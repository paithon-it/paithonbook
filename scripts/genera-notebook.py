#!/usr/bin/env python3
"""Genera i notebook compagni dei capitoli del libro.

Il libro è testo: i blocchi ```python si leggono e si copiano. Questo script ne
ricava **un notebook per capitolo**, così chi vuole *provare* il codice lo apre
su Colab (dove torch è già installato e la GPU è gratis) invece di
ricostruirlo a mano.

    python3 scripts/genera-notebook.py                 # tutti i capitoli
    python3 scripts/genera-notebook.py PyTorch Python  # solo questi
    python3 scripts/genera-notebook.py --verifica      # genera ed ESEGUE
    python3 scripts/genera-notebook.py --esistenti     # solo i capitoli che
                                                       # hanno già un notebook
    python3 scripts/genera-notebook.py --severo        # un modulo mancante è
                                                       # un errore (modalità CI)

Con `--verifica` lo script esce con codice 1 se una cella si rompe: è così che
`verifica-notebook.yml` impedisce di pubblicare un notebook che non gira.

## Perché il capitolo e non la pagina

Perché i numeri lo dicono. Su 28 capitoli con codice, quasi tutte le *pagine*
hanno uno o due blocchi: Matematica ne ha 6 distribuiti su 6 pagine, Machine
Learning 11 su 10. Un notebook per pagina sarebbe stato quasi sempre un notebook
da due celle, che si copia a mano in meno tempo di quanto ci voglia ad aprire
Colab. Il capitolo invece è un'unità che il lettore riconosce.

## Le regole, che sono la sostanza

* **Celle nell'ordine del libro**, secondo `_toc.yml`: il terzo blocco usa la
  variabile del primo, ed è per questo che l'ordine non è un dettaglio.
* **Tre stati per un blocco, non due.** Il default è cella eseguibile e
  verificata. Chi non può girare si marca nel sorgente, perché nessun euristico
  può sapere che `read_csv("vendite.csv")` non troverà il file:

      ```{code-block} python
      :class: pt-non-eseguibile     # frammento illustrativo: entra come testo
      ```
      ```{code-block} python
      :class: pt-lento              # gira su Colab, non in un job da un minuto
      ```

  `pt-lento` esiste perché «verificabile in CI» e «eseguibile su Colab» non sono
  la stessa cosa: scaricare MNIST, addestrare dieci epoche o dipendere da un
  pacchetto che in CI non vale 250 MB di wheel va benissimo su Colab, dove è
  già installato, e non va in una pipeline. Confonderli significa mentire in un
  verso o nell'altro.
* **Preludio facoltativo, e ripristinato a ogni pagina.**
  `notebooks/_preludi/<Capitolo>.py` crea i dati e i nomi che il testo dà per
  esistenti. Ma in un notebook di capitolo una pagina più avanti ridefinisce
  `X_train` con un altro dataset, e da lì le pagine precedenti non tornerebbero
  più: per questo il preludio viene ri-eseguito all'inizio di ogni pagina, così
  ognuna riparte dallo stesso stato. È impalcatura del notebook, dichiarata come
  tale, NON contenuto del libro.
* **Meno di tre celle, niente notebook.** Per `import this` aprire Colab è più
  lavoro che copiare.

Il notebook è JSON scritto a mano: nessuna dipendenza da nbformat, così lo
script gira anche in un ambiente spoglio.
"""

import json
import pathlib
import re
import subprocess
import sys
import tempfile

import yaml

RADICE = pathlib.Path(__file__).resolve().parent.parent
LIBRO = RADICE / "book"
USCITA = RADICE / "notebooks"
PRELUDI = USCITA / "_preludi"
SITO = "https://book.paithon.it/main"

# Sotto questa soglia un notebook non vale il pulsante.
MINIMO_CELLE = 3

# Capitoli rinviati: il loro codice dipende da librerie che non possiamo
# eseguire in CI, quindi il notebook non sarebbe verificato, e un pulsante
# "Esegui" su codice non provato e' una promessa che non possiamo mantenere.
# Il motivo sta accanto al nome, cosi' si sa cosa serve per sbloccarli.
RINVIATI = {
    # Resta solo questo: i kernel Triton non si compilano senza una GPU, e in
    # un runner non c'e'. Le celle non sarebbero verificate, quindi il capitolo
    # non pubblica il notebook.
    "GPU": "triton e una GPU vera per compilare i kernel",
}

# Pacchetti da installare, dedotti dagli import del codice.
PACCHETTI = {
    "torch": "torch torchvision",
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "sklearn": "scikit-learn",
    "torchinfo": "torchinfo",
    "PIL": "pillow",
    "networkx": "networkx",
    "scipy": "scipy",
    "xgboost": "xgboost",
    "transformers": "transformers",
    "diffusers": "diffusers",
    "librosa": "librosa",
    "triton": "triton",
    "torch_geometric": "torch-geometric",
    "gymnasium": "gymnasium",
}

# Le versioni dichiarate: finiscono in testa a ogni notebook, così chi lo apre
# fra un anno sa contro cosa era stato provato, e se qualcosa non gira sa dove
# guardare invece di dare la colpa al libro.
VERSIONI = RADICE / "requirements-notebook.txt"


def versioni_dichiarate() -> tuple[str, str]:
    """(elenco leggibile, data dell'ultima verifica)."""
    if not VERSIONI.exists():
        return "", ""
    testo = VERSIONI.read_text()
    data = re.search(r"# Ultima verifica completa: (\S+)", testo)
    coppie = [r.strip() for r in testo.split("\n")
              if "==" in r and not r.lstrip().startswith("#")]
    interessanti = ("torch==", "transformers==", "numpy==", "scikit-learn==",
                    "pandas==", "diffusers==", "librosa==", "torch-geometric==")
    scelte = [c.replace("==", " ") for c in coppie if c.startswith(interessanti)]
    return ", ".join(scelte), (data.group(1) if data else "")


RE_SEMPLICE = re.compile(r"^```python\n(.*?)^```", re.S | re.M)
RE_DIRETTIVA = re.compile(
    r"^```\{code-block\}\s+python\n((?:\s*:.*\n)*)(.*?)^```", re.S | re.M
)
RE_TITOLO = re.compile(r"^(#{1,4})\s+(.*)$", re.M)

# Le tre virgolette servono a incapsulare il preludio in una stringa
# ri-eseguibile; qui come costante per non doverle scrivere annidate.
TRIPLE = chr(39) * 3


def capitoli() -> dict[str, list[pathlib.Path]]:
    """{nome del capitolo: pagine, nell'ordine del toc}."""
    toc = yaml.safe_load((LIBRO / "_toc.yml").read_text())
    # Il toc raggruppa i capitoli in `parts` (con la caption che intesta
    # l'indice); la forma piatta `chapters` resta accettata perche' e' quella
    # di jb-book senza parti: le parti qui non contano, contano i capitoli.
    elenco = [c for p in toc.get("parts", []) for c in p["chapters"]] \
        or toc.get("chapters", [])
    fuori: dict[str, list[pathlib.Path]] = {}
    for cap in elenco:
        if "/" not in cap["file"]:          # references.md e simili: non capitoli
            continue
        files = [cap["file"]] + [s["file"] for s in cap.get("sections", [])]
        pagine = [LIBRO / f for f in files
                  if f.endswith(".md") and (LIBRO / f).exists()]
        if pagine:
            fuori[cap["file"].split("/")[0]] = pagine
    return fuori


def blocchi(testo: str):
    """[(posizione, codice, stato)] in ordine di documento.

    stato: "cella" (eseguibile e verificata), "lento" (cella vera, saltata dal
    verificatore), "testo" (frammento illustrativo, non è una cella).
    """
    trovati = []
    for m in RE_SEMPLICE.finditer(testo):
        trovati.append((m.start(), m.group(1), "cella"))
    for m in RE_DIRETTIVA.finditer(testo):
        classi = m.group(1)
        if "pt-non-eseguibile" in classi:
            stato = "testo"
        elif "pt-lento" in classi:
            stato = "lento"
        else:
            stato = "cella"
        trovati.append((m.start(), m.group(2), stato))
    return sorted(trovati)


def titolo_di(testo: str) -> str | None:
    m = RE_TITOLO.search(testo)
    return m.group(2) if m else None


def titolo_prima_di(testo: str, posizione: int) -> str | None:
    ultimo = None
    for m in RE_TITOLO.finditer(testo, 0, posizione):
        ultimo = m
    return ultimo.group(2) if ultimo else None


def pacchetti_usati(codice: str) -> list[str]:
    trovati = []
    for modulo, pacchetto in PACCHETTI.items():
        if re.search(rf"^\s*(import|from)\s+{re.escape(modulo)}\b", codice, re.M):
            trovati.extend(pacchetto.split())
    return sorted(set(trovati))


def cella_testo(righe: list[str]) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": righe}


def cella_codice(codice: str, tag: str | None = None,
                 pagina: str | None = None) -> dict:
    """Una cella. `pagina` finisce nei metadati: serve al verificatore per dire
    non solo QUALE cella si rompe ma in quale pagina sta, altrimenti trovare il
    blocco fra dieci pagine che cominciano tutte con `import torch` è una
    caccia al tesoro."""
    meta: dict = {}
    if tag:
        meta["tags"] = [tag]
    if pagina:
        meta["pt-pagina"] = pagina
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": meta,
        "outputs": [],
        "source": codice.rstrip("\n").split("\n"),
    }


def componi(nome: str, pagine: list[pathlib.Path]):
    """Costruisce il notebook di un capitolo. Restituisce (json, n_celle)."""
    raccolto = []          # (titolo, url, rel, [(codice, stato, titoletto)])
    for pagina in pagine:
        testo = pagina.read_text()
        pezzi = blocchi(testo)
        if not pezzi:
            continue
        rel = pagina.relative_to(LIBRO).with_suffix("")
        voci = [(c, s, titolo_prima_di(testo, pos)) for pos, c, s in pezzi]
        raccolto.append((titolo_di(testo) or str(rel), f"{SITO}/{rel}.html",
                         str(rel), voci))

    celle_vere = sum(1 for *_, voci in raccolto for _, s, _ in voci if s != "testo")
    if celle_vere < MINIMO_CELLE:
        return None, 0

    tutto = "\n".join(c for *_, voci in raccolto for c, s, _ in voci if s != "testo")
    pacchetti = pacchetti_usati(tutto)
    titolo_capitolo, url_capitolo, _, _ = raccolto[0]

    celle = [cella_testo([
        f"# {titolo_capitolo}\n", "\n",
        f"Il codice del capitolo [«{titolo_capitolo}»]({url_capitolo}), "
        "*Paithon Book*.\n", "\n",
        "Le celle sono quelle del libro, nell'ordine in cui compaiono: il testo "
        "che le spiega sta nelle pagine, qui c'è solo la parte da eseguire e da "
        "rompere.\n", "\n",
        "Generato da `scripts/genera-notebook.py`: le correzioni vanno fatte "
        "nelle pagine del libro, non qui.\n",
    ])]

    elenco_versioni, data_verifica = versioni_dichiarate()
    if elenco_versioni:
        celle.append(cella_testo([
            f"> **Verificato il {data_verifica}** con {elenco_versioni}. "
            "Tutte le celle di questo notebook sono state eseguite senza errori "
            "con quelle versioni; le librerie si muovono, e se qualcosa qui non "
            "gira piu' e' un errore del libro: "
            "[segnalalo](https://github.com/paithon-it/paithonbook/issues).\n"]))

    if pacchetti:
        celle.append(cella_codice(
            "# Su Colab quasi tutto c'è già; questa riga serve altrove.\n"
            f"%pip install -q {' '.join(pacchetti)}", tag="pt-setup"))

    preludio = PRELUDI / f"{nome}.py"
    testo_preludio = preludio.read_text() if preludio.exists() else None
    if testo_preludio:
        # Il preludio viaggia dentro una stringa per poter essere ri-eseguito
        # all'inizio di ogni pagina: senza, la prima pagina che ridefinisce
        # `X_train` romperebbe tutte le successive.
        assert TRIPLE not in testo_preludio, "il preludio non puo contenere " + TRIPLE
        celle.append(cella_testo([
            "> **Cella di preparazione.** Crea i dati e i nomi che il testo da per "
            "esistenti. Non fa parte del libro: serve a far girare il notebook, e "
            "viene ripetuta all'inizio di ogni pagina perche ognuna riparta dallo "
            "stesso stato.\n"]))
        celle.append(cella_codice(
            "_PRELUDIO = r" + TRIPLE + "\n" + testo_preludio.rstrip() + "\n"
            + TRIPLE + "\nexec(_PRELUDIO)", tag="pt-setup-dati"))

    for n_pagina, (titolo_pagina, url, rel_pagina, voci) in enumerate(raccolto):
        celle.append(cella_testo([f"## {titolo_pagina}\n", "\n",
                                  f"[Leggi la pagina]({url})\n"]))
        if testo_preludio and n_pagina:
            celle.append(cella_codice(
                "exec(_PRELUDIO)   # ripristina i nomi di partenza della pagina"))
        ultimo = None
        for codice, stato, titoletto in voci:
            if titoletto and titoletto not in (titolo_pagina, ultimo):
                celle.append(cella_testo([f"### {titoletto}\n"]))
                ultimo = titoletto
            if stato == "cella":
                celle.append(cella_codice(codice, pagina=rel_pagina))
            elif stato == "lento":
                celle.append(cella_codice(codice, tag="pt-lento",
                                          pagina=rel_pagina))
            else:
                celle.append(cella_testo([
                    "*Frammento illustrativo: nel libro mostra la forma, qui non "
                    "si esegue.*\n", "\n", "```python\n",
                    *[r + "\n" for r in codice.rstrip().split("\n")], "```\n"]))

    nb = {
        "cells": celle,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return nb, sum(1 for c in celle if c["cell_type"] == "code")


def verifica(nb: dict) -> str:
    """Esegue le celle in fila e dice QUALE si rompe.

    Non serve un kernel: le celle di un capitolo sono uno script. Ma un solo
    messaggio di stderr non basta a trovare il colpevole fra quaranta, così le
    celle si eseguono una alla volta nello stesso spazio dei nomi e, al primo
    errore, si stampano numero, eccezione e prima riga.

    Si saltano: la cella di installazione (`pt-setup`), quelle marcate
    `pt-lento`, e le righe che cominciano con `%` o `!`, le magie IPython sono
    legittime in un notebook ma non in Python puro. E un modulo mancante viene
    distinto da un errore vero: su una macchina spoglia può solo mancare, non
    essere rotto.
    """
    celle, magie, lente = [], 0, 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        tag = c["metadata"].get("tags", [])
        if "pt-setup" in tag:
            continue
        if "pt-lento" in tag:
            lente += 1
            continue
        righe = []
        for r in c["source"]:
            if re.match(r"^\s*[%!]", r):
                magie += 1
                righe.append("# " + r)
            else:
                righe.append(r)
        celle.append(("\n".join(righe), c["metadata"].get("pt-pagina", "?")))

    coda = ""
    if magie:
        coda += f", {magie} magie non verificabili qui"
    if lente:
        coda += f", {lente} lente saltate"
    fra_parentesi = f" ({coda.lstrip(', ')})" if coda else ""

    if not celle:
        return "nessuna cella verificabile" + fra_parentesi

    driver = (
        "import json, sys\n"
        "celle = json.load(open(sys.argv[1]))\n"
        "spazio = {'__name__': '__main__'}\n"
        "for i, codice in enumerate(celle, 1):\n"
        "    try:\n"
        "        exec(compile(codice, f'<cella {i}>', 'exec'), spazio)\n"
        "    except BaseException as e:\n"
        "        prima = next((r for r in codice.split('\\n') if r.strip()), '')\n"
        "        print(f'CELLA {i} | {type(e).__name__}: {e} | {prima[:70]}')\n"
        "        sys.exit(3)\n"
    )

    with tempfile.TemporaryDirectory() as cartella:
        d = pathlib.Path(cartella)
        (d / "celle.json").write_text(json.dumps([c for c, _ in celle]))
        (d / "driver.py").write_text(driver)
        try:
            esito = subprocess.run(
                [sys.executable, str(d / "driver.py"), str(d / "celle.json")],
                capture_output=True, text=True, timeout=600, cwd=cartella)
        except subprocess.TimeoutExpired:
            return "TIMEOUT"

    if esito.returncode == 0:
        return "OK" + fra_parentesi

    riga = next((r for r in esito.stdout.split("\n") if r.startswith("CELLA")), "")
    if riga:
        n = int(riga.split()[1])
        if 1 <= n <= len(celle):
            riga = riga.replace(f"CELLA {n} |", f"CELLA {n} [{celle[n - 1][1]}] |", 1)
    if not riga:
        coda_err = (esito.stderr.strip().split("\n") or ["?"])[-1]
        return "ERRORE: " + coda_err[:110]
    if "ModuleNotFoundError" in riga:
        modulo = re.search(r"No module named '([^']+)'", riga)
        return f"NON VERIFICABILE QUI: manca {modulo.group(1) if modulo else '?'}"
    return riga


def aggiorna_manifesto() -> None:
    """Scrive in `_config.yml` l'elenco dei capitoli che hanno un notebook.

    Il pulsante deve comparire solo dove il notebook esiste, e un template Jinja
    non può guardare il disco: l'elenco glielo passa `html_context`. La
    sostituzione riscrive solo le voci sotto `pt_notebooks:`, così i commenti del
    file restano dove sono. L'elenco è quello che ESISTE su disco, non quello
    prodotto da questa passata: generare un capitolo alla volta non cancella gli
    altri.
    """
    presenti = sorted(n.stem for n in USCITA.glob("*.ipynb"))
    percorso = LIBRO / "_config.yml"
    righe = percorso.read_text().split("\n")
    try:
        i = next(n for n, r in enumerate(righe) if r.strip() == "pt_notebooks:")
    except StopIteration:
        print("  ATTENZIONE: manca `pt_notebooks:` in _config.yml")
        return
    rientro = " " * (len(righe[i]) - len(righe[i].lstrip()))
    j = i + 1
    while j < len(righe) and righe[j].lstrip().startswith("- "):
        j += 1
    percorso.write_text("\n".join(
        righe[:i + 1] + [f"{rientro}- {c}" for c in presenti] + righe[j:]))
    print(f"  manifesto in _config.yml: {len(presenti)} capitoli")


def main() -> None:
    argomenti = [a for a in sys.argv[1:] if not a.startswith("--")]
    con_verifica = "--verifica" in sys.argv
    solo_esistenti = "--esistenti" in sys.argv
    # In CI le librerie ci sono tutte: se un modulo manca là non è "non
    # verificabile", è la lista dei pacchetti del workflow che è incompleta.
    severo = "--severo" in sys.argv

    elenco = capitoli()
    for nome, motivo in RINVIATI.items():
        if nome in elenco and nome not in argomenti:
            del elenco[nome]
            obsoleto = USCITA / f"{nome}.ipynb"
            if obsoleto.exists():
                obsoleto.unlink()
            print(f"  ⏸  {nome}: rinviato, serve {motivo}")
    if argomenti:
        elenco = {k: v for k, v in elenco.items() if k in argomenti}
    if solo_esistenti:
        gia = {n.stem for n in USCITA.glob("*.ipynb")}
        elenco = {k: v for k, v in elenco.items() if k in gia}

    scritti = saltati = rotti = 0
    for nome, pagine in elenco.items():
        nb, n = componi(nome, pagine)
        if nb is None:
            saltati += 1
            # Se il capitolo era già pubblicato e ora scende sotto soglia, di
            # solito perché i suoi blocchi sono stati marcati, il notebook
            # vecchio va rimosso: altrimenti resta su disco, entra nel
            # manifesto e il pulsante porta a un file che non rispecchia più
            # il libro.
            vecchio = USCITA / f"{nome}.ipynb"
            if vecchio.exists():
                vecchio.unlink()
                print(f"  ✗  {nome}: sotto soglia, notebook rimosso")
            else:
                print(f"  –  {nome}: meno di {MINIMO_CELLE} celle, niente notebook")
            continue
        USCITA.mkdir(parents=True, exist_ok=True)
        (USCITA / f"{nome}.ipynb").write_text(
            json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
        scritti += 1
        esito = ""
        if con_verifica:
            risultato = verifica(nb)
            if (risultato.startswith(("ERRORE", "CELLA")) or risultato == "TIMEOUT"
                    or (severo and risultato.startswith("NON VERIFICABILE"))):
                rotti += 1
            esito = f"  {risultato}"
        print(f"  ✓  {nome}: {n} celle{esito}")

    aggiorna_manifesto()
    print(f"\n  notebook scritti: {scritti}   capitoli saltati: {saltati}")
    if not con_verifica:
        print("  (rilancia con --verifica per eseguirli davvero)")
    elif rotti:
        print(f"  {rotti} notebook NON eseguono: non vanno pubblicati così")
        sys.exit(1)
    else:
        print("  tutti eseguiti senza errori")


if __name__ == "__main__":
    main()
