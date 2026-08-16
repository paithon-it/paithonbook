#!/usr/bin/env python3
"""I file che a un sito servono alla radice: robots.txt, sitemap.xml, llms.txt.

    python3 scripts/genera-radice.py             # scrive book/_extra/
    python3 scripts/genera-radice.py --verifica  # sono allineati al _toc.yml?

## Perche' non si scrivono a mano

Sono l'elenco delle pagine del libro, e l'elenco delle pagine del libro sta nel
`_toc.yml`. Scritti a mano invecchierebbero al primo capitolo nuovo, e
invecchierebbero *in silenzio*: un sitemap che non elenca una pagina non da'
errore, semplicemente quella pagina non viene indicizzata.

## Come arrivano alla radice, che e' la parte non ovvia

Vanno in `book/_extra/`, che `_config.yml` passa a Sphinx come
`html_extra_path`: quella cartella viene ricopiata **tal quale** nella radice
della build, cioe' in `book.paithon.it/main/`. Ma `robots.txt` conta solo se
sta nella radice del *dominio*, non in una sottocartella, e il libro vive in
`/main/` perche' il workflow di TeachBooks costruisce ogni branch in una
cartella sua.

Il pezzo che chiude il cerchio e' la variabile di repository
**`BEHAVIOR_PRIMARY=copy`** (in `paithon-it/paithonbook`): con quella il
workflow ricopia il branch primario anche nella radice del sito, e i tre file
finiscono dove servono. Con il valore di prima (`redirect`) la radice non era
una pagina ma la 404 di TeachBooks con un rimbalzo JavaScript: un browser ci
passava sopra senza accorgersene, un crawler ci leggeva `HTTP 404` e se ne
andava.

Conseguenza: la radice ora *serve* il libro (non ci redirige), e le due copie
si consolidano con il `canonical`, che punta sempre a `/main/`. Il `canonical`
lo scrive Sphinx da `html.baseurl`, ed e' l'unica ragione per cui avere due
copie non e' un danno.

## Nessun index.html

Non serve: la build ne ha gia' uno, sono 58 byte di
`<meta http-equiv="Refresh" content="0; url=intro.html" />`, ed essendo
relativo funziona identico nella radice e in `/main/`. Sovrascriverlo con uno
nostro vorrebbe dire litigare con Sphinx per niente.

## Niente llms-full.txt

Il libro intero in un file solo, che qualche sito serve accanto a `llms.txt`,
qui non si fa: il testo e' CC BY-NC-ND, e servirlo in un unico blocco invita
esattamente la cosa che la licenza esclude. La mappa con i link, si'.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import yaml

RADICE = Path(__file__).resolve().parent.parent
LIBRO = RADICE / "book"
DOVE_DEFAULT = LIBRO / "_extra"

# La landing apre con l'istruzione per l'interruttore dei livelli, che come
# descrizione non dice niente: per lei vale la riga curata del sito (la stessa
# di `_static/head_custom.html`).
DESCRIZIONE_LIBRO = (
    "Libro di Intelligenza Artificiale in italiano, su due livelli: Elementare "
    "per chi parte da zero, Superiore per la trattazione formale. Machine "
    "Learning, Deep Learning e Reinforcement Learning con Python. Online, "
    "gratuito, in aggiornamento continuo.")


def base_url() -> str:
    dati = yaml.safe_load((LIBRO / "_config.yml").read_text(encoding="utf-8"))
    return (dati.get("html", {}) or {}).get("baseurl", "").rstrip("/")


def dominio(url: str) -> str:
    """`https://book.paithon.it/main` -> `https://book.paithon.it`."""
    pezzi = url.split("/")
    return "/".join(pezzi[:3])


def pagine() -> list[dict]:
    """Le pagine del libro nell'ordine di lettura, con parte e capitolo."""
    dati = yaml.safe_load((LIBRO / "_toc.yml").read_text(encoding="utf-8"))
    fuori = [{"file": dati["root"], "parte": "", "capitolo": "",
              "titolo": "Paithon Book", "radice": True}]
    for parte in dati.get("parts", []) or []:
        for cap in parte.get("chapters", []) or []:
            fuori.append({"file": cap["file"], "parte": parte.get("caption", ""),
                          "capitolo": cap.get("title", ""),
                          "titolo": cap.get("title", ""), "radice": False,
                          "capo": True})
            for sez in cap.get("sections", []) or []:
                fuori.append({"file": sez["file"], "parte": parte.get("caption", ""),
                              "capitolo": cap.get("title", ""),
                              "titolo": sez.get("title", ""), "radice": False})
    return fuori


# --------------------------------------------------------------------------
# la prima frase di una pagina
# --------------------------------------------------------------------------

RX_FENCE = re.compile(r"^(`{3,})")
RX_INLINE = [
    (re.compile(r"\{[a-z-]+\}`[^`]*`"), ""),          # ruoli MyST
    (re.compile(r"\[([^\]]+)\]\([^)]+\)"), r"\1"),    # link
    (re.compile(r"</?[a-zA-Z][^>]*>"), ""),           # html inline
    (re.compile(r"[*`_]"), ""),
    (re.compile(r"\s+"), " "),
]


def prima_frase(percorso: Path, limite: int = 165) -> str:
    """Il primo paragrafo di testo della pagina, che nel libro e' l'aggancio.

    Sta anche in `strumenti/compila_skill.py` (`sommario`), sul testo ripulito:
    qui la sorgente e' il MyST grezzo, e i due lavori sono abbastanza diversi da
    non valere un modulo in comune.
    """
    if not percorso.is_file():
        return ""
    if percorso.suffix == ".ipynb":
        import json
        celle = json.loads(percorso.read_text(encoding="utf-8")).get("cells", [])
        righe = "".join("".join(c.get("source", [])) + "\n\n" for c in celle
                        if c.get("cell_type") == "markdown").splitlines()
    else:
        righe = percorso.read_text(encoding="utf-8").splitlines()

    # Un commento HTML su piu' righe: la prima riga cade da se' (comincia per
    # `<`), le altre no, e senza questo la descrizione di una pagina generata
    # diventava l'avviso «non modificare a mano».
    pila, paragrafo, commento = [], [], False
    for riga in righe:
        if commento:
            commento = "-->" not in riga
            continue
        if riga.lstrip().startswith("<!--") and "-->" not in riga:
            commento = True
            continue
        fence = RX_FENCE.match(riga)
        if fence:
            n = len(fence.group(1))
            if pila and pila[-1] == n and not riga.strip()[n:].strip():
                pila.pop()
            else:
                pila.append(n)
            continue
        if pila:
            continue
        nudo = riga.strip()
        if not nudo:
            if paragrafo:
                break
            continue
        if nudo.startswith(("#", ":", "$$", "|", ">", "(", "<", "!", "%")):
            if paragrafo:
                break
            continue
        paragrafo.append(nudo)

    frase = " ".join(paragrafo)
    for rx, con in RX_INLINE:
        frase = rx.sub(con, frase)
    frase = frase.strip()
    if len(frase) <= limite:
        return frase
    taglio = frase.rfind(". ", 0, limite)
    if taglio > limite // 2:
        return frase[:taglio + 1]
    return frase[:limite - 1].rsplit(" ", 1)[0] + "…"


def _iso_utc(quando: str) -> str:
    """La stessa data scritta sempre allo stesso modo.

    `git log --format=%cI` non e' stabile fra versioni di git: la 2.43 stampa
    l'UTC come `+00:00`, la 2.54 come `Z`. Sono lo stesso istante e per la
    sitemap valgono uguale, ma il file cambia byte, e allora `--verifica`
    dichiara disallineato un file che nessuno ha toccato: passava in locale e
    falliva in CI, dove git e' piu' recente. Qui la data si riscrive noi, in
    UTC, e il risultato non dipende ne' dalla versione di git ne' dal fuso di
    chi ha committato.
    """
    try:
        return (datetime.fromisoformat(quando)
                .astimezone(timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%S+00:00"))
    except ValueError:
        return quando


def date_git() -> dict[str, str]:
    """Data dell'ultima modifica per ogni file sotto `book/`, da git.

    Una passata sola sulla storia: 160 invocazioni di `git log` costavano
    secondi e lo stesso risultato.
    """
    fuori: dict[str, str] = {}
    try:
        uscita = subprocess.run(
            ["git", "-C", str(RADICE), "log", "--format=%cI", "--name-only",
             "--no-renames", "--", "book"],
            capture_output=True, text=True, check=True, timeout=120).stdout
    except Exception:
        return fuori
    data = ""
    for riga in uscita.splitlines():
        if not riga.strip():
            continue
        if re.match(r"^\d{4}-\d{2}-\d{2}T", riga):
            data = riga.strip()
        elif riga.startswith("book/"):
            fuori.setdefault(riga[len("book/"):], _iso_utc(data))
    return fuori


# --------------------------------------------------------------------------
# i tre file
# --------------------------------------------------------------------------

def robots(url: str) -> str:
    return "\n".join([
        "# Il libro e' pubblico e si puo' indicizzare, tutto quanto.",
        "#",
        "# Anche i crawler degli assistenti (ClaudeBot, GPTBot, PerplexityBot,",
        "# Google-Extended, Applebot-Extended) sono ammessi di proposito: sono",
        "# il canale da cui il libro viene citato quando qualcuno chiede una",
        "# cosa che il libro spiega.",
        "#",
        "# Le pagine esistono a piu' indirizzi (la radice, /main/, l'alias",
        "# /book/) perche' il workflow costruisce ogni branch in una cartella",
        "# sua e ricopia il primario nella radice. NON si bloccano qui: a",
        "# consolidarle c'e' il <link rel=\"canonical\">, che punta sempre a",
        "# /main/. Bloccarle impedirebbe di leggere proprio quel canonical.",
        "",
        "User-agent: *",
        "Allow: /",
        "",
        f"Sitemap: {dominio(url)}/sitemap.xml",
        "",
    ])


def sitemap(url: str, elenco: list[dict], date: dict[str, str]) -> str:
    ns = "http://www.sitemaps.org/schemas/sitemap/0.9"
    ET.register_namespace("", ns)
    radice = ET.Element(f"{{{ns}}}urlset")
    for pagina in elenco:
        html = Path(pagina["file"]).with_suffix(".html").as_posix()
        voce = ET.SubElement(radice, f"{{{ns}}}url")
        ET.SubElement(voce, f"{{{ns}}}loc").text = f"{url}/{html}"
        quando = date.get(pagina["file"])
        if quando:
            ET.SubElement(voce, f"{{{ns}}}lastmod").text = quando
    ET.indent(radice, space="  ")
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            + ET.tostring(radice, encoding="unicode") + "\n")


def llms(url: str, elenco: list[dict]) -> str:
    fuori = [
        "# Paithon Book",
        "",
        f"> {DESCRIZIONE_LIBRO}",
        "",
        "Di Francesco Messina, su book.paithon.it. Ogni concetto e' scritto due",
        "volte, su due livelli che il lettore sceglie da un interruttore:",
        "**Elementare** (analogia concreta, nessun prerequisito) e **Superiore**",
        "(definizioni, formule, complessita'). Il framework e' PyTorch.",
        "",
        "Cosa NON c'e', per scelta: benchmark, classifiche, prezzi e novita' del",
        "mese. Il libro spiega i meccanismi, che restano veri; la cronaca sta sul",
        "magazine paithon.it. Se cercate «quale modello e' il migliore oggi»,",
        "questo non e' il posto.",
        "",
        "Licenza CC BY-NC-ND 4.0: si cita e si parafrasa attribuendo (titolo",
        "della sezione e URL), non si ridistribuisce il testo.",
        "",
    ]
    # Il capitolo porta la sua descrizione (l'`overview.md` dice di che parla);
    # le sezioni no, e stanno rientrate sotto di lui. Con una descrizione per
    # sezione il file raddoppiava, e a descriverle c'erano gli agganci narrativi
    # del libro («San Pietroburgo, inverno 1913»), che sono belli da leggere e
    # non dicono a un agente niente che il titolo non dica meglio.
    parte_corrente = None
    for pagina in elenco:
        if pagina["radice"]:
            continue
        if pagina["parte"] != parte_corrente:
            parte_corrente = pagina["parte"]
            if fuori[-1]:
                fuori.append("")
            fuori += [f"## {parte_corrente}", ""]
        html = Path(pagina["file"]).with_suffix(".html").as_posix()
        titolo = pagina["titolo"] or Path(pagina["file"]).stem
        if pagina.get("capo"):
            frase = prima_frase(LIBRO / pagina["file"], 200)
            fuori.append(f"- [{titolo}]({url}/{html})"
                         + (f": {frase}" if frase else ""))
        else:
            fuori.append(f"  - [{titolo}]({url}/{html})")
    fuori.append("")
    return "\n".join(fuori)


# --------------------------------------------------------------------------

def genera() -> dict[str, str]:
    url = base_url()
    elenco = pagine()
    return {
        "robots.txt": robots(url),
        "sitemap.xml": sitemap(url, elenco, date_git()),
        "llms.txt": llms(url, elenco),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Genera i file di radice del sito.")
    ap.add_argument("--dove", type=Path, default=DOVE_DEFAULT)
    ap.add_argument("--verifica", action="store_true",
                    help="esce con 1 se i file sul disco non sono quelli attesi")
    argomenti = ap.parse_args()
    dove = argomenti.dove
    file = genera()

    if argomenti.verifica:
        diversi = []
        for nome, atteso in file.items():
            percorso = dove / nome
            if not percorso.is_file():
                diversi.append(f"{nome} (manca)")
            elif percorso.read_text(encoding="utf-8") != atteso:
                diversi.append(nome)
        if diversi:
            print("da rigenerare: " + ", ".join(diversi))
            print("  python3 scripts/genera-radice.py")
            return 1
        print(f"allineati al _toc.yml: {', '.join(file)}")
        return 0

    dove.mkdir(parents=True, exist_ok=True)
    for nome, testo in file.items():
        (dove / nome).write_text(testo, encoding="utf-8")
        print(f"  {dove.relative_to(RADICE) / nome}  "
              f"{len(testo.encode()) / 1000:.1f} kB")
    print(f"scritti in {dove}; finiscono nella radice del sito grazie a "
          f"BEHAVIOR_PRIMARY=copy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
