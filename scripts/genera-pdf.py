#!/usr/bin/env python3
"""Il libro intero in un PDF: `paithon-book.pdf`.

    python3 scripts/genera-pdf.py              # costruisce il PDF
    python3 scripts/genera-pdf.py --solo-tex   # si ferma al sorgente LaTeX
    python3 scripts/genera-pdf.py --pulisci    # butta la cache delle figure
    python3 scripts/genera-pdf.py --verifica   # il PDF c'e' ed e' aggiornato?
    python3 scripts/genera-pdf.py --rilascia   # lo allega alla release

Il libro online si legge una pagina alla volta; questo e' il secondo formato,
impaginato come un libro vero. Esce in `book/_build/stampa/paithon-book.pdf`,
e da li' lo allega a una release il comando `--rilascia`, che e' un comando a
parte apposta: le azioni verso l'esterno non si infilano in un comando
composto.

## I passi

1. `sphinx -b latex` sul libro, con l'estensione `_ext/pt_stampa.py` che
   converte le figure (Chromium), da' un nome alle tab e sostituisce le
   animazioni con i fermi immagine.
2. LuaLaTeX, tre passate: la prima scrive gli ausiliari, la seconda risolve
   l'indice e i riferimenti, la terza sistema i numeri di pagina che le
   prime due hanno fatto ballare.

## Perche' si legge il .log e non il codice di uscita

LuaLaTeX gira in `nonstopmode`, che vuol dire che **non si ferma mai**: davanti
a una figura che non trova o a un ambiente che non conosce si aggiusta da se',
tira dritto, e alla fine esce con codice 0 e un PDF in mano. Un PDF con dentro
i riquadri neri al posto delle figure e' comunque un PDF. Quindi il codice di
uscita non dice niente e l'unica fonte sono le righe del log, che `errori_veri`
setaccia.

## Che cosa serve installato

TeX Live (`lualatex`), Playwright con Chromium, e `jupyter-book` della serie 1
(la 2 e' mystmd, un altro programma). Se il log si lamenta di un pacchetto TeX
mancante, si installa e si annota qui, perche' il prossimo che clona lo trovi
scritto:

    sudo apt-get install -y texlive-luatex texlive-latex-extra
"""

import argparse
import pathlib
import re
import shutil
import subprocess
import sys

RADICE = pathlib.Path(__file__).resolve().parent.parent
LIBRO = RADICE / "book"
USCITA = LIBRO / "_build" / "stampa"
TEX = USCITA / "tex"
NOME = "paithon-book"
CASA = "book.paithon.it"
PUBBLICO = "paithon-it/paithonbook"
# Chi firma le release. Vedi `rilascia()`: si controlla, non si spera.
FIRMA = "paithon-it"

# Le righe del log che contano. In `nonstopmode` LaTeX segnala e prosegue,
# quindi queste sono l'unico modo di sapere se il PDF e' venuto bene.
#
# La riga si prende INTERA (`.*`): la prima versione di questa espressione
# si fermava al prefisso, e il rapporto degli errori era una colonna di punti
# esclamativi. Un setaccio che non dice che cosa ha trovato non serve.
GUAI = re.compile(
    r"^(?:!.*|.*?Package \S+ Error.*|.*?LaTeX Error.*"
    r"|.*?File .* not found.*|.*?Missing character.*)$", re.M)


def esegui(comando: list[str], dove: pathlib.Path | None = None,
           ambiente: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(comando, cwd=dove, env=ambiente,
                          capture_output=True, text=True)


def costruisci_tex(pulisci: bool = False) -> pathlib.Path:
    """`sphinx -b latex`. Torna la cartella che contiene il .tex.

    `jupyter-book config sphinx` scrive `book/conf.py`, che non e' tracciato e
    non deve esserlo: si cancella sempre, anche se qualcosa va storto.
    """
    import os

    if pulisci and USCITA.exists():
        shutil.rmtree(USCITA)
    TEX.mkdir(parents=True, exist_ok=True)

    conf = LIBRO / "conf.py"
    fatto = esegui(["jupyter-book", "config", "sphinx", str(LIBRO)])
    if fatto.returncode != 0:
        conf.unlink(missing_ok=True)
        sys.exit(f"la configurazione non si scrive:\n{fatto.stderr}")

    ambiente = dict(os.environ, PYTHONPATH=str(LIBRO / "_ext"))
    try:
        fatto = esegui(["python3", "-m", "sphinx", "-b", "latex",
                        str(LIBRO), str(TEX)], ambiente=ambiente)
    finally:
        conf.unlink(missing_ok=True)

    sorgente = TEX / f"{NOME}.tex"
    # Si guarda il CODICE DI USCITA, non se il file c'e'. La prima versione
    # guardava solo l'esistenza, e quando l'estensione moriva a `config-inited`
    # trovava il .tex della build precedente e annunciava che era andato tutto
    # bene. Uno strumento che verifica l'esistenza invece dell'esito e' peggio
    # di nessuno strumento: fa credere di aver controllato.
    if fatto.returncode != 0:
        coda = "\n".join((fatto.stdout + fatto.stderr).splitlines()[-30:])
        sys.exit(f"sphinx si e' fermato (uscita {fatto.returncode}):\n{coda}")
    if not sorgente.exists():
        sys.exit(f"il sorgente LaTeX non e' stato scritto in {sorgente}")

    # Gli avvisi di Sphinx vanno su stderr, non su stdout: cercarli solo in
    # stdout vuol dire non trovarne mai.
    avvisi = [r for r in (fatto.stdout + fatto.stderr).splitlines()
              if "WARNING" in r]
    print(f"  sorgente: {sorgente.relative_to(RADICE)} "
          f"({sorgente.stat().st_size // 1024} KB, {len(avvisi)} avvisi)")
    for r in avvisi[:10]:
        print(f"    {r}")
    return TEX


def ritaglia_capitolo(sorgente: pathlib.Path, nome: str) -> pathlib.Path:
    """Un .tex con dentro un capitolo solo, per i provini.

    Serve perche' una build intera costa minuti e chi lavora sullo stile ne
    fa venti di seguito. Si prende il preambolo (tutto fino a
    `\\begin{document}`) e la fetta fra il `\\chapter` scelto e il successivo.

    **I numeri di pagina e i riferimenti fuori dal capitolo non tornano**, ed
    e' inevitabile: un libro e' impaginato globalmente, l'indice e i
    `{numref}` dipendono da dove cade tutto il resto. Per guardare un font,
    un riquadro o un'apertura di capitolo va benissimo lo stesso; per il
    libro vero c'e' la build intera.
    """
    testo = sorgente.read_text(encoding="utf-8")
    inizio_corpo = testo.index(r"\begin{document}") + len(r"\begin{document}")
    preambolo = testo[:inizio_corpo]

    tagli = [m.start() for m in re.finditer(r"^\\chapter[\{\[]", testo, re.M)]
    if not tagli:
        sys.exit("nel sorgente non ci sono capitoli")

    scelto = None
    for numero, taglio in enumerate(tagli):
        fine = tagli[numero + 1] if numero + 1 < len(tagli) else len(testo)
        if f"{{{nome}/" in testo[taglio:fine] or f"{nome}/" in testo[taglio:fine]:
            scelto = (taglio, fine)
            break
    if scelto is None:
        sys.exit(f"capitolo «{nome}» non trovato (e' il nome della cartella "
                 f"sotto book/, per esempio VisioneArtificiale)")

    fetta = testo[scelto[0]:scelto[1]]
    provino = sorgente.with_name(f"{NOME}-provino.tex")
    provino.write_text(preambolo + "\n" + fetta + "\n\\end{document}\n",
                       encoding="utf-8")
    return provino


def errori_veri(log: pathlib.Path) -> list[str]:
    """Le righe del .log che dicono davvero che qualcosa non va."""
    if not log.exists():
        return ["il log non esiste: LuaLaTeX non e' proprio partito"]
    testo = log.read_text(encoding="utf-8", errors="ignore")
    righe = [r.strip() for r in GUAI.findall(testo)]
    # `Missing character` esce una volta per carattere: si contano, non si
    # elencano, o il rapporto diventa illeggibile.
    mancanti = [r for r in righe if "Missing character" in r]
    altri = [r for r in righe if "Missing character" not in r]
    if mancanti:
        altri.append(f"{len(mancanti)} caratteri che i font non hanno")
    return altri


def compila(cartella: pathlib.Path, passate: int = 3,
            nome: str = NOME) -> pathlib.Path:
    """LuaLaTeX, N passate. Torna il PDF."""
    pdf = cartella / f"{nome}.pdf"
    for numero in range(1, passate + 1):
        print(f"  passata {numero}/{passate}...", flush=True)
        esegui(["lualatex", "-interaction=nonstopmode",
                "-halt-on-error=false", f"{nome}.tex"], dove=cartella)

    if not pdf.exists():
        guai = errori_veri(cartella / f"{nome}.log")
        sys.exit("il PDF non e' stato scritto:\n  " + "\n  ".join(guai[:20]))

    finale = USCITA / f"{nome}.pdf"
    shutil.copy2(pdf, finale)
    return finale


def versione() -> tuple[str, str]:
    """Numero e data dalla voce in cima al registro degli aggiornamenti.

    E' la stessa fonte della pagina degli aggiornamenti e della scheda in
    cima all'indice: il numero non si scrive in nessun altro posto.
    """
    sys.path.insert(0, str(LIBRO / "_ext"))
    import pt_conteggi

    marca = pt_conteggi.versione_corrente(LIBRO / "_dati" / "aggiornamenti.yml")
    return marca.get("versione", ""), marca.get("data_versione", "")


def note_di_rilascio(numero: str) -> str:
    """Le note della release, dalla voce del registro.

    Si riusa quello che e' gia' scritto per i lettori invece di riscriverlo:
    una voce dice «che cosa e' cambiato», ed e' esattamente quello che uno si
    aspetta di leggere in una release.
    """
    import yaml

    registro = yaml.safe_load(
        (LIBRO / "_dati" / "aggiornamenti.yml").read_text(encoding="utf-8"))
    voce = next((v for v in registro.get("versioni", [])
                 if str(v.get("versione")) == numero), None)
    if not voce:
        return f"Paithon Book {numero}."

    righe = [f"Il libro completo in PDF, versione {numero}.", ""]
    for v in voce.get("voci", []):
        testo = (v.get("testo") or "").strip()
        if testo:
            righe.append(f"- {testo}")
    righe += ["", f"La versione online, sempre aggiornata e con il codice da "
                  f"eseguire: https://{CASA}/main/"]
    return "\n".join(righe)


def rilascia(pdf: pathlib.Path, conferma: bool) -> None:
    """Allega il PDF alla release del repository pubblico.

    E' un comando a parte, non un passo di `pubblica.py`: le azioni verso
    l'esterno si fanno in modo esplicito, non si infilano in un comando
    composto. Vale per il push pubblico e vale identico qui.
    """
    numero, data = versione()
    if not numero:
        sys.exit("il registro degli aggiornamenti non dice la versione")
    if not pdf.exists():
        sys.exit(f"il PDF non c'e': {pdf}. Prima si costruisce.")

    # CHI firma. Una release porta il nome dell'account attivo di `gh`, e su
    # questa macchina ce n'e' piu' di uno: il primo giro sarebbe uscito a nome
    # di un account personale invece che del progetto. Non e' un dettaglio
    # cosmetico, e' la firma di quello che il pubblico scarica.
    chi = esegui(["gh", "api", "user", "--jq", ".login"]).stdout.strip()
    if chi != FIRMA:
        sys.exit(f"gh e' attivo come «{chi}», e una release del libro si firma "
                 f"«{FIRMA}».\n  gh auth switch --user {FIRMA}")

    tag = f"v{numero}"
    esiste = esegui(["gh", "release", "view", tag, "--repo", PUBBLICO])
    peso = pdf.stat().st_size / 1024 / 1024

    print(f"  versione {numero} ({data}), {peso:.1f} MB")
    print(f"  release {tag} su {PUBBLICO}: "
          f"{'esiste già, sostituisco il file' if esiste.returncode == 0 else 'da creare'}")
    if not conferma:
        print("\n  prova a vuoto. Per farlo davvero: --rilascia --conferma")
        return

    note = USCITA / "note.md"
    note.write_text(note_di_rilascio(numero), encoding="utf-8")
    if esiste.returncode == 0:
        fatto = esegui(["gh", "release", "upload", tag, str(pdf),
                        "--repo", PUBBLICO, "--clobber"])
    else:
        fatto = esegui(["gh", "release", "create", tag,
                        "--repo", PUBBLICO,
                        "--title", f"Paithon Book {numero}",
                        "--notes-file", str(note), str(pdf)])
    if fatto.returncode != 0:
        sys.exit(f"la release non e' andata:\n{fatto.stderr}")
    print(f"  fatto: https://github.com/{PUBBLICO}/releases/tag/{tag}")


def racconta(pdf: pathlib.Path) -> None:
    try:
        import fitz
    except ImportError:
        print(f"  {pdf.relative_to(RADICE)} "
              f"({pdf.stat().st_size // 1024 // 1024} MB)")
        return
    documento = fitz.open(pdf)
    print(f"  {pdf.relative_to(RADICE)}: {documento.page_count} pagine, "
          f"{pdf.stat().st_size / 1024 / 1024:.1f} MB")


def main() -> None:
    argomenti = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    argomenti.add_argument("--solo-tex", action="store_true",
                           help="si ferma al sorgente LaTeX")
    argomenti.add_argument("--pulisci", action="store_true",
                           help="butta la cartella di build e la cache")
    argomenti.add_argument("--passate", type=int, default=3)
    argomenti.add_argument("--capitolo", metavar="CARTELLA",
                           help="provino di un capitolo solo (per es. "
                                "VisioneArtificiale): compila in secondi, "
                                "ma i numeri di pagina non sono quelli veri")
    argomenti.add_argument("--rilascia", action="store_true",
                           help="allega il PDF alla release del repo pubblico")
    argomenti.add_argument("--conferma", action="store_true",
                           help="con --rilascia: lo fa davvero")
    scelte = argomenti.parse_args()

    if scelte.rilascia:
        rilascia(USCITA / f"{NOME}.pdf", conferma=scelte.conferma)
        return

    print("costruisco il sorgente LaTeX...")
    cartella = costruisci_tex(pulisci=scelte.pulisci)
    if scelte.solo_tex:
        return

    nome = NOME
    passate = scelte.passate
    if scelte.capitolo:
        provino = ritaglia_capitolo(cartella / f"{NOME}.tex", scelte.capitolo)
        nome = provino.stem
        passate = min(passate, 2)   # niente indice generale da far quadrare
        print(f"provino del capitolo {scelte.capitolo}")

    print("compilo con LuaLaTeX...")
    pdf = compila(cartella, passate=passate, nome=nome)

    guai = errori_veri(cartella / f"{nome}.log")
    racconta(pdf)
    if guai:
        print(f"\n{len(guai)} cose che LuaLaTeX non ha digerito:")
        for riga in guai[:25]:
            print(f"  {riga}")
        if len(guai) > 25:
            print(f"  ... e altre {len(guai) - 25}")


if __name__ == "__main__":
    main()
