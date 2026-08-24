#!/usr/bin/env python3
"""Il libro intero in un PDF: `paithon-book.pdf`.

    python3 scripts/genera-pdf.py              # costruisce il PDF
    python3 scripts/genera-pdf.py --solo-tex   # si ferma al sorgente LaTeX
    python3 scripts/genera-pdf.py --pulisci    # butta la cache delle figure
    python3 scripts/genera-pdf.py --verifica   # il PDF c'e' ed e' aggiornato?
    python3 scripts/genera-pdf.py --capitoli  # ritaglia un PDF per capitolo
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
CAPITOLI = USCITA / "capitoli"
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
#
# `not found in language.dat.lua` e' entrato dopo: e' un *Warning*, quindi non
# somigliava a un guasto, e per questo e' rimasto in pagina a lungo. Senza i
# pattern di sillabazione LuaLaTeX non spezza le parole a fine riga, e il libro
# esce con centinaia di righe che sbordano o troppo spaziate; ma la build
# riesce, il codice di uscita e' 0 e il PDF c'e'. Vale la pena tenere presente
# come si e' scoperto: non da un controllo, ma aprendo il PDF e leggendolo.
# Si ripara sulla macchina (`texlive-lang-italian`), non nel libro.
#
# `undefined` prende i rimandi ciechi: un {doc} verso una pagina che in stampa
# non c'e' (FUORI_STAMPA in pt_stampa.py) non rompe niente, ma promette al
# lettore un capitolo che nel file che ha in mano non esiste.
GUAI = re.compile(
    r"^(?:!.*|.*?Package \S+ Error.*|.*?LaTeX Error.*"
    r"|.*?File .* not found.*|.*?Missing character.*"
    r"|.*?not found in language\.dat\.lua.*"
    r"|.*?(?:Hyper reference|Citation|Reference) .*undefined.*)$", re.M)


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

    # I fermi immagine delle animazioni li mette in pagina `pt_stampa`, che
    # gira a doctree RISOLTO, cioe' dopo che Sphinx ha gia' raccolto e copiato
    # le immagini: quei PNG non li vede nessuno e nella cartella del .tex non
    # arrivano. Non se n'era accorto nessuno perche' la build e' incrementale e
    # li trovava avanzati da un giro precedente; alla prima build pulita sono
    # spariti tutti e 132 in una volta, con 365 errori di LuaLaTeX che pero'
    # non fermano niente (gira in nonstopmode) e un PDF che esce lo stesso, coi
    # buchi. Si copiano qui, esplicitamente.
    fermi = LIBRO / "figures" / "fermi"
    if fermi.is_dir():
        dove = TEX / "figures" / "fermi"
        dove.mkdir(parents=True, exist_ok=True)
        for png in fermi.glob("*.png"):
            shutil.copy2(png, dove / png.name)
        print(f"  fermi immagine copiati: {len(list(dove.glob('*.png')))}")
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
        fetta_test = testo[taglio:fine]
        # Le pagine di radice (prefazione, intro) non stanno in una
        # cartella: la loro etichetta e' `nome:`, non `nome/`.
        if any(m in fetta_test for m in (f"{{{nome}/", f"{nome}/", f"{nome}:")):
            scelto = (taglio, fine)
            break
    if scelto is None:
        sys.exit(f"capitolo «{nome}» non trovato (e' il nome della cartella "
                 f"sotto book/, per esempio VisioneArtificiale, o della pagina "
                 f"di radice, per esempio prefazione)")

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

    # IL NUMERO E IL FILE ARRIVANO DA DUE POSTI DIVERSI, e non e' un dettaglio:
    # il numero lo dice il registro, il file sta su disco, e finche' non li si
    # confronta possono raccontare due versioni diverse. E' successo: la 1.5.7
    # e' uscita con dentro un PDF che in copertina diceva 1.5.6, perche' era
    # stato costruito prima di aggiungere la voce al registro. Nessuno se n'era
    # accorto perche' tutto funzionava, e il numero giusto lo scriveva il tag.
    # La copertina il numero lo stampa: si legge di li'.
    stampata = versione_stampata(pdf)
    if stampata and stampata != numero:
        sys.exit(f"il PDF in copertina dice {stampata}, il registro {numero}: "
                 f"e' il PDF di un'altra versione.\n"
                 f"  Ricostruiscilo: python3 scripts/genera-pdf.py")

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

    # SOLO I CAPITOLI CAMBIATI. Il libro si ricompone per intero a ogni giro
    # (e' impaginato globalmente: un paragrafo in piu' al capitolo 3 sposta
    # tutto quello dopo), ma di ritagli ne cambiano pochi, e non ha senso
    # rispedire quaranta megabyte per tre capitoli toccati. Chi c'e' gia' e
    # com'e' fatto lo dice la release, che di ogni allegato espone lo sha256:
    # lo stato vero e' quello che il lettore scarica, non quello che questa
    # macchina ricorda.
    gia = allegati(tag)
    capitoli = sorted(CAPITOLI.glob(f"{NOME}-*.pdf"))
    # I ritagli li scrive la stessa build che scrive il libro, quindi vecchi
    # non dovrebbero esserlo mai. Se lo sono e' successo qualcosa (una build
    # interrotta, una cartella copiata a mano), e caricarli vorrebbe dire
    # mettere sulla release capitoli di due edizioni diverse.
    vecchi = [f for f in capitoli if f.stat().st_mtime < pdf.stat().st_mtime]
    if vecchi:
        sys.exit(f"{len(vecchi)} ritagli sono piu' vecchi del libro "
                 f"({vecchi[0].name} e gli altri): rifalli con\n"
                 f"  python3 scripts/genera-pdf.py --capitoli")
    da_caricare = [f for f in capitoli if gia.get(f.name) != impronta(f)]

    print(f"  versione {numero} ({data}), {peso:.1f} MB")
    print(f"  release {tag} su {PUBBLICO}: "
          f"{'esiste già, sostituisco il file' if esiste.returncode == 0 else 'da creare'}")
    if capitoli:
        nomi = ", ".join(f.stem.split("-", 2)[-1] for f in da_caricare)
        print(f"  capitoli: {len(capitoli)} sul disco, "
              f"{len(da_caricare)} da caricare"
              + (f" ({nomi})" if 0 < len(da_caricare) < len(capitoli) else ""))
    else:
        print(f"  capitoli: nessuno in {CAPITOLI.relative_to(RADICE)}, "
              f"li ritaglia `genera-pdf.py` (anche da solo, con --capitoli)")
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

    if da_caricare:
        fatto = esegui(["gh", "release", "upload", tag, "--repo", PUBBLICO,
                        "--clobber", *[str(f) for f in da_caricare]])
        if fatto.returncode != 0:
            sys.exit(f"i capitoli non sono saliti:\n{fatto.stderr}")
        print(f"  capitoli caricati: {len(da_caricare)}")
    print(f"  fatto: https://github.com/{PUBBLICO}/releases/tag/{tag}")


def ricompatta(pdf: pathlib.Path) -> None:
    """Riscrive il PDF eliminando l'imbottitura che LuaTeX ci lascia dentro.

    LuaTeX ogni tanto scrive un oggetto compresso seguito da decine di megabyte
    di byte a zero, e dichiara nella `/Length` anche quelli. Il PDF resta
    perfettamente valido e completo (stesse pagine, stessi segnalibri, stessi
    collegamenti): a cambiare e' solo il peso, che qui e' passato da 66 MB a 29
    per 39 MB di zeri in un oggetto solo. Non da' nessun errore, e l'unico
    sintomo e' la dimensione del file, che se ne accorge solo chi la confronta
    con quella della volta prima.

    Riscrivere tutto con PyMuPDF li toglie, e in piu' fissa l'identificatore
    del documento, cosi' due costruzioni dello stesso testo danno file uguali:
    e' lo stesso trattamento che `_stampa/copertina.py` fa alle copertine.
    """
    try:
        import fitz
    except ImportError:
        return
    prima = pdf.stat().st_size
    documento = fitz.open(pdf)
    pagine, segnalibri = documento.page_count, len(documento.get_toc())
    documento.xref_set_key(-1, "ID", "[<70616974686F6E><626F6F6B>]")
    rimessi = length1(documento)
    if rimessi:
        print(f"  /Length1 rimesso su {rimessi} font")
    temporaneo = pdf.with_suffix(".compatto.pdf")
    documento.save(temporaneo, garbage=3, deflate=True, no_new_id=True)
    documento.close()
    controllo = fitz.open(temporaneo)
    intatto = (controllo.page_count == pagine
               and len(controllo.get_toc()) == segnalibri)
    controllo.close()
    if not intatto:                      # non e' mai successo, ma se succede
        temporaneo.unlink()              # si tiene quello di LuaTeX
        print("  ricompattazione saltata: il risultato non combaciava")
        return
    temporaneo.replace(pdf)
    if prima - pdf.stat().st_size > 1024 * 1024:
        print(f"  ricompattato: {prima / 1024 / 1024:.0f} MB "
              f"-> {pdf.stat().st_size / 1024 / 1024:.0f} MB")


def length1(documento) -> int:
    """Rimette il `/Length1` che LuaTeX non scrive nei font che incorpora lui.

    Un `FontFile2` deve dichiarare in `/Length1` la lunghezza del programma
    TrueType non compresso: e' obbligatorio, e Acrobat lo usa per estrarre il
    font. Senza, apre il libro e avvisa che «non e' possibile estrarre il font
    incorporato», con il nome del carattere dei titoli, e aggiunge che dei
    caratteri potrebbero non stamparsi bene.

    Non lo vede nessuno degli strumenti della catena: `qpdf --check` dice che
    il file e' a posto (controlla la sintassi, non le chiavi obbligatorie), e
    gli altri lettori il font lo disegnano lo stesso. Lo dice solo Acrobat, a
    chi scarica il PDF.

    Sono i font che incorpora LuaTeX, cioe' quelli del testo; quelli che
    arrivano dentro le figure convertite da Chromium il `/Length1` ce l'hanno,
    ed e' il motivo per cui l'avviso nomina sempre i primi.
    """
    import re
    rimessi = 0
    for xref in range(1, documento.xref_length()):
        try:
            oggetto = documento.xref_object(xref, compressed=False)
        except Exception:
            continue
        if "/FontFile2" not in oggetto or "/FontName" not in oggetto:
            continue
        programma = re.search(r"/FontFile2 (\d+) 0 R", oggetto)
        if not programma:
            continue
        font = int(programma.group(1))
        if "/Length1" in documento.xref_object(font, compressed=False):
            continue
        documento.xref_set_key(font, "Length1",
                               str(len(documento.xref_stream(font))))
        rimessi += 1
    return rimessi


def pagine_dei_capitoli(documento, sorgente: pathlib.Path) -> list[tuple]:
    """Per ogni capitolo del libro: cartella, prima e ultima pagina (0-based).

    I confini non si indovinano dai titoli dell'indice, che sono prosa e
    cambiano: si prendono dalle **ancore** che hyperref lascia nel PDF
    (`chapter.1`, `chapter.2`, ...), messe in fila per pagina e accoppiate ai
    `\\chapter` del sorgente LaTeX, che portano subito sotto il `\\label` con
    il nome del file da cui vengono. Se i due conti non tornano ci si ferma:
    accoppiare per posizione due elenchi di lunghezza diversa vuol dire dare a
    ogni capitolo il PDF di quello accanto, e nessuno se ne accorgerebbe.

    Il confine di destra e' la prossima ancora, di capitolo **o di parte**: il
    frontespizio di una parte cade fra la fine di un capitolo e l'inizio del
    successivo, e senza contarlo finirebbe in coda al capitolo precedente.
    """
    testo = sorgente.read_text(encoding="utf-8")
    etichette = re.findall(
        r"^\\chapter[\[{].*\n\\label\{\\detokenize\{([^:}]+):", testo, re.M)
    ancore = sorted((v["page"], k) for k, v in documento.resolve_names().items()
                    if k.startswith("chapter."))
    if len(ancore) != len(etichette):
        sys.exit(f"nel PDF ci sono {len(ancore)} ancore di capitolo e nel "
                 f"sorgente {len(etichette)} capitoli: non si accoppiano.")
    confini = sorted(v["page"] for k, v in documento.resolve_names().items()
                     if k.startswith(("chapter.", "part.")))

    capitoli = []
    for (prima, _), etichetta in zip(ancore, etichette):
        # Le pagine di radice (prefazione, bibliografia) non stanno in una
        # cartella, e un capitolo del libro si', sempre: e' il filtro.
        if "/" not in etichetta:
            continue
        dopo = [p for p in confini if p > prima]
        ultima = (dopo[0] - 1) if dopo else documento.page_count - 1
        capitoli.append((etichetta.split("/")[0], prima, ultima))
    return capitoli


def pagina_vuota(pagina) -> bool:
    """La verso bianca che `openright` lascia in fondo a un capitolo."""
    return (not pagina.get_text().strip() and not pagina.get_images()
            and not pagina.get_drawings())


def spezza(pdf: pathlib.Path, sorgente: pathlib.Path) -> dict:
    """Il libro impaginato, tagliato in un PDF per capitolo.

    Si RITAGLIA dal libro gia' composto invece di compilare ogni capitolo per
    conto suo, e la ragione e' che un libro e' impaginato globalmente: un
    capitolo compilato da solo riparte da pagina 1, si numera «Capitolo 1» e
    ai rimandi fuori dalle sue pagine risponde `??`. Ritagliato, porta i
    numeri veri e i riferimenti risolti, e costa un secondo e mezzo per tutto
    il libro invece di una passata di LuaLaTeX per capitolo.

    Nei metadati non finisce nessuna data: il libro intero la sua ce l'ha, ma
    qui la data cambierebbe a ogni costruzione, e con essa il codice di
    controllo di ogni ritaglio, che e' proprio quello con cui `rilascia()`
    decide chi ricaricare e chi no.
    """
    import fitz

    documento = fitz.open(pdf)
    indice = documento.get_toc()
    titoli = {p: t for livello, t, p in indice if livello == 2}
    marca = dict(documento.metadata)
    for campo in ("creationDate", "modDate", "producer"):
        marca.pop(campo, None)

    CAPITOLI.mkdir(parents=True, exist_ok=True)
    fatti = {}
    for cartella, prima, ultima in pagine_dei_capitoli(documento, sorgente):
        while ultima > prima and pagina_vuota(documento[ultima]):
            ultima -= 1
        pezzo = fitz.open()
        pezzo.insert_pdf(documento, from_page=prima, to_page=ultima)
        # L'indice del pezzo e' quello del libro ristretto alle sue pagine e
        # rialzato di un livello: il capitolo, che nel libro e' figlio di una
        # parte, qui e' la radice.
        pezzo.set_toc([[livello - 1, testo, pagina - prima]
                       for livello, testo, pagina in indice
                       if livello >= 2 and prima < pagina <= ultima + 1])
        pezzo.set_metadata(dict(
            marca, title=f"{titoli.get(prima + 1, cartella)} · Paithon Book"))
        length1(pezzo)
        pezzo.xref_set_key(-1, "ID", "[<70616974686F6E><626F6F6B>]")
        file = CAPITOLI / f"{NOME}-{cartella}.pdf"
        pezzo.save(file, garbage=4, deflate=True, no_new_id=True)
        pezzo.close()
        fatti[cartella] = file

    # Un capitolo tolto dal libro lascerebbe qui il suo ritaglio, e da li'
    # salirebbe sulla release: sulla home la scheda non c'e' piu', il file si
    # scarica ancora.
    for avanzo in CAPITOLI.glob(f"{NOME}-*.pdf"):
        if avanzo not in fatti.values():
            avanzo.unlink()
            print(f"  tolto il ritaglio di un capitolo che non c'e' piu': "
                  f"{avanzo.name}")

    peso = sum(f.stat().st_size for f in fatti.values()) / 1024 / 1024
    print(f"  capitoli: {len(fatti)} file in "
          f"{CAPITOLI.relative_to(RADICE)} ({peso:.0f} MB)")
    return fatti


def impronta(file: pathlib.Path) -> str:
    """Lo stesso codice di controllo che GitHub espone sugli allegati."""
    import hashlib
    return "sha256:" + hashlib.sha256(file.read_bytes()).hexdigest()


def allegati(tag: str) -> dict:
    """Nome -> impronta di quello che sulla release c'e' gia'.

    E' la lista che permette di ricaricare **solo i capitoli cambiati**, e la
    si chiede alla release invece di tenersene una copia qui: lo stato vero e'
    quello che il lettore scarica, non quello che questa macchina ricorda.
    Se la release non esiste ancora, non c'e' niente e si carica tutto.
    """
    import json
    fatto = esegui(["gh", "api", f"repos/{PUBBLICO}/releases/tags/{tag}",
                    "--jq", "[.assets[] | {name, digest}]"])
    if fatto.returncode != 0:
        return {}
    try:
        return {a["name"]: a.get("digest") or "" for a in json.loads(fatto.stdout)}
    except (ValueError, TypeError):
        return {}


def versione_stampata(pdf: pathlib.Path) -> str | None:
    """Il numero che il PDF si porta stampato sulla copertina.

    Il nome del file non lo dice (e' sempre lo stesso) e la data di modifica
    nemmeno. La copertina invece porta «1.5.7 del 15 agosto 2026», scritto dal
    registro nel momento in cui il libro e' stato composto: e' l'unica prova
    di quale versione sia davvero quel file.
    """
    try:
        import fitz
    except ImportError:
        return None
    trovato = re.search(r"\b(\d+\.\d+\.\d+) del ", fitz.open(pdf)[0].get_text())
    return trovato.group(1) if trovato else None


def copertina_intera(pdf: pathlib.Path) -> str | None:
    """La copertina sta ancora in una pagina sola?

    Sta in una pagina per pochi millimetri: il fregio da solo prende 111mm dei
    297 del foglio. Aggiungendoci il claim, fregio e riga della versione sono
    scivolati sulla seconda pagina, e nessuno se n'e' accorto perche' nessuno
    si lamenta: LuaLaTeX non emette avvisi (dentro un `titlepage` il contenuto
    che avanza semplicemente prosegue), il codice di uscita e' 0 e il PDF c'e'.
    Si vede solo aprendolo, e la copertina uno la guarda una volta l'anno.

    Quindi si controlla: sulla prima pagina devono esserci tutte e tre le cose
    che ci mettiamo, il claim, l'indirizzo del libro e il fregio, che non e'
    testo e si riconosce dalla sua texture di punti (centinaia di tracciati:
    il resto della copertina ne ha una quindicina).
    """
    try:
        import fitz
    except ImportError:
        return None
    prima = fitz.open(pdf)[0]
    testo = prima.get_text()
    manca = [nome for nome, presente in (
        ("il claim", "che spiega due volte" in testo),
        ("l'autore", "Francesco Messina" in testo),
        ("l'indirizzo del libro", "book.paithon.it" in testo),
        ("il fregio", len(prima.get_drawings()) > 100),
    ) if not presente]
    return ", ".join(manca) or None


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
    argomenti.add_argument("--capitoli", action="store_true",
                           help="ritaglia dal PDF gia' costruito un file per "
                                "capitolo, senza ricompilare niente")
    argomenti.add_argument("--rilascia", action="store_true",
                           help="allega il PDF alla release del repo pubblico")
    argomenti.add_argument("--conferma", action="store_true",
                           help="con --rilascia: lo fa davvero")
    scelte = argomenti.parse_args()

    if scelte.rilascia:
        rilascia(USCITA / f"{NOME}.pdf", conferma=scelte.conferma)
        return

    if scelte.capitoli:
        libro, sorgente = USCITA / f"{NOME}.pdf", TEX / f"{NOME}.tex"
        if not libro.exists() or not sorgente.exists():
            sys.exit("prima si costruisce il libro: python3 scripts/genera-pdf.py")
        print("ritaglio i capitoli...")
        spezza(libro, sorgente)
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
    ricompatta(pdf)
    racconta(pdf)

    # I capitoli si ritagliano dal libro appena composto: costa un secondo e
    # mezzo, quindi non e' un passo a parte da ricordarsi.
    if not scelte.capitolo:
        spezza(pdf, cartella / f"{NOME}.tex")

    # Il provino di un capitolo la copertina non ce l'ha: si controlla il
    # libro intero, che e' poi quello che va in release.
    if not scelte.capitolo and (mancante := copertina_intera(pdf)):
        print(f"\nATTENZIONE: sulla prima pagina manca {mancante}.")
        print("  La copertina e' scivolata su due pagine: sta in una per "
              "pochi millimetri.")
        print("  Si rimedia in `book/_ext/pt_stampa.py`, stringendo la "
              "larghezza del marchio,")
        print("  il corpo del claim o i \\vspace del frontespizio.")

    if guai:
        print(f"\n{len(guai)} cose che LuaLaTeX non ha digerito:")
        for riga in guai[:25]:
            print(f"  {riga}")
        if len(guai) > 25:
            print(f"  ... e altre {len(guai) - 25}")


if __name__ == "__main__":
    main()
