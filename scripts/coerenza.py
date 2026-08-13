#!/usr/bin/env python3
"""
Cerca incoerenze nel libro: riferimenti che non portano da nessuna parte.

Nasce da un'osservazione fatta assorbendo il corpus del magazine: quasi tutte
le sezioni che mancavano davvero non erano temi dimenticati, erano **buchi di
coerenza**. Il libro usava i decoratori senza averli spiegati, rimandava LoRA a
una sezione che non esisteva, insegnava la U bias-varianza e altrove descriveva
modelli da miliardi di parametri che la infrangono.

Quelli si trovavano solo leggendo. Questi si trovano contando, e vale la pena
toglierli di mezzo prima di rileggere:

  numref     riferimenti a figure/tabelle senza un :name: corrispondente
  cite       chiavi bibliografiche assenti dal .bib
  ref/doc    riferimenti interni a target inesistenti
  figure     file in book/figures/ che nessuno richiama
  toc        file sotto book/ non elencati in _toc.yml
  landing    schede di intro.md che non corrispondono ai capitoli del _toc.yml
             (mancanti, di troppo, fuori ordine, o col numero scritto a mano)
  animazioni capitoli senza nemmeno una figura animata (solo elenco: zero clip
             puo' essere una scelta), e capitoli oltre il tetto di 10
  schede     gruppi di tab contigui, che `sphinx-inline-tabs` FONDE in un
             gruppo solo con quattro linguette (si vede solo in build)
  ricordare  riquadri «Da ricordare» FUORI dalle schede, mentre
             `aggiornamenti.md` promette al lettore che sono sui due livelli
             (solo elenco: fuori dalle schede non e' di per se' un difetto)
  clip       scene Manim il cui sorgente e' piu' recente della GIF pubblicata,
             e con un diff che il render puo' vedere: la clip online non e'
             quella che il sorgente descrive
  palette    colori fuori palette nelle figure scritte a mano (quelle generate
             le rifiuta gia' `scrivi()` in paithon_svg.py)
  ambiente   scene Manim che l'ambiente di oggi non saprebbe piu' rendere
  lineette   i due travestimenti della lineetta che `CLAUDE.md` vieta: la
             doppia `--` e il trattino singolo spaziato
  stampa     cio' che serve al PDF e puo' restare indietro in silenzio: i tre
             fermi immagine di ogni animazione (mancanti, piu' vecchi
             dell'animazione, orfani) e le figure che dichiarano una famiglia
             di font generica invece di quelle del brand
  avanti     rimandi in avanti ("vedremo", "prossima sezione"), solo elenco,
             la verifica resta umana

  scripts/coerenza.py            # tutto
  scripts/coerenza.py --solo numref,cite

Un asse PROVATO E SCARTATO, per non rifarlo: "termini usati prima di essere
definiti", ricostruendo l'ordine di lettura dal _toc.yml e usando il grassetto
come marca di introduzione (e' la convenzione del libro). Ha prodotto 474
risultati quasi tutti rumore, parole comuni che capita siano in grassetto
("ottimizzazione", "stessa", "a mano") e la front matter che anticipa tutto il
libro per mestiere. Filtrando su intro/Introduzione e richiedendo che il
termine sia anche un titolo di sezione si scende a 11, e di quegli 11 **nessuno
e' un difetto reale**: l'unico plausibile, `transfer learning`, e' usato in
corsivo tre volte prima del suo capitolo ma gia' glossato in una riga in
DeepLearning/overview.

La differenza col controllo sul codice e' il vocabolario: i costrutti Python
sono un insieme chiuso e riconoscibile, i termini tecnici no. Su questo libro
l'asse dell'ordine di introduzione e' esaurito.

Un SECONDO asse provato e scartato (agosto 2026), che sembrava aggirare il
problema di sopra e non lo aggira. La revisione a tre lenti ha trovato che il
difetto piu' diffuso del libro e' un termine introdotto **solo dentro una scheda
Superiore** e poi usato nel testo comune, dove il lettore Elementare lo incontra
senza averlo mai visto: succede con GLA, Mamba-2, NT-Xent, «bias induttivo».
Sembra contabile: si guarda in quale zona (fuori dalle schede / Elementare /
Superiore) cade ogni occorrenza in grassetto, e si segnala chi e' in grassetto
solo in Superiore ma compare anche altrove.

Ha dato **625 risultati, quasi tutti rumore**, e per la stessa ragione di
prima: nel libro il grassetto non marca solo l'introduzione di un termine, marca
anche l'enfasi. Fra i primi risultati ci sono «a sinistra», «accanto»,
«aggiorna», «10 minuti». Il segnale c'e' ma sta sotto il rumore, e un elenco di
625 righe di cui novanta per cento false non lo legge nessuno.

Quel difetto lo trovano bene i lettori (un agente che legge il capitolo come
farebbe un tredicenne ne ha trovati 118 bloccanti in undici capitoli, con la
causa giusta): e' una cosa che si vede leggendo, non contando. Il conteggio che
invece funziona, e che sta qui sotto, e' quello dei riquadri «Da ricordare»
fuori dalle schede, perche' li' la marca e' una direttiva MyST e non una
convenzione tipografica.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

QUI = Path(__file__).resolve().parent.parent
LIBRO = QUI / "book"

RX = {
    "numref": re.compile(r"\{numref\}`([^`<]+?)`"),
    "ref": re.compile(r"\{ref\}`(?:[^<`]*<)?([^`>]+)>?`"),
    "doc": re.compile(r"\{doc\}`(?:[^<`]*<)?([^`>]+)>?`"),
    "cite": re.compile(r"\{cite\}`([^`]+)`"),
    "name": re.compile(r"^\s*:name:\s*(\S+)", re.M),
    "label": re.compile(r"^\(([^)]+)\)=\s*$", re.M),
    # \S* GREEDY fino all'ultimo slash: con il non-greedy si catturava
    # "figures/nome.svg" invece di "nome.svg" e ogni figura risultava orfana.
    "fig_uso": re.compile(r"\{figure\}\s*\S*/([^\s`/]+)"),
    "img_uso": re.compile(r"!\[[^\]]*\]\(\S*/([^\s)/]+)\)"),
    "avanti": re.compile(
        r"[^.\n]*\b(vedremo|vedrai|prossima sezione|prossimo capitolo|"
        r"piu[' ]avanti|torneremo|approfondiremo)\b[^.\n]*\.", re.I),
}


def sorgenti() -> dict[str, str]:
    # _static/ contiene il submodule `brand` con i suoi README: non e' testo del
    # libro e non va nel _toc.yml.
    #
    # _build/ e' l'uscita di Sphinx, e va esclusa perche' contiene una COPIA di
    # ogni sorgente sotto `_sources/`: una cartella di build dimenticata in giro
    # fa contare ogni file due volte e il totale passa da 6 problemi a 258, tutti
    # fantasmi. Chi ci casca smette di fidarsi del controllo, che e' il danno
    # peggiore che un controllo possa fare.
    return {str(p.relative_to(LIBRO)): p.read_text(encoding="utf-8", errors="ignore")
            for p in sorted(list(LIBRO.rglob("*.md")) + list(LIBRO.rglob("*.ipynb")))
            if "_static" not in p.parts and "_build" not in p.parts}


def chiavi_bib() -> set[str]:
    chiavi = set()
    for b in LIBRO.rglob("*.bib"):
        chiavi |= set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", b.read_text(
            encoding="utf-8", errors="ignore")))
    return chiavi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo", help="controlli da eseguire, separati da virgola")
    args = ap.parse_args()
    attivi = set(args.solo.split(",")) if args.solo else {
        "numref", "cite", "ref", "figure", "toc", "landing", "animazioni",
        "schede", "ricordare", "avanti", "palette", "clip", "ambiente",
        "lineette", "stampa"}

    testi = sorgenti()
    problemi = defaultdict(list)

    # target definiti ovunque nel libro (i riferimenti sono globali)
    nomi = set()
    for t in testi.values():
        nomi |= set(RX["name"].findall(t))
        nomi |= set(RX["label"].findall(t))

    if "numref" in attivi:
        for f, t in testi.items():
            for r in RX["numref"].findall(t):
                if r.strip() not in nomi:
                    problemi["numref senza :name:"].append(f"{f}  ->  {r}")

        # Due difetti che la build non segnala e che si trovano solo contando.
        # I target di Sphinx sono GLOBALI: due figure diverse con lo stesso
        # :name: si costruiscono senza un avviso, e uno dei due {numref}
        # stampa il numero dell'altra. Lo stesso file richiamato da due
        # `{figure}` produce invece due figure numerate diverse con la stessa
        # immagine, che al lettore sembra una svista, e di solito lo e'.
        dove_nome, dove_file = defaultdict(list), defaultdict(list)
        for f, t in testi.items():
            for n in RX["name"].findall(t):
                dove_nome[n].append(f)
            for g in RX["fig_uso"].findall(t):
                dove_file[g].append(f)
        for n, dove in sorted(dove_nome.items()):
            if len(dove) > 1:
                problemi[":name: usato piu' volte"].append(
                    f"{n}  ->  {', '.join(dove)}")
        for g, dove in sorted(dove_file.items()):
            if len(dove) > 1:
                problemi["stessa figura richiamata piu' volte"].append(
                    f"{g}  ->  {', '.join(dove)}")

    if "ref" in attivi:
        for f, t in testi.items():
            for r in RX["ref"].findall(t) + RX["doc"].findall(t):
                r = r.strip()
                if r in nomi:
                    continue
                # {doc} punta a un file: accetta se esiste
                if any(k.startswith(r.lstrip("/")) or Path(k).stem == Path(r).stem
                       for k in testi):
                    continue
                problemi["ref/doc senza target"].append(f"{f}  ->  {r}")

    if "cite" in attivi:
        bib = chiavi_bib()
        if bib:
            for f, t in testi.items():
                for gruppo in RX["cite"].findall(t):
                    for c in (x.strip() for x in gruppo.split(",")):
                        if c and c not in bib:
                            problemi["cite senza voce nel .bib"].append(f"{f}  ->  {c}")
        else:
            problemi["cite senza voce nel .bib"].append("(nessun .bib trovato)")

    if "figure" in attivi:
        usate = set()
        for t in testi.values():
            usate |= set(RX["fig_uso"].findall(t)) | set(RX["img_uso"].findall(t))
        # Non tutto ciò che sta in figures/ lo richiama una `{figure}`: la
        # favicon la nomina `_config.yml`, e un template può linkare un'icona.
        # Cercarle solo nelle pagine le farebbe risultare orfane per sempre.
        for extra in [LIBRO / "_config.yml", *(LIBRO / "_templates").rglob("*.html"),
                      *(LIBRO / "_static").glob("*.html")]:
            if extra.is_file():
                testo = extra.read_text(encoding="utf-8", errors="replace")
                usate |= {m for m in re.findall(r"figures/([\w.-]+)", testo)}
        for p in sorted((LIBRO / "figures").glob("*")):
            if not p.is_file() or p.name in usate:
                continue
            # I fermi immagine (`figures/fermi/`) non li richiama nessuna
            # pagina per progetto: li mette in pagina il PDF, e chi li
            # controlla e' l'asse `stampa`. Segnalarli qui ogni volta
            # insegnerebbe solo a ignorare l'elenco.
            if p.is_dir() or p.parent.name == "fermi":
                continue
            problemi["figure mai richiamate"].append(p.name)

    if "toc" in attivi:
        toc = (LIBRO / "_toc.yml")
        if toc.is_file():
            testo_toc = toc.read_text(encoding="utf-8")
            elencati = set(re.findall(r"file:\s*(\S+)", testo_toc))
            elencati |= set(re.findall(r"^root:\s*(\S+)", testo_toc, re.M))
            elencati = {e if "." in Path(e).name else e for e in elencati}
            for f in testi:
                stem = re.sub(r"\.(md|ipynb)$", "", f)
                if stem not in elencati and f not in elencati:
                    problemi["file non nel _toc.yml"].append(f)

    if "landing" in attivi:
        # La griglia della landing e' l'indice del libro scritto a mano: i
        # numeri li conta il CSS (contatore `pt-scheda`), ma che ci sia una
        # scheda per capitolo, nello stesso ordine, non lo controlla nessuno.
        toc = LIBRO / "_toc.yml"
        landing = LIBRO / "intro.md"
        if toc.is_file() and landing.is_file():
            testo_toc = toc.read_text(encoding="utf-8")
            # capitoli = voci a due spazi di rientro (le sezioni ne hanno quattro)
            capitoli = [f for f in re.findall(r"^  - file:\s*(\S+)", testo_toc, re.M)
                        if "/" in f]
            attese = [re.sub(r"\.(md|ipynb)$", ".html", f) for f in capitoli]
            testo_landing = landing.read_text(encoding="utf-8")
            schede = re.findall(r'class="pt-card"\s+href="([^"]+)"', testo_landing)
            for mancante in [a for a in attese if a not in schede]:
                problemi["capitoli senza scheda nella landing"].append(mancante)
            for extra in [s for s in schede if s not in attese]:
                problemi["schede della landing senza capitolo"].append(extra)
            if schede != attese and not (set(schede) ^ set(attese)):
                problemi["schede della landing fuori ordine"].append(
                    "l'ordine delle schede non e' quello del _toc.yml")
            for numero in re.findall(r'<span class="pt-card-num">([^<]+)</span>',
                                     testo_landing):
                problemi["numero scritto a mano nella scheda"].append(
                    f"«{numero}», lo conta il CSS, lo <span> va lasciato vuoto")

    if "schede" in attivi:
        # `sphinx-inline-tabs` unisce in UN SOLO gruppo le schede che si
        # toccano: due coppie Elementare/Superiore separate solo da righe
        # vuote diventano una barra da quattro linguette, con «Elementare»
        # scritto due volte. Il sorgente e' corretto, la build non protesta, e
        # il difetto si vede solo aprendo la pagina: infatti quattro casi erano
        # online da mesi. Il rimedio e' una riga di testo comune in mezzo, che
        # e' anche la regola editoriale («le tab avvolgono i passaggi dove la
        # profondita' fa la differenza, intervallati da testo universale»).
        #
        # Il rischio non e' accidentale ma sistematico: sdoppiare un riquadro
        # «Da ricordare» significa aggiungere una coppia di schede in fondo a
        # una pagina, cioe' esattamente dove ce n'e' gia' un'altra.
        apre = re.compile(r"^(`{4,})\{tab\}\s*(\S+)")
        for f, t in testi.items():
            if not f.endswith(".md"):
                continue
            righe = t.split("\n")
            i, gruppo, riga_inizio = 0, [], None
            def resoconto():
                if len(gruppo) > 2:
                    problemi["schede contigue, che la build fonde"].append(
                        f"{f}:{riga_inizio}  ->  {len(gruppo)} linguette: "
                        f"{' | '.join(gruppo)}")
            while i < len(righe):
                m = apre.match(righe[i])
                if not m:
                    if righe[i].strip() and gruppo:
                        resoconto()
                        gruppo, riga_inizio = [], None
                    i += 1
                    continue
                if riga_inizio is None:
                    riga_inizio = i + 1
                gruppo.append(m.group(2))
                tick = m.group(1)
                j = i + 1
                while j < len(righe) and righe[j].rstrip() != tick:
                    j += 1
                i = j + 1
            resoconto()

    if "ricordare" in attivi:
        # `book/aggiornamenti.md` promette al lettore, nero su bianco: «Ogni
        # capitolo si chiude con un riquadro "Da ricordare", scritto sui due
        # livelli come il resto del libro». Una promessa pubblica che nessuno
        # verificava: alla revisione dell'11 agosto 2026 erano 111 riquadri su
        # 237 fuori dalle schede, e cinque capitoli non ne avevano NEMMENO UNO
        # sdoppiato.
        #
        # Fuori dalle schede non e' di per se' un difetto: un riquadro scritto
        # in lingua comune va benissimo. Lo diventa quando e' scritto in lingua
        # Superiore, e quello un programma non lo sa giudicare. Quindi qui si
        # elenca e basta, e la lettura resta umana; ma l'elenco c'e', e questo
        # basta a non dimenticarsene per centoundici volte.
        apre = re.compile(r"^\s*(`{3,})(.*)$")
        for f, t in sorted(testi.items()):
            if not f.endswith(".md"):
                continue
            pila, fuori = [], 0
            for riga in t.split("\n"):
                m = apre.match(riga)
                if not m:
                    continue
                n, info = len(m.group(1)), m.group(2).strip()
                if "{admonition} Da ricordare" in info:
                    fuori += not any(pila)
                    pila.append((n, False))
                elif pila and not info and n >= pila[-1][0]:
                    pila.pop()
                else:
                    pila.append((n, info.startswith("{tab}")))
            if fuori:
                problemi["«Da ricordare» fuori dalle schede"].append(
                    f"{f}  ->  {fuori}")

    if "animazioni" in attivi:
        # Le clip non le contava nessuno, ed e' cosi' che venti capitoli sono
        # nati senza. `animazioni/README.md` scriveva gia' «i capitoli non
        # ancora coperti sono la lista dei prossimi»: una lista scritta a mano,
        # che infatti e' rimasta ferma mentre il libro raddoppiava. Contarla
        # non invecchia.
        #
        # Zero clip NON e' un errore e non entra nel totale: il tetto di 5-10
        # per capitolo e' un tetto, non una quota, e un capitolo dove il tempo
        # non e' mai il contenuto sta bene fermo. E' un elenco da guardare, per
        # decidere una volta e non dimenticarsene per venti capitoli.
        toc = LIBRO / "_toc.yml"
        if toc.is_file():
            animate = set()
            for p in sorted((LIBRO / "figures").glob("*")):
                if not p.is_file():
                    continue
                if p.suffix == ".gif":
                    animate.add(p.name)
                elif p.suffix == ".svg" and "@keyframes" in p.read_text(
                        encoding="utf-8", errors="ignore"):
                    animate.add(p.name)
            capitoli = [f for f in re.findall(
                r"^  - file:\s*(\S+)", toc.read_text(encoding="utf-8"), re.M)
                if "/" in f]
            for cap in capitoli:
                cartella = cap.split("/")[0]
                usate = set()
                for f, t in testi.items():
                    if f.startswith(cartella + "/"):
                        usate |= set(RX["fig_uso"].findall(t))
                        usate |= set(RX["img_uso"].findall(t))
                clip = sorted(usate & animate)
                if not clip:
                    problemi["capitoli senza figure animate"].append(cartella)
                elif len(clip) > 10:
                    problemi["capitoli oltre il tetto di 10 clip"].append(
                        f"{cartella}  ->  {len(clip)}: {', '.join(clip)}")

        # La tabella in fondo ad `animazioni/README.md` elenca le clip che
        # esistono, ed e' scritta a mano tre paragrafi sotto la riga che spiega
        # perche' gli elenchi scritti a mano invecchiano. Infatti e' invecchiata
        # anche lei: a fine campagna dichiarava diciassette clip su
        # trentaquattro generatori. Non la si genera (a un umano che apre il
        # README serve leggerla li'), la si verifica.
        readme = QUI / "animazioni" / "README.md"
        if readme.is_file():
            elencati = set(re.findall(r"^\|\s*`([\w./-]+\.py)`",
                                      readme.read_text(encoding="utf-8"), re.M))
            for gen in sorted(list((QUI / "animazioni").glob("*.py"))
                              + list((QUI / "animazioni/svg").glob("*.py"))):
                nome = gen.name if gen.parent.name == "animazioni" \
                    else f"svg/{gen.name}"
                if gen.stem == "genera":
                    continue
                # Un generatore la cui figura non e' richiamata da nessuna
                # pagina non e' una clip del libro: lo dice gia' l'asse
                # `figure`, e ripeterlo qui direbbe due volte la stessa cosa.
                prodotta = (LIBRO / "figures" / f"{gen.stem}.svg").is_file() or \
                           (LIBRO / "figures" / f"{gen.stem}.gif").is_file()
                if prodotta and nome not in elencati:
                    problemi["clip fuori dalla tabella del README"].append(nome)

    if "palette" in attivi:
        # Il motore delle figure generate verifica la palette e rifiuta il file
        # (`scrivi()` in paithon_svg.py). Le figure scritte a mano non passavano
        # da nessun controllo, ed e' cosi' che 172 file su 306 hanno finito per
        # usare tre grigi che nessun documento autorizza, e cinque hanno
        # attraversato due campagne di ripulitura con le lineette ancora dentro.
        #
        # Le due cose non pesano uguale, e infatti qui non pesano uguale.
        #
        # La LINEETTA e' un difetto e basta: `CLAUDE.md` la vieta, non c'e'
        # nessuna decisione in sospeso, e chi corregge il testo non ha modo di
        # sapere che quella lineetta e' disegnata anche altrove. Conta.
        #
        # Il COLORE fuori palette e' un elenco da guardare: `#5E5852` con 927
        # usi e' una convenzione stabilita che pero' esiste solo nei file, e
        # decidere se diventa un token o si accorpa a `--pt-fg-muted` e' una
        # decisione del repository `brand`, non di questo. Finche' non e'
        # presa, farla fallire qui vorrebbe dire tenere il controllo rosso per
        # una cosa che da qui non si puo' chiudere.
        AMMESSI = {"#B5532C", "#2D5A5C", "#C9A961", "#1A1A1A", "#F8F5EE",
                   "#5A524A", "#E2DCC9", "#C5BEAA", "#FFFFFF"}
        rx_col = re.compile(r"#[0-9A-Fa-f]{6}")
        fuori = defaultdict(int)
        for p in sorted((LIBRO / "figures").glob("*.svg")):
            t = p.read_text(encoding="utf-8", errors="ignore")
            if "\u2014" in t or "&#8212;" in t:
                problemi["lineette dentro le figure"].append(p.name)
            if "<script" in t:
                problemi["script dentro le figure"].append(p.name)
            for c in {x.upper() for x in rx_col.findall(t)} - AMMESSI:
                fuori[c] += 1
        for c, n in sorted(fuori.items(), key=lambda kv: -kv[1]):
            problemi["colori fuori palette (decisione del repo brand)"].append(
                f"{c}  ->  in {n} figure")


    if "clip" in attivi:
        # Le SVG generate hanno `animazioni/svg/genera.py --verifica`, che le
        # rigenera in un temporaneo e le confronta. Le clip Manim no: sono
        # video, rigenerarle costa, e due render della stessa scena non danno
        # comunque file identici. Risultato: fra `animazioni/*.py` e
        # `book/figures/*.gif` non c'era **nessun** legame verificabile, e si
        # poteva correggere il sorgente, committare, e lasciare online la clip
        # vecchia senza che niente protestasse. E' successo con
        # `diffusione-denoising`, che mostrava il passo deterministico di DDIM
        # sotto un testo che spiega perche' quello di DDPM e' un altro: una
        # formula dentro un'immagine, che nessuna grep puo' trovare.
        #
        # Qui non si confrontano i video, si confrontano le date dei commit. Ma
        # la sola data grida al lupo, e un controllo che segnala cose che non
        # sono difetti viene disattivato, il che e' peggio di non averlo: la
        # campagna di luglio contro le lineette ha toccato la **docstring** di
        # una scena e un **commento** di un'altra, e ne' l'una ne' l'altro
        # finiscono nel video. Quindi si guarda anche il diff, e si segnala solo
        # quando cambia una riga che il render puo' vedere.
        import ast

        def _ct(rel):
            r = subprocess.run(["git", "log", "-1", "--format=%ct", "--", rel],
                               cwd=QUI, capture_output=True, text=True)
            try:
                return int(r.stdout.strip())
            except ValueError:
                return 0

        for sorg in sorted((QUI / "animazioni").glob("*.py")):
            gif = LIBRO / "figures" / (sorg.stem + ".gif")
            if not gif.is_file():
                continue
            if not (_ct("animazioni/" + sorg.name) > _ct("book/figures/" + gif.name)):
                continue
            sha = subprocess.run(
                ["git", "log", "-1", "--format=%H", "--", "book/figures/" + gif.name],
                cwd=QUI, capture_output=True, text=True).stdout.strip()
            diff = subprocess.run(
                ["git", "log", sha + "..HEAD", "-U0", "--format=", "--",
                 "animazioni/" + sorg.name],
                cwd=QUI, capture_output=True, text=True).stdout
            try:
                testa = ast.parse(sorg.read_text(encoding="utf-8")).body[0]
                fine_doc = (testa.end_lineno or 0) if (
                    isinstance(testa, ast.Expr)
                    and isinstance(getattr(testa, "value", None), ast.Constant)
                    and isinstance(testa.value.value, str)) else 0
            except (SyntaxError, IndexError):
                fine_doc = 0
            vive, n = [], 0
            for r in diff.split("\n"):
                if r.startswith("@@"):
                    try:
                        n = int(r.split("+")[1].split(",")[0].split(" ")[0])
                    except (IndexError, ValueError):
                        n = fine_doc + 1
                    continue
                if r[:3] in ("+++", "---") or r[:1] not in "+-":
                    continue
                if n > fine_doc:
                    vive.append(r[1:].strip())
                if r[:1] == "+":
                    n += 1
            if any(x and not x.startswith("#") for x in vive):
                problemi["clip piu' vecchie del loro sorgente"].append(
                    sorg.name + "  ->  il sorgente e' cambiato dopo il .gif, e "
                    "non solo nei commenti: va rigenerato")


    if "ambiente" in attivi:
        # L'altra meta' del problema delle clip, e la peggiore, perche' non
        # lascia tracce in git: l'**ambiente** cambia sotto le scene. Una
        # versione nuova di Manim rinomina una classe, un LaTeX nuovo compone
        # una formula in un altro modo, e le scene smettono di partire senza
        # che nessun file del repository sia stato toccato. L'asse `clip` qui
        # sopra non puo' vederlo: guarda le date dei commit, e di commit non ce
        # ne sono.
        #
        # `animazioni/ambiente.json` e' il manifesto delle versioni con cui le
        # scene sono state viste girare; `collaudo.py --verifica` lo confronta
        # con l'immagine di oggi. Se Docker non c'e' non e' un difetto del
        # libro, e' una mancanza della macchina: si tace.
        collaudo = QUI / "animazioni" / "collaudo.py"
        if collaudo.is_file() and shutil.which("docker"):
            r = subprocess.run([sys.executable, str(collaudo), "--verifica"],
                               cwd=QUI, capture_output=True, text=True)
            if r.returncode != 0:
                for riga in r.stdout.splitlines():
                    if riga.strip() and not riga.startswith("   python3"):
                        problemi["ambiente delle clip cambiato"].append(riga.strip())


    if "stampa" in attivi:
        # Il libro esiste anche come PDF (`scripts/genera-pdf.py`), e due cose
        # che gli servono possono restare indietro in silenzio: i fermi
        # immagine delle animazioni, che sono tracciati, e i font dichiarati
        # dalle figure.
        #
        # Il controllo sui fermi non si limita a chiedere se i file ci sono.
        # Tre PNG identici passerebbero qualunque verifica sull'esistenza e
        # non racconterebbero niente: il senso della terzina e' far vedere che
        # c'era un prima e un dopo.
        fermi = LIBRO / "figures" / "fermi"
        anima = [p for p in sorted((LIBRO / "figures").glob("*.gif"))]
        for svg in sorted((LIBRO / "figures").glob("*.svg")):
            if "@keyframes" in svg.read_text(encoding="utf-8", errors="ignore"):
                anima.append(svg)

        attesi = set()
        for sorgente in anima:
            tre = [fermi / f"{sorgente.stem}-{n}.png" for n in (1, 2, 3)]
            attesi.update(f.name for f in tre)
            mancano = [f.name for f in tre if not f.is_file()]
            if mancano:
                problemi["fermi immagine mancanti"].append(
                    f"{sorgente.name}  ->  {', '.join(mancano)}")
            elif min(f.stat().st_mtime for f in tre) < sorgente.stat().st_mtime:
                problemi["fermi immagine piu' vecchi dell'animazione"].append(
                    f"{sorgente.name}  (python3 animazioni/fermi.py)")

        if fermi.is_dir():
            for p in sorted(fermi.glob("*.png")):
                # I fogli a contatto cominciano per punto e non sono fermi:
                # si guardano e si buttano, e infatti non sono tracciati.
                if p.name.startswith(".") or p.name in attesi:
                    continue
                problemi["fermi immagine orfani"].append(p.name)

        # Le figure devono dichiarare i font del brand. Non e' un vezzo da
        # stampa: un `font-family="sans-serif"` generico, come file autonomo,
        # prende il font di sistema di chi legge, ed e' cosi' che per mesi
        # meta' delle figure del libro sono uscite in Arial anche online.
        # La lineetta SPAZIATA dentro una figura. L'asse `palette` cerca la
        # doppia `--` e la lineetta lunga; questa terza forma passava, ed e'
        # la piu' facile da scrivere per sbaglio. Le sottrazioni fra numeri
        # (`2026 - 2017`) non contano: li' e' un segno meno.
        rx_lin = re.compile(r"(?<![\d>])\s\-\s(?![\d<])")
        for p in sorted((LIBRO / "figures").glob("*.svg")):
            for riga in re.findall(r">([^<>]{3,120})<",
                                   p.read_text(encoding="utf-8", errors="ignore")):
                if rx_lin.search(riga):
                    problemi["lineette dentro le figure"].append(
                        f"{p.name}  ->  {riga.strip()[:60]}")

        rx_fam = re.compile(r'font-family\s*[:=]\s*["\']?([^;"\'>}]+)', re.I)
        for p in sorted((LIBRO / "figures").glob("*.svg")):
            testo = p.read_text(encoding="utf-8", errors="ignore")
            for famiglia in rx_fam.findall(testo):
                prima = famiglia.split(",")[0].strip().strip("'\"")
                if prima.lower() in ("sans-serif", "serif", "monospace",
                                     "georgia", "arial", "helvetica",
                                     "times new roman", ""):
                    problemi["figure che non dichiarano i font del brand"].append(
                        f"{p.name}  ->  {famiglia.strip()}")
                    break

    if "lineette" in attivi:
        # `CLAUDE.md` vieta la lineetta lunga (—) perche' non e' nello stile
        # dell'autore. Ma la stessa cosa si scrive in altri due modi, e sono i
        # due che una macchina produce piu' volentieri: la **doppia lineetta**
        # `--` e il **trattino singolo spaziato** ` - `. Vanno tenuti fuori con
        # lo stesso rigore: sono la firma piu' riconoscibile di un testo
        # generato, e a differenza della lineetta lunga non si vedono a colpo
        # d'occhio scorrendo una pagina.
        #
        # Il libro oggi e' pulito su tutte e tre le forme (misurato: zero
        # occorrenze in prosa), quindi questo controllo non ripara, impedisce di
        # cominciare.
        #
        # La parte delicata e' non gridare al lupo, perche' in un libro tecnico
        # quei segni compaiono legittimamente dappertutto: segni meno nelle
        # formule, opzioni da riga di comando, indici, commenti HTML, classi
        # CSS, separatori di tabella, elenchi. Al primo giro un interruttore
        # acceso/spento sui recinti ne ha lasciati passare cinque, perche' le
        # schede a cinque backtick lo desincronizzavano; e la matematica in
        # linea che va a capo non si vede guardando una riga per volta. Quindi
        # qui si maschera il testo **una volta sola**, con una pila per i
        # recinti e sostituendo con spazi per non spostare i numeri di riga.
        def maschera(testo: str) -> str:
            fuori, pila = [], []
            for riga in testo.split("\n"):
                m = re.match(r"^(\s*)(`{3,}|~{3,})(.*)$", riga)
                if m:
                    tick, info = m.group(2), m.group(3).strip()
                    if pila and len(tick) >= len(pila[-1]) and not info:
                        pila.pop()
                    elif info or not pila:
                        pila.append(tick)
                    else:
                        pila.pop()
                    fuori.append(" " * len(riga))
                    continue
                fuori.append(" " * len(riga) if pila else riga)
            t = "\n".join(fuori)
            # matematica: $$...$$ prima di $...$, e su tutto il testo perche'
            # una formula in linea puo' andare a capo
            for rx in (r"\$\$.*?\$\$", r"\$[^$]*?\$", r"`[^`]*`", r"https?://\S+"):
                t = re.sub(rx, lambda m: re.sub(r"[^\n]", " ", m.group(0)), t, flags=re.S)
            return t

        doppia = re.compile(r"(?<![-\w])--(?![-\w])|(?<=\w)--(?=\w)")
        singola = re.compile(r"(?<=\w) - (?=\w)")
        for f, t in sorted(testi.items()):
            if not f.endswith(".md"):
                continue
            for n, riga in enumerate(maschera(t).split("\n"), 1):
                if "<!--" in riga or "-->" in riga or '="' in riga:
                    continue
                if re.match(r"^\s*\|[\s|:-]+\|\s*$", riga) or re.match(r"^\s*-{3,}\s*$", riga):
                    continue
                pulita = re.sub(r"^\s*[-*+]\s", "", riga)
                if doppia.search(pulita) or singola.search(pulita):
                    problemi["lineette scritte in ASCII (-- oppure - )"].append(
                        f"{f}:{n}  {pulita.strip()[:88]}")

    if "avanti" in attivi:
        for f, t in testi.items():
            for frase in RX["avanti"].findall(t):
                pass  # findall coi gruppi non serve: conta il testo intero
            for m in RX["avanti"].finditer(t):
                problemi["rimandi in avanti (da leggere)"].append(
                    f"{f}: …{m.group(0).strip()[:110]}")

    # `ordine` non e' solo l'ordine di stampa: e' anche il filtro. Una chiave
    # che non compare qui viene calcolata e poi buttata via in silenzio, ed e'
    # esattamente quel che e' successo ai due controlli sui doppioni.
    ordine = ["numref senza :name:", ":name: usato piu' volte",
              "stessa figura richiamata piu' volte",
              "cite senza voce nel .bib",
              "ref/doc senza target", "figure mai richiamate",
              "file non nel _toc.yml",
              "capitoli senza scheda nella landing",
              "schede della landing senza capitolo",
              "schede della landing fuori ordine",
              "numero scritto a mano nella scheda",
              "schede contigue, che la build fonde",
              "capitoli oltre il tetto di 10 clip",
              "clip fuori dalla tabella del README",
              "capitoli senza figure animate",
              "lineette scritte in ASCII (-- oppure - )",
              "«Da ricordare» fuori dalle schede",
              "clip piu' vecchie del loro sorgente",
              "ambiente delle clip cambiato",
              "lineette dentro le figure",
              "script dentro le figure",
              "colori fuori palette (decisione del repo brand)",
              "fermi immagine mancanti",
              "fermi immagine piu' vecchi dell'animazione",
              "fermi immagine orfani",
              "figure che non dichiarano i font del brand",
              "rimandi in avanti (da leggere)"]
    # Assi che elencano e basta: dicono cosa guardare, non cosa e' rotto, e
    # quindi non fanno fallire niente.
    solo_elenco = {"rimandi in avanti (da leggere)",
                   "capitoli senza figure animate",
                   "colori fuori palette (decisione del repo brand)",
                   "«Da ricordare» fuori dalle schede"}
    totale = 0
    for k in ordine:
        v = problemi.get(k)
        if not v:
            continue
        print(f"\n=== {k} ({len(v)})")
        mostra = v if k != "rimandi in avanti (da leggere)" else v[:12]
        for riga in mostra:
            print(f"   {riga}")
        if len(v) > len(mostra):
            print(f"   … e altri {len(v) - len(mostra)}")
        if k not in solo_elenco:
            totale += len(v)

    print(f"\n{totale} problemi da correggere"
          f" ({len(problemi.get('rimandi in avanti (da leggere)', []))} rimandi da leggere a mano,"
          f" {len(problemi.get('capitoli senza figure animate', []))} capitoli senza clip)")
    return 1 if totale else 0


if __name__ == "__main__":
    sys.exit(main())
