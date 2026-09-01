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
  landing    schede di intro.md che non corrispondono ai capitoli del _toc.yml,
             o a cui manca il collegamento al proprio capitolo in PDF
             (mancanti, di troppo, fuori ordine, o col numero scritto a mano)
  animazioni capitoli senza nemmeno una figura animata: zero clip puo' essere
             la scelta giusta, ma va DICHIARATA in animazioni/senza-clip.toml
             con la ragione, altrimenti e' un difetto (perche' «ci abbiamo
             pensato» e «non ce lo siamo chiesti» da fuori si somigliano);
             e capitoli oltre il tetto di 10
  schede     gruppi di tab contigui, che `sphinx-inline-tabs` FONDE in un
             gruppo solo con quattro linguette; e il difetto speculare, una
             coppia SPEZZATA da un capoverso, che diventa due gruppi da una
             linguetta e lascia la Superiore visibile a chi legge in
             Elementare (tutti e due si vedono solo in build)
  simboli    una lettera che riceve DUE GLOSSE diverse nello stesso capitolo
             («$x$ e' il ...» detto in due modi): o e' una collisione, e la
             medicina e' rinominare, o e' la stessa cosa spiegata due volte
             (solo elenco: distinguerle richiede di leggere)
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
  matematica segni di prosa finiti dentro `$...$`: l'apostrofo tipografico
             al posto della derivata prima, virgolette, lineette. Nessuno se ne
             accorge (Sphinx non avvisa, MathJax li disegna), e una campagna di
             ripulitura tipografica ne ha piazzati 36 in una volta sola
  doppioni   la stessa frase scritta due volte nello stesso capoverso. Nasce
             dalle riscritture: si sostituisce un pezzo di testo con uno nuovo
             che riprende quello che diceva la frase dopo, e il capoverso
             finisce per dirla due volte. Ne sono state trovate quattordici in
             una passata sola, in nove capitoli, e nessuna l'ha vista un
             controllo: la build non se ne accorge e il testo resta valido
  verso      rimandi che vanno dalla parte sbagliata: «come abbiamo visto»
             seguito da un capitolo che il lettore non ha ancora letto (e il
             contrario, «vedremo» verso un capitolo gia' letto). L'ordine di
             lettura sta nel _toc.yml, il link risolve benissimo e nessuna
             build se ne accorge: trovato a mano in Visione e linguaggio (4
             casi, verso i capitoli 21 e 24), in Audio (6) e in Speech (2)

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

Un TERZO asse provato e scartato (agosto 2026), e questo e' un buco noto che
resta aperto apposta. L'asse `lineette` maschera i blocchi di codice, per non
segnalare i segni meno e le opzioni da riga di comando: quindi non vede una
lineetta vietata scritta dentro un **commento**, che invece e' prosa a tutti
gli effetti (in Transformers ne e' stata trovata una a mano).

Estendere il controllo ai soli commenti sembra ovvio e non lo e'. Misurato su
tutto il libro, togliendo i separatori `# ---` e gli intervalli numerici,
resta **un solo** candidato, ed e' un falso positivo:
`# modellino lineare ... normalizzata z = (x - mu)/sigma`. Nei commenti si
scrivono formule, e in una formula il trattino spaziato e' un segno meno: il
controllo segnalerebbe soprattutto quelle. Il libro e' pulito su questo asse, e
il costo di tenerlo pulito con gli occhi e' minore del rumore che produrrebbe
un controllo che sbaglia piu' spesso di quanto azzecchi.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
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
    # Le due formule con cui una frase dichiara da che parte sta il rimando.
    # Servono all'asse `verso`, e sono volutamente strette: meglio non vedere
    # un difetto che accusarne uno che non c'e', perche' un elenco con dentro
    # del rumore lo si smette di leggere.
    "indietro": re.compile(
        r"\b(abbiamo (gia' |gia |già )?(visto|incontrato|derivato|costruito|"
        r"definito|studiato|introdotto)|come sappiamo|(gia'|gia|già) "
        r"(visto|vista|viste|visti|incontrato|incontrata)|"
        r"(nel capitolo|nella sezione) precedente|poco fa|"
        r"come si (e'|è) visto|che conosciamo)", re.I),
    # NON e' una formula di richiamo all'indietro «del capitolo su X»: nomina
    # un capitolo e basta, e nel libro compare piu' spesso per annunciare
    # («lo raccontano il capitolo sui Transformer e il capitolo su MLOps»)
    # che per ricordare. Messa fra le formule all'indietro produceva due
    # segnalazioni su tre false.
    "poi": re.compile(
        r"\b(vedremo|vedrai|lo vedremo|piu' avanti|più avanti|"
        r"nel prossimo capitolo|nella prossima sezione|torneremo|"
        r"approfondiremo|riprenderemo|lo riprende|ne parliamo)", re.I),
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


_RX_RECINTO_CODICE = re.compile(
    r"^\{?(?:code-block|code|literalinclude|eval-rst)\}?\b"
    r"|^(?:python|py|ipython3?|text|bash|sh|shell|console|json|ya?ml|toml"
    r"|latex|tex|html|css|js|javascript|c|cpp|sql|diff|ini|makefile"
    r"|dockerfile|xml|markdown|md|r|julia|rust|go)\b", re.I)


def _maschera_prosa(testo: str) -> str:
    """Il testo con **il solo codice**, la matematica e gli URL messi a spazi.

    Serve a ogni asse che cerca un segno di prosa, perche' in un libro tecnico
    gli stessi caratteri compaiono legittimamente dappertutto: segni meno nelle
    formule, opzioni da riga di comando, separatori di tabella.

    Tre trappole, e tutte e tre sono costate.

    La prima: un interruttore acceso/spento sui recinti si **desincronizza**
    sulle schede a cinque backtick, quindi ci vuole una pila.

    La seconda: una formula in linea puo' andare a capo, quindi la matematica
    si maschera sul testo intero e non riga per riga, con `$$...$$` prima di
    `$...$`. Si sostituisce con spazi, e non si cancella, per non spostare i
    numeri di riga.

    La terza, la piu' cara, e' rimasta in piedi per parecchio: **non tutte le
    recinzioni contengono codice**. Le schede `{tab}` e i riquadri
    `{admonition}` sono recinzioni, e dentro c'e' quasi tutta la prosa del
    libro. Chi le maschera insieme ai blocchi `python` costruisce un controllo
    che gira su una frazione del testo e torna verde, e nessuno se ne accorge
    perche' verde e' la risposta che ci si aspetta. Qui si guarda **l'etichetta
    del recinto**: si azzera solo dove quell'etichetta dice codice (o dove non
    dice niente, che e' il caso dei blocchi di uscita), e la prosa dentro le
    schede resta visibile. Le righe di apertura e chiusura si azzerano sempre,
    perche' i backtick non sono testo.

    La quarta l'ha portata una riga di uscita vera: **dentro un recinto di
    codice non si apre niente**. Dal 3.11 il traceback di Python sottolinea il
    pezzo che ha ceduto con `~~~~^^^^^^`, e quella riga, dentro un blocco
    ` ```text `, questo regex la prende per una recinzione a tilde: da li' in
    poi la pila e' sfasata di uno e il recinto di codice non si chiude piu'.
    In `Python/basi.md` restavano azzerate 741 righe di prosa su 1092, due
    terzi del capitolo, per ogni asse che passa di qui. Adesso, quando in cima
    alla pila c'e' un recinto di **codice**, l'unica riga che conta e' la sua
    chiusura (stesso carattere, lunghezza non minore, nessuna etichetta): il
    resto e' contenuto, che e' poi la regola di CommonMark.
    """
    fuori, pila = [], []          # pila di (backtick, e_codice)
    for riga in testo.split("\n"):
        m = re.match(r"^(\s*)(`{3,}|~{3,})(.*)$", riga)
        if m and pila and pila[-1][1] and not (
                m.group(2)[0] == pila[-1][0][0]
                and len(m.group(2)) >= len(pila[-1][0])
                and not m.group(3).strip()):
            m = None              # dentro il codice, e' contenuto e basta
        if m:
            tick, info = m.group(2), m.group(3).strip()
            if pila and len(tick) >= len(pila[-1][0]) and not info:
                pila.pop()
            elif info or not pila:
                pila.append((tick, bool(_RX_RECINTO_CODICE.match(info)) or not info))
            else:
                pila.pop()
            fuori.append(" " * len(riga))
            continue
        dentro_codice = any(e_codice for _, e_codice in pila)
        fuori.append(" " * len(riga) if dentro_codice else riga)
    t = "\n".join(fuori)
    for rx in (r"\$\$.*?\$\$", r"\$[^$]*?\$", r"`[^`]*`", r"https?://\S+"):
        t = re.sub(rx, lambda m: re.sub(r"[^\n]", " ", m.group(0)), t, flags=re.S)
    return t


# Lo spazio della **prosa**: orizzontale, oppure un solo a capo. Non due: una
# riga vuota e' un cambio di capoverso, e «non» in fondo a un capoverso con
# «e'» in cima al successivo sono due frasi diverse, non un costrutto.
#
# Serve perche' il libro va a capo a ottanta colonne, quindi la stessa frase
# si scrive «non e' un caso» su una riga e «non e' un\ncaso» sulla riga dopo,
# a seconda di dove cade il margine. Con lo spazio **letterale** un
# rilevatore ne vede una e non l'altra, e dichiara dentro il tetto capitoli
# che non ci sono: il costrutto e' lo stesso, cambia solo dove cade l'a capo.
_SP = r"(?:[^\S\n]+|[^\S\n]*\n[^\S\n]*)"

# `(?![^\S\n]{3,})` esclude i falsi positivi che la **mascheratura stessa
# fabbrica**: azzerando il codice in linea, «non sono Python: `%timeit` e' un
# comando» diventa «non sono Python:        e' un comando», e il costrutto si
# forma dal nulla. Sul grezzo quelle frasi non hanno nessun match. La prosa
# vera fra la punteggiatura e il verbo mette uno spazio solo, o un a capo:
# tre o piu' spazi di fila sono sempre roba mascherata.
_RX_TIC = re.compile(
    rf"\b[Nn]on{_SP}(?:è|sono|era|erano|si{_SP}tratta{_SP}di)\b"
    r"[^.;!?:,]{0,120}?[,:;](?![^\S\n]{3,})\s*(?:è|sono|era|erano)\b",
    re.S)

# `(?!...particolare)` perche' «non e' un **caso particolare** della GCN» e'
# un'affermazione tecnica, non la sorella retorica di «non e' un caso» (= non
# e' una coincidenza). Senza l'esclusione il rilevatore fa «riparare» una
# frase giusta, che e' il modo in cui un controllo costa piu' di quanto rende.
# E anche qui lo spazio e' `_SP`, perche' il difetto morde al contrario:
# l'esclusione saltava appena l'a capo cadeva fra «caso» e «particolare».
_RX_SORELLE = re.compile(
    rf"[Nn]on{_SP}è{_SP}un{_SP}(?:dettaglio|caso|teoria|zelo|cosmesi"
    rf"|pignoleria|capriccio)\b(?!{_SP}particolare)")


def _senza_citazioni(t: str) -> str:
    """Il testo con le citazioni messe a spazi, i numeri di riga fermi.

    Una negazione dentro «virgolette» e' un riferimento a una frase gia'
    coniata, non una mossa fatta adesso: chi la «ripara» rompe una citazione.
    """
    for rx in (r"«[^»]*»", r'"[^"\n]*"', r"“[^”]*”"):
        t = re.sub(rx, lambda m: re.sub(r"[^\n]", " ", m.group(0)), t)
    return t


def contrapposizioni(testo: str) -> list[tuple[int, str]]:
    """[(riga, frase)] dei «non e' X, e' Y» di una pagina, criterio ufficiale.

    Sta qui, e non dentro l'asse, perche' **chi misura deve poter eseguire il
    rilevatore vero** invece di riscriverne una copia: due copie divergono, e
    a quel punto non si sa piu' quale dei due conti abbia ragione.
    """
    pulito = _senza_citazioni(_maschera_prosa(testo))
    trovati = {m.start(): m for m in _RX_TIC.finditer(pulito)}
    trovati.update({m.start(): m for m in _RX_SORELLE.finditer(pulito)})
    return [(pulito[:pos].count("\n") + 1,
             " ".join(trovati[pos].group(0).split()))
            for pos in sorted(trovati)]


@functools.lru_cache(maxsize=1)
def _tempi_git() -> dict[str, int]:
    """{percorso relativo: data dell'ultimo commit che lo tocca}, in una passata.

    Serve perche' **la data di modifica di un file non dice niente in CI**: un
    `git checkout` scrive tutto l'albero in un istante solo, quindi ogni file
    risulta modificato nello stesso momento e ogni confronto fra due date
    diventa un sorteggio. In locale funzionava e sul runner no: l'asse `stampa`
    dichiarava «22 fermi immagine piu' vecchi dell'animazione» su un albero in
    cui nessuno aveva toccato niente.

    Una sola invocazione di `git log` per tutto il repository: farne una per
    file (qui sarebbero un centinaio e passa) costa secondi e da' lo stesso
    risultato. E' la stessa mossa che fa `date_git()` in `genera-radice.py`,
    per la stessa ragione.
    """
    fuori: dict[str, int] = {}
    try:
        uscita = subprocess.run(
            ["git", "-C", str(QUI), "log", "--format=%ct", "--name-only",
             "--no-renames"],
            capture_output=True, text=True, check=True, timeout=180).stdout
    except Exception:
        return fuori          # senza git si ripiega sulle date dei file
    ct = 0
    for riga in uscita.splitlines():
        riga = riga.rstrip()
        if not riga:
            continue
        if riga.isdigit():
            ct = int(riga)
        else:
            fuori.setdefault(riga, ct)   # la prima e' la piu' recente
    return fuori


@functools.lru_cache(maxsize=1)
def _sporchi() -> set[str]:
    """I file che in questo momento divergono da HEAD (o non sono tracciati)."""
    try:
        uscita = subprocess.run(
            ["git", "-C", str(QUI), "status", "--porcelain"],
            capture_output=True, text=True, check=True, timeout=60).stdout
    except Exception:
        return set()
    return {r[3:].strip().strip('"') for r in uscita.splitlines() if len(r) > 3}


def quando(percorso: Path) -> float:
    """Quanto e' «recente» un file, in modo che regga anche in CI.

    Per un file committato e pulito vale la data del commit; per uno modificato
    o non ancora tracciato vale la data di modifica, che e' recentissima e
    quindi lo fa risultare piu' nuovo di qualunque cosa sia committata. Le due
    scale sono confrontabili perche' sono entrambe epoch.
    """
    try:
        rel = str(percorso.resolve().relative_to(QUI))
    except ValueError:
        return percorso.stat().st_mtime
    if rel in _sporchi():
        return percorso.stat().st_mtime
    ct = _tempi_git().get(rel)
    return float(ct) if ct is not None else percorso.stat().st_mtime


@functools.lru_cache(maxsize=1)
def _impronte_fermi() -> dict[str, str]:
    """Il registro che `animazioni/fermi.py` scrive quando genera i fermi."""
    reg = QUI / "animazioni" / "impronte-fermi.json"
    if not reg.is_file():
        return {}
    try:
        return json.loads(reg.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _impronta(file: Path) -> str:
    return hashlib.sha256(file.read_bytes()).hexdigest()[:16]


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
        "schede", "ricordare", "simboli", "avanti", "palette", "clip", "ambiente",
        "lineette", "stampa", "verso", "matematica", "doppioni",
        "contrapposizioni"}

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

            # Accanto a ogni scheda si scarica il capitolo in PDF, e il nome
            # del file lo decide la cartella. Se i due si scollano il lettore
            # clicca su un 404, che dalla pagina non si vede: il collegamento
            # c'e', il file no, e se ne accorge solo chi ci prova.
            for scheda in schede:
                atteso = f"paithon-book-{scheda.split('/')[0]}.pdf"
                if atteso not in testo_landing:
                    problemi["scheda senza il suo capitolo in PDF"].append(atteso)

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

        # E il difetto speculare, che nessuno cercava perche' nel sorgente si
        # legge benissimo: una Elementare e la sua Superiore separate da un
        # capoverso di prosa. Sono due gruppi da UNA linguetta, non una coppia:
        # la Superiore resta visibile a chi ha messo l'interruttore globale su
        # Elementare, e l'interruttore non puo' nasconderla perche' nel suo
        # gruppo non ha un'alternativa. Il sorgente non protesta, la build
        # nemmeno, e si vede solo contando le linguette nell'HTML. Al 20 agosto
        # 2026 ce n'era una sola in 217 file, prodotta da una riscrittura che
        # aveva infilato la prosa in mezzo: cioe' e' un guasto da revisione, e
        # arriva proprio quando qualcuno rimette mano a una pagina gia' buona.
        for f, t in testi.items():
            if not f.endswith(".md"):
                continue
            righe = t.split("\n")
            apertura, chiusura = None, None
            for i, riga in enumerate(righe):
                m = apre.match(riga)
                if m:
                    livello = m.group(2).lower()
                    if livello.startswith("element"):
                        apertura, chiusura = m.group(1), None
                    elif livello.startswith("superior") and chiusura is not None:
                        prosa = [x for x in righe[chiusura + 1:i] if x.strip()]
                        if prosa:
                            problemi["coppia di schede spezzata da prosa"].append(
                                f"{f}:{i + 1}  ->  {len(prosa)} righe fra "
                                f"Elementare e Superiore: {prosa[0][:60]}")
                        apertura = None
                    continue
                if apertura and riga.rstrip() == apertura:
                    chiusura = i

    if "schede" in attivi:
        # La lunghezza di una scheda Elementare non e' un difetto di per se',
        # ed e' per questo che qui si elenca e basta. Ma e' la deriva che la
        # campagna dell'isomorfismo ha prodotto senza che nessuno la vedesse:
        # riparando un buco si aggiunge un capoverso, il capoverso e' giusto, e
        # dopo undici ondate la mediana era passata da 195 a 282 parole e le
        # schede oltre le 300 erano raddoppiate. Oltre una certa lunghezza la
        # scena evapora e resta un elenco di condizioni in parole piane, che e'
        # una Superiore tradotta: il livello Elementare a quel punto non fa piu'
        # il mestiere per cui esiste (il test dell'ombrellone). Il numero non
        # decide, ma dice dove guardare.
        TETTO = 400
        for f, testo in testi.items():
            if not f.endswith(".md"):
                continue
            corpo, dentro, inizio = [], False, 0
            for n, riga in enumerate(testo.split("\n"), 1):
                m = re.match(r"^(`{4,})\{tab\}\s*(\S+)", riga)
                if m:
                    dentro = m.group(2).lower().startswith("element")
                    corpo, inizio = [], n
                    continue
                if re.match(r"^`{4,}\s*$", riga):
                    parole = len(" ".join(corpo).split())
                    if dentro and parole > TETTO:
                        problemi["schede Elementari molto lunghe (da rileggere)"].append(
                            f"{f}:{inizio}  ->  {parole} parole")
                    dentro = False
                    continue
                if dentro:
                    corpo.append(riga)

    if "simboli" in attivi:
        # Una lettera fa un mestiere solo per capitolo, e la medicina e'
        # rinominare, non avvertire. Il difetto e' il secondo per frequenza fra
        # quelli che una campagna di rilettura integrale ha trovato (dieci
        # capitoli su venti tornate), e nasce naturale: due sottocampi usano la
        # stessa lettera per convenzione, il capitolo li mette nella stessa
        # pagina, e nessuno dei due autori sta sbagliando.
        #
        # Non si cercano le occorrenze, che sono migliaia: si cercano le
        # GLOSSE. Il libro introduce un simbolo scrivendo «$x$ e' il ...», e se
        # nello stesso capitolo lo stesso simbolo riceve due glosse diverse,
        # o e' una collisione o e' la stessa cosa detta due volte.
        #
        # **Elenca e basta, e la meta' delle righe non e' un difetto.** Le
        # ventisette del 29 agosto 2026 sono state aperte una per una:
        #
        #   6  artefatti della regex: cinque volte il simbolo non e' il
        #      soggetto ma la coda di un complemento («la media di tutti gli
        #      $\ell$ **e' la** loss», «Cio' che risale da $D$ verso $G$ **e'
        #      il** gradiente»: li' $\ell$ e $G$ fanno un mestiere solo), una
        #      volta e' il pedice tolto ($F_1$ la metrica contro $F$ il simbolo
        #      iniziale di una grammatica);
        #  13  lo stesso mestiere glossato due volte, che e' spreco e non
        #      errore: quattro «$r$ e' la ricompensa...» in DeepRL, tre
        #      «$\sigma$ e' la sigmoide» nei Transformer;
        #   4  l'ambiguita' **e' l'argomento** del paragrafo, e toglierla
        #      toglierebbe la lezione: la $p$ della cross-entropia in
        #      Matematica spiega con quelle due letture perche' la loss di
        #      training scenda a zero e quella di validazione no;
        #   4  riuso vero di una lettera consolidata in sezioni lontane, ognuna
        #      con la sua glossa.
        #
        # Nessuna delle ventisette ha richiesto di rinominare. Prima di
        # ripararne una si legga in `CLAUDE.md` quando una collisione e' un
        # difetto e quando e' una convenzione: la medicina non e' sempre
        # rinominare, e questo asse non lo sa.
        glossa = re.compile(
            r"\$\\?([A-Za-z]|\\[a-zA-Z]+)(?:_\{?\w+\}?)?\$\s*(?:è|e')\s+"
            r"((?:il|la|lo|l'|un|una|uno|i|le|gli)\s+\w+(?:\s+\w+)?)", re.I)
        per_capitolo: dict = defaultdict(lambda: defaultdict(set))
        for f, testo in testi.items():
            if not f.endswith(".md") or "/" not in f:
                continue
            capitolo = f.split("/")[0]
            for m in glossa.finditer(testo):
                per_capitolo[capitolo][m.group(1)].add(m.group(2).lower().strip())
        for capitolo, simboli in sorted(per_capitolo.items()):
            for simbolo, glosse in sorted(simboli.items()):
                if len(glosse) >= 2:
                    problemi["una lettera con due glosse nello stesso capitolo"].append(
                        f"{capitolo}: ${simbolo}$  ->  "
                        + " | ".join(sorted(glosse)[:3]))

    if "schede" in attivi:
        # Una recinzione a backtick con del testo attaccato sulla stessa riga.
        # Se chiude una direttiva, Sphinx legge quel testo come l'APERTURA di
        # un blocco di codice (`\`\`\` La via dei` diventa un blocco in
        # linguaggio «La»), e da li' in poi un capoverso di prosa viene
        # impaginato come codice.
        #
        # Il difetto e' invisibile a tutto: la build esce a zero, la pagina si
        # vede solo aprendola, e il controllo sui notebook non lo trova perche'
        # il notebook e' COERENTE con il sorgente rotto. Al 22 agosto 2026 ce
        # n'era una sola in tutto il libro, e per settimane aveva messo
        # l'etichetta di sezione sbagliata su una cella di `Audio.ipynb`. Si e'
        # vista costruendo il PDF, fra gli avvisi di LaTeX.
        for f, testo in testi.items():
            if not f.endswith(".md"):
                continue
            for n, riga in enumerate(testo.split("\n"), 1):
                m = re.match(r"^(`{3,})\s*(\S.*)$", riga.rstrip())
                if m and not m.group(2).startswith("{") and " " in m.group(2).strip():
                    problemi["recinzione con del testo attaccato"].append(
                        f"{f}:{n}  {riga.strip()[:60]}")

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
        # Zero clip era un avviso e basta, e non e' bastato: elencare i
        # capitoli scoperti senza chiedere niente a nessuno li ha lasciati
        # scoperti. Adesso e' un difetto, con una sola via d'uscita: dichiarare
        # il capitolo in `animazioni/senza-clip.toml` scrivendo perche' li' il
        # tempo non e' il contenuto. La differenza che conta non e' fra un
        # capitolo animato e uno fermo (un capitolo fermo puo' essere la scelta
        # giusta), e' fra essersela chiesta e non essersela chiesta, e dal di
        # fuori le due cose si somigliano. Una riga in un file le separa.
        esenti = {}
        senza = QUI / "animazioni" / "senza-clip.toml"
        if senza.is_file():
            try:
                import tomllib
                esenti = tomllib.loads(senza.read_text(encoding="utf-8"))
            except Exception as e:      # un toml rotto non deve zittire l'asse
                problemi["senza-clip.toml illeggibile"].append(str(e))
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
                if not clip and cartella not in esenti:
                    problemi["capitoli senza figure animate (dichiarali in "
                             "animazioni/senza-clip.toml, con la ragione)"
                             ].append(cartella)
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

        for sorg in sorted((QUI / "animazioni").glob("*.py")):
            gif = LIBRO / "figures" / (sorg.stem + ".gif")
            if not gif.is_file():
                continue
            if not (quando(sorg) > quando(gif)):
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
        # L'immagine e' attrezzatura della macchina come Docker stesso: dove
        # manca (un runner CI appena nato) il controllo non puo' dire niente
        # sull'ambiente, e un "non posso misurare" non e' un rosso. Si tace
        # CON NOTA, mai in silenzio (misurato 2026-08-30: il cancello della
        # pubblicazione andava rosso su ogni runner, sempre).
        ha_immagine = shutil.which("docker") and subprocess.run(
            ["docker", "image", "inspect", "paithon-manim"],
            capture_output=True).returncode == 0
        if collaudo.is_file() and shutil.which("docker") and not ha_immagine:
            print("   (ambiente delle clip: immagine paithon-manim assente su"
                  " questa macchina, controllo non eseguito —"
                  " docker build -t paithon-manim animazioni/)")
        if collaudo.is_file() and ha_immagine:
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
            else:
                # Non si confrontano le date: si confronta l'impronta del
                # contenuto, registrata da fermi.py quando ha scattato i tre
                # fotogrammi. Le date qui non funzionano in nessuna delle due
                # versioni: la data di modifica la azzera il checkout della CI,
                # e la data del commit puo' risultare invertita se i PNG e la
                # loro animazione finiscono in commit diversi (e' successo).
                # L'impronta e' l'unica cosa che dice davvero da quale
                # animazione vengono quei fermi.
                registrata = _impronte_fermi().get(sorgente.name)
                if registrata is None:
                    problemi["fermi immagine senza impronta"].append(
                        f"{sorgente.name}  (python3 animazioni/fermi.py)")
                elif registrata != _impronta(sorgente):
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
        # linea che va a capo non si vede guardando una riga per volta. La
        # mascheratura sta in `_maschera_prosa`, che la fa **una volta sola**.
        doppia = re.compile(r"(?<![-\w])--(?![-\w])|(?<=\w)--(?=\w)")
        singola = re.compile(r"(?<=\w) - (?=\w)")
        for f, t in sorted(testi.items()):
            if not f.endswith(".md"):
                continue
            for n, riga in enumerate(_maschera_prosa(t).split("\n"), 1):
                if "<!--" in riga or "-->" in riga or '="' in riga:
                    continue
                if re.match(r"^\s*\|[\s|:-]+\|\s*$", riga) or re.match(r"^\s*-{3,}\s*$", riga):
                    continue
                pulita = re.sub(r"^\s*[-*+]\s", "", riga)
                if doppia.search(pulita) or singola.search(pulita):
                    problemi["lineette scritte in ASCII (-- oppure - )"].append(
                        f"{f}:{n}  {pulita.strip()[:88]}")

    if "contrapposizioni" in attivi:
        # «non e' X, e' Y». `CLAUDE.md` ne ammette **due per capitolo**, e solo
        # dove la contrapposizione *e'* l'argomento. Fuori di li' e' la cadenza
        # da diapositiva: il paragrafo si presenta prima di parlare.
        #
        # Questo asse nasce da una campagna, e la campagna ha insegnato tre
        # cose che stanno tutte nel codice qui sotto.
        #
        # **La prima: il costrutto ha tre punteggiature, non una.** Un
        # rilevatore che cerca solo la virgola ne trova meno della meta'. Su
        # Machine Learning ha detto «ventuno, adesso due» quando ne restavano
        # diciotto, tutti con i due punti o con il punto e virgola. Ogni volta
        # che il pattern e' stato allargato ne sono usciti altri, tre volte di
        # fila: e' la ragione per cui qui le tre forme sono nella stessa regex
        # invece che in tre passate successive.
        #
        # **La seconda: la negazione dentro una citazione non e' il
        # costrutto.** «una su due non e' a posto»: sono due bocciature
        # diverse» e la frase del capitolo sulle GPU richiamata fra virgolette
        # («il collo di bottiglia non sono i conti, sono i byte») sono
        # riferimenti, non mosse fatte adesso. Chi le «ripara» rompe una
        # citazione, quindi il testo fra «» e fra "" si maschera prima.
        #
        # **La terza: sopra il tetto non si fallisce, si elenca.** Quando
        # l'asse e' nato, trentacinque capitoli su quarantuno erano oltre, e un
        # controllo sempre rosso non avvisa di niente. Falliscono i soli
        # capitoli gia' portati dentro il tetto, elencati in CHIUSI: li' un
        # ritorno sopra i due e' una regressione e va vista subito. Gli altri
        # si contano e basta, e la lista si accorcia man mano.
        # `CHIUSI` era diventato `None`, cioe' «tutti», perche' col rilevatore
        # vecchio ogni capitolo risultava dentro il tetto. Riparato lo spazio
        # il conto vero e' un altro, e tenere `None` renderebbe l'asse rosso
        # per chiunque fino alla fine della campagna: un controllo sempre rosso
        # non avvisa di niente, e' soltanto un cancello che si impara ad
        # aggirare. Vale quindi il meccanismo che questo asse ha sempre avuto,
        # e la lista **cresce** man mano che un capitolo rientra: dentro
        # `CHIUSI` si fallisce, perche' li' tornare sopra il tetto e' una
        # regressione; fuori si elenca e basta.
        CHIUSI = {"Agenti", "AttenzioneLineare", "Audio",
                  "AutoSupervisione", "Conclusioni", "Efficienza", "GAN",
                  "GPU", "GraphNeuralNetwork", "Interpretabilita",
                  "Introduzione", "MachineLearning", "ModelliLatenti",
                  "Python", "ReinforcementLearning", "RetiNeurali",
                  "Ricerca", "SerieTemporali", "SpeechRecognition",
                  "VerosimiglianzaEsatta"}
        TETTO = 2
        # **La quarta: dentro il costrutto lo spazio non e' uno spazio.**
        # Il rilevatore vero sta in `contrapposizioni()`, a livello di
        # modulo, con `_SP` e le due regex: li' lo puo' eseguire anche chi
        # misura, invece di riscriverne una copia che poi diverge.
        per_capitolo = defaultdict(list)
        for f, t in sorted(testi.items()):
            if "/" not in f:
                continue
            capitolo = f.split("/")[0]
            for n, frase in contrapposizioni(t):
                per_capitolo[capitolo].append(f"{f}:{n}  {frase[:78]}")

        for capitolo, righe in sorted(per_capitolo.items()):
            if len(righe) <= TETTO:
                continue
            chiave = ("«non e' X, e' Y» oltre il tetto di due"
                      if CHIUSI is None or capitolo in CHIUSI else
                      "«non e' X, e' Y» oltre il tetto (campagna in corso)")
            problemi[chiave].append(
                f"{capitolo}: {len(righe)} (tetto {TETTO})")
            for r in righe[:4]:
                problemi[chiave].append(f"    {r}")
            if len(righe) > 4:
                problemi[chiave].append(f"    … e altre {len(righe) - 4}")

    if "matematica" in attivi:
        # Dentro `$...$` la tipografia italiana non vale, e nessuno se ne
        # accorge. Una campagna che ha sostituito 1086 apostrofi ASCII con
        # quello tipografico (perche' `smartquotes`, in italiano, rende `'` come
        # `”` a fine parola) e' entrata anche in **36 formule**, dove `'` non e'
        # un apostrofo ma la derivata prima: `s'`, `a'`, `w'` del Reinforcement
        # Learning sono diventati `s’`, `a’`, `w’`.
        #
        # E sono passati sotto tre reti: `coerenza.py` non guardava dentro la
        # matematica (l'asse `lineette` la maschera apposta), la build Sphinx
        # non ha alzato un avviso, e MathJax `’` lo disegna e basta. Il difetto
        # si vede solo aprendo la pagina e riconoscendo che quel segno non e'
        # un primo. In stampa e' peggio, perche' LuaLaTeX gira in nonstopmode.
        #
        # Il filtro che ha fatto il danno ragionava riga per riga, e riga per
        # riga il `$$` di blocco non si vede: sta su piu' righe. Qui la
        # matematica si estrae dal testo intero, con la stessa pila di recinti
        # dell'asse `lineette`, e i numeri di riga si ricavano dagli offset.
        #
        # Fuori dal setaccio resta `\text{...}` (e `\textrm`, `\mbox`), che e'
        # il posto in cui l'italiano dentro una formula ci sta di diritto: li'
        # un apostrofo tipografico e' giusto, non sbagliato.
        SEGNI = {"’": "apostrofo tipografico (in matematica ' e' la derivata)",
                 "‘": "virgoletta singola tipografica",
                 "“": "virgoletta doppia tipografica",
                 "”": "virgoletta doppia tipografica",
                 "«": "virgoletta caporale",
                 "»": "virgoletta caporale",
                 "—": "lineetta lunga",
                 "–": "lineetta breve",
                 "…": "puntini di sospensione"}

        def solo_prosa(testo: str) -> str:
            """Spegne i recinti di codice lasciando intatte le posizioni."""
            fuori, pila = [], []
            for riga in testo.split("\n"):
                m = re.match(r"^(\s*)(`{3,}|~{3,})(.*)$", riga)
                if m:
                    tick, info = m.group(2), m.group(3).strip()
                    dentro_direttiva = info.startswith("{") and not info.startswith("{code")
                    if pila and len(tick) >= len(pila[-1][0]) and not info:
                        pila.pop()
                    elif info or not pila:
                        pila.append((tick, dentro_direttiva))
                    else:
                        pila.pop()
                    fuori.append(" " * len(riga))
                    continue
                codice = any(not direttiva for _, direttiva in pila)
                fuori.append(" " * len(riga) if codice else riga)
            return "\n".join(fuori)

        formule = re.compile(r"\$\$.*?\$\$|\$[^$]+?\$", re.S)
        testuale = re.compile(r"\\(?:text|textrm|textit|textbf|mbox)\{[^{}]*\}")
        for f, t in sorted(testi.items()):
            if not f.endswith(".md"):
                continue
            prosa = solo_prosa(t)
            for m in formule.finditer(prosa):
                corpo = testuale.sub(lambda x: " " * len(x.group(0)), m.group(0))
                for i, ch in enumerate(corpo):
                    if ch in SEGNI:
                        n = prosa.count("\n", 0, m.start() + i) + 1
                        estratto = " ".join(m.group(0).split())[:70]
                        problemi["segni di prosa dentro una formula"].append(
                            f"{f}:{n}  {ch}  {SEGNI[ch]}\n      {estratto}")

    if "doppioni" in attivi:
        # Due finestre di otto parole identiche a meno di novanta parole di
        # distanza, dentro lo stesso capoverso di prosa: e' quasi sempre una
        # frase incollata due volte da una riscrittura. Il codice e le formule
        # restano fuori, che li' ripetersi e' normale.
        for f, t in testi.items():
            if f.endswith("aggiornamenti.md"):
                continue
            dentro = False
            for blocco in re.split(r"\n\s*\n", t):
                testa = blocco.lstrip()
                # le recinzioni a 4+ backtick sono le schede `{tab}`, e il
                # loro contenuto e' prosa: solo quelle a 3 aprono codice
                if testa.startswith("```") and not testa.startswith("````"):
                    dentro = not dentro
                if dentro or testa.startswith((":", "|", "#", "$$", "```",
                                               "-", "*", ">", "%")):
                    continue
                parole = " ".join(blocco.split()).split()
                visti, gia = {}, False
                for i in range(len(parole) - 7):
                    chiave = " ".join(parole[i:i + 8]).lower()
                    # una finestra con dentro codice o formule non e' prosa:
                    # li' ripetersi e' normale e non dice niente
                    if any(c in chiave for c in "`=#\\$"):
                        continue
                    if chiave in visti and i - visti[chiave] < 90 and not gia:
                        problemi["frasi scritte due volte"].append(
                            f"{f}  ->  «{chiave}…»")
                        gia = True
                    visti[chiave] = i

    if "verso" in attivi:
        # Il difetto piu' insidioso di questo libro non e' un fatto sbagliato:
        # e' una frase giusta messa nel verso sbagliato. «Come abbiamo visto»
        # seguito da un rimando a un capitolo che il lettore non ha ancora
        # letto, e che gli chiede un ricordo che non puo' avere. In Visione e
        # linguaggio (capitolo 17) ce n'erano quattro, verso i capitoli 21 e
        # 24; in Audio sei, in Speech due. Nessuna build se ne accorge, perche'
        # il link risolve benissimo: e' l'ordine di lettura a non tornare, e
        # l'ordine di lettura sta nel _toc.yml.
        toc = LIBRO / "_toc.yml"
        if toc.is_file():
            capitoli = [f for f in re.findall(
                r"^  - file:\s*(\S+)", toc.read_text(encoding="utf-8"), re.M)
                if "/" in f]
            pos: dict[str, int] = {}
            for i, f in enumerate(capitoli):
                pos.setdefault(f.split("/")[0], i)
            # in quale capitolo vive ciascun :name: (i target sono globali,
            # quindi il nome da solo non dice dove sta)
            cap_di_nome = {n: f.split("/")[0]
                           for f, t in testi.items()
                           for n in RX["name"].findall(t)}
            for f, t in testi.items():
                mio = pos.get(f.split("/")[0])
                if mio is None:
                    continue
                for chiave in ("numref", "ref", "doc"):
                    for m in RX[chiave].finditer(t):
                        bersaglio = m.group(1).strip()
                        if chiave == "doc":
                            parti = [p for p in bersaglio.split("/")
                                     if p not in ("", ".", "..")]
                            cap = parti[0] if len(parti) > 1 else None
                        else:
                            cap = cap_di_nome.get(bersaglio)
                        suo = pos.get(cap)
                        if suo is None or suo == mio:
                            continue
                        # solo la frase che PRECEDE il rimando: una frase che
                        # dice «vedremo» dopo il link parla d'altro. Si taglia
                        # sul punto, non sui due punti e sul punto e virgola:
                        # in italiano non chiudono la frase, e la formula del
                        # richiamo sta quasi sempre prima di loro («come
                        # abbiamo visto, vale anche qui: {numref}...»).
                        prima = t[max(0, m.start() - 220):m.start()]
                        prima = re.split(r"(?<=[.!?])\s|\n\n", prima)[-1]
                        coda = prima.strip()[-90:]
                        if suo > mio and RX["indietro"].search(prima):
                            problemi["rimandi all'indietro che puntano in avanti"].append(
                                f"{f}  ->  {cap}  …{coda}")
                        elif suo < mio and RX["poi"].search(prima):
                            problemi["rimandi in avanti che puntano indietro"].append(
                                f"{f}  ->  {cap}  …{coda}")

            # I richiami in PROSA, che sono la forma vera: il libro scrive
            # «come abbiamo visto nel capitolo sulle GAN», non un {doc}.
            # Misurato prima di scrivere questo pezzo: 409 formule di richiamo
            # all'indietro in tutto il libro, e zero con un link accanto. Un
            # asse che guardasse solo i link guarderebbe l'unica forma che il
            # libro non usa.
            #
            # Il nome del capitolo da solo non basta come indizio («PyTorch»
            # compare ovunque come parola): dev'essere preceduto da un
            # connettivo che dichiara il rimando (capitolo, sezione, parlando
            # di). Senza quel vincolo l'elenco si riempie di termini tecnici e
            # smette di essere letto.
            righe = toc.read_text(encoding="utf-8").splitlines()
            titoli: dict[str, str] = {}
            for i, r in enumerate(righe):
                mfile = re.match(r"^  - file:\s*(\S+)", r)
                if not mfile or "/" not in mfile.group(1):
                    continue
                cap = mfile.group(1).split("/")[0]
                for succ in righe[i + 1:i + 3]:
                    mtit = re.match(r"^    title:\s*(.+?)\s*$", succ)
                    if mtit:
                        titoli.setdefault(mtit.group(1).strip().lower(), cap)
                        break
            if titoli:
                rx_titoli = re.compile("|".join(
                    re.escape(a) for a in sorted(titoli, key=len, reverse=True)), re.I)
                rx_connettivo = re.compile(r"capitolo|sezione|parlando|parte", re.I)
                for f, t in testi.items():
                    mio = pos.get(f.split("/")[0])
                    if mio is None:
                        continue
                    for m in RX["indietro"].finditer(t):
                        # la finestra si ferma al primo punto: oltre c'e' una
                        # frase diversa, e una frase diversa puo' legittimamente
                        # annunciare un capitolo che viene dopo
                        finestra = re.split(
                            r"(?<=[.!?])\s", t[m.end():m.end() + 200])[0]
                        for mt in rx_titoli.finditer(finestra):
                            cap = titoli[mt.group(0).lower()]
                            suo = pos.get(cap)
                            if suo is None or suo <= mio:
                                continue
                            if not rx_connettivo.search(
                                    finestra[max(0, mt.start() - 45):mt.start()]):
                                continue
                            frase = " ".join(
                                t[m.start():m.end() + mt.end()].split())
                            problemi["rimandi all'indietro che puntano in avanti"].append(
                                f"{f}  ->  {cap}  …{frase[:120]}")
                            break

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
              "scheda senza il suo capitolo in PDF",
              "schede contigue, che la build fonde",
              "coppia di schede spezzata da prosa",
              "recinzione con del testo attaccato",
              "schede Elementari molto lunghe (da rileggere)",
              "capitoli oltre il tetto di 10 clip",
              "clip fuori dalla tabella del README",
              "capitoli senza figure animate (dichiarali in "
              "animazioni/senza-clip.toml, con la ragione)",
              "lineette scritte in ASCII (-- oppure - )",
              "«non e' X, e' Y» oltre il tetto di due",
              "«non e' X, e' Y» oltre il tetto (campagna in corso)",
              "frasi scritte due volte",
              "«Da ricordare» fuori dalle schede",
              "una lettera con due glosse nello stesso capitolo",
              "clip piu' vecchie del loro sorgente",
              "ambiente delle clip cambiato",
              "lineette dentro le figure",
              "script dentro le figure",
              "colori fuori palette (decisione del repo brand)",
              "fermi immagine mancanti",
              "fermi immagine senza impronta",
              "fermi immagine piu' vecchi dell'animazione",
              "fermi immagine orfani",
              "figure che non dichiarano i font del brand",
              "segni di prosa dentro una formula",
              "rimandi all'indietro che puntano in avanti",
              "rimandi in avanti che puntano indietro",
              "rimandi in avanti (da leggere)"]
    # Assi che elencano e basta: dicono cosa guardare, non cosa e' rotto, e
    # quindi non fanno fallire niente.
    solo_elenco = {"schede Elementari molto lunghe (da rileggere)",
                   "una lettera con due glosse nello stesso capitolo",
                   "rimandi in avanti (da leggere)",
                   "colori fuori palette (decisione del repo brand)",
                   "«non e' X, e' Y» oltre il tetto (campagna in corso)",
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
          f" ({len(problemi.get('rimandi in avanti (da leggere)', []))}"
          f" rimandi da leggere a mano)")
    return 1 if totale else 0


if __name__ == "__main__":
    sys.exit(main())
