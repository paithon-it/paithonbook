#!/usr/bin/env python3
"""
Cerca incoerenze nel libro: riferimenti che non portano da nessuna parte.

Le incoerenze di contenuto (un concetto usato prima di averlo spiegato, un
rimando in prosa a una sezione che non c'e', un'affermazione che un altro
capitolo smentisce) si trovano leggendo. Queste si trovano contando, e vale la
pena toglierle di mezzo prima di rileggere:

  numref     riferimenti a figure/tabelle senza un :name: corrispondente
  cite       chiavi bibliografiche assenti dal .bib
  ref/doc    riferimenti interni a target inesistenti
  figure     file in book/figures/ che nessuno richiama
  toc        file sotto book/ non elencati in _toc.yml
  codifica   pagine del toc con gli a capo CRLF o con un BOM in testa: non si
             vedono in nessun editor, e una patch fatta su testo a capo LF su
             quelle righe non si applica
  esterni    script e fogli di stile presi da un altro dominio, in una pagina
             o in _config.yml: un terzo che vede ogni lettore e cambia cio'
             che gira sulla pagina (un esempio in un blocco di codice no)
  landing   schede di intro.md che non corrispondono ai capitoli del _toc.yml,
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
             le rifiuta gia' `scrivi()` in paithon_svg.py). La palette e'
             quella del brand, `figure/palette.json` nel submodule: senza il
             submodule l'asse si ferma e lo dice
  ambiente   scene Manim che l'ambiente di oggi non saprebbe piu' rendere
  lineette   i due travestimenti della lineetta che `CONTRIBUTING.md` vieta:
             la doppia `--` e il trattino singolo spaziato
  stampa     cio' che serve al PDF e puo' restare indietro in silenzio: i tre
             fermi immagine di ogni animazione (mancanti, piu' vecchi
             dell'animazione, orfani) e le figure che dichiarano una famiglia
             di font generica invece di quelle del brand
  avanti     rimandi in avanti ("vedremo", "prossima sezione"), solo elenco,
             la verifica resta umana
  matematica segni di prosa finiti dentro `$...$`: l'apostrofo tipografico
             al posto della derivata prima, virgolette, lineette. Nessuno se ne
             accorge (Sphinx non avvisa, MathJax li disegna), e una
             sostituzione tipografica fatta su tutto il testo ce li porta
             tutti insieme
  doppioni   la stessa frase scritta due volte nello stesso capoverso. Nasce
             dalle riscritture: si sostituisce un pezzo di testo con uno nuovo
             che riprende quello che diceva la frase dopo, e il capoverso
             finisce per dirla due volte. La build non se ne accorge e il
             testo resta valido
  frasi      le forme fisse di «La frase» in CONTRIBUTING.md (il libro che
             parla di se', le formule d'annuncio, i calchi, il parlato
             nella Superiore), solo elenco: quasi ogni forma ha anche un
             uso giusto, e a decidere e' chi legge
  enfasi     il neretto per mille parole di prosa, capitolo per capitolo,
             contro il tetto di otto. Non giudica QUALE marcatura sia
             giusta (i due assi in fondo a questa nota ci hanno provato e
             sono affogati nel rumore): conta, e la soglia e' decisa altrove.
             `aggiornamenti.md` sta fuori, ed e' scritto nell'uscita: la
             genera un altro script e il suo neretto sono i titoli delle voci
             del registro. In piu', come elenco e non come errore, i termini
             marcati in piu' di un capitolo, che una rilettura a un capitolo
             per volta non puo' vedere
  verso      rimandi che vanno dalla parte sbagliata: «come abbiamo visto»
             seguito da un capitolo che il lettore non ha ancora letto (e il
             contrario, «vedremo» verso un capitolo gia' letto). L'ordine di
             lettura sta nel _toc.yml, il link risolve benissimo e nessuna
             build se ne accorge
  postille   le postille dei capoversi (`pagina.postille.yml`, accanto alla
             pagina): ferma un file illeggibile o senza la sua pagina, una
             postilla oltre gli 80 caratteri, con marcature o lineette,
             doppia, o su un capoverso sotto le 60 parole; elenca soltanto le
             postille spente (il capoverso e' cambiato, o e' finito in una
             scheda), le ambigue, le forme fisse e il conto per capitolo. Le
             spente non fermano niente: una correzione del testo spegne
             la postilla del capoverso che tocca, e non va bloccata

  scripts/coerenza.py            # tutto
  scripts/coerenza.py --solo numref,cite

Due assi PROVATI E SCARTATI, per non rifarli. Il primo cerca i termini usati
prima di essere definiti, ricostruendo l'ordine di lettura dal _toc.yml e
prendendo il grassetto come marca di introduzione (e' la convenzione del
libro). Il secondo sembra aggirare il problema e non lo aggira: cerca i
termini in grassetto solo dentro una scheda Superiore e usati anche nel testo
comune, dove il lettore Elementare li incontra senza averli mai visti.

Tutti e due affogano nel rumore per la stessa ragione: nel libro il grassetto
non marca solo l'introduzione di un termine, marca anche l'enfasi. Fra i
risultati escono parole comuni che capita siano in grassetto («a sinistra»,
«stessa», «10 minuti») e la front matter, che anticipa tutto il libro per
mestiere, e il segnale resta sotto il rumore. I termini tecnici non sono un
insieme chiuso e riconoscibile come i costrutti di un linguaggio, e il
grassetto non basta a riconoscerli: quel difetto si vede leggendo, non
contando. Il conteggio che invece funziona, e che sta qui sotto, e' quello
dei riquadri «Da ricordare» fuori dalle schede, perche' li' la marca e' una
direttiva MyST e non una convenzione tipografica.

E un buco noto, che resta aperto apposta. L'asse `lineette` maschera i
blocchi di codice, per non segnalare i segni meno e le opzioni da riga di
comando: quindi non vede una lineetta vietata scritta dentro un **commento**,
che invece e' prosa a tutti gli effetti. Estendere il controllo ai soli
commenti sembra ovvio e non lo e': nei commenti si scrivono formule
(`z = (x - mu)/sigma`), e in una formula il trattino spaziato e' un segno
meno, quindi il controllo segnalerebbe soprattutto quelle. Tenerli puliti con
gli occhi costa meno del rumore di un controllo che sbaglia piu' spesso di
quanto azzecchi.
"""

from __future__ import annotations

import argparse
import codecs
import functools
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
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
    # Ancorata all'inizio della frase: `(?<![^.\n])` vuol dire dopo un punto,
    # dopo un a capo o in testa al testo. Senza l'ancora `finditer` la
    # riproverebbe da ogni posizione, e ogni volta `[^.\n]*` scorrerebbe la
    # riga fino in fondo prima di cercare la parola: il costo crescerebbe con
    # il quadrato della lunghezza della riga. Le frasi trovate sono le stesse,
    # posizione per posizione.
    "avanti": re.compile(
        r"(?<![^.\n])[^.\n]*"
        r"\b(vedremo|vedrai|prossima sezione|prossimo capitolo|"
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
    # che per ricordare. Messa fra le formule all'indietro, la maggior parte
    # delle sue segnalazioni sarebbe falsa.
    "poi": re.compile(
        r"\b(vedremo|vedrai|lo vedremo|piu' avanti|più avanti|"
        r"nel prossimo capitolo|nella prossima sezione|torneremo|"
        r"approfondiremo|riprenderemo|lo riprende|ne parliamo)", re.I),
}


# L'uscita strutturata (`--json FILE`), per chi mostra le voci accanto al
# testo invece di stamparle: una voce per occorrenza, con il file e la
# posizione quando l'asse la conosce, e una misura per capitolo per gli assi
# che contano contro un tetto. La stampa per chi legge resta identica, riga
# per riga: le voci si aggiungono accanto, non al posto.
VOCI: list[dict] = []


def posizione(testo: str, inizio: int) -> tuple[int, int]:
    """(riga, colonna), da 1, di un indice del testo.

    La colonna conta i caratteri, non i byte: e' la misura di `len()`, e
    «perché» vale sei, non sette.
    """
    riga = testo.count("\n", 0, inizio) + 1
    return riga, inizio - (testo.rfind("\n", 0, inizio) + 1) + 1


def voce(asse: str, categoria: str, file: str | None = None, *,
         testo: str | None = None, inizio: int | None = None,
         fine: int | None = None, riga: int | None = None,
         colonna: int | None = None, fine_colonna: int | None = None,
         identita: str = "", messaggio: str = "",
         capitolo: str | None = None, **altro) -> None:
    """Una voce dell'uscita strutturata.

    La posizione si da' con il testo e gli indici (la fine esclusa), oppure
    con riga e colonna gia' contate. `identita` e' cio' che resta uguale
    quando la stessa voce si sposta di riga: il bersaglio di un rimando, la
    chiave di una citazione, la frase. Chi confronta due giri la usa per dire
    se un difetto e' nuovo o c'era gia'.
    """
    fine_riga = None
    if testo is not None and inizio is not None:
        riga, colonna = posizione(testo, inizio)
        if fine is not None:
            fine_riga, fine_colonna = posizione(testo, fine)
    elif riga is not None and fine_colonna is not None:
        fine_riga = riga
    if capitolo is None and file and "/" in file:
        capitolo = file.split("/")[0]
    VOCI.append({"tipo": "voce", "asse": asse, "categoria": categoria,
                 "file": file, "capitolo": capitolo, "riga": riga,
                 "colonna": colonna, "fine_riga": fine_riga,
                 "fine_colonna": fine_colonna, "identita": identita,
                 "messaggio": messaggio or categoria, **altro})


def misura(asse: str, categoria: str, capitolo: str, conteggio: float,
           tetto: float, oltre: bool, occorrenze: list | None = None) -> None:
    """Il conto di un capitolo contro il suo tetto, sopra o sotto che sia:
    chi confronta due giri deve sapere anche quanto valeva prima."""
    VOCI.append({"tipo": "misura", "asse": asse, "categoria": categoria,
                 "capitolo": capitolo, "conteggio": conteggio,
                 "tetto": tetto, "oltre": oltre,
                 "occorrenze": occorrenze or []})


def sorgenti() -> dict[str, str]:
    # _static/ contiene il submodule `brand` con i suoi README: non e' testo del
    # libro e non va nel _toc.yml.
    #
    # _build/ e' l'uscita di Sphinx, e va esclusa perche' contiene una COPIA di
    # ogni sorgente sotto `_sources/`: una cartella di build dimenticata in giro
    # fa contare ogni file due volte e riempie l'elenco di problemi fantasma.
    # Chi ci casca smette di fidarsi del controllo, che e' il danno peggiore
    # che un controllo possa fare.
    return {str(p.relative_to(LIBRO)): p.read_text(encoding="utf-8", errors="ignore")
            for p in sorted(list(LIBRO.rglob("*.md")) + list(LIBRO.rglob("*.ipynb")))
            if "_static" not in p.parts and "_build" not in p.parts}


def schede_non_chiuse(testo: str) -> list[int]:
    """Le righe in cui una `{tab}` si apre dentro un'altra ancora aperta.

    Succede quando una riscrittura toglie la chiusura di una scheda: la
    Superiore finisce annidata dentro l'Elementare, e la chiusura che segue
    chiude la scheda esterna. Nel sorgente si legge quasi bene, la build non
    protesta, e nessun altro controllo lo vede: le due schede contano come
    una sola linguetta, quindi l'asse delle schede contigue resta muto. Una
    `{tab}` dentro un'altra con lo STESSO numero di backtick non e' mai
    voluta: per annidare si allunga la recinzione esterna.
    """
    apre = re.compile(r"^(`{4,})\{tab\}")
    righe, fuori = testo.split("\n"), []
    i = 0
    while i < len(righe):
        m = apre.match(righe[i])
        if not m:
            i += 1
            continue
        tick = m.group(1)
        j = i + 1
        while j < len(righe) and righe[j].rstrip() != tick:
            dentro = apre.match(righe[j])
            if dentro and dentro.group(1) == tick:
                fuori.append(j + 1)
            j += 1
        i = j + 1
    return fuori


def pagine_del_toc() -> list[Path]:
    """Le pagine del libro, in ordine di lettura, prese dal `_toc.yml`.

    Non da un glob: `book/**/*.md` non e' il libro (ci sono anche i `.ipynb`,
    e sotto `book/` stanno anche file che il toc non elenca), e una pagina
    saltata non protesta.

    **E il toc non e' fatto di sole voci `file:`**: la prima pagina e' la
    `root:`. Un perimetro che cerca solo `file:` la perde, torna un conto piu'
    basso di uno e non solleva niente: e' un numero appena sbagliato, cioe'
    quello che si crede da se'. La pagina di apertura e' anche la prima che il
    lettore incontra, quindi e' la peggiore da non guardare.

    Sta qui in una copia sola, e non una per attrezzo, perche' due perimetri
    divergono e allora non si sa piu' quale dei due conti abbia ragione: la
    stessa funzione la usa `scripts/genera-figure-scure.py`.
    """
    nomi = re.findall(r"^\s*-?\s*(?:file|root):\s*(\S+)",
                      (LIBRO / "_toc.yml").read_text(encoding="utf-8"), re.M)
    assert nomi, "il _toc.yml non elenca nessuna pagina"
    fuori = []
    for n in nomi:
        for estensione in ("", ".md", ".ipynb"):
            p = LIBRO / (n + estensione)
            if p.is_file():
                fuori.append(p)
                break
    return list(dict.fromkeys(fuori))


_RX_RECINTO_CODICE = re.compile(
    r"^\{?(?:code-block|code|literalinclude|eval-rst)\}?\b"
    r"|^(?:python|py|ipython3?|text|bash|sh|shell|console|json|ya?ml|toml"
    r"|latex|tex|html|css|js|javascript|c|cpp|sql|diff|ini|makefile"
    r"|dockerfile|xml|markdown|md|r|julia|rust|go)\b", re.I)


_RX_RIGA_RECINTO = re.compile(r"^(\s*)(`{3,}|~{3,})(.*)$")


# Ricordata per testo e riempitivo: sei assi la chiedono per ogni pagina, e
# la risposta non cambia.
@functools.lru_cache(maxsize=None)
def _maschera_prosa(testo: str, riempitivo: str = " ") -> str:
    """Il testo con **il solo codice**, la matematica e gli URL messi a spazi.

    Serve a ogni asse che cerca un segno di prosa, perche' in un libro tecnico
    gli stessi caratteri compaiono legittimamente dappertutto: segni meno nelle
    formule, opzioni da riga di comando, separatori di tabella.

    Quattro trappole.

    La prima: un interruttore acceso/spento sui recinti si **desincronizza**
    sulle schede a cinque backtick, quindi ci vuole una pila.

    La seconda: una formula in linea puo' andare a capo, quindi la matematica
    si maschera sul testo intero e non riga per riga, con `$$...$$` prima di
    `$...$`. Si sostituisce con spazi, e non si cancella, per non spostare i
    numeri di riga.

    La terza, la piu' insidiosa: **non tutte le recinzioni contengono
    codice**. Le schede `{tab}` e i riquadri `{admonition}` sono recinzioni,
    e dentro c'e' quasi tutta la prosa del libro. Chi le maschera insieme ai
    blocchi `python` costruisce un controllo che gira su una frazione del
    testo e torna verde, e nessuno se ne accorge perche' verde e' la
    risposta che ci si aspetta. Qui si guarda **l'etichetta del recinto**:
    si azzera solo dove quell'etichetta dice codice (o dove non dice niente,
    che e' il caso dei blocchi di uscita), e la prosa dentro le schede resta
    visibile. Le righe di apertura e chiusura si azzerano sempre, perche' i
    backtick non sono testo.

    Il `riempitivo` esiste per chi cerca una **marcatura**, e non un segno di
    prosa: `**indipendentemente da $x$ e $y$**`, mascherato con gli spazi,
    finisce con uno spazio davanti al `**` di chiusura, e un pattern che
    pretende un carattere pieno li' non chiude piu' la marcatura: il match si
    allunga fino a quella dopo e **due ne diventano una**. Il conto cala, e un
    conto che cala assomiglia al lavoro che procede. Chi passa un riempitivo
    pieno lo vuole di un carattere che non sia ne' spazio ne' `*`, e le
    posizioni restano ferme in tutti e due i casi.

    La quarta: **dentro un recinto di codice non si apre niente**. Dal 3.11
    il traceback di Python sottolinea il pezzo che ha ceduto con
    `~~~~^^^^^^`, e quella riga, dentro un blocco ` ```text `, questo regex
    la prende per una recinzione a tilde: da li' in poi la pila e' sfasata
    di uno, il recinto di codice non si chiude piu', e la prosa che segue
    resta azzerata per ogni asse che passa di qui. Quando in cima alla pila
    c'e' un recinto di **codice**, l'unica riga che conta e' la sua chiusura
    (stesso carattere, lunghezza non minore, nessuna etichetta): il resto e'
    contenuto, che e' poi la regola di CommonMark.
    """
    fuori, pila = [], []          # pila di (backtick, e_codice)
    for riga in testo.split("\n"):
        m = _RX_RIGA_RECINTO.match(riga)
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
            fuori.append(riempitivo * len(riga))
            continue
        dentro_codice = any(e_codice for _, e_codice in pila)
        fuori.append(riempitivo * len(riga) if dentro_codice else riga)
    t = "\n".join(fuori)
    # Il codice in linea prima della matematica: un `$` fra due backtick (il
    # metacarattere di fine riga di una regex, una variabile di shell) e' testo
    # di codice, ma se la matematica passa per prima lo appaia con il dollaro
    # successivo, e tutta la prosa fra i due sparisce dagli assi.
    for rx in (r"`[^`]*`", r"\$\$.*?\$\$", r"\$[^$]*?\$", r"https?://\S+"):
        t = re.sub(rx, lambda m: re.sub(r"[^\n]", riempitivo, m.group(0)),
                   t, flags=re.S)
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
# E anche qui lo spazio e' `_SP`, perche' il difetto morde al contrario: con
# uno spazio letterale l'esclusione salta appena l'a capo cade fra «caso» e
# «particolare».
_RX_SORELLE = re.compile(
    rf"[Nn]on{_SP}è{_SP}un{_SP}(?:dettaglio|caso|teoria|zelo|cosmesi"
    rf"|pignoleria|capriccio)\b(?!{_SP}particolare)")


def _senza_citazioni(t: str, riempitivo: str = " ") -> str:
    """Il testo con le citazioni messe a spazi, i numeri di riga fermi.

    Una negazione dentro «virgolette» e' un riferimento a una frase gia'
    coniata, non una mossa fatta adesso: chi la «ripara» rompe una citazione.
    Il `riempitivo` pieno serve a chi conta le parole, per la ragione scritta
    in `frasi_doppie`.
    """
    for rx in (r"«[^»]*»", r'"[^"\n]*"', r"“[^”]*”"):
        t = re.sub(rx, lambda m: re.sub(r"[^\n]", riempitivo, m.group(0)), t)
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


# Le forme fisse della sezione «La frase» di `CONTRIBUTING.md`: il libro che
# parla di se', le formule d'annuncio, i calchi dall'inglese, e il parlato che
# nella scheda Superiore stona. Sono un elenco da leggere e non un errore,
# perche' quasi ogni forma ha anche un uso giusto («cade fuori dall'intervallo»
# e' italiano, «cade fuori dal conto» e' un calco): l'asse dice dove guardare.
#
# Lo spazio fra le parole e' di nuovo quello della prosa, che puo' andare a
# capo, ma qui al piu' **due** spazi di fila: la maschera mette a spazi il
# codice in linea, e «da qui in poi `x` la chiameremo» non deve diventare
# «da qui in poi la chiameremo».
_SP2 = r"(?:[^\S\n]{1,2}|[^\S\n]{0,2}\n[^\S\n]*)"


def _forme(*voci: str) -> re.Pattern:
    return re.compile("|".join(f"(?:{v.replace(' ', _SP2)})" for v in voci),
                      re.I)


# (famiglia, regex, solo nella scheda Superiore, pagine esenti). Prefazione,
# pagina di apertura e Conclusioni possono parlare del libro (`CONTRIBUTING.md`).
_ESENTI_DA_SE = {"prefazione.md", "intro.md", "Conclusioni/overview.md"}
_FAMIGLIE = [
    ("parla di se'", _forme(
        r"\bfino a qui il capitolo", r"\bfin qui il capitolo",
        r"\b(?:le|delle|nelle|dalle|alle) sezioni precedenti",
        r"\b(?:nel|il|tutto il) resto del capitolo",
        r"\bpiù avanti nel (?:capitolo|libro)", r"\balla fine del capitolo",
        r"\b(?:introdott|vist|definit)[oaie] più su\b",
        r"\bdi cui si (?:parlava|diceva) (?:sopra|più su)",
        r"\bda qui (?:in poi|in avanti),? (?:la|lo|li|le) chiameremo",
        r"\b(?:la|questa) parola tornerà", r"\btornerà più volte nel capitolo",
        r"\bil capitolo (?:ripeterà|chiama sempre)",
        r"\bcome abbiamo visto nel capitolo",
        r"\b(?:in|nell['’]|dall['’]|all['’])apertura del capitolo",
        r"\ball['’]inizio del capitolo"), False, _ESENTI_DA_SE),
    ("annuncio", _forme(
        r"\bconviene (?:dirl[oa]|vederl[oa]|notare|elencarl[ei]"
        r"|distinguerl[ei]|distinguere subito|dire subito)\b",
        r"\bvale la pena\b", r"\bva detta la cosa\b", r"\bva detto che\b",
        r"\buna premessa, che\b", r"\buna parola sui simboli\b",
        r"\bdue cose vanno dette\b", r"\b(?:il motivo|ed) è istruttivo\b",
        r"\bsembra pedante e non lo è\b",
        r"\bfermiamoci un attimo\b", r"\bdose di onestà\b",
        r"\bil punto è che\b"), False, set()),
    ("calco", _forme(
        r"\bcad(?:e|ono|ere|rebbe|uta|uto) fuori"
        r" (?:da sol[oa]|dal conto|dai conti|dalla formula)",
        r"\bnon compra\b", r"\bcompra (?:fedeltà|l['’]ancoraggio)",
        r"\bcifra titolare\b", r"\bregredirl[aoie]\b",
        r"\buccider\w* (?:la|il|un|una|le|i) (?:richiest|process|job)\w*",
        r"\briporta(?:no)? un (?:classificatore|modello)\b",
        r"\bsi legge \d", r"\bda portarsi via\b", r"\bonestamente aperta\b",
        r"\bpinnat[eoai]\b", r"\bminor version\b", r"\bdiffabil[ei]\b",
        r"\bconvergit[aoie]\b"), False, set()),
    ("parlato nella Superiore", _forme(
        r"\bper dire\s*[,)]", r"\bsenza storie\b", r"\bbasta e avanza\b",
        r"\btirar(?:e|la|lo|li|le) su\b", r"\bpari pari\b",
        r"\bparente stretto\b", r"\bsi tocca con mano\b",
        r"\bci sta anche (?:lei|lui)\b", r"\bnon ci sta(?:nno)?\b",
        r"\b(?:dare|dato|data|dati|date|diamo|dà) in pasto\b",
        r"\btraslare in su\b", r"\bun bel niente\b"), True, set()),
]


def _righe_superiore(testo: str) -> set[int]:
    """I numeri di riga (da 1) che stanno dentro una scheda `{tab} Superiore`.

    La scheda si chiude con una recinzione dello stesso numero di backtick e
    senza etichetta; i riquadri e i blocchi di codice che ci stanno dentro ne
    usano di meno, quindi non la chiudono.
    """
    dentro, aperta = set(), None
    for n, riga in enumerate(testo.split("\n"), 1):
        if aperta is None:
            m = re.match(r"^(`{4,})\{tab\}\s+Superiore\b", riga)
            if m:
                aperta = m.group(1)
            continue
        if riga.strip() == aperta:
            aperta = None
            continue
        dentro.add(n)
    return dentro


def frasi_da_rivedere(testo: str, pagina: str = "") -> list[tuple[int, str, str]]:
    """[(riga, famiglia, forma)] delle forme fisse di «La frase» in una pagina.

    Come `contrapposizioni()`, sta a livello di modulo perche' chi misura possa
    eseguire il rilevatore vero invece di riscriverne una copia.
    """
    pulito = _senza_citazioni(_maschera_prosa(testo))
    # Il titolo di un riquadro sta sulla riga di apertura del recinto, che la
    # maschera comune azzera (i backtick non sono testo), e con lei il titolo:
    # «Il gesto da portarsi via» resterebbe invisibile. Si rimette qui, alla
    # stessa posizione, senza toccare la maschera che usano gli altri assi.
    righe = pulito.split("\n")
    for i, riga in enumerate(testo.split("\n")):
        m = re.match(r"^(\s*`{3,}\{admonition\}\s+)(.*)$", riga)
        if m:
            righe[i] = " " * len(m.group(1)) + m.group(2)
    pulito = _senza_citazioni("\n".join(righe))
    superiore = _righe_superiore(testo)
    fuori = []
    for famiglia, rx, solo_superiore, esenti in _FAMIGLIE:
        if pagina in esenti:
            continue
        for m in rx.finditer(pulito):
            riga = pulito[:m.start()].count("\n") + 1
            if solo_superiore and riga not in superiore:
                continue
            fuori.append((riga, famiglia, " ".join(m.group(0).split())))
    return sorted(fuori)


# I capoversi che l'asse `doppioni` non legge, per scelta: elenchi puntati,
# tabelle, titoli, righe di opzione (`:name:`), citazioni e commenti. Il
# corsivo e il neretto in testa a un capoverso **non** ne fanno parte: `*` e
# `-` contano solo come segno d'elenco, cioe' seguiti da uno spazio.
_RX_NON_PROSA = re.compile(r"(?:[-*+]\s|[|#:>%])")


def frasi_doppie(testo: str) -> list[tuple[int, str]]:
    """[(riga, otto parole)] delle frasi scritte due volte in un capoverso.

    Due finestre di otto parole identiche a meno di novanta parole di
    distanza, dentro lo stesso capoverso di prosa: e' quasi sempre una frase
    incollata due volte da una riscrittura. Una voce per capoverso.

    Il capoverso si cerca sul testo **mascherato**, e non con un interruttore
    acceso/spento girato dai capoversi che cominciano con tre backtick. La
    chiusura di un recinto sta in coda allo stesso capoverso dell'apertura,
    quindi l'interruttore non torna indietro: dopo il primo blocco di codice
    la prosa non si legge piu' fino al recinto successivo, ogni `{figure}` lo
    gira a sua volta, e il codice dopo una riga vuota dentro un blocco si
    legge come prosa. E' la prima trappola di `_maschera_prosa`, e la pila
    che la evita sta gia' la'.

    Il riempitivo e' pieno, non uno spazio, perche' la maschera **inventa**:
    azzerando con gli spazi, «il valore di $x$ cresce…» e «il valore di $y$
    cresce…» diventano la stessa frase. Con un carattere pieno la finestra
    che tocca codice, formule o URL resta riconoscibile, e si salta.

    E si salta anche la finestra che tocca una citazione. Due frasi fra
    virgolette che differiscono per una parola sono quasi sempre un parallelo
    voluto («il primo barattolo e il secondo, insieme, devono fare due litri»
    contro «…tre litri», in `Matematica/sistemi-lineari.md`, dove il punto e'
    proprio che le due promesse si somigliano), e non la firma di una
    riscrittura. Il prezzo, dichiarato: un doppione tutto dentro le virgolette
    qui non si vede.

    Sta a livello di modulo per la ragione di `contrapposizioni`: chi lo
    collauda deve eseguire il rilevatore vero.
    """
    maschera = _senza_citazioni(_maschera_prosa(testo, _PIENO), _PIENO)
    # le righe tutte piene in testa a un capoverso sono la recinzione che lo
    # apre (una scheda, un riquadro, una figura): a dire che cosa c'e' dentro
    # e' la prima riga dopo, `:class:` o `- ` o prosa che sia
    recinto = re.compile(rf"\A(?:[^\S\n]*{re.escape(_PIENO)}+[^\S\n]*\n)+")
    trovate, inizio, riga = [], 0, 1
    for m in re.finditer(r"\n[^\S\n]*\n|\Z", maschera):
        blocco, prima = maschera[inizio:m.start()], riga
        riga += maschera.count("\n", inizio, m.end())
        inizio = m.end()
        if _RX_NON_PROSA.match(recinto.sub("", blocco.lstrip(), count=1)):
            continue
        parole, visti = blocco.split(), {}
        for i in range(len(parole) - 7):
            chiave = " ".join(parole[i:i + 8]).lower()
            # una finestra con dentro codice o formule non e' prosa: li'
            # ripetersi e' normale e non dice niente
            if _RX_SEGNI_NON_PROSA.search(chiave):
                continue
            if chiave in visti and i - visti[chiave] < 90:
                trovate.append((prima, chiave))
                break
            visti[chiave] = i
    return trovate


# La marcatura di enfasi, contro cui `CONTRIBUTING.md` mette un tetto. `(?!\s)`
# e `(?<!\s)` perche' `** ` e ` **` in CommonMark non aprono e non chiudono
# niente, e `re.S` perche' il libro va a capo a ottanta colonne: una
# marcatura lunga sta spesso a cavallo di un a capo, e un pattern che non lo
# attraversa non la conta.
RX_NERETTO = re.compile(r"\*\*(?!\s)(?:(?!\*\*).)+?(?<!\s)\*\*", re.S)

# Il riempitivo con cui questo asse maschera: un carattere pieno, per la
# ragione scritta in `_maschera_prosa`. E prima della maschera va tolto `\*`,
# l'asterisco protetto di `**A\***`: lasciato li' fa chiudere il match un
# carattere prima e la marcatura si conta male. Si sostituisce con due
# caratteri, non con uno, cosi' le posizioni non si spostano.
_PIENO = "~"
_ASTERISCO_PROTETTO = ".."
# I segni che in una finestra di `frasi_doppie` dicono codice o formule.
_RX_SEGNI_NON_PROSA = re.compile("[" + re.escape("`=#\\$" + _PIENO) + "]")


def parole_e_marcature(testo: str) -> tuple[int, list[str]]:
    """(parole di prosa, testo di ogni marcatura `**...**`) di una pagina.

    Sta qui, a livello di modulo, perche' **chi misura deve poter eseguire il
    rilevatore vero** invece di riscriverne una copia: due copie divergono, e
    a quel punto non si sa piu' quale dei due conti abbia ragione.

    Le parole si contano sulla maschera a spazi e le marcature su quella
    piena, e non e' un capriccio: a spazi la matematica non conta come parola
    (giusto, non e' prosa), piena non spezza una marcatura che se la porta
    dentro (giusto, e' una marcatura sola). «Parola» qui e' una sequenza di
    caratteri non-spazio, quindi il denominatore comprende anche la
    punteggiatura e i segni del Markdown: e' una stima **larga**, cioe' un
    tasso prudente, ed e' quella su cui il tetto e' definito: cambiarla vuol
    dire cambiare il tetto.
    """
    parole = len(re.findall(r"\S+", _maschera_prosa(testo)))
    pieno = _maschera_prosa(testo.replace("\\*", _ASTERISCO_PROTETTO), _PIENO)
    return parole, [m[2:-2] for m in RX_NERETTO.findall(pieno)]


def termine_marcato(marcatura: str) -> str:
    """Il termine dietro una marcatura: minuscolo, senza i segni ai bordi.

    `**Il peso.**` e `**peso**` sono lo stesso termine, e la regola del libro
    («il grassetto marca un termine al suo primo uso, una volta sola») parla
    del termine, non della stringa.

    **E lo spazio dentro la marcatura non e' uno spazio**: il libro va a capo
    a ottanta colonne, quindi lo stesso termine si scrive `**stato\nnascosto**`
    in un capitolo e `**stato nascosto**` in un altro, a seconda di dove cade
    il margine. Chi non appiana quell'a capo si ritrova due termini diversi e
    conta **meno del vero**, che e' la direzione in cui un rilevatore mente
    piu' volentieri.
    """
    return re.sub(r"^[^\w$]+|[^\w$]+$", "",
                  re.sub(r"\s+", " ", marcatura).strip().lower())


def prosa_della_pagina(percorso: Path) -> str:
    """Il testo di una pagina del libro, notebook compresi.

    In un `.ipynb` la prosa sta nelle celle `markdown`: contare il resto vuol
    dire mettere nel denominatore le virgolette e i `"cell_type"` del JSON. E
    il campo `source` di una cella e' una lista di righe **oppure** una
    stringa, quindi vanno prese tutt'e due, o lo stesso ciclo conta le righe
    in un file e i caratteri nell'altro senza sollevare niente.
    """
    testo = percorso.read_text(encoding="utf-8", errors="ignore")
    if percorso.suffix != ".ipynb":
        return testo
    pezzi = []
    for cella in json.loads(testo).get("cells", []):
        if cella.get("cell_type") != "markdown":
            continue
        sorgente = cella.get("source", "")
        pezzi.append("".join(sorgente) if isinstance(sorgente, list)
                     else str(sorgente))
    return "\n\n".join(pezzi)


@functools.lru_cache(maxsize=1)
def _tempi_git() -> dict[str, int]:
    """{percorso relativo: data dell'ultimo commit che lo tocca}, in una passata.

    Serve perche' **la data di modifica di un file non dice niente in CI**: un
    `git checkout` scrive tutto l'albero in un istante solo, quindi ogni file
    risulta modificato nello stesso momento e ogni confronto fra due date
    diventa un sorteggio. Con le date dei file si va bene in locale e male
    sul runner: l'asse `stampa` dichiarerebbe vecchi dei fermi immagine su
    un albero in cui nessuno ha toccato niente.

    Una sola invocazione di `git log` per tutto il repository: farne una per
    file costa secondi e da' lo stesso risultato. E' la stessa mossa che fa
    `date_git()` in `genera-radice.py`, per la stessa ragione.
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


# ------------------------------------------------------------------ postille
# Le postille dei capoversi del testo comune. La definizione di capoverso e
# l'impronta stanno in `book/_ext/pt_postille_testo.py`, accanto all'estensione
# che le aggancia, e si importano da li' anche quando `--radice` punta a un
# libro finto: il modulo e' codice, il libro finto e' testo. In stampa una
# postilla di 80 caratteri occupa cinque righe di margine, e un capoverso sotto
# le sessanta parole e' piu' basso di lei: la nota scenderebbe accanto al
# capoverso dopo.
POSTILLA_MAX_CARATTERI = 80
POSTILLA_MIN_PAROLE = 60
_EXT = Path(__file__).resolve().parent.parent / "book" / "_ext"

_CATEGORIE_POSTILLE = {
    "illeggibile": "file di postille illeggibili",
    "senza_pagina": "file di postille senza la loro pagina nel toc",
    "doppia": "postille doppie nello stesso file",
    "da_riscrivere": "postille da riscrivere (lunghezza, marcature, lineette)",
    "corto": f"postille su capoversi sotto le {POSTILLA_MIN_PAROLE} parole",
    "spenta": "postille spente (da rileggere e riconfermare)",
    "ambigua": "postille ambigue (lo stesso capoverso due volte)",
    "forme": "forme da rivedere nelle postille",
    "conto": "postille per capitolo (accese, spente, candidati senza)",
}
_POSTILLE_SOLO_ELENCO = ("spenta", "ambigua", "forme", "conto")

# Le due lineette scritte in ASCII, che l'asse `lineette` cerca nella prosa e
# l'asse `postille` nel testo delle postille.
_RX_LINEETTA_DOPPIA = re.compile(r"(?<![-\w])--(?![-\w])|(?<=\w)--(?=\w)")
_RX_LINEETTA_SINGOLA = re.compile(r"(?<=\w) - (?=\w)")
_RX_MARCATURA = re.compile(r"[`$*_{}\[\]<>|\\]|https?://")


def postille_testo():
    """Il modulo condiviso delle postille."""
    if str(_EXT) not in sys.path:
        sys.path.insert(0, str(_EXT))
    import pt_postille_testo
    return pt_postille_testo


def parole_di_prosa(testo: str) -> int:
    """Le parole di un capoverso, senza codice, matematica e indirizzi."""
    return len(re.findall(r"\w+", _maschera_prosa(testo)))


def postilla_da_rifiutare(testo: str) -> str | None:
    """Che cosa non va in una postilla, o None se va bene."""
    if not testo.strip():
        return "è vuota"
    if "\n" in testo:
        return "va a capo"
    if len(testo) > POSTILLA_MAX_CARATTERI:
        return f"ha {len(testo)} caratteri, oltre gli {POSTILLA_MAX_CARATTERI}"
    if _RX_MARCATURA.search(testo):
        return "contiene una marcatura (formule, codice, link, neretto o corsivo)"
    if (chr(0x2014) in testo or _RX_LINEETTA_DOPPIA.search(testo)
            or _RX_LINEETTA_SINGOLA.search(testo)):
        return "contiene una lineetta"
    return None


def main():
    global QUI, LIBRO
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo", help="controlli da eseguire, separati da virgola")
    ap.add_argument("--radice", type=Path,
                    help="il repository da controllare (di norma questo)")
    ap.add_argument("--json", type=Path,
                    help="scrive anche le voci, con file e posizione, in "
                         "questo file")
    args = ap.parse_args()
    if args.radice:
        QUI = args.radice.resolve()
        LIBRO = QUI / "book"
    VOCI.clear()
    attivi = set(args.solo.split(",")) if args.solo else {
        "numref", "cite", "ref", "figure", "toc", "codifica", "esterni",
        "landing", "animazioni",
        "schede", "ricordare", "simboli", "avanti", "palette", "clip", "ambiente",
        "lineette", "stampa", "verso", "matematica", "doppioni",
        "contrapposizioni", "enfasi", "frasi", "postille"}

    testi = sorgenti()
    problemi = defaultdict(list)

    # target definiti ovunque nel libro (i riferimenti sono globali)
    nomi = set()
    for t in testi.values():
        nomi |= set(RX["name"].findall(t))
        nomi |= set(RX["label"].findall(t))

    if "numref" in attivi:
        for f, t in testi.items():
            for m in RX["numref"].finditer(t):
                r = m.group(1)
                if r.strip() not in nomi:
                    problemi["numref senza :name:"].append(f"{f}  ->  {r}")
                    voce("numref", "numref senza :name:", f, testo=t,
                         inizio=m.start(), fine=m.end(), identita=r.strip(),
                         messaggio=f"{{numref}} verso «{r.strip()}»: nessuna"
                                   " figura o tabella ha questo :name:")

        # Due difetti che la build non segnala e che si trovano solo contando.
        # I target di Sphinx sono GLOBALI: due figure diverse con lo stesso
        # :name: si costruiscono senza un avviso, e uno dei due {numref}
        # stampa il numero dell'altra. Lo stesso file richiamato da due
        # `{figure}` produce invece due figure numerate diverse con la stessa
        # immagine, che al lettore sembra una svista, e di solito lo e'.
        dove_nome, dove_file = defaultdict(list), defaultdict(list)
        posti_nome, posti_file = defaultdict(list), defaultdict(list)
        for f, t in testi.items():
            for m in RX["name"].finditer(t):
                dove_nome[m.group(1)].append(f)
                posti_nome[m.group(1)].append((f, t, m.start(1), m.end(1)))
            for m in RX["fig_uso"].finditer(t):
                dove_file[m.group(1)].append(f)
                posti_file[m.group(1)].append((f, t, m.start(1), m.end(1)))
        for n, dove in sorted(dove_nome.items()):
            if len(dove) > 1:
                problemi[":name: usato piu' volte"].append(
                    f"{n}  ->  {', '.join(dove)}")
                for f, t, a, b in posti_nome[n]:
                    voce("numref", ":name: usato piu' volte", f, testo=t,
                         inizio=a, fine=b, identita=n,
                         messaggio=f"il :name: «{n}» è usato più volte "
                                   f"({', '.join(dove)}): i bersagli valgono"
                                   " per tutto il libro")
        for g, dove in sorted(dove_file.items()):
            if len(dove) > 1:
                problemi["stessa figura richiamata piu' volte"].append(
                    f"{g}  ->  {', '.join(dove)}")
                for f, t, a, b in posti_file[g]:
                    voce("numref", "stessa figura richiamata piu' volte", f,
                         testo=t, inizio=a, fine=b, identita=g,
                         messaggio=f"la figura {g} è richiamata più volte "
                                   f"({', '.join(dove)})")

    if "ref" in attivi:
        steli = {Path(k).stem for k in testi}
        for f, t in testi.items():
            for m in [*RX["ref"].finditer(t), *RX["doc"].finditer(t)]:
                r = m.group(1).strip()
                if r in nomi:
                    continue
                # {doc} punta a un file: accetta se esiste
                if Path(r).stem in steli or any(
                        k.startswith(r.lstrip("/")) for k in testi):
                    continue
                problemi["ref/doc senza target"].append(f"{f}  ->  {r}")
                voce("ref", "ref/doc senza target", f, testo=t,
                     inizio=m.start(), fine=m.end(), identita=r,
                     messaggio=f"il rimando a «{r}» non porta a nessuna "
                               "pagina né etichetta")

    if "cite" in attivi:
        bib = chiavi_bib()
        if bib:
            for f, t in testi.items():
                for m in RX["cite"].finditer(t):
                    for c in (x.strip() for x in m.group(1).split(",")):
                        if c and c not in bib:
                            problemi["cite senza voce nel .bib"].append(
                                f"{f}  ->  {c}")
                            voce("cite", "cite senza voce nel .bib", f,
                                 testo=t, inizio=m.start(), fine=m.end(),
                                 identita=c,
                                 messaggio=f"la citazione «{c}» non ha una "
                                           "voce in references.bib, e in "
                                           "pagina sparisce senza avviso")
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

    if "codifica" in attivi:
        # Gli a capo CRLF e il BOM non si vedono in nessun editor, e la build
        # li digerisce. Ma un attrezzo che corregge le pagine con una patch
        # riga per riga, fatta su testo a capo LF, su quelle righe non le
        # ritrova. Si leggono i byte: `read_text` con le newline universali
        # converte gli a capo prima di restituire il testo, e il conto
        # tornerebbe zero senza errore. Il perimetro e' quello del toc; un
        # file fuori dal toc lo segnala gia' l'asse `toc`.
        for p in pagine_del_toc():
            dati = p.read_bytes()
            difetti = []
            if dati.startswith(codecs.BOM_UTF8):
                difetti.append("un BOM in testa")
            if b"\r\n" in dati:
                difetti.append("a capo CRLF")
            elif b"\r" in dati:
                difetti.append("un \\r da solo")
            if difetti:
                problemi["pagine con a capo CRLF o un BOM"].append(
                    f"{p.relative_to(LIBRO).as_posix()}: {', '.join(difetti)}")

    if "esterni" in attivi:
        # Il libro carica font e script da se': i font stanno nel brand, e la
        # CDN di Google e' uscita apposta. Uno script o un foglio di stile di
        # un altro dominio, scritto in una pagina o in `_config.yml`, rimette
        # in mezzo un terzo che riceve l'indirizzo di ogni lettore e puo'
        # cambiare cio' che gira sulla pagina senza passare da qui. Il bottone
        # di Buy Me a Coffee lo faceva sulla landing, e il suo script si
        # portava dietro il font Cookie da Google. La build non protesta e la
        # pagina si vede bene. Un link normale verso un altro sito va bene: lo
        # segue chi ci clicca. Gli esempi dentro un blocco di codice non
        # girano, quindi la posizione si guarda sulla maschera: un `<` che li'
        # e' diventato spazio sta nel codice. L'esca e' `strumenti/esca-esterni.py`.
        rx_est = re.compile(
            r"<(?:script|link)\b[^>]*?\b(?:src|href)\s*=\s*[\"']?"
            r"((?:https?:)?//[^\"'\s>]+)", re.I)
        config = LIBRO / "_config.yml"
        da_guardare = [("_config.yml", t, t) for t in
                       [config.read_text(encoding="utf-8")] if config.is_file()]
        da_guardare += [(nome, t, _maschera_prosa(t))
                        for nome, t in testi.items() if nome.endswith(".md")]
        for nome, t, maschera in da_guardare:
            for m in rx_est.finditer(t):
                if maschera[m.start()] != "<":
                    continue
                riga = t.count("\n", 0, m.start()) + 1
                problemi["script e fogli di stile da un altro dominio"].append(
                    f"{Path(nome).as_posix()}:{riga}: {m.group(1)}")

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
            posti = {}
            for m in re.finditer(r'class="pt-card"\s+href="([^"]+)"',
                                 testo_landing):
                posti.setdefault(m.group(1), m)
            for mancante in [a for a in attese if a not in schede]:
                problemi["capitoli senza scheda nella landing"].append(mancante)
                voce("landing", "capitoli senza scheda nella landing",
                     "intro.md", identita=mancante,
                     messaggio=f"manca la scheda del capitolo {mancante}")
            for extra in [s for s in schede if s not in attese]:
                problemi["schede della landing senza capitolo"].append(extra)
                voce("landing", "schede della landing senza capitolo",
                     "intro.md", testo=testo_landing,
                     inizio=posti[extra].start(1), fine=posti[extra].end(1),
                     identita=extra,
                     messaggio=f"la scheda {extra} non è un capitolo del "
                               "_toc.yml")
            if schede != attese and not (set(schede) ^ set(attese)):
                problemi["schede della landing fuori ordine"].append(
                    "l'ordine delle schede non e' quello del _toc.yml")
                voce("landing", "schede della landing fuori ordine",
                     "intro.md", identita="ordine",
                     messaggio="le schede non sono nell'ordine del _toc.yml")
            for m in re.finditer(r'<span class="pt-card-num">([^<]+)</span>',
                                 testo_landing):
                numero = m.group(1)
                problemi["numero scritto a mano nella scheda"].append(
                    f"«{numero}», lo conta il CSS, lo <span> va lasciato vuoto")
                voce("landing", "numero scritto a mano nella scheda",
                     "intro.md", testo=testo_landing, inizio=m.start(1),
                     fine=m.end(1), identita=numero,
                     messaggio=f"il numero «{numero}» è scritto a mano: lo "
                               "conta il CSS, lo <span> va lasciato vuoto")

            # Accanto a ogni scheda si scarica il capitolo in PDF, e il nome
            # del file lo decide la cartella. Se i due si scollano il lettore
            # clicca su un 404, che dalla pagina non si vede: il collegamento
            # c'e', il file no, e se ne accorge solo chi ci prova.
            for scheda in schede:
                atteso = f"paithon-book-{scheda.split('/')[0]}.pdf"
                if atteso not in testo_landing:
                    problemi["scheda senza il suo capitolo in PDF"].append(atteso)
                    voce("landing", "scheda senza il suo capitolo in PDF",
                         "intro.md", identita=atteso,
                         messaggio=f"la scheda non porta al suo capitolo in "
                                   f"PDF ({atteso})")

    if "schede" in attivi:
        # `sphinx-inline-tabs` unisce in UN SOLO gruppo le schede che si
        # toccano: due coppie Elementare/Superiore separate solo da righe
        # vuote diventano una barra da quattro linguette, con «Elementare»
        # scritto due volte. Il sorgente e' corretto, la build non protesta, e
        # il difetto si vede solo aprendo la pagina. Il rimedio e' una riga di
        # testo comune in mezzo, che e' anche la regola editoriale («le tab
        # avvolgono i passaggi dove la profondita' fa la differenza,
        # intervallati da testo universale»).
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
                    voce("schede", "schede contigue, che la build fonde", f,
                         riga=riga_inizio, identita="contigue",
                         messaggio=f"{len(gruppo)} linguette di fila "
                                   f"({' | '.join(gruppo)}): la build le "
                                   "fonde in un gruppo solo")
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
            for riga in schede_non_chiuse(t):
                problemi["scheda non chiusa, un'altra si apre dentro"].append(
                    f"{f}:{riga}")
                voce("schede", "scheda non chiusa, un'altra si apre dentro",
                     f, riga=riga, identita="non chiusa",
                     messaggio="una scheda si apre dentro un'altra rimasta "
                               "aperta: manca una chiusura")

        # E il difetto speculare, che nel sorgente si legge benissimo: una
        # Elementare e la sua Superiore separate da un capoverso di prosa.
        # Sono due gruppi da UNA linguetta, non una coppia: la Superiore resta
        # visibile a chi ha messo l'interruttore globale su Elementare, e
        # l'interruttore non puo' nasconderla perche' nel suo gruppo non ha
        # un'alternativa. Il sorgente non protesta, la build nemmeno, e si
        # vede solo contando le linguette nell'HTML. Lo produce una
        # riscrittura che infila la prosa in mezzo: e' un guasto da
        # revisione, e arriva proprio quando qualcuno rimette mano a una
        # pagina gia' buona.
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
                            voce("schede",
                                 "coppia di schede spezzata da prosa", f,
                                 riga=i + 1, identita="spezzata",
                                 messaggio="un capoverso fra l'Elementare e "
                                           "la Superiore spezza la coppia in"
                                           " due gruppi da una linguetta")
                        apertura = None
                    continue
                if apertura and riga.rstrip() == apertura:
                    chiusura = i

    if "schede" in attivi:
        # La lunghezza di una scheda Elementare non e' un difetto di per se',
        # ed e' per questo che qui si elenca e basta. Ma e' la deriva che le
        # correzioni producono senza che nessuno la veda: riparando un buco si
        # aggiunge un capoverso, il capoverso e' giusto, e una riparazione
        # dopo l'altra la scheda si allunga. Oltre una certa lunghezza la
        # scena evapora e resta un elenco di condizioni in parole piane, che e'
        # una Superiore tradotta: il livello Elementare a quel punto non fa piu'
        # il mestiere per cui esiste (il test dell'ombrellone). Il numero non
        # decide, ma dice dove guardare.
        TETTO = 400
        apre_scheda = re.compile(r"^(`{4,})\{tab\}\s*(\S+)")
        chiude_scheda = re.compile(r"^`{4,}\s*$")
        for f, testo in testi.items():
            if not f.endswith(".md"):
                continue
            corpo, dentro, inizio = [], False, 0
            for n, riga in enumerate(testo.split("\n"), 1):
                m = apre_scheda.match(riga)
                if m:
                    dentro = m.group(2).lower().startswith("element")
                    corpo, inizio = [], n
                    continue
                if chiude_scheda.match(riga):
                    parole = len(" ".join(corpo).split())
                    if dentro and parole > TETTO:
                        problemi["schede Elementari molto lunghe (da rileggere)"].append(
                            f"{f}:{inizio}  ->  {parole} parole")
                        voce("schede",
                             "schede Elementari molto lunghe (da rileggere)",
                             f, riga=inizio, identita="lunga",
                             messaggio=f"una scheda Elementare di {parole} "
                                       f"parole (oltre {TETTO}): la scena "
                                       "tende a evaporare")
                    dentro = False
                    continue
                if dentro:
                    corpo.append(riga)

    if "simboli" in attivi:
        # Una lettera fa un mestiere solo per capitolo, e la medicina e'
        # rinominare, non avvertire. Il difetto nasce naturale: due sottocampi
        # usano la stessa lettera per convenzione, il capitolo li mette nella
        # stessa pagina, e nessuno dei due autori sta sbagliando.
        #
        # Non si cercano le occorrenze, che sono migliaia: si cercano le
        # GLOSSE. Il libro introduce un simbolo scrivendo «$x$ e' il ...», e se
        # nello stesso capitolo lo stesso simbolo riceve due glosse diverse,
        # o e' una collisione o e' la stessa cosa detta due volte.
        #
        # **Elenca e basta, e molte righe non sono un difetto.** Le forme che
        # ritornano:
        #
        #   - artefatti della regex: il simbolo non e' il soggetto ma la coda
        #     di un complemento («la media di tutti gli $\ell$ **e' la**
        #     loss», «Cio' che risale da $D$ verso $G$ **e' il** gradiente»:
        #     li' $\ell$ e $G$ fanno un mestiere solo), oppure il pedice tolto
        #     ($F_1$ la metrica contro $F$ il simbolo iniziale di una
        #     grammatica);
        #   - lo stesso mestiere glossato due volte, che e' spreco e non
        #     errore («$r$ e' la ricompensa...» ripetuto in DeepRL, «$\sigma$
        #     e' la sigmoide» nei Transformer);
        #   - l'ambiguita' che **e' l'argomento** del paragrafo, e toglierla
        #     toglierebbe la lezione: la $p$ della cross-entropia in
        #     Matematica spiega con quelle due letture perche' la loss di
        #     training scenda a zero e quella di validazione no;
        #   - il riuso vero di una lettera consolidata in sezioni lontane,
        #     ognuna con la sua glossa.
        #
        # Prima di ripararne una si legga in `CONTRIBUTING.md` quando una
        # collisione e' un difetto e quando e' una convenzione: la medicina
        # non e' sempre rinominare, e questo asse non lo sa.
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
                    voce("simboli",
                         "una lettera con due glosse nello stesso capitolo",
                         None, capitolo=capitolo,
                         identita=f"{simbolo}: {' | '.join(sorted(glosse))}",
                         messaggio=f"${simbolo}$ ha due glosse diverse nel "
                                   "capitolo: "
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
        # il notebook e' COERENTE con il sorgente rotto, e mette l'etichetta
        # di sezione sbagliata sulla cella che ne nasce. Lo denuncia soltanto
        # un avviso di LaTeX, costruendo il PDF.
        recinto_con_testo = re.compile(r"^(`{3,})\s*(\S.*)$")
        for f, testo in testi.items():
            if not f.endswith(".md"):
                continue
            for n, riga in enumerate(testo.split("\n"), 1):
                m = recinto_con_testo.match(riga.rstrip())
                if m and not m.group(2).startswith("{") and " " in m.group(2).strip():
                    problemi["recinzione con del testo attaccato"].append(
                        f"{f}:{n}  {riga.strip()[:60]}")
                    voce("schede", "recinzione con del testo attaccato", f,
                         riga=n, identita=riga.strip(),
                         messaggio="testo attaccato a una recinzione: Sphinx"
                                   " la legge come l'apertura di un blocco "
                                   "di codice")

    if "ricordare" in attivi:
        # `book/aggiornamenti.md` promette al lettore, nero su bianco: «Ogni
        # capitolo si chiude con un riquadro "Da ricordare", scritto sui due
        # livelli come il resto del libro». Una promessa pubblica si verifica.
        #
        # Fuori dalle schede non e' di per se' un difetto: un riquadro scritto
        # in lingua comune va benissimo. Lo diventa quando e' scritto in lingua
        # Superiore, e quello un programma non lo sa giudicare. Quindi qui si
        # elenca e basta, e la lettura resta umana; ma l'elenco c'e', e questo
        # basta a non dimenticarsene.
        apre = re.compile(r"^\s*(`{3,})(.*)$")
        for f, t in sorted(testi.items()):
            if not f.endswith(".md"):
                continue
            pila, fuori = [], 0
            for numero, riga in enumerate(t.split("\n"), 1):
                m = apre.match(riga)
                if not m:
                    continue
                n, info = len(m.group(1)), m.group(2).strip()
                if "{admonition} Da ricordare" in info:
                    if not any(pila):
                        voce("ricordare", "«Da ricordare» fuori dalle schede",
                             f, riga=numero, identita="fuori dalle schede",
                             messaggio="un riquadro «Da ricordare» fuori dalle"
                                       " schede: il libro lo promette sui due"
                                       " livelli")
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
        # Le clip si contano, perche' una lista scritta a mano dei capitoli
        # ancora scoperti resta ferma mentre il libro cresce. Contarla non
        # invecchia.
        #
        # Zero clip e' un difetto, e non un avviso: un elenco dei capitoli
        # scoperti che non chiede niente a nessuno li lascia scoperti. La via
        # d'uscita e' una sola: dichiarare il capitolo in
        # `animazioni/senza-clip.toml` scrivendo perche' li' il tempo non e'
        # il contenuto. La differenza che conta non e' fra un capitolo
        # animato e uno fermo (un capitolo fermo puo' essere la scelta
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
        # esistono, ed e' scritta a mano: invecchia come ogni elenco scritto a
        # mano. Non la si genera (a un umano che apre il README serve leggerla
        # li'), la si verifica.
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
        # (`scrivi()` in paithon_svg.py). Le figure scritte a mano da li' non
        # passano, e senza questo asse ci finiscono colori che nessun
        # documento autorizza e lineette che nessuna ripulitura del testo vede.
        #
        # Le due cose non pesano uguale, e infatti qui non pesano uguale.
        #
        # La LINEETTA e' un difetto e basta: `CONTRIBUTING.md` la vieta, non
        # c'e' nessuna decisione in sospeso, e chi corregge il testo non ha
        # modo di sapere che quella lineetta e' disegnata anche altrove. Conta.
        #
        # Il COLORE fuori palette e' un elenco da guardare: un grigio usato in
        # molte figure puo' essere una convenzione stabilita che pero' esiste
        # solo nei file, e decidere se diventa un token o si accorpa a
        # `--pt-fg-muted` e' una decisione del repository `brand`, non di
        # questo. Finche' non e' presa, farla fallire qui vorrebbe dire tenere
        # il controllo rosso per una cosa che da qui non si puo' chiudere.
        #
        # E la palette e' quella del brand, letta dal submodule: i `colori` di
        # `figure/palette.json` e i loro `sinonimi`, gli stessi che
        # `genera-figure-scure.py` sa portare sullo scuro. Qui ce n'era una
        # copia di nove colori, ferma a prima che il brand li scrivesse, e
        # l'elenco dava per «fuori palette» sei colori che il brand ammette.
        # Senza il submodule l'asse non sa cosa ammettere: lo dice e conta,
        # invece di passare in silenzio. L'esca e' `strumenti/esca-palette.py`.
        mappa = LIBRO / "_static" / "brand" / "figure" / "palette.json"
        if mappa.is_file():
            brand = json.loads(mappa.read_text(encoding="utf-8"))
            AMMESSI = {c.upper() for c in (*brand["colori"], *brand["sinonimi"])}
        else:
            AMMESSI = None
            problemi["palette del brand mancante (git submodule update "
                     "--init book/_static/brand)"].append(
                str(mappa.relative_to(QUI)))
        rx_col = re.compile(r"#[0-9A-Fa-f]{6}")
        fuori = defaultdict(int)
        for p in sorted((LIBRO / "figures").glob("*.svg")):
            t = p.read_text(encoding="utf-8", errors="ignore")
            if "\u2014" in t or "&#8212;" in t:
                problemi["lineette dentro le figure"].append(p.name)
            if "<script" in t:
                problemi["script dentro le figure"].append(p.name)
            if AMMESSI is None:
                continue
            for c in {x.upper() for x in rx_col.findall(t)} - AMMESSI:
                fuori[c] += 1
        # A pari merito, in ordine alfabetico: l'insieme dei colori di una
        # figura si scorre in un ordine che cambia da un processo all'altro
        # (`PYTHONHASHSEED`), e due giri sullo stesso libro devono stampare
        # la stessa cosa.
        for c, n in sorted(fuori.items(), key=lambda kv: (-kv[1], kv[0])):
            problemi["colori fuori palette (decisione del repo brand)"].append(
                f"{c}  ->  in {n} figure")


    if "clip" in attivi:
        # Le SVG generate hanno `animazioni/svg/genera.py --verifica`, che le
        # rigenera in un temporaneo e le confronta. Le clip Manim no: sono
        # video, rigenerarle costa, e due render della stessa scena non danno
        # comunque file identici. Risultato: fra `animazioni/*.py` e
        # `book/figures/*.gif` non c'e' **nessun** legame verificabile, e si
        # puo' correggere il sorgente, committare, e lasciare online la clip
        # vecchia senza che niente protesti: una formula dentro un'immagine
        # non la trova nessuna grep.
        #
        # Qui non si confrontano i video, si confrontano le date dei commit. Ma
        # la sola data grida al lupo, e un controllo che segnala cose che non
        # sono difetti viene disattivato, il che e' peggio di non averlo: una
        # modifica alla sola **docstring** di una scena, o a un suo
        # **commento**, sposta la data e non arriva nel video. Quindi si
        # guarda anche il diff, e si segnala solo quando cambia una riga che il
        # render puo' vedere.
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
        # CON NOTA, mai in silenzio.
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
                # loro animazione finiscono in commit diversi.
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
        # prende il font di sistema di chi legge, anche online.
        #
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
        # `CONTRIBUTING.md` vieta la lineetta lunga (—) perche' non e' nello
        # stile dell'autore. Ma la stessa cosa si scrive in altri due modi: la
        # **doppia lineetta** `--` e il **trattino singolo spaziato** ` - `.
        # Vanno tenuti fuori con lo stesso rigore, e a differenza della
        # lineetta lunga non si vedono a colpo d'occhio scorrendo una pagina.
        #
        # La parte delicata e' non gridare al lupo, perche' in un libro tecnico
        # quei segni compaiono legittimamente dappertutto: segni meno nelle
        # formule, opzioni da riga di comando, indici, commenti HTML, classi
        # CSS, separatori di tabella, elenchi. Un interruttore acceso/spento
        # sui recinti si desincronizza sulle schede a cinque backtick, e la
        # matematica in linea che va a capo non si vede guardando una riga per
        # volta: la mascheratura sta in `_maschera_prosa`, che la fa **una
        # volta sola**.
        doppia = _RX_LINEETTA_DOPPIA
        singola = _RX_LINEETTA_SINGOLA
        di_tabella = re.compile(r"^\s*\|[\s|:-]+\|\s*$")
        di_trattini = re.compile(r"^\s*-{3,}\s*$")
        segno_di_elenco = re.compile(r"^\s*[-*+]\s")
        for f, t in sorted(testi.items()):
            if not f.endswith(".md"):
                continue
            for n, riga in enumerate(_maschera_prosa(t).split("\n"), 1):
                if "<!--" in riga or "-->" in riga or '="' in riga:
                    continue
                if di_tabella.match(riga) or di_trattini.match(riga):
                    continue
                pulita = segno_di_elenco.sub("", riga)
                m = doppia.search(pulita) or singola.search(pulita)
                if m:
                    problemi["lineette scritte in ASCII (-- oppure - )"].append(
                        f"{f}:{n}  {pulita.strip()[:88]}")
                    scarto = len(riga) - len(pulita)
                    voce("lineette", "lineette scritte in ASCII (-- oppure - )",
                         f, riga=n, colonna=m.start() + scarto + 1,
                         fine_colonna=m.end() + scarto + 1,
                         identita="lineetta",
                         messaggio="una lineetta scritta in ASCII: il libro "
                                   "usa parentesi, virgole o due punti")

    if "contrapposizioni" in attivi:
        # «non e' X, e' Y». `CONTRIBUTING.md` ne ammette **due per capitolo**,
        # e solo dove la contrapposizione *e'* l'argomento. Fuori di li' e' la
        # cadenza da diapositiva: il paragrafo si presenta prima di parlare.
        #
        # Quattro cose, che stanno tutte nel codice qui sotto.
        #
        # **La prima: il costrutto ha tre punteggiature, non una.** Un
        # rilevatore che cerca solo la virgola ne trova una parte, e dichiara
        # dentro il tetto capitoli che non lo sono: le altre forme hanno i due
        # punti o il punto e virgola. E' la ragione per cui qui le tre forme
        # sono nella stessa regex invece che in tre passate successive.
        #
        # **La seconda: la negazione dentro una citazione non e' il
        # costrutto.** «una su due non e' a posto»: sono due bocciature
        # diverse» e la frase del capitolo sulle GPU richiamata fra virgolette
        # («il collo di bottiglia non sono i conti, sono i byte») sono
        # riferimenti, non mosse fatte adesso. Chi le «ripara» rompe una
        # citazione, quindi il testo fra «» e fra "" si maschera prima.
        #
        # **La terza: sopra il tetto non si fallisce, si elenca.** Un
        # controllo sempre rosso non avvisa di niente, e' soltanto un cancello
        # che si impara ad aggirare. Falliscono i soli capitoli gia' portati
        # dentro il tetto, elencati in `CHIUSI`: li' tornare sopra i due e'
        # una regressione e va vista subito. Gli altri si elencano e basta, e
        # `CHIUSI` **cresce** man mano che un capitolo rientra. Con `None` al
        # posto dell'insieme varrebbe per tutti, e l'asse resterebbe rosso per
        # chiunque finche' un capitolo solo sta sopra il tetto.
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
        occorrenze = defaultdict(list)
        for f, t in sorted(testi.items()):
            if "/" not in f:
                continue
            capitolo = f.split("/")[0]
            for n, frase in contrapposizioni(t):
                per_capitolo[capitolo].append(f"{f}:{n}  {frase[:78]}")
                occorrenze[capitolo].append(
                    {"file": f, "riga": n, "identita": frase})

        for capitolo, righe in sorted(per_capitolo.items()):
            chiave = ("«non e' X, e' Y» oltre il tetto di due"
                      if CHIUSI is None or capitolo in CHIUSI else
                      "«non e' X, e' Y» oltre il tetto (da leggere)")
            # La misura c'e' anche sotto il tetto: chi confronta due giri deve
            # sapere da quanto e' partito.
            misura("contrapposizioni", chiave, capitolo, len(righe), TETTO,
                   len(righe) > TETTO, occorrenze[capitolo])
            if len(righe) <= TETTO:
                continue
            problemi[chiave].append(
                f"{capitolo}: {len(righe)} (tetto {TETTO})")
            for r in righe[:4]:
                problemi[chiave].append(f"    {r}")
            if len(righe) > 4:
                problemi[chiave].append(f"    … e altre {len(righe) - 4}")

    if "frasi" in attivi:
        # Elenco e basta, come le contrapposizioni fuori da CHIUSI: quasi ogni
        # forma ha anche un uso giusto, e un controllo che fallisce su una
        # frase giusta si impara ad aggirare. Il perimetro e' quello del toc,
        # e il conto dichiara quante pagine ha letto, perche' zero non si
        # verifica da se'.
        chiave = "forme da rivedere («La frase» di CONTRIBUTING.md)"
        lette, trovate = 0, []
        for p in pagine_del_toc():
            pagina = str(p.relative_to(LIBRO))
            if pagina == "aggiornamenti.md":
                continue
            lette += 1
            for n, famiglia, forma in frasi_da_rivedere(
                    prosa_della_pagina(p), pagina):
                trovate.append(f"{pagina}:{n}  [{famiglia}]  {forma}")
                voce("frasi", chiave, pagina, riga=n,
                     identita=f"{famiglia}: {forma.lower()}",
                     messaggio=f"[{famiglia}] «{forma}»: una forma fissa da "
                               "rileggere (la frase, in CONTRIBUTING.md)")
        assert lette, "il toc non ha dato nessuna pagina da leggere"
        if trovate:
            problemi[chiave] = [f"(lette {lette} pagine del toc;"
                                " aggiornamenti.md esclusa: e' generata)"] + trovate

    if "enfasi" in attivi:
        # Il tetto e' scritto in `CONTRIBUTING.md`: **otto marcature di
        # enfasi ogni mille parole di prosa**. Il neretto marca un termine
        # dove viene definito, e le etichette di un blocco (Definizione,
        # Esempio, Osservazione) che fanno da appiglio all'occhio; non marca
        # il ritmo del ragionamento.
        #
        # Questo asse non giudica **quale** marcatura sia giusta, e non e' una
        # rinuncia: i due assi provati e scartati che stanno in cima a questo
        # file volevano proprio quello (un termine usato prima di essere
        # definito, un termine coniato nella sola Superiore) e sono affogati
        # nel rumore, perche' nel libro il grassetto non marca solo
        # l'introduzione di un termine. Un tasso non ha quel problema: conta,
        # e la soglia e' gia' decisa altrove.
        #
        # Il perimetro viene dal `_toc.yml` (`file:` e `root:`), non da un
        # glob, e il conto stampa quante pagine ha letto: un conteggio senza
        # il suo denominatore non e' una misura, e zero e' l'unico numero che
        # non si verifica da se'.
        TETTO = 8.0
        # `aggiornamenti.md` sta fuori, e l'esclusione e' **dichiarata**: la
        # scrive `scripts/genera-aggiornamenti.py` dal registro delle
        # versioni, e il suo neretto e' il titolo di ciascuna voce, cioe'
        # struttura. Senza, un registro lungo diventa un muro di prosa.
        # Ripararla a mano non avrebbe senso (la prossima rigenerazione lo
        # cancella) e lasciarla dentro darebbe un rosso che nessuno puo'
        # togliere, cioe' un controllo che si impara ad aggirare.
        # Un'esclusione taciuta e' un buco; dichiarata e' una decisione.
        ESCLUSE = {"aggiornamenti.md":
                   "generata da scripts/genera-aggiornamenti.py: il suo "
                   "neretto sono i titoli delle voci del registro"}
        # Sotto questa soglia il tasso non e' una misura: una marcatura sola
        # lo sposta di piu' di due per mille. Riguarda le pagine che non sono
        # prosa (`references.md` e' una direttiva `{bibliography}`), e anche
        # questa esclusione si stampa invece di tacere.
        MINIMO = 500

        pagine = pagine_del_toc()
        assert pagine, "perimetro vuoto: il _toc.yml non elenca nessuna pagina"
        gruppi: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        capitoli_del_termine = defaultdict(list)
        lette = 0
        for pagina in pagine:
            rel = pagina.relative_to(LIBRO).as_posix()
            if rel in ESCLUSE:
                continue
            lette += 1
            parole, marcature = parole_e_marcature(prosa_della_pagina(pagina))
            gruppo = rel.split("/")[0] if "/" in rel else rel
            gruppi[gruppo][0] += len(marcature)
            gruppi[gruppo][1] += parole
            # I termini si contano nei soli **capitoli**: la regola «una volta
            # sola» parla di un termine presentato al lettore, e la pagina di
            # apertura per mestiere nomina tutto il libro.
            if "/" in rel:
                for m in marcature:
                    termine = termine_marcato(m)
                    if termine:
                        capitoli_del_termine[termine].append(gruppo)

        marcature_totali = sum(m for m, _ in gruppi.values())
        parole_totali = sum(w for _, w in gruppi.values())
        piccoli = sorted(g for g, (_, w) in gruppi.items() if w < MINIMO)
        # Il conto dice **quante pagine ha letto**, e le esclusioni le nomina
        # una per una con la loro ragione: un conteggio senza il suo
        # denominatore non e' una misura, e un'esclusione taciuta e' un buco.
        fuori = [f"{nome} ({ragione})" for nome, ragione in ESCLUSE.items()]
        if piccoli:
            fuori.append(f"{', '.join(piccoli)} sotto le {MINIMO} parole,"
                         f" dove il tasso non e' una misura")
        print(f"   (enfasi: {lette} pagine lette su {len(pagine)} del toc,"
              f" {marcature_totali} marcature su {parole_totali} parole di"
              f" prosa, {1000 * marcature_totali / parole_totali:.2f} per"
              f" mille. Fuori: {'; '.join(fuori)})")

        for gruppo, (marcature, parole) in sorted(gruppi.items()):
            if parole < MINIMO:
                continue
            tasso = 1000 * marcature / parole
            misura("enfasi", "enfasi oltre il tetto di otto per mille",
                   gruppo, round(tasso, 3), TETTO, tasso > TETTO)
            if tasso > TETTO:
                problemi["enfasi oltre il tetto di otto per mille"].append(
                    f"{gruppo}: {tasso:.2f} per mille ({marcature} marcature"
                    f" su {parole} parole di prosa, tetto {TETTO:.0f})")

        # E l'altra meta', che una rilettura a un capitolo per volta non puo'
        # vedere: **lo stesso termine marcato in due capitoli**. La regola e'
        # una proprieta' del libro, non di un capitolo, e chi legge un
        # capitolo per volta non la puo' controllare. Non e' un errore, ed e'
        # per questo che qui si elenca: molti sono omonimi legittimi (il
        # `peso` di una rete e il `peso` di un campione), e a distinguerli
        # bisogna leggere. Quello che l'elenco dice e' **dove guardare**.
        multi = {t: c for t, c in capitoli_del_termine.items()
                 if len(set(c)) > 1}
        if multi:
            for termine, dove in sorted(multi.items()):
                voce("enfasi",
                     "termini in neretto in piu' di un capitolo (da leggere)",
                     None, identita=termine, capitoli=sorted(set(dove)),
                     messaggio=f"«{termine}» è in neretto in "
                               f"{len(set(dove))} capitoli "
                               f"({', '.join(sorted(set(dove)))}): il neretto"
                               " marca un termine una volta sola, al primo "
                               "uso")
            chiave = "termini in neretto in piu' di un capitolo (da leggere)"
            problemi[chiave].append(
                f"{len(multi)} termini, per "
                f"{sum(len(c) for c in multi.values())} marcature su "
                f"{sum(len(c) for c in capitoli_del_termine.values())} "
                f"nei capitoli")
            for termine, dove in sorted(
                    multi.items(),
                    key=lambda kv: (-len(set(kv[1])), -len(kv[1]), kv[0])):
                if len(set(dove)) < 3:
                    continue
                problemi[chiave].append(
                    f"{termine}: {len(set(dove))} capitoli"
                    f" ({', '.join(sorted(set(dove)))})")
            # Qui sopra si elencano i termini che stanno in **tre o piu'**
            # capitoli, che sono l'elenco corto e ad alto segnale. Quelli in
            # due si contano e basta: sono la coda lunga, e un elenco
            # lunghissimo non lo legge nessuno.
            due = sum(1 for c in multi.values() if len(set(c)) == 2)
            problemi[chiave].append(
                f"… e {due} termini marcati in due capitoli soltanto,"
                f" contati e non elencati")

    if "matematica" in attivi:
        # Dentro `$...$` la tipografia italiana non vale, e nessuno se ne
        # accorge. Una sostituzione degli apostrofi ASCII con quello
        # tipografico (che serve, perche' `smartquotes`, in italiano, rende
        # `'` come `”` a fine parola) che entra in una formula trasforma la
        # derivata prima in un apostrofo: `s'`, `a'`, `w'` del Reinforcement
        # Learning diventano `s’`, `a’`, `w’`.
        #
        # La build Sphinx non alza un avviso, e MathJax `’` lo disegna e basta:
        # il difetto si vede solo aprendo la pagina e riconoscendo che quel
        # segno non e' un primo. In stampa e' peggio, perche' LuaLaTeX gira in
        # nonstopmode. E l'asse `lineette` qui non guarda, perche' maschera la
        # matematica apposta.
        #
        # Riga per riga il `$$` di blocco non si vede: sta su piu' righe. Qui
        # la matematica si estrae dal testo intero, con la stessa pila di
        # recinti dell'asse `lineette`, e i numeri di riga si ricavano dagli
        # offset.
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
        FORMULA_A_CAPO = re.compile(
            r"[ \t]*(?:[-+*][ \t]|\d+[.)][ \t]|>|#+[ \t])")
        testuale = re.compile(r"\\(?:text|textrm|textit|textbf|mbox)\{[^{}]*\}")
        LINK_TESTO = re.compile(r"\{(?:doc|ref|numref)\}`([^`]*)`"
                                r"|\[([^\]\n]*(?:\n[^\]\n]*)?)\]\(")
        for f, t in sorted(testi.items()):
            if not f.endswith(".md"):
                continue
            prosa = solo_prosa(t)
            # Le formule si cercano a codice in linea spento, per la ragione
            # scritta in `_maschera_prosa`: un `$` fra due backtick si
            # appaierebbe con il dollaro dopo, e il pezzo di prosa in mezzo
            # diventerebbe una formula piena di segni di prosa. Le posizioni
            # restano quelle di `prosa`, che serve intera ai rimandi qui sotto.
            formule_prosa = re.sub(
                r"`[^`]*`", lambda x: re.sub(r"[^\n]", " ", x.group(0)), prosa)
            for m in formule.finditer(formule_prosa):
                corpo = testuale.sub(lambda x: " " * len(x.group(0)), m.group(0))
                for i, ch in enumerate(corpo):
                    if ch in SEGNI:
                        n = prosa.count("\n", 0, m.start() + i) + 1
                        estratto = " ".join(m.group(0).split())[:70]
                        problemi["segni di prosa dentro una formula"].append(
                            f"{f}:{n}  {ch}  {SEGNI[ch]}\n      {estratto}")
                        voce("matematica", "segni di prosa dentro una formula",
                             f, testo=t, inizio=m.start() + i,
                             fine=m.start() + i + 1, identita=ch,
                             messaggio=f"«{ch}» dentro una formula: "
                                       f"{SEGNI[ch]}")
            # Una formula in linea che va a capo su una riga che comincia
            # con `- `, `+ `, `* `, `1. `, `>` o `# ` non e' piu' una
            # formula: Markdown legge quell'inizio di riga come una voce
            # d'elenco, una citazione o un titolo, e la spezza in due. Online
            # e in stampa escono il sorgente grezzo, un pallino e la frase
            # dopo in corsivo matematico, e la build non protesta. La cura e'
            # sempre la stessa: il segno in coda alla riga prima, o l'a capo
            # in un altro punto. Una coppia di `$` che scavalca una riga vuota
            # non e' una formula ma due dollari spaiati, e si salta.
            for m in formule.finditer(formule_prosa):
                if m.group(0).startswith("$$"):
                    continue
                if re.search(r"\n[ \t]*\n", m.group(0)):
                    continue
                corpo = m.group(0)
                for k in [i for i, ch in enumerate(corpo) if ch == "\n"]:
                    resto = corpo[k + 1:]
                    if FORMULA_A_CAPO.match(resto):
                        n = prosa.count("\n", 0, m.start() + k) + 2
                        estratto = resto.lstrip()[:60].split("\n")[0]
                        problemi["formula in linea spezzata da un a capo"
                                 ].append(f"{f}:{n}  {estratto}")
                        voce("matematica",
                             "formula in linea spezzata da un a capo", f,
                             testo=t, inizio=m.start() + k + 1,
                             identita="a capo",
                             messaggio="una formula in linea va a capo su un "
                                       "segno che Markdown legge come inizio"
                                       " di un elenco, di una citazione o di "
                                       "un titolo")
            # Il testo di un rimando ({doc}, {ref}, un link Markdown) e' testo
            # letterale: una `$n$` li' dentro non si compone, e in pagina e
            # sul sito escono i dollari. Si scrive senza matematica («modelli
            # n-gram»), o la formula si porta fuori dal link.
            for m in LINK_TESTO.finditer(prosa):
                etichetta = m.group(1) if m.group(1) is not None \
                    else m.group(2)
                etichetta = etichetta.split("<")[0]
                if "$" in etichetta:
                    n = prosa.count("\n", 0, m.start()) + 1
                    problemi["formula dentro il testo di un rimando"].append(
                        f"{f}:{n}  {' '.join(etichetta.split())[:60]}")
                    voce("matematica", "formula dentro il testo di un rimando",
                         f, testo=t, inizio=m.start(), fine=m.end(),
                         identita=" ".join(etichetta.split()),
                         messaggio="una formula nel testo di un rimando: resta"
                                   " testo, e in pagina si vedono i dollari")

    if "doppioni" in attivi:
        # Il rilevatore e le sue ragioni stanno in `frasi_doppie`. Il registro
        # degli aggiornamenti resta fuori: e' generato, e le sue voci si
        # ripetono per costruzione.
        for f, t in testi.items():
            if f.endswith("aggiornamenti.md"):
                continue
            for riga, chiave in frasi_doppie(t):
                problemi["frasi scritte due volte"].append(
                    f"{f}:{riga}  ->  «{chiave}…»")
                voce("doppioni", "frasi scritte due volte", f, riga=riga,
                     identita=chiave,
                     messaggio=f"la stessa frase due volte nello stesso "
                               f"capoverso: «{chiave}…»")

    if "verso" in attivi:
        # Una frase giusta nel verso sbagliato: «Come abbiamo visto» seguito
        # da un rimando a un capitolo che il lettore non ha ancora letto, e
        # che gli chiede un ricordo che non puo' avere. Nessuna build se ne
        # accorge, perche' il link risolve benissimo: e' l'ordine di lettura a
        # non tornare, e l'ordine di lettura sta nel _toc.yml.
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
                            voce("verso",
                                 "rimandi all'indietro che puntano in avanti",
                                 f, testo=t, inizio=m.start(), fine=m.end(),
                                 identita=f"{cap} {bersaglio}",
                                 messaggio="«come abbiamo visto» verso un "
                                           "capitolo che il lettore non ha "
                                           f"ancora letto ({cap})")
                        elif suo < mio and RX["poi"].search(prima):
                            problemi["rimandi in avanti che puntano indietro"].append(
                                f"{f}  ->  {cap}  …{coda}")
                            voce("verso",
                                 "rimandi in avanti che puntano indietro",
                                 f, testo=t, inizio=m.start(), fine=m.end(),
                                 identita=f"{cap} {bersaglio}",
                                 messaggio="«vedremo» verso un capitolo che "
                                           f"il lettore ha già letto ({cap})")

            # I richiami in PROSA, che sono la forma vera: il libro scrive
            # «come abbiamo visto nel capitolo sulle GAN», non un {doc}, e un
            # asse che guardasse solo i link guarderebbe la forma che il libro
            # non usa.
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
                            voce("verso",
                                 "rimandi all'indietro che puntano in avanti",
                                 f, testo=t, inizio=m.start(),
                                 fine=m.end() + mt.end(),
                                 identita=f"{cap} {m.group(0).lower()}",
                                 messaggio="«come abbiamo visto» verso un "
                                           "capitolo che il lettore non ha "
                                           f"ancora letto ({cap})")
                            break

    if "avanti" in attivi:
        for f, t in testi.items():
            for m in RX["avanti"].finditer(t):
                problemi["rimandi in avanti (da leggere)"].append(
                    f"{f}: …{m.group(0).strip()[:110]}")
                voce("avanti", "rimandi in avanti (da leggere)", f, testo=t,
                     inizio=m.start(), fine=m.end(),
                     identita=" ".join(m.group(0).lower().split()),
                     messaggio=f"un rimando in avanti, da rileggere: "
                               f"«{m.group(0).strip()[:110]}»")

    if "postille" in attivi:
        # Ferma soltanto cio' che sta nel file delle postille. Una postilla
        # spenta, o finita in una scheda, si elenca: la spegne chi corregge il
        # capoverso, e una correzione dell'autore non va bloccata da una riga
        # che non ha scritto lui.
        pp = postille_testo()
        cat = _CATEGORIE_POSTILLE
        pagine_toc = {str(p.relative_to(LIBRO)): p for p in pagine_del_toc()}
        conti: dict[str, list[int]] = {}
        for file in sorted(LIBRO.rglob("*" + pp.ESTENSIONE)):
            if "_build" in file.parts or "_static" in file.parts:
                continue
            nome_file = str(file.relative_to(LIBRO))
            pagina = nome_file[: -len(pp.ESTENSIONE)] + ".md"
            if pagina not in pagine_toc:
                problemi[cat["senza_pagina"]].append(f"{nome_file}  ->  {pagina}")
                voce("postille", cat["senza_pagina"], nome_file, identita=pagina,
                     messaggio=f"la pagina {pagina} non è nel _toc.yml")
                continue
            try:
                voci = pp.leggi_postille(file)
            except ValueError as errore:
                problemi[cat["illeggibile"]].append(f"{nome_file}: {errore}")
                voce("postille", cat["illeggibile"], nome_file, identita=nome_file,
                     messaggio=f"il file delle postille non si legge: {errore}")
                continue
            capoversi = pp.capoversi(pagine_toc[pagina].read_text(encoding="utf-8"))
            per_impronta = defaultdict(list)
            for c in capoversi:
                per_impronta[c.impronta].append(c)
            quante = Counter(v.impronta for v in voci)
            for v in voci:
                dove = f"{pagina}: «{v.inizio}»"
                trovati = per_impronta.get(v.impronta, [])
                riga = trovati[0].riga if len(trovati) == 1 else None
                if quante[v.impronta] > 1:
                    problemi[cat["doppia"]].append(dove)
                    voce("postille", cat["doppia"], pagina, riga=riga,
                         identita=v.impronta,
                         messaggio="due postille per lo stesso capoverso")
                rifiuto = postilla_da_rifiutare(v.postilla)
                if rifiuto:
                    problemi[cat["da_riscrivere"]].append(f"{dove}: {rifiuto}")
                    voce("postille", cat["da_riscrivere"], pagina, riga=riga,
                         identita=v.impronta, messaggio=f"la postilla {rifiuto}")
                for _, famiglia, forma in frasi_da_rivedere(v.postilla):
                    problemi[cat["forme"]].append(f"{dove}: {famiglia}: «{forma}»")
                    voce("postille", cat["forme"], pagina, riga=riga,
                         identita=forma, messaggio=f"{famiglia}: «{forma}»")
                if not trovati:
                    problemi[cat["spenta"]].append(f"{dove}: il capoverso è cambiato")
                    voce("postille", cat["spenta"], pagina, identita=v.impronta,
                         messaggio=f"postilla spenta: «{v.inizio}» è cambiato")
                    continue
                if len(trovati) > 1:
                    righe = ", ".join(str(c.riga) for c in trovati)
                    problemi[cat["ambigua"]].append(f"{dove}: righe {righe}")
                    voce("postille", cat["ambigua"], pagina, identita=v.impronta,
                         messaggio=f"lo stesso capoverso alle righe {righe}")
                    continue
                c = trovati[0]
                if c.zona != pp.COMUNE:
                    problemi[cat["spenta"]].append(
                        f"{dove}: il capoverso ora sta in {c.zona}")
                    voce("postille", cat["spenta"], pagina, riga=c.riga,
                         identita=v.impronta,
                         messaggio=f"postilla spenta: il capoverso ora sta in {c.zona}")
                    continue
                parole = parole_di_prosa(c.testo)
                if parole < POSTILLA_MIN_PAROLE:
                    problemi[cat["corto"]].append(
                        f"{pagina}:{c.riga}  «{v.inizio}» ({parole} parole)")
                    voce("postille", cat["corto"], pagina, riga=c.riga,
                         identita=v.impronta,
                         messaggio=f"il capoverso ha {parole} parole, "
                                   f"sotto le {POSTILLA_MIN_PAROLE}")
            accese, spente = pp.abbina(capoversi, voci)
            capitolo = pagina.split("/")[0]
            conto = conti.setdefault(capitolo, [0, 0])
            conto[0] += len(accese)
            conto[1] += len(spente)
        for capitolo, (n_accese, n_spente) in sorted(conti.items()):
            senza = 0
            for nome, p in pagine_toc.items():
                if p.suffix != ".md" or nome.split("/")[0] != capitolo:
                    continue
                file = pp.file_delle_postille(p)
                try:
                    voci = pp.leggi_postille(file) if file.is_file() else []
                except ValueError:
                    voci = []
                capoversi = pp.capoversi(p.read_text(encoding="utf-8"))
                accese, _ = pp.abbina(capoversi, voci)
                senza += sum(1 for c in capoversi
                             if c.zona == pp.COMUNE and c.riga not in accese
                             and parole_di_prosa(c.testo) >= POSTILLA_MIN_PAROLE)
            problemi[cat["conto"]].append(
                f"{capitolo}: {n_accese} accese, {n_spente} spente, "
                f"{senza} capoversi candidati senza postilla")

    # `ordine` non e' solo l'ordine di stampa: e' anche il filtro. Una chiave
    # che non compare qui verrebbe calcolata e poi buttata via in silenzio, e
    # per questo in fondo le chiavi senza posto si stampano lo stesso.
    ordine = ["numref senza :name:", ":name: usato piu' volte",
              "stessa figura richiamata piu' volte",
              "cite senza voce nel .bib",
              "ref/doc senza target", "figure mai richiamate",
              "file non nel _toc.yml",
              "pagine con a capo CRLF o un BOM",
              "capitoli senza scheda nella landing",
              "schede della landing senza capitolo",
              "schede della landing fuori ordine",
              "numero scritto a mano nella scheda",
              "scheda senza il suo capitolo in PDF",
              "schede contigue, che la build fonde",
              "coppia di schede spezzata da prosa",
              "scheda non chiusa, un'altra si apre dentro",
              "recinzione con del testo attaccato",
              "schede Elementari molto lunghe (da rileggere)",
              "capitoli oltre il tetto di 10 clip",
              "clip fuori dalla tabella del README",
              "capitoli senza figure animate (dichiarali in "
              "animazioni/senza-clip.toml, con la ragione)",
              "lineette scritte in ASCII (-- oppure - )",
              "«non e' X, e' Y» oltre il tetto di due",
              "«non e' X, e' Y» oltre il tetto (da leggere)",
              "enfasi oltre il tetto di otto per mille",
              "termini in neretto in piu' di un capitolo (da leggere)",
              "frasi scritte due volte",
              "«Da ricordare» fuori dalle schede",
              "una lettera con due glosse nello stesso capitolo",
              "clip piu' vecchie del loro sorgente",
              "ambiente delle clip cambiato",
              "lineette dentro le figure",
              "script dentro le figure",
              "script e fogli di stile da un altro dominio",
              "palette del brand mancante (git submodule update "
              "--init book/_static/brand)",
              "colori fuori palette (decisione del repo brand)",
              "fermi immagine mancanti",
              "fermi immagine senza impronta",
              "fermi immagine piu' vecchi dell'animazione",
              "fermi immagine orfani",
              "figure che non dichiarano i font del brand",
              "segni di prosa dentro una formula",
              "formula in linea spezzata da un a capo",
              "formula dentro il testo di un rimando",
              "rimandi all'indietro che puntano in avanti",
              "rimandi in avanti che puntano indietro",
              "rimandi in avanti (da leggere)",
              "forme da rivedere («La frase» di CONTRIBUTING.md)"]
    # Assi che elencano e basta: dicono cosa guardare, non cosa e' rotto, e
    # quindi non fanno fallire niente.
    solo_elenco = {"schede Elementari molto lunghe (da rileggere)",
                   "termini in neretto in piu' di un capitolo (da leggere)",
                   "una lettera con due glosse nello stesso capitolo",
                   "rimandi in avanti (da leggere)",
                   "colori fuori palette (decisione del repo brand)",
                   "«non e' X, e' Y» oltre il tetto (da leggere)",
                   "«Da ricordare» fuori dalle schede",
                   "forme da rivedere («La frase» di CONTRIBUTING.md)"}
    ordine += list(_CATEGORIE_POSTILLE.values())
    solo_elenco |= {_CATEGORIE_POSTILLE[k] for k in _POSTILLE_SOLO_ELENCO}
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
    # E una chiave che manca da `ordine` non deve sparire: la si stampa e la
    # si conta, perche' un asse che calcola e non stampa resta muto proprio
    # quando trova qualcosa.
    for k, v in sorted(problemi.items()):
        if k in ordine or not v:
            continue
        print(f"\n=== {k} ({len(v)}) [chiave senza posto in `ordine`: "
              f"aggiungerla alla lista]")
        for riga in v:
            print(f"   {riga}")
        totale += len(v)

    print(f"\n{totale} problemi da correggere"
          f" ({len(problemi.get('rimandi in avanti (da leggere)', []))}"
          f" rimandi da leggere a mano)")
    if args.json:
        coperte = {v["categoria"] for v in VOCI}
        for k, righe in sorted(problemi.items()):
            # Un asse che non sa ancora dare le posizioni: la riga stampata,
            # intera. Una voce che manca dall'uscita strutturata non la vede
            # nessuno di quelli che la leggono, ed e' la stessa trappola
            # delle chiavi senza posto in `ordine`.
            if k not in coperte:
                for r in righe:
                    voce("coerenza", k, identita=r, messaggio=f"{k}: {r}")
        for v in VOCI:
            v["solo_elenco"] = v["categoria"] in solo_elenco
        args.json.write_text(json.dumps(
            {"radice": str(QUI), "assi": sorted(attivi), "voci": VOCI},
            ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 1 if totale else 0


if __name__ == "__main__":
    sys.exit(main())
