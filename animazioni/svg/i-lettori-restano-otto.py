"""Le quattro formulazioni dell'attenzione, e la cosa che nessuna tocca.

`Transformers/attenzione-in-pratica.md` racconta MHA, MQA, GQA e MLA in
parole, e chiude avvertendo dell'equivoco che si sente ripetere: che
condividere i taccuini voglia dire avere meno punti di vista. L'avvertimento
regge finché il lettore riesce a tenere in testa quattro schemi di
collegamento; disegnati uno accanto all'altro, la fila dei lettori che resta
lunga uguale **è** l'avvertimento, e non serve crederci sulla parola.

La figura serve anche a un secondo mestiere, ed è quello per cui è nata: le
quattro sigle stanno nel titolo della sezione e nella tabella che il codice
stampa, e chi legge la scena dei taccuini non aveva modo di sapere quale
sigla fosse quale. Qui ciascuna sta sopra il proprio schema.

Quello che qui non si disegna, e resta nella tabella del codice, è quanto
pesa ciascuna in memoria: quei numeri valgono per un modello da trentadue
teste e trentadue strati, e disegnarli sopra uno schema da otto lettori
farebbe credere che vengano da qui.

I numeri li calcola la scena da LETTORI e GRUPPI, e `verifica()` pretende
quelli della didascalia: otto lettori in tutti e quattro gli schemi, otto
taccuini contro due contro uno, e le frazioni di spazio che ne discendono.

Ferma: non è una progressione nel tempo da guardare scorrere, sono quattro
schemi da confrontare a colpo d'occhio.
"""

from paithon_svg import *

NOME = "i-lettori-restano-otto"
TITOLO = "le quattro formulazioni, e i lettori che restano otto"

# --------------------------------------------------------------------------
# I parametri della scena: da questi discende ogni numero che si legge
# --------------------------------------------------------------------------
LETTORI = 8          # le teste di query, le stesse in tutti e quattro
GRUPPI = 2           # in quanti gruppi GQA divide i lettori

PER_GRUPPO = LETTORI // GRUPPI
FRAZIONI = {2: "metà", 4: "un quarto", 8: "un ottavo"}

# (sigla, quanti taccuini, righe di didascalia sotto lo schema)
SCHEMI = [
    ("MHA", LETTORI, ["uno per lettore"]),
    ("GQA", GRUPPI, [f"uno ogni {PER_GRUPPO},",
                     f"lo spazio a {FRAZIONI[LETTORI // GRUPPI]}"]),
    ("MQA", 1, ["uno per tutti,", f"lo spazio a {FRAZIONI[LETTORI]}"]),
    ("MLA", 1, ["stenografato, più", "una riga per l'ordine"]),
]

# --------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------
LARG, ALT = 880, 292
PANNELLO, VUOTO, X0 = 200, 20, 12
PASSO = 20                       # quanto distano due lettori
RAGGIO = 7
Y_SIGLA, Y_LETTORI, Y_TACC, ALTA = 26, 62, 130, 36
LARGA = 14                       # quanto è largo un taccuino singolo
Y_SOTTO = Y_TACC + ALTA + 24
Y_NOTA = ALT - 20


def x_lettore(x0: float, i: int) -> float:
    """L'ascissa dell'i-esimo lettore dentro il pannello che parte in x0."""
    primo = x0 + (PANNELLO - (LETTORI - 1) * PASSO) / 2
    return primo + i * PASSO


def taccuino(x0: float, primo: int, quanti: int) -> tuple[float, float]:
    """Il taccuino che copre i lettori da `primo` a `primo + quanti - 1`."""
    sinistra = x_lettore(x0, primo) - LARGA / 2
    larghezza = (quanti - 1) * PASSO + LARGA
    return sinistra, larghezza


def pannello(indice: int, sigla: str, taccuini: int,
             sotto: list[str]) -> tuple[str, int]:
    """Uno dei quattro schemi. Torna gli elementi e quante linee ha tirato."""
    x0 = X0 + indice * (PANNELLO + VUOTO)
    centro = x0 + PANNELLO / 2
    p = [f'<text class="sg" x="{centro:.0f}" y="{Y_SIGLA}" '
         f'text-anchor="middle">{sigla}</text>']

    # i taccuini, e per ciascuno da quale lettore a quale. La larghezza dice
    # sempre e solo *quanti lettori serve* quel taccuino, mai quanto pesa: il
    # taccuino di MLA copre tutti e otto come quello di MQA, ed è largo
    # uguale, perché disegnarlo più stretto direbbe che occupa meno memoria,
    # che è il contrario di quello che la tabella del capitolo misura.
    if sigla == "MLA":
        _, larghezza = taccuino(x0, 0, LETTORI)
        sinistra = centro - (larghezza + 10 + LARGA) / 2
        coperture = [(sinistra, larghezza, 0, LETTORI)]
        p.append(f'<rect class="tc st" x="{sinistra:.1f}" y="{Y_TACC}" '
                 f'width="{larghezza:.1f}" height="{ALTA}" rx="3"/>')
        # la riga tenuta fuori dal riassunto: il segnale dell'ordine
        p.append(f'<rect class="or" x="{sinistra + larghezza + 10:.1f}" '
                 f'y="{Y_TACC}" width="{LARGA}" height="{ALTA}" rx="3"/>')
    else:
        quanti = LETTORI // taccuini
        coperture = []
        for g in range(taccuini):
            sinistra, larghezza = taccuino(x0, g * quanti, quanti)
            coperture.append((sinistra, larghezza, g * quanti, quanti))
            p.append(f'<rect class="tc" x="{sinistra:.1f}" y="{Y_TACC}" '
                     f'width="{larghezza:.1f}" height="{ALTA}" rx="3"/>')

    # una linea per lettore, verso il taccuino che lo serve
    linee = 0
    for i in range(LETTORI):
        for sinistra, larghezza, primo, quanti in coperture:
            if primo <= i < primo + quanti:
                p.append(f'<line class="cn" x1="{x_lettore(x0, i):.1f}" '
                         f'y1="{Y_LETTORI + RAGGIO}" '
                         f'x2="{sinistra + larghezza / 2:.1f}" '
                         f'y2="{Y_TACC}"/>')
                linee += 1
                break

    # i lettori, disegnati sopra le linee perché le coprano
    for i in range(LETTORI):
        p.append(f'<circle class="lt" cx="{x_lettore(x0, i):.1f}" '
                 f'cy="{Y_LETTORI}" r="{RAGGIO}"/>')

    p.append(f'<text class="lb" x="{centro:.0f}" y="{Y_SOTTO}" '
             f'text-anchor="middle">{taccuini} '
             f'{"taccuino" if taccuini == 1 else "taccuini"}</text>')
    for k, riga in enumerate(sotto):
        p.append(f'<text class="ls" x="{centro:.0f}" '
                 f'y="{Y_SOTTO + 19 + k * 16}" '
                 f'text-anchor="middle">{riga}</text>')
    return "".join(p), linee


def costruisci() -> Figura:
    corpi, linee = [], []
    for i, (sigla, taccuini, sotto) in enumerate(SCHEMI):
        corpo, tirate = pannello(i, sigla, taccuini, sotto)
        corpi.append(corpo)
        linee.append(tirate)

    nota = (f'<text class="ls" x="{LARG / 2:.0f}" y="{Y_NOTA}" '
            f'text-anchor="middle">In alto i lettori: sono {LETTORI} in tutti '
            f'e quattro gli schemi. A cambiare è solo che cosa leggono.</text>')

    verifica(linee)

    return Figura(
        larghezza=LARG,
        altezza=ALT,
        alt="Quattro schemi affiancati, uno per formulazione. In ciascuno, in "
            "alto, la stessa fila di otto pallini: i lettori, cioè le teste di "
            "query. Sotto, i taccuini di chiave e valore, e una linea porta "
            "ogni lettore al taccuino che consulta. In MHA i taccuini sono "
            "otto, uno per lettore. In GQA sono due, ciascuno condiviso da "
            "quattro lettori, e lo spazio scende a un quarto. In MQA è uno "
            "solo, letto da tutti e otto, e lo spazio scende a un ottavo. In "
            "MLA è ancora uno, letto da tutti e otto, ma tratteggiato perché "
            "dentro c'è un riassunto stenografato, e accanto sta una striscia "
            "ocra a parte che porta il segnale dell'ordine delle parole. La "
            "fila di otto pallini in cima resta identica nei quattro schemi.",
        corpo="".join(corpi) + nota,
        stile=f"""    .sg  {{ font-family:{SANS}; font-size:14px; font-weight:600;
            letter-spacing:0.06em; fill:{INK}; }}
    .lt  {{ fill:{TERRACOTTA}; fill-opacity:0.20; stroke:{TERRACOTTA};
            stroke-width:1.6; }}
    .tc  {{ fill:{CREAM}; stroke:{TEAL}; stroke-width:1.6; }}
    .st  {{ stroke-dasharray:5 3; }}
    .or  {{ fill:{CREAM}; stroke:{OCRA}; stroke-width:1.6; }}
    .cn  {{ stroke:{BORDER_STRONG}; stroke-width:1.1; }}
    .lb  {{ font-family:{SANS}; font-size:13px; fill:{INK}; }}
    .ls  {{ font-family:{SANS}; font-size:11.5px; fill:{FG_MUTED}; }}""",
    )


def verifica(linee: list[int]) -> None:
    """Difende quello che la didascalia promette, numero per numero."""
    # il fatto per cui la figura esiste: i lettori non cambiano mai
    assert LETTORI == 8, LETTORI
    assert linee == [LETTORI] * len(SCHEMI), linee
    # e i taccuini sì
    assert [t for _, t, _ in SCHEMI] == [8, 2, 1, 1], SCHEMI
    assert [s for s, _, _ in SCHEMI] == ["MHA", "GQA", "MQA", "MLA"], SCHEMI
    # i gruppi di GQA partizionano i lettori, o lo schema mentirebbe
    assert LETTORI % GRUPPI == 0, (LETTORI, GRUPPI)
    assert PER_GRUPPO == 4, PER_GRUPPO
    # le due frazioni scritte sotto gli schemi
    assert FRAZIONI[LETTORI // GRUPPI] == "un quarto", FRAZIONI
    assert FRAZIONI[LETTORI] == "un ottavo", FRAZIONI
    # la larghezza dice quanti lettori serve un taccuino, e nient'altro: MLA
    # ne serve otto come MQA, quindi i due sono larghi uguale. Un MLA più
    # stretto prometterebbe una cache più piccola di quella di MQA, mentre la
    # tabella del capitolo la misura più grande (36 KiB contro 16)
    _, largo_mqa = taccuino(0, 0, LETTORI)
    _, largo_mla = taccuino(0, 0, LETTORI)
    assert largo_mla == largo_mqa, (largo_mla, largo_mqa)
    # e i due oggetti di MLA devono stare dentro il pannello
    assert largo_mla + 10 + LARGA <= PANNELLO, largo_mla
    # e tutto deve stare nella tela
    assert X0 + 4 * PANNELLO + 3 * VUOTO <= LARG, LARG
    assert Y_NOTA < ALT, (Y_NOTA, ALT)
