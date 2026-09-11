"""Un filtro non e' un quadrato: e' un mazzo alto quanto le mappe che legge.

`DeepLearning/reti-convoluzionali.md` lo dice tre volte e non lo mostra mai: la
scheda Elementare («lo stampino del secondo strato non scorre su un foglio
solo: scorre su tutti e 32 insieme»), la Superiore (il kernel a quattro indici,
$K_{f,c,m,n}$, e «il $C$ di uno strato e' l'$F$ dello strato sotto») e il
commento del blocco PyTorch («Il primo numero e' 32: ogni filtro di questo
strato legge tutte e 32 le mappe»). E' il passaggio su cui il lettore si ferma
davanti a `nn.Conv2d(32, 64, 3)`, e la pagina aveva due figure, tutte e due
sulla finestra che scorre.

I riferimenti gli dedicano una sezione con la sua figura (d2l, §7.4 «Multiple
Input and Multiple Output Channels», fig. 7.4.1): e' il caso in cui la fonte
spiega meglio con un'immagine, ed e' questa.

Il mazzo e' **sfalsato quel tanto che basta perche' le finestre delle mappe
dietro si vedano**, e non e' una scelta di gusto: la didascalia promette che la
finestra si affaccia su tutte nello stesso punto, e con lo sfalsamento piccolo
che verrebbe naturale le mappe davanti coprirebbero le finestre di quelle
dietro. La figura prometterebbe una cosa che il disegno non contiene.

I numeri li calcola la scena da CANALI, LATO e FILTRI, e `verifica()` pretende
quelli della didascalia: i 27 pesi, i 28 col bias, gli 896 dello strato,
l'altezza 32 del mazzo dopo, e che la finestra cada nello stesso punto di ogni
mappa del mazzo.

Ferma: e' un confronto fra due strati, cioe' geometria, e il tempo non
c'entra.
"""

from paithon_svg import *

NOME = "filtro-alto-quanto-le-mappe"
TITOLO = "il filtro alto quanto le mappe che legge"

# --------------------------------------------------------------------------
# I parametri della scena: da questi discende ogni numero che si legge
# --------------------------------------------------------------------------
CANALI_1 = 3        # le mappe che arrivano al primo strato: i tre colori
LATO = 3            # il kernel 3x3
FILTRI_1 = 32       # gli stampini del primo strato
CANALI_2 = FILTRI_1  # e quindi le mappe che arrivano allo strato dopo
VISIBILI = 4        # quante mappe si disegnano di un mazzo da trentadue

PESI_FILTRO = CANALI_1 * LATO * LATO
PESI_CON_BIAS = PESI_FILTRO + 1
PESI_STRATO = FILTRI_1 * PESI_CON_BIAS

# --------------------------------------------------------------------------
# Geometria. La riga su cui tutto e' centrato, e le due sfalsature dei mazzi.
# --------------------------------------------------------------------------
LARG, ALT = 880, 340
RIGA = 150
PANNELLO = 416                   # la larghezza utile di un pannello
SINISTRE = (16, 456)             # dove cominciano i due pannelli
DIVISORIO = 442
CELLE_MAPPA = 5                  # una mappa e' una griglia 5x5
LATO_MAPPA, LATO_FILTRO = 74, 38
SFALSO_MAPPE, SFALSO_FILTRO = 16, 13
FINESTRA = (1, 0)                # colonna 1, riga 0: la fila di sopra
LATO_USCITA, CELLE_USCITA = 40, 4
Y_ETICHETTE, Y_NOTE = 232, 266


def griglia(x: float, y: float, lato: float, celle: int, classe: str) -> str:
    """Una mappa quadrata divisa in `celle` per lato. Torna gli elementi."""
    passo = lato / celle
    p = [f'<rect class="{classe}" x="{x:.1f}" y="{y:.1f}" '
         f'width="{lato:.1f}" height="{lato:.1f}"/>']
    for i in range(1, celle):
        d = i * passo
        p.append(f'<line class="{classe}l" x1="{x + d:.1f}" y1="{y:.1f}" '
                 f'x2="{x + d:.1f}" y2="{y + lato:.1f}"/>')
        p.append(f'<line class="{classe}l" x1="{x:.1f}" y1="{y + d:.1f}" '
                 f'x2="{x + lato:.1f}" y2="{y + d:.1f}"/>')
    return "".join(p)


def freccia(x1: float, y: float, x2: float) -> str:
    return (f'<line class="ar" x1="{x1:.1f}" y1="{y:.1f}" x2="{x2 - 6:.1f}" '
            f'y2="{y:.1f}"/>'
            f'<path class="ar" d="M {x2 - 7:.1f} {y - 4.5:.1f} L {x2:.1f} '
            f'{y:.1f} L {x2 - 7:.1f} {y + 4.5:.1f}" fill="none"/>')


def mazzo(x: float, y: float, lato: float, celle: int, quante: int,
          sfalso: float, classe: str, finestra: bool, puntini: bool):
    """Mappe sfalsate in un mazzo, le lontane disegnate prima.

    Torna gli elementi e, se c'e' la finestra, la sua posizione **relativa** a
    ogni mappa: e' quello che `verifica()` controlla.
    """
    p, dove = [], []
    for i in reversed(range(quante)):
        mx, my = x + i * sfalso, y - i * sfalso
        p.append(griglia(mx, my, lato, celle, classe))
        if finestra:
            passo = lato / celle
            fx, fy = mx + FINESTRA[0] * passo, my + FINESTRA[1] * passo
            fl = LATO * passo
            p.append(f'<rect class="fn" x="{fx:.1f}" y="{fy:.1f}" '
                     f'width="{fl:.1f}" height="{fl:.1f}"/>')
            dove.append((round(fx - mx, 3), round(fy - my, 3)))
    if puntini:
        bx, by = x + (quante - 1) * sfalso, y - (quante - 1) * sfalso
        p.append(f'<text class="ls" x="{bx + lato * 0.42:.1f}" y="{by - 7:.1f}" '
                 f'text-anchor="middle">&#8943;</text>')
    return "".join(p), dove


def pannello(sinistra: float, titolo: str, classe_titolo: str, canali: int,
             puntini: bool, note: list[str]):
    """Un pannello: mazzo di mappe, filtro, somma, casella in uscita."""
    centro = sinistra + PANNELLO / 2
    quante = VISIBILI if puntini else canali
    alza_m = (quante - 1) * SFALSO_MAPPE
    alza_f = (quante - 1) * SFALSO_FILTRO
    p = [f'<text class="{classe_titolo}" x="{centro:.0f}" y="36" '
         f'text-anchor="middle">{titolo}</text>']

    x1 = sinistra + 24                                  # le mappe in ingresso
    corpo, dove = mazzo(x1, RIGA - LATO_MAPPA / 2 + alza_m / 2, LATO_MAPPA,
                        CELLE_MAPPA, quante, SFALSO_MAPPE, "mp", True, puntini)
    p.append(corpo)
    p.append(f'<text class="lb" x="{x1 + (LATO_MAPPA + alza_m) / 2:.1f}" '
             f'y="{Y_ETICHETTE}" text-anchor="middle">{canali} mappe</text>')

    x2 = x1 + LATO_MAPPA + alza_m + 46                  # il mazzo del filtro
    p.append(freccia(x1 + LATO_MAPPA + alza_m + 10, RIGA, x2 - 10))
    corpo, _ = mazzo(x2, RIGA - LATO_FILTRO / 2 + alza_f / 2, LATO_FILTRO,
                     LATO, quante, SFALSO_FILTRO, "ff", False, puntini)
    p.append(corpo)
    p.append(f'<text class="lb" x="{x2 + (LATO_FILTRO + alza_f) / 2:.1f}" '
             f'y="{Y_ETICHETTE}" text-anchor="middle">un filtro</text>')

    cx = x2 + LATO_FILTRO + alza_f + 42                 # la somma
    p.append(freccia(x2 + LATO_FILTRO + alza_f + 10, RIGA, cx - 20))
    p.append(f'<circle class="sm" cx="{cx:.1f}" cy="{RIGA}" r="16"/>')
    p.append(f'<text class="sg" x="{cx:.1f}" y="{RIGA + 6.5}" '
             f'text-anchor="middle">&#931;</text>')
    p.append(f'<text class="ls" x="{cx:.1f}" y="{RIGA - 26}" '
             f'text-anchor="middle">si somma</text>')

    x4 = cx + 42                                        # la mappa in uscita
    p.append(freccia(cx + 20, RIGA, x4 - 8))
    p.append(griglia(x4, RIGA - 20, LATO_USCITA, CELLE_USCITA, "mp"))
    passo = LATO_USCITA / CELLE_USCITA
    p.append(f'<rect class="us" x="{x4 + passo:.1f}" y="{RIGA - 20 + passo:.1f}" '
             f'width="{passo:.1f}" height="{passo:.1f}"/>')
    p.append(f'<text class="lb" x="{x4 + LATO_USCITA / 2:.1f}" '
             f'y="{Y_ETICHETTE}" text-anchor="middle">una casella</text>')

    for i, r in enumerate(note):
        p.append(f'<text class="ls" x="{centro:.0f}" y="{Y_NOTE + i * 18}" '
                 f'text-anchor="middle">{r}</text>')
    return "".join(p), dove, x4 + LATO_USCITA


def costruisci() -> Figura:
    primo, dove_1, fine_1 = pannello(
        SINISTRE[0], "il primo strato", "t1", CANALI_1, False,
        ["una griglia per ogni mappa che arriva:",
         f"{CANALI_1} × {LATO} × {LATO} = {PESI_FILTRO} pesi, "
         f"e col bias fanno {PESI_CON_BIAS}",
         f"{FILTRI_1} filtri, {FILTRI_1} mappe in uscita, "
         f"{FILTRI_1} × {PESI_CON_BIAS} = {PESI_STRATO} pesi"])
    dopo, dove_2, fine_2 = pannello(
        SINISTRE[1], "lo strato dopo", "t2", CANALI_2, True,
        [f"le mappe che arrivano sono {CANALI_2},",
         f"quindi il mazzo è alto {CANALI_2}: la finestra si affaccia",
         "su tutte nello stesso punto, e ne esce un numero solo"])

    verifica(dove_1, dove_2, fine_1, fine_2)

    divisorio = (f'<line class="dv" x1="{DIVISORIO}" y1="24" '
                 f'x2="{DIVISORIO}" y2="312"/>')
    return Figura(
        larghezza=LARG,
        altezza=ALT,
        alt="Due pannelli affiancati mostrano lo stesso percorso a due strati "
            "diversi. A sinistra, il primo strato: tre mappe quadrate sfalsate "
            "in un mazzo, ciascuna divisa in venticinque celle, con la stessa "
            "finestra da tre per tre evidenziata nello stesso punto di tutte e "
            "tre. Una freccia porta a un mazzo di tre griglie da nove, che è "
            "il filtro, poi a un cerchio col simbolo di somma, poi a una mappa "
            "in uscita in cui una sola casella è colorata. Sotto, il conto: "
            "una griglia per ogni mappa che arriva, tre per tre per tre fa "
            "ventisette pesi, ventotto col bias, e con trentadue filtri escono "
            "trentadue mappe e trentadue per ventotto fa ottocentonovantasei "
            "pesi. A destra, lo strato dopo: identico, ma le mappe in ingresso "
            "sono trentadue, disegnate come quattro mappe più dei puntini di "
            "continuazione, e il mazzo del filtro è alto trentadue allo stesso "
            "modo; l'uscita resta una casella sola.",
        corpo=primo + divisorio + dopo,
        stile=f"""    .mp  {{ fill:{CREAM}; stroke:{BORDER_STRONG}; stroke-width:1.1; }}
    .mpl {{ stroke:{BORDER_STRONG}; stroke-width:0.9; }}
    .ff  {{ fill:{CREAM}; stroke:{TERRACOTTA}; stroke-width:1.6; }}
    .ffl {{ stroke:{TERRACOTTA}; stroke-width:1.3; }}
    .fn  {{ fill:{TERRACOTTA}; fill-opacity:0.16; stroke:{TERRACOTTA};
            stroke-width:2; }}
    .us  {{ fill:{TEAL}; fill-opacity:0.5; stroke:{TEAL}; stroke-width:2; }}
    .sm  {{ fill:none; stroke:{TEAL}; stroke-width:1.8; }}
    .sg  {{ font-family:{SANS}; font-size:18px; font-weight:700; fill:{TEAL}; }}
    .ar  {{ stroke:{FG_MUTED}; stroke-width:1.8; stroke-linecap:round;
            stroke-linejoin:round; }}
    .dv  {{ stroke:{BORDER_STRONG}; stroke-width:1; stroke-dasharray:3 5; }}
    .lb  {{ font-family:{SANS}; font-size:12px; fill:{INK}; }}
    .ls  {{ font-family:{SANS}; font-size:11.5px; fill:{FG_MUTED}; }}
    .t1  {{ font-family:{SANS}; font-size:13.5px; font-weight:700;
            fill:{TERRACOTTA}; }}
    .t2  {{ font-family:{SANS}; font-size:13.5px; font-weight:700;
            fill:{TEAL}; }}""",
    )


def verifica(dove_1, dove_2, fine_1: float, fine_2: float) -> None:
    """Difende quello che la didascalia promette, numero per numero."""
    assert (PESI_FILTRO, PESI_CON_BIAS, PESI_STRATO) == (27, 28, 896), \
        (PESI_FILTRO, PESI_CON_BIAS, PESI_STRATO)
    assert CANALI_2 == FILTRI_1 == 32, (CANALI_2, FILTRI_1)
    # la finestra cade nello stesso punto di **ogni** mappa del mazzo: e'
    # l'affermazione della didascalia, e senza questa il disegno mostrerebbe
    # tre finestre in tre posti diversi
    assert len(set(dove_1)) == 1 and len(dove_1) == CANALI_1, dove_1
    assert len(set(dove_2)) == 1 and len(dove_2) == VISIBILI, dove_2
    assert set(dove_1) == set(dove_2), "i due pannelli guardano punti diversi"
    # e le finestre delle mappe dietro devono **vedersi**: lo sfalsamento
    # supera il bordo che la mappa davanti copre
    assert SFALSO_MAPPE > FINESTRA[1] * LATO_MAPPA / CELLE_MAPPA, \
        "le mappe davanti coprono le finestre di quelle dietro"
    # i due pannelli stanno nella tela e non si toccano
    assert fine_1 < DIVISORIO, f"il primo pannello sfora: {fine_1}"
    assert fine_2 <= LARG, f"il secondo pannello sfora: {fine_2}"
