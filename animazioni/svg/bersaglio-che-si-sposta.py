"""Apprendimento online: il passo che si accorcia e il passo fisso, davanti a un cambio.

È la scena di `MachineLearning/dati-che-cambiano.md`, sezione «Imparare un
esempio alla volta». Un flusso di esempi etichettati da una regressione
logistica in cinque dimensioni, la stessa del blocco di codice della pagina:
fino a metà la regola è `W_PRIMA`, poi diventa `W_DOPO`. Due allievi imparano
con la discesa del gradiente online, uno col passo 1/radice(t) e uno col passo
fisso 0,2, e la figura traccia per ciascuno la distanza fra i suoi pesi e quelli
della regola vera, un punto ogni `OGNI` esempi.

Il flusso si sorteggia con `random.Random(SEME)` e i conti si fanno con
`math`, in Python puro: niente BLAS e niente SIMD, quindi lo stesso SVG su ogni
macchina che abbia la stessa libm. È un addestramento, ma di cinque pesi e
senza riduzioni in ordine variabile, e non ha bisogno di un dato misurato.

La didascalia promette tre cose, e gli assert le difendono: prima del cambio il
passo che si accorcia sta più vicino alla regola del passo fisso; dopo il cambio
il passo fisso si riavvicina in poche centinaia di esempi e l'altro ci mette
migliaia; alla fine il passo che si accorcia torna il più vicino dei due. Il
disegno fermo è lo stato finale, con le due curve intere.
"""

import math
import random

from paithon_svg import *

NOME = "bersaglio-che-si-sposta"
TITOLO = "il passo che si accorcia e il passo fisso, davanti a un cambio di regola"

SEME = 0
T, CAMBIO, OGNI = 20_000, 10_000, 100
W_PRIMA = [2.0, -1.5, 1.0, 0.5, -0.5]
W_DOPO = [-1.0, -1.5, 2.0, 0.5, 1.5]
FISSO = 0.2
VICINO = 1.0                 # «riavvicinarsi» vuol dire tornare sotto questa distanza
RAGGIO = 5.0                 # la palla su cui proietta anche il blocco della pagina

X0, X1 = 90, 640
Y0, ALT = 40, 250            # la fascia: distanza da 0 (in basso) a YMAX (in alto)
YMAX = 4.5


def flusso():
    rnd = random.Random(SEME)
    esempi = []
    for t in range(T):
        x = [rnd.uniform(-1, 1) for _ in range(5)]
        w = W_PRIMA if t < CAMBIO else W_DOPO
        z = sum(a * b for a, b in zip(x, w))
        y = 1.0 if rnd.random() < 1 / (1 + math.exp(-z)) else 0.0
        esempi.append((x, y))
    return esempi


def distanze(esempi, passo):
    """La distanza dalla regola vera dopo ogni esempio (un valore ogni OGNI)."""
    w = [0.0] * 5
    fuori = []
    for t, (x, y) in enumerate(esempi, start=1):
        z = sum(a * b for a, b in zip(x, w))
        g = 1 / (1 + math.exp(-z)) - y
        w = [wi - passo(t) * g * xi for wi, xi in zip(w, x)]
        norma = math.hypot(*w)
        if norma > RAGGIO:                     # la proiezione, come nella pagina
            w = [wi * RAGGIO / norma for wi in w]
        if t % OGNI == 0:
            fuori.append(math.dist(w, W_PRIMA if t < CAMBIO else W_DOPO))
    return fuori


def px(k):                   # k-esimo punto, cioè dopo (k + 1) * OGNI esempi
    return X0 + (k + 1) * OGNI / T * (X1 - X0)


def py(d):
    return Y0 + (YMAX - d) / YMAX * ALT


def costruisci() -> Figura:
    esempi = flusso()
    corto = distanze(esempi, lambda t: 1 / math.sqrt(t))
    fisso = distanze(esempi, lambda t: FISSO)
    media = lambda v, a, b: sum(v[a:b]) / (b - a)
    k_cambio = CAMBIO // OGNI
    prima = slice(k_cambio - 20, k_cambio)                 # i 2000 esempi prima
    assert media(corto, prima.start, prima.stop) < 0.6 * media(fisso, prima.start, prima.stop), \
        "prima del cambio il passo che si accorcia non è più vicino"
    rientro = lambda v: next(k for k in range(k_cambio, len(v)) if v[k] < VICINO) - k_cambio + 1
    assert rientro(fisso) * OGNI <= 300, f"il passo fisso ci mette {rientro(fisso) * OGNI} esempi"
    assert rientro(corto) * OGNI >= 2000, f"il passo corto ci mette solo {rientro(corto) * OGNI} esempi"
    n = len(corto)
    assert media(corto, n - 20, n) < media(fisso, n - 20, n), "alla fine il passo corto non è il più vicino"
    assert max(corto + fisso) < YMAX, "una curva esce dalla fascia"
    # «e in cambio non smette mai di oscillare»: negli ultimi cinquemila esempi
    # il passo fisso si muove da un punto all'altro molto più del passo corto
    salti = lambda v: sum(abs(a - b) for a, b in zip(v[n - 50:], v[n - 49:])) / 49
    assert salti(fisso) > 5 * salti(corto), "il passo fisso non oscilla più dell'altro"

    corpo, anim = [], []
    corpo.append(f'<clipPath id="scorre"><rect id="scorre-r" x="{X0 - 4}" y="0" '
                 f'width="{X1 - X0 + 8}" height="{Y0 + ALT + 10}" '
                 f'style="animation:scorre var(--d) linear infinite"/></clipPath>')
    corpo.append(f'<line class="asse" x1="{X0}" y1="{py(0):.1f}" x2="{X1}" y2="{py(0):.1f}"/>')
    corpo.append(f'<line class="asse" x1="{X0}" y1="{Y0}" x2="{X0}" y2="{py(0):.1f}"/>')
    for d in (1, 2, 3, 4):
        corpo.append(f'<line class="guida" x1="{X0}" y1="{py(d):.1f}" x2="{X1}" y2="{py(d):.1f}"/>')
        corpo.append(f'<text class="lbs" x="{X0 - 8}" y="{py(d) + 4:.1f}" text-anchor="end">{d}</text>')
    xc = X0 + CAMBIO / T * (X1 - X0)
    corpo.append(f'<line class="cambio" x1="{xc:.1f}" y1="{Y0 - 8}" x2="{xc:.1f}" y2="{py(0) + 6:.1f}"/>')
    corpo.append(f'<text class="lbs" x="{xc + 6:.1f}" y="{Y0 + 2}">qui cambia la regola</text>')
    corpo.append(f'<text class="lbs" x="{X1}" y="{py(0) + 22:.1f}" text-anchor="end">'
                 f'{T // 1000} mila esempi</text>')
    corpo.append(f'<text class="lbs" x="{X0}" y="{py(0) + 22:.1f}">0</text>')
    corpo.append(f'<text class="lbs" x="{X0 - 8}" y="{Y0 - 14}" text-anchor="start">'
                 f'distanza dalla regola vera</text>')
    for classe, v in (("fisso", fisso), ("corto", corto)):
        punti = " ".join(f"{px(k):.1f},{py(d):.1f}" for k, d in enumerate(v))
        corpo.append(f'<polyline class="{classe}" points="{punti}" clip-path="url(#scorre)"/>')
    corpo.append(f'<text class="etc" x="{X1 + 8}" y="{py(corto[-1]) + 4:.1f}">passo che</text>'
                 f'<text class="etc" x="{X1 + 8}" y="{py(corto[-1]) + 20:.1f}">si accorcia</text>')
    corpo.append(f'<text class="etf" x="{X1 + 8}" y="{py(fisso[-1]) - 12:.1f}">passo fisso</text>')
    anim.append(keyframes("scorre", [(0.0, "transform:scaleX(0)"), (6.0, "transform:scaleX(0)"),
                                     (90.0, "transform:scaleX(1)"), (100.0, "transform:scaleX(1)")]))

    return Figura(
        larghezza=760, altezza=Y0 + ALT + 40,
        alt="Animazione: due curve avanzano da sinistra a destra, la distanza fra i "
            "pesi di due modelli e quelli della regola vera, esempio dopo esempio. "
            "La curva teal, col passo che si accorcia, scende presto e resta bassa e "
            "liscia; la curva terracotta, col passo fisso, scende altrettanto presto "
            "ma resta più alta e tremolante. A metà, dove una linea verticale segna "
            "il cambio della regola, tutte e due saltano in alto, perché la regola "
            "vera si è spostata: la terracotta torna giù in poche centinaia di "
            "esempi, la teal ridiscende lentamente per migliaia. Alla fine la teal è "
            "di nuovo la più bassa.",
        corpo="".join(corpo),
        stile=f"""    .asse   {{ stroke:{BORDER_STRONG}; stroke-width:1.2; }}
    .guida  {{ stroke:{BORDER}; stroke-width:1; stroke-dasharray:3 4; }}
    .cambio {{ stroke:{FG_MUTED}; stroke-width:1.2; stroke-dasharray:2 4; }}
    .corto  {{ fill:none; stroke:{TEAL}; stroke-width:2.6; stroke-linejoin:round; }}
    .fisso  {{ fill:none; stroke:{TERRACOTTA}; stroke-width:1.8; stroke-linejoin:round; opacity:.85; }}
    .etc    {{ font-family:{SANS}; font-size:14px; font-weight:700; fill:{TEAL}; }}
    .etf    {{ font-family:{SANS}; font-size:14px; font-weight:700; fill:{TERRACOTTA}; }}
    #scorre-r {{ transform-box:fill-box; transform-origin:0% 50%; }}""",
        animazioni=anim,
        durata=9.0,
        fermi="#scorre-r",
    )
