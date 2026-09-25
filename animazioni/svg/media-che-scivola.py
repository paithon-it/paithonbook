"""La regressione a nucleo: una media pesata che scivola lungo l'asse.

È la scena di `MachineLearning/curve-al-posto-di-rette.md`: cento esempi
rumorosi attorno a un'onda, e lo stimatore di Nadaraya-Watson con una campana
gaussiana di larghezza 0,05. La campana scorre da sinistra a destra e dietro di
lei si traccia la stima; il tracciato è sincronizzato con la campana perché lo
scopre un ritaglio che si allarga alla stessa velocità, non un tratteggio che
avanzerebbe con la lunghezza della curva.

Il sorteggio è `numpy.random.default_rng(SEME)`, deterministico. La didascalia
promette tre cose e tre assert le difendono: all'interno la stima segue l'onda
vera, all'estremo sinistro se ne stacca verso l'alto, all'estremo destro verso
il basso. Il seme è scelto perché il sorteggio sia tipico ai due bordi: su
duecento semi lo scarto medio vale circa +0,26 a sinistra e -0,24 a destra
(asintoticamente 0,8 h f' = 0,25), e col seme 0 il bordo sinistro quasi non si
staccava (+0,10), cioè il disegno mostrava meno di quello che il meccanismo fa.
Il disegno fermo è lo stato finale: la curva intera e la campana al bordo
destro.
"""

import numpy as np

from paithon_svg import *

NOME = "media-che-scivola"
TITOLO = "la regressione a nucleo, una media pesata che scivola"

SEME = 1
H = 0.05
N = 100
X0, X1, Y0, ALT = 100, 680, 40, 240     # il riquadro: x in [0, 1], y in [-2, 2]
YMIN, YMAX = -2.0, 2.0


def px(x):
    return X0 + x * (X1 - X0)


def py(y):
    return Y0 + (YMAX - y) / (YMAX - YMIN) * ALT


def dati():
    rng = np.random.default_rng(SEME)
    x = np.sort(rng.random(N))
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, size=N)
    return x, y


def stima(griglia, x, y):
    w = np.exp(-0.5 * ((griglia[:, None] - x[None, :]) / H) ** 2)
    return (w * y).sum(axis=1) / w.sum(axis=1)


def costruisci() -> Figura:
    x, y = dati()
    griglia = np.linspace(0, 1, 241)
    f = stima(griglia, x, y)
    vera = np.sin(2 * np.pi * griglia)
    dentro = (griglia > 0.1) & (griglia < 0.9)
    scarto_dentro = np.max(np.abs(f - vera)[dentro])
    scarto_bordo = f[-1] - vera[-1]
    scarto_sx = f[0] - vera[0]
    assert scarto_dentro < 0.2, "all'interno la stima non segue l'onda"
    # al bordo sinistro i vicini stanno tutti a destra, dove l'onda sale: la
    # stima è tirata verso l'alto
    assert scarto_sx > 0.2 and scarto_sx > 1.4 * scarto_dentro, \
        "al bordo sinistro la stima non si stacca"
    # al bordo la media prende i vicini solo da sinistra, dove l'onda è più bassa:
    # la stima è tirata verso il basso, e di parecchio più che in qualunque punto
    # dell'interno
    assert scarto_bordo < -0.25 and abs(scarto_bordo) > 1.8 * scarto_dentro, \
        "al bordo destro la stima non si stacca"

    assert np.all((YMIN < y) & (y < YMAX)), "un punto esce dal riquadro"

    corpo, anim = [], []
    durata = 7.0
    corpo.append(f'<clipPath id="scopre"><rect id="scopre-r" x="{X0 - 2}" y="{Y0 - 10}" '
                 f'width="{X1 - X0 + 4}" height="{ALT + 20}" '
                 f'style="animation:scopri var(--d) infinite"/></clipPath>')
    # l'asse e l'onda vera
    corpo.append(f'<line class="asse" x1="{X0}" y1="{py(0):.1f}" x2="{X1}" y2="{py(0):.1f}"/>')
    onda = " ".join(f"{px(g):.1f},{py(v):.1f}" for g, v in zip(griglia, vera))
    corpo.append(f'<polyline class="vera" points="{onda}"/>')
    # i punti
    for xi, yi in zip(x, y):
        corpo.append(f'<circle class="punto" cx="{px(xi):.1f}" cy="{py(yi):.1f}" r="3.4"/>')
    # la stima, scoperta dal ritaglio
    curva = " ".join(f"{px(g):.1f},{py(v):.1f}" for g, v in zip(griglia, f))
    corpo.append(f'<polyline class="stima" points="{curva}" clip-path="url(#scopre)"/>')
    # la campana, disegnata al riposo sul bordo destro
    xs = np.linspace(-3.2 * H, 3.2 * H, 61)
    campana = " ".join(f"{px(1 + u):.1f},{py(-1.95 + 0.9 * np.exp(-0.5 * (u / H) ** 2)):.1f}" for u in xs)
    corpo.append(f'<polyline class="campana" points="{campana}" '
                 f'style="animation:scorri var(--d) infinite"/>')
    corpo.append(f'<text class="lbs" x="{X0}" y="{Y0 + ALT + 34}">in ocra la campana dei pesi, '
                 f'in terracotta la media che disegna, tratteggiata l’onda vera</text>')

    larg = X1 - X0
    anim.append(keyframes("scorri", [
        (0.0, f"transform:translateX({-larg}px)"), (8.0, f"transform:translateX({-larg}px)"),
        (88.0, "transform:translateX(0px)"), (100.0, "transform:translateX(0px)")]))
    anim.append(keyframes("scopri", [
        (0.0, "transform:scaleX(0)"), (8.0, "transform:scaleX(0)"),
        (88.0, "transform:scaleX(1)"), (100.0, "transform:scaleX(1)")]))

    return Figura(
        larghezza=780, altezza=Y0 + ALT + 50,
        alt="Animazione: cento punti sparsi attorno a un'onda sinusoidale tratteggiata. "
            "Una campana ocra scorre da sinistra a destra lungo l'asse orizzontale, e "
            "mentre avanza si traccia dietro di lei una curva terracotta, la media dei "
            "punti pesata dalla campana. La curva segue l'onda all'interno e se ne "
            "stacca ai due estremi, dove la campana ha punti da una parte sola: "
            "all'estremo sinistro resta sopra l'onda, all'estremo destro sotto.",
        corpo="".join(corpo),
        stile=f"""    .asse    {{ stroke:{BORDER_STRONG}; stroke-width:1.2; }}
    .vera    {{ fill:none; stroke:{INK}; stroke-width:1.6; stroke-dasharray:5 5; opacity:.7; }}
    .punto   {{ fill:{TEAL}; opacity:.75; }}
    .stima   {{ fill:none; stroke:{TERRACOTTA}; stroke-width:3; stroke-linejoin:round; }}
    .campana {{ fill:none; stroke:{OCRA}; stroke-width:3; }}
    #scopre-r {{ transform-box:fill-box; transform-origin:0% 50%; }}""",
        animazioni=anim,
        durata=durata,
        fermi=".campana, #scopre-r",
    )
