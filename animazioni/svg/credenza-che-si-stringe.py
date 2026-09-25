"""La credenza su una moneta che si stringe lancio dopo lancio.

È la scena di `Matematica/probabilita-statistica.md`, nella stima bayesiana:
si parte da una Beta(2, 2) sulla probabilità p di testa (un'opinione larga,
che per la media vale come quattro lanci immaginati, due teste e due croci), e
a ogni tappa si aggiungono
i lanci di una moneta con p = 0,7. La distribuzione a posteriori è
Beta(2 + teste, 2 + croci), e la figura la disegna a sei tappe: 0, 1, 3, 10,
30 e 100 lanci. La curva di ogni tappa si accende, resta un momento e scende
a traccia sbiadita; l'ultima resta piena.

Il sorteggio è `numpy.random.default_rng(SEME)`, le densità si calcolano con
`math.lgamma`, quindi il disegno non dipende da nessuna libreria di algebra
lineare. La didascalia promette tre cose e gli assert le difendono: la curva
si stringe a ogni tappa (la deviazione standard scende sempre), alla fine sta
attorno al valore vero (la media a posteriori a meno di 0,05 da 0,7, e 0,7
dentro l'intervallo centrale al 95%), e all'inizio è larga (la Beta(2, 2) dà
più di metà della probabilità fuori dall'intervallo fra 0,4 e 0,6). Il
disegno fermo è lo stato finale: tutte le tappe come tracce, l'ultima piena,
la sua etichetta accesa.
"""

import math

import numpy as np

from paithon_svg import *

NOME = "credenza-che-si-stringe"
TITOLO = "la distribuzione a posteriori si stringe attorno al valore vero"

SEME = 3
P_VERO = 0.7
A0, B0 = 2, 2
TAPPE = [0, 1, 3, 10, 30, 100]

X0, X1 = 80, 700
Y0, Y1 = 330, 40           # base e cima del grafico
DMAX = 10.0                # densità in cima


def lanci():
    rng = np.random.default_rng(SEME)
    return (rng.random(max(TAPPE)) < P_VERO).astype(int)


def densita(p, a, b):
    if p <= 0 or p >= 1:
        return 0.0
    return math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                    + (a - 1) * math.log(p) + (b - 1) * math.log(1 - p))


def quantile(a, b, q, passi=20000):
    """Quantile della Beta per somma cumulata su una griglia fine (basta a due decimali)."""
    tot, cum = 0.0, []
    for i in range(1, passi):
        tot += densita(i / passi, a, b) / passi
        cum.append(tot)
    for i, c in enumerate(cum):
        if c / tot >= q:
            return (i + 1) / passi
    return 1.0


def gx(p):
    return X0 + p * (X1 - X0)


def gy(d):
    return Y0 + d / DMAX * (Y1 - Y0)


def costruisci() -> Figura:
    x = lanci()
    stati = []
    for n in TAPPE:
        t = int(x[:n].sum())
        a, b = A0 + t, B0 + n - t
        media = a / (a + b)
        sd = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
        stati.append((n, t, a, b, media, sd))
    sd = [s[5] for s in stati]
    assert all(s2 < s1 for s1, s2 in zip(sd, sd[1:])), "la curva non si stringe a ogni tappa"
    n, t, a, b, media, _ = stati[-1]
    assert abs(media - P_VERO) < 0.05, f"media finale {media:.3f} lontana da {P_VERO}"
    lo, hi = quantile(a, b, 0.025), quantile(a, b, 0.975)
    assert lo < P_VERO < hi, "il valore vero è fuori dall'intervallo al 95%"
    dentro = quantile(A0, B0, 0.5)  # simmetrica: la mediana è 0,5
    assert abs(dentro - 0.5) < 1e-3
    massa_centrale = sum(densita(i / 10000, A0, B0) for i in range(4000, 6001)) / 10000
    assert massa_centrale < 0.5, "la credenza di partenza non è larga"
    picchi = [max(densita(i / 1000, s[2], s[3]) for i in range(1, 1000)) for s in stati]
    assert max(picchi) < DMAX * 0.95, "una curva esce dal grafico"

    corpo, anim = [], []
    corpo.append(f'<line class="asse" x1="{X0}" y1="{Y0}" x2="{X1}" y2="{Y0}"/>')
    for v in (0, 0.5, 1):
        corpo.append(f'<text class="tick" x="{gx(v):.1f}" y="{Y0 + 18}" text-anchor="middle">'
                     f'{str(v).replace(".", ",")}</text>')
    corpo.append(f'<text class="lbs" x="{(X0 + X1) / 2:.1f}" y="{Y0 + 38}" text-anchor="middle">'
                 f'probabilità di testa</text>')
    corpo.append(f'<line class="vero" x1="{gx(P_VERO):.1f}" y1="{Y0}" x2="{gx(P_VERO):.1f}" y2="{Y1 - 6}"/>')
    corpo.append(f'<text class="lbs" x="{gx(P_VERO) + 6:.1f}" y="{Y1 + 4}">valore vero 0,7</text>')

    k = len(stati)
    passo = 84.0 / k                     # le tappe occupano l'84% del ciclo, poi sosta
    for i, (n, t, a, b, media, _) in enumerate(stati):
        punti = " ".join(f"{gx(j / 400):.1f},{gy(densita(j / 400, a, b)):.1f}" for j in range(1, 400))
        ultima = i == k - 1
        classe = "curva ultima" if ultima else "curva"
        inizio, fine = 4 + i * passo, 4 + (i + 1) * passo
        if ultima:
            fotogrammi = [(0.0, "opacity:0"), (inizio - 0.1, "opacity:0"), (inizio + 2, "opacity:1"),
                          (100.0, "opacity:1")]
        else:
            fotogrammi = [(0.0, "opacity:0"), (inizio - 0.1, "opacity:0"), (inizio + 2, "opacity:1"),
                          (fine - 1, "opacity:1"), (fine + 2, "opacity:.22"), (100.0, "opacity:.22")]
        anim.append(keyframes(f"tappa{i}", fotogrammi))
        # lo stato di riposo sta sull'elemento animato, non su un gruppo attorno:
        # l'animazione lo scavalca mentre gira, e un'opacità sul gruppo invece si
        # moltiplicherebbe con la sua, spegnendo la curva attiva
        riposo = "" if ultima else "opacity:.22;"
        corpo.append(f'<polyline class="{classe}" points="{punti}" '
                     f'style="{riposo}animation:tappa{i} var(--d) infinite"/>')
        # l'etichetta della tappa: accesa solo nel suo intervallo, l'ultima resta
        testo = (f"prima dei lanci: Beta(2, 2)" if n == 0 else
                 f"dopo {n} {'lancio' if n == 1 else 'lanci'}, {t} {'testa' if t == 1 else 'teste'}")
        if ultima:
            fl = [(0.0, "opacity:0"), (inizio - 0.1, "opacity:0"), (inizio + 2, "opacity:1"), (100.0, "opacity:1")]
            riposo = ""
        else:
            fl = [(0.0, "opacity:0"), (inizio - 0.1, "opacity:0"), (inizio + 1, "opacity:1"),
                  (fine - 0.5, "opacity:1"), (fine + 0.5, "opacity:0"), (100.0, "opacity:0")]
            riposo = "opacity:0;"
        anim.append(keyframes(f"scritta{i}", fl))
        corpo.append(f'<text class="scritta" x="{X0 + 8}" y="{Y1 + 4}" '
                     f'style="{riposo}animation:scritta{i} var(--d) infinite">{testo}</text>')

    return Figura(
        larghezza=780, altezza=390,
        alt="Animazione su un asse orizzontale che va da 0 a 1, la probabilità di "
            "testa, con una linea verticale tratteggiata sul valore vero 0,7. Una "
            "curva terracotta bassa e larga, la credenza prima dei lanci, lascia il "
            "posto tappa dopo tappa a curve sempre più alte e strette, dopo 1, 3, 10, "
            "30 e 100 lanci; ognuna resta come traccia sbiadita, e l'ultima, piena, "
            "sta stretta attorno al valore vero. Una scritta in alto a sinistra dice "
            "quanti lanci e quante teste si sono visti.",
        corpo="".join(corpo),
        stile=f"""    .asse    {{ stroke:{BORDER_STRONG}; stroke-width:1.2; }}
    .vero    {{ stroke:{TEAL}; stroke-width:1.6; stroke-dasharray:6 5; }}
    .curva   {{ fill:none; stroke:{TERRACOTTA}; stroke-width:2.2; stroke-linejoin:round; }}
    .ultima  {{ stroke-width:3; }}
    .tick    {{ font-family:{SANS}; font-size:12px; fill:{FG_MUTED}; }}
    .scritta {{ font-family:{SANS}; font-size:15px; font-weight:700; fill:{INK}; }}""",
        animazioni=anim,
        durata=9.0,
        fermi=".curva, .scritta",
    )
