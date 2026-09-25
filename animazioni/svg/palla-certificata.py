"""Randomized smoothing: il voto sotto rumore e la palla entro cui non cambia.

Figura ferma. Attorno all'ingresso x si sorteggiano copie sporcate di rumore
gaussiano; il classificatore f le vota, e la classe più votata è la risposta
del classificatore lisciato g. Se la classe A prende la frazione p_A dei voti
(e le altre il resto), la risposta di g non cambia dentro la palla di raggio
R = sigma/2 (Phi^-1(p_A) - Phi^-1(p_B)).

Con un confine lineare e due classi il conto si chiude, ed è il controllo che
gli `assert` fanno: p_A = Phi(d/sigma), dove d è la distanza di x dal
confine, quindi R = d. La palla tocca il confine e non lo passa: la garanzia,
qui, è stretta. I voti disegnati sono un sorteggio vero (seme fisso), e la
loro frazione deve ritrovare p_A.

Quel raggio però viene dalla probabilità esatta, che nessuno conosce: chi
certifica ha solo i voti. Con i 400 voti e la procedura di Cohen e colleghi
(il limite inferiore di Clopper-Pearson su p_A, errore un millesimo) il raggio
garantito è più piccolo, e la figura lo disegna tratteggiato; l'`assert`
controlla che resti dentro la distanza dal confine, che è ciò che la
didascalia promette. La stima ingenua dalla sola frazione dei voti, invece,
qui supererebbe il confine.
"""

import math
import random
from statistics import NormalDist

from paithon_svg import *

NOME = "palla-certificata"
TITOLO = "il raggio certificato del randomized smoothing"

SIGMA = 0.5
X = (0.0, 0.0)
W, B = (1.0, 0.6), -0.55             # confine: w . z + b = 0, classe A dove < 0
CAMPIONI, SEME = 400, 0

_nw = math.hypot(*W)
D = abs(W[0] * X[0] + W[1] * X[1] + B) / _nw          # distanza dal confine
P_A = NormalDist().cdf(D / SIGMA)
P_B = 1 - P_A
R = SIGMA / 2 * (NormalDist().inv_cdf(P_A) - NormalDist().inv_cdf(P_B))
if abs(R - D) > 1e-9:
    raise AssertionError(f"il raggio {R:.4f} non coincide con la distanza {D:.4f}")

_rng = random.Random(SEME)
PUNTI = [(X[0] + _rng.gauss(0, SIGMA), X[1] + _rng.gauss(0, SIGMA))
         for _ in range(CAMPIONI)]
CLASSE_A = [W[0] * px + W[1] * py + B < 0 for px, py in PUNTI]
FRAZIONE = sum(CLASSE_A) / CAMPIONI
if abs(FRAZIONE - P_A) > 0.05:
    raise AssertionError(f"i voti sorteggiati danno {FRAZIONE:.3f}, non {P_A:.3f}")

# il raggio garantito dai soli voti: limite inferiore di Clopper-Pearson su p_A
ALFA = 0.001
_k = sum(CLASSE_A)


def _coda(p: float) -> float:
    """P(almeno _k voti per A su CAMPIONI, se ciascuno va ad A con prob. p)."""
    return sum(math.comb(CAMPIONI, j) * p ** j * (1 - p) ** (CAMPIONI - j)
               for j in range(_k, CAMPIONI + 1))


_lo, _hi = 0.0, _k / CAMPIONI
for _ in range(100):
    _m = (_lo + _hi) / 2
    _lo, _hi = (_m, _hi) if _coda(_m) < ALFA else (_lo, _m)
P_A_BASSA = (_lo + _hi) / 2
R_VOTI = SIGMA * NormalDist().inv_cdf(P_A_BASSA)
if not 0 < R_VOTI < D:
    raise AssertionError(f"il raggio garantito dai voti {R_VOTI:.4f} "
                         f"non sta dentro la distanza {D:.4f}")


def costruisci() -> Figura:
    rq = Riquadro(x=40, y=30, larg=380, alt=380,
                  xmin=-1.6, xmax=1.6, ymin=-1.6, ymax=1.6)
    c = []
    # il confine del classificatore f, dentro il riquadro
    # gli estremi sono dove la retta incontra i bordi del riquadro
    estremi = []
    for x in (rq.xmin, rq.xmax):
        y = -(W[0] * x + B) / W[1]
        if rq.ymin <= y <= rq.ymax:
            estremi.append((x, y))
    for y in (rq.ymin, rq.ymax):
        x = -(W[1] * y + B) / W[0]
        if rq.xmin <= x <= rq.xmax:
            estremi.append((x, y))
    (xa, ya), (xb, yb) = estremi[:2]
    c.append(f'<line class="confine" x1="{rq.sx(xa):.1f}" y1="{rq.sy(ya):.1f}" '
             f'x2="{rq.sx(xb):.1f}" y2="{rq.sy(yb):.1f}"/>')
    c.append(f'<rect class="cornice" x="{rq.x:.0f}" y="{rq.y:.0f}" '
             f'width="{rq.larg:.0f}" height="{rq.alt:.0f}"/>')
    for (px, py), a in zip(PUNTI, CLASSE_A):
        if rq.xmin < px < rq.xmax and rq.ymin < py < rq.ymax:
            c.append(f'<circle class="{"va" if a else "vb"}" cx="{rq.sx(px):.1f}" '
                     f'cy="{rq.sy(py):.1f}" r="2.6"/>')
    scala = rq.larg / (rq.xmax - rq.xmin)
    c.append(f'<circle class="palla" cx="{rq.sx(X[0]):.1f}" cy="{rq.sy(X[1]):.1f}" '
             f'r="{R * scala:.1f}"/>')
    c.append(f'<circle class="garantita" cx="{rq.sx(X[0]):.1f}" '
             f'cy="{rq.sy(X[1]):.1f}" r="{R_VOTI * scala:.1f}"/>')
    c.append(f'<circle class="x" cx="{rq.sx(X[0]):.1f}" cy="{rq.sy(X[1]):.1f}" r="5"/>')
    tx = rq.x + rq.larg + 30
    virgola = lambda v, k: f"{v:.{k}f}".replace(".", ",")
    righe = [("lbl", "il voto sotto rumore"),
             ("lbs", f"il punto nero è l’ingresso x; σ = {virgola(SIGMA, 1)}"),
             ("lbs", f"{CAMPIONI} copie di x sporcate di rumore"),
             ("lbs", f"classe A: {virgola(FRAZIONE, 2)} dei voti"),
             ("lbs", f"(la probabilità esatta è {virgola(P_A, 2)})"),
             ("lbl", "con la probabilità esatta"),
             ("lbs", f"R = σ/2 · (Φ⁻¹(p_A) − Φ⁻¹(p_B)) = {virgola(R, 2)}"),
             ("lbs", "tocca il confine e non lo passa:"),
             ("lbs", "con un confine dritto la garanzia è stretta"),
             ("lbl", "dai soli voti (tratteggiato)"),
             ("lbs", f"con errore al più {virgola(ALFA, 3)}: "
                     f"R = {virgola(R_VOTI, 2)}")]
    for k, (cls, testo) in enumerate(righe):
        c.append(f'<text class="{cls}" x="{tx:.0f}" y="{rq.y + 40 + 24 * k:.0f}">'
                 f'{testo}</text>')
    return Figura(
        larghezza=780, altezza=rq.y + rq.alt + 30,
        alt=f"Un piano con un confine diritto fra due classi e un punto x. "
            f"Attorno a x, {CAMPIONI} copie sporcate di rumore gaussiano, colorate "
            f"secondo la classe che il classificatore dà loro: la classe di x "
            f"prende {virgola(FRAZIONE, 2)} dei voti. Un cerchio attorno a x, "
            f"di raggio {virgola(R, 2)}, è il raggio che dà la probabilità "
            f"esatta del voto: tocca il confine senza superarlo, perché con un "
            f"confine diritto quel raggio è esattamente la distanza dal confine. "
            f"Un cerchio tratteggiato più piccolo, di raggio "
            f"{virgola(R_VOTI, 2)}, è il raggio garantito dai soli "
            f"{CAMPIONI} voti con una probabilità di errore di un millesimo.",
        corpo="".join(c),
        stile=f"""    .confine {{ stroke:{INK}; stroke-width:2; }}
    .cornice {{ fill:none; stroke:{BORDER_STRONG}; stroke-width:1.5; }}
    .va {{ fill:{TEAL}; opacity:.55; }}
    .vb {{ fill:{TERRACOTTA}; opacity:.7; }}
    .palla {{ fill:none; stroke:{OCRA}; stroke-width:3; }}
    .garantita {{ fill:none; stroke:{OCRA}; stroke-width:2; stroke-dasharray:6 4; }}
    .x {{ fill:{INK}; }}""",
    )
