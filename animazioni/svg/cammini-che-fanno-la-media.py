"""La formula di Feynman-Kac per l'equazione del calore: la soluzione in un punto
è la media della condizione iniziale nei punti in cui arrivano cammini casuali.

È la scena di `PINN/equazioni-come-medie.md`. L'equazione è quella della
sbarra del capitolo, du/dt = alfa d2u/dx2, con alfa = 1/4, la condizione iniziale
phi(y) = y^2 e il punto x0 = 0,3 al tempo t = 1. La soluzione esatta è
u(x0, t) = x0^2 + 2 alfa t, cioè 0,59.

A sinistra partono da x0 i cammini, a gruppi di VENTI, e ciascuno è una
passeggiata casuale a `PASSI` passi con incrementi gaussiani di varianza
2 alfa dt: al tempo t la posizione ha la legge di x0 + radice(2 alfa t) Z. A
destra la media di phi nei punti di arrivo, dopo n cammini, con la retta
tratteggiata del valore esatto e la fascia di due deviazioni standard della
media, larga come 1/radice(n): la stima ci entra e ci resta, e la fascia si
stringe.

Il sorteggio è `random.Random(SEME)` e i conti sono in Python puro: lo stesso
SVG su ogni macchina. Il seme è il primo provato, non uno scelto fra tanti: la
fascia a due deviazioni standard contiene la media con probabilità di circa 95
su cento a ogni gruppo, e l'assert lo pretende a tutti e dieci: sui semi da 0 a
49 lo passano 43. Gli assert difendono quello che la didascalia promette:
la media finale sta nella fascia, e la fascia finale è meno di un terzo di
quella iniziale.
"""

import math
import random

from paithon_svg import *

NOME = "cammini-che-fanno-la-media"
TITOLO = "la soluzione dell'equazione del calore come media sui cammini casuali"

SEME = 4
ALFA, T, X0 = 0.25, 1.0, 0.3
PASSI = 24
GRUPPI, VENTI = 10, 20
ESATTA = X0 ** 2 + 2 * ALFA * T
VAR_PHI = 4 * X0 ** 2 * (2 * ALFA * T) + 2 * (2 * ALFA * T) ** 2   # varianza di phi(X0 + sqrt(2 alfa t) Z)

# pannello dei cammini
PX0, PX1, PY0, PALT = 70, 380, 40, 260
YMIN, YMAX = -2.6, 3.2
# pannello della media
MX0, MX1 = 470, 780
MMIN, MMAX = 0.0, 1.4


def cammini():
    rnd = random.Random(SEME)
    dt = T / PASSI
    tutti = []
    for _ in range(GRUPPI * VENTI):
        y, traccia = X0, [X0]
        for _ in range(PASSI):
            y += rnd.gauss(0.0, math.sqrt(2 * ALFA * dt))
            traccia.append(y)
        tutti.append(traccia)
    return tutti


def px(k):
    return PX0 + k / PASSI * (PX1 - PX0)


def py(y):
    return PY0 + (YMAX - y) / (YMAX - YMIN) * PALT


def mx(n):
    return MX0 + (n / (GRUPPI * VENTI)) * (MX1 - MX0)


def my(v):
    return PY0 + (MMAX - v) / (MMAX - MMIN) * PALT


def costruisci() -> Figura:
    tutti = cammini()
    finali = [c[-1] for c in tutti]
    assert all(YMIN < y < YMAX for c in tutti for y in c), "un cammino esce dal pannello"
    medie = []
    for g in range(1, GRUPPI + 1):
        n = g * VENTI
        medie.append((n, sum(y * y for y in finali[:n]) / n))
    fascia = lambda n: 2 * math.sqrt(VAR_PHI / n)
    assert all(abs(m - ESATTA) < fascia(n) for n, m in medie), "la media esce dalla fascia"
    assert fascia(GRUPPI * VENTI) < fascia(VENTI) / 3, "la fascia non si stringe abbastanza"
    assert all(MMIN < ESATTA - fascia(n) and ESATTA + fascia(n) < MMAX for n, _ in medie)

    n_tempi = GRUPPI
    corpo, anim = [], []

    def dal_gruppo(g):
        """Visibile dal gruppo g in poi; a riposo visibile."""
        if g == 0:
            return ""
        t0, _ = sosta(g, n_tempi)
        nome = f"g{g}"
        anim.append(keyframes(nome, [(0.0, "opacity:0"), (t0 - 0.8, "opacity:0"),
                                     (t0 + 0.8, "opacity:1"), (100.0, "opacity:1")]))
        return f' style="animation:{nome} var(--d) infinite"'

    def solo_nel_gruppo(g):
        """Visibile solo durante il gruppo g; a riposo visibile l'ultimo."""
        t0, _ = sosta(g, n_tempi)
        t1 = t0 + 100.0 / n_tempi
        nome = f"s{g}"
        if g == n_tempi - 1:
            tappe = [(0.0, "opacity:0"), (t0 - 0.4, "opacity:0"), (t0, "opacity:1"), (100.0, "opacity:1")]
        elif g == 0:
            tappe = [(0.0, "opacity:1"), (t1 - 0.4, "opacity:1"), (t1, "opacity:0"), (100.0, "opacity:0")]
        else:
            tappe = [(0.0, "opacity:0"), (t0 - 0.4, "opacity:0"), (t0, "opacity:1"),
                     (t1 - 0.4, "opacity:1"), (t1, "opacity:0"), (100.0, "opacity:0")]
        anim.append(keyframes(nome, tappe))
        fermo = "" if g == n_tempi - 1 else ";opacity:0"
        return f' style="animation:{nome} var(--d) infinite{fermo}"'

    # pannello dei cammini: assi e il punto di partenza
    corpo.append(f'<line class="asse" x1="{PX0}" y1="{PY0}" x2="{PX0}" y2="{PY0 + PALT}"/>')
    corpo.append(f'<line class="asse" x1="{PX1}" y1="{PY0}" x2="{PX1}" y2="{PY0 + PALT}"/>')
    corpo.append(f'<text class="lbs" x="{PX0}" y="{PY0 + PALT + 20}" text-anchor="middle">0</text>')
    corpo.append(f'<text class="lbs" x="{PX1}" y="{PY0 + PALT + 20}" text-anchor="middle">t</text>')
    corpo.append(f'<text class="lbs" x="{(PX0 + PX1) / 2:.0f}" y="{PY0 - 14}" text-anchor="middle">'
                 f'cammini casuali da x₀</text>')
    for g in range(GRUPPI):
        linee = "".join(
            f'<polyline class="cam" points="{" ".join(f"{px(k):.1f},{py(y):.1f}" for k, y in enumerate(c))}"/>'
            for c in tutti[g * VENTI:(g + 1) * VENTI])
        punti = "".join(f'<circle class="fine" cx="{PX1:.1f}" cy="{py(c[-1]):.1f}" r="2.4"/>'
                        for c in tutti[g * VENTI:(g + 1) * VENTI])
        corpo.append(f'<g{dal_gruppo(g)}>{linee}{punti}</g>')
    corpo.append(f'<circle class="x0" cx="{PX0}" cy="{py(X0):.1f}" r="5"/>')

    # pannello della media: fascia, valore esatto, stima
    alto = " ".join(f"{mx(n):.1f},{my(ESATTA + fascia(n)):.1f}" for n, _ in medie)
    basso = " ".join(f"{mx(n):.1f},{my(ESATTA - fascia(n)):.1f}" for n, _ in reversed(medie))
    corpo.append(f'<polygon class="fascia" points="{alto} {basso}"/>')
    corpo.append(f'<line class="esatta" x1="{MX0}" y1="{my(ESATTA):.1f}" x2="{MX1}" y2="{my(ESATTA):.1f}"/>')
    corpo.append(f'<text class="lbs" x="{MX1 + 6}" y="{my(ESATTA) + 4:.1f}">u(x₀, t)</text>')
    corpo.append(f'<line class="asse" x1="{MX0}" y1="{my(MMIN):.1f}" x2="{MX1}" y2="{my(MMIN):.1f}"/>')
    corpo.append(f'<text class="lbs" x="{(MX0 + MX1) / 2:.0f}" y="{PY0 - 14}" text-anchor="middle">'
                 f'media di φ nei punti di arrivo</text>')
    for g, (n, m) in enumerate(medie):
        if g > 0:
            n0, m0 = medie[g - 1]
            corpo.append(f'<line class="stima" x1="{mx(n0):.1f}" y1="{my(m0):.1f}" x2="{mx(n):.1f}" '
                         f'y2="{my(m):.1f}"{dal_gruppo(g)}/>')
        corpo.append(f'<circle class="pstima" cx="{mx(n):.1f}" cy="{my(m):.1f}" r="4"{dal_gruppo(g)}/>')
        testo = f"{n} cammini: media {m:.3f}".replace(".", ",")
        corpo.append(f'<text class="cont" x="{MX0}" y="{my(MMIN) + 20:.1f}"{solo_nel_gruppo(g)}>{testo}</text>')

    return Figura(
        larghezza=860, altezza=PY0 + PALT + 36,
        alt="Animazione in due pannelli. A sinistra, da un punto sull'asse verticale "
            "partono gruppi di venti cammini casuali teal che si aprono a ventaglio "
            "fino al tempo t, dove ciascuno lascia un punto. A destra una retta "
            "tratteggiata segna il valore esatto della soluzione, attorno a lei una "
            "fascia ocra si stringe man mano che i cammini aumentano, e una "
            "spezzata terracotta segue la media della condizione iniziale nei punti "
            f"di arrivo: dopo {GRUPPI * VENTI} cammini sta dentro la fascia, vicino "
            "al valore esatto.",
        corpo="".join(corpo),
        stile=f"""    .asse   {{ stroke:{BORDER_STRONG}; stroke-width:1.2; }}
    .cam    {{ fill:none; stroke:{TEAL}; stroke-width:1; opacity:.35; }}
    .fine   {{ fill:{TEAL}; opacity:.7; }}
    .x0     {{ fill:{TERRACOTTA}; }}
    .fascia {{ fill:{OCRA}; opacity:.35; }}
    .esatta {{ stroke:{INK}; stroke-width:1.4; stroke-dasharray:6 5; }}
    .stima  {{ stroke:{TERRACOTTA}; stroke-width:2.4; }}
    .pstima {{ fill:{TERRACOTTA}; }}
    .cont   {{ font-family:{SANS}; font-size:14px; font-weight:700; fill:{TERRACOTTA}; }}""",
        animazioni=anim,
        durata=GRUPPI * 1.2,
        fermi="g, .stima, .pstima, .cont",
    )
