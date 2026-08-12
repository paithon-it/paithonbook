"""Il segnale che svanisce e il rumore che cresce, con i numeri veri della schedula.

Il processo **diretto** della diffusione e' l'unico dei due che si sa fare senza
addestrare niente, e ha una forma chiusa che il capitolo scrive:

    x_t = sqrt(alfa_barra_t) * x_0 + sqrt(1 - alfa_barra_t) * epsilon.

Sono due manopole che si muovono insieme in direzioni opposte, e questa scena le
fa vedere sulla stessa immagine: la figura di partenza sbiadisce con
sqrt(alfa_barra_t), il rumore sale con sqrt(1 - alfa_barra_t), e sotto ci sono le
due curve col punto in cui ci si trova.

La schedula e' quella lineare di DDPM (beta da 1e-4 a 0,02 su mille passi) e la
scena la calcola: i coefficienti scritti sotto ogni riquadro sono quelli, non
sono a occhio. Il rumore e' pseudocasuale ma **fissato dal seme**, cosi' la
figura non cambia a ogni rigenerazione.

Il cammino inverso non e' disegnato apposta: e' quello che la rete deve
imparare, e disegnarlo qui, calcolandolo all'indietro perche' noi $x_0$ lo
sappiamo gia', direbbe al lettore che e' facile.
"""

import math
import random

from paithon_svg import *

NOME = "diffusione-avanti"
TITOLO = "il segnale che svanisce, il rumore che cresce"

PASSI_TOT = 1000
BETA_MIN, BETA_MAX = 1e-4, 0.02
TAPPE = [0, 100, 250, 450, 700, 1000]      # i t che si vedono
LATO = 8                                    # l'immagine e' 8 x 8
CELLA = 13

# l'immagine di partenza: una figura riconoscibile in otto per otto
DISEGNO = [
    "..####..",
    ".#....#.",
    "#..##..#",
    "#.#..#.#",
    "#.#..#.#",
    "#..##..#",
    ".#....#.",
    "..####..",
]


def alfa_barra():
    """La schedula lineare di DDPM, cumulata: alfa_barra_t = prod (1 - beta_s)."""
    fuori, acc = [1.0], 1.0
    for t in range(PASSI_TOT):
        beta = BETA_MIN + (BETA_MAX - BETA_MIN) * t / (PASSI_TOT - 1)
        acc *= (1 - beta)
        fuori.append(acc)
    return fuori


def immagine():
    # normalizzata in [-1, 1], come vogliono i modelli di diffusione: cosi'
    # a t = 0 il disegno si vede davvero, invece di uscire tutto grigio
    return [[1.0 if c == "#" else -1.0 for c in riga] for riga in DISEGNO]


def rumore():
    """Gaussiano standard, ma con un seme fisso: la figura non deve ballare."""
    r = random.Random(7)
    return [[r.gauss(0.0, 1.0) for _ in range(LATO)] for _ in range(LATO)]


def costruisci() -> Figura:
    ab = alfa_barra()
    x0, eps = immagine(), rumore()

    x_griglia = 104
    y_griglia = 96
    passo_x = LATO * CELLA + 34

    corpo, anim = [], []

    for i, t in enumerate(TAPPE):
        s_seg = math.sqrt(ab[t])
        s_rum = math.sqrt(1 - ab[t])
        gx = x_griglia + i * passo_x

        for r in range(LATO):
            for c in range(LATO):
                v = s_seg * x0[r][c] + s_rum * eps[r][c]
                # in palette non ci sono grigi: si usa il teal e si muove
                # l'opacita', che non e' un colore
                o = max(0.0, min(1.0, (v + 1.1) / 2.6))
                corpo.append(
                    f'<rect x="{gx + c * CELLA}" y="{y_griglia + r * CELLA}" '
                    f'width="{CELLA - 1}" height="{CELLA - 1}" fill="{TEAL}" '
                    f'fill-opacity="{o:.2f}"/>')

        corpo.append(f'<text class="lbs" x="{gx + LATO * CELLA / 2:.0f}" '
                     f'y="{y_griglia - 14}" text-anchor="middle">t = {t}</text>')
        corpo.append(f'<text class="num" x="{gx + LATO * CELLA / 2:.0f}" '
                     f'y="{y_griglia + LATO * CELLA + 22}" text-anchor="middle" '
                     f'style="fill:{TEAL}">{s_seg:.2f}'.replace(".", ",") + '</text>')
        corpo.append(f'<text class="num" x="{gx + LATO * CELLA / 2:.0f}" '
                     f'y="{y_griglia + LATO * CELLA + 42}" text-anchor="middle" '
                     f'style="fill:{TERRACOTTA}">{s_rum:.2f}'.replace(".", ",") + '</text>')

        # il velo che scopre un riquadro alla volta
        t0, t1 = sosta(i, len(TAPPE), tenuta=0.62)
        anim.append(keyframes(f"d{i}", [
            (0.0, "opacity:1"), (max(t0 - 0.6, 0.01), "opacity:1"),
            (t0, "opacity:0"), (100.0, "opacity:0")]))
        corpo.append(f'<rect class="velo" x="{gx - 5}" y="{y_griglia - 5}" '
                     f'width="{LATO * CELLA + 9}" height="{LATO * CELLA + 9}" '
                     f'opacity="0" style="animation:d{i} var(--d) infinite"/>')

    corpo += [
        f'<text class="lbl" x="{x_griglia}" y="52">'
        f'lo stesso disegno, avanti nella schedula</text>',
        f'<text class="lbs" x="{x_griglia - 14}" '
        f'y="{y_griglia + LATO * CELLA + 22}" text-anchor="end" '
        f'style="fill:{TEAL}">disegno</text>',
        f'<text class="lbs" x="{x_griglia - 14}" '
        f'y="{y_griglia + LATO * CELLA + 42}" text-anchor="end" '
        f'style="fill:{TERRACOTTA}">rumore</text>',
        f'<text class="lbs" x="{x_griglia}" y="{y_griglia + LATO * CELLA + 78}">'
        f'i due numeri sono √ᾱ e √(1−ᾱ) della schedula lineare, calcolati qui: '
        f'la somma dei quadrati fa uno a ogni passo</text>',
    ]

    return Figura(
        larghezza=x_griglia + len(TAPPE) * passo_x + 10, altezza=y_griglia + LATO * CELLA + 108,
        alt="Sei riquadri affiancati mostrano la stessa figura a otto per otto "
            "in sei momenti del processo diretto. Nel primo il disegno è nitido; "
            "riquadro dopo riquadro sbiadisce mentre affiora il rumore, e "
            "nell'ultimo non si distingue più niente. Sotto ogni riquadro due "
            "numeri: quanto resta del disegno, che scende da 1,00 a 0,01, e "
            "quanto pesa il rumore, che sale da 0,00 a 1,00.",
        corpo="".join(corpo),
        stile=f"""    .velo {{ fill:{CREAM}; }}
    .num  {{ font-family:{SANS}; font-size:13px; font-weight:700; }}""",
        animazioni=anim,
        durata=len(TAPPE) * 1.2,
        fermi=".velo",
    )
