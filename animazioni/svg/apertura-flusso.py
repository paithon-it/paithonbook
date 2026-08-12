"""Il problema dell'apertura: quello che si vede non e' quello che succede.

Lo stesso palo, lo stesso identico movimento (orizzontale, verso destra), visto
attraverso due finestrelle diverse. A sinistra la finestrella inquadra solo un
tratto di **bordo**, e il movimento lungo il bordo non produce nessun
cambiamento: quello che si misura e' la sola componente perpendicolare, che
punta in diagonale. A destra la finestrella inquadra la **punta** del palo, e li'
il movimento vero si legge per intero.

E' il tubo di cartone della scheda Elementare e l'equazione a due incognite di
quella Superiore, che sono la stessa cosa: una sola equazione per pixel
determina solo la componente parallela al gradiente.

I due numeri sotto le finestrelle li calcola la scena, proiettando lo
spostamento vero sulla normale al bordo.
"""

import math

from paithon_svg import *

NOME = "apertura-flusso"
TITOLO = "il problema dell'apertura"

ANGOLO = 62.0          # inclinazione del palo sull'orizzontale, in gradi
SPOSTAMENTO = 96.0     # lo spostamento vero, in pixel, orizzontale verso destra

A = (188.0, 190.0)     # centro della finestrella di sinistra (solo bordo)
B = (486.0, 190.0)     # centro della finestrella di destra (c'e' la punta)
R = 76.0
SPESSORE = 34.0        # larghezza del palo


def geometria():
    """Componente misurabile: la proiezione del moto vero sulla normale al bordo.

    Il bordo ha direzione (cos a, -sin a) nelle coordinate SVG, dove la y cresce
    in giu'. La normale e' (sin a, cos a). Il moto vero e' (SPOSTAMENTO, 0),
    quindi la proiezione vale SPOSTAMENTO * sin a, e punta in basso a destra.
    """
    a = math.radians(ANGOLO)
    misurato = SPOSTAMENTO * math.sin(a)
    direzione = (math.sin(a), math.cos(a))
    return misurato, direzione


def palo(cx, cy, avanti, indietro):
    """Il palo come segmento inclinato, da `indietro` a `avanti` lungo la sua direzione."""
    a = math.radians(ANGOLO)
    dx, dy = math.cos(a), -math.sin(a)
    return (f"M {cx - indietro * dx:.1f} {cy - indietro * dy:.1f} "
            f"L {cx + avanti * dx:.1f} {cy + avanti * dy:.1f}")


def costruisci() -> Figura:
    misurato, (nx, ny) = geometria()
    corpo, anim = [], []

    corpo.append(
        '<defs>'
        f'<marker id="pf" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
        f'markerHeight="6" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{TERRACOTTA}"/></marker>'
        f'<marker id="po" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
        f'markerHeight="6" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{OCRA}"/></marker>'
        f'<clipPath id="apA"><circle cx="{A[0]}" cy="{A[1]}" r="{R}"/></clipPath>'
        f'<clipPath id="apB"><circle cx="{B[0]}" cy="{B[1]}" r="{R}"/></clipPath>'
        '</defs>')

    # Lo stato di riposo e' la posizione finale, scritta in coordinate vere;
    # l'animazione parte dall'inverso (il palo arretrato) e finisce sull'identita'.
    anim.append(keyframes("scorri", [
        (0.0, f"transform:translateX({-SPOSTAMENTO:.0f}px)"),
        (6.0, f"transform:translateX({-SPOSTAMENTO:.0f}px)"),
        (86.0, "transform:translateX(0)"),
        (100.0, "transform:translateX(0)")]))

    # --- sinistra: la finestrella vede solo un tratto di bordo
    corpo.append(f'<g clip-path="url(#apA)">'
                 f'<path class="palo" d="{palo(A[0], A[1], 240, 240)}" '
                 f'style="animation:scorri var(--d) infinite"/></g>')
    corpo.append(f'<circle class="apertura" cx="{A[0]}" cy="{A[1]}" r="{R}"/>')

    # --- destra: la finestrella vede la punta, cioe' uno spigolo
    corpo.append(f'<g clip-path="url(#apB)">'
                 f'<path class="palo" d="{palo(B[0], B[1], 26, 240)}" '
                 f'style="animation:scorri var(--d) infinite"/></g>')
    corpo.append(f'<circle class="apertura" cx="{B[0]}" cy="{B[1]}" r="{R}"/>')

    # --- le frecce: a sinistra due (vero e misurato), a destra una sola
    y_fr = 320.0
    x0a = A[0] - SPOSTAMENTO / 2
    corpo.append(f'<line class="vero" x1="{x0a:.1f}" y1="{y_fr}" '
                 f'x2="{x0a + SPOSTAMENTO:.1f}" y2="{y_fr}" marker-end="url(#pf)"/>')
    corpo.append(f'<line class="mis" x1="{x0a:.1f}" y1="{y_fr}" '
                 f'x2="{x0a + misurato * nx:.1f}" y2="{y_fr + misurato * ny:.1f}" '
                 f'marker-end="url(#po)"/>')

    x0b = B[0] - SPOSTAMENTO / 2
    corpo.append(f'<line class="vero" x1="{x0b:.1f}" y1="{y_fr}" '
                 f'x2="{x0b + SPOSTAMENTO:.1f}" y2="{y_fr}" marker-end="url(#pf)"/>')

    # --- scritte
    corpo += [
        f'<text class="lbl" x="{A[0]:.0f}" y="52" text-anchor="middle">'
        f'la finestrella vede un bordo</text>',
        f'<text class="lbl" x="{B[0]:.0f}" y="52" text-anchor="middle">'
        f'la finestrella vede una punta</text>',
        f'<text class="lbs" x="{A[0]:.0f}" y="380" text-anchor="middle" '
        f'style="fill:{OCRA}">si misura {misurato:.0f} px, in diagonale</text>',
        f'<text class="lbs" x="{A[0]:.0f}" y="400" text-anchor="middle" '
        f'style="fill:{TERRACOTTA}">ne sono successi {SPOSTAMENTO:.0f}, '
        f'in orizzontale</text>',
        f'<text class="lbs" x="{B[0]:.0f}" y="380" text-anchor="middle" '
        f'style="fill:{TERRACOTTA}">si misurano tutti e {SPOSTAMENTO:.0f}</text>',
        f'<text class="lbs" x="{B[0]:.0f}" y="400" text-anchor="middle">'
        f'lo spigolo chiude le due incognite</text>',
        '<text class="lbs" x="340" y="434" text-anchor="middle">'
        'stesso palo, stesso movimento: cambia solo da dove lo si guarda</text>',
    ]

    return Figura(
        larghezza=680, altezza=452,
        alt="Due finestrelle rotonde affiancate, e in ciascuna scorre lo stesso "
            "palo inclinato che si muove verso destra. A sinistra si vede solo "
            "un tratto di bordo, e la freccia di ciò che si misura punta in "
            "diagonale, più corta della freccia del movimento vero, che è "
            "orizzontale. A destra si vede la punta del palo, e la freccia del "
            "misurato coincide con quella del movimento vero.",
        corpo="".join(corpo),
        stile=f"""    .palo     {{ fill:none; stroke:{TEAL}; stroke-width:{SPESSORE};
                 opacity:0.9; }}
    .apertura {{ fill:none; stroke:{BORDER_STRONG}; stroke-width:2.5; }}
    .vero     {{ stroke:{TERRACOTTA}; stroke-width:3.5; }}
    .mis      {{ stroke:{OCRA}; stroke-width:3.5; }}""",
        animazioni=anim,
        durata=5.0,
        fermi=".palo",
    )
