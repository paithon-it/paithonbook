"""Il cammino dalla baseline all'ingresso, e il gradiente che lungo la strada è vivo.

Il gradiente valutato **nel punto** $\\mathbf{x}$ non dice niente quando la rete
e' satura: la curva li' e' piatta, la pendenza e' quasi zero, e l'attribuzione
pure. Gli Integrated Gradients non guardano quel punto: percorrono il segmento
dalla baseline all'ingresso e sommano la pendenza lungo tutta la strada, dove il
segnale c'e' ancora.

E' un metodo che **non si vede in un fotogramma**, perche' il fotogramma e'
esattamente il punto in cui il gradiente non dice niente.

La scena calcola tutto: la funzione e' $f(z) = \\tanh(4z)$ sul segmento da
$x' = 0$ a $x = 1$, la somma di Riemann e' a punto medio su otto passi, e il
riscontro in basso e' l'assioma di completezza, cioe' che la somma delle
attribuzioni fa esattamente $f(x) - f(x')$.
"""

import math

from paithon_svg import *

NOME = "gradienti-integrati"
TITOLO = "il cammino dalla baseline all'ingresso"

PASSI = 8


def f(z):
    return math.tanh(4 * z)


def df(z):
    return 4 * (1 - math.tanh(4 * z) ** 2)


SU = Riquadro(x=96, y=88, larg=300, alt=186, xmin=0.0, xmax=1.0, ymin=-0.04, ymax=1.06)
BARRA_X, BARRA_L = 520, 74
BARRA_ALTO, BARRA_BASSO = 88, 274


def riemann():
    """Somma a punto medio, passo per passo: e' cio' che il testo chiama m passi."""
    d = 1.0 / PASSI
    parziali, tot = [], 0.0
    for k in range(PASSI):
        tot += df((k + 0.5) * d) * d
        parziali.append(tot)
    return parziali


def costruisci() -> Figura:
    parziali = riemann()
    obiettivo = f(1.0) - f(0.0)
    scala = (BARRA_BASSO - BARRA_ALTO) / 1.06        # la barra e' alta quanto il riquadro

    corpo, anim = [], []
    corpo.append(SU.cornice())

    # la curva lungo il cammino: ferma, sempre visibile
    punti = " ".join(f"{SU.sx(k / 120):.1f},{SU.sy(f(k / 120)):.1f}" for k in range(121))
    corpo.append(f'<polyline class="curva" points="{punti}"/>')

    # il traguardo: f(x) - f(x'), che la somma deve raggiungere esattamente
    y_obb = BARRA_BASSO - obiettivo * scala
    corpo.append(f'<line class="obb" x1="{BARRA_X - 14}" y1="{y_obb:.1f}" '
                 f'x2="{BARRA_X + BARRA_L + 14}" y2="{y_obb:.1f}"/>')
    corpo.append(f'<text class="lbs" x="{BARRA_X - 18}" y="{y_obb + 5:.0f}" '
                 f'text-anchor="end" style="fill:{TERRACOTTA}">f(x) − f(x′)</text>')
    corpo.append(f'<line class="asse" x1="{BARRA_X - 14}" y1="{BARRA_BASSO}" '
                 f'x2="{BARRA_X + BARRA_L + 14}" y2="{BARRA_BASSO}"/>')

    for k in range(PASSI):
        a0, a1 = k / PASSI, (k + 1) / PASSI
        am = (k + 0.5) / PASSI
        t0, t1 = sosta(k, PASSI, tenuta=0.5)
        tappe = [(0.0, "opacity:0")]
        if t0 > 1.0:
            tappe.append((max(t0 - 1.4, 0.01), "opacity:0"))
        tappe += [(t0, "opacity:1"), (t1, "opacity:1")]
        if k < PASSI - 1:
            tappe += [(min(t1 + 1.4, 99.9), "opacity:0"), (100.0, "opacity:0")]
        else:
            tappe.append((100.0, "opacity:1"))
        anim.append(keyframes(f"p{k}", tappe))
        op = 1 if k == PASSI - 1 else 0
        stile = f'opacity="{op}" style="animation:p{k} var(--d) infinite"'

        # la fettina di cammino appena percorsa, e la pendenza li' in mezzo
        corpo.append(f'<rect class="fetta" x="{SU.sx(a0):.1f}" y="{SU.y}" '
                     f'width="{SU.sx(a1) - SU.sx(a0):.1f}" height="{SU.alt}" {stile}/>')
        dx = 0.09
        corpo.append(
            f'<line class="tang" x1="{SU.sx(am - dx):.1f}" '
            f'y1="{SU.sy(f(am) - df(am) * dx):.1f}" x2="{SU.sx(am + dx):.1f}" '
            f'y2="{SU.sy(f(am) + df(am) * dx):.1f}" {stile}/>')
        corpo.append(f'<circle class="pallino" cx="{SU.sx(am):.1f}" '
                     f'cy="{SU.sy(f(am)):.1f}" r="5.5" {stile}/>')

        # la barra che accumula, e i due numeri
        h = parziali[k] * scala
        corpo.append(f'<rect class="somma" x="{BARRA_X}" y="{BARRA_BASSO - h:.1f}" '
                     f'width="{BARRA_L}" height="{h:.1f}" {stile}/>')
        corpo.append(f'<text class="num" x="{BARRA_X + BARRA_L / 2:.0f}" '
                     f'y="{BARRA_BASSO - h - 12:.0f}" text-anchor="middle" {stile}>'
                     f'{parziali[k]:.3f}'.replace(".", ",") + '</text>')
        corpo.append(f'<text class="lbs" x="{SU.x + SU.larg / 2:.0f}" '
                     f'y="{SU.y + SU.alt + 30}" text-anchor="middle" {stile}>'
                     f'pendenza qui: {df(am):.2f}'.replace(".", ",") + '</text>')

    corpo += [
        f'<text class="lbl" x="{SU.x}" y="{SU.y - 30}">'
        f'lungo il cammino da x′ a x</text>',
        f'<text class="lbs" x="{SU.x}" y="{SU.y + SU.alt + 54}">'
        f'x′ (baseline)</text>',
        f'<text class="lbs" x="{SU.x + SU.larg}" y="{SU.y + SU.alt + 54}" '
        f'text-anchor="end">x (l\'ingresso da spiegare)</text>',
        f'<text class="lbl" x="{BARRA_X + BARRA_L / 2:.0f}" y="{SU.y - 30}" '
        f'text-anchor="middle">quanto si è sommato</text>',
        '<text class="lbs" x="340" y="358" text-anchor="middle">'
        'nel punto x la pendenza è quasi zero: è lì che il gradiente da solo non '
        'dice niente</text>',
    ]

    return Figura(
        larghezza=680, altezza=384,
        alt="A sinistra la curva della rete lungo il segmento che va dalla "
            "baseline all'ingresso: parte ripida e si appiattisce. Un pallino la "
            "percorre a passi, e a ogni passo un segmento mostra la pendenza in "
            "quel punto, che all'inizio è grande e alla fine quasi nulla. A "
            "destra una barra accumula la somma delle pendenze e si ferma "
            "esattamente sulla riga che segna la differenza fra l'uscita "
            "sull'ingresso e quella sulla baseline.",
        corpo="".join(corpo),
        stile=f"""    .curva   {{ fill:none; stroke:{TEAL}; stroke-width:3; }}
    .tang    {{ stroke:{TERRACOTTA}; stroke-width:3; stroke-linecap:round; }}
    .pallino {{ fill:{TERRACOTTA}; }}
    .fetta   {{ fill:{OCRA}; fill-opacity:0.22; }}
    .somma   {{ fill:{TEAL}; fill-opacity:0.9; }}
    .obb     {{ stroke:{TERRACOTTA}; stroke-width:2; stroke-dasharray:6 4; }}
    .asse    {{ stroke:{BORDER_STRONG}; stroke-width:2; }}
    .num     {{ font-family:{SANS}; font-size:14px; font-weight:700; fill:{TEAL}; }}""",
        animazioni=anim,
        durata=PASSI * 1.25,
        fermi=".fetta, .tang, .pallino, .somma, .num, .lbs",
    )
