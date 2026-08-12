"""La deriva che scivola, e il numero che la misura.

Due curve cumulative: quella della finestra di riferimento (il passato su cui il
modello e' tarato) sta ferma, quella della finestra corrente scivola mese dopo
mese. Il segmento verticale in mezzo e' la statistica di Kolmogorov-Smirnov, che
non e' un'astrazione: e' letteralmente **la distanza verticale massima fra le
due curve**, e questa scena la disegna dove si trova.

Tutti i numeri sono calcolati: per due normali di uguale varianza sfalsate di
$\\mu$, il massimo scarto vale $2\\Phi(\\mu/2) - 1$ e cade in $x = \\mu/2$.

La soglia disegnata **non** e' quella del test. Alle taglie di una finestra di
produzione il $p$-value del KS sta sotto qualunque soglia anche per scostamenti
che nessun modello sente, ed e' esattamente il riflesso sbagliato contro cui la
sezione mette in guardia; quella disegnata e' una soglia di **ampiezza**, decisa
sul significato pratico, come il testo chiede di fare.
"""

import math

from paithon_svg import *

NOME = "deriva-ks"
TITOLO = "la deriva e il numero che la misura"

# Lo scostamento della finestra corrente, mese per mese: il mondo si sposta
# piano, e il primo mese e' identico al riferimento.
MU = [0.0, 0.15, 0.30, 0.50, 0.75, 1.05]
SOGLIA = 0.10        # ampiezza, non p-value: la sezione spiega perche'

RIQ = Riquadro(x=92, y=76, larg=496, alt=232, xmin=-3.4, xmax=4.4, ymin=0.0, ymax=1.0)
PUNTI = 161


def phi(x):
    """Cumulativa della normale standard."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def d_massimo(mu):
    """Lo scarto verticale massimo fra N(0,1) e N(mu,1): cade in x = mu/2."""
    return 2 * phi(mu / 2) - 1


def griglia():
    return [RIQ.xmin + k * (RIQ.xmax - RIQ.xmin) / (PUNTI - 1) for k in range(PUNTI)]


def polilinea(xs, ys):
    return " ".join(f"{RIQ.sx(x):.1f},{RIQ.sy(y):.1f}" for x, y in zip(xs, ys))


def costruisci() -> Figura:
    xs = griglia()
    corpo, anim = [], []
    corpo.append(RIQ.cornice())

    # --- il riferimento: fermo, sempre visibile
    corpo.append(f'<polyline class="rif" points="{polilinea(xs, [phi(x) for x in xs])}"/>')

    for i, mu in enumerate(MU):
        d = d_massimo(mu)
        x_d = mu / 2                       # dove cade il massimo scarto
        y_bas, y_alt = phi(-mu / 2), phi(mu / 2)

        t0, t1 = sosta(i, len(MU), tenuta=0.5)
        tappe = [(0.0, "opacity:0")]
        if t0 > 1.0:
            tappe.append((max(t0 - 1.5, 0.01), "opacity:0"))
        tappe += [(t0, "opacity:1"), (t1, "opacity:1")]
        if i < len(MU) - 1:
            tappe += [(min(t1 + 1.5, 99.9), "opacity:0"), (100.0, "opacity:0")]
        else:
            tappe.append((100.0, "opacity:1"))
        anim.append(keyframes(f"m{i}", tappe))
        op = 1 if i == len(MU) - 1 else 0
        stile = f'opacity="{op}" style="animation:m{i} var(--d) infinite"'

        corpo.append(f'<polyline class="cor" '
                     f'points="{polilinea(xs, [phi(x - mu) for x in xs])}" {stile}/>')

        # il segmento della distanza: e' *la* statistica, disegnata dov'e'
        if d > 0.004:
            corpo.append(
                f'<line class="dist" x1="{RIQ.sx(x_d):.1f}" y1="{RIQ.sy(y_bas):.1f}" '
                f'x2="{RIQ.sx(x_d):.1f}" y2="{RIQ.sy(y_alt):.1f}" {stile}/>')
            corpo.append(
                f'<text class="dval" x="{RIQ.sx(x_d) + 12:.0f}" '
                f'y="{(RIQ.sy(y_bas) + RIQ.sy(y_alt)) / 2 + 5:.0f}" {stile}>D</text>')

        # la didascalia che cambia: un testo per mese, sovrapposti
        verdetto = "sopra la soglia" if d > SOGLIA else "sotto la soglia"
        colore = TERRACOTTA if d > SOGLIA else FG_MUTED
        y_txt = RIQ.y + RIQ.alt + 44
        corpo.append(
            f'<text class="cap" x="{RIQ.x + RIQ.larg / 2:.0f}" y="{y_txt}" '
            f'text-anchor="middle" style="fill:{colore};animation:m{i} '
            f'var(--d) infinite" opacity="{op}">'
            f'mese {i} · D = {d:.2f} · {verdetto}</text>'.replace("D = 0.", "D = 0,"))

    corpo += [
        f'<text class="lbl" x="{RIQ.x}" y="{RIQ.y - 36}">'
        f'le due finestre, in cumulata</text>',
        f'<text class="lbs" x="{RIQ.x + RIQ.larg}" y="{RIQ.y - 36}" text-anchor="end" '
        f'style="fill:{TERRACOTTA}">finestra corrente</text>',
        f'<text class="lbs" x="{RIQ.x + RIQ.larg - 148}" y="{RIQ.y - 36}" '
        f'text-anchor="end" style="fill:{TEAL}">riferimento</text>',
        f'<text class="lbs" x="{RIQ.x + RIQ.larg / 2:.0f}" '
        f'y="{RIQ.y + RIQ.alt + 70}" text-anchor="middle">'
        f'la soglia è sull\'ampiezza di D, non sul p-value: '
        f'qui 0,10</text>',
    ]

    return Figura(
        larghezza=680, altezza=430,
        alt="Una curva cumulativa teal sta ferma; una curva terracotta, che "
            "all'inizio le sta sopra esattamente, scivola verso destra mese "
            "dopo mese. Un segmento verticale ocra unisce le due curve nel "
            "punto in cui sono più distanti, e la scritta sotto dice di quanto: "
            "si parte da zero e si arriva a 0,40, ben oltre la soglia di 0,10.",
        corpo="".join(corpo),
        stile=f"""    .rif  {{ fill:none; stroke:{TEAL}; stroke-width:3; }}
    .cor  {{ fill:none; stroke:{TERRACOTTA}; stroke-width:3; }}
    .dist {{ stroke:{OCRA}; stroke-width:4; stroke-linecap:round; }}
    .dval {{ font-family:{SANS}; font-size:15px; font-weight:700; fill:{OCRA}; }}
    .cap  {{ font-family:{SANS}; font-size:16px; font-weight:700; }}""",
        animazioni=anim,
        durata=len(MU) * 1.6,
        fermi=".cor, .dist, .dval, .cap",
    )
