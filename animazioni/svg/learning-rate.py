"""Il learning rate: troppo piccolo, giusto, troppo grande.

Tre discese sulla stessa parabola, con lo stesso punto di partenza e tre passi
diversi. Le traiettorie non sono disegnate a occhio: si itera davvero
x ← x − η f'(x) su f(x) = x², dove f'(x) = 2x, quindi x ← x(1 − 2η).

Il caso divergente non è un'esagerazione grafica: la successione è
x_k = x_0 (1 − 2η)^k, quindi diverge esattamente quando |1 − 2η| > 1, cioè per
η > 1. Con η = 1,05 il fattore vale −1,1 e ogni passo scavalca il minimo più
lontano di prima. (Con η = 0,55 il fattore sarebbe −0,1: oscilla, ma converge.)
"""

from paithon_svg import *

NOME = "learning-rate"
TITOLO = "il learning rate: piccolo, giusto, troppo grande"

X0 = 0.82
PASSI = 6
CASI = [(0.05, "η troppo piccolo", "striscia"),
        (0.40, "η giusto", "arriva"),
        (1.05, "η troppo grande", "diverge")]


def discesa(eta):
    """x ← x − η·f'(x) con f(x) = x². Restituisce la successione."""
    x, seq = X0, [X0]
    for _ in range(PASSI):
        x = x - eta * 2 * x
        seq.append(x)
    return seq


def costruisci() -> Figura:
    larg_pan, gap, y_top = 200, 30, 56
    corpo, anim = [], []

    for c, (eta, titolo, esito) in enumerate(CASI):
        x_off = 30 + c * (larg_pan + gap)
        r = Riquadro(x=x_off, y=y_top, larg=larg_pan, alt=210,
                     xmin=-1.55, xmax=1.55, ymin=-0.15, ymax=2.35)
        corpo.append(r.clip(f'p{c}') + r.cornice())

        # la parabola
        pts = []
        u = -1.5
        while u <= 1.5001:
            pts.append(f"{r.sx(u):.1f},{r.sy(u * u):.1f}")
            u += 0.05
        corpo.append(f'<g clip-path="url(#p{c})">'
                     f'<polyline class="par" points="{" ".join(pts)}"/></g>')

        seq = discesa(eta)
        vis = [min(max(x, -1.5), 1.5) for x in seq]     # niente esce dal riquadro
        corpo.append('<polyline class="tra" points="'
                     + " ".join(f"{r.sx(x):.1f},{r.sy(x * x):.1f}" for x in vis) + '"/>')

        # il punto: riposo sull'ultima posizione, animazione a ritroso fino alla prima
        xf, yf = r.sx(vis[-1]), r.sy(vis[-1] ** 2)
        n = len(vis)
        tappe = []
        for i, x in enumerate(vis):
            t0, t1 = sosta(i, n, tenuta=0.5)
            d = (f"transform:translate({r.sx(x) - xf:.1f}px,"
                 f"{r.sy(x * x) - yf:.1f}px)")
            tappe += [(t0, d), (t1, d)]
        tappe.append((100.0, "transform:translate(0px,0px)"))
        anim.append(keyframes(f"pt{c}", tappe))
        corpo.append(f'<circle class="dot" cx="{xf:.1f}" cy="{yf:.1f}" r="7" '
                     f'style="animation:pt{c} var(--d) infinite"/>')

        cls = "esito ko" if esito != "arriva" else "esito ok"
        corpo += [
            f'<text class="ttl" x="{x_off + larg_pan / 2:.0f}" y="{y_top - 22}" '
            f'text-anchor="middle">{titolo}</text>',
            f'<text class="lbs" x="{x_off + larg_pan / 2:.0f}" y="{y_top - 4}" '
            f'text-anchor="middle">η = {eta:g}</text>'.replace(".", ","),
            f'<text class="{cls}" x="{x_off + larg_pan / 2:.0f}" y="{y_top + 246}" '
            f'text-anchor="middle">{esito}</text>',
        ]

    corpo.append('<text class="lbl" x="30" y="340">'
                 'x ← x − η f′(x)   su   f(x) = x²</text>')

    return Figura(
        larghezza=720, altezza=366,
        alt="Tre parabole affiancate: con un learning rate piccolo il punto "
            "striscia senza arrivare al minimo, con quello giusto ci arriva in "
            "pochi passi, con quello troppo grande rimbalza allargandosi.",
        corpo="".join(corpo),
        stile=f"""    .par {{ fill:none; stroke:{TEAL}; stroke-width:2.5; }}
    .tra {{ fill:none; stroke:{TERRACOTTA}; stroke-width:2; stroke-opacity:0.45;
            stroke-dasharray:4 4; }}
    .dot {{ fill:{TERRACOTTA}; stroke:{CREAM}; stroke-width:2;
            transform-box:view-box; }}
    .esito {{ font-family:{SANS}; font-size:14px; font-weight:700; }}
    .ok  {{ fill:{TEAL}; }}
    .ko  {{ fill:{TERRACOTTA}; }}""",
        animazioni=anim,
        durata=8.0,
        fermi=".dot",
    )
