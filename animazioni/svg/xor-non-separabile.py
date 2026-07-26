"""Il muro dello XOR: nessuna retta ce la fa, e si vede provandole tutte.

Il contraltare del percettrone: là la retta convergeva, qui gira su sé stessa
e resta sempre con due punti sbagliati. Il conteggio degli errori non è
scritto a mano: si calcola per ogni orientamento mostrato.
"""

import math

from paithon_svg import *

NOME = "xor-non-separabile"
TITOLO = "lo XOR non è separabile linearmente"

DATI = [((1.2, 1.2), 1), ((-1.2, -1.2), 1),      # stessa classe: diagonale
        ((1.2, -1.2), -1), ((-1.2, 1.2), -1)]    # l'altra: antidiagonale

ANGOLI = [20, 65, 110, 155, 200]   # gli orientamenti provati, in gradi


def prova(alpha_gradi):
    """Errori di una retta per l'origine con normale ad angolo alpha."""
    a = math.radians(alpha_gradi)
    w = (math.cos(a), math.sin(a))
    sbagliati = [i for i, ((x, y), t) in enumerate(DATI)
                 if t * (w[0] * x + w[1] * y) <= 0]
    return w, sbagliati


def costruisci() -> Figura:
    r = Riquadro(xmin=-1.9, xmax=1.9, ymin=-1.9, ymax=1.9)
    stati = [(a, *prova(a)) for a in ANGOLI]
    n = len(stati)

    peggio = min(len(s[2]) for s in stati)
    if peggio < 2:
        raise AssertionError(f"uno degli orientamenti sbaglia solo {peggio} punti: "
                             "con lo XOR non può succedere")

    pose = [r.posa_retta(w, 0.0) for _, w, _ in stati]
    tappe = []
    for i, (px, py, ang) in enumerate(pose):
        t0, t1 = sosta(i, n)
        d = f"transform:translate({px:.1f}px,{py:.1f}px) rotate({ang:.1f}deg)"
        tappe += [(t0, d), (t1, d)]
    px, py, ang = pose[-1]
    tappe.append((100.0, f"transform:translate({px:.1f}px,{py:.1f}px) rotate({ang:.1f}deg)"))

    anim = [keyframes("gira", tappe)]
    corpo = [r.clip("campo"), r.cornice(croce=True),
             f'<g clip-path="url(#campo)"><g id="retta" '
             f'transform="translate({px:.1f},{py:.1f}) rotate({ang:.1f})">'
             f'<line class="sep" x1="-620" y1="0" x2="620" y2="0"/></g></g>']

    # ogni punto si cerchia nelle fasi in cui è dalla parte sbagliata
    for i, ((x, y), t) in enumerate(DATI):
        fasi = [k for k, (_, _, sb) in enumerate(stati) if i in sb]
        cls = "pos" if t > 0 else "neg"
        extra = ""
        if fasi:
            tappe_p = [(0.0, "stroke-opacity:0")]
            for k in fasi:
                t0, t1 = sosta(k, n, tenuta=0.92)
                tappe_p += [(max(t0 - 0.8, 0.01), "stroke-opacity:0"),
                            (t0, "stroke-opacity:0.55"),
                            (t1, "stroke-opacity:0.55"),
                            (min(t1 + 0.8, 99.9), "stroke-opacity:0")]
            tappe_p.append((100.0, "stroke-opacity:0"))
            tappe_p.sort(key=lambda x: x[0])
            anim.append(keyframes(f"err{i}", tappe_p))
            fermo = ";stroke-opacity:0.55" if i in stati[-1][2] else ""
            corpo.append(f'<circle class="err" cx="{r.sx(x):.1f}" cy="{r.sy(y):.1f}" '
                         f'r="9" style="animation:err{i} var(--d) infinite{fermo}"/>')
        corpo.append(f'<circle class="pt {cls}" cx="{r.sx(x):.1f}" cy="{r.sy(y):.1f}" '
                     f'r="9"{extra}/>')

    corpo.append(f'<text class="cnt" x="{r.x}" y="{r.y + r.alt + 36}">'
                 f'sempre {peggio} punti sbagliati</text>')
    corpo.append(f'<text class="lbl" x="{r.x}" y="{r.y + r.alt + 64}">'
                 f'nessuna retta mette le due classi da parti opposte</text>')

    lx = r.x + r.larg + 26
    corpo += [
        f'<circle class="pt pos" cx="{lx}" cy="{r.y + 20}" r="9"/>',
        f'<text class="lbs" x="{lx + 18}" y="{r.y + 25}">XOR = 0</text>',
        f'<circle class="pt neg" cx="{lx}" cy="{r.y + 50}" r="9"/>',
        f'<text class="lbs" x="{lx + 18}" y="{r.y + 55}">XOR = 1</text>',
    ]

    return Figura(
        larghezza=664, altezza=500,
        alt="Una retta ruota su quattro punti disposti a XOR: a ogni "
            "orientamento restano due punti dalla parte sbagliata, cerchiati.",
        corpo="".join(corpo),
        stile=f"""    .pt  {{ stroke:{CREAM}; }}
    .pos {{ fill:{TERRACOTTA}; stroke:{TERRACOTTA}; }}
    .neg {{ fill:{TEAL}; stroke:{TEAL}; }}
    .sep {{ stroke:{INK}; stroke-width:3; }}
    .err {{ fill:none; stroke:{INK}; stroke-width:11; stroke-opacity:0; }}
    .cnt {{ font-family:{SANS}; font-size:16px; font-weight:700; fill:{TERRACOTTA}; }}
    #retta {{ animation:gira var(--d) infinite; transform-box:view-box; }}""",
        animazioni=anim,
        durata=n * 1.5,
        fermi="#retta, .err",
    )
