"""Autovettori: le due direzioni che una matrice non riesce a girare.

Ogni vettore unitario diventa Av. Quasi tutti cambiano direzione; quelli sulle
diagonali no, restano sulla propria retta e si limitano ad allungarsi (λ = 3)
o a non muoversi affatto (λ = 1). Angoli e lunghezze si calcolano applicando
davvero la matrice.

Nota di tecnica, valida per tutte le figure di questa cartella: il disegno
*fermo* usa coordinate vere, senza nessun `transform`, ed è lo stato finale.
L'animazione parte dalla trasformazione **inversa** e finisce sull'identità.
Così il riposo non dipende dal CSS, cioè regge in stampa, nei PDF e con
`prefers-reduced-motion`, e non c'è modo che i due stati divergano.
"""

import math

from paithon_svg import *

NOME = "autovettori"
TITOLO = "gli autovettori di una matrice"

A = ((2.0, 1.0), (1.0, 2.0))   # autovalori 3 e 1, autovettori (1,1) e (1,-1)
# 45/135/225/315 SONO le direzioni invarianti: devono comparire nel campione,
# altrimenti la figura non mostra ciò che promette
ANGOLI = [0, 30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 315, 330]


def applica(v):
    return (A[0][0] * v[0] + A[0][1] * v[1], A[1][0] * v[0] + A[1][1] * v[1])


def costruisci() -> Figura:
    r = Riquadro(x=100, y=34, larg=392, alt=392,
                 xmin=-3.3, xmax=3.3, ymin=-3.3, ymax=3.3)
    cx, cy = r.sx(0), r.sy(0)
    unita = r.scala_x

    anim, corpo = [], [r.cornice(croce=True)]
    corpo.append(f'<circle class="ghost" cx="{cx:.1f}" cy="{cy:.1f}" r="{unita:.1f}"/>')

    # le due rette invarianti, sotto ai vettori
    for g in (45, -45):
        a = math.radians(g)
        dx, dy = math.cos(a) * 3.2, math.sin(a) * 3.2
        corpo.append(f'<line class="inv" x1="{r.sx(-dx):.1f}" y1="{r.sy(-dy):.1f}" '
                     f'x2="{r.sx(dx):.1f}" y2="{r.sy(dy):.1f}"/>')

    for k, g in enumerate(ANGOLI):
        a = math.radians(g)
        v = (math.cos(a), math.sin(a))
        av = applica(v)
        lung = math.hypot(*av)

        # in SVG la y cresce in giù: gli angoli cambiano segno
        g0 = -g
        g1 = -math.degrees(math.atan2(av[1], av[0]))
        delta = g0 - g1                       # da dove deve *partire* l'animazione
        while delta > 180:
            delta -= 360
        while delta < -180:
            delta += 360

        # stato di riposo: il vettore trasformato, in coordinate esplicite
        x2 = cx + lung * unita * math.cos(math.radians(g1))
        y2 = cy + lung * unita * math.sin(math.radians(g1))

        anim.append(keyframes(f"tr{k}", [
            (0.0, f"transform:rotate({delta:.2f}deg) scale({1 / lung:.4f})"),
            (18.0, f"transform:rotate({delta:.2f}deg) scale({1 / lung:.4f})"),
            (58.0, "transform:rotate(0deg) scale(1)"),
            (100.0, "transform:rotate(0deg) scale(1)")]))

        cls = "eig" if g % 180 in (45, 135) else "vec"
        corpo.append(
            f'<line class="{cls}" x1="{cx:.1f}" y1="{cy:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'style="animation:tr{k} var(--d) infinite"/>')

    corpo.append(f'<text class="lbl" x="{r.x}" y="{r.y + r.alt + 40}">'
                 f'A v = λ v   con   λ = 3   e   λ = 1</text>')
    lx = r.x + r.larg + 26
    corpo += [
        f'<line class="eigl" x1="{lx}" y1="{r.y + 18}" x2="{lx + 22}" y2="{r.y + 18}"/>',
        f'<text class="lbs" x="{lx + 30}" y="{r.y + 23}">autovettori</text>',
        f'<line class="vecl" x1="{lx}" y1="{r.y + 46}" x2="{lx + 22}" y2="{r.y + 46}"/>',
        f'<text class="lbs" x="{lx + 30}" y="{r.y + 51}">tutti gli altri</text>',
    ]

    return Figura(
        larghezza=680, altezza=490,
        alt="Sedici vettori unitari vengono trasformati da una matrice: quasi "
            "tutti ruotano, quelli sulle due diagonali restano sulla propria "
            "retta e si limitano ad allungarsi.",
        corpo="".join(corpo),
        stile=f"""    .vec, .eig {{ transform-box:view-box; transform-origin:{cx:.1f}px {cy:.1f}px; }}
    .vec  {{ stroke:{TEAL}; stroke-width:3; }}
    .eig  {{ stroke:{TERRACOTTA}; stroke-width:5; }}
    .ghost {{ fill:none; stroke:{BORDER}; stroke-width:2; }}
    .inv  {{ stroke:{TERRACOTTA}; stroke-width:1.5; stroke-dasharray:6 6; opacity:0.5; }}
    .eigl {{ stroke:{TERRACOTTA}; stroke-width:5; }}
    .vecl {{ stroke:{TEAL}; stroke-width:3; }}""",
        animazioni=anim,
        durata=7.0,
        fermi=".vec, .eig",
    )
