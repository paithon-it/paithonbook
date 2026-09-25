"""Over-squashing: due gruppi densi uniti da un arco solo.

Figura ferma. Il limite di Topping e colleghi dice che la sensibilità dello
stato di v alla feature di u, a distanza K, sta sotto (A^K)_vu, con A la
matrice di adiacenza normalizzata con i cappi. Qui u e v stanno in due gruppi
densi collegati da un ponte, e quell'elemento è piccolo; il ponte è l'arco con
la curvatura più negativa, ed è lì che il rewiring aggiunge un arco.

I numeri li calcola il generatore: la curvatura Balanced Forman di Topping e
colleghi per ogni arco (la stessa che il testo nomina e che guida il loro
rewiring; qui i termini sui quadrilateri senza diagonali valgono zero, e un
`assert` lo controlla), l'elemento (A^K)_vu prima e dopo l'arco aggiunto. Gli `assert` difendono quello che la
figura promette: il ponte è l'arco più curvo in negativo, e l'arco aggiunto
fa crescere (A^K)_vu.
"""

import math

import numpy as np

from paithon_svg import *

NOME = "collo-di-bottiglia"
TITOLO = "il collo di bottiglia dell'over-squashing"

SX = [0, 1, 2, 3, 4]          # gruppo di sinistra
DX = [5, 6, 7, 8, 9]          # gruppo di destra
PONTE = (4, 5)
U, V = 0, 9
AGGIUNTO = (3, 8)             # l'arco che il rewiring aggiunge, sotto il ponte


def archi(extra=()):
    e = {(i, j) for g in (SX, DX) for i in g for j in g if i < j}
    e.add(PONTE)
    e.update(extra)
    return sorted(e)


def adiacenza(e, n=10):
    A = np.zeros((n, n))
    for i, j in e:
        A[i, j] = A[j, i] = 1.0
    return A


def normalizzata(A):
    At = A + np.eye(len(A))
    d = At.sum(1)
    return At / np.sqrt(np.outer(d, d))


def balanced_forman(A, i, j):
    """Curvatura Balanced Forman (Topping e colleghi, Def. 1) dell'arco (i, j).

    Si calcola la parte con gradi e triangoli; i termini sui quadrilateri senza
    diagonali qui non servono, e l'assert si ferma se un grafo li avesse."""
    d = A.sum(1)
    di, dj = d[i], d[j]
    triangoli = (A[i] * A[j]).sum()
    for k in np.nonzero(A[i])[0]:
        if k == j or A[j, k]:
            continue
        for w in np.nonzero(A[j])[0]:
            if w != i and not A[i, w] and A[k, w]:
                raise AssertionError(f"quadrilatero senza diagonali su {i}-{j}")
    return (2 / di + 2 / dj - 2 + 2 * triangoli / max(di, dj)
            + triangoli / min(di, dj))


def distanza(A, s, t):
    fronte, visti, k = {s}, {s}, 0
    while t not in fronte:
        fronte = {j for i in fronte for j in np.nonzero(A[i])[0]} - visti
        visti |= fronte
        k += 1
    return k


A0 = adiacenza(archi())
A1 = adiacenza(archi([AGGIUNTO]))
K = distanza(A0, U, V)
PRIMA = np.linalg.matrix_power(normalizzata(A0), K)[V, U]
DOPO = np.linalg.matrix_power(normalizzata(A1), K)[V, U]
CURV = {e: balanced_forman(A0, *e) for e in archi()}
if min(CURV, key=CURV.get) != PONTE:
    raise AssertionError("il ponte non è l'arco con la curvatura più negativa")
if DOPO <= 1.5 * PRIMA:
    raise AssertionError("l'arco aggiunto non allarga il collo di bottiglia")


def posizioni():
    pos = {}
    for k, i in enumerate(SX):
        a = 2 * math.pi * (k - 4) / 5          # il nodo del ponte guarda a destra
        pos[i] = (170 + 70 * math.cos(a), 130 - 70 * math.sin(a))
    for k, i in enumerate(DX):
        a = 2 * math.pi * k / 5                # e questo a sinistra
        pos[i] = (470 - 70 * math.cos(a), 130 - 70 * math.sin(a))
    return pos


def cifra(x):
    m, e = f"{x:.1e}".split("e")
    apici = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    return f"{m.replace('.', ',')} · 10{str(int(e)).translate(apici)}"


def decimale(x):
    return f"{x:.1f}".replace(".", ",").replace("-", "−")


def costruisci() -> Figura:
    pos = posizioni()
    c = []
    for (i, j) in archi([AGGIUNTO]):
        cls = ("ponte" if (i, j) == PONTE else
               "nuovo" if (i, j) == AGGIUNTO else "arco")
        (x1, y1), (x2, y2) = pos[i], pos[j]
        if cls == "nuovo":                 # passa sotto i due gruppi, in curva
            c.append(f'<path class="nuovo" d="M{x1:.1f},{y1:.1f} Q{(x1 + x2) / 2:.1f},'
                     f'{max(y1, y2) + 90:.1f} {x2:.1f},{y2:.1f}"/>')
            continue
        c.append(f'<line class="{cls}" x1="{x1:.1f}" y1="{y1:.1f}" '
                 f'x2="{x2:.1f}" y2="{y2:.1f}"/>')
    for i, (x, y) in pos.items():
        cls = "nodo-uv" if i in (U, V) else "nodo"
        c.append(f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="11"/>')
        if i in (U, V):
            c.append(f'<text class="lbl bianco" x="{x:.1f}" y="{y + 5:.1f}" '
                     f'text-anchor="middle">{"u" if i == U else "v"}</text>')
    (x1, y1), (x2, y2) = pos[PONTE[0]], pos[PONTE[1]]
    c.append(f'<text class="lbs" x="{(x1 + x2) / 2:.0f}" y="{y1 - 16:.0f}" '
             f'text-anchor="middle">ponte, curvatura '
             f'{decimale(CURV[PONTE])}</text>')
    (x1, y1), (x2, y2) = pos[AGGIUNTO[0]], pos[AGGIUNTO[1]]
    c.append(f'<text class="lbs nuovo-t" x="{(x1 + x2) / 2:.0f}" '
             f'y="{max(y1, y2) + 64:.0f}" text-anchor="middle">'
             f'arco aggiunto dal rewiring</text>')
    ly = 322
    apice = str(K).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
    c += [f'<text class="lbl" x="40" y="{ly:.0f}">u e v distano {K} passi; il limite '
          f'sulla sensibilità di v a u, (Â{apice})ᵥᵤ, vale</text>',
          f'<text class="lbs" x="40" y="{ly + 24:.0f}">{cifra(PRIMA)} con il solo '
          f'ponte, {cifra(DOPO)} con l’arco aggiunto</text>']
    return Figura(
        larghezza=720, altezza=ly + 44,
        alt=f"Due gruppi di cinque nodi, ognuno collegato al proprio interno, "
            f"uniti da un solo arco, il ponte, che ha la curvatura più negativa "
            f"del grafo ({decimale(CURV[PONTE])}). Il nodo u sta a sinistra e v a destra, "
            f"a {K} passi. Un arco tratteggiato, aggiunto dal rewiring accanto al "
            f"ponte, apre una seconda strada e fa crescere il limite sulla "
            f"sensibilità di v a u da {cifra(PRIMA)} a {cifra(DOPO)}.",
        corpo="".join(c),
        stile=f"""    .arco {{ stroke:{BORDER_STRONG}; stroke-width:1.5; }}
    .ponte {{ stroke:{TERRACOTTA}; stroke-width:3.5; }}
    .nuovo {{ stroke:{TEAL}; stroke-width:2.5; stroke-dasharray:6 4; fill:none; }}
    .nuovo-t {{ fill:{TEAL}; }}
    .nodo {{ fill:{CREAM}; stroke:{INK}; stroke-width:1.5; }}
    .nodo-uv {{ fill:{TEAL}; stroke:{INK}; stroke-width:1.5; }}
    .bianco {{ fill:{CREAM}; font-size:13px; }}""",
    )
