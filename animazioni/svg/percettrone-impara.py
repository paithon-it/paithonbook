"""La regola del percettrone: la retta che ruota finché separa.

L'algoritmo gira davvero (vedi `addestra`); l'SVG ne mostra gli stati.
"""

from paithon_svg import *

NOME = "percettrone-impara"
TITOLO = "la regola del percettrone"

# centrate sull'origine: così la retta ruota invece di traslare a lungo, ed è
# la rotazione la cosa da guardare
POS = [(1.1, 0.8), (1.7, 0.1), (0.6, 1.7), (1.4, 1.4)]
NEG = [(-1.1, -0.8), (-1.7, 0.0), (-0.6, -1.7), (-1.4, -1.4)]
# etichette 0/1, la convenzione del capitolo: il gradino vale 1 se z >= 0 e 0
# altrimenti, e la regola aggiorna di eta*(y - yhat)*x. Con le etichette +-1 e
# w += eta*t*x (come era qui) i passi sono metà di quelli che la formula
# stampata sotto la figura prescrive: la figura smentiva sé stessa.
DATI = [(p, 1) for p in POS] + [(p, 0) for p in NEG]

W0, B0, ETA = (-0.9, 0.5), 0.0, 0.25   # partenza deliberatamente sbagliata

CORREZIONI_ATTESE = 4      # quante ne annuncia percettrone.md


def addestra():
    """Regola del percettrone. Restituisce (w, b, indice del punto corretto)."""
    w, b = list(W0), B0
    stati = [(tuple(w), b, None)]
    for _ in range(20):
        errori = 0
        for i, ((x, y), y_vero) in enumerate(DATI):
            y_hat = 1 if w[0] * x + w[1] * y + b >= 0 else 0
            if y_hat != y_vero:
                d = y_vero - y_hat
                w[0] += ETA * d * x
                w[1] += ETA * d * y
                b += ETA * d
                stati.append((tuple(w), b, i))
                errori += 1
        if not errori:
            break
    return stati


def costruisci() -> Figura:
    r = Riquadro()
    stati = addestra()
    n = len(stati)
    pose = [r.posa_retta(w, b) for w, b, _ in stati]

    if not r.separa(pose[-1], DATI):
        raise AssertionError("la retta disegnata non separa le classi")
    if n - 1 != CORREZIONI_ATTESE:
        raise AssertionError(f"la figura disegna {n - 1} correzioni, il capitolo "
                             f"ne annuncia {CORREZIONI_ATTESE}")

    tappe = []
    for i, (px, py, a) in enumerate(pose):
        t0, t1 = sosta(i, n)
        d = f"transform:translate({px:.1f}px,{py:.1f}px) rotate({a:.1f}deg)"
        tappe += [(t0, d), (t1, d)]
    px, py, a = pose[-1]
    tappe.append((100.0, f"transform:translate({px:.1f}px,{py:.1f}px) rotate({a:.1f}deg)"))

    anim = [keyframes("ruota", tappe)]
    corpo = [r.clip("campo"), r.cornice(croce=True),
             f'<g clip-path="url(#campo)"><g id="retta" '
             f'transform="translate({px:.1f},{py:.1f}) rotate({a:.1f})">'
             f'<line class="sep" x1="-620" y1="0" x2="620" y2="0"/></g></g>']

    # il punto corretto a ogni passo si accende nella propria fetta di timeline
    primo_uso = {}
    for i, (_, _, idx) in enumerate(stati):
        if idx is not None:
            primo_uso.setdefault(idx, i)
    for i, ((x, y), y_vero) in enumerate(DATI):
        cls = "pos" if y_vero else "neg"
        extra = ""
        if i in primo_uso:
            k = primo_uso[i]
            t0, t1 = sosta(k, n)
            anim.append(keyframes(f"lampo{k}", [
                (0.0, "stroke-width:1;stroke-opacity:0"),
                (max(t0 - 1.2, 0.01), "stroke-width:1;stroke-opacity:0"),
                (t0, "stroke-width:12;stroke-opacity:0.4"),
                (t1, "stroke-width:1;stroke-opacity:0"),
                (100.0, "stroke-width:1;stroke-opacity:0")]))
            extra = f' style="animation:lampo{k} var(--d) infinite"'
        corpo.append(f'<circle class="pt {cls}" cx="{r.sx(x):.1f}" cy="{r.sy(y):.1f}" '
                     f'r="8"{extra}/>')

    # l'etichetta del passo: una per stato, visibile nella propria fetta
    for i in range(n):
        t0, _ = sosta(i, n)
        passo = 100.0 / n
        anim.append(keyframes(f"eti{i}", [
            (0.0, "opacity:0"), (max(t0 - 0.6, 0.01), "opacity:0"), (t0, "opacity:1"),
            (min(t0 + passo - 0.6, 99.9), "opacity:1"),
            (min(t0 + passo, 100.0), "opacity:0"), (100.0, "opacity:0")]))
        testo = "partenza" if i == 0 else f"correzione {i}"
        fermo = ";opacity:1" if i == n - 1 else ""
        corpo.append(f'<text class="eti" x="{r.x}" y="{r.y + r.alt + 34}" '
                     f'style="animation:eti{i} var(--d) infinite{fermo}">{testo}</text>')

    lx = r.x + r.larg + 26
    corpo += [
        f'<circle class="pt pos" cx="{lx}" cy="{r.y + 20}" r="8"/>',
        f'<text class="lbs" x="{lx + 16}" y="{r.y + 25}">classe 1</text>',
        f'<circle class="pt neg" cx="{lx}" cy="{r.y + 48}" r="8"/>',
        f'<text class="lbs" x="{lx + 16}" y="{r.y + 53}">classe 0</text>',
        f'<text class="lbl" x="{r.x}" y="{r.y + r.alt + 62}">w ← w + η (y − ŷ) x</text>',
    ]

    return Figura(
        larghezza=664, altezza=500,
        alt="La retta di separazione di un percettrone ruota a ogni correzione "
            "finché separa i punti terracotta dai punti teal.",
        corpo="".join(corpo),
        stile=f"""    .pt  {{ stroke:{CREAM}; }}
    .pos {{ fill:{TERRACOTTA}; stroke:{TERRACOTTA}; }}
    .neg {{ fill:{TEAL}; stroke:{TEAL}; }}
    .sep {{ stroke:{INK}; stroke-width:3; }}
    .eti {{ font-family:{SANS}; font-size:15px; fill:{FG_MUTED}; opacity:0; }}
    #retta {{ animation:ruota var(--d) infinite; transform-box:view-box; }}""",
        animazioni=anim,
        durata=n * 1.5,
        fermi="#retta, .pt, .eti",
    )
