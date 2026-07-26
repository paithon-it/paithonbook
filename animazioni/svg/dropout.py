"""Dropout: a ogni passo la rete che si allena è un'altra rete.

Le maschere non sono disegnate a mano: si estraggono con un seme fisso, così la
figura è riproducibile e i conti tornano con quello che dice il testo.

Come le altre figure animate, il disegno fermo è l'**ultimo** passo: chi non
anima vede una rete con metà neuroni spenti, che è esattamente il concetto.
"""

import random

from paithon_svg import *

NOME = "dropout"
TITOLO = "il dropout spegne neuroni a caso"

STRATI = [3, 4, 4, 2]
P = 0.5          # probabilità di spegnimento sugli strati nascosti
PASSI = 4
SEME = 5


def maschere():
    """Per ogni passo, quali neuroni nascosti restano accesi."""
    rnd = random.Random(SEME)
    fuori = []
    for _ in range(PASSI):
        m = {}
        for s in (1, 2):
            # si evita che uno strato si spenga tutto: la rete non passerebbe nulla
            while True:
                riga = [rnd.random() > P for _ in range(STRATI[s])]
                if any(riga):
                    break
            m[s] = riga
        fuori.append(m)
    return fuori


def costruisci() -> Figura:
    masc = maschere()
    xs = [90, 250, 410, 570]
    r_nodo = 16
    alt = 300
    corpo, anim = [], []
    nomi_pattern = {}

    def pos(s, i):
        n = STRATI[s]
        passo_y = alt / (max(n, 2) + 1)
        return xs[s], 60 + (alt - (n - 1) * passo_y) / 2 + i * passo_y

    def acceso(s, i, k):
        return True if s in (0, 3) else masc[k][s][i]

    def pattern(stati):
        """Un @keyframes per ogni sequenza acceso/spento distinta."""
        chiave = tuple(stati)
        if chiave not in nomi_pattern:
            nome = f"m{len(nomi_pattern)}"
            nomi_pattern[chiave] = nome
            tappe = [(0.0, f"opacity:{1 if stati[0] else 0.12}")]
            for k, on in enumerate(stati):
                t0, t1 = sosta(k, PASSI, tenuta=0.82)
                d = f"opacity:{1 if on else 0.12}"
                tappe += [(max(t0 - 0.5, 0.01), d), (t0, d), (t1, d)]
            tappe.append((100.0, f"opacity:{1 if stati[-1] else 0.12}"))
            tappe.sort(key=lambda x: x[0])
            anim.append(keyframes(nome, tappe))
        return nomi_pattern[chiave]

    # archi sotto, nodi sopra
    for s in range(len(STRATI) - 1):
        for i in range(STRATI[s]):
            for j in range(STRATI[s + 1]):
                stati = [acceso(s, i, k) and acceso(s + 1, j, k) for k in range(PASSI)]
                nome = pattern(stati)
                x1, y1 = pos(s, i)
                x2, y2 = pos(s + 1, j)
                op = 1 if stati[-1] else 0.12
                corpo.append(
                    f'<line class="arc" x1="{x1 + r_nodo:.0f}" y1="{y1:.0f}" '
                    f'x2="{x2 - r_nodo:.0f}" y2="{y2:.0f}" opacity="{op}" '
                    f'style="animation:{nome} var(--d) infinite"/>')

    for s, n in enumerate(STRATI):
        for i in range(n):
            x, y = pos(s, i)
            fisso = s in (0, 3)
            cls = "nodo fisso" if fisso else "nodo"
            if fisso:
                corpo.append(f'<circle class="{cls}" cx="{x:.0f}" cy="{y:.0f}" r="{r_nodo}"/>')
            else:
                stati = [acceso(s, i, k) for k in range(PASSI)]
                nome = pattern(stati)
                op = 1 if stati[-1] else 0.12
                corpo.append(
                    f'<circle class="{cls}" cx="{x:.0f}" cy="{y:.0f}" r="{r_nodo}" '
                    f'opacity="{op}" style="animation:{nome} var(--d) infinite"/>')

    for i, nome in enumerate(("input", "nascosto", "nascosto", "output")):
        corpo.append(f'<text class="lbs" x="{xs[i]}" y="42" text-anchor="middle">{nome}</text>')

    # il contatore dei passi
    for k in range(PASSI):
        t0, _ = sosta(k, PASSI)
        p = 100.0 / PASSI
        anim.append(keyframes(f"pk{k}", [
            (0.0, "opacity:0"), (max(t0 - 0.4, 0.01), "opacity:0"), (t0, "opacity:1"),
            (min(t0 + p - 0.4, 99.9), "opacity:1"),
            (min(t0 + p, 100.0), "opacity:0"), (100.0, "opacity:0")]))
        fermo = ";opacity:1" if k == PASSI - 1 else ""
        corpo.append(f'<text class="pas" x="90" y="410" '
                     f'style="animation:pk{k} var(--d) infinite{fermo}">'
                     f'mini-batch {k + 1}</text>')

    corpo.append(f'<text class="lbl" x="90" y="440">p = {str(P).replace(chr(46), chr(44))}: a ogni passo metà '
                 f'dei neuroni nascosti sparisce</text>')

    return Figura(
        larghezza=680, altezza=470,
        alt="Una rete con due strati nascosti: a ogni mini-batch una metà "
            "diversa dei neuroni nascosti si spegne, insieme alle sue connessioni.",
        corpo="".join(corpo),
        stile=f"""    .arc  {{ stroke:{BORDER_STRONG}; stroke-width:1.6; }}
    .nodo {{ fill:{CREAM}; stroke:{TEAL}; stroke-width:3; }}
    .fisso {{ stroke:{FG_MUTED}; }}
    .pas  {{ font-family:{SANS}; font-size:16px; font-weight:700; fill:{TERRACOTTA}; opacity:0; }}""",
        animazioni=anim,
        durata=PASSI * 1.6,
        fermi=".arc, .nodo, .pas",
    )
