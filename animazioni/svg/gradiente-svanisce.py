"""Il gradiente che svanisce: perché la sigmoide affonda le reti profonde.

Il segnale d'errore torna indietro moltiplicando, strato dopo strato, la
derivata dell'attivazione. Con la sigmoide quella derivata vale al massimo
0,25: cinque strati e il gradiente è già sceso di un fattore quattromila. Con
la ReLU vale 1 sulla parte attiva, e il prodotto non si consuma.

I numeri sono prodotti veri, non stime: 0,25⁵ = 0,00098.
"""

from paithon_svg import *

NOME = "gradiente-svanisce"
TITOLO = "il gradiente che svanisce"

STRATI = 6
D_SIGMOIDE = 0.25    # max di σ'(z), raggiunto in z = 0
D_RELU = 1.0


def catena(derivata):
    """Il fattore che arriva allo strato k risalendo dall'uscita."""
    v, out = 1.0, []
    for _ in range(STRATI):
        out.append(v)
        v *= derivata
    return out


def costruisci() -> Figura:
    righe = [("sigmoide", catena(D_SIGMOIDE), TERRACOTTA),
             ("ReLU", catena(D_RELU), TEAL)]
    larg_b, gap = 62, 22
    x0, alt_max = 130, 90
    corpo, anim = [], []

    for ri, (nome, vals, colore) in enumerate(righe):
        y_base = 152 + ri * 170
        corpo.append(f'<text class="nome" x="{x0 - 18}" y="{y_base + 6}" '
                     f'text-anchor="end" fill="{colore}">{nome}</text>')
        corpo.append(f'<line class="asse" x1="{x0 - 8}" y1="{y_base}" '
                     f'x2="{x0 + STRATI * (larg_b + gap) - gap + 8}" y2="{y_base}"/>')

        for k, v in enumerate(vals):
            # altezza in scala logaritmica: in scala lineare le ultime barre
            # sarebbero invisibili, e sparirebbe proprio ciò che si vuole mostrare
            import math
            h = max(alt_max * (1 + math.log10(max(v, 1e-4)) / 4), 3)
            x = x0 + k * (larg_b + gap)
            # il gradiente risale: lo strato piu' vicino all'uscita si accende per primo
            i = STRATI - 1 - k
            t0, t1 = sosta(i, STRATI, tenuta=0.9)
            anim.append(keyframes(f"b{ri}{k}", [
                (0.0, "opacity:0"), (max(t0 - 0.5, 0.01), "opacity:0"),
                (t0, "opacity:1"), (100.0, "opacity:1")]))
            corpo.append(
                f'<rect class="bar" x="{x}" y="{y_base - h:.1f}" width="{larg_b}" '
                f'height="{h:.1f}" fill="{colore}" '
                f'style="animation:b{ri}{k} var(--d) infinite"/>')
            # due cifre significative: "0,001" spaccerebbe 0,00098 per un millesimo
            testo = f"{v:#.2g}".rstrip("0").rstrip(".") if v < 1 else "1"
            corpo.append(
                f'<text class="val" x="{x + larg_b / 2:.0f}" y="{y_base - h - 8:.1f}" '
                f'text-anchor="middle" style="animation:b{ri}{k} var(--d) infinite">'
                f'{testo.replace(".", ",")}</text>')
            if ri == len(righe) - 1:
                corpo.append(f'<text class="lbs" x="{x + larg_b / 2:.0f}" '
                             f'y="{y_base + 22}" text-anchor="middle">strato '
                             f'{STRATI - k}</text>')

    corpo += [
        '<text class="lbl" x="30" y="34">il gradiente parte dall\'uscita '
        'e risale verso l\'ingresso</text>',
        f'<text class="lbl" x="30" y="{152 + 170 + 62}">'
        f'a ogni strato si moltiplica per σ′ ≤ 0,25 — oppure per 1, con la ReLU</text>',
    ]

    return Figura(
        larghezza=680, altezza=440,
        alt="Due file di barre, una per la sigmoide e una per la ReLU. "
            "Risalendo dall'uscita verso l'ingresso le barre della sigmoide si "
            "accorciano fino a sparire, quelle della ReLU restano intere.",
        corpo="".join(corpo),
        stile=f"""    .bar  {{ opacity:1; }}
    .asse {{ stroke:{BORDER_STRONG}; stroke-width:2; }}
    .nome {{ font-family:{SANS}; font-size:15px; font-weight:700; }}
    .val  {{ font-family:{SANS}; font-size:12px; fill:{FG_MUTED}; }}""",
        animazioni=anim,
        durata=STRATI * 1.1,
        fermi=".bar, .val",
    )
