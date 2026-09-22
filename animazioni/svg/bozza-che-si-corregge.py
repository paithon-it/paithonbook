"""Perché lo speculative decoding restituisce il testo del modello grande.

Il tempo è il contenuto: la distribuzione che esce si costruisce in due
tempi. Prima la parola proposta dalla bozza viene tenuta con probabilità
min(1, p_t/p_b), e di ogni barra della bozza resta la parte che il grande
condivide, min(p_b, p_t). Poi, quando la proposta è scartata, si sorteggia
dal residuo [p_t - p_b]_+, e quel residuo riempie esattamente quello che
mancava. Alla fine le barre coincidono con quelle del modello grande.

Lo stato di riposo è quello finale: barre alte quanto p_t, divise nella parte
accettata e in quella ricampionata. I numeri li calcola il generatore; due
`assert` li difendono, uno esatto (la somma dà p_t) e uno empirico (il
sorteggio fatto davvero, con la regola di accettazione, ritrova p_t).
"""

import random

from paithon_svg import *

NOME = "bozza-che-si-corregge"
TITOLO = "la regola di accettazione dello speculative decoding"

PAROLE = ["il", "la", "un", "questo", "quel", "molto"]
P_T = [0.40, 0.25, 0.15, 0.10, 0.06, 0.04]     # il modello grande
P_B = [0.55, 0.10, 0.20, 0.05, 0.07, 0.03]     # la bozza
SORTEGGI, SEME = 200_000, 0


def scomponi(p_t, p_b):
    """Parte accettata e parte ricampionata di ciascuna parola."""
    accettata = [min(a, b) for a, b in zip(p_b, p_t)]
    beta = sum(accettata)                            # probabilità di accettare
    residuo = [max(0.0, t - b) for t, b in zip(p_t, p_b)]
    norma = sum(residuo)
    ricampionata = [(1 - beta) * r / norma for r in residuo]
    return accettata, ricampionata, beta


def sorteggia(p_t, p_b, n, seme):
    """La regola vera, eseguita: frequenza di ogni parola emessa."""
    rng = random.Random(seme)
    residuo = [max(0.0, t - b) for t, b in zip(p_t, p_b)]
    conte = [0] * len(p_t)
    indici = range(len(p_t))
    for _ in range(n):
        x = rng.choices(indici, weights=p_b)[0]
        if rng.random() < min(1.0, p_t[x] / p_b[x]):
            conte[x] += 1
        else:
            conte[rng.choices(indici, weights=residuo)[0]] += 1
    return [c / n for c in conte]


ACCETTATA, RICAMPIONATA, BETA = scomponi(P_T, P_B)
for _a, _r, _t in zip(ACCETTATA, RICAMPIONATA, P_T):
    if abs(_a + _r - _t) > 1e-12:
        raise AssertionError("parte accettata più ricampionata non dà p_t")
FREQ = sorteggia(P_T, P_B, SORTEGGI, SEME)
if max(abs(f - t) for f, t in zip(FREQ, P_T)) > 0.005:
    raise AssertionError("il sorteggio con la regola non ritrova p_t")


def costruisci() -> Figura:
    x0, base, alto = 60.0, 330.0, 520.0       # alto: pixel per probabilità 1
    passo, larg = 96.0, 40.0
    corpo = []
    n = 3
    anim = [
        # la parte accettata parte alta quanto la bozza e si accorcia al min
        keyframes("sgonfia", [(0, "transform:scaleY(var(--k))"),
                              (sosta(0, n)[1], "transform:scaleY(var(--k))"),
                              (sosta(1, n)[0], "transform:scaleY(1)"),
                              (100, "transform:scaleY(1)")]),
        # il residuo arriva dopo, e riempie
        keyframes("riempie", [(0, "transform:scaleY(0)"),
                              (sosta(1, n)[1], "transform:scaleY(0)"),
                              (sosta(2, n)[0], "transform:scaleY(1)"),
                              (100, "transform:scaleY(1)")]),
    ]
    for i, parola in enumerate(PAROLE):
        x = x0 + i * passo
        ha = ACCETTATA[i] * alto
        hr = RICAMPIONATA[i] * alto
        ht = P_T[i] * alto
        hb = P_B[i] * alto
        k = P_B[i] / ACCETTATA[i]
        # la bozza, in trasparenza, per confronto
        corpo.append(f'<rect class="bozza" x="{x + larg + 4:.1f}" y="{base - hb:.1f}" '
                     f'width="10" height="{hb:.1f}"/>')
        corpo.append(f'<rect class="acc" x="{x:.1f}" y="{base - ha:.1f}" '
                     f'width="{larg:.0f}" height="{ha:.1f}" '
                     f'style="--k:{k:.4f};animation:sgonfia var(--d) infinite;'
                     f'transform-box:fill-box;transform-origin:50% 100%"/>')
        if hr > 0:
            corpo.append(f'<rect class="ric" x="{x:.1f}" y="{base - ha - hr:.1f}" '
                         f'width="{larg:.0f}" height="{hr:.1f}" '
                         f'style="animation:riempie var(--d) infinite;'
                         f'transform-box:fill-box;transform-origin:50% 100%"/>')
        # il contorno del modello grande: il bersaglio
        corpo.append(f'<rect class="tgt" x="{x:.1f}" y="{base - ht:.1f}" '
                     f'width="{larg:.0f}" height="{ht:.1f}"/>')
        corpo.append(f'<text class="lbl" x="{x + larg / 2 + 7:.1f}" y="{base + 22:.0f}" '
                     f'text-anchor="middle">{parola}</text>')
        corpo.append(f'<text class="lbs" x="{x + larg / 2:.1f}" y="{base - ht - 8:.1f}" '
                     f'text-anchor="middle">{f"{FREQ[i]:.3f}".replace(".", ",")}</text>')
    corpo.append(f'<line class="asse" x1="{x0 - 10:.0f}" y1="{base:.0f}" '
                 f'x2="{x0 + len(PAROLE) * passo - 30:.0f}" y2="{base:.0f}"/>')
    # legenda
    ly = base + 56
    voci = (("acc", "accettata dalla bozza: min(pᵇ, pᵗ)"),
            ("ric", "ricampionata dal residuo, se scartata"),
            ("bozza", "la bozza pᵇ"),
            ("tgt", "il modello grande pᵗ"))
    for k, (cls, testo) in enumerate(voci):
        xx = x0 + (k % 2) * 300
        yy = ly + (k // 2) * 22
        corpo.append(f'<rect class="{cls}" x="{xx:.0f}" y="{yy - 11:.0f}" '
                     f'width="13" height="13"/>')
        corpo.append(f'<text class="lbs" x="{xx + 20:.0f}" y="{yy:.0f}">{testo}</text>')
    corpo.append(f'<text class="lbs" x="{x0:.0f}" y="{ly + 52:.0f}">'
                 f'sopra le barre: frequenze in {SORTEGGI // 1000} mila sorteggi con la '
                 f'regola (la bozza passa {round(100 * BETA)} volte su 100)</text>')
    return Figura(
        larghezza=680, altezza=ly + 70,
        alt=f"Sei parole, ciascuna con una barra. Parte dalla probabilità della "
            f"bozza; si accorcia alla parte che il modello grande condivide, "
            f"quella accettata; poi le si aggiunge sopra la parte ricampionata "
            f"dal residuo. Alla fine ogni barra è alta esattamente quanto la "
            f"probabilità del modello grande, e le frequenze di {SORTEGGI // 1000} mila "
            f"sorteggi fatti con la regola lo confermano.",
        corpo="".join(corpo),
        stile=f"""    .acc {{ fill:{TEAL}; }}
    .ric {{ fill:{TERRACOTTA}; }}
    .bozza {{ fill:{OCRA}; opacity:.8; }}
    .tgt {{ fill:none; stroke:{INK}; stroke-width:1.5; }}
    .asse {{ stroke:{BORDER_STRONG}; stroke-width:2; }}""",
        animazioni=anim,
        durata=9.0,
        fermi=".acc, .ric",
    )
