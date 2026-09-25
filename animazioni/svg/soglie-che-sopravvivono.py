"""Le soglie compatibili con gli esempi, mentre gli esempi arrivano.

È la scena di `TeoriaApprendimento/pac.md`: esempi estratti uniformi in [0, 1],
etichettati dalla soglia vera 0,371, e la famiglia delle soglie h_t(x) = 1 se
x >= t. Le soglie ancora compatibili con tutti gli esempi visti sono quelle fra
il negativo più a destra e il positivo più a sinistra: la banda ocra. Gli
esempi arrivano a gruppi che raddoppiano (5, 10, 20, 40, 80), e a ogni gruppo
la banda si stringe.

Il sorteggio è quello di `random.Random(SEME)`, identico su ogni macchina. La
didascalia promette due cose, e le difendono due assert: all'inizio la banda
contiene soglie con errore oltre la tolleranza, alla fine non più. Che alla
fine non ne resti nessuna non è garantito (con ottanta esempi succede circa 97
volte su cento, lo stima il blocco di codice della pagina): è vero per questa
estrazione, e l'assert impedisce che un seme diverso lo renda falso in
silenzio.

Il disegno fermo è lo stato finale (ottanta esempi, banda stretta): chi non
anima vede la conclusione.
"""

import random

from paithon_svg import *

NOME = "soglie-che-sopravvivono"
TITOLO = "le soglie compatibili si stringono mentre arrivano gli esempi"

VERA = 0.371
EPS = 0.05
GRUPPI = [5, 10, 20, 40, 80]
SEME = 7

X0, X1 = 60, 660          # il segmento [0, 1] in pixel
Y = 170                   # l'asse
YB = Y - 34               # la corsia della banda, sopra i puntini


def px(t: float) -> float:
    return X0 + t * (X1 - X0)


def esempi():
    rnd = random.Random(SEME)
    return [(rnd.random(), rnd.uniform(-9, 9)) for _ in range(GRUPPI[-1])]


def banda(xs):
    """Le soglie compatibili: da dopo l'ultimo negativo al primo positivo."""
    neg = [x for x in xs if x < VERA]
    pos = [x for x in xs if x >= VERA]
    return (max(neg) if neg else 0.0), (min(pos) if pos else 1.0)


def costruisci() -> Figura:
    punti = esempi()
    bande = [banda([x for x, _ in punti[:n]]) for n in GRUPPI]

    # quello che la didascalia promette
    lo0, hi0 = bande[0]
    assert lo0 < VERA - EPS or hi0 > VERA + EPS, "all'inizio non c'è nessuna soglia cattiva"
    lo, hi = bande[-1]
    assert VERA - EPS <= lo and hi <= VERA + EPS, "alla fine sopravvive una soglia cattiva"
    # la didascalia dice che la banda non si allarga mai, e che fra dieci e
    # venti esempi resta ferma: tutte e due le cose si difendono qui
    assert all(b[1] - b[0] >= c[1] - c[0] for b, c in zip(bande, bande[1:])), \
        "la banda si allarga"
    assert bande[1] == bande[2], "fra dieci e venti esempi la banda si muove"

    n = len(GRUPPI)
    corpo, anim = [], []

    # la banda: il riposo è l'ultima, e le altre ne sono trasformazioni
    L, H = px(lo), px(hi)
    # la transizione sta fra la fine di una sosta e l'inizio della successiva
    ordinate = []
    for k in range(n):
        t0, _ = sosta(k, n)
        l, h = bande[k]
        tr = f"transform:translateX({px(l) - L:.1f}px) scaleX({(px(h) - px(l)) / (H - L):.3f})"
        ordinate.append((t0, tr))
    passi = [(0.0, ordinate[0][1])]
    for k in range(1, n):
        t0 = ordinate[k][0]
        passi += [(t0 - 2.0, ordinate[k - 1][1]), (t0 + 3.0, ordinate[k][1])]
    passi.append((100.0, "transform:translateX(0px) scaleX(1)"))
    anim.append(keyframes("banda", passi))
    corpo.append(f'<line class="guida" x1="{X0}" y1="{YB}" x2="{X1}" y2="{YB}"/>')
    corpo.append(f'<rect class="banda" x="{L:.1f}" y="{YB - 7}" width="{H - L:.1f}" '
                 f'height="14" style="animation:banda var(--d) infinite"/>')

    # l'asse e la tolleranza attorno alla soglia vera
    corpo.append(f'<line class="asse" x1="{X0}" y1="{Y}" x2="{X1}" y2="{Y}"/>')
    for t, et in ((0, "0"), (0.5, "0,5"), (1, "1")):
        corpo.append(f'<line class="asse" x1="{px(t):.1f}" y1="{Y - 5}" x2="{px(t):.1f}" y2="{Y + 5}"/>')
        corpo.append(f'<text class="lbs" x="{px(t):.1f}" y="{Y + 44}" text-anchor="middle">{et}</text>')
    a, b = px(VERA - EPS), px(VERA + EPS)
    corpo.append(f'<path class="toll" d="M{a:.1f},{YB - 14} L{a:.1f},{YB - 22} L{b:.1f},{YB - 22} L{b:.1f},{YB - 14}"/>')
    corpo.append(f'<text class="lbs" x="{(a + b) / 2:.1f}" y="{YB - 30}" text-anchor="middle">'
                 f'errore al massimo {EPS:.0%}</text>')
    v = px(VERA)
    corpo.append(f'<path class="vera" d="M{v:.1f},{Y + 13} L{v - 6:.1f},{Y + 23} L{v + 6:.1f},{Y + 23} Z"/>')
    corpo.append(f'<text class="lbs" x="{v:.1f}" y="{Y + 64}" text-anchor="middle">soglia vera</text>')

    # gli esempi, che compaiono gruppo per gruppo
    inizio = 0
    for k, fine in enumerate(GRUPPI):
        t0, _ = sosta(k, n)
        if k > 0:
            anim.append(keyframes(f"g{k}", [(0.0, "opacity:0"), (t0 - 2.0, "opacity:0"),
                                            (t0, "opacity:1"), (100.0, "opacity:1")]))
        for x, dy in punti[inizio:fine]:
            cls = "pos" if x >= VERA else "neg"
            st = f' style="animation:g{k} var(--d) infinite"' if k > 0 else ""
            corpo.append(f'<circle class="{cls}" cx="{px(x):.1f}" cy="{Y + dy:.1f}" r="4.2"{st}/>')
        inizio = fine

    # il contatore: ogni scritta resta accesa per il suo gruppo, l'ultima a riposo
    for k, m in enumerate(GRUPPI):
        t0, _ = sosta(k, n)
        p = 100.0 / n
        if k < n - 1:
            tappe = [(0.0, "opacity:0"), (max(t0 - 0.4, 0.01), "opacity:0"),
                     (max(t0, 0.02), "opacity:1"), (t0 + p - 0.4, "opacity:1"),
                     (t0 + p, "opacity:0"), (100.0, "opacity:0")]
            if k == 0:
                tappe = [(0.0, "opacity:1"), (t0 + p - 0.4, "opacity:1"),
                         (t0 + p, "opacity:0"), (100.0, "opacity:0")]
        else:
            tappe = [(0.0, "opacity:0"), (t0 - 0.4, "opacity:0"), (t0, "opacity:1"),
                     (100.0, "opacity:1")]
        anim.append(keyframes(f"c{k}", tappe))
        fermo = ";opacity:1" if k == n - 1 else ""
        corpo.append(f'<text class="cont" x="{X0}" y="44" '
                     f'style="animation:c{k} var(--d) infinite{fermo}">{m} esempi</text>')

    corpo.append(f'<text class="lbs" x="{X0}" y="{Y + 92}">in ocra, sopra: le soglie ancora '
                 f'compatibili con tutti gli esempi visti</text>')
    corpo.append(f'<text class="lbs" x="{X0}" y="{Y + 110}">sull\u2019asse: gli esempi, '
                 f'terracotta i positivi e teal i negativi</text>')

    return Figura(
        larghezza=720, altezza=296,
        alt="Animazione su un segmento orizzontale da zero a uno. Gli esempi "
            "compaiono a gruppi, cinque, dieci, venti, quaranta e ottanta, come "
            "puntini color terracotta se positivi e teal se negativi; una banda ocra "
            "segna le soglie ancora compatibili con tutti gli esempi e a ogni gruppo "
            "si restringe o resta ferma, senza mai allargarsi. Sopra il segmento una "
            "parentesi segna le soglie con errore al massimo del cinque per cento "
            "attorno alla soglia vera. Alla fine la banda ocra sta tutta dentro la "
            "parentesi.",
        corpo="".join(corpo),
        stile=f"""    .asse  {{ stroke:{INK}; stroke-width:1.6; }}
    .guida {{ stroke:{BORDER_STRONG}; stroke-width:1.2; stroke-dasharray:3 4; }}
    .toll  {{ stroke:{INK}; stroke-width:1.4; fill:none; }}
    .vera  {{ fill:{INK}; }}
    .banda {{ fill:{OCRA}; opacity:.75; transform-box:fill-box; transform-origin:0% 50%; }}
    .pos   {{ fill:{TERRACOTTA}; stroke:{CREAM}; stroke-width:.8; }}
    .neg   {{ fill:{TEAL}; stroke:{CREAM}; stroke-width:.8; }}
    .cont  {{ font-family:{SANS}; font-size:16px; font-weight:700; fill:{TERRACOTTA}; opacity:0; }}""",
        animazioni=anim,
        durata=len(GRUPPI) * 1.8,
        fermi=".banda, .pos, .neg, .cont",
    )
