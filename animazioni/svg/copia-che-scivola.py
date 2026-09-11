"""Autocorrelazione: la copia scivola indietro, e il correlogramma si riempie.

Il tempo è il contenuto due volte. È l'asse su cui sta la serie, ed è il
meccanismo: l'autocorrelazione *è* uno scorrimento, «fanne una copia e falla
scivolare indietro di un giorno, di due, di dodici». Una figura ferma può
mostrare uno scorrimento solo, e il punto è che sono venticinque e che la
somiglianza cambia segno lungo la strada: a mezzo ciclo le due curve sono
capovolte, a un ciclo intero tornano a sovrapporsi.

I numeri non sono disegnati a mano. La serie è quella del blocco di
`SerieTemporali/overview.md` (stessa formula, stesso seme, stessi 200 punti) e
i coefficienti li calcola la stessa `autocorr` del capitolo: le asserzioni qui
sotto pretendono che i tre valori della figura siano quelli che la pagina
stampa, così che ritoccare l'uno senza l'altro fermi la generazione.

Lo stato di riposo è l'ultimo scorrimento: la copia è arrivata in fondo e il
correlogramma è completo. Chi non anima vede una figura conclusa, che è quella
che finisce in stampa.
"""

import math

from paithon_svg import *

NOME = "copia-che-scivola"
TITOLO = "la copia che scivola, e il correlogramma"

# I parametri sono quelli del capitolo (SerieTemporali/overview.md, sezione
# "Perché è un problema diverso (e difficile)").
N, PERIODO, SEME = 200, 12, 0
RITARDO_MAX = 24          # due cicli interi: la seconda cresta si deve vedere
FINESTRA = 96             # quanti punti si disegnano, cioè otto cicli

# Quello che la pagina stampa, e che questa figura non può smentire.
ATTESI = {1: 0.778, 6: -0.869, 12: 0.826}


def serie_del_capitolo(n=N, seme=SEME):
    """La stessa serie sintetica del blocco: tendenza + stagionalità + rumore."""
    import numpy as np

    rng = np.random.default_rng(seme)
    t = np.arange(n)
    grezza = 0.05 * t + 2.0 * np.sin(2 * np.pi * t / PERIODO) + rng.normal(0, 0.5, n)
    # la tendenza gonfia ogni confronto: si toglie, come fa il capitolo
    return grezza - np.polyval(np.polyfit(t, grezza, 1), t)


def autocorr(x, ritardo):
    """La stessa funzione del capitolo, riga per riga."""
    x = x - x.mean()
    if ritardo == 0:
        return 1.0
    return float((x[ritardo:] * x[:-ritardo]).sum() / (x * x).sum())


SERIE = serie_del_capitolo()
COEFF = [autocorr(SERIE, k) for k in range(RITARDO_MAX + 1)]

# La fascia del caso: rimescolando, il coefficiente oscilla di 1/radice(n), e
# sotto il doppio di quell'oscillazione un valore non si distingue da zero.
FASCIA = 2.0 / math.sqrt(N)

# --- le asserzioni, che difendono quello che la didascalia promette ---------

for _k, _atteso in ATTESI.items():
    if abs(round(COEFF[_k], 3) - _atteso) > 5e-4:
        raise AssertionError(
            f"a ritardo {_k} la figura calcola {COEFF[_k]:.3f}, la pagina "
            f"stampa {_atteso:.3f}: uno dei due è cambiato senza l'altro")

# la didascalia dice «a mezzo ciclo capovolte»: il minimo deve stare lì
_minimo = min(range(1, RITARDO_MAX + 1), key=lambda k: COEFF[k])
if _minimo != PERIODO // 2:
    raise AssertionError(f"il minimo cade a ritardo {_minimo}, non a "
                         f"{PERIODO // 2}: la scena non mostra più il capovolgimento")
if COEFF[PERIODO // 2] > -FASCIA:
    raise AssertionError("a mezzo ciclo la somiglianza non è negativa per davvero")

# e «a un ciclo intero tornano a sovrapporsi»: il massimo dopo il ritardo 0
_massimo = max(range(1, RITARDO_MAX + 1), key=lambda k: COEFF[k])
if _massimo != PERIODO:
    raise AssertionError(f"il massimo cade a ritardo {_massimo}, non a {PERIODO}: "
                         f"la figura non mostra più il ritorno del ciclo")

# la fascia serve solo se qualcosa la supera e qualcosa ci sta dentro
if not any(abs(c) > FASCIA for c in COEFF[1:]):
    raise AssertionError("nessun coefficiente esce dalla fascia del caso")
if not any(abs(c) <= FASCIA for c in COEFF[1:]):
    raise AssertionError("nessun coefficiente cade dentro la fascia del caso")


def costruisci() -> Figura:
    x0, larg = 60.0, 600.0
    px = larg / FINESTRA                      # pixel per osservazione

    # ---- pannello di sopra: la serie e la sua copia ------------------------
    ya, alta = 44.0, 118.0
    ultimo = RITARDO_MAX + FINESTRA          # l'ultimo punto che serve disegnare
    ampiezza = max(abs(v) for v in SERIE[:ultimo + 1])

    def sx(i):
        """Il punto i della serie, con il ritardo massimo all'inizio del riquadro."""
        return x0 + (i - RITARDO_MAX) * px

    def sy(v):
        return ya + alta / 2 - v / ampiezza * (alta / 2 - 6)

    def spezzata(primo, ultimo, sposta=0):
        return " ".join(f"{sx(i + sposta):.1f},{sy(SERIE[i]):.1f}"
                        for i in range(primo, ultimo + 1))

    corpo = [
        f'<text class="lbl" x="{x0:.0f}" y="{ya - 14:.0f}">'
        f'la serie, e la stessa serie in ritardo di qualche passo</text>',
        f'<line class="zero" x1="{x0:.0f}" y1="{sy(0):.1f}" '
        f'x2="{x0 + larg:.0f}" y2="{sy(0):.1f}"/>',
        f'<clipPath id="dentro"><rect x="{x0:.0f}" y="{ya - 6:.0f}" '
        f'width="{larg:.0f}" height="{alta + 12:.0f}"/></clipPath>',
        f'<polyline class="serie" points="{spezzata(RITARDO_MAX, ultimo)}"/>',
    ]

    # La copia è disegnata dove finisce, cioè al ritardo massimo, e l'animazione
    # la porta lì partendo da sovrapposta: il riposo non dipende dal CSS. È
    # lunga un ritardo massimo in più della finestra, così copre il riquadro
    # anche quando è tutta scivolata a sinistra.
    corpo.append(f'<polyline class="copia" clip-path="url(#dentro)" '
                 f'points="{spezzata(0, ultimo, sposta=RITARDO_MAX)}" '
                 f'style="animation:scivola var(--d) infinite"/>')

    tappe = []
    for k in range(RITARDO_MAX + 1):
        t0, t1 = sosta(k, RITARDO_MAX + 1, tenuta=0.55)
        dx = (k - RITARDO_MAX) * px
        tappe += [(t0, f"transform:translateX({dx:.1f}px)"),
                  (t1, f"transform:translateX({dx:.1f}px)")]
    tappe.append((100.0, "transform:translateX(0px)"))
    anim = [keyframes("scivola", tappe)]

    # ---- pannello di sotto: il correlogramma -------------------------------
    yb, altb = 232.0, 150.0
    mezzo = yb + altb / 2
    passo = larg / RITARDO_MAX

    def sy2(v):
        return mezzo - v * (altb / 2 - 8)

    corpo += [
        f'<text class="lbl" x="{x0:.0f}" y="{yb - 16:.0f}">'
        f'quanto si somigliano, a ogni ritardo</text>',
        f'<text class="lbs" x="{x0 + larg:.0f}" y="{yb - 16:.0f}" '
        f'text-anchor="end">nella fascia grigia non si distingue dal caso</text>',
        f'<rect class="fascia" x="{x0 - 6:.0f}" y="{sy2(FASCIA):.1f}" '
        f'width="{larg + 12:.0f}" height="{sy2(-FASCIA) - sy2(FASCIA):.1f}"/>',
        f'<line class="zero" x1="{x0 - 6:.0f}" y1="{mezzo:.1f}" '
        f'x2="{x0 + larg + 6:.0f}" y2="{mezzo:.1f}"/>',
    ]

    for k, c in enumerate(COEFF):
        t0, _ = sosta(k, RITARDO_MAX + 1, tenuta=0.55)
        prima = max(t0 - 0.01, 0.0)
        anim.append(keyframes(f"b{k}", [(0.0, "opacity:0"), (prima, "opacity:0"),
                                        (t0, "opacity:1"), (100.0, "opacity:1")]))
        x = x0 + k * passo
        y1, y2 = (sy2(c), mezzo) if c >= 0 else (mezzo, sy2(c))
        corpo.append(f'<line class="barra" x1="{x:.1f}" y1="{y1:.1f}" '
                     f'x2="{x:.1f}" y2="{y2:.1f}" '
                     f'style="animation:b{k} var(--d) infinite"/>')
        corpo.append(f'<circle class="punto" cx="{x:.1f}" cy="{sy2(c):.1f}" r="3.5" '
                     f'style="animation:b{k} var(--d) infinite"/>')
        if k % 6 == 0:
            corpo.append(f'<text class="lbs" x="{x:.1f}" y="{yb + altb + 20:.0f}" '
                         f'text-anchor="middle">{k}</text>')

    corpo += [
        f'<text class="lbs" x="{x0 + larg / 2:.0f}" y="{yb + altb + 40:.0f}" '
        f'text-anchor="middle">di quanti passi è in ritardo la copia</text>',
    ]

    return Figura(
        larghezza=700, altezza=yb + altb + 56,
        alt="In alto una serie temporale ondulata e una sua copia di un altro "
            "colore, che scivola verso destra un passo alla volta: "
            "quando lo scorrimento è di sei passi le due onde sono capovolte, "
            "cresta contro avvallamento, e quando è di dodici tornano a "
            "sovrapporsi. In basso un correlogramma si riempie di pari passo, "
            "una barra per ogni scorrimento: parte da uno, scende fino a un "
            "minimo profondamente negativo a sei, risale a un massimo a dodici "
            "e ripete l'onda più smorzata fino a ventiquattro. Una fascia "
            "grigia chiara attorno allo zero segna i valori che non si "
            "distinguono dal caso.",
        corpo="".join(corpo),
        stile=f"""    .serie {{ fill:none; stroke:{TEAL}; stroke-width:2; }}
    .copia {{ fill:none; stroke:{TERRACOTTA}; stroke-width:2; }}
    .zero {{ stroke:{BORDER_STRONG}; stroke-width:1; }}
    .fascia {{ fill:{BORDER}; }}
    .barra {{ stroke:{OCRA}; stroke-width:3; }}
    .punto {{ fill:{OCRA}; }}""",
        animazioni=anim,
        durata=(RITARDO_MAX + 1) * 0.42,
        fermi=".copia, .barra, .punto",
    )
