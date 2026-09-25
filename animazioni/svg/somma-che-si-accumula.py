"""La somma cumulata di Page (CUSUM) che si accumula dopo un cambio di regime.

È la scena di `MLOps/monitoring-e-drift.md`: ogni giorno arriva un errore,
che fino al giorno CAMBIO oscilla attorno a zero e poi attorno a DELTA. La
somma S_t = max(0, S_{t-1} + x_t - K) si traccia man mano, e l'allarme scatta
al primo giorno in cui supera H. I parametri sono quelli del blocco di codice
della pagina con la soglia più alta (salto di mezza deviazione standard,
K = 0,25, H = 8): la figura è un flusso costruito con quei parametri, con la
quiete e il cambio nella stessa serie (il blocco li misura separati).

Il sorteggio è `numpy.random.default_rng(SEME)`. La didascalia promette
quattro cose e gli assert le difendono: prima del cambio la somma sale e ricade
senza arrivare alla soglia, dopo il cambio la supera, ci mette qualche
settimana (il blocco della pagina stima 27,6 giorni in media, contando dal
primo giorno cambiato; un flusso singolo può scostarsene di parecchio, e
l'intervallo dell'assert lo dice), e dopo il cambio torna a zero un'ultima
volta, qualche giorno dopo il cambio vero: il giorno dopo quell'ultimo zero è
la stima di massima verosimiglianza dell'inizio del cambio. Il seme non è il
primo: con SEME = 0 un falso allarme scatta prima del cambio, che a questa
soglia capita in circa il 6% dei flussi, e la figura racconterebbe un'altra
cosa. Il disegno fermo è lo stato finale: tutti i giorni visibili, la
somma intera, il segno di allarme acceso.
"""

import numpy as np

from paithon_svg import *

NOME = "somma-che-si-accumula"
TITOLO = "la somma cumulata si accumula dopo un cambio di regime"

SEME = 2
GIORNI = 140
CAMBIO = 60
DELTA = 0.5
K = DELTA / 2
H = 8.0
DOPO = 8                  # giorni disegnati dopo l'allarme

X0, X1 = 80, 700
ALTO_Y0, ALTO_H = 40, 120          # fascia dei dati, y in [-3, 3.5]
BASSO_Y0, BASSO_H = 200, 150       # fascia della somma, S in [0, SMAX]
SMAX = 11.0


FINE = None


def serie():
    rng = np.random.default_rng(SEME)
    x = rng.normal(0, 1, GIORNI)
    x[CAMBIO:] += DELTA
    S, s = [], 0.0
    for v in x:
        s = max(0.0, s + v - K)
        S.append(s)
    S = np.array(S)
    allarme = int(np.argmax(S > H)) if (S > H).any() else None
    return x, S, allarme


def gx(t):
    return X0 + t / FINE * (X1 - X0)


def gy_alto(v):
    return ALTO_Y0 + (3.5 - v) / 6.5 * ALTO_H


def gy_basso(s):
    return BASSO_Y0 + (SMAX - s) / SMAX * BASSO_H


def costruisci() -> Figura:
    global FINE
    x, S, allarme = serie()
    assert allarme is not None and allarme > CAMBIO, "allarme prima del cambio, o mai"
    assert S[:CAMBIO].max() > 1.0, "prima del cambio la somma non sale mai: la scena non lo mostra"
    assert S[:CAMBIO].max() < H * 0.6, "prima del cambio la somma arriva troppo vicino alla soglia"
    # «sale e ricade»: prima del cambio la somma torna a zero più volte
    assert (S[1:CAMBIO] == 0).sum() >= 5, "prima del cambio la somma non ricade"
    assert 14 <= allarme - CAMBIO <= 42, f"ritardo {allarme - CAMBIO}: non è 'qualche settimana'"
    # l'alt lo dice: dopo il cambio la somma torna a zero ancora una volta, poi sale
    assert (S[CAMBIO:allarme] == 0).any(), "dopo il cambio la somma non torna mai a zero"
    # la didascalia: il giorno dopo l'ultimo zero stima l'inizio del cambio,
    # «qualche giorno dopo quello vero»
    ultimo_zero = max(t for t in range(allarme) if S[t] == 0)
    assert 2 <= ultimo_zero + 1 - CAMBIO <= 10, f"stima del cambio a {ultimo_zero + 1 - CAMBIO} giorni"
    fine = FINE = allarme + DOPO
    assert fine < GIORNI
    assert np.all((-3 < x[:fine + 1]) & (x[:fine + 1] < 3.5)), "un punto esce dalla fascia"
    assert S[:fine + 1].max() < SMAX, "la somma esce dalla fascia"

    corpo, anim = [], []
    frazione = lambda t: t / fine
    corpo.append(f'<clipPath id="avanza"><rect id="avanza-r" x="{X0 - 6}" y="0" '
                 f'width="{gx(fine) - X0 + 12:.1f}" height="360" '
                 f'style="animation:avanza var(--d) linear infinite"/></clipPath>')
    # le due fasce
    corpo.append(f'<line class="asse" x1="{X0}" y1="{gy_alto(0):.1f}" x2="{X1}" y2="{gy_alto(0):.1f}"/>')
    corpo.append(f'<line class="asse" x1="{X0}" y1="{gy_basso(0):.1f}" x2="{X1}" y2="{gy_basso(0):.1f}"/>')
    corpo.append(f'<line class="soglia" x1="{X0}" y1="{gy_basso(H):.1f}" x2="{X1}" y2="{gy_basso(H):.1f}"/>')
    corpo.append(f'<text class="lbs" x="{X1 + 6}" y="{gy_basso(H) + 4:.1f}">soglia</text>')
    corpo.append(f'<text class="lbs" x="{X1 + 6}" y="{gy_basso(H) + 20:.1f}">h = 8</text>')
    corpo.append(f'<line class="cambio" x1="{gx(CAMBIO):.1f}" y1="{ALTO_Y0 - 6}" '
                 f'x2="{gx(CAMBIO):.1f}" y2="{gy_basso(0) + 6:.1f}"/>')
    corpo.append(f'<text class="lbs" x="{gx(CAMBIO) + 6:.1f}" y="{ALTO_Y0 + 4}">qui cambia</text>')
    corpo.append(f'<text class="lbs" x="{X0 - 10}" y="{gy_alto(0) + 4:.1f}" text-anchor="end">errore</text>')
    corpo.append(f'<text class="lbs" x="{X0 - 10}" y="{gy_basso(0) + 4:.1f}" text-anchor="end">somma</text>')
    # i dati e la somma, scoperti dal ritaglio che avanza
    punti = "".join(f'<circle class="dato" cx="{gx(t):.1f}" cy="{gy_alto(v):.1f}" r="2.6"/>'
                    for t, v in enumerate(x[:fine + 1]))
    linea = " ".join(f"{gx(t):.1f},{gy_basso(s):.1f}" for t, s in enumerate(S[:fine + 1]))
    corpo.append(f'<g clip-path="url(#avanza)">{punti}'
                 f'<polyline class="somma" points="{linea}"/></g>')
    # l'allarme, che si accende quando il ritaglio lo raggiunge: per questo il
    # ritaglio avanza `linear`, perché con l'`ease` di default la sua posizione
    # non sarebbe più proporzionale al tempo e l'allarme si accenderebbe fuori
    # sincrono (e i fermi cadrebbero su giorni diversi da quelli scelti)
    ax, ay = gx(allarme), gy_basso(S[allarme])
    quando = 8 + 80 * frazione(allarme)
    anim.append(keyframes("accendi", [(0.0, "opacity:0"), (quando - 0.5, "opacity:0"),
                                      (quando, "opacity:1"), (100.0, "opacity:1")]))
    corpo.append(f'<g class="acceso" style="animation:accendi var(--d) infinite">'
                 f'<circle class="allarme" cx="{ax:.1f}" cy="{ay:.1f}" r="7"/>'
                 f'<text class="all" x="{ax:.1f}" y="{ay - 14:.1f}" text-anchor="middle">allarme</text></g>')
    anim.append(keyframes("avanza", [(0.0, "transform:scaleX(0)"), (8.0, "transform:scaleX(0)"),
                                     (88.0, "transform:scaleX(1)"), (100.0, "transform:scaleX(1)")]))

    return Figura(
        larghezza=780, altezza=370,
        alt="Animazione in due fasce sovrapposte. In alto, un giorno dopo l'altro "
            "compaiono puntini teal che oscillano attorno a una linea, e dopo una "
            "linea verticale segnata qui cambia si spostano appena verso l'alto; in "
            "basso, una spezzata terracotta traccia la somma cumulata, che prima del "
            "cambio sale e ricade restando sotto una linea tratteggiata, la soglia, e "
            "dopo il cambio, tornata a zero un'ultima volta, sale fino a "
            "oltrepassarla, dove compare un segno di allarme.",
        corpo="".join(corpo),
        stile=f"""    .asse    {{ stroke:{BORDER_STRONG}; stroke-width:1.2; }}
    .soglia  {{ stroke:{INK}; stroke-width:1.4; stroke-dasharray:6 5; }}
    .cambio  {{ stroke:{FG_MUTED}; stroke-width:1.2; stroke-dasharray:2 4; }}
    .dato    {{ fill:{TEAL}; opacity:.8; }}
    .somma   {{ fill:none; stroke:{TERRACOTTA}; stroke-width:2.6; stroke-linejoin:round; }}
    .allarme {{ fill:{OCRA}; stroke:{INK}; stroke-width:1.2; }}
    .all     {{ font-family:{SANS}; font-size:14px; font-weight:700; fill:{INK}; }}
    #avanza-r {{ transform-box:fill-box; transform-origin:0% 50%; }}""",
        animazioni=anim,
        durata=8.0,
        fermi="#avanza-r, .acceso",
    )
