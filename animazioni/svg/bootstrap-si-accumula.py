"""Il bootstrap che accumula: da un campione solo, una distribuzione.

I dati sono i sessanta stipendi di `MachineLearning/il-bootstrap.md`, generati
qui con lo stesso seme e la stessa formula: se il capitolo cambia i numeri,
`verifica` se ne accorge e la figura non si genera.

Quello che il fermo immagine non può mostrare, ed e' tutta la ragione di questa
clip, e' il **gesto**: il campione resta sempre quello, e a cambiare e' solo
quali dei suoi punti vengono pescati (con reimmissione, quindi qualcuno due
volte e qualcuno mai). Ogni pescata produce una mediana, la mediana cade
nell'istogramma, e l'istogramma cresce. Da una fotografia sola, mille
fotografie.

Lo stato di riposo e' l'istogramma completo con l'intervallo al 95%: chi non
anima (stampa, PDF, prefers-reduced-motion) vede la conclusione, cioe' la cosa
che serve davvero, e le barre sono disegnate alla loro altezza finale con
coordinate vere.
"""

import numpy as np

from paithon_svg import *

NOME = "bootstrap-si-accumula"
TITOLO = "Il bootstrap accumula: da un campione solo, una distribuzione"

# --- gli stessi dati del capitolo, rifatti qui invece che trascritti ---------
N, GIRI, CELLE = 60, 10_000, 20
TAPPE = [1, 3, 8, 25, 80, 300, 1200, GIRI]   # quanti ricampionamenti a ogni scatto

# Quello che il capitolo dichiara: se cambia, la figura non si genera.
ATTESI = {"mediana": 29282.0, "lo": 25720.0, "hi": 32885.0}


def campione() -> np.ndarray:
    r = np.random.default_rng(0)
    return np.round(np.exp(r.normal(np.log(28_000), 0.45, N)))


def sorteggi() -> np.ndarray:
    """Gli indici pescati a ogni giro: (GIRI, N), con reimmissione.

    Uno solo, condiviso: cosi' i punti che il pannello di sinistra accende sono
    esattamente quelli che hanno prodotto la mediana caduta a destra.
    """
    return np.random.default_rng(0).integers(0, N, (GIRI, N))


def verifica(dati, med, lo, hi, idx) -> None:
    """I numeri della figura sono quelli stampati nel capitolo?"""
    assert len(dati) == N, f"il capitolo usa {N} stipendi, qui {len(dati)}"
    for nome, atteso, avuto in (("mediana", ATTESI["mediana"], np.median(dati)),
                                ("lo", ATTESI["lo"], lo), ("hi", ATTESI["hi"], hi)):
        assert abs(avuto - atteso) < 1, \
            f"{nome}: la figura calcola {avuto:.0f}, il capitolo stampa {atteso:.0f}"
    # e la promessa dell'alt: l'intervallo contiene la mediana del campione
    assert lo < np.median(dati) < hi, "l'intervallo deve contenere la mediana osservata"
    # con la reimmissione, a ogni giro qualcuno resta fuori: e' il punto del disegno
    fuori = [N - len(set(p)) for p in idx[:20]]
    assert min(fuori) > 10, f"troppo pochi esclusi per giro: {min(fuori)}"
    # l'alt dice «campana stretta e quasi simmetrica»: la coda a destra ce
    # l'hanno gli stipendi (pannello di sinistra), non le loro mediane
    asimmetria = ((med - med.mean())**3).mean() / med.std()**3
    assert abs(asimmetria) < 0.1, \
        f"l'alt promette una campana simmetrica, ma l'asimmetria vale {asimmetria:.3f}"


def costruisci() -> Figura:
    dati = campione()
    idx = sorteggi()
    med = np.median(dati[idx], axis=1)
    lo, hi = np.percentile(med, [2.5, 97.5])
    verifica(dati, med, lo, hi, idx)

    bordi = np.linspace(med.min(), med.max(), CELLE + 1)
    finale, _ = np.histogram(med, bins=bordi)
    conteggi = [np.histogram(med[:t], bins=bordi)[0] for t in TAPPE]

    n = len(TAPPE)
    passo = 100.0 / n
    corpo, anim = [], []

    # ---- pannello di sinistra: il campione, sempre lo stesso ---------------
    rs = Riquadro(x=52, y=76, larg=250, alt=250,
                  xmin=dati.min() * 0.92, xmax=dati.max() * 1.03,
                  ymin=-0.6, ymax=N / 6.0)
    corpo.append(rs.cornice())
    corpo.append(f'<text class="ttl" x="{rs.x}" y="{rs.y - 34}">'
                 f'il campione, uno solo</text>')
    corpo.append(f'<text class="lbs" x="{rs.x}" y="{rs.y - 14}">'
                 f'{N} stipendi, che non cambiano mai</text>')

    # i punti si dispongono a colonne, per non sovrapporsi
    ordinati = np.argsort(dati)
    riga = {}
    posti = {}
    for i in ordinati:
        col = int((dati[i] - rs.xmin) / (rs.xmax - rs.xmin) * 26)
        riga[col] = riga.get(col, -1) + 1
        posti[i] = (dati[i], riga[col] * 0.9)

    for i in range(N):
        px, py = posti[i]
        # quante volte l'esempio i e' stato pescato all'ULTIMA pescata di ogni
        # scatto: e' la pescata che ha prodotto la mediana appena caduta a destra
        volte = [int((idx[t - 1] == i).sum()) for t in TAPPE]
        tappe = []
        for s, v in enumerate(volte):
            t0, t1 = sosta(s, n)
            colore = (FG_MUTED if v < 0 else TEAL if v == 0 else TERRACOTTA)
            op = "0.28" if v == 0 else "0.9"
            r_ = 3.0 if v <= 0 else 3.0 + 1.6 * min(v, 3)
            tappe += [(max(t0 - 0.4, 0.01), tappe[-1][1] if tappe else
                       f"fill:{colore};opacity:{op};r:{r_:.1f}px"),
                      (t0, f"fill:{colore};opacity:{op};r:{r_:.1f}px"),
                      (t1, f"fill:{colore};opacity:{op};r:{r_:.1f}px")]
        tappe += [(100.0, f"fill:{FG_MUTED};opacity:0.55;r:3.0px")]
        nome = f"pt{i}"
        anim.append(keyframes(nome, tappe))
        corpo.append(f'<circle class="pt" cx="{rs.sx(px):.1f}" cy="{rs.sy(py):.1f}" '
                     f'r="3" fill="{FG_MUTED}" opacity="0.55" '
                     f'style="animation:{nome} var(--d) infinite"/>')

    corpo += [f'<text class="lbs" x="{rs.x}" y="{rs.y + rs.alt + 22}">'
              f'pescato una volta</text>',
              f'<circle class="pt" cx="{rs.x - 10:.0f}" cy="{rs.y + rs.alt + 17}" '
              f'r="4.6" fill="{TERRACOTTA}" opacity="0.9"/>',
              f'<text class="lbs" x="{rs.x}" y="{rs.y + rs.alt + 42}">'
              f'rimasto fuori</text>',
              f'<circle class="pt" cx="{rs.x - 10:.0f}" cy="{rs.y + rs.alt + 37}" '
              f'r="3" fill="{TEAL}" opacity="0.28"/>']

    # ---- pannello di destra: le mediane che si accumulano ------------------
    rd = Riquadro(x=372, y=76, larg=290, alt=250,
                  xmin=bordi[0], xmax=bordi[-1], ymin=0, ymax=finale.max() * 1.06)
    corpo.append(rd.cornice())
    corpo.append(f'<text class="ttl" x="{rd.x}" y="{rd.y - 34}">'
                 f'le mediane, una per pescata</text>')

    larg = (rd.larg / CELLE) - 1.4
    for c in range(CELLE):
        h = finale[c]
        if h == 0:
            continue
        bx = rd.sx(bordi[c]) + 0.7
        # disegnata all'altezza FINALE, con coordinate vere; l'animazione la
        # comprime all'indietro e la lascia sull'identita'
        by, bh = rd.sy(h), rd.sy(0) - rd.sy(h)
        tappe = []
        for s, cont in enumerate(conteggi):
            t0, t1 = sosta(s, n)
            # ogni fotogramma e' normalizzato al proprio massimo: in scala
            # assoluta le prime pescate sarebbero invisibili, e la cosa da
            # guardare non e' l'altezza, e' la FORMA che si stabilizza
            k = (cont[c] / max(cont.max(), 1)) * (finale.max() / h)
            v = f"transform:scaleY({k:.4f})"
            tappe += [(t0, v), (t1, v)]
        tappe.append((100.0, "transform:scaleY(1)"))
        nome = f"ba{c}"
        anim.append(keyframes(nome, tappe))
        dentro = bordi[c] >= lo and bordi[c + 1] <= hi
        corpo.append(
            f'<rect class="bar" x="{bx:.1f}" y="{by:.1f}" width="{larg:.1f}" '
            f'height="{bh:.1f}" fill="{TERRACOTTA if dentro else OCRA}" '
            f'style="transform-origin:{bx + larg/2:.1f}px {rd.sy(0):.1f}px;'
            f'animation:{nome} var(--d) infinite"/>')

    # l'intervallo al 95%, che e' la conclusione: compare all'ultimo scatto
    t_ult, _ = sosta(n - 1, n)
    anim.append(keyframes("cin", [(0.0, "opacity:0"), (max(t_ult - 0.4, 0.01), "opacity:0"),
                                  (t_ult, "opacity:1"), (100.0, "opacity:1")]))
    yb = rd.y + rd.alt + 14
    corpo.append(
        f'<g style="animation:cin var(--d) infinite">'
        f'<line class="int" x1="{rd.sx(lo):.1f}" y1="{yb}" x2="{rd.sx(hi):.1f}" y2="{yb}"/>'
        f'<line class="int" x1="{rd.sx(lo):.1f}" y1="{yb-6}" x2="{rd.sx(lo):.1f}" y2="{yb+6}"/>'
        f'<line class="int" x1="{rd.sx(hi):.1f}" y1="{yb-6}" x2="{rd.sx(hi):.1f}" y2="{yb+6}"/>'
        f'<text class="lbs" x="{rd.sx(lo):.1f}" y="{yb+24}" text-anchor="middle">'
        f'{lo:.0f}</text>'
        f'<text class="lbs" x="{rd.sx(hi):.1f}" y="{yb+24}" text-anchor="middle">'
        f'{hi:.0f}</text>'
        f'<text class="lbl" x="{(rd.sx(lo)+rd.sx(hi))/2:.1f}" y="{yb+44}" '
        f'text-anchor="middle">intervallo al 95%</text></g>')

    # ---- il contatore: quante pescate finora -------------------------------
    for s, t in enumerate(TAPPE):
        t0, t1 = sosta(s, n)
        fermo = ";opacity:1" if s == n - 1 else ""
        tappe = [(0.0, "opacity:0"), (max(t0 - 0.4, 0.01), "opacity:0"), (t0, "opacity:1"),
                 (t1 + passo * 0.4, "opacity:1"), (min(t1 + passo * 0.45, 99.9), "opacity:0"),
                 (100.0, "opacity:0")] if s < n - 1 else [
                 (0.0, "opacity:0"), (max(t0 - 0.4, 0.01), "opacity:0"),
                 (t0, "opacity:1"), (100.0, "opacity:1")]
        anim.append(keyframes(f"cnt{s}", tappe))
        etichetta = "1 pescata" if t == 1 else f"{t:,} pescate".replace(",", ".")
        # a riposo si vede solo l'ultimo: il disegno fermo e' lo stato finale
        spento = "" if s == n - 1 else ' opacity="0"'
        corpo.append(f'<text class="cnt" x="{rd.x + rd.larg}" y="{rd.y - 14}" '
                     f'text-anchor="end"{spento} style="animation:cnt{s} var(--d) '
                     f'infinite{fermo}">{etichetta}</text>')

    return Figura(
        larghezza=700, altezza=420, durata=11.0,
        alt=(f"A sinistra i {N} stipendi del campione, sempre gli stessi, disposti "
             "in colonnine lungo un asse orizzontale. A ogni scatto alcuni di essi "
             "si accendono in terracotta perche' sono stati pescati, e diventano "
             "piu' grandi se pescati piu' volte, mentre quelli rimasti fuori "
             "restano pallidi: e' il ricampionamento con reimmissione. A destra "
             "un istogramma cresce di scatto in scatto, una barra per ogni "
             "mediana calcolata, da una sola pescata fino a "
             # il .replace va applicato SOLO a questo pezzo: attaccato alla
             # catena di letterali qui sopra girava su tutta la frase e le
             # mangiava sei virgole
             + f"{GIRI:,}".replace(",", ".") +
             " pescate, e prende una forma a campana stretta e quasi simmetrica. "
             f"Alla fine compare sotto l'istogramma l'intervallo al 95 per cento, "
             f"da {lo:.0f} a {hi:.0f}, che e' la risposta cercata: quanto balla "
             "la mediana."),
        corpo="".join(corpo),
        stile=f"""    .pt  {{ }}
    .bar {{ }}
    .int {{ stroke:{INK}; stroke-width:2; stroke-linecap:round; }}
    .cnt {{ font-family:{SANS}; font-size:14px; font-weight:600; fill:{TERRACOTTA}; }}""",
        animazioni=anim,
        fermi=".pt, .bar, .cnt, g")
