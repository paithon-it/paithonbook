"""La finestra che scorre sul segnale e riempie lo spettrogramma (STFT).

Il tempo è il contenuto due volte: la finestra avanza sul segnale, e ogni sua
posizione diventa **una** colonna dello spettrogramma. Una figura ferma mostra
il risultato e nasconde il meccanismo.

Nulla è disegnato a mano. Il segnale è sintetizzato qui (tre note che salgono,
ciascuna con la sua armonica, più un fondo di rumore), la finestra di Hann è
applicata davvero, e la colonna è il modulo della DFT calcolata con
`numpy.fft.rfft`. Se il segnale cambia frequenza a metà, lo spettrogramma lo
mostra perché lo ha misurato.

I parametri sono quelli del capitolo (`Audio/dal-suono-alle-feature.md`, il
blocco `librosa`): $f_s = 16$ kHz, finestra di 400 campioni (25 ms), passo di
160 (10 ms).

Due scelte di colore, e sono una convenzione: **teal = i dati** (la forma
d'onda e le celle dello spettrogramma sono la stessa cosa misurata due volte),
**terracotta = il meccanismo che si muove** (la finestra e la colonna che ne
esce).

Le intensità sono quantizzate in cinque livelli di opacità del teal: non è una
scorciatoia grafica, è ciò che tiene il file sotto i 15 KB, perché le celle
contigue allo stesso livello diventano un solo tratto di path.
"""

import numpy as np

from paithon_svg import *

NOME = "finestra-spettrogramma"
TITOLO = "la finestra scorre e riempie lo spettrogramma"

# --- parametri della STFT: gli stessi del capitolo -------------------------
FS = 16000          # frequenza di campionamento, Hz
N_FFT = 400         # finestra: 400 campioni = 25 ms
HOP = 160           # passo: 160 campioni = 10 ms
N_COL = 18          # quante finestre si vedono scorrere
N_BANDE = 12        # bande di frequenza mostrate...
F_MAX = 3000        # ... da 0 a 3 kHz, 250 Hz l'una
PAVIMENTO_DB = 40.0  # sotto questa soglia la cella non si disegna

# Tre note che salgono: (inizio ms, fine ms, decadimento, [(Hz, ampiezza)]).
# L'ultima dura oltre la fine del brano e decade piano: così l'**ultima**
# finestra, che è lo stato di riposo della figura, trova ancora del segnale e
# la sua colonna si legge da sola.
NOTE = [
    (0, 72, 2.4, [(400, 1.00), (800, 0.40)]),
    (68, 140, 2.4, [(1100, 0.95), (2200, 0.36)]),
    (136, 230, 1.1, [(2600, 0.95)]),
]
RUMORE = 0.008
SEME = 3

# --- geometria -------------------------------------------------------------
X0 = 76             # x dell'istante t = 0
PX_HOP = 28         # pixel per passo (10 ms): la colonna è larga così
ONDA_Y, ONDA_H = 66, 96
SPEC_Y, BANDA_H = 222, 16
LIVELLI = [0.14, 0.32, 0.52, 0.74, 0.96]


def segnale() -> np.ndarray:
    """Le tre note, con attacco e rilascio dolci per non produrre clic."""
    n_tot = (N_COL - 1) * HOP + N_FFT
    t = np.arange(n_tot) / FS
    x = np.zeros(n_tot)
    for t0, t1, dec, righe in NOTE:
        u = (t - t0 / 1000) / ((t1 - t0) / 1000)
        dentro = (u >= 0) & (u <= 1)
        # seno rialzato: sale e scende senza discontinuità; l'esponenziale dà
        # il decadimento del pizzicato, così l'inviluppo si legge nell'onda
        env = np.where(dentro, np.sin(np.pi * np.clip(u, 0, 1)) ** 0.5, 0.0)
        env = env * np.exp(-dec * np.clip(u, 0, 1))
        for f, a in righe:
            x += a * env * np.sin(2 * np.pi * f * t)
    x += RUMORE * np.random.default_rng(SEME).standard_normal(n_tot)
    return x / np.abs(x).max()


def spettrogramma(x: np.ndarray) -> np.ndarray:
    """STFT vera: finestra di Hann, rfft, energia raccolta in N_BANDE bande.

    Restituisce una matrice (N_COL, N_BANDE) di valori in [0, 1], dove 1 è il
    massimo dell'intera figura e 0 il pavimento a −40 dB.
    """
    w = np.hanning(N_FFT + 1)[:-1]          # Hann periodica, come in analisi
    bordi = np.linspace(0, F_MAX, N_BANDE + 1)
    freq = np.fft.rfftfreq(N_FFT, 1 / FS)
    pot = np.zeros((N_COL, N_BANDE))
    for i in range(N_COL):
        pezzo = x[i * HOP: i * HOP + N_FFT] * w
        spettro = np.abs(np.fft.rfft(pezzo)) ** 2
        for b in range(N_BANDE):
            sel = (freq >= bordi[b]) & (freq < bordi[b + 1])
            pot[i, b] = spettro[sel].sum()
    db = 10 * np.log10(pot / pot.max() + 1e-12)
    return np.clip((db + PAVIMENTO_DB) / PAVIMENTO_DB, 0.0, 1.0)


def tratti(col: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    """Per una colonna: per ogni livello, le corse verticali contigue.

    Fondere le celle contigue allo stesso livello in un unico rettangolo è la
    ragione per cui la figura pesa poco: uno spettrogramma è fatto di macchie,
    non di celle indipendenti.
    """
    liv = [min(int(v * len(LIVELLI)), len(LIVELLI) - 1) if v > 0.06 else -1
           for v in col]
    fuori: dict[int, list[tuple[int, int]]] = {}
    b = 0
    while b < N_BANDE:
        k, e = liv[b], b
        while e + 1 < N_BANDE and liv[e + 1] == k:
            e += 1
        if k >= 0:
            fuori.setdefault(k, []).append((b, e - b + 1))
        b = e + 1
    return fuori


def costruisci() -> Figura:
    x = segnale()
    S = spettrogramma(x)

    spec_h = N_BANDE * BANDA_H
    spec_x0 = X0 + round(0.75 * PX_HOP)          # la colonna 0 comincia qui
    onda_x1 = X0 + round(len(x) / HOP * PX_HOP)  # 195 ms = 19,5 passi
    zero_y = ONDA_Y + ONDA_H / 2
    y_fondo = SPEC_Y + spec_h
    corpo, anim = [], []

    # ---- pannello del segnale ---------------------------------------------
    corpo.append(f'<rect class="ax" x="{X0}" y="{ONDA_Y}" width="{onda_x1 - X0}" '
                 f'height="{ONDA_H}" rx="3"/>')
    corpo.append(f'<line class="axc" x1="{X0}" y1="{zero_y:.0f}" '
                 f'x2="{onda_x1}" y2="{zero_y:.0f}"/>')

    # ---- la forma d'onda ---------------------------------------------------
    # 56 secchielli: ognuno copre ~1,4 periodi della nota più grave, quel tanto
    # che basta perché il massimo non oscilli con la fase
    ampiezza = ONDA_H / 2 - 8
    n_buc, su, giu = 56, [], []
    for j in range(n_buc):
        a = int(j * len(x) / n_buc)
        b = max(int((j + 1) * len(x) / n_buc), a + 1)
        e = float(np.abs(x[a:b]).max())
        px = X0 + j * (onda_x1 - X0) / (n_buc - 1)
        su.append(f"{px:.0f},{zero_y - e * ampiezza:.0f}")
        giu.append(f"{px:.0f},{zero_y + e * ampiezza:.0f}")
    corpo.append(f'<polygon class="onda" points="{" ".join(su + giu[::-1])}"/>')

    # ---- la finestra di Hann, alla sua ultima posizione (stato di riposo) ---
    # Disegnata come ±w[n] attorno allo zero, e sopra la forma d'onda: è
    # esattamente ciò che la finestra fa al segnale, moltiplicarlo per un peso
    # che vale 1 al centro e si spegne ai bordi.
    centri = [X0 + (i + 1.25) * PX_HOP for i in range(N_COL)]
    cf = centri[-1]
    mezza = 1.25 * PX_HOP                        # 25 ms = 2,5 passi
    alto, basso = [], []
    for j in range(41):
        u = j / 40
        h = 0.5 - 0.5 * np.cos(2 * np.pi * u)    # la Hann, davvero
        px = cf - mezza + u * 2 * mezza
        alto.append(f"{px:.1f},{zero_y - h * ampiezza:.1f}")
        basso.append(f"{px:.1f},{zero_y + h * ampiezza:.1f}")
    corpo.append(f'<g id="fin"><polygon class="hann" points='
                 f'"{" ".join(alto + basso[::-1])}"/>'
                 f'<line class="bordo" x1="{cf - mezza:.0f}" y1="{ONDA_Y}" '
                 f'x2="{cf - mezza:.0f}" y2="{ONDA_Y + ONDA_H}"/>'
                 f'<line class="bordo" x1="{cf + mezza:.0f}" y1="{ONDA_Y}" '
                 f'x2="{cf + mezza:.0f}" y2="{ONDA_Y + ONDA_H}"/>'
                 f'<text class="eti" x="{cf:.0f}" y="{ONDA_Y - 10}" '
                 f'text-anchor="middle">la finestra</text></g>')

    # ---- lo spettrogramma: una colonna per finestra ------------------------
    corpo.append(f'<rect class="ax" x="{spec_x0}" y="{SPEC_Y}" '
                 f'width="{N_COL * PX_HOP}" height="{spec_h}" rx="3"/>')
    corpo.append('<g class="spec">')
    passo = 100.0 / N_COL
    for i in range(N_COL):
        t0, _ = sosta(i, N_COL)
        anim.append(keyframes(f"s{i}", [
            (0.0, "opacity:0"), (max(t0 - 0.4, 0.01), "opacity:0"),
            (min(t0 + passo * 0.22, 99.9), "opacity:1"), (100.0, "opacity:1")]))
        cx = spec_x0 + i * PX_HOP
        pezzi = []
        for k, corse in sorted(tratti(S[i]).items()):
            d = "".join(f"M{cx} {y_fondo - (b + n) * BANDA_H}"
                        f"h{PX_HOP}v{n * BANDA_H}h-{PX_HOP}z" for b, n in corse)
            pezzi.append(f'<path fill-opacity="{LIVELLI[k]}" d="{d}"/>')
        corpo.append(f'<g style="animation:s{i} var(--d) infinite">'
                     f'{"".join(pezzi)}</g>')
    corpo.append('</g>')

    # ---- il legame: dalla finestra alla sua colonna -------------------------
    corpo.append(f'<g id="col"><line class="giu" x1="{cf:.0f}" '
                 f'y1="{ONDA_Y + ONDA_H + 6}" x2="{cf:.0f}" y2="{SPEC_Y - 12}"/>'
                 f'<polygon class="punta" points="'
                 f'{cf - 5:.0f},{SPEC_Y - 13} {cf + 5:.0f},{SPEC_Y - 13} '
                 f'{cf:.0f},{SPEC_Y - 3}"/>'
                 f'<rect class="sel" x="{cf - PX_HOP / 2:.0f}" y="{SPEC_Y}" '
                 f'width="{PX_HOP}" height="{spec_h}"/>'
                 f'<text class="eti" x="{cf:.0f}" y="{y_fondo + 22}" '
                 f'text-anchor="middle">la sua colonna</text></g>')

    # ---- la finestra scorre: riposo all'ultima posizione, identità in fondo -
    tappe = []
    for i, c in enumerate(centri):
        t0, t1 = sosta(i, N_COL, tenuta=0.62)
        d = f"transform:translate({c - cf:.0f}px,0px)"
        tappe += [(t0, d), (t1, d)]
    tappe.append((100.0, "transform:translate(0px,0px)"))
    anim.append(keyframes("scorre", tappe))

    # ---- assi e didascalie --------------------------------------------------
    for b in range(0, N_BANDE + 1, 4):
        corpo.append(f'<text class="lbs" x="{spec_x0 - 10}" '
                     f'y="{y_fondo - b * BANDA_H + 4}" text-anchor="end">'
                     f'{b * F_MAX // N_BANDE // 1000}</text>')
    corpo += [
        # A sinistra non c'e' posto: appesa all'asse con text-anchor="end"
        # questa etichetta esce dal viewBox, e la figura arriva in stampa
        # tagliata. Sta sopra il grafico, allineata a sinistra come le altre.
        f'<text class="lbs" x="{X0}" y="{SPEC_Y - 12}">'
        f'in verticale: migliaia di oscillazioni al secondo</text>',
        f'<text class="lbl" x="{X0}" y="{ONDA_Y - 32}">il segnale: tre note '
        f'che salgono, in 195 millesimi di secondo</text>',
        f'<text class="lbl" x="{X0}" y="{y_fondo + 50}">'
        f'lo spettrogramma: {N_COL} finestre, {N_COL} colonne</text>',
        f'<text class="lbs" x="{X0}" y="{y_fondo + 72}">'
        f'la finestra è lunga 400 campioni (25 ms) e avanza di 160 (10 ms):'
        f'</text>',
        f'<text class="lbs" x="{X0}" y="{y_fondo + 90}">'
        f'due finestre vicine si sovrappongono, le loro colonne no</text>',
    ]

    return Figura(
        larghezza=onda_x1 + 42, altezza=y_fondo + 106,
        alt="Una finestra larga 25 millisecondi scorre sulla forma d'onda di "
            "tre note che salgono; a ogni sua posizione si accende una colonna "
            "dello spettrogramma sottostante, e la banda scura sale di gradino "
            "in gradino come le note del segnale.",
        corpo="".join(corpo),
        stile=f"""    .onda {{ fill:{TEAL}; fill-opacity:0.9; }}
    .spec path {{ fill:{TEAL}; }}
    .hann {{ fill:none; stroke:{TERRACOTTA}; stroke-width:3;
            stroke-linejoin:round; }}
    .bordo {{ stroke:{TERRACOTTA}; stroke-width:1.5; stroke-dasharray:3 3; }}
    .giu  {{ stroke:{TERRACOTTA}; stroke-width:2; stroke-dasharray:5 4; }}
    .punta {{ fill:{TERRACOTTA}; }}
    .sel  {{ fill:none; stroke:{TERRACOTTA}; stroke-width:2.5; }}
    .eti  {{ font-family:{SANS}; font-size:13px; font-weight:700;
            fill:{TERRACOTTA}; }}
    #fin, #col {{ animation:scorre var(--d) infinite; transform-box:view-box; }}""",
        animazioni=anim,
        durata=N_COL * 0.55,
        fermi="#fin, #col, .spec g",
    )
