"""Le quattro linee di base, disegnate sulla stessa serie.

La sezione «Le linee di base che bisogna sempre battere» elenca quattro
previsori volutamente stupidi e li scrive in formula, uno per punto elenco.
Disegnate sulla stessa serie sono quattro forme, e una volta viste non si
dimenticano più: una linea piatta a mezz'altezza, una linea piatta all'ultimo
valore, l'ultimo anno ricopiato in avanti, una retta che prolunga la salita.
Una forma si ricorda meglio di una formula, e le quattro insieme si
confrontano a colpo d'occhio.

**La scena è la stessa dell'esempio numerico del testo.** La serie finisce a
dicembre, la previsione va a quindici mesi, cioè al marzo dell'anno dopo il
prossimo, e il naive stagionale va a pescare il marzo di nove mesi prima: è
esattamente il conto che il testo svolge con $k = \\lfloor 14/12 \\rfloor = 1$ e
$t + 15 - 12\\cdot 2 = t - 9$. Disegnandolo si vede anche la cosa che quel conto
serve a evitare, e che nel testo si legge e non si tocca: passato un anno
intero, il ciclo che si ricicla è **sempre** l'ultimo osservato, perché quello
del calendario cadrebbe a sua volta nel futuro.

Le cinque tappe sono un ciclo solo: la serie e la sua origine, poi le quattro
linee una per volta, e a riposo tutte e quattro insieme, che è la figura che
va in stampa.

Niente numeri scritti a mano. La serie la costruisce il generatore da un seme
dichiarato (livello, salita, onda annuale con la punta a dicembre, rumore) e
`verifica()` pretende quello che la didascalia promette: che l'ultimo mese
osservato sia un dicembre, che il mese previsto sia un marzo, che il valore
ricopiato dal naive stagionale sia proprio quello di nove mesi prima, e che le
quattro linee arrivino a quota diverse abbastanza da distinguersi.
"""

import math
import random

from paithon_svg import *

NOME = "linee-di-base"
TITOLO = "Le quattro linee di base, sulla stessa serie"

# --------------------------------------------------------------------------
# La serie
# --------------------------------------------------------------------------
SEME = 11
M = 12                     # il ciclo stagionale: dodici mesi
N_OSS = 36                 # tre anni osservati, l'ultimo mese e' un dicembre
H = 15                     # l'orizzonte dell'esempio del testo
LIVELLO = 100.0
SALITA = 0.75              # per mese
AMPIEZZA = 14.0            # l'onda annuale, con la punta a dicembre
RUMORE = 2.4

MESI = ("gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
        "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre")


def mese_di(t: int) -> str:
    """Il nome del mese dell'istante t, con t = 12 che e' un dicembre."""
    return MESI[(t - 1) % M]


def serie() -> list[float]:
    """Livello piu' salita piu' onda annuale piu' rumore, da un seme solo."""
    rnd = random.Random(SEME)
    fuori = []
    for t in range(1, N_OSS + 1):
        stagione = AMPIEZZA * math.cos(2 * math.pi * (t - M) / M)
        fuori.append(LIVELLO + SALITA * t + stagione + rnd.gauss(0, RUMORE))
    return fuori


# --------------------------------------------------------------------------
# Le quattro linee di base
# --------------------------------------------------------------------------
def k_di(h: int) -> int:
    """I cicli interi che si chiudono prima dell'istante da prevedere."""
    return (h - 1) // M


def indice_stagionale(h: int) -> int:
    """L'indice (base 1) da cui il naive stagionale copia, per l'orizzonte h."""
    return N_OSS + h - M * (k_di(h) + 1)


def previsioni(y: list[float]) -> dict[str, list[float]]:
    """Le quattro classiche, per h da 1 a H."""
    media = sum(y) / len(y)
    ultimo = y[-1]
    pendenza = (y[-1] - y[0]) / (N_OSS - 1)
    return {
        "media": [media for _ in range(1, H + 1)],
        "naive": [ultimo for _ in range(1, H + 1)],
        "stagionale": [y[indice_stagionale(h) - 1] for h in range(1, H + 1)],
        "drift": [ultimo + h * pendenza for h in range(1, H + 1)],
    }


def verifica(y: list[float], prev: dict[str, list[float]]) -> None:
    """Quello che la didascalia promette, preteso sui numeri della scena."""
    assert mese_di(N_OSS) == "dicembre", \
        f"l'ultimo mese osservato e' {mese_di(N_OSS)}, la figura dice dicembre"
    assert mese_di(N_OSS + H) == "marzo", \
        f"il mese previsto e' {mese_di(N_OSS + H)}, la figura dice marzo"

    # Il conto del testo: k = 1, e si pesca nove mesi prima dell'origine.
    assert k_di(H) == 1, f"k vale {k_di(H)}, il testo dice 1"
    sorgente = indice_stagionale(H)
    assert sorgente == N_OSS - 9, \
        f"si pesca dall'indice {sorgente}, il testo dice {N_OSS - 9}"
    assert mese_di(sorgente) == "marzo", \
        f"il mese pescato e' {mese_di(sorgente)}, e doveva essere un marzo"
    assert prev["stagionale"][H - 1] == y[sorgente - 1], \
        "il naive stagionale non sta copiando il valore che la figura indica"

    # Passato un anno il ciclo si ricicla: h = 13 ripesca l'indice di h = 1.
    assert indice_stagionale(M + 1) == indice_stagionale(1), \
        "oltre il ciclo il naive stagionale non ricicla l'ultimo anno osservato"

    # Le quattro linee devono distinguersi a occhio all'arrivo.
    arrivi = sorted(p[-1] for p in prev.values())
    minimo = min(b - a for a, b in zip(arrivi, arrivi[1:]))
    ampiezza = max(y) - min(y)
    assert minimo > ampiezza * 0.03, \
        (f"due linee di base arrivano a {minimo:.1f} di distanza, meno del "
         f"tre per cento dell'ampiezza della serie: si sovrappongono")

    # La serie sale, o drift e naive sarebbero la stessa linea.
    assert prev["drift"][-1] - prev["naive"][-1] > ampiezza * 0.08, \
        "il drift non si stacca dal naive: la serie non ha abbastanza salita"

    # La punta della stagione sta a dicembre, o la storia del testo non regge.
    ultimo_anno = y[N_OSS - M:]
    assert ultimo_anno.index(max(ultimo_anno)) == M - 1, \
        (f"nell'ultimo anno il massimo cade a "
         f"{mese_di(N_OSS - M + 1 + ultimo_anno.index(max(ultimo_anno)))}, "
         f"non a dicembre")


# --------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------
LARG, ALT = 700, 452

X0, Y0 = 66, 40            # angolo alto-sinistro del riquadro
W, HH = 566, 250           # riquadro del grafico
Y_LEG = 322                # la legenda
Y_TIT, Y_RIG = 396, 420    # il racconto della tappa

COLORI = {
    "media": FG_MUTED,
    "naive": OCRA,
    "stagionale": TEAL,
    "drift": TERRACOTTA,
}
ETICHETTE = {
    "media": "la media di tutta la serie",
    "naive": "l'ultimo valore, ripetuto",
    "stagionale": "l'ultimo anno osservato, ricopiato",
    "drift": "la retta fra il primo e l'ultimo punto",
}
ORDINE = ("media", "naive", "stagionale", "drift")


def costruisci() -> Figura:
    y = serie()
    prev = previsioni(y)
    verifica(y, prev)

    tutti = y + [v for p in prev.values() for v in p]
    ymin, ymax = min(tutti), max(tutti)
    margine = (ymax - ymin) * 0.10
    ymin, ymax = ymin - margine, ymax + margine

    def sx(t: float) -> float:
        return X0 + (t - 1) / (N_OSS + H - 1) * W

    def sy(v: float) -> float:
        return Y0 + HH - (v - ymin) / (ymax - ymin) * HH

    def spezzata(punti) -> str:
        return " ".join(f"{sx(t):.1f},{sy(v):.1f}" for t, v in punti)

    corpo, anim = [], []

    # ---- il riquadro, l'origine e le due etichette di calendario ----------
    corpo.append(f'<line class="axc" x1="{X0}" y1="{Y0 + HH}" '
                 f'x2="{X0 + W}" y2="{Y0 + HH}"/>')
    x_orig = sx(N_OSS)
    corpo.append(f'<line class="org" x1="{x_orig:.1f}" y1="{Y0 - 4}" '
                 f'x2="{x_orig:.1f}" y2="{Y0 + HH}"/>')
    corpo.append(f'<text class="lbs" x="{x_orig - 14:.1f}" y="{Y0 - 12}" '
                 f'text-anchor="end">osservato, fino a {mese_di(N_OSS)}</text>')
    corpo.append(f'<text class="lbs" x="{x_orig + 14:.1f}" y="{Y0 - 12}">'
                 f'previsione, {H} mesi</text>')

    # ---- la serie osservata: c'e' fin dalla prima tappa e non si muove ----
    corpo.append(f'<polyline class="oss" points="'
                 f'{spezzata([(t + 1, v) for t, v in enumerate(y)])}"/>')

    # ---- il marzo pescato e il marzo previsto -----------------------------
    sorgente = indice_stagionale(H)
    x_s, y_s = sx(sorgente), sy(y[sorgente - 1])
    x_a, y_a = sx(N_OSS + H), sy(prev["stagionale"][H - 1])
    arco = (f'M {x_s:.1f} {y_s:.1f} C {x_s:.1f} {y_s - 58:.1f} '
            f'{x_a:.1f} {y_a - 58:.1f} {x_a:.1f} {y_a:.1f}')

    # ---- le quattro linee: una per tappa, e a riposo ci sono tutte --------
    n = 1 + len(ORDINE)                    # la serie sola, poi le quattro
    passo = 100.0 / n

    for i, nome in enumerate(ORDINE):
        s = i + 1                          # la tappa in cui compare
        # Le previsioni partono dal primo mese previsto, non dall'ultimo
        # osservato: agganciarle li' disegnerebbe un tratto quasi verticale
        # (la media sta molto sotto la punta di dicembre) e quel tratto si
        # leggerebbe come un crollo previsto, che nessuno dei quattro predice.
        punti = [(N_OSS + h, prev[nome][h - 1]) for h in range(1, H + 1)]
        anim.append(keyframes(f"ln{i}", compare(s, n, passo)))
        corpo.append(f'<polyline class="prev" stroke="{COLORI[nome]}" '
                     f'points="{spezzata(punti)}" '
                     f'style="animation:ln{i} var(--d) infinite;opacity:1"/>')

        if nome == "stagionale":
            anim.append(keyframes("arco", solo_in(s, n, passo)))
            corpo.append(f'<path class="arc" d="{arco}" '
                         f'style="animation:arco var(--d) infinite"/>')
            for x, yy, testo, anc in (
                    (x_s, y_s, f"il {mese_di(sorgente)} osservato", "middle"),
                    (x_a, y_a, f"il {mese_di(N_OSS + H)} previsto", "end")):
                corpo.append(f'<circle class="pun" cx="{x:.1f}" cy="{yy:.1f}" '
                             f'r="4" style="animation:arco var(--d) infinite"/>')
                corpo.append(f'<text class="arl" x="{x:.1f}" '
                             f'y="{yy - 66:.1f}" text-anchor="{anc}" '
                             f'style="animation:arco var(--d) infinite">'
                             f'{testo}</text>')

    # ---- la legenda: ogni voce si accende con la sua linea e resta --------
    for i, nome in enumerate(ORDINE):
        s = i + 1
        yl = Y_LEG + (i // 2) * 24
        xl = X0 + (i % 2) * 300
        anim.append(keyframes(f"lg{i}", compare(s, n, passo)))
        m = f' style="animation:lg{i} var(--d) infinite;opacity:1"'
        corpo.append(f'<line class="lgl" x1="{xl}" y1="{yl - 4}" '
                     f'x2="{xl + 26}" y2="{yl - 4}" stroke="{COLORI[nome]}"'
                     f'{m}/>')
        corpo.append(f'<text class="lgt" x="{xl + 34}" y="{yl}"{m}>'
                     f'{ETICHETTE[nome]}</text>')

    # ---- il racconto ------------------------------------------------------
    racconto = [
        ("la serie, e il punto da cui si guarda avanti",
         f"tre anni di mesi, l'ultimo osservato è un {mese_di(N_OSS)}"),
    ] + [
        (f"la {i + 1}ª linea di base: {ETICHETTE[nome]}",
         riga_di(nome, y, prev, sorgente))
        for i, nome in enumerate(ORDINE)
    ]
    racconto[-1] = ("le quattro linee di base, tutte insieme",
                    "un modello che non le batte non serve: quello che sa "
                    "fare lo fa già una riga di codice")

    for s, (titolo, riga) in enumerate(racconto):
        anim.append(keyframes(f"tx{s}", solo_in(s, n, passo)))
        fermo = ";opacity:1" if s == n - 1 else ""
        m = f' style="animation:tx{s} var(--d) infinite{fermo}"'
        corpo.append(f'<text class="tit" x="{X0}" y="{Y_TIT}"{m}>{titolo}</text>')
        corpo.append(f'<text class="rig" x="{X0}" y="{Y_RIG}"{m}>{riga}</text>')

    return Figura(
        larghezza=LARG, altezza=ALT,
        alt=(f"Un grafico a linee. A sinistra la serie osservata, {N_OSS} mesi "
             f"che salgono lentamente con un'onda annuale che ha la punta a "
             f"dicembre; una riga verticale segna l'ultimo mese osservato, un "
             f"{mese_di(N_OSS)}. A destra della riga, {H} mesi di previsione, "
             f"con le quattro linee di base sovrapposte alla stessa scala: "
             f"grigia e piatta a mezz'altezza la media di tutta la serie, ocra "
             f"e piatta all'altezza dell'ultimo valore il naive, teal e "
             f"ondulata il naive stagionale, che ricopia in avanti i mesi "
             f"dell'ultimo anno osservato, e terracotta in salita il drift, "
             f"che prolunga la retta fra il primo e l'ultimo punto. Un arco "
             f"collega il {mese_di(sorgente)} osservato, nove mesi prima "
             f"dell'ultimo, al {mese_di(N_OSS + H)} previsto quindici mesi "
             f"dopo: è il valore che il naive stagionale copia, e la ragione "
             f"per cui oltre un ciclo intero si ricicla sempre l'ultimo anno "
             f"osservato invece di un anno che non è ancora accaduto."),
        corpo="".join(corpo),
        stile=f"""    .oss {{ fill:none; stroke:{INK}; stroke-width:2;
            stroke-linejoin:round; }}
    .prev {{ fill:none; stroke-width:2.5; stroke-linejoin:round;
            stroke-linecap:round; opacity:0; }}
    .org {{ stroke:{BORDER_STRONG}; stroke-width:1.5; stroke-dasharray:4 4; }}
    .arc {{ fill:none; stroke:{TEAL}; stroke-width:1.5; stroke-dasharray:5 4;
            opacity:0; }}
    .pun {{ fill:{TEAL}; opacity:0; }}
    .arl {{ font-family:{SANS}; font-size:12px; fill:{TEAL}; opacity:0; }}
    .lgl {{ stroke-width:3; stroke-linecap:round; opacity:0; }}
    .lgt {{ font-family:{SANS}; font-size:13px; fill:{FG_MUTED}; opacity:0; }}
    .tit {{ font-family:{SANS}; font-size:16px; font-weight:700;
            fill:{TERRACOTTA}; opacity:0; }}
    .rig {{ font-family:{SANS}; font-size:13.5px; fill:{FG_MUTED};
            opacity:0; }}""",
        animazioni=anim,
        durata=n * 2.2,
        fermi=".prev, .arc, .pun, .arl, .lgl, .lgt, .tit, .rig",
    )


def riga_di(nome, y, prev, sorgente) -> str:
    """La riga che spiega, calcolata dalla scena."""
    if nome == "media":
        return "non guarda né la salita né la stagione: sta ferma a mezz'altezza"
    if nome == "naive":
        return ("durissima da battere dove ogni scossa sposta il livello per "
                "sempre")
    if nome == "stagionale":
        return (f"il {mese_di(N_OSS + H)} fra {H} mesi lo prevede con il "
                f"{mese_di(sorgente)} di nove mesi fa, l'ultimo davvero visto")
    return "la sola delle quattro che tiene conto della salita"


# --------------------------------------------------------------------------
# Timeline
# --------------------------------------------------------------------------
def compare(s: int, n: int, passo: float):
    """Compare alla tappa s e resta acceso fino alla fine, riposo compreso."""
    t0 = s * passo
    return [(0.0, "opacity:0"), (max(t0 - passo * 0.28, 0.01), "opacity:0"),
            (t0, "opacity:1"), (100.0, "opacity:1")]


def solo_in(s: int, n: int, passo: float):
    """Acceso nella sola tappa s; l'ultima resta accesa anche a riposo."""
    t0 = s * passo
    if s == n - 1:
        return [(0.0, "opacity:0"), (max(t0 - passo * 0.09, 0.01), "opacity:0"),
                (t0, "opacity:1"), (100.0, "opacity:1")]
    if s == 0:
        return [(0.0, "opacity:1"), (passo * 0.92, "opacity:1"),
                (passo, "opacity:0"), (100.0, "opacity:0")]
    return [(0.0, "opacity:0"), (t0 - passo * 0.09, "opacity:0"),
            (t0, "opacity:1"), (t0 + passo * 0.92, "opacity:1"),
            (t0 + passo, "opacity:0"), (100.0, "opacity:0")]
