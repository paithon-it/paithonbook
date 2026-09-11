"""Il peso del trapezio: dove sta il minimo, e quanto costa allontanarsene.

`StateSpaceModel/dualita-e-mamba-2-3.md` racconta la discretizzazione di
Mamba-3 come un conto a trapezi al posto di un conto a rettangoli, e poi
aggiunge la cosa che conta: i due estremi non pesano metà e metà, il peso
$\\lambda_t$ lo sceglie il modello a ogni passo, e il trapezio della geometria è
solo il caso in cui i due pesi sono uguali.

Una figura che disegnasse il solo trapezio simmetrico direbbe che Mamba-3
integra meglio, ed è proprio la convinzione che la pagina toglie dodici righe
più in là: il paper riporta che obbligare il peso a stare vicino a metà
peggiora i risultati. Quindi qui non si disegnano due metodi, si disegna la
**famiglia**: la stessa quantità stimata con tre pesi diversi sullo stesso
passo, e poi lo scarto per ogni peso fra zero e uno.

Il fatto che si vede, e che non è ovvio: la curva dello scarto è una conca con
il minimo quasi a metà, e **i due estremi sbagliano quasi uguale**. Il peso
zero, cioè «guarda solo il campione di prima», non è meglio del peso uno di
Mamba-2: è la stessa cosa dall'altra parte. Libertà e precisione tirano da
parti opposte, che è la frase della pagina, e qui si misura di quanto.

Che cosa si approssima. La transizione resta esatta, $e^{\\Delta A}$; a essere
stimato è il solo contributo dell'ingresso, cioè l'integrale su un passo di
$g(\\tau) = e^{(t_k - \\tau)A} B(\\tau) x(\\tau)$. E la regola di Mamba-3 è una
combinazione convessa dei due valori di $g$ agli estremi: per ogni $\\lambda$ è
il rettangolo di altezza $(1-\\lambda)g_0 + \\lambda g_1$, e per
$\\lambda = 1/2$ quel rettangolo ha la stessa area del trapezio classico.

Il banco di prova è un sistema scalare tempo-variante, dichiarato nella
didascalia: non è Mamba-3, è la quadratura che Mamba-3 usa. Verifica la
Proposizione 1 del paper, non l'implementazione.

Una trappola, che è costata un falso verde a chi ha misurato prima di questo
disegno: su un solo punto di partenza l'ordine si misura sbagliato, perché
capita che la derivata seconda dell'integranda si annulli lì per caso. Lo
scarto del pannello di destra è quindi una media su un giro intero di passi, e
`verifica()` controlla anche che sul passo disegnato le due derivate non siano
degeneri.

Ferma: qui non scorre niente, sono due grafici.
"""

import math
import re

from paithon_svg import *

NOME = "trapezio-e-il-peso"
TITOLO = "il peso del trapezio, e la conca dello scarto"

# --------------------------------------------------------------------------
# Il banco di prova: h' = a h + u(t), scalare, tempo-variante.
# --------------------------------------------------------------------------
A = -0.5
PASSO = 0.4                      # il passo, grande apposta: si deve vedere
T0 = 0.0                         # il passo disegnato a sinistra
PERIODO = 2 * math.pi / 3        # un giro intero di u
QUANTI = 120                     # i punti di partenza su cui si media
PESI = (0.0, 0.5, 1.0)           # i tre pesi disegnati sul passo


def ingresso(t: float) -> float:
    return math.sin(3 * t) + 0.5


def integranda(t0: float, s: float) -> float:
    """g(s) = e^{(t0+PASSO-s)A} u(s): l'ingresso, già pesato da quanto defluisce."""
    return math.exp((t0 + PASSO - s) * A) * ingresso(s)


def vera(t0: float, nodi: int = 4000) -> float:
    """L'integrale su un passo, per quadratura fitta: il termine di riferimento."""
    h = PASSO / nodi
    tot = 0.5 * (integranda(t0, t0) + integranda(t0, t0 + PASSO))
    tot += sum(integranda(t0, t0 + i * h) for i in range(1, nodi))
    return tot * h


def stima(t0: float, peso: float) -> float:
    """La regola di Mamba-3: i due valori agli estremi, pesati da lambda."""
    g0 = math.exp(PASSO * A) * ingresso(t0)      # con lo sconto del deflusso
    g1 = ingresso(t0 + PASSO)
    return PASSO * ((1 - peso) * g0 + peso * g1)


def scarto_medio(peso: float) -> float:
    """Lo scarto quadratico medio su un giro intero di punti di partenza."""
    passi = [PERIODO * i / QUANTI for i in range(QUANTI)]
    return math.sqrt(sum((stima(t, peso) - vera(t)) ** 2 for t in passi) / QUANTI)


CONCA = [(i / 100, scarto_medio(i / 100)) for i in range(101)]


def num(v: float, cifre: int = 3) -> str:
    return f"{v:.{cifre}f}".replace(".", ",")


# --------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------
LARG, ALT = 900, 442
MARGINE = 16.0                   # quanto deve restare libero a destra
SIN = Riquadro(x=66, y=76, larg=318, alt=212,
               xmin=T0, xmax=T0 + PASSO, ymin=0.0, ymax=1.62)
DES = Riquadro(x=520, y=76, larg=300, alt=212,
               xmin=0.0, xmax=1.0, ymin=0.0, ymax=0.175)

# Larghezza media di un carattere, in frazioni della dimensione del font. Non
# è una stima a occhio: viene dalle metriche di DejaVu Sans, che è il ripiego
# che vedono sia il rasterizzatore sia il lettore che non ha Inter, e sta
# sopra il caso peggiore misurato (0,52 per il tondo, 0,64 per il grassetto
# delle cifre). Serve a `verifica_testi()`, che senza un numero qui non
# potrebbe dire niente.
LARGO = {"lbs": (13.0, 0.54), "val": (13.0, 0.66), "ttl": (16.0, 0.62)}
TINTA = {0.0: "pB", 0.5: "pC", 1.0: "pA"}      # teal, ocra, terracotta


def verifica() -> None:
    """Difende quello che la didascalia promette."""
    g0 = math.exp(PASSO * A) * ingresso(T0)
    g1 = ingresso(T0 + PASSO)

    # Il passo disegnato non è un punto fortunato: se g' o g'' si annullassero
    # lì, il trapezio sembrerebbe esatto per caso. È l'errore che ha prodotto un
    # falso verde, e da qui in poi non lo può più produrre.
    s, e = T0 + PASSO / 2, 1e-4
    d1 = (integranda(T0, s + e) - integranda(T0, s - e)) / (2 * e)
    d2 = (integranda(T0, s + e) - 2 * integranda(T0, s)
          + integranda(T0, s - e)) / e ** 2
    assert abs(d1) > 0.5 and abs(d2) > 0.5, \
        f"il passo disegnato è degenere: g'={d1:.3f}, g''={d2:.3f}"

    # Sul passo disegnato i due estremi sbagliano in versi opposti, e il peso a
    # metà è quello vicino: è il disegno di sinistra.
    riferimento = vera(T0)
    assert stima(T0, 0.0) < riferimento < stima(T0, 1.0), \
        "i due estremi non racchiudono il valore vero"
    assert abs(stima(T0, 0.5) - riferimento) < abs(stima(T0, 0.0) - riferimento) / 10, \
        "sul passo disegnato il peso a metà non è dieci volte più vicino"
    assert g0 < g1, "le due altezze non sono distinte: le tre righe si sovrappongono"
    # Lo sconto del deflusso, che la legenda promette: l'altezza di sinistra è
    # l'ingresso già sgonfiato, non l'ingresso nudo. Senza questi tre, togliere
    # l'esponenziale da `stima` lasciava tutto verde.
    assert math.isclose(stima(T0, 0.0) / PASSO, integranda(T0, T0)), \
        "l'altezza del peso zero non è il valore di g all'inizio del passo"
    assert math.isclose(stima(T0, 1.0) / PASSO, integranda(T0, T0 + PASSO)), \
        "l'altezza del peso uno non è il valore di g alla fine del passo"
    assert integranda(T0, T0) < 0.9 * ingresso(T0), \
        "lo sconto del deflusso non si vede: la legenda promette più del disegno"
    assert g1 < SIN.ymax, "la riga del peso uno esce dal riquadro"

    # A destra: la conca, il minimo quasi a metà, e i due estremi quasi pari.
    pesi = [p for p, _ in CONCA]
    valori = [v for _, v in CONCA]
    argmin = pesi[valori.index(min(valori))]
    assert 0.40 < argmin < 0.60, f"il minimo non cade quasi a metà: {argmin}"
    agli_estremi = scarto_medio(0.0), scarto_medio(1.0)
    rapporto = min(agli_estremi) / max(agli_estremi)
    assert rapporto > 0.90, \
        f"i due estremi non sbagliano quasi uguale: {agli_estremi}"
    assert min(valori) * 4 < min(agli_estremi), \
        "il minimo non è nettamente sotto gli estremi: la conca non si vede"
    assert scarto_medio(0.5) * 4 < min(agli_estremi), \
        "il peso un mezzo non sbaglia quasi cinque volte meno degli estremi"
    # È una conca sola: si scende fino al minimo e poi si risale, sempre.
    giro = [valori[i + 1] - valori[i] for i in range(len(valori) - 1)]
    cambi = sum(1 for i in range(len(giro) - 1) if giro[i] * giro[i + 1] < 0)
    assert cambi == 1, f"la curva non è una conca sola: {cambi} inversioni"
    assert max(valori) < DES.ymax, "la conca esce dal riquadro"


def verifica_testi(disegno: str) -> None:
    """Nessun testo esce dalla tela, nemmeno con il font di ripiego.

    Il generatore non può misurare il font che vedrà il lettore, quindi stima
    la larghezza dal numero di caratteri con le metriche del ripiego più
    largo. È un limite superiore grossolano e va bene così: quello che deve
    impedire è il caso in cui la stima ci sta e la resa no, e per questo la
    costante sta sopra il peggio misurato. Il difetto che chiude si era visto
    solo aprendo la figura: un sottotitolo tagliato a metà parola.
    """
    righe: dict[float, list] = {}
    for attributi, testo in re.findall(r"<text([^>]*)>([^<]*)</text>", disegno):
        if "transform=" in attributi:          # i nomi degli assi, ruotati
            continue
        classe = re.search(r'class="(\w+)"', attributi).group(1)
        px, quanto = LARGO[classe]
        larghezza = len(testo) * px * quanto
        x = float(re.search(r'\sx="([-\d.]+)"', attributi).group(1))
        y = float(re.search(r'\sy="([-\d.]+)"', attributi).group(1))
        ancora = re.search(r'text-anchor="(\w+)"', attributi)
        ancora = ancora.group(1) if ancora else "start"
        sinistra = {"start": x, "middle": x - larghezza / 2,
                    "end": x - larghezza}[ancora]
        assert sinistra >= MARGINE / 2, \
            f"esce a sinistra ({sinistra:.0f}): {testo[:40]}"
        assert sinistra + larghezza <= LARG - MARGINE, \
            (f"esce a destra ({sinistra + larghezza:.0f} su {LARG - MARGINE:.0f}): "
             f"{testo[:40]}")
        righe.setdefault(y, []).append((sinistra, sinistra + larghezza, testo))

    # E due testi sulla stessa riga non si toccano. È l'altra metà del
    # difetto: «quello che entra davvero» finiva a 252 e «0,380», ancorato a
    # destra, cominciava a 253. Il bordo della tela stava benissimo.
    for y, pezzi in righe.items():
        pezzi.sort()
        for (_, fine, primo), (inizio, _, dopo) in zip(pezzi, pezzi[1:]):
            assert inizio - fine >= 10.0, \
                (f"a y={y:.0f} «{primo[:24]}» e «{dopo[:24]}» distano "
                 f"{inizio - fine:.0f} px")


def curva(riquadro: Riquadro, punti) -> str:
    d = " ".join(f"{'M' if i == 0 else 'L'} {riquadro.sx(x):.1f} "
                 f"{riquadro.sy(y):.1f}" for i, (x, y) in enumerate(punti))
    return d


def costruisci() -> Figura:
    verifica()

    g0 = math.exp(PASSO * A) * ingresso(T0)
    g1 = ingresso(T0 + PASSO)
    riferimento = vera(T0)
    campioni = [(T0 + PASSO * i / 200, integranda(T0, T0 + PASSO * i / 200))
                for i in range(201)]

    corpo = []

    # ---- pannello di sinistra: un passo, e le tre altezze ------------------
    corpo.append(f'<text class="ttl" x="{SIN.x:.1f}" y="{SIN.y - 34:.1f}">'
                 f'un passo solo</text>')
    corpo.append(f'<text class="lbs" x="{SIN.x:.1f}" y="{SIN.y - 16:.1f}">'
                 f'quello che entra, e le tre altezze con cui lo si stima'
                 f'</text>')
    corpo.append(f'<text class="lbs" transform="rotate(-90 {SIN.x - 14:.1f} '
                 f'{SIN.y + SIN.alt / 2:.1f})" x="{SIN.x - 14:.1f}" '
                 f'y="{SIN.y + SIN.alt / 2:.1f}" text-anchor="middle">'
                 f'altezza</text>')
    corpo.append(SIN.cornice())

    area = (f'M {SIN.sx(T0):.1f} {SIN.sy(0):.1f} ' + curva(SIN, campioni)[1:]
            + f' L {SIN.sx(T0 + PASSO):.1f} {SIN.sy(0):.1f} Z')
    corpo.append(f'<path class="area" d="{area}"/>')
    corpo.append(f'<path class="gcur" d="{curva(SIN, campioni)}"/>')

    corpo.append(f'<line class="trap" x1="{SIN.sx(T0):.1f}" '
                 f'y1="{SIN.sy(g0):.1f}" x2="{SIN.sx(T0 + PASSO):.1f}" '
                 f'y2="{SIN.sy(g1):.1f}"/>')
    for peso in PESI:
        h = (1 - peso) * g0 + peso * g1
        y = SIN.sy(h)
        corpo.append(f'<line class="{TINTA[peso]}l" x1="{SIN.sx(T0):.1f}" '
                     f'y1="{y:.1f}" x2="{SIN.sx(T0 + PASSO):.1f}" '
                     f'y2="{y:.1f}"/>')

    corpo.append(f'<text class="lbs" x="{SIN.sx(T0) + 6:.1f}" '
                 f'y="{SIN.sy(0) + 18:.1f}">inizio del passo</text>')
    corpo.append(f'<text class="lbs" x="{SIN.sx(T0 + PASSO):.1f}" '
                 f'y="{SIN.sy(0) + 18:.1f}" text-anchor="end">fine</text>')

    # la legenda: ogni riga porta la sua area, calcolata qui sopra
    righe = [("area", "quello che entra davvero", riferimento, ""),
             ("pB", "peso 0", stima(T0, 0.0),
              "solo il campione di prima, già sgonfiato da quello che è defluito"),
             ("pC", "peso ½", stima(T0, 0.5),
              "il trapezio della geometria, tratteggiato: stessa area"),
             ("pA", "peso 1", stima(T0, 1.0),
              "solo il campione di adesso: il conto di Mamba-2")]
    for i, (classe, nome, valore, nota) in enumerate(righe):
        y = SIN.y + SIN.alt + 34 + i * 19
        corpo.append(f'<rect class="{classe}s" x="{SIN.x:.1f}" y="{y - 10:.1f}" '
                     f'width="15" height="12"/>')
        corpo.append(f'<text class="lbs" x="{SIN.x + 23:.1f}" y="{y:.1f}">'
                     f'{nome}</text>')
        corpo.append(f'<text class="val" x="{SIN.x + 300:.1f}" y="{y:.1f}" '
                     f'text-anchor="end">{num(valore)}</text>')
        if nota:
            corpo.append(f'<text class="lbs" x="{SIN.x + 312:.1f}" '
                         f'y="{y:.1f}">{nota}</text>')

    # ---- pannello di destra: la conca ---------------------------------------
    corpo.append(f'<text class="ttl" x="{DES.x:.1f}" y="{DES.y - 34:.1f}">'
                 f'tutti i pesi</text>')
    corpo.append(f'<text class="lbs" x="{DES.x:.1f}" y="{DES.y - 16:.1f}">'
                 f'scarto quadratico medio su {QUANTI} punti di '
                 f'partenza</text>')
    corpo.append(f'<text class="lbs" transform="rotate(-90 {DES.x - 14:.1f} '
                 f'{DES.y + DES.alt / 2:.1f})" x="{DES.x - 14:.1f}" '
                 f'y="{DES.y + DES.alt / 2:.1f}" text-anchor="middle">'
                 f'quanto si sbaglia</text>')
    corpo.append(DES.cornice())
    corpo.append(f'<path class="gcur" d="{curva(DES, CONCA)}"/>')

    minimo = min(CONCA, key=lambda pv: pv[1])
    corpo.append(f'<line class="tick" x1="{DES.sx(minimo[0]):.1f}" '
                 f'y1="{DES.sy(minimo[1]):.1f}" x2="{DES.sx(minimo[0]):.1f}" '
                 f'y2="{DES.sy(0):.1f}"/>')
    for peso, dove in [(0.0, "start"), (0.5, "middle"), (1.0, "end")]:
        x, y = DES.sx(peso), DES.sy(scarto_medio(peso))
        corpo.append(f'<circle class="{TINTA[peso]}p" cx="{x:.1f}" '
                     f'cy="{y:.1f}" r="4.5"/>')
        dx = {"start": 8, "middle": 0, "end": -8}[dove]
        corpo.append(f'<text class="val" x="{x + dx:.1f}" y="{y - 11:.1f}" '
                     f'text-anchor="{dove}">{num(scarto_medio(peso))}</text>')
    corpo.append(f'<text class="lbs" x="{DES.sx(0):.1f}" '
                 f'y="{DES.sy(0) + 18:.1f}">peso 0</text>')
    corpo.append(f'<text class="lbs" x="{DES.sx(0.5):.1f}" '
                 f'y="{DES.sy(0) + 18:.1f}" text-anchor="middle">'
                 f'½, e il minimo a {num(minimo[0], 2)}</text>')
    corpo.append(f'<text class="lbs" x="{DES.sx(1):.1f}" '
                 f'y="{DES.sy(0) + 18:.1f}" text-anchor="end">1</text>')

    # ---- le due righe che tengono la figura onesta --------------------------
    corpo.append(f'<text class="lbs" x="{SIN.x:.1f}" y="{ALT - 34:.1f}">'
                 f'Si stima solo l’acqua che entra in questo tratto: quella '
                 f'che c’era già nella vasca si porta avanti esatta.</text>')
    corpo.append(f'<text class="lbs" x="{SIN.x:.1f}" y="{ALT - 14:.1f}">'
                 f'Il peso lo sceglie il modello a ogni passo, e obbligarlo a '
                 f'stare a metà, riferisce il paper, peggiora i risultati.'
                 f'</text>')

    disegno = "".join(corpo)
    verifica_testi(disegno)

    return Figura(
        larghezza=LARG, altezza=ALT,
        alt=f"Due grafici affiancati. A sinistra, un passo solo di una "
            f"curva di prova: una curva che sale, e l'area sotto di lei, "
            f"tinta, è quello che entra davvero, {num(riferimento)}. Tre "
            f"righe orizzontali la attraversano, e ciascuna è l'altezza con "
            f"cui un peso stima quella stessa area: la riga bassa, in teal, è "
            f"il peso 0, che guarda solo il campione di prima e dà "
            f"{num(stima(T0, 0.0))}; quella di mezzo, in ocra, è il peso un "
            f"mezzo, il trapezio della geometria, e dà {num(stima(T0, 0.5))}; "
            f"quella alta, in terracotta, è il peso 1, cioè il conto di "
            f"Mamba-2, e dà {num(stima(T0, 1.0))}. Le due righe estreme "
            f"stanno una tutta sotto e una tutta sopra la curva; quella di "
            f"mezzo la taglia, e il pezzo che avanza da una parte compensa "
            f"quello che manca dall'altra. Un segmento tratteggiato in ocra "
            f"unisce i due estremi della curva: è il trapezio della "
            f"geometria, e racchiude la stessa area della riga di mezzo. A "
            f"destra, lo scarto quadratico medio su {QUANTI} punti di "
            f"partenza al variare del peso fra zero e uno: una conca. Vale "
            f"{num(scarto_medio(0.0))} al peso zero, "
            f"{num(scarto_medio(0.5))} al peso un mezzo e "
            f"{num(scarto_medio(1.0))} al peso uno, e un trattino segna il "
            f"minimo vero, che cade a {num(minimo[0], 2)}. I due estremi "
            f"sbagliano quasi uguale, e il fondo della conca sbaglia quasi "
            f"cinque volte meno di tutti e due.",
        corpo=disegno,
        stile=f"""    .area {{ fill:{FG_MUTED}; fill-opacity:0.16; stroke:none; }}
    .areas{{ fill:{FG_MUTED}; fill-opacity:0.16; stroke:{FG_MUTED};
            stroke-width:1.2; }}
    .gcur {{ fill:none; stroke:{INK}; stroke-width:2.2;
            stroke-linejoin:round; }}
    .trap {{ stroke:{OCRA}; stroke-width:1.8; stroke-dasharray:6 4; }}
    .pAl  {{ stroke:{TERRACOTTA}; stroke-width:2.4; }}
    .pBl  {{ stroke:{TEAL}; stroke-width:2.4; }}
    .pCl  {{ stroke:{OCRA}; stroke-width:2.4; }}
    .pAs  {{ fill:{TERRACOTTA}; stroke:none; }}
    .pBs  {{ fill:{TEAL}; stroke:none; }}
    .pCs  {{ fill:{OCRA}; stroke:none; }}
    .pAp  {{ fill:{TERRACOTTA}; stroke:none; }}
    .pBp  {{ fill:{TEAL}; stroke:none; }}
    .pCp  {{ fill:{OCRA}; stroke:none; }}
    .tick {{ stroke:{FG_MUTED}; stroke-width:1.2; stroke-dasharray:3 3; }}
    .val  {{ font-family:{SANS}; font-size:13px; font-weight:700;
            fill:{INK}; }}""",
    )
