"""La stessa nuvola di schede, letta con due coppie di assi.

La sezione «Il latente che si usa» dimostra che separare i fattori senza
supervisione è impossibile, e lo fa con un argomento di una riga: prendi le due
righe della scheda e falle ruotare insieme, e non cambia niente di quello che
abbiamo chiesto alla macchina. Letto, quell'argomento inciampa due volte. La
prima: «far ruotare due righe» non si vede, perché le righe sono numeri e i
numeri non ruotano. La seconda: «girarlo lo lascia identico a prima» suona come
una contraddizione, finché non si vede che a girare sono gli assi e non la
nuvola.

La figura fa vedere esattamente quelle due cose. A sinistra le schede lette con
gli assi che avremmo voluto, l'inclinazione e la luce; a destra le stesse
schede, nello stesso posto, lette con una coppia di assi girata di trenta
gradi. Una scheda è marcata in tutti e due i riquadri: è lo stesso punto, alla
stessa distanza dal centro, e cambiano soltanto i due numeri con cui la si
scrive.

Il calcolo è tutto qui e non c'è niente di scritto a mano: la nuvola esce da un
seme dichiarato, le coordinate girate sono la rotazione applicata davvero, e le
asserzioni pretendono quello che la didascalia promette, cioè che il punto non
si sposti (raggio invariato) e che tutti e due i numeri cambino in modo
visibile.

La rotazione è l'unica cosa che si muove: gli assi di destra partono
sovrapposti a quelli di sinistra e finiscono a trenta gradi, mentre la nuvola
resta ferma. Lo stato di riposo, quello che va in stampa, è l'ultimo: i due
riquadri a confronto, con tutti e quattro i numeri leggibili.
"""

import math
import random

from paithon_svg import *

NOME = "assi-girati"
TITOLO = "la stessa nuvola di schede, letta con due coppie di assi"

SEME = 11
N_PUNTI = 130
SIGMA = 0.82
RAGGIO_MAX = 2.05          # oltre, il punto uscirebbe dal riquadro

ANGOLO = 30.0              # di quanto si girano gli assi
SCHEDA = (0.50, 0.90)      # la scheda marcata: prima riga, seconda riga


def nuvola() -> list[tuple[float, float]]:
    """Le schede: un campione isotropo, cioè senza una direzione preferita."""
    rng = random.Random(SEME)
    punti = []
    while len(punti) < N_PUNTI:
        x, y = rng.gauss(0, SIGMA), rng.gauss(0, SIGMA)
        if math.hypot(x, y) <= RAGGIO_MAX:
            punti.append((x, y))
    return punti


def leggi_girato(p: tuple[float, float], gradi: float) -> tuple[float, float]:
    """Le coordinate di `p` nel sistema di assi ruotato di `gradi`."""
    a = math.radians(gradi)
    return (p[0] * math.cos(a) + p[1] * math.sin(a),
            -p[0] * math.sin(a) + p[1] * math.cos(a))


def anisotropia(punti) -> float:
    """Rapporto fra i due autovalori della covarianza campionaria.

    Vale 1 per una nuvola perfettamente tonda. Serve a difendere l'unica
    affermazione della scena che non si legge da un numero stampato: che la
    nuvola non abbia un verso suo, cioè che girarla la lasci uguale.
    """
    n = len(punti)
    mx = sum(x for x, _ in punti) / n
    my = sum(y for _, y in punti) / n
    sxx = sum((x - mx) ** 2 for x, _ in punti) / n
    syy = sum((y - my) ** 2 for _, y in punti) / n
    sxy = sum((x - mx) * (y - my) for x, y in punti) / n
    tr, det = sxx + syy, sxx * syy - sxy * sxy
    rad = math.sqrt(max(tr * tr / 4 - det, 0.0))
    return (tr / 2 + rad) / (tr / 2 - rad)


def verifica(punti, girata) -> None:
    """Quello che la didascalia promette, trasformato in assert."""
    r0 = math.hypot(*SCHEDA)
    r1 = math.hypot(*girata)
    assert abs(r0 - r1) < 1e-12, f"la scheda si è spostata: {r0} -> {r1}"
    for prima, dopo in zip(SCHEDA, girata):
        assert abs(prima - dopo) >= 0.3, (
            f"il numero cambia troppo poco per vedersi: {prima} -> {dopo}")
    a = anisotropia(punti)
    assert a <= 1.30, f"la nuvola ha un verso suo: rapporto {a:.3f}"
    assert len(punti) == N_PUNTI
    assert all(math.hypot(x, y) <= RAGGIO_MAX for x, y in punti)


def _assi(r: Riquadro, gradi: float, cls: str, anim: str = "") -> list[str]:
    """Le due rette del sistema di assi, in coordinate vere."""
    cx, cy = r.sx(0), r.sy(0)
    lung = r.larg / 2 - 8
    fuori = []
    for k in (0, 90):
        # in SVG la y cresce in giù, quindi l'angolo cambia segno
        a = math.radians(-(gradi + k))
        dx, dy = math.cos(a) * lung, math.sin(a) * lung
        fuori.append(f'<line class="{cls}" x1="{cx - dx:.1f}" y1="{cy - dy:.1f}" '
                     f'x2="{cx + dx:.1f}" y2="{cy + dy:.1f}"{anim}/>')
    return fuori


def _guide(r: Riquadro, p: tuple[float, float], gradi: float,
           cls: str, anim: str = "") -> list[str]:
    """Le due proiezioni della scheda sugli assi: è la lettura, disegnata."""
    cx, cy = r.sx(0), r.sy(0)
    u = r.scala_x
    px, py = r.sx(p[0]), r.sy(p[1])
    letto = leggi_girato(p, gradi)
    fuori = []
    for k, valore in enumerate(letto):
        a = math.radians(-(gradi + 90 * k))
        qx, qy = cx + valore * u * math.cos(a), cy + valore * u * math.sin(a)
        fuori.append(f'<line class="{cls}" x1="{px:.1f}" y1="{py:.1f}" '
                     f'x2="{qx:.1f}" y2="{qy:.1f}"{anim}/>')
        fuori.append(f'<circle class="tick" cx="{qx:.1f}" cy="{qy:.1f}" r="4"'
                     f'{anim}/>')
    return fuori


def _num(v: float) -> str:
    return f"{v:.2f}".replace(".", ",").replace("-", "−")


def costruisci() -> Figura:
    punti = nuvola()
    girata = leggi_girato(SCHEDA, ANGOLO)
    verifica(punti, girata)

    a = Riquadro(x=52, y=68, larg=300, alt=300, xmin=-2.3, xmax=2.3,
                 ymin=-2.3, ymax=2.3)
    b = Riquadro(x=408, y=68, larg=300, alt=300, xmin=-2.3, xmax=2.3,
                 ymin=-2.3, ymax=2.3)

    corpo = [a.cornice(), b.cornice()]

    # le stesse schede, negli stessi posti, in tutti e due i riquadri
    for r in (a, b):
        for x, y in punti:
            corpo.append(f'<circle class="pt" cx="{r.sx(x):.1f}" '
                         f'cy="{r.sy(y):.1f}" r="3.2"/>')

    corpo += _assi(a, 0.0, "asse")
    corpo += _guide(a, SCHEDA, 0.0, "guida")

    # Gli assi di destra girano; la lettura appare solo quando sono arrivati.
    # Se girasse anche lei, a meta' corsa il disegno mostrerebbe le due
    # proiezioni girate misurate su assi che non sono ancora quelli: un fermo
    # immagine che dice il falso, e i fermi immagine vanno in stampa.
    gir = ' style="animation:gira var(--d) infinite"'
    tardi = ' style="animation:appare var(--d) infinite"'
    corpo += _assi(b, ANGOLO, "asse", gir)
    corpo += _guide(b, SCHEDA, ANGOLO, "guida", tardi)

    # la scheda marcata, sopra a tutto
    for r in (a, b):
        corpo.append(f'<circle class="scheda" cx="{r.sx(SCHEDA[0]):.1f}" '
                     f'cy="{r.sy(SCHEDA[1]):.1f}" r="6.5"/>')

    # titoli e letture
    corpo += [
        f'<text class="ttl" x="{a.x}" y="{a.y - 26}">gli assi che avremmo '
        f'voluto</text>',
        f'<text class="ttl" x="{b.x}" y="{b.y - 26}">gli stessi dati, assi '
        f'girati di {ANGOLO:.0f}°</text>',
        f'<text class="lbs" x="{a.x}" y="{a.y - 6}">prima riga: inclinazione · '
        f'seconda riga: luce</text>',
        f'<text class="lbs" x="{b.x}" y="{b.y - 6}">due righe che ne portano '
        f'un po\' per una</text>',
        f'<text class="lbl" x="{a.x}" y="{a.y + a.alt + 34}">la scheda marcata '
        f'dice {_num(SCHEDA[0])} e {_num(SCHEDA[1])}</text>',
        f'<text class="lbl lett" x="{b.x}" y="{b.y + b.alt + 34}"{tardi}>la '
        f'stessa scheda dice {_num(girata[0])} e {_num(girata[1])}</text>',
        f'<text class="lbs" x="{a.x}" y="{a.y + a.alt + 58}">distanza dal '
        f'centro {_num(math.hypot(*SCHEDA))}</text>',
        f'<text class="lbs lett" x="{b.x}" y="{b.y + b.alt + 58}"{tardi}>'
        f'distanza dal centro {_num(math.hypot(*girata))}</text>',
    ]

    return Figura(
        larghezza=760, altezza=470,
        alt=f"Due riquadri affiancati con dentro la stessa nuvola di punti, "
            f"negli stessi posti. A sinistra la nuvola è letta con una coppia "
            f"di assi orizzontale e verticale, intestati «prima riga: "
            f"inclinazione» e «seconda riga: luce»; a destra con una coppia di "
            f"assi girata di {ANGOLO:.0f} gradi. In tutti e due i riquadri è "
            f"marcato lo stesso punto, con le linee tratteggiate che lo "
            f"proiettano sui due assi: a sinistra si legge "
            f"{_num(SCHEDA[0])} e {_num(SCHEDA[1])}, a destra "
            f"{_num(girata[0])} e {_num(girata[1])}, e la sua distanza dal "
            f"centro resta {_num(math.hypot(*SCHEDA))} in tutti e due.",
        corpo="".join(corpo),
        stile=f"""    .pt {{ fill:{TEAL}; fill-opacity:0.38; }}
    .asse {{ stroke:{TERRACOTTA}; stroke-width:2.5; }}
    .guida {{ stroke:{OCRA}; stroke-width:2; stroke-dasharray:5 5; }}
    .tick {{ fill:{OCRA}; }}
    .scheda {{ fill:{OCRA}; stroke:{INK}; stroke-width:2; }}
    .lett {{ fill:{TERRACOTTA}; }}
    .asse {{ transform-box:view-box;
        transform-origin:{b.sx(0):.1f}px {b.sy(0):.1f}px; }}""",
        animazioni=[
            keyframes("gira", [
                (0.0, f"transform:rotate({ANGOLO:.0f}deg)"),
                (16.0, f"transform:rotate({ANGOLO:.0f}deg)"),
                (62.0, "transform:rotate(0deg)"),
                (100.0, "transform:rotate(0deg)")]),
            keyframes("appare", [
                (0.0, "opacity:0"),
                (56.0, "opacity:0"),
                (70.0, "opacity:1"),
                (100.0, "opacity:1")]),
        ],
        durata=7.0,
        fermi=".asse, .guida, .tick, .lett",
    )
