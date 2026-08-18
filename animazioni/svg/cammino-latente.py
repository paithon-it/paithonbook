"""Lo stesso cammino in due archivi: senza alone e con alone.

La sezione «Il salto probabilistico» misura, sulle cifre scritte a mano, che un
codice sorteggiato cade a 2,2 spaziature dai codici veri quando l’archivista
scrive un punto, e a 1,0 quando scrive un punto piu' un alone attorno. Questa figura
mostra la stessa cosa da dentro: un segnalino cammina in linea retta da un
codice all’altro, e sotto si legge quanto il punto in cui si trova sia terra
già battuta dal decoder.

I due pannelli differiscono per quello che differisce davvero, e sono due cose
non una: a destra i codici stanno più vicini fra loro (il costo di descrizione
li tira verso il centro) e ciascuno occupa un alone invece che un punto. Il
profilo sotto i pannelli è calcolato, non disegnato a mano: è la stessa
funzione valutata lungo il cammino, e a sinistra crolla a zero.

Lo stato di riposo è l’arrivo, con il cammino percorso e i due profili interi:
chi non anima vede la storia completa, che è il confronto fra le due curve.
"""

import math

from paithon_svg import *

NOME = "cammino-latente"
TITOLO = "Camminare nel latente, con e senza l'alone"

# --------------------------------------------------------------------------
# I dati: gli stessi otto codici per pannello, in due disposizioni
# --------------------------------------------------------------------------
SCARTI = [(0.00, 0.00), (0.72, 0.34), (-0.54, 0.62), (0.30, -0.74),
          (-0.80, -0.28), (0.86, -0.22), (-0.18, 0.86), (0.16, 0.18)]

# (centro del primo gruppo, centro del secondo, sparpagliamento, raggio dell’alone)
SENZA = ((-1.42, -0.34), (1.42, 0.34), 0.42, 0.0)
CON = ((-0.60, -0.15), (0.60, 0.15), 0.42, 0.22)

PORTATA = 0.55          # oltre questa distanza il decoder non c’è mai stato
TAPPE = 44              # in quanti punti si valuta il cammino


def codici(disposizione):
    (ax, ay), (bx, by), s, _ = disposizione
    return ([(ax + s * dx, ay + s * dy) for dx, dy in SCARTI]
            + [(bx + s * dx, by + s * dy) for dx, dy in SCARTI])


def cammino(disposizione):
    """Dal primo codice del gruppo di sinistra al primo di quello di destra."""
    punti = codici(disposizione)
    return punti[0], punti[len(SCARTI)]


def familiarita(punto, disposizione) -> float:
    """1 se il decoder è già stato lì, 0 se non ci ha mai messo piede."""
    raggio = disposizione[3]
    d = min(math.dist(punto, p) for p in codici(disposizione))
    return max(0.0, min(1.0, 1.0 - max(0.0, d - raggio) / PORTATA))


def profilo(disposizione):
    (px, py), (qx, qy) = cammino(disposizione)
    return [familiarita((px + t * (qx - px), py + t * (qy - py)), disposizione)
            for t in (i / (TAPPE - 1) for i in range(TAPPE))]


def verifica(sinistra, destra) -> None:
    """Il confronto che la figura promette c’è davvero?"""
    assert min(sinistra) < 0.05, \
        f"senza alone il cammino non passa da nessun buco: minimo {min(sinistra):.2f}"
    assert 0.80 < min(destra) < 0.98, \
        (f"con l'alone il profilo deve scendere un poco senza toccare il fondo: "
         f"minimo {min(destra):.2f}. A 1,00 l'alone e' troppo largo e la figura "
         f"promette una copertura piena, che la sezione non misura")
    assert sinistra[0] > 0.95 and sinistra[-1] > 0.95, \
        "gli estremi del cammino devono essere codici veri in tutti e due i casi"


# --------------------------------------------------------------------------
# La figura
# --------------------------------------------------------------------------
def pannello(r, disposizione, colore, nome) -> tuple[list, list]:
    """Il piano dei codici: aloni, punti, retta del cammino, segnalino."""
    corpo, anim = [r.cornice()], []
    punti = codici(disposizione)
    raggio = disposizione[3]

    if raggio > 0:
        for x, y in punti:
            corpo.append(f'<circle class="alo" cx="{r.sx(x):.1f}" cy="{r.sy(y):.1f}" '
                         f'r="{raggio * r.scala_x:.1f}" fill="{colore}"/>')

    (px, py), (qx, qy) = cammino(disposizione)
    corpo.append(f'<line class="via" x1="{r.sx(px):.1f}" y1="{r.sy(py):.1f}" '
                 f'x2="{r.sx(qx):.1f}" y2="{r.sy(qy):.1f}"/>')

    for x, y in punti:
        corpo.append(f'<circle class="cod" cx="{r.sx(x):.1f}" cy="{r.sy(y):.1f}" '
                     f'r="5" fill="{colore}"/>')

    # il segnalino: disegnato all’arrivo, l’animazione parte dalla partenza
    dx, dy = r.sx(px) - r.sx(qx), r.sy(py) - r.sy(qy)
    anim.append(keyframes(f"corsa{nome}", [
        (0.0, f"transform:translate({dx:.1f}px,{dy:.1f}px)"),
        (8.0, f"transform:translate({dx:.1f}px,{dy:.1f}px)"),
        (88.0, "transform:translate(0px,0px)"),
        (100.0, "transform:translate(0px,0px)")]))
    corpo.append(
        f'<g style="animation:corsa{nome} var(--d) linear infinite">'
        f'<circle class="seg" cx="{r.sx(qx):.1f}" cy="{r.sy(qy):.1f}" r="9"/></g>')
    return corpo, anim


def curva(x, y, larg, alt, valori, colore, nome) -> tuple[list, list]:
    """Il profilo della familiarità lungo il cammino, più il cursore."""
    passi = [(x + i / (len(valori) - 1) * larg, y + alt - v * alt)
             for i, v in enumerate(valori)]
    poli = " ".join(f"{a:.1f},{b:.1f}" for a, b in passi)
    corpo = [
        f'<line class="base" x1="{x}" y1="{y + alt}" x2="{x + larg}" y2="{y + alt}"/>',
        f'<polyline class="pro" points="{poli}" stroke="{colore}"/>',
    ]
    # il cursore corre sul profilo: una tappa per campione, così segue la curva
    tappe = [(0.0, f"transform:translate({passi[0][0] - passi[-1][0]:.1f}px,"
                   f"{passi[0][1] - passi[-1][1]:.1f}px)"),
             (8.0, f"transform:translate({passi[0][0] - passi[-1][0]:.1f}px,"
                   f"{passi[0][1] - passi[-1][1]:.1f}px)")]
    for i, (a, b) in enumerate(passi):
        t = 8.0 + 80.0 * i / (len(passi) - 1)
        tappe.append((t, f"transform:translate({a - passi[-1][0]:.1f}px,"
                         f"{b - passi[-1][1]:.1f}px)"))
    tappe.append((100.0, "transform:translate(0px,0px)"))
    anim = [keyframes(f"cur{nome}", tappe)]
    corpo.append(f'<g style="animation:cur{nome} var(--d) linear infinite">'
                 f'<circle class="cur" cx="{passi[-1][0]:.1f}" cy="{passi[-1][1]:.1f}" '
                 f'r="5.5" fill="{colore}"/></g>')
    return corpo, anim


def costruisci() -> Figura:
    sinistra, destra = profilo(SENZA), profilo(CON)
    verifica(sinistra, destra)

    rs = Riquadro(x=54, y=58, larg=300, alt=240, xmin=-2.3, xmax=2.3, ymin=-1.84, ymax=1.84)
    rd = Riquadro(x=414, y=58, larg=300, alt=240, xmin=-2.3, xmax=2.3, ymin=-1.84, ymax=1.84)

    corpo, anim = [], []
    for r, disp, colore, nome, titolo, sotto in (
            (rs, SENZA, TERRACOTTA, "s", "senza l’alone",
             "una scheda è un punto"),
            (rd, CON, TEAL, "d", "con l’alone",
             "una scheda è un punto e un alone")):
        c, a = pannello(r, disp, colore, nome)
        corpo += c
        anim += a
        corpo += [
            f'<text class="ttl" x="{r.x}" y="{r.y - 24}">{titolo}</text>',
            f'<text class="lbs" x="{r.x}" y="{r.y - 6}">{sotto}</text>']

    for r, valori, colore, nome in ((rs, sinistra, TERRACOTTA, "s"),
                                    (rd, destra, TEAL, "d")):
        c, a = curva(r.x, 352, r.larg, 62, valori, colore, nome)
        corpo += c
        anim += a
        corpo.append(f'<text class="lbs" x="{r.x}" y="{352 - 10}">'
                     f'quanto quel punto è terra già battuta</text>')
        corpo += [f'<text class="tic" x="{r.x - 8}" y="356" text-anchor="end">1</text>',
                  f'<text class="tic" x="{r.x - 8}" y="418" text-anchor="end">0</text>']

    # la virgola decimale, come in tutte le altre figure del libro
    def dec(v: float) -> str:
        return f"{v:.2f}".replace(".", ",")

    corpo.append(f'<text class="lbs" x="{rs.x}" y="446">il minimo lungo il cammino: '
                 f'{dec(min(sinistra))} a sinistra, {dec(min(destra))} a destra</text>')

    return Figura(
        larghezza=760, altezza=470,
        alt="Due riquadri affiancati, ciascuno un piano dei codici con sedici "
            "codici disposti in due gruppi. A sinistra, «senza l’alone», i "
            "codici sono punti isolati e i due gruppi sono lontani, con un largo "
            "vuoto in mezzo; un segnalino cammina in linea retta da un codice "
            "del gruppo di sinistra a uno del gruppo di destra e attraversa quel "
            "vuoto. A destra, «con l’alone», i gruppi sono più vicini e ogni "
            "codice occupa un alone che si sovrappone a quelli dei vicini, "
            "cosicché il segnalino percorre un cammino altrettanto dritto "
            "senza mai uscire davvero allo scoperto. Sotto ogni riquadro un profilo "
            "misura quanto il punto in cui si trova il segnalino sia terra già "
            "battuta: a sinistra la curva crolla a zero a metà strada, a destra "
            "scende appena e non tocca mai il fondo.",
        corpo="".join(corpo),
        stile=f"""    .cod  {{ stroke:{CREAM}; stroke-width:1.2; }}
    .alo  {{ opacity:0.20; }}
    .via  {{ stroke:{FG_MUTED}; stroke-width:1.6; stroke-dasharray:6 5; }}
    .seg  {{ fill:{OCRA}; stroke:{INK}; stroke-width:2; }}
    .base {{ stroke:{BORDER_STRONG}; stroke-width:1.5; }}
    .pro  {{ fill:none; stroke-width:2.5; }}
    .cur  {{ stroke:{CREAM}; stroke-width:1.5; }}
    .tic  {{ font-family:{SANS}; font-size:12px; fill:{FG_MUTED}; }}""",
        animazioni=anim,
        durata=9.0,
        fermi=".seg, .cur, g",
    )
