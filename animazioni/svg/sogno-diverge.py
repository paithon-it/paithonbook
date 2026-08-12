"""Il sogno che si stacca dalla realta', un passo alla volta.

Due traiettorie dello stesso mondo, dalla stessa identica partenza. Una e' il
mondo vero; l'altra e' il mondo come il modello se lo immagina, con un solo
difetto: sbaglia di **meno del tre per cento** un solo coefficiente, quello che
dice quanta velocita' sopravvive a ogni passo. Per una decina di
passi le due curve sono indistinguibili; poi si separano, e lo scarto peggiore
non torna piu' indietro.

E' la ragione, resa visibile, per cui i sogni utili sono brevi: l'errore del
modello non e' un rumore che si media a zero, e' un errore **sistematico** che
al passo dopo entra come stato iniziale. Il segno di spunta non e' un
fotogramma: e' il momento in cui lo scarto supera la tolleranza dichiarata, e
per saperlo bisogna guardare la scena scorrere.

Tutti i numeri li calcola questa scena, integrando le due dinamiche.
"""

import math

from paithon_svg import *

NOME = "sogno-diverge"
TITOLO = "il sogno che si stacca dalla realtà"

PASSI = 28
# Quanta velocita' sopravvive a un passo. ATTENZIONE a come si racconta: fra i
# due valori c'e' il 2,8% di scarto, ma il *coefficiente di smorzamento* e'
# (1 - a), e li' lo scarto e' del 91%. Il testo deve nominare la grandezza
# giusta, altrimenti dice una cosa falsa con un numero vero.
A_VERO = 0.97
A_SOGNO = 0.9973
TOLLERANZA = 0.25      # quanto scarto si e' disposti a sopportare

RIQ = Riquadro(x=88, y=96, larg=504, alt=248,
               xmin=0, xmax=PASSI - 1, ymin=-7.2, ymax=7.2)


def traiettoria(a):
    """Oscillatore smorzato e forzato, integrato a passi interi."""
    x, v, fuori = 0.0, 0.0, []
    for t in range(PASSI):
        fuori.append(x)
        v += -0.35 * x - (1 - a) * v + 0.30 * math.sin(0.55 * t)
        x += v
    return fuori


def scarti():
    vero, sogno = traiettoria(A_VERO), traiettoria(A_SOGNO)
    peggiore = []
    corrente = 0.0
    for a, b in zip(vero, sogno):
        corrente = max(corrente, abs(a - b))
        peggiore.append(corrente)
    return vero, sogno, peggiore


def polilinea(ys, fino=None):
    n = len(ys) if fino is None else fino + 1
    return " ".join(f"{RIQ.sx(t):.1f},{RIQ.sy(ys[t]):.1f}" for t in range(n))


def costruisci() -> Figura:
    vero, sogno, peggiore = scarti()
    rottura = next(t for t, s in enumerate(peggiore) if s > TOLLERANZA)

    corpo, anim = [], []
    corpo.append(RIQ.cornice(croce=True))

    # --- il momento della rottura: una banda che copre i passi non piu' fidati
    corpo.append(f'<rect class="ombra" x="{RIQ.sx(rottura):.1f}" y="{RIQ.y}" '
                 f'width="{RIQ.x + RIQ.larg - RIQ.sx(rottura):.1f}" '
                 f'height="{RIQ.alt}"/>')
    corpo.append(f'<line class="taglio" x1="{RIQ.sx(rottura):.1f}" y1="{RIQ.y}" '
                 f'x2="{RIQ.sx(rottura):.1f}" y2="{RIQ.y + RIQ.alt}"/>')
    corpo.append(f'<text class="lbs" x="{RIQ.x + RIQ.larg - 26:.0f}" '
                 f'y="{RIQ.y + 22}" text-anchor="end">'
                 f'da qui il sogno non serve più</text>')

    # --- le due traiettorie: lo stato di riposo e' quella completa, e l'animazione
    # scopre i passi uno alla volta con stroke-dashoffset
    anim.append(keyframes("scopri", [
        (0.0, f"transform:translateX({-RIQ.larg - 8:.0f}px)"),
        (4.0, f"transform:translateX({-RIQ.larg - 8:.0f}px)"),
        (88.0, "transform:translateX(0)"), (100.0, "transform:translateX(0)")]))
    corpo.insert(0, f'<defs><clipPath id="scopri"><rect x="{RIQ.x - 4}" '
                    f'y="{RIQ.y - 4}" width="{RIQ.larg + 8}" '
                    f'height="{RIQ.alt + 8}" '
                    f'style="animation:scopri var(--d) infinite"/></clipPath></defs>')
    corpo.append('<g clip-path="url(#scopri)">')
    for ys, classe in ((vero, "vero"), (sogno, "sogno")):
        corpo.append(f'<polyline class="{classe}" points="{polilinea(ys)}"/>')
    corpo.append('</g>')

    # --- scritte
    corpo += [
        f'<text class="lbl" x="{RIQ.x}" y="{RIQ.y - 52}">'
        f'lo stesso mondo, dalla stessa partenza</text>',
        f'<text class="lbs" x="{RIQ.x + RIQ.larg}" y="{RIQ.y - 24}" '
        f'text-anchor="end" style="fill:{TERRACOTTA}">come il modello se lo '
        f'immagina</text>',
        f'<text class="lbs" x="{RIQ.x + RIQ.larg - 232}" y="{RIQ.y - 24}" '
        f'text-anchor="end" style="fill:{TEAL}">com\'è davvero</text>',
        f'<text class="cap" x="{RIQ.x + RIQ.larg / 2:.0f}" '
        f'y="{RIQ.y + RIQ.alt + 46}" text-anchor="middle">'
        f'{rottura} passi di sogno affidabile</text>',
        f'<text class="lbs" x="{RIQ.x + RIQ.larg / 2:.0f}" '
        f'y="{RIQ.y + RIQ.alt + 72}" text-anchor="middle">'
        f'al modello basta sbagliare del {100 * (A_SOGNO - A_VERO) / A_VERO:.1f} '
        f'per cento un solo numero: quanta spinta resta a ogni passo</text>'
        .replace("2.8", "2,8"),
        # gli assi vanno nominati: un lettore che salta le schede non ha modo
        # di sapere che cosa sia la quantita' disegnata
        f'<text class="lbs" x="{RIQ.x + 10}" y="{RIQ.y + 20}">dove si trova</text>',
        f'<text class="lbs" x="{RIQ.x + RIQ.larg}" y="{RIQ.y + RIQ.alt + 22}" '
        f'text-anchor="end">passi</text>',
    ]

    return Figura(
        larghezza=680, altezza=452,
        alt="Due curve oscillanti partono sovrapposte e restano indistinguibili "
            "per una quindicina di passi, poi si separano sempre di più. Una "
            "banda chiara copre la parte finale del grafico, da dove lo scarto "
            "ha superato la tolleranza in poi.",
        corpo="".join(corpo),
        stile=f"""    .vero   {{ fill:none; stroke:{TEAL}; stroke-width:3.5; }}
    .sogno  {{ fill:none; stroke:{TERRACOTTA}; stroke-width:2.6; }}
    .ombra  {{ fill:{BORDER}; opacity:0.45; }}
    .taglio {{ stroke:{OCRA}; stroke-width:2.5; stroke-dasharray:6 5; }}
    .cap    {{ font-family:{SANS}; font-size:17px; font-weight:700;
              fill:{TERRACOTTA}; }}""",
        animazioni=anim,
        durata=7.0,
        fermi="#scopri rect",
    )
