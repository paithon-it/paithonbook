"""Dove sta la taratura decide se la scorciatoia resta libera.

La sezione «Il blocco: scorciatoia e taratura» racconta un palazzo: le stanze
dove si lavora, la scala che le scavalca, e su ogni pianerottolo una bilancia.
Nel montaggio del 2017 la bilancia sta *sulla* scala, quindi chi scende deve
attraversarla a ogni piano e il corrimano non è sgombro fino in fondo; nei
modelli di oggi la bilancia è stata portata all'ingresso delle stanze, e la
scala è tornata libera da cima a fondo. In formule sono
`LayerNorm(x + Sub(x))` e `x + Sub(LayerNorm(x))`, che a leggerle di fila si
somigliano parecchio.

È una differenza di **percorso**, non di ingredienti: gli stessi tre pezzi
montati in due modi, e a cambiare è che cosa incontra il gradiente scendendo.
In formule si somigliano; in un disegno la differenza si vede in un secondo.

Il conto lo fa la scena e non è scritto a mano: le interruzioni del corrimano
si contano sui segmenti disegnati, e il numero per una pila vera si ricava da
`BLOCCHI_PILA`. Le asserzioni pretendono quello che la didascalia promette,
cioè che a sinistra ce ne sia una per sommatoria e a destra nessuna, e che
nessuna scatola si sovrapponga a un'altra, che è il modo in cui questo tipo di
schema si rompe senza dirlo.

Il tempo qui è il contenuto: si guarda qualcuno scendere. Il segnalino parte
dall'uscita, in alto, e arriva all'ingresso; lo stato di riposo, quello che va
in stampa, è l'arrivo.
"""

from paithon_svg import *

NOME = "corrimano-e-taratura"
TITOLO = "post-LN e pre-LN: dove sta la taratura, e che cosa incontra chi scende"

BLOCCHI = 2                # i blocchi disegnati
SOTTOSTRATI = 2            # attenzione e lavoro individuale, per blocco
BLOCCHI_PILA = 96          # una pila vera, quella di GPT-3

STAZIONI = BLOCCHI * SOTTOSTRATI
Y_BASSO, PASSO = 470, 96
SU, GIU = 64, 88           # quanto il corrimano esce sopra e sotto: sotto deve
                           # superare STACCO, o l'ultimo ramo si stacca nel vuoto
RAMO = 120                 # quanto il ramo si scosta dal corrimano
STANZA_L, STANZA_A = 132, 32
TAR_L, TAR_A = 58, 22
STACCO = 70                # sotto la sommatoria, dove il ramo si stacca


def _stazioni() -> list[float]:
    """L'ordinata di ogni sommatoria, dal basso verso l'alto."""
    return [Y_BASSO - k * PASSO for k in range(STAZIONI)]


def _rett(x, y, larg, alt):
    return (x - larg / 2, y - alt / 2, x + larg / 2, y + alt / 2)


def _si_toccano(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _torre(cx: float, sul_corrimano: bool) -> tuple[list[str], list, list]:
    """Una torre. Torna il disegno, le interruzioni e le scatole piazzate.

    Il disegno esce in due strati, e l'ordine non e' un dettaglio di gusto: le
    righe prima, le scatole e le scritte dopo. Il ramo di un sotto-strato si
    stacca dal corrimano all'altezza della taratura di quello sotto (in post-LN
    e' esatto: quel ramo prende proprio l'uscita della taratura), e il
    corrimano passa dentro la sommatoria. Disegnati nell'ordine in cui si
    calcolano, la riga cancellava la parola «taratura» e il tratto da cinque
    punti del corrimano copriva il piu' dentro il cerchio, che finiva per
    leggersi come dei puntini.
    """
    linee, sopra, buchi, scatole = [], [], [], []
    bx = cx + RAMO
    for k, y in enumerate(_stazioni()):
        y_dip = y + STACCO
        y_mez = (y + y_dip) / 2
        # il ramo: si stacca, sale attraverso la stanza, rientra nella somma
        linee.append(
            f'<path class="ramo" d="M {cx} {y_dip:.0f} H {bx} '
            f'V {y_mez + STANZA_A / 2:.0f} M {bx} {y_mez - STANZA_A / 2:.0f} '
            f'V {y:.0f} H {cx}"/>')
        sopra.append(f'<rect class="stanza" x="{bx - STANZA_L / 2:.0f}" '
                     f'y="{y_mez - STANZA_A / 2:.0f}" width="{STANZA_L}" '
                     f'height="{STANZA_A}" rx="5"/>')
        sopra.append(f'<text class="lbs stz" x="{bx}" y="{y_mez + 5:.0f}">'
                     f'{"attenzione" if k % 2 == 0 else "lavoro a parte"}</text>')
        scatole.append(_rett(bx, y_mez, STANZA_L, STANZA_A))

        if sul_corrimano:
            yt = y - 26
            buchi.append((yt - TAR_A / 2, yt + TAR_A / 2))
            tx = cx
        else:
            yt, tx = y_dip, cx + RAMO / 2
        sopra.append(f'<rect class="tar" x="{tx - TAR_L / 2:.0f}" '
                     f'y="{yt - TAR_A / 2:.0f}" width="{TAR_L}" '
                     f'height="{TAR_A}" rx="4"/>')
        sopra.append(f'<text class="lbs trt" x="{tx}" y="{yt + 5:.0f}">'
                     f'taratura</text>')
        scatole.append(_rett(tx, yt, TAR_L, TAR_A))

        sopra.append(f'<circle class="somma" cx="{cx}" cy="{y:.0f}" r="10"/>')
        sopra.append(f'<text class="lbl sgn" x="{cx}" y="{y + 5:.0f}">+</text>')
        scatole.append(_rett(cx, y, 20, 20))

    alto, basso = _stazioni()[-1] - SU, Y_BASSO + GIU
    y0, segmenti = alto, []
    for a, b in sorted(buchi):
        segmenti.append((y0, a))
        y0 = b
    segmenti.append((y0, basso))
    for a, b in segmenti:
        linee.append(f'<line class="corrimano" x1="{cx}" y1="{a:.0f}" '
                     f'x2="{cx}" y2="{b:.0f}"/>')
    return linee + sopra, buchi, scatole


def costruisci() -> Figura:
    sx, dx = 150, 452
    corpo, anim = [], []

    sinistra, buchi_sx, scat_sx = _torre(sx, True)
    destra, buchi_dx, scat_dx = _torre(dx, False)
    corpo += sinistra + destra

    assert len(buchi_sx) == STAZIONI, f"{len(buchi_sx)} interruzioni su {STAZIONI}"
    assert not buchi_dx, f"a destra il corrimano e' interrotto {len(buchi_dx)} volte"
    # le righe stanno sotto, le scatole e le scritte sopra: se qualcuno rimette
    # il disegno nell'ordine in cui si calcola, la parola «taratura» torna
    # sbarrata e il piu' della sommatoria torna a leggersi come dei puntini.
    for nome, torre in (("sinistra", sinistra), ("destra", destra)):
        righe = [i for i, e in enumerate(torre)
                 if 'class="ramo"' in e or 'class="corrimano"' in e]
        scatole = [i for i, e in enumerate(torre)
                   if 'class="tar"' in e or 'class="somma"' in e]
        assert righe and scatole, nome
        assert max(righe) < min(scatole), (
            f"torre {nome}: una riga e' disegnata dopo una scatola "
            f"({max(righe)} > {min(scatole)}), quindi la attraversa")
    pila = SOTTOSTRATI * BLOCCHI_PILA
    assert pila == 192, pila
    for scat in (scat_sx, scat_dx):
        for i, a in enumerate(scat):
            for b in scat[i + 1:]:
                assert not _si_toccano(a, b), f"scatole sovrapposte: {a} {b}"

    alto, basso = _stazioni()[-1] - SU, Y_BASSO + GIU
    for cx in (sx, dx):
        corpo.append(f'<circle class="giu" cx="{cx}" cy="{basso:.0f}" r="7" '
                     f'style="animation:scende var(--d) infinite"/>')
    anim.append(keyframes("scende", [
        (0.0, f"transform:translateY({alto - basso:.0f}px)"),
        (10.0, f"transform:translateY({alto - basso:.0f}px)"),
        (90.0, "transform:translateY(0px)"),
        (100.0, "transform:translateY(0px)")]))

    for cx, titolo, formula, conto in (
            (sx, "post-LN, il montaggio del 2017",
             "LayerNorm(x + Sub(x))",
             f"il corrimano attraversa {len(buchi_sx)} tarature"),
            (dx, "pre-LN, i modelli di oggi",
             "x + Sub(LayerNorm(x))",
             "il corrimano non ne attraversa nessuna")):
        corpo += [
            f'<text class="ttl" x="{cx - 96}" y="34">{titolo}</text>',
            f'<text class="lbs" x="{cx - 96}" y="56">{formula}</text>',
            f'<text class="lbl" x="{cx - 96}" y="{basso + 56:.0f}">{conto}</text>',
            f'<text class="lbs stz" x="{cx}" y="{alto - 12:.0f}">uscita</text>',
            f'<text class="lbs stz" x="{cx}" y="{basso + 26:.0f}">ingresso</text>',
        ]

    corpo += [
        f'<text class="lbs" x="{sx - 96}" y="{basso + 82:.0f}">il pallino è il '
        f'gradiente, che scende dall\'uscita verso l\'ingresso</text>',
        f'<text class="lbs" x="{sx - 96}" y="{basso + 104:.0f}">in una pila da '
        f'{BLOCCHI_PILA} blocchi, due sotto-strati per blocco, fanno {pila} '
        f'interruzioni contro nessuna</text>',
        f'<text class="lbs" x="{sx - 96}" y="{basso + 126:.0f}">(qui i blocchi '
        f'disegnati sono {BLOCCHI}, cioè {STAZIONI} sommatorie)</text>',
    ]

    return Figura(
        larghezza=760, altezza=int(basso + 144),
        alt=f"Due torri affiancate, ciascuna con {BLOCCHI} blocchi da "
            f"{SOTTOSTRATI} sotto-strati. In ognuna una retta verticale, la "
            f"scorciatoia, va dall'ingresso in fondo all'uscita in cima, e da "
            f"essa si stacca a ogni sotto-strato un ramo che attraversa un "
            f"riquadro («attenzione» o «lavoro a parte») e rientra in una "
            f"sommatoria segnata con un più. A sinistra, intestata «post-LN, "
            f"il montaggio del 2017» e «LayerNorm(x + Sub(x))», una taratura "
            f"sta sulla scorciatoia subito sopra ogni sommatoria e la "
            f"interrompe {len(buchi_sx)} volte. A destra, intestata «pre-LN, i "
            f"modelli di oggi» e «x + Sub(LayerNorm(x))», le tarature stanno "
            f"sui rami all'ingresso dei sotto-strati e la scorciatoia corre "
            f"intera dall'alto in basso. Un pallino, il gradiente, scende "
            f"lungo le due scorciatoie. In fondo il conto: in una pila da "
            f"{BLOCCHI_PILA} blocchi sono {pila} interruzioni contro nessuna.",
        corpo="".join(corpo),
        stile=f"""    .corrimano {{ stroke:{TERRACOTTA}; stroke-width:5; stroke-linecap:round; }}
    .ramo {{ stroke:{TEAL}; stroke-width:2; fill:none; }}
    .stanza {{ fill:{TEAL}; fill-opacity:0.14; stroke:{TEAL}; stroke-width:1.5; }}
    .tar {{ fill:{CREAM}; stroke:{OCRA}; stroke-width:2; }}
    .somma {{ fill:{CREAM}; stroke:{TERRACOTTA}; stroke-width:2; }}
    .sgn, .stz, .trt {{ text-anchor:middle; }}
    .giu {{ fill:{INK}; }}""",
        animazioni=anim,
        durata=7.0,
        fermi=".giu",
    )
