"""I quattro ingredienti di un problema, ciascuno nel suo posto del rettangolo.

`PINN/applicazioni-limiti.md` elenca le quattro cose che restano fissate quando
l'addestramento di una PINN finisce (il dominio, la condizione iniziale, le
condizioni al contorno, la sorgente) e le racconta con la sbarra di ferro che
si scalda. Tre delle quattro il lettore se le vede (la sbarra sul tavolo, la
fiamma e il ghiaccio ai due capi, com'era calda prima); la quarta, la sorgente,
gli si dice e basta, perche' nella scena non ha un posto.

Il posto ce l'ha, e questa figura e' quello: il rettangolo spazio-tempo che
`PINN/overview.md` descrive a parole («per la sbarra ne servono due, il tempo e
il punto lungo la sbarra, e i puntini riempiono un rettangolo»). Messi li', i
quattro ingredienti smettono di essere un elenco e diventano quattro posti:
il rettangolo intero, il suo bordo di sotto, i suoi due fianchi, e il dentro.

E si vede la cosa che l'elenco non dice: **in cima non si impone niente**. Le
condizioni stanno sul bordo di sotto e sui fianchi, non sul lato finale; il
tempo finale non e' un dato del problema, e' dove il conto arriva.

Niente numeri: la figura non ne mostra nessuno, quindi non ce ne sono da
calcolare. Quello che `verifica()` difende e' la geometria, che qui e' la
sostanza: che i quattro posti siano davvero quattro (nessuno dentro l'altro,
nessuno sovrapposto a un altro se non negli spigoli), che insieme facciano il
rettangolo intero, e che il lato di sopra resti nudo.

Ferma, e non poteva essere altro: qui il tempo non scorre, e' un asse.
"""

from paithon_svg import *

NOME = "quattro-ingredienti"
TITOLO = "i quattro ingredienti, sul rettangolo spazio-tempo"

# --------------------------------------------------------------------------
# Geometria del rettangolo
# --------------------------------------------------------------------------
LARG, ALT = 760, 420
XS, XD = 250.0, 560.0          # i due capi della sbarra
YB, YA = 310.0, 110.0          # l'istante iniziale (in basso) e quello finale

# I quattro posti, come dati: tre segmenti del bordo e una regione.
FIANCO_SX = ("condizioni al contorno", (XS, YB), (XS, YA))
FIANCO_DX = ("condizioni al contorno", (XD, YB), (XD, YA))
SOTTO = ("condizione iniziale", (XS, YB), (XD, YB))
SOPRA = ("niente", (XS, YA), (XD, YA))
DENTRO = ("sorgente", (XS, YA), (XD, YB))      # il rettangolo aperto

# Le crocette della sorgente: il calore che entra da dentro, punto per punto.
CROCETTE = [(0.16, 0.14), (0.40, 0.18), (0.64, 0.12), (0.86, 0.17),
            (0.14, 0.86), (0.38, 0.82), (0.62, 0.88), (0.88, 0.84)]


def verifica() -> None:
    """Difende quello che la didascalia promette: quattro posti, e la cima nuda."""
    # I tre segmenti con una condizione sono davvero tre lati distinti.
    lati = [FIANCO_SX, FIANCO_DX, SOTTO]
    assert len({(a, b) for _, a, b in lati}) == 3, \
        "due delle condizioni stanno sullo stesso lato"
    assert FIANCO_SX[1][0] == FIANCO_SX[2][0] == XS, "il fianco sinistro non e' verticale"
    assert FIANCO_DX[1][0] == FIANCO_DX[2][0] == XD, "il fianco destro non e' verticale"
    assert SOTTO[1][1] == SOTTO[2][1] == YB, "il bordo di sotto non e' orizzontale"

    # I due fianchi coprono tutta l'altezza, il bordo di sotto tutta la
    # larghezza: le condizioni valgono per ogni istante e per ogni punto.
    for nome, a, b in (FIANCO_SX, FIANCO_DX):
        assert {a[1], b[1]} == {YA, YB}, f"{nome}: il fianco non arriva ai due estremi"
    assert {SOTTO[1][0], SOTTO[2][0]} == {XS, XD}, \
        "la condizione iniziale non copre tutta la sbarra"

    # Il lato di sopra resta nudo: nessuna condizione ci sta sopra.
    assert SOPRA[0] == "niente", "in cima e' finita una condizione"
    for nome, a, b in lati:
        assert not (a[1] == YA and b[1] == YA), \
            f"{nome} sta sul lato di sopra, dove non si impone niente"

    # E il dentro e' il rettangolo aperto: nessuna crocetta tocca un bordo.
    for u, v in CROCETTE:
        assert 0.0 < u < 1.0 and 0.0 < v < 1.0, \
            f"una crocetta della sorgente cade sul bordo invece che dentro ({u}, {v})"
    assert len(set(CROCETTE)) == len(CROCETTE), "due crocette nello stesso punto"


def crocetta(x: float, y: float, r: float = 6.0) -> str:
    return (f'<path class="src" d="M {x - r:.1f} {y:.1f} h {2 * r:.1f} '
            f'M {x:.1f} {y - r:.1f} v {2 * r:.1f}"/>')


def costruisci() -> Figura:
    verifica()

    corpo = ['<defs>'
             '<marker id="qas" viewBox="0 0 10 10" refX="8.5" refY="5" '
             'markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
             f'<path d="M 0 1 L 9 5 L 0 9 z" fill="{FG_MUTED}"/></marker>'
             '</defs>']

    # ---- il dentro, e le crocette che sono la sorgente --------------------
    corpo.append(f'<rect class="dom" x="{XS:.1f}" y="{YA:.1f}" '
                 f'width="{XD - XS:.1f}" height="{YB - YA:.1f}"/>')
    for u, v in CROCETTE:
        corpo.append(crocetta(XS + u * (XD - XS), YA + v * (YB - YA)))

    # ---- i tre lati con una condizione, e quello nudo ---------------------
    corpo.append(f'<line class="cont" x1="{XS:.1f}" y1="{YB:.1f}" '
                 f'x2="{XS:.1f}" y2="{YA:.1f}"/>')
    corpo.append(f'<line class="cont" x1="{XD:.1f}" y1="{YB:.1f}" '
                 f'x2="{XD:.1f}" y2="{YA:.1f}"/>')
    corpo.append(f'<line class="iniz" x1="{XS:.1f}" y1="{YB:.1f}" '
                 f'x2="{XD:.1f}" y2="{YB:.1f}"/>')
    corpo.append(f'<line class="nudo" x1="{XS:.1f}" y1="{YA:.1f}" '
                 f'x2="{XD:.1f}" y2="{YA:.1f}"/>')

    # ---- i due assi -------------------------------------------------------
    corpo.append(f'<line class="ass" x1="{XS - 104:.1f}" y1="{YB:.1f}" '
                 f'x2="{XS - 104:.1f}" y2="{YA:.1f}" marker-end="url(#qas)"/>')
    corpo.append(f'<text class="lbs" x="{XS - 104:.1f}" y="{YA - 12:.1f}" '
                 f'text-anchor="middle">il tempo</text>')
    corpo.append(f'<line class="ass" x1="{XS:.1f}" y1="{YB + 48:.1f}" '
                 f'x2="{XD:.1f}" y2="{YB + 48:.1f}" marker-end="url(#qas)"/>')
    corpo.append(f'<text class="lbs" x="{(XS + XD) / 2:.1f}" '
                 f'y="{YB + 68:.1f}" text-anchor="middle">'
                 f'lungo la sbarra</text>')

    # ---- le etichette dei quattro posti -----------------------------------
    corpo.append(f'<text class="et" x="{(XS + XD) / 2:.1f}" y="{YA - 36:.1f}" '
                 f'text-anchor="middle">in cima non si impone niente</text>')
    corpo.append(f'<text class="ets" x="{(XS + XD) / 2:.1f}" y="{YA - 18:.1f}" '
                 f'text-anchor="middle">il tempo finale non è un dato: '
                 f'è dove il conto arriva</text>')

    corpo.append(f'<text class="etc" x="{XS - 14:.1f}" y="{YA + 46:.1f}" '
                 f'text-anchor="end">la fiamma</text>')
    corpo.append(f'<text class="etc" x="{XD + 14:.1f}" y="{YA + 46:.1f}">'
                 f'il ghiaccio</text>')
    corpo.append(f'<text class="ets" x="{XS - 14:.1f}" y="{YA + 65:.1f}" '
                 f'text-anchor="end">condizione</text>')
    corpo.append(f'<text class="ets" x="{XS - 14:.1f}" y="{YA + 81:.1f}" '
                 f'text-anchor="end">al contorno</text>')
    corpo.append(f'<text class="ets" x="{XD + 14:.1f}" y="{YA + 65:.1f}">'
                 f'condizione</text>')
    corpo.append(f'<text class="ets" x="{XD + 14:.1f}" y="{YA + 81:.1f}">'
                 f'al contorno</text>')

    mezzo = (XS + XD) / 2
    corpo.append(f'<text class="etd" x="{mezzo:.1f}" '
                 f'y="{(YA + YB) / 2 - 14:.1f}" text-anchor="middle">'
                 f'la sorgente</text>')
    corpo.append(f'<text class="ets" x="{mezzo:.1f}" '
                 f'y="{(YA + YB) / 2 + 6:.1f}" text-anchor="middle" '
                 f'fill="{TEAL}">che cosa la scalda da dentro,</text>')
    corpo.append(f'<text class="ets" x="{mezzo:.1f}" '
                 f'y="{(YA + YB) / 2 + 24:.1f}" text-anchor="middle" '
                 f'fill="{TEAL}">in ogni punto e in ogni istante</text>')

    corpo.append(f'<text class="eti" x="{(XS + XD) / 2:.1f}" y="{YB + 26:.1f}" '
                 f'text-anchor="middle">la condizione iniziale: '
                 f'com\'era calda prima che tutto cominciasse</text>')

    corpo.append(f'<text class="et" x="{XD + 14:.1f}" y="{YB - 22:.1f}">'
                 f'il dominio</text>')
    corpo.append(f'<text class="ets" x="{XD + 14:.1f}" y="{YB - 4:.1f}">'
                 f'tutto il rettangolo</text>')

    return Figura(
        larghezza=LARG, altezza=ALT,
        alt="Un rettangolo: in orizzontale la posizione lungo la sbarra, in "
            "verticale il tempo, che sale. Il rettangolo intero è etichettato "
            "«il dominio». Il suo lato di sotto, marcato in ocra, è «la "
            "condizione iniziale: com'era calda prima che tutto cominciasse». "
            "I due fianchi, marcati in terracotta, sono le «condizioni al "
            "contorno», e portano l'uno l'etichetta «la fiamma», l'altro «il "
            "ghiaccio». Dentro il rettangolo otto crocette in teal, sparse e "
            "mai appoggiate a un bordo, sono «la sorgente: che cosa la scalda "
            "da dentro, in ogni punto e in ogni istante». Il lato di sopra è "
            "l'unico lasciato con il tratto sottile del bordo, e sopra di lui "
            "c'è scritto che lì non si impone niente, perché il tempo finale "
            "non è un dato del problema ma il punto in cui il conto arriva.",
        corpo="".join(corpo),
        stile=f"""    .dom  {{ fill:{TEAL}; fill-opacity:0.06; stroke:none; }}
    .nudo {{ stroke:{BORDER_STRONG}; stroke-width:1.4; stroke-dasharray:5 4; }}
    .cont {{ stroke:{TERRACOTTA}; stroke-width:5; stroke-linecap:round; }}
    .iniz {{ stroke:{OCRA}; stroke-width:5; stroke-linecap:round; }}
    .src  {{ stroke:{TEAL}; stroke-width:2; stroke-linecap:round; fill:none; }}
    .ass  {{ stroke:{FG_MUTED}; stroke-width:1.4; }}
    .et   {{ font-family:{SANS}; font-size:15px; font-weight:700; fill:{INK}; }}
    .etc  {{ font-family:{SANS}; font-size:15px; font-weight:700;
            fill:{TERRACOTTA}; }}
    .eti  {{ font-family:{SANS}; font-size:14px; font-weight:700;
            fill:{INK}; }}
    .etd  {{ font-family:{SANS}; font-size:15px; font-weight:700; fill:{TEAL}; }}
    .ets  {{ font-family:{SANS}; font-size:13px; fill:{FG_MUTED}; }}""",
    )
