"""Il dibattito che si stringe su un ramo solo, turno per turno.

Il tempo qui è il contenuto. L'albero delle affermazioni non si percorre: si
apre un ramo per volta, e a scegliere quale non è il giudice ma il contendente
che vuole vincere, cioè quello che ha tutto l'interesse a puntare il dito dove
crede ci sia la falla. Su una figura ferma resterebbe un albero con una strada
colorata, e non si vedrebbe *chi* la sceglie né *quando*, che è tutto quello
che c'è da capire.

I numeri non sono scritti a mano: `traccia()` percorre davvero l'albero e
restituisce l'ordine dei turni, il ramo aperto a ogni giro e la foglia su cui
il dibattito si ferma. Da lì escono anche i due conti della didascalia, le
catene di ragionamento possibili e le affermazioni che il giudice si trova
davanti.

Lo stato di riposo è la fine: l'albero aperto per intero, il ramo contestato
acceso, la foglia sotto la lente del giudice.
"""

from paithon_svg import *

NOME = "ramo-contestato"
TITOLO = "Il ramo che i due scelgono di contestare"

# Quante alternative offre ogni scomposizione, e quanti giri dura il dibattito.
RAMI, GIRI = 3, 2

# Quale ramo contesta chi sta all'opposizione, giro per giro (0 = il primo a
# sinistra). Sono le sue mosse: le sceglie lui, ed è il punto della figura.
CONTESTATI = [1, 2]


# --------------------------------------------------------------------------
# Il dibattito, percorso per davvero: da qui escono tutti i numeri
# --------------------------------------------------------------------------
def traccia():
    """La sceneggiatura del dibattito, in ordine di accadimento.

    Ogni evento e' una tupla `(cosa, giro)` e diventa un momento della
    timeline: `("afferma", 0)` la tesi di partenza, poi per ogni giro
    `("scompone", g)` e `("contesta", g)`, infine `("giudica", GIRI)`.

    Torna anche il cammino: la lista dei rami scelti, che dice quale nodo di
    ogni livello sta sulla strada aperta.
    """
    eventi = [("afferma", 0)]
    cammino = []
    for g in range(GIRI):
        eventi.append(("scompone", g))
        eventi.append(("contesta", g))
        cammino.append(CONTESTATI[g])
    eventi.append(("giudica", GIRI))
    return eventi, cammino


def conti(cammino):
    """I due numeri che la didascalia promette.

    `catene` sono le linee di ragionamento complete che un giudice dovrebbe
    percorrere da solo per decidere; `scritte` sono le affermazioni che i due
    contendenti mettono nero su bianco, cioe' quello che il giudice si trova
    davanti; `controllate` e' la sola su cui deve pronunciarsi.
    """
    catene = RAMI ** GIRI
    scritte = RAMI * GIRI
    return catene, scritte, 1


def verifica(eventi, cammino) -> None:
    """La figura promette una strettoia: c'e' davvero, ed e' quella giusta?"""
    catene, scritte, controllate = conti(cammino)
    assert len(cammino) == GIRI, \
        f"il dibattito deve durare {GIRI} giri, ne ha percorsi {len(cammino)}"
    assert all(0 <= r < RAMI for r in cammino), \
        "un ramo contestato cade fuori dalla scomposizione"
    assert controllate < scritte < catene, (
        "questa figura esiste per mostrare una strettoia: se le affermazioni "
        f"scritte ({scritte}) non sono meno delle catene possibili ({catene}), "
        "o se il giudice non ne controlla una sola, non c'e' niente da vedere")
    # La didascalia e l'`:alt:` della pagina promettono tre numeri precisi:
    # nove catene, sei affermazioni, una controllata. Un collaudo si scrive
    # leggendo la didascalia, non piu' debole di quella.
    assert (catene, scritte, controllate) == (9, 6, 1), (
        f"la didascalia dice nove catene, sei affermazioni e una controllata: "
        f"qui sono {catene}, {scritte} e {controllate}")
    assert len(set(cammino)) > 1, (
        "l'`:alt:` dice che i due giri contestano rami diversi: con lo stesso "
        "indice due volte la figura racconta una strada dritta")


# --------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------
LARG, ALT = 760, 470
Y = [92, 226, 360]              # tesi, primo giro, secondo giro
MEZZO = LARG / 2
PASSO_1, PASSO_2 = 232, 76      # apertura dei rami, livello per livello
R_NODO = 26

INIZIO, FINE = 6.0, 86.0        # la coda ferma serve a leggere la conclusione


def x_nodo(livello: int, cammino: list[int], indice: int) -> float:
    """Ascissa del nodo `indice` al `livello`, sotto il ramo gia' aperto."""
    if livello == 0:
        return MEZZO
    if livello == 1:
        return MEZZO + (indice - (RAMI - 1) / 2) * PASSO_1
    padre = x_nodo(1, cammino, cammino[0])
    return padre + (indice - (RAMI - 1) / 2) * PASSO_2


def costruisci() -> Figura:
    eventi, cammino = traccia()
    verifica(eventi, cammino)
    catene, scritte, controllate = conti(cammino)

    istante = {e: INIZIO + (FINE - INIZIO) * k / (len(eventi) - 1)
               for k, e in enumerate(eventi)}

    corpo, anim = [], []

    def acceso(nome: str, evento, dentro: str) -> str:
        """Invisibile fino all'istante dell'evento, poi visibile fino in fondo.

        Lo stato di riposo (nessuna animazione) e' quindi «visibile»: e' cio'
        che la stampa deve vedere.
        """
        t = istante[evento]
        anim.append(keyframes(nome, [(0.0, "opacity:0"),
                                     (max(t - 1.4, 0.0), "opacity:0"),
                                     (t, "opacity:1"),
                                     (100.0, "opacity:1")]))
        return (f'<g style="animation:{nome} var(--d) linear infinite">'
                f'{dentro}</g>')

    # --- i rami, disegnati per primi cosi' restano sotto ai nodi -----------
    for g in range(GIRI):
        padre_i = cammino[g - 1] if g else 0
        x_padre = x_nodo(g, cammino, padre_i)
        for i in range(RAMI):
            x = x_nodo(g + 1, cammino, i)
            classe = "ram" if i == cammino[g] else "ram morto"
            corpo.append(acceso(
                f"ram{g}{i}", ("scompone", g),
                f'<line class="{classe}" x1="{x_padre}" y1="{Y[g] + R_NODO}" '
                f'x2="{x:.1f}" y2="{Y[g + 1] - R_NODO}"/>'))

    # --- la tesi di partenza ----------------------------------------------
    corpo.append(acceso(
        "tesi", ("afferma", 0),
        f'<circle class="nodo afferma" cx="{MEZZO}" cy="{Y[0]}" r="{R_NODO}"/>'
        f'<text class="val" x="{MEZZO}" y="{Y[0] + 7}">?</text>'))
    corpo.append(acceso(
        "etesi", ("afferma", 0),
        f'<text class="lbl" x="{MEZZO}" y="{Y[0] - 42}" text-anchor="middle">'
        f'la domanda, e la risposta che il primo sostiene</text>'))

    # --- i nodi dei due giri, e l'etichetta di riga a sinistra -------------
    # Le etichette stanno nel margine sinistro, sopra la loro riga di nodi, e
    # vanno tenute corte: piu' lunghe di cosi' la coda della prima incrocia il
    # ramo che scende dalla radice al nodo 1, e nei fermi immagine si legge una
    # parola appoggiata sopra una linea.
    for g, riga in enumerate(("primo giro: tre parti",
                              "secondo giro: altre tre parti")):
        corpo.append(acceso(
            f"riga{g}", ("scompone", g),
            f'<text class="lbs" x="26" y="{Y[g + 1] - R_NODO - 16}">'
            f'{riga}</text>'))
    for g in range(GIRI):
        for i in range(RAMI):
            x = x_nodo(g + 1, cammino, i)
            scelto = i == cammino[g]
            corpo.append(acceso(
                f"nodo{g}{i}", ("scompone", g),
                f'<circle class="nodo {"afferma" if scelto else "morto"}" '
                f'cx="{x:.1f}" cy="{Y[g + 1]}" r="{R_NODO}"/>'
                f'<text class="val {"" if scelto else "fioco"}" x="{x:.1f}" '
                f'y="{Y[g + 1] + 7}">{i + 1}</text>'))
            if scelto:
                # l'anello che segna la contestazione arriva un turno dopo la
                # scomposizione: prima il primo parla, poi il secondo sceglie
                corpo.append(acceso(
                    f"segno{g}", ("contesta", g),
                    f'<circle class="contesta" cx="{x:.1f}" cy="{Y[g + 1]}" '
                    f'r="{R_NODO + 7}"/>'))

    # --- il giudice, sull'unica affermazione su cui deve pronunciarsi ------
    x_fine = x_nodo(GIRI, cammino, cammino[-1])
    corpo.append(acceso(
        "giudice", ("giudica", GIRI),
        f'<line class="tirante" x1="{x_fine:.1f}" y1="{Y[GIRI] + R_NODO}" '
        f'x2="{x_fine:.1f}" y2="{Y[GIRI] + 52}"/>'
        f'<text class="lbl" x="{x_fine:.1f}" y="{Y[GIRI] + 72}" '
        f'text-anchor="middle">il giudice controlla questa, e basta</text>'))

    # --- le due legende, in basso a sinistra dove non c'e' niente ----------
    for dy, colore, testo in ((0, "afferma", "il primo afferma e scompone"),
                              (22, "contesta", "il secondo sceglie dove attaccare")):
        y_leg = ALT - 68 + dy
        corpo.append(f'<circle class="nodo {colore}" cx="34" cy="{y_leg}" r="8"/>'
                     f'<text class="lbs" x="50" y="{y_leg + 4}">{testo}</text>')

    # --- la riga in basso: i due conti -------------------------------------
    y = ALT - 18
    corpo.append(f'<text class="lbs" x="30" y="{y}">'
                 f'catene di ragionamento possibili: {catene}</text>')
    corpo.append(acceso(
        "conto", ("giudica", GIRI),
        f'<text class="lbs" x="{LARG - 30}" y="{y}" text-anchor="end">'
        f'affermazioni messe per iscritto: {scritte} '
        f'· controllate dal giudice: {controllate}</text>'))

    return Figura(
        larghezza=LARG, altezza=ALT,
        alt="Un albero di affermazioni che si apre un ramo per volta. In cima "
            "un nodo con un punto interrogativo, la domanda e la risposta che "
            "il primo contendente sostiene. Sotto compaiono tre nodi, la sua "
            "scomposizione in tre parti; il secondo contendente ne cerchia uno, "
            "quello centrale, e gli altri due restano grigi e non vengono più "
            "aperti. Sotto il nodo cerchiato compaiono altri tre nodi, e il "
            "secondo ne cerchia uno diverso, il terzo da sinistra; gli altri "
            "due restano grigi. Da quel nodo scende un tratto fino alla scritta "
            "«il giudice controlla questa, e basta». In basso i due conti: "
            "catene di ragionamento possibili nove, affermazioni messe per "
            "iscritto sei, controllate dal giudice una.",
        corpo="".join(corpo),
        stile=f"""    .ram      {{ stroke:{BORDER_STRONG}; stroke-width:2; fill:none; }}
    .morto    {{ stroke:{BORDER}; }}
    .nodo     {{ stroke-width:2.5; fill:{CREAM}; }}
    .afferma  {{ stroke:{TERRACOTTA}; }}
    .contesta {{ stroke:{TEAL}; stroke-width:2.5; fill:none;
                 stroke-dasharray:6 5; }}
    .tirante  {{ stroke:{TEAL}; stroke-width:2; }}
    .val      {{ font-family:{SANS}; font-size:17px; font-weight:600;
                 fill:{INK}; text-anchor:middle; }}
    .fioco    {{ fill:{FG_MUTED}; }}""",
        animazioni=anim,
        durata=10.0,
        fermi="g",
    )
