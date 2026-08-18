"""La potatura alfa-beta mentre avviene, sull'albero d'esempio della sezione.

Il tempo qui è il contenuto: alfa-beta non è una formula, è un *ordine*. Le
foglie si scoprono da sinistra a destra, e a un certo punto ne restano due che
non si guardano più, perché si è già dimostrato che non possono cambiare la
risposta. Su una figura ferma quelle due sarebbero grigie e basta, e non si
capirebbe *quando* sono diventate inutili, che è tutto quello che c'è da
capire.

I numeri non sono disegnati a mano: `traccia()` esegue davvero alfa-beta sulle
nove foglie e restituisce l'ordine di visita, il taglio, i valori dei nodi
interni e quello della radice. Cambiando una foglia la figura si riadatta, o
un'asserzione si lamenta.

Lo stato di riposo è la fine: sette foglie scoperte, due barrate, il 3 alla
radice.
"""

from paithon_svg import *

NOME = "alfabeta-pota"
TITOLO = "La potatura alfa-beta mentre avviene"

# L'albero d'esempio della sezione: tre mosse per chi muove per primo, tre
# risposte dell'avversario sotto ciascuna. Sono gli stessi numeri del testo.
FOGLIE = [[3, 12, 8], [2, 4, 6], [14, 5, 2]]

MOLTO = 10 ** 9


# --------------------------------------------------------------------------
# L'algoritmo, eseguito per davvero: da qui escono tutti i numeri della figura
# --------------------------------------------------------------------------
def traccia():
    """Alfa-beta su FOGLIE: la sceneggiatura, in ordine di accadimento.

    Ogni evento è una tupla, e diventa un momento della timeline:
    `("foglia", g, i)`, `("taglio", g, None)`, `("min", g, None)`,
    `("alfa", g, None)`, `("radice", None, None)`.
    """
    eventi, minimi, tagliate = [], [], []
    alfa = -MOLTO
    for g, gruppo in enumerate(FOGLIE):
        v = MOLTO
        for i, foglia in enumerate(gruppo):
            eventi.append(("foglia", g, i))
            v = min(v, foglia)
            if v <= alfa:                    # l'avversario non mi ci farebbe arrivare
                salti = list(range(i + 1, len(gruppo)))
                if salti:
                    eventi.append(("taglio", g, None))
                    tagliate += [(g, k) for k in salti]
                break
        minimi.append(v)
        eventi.append(("min", g, None))
        if v > alfa:
            alfa = v
            eventi.append(("alfa", g, None))
    eventi.append(("radice", None, None))
    return eventi, minimi, alfa, tagliate


def verifica(eventi, minimi, radice, tagliate) -> None:
    """La figura promette una potatura: c'è davvero, ed è quella giusta?"""
    guardate = sum(1 for e in eventi if e[0] == "foglia")
    totale = sum(len(g) for g in FOGLIE)
    assert tagliate, "senza nessun taglio questa figura non ha niente da mostrare"
    assert guardate + len(tagliate) == totale, \
        f"foglie guardate {guardate} + tagliate {len(tagliate)} != {totale}"
    assert radice == max(min(g) for g in FOGLIE), \
        "alfa-beta ha restituito un valore diverso dal minimax completo"
    assert minimi[0] == radice, \
        ("la figura racconta che a vincere è il primo gruppo, ed è per questo "
         "che il taglio scatta nel secondo: se cambia, cambia la didascalia")
    # Gli assert qui sopra dicevano solo che un taglio ci fosse. Ma la
    # didascalia e l'`:alt:` della pagina promettono tre numeri precisi (sette
    # foglie su nove, il taglio nel secondo gruppo, il terzo scoperto tutto), e
    # con FOGLIE = [[3,12,8],[7,1,6],[2,5,5]] passavano tutti lasciando l'alt
    # falso su tre punti. Un collaudo si scrive leggendo la didascalia.
    assert guardate == 7 and tagliate == [(1, 1), (1, 2)], \
        (f"l'`:alt:` della pagina dice «sette foglie su nove», il taglio nel "
         f"secondo gruppo e il terzo scoperto tutto: qui sono {guardate} "
         f"guardate e tagliate {tagliate}")


# --------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------
LARG, ALT = 760, 430
Y_RADICE, Y_MIN, Y_FOGLIA = 74, 202, 330
X_MIN = [148, 380, 612]
PASSO = 64

INIZIO, FINE = 5.0, 88.0      # la coda ferma serve a leggere la conclusione


def x_foglia(g: int, i: int) -> float:
    return X_MIN[g] + (i - 1) * PASSO


def appari(nome: str, t: float) -> str:
    """Invisibile fino all'istante `t`, poi visibile fino alla fine.

    Lo stato di riposo (nessuna animazione) è quindi «visibile», che è ciò che
    la stampa deve vedere.
    """
    return keyframes(nome, [(0.0, "opacity:0"),
                            (max(t - 1.2, 0.0), "opacity:0"),
                            (t, "opacity:1"),
                            (100.0, "opacity:1")])


def costruisci() -> Figura:
    eventi, minimi, radice, tagliate = traccia()
    verifica(eventi, minimi, radice, tagliate)

    istante = {e: INIZIO + (FINE - INIZIO) * k / (len(eventi) - 1)
               for k, e in enumerate(eventi)}

    corpo, anim = [], []

    def acceso(nome: str, evento, dentro: str) -> str:
        anim.append(appari(nome, istante[evento]))
        return (f'<g style="animation:{nome} var(--d) linear infinite">'
                f'{dentro}</g>')

    def spegni(nome: str, evento, dentro: str) -> str:
        """Visibile fino all'istante dell'evento, poi via. A riposo: via."""
        t = istante[evento]
        anim.append(keyframes(nome, [(0.0, "opacity:1"),
                                     (max(t - 1.2, 0.0), "opacity:1"),
                                     (t, "opacity:0"),
                                     (100.0, "opacity:0")]))
        return (f'<g style="opacity:0;animation:{nome} var(--d) linear infinite">'
                f'{dentro}</g>')

    # --- i rami, disegnati per primi così restano sotto ai nodi ------------
    # Quelli potati esistono in due copie sovrapposte: quella piena si spegne
    # nell'istante del taglio e quella tratteggiata si accende. Disegnare
    # subito il tratteggio annuncerebbe il taglio prima che accada, cioè
    # racconterebbe la fine all'inizio, che è esattamente ciò che questa
    # figura esiste per non fare.
    for g in range(3):
        corpo.append(f'<line class="ram" x1="{LARG / 2}" y1="{Y_RADICE + 22}" '
                     f'x2="{X_MIN[g]}" y2="{Y_MIN - 22}"/>')
        for i in range(len(FOGLIE[g])):
            geom = (f'x1="{X_MIN[g]}" y1="{Y_MIN + 22}" '
                    f'x2="{x_foglia(g, i)}" y2="{Y_FOGLIA - 20}"')
            if (g, i) in tagliate:
                corpo.append(spegni(f"sram{g}{i}", ("taglio", g, None),
                                    f'<line class="ram" {geom}/>'))
                corpo.append(acceso(f"tram{g}{i}", ("taglio", g, None),
                                    f'<line class="ram spento" {geom}/>'))
            else:
                corpo.append(f'<line class="ram" {geom}/>')

    # --- la radice ---------------------------------------------------------
    corpo.append(f'<circle class="nodo max" cx="{LARG / 2}" cy="{Y_RADICE}" r="22"/>')
    corpo.append(acceso(
        "vradice", ("radice", None, None),
        f'<text class="val" x="{LARG / 2}" y="{Y_RADICE + 6}">{radice}</text>'))
    # la legenda in alto a sinistra, dove non c'è niente: appesa alle righe
    # dell'albero i due cartigli finivano fuori dal foglio a destra
    for dy, colore, testo in ((0, "max", "tocca a me, e prendo il massimo"),
                              (24, "min", "tocca a lui, e prende il minimo")):
        corpo.append(f'<circle class="nodo {colore}" cx="46" cy="{34 + dy}" r="8"/>'
                     f'<text class="lbs" x="62" y="{38 + dy}">{testo}</text>')

    # --- i tre nodi dell'avversario ---------------------------------------
    for g in range(3):
        corpo.append(f'<circle class="nodo min" cx="{X_MIN[g]}" cy="{Y_MIN}" r="22"/>')
        # dove il taglio è scattato il valore non è misurato, è un tetto:
        # scriverlo secco sarebbe una bugia, perché sotto c'è roba mai guardata
        potato = any(t[0] == g for t in tagliate)
        testo = f"≤{minimi[g]}" if potato else f"{minimi[g]}"
        corpo.append(acceso(
            f"vmin{g}", ("min", g, None),
            f'<text class="val" x="{X_MIN[g]}" y="{Y_MIN + 6}">{testo}</text>'))

    # --- le foglie ---------------------------------------------------------
    for g in range(3):
        for i, foglia in enumerate(FOGLIE[g]):
            x = x_foglia(g, i)
            if (g, i) in tagliate:
                corpo.append(acceso(
                    f"vtag{g}{i}", ("taglio", g, None),
                    f'<rect class="cassa spenta" x="{x - 22}" y="{Y_FOGLIA - 20}" '
                    f'width="44" height="40" rx="5"/>'
                    f'<line class="croce" x1="{x - 12}" y1="{Y_FOGLIA - 10}" '
                    f'x2="{x + 12}" y2="{Y_FOGLIA + 12}"/>'
                    f'<line class="croce" x1="{x + 12}" y1="{Y_FOGLIA - 10}" '
                    f'x2="{x - 12}" y2="{Y_FOGLIA + 12}"/>'))
            else:
                corpo.append(acceso(
                    f"vfog{g}{i}", ("foglia", g, i),
                    f'<rect class="cassa" x="{x - 22}" y="{Y_FOGLIA - 20}" '
                    f'width="44" height="40" rx="5"/>'
                    f'<text class="val" x="{x}" y="{Y_FOGLIA + 6}">{foglia}</text>'))

    # --- la riga in basso: che cosa mi sono già assicurato -----------------
    y = 398
    corpo.append(f'<text class="lbs" x="40" y="{y}">'
                 f'quello che mi sono già assicurato:</text>')
    x_al = 322
    for g in range(3):
        if ("alfa", g, None) not in istante:
            continue
        corpo.append(acceso(f"valfa{g}", ("alfa", g, None),
                            f'<text class="alfa" x="{x_al}" y="{y}">{minimi[g]}</text>'))
        x_al += 28
    guardate = sum(1 for e in eventi if e[0] == "foglia")
    totale = sum(len(g) for g in FOGLIE)
    corpo.append(acceso(
        "vconto", ("radice", None, None),
        f'<text class="lbs" x="{LARG - 40}" y="{y}" text-anchor="end">'
        f'foglie guardate: {guardate} su {totale}</text>'))

    return Figura(
        larghezza=LARG, altezza=ALT,
        alt="Un albero a due livelli. In cima un nodo tondo, chi muove per "
            "primo, che prende il massimo; sotto, tre nodi tondi dell'avversario, "
            "che prendono il minimo; sotto ancora nove caselle con i numeri 3, "
            "12, 8, poi 2, 4, 6, poi 14, 5, 2. Le caselle si scoprono da "
            "sinistra a destra. Scoperte le prime tre, il nodo sopra di esse "
            "segna 3, e in basso compare il 3 come guadagno già assicurato. Nel "
            "secondo gruppo si scopre soltanto il 2: le due caselle che restano "
            "e i loro rami diventano grigi e barrati, e il loro nodo segna "
            "«minore o uguale a 2», perché quel valore nessuno l'ha misurato "
            "fino in fondo. Il terzo gruppo si scopre tutto, 14, 5 e 2, e segna "
            "2. Alla fine la radice segna 3, e la riga in basso conta sette "
            "foglie guardate su nove.",
        corpo="".join(corpo),
        stile=f"""    .ram    {{ stroke:{BORDER_STRONG}; stroke-width:2; }}
    .spento {{ stroke:{BORDER}; stroke-dasharray:5 5; }}
    .nodo   {{ stroke-width:2.5; fill:{CREAM}; }}
    .max    {{ stroke:{TERRACOTTA}; }}
    .min    {{ stroke:{TEAL}; }}
    .cassa  {{ fill:{CREAM}; stroke:{BORDER_STRONG}; stroke-width:2; }}
    .spenta {{ stroke:{BORDER}; }}
    .croce  {{ stroke:{BORDER_STRONG}; stroke-width:2.5; }}
    .val    {{ font-family:{SANS}; font-size:17px; font-weight:600;
               fill:{INK}; text-anchor:middle; }}
    .alfa   {{ font-family:{SANS}; font-size:17px; font-weight:600;
               fill:{TERRACOTTA}; text-anchor:middle; }}""",
        animazioni=anim,
        durata=11.0,
        fermi="g",
    )
