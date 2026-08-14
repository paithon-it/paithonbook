"""Decodifica per differenza: due letture, una sottrazione, una parola.

Il tempo qui è il contenuto due volte. La prima perché la decodifica **genera un
token alla volta**, e il rimedio della sezione non tocca i pesi: agisce a ogni
passo, dentro la scelta. La seconda perché ogni passo è fatto di due battute
(«si legge due volte», poi «si sottrae e si sceglie»), e il momento che vale la
clip è uno solo: quello in cui «forchetta» sarebbe stata scritta e non lo è.
Fermo servirebbero otto pannelli.

Tre passi di decodifica, due battute ciascuno, sei stati in tutto:

1. **piatto**: vince a occhi aperti e a occhi chiusi; la sottrazione conferma.
2. **coltello**: alto a occhi aperti e basso a occhi chiusi; la sottrazione lo
   premia, e il margine sulla forchetta passa da 0,04 a 0,45.
3. **bicchiere**: «forchetta» è la più alta *in tutte e due* le letture, e la
   sottrazione la schiaccia sotto «bicchiere», che era seconda.

I numeri non sono scritti a mano. Si dichiarano le due distribuzioni giocattolo
(a occhi aperti e a occhi chiusi) e `contrastiva()` esegue davvero la formula
del capitolo: la combinazione dei due punteggi con il peso α, ristretta alla
rosa dei candidati plausibili (p ≥ β·max) e rinormalizzata. `controlla()`
verifica, fra l'altro, che al terzo passo «forchetta» sia **prima** prima della
correzione e **non** prima dopo: se i valori giocattolo cambiassero rendendo la
dimostrazione falsa, la figura non si genera.

Lo stato di riposo è l'ultima battuta dell'ultimo passo, ed è il fotogramma che
finisce in stampa: la frase completa in alto, e sotto le tre colonne con
«forchetta» altissima nelle prime due e sprofondata nella terza, più la riga che
dice perché. Il fermo insegna da solo il meccanismo intero.
"""

import math

from paithon_svg import *

NOME = "decodifica-per-differenza"
TITOLO = "la decodifica contrastiva sottrae ciò che il modello direbbe comunque"

# Il vocabolario giocattolo. L'ordine delle righe è fisso in tutti i passi:
# è quello che permette all'occhio di seguire una parola da una colonna
# all'altra e da un passo al successivo.
PAROLE = ("piatto", "coltello", "bicchiere", "forchetta", "tovagliolo", "tavolo")

ALFA = 1.0     # α della formula del capitolo: quanto pesa la correzione
BETA = 0.1     # β: la rosa sono le parole che valgono almeno un decimo della prima

# Le due distribuzioni giocattolo, passo per passo. Sono probabilità: la somma
# fa uno, e `controlla()` lo verifica.
PASSI = (
    {
        "scrive": "piatto",
        "aperti": {"piatto": .52, "coltello": .12, "bicchiere": .06,
                   "forchetta": .16, "tovagliolo": .10, "tavolo": .04},
        "chiusi": {"piatto": .30, "coltello": .12, "bicchiere": .09,
                   "forchetta": .26, "tovagliolo": .16, "tavolo": .07},
        "frase": "un piatto,",
        "letture": "due letture: la stessa domanda, con la foto e con la foto illeggibile",
        "scelta": "«piatto» vince in tutte e due le letture: la sottrazione lo conferma",
    },
    {
        "scrive": "coltello",
        "aperti": {"piatto": .13, "coltello": .34, "bicchiere": .10,
                   "forchetta": .30, "tovagliolo": .10, "tavolo": .03},
        "chiusi": {"piatto": .16, "coltello": .14, "bicchiere": .10,
                   "forchetta": .40, "tovagliolo": .12, "tavolo": .08},
        "frase": " un coltello",
        "letture": "di nuovo due letture, e questa volta le due non si somigliano",
        "scelta": "«coltello» è alto solo a occhi aperti: la sottrazione lo premia",
    },
    {
        "scrive": "bicchiere",
        "aperti": {"piatto": .11, "coltello": .15, "bicchiere": .28,
                   "forchetta": .34, "tovagliolo": .09, "tavolo": .03},
        "chiusi": {"piatto": .10, "coltello": .16, "bicchiere": .14,
                   "forchetta": .46, "tovagliolo": .08, "tavolo": .06},
        "frase": " e un bicchiere",
        "letture": "«forchetta» è la più alta in tutte e due le letture",
        "scelta": "la forchetta non c'era: era probabile anche a occhi chiusi",
    },
)

STATI = 2 * len(PASSI)     # due battute per passo: le letture, poi la sottrazione
TENUTA = 0.7               # quanto di ogni fetta è sosta e quanto transizione

# ------------------------------------------------------------------ geometria
LARG = 740
X_LAB = 150                # le etichette di riga finiscono qui (allineate a destra)
COL_X = (162, 352, 542)    # occhi aperti, occhi chiusi, differenza
COL_W = 168
VMAX = 1.0                 # le barre sono probabilità: fondo scala 1
Y_R0, DY_R, H_BAR = 164, 27, 17
Y_TIT, Y_SUB = 122, 138
Y_CAP, Y_N1, Y_N2 = 344, 370, 388
ALT = 404


# ---------------------------------------------------------------- il conto
def contrastiva(passo):
    """La decodifica contrastiva visiva del capitolo, eseguita davvero.

    Restituisce (aperti, chiusi, rosa, p_cd). I logit qui sono i logaritmi
    delle probabilità: differiscono da quelli veri per una costante additiva,
    che la softmax finale cancella.
    """
    aperti = [passo["aperti"][p] for p in PAROLE]
    chiusi = [passo["chiusi"][p] for p in PAROLE]

    soglia = BETA * max(aperti)                       # la rosa dei plausibili
    rosa = [a >= soglia for a in aperti]

    punteggi = [(1 + ALFA) * math.log(a) - ALFA * math.log(c)
                for a, c in zip(aperti, chiusi)]
    m = max(s for s, dentro in zip(punteggi, rosa) if dentro)
    pesi = [math.exp(s - m) if dentro else 0.0
            for s, dentro in zip(punteggi, rosa)]
    tot = sum(pesi)
    return aperti, chiusi, rosa, [p / tot for p in pesi]


def ordina(valori):
    """Le parole dalla più probabile alla meno probabile."""
    return [PAROLE[i] for i in sorted(range(len(PAROLE)),
                                      key=lambda i: -valori[i])]


def controlla(conti):
    """Se la dimostrazione non regge, la figura non si genera.

    Il punto della sezione è uno solo, e o è vero nei numeri o la figura sta
    insegnando una cosa che i suoi stessi dati smentiscono.
    """
    for k, (passo, (aperti, chiusi, rosa, cd)) in enumerate(zip(PASSI, conti), start=1):
        for nome, dist in (("aperti", aperti), ("chiusi", chiusi)):
            if abs(sum(dist) - 1.0) > 1e-9:
                raise ValueError(f"{NOME}: passo {k}, {nome} somma {sum(dist)}")
        if ordina(cd)[0] != passo["scrive"]:
            raise ValueError(f"{NOME}: passo {k} scriverebbe «{ordina(cd)[0]}» "
                             f"e non «{passo['scrive']}»")
        if PAROLE[rosa.index(False)] != "tavolo" or rosa.count(False) != 1:
            raise ValueError(f"{NOME}: passo {k}, fuori dalla rosa doveva restare "
                             f"solo «tavolo»")

    # passo 1: la correzione non ribalta niente, conferma
    if ordina(conti[0][0])[0] != "piatto":
        raise ValueError(f"{NOME}: al passo 1 «piatto» doveva essere già prima")

    # passo 2: la correzione allarga il margine sulla forchetta
    ap, _, _, cd = conti[1]
    i_col, i_for = PAROLE.index("coltello"), PAROLE.index("forchetta")
    if not cd[i_col] - cd[i_for] > ap[i_col] - ap[i_for]:
        raise ValueError(f"{NOME}: al passo 2 la sottrazione doveva premiare "
                         f"«coltello», non pareggiare")

    # passo 3: il fotogramma che vale la clip
    ap, ch, _, cd = conti[2]
    if ordina(ap)[0] != "forchetta" or ordina(ch)[0] != "forchetta":
        raise ValueError(f"{NOME}: al passo 3 «forchetta» doveva essere prima "
                         f"in tutte e due le letture")
    if ordina(ap)[1] != "bicchiere":
        raise ValueError(f"{NOME}: al passo 3 «bicchiere» doveva essere seconda "
                         f"a occhi aperti")
    if ordina(cd)[0] != "bicchiere" or ordina(cd)[1] != "forchetta":
        raise ValueError(f"{NOME}: al passo 3 la sottrazione doveva schiacciare "
                         f"«forchetta» sotto «bicchiere»")

    if any(p["scrive"] == "forchetta" for p in PASSI):
        raise ValueError(f"{NOME}: la forchetta non deve mai essere scritta")


# ------------------------------------------------------------------ tempo
def visibile(da: int, a: int, ritardo: float = 0.0):
    """Tappe di opacità per un gruppo che vive dallo stato `da` allo stato `a`."""
    inizio = sosta(da, STATI, TENUTA)[0] + ritardo
    fine = sosta(a, STATI, TENUTA)[1]
    tappe = {}

    if da == 0 and ritardo <= 0.01:
        tappe[0.0] = "opacity:1"
    else:
        tappe[0.0] = "opacity:0"
        tappe[sosta(da - 1, STATI, TENUTA)[1] if da else 0.0] = "opacity:0"
        tappe[inizio] = "opacity:1"

    if a >= STATI - 1:
        tappe[100.0] = "opacity:1"
    else:
        tappe[fine] = "opacity:1"
        tappe[sosta(a + 1, STATI, TENUTA)[0]] = "opacity:0"
        tappe[100.0] = "opacity:0"

    return sorted(tappe.items())


def scrittura(k: int):
    """La parola k della frase: compare quando viene scelta, e resta.

    Mentre è l'ultima scritta è in terracotta; al passo dopo passa in nero. Il
    riposo è l'ultima parola, che quindi in terracotta ci resta.
    """
    stato = 2 * k + 1
    inizio, fine = sosta(stato, STATI, TENUTA)
    calda = f"fill-opacity:1;fill:{TERRACOTTA}"
    tappe = {0.0: f"fill-opacity:0;fill:{TERRACOTTA}",
             sosta(stato - 1, STATI, TENUTA)[1]: f"fill-opacity:0;fill:{TERRACOTTA}",
             inizio: calda}
    if stato + 1 < STATI:
        tappe[fine] = calda
        tappe[sosta(stato + 1, STATI, TENUTA)[0]] = f"fill-opacity:1;fill:{INK}"
        tappe[100.0] = f"fill-opacity:1;fill:{INK}"
    else:
        tappe[100.0] = calda
    return sorted(tappe.items())


# ------------------------------------------------------------------ disegno
def num(x: float) -> str:
    return f"{x:.2f}".replace(".", ",")


def y_riga(i: int) -> float:
    return Y_R0 + i * DY_R


def barre(colonna: int, valori, rosa, classe: str, vincitore: int = -1):
    """Una colonna di sei barre, con il numero in fondo a ciascuna."""
    x0 = COL_X[colonna]
    pezzi = []
    for i, v in enumerate(valori):
        cy = y_riga(i)
        if classe == "bc" and not rosa[i]:
            pezzi.append(f'<text class="escl" x="{x0 + 2}" y="{cy + 4}">'
                         f'fuori dalla rosa</text>')
            continue
        w = v / VMAX * COL_W
        cls = classe if rosa[i] else f"{classe} spenta"
        if i == vincitore:
            cls += " vince"
        pezzi.append(f'<rect class="{cls}" x="{x0}" y="{cy - H_BAR / 2:.0f}" '
                     f'width="{w:.1f}" height="{H_BAR}" rx="2"/>')
        pezzi.append(f'<text class="{"num vinto" if i == vincitore else "num"}" '
                     f'x="{x0 + w + 5:.1f}" y="{cy + 4}">{num(v)}</text>')
    return "".join(pezzi)


def costruisci() -> Figura:
    conti = [contrastiva(p) for p in PASSI]
    controlla(conti)

    corpo, anim = [], []

    # --------------------------------------------------------------- la frase
    corpo.append('<text class="lbs" x="30" y="30">la risposta, scritta una '
                 'parola alla volta</text>')
    pezzi = []
    for k, passo in enumerate(PASSI):
        anim.append(keyframes(f"pw{k}", scrittura(k)))
        colore = TERRACOTTA if k == len(PASSI) - 1 else INK
        pezzi.append(f'<tspan fill="{colore}" fill-opacity="1" '
                     f'style="animation:pw{k} var(--d) infinite">'
                     f'{passo["frase"]}</tspan>')
    corpo.append(f'<text class="frase" x="30" y="62">{"".join(pezzi)}</text>')
    corpo.append('<text class="prem" x="30" y="86">nella foto ci sono un piatto, '
                 'un coltello e un bicchiere; nessuna forchetta.</text>')
    corpo.append(f'<line class="axc" x1="30" y1="100" x2="{LARG - 30}" y2="100"/>')

    # ------------------------------------------------------------- le colonne
    testate = (("a occhi aperti", "con la foto", "ta"),
               ("a occhi chiusi", "con la foto illeggibile", "oc"),
               ("la differenza", "quello che viene dalla foto", "te"))
    for c, (titolo, gloss, cls) in enumerate(testate):
        corpo.append(f'<text class="tit {cls}" x="{COL_X[c]}" y="{Y_TIT}">'
                     f'{titolo}</text>')
        corpo.append(f'<text class="glo" x="{COL_X[c]}" y="{Y_SUB}">{gloss}</text>')
        corpo.append(f'<line class="ass" x1="{COL_X[c]}" y1="{y_riga(0) - 16}" '
                     f'x2="{COL_X[c]}" y2="{y_riga(len(PAROLE) - 1) + 16}"/>')

    for i, parola in enumerate(PAROLE):
        corpo.append(f'<text class="wl" x="{X_LAB}" y="{y_riga(i) + 4}" '
                     f'text-anchor="end">{parola}</text>')

    # ------------------------------------------------------- le barre, passo per passo
    for k, (passo, (aperti, chiusi, rosa, cd)) in enumerate(zip(PASSI, conti)):
        letture, sottrazione = 2 * k, 2 * k + 1
        vince = cd.index(max(cd))

        # le due letture: la seconda entra mezza battuta dopo la prima, perché
        # sono due passaggi del modello, non uno.
        anim.append(keyframes(f"la{k}", visibile(letture, sottrazione)))
        anim.append(keyframes(f"lb{k}", visibile(letture, sottrazione, ritardo=5.0)))
        for colonna, (valori, sigla) in enumerate(((aperti, "la"), (chiusi, "lb"))):
            op = 1 if sottrazione == STATI - 1 else 0
            corpo.append(
                f'<g class="grp" opacity="{op}" '
                f'style="animation:{sigla}{k} var(--d) infinite">'
                f'{barre(colonna, valori, rosa, "ba" if colonna == 0 else "bb")}</g>')

        # la sottrazione: la terza colonna, e la punta che indica chi si scrive
        anim.append(keyframes(f"dc{k}", visibile(sottrazione, sottrazione)))
        op = 1 if sottrazione == STATI - 1 else 0
        cy = y_riga(vince)
        punta = (f'<path class="punta" d="M {COL_X[2] - 15} {cy - 5.5} '
                 f'L {COL_X[2] - 5} {cy} L {COL_X[2] - 15} {cy + 5.5} Z"/>')
        corpo.append(
            f'<g class="grp" opacity="{op}" '
            f'style="animation:dc{k} var(--d) infinite">'
            f'{barre(2, cd, rosa, "bc", vincitore=vince)}{punta}</g>')

    # ------------------------------------------------------------ didascalie
    for i in range(STATI):
        passo, battuta = PASSI[i // 2], i % 2
        testo = passo["letture"] if battuta == 0 else passo["scelta"]
        anim.append(keyframes(f"cp{i}", visibile(i, i)))
        op = 1 if i == STATI - 1 else 0
        cls = "cap forte" if battuta else "cap"
        corpo.append(f'<text class="{cls}" x="30" y="{Y_CAP}" opacity="{op}" '
                     f'style="animation:cp{i} var(--d) infinite">'
                     f'<tspan class="capn">passo {i // 2 + 1} di {len(PASSI)} · '
                     f'</tspan>{testo}</text>')

    corpo.append(f'<text class="lbs" x="30" y="{Y_N1}">la differenza è '
                 f'(1 + α)·log p(aperti) − α·log p(chiusi), rinormalizzata sulla '
                 f'rosa; qui α = 1.</text>')
    corpo.append(f'<text class="lbs" x="30" y="{Y_N2}">la rosa sono le parole che '
                 f'a occhi aperti valgono almeno β = 0,1 della prima; il resto '
                 f'non si guarda.</text>')

    return Figura(
        larghezza=LARG, altezza=ALT,
        alt="La decodifica di una didascalia, parola per parola. In alto la "
            "risposta cresce: un piatto, un coltello e un bicchiere. Sotto, per "
            "ogni passo, tre colonne di barre sullo stesso vocabolario di sei "
            "parole: le probabilità con la foto, quelle con la foto resa "
            "illeggibile e quelle che restano dopo aver sottratto le seconde "
            "dalle prime. Ai primi due passi la sottrazione conferma o premia la "
            "parola che si vede davvero. Al terzo passo forchetta è la più alta "
            "in tutte e due le letture, quindi non viene dalla foto ma "
            "dall'abitudine: sottraendo, sprofonda sotto bicchiere, che era "
            "seconda, e la parola scritta è bicchiere.",
        corpo="".join(corpo),
        stile=f"""    .frase {{ font-family:{SERIF}; font-size:22px; fill:{INK}; }}
    .prem {{ font-family:{SANS}; font-size:12px; font-style:italic; fill:{FG_MUTED}; }}
    .wl   {{ font-family:{SANS}; font-size:13.5px; fill:{INK}; }}
    .tit  {{ font-family:{SANS}; font-size:14px; font-weight:700; }}
    .ta   {{ fill:{TEAL}; }}
    .oc   {{ fill:{OCRA}; }}
    .te   {{ fill:{TERRACOTTA}; }}
    .glo  {{ font-family:{SANS}; font-size:11px; fill:{FG_MUTED}; }}
    .ass  {{ stroke:{BORDER_STRONG}; stroke-width:1.5; }}
    .ba   {{ fill:{TEAL}; fill-opacity:0.68; stroke:{TEAL}; stroke-width:1; }}
    .bb   {{ fill:{OCRA}; fill-opacity:0.72; stroke:{OCRA}; stroke-width:1; }}
    .bc   {{ fill:{TERRACOTTA}; fill-opacity:0.55; stroke:{TERRACOTTA}; stroke-width:1; }}
    .bc.vince {{ fill-opacity:0.9; stroke-width:2; }}
    .spenta {{ fill-opacity:0.2; stroke-dasharray:3 3; }}
    .num  {{ font-family:{SANS}; font-size:11px; fill:{FG_MUTED}; }}
    .vinto {{ fill:{TERRACOTTA}; font-weight:700; }}
    .escl {{ font-family:{SANS}; font-size:11px; font-style:italic; fill:{FG_MUTED}; }}
    .punta {{ fill:{TERRACOTTA}; }}
    .cap  {{ font-family:{SANS}; font-size:14.5px; fill:{INK}; }}
    .forte {{ font-weight:700; fill:{TERRACOTTA}; }}
    .capn {{ font-weight:400; fill:{FG_MUTED}; }}""",
        animazioni=anim,
        durata=STATI * 1.8,
        fermi=".grp, .cap, .frase tspan",
    )
