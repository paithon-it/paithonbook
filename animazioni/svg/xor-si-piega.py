"""Lo XOR si piega: il primo strato cambia le coordinate, e una retta basta.

Il seguito di `xor-non-separabile`, che mostra la retta girare e sbagliare
sempre. Qui la risposta: a sinistra il piano degli ingressi con **due** rette
invece di una (i due neuroni nascosti), a destra gli stessi quattro punti che
si spostano nelle coordinate calcolate da quei neuroni, e in quelle coordinate
la retta del neurone di uscita, una sola, che li separa.

Il movimento è il contenuto: «un primo strato costruisce rappresentazioni
intermedie, un secondo le combina» è la frase che il capitolo scrive, e ferma
si può soltanto raccontare.

I pesi non sono scritti a mano.

* Il **primo strato** è la costruzione classica: un neurone che si accende se
  almeno un ingresso vale 1 (la retta x₁ + x₂ = 0,5) e uno che si accende se
  non valgono 1 tutti e due (la retta x₁ + x₂ = 1,5). Fra le due rette resta
  la fascia in cui cadono i due casi con uscita 1: sono le due righe che il
  lettore, col foglio a quadretti in mano, chiede di vedere tirare.
* Il **neurone di uscita** si addestra invece per discesa del gradiente
  (`uscita_addestrata`), perché il punto è proprio che nelle coordinate nuove
  il problema è diventato risolvibile: se non lo fosse, non si troverebbe
  nessuna retta e la figura non nascerebbe.
* `rete()` esegue la rete sui quattro ingressi e verifica con un `assert` che
  le uscite siano 0, 1, 1, 0. Le coordinate disegnate a destra sono quelle che
  escono davvero dai due neuroni nascosti, e anche i numeri **arrotondati**
  che finiscono nella formula stampata vengono riprovati: una figura non può
  stampare una formula che classifica diversamente da quella che disegna.
"""

import math

from paithon_svg import *

NOME = "xor-si-piega"
TITOLO = "due strati piegano lo XOR"

# Gli stessi quattro casi della figura sorella, nello stesso ordine e con la
# stessa convenzione: terracotta la classe di uscita 1, teal quella di uscita 0.
DATI = [((0, 1), 1), ((1, 0), 1),      # XOR = 1: antidiagonale
        ((0, 0), 0), ((1, 1), 0)]      # XOR = 0: diagonale

# Il primo strato: (pesi, bias) dei due neuroni nascosti, con la ReLU. Le due
# rette che ne escono sono parallele, ed è quello che rende la fascia in mezzo
# una fascia: il disegno più sotto ci conta.
STRATO_1 = [((1.0, 1.0), -0.5),        # h₁: «almeno uno acceso»
            ((-1.0, -1.0), 1.5)]       # h₂: «non tutti e due accesi»

PASSI, ETA = 4000, 0.5                 # la discesa del gradiente dell'uscita

# La tabella di verità dello XOR, nell'ordine in cui la si legge.
TAVOLA = [(0, 0), (0, 1), (1, 0), (1, 1)]
XOR = [0, 1, 1, 0]

# Le quattro fasi, con l'istante in cui ciascuna comincia (in % del ciclo).
FASI = [
    (0.0, "i quattro casi dello XOR: le due classi occupano gli angoli opposti"),
    (22.0, "due neuroni nascosti, due rette: in mezzo restano i casi con uscita 1"),
    (42.0, "il primo strato dà a ogni punto due coordinate nuove"),
    (72.0, "nelle coordinate nuove una retta sola basta, e i due casi con "
           "uscita 1 coincidono"),
]
# Quando i punti viaggiano. A cavallo di metà ciclo apposta: il fermo immagine
# di mezzo (`animazioni/fermi.py` scatta al 50%) li prende in volo.
MOSSA = (44.0, 58.0)


# --------------------------------------------------------------------------
# La rete: si costruisce, si esegue, si verifica
# --------------------------------------------------------------------------
def relu(z: float) -> float:
    return z if z > 0.0 else 0.0


def nascosto(x) -> tuple[float, float]:
    """Le due coordinate nuove di un ingresso: h = ReLU(W x + b)."""
    return tuple(relu(w[0] * x[0] + w[1] * x[1] + b) for w, b in STRATO_1)


def uscita_addestrata(h_dati, bersagli):
    """Un neurone sigmoide sulle coordinate nuove, per discesa del gradiente.

    Parte da zero (nessun peso scelto a mano) e minimizza l'entropia
    incrociata. Alla fine si divide tutto per ‖w‖: la retta non cambia (è la
    stessa disuguaglianza moltiplicata per una costante positiva) e i numeri
    stampati smettono di dipendere da quanto a lungo si è addestrato.
    """
    w, b = [0.0, 0.0], 0.0
    n = len(h_dati)
    for _ in range(PASSI):
        gw, gb = [0.0, 0.0], 0.0
        for (h1, h2), y in zip(h_dati, bersagli):
            errore = 1.0 / (1.0 + math.exp(-(w[0] * h1 + w[1] * h2 + b))) - y
            gw[0] += errore * h1
            gw[1] += errore * h2
            gb += errore
        w[0] -= ETA * gw[0] / n
        w[1] -= ETA * gw[1] / n
        b -= ETA * gb / n
    norma = math.hypot(*w) or 1e-9
    return (w[0] / norma, w[1] / norma), b / norma


def classifica(w, b, h) -> int:
    """Il gradino del capitolo: 1 se w·h + b ≥ 0, altrimenti 0."""
    return 1 if w[0] * h[0] + w[1] * h[1] + b >= 0 else 0


def rete():
    """Costruisce la rete, la esegue sui quattro ingressi, verifica lo XOR."""
    h_dati = [nascosto(x) for x, _ in DATI]
    w, b = uscita_addestrata(h_dati, [y for _, y in DATI])

    uscite = [classifica(w, b, nascosto(x)) for x in TAVOLA]
    if uscite != XOR:
        raise AssertionError(f"la rete calcola {uscite}, lo XOR è {XOR}: "
                             "senza XOR risolto questa figura non ha senso")

    # I due casi con uscita 1 devono finire nello stesso punto: la didascalia
    # lo dice, e allora dev'essere vero anche nei numeri.
    if nascosto((0, 1)) != nascosto((1, 0)):
        raise AssertionError("i due casi con uscita 1 non si sovrappongono più: "
                             "la didascalia mentirebbe")

    # E la formula stampata, che è arrotondata, deve classificare come quella
    # disegnata: una figura non può contraddire la propria legenda.
    w_st, b_st = (round(w[0], 2), round(w[1], 2)), round(b, 2)
    if [classifica(w_st, b_st, nascosto(x)) for x in TAVOLA] != XOR:
        raise AssertionError("i pesi arrotondati per la stampa non fanno lo XOR")

    return h_dati, w, b, uscite


# --------------------------------------------------------------------------
# Numeri e formule, ricavati dai pesi veri e mai scritti a mano
# --------------------------------------------------------------------------
def _num(v: float, cifre: int = 2) -> str:
    s = f"{abs(v):.{cifre}f}".rstrip("0").rstrip(".")
    return s.replace(".", ",")


def _lineare(w, b, nomi) -> str:
    """«x₁ + x₂ − 0,5» dai pesi, così la formula non può divergere dal disegno."""
    pezzi = []
    for i, (c, nome) in enumerate(zip(w, nomi)):
        corpo = nome if abs(abs(c) - 1) < 1e-9 else f"{_num(c)}·{nome}"
        pezzi.append(("−" + corpo if c < 0 else corpo) if i == 0
                     else ("− " if c < 0 else "+ ") + corpo)
    if abs(b) > 5e-3:
        pezzi.append(("− " if b < 0 else "+ ") + _num(b))
    return " ".join(pezzi)


# --------------------------------------------------------------------------
# Il disegno
# --------------------------------------------------------------------------
def tacche(r: Riquadro) -> list[str]:
    """Le tacche 0 e 1 sui due assi, etichette all'interno come nella sorella."""
    segni = []
    giu, sin = r.y + r.alt, r.x
    for v in (0, 1):
        segni += [
            f'<line class="ax" x1="{r.sx(v):.1f}" y1="{giu - 6}" '
            f'x2="{r.sx(v):.1f}" y2="{giu}"/>',
            f'<text class="lbs" x="{r.sx(v):.1f}" y="{giu - 14}" '
            f'text-anchor="middle">{v}</text>',
            f'<line class="ax" x1="{sin}" y1="{r.sy(v):.1f}" '
            f'x2="{sin + 6}" y2="{r.sy(v):.1f}"/>',
            f'<text class="lbs" x="{sin + 12}" y="{r.sy(v) + 5:.1f}">{v}</text>',
        ]
    return segni


def retta(cls: str, posa, extra: str = "") -> str:
    """Un segmento lunghissimo, posato e ruotato: si affida al clip del campo."""
    px, py, ang = posa
    return (f'<line class="{cls}" x1="-620" y1="0" x2="620" y2="0" '
            f'transform="translate({px:.1f},{py:.1f}) rotate({ang:.1f})"{extra}/>')


def apparizione(nome: str, quando: float, valore: str = "opacity:1") -> str:
    """@keyframes di una cosa che compare a `quando` e poi resta."""
    return keyframes(nome, [(0.0, "opacity:0"), (quando - 2.0, "opacity:0"),
                            (quando, valore), (100.0, valore)])


def costruisci() -> Figura:
    h_dati, w_out, b_out, uscite = rete()

    # Stesso riquadro per i due pannelli: la scala non cambia, quindi lo
    # spostamento dei punti si legge per quello che è. Gli estremi non sono
    # simmetrici attorno a 0,75 apposta: se lo fossero, la retta x₁ + x₂ = 1,5
    # passerebbe esattamente per due angoli del riquadro.
    limiti = dict(xmin=-0.4, xmax=1.8, ymin=-0.4, ymax=1.8)
    sin = Riquadro(x=56, y=84, larg=300, alt=300, **limiti)
    des = Riquadro(x=424, y=84, larg=300, alt=300, **limiti)

    corpo, anim = [], []

    # ---------------------------------------------------------------- sinistra
    # Il piano di partenza. Al riposo resta acceso ma in tono più tenue
    # (l'attributo `opacity` del gruppo), perché l'azione si è spostata a destra.
    a_sin = [sin.clip("campoS"), sin.cornice(croce=True), *tacche(sin)]

    pose = [sin.posa_retta(w, b) for w, b in STRATO_1]
    # La fascia fra le due rette: è lì che cadono i due casi con uscita 1.
    # `posa_retta` dà il piede della perpendicolare dal centro del riquadro, e
    # per due rette parallele i due piedi stanno sulla stessa perpendicolare:
    # la loro distanza *è* lo spessore della fascia, e il loro punto di mezzo
    # ne è l'asse. Nessun numero da riscrivere se le soglie cambiano.
    (p1x, p1y, ang), (p2x, p2y, _) = pose
    spessore = math.hypot(p2x - p1x, p2y - p1y)
    a_sin.append(f'<g clip-path="url(#campoS)" class="mobile nasc">'
                 + retta("banda", ((p1x + p2x) / 2, (p1y + p2y) / 2, ang),
                         f' stroke-width="{spessore:.1f}"')
                 + "".join(retta("nas", p) for p in pose)
                 + "</g>")

    # I nomi delle due rette, ciascuno accostato alla propria: la retta di h₁ è
    # quella vicina all'origine, in basso a sinistra; quella di h₂ è la lontana.
    # (Sbagliare qui non si vede rileggendo il codice, si vede solo guardando il
    # disegno: la prima stesura appendeva «h₁» alla retta di h₂.)
    a_sin += [
        f'<text class="nom mobile nasc" x="{sin.x + 14}" y="{sin.y + 172}">h₁</text>',
        f'<text class="nom mobile nasc" x="{sin.x + sin.larg - 36}" '
        f'y="{sin.y + sin.alt - 22}" text-anchor="end">h₂</text>',
    ]

    for (x, y), xor in DATI:
        a_sin.append(f'<circle class="pt {"pos" if xor else "neg"}" '
                     f'cx="{sin.sx(x):.1f}" cy="{sin.sy(y):.1f}" r="9"/>')
    a_sin.append(f'<text class="lbl" x="{sin.x}" y="{sin.y - 16}">'
                 f'il piano degli ingressi (x₁, x₂)</text>')

    corpo.append(f'<g id="partenza" opacity="0.8">{"".join(a_sin)}</g>')
    anim += [
        keyframes("attenua", [(0.0, "opacity:1"), (FASI[3][0] - 4, "opacity:1"),
                              (FASI[3][0], "opacity:0.8"), (100.0, "opacity:0.8")]),
        apparizione("nasc", FASI[1][0]),
    ]

    # ----------------------------------------------------------------- destra
    corpo += [des.clip("campoD"), des.cornice(croce=True), *tacche(des)]

    # Le tracce: da dove ogni punto è partito, a dove è arrivato. Nel fermo
    # immagine sono la sola cosa che dice che c'è stato un movimento.
    for ((x, y), _), (h1, h2) in zip(DATI, h_dati):
        x0, y0, x1, y1 = des.sx(x), des.sy(y), des.sx(h1), des.sy(h2)
        lung = math.hypot(x1 - x0, y1 - y0) or 1.0
        ux, uy = (x1 - x0) / lung, (y1 - y0) / lung
        corpo += [
            f'<line class="scia mobile via" x1="{x0 + ux * 12:.1f}" '
            f'y1="{y0 + uy * 12:.1f}" x2="{x1 - ux * 12:.1f}" y2="{y1 - uy * 12:.1f}"/>',
            f'<circle class="ombra mobile via" cx="{x0:.1f}" cy="{y0:.1f}" r="9"/>',
        ]
    anim.append(apparizione("via", MOSSA[0]))

    # La retta del neurone di uscita, che compare quando i punti sono arrivati,
    # e la metà di piano in cui risponde 1. Da che parte stia non si scrive: si
    # chiede alla posa disegnata, con lo stesso conto di `separa`, cioè la
    # normale ricostruita dalla rotazione che finisce nell'SVG.
    posa_out = des.posa_retta(w_out, b_out)
    if not des.separa(posa_out, list(zip(h_dati, [y for _, y in DATI]))):
        raise AssertionError("la retta disegnata non separa le classi nelle "
                             "coordinate nuove: c'è un segno sbagliato")
    px, py, ang = posa_out
    a = math.radians(ang)
    uno = h_dati[0]                     # un caso con uscita 1, per orientarsi
    sopra = ((des.sx(uno[0]) - px) * -math.sin(a)
             + (des.sy(uno[1]) - py) * math.cos(a)) > 0
    corpo.append(
        f'<g clip-path="url(#campoD)">'
        f'<rect class="zona mobile esce" x="-620" y="{0 if sopra else -620}" '
        f'width="1240" height="620" '
        f'transform="translate({px:.1f},{py:.1f}) rotate({ang:.1f})"/>'
        + retta("sep mobile esce", posa_out) + "</g>")
    # Che cosa risponde la rete di qua e di là: la tinta da sola dice che c'è
    # una frontiera, non che cosa c'è dalle due parti.
    corpo += [
        f'<text class="lbs mobile esce" x="{des.x + 116}" '
        f'y="{des.y + 236}">ŷ = 1</text>',
        f'<text class="lbs mobile esce" x="{des.x + des.larg - 14}" y="{des.y + 46}" '
        f'text-anchor="end">ŷ = 0</text>',
    ]
    anim.append(apparizione("esce", FASI[3][0]))

    # L'anello sui due punti che finiscono sovrapposti: senza, al riposo si
    # vede un punto solo e la sparizione dell'altro sembra un errore.
    corpo.append(f'<circle class="anello mobile giunti" '
                 f'cx="{des.sx(h_dati[0][0]):.1f}" cy="{des.sy(h_dati[0][1]):.1f}" '
                 f'r="16"/>')
    anim.append(apparizione("giunti", MOSSA[1]))

    # I quattro punti: al riposo stanno nelle coordinate nuove, senza alcun
    # `transform`. L'animazione parte dallo scarto inverso, cioè da dov'erano.
    for i, (((x, y), xor), (h1, h2)) in enumerate(zip(DATI, h_dati)):
        dx, dy = des.sx(x) - des.sx(h1), des.sy(y) - des.sy(h2)
        prima = f"transform:translate({dx:.1f}px,{dy:.1f}px)"
        anim.append(keyframes(f"vola{i}", [
            (0.0, prima), (MOSSA[0], prima),
            (MOSSA[1], "transform:translate(0px,0px)"),
            (100.0, "transform:translate(0px,0px)")]))
        corpo.append(f'<circle class="pt mobile {"pos" if xor else "neg"} v{i}" '
                     f'cx="{des.sx(h1):.1f}" cy="{des.sy(h2):.1f}" r="9"/>')

    corpo.append(f'<text class="lbl" x="{des.x}" y="{des.y - 16}">'
                 f'le rappresentazioni intermedie (h₁, h₂)</text>')

    # ------------------------------------------------------------- didascalie
    for i, (quando, testo) in enumerate(FASI):
        fine = FASI[i + 1][0] if i + 1 < len(FASI) else None
        tappe = [(0.0, "opacity:1" if i == 0 else "opacity:0")]
        if i:
            tappe += [(quando - 2.0, "opacity:0"), (quando, "opacity:1")]
        if fine is None:
            tappe.append((100.0, "opacity:1"))
        else:
            tappe += [(fine - 2.0, "opacity:1"), (fine, "opacity:0"),
                      (100.0, "opacity:0")]
        anim.append(keyframes(f"dida{i}", tappe))
        resta = ";opacity:1" if fine is None else ""
        corpo.append(f'<text class="eti mobile" x="{sin.x}" y="{sin.y + sin.alt + 34}" '
                     f'style="animation:dida{i} var(--d) infinite{resta}">{testo}</text>')

    corpo.append(f'<text class="cnt mobile esce" x="{sin.x}" y="{sin.y + sin.alt + 64}">'
                 f'nessun punto sbagliato: la rete calcola '
                 f'{", ".join(str(u) for u in uscite)}, cioè lo XOR</text>')

    # Una formula per pannello, incolonnata sotto il suo: due `<text>` e non uno,
    # perché l'SVG collassa gli spazi e la spaziatura larga sparirebbe.
    corpo += [
        f'<text class="lbs" x="{r.x}" y="{sin.y + sin.alt + 94}">'
        f'h{p} = ReLU({_lineare(w, b, ("x₁", "x₂"))})</text>'
        for r, p, (w, b) in zip((sin, des), "₁₂", STRATO_1)
    ]
    corpo += [
        f'<text class="lbs" x="{sin.x}" y="{sin.y + sin.alt + 114}">'
        f'ŷ = g({_lineare(w_out, b_out, ("h₁", "h₂"))}), con g il gradino e i '
        f'pesi dell\'uscita trovati addestrando</text>',
    ]

    # ---------------------------------------------------------------- legende
    corpo += [
        f'<circle class="pt pos" cx="{sin.x + 6}" cy="30" r="9"/>',
        f'<text class="lbs" x="{sin.x + 24}" y="35">XOR = 1</text>',
        f'<circle class="pt neg" cx="{sin.x + 118}" cy="30" r="9"/>',
        f'<text class="lbs" x="{sin.x + 136}" y="35">XOR = 0</text>',
        f'<line class="nas" x1="{des.x - 48}" y1="30" x2="{des.x - 22}" y2="30"/>',
        f'<text class="lbs" x="{des.x - 14}" y="35">neuroni nascosti</text>',
        f'<line class="sep" x1="{des.x + 138}" y1="30" x2="{des.x + 164}" y2="30"/>',
        f'<text class="lbs" x="{des.x + 172}" y="35">neurone di uscita</text>',
    ]

    return Figura(
        larghezza=756, altezza=524,
        alt="Due pannelli affiancati. A sinistra i quattro casi dello XOR agli "
            "angoli del quadrato unitario, in terracotta i due con uscita 1 e in "
            "teal i due con uscita 0, tagliati da due rette parallele, i due "
            "neuroni nascosti, che lasciano in mezzo una fascia con i soli punti "
            "terracotta. A destra gli stessi quattro punti si spostano nelle "
            "coordinate calcolate da quei neuroni: i due terracotta finiscono "
            "esattamente nello stesso posto e i due teal si allontanano ai lati "
            "opposti, e a quel punto una sola retta li separa.",
        corpo="".join(corpo),
        stile=f"""    .pt  {{ stroke:{CREAM}; stroke-width:1.5; }}
    .pos {{ fill:{TERRACOTTA}; }}
    .neg {{ fill:{TEAL}; }}
    .sep {{ stroke:{INK}; stroke-width:3; fill:none; }}
    .nas {{ stroke:{OCRA}; stroke-width:2.5; fill:none; }}
    .banda {{ stroke:{OCRA}; stroke-opacity:0.16; fill:none; }}
    .zona {{ fill:{TERRACOTTA}; fill-opacity:0.09; }}
    .scia {{ stroke:{BORDER_STRONG}; stroke-width:1.6; stroke-dasharray:5 5; }}
    .ombra {{ fill:none; stroke:{BORDER_STRONG}; stroke-width:1.6; }}
    .anello {{ fill:none; stroke:{OCRA}; stroke-width:2; }}
    .nom {{ font-family:{SANS}; font-size:15px; font-weight:700; fill:{OCRA}; }}
    .eti {{ font-family:{SANS}; font-size:15px; fill:{FG_MUTED}; opacity:0; }}
    .cnt {{ font-family:{SANS}; font-size:16px; font-weight:700; fill:{TERRACOTTA}; }}
    #partenza {{ animation:attenua var(--d) infinite; }}
    .nasc   {{ animation:nasc var(--d) infinite; }}
    .via    {{ animation:via var(--d) infinite; }}
    .esce   {{ animation:esce var(--d) infinite; }}
    .giunti {{ animation:giunti var(--d) infinite; }}
""" + "".join(f"    .v{i} {{ animation:vola{i} var(--d) infinite; "
              f"transform-box:view-box; }}\n"
              for i in range(len(DATI))).rstrip("\n"),
        animazioni=anim,
        durata=9.0,
        fermi="#partenza, .mobile",
    )
