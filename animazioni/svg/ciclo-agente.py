"""Il ciclo dell'agente: quello che gira è sempre uguale, il contesto no.

Una figura ferma di un agente è un cerchio con tre frecce, e nasconde proprio
la cosa che distingue un agente da una singola chiamata al modello: a ogni giro
il modello **rilegge più di prima**, perché ogni osservazione resta nel
contesto. Qui il cerchio gira a sinistra e la colonna di destra si allunga di
un blocco per giro, che è la chiamata con ciò che ha risposto.

I passi non sono scritti a mano: questo file esegue **l'agente giocattolo del
capitolo** (lo stesso `llm_finto`, la stessa calcolatrice senza `eval`, lo
stesso archivio) e disegna la traccia che ne esce. `controlla()` verifica che
sia quella stampata in `Agenti/agenti-e-tool-use.md`: le osservazioni 2017 e 9,
tre giri, e la risposta parola per parola. Se un giorno il capitolo cambiasse
l'esempio, la figura non si genererebbe più invece di smentirlo in silenzio.

Il disegno fermo è l'ultimo stato, ed è la parte che la divulgazione dimentica:
il ciclo **finisce**. Al terzo giro il modello non chiama nessuno strumento, e
il ramo di uscita porta alla risposta; il limite (`max_passi = 5`) resta lì a
vista, perché è l'altro modo in cui un agente si ferma.
"""

import ast
import operator

from paithon_svg import *

NOME = "ciclo-agente"
TITOLO = "il ciclo dell'agente, e il contesto che si allunga"

DOMANDA = ("In che anno è uscito 'Attention Is All You Need' "
           "e quanti anni fa è, nel 2026?")
MAX_PASSI = 5                     # il `max_passi` del capitolo: il guardrail

# --------------------------------------------------------------------------
# Gli strumenti veri del capitolo (ricopiati, non riassunti: se divergono, la
# guardia sotto se ne accorge)
# --------------------------------------------------------------------------
_OP = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.USub: operator.neg,
}


def _operatore(op):
    if type(op) not in _OP:
        raise ValueError(f"operatore non ammesso: {type(op).__name__}")
    return _OP[type(op)]


def _valuta(nodo):
    if isinstance(nodo, ast.Constant):
        if not isinstance(nodo.value, (int, float)):
            raise ValueError("ammessi solo numeri")
        return nodo.value
    if isinstance(nodo, ast.BinOp):
        return _operatore(nodo.op)(_valuta(nodo.left), _valuta(nodo.right))
    if isinstance(nodo, ast.UnaryOp):
        return _operatore(nodo.op)(_valuta(nodo.operand))
    raise ValueError("espressione non ammessa")


def calcola(espressione):
    return _valuta(ast.parse(espressione, mode="eval").body)


ARCHIVIO = {
    "attention is all you need": "2017",
    "gpt-3": "2020",
    "react": "2022",
}


def cerca(chiave):
    return ARCHIVIO.get(chiave.lower().strip(), "non trovato")


STRUMENTI = {"calcola": calcola, "cerca": cerca}


def llm_finto(traccia):
    ultima = traccia[-1]["osservazione"] if traccia else None
    if ultima is None:
        return ("Non conosco a memoria l'anno del paper: lo cerco.",
                "cerca", "attention is all you need")
    if ultima == "2017":
        return ("Il paper è del 2017. Calcolo quanti anni fa, dal 2026.",
                "calcola", "2026 - 2017")
    return (f"Il calcolo dice {ultima}: ho tutto per rispondere.",
            "Answer", "'Attention Is All You Need' è del 2017: 9 anni fa nel 2026.")


def esegui_agente(domanda, max_passi=MAX_PASSI):
    """Il loop del capitolo, che invece di stampare restituisce i suoi passi."""
    traccia, passi = [], []
    for _ in range(max_passi):
        pensiero, azione, argomento = llm_finto(traccia)
        if azione == "Answer":
            passi.append({"pensiero": pensiero, "azione": azione,
                          "argomento": argomento, "osservazione": None})
            return passi, "risposta"
        osservazione = str(STRUMENTI[azione](argomento))
        passi.append({"pensiero": pensiero, "azione": azione,
                      "argomento": argomento, "osservazione": osservazione})
        traccia.append({"azione": azione, "argomento": argomento,
                        "osservazione": osservazione})
    return passi, "limite"


def controlla(passi, esito):
    """La traccia disegnata è quella stampata nel capitolo? Altrimenti niente figura."""
    if esito != "risposta":
        raise ValueError(f"{NOME}: l'agente non ha chiuso il compito ({esito})")
    osservazioni = [p["osservazione"] for p in passi if p["osservazione"] is not None]
    if osservazioni != ["2017", "9"]:
        raise ValueError(f"{NOME}: osservazioni {osservazioni}, il capitolo "
                         f"stampa ['2017', '9']")
    if len(passi) != 3:
        raise ValueError(f"{NOME}: {len(passi)} giri, il capitolo ne stampa 3")
    atteso = "'Attention Is All You Need' è del 2017: 9 anni fa nel 2026."
    if passi[-1]["argomento"] != atteso:
        raise ValueError(f"{NOME}: la risposta non è quella del capitolo")
    if calcola("2026 - 2017") != 9:
        raise ValueError(f"{NOME}: la calcolatrice del capitolo non torna")


# --------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------
BOX_W, BOX_H = 120, 44
OSS = (125, 118)          # centri delle tre caselle del ciclo
AGI = (305, 118)
PEN = (215, 248)

Y_USC = 308               # la casella della risposta, sotto il ciclo
H_USC = 66

XB, WB = 420, 320         # la colonna del contesto
Y_BLK = [70, 149, 228]
H_BLK = 64
X_BAR = 402

CAR_BLK = 42              # caratteri per riga dentro un blocco
CAR_USC = 40              # e dentro la casella della risposta


def spezza(testo: str, massimo: int) -> list[str]:
    """A capo sulle parole, senza tagliarne nessuna."""
    righe, riga = [], ""
    for parola in testo.split():
        prova = f"{riga} {parola}".strip()
        if len(prova) <= massimo or not riga:
            riga = prova
        else:
            righe.append(riga)
            riga = parola
    if riga:
        righe.append(riga)
    if len(righe) > 2 or any(len(r) > massimo for r in righe):
        raise ValueError(f"{NOME}: non ci sta in due righe da {massimo}: {righe}")
    return righe


def acceso_su(stati: set[int], n: int) -> list[tuple[float, str]]:
    """Opacità 1 nelle fette elencate, 0 nelle altre, con un cambio rapido.

    Il 100% ripete il valore dell'ultimo stato, che è quello di riposo: così
    l'animazione finisce esattamente dov'è scritto l'SVG.
    """
    passo = 100.0 / n
    q = {}
    for k in range(n):
        d = f"opacity:{1 if k in stati else 0}"
        q[round(k * passo, 3)] = d
        q[round(min((k + 1) * passo - 1.2, 99.9), 3)] = d
    q[100.0] = f"opacity:{1 if (n - 1) in stati else 0}"
    return sorted(q.items())


def costruisci() -> Figura:
    passi, esito = esegui_agente(DOMANDA)
    controlla(passi, esito)

    # gli stati dell'animazione: pensa/agisce/osserva per ogni giro, e il
    # terzo pensa che invece di agire risponde
    stati, blocchi_visti = [], []
    n_blocchi = 1                            # la domanda c'è dall'inizio
    for p in passi:
        stati.append("pensa")
        blocchi_visti.append(n_blocchi)
        if p["azione"] != "Answer":
            stati.append("agisce")
            blocchi_visti.append(n_blocchi)
            stati.append("osserva")
            n_blocchi += 1                   # l'osservazione entra nel contesto
            blocchi_visti.append(n_blocchi)
    n = len(stati)
    giro_di = []                             # a che giro appartiene ogni stato
    g = 0
    for s in stati:
        if s == "pensa":
            g += 1
        giro_di.append(g)

    corpo, anim = [], []

    # ------------------------------------------------------------------ defs
    corpo.append(
        '<defs>'
        f'<marker id="cf-n" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
        f'markerHeight="6" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{BORDER_STRONG}"/></marker>'
        f'<marker id="cf-t" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
        f'markerHeight="6" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{TERRACOTTA}"/></marker>'
        f'<marker id="cf-v" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
        f'markerHeight="6" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{TEAL}"/></marker>'
        '</defs>')

    # ------------------------------------------------------------- intestazioni
    corpo += [
        '<text class="ttl" x="30" y="40">il ciclo</text>',
        '<text class="lbs" x="30" y="58">pensa, agisce, osserva, e si ricomincia</text>',
        f'<text class="ttl" x="{XB}" y="40">il contesto</text>',
        f'<text class="lbs" x="{XB}" y="58">si allunga di un blocco a ogni giro</text>',
    ]

    # --------------------------------------------------------- le tre caselle
    caselle = [("osserva", "Observation", OSS, {"osserva"}),
               ("agisce", "Action", AGI, {"agisce"}),
               ("pensa", "Thought", PEN, {"pensa"})]
    for nome, inglese, (cx, cy), quali in caselle:
        x, y = cx - BOX_W / 2, cy - BOX_H / 2
        attivi = {i for i, s in enumerate(stati) if s in quali}
        anim.append(keyframes(f"hl{nome}", acceso_su(attivi, n)))
        op = 1 if (n - 1) in attivi else 0
        corpo.append(f'<rect class="box" x="{x:.0f}" y="{y:.0f}" '
                     f'width="{BOX_W}" height="{BOX_H}" rx="9"/>')
        corpo.append(f'<rect class="hl" x="{x:.0f}" y="{y:.0f}" '
                     f'width="{BOX_W}" height="{BOX_H}" rx="9" opacity="{op}" '
                     f'style="animation:hl{nome} var(--d) infinite"/>')
        corpo.append(f'<text class="nome" x="{cx}" y="{cy + 1}" '
                     f'text-anchor="middle">{nome}</text>')
        corpo.append(f'<text class="eng" x="{cx}" y="{cy + 16}" '
                     f'text-anchor="middle">{inglese}</text>')

    # ------------------------------------------------------------- le frecce
    # (x1, y1, x2, y2, in quali stati si accende)
    frecce = [
        (258, 224, 293, 144, {"agisce"}),      # pensa  -> agisce
        (241, 118, 191, 118, {"osserva"}),     # agisce -> osserva
        (135, 142, 168, 224, None),            # osserva -> pensa: il ritorno
    ]
    for k, (x1, y1, x2, y2, quali) in enumerate(frecce):
        if quali is None:                      # il ritorno si accende sul pensa
            attivi = {i for i, s in enumerate(stati)
                      if s == "pensa" and i > 0 and stati[i - 1] == "osserva"}
        else:
            attivi = {i for i, s in enumerate(stati) if s in quali}
        anim.append(keyframes(f"fr{k}", acceso_su(attivi, n)))
        op = 1 if (n - 1) in attivi else 0
        corpo.append(f'<line class="fre" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                     f'marker-end="url(#cf-n)"/>')
        # la freccia accesa sta dentro un <g>: l'opacità su una <line> non
        # ferma il suo marker in ogni rasterizzatore (cairosvg lo disegna lo
        # stesso), e a riposo si vedrebbero punte accese di frecce spente
        corpo.append(f'<g class="fre2" opacity="{op}" '
                     f'style="animation:fr{k} var(--d) infinite">'
                     f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                     f'marker-end="url(#cf-t)"/></g>')

    corpo += [
        '<text class="mini" x="215" y="88" text-anchor="middle">il sistema esegue</text>',
        '<text class="mini" x="305" y="177">chiama uno</text>',
        '<text class="mini" x="305" y="191">strumento</text>',
        '<text class="mini" x="124" y="177" text-anchor="end">l\'osservazione</text>',
        '<text class="mini" x="124" y="191" text-anchor="end">entra nel contesto</text>',
    ]

    # il contagiri, al centro del ciclo: è il guardrail, visto da dentro
    for giro in sorted(set(giro_di)):
        attivi = {j for j in range(n) if giro_di[j] == giro}
        anim.append(keyframes(f"gi{giro}", acceso_su(attivi, n)))
        op = 1 if (n - 1) in attivi else 0
        corpo.append(f'<text class="giro" x="215" y="180" text-anchor="middle" '
                     f'opacity="{op}" style="animation:gi{giro} var(--d) infinite">'
                     f'giro {giro} di {MAX_PASSI}</text>')

    # ------------------------------------------------------- l'uscita dal ciclo
    anim.append(keyframes("usc", acceso_su({n - 1}, n)))
    righe_ans = spezza(passi[-1]["argomento"], CAR_USC)
    pezzi = [f'<line class="frv" x1="215" y1="272" x2="215" y2="{Y_USC - 2}" '
             f'marker-end="url(#cf-v)"/>',
             '<text class="miniv" x="230" y="291">ho tutto: rispondo</text>',
             f'<rect class="usb" x="45" y="{Y_USC}" width="300" height="{H_USC}" rx="9"/>',
             f'<text class="anst" x="59" y="{Y_USC + 22}">Answer</text>']
    for j, riga in enumerate(righe_ans):
        pezzi.append(f'<text class="riga" x="59" y="{Y_USC + 42 + j * 17}">{riga}</text>')
    corpo.append(f'<g class="fine" style="animation:usc var(--d) infinite">'
                 f'{"".join(pezzi)}</g>')

    # --------------------------------------------------- la colonna del contesto
    # un blocco per giro: la chiamata e ciò che ha risposto
    blocchi = [("Domanda", spezza(DOMANDA, CAR_BLK), None)]
    for i, p in enumerate(passi):
        if p["osservazione"] is None:
            continue
        blocchi.append((f"giro {i + 1}",
                        [f'Action: {p["azione"]}[{p["argomento"]}]'],
                        f'Observation: {p["osservazione"]}'))
    if len(blocchi) != len(Y_BLK):
        raise ValueError(f"{NOME}: {len(blocchi)} blocchi, la colonna ne tiene {len(Y_BLK)}")

    for j, (tag, righe, osservazione) in enumerate(blocchi):
        y = Y_BLK[j]
        primo = next(i for i, b in enumerate(blocchi_visti) if b >= j + 1)
        anim.append(keyframes(f"bl{j}", acceso_su(set(range(primo, n)), n)))
        dentro = [f'<rect class="blk" x="{XB}" y="{y}" width="{WB}" '
                  f'height="{H_BLK}" rx="8"/>',
                  f'<text class="tag" x="{XB + 14}" y="{y + 19}">{tag}</text>']
        for k, riga in enumerate(righe):
            dentro.append(f'<text class="riga" x="{XB + 14}" y="{y + 39 + k * 17}">'
                          f'{riga}</text>')
        if osservazione:
            dentro.append(f'<text class="riga oss" x="{XB + 14}" y="{y + 56}">'
                          f'{osservazione}</text>')
        # a riposo ci sono tutti: nessun attributo di opacità da scrivere
        corpo.append(f'<g class="blocco" style="animation:bl{j} var(--d) infinite">'
                     f'{"".join(dentro)}</g>')

    # la barra che cresce. Si allunga scoprendo un tratteggio invece che con un
    # `transform`: così il riposo è un attributo (`stroke-dashoffset="0"`, cioè
    # tutta scoperta) e non dipende da come si risolve l'origine di una scala.
    y0, y1 = Y_BLK[0], Y_BLK[-1] + H_BLK
    lungo = y1 - y0
    tappe = {}
    passo = 100.0 / n
    for i in range(n):
        alto = Y_BLK[blocchi_visti[i] - 1] + H_BLK - y0
        d = f"stroke-dashoffset:{lungo - alto:.1f}"
        tappe[round(i * passo, 3)] = d
        tappe[round(min((i + 1) * passo - 1.2, 99.9), 3)] = d
    tappe[100.0] = "stroke-dashoffset:0"
    anim.append(keyframes("bar", sorted(tappe.items())))
    corpo.append(f'<line class="bar" x1="{X_BAR + 3}" y1="{y0}" x2="{X_BAR + 3}" '
                 f'y2="{y1}" stroke-dasharray="{lungo} {lungo}" '
                 f'stroke-dashoffset="0" style="animation:bar var(--d) infinite"/>')

    # il conto dei blocchi, che è il modo onesto di dire «cresce»: quanti
    # pezzi rilegge il modello, non quanti token (che questo agente non ha)
    for b in sorted(set(blocchi_visti)):
        attivi = {i for i, v in enumerate(blocchi_visti) if v == b}
        anim.append(keyframes(f"cn{b}", acceso_su(attivi, n)))
        op = 1 if (n - 1) in attivi else 0
        testo = ("1 blocco, riletto a ogni giro" if b == 1
                 else f"{b} blocchi, riletti a ogni giro")
        corpo.append(f'<text class="cnt" x="{XB}" y="{y1 + 26}" opacity="{op}" '
                     f'style="animation:cn{b} var(--d) infinite">{testo}</text>')

    # ---------------------------------------------------------------- la coda
    corpo += [
        f'<text class="note" x="30" y="400">Un ciclo finisce in due modi. Qui al '
        f'giro {len(passi)}, perché l\'agente ha la risposta;</text>',
        f'<text class="note" x="30" y="418">se non l\'avesse, lo fermerebbe il '
        f'limite: max_passi = {MAX_PASSI}.</text>',
    ]

    return Figura(
        larghezza=760, altezza=434,
        alt="A sinistra il ciclo di un agente: tre caselle collegate in cerchio, "
            "pensa (Thought), agisce (Action) e osserva (Observation). "
            "L'evidenziazione gira di casella in casella: il modello pensa, "
            "chiama uno strumento, il sistema lo esegue e il risultato torna "
            "indietro. Al centro un contagiri arriva a giro 3 di 5. Al terzo "
            "giro il modello non chiama nessuno strumento: un ramo scende fuori "
            "dal cerchio verso la risposta finale, cioè che il paper è del 2017 "
            "e sono 9 anni fa nel 2026. A destra la colonna del contesto si "
            "allunga di un blocco a ogni giro: prima la sola domanda, poi la "
            "ricerca con la sua osservazione 2017, poi il calcolo con la sua "
            "osservazione 9, e una barra verticale accanto cresce insieme a "
            "loro fino a tre blocchi.",
        corpo="".join(corpo),
        stile=f"""    .box  {{ fill:{CREAM}; stroke:{BORDER_STRONG}; stroke-width:2; }}
    .hl   {{ fill:{TERRACOTTA}; fill-opacity:0.15; stroke:{TERRACOTTA}; stroke-width:3; }}
    .nome {{ font-family:{SANS}; font-size:16px; font-weight:700; fill:{INK}; }}
    .eng  {{ font-family:{SANS}; font-size:11px; fill:{FG_MUTED}; }}
    .fre  {{ stroke:{BORDER_STRONG}; stroke-width:2.2; fill:none; }}
    .fre2 {{ stroke:{TERRACOTTA}; stroke-width:3; fill:none; }}
    .frv  {{ stroke:{TEAL}; stroke-width:3; fill:none; }}
    .mini {{ font-family:{SANS}; font-size:11px; fill:{FG_MUTED}; }}
    .miniv {{ font-family:{SANS}; font-size:11px; fill:{TEAL}; }}
    .giro {{ font-family:{SANS}; font-size:14px; font-weight:700; fill:{TERRACOTTA}; }}
    .usb  {{ fill:{CREAM}; stroke:{TEAL}; stroke-width:2.5; }}
    .anst {{ font-family:{SANS}; font-size:11px; font-weight:700; fill:{TEAL}; }}
    .blk  {{ fill:{CREAM}; stroke:{BORDER_STRONG}; stroke-width:1.6; }}
    .tag  {{ font-family:{SANS}; font-size:11px; font-weight:700; fill:{FG_MUTED}; }}
    .riga {{ font-family:{SANS}; font-size:12.5px; fill:{INK}; }}
    .oss  {{ fill:{TERRACOTTA}; font-weight:700; }}
    .bar  {{ stroke:{TERRACOTTA}; stroke-width:6; stroke-opacity:0.55;
            stroke-linecap:round; fill:none; }}
    .cnt  {{ font-family:{SANS}; font-size:12.5px; font-weight:700; fill:{TERRACOTTA}; }}
    .note {{ font-family:{SANS}; font-size:12.5px; fill:{FG_MUTED}; }}""",
        animazioni=anim,
        durata=n * 1.35,
        fermi=".hl, .fre2, .giro, .fine, .blocco, .bar, .cnt",
    )
