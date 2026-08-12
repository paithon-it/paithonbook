"""Byte Pair Encoding: le fusioni si accumulano e le parole si accorciano.

L'algoritmo è quello della sezione `NaturalLanguageProcessing/tokenizzatori.md`
(stesso corpus giocattolo, stessa regola di spareggio) e **gira davvero**: le
coppie, i conteggi, le segmentazioni a ogni passo e la lunghezza totale del
corpus escono da qui. Nessun numero è scritto a mano, quindi la figura non può
smentire il testo; un `assert` lo verifica contro le quattro fusioni che la
sezione svolge a penna.

Il movimento è una dissolvenza fra stati: ogni segmentazione è disegnata alle
proprie coordinate vere, e nel tempo cambia soltanto l'opacità. Lo stato di
riposo è l'ultimo, quattro fusioni fatte, dove `rosso` è ormai un token solo:
è quello che vedono la stampa, il PDF e `prefers-reduced-motion`, ed è
esattamente il punto della sezione.
"""

from collections import Counter

from paithon_svg import *

NOME = "bpe-fusioni"
TITOLO = "BPE impara il vocabolario, una fusione per volta"

# Il corpus giocattolo della sezione, parola -> frequenza.
CORPUS = {"basso": 6, "bassotto": 2, "bosso": 3, "rosso": 9, "rossetto": 5}
FUSIONI = 4

# Le quattro fusioni che la sezione calcola a mano. Qui non si disegnano: si
# usano per fermare la figura se un giorno il corpus o la regola di spareggio
# cambiassero senza che il testo lo sappia.
ATTESE = [("s", "s", 25), ("ss", "o", 20), ("r", "o", 14), ("ro", "sso", 9)]


# --------------------------------------------------------------------------
# L'algoritmo (le stesse tre funzioni del capitolo)
# --------------------------------------------------------------------------
def conta_coppie(pezzi):
    """Frequenza di ogni coppia adiacente, pesata sulle occorrenze della parola."""
    coppie = Counter()
    for parola, simboli in pezzi.items():
        for coppia in zip(simboli, simboli[1:]):
            coppie[coppia] += CORPUS[parola]
    return coppie


def fondi(simboli, coppia):
    """Sostituisce ogni occorrenza della coppia con il simbolo unito."""
    uniti, i = [], 0
    while i < len(simboli):
        if i < len(simboli) - 1 and (simboli[i], simboli[i + 1]) == coppia:
            uniti.append(simboli[i] + simboli[i + 1])
            i += 2
        else:
            uniti.append(simboli[i])
            i += 1
    return tuple(uniti)


def addestra():
    """Stati del corpus (uno per passo) e fusioni scelte, col loro conteggio."""
    pezzi = {parola: tuple(parola) for parola in CORPUS}
    stati, fusioni = [pezzi], []
    for _ in range(FUSIONI):
        coppie = conta_coppie(pezzi)
        # la piu' frequente; a parita' di conteggio, la prima in ordine alfabetico
        a, b = min(coppie, key=lambda c: (-coppie[c], c))
        fusioni.append((a, b, coppie[(a, b)]))
        pezzi = {p: fondi(s, (a, b)) for p, s in pezzi.items()}
        stati.append(pezzi)
    return stati, fusioni


def lunghezza(pezzi):
    """Quanto è lungo il corpus, in token, pesando ogni parola per la frequenza."""
    return sum(len(s) * CORPUS[p] for p, s in pezzi.items())


# --------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------
W, H = 676, 330
X_FREQ, X_TOKEN = 66, 80          # colonna delle frequenze, poi le scatole
Y_HEAD, Y0, PITCH, ALT_BOX = 28, 52, 48, 32
PAD, CH, GAP = 8.0, 11.0, 6.0     # imbottitura, larghezza per carattere, spazio
X_LISTA, X_CONT = 400, 646        # elenco delle fusioni e colonna dei conteggi

TEN, FADE = 0.60, 0.20            # tenuta e dissolvenza, in frazioni di passo


def costruisci() -> Figura:
    stati, fusioni = addestra()
    assert [(a, b, c) for a, b, c in fusioni] == ATTESE, \
        f"le fusioni non sono più quelle della sezione: {fusioni}"

    n = len(stati)                # 5: i caratteri più le quattro fusioni
    passo = 100.0 / n
    corpo, anim, nomi = [], [], {}

    def visibile(a: int, b: int) -> tuple[str, int]:
        """(@keyframes, opacità di riposo) per chi si vede dallo stato a al b.

        Il valore al 100% è quello dell'ultimo stato, ed è lo stesso che
        l'elemento porta come attributo: riposo e fine dell'animazione non
        possono divergere.
        """
        finale = 1 if b == n - 1 else 0
        if (a, b) not in nomi:
            nome = f"v{a}{b}"
            tappe = [(0.0, f"opacity:{1 if a == 0 else 0}")]
            if a > 0:
                tappe.append((a * passo - passo * (FADE + 0.15), "opacity:0"))
                tappe.append((a * passo, "opacity:1"))
            tappe.append((b * passo + passo * TEN, "opacity:1"))
            if finale:
                tappe.append((100.0, "opacity:1"))
            else:
                tappe.append((b * passo + passo * (TEN + FADE), "opacity:0"))
                tappe.append((100.0, "opacity:0"))
            anim.append(keyframes(nome, tappe))
            nomi[(a, b)] = nome
        return nomi[(a, b)], finale

    def scatole(simboli, ytop: float, neo: str | None) -> str:
        """Una segmentazione, disegnata come scatole affiancate."""
        pezzi, x = [], float(X_TOKEN)
        for s in simboli:
            larg = 2 * PAD + CH * len(s)
            stile = "neo" if s == neo else ("fusa" if len(s) > 1 else "")
            pezzi.append(f'<rect class="{("box " + stile).strip()}" x="{x:.0f}" '
                         f'y="{ytop:.0f}" width="{larg:.0f}" height="{ALT_BOX}" rx="6"/>')
            pezzi.append(f'<text class="{("sim " + stile).strip()}" '
                         f'x="{x + larg / 2:.1f}" y="{ytop + 22:.0f}" '
                         f'text-anchor="middle">{s}</text>')
            x += larg + GAP
        return "".join(pezzi)

    def gruppo(dentro: str, classe: str, a: int, b: int) -> str:
        nome, op = visibile(a, b)
        return (f'<g class="{classe}" opacity="{op}" '
                f'style="animation:{nome} var(--d) infinite">{dentro}</g>')

    # ---- intestazioni ----------------------------------------------------
    corpo.append(f'<text class="lbs" x="{X_FREQ}" y="{Y_HEAD}" text-anchor="end">freq.</text>')
    corpo.append(f'<text class="lbs" x="{X_TOKEN}" y="{Y_HEAD}">corpus: una scatola, un token</text>')
    corpo.append(f'<text class="lbs" x="{X_LISTA}" y="{Y_HEAD}">le fusioni, in ordine</text>')
    corpo.append(f'<text class="lbs" x="{X_CONT}" y="{Y_HEAD}" text-anchor="end">conteggio</text>')

    # ---- le cinque parole, uno stato per volta ---------------------------
    for i, parola in enumerate(CORPUS):
        seq = [stati[k][parola] for k in range(n)]
        ytop = Y0 + i * PITCH
        corpo.append(f'<text class="lbs" x="{X_FREQ}" y="{ytop + 22}" '
                     f'text-anchor="end">×{CORPUS[parola]}</text>')

        k = 0
        while k < n:
            j = k                       # gli stati in cui la parola non cambia
            while j + 1 < n and seq[j + 1] == seq[k]:
                j += 1
            # il token appena nato si accende in terracotta, ma solo nel passo
            # che lo ha creato: se la parola resta ferma anche dopo, il resto
            # della sosta va disegnato a parte, in teal come gli altri.
            neo = fusioni[k - 1][0] + fusioni[k - 1][1] if k else None
            neo = neo if neo in seq[k] else None
            if neo and j > k:
                corpo.append(gruppo(scatole(seq[k], ytop, neo), "st", k, k))
                corpo.append(gruppo(scatole(seq[k], ytop, None), "st", k + 1, j))
            else:
                corpo.append(gruppo(scatole(seq[k], ytop, neo), "st", k, j))
            k = j + 1

    # ---- l'elenco delle fusioni, che si allunga --------------------------
    centro = Y0 + (n - 1) * PITCH / 2 + ALT_BOX / 2      # centro del blocco parole
    corpo.append(gruppo(f'<text class="fu-n" x="{X_LISTA}" y="{centro + 5.5:.1f}">'
                        f'ancora nessuna: solo caratteri</text>', "fus", 0, 0))
    for k, (a, b, conteggio) in enumerate(fusioni, start=1):
        y = centro + (k - 1 - (len(fusioni) - 1) / 2) * PITCH + 5.5
        riga = (f'<text class="fu-n" x="{X_LISTA}" y="{y:.1f}">{k}.</text>'
                f'<text class="fu-p" x="{X_LISTA + 24}" y="{y:.1f}">{a} + {b}</text>'
                f'<text class="fu-n" x="{X_LISTA + 116}" y="{y:.1f}">→</text>'
                f'<text class="fu-r" x="{X_LISTA + 140}" y="{y:.1f}">{a + b}</text>'
                f'<text class="fu-c" x="{X_CONT}" y="{y:.1f}" '
                f'text-anchor="end">{conteggio}</text>')
        corpo.append(gruppo(riga, "fus", k, n - 1))

    # ---- quanto è lungo il corpus, passo per passo -----------------------
    for k, pezzi in enumerate(stati):
        testo = (f'<text class="cnt" x="{X_TOKEN}" y="{H - 24}">'
                 f'tutto il corpus: {lunghezza(pezzi)} token</text>')
        corpo.append(gruppo(testo, "cnt-g", k, k))

    return Figura(
        larghezza=W, altezza=H,
        alt="Cinque parole di un corpus giocattolo, spezzate in caratteri. A "
            "ogni passo la coppia adiacente più frequente diventa un simbolo "
            "solo: le scatole si saldano, l'elenco delle fusioni si allunga e "
            "il corpus si accorcia, finché la parola «rosso» sta in un token "
            "solo.",
        corpo="".join(corpo),
        stile=f"""    rect.box  {{ fill:none; stroke:{BORDER_STRONG}; stroke-width:1.6; }}
    rect.fusa {{ stroke:{TEAL}; stroke-width:2.4; }}
    rect.neo  {{ stroke:{TERRACOTTA}; stroke-width:2.8; }}
    text.sim  {{ font-family:{SANS}; font-size:16px; fill:{INK}; }}
    text.fusa {{ fill:{TEAL}; }}
    text.neo  {{ fill:{TERRACOTTA}; font-weight:600; }}
    .fu-n {{ font-family:{SANS}; font-size:15px; fill:{FG_MUTED}; }}
    .fu-p {{ font-family:{SANS}; font-size:15px; fill:{INK}; }}
    .fu-r {{ font-family:{SANS}; font-size:15px; font-weight:600; fill:{TEAL}; }}
    .fu-c {{ font-family:{SANS}; font-size:15px; fill:{FG_MUTED}; }}
    .cnt  {{ font-family:{SANS}; font-size:15px; fill:{FG_MUTED}; }}""",
        animazioni=anim,
        durata=n * 2.0,
        fermi=".st, .fus, .cnt-g",
    )
