"""La generazione a blocchi di un video: i blocchi nascono, la finestra scorre.

Una figura ferma di una generazione autoregressiva a blocchi e' una fila di
riquadri con una cornice sopra, e nasconde le due cose che sono fatte di tempo:
ogni blocco comincia come rumore e si ripulisce in pochi passi, e la finestra
della cache, larga w fotogrammi latenti (cioe' w / 3 blocchi da tre), segue il
blocco che nasce e a ogni blocco finito lascia fuori il piu' vecchio, che resta
sullo schermo ma il modello non lo guarda piu'. La finestra e' quella del testo,
w = 6, che e' anche la scelta di Matrix-Game 2.0 con blocchi da tre.

Il conto della scena e' tutto qui, e le guardie lo difendono:

* i blocchi nascono uno alla volta da sinistra a destra, e ciascuno attraversa
  T livelli di rumore strettamente decrescenti fino a zero (T passi);
* mentre si ripulisce il blocco k la finestra contiene i blocchi finiti
  precedenti, al piu' w; a blocco finito il piu' vecchio esce;
* a fine corsa la finestra copre esattamente gli ultimi w blocchi, e gli altri
  N - w sono quelli scartati.

Lo stato di riposo e' l'ultimo istante: tutti i blocchi puliti, la finestra
sugli ultimi w, i piu' vecchi attenuati.
"""

import math
import random

from paithon_svg import *

NOME = "video-cache-che-scorre"
TITOLO = "i blocchi nascono da rumore, la finestra della cache scorre"

N = 7            # blocchi mostrati
T = 4            # passi di denoising per blocco (i modelli distillati)
FOT = 3          # fotogrammi latenti per blocco
W_LAT = 6        # la finestra w del testo, in fotogrammi latenti
W = W_LAT // FOT # la stessa finestra, in blocchi
SEME = 11

A_PAROLE = {2: "due", 3: "tre", 4: "quattro", 5: "cinque", 6: "sei",
            7: "sette", 8: "otto"}


def simula():
    """Per ogni slot (un passo di un blocco): rumore, finestra, scartati."""
    livelli = [(T - j) / T for j in range(T)]          # 1, .75, .5, .25
    slot = []                                          # (blocco, passo)
    for k in range(N):
        for j in range(T):
            slot.append((k, j))
    finestre = []                                      # W_k, dopo il blocco k
    for k in range(N):
        finestre.append((max(0, k - W + 1), k))

    # --- guardie: la figura non nasce se smette di dire il vero ------------
    if W * FOT != W_LAT or W < 2:
        raise AssertionError("la finestra deve contenere blocchi interi, e piu' "
                             "di uno: w e' in fotogrammi latenti")
    if any(not (a > b) for a, b in zip(livelli, livelli[1:] + [0.0])):
        raise AssertionError("i livelli di rumore non scendono strettamente")
    if len(livelli) != T or livelli[0] != 1.0:
        raise AssertionError("ogni blocco deve partire da rumore pieno in T passi")
    for k, (a, b) in enumerate(finestre):
        if b != k:
            raise AssertionError("la finestra non segue il blocco finito")
        if b - a + 1 > W:
            raise AssertionError(f"finestra di {b - a + 1} blocchi, oltre w = {W}")
        if k >= W and (a, b) != (k - W + 1, k):
            raise AssertionError("a regime la finestra deve avere w blocchi")
    a, b = finestre[-1]
    scartati = list(range(0, a))
    if (b - a + 1, len(scartati)) != (W, N - W):
        raise AssertionError("a fine corsa: w dentro e N - w fuori")
    if [x for x, _ in slot[::T]] != list(range(N)):
        raise AssertionError("i blocchi devono nascere in ordine, da sinistra")
    return livelli, finestre, scartati


# --- geometria --------------------------------------------------------------
FW, FH, FG = 18, 46, 3                  # fotogramma latente, e spazio fra due
BW = FOT * FW + (FOT - 1) * FG          # larghezza del blocco
BG = 16                                 # spazio fra blocchi
X0, Y0 = 40, 92
NSLOT = N * T + 1                       # l'ultimo e' la sosta a corsa finita


def bx(k):
    return X0 + k * (BW + BG)


def pct(s):
    return s * 100.0 / NSLOT


def tenuto(nome, stati, anim):
    """`@keyframes` a gradini: stati = [(slot, dichiarazione)], tenuti fino al
    successivo; l'ultimo vale fino a fine ciclo, cioe' e' il riposo."""
    tappe = []
    for i, (s, d) in enumerate(stati):
        tappe.append((pct(s) if i else 0.0, d))
        fine = pct(stati[i + 1][0]) - 0.01 if i + 1 < len(stati) else 100.0
        tappe.append((fine, d))
    anim.append(keyframes(nome, tappe))
    return f"animation:{nome} var(--d) linear infinite"


def costruisci() -> Figura:
    livelli, finestre, scartati = simula()
    rnd = random.Random(SEME)
    corpo, anim = [], []

    # il rumore: una texture fitta, un solo tassello
    cel = 6
    celle = "".join(
        f'<rect x="{c * cel}" y="{r * cel}" width="{cel - 1}" height="{cel - 1}" '
        f'fill="{INK}" fill-opacity="{rnd.uniform(0.25, 0.95):.2f}"/>'
        for r in range(3) for c in range(3))
    corpo.append(f'<defs><pattern id="rum" width="{3 * cel}" height="{3 * cel}" '
                 f'patternUnits="userSpaceOnUse"><rect width="{3 * cel}" '
                 f'height="{3 * cel}" fill="{CREAM}"/>{celle}</pattern></defs>')

    corpo.append(f'<text class="ttl" x="{X0}" y="32">un blocco alla volta, da '
                 f'sinistra a destra</text>')
    corpo.append(f'<text class="lbs" x="{X0}" y="54">ogni blocco parte da rumore '
                 f'e si ripulisce in {T} passi</text>')

    # --- i blocchi ---------------------------------------------------------
    for k in range(N):
        x = bx(k)
        nasce = k * T
        base = tenuto(f"b{k}", [(0, "opacity:0"), (nasce, "opacity:1")], anim) \
            if k else ""
        g = [f'<text class="num" x="{x + BW / 2:.1f}" y="{Y0 - 20}" '
             f'text-anchor="middle">{k + 1}</text>']
        for i in range(FOT):
            f = k * FOT + i
            fx = x + i * (FW + FG)
            cy = Y0 + FH / 2 + 14 * math.sin(f * 0.55)
            g.append(f'<rect class="fr" x="{fx}" y="{Y0}" width="{FW}" '
                     f'height="{FH}" rx="2"/>'
                     f'<circle class="dot" cx="{fx + FW / 2}" cy="{cy:.1f}" r="5"/>')
        corpo.append(f'<g style="{base}">{"".join(g)}</g>' if k
                     else f'<g>{"".join(g)}</g>')

        # il rumore sopra il blocco, che scende di livello a ogni passo
        stati = [(0, "opacity:0")]
        stati += [(nasce + j, f"opacity:{lv:.2f}") for j, lv in enumerate(livelli)]
        stati += [(nasce + T, "opacity:0")]
        st = tenuto(f"r{k}", stati, anim)
        for i in range(FOT):
            fx = x + i * (FW + FG)
            corpo.append(f'<rect fill="url(#rum)" x="{fx}" y="{Y0}" width="{FW}" '
                         f'height="{FH}" rx="2" opacity="0" style="{st}"/>')

    # --- gli scartati: attenuati dal momento in cui escono dalla finestra ----
    for k in scartati:
        esce = (k + W + 1) * T          # W_{k+W} e' la prima che non lo contiene
        st = tenuto(f"s{k}", [(0, "opacity:0"), (esce, "opacity:0.62")], anim)
        corpo.append(f'<rect class="dim" x="{bx(k) - 4}" y="{Y0 - 4}" '
                     f'width="{BW + 8}" height="{FH + 8}" rx="4" '
                     f'opacity="0.62" style="{st}"/>')

    # --- la finestra: stessa larghezza a regime, piu' stretta all'inizio ----
    a_fin, b_fin = finestre[-1]
    wx = bx(a_fin) - 7
    ww = (b_fin - a_fin) * (BW + BG) + BW + 14
    stati_t, stati_o = [], [(0, "opacity:0"), (T, "opacity:1")]
    for k, (a, b) in enumerate(finestre):
        dx = bx(a) - bx(a_fin)
        s = ((b - a) * (BW + BG) + BW + 14) / ww
        stati_t.append(((k + 1) * T,
                        f"opacity:1;transform:translate({dx:.1f}px,0px) "
                        f"scale({s:.4f},1)"))
    stati_t.insert(0, (0, stati_t[0][1].replace("opacity:1", "opacity:0")))
    st = tenuto("fin", stati_t, anim)
    org = f"transform-origin:{wx}px 0px;"
    corpo.append(f'<rect class="fin" x="{wx}" y="{Y0 - 10}" width="{ww}" '
                 f'height="{FH + 20}" rx="6" style="{org}{st}"/>')

    st = tenuto("fet", [(0, "opacity:0;transform:translate(0px,0px)")] +
                [((k + 1) * T, f"opacity:1;transform:translate("
                  f"{bx(a) - bx(a_fin):.1f}px,0px)")
                 for k, (a, b) in enumerate(finestre)], anim)
    corpo.append(f'<text class="etf" x="{wx}" y="{Y0 + FH + 30}" '
                 f'style="{st}">finestra della cache<tspan class="etf2" '
                 f'x="{wx}" dy="16">{W} blocchi, w = {W_LAT}</tspan></text>')

    # --- legenda: che cosa e' fuori dalla finestra ---------------------------
    xl = bx(0)
    st = tenuto("fuo", [(0, "opacity:0"), ((W + 1) * T, "opacity:1")], anim)
    corpo.append(f'<text class="lbs" x="{xl}" y="{Y0 + FH + 66}" '
                 f'style="{st}">fuori dalla finestra</text>')
    corpo.append(f'<text class="lbs" x="{X0}" y="{Y0 + FH + 98}">i blocchi scartati '
                 f'restano nel video; il modello non li guarda più</text>')

    larg = bx(N - 1) + BW + 40
    return Figura(
        larghezza=larg, altezza=Y0 + FH + 118,
        alt=f"Una fila di {A_PAROLE[N]} blocchi di fotogrammi latenti, "
            f"{A_PAROLE[FOT]} fotogrammi per blocco. I blocchi nascono da sinistra a "
            f"destra: ciascuno comincia come una texture fitta di rumore e in "
            f"{A_PAROLE[T]} passi diventa pulito. Una cornice, la finestra "
            f"della cache, larga {A_PAROLE[W]} blocchi, cioè "
            f"{A_PAROLE[W_LAT]} fotogrammi latenti, segue il blocco che "
            f"si sta generando; a ogni blocco nuovo lascia fuori il più "
            f"vecchio, che resta visibile ma attenuato. Alla fine i blocchi "
            f"sono tutti puliti, la finestra copre gli ultimi {A_PAROLE[W]} e "
            f"i primi {A_PAROLE[N - W]} sono attenuati, fuori dalla finestra.",
        corpo="".join(corpo),
        stile=f"""    .fr {{ fill:{CREAM}; stroke:{BORDER_STRONG}; stroke-width:1.5; }}
    .dot {{ fill:{TERRACOTTA}; }}
    .dim {{ fill:{CREAM}; stroke:none; }}
    .fin {{ fill:{OCRA}; fill-opacity:0.16; stroke:{OCRA}; stroke-width:3;
            vector-effect:non-scaling-stroke; }}
    .num {{ font-family:{SANS}; font-size:12px; fill:{FG_MUTED}; }}
    .etf {{ font-family:{SANS}; font-size:13px; font-weight:700; fill:{INK}; }}
    .etf2 {{ font-size:12px; font-weight:400; }}""",
        animazioni=anim,
        durata=NSLOT * 0.42,
        fermi='[style*="animation"]',
    )
