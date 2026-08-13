"""DQN: il buffer pesca a caso, la copia congelata resta indietro.

I due accorgimenti che rendono stabile il DQN rompono la stessa cosa, una
correlazione nel tempo, in due modi diversi: il buffer scrive in ordine e
**legge a caso**, la rete-target **resta ferma** mentre la online si muove e
ogni $C$ passi viene ricopiata. Nessuno dei due si vede in una figura ferma:
la pescata casuale e il ritardo della copia sono fatti di tempo.

Qui non c'e' niente di disegnato a mano. Le celle che si accendono sono gli
indici che `random.Random(SEME).sample` estrae davvero, e sono gli stessi che
alimentano l'aggiornamento; le due curve sono il valore $Q$ di una stessa
coppia $(s,a)$ calcolato con $\\theta$ e con $\\theta^{-}$, cioe' la stessa
quantita' vista dalle due reti. Cosi' la scaletta ocra non e' un'illustrazione
del ritardo: **e'** il ritardo, e infatti tocca la curva teal esattamente nei
passi in cui avviene la copia (`assert` in `addestra()`).

Il capitolo non fissa un numero per $C$ (scrive «ogni $C$ passi») ne' puo'
mostrare un milione di celle: la figura e' in scala ridotta e lo dice, con i
numeri veri del testo fra parentesi (32 transizioni su un milione).

Lo stato di riposo e' l'ultimo passo: l'ultimo minibatch acceso, le due curve
intere, la target che ha appena preso due passi di ritardo.
"""

import math
import random

from paithon_svg import *

NOME = "dqn-stabilita"
TITOLO = "i due accorgimenti che rendono stabile il DQN"

# --- la miniatura -----------------------------------------------------------
N_BUFFER = 24        # celle del buffer (nel libro: un milione di transizioni)
COLONNE = 6
BATCH = 4            # celle pescate a ogni passo (nel libro: 32)
PASSI = 8            # passi di addestramento mostrati
C = 3                # la copia theta^- <- theta ogni C passi
SEME = 11

# --- l'ambiente giocattolo su cui l'algoritmo gira davvero ------------------
# Un corridoio di cinque posizioni: piu' a destra si arriva, piu' si guadagna.
# Serve solo a far muovere dei numeri veri: il capitolo parla di pixel Atari,
# e la figura infatti non nomina l'ambiente, mostra una Q che sale.
STATI = 5
GAMMA = 0.9
ETA = 0.3
S_RIF, A_RIF = 2, 1   # la coppia (s, a) di cui si segue il valore


def raccogli():
    """Le transizioni del buffer, nell'ordine in cui sono state vissute."""
    rnd = random.Random(SEME)
    dati, s = [], 1
    while len(dati) < N_BUFFER:
        a = rnd.randint(0, 1)                       # politica di esplorazione
        s2 = min(STATI - 1, max(0, s + (1 if a else -1)))
        dati.append((s, a, s2 / (STATI - 1), s2))   # (s, a, r, s')
        s = s2
    return dati


def q(th, s, a):
    """Approssimatore lineare a parametri condivisi fra le azioni."""
    return th[0] + th[1] * s / (STATI - 1) + th[2] * a


def addestra():
    """Il ciclo di DQN in miniatura: pesca, aggiorna, ogni C passi ricopia.

    Restituisce le pescate vere e il valore della stessa coppia (s, a) visto
    dalla rete online e dalla rete-target.
    """
    dati = raccogli()
    rnd = random.Random(SEME + 1000)
    theta = [0.0, 0.0, 0.0]
    theta_m = list(theta)                       # theta^-, la copia congelata
    pescate = []
    q_on, q_tg = [q(theta, S_RIF, A_RIF)], [q(theta_m, S_RIF, A_RIF)]
    storia = [list(theta)]                      # theta dopo k aggiornamenti

    for k in range(1, PASSI + 1):
        idx = sorted(rnd.sample(range(N_BUFFER), BATCH))
        g = [0.0, 0.0, 0.0]
        for i in idx:
            s, a, r, s2 = dati[i]
            # bersaglio con i pesi congelati: r + gamma max_a' Q(s', a'; theta^-)
            y = r + GAMMA * max(q(theta_m, s2, 0), q(theta_m, s2, 1))
            delta = y - q(theta, s, a)
            grad = (1.0, s / (STATI - 1), float(a))
            for j in range(3):
                g[j] += delta * grad[j] / BATCH
        for j in range(3):
            theta[j] += ETA * g[j]
        if k % C == 0:
            theta_m = list(theta)               # la copia, ogni C passi
        pescate.append(idx)
        storia.append(list(theta))
        q_on.append(q(theta, S_RIF, A_RIF))
        q_tg.append(q(theta_m, S_RIF, A_RIF))

    # --- guardie: se un giorno la figura smettesse di dire il vero, non nasce
    for k in range(PASSI + 1):
        # la target e' esattamente la online dell'ultima copia: e' il ritardo
        atteso = q(storia[C * (k // C)], S_RIF, A_RIF)
        if abs(q_tg[k] - atteso) > 1e-12:
            raise AssertionError(f"passo {k}: la target non e' la copia del "
                                 f"passo {C * (k // C)}")
        if k and k % C and abs(q_tg[k] - q_tg[k - 1]) > 1e-12:
            raise AssertionError(f"passo {k}: la target si e' mossa fuori dalla copia")
        if k and abs(q_on[k] - q_on[k - 1]) < 1e-6:
            raise AssertionError(f"passo {k}: la online non si e' mossa")
    if sum(1 for k in range(1, PASSI + 1) if k % C == 0) != PASSI // C:
        raise AssertionError("le copie non sono una ogni C passi")
    if any(b - a < 0 for a, b in zip(q_on, q_on[1:])):
        raise AssertionError("la curva online non sale: il disegno la dà in salita")
    for idx in pescate:
        # una pescata contigua sarebbe la correlazione che il buffer rompe
        if idx[-1] - idx[0] == BATCH - 1:
            raise AssertionError(f"pescata contigua: {idx}")
    return pescate, q_on, q_tg


# --- geometria --------------------------------------------------------------
GX, GY, CW, CH, GAP = 40, 66, 36, 28, 6
RIGHE = N_BUFFER // COLONNE
GW = COLONNE * CW + (COLONNE - 1) * GAP
GH = RIGHE * CH + (RIGHE - 1) * GAP


def cella(i):
    return GX + (i % COLONNE) * (CW + GAP), GY + (i // COLONNE) * (CH + GAP)


def percorso(punti):
    """Lunghezza totale e lunghezze cumulate di una spezzata."""
    cum, tot = [0.0], 0.0
    for (x1, y1), (x2, y2) in zip(punti, punti[1:]):
        tot += math.hypot(x2 - x1, y2 - y1)
        cum.append(tot)
    return tot, cum


def fetta(k, n, anim, visti):
    """`@keyframes` che accende qualcosa nella fetta k e la spegne dopo.

    Il contatore in basso e il minibatch acceso hanno la stessa identica
    tempistica, ed e' il punto della figura: sono lo stesso passo visto da due
    parti. Un solo `@keyframes` per fetta, condiviso.
    """
    nome = f"f{k}"
    if nome in visti:
        return nome
    visti.add(nome)
    t0, _ = sosta(k, n)
    p = 100.0 / n
    fine = "opacity:1" if k == n - 1 else "opacity:0"
    tappe = []
    if t0 > 0.5:                       # la prima fetta e' gia' accesa a 0%
        tappe += [(0.0, "opacity:0"), (t0 - 0.4, "opacity:0")]
    tappe += [(t0, "opacity:1"), (min(t0 + p - 0.4, 99.9), "opacity:1")]
    if k < n - 1:
        tappe.append((min(t0 + p, 100.0), "opacity:0"))
    tappe.append((100.0, fine))
    anim.append(keyframes(nome, tappe))
    return nome


def rivela(nome, cum, tappe_idx, n, anim):
    """Scopre una spezzata passo per passo, con lo stato di riposo intera."""
    L = cum[-1]
    tappe = []
    for k in range(n):
        t0, t1 = sosta(k, n)
        d = f"stroke-dashoffset:{L - cum[tappe_idx[k]]:.1f}"
        tappe += [(t0, d), (t1, d)]
    tappe.append((100.0, "stroke-dashoffset:0"))
    anim.append(keyframes(nome, tappe))
    return L


def costruisci() -> Figura:
    pescate, q_on, q_tg = addestra()
    n = PASSI + 1                      # fette di timeline: partenza + PASSI
    corpo, anim, visti = [], [], set()

    # ---------------- pannello 1: il buffer -------------------------------
    corpo.append(f'<text class="ttl" x="{GX}" y="34">experience replay</text>')
    corpo.append(f'<text class="lbs" x="{GX}" y="56">{N_BUFFER} transizioni, '
                 f'in ordine di arrivo →</text>')
    for i in range(N_BUFFER):
        x, y = cella(i)
        corpo.append(f'<rect class="cel" x="{x}" y="{y}" width="{CW}" '
                     f'height="{CH}" rx="3"/>')

    for k, idx in enumerate(pescate, start=1):
        nome = fetta(k, n, anim, visti)
        celle = "".join(
            f'<rect x="{cella(i)[0] + 3}" y="{cella(i)[1] + 3}" '
            f'width="{CW - 6}" height="{CH - 6}" rx="2"/>' for i in idx)
        corpo.append(f'<g class="pes" opacity="{1 if k == PASSI else 0}" '
                     f'style="animation:{nome} var(--d) infinite">{celle}</g>')

    corpo.append(f'<rect class="pes leg" x="{GX}" y="228" width="14" height="14" rx="2"/>')
    corpo.append(f'<text class="lbs" x="{GX + 22}" y="240">a ogni passo, '
                 f'{BATCH} celle a caso</text>')
    corpo.append(f'<text class="lbs" x="{GX + 22}" y="258">nel testo: 32 su un milione</text>')

    # ---------------- pannello 2: le due reti ------------------------------
    vmax = max(q_on) * 1.14
    r = Riquadro(x=400, y=GY, larg=262, alt=GH,
                 xmin=-0.3, xmax=PASSI + 0.3, ymin=-0.10 * vmax, ymax=vmax)
    corpo.append(f'<text class="ttl" x="{r.x}" y="34">rete-target</text>')
    corpo.append(f'<text class="lbs" x="{r.x}" y="56">il valore Q di una stessa '
                 f'coppia (s, a)</text>')
    corpo.append(r.cornice())
    corpo.append(f'<line class="axc" x1="{r.x}" y1="{r.sy(0):.1f}" '
                 f'x2="{r.x + r.larg}" y2="{r.sy(0):.1f}"/>')
    for k in range(n):
        corpo.append(f'<line class="axc" x1="{r.sx(k):.1f}" y1="{r.y + r.alt}" '
                     f'x2="{r.sx(k):.1f}" y2="{r.y + r.alt + 5}"/>')
    corpo.append(f'<text class="lbs" x="{r.x + r.larg}" y="{r.y + r.alt + 22}" '
                 f'text-anchor="end">{PASSI} passi di addestramento</text>')

    # la online: un punto per passo
    p_on = [(r.sx(k), r.sy(q_on[k])) for k in range(n)]
    L_on = rivela("cre", percorso(p_on)[1], list(range(n)), n, anim)
    corpo.append('<polyline class="on" points="'
                 + " ".join(f"{x:.1f},{y:.1f}" for x, y in p_on)
                 + f'" stroke-dasharray="{L_on:.1f}" stroke-dashoffset="0"'
                 f' style="animation:cre var(--d) infinite"/>')

    # la target: ferma per C passi, poi lo scatto della copia
    p_tg, tappe_tg = [(r.sx(0), r.sy(q_tg[0]))], [0]
    for k in range(1, n):
        p_tg.append((r.sx(k), r.sy(q_tg[k - 1])))
        if abs(q_tg[k] - q_tg[k - 1]) > 1e-12:
            p_tg.append((r.sx(k), r.sy(q_tg[k])))
        tappe_tg.append(len(p_tg) - 1)
    L_tg = rivela("crt", percorso(p_tg)[1], tappe_tg, n, anim)
    corpo.append('<polyline class="tg" points="'
                 + " ".join(f"{x:.1f},{y:.1f}" for x, y in p_tg)
                 + f'" stroke-dasharray="{L_tg:.1f}" stroke-dashoffset="0"'
                 f' style="animation:crt var(--d) infinite"/>')

    # le due teste: quella teal si muove ogni passo, quella ocra scatta
    for nome, serie, cls in (("ton", q_on, "on"), ("ttg", q_tg, "tg")):
        xf, yf = r.sx(PASSI), r.sy(serie[PASSI])
        tappe = []
        for k in range(n):
            t0, t1 = sosta(k, n)
            d = (f"transform:translate({r.sx(k) - xf:.1f}px,"
                 f"{r.sy(serie[k]) - yf:.1f}px)")
            tappe += [(t0, d), (t1, d)]
        tappe.append((100.0, "transform:translate(0px,0px)"))
        anim.append(keyframes(nome, tappe))
        corpo.append(f'<circle class="testa {cls}" cx="{xf:.1f}" cy="{yf:.1f}" '
                     f'r="6" style="animation:{nome} var(--d) infinite"/>')

    # l'istante della copia: compare quando avviene e resta
    for k in range(C, PASSI + 1, C):
        t0, _ = sosta(k, n)
        anim.append(keyframes(f"cp{k}", [
            (0.0, "opacity:0"), (max(t0 - 0.4, 0.01), "opacity:0"),
            (t0, "opacity:1"), (100.0, "opacity:1")]))
        corpo.append(f'<text class="cop" x="{r.sx(k) + 6:.1f}" '
                     f'y="{r.sy(q_tg[k]) + 17:.1f}" '
                     f'style="animation:cp{k} var(--d) infinite">copia</text>')

    corpo.append(f'<line class="on" x1="{r.x}" y1="235" x2="{r.x + 18}" y2="235"/>')
    corpo.append(f'<text class="lbs" x="{r.x + 26}" y="240">θ  si muove a ogni passo</text>')
    corpo.append(f'<line class="tg" x1="{r.x}" y1="253" x2="{r.x + 18}" y2="253"/>')
    corpo.append(f'<text class="lbs" x="{r.x + 26}" y="258">θ⁻  ferma per C = {C} passi</text>')

    # ---------------- il contatore, comune ai due pannelli -----------------
    for k in range(n):
        nome = fetta(k, n, anim, visti)
        testo = "il buffer è pieno" if k == 0 else f"passo {k}"
        corpo.append(f'<text class="pas" x="351" y="294" text-anchor="middle" '
                     f'opacity="{1 if k == PASSI else 0}" '
                     f'style="animation:{nome} var(--d) infinite">{testo}</text>')

    return Figura(
        larghezza=702, altezza=312,
        alt="A sinistra il buffer di ventiquattro transizioni disposte in ordine "
            "di arrivo: a ogni passo si accendono quattro celle sparse, mai "
            "consecutive, che sono il minibatch pescato a caso. A destra il "
            "valore Q di una stessa coppia stato-azione: la curva della rete che "
            "impara sale a ogni passo, la scaletta della copia congelata resta "
            "ferma per tre passi e poi scatta a raggiungerla.",
        corpo="".join(corpo),
        stile=f"""    .cel {{ fill:{CREAM}; stroke:{BORDER_STRONG}; stroke-width:1.5; }}
    .pes {{ fill:{TEAL}; stroke:none; }}
    .leg {{ stroke:none; }}
    .on  {{ fill:none; stroke:{TEAL}; stroke-width:3; stroke-linejoin:round; }}
    .tg  {{ fill:none; stroke:{OCRA}; stroke-width:3; stroke-linejoin:round; }}
    .testa {{ stroke:{CREAM}; stroke-width:2; }}
    .testa.on {{ fill:{TEAL}; }}
    .testa.tg {{ fill:{OCRA}; }}
    .cop {{ font-family:{SANS}; font-size:13px; fill:{OCRA}; opacity:1; }}
    .pas {{ font-family:{SANS}; font-size:16px; font-weight:700; fill:{TERRACOTTA}; }}""",
        animazioni=anim,
        durata=n * 1.15,
        fermi=".pes, .on, .tg, .testa, .cop, .pas",
    )
