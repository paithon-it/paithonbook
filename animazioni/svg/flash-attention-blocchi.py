"""Flash Attention: la online softmax che scorre a blocchi e ricalibra.

Il tempo qui è il contenuto. La riga di punteggi non viene mai scritta per
intero: in ogni istante esiste solo il blocco che sta nella memoria veloce, e
quando quel blocco porta un massimo più grande di quello visto finora,
l'accumulatore si **ricalibra**, cioè `l` e `O` già sommati vengono riscalati
per il fattore α prima di aggiungere i nuovi termini.

I numeri non sono scritti a mano: `passo_online()` esegue davvero la ricorrenza
del capitolo (`m`, `l`, `O`, α) e `controlla()` verifica che il risultato finale
`O/l` coincida, a meno dell'errore di macchina, con la softmax calcolata in un
colpo solo su tutti e otto i punteggi. Se non coincidesse, la figura non si
genera.

I primi quattro punteggi sono quelli dell'esempio del capitolo, s = (1, 3, 2, 4)
a blocchi di due: al secondo blocco la figura mostra α ≈ 0,368 e l ≈ 1,553,
cioè esattamente i numeri del testo. Gli altri quattro continuano l'esempio per
far vedere i due casi che mancavano: un blocco che non cambia il massimo (α = 1,
niente da riscalare) e un blocco che lo cambia di nuovo.

Il disegno fermo è l'ultimo passo: la tabella completa della ricorrenza, la
finestra sull'ultimo blocco e il confronto finale. È ciò che finisce in stampa.
"""

import math

from paithon_svg import *

NOME = "flash-attention-blocchi"
TITOLO = "la online softmax scorre i blocchi e ricalibra"

# I punteggi di una query contro otto chiavi (una riga di S) e i valori
# corrispondenti. Nel kernel i v sono righe di V: qui sono scalari, così O si
# legge come un numero solo.
S = [1.0, 3.0, 2.0, 4.0, 0.0, 1.0, 5.0, 2.0]
V = [1.0, 4.0, 2.0, 5.0, 0.0, 3.0, 6.0, 2.0]
B = 2                      # quante chiavi entrano insieme in memoria veloce

# geometria della striscia
X0, PITCH = 118, 152
CELL_W, GAP = 58, 8
BLOCCO_W = 2 * CELL_W + GAP
Y_S, H_S = 58, 44
Y_V, H_V = 108, 30
Y_FIN, H_FIN = 48, 98

# geometria della tabella
COL = {"blocco": 90, "mt": 202, "m": 320, "alfa": 420, "l": 530, "O": 645}
Y_TIT, Y_TH1, Y_TH2, Y_RULE, Y_R0, DY_R = 222, 246, 262, 272, 294, 28


def passo_online():
    """La ricorrenza del capitolo, eseguita davvero, blocco per blocco."""
    m, l, O = -math.inf, 0.0, 0.0
    passi = []
    for j in range(0, len(S), B):
        indici = list(range(j, min(j + B, len(S))))
        m_bloc = max(S[i] for i in indici)              # il massimo del blocco
        m_nuovo = max(m, m_bloc)
        alfa = math.exp(m - m_nuovo) if math.isfinite(m) else None
        f = alfa if alfa is not None else 0.0           # al primo blocco non c'è nulla
        l = f * l + sum(math.exp(S[i] - m_nuovo) for i in indici)
        O = f * O + sum(math.exp(S[i] - m_nuovo) * V[i] for i in indici)
        m = m_nuovo
        passi.append({"indici": indici, "mt": m_bloc, "m": m,
                      "alfa": alfa, "l": l, "O": O})
    return passi


def controlla(passi) -> float:
    """La softmax in un colpo solo: se non coincide, la figura è sbagliata."""
    m = max(S)
    den = sum(math.exp(s - m) for s in S)
    pesi = [math.exp(s - m) / den for s in S]
    atteso = sum(p * v for p, v in zip(pesi, V))
    if abs(passi[-1]["l"] - den) > 1e-12:
        raise ValueError(f"{NOME}: somma online {passi[-1]['l']} ≠ {den}")
    ottenuto = passi[-1]["O"] / passi[-1]["l"]
    if abs(ottenuto - atteso) > 1e-12:
        raise ValueError(f"{NOME}: uscita online {ottenuto} ≠ {atteso}")
    return atteso


def num(x: float, dec: int = 3) -> str:
    return f"{x:.{dec}f}".replace(".", ",")


def appare(t0: float, t1: float | None = None, resta: bool = False):
    """Tappe di opacità: compare in t0 e, se non `resta`, sparisce dopo t1."""
    q = {0.0: f"opacity:{1 if t0 <= 0.01 else 0}"}
    if t0 > 0.01:
        q[max(t0 - 2.5, 0.3)] = "opacity:0"
    q[t0] = "opacity:1"
    if resta:
        q[100.0] = "opacity:1"
    else:
        q[t1] = "opacity:1"
        q[min(t1 + 2.5, 99.7)] = "opacity:0"
        q[100.0] = "opacity:0"
    return sorted(q.items())


def costruisci() -> Figura:
    passi = passo_online()
    esatto = controlla(passi)
    n_bloc = len(passi)
    n_stati = n_bloc + 1              # un blocco per stato, più la divisione finale
    ultimo = n_bloc - 1

    corpo, anim = [], []

    # ------------------------------------------------------------------ testa
    corpo.append('<text class="lbl" x="30" y="26">s: i punteggi di una query '
                 'contro otto chiavi (una riga di S), letti a blocchi di due</text>')

    # le celle vuote: la riga di S non viene scritta, i valori esistono ma
    # stanno in HBM. Due tratti diversi, perché sono due assenze diverse.
    for j in range(n_bloc):
        bx = X0 + j * PITCH
        for k in range(B):
            cx = bx + k * (CELL_W + GAP)
            corpo.append(f'<rect class="vuS" x="{cx}" y="{Y_S}" '
                         f'width="{CELL_W}" height="{H_S}" rx="5"/>')
            corpo.append(f'<rect class="vuV" x="{cx}" y="{Y_V}" '
                         f'width="{CELL_W}" height="{H_V}" rx="5"/>')

    # le celle piene: solo il blocco che in quel momento sta in memoria veloce
    for j, p in enumerate(passi):
        bx = X0 + j * PITCH
        t0, t1 = sosta(j, n_stati, tenuta=0.62)
        anim.append(keyframes(f"c{j}", appare(t0, t1, resta=(j == ultimo))))
        pezzi = []
        for k, i in enumerate(p["indici"]):
            cx = bx + k * (CELL_W + GAP)
            pezzi.append(f'<rect class="pieS" x="{cx}" y="{Y_S}" '
                         f'width="{CELL_W}" height="{H_S}" rx="5"/>')
            pezzi.append(f'<text class="valS" x="{cx + CELL_W / 2:.0f}" y="{Y_S + 29}" '
                         f'text-anchor="middle">{S[i]:g}</text>')
            pezzi.append(f'<rect class="pieV" x="{cx}" y="{Y_V}" '
                         f'width="{CELL_W}" height="{H_V}" rx="5"/>')
            pezzi.append(f'<text class="valV" x="{cx + CELL_W / 2:.0f}" y="{Y_V + 20}" '
                         f'text-anchor="middle">{V[i]:g}</text>')
        op = 1 if j == ultimo else 0
        corpo.append(f'<g class="cel" opacity="{op}" '
                     f'style="animation:c{j} var(--d) infinite">{"".join(pezzi)}</g>')

    corpo.append(f'<text class="lbs" x="100" y="{Y_S + 28}" text-anchor="end">'
                 f'punteggi s</text>')
    corpo.append(f'<text class="lbs" x="100" y="{Y_V + 20}" text-anchor="end">'
                 f'valori v</text>')

    # le etichette dei blocchi: quelli, in HBM, ci sono sempre
    for j in range(n_bloc):
        cx = X0 + j * PITCH + BLOCCO_W / 2
        corpo.append(f'<text class="bloc" x="{cx:.0f}" y="{Y_V + H_V + 26}" '
                     f'text-anchor="middle">K{chr(0x2080 + j + 1)},V{chr(0x2080 + j + 1)}</text>')

    # la finestra della memoria veloce: riposo sull'ultimo blocco, e l'animazione
    # la riporta indietro fino al primo. Coordinate vere, nessun transform fermo.
    x_fin = X0 - 8 + ultimo * PITCH
    tappe = []
    for i in range(n_stati):
        j = min(i, ultimo)
        t0, t1 = sosta(i, n_stati, tenuta=0.62)
        d = f"transform:translate({(j - ultimo) * PITCH:.0f}px,0px)"
        tappe += [(t0, d), (t1, d)]
    tappe.append((100.0, "transform:translate(0px,0px)"))
    anim.append(keyframes("fin", sorted(dict(tappe).items())))
    corpo.append(
        f'<g class="fin" style="animation:fin var(--d) infinite">'
        f'<rect class="fbox" x="{x_fin}" y="{Y_FIN}" width="{BLOCCO_W + 16}" '
        f'height="{H_FIN}" rx="8"/>'
        f'<text class="ftag" x="{x_fin + (BLOCCO_W + 16) / 2:.0f}" y="{Y_FIN - 6}" '
        f'text-anchor="middle">in memoria veloce adesso</text></g>')

    corpo.append(f'<text class="lbs" x="30" y="{Y_V + H_V + 54}">la finestra '
                 'scorre da sinistra a destra; fuori, quei punteggi non esistono: '
                 'la matrice N×N non viene mai scritta.</text>')

    # --------------------------------------------------------------- tabella
    corpo.append(f'<text class="ttl" x="30" y="{Y_TIT}">l\'accumulatore, '
                 f'blocco per blocco</text>')
    intest = [("blocco", "", "blocco"),
              ("mt", "m̃", "max del blocco"),
              ("m", "m", "massimo corrente"),
              ("alfa", "α", "riscalatura"),
              ("l", "l", "somma corrente"),
              ("O", "O", "output accumulato")]
    for chiave, simbolo, gloss in intest:
        x = COL[chiave]
        if simbolo:
            corpo.append(f'<text class="sim" x="{x}" y="{Y_TH1}" '
                         f'text-anchor="middle">{simbolo}</text>')
        corpo.append(f'<text class="glo" x="{x}" y="{Y_TH2}" '
                     f'text-anchor="middle">{gloss}</text>')
    corpo.append(f'<line class="axc" x1="55" y1="{Y_RULE}" x2="700" y2="{Y_RULE}"/>')

    for j, p in enumerate(passi):
        y = Y_R0 + j * DY_R
        t0, _ = sosta(j, n_stati, tenuta=0.62)
        anim.append(keyframes(f"r{j}", appare(t0, resta=True)))
        cambia = j == 0 or p["m"] > passi[j - 1]["m"]
        celle = [
            f'<text class="cel1" x="{COL["blocco"]}" y="{y}" text-anchor="middle">'
            f'K{chr(0x2080 + j + 1)},V{chr(0x2080 + j + 1)}</text>',
            f'<text class="cel1" x="{COL["mt"]}" y="{y}" text-anchor="middle">'
            f'{p["mt"]:g}</text>',
            f'<text class="{"cel1 su" if cambia else "cel1"}" x="{COL["m"]}" y="{y}" '
            f'text-anchor="middle">{p["m"]:g}</text>',
            f'<text class="cel1" x="{COL["l"]}" y="{y}" text-anchor="middle">'
            f'{num(p["l"])}</text>',
            f'<text class="cel1" x="{COL["O"]}" y="{y}" text-anchor="middle">'
            f'{num(p["O"])}</text>',
        ]
        if p["alfa"] is None:
            celle.append(f'<text class="glo" x="{COL["alfa"]}" y="{y}" '
                         f'text-anchor="middle">niente ancora</text>')
        else:
            cls = "cel1 su" if p["alfa"] < 1 else "cel1 spento"
            celle.append(f'<text class="{cls}" x="{COL["alfa"]}" y="{y}" '
                         f'text-anchor="middle">{num(p["alfa"])}</text>')
        corpo.append(f'<g class="rig" style="animation:r{j} var(--d) infinite">'
                     f'{"".join(celle)}</g>')

    # ------------------------------------------------------------------ coda
    y_coda = Y_R0 + n_bloc * DY_R
    corpo.append(
        f'<text class="lbs" x="30" y="{y_coda + 4}">α = '
        f'e<tspan dy="-6" font-size="11">m vecchio − m nuovo</tspan>'
        f'<tspan dy="6"> riscala ciò che era già stato accumulato, ogni volta '
        f'che arriva un massimo più grande;</tspan></text>')
    corpo.append(
        f'<text class="lbs" x="30" y="{y_coda + 24}">se il massimo '
        f'non cambia, α = 1 e non c\'è niente da riscalare.</text>')
    corpo.append(
        f'<text class="lbs" x="30" y="{y_coda + 44}">i v qui sono scalari, così '
        f'O si legge come un numero: nel kernel sono righe di V.</text>')

    t_fin, _ = sosta(n_stati - 1, n_stati, tenuta=0.62)
    anim.append(keyframes("esi", appare(t_fin, resta=True)))
    corpo.append(
        f'<text class="esito" x="30" y="{y_coda + 78}" '
        f'style="animation:esi var(--d) infinite">alla fine O / l = {num(esatto)}: '
        f'lo stesso numero della softmax calcolata in un colpo solo.</text>')

    return Figura(
        larghezza=760, altezza=y_coda + 100,
        alt="Una riga di otto punteggi divisa in quattro blocchi da due. Una "
            "finestra scorre da sinistra a destra e in ogni istante mostra i "
            "numeri di un solo blocco: fuori dalla finestra le celle restano "
            "vuote, perché la matrice dei punteggi non viene mai scritta. Sotto, "
            "una tabella si riempie riga per riga con il massimo del blocco, il "
            "massimo corrente m, il fattore di riscalatura alfa, la somma "
            "corrente l e l'output accumulato O: quando arriva un massimo più "
            "grande, alfa scende sotto 1 e l'accumulatore viene riscalato. "
            "L'ultima riga dà O diviso l, che coincide con la softmax calcolata "
            "in un colpo solo.",
        corpo="".join(corpo),
        stile=f"""    .vuS  {{ fill:none; stroke:{TERRACOTTA}; stroke-width:1.4;
            stroke-opacity:0.35; stroke-dasharray:4 4; }}
    .vuV  {{ fill:none; stroke:{BORDER_STRONG}; stroke-width:1.4; }}
    .pieS {{ fill:{TERRACOTTA}; fill-opacity:0.18; stroke:{TERRACOTTA}; stroke-width:2; }}
    .pieV {{ fill:{OCRA}; fill-opacity:0.3; stroke:{OCRA}; stroke-width:1.8; }}
    .valS {{ font-family:{SANS}; font-size:19px; font-weight:700; fill:{INK}; }}
    .valV {{ font-family:{SANS}; font-size:14px; fill:{INK}; }}
    .bloc {{ font-family:{SANS}; font-size:13px; fill:{FG_MUTED}; }}
    .fbox {{ fill:none; stroke:{TEAL}; stroke-width:3; }}
    .ftag {{ font-family:{SANS}; font-size:12.5px; font-weight:700; fill:{TEAL}; }}
    .fin  {{ transform-box:view-box; }}
    .sim  {{ font-family:{SERIF}; font-size:16px; font-style:italic; fill:{INK}; }}
    .glo  {{ font-family:{SANS}; font-size:11px; fill:{FG_MUTED}; }}
    .cel1 {{ font-family:{SANS}; font-size:14.5px; fill:{INK}; }}
    .su   {{ fill:{TERRACOTTA}; font-weight:700; }}
    .spento {{ fill:{FG_MUTED}; }}
    .esito {{ font-family:{SANS}; font-size:14.5px; font-weight:700; fill:{TEAL}; }}""",
        animazioni=anim,
        durata=n_stati * 1.8,
        fermi=".cel, .fin, .rig, .esito",
    )
