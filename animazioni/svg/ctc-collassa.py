"""CTC: la regola di collasso, e perché l'ordine dei due passi conta.

Il tempo qui è il contenuto in senso quasi letterale. La rete emette un simbolo
per ogni frame acustico, e la sequenza grezza è piena di ripetizioni e di
vuoti; la regola $\\mathcal{B}$ la riduce alla parola in due passi, e i due
passi vanno in quest'ordine: **prima** si uniscono i simboli uguali
consecutivi, **poi** si tolgono i vuoti. Una figura ferma mostra il risultato e
nasconde proprio la cosa che si deve capire, cioè che invertire i due passi dà
una parola diversa.

L'allineamento è quello della sezione `SpeechRecognition/modelli-asr.md`,
`P A A L ∅ L A` per «PALLA», e la regola **gira davvero**: `unisci`,
`togli_vuoti` e `collassa` sono dieci righe di Python, e quello che la figura
disegna è il loro output. Tre `assert` la fermano se un giorno divergesse dal
testo: il collasso deve dare PALLA, l'altro allineamento che la sezione cita
(`P P A L ∅ L A`) deve dare PALLA anche lui, e l'ordine invertito deve dare
PALA, cioè la parola sbagliata che è la ragione per cui il vuoto esiste.

Lo stato di riposo è il diagramma completo: le tre righe, il risultato e il
riquadro dell'ordine invertito. Chi non anima (stampa, PDF,
`prefers-reduced-motion`) vede l'intera regola, e l'animazione non aggiunge
disegno, aggiunge l'**ordine** in cui il disegno si costruisce.

Convenzione di colore, la stessa delle altre figure animate: teal per i dati
(l'onda, la parola che ne esce), terracotta per il meccanismo che agisce (i
passi, la fusione, la cancellazione), ocra per i frame e per il vuoto.
"""

import math

from paithon_svg import *

NOME = "ctc-collassa"
TITOLO = "la regola di collasso della CTC"

# --- l'allineamento del capitolo -------------------------------------------
VUOTO = "∅"
PI = ("P", "A", "A", "L", VUOTO, "L", "A")          # la sequenza grezza
ALTRO = ("P", "P", "A", "L", VUOTO, "L", "A")       # l'altra citata nel testo
PAROLA = "PALLA"
INVERTITA = "PALA"                                   # cosa esce a passi scambiati


# --------------------------------------------------------------------------
# La regola di collasso, in dieci righe
# --------------------------------------------------------------------------
def unisci(seq):
    """Passo 1: ogni corsa di simboli uguali consecutivi diventa un simbolo."""
    fuori = []
    for s in seq:
        if not fuori or s != fuori[-1]:
            fuori.append(s)
    return tuple(fuori)


def togli_vuoti(seq):
    """Passo 2: i vuoti spariscono."""
    return tuple(s for s in seq if s != VUOTO)


def collassa(seq):
    """La regola B: prima si uniscono i ripetuti, poi si tolgono i vuoti."""
    return togli_vuoti(unisci(seq))


def corse(seq):
    """Gli indici della sequenza, raggruppati per corsa di simboli uguali."""
    fuori = []
    for i, s in enumerate(seq):
        if fuori and seq[fuori[-1][-1]] == s:
            fuori[-1].append(i)
        else:
            fuori.append([i])
    return fuori


# --------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------
X0, PITCH = 66, 62
N = len(PI)
STRIP_Y, STRIP_H = 36, 44
BOX_W, BOX_H = 44, 34
RAW_Y, MERGE_Y, FIN_Y = 100, 186, 272
DX = 512                       # colonna delle note, a destra del diagramma
W, H = 668, 470
PAN_Y, PAN_H = 342, 104        # il riquadro dell'ordine invertito
MINI_W, MINI_PITCH = 26, 30

FADE = 2.2                     # durata della dissolvenza, in punti percentuali


def onda(n: int = 140):
    """Una forma d'onda plausibile: dà il contesto, non porta informazione."""
    val = []
    for i in range(n):
        u = i / (n - 1)
        env = 0.30 + 0.70 * math.sin(math.pi * u) ** 0.45
        v = (0.60 * math.sin(2 * math.pi * 9.5 * u)
             + 0.28 * math.sin(2 * math.pi * 23.0 * u + 1.1)
             + 0.16 * math.sin(2 * math.pi * 41.0 * u + 2.4))
        val.append(env * v)
    m = max(abs(x) for x in val)
    return [x / m for x in val]


def costruisci() -> Figura:
    unita = unisci(PI)
    finale = collassa(PI)

    assert "".join(finale) == PAROLA, \
        f"il collasso non dà più {PAROLA}: {''.join(finale)}"
    assert "".join(collassa(ALTRO)) == PAROLA, \
        "l'altro allineamento citato nella sezione non dà più la stessa parola"
    assert "".join(unisci(togli_vuoti(PI))) == INVERTITA != PAROLA, \
        "a passi invertiti non esce più la parola sbagliata che il testo cita"

    corpo, anim = [], []
    visti = set()

    def appare(t: float) -> str:
        """Chi compare all'istante t e poi resta: il riposo è l'ultimo stato."""
        nome = f"a{int(round(t * 10))}"
        if nome not in visti:
            visti.add(nome)
            anim.append(keyframes(nome, [
                (0.0, "opacity:0"), (t, "opacity:0"),
                (min(t + FADE, 100.0), "opacity:1"), (100.0, "opacity:1")]))
        return nome

    def a(dentro: str, t: float) -> str:
        return (f'<g class="ap" style="animation:{appare(t)} var(--d) infinite">'
                f'{dentro}</g>')

    def scatola(cx: float, y: float, s: str, cls: str = "",
                larg: float = BOX_W, alt: float = BOX_H) -> str:
        c = ("box " + cls).strip()
        t = ("sim " + cls).strip()
        return (f'<rect class="{c}" x="{cx - larg / 2:.0f}" y="{y:.0f}" '
                f'width="{larg:.0f}" height="{alt:.0f}" rx="5"/>'
                f'<text class="{t}" x="{cx:.0f}" y="{y + alt * 0.72:.0f}" '
                f'text-anchor="middle">{s}</text>')

    cx = [X0 + (i + 0.5) * PITCH for i in range(N)]
    strip_x1 = X0 + N * PITCH

    # ---- il segnale, tagliato in frame ------------------------------------
    mid, amp = STRIP_Y + STRIP_H / 2, 15.0
    val = onda()
    punti = " ".join(
        f"{X0 + j / (len(val) - 1) * (strip_x1 - X0):.0f},{mid - v * amp:.1f}"
        for j, v in enumerate(val))
    corpo.append(f'<polyline class="onda" points="{punti}"/>')
    for i in range(N):
        corpo.append(f'<rect class="cella" x="{X0 + i * PITCH}" y="{STRIP_Y}" '
                     f'width="{PITCH}" height="{STRIP_H}"/>')
    # Niente durate qui dentro: con 25 ms l'uno sette frame non bastano a dire
    # «palla», e un lettore che moltiplica se ne accorge. Sopra il disegno non
    # c'e' spazio per la precisazione, quindi la porta la didascalia nel testo.
    corpo.append(f'<text class="lbs" x="{X0}" y="{STRIP_Y - 12}">'
                 f'sette frame acustici (pochi, per stare in figura)</text>')

    # ---- riga 1: un simbolo per frame -------------------------------------
    t_raw = [2.0 + 2.5 * i for i in range(N)]
    for i, s in enumerate(PI):
        cls = "vuoto" if s == VUOTO else ""
        freccia = (f'<line class="giu" x1="{cx[i]:.0f}" y1="{STRIP_Y + STRIP_H + 3}" '
                   f'x2="{cx[i]:.0f}" y2="{RAW_Y - 8}"/>'
                   f'<polygon class="punta" points="'
                   f'{cx[i] - 4:.0f},{RAW_Y - 9} {cx[i] + 4:.0f},{RAW_Y - 9} '
                   f'{cx[i]:.0f},{RAW_Y - 1}"/>')
        corpo.append(a(freccia + scatola(cx[i], RAW_Y, s, cls), t_raw[i]))
    corpo.append(a(f'<text class="nota" x="{DX}" y="114">la sequenza emessa,</text>'
                   f'<text class="nota" x="{DX}" y="130">un simbolo per frame</text>',
                   t_raw[0]))

    # ---- passo 1: le corse di simboli uguali diventano una sola ----------
    gruppi = corse(PI)
    mx = [sum(cx[i] for i in g) / len(g) for g in gruppi]
    t_mer = [25.0 + 2.2 * k for k in range(len(gruppi))]
    for k, g in enumerate(gruppi):
        cls_conn = "fonde" if len(g) > 1 else ""
        righe = "".join(
            f'<line class="conn {cls_conn}" x1="{cx[i]:.0f}" y1="{RAW_Y + BOX_H}" '
            f'x2="{mx[k]:.0f}" y2="{MERGE_Y}"/>' for i in g)
        cls = "vuoto" if unita[k] == VUOTO else ("fusa" if len(g) > 1 else "")
        corpo.append(a(righe + scatola(mx[k], MERGE_Y, unita[k], cls), t_mer[k]))
    corpo.append(a(f'<text class="passo" x="{DX}" y="150">passo 1</text>'
                   f'<text class="nota" x="{DX}" y="167">unisci i simboli</text>'
                   f'<text class="nota" x="{DX}" y="183">uguali consecutivi</text>',
                   22.0))

    # ---- passo 2: i vuoti spariscono --------------------------------------
    corpo.append(a(f'<text class="passo" x="{DX}" y="236">passo 2</text>'
                   f'<text class="nota" x="{DX}" y="253">togli i vuoti {VUOTO}</text>',
                   42.0))
    k_vuoto = unita.index(VUOTO)
    xv, yv = mx[k_vuoto], MERGE_Y
    corpo.append(a(f'<line class="croce" x1="{xv - 15:.0f}" y1="{yv + 5}" '
                   f'x2="{xv + 15:.0f}" y2="{yv + BOX_H - 5}"/>'
                   f'<line class="croce" x1="{xv + 15:.0f}" y1="{yv + 5}" '
                   f'x2="{xv - 15:.0f}" y2="{yv + BOX_H - 5}"/>', 45.0))

    resta = [k for k, s in enumerate(unita) if s != VUOTO]
    assert tuple(unita[k] for k in resta) == finale, "le due strade non coincidono"
    t_fin = [48.0 + 2.2 * j for j in range(len(resta))]
    for j, k in enumerate(resta):
        conn = (f'<line class="conn" x1="{mx[k]:.0f}" y1="{MERGE_Y + BOX_H}" '
                f'x2="{mx[k]:.0f}" y2="{FIN_Y}"/>')
        corpo.append(a(conn + scatola(mx[k], FIN_Y, unita[k], "tenuta"), t_fin[j]))

    # ---- la parola che ne esce --------------------------------------------
    parola = "".join(finale)
    corpo.append(a(f'<line class="verso" x1="{mx[resta[-1]] + BOX_W / 2 + 6:.0f}" '
                   f'y1="{FIN_Y + BOX_H / 2:.0f}" x2="{DX - 14}" '
                   f'y2="{FIN_Y + BOX_H / 2:.0f}"/>'
                   f'<polygon class="pverso" points="'
                   f'{DX - 13},{FIN_Y + BOX_H / 2 - 4:.0f} '
                   f'{DX - 13},{FIN_Y + BOX_H / 2 + 4:.0f} '
                   f'{DX - 3},{FIN_Y + BOX_H / 2:.0f}"/>'
                   f'<rect class="esito" x="{DX}" y="{FIN_Y}" width="128" '
                   f'height="{BOX_H}" rx="7"/>'
                   f'<text class="parola" x="{DX + 64}" y="{FIN_Y + 25}" '
                   f'text-anchor="middle">{parola}</text>', 62.0))

    # ---- e se l'ordine fosse l'altro? -------------------------------------
    senza = togli_vuoti(PI)
    sbagliata = unisci(senza)
    corpo.append(a(f'<rect class="pan" x="{X0}" y="{PAN_Y}" '
                   f'width="{DX + 128 - X0}" height="{PAN_H}" rx="8"/>'
                   f'<text class="passo" x="{X0 + 20}" y="{PAN_Y + 26}">'
                   f'e se si invertisse l\'ordine? prima togliere i vuoti, '
                   f'poi unire i ripetuti</text>', 70.0))

    y_mini = PAN_Y + 44
    xs = X0 + 34
    pezzi = "".join(scatola(xs + i * MINI_PITCH, y_mini, s, "mini",
                            larg=MINI_W, alt=MINI_W)
                    for i, s in enumerate(senza))
    corpo.append(a(pezzi, 73.0))

    x_fr = xs + len(senza) * MINI_PITCH + 2
    corpo.append(a(f'<line class="verso" x1="{x_fr:.0f}" y1="{y_mini + 13}" '
                   f'x2="{x_fr + 22:.0f}" y2="{y_mini + 13}"/>'
                   f'<polygon class="pverso" points="'
                   f'{x_fr + 23:.0f},{y_mini + 9} {x_fr + 23:.0f},{y_mini + 17} '
                   f'{x_fr + 33:.0f},{y_mini + 13}"/>', 76.0)),

    xs2 = x_fr + 48
    pezzi2 = "".join(scatola(xs2 + i * MINI_PITCH, y_mini, s, "mini errata",
                             larg=MINI_W, alt=MINI_W)
                     for i, s in enumerate(sbagliata))
    x_testo = xs2 + len(sbagliata) * MINI_PITCH + 6
    corpo.append(a(pezzi2 + f'<text class="errata" x="{x_testo:.0f}" '
                   f'y="{y_mini + 18}">{"".join(sbagliata)}, non {parola}</text>',
                   79.0))
    corpo.append(a(f'<text class="nota" x="{X0 + 20}" y="{PAN_Y + PAN_H - 14}">'
                   f'senza il vuoto in mezzo, il passo 1 fonde le due L: '
                   f'il {VUOTO} esiste per impedirlo</text>',
                   82.0))

    return Figura(
        larghezza=W, altezza=H,
        alt="Sette frame acustici con il simbolo che la rete emette per "
            "ciascuno, P A A L vuoto L A. Nel primo passo le due A consecutive "
            "si fondono in una sola, mentre le due L restano separate dal "
            "simbolo vuoto; nel secondo passo il vuoto viene cancellato e "
            "resta la parola PALLA. In basso, invertendo i due passi, esce "
            "invece PALA.",
        corpo="".join(corpo),
        stile=f"""    .onda  {{ fill:none; stroke:{TEAL}; stroke-width:1.6;
             stroke-linejoin:round; }}
    .cella {{ fill:none; stroke:{OCRA}; stroke-width:1.6; }}
    .giu   {{ stroke:{BORDER_STRONG}; stroke-width:1.4; }}
    .punta {{ fill:{BORDER_STRONG}; }}
    rect.box {{ fill:none; stroke:{BORDER_STRONG}; stroke-width:1.8; }}
    rect.vuoto {{ stroke:{OCRA}; }}
    rect.fusa, rect.tenuta {{ stroke:{TEAL}; stroke-width:2.4; }}
    rect.errata {{ stroke:{TERRACOTTA}; stroke-width:2.2; }}
    text.sim {{ font-family:{SANS}; font-size:19px; font-weight:600;
             fill:{INK}; }}
    text.mini {{ font-size:14px; font-weight:500; }}
    text.vuoto {{ fill:{FG_MUTED}; }}
    text.fusa, text.tenuta {{ fill:{TEAL}; }}
    text.errata {{ fill:{TERRACOTTA}; }}
    .conn  {{ stroke:{BORDER_STRONG}; stroke-width:1.6; }}
    .fonde {{ stroke:{TERRACOTTA}; stroke-width:2.4; }}
    .croce {{ stroke:{TERRACOTTA}; stroke-width:2.6; stroke-linecap:round; }}
    .verso {{ stroke:{TERRACOTTA}; stroke-width:2; }}
    .pverso {{ fill:{TERRACOTTA}; }}
    .esito {{ fill:none; stroke:{TEAL}; stroke-width:2.6; }}
    .parola {{ font-family:{SANS}; font-size:22px; font-weight:700;
             fill:{TEAL}; letter-spacing:3px; }}
    .passo {{ font-family:{SANS}; font-size:13.5px; font-weight:700;
             fill:{TERRACOTTA}; }}
    .nota  {{ font-family:{SANS}; font-size:12.5px; fill:{FG_MUTED}; }}
    .errata {{ font-family:{SANS}; font-size:14px; font-weight:600;
             fill:{TERRACOTTA}; }}
    .pan   {{ fill:none; stroke:{BORDER_STRONG}; stroke-width:1.4;
             stroke-dasharray:5 4; }}""",
        animazioni=anim,
        durata=13.0,
        fermi=".ap",
    )
