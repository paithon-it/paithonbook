"""Una rete di Hopfield ripara un ricordo, e l'energia può soltanto scendere.

Niente qui è trascritto dal capitolo: la rete gira davvero. I pesi escono dalla
regola di Hebb sulle tre lettere di `ModelliEnergia/memoria-associativa.md`, la
corruzione usa lo stesso seme (42) e la stessa sequenza di estrazioni del codice
stampato là, e l'aggiornamento è quello asincrono, un neurone alla volta. Gli
stati intermedi che la figura mostra sono quelli veri, e `verifica` li confronta
con i numeri che il capitolo dichiara: se un giorno il testo cambia e la figura
no, la figura non si genera nemmeno.

Quello che il fermo immagine non può dire, e che qui si vede: la discesa
avviene **a scatti**, uno per neurone, ed è **monotona**. È la proprietà su cui
poggia tutto il resto (E non può salire, quindi la rete si ferma da sola in un
minimo), e in una figura ferma resta una promessa. L'assert su ΔE < 0 sta qui
per questo: una figura che mostrasse l'energia risalire sarebbe sbagliata, e
non deve nascere.

Lo stato di riposo è il punto fisso: chi non anima vede la T ricomposta e
l'intera scala dell'energia già percorsa, coi sette valori.
"""

import numpy as np

from paithon_svg import *

NOME = "hopfield-ricorda"
TITOLO = "una rete di Hopfield ripara un ricordo"

# Le tre lettere memorizzate, identiche al codice del capitolo.
LETTERE = {
    "T": ["#####",
          "..#..",
          "..#..",
          "..#..",
          "..#.."],
    "L": ["#....",
          "#....",
          "#....",
          "#....",
          "#####"],
    "X": ["#...#",
          ".#.#.",
          "..#..",
          ".#.#.",
          "#...#"],
}
SEME = 42          # lo stesso del capitolo: la corruzione è quella, non un'altra
QUANTI = 6         # 6 bit su 25 invertiti, cioè il 24%
LATO = 5

# La guardia: quello che il capitolo stampa per la T. Sono i numeri e il
# disegno che il lettore ha davanti nella pagina, non un'approssimazione.
E_CORROTTO, E_RICHIAMATO = -2.08, -11.20
CORROTTO_ATTESO = ["#.###", "..#..", "#.#.#", ".##..", ".###."]
ALFA_C = 0.138     # la capienza dichiarata nel capitolo


# --------------------------------------------------------------------------
# La rete
# --------------------------------------------------------------------------
def a_vettore(disegno):
    return np.array([1 if c == "#" else -1 for riga in disegno for c in riga])


def a_righe(s):
    griglia = np.where(s == 1, "#", ".").reshape(LATO, LATO)
    return ["".join(riga) for riga in griglia]


def discesa():
    """La rete gira: Hebb, corruzione, aggiornamento asincrono.

    Restituisce (stati, N), dove ogni stato è (vettore, energia, neurone
    appena capovolto). Il primo è quello corrotto, l'ultimo il punto fisso.
    La sequenza di estrazioni è quella del capitolo: prima `corrompi`, poi le
    permutazioni di `richiama`, dallo stesso generatore.
    """
    pattern = np.array([a_vettore(d) for d in LETTERE.values()])
    n = pattern.shape[1]
    W = (pattern.T @ pattern) / n
    np.fill_diagonal(W, 0.0)

    def energia(s):
        return float(-0.5 * s @ W @ s)

    rng = np.random.default_rng(SEME)
    s = pattern[0].copy()                       # la T
    s[rng.choice(n, size=QUANTI, replace=False)] *= -1

    stati = [(s.copy(), energia(s), None)]
    for _ in range(10):
        cambiato = False
        for i in rng.permutation(n):
            campo = W[i] @ s
            nuovo = np.sign(campo) if campo != 0 else s[i]
            if nuovo != s[i]:
                # ΔE = -2|h_i|, la formula del capitolo: la si verifica qui,
                # dove il campo locale è ancora quello di prima del ribaltamento
                atteso = stati[-1][1] - 2 * abs(float(campo))
                s[i] = nuovo
                stati.append((s.copy(), energia(s), int(i)))
                assert abs(stati[-1][1] - atteso) < 1e-9, (
                    f"neurone {i}: E passa a {stati[-1][1]:.4f}, ma "
                    f"ΔE = −2|h| ne prevede {atteso:.4f}")
                cambiato = True
        if not cambiato:
            break
    return stati, n, W, pattern


def verifica(stati, n, pattern):
    """I conti della figura coincidono con quelli che il capitolo dichiara?"""
    energie = [e for _, e, _ in stati]

    # la proprietà che rende la rete una memoria: E non può risalire, mai
    for k in range(1, len(energie)):
        assert energie[k] < energie[k - 1], (
            f"al passo {k} l'energia sale da {energie[k - 1]:.4f} a "
            f"{energie[k]:.4f}: la rete non sarebbe una memoria")

    assert a_righe(stati[0][0]) == CORROTTO_ATTESO, (
        f"lo stato corrotto è {a_righe(stati[0][0])}, il capitolo stampa "
        f"{CORROTTO_ATTESO}")
    assert abs(energie[0] - E_CORROTTO) < 5e-3, (
        f"E dello stato corrotto è {energie[0]:.2f}, il capitolo scrive {E_CORROTTO}")
    assert abs(energie[-1] - E_RICHIAMATO) < 5e-3, (
        f"E del pattern richiamato è {energie[-1]:.2f}, il capitolo scrive "
        f"{E_RICHIAMATO}")
    assert np.array_equal(stati[-1][0], pattern[0]), \
        "la rete non richiama la T: la figura mostrerebbe un minimo spurio"
    assert len(pattern) < ALFA_C * n, (
        f"{len(pattern)} ricordi su {n} neuroni sfondano la capienza "
        f"{ALFA_C}·{n} = {ALFA_C * n:.2f} che il capitolo dichiara")


# --------------------------------------------------------------------------
# La figura
# --------------------------------------------------------------------------
def numero(v: float) -> str:
    return f"{v:+.2f}".replace("+", "").replace("-", "−").replace(".", ",")


def traccia(valori, passo, transito):
    """(tempo, dichiarazione) per una successione discreta di stati.

    `transito` è la frazione di fetta usata per passare al valore nuovo: 0 per
    ciò che scatta (un neurone si capovolge, non sfuma), un pelo di più per la
    pallina, che deve *rotolare* per far vedere che sta scendendo.
    """
    tappe = [(0.0, valori[0])]
    for k in range(1, len(valori)):
        if valori[k] == valori[k - 1]:
            continue
        t0 = k * passo
        tappe += [(max(t0 - passo * 0.02, 0.01), valori[k - 1]),
                  (min(t0 + passo * transito, 99.9), valori[k])]
    tappe.append((100.0, valori[-1]))
    return tappe


def costruisci() -> Figura:
    stati, n, _W, pattern = discesa()
    verifica(stati, n, pattern)

    passi = len(stati)
    passo = 100.0 / passi
    energie = [e for _, e, _ in stati]

    # --- la griglia dei neuroni
    gx, gy, pitch, cella = 44, 88, 44, 38
    corpo, anim = [], []

    def acceso(k, i):
        return stati[k][0][i] == 1

    nomi = {}
    for i in range(n):
        x = gx + (i % LATO) * pitch
        y = gy + (i // LATO) * pitch
        colori = [TEAL if acceso(k, i) else CREAM for k in range(passi)]
        moto = ""
        if len(set(colori)) > 1:                # solo i sei che cambiano si animano
            chiave = tuple(colori)
            if chiave not in nomi:
                nomi[chiave] = f"px{len(nomi)}"
                anim.append(keyframes(nomi[chiave],
                                      [(t, f"fill:{v}") for t, v
                                       in traccia(list(chiave), passo, 0.0)]))
            moto = f' style="animation:{nomi[chiave]} var(--d) infinite"'
        corpo.append(f'<rect class="cel" x="{x}" y="{y}" width="{cella}" '
                     f'height="{cella}" rx="3" fill="{colori[-1]}"{moto}/>')

    # --- l'alone sul neurone che si sta capovolgendo: è la mossa, e senza di
    #     lui in una griglia di venticinque quadrati non si vede quale cambia
    for k in range(1, passi):
        i = stati[k][2]
        x = gx + (i % LATO) * pitch
        y = gy + (i // LATO) * pitch
        t0 = k * passo
        anim.append(keyframes(f"alo{k}", [
            (0.0, "stroke-opacity:0"), (max(t0 - passo * 0.02, 0.01), "stroke-opacity:0"),
            (t0, "stroke-opacity:1"), (min(t0 + passo * 0.62, 99.9), "stroke-opacity:0"),
            (100.0, "stroke-opacity:0")]))
        corpo.append(f'<rect class="alo" x="{x - 4}" y="{y - 4}" width="{cella + 8}" '
                     f'height="{cella + 8}" rx="5" stroke-opacity="0" '
                     f'style="animation:alo{k} var(--d) infinite"/>')

    corpo.append(f'<text class="lbs" x="{gx}" y="{gy - 16}">'
                 f'lo stato della rete: 25 neuroni, acceso o spento</text>')

    # --- la scala dell'energia
    r = Riquadro(x=376, y=gy, larg=232, alt=LATO * pitch - (pitch - cella),
                 xmin=-0.4, xmax=passi - 0.6,
                 # in alto ci sta l'etichetta del primo valore, che altrimenti
                 # finisce a cavallo della cornice
                 ymin=min(energie) - 0.7, ymax=max(energie) + 1.3)
    px = [r.sx(k) for k in range(passi)]
    py = [r.sy(e) for e in energie]

    corpo.insert(0, r.cornice())
    corpo.append('<polyline class="via" points="'
                 + " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(px, py)) + '"/>')
    for k in range(passi):
        corpo.append(f'<circle class="tap" cx="{px[k]:.1f}" cy="{py[k]:.1f}" r="3.5"/>')
        corpo.append(f'<text class="val" x="{px[k] + 9:.1f}" y="{py[k] - 7:.1f}">'
                     f'{numero(energie[k])}</text>')
        corpo.append(f'<text class="tic" x="{px[k]:.1f}" y="{r.y + r.alt + 18}" '
                     f'text-anchor="middle">{k}</text>')

    # La pallina, disegnata dov'è arrivata: l'animazione parte dall'inverso.
    # Ocra mentre rotola e teal quando si ferma, come nel paesaggio di energia
    # della stessa sezione: là l'ocra è lo stato rumoroso e il teal un ricordo.
    salti = [f"transform:translate({px[k] - px[-1]:.1f}px,{py[k] - py[-1]:.1f}px);"
             f"fill:{TEAL if k == passi - 1 else OCRA}" for k in range(passi)]
    anim.append(keyframes("rotola", traccia(salti, passo, 0.22)))
    corpo.append(f'<circle id="pallina" cx="{px[-1]:.1f}" cy="{py[-1]:.1f}" r="8"/>')

    corpo += [
        f'<text class="lbs" x="{r.x}" y="{gy - 16}">l\'energia E, che può soltanto '
        f'scendere</text>',
        f'<text class="tic" x="{r.x + r.larg / 2:.0f}" y="{r.y + r.alt + 38}" '
        f'text-anchor="middle">aggiornamenti, un neurone alla volta</text>',
    ]

    # --- che cosa sta succedendo, un cartello per stato
    for k in range(passi):
        t0 = k * passo
        if k == 0:
            tappe = [(0.0, "opacity:1"), (passo * 0.96, "opacity:1"),
                     (passo, "opacity:0"), (100.0, "opacity:0")]
        elif k == passi - 1:
            tappe = [(0.0, "opacity:0"), (t0 - passo * 0.02, "opacity:0"),
                     (t0, "opacity:1"), (100.0, "opacity:1")]
        else:
            tappe = [(0.0, "opacity:0"), (t0 - passo * 0.02, "opacity:0"),
                     (t0, "opacity:1"), (t0 + passo * 0.96, "opacity:1"),
                     (t0 + passo, "opacity:0"), (100.0, "opacity:0")]
        anim.append(keyframes(f"eti{k}", tappe))
        fermo = ";opacity:1" if k == passi - 1 else ""
        moto = f'style="animation:eti{k} var(--d) infinite{fermo}"'

        if k == 0:
            titolo = "stato corrotto"
            sotto = f"{QUANTI} pixel su {n} invertiti: la T non si legge più"
        elif k == passi - 1:
            titolo = f"aggiornamento {k}: punto fisso"
            sotto = "nessun neurone vuole più cambiare, la T è tornata"
        else:
            titolo = f"aggiornamento {k}"
            sotto = ("un neurone cambia segno, E scende di "
                     f"{numero(energie[k - 1] - energie[k])}")
        corpo += [
            f'<text class="pas" x="{gx}" y="{r.y + r.alt + 66}" {moto}>{titolo}</text>',
            f'<text class="spg" x="{gx}" y="{r.y + r.alt + 90}" {moto}>{sotto}</text>']

    # Niente formula qui dentro. Il lettore Elementare che ha collaudato questa
    # figura l'ha detto meglio di come lo direi io: la didascalia si legge, ma
    # «E(s) = −½ sᵀWs» insieme a «regola di Hebb» viene dalla scheda Superiore, e
    # una figura non si salta come si salta una scheda. La formula sta nel testo,
    # a un livello dichiarato; qui resta quel che serve a leggere il disegno.
    corpo.append(f'<text class="lbs" x="{gx}" y="{r.y + r.alt + 122}">'
                 f'la rete ha imparato tre ricordi: le lettere T, L e X</text>')

    return Figura(
        larghezza=684, altezza=r.y + r.alt + 142,
        alt="A sinistra una griglia di cinque per cinque neuroni: parte da una "
            "lettera T con sei pixel invertiti, e a ogni passo un solo neurone "
            "si capovolge finché la T è ricomposta. A destra l'energia della "
            "rete, che a ogni aggiornamento scende a scatti e non risale mai: "
            "da −2,08 dello stato corrotto a −11,20 del ricordo richiamato, in "
            "sei aggiornamenti. Arrivata in fondo, la rete si ferma da sola.",
        corpo="".join(corpo),
        stile=f"""    .cel {{ stroke:{BORDER_STRONG}; stroke-width:1.5; }}
    .alo {{ fill:none; stroke:{TERRACOTTA}; stroke-width:3; }}
    .via {{ fill:none; stroke:{BORDER_STRONG}; stroke-width:2; }}
    .tap {{ fill:{FG_MUTED}; }}
    #pallina {{ fill:{TEAL}; stroke:{CREAM}; stroke-width:1.5;
               animation:rotola var(--d) infinite; }}
    .val {{ font-family:{SANS}; font-size:12px; fill:{FG_MUTED}; }}
    .tic {{ font-family:{SANS}; font-size:12px; fill:{FG_MUTED}; }}
    .pas {{ font-family:{SANS}; font-size:17px; font-weight:700;
           fill:{TERRACOTTA}; opacity:0; }}
    .spg {{ font-family:{SANS}; font-size:14px; fill:{FG_MUTED}; opacity:0; }}""",
        animazioni=anim,
        durata=passi * 1.5,
        fermi=".cel, .alo, #pallina, .pas, .spg",
    )
