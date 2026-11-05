"""Una canzone a token e una per diffusione, sullo stesso orologio.

Il tempo e' il contenuto: sopra la canzone nasce da sinistra a destra, un pezzo
alla volta, e ogni pezzo e' nitido e definitivo appena scritto, quindi serve un
passo per ogni pezzo. Sotto la canzone intera esiste fin dal primo passo come
una macchia sfocata su tutta la durata, e a ogni passo diventa piu' nitida
dappertutto insieme.

Le due righe girano con lo stesso orologio: finiscono nello stesso istante, ma
sopra gli scatti sono PEZZI (24) e sotto sono PASSI (8), ciascuno su tutta la
canzone. Non si confrontano i lavori (un passo di diffusione costa molto piu'
di un passo con la cache), si confrontano i passi in fila, come fa il testo.

Il profilo della canzone e' un'onda inventata con un seme fisso; la sfocatura
della riga sotto e' una media mobile del profilo vero, piu' un rumore
smorzato, che si spegne a mano a mano che i passi avanzano. Lo stato di riposo
e' l'ultimo istante: tutti i pezzi scritti, tutti e otto i passi fatti.
"""

import math
import random

from paithon_svg import *

NOME = "musica-tutta-insieme"
TITOLO = "una canzone a token e una per diffusione, sullo stesso orologio"

PEZZI = 24             # i pezzi della canzone, cioe' i passi della riga a token
PASSI_DIFF = 8         # i passi della riga a diffusione
SEME = 11
CHIUSURA = 0.85        # frazione del ciclo in cui si lavora; il resto e' sosta

X0, W_PEZZO, GAP = 40, 24.0, 2.0
LARG_TOT = PEZZI * (W_PEZZO + GAP) - GAP
MEZZA = 30.0           # mezza altezza massima di una barra
Y_TOK, Y_DIF = 108.0, 272.0     # linea di mezzo delle due righe

A_PAROLE = {8: "otto", 24: "ventiquattro"}


def profilo():
    """Il profilo vero della canzone: un'onda con sezioni piu' forti e piu' piane."""
    r = random.Random(SEME)
    out = []
    for i in range(PEZZI):
        inviluppo = 0.55 + 0.35 * math.sin(i / PEZZI * 3 * math.pi + 0.6)
        out.append(max(0.12, min(1.0, inviluppo * (0.6 + 0.4 * r.random()))))
    return out


def rumore():
    r = random.Random(SEME + 1)
    return [r.uniform(-1.0, 1.0) for _ in range(PEZZI)]


def media_mobile(v, raggio):
    if raggio == 0:
        return list(v)
    n = len(v)
    return [sum(v[max(0, i - raggio):min(n, i + raggio + 1)])
            / (min(n, i + raggio + 1) - max(0, i - raggio)) for i in range(n)]


def stati_diffusione(vero, rum):
    """Lo stato dopo ogni passo s = 0..PASSI_DIFF: (altezze, opacita')."""
    stati = []
    for s in range(PASSI_DIFF + 1):
        nit = (s / PASSI_DIFF) ** 1.4              # nitidezza: 0 rumore, 1 nitido
        raggio = round(7 * (1 - s / PASSI_DIFF))
        base = media_mobile(vero, raggio)
        macchia = media_mobile(rum, max(2, raggio))
        alt = [max(0.06, min(1.0, b * (0.35 + 0.65 * nit)
                             + 0.5 * (1 - nit) * (0.5 + 0.5 * m)))
               for b, m in zip(base, macchia)]
        if s == PASSI_DIFF:
            alt = list(vero)
        stati.append((alt, 0.30 + 0.70 * nit))
    return stati


VERO = profilo()
RUM = rumore()
STATI = stati_diffusione(VERO, RUM)

# --- guardie: la figura non nasce se smette di dire il vero --------------
if len(VERO) != PEZZI:
    raise AssertionError("la riga a token non ha un passo per ogni pezzo")
if len(STATI) - 1 != PASSI_DIFF:
    raise AssertionError("la riga a diffusione non ha i passi dichiarati")
if STATI[-1][0] != VERO or STATI[-1][1] != 1.0:
    raise AssertionError("alla fine la riga a diffusione non e' nitida")
if not PASSI_DIFF < PEZZI:
    raise AssertionError("la diffusione deve usare meno passi dei pezzi")
if any(a == b for a, b in zip(STATI[0][0], VERO)):
    raise AssertionError("lo stato iniziale non e' una macchia: coincide col vero")
scarto = [sum(abs(a - b) for a, b in zip(alt, VERO)) for alt, _ in STATI]
if any(scarto[i] <= scarto[i + 1] for i in range(PASSI_DIFF)):
    raise AssertionError("la diffusione non si avvicina al vero a ogni passo")
# lo stato 0 esiste fin dall'inizio e copre TUTTA la durata, non una parte
if len(STATI[0][0]) != PEZZI:
    raise AssertionError("la macchia non copre tutta la canzone")


def tempo(k, n):
    """L'istante (in %) in cui finisce il passo k di n, sullo stesso orologio."""
    return 100.0 * CHIUSURA * k / n


def finestra(nome, t0, t1, anim):
    """Acceso fra t0 e t1 (in %), spento fuori. t1=None: acceso fino alla fine."""
    tappe = [(0.0, "opacity:1" if t0 == 0 else "opacity:0")]
    if t0 > 0:
        tappe += [(max(t0 - 0.05, 0.01), "opacity:0"), (t0, "opacity:1")]
    if t1 is not None:
        tappe += [(t1 - 0.05, "opacity:1"), (t1, "opacity:0"), (100.0, "opacity:0")]
    else:
        tappe += [(100.0, "opacity:1")]
    anim.append(keyframes(nome, tappe))


def px(i):
    return X0 + i * (W_PEZZO + GAP)


def barra(i, h, y_mezzo, cls, extra=""):
    mh = h * MEZZA
    return (f'<rect class="{cls}" x="{px(i):.1f}" y="{y_mezzo - mh:.1f}" '
            f'width="{W_PEZZO:.0f}" height="{2 * mh:.1f}" rx="3"{extra}/>')


def costruisci() -> Figura:
    corpo, anim = [], []
    scatta = "animation-timing-function:linear"

    corpo.append(f'<text class="lbl" x="{X0}" y="32">la strada a token: un pezzo '
                 f'alla volta</text>')
    corpo.append(f'<text class="lbl" x="{X0}" y="198">la diffusione: tutta la '
                 f'canzone insieme</text>')

    # --- riga a token: una casella vuota per ogni pezzo, e il pezzo scritto
    for i in range(PEZZI):
        corpo.append(f'<rect class="vuoto" x="{px(i):.1f}" y="{Y_TOK - MEZZA:.1f}" '
                     f'width="{W_PEZZO:.0f}" height="{2 * MEZZA:.0f}" rx="3"/>')
    for i in range(PEZZI):
        nome = f"p{i}"
        t = tempo(i + 1, PEZZI)
        anim.append(keyframes(nome, [(0.0, "opacity:0"), (t - 0.05, "opacity:0"),
                                     (t, "opacity:1"), (100.0, "opacity:1")]))
        corpo.append(barra(i, VERO[i], Y_TOK, "nit",
                           f' opacity="1" style="animation:{nome} var(--d) infinite;'
                           f'{scatta}"'))

    # --- riga a diffusione: una macchia per stato, la nitida e' l'ultima
    for s, (alt, op) in enumerate(STATI):
        nome = f"s{s}"
        t0 = 0.0 if s == 0 else tempo(s, PASSI_DIFF)
        t1 = None if s == PASSI_DIFF else tempo(s + 1, PASSI_DIFF)
        finestra(nome, t0, t1, anim)
        barre = "".join(barra(i, alt[i], Y_DIF, "dif") for i in range(PEZZI))
        corpo.append(f'<g opacity="{1 if s == PASSI_DIFF else 0}" '
                     f'fill-opacity="{op:.2f}" '
                     f'style="animation:{nome} var(--d) infinite;{scatta}">{barre}</g>')

    # --- i contatori dei passi, uno per riga, che scattano col loro orologio
    def contatore(y, n, prefisso):
        for k in range(n + 1):
            nome = f"{prefisso}{k}"
            t0 = 0.0 if k == 0 else tempo(k, n)
            t1 = None if k == n else tempo(k + 1, n)
            finestra(nome, t0, t1, anim)
            corpo.append(f'<text class="cnt" x="{X0 + LARG_TOT:.0f}" y="{y}" '
                         f'text-anchor="end" opacity="{1 if k == n else 0}" '
                         f'style="animation:{nome} var(--d) infinite;{scatta}">'
                         f'passo {k} di {n}</text>')
    contatore(32, PEZZI, "ct")
    contatore(198, PASSI_DIFF, "cd")

    # --- le tacche dei passi, lo stesso orologio letto due volte
    def tacche(y, n):
        larg = (LARG_TOT + GAP) / n
        for k in range(n):
            nome = f"{'tt' if n == PEZZI else 'td'}{k}"
            t = tempo(k + 1, n)
            anim.append(keyframes(nome, [(0.0, "opacity:0"), (t - 0.05, "opacity:0"),
                                         (t, "opacity:1"), (100.0, "opacity:1")]))
            corpo.append(f'<rect class="tacca" x="{X0 + k * larg:.1f}" y="{y}" '
                         f'width="{larg - GAP:.1f}" height="7" rx="2" opacity="1" '
                         f'style="animation:{nome} var(--d) infinite;{scatta}"/>')
    corpo.append(f'<text class="lbs" x="{X0}" y="{Y_TOK + MEZZA + 38:.0f}">'
                 f'{PEZZI} passi in fila</text>')
    tacche(Y_TOK + MEZZA + 12, PEZZI)
    corpo.append(f'<text class="lbs" x="{X0}" y="{Y_DIF + MEZZA + 38:.0f}">'
                 f'{PASSI_DIFF} passi, ciascuno su tutta la canzone</text>')
    tacche(Y_DIF + MEZZA + 12, PASSI_DIFF)

    # --- l'asse del tempo della canzone, uguale per le due righe
    y_asse = Y_DIF + MEZZA + 62
    corpo.append(f'<line class="ax" x1="{X0}" y1="{y_asse}" x2="{X0 + LARG_TOT:.0f}" '
                 f'y2="{y_asse}"/>')
    corpo.append(f'<text class="lbs" x="{X0}" y="{y_asse + 18}">inizio della '
                 f'canzone</text>')
    corpo.append(f'<text class="lbs" x="{X0 + LARG_TOT:.0f}" y="{y_asse + 18}" '
                 f'text-anchor="end">fine della canzone</text>')

    return Figura(
        larghezza=X0 + LARG_TOT + 40, altezza=y_asse + 34,
        alt=f"Due righe una sopra l'altra, sullo stesso asse del tempo della "
            f"canzone, divisa in {A_PAROLE[PEZZI]} pezzi. Nella riga in alto, "
            f"la strada a token, la canzone nasce da sinistra a destra un pezzo "
            f"alla volta: ogni pezzo compare nitido e definitivo, e servono "
            f"{A_PAROLE[PEZZI]} passi in fila. Nella riga in basso, la "
            f"diffusione, la canzone intera esiste fin dal primo passo come "
            f"una macchia sfocata su tutta la durata, e in {A_PAROLE[PASSI_DIFF]} "
            f"passi diventa nitida dappertutto insieme. Le due righe seguono "
            f"lo stesso orologio e finiscono nello stesso istante, ma sopra "
            f"i passi sono {A_PAROLE[PEZZI]} e sotto sono {A_PAROLE[PASSI_DIFF]}, "
            f"ciascuno su tutta la canzone. Alla fine le due righe sono "
            f"entrambe nitide.",
        corpo="".join(corpo),
        stile=f"""    .vuoto {{ fill:none; stroke:{BORDER_STRONG}; stroke-width:1; stroke-dasharray:3 3; }}
    .nit {{ fill:{TEAL}; }}
    .dif {{ fill:{TEAL}; }}
    .tacca {{ fill:{TERRACOTTA}; }}
    .cnt {{ font-family:{SANS}; font-size:13px; font-weight:700; fill:{TERRACOTTA}; }}""",
        animazioni=anim,
        durata=14.0,
        fermi=".nit, .tacca, .cnt, g[style]",
    )
