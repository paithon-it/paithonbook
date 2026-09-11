"""Il boosting che si somma: la scaletta che rincorre la curva, un albero per volta.

Il tempo qui e' il contenuto, e non per modo di dire: il boosting *e'* un
ordine. Ogni albero non guarda i dati, guarda quello che i precedenti hanno
lasciato indietro, e la somma delle correzioni assomiglia alla curva sempre di
piu'. Su una figura ferma resterebbe una scaletta accanto a una curva, e non si
vedrebbe che ogni gradino e' nato per riparare un pezzo, che e' tutto quello
che c'e' da capire.

I numeri non sono disegnati a mano: `addestra()` esegue davvero il gradient
boosting con la loss quadratica, alberi profondi uno (i *ceppi*) e il learning
rate di `NU`, e restituisce il modello dopo ogni giro e l'errore tipico che
resta. Da li' escono anche i due numeri della didascalia.

Lo stato di riposo e' l'ultimo giro: la scaletta finita, i residui corti, il
conteggio dei giri e l'errore finale.
"""

import numpy as np

from paithon_svg import *

NOME = "boosting-si-somma"
TITOLO = "Il boosting si somma, un albero per volta"

GIRI = 10           # quanti ceppi si sommano
NU = 0.8            # learning rate: quanta parte di ogni correzione si prende
N = 28              # punti del campione


# --------------------------------------------------------------------------
# Il boosting, eseguito per davvero: da qui escono tutti i numeri
# --------------------------------------------------------------------------
def dati():
    """Il campione: una curva liscia, campionata a passo fisso.

    Niente sorteggi: la figura deve uscire identica su qualunque macchina,
    quindi il campione e' deterministico e la scaletta pure.
    """
    x = np.linspace(0.0, 1.0, N)
    y = np.sin(2 * np.pi * x) + 0.45 * x
    return x, y


def ceppo(x, r):
    """Il ceppo che meglio approssima i residui `r`: soglia e due costanti.

    E' un albero profondo uno, cioe' una domanda sola: la soglia si cerca fra i
    punti medi di due valori consecutivi, e per ciascuna si prende la coppia di
    medie, che e' la costante che minimizza lo scarto quadratico in ogni parte.
    """
    ordine = np.argsort(x)
    xs, rs = x[ordine], r[ordine]
    soglie = (xs[:-1] + xs[1:]) / 2
    migliore, sse_min = None, np.inf
    for s in soglie:
        sin_, des = rs[xs < s], rs[xs >= s]
        if len(sin_) == 0 or len(des) == 0:
            continue
        sse = ((sin_ - sin_.mean()) ** 2).sum() + ((des - des.mean()) ** 2).sum()
        if sse < sse_min:
            sse_min, migliore = sse, (s, sin_.mean(), des.mean())
    return migliore


def addestra():
    """Gradient boosting con loss quadratica: il modello dopo ogni giro.

    Con la loss quadratica lo pseudo-residuo e' lo scarto che resta, quindi
    ogni ceppo si addestra proprio su quello che i precedenti hanno sbagliato.
    """
    x, y = dati()
    f = np.full_like(y, y.mean())          # $F_0$: la costante che minimizza
    stati = [(f.copy(), float(np.sqrt(((y - f) ** 2).mean())))]
    tagli = []
    for _ in range(GIRI):
        r = y - f
        s, a_sin, a_des = ceppo(x, r)
        f = f + NU * np.where(x < s, a_sin, a_des)
        tagli.append(float(s))
        stati.append((f.copy(), float(np.sqrt(((y - f) ** 2).mean()))))
    return x, y, stati, tagli


def verifica(x, y, stati, tagli) -> None:
    """La figura promette una rincorsa: c'e' davvero, ed e' quella giusta?"""
    errori = [e for _, e in stati]
    assert len(stati) == GIRI + 1, \
        f"servono {GIRI} giri piu' la costante di partenza, ce ne sono {len(stati)}"
    assert all(b <= a + 1e-12 for a, b in zip(errori, errori[1:])), (
        "l'errore deve calare a ogni giro, o la figura racconta una rincorsa "
        f"che non c'e': {[round(e, 3) for e in errori]}")
    # La didascalia e l'`:alt:` promettono due numeri, l'errore di partenza e
    # quello finale, e la parola «un quarto». Un collaudo si scrive leggendo la
    # didascalia, non piu' debole di quella.
    assert round(errori[0], 2) == 0.60 and round(errori[-1], 2) == 0.14, (
        f"la didascalia dice da 0,60 a 0,14: qui sono {errori[0]:.2f} e "
        f"{errori[-1]:.2f}")
    assert errori[-1] < errori[0] / 4, (
        "la didascalia dice che l'errore scende sotto un quarto: qui va da "
        f"{errori[0]:.3f} a {errori[-1]:.3f}")
    assert len(set(round(s, 6) for s in tagli)) > 1, (
        "l'`:alt:` dice che i ceppi tagliano in punti diversi: qui tagliano "
        "sempre nello stesso")


# --------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------
LARG, ALT = 760, 360
RIQ = Riquadro(x=60, y=30, larg=660, alt=262,
               xmin=-0.045, xmax=1.045, ymin=-0.92, ymax=1.55)

INIZIO, FINE = 6.0, 88.0     # la coda ferma serve a leggere la conclusione


def px(v: float) -> float:
    return RIQ.x + (v - RIQ.xmin) / (RIQ.xmax - RIQ.xmin) * RIQ.larg


def py(v: float) -> float:
    return RIQ.y + RIQ.alt - (v - RIQ.ymin) / (RIQ.ymax - RIQ.ymin) * RIQ.alt


def scaletta(x, f) -> str:
    """Il modello corrente come spezzata a gradini, un tratto per punto."""
    ordine = np.argsort(x)
    xs, fs = x[ordine], f[ordine]
    bordi = np.concatenate([[RIQ.xmin], (xs[:-1] + xs[1:]) / 2, [RIQ.xmax]])
    d = []
    for i, v in enumerate(fs):
        d.append(f"{'M' if i == 0 else 'L'}{px(bordi[i]):.1f},{py(v):.1f}")
        d.append(f"L{px(bordi[i + 1]):.1f},{py(v):.1f}")
    return " ".join(d)


def costruisci() -> Figura:
    x, y, stati, tagli = addestra()
    verifica(x, y, stati, tagli)

    corpo, anim = [], []
    n = len(stati)
    istante = [INIZIO + (FINE - INIZIO) * k / (n - 1) for k in range(n)]

    def sosta_giro(k: int) -> list[tuple[float, str]]:
        """Visibile solo durante il giro `k`; l'ultimo resta, ed e' il riposo."""
        t = istante[k]
        if k == n - 1:
            return [(0.0, "opacity:0"), (max(t - 1.0, 0.0), "opacity:0"),
                    (t, "opacity:1"), (100.0, "opacity:1")]
        dopo = istante[k + 1]
        return [(0.0, "opacity:0"), (max(t - 1.0, 0.0), "opacity:0"),
                (t, "opacity:1"), (dopo - 1.0, "opacity:1"),
                (dopo, "opacity:0"), (100.0, "opacity:0")]

    # --- la cornice e i due assi -------------------------------------------
    corpo.append(f'<line class="ax" x1="{RIQ.x}" y1="{py(0):.1f}" '
                 f'x2="{RIQ.x + RIQ.larg}" y2="{py(0):.1f}"/>')

    # --- i punti da approssimare, sempre visibili --------------------------
    for xv, yv in zip(x, y):
        corpo.append(f'<circle class="dato" cx="{px(xv):.1f}" cy="{py(yv):.1f}" '
                     f'r="4"/>')

    # --- una scaletta e i suoi residui per ogni giro -----------------------
    for k, (f, err) in enumerate(stati):
        nome = f"giro{k}"
        anim.append(keyframes(nome, sosta_giro(k)))
        dentro = [f'<path class="mod" d="{scaletta(x, f)}"/>']
        for xv, yv, fv in zip(x, y, f):
            dentro.append(f'<line class="res" x1="{px(xv):.1f}" '
                          f'y1="{py(yv):.1f}" x2="{px(xv):.1f}" '
                          f'y2="{py(fv):.1f}"/>')
        etichetta = ("il modello di partenza: la media, e basta" if k == 0
                     else f"dopo {k} alber{'o' if k == 1 else 'i'}")
        dentro.append(f'<text class="lbl" x="{RIQ.x + 8}" y="{RIQ.y + 20}">'
                      f'{etichetta}</text>')
        dentro.append(f'<text class="err" x="{RIQ.x + RIQ.larg - 8}" '
                      f'y="{RIQ.y + 20}" text-anchor="end">'
                      f'errore tipico {err:.2f}</text>'.replace(".", ","))
        # l'ultimo giro non porta l'`opacity:0` in linea: senza animazione
        # (stampa, PDF, `prefers-reduced-motion`) e' l'unico che resta acceso,
        # ed e' lo stato di riposo che la figura deve mostrare
        riposo = "" if k == n - 1 else "opacity:0;"
        corpo.append(f'<g style="{riposo}animation:{nome} var(--d) '
                     f'linear infinite">{"".join(dentro)}</g>')

    # --- la legenda in basso ----------------------------------------------
    y_leg = ALT - 22
    corpo.append(f'<circle class="dato" cx="{RIQ.x + 6}" cy="{y_leg - 4}" r="4"/>'
                 f'<text class="lbs" x="{RIQ.x + 20}" y="{y_leg}">'
                 f'quello che bisogna indovinare</text>')
    corpo.append(f'<line class="mod" x1="{RIQ.x + 218}" y1="{y_leg - 4}" '
                 f'x2="{RIQ.x + 250}" y2="{y_leg - 4}"/>'
                 f'<text class="lbs" x="{RIQ.x + 260}" y="{y_leg}">'
                 f'la somma degli alberi finora</text>')
    corpo.append(f'<line class="res" x1="{RIQ.x + 540}" y1="{y_leg - 12}" '
                 f'x2="{RIQ.x + 540}" y2="{y_leg + 2}"/>'
                 f'<text class="lbs" x="{RIQ.x + 552}" y="{y_leg}">'
                 f'quello che resta</text>')

    return Figura(
        larghezza=LARG, altezza=ALT,
        alt="Ventotto punti disposti lungo una curva che sale, scende e "
            "risale. Sopra di essi una linea a gradini, il modello. Al primo "
            "fotogramma la linea è piatta, all'altezza della media dei punti, e "
            "da ciascun punto scende o sale un trattino verticale che dice "
            "quanto il modello lo manca: l'errore tipico segnato in alto a "
            "destra vale 0,60. A ogni fotogramma si aggiunge un albero, la "
            "linea guadagna un gradino in un punto diverso e i trattini si "
            "accorciano. Dopo dieci alberi la linea a gradini segue la curva "
            "dei punti e l'errore tipico è sceso a 0,14.",
        corpo="".join(corpo),
        stile=f"""    .dato {{ fill:{TEAL}; }}
    .mod  {{ stroke:{TERRACOTTA}; stroke-width:2.5; fill:none;
             stroke-linejoin:round; }}
    .res  {{ stroke:{BORDER_STRONG}; stroke-width:1.5; }}
    .err  {{ font-family:{SANS}; font-size:14px; font-weight:600;
             fill:{TERRACOTTA}; }}""",
        animazioni=anim,
        durata=13.0,
        fermi="g",
    )
