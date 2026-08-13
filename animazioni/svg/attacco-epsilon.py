"""L'attacco avversario che cresce: quanto poco basta a ribaltare la risposta.

Il fatto sorprendente non è che si possa ingannare un classificatore, è la
**taglia** della spinta che serve: al crescere di ρ le trenta caratteristiche
si spostano di un soffio ciascuna, la riga della perturbazione resta minuscola
accanto a quella dell'ingresso, e a un certo punto la probabilità passa sotto
la metà e la classificazione si ribalta. Una figura ferma mostra il prima e il
dopo, e perde esattamente la soglia, che è il punto.

Niente qui è trascritto dal capitolo: l'esperimento gira davvero, ed è **lo
stesso** codice di `AIResponsabile/privacy-e-robustezza.md` (stesso seme, stesse
dimensioni, stessa scelta dell'esempio per criterio e non per indice). Il
gradiente della cross-entropia rispetto all'input è $(\\hat{y}-y)\\,w$, e FGSM ne
prende il segno. Le guardie in `verifica()` confrontano i risultati con i
numeri che il capitolo stampa: se un giorno il testo cambia e la figura no, la
figura non si genera nemmeno.

Perché non una foto. La palette è di cinque colori e una griglia di pixel
perturbati qui sarebbe illeggibile (e fuori stile): l'ingresso del capitolo,
del resto, non è un'immagine ma trenta numeri. Si disegnano quelli, una barra
per caratteristica, e la perturbazione come una seconda riga **alla stessa
scala**: tutte le spinte hanno la stessa ampiezza ρ, perché è ciò che fa il
segno del gradiente, e si vedono minuscole accanto alle barre. Il ribaltamento
sta nel terzo pannello, dove la probabilità scende e attraversa la soglia.

Lo stato di riposo è l'ultimo passo, ρ = 0,15: chi non anima vede la
perturbazione al massimo, la curva intera, il punto in terracotta sotto la
soglia e il valore di ρ a cui si è ribaltata.
"""

import numpy as np

from paithon_svg import *

NOME = "attacco-epsilon"
TITOLO = "l'attacco avversario al crescere di ρ"

# --- l'esperimento del capitolo, riga per riga -----------------------------
D, N = 30, 500
SEME = 0
ITERAZIONI, ETA = 3000, 0.2
RHO_MAX = 0.15          # il budget di perturbazione del capitolo
N_STATI = 7             # ρ = 0, 0,025, ... 0,15

# Le guardie: i numeri che il lettore ha davanti nella pagina.
I_ATTESO = 1            # l'esempio che il criterio sceglie
P0_ATTESO, P1_ATTESO = 0.890, 0.190
NORMA_SPINTA, NORMA_INGRESSO = 0.82, 6.00
CALO_LOGIT = 3.54       # ρ ‖w‖₁, il calo del logit dichiarato in Superiore

# --- geometria -------------------------------------------------------------
X0, LARG_RIGA = 60.0, 600.0
PASSO = LARG_RIGA / D
LARG_BARRA = 9.0
SCALA = 32.0            # px per unità: la stessa per l'ingresso e per la spinta
Y_ING = 118.0           # linea di base delle caratteristiche
Y_SPI = 240.0           # linea di base della perturbazione


def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-z))


def esperimento():
    """Addestra, sceglie l'esempio e attacca. Restituisce tutto il necessario."""
    rng = np.random.default_rng(SEME)

    w_vero = rng.normal(size=D)
    X = rng.normal(size=(N, D))
    y = (rng.random(N) < sigmoide(X @ w_vero)).astype(float)

    w, b = np.zeros(D), 0.0
    for _ in range(ITERAZIONI):
        p = sigmoide(X @ w + b)
        w -= ETA * (X.T @ (p - y) / N)
        b -= ETA * np.mean(p - y)

    # l'esempio non è scelto a mano: azzeccato e con fiducia fra 0,85 e 0,95
    p_tutti = sigmoide(X @ w + b)
    azzeccati = (p_tutti > 0.5) == (y == 1)
    fiducia = np.maximum(p_tutti, 1 - p_tutti)
    i = int(np.flatnonzero(azzeccati & (fiducia > 0.85) & (fiducia < 0.95))[0])

    x, y_vero = X[i].copy(), float(y[i])
    p0 = float(sigmoide(x @ w + b))

    # FGSM: un passo lungo il segno del gradiente della loss rispetto a x
    segni = np.sign((p0 - y_vero) * w)
    rhos = [k * RHO_MAX / (N_STATI - 1) for k in range(N_STATI)]
    p_rho = [float(sigmoide((x + r * segni) @ w + b)) for r in rhos]

    # il logit è lineare in ρ (cala di ρ‖w‖₁), quindi la soglia si risolve
    logit0 = float(x @ w + b)
    norma1 = float(np.abs(w).sum())
    rho_star = logit0 / norma1

    return dict(x=x, segni=segni, w=w, b=float(b), y_vero=y_vero, i=i,
                p0=p0, rhos=rhos, p_rho=p_rho, rho_star=rho_star, norma1=norma1,
                spinta=float(np.linalg.norm(RHO_MAX * segni)),
                ingresso=float(np.linalg.norm(x)))


def verifica(e):
    """La figura non nasce se l'attacco non fa quello che il capitolo dice."""
    if e["i"] != I_ATTESO:
        raise AssertionError(f"il criterio sceglie l'esempio {e['i']}, "
                             f"il capitolo stampa {I_ATTESO}")
    if e["y_vero"] != 1.0:
        raise AssertionError("l'esempio scelto non ha etichetta 1")
    # a ρ = 0 la classificazione è quella giusta
    if not (e["p_rho"][0] > 0.5 and e["p_rho"][0] == e["p0"]):
        raise AssertionError("a ρ = 0 il modello non classifica correttamente")
    # oltre la soglia è ribaltata
    if not e["p_rho"][-1] < 0.5:
        raise AssertionError(f"a ρ = {RHO_MAX} la classificazione non si ribalta "
                             f"(p = {e['p_rho'][-1]:.3f})")
    if not 0 < e["rho_star"] < RHO_MAX:
        raise AssertionError(f"la soglia ρ* = {e['rho_star']:.4f} cade fuori scala")
    if any(a <= b for a, b in zip(e["p_rho"], e["p_rho"][1:])):
        raise AssertionError("la probabilità non scende in modo monotono")
    # e i numeri stampati nel capitolo sono questi
    coppie = [(round(e["p0"], 3), P0_ATTESO), (round(e["p_rho"][-1], 3), P1_ATTESO),
              (round(e["spinta"], 2), NORMA_SPINTA),
              (round(e["ingresso"], 2), NORMA_INGRESSO),
              (round(RHO_MAX * e["norma1"], 2), CALO_LOGIT)]
    for ottenuto, atteso in coppie:
        if ottenuto != atteso:
            raise AssertionError(f"la figura calcola {ottenuto}, il capitolo "
                                 f"dichiara {atteso}")


def virgola(v, cifre=3):
    return f"{v:.{cifre}f}".replace(".", ",")


def costruisci() -> Figura:
    e = esperimento()
    verifica(e)

    x, segni = e["x"], e["segni"]
    rhos, p_rho = e["rhos"], e["p_rho"]
    n = N_STATI
    corpo, anim = [], []

    def centro(k):
        return X0 + PASSO / 2 + k * PASSO

    # ---- riga 1: l'ingresso, una barra per caratteristica ----------------
    corpo.append(f'<text class="lbs" x="{X0:.0f}" y="30">l\'esempio scelto, '
                 f'con le sue trenta caratteristiche</text>')
    corpo.append(f'<line class="asse" x1="{X0:.0f}" y1="{Y_ING:.0f}" '
                 f'x2="{X0 + LARG_RIGA:.0f}" y2="{Y_ING:.0f}"/>')
    for k, v in enumerate(x):
        h = abs(float(v)) * SCALA
        y = Y_ING - h if v > 0 else Y_ING
        corpo.append(f'<rect class="car" x="{centro(k) - LARG_BARRA / 2:.1f}" '
                     f'y="{y:.1f}" width="{LARG_BARRA:.0f}" height="{h:.1f}"/>')

    # ---- riga 2: la perturbazione, tutte le spinte della stessa ampiezza --
    corpo.append(f'<text class="lbs" x="{X0:.0f}" y="210">la spinta: ogni '
                 f'caratteristica si sposta di ρ, nel verso che danneggia il '
                 f'modello</text>')
    corpo.append(f'<line class="asse" x1="{X0:.0f}" y1="{Y_SPI:.0f}" '
                 f'x2="{X0 + LARG_RIGA:.0f}" y2="{Y_SPI:.0f}"/>')
    h_sp = RHO_MAX * SCALA
    for k, s in enumerate(segni):
        su = s > 0
        y = Y_SPI - h_sp if su else Y_SPI
        corpo.append(f'<rect class="sp {"su" if su else "giu"}" '
                     f'x="{centro(k) - LARG_BARRA / 2:.1f}" y="{y:.1f}" '
                     f'width="{LARG_BARRA:.0f}" height="{h_sp:.1f}"/>')

    # una sola animazione per tutte e trenta: cresce con ρ, e finisce intera
    tappe = []
    for k in range(n):
        t0, t1 = sosta(k, n)
        d = f"transform:scaleY({rhos[k] / RHO_MAX:.3f})"
        tappe += [(t0, d), (t1, d)]
    tappe.append((100.0, "transform:scaleY(1)"))
    anim.append(keyframes("cresce", tappe))

    corpo.append(f'<text class="lbs" x="{X0:.0f}" y="266">alla stessa scala '
                 f'della riga sopra: in tutto {virgola(e["spinta"], 2)} di spinta, '
                 f'contro {virgola(e["ingresso"], 2)} dell\'ingresso</text>')

    # ---- riga 3: la probabilità, e la soglia dove si ribalta -------------
    r = Riquadro(x=X0, y=312, larg=520, alt=148,
                 xmin=-0.007, xmax=0.163, ymin=0.0, ymax=1.0)
    corpo.append(f'<text class="lbs" x="{X0:.0f}" y="302">p(classe 1): la '
                 f'probabilità che il modello dà alla classe giusta</text>')
    corpo.append(r.cornice())

    # la soglia di decisione
    corpo.append(f'<line class="mezza" x1="{r.x:.0f}" y1="{r.sy(0.5):.1f}" '
                 f'x2="{r.x + r.larg:.0f}" y2="{r.sy(0.5):.1f}"/>')
    corpo.append(f'<text class="lbs" x="{r.x + r.larg + 8:.0f}" '
                 f'y="{r.sy(0.5) + 5:.1f}">soglia 0,5</text>')

    # la curva vera, campionata fitta; i punti di sosta cadono sui campioni
    campioni = 60
    pts = []
    for j in range(campioni + 1):
        rho = j * RHO_MAX / campioni
        p = float(sigmoide((x + rho * segni) @ e["w"] + e["b"]))
        pts.append((r.sx(rho), r.sy(p)))
    lung = [0.0]
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        lung.append(lung[-1] + math.hypot(x2 - x1, y2 - y1))
    totale = lung[-1]
    via = "M" + " L".join(f"{px:.1f},{py:.1f}" for px, py in pts)
    corpo.append(f'<path id="curva" d="{via}"/>')

    tappe = []
    for k in range(n):
        t0, t1 = sosta(k, n)
        d = f"stroke-dashoffset:{totale - lung[k * campioni // (n - 1)]:.1f}"
        tappe += [(t0, d), (t1, d)]
    tappe.append((100.0, "stroke-dashoffset:0"))
    anim.append(keyframes("disegna", tappe))

    # il momento esatto del ribaltamento, sulla timeline
    k_dopo = next(k for k, p in enumerate(p_rho) if p < 0.5)
    frazione = (e["rho_star"] - rhos[k_dopo - 1]) / (rhos[k_dopo] - rhos[k_dopo - 1])
    t_giro = sosta(k_dopo - 1, n)[1] + frazione * (sosta(k_dopo, n)[0]
                                                   - sosta(k_dopo - 1, n)[1])

    # il segno della soglia compare quando la soglia viene attraversata
    xs = r.sx(e["rho_star"])
    corpo.append(
        f'<g id="soglia">'
        f'<rect class="oltre" x="{xs:.1f}" y="{r.y:.0f}" '
        f'width="{r.x + r.larg - xs:.1f}" height="{r.alt:.0f}"/>'
        f'<line class="rib" x1="{xs:.1f}" y1="{r.y:.0f}" x2="{xs:.1f}" '
        f'y2="{r.y + r.alt:.0f}"/>'
        f'<text class="rib-t" x="{xs + 8:.1f}" y="{r.y + 20:.0f}">qui si '
        f'ribalta: ρ = {virgola(e["rho_star"])}</text></g>')
    anim.append(keyframes("svela", [
        (0.0, "opacity:0"), (max(t_giro - 0.01, 0.01), "opacity:0"),
        (t_giro, "opacity:1"), (100.0, "opacity:1")]))

    # il punto che percorre la curva: a riposo è l'ultimo, in terracotta
    px_f, py_f = r.sx(rhos[-1]), r.sy(p_rho[-1])
    corpo.append(f'<circle id="pallino" cx="{px_f:.1f}" cy="{py_f:.1f}" r="6.5"/>')
    tappe = []
    for k in range(n):
        t0, t1 = sosta(k, n)
        d = (f"transform:translate({r.sx(rhos[k]) - px_f:.1f}px,"
             f"{r.sy(p_rho[k]) - py_f:.1f}px)")
        tappe += [(t0, d), (t1, d)]
    tappe.append((100.0, "transform:translate(0px,0px)"))
    anim.append(keyframes("corri", tappe))
    anim.append(keyframes("vira", [
        (0.0, f"fill:{TEAL}"), (max(t_giro - 0.01, 0.01), f"fill:{TEAL}"),
        (t_giro, f"fill:{TERRACOTTA}"), (100.0, f"fill:{TERRACOTTA}")]))

    # tacche e nomi degli assi
    for v in (0.0, 0.05, 0.10, 0.15):
        corpo.append(f'<line class="asse" x1="{r.sx(v):.1f}" y1="{r.y + r.alt:.0f}" '
                     f'x2="{r.sx(v):.1f}" y2="{r.y + r.alt + 6:.0f}"/>')
        eti = "0" if v == 0 else virgola(v, 2)
        corpo.append(f'<text class="lbs" x="{r.sx(v):.1f}" y="{r.y + r.alt + 22:.0f}" '
                     f'text-anchor="middle">{eti}</text>')
    corpo.append(f'<text class="lbs" x="{r.x + r.larg + 8:.0f}" '
                 f'y="{r.y + r.alt + 22:.0f}">ρ</text>')
    for v, eti in ((1.0, "1"), (0.5, "0,5"), (0.0, "0")):
        corpo.append(f'<text class="lbs" x="{r.x - 8:.0f}" y="{r.sy(v) + 5:.1f}" '
                     f'text-anchor="end">{eti}</text>')

    # l'etichetta del passo: una per stato, visibile nella propria fetta.
    # La prima è già accesa a 0%, l'ultima resta accesa fino a 100%: così
    # nessuna delle due chiede alla timeline di tornare indietro, e il
    # fotogramma finale coincide con lo stato di riposo.
    for k in range(n):
        t0, _ = sosta(k, n)
        passo = 100.0 / n
        tappe = ([(0.0, "opacity:1")] if k == 0 else
                 [(0.0, "opacity:0"), (t0 - 0.4, "opacity:0"), (t0, "opacity:1")])
        if k == n - 1:
            tappe.append((100.0, "opacity:1"))
        else:
            tappe += [(t0 + passo - 0.4, "opacity:1"),
                      (t0 + passo, "opacity:0"), (100.0, "opacity:0")]
        anim.append(keyframes(f"eti{k}", tappe))
        giusto = (p_rho[k] > 0.5) == (e["y_vero"] == 1)
        fermo = ";opacity:1" if k == n - 1 else ""
        corpo.append(
            f'<text class="eti" x="{X0:.0f}" y="504" '
            f'fill="{FG_MUTED if giusto else TERRACOTTA}" '
            f'style="animation:eti{k} var(--d) infinite{fermo}">'
            f'ρ = {virgola(rhos[k])} · p(classe 1) = {virgola(p_rho[k])} · '
            f'predice {int(p_rho[k] > 0.5)}, '
            f'{"corretto" if giusto else "SBAGLIATO"}</text>')

    return Figura(
        larghezza=700, altezza=526,
        alt="Al crescere della perturbazione ρ la spinta su ogni caratteristica "
            "resta minuscola accanto al valore della caratteristica stessa, ma la "
            f"probabilità della classe giusta scende e a ρ = {virgola(e['rho_star'])} "
            "passa sotto la soglia di 0,5: da lì il modello sbaglia.",
        corpo="".join(corpo),
        stile=f"""    .car  {{ fill:{TEAL}; }}
    .sp   {{ fill:{TERRACOTTA}; transform-box:fill-box;
            animation:cresce var(--d) infinite; }}
    .su   {{ transform-origin:50% 100%; }}
    .giu  {{ transform-origin:50% 0%; }}
    .asse {{ stroke:{BORDER_STRONG}; stroke-width:1.5; }}
    .mezza {{ stroke:{FG_MUTED}; stroke-width:1.5; stroke-dasharray:6 5; }}
    .oltre {{ fill:{TERRACOTTA}; opacity:0.08; }}
    .rib  {{ stroke:{TERRACOTTA}; stroke-width:2; stroke-dasharray:5 4; }}
    .rib-t {{ font-family:{SANS}; font-size:14px; font-weight:700;
            fill:{TERRACOTTA}; }}
    .eti  {{ font-family:{SANS}; font-size:15px; opacity:0; }}
    #curva {{ fill:none; stroke:{INK}; stroke-width:2.5;
            stroke-dasharray:{totale:.1f}; animation:disegna var(--d) infinite; }}
    #pallino {{ fill:{TERRACOTTA}; stroke:{CREAM}; stroke-width:2;
            animation:corri var(--d) infinite, vira var(--d) infinite; }}
    #soglia {{ animation:svela var(--d) infinite; }}""",
        animazioni=anim,
        durata=n * 1.3,
        fermi=".sp, #curva, #pallino, #soglia, .eti",
    )
