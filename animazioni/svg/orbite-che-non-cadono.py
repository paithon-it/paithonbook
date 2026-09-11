"""Le due orbite chiuse della sezione: prede e predatori, sasso carta forbici.

La sezione promette in apertura («tenete a mente quelle orbite») e paga in
chiusura, dicendo che la dinamica del replicatore su sasso-carta-forbici dà
formule della stessa famiglia di quelle di Volterra, con le stesse orbite
chiuse attorno all'equilibrio e nessuna che ci cada dentro. Finora quella
promessa arrivava senza un disegno, e chi non sa che cosa sta sui due assi non
può immaginarsi che cosa giri attorno a che cosa.

I due sistemi non sono trascritti: si integrano qui (Runge-Kutta a passo
piccolo) e sono

    preda:      x' = a x - b x y            replicatore:  x_i' = x_i (A x)_i
    predatore:  y' = d x y - c y            con A antisimmetrica di
                                            sasso-carta-forbici

e ciascuno dei due ha una quantita' che non cambia lungo il moto: per Volterra
V = d x - c ln x + b y - a ln y, per il replicatore il prodotto x1 x2 x3. Gli
assert difendono le due cose che la didascalia promette, e le difendono sul
dato disegnato: che l'orbita si chiuda (dopo un giro si torna da dove si e'
partiti) e che non caschi nel centro (la distanza dall'equilibrio non scende
mai sotto una soglia). Se un giorno i parametri cambiassero e le orbite
diventassero spirali, la figura non si genererebbe nemmeno.
"""

import math

from paithon_svg import *

NOME = "orbite-che-non-cadono"
TITOLO = "prede e predatori, sasso e carta: si gira e non si cade"

# --- Volterra. Equilibrio in (c/d, a/b) = (2, 2).
A_PREDA, B_PREDA = 1.0, 0.5
C_PRED, D_PRED = 0.8, 0.4
EQ_LV = (C_PRED / D_PRED, A_PREDA / B_PREDA)
AVVII_LV = [(2.0, 3.0), (2.0, 3.7)]        # due orbite annidate, stesso centro

# --- Replicatore su sasso-carta-forbici: A[i][j] e' il guadagno di i contro j.
PAYOFF = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]
EQ_RPS = (1 / 3, 1 / 3, 1 / 3)
AVVII_RPS = [(0.48, 0.30, 0.22), (0.70, 0.20, 0.10)]

PASSO = 0.0005          # passo di integrazione: piccolo, gli invarianti lo dicono
GIRI = 60000            # passi massimi cercando il ritorno al punto di partenza
POSE = 96               # quanti fotogrammi finiscono nei keyframes


# --------------------------------------------------------------------------
# I due campi vettoriali
# --------------------------------------------------------------------------
def campo_lv(s):
    x, y = s
    return (A_PREDA * x - B_PREDA * x * y, D_PRED * x * y - C_PRED * y)


def campo_rps(s):
    guadagni = [sum(PAYOFF[i][j] * s[j] for j in range(3)) for i in range(3)]
    # A e' antisimmetrica, quindi la media s.As vale zero e non compare.
    return tuple(s[i] * guadagni[i] for i in range(3))


def rk4(campo, s, h):
    def somma(a, b, k):
        return tuple(ai + k * bi for ai, bi in zip(a, b))
    k1 = campo(s)
    k2 = campo(somma(s, k1, h / 2))
    k3 = campo(somma(s, k2, h / 2))
    k4 = campo(somma(s, k3, h))
    return tuple(si + h / 6 * (a + 2 * b + 2 * c + d)
                 for si, a, b, c, d in zip(s, k1, k2, k3, k4))


def orbita(campo, avvio, h=PASSO):
    """Integra finche' non si torna al punto di partenza: un giro esatto."""
    s, traccia = avvio, [avvio]
    for i in range(GIRI):
        s = rk4(campo, s, h)
        traccia.append(s)
        d = math.dist(s, avvio)
        if i > 200 and d < 2e-3:
            return traccia
    raise AssertionError(f"{NOME}: l'orbita non si chiude in {GIRI} passi")


def invariante_lv(s):
    x, y = s
    return D_PRED * x - C_PRED * math.log(x) + B_PREDA * y - A_PREDA * math.log(y)


def invariante_rps(s):
    return s[0] * s[1] * s[2]


def scarto(valori):
    """Quanto l'invariante si e' mosso, in rapporto al suo valore."""
    return (max(valori) - min(valori)) / abs(sum(valori) / len(valori))


def assottiglia(traccia, n):
    passo = (len(traccia) - 1) / (n - 1)
    return [traccia[round(i * passo)] for i in range(n)]


# --------------------------------------------------------------------------
# Il disegno
# --------------------------------------------------------------------------
SU = Riquadro(x=64, y=58, larg=250, alt=250, xmin=0.55, xmax=4.25,
              ymin=0.55, ymax=4.25)
GIU = Riquadro(x=396, y=58, larg=250, alt=250, xmin=-0.70, xmax=0.70,
               ymin=-0.545, ymax=0.855)


def sul_triangolo(s):
    """Dalle tre quote alle coordinate del triangolo, vertici a 90, 210, 330."""
    vertici = [(0.0, 0.62), (-0.537, -0.31), (0.537, -0.31)]
    return (sum(s[i] * vertici[i][0] for i in range(3)),
            sum(s[i] * vertici[i][1] for i in range(3)))


def polilinea(riq, punti):
    return " ".join(f"{riq.sx(x):.1f},{riq.sy(y):.1f}" for x, y in punti)


def punta(riq, punti, frazione):
    """Una freccetta sull'anello: nel fermo immagine e' l'unica cosa che dice
    da che parte si gira, perche' li' il punto non si muove."""
    i = int(frazione * (len(punti) - 1))
    x0, y0 = riq.sx(punti[i][0]), riq.sy(punti[i][1])
    x1, y1 = riq.sx(punti[i + 1][0]), riq.sy(punti[i + 1][1])
    ang = math.degrees(math.atan2(y1 - y0, x1 - x0))
    return (f'<path class="fre" d="M-6,-5 L7,0 L-6,5 Z" '
            f'transform="translate({x0:.1f},{y0:.1f}) rotate({ang:.1f})"/>')


def moto(riq, punti, nome):
    """Keyframes del punto che percorre l'orbita: riposo sull'ultima posa."""
    xf, yf = riq.sx(punti[-1][0]), riq.sy(punti[-1][1])
    tappe = []
    for i, (x, y) in enumerate(punti):
        p = 100.0 * i / (len(punti) - 1)
        tappe.append((p, f"transform:translate({riq.sx(x) - xf:.1f}px,"
                         f"{riq.sy(y) - yf:.1f}px)"))
    return (xf, yf), keyframes(nome, tappe)


def costruisci() -> Figura:
    # --- Volterra
    orbite_lv = [orbita(campo_lv, s) for s in AVVII_LV]
    for tr in orbite_lv:
        assert scarto([invariante_lv(s) for s in tr]) < 1e-6, "Volterra: orbita che deriva"
        assert min(math.dist(s, EQ_LV) for s in tr) > 0.4, "Volterra: cade nel centro"

    # --- replicatore
    orbite_rps = [orbita(campo_rps, s) for s in AVVII_RPS]
    for tr in orbite_rps:
        assert scarto([invariante_rps(s) for s in tr]) < 1e-6, "replicatore: orbita che deriva"
        assert min(math.dist(s, EQ_RPS) for s in tr) > 0.06, "replicatore: cade nel centro"
        assert all(abs(sum(s) - 1) < 1e-9 for s in tr), "replicatore: le tre quote non fanno uno"

    # Il triangolo e' gia' il bordo del pannello di destra, e non ne serve un altro.
    corpo, anim = [SU.cornice()], []

    # ---------------- pannello di sinistra: prede e predatori
    for k, tr in enumerate(orbite_lv):
        classe = "orbita" if k == len(orbite_lv) - 1 else "orbita2"
        corpo.append(f'<polyline class="{classe}" '
                     f'points="{polilinea(SU, assottiglia(tr, 400))}"/>')
    ex, ey = SU.sx(EQ_LV[0]), SU.sy(EQ_LV[1])
    corpo.append(f'<path class="cro" d="M{ex - 7:.1f},{ey}h14M{ex},{ey - 7:.1f}v14"/>')
    corpo.append(f'<text class="lbs" x="{ex + 11:.0f}" y="{ey + 5:.0f}">equilibrio</text>')

    pose_lv = [(s[0], s[1]) for s in assottiglia(orbite_lv[-1], POSE)]
    corpo.append(punta(SU, pose_lv, 0.30))
    (px, py), kf = moto(SU, pose_lv, "olv")
    anim.append(kf)
    corpo.append(f'<circle class="dot" cx="{px:.1f}" cy="{py:.1f}" r="7" '
                 f'style="animation:olv var(--d) infinite linear"/>')

    corpo += [
        f'<text class="ttl" x="{SU.x}" y="{SU.y - 30}">prede e predatori</text>',
        f'<text class="lbs" x="{SU.x}" y="{SU.y - 12}">Lotka 1925, Volterra 1926</text>',
        f'<text class="lbl" x="{SU.x + SU.larg / 2:.0f}" y="{SU.y + SU.alt + 30}" '
        f'text-anchor="middle">quante prede ci sono</text>',
        f'<text class="lbl" transform="translate({SU.x - 32},{SU.y + SU.alt / 2:.0f}) '
        f'rotate(-90)" text-anchor="middle">quanti predatori</text>',
    ]

    # ---------------- pannello di destra: sasso, carta, forbici
    vertici = [sul_triangolo((1, 0, 0)), sul_triangolo((0, 1, 0)),
               sul_triangolo((0, 0, 1))]
    corpo.append('<polygon class="tri" points="'
                 + " ".join(f"{GIU.sx(x):.1f},{GIU.sy(y):.1f}" for x, y in vertici)
                 + '"/>')
    for k, tr in enumerate(orbite_rps):
        classe = "orbita" if k == len(orbite_rps) - 1 else "orbita2"
        corpo.append(f'<polyline class="{classe}" points="'
                     + polilinea(GIU, [sul_triangolo(s)
                                       for s in assottiglia(tr, 400)]) + '"/>')
    cx, cy = sul_triangolo(EQ_RPS)
    ex, ey = GIU.sx(cx), GIU.sy(cy)
    corpo.append(f'<path class="cro" d="M{ex - 7:.1f},{ey}h14M{ex},{ey - 7:.1f}v14"/>')

    pose_rps = [sul_triangolo(s) for s in assottiglia(orbite_rps[-1], POSE)]
    corpo.append(punta(GIU, pose_rps, 0.30))
    (px, py), kf = moto(GIU, pose_rps, "orps")
    anim.append(kf)
    corpo.append(f'<circle class="dot" cx="{px:.1f}" cy="{py:.1f}" r="7" '
                 f'style="animation:orps var(--d) infinite linear"/>')

    for (vx, vy), nome, dx, dy, anc in zip(
            vertici, ("tutti sasso", "tutti carta", "tutti forbici"),
            (0, 4, -4), (-13, 20, 20), ("middle", "start", "end")):
        corpo.append(f'<text class="lbs" x="{GIU.sx(vx) + dx:.0f}" '
                     f'y="{GIU.sy(vy) + dy:.0f}" text-anchor="{anc}">{nome}</text>')

    corpo += [
        f'<text class="ttl" x="{GIU.x}" y="{GIU.y - 30}">sasso, carta, forbici</text>',
        f'<text class="lbs" x="{GIU.x}" y="{GIU.y - 12}">'
        f'la quota di chi gioca ciascuna mossa</text>',
        f'<text class="lbl" x="{ex:.0f}" y="{GIU.y + GIU.alt + 30}" '
        f'text-anchor="middle">al centro, un terzo ciascuno</text>',
    ]

    return Figura(
        larghezza=700, altezza=400,
        alt="Due scene affiancate, e in ciascuna un punto percorre all'infinito "
            "un anello chiuso attorno a una crocetta, senza finirci mai dentro. "
            "A sinistra il piano di Volterra: in orizzontale quante prede ci "
            "sono, in verticale quanti predatori, e due anelli annidati girano "
            "in senso antiorario attorno alla crocetta dell'equilibrio, così "
            "che dove i predatori sono tanti le prede calano e viceversa. A "
            "destra un triangolo con i vertici «tutti sasso», «tutti carta» e "
            "«tutti forbici»: ogni punto dentro il triangolo dice quanta parte "
            "della popolazione gioca ciascuna mossa, la crocetta al centro è un "
            "terzo ciascuno, e anche qui i due anelli girano attorno al centro "
            "senza cadervi dentro. Su ciascun anello esterno una punta di "
            "freccia dice da che parte si gira.",
        corpo="".join(corpo),
        stile=f"""    .orbita  {{ fill:none; stroke:{TEAL}; stroke-width:2.6; }}
    .orbita2 {{ fill:none; stroke:{OCRA}; stroke-width:2.2; }}
    .tri     {{ fill:none; stroke:{BORDER_STRONG}; stroke-width:1.6; }}
    .cro     {{ stroke:{FG_MUTED}; stroke-width:2; }}
    .fre     {{ fill:{TEAL}; }}
    .dot     {{ fill:{TERRACOTTA}; }}""",
        animazioni=anim,
        durata=9.0,
        fermi=".dot",
    )
