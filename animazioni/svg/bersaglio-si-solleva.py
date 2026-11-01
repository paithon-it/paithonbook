"""Il bersaglio che si solleva: il kernel trick visto dal vivo.

Qui il tempo è il contenuto perché la figura deve far vedere che **i punti
sono gli stessi**. Ferma, la scena mostra due disegni affiancati e lascia al
lettore il compito di credere che la nuvola di destra sia quella di sinistra
dopo la mappa; in movimento lo si vede, perché ogni punto sale dal posto in cui
stava. E si vede anche il ritorno, che la figura ferma non poteva mostrare: il
taglio piatto dello spazio sollevato, riportato nello spazio di partenza, è un
cerchio.

La scena ha quattro tempi:

1. lo spazio di partenza visto dall'alto: un disco al centro, un anello
   intorno, e nessuna retta che li separi;
2. la camera si inclina, e il pavimento diventa un'ellisse;
3. la mappa $\\phi(x_1, x_2) = (x_1, x_2, x_1^2 + x_2^2)$ solleva ogni punto
   all'altezza $r^2$, e compare la superficie su cui i punti stanno;
4. un piano orizzontale separa le due classi, e il cerchio in cui taglia la
   superficie scende sul pavimento.

Lo stato di riposo è l'ultimo: i punti sollevati, il piano, e il cerchio sul
pavimento.

## La geometria

La proiezione è ortografica, con la camera all'elevazione `e` sopra il
pavimento. Un punto $(x, y, z)$, già ruotato dell'azimut, finisce sullo
schermo in

    X = Ox + S x,    Y = Oy - S (y sin e + A z cos e)

dove `A` schiaccia la quota (la superficie vera sarebbe alta il doppio della
larghezza della figura). Il pavimento ($z = 0$) è quindi lo stesso disegno
visto dall'alto, schiacciato in verticale di $\\sin e$: per questo il suo
gruppo si anima con un solo `scale(1, k)`, e i punti con una `translate`
ciascuno. Fra due tappe le due trasformazioni variano tutte e due in modo
lineare, così i punti restano esattamente sul pavimento mentre la camera si
inclina.

Niente si addestra e niente si sorteggia con librerie di algebra lineare: i
punti vengono da `random.Random(SEME).random()`, l'unico metodo di cui Python
garantisce la stessa sequenza fra una versione e l'altra, e l'SVG esce identico
su ogni macchina.
"""

import math
import random

from paithon_svg import *

NOME = "bersaglio-si-solleva"
TITOLO = "il bersaglio si solleva, e un piano lo separa"

# --------------------------------------------------------------------------
# I dati: un disco e un anello, le due classi del bersaglio
# --------------------------------------------------------------------------
SEME = 7
N_DISCO, N_ANELLO = 34, 54
R_DISCO = 0.85               # il disco: raggio da 0 a R_DISCO
R_ANELLO = (1.3, 1.8)        # l'anello: raggio fra questi due


def bersaglio() -> list[tuple[float, float, int]]:
    """(x1, x2, classe): 0 il disco al centro, 1 l'anello intorno.

    Uniformi in area: il raggio è la radice di un'uniforme, o i punti si
    affollerebbero al centro e il disco sembrerebbe più piccolo di com'è.
    """
    g = random.Random(SEME)
    punti = []
    for _ in range(N_DISCO):
        r = R_DISCO * math.sqrt(g.random())
        a = 2 * math.pi * g.random()
        punti.append((round(r * math.cos(a), 4), round(r * math.sin(a), 4), 0))
    lo2, hi2 = R_ANELLO[0] ** 2, R_ANELLO[1] ** 2
    for _ in range(N_ANELLO):
        r = math.sqrt(lo2 + (hi2 - lo2) * g.random())
        a = 2 * math.pi * g.random()
        punti.append((round(r * math.cos(a), 4), round(r * math.sin(a), 4), 1))
    return punti


def phi(x1: float, x2: float) -> tuple[float, float, float]:
    """La mappa della sezione: aggiunge la coordinata r² = x1² + x2²."""
    return x1, x2, x1 * x1 + x2 * x2


def soglia(punti) -> float:
    """L'altezza del piano: a metà fra il disco più alto e l'anello più basso."""
    alto_disco = max(phi(x, y)[2] for x, y, c in punti if c == 0)
    basso_anello = min(phi(x, y)[2] for x, y, c in punti if c == 1)
    return round((alto_disco + basso_anello) / 2, 2)


# --------------------------------------------------------------------------
# La verità: quello che la didascalia promette
# --------------------------------------------------------------------------
def dentro_inviluppo(p, altri) -> bool:
    """`p` sta dentro l'inviluppo convesso di `altri`?

    Esatto, senza costruire l'inviluppo: `p` ne sta fuori se e solo se esiste
    un semipiano che passa per `p` e contiene tutti gli altri, cioè se le
    direzioni da `p` verso gli altri lasciano libero un arco di almeno mezzo
    giro.
    """
    angoli = sorted(math.atan2(y - p[1], x - p[0]) for x, y in altri)
    salti = [b - a for a, b in zip(angoli, angoli[1:])]
    salti.append(angoli[0] + 2 * math.pi - angoli[-1])
    return max(salti) < math.pi


def verifica(punti, c: float) -> None:
    """La figura promette tre cose. Ci sono davvero?

    Che nello spazio di partenza nessuna retta separi le due classi; che dopo
    la mappa il piano all'altezza `c` le separi; e che il cerchio in cui quel
    piano taglia la superficie, riportato sul pavimento, passi fra il disco e
    l'anello.
    """
    disco = [(x, y) for x, y, k in punti if k == 0]
    anello = [(x, y) for x, y, k in punti if k == 1]
    assert len(disco) == N_DISCO and len(anello) == N_ANELLO

    # Nessuna retta separa due insiemi i cui inviluppi convessi si toccano, e
    # qui ogni punto del disco sta dentro l'inviluppo dell'anello.
    for p in disco:
        assert dentro_inviluppo(p, anello), \
            (f"il punto del disco {p} sta fuori dall'inviluppo dell'anello: "
             f"una retta potrebbe separarli, e la prima didascalia mentirebbe")

    # Il piano separa: tutto il disco sotto, tutto l'anello sopra, con un
    # margine che si veda nel disegno.
    alto = max(phi(*p)[2] for p in disco)
    basso = min(phi(*p)[2] for p in anello)
    assert alto < c < basso, \
        f"il piano r² = {c} non separa: il disco arriva a {alto:.3f}, " \
        f"l'anello parte da {basso:.3f}"
    assert basso - alto > 0.5, \
        f"fra il disco e l'anello c'è un salto di quota di soli {basso - alto:.2f}"

    # La mappa è quella della sezione, e la terza coordinata è proprio r².
    for x, y in disco + anello:
        assert phi(x, y) == (x, y, x * x + y * y)

    # Il cerchio sul pavimento passa fra le due classi.
    raggio = math.sqrt(c)
    assert max(math.hypot(*p) for p in disco) < raggio \
        < min(math.hypot(*p) for p in anello)


# --------------------------------------------------------------------------
# La camera
# --------------------------------------------------------------------------
S = 104                       # pixel per unità
A = 0.8                       # quanto si schiaccia la quota nel disegno
AZIMUT = math.radians(-30)    # gira gli assi del pavimento, perché si legga 3D
ELEVAZIONI = [90.0, 73.0, 56.0, 39.0, 22.0]   # le tappe della camera, in gradi
E = math.radians(ELEVAZIONI[-1])               # quella finale

R_PAVIMENTO = 2.0
R_CIOTOLA = 1.9               # il bordo della superficie disegnata
R_PIANO = 1.75

LARG = 680
OX = LARG / 2
Y_TITOLO = 34
Y_DISEGNO = 80                # dove comincia il disegno
ALT_DISEGNO = 4.2 * S         # la vista dall'alto: ±2 unità più il margine
OY_ALTO = Y_DISEGNO + ALT_DISEGNO / 2


def gira(x: float, y: float) -> tuple[float, float]:
    ca, sa = math.cos(AZIMUT), math.sin(AZIMUT)
    return x * ca - y * sa, x * sa + y * ca


def schermo(x: float, y: float, z: float, e: float, oy: float):
    """Il punto (x, y, z) dei dati, visto all'elevazione `e`."""
    u, v = gira(x, y)
    return OX + S * u, oy - S * (v * math.sin(e) + A * z * math.cos(e))


def oy_obliqua(c: float) -> float:
    """L'origine del pavimento nella vista finale, con la scena centrata.

    Il disegno obliquo va da sotto il pavimento (il bordo davanti
    dell'ellisse) fino in cima al bordo della superficie: lo si centra nella
    stessa fascia in cui sta la vista dall'alto, così la camera che si inclina
    non porta la scena fuori dal riquadro.
    """
    sotto = R_PAVIMENTO * math.sin(E)
    sopra = max(A * R_CIOTOLA ** 2 * math.cos(E) + R_CIOTOLA * math.sin(E),
                A * c * math.cos(E) + R_PIANO * math.sin(E))
    return OY_ALTO + S * (sopra - sotto) / 2


# --------------------------------------------------------------------------
# Il tempo, in percentuali del ciclo
# --------------------------------------------------------------------------
T_FERMO = 10.0                # fino a qui la vista dall'alto
T_INCLINATA = 26.0            # la camera ha finito di inclinarsi
T_SALE = (30.0, 48.0)         # la mappa solleva i punti
T_CIOTOLA = (42.0, 50.0)      # compare la superficie
T_PIANO = (52.0, 58.0)        # compare il piano
T_TAGLIO = (70.0, 80.0)       # il cerchio scende sul pavimento
TESTI = [0.0, 30.0, 52.0, 70.0]   # quando cambia la riga sotto il disegno
DISSOLVENZA = 2.0


def tappe_camera(oy_fine: float) -> list[tuple[float, float, float]]:
    """(percentuale, elevazione, origine del pavimento) per ogni tappa."""
    n = len(ELEVAZIONI) - 1
    tappe = [(0.0, math.radians(ELEVAZIONI[0]), OY_ALTO)]
    for j, gradi in enumerate(ELEVAZIONI):
        t = T_FERMO + (T_INCLINATA - T_FERMO) * j / n
        oy = OY_ALTO + (oy_fine - OY_ALTO) * j / n
        tappe.append((t, math.radians(gradi), oy))
    return tappe


def compare(nome: str, inizio: float, fine: float) -> str:
    """Invisibile fino a `inizio`, pieno da `fine` in poi."""
    return keyframes(nome, [(0.0, "opacity:0"), (inizio, "opacity:0"),
                            (fine, "opacity:1"), (100.0, "opacity:1")])


def finestra(nome: str, inizio: float, fine: float | None) -> str:
    """Visibile fra `inizio` e `fine`, con una dissolvenza ai due capi."""
    tappe = [(0.0, "opacity:1" if inizio == 0 else "opacity:0")]
    if inizio > 0:
        tappe += [(inizio - DISSOLVENZA, "opacity:0"), (inizio, "opacity:1")]
    if fine is None:
        tappe.append((100.0, "opacity:1"))
    else:
        tappe += [(fine - DISSOLVENZA, "opacity:1"), (fine, "opacity:0"),
                  (100.0, "opacity:0")]
    return keyframes(nome, tappe)


def dec(v: float, cifre: int = 2) -> str:
    return f"{v:.{cifre}f}".replace(".", ",")


def ellisse(cls: str, cy: float, r: float, extra: str = "") -> str:
    return (f'<ellipse class="{cls}" cx="{OX:.1f}" cy="{cy:.1f}" '
            f'rx="{S * r:.1f}" ry="{S * r * math.sin(E):.1f}"{extra}/>')


# --------------------------------------------------------------------------
# La figura
# --------------------------------------------------------------------------
def costruisci() -> Figura:
    punti = bersaglio()
    c = soglia(punti)
    verifica(punti, c)

    oy = oy_obliqua(c)
    camera = tappe_camera(oy)
    y_riga1 = Y_DISEGNO + ALT_DISEGNO + 30
    y_riga2 = y_riga1 + 24
    altezza = y_riga2 + 18

    anim, moto, corpo = [], [], []
    quota = lambda z: S * A * z * math.cos(E)   # noqa: E731

    # Il pavimento: a riposo è l'ellisse obliqua; all'inizio lo stesso disegno
    # allungato in verticale di 1/sin(e), cioè visto dall'alto. Il tratto non
    # deve ingrossarsi con la scala, da cui `non-scaling-stroke`.
    tappe = []
    for t, e, o in camera:
        k = math.sin(e) / math.sin(E)
        tappe.append((t, f"transform:translate(0px,{o:.1f}px) scale(1,{k:.4f}) "
                         f"translate(0px,{-oy:.1f}px)"))
    tappe.append((100.0, tappe[-1][1]))
    anim.append(keyframes("pav", tappe))
    assi = []
    for dx, dy in ((1, 0), (0, 1)):
        x1, y1 = schermo(-R_PAVIMENTO * dx, -R_PAVIMENTO * dy, 0, E, oy)
        x2, y2 = schermo(R_PAVIMENTO * dx, R_PAVIMENTO * dy, 0, E, oy)
        assi.append(f'<line class="asse" x1="{x1:.1f}" y1="{y1:.1f}" '
                    f'x2="{x2:.1f}" y2="{y2:.1f}"/>')
    corpo.append(f'<g class="pav">{ellisse("pavimento", oy, R_PAVIMENTO)}'
                 f'{"".join(assi)}</g>')

    # Il cerchio sul pavimento: a riposo sta lì, e arriva dall'altezza del
    # piano. Sta sotto a tutto il resto, come il pavimento.
    salto = quota(c)
    anim.append(keyframes("scende", [
        (0.0, f"transform:translate(0px,{-salto:.1f}px);opacity:0"),
        (T_TAGLIO[0], f"transform:translate(0px,{-salto:.1f}px);opacity:0"),
        (T_TAGLIO[0] + 1.5, f"transform:translate(0px,{-salto:.1f}px);opacity:1"),
        (T_TAGLIO[1], "transform:translate(0px,0px);opacity:1"),
        (100.0, "transform:translate(0px,0px);opacity:1")]))
    corpo.append(ellisse("taglio", oy, math.sqrt(c),
                         ' style="animation:scende var(--d) linear infinite"'))

    # Le due verticali che portano il cerchio giù: ai suoi estremi destro e
    # sinistro, dove l'ellisse tocca la sua larghezza piena.
    anim.append(compare("cala", T_TAGLIO[0], T_TAGLIO[0] + 3))
    for segno in (-1, 1):
        x = OX + segno * S * math.sqrt(c)
        corpo.append(f'<line class="cala" x1="{x:.1f}" y1="{oy - salto:.1f}" '
                     f'x2="{x:.1f}" y2="{oy:.1f}"/>')

    # La superficie r² = x1² + x2²: il bordo, e il contorno di sotto, che in
    # proiezione ortografica è una parabola e quindi una Bézier quadratica
    # esatta. y* è la profondità a cui la superficie è più bassa sullo schermo.
    anim.append(compare("ciotola", *T_CIOTOLA))
    y_bordo = oy - quota(R_CIOTOLA ** 2)
    ystar = math.sin(E) / (2 * A * math.cos(E))
    assert ystar < R_CIOTOLA
    u = math.sqrt(R_CIOTOLA ** 2 - ystar ** 2)
    alto_u = A * u * u * math.cos(E) - math.sin(E) ** 2 / (4 * A * math.cos(E))
    fondo = -math.sin(E) ** 2 / (4 * A * math.cos(E))
    p0 = (OX - S * u, oy - S * alto_u)
    p1 = (OX, oy - S * (2 * fondo - alto_u))
    p2 = (OX + S * u, oy - S * alto_u)
    corpo.append(
        f'<g class="ciotola">{ellisse("bordo", y_bordo, R_CIOTOLA)}'
        f'<path class="bordo" d="M{p0[0]:.1f},{p0[1]:.1f} '
        f'Q{p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}"/>'
        f'<text class="lbs" x="{OX + S * R_CIOTOLA + 8:.1f}" '
        f'y="{y_bordo + 4:.1f}">r² = x₁² + x₂²</text></g>')

    # I punti. A riposo stanno sollevati e visti di sbieco, senza transform;
    # l'animazione parte dalla vista dall'alto a quota zero, segue la camera
    # che si inclina, e poi sale.
    def un_punto(i: int, x: float, y: float, classe: int) -> str:
        z = phi(x, y)[2]
        px, py = schermo(x, y, z, E, oy)
        tappe = []
        for t, e, o in camera:
            sx, sy = schermo(x, y, 0, e, o)
            tappe.append((t, f"transform:translate({sx - px:.1f}px,{sy - py:.1f}px)"))
        tappe.append((T_SALE[0], tappe[-1][1]))
        tappe.append((T_SALE[1], "transform:translate(0px,0px)"))
        tappe.append((100.0, "transform:translate(0px,0px)"))
        anim.append(keyframes(f"p{i}", tappe))
        moto.append(f"    .v{i} {{ animation:p{i} var(--d) linear infinite; }}")
        return f'<circle class="pt c{classe} v{i}" cx="{px:.1f}" cy="{py:.1f}" r="4.6"/>'

    # Prima il disco, poi il piano, poi l'anello: il piano copre chi gli sta
    # sotto e viene coperto da chi gli sta sopra.
    for i, (x, y, k) in enumerate(punti):
        if k == 0:
            corpo.append(un_punto(i, x, y, k))

    anim.append(compare("piano", *T_PIANO))
    y_piano = oy - quota(c)
    corpo.append(
        f'<g class="piano">{ellisse("lastra", y_piano, R_PIANO)}'
        f'{ellisse("taglio", y_piano, math.sqrt(c))}'
        f'<text class="lbs" x="{OX - S * R_PIANO - 8:.1f}" y="{y_piano + 4:.1f}" '
        f'text-anchor="end">r² = {dec(c)}</text></g>')

    for i, (x, y, k) in enumerate(punti):
        if k == 1:
            corpo.append(un_punto(i, x, y, k))

    # I nomi degli assi del pavimento, fuori dal gruppo che si schiaccia (il
    # testo si deformerebbe): si spostano come i punti a quota zero.
    for j, (nome, dx, dy) in enumerate((("x₁", 1, 0), ("x₂", 0, 1))):
        r = R_PAVIMENTO + 0.16
        px, py = schermo(r * dx, r * dy, 0, E, oy)
        tappe = []
        for t, e, o in camera:
            sx, sy = schermo(r * dx, r * dy, 0, e, o)
            tappe.append((t, f"transform:translate({sx - px:.1f}px,{sy - py:.1f}px)"))
        tappe.append((100.0, tappe[-1][1]))
        anim.append(keyframes(f"a{j}", tappe))
        corpo.append(f'<text class="lbs" x="{px:.1f}" y="{py + 4:.1f}" '
                     f'text-anchor="middle" style="animation:a{j} var(--d) '
                     f'linear infinite">{nome}</text>')

    # Le righe sotto il disegno, una coppia per tempo della scena. I numeri
    # sono quelli del piano calcolato, non scritti a mano.
    righe = [
        ("nello spazio di partenza (x₁, x₂): un disco al centro, un anello intorno",
         "nessuna retta separa i punti teal da quelli terracotta"),
        ("la mappa φ aggiunge una coordinata, r² = x₁² + x₂²",
         "ogni punto sale tanto più quanto è lontano dal centro"),
        ("nello spazio sollevato un piano orizzontale separa le due classi",
         f"il piano r² = {dec(c)}: sotto restano i punti teal, sopra quelli "
         f"terracotta"),
        ("riportato nello spazio di partenza, il taglio piatto è un cerchio",
         f"x₁² + x₂² = {dec(c)}: lì la frontiera è curva"),
    ]
    for k, (riga1, riga2) in enumerate(righe):
        fine = TESTI[k + 1] if k + 1 < len(TESTI) else None
        anim.append(finestra(f"t{k}", TESTI[k], fine))
        base = "" if fine is None else "opacity:0;"
        corpo.append(
            f'<g style="{base}animation:t{k} var(--d) linear infinite">'
            f'<text class="lbl" x="40" y="{y_riga1:.1f}">{riga1}</text>'
            f'<text class="lbs" x="40" y="{y_riga2:.1f}">{riga2}</text></g>')

    corpo += [
        f'<text class="ttl" x="40" y="{Y_TITOLO}">'
        f'Una coordinata in più, e la frontiera torna dritta</text>',
        f'<text class="lbs" x="40" y="{Y_TITOLO + 22}">gli stessi punti, prima '
        f'e dopo la mappa φ(x₁, x₂) = (x₁, x₂, x₁² + x₂²)</text>']

    # Nessun punto esce dall'inquadratura, in nessuna tappa.
    for x, y, _ in punti:
        for _, e, o in camera:
            sx, sy = schermo(x, y, 0, e, o)
            assert 20 < sx < LARG - 20 and Y_DISEGNO - 10 < sy < y_riga1 - 20, \
                f"il punto ({x}, {y}) esce dall'inquadratura: ({sx:.0f}, {sy:.0f})"
        sx, sy = schermo(x, y, phi(x, y)[2], E, oy)
        assert Y_DISEGNO - 10 < sy < y_riga1 - 20

    return Figura(
        larghezza=LARG, altezza=altezza,
        alt="Una nuvola di punti vista prima dall'alto: un disco di punti teal "
            "al centro e un anello di punti terracotta intorno, che nessuna "
            "retta può separare. La vista si inclina e il pavimento diventa "
            "un'ellisse. Poi ogni punto sale a un'altezza pari al quadrato "
            "della sua distanza dal centro: i punti del disco restano in basso, "
            "quelli dell'anello salgono, e compare la superficie a forma di "
            "ciotola su cui stanno tutti. Un piano orizzontale semitrasparente "
            f"si infila a quota {dec(c)}, sopra il disco e sotto l'anello, e "
            "li separa. Infine il cerchio in cui il piano taglia la ciotola "
            "scende sul pavimento: nello spazio di partenza la frontiera è "
            "quel cerchio, fra il disco e l'anello.",
        corpo="".join(corpo),
        stile=f"""    .pt  {{ opacity:0.9; }}
    .c0  {{ fill:{TEAL}; }}
    .c1  {{ fill:{TERRACOTTA}; }}
    .pav {{ animation:pav var(--d) linear infinite; }}
    .pavimento {{ fill:none; stroke:{BORDER_STRONG}; stroke-width:1.5;
                 vector-effect:non-scaling-stroke; }}
    .asse {{ stroke:{BORDER}; stroke-width:1.5; vector-effect:non-scaling-stroke; }}
    .ciotola {{ animation:ciotola var(--d) linear infinite; }}
    .bordo {{ fill:none; stroke:{BORDER_STRONG}; stroke-width:1.6; }}
    .piano {{ animation:piano var(--d) linear infinite; }}
    .lastra {{ fill:{OCRA}; fill-opacity:0.22; stroke:{OCRA}; stroke-width:1.6; }}
    .taglio {{ fill:none; stroke:{INK}; stroke-width:2; stroke-dasharray:6 4; }}
    .cala {{ stroke:{FG_MUTED}; stroke-width:1.2; stroke-dasharray:3 4;
            animation:cala var(--d) linear infinite; }}
""" + "\n".join(moto),
        animazioni=anim,
        durata=16.0,
    )
