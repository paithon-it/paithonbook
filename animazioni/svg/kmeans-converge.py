"""k-means che converge, sull'esempio svolto a mano nel capitolo.

I sei punti A..F, i due centroidi di partenza (scelti male di proposito) e le
due iterazioni sono quelli di `MachineLearning/riduzione-clustering.md`. Qui
però non sono trascritti: l'algoritmo di Lloyd gira davvero (`lloyd`) e
`verifica` confronta con un assert i centroidi calcolati e i due cluster con
quelli che il capitolo dichiara. Se un giorno il testo cambia i numeri e la
figura no, la figura non si genera nemmeno.

Le due cose che il fermo immagine non può mostrare, e che qui si vedono:
l'alternanza delle due mosse (nell'assegnazione cambiano colore i punti, e i
centroidi stanno fermi; nell'aggiornamento si spostano i centroidi, e i colori
stanno fermi) e il punto C che alla seconda assegnazione cambia gruppo, cioè la
partenza sbagliata che si corregge da sola.

Lo stato di riposo è la convergenza, con la scia che dice da dove è passato
ciascun centroide: chi non anima (stampa, PDF, `prefers-reduced-motion`) vede il
risultato e la storia che lo ha prodotto.
"""

from paithon_svg import *

NOME = "kmeans-converge"
TITOLO = "k-means converge: assegna, ricalcola, ripeti"

K = 2
COLORI = (TERRACOTTA, TEAL)

NOMI = "ABCDEF"
PUNTI = [(1, 1), (1, 2), (2, 1), (8, 8), (9, 8), (8, 9)]
INIZIALI = [(1, 1), (2, 1)]      # entrambi nel gruppo di sinistra: partenza cattiva

# La guardia. Quello che il capitolo scrive dopo ciascun aggiornamento, e i
# cluster dopo ciascuna assegnazione: se il testo cambia, questi assert saltano.
ATTESI = [((1, 1.5), (6.75, 6.5)), ((4 / 3, 4 / 3), (25 / 3, 25 / 3))]
ATTESI_GRUPPI = ["AB|CDEF", "ABC|DEF", "ABC|DEF"]
CAMBIA_GRUPPO = ["C"]            # alla seconda assegnazione, e nessun altro

# Dove sta l'etichetta di ciascun punto: (dx, dy, ancoraggio). Fuori dal proprio
# gruppo, per non finire sotto un centroide né oltre il bordo del riquadro.
ETICHETTE = {"A": (-14, 5, "end"), "B": (-14, 5, "end"), "C": (14, 5, "start"),
             "D": (-14, 5, "end"), "E": (14, 5, "start"), "F": (0, -15, "middle")}


# --------------------------------------------------------------------------
# L'algoritmo
# --------------------------------------------------------------------------
def d2(p, q) -> float:
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2


def lloyd(punti, centri):
    """Alterna assegnazione e aggiornamento finché l'assegnazione non cambia.

    Restituisce la successione degli stati (tipo, centroidi, assegnazione):
    l'ultimo è l'assegnazione che non muove più niente, cioè la convergenza.
    """
    stati, prec, centri = [], None, list(centri)
    for _ in range(30):
        a = [min(range(K), key=lambda j: d2(p, centri[j])) for p in punti]
        stati.append(("assegnazione", list(centri), a))
        if a == prec:
            return stati
        prec = a
        centri = [(sum(x for x, _ in g) / len(g), sum(y for _, y in g) / len(g))
                  for g in ([p for p, c in zip(punti, a) if c == j] for j in range(K))]
        stati.append(("aggiornamento", list(centri), a))
    raise AssertionError("Lloyd non converge: dati o partenza da rivedere")


def inerzia(punti, centri, a) -> float:
    """La L del capitolo: somma delle distanze quadrate dal proprio centroide."""
    return sum(d2(p, centri[c]) for p, c in zip(punti, a))


def gruppi(a) -> str:
    """I cluster come li scrive il capitolo: 'AB|CDEF'."""
    return "|".join("".join(NOMI[i] for i, c in enumerate(a) if c == j)
                    for j in range(K))


def verifica(stati) -> None:
    """I numeri calcolati coincidono con quelli scritti nel capitolo?"""
    agg = [c for t, c, _ in stati if t == "aggiornamento"]
    ass = [a for t, _, a in stati if t == "assegnazione"]

    assert len(agg) == len(ATTESI), \
        f"il capitolo racconta {len(ATTESI)} aggiornamenti, qui ne servono {len(agg)}"
    for k, (calcolati, attesi) in enumerate(zip(agg, ATTESI), 1):
        for j in range(K):
            assert all(abs(v - w) < 5e-4 for v, w in zip(calcolati[j], attesi[j])), \
                (f"aggiornamento {k}: mu{j + 1} calcolato {calcolati[j]}, "
                 f"il capitolo scrive {attesi[j]}")

    assert len(ass) == len(ATTESI_GRUPPI), \
        f"il capitolo arriva a convergenza in {len(ATTESI_GRUPPI)} assegnazioni, qui {len(ass)}"
    for k, atteso in enumerate(ATTESI_GRUPPI, 1):
        assert gruppi(ass[k - 1]) == atteso, \
            f"assegnazione {k}: cluster {gruppi(ass[k - 1])}, il capitolo scrive {atteso}"

    cambiati = [NOMI[i] for i in range(len(PUNTI)) if ass[1][i] != ass[0][i]]
    assert cambiati == CAMBIA_GRUPPO, \
        f"alla seconda assegnazione cambia gruppo {cambiati}, il capitolo dice {CAMBIA_GRUPPO}"


# --------------------------------------------------------------------------
# La figura
# --------------------------------------------------------------------------
def traccia(valori, passo, modo):
    """(tempo, valore) per una successione di stati, saltando quelli fermi.

    `snap` cambia di scatto all'inizio della fetta (l'assegnazione è discreta),
    `glide` transita dentro la fetta (l'aggiornamento è uno spostamento, e si
    deve vedere mentre avviene: se transitasse *fra* le fette il centroide si
    muoverebbe mentre l'etichetta dice ancora «assegnazione»).
    """
    tappe = [(0.0, valori[0])]
    for s in range(1, len(valori)):
        if valori[s] == valori[s - 1]:
            continue
        t0 = s * passo
        if modo == "snap":
            tappe += [(max(t0 - passo * 0.09, 0.01), valori[s - 1]), (t0, valori[s])]
        else:
            tappe += [(t0, valori[s - 1]), (min(t0 + passo * 0.7, 99.9), valori[s])]
    tappe.append((100.0, valori[-1]))
    return tappe


def numero(v: float) -> str:
    return f"{v:.1f}".replace(".", ",")


def mu(j: int) -> str:
    """La lettera greca col pedice, come nel capitolo. Il font di sistema non
    ha i pedici Unicode: il pedice si fa con un tspan."""
    return f'&#956;<tspan class="idx" dy="3">{j + 1}</tspan>'


def sottotitolo(stati, s) -> str:
    """Che cosa è appena successo, detto in lingua comune e ricavato dagli stati."""
    tipo, _, a = stati[s]
    if s == 0:
        return "i due centroidi capitano tutti e due nel gruppo di sinistra"
    if tipo == "aggiornamento":
        return "ogni centroide si sposta nella media dei suoi punti"
    if s == 1:
        return "ogni punto prende il colore del centroide più vicino"
    cambiati = [NOMI[i] for i in range(len(PUNTI)) if a[i] != stati[s - 2][2][i]]
    if not cambiati:
        return "nessun punto cambia gruppo: l'algoritmo si ferma"
    return (f"{' e '.join(cambiati)} cambia gruppo: adesso è più vicino a "
            f"{mu(a[NOMI.index(cambiati[0])])}")


def inquadra(punti, x, y, lato, margine) -> Riquadro:
    """Riquadro centrato sui dati, con la stessa scala sui due assi.

    Le due scale devono restare uguali: qui si guardano distanze, e un riquadro
    che allunga un asse mostrerebbe come «più vicino» ciò che non lo è.
    """
    xs = [p[0] for p in punti]
    ys = [p[1] for p in punti]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    mezzo = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 + margine
    return Riquadro(x=x, y=y, larg=lato, alt=lato,
                    xmin=cx - mezzo, xmax=cx + mezzo,
                    ymin=cy - mezzo, ymax=cy + mezzo)


def segno(px, py, colore, raggio=10, braccio=5.5) -> str:
    """Il segno di un centroide: una x dentro un cerchietto, come nel testo.

    Sotto ci va un alone chiaro invece di un disco pieno: alla partenza i due
    centroidi stanno esattamente su A e su C (il capitolo lo dice), e un disco
    pieno cancellerebbe proprio i due punti che in quel momento si guardano.
    Con l'alone il pallino sotto resta visibile fra i bracci della x.
    """
    d = (f'M{px - braccio:.1f},{py - braccio:.1f}'
         f'L{px + braccio:.1f},{py + braccio:.1f}'
         f'M{px - braccio:.1f},{py + braccio:.1f}'
         f'L{px + braccio:.1f},{py - braccio:.1f}')
    return (f'<circle class="ceh" cx="{px:.1f}" cy="{py:.1f}" r="{raggio}"/>'
            f'<path class="ceh" d="{d}"/>'
            f'<circle class="cea" cx="{px:.1f}" cy="{py:.1f}" r="{raggio}" '
            f'stroke="{colore}"/>'
            f'<path class="cex" stroke="{colore}" d="{d}"/>')


def costruisci() -> Figura:
    stati = [("partenza", list(INIZIALI), None)] + lloyd(PUNTI, INIZIALI)
    verifica(stati[1:])
    n = len(stati)
    passo = 100.0 / n
    r = inquadra(PUNTI, x=48, y=30, lato=380, margine=0.9)

    corpo, anim = [r.cornice()], []

    # --- i valori sugli assi: servono a leggere sulla figura le coordinate che
    #     il testo scrive accanto (A(1,1), B(1,2), ...)
    for v in (2, 4, 6, 8):
        corpo += [
            f'<text class="tic" x="{r.sx(v):.1f}" y="{r.y + r.alt + 18}" '
            f'text-anchor="middle">{v}</text>',
            f'<text class="tic" x="{r.x - 9}" y="{r.sy(v) + 4:.1f}" '
            f'text-anchor="end">{v}</text>']

    # --- le scie: un segmento per ogni spostamento, che compare quando avviene
    for s in range(1, n):
        tipo, centri, _ = stati[s]
        if tipo != "aggiornamento":
            continue
        t0 = s * passo
        anim.append(keyframes(f"sci{s}", [
            (0.0, "opacity:0"), (t0, "opacity:0"),
            (min(t0 + passo * 0.7, 99.9), "opacity:0.5"), (100.0, "opacity:0.5")]))
        for j in range(K):
            a, b = stati[s - 1][1][j], centri[j]
            corpo.append(
                f'<line class="scia" x1="{r.sx(a[0]):.1f}" y1="{r.sy(a[1]):.1f}" '
                f'x2="{r.sx(b[0]):.1f}" y2="{r.sy(b[1]):.1f}" stroke="{COLORI[j]}" '
                f'opacity="0.5" style="animation:sci{s} var(--d) infinite"/>')

    # --- i punti: il colore è il gruppo, e cambia solo nei passi di assegnazione
    colori = [[FG_MUTED if a is None else COLORI[a[i]] for _, _, a in stati]
              for i in range(len(PUNTI))]
    nomi = {}
    for i, (x, y) in enumerate(PUNTI):
        chiave = tuple(colori[i])
        if chiave not in nomi:                     # più punti fanno la stessa storia
            nomi[chiave] = f"col{len(nomi)}"
            anim.append(keyframes(nomi[chiave], [
                (t, f"fill:{v}") for t, v in traccia(list(chiave), passo, "snap")]))
        dx, dy, anc = ETICHETTE[NOMI[i]]
        corpo += [
            f'<circle class="pt" cx="{r.sx(x):.1f}" cy="{r.sy(y):.1f}" r="7" '
            f'fill="{chiave[-1]}" style="animation:{nomi[chiave]} var(--d) infinite"/>',
            f'<text class="nom" x="{r.sx(x) + dx:.1f}" y="{r.sy(y) + dy:.1f}" '
            f'text-anchor="{anc}">{NOMI[i]}</text>']

    # --- chi cambia gruppo si accende: è la mossa di assegnazione che si vede,
    #     ed è l'unico momento in cui succede qualcosa di non ovvio
    lampi = set()
    for s in range(2, n):
        for i, (x, y) in enumerate(PUNTI):
            if colori[i][s] == colori[i][s - 1]:
                continue
            lampi.add(s)
            corpo.append(f'<circle class="alo" cx="{r.sx(x):.1f}" cy="{r.sy(y):.1f}" '
                         f'r="16" stroke="{colori[i][s]}" stroke-opacity="0" '
                         f'style="animation:alo{s} var(--d) infinite"/>')
    for s in sorted(lampi):
        t0 = s * passo
        anim.append(keyframes(f"alo{s}", [
            (0.0, "stroke-opacity:0"), (max(t0 - passo * 0.09, 0.01), "stroke-opacity:0"),
            (t0, "stroke-opacity:0.75"), (min(t0 + passo * 0.6, 99.9), "stroke-opacity:0"),
            (100.0, "stroke-opacity:0")]))

    # --- i centroidi: disegnati dove arrivano, l'animazione parte dall'inverso.
    #     Nessuna etichetta accanto al segno: un mu del colore del gruppo cade
    #     sopra i punti di quel gruppo e sparisce, e la targhetta chiara che lo
    #     renderebbe leggibile copre i punti. Il colore dice già quale centroide
    #     è quale, e mu sta nella legenda.
    for j in range(K):
        fx, fy = stati[-1][1][j]
        px, py = r.sx(fx), r.sy(fy)
        offset = [f"transform:translate({r.sx(c[j][0]) - px:.1f}px,"
                  f"{r.sy(c[j][1]) - py:.1f}px)" for _, c, _ in stati]
        anim.append(keyframes(f"cen{j}", traccia(offset, passo, "glide")))
        corpo.append(f'<g style="animation:cen{j} var(--d) infinite">'
                     f'{segno(px, py, COLORI[j])}</g>')

    # --- le etichette: che mossa si sta guardando, e a quanto sta l'inerzia
    lx = r.x + r.larg + 26
    for s, (tipo, centri, a) in enumerate(stati):
        t0 = s * passo
        if s == 0:                       # la prima è già accesa quando parte
            tappe = [(0.0, "opacity:1"), (passo * 0.94, "opacity:1"),
                     (passo, "opacity:0"), (100.0, "opacity:0")]
        elif s == n - 1:                 # l'ultima resta accesa: è lo stato di riposo
            tappe = [(0.0, "opacity:0"), (t0 - passo * 0.09, "opacity:0"),
                     (t0, "opacity:1"), (100.0, "opacity:1")]
        else:
            tappe = [(0.0, "opacity:0"), (t0 - passo * 0.09, "opacity:0"),
                     (t0, "opacity:1"), (t0 + passo * 0.94, "opacity:1"),
                     (t0 + passo, "opacity:0"), (100.0, "opacity:0")]
        anim.append(keyframes(f"eti{s}", tappe))
        if s == 0:
            titolo = "partenza"
        elif s == n - 1:
            titolo = "convergenza"
        else:
            titolo = f"iterazione {(s + 1) // 2}: {tipo}"
        fermo = ";opacity:1" if s == n - 1 else ""
        moto = f'style="animation:eti{s} var(--d) infinite{fermo}"'
        corpo += [
            f'<text class="pas" x="{r.x}" y="{r.y + r.alt + 46}" {moto}>{titolo}</text>',
            f'<text class="spg" x="{r.x}" y="{r.y + r.alt + 70}" {moto}>'
            f'{sottotitolo(stati, s)}</text>']
        if a is not None:
            corpo.append(f'<text class="val" x="{lx}" y="228" {moto}>'
                         f'{numero(inerzia(PUNTI, centri, a))}</text>')

    # --- la colonna di destra: come si leggono i segni, e cosa misura l'inerzia
    for j in range(K):
        y = 60 + j * 36
        corpo += [
            f'<circle class="pt" cx="{lx + 7}" cy="{y - 5}" r="6.5" fill="{COLORI[j]}"/>',
            segno(lx + 33, y - 5, COLORI[j], raggio=9, braccio=4.5),
            f'<text class="lbs" x="{lx + 52}" y="{y}">gruppo {j + 1}, '
            f'centro {mu(j)}</text>']
    corpo += [
        f'<line class="scia" x1="{lx}" y1="{60 + K * 36}" x2="{lx + 22}" '
        f'y2="{60 + K * 36}" stroke="{FG_MUTED}" opacity="0.7"/>',
        f'<text class="lbs" x="{lx + 32}" y="{65 + K * 36}">il loro cammino</text>',
        f'<text class="lbs" x="{lx}" y="200">inerzia</text>',
        f'<text class="lbs" x="{lx}" y="256">la somma delle distanze</text>',
        f'<text class="lbs" x="{lx}" y="274">al quadrato di ogni punto</text>',
        f'<text class="lbs" x="{lx}" y="292">dal proprio centroide</text>',
        f'<text class="lbs" x="{lx}" y="348">k = {K} lo decidiamo noi</text>',
        f'<text class="lbs" x="{lx}" y="366">prima di cominciare</text>',
    ]

    return Figura(
        larghezza=700, altezza=500,
        alt=f"I {len(PUNTI)} punti A, B, C, D, E, F su un piano, e {K} centroidi "
            "segnati con una x. I due centroidi partono tutti e due nel gruppo di "
            "sinistra, poi si alternano due mosse: l'assegnazione, in cui ogni "
            "punto prende il colore del centroide più vicino, e l'aggiornamento, "
            "in cui ogni centroide si sposta nella media dei suoi punti. Alla "
            "seconda assegnazione il punto C passa dal gruppo di destra a quello "
            "di sinistra; dopo il secondo aggiornamento nessun punto cambia più "
            "colore e l'algoritmo si ferma. Una scia tratteggiata mostra il "
            "cammino percorso da ciascun centroide, e l'inerzia scende a ogni "
            "mezzo passo.",
        corpo="".join(corpo),
        stile=f"""    .pt   {{ stroke:{CREAM}; stroke-width:1.2; }}
    .alo  {{ fill:none; stroke-width:3; }}
    .scia {{ stroke-width:2; stroke-dasharray:5 5; }}
    .ceh  {{ fill:none; stroke:{CREAM}; stroke-width:6; stroke-linecap:round; }}
    .cea  {{ fill:none; stroke-width:2; }}
    .cex  {{ fill:none; stroke-width:3.2; stroke-linecap:round; }}
    .nom  {{ font-family:{SANS}; font-size:14px; font-weight:700; fill:{INK}; }}
    .tic  {{ font-family:{SANS}; font-size:12px; fill:{FG_MUTED}; }}
    .idx  {{ font-size:11px; }}
    .pas  {{ font-family:{SANS}; font-size:17px; font-weight:700;
            fill:{TERRACOTTA}; opacity:0; }}
    .spg  {{ font-family:{SANS}; font-size:14px; fill:{FG_MUTED}; opacity:0; }}
    .val  {{ font-family:{SANS}; font-size:26px; font-weight:700;
            fill:{TEAL}; opacity:0; }}""",
        animazioni=anim,
        durata=n * 1.4,
        fermi=".pt, .alo, .scia, .pas, .spg, .val, g",
    )
