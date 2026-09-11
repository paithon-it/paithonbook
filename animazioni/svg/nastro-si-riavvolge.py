"""Il nastro di autograd: appunti presi all'andata, riletti e consumati al ritorno.

Il tempo è il contenuto, e in due sensi diversi. Il primo è l'ordine: la
pagina dice che «per poter tornare indietro, l'andata deve ricordare», e la
figura ferma che le sta accanto (i due sensi di marcia) quella cosa non la può
mostrare, perché l'appunto esiste solo *fra* le due passate. Il secondo è la
somma: lo stesso $x$ comanda in due punti del conto, quindi al ritorno riceve
un contributo per ramo, e i due arrivano **uno dopo l'altro**. È la ragione
per cui `.grad` è un `+=` e non un `=`, e a figura ferma sarebbe soltanto un
numero già sommato.

Il conto è quello della pagina, $y = x^2 + 2x$ con $x = 3$, quindi la scena
non introduce niente che il lettore non abbia già davanti. Il grafo ha due
rami che partono dallo stesso nodo: il ramo del quadrato restituisce
$1 \\cdot 2x = 6$, quello lineare $1 \\cdot 2 = 2$, e in `x.grad` finisce $8$.

**I numeri li calcola la scena**, ramo per ramo, con la regola della catena
applicata a mano su ciascun nodo. Le guardie sono quattro e difendono la
didascalia: i valori dell'andata, i due contributi del ritorno, la loro somma,
e il confronto con la derivata ottenuta per differenze finite centrate, che è
una strada indipendente dalla prima (su una parabola è esatta a meno
dell'errore di macchina). Se un giorno i due conti non coincidessero, la
figura non si genererebbe.
"""

import math

from paithon_svg import *

NOME = "nastro-si-riavvolge"
TITOLO = "il nastro di autograd: gli appunti dell'andata, consumati al ritorno"

X = 3.0                      # il punto in cui la pagina calcola la derivata


def conta():
    """Andata e ritorno sul grafo di $y = x^2 + 2x$, un nodo per volta.

    Torna i valori dell'andata, i due contributi del ritorno e la loro somma.
    Niente di simbolico: a ogni nodo si applica la derivata locale, che è la
    stessa tabellina che il registratore rilegge.
    """
    # andata
    u = X * X                       # ramo del quadrato
    v = 2.0 * X                     # ramo lineare
    y = u + v

    # gli appunti: il numero che ogni stazione deve tenere da parte per
    # sapere, al ritorno, per quanto moltiplicare
    appunto_quadrato = X            # d(x·x)/dx = 2x, e serve x
    appunto_lineare = 2.0           # d(2x)/dx = 2, e serve il 2

    # ritorno, dal fondo: la derivata di y rispetto a sé stesso è 1
    seme = 1.0
    ramo_quadrato = seme * 2.0 * appunto_quadrato
    ramo_lineare = seme * appunto_lineare
    grad = ramo_quadrato + ramo_lineare

    assert (u, v, y) == (9.0, 6.0, 15.0), f"andata: {(u, v, y)}"
    assert (ramo_quadrato, ramo_lineare) == (6.0, 2.0), \
        f"i due contributi: {(ramo_quadrato, ramo_lineare)}"
    assert grad == 8.0, f"la somma dei due rami: {grad}"

    # la seconda strada: differenze finite centrate, che su una parabola
    # danno la derivata esatta a meno dell'errore di macchina. Non condivide
    # niente con il conto qui sopra, quindi se le due coincidono il grafo è
    # percorso bene.
    def f(t):
        return t * t + 2.0 * t
    h = 1e-3
    stimato = (f(X + h) - f(X - h)) / (2.0 * h)
    assert math.isclose(stimato, grad, rel_tol=1e-9), \
        f"differenze finite {stimato} contro il grafo {grad}"

    return u, v, y, appunto_quadrato, appunto_lineare, ramo_quadrato, ramo_lineare, grad


def n(v: float) -> str:
    """Un numero come lo scrive il libro: senza decimali se è intero."""
    return f"{v:g}".replace(".", ",")


# --------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------
LARG, ALT = 700, 436

X_BOX = (40, 178, 112, 58)          # x, y, larghezza, altezza
GRAD_BOX = (26, 302, 168, 54)
SU_BOX = (272, 86, 140, 58)
GIU_BOX = (272, 264, 140, 58)
Y_BOX = (528, 178, 132, 58)
SU_NOTA = (268, 152, 148, 30)
GIU_NOTA = (268, 330, 148, 30)

STATI = 8
TENUTA = 0.60


def centro(b):
    x, y, w, h = b
    return x + w / 2, y + h / 2


def riquadro(b, cls="lbl", testo="", forte=True, dash=None, colore=None):
    x, y, w, h = b
    tratto = colore or (BORDER_STRONG if forte else BORDER)
    d = f' stroke-dasharray="{dash}"' if dash else ""
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="none" '
           f'stroke="{tratto}" stroke-width="2"{d}/>']
    if testo:
        cx, cy = centro(b)
        out.append(f'<text class="{cls}" x="{cx:.0f}" y="{cy + 5:.0f}" '
                   f'text-anchor="middle">{testo}</text>')
    return out


def scosta(p0, p1, d):
    """I due estremi spostati di `d` pixel in perpendicolare alla congiungente.

    Serve a separare le due direzioni di marcia: andata e ritorno percorrono
    lo stesso tratto, e sovrapposte darebbero una riga sola con due punte
    opposte, cioè un disegno che non si legge.
    """
    (x0, y0), (x1, y1) = p0, p1
    L = math.hypot(x1 - x0, y1 - y0)
    nx, ny = -(y1 - y0) / L, (x1 - x0) / L
    return (x0 + nx * d, y0 + ny * d), (x1 + nx * d, y1 + ny * d)


def mezzo(p0, p1, d=0.0):
    """Il punto di mezzo del tratto, scostato di `d` in perpendicolare."""
    (x0, y0), (x1, y1) = scosta(p0, p1, d)
    return (x0 + x1) / 2, (y0 + y1) / 2


def freccia(p0, p1, colore, larghezza=2.0):
    """Un segmento con la punta, accorciato alle due estremità."""
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    x0, y0 = x0 + ux * 6, y0 + uy * 6
    x1, y1 = x1 - ux * 10, y1 - uy * 10
    ang = math.degrees(math.atan2(uy, ux))
    return [f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" '
            f'stroke="{colore}" stroke-width="{larghezza}" stroke-linecap="round"/>',
            f'<path d="M {x1 - 8:.1f} {y1 - 5.5:.1f} L {x1:.1f} {y1:.1f} '
            f'L {x1 - 8:.1f} {y1 + 5.5:.1f}" fill="none" stroke="{colore}" '
            f'stroke-width="{larghezza}" stroke-linecap="round" stroke-linejoin="round" '
            f'transform="rotate({ang:.1f} {x1:.1f} {y1:.1f})"/>']


def acceso(nome, accensioni, anim):
    """`@keyframes` che tiene un elemento visibile negli stati elencati."""
    tappe = []
    for s in range(STATI):
        t0, t1 = sosta(s, STATI, TENUTA)
        op = "1" if s in accensioni else "0"
        tappe += [(t0, f"opacity:{op}"), (t1, f"opacity:{op}")]
    tappe.append((100.0, "opacity:1" if (STATI - 1) in accensioni else "opacity:0"))
    anim.append(keyframes(nome, tappe))
    return f' opacity="{1 if (STATI - 1) in accensioni else 0}" ' \
           f'style="animation:{nome} var(--d) infinite"'


def costruisci() -> Figura:
    u, v, y, ap_q, ap_l, ramo_q, ramo_l, grad = conta()
    corpo, anim = [], []

    # --- la struttura, sempre in vista -----------------------------------
    corpo += riquadro(X_BOX, testo=f"x = {n(X)}")
    corpo += riquadro(SU_BOX, testo="x · x")
    corpo += riquadro(GIU_BOX, testo="2 · x")
    corpo += riquadro(Y_BOX, testo=f"y = {n(y)}")

    cx, cy = centro(X_BOX)
    csu, csuy = centro(SU_BOX)
    cgiu, cgiuy = centro(GIU_BOX)
    cy_, cyy = centro(Y_BOX)

    p_x_su = ((X_BOX[0] + X_BOX[2], X_BOX[1] + 16), (SU_BOX[0], SU_BOX[1] + SU_BOX[3] - 12))
    p_x_giu = ((X_BOX[0] + X_BOX[2], X_BOX[1] + X_BOX[3] - 16), (GIU_BOX[0], GIU_BOX[1] + 12))
    p_su_y = ((SU_BOX[0] + SU_BOX[2], SU_BOX[1] + SU_BOX[3] - 12), (Y_BOX[0], Y_BOX[1] + 16))
    p_giu_y = ((GIU_BOX[0] + GIU_BOX[2], GIU_BOX[1] + 12), (Y_BOX[0], Y_BOX[1] + Y_BOX[3] - 16))

    SCOSTO = 9                       # quanto si separano le due direzioni
    for p0, p1 in (p_x_su, p_x_giu, p_su_y, p_giu_y):
        a, b = scosta(p0, p1, -SCOSTO)
        corpo += freccia(a, b, BORDER_STRONG, 2.0)

    # il contatore di x.grad
    corpo += riquadro(GRAD_BOX, forte=False)
    gx, gy = GRAD_BOX[0] + 14, GRAD_BOX[1] + 22
    corpo.append(f'<text class="lbs" x="{gx}" y="{gy}">x.grad</text>')

    # --- l'andata: i valori che passano ----------------------------------
    def etichetta(x, y_, testo, cls, accensioni, chiave, ancora="middle"):
        a = acceso(chiave, accensioni, anim)
        corpo.append(f'<text class="{cls}" x="{x:.0f}" y="{y_:.0f}" '
                     f'text-anchor="{ancora}"{a}>{testo}</text>')

    AVANTI = set(range(1, STATI))          # una volta comparsi, restano
    for tratto, valore, accensioni, chiave in (
            (p_x_su, X, AVANTI, "a1"),
            (p_x_giu, X, AVANTI, "a2"),
            (p_su_y, u, set(range(2, STATI)), "a3"),
            (p_giu_y, v, set(range(2, STATI)), "a4")):
        mx, my = mezzo(*tratto, -SCOSTO - 12)
        etichetta(mx, my, n(valore), "av", accensioni, chiave)

    # il valore di y compare solo quando i due rami si sono ritrovati
    corpo.append('<rect x="529" y="179" width="130" height="56" rx="7" '
                 f'fill="{CREAM}"' + acceso("coprey", {0, 1}, anim) + '/>')
    etichetta(cy_, cyy + 5, "y = ?", "lbl", {0, 1}, "yq")

    # --- il ritorno -------------------------------------------------------
    # il seme: la derivata di y rispetto a se' stesso
    etichetta(cy_, Y_BOX[1] - 14, "1", "ind", set(range(3, STATI)), "s1")
    etichetta(cy_, Y_BOX[1] - 30, "si riparte da qui", "lbs", set(range(3, STATI)), "s0")

    # il ritorno percorre gli stessi tratti al contrario, spostato dall'altra
    # parte della congiungente
    for tratto, chiave in ((p_su_y, "r1"), (p_giu_y, "r2"),
                           (p_x_su, "r3"), (p_x_giu, "r4")):
        a = acceso(chiave, set(range(4, STATI)), anim)
        b, c = scosta(tratto[1], tratto[0], -SCOSTO)
        for el in freccia(b, c, TEAL, 2.2):
            corpo.append(el.replace("/>", a + "/>", 1))

    for tratto, valore, accensioni, chiave in (
            (p_x_su, ramo_q, set(range(5, STATI)), "rq"),
            (p_x_giu, ramo_l, set(range(6, STATI)), "rl")):
        mx, my = mezzo(*tratto, SCOSTO + 16)
        etichetta(mx, my, n(valore), "ind", accensioni, chiave)

    # --- gli appunti, che diventano il conto del ritorno -------------------
    for box, ap, conto_, chiavi in (
            (SU_NOTA, ap_q, f"1 · 2·{n(ap_q)} = {n(ramo_q)}", ("nq", "cq")),
            (GIU_NOTA, ap_l, f"1 · {n(ap_l)} = {n(ramo_l)}", ("nl", "cl"))):
        bx, by, bw, bh = box
        a_nota = acceso(chiavi[0], set(range(1, 4)), anim)
        corpo.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="6" '
                     f'fill="none" stroke="{OCRA}" stroke-width="1.8" '
                     f'stroke-dasharray="4 4"{a_nota}/>')
        corpo.append(f'<text class="nota" x="{bx + bw / 2:.0f}" y="{by + 20}" '
                     f'text-anchor="middle"{a_nota}>appunto: {n(ap)}</text>')
        a_conto = acceso(chiavi[1], set(range(4, STATI)), anim)
        corpo.append(f'<text class="ind" x="{bx + bw / 2:.0f}" y="{by + 20}" '
                     f'text-anchor="middle"{a_conto}>{conto_}</text>')

    # --- il contatore, che si riempie in due tempi ------------------------
    vx = GRAD_BOX[0] + 14
    vy = GRAD_BOX[1] + 44
    for testo, accensioni, chiave in (
            ("vuoto", set(range(0, 5)), "g0"),
            (n(ramo_q), {5}, "g1"),
            (f"{n(ramo_q)} + {n(ramo_l)} = {n(grad)}", set(range(6, STATI)), "g2")):
        a = acceso(chiave, accensioni, anim)
        corpo.append(f'<text class="ind" x="{vx}" y="{vy}"{a}>{testo}</text>')

    # --- le didascalie dei momenti ---------------------------------------
    momenti = [
        "il conto, prima di farlo: x comanda in due punti",
        "andata: x entra nei due rami",
        "andata: i rami si ritrovano, e y vale " + n(y),
        "ritorno: si riparte dalla fine, dove la derivata vale 1",
        "ritorno: ogni stazione rilegge il proprio appunto e lo consuma",
        f"il ramo del quadrato deposita {n(ramo_q)}",
        f"il ramo lineare deposita {n(ramo_l)}, e si somma al primo",
        f"x.grad vale {n(grad)}: un contributo per ramo, sommati",
    ]
    for m, testo in enumerate(momenti):
        ultimo = (m == len(momenti) - 1)
        t0, t1 = sosta(m, STATI, TENUTA)
        t_fine = 100.0 if ultimo else sosta(m + 1, STATI, TENUTA)[0]
        anim.append(keyframes(f"m{m}", [
            (0.0, "opacity:1" if m == 0 else "opacity:0"),
            (max(t0 - 0.4, 0.01), "opacity:1" if m == 0 else "opacity:0"),
            (t0, "opacity:1"),
            (max(t_fine - 0.4, t0 + 0.1), "opacity:1"),
            (min(t_fine, 100.0), "opacity:1" if ultimo else "opacity:0"),
            (100.0, "opacity:1" if ultimo else "opacity:0")]))
        corpo.append(f'<text class="ttl" x="{LARG / 2:.0f}" y="34" text-anchor="middle" '
                     f'opacity="{1 if ultimo else 0}" '
                     f'style="animation:m{m} var(--d) infinite">{testo}</text>')

    corpo.append(f'<text class="lbs" x="26" y="{ALT - 32}">'
                 f'gli appunti dell\'andata li consuma il ritorno: per riavvolgere una '
                 f'seconda volta</text>')
    corpo.append(f'<text class="lbs" x="26" y="{ALT - 12}">'
                 f'bisogna rifare l\'andata, oppure chiedere prima che restino</text>')

    return Figura(
        larghezza=LARG, altezza=ALT,
        alt="Il grafo del conto y uguale a x al quadrato più due x, con x che "
            "vale 3. A sinistra il riquadro di x, da cui partono due frecce "
            "verso due stazioni: quella in alto moltiplica x per x, quella in "
            "basso moltiplica x per 2; le due si ritrovano nel riquadro di y a "
            "destra. All'andata lungo le frecce compaiono i valori 3 e 3, poi "
            "9 e 6, e y vale 15; sotto ciascuna stazione compare un appunto "
            "tratteggiato, 3 per il quadrato e 2 per il ramo lineare. Al "
            "ritorno le frecce si percorrono al contrario partendo da 1 dal "
            "fondo, e ogni appunto lascia il posto al proprio conto: uno per "
            "due per tre uguale sei nel ramo del quadrato, uno per due uguale "
            "due in quello lineare. In basso a sinistra il contatore x.grad "
            "si riempie in due tempi, prima 6 e poi 6 più 2 uguale 8. Una "
            "riga in fondo ricorda che gli appunti dell'andata li consuma il "
            "ritorno, e che per riavvolgere una seconda volta bisogna rifare "
            "l'andata oppure chiedere prima che restino.",
        corpo="".join(corpo),
        stile=f"""    .lbl {{ font-size:17px; }}
    .av  {{ font-family:{SANS}; font-size:15px; fill:{TERRACOTTA}; font-weight:600; }}
    .ind {{ font-family:{SANS}; font-size:15px; fill:{TEAL}; font-weight:600; }}
    .nota{{ font-family:{SANS}; font-size:13px; fill:{OCRA}; }}""",
        animazioni=anim,
        durata=STATI * 1.25,
        fermi="rect, text, line, path",
    )
