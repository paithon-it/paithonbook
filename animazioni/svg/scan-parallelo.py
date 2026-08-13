"""Lo scan: la stessa ricorrenza in fila oppure a raddoppio.

Il tempo qui è il contenuto due volte, ed è per questo che le due strade stanno
sullo stesso orologio: l'asse verticale è il *turno*, uguale per entrambe. A
sinistra la catena avanza di una posizione per turno e ne consuma
$L - 1$; a destra ogni turno raddoppia il salto (1, 2, 4, 8) e dopo
$\\lceil \\log_2 L \\rceil$ turni ha finito. Con la lunghezza dell'esempio di
`StateSpaceModel/mamba.md` (L = 12) fanno 11 turni contro 4, e nell'animazione
il pannello destro sta fermo per gli altri sette: è tutto il punto.

Lo scan **gira davvero**, in tutte e due le versioni, con l'operatore vero
della ricorrenza lineare, quello scritto nel capitolo:

    (a1, b1) . (a2, b2) = (a2 a1, a2 b1 + b2)

che è associativo (e non commutativo). La versione a raddoppio è quella
inclusiva di Hillis e Steele, cioè esattamente la `scan_parallelo` che il
capitolo stampa qualche riga più su: `salto` che raddoppia, e a ogni giro
`b[salto:] = a[salto:] * b_prec + b[salto:]`. Non è la variante di Blelloch (a
due passate, che risparmia lavoro ma raddoppia i turni): quella non è il codice
del capitolo, e chi confronta la figura con la pagina troverebbe due algoritmi
diversi.

Le guardie sono tre, e la prima è la ragione per cui la figura esiste: i due
scan devono dare lo **stesso risultato elemento per elemento** (verificato in
aritmetica esatta, con le frazioni, così non resta il dubbio del confronto fra
numeri in virgola mobile; e una seconda volta in virgola mobile, sui parametri
del capitolo, dove si chiede solo che lo scarto sia sotto 1e-12). Se non
coincidessero, la figura sarebbe una bugia e non si genera. Le altre due sono
il conto dei turni ($\\lceil \\log_2 12 \\rceil = 4$) e la legge del raddoppio:
quante posizioni hanno il risultato definitivo dopo ogni turno non si scrive a
mano, si ricava facendo girare lo stesso scan su un secondo monoide, quello
degli intervalli, che dice quali passi originali ciascuna posizione ha dentro.
"""

import math
import random
from fractions import Fraction

from paithon_svg import *

NOME = "scan-parallelo"
TITOLO = "lo scan in fila e lo scan a raddoppio"

L = 12          # la lunghezza di sequenza dell'esempio di mamba.md (L, N = 12, 4)
SEME = 7

# ---------------------------------------------------------------------------
# L'algoritmo, che gira davvero
# ---------------------------------------------------------------------------


def comporre(p, q):
    """(a1, b1) . (a2, b2) = (a2 a1, a2 b1 + b2), con p che viene *prima* di q.

    È l'operatore della ricorrenza h_t = a_t h_{t-1} + b_t: comporre due passi
    dà un passo dello stesso tipo. Associativo, non commutativo.
    """
    (a1, b1), (a2, b2) = p, q
    return (a2 * a1, a2 * b1 + b2)


def unisci(p, q):
    """Lo stesso operatore su un altro monoide: gli intervalli di passi coperti.

    Serve a sapere, senza scriverlo a mano, quali passi originali ciascuna
    posizione ha già dentro di sé. E fa da guardia: se lo scan componesse due
    tratti non contigui, o nell'ordine sbagliato, qui salterebbe.
    """
    (i1, f1), (i2, f2) = p, q
    if f1 + 1 != i2:
        raise AssertionError(f"tratti non contigui: {p} prima di {q}")
    return (i1, f2)


def in_fila(passi, op=comporre):
    """Da sinistra a destra, un passo alla volta: L - 1 composizioni in fila.

    Ognuna aspetta la precedente, quindi le composizioni sono anche i turni.
    """
    acc, stati, fatte = passi[0], [passi[0]], 0
    for p in passi[1:]:
        acc = op(acc, p)
        fatte += 1
        stati.append(acc)
    return stati, fatte


def a_raddoppio(passi, op=comporre):
    """Scan inclusivo a raddoppio (Hillis-Steele), lo stesso di `scan_parallelo`.

    A ogni turno ogni posizione si compone con quella distante `salto`, e il
    salto raddoppia: 1, 2, 4, 8. Dentro un turno le composizioni non dipendono
    l'una dall'altra, quindi si fanno tutte insieme.
    """
    cur, livelli, fatte = list(passi), [], 0
    salto = 1
    while salto < len(cur):
        nuovo = list(cur)
        for i in range(salto, len(cur)):
            nuovo[i] = op(cur[i - salto], cur[i])
            fatte += 1
        cur = nuovo
        livelli.append((salto, cur))
        salto *= 2
    return cur, livelli, fatte


def passi_esatti():
    """I passi (a_t, b_t) in aritmetica esatta: a_t in (0, 1), b_t qualunque."""
    rnd = random.Random(SEME)
    return [(Fraction(rnd.randrange(3, 10), 10), Fraction(rnd.randrange(-9, 10), 4))
            for _ in range(L)]


def passi_float():
    """Gli stessi passi come li produce Mamba: a_t = exp(delta_t A), b_t = delta_t B_t x_t."""
    rnd = random.Random(SEME + 1)
    fuori = []
    for _ in range(L):
        A = -(rnd.random() + 0.5)              # autovalore negativo, come nel capitolo
        delta = rnd.random() * 0.5 + 0.1       # passo di discretizzazione, positivo
        fuori.append((math.exp(delta * A), delta * rnd.gauss(0, 1) * rnd.gauss(0, 1)))
    return fuori


def scan():
    """Fa girare i due scan, li confronta, e restituisce ciò che la figura disegna."""
    passi = passi_esatti()
    fila, turni_fila = in_fila(passi)
    albero, livelli, lavoro_albero = a_raddoppio(passi)

    # la proprietà per cui la figura esiste: elemento per elemento, esatto.
    for t, (u, v) in enumerate(zip(fila, albero), start=1):
        if u != v:
            raise AssertionError(f"i due scan divergono alla posizione {t}: {u} != {v}")

    # e una seconda volta in virgola mobile, sui parametri del capitolo
    fila_f, _ = in_fila(passi_float())
    albero_f, _, _ = a_raddoppio(passi_float())
    scarto = max(abs(u[1] - v[1]) for u, v in zip(fila_f, albero_f))
    if scarto > 1e-12:
        raise AssertionError(f"in virgola mobile i due scan divergono di {scarto:.2e}")

    if len(livelli) != math.ceil(math.log2(L)):
        raise AssertionError(f"{len(livelli)} turni invece di ceil(log2 {L})")
    if turni_fila != L - 1:
        raise AssertionError(f"{turni_fila} turni in fila invece di {L - 1}")

    # quanti risultati sono definitivi dopo ogni turno: non si scrive, si conta
    # facendo girare lo stesso scan sugli intervalli.
    tratti = [(i, i) for i in range(L)]
    _, livelli_tr, _ = a_raddoppio(tratti, unisci)
    fronte = [sum(1 for t in stati if t[0] == 0) for _, stati in livelli_tr]
    if fronte != [min(2 ** k, L) for k in range(1, len(livelli) + 1)]:
        raise AssertionError(f"il fronte non raddoppia: {fronte}")

    salti = [s for s, _ in livelli]
    definitivi = [[t[0] == 0 for t in stati] for _, stati in livelli_tr]
    return turni_fila, len(livelli), salti, definitivi, lavoro_albero


# ---------------------------------------------------------------------------
# Il disegno
# ---------------------------------------------------------------------------
PX, PY, R = 20, 32, 6           # passo fra le posizioni, fra i turni, raggio
X1, X2 = 64, 344                # colonna 0 dei due pannelli
TOP = 100                       # riga del turno 0


def costruisci() -> Figura:
    turni_fila, turni_albero, salti, definitivi, _ = scan()
    righe = turni_fila + 1                      # turno 0 (partenza) + i turni

    def cx(x0, i):
        return x0 + i * PX

    def cy(r):
        return TOP + r * PY

    corpo, anim, stile = [], [], []
    largh = cx(X2, L - 1) + 100

    # --- la riga del turno corrente, dietro a tutto: un solo orologio per due
    anim.append(keyframes("scorre", [
        v for r in range(righe)
        for t in sosta(r, righe, tenuta=0.72)
        for v in [(t, f"transform:translateY({(r - turni_fila) * PY}px)")]] +
        [(100.0, "transform:translateY(0px)")]))
    corpo.append(f'<rect class="band" x="14" y="{cy(turni_fila) - 12}" '
                 f'width="{cx(X2, L - 1) + 18 - 14}" height="24" rx="5"/>')

    # --- i binari: le 12 posizioni della sequenza, in tutti e due i pannelli
    corpo.append('<path class="rail" d="' + "".join(
        f"M{cx(x0, i)} {cy(0)}V{cy(ultimo)}"
        for x0, ultimo in ((X1, turni_fila), (X2, turni_albero))
        for i in range(L)) + '"/>')

    # --- comparse: un @keyframes per turno, gli elementi lo condividono
    for k in range(1, righe):
        t0, _ = sosta(k, righe)
        anim.append(keyframes(f"ap{k}", [
            (0.0, "opacity:0"), (max(t0 - 0.4, 0.01), "opacity:0"),
            (t0, "opacity:1"), (100.0, "opacity:1")]))
        stile.append(f".k{k}{{animation:ap{k} var(--d) infinite}}")

    def barra(x0, r, quanti, cls):
        """Il fronte: fin dove arriva, dopo questo turno, il risultato definitivo."""
        largo = max((quanti - 1) * PX, 5)
        corpo.append(f'<rect class="{cls}" x="{cx(x0, 0) - 3}" y="{cy(r) + R + 5}" '
                     f'width="{largo + 6}" height="3" rx="1.5"/>')

    # --- pannello di sinistra: in fila, una posizione per turno
    corpo.append('<g class="n">')
    for i in range(1, L):
        corpo.append(f'<circle cx="{cx(X1, i)}" cy="{cy(0)}" r="{R}"/>')
    corpo.append(f'</g><g class="n d"><circle cx="{cx(X1, 0)}" cy="{cy(0)}" r="{R}"/></g>')
    barra(X1, 0, 1, "fronte")
    for t in range(1, righe):
        corpo.append(f'<line class="c k{t}" x1="{cx(X1, t - 1)}" y1="{cy(t - 1) + R}" '
                     f'x2="{cx(X1, t)}" y2="{cy(t) - R}"/>')
        corpo.append(f'<circle class="n d k{t}" cx="{cx(X1, t)}" cy="{cy(t)}" r="{R}"/>')
        barra(X1, t, t + 1, f"fronte k{t}")

    # --- pannello di destra: a raddoppio, il salto che raddoppia a ogni turno
    corpo.append('<g class="n">')
    for i in range(1, L):
        corpo.append(f'<circle cx="{cx(X2, i)}" cy="{cy(0)}" r="{R}"/>')
    corpo.append(f'</g><g class="n d"><circle cx="{cx(X2, 0)}" cy="{cy(0)}" r="{R}"/></g>')
    barra(X2, 0, 1, "fronte")
    for k, (salto, pieni) in enumerate(zip(salti, definitivi), start=1):
        corpo.append(f'<path class="c k{k}" d="' + "".join(
            f"M{cx(X2, i - salto)} {cy(k - 1) + R}L{cx(X2, i)} {cy(k) - R}"
            for i in range(salto, L)) + f'"/><g class="n k{k}">')
        for i in range(L):
            if not pieni[i]:
                corpo.append(f'<circle cx="{cx(X2, i)}" cy="{cy(k)}" r="{R}"/>')
        corpo.append(f'</g><g class="n d k{k}">')
        for i in range(L):
            if pieni[i]:
                corpo.append(f'<circle cx="{cx(X2, i)}" cy="{cy(k)}" r="{R}"/>')
        corpo.append('</g>')
        barra(X2, k, sum(pieni), f"fronte k{k}")
        corpo.append(f'<text class="lbs k{k}" x="{cx(X2, L - 1) + 18}" '
                     f'y="{cy(k) + 5}">salto {salto}</text>')

    # --- l'orologio comune e le intestazioni
    corpo.append(f'<text class="lbs" x="46" y="{cy(0) - 22}" text-anchor="end">turno</text>')
    for r in range(righe):
        corpo.append(f'<text class="lbs" x="46" y="{cy(r) + 5}" text-anchor="end">{r}</text>')

    for x0, titolo, quanti, sotto in (
            (X1, "in fila", turni_fila, "una composizione per turno"),
            (X2, "a raddoppio", turni_albero, "tante composizioni per turno")):
        mezzo = (cx(x0, 0) + cx(x0, L - 1)) // 2
        corpo += [
            f'<text class="ttl" x="{mezzo}" y="{cy(0) - 68}" '
            f'text-anchor="middle">{titolo}</text>',
            f'<text class="cnt k{quanti}" x="{mezzo}" y="{cy(0) - 44}" '
            f'text-anchor="middle">{quanti} turni</text>',
            f'<text class="lbs" x="{mezzo}" y="{cy(0) - 22}" '
            f'text-anchor="middle">{sotto}</text>',
        ]

    # --- il pannello destro ha finito, e resta a guardare
    for j, riga in enumerate(("qui è già tutto calcolato:",
                              f"i turni da {turni_albero + 1} a {turni_fila} non servono")):
        corpo.append(f'<text class="lbs k{turni_albero}" x="{cx(X2, 0)}" '
                     f'y="{cy(turni_albero) + 62 + j * 20}">{riga}</text>')

    # --- legenda e operatore
    base = cy(turni_fila) + 44
    corpo += [
        f'<g class="n"><circle cx="{X1}" cy="{base - 5}" r="{R}"/></g>',
        f'<text class="lbs" x="{X1 + 14}" y="{base}">risultato parziale</text>',
        f'<g class="n d"><circle cx="{X1 + 168}" cy="{base - 5}" r="{R}"/></g>',
        f'<text class="lbs" x="{X1 + 182}" y="{base}">risultato definitivo</text>',
        f'<text class="lbl" x="{X1}" y="{base + 32}">'
        f'(a₁, b₁) • (a₂, b₂) = (a₂ a₁, a₂ b₁ + b₂)</text>',
        f'<text class="lbs" x="{X1}" y="{base + 56}">l\'operazione è associativa, '
        f'quindi i passi si possono raggruppare a piacere:</text>',
        f'<text class="lbs" x="{X1}" y="{base + 76}">dentro un turno le composizioni '
        f'non dipendono l\'una dall\'altra e si fanno tutte insieme.</text>',
    ]

    return Figura(
        larghezza=largh, altezza=base + 92,
        alt=f"Due schemi affiancati della stessa ricorrenza su {L} passi, con lo stesso "
            f"asse verticale dei turni. A sinistra, in fila: una scala di frecce scende "
            f"in diagonale, una posizione per turno, e ne servono {turni_fila}. A destra, "
            f"a raddoppio: {turni_albero} righe di frecce in cui il salto raddoppia "
            f"(1, 2, 4, 8), e già dopo {turni_albero} turni tutte le posizioni hanno il "
            f"risultato definitivo; le righe sotto restano vuote.",
        corpo="".join(corpo),
        stile=f"""    .rail   {{ stroke:{BORDER}; stroke-width:1; fill:none; }}
    .band   {{ fill:{BORDER}; fill-opacity:.5;
              animation:scorre var(--d) infinite; }}
    .n      {{ fill:{CREAM}; stroke:{TEAL}; stroke-width:2; }}
    .d      {{ fill:{TERRACOTTA}; stroke:{TERRACOTTA}; }}
    .c      {{ stroke:{FG_MUTED}; stroke-width:1.3; fill:none; }}
    .fronte {{ fill:{TERRACOTTA}; fill-opacity:.35; }}
    .cnt    {{ font-family:{SANS}; font-size:17px; font-weight:700;
              fill:{TERRACOTTA}; }}
{chr(10).join("    " + s for s in stile)}""",
        animazioni=anim,
        durata=righe * 0.8,
        fermi=".band, .c, .n, .fronte, .cnt, .lbs",
    )
