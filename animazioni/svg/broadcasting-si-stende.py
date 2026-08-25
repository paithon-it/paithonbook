"""Il broadcasting si stende, ma i dati restano dove sono.

Il capitolo lo dice a parole (gli stride del lato trasmesso sono posti a zero,
così lo stesso dato viene riletto più volte) e la figura ferma `fig-broadcasting`
mostra il prima e il dopo. Qui c'è il **durante**: la riga scende di riga in
riga, la colonna attraversa le colonne, e dove sono passate resta una coppia di
caselle tratteggiate con dentro i due valori in tono tenue. Quelle caselle sono
letture, non copie: è la differenza che il testo chiede di immaginare.

I numeri sono quelli dell'esempio di `book/Python/numpy.md` (quattro prezzi base
e tre sovrapprezzi) e li calcola NumPy: la somma non è scritta a mano, e
`conti()` verifica anche ciò che la figura afferma, cioè che gli array estesi
abbiano stride nullo sul lato trasmesso. Se un giorno non fosse più vero, la
figura non si genera.

Lo stato di riposo è l'ultimo: la griglia (3,4) piena, i due array veri ancora
sui bordi e le letture ancora riconoscibili come tali. È ciò che finisce in
stampa.
"""

import numpy as np

from paithon_svg import *

NOME = "broadcasting-si-stende"
TITOLO = "il broadcasting si stende, i dati restano dove sono"

# Gli array dell'esempio del capitolo: quattro prezzi base (una riga) e tre
# sovrapprezzi regionali (una colonna).
A = np.array([10, 20, 30, 40])
B = np.array([[1], [2], [3]])

# --------------------------------------------------------------------- misure
CW, CH = 78, 64            # cella del risultato
PX, PY = 86, 74            # passo della griglia
GX, GY = 214, 132          # angolo alto-sinistro della griglia
AY, AH = 64, 48            # la riga a, sopra la griglia
BX = 102                   # la colonna b, a sinistra della griglia
Y_RIS, Y_FASE = 372, 402   # etichetta del risultato, testo di fase
# Qui stavano due righe di chiusura fisse. Erano fisse, quindi comparivano
# IDENTICHE in tutti e tre i fermi immagine della stampa, in corpo minuscolo, e
# ripetevano parola per parola quello che dice gia' la didascalia della pagina.
# Tolte: in figura resta il movimento, il commento sta sotto una volta sola.

# --------------------------------------------------------------- la scaletta
# Quattro tempi: le forme, la riga che scende, la colonna che attraversa, le
# celle che si riempiono. L'ultimo è lo stato di riposo.
T_FORME = (0.0, 8.0)
T_RIGA = (8.0, 38.0)
T_COL = (38.0, 66.0)
T_CELLE = (66.0, 94.0)


def conti():
    """La somma la fa NumPy, e la figura verifica quello che dichiara."""
    assert A.shape == (4,) and B.shape == (3, 1), "gli array di partenza sono cambiati"
    C = A + B
    assert C.shape == (3, 4), f"forma inattesa: {C.shape}"
    a_est, b_est = np.broadcast_arrays(A, B)
    # il punto didattico: sul lato trasmesso lo stride è zero, cioè lo stesso
    # dato viene riletto invece di essere copiato.
    assert 0 in a_est.strides and 0 in b_est.strides, "il lato trasmesso non ha stride nullo"
    return C


def entra(t0: float, fade: float = 2.0):
    """Compare in t0 e resta: il riposo è lo stato finale."""
    q = {0.0: "opacity:0", max(t0 - fade, 0.01): "opacity:0",
         t0: "opacity:1", 100.0: "opacity:1"}
    return sorted(q.items())


def finestra(t0: float, t1: float, fade: float = 1.6):
    """Visibile solo fra t0 e t1: per i testi che si danno il cambio."""
    q = {0.0: f"opacity:{1 if t0 <= 0.01 else 0}"}
    if t0 > 0.01:
        q[max(t0 - fade, 0.01)] = "opacity:0"
    q[t0] = "opacity:1"
    q[t1] = "opacity:1"
    q[min(t1 + fade, 99.8)] = "opacity:0"
    q[100.0] = "opacity:0"
    return sorted(q.items())


def scorre(t0: float, t1: float, salti, verticale: bool, fade: float = 2.5):
    """La copia che si sposta di posto in posto: tappe e istanti di arrivo.

    Parte sovrapposta al dato vero (nessuno spostamento), si ferma su ogni
    posizione della griglia e svanisce. A riposo non c'è: il suo compito è
    mostrare il movimento, e quello che lascia dietro di sé resta.
    """
    n = len(salti)
    passo = (t1 - t0 - 2 * fade) / n

    def tr(d):
        return f"transform:translate({0 if verticale else d}px,{d if verticale else 0}px)"

    q = {0.0: f"opacity:0;{tr(0)}", t0: f"opacity:0;{tr(0)}"}
    t_in = t0 + fade
    q[t_in] = f"opacity:1;{tr(0)}"

    arrivi, prec = [], 0
    for i, d in enumerate(salti):
        inizio = t_in + i * passo
        arrivo = inizio + 0.45 * passo
        q.setdefault(inizio, f"opacity:1;{tr(prec)}")
        q[arrivo] = f"opacity:1;{tr(d)}"
        arrivi.append(arrivo)
        prec = d

    t_out = t_in + n * passo
    q[t_out] = f"opacity:1;{tr(prec)}"
    q[min(t_out + fade, 99.8)] = f"opacity:0;{tr(prec)}"
    q[100.0] = f"opacity:0;{tr(prec)}"
    return sorted(q.items()), arrivi


def costruisci() -> Figura:
    C = conti()
    n_righe, n_colonne = C.shape
    colonne = [GX + j * PX for j in range(n_colonne)]
    righe = [GY + i * PY for i in range(n_righe)]
    corpo, anim = [], []

    corpo.append('<text class="ttl" x="30" y="28">le forme si stendono, '
                 'i dati restano dove sono</text>')

    # ------------------------------------------------- lo scheletro della griglia
    for y in righe:
        for x in colonne:
            corpo.append(f'<rect class="vuo" x="{x}" y="{y}" '
                         f'width="{CW}" height="{CH}" rx="5"/>')

    # ------------------------------- le celle del risultato, una dopo l'altra
    passo_cella = (T_CELLE[1] - T_CELLE[0]) / (n_righe * n_colonne)
    for i in range(n_righe):
        for j in range(n_colonne):
            k = i * n_colonne + j
            anim.append(keyframes(f"ce{k}", entra(T_CELLE[0] + k * passo_cella, fade=1.5)))
            x, y = colonne[j], righe[i]
            corpo.append(
                f'<g class="cel" style="animation:ce{k} var(--d) infinite">'
                f'<rect class="pie" x="{x}" y="{y}" width="{CW}" height="{CH}" rx="5"/>'
                f'<text class="som" x="{x + CW / 2:.0f}" y="{y + 52}" '
                f'text-anchor="middle">{C[i, j]}</text></g>')

    # --------------------------------------------------- le letture di a e di b
    # Una casella tratteggiata per ogni valore riletto: quattro per riga e tre
    # per colonna, dodici in tutto da sette numeri veri.
    tappe_a, arrivi_a = scorre(*T_RIGA, [y - AY for y in righe], verticale=True)
    tappe_b, arrivi_b = scorre(*T_COL, [x - BX for x in colonne], verticale=False)
    anim.append(keyframes("scoA", tappe_a))
    anim.append(keyframes("scoB", tappe_b))

    for i, y in enumerate(righe):
        anim.append(keyframes(f"ga{i}", entra(arrivi_a[i], fade=1.5)))
        pezzi = []
        for j, x in enumerate(colonne):
            pezzi.append(f'<rect class="lea" x="{x + 2}" y="{y + 7}" '
                         f'width="30" height="20" rx="3"/>')
            pezzi.append(f'<text class="vla" x="{x + 17}" y="{y + 22}" '
                         f'text-anchor="middle">{A[j]}</text>')
        corpo.append(f'<g class="gh" style="animation:ga{i} var(--d) infinite">'
                     f'{"".join(pezzi)}</g>')

    for j, x in enumerate(colonne):
        anim.append(keyframes(f"gb{j}", entra(arrivi_b[j], fade=1.5)))
        pezzi = []
        for i, y in enumerate(righe):
            pezzi.append(f'<text class="piu" x="{x + 39}" y="{y + 22}" '
                         f'text-anchor="middle">+</text>')
            pezzi.append(f'<rect class="leb" x="{x + 46}" y="{y + 7}" '
                         f'width="30" height="20" rx="3"/>')
            pezzi.append(f'<text class="vlb" x="{x + 61}" y="{y + 22}" '
                         f'text-anchor="middle">{B[i, 0]}</text>')
        corpo.append(f'<g class="gh" style="animation:gb{j} var(--d) infinite">'
                     f'{"".join(pezzi)}</g>')

    # ------------------------------------------------------- i due array veri
    for j, x in enumerate(colonne):
        corpo.append(f'<rect class="rea" x="{x}" y="{AY}" '
                     f'width="{CW}" height="{AH}" rx="5"/>')
        corpo.append(f'<text class="dato" x="{x + CW / 2:.0f}" y="{AY + 31}" '
                     f'text-anchor="middle">{A[j]}</text>')
    for i, y in enumerate(righe):
        corpo.append(f'<rect class="reb" x="{BX}" y="{y}" '
                     f'width="{CW}" height="{CH}" rx="5"/>')
        corpo.append(f'<text class="dato" x="{BX + CW / 2:.0f}" y="{y + 40}" '
                     f'text-anchor="middle">{B[i, 0]}</text>')

    corpo.append(f'<text class="nome noma" x="{GX - 12}" y="{AY + 35}" '
                 f'text-anchor="end">a (4,)</text>')
    corpo.append(f'<text class="nome nomb" x="{BX - 12}" y="{(righe[0] + righe[-1] + CH) / 2 + 5:.0f}" '
                 f'text-anchor="end">b (3,1)</text>')
    corpo.append(f'<text class="nome" x="{(GX + colonne[-1] + CW) / 2:.0f}" y="{Y_RIS}" '
                 f'text-anchor="middle">a + b (3,4)</text>')

    # ----------------------------------------- le copie che passano e se ne vanno
    pezzi = []
    for j, x in enumerate(colonne):
        pezzi.append(f'<rect class="rea" x="{x}" y="{AY}" '
                     f'width="{CW}" height="{AH}" rx="5"/>')
        pezzi.append(f'<text class="dato" x="{x + CW / 2:.0f}" y="{AY + 31}" '
                     f'text-anchor="middle">{A[j]}</text>')
    corpo.append(f'<g class="sli" opacity="0" style="animation:scoA var(--d) infinite">'
                 f'{"".join(pezzi)}</g>')

    pezzi = []
    for i, y in enumerate(righe):
        pezzi.append(f'<rect class="reb" x="{BX}" y="{y}" '
                     f'width="{CW}" height="{CH}" rx="5"/>')
        pezzi.append(f'<text class="dato" x="{BX + CW / 2:.0f}" y="{y + 40}" '
                     f'text-anchor="middle">{B[i, 0]}</text>')
    corpo.append(f'<g class="sli" opacity="0" style="animation:scoB var(--d) infinite">'
                 f'{"".join(pezzi)}</g>')

    # ------------------------------------------------------- il testo che racconta
    fasi = [(T_FORME, "a ha forma (4,), b ha forma (3,1): gli assi si allineano da destra"),
            (T_RIGA, "la riga si stende verso il basso"),
            (T_COL, "la colonna si stende verso destra"),
            (T_CELLE, "ogni cella somma i due valori che legge")]
    for k, ((t0, t1), testo) in enumerate(fasi):
        ultimo = k == len(fasi) - 1
        anim.append(keyframes(f"fa{k}", entra(t0) if ultimo else finestra(t0, t1)))
        corpo.append(f'<text class="fase" x="30" y="{Y_FASE}" opacity="{1 if ultimo else 0}" '
                     f'style="animation:fa{k} var(--d) infinite">{testo}</text>')

    return Figura(
        larghezza=590, altezza=424,
        alt="Una riga di quattro numeri (10, 20, 30, 40) sta sopra una griglia "
            "vuota di tre righe per quattro colonne, e una colonna di tre numeri "
            "(1, 2, 3) sta alla sua sinistra. La riga scende di riga in riga e "
            "lascia in ogni cella una casella tratteggiata con il proprio valore; "
            "poi la colonna attraversa le colonne e lascia accanto la seconda "
            "casella tratteggiata. Infine ogni cella si riempie con la somma dei "
            "due valori che legge, dando la matrice tre per quattro da 11 a 43. "
            "I due array di partenza restano al loro posto: le caselle "
            "tratteggiate sono letture ripetute dello stesso dato, non copie in "
            "memoria, perché sul lato trasmesso lo stride vale zero.",
        corpo="".join(corpo),
        stile=f"""    .vuo  {{ fill:none; stroke:{BORDER}; stroke-width:1.4; stroke-dasharray:5 5; }}
    .pie  {{ fill:{OCRA}; fill-opacity:0.16; stroke:{INK}; stroke-width:1.8; }}
    .som  {{ font-family:{SANS}; font-size:21px; font-weight:700; fill:{INK}; }}
    .lea  {{ fill:none; stroke:{TEAL}; stroke-width:1.3; stroke-dasharray:3 3; opacity:0.7; }}
    .leb  {{ fill:none; stroke:{TERRACOTTA}; stroke-width:1.3; stroke-dasharray:3 3; opacity:0.7; }}
    .vla  {{ font-family:{SANS}; font-size:12px; fill:{TEAL}; opacity:0.85; }}
    .vlb  {{ font-family:{SANS}; font-size:12px; fill:{TERRACOTTA}; opacity:0.85; }}
    .piu  {{ font-family:{SANS}; font-size:11px; fill:{FG_MUTED}; }}
    .rea  {{ fill:{CREAM}; stroke:{TEAL}; stroke-width:2.4; }}
    .reb  {{ fill:{CREAM}; stroke:{TERRACOTTA}; stroke-width:2.4; }}
    .dato {{ font-family:{SANS}; font-size:19px; font-weight:700; fill:{INK}; }}
    .nome {{ font-family:{SANS}; font-size:14px; font-weight:700; fill:{INK}; }}
    .noma {{ fill:{TEAL}; }}
    .nomb {{ fill:{TERRACOTTA}; }}
    .fase {{ font-family:{SANS}; font-size:15px; font-weight:700; fill:{TERRACOTTA}; }}
    .sli  {{ transform-box:view-box; }}""",
        animazioni=anim,
        durata=11.0,
        fermi=".cel, .gh, .sli, .fase",
    )
