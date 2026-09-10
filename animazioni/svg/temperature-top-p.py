"""Le due manopole in fila: la temperatura ripesa, poi il top_p taglia.

La sezione «Le due manopole del campionamento» dice una cosa sola, e la dice
contro l'intuizione comune: temperatura e top_p **non agiscono in parallelo**.
La prima ripesa tutta la classifica, la seconda taglia su quella *già ripesata*,
sicché a soglia ferma il taglio cade in un altro punto. Il tempo, qui, è il
contenuto: una figura a pannelli affiancati mostra i due gesti l'uno accanto
all'altro, che è esattamente la lettura sbagliata.

Sei tappe, e sono un ciclo solo:

1. la classifica di partenza, a $T = 1$;
2. il taglio a $p = 0{,}9$: la cumulata arriva a 91 alla terza riga, quindi il
   nucleo è di tre;
3. la massa si ridistribuisce sui tre superstiti;
4. $T = 2$: la stessa classifica ripesata, i distacchi si accorciano;
5. **la stessa** soglia: adesso alla terza riga la cumulata fa 80, e per
   arrivare a novanta serve anche la quarta;
6. la massa si ridistribuisce sui quattro.

Nessun numero è scritto a mano. Le percentuali di partenza sono quelle della
sezione (45, 30, 16, 6, 3); tutto il resto lo calcola il generatore, e in
particolare la ripesatura, che per la temperatura $T$ è $p_i^{1/T}$
rinormalizzata, cioè la softmax sui logit divisi per $T$ scritta in funzione
delle sole probabilità.

**Le percentuali stampate sommano a 100 per costruzione.** Non è un dettaglio:
arrotondando ciascuna al più vicino, la classifica a $T = 2$ dà 33, 27, 20, 12
e 9, che fanno 101, ed è quello che la figura ferma di prima stampava. Qui
l'arrotondamento è a resto massimo (`arrotonda_a_cento`), le cumulate si
ricavano dalle percentuali già arrotondate, e un `assert` pretende sia la somma
sia che il taglio dichiarato segua dalle cumulate stampate. Così quello che il
lettore somma con gli occhi torna, e la didascalia non può mentire.
"""

from paithon_svg import *

NOME = "temperature-top-p"
TITOLO = "Le due manopole in fila: la temperatura ripesa, il top_p taglia"

# --------------------------------------------------------------------------
# I numeri
# --------------------------------------------------------------------------
PARTENZA = (0.45, 0.30, 0.16, 0.06, 0.03)   # la classifica della sezione
SOGLIA = 0.9                                # il top_p
T_FREDDA, T_CALDA = 1.0, 2.0


def ripesa(p, temperatura):
    """La classifica a temperatura T: $p_i^{1/T}$ rinormalizzata.

    E' la softmax sui logit divisi per T scritta senza i logit: se
    $p_i \\propto e^{z_i}$, allora $e^{z_i/T} = (e^{z_i})^{1/T} \\propto
    p_i^{1/T}$. Vale per qualunque insieme di logit che dia quelle $p_i$,
    perche' una costante additiva sui logit sparisce nella rinormalizzazione.
    """
    grezze = [x ** (1.0 / temperatura) for x in p]
    totale = sum(grezze)
    return [x / totale for x in grezze]


def nucleo(p, soglia):
    """Quante voci servono perche' la cumulata raggiunga la soglia."""
    somma = 0.0
    for i, x in enumerate(p, start=1):
        somma += x
        if somma >= soglia:
            return i
    return len(p)


def arrotonda_a_cento(p):
    """Percentuali intere che sommano esattamente a 100 (resto massimo).

    Arrotondare ciascuna per conto suo da' 101 sulla classifica calda, ed e'
    il difetto che questa figura esiste anche per togliere: il lettore somma
    le etichette e trova un numero che non esiste.
    """
    esatte = [x * 100 for x in p]
    interi = [int(x) for x in esatte]
    mancano = 100 - sum(interi)
    ordine = sorted(range(len(p)), key=lambda i: -(esatte[i] - interi[i]))
    for i in ordine[:mancano]:
        interi[i] += 1
    return interi


def cumulate(interi):
    """Le somme parziali delle percentuali *stampate*, non di quelle esatte."""
    fuori, somma = [], 0
    for x in interi:
        somma += x
        fuori.append(somma)
    return fuori


def tappe():
    """Le sei tappe, ciascuna con quello che la figura deve disegnare."""
    fredda, calda = PARTENZA, ripesa(PARTENZA, T_CALDA)
    k_fredda, k_calda = nucleo(fredda, SOGLIA), nucleo(calda, SOGLIA)

    def ridistribuisci(p, k):
        totale = sum(p[:k])
        return [x / totale if i < k else 0.0 for i, x in enumerate(p)]

    return [
        dict(p=fredda, k=None, T=T_FREDDA,
             titolo=f"temperatura {num(T_FREDDA, 0)}",
             riga="la classifica sulla parola dopo, com'è"),
        dict(p=fredda, k=k_fredda, T=T_FREDDA,
             titolo=f"il top_p taglia a {num(SOGLIA)}",
             riga=f"alla riga {k_fredda} la cumulata fa "
                  f"{cumulate(arrotonda_a_cento(fredda))[k_fredda - 1]}, e "
                  f"novanta è passato"),
        dict(p=ridistribuisci(fredda, k_fredda), k=k_fredda, T=T_FREDDA,
             titolo="e la massa si ridistribuisce",
             riga=f"si pesca fra {PAROLE[k_fredda]}, e sono di nuovo cento"),
        dict(p=calda, k=None, T=T_CALDA,
             titolo=f"temperatura {num(T_CALDA, 0)}: si ripesa tutto",
             riga="stessa classifica, distacchi più corti; nessuno escluso"),
        dict(p=calda, k=k_calda, T=T_CALDA,
             titolo=f"la stessa soglia, {num(SOGLIA)}",
             riga=f"adesso alla riga {k_fredda} la cumulata non basta più"),
        dict(p=ridistribuisci(calda, k_calda), k=k_calda, T=T_CALDA,
             titolo=f"temperatura {num(T_CALDA, 0)}: il nucleo è di "
                    f"{PAROLE[k_calda]}",
             riga=f"stessa soglia {num(SOGLIA)}, e la massa si è "
                  f"ridistribuita sui {PAROLE[k_calda]}"),
    ]


def verifica(vista):
    """Quello che la didascalia promette, preteso sui numeri stampati."""
    fredda, calda = PARTENZA, ripesa(PARTENZA, T_CALDA)
    k_fredda, k_calda = nucleo(fredda, SOGLIA), nucleo(calda, SOGLIA)

    assert abs(sum(PARTENZA) - 1.0) < 1e-12, "la classifica di partenza non fa 1"
    assert list(PARTENZA) == sorted(PARTENZA, reverse=True), \
        "la classifica non e' ordinata"
    assert k_calda == k_fredda + 1, (
        f"il nucleo passa da {k_fredda} a {k_calda}: la figura dice «uno in "
        f"piu'», e senza quello non racconta niente")

    # I distacchi si accorciano: e' la mossa della temperatura.
    assert calda[0] < fredda[0] and calda[-1] > fredda[-1], \
        "alzando la temperatura la classifica non si e' appiattita"

    for t in vista:
        interi = arrotonda_a_cento([x for x in t["p"] if x > 0])
        assert sum(interi) == 100, \
            f"«{t['titolo']}»: le percentuali stampate fanno {sum(interi)}"

    # Il taglio dichiarato deve seguire dalle cumulate *stampate*, o la riga
    # in fondo dice una cosa e le etichette un'altra.
    for p, k in ((fredda, k_fredda), (calda, k_calda)):
        cum = cumulate(arrotonda_a_cento(p))
        assert cum[k - 1] >= SOGLIA * 100, \
            f"la cumulata stampata alla riga {k} fa {cum[k - 1]}, sotto la soglia"
        if k > 1:
            assert cum[k - 2] < SOGLIA * 100, \
                f"bastava gia' la riga {k - 1}: cumulata stampata {cum[k - 2]}"


# --------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------
LARG, ALT = 700, 404

N = len(PARTENZA)
X_RIGA = 62                      # il numero d'ordine della riga
X_BAR = 84                       # dove partono le barre
# La barra piu' larga che la figura disegna e' meno della meta' del totale
# (la prima voce dopo la ridistribuzione a T = 1), quindi la scala del cento
# per cento vale il doppio della larghezza che si vede.
W_PIENA = 560                    # la larghezza del cento per cento
H_BAR = 30
PASSO_Y = 44
Y0 = 76                          # centro della prima riga

X_PCT = 392                      # la percentuale, dopo la barra piu' lunga
X_CUM = 486                      # la colonna delle cumulate
X_TAG = 560                      # dove finisce la linea del taglio

Y_TIT = 316                      # il titolo della tappa
Y_RIG = 340                      # la riga che spiega
Y_FIN = 380                      # la riga fissa in fondo

PAROLE = ("zero", "una", "due", "tre", "quattro", "cinque")


def num(v: float, cifre: int = 1) -> str:
    return f"{v:.{cifre}f}".replace(".", ",")


def cy(riga: int) -> float:
    return Y0 + riga * PASSO_Y


def y_taglio(k: int) -> float:
    """La linea cade fra l'ultima riga dentro e la prima fuori."""
    return cy(k - 1) + PASSO_Y / 2


# --------------------------------------------------------------------------
# La figura
# --------------------------------------------------------------------------
def costruisci() -> Figura:
    vista = tappe()
    verifica(vista)
    n = len(vista)
    passo = 100.0 / n

    corpo, anim = [], []

    corpo.append(f'<text class="lbs" x="{X_BAR}" y="38">'
                 f'i candidati, ordinati per probabilità</text>')
    corpo.append(f'<text class="lbs" x="{X_CUM}" y="38">cumulata</text>')

    # ---- le cinque barre: a riposo lo stato finale, l'animazione lo precede
    finale = vista[-1]
    for i in range(N):
        larghezze, opacita, colori, pct, cum = [], [], [], [], []
        for t in vista:
            vivi = [x for x in t["p"] if x > 0]
            interi = arrotonda_a_cento(vivi)
            somme = cumulate(interi)
            dentro = i < len(vivi)
            larghezze.append(f"width:{t['p'][i] * W_PIENA:.1f}px")
            opacita.append("opacity:1" if dentro else "opacity:0")
            colori.append(TEAL if (t["k"] is None or i < t["k"]) else FG_MUTED)
            pct.append(f"{interi[i]}%" if dentro else "")
            cum.append(f"{somme[i]}" if dentro else "")

        vivo_fine = i < len([x for x in finale["p"] if x > 0])

        # la barra: si anima `width`, non una scaleX, o gli angoli arrotondati
        # si stirerebbero in ellissi
        anim.append(keyframes(f"bar{i}", scorre(larghezze, passo)))
        anim.append(keyframes(f"col{i}", scorre(
            [f"fill:{c}" for c in colori], passo)))
        corpo.append(
            f'<rect class="bar" x="{X_BAR}" y="{cy(i) - H_BAR / 2:.0f}" '
            f'width="{finale["p"][i] * W_PIENA:.1f}" height="{H_BAR}" rx="4" '
            f'fill="{TEAL if vivo_fine else FG_MUTED}" '
            f'style="animation:bar{i} var(--d) infinite, '
            f'col{i} var(--d) infinite"/>')

        # il numero d'ordine, la percentuale e la cumulata
        anim.append(keyframes(f"viv{i}", scorre(opacita, passo)))
        moto = (f' style="animation:viv{i} var(--d) infinite;'
                f'opacity:{1 if vivo_fine else 0}"')
        corpo.append(f'<text class="ord" x="{X_RIGA}" y="{cy(i) + 5:.0f}" '
                     f'text-anchor="end"{moto}>{i + 1}</text>')

        for s in range(n):
            if not pct[s]:
                continue
            anim.append(keyframes(f"p{i}_{s}", accende(s, n, passo)))
            fermo = ";opacity:1" if s == n - 1 else ""
            m = f' style="animation:p{i}_{s} var(--d) infinite{fermo}"'
            corpo.append(f'<text class="pct" x="{X_PCT}" y="{cy(i) + 6:.0f}"'
                         f'{m}>{pct[s]}</text>')
            corpo.append(f'<text class="cum" x="{X_CUM}" y="{cy(i) + 6:.0f}"'
                         f'{m}>{cum[s]}</text>')

    # ---- la linea del taglio: e' lei che si sposta, ed e' il punto ---------
    tagli = [t["k"] for t in vista]
    posizioni = [y_taglio(k) if k else y_taglio(N) for k in tagli]
    visibile = ["opacity:1" if k else "opacity:0" for k in tagli]
    anim.append(keyframes("tag", scorre(
        [f"transform:translateY({p - posizioni[-1]:.1f}px)" for p in posizioni],
        passo)))
    anim.append(keyframes("tagv", scorre(visibile, passo)))
    corpo.append(
        f'<g class="tgr" style="animation:tag var(--d) infinite, '
        f'tagv var(--d) infinite">'
        f'<line class="tag" x1="{X_RIGA - 12}" y1="{posizioni[-1]:.1f}" '
        f'x2="{X_TAG}" y2="{posizioni[-1]:.1f}"/>'
        f'<text class="tagl" x="{X_TAG + 6}" y="{posizioni[-1] + 5:.1f}">'
        f'taglio</text></g>')

    # ---- il racconto: titolo della tappa e riga che spiega ----------------
    for s, t in enumerate(vista):
        anim.append(keyframes(f"txt{s}", accende(s, n, passo)))
        fermo = ";opacity:1" if s == n - 1 else ""
        m = f' style="animation:txt{s} var(--d) infinite{fermo}"'
        corpo.append(f'<text class="tit" x="{X_BAR}" y="{Y_TIT}"{m}>'
                     f'{t["titolo"]}</text>')
        corpo.append(f'<text class="rig" x="{X_BAR}" y="{Y_RIG}"{m}>'
                     f'{t["riga"]}</text>')

    corpo.append(f'<text class="fin" x="{X_BAR}" y="{Y_FIN}">'
                 f'i due gesti stanno in fila, non affiancati: il taglio cade '
                 f'sulla classifica già ripesata</text>')

    # ---- l'alt, calcolato ------------------------------------------------
    fredda = arrotonda_a_cento(PARTENZA)
    calda = arrotonda_a_cento(ripesa(PARTENZA, T_CALDA))
    k_f, k_c = nucleo(PARTENZA, SOGLIA), nucleo(ripesa(PARTENZA, T_CALDA), SOGLIA)
    cum_f, cum_c = cumulate(fredda), cumulate(calda)

    return Figura(
        larghezza=LARG, altezza=ALT,
        alt=(f"Cinque barre orizzontali, una per candidato, ordinate per "
             f"probabilità, con accanto la percentuale e la somma cumulata. "
             f"La figura ripete un ciclo di sei tappe. A temperatura "
             f"{num(T_FREDDA, 0)} le percentuali sono "
             f"{' · '.join(str(x) for x in fredda)} e le cumulate "
             f"{' · '.join(str(x) for x in cum_f)}: una linea orizzontale "
             f"segnata «taglio» scende sotto la riga {k_f}, perché lì la "
             f"cumulata fa {cum_f[k_f - 1]} e supera la soglia di novanta; le "
             f"barre escluse spariscono e le altre si allargano fino a "
             f"ricomporre cento. Poi la temperatura sale a "
             f"{num(T_CALDA, 0)}: tornano tutte e cinque le barre, più vicine "
             f"fra loro ({' · '.join(str(x) for x in calda)}, cumulate "
             f"{' · '.join(str(x) for x in cum_c)}), e la stessa linea del "
             f"taglio, con la stessa soglia, scende di una riga, sotto la "
             f"{k_c}, perché alla riga {k_f} la cumulata adesso fa "
             f"{cum_c[k_f - 1]} e non basta più."),
        corpo="".join(corpo),
        stile=f"""    .bar {{ stroke:none; }}
    .ord {{ font-family:{SANS}; font-size:13px; fill:{FG_MUTED}; }}
    .pct {{ font-family:{SANS}; font-size:15px; font-weight:700;
            fill:{INK}; opacity:0; }}
    .cum {{ font-family:{SANS}; font-size:15px; fill:{FG_MUTED}; opacity:0; }}
    .tag {{ stroke:{TERRACOTTA}; stroke-width:2.5; stroke-dasharray:7 5;
            stroke-linecap:round; }}
    .tagl {{ font-family:{SANS}; font-size:13px; font-weight:700;
            fill:{TERRACOTTA}; }}
    .tit {{ font-family:{SANS}; font-size:17px; font-weight:700;
            fill:{TERRACOTTA}; opacity:0; }}
    .rig {{ font-family:{SANS}; font-size:13.5px; fill:{FG_MUTED}; opacity:0; }}
    .fin {{ font-family:{SANS}; font-size:14px; fill:{INK}; }}""",
        animazioni=anim,
        durata=n * 1.8,
        fermi=".bar, .ord, .pct, .cum, .tit, .rig, .tgr",
    )


# --------------------------------------------------------------------------
# Timeline: le stesse due funzioni della vetrina, che qui servono uguali
# --------------------------------------------------------------------------
def scorre(valori, passo, quota=0.34):
    """(tempo, valore) per una successione che transita dentro la propria fetta."""
    tappe_ = [(0.0, valori[0])]
    for s in range(1, len(valori)):
        if valori[s] == valori[s - 1]:
            continue
        t0 = s * passo
        tappe_ += [(t0, valori[s - 1]), (min(t0 + passo * quota, 99.9), valori[s])]
    tappe_.append((100.0, valori[-1]))
    return tappe_


def accende(s, n, passo):
    """Opacita' di cio' che vale solo nella tappa s: l'ultima resta accesa."""
    t0 = s * passo
    if s == 0:
        return [(0.0, "opacity:1"), (passo * 0.94, "opacity:1"),
                (passo, "opacity:0"), (100.0, "opacity:0")]
    if s == n - 1:
        return [(0.0, "opacity:0"), (t0 - passo * 0.09, "opacity:0"),
                (t0, "opacity:1"), (100.0, "opacity:1")]
    return [(0.0, "opacity:0"), (t0 - passo * 0.09, "opacity:0"),
            (t0, "opacity:1"), (t0 + passo * 0.94, "opacity:1"),
            (t0 + passo, "opacity:0"), (100.0, "opacity:0")]
