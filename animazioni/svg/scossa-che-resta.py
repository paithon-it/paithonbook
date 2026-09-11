"""La stessa scossa su due tendenze: una si riassorbe, l'altra resta.

`SerieTemporali/componenti-e-classici.md` distingue la tendenza
**deterministica** («la retta c'e' davvero, le scosse la fanno sbandare ma non
la spostano») da quella **stocastica** («ogni scossa le sposta il livello per
sempre»), e sceglie come esempio della prima la serie $100, 110, 120, 130$.
Quella serie pero' e' liscia: di scosse non ne ha nessuna, quindi la frase che
la definisce non ci si puo' vedere. E' il buco che questa figura chiude.

La scena e' un esperimento a una variabile: le due serie ricevono **le stesse
identiche scosse**, la stessa salita di fondo e lo stesso punto di partenza.
Cambia solo dove la scossa finisce. Nella deterministica si somma alla retta,
quindi al passo dopo e' sparita; nella stocastica si somma al valore
precedente, quindi entra nel livello e ci resta.

Il tratteggio non e' lo stesso nei due riquadri, ed e' il punto: a sinistra e'
la retta attorno a cui la serie oscilla, a destra e' il **controfattuale**,
cioe' il cammino che quella serie avrebbe fatto se la scossa grossa non fosse
arrivata. Da li' in poi lo scarto vale esattamente la scossa, per sempre:
`verifica()` lo pretende passo per passo.

Ferma di proposito. Il tempo sta gia' sull'asse orizzontale, quindi
l'animazione mostrerebbe una seconda volta cio' che il disegno mostra tutto
insieme, e il confronto fra i due riquadri si legge in una volta sola.
"""

from paithon_svg import *

NOME = "scossa-che-resta"
TITOLO = "la stessa scossa su due tendenze"

# --------------------------------------------------------------------------
# La scena
# --------------------------------------------------------------------------
LIVELLO = 100.0            # il punto di partenza, uguale per le due serie
SALITA = 4.0               # la spinta di fondo, uguale per le due serie
QUANDO = 6                 # l'istante in cui arriva la scossa grossa
SCOSSA = 18.0              # quanto vale

# Le scosse piccole, scritte a mano: nessun sorteggio, cosi' i numeri della
# figura si rifanno a mente. Quella dell'istante QUANDO vale zero, perche' li'
# la sola cosa che succede dev'essere la scossa grossa.
PICCOLE = [0.0, 3.0, -4.0, 2.0, -3.0, 4.0,
           0.0, -3.0, 2.0, -4.0, 3.0, -2.0, 4.0, -3.0]
N = len(PICCOLE)


def scosse() -> list[float]:
    """Le scosse che ricevono tutt'e due le serie: le piccole, piu' la grossa."""
    fuori = list(PICCOLE)
    fuori[QUANDO] += SCOSSA
    return fuori


def retta(t: int) -> float:
    """La retta attorno a cui oscilla la tendenza deterministica."""
    return LIVELLO + SALITA * t


def deterministica(urti: list[float]) -> list[float]:
    """La scossa si somma **alla retta**: non entra mai nel livello."""
    return [retta(t) + urti[t] for t in range(N)]


def stocastica(urti: list[float]) -> list[float]:
    """La scossa si somma **al valore precedente**: entra nel livello."""
    fuori = [LIVELLO + urti[0]]
    for t in range(1, N):
        fuori.append(fuori[-1] + SALITA + urti[t])
    return fuori


def verifica(urti, det, sto, controf) -> None:
    """Difende quello che la didascalia promette, numero per numero."""
    assert PICCOLE[QUANDO] == 0.0, \
        "all'istante della scossa grossa non deve succedere nient'altro"
    assert urti[QUANDO] == SCOSSA, \
        f"la scossa grossa vale {urti[QUANDO]}, la figura dice {SCOSSA}"
    assert max(abs(p) for p in PICCOLE) * 3 < SCOSSA, \
        "la scossa grossa non si distingue dalle piccole"

    # Le due serie ricevono le stesse scosse: e' l'unico modo perche' il
    # confronto dica qualcosa.
    assert deterministica(urti) == det and stocastica(urti) == sto, \
        "le due serie non sono state costruite dalle stesse scosse"

    # A sinistra lo scarto dalla retta **e'** la scossa di quell'istante,
    # quindi al passo dopo della scossa grossa non resta niente.
    for t in range(N):
        assert abs((det[t] - retta(t)) - urti[t]) < 1e-9, \
            f"a sinistra lo scarto dalla retta all'istante {t} non e' la scossa"
    assert det[QUANDO] - retta(QUANDO) == SCOSSA, \
        "a sinistra la scossa non si vede nell'istante in cui arriva"
    resti = [abs(det[t] - retta(t)) for t in range(QUANDO + 1, N)]
    assert max(resti) <= max(abs(p) for p in PICCOLE), \
        (f"a sinistra, dopo la scossa, la serie resta a {max(resti):.1f} dalla "
         f"retta: non ci e' tornata sopra")

    # A destra lo scarto dal controfattuale e' zero prima e la scossa dopo,
    # esatto a ogni passo: e' la definizione di «sposta il livello per sempre».
    for t in range(QUANDO):
        assert abs(sto[t] - controf[t]) < 1e-9, \
            f"a destra le due serie divergono gia' all'istante {t}"
    for t in range(QUANDO, N):
        assert abs((sto[t] - controf[t]) - SCOSSA) < 1e-9, \
            (f"a destra all'istante {t} lo scarto vale "
             f"{sto[t] - controf[t]:.1f} invece di {SCOSSA}")

    # E la differenza fra i due meccanismi, enunciata sui numeri: la stessa
    # retta piu' la scossa **di adesso** contro la stessa retta piu' le scosse
    # **accumulate**. E' la ragione per cui una si riassorbe e l'altra no.
    assert det[0] == controf[0], "le due serie non partono dallo stesso valore"
    somma = 0.0
    for t in range(N):
        somma += PICCOLE[t]
        assert abs(controf[t] - (retta(t) + somma)) < 1e-9, \
            (f"a destra all'istante {t} la serie non e' la retta piu' le "
             f"scosse accumulate: la salita di fondo non e' la stessa")


# --------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------
LARG, ALT = 760, 352
X0, Y0 = 40.0, 62.0        # angolo alto-sinistro del riquadro di sinistra
W, HH = 280.0, 196.0       # un riquadro
SALTO = 336.0              # da un riquadro all'altro


def it(v: float) -> str:
    return f"{v:g}".replace(".", ",")


def costruisci() -> Figura:
    urti = scosse()
    det = deterministica(urti)
    sto = stocastica(urti)
    controf = stocastica(PICCOLE)
    verifica(urti, det, sto, controf)

    tutti = det + sto + controf + [retta(t) for t in range(N)]
    ymin, ymax = min(tutti), max(tutti)
    margine = (ymax - ymin) * 0.12
    ymin, ymax = ymin - margine, ymax + margine

    def sy(v: float) -> float:
        return Y0 + HH - (v - ymin) / (ymax - ymin) * HH

    def sx(base: float, t: float) -> float:
        return base + t / (N - 1) * W

    def spezzata(base, valori) -> str:
        return " ".join(f"{sx(base, t):.1f},{sy(v):.1f}"
                        for t, v in enumerate(valori))

    corpo = ['<defs>'
             '<marker id="pta" viewBox="0 0 10 10" refX="8.5" refY="5" '
             'markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
             f'<path d="M 0 1 L 9 5 L 0 9 z" fill="{TERRACOTTA}"/></marker>'
             '</defs>']

    for base, titolo in ((X0, "tendenza deterministica"),
                         (X0 + SALTO, "tendenza stocastica")):
        corpo.append(f'<text class="pan" x="{base:.1f}" y="{Y0 - 30:.1f}">'
                     f'{titolo}</text>')
        corpo.append(f'<line class="axc" x1="{base:.1f}" y1="{Y0 + HH:.1f}" '
                     f'x2="{base + W:.1f}" y2="{Y0 + HH:.1f}"/>')

    # ---- sinistra: la retta, la serie, la scossa che sbanda e rientra -----
    corpo.append(f'<polyline class="rif" points="'
                 f'{spezzata(X0, [retta(t) for t in range(N)])}"/>')
    corpo.append(f'<text class="lbs" x="{sx(X0, N - 1) - 4:.1f}" '
                 f'y="{sy(retta(N - 1)) + 34:.1f}" text-anchor="end">'
                 f'la retta</text>')
    corpo.append(f'<polyline class="ser" points="{spezzata(X0, det)}"/>')

    xq = sx(X0, QUANDO)
    corpo.append(f'<line class="sco" x1="{xq:.1f}" y1="{sy(retta(QUANDO)):.1f}" '
                 f'x2="{xq:.1f}" y2="{sy(det[QUANDO]) + 5:.1f}" '
                 f'marker-end="url(#pta)"/>')
    corpo.append(f'<text class="num" x="{xq - 13:.1f}" '
                 f'y="{sy((retta(QUANDO) + det[QUANDO]) / 2) + 4:.1f}" '
                 f'text-anchor="end">+{it(SCOSSA)}</text>')
    corpo.append(f'<circle class="pun" cx="{sx(X0, QUANDO + 1):.1f}" '
                 f'cy="{sy(det[QUANDO + 1]):.1f}" r="4.5"/>')

    # ---- destra: il controfattuale, la serie, lo scarto che non si chiude --
    corpo.append(f'<polyline class="rif" points="{spezzata(X0 + SALTO, controf)}"/>')
    corpo.append(f'<text class="lbs" x="{sx(X0 + SALTO, N - 1):.1f}" '
                 f'y="{sy(controf[N - 1]) + 34:.1f}" text-anchor="end">'
                 f'senza la scossa</text>')
    corpo.append(f'<polyline class="ser" points="{spezzata(X0 + SALTO, sto)}"/>')

    xq2 = sx(X0 + SALTO, QUANDO)
    corpo.append(f'<line class="sco" x1="{xq2:.1f}" '
                 f'y1="{sy(controf[QUANDO]):.1f}" x2="{xq2:.1f}" '
                 f'y2="{sy(sto[QUANDO]) + 5:.1f}" marker-end="url(#pta)"/>')
    corpo.append(f'<text class="num" x="{xq2 - 13:.1f}" '
                 f'y="{sy((controf[QUANDO] + sto[QUANDO]) / 2) + 4:.1f}" '
                 f'text-anchor="end">+{it(SCOSSA)}</text>')

    xf = sx(X0 + SALTO, N - 1)
    corpo.append(f'<line class="sca" x1="{xf:.1f}" y1="{sy(controf[-1]):.1f}" '
                 f'x2="{xf:.1f}" y2="{sy(sto[-1]):.1f}"/>')
    corpo.append(f'<text class="num" x="{xf + 8:.1f}" '
                 f'y="{sy((controf[-1] + sto[-1]) / 2) + 4:.1f}">'
                 f'ancora +{it(SCOSSA)}</text>')

    # ---- le due righe che dicono che cosa e' successo ---------------------
    y_dida = Y0 + HH + 30
    corpo.append(f'<text class="lbs" x="{X0:.1f}" y="{y_dida:.1f}">'
                 f'il passo dopo la serie è già tornata sulla retta</text>')
    corpo.append(f'<text class="lbs" x="{X0 + SALTO:.1f}" y="{y_dida:.1f}">'
                 f'lo scarto non si chiude più, a nessun passo</text>')

    return Figura(
        larghezza=LARG, altezza=ALT,
        alt="Due grafici affiancati, stesso asse verticale, con il tempo in "
            "orizzontale. Le due serie partono dallo stesso valore, salgono "
            "con la stessa spinta e ricevono le stesse scosse, fra cui una "
            f"grossa, di più {it(SCOSSA)}, al settimo istante, segnata in "
            "tutti e due i grafici da una freccia verticale. A sinistra, "
            "«tendenza deterministica», la serie oscilla attorno a una retta "
            "tratteggiata: nell'istante della scossa schizza sopra la retta, e "
            "già il punto dopo, cerchiato, è tornato ad appoggiarsi alla "
            "retta come tutti gli altri. A destra, «tendenza stocastica», il "
            "tratteggio è il cammino che la serie avrebbe fatto senza quella "
            "scossa: dall'istante della scossa in poi le due linee corrono "
            "parallele, e la distanza fra loro all'ultimo istante è segnata "
            f"«ancora più {it(SCOSSA)}», la stessa di quando la scossa è "
            "arrivata.",
        corpo="".join(corpo),
        stile=f"""    .ser  {{ fill:none; stroke:{TERRACOTTA}; stroke-width:2.6;
            stroke-linejoin:round; }}
    .rif  {{ fill:none; stroke:{FG_MUTED}; stroke-width:1.6;
            stroke-dasharray:6 5; stroke-linejoin:round; }}
    .sco  {{ stroke:{TERRACOTTA}; stroke-width:2; }}
    .sca  {{ stroke:{TEAL}; stroke-width:2.4; stroke-dasharray:3 3; }}
    .pun  {{ fill:{CREAM}; stroke:{TEAL}; stroke-width:2.4; }}
    .num  {{ font-family:{SANS}; font-size:13px; font-weight:700;
            fill:{TERRACOTTA}; }}
    .pan  {{ font-family:{SANS}; font-size:16px; font-weight:700;
            fill:{INK}; }}""",
    )
