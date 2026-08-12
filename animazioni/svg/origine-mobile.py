"""Validazione a origine mobile: il taglio avanza, il test non guarda indietro.

Il tempo qui è il contenuto due volte: è l'asse dei dati ed è il meccanismo,
perché il confine fra training e test si sposta in avanti a ogni giro. Una
figura ferma mostra un taglio solo; il punto è che i tagli sono sedici e che
in nessuno di essi il training arriva dopo il test.

Gli indici non sono disegnati a mano: li calcola `tagli_origine_mobile()`, la
stessa procedura di `walk_forward_split()` nel capitolo, con gli stessi
parametri. La proprietà che la figura esiste per mostrare è garantita da un
`assert`, non dall'occhio di chi ha disegnato.

Lo stato di riposo è l'ultimo taglio: la serie è consumata fino in fondo, il
test è il blocco finale. Chi non anima vede una figura conclusa e sensata.
"""

import random

from paithon_svg import *

NOME = "origine-mobile"
TITOLO = "origine mobile contro k-fold mescolata"

# I parametri sono quelli del capitolo (SerieTemporali/validazione-e-feature.md,
# sezione "In pratica: walk-forward e MASE con NumPy").
N, MIN_TRAIN, ORIZZONTE = 140, 28, 7
AMPIEZZA = MIN_TRAIN     # finestra scorrevole: lunga quanto il primo training
K, SEME = 5, 0           # la k-fold mescolata da contrapporre
TAGLI_ATTESI = 16        # quanti ne conta il codice del capitolo


def tagli_origine_mobile(n, min_train, orizzonte, ampiezza=None):
    """Coppie (train, test) del walk-forward, col test sempre nel futuro.

    `ampiezza=None` dà la finestra espansa (tutto il passato disponibile),
    un valore dà quella scorrevole (training di lunghezza costante).
    """
    tagli = []
    for t in range(min_train, n - orizzonte + 1, orizzonte):
        inizio = 0 if ampiezza is None else max(0, t - ampiezza)
        tagli.append((list(range(inizio, t)), list(range(t, t + orizzonte))))
    return tagli


def tagli_k_fold_mescolata(n, k, seme):
    """Le stesse osservazioni, ma assegnate ai fold a sorte: l'errore da evitare."""
    idx = list(range(n))
    random.Random(seme).shuffle(idx)
    tagli = []
    for f in range(k):
        test = sorted(idx[f::k])
        tagli.append((sorted(set(range(n)) - set(test)), test))
    return tagli


def test_con_futuro_nel_training(train, test):
    """Quanti punti di test hanno dati di addestramento nel proprio futuro."""
    ultimo = max(train)
    return sum(1 for i in test if i < ultimo)


ESPANSA = tagli_origine_mobile(N, MIN_TRAIN, ORIZZONTE)
SCORREVOLE = tagli_origine_mobile(N, MIN_TRAIN, ORIZZONTE, AMPIEZZA)
MESCOLATA = tagli_k_fold_mescolata(N, K, SEME)

# La proprietà per cui la figura esiste: in nessun taglio un indice di test
# viene prima di un indice di training.
for _nome, _tagli in (("espansa", ESPANSA), ("scorrevole", SCORREVOLE)):
    for _i, (_tr, _te) in enumerate(_tagli):
        if min(_te) <= max(_tr):
            raise AssertionError(
                f"finestra {_nome}, taglio {_i + 1}: il training arriva fino a "
                f"{max(_tr)} ma il test comincia a {min(_te)}")

if len(ESPANSA) != TAGLI_ATTESI:
    raise AssertionError(f"la figura disegna {len(ESPANSA)} tagli, il codice del "
                         f"capitolo ne conta {TAGLI_ATTESI}")

# E il contrario, sul fold mescolato: se un giorno non violasse più nulla, la
# metà alta della figura non avrebbe più niente da mostrare.
VIOLAZIONI = test_con_futuro_nel_training(*MESCOLATA[0])
if VIOLAZIONI == 0:
    raise AssertionError("il fold mescolato disegnato non viola l'ordine temporale")


def blocchi(indici):
    """Indici consecutivi accorpati in (inizio, fine): meno rettangoli, stesso disegno."""
    fuori = []
    for i in indici:
        if fuori and fuori[-1][1] == i:
            fuori[-1][1] = i + 1
        else:
            fuori.append([i, i + 1])
    return fuori


def costruisci() -> Figura:
    x0, larg, h = 40.0, 620.0, 30.0
    px = larg / N                       # pixel per osservazione
    basi = (26.0, 112.0, 198.0)         # le tre righe
    corpo = []

    def sx(i):
        return x0 + i * px

    def riga(base, etichetta, nota):
        corpo.append(f'<text class="lbl" x="{x0:.0f}" y="{base:.0f}">{etichetta}</text>')
        corpo.append(f'<text class="lbs" x="{x0:.0f}" y="{base + 56:.0f}">{nota}</text>')
        return base + 8

    # ---- riga 1: la k-fold mescolata, ferma: il disordine si vede in un colpo
    train, test = MESCOLATA[0]
    y = riga(basi[0], "k-fold mescolata (k = 5)",
             f"{VIOLAZIONI} punti di test su {len(test)} hanno dati di "
             f"addestramento nel proprio futuro")
    corpo.append(f'<rect class="tr" x="{x0:.0f}" y="{y:.0f}" '
                 f'width="{larg:.0f}" height="{h:.0f}"/>')
    for a, b in blocchi(test):
        corpo.append(f'<rect class="te" x="{sx(a):.1f}" y="{y:.0f}" '
                     f'width="{(b - a) * px:.1f}" height="{h:.0f}"/>')
    corpo.append(f'<rect class="cornice" x="{x0:.0f}" y="{y:.0f}" '
                 f'width="{larg:.0f}" height="{h:.0f}"/>')

    # ---- righe 2 e 3: l'origine che si muove
    ultimo_train, ultimo_test = ESPANSA[-1]
    tappe_cresce, tappe_scorre = [], []
    for i, (tr, te) in enumerate(ESPANSA):
        t0, t1 = sosta(i, len(ESPANSA), tenuta=0.55)
        k = len(tr) / len(ultimo_train)                  # il training si allunga
        dx = (te[0] - ultimo_test[0]) * px               # il taglio scivola
        tappe_cresce += [(t0, f"transform:scaleX({k:.4f})"),
                         (t1, f"transform:scaleX({k:.4f})")]
        tappe_scorre += [(t0, f"transform:translateX({dx:.1f}px)"),
                         (t1, f"transform:translateX({dx:.1f}px)")]
    tappe_cresce.append((100.0, "transform:scaleX(1)"))
    tappe_scorre.append((100.0, "transform:translateX(0px)"))
    anim = [keyframes("cresce", tappe_cresce), keyframes("scorre", tappe_scorre)]

    for base, (etichetta, nota), tagli in (
        (basi[1], ("origine mobile, finestra espansa",
                   f"{len(ESPANSA)} tagli: il training si allunga, il test sta "
                   f"sempre dopo"), ESPANSA),
        (basi[2], ("origine mobile, finestra scorrevole",
                   f"stessi tagli, ma il training resta lungo {AMPIEZZA} "
                   f"osservazioni"), SCORREVOLE),
    ):
        tr, te = tagli[-1]
        y = riga(base, etichetta, nota)
        corpo.append(f'<rect class="futuro" x="{x0:.0f}" y="{y:.0f}" '
                     f'width="{larg:.0f}" height="{h:.0f}"/>')
        # il training: si allunga se è espanso, scivola se è scorrevole
        moto = "cresce" if tr[0] == 0 else "scorre"
        corpo.append(f'<rect class="tr" x="{sx(tr[0]):.1f}" y="{y:.0f}" '
                     f'width="{len(tr) * px:.1f}" height="{h:.0f}" '
                     f'style="animation:{moto} var(--d) infinite;'
                     f'transform-box:view-box;transform-origin:{x0:.0f}px 0"/>')
        corpo.append(f'<rect class="te" x="{sx(te[0]):.1f}" y="{y:.0f}" '
                     f'width="{len(te) * px:.1f}" height="{h:.0f}" '
                     f'style="animation:scorre var(--d) infinite"/>')
        corpo.append(f'<rect class="cornice" x="{x0:.0f}" y="{y:.0f}" '
                     f'width="{larg:.0f}" height="{h:.0f}"/>')

    # ---- legenda, in alto a destra, sulla riga della prima etichetta
    lx = x0 + larg
    for dx, cls, testo in ((-236, "tr", "training"), (-92, "te", "test")):
        corpo.append(f'<rect class="{cls}" x="{lx + dx:.0f}" y="{basi[0] - 11:.0f}" '
                     f'width="13" height="13"/>')
        corpo.append(f'<text class="lbs" x="{lx + dx + 20:.0f}" '
                     f'y="{basi[0]:.0f}">{testo}</text>')

    # ---- l'asse del tempo, che vale per tutte e tre le righe
    ya = basi[2] + 76
    corpo += [
        f'<line class="freccia" x1="{x0:.0f}" y1="{ya:.0f}" '
        f'x2="{x0 + larg - 16:.0f}" y2="{ya:.0f}"/>',
        f'<polygon class="punta" points="{x0 + larg - 16:.0f},{ya - 6:.0f} '
        f'{x0 + larg:.0f},{ya:.0f} {x0 + larg - 16:.0f},{ya + 6:.0f}"/>',
        f'<text class="lbs" x="{x0:.0f}" y="{ya + 20:.0f}">passato</text>',
        f'<text class="lbs" x="{x0 + larg / 2:.0f}" y="{ya + 20:.0f}" '
        f'text-anchor="middle">una serie di {N} osservazioni</text>',
        f'<text class="lbs" x="{x0 + larg:.0f}" y="{ya + 20:.0f}" '
        f'text-anchor="end">futuro</text>',
    ]

    return Figura(
        larghezza=700, altezza=ya + 36,
        alt="Tre barre del tempo sulla stessa serie. In alto la k-fold "
            "mescolata: i blocchi di test terracotta sono sparsi ovunque, "
            "anche prima dei dati di addestramento teal. Sotto la validazione "
            "a origine mobile, a finestra espansa e a finestra scorrevole: il "
            "training avanza da sinistra e il blocco di test gli sta sempre "
            "subito a destra, cioè nel futuro.",
        corpo="".join(corpo),
        stile=f"""    .tr {{ fill:{TEAL}; }}
    .te {{ fill:{TERRACOTTA}; }}
    .futuro {{ fill:{CREAM}; }}
    .cornice {{ fill:none; stroke:{BORDER_STRONG}; stroke-width:1.5; }}
    .freccia {{ stroke:{FG_MUTED}; stroke-width:1.5; }}
    .punta {{ fill:{FG_MUTED}; }}""",
        animazioni=anim,
        durata=len(ESPANSA) * 0.55,
        fermi=".tr, .te",
    )
