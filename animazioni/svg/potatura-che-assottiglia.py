"""La potatura iterativa: si pota, si riaddestra, si pota ancora.

Il tempo qui è il contenuto in senso stretto, perché la potatura iterativa
**è** un ciclo: a ogni giro si toglie una fetta dei pesi rimasti e si lascia
alla rete qualche centinaio di passi per rimettersi in sesto. Su una figura
ferma si vedrebbe solo il punto d'arrivo, e si perderebbe la cosa che la
sezione racconta: che l'accuratezza non scende, **non scende, non scende, e poi
cade di colpo**.

I numeri li calcola la scena, addestrando davvero una rete piccola sulle cifre
scritte a mano (due secondi e mezzo). Il riquadro di sinistra mostra
duecentocinquantasei pesi presi a caso fra i quattromila del primo strato, e si
spengono quando la potatura li toglie; quello di destra è la curva
dell'accuratezza contro la frazione di pesi tolti.

Lo stato di riposo è l'ultimo giro: la griglia quasi vuota e la curva intera.
"""

import torch
from torch import nn
from torch.nn import functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from paithon_svg import *

NOME = "potatura-che-assottiglia"
TITOLO = "La potatura iterativa, giro dopo giro"

GIRI = 9
FETTA = 0.35          # quanta parte dei pesi rimasti si toglie a ogni giro
MOSTRATI = 256        # quanti pesi si disegnano, presi a caso


# --------------------------------------------------------------------------
# Il ciclo, eseguito per davvero
# --------------------------------------------------------------------------
def storia():
    """Potatura iterativa su una rete piccola: [(frazione tolta, accuratezza,
    maschera dei pesi mostrati)] giro per giro."""
    dati = load_digits()
    Xtr, Xte, ytr, yte = train_test_split(dati.data / 16.0, dati.target,
                                          test_size=0.3, random_state=0)
    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    Xte = torch.tensor(Xte, dtype=torch.float32)
    ytr, yte = torch.tensor(ytr), torch.tensor(yte)

    torch.manual_seed(0)
    torch.set_num_threads(1)
    rete = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
    W = rete[0].weight

    def addestra(passi, maschera=None):
        # `W` e' quello della funzione che racchiude questa: senza `nonlocal`
        # il `W *= maschera` piu' sotto lo renderebbe una variabile locale
        nonlocal W
        opt = torch.optim.Adam(rete.parameters(), lr=3e-3)
        for _ in range(passi):
            F.cross_entropy(rete(Xtr), ytr).backward()
            opt.step()
            opt.zero_grad()
            if maschera is not None:
                with torch.no_grad():
                    W *= maschera

    def accuratezza():
        with torch.no_grad():
            return (rete(Xte).argmax(1) == yte).float().mean().item() * 100

    addestra(400)
    scelti = torch.randperm(W.numel(), generator=torch.Generator().manual_seed(1))
    scelti = scelti[:MOSTRATI]
    maschera = torch.ones_like(W)
    passi = [(0.0, accuratezza(), maschera.flatten()[scelti].clone())]
    for _ in range(GIRI):
        with torch.no_grad():
            vivi = W[maschera.bool()].abs()
            soglia = vivi.kthvalue(max(int(FETTA * vivi.numel()), 1)).values
            maschera = maschera * (W.abs() >= soglia).float()
            W *= maschera
        addestra(150, maschera)
        passi.append((1 - maschera.mean().item(), accuratezza(),
                      maschera.flatten()[scelti].clone()))
    return passi


def verifica(passi) -> None:
    """La figura promette un altopiano e poi un crollo: ci sono davvero?"""
    tolti = [p[0] for p in passi]
    acc = [p[1] for p in passi]
    assert tolti == sorted(tolti) and tolti[-1] > 0.95, \
        f"la potatura deve crescere e arrivare in alto, arriva a {tolti[-1]:.2f}"
    # L'altopiano va verificato **alla sparsita' che la didascalia nomina**, non
    # sulla prima meta' dei giri: i giri si mappano sulla sparsita' in modo
    # esponenziale in FETTA, quindi contarli lascia passare parametri con cui la
    # curva e' gia' crollata al 90% e la didascalia diventa falsa. Provato: con
    # FETTA=0.22 e GIRI=16 gli assert vecchi passavano tutti, e a nove pesi su
    # dieci tolti la curva era gia' scesa di quasi cinque punti.
    def a_sparsita(s: float) -> float:
        for i in range(1, len(tolti)):
            if tolti[i] >= s:
                q = (s - tolti[i - 1]) / (tolti[i] - tolti[i - 1])
                return acc[i - 1] + q * (acc[i] - acc[i - 1])
        return acc[-1]

    caduta = acc[0] - a_sparsita(0.90)
    assert caduta < 2.0, \
        (f"la didascalia dice che la curva resta piatta finche' si tolgono i "
         f"primi nove pesi su dieci: a quel punto e' gia' scesa di "
         f"{caduta:.1f} punti")
    assert acc[0] - acc[-1] > 15.0, \
        (f"senza un crollo alla fine questa figura non racconta niente: "
         f"si passa da {acc[0]:.1f} a {acc[-1]:.1f}")


# --------------------------------------------------------------------------
# La figura
# --------------------------------------------------------------------------
LARG, ALT = 760, 430
LATO = 16                     # la griglia dei pesi mostrati
CELLA = 13
GX, GY = 64, 96
INIZIO, FINE = 6.0, 86.0


def dec(v: float, cifre: int = 0) -> str:
    return f"{v:.{cifre}f}".replace(".", ",")


def costruisci() -> Figura:
    passi = storia()
    verifica(passi)
    n = len(passi)
    istante = [INIZIO + (FINE - INIZIO) * i / (n - 1) for i in range(n)]

    corpo, anim = [], []
    for i, t in enumerate(istante):
        anim.append(keyframes(f"g{i}", [
            (0.0, "opacity:0"),
            (max(t - 0.8, 0.0), "opacity:0"),
            (t, "opacity:1"),
            (istante[i + 1] - 0.8 if i + 1 < n else 100.0, "opacity:1"),
            *([(istante[i + 1], "opacity:0"), (100.0, "opacity:0")] if i + 1 < n else [])]))

    def solo_al_giro(i, dentro, permanente=False):
        """Visibile al giro i. Se `permanente`, da lì in poi non si spegne."""
        nome = f"p{i}" if permanente else f"g{i}"
        if permanente and f"@keyframes p{i}" not in "".join(anim):
            t = istante[i]
            anim.append(keyframes(nome, [(0.0, "opacity:0"),
                                         (max(t - 0.8, 0.0), "opacity:0"),
                                         (t, "opacity:1"), (100.0, "opacity:1")]))
        # A riposo si vede lo stato finale: i pezzi permanenti (la curva) ci
        # sono tutti, quelli che si avvicendano (il cartiglio) solo l'ultimo.
        base = "" if permanente or i == n - 1 else "opacity:0;"
        return (f'<g style="{base}animation:{nome} var(--d) linear infinite">'
                f'{dentro}</g>')

    # --- la griglia dei pesi ----------------------------------------------
    # Ogni cella si disegna UNA volta e si spegne al giro in cui la potatura
    # la toglie, raggruppata con le altre che muoiono insieme. Ridisegnare
    # tutta la griglia a ogni giro dava un file di 246 KB, quaranta volte la
    # norma delle figure del libro, per una animazione identica.
    corpo.append(f'<rect class="ax" x="{GX - 8}" y="{GY - 8}" '
                 f'width="{LATO * CELLA + 16}" height="{LATO * CELLA + 16}" rx="4"/>')
    for j in range(LATO + 1):
        d = j * CELLA - 1.5
        corpo.append(f'<line class="griglia" x1="{GX + d}" y1="{GY - 2}" '
                     f'x2="{GX + d}" y2="{GY + LATO * CELLA - 2}"/>'
                     f'<line class="griglia" x1="{GX - 2}" y1="{GY + d}" '
                     f'x2="{GX + LATO * CELLA - 2}" y2="{GY + d}"/>')

    def quando_muore(k):
        for i, (_, _, maschera) in enumerate(passi):
            if not maschera[k]:
                return i
        return None                                   # sopravvive fino alla fine

    gruppi = {}
    for k in range(LATO * LATO):
        gruppi.setdefault(quando_muore(k), []).append(k)

    def quadretto(k):
        x, y = GX + (k % LATO) * CELLA, GY + (k // LATO) * CELLA
        return (f'<rect x="{x}" y="{y}" width="{CELLA - 3}" height="{CELLA - 3}" '
                f'rx="1.5" fill="{TEAL}"/>')

    for i, chiavi in sorted(gruppi.items(), key=lambda v: (v[0] is None, v[0])):
        celle = "".join(quadretto(k) for k in chiavi)
        if i is None:                                  # i superstiti non si spengono
            corpo.append(celle)
            continue
        t_morte = istante[i]
        anim.append(keyframes(f"m{i}", [(0.0, "opacity:1"),
                                        (max(t_morte - 0.8, 0.0), "opacity:1"),
                                        (t_morte, "opacity:0"),
                                        (100.0, "opacity:0")]))
        corpo.append(f'<g style="opacity:0;animation:m{i} var(--d) linear infinite">'
                     f'{celle}</g>')

    # --- la curva ----------------------------------------------------------
    r = Riquadro(x=404, y=GY - 8, larg=292, alt=LATO * CELLA + 16,
                 xmin=-0.04, xmax=1.02, ymin=60, ymax=100)
    corpo.append(r.cornice())
    punti = [(r.sx(t), r.sy(max(a, 60))) for t, a, _ in passi]
    for i in range(1, n):
        (x1, y1), (x2, y2) = punti[i - 1], punti[i]
        corpo.append(solo_al_giro(
            i, f'<line class="cur" x1="{x1:.1f}" y1="{y1:.1f}" '
               f'x2="{x2:.1f}" y2="{y2:.1f}"/>', permanente=True))
    for i, (x, y) in enumerate(punti):
        corpo.append(solo_al_giro(
            i, f'<circle class="pun" cx="{x:.1f}" cy="{y:.1f}" r="4"/>',
            permanente=True))

    for v in (100, 90, 80, 70, 60):
        corpo.append(f'<text class="tic" x="{r.x - 8}" y="{r.sy(v) + 4:.1f}" '
                     f'text-anchor="end">{v}%</text>')
    corpo += [
        f'<text class="ttl" x="{GX - 8}" y="{GY - 44}">i pesi che restano</text>',
        f'<text class="lbs" x="{GX - 8}" y="{GY - 26}">'
        f'{MOSTRATI} presi a caso fra i {64 * 64} del primo strato</text>',
        f'<text class="ttl" x="{r.x}" y="{GY - 44}">che cosa costa</text>',
        f'<text class="lbs" x="{r.x}" y="{GY - 26}">'
        f'accuratezza contro pesi tolti</text>',
        f'<text class="lbs" x="{r.x}" y="{r.y + r.alt + 20}">nessuno tolto</text>',
        f'<text class="lbs" x="{r.x + r.larg}" y="{r.y + r.alt + 20}" '
        f'text-anchor="end">tolti tutti</text>']

    # --- il cartiglio del giro --------------------------------------------
    y = GY + LATO * CELLA + 44
    corpo.append(f'<text class="lbs" x="{GX - 8}" y="{y}">al giro</text>')
    for i, (tolti, acc, _) in enumerate(passi):
        corpo.append(solo_al_giro(
            i, f'<text class="big" x="{GX + 48}" y="{y}">'
               f'{dec(tolti * 100)}% tolti</text>'
               f'<text class="big2" x="{GX + 190}" y="{y}">'
               f'accuratezza {dec(acc, 1)}%</text>'))

    return Figura(
        larghezza=LARG, altezza=ALT,
        alt="Due riquadri affiancati. A sinistra una griglia di sedici per sedici "
            "quadratini, un campione dei pesi del primo strato di una rete: "
            "all'inizio sono tutti pieni, e giro dopo giro se ne svuotano sempre "
            "di più, fino a restarne pochissimi. A destra la curva "
            "dell'accuratezza contro la frazione di pesi tolti, tracciata un "
            "punto per giro: resta piatta poco sotto il cento per cento mentre "
            "si tolgono i primi nove pesi su dieci, e poi precipita negli ultimi "
            "giri. Sotto, a ogni giro, quanti pesi sono stati tolti e "
            "l'accuratezza corrispondente.",
        corpo="".join(corpo),
        stile=f"""    .griglia {{ stroke:{BORDER}; stroke-width:1; }}
    .cur  {{ stroke:{TERRACOTTA}; stroke-width:2.5; }}
    .pun  {{ fill:{TERRACOTTA}; stroke:{CREAM}; stroke-width:1.5; }}
    .tic  {{ font-family:{SANS}; font-size:12px; fill:{FG_MUTED}; }}
    .big  {{ font-family:{SANS}; font-size:15px; font-weight:600; fill:{TEAL}; }}
    .big2 {{ font-family:{SANS}; font-size:15px; font-weight:600;
             fill:{TERRACOTTA}; }}""",
        animazioni=anim,
        durata=12.0,
        fermi="g",
    )
