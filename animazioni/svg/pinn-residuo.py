"""Una PINN impara la fisica: il residuo che si spegne, la curva che si incolla.

I fotogrammi sono stati veri dell'addestramento, non un'interpolazione fra
inizio e fine: `addestra()` è la PINN di `PINN/come-funziona.md` riga per riga
(stessa equazione, stessa rete, stesso seme, stessa loss) e la figura fotografa
i pesi a cinque epoche.

Le due cose che il capitolo vuole far vedere insieme stanno una sopra l'altra:
in alto la curva della rete che si sovrappone alla soluzione esatta, in basso
il residuo nei punti di collocazione che cala. Sono lo stesso fatto visto da
due lati, ed è il motivo per cui una figura ferma qui perde quasi tutto.

Costa una quarantina di secondi (trentamila epoche su CPU) e richiede `torch`:
è il prezzo di numeri veri. Alla fine la rete riproduce i valori che il
capitolo stampa, e gli `assert` in fondo a `addestra()` lo verificano: se un
giorno l'addestramento non converge più, la figura non nasce.
"""

import math

from paithon_svg import *

NOME = "pinn-residuo"
TITOLO = "una PINN impara la fisica"

# Il problema del capitolo: oscillatore armonico smorzato, m u'' + c u' + k u = 0
M, C, K = 1.0, 0.4, 4.0
U_0, V_0 = 1.0, 0.0
T_MAX = 10.0

# L'addestramento del capitolo, invariato
SEME = 42
EPOCHE = 30_000
LR = 1e-3
LAMBDA_0 = 100.0
N_COLLOCAZIONE = 200

TAPPE = (0, 4_000, 8_000, 16_000, 30_000)   # le epoche fotografate
N_GRIGLIA = 71      # punti con cui si disegna una curva
N_BARRE = 16        # quanti punti di collocazione mostra il pannello in basso

# Quello che il capitolo stampa a fine addestramento, con questo stesso seme:
# residuo (media di r^2) sui 200 punti 7,77e-3 e scarto massimo 0,154.
RESIDUO_CAPITOLO = 7.8e-3
ERRORE_CAPITOLO = 0.154
TOLLERANZA_CURVA = 0.20    # scarto massimo ammesso dalla soluzione esatta
CALO_MINIMO = 20.0         # di quante volte il residuo deve almeno scendere


def soluzione_esatta(t: float) -> float:
    """u(t) = e^{-γt}(cos ω_d t + (γ/ω_d) sin ω_d t), la formula del capitolo."""
    gamma = C / (2 * M)
    omega_d = math.sqrt(K / M - gamma ** 2)
    return math.exp(-gamma * t) * (math.cos(omega_d * t)
                                   + (gamma / omega_d) * math.sin(omega_d * t))


def addestra():
    """La PINN di `PINN/come-funziona.md`, con le fotografie alle epoche di TAPPE.

    Restituisce (istanti mostrati, stati), dove ogni stato porta l'epoca, la
    curva sulla griglia di disegno, il residuo nei punti mostrati, la media di
    r^2 su tutti i punti di collocazione e lo scarto massimo dalla soluzione
    esatta.
    """
    import torch
    from torch import nn

    torch.set_num_threads(1)        # una sola somma parziale: risultati ripetibili
    torch.manual_seed(SEME)

    rete = nn.Sequential(
        nn.Linear(1, 32), nn.Tanh(),
        nn.Linear(32, 32), nn.Tanh(),
        nn.Linear(32, 32), nn.Tanh(),
        nn.Linear(32, 1),
    )
    t_c = T_MAX * torch.rand(N_COLLOCAZIONE, 1)
    t_c.requires_grad_(True)
    t_0 = torch.zeros(1, 1, requires_grad=True)
    ottimizzatore = torch.optim.Adam(rete.parameters(), lr=LR)

    # la griglia su cui si disegna, e i punti di collocazione che finiscono a barra
    griglia = [T_MAX * i / (N_GRIGLIA - 1) for i in range(N_GRIGLIA)]
    t_g = torch.tensor(griglia, dtype=torch.float32).reshape(-1, 1)
    esatta = [soluzione_esatta(t) for t in griglia]
    # lo scarto si misura sulla griglia fitta del capitolo (500 istanti), non su
    # quella di disegno: altrimenti il numero mostrato dipenderebbe da quanti
    # punti servono a tracciare una linea liscia
    fitta = [T_MAX * i / 499 for i in range(500)]
    t_f = torch.tensor(fitta, dtype=torch.float32).reshape(-1, 1)
    esatta_fitta = [soluzione_esatta(t) for t in fitta]
    istanti = t_c.detach().squeeze(1).tolist()
    ordinati = sorted(range(N_COLLOCAZIONE), key=lambda i: istanti[i])
    mostrati = [ordinati[round(i * (N_COLLOCAZIONE - 1) / (N_BARRE - 1))]
                for i in range(N_BARRE)]

    stati = []

    def fotografa(epoca):
        u = rete(t_c)
        u_t = torch.autograd.grad(u, t_c, torch.ones_like(u), create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, t_c, torch.ones_like(u_t))[0]
        residuo = (M * u_tt + C * u_t + K * u).detach().squeeze(1)
        with torch.no_grad():
            curva = rete(t_g).squeeze(1).tolist()
            densa = rete(t_f).squeeze(1).tolist()
        stati.append({
            "epoca": epoca,
            "curva": curva,
            "residui": [abs(float(residuo[i])) for i in mostrati],
            "medio": float((residuo ** 2).mean()),
            "scarto": max(abs(a - b) for a, b in zip(densa, esatta_fitta)),
        })

    for epoca in range(EPOCHE + 1):
        if epoca in TAPPE:
            fotografa(epoca)
        if epoca == EPOCHE:
            break
        ottimizzatore.zero_grad()

        u = rete(t_c)
        u_t = torch.autograd.grad(u, t_c, torch.ones_like(u), create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, t_c, torch.ones_like(u_t), create_graph=True)[0]
        loss_fisica = ((M * u_tt + C * u_t + K * u) ** 2).mean()

        u_0 = rete(t_0)
        u_t0 = torch.autograd.grad(u_0, t_0, torch.ones_like(u_0), create_graph=True)[0]
        loss_iniziale = (u_0 - U_0).pow(2).mean() + (u_t0 - V_0).pow(2).mean()

        (loss_fisica + LAMBDA_0 * loss_iniziale).backward()
        ottimizzatore.step()

    primo, ultimo = stati[0], stati[-1]
    # Le due promesse della figura. Se l'addestramento non converge piu', meglio
    # nessuna figura che una figura che dice il contrario del capitolo.
    if ultimo["medio"] * CALO_MINIMO > primo["medio"]:
        raise AssertionError(
            f"il residuo e' passato da {primo['medio']:.2e} a {ultimo['medio']:.2e}: "
            f"meno di {CALO_MINIMO:g} volte, la figura non mostrerebbe nulla")
    if ultimo["scarto"] > TOLLERANZA_CURVA:
        raise AssertionError(
            f"scarto massimo dalla soluzione esatta {ultimo['scarto']:.3f} > "
            f"{TOLLERANZA_CURVA}: la rete non si e' incollata alla curva vera")
    # E la terza: i numeri devono restare quelli che il capitolo stampa.
    if not 0.6 < ultimo["medio"] / RESIDUO_CAPITOLO < 1.6:
        raise AssertionError(
            f"residuo finale {ultimo['medio']:.2e}, il capitolo stampa "
            f"{RESIDUO_CAPITOLO:.1e}: uno dei due va rifatto")
    if not 0.7 < ultimo["scarto"] / ERRORE_CAPITOLO < 1.4:
        raise AssertionError(
            f"scarto finale {ultimo['scarto']:.3f}, il capitolo stampa "
            f"{ERRORE_CAPITOLO}: uno dei due va rifatto")
    for prima, dopo in zip(stati, stati[1:]):
        if dopo["medio"] >= prima["medio"] or dopo["scarto"] >= prima["scarto"]:
            raise AssertionError(
                f"fra l'epoca {prima['epoca']} e la {dopo['epoca']} qualcosa "
                "risale: le tappe vanno riscelte, la figura racconta un calo")

    return griglia, esatta, [istanti[i] for i in mostrati], stati


def _numero(x: float, cifre: int = 3) -> str:
    return f"{x:.{cifre}f}".replace(".", ",")


def _potenza(x: float) -> str:
    """2,56e-01 → «2,6·10⁻¹»: due cifre, come le stampa il capitolo."""
    esp = math.floor(math.log10(x))
    mant = x / 10 ** esp
    apici = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    return f"{mant:.1f}".replace(".", ",") + "·10" + str(esp).translate(apici)


def _epoca(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def _polilinea(r: Riquadro, xs, ys) -> str:
    return " ".join(f"{r.sx(x):.0f},{r.sy(y):.1f}" for x, y in zip(xs, ys))


def costruisci() -> Figura:
    griglia, esatta, t_barre, stati = addestra()
    n = len(stati)

    y_min = min(min(min(s["curva"]) for s in stati), min(esatta)) - 0.12
    y_max = max(max(max(s["curva"]) for s in stati), max(esatta)) + 0.12
    r_max = max(max(s["residui"]) for s in stati) * 1.06

    alto = Riquadro(x=88, y=44, larg=560, alt=190,
                    xmin=-0.25, xmax=T_MAX + 0.25, ymin=y_min, ymax=y_max)
    basso = Riquadro(x=88, y=298, larg=560, alt=92,
                     xmin=-0.25, xmax=T_MAX + 0.25, ymin=0.0, ymax=r_max)

    corpo = [alto.clip("su"), alto.cornice(croce=True), basso.cornice()]
    anim = []

    # una fetta di timeline per stato: si vede uno alla volta, e l'ultimo resta
    passo = 100.0 / n
    for i in range(n):
        t0 = i * passo
        tappe = [(0.0, "opacity:1" if i == 0 else "opacity:0")]
        if i:
            tappe += [(t0 - 0.6, "opacity:0"), (t0, "opacity:1")]
        if i == n - 1:
            tappe.append((100.0, "opacity:1"))
        else:
            tappe += [(t0 + passo - 0.6, "opacity:1"), (t0 + passo, "opacity:0"),
                      (100.0, "opacity:0")]
        anim.append(keyframes(f"f{i}", tappe))

    # la soluzione esatta: ferma, è il bersaglio
    corpo.append(f'<g clip-path="url(#su)"><polyline class="esatta" points="'
                 f'{_polilinea(alto, griglia, esatta)}"/></g>')

    for i, s in enumerate(stati):
        # l'animazione sta in una classe, non in uno `style` ripetuto cinque
        # volte per stato: sono settecento byte di differenza
        moto = f' opacity="{"1" if i == n - 1 else "0"}"'
        # la curva della rete a quell'epoca
        corpo.append(f'<g class="mobile f{i}" clip-path="url(#su)"{moto}>'
                     f'<polyline class="rete" points="'
                     f'{_polilinea(alto, griglia, s["curva"])}"/></g>')
        # il residuo nei punti di collocazione, alla stessa epoca. L'altezza si
        # ricava dalla y arrotondata (così le barre restano appoggiate alla
        # stessa base) e non scende sotto il pixel: un punto di collocazione
        # dove il residuo è quasi zero resta comunque un punto controllato.
        barre = "".join(
            f'<rect x="{basso.sx(t) - 4:.0f}" y="{min(round(basso.sy(v)), basso.sy(0) - 1):.0f}"'
            f' width="8" height="{max(basso.sy(0) - round(basso.sy(v)), 1):.0f}"/>'
            for t, v in zip(t_barre, s["residui"]))
        corpo.append(f'<g class="mobile barre f{i}"{moto}>{barre}</g>')
        # e i due numeri che dicono la stessa cosa in cifre
        corpo += [
            f'<text class="eti mobile f{i}" x="{alto.x + alto.larg}" y="30" '
            f'text-anchor="end"{moto}>epoca {_epoca(s["epoca"])}</text>',
            f'<text class="val mobile f{i}" x="{alto.x + alto.larg}" y="258" '
            f'text-anchor="end"{moto}>scarto massimo dalla soluzione esatta: '
            f'{_numero(s["scarto"])}</text>',
            f'<text class="val mobile f{i}" x="{basso.x + basso.larg}" y="288" '
            f'text-anchor="end"{moto}>media di r²: {_potenza(s["medio"])}</text>',
        ]

    # il livello di partenza del residuo: senza, il fotogramma finale non ha
    # con che cosa confrontare le sue barre quasi invisibili
    partenza = sum(stati[0]["residui"]) / N_BARRE
    y_rif = basso.sy(partenza)

    # legenda, assi, didascalie fisse
    lx = alto.x
    corpo += [
        f'<line class="esatta" x1="{lx}" y1="25" x2="{lx + 26}" y2="25"/>',
        f'<text class="lbs" x="{lx + 34}" y="30">soluzione esatta</text>',
        f'<line class="rete" x1="{lx + 150}" y1="25" x2="{lx + 176}" y2="25"/>',
        f'<text class="lbs" x="{lx + 184}" y="30">rete u'
        f'<tspan class="ped" dy="3">θ</tspan><tspan dy="-3">(t)</tspan></text>',
        f'<text class="lbs" x="{alto.x - 10}" y="{alto.sy(1.0) + 5:.1f}" '
        f'text-anchor="end">1</text>',
        f'<text class="lbs" x="{alto.x - 10}" y="{alto.sy(0.0) + 5:.1f}" '
        f'text-anchor="end">0</text>',
        f'<text class="lbs" x="{basso.x}" y="288">'
        f'residuo |r(t)| in {N_BARRE} dei {N_COLLOCAZIONE} punti di collocazione</text>',
        f'<line class="rif" x1="{basso.x}" y1="{y_rif:.1f}" '
        f'x2="{basso.x + basso.larg}" y2="{y_rif:.1f}"/>',
    ]
    for tick in range(0, 11, 2):
        corpo.append(f'<text class="lbs" x="{basso.sx(tick):.0f}" y="412" '
                     f'text-anchor="middle">{tick}</text>')
    corpo += [
        f'<text class="lbs" x="{basso.x + basso.larg / 2:.0f}" y="434" '
        f'text-anchor="middle">tempo t (secondi)</text>',
        f'<text class="lbl" x="{alto.x}" y="464">'
        f'r(t) = m u″ + c u′ + k u,   con m = 1, c = 0,4, k = 4,   '
        f'u(0) = 1, u′(0) = 0</text>',
        f'<text class="lbs" x="{alto.x}" y="488">'
        f'il tratteggio segna il residuo medio alla partenza, '
        f'{_numero(partenza, 2)}</text>',
        f'<text class="lbs" x="{alto.x}" y="508">'
        f'la rete non vede un solo valore della soluzione: solo la legge e la '
        f'partenza</text>',
    ]

    return Figura(
        larghezza=720, altezza=528,
        alt="In alto la curva della rete, che addestrandosi passa da quasi "
            "piatta a sovrapposta all'oscillazione smorzata della soluzione "
            "esatta; in basso le barre del residuo nei punti di collocazione, "
            "alte all'inizio e quasi spente alla fine.",
        corpo="".join(corpo),
        stile=f"""    .esatta, .rete {{ fill:none; stroke-width:2.5;
                      stroke-linejoin:round; }}
    .esatta {{ stroke:{TEAL}; }}
    .rete   {{ stroke:{TERRACOTTA}; }}
    .barre  {{ fill:{TERRACOTTA}; fill-opacity:0.85; }}
    .rif    {{ stroke:{BORDER_STRONG}; stroke-width:1.5; stroke-dasharray:5 4; }}
    .eti    {{ font-family:{SANS}; font-size:15px; font-weight:700; fill:{INK}; }}
    .val    {{ font-family:{SANS}; font-size:13px; fill:{FG_MUTED}; }}
    .ped    {{ font-size:11px; }}
""" + "".join(f"    .f{i} {{ animation:f{i} var(--d) infinite; }}\n"
              for i in range(n)).rstrip("\n"),
        animazioni=anim,
        durata=n * 1.7,
        fermi=".mobile",
    )
