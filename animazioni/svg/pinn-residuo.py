"""Una PINN impara la fisica: il residuo che si spegne, la curva che si incolla.

I fotogrammi sono stati veri dell'addestramento, non un'interpolazione fra
inizio e fine: l'esperimento è la PINN di `PINN/come-funziona.md` riga per riga
(stessa equazione, stessa rete, stesso seme, stessa loss) e la figura fotografa
i pesi a cinque epoche.

Le due cose che il capitolo vuole far vedere insieme stanno una sopra l'altra:
in alto la curva della rete che si sovrappone alla soluzione esatta, in basso
il residuo nei punti di collocazione che cala. Sono lo stesso fatto visto da
due lati, ed è il motivo per cui una figura ferma qui perde quasi tutto.

## Il dato si misura una volta, il disegno è una funzione pura

L'addestramento non gira più mentre la figura si costruisce, e la ragione non
sono i quaranta secondi che costa: sono trentamila passi di Adam su una loss
che deriva due volte la rete, cioè trentamila occasioni perché l'ultimo bit di
una somma parziale mandi i pesi da un'altra parte. Fra due CPU con ordini di
riduzione BLAS diversi (stesso torch, stesso seme) la curva esce diversa alla
terza cifra, l'SVG cambia, e `genera.py --verifica` dichiara «da rigenerare»
su ogni macchina che non sia quella che ha committato. Un cancello che non può
tornare verde non protegge niente: prima o poi lo si aggira, ed è la fine
peggiore per un controllo.

Quindi la misura e il disegno si separano, come per la potatura iterativa:

- **il dato**: `misura()` addestra la rete, la collauda e scrive
  `animazioni/dati/pinn-residuo.json`, che è committato e porta dentro il
  seme, la configurazione e la data (`genera.py --misura pinn-residuo`);
- **il disegno**: `costruisci()` legge quel json e basta. Funzione pura dei
  dati, quindi stessi byte su ogni macchina, e nemmeno un `import torch`;
- **la verità**: `verifica()` gira in tutti e due i posti. Sui dati committati
  a ogni disegno, perché un dato committato è un dato che nessuno riapre più;
  sull'addestramento appena fatto quando si rimisura, prima che tocchi il
  disco.

Il prezzo è che l'addestramento non lo riesegue più nessun controllo
automatico, e «identico» non vuol dire «giusto»: un json vecchio verrebbe
ridisegnato fedelmente per sempre. A difendere la pagina restano tre cose. Le
asserzioni girano sul dato committato a ogni disegno, quindi un dato che non
mostra il fenomeno non arriva in pagina. Il json porta dentro la
configurazione con cui è stato misurato e il caricamento la riconfronta con
quella scritta qui, quindi chi ritocca un parametro e non rimisura trova un
rifiuto. E senza il json il generatore si ferma e dice come produrlo: una
figura che si inventa i propri numeri è peggio di una figura che manca.
"""

import json
import math
import sys
from datetime import date
from pathlib import Path

from paithon_svg import *

NOME = "pinn-residuo"
TITOLO = "una PINN impara la fisica"

QUI = Path(__file__).resolve()
RADICE = QUI.parents[2]
DATI = QUI.parents[1] / "dati" / f"{NOME}.json"

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
# Un thread. Non è una scelta di velocità: il numero di thread cambia l'ordine
# di riduzione, cioè l'ultimo bit, e su trentamila passi da lì in poi la rete
# non è più la stessa. Sta fra i parametri perché cambiarlo cambia i numeri.
THREAD = 1

TAPPE = (0, 4_000, 8_000, 16_000, 30_000)   # le epoche fotografate
N_GRIGLIA = 71      # punti con cui si disegna una curva
N_BARRE = 16        # quanti punti di collocazione mostra il pannello in basso
N_FITTA = 500       # la griglia su cui il capitolo misura lo scarto massimo

TOLLERANZA_CURVA = 0.20    # scarto massimo ammesso dalla soluzione esatta
CALO_MINIMO = 20.0         # di quante volte il residuo deve almeno scendere

# I quattro numeri che la pagina nomina, nel testo e nell'`:alt:` del `.md`,
# che è la copia che nessuna macchina confronta con la figura. Sono scritti
# **come la figura li stampa**, così il confronto è quello che il lettore fa
# davvero, e girano a ogni disegno. Se un giorno la misura li spostasse, a
# essere falsa sarebbe la pagina: si riscrive quella, non si allargano queste
# righe (e sono due misure gemelle, il residuo e lo scarto, che tirano in
# versi opposti: si rimisurano insieme).
SCARTO_PAGINA = ("1,009", "0,154")            # prima e ultima tappa
RESIDUO_PAGINA = ("2,6·10⁻¹", "7,8·10⁻³")     # media di r^2, prima e ultima


def configurazione() -> dict:
    """Tutto ciò che, cambiando, cambia i numeri: va nel json e si riconfronta.

    Serve a impedire il guasto che la separazione fra dato e disegno rende
    possibile: si ritocca un parametro qui, non si rimisura, e la figura
    continua a disegnare l'addestramento vecchio mentre le etichette
    raccontano quello nuovo (la riga dell'equazione in fondo alla figura, per
    dirne una, legge `M`, `C` e `K` e non il json).
    """
    return {
        "equazione": f"m u'' + c u' + k u = 0, m {M}, c {C}, k {K}",
        "iniziali": f"u(0) {U_0}, u'(0) {V_0}, t in [0, {T_MAX}]",
        "rete": "Linear(1,32), Tanh, Linear(32,32), Tanh, "
                "Linear(32,32), Tanh, Linear(32,1)",
        "ottimizzatore": f"Adam, lr {LR}",
        "lambda_iniziale": LAMBDA_0,
        "collocazione": N_COLLOCAZIONE,
        "epoche": EPOCHE,
        "tappe": list(TAPPE),
        "griglia": N_GRIGLIA,
        "barre": N_BARRE,
        "fitta": N_FITTA,
        "seme": SEME,
        "thread": THREAD,
    }


def soluzione_esatta(t: float) -> float:
    """u(t) = e^{-γt}(cos ω_d t + (γ/ω_d) sin ω_d t), la formula del capitolo."""
    gamma = C / (2 * M)
    omega_d = math.sqrt(K / M - gamma ** 2)
    return math.exp(-gamma * t) * (math.cos(omega_d * t)
                                   + (gamma / omega_d) * math.sin(omega_d * t))


# --------------------------------------------------------------------------
# Il dato: l'addestramento eseguito per davvero, una volta, e committato
# --------------------------------------------------------------------------
def esperimento() -> dict:
    """La PINN di `PINN/come-funziona.md`, con le fotografie alle epoche di TAPPE.

    Restituisce la griglia di disegno, la soluzione esatta su quella griglia,
    gli istanti dei punti di collocazione mostrati, e uno stato per tappa:
    l'epoca, la curva della rete, il residuo nei punti mostrati, la media di
    r^2 su tutti i punti di collocazione e lo scarto massimo dalla soluzione
    esatta.

    `torch` si importa qui dentro e non in cima al file: chi disegna la figura
    non ne ha bisogno, e una verifica che non ha bisogno di un ambiente di
    calcolo è una verifica che gira dappertutto.
    """
    import torch
    from torch import nn

    torch.set_num_threads(THREAD)   # una somma parziale sola: risultati ripetibili
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
    fitta = [T_MAX * i / (N_FITTA - 1) for i in range(N_FITTA)]
    t_f = torch.tensor(fitta, dtype=torch.float32).reshape(-1, 1)
    esatta_fitta = [soluzione_esatta(t) for t in fitta]
    istanti = t_c.detach().squeeze(1).tolist()
    ordinati = sorted(range(N_COLLOCAZIONE), key=lambda i: istanti[i])
    mostrati = [ordinati[round(i * (N_COLLOCAZIONE - 1) / (N_BARRE - 1))]
                for i in range(N_BARRE)]

    stati = []

    def tondo(v):
        # Nove decimali: questi valori escono da tensori float32, che portano
        # circa nove cifre decimali significative, quindi arrotondare piu'
        # corto butterebbe via misura, e a un decimo di pixel di distanza da
        # un arrotondamento sposterebbe anche un punto disegnato.
        return round(float(v), 9)

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
            "curva": [tondo(v) for v in curva],
            "residui": [tondo(abs(float(residuo[i]))) for i in mostrati],
            "medio": tondo((residuo ** 2).mean()),
            "scarto": tondo(max(abs(a - b) for a, b in zip(densa, esatta_fitta))),
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

    return {
        "griglia": [tondo(t) for t in griglia],
        "esatta": [tondo(v) for v in esatta],
        "t_barre": [tondo(istanti[i]) for i in mostrati],
        "stati": stati,
    }


def misura() -> Path:
    """Riesegue l'addestramento, lo collauda, e riscrive il dato committato.

        python3 animazioni/svg/genera.py --misura pinn-residuo

    Il collaudo sta **qui**, prima della scrittura: un dato che non mostra il
    fenomeno che la didascalia promette non deve arrivare al disco, o il
    prossimo che rigenera la figura si ritrova con una pagina che mente e
    nessun cancello rosso.
    """
    import torch

    corsa = esperimento()
    verifica(corsa)
    dato = {
        "_": ("L'addestramento misurato della PINN, disegnato da "
              f"animazioni/svg/{NOME}.py. Non si scrive a mano: lo riscrive "
              f"`python3 animazioni/svg/genera.py --misura {NOME}`, che prima "
              "di scrivere collauda che il fenomeno ci sia."),
        "data": date.today().isoformat(),
        "configurazione": configurazione(),
        "versioni": {
            "python": ".".join(str(v) for v in sys.version_info[:3]),
            "torch": torch.__version__,
        },
        **corsa,
    }
    DATI.parent.mkdir(parents=True, exist_ok=True)
    DATI.write_text(json.dumps(dato, indent=1, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return DATI


def dati() -> dict:
    """L'addestramento committato, con il rifiuto al posto dell'invenzione."""
    if not DATI.is_file():
        raise FileNotFoundError(
            f"manca il dato misurato: {DATI.relative_to(RADICE)}\n"
            f"    Questa figura disegna un addestramento **misurato**, non "
            f"calcolato: senza il suo json non c'è niente da disegnare, e dei "
            f"numeri inventati sarebbero peggio della figura che manca.\n"
            f"    python3 animazioni/svg/genera.py --misura {NOME}")

    dato = json.loads(DATI.read_text(encoding="utf-8"))
    if dato.get("configurazione") != configurazione():
        raise ValueError(
            f"{DATI.relative_to(RADICE)} è stato misurato con un'altra "
            f"configurazione:\n"
            f"    committata: {dato.get('configurazione')}\n"
            f"    nel file:   {configurazione()}\n"
            f"    Il disegno legge il json ma le etichette leggono le costanti "
            f"del generatore: disegnerebbe una rete e ne racconterebbe "
            f"un'altra.\n"
            f"    python3 animazioni/svg/genera.py --misura {NOME}")
    return dato


# --------------------------------------------------------------------------
# La verità: le stesse asserzioni sul dato committato e sull'esperimento
# --------------------------------------------------------------------------
def verifica(corsa: dict) -> None:
    """La figura promette un residuo che si spegne e una curva che si incolla.

    Gira in due momenti, e sono due mestieri diversi. Quando si rimisura,
    difende il dato che sta per finire sul disco. Quando si disegna, difende
    la pagina: un json committato è un file che nessuno riapre più, e la sola
    cosa che impedisce a un dato rimasto indietro di diventare una didascalia
    falsa è che il disegno si rifiuti di disegnarlo.
    """
    stati = corsa["stati"]
    assert [s["epoca"] for s in stati] == list(TAPPE), \
        f"le tappe fotografate non sono {list(TAPPE)}"
    assert len(corsa["griglia"]) == N_GRIGLIA == len(corsa["esatta"]), \
        f"la curva si disegna su {N_GRIGLIA} punti, il dato ne porta altri"
    assert len(corsa["t_barre"]) == N_BARRE, \
        f"il pannello in basso disegna {N_BARRE} barre, il dato ne porta altre"
    assert all(len(s["curva"]) == N_GRIGLIA and len(s["residui"]) == N_BARRE
               for s in stati), "una tappa porta un numero di punti diverso"
    assert all(v >= 0 for s in stati for v in s["residui"]), \
        "il pannello in basso disegna |r(t)|: un residuo negativo non ci sta"

    primo, ultimo = stati[0], stati[-1]
    # Le due promesse della figura. Se l'addestramento non converge piu', meglio
    # nessuna figura che una figura che dice il contrario del capitolo.
    assert ultimo["medio"] * CALO_MINIMO <= primo["medio"], \
        (f"il residuo è passato da {primo['medio']:.2e} a {ultimo['medio']:.2e}: "
         f"meno di {CALO_MINIMO:g} volte, la figura non mostrerebbe nulla")
    assert ultimo["scarto"] <= TOLLERANZA_CURVA, \
        (f"scarto massimo dalla soluzione esatta {ultimo['scarto']:.3f} > "
         f"{TOLLERANZA_CURVA}: la rete non si è incollata alla curva vera")
    for prima, dopo in zip(stati, stati[1:]):
        assert dopo["medio"] < prima["medio"] and dopo["scarto"] < prima["scarto"], \
            (f"fra l'epoca {prima['epoca']} e la {dopo['epoca']} qualcosa "
             f"risale: le tappe vanno riscelte, la figura racconta un calo")

    # E i numeri che la pagina nomina, confrontati **come la figura li stampa**.
    detti_scarto = (_numero(primo["scarto"]), _numero(ultimo["scarto"]))
    assert detti_scarto == SCARTO_PAGINA, \
        (f"la pagina nomina uno scarto che va da {SCARTO_PAGINA[0]} a "
         f"{SCARTO_PAGINA[1]}, la misura dà {detti_scarto[0]} e "
         f"{detti_scarto[1]}: va riscritta la pagina, non questa riga")
    detti_residuo = (_potenza(primo["medio"]), _potenza(ultimo["medio"]))
    assert detti_residuo == RESIDUO_PAGINA, \
        (f"la pagina nomina un residuo che va da {RESIDUO_PAGINA[0]} a "
         f"{RESIDUO_PAGINA[1]}, la misura dà {detti_residuo[0]} e "
         f"{detti_residuo[1]}: va riscritta la pagina, non questa riga")


# --------------------------------------------------------------------------
# Il disegno
# --------------------------------------------------------------------------
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
    corsa = dati()
    verifica(corsa)
    griglia, esatta = corsa["griglia"], corsa["esatta"]
    t_barre, stati = corsa["t_barre"], corsa["stati"]
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
