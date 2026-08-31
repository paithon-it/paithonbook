"""La potatura iterativa: si pota, si riaddestra, si pota ancora.

Il tempo qui è il contenuto in senso stretto, perché la potatura iterativa
**è** un ciclo: a ogni giro si toglie una fetta dei pesi rimasti e si lascia
alla rete qualche centinaio di passi per rimettersi in sesto. Su una figura
ferma si vedrebbe solo il punto d'arrivo, e si perderebbe la cosa che la
sezione racconta: che l'accuratezza non scende, **non scende, non scende, e poi
cade di colpo**.

I numeri vengono da un addestramento vero su una rete piccola, con le cifre
scritte a mano. Il riquadro di sinistra mostra duecentocinquantasei pesi presi
a caso fra i sedicimila del primo strato, e si spengono quando la potatura li
toglie; quello di destra è la curva dell'accuratezza contro la frazione di
pesi tolti.

Lo stato di riposo è l'ultimo giro: la griglia vuota (dei
duecentocinquantasei pesi mostrati, all'ultimo giro non ne sopravvive nessuno)
e la curva intera.

## Il dato si misura una volta, il disegno è una funzione pura

L'addestramento non gira più mentre la figura si costruisce, e non è una
questione di secondi. La potatura sceglie che cosa togliere con un `kthvalue`
su valori quasi appaiati: basta l'ultimo bit perché sopravviva un peso invece
di un altro, e da lì in poi la rete non è più la stessa. Fra due CPU con
ordini di riduzione BLAS diversi (stesso torch, stesso seme, un thread) le
coordinate arrivano a `.1f` diverse, e `genera.py --verifica` dichiara «da
rigenerare» su ogni macchina che non sia quella che ha committato. Un
cancello che non può tornare verde non protegge niente: prima o poi lo si
aggira, ed è la fine peggiore per un controllo.

Quindi la misura e il disegno si separano, che è la stessa mossa dei fermi
immagine con `impronte-fermi.json` e del collaudo delle clip con
`ambiente.json`:

- **il dato**: `misura()` esegue l'addestramento, collauda quello che ne esce
  e scrive `animazioni/dati/potatura-che-assottiglia.json`, che è committato
  e porta dentro il seme, la configurazione e la data
  (`genera.py --misura potatura-che-assottiglia`);
- **il disegno**: `costruisci()` legge quel json e basta. Funzione pura dei
  dati, quindi stessi byte su ogni macchina, e `--verifica` torna a fare il
  suo mestiere, che è dire se la figura è rimasta indietro rispetto al suo
  sorgente;
- **la verità**: `verifica()` gira in tutti e due i posti. Sui dati committati
  **sempre**, perché un dato committato è un dato che nessuno riapre e il
  disegno deve promettere il vero anche fra un anno; sull'esperimento appena
  fatto quando si rimisura, con le soglie tarate sulla distribuzione e non sul
  risultato di questa macchina.

Il rischio che questa separazione introduce va detto per intero, perché è il
prezzo pagato: l'addestramento non lo riesegue più nessun controllo
automatico, e «identico» non vuol dire «giusto», quindi un json vecchio
verrebbe ridisegnato fedelmente per sempre. Quello che difende la pagina sono
tre cose. Le asserzioni girano sul dato committato a ogni disegno, quindi un
dato che non mostra il fenomeno non arriva in pagina. Il json porta dentro la
configurazione con cui è stato misurato e il caricamento la confronta con
quella scritta qui sopra, quindi chi tocca un parametro e non rimisura trova
un rifiuto invece di una figura che disegna una rete che non esiste più. E chi
rimisura fa girare le asserzioni sull'esperimento vero, prima che il dato
tocchi il disco.

E senza il json il generatore si ferma e dice come produrlo. Una figura che si
inventa i propri numeri sarebbe peggio di una figura che manca.
"""

import json
import sys
from datetime import date
from pathlib import Path

from paithon_svg import *

NOME = "potatura-che-assottiglia"
TITOLO = "La potatura iterativa, giro dopo giro"

QUI = Path(__file__).resolve()
RADICE = QUI.parents[2]
DATI = QUI.parents[1] / "dati" / f"{NOME}.json"

# Lo strato nascosto e' largo, e i giri sono tanti, per una ragione sola: e'
# la **ridondanza** a fare l'altopiano, e senza margine l'altopiano non c'e'
# su tutte le macchine. Con 64 unita' e 9 giri la caduta al 90% di sparsita'
# valeva 1,56 punti col seme 0 e fra 2,3 e 5,5 con i semi 1..4: l'asserzione
# passava per 0,44 punti, cioe' per caso, e sul runner (stesso seme, stesso
# torch, altra CPU) la stessa curva era gia' scesa di 3,2. Un cambio di
# macchina non sposta il risultato di poco: cambiando l'ordine di riduzione
# BLAS la prima potatura sceglie pesi diversi e la traiettoria riparte da
# un'altra parte, quindi vale come un seme nuovo. Misurato su venti estrazioni
# (dieci semi per due ordini di riduzione), con 256 unita' e 13 giri la caduta
# sta fra -0,04 e 1,45 punti, media 0,53, contro una soglia di 2,0; e il crollo
# finale sta fra 23 e 49 punti, contro una soglia di 15. Non allargare l'uno
# senza rimisurare l'altro: la ridondanza che protegge l'altopiano protegge
# anche il finale, e a 256 unita' con 9 giri il crollo scendeva a 4 punti.
NASCOSTO = 256        # quanto e' largo lo strato nascosto
GIRI = 13
FETTA = 0.35          # quanta parte dei pesi rimasti si toglie a ogni giro
MOSTRATI = 256        # quanti pesi si disegnano, presi a caso
SEME = 0              # l'inizializzazione della rete
SEME_CAMPIONE = 1     # quali dei sedicimila pesi finiscono nella griglia
PASSI_PRIMA = 400     # l'addestramento prima della prima potatura
PASSI_PER_GIRO = 150  # il riaddestramento dopo ogni potatura
# Un thread. Non e' una scelta di velocita': il numero di thread cambia
# l'ordine di riduzione, cioe' l'ultimo bit, cioe' quale peso sopravvive alla
# prima potatura. Sta fra i parametri perche' cambiarlo cambia i numeri.
THREAD = 1


def configurazione() -> dict:
    """Tutto ciò che, cambiando, cambia i numeri: va nel json e si riconfronta.

    Serve a impedire il guasto che la separazione fra dato e disegno rende
    possibile: si ritocca un parametro qui, non si rimisura, e la figura
    continua a disegnare l'esperimento vecchio mentre le etichette raccontano
    quello nuovo (l'etichetta dei pesi del primo strato, per dirne una, legge
    `NASCOSTO` e non il json).
    """
    return {
        "dati": "sklearn load_digits, 30% di prova, random_state 0",
        "nascosto": NASCOSTO,
        "giri": GIRI,
        "fetta": FETTA,
        "mostrati": MOSTRATI,
        "seme": SEME,
        "seme_campione": SEME_CAMPIONE,
        "passi_prima": PASSI_PRIMA,
        "passi_per_giro": PASSI_PER_GIRO,
        "thread": THREAD,
    }


# --------------------------------------------------------------------------
# Il dato: il ciclo eseguito per davvero, una volta, e committato
# --------------------------------------------------------------------------
def esperimento() -> list[dict]:
    """Potatura iterativa su una rete piccola, giro per giro.

    Torna un passo per giro: la frazione di pesi tolti, l'accuratezza sul
    campione di prova, e quali dei `MOSTRATI` pesi disegnati sono ancora vivi
    (una stringa di uni e zeri, che in un diff si legge).

    `torch` e `sklearn` si importano qui dentro e non in cima al file: chi
    disegna la figura non ne ha bisogno, e una verifica che non ha bisogno di
    un ambiente di calcolo è una verifica che gira dappertutto.
    """
    import torch
    from torch import nn
    from torch.nn import functional as F
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    cifre = load_digits()
    Xtr, Xte, ytr, yte = train_test_split(cifre.data / 16.0, cifre.target,
                                          test_size=0.3, random_state=0)
    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    Xte = torch.tensor(Xte, dtype=torch.float32)
    ytr, yte = torch.tensor(ytr), torch.tensor(yte)

    torch.manual_seed(SEME)
    torch.set_num_threads(THREAD)
    rete = nn.Sequential(nn.Linear(64, NASCOSTO), nn.ReLU(),
                         nn.Linear(NASCOSTO, 10))
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

    def passo(maschera):
        # `campione` e non `vivi`: dieci righe piu' sotto `vivi` sono i moduli
        # dei pesi sopravvissuti, che e' un'altra cosa
        campione = maschera.flatten()[scelti].tolist()
        return {"sparsita": round(1 - maschera.mean().item(), 6),
                "accuratezza": round(accuratezza(), 6),
                "vivi": "".join("1" if v else "0" for v in campione)}

    addestra(PASSI_PRIMA)
    scelti = torch.randperm(
        W.numel(), generator=torch.Generator().manual_seed(SEME_CAMPIONE))
    scelti = scelti[:MOSTRATI]
    maschera = torch.ones_like(W)
    passi = [passo(maschera)]
    for _ in range(GIRI):
        with torch.no_grad():
            vivi = W[maschera.bool()].abs()
            soglia = vivi.kthvalue(max(int(FETTA * vivi.numel()), 1)).values
            maschera = maschera * (W.abs() >= soglia).float()
            W *= maschera
        addestra(PASSI_PER_GIRO, maschera)
        passi.append(passo(maschera))
    return passi


def misura() -> Path:
    """Riesegue l'esperimento, lo collauda, e riscrive il dato committato.

        python3 animazioni/svg/genera.py --misura potatura-che-assottiglia

    Il collaudo sta **qui**, prima della scrittura: un dato che non mostra il
    fenomeno che la didascalia promette non deve arrivare al disco, o il
    prossimo che rigenera la figura si ritrova con una pagina che mente e
    nessun cancello rosso.
    """
    import sklearn
    import torch

    passi = esperimento()
    verifica(passi)
    dato = {
        "_": ("La curva misurata della potatura iterativa, disegnata da "
              f"animazioni/svg/{NOME}.py. Non si scrive a mano: la riscrive "
              f"`python3 animazioni/svg/genera.py --misura {NOME}`, che prima "
              "di scrivere collauda che il fenomeno ci sia."),
        "data": date.today().isoformat(),
        "configurazione": configurazione(),
        "versioni": {
            "python": ".".join(str(v) for v in sys.version_info[:3]),
            "torch": torch.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "passi": passi,
    }
    DATI.parent.mkdir(parents=True, exist_ok=True)
    DATI.write_text(json.dumps(dato, indent=1, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return DATI


def dati() -> list[dict]:
    """I passi committati, con il rifiuto al posto dell'invenzione."""
    if not DATI.is_file():
        raise FileNotFoundError(
            f"manca il dato misurato: {DATI.relative_to(RADICE)}\n"
            f"    Questa figura disegna una curva **misurata**, non calcolata: "
            f"senza il suo json non c'è niente da disegnare, e dei numeri "
            f"inventati sarebbero peggio della figura che manca.\n"
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
    return dato["passi"]


# --------------------------------------------------------------------------
# La verità: le stesse asserzioni sul dato committato e sull'esperimento
# --------------------------------------------------------------------------
def verifica(passi) -> None:
    """La figura promette un altopiano e poi un crollo: ci sono davvero?

    Gira in due momenti, e sono due mestieri diversi. Quando si rimisura,
    difende il dato che sta per finire sul disco. Quando si disegna, difende
    la pagina: un json committato è un file che nessuno riapre più, e la sola
    cosa che impedisce a un dato rimasto indietro di diventare una didascalia
    falsa è che il disegno si rifiuti di disegnarlo.

    Le soglie non sono tarate sul risultato di questa macchina: stanno dove
    stanno perche' venti estrazioni (dieci semi per due ordini di riduzione
    BLAS) ci passano sotto con margine, e la macchina che costruisce il libro
    non e' questa. Chi tocca i parametri dell'esperimento rifa' quella misura
    invece di allargare la soglia: se la dimostrazione non mostra il fenomeno,
    a essere sbagliato e' l'esperimento, non la frase.
    """
    assert len(passi) == GIRI + 1, \
        (f"servono {GIRI + 1} passi (il giro zero piu' i {GIRI} giri), "
         f"ce ne sono {len(passi)}")
    tolti = [p["sparsita"] for p in passi]
    acc = [p["accuratezza"] for p in passi]
    vivi = [p["vivi"] for p in passi]

    # La griglia di sinistra: i pesi si spengono e non si riaccendono, che e'
    # esattamente la potatura iterativa. Un dato che dicesse il contrario
    # farebbe raccontare alla scena un algoritmo diverso da quello della
    # pagina, e la curva accanto continuerebbe a sembrare giusta.
    assert all(len(v) == MOSTRATI for v in vivi), \
        f"la griglia disegna {MOSTRATI} pesi, il dato ne porta altri"
    assert vivi[0] == "1" * MOSTRATI, \
        ("al giro zero non e' stato tolto ancora niente: i pesi mostrati "
         "sono tutti vivi")
    for i in range(1, len(vivi)):
        assert all(b <= a for a, b in zip(vivi[i - 1], vivi[i])), \
            f"al giro {i} un peso potato torna vivo: non si torna indietro"

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
    # Il grafico non ha un fondo elastico: un punto sotto YMIN uscirebbe dal
    # riquadro, e prima ancora vorrebbe dire che il crollo e' andato piu' giu'
    # di quanto si sia mai misurato, cioe' che l'esperimento e' cambiato.
    assert min(acc) > YMIN, \
        (f"la curva scende a {min(acc):.1f}%, sotto il fondo del grafico "
         f"({YMIN}%): il tratto finale verrebbe disegnato piu' dolce di com'e'")


# --------------------------------------------------------------------------
# La figura
# --------------------------------------------------------------------------
LARG, ALT = 760, 430
LATO = 16                     # la griglia dei pesi mostrati
CELLA = 13
GX, GY = 64, 96
INIZIO, FINE = 6.0, 86.0
# Il fondo del grafico sta sotto il punto d'arrivo peggiore che si sia
# misurato (48,9%), perche' un fondo piu' alto non taglia la curva: la
# **schiaccia** contro il bordo, e la pendenza disegnata diventa piu' dolce di
# quella vera proprio nel tratto che la figura esiste per mostrare. `verifica`
# controlla che nessun punto ci finisca sotto.
YMIN = 40


def dec(v: float, cifre: int = 0) -> str:
    return f"{v:.{cifre}f}".replace(".", ",")


def costruisci() -> Figura:
    passi = dati()
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
        for i, p in enumerate(passi):
            if p["vivi"][k] == "0":
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
                 xmin=-0.04, xmax=1.02, ymin=YMIN, ymax=100)
    corpo.append(r.cornice())
    punti = [(r.sx(p["sparsita"]), r.sy(p["accuratezza"])) for p in passi]
    for i in range(1, n):
        (x1, y1), (x2, y2) = punti[i - 1], punti[i]
        corpo.append(solo_al_giro(
            i, f'<line class="cur" x1="{x1:.1f}" y1="{y1:.1f}" '
               f'x2="{x2:.1f}" y2="{y2:.1f}"/>', permanente=True))
    for i, (x, y) in enumerate(punti):
        corpo.append(solo_al_giro(
            i, f'<circle class="pun" cx="{x:.1f}" cy="{y:.1f}" r="4"/>',
            permanente=True))

    for v in range(100, YMIN - 1, -10):
        corpo.append(f'<text class="tic" x="{r.x - 8}" y="{r.sy(v) + 4:.1f}" '
                     f'text-anchor="end">{v}%</text>')
    corpo += [
        f'<text class="ttl" x="{GX - 8}" y="{GY - 44}">i pesi che restano</text>',
        f'<text class="lbs" x="{GX - 8}" y="{GY - 26}">'
        f'{MOSTRATI} presi a caso fra i {64 * NASCOSTO:,} del primo strato</text>'
        .replace(",", "."),
        f'<text class="ttl" x="{r.x}" y="{GY - 44}">che cosa costa</text>',
        f'<text class="lbs" x="{r.x}" y="{GY - 26}">'
        f'accuratezza contro pesi tolti</text>',
        f'<text class="lbs" x="{r.x}" y="{r.y + r.alt + 20}">nessuno tolto</text>',
        f'<text class="lbs" x="{r.x + r.larg}" y="{r.y + r.alt + 20}" '
        f'text-anchor="end">tolti tutti</text>']

    # --- il cartiglio del giro --------------------------------------------
    y = GY + LATO * CELLA + 44
    corpo.append(f'<text class="lbs" x="{GX - 8}" y="{y}">al giro</text>')
    for i, p in enumerate(passi):
        corpo.append(solo_al_giro(
            i, f'<text class="big" x="{GX + 48}" y="{y}">'
               f'{dec(p["sparsita"] * 100, 1)}% tolti</text>'
               f'<text class="big2" x="{GX + 190}" y="{y}">'
               f'accuratezza {dec(p["accuratezza"], 1)}%</text>'))

    return Figura(
        larghezza=LARG, altezza=ALT,
        alt="Due riquadri affiancati. A sinistra una griglia di sedici per sedici "
            "quadratini, un campione dei pesi del primo strato di una rete: "
            "all'inizio sono tutti pieni, e giro dopo giro se ne svuotano sempre "
            "di più, fino a restare vuota o quasi. A destra la curva "
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
