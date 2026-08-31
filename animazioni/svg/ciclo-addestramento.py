"""Il rito: i cinque passi del training loop, e cosa cambia a ogni giro.

Una figura ferma del ciclo di addestramento è un cerchio con cinque frecce, e
dice soltanto in che ordine si scrivono le righe. Quello che il ciclo *fa* sta
nel tempo: i gradienti che compaiono al `backward()`, i pesi che si spostano
allo `step()`, i gradienti del giro prima che spariscono allo `zero_grad()`. È
anche la ragione per cui `zero_grad()` sta prima del `backward()` e non dopo lo
`step()`: qui si vede che fra un giro e l'altro i gradienti vecchi ci sono
ancora, e che qualcuno li deve togliere di mezzo.

**Il ciclo gira davvero.** La rete è il percettrone multistrato del capitolo in
miniatura: `Flatten → Linear(64, 16) → ReLU → Linear(16, 10)`, loss
`CrossEntropyLoss`, mini-batch da un `DataLoader`. Al posto di MNIST ci sono le
cifre 8×8 di scikit-learn, che stanno già dentro la libreria e non vanno
scaricate: stesso compito (riconoscere una cifra fra dieci), un ottavo dei
pixel. I tre valori della loss sono quelli veri dei primi tre giri e un
controllo li vuole in discesa: se un giorno non scendessero più, la figura non
si genera.

**Perché SGD e non Adam**, che è l'ottimizzatore del programma completo. Con
SGD l'aggiornamento è esattamente $w \\leftarrow w - \\eta\\,g$, quindi la barra
del gradiente e lo spostamento del peso sotto raccontano la stessa cosa: barra
in su, peso che scende. Con Adam il passo vale circa $\\eta$ qualunque sia il
gradiente, e il momento muove anche i pesi il cui gradiente è nullo: è vero, ed
è il contenuto della figura di Adam poche righe più su nel capitolo, ma qui
smentirebbe la figura invece di spiegarla. Un controllo verifica che il segno
torni per tutti i pesi mostrati.

**I sei pesi** sono presi dall'ultimo strato, i sei col gradiente più grande
all'ultimo giro: così nessuna barra è schiacciata a zero proprio nel fotogramma
che finisce in stampa. Le altezze delle barre e gli spostamenti dei pesi sono
in scala relativa, cioè le proporzioni sono vere e il fattore di scala no: uno
spostamento da 0,04 su una pista larga quanto l'intero peso sarebbe invisibile.

## Il dato si misura una volta, il disegno è una funzione pura

Il ciclo non gira più mentre la figura si costruisce, e non è una questione di
secondi (qui sono tre mini-batch, un istante). È che i sei pesi disegnati li
sceglie un `topk` sui gradienti dell'ultimo giro, cioè un ordinamento fra
valori appaiati: basta l'ultimo bit perché entri nella figura un peso invece
di un altro, e da lì cambiano sei colonne su sei. Fra due CPU con ordini di
riduzione BLAS diversi (stesso torch, stesso seme) l'SVG esce diverso, e
`genera.py --verifica` dichiara «da rigenerare» su ogni macchina che non sia
quella che ha committato. Un cancello che non può tornare verde non protegge
niente: prima o poi lo si aggira, ed è la fine peggiore per un controllo.

Quindi la misura e il disegno si separano, come per la potatura iterativa:

- **il dato**: `misura()` esegue i tre giri, li collauda e scrive
  `animazioni/dati/ciclo-addestramento.json`, che è committato e porta dentro
  il seme, la configurazione e la data
  (`genera.py --misura ciclo-addestramento`);
- **il disegno**: `costruisci()` legge quel json e basta. Funzione pura dei
  dati, quindi stessi byte su ogni macchina;
- **la verità**: `verifica()` gira in tutti e due i posti. Sui dati committati
  a ogni disegno, perché un dato committato è un dato che nessuno riapre più;
  sull'esperimento appena fatto quando si rimisura, prima che tocchi il disco.

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
import sys
from datetime import date
from pathlib import Path

from paithon_svg import *

NOME = "ciclo-addestramento"
TITOLO = "il ciclo di addestramento: cinque passi, tre giri"

QUI = Path(__file__).resolve()
RADICE = QUI.parents[2]
DATI = QUI.parents[1] / "dati" / f"{NOME}.json"

SEME, LR, BATCH = 0, 1.0, 256
GIRI, N_PESI = 3, 6
N_STATI = GIRI * 5
# Un thread. Non è una scelta di velocità: il numero di thread cambia
# l'ordine di riduzione, cioè l'ultimo bit, cioè quali sei pesi il `topk`
# porta nella figura. Sta fra i parametri perché cambiarlo può cambiare i
# numeri (su questa macchina, con questo lotto, non li cambia: fissarlo serve
# a chi rimisura altrove).
THREAD = 1

# I tre valori che la pagina nomina, nella didascalia e nell'`:alt:` del
# `.md`, che è la copia che nessuna macchina confronta con la figura. Stanno
# qui perché un'asserzione li confronti a ogni disegno: se un giorno la misura
# li spostasse, a essere falsa sarebbe la pagina, e allora si riscrive quella,
# non si allarga questa riga.
LOSS_PAGINA = ("2,35", "2,29", "2,25")

# I cinque passi nell'ordine del capitolo, con le parole del capitolo.
PASSI = [
    ("forward", "la previsione", ""),
    ("loss", "quanto abbiamo", "sbagliato"),
    ("zero_grad()", "azzera i gradienti", "vecchi"),
    ("backward()", "calcola i", "gradienti"),
    ("step()", "aggiorna", "i pesi"),
]

# Geometria
LARG, ALT = 700, 412
X0, W_BOX, H_BOX, PASSO_X, Y_BOX = 26, 112, 66, 134, 76
X_ETI = 178                      # etichette di riga, allineate a destra
X_COL, PASSO_COL = 214, 50       # le sei colonne dei pesi
Y_GRAD, H_GRAD = 240, 34         # linea dello zero dei gradienti, barra massima
Y_PESO, H_PESO = 322, 30         # centro e semiampiezza delle piste dei pesi
ESCURSIONE = 20                  # a quanti pixel corrisponde lo scostamento massimo
X_DIV, X_LOSS, X_VAL = 496, 520, 674
Y_LOSS = [226, 264, 302]
DURATA = N_STATI * 0.7           # e un giro dura un terzo del totale


def configurazione() -> dict:
    """Tutto ciò che, cambiando, cambia i numeri: va nel json e si riconfronta.

    Serve a impedire il guasto che la separazione fra dato e disegno rende
    possibile: si ritocca un parametro qui, non si rimisura, e la figura
    continua a disegnare l'esperimento vecchio mentre le etichette raccontano
    quello nuovo (l'etichetta dei pesi, per dirne una, legge `N_PESI` e non il
    json).
    """
    return {
        "dati": "sklearn load_digits, immagini 8x8 divise per 16",
        "rete": "Flatten, Linear(64,16), ReLU, Linear(16,10)",
        "loss": "CrossEntropyLoss",
        "ottimizzatore": f"SGD, lr {LR}",
        "batch": BATCH,
        "giri": GIRI,
        "n_pesi": N_PESI,
        "seme": SEME,
        "thread": THREAD,
    }


# --------------------------------------------------------------------------
# Il dato: i tre giri eseguiti per davvero, una volta, e committati
# --------------------------------------------------------------------------
def esperimento() -> dict:
    """Tre giri veri del training loop, nell'ordine in cui li scrive il capitolo.

    Restituisce la loss di ogni giro, i gradienti dei pesi mostrati a ogni
    `backward()` e lo scostamento di quei pesi dal valore di partenza.

    `torch` e `sklearn` si importano qui dentro e non in cima al file: chi
    disegna la figura non ne ha bisogno, e una verifica che non ha bisogno di
    un ambiente di calcolo è una verifica che gira dappertutto.
    """
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.datasets import load_digits

    torch.set_num_threads(THREAD)
    torch.manual_seed(SEME)
    cifre = load_digits()
    X = torch.tensor(cifre.images, dtype=torch.float32).unsqueeze(1) / 16.0
    y = torch.tensor(cifre.target, dtype=torch.long)
    dati = DataLoader(TensorDataset(X, y), batch_size=BATCH, shuffle=True,
                      generator=torch.Generator().manual_seed(SEME))

    modello = nn.Sequential(nn.Flatten(), nn.Linear(64, 16), nn.ReLU(),
                            nn.Linear(16, 10))
    criterio = nn.CrossEntropyLoss()
    ottimizzatore = optim.SGD(modello.parameters(), lr=LR)

    W = modello[3].weight                      # l'ultimo strato: 10 x 16
    perdite, gradienti = [], []
    pesi = [W.detach().flatten().clone()]

    for giro, (X_batch, y_batch) in enumerate(dati):
        if giro == GIRI:
            break
        y_pred = modello(X_batch)              # 1. forward
        perdita = criterio(y_pred, y_batch)    # 2. loss
        ottimizzatore.zero_grad()              # 3. azzera i gradienti vecchi
        perdita.backward()                     # 4. backward
        gradienti.append(W.grad.detach().flatten().clone())
        ottimizzatore.step()                   # 5. aggiorna i pesi
        perdite.append(perdita.item())
        pesi.append(W.detach().flatten().clone())

    # i sei pesi da mostrare: quelli col gradiente piu' grande all'ultimo giro
    scelti = gradienti[-1].abs().topk(N_PESI).indices.tolist()

    def arr(riga):
        # Nove decimali, e non sei: questi valori escono da tensori float32,
        # che portano circa nove cifre decimali significative, quindi
        # arrotondare piu' corto butta via misura. Non e' pignoleria di
        # rappresentazione: con sei decimali lo scostamento del secondo peso
        # all'ultimo giro (-0,060395777) diventa -0,060396, e la sua ordinata
        # passa da 336,4499894 a 336,4500138, cioe' scavalca l'arrotondamento
        # a un decimo di pixel e il cerchio si sposta.
        return [round(v, 9) for v in riga]

    return {
        "perdite": arr(perdite),
        "gradienti": [arr(g[i].item() for i in scelti) for g in gradienti],
        "scostamenti": [arr((p[i] - pesi[0][i]).item() for i in scelti)
                        for p in pesi],
        "parametri": sum(p.numel() for p in modello.parameters()),
    }


def misura() -> Path:
    """Riesegue i tre giri, li collauda, e riscrive il dato committato.

        python3 animazioni/svg/genera.py --misura ciclo-addestramento

    Il collaudo sta **qui**, prima della scrittura: un dato che non mostra il
    fenomeno che la didascalia promette non deve arrivare al disco, o il
    prossimo che rigenera la figura si ritrova con una pagina che mente e
    nessun cancello rosso.
    """
    import sklearn
    import torch

    corsa = esperimento()
    verifica(corsa)
    dato = {
        "_": ("I tre giri misurati del ciclo di addestramento, disegnati da "
              f"animazioni/svg/{NOME}.py. Non si scrive a mano: lo riscrive "
              f"`python3 animazioni/svg/genera.py --misura {NOME}`, che prima "
              "di scrivere collauda che il fenomeno ci sia."),
        "data": date.today().isoformat(),
        "configurazione": configurazione(),
        "versioni": {
            "python": ".".join(str(v) for v in sys.version_info[:3]),
            "torch": torch.__version__,
            "scikit-learn": sklearn.__version__,
        },
        **corsa,
    }
    DATI.parent.mkdir(parents=True, exist_ok=True)
    DATI.write_text(json.dumps(dato, indent=1, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return DATI


def dati() -> dict:
    """La corsa committata, con il rifiuto al posto dell'invenzione."""
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
    """La figura promette una loss in discesa e il segno del gradiente: ci sono?

    Gira in due momenti, e sono due mestieri diversi. Quando si rimisura,
    difende il dato che sta per finire sul disco. Quando si disegna, difende
    la pagina: un json committato è un file che nessuno riapre più, e la sola
    cosa che impedisce a un dato rimasto indietro di diventare una didascalia
    falsa è che il disegno si rifiuti di disegnarlo.
    """
    perdite = corsa["perdite"]
    grad, scost = corsa["gradienti"], corsa["scostamenti"]

    assert len(perdite) == GIRI, \
        f"servono {GIRI} giri, il dato ne porta {len(perdite)}"
    assert len(grad) == GIRI and all(len(g) == N_PESI for g in grad), \
        f"la figura disegna {N_PESI} pesi per giro, il dato ne porta altri"
    assert len(scost) == GIRI + 1 and all(len(s) == N_PESI for s in scost), \
        "gli scostamenti sono uno per giro piu' quello di partenza"
    assert all(v == 0 for v in scost[0]), \
        "al giro zero i pesi non si sono ancora spostati da sé"

    assert all(perdite[i] > perdite[i + 1] for i in range(GIRI - 1)), \
        f"la figura mostra una loss in discesa, l'addestramento dà {perdite}"

    # La didascalia e l'`:alt:` del `.md` nominano i tre valori: se la misura
    # li spostasse, la pagina direbbe numeri che la figura non disegna, e non
    # c'è nessun altro controllo che confronti quelle due copie.
    detti = tuple(f"{p:.2f}".replace(".", ",") for p in perdite)
    assert detti == LOSS_PAGINA, \
        (f"la pagina nomina una loss {', '.join(LOSS_PAGINA)} e la misura dà "
         f"{', '.join(detti)}: va riscritta la pagina, non questa riga")

    # La figura dice: barra del gradiente in su, peso che scende. Con SGD è
    # vero per costruzione, ma è la cosa che la figura insegna: si controlla.
    for t in range(GIRI):
        for i in range(N_PESI):
            passo = scost[t + 1][i] - scost[t][i]
            g = grad[t][i]
            assert passo * g <= 0, \
                (f"peso {i}, giro {t + 1}: gradiente {g:+.4f} e spostamento "
                 f"{passo:+.4f} hanno lo stesso segno, la figura mentirebbe")

    # Le due scale del disegno dividono per il massimo: un massimo nullo
    # sarebbe una divisione per zero, e prima ancora una figura senza niente
    # da mostrare.
    assert max(abs(v) for riga in grad for v in riga) > 0, \
        "tutti i gradienti mostrati sono nulli: non c'è nessuna barra"
    assert max(abs(v) for riga in scost for v in riga) > 0, \
        "nessuno dei pesi mostrati si è spostato"


# --------------------------------------------------------------------------
# Timeline
# --------------------------------------------------------------------------
def tappe(valori, tenuta: float = 0.72, fmt=str) -> list[tuple[float, str]]:
    """Tappe compatte da un valore per stato: una coppia a ogni cambio.

    Fra due cambi il valore resta lo stesso, e interpolare fra due keyframe
    uguali dà una costante: quindi gli stati fermi non si scrivono. Con quindici
    stati e cinque cambi si passa da trenta tappe a dodici, che sono chilobyte.
    """
    n = len(valori)
    passo = 100.0 / n
    transizione = passo * (1 - tenuta)
    out = [(0.0, fmt(valori[0]))]
    for i in range(1, n):
        if valori[i] != valori[i - 1]:
            t0 = i * passo
            out.append((max(t0 - transizione, 0.01), fmt(valori[i - 1])))
            out.append((t0, fmt(valori[i])))
    out.append((100.0, fmt(valori[-1])))
    return out


def num(v: float, cifre: int = 3) -> str:
    """Numero compatto per il CSS: niente zeri di coda, niente zero davanti al punto.

    Sono quattrocento byte su una figura che ne ha quattordicimila, e il tetto
    per una figura animata sta a quindicimila.
    """
    s = f"{v:.{cifre}f}".rstrip("0").rstrip(".")
    if s in ("", "-", "-0"):
        return "0"
    if s.startswith("0."):
        return s[1:]
    if s.startswith("-0."):
        return "-" + s[2:]
    return s


def quali_gradienti() -> list[int]:
    """Per ogni stato, di quale giro sono i gradienti in memoria (-1: nessuno).

    È il cuore didattico: al forward e alla loss ci sono ancora quelli del giro
    prima, `zero_grad()` li toglie, `backward()` mette i nuovi.
    """
    dentro = []
    for giro in range(GIRI):
        dentro += [giro - 1, giro - 1,   # forward, loss: i vecchi sono lì
                   -1,                   # zero_grad: spariscono
                   giro, giro]           # backward, step: ci sono i nuovi
    return dentro


def quali_pesi() -> list[int]:
    """Per ogni stato, quale versione dei pesi è nel modello."""
    dentro = []
    for giro in range(GIRI):
        dentro += [giro] * 4 + [giro + 1]     # cambiano solo allo step
    return dentro


# --------------------------------------------------------------------------
# Il disegno
# --------------------------------------------------------------------------
def costruisci() -> Figura:
    corsa = dati()
    verifica(corsa)
    perdite, grad = corsa["perdite"], corsa["gradienti"]
    scost, n_par = corsa["scostamenti"], corsa["parametri"]

    corpo, anim = [], []
    centri = [X0 + k * PASSO_X + W_BOX / 2 for k in range(5)]

    # --- la freccia di ritorno, che è il "e poi da capo" ---
    corpo.append(f'<path class="arco" d="M {centri[4]:.0f} {Y_BOX - 2} '
                 f'C {centri[4]:.0f} 36, {centri[0]:.0f} 36, '
                 f'{centri[0]:.0f} {Y_BOX - 12}"/>')
    corpo.append(f'<polygon class="frec" points="'
                 f'{centri[0] - 6:.0f},{Y_BOX - 14} {centri[0] + 6:.0f},{Y_BOX - 14} '
                 f'{centri[0]:.0f},{Y_BOX - 2}"/>')
    corpo.append(f'<text class="lbs" x="{(centri[0] + centri[4]) / 2:.0f}" y="32" '
                 f'text-anchor="middle">e poi da capo, col mini-batch dopo</text>')

    # --- i cinque passi ---
    for k, (nome, gloss1, gloss2) in enumerate(PASSI):
        x = X0 + k * PASSO_X
        attivo = k == 4                      # a riposo il ciclo è appena finito
        att = " att" if attivo else ""

        anim.append(keyframes(f"bx{k}", tappe(
            [k == j for j in range(5)], tenuta=0.82,
            fmt=lambda v: (f"stroke:{TERRACOTTA};stroke-width:3.5" if v
                           else f"stroke:{BORDER_STRONG};stroke-width:2"))))
        anim.append(keyframes(f"nm{k}", tappe(
            [k == j for j in range(5)], tenuta=0.82,
            fmt=lambda v: f"fill:{TERRACOTTA if v else INK}")))

        corpo.append(f'<rect class="box{att}" x="{x}" y="{Y_BOX}" width="{W_BOX}" '
                     f'height="{H_BOX}" rx="8" '
                     f'style="animation:bx{k} var(--g) infinite"/>')
        corpo.append(f'<circle class="badge" cx="{x + 15}" cy="{Y_BOX}" r="11"/>')
        corpo.append(f'<text class="num" x="{x + 15}" y="{Y_BOX + 4.5:.0f}" '
                     f'text-anchor="middle">{k + 1}</text>')
        corpo.append(f'<text class="nome{att}" x="{centri[k]:.0f}" y="{Y_BOX + 28}" '
                     f'text-anchor="middle" '
                     f'style="animation:nm{k} var(--g) infinite">{nome}</text>')
        if gloss2:
            corpo.append(f'<text class="gls" x="{centri[k]:.0f}" y="{Y_BOX + 46}" '
                         f'text-anchor="middle">{gloss1}</text>')
            corpo.append(f'<text class="gls" x="{centri[k]:.0f}" y="{Y_BOX + 60}" '
                         f'text-anchor="middle">{gloss2}</text>')
        else:
            corpo.append(f'<text class="gls" x="{centri[k]:.0f}" y="{Y_BOX + 53}" '
                         f'text-anchor="middle">{gloss1}</text>')

        if k < 4:                            # la frecciolina nello spazio dopo
            xf = x + W_BOX + (PASSO_X - W_BOX) / 2 - 4
            yc = Y_BOX + H_BOX / 2
            corpo.append(f'<polygon class="frec" points="{xf:.0f},{yc - 6:.0f} '
                         f'{xf:.0f},{yc + 6:.0f} {xf + 9:.0f},{yc:.0f}"/>')

    # --- lo stato del modello: gradienti sopra, pesi sotto ---
    quanti = f"{n_par:,}".replace(",", ".")      # 1.210, all'italiana
    corpo.append(f'<text class="lbs" x="{X_COL - 9}" y="190">'
                 f'sei dei {quanti} pesi del modello</text>')
    corpo.append(f'<line class="base" x1="{X_COL - 20}" y1="{Y_GRAD}" '
                 f'x2="{X_COL + (N_PESI - 1) * PASSO_COL + 20}" y2="{Y_GRAD}"/>')
    corpo.append(f'<line class="div" x1="{X_DIV}" y1="180" x2="{X_DIV}" y2="352"/>')

    for testo, y, cls in (("gradienti", 236, "lbl"), ("p.grad", 254, "lbs"),
                          ("pesi", 318, "lbl"),
                          ("quanto si sono spostati", 336, "lbs")):
        corpo.append(f'<text class="{cls}" x="{X_ETI}" y="{y}" '
                     f'text-anchor="end">{testo}</text>')

    scala_g = H_GRAD / max(abs(v) for riga in grad for v in riga)
    scala_w = ESCURSIONE / max(abs(v) for riga in scost for v in riga)
    in_memoria, versione = quali_gradienti(), quali_pesi()

    for i in range(N_PESI):
        cx = X_COL + i * PASSO_COL

        # la barra del gradiente: a riposo quella dell'ultimo backward
        g_fin = grad[-1][i]
        h = abs(g_fin) * scala_g
        y_barra = Y_GRAD - h if g_fin > 0 else Y_GRAD
        anim.append(keyframes(f"gb{i}", tappe(
            [0.0 if t < 0 else grad[t][i] / g_fin for t in in_memoria],
            fmt=lambda v: f"transform:scaleY({num(v)})")))
        corpo.append(f'<rect class="gbar" x="{cx - 9}" y="{y_barra:.1f}" '
                     f'width="18" height="{h:.1f}" '
                     f'style="animation:gb{i} var(--d) infinite"/>')

        # la pista del peso, col trattino sul valore di partenza
        corpo.append(f'<line class="pista" x1="{cx}" y1="{Y_PESO - H_PESO}" '
                     f'x2="{cx}" y2="{Y_PESO + H_PESO}"/>')
        corpo.append(f'<line class="tacca" x1="{cx - 10}" y1="{Y_PESO}" '
                     f'x2="{cx + 10}" y2="{Y_PESO}"/>')
        cy = [Y_PESO - s[i] * scala_w for s in scost]
        anim.append(keyframes(f"pw{i}", tappe(
            [cy[t] for t in versione],
            fmt=lambda v: f"transform:translateY({num(v - cy[-1], 1)}px)")))
        corpo.append(f'<circle class="peso" cx="{cx}" cy="{cy[-1]:.1f}" r="7" '
                     f'style="animation:pw{i} var(--d) infinite"/>')

    # --- la loss, un valore per giro ---
    corpo.append(f'<text class="lbl" x="{X_LOSS}" y="194">loss</text>')
    for g, perdita in enumerate(perdite):
        corpo.append(f'<text class="lbs" x="{X_LOSS}" y="{Y_LOSS[g]}">'
                     f'giro {g + 1}</text>')
        # compare quando si accende il passo «loss» del proprio giro
        stato = ["fuori"] * (g * 5 + 1) + ["ora"] * 5 + ["prima"] * N_STATI
        anim.append(keyframes(f"ls{g}", tappe(
            stato[:N_STATI], fmt=lambda v: (
                f"opacity:0;fill:{TERRACOTTA}" if v == "fuori" else
                f"opacity:1;fill:{TERRACOTTA}" if v == "ora" else
                f"opacity:1;fill:{FG_MUTED}"))))
        colore = TERRACOTTA if g == GIRI - 1 else FG_MUTED
        valore = f"{perdita:.2f}".replace(".", ",")
        corpo.append(f'<text class="val" x="{X_VAL}" y="{Y_LOSS[g]}" '
                     f'text-anchor="end" fill="{colore}" '
                     f'style="animation:ls{g} var(--d) infinite">{valore}</text>')

    corpo.append('<text class="lbl" x="26" y="394">senza zero_grad() i gradienti '
                 'nuovi si sommerebbero a quelli del giro prima</text>')

    return Figura(
        larghezza=LARG, altezza=ALT,
        alt="Il ciclo di addestramento in cinque passi (forward, loss, "
            "zero_grad, backward, step) percorso tre volte. Al backward le "
            "barre dei gradienti si riempiono, allo step i sei pesi mostrati "
            "si spostano lungo la loro pista, allo zero_grad del giro dopo le "
            "barre tornano a zero. La loss dei tre giri scende: "
            + ", ".join(f"{p:.2f}".replace(".", ",") for p in perdite) + ".",
        corpo="".join(corpo),
        stile=f"""    svg   {{ --g: {DURATA / GIRI:.2f}s; }}
    .box  {{ fill:{CREAM}; stroke:{BORDER_STRONG}; stroke-width:2; }}
    .box.att {{ stroke:{TERRACOTTA}; stroke-width:3.5; }}
    .badge {{ fill:{CREAM}; stroke:{BORDER_STRONG}; stroke-width:1.5; }}
    .num  {{ font-family:{SANS}; font-size:12px; font-weight:700; fill:{FG_MUTED}; }}
    .nome {{ font-family:{SANS}; font-size:14px; font-weight:700; fill:{INK}; }}
    .nome.att {{ fill:{TERRACOTTA}; }}
    .gls  {{ font-family:{SANS}; font-size:11.5px; fill:{FG_MUTED}; }}
    .arco {{ stroke:{BORDER_STRONG}; stroke-width:2; fill:none; }}
    .frec {{ fill:{BORDER_STRONG}; }}
    .base {{ stroke:{BORDER_STRONG}; stroke-width:2; }}
    .div  {{ stroke:{BORDER}; stroke-width:1.5; }}
    .gbar {{ fill:{OCRA}; transform-box:view-box; transform-origin:0px {Y_GRAD}px; }}
    .pista {{ stroke:{BORDER}; stroke-width:2; }}
    .tacca {{ stroke:{BORDER_STRONG}; stroke-width:2.5; }}
    .peso {{ fill:{TEAL}; }}
    .val  {{ font-family:{SANS}; font-size:17px; font-weight:700; }}""",
        animazioni=anim,
        durata=DURATA,
        fermi=".box, .nome, .gbar, .peso, .val",
    )
