"""Il grokking: le due curve che restano separate, e poi si ricongiungono.

Il tempo qui è il contenuto in senso letterale, ed è l'unico modo di far
vedere la cosa che la sezione racconta: l'accuratezza sulle somme già viste
arriva al cento per cento **subito**, quella sulle somme mai viste resta
sotto il livello del sorteggio per decine di migliaia di passi, e poi sale di
colpo.
Ferma, una figura del genere si legge come un grafico qualunque; in movimento
si vive la lunga attesa, che è la parte che sorprende.

Lo stato di riposo è l'ultimo istante: le due curve intere, e il cartiglio
dell'ultimo passo.

## Il dato si misura una volta, il disegno è una funzione pura

Questa figura disegna un addestramento vero, e un addestramento non dà gli
stessi byte su due macchine: l'`argmax` con cui si conta l'accuratezza sceglie
fra logit quasi appaiati, e due ordini di riduzione BLAS diversi (stesso
torch, stesso seme) danno da lì in poi due curve diverse. Vale la regola delle
altre figure che addestrano: il dato si misura una volta e si committa,
`costruisci()` legge quel json e non importa nemmeno `torch`.

- **il dato**: `misura()` esegue l'addestramento, collauda quello che ne esce e
  scrive `animazioni/dati/grokking-tarda.json`
  (`genera.py --misura grokking-tarda`);
- **il disegno**: `costruisci()` legge quel json e basta;
- **la verità**: `verifica()` gira in tutti e due i posti, sul dato committato
  a ogni disegno e sull'esperimento quando lo si rimisura.

Il prezzo è lo stesso delle altre: l'addestramento non lo riesegue più nessun
controllo automatico. A difendere la pagina sono le asserzioni, che girano sul
dato committato a ogni disegno, e il confronto fra la configurazione scritta
qui e quella dentro il json, che rifiuta un dato misurato con altri parametri.
"""

import json
import sys
from datetime import date
from pathlib import Path

from paithon_svg import *

NOME = "grokking-tarda"
TITOLO = "Il grokking: la generalizzazione che arriva tardi"

QUI = Path(__file__).resolve()
RADICE = QUI.parents[2]
DATI = QUI.parents[1] / "dati" / f"{NOME}.json"

# Il compito e' l'addizione modulo un primo: 97 x 97 coppie, se ne mostra il
# 30% e si chiede il resto. La rete e' piccola apposta, perche' il fenomeno si
# vede su reti piccole e dataset generati a tavolino.
#
# Il `weight_decay` alto non e' un dettaglio di taratura, e' il soggetto della
# figura: e' la multa sui pesi a guidare la fase in cui il circuito che
# memorizza viene smontato. Con `weight_decay` piccolo l'attesa si allunga
# oltre ogni budget di calcolo ragionevole, e la figura non mostrerebbe niente.
P = 97                  # il modulo, primo
FRAZIONE = 0.3          # quanta parte delle coppie finisce in addestramento
LARGHEZZA = 128         # la dimensione dell'embedding e dello strato nascosto
PASSI = 26000           # a batch pieno: un passo vede tutte le coppie viste
CAMPIONE = 250          # ogni quanti passi si misura l'accuratezza
LR = 1e-3
WEIGHT_DECAY = 1.0
BETA1, BETA2 = 0.9, 0.98
SEME = 0                # l'inizializzazione della rete
SEME_SPLIT = 0          # quali coppie finiscono in addestramento
# Un thread. Come nelle altre figure che addestrano, non e' una scelta di
# velocita': il numero di thread cambia l'ordine di riduzione, cioe' l'ultimo
# bit, cioe' quale logit vince un `argmax` fra valori appaiati.
THREAD = 1


def configurazione() -> dict:
    """Tutto ciò che, cambiando, cambia i numeri: va nel json e si riconfronta."""
    return {
        "compito": f"addizione modulo {P}, tutte le {P * P} coppie",
        "p": P,
        "frazione": FRAZIONE,
        "larghezza": LARGHEZZA,
        "passi": PASSI,
        "campione": CAMPIONE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "betas": [BETA1, BETA2],
        "seme": SEME,
        "seme_split": SEME_SPLIT,
        "thread": THREAD,
    }


# --------------------------------------------------------------------------
# Il dato: l'addestramento eseguito per davvero, una volta, e committato
# --------------------------------------------------------------------------
def esperimento() -> list[dict]:
    """Addestra e campiona le due accuratezze ogni `CAMPIONE` passi.

    `torch` si importa qui dentro e non in cima al file: chi disegna la figura
    non ne ha bisogno, e una verifica che non chiede un ambiente di calcolo è
    una verifica che gira dappertutto.
    """
    import torch
    from torch import nn
    from torch.nn import functional as F

    torch.set_num_threads(THREAD)
    a = torch.arange(P).repeat_interleave(P)
    b = torch.arange(P).repeat(P)
    y = (a + b) % P
    ordine = torch.randperm(
        P * P, generator=torch.Generator().manual_seed(SEME_SPLIT))
    n_tr = int(FRAZIONE * P * P)
    tr, te = ordine[:n_tr], ordine[n_tr:]

    torch.manual_seed(SEME)
    emb = nn.Embedding(P, LARGHEZZA)
    testa = nn.Sequential(nn.Linear(2 * LARGHEZZA, LARGHEZZA), nn.ReLU(),
                          nn.Linear(LARGHEZZA, P))
    parametri = list(emb.parameters()) + list(testa.parameters())
    opt = torch.optim.AdamW(parametri, lr=LR, weight_decay=WEIGHT_DECAY,
                            betas=(BETA1, BETA2))

    def uscita(idx):
        return testa(torch.cat([emb(a[idx]), emb(b[idx])], dim=-1))

    def accuratezza(idx):
        with torch.no_grad():
            return (uscita(idx).argmax(1) == y[idx]).float().mean().item() * 100

    campioni = []
    for passo in range(PASSI + 1):
        if passo % CAMPIONE == 0:
            campioni.append({"passo": passo,
                             "viste": round(accuratezza(tr), 6),
                             "nuove": round(accuratezza(te), 6)})
        if passo == PASSI:
            break
        opt.zero_grad()
        F.cross_entropy(uscita(tr), y[tr]).backward()
        opt.step()
    return campioni


def misura() -> Path:
    """Riesegue l'addestramento, lo collauda, e riscrive il dato committato.

        python3 animazioni/svg/genera.py --misura grokking-tarda

    Il collaudo sta **qui**, prima della scrittura: un dato in cui il grokking
    non si vede non deve arrivare al disco, o chi rigenera la figura si ritrova
    una didascalia che promette un fenomeno che il disegno non mostra.
    """
    import torch

    campioni = esperimento()
    verifica(campioni)
    dato = {
        "_": ("Le due curve misurate del grokking, disegnate da "
              f"animazioni/svg/{NOME}.py. Non si scrive a mano: la riscrive "
              f"`python3 animazioni/svg/genera.py --misura {NOME}`, che prima "
              "di scrivere collauda che il fenomeno ci sia."),
        "data": date.today().isoformat(),
        "configurazione": configurazione(),
        "versioni": {
            "python": ".".join(str(v) for v in sys.version_info[:3]),
            "torch": torch.__version__,
        },
        "campioni": campioni,
    }
    DATI.parent.mkdir(parents=True, exist_ok=True)
    DATI.write_text(json.dumps(dato, indent=1, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return DATI


def dati() -> list[dict]:
    """I campioni committati, con il rifiuto al posto dell'invenzione."""
    if not DATI.is_file():
        raise FileNotFoundError(
            f"manca il dato misurato: {DATI.relative_to(RADICE)}\n"
            f"    Questa figura disegna due curve **misurate**, non calcolate: "
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
            f"del generatore: disegnerebbe un addestramento e ne racconterebbe "
            f"un altro.\n"
            f"    python3 animazioni/svg/genera.py --misura {NOME}")
    return dato["campioni"]


# --------------------------------------------------------------------------
# La verità: le stesse asserzioni sul dato committato e sull'esperimento
# --------------------------------------------------------------------------
CASO = 100.0 / P          # tirando a caso si indovina una volta su P


def verifica(campioni) -> None:
    """La figura promette tre cose. Ci sono davvero?

    Che le somme già viste si imparino presto; che quelle mai viste restino
    **sotto** il livello del sorteggio per un tratto lungo **dopo** che le
    prime sono a posto, ed è la parte che dà il nome alla figura; e che alla
    fine salgano. Le soglie sono larghe apposta: servono a dire se il fenomeno
    c'è, non a fotografare i numeri di questa macchina.
    """
    assert campioni, "nessun campione: il dato è vuoto"
    assert campioni[0]["passo"] == 0 and campioni[-1]["passo"] == PASSI, \
        (f"i campioni devono coprire da 0 a {PASSI}, coprono da "
         f"{campioni[0]['passo']} a {campioni[-1]['passo']}")

    viste = [c["viste"] for c in campioni]
    nuove = [c["nuove"] for c in campioni]

    memorizza = next((c["passo"] for c, v in zip(campioni, viste) if v > 99.0),
                     None)
    assert memorizza is not None and memorizza < 0.1 * PASSI, \
        (f"la rete deve imparare a memoria presto, e il primo campione sopra "
         f"il 99% sulle viste è a {memorizza}")

    # L'attesa: dopo aver memorizzato, sulle nuove si resta **sotto** il
    # livello del sorteggio per un tratto lungo. E' la promessa della
    # didascalia, ed e' cio' che distingue questa curva da una qualunque curva
    # che sale: una rete che ha imparato a memoria non tira a indovinare sulle
    # coppie nuove, sbaglia, e sbaglia piu' spesso di chi tirasse a caso.
    # La soglia e' CASO e non un suo multiplo proprio perche' e' quello che la
    # didascalia promette; il margine c'e', perche' nel tratto piatto si sta
    # a un decimo del sorteggio, non a un pelo sotto.
    attesa = [c["passo"] for c in campioni
              if c["passo"] > memorizza and c["nuove"] < CASO]
    assert attesa and max(attesa) > 0.3 * PASSI, \
        (f"l'attesa è troppo corta: sulle nuove si torna sopra il "
         f"{CASO:.2f}% del sorteggio già a {max(attesa) if attesa else 0} passi")

    assert nuove[-1] > 95.0, \
        f"alla fine il grokking deve essere avvenuto, e si arriva a {nuove[-1]:.1f}%"
    assert viste[-1] > 99.0, \
        f"le somme viste restano imparate, e si arriva a {viste[-1]:.1f}%"

    # Il salto e' un salto: dal 10% al 90% in una frazione piccola della corsa.
    su10 = next(c["passo"] for c in campioni if c["nuove"] > 10.0)
    su90 = next(c["passo"] for c in campioni if c["nuove"] > 90.0)
    assert su90 - su10 < 0.25 * PASSI, \
        (f"la transizione deve essere netta, e qui va da {su10} a {su90} "
         f"passi, cioè {(su90 - su10) / PASSI:.0%} della corsa")

    # Il grafico non ha un fondo elastico: parte da zero e arriva a cento.
    assert min(min(viste), min(nuove)) >= 0.0, "un'accuratezza negativa"
    assert max(max(viste), max(nuove)) <= 100.0, "un'accuratezza sopra il 100%"


# --------------------------------------------------------------------------
# La figura
# --------------------------------------------------------------------------
LARG, ALT = 760, 420
TAPPE = 14                # in quanti scatti si rivelano le curve
INIZIO, FINE = 4.0, 88.0


def dec(v: float, cifre: int = 0) -> str:
    return f"{v:.{cifre}f}".replace(".", ",")


def migliaia(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def costruisci() -> Figura:
    campioni = dati()
    verifica(campioni)

    # Un po' di margine sopra e sotto: a `ymin=0` e `ymax=100` le due curve
    # correrebbero **sulla** cornice, e il tratto piatto delle nuove si
    # confonderebbe con il bordo del riquadro invece di leggersi come uno zero.
    r = Riquadro(x=92, y=70, larg=568, alt=252, xmin=0, xmax=PASSI,
                 ymin=-4, ymax=104)
    corpo = [r.cornice()]

    # Le tacche orizzontali. Il livello del sorteggio non si disegna, e non
    # perche' non ci sia posto: l'1,03% di una su 97 cade a due pixel e mezzo
    # dallo zero, e il tratto piatto delle nuove sta ancora piu' giu' (0,1% in
    # media), cioe' sotto il tiro a indovinare. Una riga li' non aggiungerebbe
    # niente da vedere; a dirlo e' la didascalia.
    for v in (0, 25, 50, 75, 100):
        corpo.append(f'<line class="axc" x1="{r.x}" y1="{r.sy(v):.1f}" '
                     f'x2="{r.x + r.larg}" y2="{r.sy(v):.1f}"/>'
                     f'<text class="tic" x="{r.x - 8}" y="{r.sy(v) + 4:.1f}" '
                     f'text-anchor="end">{v}%</text>')
    # Le tacche verticali cadono su multipli tondi di mille, non su quarti di
    # `PASSI`: con 26.000 passi i quarti danno 6.500 e 19.500, e l'ultima
    # finiva addosso all'etichetta dell'asse.
    for v in range(0, PASSI + 1, 5000):
        corpo.append(f'<text class="tic" x="{r.sx(v):.1f}" '
                     f'y="{r.y + r.alt + 20}" text-anchor="middle">'
                     f'{migliaia(v)}</text>')

    # I due tracciati, spezzati in TAPPE gruppi che si accendono uno dopo
    # l'altro e non si spengono piu': a riposo ci sono tutti, cioe' le curve
    # intere, che e' lo stato finale.
    n = len(campioni)
    confini = [round(k * (n - 1) / TAPPE) for k in range(TAPPE + 1)]
    istante = [INIZIO + (FINE - INIZIO) * k / TAPPE for k in range(TAPPE + 1)]

    anim = []
    for k in range(1, TAPPE + 1):
        t = istante[k]
        anim.append(keyframes(f"c{k}", [(0.0, "opacity:0"),
                                        (max(t - 0.6, 0.0), "opacity:0"),
                                        (t, "opacity:1"), (100.0, "opacity:1")]))

    def tratto(chiave: str, i0: int, i1: int) -> str:
        punti = " ".join(f"{r.sx(c['passo']):.1f},{r.sy(c[chiave]):.1f}"
                         for c in campioni[i0:i1 + 1])
        return f'<polyline class="{chiave}" points="{punti}"/>'

    for k in range(1, TAPPE + 1):
        i0, i1 = confini[k - 1], confini[k]
        corpo.append(f'<g style="animation:c{k} var(--d) linear infinite">'
                     f'{tratto("viste", i0, i1)}{tratto("nuove", i0, i1)}</g>')

    # Il cartiglio: quanti passi, e le due accuratezze a quel punto.
    y = r.y + r.alt + 72
    for k in range(TAPPE + 1):
        c = campioni[confini[k]]
        t = istante[k]
        fine_k = istante[k + 1] if k < TAPPE else 100.0
        anim.append(keyframes(f"k{k}", [
            (0.0, "opacity:0"), (max(t - 0.6, 0.0), "opacity:0"),
            (t, "opacity:1"),
            *([(fine_k - 0.6, "opacity:1"), (fine_k, "opacity:0"),
               (100.0, "opacity:0")] if k < TAPPE else [(100.0, "opacity:1")])]))
        base = "" if k == TAPPE else "opacity:0;"
        corpo.append(
            f'<g style="{base}animation:k{k} var(--d) linear infinite">'
            f'<text class="lbs" x="{r.x}" y="{y}">al passo</text>'
            f'<text class="big" x="{r.x + 66}" y="{y}">'
            f'{migliaia(c["passo"])}</text>'
            f'<text class="cviste" x="{r.x + 190}" y="{y}">'
            f'somme viste {dec(c["viste"])}%</text>'
            f'<text class="cnuove" x="{r.x + 380}" y="{y}">'
            f'somme nuove {dec(c["nuove"])}%</text></g>')

    corpo += [
        f'<text class="ttl" x="{r.x}" y="{r.y - 40}">'
        f'Addizione modulo {P}: quando arriva la generalizzazione</text>',
        f'<text class="lbs" x="{r.x}" y="{r.y - 20}">'
        f'accuratezza sulle {migliaia(int(FRAZIONE * P * P))} somme mostrate '
        f'e sulle {migliaia(P * P - int(FRAZIONE * P * P))} mai viste, '
        f'passo dopo passo</text>',
        f'<text class="lbs" x="{r.x + r.larg / 2:.1f}" '
        f'y="{r.y + r.alt + 42}" text-anchor="middle">'
        f'passi di addestramento</text>',
        f'<text class="cviste" x="{r.x + r.larg - 10}" y="{r.sy(100) + 34:.1f}" '
        f'text-anchor="end">somme già viste</text>',
        f'<text class="cnuove" x="{r.x + 10}" y="{r.sy(0) - 14:.1f}">'
        f'somme mai viste</text>']

    return Figura(
        larghezza=LARG, altezza=ALT,
        alt="Due curve di accuratezza contro i passi di addestramento di una "
            "rete che impara l'addizione modulo 97. La curva delle somme già "
            "mostrate sale al cento per cento nei primissimi passi e ci resta. "
            "Quella delle somme mai viste resta appiattita sullo zero per più "
            "di metà del grafico, poi sale ripidamente e raggiunge il cento per "
            "cento verso la fine. Un cartiglio sotto il "
            "grafico segue l'avanzamento e riporta, a ogni scatto, il numero "
            "di passi e le due accuratezze.",
        corpo="".join(corpo),
        stile=f"""    .viste  {{ fill:none; stroke:{TEAL}; stroke-width:2.5; }}
    .nuove  {{ fill:none; stroke:{TERRACOTTA}; stroke-width:2.5; }}
    .tic    {{ font-family:{SANS}; font-size:12px; fill:{FG_MUTED}; }}
    .big    {{ font-family:{SANS}; font-size:15px; font-weight:600; fill:{INK}; }}
    .cviste {{ font-family:{SANS}; font-size:13px; font-weight:600; fill:{TEAL}; }}
    .cnuove {{ font-family:{SANS}; font-size:13px; font-weight:600;
               fill:{TERRACOTTA}; }}""",
        animazioni=anim,
        durata=14.0,
        fermi="g",
    )
