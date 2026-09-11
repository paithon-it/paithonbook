"""Le due lune che si raddrizzano, un accoppiamento alla volta.

Qui il tempo è il contenuto per la ragione che dà il nome alla famiglia: un
flusso non è una trasformazione sola, è una successione, e ciò che si vuole far
vedere è la nuvola che *scorre* dai dati verso la gaussiana. Ferma, la scena
sarebbe una nuvola di punti qualunque; in movimento si vede la cosa che il
testo racconta a parole e che nessun numero stampato mostra: a ogni strato si
muove **una sola** delle due coordinate, l'altra sta ferma dov'è, ed è
esattamente il vincolo che rende il determinante gratis.

Lo stato di riposo è l'ultimo, cioè il latente: la nuvola gaussiana con la sua
deviazione misurata.

I due colori sono le due lune di partenza, e servono a riconoscere la forma
iniziale. Che restino separate alla fine la scena non lo dice, perché non è
vero: si sovrappongono, come devono, essendo il latente una gaussiana sola.

## Il dato si misura una volta, il disegno è una funzione pura

Questa figura disegna un **addestramento vero**, quello del blocco della
sezione, con il suo seme. Un addestramento non dà gli stessi byte su due
macchine, quindi vale la regola delle altre figure che addestrano: il dato si
misura una volta e si committa, `costruisci()` legge quel json e non importa
nemmeno `torch`.

- **il dato**: `misura()` riesegue l'addestramento, collauda quello che ne esce
  e scrive `animazioni/dati/lune-si-raddrizzano.json`
  (`genera.py --misura lune-si-raddrizzano`);
- **il disegno**: `costruisci()` legge quel json e basta;
- **la verità**: `verifica()` gira in tutti e due i posti, sul dato committato a
  ogni disegno e sull'esperimento quando lo si rimisura.

L'asserzione che conta è quella sulla coordinata ferma, e non è una tolleranza:
lo scarto dev'essere **zero**, perché quella metà non viene toccata affatto.
Se un domani la costruzione cambiasse, la figura si rifiuterebbe di uscire
invece di mostrare una scena che promette una cosa e ne mostra un'altra.
"""

import json
import sys
from datetime import date
from pathlib import Path

from paithon_svg import *

NOME = "lune-si-raddrizzano"
TITOLO = "le due lune si raddrizzano, un accoppiamento alla volta"

QUI = Path(__file__).resolve()
RADICE = QUI.parents[2]
DATI = QUI.parents[1] / "dati" / f"{NOME}.json"

# La configurazione è quella del blocco della sezione: stesso dataset, stesso
# seme, stessa rete, stessi passi. Cambiarla qui senza rimisurare fa fallire il
# caricamento, e cambiarla senza cambiarla anche là farebbe raccontare alla
# figura un addestramento diverso da quello che il lettore può rifare.
N_PUNTI = 2000          # le lune su cui si addestra
RUMORE = 0.06
SEME_LUNE = 0
N_ACCOPPIAMENTI = 6
NASCOSTO = 64
PASSI = 1500
LR = 3e-3
SEME = 0
# Un thread, come nelle altre figure che addestrano: il numero di thread cambia
# l'ordine di riduzione, cioè l'ultimo bit, cioè la traiettoria.
THREAD = 1
# Quanti punti finiscono nel disegno: uno ogni PASSO_DISEGNO. Non è una scelta
# estetica ma di peso, perché ogni punto disegnato si porta dietro il proprio
# @keyframes con sette tappe. Le statistiche restano quelle della nuvola
# intera, che è ciò di cui parla la didascalia.
PASSO_DISEGNO = 20


def configurazione() -> dict:
    """Tutto ciò che, cambiando, cambia i numeri: va nel json e si riconfronta."""
    return {
        "dati": f"two moons, {N_PUNTI} punti, rumore {RUMORE}, "
                f"seme {SEME_LUNE}, standardizzati",
        "accoppiamenti": N_ACCOPPIAMENTI,
        "nascosto": NASCOSTO,
        "passi": PASSI,
        "lr": LR,
        "seme": SEME,
        "thread": THREAD,
        "passo_disegno": PASSO_DISEGNO,
    }


# --------------------------------------------------------------------------
# Il dato: l'addestramento del blocco della sezione, eseguito una volta
# --------------------------------------------------------------------------
def esperimento() -> dict:
    """Addestra il flusso e raccoglie la nuvola dopo ogni accoppiamento.

    Il modello è quello della sezione, ricopiato senza cambiare niente: se le
    due versioni divergessero, la figura mostrerebbe un flusso e la pagina ne
    stamperebbe un altro. `torch` si importa qui dentro e non in cima al file,
    perché chi disegna la figura non ne ha bisogno.
    """
    import math

    import torch
    from torch import nn
    from sklearn.datasets import make_moons

    torch.set_num_threads(THREAD)

    class Accoppiamento(nn.Module):
        def __init__(self, scambia, nascosto=NASCOSTO):
            super().__init__()
            self.scambia = scambia
            self.rete = nn.Sequential(nn.Linear(1, nascosto), nn.Tanh(),
                                      nn.Linear(nascosto, nascosto), nn.Tanh(),
                                      nn.Linear(nascosto, 2))

        def _st(self, fissa):
            s, t = self.rete(fissa.unsqueeze(1)).chunk(2, dim=1)
            return torch.tanh(s).squeeze(1), t.squeeze(1)

        def _ricomponi(self, fissa, mobile):
            return (torch.stack([mobile, fissa], 1) if self.scambia
                    else torch.stack([fissa, mobile], 1))

        def avanti(self, x):
            fissa, mobile = ((x[:, 1], x[:, 0]) if self.scambia
                             else (x[:, 0], x[:, 1]))
            s, t = self._st(fissa)
            return self._ricomponi(fissa, mobile * torch.exp(s) + t), s

    class Flusso(nn.Module):
        def __init__(self, n=N_ACCOPPIAMENTI):
            super().__init__()
            self.passi = nn.ModuleList(Accoppiamento(i % 2 == 1)
                                       for i in range(n))

        def avanti(self, x):
            logdet = torch.zeros(len(x))
            for p in self.passi:
                x, s = p.avanti(x)
                logdet = logdet + s
            return x, logdet

        def log_densita(self, x):
            z, logdet = self.avanti(x)
            log_gauss = -0.5 * (z ** 2).sum(1) - math.log(2 * math.pi)
            return log_gauss + logdet

    torch.manual_seed(SEME)
    X, luna = make_moons(N_PUNTI, noise=RUMORE, random_state=SEME_LUNE)
    X = torch.tensor(X, dtype=torch.float32)
    X = (X - X.mean(0)) / X.std(0)

    flusso = Flusso()
    opt = torch.optim.Adam(flusso.parameters(), lr=LR)
    for _ in range(PASSI):
        perdita = -flusso.log_densita(X).mean()
        opt.zero_grad()
        perdita.backward()
        opt.step()

    # La nuvola dopo ogni accoppiamento, e quale coordinata quello strato
    # tiene ferma: 0 è l'orizzontale, 1 la verticale.
    stadi, ferma = [], []
    with torch.no_grad():
        x = X.clone()
        stadi.append(x)
        for p in flusso.passi:
            x, _ = p.avanti(x)
            stadi.append(x)
            ferma.append(1 if p.scambia else 0)

    scelti = list(range(0, N_PUNTI, PASSO_DISEGNO))
    return {
        "ferma": ferma,
        "luna": [int(luna[i]) for i in scelti],
        "stadi": [{"dev": [round(v, 6) for v in s.std(0).tolist()],
                   "media": [round(v, 6) for v in s.mean(0).tolist()],
                   "punti": [[round(s[i, 0].item(), 6),
                              round(s[i, 1].item(), 6)] for i in scelti]}
                  for s in stadi],
    }


def misura() -> Path:
    """Riesegue l'addestramento, lo collauda, e riscrive il dato committato.

        python3 animazioni/svg/genera.py --misura lune-si-raddrizzano

    Il collaudo sta **qui**, prima della scrittura: una nuvola che alla fine non
    è gaussiana, o uno strato che muove tutte e due le coordinate, non deve
    arrivare al disco, o chi rigenera si ritrova una didascalia che promette
    quello che il disegno non fa.
    """
    import torch

    misurato = esperimento()
    verifica(misurato)
    dato = {
        "_": ("La nuvola delle due lune dopo ogni accoppiamento del flusso, "
              f"misurata da animazioni/svg/{NOME}.py. Non si scrive a mano: la "
              f"riscrive `python3 animazioni/svg/genera.py --misura {NOME}`, "
              "che prima di scrivere collauda che la scena ci sia."),
        "data": date.today().isoformat(),
        "configurazione": configurazione(),
        "versioni": {
            "python": ".".join(str(v) for v in sys.version_info[:3]),
            "torch": torch.__version__,
        },
        **misurato,
    }
    DATI.parent.mkdir(parents=True, exist_ok=True)
    DATI.write_text(json.dumps(dato, indent=1, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return DATI


def dati() -> dict:
    """Il dato committato, con il rifiuto al posto dell'invenzione."""
    if not DATI.is_file():
        raise FileNotFoundError(
            f"manca il dato misurato: {DATI.relative_to(RADICE)}\n"
            f"    Questa figura disegna un addestramento **vero**, non un "
            f"calcolo: senza il suo json non c'è niente da disegnare, e dei "
            f"punti inventati sarebbero peggio della figura che manca.\n"
            f"    python3 animazioni/svg/genera.py --misura {NOME}")

    dato = json.loads(DATI.read_text(encoding="utf-8"))
    if dato.get("configurazione") != configurazione():
        raise ValueError(
            f"{DATI.relative_to(RADICE)} è stato misurato con un'altra "
            f"configurazione:\n"
            f"    committata: {dato.get('configurazione')}\n"
            f"    nel file:   {configurazione()}\n"
            f"    Il disegno legge il json ma le etichette leggono le costanti "
            f"del generatore: disegnerebbe un flusso e ne racconterebbe un "
            f"altro.\n"
            f"    python3 animazioni/svg/genera.py --misura {NOME}")
    return dato


# --------------------------------------------------------------------------
# La verità: le stesse asserzioni sul dato committato e sull'esperimento
# --------------------------------------------------------------------------
# L'inquadratura, nelle unità dei dati. Non è una scelta di gusto: la nuvola
# arriva a sei unità di larghezza a metà strada, e un riquadro più stretto
# taglierebbe fuori proprio i punti che si sono allontanati di più, cioè il
# fenomeno. `verifica` pretende che ogni punto disegnato ci stia dentro.
XMIN, XMAX = -7.7, 7.7
YMIN, YMAX = -4.4, 4.4


def verifica(dato: dict) -> None:
    """La figura promette tre cose. Ci sono davvero?

    Che a ogni accoppiamento si muova **una sola** coordinata (ed è
    un'uguaglianza esatta, non una tolleranza: quella metà non viene toccata);
    che l'altra invece si muova davvero, o la scena mostrerebbe una nuvola
    ferma; e che alla fine la nuvola sia una gaussiana standard, cioè media
    zero e deviazione uno su tutti e due gli assi.
    """
    stadi, ferma = dato["stadi"], dato["ferma"]
    assert len(stadi) == N_ACCOPPIAMENTI + 1, \
        f"servono {N_ACCOPPIAMENTI + 1} stadi, ce ne sono {len(stadi)}"
    assert len(ferma) == N_ACCOPPIAMENTI
    n = len(stadi[0]["punti"])
    assert n > 40, f"con {n} punti la forma delle lune non si riconosce"
    assert len(dato["luna"]) == n

    # Lo stadio zero sono i dati standardizzati: media zero e deviazione uno.
    for asse, (m, d) in enumerate(zip(stadi[0]["media"], stadi[0]["dev"])):
        assert abs(m) < 0.01 and abs(d - 1) < 0.01, \
            (f"i dati di partenza sono standardizzati, e sull'asse {asse} "
             f"hanno media {m:.3f} e deviazione {d:.3f}")

    for k, j in enumerate(ferma):
        prima, dopo = stadi[k], stadi[k + 1]
        assert len(dopo["punti"]) == n
        # La coordinata ferma: scarto **zero**, non piccolo.
        scarto = max(abs(a[j] - b[j]) for a, b in zip(prima["punti"],
                                                     dopo["punti"]))
        assert scarto == 0.0, \
            (f"l'accoppiamento {k + 1} dovrebbe lasciare intatta la coordinata "
             f"{j}, e invece la sposta di {scarto:.2e}")
        assert prima["dev"][j] == dopo["dev"][j], \
            (f"l'accoppiamento {k + 1} lascia intatta la coordinata {j} ma ne "
             f"cambia la deviazione, da {prima['dev'][j]} a {dopo['dev'][j]}")
        # L'altra si muove, o non ci sarebbe niente da vedere.
        mosso = max(abs(a[1 - j] - b[1 - j]) for a, b in zip(prima["punti"],
                                                            dopo["punti"]))
        assert mosso > 0.05, \
            (f"l'accoppiamento {k + 1} non muove la coordinata {1 - j}: lo "
             f"scarto massimo è {mosso:.2e}, la scena non mostrerebbe niente")

    # Nel mezzo la nuvola si allarga a più del doppio di com'era: è la parte
    # che il testo commenta, e senza questa riga sarebbe un numero letto a
    # occhio sul disegno.
    largo = max(max(s["dev"]) for s in stadi)
    assert largo > 2 * max(stadi[0]["dev"]), \
        (f"la nuvola non si allarga: il massimo delle deviazioni è {largo:.2f} "
         f"contro il {max(stadi[0]['dev']):.2f} di partenza")

    # L'arrivo è una gaussiana standard. Le soglie sono larghe: dicono se la
    # nuvola è quella promessa, non fotografano i numeri di questa macchina.
    for asse, (m, d) in enumerate(zip(stadi[-1]["media"], stadi[-1]["dev"])):
        assert abs(m) < 0.3, \
            f"il latente dovrebbe essere centrato, e sull'asse {asse} vale {m:.2f}"
        assert abs(d - 1) < 0.25, \
            (f"il latente dovrebbe avere deviazione uno, e sull'asse {asse} "
             f"vale {d:.2f}")

    # Nessun punto disegnato esce dall'inquadratura: un punto tagliato via
    # sarebbe proprio quello che si è allontanato di più.
    for k, s in enumerate(stadi):
        for x, y in s["punti"]:
            assert XMIN <= x <= XMAX and YMIN <= y <= YMAX, \
                (f"allo stadio {k} il punto ({x:.2f}, {y:.2f}) esce "
                 f"dall'inquadratura [{XMIN}, {XMAX}] x [{YMIN}, {YMAX}]")


# --------------------------------------------------------------------------
# La figura
# --------------------------------------------------------------------------
# La scala è la stessa sui due assi **per costruzione**: l'altezza del riquadro
# si ricava dalla larghezza e dai due intervalli, invece di essere scritta a
# mano. Senza, una nuvola tonda uscirebbe ovale e lo stiramento, che è il
# soggetto, si leggerebbe sbagliato.
X0, Y0, LARG_R = 39.0, 64.0, 682.0
ALT_R = LARG_R * (YMAX - YMIN) / (XMAX - XMIN)
LARG, ALT = 760, 552
TENUTA = 0.5              # quanta parte di ogni tappa è sosta, quanta è moto

ORDINALI = ["1º", "2º", "3º", "4º", "5º", "6º"]
NOMI_ASSE = ["in orizzontale", "in verticale"]


def dec(v: float, cifre: int = 2) -> str:
    return f"{v:.{cifre}f}".replace(".", ",")


def costruisci() -> Figura:
    dato = dati()
    verifica(dato)
    stadi, ferma, luna = dato["stadi"], dato["ferma"], dato["luna"]
    n_stadi = len(stadi)

    r = Riquadro(x=X0, y=Y0, larg=LARG_R, alt=ALT_R,
                 xmin=XMIN, xmax=XMAX, ymin=YMIN, ymax=YMAX)
    assert abs(r.scala_x - r.scala_y) < 1e-6, \
        f"scale diverse sui due assi: {r.scala_x:.3f} e {r.scala_y:.3f}"
    corpo = [r.cornice()]

    # Il reticolo alle unità pari: senza un riferimento di scala lo stiramento
    # non si vede, perché la nuvola resta comunque una nuvola.
    for v in (-6, -4, -2, 2, 4, 6):
        corpo.append(f'<line class="axc" x1="{r.sx(v):.1f}" y1="{r.y}" '
                     f'x2="{r.sx(v):.1f}" y2="{r.y + r.alt}"/>')
    for v in (-4, -2, 2, 4):
        corpo.append(f'<line class="axc" x1="{r.x}" y1="{r.sy(v):.1f}" '
                     f'x2="{r.x + r.larg}" y2="{r.sy(v):.1f}"/>')
    corpo.append(f'<line class="axz" x1="{r.x}" y1="{r.sy(0):.1f}" '
                 f'x2="{r.x + r.larg}" y2="{r.sy(0):.1f}"/>'
                 f'<line class="axz" x1="{r.sx(0):.1f}" y1="{r.y}" '
                 f'x2="{r.sx(0):.1f}" y2="{r.y + r.alt}"/>')

    anim, moto = [], []

    # I punti. A riposo stanno nelle coordinate del latente, senza nessun
    # `transform`; l'animazione parte dallo scarto inverso, cioè da dov'erano
    # sulle lune, e passa per uno stadio alla volta finendo sull'identità.
    finale = stadi[-1]["punti"]
    for i, (xf, yf) in enumerate(finale):
        px, py = r.sx(xf), r.sy(yf)
        tappe = []
        for k in range(n_stadi):
            inizio, fine_sosta = sosta(k, n_stadi, TENUTA)
            x, y = stadi[k]["punti"][i]
            spo = (f"transform:translate({r.sx(x) - px:.1f}px,"
                   f"{r.sy(y) - py:.1f}px)")
            if k == 0:
                tappe.append((0.0, spo))
            else:
                tappe.append((inizio, spo))
            tappe.append((fine_sosta, spo))
        tappe.append((100.0, tappe[-1][1]))
        anim.append(keyframes(f"p{i}", tappe))
        moto.append(f"    .v{i} {{ animation:p{i} var(--d) linear infinite; }}")
        corpo.append(f'<circle class="pt l{luna[i]} v{i}" cx="{px:.1f}" '
                     f'cy="{py:.1f}" r="4.2"/>')

    # Le due righe che cambiano a ogni stadio: che cosa si muove, e quanto è
    # larga la nuvola. Il numero della coordinata ferma non cambia, ed è la
    # conferma numerica di quello che il disegno fa vedere.
    y1, y2 = r.y + r.alt + 44, r.y + r.alt + 68
    for k in range(n_stadi):
        inizio, _ = sosta(k, n_stadi, TENUTA)
        prossimo = sosta(k + 1, n_stadi, TENUTA)[0] if k + 1 < n_stadi else None
        tappe = [(0.0, "opacity:1" if k == 0 else "opacity:0")]
        if k:
            tappe += [(inizio - 3.0, "opacity:0"), (inizio, "opacity:1")]
        if prossimo is None:
            tappe.append((100.0, "opacity:1"))
        else:
            tappe += [(prossimo - 3.0, "opacity:1"), (prossimo, "opacity:0"),
                      (100.0, "opacity:0")]
        anim.append(keyframes(f"d{k}", tappe))

        if k == 0:
            titolo = "i dati: due lune"
        else:
            titolo = (f"{ORDINALI[k - 1]} accoppiamento: la nuvola si muove "
                      f"solo {NOMI_ASSE[1 - ferma[k - 1]]}")
            if k == n_stadi - 1:
                titolo += ", ed è la gaussiana"
        dev = stadi[k]["dev"]
        base = "" if k == n_stadi - 1 else "opacity:0;"
        corpo.append(
            f'<g style="{base}animation:d{k} var(--d) linear infinite">'
            f'<text class="lbl" x="{r.x}" y="{y1}">{titolo}</text>'
            f'<text class="lbs" x="{r.x}" y="{y2}">quanto è larga la nuvola '
            f'(la deviazione standard): {dec(dev[0])} in orizzontale e '
            f'{dec(dev[1])} in verticale</text>'
            f'</g>')

    corpo += [
        f'<text class="ttl" x="{r.x}" y="{r.y - 34}">'
        f'Sei accoppiamenti, e le lune diventano una gaussiana</text>',
        f'<text class="lbs" x="{r.x}" y="{r.y - 14}">'
        f'gli stessi punti dopo ogni strato del flusso, con il reticolo alle '
        f'unità pari come riferimento</text>']

    return Figura(
        larghezza=LARG, altezza=ALT,
        alt="Una nuvola di punti dentro un riquadro con un reticolo di "
            "riferimento. All'inizio i punti disegnano due archi intrecciati, "
            "le due lune, uno in un colore e uno nell'altro. A ogni passo "
            "l'intera nuvola si deforma, ma si sposta lungo una sola "
            "direzione per volta: prima solo in verticale, poi solo in "
            "orizzontale, e così alternando per sei passi. A metà strada la "
            "nuvola si allarga fino a occupare quasi tutto il riquadro, poi "
            "si richiude. Alla fine gli archi non ci sono più e i punti "
            "formano una macchia tonda centrata sull'origine, con i due "
            "colori mescolati. Due righe di testo sotto il riquadro dicono, "
            "a ogni passo, quale delle due direzioni si sta muovendo e "
            "quanto è larga la nuvola nei due sensi.",
        corpo="".join(corpo),
        stile=f"""    .pt  {{ opacity:0.8; }}
    .l0  {{ fill:{TEAL}; }}
    .l1  {{ fill:{TERRACOTTA}; }}
    .axz {{ stroke:{BORDER_STRONG}; stroke-width:1.2; }}
""" + "\n".join(moto),
        animazioni=anim,
        durata=17.0,
        fermi="g, circle",
    )
