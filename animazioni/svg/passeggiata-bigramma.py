"""La passeggiata del bigramma: si pesca dal sacchetto, e si gira in tondo.

La sezione racconta la generazione come una fila di sorteggi truccati («un
sacchetto con dentro quattro foglietti, tre con scritto gatto e uno con scritto
cane») e poi mostra che con il seme 2 il testo inciampa nell'anello «il cane
guarda il cane guarda…». Sono due cose che una figura ferma non fa vedere: il
pescare, e il tornare al punto di partenza.

Qui non c'è niente di trascritto. I conteggi si contano sul corpus di tre frasi
della sezione, e la passeggiata è quella del blocco di codice, rifatta con lo
stesso `random.Random(2)` e la stessa `rng.choices`: l'assert pretende che
esca parola per parola la frase che la sezione stampa. Se un giorno il corpus,
il seme o la frase cambiassero, la figura non si genererebbe nemmeno.

Lo stato di riposo è la frase finita con l'ultima pagina del quaderno aperta,
cioè quella in cui la scommessa è fra «salta» e la fine della frase.
"""

import random
from collections import Counter, defaultdict

from paithon_svg import *

NOME = "passeggiata-bigramma"
TITOLO = "la passeggiata del bigramma, e l'anello in cui inciampa"

CORPUS = [
    "il gatto nero salta sul muro",
    "il gatto bianco dorme sul divano",
    "il cane guarda il gatto nero",
]
INIZIO, FINE = "<s>", "</s>"
SEME = 2                       # il seme che nella sezione inciampa nell'anello
ATTESA = "il cane guarda il cane guarda il gatto nero"

AVANZO = 11.9                  # larghezza media di un carattere a 20px, con
SPAZIO = 14                    # abbondanza: qui non si misura il testo davvero


def quaderno():
    conta = defaultdict(Counter)
    for frase in CORPUS:
        parole = [INIZIO] + frase.split() + [FINE]
        for w1, w2 in zip(parole, parole[1:]):
            conta[w1][w2] += 1
    return conta


def passeggiata(conta):
    """La stessa di `genera(2)` nella sezione: contesto, pagina, parola pescata."""
    rng = random.Random(SEME)
    parola, passi, frase = INIZIO, [], []
    while len(frase) < 20:
        seguiti = conta[parola]
        pescata = rng.choices(list(seguiti), weights=seguiti.values())[0]
        passi.append((parola, pescata))
        if pescata == FINE:
            break
        frase.append(pescata)
        parola = pescata
    return passi, " ".join(frase)


def etichetta(w):
    return "fine frase" if w == FINE else ("inizio" if w == INIZIO else w)


def costruisci() -> Figura:
    conta = quaderno()
    passi, frase = passeggiata(conta)
    assert frase == ATTESA, f"la passeggiata non è quella della sezione: {frase}"
    assert conta["il"]["gatto"] == 3 and conta["il"]["cane"] == 1, \
        "i foglietti di «il» non sono tre a uno come dice la sezione"
    anelli = [i for i, (c, _) in enumerate(passi) if c == "guarda"]
    assert len(anelli) == 2, "l'anello «guarda → il» dovrebbe chiudersi due volte"

    corpo, anim = [], []

    # ---------------- la frase che cresce
    parole = [p for _, p in passi if p != FINE]
    larghezze = [len(w) * AVANZO for w in parole]
    totale = sum(larghezze) + SPAZIO * (len(parole) - 1)
    x = 350 - totale / 2
    posizioni = []
    for w, lar in zip(parole, larghezze):
        posizioni.append((x, lar))
        x += lar + SPAZIO

    for i, (w, (px, lar)) in enumerate(zip(parole, posizioni)):
        t0, _ = sosta(i, len(passi), tenuta=0.0)
        anim.append(keyframes(f"w{i}", [(0.0, "opacity:0"),
                                        (max(t0 - 0.5, 0.01), "opacity:0"),
                                        (t0, "opacity:1"),
                                        (100.0, "opacity:1")]))
        # riposo = frase finita: l'attributo dice 1, e l'animazione parte da 0
        corpo.append(f'<text class="par" x="{px + lar / 2:.1f}" y="66" '
                     f'text-anchor="middle" '
                     f'style="animation:w{i} var(--d) infinite">{w}</text>')

    # l'anello: due archi sotto la frase, da «guarda» al «il» che segue. Ognuno
    # compare quando il giro si chiude, cioè quando quel «il» viene scritto.
    for k, i in enumerate(anelli):
        da = posizioni[i - 1]        # «guarda» appena scritto
        a = posizioni[i]             # il «il» che ne esce
        x0, x1 = da[0] + da[1] / 2, a[0] + a[1] / 2
        t0, _ = sosta(i, len(passi), tenuta=0.0)
        anim.append(keyframes(f"a{k}", [(0.0, "opacity:0"),
                                        (max(t0 - 0.5, 0.01), "opacity:0"),
                                        (t0, "opacity:1"),
                                        (100.0, "opacity:1")]))
        corpo.append(f'<path class="anello" d="M{x0:.1f},78 Q{(x0 + x1) / 2:.1f},'
                     f'{100 + 8 * k},{x1:.1f},78" '
                     f'style="animation:a{k} var(--d) infinite"/>')

    # ---------------- la pagina del quaderno, una per passo
    RIQ = Riquadro(x=190, y=152, larg=320, alt=176, xmin=0, xmax=1, ymin=0, ymax=1)
    corpo.append(RIQ.cornice())

    for i, (contesto, pescata) in enumerate(passi):
        voci = sorted(conta[contesto].items(), key=lambda kv: (-kv[1], kv[0]))
        tot = sum(c for _, c in voci)
        t0, t1 = sosta(i, len(passi), tenuta=0.62)
        tappe = [(0.0, "opacity:0")]
        if t0 > 0.6:
            tappe.append((max(t0 - 0.6, 0.01), "opacity:0"))
        tappe += [(t0, "opacity:1"), (t1, "opacity:1")]
        if i < len(passi) - 1:
            tappe += [(min(t1 + 0.6, 99.9), "opacity:0"), (100.0, "opacity:0")]
        else:
            tappe.append((100.0, "opacity:1"))
        anim.append(keyframes(f"p{i}", tappe))
        op = 1 if i == len(passi) - 1 else 0
        stile = f'opacity="{op}" style="animation:p{i} var(--d) infinite"'

        dentro = [f'<text class="lbl" x="{RIQ.x + 18}" y="{RIQ.y + 30}">'
                  f'la pagina di «{etichetta(contesto)}»</text>']
        for k, (w, c) in enumerate(voci):
            y = RIQ.y + 62 + k * 40
            largh = 128 * c / tot
            classe = "vinta" if w == pescata else "persa"
            dentro.append(
                f'<rect class="{classe}" x="{RIQ.x + 128}" y="{y - 14}" '
                f'width="{largh:.1f}" height="20" rx="3"/>'
                f'<text class="lbs" x="{RIQ.x + 118}" y="{y + 2}" '
                f'text-anchor="end">{etichetta(w)}</text>'
                f'<text class="lbs" x="{RIQ.x + 134 + largh:.1f}" y="{y + 2}">'
                f'{c} su {tot}</text>')
        corpo.append(f'<g {stile}>' + "".join(dentro) + '</g>')

    corpo += [
        f'<text class="lbs" x="350" y="30" text-anchor="middle">'
        f'la frase che esce, parola per parola</text>',
        f'<text class="lbs" x="350" y="{RIQ.y + RIQ.alt + 28}" '
        f'text-anchor="middle">i foglietti nel sacchetto, e quello pescato</text>',
    ]

    return Figura(
        larghezza=700, altezza=360,
        alt="In alto una frase si scrive parola per parola, «il cane guarda il "
            "cane guarda il gatto nero», e due archetti sotto di essa collegano "
            "ciascun «guarda» al «il» che lo segue, cioè le due volte in cui la "
            "passeggiata torna sui propri passi. Sotto, un riquadro mostra a "
            "ogni passo la pagina del quaderno intestata all'ultima parola "
            "scritta, con le continuazioni viste nel corpus disegnate come "
            "barrette lunghe quanto i loro foglietti: dopo «il» ci sono «gatto» "
            "tre su quattro e «cane» uno su quattro, dopo «cane» soltanto "
            "«guarda», e alla fine, dopo «nero», la scommessa è pari fra "
            "«salta» e la fine della frase. La barretta della parola pescata è "
            "in terracotta, le altre smorzate.",
        corpo="".join(corpo),
        stile=f"""    .par    {{ font-family:{SANS}; font-size:20px; fill:{INK}; }}
    .anello {{ fill:none; stroke:{TERRACOTTA}; stroke-width:1.8; }}
    .vinta  {{ fill:{TERRACOTTA}; }}
    .persa  {{ fill:{BORDER_STRONG}; }}""",
        animazioni=anim,
        durata=len(passi) * 1.15,
        fermi=".par, .anello, g",
    )
