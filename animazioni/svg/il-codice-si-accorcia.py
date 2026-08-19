"""Il prezzo di imparare, pagato mentre si legge.

I dati sono quelli di `AutoSupervisione/capire-e-accorciare.md`, rifatti qui
con la stessa tabella e lo stesso seme invece che trascritti: se il capitolo
cambia i numeri, `verifica` se ne accorge e la figura non si genera.

Quello che il fermo immagine non puo' mostrare, ed e' tutta la ragione della
clip, e' che il costo di un modello **scende leggendo**. Il modello non viene
consegnato a parte: nasce ignorante, paga caro le prime lettere, e con quelle
si costruisce. La curva che scende E' il prezzo del dizionario, pagato a rate.

Le tre curve dicono tre cose diverse. Ordine 0 resta in alto per sempre: sta
contando una cosa che in questa lingua non porta informazione. Ordine 1 ha la
forma della regola e scende fino a sfiorare il fondo. Ordine 2 ha piu' memoria
del necessario, ci mette molto piu' tempo a riempire sedici contesti invece di
quattro, e finisce un filo sopra: e' il rasoio di Occam disegnato.

Lo stato di riposo sono le tre curve intere con il fondo tratteggiato: chi non
anima (stampa, PDF, prefers-reduced-motion) vede la conclusione.
"""

import math
from collections import defaultdict
from random import Random

from paithon_svg import *

NOME = "il-codice-si-accorcia"
TITOLO = "Il prezzo di imparare, pagato mentre si legge"

# --- la stessa sorgente del capitolo, rifatta qui invece che trascritta -----
REGOLA = {
    "a": {"a": 0.05, "e": 0.05, "r": 0.35, "t": 0.55},
    "e": {"a": 0.05, "e": 0.05, "r": 0.55, "t": 0.35},
    "r": {"a": 0.35, "e": 0.55, "r": 0.05, "t": 0.05},
    "t": {"a": 0.55, "e": 0.35, "r": 0.05, "t": 0.05},
}
LETTERE = sorted(REGOLA)
N = 200_000

# Quello che il capitolo stampa: se cambia, la figura non si genera.
ATTESI = {"fondo": 1.4367, 0: 2.0001, 1: 1.4402, 2: 1.4411}

ASPETTO = {0: (OCRA, "ordine 0"), 1: (TEAL, "ordine 1"), 2: (TERRACOTTA, "ordine 2")}


def genera(n, seme=0):
    r, seq = Random(seme), ["a"]
    for _ in range(n - 1):
        p, soglia, cumulata = REGOLA[seq[-1]], r.random(), 0.0
        for lettera in LETTERE:
            cumulata += p[lettera]
            if soglia < cumulata:
                seq.append(lettera)
                break
    return "".join(seq)


def entropia(p):
    return -sum(q * math.log2(q) for q in p if q > 0)


def it(x, cifre):
    """Un numero con la virgola, come lo scrive il libro.

    Sta qui in una funzione e non in una catena di `.replace` sulla frase
    intera: il punto va cambiato nel NUMERO, non nel testo, se no si mangia
    anche quello di fine periodo. E' gia' successo altrove, e l'alt che ne
    usciva erano frammenti al posto di una frase.
    """
    return f"{x:.{cifre}f}".replace(".", ",")


def mille(n):
    """Un intero col punto a separare le migliaia, come in italiano."""
    return f"{n:,}".replace(",", ".")


def corsa(testo, ordine):
    """Il costo medio per lettera accumulato dall'inizio, lettera per lettera."""
    conte = defaultdict(lambda: dict.fromkeys(LETTERE, 1))
    totale = defaultdict(lambda: len(LETTERE))
    bit, fuori = 0.0, []
    for i, lettera in enumerate(testo):
        contesto = testo[max(0, i - ordine):i]
        bit -= math.log2(conte[contesto][lettera] / totale[contesto])
        conte[contesto][lettera] += 1
        totale[contesto] += 1
        fuori.append(bit / (i + 1))
    return fuori


def verifica(curve, fondo):
    """La figura dice quello che la didascalia promette?"""
    assert abs(fondo - ATTESI["fondo"]) < 5e-5, \
        f"il fondo qui e' {fondo:.4f}, il capitolo stampa {ATTESI['fondo']}"
    for k in (0, 1, 2):
        avuto = curve[k][-1]
        assert abs(avuto - ATTESI[k]) < 5e-5, \
            f"ordine {k}: la figura calcola {avuto:.4f}, il capitolo stampa {ATTESI[k]}"
    # l'alt promette che tutte partano da 2 bit, cioe' da «non so niente»
    for k in (0, 1, 2):
        assert abs(curve[k][0] - 2.0) < 1e-9, \
            f"ordine {k} parte da {curve[k][0]:.4f}, non da 2"
    # l'alt promette che ordine 0 resti PIATTO: mai sotto 1,99
    assert min(curve[0]) > 1.99, f"ordine 0 scende fino a {min(curve[0]):.4f}"
    # e che ordine 2 stia SEMPRE sopra ordine 1 dopo l'avvio, e finisca sopra
    assert curve[2][-1] > curve[1][-1], "ordine 2 dovrebbe finire sopra ordine 1"
    assert all(b >= a - 1e-12 for a, b in zip(curve[1][50:], curve[2][50:])), \
        "ordine 2 dovrebbe stare sempre sopra ordine 1"
    # L'alt dice che ordine 2 «ci mette circa il doppio». Non e' un'impressione:
    # e' quante lettere serve leggere per arrivare a 0,05 bit dal fondo. La
    # prima stesura diceva «molto piu' lentamente» e si appoggiava al divario a
    # diecimila lettere, che vale 0,0100 e non regge quell'avverbio: il divario
    # e' grande all'inizio e si chiude, quindi misurato tardi dice poco.
    def quando(k, soglia=0.05):
        return next(i for i, v in enumerate(curve[k]) if v - fondo < soglia) + 1

    lento, svelto = quando(2), quando(1)
    assert lento / svelto > 1.7, \
        f"ordine 2 arriva in {lento} lettere contro {svelto}: non e' «circa il doppio»"
    # nessuna curva finisce sotto il fondo. Non e' un teorema (il fondo di
    # Shannon limita la media sulla sorgente, non il costo su UNA stringa
    # sorteggiata): e' che a duecentomila lettere lo scarto tipico e' di due
    # millesimi, quindi finirci sotto vorrebbe dire un errore nel conto.
    for k in (0, 1, 2):
        assert curve[k][-1] > fondo, f"ordine {k} finisce sotto il fondo"
    return lento, svelto


def costruisci() -> Figura:
    testo = genera(N)
    curve = {k: corsa(testo, k) for k in (0, 1, 2)}
    fondo = sum(entropia(list(REGOLA[s].values())) for s in LETTERE) / len(LETTERE)
    lento, svelto = verifica(curve, fondo)

    # Ascissa logaritmica: la discesa succede tutta nelle prime migliaia di
    # lettere, e in scala lineare sarebbe schiacciata contro il margine
    # sinistro, cioe' sparirebbe proprio la cosa da guardare.
    # ymax sta sopra il picco della curva di ordine 0, che nelle prime decine di
    # lettere sale a 2,14: con un tetto piu' basso usciva dal riquadro e finiva
    # a coprire il sottotitolo. Il provino lo ha mostrato subito, il sorgente no.
    tetto = max(max(c) for c in curve.values())
    r = Riquadro(x=92, y=76, larg=548, alt=292,
                 xmin=1.0, xmax=math.log10(N), ymin=1.38, ymax=tetto + 0.08)
    corpo, anim = [r.cornice()], []

    corpo.append(f'<text class="ttl" x="{r.x}" y="{r.y - 40}">'
                 f'quanto costa scrivere il testo, mentre lo si legge</text>')
    corpo.append(f'<text class="lbs" x="{r.x}" y="{r.y - 20}">'
                 f'bit per lettera, media dall&#8217;inizio</text>')

    # tacche verticali: potenze di dieci
    for e in range(1, int(math.log10(N)) + 1):
        px = r.sx(e)
        corpo.append(f'<line class="axc" x1="{px:.1f}" y1="{r.y}" '
                     f'x2="{px:.1f}" y2="{r.y + r.alt}"/>')
        etichetta = f"10{'⁰¹²³⁴⁵⁶⁷⁸⁹'[e]}"
        corpo.append(f'<text class="lbs" x="{px:.1f}" y="{r.y + r.alt + 20}" '
                     f'text-anchor="middle">{etichetta}</text>')
    corpo.append(f'<text class="lbs" x="{r.x + r.larg}" y="{r.y + r.alt + 40}" '
                 f'text-anchor="end">lettere lette</text>')

    # tacche orizzontali
    for v in (1.5, 1.75, 2.0):
        py = r.sy(v)
        corpo.append(f'<line class="axc" x1="{r.x}" y1="{py:.1f}" '
                     f'x2="{r.x + r.larg}" y2="{py:.1f}"/>')
        corpo.append(f'<text class="lbs" x="{r.x - 10}" y="{py + 4:.1f}" '
                     f'text-anchor="end">{it(v, 2)}</text>')

    # il fondo: la riga che nessuno puo' oltrepassare
    py = r.sy(fondo)
    corpo.append(f'<line class="fondo" x1="{r.x}" y1="{py:.1f}" '
                 f'x2="{r.x + r.larg}" y2="{py:.1f}"/>')
    corpo.append(f'<text class="lbf" x="{r.x + 8}" y="{py + 18:.1f}">'
                 f'il fondo della sorgente, {it(fondo, 4)} bit</text>')

    # le tre curve, campionate su una griglia logaritmica
    campioni = [10 ** (1 + i * (math.log10(N) - 1) / 260) for i in range(261)]
    for k in (2, 1, 0):            # ordine 0 disegnato per ultimo, sta sopra
        colore, _ = ASPETTO[k]
        punti = " ".join(f"{r.sx(math.log10(c)):.1f},{r.sy(curve[k][int(c) - 1]):.1f}"
                         for c in campioni)
        corpo.append(f'<polyline class="curva" points="{punti}" stroke="{colore}" '
                     f'pathLength="1000" style="animation:tira{k} var(--d) infinite"/>')
        anim.append(keyframes(f"tira{k}", [(0.0, "stroke-dashoffset:1000"),
                                           (78.0, "stroke-dashoffset:0"),
                                           (100.0, "stroke-dashoffset:0")]))

    # Le etichette a fine curva. Ordine 1 e ordine 2 finiscono a nove
    # decimillesimi di bit l'una dall'altra, cioe' a un terzo di pixel: messe
    # alla loro altezza vera si sovrappongono e si leggono «ordine 21». Si
    # scostano quel tanto che basta, e un trattino sottile dice a quale curva
    # ciascuna appartiene, perche' l'altezza da sola non lo dice piu'.
    posti, occupato = {}, []
    for k in sorted((0, 1, 2), key=lambda j: r.sy(curve[j][-1])):
        y = r.sy(curve[k][-1])
        while any(abs(y - v) < 17 for v in occupato):
            y += 17
        occupato.append(y)
        posti[k] = y

    for k in (0, 1, 2):
        colore, nome = ASPETTO[k]
        y_curva, y_etichetta = r.sy(curve[k][-1]), posti[k]
        if abs(y_curva - y_etichetta) > 2:
            corpo.append(f'<line class="guida" x1="{r.x + r.larg + 2}" '
                         f'y1="{y_curva:.1f}" x2="{r.x + r.larg + 12}" '
                         f'y2="{y_etichetta - 4:.1f}" stroke="{colore}" '
                         f'style="animation:vieni{k} var(--d) infinite"/>')
        # Il colore va nello `style` e non in un attributo `fill`: nel foglio
        # del brand `.lbl` fissa gia' un fill, e una regola di classe batte
        # sempre un attributo di presentazione. Scritto come attributo, le tre
        # etichette uscivano tutte nere, cioe' la figura perdeva la sua unica
        # legenda: quale curva sia quale lo dice solo il colore.
        corpo.append(f'<text class="lbl" x="{r.x + r.larg + 16}" '
                     f'y="{y_etichetta:.1f}" '
                     f'style="fill:{colore};animation:vieni{k} '
                     f'var(--d) infinite">{nome}</text>')
        anim.append(keyframes(f"vieni{k}", [(0.0, "opacity:0"), (74.0, "opacity:0"),
                                            (84.0, "opacity:1"), (100.0, "opacity:1")]))

    stile = (f'    .curva {{ fill:none; stroke-width:2.6; stroke-linejoin:round; '
             f'stroke-dasharray:1000; }}\n'
             f'    .guida {{ stroke-width:1.2; opacity:0.7; }}\n'
             f'    .fondo {{ stroke:{FG_MUTED}; stroke-width:1.6; '
             f'stroke-dasharray:7 5; }}\n'
             f'    .lbf   {{ font-family:{SANS}; font-size:12px; fill:{FG_MUTED}; }}')

    # Ogni numero e' gia' formattato quando entra nella frase: sulla frase
    # intera non gira nessun `.replace`, che e' il modo tipico di mangiarsi le
    # virgole di un periodo e consegnare tre frammenti a chi usa un lettore di
    # schermo. Il commento di `it` dice perche' questa riga sta cosi'.
    alt = (
        "Tre curve mostrano quanti bit per lettera costa scrivere il testo man "
        "mano che le lettere scorrono, con le lettere lette in scala "
        "logaritmica. Tutte e tre partono da 2 bit, cioè da nessuna conoscenza. "
        "Il modello di ordine 0 resta piatto a 2 bit e non impara niente. Il "
        "modello di ordine 1 scende rapidamente verso la linea tratteggiata del "
        f"fondo teorico, {it(fondo, 4)} bit, e si ferma a {it(curve[1][-1], 4)}. "
        "Il modello di ordine 2 scende più lentamente, perché ha sedici "
        "contesti da riempire invece di quattro: per portarsi a 0,05 bit dal "
        f"fondo gli servono {mille(lento)} lettere contro {mille(svelto)}, cioè "
        f"circa il doppio, e si assesta poco sopra, a {it(curve[2][-1], 4)}."
    )

    return Figura(larghezza=744, altezza=438, alt=alt,
                  corpo="\n  ".join(corpo), stile=stile,
                  animazioni=anim, durata=9.0,
                  fermi=".curva, .guida, text")
