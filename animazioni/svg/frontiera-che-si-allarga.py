"""Dove guarda A*, con e senza il fiuto, sullo stesso rompicapo.

Il tempo qui è il contenuto due volte. La prima: una ricerca **si allarga**, e
il modo in cui si allarga (in tondo contro in avanti) è tutta la sezione. La
seconda: il prezzo si paga a ogni passo, quindi il contatore che sale è
l'argomento, non un ornamento.

Il piano è lo stesso nei due riquadri: in ascissa i passi già fatti, in
ordinata la **stima** di quanto la posizione disti ancora dalla meta, contata a
isolati. In ordinata c'e' la stima e non la distanza vera, ed e' la differenza
che rende la figura vera: la fascia stretta del pannello destro dice che con la
stima l'algoritmo non apre mai uno stato per cui «passi fatti piu' stima»
superi il costo della soluzione, che e' la proprieta' di A*. Con la distanza
vera al posto della stima il disegno direbbe il contrario (il 92,9% degli stati
finirebbe sopra la diagonale). Cambia una cosa sola fra i due pannelli, ed e'
se l'algoritmo quel numero lo guarda o no.

Tutti i numeri li calcola la scena, rieseguendo la ricerca del capitolo: le
celle toccate, l'ordine in cui lo sono state, e i due totali (48.389 e 282) che
il codice della sezione stampa.
"""

import heapq

from paithon_svg import *

NOME = "frontiera-che-si-allarga"
TITOLO = "Dove guarda la ricerca, con e senza il fiuto"

META = (1, 2, 3, 4, 5, 6, 7, 8, 0)
PARTENZA = (7, 2, 4, 5, 0, 6, 8, 3, 1)     # la stessa della sezione

TAPPE = 24                                  # in quanti scatti si scopre la ricerca
SPIE = 6                                    # quante volte si aggiorna il contatore


# --------------------------------------------------------------------------
# La ricerca, rieseguita qui: da qui escono tutti i numeri della figura
# --------------------------------------------------------------------------
def mosse(s):
    v = s.index(0)
    r, c = divmod(v, 3)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            n = nr * 3 + nc
            t = list(s)
            t[v], t[n] = t[n], t[v]
            yield tuple(t)


def isolati(s) -> int:
    """Quanti passi in orizzontale e in verticale mancano, tessera per tessera."""
    return sum(abs(i // 3 - META.index(v) // 3) + abs(i % 3 - META.index(v) % 3)
               for i, v in enumerate(s) if v)


def cerca(stima):
    """A* con la stima data. Restituisce la lunghezza della soluzione e la
    successione degli stati aperti, ciascuno come (passi fatti, quanto manca)."""
    coda = [(stima(PARTENZA), 0, PARTENZA)]
    costo = {PARTENZA: 0}
    aperti = []
    while coda:
        _, fatti, s = heapq.heappop(coda)
        if s == META:
            return fatti, aperti
        if fatti > costo[s]:
            continue
        aperti.append((fatti, isolati(s)))
        for t in mosse(s):
            if fatti + 1 < costo.get(t, 10 ** 9):
                costo[t] = fatti + 1
                heapq.heappush(coda, (fatti + 1 + stima(t), fatti + 1, t))
    raise AssertionError("nessuna soluzione")


def prima_volta(aperti) -> dict:
    """Per ogni cella del piano, a che punto della ricerca è stata toccata."""
    quando = {}
    for k, cella in enumerate(aperti):
        quando.setdefault(cella, k / max(len(aperti) - 1, 1))
    return quando


def verifica(costo, senza, con) -> None:
    """La figura promette due cose: che il fiuto restringa, e che restringa
    **alla diagonale**. Sono due affermazioni diverse e vanno provate qui."""
    assert costo == 20, f"la sezione dice venti mosse, la ricerca ne dà {costo}"
    assert len(senza) == 48389 and len(con) == 282, \
        (f"i totali non sono quelli stampati dalla sezione: "
         f"{len(senza)} e {len(con)}")
    sopra = [c for c in set(con) if sum(c) > costo]
    assert not sopra, \
        f"con la stima la ricerca esce dalla diagonale in {len(sopra)} celle"
    fuori = [c for c in set(senza) if sum(c) > costo]
    assert len(fuori) > len(set(senza)) / 2, \
        (f"l'`:alt:` promette che senza stima la ricerca finisca *in "
         f"maggioranza* sopra la diagonale: sono {len(fuori)} celle su "
         f"{len(set(senza))}")


# --------------------------------------------------------------------------
# La figura
# --------------------------------------------------------------------------
def pannello(r, quando, colore, sigla, costo) -> tuple[list, list]:
    corpo, anim = [r.cornice()], []
    larg = r.scala_x * 0.86
    alt = r.scala_y * 0.86
    # le celle si scoprono a scatti invece che una per una: 179 animazioni
    # separate peserebbero cinque volte tanto e nessuno vedrebbe la differenza
    scatti = {}
    for (g, h), t in quando.items():
        scatti.setdefault(min(int(t * TAPPE), TAPPE - 1), []).append((g, h))
    for i in sorted(scatti):
        celle = "".join(
            f'<rect class="cel" x="{r.sx(g) - larg / 2:.1f}" '
            f'y="{r.sy(h) - alt / 2:.1f}" width="{larg:.1f}" height="{alt:.1f}" '
            f'fill="{colore}"/>' for g, h in scatti[i])
        istante = 6.0 + 82.0 * i / TAPPE
        anim.append(keyframes(f"sc{sigla}{i}", [
            (0.0, "opacity:0"),
            (max(istante - 1.0, 0.0), "opacity:0"),
            (istante, "opacity:1"),
            (100.0, "opacity:1")]))
        corpo.append(f'<g style="animation:sc{sigla}{i} var(--d) linear infinite">'
                     f'{celle}</g>')

    # la diagonale del costo, in ultimo: sotto c'e' lavoro inevitabile, sopra
    # lavoro sprecato, ed e' la riga che da' senso alle due macchie. Disegnata
    # per prima spariva sotto le celle.
    corpo.append(f'<line class="dia" x1="{r.sx(0):.1f}" y1="{r.sy(costo):.1f}" '
                 f'x2="{r.sx(costo):.1f}" y2="{r.sy(0):.1f}"/>')
    return corpo, anim


def contatore(x, y, aperti, sigla) -> tuple[list, list]:
    """Il conto degli stati aperti, aggiornato a scatti fino al totale vero."""
    corpo, anim = [], []
    for k in range(SPIE + 1):
        frazione = k / SPIE
        quanti = len(aperti) if k == SPIE else int(len(aperti) * frazione)
        istante = 6.0 + 82.0 * frazione
        fine = 6.0 + 82.0 * (k + 1) / SPIE if k < SPIE else 100.0
        anim.append(keyframes(f"ct{sigla}{k}", [
            (0.0, "opacity:0"),
            (max(istante - 0.5, 0.0), "opacity:0"),
            (istante, "opacity:1"),
            (min(fine, 100.0), "opacity:1" if k == SPIE else "opacity:1"),
            (min(fine + 0.01, 100.0), "opacity:1" if k == SPIE else "opacity:0"),
            (100.0, "opacity:1" if k == SPIE else "opacity:0")]))
        corpo.append(
            f'<g style="{"" if k == SPIE else "opacity:0;"}'
            f'animation:ct{sigla}{k} var(--d) linear infinite">'
            f'<text class="cnt" x="{x}" y="{y}">{mille(quanti)}</text></g>')
    return corpo, anim


def mille(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def costruisci() -> Figura:
    costo, senza = cerca(lambda s: 0)
    _, con = cerca(isolati)
    verifica(costo, senza, con)

    gmax = max(g for g, _ in senza + con)
    hmax = max(h for _, h in senza + con)

    corpo, anim = [], []
    for x0, aperti, colore, sigla, titolo, sotto in (
            (64, senza, TERRACOTTA, "s", "senza stima",
             "l’algoritmo non sa da che parte guardare"),
            (414, con, TEAL, "c", "con la distanza a isolati",
             "l’algoritmo guarda quanto manca")):
        r = Riquadro(x=x0, y=74, larg=282, alt=228,
                     xmin=-0.8, xmax=gmax + 0.8, ymin=-0.8, ymax=hmax + 0.8)
        c, a = pannello(r, prima_volta(aperti), colore, sigla, costo)
        corpo += c
        anim += a
        corpo += [
            f'<text class="ttl" x="{r.x}" y="{r.y - 26}">{titolo}</text>',
            f'<text class="lbs" x="{r.x}" y="{r.y - 8}">{sotto}</text>',
            f'<text class="lbs" x="{r.x}" y="{r.y + r.alt + 20}">'
            f'posizioni aperte:</text>']
        c, a = contatore(r.x + 118, r.y + r.alt + 20, aperti, sigla)
        corpo += c
        anim += a

    corpo += [
        f'<text class="lbs" x="64" y="{74 + 228 + 48}">'
        f'In orizzontale i passi già fatti, in verticale quanti se ne stima '
        f'che ne manchino.</text>',
        f'<text class="lbs" x="64" y="{74 + 228 + 66}">'
        f'La riga obliqua è la soluzione, {costo} mosse: sotto c’è lavoro che '
        f'nessuno può evitare, sopra lavoro sprecato.</text>']

    return Figura(
        larghezza=760, altezza=430,
        alt="Due riquadri affiancati con lo stesso piano: in orizzontale i passi "
            "già fatti dalla partenza, in verticale la stima di quanto la "
            "posizione disti ancora dalla meta, contata a isolati. In ciascuno "
            "una riga obliqua segna le venti mosse "
            "della soluzione. Le posizioni che la ricerca apre si accendono a "
            "poco a poco. A sinistra, «senza stima», si accendono dappertutto, e "
            "la maggior parte finisce sopra la riga obliqua, cioè in posizioni per cui "
            "i passi fatti più quelli stimati superano già il costo della "
            "soluzione; il contatore sotto arriva a 48.389. "
            "A destra, «con la distanza a isolati», si accendono soltanto lungo "
            "una fascia stretta che segue la riga obliqua e non la supera mai, e "
            "il contatore si ferma a 282.",
        corpo="".join(corpo),
        stile=f"""    .cel {{ stroke:none; }}
    .dia {{ stroke:{OCRA}; stroke-width:2.5; }}
    .cnt {{ font-family:{SANS}; font-size:14px; font-weight:600; fill:{INK}; }}""",
        animazioni=anim,
        durata=10.0,
        fermi="g",
    )
