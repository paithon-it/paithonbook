"""Il ring all-reduce: due giri attorno al tavolo, e nessuno fa da imbuto.

Il tempo qui è il contenuto: la pagina descrive in prosa *due* giri e la
differenza fra loro, ed è la cosa che a parole non si vede. Nel primo giro
(reduce-scatter) ogni scheda passa al vicino di destra un pezzo per volta e
il vicino lo somma al proprio; dopo $K-1$ passaggi ciascuna tiene **un** pezzo
completo, e nessuna tiene il totale. Nel secondo (all-gather) quei pezzi finiti
girano com'erano, e dopo altri $K-1$ passaggi ce li hanno tutte.

L'anello **gira davvero**: la somma è simulata pezzo per pezzo, e sono le
guardie a dire se il disegno è vero. Sono tre, e ognuna difende una frase
della didascalia:

- dopo il primo giro ogni scheda ha esattamente un pezzo completo, e i quattro
  pezzi completi sono tutti diversi (se due schede tenessero lo stesso pezzo,
  il secondo giro non basterebbe);
- alla fine tutte le celle valgono $K$, cioè contengono il contributo di tutte;
- ogni scheda spedisce $2(K-1)$ pezzi su $K$, che con $K=4$ fa **sei quarti**,
  una volta e mezza la propria lista: è il numero che la pagina calcola a mano
  poche righe più sopra, e non va scritto a mano nemmeno qui.

$K = 4$ e non 8 perché quattro è la taglia su cui la pagina fa il conto («In
quattro, ciascuno taglia la propria lista in quattro pezzetti e nei due giri
ne spedisce tre più tre»), e perché con otto nodi le celle diventano
illeggibili.
"""

import math

from paithon_svg import *

NOME = "anello-somma"
TITOLO = "il ring all-reduce, due giri attorno al tavolo"

K = 4                    # schede, e quindi anche pezzi per scheda
STATI = 2 * (K - 1) + 2  # riposo iniziale, i sei passaggi, riposo finale


def simula():
    """Il ring all-reduce, eseguito.

    `n[i][c]` è quanti contributi ha dentro il pezzo `c` della scheda `i`.
    Torna la successione degli stati e quanti pezzi ha spedito ogni scheda.
    """
    n = [[1] * K for _ in range(K)]
    stati = [[riga[:] for riga in n]]
    passaggi = []            # (scheda che spedisce, pezzo, scheda che riceve)
    spediti = [0] * K

    # primo giro: reduce-scatter. Al passo s la scheda i manda il pezzo
    # (i - s) mod K al vicino, che lo somma al proprio.
    for s in range(K - 1):
        mosse = [(i, (i - s) % K, (i + 1) % K) for i in range(K)]
        for mitt, c, dest in mosse:
            n[dest][c] += n[mitt][c]
            spediti[mitt] += 1
        passaggi.append(mosse)
        stati.append([riga[:] for riga in n])

    completi = [[c for c in range(K) if n[i][c] == K] for i in range(K)]
    assert all(len(v) == 1 for v in completi), \
        f"dopo il primo giro ogni scheda deve avere un pezzo completo: {completi}"
    assert len({v[0] for v in completi}) == K, \
        f"i pezzi completi devono essere tutti diversi: {completi}"

    # secondo giro: all-gather. Il pezzo finito passa al vicino, che lo prende
    # com'è (non lo somma: è già il totale).
    posseduto = [v[0] for v in completi]
    for s in range(K - 1):
        mosse = [(i, (posseduto[i] - s) % K, (i + 1) % K) for i in range(K)]
        for mitt, c, dest in mosse:
            n[dest][c] = K
            spediti[mitt] += 1
        passaggi.append(mosse)
        stati.append([riga[:] for riga in n])

    assert all(n[i][c] == K for i in range(K) for c in range(K)), \
        f"alla fine tutte le celle devono valere {K}: {n}"
    assert spediti == [2 * (K - 1)] * K, \
        f"ogni scheda spedisce {2 * (K - 1)} pezzi: {spediti}"

    stati.append([riga[:] for riga in n])   # il riposo finale, uguale all'ultimo
    return stati, passaggi, spediti[0]


# --------------------------------------------------------------------------
# Geometria: quattro schede sui quattro punti cardinali di un anello
# --------------------------------------------------------------------------
CX, CY, R = 340, 246, 148
LARG_N, ALT_C, GAP_C = 84, 19, 3
ALT_N = K * ALT_C + (K - 1) * GAP_C

ANGOLI = [-90, 0, 90, 180]      # su, destra, giù, sinistra: il giro è orario


def centro(i):
    a = math.radians(ANGOLI[i])
    return CX + R * math.cos(a), CY + R * math.sin(a)


def cella(i, c):
    """Angolo in alto a sinistra della cella `c` della scheda `i`."""
    x, y = centro(i)
    return x - LARG_N / 2, y - ALT_N / 2 + c * (ALT_C + GAP_C)


OPACITA = {1: 0.20, 2: 0.42, 3: 0.64, 4: 0.92}


def costruisci() -> Figura:
    stati, passaggi, spediti = simula()
    corpo, anim = [], []

    # l'anello, spezzato in quattro archi perche' non passi dentro le schede,
    # e le frecce che dicono il verso
    for i in range(K):
        a0, a1 = math.radians(ANGOLI[i] + 17), math.radians(ANGOLI[i] + 73)
        x0, y0 = CX + R * math.cos(a0), CY + R * math.sin(a0)
        x1, y1 = CX + R * math.cos(a1), CY + R * math.sin(a1)
        corpo.append(f'<path d="M {x0:.1f} {y0:.1f} A {R} {R} 0 0 1 {x1:.1f} {y1:.1f}" '
                     f'fill="none" stroke="{BORDER_STRONG}" stroke-width="2" '
                     f'stroke-dasharray="5 6"/>')
    for i in range(K):
        m = ANGOLI[i] + 45          # a meta' strada fra la scheda i e la i+1
        a = math.radians(m)
        px, py = CX + R * math.cos(a), CY + R * math.sin(a)
        corpo.append(f'<path d="M {px - 7:.1f} {py - 7:.1f} L {px + 7:.1f} {py:.1f} '
                     f'L {px - 7:.1f} {py + 7:.1f}" fill="none" stroke="{BORDER_STRONG}" '
                     f'stroke-width="2.4" stroke-linecap="round" '
                     f'transform="rotate({m + 90} {px:.1f} {py:.1f})"/>')

    # le quattro schede
    for i in range(K):
        x, y = centro(i)
        corpo.append(f'<rect x="{x - LARG_N / 2 - 6:.1f}" y="{y - ALT_N / 2 - 6:.1f}" '
                     f'width="{LARG_N + 12}" height="{ALT_N + 12}" rx="6" fill="none" '
                     f'stroke="{BORDER_STRONG}" stroke-width="2"/>')
        corpo.append(f'<text class="lbs" x="{x:.1f}" y="{y + ALT_N / 2 + 24:.1f}" '
                     f'text-anchor="middle">scheda {i}</text>')
        for c in range(K):
            cx0, cy0 = cella(i, c)
            valori = [OPACITA[s[i][c]] for s in stati]
            tappe = []
            for s, v in enumerate(valori):
                t0, t1 = sosta(s, STATI, tenuta=0.62)
                tappe += [(t0, f"fill-opacity:{v}"), (t1, f"fill-opacity:{v}")]
            tappe.append((100.0, f"fill-opacity:{valori[-1]}"))
            anim.append(keyframes(f"c{i}{c}", tappe))
            corpo.append(
                f'<rect x="{cx0:.1f}" y="{cy0:.1f}" width="{LARG_N}" height="{ALT_C}" '
                f'rx="3" fill="{TEAL}" fill-opacity="{valori[-1]}" '
                f'style="animation:c{i}{c} var(--d) infinite"/>')

    # i pezzi in viaggio: uno per scheda a ogni passaggio
    for p, mosse in enumerate(passaggi):
        t_via, t_arr = sosta(p, STATI, tenuta=0.62)[1], sosta(p + 1, STATI, tenuta=0.62)[0]
        for mitt, c, dest in mosse:
            x0, y0 = cella(mitt, c)
            x1, y1 = cella(dest, c)
            dx, dy = x0 - x1, y0 - y1
            nome = f"p{p}{mitt}"
            anim.append(keyframes(nome, [
                (0.0, "opacity:0"),
                (max(t_via - 0.4, 0.01), f"opacity:0;transform:translate({dx:.1f}px,{dy:.1f}px)"),
                (t_via, f"opacity:1;transform:translate({dx:.1f}px,{dy:.1f}px)"),
                (t_arr, "opacity:1;transform:translate(0px,0px)"),
                (min(t_arr + 0.4, 100.0), "opacity:0;transform:translate(0px,0px)"),
                (100.0, "opacity:0")]))
            # il pezzo in viaggio e' piu' stretto della cella, cosi' si vede
            # che sta viaggiando invece di sembrare una cella di un altro colore
            corpo.append(
                f'<rect x="{x1 + LARG_N * 0.22:.1f}" y="{y1 + 3:.1f}" '
                f'width="{LARG_N * 0.56:.1f}" height="{ALT_C - 6}" '
                f'rx="3" fill="{TERRACOTTA}" fill-opacity="0.95" opacity="0" '
                f'style="animation:{nome} var(--d) infinite"/>')

    # le didascalie dei momenti. I confini non sono gli stati ma le
    # *transizioni*, perche' e' li' che i pezzi viaggiano: una didascalia che
    # cambiasse a stato gia' cambiato arriverebbe sempre un giro in ritardo.
    TEN = 0.62
    part1 = sosta(0, STATI, TEN)[1]                 # parte il primo passaggio
    part2 = sosta(K - 1, STATI, TEN)[1]             # parte il secondo giro
    fine = sosta(2 * (K - 1), STATI, TEN)[0]        # ultimo pezzo arrivato
    momenti = [(0.0, part1, "ogni scheda ha la sua lista, tagliata in quattro"),
               (part1, part2, f"primo giro: ogni pezzo passa al vicino e si somma "
                              f"({K - 1} passaggi)"),
               (part2, fine, f"secondo giro: i pezzi finiti girano com'erano "
                             f"(altri {K - 1})"),
               (fine, 100.0, "tutte hanno il totale, e nessuna ha fatto da imbuto")]
    for m, (t0, t1, testo) in enumerate(momenti):
        ultimo = (m == len(momenti) - 1)
        anim.append(keyframes(f"m{m}", [
            (0.0, "opacity:1" if m == 0 else "opacity:0"),
            (max(t0 - 0.4, 0.01), "opacity:1" if m == 0 else "opacity:0"),
            (t0, "opacity:1"), (max(t1 - 0.4, t0 + 0.1), "opacity:1"),
            (min(t1, 100.0), "opacity:1" if ultimo else "opacity:0"),
            (100.0, "opacity:1" if ultimo else "opacity:0")]))
        corpo.append(f'<text class="lbl" x="{CX}" y="34" text-anchor="middle" '
                     f'opacity="{1 if ultimo else 0}" '
                     f'style="animation:m{m} var(--d) infinite">{testo}</text>')

    volte = f"{spediti / K:g}".replace(".", ",")
    corpo.append(f'<text class="lbs" x="22" y="500">'
                 f'ogni scheda spedisce {spediti} pezzi su {K}, cioè {volte} volte la '
                 f'propria lista,</text>')
    corpo.append(f'<text class="lbs" x="22" y="520">'
                 f'e il conto non cambia se le schede sono di più</text>')

    return Figura(
        larghezza=680, altezza=538,
        alt="Quattro schede disposte su un anello tratteggiato con le frecce "
            "che dicono il verso, ciascuna con la propria lista tagliata in "
            "quattro celle. Nel primo giro un pezzo per volta passa al vicino "
            "di destra e si somma al suo, e le celle si scuriscono via via che "
            "raccolgono i contributi; dopo tre passaggi ogni scheda ha una sola "
            "cella piena, e sono quattro celle diverse. Nel secondo giro quelle "
            "celle piene fanno il giro com'erano, e dopo altri tre passaggi "
            "tutte le celle di tutte le schede sono piene. In basso il conto: "
            "ogni scheda spedisce sei pezzi su quattro, cioè una volta e mezza "
            "la propria lista, e il conto non cambia se le schede sono di più.",
        corpo="".join(corpo),
        stile=f"""    .lbl {{ font-size:16px; }}""",
        animazioni=anim,
        durata=STATI * 1.15,
        fermi="rect, text",
    )
