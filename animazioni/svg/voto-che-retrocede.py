"""Il voto della meta che retrocede casella per casella, partita dopo partita.

La pagina `ReinforcementLearning/q-learning.md` fa due conti a mano (il $0{,}5$
della casella accanto alla meta, il $0{,}225$ di quella prima) e poi stampa la
griglia dopo cinquemila partite. Fra i due c'e' un buco che la didascalia della
figura ferma dichiara: «il disegno mostra il punto d'arrivo, non la strada per
arrivarci: quella si vede solo guardando i voti cambiare partita dopo partita».
Questa figura e' quella strada.

Non c'e' niente di trascritto. Gira **lo stesso identico programma della
pagina**, con il suo seme, e le sette istantanee sono le sue: dopo sette
partite tutte le caselle sono ancora a zero, all'ottava la casella accanto alla
meta prende $0{,}50$ (il conto a mano della scheda), all'undicesima quella
prende $0{,}75$ e la seconda casella prende $0{,}23$ (l'altro conto a mano), e
avanti cosi' finche' la griglia non e' quella stampata nel blocco ` ```text `.

Quello che il fermo immagine non puo' mostrare, ed e' la ragione per cui la
figura si muove: il voto non arriva dappertutto insieme. Torna indietro **di
una casella alla volta**, e ogni casella lo riceve solo dopo esserci passata
sopra, quindi fra un passo indietro e il successivo passano piu' partite
(sette per la prima, tre per la seconda, dieci per l'ultima). Nel frattempo i
voti gia' assegnati continuano a salire, perche' ciascuno rincorre un bersaglio
che a sua volta sta salendo: e' il capitolo in miniatura.

E la macchia non si chiude dove verrebbe da dire. L'angolo piu' lontano dalla
meta, quello in basso a sinistra, il voto ce l'ha gia' alla ventunesima
partita; l'ultima a riceverlo e' la casella in basso a **destra**, sotto la
trappola, perche' a decidere non e' la distanza ma quando l'agente ci passa
sopra. Un `assert` lo difende, perche' l'`alt` di questa figura ha detto il
contrario per un pomeriggio e non se n'era accorto nessun controllo.

Lo stato di riposo e' l'ultima istantanea, cioe' la griglia che il programma
della pagina stampa: chi non anima (stampa, PDF, `prefers-reduced-motion`)
vede il risultato, e lo puo' confrontare riga per riga con il blocco di codice.
"""

import numpy as np

from paithon_svg import *

NOME = "voto-che-retrocede"
TITOLO = "il voto retrocede dalla meta, partita dopo partita"

# --- il mondo, identico a quello del blocco di codice della pagina
RIGHE, COLONNE = 3, 4
MURO, META, TRAPPOLA = (1, 1), (0, 3), (1, 3)
MOSSE = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALFA, GAMMA, EPSILON = 0.5, 0.9, 0.1
SEME = 20260807
PARTITE = 5000

# le istantanee che si vedono, scelte sui momenti in cui la macchia avanza
TAPPE = [7, 8, 11, 13, 21, 60, PARTITE]


def _libere():
    return [(i, j) for i in range(RIGHE) for j in range(COLONNE)
            if (i, j) not in (MURO, META, TRAPPOLA)]


def _indice(cella):
    return cella[0] * COLONNE + cella[1]


def storia():
    """Riesegue il Q-learning della pagina e restituisce i voti alle tappe.

    E' lo stesso codice del libro, riga per riga: stessa griglia, stessi
    parametri, stesso seme, stessi inizi esplorativi. Se un giorno la pagina
    cambiasse un parametro e la figura no, `verifica()` se ne accorgerebbe,
    perche' l'ultima istantanea non sarebbe piu' la griglia stampata.
    """
    libere = _libere()
    Q = np.zeros((RIGHE * COLONNE, len(MOSSE)))
    rng = np.random.default_rng(SEME)

    def ambiente(cella, a):
        i, j = cella[0] + MOSSE[a][0], cella[1] + MOSSE[a][1]
        if not (0 <= i < RIGHE and 0 <= j < COLONNE) or (i, j) == MURO:
            i, j = cella
        if (i, j) == META:
            return (i, j), 1.0, True
        if (i, j) == TRAPPOLA:
            return (i, j), -1.0, True
        return (i, j), 0.0, False

    fuori, prima_volta = {}, {}
    for partita in range(1, PARTITE + 1):
        cella = libere[rng.integers(len(libere))]
        for _ in range(100):
            s = _indice(cella)
            if rng.random() < EPSILON:
                a = int(rng.integers(len(MOSSE)))
            else:
                a = int(np.argmax(Q[s]))
            cella_dopo, r, fine = ambiente(cella, a)
            s2 = _indice(cella_dopo)
            bersaglio = r if fine else r + GAMMA * np.max(Q[s2])
            Q[s, a] += ALFA * (bersaglio - Q[s, a])
            if fine:
                break
            cella = cella_dopo
        for c in libere:
            if c not in prima_volta and Q[_indice(c)].max() > 0:
                prima_volta[c] = partita
        if partita in TAPPE:
            fuori[partita] = {c: float(Q[_indice(c)].max()) for c in libere}

    assert len(fuori) == len(TAPPE), f"tappe mancanti: {sorted(fuori)}"
    assert len(prima_volta) == len(libere), "una casella resta senza voto"
    return [fuori[t] for t in TAPPE], prima_volta


# l'angolo che riceve un voto per ultimo: non e' quello piu' lontano dalla
# meta (l'angolo in basso a sinistra ce l'ha gia' alla ventunesima partita),
# perche' a decidere non e' la distanza ma quando l'agente ci passa sopra
ULTIMA_A_ILLUMINARSI = (2, 3)

# la griglia che il blocco ```text della pagina stampa, riga per riga
STAMPATA = {
    (0, 0): 0.81, (0, 1): 0.90, (0, 2): 1.00,
    (1, 0): 0.73, (1, 2): 0.90,
    (2, 0): 0.66, (2, 1): 0.73, (2, 2): 0.81, (2, 3): 0.73,
}


def verifica(passi, prima_volta):
    """Difende quello che la didascalia promette, e nient'altro.

    Tre cose. Che l'ultima istantanea sia **la griglia stampata nella pagina**,
    alle due cifre con cui la pagina la stampa: e' il solo modo di accorgersi
    che la figura e il testo hanno smesso di raccontare la stessa esecuzione.
    Che i due conti a mano delle schede (0,50 all'ottava partita, 0,225
    all'undicesima) siano davvero li'. E che la macchia **avanzi**, cioe' che
    il numero di caselle con un voto non cali mai e arrivi a coprirle tutte.
    """
    ultimo = passi[-1]
    for cella, atteso in STAMPATA.items():
        letto = round(ultimo[cella], 2)
        assert letto == atteso, (
            f"la casella {cella} vale {letto} e la pagina stampa {atteso}")

    ottava = passi[TAPPE.index(8)]
    assert round(ottava[(0, 2)], 2) == 0.50, (
        f"il conto a mano dice 0,50 e qui e' {ottava[(0, 2)]:.3f}")
    assert all(v == 0.0 for c, v in ottava.items() if c != (0, 2)), (
        "all'ottava partita solo la casella accanto alla meta ha un voto")

    undicesima = passi[TAPPE.index(11)]
    assert round(undicesima[(1, 2)], 3) == 0.225, (
        f"il conto a mano dice 0,225 e qui e' {undicesima[(1, 2)]:.3f}")
    assert round(undicesima[(0, 2)], 2) == 0.75, (
        "alla ripassata la casella accanto alla meta recupera meta' strada")

    quante = [sum(1 for v in p.values() if v > 0) for p in passi]
    assert quante[0] == 0, "alla settima partita nessuna casella ha un voto"
    assert quante == sorted(quante), f"la macchia arretra: {quante}"
    assert quante[-1] == len(_libere()), (
        f"alla fine {quante[-1]} caselle su {len(_libere())} hanno un voto")

    # la didascalia dice DOVE la macchia si chiude, e senza questo assert quella
    # promessa non la difendeva niente: l'alt diceva «in basso a sinistra», e
    # l'ultima a illuminarsi e' l'altro angolo. L'ordine si legge dal registro
    # partita per partita, perche' alle sole tappe le ultime tre sono gia' pari.
    ultima = max(prima_volta, key=prima_volta.get)
    assert ultima == ULTIMA_A_ILLUMINARSI, (
        f"l'ultima casella a prendere un voto e' {ultima} "
        f"(alla {prima_volta[ultima]}ª), e la didascalia dice "
        f"{ULTIMA_A_ILLUMINARSI}")
    lontana = max(prima_volta,
                  key=lambda c: abs(c[0] - META[0]) + abs(c[1] - META[1]))
    assert prima_volta[lontana] < prima_volta[ultima], (
        "la casella piu' lontana dalla meta non e' l'ultima a illuminarsi, "
        "ed e' il punto della didascalia")

    for p in passi:
        assert all(v >= 0.0 for v in p.values()), "un voto negativo"
        assert all(v <= 1.0 for v in p.values()), "un voto sopra il premio"
    return quante, lontana


def it(v: float) -> str:
    return f"{v:.2f}".replace(".", ",")


def costruisci() -> Figura:
    passi, prima_volta = storia()
    quante, lontana = verifica(passi, prima_volta)
    finale = passi[-1]

    lato, x0, y0 = 92.0, 54.0, 96.0
    # la tinta della casella dice il voto: trasparente quando e' zero, teal
    # pieno quando vale il premio. E' quella che fa la macchia.
    tinta = lambda v: 0.45 * v

    def ang(cella):
        return x0 + cella[1] * lato, y0 + cella[0] * lato

    corpo, anim = [], []

    # le tre caselle che non hanno voti: muro, meta, trappola
    for cella, testo, classe in ((MURO, "muro", "mur"),
                                 (META, "+1", "prem"),
                                 (TRAPPOLA, "−1", "pena")):
        cx, cy = ang(cella)
        corpo.append(f'<rect class="{classe}" x="{cx:.1f}" y="{cy:.1f}" '
                     f'width="{lato:.1f}" height="{lato:.1f}"/>')
        corpo.append(f'<text class="{classe}t" x="{cx + lato / 2:.1f}" '
                     f'y="{cy + lato / 2 + 8:.1f}" text-anchor="middle">'
                     f'{testo}</text>')

    for k, cella in enumerate(_libere()):
        cx, cy = ang(cella)

        # RIPOSO: la tinta del voto finale, scritta nell'attributo e non nel
        # CSS, cosi' chi non anima vede la griglia arrivata
        corpo.append(f'<rect class="cel" x="{cx:.1f}" y="{cy:.1f}" '
                     f'width="{lato:.1f}" height="{lato:.1f}" '
                     f'fill-opacity="{tinta(finale[cella]):.3f}" '
                     f'style="animation:b{k} var(--d) ease-in-out infinite;"/>')
        tappe = []
        for i, dist in enumerate(passi):
            a, b = sosta(i, len(passi))
            o = tinta(dist[cella])
            tappe.append((a, f"fill-opacity:{o:.3f}"))
            tappe.append((b, f"fill-opacity:{o:.3f}"))
        tappe.append((100.0, f"fill-opacity:{tinta(finale[cella]):.3f}"))
        anim.append(keyframes(f"b{k}", tappe))

        # i numeri: un elemento per valore, non per fotogramma, cosi' una
        # casella che non si muove per cinque tappe costa un testo solo
        gruppi, corrente = [], None
        for i, dist in enumerate(passi):
            v = it(dist[cella])
            if corrente is None or corrente[0] != v:
                corrente = [v, i, i]
                gruppi.append(corrente)
            else:
                corrente[2] = i
        for g, (v, primo, ultimo) in enumerate(gruppi):
            a, _ = sosta(primo, len(passi))
            _, b = sosta(ultimo, len(passi))
            riposo = ultimo == len(passi) - 1
            classe = "vf" if riposo else ("v" if v != "0,00" else "vz")
            corpo.append(f'<text class="{classe}" x="{cx + lato / 2:.1f}" '
                         f'y="{cy + lato / 2 + 2:.1f}" text-anchor="middle" '
                         f'style="animation:n{k}_{g} var(--d) '
                         f'ease-in-out infinite;">{v}</text>')
            # gli estremi vanno dichiarati: senza, il browser li sintetizza
            # dallo stile base e in un fermo immagine si leggono due numeri
            # sovrapposti nella stessa casella.
            passo = 100.0 / len(passi)
            tl = [(0.0, "opacity:1" if primo == 0 else "opacity:0")]
            if primo > 0:
                tl.append((max(0.1, a - passo * 0.25), "opacity:0"))
            tl.append((a, "opacity:1"))
            tl.append((min(99.9, b), "opacity:1"))
            if riposo:
                tl.append((100.0, "opacity:1"))
            else:
                tl.append((min(99.9, b + passo * 0.25), "opacity:0"))
                tl.append((100.0, "opacity:0"))
            anim.append(keyframes(f"n{k}_{g}", sorted(set(tl))))

    # il contatore delle partite, che cambia a ogni istantanea
    mx = x0 + COLONNE * lato / 2
    for i, t in enumerate(TAPPE):
        a, b = sosta(i, len(TAPPE))
        riposo = i == len(TAPPE) - 1
        testo = f"dopo {t} partite" if t != 1 else "dopo 1 partita"
        corpo.append(f'<text class="{"paf" if riposo else "pas"}" x="{mx:.1f}" '
                     f'y="{y0 - 34:.1f}" text-anchor="middle" '
                     f'style="animation:p{i} var(--d) ease-in-out infinite;">'
                     f'{testo}</text>')
        passo = 100.0 / len(TAPPE)
        tl = [(0.0, "opacity:1" if i == 0 else "opacity:0")]
        if i > 0:
            tl.append((max(0.1, a - passo * 0.25), "opacity:0"))
        tl += [(a, "opacity:1"), (min(99.9, b), "opacity:1")]
        if riposo:
            tl.append((100.0, "opacity:1"))
        else:
            tl += [(min(99.9, b + passo * 0.25), "opacity:0"),
                   (100.0, "opacity:0")]
        anim.append(keyframes(f"p{i}", sorted(set(tl))))

    corpo.append(f'<text class="lbs" x="{x0:.1f}" '
                 f'y="{y0 + RIGHE * lato + 26:.1f}">'
                 f'in ogni casella, il voto della mossa migliore</text>')

    # la legenda, che dice che cosa si sta guardando
    lx = x0 + COLONNE * lato + 34
    righe = ["la macchia parte dalla meta",
             "e retrocede di una casella",
             "alla volta: ogni casella",
             "aspetta di esserci passata",
             "sopra, e intanto i voti gia'",
             "assegnati continuano a salire"]
    for i, r in enumerate(righe):
        corpo.append(f'<text class="cod" x="{lx:.1f}" y="{y0 + 26 + i * 20:.1f}">'
                     f'{r.replace("gia'", "già")}</text>')

    parole = {0: "nessuna", 1: "una", 2: "due", 3: "tre", 4: "quattro",
              5: "cinque", 6: "sei", 7: "sette", 8: "otto", 9: "nove"}
    righe_finali = "; ".join(
        ", ".join(it(finale[(i, j)]) for j in range(COLONNE)
                  if (i, j) in finale)
        for i in range(RIGHE))
    return Figura(
        larghezza=760, altezza=420,
        alt="Griglia di tre righe per quattro colonne: la meta che paga più "
            "uno in alto a destra, la trappola che ne toglie uno subito sotto, "
            "un muro al centro, e nove caselle libere in ciascuna delle quali "
            "si legge il voto della mossa migliore, con la casella tinta tanto "
            "più intensamente quanto più quel voto è alto. Si parte con tutte "
            f"e nove le caselle a zero dopo {TAPPE[0]} partite; alla "
            f"{TAPPE[1]}ª la sola casella accanto alla meta prende "
            f"{it(passi[1][(0, 2)])}; alla {TAPPE[2]}ª quella sale a "
            f"{it(passi[2][(0, 2)])} e la casella prima prende "
            f"{it(passi[2][(1, 2)])}; alla {TAPPE[3]}ª le caselle con un voto "
            f"sono {parole[quante[3]]}, alla {TAPPE[4]}ª {parole[quante[4]]}, "
            f"alla {TAPPE[5]}ª tutte e {parole[quante[5]]}. L'ultima a "
            "prendere un voto è la casella in basso a destra, sotto la "
            "trappola, e non l'angolo più lontano dalla meta, che ce l'ha già "
            f"alla {prima_volta[lontana]}ª. Dopo {TAPPE[-1]} partite i voti "
            f"sono {righe_finali}, "
            "riga per riga dall'alto in basso.",
        corpo="".join(corpo),
        stile=f"""    .cel  {{ fill:{TEAL}; stroke:{BORDER_STRONG}; stroke-width:1.5; }}
    .mur  {{ fill:{BORDER_STRONG}; stroke:{BORDER_STRONG}; stroke-width:1.5; }}
    .prem {{ fill:{OCRA}; fill-opacity:0.45; stroke:{OCRA}; stroke-width:2; }}
    .pena {{ fill:{TERRACOTTA}; fill-opacity:0.3; stroke:{TERRACOTTA};
            stroke-width:2; }}
    .murt {{ font-family:{SANS}; font-size:14px; fill:{INK}; }}
    .premt,.penat {{ font-family:{SANS}; font-size:20px; font-weight:700;
            fill:{INK}; }}
    .bar  {{ fill:{TEAL}; fill-opacity:0.85; }}
    .v    {{ font-family:{SANS}; font-size:19px; font-weight:700; fill:{INK};
            opacity:0; }}
    .vz   {{ font-family:{SANS}; font-size:19px; fill:{FG_MUTED}; opacity:0; }}
    .vf   {{ font-family:{SANS}; font-size:19px; font-weight:700; fill:{INK};
            opacity:1; }}
    .pas  {{ font-family:{SANS}; font-size:18px; font-weight:700;
            fill:{TERRACOTTA}; opacity:0; }}
    .paf  {{ font-family:{SANS}; font-size:18px; font-weight:700;
            fill:{TERRACOTTA}; opacity:1; }}
    .cod  {{ font-family:{SANS}; font-size:12.5px; fill:{FG_MUTED}; }}""",
        animazioni=anim,
        durata=len(TAPPE) * 1.7,
        fermi=".cel, .v, .vz, .vf, .pas, .paf",
    )
