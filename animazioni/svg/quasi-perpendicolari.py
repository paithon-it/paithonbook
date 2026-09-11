"""In molte dimensioni quasi ogni coppia di direzioni e' quasi perpendicolare.

`Matematica/ortogonalita-proiezioni.md` lo afferma nella sezione «In molte
dimensioni quasi tutto e' perpendicolare», e i numeri che stampa sono tre
medie: 0,50 in tre dimensioni, 0,08 in cento, 0,03 in mille. Una media pero'
non dice se il coseno si e' **stretto** o se si e' solo spostato, ed e' lo
stringersi la cosa su cui poggia il resto (che uno spazio a mille dimensioni
ospiti molte piu' di mille direzioni quasi perpendicolari). La pagina aveva
una figura sola, sulla proiezione ai minimi quadrati, e questo passaggio
nessuna.

La distribuzione lo fa vedere in un colpo: in tre dimensioni il coseno e'
**uniforme**, cioe' un tappeto piatto in cui ogni fascia vale quanto le altre
e due direzioni possono formare qualunque angolo; in mille dimensioni resta un
picco solo, appoggiato allo zero.

I numeri li calcola la scena, rieseguendo l'esperimento che sta nella pagina
(stesso seme, stesse tremila direzioni), e `verifica()` pretende quelli della
didascalia: che a tre dimensioni le venti fasce siano piatte, le tre medie, le
tre quote sotto un decimo e il coseno massimo a mille dimensioni.

Ferma, e non e' un ripiego. Qui a cambiare non e' il tempo ma la dimensione, e
ce ne sono tre: il contenuto e' il **confronto**, che si legge tenendo i tre
istogrammi sott'occhio insieme. Una clip che li mostrasse uno per volta
toglierebbe proprio quello.
"""

import numpy as np

from paithon_svg import *

NOME = "quasi-perpendicolari"
TITOLO = "il coseno fra due direzioni a caso, al crescere delle dimensioni"

# --------------------------------------------------------------------------
# L'esperimento: e' quello della pagina, riga per riga, con lo stesso seme.
# --------------------------------------------------------------------------
DIMENSIONI = (3, 100, 1000)
DIREZIONI = 3000
SEME = 0
FASCE = 20                 # le fasce in cui si divide il coseno, da 0 a 1
SOGLIA = 0.1               # «sotto un decimo», la prima fascia


def misura_coseni() -> dict:
    """Per ogni dimensione: quota per fascia, media e massimo del coseno.

    Il coseno si prende in valore assoluto, come nella pagina: due direzioni
    opposte sono allineate quanto due coincidenti, e quello che interessa e'
    l'allineamento, non il verso.
    """
    rng = np.random.default_rng(SEME)
    fuori = {}
    for d in DIMENSIONI:
        V = rng.normal(size=(DIREZIONI, d))
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        G = np.abs(V @ V.T)
        coppie = G[np.triu_indices(DIREZIONI, 1)]
        conte, _ = np.histogram(coppie, bins=FASCE, range=(0.0, 1.0))
        fuori[d] = {
            "quote": (conte / conte.sum() * 100).tolist(),
            "media": float(coppie.mean()),
            "massimo": float(coppie.max()),
            "sotto": float((coppie < SOGLIA).mean() * 100),
        }
    return fuori


# --------------------------------------------------------------------------
# Geometria. Tre riquadri identici, stessa scala verticale: e' la scala
# condivisa a far vedere il tappeto che diventa picco.
# --------------------------------------------------------------------------
LARG, ALT = 880, 356
X0, PASSO, LARG_RIQ = 76, 270, 212
Y_ALTO, Y_BASSO = 70, 258        # il riquadro: 100% in alto, 0 in basso
Y_TITOLO = 40
Y_TACCHE, Y_MEDIA, Y_SOTTO = 276, 300, 320
GRIGLIA = (25, 50, 75, 100)      # le quote su cui passa una riga chiara


def virgola(x: float, cifre: int) -> str:
    return f"{x:.{cifre}f}".replace(".", ",")


def riquadro(i: int, d: int, dato: dict) -> str:
    """Un istogramma: le venti fasce, la griglia, le tacche, due note."""
    sx = X0 + i * PASSO
    alt = Y_BASSO - Y_ALTO
    larghezza_fascia = LARG_RIQ / FASCE
    p = [f'<text class="tt" x="{sx + LARG_RIQ / 2:.0f}" y="{Y_TITOLO}" '
         f'text-anchor="middle">{d} dimensioni</text>']

    for q in GRIGLIA:
        y = Y_BASSO - alt * q / 100
        p.append(f'<line class="gr" x1="{sx:.1f}" y1="{y:.1f}" '
                 f'x2="{sx + LARG_RIQ:.1f}" y2="{y:.1f}"/>')
        if i == 0:
            p.append(f'<text class="qy" x="{sx - 8:.1f}" y="{y + 4:.1f}" '
                     f'text-anchor="end">{q}%</text>')

    for k, quota in enumerate(dato["quote"]):
        if quota <= 0:
            continue
        h = alt * quota / 100
        p.append(f'<rect class="ba" x="{sx + k * larghezza_fascia:.2f}" '
                 f'y="{Y_BASSO - h:.2f}" width="{larghezza_fascia - 1:.2f}" '
                 f'height="{h:.2f}"/>')

    p.append(f'<line class="as" x1="{sx:.1f}" y1="{Y_BASSO:.1f}" '
             f'x2="{sx + LARG_RIQ:.1f}" y2="{Y_BASSO:.1f}"/>')
    for frazione, etichetta in ((0.0, "0"), (0.5, "0,5"), (1.0, "1")):
        x = sx + LARG_RIQ * frazione
        p.append(f'<line class="as" x1="{x:.1f}" y1="{Y_BASSO:.1f}" '
                 f'x2="{x:.1f}" y2="{Y_BASSO + 5:.1f}"/>')
        p.append(f'<text class="qy" x="{x:.1f}" y="{Y_TACCHE}" '
                 f'text-anchor="middle">{etichetta}</text>')

    p.append(f'<text class="no" x="{sx + LARG_RIQ / 2:.0f}" y="{Y_MEDIA}" '
             f'text-anchor="middle">coseno medio '
             f'{virgola(dato["media"], 2)}</text>')
    p.append(f'<text class="no" x="{sx + LARG_RIQ / 2:.0f}" y="{Y_SOTTO}" '
             f'text-anchor="middle">sotto {virgola(SOGLIA, 1)}: '
             f'{virgola(dato["sotto"], 1)}% delle coppie</text>')
    return "".join(p)


def costruisci() -> Figura:
    dati = misura_coseni()
    verifica(dati)

    corpo = "".join(riquadro(i, d, dati[d]) for i, d in enumerate(DIMENSIONI))
    mezzo = (Y_ALTO + Y_BASSO) / 2
    corpo += (f'<text class="at" transform="rotate(-90 18 {mezzo:.0f})" '
              f'x="18" y="{mezzo:.0f}" text-anchor="middle">quota delle '
              f'coppie</text>')
    corpo += (f'<text class="at" x="{LARG / 2:.0f}" y="{ALT - 8}" '
              f'text-anchor="middle">coseno fra due direzioni sorteggiate a '
              f'caso, in valore assoluto</text>')

    tre, cento, mille = (dati[d] for d in DIMENSIONI)
    return Figura(
        larghezza=LARG,
        altezza=ALT,
        alt="Tre istogrammi affiancati, con la stessa scala verticale da zero "
            "a cento per cento, mostrano come si distribuisce il coseno fra "
            "due direzioni sorteggiate a caso. A sinistra, in tre dimensioni, "
            "le venti fasce sono tutte alte uguali, un tappeto piatto al "
            "cinque per cento: ogni angolo è ugualmente probabile, e il "
            "coseno medio vale zero virgola cinquanta. Al centro, in cento "
            "dimensioni, le fasce calano da sinistra a destra e dopo la "
            "sesta non ce n'è quasi più nessuna: il coseno medio vale zero "
            "virgola zero otto. A destra, in mille dimensioni, resta un picco "
            "solo appoggiato allo zero, alto quasi il novanta per cento, e il "
            "coseno medio vale zero virgola zero tre. Sotto ogni istogramma, "
            "la quota di coppie con coseno sotto un decimo: dieci per cento "
            "in tre dimensioni, sessantotto in cento, novantanove virgola "
            "otto in mille.",
        corpo=corpo,
        stile=f"""    .ba  {{ fill:{TEAL}; fill-opacity:0.55; stroke:{TEAL};
            stroke-width:0.8; }}
    .gr  {{ stroke:{BORDER}; stroke-width:1; }}
    .as  {{ stroke:{BORDER_STRONG}; stroke-width:1.4; }}
    .tt  {{ font-family:{SANS}; font-size:14px; font-weight:700; fill:{INK}; }}
    .qy  {{ font-family:{SANS}; font-size:11px; fill:{FG_MUTED}; }}
    .no  {{ font-family:{SANS}; font-size:12px; fill:{FG_MUTED}; }}
    .at  {{ font-family:{SANS}; font-size:12.5px; fill:{INK};
            stroke:none; }}""",
    )


def verifica(dati: dict) -> None:
    """Difende quello che la didascalia promette, numero per numero."""
    tre, cento, mille = (dati[d] for d in DIMENSIONI)

    # la didascalia dice «tremila direzioni per ogni spazio» e «venti fasce»
    assert DIREZIONI == 3000 and FASCE == 20, (DIREZIONI, FASCE)

    # «in tre dimensioni le fasce sono tutte ugualmente piene»: e' la frase
    # piu' forte della didascalia, ed e' un fatto (il coseno fra due direzioni
    # uniformi sulla sfera di R^3 e' uniforme). Senza questa asserzione un
    # disegno storto la smentirebbe e nessuno se ne accorgerebbe.
    assert max(tre["quote"]) - min(tre["quote"]) < 0.2, tre["quote"]
    assert abs(sum(tre["quote"]) / FASCE - 100 / FASCE) < 1e-6

    assert round(tre["media"], 2) == 0.50, tre["media"]
    assert round(cento["media"], 2) == 0.08, cento["media"]
    assert round(mille["media"], 2) == 0.03, mille["media"]

    assert round(tre["sotto"], 1) == 10.0, tre["sotto"]
    assert round(cento["sotto"], 1) == 68.1, cento["sotto"]
    assert round(mille["sotto"], 1) == 99.8, mille["sotto"]

    # il massimo che la pagina stampa, e che la didascalia cita
    assert round(mille["massimo"], 3) == 0.166, mille["massimo"]

    # e la scala condivisa: il picco piu' alto deve stare dentro il riquadro,
    # o il disegno mostrerebbe una barra tagliata invece di un picco
    assert max(mille["quote"]) <= 100.0, max(mille["quote"])
    assert max(mille["quote"]) > 4 * max(cento["quote"]) / 2, "il picco non si stacca"
