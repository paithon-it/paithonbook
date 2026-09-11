"""Le tre mosse dello split-apply-combine, sui dati della pagina.

`Python/pandas-matplotlib.md` raggruppa `vendite.csv` per città e ne fa la
media della spesa, e stampa il risultato: Milano 223.50, Napoli NaN, Torino
81.95. La figura che accompagnava quel passaggio raggruppava invece una tabella
sua, con prodotti e incassi: un terzo insieme di dati in una pagina che ne ha
già uno che ricorre, e per giunta senza il caso che la pagina spiega subito
dopo, cioè il gruppo che risponde NaN.

Qui i dati sono quelli, e il gruppo di Napoli c'è: un cliente solo, con la
casella della spesa vuota, e una media di niente non esiste. Le tre mosse
restano le stesse, dividere, applicare, ricomporre.

Nessun numero è scritto a mano: i gruppi e le medie li fa pandas con la stessa
riga della pagina. `verifica()` difende quello che la didascalia promette, cioè
che la tabella finale abbia una riga per gruppo e che la colonna su cui si è
diviso sia diventata il suo indice.

Ferma: le tre mosse sono un ordine logico, non un tempo che scorre.
"""

import tempfile
from pathlib import Path

import pandas as pd

from paithon_svg import *

NOME = "pandas-selezione-filtri-groupby"
TITOLO = "split, apply, combine: la spesa media per città"

GREZZI = {
    "nome":  ["Ada", "Bruno", "Carla", "Dario", "Elena", "Furio"],
    "eta":   [34, None, 41, 36, 52, 23],
    "citta": ["Milano", "Torino", "Milano", "Napoli", "Milano", "Torino"],
    "spesa": [120.5, 89.0, 240.0, None, 310.0, 74.9],
}
CHIAVE, VALORE = "citta", "spesa"
TINTE = {"Milano": "grA", "Napoli": "grB", "Torino": "grC"}


def tabella():
    """La tabella come la legge la pagina, i suoi gruppi e le sue medie."""
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "vendite.csv"
        pd.DataFrame(GREZZI).to_csv(csv, index=False)
        df = pd.read_csv(csv)
    gruppi = {k: g for k, g in df.groupby(CHIAVE)[VALORE]}
    medie = df.groupby(CHIAVE)[VALORE].mean()
    return df, gruppi, medie


def num(v, cifre=1) -> str:
    return "NaN" if pd.isna(v) else f"{v:.{cifre}f}"


# --------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------
LARG, ALT = 780, 372
X_ORIG, W_ORIG, H_RIGA = 26.0, 138.0, 28.0
Y_ORIG = 92.0

X_GRUP, W_GRUP, H_GRUP = 236.0, 146.0, 26.0
Y_GRUP, SALTO = 60.0, 34.0

X_APP, X_RIS, W_RIS = 390.0, 452.0, 84.0
X_FIN, W_FIDX, W_FVAL, H_FIN = 600.0, 78.0, 82.0, 30.0
Y_FIN = 128.0


def posa_gruppi(gruppi) -> dict:
    """A ogni gruppo la sua ordinata di partenza, dall'alto in basso."""
    y, out = Y_GRUP, {}
    for k, g in gruppi.items():
        out[k] = y
        y += len(g) * H_GRUP + SALTO
    return out


def verifica(df, gruppi, medie) -> None:
    """Difende quello che la didascalia promette."""
    # Tre gruppi, nell'ordine in cui pandas li restituisce (le chiavi ordinate).
    assert list(gruppi) == ["Milano", "Napoli", "Torino"], \
        f"i gruppi non sono i tre della pagina: {list(gruppi)}"
    assert [len(g) for g in gruppi.values()] == [3, 1, 2], \
        "i gruppi non hanno le dimensioni della tabella della pagina"

    # Split: niente si perde e niente si duplica.
    assert sum(len(g) for g in gruppi.values()) == len(df) == 6, \
        "i gruppi non ricompongono la tabella"
    indici = [i for g in gruppi.values() for i in g.index]
    assert sorted(indici) == list(df.index), "una riga è finita in due gruppi"

    # Combine: una riga per gruppo, e la chiave è diventata l'indice.
    assert len(medie) == len(gruppi) == 3, "la tabella finale non ha una riga per gruppo"
    assert medie.index.name == CHIAVE, "la colonna di raggruppamento non è l'indice"
    # «la chiave non è più una colonna» va provata su un oggetto che le colonne
    # ce le ha: su una Series `columns` non esiste, e l'assert che la cercava
    # lì non poteva fallire. Il secondo blocco della pagina produce il
    # DataFrame giusto per la prova.
    agg = df.groupby(CHIAVE).agg(spesa_media=(VALORE, "mean"),
                                 clienti=("nome", "count"))
    assert CHIAVE not in agg.columns and agg.index.name == CHIAVE, \
        "dopo il raggruppamento la città è rimasta anche una colonna"
    assert CHIAVE in df.columns, "la tabella di partenza ha perso la colonna citta"
    # le frecce del «ricomponi» si appoggiano a questo: la riga i-esima della
    # tabella finale è il gruppo i-esimo.
    assert list(medie.index) == list(gruppi), \
        "le frecce del ricomponi puntano alla riga sbagliata"

    # I tre numeri che la pagina stampa, e il NaN che spiega.
    assert num(medie["Milano"], 2) == "223.50", num(medie["Milano"], 2)
    assert num(medie["Torino"], 2) == "81.95", num(medie["Torino"], 2)
    assert pd.isna(medie["Napoli"]), "la media di Napoli non è NaN"
    assert gruppi["Napoli"].isna().all(), \
        "il gruppo di Napoli ha un valore: allora la media esisterebbe"

    # Il disegno sta dentro la tela, e le colonne non si accavallano.
    fondo = max(y + len(gruppi[k]) * H_GRUP for k, y in posa_gruppi(gruppi).items())
    assert fondo < ALT - 60, "i gruppi arrivano troppo in basso"
    assert X_ORIG + W_ORIG < X_GRUP < X_GRUP + W_GRUP < X_APP < X_RIS
    assert X_FIN + W_FIDX + W_FVAL <= LARG - 10, "la tabella finale esce dalla tela"


def freccia(x1, y1, x2, y2) -> str:
    return (f'<line class="frec" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" '
            f'y2="{y2:.1f}" marker-end="url(#psf)"/>')


def costruisci() -> Figura:
    df, gruppi, medie = tabella()
    verifica(df, gruppi, medie)
    ygr = posa_gruppi(gruppi)

    corpo = ['<defs><marker id="psf" viewBox="0 0 10 10" refX="8.5" refY="5" '
             'markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
             f'<path d="M 0 1 L 9 5 L 0 9 z" fill="{INK}"/></marker></defs>']

    for x, testo in [(X_ORIG + W_ORIG / 2, "la tabella"),
                     (X_GRUP + W_GRUP / 2, "1. dividi"),
                     (X_RIS + W_RIS / 2, "2. applica"),
                     (X_FIN + (W_FIDX + W_FVAL) / 2, "3. ricomponi")]:
        corpo.append(f'<text class="fase" x="{x:.1f}" y="34" '
                     f'text-anchor="middle">{testo}</text>')

    # ---- la tabella di partenza, nell'ordine in cui sta nel file -----------
    for r, i in enumerate(df.index):
        y = Y_ORIG + r * H_RIGA
        citta = df.at[i, CHIAVE]
        corpo.append(f'<rect class="{TINTE[citta]}" x="{X_ORIG:.1f}" y="{y:.1f}" '
                     f'width="{W_ORIG:.1f}" height="{H_RIGA:.1f}"/>')
        corpo.append(f'<text class="t{TINTE[citta]}" x="{X_ORIG + W_ORIG / 2:.1f}" '
                     f'y="{y + 19:.1f}" text-anchor="middle">'
                     f'{citta} · {num(df.at[i, VALORE])}</text>')

    corpo.append(f'<text class="lbs" x="{X_ORIG:.1f}" '
                 f'y="{Y_ORIG + len(df) * H_RIGA + 20:.1f}">'
                 f'gli stessi sei clienti,</text>')
    corpo.append(f'<text class="lbs" x="{X_ORIG:.1f}" '
                 f'y="{Y_ORIG + len(df) * H_RIGA + 36:.1f}">'
                 f'con la sola città e la spesa</text>')

    # ---- le frecce dello split, una per gruppo -----------------------------
    x_da = X_ORIG + W_ORIG + 8
    y_da = Y_ORIG + len(df) * H_RIGA / 2
    for k, g in gruppi.items():
        y_a = ygr[k] + len(g) * H_GRUP / 2
        corpo.append(freccia(x_da, y_da, X_GRUP - 8, y_a))

    # ---- i gruppi, poi la media di ciascuno --------------------------------
    for k, g in gruppi.items():
        for j, i in enumerate(g.index):
            y = ygr[k] + j * H_GRUP
            corpo.append(f'<rect class="{TINTE[k]}" x="{X_GRUP:.1f}" y="{y:.1f}" '
                         f'width="{W_GRUP:.1f}" height="{H_GRUP:.1f}"/>')
            corpo.append(f'<text class="t{TINTE[k]}" '
                         f'x="{X_GRUP + W_GRUP / 2:.1f}" y="{y + 18:.1f}" '
                         f'text-anchor="middle">{k} · {num(g.loc[i])}</text>')

        ym = ygr[k] + len(g) * H_GRUP / 2
        corpo.append(freccia(X_GRUP + W_GRUP + 6, ym, X_APP + 42, ym))
        corpo.append(f'<text class="lbs" x="{X_APP + 24:.1f}" '
                     f'y="{ym - 10:.1f}" text-anchor="middle">la media</text>')
        corpo.append(f'<rect class="ris {TINTE[k]}b" x="{X_RIS:.1f}" '
                     f'y="{ym - 15:.1f}" width="{W_RIS:.1f}" height="30"/>')
        media = num(medie[k], 2)
        corpo.append(f'<text class="{"tnan" if media == "NaN" else "tcel"}" '
                     f'x="{X_RIS + W_RIS / 2:.1f}" '
                     f'y="{ym + 5:.1f}" text-anchor="middle">{media}</text>')

    # ---- la tabella finale --------------------------------------------------
    for r, k in enumerate(medie.index):
        y = Y_FIN + r * H_FIN
        ym = ygr[k] + len(gruppi[k]) * H_GRUP / 2
        corpo.append(freccia(X_RIS + W_RIS + 6, ym, X_FIN - 8, y + H_FIN / 2))
        corpo.append(f'<rect class="{TINTE[k]}" x="{X_FIN:.1f}" y="{y:.1f}" '
                     f'width="{W_FIDX:.1f}" height="{H_FIN:.1f}"/>')
        corpo.append(f'<text class="t{TINTE[k]}" x="{X_FIN + W_FIDX / 2:.1f}" '
                     f'y="{y + 20:.1f}" text-anchor="middle">{k}</text>')
        corpo.append(f'<rect class="cel" x="{X_FIN + W_FIDX:.1f}" y="{y:.1f}" '
                     f'width="{W_FVAL:.1f}" height="{H_FIN:.1f}"/>')
        testo = num(medie[k], 2)
        stile = "tnan" if testo == "NaN" else "tcel"
        corpo.append(f'<text class="{stile}" '
                     f'x="{X_FIN + W_FIDX + W_FVAL / 2:.1f}" y="{y + 20:.1f}" '
                     f'text-anchor="middle">{testo}</text>')
    corpo.append(f'<text class="lbs" x="{X_FIN:.1f}" '
                 f'y="{Y_FIN + 3 * H_FIN + 22:.1f}">indice: la città</text>')

    corpo.append(f'<text class="cod" x="{LARG / 2:.1f}" y="{ALT - 46:.1f}" '
                 f'text-anchor="middle">'
                 f'df.groupby(&quot;{CHIAVE}&quot;)[&quot;{VALORE}&quot;]'
                 f'.mean()</text>')
    corpo.append(f'<text class="lbs" x="{LARG / 2:.1f}" y="{ALT - 22:.1f}" '
                 f'text-anchor="middle">Napoli ha un cliente solo e la sua '
                 f'spesa manca: senza nemmeno un valore da mediare, il gruppo '
                 f'risponde NaN.</text>')

    return Figura(
        larghezza=LARG, altezza=ALT,
        alt="Le tre mosse in fila, da sinistra a destra. A sinistra la tabella "
            "di sei clienti, una riga per ciascuno, con la città e la spesa: "
            "Milano 120.5, Torino 89.0, Milano 240.0, Napoli NaN, Milano "
            "310.0, Torino 74.9, e ogni riga tinta del colore della sua "
            "città. Tre frecce la dividono in tre gruppi: Milano con tre "
            "righe, Napoli con una sola, che porta NaN, Torino con due. Su "
            "ogni gruppo una freccia etichettata «la media» lo riduce a un "
            "numero: "
            "223.50 per Milano, NaN per Napoli, 81.95 per Torino. Tre frecce "
            "ricompongono i tre numeri in una tabella finale di tre righe, "
            "dove la città non è più una colonna ma l'etichetta di riga.",
        corpo="".join(corpo),
        stile=f"""    .grA  {{ fill:{TERRACOTTA}; fill-opacity:0.30; stroke:{TERRACOTTA};
            stroke-width:1.8; }}
    .grB  {{ fill:{OCRA}; fill-opacity:0.38; stroke:{OCRA};
            stroke-width:1.8; }}
    .grC  {{ fill:{TEAL}; fill-opacity:0.26; stroke:{TEAL};
            stroke-width:1.8; }}
    .ris  {{ fill:none; stroke-width:2; }}
    .grAb {{ stroke:{TERRACOTTA}; }}
    .grBb {{ stroke:{OCRA}; }}
    .grCb {{ stroke:{TEAL}; }}
    .cel  {{ fill:none; stroke:{BORDER_STRONG}; stroke-width:1.4; }}
    .tgrA {{ font-family:{SANS}; font-size:12.5px; fill:{INK}; }}
    .tgrB {{ font-family:{SANS}; font-size:12.5px; fill:{INK}; }}
    .tgrC {{ font-family:{SANS}; font-size:12.5px; fill:{INK}; }}
    .tcel {{ font-family:{SANS}; font-size:13px; fill:{INK}; }}
    .tnan {{ font-family:{SANS}; font-size:13px; font-style:italic;
            fill:{FG_MUTED}; }}
    .frec {{ stroke:{INK}; stroke-width:1.8; }}
    .fase {{ font-family:{SANS}; font-size:14.5px; font-weight:700;
            fill:{INK}; }}
    .cod  {{ font-family:{SANS}; font-size:12.5px; fill:{INK}; }}""",
    )
