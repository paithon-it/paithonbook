"""Un DataFrame e la Series che se ne stacca, sui dati della pagina.

`Python/pandas-matplotlib.md` costruisce un file `vendite.csv` con sei clienti
(nome, eta, citta, spesa) e due caselle lasciate vuote, e da lì in avanti ogni
riga della pagina lavora su quella tabella. La figura che apriva la sezione
lavorava invece su una tabella sua, con temperatura, umidità e regione: giusta
in sé, ma un terzo insieme di dati in una pagina che ne ha già uno che ricorre.

Qui la tabella è quella. E siccome è quella, la figura mostra due cose che
prima non poteva mostrare: le due caselle vuote, che sono la ragione per cui
l'età compare come 34.0 e non come 34; e l'indice 0, 1, 2, …, che pandas mette
da sé perché il file non ne portava uno.

I numeri non sono scritti a mano: la tabella la costruisce pandas con lo stesso
dizionario della pagina, la salva in CSV e la rilegge da lì, esattamente come
fa il lettore. `verifica()` difende quello che la didascalia promette, cioè che
la colonna staccata si porti dietro l'indice della tabella da cui viene.

Ferma: qui non scorre niente, è una tabella.
"""

import tempfile
from pathlib import Path

import pandas as pd

from paithon_svg import *

NOME = "pandas-series-dataframe"
TITOLO = "un DataFrame, e la Series che se ne stacca"

# --------------------------------------------------------------------------
# I dati: lo stesso dizionario che la pagina salva in vendite.csv, riletto da
# CSV perché è il giro che fa il lettore (ed è quel giro a produrre l'indice
# 0, 1, 2, … e la colonna dell'età in virgola mobile).
# --------------------------------------------------------------------------
GREZZI = {
    "nome":  ["Ada", "Bruno", "Carla", "Dario", "Elena", "Furio"],
    "eta":   [34, None, 41, 36, 52, 23],
    "citta": ["Milano", "Torino", "Milano", "Napoli", "Milano", "Torino"],
    "spesa": [120.5, 89.0, 240.0, None, 310.0, 74.9],
}
ESTRATTA = "spesa"

# Quello che la figura scrive nelle celle, e che il testo alternativo ripete a
# voce: le due copie si controllano a vicenda.
ENUMERATI = {
    "nome":  ["Ada", "Bruno", "Carla", "Dario", "Elena", "Furio"],
    "eta":   ["34.0", "NaN", "41.0", "36.0", "52.0", "23.0"],
    "citta": ["Milano", "Torino", "Milano", "Napoli", "Milano", "Torino"],
    "spesa": ["120.5", "89.0", "240.0", "NaN", "310.0", "74.9"],
}


def tabella() -> tuple[pd.DataFrame, pd.Series]:
    """La tabella come la legge la pagina, e la colonna che se ne stacca."""
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "vendite.csv"
        pd.DataFrame(GREZZI).to_csv(csv, index=False)
        df = pd.read_csv(csv)
    return df, df[ESTRATTA]


def cella(v) -> str:
    """Come pandas scrive un valore quando lo stampa."""
    if pd.isna(v):
        return "NaN"
    if isinstance(v, float):
        return f"{v:g}" if v != int(v) else f"{v:.1f}"
    return str(v)


# --------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------
LARG, ALT = 760, 404
X0, Y0 = 26.0, 74.0            # angolo alto-sinistro dell'intestazione
H_INT, H_RIGA = 32.0, 30.0
LARGHEZZE = {"__idx__": 42.0, "nome": 78.0, "eta": 60.0,
             "citta": 84.0, "spesa": 76.0}

X_SERIE = 536.0
W_SIDX, W_SVAL = 46.0, 84.0


def bordi() -> list[float]:
    """Le ascisse dei confini di colonna, da sinistra a destra."""
    x, out = X0, [X0]
    for c in ["__idx__"] + list(GREZZI):
        x += LARGHEZZE[c]
        out.append(x)
    return out


def verifica(df: pd.DataFrame, serie: pd.Series) -> None:
    """Difende quello che la didascalia promette."""
    # La promessa della didascalia: la colonna staccata si porta dietro
    # l'indice, ed è lo stesso della tabella.
    assert serie.index.equals(df.index), \
        "la Series estratta non ha l'indice del DataFrame"
    assert serie.name == ESTRATTA, "la Series ha perso il nome della colonna"
    # I valori disegnati sono quelli che il testo alternativo enumera uno per
    # uno, e nessun assert li pesava: falsando una spesa la figura ridisegnava
    # il dato nuovo e l'alt continuava a dire il vecchio.
    for col, atteso in ENUMERATI.items():
        letto = [cella(df.at[i, col]) for i in df.index]
        assert letto == atteso, \
            f"la colonna {col} non è quella che il testo alternativo dice: {letto}"

    # L'indice è 0, 1, 2, …: il file non ne portava uno, e lo mette pandas.
    assert list(df.index) == list(range(len(df))), \
        "l'indice non è quello che pandas mette da sé"
    assert len(df) == 6, f"la tabella della pagina ha sei clienti, non {len(df)}"
    assert list(df.columns) == list(GREZZI), "le colonne non sono quelle della pagina"

    # Le due caselle vuote, che la pagina commenta: una nell'età, una nella
    # spesa, e nessun'altra.
    vuote = {(c, i) for c in df.columns for i in df.index if pd.isna(df.at[i, c])}
    assert vuote == {("eta", 1), ("spesa", 3)}, \
        f"le caselle vuote non sono le due della pagina: {sorted(vuote)}"

    # È per quella casella che l'età si legge 34.0 e non 34: la figura lo mostra,
    # quindi lo deve anche essere.
    assert cella(df.at[0, "eta"]) == "34.0", "l'età non è passata alla virgola mobile"
    assert cella(df.at[0, "spesa"]) == "120.5", "la spesa di Ada non è 120.5"
    # e la ragione è quella casella, non un'altra: riempiendola, la colonna
    # torna intera.
    with tempfile.TemporaryDirectory() as tmp:
        piena = Path(tmp) / "piena.csv"
        pd.DataFrame({**GREZZI, "eta": [34, 39, 41, 36, 52, 23]}).to_csv(
            piena, index=False)
        assert cella(pd.read_csv(piena).at[0, "eta"]) == "34", \
            "senza la casella vuota l'età non torna intera"

    # Il disegno sta dentro la tela, e le due parti non si sovrappongono.
    assert bordi()[-1] < X_SERIE, "la tabella e la Series si accavallano"
    assert X_SERIE + W_SIDX + W_SVAL <= LARG - 10, "la Series esce dalla tela"


def costruisci() -> Figura:
    df, serie = tabella()
    verifica(df, serie)

    b = bordi()
    y_dati = Y0 + H_INT
    fondo = y_dati + H_RIGA * len(df)
    corpo = ['<defs><marker id="psd" viewBox="0 0 10 10" refX="8.5" refY="5" '
             'markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
             f'<path d="M 0 1 L 9 5 L 0 9 z" fill="{INK}"/></marker></defs>']

    corpo.append(f'<text class="ttl" x="{X0:.1f}" y="{Y0 - 24:.1f}">'
                 f'DataFrame: sei clienti, quattro colonne, un indice</text>')

    # ---- intestazioni di colonna ------------------------------------------
    for k, col in enumerate(df.columns):
        x, w = b[k + 1], LARGHEZZE[col]
        corpo.append(f'<rect class="int" x="{x:.1f}" y="{Y0:.1f}" '
                     f'width="{w:.1f}" height="{H_INT:.1f}"/>')
        corpo.append(f'<text class="tint" x="{x + w / 2:.1f}" '
                     f'y="{Y0 + 21:.1f}" text-anchor="middle">{col}</text>')

    # ---- indice e celle ----------------------------------------------------
    for r, i in enumerate(df.index):
        y = y_dati + r * H_RIGA
        corpo.append(f'<rect class="idx" x="{X0:.1f}" y="{y:.1f}" '
                     f'width="{LARGHEZZE["__idx__"]:.1f}" height="{H_RIGA:.1f}"/>')
        corpo.append(f'<text class="tidx" x="{X0 + LARGHEZZE["__idx__"] / 2:.1f}" '
                     f'y="{y + 20:.1f}" text-anchor="middle">{i}</text>')
        for k, col in enumerate(df.columns):
            x, w = b[k + 1], LARGHEZZE[col]
            classe = "estr" if col == ESTRATTA else "cel"
            corpo.append(f'<rect class="{classe}" x="{x:.1f}" y="{y:.1f}" '
                         f'width="{w:.1f}" height="{H_RIGA:.1f}"/>')
            testo = cella(df.at[i, col])
            stile = "tnan" if testo == "NaN" else "tcel"
            corpo.append(f'<text class="{stile}" x="{x + w / 2:.1f}" '
                         f'y="{y + 20:.1f}" text-anchor="middle">{testo}</text>')

    # ---- la freccia dell'estrazione ---------------------------------------
    xa, xb = b[-1] + 16, X_SERIE - 16
    ym = (y_dati + fondo) / 2
    corpo.append(f'<line class="frec" x1="{xa:.1f}" y1="{ym:.1f}" '
                 f'x2="{xb:.1f}" y2="{ym:.1f}" marker-end="url(#psd)"/>')
    corpo.append(f'<text class="lbs" x="{(xa + xb) / 2:.1f}" '
                 f'y="{ym - 30:.1f}" text-anchor="middle">'
                 f'si estrae la colonna</text>')
    corpo.append(f'<text class="cod" x="{(xa + xb) / 2:.1f}" y="{ym - 12:.1f}" '
                 f'text-anchor="middle">df[&quot;{ESTRATTA}&quot;]</text>')

    # ---- la Series estratta -------------------------------------------------
    corpo.append(f'<text class="ser" x="{X_SERIE:.1f}" y="{Y0 + 21:.1f}">'
                 f'Series</text>')
    for r, i in enumerate(serie.index):
        y = y_dati + r * H_RIGA
        corpo.append(f'<rect class="idx" x="{X_SERIE:.1f}" y="{y:.1f}" '
                     f'width="{W_SIDX:.1f}" height="{H_RIGA:.1f}"/>')
        corpo.append(f'<text class="tidx" x="{X_SERIE + W_SIDX / 2:.1f}" '
                     f'y="{y + 20:.1f}" text-anchor="middle">{i}</text>')
        corpo.append(f'<rect class="val" x="{X_SERIE + W_SIDX:.1f}" y="{y:.1f}" '
                     f'width="{W_SVAL:.1f}" height="{H_RIGA:.1f}"/>')
        testo = cella(serie.iloc[r])
        stile = "tnan" if testo == "NaN" else "tcel"
        corpo.append(f'<text class="{stile}" '
                     f'x="{X_SERIE + W_SIDX + W_SVAL / 2:.1f}" '
                     f'y="{y + 20:.1f}" text-anchor="middle">{testo}</text>')
    corpo.append(f'<text class="lbs" x="{X_SERIE:.1f}" y="{fondo + 20:.1f}">'
                 f'stesso indice, stessi valori</text>')

    # ---- legenda ------------------------------------------------------------
    y_leg = fondo + 44
    for x, classe, testo in [(X0, "idx", "indice (etichette di riga)"),
                             (X0 + 226, "int", "nomi di colonna"),
                             (X0 + 400, "val", "valori della Series estratta")]:
        corpo.append(f'<rect class="{classe}" x="{x:.1f}" y="{y_leg:.1f}" '
                     f'width="17" height="17"/>')
        corpo.append(f'<text class="lbs" x="{x + 25:.1f}" '
                     f'y="{y_leg + 13:.1f}">{testo}</text>')
    corpo.append(f'<text class="lbs" x="{X0:.1f}" y="{y_leg + 42:.1f}">'
                 f'Ogni colonna è una Series; tutte condividono lo stesso '
                 f'indice.</text>')
    corpo.append(f'<text class="lbs" x="{X0:.1f}" y="{y_leg + 62:.1f}">'
                 f'Le due caselle vuote si scrivono NaN.</text>')

    return Figura(
        larghezza=LARG, altezza=ALT,
        alt="A sinistra un DataFrame con sei righe e quattro colonne: nome, "
            "eta, citta, spesa, per Ada, Bruno, Carla, Dario, Elena e Furio. "
            "Sul fianco sinistro, in terracotta, la fascia delle etichette "
            "di riga, 0, 1, 2, 3, 4, 5, che pandas ha messo da sé perché "
            "il file non ne portava. Due caselle sono NaN: l'età di Bruno e "
            "la spesa di Dario. Le età si leggono 34.0, 41.0, 36.0, 52.0, "
            "23.0, in virgola mobile. La "
            "colonna della spesa è tinta di ocra. Una freccia etichettata "
            "«si estrae la colonna» e df tra quadre spesa la porta a destra, "
            "dove la stessa colonna "
            "compare da sola come Series: sei valori, 120.5, 89.0, 240.0, "
            "NaN, 310.0, 74.9, e accanto a ciascuno la stessa etichetta di "
            "riga che aveva nella tabella, da 0 a 5.",
        corpo="".join(corpo),
        stile=f"""    .int  {{ fill:{TEAL}; stroke:{INK}; stroke-width:1.4; }}
    .idx  {{ fill:{TERRACOTTA}; stroke:{INK}; stroke-width:1.4; }}
    .cel  {{ fill:none; stroke:{BORDER_STRONG}; stroke-width:1.4; }}
    .estr {{ fill:{OCRA}; fill-opacity:0.22; stroke:{BORDER_STRONG};
            stroke-width:1.4; }}
    .val  {{ fill:{OCRA}; fill-opacity:0.38; stroke:{OCRA};
            stroke-width:1.8; }}
    .tint {{ font-family:{SANS}; font-size:13.5px; font-weight:700;
            fill:{CREAM}; }}
    .tidx {{ font-family:{SANS}; font-size:13px; fill:{CREAM}; }}
    .tcel {{ font-family:{SANS}; font-size:13px; fill:{INK}; }}
    .tnan {{ font-family:{SANS}; font-size:13px; font-style:italic;
            fill:{FG_MUTED}; }}
    .frec {{ stroke:{INK}; stroke-width:2; }}
    .cod  {{ font-family:{SANS}; font-size:12.5px; fill:{INK}; }}
    .ser  {{ font-family:{SANS}; font-size:14px; font-weight:700; fill:{TEAL}; }}""",
    )
