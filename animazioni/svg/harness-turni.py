"""La finestra che si riempie: tre strategie per sessanta iterazioni.

La sezione sul passaggio di consegne fa il conto dei token letti con più
strategie (nessuna, la compattazione, la ripartenza da capo), e il conto ha una
forma che una tabella di totali non mostra: senza strategie la linea sale
dritta e si ferma contro il tetto della finestra; con la compattazione crolla
al riassunto e risale fino alla soglia, due volte; con la ripartenza disegna
denti di sega che restano bassi. Qui il tempo è il contenuto, perché le tre
corsie avanzano insieme un'iterazione dopo l'altra, e la differenza fra le
strategie è proprio in come il contesto cambia mentre il lavoro procede.

I numeri non sono scritti a mano: questo file **esegue il blocco di codice
della sezione**, preso dal `.md`, e disegna le serie che ne escono.
`controlla()` confronta ciò che il programma stampa con il blocco ```text della
sezione: se un giorno il codice del libro cambiasse, la figura non si
genererebbe più invece di smentirlo in silenzio.

Lo stato di riposo è il conto concluso: le tre linee disegnate per intero e i
tre totali in vista. L'animazione scopre le linee da sinistra a destra con un
ritaglio che parte schiacciato e finisce sull'identità, con andamento lineare,
così che a metà della durata le tre corsie siano tutte a metà del compito.
"""

import contextlib
import io
import re
from pathlib import Path

from paithon_svg import *

NOME = "harness-turni"
TITOLO = "la finestra che si riempie, con tre strategie diverse"

QUI = Path(__file__).resolve().parent
SEZIONE = QUI.parents[1] / "book" / "Harness" / "passaggio-di-consegne.md"
MARCA = "def con_ripartenza("
SOGLIA = 0.8                     # la compattazione che la figura disegna


# --------------------------------------------------------------------------
# Il codice del capitolo, eseguito davvero
# --------------------------------------------------------------------------
def _blocco(marcatore: str) -> str:
    """Il blocco con la marca (``python``) o la sua uscita (``text``)."""
    testo = SEZIONE.read_text(encoding="utf-8")
    trovati = [m for m in re.finditer(r"^```python\n(.*?)^```", testo,
                                      re.S | re.M)
               if MARCA in m.group(1)]
    if len(trovati) != 1:
        raise ValueError(f"{NOME}: {len(trovati)} blocchi ```python con "
                         f"{MARCA!r} in {SEZIONE.name}, ne serve esattamente 1")
    codice = trovati[0]
    if marcatore == "python":
        return codice.group(1)
    uscita = re.match(r"\n*```text\n(.*?)^```", testo[codice.end():],
                      re.S | re.M)
    if not uscita:
        raise ValueError(f"{NOME}: il blocco in {SEZIONE.name} non ha il suo "
                         f"```text subito dopo")
    return uscita.group(1)


def esegui_il_capitolo():
    codice = _blocco("python")
    spazio = {"__name__": "passaggio_di_consegne"}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(codice, str(SEZIONE), "exec"), spazio)   # noqa: S102
    return spazio, buf.getvalue().splitlines()


def controlla(spazio, stampate):
    atteso = _blocco("text").splitlines()
    if stampate != atteso:
        raise ValueError(f"{NOME}: il capitolo stampa righe diverse dal suo "
                         f"blocco di uscita:\n  {stampate}\n  {atteso}")
    if spazio["con_compattazione"](SOGLIA)[1] != spazio["cb"]:
        raise ValueError(f"{NOME}: la corsia della compattazione non è quella "
                         f"con la soglia al {SOGLIA:.0%} che il capitolo stampa")
    if len(spazio["ca"]) >= spazio["ITERAZIONI"]:
        raise ValueError(f"{NOME}: la prima corsia deve fermarsi prima della "
                         f"fine, si ferma dopo {len(spazio['ca'])} iterazioni")
    if spazio["volte"] != 2 or spazio["sessioni"] != 4:
        raise ValueError(f"{NOME}: la figura disegna 2 compattazioni e 4 "
                         f"sessioni, il capitolo ne fa {spazio['volte']} e "
                         f"{spazio['sessioni']}")
    if max(spazio["cc"]) * 2 > spazio["FINESTRA"]:
        raise ValueError(f"{NOME}: la didascalia promette che la ripartenza non "
                         f"supera la metà della finestra")


def it(n: int) -> str:
    return f"{n:,}".replace(",", ".")


# --------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------
W, H = 800, 466
X0, X1 = 92, 760                 # l'asse delle iterazioni
H_CORSIA, ALTO = 96, 74          # altezza di una corsia e del suo grafico
Y_CORSIE = [80, 212, 344]        # la y del tetto della finestra in ciascuna corsia
RIVELA = 0.86                    # la frazione della durata in cui si scopre


def costruisci() -> Figura:
    spazio, stampate = esegui_il_capitolo()
    controlla(spazio, stampate)

    iterazioni, finestra = spazio["ITERAZIONI"], spazio["FINESTRA"]
    soglia = SOGLIA * finestra
    fermo = len(spazio["ca"])
    passo = (X1 - X0) / (iterazioni - 1)

    def px(t: float) -> float:
        return X0 + t * passo

    def py(y0: float, valore: float) -> float:
        return y0 + ALTO - valore / finestra * ALTO

    corsie = [
        ("senza strategie", spazio["ca"], spazio["a"],
         f"si ferma dopo {fermo} iterazioni"),
        (f"con la compattazione all'{SOGLIA:.0%}",
         spazio["cb"], spazio["b"], f"{spazio['volte']} riassunti"),
        ("con la ripartenza a ogni funzionalità", spazio["cc"], spazio["c"],
         f"{spazio['sessioni']} sessioni"),
    ]

    corpo, anim = [], []
    anim.append(keyframes("scopri", [
        (0.0, "transform:scaleX(0)"),
        (RIVELA * 100, "transform:scaleX(1)"),
        (100.0, "transform:scaleX(1)")]))
    anim.append(keyframes("totale", [
        (0.0, "opacity:0"), (RIVELA * 100, "opacity:0"),
        (min(100.0, RIVELA * 100 + 6), "opacity:1"), (100.0, "opacity:1")]))

    corpo.append(f'<text class="ttl" x="{X0}" y="26">quanto testo il modello '
                 f'rilegge a ogni iterazione</text>')

    for k, (nome, serie, letti, nota) in enumerate(corsie):
        y0 = Y_CORSIE[k]
        clip = f"cl{k}"
        corpo.append(
            f'<clipPath id="{clip}"><rect class="rit" x="{X0 - 2}" '
            f'y="{y0 - 6}" width="{X1 - X0 + 14}" height="{ALTO + 12}" '
            f'style="animation:scopri var(--d) linear infinite"/></clipPath>')
        # il tetto della finestra, la base, e per la compattazione la soglia
        corpo.append(f'<line class="tetto" x1="{X0}" y1="{y0}" x2="{X1}" '
                     f'y2="{y0}"/>')
        corpo.append(f'<line class="ax" x1="{X0}" y1="{y0 + ALTO}" x2="{X1}" '
                     f'y2="{y0 + ALTO}"/>')
        corpo.append(f'<text class="lbs" x="{X0 - 8}" y="{y0 + 4}" '
                     f'text-anchor="end">finestra</text>')
        if k == 1:
            ys = py(y0, soglia)
            corpo.append(f'<line class="sogl" x1="{X0}" y1="{ys:.1f}" '
                         f'x2="{X1}" y2="{ys:.1f}"/>')
            corpo.append(f'<text class="lbs" x="{X0 + 4}" y="{ys + 15:.1f}">'
                         f'soglia di compattazione</text>')
        if k == 2:
            ym = py(y0, finestra / 2)
            corpo.append(f'<line class="meta" x1="{X0}" y1="{ym:.1f}" '
                         f'x2="{X1}" y2="{ym:.1f}"/>')
            corpo.append(f'<text class="lbs" x="{X0 - 8}" y="{ym + 4:.1f}" '
                         f'text-anchor="end">metà</text>')
        corpo.append(f'<text class="lbl nome" x="{X0}" y="{y0 - 14}">{nome}'
                     f'</text>')

        # La linea: un punto per iterazione. Dove il contesto crolla (riassunto
        # o sessione nuova) la linea scende ripida fra due iterazioni.
        punti = [(px(t), py(y0, v)) for t, v in enumerate(serie)]
        d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in punti)
        classe = ["lin-t", "lin-o", "lin-v"][k]
        corpo.append(f'<g clip-path="url(#{clip})"><path class="{classe}" '
                     f'd="{d}"/>')
        if k == 0:
            xf, yf = punti[-1]
            r = 7
            corpo.append(f'<path class="stop" d="M {xf + 8 - r:.1f} {yf - r:.1f} '
                         f'L {xf + 8 + r:.1f} {yf + r:.1f} M {xf + 8 + r:.1f} '
                         f'{yf - r:.1f} L {xf + 8 - r:.1f} {yf + r:.1f}"/>')
        corpo.append('</g>')
        if k == 0:
            # fuori dal ritaglio, che la taglierebbe sopra il tetto: compare
            # con i totali, a conto finito
            corpo.append(f'<g class="tot" opacity="1" style="animation:totale '
                         f'var(--d) infinite"><text class="lbs stopl" '
                         f'x="{xf + 22:.1f}" y="{yf + 18:.1f}">non ci sta più'
                         f'</text></g>')

        # i totali, che compaiono a conto finito
        corpo.append(
            f'<g class="tot" opacity="1" style="animation:totale var(--d) '
            f'infinite"><text class="lbl" x="{X1}" y="{y0 + ALTO + 22}" '
            f'text-anchor="end">{nota}, letti {it(letti)} token</text></g>')

    corpo.append(f'<text class="lbs" x="{X0}" y="{H - 8}">iterazione 1</text>')
    corpo.append(f'<text class="lbs" x="{X1}" y="{H - 8}" text-anchor="end">'
                 f'iterazione {iterazioni}</text>')

    return Figura(
        larghezza=W, altezza=H,
        alt=f"Tre corsie, una per strategia, con le iterazioni del compito in "
            f"orizzontale e l'occupazione della finestra in verticale. Nella "
            f"prima, senza strategie, la linea sale dritta fino al tetto della "
            f"finestra e si interrompe dopo {fermo} iterazioni, dopo aver "
            f"letto {it(spazio['a'])} token. Nella seconda, con la "
            f"compattazione all'ottanta per cento, la linea sale fino alla "
            f"soglia, crolla al riassunto e risale, {spazio['volte']} volte, "
            f"per {it(spazio['b'])} token letti in tutto, compresi quelli "
            f"riletti per scrivere i riassunti. Nella terza, con la "
            f"ripartenza a ogni funzionalità, la linea sale e torna in basso "
            f"a ogni nuova sessione, {spazio['sessioni']} denti di sega che "
            f"restano sotto la linea della metà della finestra, per "
            f"{it(spazio['c'])} token letti.",
        corpo="".join(corpo),
        stile=f"""    .rit  {{ transform-box:view-box; transform-origin:{X0 - 2}px 0px; }}
    .tetto {{ stroke:{TERRACOTTA}; stroke-width:1.4; stroke-dasharray:6 4; }}
    .sogl {{ stroke:{OCRA}; stroke-width:1.4; stroke-dasharray:3 4; }}
    .meta {{ stroke:{TEAL}; stroke-width:1.2; stroke-dasharray:2 5; }}
    .nome {{ font-weight:700; }}
    .lin-t {{ fill:none; stroke:{TERRACOTTA}; stroke-width:3; stroke-linejoin:round; }}
    .lin-o {{ fill:none; stroke:{OCRA}; stroke-width:3; stroke-linejoin:round; }}
    .lin-v {{ fill:none; stroke:{TEAL}; stroke-width:3; stroke-linejoin:round; }}
    .stop {{ stroke:{TERRACOTTA}; stroke-width:3; fill:none; stroke-linecap:round; }}
    .stopl {{ fill:{TERRACOTTA}; font-weight:700; }}""",
        animazioni=anim,
        durata=9.0,
        fermi=".rit, .tot",
    )
