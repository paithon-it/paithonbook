"""La deriva del codice: pulizia continua e pulizia a ondate, a pari permanenza.

La sezione sull'entropia fa il conto con un modello a campo medio: la frazione
di codice che segue uno schema storto, sessione dopo sessione, con una pulizia
continua e con una a ondate che lasciano a ogni schema storto la stessa
permanenza media. Le medie le stampa il programma; la forma no, ed è il
contenuto: la continua si assesta su un valore, quella a ondate sale fra un
passaggio e l'altro e ricade, con la stessa media e picchi più alti, e con il
contagio tutto il disegno sale.

I numeri non sono scritti a mano: questo file **esegue il blocco di codice
della sezione**, preso dal `.md`, e disegna le serie che ne escono.
`controlla()` confronta ciò che il programma stampa con il blocco ```text della
sezione, e difende quello che la didascalia promette.

Lo stato di riposo è il conto concluso: le curve disegnate per intero e le
medie in vista. L'animazione scopre le curve da sinistra a destra con un
ritaglio che parte schiacciato e finisce sull'identità, con andamento lineare,
così che a metà della durata le curve siano a metà delle sessioni disegnate.
"""

import contextlib
import io
import re
from pathlib import Path

from paithon_svg import *

NOME = "harness-deriva"
TITOLO = "la deriva del codice con la pulizia continua e con quella a ondate"

QUI = Path(__file__).resolve().parent
SEZIONE = QUI.parents[1] / "book" / "Harness" / "quando-nessuno-guarda.md"
MARCA = "def deriva("
PERMANENZA = 10                  # quella che la figura disegna
DISEGNATE = 60                   # le sessioni in orizzontale


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
    spazio = {"__name__": "quando_nessuno_guarda"}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(codice, str(SEZIONE), "exec"), spazio)   # noqa: S102
    return spazio, buf.getvalue().splitlines()


def serie(spazio):
    """Per ciascun mondo: le due curve disegnate e le loro medie a regime."""
    mondi = []
    for contagio, nome in [(0, "copia in proporzione"),
                           (spazio["CONTAGIO"], "copia con contagio")]:
        lunga_c = spazio["deriva"](contagio, PERMANENZA)
        lunga_o = spazio["deriva"](contagio, PERMANENZA, a_ondate=True)
        mondi.append({
            "nome": nome,
            "continua": lunga_c[:DISEGNATE],
            "ondate": lunga_o[:DISEGNATE],
            "media_c": spazio["media"](lunga_c),
            "media_o": spazio["media"](lunga_o),
            "picco_o": max(lunga_o[-spazio["REGIME"]:]),
        })
    return mondi


def controlla(spazio, stampate, mondi):
    atteso = _blocco("text").splitlines()
    if stampate != atteso:
        raise ValueError(f"{NOME}: il capitolo stampa righe diverse dal suo "
                         f"blocco di uscita:\n  {stampate}\n  {atteso}")
    for m in mondi:
        # la didascalia: stessa permanenza, quasi la stessa media
        if abs(m["media_o"] - m["media_c"]) > 0.02 * m["media_c"]:
            raise ValueError(f"{NOME}: {m['nome']}, le medie delle due pulizie "
                             f"non sono quasi uguali: {m['media_c']:.3f} e "
                             f"{m['media_o']:.3f}")
        # la didascalia: le ondate salgono e ricadono sopra la continua
        if m["picco_o"] <= m["media_c"] * 1.1:
            raise ValueError(f"{NOME}: {m['nome']}, le ondate non hanno picchi "
                             f"visibili sopra la media")
    # la didascalia: con il contagio tutto il disegno sale
    if mondi[1]["media_c"] <= mondi[0]["media_c"] * 1.3:
        raise ValueError(f"{NOME}: il contagio non alza abbastanza la media "
                         f"per vederlo")
    if max(max(m["ondate"]) for m in mondi) >= ALTEZZA_B:
        raise ValueError(f"{NOME}: una curva esce dal grafico, che arriva a "
                         f"{ALTEZZA_B}")


def it(x: float) -> str:
    return f"{x:.3f}".replace(".", ",")


# --------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------
W, H = 800, 430
X0, X1 = 92, 700                 # l'asse delle sessioni
ALTO = 120                       # altezza di un grafico
Y_CORSIE = [70, 250]             # la y del bordo alto di ciascun grafico
ALTEZZA_B = 0.4                  # la frazione storta in cima al grafico
RIVELA = 0.86                    # la frazione della durata in cui si scopre


def costruisci() -> Figura:
    spazio, stampate = esegui_il_capitolo()
    mondi = serie(spazio)
    controlla(spazio, stampate, mondi)

    passo = (X1 - X0) / (DISEGNATE - 1)

    def px(t: float) -> float:
        return X0 + t * passo

    def py(y0: float, valore: float) -> float:
        return y0 + ALTO - valore / ALTEZZA_B * ALTO

    corpo, anim = [], []
    anim.append(keyframes("scopri", [
        (0.0, "transform:scaleX(0)"),
        (RIVELA * 100, "transform:scaleX(1)"),
        (100.0, "transform:scaleX(1)")]))
    anim.append(keyframes("medie", [
        (0.0, "opacity:0"), (RIVELA * 100, "opacity:0"),
        (min(100.0, RIVELA * 100 + 6), "opacity:1"), (100.0, "opacity:1")]))

    corpo.append(f'<text class="ttl" x="{X0}" y="26">la frazione di codice '
                 f'storto, sessione dopo sessione</text>')

    for k, m in enumerate(mondi):
        y0 = Y_CORSIE[k]
        clip = f"cl{k}"
        corpo.append(
            f'<clipPath id="{clip}"><rect class="rit" x="{X0 - 2}" '
            f'y="{y0 - 8}" width="{X1 - X0 + 10}" height="{ALTO + 16}" '
            f'style="animation:scopri var(--d) linear infinite"/></clipPath>')
        corpo.append(f'<line class="ax" x1="{X0}" y1="{y0 + ALTO}" x2="{X1}" '
                     f'y2="{y0 + ALTO}"/>')
        for v in (0.1, 0.2, 0.3):
            yg = py(y0, v)
            corpo.append(f'<line class="griglia" x1="{X0}" y1="{yg:.1f}" '
                         f'x2="{X1}" y2="{yg:.1f}"/>')
            corpo.append(f'<text class="lbs" x="{X0 - 8}" y="{yg + 4:.1f}" '
                         f'text-anchor="end">{it(v)[:3]}</text>')
        corpo.append(f'<text class="lbl nome" x="{X0}" y="{y0 - 14}">'
                     f'{m["nome"]}</text>')

        # la media della pulizia continua, la linea di riferimento
        ym = py(y0, m["media_c"])
        corpo.append(f'<line class="media" x1="{X0}" y1="{ym:.1f}" x2="{X1}" '
                     f'y2="{ym:.1f}"/>')
        corpo.append(f'<text class="lbs med" x="{X0 + 6}" y="{ym - 6:.1f}">'
                     f'media a regime</text>')

        corpo.append(f'<g clip-path="url(#{clip})">')
        for chiave, classe in [("ondate", "lin-o"), ("continua", "lin-v")]:
            punti = [(px(t), py(y0, v)) for t, v in enumerate(m[chiave])]
            d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in punti)
            corpo.append(f'<path class="{classe}" d="{d}"/>')
        corpo.append('</g>')

        # le etichette in fondo alle curve, e le medie a conto finito
        yc = py(y0, m["continua"][-1])
        yo = py(y0, max(m["ondate"][-6:]))
        corpo.append(f'<text class="lbs fin-v" x="{X1 + 8}" y="{yc + 4:.1f}">'
                     f'continua</text>')
        corpo.append(f'<text class="lbs fin-o" x="{X1 + 8}" '
                     f'y="{min(yo, yc - 14) + 4:.1f}">a ondate</text>')
        corpo.append(
            f'<g class="tot" opacity="1" style="animation:medie var(--d) '
            f'infinite"><text class="lbl" x="{X1}" y="{y0 + ALTO + 22}" '
            f'text-anchor="end">media: continua {it(m["media_c"])}, a ondate '
            f'{it(m["media_o"])}; picco delle ondate {it(m["picco_o"])}'
            f'</text></g>')

    corpo.append(f'<text class="lbs" x="{X0}" y="{H - 8}">sessione 1</text>')
    corpo.append(f'<text class="lbs" x="{X1}" y="{H - 8}" text-anchor="end">'
                 f'sessione {DISEGNATE}</text>')

    p, c = mondi
    return Figura(
        larghezza=W, altezza=H,
        alt=f"Due grafici, uno sopra l'altro, con le sessioni in orizzontale e "
            f"la frazione di codice storto in verticale, da zero a "
            f"{it(ALTEZZA_B)[:3]}. In ciascuno due curve con la stessa "
            f"permanenza media di uno schema storto, {PERMANENZA} sessioni: "
            f"la pulizia continua sale e si assesta su un valore, quella a "
            f"ondate sale fra un passaggio e l'altro e ricade, a dente di "
            f"sega. In alto, con la copia in proporzione, la media è "
            f"{it(p['media_c'])} per la continua e {it(p['media_o'])} per le "
            f"ondate, che toccano {it(p['picco_o'])}. In basso, con il "
            f"contagio, tutto sale: media {it(c['media_c'])} e "
            f"{it(c['media_o'])}, picchi a {it(c['picco_o'])}.",
        corpo="".join(corpo),
        stile=f"""    .rit  {{ transform-box:view-box; transform-origin:{X0 - 2}px 0px; }}
    .griglia {{ stroke:{BORDER}; stroke-width:1; }}
    .media {{ stroke:{TERRACOTTA}; stroke-width:1.4; stroke-dasharray:6 4; }}
    .nome {{ font-weight:700; }}
    .lin-o {{ fill:none; stroke:{OCRA}; stroke-width:3; stroke-linejoin:round; }}
    .lin-v {{ fill:none; stroke:{TEAL}; stroke-width:3; stroke-linejoin:round; }}
    .fin-v {{ fill:{TEAL}; font-weight:700; }}
    .fin-o {{ fill:{FG_MUTED}; font-weight:700; }}
    .med {{ fill:{TERRACOTTA}; }}""",
        animazioni=anim,
        durata=9.0,
        fermi=".rit, .tot",
    )
