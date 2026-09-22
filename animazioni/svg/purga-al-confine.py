"""La purga al confine fra training e test: quante righe, e perché proprio quelle.

Figura ferma. Ogni riga della tabella di una serie temporale guarda indietro
(la finestra delle feature) e avanti (il bersaglio, h passi dopo). Al confine
del training si buttano le righe il cui bersaglio cade oltre l'ultimo istante
che la prima riga di test conosce, cioè esattamente h. La figura mostra anche
il caso che fa sbagliare il conto: la prima riga di test legge fra le sue
feature valori che sono bersagli di righe di training tenute, e non c'è fuga,
perché quando si prevede quei valori sono già osservati.

La regola non è disegnata a mano: `purgate()` la applica agli indici, e gli
`assert` controllano che le righe tolte siano h e che nessuna riga tenuta
abbia un bersaglio oltre il confine.
"""

from paithon_svg import *

NOME = "purga-al-confine"
TITOLO = "la purga al confine fra training e test"

H = 7          # orizzonte: una settimana d'anticipo
R = 7          # quanto guarda indietro una riga (medie a sette giorni)
T0 = 0         # ultimo istante del training (gli indici sono relativi)
RIGHE = list(range(T0 - 9, T0 + 1))    # le ultime dieci righe di training
TEST = T0 + 1                          # la prima riga di test


def purgate(righe, h, t0):
    """Le righe di training il cui bersaglio y[t+h] cade dopo t0."""
    return [t for t in righe if t + h > t0]


TOLTE = purgate(RIGHE, H, T0)
if len(TOLTE) != H:
    raise AssertionError(f"la purga toglie {len(TOLTE)} righe invece di h = {H}")
if any(t + H > T0 for t in RIGHE if t not in TOLTE):
    raise AssertionError("una riga tenuta ha il bersaglio oltre il confine")
# il caso che la figura esiste per mostrare: la prima riga di test legge
# bersagli di righe tenute, e questo è lecito
LETTI = [t for t in RIGHE if t not in TOLTE and TEST - R <= t + H <= TEST - 1]
if not LETTI:
    raise AssertionError("la riga di test non legge nessun bersaglio di training")


def costruisci() -> Figura:
    primo = min(RIGHE) - R
    ultimo = TEST + H
    x0, cella, alt_riga = 150.0, 17.0, 22.0
    sx = lambda i: x0 + (i - primo) * cella
    y0 = 58.0
    corpo = []
    tutte = [(t, "train") for t in RIGHE] + [(TEST, "test")]
    for k, (t, tipo) in enumerate(tutte):
        y = y0 + k * alt_riga + (10 if tipo == "test" else 0)
        tolta = tipo == "train" and t in TOLTE
        cls = "tolta" if tolta else ""
        nome = (f"test t₀+1" if tipo == "test"
                else (f"t₀−{T0 - t}" if t != T0 else "t₀"))
        corpo.append(f'<text class="lbs {cls}" x="{x0 - 12:.0f}" y="{y + 14:.0f}" '
                     f'text-anchor="end">{nome}</text>')
        g = f'<g class="{cls}">' if tolta else "<g>"
        corpo.append(g)
        # la finestra delle feature: da t-R a t-1
        corpo.append(f'<rect class="feat{" feat-test" if tipo == "test" else ""}" '
                     f'x="{sx(t - R):.1f}" y="{y + 4:.0f}" '
                     f'width="{R * cella - 2:.1f}" height="{alt_riga - 8:.0f}"/>')
        # il bersaglio, h passi dopo
        corpo.append(f'<rect class="bers" x="{sx(t + H):.1f}" y="{y + 4:.0f}" '
                     f'width="{cella - 2:.1f}" height="{alt_riga - 8:.0f}"/>')
        corpo.append("</g>")
        if tolta:
            corpo.append(f'<line class="barra" x1="{sx(t - R) - 4:.1f}" '
                         f'y1="{y + 11:.0f}" x2="{sx(t + H) + cella + 2:.1f}" '
                         f'y2="{y + 11:.0f}"/>')
    fondo = y0 + len(tutte) * alt_riga + 16
    # il confine
    xc = sx(T0 + 1) - 1
    corpo.append(f'<line class="confine" x1="{xc:.1f}" y1="{y0 - 8:.0f}" '
                 f'x2="{xc:.1f}" y2="{fondo:.0f}"/>')
    corpo.append(f'<text class="lbl" x="{xc:.1f}" y="{y0 - 16:.0f}" '
                 f'text-anchor="middle">confine</text>')
    # legenda
    ly = fondo + 30
    for dx, cls, testo in ((0, "feat", "feature: i giorni che la riga legge"),
                           (250, "bers", "bersaglio, h giorni dopo")):
        corpo.append(f'<rect class="{cls}" x="{x0 + dx:.0f}" y="{ly - 11:.0f}" '
                     f'width="13" height="13"/>')
        corpo.append(f'<text class="lbs" x="{x0 + dx + 20:.0f}" y="{ly:.0f}">'
                     f'{testo}</text>')
    corpo.append(f'<text class="lbs" x="{x0:.0f}" y="{ly + 24:.0f}">'
                 f'barrate: le {len(TOLTE)} righe tolte, il cui bersaglio cade oltre '
                 f'il confine (h = {H})</text>')
    corpo.append(f'<text class="lbs" x="{x0:.0f}" y="{ly + 44:.0f}">'
                 f'la riga di test legge {len(LETTI)} bersagli di righe tenute: '
                 f'è lecito, quei giorni sono già passati</text>')
    return Figura(
        larghezza=max(sx(ultimo + 1) + 20, 780), altezza=ly + 60,
        alt=f"Le ultime dieci righe di training e la prima di test, una per "
            f"riga. Ogni riga ha a sinistra la finestra di {R} giorni che legge "
            f"e a destra il bersaglio, {H} giorni dopo. Una linea verticale "
            f"segna il confine. Le {len(TOLTE)} righe più vicine al confine hanno "
            f"il bersaglio oltre il confine e sono barrate: sono quelle da "
            f"togliere, esattamente h. La riga di test legge {len(LETTI)} giorni "
            f"che sono bersagli di righe tenute, e non è una fuga.",
        corpo="".join(corpo),
        stile=f"""    .feat {{ fill:{TEAL}; opacity:.35; }}
    .feat-test {{ fill:{OCRA}; opacity:.7; }}
    .bers {{ fill:{TERRACOTTA}; }}
    .tolta {{ opacity:.45; }}
    .barra {{ stroke:{INK}; stroke-width:1.5; }}
    .confine {{ stroke:{INK}; stroke-width:2; stroke-dasharray:5 4; }}""",
    )
