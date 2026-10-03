"""La purga al confine fra training e test: quante righe, e perché proprio quelle.

Figura ferma. Ogni riga della tabella di una serie temporale guarda indietro
(la finestra delle feature) e avanti (il bersaglio, h passi dopo). La riga di
un istante legge fino a quell'istante compreso, che è quanto si sa quando si
emette la previsione. Al confine del training si buttano le righe il cui
bersaglio cade oltre l'ultimo istante che la prima riga di test conosce, cioè
h - 1. La figura mostra anche
il caso che fa sbagliare il conto: la prima riga di test legge fra le sue
feature valori che sono bersagli di righe di training tenute, e non c'è fuga,
perché quando si prevede quei valori sono già osservati.

La regola non è disegnata a mano: `purgate()` la applica agli indici, e gli
`assert` controllano che le righe tolte siano h - 1 e che nessuna riga tenuta
abbia un bersaglio oltre il confine.
"""

from paithon_svg import *

NOME = "purga-al-confine"
TITOLO = "la purga al confine fra training e test"

H = 7          # orizzonte: una settimana d'anticipo
R = 7          # quanto guarda indietro una riga (medie a sette giorni)
T0 = 0         # la prima riga di test: l'origine della previsione
RIGHE = list(range(T0 - 10, T0))       # le ultime dieci righe di training
TEST = T0                              # la prima riga di test


def purgate(righe, h, t0):
    """Le righe di training il cui bersaglio y[t+h] cade dopo t0, l'ultimo
    istante che la prima riga di test conosce."""
    return [t for t in righe if t + h > t0]


TOLTE = purgate(RIGHE, H, T0)
if len(TOLTE) != H - 1:
    raise AssertionError(f"la purga toglie {len(TOLTE)} righe "
                         f"invece di h - 1 = {H - 1}")
if any(t + H > T0 for t in RIGHE if t not in TOLTE):
    raise AssertionError("una riga tenuta ha il bersaglio oltre il confine")
# il caso che la figura esiste per mostrare: la prima riga di test legge
# bersagli di righe tenute, e questo è lecito
LETTI = [t for t in RIGHE if t not in TOLTE and TEST - R + 1 <= t + H <= TEST]
if not LETTI:
    raise AssertionError("la riga di test non legge nessun bersaglio di training")


def costruisci() -> Figura:
    primo = min(RIGHE) - R + 1
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
        nome = "test t₀" if tipo == "test" else f"t₀−{T0 - t}"
        corpo.append(f'<text class="lbs {cls}" x="{x0 - 12:.0f}" y="{y + 14:.0f}" '
                     f'text-anchor="end">{nome}</text>')
        g = f'<g class="{cls}">' if tolta else "<g>"
        corpo.append(g)
        # la finestra delle feature: da t-R+1 a t compreso
        corpo.append(f'<rect class="feat{" feat-test" if tipo == "test" else ""}" '
                     f'x="{sx(t - R + 1):.1f}" y="{y + 4:.0f}" '
                     f'width="{R * cella - 2:.1f}" height="{alt_riga - 8:.0f}"/>')
        # il bersaglio, h passi dopo
        corpo.append(f'<rect class="bers" x="{sx(t + H):.1f}" y="{y + 4:.0f}" '
                     f'width="{cella - 2:.1f}" height="{alt_riga - 8:.0f}"/>')
        corpo.append("</g>")
        if tolta:
            corpo.append(f'<line class="barra" x1="{sx(t - R + 1) - 4:.1f}" '
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
                 f'il confine (h − 1, con h = {H})</text>')
    corpo.append(f'<text class="lbs" x="{x0:.0f}" y="{ly + 44:.0f}">'
                 f'la riga di test legge {len(LETTI)} bersagli di righe tenute: '
                 f'è lecito, quei giorni sono già passati</text>')
    return Figura(
        larghezza=max(sx(ultimo + 1) + 20, 780), altezza=ly + 60,
        alt=f"Le ultime dieci righe di training e la prima di test, una per "
            f"riga. Ogni riga ha a sinistra la finestra di {R} giorni che legge, "
            f"fino al proprio giorno compreso, e a destra il bersaglio, {H} giorni "
            f"dopo. Una linea verticale segna il confine, subito dopo l'ultimo "
            f"giorno che la riga di test conosce. Le {len(TOLTE)} righe più vicine "
            f"al confine hanno il bersaglio oltre il confine e sono barrate: sono "
            f"quelle da togliere, una meno dei giorni d'anticipo. La riga di test "
            f"legge {len(LETTI)} giorni che sono bersagli di righe tenute, e non "
            f"è una fuga.",
        corpo="".join(corpo),
        stile=f"""    .feat {{ fill:{TEAL}; opacity:.35; }}
    .feat-test {{ fill:{OCRA}; opacity:.7; }}
    .bers {{ fill:{TERRACOTTA}; }}
    .tolta {{ opacity:.45; }}
    .barra {{ stroke:{INK}; stroke-width:1.5; }}
    .confine {{ stroke:{INK}; stroke-width:2; stroke-dasharray:5 4; }}""",
    )
