"""Due griglie di passi sullo stesso intervallo: uniforme nel tempo, uniforme in lambda.

Figura ferma. Il punto che mostra è quello che la tabella dei campionatori
mescola: il metodo e la griglia sono due scelte diverse. La griglia uniforme
in lambda (il logaritmo del rapporto segnale-rumore) infittisce i passi
verso il dato pulito, dove il termine della rete è ripido, e sotto t = 0,1 ne
mette più del triplo di quella uniforme in t, che li spende in parti uguali.

Niente è ricopiato a mano. Il generatore esegue il blocco di codice della
pagina (`ModelliDiffusione/campionatori-veloci.md`, quello che comincia con
`MODI = np.array`), prende da lì le griglie e i quattro scarti, e controlla
che gli scarti siano gli stessi che la pagina stampa nei suoi commenti
`# ->`. Se la pagina cambia e la figura no, il generatore si ferma.
"""

import contextlib
import io
import re
from pathlib import Path

from paithon_svg import *

NOME = "griglie-di-passi"
TITOLO = "la stessa integrazione su due griglie di passi"

PAGINA = (Path(__file__).resolve().parents[2] / "book" / "ModelliDiffusione"
          / "campionatori-veloci.md")
PASSI_DISEGNATI = 16      # i passi mostrati sulle due righe
VALUTAZIONI = 32          # quelle a cui la pagina confronta gli scarti


def esegui_pagina():
    """Esegue il blocco della pagina e restituisce il suo spazio dei nomi."""
    testo = PAGINA.read_text(encoding="utf-8")
    blocchi = re.findall(r"```python\n(.*?)```", testo, re.S)
    scelti = [b for b in blocchi if "MODI = np.array" in b]
    if len(scelti) != 1:
        raise AssertionError(f"blocco dei campionatori trovato {len(scelti)} volte")
    spazio = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(scelti[0], spazio)
    return spazio, scelti[0]


def stampato(codice, etichetta, colonna):
    """Legge dalla pagina il numero stampato nel commento `# ->`."""
    for riga in codice.splitlines():
        if riga.startswith("# ->") and etichetta in riga:
            return float(riga.split()[colonna])
    raise AssertionError(f"riga '{etichetta}' non trovata nei commenti della pagina")


def dati():
    ns, codice = esegui_pagina()
    np = ns["np"]
    lam, t_di_lam, T_MIN = ns["lam"], ns["t_di_lam"], ns["T_MIN"]
    scarto = ns["scarto"]
    in_t = list(np.linspace(1.0, T_MIN, PASSI_DISEGNATI + 1))
    in_lam = [t_di_lam(L) for L in np.linspace(lam(1.0), lam(T_MIN),
                                                PASSI_DISEGNATI + 1)]
    errori = {
        ("t", "Eulero"): scarto(ns["eulero"](VALUTAZIONI)),
        ("t", "DDIM"): scarto(ns["ddim_in_t"](VALUTAZIONI)),
        ("lambda", "Eulero"): scarto(ns["eulero_in_lambda"](VALUTAZIONI)),
        ("lambda", "DDIM"): scarto(ns["ddim"](VALUTAZIONI)),
    }
    # gli stessi numeri che la pagina stampa, alla precisione a cui li stampa
    attesi = {
        ("t", "Eulero"): stampato(codice, f"{VALUTAZIONI:>11d}   ", 3),
        ("lambda", "DDIM"): stampato(codice, f"{VALUTAZIONI:>11d}   ", 4),
        ("lambda", "Eulero"): stampato(codice, "Eulero in lambda", 7),
        ("t", "DDIM"): stampato(codice, "DDIM in t", 11),
    }
    for chiave, v in errori.items():
        if f"{v:.3e}" != f"{attesi[chiave]:.3e}":
            raise AssertionError(f"{chiave}: la figura calcola {v:.3e}, la pagina "
                                 f"stampa {attesi[chiave]:.3e}")
    # le due proprietà che la didascalia promette
    ultimo_decimo = sum(1 for t in in_lam[1:] if t < 0.1)
    in_t_decimo = sum(1 for t in in_t[1:] if t < 0.1)
    if ultimo_decimo < 3 * in_t_decimo:
        raise AssertionError("la griglia in lambda non infittisce i passi vicino a t = 0")
    if not (errori[("lambda", "Eulero")] < errori[("t", "Eulero")]
            and errori[("t", "DDIM")] < errori[("lambda", "DDIM")]):
        raise AssertionError("le due griglie non premiano più i due metodi al contrario")
    guadagno = errori[("t", "Eulero")] / errori[("lambda", "Eulero")]
    if not 3.0 < guadagno < 4.0:
        raise AssertionError(f"Eulero migliora di {guadagno:.2f} volte, non «quasi quattro»")
    return in_t, in_lam, errori, (ultimo_decimo, in_t_decimo)


def cifra(v):
    """3,1 · 10⁻³, alla maniera del testo."""
    m, e = f"{v:.1e}".split("e")
    esp = str(int(e)).replace("-", "−")
    apici = str.maketrans("0123456789−", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    return f"{m.replace('.', ',')} · 10{esp.translate(apici)}"


def costruisci() -> Figura:
    in_t, in_lam, errori, ultimo_decimo = dati()
    x0, larg = 40.0, 400.0
    sx = lambda t: x0 + (1.0 - t) * larg          # t = 1 a sinistra, 0 a destra
    righe = (("uniforme in t", in_t, 70.0, "t"),
             ("uniforme in λ", in_lam, 170.0, "lambda"))
    corpo = []
    for etichetta, griglia, y, chiave in righe:
        corpo.append(f'<text class="lbl" x="{x0:.0f}" y="{y - 26:.0f}">'
                     f'griglia {etichetta}</text>')
        corpo.append(f'<line class="asse" x1="{x0:.0f}" y1="{y:.0f}" '
                     f'x2="{x0 + larg:.0f}" y2="{y:.0f}"/>')
        for t in griglia:
            corpo.append(f'<line class="passo" x1="{sx(t):.1f}" y1="{y - 11:.0f}" '
                         f'x2="{sx(t):.1f}" y2="{y + 11:.0f}"/>')
        # gli scarti a 32 valutazioni, a destra della riga
        xs = x0 + larg + 40
        for k, metodo in enumerate(("Eulero", "DDIM")):
            v = errori[(chiave, metodo)]
            migliore = v == min(errori[(chiave, "Eulero")], errori[(chiave, "DDIM")])
            corpo.append(f'<text class="{"lbl" if migliore else "lbs"}" '
                         f'x="{xs:.0f}" y="{y - 6 + 20 * k:.0f}">{metodo}  '
                         f'{cifra(v)}</text>')
    ya = 222.0
    corpo += [
        f'<line class="freccia" x1="{x0:.0f}" y1="{ya:.0f}" '
        f'x2="{x0 + larg - 14:.0f}" y2="{ya:.0f}"/>',
        f'<polygon class="punta" points="{x0 + larg - 14:.0f},{ya - 5:.0f} '
        f'{x0 + larg:.0f},{ya:.0f} {x0 + larg - 14:.0f},{ya + 5:.0f}"/>',
        f'<text class="lbs" x="{x0:.0f}" y="{ya + 20:.0f}">t = 1, rumore</text>',
        f'<text class="lbs" x="{x0 + larg:.0f}" y="{ya + 20:.0f}" '
        f'text-anchor="end">t ≈ 0, dato pulito</text>',
        f'<text class="lbs" x="{x0 + larg + 40:.0f}" y="{ya + 20:.0f}">'
        f'scarto a {VALUTAZIONI} valutazioni</text>',
    ]
    return Figura(
        larghezza=720, altezza=ya + 34,
        alt=f"Due righe con {PASSI_DISEGNATI} passi ciascuna sull'intervallo "
            f"che va dal rumore (a sinistra) al dato pulito (a destra). Nella "
            f"griglia uniforme in t i passi sono equidistanti; in quella "
            f"uniforme in lambda si infittiscono verso il dato, e nell'ultimo "
            f"decimo ne cadono {ultimo_decimo[0]} invece di {ultimo_decimo[1]}. Accanto a ogni riga lo "
            f"scarto dopo {VALUTAZIONI} valutazioni: sulla griglia in t vince "
            f"DDIM, e sulla griglia in lambda Eulero migliora di quasi quattro "
            f"volte, mentre DDIM peggiora.",
        corpo="".join(corpo),
        stile=f"""    .asse {{ stroke:{BORDER_STRONG}; stroke-width:2; }}
    .passo {{ stroke:{TEAL}; stroke-width:2; }}
    .freccia {{ stroke:{FG_MUTED}; stroke-width:1.5; }}
    .punta {{ fill:{FG_MUTED}; }}""",
    )
