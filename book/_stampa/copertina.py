#!/usr/bin/env python3
"""Le bande geometriche in cima alle aperture di capitolo.

    python3 book/_stampa/copertina.py            # le scrive tutte
    python3 book/_stampa/copertina.py --provino  # un foglio da guardare

Nella bozza del 2019 ogni capitolo si apre con una banda fotografica, e la
pillola col titolo ci sta a cavallo. La banda si tiene, la fotografia no: il
libro vieta le foto stock e le immagini generate, e quelle della bozza erano
per giunta protette.

Al loro posto una composizione geometrica per capitolo, disegnata con un seme
derivato dal numero: **aria di famiglia, nessuna uguale**. Il vocabolario e'
piccolo apposta (archi, punti, onde, barre, triangoli annidati), perche' una
banda deve dire «comincia un capitolo», non raccontare qualcosa.

## Perche' passa da qui e non dal convertitore delle figure

Perche' quello lavora sulle immagini che stanno nell'albero del documento, e
queste le chiama una macro LaTeX (`\\ptBanda`). Quindi si convertono qui, con
lo stesso Chromium, e si copiano nella build con `latex_additional_files`.

## Se le bande non convincono

Si cancella `book/_stampa/bande/` e non succede niente: `\\ptBanda` ha il suo
`\\IfFileExists` e l'apertura resta la pillola col numero. E' un componente
isolato apposta.
"""

import argparse
import math
import pathlib
import sys

QUI = pathlib.Path(__file__).resolve().parent
RADICE = QUI.parent.parent
BANDE = QUI / "bande"

# A4 in px a 96 dpi: 210 mm di larghezza. L'altezza e' un quinto scarso della
# pagina: la banda deve annunciare, non occupare.
LARGA, ALTA = 794, 132

# Gli unici colori ammessi. Vedi `_static/brand/tokens.css`.
TERRACOTTA, TEAL, OCRA, NERO, CREMA = (
    "#B5532C", "#2D5A5C", "#C9A961", "#1A1A1A", "#F8F5EE")
INCHIOSTRI = (TERRACOTTA, TEAL, OCRA)


def caso(seme: int):
    """Un generatore deterministico, che non dipende da `random`.

    Serve che la banda del capitolo 7 sia sempre la stessa: se cambiasse a
    ogni build, ogni ricostruzione del libro produrrebbe un PDF diverso senza
    che nessuno abbia cambiato niente.
    """
    stato = (seme * 2654435761) % (2 ** 32)

    def prossimo(n: int) -> int:
        nonlocal stato
        stato = (stato * 1103515245 + 12345) % (2 ** 31)
        return stato % n

    return prossimo


def archi(d, colori) -> list[str]:
    """Archi concentrici che escono dal bordo."""
    pezzi, cx, cy = [], 60 + d(200), ALTA + 10
    for i in range(5 + d(3)):
        r = 40 + i * (18 + d(10))
        pezzi.append(
            f'<path d="M {cx - r} {cy} A {r} {r} 0 0 1 {cx + r} {cy}" '
            f'fill="none" stroke="{colori[i % len(colori)]}" '
            f'stroke-width="{2 + d(3)}" opacity="0.{5 + d(4)}"/>')
    return pezzi


def punti(d, colori) -> list[str]:
    """Una griglia di punti che si dirada."""
    pezzi = []
    passo = 26 + d(10)
    for x in range(30, LARGA - 20, passo):
        for y in range(24, ALTA - 10, passo):
            r = 1.5 + (x / LARGA) * (3 + d(3))
            pezzi.append(f'<circle cx="{x}" cy="{y}" r="{r:.1f}" '
                         f'fill="{colori[(x + y) % len(colori)]}" '
                         f'opacity="0.{4 + d(5)}"/>')
    return pezzi


def onde(d, colori) -> list[str]:
    """Sinusoidi sfasate."""
    pezzi = []
    for i in range(3 + d(2)):
        ampiezza = 12 + d(16)
        periodo = 90 + d(70)
        base = 30 + i * (22 + d(8))
        punti_curva = " ".join(
            f"{x},{base + ampiezza * math.sin(x / periodo + i):.1f}"
            for x in range(0, LARGA + 1, 8))
        pezzi.append(f'<polyline points="{punti_curva}" fill="none" '
                     f'stroke="{colori[i % len(colori)]}" '
                     f'stroke-width="{1.5 + d(2)}" opacity="0.{5 + d(4)}"/>')
    return pezzi


def barre(d, colori) -> list[str]:
    """Barre verticali di altezza variabile, come uno spettro."""
    pezzi, x, i = [], 24, 0
    larghezza = 8 + d(8)
    while x < LARGA - 20:
        h = 18 + d(ALTA - 40)
        pezzi.append(f'<rect x="{x}" y="{ALTA - h - 8}" width="{larghezza}" '
                     f'height="{h}" fill="{colori[i % len(colori)]}" '
                     f'opacity="0.{3 + d(6)}"/>')
        x += larghezza + 6 + d(12)
        i += 1
    return pezzi


def triangoli(d, colori) -> list[str]:
    """Triangoli annidati: e' il segno del marchio, ripetuto."""
    pezzi, x = [], 40
    while x < LARGA - 40:
        lato = 30 + d(50)
        for i in range(2 + d(2)):
            l = lato - i * 9
            if l < 8:
                break
            cy = ALTA - 18
            pezzi.append(
                f'<path d="M {x} {cy} L {x + l} {cy} '
                f'L {x + l / 2:.1f} {cy - l * 0.87:.1f} Z" fill="none" '
                f'stroke="{colori[i % len(colori)]}" stroke-width="2" '
                f'stroke-linejoin="round" opacity="0.{5 + d(4)}"/>')
        x += lato + 24 + d(40)
    return pezzi


DISEGNI = (archi, punti, onde, barre, triangoli)


def banda(numero: int) -> str:
    d = caso(numero)
    colori = list(INCHIOSTRI)
    # Ogni capitolo comincia da un inchiostro diverso: due capitoli vicini non
    # devono sembrare la stessa banda.
    colori = colori[numero % 3:] + colori[:numero % 3]
    disegno = DISEGNI[numero % len(DISEGNI)]
    corpo = "\n  ".join(disegno(d, colori))
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!--
  paithon book, banda di apertura del capitolo {numero}.
  Generata da `book/_stampa/copertina.py`: non modificare a mano.
  Palette-locked. Il seme e' il numero del capitolo, quindi la banda di un
  capitolo e' sempre la stessa: una build non deve produrre un PDF diverso
  senza che nessuno abbia cambiato niente.
-->
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {LARGA} {ALTA}"
     role="presentation">
  <rect width="{LARGA}" height="{ALTA}" fill="{CREMA}"/>
  {corpo}
  <rect x="0" y="{ALTA - 3}" width="{LARGA}" height="3" fill="{TEAL}"
        opacity="0.85"/>
</svg>
"""


def converti(svg: str, dove: pathlib.Path, pagina) -> None:
    """SVG verso PDF, con lo stesso Chromium che converte le figure."""
    sys.path.insert(0, str(RADICE / "book" / "_ext"))
    from pt_stampa import PAGINA

    temporanea = dove.with_suffix(".html")
    temporanea.write_text(
        PAGINA.format(facce="", svg=svg, w=LARGA, h=ALTA), encoding="utf-8")
    pagina.goto(temporanea.as_uri())
    pagina.pdf(path=str(dove), width=f"{LARGA}px", height=f"{ALTA}px",
               print_background=True,
               margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
    temporanea.unlink(missing_ok=True)


def quanti_capitoli() -> int:
    """Quanti ne servono. Non si scrive: si conta dal toc, piu' un margine
    per la prefazione e la bibliografia, che sono capitoli anche loro."""
    sys.path.insert(0, str(RADICE / "book" / "_ext"))
    import pt_conteggi

    return pt_conteggi.conta_capitoli(RADICE / "book" / "_toc.yml") + 4


def main() -> None:
    argomenti = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    argomenti.add_argument("--provino", action="store_true",
                           help="un PNG con le prime bande, da guardare")
    scelte = argomenti.parse_args()

    from playwright.sync_api import sync_playwright

    BANDE.mkdir(parents=True, exist_ok=True)
    totale = quanti_capitoli()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        pagina = browser.new_page()
        for numero in range(1, totale + 1):
            converti(banda(numero), BANDE / f"capitolo-{numero}.pdf", pagina)
        if scelte.provino:
            provino = "\n".join(
                f'<div style="margin-bottom:10px">{banda(n)}</div>'
                for n in range(1, 9))
            pagina.set_viewport_size({"width": LARGA, "height": (ALTA + 10) * 8})
            pagina.set_content(f'<body style="margin:0">{provino}</body>')
            pagina.screenshot(path=str(BANDE / ".provino.png"), full_page=True)
            print(f"  provino: {BANDE / '.provino.png'}")
        pagina.close()
        browser.close()

    print(f"{totale} bande in {BANDE.relative_to(RADICE)}")


if __name__ == "__main__":
    main()
