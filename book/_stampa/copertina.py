#!/usr/bin/env python3
"""Il fregio di copertina e le bande in cima alle aperture di capitolo.

    python3 book/_stampa/copertina.py            # le scrive tutte
    python3 book/_stampa/copertina.py --provino  # un foglio da guardare

Due cose, non una, e la differenza conta. Le **bande** sono trentasette e
nascono da un seme (il numero del capitolo): aria di famiglia, nessuna
uguale, e nessuna deve raccontare qualcosa. Il **fregio** è uno solo, sta in
copertina, e invece racconta: e' composto a mano, e il soggetto e' il gesto
che il libro spiega dal capitolo di matematica fino all'ultimo, cioe' una
discesa che trova il fondo di una conca.

La sua traiettoria e' **calcolata**, non disegnata a occhio: le curve di
livello sono quelle della quadratica su cui la discesa scende, la discesa e'
un'iterazione con momento, e le oscillazioni si vedono perche' la conca e'
sette volte piu' larga che alta. Vale la regola delle animazioni: se una
figura del libro mostra un numero, quel numero lo calcola lei.

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
import re
import math
import pathlib
import sys

QUI = pathlib.Path(__file__).resolve().parent
RADICE = QUI.parent.parent
BANDE = QUI / "bande"

# A4 in px a 96 dpi: 210 mm di larghezza. L'altezza e' un quinto scarso della
# pagina: la banda deve annunciare, non occupare.
LARGA, ALTA = 794, 132

# Il fregio di copertina: stessa larghezza, ma alto, perche' li' lo spazio
# c'e' e la prima pagina del libro era due terzi di crema vuota.
FREGIO_LARGO, FREGIO_ALTO = 794, 420

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
  Paithon Book, banda di apertura del capitolo {numero}.
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


def discesa(cx, cy, a, b, partenza, lr, mu, passi):
    """La traiettoria del fregio: una discesa del gradiente vera.

    Non e' una polilinea disegnata a occhio che «sembra» una discesa. La
    funzione e' la quadratica di cui il fregio disegna le curve di livello,
    e i punti escono dall'iterazione con momento: le oscillazioni si vedono
    perche' la conca e' piu' stretta in verticale che in orizzontale, che e'
    esattamente la ragione per cui esistono. Un disegno del gesto centrale
    del libro non puo' essere finto.
    """
    (x, y), (vx, vy) = partenza, (0.0, 0.0)
    punti_traccia = [(x, y)]
    for _ in range(passi):
        gx, gy = 2 * (x - cx) / a ** 2, 2 * (y - cy) / b ** 2
        vx, vy = mu * vx - lr * gx, mu * vy - lr * gy
        x, y = x + vx, y + vy
        punti_traccia.append((x, y))
    return punti_traccia


def fregio() -> str:
    """La composizione della copertina: una conca e la discesa che la trova.

    Il vocabolario e' quello delle bande (curve, punti, il segno del
    marchio), ma qui i pezzi non sono presi a caso da un seme: la copertina
    e' una sola e va composta. Il soggetto e' il gesto che il libro racconta
    per intero, dal capitolo sulla matematica fino all'ultimo: si parte
    lontano, si scende, si oscilla nella valle stretta, si arriva.
    """
    # La conca e' molto piu' larga che alta (a/b vale sette), e non e' una
    # scelta estetica: e' la condizione che fa oscillare la discesa. In una
    # conca tonda la traiettoria scenderebbe dritta e non ci sarebbe niente
    # da vedere, ne' da spiegare.
    cx, cy, a, b = 520, 230, 320, 64
    livelli = (0.06, 0.25, 0.6, 1.1, 1.8, 2.8, 4.2)
    pezzi = []

    # Texture di fondo: la griglia dei dati, che si dirada dove il disegno
    # ha qualcosa da dire. Sta sotto tutto, e infatti si disegna per prima.
    for gx in range(26, FREGIO_LARGO - 10, 34):
        for gy in range(26, FREGIO_ALTO - 10, 34):
            lontano = min(1.0, (((gx - cx) / a) ** 2
                                + ((gy - cy) / b) ** 2) / 12.0)
            if lontano < 0.35:
                continue
            pezzi.append(
                f'<circle cx="{gx}" cy="{gy}" r="{1.4 + lontano * 1.6:.1f}" '
                f'fill="{OCRA}" opacity="{0.10 + lontano * 0.30:.2f}"/>')

    # Le curve di livello: le stesse della quadratica su cui poi si scende.
    # Le piu' esterne escono dal foglio, ed e' voluto: la conca continua.
    for i, f in enumerate(reversed(livelli)):
        colore = (TEAL, OCRA, TERRACOTTA)[i % 3]
        pezzi.append(
            f'<ellipse cx="{cx}" cy="{cy}" rx="{a * f ** 0.5:.1f}" '
            f'ry="{b * f ** 0.5:.1f}" fill="none" stroke="{colore}" '
            f'stroke-width="{1.2 + i * 0.25:.1f}" '
            f'opacity="{0.22 + i * 0.09:.2f}"/>')

    traccia = discesa(cx, cy, a, b, (55, 95), lr=1400, mu=0.72, passi=70)
    dentro = [(x, y) for x, y in traccia
              if -40 < x < FREGIO_LARGO + 40 and -40 < y < FREGIO_ALTO + 40]
    pezzi.append(
        '<polyline points="'
        + " ".join(f"{x:.1f},{y:.1f}" for x, y in dentro)
        + f'" fill="none" stroke="{TERRACOTTA}" stroke-width="2.6" '
        'stroke-linejoin="round" stroke-linecap="round" opacity="0.9"/>')
    # I passi, uno ogni due: si vedono fitti dove la discesa rallenta, ed e'
    # l'unica cosa che una polilinea da sola non direbbe.
    for i, (x, y) in enumerate(dentro[::2]):
        pezzi.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" '
                     f'r="{max(1.3, 3.4 - i * 0.055):.1f}" '
                     f'fill="{TERRACOTTA}" '
                     f'opacity="{max(0.35, 0.85 - i * 0.012):.2f}"/>')

    # Il minimo, col segno del marchio: un triangolo, e dentro il suo
    # rimpicciolito. E' l'unico posto del fregio in cui il disegno cita il
    # logo, e cade dove la discesa finisce.
    for lato, colore, spessore in ((34, TEAL, 2.4), (20, TERRACOTTA, 2.2)):
        pezzi.append(
            f'<path d="M {cx - lato / 2:.1f} {cy + lato * 0.29:.1f} '
            f'L {cx + lato / 2:.1f} {cy + lato * 0.29:.1f} '
            f'L {cx} {cy - lato * 0.58:.1f} Z" fill="none" stroke="{colore}" '
            f'stroke-width="{spessore}" stroke-linejoin="round"/>')

    corpo = "\n  ".join(pezzi)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!--
  Paithon Book, il fregio di copertina.
  Generato da `book/_stampa/copertina.py`: non modificare a mano.
  Palette-locked. La traiettoria e' una discesa del gradiente calcolata, non
  disegnata: le curve di livello sono quelle della funzione su cui scende.
-->
<svg xmlns="http://www.w3.org/2000/svg"
     viewBox="0 0 {FREGIO_LARGO} {FREGIO_ALTO}" role="presentation">
  <rect width="{FREGIO_LARGO}" height="{FREGIO_ALTO}" fill="{CREMA}"/>
  {corpo}
</svg>
"""


def senza_data(pdf: pathlib.Path) -> None:
    """Toglie la data di creazione dal PDF, che e' l'unica cosa che cambia.

    Il seme rende il disegno di una banda sempre lo stesso, ma Chromium
    stampa dentro il PDF l'istante in cui l'ha scritto: due rigenerazioni a
    disegno immutato producono trentasette blob nuovi, che in un repository
    dove i PDF sono **tracciati** significa qualche megabyte di storia per
    niente, e un `git status` che grida al cambiamento quando non e'
    cambiato nulla. Misurato: il disegno era identico pixel per pixel, e la
    sola differenza erano `creationDate` e `modDate`.

    Se PyMuPDF non c'e' non succede niente di grave: si torna al
    comportamento di prima, che e' rumoroso ma corretto.
    """
    try:
        import fitz
    except ImportError:
        return
    documento = fitz.open(pdf)
    documento.set_metadata({"producer": "Paithon Book", "creator": "",
                            "creationDate": "", "modDate": ""})
    # Anche l'identificativo del documento va fissato: Chromium ne genera uno
    # a caso a ogni stampa e PyMuPDF ne genera un altro a ogni salvataggio.
    # Senza questa riga le date spariscono e i file restano lo stesso tutti
    # diversi, che e' il modo peggiore di risolvere un problema: sembra fatto.
    documento.xref_set_key(-1, "ID", "[<70616974686F6E><626F6F6B>]")
    documento.save(pdf.with_suffix(".tmp.pdf"), garbage=3, deflate=True,
                   no_new_id=True)
    documento.close()
    pdf.with_suffix(".tmp.pdf").replace(pdf)


def converti(svg: str, dove: pathlib.Path, pagina,
             larga: int = LARGA, alta: int = ALTA) -> None:
    """SVG verso PDF, con lo stesso Chromium che converte le figure."""
    sys.path.insert(0, str(RADICE / "book" / "_ext"))
    from pt_stampa import PAGINA

    temporanea = dove.with_suffix(".html")
    temporanea.write_text(
        PAGINA.format(facce="", svg=svg, w=larga, h=alta), encoding="utf-8")
    pagina.goto(temporanea.as_uri())
    pagina.pdf(path=str(dove), width=f"{larga}px", height=f"{alta}px",
               print_background=True,
               margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
    temporanea.unlink(missing_ok=True)
    senza_data(dove)


def quanti_capitoli() -> int:
    """Quanti ne servono. Non si scrive: si conta dal toc, piu' un margine
    per la prefazione e la bibliografia, che sono capitoli anche loro."""
    sys.path.insert(0, str(RADICE / "book" / "_ext"))
    import pt_conteggi

    return pt_conteggi.conta_capitoli(RADICE / "book" / "_toc.yml") + 4


def marchio() -> tuple[str, int, int]:
    """Il logo del libro per la copertina a stampa, dal file del sito.

    Prima questo PDF non lo generava nessuno: era stato fatto a mano una volta
    e committato, e quando il logo e' cambiato (la «a» diventata il tribar, il
    15 agosto 2026) la copertina del PDF ha continuato a mostrare quello
    vecchio senza dirlo, perche' la ricetta LaTeX trova il file e lo mette.
    Adesso si rifa' da `logo-light.svg`, cosi' non puo' piu' restare indietro.

    Si prende la versione chiara perche' la copertina e' su fondo crema; le
    misure escono dal `viewBox`, non si scrivono.
    """
    svg = (RADICE / "book" / "_static" / "logo-light.svg").read_text()
    numeri = re.search(r'viewBox="[-\d.]+ [-\d.]+ ([\d.]+) ([\d.]+)"', svg)
    larga, alta = (float(v) for v in numeri.groups())
    scala = 900.0 / larga            # abbastanza grande da non impastare i tratti
    return svg, round(larga * scala), round(alta * scala)


def main() -> None:
    argomenti = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    argomenti.add_argument("--provino", action="store_true",
                           help="un PNG con le prime bande e il fregio, "
                                "da guardare")
    scelte = argomenti.parse_args()

    from playwright.sync_api import sync_playwright

    BANDE.mkdir(parents=True, exist_ok=True)
    totale = quanti_capitoli()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        pagina = browser.new_page()
        for numero in range(1, totale + 1):
            converti(banda(numero), BANDE / f"capitolo-{numero}.pdf", pagina)
        converti(fregio(), QUI / "fregio.pdf", pagina,
                 FREGIO_LARGO, FREGIO_ALTO)
        svg_marchio, m_largo, m_alto = marchio()
        converti(svg_marchio, QUI / "marchio.pdf", pagina, m_largo, m_alto)
        if scelte.provino:
            provino = (f'<div style="margin-bottom:16px">{fregio()}</div>'
                       + "\n".join(
                           f'<div style="margin-bottom:10px">{banda(n)}</div>'
                           for n in range(1, 9)))
            pagina.set_viewport_size(
                {"width": LARGA, "height": FREGIO_ALTO + (ALTA + 10) * 8})
            pagina.set_content(f'<body style="margin:0">{provino}</body>')
            pagina.screenshot(path=str(BANDE / ".provino.png"), full_page=True)
            print(f"  provino: {BANDE / '.provino.png'}")
        pagina.close()
        browser.close()

    print(f"{totale} bande in {BANDE.relative_to(RADICE)}, "
          f"piu' il fregio e il marchio di copertina")


if __name__ == "__main__":
    main()
