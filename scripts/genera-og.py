#!/usr/bin/env python3
"""Genera l'immagine di anteprima per la condivisione (Open Graph).

Quando si manda il link del libro su WhatsApp, Telegram, LinkedIn o Slack,
l'app scarica la pagina e mostra la card con l'immagine dichiarata in
`og:image`. Prima quel meta puntava a un file che non esisteva (404), quindi la
card usciva senza immagine: solo titolo e URL.

Qui la card si compone e si rasterizza, e con lei l'icona che iOS mette
sulla schermata Home quando si aggiunge il libro:

    python3 scripts/genera-og.py     ->  book/_static/social/og-book.png
                                         book/_static/social/apple-touch-icon.png

Due scelte tecniche che vale la pena spiegare.

1. **PNG, 1200x630.** È il formato che tutte le app capiscono: WhatsApp non
   rasterizza l'SVG, e sotto i 300x200 mostra la miniatura piccola invece della
   card grande.
2. **Nessun font a runtime.** cairosvg userebbe i font di sistema, e Fraunces
   qui non c'è: il testo uscirebbe in un ripiego che non è la nostra voce.
   Quindi i glifi si vettorializzano con fontTools dal font del design system,
   esattamente come per il logo (vedi `genera-logo-book.py`).
"""

import pathlib
import re

import cairosvg
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont

RADICE = pathlib.Path(__file__).resolve().parent.parent
STATICO = RADICE / "book/_static"
FONT_DISPLAY = STATICO / "brand/motion/fonts/fraunces-600-og.ttf"
FONT_TESTO = STATICO / "brand/motion/fonts/inter-400-og.ttf"
USCITA = STATICO / "social/og-book.png"
TOKENS = STATICO / "brand/tokens.css"
SEGNO = STATICO / "brand/logos/paithon-mark.svg"
ICONA = STATICO / "social/apple-touch-icon.png"

L, H = 1200, 630                      # misura canonica delle card social



def token_scuro(nome: str) -> str:
    """Il valore di un token nel blocco `[data-theme="dark"]` del brand."""
    css = TOKENS.read_text(encoding="utf-8")
    inizio = css.index('[data-theme="dark"] {')
    blocco = css[inizio:css.index("}", inizio)]
    trovato = re.search(rf"{re.escape(nome)}:\s*(#[0-9A-Fa-f]{{6}})\b", blocco)
    assert trovato, f"{nome} non e' nel blocco scuro di {TOKENS.name}"
    return trovato.group(1).upper()


# Palette: i token del tema scuro del libro, che è come il lettore vedrà la
# pagina appena apre il link. Si leggono dal brand invece di ricopiarli.
FONDO = token_scuro("--pt-bg")
INCHIOSTRO = token_scuro("--pt-fg")
TENUE = token_scuro("--pt-fg-muted")
TEAL = token_scuro("--pt-accent-2")
TERRACOTTA = token_scuro("--pt-accent")

# L'icona della schermata Home: il segno su carta crema, come quella del sito.
# iOS non la ritaglia ma ne arrotonda gli angoli, quindi il segno sta al
# centro con un margine che gli angoli non tocchino.
LATO_ICONA = 180
LATO_SEGNO = 132
CREMA = "#F8F5EE"                     # --pt-illus-cream, la carta del segno

# Ogni riga è una sequenza di (testo, colore): "due" e "volte" portano i colori
# dei due livelli (teal l'Elementare, terracotta il Superiore) come le
# linguette nel libro.
CLAIM = [
    [("Il Libro di Intelligenza Artificiale", INCHIOSTRO)],
    [("che spiega ", INCHIOSTRO), ("due ", TEAL), ("volte.", TERRACOTTA)],
]
# Il sottotesto («l'AI che spiega se stessa… due volte») qui NON ci va, e per
# un po' c'è stato: sotto al claim diventa un'eco, perché finiscono tutte e due
# su «due volte» e la seconda sembra la prima detta peggio. Una card dice una
# cosa sola. Il sottotesto vive sulla landing, dov'è una postilla.
PIEDE = "book.paithon.it"


def testo_vettoriale(parole: str, font: TTFont, em: float, x: float, y: float,
                     colore: str) -> tuple[str, float]:
    """Compone `parole` come path SVG. Restituisce (markup, larghezza)."""
    glifi = font.getGlyphSet()
    cmap = font.getBestCmap()
    upem = font["head"].unitsPerEm
    scala = em / upem

    pezzi, avanzamento = [], 0.0
    for ch in parole:
        nome = cmap.get(ord(ch))
        if nome is None:
            avanzamento += upem * 0.3
            continue
        penna = SVGPathPen(glifi)
        glifi[nome].draw(penna)
        d = penna.getCommands()
        if d:
            pezzi.append(f'<path transform="translate({avanzamento:.0f},0)" d="{d}"/>')
        avanzamento += glifi[nome].width

    gruppo = (f'<g transform="translate({x:.2f},{y:.2f}) scale({scala:.6f},-{scala:.6f})" '
              f'fill="{colore}">' + "".join(pezzi) + "</g>")
    return gruppo, avanzamento * scala


def logo(x: float, y: float, larghezza: float) -> str:
    """Il lockup del libro, riusato dal suo SVG: geometria, non font."""
    svg = (STATICO / "logo-dark.svg").read_text()
    vb = re.search(r'viewBox="([-\d.]+) ([-\d.]+) ([-\d.]+) ([-\d.]+)"', svg)
    x0, y0, w, _ = (float(v) for v in vb.groups())
    corpo = svg[svg.index(">", svg.index("<svg")) + 1: svg.rindex("</svg>")]
    s = larghezza / w
    return (f'<g transform="translate({x - x0 * s:.2f},{y - y0 * s:.2f}) '
            f'scale({s:.6f})">{corpo}</g>')


def icona() -> None:
    """Il segno del brand, centrato su carta crema, in PNG 180x180."""
    svg = SEGNO.read_text(encoding="utf-8")
    vb = re.search(r'viewBox="([^"]+)"', svg).group(1)
    corpo = svg[svg.index(">", svg.index("<svg")) + 1: svg.rindex("</svg>")]
    margine = (LATO_ICONA - LATO_SEGNO) / 2
    tela = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{LATO_ICONA}" '
            f'height="{LATO_ICONA}" viewBox="0 0 {LATO_ICONA} {LATO_ICONA}">'
            f'<rect width="{LATO_ICONA}" height="{LATO_ICONA}" fill="{CREMA}"/>'
            f'<svg x="{margine}" y="{margine}" width="{LATO_SEGNO}" '
            f'height="{LATO_SEGNO}" viewBox="{vb}">{corpo}</svg></svg>')
    cairosvg.svg2png(bytestring=tela.encode(), write_to=str(ICONA),
                     output_width=LATO_ICONA, output_height=LATO_ICONA)
    print(f"  scritto {ICONA.relative_to(RADICE)}  {LATO_ICONA}x{LATO_ICONA}")


def main() -> None:
    display = TTFont(FONT_DISPLAY)
    testo = TTFont(FONT_TESTO)

    parti = [
        f'<rect width="{L}" height="{H}" fill="{FONDO}"/>',
        # Filo terracotta in alto: la stessa riga che il libro ha sopra la barra.
        f'<rect width="{L}" height="6" fill="{TERRACOTTA}"/>',
        logo(84, 74, 430),
    ]

    y = 372
    for riga in CLAIM:
        x = 84.0
        for parole, colore in riga:
            gruppo, larghezza = testo_vettoriale(parole, display, 58, x, y, colore)
            parti.append(gruppo)
            x += larghezza
        y += 74

    # Niente simbolo accanto al claim: i due colori dicono già che i livelli
    # sono due, e il segno accanto era una ripetizione.

    piede, _ = testo_vettoriale(PIEDE, testo, 26, 84, H - 62, TENUE)
    parti.append(piede)

    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{L}" height="{H}" '
           f'viewBox="0 0 {L} {H}">' + "".join(parti) + "</svg>")

    USCITA.parent.mkdir(parents=True, exist_ok=True)
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(USCITA),
                     output_width=L, output_height=H)
    peso = USCITA.stat().st_size
    print(f"  scritto {USCITA.relative_to(RADICE)}  {L}x{H}  {peso/1024:.0f} kB")
    if peso > 500_000:
        print("  ATTENZIONE: sopra i 500 kB, alcune app non la scaricano")
    icona()


if __name__ == "__main__":
    main()
