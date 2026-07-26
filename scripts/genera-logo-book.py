#!/usr/bin/env python3
"""Compone i loghi del libro: il lockup paithon + la parola «book».

Il logo del sito è il bollo tribar più il wordmark «paithon» in Fraunces
vettorializzato (nessuna dipendenza da font a runtime). Qui si aggiunge
«book» ricavando i contorni dei glifi dal Fraunces SemiBold del design
system — così anche la parola nuova è geometria, non testo.

Escono due lockup dagli stessi pezzi:

* **impilato** (`logo-light.svg`, `logo-dark.svg`) — «book» su una seconda
  riga allineata a destra. È il logo della sidebar, dove c'è larghezza.
* **in linea** (`logo-inline-light.svg`, `logo-inline-dark.svg`) — «book»
  sulla stessa linea di base, dopo uno spazio. Serve alla barra in alto su
  schermo piccolo, dove l'altezza è quella di un'icona.

Lo script è **idempotente**: rimuove il gruppo `pt-book` se già presente e
riporta il viewBox alla base, poi ricompone. Si può rilanciare a piacere dopo
aver toccato i parametri qui sotto.

    python3 scripts/genera-logo-book.py

Font: book/_static/brand/motion/fonts/fraunces-600-og.ttf (OFL, vedi
NOTICE.md nel design system). Il submodule non viene modificato: si legge.
"""

import pathlib
import re

from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont

RADICE = pathlib.Path(__file__).resolve().parent.parent
STATICO = RADICE / "book/_static"
FONT = STATICO / "brand/motion/fonts/fraunces-600-og.ttf"

# L'inchiostro del wordmark, per tema. Le chiavi sono i suffissi dei file.
INCHIOSTRO = {"light": "#1A1714", "dark": "#F4ECDD"}

# --- geometria del logo esistente (misurata, non indovinata) ---------------
# Il wordmark «paithon» vive in un gruppo `translate(166.189,0) scale(0.06,-0.06)`
# su un font da 2000 unità per em: quindi un em vale 120 unità SVG, la linea di
# base è y=0 e le ascendenti arrivano a y=-90.
EM_PAITHON = 120.0
X_FINE_PAITHON = 641.3        # bordo destro del wordmark, in unità SVG
VIEWBOX_BASE = (-8.00, -98.24, 660.15, 134.20)

# --- parametri regolabili --------------------------------------------------
PAROLA = "book"
MARGINE = 8.0                 # margine del viewBox, come nell'originale

RAPPORTO_IMPILATO = 0.78      # grandezza di «book» rispetto a «paithon»
ARIA_SOPRA = 6.0              # fra la linea di base di «paithon» e la cima
                              # reale dell'inchiostro di «book»

RAPPORTO_LINEA = 1.0          # in linea: stessa grandezza, peso più leggero
SPAZIO_LINEA = 0.24           # spazio fra le due parole, in em

MARCATORE = '<g id="pt-book"'


def contorni(parola: str, font: TTFont):
    """(lista di (dx, path), advance totale, cima dell'inchiostro), unità font.

    La cima si misura sui contorni veri, non su `hhea.ascent`: quel valore
    include l'interlinea del font e lascerebbe un buco fra le due righe.
    """
    glifi = font.getGlyphSet()
    cmap = font.getBestCmap()
    pezzi, x, cima = [], 0.0, 0.0
    for ch in parola:
        nome = cmap[ord(ch)]
        penna = SVGPathPen(glifi)
        glifi[nome].draw(penna)
        pezzi.append((x, penna.getCommands()))

        limiti = BoundsPen(glifi)
        glifi[nome].draw(limiti)
        if limiti.bounds:
            cima = max(cima, limiti.bounds[3])
        x += glifi[nome].width
    return pezzi, x, cima


def base(tema: str) -> str:
    """Il logo senza «book»: si ricava dal file su disco, non da una copia."""
    testo = (STATICO / f"logo-{tema}.svg").read_text()
    testo = re.sub(r"\n?    " + re.escape(MARCATORE) + r".*?</g>\n", "\n", testo, flags=re.S)
    x0, y0, w, h = VIEWBOX_BASE
    return re.sub(r'viewBox="[^"]*"', f'viewBox="{x0:.2f} {y0:.2f} {w:.2f} {h:.2f}"',
                  testo, count=1)


def componi(tema: str, gruppo: str, viewbox: str, etichetta: str, uscita: pathlib.Path):
    testo = base(tema)
    testo = re.sub(r'viewBox="[^"]*"', f'viewBox="{viewbox}"', testo, count=1)
    testo = re.sub(r'aria-label="[^"]*"', f'aria-label="{etichetta}"', testo, count=1)
    uscita.write_text(testo.replace("</svg>", gruppo + "</svg>", 1))
    print(f"  scritto {uscita.relative_to(RADICE)}")


def main() -> None:
    font = TTFont(FONT)
    upem = font["head"].unitsPerEm
    pezzi, advance, cima = contorni(PAROLA, font)

    def gruppo(x: float, y: float, scala: float, inchiostro: str) -> str:
        corpo = "\n".join(
            f'      <path transform="translate({dx:.0f},0)" d="{d}"/>' for dx, d in pezzi
        )
        return (
            f'    {MARCATORE} transform="translate({x:.3f},{y:.3f}) '
            f'scale({scala:.6f},-{scala:.6f})" fill="{inchiostro}">\n'
            f"{corpo}\n    </g>\n"
        )

    x0, y0, larghezza_base, _ = VIEWBOX_BASE

    # --- impilato: «book» sotto, allineato a destra ---
    scala_i = EM_PAITHON * RAPPORTO_IMPILATO / upem
    largh_i = advance * scala_i
    x_i = X_FINE_PAITHON - largh_i
    base_i = ARIA_SOPRA + cima * scala_i
    vb_i = f"{x0:.2f} {y0:.2f} {larghezza_base:.2f} {(base_i + MARGINE) - y0:.2f}"

    # --- in linea: «book» dopo «paithon», stessa linea di base ---
    scala_l = EM_PAITHON * RAPPORTO_LINEA / upem
    largh_l = advance * scala_l
    x_l = X_FINE_PAITHON + EM_PAITHON * SPAZIO_LINEA
    vb_l = f"{x0:.2f} {y0:.2f} {(x_l + largh_l + MARGINE) - x0:.2f} {VIEWBOX_BASE[3]:.2f}"

    for tema, inchiostro in INCHIOSTRO.items():
        componi(tema, gruppo(x_i, base_i, scala_i, inchiostro), vb_i,
                "paithon book", STATICO / f"logo-{tema}.svg")
        componi(tema, gruppo(x_l, 0.0, scala_l, inchiostro), vb_l,
                "paithon book", STATICO / f"logo-inline-{tema}.svg")

    print(f"\n  impilato: em {EM_PAITHON * RAPPORTO_IMPILATO:.1f}, "
          f"«book» x {x_i:.1f}–{X_FINE_PAITHON:.1f}, base {base_i:.1f}")
    print(f"            viewBox {vb_i}")
    print(f"  in linea: em {EM_PAITHON * RAPPORTO_LINEA:.1f}, "
          f"«book» x {x_l:.1f}–{x_l + largh_l:.1f}")
    print(f"            viewBox {vb_l}  (rapporto "
          f"{(x_l + largh_l + MARGINE - x0) / VIEWBOX_BASE[3]:.2f}:1)")


if __name__ == "__main__":
    main()
