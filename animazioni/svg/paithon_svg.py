"""Figure animate in SVG: la controparte leggera delle scene Manim.

Perché esistono due strade. Manim rende **fotogrammi**: ottimo per LaTeX denso,
curve e coreografie, ma l'uscita è raster e pesa ~570 KB a clip. Qui invece si
descrive il movimento con qualche `@keyframes` CSS su pochi elementi: il file
resta sotto i 6 KB, è testo (quindi diffabile in git) e i fotogrammi intermedi
li calcola il browser.

Regola fondativa, che vale come "l'ultimo fotogramma deve reggere da solo" per
Manim: **lo stato di riposo dell'SVG è lo stato finale**. Si scrive come
attributo di presentazione, e l'animazione CSS lo sovrascrive solo mentre gira.
Così la stampa, il PDF, `prefers-reduced-motion` e qualunque rasterizzatore
vedono la figura *conclusa* — e non serve salvare nessun PNG a parte.

    python3 animazioni/svg/genera.py            # rigenera tutte le figure
    python3 animazioni/svg/genera.py percettrone-impara

Ogni generatore espone `NOME`, `TITOLO` e `costruisci() -> Figura`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Palette — la stessa, fissa, delle illustrazioni editoriali. Nessun altro
# colore può comparire: `scrivi()` lo verifica e rifiuta il file.
# --------------------------------------------------------------------------
TERRACOTTA = "#B5532C"
TEAL = "#2D5A5C"
OCRA = "#C9A961"
INK = "#1A1A1A"
CREAM = "#F8F5EE"
FG_MUTED = "#5E5852"
BORDER = "#E2DCC9"
BORDER_STRONG = "#C5BEAA"

AMMESSI = {TERRACOTTA, TEAL, OCRA, INK, CREAM, FG_MUTED, BORDER, BORDER_STRONG}

# I font: un SVG caricato in <img> non vede i webfont della pagina e ripiega
# sui font di sistema. Le 108 figure statiche del libro fanno già così.
SANS = "Inter, system-ui, sans-serif"
SERIF = "Fraunces, Georgia, serif"

RADICE = Path(__file__).resolve().parents[2]
FIGURE = RADICE / "book" / "figures"
PROVINI = Path.home() / ".cache" / "paithon-svg"


# --------------------------------------------------------------------------
# Geometria: dal piano dei dati a quello dell'SVG (dove la y cresce in giù)
# --------------------------------------------------------------------------
@dataclass
class Riquadro:
    """Area di disegno più la mappa dati → SVG."""

    x: float = 78
    y: float = 34
    larg: float = 392
    alt: float = 392
    xmin: float = -2.15
    xmax: float = 2.15
    ymin: float = -2.15
    ymax: float = 2.15

    def sx(self, x: float) -> float:
        return self.x + (x - self.xmin) / (self.xmax - self.xmin) * self.larg

    def sy(self, y: float) -> float:
        return self.y + self.alt - (y - self.ymin) / (self.ymax - self.ymin) * self.alt

    @property
    def scala_x(self) -> float:
        return self.larg / (self.xmax - self.xmin)

    @property
    def scala_y(self) -> float:
        return self.alt / (self.ymax - self.ymin)

    def cornice(self, croce: bool = False) -> str:
        s = (f'<rect class="ax" x="{self.x}" y="{self.y}" '
             f'width="{self.larg}" height="{self.alt}" rx="4"/>')
        if croce:
            s += (f'<line class="axc" x1="{self.x}" y1="{self.sy(0):.1f}" '
                  f'x2="{self.x + self.larg}" y2="{self.sy(0):.1f}"/>'
                  f'<line class="axc" x1="{self.sx(0):.1f}" y1="{self.y}" '
                  f'x2="{self.sx(0):.1f}" y2="{self.y + self.alt}"/>')
        return s

    def clip(self, nome: str) -> str:
        return (f'<clipPath id="{nome}"><rect x="{self.x}" y="{self.y}" '
                f'width="{self.larg}" height="{self.alt}" rx="4"/></clipPath>')

    def posa_retta(self, w: tuple[float, float], b: float) -> tuple[float, float, float]:
        """(tx, ty, gradi) per disegnare w·x + b = 0 come segmento ruotato.

        Il punto scelto è quello della retta più vicino al centro del riquadro,
        così la trasformazione resta sempre dentro l'inquadratura. La direzione
        nei dati è (-w1, w0); passando all'SVG la y si ribalta, quindi diventa
        (-w1, -w0) — è qui che è facilissimo sbagliare il segno.
        """
        n2 = w[0] ** 2 + w[1] ** 2 or 1e-9
        cx, cy = (self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2
        k = (w[0] * cx + w[1] * cy + b) / n2
        px, py = cx - k * w[0], cy - k * w[1]
        return self.sx(px), self.sy(py), math.degrees(math.atan2(-w[0], -w[1]))

    def separa(self, posa: tuple[float, float, float], punti) -> bool:
        """La retta *disegnata* separa davvero le classi?

        Non si fida dei pesi: ricostruisce la normale dalla posa che finisce
        nell'SVG e prova tutti i punti. Serve a intercettare gli errori di
        segno, che a occhio non si vedono e in stampa restano.
        """
        px, py, ang = posa
        a = math.radians(ang)
        nx, ny = -math.sin(a), math.cos(a)
        lati = {(t, ((self.sx(x) - px) * nx + (self.sy(y) - py) * ny) > 0)
                for (x, y), t in punti}
        return len(lati) == 2


# --------------------------------------------------------------------------
# Animazione
# --------------------------------------------------------------------------
def keyframes(nome: str, tappe: list[tuple[float, str]]) -> str:
    """`@keyframes` da una lista di (percentuale, dichiarazioni)."""
    corpo = "".join(f"{p:.2f}%{{{d}}}" for p, d in tappe)
    return f"@keyframes {nome}{{{corpo}}}"


def sosta(i: int, n: int, tenuta: float = 0.55) -> tuple[float, float]:
    """Fetta di timeline dello stato i-esimo: (inizio, fine della sosta).

    Ogni stato occupa 1/n della durata: prima resta fermo (perché si legga),
    poi transita verso il successivo.
    """
    passo = 100.0 / n
    return i * passo, i * passo + passo * tenuta


@dataclass
class Figura:
    """Una figura animata pronta da scrivere."""

    larghezza: float
    altezza: float
    alt: str                       # descrizione per l'accessibilità
    corpo: str                     # gli elementi SVG
    stile: str = ""                # regole CSS aggiuntive
    animazioni: list[str] = field(default_factory=list)   # i @keyframes
    durata: float = 8.0
    fermi: str = ""                # cosa fermare con prefers-reduced-motion

    def testo(self, nome: str, titolo: str) -> str:
        stop = self.fermi or "*"
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<!--
  paithon book — figura animata: {titolo}
  Generata da animazioni/svg/{nome}.py — non modificare a mano.
  Palette-locked, fondo trasparente, animazione in CSS puro (nessuno script).
  Lo stato di riposo è quello finale: chi non anima (stampa, PDF,
  prefers-reduced-motion) vede la figura conclusa.
-->
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.larghezza:.0f} {self.altezza:.0f}"
     role="img" aria-label="{self.alt}">
  <style>
    svg  {{ --d: {self.durata:.1f}s; }}
    .ax  {{ stroke:{BORDER_STRONG}; stroke-width:2; fill:none; }}
    .axc {{ stroke:{BORDER}; stroke-width:1.5; }}
    .lbl {{ font-family:{SANS}; font-size:15px; fill:{INK}; }}
    .lbs {{ font-family:{SANS}; font-size:13px; fill:{FG_MUTED}; }}
    .ttl {{ font-family:{SERIF}; font-size:16px; font-style:italic; fill:{INK}; }}
{self.stile}
    {chr(10).join("    " + a for a in self.animazioni)}
    @media (prefers-reduced-motion:reduce) {{ {stop} {{ animation:none !important; }} }}
  </style>
  {self.corpo}
</svg>
"""


def scrivi(nome: str, titolo: str, fig: Figura, provino: bool = True) -> Path:
    """Scrive la figura in book/figures/ dopo averla verificata.

    Tre controlli, tutti e tre già serviti almeno una volta: XML ben formato,
    nessun colore fuori palette, nessuno script. E poi rasterizza lo stato di
    riposo: è il modo di *vedere* cosa finirà in stampa.
    """
    testo = fig.testo(nome, titolo)

    fuori = {c.upper() for c in re.findall(r"#[0-9A-Fa-f]{6}", testo)} - AMMESSI
    if fuori:
        raise ValueError(f"{nome}: colori fuori palette: {sorted(fuori)}")
    if "<script" in testo:
        raise ValueError(f"{nome}: niente script negli SVG del libro")

    import xml.dom.minidom as minidom
    minidom.parseString(testo)          # solleva se l'XML è malformato

    FIGURE.mkdir(parents=True, exist_ok=True)
    out = FIGURE / f"{nome}.svg"
    out.write_text(testo, encoding="utf-8")

    if provino:
        try:
            import cairosvg
            PROVINI.mkdir(parents=True, exist_ok=True)
            cairosvg.svg2png(url=str(out), write_to=str(PROVINI / f"{nome}.png"),
                             output_width=860, background_color=CREAM)
        except ImportError:
            pass
    return out
