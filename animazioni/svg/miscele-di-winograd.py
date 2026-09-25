"""L'algoritmo di Winograd F(2,3): due uscite di un filtro da tre pesi con
quattro moltiplicazioni invece di sei.

È la scena di `DeepLearning/reti-convoluzionali.md`, sezione «Lo stesso conto
con meno moltiplicazioni»: la striscia 1, 2, 3, 4 e lo stampino 1, 2, 3. Il
conto va in quattro tempi, ed è per questo che è animato: prima striscia e
stampino, poi le quattro miscele di ciascuno, poi i quattro prodotti, infine
la ricomposizione nelle due uscite.

Tutti i numeri li calcola il generatore con le matrici B^T, G e A^T della
pagina, e gli assert controllano quello che la didascalia promette: che le due
uscite siano quelle del conto diretto (14 e 20), che le moltiplicazioni siano
quattro contro sei, e che le miscele siano quelle scritte nella scheda
Elementare. Il disegno fermo è lo stato finale, con tutti e quattro i tempi
visibili.
"""

from fractions import Fraction as Fr

from paithon_svg import *

NOME = "miscele-di-winograd"
TITOLO = "l'algoritmo di Winograd F(2,3), quattro moltiplicazioni invece di sei"

STRISCIA = [1, 2, 3, 4]
STAMPINO = [1, 2, 3]
BT = [[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]]
G = [[1, 0, 0], [Fr(1, 2), Fr(1, 2), Fr(1, 2)], [Fr(1, 2), Fr(-1, 2), Fr(1, 2)], [0, 0, 1]]
AT = [[1, 1, 1, 0], [0, 1, -1, -1]]

COLONNE = [150, 300, 450, 600]     # le quattro miscele
Y_IN, Y_MIX, Y_PROD, Y_OUT = 62, 170, 272, 350
LATO = 40


def per(M, v):
    return [sum(Fr(a) * b for a, b in zip(riga, v)) for riga in M]


def num(x) -> str:
    """Un numero come lo si scrive: intero se lo è, meno tipografico."""
    x = Fr(x)
    s = str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"
    return s.replace("-", "−")


def conti():
    mix_d = per(BT, STRISCIA)
    mix_g = per(G, STAMPINO)
    prodotti = [a * b for a, b in zip(mix_d, mix_g)]
    uscite = per(AT, prodotti)
    diretta = [sum(STRISCIA[i + k] * STAMPINO[k] for k in range(3)) for i in range(2)]
    return mix_d, mix_g, prodotti, uscite, diretta


def costruisci() -> Figura:
    mix_d, mix_g, prodotti, uscite, diretta = conti()
    assert uscite == diretta == [14, 20], f"uscite {uscite}, conto diretto {diretta}"
    assert mix_d == [-2, 5, 1, -2] and mix_g == [1, 3, 1, 3], "le miscele non sono quelle della scheda"
    m, r = len(uscite), len(STAMPINO)
    assert len(prodotti) == m + r - 1 == 4 and m * r == 6, "quattro moltiplicazioni contro sei"

    n = 4
    corpo, anim = [], []

    def tempo(k):
        """Visibile dal tempo k in poi; a riposo visibile."""
        if k == 0:
            return ""
        t0, _ = sosta(k, n)
        nome = f"t{k}"
        if not any(a.startswith(f"@keyframes {nome}{{") for a in anim):
            anim.append(keyframes(nome, [(0.0, "opacity:0"), (t0 - 1.0, "opacity:0"),
                                         (t0 + 2.0, "opacity:1"), (100.0, "opacity:1")]))
        return f' style="animation:{nome} var(--d) infinite"'

    def casella(x, y, testo, classe, larg=LATO):
        return (f'<rect class="{classe}" x="{x - larg / 2:.1f}" y="{y - LATO / 2:.1f}" '
                f'width="{larg}" height="{LATO}" rx="4"/>'
                f'<text class="val" x="{x:.1f}" y="{y + 7:.1f}" text-anchor="middle">{testo}</text>')

    # primo tempo: striscia e stampino
    g = [f'<text class="lbl" x="40" y="{Y_IN + 6}">striscia</text>']
    for i, v in enumerate(STRISCIA):
        g.append(casella(130 + i * 44, Y_IN, num(v), "cd"))
    g.append(f'<text class="lbl" x="420" y="{Y_IN + 6}">stampino</text>')
    for i, v in enumerate(STAMPINO):
        g.append(casella(530 + i * 44, Y_IN, num(v), "cg"))
    corpo.append("<g>" + "".join(g) + "</g>")

    # secondo tempo: le miscele
    etich_d = [f"{num(STRISCIA[0])}−{num(STRISCIA[2])}", f"{num(STRISCIA[1])}+{num(STRISCIA[2])}",
               f"{num(STRISCIA[2])}−{num(STRISCIA[1])}", f"{num(STRISCIA[1])}−{num(STRISCIA[3])}"]
    g0, g1, g2 = (num(v) for v in STAMPINO)
    etich_g = [g0, f"({g0}+{g1}+{g2})/2", f"({g0}\u2212{g1}+{g2})/2", g2]
    g = [f'<text class="lbs" x="40" y="{Y_MIX + 5}">miscele</text>']
    for k, x in enumerate(COLONNE):
        g.append(f'<text class="lbs" x="{x - 26}" y="{Y_MIX - 28}" text-anchor="middle">{etich_d[k]}</text>')
        g.append(f'<text class="lbs" x="{x + 26}" y="{Y_MIX + 38}" text-anchor="middle">{etich_g[k]}</text>')
        g.append(casella(x - 26, Y_MIX, num(mix_d[k]), "cd"))
        g.append(casella(x + 26, Y_MIX, num(mix_g[k]), "cg"))
    corpo.append(f'<g{tempo(1)}>' + "".join(g) + "</g>")

    # terzo tempo: i quattro prodotti, le sole moltiplicazioni
    g = [f'<text class="lbs" x="40" y="{Y_PROD + 5}">prodotti</text>']
    for k, x in enumerate(COLONNE):
        g.append(f'<line class="filo" x1="{x}" y1="{Y_MIX + LATO / 2 + 22}" x2="{x}" y2="{Y_PROD - 22}"/>')
        g.append(f'<text class="per" x="{x}" y="{Y_MIX + 6}" text-anchor="middle">×</text>')
        g.append(f'<circle class="prod" cx="{x}" cy="{Y_PROD}" r="21"/>')
        g.append(f'<text class="val" x="{x}" y="{Y_PROD + 7}" text-anchor="middle">{num(prodotti[k])}</text>')
    g.append(f'<text class="cont" x="{COLONNE[-1] + 60}" y="{Y_PROD + 6}">4 moltiplicazioni</text>')
    g.append(f'<text class="lbs" x="{COLONNE[-1] + 60}" y="{Y_PROD + 24}">il conto diretto ne fa 6</text>')
    corpo.append(f'<g{tempo(2)}>' + "".join(g) + "</g>")

    # quarto tempo: la ricomposizione, con sole somme e differenze
    segni = [[("+", 0), ("+", 1), ("+", 2)], [("+", 1), ("−", 2), ("−", 3)]]
    xo = [COLONNE[0] - 30, COLONNE[2] - 30]           # testi allineati a sinistra
    g = [f'<text class="lbs" x="40" y="{Y_OUT + 5}">uscite</text>']
    for o in range(2):
        pezzi = []
        for s, k in segni[o]:
            v = prodotti[k]
            pezzi.append((("" if not pezzi and s == "+" else f" {s} ")) + (f"({num(v)})" if v < 0 else num(v)))
        g.append(f'<text class="espr" x="{xo[o]:.1f}" y="{Y_OUT + 6}">'
                 f'{"".join(pezzi)} = <tspan class="usc">{num(uscite[o])}</tspan></text>')
    corpo.append(f'<g{tempo(3)}>' + "".join(g) + "</g>")

    return Figura(
        larghezza=820, altezza=380,
        alt="Animazione in quattro tempi. In alto una striscia di quattro caselle "
            "teal con 1, 2, 3, 4 e uno stampino di tre caselle terracotta con 1, 2, "
            "3. Poi compaiono quattro coppie di caselle: le miscele della striscia, "
            "meno 2, 5, 1 e meno 2, fatte di sole somme e differenze, accanto alle "
            "miscele dello stampino, 1, 3, 1 e 3. Poi, sotto ogni coppia, un cerchio "
            "ocra con il prodotto: meno 2, 15, 1 e meno 6, e la scritta quattro "
            "moltiplicazioni, il conto diretto ne fa sei. Infine le due uscite: meno "
            "2 più 15 più 1 fa 14, e 15 meno 1 meno meno 6 fa 20.",
        corpo="".join(corpo),
        stile=f"""    .cd   {{ fill:none; stroke:{TEAL}; stroke-width:2; }}
    .cg   {{ fill:none; stroke:{TERRACOTTA}; stroke-width:2; }}
    .val  {{ font-family:{SANS}; font-size:19px; font-weight:700; fill:{INK}; }}
    .prod {{ fill:none; stroke:{OCRA}; stroke-width:2.6; }}
    .filo {{ stroke:{BORDER_STRONG}; stroke-width:1.2; }}
    .per  {{ font-family:{SANS}; font-size:18px; fill:{FG_MUTED}; }}
    .cont {{ font-family:{SANS}; font-size:15px; font-weight:700; fill:{OCRA}; }}
    .espr {{ font-family:{SANS}; font-size:17px; fill:{INK}; }}
    .usc  {{ font-weight:700; fill:{TERRACOTTA}; }}""",
        animazioni=anim,
        durata=n * 2.2,
        fermi="g",
    )
