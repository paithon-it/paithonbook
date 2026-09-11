"""Che cosa un episodio rinforza: la sola ultima mossa, o tutta la scia.

`ReinforcementLearning/q-learning.md` promette due volte una cosa da guardare
(«il primo episodio che tocca la meta non illumina solo l'ultima casella,
illumina tutta la strada percorsa, in dissolvenza») e non gliela dava. Questa
figura e' quella cosa, ed e' la stessa che Sutton e Barto mettono in fondo al
capitolo sui metodi a n passi: lo stesso cammino, e a fianco quali mosse ne
escono rinforzate con l'uno e con l'altro metodo.

Non e' una misura, e' il meccanismo: il cammino disegnato e' il piu' corto che
porta dalla partenza alla meta nel labirinto della pagina, cinque mosse. Con
la tabella tutta a zero, il bersaglio a un passo e' zero dappertutto tranne
sull'ultima mossa, che incassa il premio: quindi si muove quella e nient'altro.
Con le tracce si muovono tutte, e il peso di ciascuna e' la sua traccia al
momento della sorpresa, cioe' $(\\gamma\\lambda)^k$ con $k$ i passi che la
separano dalla fine. I numeri sotto le frecce sono quelli, calcolati qui.

E' una figura ferma di proposito: qui il tempo non e' il contenuto, il
confronto si legge tutto in una volta, e la sezione ha gia' la sua clip.
"""

from paithon_svg import *

NOME = "tracce-illuminano"
TITOLO = "un passo alla volta contro le tracce: chi si rinforza"

RIGHE, COLONNE = 3, 4
MURO, META, TRAPPOLA = (1, 1), (0, 3), (1, 3)
PARTENZA = (2, 0)
GAMMA, LAMBDA = 0.9, 0.9

# il cammino piu' corto dalla partenza alla meta, aggirando muro e trappola
CAMMINO = [(2, 0), (1, 0), (0, 0), (0, 1), (0, 2), (0, 3)]


def pesi():
    """Il peso con cui ciascuna mossa del cammino viene rinforzata.

    A un passo: solo l'ultima, che e' l'unica il cui bersaglio non e' zero.
    Con le tracce: tutte, e la k-esima dalla fine vale $(\\gamma\\lambda)^k$.
    """
    n = len(CAMMINO) - 1
    uno = [0.0] * (n - 1) + [1.0]
    scia = [(GAMMA * LAMBDA) ** (n - 1 - k) for k in range(n)]
    return uno, scia


def verifica(uno, scia):
    """Difende quello che la didascalia promette.

    Che il cammino sia davvero un cammino (mosse di una casella, mai dentro il
    muro o la trappola) e che finisca sulla meta; che a un passo si muova una
    mossa sola; che le tracce sfumino di $\\gamma\\lambda$ esatto a ogni passo
    indietro e non arrivino mai a zero, perche' il punto della scena e' che
    nessuna mossa resta al buio.
    """
    assert CAMMINO[0] == PARTENZA, "il cammino non parte dalla partenza"
    assert CAMMINO[-1] == META, "il cammino non arriva alla meta"
    for prima, dopo in zip(CAMMINO, CAMMINO[1:]):
        passo = abs(prima[0] - dopo[0]) + abs(prima[1] - dopo[1])
        assert passo == 1, f"da {prima} a {dopo} non e' una mossa sola"
        assert dopo not in (MURO, TRAPPOLA), f"il cammino passa da {dopo}"
    assert len(CAMMINO) - 1 == 5, "il cammino piu' corto e' di cinque mosse"

    assert sum(1 for v in uno if v > 0) == 1, "a un passo si muove una sola mossa"
    assert uno[-1] == 1.0, "a un passo si muove l'ultima"

    assert scia[-1] == 1.0, "l'ultima mossa e' rinforzata per intero"
    for prima, dopo in zip(scia, scia[1:]):
        assert abs(prima - dopo * GAMMA * LAMBDA) < 1e-12, (
            "la traccia non sfuma esattamente di gamma per lambda")
    assert min(scia) > 0.4, f"la mossa piu' vecchia scende a {min(scia):.3f}"
    return min(scia)


def it(v: float) -> str:
    return f"{v:.2f}".replace(".", ",")


def pannello(x0, y0, lato, titolo, peso, prefisso) -> str:
    fuori = [f'<text class="pan" x="{x0 + COLONNE * lato / 2:.1f}" '
             f'y="{y0 - 16:.1f}" text-anchor="middle">{titolo}</text>']

    for i in range(RIGHE):
        for j in range(COLONNE):
            cx, cy = x0 + j * lato, y0 + i * lato
            classe = ("mur" if (i, j) == MURO else
                      "prem" if (i, j) == META else
                      "pena" if (i, j) == TRAPPOLA else "cel")
            fuori.append(f'<rect class="{classe}" x="{cx:.1f}" y="{cy:.1f}" '
                         f'width="{lato:.1f}" height="{lato:.1f}"/>')
    for cella, testo in ((META, "+1"), (TRAPPOLA, "−1"), (PARTENZA, "S")):
        cx = x0 + cella[1] * lato + lato / 2
        cy = y0 + cella[0] * lato + lato / 2
        classe = "seg" if cella == PARTENZA else "trg"
        fuori.append(f'<text class="{classe}" x="{cx:.1f}" y="{cy + 6:.1f}" '
                     f'text-anchor="middle">{testo}</text>')

    for k, (prima, dopo) in enumerate(zip(CAMMINO, CAMMINO[1:])):
        x1 = x0 + prima[1] * lato + lato / 2
        y1 = y0 + prima[0] * lato + lato / 2
        x2 = x0 + dopo[1] * lato + lato / 2
        y2 = y0 + dopo[0] * lato + lato / 2
        # la freccia si ferma prima del centro della casella d'arrivo, cosi'
        # non copre quello che c'e' scritto dentro
        dx, dy = x2 - x1, y2 - y1
        norma = (dx * dx + dy * dy) ** 0.5
        ax1, ay1 = x1 + dx / norma * 16, y1 + dy / norma * 16
        ax2, ay2 = x2 - dx / norma * 20, y2 - dy / norma * 20
        p = peso[k]
        if p > 0:
            fuori.append(f'<line class="acc" x1="{ax1:.1f}" y1="{ay1:.1f}" '
                         f'x2="{ax2:.1f}" y2="{ay2:.1f}" '
                         f'stroke-opacity="{0.25 + 0.75 * p:.3f}" '
                         f'marker-end="url(#{prefisso}p)"/>')
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            # sopra le mosse orizzontali, di fianco a quelle verticali: sulla
            # linea il numero finiva addosso alla freccia, e l'ultimo addosso
            # al «+1» della meta
            if abs(dy) < 1e-9:
                fuori.append(f'<text class="pes" x="{mx:.1f}" y="{my - 13:.1f}" '
                             f'text-anchor="middle">{it(p)}</text>')
            else:
                fuori.append(f'<text class="pes" x="{mx + 14:.1f}" '
                             f'y="{my + 4:.1f}">{it(p)}</text>')
        else:
            fuori.append(f'<line class="spe" x1="{ax1:.1f}" y1="{ay1:.1f}" '
                         f'x2="{ax2:.1f}" y2="{ay2:.1f}" '
                         f'marker-end="url(#{prefisso}s)"/>')
    return "".join(fuori)


def punta(nome: str, colore: str, opacita: float) -> str:
    return (f'<marker id="{nome}" viewBox="0 0 10 10" refX="8.5" refY="5" '
            f'markerWidth="5.5" markerHeight="5.5" orient="auto-start-reverse">'
            f'<path d="M 0 1 L 9 5 L 0 9 z" fill="{colore}" '
            f'fill-opacity="{opacita}"/></marker>')


def costruisci() -> Figura:
    uno, scia = pesi()
    piu_vecchia = verifica(uno, scia)

    lato = 62.0
    y0 = 76.0
    sinistra, destra = 40.0, 40.0 + COLONNE * lato + 74.0

    corpo = ["<defs>",
             punta("ap", TERRACOTTA, 1.0), punta("as", BORDER_STRONG, 1.0),
             punta("bp", TERRACOTTA, 1.0), punta("bs", BORDER_STRONG, 1.0),
             "</defs>"]
    corpo.append(pannello(sinistra, y0, lato, "un passo alla volta", uno, "a"))
    corpo.append(pannello(destra, y0, lato, "con le tracce", scia, "b"))

    corpo.append(f'<text class="lbs" x="{sinistra:.1f}" '
                 f'y="{y0 + RIGHE * lato + 28:.1f}">'
                 f'lo stesso episodio, con la tabella dei voti tutta a zero: '
                 f'accanto a ogni mossa rinforzata, di quanto</text>')

    return Figura(
        larghezza=720, altezza=330,
        alt="Due griglie identiche di tre righe per quattro colonne, con la "
            "partenza in basso a sinistra, la meta che paga più uno in alto a "
            "destra, la trappola sotto di lei e un muro al centro. In tutte e "
            "due, cinque frecce disegnano lo stesso cammino dalla partenza "
            "alla meta. Nella griglia di sinistra, «un passo alla volta», una "
            "sola freccia è colorata, l'ultima, quella che entra nella meta, "
            "con peso 1,00; le altre quattro sono grigie. Nella griglia di "
            "destra, «con le tracce», sono colorate tutte e cinque, e il loro "
            f"peso sfuma all'indietro: {it(scia[-1])}, {it(scia[-2])}, "
            f"{it(scia[-3])}, {it(scia[-4])} e {it(scia[-5])} per la prima "
            "mossa del cammino, che resta comunque accesa.",
        corpo="".join(corpo),
        stile=f"""    .cel  {{ fill:none; stroke:{BORDER_STRONG}; stroke-width:1.4; }}
    .mur  {{ fill:{BORDER_STRONG}; stroke:{BORDER_STRONG}; stroke-width:1.4; }}
    .prem {{ fill:{OCRA}; fill-opacity:0.45; stroke:{OCRA}; stroke-width:1.8; }}
    .pena {{ fill:{TERRACOTTA}; fill-opacity:0.28; stroke:{TERRACOTTA};
            stroke-width:1.8; }}
    .trg  {{ font-family:{SANS}; font-size:15px; font-weight:700; fill:{INK}; }}
    .seg  {{ font-family:{SANS}; font-size:14px; font-weight:700;
            fill:{FG_MUTED}; }}
    .acc  {{ stroke:{TERRACOTTA}; stroke-width:3; stroke-linecap:round; }}
    .spe  {{ stroke:{BORDER_STRONG}; stroke-width:1.6; stroke-linecap:round; }}
    .pes  {{ font-family:{SANS}; font-size:12px; font-weight:700;
            fill:{TERRACOTTA}; }}
    .pan  {{ font-family:{SANS}; font-size:16px; font-weight:700;
            fill:{INK}; }}""",
    )
