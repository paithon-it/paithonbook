"""Il torneo a eliminazione: cinque turni, e ogni turno costa quanto gli altri.

`MachineLearning/iperparametri.md` lo dice con una fila di prodotti
($81 \\times 1$, $27 \\times 3$, $9 \\times 9$, $3 \\times 27$, $1 \\times 81$) e
poi tira la somma. È il passaggio in cui un lettore senza carta e penna si
ferma: i due numeri si muovono insieme in versi opposti, e che il prodotto
resti lo stesso non si vede finché non lo si fa cinque volte.

Disegnato, il fatto è immediato: cinque righe lunghe uguali. La prima è pettinata
in ottantuno celle sottili, l'ultima è un blocco solo, e in mezzo le celle si
allargano mentre diventano meno. La riga che resta lunga uguale **è** la
frase «ogni turno costa quanto gli altri».

Quello che qui non si disegna, e resta nel testo, è il confronto con la
valutazione completa: 6.561 epoche contro 405 sono sedici volte, e a disegnarle
nella stessa unità la figura sarebbe alta due metri. Un secondo riquadro con
un'altra scala direbbe una bugia sulla proporzione, quindi il conto resta dov'è,
scritto per esteso.

I numeri li calcola la scena da CANDIDATE e FATTORE, e `verifica()` pretende
quelli della didascalia: le cinque coppie, il prodotto costante, i 405 in
totale e le cinque barre larghe uguali.

Ferma: non è una progressione nel tempo da guardare scorrere, è un confronto
fra cinque righe, e si legge tutto in una volta.
"""

from paithon_svg import *

NOME = "ogni-turno-costa-uguale"
TITOLO = "cinque turni, e ogni turno costa quanto gli altri"

# --------------------------------------------------------------------------
# I parametri della scena: da questi discende ogni numero che si legge
# --------------------------------------------------------------------------
CANDIDATE = 81      # quante partono
FATTORE = 3         # a ogni turno ne sopravvive una su tre, e il budget si triplica

TURNI = []          # (candidate rimaste, epoche a testa)
n, r = CANDIDATE, 1
while n >= 1:
    TURNI.append((n, r))
    n, r = n // FATTORE, r * FATTORE
COSTO_TURNO = CANDIDATE                  # n * r, lo stesso a ogni turno
COSTO_TORNEO = COSTO_TURNO * len(TURNI)

# --------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------
LARG, ALT = 880, 348
X_BARRA, BARRA = 236, 512                # dove comincia la fila, e quanto è lunga
Y_PRIMA, PASSO_RIGA, ALTA = 54, 48, 32   # la prima riga, il passo, l'altezza
X_SINISTRA, X_DESTRA = 220, X_BARRA + BARRA + 14
Y_NOTA = Y_PRIMA + len(TURNI) * PASSO_RIGA + 22


def riga(indice: int, rimaste: int, epoche: int) -> tuple[str, float]:
    """Una riga del torneo. Torna gli elementi e la larghezza totale usata."""
    y = Y_PRIMA + indice * PASSO_RIGA
    cella = BARRA / rimaste
    vincitrice = rimaste == 1
    classe = "vn" if vincitrice else "cl"
    p = []
    for i in range(rimaste):
        p.append(f'<rect class="{classe}" x="{X_BARRA + i * cella:.2f}" '
                 f'y="{y:.0f}" width="{cella:.2f}" height="{ALTA}"/>')
    p.append(f'<text class="lb" x="{X_SINISTRA}" y="{y + ALTA / 2 + 4.5:.0f}" '
             f'text-anchor="end">{rimaste} '
             f'{"candidata" if rimaste == 1 else "candidate"}, '
             f'{epoche} {"epoca" if epoche == 1 else "epoche"} a testa</text>')
    p.append(f'<text class="ls" x="{X_DESTRA}" y="{y + ALTA / 2 + 4.5:.0f}">'
             f'{rimaste} × {epoche} = {rimaste * epoche}</text>')
    return "".join(p), rimaste * cella


def costruisci() -> Figura:
    corpi, larghezze = [], []
    for i, (rimaste, epoche) in enumerate(TURNI):
        corpo, larghezza = riga(i, rimaste, epoche)
        corpi.append(corpo)
        larghezze.append(round(larghezza, 6))

    intestazione = (
        f'<text class="ttl" x="{X_BARRA}" y="32">ogni turno costa quanto gli '
        f'altri</text>'
        f'<text class="ls" x="{X_DESTRA}" y="32">epoche spese</text>')
    nota = (f'<text class="ls" x="{X_BARRA}" y="{Y_NOTA}">'
            f'{len(TURNI)} turni da {COSTO_TURNO}: {COSTO_TORNEO} epoche in '
            f'tutto. La riga resta lunga uguale perché le candidate si '
            f'riducono</text>'
            f'<text class="ls" x="{X_BARRA}" y="{Y_NOTA + 17}">'
            f'a un {"terzo" if FATTORE == 3 else FATTORE} mentre le epoche a '
            f'testa si moltiplicano per {FATTORE}.</text>')

    verifica(larghezze)

    return Figura(
        larghezza=LARG,
        altezza=ALT,
        alt="Cinque righe orizzontali lunghe uguali, una per turno del torneo. "
            "La prima è divisa in ottantun celle sottilissime, ed è etichettata "
            "ottantuno candidate con una epoca a testa; la seconda in "
            "ventisette celle tre volte più larghe, con tre epoche a testa; la "
            "terza in nove celle, con nove epoche; la quarta in tre celle, con "
            "ventisette epoche; l'ultima è un blocco solo, in terracotta, con "
            "ottantuno epoche. A destra di ogni riga il prodotto, che vale "
            "ottantuno tutte e cinque le volte. Sotto, la somma: cinque turni "
            "da ottantuno fanno quattrocentocinque epoche in tutto.",
        corpo=intestazione + "".join(corpi) + nota,
        stile=f"""    .cl  {{ fill:{CREAM}; stroke:{TEAL}; stroke-width:1.2; }}
    .vn  {{ fill:{TERRACOTTA}; fill-opacity:0.18; stroke:{TERRACOTTA};
            stroke-width:1.6; }}
    .lb  {{ font-family:{SANS}; font-size:12.5px; fill:{INK}; }}
    .ls  {{ font-family:{SANS}; font-size:11.5px; fill:{FG_MUTED}; }}""",
    )


def verifica(larghezze: list[float]) -> None:
    """Difende quello che la didascalia promette, numero per numero."""
    assert [n for n, _ in TURNI] == [81, 27, 9, 3, 1], TURNI
    assert [r for _, r in TURNI] == [1, 3, 9, 27, 81], TURNI
    # il fatto che la figura esiste per mostrare: il prodotto non cambia mai
    for n, r in TURNI:
        assert n * r == COSTO_TURNO, (n, r, COSTO_TURNO)
    assert COSTO_TORNEO == 405, COSTO_TORNEO
    assert CANDIDATE * CANDIDATE == 6561, CANDIDATE
    # e la sua controparte nel disegno: cinque barre larghe uguali. Senza
    # questa, un arrotondamento delle celle farebbe righe diverse e la figura
    # direbbe il contrario della didascalia.
    assert len(set(larghezze)) == 1, larghezze
    assert abs(larghezze[0] - BARRA) < 1e-6, larghezze[0]
    # la riga più fitta deve restare leggibile: celle sopra i cinque punti
    assert BARRA / CANDIDATE > 5, BARRA / CANDIDATE
    # e tutto deve stare nella tela
    assert Y_NOTA + 17 < ALT, Y_NOTA
