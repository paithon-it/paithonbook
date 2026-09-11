"""Come il picco della goccia si ricompra il controllo sull'ampiezza.

`GAN/applicazioni-evoluzioni.md` racconta il trucco del contrabbandiere: il
falsario fabbrica in un punto qualunque un picco enorme, la taratura divide
per quell'ampiezza gonfiata, e «tutto il resto della corsia esce schiacciato
della quantità che serviva». È il passaggio in cui il lettore si ferma, perché
la frase dice che cosa succede e non fa vedere *perché* così l'informazione
che la taratura toglieva torni dentro: bisogna tenere a mente due deviazioni
standard e il loro rapporto.

Disegnato, il fatto è immediato: due corsie identiche tranne una casella, e
sotto quello che ne esce dopo la taratura. Senza il picco la corsia esce
piena; con il picco esce schiacciata, e il fattore di schiacciamento è il
rapporto fra le due ampiezze, cioè esattamente il numero che il falsario
sceglie decidendo quanto alto farlo.

Quello che qui non si disegna, e resta nel testo, è che cosa il falsario ci
faccia poi con quel controllo: la figura mostra il meccanismo, non il movente.

I numeri li calcola la scena da CORSIA e PICCO, e `verifica()` pretende
quelli della didascalia: il picco alto quanto dichiarato, il fattore di
schiacciamento, e che la corsia schiacciata stia davvero sotto quella piena
casella per casella.

Ferma: non è una progressione nel tempo, sono due colonne da confrontare.
"""

import statistics

from paithon_svg import *

NOME = "il-picco-che-schiaccia"
TITOLO = "il picco che si ricompra l'ampiezza"

# --------------------------------------------------------------------------
# I parametri della scena: da questi discende ogni numero che si legge
# --------------------------------------------------------------------------
# una corsia, cioè i valori che un canale porta nelle sue caselle
CORSIA = [0.6, -0.4, 0.9, -0.7, 0.3, -0.2, 0.8, -0.9,
          0.5, -0.6, 0.2, -0.3, 0.7, -0.8, 0.4, -0.5]
DOVE = 8            # in quale casella il falsario mette il picco
PICCO = 8.0         # quanto lo fa alto


def cifre(x: float, decimali: int = 2) -> str:
    """Il numero come si scrive in italiano: virgola, non punto."""
    return f"{x:.{decimali}f}".replace(".", ",")


def tarata(valori: list[float]) -> list[float]:
    """Il primo tempo di AdaIN: media a zero, ampiezza a uno."""
    media = statistics.fmean(valori)
    ampiezza = statistics.pstdev(valori)
    return [(v - media) / ampiezza for v in valori]


CON_PICCO = list(CORSIA)
CON_PICCO[DOVE] = PICCO

AMPIEZZA_PULITA = statistics.pstdev(CORSIA)
AMPIEZZA_GONFIA = statistics.pstdev(CON_PICCO)
SCHIACCIAMENTO = AMPIEZZA_GONFIA / AMPIEZZA_PULITA

USCITA_PULITA = tarata(CORSIA)
USCITA_GONFIA = tarata(CON_PICCO)

# --------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------
LARG, ALT = 880, 372
COLONNA, VUOTO, X0 = 400, 50, 20
PASSO = COLONNA / len(CORSIA)
BARRA = PASSO - 5
Y_TIT, Y_SU, Y_GIU = 26, 112, 252     # i titoli, e le due mezzerie
MEZZA = 44                            # quanto può salire o scendere una barra
TAGLIA = 3.2                          # oltre questa la barretta si ferma
SCALA = MEZZA / TAGLIA                # quanto vale un'unità sulla mezzeria
Y_NOTA = ALT - 16


def fila(x0: float, y: float, valori: list[float], evidenzia: int | None,
         taglia: float) -> tuple[str, list[float]]:
    """Una fila di barre attorno alla mezzeria y. Torna anche le altezze."""
    p = [f'<line class="zr" x1="{x0 - 6:.0f}" y1="{y:.0f}" '
         f'x2="{x0 + COLONNA + 6:.0f}" y2="{y:.0f}"/>']
    altezze = []
    for i, v in enumerate(valori):
        h = min(abs(v), taglia) * SCALA
        altezze.append(h)
        cima = y - h if v >= 0 else y
        classe = "pk" if i == evidenzia else "br"
        p.append(f'<rect class="{classe}" x="{x0 + i * PASSO + 2.5:.1f}" '
                 f'y="{cima:.1f}" width="{BARRA:.1f}" height="{h:.1f}"/>')
        if abs(v) > taglia and i == evidenzia:
            # il picco esce dalla scala, non dalla tela: la barretta si ferma
            # al massimo che la mezzeria consente, e il valore vero si scrive
            # sopra, o il disegno direbbe che il picco vale tre e due
            p.append(f'<text class="ls" x="{x0 + i * PASSO + PASSO / 2:.1f}" '
                     f'y="{cima - 5:.0f}" text-anchor="middle">'
                     f'{cifre(v, 1)}</text>')
    return "".join(p), altezze


def colonna(x0: float, titolo: str, dentro: list[float], fuori: list[float],
            evidenzia: int | None, sotto: str) -> tuple[str, list[float]]:
    p = [f'<text class="ttl" x="{x0 + COLONNA / 2:.0f}" y="{Y_TIT}" '
         f'text-anchor="middle">{titolo}</text>']
    su, _ = fila(x0, Y_SU, dentro, evidenzia, TAGLIA)
    giu, altezze = fila(x0, Y_GIU, fuori, evidenzia, TAGLIA)
    p += [su, giu]
    p.append(f'<text class="lb" x="{x0:.0f}" y="{Y_SU + 78:.0f}">'
             f'la taratura divide per {sotto}</text>')
    p.append(f'<path class="fr" d="M {x0 + COLONNA * 0.82:.0f} {Y_SU + 86} '
             f'l 0 26 m -6 -8 l 6 8 l 6 -8"/>')
    return "".join(p), altezze


def costruisci() -> Figura:
    sinistra, alte_pulite = colonna(
        X0, "la corsia com'è", CORSIA, USCITA_PULITA, None,
        cifre(AMPIEZZA_PULITA))
    destra, alte_gonfie = colonna(
        X0 + COLONNA + VUOTO, "la stessa corsia, più il picco",
        CON_PICCO, USCITA_GONFIA, DOVE, cifre(AMPIEZZA_GONFIA))

    pulita, gonfia = cifre(AMPIEZZA_PULITA), cifre(AMPIEZZA_GONFIA)
    nota = (
        f'<text class="lb" x="{LARG / 2:.0f}" y="{Y_NOTA - 20}" '
        f'text-anchor="middle">Un picco alto {PICCO:.0f} decide da solo di '
        f'quanto si divide: l\'ampiezza passa da {pulita} a {gonfia}.</text>'
        f'<text class="lb" x="{LARG / 2:.0f}" y="{Y_NOTA}" '
        f'text-anchor="middle">Tutto il resto della corsia esce '
        f'{cifre(SCHIACCIAMENTO, 1)} volte più piccolo, ed è quel numero che '
        f'il falsario si è ricomprato.</text>')

    verifica(alte_pulite, alte_gonfie)

    return Figura(
        larghezza=LARG,
        altezza=ALT,
        alt="Due colonne a confronto. A sinistra, in alto, una corsia di "
            "sedici caselle disegnate come barrette sopra e sotto una "
            "mezzeria; in basso la stessa corsia dopo la taratura, che divide "
            "per quanto la corsia oscilla, e le barrette restano ben "
            "visibili. A destra la stessa corsia con una sola casella "
            "cambiata, un picco in terracotta molto più alto di tutte le "
            "altre barrette, con il suo valore, 8,0, scritto sopra perché la "
            "scala del disegno non arriva fin lì. Dopo la taratura, che ora "
            "divide per un numero molto più grande, tutte le altre barrette "
            "escono schiacciate, di poco più di tre volte, mentre il picco "
            "resta altissimo, col suo nuovo valore, 3,7, scritto sopra. "
            "In fondo il conto: l'ampiezza passa da 0,60 a 2,03 e il resto "
            "della corsia esce 3,4 volte più piccolo.",
        corpo=sinistra + destra + nota,
        stile=f"""    .br  {{ fill:{TEAL}; fill-opacity:0.30; stroke:{TEAL};
            stroke-width:1; }}
    .pk  {{ fill:{TERRACOTTA}; fill-opacity:0.28; stroke:{TERRACOTTA};
            stroke-width:1.4; }}
    .zr  {{ stroke:{BORDER_STRONG}; stroke-width:1.2; }}
    .fr  {{ stroke:{FG_MUTED}; stroke-width:1.4; fill:none; }}
    .lb  {{ font-family:{SANS}; font-size:12.5px; fill:{INK}; }}
    .ls  {{ font-family:{SANS}; font-size:11.5px; fill:{FG_MUTED}; }}""",
    )


def verifica(alte_pulite: list[float], alte_gonfie: list[float]) -> None:
    """Difende quello che la didascalia promette, numero per numero."""
    assert len(CORSIA) == 16, len(CORSIA)
    assert CON_PICCO[DOVE] == PICCO, CON_PICCO[DOVE]
    # il picco è l'unica casella cambiata: le due corsie sono identiche altrove
    diverse = [i for i, (a, b) in enumerate(zip(CORSIA, CON_PICCO)) if a != b]
    assert diverse == [DOVE], diverse
    # i due numeri stampati in fondo
    assert f"{AMPIEZZA_PULITA:.2f}" == "0.60", AMPIEZZA_PULITA
    assert f"{AMPIEZZA_GONFIA:.2f}" == "2.03", AMPIEZZA_GONFIA
    assert f"{SCHIACCIAMENTO:.1f}" == "3.4", SCHIACCIAMENTO
    # e il fatto per cui la figura esiste: dopo la taratura, ogni casella
    # della corsia col picco esce più bassa della sua gemella. Senza questo
    # il disegno potrebbe mostrare lo schiacciamento solo dove fa comodo
    for i in range(len(CORSIA)):
        if i == DOVE:
            continue
        assert abs(USCITA_GONFIA[i]) < abs(USCITA_PULITA[i]), i
        assert alte_gonfie[i] < alte_pulite[i], i
    # lo schiacciamento si deve vedere: almeno il doppio, o è un disegno muto
    assert SCHIACCIAMENTO > 2, SCHIACCIAMENTO
    # il picco esce dalla scala in tutt'e due le file, quindi la sua barretta
    # si ferma al massimo e il valore gli va scritto sopra: è quello che
    # l'`alt` descrive, e senza queste due righe può smettere di essere vero
    assert PICCO > TAGLIA and abs(USCITA_GONFIA[DOVE]) > TAGLIA, PICCO
    # e i due numeri scritti sopra la barretta sono quelli, non i loro
    # arrotondamenti all'intero: con una cifra sola il picco tarato direbbe
    # «4», che è 8 diviso l'ampiezza, cioè il conto senza la sottrazione
    # della media, ed è proprio il passaggio che la figura esiste per mostrare
    assert cifre(PICCO, 1) == "8,0", PICCO
    assert cifre(USCITA_GONFIA[DOVE], 1) == "3,7", USCITA_GONFIA[DOVE]
    assert alte_gonfie[DOVE] == MEZZA, alte_gonfie[DOVE]
    # e dalla tela non esce: la cima della barretta resta sotto il titolo
    assert Y_SU - MEZZA > Y_TIT, (Y_SU - MEZZA, Y_TIT)
    # e tutto deve stare nella tela
    assert X0 + 2 * COLONNA + VUOTO <= LARG, LARG
    assert Y_GIU + MEZZA < Y_NOTA - 34, (Y_GIU, Y_NOTA)
