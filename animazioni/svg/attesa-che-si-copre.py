"""L'attesa non si accorcia: la copre il lavoro di chi non sta aspettando.

`GPU/overview.md` lo afferma due volte, una per scheda: «l'attesa del singolo
dura esattamente quanto durava prima, e intanto la sala lavora lo stesso»
nell'Elementare, «non accorcia l'attesa del singolo: la *copre* con il lavoro
degli altri» nella Superiore. E' il guadagno su cui poggia mezzo capitolo (e'
la ragione dei warp che si danno il cambio, dell'occupancy, del rifornire la
GPU) e non aveva nessun disegno.

Quello che il testo chiede al lettore e' di tenere in testa due cose insieme:
che l'attesa di ciascuno resti la stessa, **e** che la macchina non si fermi.
A parole sembra una contraddizione; su una fila di istanti si vede in un
secondo, perche' sono due righe diverse dello stesso disegno.

I numeri li calcola la scena da LAVORO, ATTESA ed ESECUTORI, e `verifica()`
pretende quelli che la didascalia promette: che l'attesa di ciascuna unita'
sia la stessa nei due pannelli (nove istanti su dodici), che la macchina stia
ferma nove istanti con una unita' sola e nessuno con quattro, e che in ogni
istante del secondo pannello conti **esattamente** un'unita'. Quest'ultima e'
la piu' importante: se due unita' contassero nello stesso istante il disegno
mostrerebbe un parallelismo di calcolo, che qui non c'entra, invece della
copertura di un'attesa.

Ferma, e la scelta e' motivata: il tempo qui e' il contenuto, ma e' un tempo
che va letto **tutto insieme**, come una tabella di turni. Una clip mostra un
istante per volta, cioe' meno: il confronto fra le due righe «la macchina
conta» si perderebbe, perche' una delle due si vedrebbe solo a cose fatte.

Le etichette stanno fuori dall'analogia della pagina («unita' di calcolo», non
«i cuochi» ne' «la sala»): la figura sta nella spina dorsale, e la guarda anche
chi tiene l'interruttore su Superiore.
"""

from paithon_svg import *

NOME = "attesa-che-si-copre"
TITOLO = "l'attesa che il cambio copre"

# --------------------------------------------------------------------------
# I parametri della scena: da questi tre discende ogni numero che si legge
# --------------------------------------------------------------------------
LAVORO = 1          # istanti che un'unita' passa a contare
ATTESA = 3          # istanti che passa ad aspettare il dato
ESECUTORI = 4       # quante si danno il cambio nel pannello di sotto
PERIODO = LAVORO + ATTESA
ISTANTI = PERIODO * 3                     # tre giri, cioe' dodici istanti

# Chi conta in quale istante. Nel primo pannello c'e' un'unita' sola; nel
# secondo le stesse unita' sfalsate di un istante l'una dall'altra.
CONTA_SOLA = [t % PERIODO < LAVORO for t in range(ISTANTI)]
CONTA_MOLTE = [[(t - i) % PERIODO < LAVORO for t in range(ISTANTI)]
               for i in range(ESECUTORI)]
MACCHINA_SOLA = CONTA_SOLA
MACCHINA_MOLTE = [any(r[t] for r in CONTA_MOLTE) for t in range(ISTANTI)]

FERMA_SOLA = MACCHINA_SOLA.count(False)
FERMA_MOLTE = MACCHINA_MOLTE.count(False)
ATTESE_SOLA = CONTA_SOLA.count(False)
ATTESE_CIASCUNA = CONTA_MOLTE[0].count(False)

PAROLE = {1: "una", 2: "due", 3: "tre", 4: "quattro"}

# --------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------
LARG, ALT = 880, 420
X0 = 236            # dove comincia la prima casella
CELLA = 44          # passo di una casella, bordo compreso
ALTEZZA = 26        # altezza di una riga di istanti
RIGHE_SOPRA = (50, 92)                     # l'unita' e la macchina
RIGA_TITOLO_1, RIGA_TITOLO_2 = 34, 190
PRIMA_MOLTE = 206                          # la prima delle quattro unita'
PASSO_MOLTE = 32
MACCHINA_SOTTO = PRIMA_MOLTE + ESECUTORI * PASSO_MOLTE + 10
LEGENDA = 408


def verifica() -> None:
    """Difende quello che la didascalia promette, numero per numero."""
    assert (LAVORO, ATTESA, ESECUTORI) == (1, 3, 4), "i parametri sono cambiati"
    assert ISTANTI == 12, ISTANTI
    # l'attesa di ciascuna e' la stessa nei due pannelli: e' la meta' della
    # tesi, e senza di lei la figura direbbe che il cambio accorcia le attese
    assert ATTESE_SOLA == ATTESE_CIASCUNA == 9, (ATTESE_SOLA, ATTESE_CIASCUNA)
    # l'altra meta': la macchina passa da ferma tre quarti del tempo a mai
    assert FERMA_SOLA == 9, FERMA_SOLA
    assert FERMA_MOLTE == 0, FERMA_MOLTE
    # e conta sempre UNA unita' per volta: due sarebbe un'altra cosa
    assert all(sum(r[t] for r in CONTA_MOLTE) == 1 for t in range(ISTANTI)), \
        "nel secondo pannello c'e' un istante con zero o due unita' che contano"
    # il disegno sta dentro la tela
    assert X0 + ISTANTI * CELLA <= LARG, "le caselle sforano a destra"
    assert MACCHINA_SOTTO + ALTEZZA < LEGENDA - 10, "la legenda tocca l'ultima riga"


def riga(y: float, conta: list[bool], etichetta: str, forte: bool = False) -> str:
    """Una fila di istanti: casella piena dove conta, tratteggiata dove aspetta."""
    pezzi = []
    for t, c in enumerate(conta):
        x = X0 + t * CELLA
        classe = ("tf" if forte else "tc") if c else "ta"
        pezzi.append(f'<rect class="{classe}" x="{x:.0f}" y="{y:.0f}" '
                     f'width="{CELLA - 3}" height="{ALTEZZA}"/>')
    pezzi.append(f'<text class="lb" x="{X0 - 16}" y="{y + ALTEZZA * 0.7:.1f}" '
                 f'text-anchor="end">{etichetta}</text>')
    return "".join(pezzi)


def nota(y: float, ferma: int) -> str:
    return (f'<text class="ls" x="{X0 + ISTANTI * CELLA / 2:.0f}" y="{y:.0f}" '
            f'text-anchor="middle">ferma {ferma} istanti su {ISTANTI}</text>')


def costruisci() -> Figura:
    verifica()
    c = []

    # il pannello di sopra: un'unita' sola, e la macchina che la segue
    c.append(f'<text class="t1" x="{X0}" y="{RIGA_TITOLO_1}">'
             f'una sola unità di calcolo</text>')
    c.append(riga(RIGHE_SOPRA[0], CONTA_SOLA, "l'unità"))
    c.append(riga(RIGHE_SOPRA[1], MACCHINA_SOLA, "la macchina conta", forte=True))
    c.append(nota(RIGHE_SOPRA[1] + ALTEZZA + 22, FERMA_SOLA))

    # il pannello di sotto: le stesse attese, sfalsate
    c.append(f'<text class="t2" x="{X0}" y="{RIGA_TITOLO_2}">'
             f'{PAROLE[ESECUTORI]} unità che si danno il cambio</text>')
    for i, r in enumerate(CONTA_MOLTE):
        c.append(riga(PRIMA_MOLTE + i * PASSO_MOLTE, r, f"unità {i + 1}"))
    c.append(riga(MACCHINA_SOTTO, MACCHINA_MOLTE, "la macchina conta", forte=True))
    c.append(nota(MACCHINA_SOTTO + ALTEZZA + 22, FERMA_MOLTE))

    # la legenda, e il verso del tempo
    c.append(f'<rect class="tc" x="{X0}" y="{LEGENDA - 10}" width="16" height="12"/>')
    c.append(f'<text class="ls" x="{X0 + 22}" y="{LEGENDA}">conta</text>')
    c.append(f'<rect class="ta" x="{X0 + 90}" y="{LEGENDA - 10}" '
             f'width="16" height="12"/>')
    c.append(f'<text class="ls" x="{X0 + 112}" y="{LEGENDA}">aspetta un dato</text>')
    c.append(f'<text class="ls" x="{X0 + ISTANTI * CELLA}" y="{LEGENDA}" '
             f'text-anchor="end">il tempo</text>')

    return Figura(
        larghezza=LARG,
        altezza=ALT,
        alt="Due pannelli sovrapposti, ciascuno una linea del tempo di dodici "
            "istanti disegnati come caselle. Nel pannello di sopra, intestato "
            "«una sola unità di calcolo», una riga mostra l'unità che conta in "
            "un istante su quattro (casella piena) e aspetta un dato negli "
            "altri tre (casella tratteggiata); la riga sotto, «la macchina "
            "conta», ha lo stesso disegno, e una nota dice che la macchina "
            "resta ferma nove istanti su dodici. Nel pannello di sotto, "
            "intestato «quattro unità che si danno il cambio», quattro righe "
            "mostrano quattro unità sfalsate di un istante l'una dall'altra: "
            "ognuna conta in un istante su quattro e aspetta negli altri tre, "
            "esattamente come l'unità sola di sopra, ma in ogni istante è il "
            "turno di una di loro. La riga «la macchina conta» è quindi piena "
            "da un capo all'altro, e la nota dice che resta ferma zero istanti "
            "su dodici. In basso la legenda: casella piena vuol dire che "
            "conta, casella tratteggiata che aspetta un dato.",
        corpo="".join(c),
        stile=f"""    .tc {{ fill:{TEAL}; fill-opacity:0.55; stroke:{TEAL};
           stroke-width:1.4; }}
    .tf {{ fill:{TEAL}; fill-opacity:0.85; stroke:{TEAL}; stroke-width:1.4; }}
    .ta {{ fill:{CREAM}; stroke:{BORDER_STRONG}; stroke-width:1.1;
           stroke-dasharray:3 3; }}
    .lb {{ font-family:{SANS}; font-size:12px; fill:{INK}; }}
    .ls {{ font-family:{SANS}; font-size:11.5px; fill:{FG_MUTED}; }}
    .t1 {{ font-family:{SANS}; font-size:13.5px; font-weight:700;
           fill:{TERRACOTTA}; }}
    .t2 {{ font-family:{SANS}; font-size:13.5px; font-weight:700;
           fill:{TEAL}; }}""",
    )
