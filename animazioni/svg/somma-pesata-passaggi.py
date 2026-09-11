"""Una risposta sola, e ogni passaggio ci mette quanto pesa.

`Transformers/rag.md` dice, in una riga della scheda Superiore, che la risposta
si ottiene sommando sui passaggi recuperati, e nella scheda Elementare che lo
studente «dà più retta» alla pagina che l'indice segnalava con più decisione.
Fra le due c'è un gesto che nessuna delle due racconta, ed è quello che decide
tutto: i passaggi **non finiscono nello stesso foglio**. Ciascuno produce la
sua risposta per conto proprio, e solo alla fine le risposte si sommano, ognuna
contata per il peso che il recupero le ha dato.

Qui il tempo è il contenuto per una ragione precisa: il fatto da vedere è il
passaggio da «tre risposte, una per pagina» a «una risposta sola», e in mezzo
c'è la somma. Un fermo immagine mostrerebbe o l'inizio o la fine.

Tre cose che questa scena si guarda dal dire, e sono le tre in cui una scena
sbagliata mentirebbe:

- **i rami non arrivano uno alla volta.** Le tre bozze crescono insieme, nella
  stessa fetta di tempo, perché nel modello girano separate e nessuna corregge
  l'altra. Una scena in cui il secondo passaggio «aggiusta» la risposta
  costruita sul primo racconterebbe un aggiornamento bayesiano, che è un altro
  algoritmo;
- **i pesi non cambiano strada facendo.** Sono fissati dal recupero prima che
  il generatore scriva una parola, e nella scena compaiono già fatti;
- **i passaggi non si vedono fra loro.** Ogni ramo ha davanti un passaggio
  solo. Mettere i tre passaggi nello stesso contesto è l'altra ricetta, quella
  a prompt aumentato, che la pagina distingue.

Il fatto che la scena porta a casa, e che non è ovvio: la somma pesata **non è
un voto**. Due passaggi su tre preferiscono la stessa risposta e quella
risposta perde, perché il terzo pesa più di loro due messi insieme. È il
rovescio esatto della rassicurazione che il lettore si porterebbe via da
«guarda più fonti», ed è il punto di rottura della pagina: un recupero
sbagliato ma sicuro di sé porta a casa la risposta.

I numeri sono di un esempio minimo, dichiarato in didascalia: la pagina non ha
pesi da riusare, e i punteggi di somiglianza che stampa non sono pesi. Li
calcola `distribuzioni()` e li difendono gli `assert`.

Lo stato di riposo è la fine: le tre bozze scritte, i pesi applicati, la somma
composta.
"""

from paithon_svg import *

NOME = "somma-pesata-passaggi"
TITOLO = "tre passaggi, tre bozze, una risposta sola"

# --------------------------------------------------------------------------
# L'esempio minimo: i pesi del recupero, e la risposta che ogni passaggio
# preferisce. Nient'altro è scritto a mano.
# --------------------------------------------------------------------------
DOMANDA = "dove avviene la fotosintesi?"
PESI = {"pag. 214": 0.60, "pag. 380": 0.22, "pag. 91": 0.18}
RAMI = {
    "pag. 214": {"nelle foglie": 0.70, "nel fusto": 0.20, "nelle radici": 0.10},
    "pag. 380": {"nelle foglie": 0.15, "nel fusto": 0.75, "nelle radici": 0.10},
    "pag. 91":  {"nelle foglie": 0.10, "nel fusto": 0.70, "nelle radici": 0.20},
}
RISPOSTE = ("nelle foglie", "nel fusto", "nelle radici")
TINTA = {"pag. 214": "pA", "pag. 380": "pB", "pag. 91": "pC"}


def distribuzioni():
    """La somma pesata, e il confronto con il voto a maggioranza."""
    somma = {r: sum(PESI[p] * RAMI[p][r] for p in PESI) for r in RISPOSTE}
    media = {r: sum(RAMI[p][r] for p in PESI) / len(PESI) for r in RISPOSTE}
    preferite = [max(RAMI[p], key=RAMI[p].get) for p in PESI]
    return somma, media, preferite


def num(v: float) -> str:
    return f"{v:.3f}".replace(".", ",")


def due(v: float) -> str:
    """Due decimali, per il conto scritto per esteso."""
    return f"{v:.2f}".replace(".", ",")


# --------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------
LARG, ALT = 900, 530
X_RAMO, W_PANN, H_PANN, SALTO = 34.0, 382.0, 110.0, 14.0
Y_RAMO = 106.0
W_ETICH = 96.0                   # la colonna dei nomi delle risposte
W_BARRA = 150.0                  # la barra più lunga di un ramo
W_PESO = W_BARRA                 # il peso sta sulla stessa scala: dentro un
                                 # riquadro due barre lunghe uguali devono
                                 # dire due numeri uguali

X_SOMMA, W_SOMMA = 470.0, 330.0  # il pannello della somma
W_SETICH, W_SBARRA = 96.0, 176.0
Y_SOMMA, H_SOMMA, PASSO_S = 150.0, 34.0, 76.0

FETTE = 4                        # ferme, rami, pesi, somma


def verifica(somma, media, preferite) -> None:
    """Difende quello che la didascalia promette."""
    # Sono distribuzioni: se non sommano a uno, le barre mentono sulle aree.
    assert abs(sum(PESI.values()) - 1) < 1e-9, "i pesi non sommano a uno"
    for p, d in RAMI.items():
        assert abs(sum(d.values()) - 1) < 1e-9, f"{p}: le risposte non sommano a uno"
    assert abs(sum(somma.values()) - 1) < 1e-9, "la somma pesata non fa uno"

    # La somma è davvero la somma: ricontata senza passare da distribuzioni().
    for r in RISPOSTE:
        atteso = 0.0
        for p in PESI:
            atteso += PESI[p] * RAMI[p][r]
        assert abs(somma[r] - atteso) < 1e-12, f"la colonna di {r} non torna"

    # Il fatto della didascalia, nei suoi tre pezzi.
    vince = max(somma, key=somma.get)
    assert vince == "nelle foglie", f"la somma pesata non vince su «nelle foglie» ma su «{vince}»"
    maggioranza = max(set(preferite), key=preferite.count)
    assert maggioranza == "nel fusto" and preferite.count("nel fusto") == 2, \
        f"le pagine preferite non sono due su tre per «nel fusto»: {preferite}"
    assert max(media, key=media.get) == "nel fusto", \
        "senza i pesi non vincerebbe «nel fusto»: il confronto della didascalia cade"
    primo = max(PESI, key=PESI.get)
    assert PESI[primo] > sum(v for k, v in PESI.items() if k != primo), \
        "il primo passaggio non pesa più degli altri due messi insieme"

    # Il conto scritto per esteso arrotonda i fattori a due decimali: se i
    # fattori arrotondati non riproducono il totale, la figura stampa
    # un'addizione che non torna.
    vince = max(somma, key=somma.get)
    arr = lambda v: float(due(v).replace(",", "."))
    rifatto = sum(arr(PESI[pa]) * arr(RAMI[pa][vince]) for pa in PESI)
    assert num(rifatto) == num(somma[vince]), \
        (f"il conto disegnato non torna: i fattori arrotondati danno "
         f"{num(rifatto)}, la figura stampa {num(somma[vince])}")

    # Dentro un riquadro le lunghezze disegnate devono stare nello stesso
    # ordine dei numeri, peso compreso: con due scale diverse non era così.
    assert W_PESO == W_BARRA, "il peso sta su una scala sua: le barre mentono"

    # E il punto di rottura: sommare non filtra. Chi perde resta grosso.
    assert somma["nel fusto"] > 0.35, \
        "la risposta che perde sparisce: la scena direbbe che la somma protegge"

    # Il disegno sta dentro la tela.
    fondo = Y_RAMO + len(RAMI) * H_PANN + (len(RAMI) - 1) * SALTO
    assert fondo < ALT - 60, "i tre pannelli arrivano troppo in basso"
    assert X_SOMMA + W_SOMMA <= LARG - 20, "il pannello della somma esce dalla tela"


def verifica_alt(alt: str, disegno: str) -> None:
    """Il testo alternativo è una copia a mano dei numeri: qui si controlla.

    Il confronto è con i numeri **disegnati**, ritagliati dal corpo dell'SVG,
    non con quelli calcolati: così copre anche il conto scritto per esteso,
    che arrotonda i fattori a due decimali. Senza, falsando un peso la figura
    si ridisegnava e l'alt continuava a dire il vecchio, che è quello che il
    lettore non vedente sente al posto della figura.
    """
    scritti = set(re.findall(r"\d,\d+", alt))
    disegnati = set(re.findall(r"\d,\d+", re.sub(r"<[^>]*>", " ", disegno)))
    assert disegnati, "nessun numero nel disegno: il ritaglio del corpo è sbagliato"
    assert not disegnati - scritti, \
        f"numeri disegnati che l'alt non dice: {sorted(disegnati - scritti)}"
    assert not scritti - disegnati, \
        f"numeri che l'alt dice e la figura non disegna: {sorted(scritti - disegnati)}"


def costruisci() -> Figura:
    somma, media, preferite = distribuzioni()
    verifica(somma, media, preferite)

    corpo, anim = [], []
    # L'ordine è quello vero: prima il recupero fissa i pesi, poi le tre bozze
    # nascono (tutte nella stessa fetta, perché nascono insieme), e per ultima
    # arriva la somma.
    t_pesi, f_pesi = sosta(1, FETTE)
    t_rami, f_rami = sosta(2, FETTE)
    t_som, f_som = sosta(3, FETTE)

    anim.append(keyframes("pesi", [(0.0, "opacity:0"),
                                   (t_pesi, "opacity:0"),
                                   (f_pesi, "opacity:1"),
                                   (100.0, "opacity:1")]))
    anim.append(keyframes("rami", [(0.0, "transform:scaleX(0)"),
                                   (t_rami, "transform:scaleX(0)"),
                                   (f_rami, "transform:scaleX(1)"),
                                   (100.0, "transform:scaleX(1)")]))
    anim.append(keyframes("somm", [(0.0, "transform:scaleX(0)"),
                                   (t_som, "transform:scaleX(0)"),
                                   (f_som, "transform:scaleX(1)"),
                                   (100.0, "transform:scaleX(1)")]))
    # I numeri compaiono insieme alla cosa che misurano: un fermo immagine con
    # la cifra già scritta e la barra ancora vuota direbbe che il conto c'era
    # prima della bozza.
    anim.append(keyframes("nrami", [(0.0, "opacity:0"),
                                    (t_rami, "opacity:0"),
                                    (f_rami, "opacity:1"),
                                    (100.0, "opacity:1")]))
    anim.append(keyframes("nsomm", [(0.0, "opacity:0"),
                                    (t_som, "opacity:0"),
                                    (f_som, "opacity:1"),
                                    (100.0, "opacity:1")]))

    corpo.append(f'<text class="ttl" x="{X_RAMO:.1f}" y="36">'
                 f'«{DOMANDA.capitalize()}» Ogni pagina risponde da sola, poi '
                 f'le risposte si sommano</text>')
    corpo.append(f'<text class="lbs" x="{X_RAMO:.1f}" y="58">'
                 f'le tre bozze nascono insieme e nessuna corregge le altre; '
                 f'i pesi vengono dal recupero, e non cambiano più</text>')
    corpo.append(f'<text class="et" x="{X_RAMO:.1f}" y="{Y_RAMO - 14:.1f}">'
                 f'le tre bozze, una per pagina</text>')

    # ---- i tre rami ---------------------------------------------------------
    for i, passaggio in enumerate(PESI):
        y0 = Y_RAMO + i * (H_PANN + SALTO)
        corpo.append(f'<rect class="pann" x="{X_RAMO:.1f}" y="{y0:.1f}" '
                     f'width="{W_PANN:.1f}" height="{H_PANN:.1f}" rx="5"/>')
        corpo.append(f'<text class="{TINTA[passaggio]}t" x="{X_RAMO + 14:.1f}" '
                     f'y="{y0 + 24:.1f}">{passaggio}</text>')

        # il peso, che arriva già fatto e non si muove più
        xp = X_RAMO + 108
        corpo.append(f'<g style="animation:pesi var(--d) infinite" opacity="1">'
                     f'<text class="lbs" x="{xp:.1f}" y="{y0 + 24:.1f}">peso'
                     f'</text>'
                     f'<rect class="{TINTA[passaggio]}b" x="{xp + 40:.1f}" '
                     f'y="{y0 + 13:.1f}" '
                     f'width="{W_PESO * PESI[passaggio]:.1f}" height="13"/>'
                     f'<text class="val" x="{xp + 60 + W_PESO:.1f}" '
                     f'y="{y0 + 24:.1f}" text-anchor="end">'
                     f'{num(PESI[passaggio])}</text></g>')

        # le tre risposte del ramo, che crescono tutte nella stessa fetta
        for j, r in enumerate(RISPOSTE):
            y = y0 + 42 + j * 21
            corpo.append(f'<text class="et" x="{X_RAMO + 14:.1f}" '
                         f'y="{y + 11:.1f}">{r}</text>')
            xb = X_RAMO + 14 + W_ETICH
            corpo.append(f'<rect class="{TINTA[passaggio]}b cresce" '
                         f'x="{xb:.1f}" y="{y:.1f}" '
                         f'width="{W_BARRA * RAMI[passaggio][r]:.1f}" '
                         f'height="14" style="animation:rami var(--d) '
                         f'infinite"/>')
            corpo.append(f'<text class="lbs" '
                         f'x="{xb + W_BARRA + 8:.1f}" y="{y + 11:.1f}" '
                         f'opacity="1" style="animation:nrami var(--d) '
                         f'infinite">{num(RAMI[passaggio][r])}</text>')

    # ---- la somma -----------------------------------------------------------
    corpo.append(f'<text class="et" x="{X_SOMMA:.1f}" y="{Y_RAMO - 14:.1f}">'
                 f'la somma, ciascuna contata per quanto pesa</text>')

    ordine = sorted(RISPOSTE, key=lambda r: -somma[r])
    for i, r in enumerate(ordine):
        y = Y_SOMMA + i * PASSO_S
        corpo.append(f'<text class="et" x="{X_SOMMA:.1f}" y="{y + 23:.1f}" '
                     f'opacity="1" style="animation:nsomm var(--d) infinite">'
                     f'{r}</text>')
        pezzi, x = [], X_SOMMA + W_SETICH
        for passaggio in PESI:
            w = W_SBARRA * PESI[passaggio] * RAMI[passaggio][r]
            pezzi.append(f'<rect class="{TINTA[passaggio]}b" x="{x:.1f}" '
                         f'y="{y:.1f}" width="{w:.1f}" height="{H_SOMMA:.1f}"/>')
            x += w
        corpo.append(f'<g class="cresce" style="animation:somm var(--d) '
                     f'infinite">{"".join(pezzi)}</g>')
        corpo.append(f'<text class="val" x="{X_SOMMA + W_SOMMA:.1f}" '
                     f'y="{y + 23:.1f}" text-anchor="end" opacity="1" '
                     f'style="animation:nsomm var(--d) infinite">'
                     f'{num(somma[r])}</text>')
        # il conto per esteso, una volta sola, sotto chi vince: senza, il
        # lettore vede sommare e non sa che prima si moltiplica
        if i == 0:
            conto = " + ".join(f"{due(PESI[pa])}×{due(RAMI[pa][r])}"
                               for pa in PESI)
            corpo.append(f'<text class="lbs" x="{X_SOMMA + W_SETICH:.1f}" '
                         f'y="{y + H_SOMMA + 16:.1f}" opacity="1" '
                         f'style="animation:nsomm var(--d) infinite">'
                         f'{conto} = {num(somma[r])}</text>')

    corpo.append(f'<text class="lbs" x="{X_SOMMA:.1f}" '
                 f'y="{Y_SOMMA + 3 * PASSO_S + 4:.1f}" opacity="1" '
                 f'style="animation:nsomm var(--d) infinite">'
                 f'«{max(set(preferite), key=preferite.count)}» piace a due '
                 f'pagine su tre, e perde.</text>')

    # ---- la legenda dei tre colori -----------------------------------------
    y_leg = ALT - 34
    x = X_RAMO
    for passaggio in PESI:
        corpo.append(f'<rect class="{TINTA[passaggio]}b" x="{x:.1f}" '
                     f'y="{y_leg - 11:.1f}" width="15" height="13"/>')
        corpo.append(f'<text class="lbs" x="{x + 22:.1f}" y="{y_leg:.1f}">'
                     f'{passaggio}</text>')
        x += 132
    corpo.append(f'<text class="lbs" x="{X_SOMMA:.1f}" y="{y_leg:.1f}">'
                 f'le tre risposte sommano a uno.</text>')

    alt = ("La domanda è «Dove avviene la fotosintesi?». A sinistra tre "
       "riquadri, uno per pagina recuperata: pag. 214 in terracotta con "
       "peso 0,600, pag. 380 in teal con peso 0,220, pag. 91 in ocra con "
       "peso 0,180. Dentro ogni riquadro tre barre, una per risposta "
       "possibile: pag. 214 dà 0,700 a «nelle foglie», 0,200 a «nel "
       "fusto» e 0,100 a «nelle radici»; pag. 380 dà 0,750 a «nel "
       "fusto», 0,150 a «nelle foglie» e 0,100 a «nelle radici»; pag. 91 "
       "dà 0,700 a «nel fusto», 0,200 a «nelle radici» e 0,100 a «nelle "
       "foglie». Le tre barre di ogni riquadro crescono tutte nello "
       "stesso momento, e le barre dei pesi compaiono già fatte. A "
       "destra le tre risposte sommate, ciascuna una barra composta di "
       "tre segmenti, uno per pagina: «nelle foglie» vale 0,471, «nel "
       "fusto» 0,411, «nelle radici» 0,118, e sotto la prima è scritto "
       "il conto per esteso, 0,60×0,70 più 0,22×0,15 più 0,18×0,10. "
       "Vince «nelle foglie», benché due pagine su tre preferiscano «nel "
       "fusto», perché il segmento di pag. 214 è più lungo degli altri "
       "due messi insieme; e «nel fusto», che perde, si porta via lo "
       "stesso quattro decimi.")

    disegno = "".join(corpo)
    verifica_alt(alt, disegno)

    return Figura(
        larghezza=LARG, altezza=ALT,
        alt=alt,
        corpo=disegno,
        stile=f"""    .pann {{ fill:none; stroke:{BORDER_STRONG}; stroke-width:1.4; }}
    .pAb  {{ fill:{TERRACOTTA}; }}
    .pBb  {{ fill:{TEAL}; }}
    .pCb  {{ fill:{OCRA}; }}
    .pAt  {{ font-family:{SANS}; font-size:14px; font-weight:700;
            fill:{TERRACOTTA}; }}
    .pBt  {{ font-family:{SANS}; font-size:14px; font-weight:700;
            fill:{TEAL}; }}
    .pCt  {{ font-family:{SANS}; font-size:14px; font-weight:700;
            fill:{OCRA}; }}
    .et   {{ font-family:{SANS}; font-size:13.5px; font-weight:700;
            fill:{INK}; }}
    .val  {{ font-family:{SANS}; font-size:13px; font-weight:700;
            fill:{INK}; }}
    .cresce {{ transform-box:fill-box; transform-origin:left center; }}""",
        animazioni=anim,
        durata=9.0,
        fermi=".cresce, g[style], text[style]",
    )
