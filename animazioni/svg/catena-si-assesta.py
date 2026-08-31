"""Una catena di Markov che si assesta: la distribuzione dimentica da dove è
partita.

Le quattro colonne sono le quattro pagine del web in miniatura di
`Matematica/catene-di-markov.md`, e le altezze sono le probabilità dopo 0, 1,
2, 3, 5 e 10 passi. Non sono trascritte: la matrice del passaggio è costruita
dai link, lo smorzamento è quello della pagina, e `verifica()` confronta con
un assert lo stato di riposo con l'autovettore di autovalore 1 calcolato per
un'altra strada. Se un giorno il testo cambia i link e la figura no, la figura
non si genera nemmeno.

Quello che il fermo immagine non può mostrare, e che qui si vede: le colonne
non salgono verso il bersaglio, ci **girano attorno**. La colonna C lo scavalca
al primo passo (0,4625 contro 0,3839), gli passa sotto al secondo (0,327), e
continua ad alternare stringendo. Chi guarda solo il risultato crede che la
convergenza sia monotona, e non lo è: la ragione sta nella coppia di autovalori
complessi coniugati che la pagina stampa poco più avanti.

Lo stato di riposo è la distribuzione stazionaria, con la riga tratteggiata
che dice dove ciascuna colonna doveva arrivare: chi non anima (stampa, PDF,
`prefers-reduced-motion`) vede il risultato e la soglia che lo giudica.
"""

from paithon_svg import *

NOME = "catena-si-assesta"
TITOLO = "una catena di Markov si assesta"

# chi punta a chi, nel web in miniatura della pagina
LINK = {0: [1, 2], 1: [2], 2: [0], 3: [0, 2]}
NOMI = ["A", "B", "C", "D"]
SMORZAMENTO = 0.85
TAPPE = [0, 1, 2, 3, 5, 10]        # dopo quanti passi guardiamo


def passaggio():
    """La matrice di Google: colonne che sommano a uno."""
    n = len(NOMI)
    M = [[0.0] * n for _ in range(n)]
    for da, verso in LINK.items():
        for a in verso:
            M[a][da] = 1.0 / len(verso)
    salto = (1.0 - SMORZAMENTO) / n
    return [[SMORZAMENTO * M[i][j] + salto for j in range(n)] for i in range(n)]


def applica(P, x):
    return [sum(P[i][j] * x[j] for j in range(len(x))) for i in range(len(x))]


def storia():
    """Le distribuzioni alle tappe, piu' quella a regime.

    L'ultimo fotogramma **non** e' una delle tappe: e' la stazionaria vera,
    ottenuta iterando finche' non si muove piu'. Serve perche' lo stato di
    riposo di questa figura e' quello che vede la stampa, e la didascalia
    promette la stazionaria: dopo dieci passi lo scarto e' ancora due
    millesimi, cioe' visibile.
    """
    n = len(NOMI)
    P = passaggio()
    x = [1.0 / n] * n
    fuori, letti = {0: list(x)}, 0
    for t in range(1, 400):
        y = applica(P, x)
        mosso = max(abs(a - b) for a, b in zip(y, x))
        x = y
        letti = t
        if t in TAPPE:
            fuori[t] = list(x)
        if mosso < 1e-13 and t >= max(TAPPE):
            break
    assert letti >= max(TAPPE), f"iterate solo {letti} tappe"
    return [fuori[t] for t in TAPPE] + [list(x)]


def verifica(finale):
    """Lo stato di riposo è davvero la stazionaria?

    Confronto per una strada diversa da quella che l'ha prodotto: invece di
    iterare, si chiede che P applicata a quel vettore lo lasci fermo. Se le due
    strade non coincidono, la figura mentirebbe sul suo unico contenuto.
    """
    P = passaggio()
    y = applica(P, finale)
    scarto = max(abs(a - b) for a, b in zip(y, finale))
    assert scarto < 1e-4, f"lo stato finale non e' stazionario: scarto {scarto}"
    assert abs(sum(finale) - 1.0) < 1e-9, "le probabilita' non sommano a uno"
    for col in zip(*P):
        assert abs(sum(col) - 1.0) < 1e-9, "una colonna non somma a uno"
    return scarto


def costruisci() -> Figura:
    passi = storia()
    finale = passi[-1]
    scarto = verifica(finale)

    r = Riquadro(x=96, y=56, larg=470, alt=300,
                 xmin=0.0, xmax=4.0, ymin=0.0, ymax=0.55)
    n = len(NOMI)
    largh = r.larg / n * 0.46
    corpo = [r.cornice()]
    anim = []

    # le tacche dell'asse verticale
    for v in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        y = r.sy(v)
        if v > 0:
            corpo.append(f'<line class="axc" x1="{r.x:.1f}" y1="{y:.1f}" '
                         f'x2="{r.x + r.larg:.1f}" y2="{y:.1f}"/>')
        corpo.append(f'<text class="lbs" x="{r.x - 12:.1f}" y="{y + 4:.1f}" '
                     f'text-anchor="end">{v:.1f}</text>')

    base = r.sy(0.0)
    for i, (nome, alt) in enumerate(zip(NOMI, finale)):
        cx = r.sx(i + 0.5)
        x0 = cx - largh / 2
        h = base - r.sy(alt)

        # RIPOSO: la colonna alta quanto la stazionaria, in coordinate vere.
        corpo.append(f'<rect class="col" x="{x0:.1f}" y="{r.sy(alt):.1f}" '
                     f'width="{largh:.1f}" height="{h:.1f}" '
                     f'style="transform-origin:{cx:.1f}px {base:.1f}px;'
                     f'animation:col{i} var(--d) ease-in-out infinite;"/>')
        # la soglia: dove la colonna doveva arrivare
        corpo.append(f'<line class="meta" x1="{x0 - 8:.1f}" y1="{r.sy(alt):.1f}" '
                     f'x2="{x0 + largh + 8:.1f}" y2="{r.sy(alt):.1f}"/>')
        corpo.append(f'<text class="nom" x="{cx:.1f}" y="{base + 24:.1f}" '
                     f'text-anchor="middle">{nome}</text>')
        corpo.append(f'<text class="val" x="{cx:.1f}" y="{r.sy(alt) - 10:.1f}" '
                     f'text-anchor="middle">{alt:.3f}</text>')

        # l'animazione parte dall'inverso e finisce sull'identita'
        tappe = []
        for k, dist in enumerate(passi):
            a, b = sosta(k, len(passi))   # passi include la stazionaria
            fattore = dist[i] / alt
            tappe.append((a, f"transform:scaleY({fattore:.4f})"))
            tappe.append((b, f"transform:scaleY({fattore:.4f})"))
        tappe.append((100.0, "transform:scaleY(1)"))
        # `passi` finisce con la stazionaria, quindi l'ultima tappa e' gia'
        # l'identita': il riposo e il fotogramma finale coincidono.
        anim.append(keyframes(f"col{i}", tappe))

    # l'etichetta del passo, che compare a turno
    for k, t in enumerate(TAPPE + [None]):
        a, b = sosta(k, len(TAPPE) + 1)
        testo = ("a regime" if t is None
                 else "partenza: tutte uguali" if t == 0
                 else "dopo 1 passo" if t == 1 else f"dopo {t} passi")
        # l'ultima etichetta e' lo stato di RIPOSO, quindi nasce visibile:
        # e' quella che vedra' la stampa.
        classe = "pas" if t is not None else "paf"
        corpo.append(f'<text class="{classe}" x="{r.x + r.larg / 2:.1f}" '
                     f'y="{r.y - 18:.1f}" text-anchor="middle" '
                     f'style="animation:pas{k} var(--d) ease-in-out infinite;">'
                     f'{testo}</text>')
        # Gli estremi 0% e 100% vanno dichiarati, altrimenti il browser li
        # sintetizza dallo stile base e in un fermo immagine si vedono due
        # etichette insieme. Il provino lo ha mostrato al primo giro.
        tappe_lab = [(0.0, "opacity:0")] if k > 0 else [(0.0, "opacity:1")]
        tappe_lab += [(max(0.1, a - 1.5), "opacity:0"), (a, "opacity:1"),
                      (b, "opacity:1")]
        if t is not None:
            tappe_lab += [(min(99.9, b + 1.5), "opacity:0"),
                          (100.0, "opacity:0")]
        else:
            tappe_lab += [(100.0, "opacity:1")]
        anim.append(keyframes(f"pas{k}", sorted(tappe_lab)))

    corpo.append(f'<text class="lbs" x="{r.x - 12:.1f}" y="{r.y - 18:.1f}" '
                 f'text-anchor="end">probabilità</text>')
    corpo.append(f'<text class="cod" x="{r.x + r.larg + 24:.1f}" y="{r.y + 40:.1f}">'
                 f'la riga tratteggiata è</text>')
    corpo.append(f'<text class="cod" x="{r.x + r.larg + 24:.1f}" y="{r.y + 58:.1f}">'
                 f'la distribuzione</text>')
    corpo.append(f'<text class="cod" x="{r.x + r.larg + 24:.1f}" y="{r.y + 76:.1f}">'
                 f'stazionaria: dove la</text>')
    corpo.append(f'<text class="cod" x="{r.x + r.larg + 24:.1f}" y="{r.y + 94:.1f}">'
                 f'catena finisce sempre,</text>')
    corpo.append(f'<text class="cod" x="{r.x + r.larg + 24:.1f}" y="{r.y + 112:.1f}">'
                 f'da qualunque parte</text>')
    corpo.append(f'<text class="cod" x="{r.x + r.larg + 24:.1f}" y="{r.y + 130:.1f}">'
                 f'sia partita</text>')

    dopo_uno = passi[1]
    it = lambda x: f"{x:.3f}".replace(".", ",")   # separatore decimale italiano
    return Figura(
        larghezza=760, altezza=400,
        alt="Quattro colonne, una per pagina del web in miniatura, alte quanto "
            "la probabilità di trovarsi su quella pagina. Si parte da quattro "
            "colonne uguali, alte 0,25 ciascuna; dopo un passo la colonna C "
            f"scavalca il proprio bersaglio salendo a {it(dopo_uno[2])} e la B "
            f"scende sotto il suo a {it(dopo_uno[1])}; nei passi successivi A, "
            "B e C continuano a passare sopra e sotto il proprio bersaglio "
            "oscillando sempre meno, mentre D, che nessuno linka, è già "
            "arrivata al suo dopo un passo solo e non si muove più. Le "
            f"quattro si assestano sulla distribuzione stazionaria {it(finale[0])}, "
            f"{it(finale[1])}, {it(finale[2])}, {it(finale[3])}, marcata da una "
            "riga tratteggiata sopra ciascuna colonna.",
        corpo="".join(corpo),
        stile=f"""    .col  {{ fill:{TEAL}; fill-opacity:0.75; stroke:{TEAL};
            stroke-width:1.5; }}
    .meta {{ stroke:{TERRACOTTA}; stroke-width:2; stroke-dasharray:6 4; }}
    .nom  {{ font-family:{SANS}; font-size:16px; font-weight:700; fill:{INK}; }}
    .val  {{ font-family:{SANS}; font-size:13px; font-weight:700;
            fill:{TERRACOTTA}; }}
    .pas  {{ font-family:{SANS}; font-size:17px; font-weight:700;
            fill:{INK}; opacity:0; }}
    .paf  {{ font-family:{SANS}; font-size:17px; font-weight:700;
            fill:{INK}; opacity:1; }}
    .cod  {{ font-family:{SANS}; font-size:12.5px; fill:{FG_MUTED}; }}""",
        animazioni=anim,
        durata=(len(TAPPE) + 1) * 1.5,
        fermi=".col, .pas, .paf",
    )
