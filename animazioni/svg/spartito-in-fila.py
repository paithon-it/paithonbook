"""Lo spartito che scorre contro la fila di token che si allunga.

Il tempo e' il contenuto, e in un modo che una figura ferma non puo' dire: le
due file **crescono a ritmi diversi**. La griglia aggiunge due caselle a ogni
sedicesimo, che ci sia o non ci sia qualcosa da dire; gli eventi non aggiungono
niente finche' non succede niente, e poi ne aggiungono tre in un colpo solo.
Il blocco di codice della sezione (`Audio/generazione-audio.md`) stampa i totali,
32 e 8: quello che non puo' stampare e' il **come ci si arriva**, ed e'
esattamente cio' che qui si vede.

La battuta e' quella della sezione, variante `ribattuta`: un DO grave tenuto
per tutta la battuta e un SOL suonato due volte. Le due funzioni sono le stesse
del capitolo, copiate riga per riga, e girano davvero: le caselle con dentro
l'altezza MIDI, la fila di eventi e i due contatori escono da qui. Tre `assert`
li confrontano con cio' che la sezione stampa, cosi' la figura non puo'
smentire il testo.

Il rullo mostra tre rettangoli e la fila a griglia non ne porta traccia: fra i
sedici `67` della voce acuta non c'e' nessuno stacco a meta' battuta, mentre nel
disegno si vede e fra gli eventi si legge (`NOTE_OFF<67>` seguito da
`NOTE_ON<67>`). E' l'ambiguita' sugli attacchi, disegnata invece che raccontata.

Due scelte di colore, che sono una convenzione: **teal = cio' che e' scritto**
(le note, le caselle, le scatole), **terracotta = cio' che si muove o cresce**
(il cursore e i due contatori). L'**ocra** e' il metro, cioe' le linee dei
quattro movimenti della battuta: si vedono nel rullo e nella griglia, e sotto,
fra gli eventi, non c'e' piu' niente che le ricordi.
"""

from paithon_svg import *

NOME = "spartito-in-fila"
TITOLO = "lo stesso spartito in fila: a griglia o a eventi"

# Una nota e' (altezza MIDI, istante d'inizio, durata); i tempi in sedicesimi.
RIBATTUTA = [(67, 0, 8), (67, 8, 8), (48, 0, 16)]
# Il SOL tenuto invece che ribattuto: non si disegna, serve all'assert che
# tiene la figura attaccata al testo (a griglia le due battute sono identiche).
TENUTA = [(67, 0, 16), (48, 0, 16)]

PASSI, VOCI = 16, 2

# Cio' che il blocco di codice della sezione stampa. Se un giorno il testo
# cambiasse esempio, la figura non lo seguirebbe in silenzio: si ferma qui.
ATTESI = ["NOTE_ON<48>", "NOTE_ON<67>", "TIME_SHIFT<8>", "NOTE_OFF<67>",
          "NOTE_ON<67>", "TIME_SHIFT<8>", "NOTE_OFF<48>", "NOTE_OFF<67>"]


# --------------------------------------------------------------------------
# Le due codifiche (le stesse due funzioni del capitolo)
# --------------------------------------------------------------------------
def a_griglia(note, passi=16, voci=2):
    """Fotografia: per ogni istante l'altezza che ciascuna voce sta suonando."""
    griglia = [[0] * voci for _ in range(passi)]     # 0 = silenzio
    scala = passi // 16                              # quanti passi vale un sedicesimo
    for altezza, inizio, durata in note:
        v = 0 if altezza >= 60 else 1                # voce acuta / voce grave
        for t in range(inizio * scala, (inizio + durata) * scala):
            griglia[t][v] = altezza
    return [x for riga in griglia for x in riga]     # srotolata istante per istante


def a_eventi(note):
    """Ricetta: che cosa accade, e quanto si aspetta fra un fatto e il successivo."""
    fatti = []
    for altezza, inizio, durata in note:
        fatti.append((inizio, f"NOTE_ON<{altezza}>"))
        fatti.append((inizio + durata, f"NOTE_OFF<{altezza}>"))
    sequenza, adesso = [], 0
    for istante, evento in sorted(fatti):
        if istante > adesso:
            sequenza.append(f"TIME_SHIFT<{istante - adesso}>")
            adesso = istante
        sequenza.append(evento)
    return sequenza


def a_eventi_datati(note):
    """La stessa fila, con accanto l'istante in cui ogni token viene emesso.

    Serve solo all'animazione (per sapere *quando* una scatola compare) e un
    `assert` la incolla alla funzione del capitolo: se le due divergessero, la
    figura non partirebbe.
    """
    fatti = []
    for altezza, inizio, durata in note:
        fatti.append((inizio, f"NOTE_ON<{altezza}>"))
        fatti.append((inizio + durata, f"NOTE_OFF<{altezza}>"))
    sequenza, adesso = [], 0
    for istante, evento in sorted(fatti):
        if istante > adesso:
            sequenza.append((istante, f"TIME_SHIFT<{istante - adesso}>"))
            adesso = istante
        sequenza.append((istante, evento))
    return sequenza


# --------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------
W = 728
X0 = 112                      # dove comincia il tempo: a sinistra le etichette
W_COL = 32                    # un sedicesimo, in pixel
X_FINE = X0 + PASSI * W_COL   # 624: fine della battuta
X_DES = W - 16                # colonna dei contatori, allineati a destra

ROLL_Y, LANE = 46, 6          # il rullo: una corsia per semitono
P_MIN, P_MAX = 46, 69
ROLL_H = (P_MAX - P_MIN + 1) * LANE

Y_TICK = ROLL_Y + ROLL_H + 20         # i numeri dei sedicesimi
Y_GRI = ROLL_Y + ROLL_H + 62          # intestazione della griglia
GRI_Y, GRI_H = Y_GRI + 12, 21         # le due righe di caselle
Y_EVE = GRI_Y + 2 * GRI_H + 46        # intestazione degli eventi
EVE_Y, EVE_H, EVE_PASSO = Y_EVE + 12, 28, 34
H = EVE_Y + 3 * EVE_PASSO + 12        # tre righe di scatole, una per istante

CH, PAD, GAP = 8.0, 11.0, 8.0         # scatole: larghezza per carattere e spazi
TEN, FADE = 0.86, 0.14                # tenuta e dissolvenza, in frazioni di passo


def y_di(altezza: float) -> float:
    """Il bordo alto della corsia di un'altezza MIDI."""
    return ROLL_Y + (P_MAX - altezza) * LANE


def costruisci() -> Figura:
    griglia = a_griglia(RIBATTUTA)
    eventi = a_eventi(RIBATTUTA)
    datati = a_eventi_datati(RIBATTUTA)

    # I tre numeri che la sezione stampa, ricontati qui.
    assert len(griglia) == PASSI * VOCI, f"caselle: {len(griglia)}"
    assert eventi == ATTESI, f"la fila di eventi non e' piu' quella: {eventi}"
    assert [t for _, t in datati] == eventi, "le due versioni sono divergite"
    assert a_griglia(TENUTA) == griglia, \
        "a griglia le due battute non sono piu' identiche: il testo dice che lo sono"

    n = PASSI + 1              # le posizioni del cursore: da 0 a 16 compresi
    passo = 100.0 / n
    corpo, anim, nomi = [], [], {}

    def finestra(a: int, b: int) -> tuple[str, int]:
        """(@keyframes, opacita' di riposo) per chi si vede dal passo a al b.

        Il valore al 100% e' lo stesso che l'elemento porta come attributo:
        riposo e fine dell'animazione non possono divergere.
        """
        finale = 1 if b == n - 1 else 0
        if (a, b) not in nomi:
            nome = f"v{a}_{b}"
            tappe = [(0.0, f"opacity:{1 if a == 0 else 0}")]
            if a > 0:
                tappe.append((a * passo - passo * FADE, "opacity:0"))
                tappe.append((a * passo, "opacity:1"))
            if finale:
                tappe.append((100.0, "opacity:1"))
            else:
                tappe.append(((b + 1) * passo - passo * FADE, "opacity:1"))
                tappe.append(((b + 1) * passo, "opacity:0"))
                tappe.append((100.0, "opacity:0"))
            anim.append(keyframes(nome, tappe))
            nomi[(a, b)] = nome
        return nomi[(a, b)], finale

    def gruppo(dentro: str, classe: str, a: int, b: int) -> str:
        nome, op = finestra(a, b)
        return (f'<g class="{classe}" opacity="{op}" '
                f'style="animation:{nome} var(--d) infinite">{dentro}</g>')

    # ---- il rullo: cornice, metro, corsie delle due altezze ----------------
    corpo.append(f'<text class="lbl" x="{X0}" y="{ROLL_Y - 22}">lo spartito: '
                 f'{len(RIBATTUTA)} note in una battuta</text>')
    corpo.append(f'<rect class="ax" x="{X0}" y="{ROLL_Y}" '
                 f'width="{PASSI * W_COL}" height="{ROLL_H}" rx="3"/>')
    for t in range(4, PASSI, 4):                       # i quattro movimenti
        x = X0 + t * W_COL
        corpo.append(f'<line class="metro" x1="{x}" y1="{ROLL_Y}" '
                     f'x2="{x}" y2="{ROLL_Y + ROLL_H}"/>')
    for altezza, nome in ((67, "SOL 67"), (48, "DO 48")):
        y = y_di(altezza) + LANE / 2
        corpo.append(f'<line class="corsia" x1="{X0}" y1="{y:.1f}" '
                     f'x2="{X_FINE}" y2="{y:.1f}"/>')
        corpo.append(f'<text class="lbs" x="{X0 - 12}" y="{y + 4:.1f}" '
                     f'text-anchor="end">{nome}</text>')

    # ---- le tre note: rettangoli veri, con lo stacco a meta' battuta -------
    for altezza, inizio, durata in RIBATTUTA:
        corpo.append(f'<rect class="nota" x="{X0 + inizio * W_COL + 2}" '
                     f'y="{y_di(altezza):.0f}" width="{durata * W_COL - 4}" '
                     f'height="{LANE - 1}" rx="2"/>')

    # ---- i sedicesimi, sotto il rullo --------------------------------------
    for t in range(0, PASSI + 1, 4):
        corpo.append(f'<text class="lbs" x="{X0 + t * W_COL}" y="{Y_TICK}" '
                     f'text-anchor="middle">{t}</text>')
    corpo.append(f'<text class="lbs" x="{X0 - 12}" y="{Y_TICK}" '
                 f'text-anchor="end">sedicesimi</text>')

    # ---- la griglia: una casella per istante e per voce ---------------------
    corpo.append(f'<text class="lbl" x="{X0}" y="{Y_GRI}">a griglia: una casella '
                 f'per ogni istante e per ogni voce</text>')
    for v, voce in enumerate(("acuta", "grave")):
        corpo.append(f'<text class="lbs" x="{X0 - 12}" '
                     f'y="{GRI_Y + v * GRI_H + 15}" text-anchor="end">{voce}</text>')
    for t in range(PASSI):
        celle = []
        for v in range(VOCI):
            x, y = X0 + t * W_COL, GRI_Y + v * GRI_H
            celle.append(f'<rect class="cel" x="{x}" y="{y}" '
                         f'width="{W_COL}" height="{GRI_H}"/>')
            celle.append(f'<text class="num" x="{x + W_COL / 2:.0f}" '
                         f'y="{y + 15}" text-anchor="middle">'
                         f'{griglia[t * VOCI + v]}</text>')
        # la colonna t e' scritta quando il cursore l'ha attraversata
        corpo.append(gruppo("".join(celle), "gri", t + 1, n - 1))
    for i in range(n):
        corpo.append(gruppo(f'<text class="cnt" x="{X_DES}" y="{Y_GRI}" '
                            f'text-anchor="end">{i * VOCI} token</text>',
                            "cnt-g", i, i))

    # ---- gli eventi: una scatola per fatto, una riga per istante -----------
    corpo.append(f'<text class="lbl" x="{X0}" y="{Y_EVE}">a eventi: una scatola '
                 f'solo dove accade qualcosa</text>')
    istanti = sorted({t for t, _ in datati})
    for riga, istante in enumerate(istanti):
        y = EVE_Y + riga * EVE_PASSO
        pezzi = [f'<text class="lbs" x="{X0 - 12}" y="{y + 19}" '
                 f'text-anchor="end">istante {istante}</text>']
        x = float(X0)
        for quando, token in datati:
            if quando != istante:
                continue
            larg = 2 * PAD + CH * len(token)
            pezzi.append(f'<rect class="sca" x="{x:.0f}" y="{y}" '
                         f'width="{larg:.0f}" height="{EVE_H}" rx="6"/>')
            pezzi.append(f'<text class="tok" x="{x + larg / 2:.0f}" '
                         f'y="{y + 19}" text-anchor="middle">'
                         f'{token.replace("<", "&lt;").replace(">", "&gt;")}'
                         f'</text>')
            x += larg + GAP
        corpo.append(gruppo("".join(pezzi), "evt", istante, n - 1))
    for i in range(n):
        quanti = sum(1 for t, _ in datati if t <= i)
        if i and quanti == sum(1 for t, _ in datati if t <= i - 1):
            continue                                   # niente e' cambiato
        fino = max([j for j in range(i, n)
                    if sum(1 for t, _ in datati if t <= j) == quanti])
        corpo.append(gruppo(f'<text class="cnt" x="{X_DES}" y="{Y_EVE}" '
                            f'text-anchor="end">{quanti} token</text>',
                            "cnt-g", i, fino))

    # ---- il cursore: attraversa il rullo e la griglia, gli eventi no -------
    # In due segmenti, con lo stacco sulle righe di testo in mezzo: intero,
    # coprirebbe il sedicesimo «16», che cade proprio dove si ferma.
    # L'etichetta sta in fondo e non in cima: in cima, nei primi sedicesimi,
    # finisce addosso al titolo della figura (visto nel provino dei fermi).
    y_alto, y_basso = ROLL_Y - 12, GRI_Y + VOCI * GRI_H + 8
    corpo.append(f'<g id="cursore">'
                 f'<line class="cur" x1="{X_FINE}" y1="{y_alto}" '
                 f'x2="{X_FINE}" y2="{ROLL_Y + ROLL_H}"/>'
                 f'<line class="cur" x1="{X_FINE}" y1="{GRI_Y - 8}" '
                 f'x2="{X_FINE}" y2="{y_basso}"/>'
                 f'<polygon class="punta" points="{X_FINE - 6},{y_alto - 9} '
                 f'{X_FINE + 6},{y_alto - 9} {X_FINE},{y_alto}"/>'
                 f'<text class="eti" x="{X_FINE}" y="{y_basso + 17}" '
                 f'text-anchor="middle">dove siamo</text></g>')
    tappe = []
    for i in range(n):
        t0, t1 = sosta(i, n, tenuta=0.66)
        d = f"transform:translate({(i - PASSI) * W_COL}px,0px)"
        tappe += [(t0, d), (t1, d)]
    tappe.append((100.0, "transform:translate(0px,0px)"))
    anim.append(keyframes("scorre", tappe))

    return Figura(
        larghezza=W, altezza=H,
        alt="Una battuta di musica disegnata come rullo di pianola: tre "
            "rettangoli, un DO grave lungo tutta la battuta e due SOL da mezza "
            "battuta ciascuno. Un cursore la percorre da sinistra a destra, e "
            "sotto crescono due file: quella a griglia aggiunge due caselle a "
            "ogni sedicesimo e arriva a trentadue, quella a eventi aggiunge "
            "scatole solo dove accade qualcosa e si ferma a otto.",
        corpo="".join(corpo),
        stile=f"""    .metro {{ stroke:{OCRA}; stroke-width:1.2; }}
    .corsia {{ stroke:{BORDER}; stroke-width:1; stroke-dasharray:3 5; }}
    .nota {{ fill:{TEAL}; }}
    rect.cel {{ fill:none; stroke:{TEAL}; stroke-width:1.2; }}
    .num  {{ font-family:{SANS}; font-size:12px; fill:{FG_MUTED}; }}
    rect.sca {{ fill:none; stroke:{TEAL}; stroke-width:1.8; }}
    .tok  {{ font-family:{SANS}; font-size:13px; fill:{INK}; }}
    .cnt  {{ font-family:{SANS}; font-size:15px; font-weight:700;
            fill:{TERRACOTTA}; }}
    .cur  {{ stroke:{TERRACOTTA}; stroke-width:2.2; }}
    .punta {{ fill:{TERRACOTTA}; }}
    .eti  {{ font-family:{SANS}; font-size:13px; font-weight:700;
            fill:{TERRACOTTA}; }}
    #cursore {{ animation:scorre var(--d) infinite; transform-box:view-box; }}""",
        animazioni=anim,
        durata=n * 0.55,
        fermi="#cursore, .gri, .evt, .cnt-g",
    )
