"""Warp divergence: stesso lavoro, tempo doppio.

Il tempo è il contenuto, ed è proprio la cosa che si paga: quando i trentadue
thread di un warp prendono strade diverse, l'hardware non li esegue insieme,
esegue prima gli uni e poi gli altri. Una figura ferma mostrerebbe due
disposizioni di caselle; il punto è che la seconda *dura il doppio*, e la
durata si vede solo se scorre.

La scena non inventa nessun numero: conta le caselle. Il lavoro utile (le
caselle accese) è identico nei due casi, il numero di passi raddoppia, e le
due cose insieme sono la definizione del costo della divergenza. Le asserzioni
pretendono esattamente questo, così una modifica al disegno che rompesse la
pari non passerebbe.

Lo stato di riposo è la fine: tutti e due i percorsi conclusi, con il ritardo
del secondo bene in vista. Chi non anima vede la figura conclusa.
"""

from paithon_svg import *

NOME = "plotone-si-divide"
TITOLO = "un warp che si divide costa il doppio"

LARGHEZZA_WARP = 32       # il plotone del capitolo: 32 thread, e non cambia
PASSI_RAMO = 3            # quanto è lungo ciascuno dei due rami


def esecuzione(divergente: bool):
    """I passi dell'esecuzione, ognuno con la maschera dei thread attivi.

    Senza divergenza tutti i thread prendono lo stesso ramo e i passi sono
    quelli di un ramo solo. Con la divergenza l'hardware serializza: prima i
    pari, con i dispari spenti, poi i dispari, con i pari spenti.
    """
    if not divergente:
        return [[True] * LARGHEZZA_WARP for _ in range(PASSI_RAMO)]
    pari = [i % 2 == 0 for i in range(LARGHEZZA_WARP)]
    dispari = [not p for p in pari]
    return [list(pari) for _ in range(PASSI_RAMO)] + \
           [list(dispari) for _ in range(PASSI_RAMO)]


COERENTE = esecuzione(False)
DIVERGENTE = esecuzione(True)

LAVORO_COERENTE = sum(sum(p) for p in COERENTE)
LAVORO_DIVERGENTE = sum(sum(p) for p in DIVERGENTE)

# --- le asserzioni, che difendono quello che la didascalia promette ---------

if LAVORO_COERENTE != LAVORO_DIVERGENTE:
    raise AssertionError(
        f"la didascalia promette lo stesso lavoro, e le caselle accese sono "
        f"{LAVORO_COERENTE} contro {LAVORO_DIVERGENTE}")

if len(DIVERGENTE) != 2 * len(COERENTE):
    raise AssertionError(
        f"la didascalia promette il tempo doppio, e i passi sono "
        f"{len(COERENTE)} contro {len(DIVERGENTE)}")

# e la metà spenta deve esserci davvero, o non c'è niente da mostrare
if any(all(passo) for passo in DIVERGENTE):
    raise AssertionError("nel caso divergente nessun passo ha thread spenti")

# I numeri che la didascalia scrive per esteso: senza queste tre, cambiare una
# delle due costanti qui sopra lascia la pagina a dire cifre che il disegno non
# porta piu', e il generatore non protesta.
if LARGHEZZA_WARP != 32:
    raise AssertionError(
        f"la didascalia dice trentadue thread per warp, e sono {LARGHEZZA_WARP}")

if (len(COERENTE), len(DIVERGENTE)) != (3, 6):
    raise AssertionError(
        f"la didascalia dice tre passi contro sei, e sono "
        f"{len(COERENTE)} contro {len(DIVERGENTE)}")

if LAVORO_COERENTE != 96:
    raise AssertionError(
        f"la didascalia dice novantasei caselle accese, e sono {LAVORO_COERENTE}")

if any(sum(passo) != LARGHEZZA_WARP // 2 for passo in DIVERGENTE):
    raise AssertionError(
        "la didascalia dice sedici caselle accese per riga nel caso divergente")


def costruisci() -> Figura:
    cella, spazio = 15.0, 2.0
    passo_x = cella + spazio
    x0 = 132.0
    larg_warp = LARGHEZZA_WARP * passo_x - spazio

    def riga(passo, y, due_strade):
        """Una riga di trentadue caselle: accese quelle attive, vuote le altre.

        Con `due_strade` il colore dice quale ramo sta percorrendo il thread;
        senza, tutti e trentadue percorrono lo stesso e il colore è uno solo.
        """
        fuori = []
        for i, attivo in enumerate(passo):
            x = x0 + i * passo_x
            if not attivo:
                classe = "spento"
            elif due_strade and i % 2:
                classe = "dispari"
            else:
                classe = "pari"
            fuori.append(f'<rect class="{classe}" x="{x:.1f}" y="{y:.1f}" '
                         f'width="{cella:.0f}" height="{cella:.0f}" rx="2"/>')
        return fuori

    corpo, anim = [], []
    totale = len(DIVERGENTE)              # i passi del percorso più lungo

    def blocco(y, titolo, nota, passi, due_strade):
        fuori = [f'<text class="lbl" x="{x0:.0f}" y="{y - 26:.0f}">{titolo}</text>',
                 f'<text class="lbs" x="{x0:.0f}" y="{y - 8:.0f}">{nota}</text>']
        for k, passo in enumerate(passi):
            yr = y + k * (cella + 5)
            fuori.append(f'<g style="animation:p{k} var(--d) infinite">'
                         + "".join(riga(passo, yr, due_strade)) + '</g>')
            fuori.append(f'<text class="lbs" x="{x0 - 14:.0f}" y="{yr + 12:.0f}" '
                         f'text-anchor="end">{k + 1}</text>')
        return fuori

    # I due blocchi condividono l'orologio: il passo k si accende allo stesso
    # istante in tutti e due, così il ritardo del secondo si vede.
    for k in range(totale):
        t0, _ = sosta(k, totale, tenuta=0.6)
        prima = max(t0 - 0.01, 0.0)
        anim.append(keyframes(f"p{k}", [(0.0, "opacity:0"), (prima, "opacity:0"),
                                        (t0, "opacity:1"), (100.0, "opacity:1")]))

    # la legenda, su una riga sua in cima
    ly = 30.0
    for dx, classe, testo in ((0, "pari", "una strada"),
                              (132, "dispari", "l'altra"),
                              (222, "spento", "fermo ad aspettare")):
        lx = x0 + dx
        corpo.append(f'<rect class="{classe}" x="{lx:.0f}" y="{ly - 11:.0f}" '
                     f'width="12" height="12" rx="2"/>')
        corpo.append(f'<text class="lbs" x="{lx + 18:.0f}" y="{ly:.0f}">{testo}</text>')

    ya = 92.0
    corpo += blocco(ya, "tutti e trentadue prendono la stessa strada",
                    f"{len(COERENTE)} passi", COERENTE, False)

    yb = ya + len(COERENTE) * (cella + 5) + 62
    corpo += blocco(yb, "metà vanno di qua, metà di là",
                    f"{len(DIVERGENTE)} passi: prima gli uni, poi gli altri, "
                    f"mai insieme", DIVERGENTE, True)

    alt_tot = yb + len(DIVERGENTE) * (cella + 5) + 46
    corpo.append(f'<text class="lbs" x="{x0:.0f}" y="{alt_tot - 16:.0f}">'
                 f'caselle accese: {LAVORO_COERENTE} sopra e '
                 f'{LAVORO_DIVERGENTE} sotto, lo stesso lavoro in due volte '
                 f'il tempo</text>')

    return Figura(
        larghezza=x0 + larg_warp + 20, altezza=alt_tot,
        alt="Due blocchi di caselle, una casella per ciascuno dei trentadue "
            "thread di un warp e una riga per ogni passo di esecuzione. Nel "
            "blocco di sopra tutti prendono la stessa strada: tre righe piene, "
            "trentadue caselle accese ciascuna. In quello di sotto il warp si "
            "divide a un bivio: sei righe invece di tre, e in ognuna solo "
            "sedici caselle sono accese mentre le altre sedici restano vuote, "
            "perché prima si esegue un ramo e poi l'altro. Le caselle accese "
            "sono novantasei in tutti e due i blocchi: stesso lavoro, tempo "
            "doppio.",
        corpo="".join(corpo),
        stile=f"""    .pari {{ fill:{TEAL}; }}
    .dispari {{ fill:{TERRACOTTA}; }}
    .spento {{ fill:none; stroke:{BORDER_STRONG}; stroke-width:1; }}""",
        animazioni=anim,
        durata=totale * 0.75,
        fermi="g",
    )
