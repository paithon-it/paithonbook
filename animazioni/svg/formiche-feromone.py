"""Il feromone che si accumula: perché la strada corta vince senza che nessuno la misuri.

La scena è il ponte doppio del capitolo: due strade fra formicaio e cibo, una
lunga il doppio dell'altra, formiche che vanno alla stessa velocità. Nessuna
formica confronta le due lunghezze; è il tempo a farlo, perché sulla strada
corta si passa più spesso e il deposito per passaggio è lo stesso.

I numeri li calcola questa scena, con le due formule della tab Superiore della
sezione: scelta proporzionale al feromone, deposito $Q/L$, evaporazione
$\\rho$. Non sono illustrativi: sono l'esito della simulazione.
"""

from paithon_svg import *

NOME = "formiche-feromone"
TITOLO = "il feromone che decide la strada"

CICLI = 6
Q = 10.0          # costante di scala del deposito
L_CORTA = 1.0     # dieci minuti di giro
L_LUNGA = 2.0     # venti: il doppio, come dice il testo
FORMICHE = 20
# Niente evaporazione, e non e' una semplificazione per pigrizia: a questo punto
# della sezione il lettore ha in mano **due** formule, la scelta proporzionale al
# feromone e il deposito Q/L. L'evaporazione arriva nella sezione successiva, e
# una figura che la usasse chiederebbe di dare per noto cio' che viene dopo.


def simula():
    """Feromone e ripartizione ciclo per ciclo. Restituisce una lista di dict."""
    tau_c = tau_l = 1.0
    storia = []
    for _ in range(CICLI):
        p_c = tau_c / (tau_c + tau_l)
        n_c = FORMICHE * p_c
        n_l = FORMICHE - n_c
        storia.append({"tau_c": tau_c, "tau_l": tau_l, "p_c": p_c})
        # deposito Q/L per formica: e' l'Ant System in due righe
        tau_c += n_c * Q / L_CORTA
        tau_l += n_l * Q / L_LUNGA
    return storia


# geometria: formicaio a sinistra, cibo a destra, due archi in mezzo
NIDO = (108, 214)
CIBO = (572, 214)
R_NODO = 30
# la strada lunga sale in alto e si vede che e' piu' lunga; la corta resta bassa
VIA_LUNGA = f"M {NIDO[0] + R_NODO} {NIDO[1]} Q 340 34 {CIBO[0] - R_NODO} {NIDO[1]}"
VIA_CORTA = f"M {NIDO[0] + R_NODO} {NIDO[1]} Q 340 292 {CIBO[0] - R_NODO} {NIDO[1]}"

SPESSORE_MIN, SPESSORE_MAX = 2.5, 17.0


def costruisci() -> Figura:
    storia = simula()
    tau_max = max(max(s["tau_c"], s["tau_l"]) for s in storia)

    def spessore(tau):
        return SPESSORE_MIN + (SPESSORE_MAX - SPESSORE_MIN) * tau / tau_max

    corpo, anim = [], []

    # --- le due strade: lo stato di riposo e' l'ultimo ciclo, come vuole la regola
    for chiave, via, colore, nome in (("tau_l", VIA_LUNGA, OCRA, "l"),
                                      ("tau_c", VIA_CORTA, TERRACOTTA, "c")):
        tappe = [(0.0, f"stroke-width:{spessore(storia[0][chiave]):.1f}")]
        for i, s in enumerate(storia):
            t0, t1 = sosta(i, CICLI, tenuta=0.45)
            w = spessore(s[chiave])
            tappe.append((t0, f"stroke-width:{w:.1f}"))
            tappe.append((t1, f"stroke-width:{w:.1f}"))
        tappe.append((100.0, f"stroke-width:{spessore(storia[-1][chiave]):.1f}"))
        anim.append(keyframes(f"via{nome}", tappe))
        corpo.append(
            f'<path class="via" d="{via}" stroke="{colore}" '
            f'stroke-width="{spessore(storia[-1][chiave]):.1f}" '
            f'style="animation:via{nome} var(--d) infinite"/>')

    # --- formicaio e cibo, disegnati sopra le strade. L'etichetta sta fuori dal
    # cerchio: dentro, "formicaio" e' piu' largo del cerchio che dovrebbe stare.
    for (cx, cy), etichetta in ((NIDO, "formicaio"), (CIBO, "cibo")):
        corpo.append(f'<circle class="nodo" cx="{cx}" cy="{cy}" r="{R_NODO}"/>')
        corpo.append(f'<text class="lbl" x="{cx}" y="{cy + R_NODO + 24}" '
                     f'text-anchor="middle">{etichetta}</text>')

    # --- le due etichette di percorso, ferme
    corpo.append(f'<text class="lbs" x="340" y="76" text-anchor="middle" '
                 f'style="fill:{OCRA}">strada lunga · venti minuti</text>')
    corpo.append(f'<text class="lbs" x="340" y="300" text-anchor="middle" '
                 f'style="fill:{TERRACOTTA}">strada corta · dieci minuti</text>')

    # --- il contatore: una scritta per ciclo, sovrapposte, una sola visibile.
    # Il testo non si anima in CSS, l'opacita' si': e' il modo di far
    # "cambiare" un numero senza uno script.
    for i, s in enumerate(storia):
        t0, t1 = sosta(i, CICLI, tenuta=0.45)
        prima = max(t0 - 1.2, 0.0)
        dopo = min(t1 + 1.2, 100.0)
        tappe = [(0.0, "opacity:0")]
        if prima > 0:
            tappe.append((prima, "opacity:0"))
        tappe += [(t0, "opacity:1"), (t1, "opacity:1")]
        if i < CICLI - 1:
            tappe += [(dopo, "opacity:0"), (100.0, "opacity:0")]
        else:
            tappe.append((100.0, "opacity:1"))
        anim.append(keyframes(f"cnt{i}", tappe))
        opac = 1 if i == CICLI - 1 else 0
        pct = f"{100 * s['p_c']:.0f}"
        # due <text> separati e non un <tspan> dentro l'altro: con
        # text-anchor="middle" il tspan lo centrano male i rasterizzatori, e il
        # provino e' esattamente cio' che va in stampa
        corpo.append(
            f'<text class="cnt" x="340" y="366" text-anchor="middle" '
            f'opacity="{opac}" style="animation:cnt{i} var(--d) infinite">'
            f'{pct} formiche su cento prendono la corta</text>')
        corpo.append(
            f'<text class="lbs" x="340" y="390" text-anchor="middle" '
            f'opacity="{opac}" style="animation:cnt{i} var(--d) infinite">'
            f'giro {i + 1}</text>')

    corpo.append('<text class="lbl" x="340" y="36" text-anchor="middle">'
                 'lo spessore è il feromone depositato</text>')
    corpo.append('<text class="lbs" x="340" y="416" text-anchor="middle">'
                 'nessuna formica ha misurato le due strade</text>')

    return Figura(
        larghezza=680, altezza=436,
        alt="Formicaio a sinistra e cibo a destra, uniti da due strade: una "
            "lunga che sale in alto e una corta in basso. Giro dopo giro "
            "entrambe si ispessiscono, perché su entrambe passano formiche, ma "
            "la corta molto più in fretta, e il divario fra le due cresce. Il "
            "contatore in basso dice quante formiche su cento scelgono la "
            "corta: si parte da cinquanta e si arriva a ottantadue.",
        corpo="".join(corpo),
        stile=f"""    .via  {{ fill:none; stroke-linecap:round; opacity:0.85; }}
    .nodo {{ fill:{CREAM}; stroke:{TEAL}; stroke-width:2.5; }}
    .cnt  {{ font-family:{SANS}; font-size:16px; font-weight:700; fill:{TERRACOTTA}; }}""",
        animazioni=anim,
        durata=CICLI * 1.6,
        fermi=".via, .cnt, .lbs",
    )
