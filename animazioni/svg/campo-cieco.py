"""Il campo visivo di una PixelCNN cresce con la profondità, il punto cieco no.

È la figura animata del capitolo `Verosimiglianza esatta`, e si muove perché
qui **il tempo è il contenuto**: la tesi della sezione non è «ecco il campo
visivo», è «il campo cresce, e a un certo punto smette di crescere in una
direzione sola». Un disegno fermo può mostrare un campo o l'altro; solo la
crescita fa vedere che a un certo punto il triangolo di destra si ferma e
resta lì per sempre, che è l'unica cosa da ricordare.

I quadretti accesi **non sono disegnati a mano**: escono dalla stessa maschera
del blocco di codice della sezione, propagata strato per strato con la somma
di Minkowski degli spostamenti ammessi. Un `assert` confronta il risultato con
i numeri che il testo stampa (0 pixel del futuro sempre, 6 pixel ciechi da 12
strati in poi): se un giorno il codice della pagina cambiasse maschera, la
figura non si genera invece di contraddirlo in silenzio.

Lo stato di riposo è l'ultimo, come vuole la regola del motore: il campo
saturo e il triangolo cieco ancora vuoto. Chi non anima (la stampa, il PDF,
`prefers-reduced-motion`) vede quindi la conclusione, che è anche la cosa che
la sezione commenta.
"""

from paithon_svg import (BORDER, BORDER_STRONG, CREAM, FG_MUTED, INK, OCRA,
                         SANS, TEAL, TERRACOTTA, Figura, keyframes, sosta)

NOME = "campo-cieco"
TITOLO = "il campo visivo di una PixelCNN, e il punto cieco che resta"

N = 9                    # lato della griglia, come nel codice della sezione
RIF = (8, 4)             # il pixel da indovinare, stessa posizione del codice
PROFONDITA = [1, 2, 3, 4, 6, 9, 14, 24]

# Gli spostamenti che una maschera 3x3 lascia passare, in (riga, colonna)
# relative. Il tipo A (primo strato) esclude il centro; il tipo B lo include.
A = {(-1, -1), (-1, 0), (-1, 1), (0, -1)}
B = A | {(0, 0)}


def campo(strati: int) -> set[tuple[int, int]]:
    """Quali celle influenzano RIF dopo `strati` convoluzioni mascherate.

    Composizione di maschere = somma di Minkowski degli spostamenti: una A per
    il primo strato, una B per ciascuno dei successivi. Si taglia alla griglia
    perché fuori dai bordi non c'è niente da guardare.
    """
    ins = set(A)
    for _ in range(strati - 1):
        ins = {(r + dr, c + dc) for (r, c) in ins for (dr, dc) in B}
    celle = {(RIF[0] + r, RIF[1] + c) for (r, c) in ins}
    return {(r, c) for (r, c) in celle if 0 <= r < N and 0 <= c < N}


def prima(cella: tuple[int, int]) -> bool:
    """Viene prima di RIF nell'ordine di lettura?"""
    return cella[0] * N + cella[1] < RIF[0] * N + RIF[1]


PASSATO = {(r, c) for r in range(N) for c in range(N) if prima((r, c))}
CAMPI = [campo(k) for k in PROFONDITA]

# I controlli che tengono la figura ancorata al testo della sezione.
for k, vis in zip(PROFONDITA, CAMPI):
    assert not (vis - PASSATO), f"{k} strati: la maschera guarda nel futuro"
assert len(PASSATO - campo(12)) == 6, "il testo dice 6 pixel ciechi con 12 strati"
assert len(PASSATO - campo(24)) == 6, "il punto cieco non deve chiudersi mai"


def costruisci() -> Figura:
    lato, x0, y0 = 34, 60, 62
    corpo, anim = [], []
    nomi = {}

    def pattern(stati: list[bool]) -> str:
        """Un @keyframes per ogni sequenza acceso/spento distinta."""
        chiave = tuple(stati)
        if chiave not in nomi:
            nome = f"c{len(nomi)}"
            nomi[chiave] = nome
            tappe = [(0.0, f"opacity:{1 if stati[0] else 0}")]
            for k, acceso in enumerate(stati):
                t0, t1 = sosta(k, len(PROFONDITA), tenuta=0.80)
                d = f"opacity:{1 if acceso else 0}"
                tappe += [(max(t0 - 0.6, 0.01), d), (t0, d), (t1, d)]
            tappe.append((100.0, f"opacity:{1 if stati[-1] else 0}"))
            tappe.sort(key=lambda t: t[0])
            anim.append(keyframes(nome, tappe))
        return nomi[chiave]

    for r in range(N):
        for c in range(N):
            x, y = x0 + c * lato, y0 + r * lato
            cella = (r, c)
            if cella == RIF:
                corpo.append(f'<rect class="q rif" x="{x}" y="{y}" '
                             f'width="{lato - 3}" height="{lato - 3}" rx="3"/>')
                continue
            if cella not in PASSATO:                       # il futuro, mai
                corpo.append(f'<rect class="q fut" x="{x}" y="{y}" '
                             f'width="{lato - 3}" height="{lato - 3}" rx="3"/>')
                continue
            stati = [cella in vis for vis in CAMPI]
            if not any(stati):                             # il punto cieco
                corpo.append(f'<rect class="q cieco" x="{x}" y="{y}" '
                             f'width="{lato - 3}" height="{lato - 3}" rx="3"/>')
                continue
            corpo.append(f'<rect class="q pas" x="{x}" y="{y}" '
                         f'width="{lato - 3}" height="{lato - 3}" rx="3"/>')
            corpo.append(f'<rect class="q vis" x="{x}" y="{y}" '
                         f'width="{lato - 3}" height="{lato - 3}" rx="3" '
                         f'opacity="{1 if stati[-1] else 0}" '
                         f'style="animation:{pattern(stati)} var(--d) infinite"/>')

    # Il contatore di profondità, con lo stesso ritmo dei quadretti.
    passo = 100.0 / len(PROFONDITA)
    for k, strati in enumerate(PROFONDITA):
        t0, _ = sosta(k, len(PROFONDITA))
        anim.append(keyframes(f"p{k}", [
            (0.0, "opacity:0"), (max(t0 - 0.4, 0.01), "opacity:0"), (t0, "opacity:1"),
            (min(t0 + passo - 0.4, 99.9), "opacity:1"),
            (min(t0 + passo, 100.0), "opacity:0"), (100.0, "opacity:0")]))
        fermo = ";opacity:1" if k == len(PROFONDITA) - 1 else ""
        corpo.append(f'<text class="pro" x="{x0}" y="42" '
                     f'style="animation:p{k} var(--d) infinite{fermo}">'
                     f'{strati} strat{"o" if strati == 1 else "i"}</text>')

    xl, yl = x0 + N * lato + 34, y0 + 8
    voci = [("vis", "già guardato"), ("rif", "il pixel da indovinare"),
            ("cieco", "punto cieco: viene prima e non lo vedrà mai"),
            ("fut", "viene dopo: giusto non guardarlo")]
    for i, (cls, testo) in enumerate(voci):
        y = yl + i * 46
        corpo.append(f'<rect class="q {cls}" x="{xl}" y="{y}" width="20" '
                     f'height="20" rx="3"/>')
        corpo.append(f'<text class="lbs" x="{xl + 30}" y="{y + 15}">{testo}</text>')

    corpo.append(f'<text class="lbs" x="{x0}" y="{y0 + N * lato + 26}">'
                 f'ordine di lettura: riga per riga, da sinistra a destra</text>')

    return Figura(
        larghezza=736, altezza=y0 + N * lato + 46,
        alt="Una griglia di nove per nove quadretti con il pixel da indovinare "
            "in basso al centro. Al crescere della profondità della rete i "
            "quadretti che entrano nel campo visivo si accendono, partendo dai "
            "vicini e allargandosi verso l'alto a sinistra; i quadretti che "
            "vengono dopo nell'ordine di lettura restano spenti, come è giusto, "
            "e un triangolo di quadretti in alto a destra resta spento pur "
            "venendo prima: è il punto cieco, e non si accende nemmeno con "
            "ventiquattro strati.",
        corpo="".join(corpo),
        stile=f"""    .q     {{ stroke-width:1.5; }}
    .pas   {{ fill:{CREAM}; stroke:{BORDER}; }}
    .vis   {{ fill:{TEAL}; stroke:{TEAL}; }}
    .rif   {{ fill:{TERRACOTTA}; stroke:{TERRACOTTA}; }}
    .cieco {{ fill:none; stroke:{OCRA}; stroke-width:2.5; stroke-dasharray:4 3; }}
    .fut   {{ fill:none; stroke:{BORDER_STRONG}; stroke-dasharray:2 4; }}
    .pro   {{ font-family:{SANS}; font-size:17px; font-weight:700;
             fill:{INK}; opacity:0; }}""",
        animazioni=anim,
        durata=len(PROFONDITA) * 1.5,
        fermi=".vis, .pro",
    )
