"""L'algoritmo di Euclide: la coppia scende, e il resto zero dice che è finita.

È la prima figura animata del capitolo `Introduzione`, e il motivo per cui si
muove è che qui **il tempo è il contenuto**: l'algoritmo non è una tabella di
tre righe, è una discesa, e ogni riga nasce da quella sopra. Le due frecce
dicono l'unica regola che serve ricordare (il divisore scala a dividendo, il
resto scala a divisore); il fermo immagine può solo elencare le tappe già
scese.

I numeri **non sono scritti a mano**: la traccia esce da `euclide()`, che è lo
stesso ciclo del notebook `Introduzione/massimo_comune_divisore.ipynb`
(`while b: a, b = b, a % b`) applicato agli stessi 60 e 48. Un `assert` la
confronta con le coppie che il libro svolge a parole e con `math.gcd`: se un
giorno il testo cambiasse numeri, la figura non si genera invece di
contraddirlo in silenzio.

Lo stato di riposo è l'ultimo, come vuole la regola del motore: la coppia
(12, 0) accesa, il 12 in terracotta come risultato, e le due tappe precedenti
ancora leggibili in tono tenue. Chi non anima (la stampa, il PDF,
`prefers-reduced-motion`) vede quindi la discesa **intera**, non solo il suo
esito.
"""

import math

from paithon_svg import *

NOME = "euclide-scende"
TITOLO = "l'algoritmo di Euclide, una divisione per volta"

A0, B0 = 60, 48          # gli stessi numeri del notebook della pagina che segue

# Le coppie che il testo svolge a parole. Non servono a disegnare (le disegna
# ciò che l'algoritmo produce): servono a fermare la figura se un giorno il
# libro cambiasse esempio senza che questo file lo sappia.
ATTESE = [(60, 48), (48, 12), (12, 0)]

# --------------------------------------------------------------------------
# Geometria: tre colonne a passo costante, così le frecce sono parallele e
# «scalare di un posto» si vede prima di leggerlo.
# --------------------------------------------------------------------------
W, H = 680, 356
XA, XB, XR = 96, 214, 332       # dividendo, divisore, resto
BW, BH = 66, 40                 # le scatole dei numeri
Y0, PITCH = 62, 92
# La colonna di destra è allineata a sinistra, e non è un capriccio: un
# `<tspan>` dentro un testo con `text-anchor="end"` cairosvg non lo disegna
# affatto (il resto in terracotta spariva dal provino, cioè dalla stampa).
X_DES = 462
Y_LEG = 316                     # la legenda, sotto il disegno

TEN, FADE = 0.62, 0.18          # tenuta e dissolvenza, in frazioni di passo
TENUE = 0.42                    # quanto resta visibile una tappa già scesa


# --------------------------------------------------------------------------
# L'algoritmo, che gira davvero
# --------------------------------------------------------------------------
def euclide(a: int, b: int):
    """Le divisioni `(a, b, quoziente, resto)` e le coppie che attraversa.

    È il ciclo del notebook, aperto: `divmod` dà in un colpo il quoziente (che
    la figura scrive, per far vedere *perché* quel resto) e il resto (che è
    l'unica cosa che al passo dopo conta).
    """
    divisioni, coppie = [], [(a, b)]
    while b:
        q, r = divmod(a, b)
        divisioni.append((a, b, q, r))
        a, b = b, r
        coppie.append((a, b))
    return divisioni, coppie, a


def costruisci() -> Figura:
    divisioni, coppie, mcd = euclide(A0, B0)
    assert coppie == ATTESE, f"la discesa non è più quella del testo: {coppie}"
    assert mcd == math.gcd(A0, B0) == 12, f"il MCD non torna: {mcd}"

    n = len(coppie) + 1          # una tappa per riga, più il verdetto finale
    passo = 100.0 / n
    corpo, anim = [], []

    # ---- visibilità: ogni gruppo entra al suo stato e poi si attenua -------
    def entra(nome: str, k: int, dopo: float | None, tenuta: float = TEN) -> float:
        """`@keyframes` di un gruppo che compare allo stato k.

        `dopo` è l'opacità che prende quando arriva lo stato successivo, e
        `None` vuol dire «resta acceso»: il valore al 100% è anche quello che
        l'elemento porta come attributo, così riposo e fine dell'animazione
        non possono divergere.
        """
        t = k * passo
        tappe = [(0.0, "opacity:1" if k == 0 else "opacity:0")]
        if k:
            tappe.append((t - passo * FADE, "opacity:0"))
            tappe.append((t, "opacity:1"))
        if dopo is None:
            tappe.append((100.0, "opacity:1"))
        else:
            tappe.append((t + passo * tenuta, "opacity:1"))
            tappe.append((t + passo * (tenuta + FADE), f"opacity:{dopo}"))
            tappe.append((100.0, f"opacity:{dopo}"))
        anim.append(keyframes(nome, tappe))
        return 1.0 if dopo is None else dopo

    def gruppo(dentro: str, k: int, dopo: float | None, tenuta: float = TEN) -> str:
        nome = f"eu{len(anim)}"
        op = entra(nome, k, dopo, tenuta)
        return (f'<g class="tappa" opacity="{op:g}" '
                f'style="animation:{nome} var(--d) infinite">{dentro}</g>')

    # ---- mattoni ----------------------------------------------------------
    def scatola(x: float, y: float, valore, stile: str) -> str:
        return (f'<rect class="box {stile}" x="{x:g}" y="{y:g}" '
                f'width="{BW}" height="{BH}" rx="7"/>'
                f'<text class="num {stile}" x="{x + BW / 2:g}" y="{y + 27:g}" '
                f'text-anchor="middle">{valore}</text>')

    def freccia(x1: float, y1: float, x2: float, y2: float, punta: str) -> str:
        return (f'<line class="fre {punta}" x1="{x1:g}" y1="{y1:g}" '
                f'x2="{x2:g}" y2="{y2:g}" marker-end="url(#eu-{punta})"/>')

    corpo.append(
        '<defs>'
        + "".join(
            f'<marker id="eu-{sigla}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{colore}"/></marker>'
            for sigla, colore in (("cop", TEAL), ("res", TERRACOTTA)))
        + '</defs>')

    # ---- intestazioni delle colonne ---------------------------------------
    for x, testo in ((XA, "dividendo"), (XB, "divisore"), (XR, "resto")):
        corpo.append(f'<text class="lbs" x="{x + BW / 2:g}" y="44" '
                     f'text-anchor="middle">{testo}</text>')
    corpo.append(f'<text class="lbs" x="{X_DES}" y="44">la divisione</text>')

    # ---- le tappe, una riga per coppia ------------------------------------
    ultima = len(coppie) - 1
    for k, (a, b) in enumerate(coppie):
        y = Y0 + k * PITCH
        riga = []

        # le due frecce che portano qui: ognuno scala di un posto a sinistra
        if k:
            y_su = y - PITCH + BH + 6
            riga.append(freccia(XB + BW / 2, y_su, XA + BW / 2, y - 8, "cop"))
            riga.append(freccia(XR + BW / 2, y_su, XB + BW / 2, y - 8, "res"))

        if k == ultima:
            # il divisore è zero: il ciclo si ferma, e il dividendo è la risposta
            riga.append(scatola(XB, y, b, "zer"))
            riga.append(f'<text class="nota" x="{X_DES}" y="{y + 14:g}">'
                        f'il divisore è 0: si ferma</text>')
        else:
            _, _, q, r = divisioni[k]
            riga.append(scatola(XA, y, a, "cop"))
            riga.append(scatola(XB, y, b, "cop"))
            riga.append(scatola(XR, y, r, "res"))
            riga.append(f'<text class="div" x="{X_DES}" y="{y + 26:g}">'
                        f'{a} = {q} × {b} + <tspan class="ter">{r}</tspan></text>')

        corpo.append(gruppo("".join(riga), k, None if k == ultima else TENUE))

    # ---- il verdetto: lo stesso 12, che da numero diventa risposta ---------
    y_fine = Y0 + ultima * PITCH
    corpo.append(gruppo(scatola(XA, y_fine, mcd, "cop"), ultima, 0.0, tenuta=1.0))
    corpo.append(gruppo(scatola(XA, y_fine, mcd, "esi"), n - 1, None))
    corpo.append(gruppo(
        f'<text class="esito" x="{X_DES}" y="{y_fine + 38:g}">'
        f'MCD({A0}, {B0}) = {mcd}</text>', n - 1, None))

    # ---- la legenda, che vale a figura ferma -------------------------------
    corpo.append(f'<text class="lbl" x="40" y="{Y_LEG}">A ogni passo si scala '
                 f'di un posto: <tspan class="tea">il divisore</tspan> diventa '
                 f'dividendo,</text>')
    corpo.append(f'<text class="lbl" x="40" y="{Y_LEG + 22}">'
                 f'<tspan class="ter">il resto</tspan> diventa divisore. Quando '
                 f'il divisore è zero, il MCD è a sinistra.</text>')

    return Figura(
        larghezza=W, altezza=H,
        alt=f"L'algoritmo di Euclide applicato a {A0} e {B0}. Ogni riga è una "
            f"coppia di numeri: {A0} diviso {B0} dà resto 12, e due frecce "
            f"portano il divisore al posto del dividendo e il resto al posto "
            f"del divisore, così la coppia diventa (48, 12). La divisione "
            f"seguente, 48 = 4 × 12 + 0, dà resto zero: la coppia diventa "
            f"(12, 0), il ciclo si ferma e il {mcd} rimasto a sinistra è il "
            f"massimo comune divisore.",
        corpo="".join(corpo),
        stile=f"""    rect.box  {{ fill:none; stroke:{BORDER_STRONG}; stroke-width:1.6; }}
    rect.cop  {{ stroke:{TEAL}; stroke-width:2.6; }}
    rect.res  {{ stroke:{TERRACOTTA}; stroke-width:2.6; }}
    rect.esi  {{ fill:{TERRACOTTA}; stroke:{TERRACOTTA}; stroke-width:2.6; }}
    text.num  {{ font-family:{SANS}; font-size:20px; font-weight:600; fill:{INK}; }}
    text.zer  {{ fill:{FG_MUTED}; }}
    text.esi  {{ fill:{CREAM}; }}
    .fre      {{ fill:none; stroke-width:1.8; }}
    .fre.cop  {{ stroke:{TEAL}; }}
    .fre.res  {{ stroke:{TERRACOTTA}; }}
    .div      {{ font-family:{SANS}; font-size:15px; fill:{FG_MUTED}; }}
    .nota     {{ font-family:{SANS}; font-size:14px; fill:{FG_MUTED}; }}
    .esito    {{ font-family:{SANS}; font-size:18px; font-weight:700; fill:{TERRACOTTA}; }}
    .ter      {{ fill:{TERRACOTTA}; font-weight:600; }}
    .tea      {{ fill:{TEAL}; font-weight:600; }}""",
        animazioni=anim,
        durata=n * 2.2,
        fermi=".tappa",
    )
