"""L'interferenza di una memoria a stato fisso cresce da subito, non a soglia.

`StateSpaceModel/panorama-e-limiti.md` dice la cosa che il lettore capisce
peggio di tutte: il foglio-registro non è uno schedario con una voce per
casella, e non si riempie a una certa soglia. Ogni voce nuova si somma a
quello che c'è, e quello che si rilegge è la voce cercata più le briciole di
tutte le altre. Il testo lo scrive per esteso («l'interferenza cresce da
subito, come la radice di N su d, e intorno a N ≈ d vale quanto il valore
cercato») e tutti e due i lettori si sono fermati lì, chiedendo la stessa cosa:
un conto piccolo, con numeri veri.

La figura è quel conto. A sinistra due riletture dello stesso foglio, una con
poche voci scritte e una con tante quante le caselle: si vede quanto della
rilettura è la voce che si cercava e quanto sono briciole. A destra la curva
intera, che non ha nessun ginocchio: parte a salire dalla prima voce.

Perché una figura e non un numero in pagina: la tesi della sezione non è un
valore, è **la forma della curva**, e una curva senza ginocchio è
esattamente ciò che una frase non riesce a far vedere. Il lettore che cerca
una soglia la cerca perché se la immagina; qui guarda e non la trova.

L'esperimento è algebra lineare, non addestramento: chiavi sorteggiate sulla
sfera, valori sorteggiati, memoria $\\mathbf{S} = \\sum \\mathbf{v}_i
\\mathbf{k}_i^\\top$, rilettura $\\mathbf{S}\\mathbf{k}_j$. Nessun ottimizzatore,
nessun `argmax`, nessun ordinamento: solo somme e prodotti su medie di molte
estrazioni, quindi il disegno è lo stesso su qualunque macchina.

Ferma: qui non scorre niente, sono due grafici.
"""

import math
import re

import numpy as np

from paithon_svg import *

NOME = "interferenza-da-subito"
TITOLO = "le briciole crescono da subito, non a soglia"

# --------------------------------------------------------------------------
# L'esperimento
# --------------------------------------------------------------------------
D = 32                 # le caselle del foglio, cioè la dimensione dello stato
PROVE = 400            # quante estrazioni si mediano per ogni punto
SEME = 7
QUANTE = [1, 2, 3, 4, 6, 8, 11, 16, 22, 32, 45, 64]   # voci scritte
POCHE, PARI = 8, 32    # le due riletture disegnate a sinistra (d/4 e d)


def briciole(n: int) -> float:
    """Quanto pesano le briciole rispetto alla voce cercata, in media.

    Si scrivono `n` coppie chiave-valore nello stesso foglio, si rilegge
    presentando una delle chiavi e si misura quanto la rilettura si scosta dal
    valore che le apparteneva, in proporzione a quel valore.
    """
    rng = np.random.default_rng(SEME + n)
    tot = 0.0
    for _ in range(PROVE):
        k = rng.standard_normal((n, D))
        k /= np.linalg.norm(k, axis=1, keepdims=True)
        v = rng.standard_normal((n, D))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        s = v.T @ k                      # la memoria: una somma di prodotti esterni
        j = 0                            # si rilegge sempre la prima voce scritta
        letto = s @ k[j]
        tot += float(np.linalg.norm(letto - v[j]) / np.linalg.norm(v[j]))
    return tot / PROVE


CURVA = [(n, briciole(n)) for n in QUANTE]
MISURE = dict(CURVA)


def attesa(n: int) -> float:
    """La forma che il testo annuncia: la radice di (n-1) su d."""
    return math.sqrt((n - 1) / D)


def num(v: float, cifre: int = 2) -> str:
    return f"{v:.{cifre}f}".replace(".", ",")


# --------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------
LARG, ALT = 900, 430
MARGINE = 16.0
SIN = Riquadro(x=70, y=80, larg=300, alt=210, xmin=0, xmax=2, ymin=0, ymax=1.35)
DES = Riquadro(x=530, y=80, larg=300, alt=210,
               xmin=0, xmax=QUANTE[-1], ymin=0, ymax=1.55)
LARGO = {"lbs": (13.0, 0.54), "val": (13.0, 0.66), "ttl": (16.0, 0.62)}


def verifica() -> None:
    """Difende quello che la didascalia promette."""
    valori = [v for _, v in CURVA]

    # 1. Non c'è nessun ginocchio: si sale dalla prima voce in poi, sempre.
    assert all(b > a for a, b in zip(valori, valori[1:])), \
        "la curva non è crescente da subito: il disegno mostrerebbe una soglia"
    assert valori[0] < 0.25, \
        f"con una voce sola le briciole non sono già piccole: {valori[0]:.3f}"

    # 2. Con tante voci quante le caselle, le briciole valgono quanto la voce.
    assert 0.90 < MISURE[PARI] < 1.10, \
        f"a n = d le briciole non valgono quanto il valore cercato: {MISURE[PARI]:.3f}"

    # 3. A un quarto delle caselle si è già a metà strada, cioè il guasto è in
    #    corso molto prima che il foglio sia «pieno».
    assert 0.40 < MISURE[POCHE] < 0.60, \
        f"a n = d/4 le briciole non sono già mezze: {MISURE[POCHE]:.3f}"

    # 4. La forma è quella annunciata dal testo, non un'altra: la misura segue
    #    la radice entro il cinque per cento su tutto il tratto disegnato.
    for n, v in CURVA:
        if n == 1:
            continue
        assert abs(v - attesa(n)) < 0.05 * attesa(n) + 0.02, \
            f"a n = {n} la misura ({v:.3f}) si stacca dalla radice ({attesa(n):.3f})"

    # 5. Niente esce dai due riquadri.
    assert max(valori) < DES.ymax, "la curva esce dal riquadro di destra"
    assert MISURE[PARI] < SIN.ymax, "la barra delle briciole esce dal riquadro"


def verifica_testi(disegno: str) -> None:
    """Nessun testo esce dalla tela, nemmeno con il font di ripiego."""
    righe: dict[float, list] = {}
    for attributi, testo in re.findall(r"<text([^>]*)>([^<]*)</text>", disegno):
        if "transform=" in attributi:
            continue
        px, quanto = LARGO[re.search(r'class="(\w+)"', attributi).group(1)]
        larghezza = len(testo) * px * quanto
        x = float(re.search(r'\sx="([-\d.]+)"', attributi).group(1))
        y = float(re.search(r'\sy="([-\d.]+)"', attributi).group(1))
        ancora = re.search(r'text-anchor="(\w+)"', attributi)
        ancora = ancora.group(1) if ancora else "start"
        sinistra = {"start": x, "middle": x - larghezza / 2,
                    "end": x - larghezza}[ancora]
        assert sinistra >= MARGINE / 2, f"esce a sinistra: {testo[:40]}"
        assert sinistra + larghezza <= LARG - MARGINE, f"esce a destra: {testo[:40]}"
        righe.setdefault(y, []).append((sinistra, sinistra + larghezza, testo))
    for y, pezzi in righe.items():
        pezzi.sort()
        for (_, fine, primo), (inizio, _, dopo) in zip(pezzi, pezzi[1:]):
            assert inizio - fine >= 10.0, \
                f"a y={y:.0f} «{primo[:22]}» e «{dopo[:22]}» si toccano"


def costruisci() -> Figura:
    verifica()
    corpo = []

    # ---- sinistra: due riletture dello stesso foglio -----------------------
    corpo.append(f'<text class="ttl" x="{SIN.x:.1f}" y="{SIN.y - 34:.1f}">'
                 f'due riletture</text>')
    corpo.append(f'<text class="lbs" x="{SIN.x:.1f}" y="{SIN.y - 16:.1f}">'
                 f'un foglio da {D} caselle, riletto dopo {POCHE} voci e dopo '
                 f'{PARI}</text>')
    corpo.append(f'<text class="lbs" transform="rotate(-90 {SIN.x - 14:.1f} '
                 f'{SIN.y + SIN.alt / 2:.1f})" x="{SIN.x - 14:.1f}" '
                 f'y="{SIN.y + SIN.alt / 2:.1f}" text-anchor="middle">'
                 f'quanto pesa</text>')
    corpo.append(SIN.cornice())

    larghezza_barra = 46.0
    for i, quante in enumerate((POCHE, PARI)):
        centro = SIN.sx(0.5 + i)
        for j, (classe, altezza, nome) in enumerate(
                (("pB", 1.0, "la voce cercata"),
                 ("pA", MISURE[quante], "le briciole"))):
            x = centro + (j - 1) * larghezza_barra - 2 + j * 4
            y = SIN.sy(altezza)
            corpo.append(f'<rect class="{classe}s" x="{x:.1f}" y="{y:.1f}" '
                         f'width="{larghezza_barra:.1f}" '
                         f'height="{SIN.sy(0) - y:.1f}"/>')
            corpo.append(f'<text class="val" x="{x + larghezza_barra / 2:.1f}" '
                         f'y="{y - 8:.1f}" text-anchor="middle">'
                         f'{num(altezza)}</text>')
        corpo.append(f'<text class="lbs" x="{centro:.1f}" '
                     f'y="{SIN.sy(0) + 18:.1f}" text-anchor="middle">'
                     f'{quante} voci scritte</text>')

    for j, (classe, nome) in enumerate((("pB", "la voce che si cercava"),
                                        ("pA", "le briciole di tutte le altre"))):
        y = SIN.y + SIN.alt + 42 + j * 19
        corpo.append(f'<rect class="{classe}s" x="{SIN.x:.1f}" y="{y - 10:.1f}" '
                     f'width="15" height="12"/>')
        corpo.append(f'<text class="lbs" x="{SIN.x + 23:.1f}" y="{y:.1f}">'
                     f'{nome}</text>')

    # ---- destra: la curva intera, senza ginocchio --------------------------
    corpo.append(f'<text class="ttl" x="{DES.x:.1f}" y="{DES.y - 34:.1f}">'
                 f'tutte le riletture</text>')
    corpo.append(f'<text class="lbs" x="{DES.x:.1f}" y="{DES.y - 16:.1f}">'
                 f'media su {PROVE} estrazioni per ogni punto</text>')
    corpo.append(f'<text class="lbs" transform="rotate(-90 {DES.x - 14:.1f} '
                 f'{DES.y + DES.alt / 2:.1f})" x="{DES.x - 14:.1f}" '
                 f'y="{DES.y + DES.alt / 2:.1f}" text-anchor="middle">'
                 f'quanto pesano le briciole</text>')
    corpo.append(DES.cornice())

    # la riga a quota uno: da lì in su le briciole pesano più della voce
    corpo.append(f'<line class="pari" x1="{DES.x:.1f}" y1="{DES.sy(1):.1f}" '
                 f'x2="{DES.x + DES.larg:.1f}" y2="{DES.sy(1):.1f}"/>')
    corpo.append(f'<text class="lbs" x="{DES.x + 8:.1f}" '
                 f'y="{DES.sy(1) - 7:.1f}">quanto la voce cercata</text>')

    d = " ".join(f"{'M' if i == 0 else 'L'} {DES.sx(n):.1f} {DES.sy(v):.1f}"
                 for i, (n, v) in enumerate(CURVA))
    corpo.append(f'<path class="gcur" d="{d}"/>')
    for n, v in CURVA:
        classe = "pAp" if n == PARI else ("pBp" if n == POCHE else "pt")
        corpo.append(f'<circle class="{classe}" cx="{DES.sx(n):.1f}" '
                     f'cy="{DES.sy(v):.1f}" r="{4.5 if n in (POCHE, PARI) else 2.6}"/>')
    corpo.append(f'<line class="tick" x1="{DES.sx(PARI):.1f}" '
                 f'y1="{DES.sy(MISURE[PARI]):.1f}" x2="{DES.sx(PARI):.1f}" '
                 f'y2="{DES.sy(0):.1f}"/>')
    for n, dove in ((0, "start"), (PARI, "middle"), (QUANTE[-1], "end")):
        etichetta = {0: "0 voci", PARI: f"{PARI}, quante le caselle",
                     QUANTE[-1]: f"{QUANTE[-1]}"}[n]
        corpo.append(f'<text class="lbs" x="{DES.sx(n):.1f}" '
                     f'y="{DES.sy(0) + 18:.1f}" text-anchor="{dove}">'
                     f'{etichetta}</text>')

    corpo.append(f'<text class="lbs" x="{SIN.x:.1f}" y="{ALT - 34:.1f}">'
                 f'La curva non ha un ginocchio: sale dalla prima voce '
                 f'scritta, e a un quarto delle caselle è già a metà '
                 f'strada.</text>')
    corpo.append(f'<text class="lbs" x="{SIN.x:.1f}" y="{ALT - 14:.1f}">'
                 f'Chiavi e valori sorteggiati, memoria fatta di somme di '
                 f'prodotti esterni: nessun addestramento, solo algebra.</text>')

    disegno = "".join(corpo)
    verifica_testi(disegno)

    return Figura(
        larghezza=LARG, altezza=ALT,
        alt=f"Due grafici affiancati, sullo stesso foglio-registro da {D} "
            f"caselle. A sinistra, due riletture messe a confronto con delle "
            f"barre: dopo {POCHE} voci scritte la voce cercata pesa 1,00 e le "
            f"briciole delle altre {num(MISURE[POCHE])}; dopo {PARI} voci, "
            f"cioè tante quante le caselle, la voce cercata pesa sempre 1,00 "
            f"e le briciole {num(MISURE[PARI])}, cioè altrettanto. A destra, "
            f"la stessa misura per ogni numero di voci da 1 a {QUANTE[-1]}: "
            f"una curva che sale da {num(MISURE[1])} e attraversa la riga "
            f"orizzontale del pari poco dopo le {PARI} voci. La curva non ha "
            f"nessun gradino e nessun ginocchio: comincia a salire dalla "
            f"prima voce scritta. Due punti marcati la segnano a {POCHE} e a "
            f"{PARI} voci, e un trattino verticale scende dal secondo.",
        corpo=disegno,
        stile=f"""    .gcur {{ fill:none; stroke:{INK}; stroke-width:2.2;
            stroke-linejoin:round; }}
    .pAs  {{ fill:{TERRACOTTA}; fill-opacity:0.75; stroke:{TERRACOTTA};
            stroke-width:1.2; }}
    .pBs  {{ fill:{TEAL}; fill-opacity:0.75; stroke:{TEAL}; stroke-width:1.2; }}
    .pAp  {{ fill:{TERRACOTTA}; stroke:none; }}
    .pBp  {{ fill:{TEAL}; stroke:none; }}
    .pt   {{ fill:{INK}; stroke:none; }}
    .pari {{ stroke:{OCRA}; stroke-width:1.8; stroke-dasharray:6 4; }}
    .tick {{ stroke:{FG_MUTED}; stroke-width:1.2; stroke-dasharray:3 3; }}
    .val  {{ font-family:{SANS}; font-size:13px; font-weight:600; fill:{INK}; }}
""")
