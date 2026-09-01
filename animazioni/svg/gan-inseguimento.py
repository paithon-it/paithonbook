"""L'inseguimento della GAN: il falso che raggiunge il vero, e l'esperto che si arrende.

Due pannelli, e sono la stessa storia vista da due parti.

Sopra: la distribuzione dei dati veri (teal, ferma) e quella del generatore
(terracotta, che si muove). Sotto: il discriminatore ottimo, che il capitolo
ricava per $G$ fissato,

    D*(x) = p_dati(x) / (p_dati(x) + p_G(x)),

cioe' il meglio che l'esperto possa fare contro quel falsario. La curva parte
da una soglia netta (a sinistra sa che e' falso, a destra che e' vero) e finisce
piatta su 1/2: e' l'equilibrio del gioco, quello in cui l'esperto puo' solo
tirare a indovinare. Non e' un disegno di fantasia, e' la formula valutata sulle
due gaussiane disegnate sopra.
"""

import math

from paithon_svg import *

NOME = "gan-inseguimento"
TITOLO = "il falsario che raggiunge il vero"

TAPPE = 7
MU_DATI, SIGMA_DATI = 0.0, 0.55
MU_G0, SIGMA_G0 = -1.75, 1.05        # il falsario alle prime armi: fuori posto e sfocato

SU = Riquadro(x=84, y=64, larg=536, alt=168, xmin=-3.3, xmax=3.3, ymin=0.0, ymax=0.80)
GIU = Riquadro(x=84, y=300, larg=536, alt=112, xmin=-3.3, xmax=3.3, ymin=0.0, ymax=1.0)

PUNTI = 121
SOGLIA = 0.02      # sotto questa densita' totale non c'e' ne' vero ne' falso da giudicare


def normale(x, mu, sigma):
    return math.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * math.sqrt(2 * math.pi))


def parametri(i):
    """Il generatore i-esimo: si sposta verso lo zero e si stringe sul vero."""
    t = i / (TAPPE - 1)
    return (MU_G0 * (1 - t), SIGMA_G0 + (SIGMA_DATI - SIGMA_G0) * t)


def campiona():
    return [SU.xmin + k * (SU.xmax - SU.xmin) / (PUNTI - 1) for k in range(PUNTI)]


def polilinea(riq, xs, ys):
    return " ".join(f"{riq.sx(x):.1f},{riq.sy(y):.1f}" for x, y in zip(xs, ys))


def costruisci() -> Figura:
    xs = campiona()
    dati = [normale(x, MU_DATI, SIGMA_DATI) for x in xs]

    corpo, anim = [], []
    corpo.append(SU.cornice())
    corpo.append(GIU.cornice())

    # --- la linea dell'indecisione, 1/2: e' il traguardo del gioco
    y_mezzo = GIU.sy(0.5)
    corpo.append(f'<line class="mezzo" x1="{GIU.x}" y1="{y_mezzo:.1f}" '
                 f'x2="{GIU.x + GIU.larg}" y2="{y_mezzo:.1f}"/>')
    corpo.append(f'<text class="lbs" x="{GIU.x - 10}" y="{y_mezzo + 5:.0f}" '
                 f'text-anchor="end">1/2</text>')

    # --- i dati veri: fermi, sempre visibili
    corpo.append(f'<polyline class="vero" points="{polilinea(SU, xs, dati)}"/>')

    # --- le sette pose del generatore, e i sette discriminatori corrispondenti
    for i in range(TAPPE):
        mu, sigma = parametri(i)
        gen = [normale(x, mu, sigma) for x in xs]
        # D*(x) del capitolo: la miglior risposta possibile a QUESTO generatore.
        # Si disegna solo dove qualcosa c'e' davvero: nelle code il rapporto fra
        # due densita' trascurabili e' matematicamente definito ma dice il falso
        # al lettore ("certamente falso" dove non c'e' ne' vero ne' falso), ed e'
        # anche il perimetro che il testo dichiara, "sul supporto dei dati".
        xs_d, disc = [], []
        for x, d, g in zip(xs, dati, gen):
            if d + g >= SOGLIA:
                xs_d.append(x)
                disc.append(d / (d + g))

        t0, t1 = sosta(i, TAPPE, tenuta=0.5)
        tappe = [(0.0, "opacity:0")]
        if t0 > 1.0:
            tappe.append((max(t0 - 1.6, 0.01), "opacity:0"))
        tappe += [(t0, "opacity:1"), (t1, "opacity:1")]
        if i < TAPPE - 1:
            tappe += [(min(t1 + 1.6, 99.9), "opacity:0"), (100.0, "opacity:0")]
        else:
            tappe.append((100.0, "opacity:1"))
        anim.append(keyframes(f"g{i}", tappe))

        op = 1 if i == TAPPE - 1 else 0
        stile = f'opacity="{op}" style="animation:g{i} var(--d) infinite"'
        corpo.append(f'<polyline class="falso" points="{polilinea(SU, xs, gen)}" {stile}/>')
        corpo.append(f'<polyline class="disc" points="{polilinea(GIU, xs_d, disc)}" {stile}/>')

    # --- etichette
    corpo += [
        f'<text class="lbl" x="{SU.x}" y="{SU.y - 26}">le due distribuzioni</text>',
        f'<text class="lbs" x="{SU.x + SU.larg}" y="{SU.y - 26}" text-anchor="end" '
        f'style="fill:{TEAL}">dati veri</text>',
        f'<text class="lbs" x="{SU.x + SU.larg - 86}" y="{SU.y - 26}" '
        f'text-anchor="end" style="fill:{TERRACOTTA}">generatore</text>',
        f'<text class="lbl" x="{GIU.x}" y="{GIU.y - 20}">'
        f'il verdetto migliore possibile dell\'esperto</text>',
        f'<text class="lbs" x="{GIU.x}" y="{GIU.y + GIU.alt + 30}">'
        f'quando il falso copre il vero, all\'esperto non resta che tirare '
        f'a indovinare</text>',
        f'<text class="lbs" x="{GIU.x + GIU.larg}" y="{GIU.y - 20}" '
        f'text-anchor="end">solo dove ci sono dati</text>',
    ]

    return Figura(
        larghezza=680, altezza=470,
        alt="Due pannelli sovrapposti. Sopra, la campana dei dati veri sta "
            "ferma al centro mentre quella del generatore, all'inizio spostata "
            "a sinistra e più larga, si sposta e si stringe fino a coprirla. "
            "Sotto, la curva del verdetto parte a gobba, alta dove "
            "prevalgono i dati veri e bassa dove prevale il generatore, e si "
            "appiattisce fino a diventare la retta orizzontale a un mezzo; è "
            "disegnata solo nel tratto in cui almeno una delle due campane ha "
            "densità apprezzabile.",
        corpo="".join(corpo),
        stile=f"""    .vero  {{ fill:none; stroke:{TEAL}; stroke-width:6.5; opacity:0.5; }}
    .falso {{ fill:none; stroke:{TERRACOTTA}; stroke-width:2.6; }}
    .disc  {{ fill:none; stroke:{OCRA}; stroke-width:3; }}
    .mezzo {{ stroke:{BORDER_STRONG}; stroke-width:1.5; stroke-dasharray:5 5; }}""",
        animazioni=anim,
        durata=TAPPE * 1.5,
        fermi=".falso, .disc",
    )
