"""Le repliche inseguono il traffico, e al salto delle nove arrivano tardi.

È la giornata simulata di `MLOps/capacita-e-costo.md`, con le stesse regole
del blocco di codice della pagina: la gobba del pomeriggio, venti minuti di
picco alle nove di sera, repliche da 8 richieste al secondo tenute all'80%, un
autoscaler che sale subito contando quelle già chiamate e scende dopo cinque
minuti di traffico basso. A sinistra la giornata intera, ferma, con la finestra
che la destra ingrandisce; a destra l'ora attorno all'evento, che si scopre
minuto per minuto.

Due strategie con lo stesso avvio da otto minuti. Quella che insegue vede il
salto alle 21:00 e chiama le repliche, che arrivano alle 21:08: in mezzo ci
sono otto minuti in cui il traffico supera la capacità. Quella col calendario
le aveva chiamate alle 20:52, otto minuti prima, e alle 21:00 il salto la trova
pronta. Gli assert
difendono quello che la didascalia promette: otto minuti sotto per la prima,
zero per la seconda, e le ore di replica della tabella della pagina (94,4 e
96,1, contando anche i minuti in cui una replica si sta avviando), così che la figura non possa raccontare una giornata diversa da quella
che il blocco stampa.

Il ritaglio che scopre la finestra avanza `linear`: con l'`ease` di default la
sua posizione non sarebbe proporzionale al tempo, e l'etichetta degli otto
minuti si accenderebbe fuori sincrono.
"""

import numpy as np

from paithon_svg import *

NOME = "repliche-in-ritardo"
TITOLO = "le repliche inseguono il traffico e al salto arrivano tardi"

MU, MIRA, AVVIO = 8.0, 0.8, 8
DA, A = 20 * 60 + 40, 21 * 60 + 40          # la finestra ingrandita, in minuti


def traffico():
    minuto = np.arange(1440)
    x = (minuto - 14 * 60) / 300
    arrivi = 12 + 48 * np.clip(1 - x**2, 0, None) ** 2
    evento = (minuto >= 21 * 60) & (minuto < 21 * 60 + 20)
    return arrivi + 30 * evento


def giornata(arrivi, calendario):
    """La stessa politica del blocco della pagina, con avvio da otto minuti."""
    servono = lambda tasso: int(np.ceil(tasso / (MIRA * MU)))
    pronte, pagate, chiamate = np.zeros(1440, dtype=int), np.zeros(1440, dtype=int), []
    n = servono(arrivi[0])
    ultime = [n] * 5
    for m in range(1440):
        n += chiamate.count(m)
        chiamate = [p for p in chiamate if p > m]
        davanti = arrivi[min(m + AVVIO, 1439)] if calendario else 0
        voglio = servono(max(arrivi[m], davanti))
        ultime = ultime[1:] + [voglio]
        if voglio > n + len(chiamate):
            chiamate += [m + AVVIO] * (voglio - n - len(chiamate))
        elif max(ultime) < n and not chiamate:
            n = max(ultime)
        pronte[m], pagate[m] = n, n + len(chiamate)
    return pronte, pagate


def scala(valori, x_di, y_di, da=0, a=1440):
    """Una spezzata a gradini: orizzontale per ogni minuto, verticale ai salti."""
    passi = [f"M{x_di(da):.1f},{y_di(valori[da]):.1f}"]
    for m in range(da + 1, a):
        if valori[m] != valori[m - 1]:
            passi.append(f"H{x_di(m):.1f}V{y_di(valori[m]):.1f}")
    passi.append(f"H{x_di(a):.1f}")
    return "".join(passi)


def costruisci() -> Figura:
    arrivi = traffico()
    (insegue, paga_i), (anticipa, paga_c) = giornata(arrivi, False), giornata(arrivi, True)
    sotto_i = arrivi > insegue * MU
    sotto_c = arrivi > anticipa * MU
    # quello che la didascalia e la tabella della pagina promettono
    assert sotto_i.sum() == 8 and sotto_c.sum() == 0
    # le ore pagate della tabella, avvii compresi
    assert round(paga_i.sum() / 60, 1) == 94.4 and round(paga_c.sum() / 60, 1) == 96.1
    buco = np.flatnonzero(sotto_i)
    assert buco[0] == 21 * 60 and buco[-1] == 21 * 60 + 7, "il buco non è 21:00-21:07"
    # col calendario le repliche si chiamano alle 20:52 e sono pronte alle 21:00
    assert anticipa[21 * 60] > anticipa[21 * 60 - 1], "il calendario non è pronto alle 21:00"
    assert anticipa[21 * 60 - 1] == insegue[21 * 60 - 1], "prima delle 21 le due scale differiscono"
    assert DA < buco[0] and buco[-1] < A

    # la giornata intera, a sinistra
    g = Riquadro(x=70, y=62, larg=250, alt=240, xmin=0, xmax=1440, ymin=0, ymax=90)
    # l'ora attorno all'evento, a destra
    z = Riquadro(x=400, y=62, larg=350, alt=240, xmin=DA, xmax=A, ymin=0, ymax=60)
    assert arrivi.max() < g.ymax and (insegue * MU).max() < g.ymax
    assert arrivi[DA:A].max() < z.ymax and (anticipa[DA:A] * MU).max() < z.ymax

    corpo, anim = [], []
    for r in (g, z):
        corpo.append(f'<line class="asse" x1="{r.x}" y1="{r.y + r.alt}" x2="{r.x + r.larg}" '
                     f'y2="{r.y + r.alt}"/><line class="asse" x1="{r.x}" y1="{r.y}" '
                     f'x2="{r.x}" y2="{r.y + r.alt}"/>')
    corpo.append(f'<text class="ttl" x="{g.x}" y="40">la giornata</text>')
    corpo.append(f'<text class="ttl" x="{z.x}" y="40">l’ora dell’evento</text>')
    for h in (0, 6, 12, 18, 24):
        corpo.append(f'<text class="lbs" x="{g.sx(h * 60):.1f}" y="{g.y + g.alt + 18}" '
                     f'text-anchor="middle">{h}</text>')
    for m, testo in ((DA, "20:40"), (21 * 60, "21:00"), (21 * 60 + 20, "21:20"), (A, "21:40")):
        corpo.append(f'<text class="lbs" x="{z.sx(m):.1f}" y="{z.y + z.alt + 18}" '
                     f'text-anchor="middle">{testo}</text>')
    for v in (0, 40, 80):
        corpo.append(f'<text class="lbs" x="{g.x - 8}" y="{g.sy(v) + 4:.1f}" text-anchor="end">{v}</text>')
    for v in (0, 20, 40, 60):
        corpo.append(f'<text class="lbs" x="{z.x - 8}" y="{z.sy(v) + 4:.1f}" text-anchor="end">{v}</text>')
    corpo.append(f'<text class="lbs" x="{g.x}" y="{g.y + g.alt + 38}">ore del giorno</text>')
    corpo.append(f'<text class="lbs" x="{z.x}" y="{z.y + z.alt + 38}">minuti attorno alle nove di sera</text>')
    corpo.append(f'<text class="lbs" x="{g.x + 6}" y="{g.y + 4}">richieste al secondo</text>')

    # a sinistra: traffico e repliche della strategia che insegue, fermi
    campioni = sorted(set(range(0, 1440, 5)) | {1259, 1260, 1279, 1280, 1439})
    linea = " ".join(f"{g.sx(m):.1f},{g.sy(arrivi[m]):.1f}" for m in campioni)
    corpo.append(f'<path class="cap" d="{scala(insegue * MU, g.sx, g.sy)}"/>')
    corpo.append(f'<polyline class="traf" points="{linea}"/>')
    corpo.append(f'<rect class="lente" x="{g.sx(DA):.1f}" y="{g.y}" '
                 f'width="{g.sx(A) - g.sx(DA):.1f}" height="{g.alt}"/>')
    corpo.append(f'<line class="rinvio" x1="{g.sx(A):.1f}" y1="{g.y + 30}" '
                 f'x2="{z.x - 4}" y2="{z.y + 30}"/>')

    # a destra, scoperte dal ritaglio che avanza
    x0, x1 = z.sx(buco[0]), z.sx(buco[-1] + 1)
    buco_d = (f'M{x0:.1f},{z.sy(insegue[buco[0]] * MU):.1f}V{z.sy(arrivi[buco[0]]):.1f}'
              f'H{x1:.1f}V{z.sy(insegue[buco[-1]] * MU):.1f}Z')
    corpo.append(f'<clipPath id="scopri"><rect id="scopri-r" style="animation:scopri var(--d) linear infinite" x="{z.x}" y="{z.y - 10}" '
                 f'width="{z.larg}" height="{z.alt + 12}"/></clipPath>')
    corpo.append(f'<g clip-path="url(#scopri)">'
                 f'<path class="buco" d="{buco_d}"/>'
                 f'<path class="cal" d="{scala(anticipa * MU, z.sx, z.sy, DA, A)}"/>'
                 f'<path class="cap" d="{scala(insegue * MU, z.sx, z.sy, DA, A)}"/>'
                 f'<path class="traf" d="{scala(arrivi, z.sx, z.sy, DA, A)}"/></g>')

    # le etichette si accendono quando il ritaglio arriva al punto che nominano
    quando = lambda m: 8 + 80 * (m - DA) / (A - DA)
    def accendi(nome, m, contenuto):
        t = quando(m)
        anim.append(keyframes(nome, [(0.0, "opacity:0"), (t - 0.5, "opacity:0"),
                                     (t, "opacity:1"), (100.0, "opacity:1")]))
        corpo.append(f'<g class="acceso" style="animation:{nome} var(--d) infinite">{contenuto}</g>')
    accendi("vede", 21 * 60,
            f'<text class="lbc" x="{z.sx(21 * 60) + 4:.1f}" y="{z.sy(anticipa[1260] * MU) - 8:.1f}">'
            f'col calendario</text>')
    accendi("manca", buco[-1] + 1,
            f'<text class="lbb" x="{x0 - 8:.1f}" y="{z.sy(31):.1f}" text-anchor="end">otto minuti</text>'
            f'<text class="lbb" x="{x0 - 8:.1f}" y="{z.sy(31) + 16:.1f}" text-anchor="end">'
            f'scoperti</text>')
    accendi("insegue", buco[-1] + 1,
            f'<text class="lbi" x="{x1 + 8:.1f}" y="{z.sy(insegue[buco[-1] + 1] * MU) + 18:.1f}">'
            f'chi insegue</text>')
    anim.append(keyframes("scopri", [(0.0, "transform:scaleX(0)"), (8.0, "transform:scaleX(0)"),
                                     (88.0, "transform:scaleX(1)"), (100.0, "transform:scaleX(1)")]))

    return Figura(
        larghezza=780, altezza=360,
        alt="Due grafici affiancati. A sinistra, fermo, una giornata intera: la curva "
            "nera del traffico sale nel pomeriggio fino a sessanta richieste al secondo "
            "e ridiscende, e una scala teal, la capacità delle repliche accese, la segue "
            "da sopra con un margine; un rettangolo stretto segna l'ora attorno alle nove di sera. A "
            "destra quell'ora ingrandita si scopre da sinistra a destra: alle 21:00 il "
            "traffico salta di colpo a oltre quaranta richieste al secondo, la scala "
            "teal di chi insegue il traffico sale solo alle 21:08 e lascia sotto di sé "
            "una fascia terracotta, otto minuti scoperti in cui la capacità non basta, mentre una scala "
            "ocra tratteggiata, col calendario, sale nello stesso minuto del traffico, "
            "perché le sue repliche erano state chiamate otto minuti prima, e copre il salto.",
        corpo="".join(corpo),
        stile=f"""    .asse  {{ stroke:{BORDER_STRONG}; stroke-width:1.2; }}
    .traf  {{ fill:none; stroke:{INK}; stroke-width:2; stroke-linejoin:round; }}
    .cap   {{ fill:none; stroke:{TEAL}; stroke-width:2.2; }}
    .cal   {{ fill:none; stroke:{OCRA}; stroke-width:2.4; stroke-dasharray:6 4; }}
    .buco  {{ fill:{TERRACOTTA}; opacity:.55; }}
    .lente {{ fill:{OCRA}; opacity:.35; stroke:{INK}; stroke-width:.8; }}
    .rinvio {{ stroke:{FG_MUTED}; stroke-width:1; stroke-dasharray:2 3; }}
    .lbb   {{ font-family:{SANS}; font-size:13px; font-weight:700; fill:{TERRACOTTA}; }}
    .lbc   {{ font-family:{SANS}; font-size:13px; fill:{INK}; }}
    .lbi   {{ font-family:{SANS}; font-size:13px; fill:{TEAL}; }}
    #scopri-r {{ transform-box:fill-box; transform-origin:0% 50%; }}""",
        animazioni=anim,
        durata=9.0,
        fermi="#scopri-r, .acceso",
    )
