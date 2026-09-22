"""I pesi causali di una PINN: un istante conta solo quando quelli prima sono risolti.

Il tempo è il contenuto due volte: è l'asse del problema, ed è l'ordine in cui
l'addestramento lo risolve. I pesi di Wang, Sankaran e Perdikaris,
w_j = exp(-eps * somma_{t_k < t_j} r_k^2), spengono il residuo di un istante
finché quelli che lo precedono restano grandi, e lo accendono man mano che il
fronte degli istanti risolti avanza.

Il residuo è schematico, e la didascalia lo dice: piccolo prima del fronte,
grande dopo. Quello che la figura calcola davvero sono i pesi, con la formula
del lavoro originale; gli `assert` difendono le due cose che la figura
promette: a ogni tappa i pesi non crescono mai andando avanti nel tempo, e
sono accesi al fronte e spenti poco oltre.
"""

import math

from paithon_svg import *

NOME = "pesi-causali"
TITOLO = "i pesi causali di una PINN"

ISTANTI = 20                      # punti di collocazione, da t = 0 a t = 10
EPS = 1.0                         # la epsilon della formula
RISOLTO, APERTO = 0.01, 1.0       # residuo al quadrato prima e dopo il fronte
FRONTI = [0, 3, 6, 10]           # quanti istanti risolti a ogni tappa


def residui(fronte):
    return [RISOLTO if j < fronte else APERTO for j in range(ISTANTI)]


def pesi(r2, eps=EPS):
    somma, w = 0.0, []
    for x in r2:
        w.append(math.exp(-eps * somma))
        somma += x
    return w


TAPPE = [(residui(f), pesi(residui(f))) for f in FRONTI]
for _f, (_r, _w) in zip(FRONTI, TAPPE):
    if any(b > a + 1e-12 for a, b in zip(_w, _w[1:])):
        raise AssertionError("un peso cresce andando avanti nel tempo")
    if _f < ISTANTI:
        if _w[_f] < 0.8:
            raise AssertionError(f"fronte {_f}: il primo istante aperto è spento")
        if _f + 3 < ISTANTI and _w[_f + 3] > 0.1:
            raise AssertionError(f"fronte {_f}: tre istanti oltre il peso è ancora acceso")


def costruisci() -> Figura:
    x0, larg = 110.0, 540.0
    passo = larg / ISTANTI
    barra = passo * 0.62
    y_r, h_r = 150.0, 110.0          # base e altezza massima dei residui
    y_w, h_w = 310.0, 110.0          # base e altezza massima dei pesi
    n = len(FRONTI)
    corpo, anim = [], []
    r_fin, w_fin = TAPPE[-1]
    for j in range(ISTANTI):
        x = x0 + j * passo + (passo - barra) / 2
        # un keyframe per barra: altezza di ogni tappa, relativa a quella finale
        for nome, riga, hmax, fin, base, cls in (
                (f"r{j}", 0, h_r, r_fin[j], y_r, "res"),
                (f"w{j}", 1, h_w, w_fin[j], y_w, "peso")):
            tappe = []
            for i, (r, w) in enumerate(TAPPE):
                v = (r, w)[riga][j]
                k = v / fin
                t0, t1 = sosta(i, n, tenuta=0.6)
                tappe += [(t0, f"transform:scaleY({k:.4f})"),
                          (t1, f"transform:scaleY({k:.4f})")]
            tappe.append((100.0, "transform:scaleY(1)"))
            anim.append(keyframes(nome, tappe))
            h = fin * hmax
            corpo.append(f'<rect class="{cls}" x="{x:.1f}" y="{base - h:.1f}" '
                         f'width="{barra:.1f}" height="{h:.1f}" '
                         f'style="animation:{nome} var(--d) infinite;'
                         f'transform-box:fill-box;transform-origin:50% 100%"/>')
    for base, testo, nota in ((y_r, "residuo r²", "per istante"),
                              (y_w, "peso w", "per istante")):
        corpo.append(f'<line class="asse" x1="{x0:.0f}" y1="{base:.0f}" '
                     f'x2="{x0 + larg:.0f}" y2="{base:.0f}"/>')
        corpo.append(f'<text class="lbl" x="{x0 - 14:.0f}" y="{base - 40:.0f}" '
                     f'text-anchor="end">{testo}</text>')
        corpo.append(f'<text class="lbs" x="{x0 - 14:.0f}" y="{base - 22:.0f}" '
                     f'text-anchor="end">{nota}</text>')
    ya = y_w + 26
    corpo += [
        f'<text class="lbs" x="{x0:.0f}" y="{ya:.0f}">t = 0</text>',
        f'<text class="lbs" x="{x0 + larg:.0f}" y="{ya:.0f}" '
        f'text-anchor="end">t = 10</text>',
        f'<text class="lbs" x="{x0:.0f}" y="{ya + 26:.0f}">wⱼ = exp(−ε '
        f'· somma dei residui degli istanti precedenti), ε = '
        f'{EPS:g}</text>',
    ]
    return Figura(
        larghezza=680, altezza=ya + 44,
        alt=f"Due file di {ISTANTI} barre, una per istante di collocazione fra "
            f"t = 0 e t = 10. In alto un residuo di esempio: piccolo negli "
            f"istanti già risolti, grande in quelli dopo, con un fronte che "
            f"avanza verso destra. In basso i pesi causali calcolati con la "
            f"formula: accesi fino al fronte e spenti poco oltre, così che "
            f"l'addestramento lavori su un istante solo quando quelli prima "
            f"sono a posto. La figura si ferma con il fronte a metà "
            f"dell'intervallo.",
        corpo="".join(corpo),
        stile=f"""    .res {{ fill:{TERRACOTTA}; }}
    .peso {{ fill:{TEAL}; }}
    .asse {{ stroke:{BORDER_STRONG}; stroke-width:2; }}""",
        animazioni=anim,
        durata=len(FRONTI) * 2.0,
        fermi=".res, .peso",
    )
