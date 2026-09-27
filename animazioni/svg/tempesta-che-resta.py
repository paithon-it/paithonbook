"""Dieci secondi di guasto: chi non ritenta si riprende, chi ritenta resta giù.

È la simulazione di `MLOps/gateway-e-affidabilita.md`, con gli stessi numeri
del blocco di codice della pagina: un modello che smaltisce 100 tentativi al
secondo, 80 richieste nuove al secondo, clienti che dopo due secondi lasciano
perdere, dieci secondi (dal trentesimo al quarantesimo) in cui il modello non
risponde. Si disegnano, secondo per secondo, le risposte utili delle due
politiche estreme e il carico che quella con i ritentativi offre al modello.

Gli assert difendono la didascalia: prima del guasto le due politiche danno
ottanta risposte utili al secondo; senza ritentativi dopo il guasto il servizio
torna a ottanta entro un minuto; con tre ritentativi le risposte utili restano
a zero fino in fondo, mentre il carico offerto si assesta sopra i trecento
tentativi al secondo, cioè più del triplo della capacità, con il guasto finito
da un pezzo.

Il ritaglio che scopre il tempo avanza `linear`, perché le etichette si
accendono quando il ritaglio arriva al secondo che nominano.
"""

from collections import deque

from paithon_svg import *

NOME = "tempesta-che-resta"
TITOLO = "dopo dieci secondi di guasto, chi ritenta non si riprende più"

CAPACITA, ARRIVI, PAZIENZA, PASSO = 100.0, 80.0, 2.0, 0.1
GUASTO = (30.0, 40.0)
FINE = 150


def simula(ritenta):
    """La stessa simulazione della pagina, che in più conta i tentativi in arrivo."""
    coda = deque()
    utili, offerti = [0.0] * FINE, [0.0] * FINE
    for k in range(FINE * 10):
        ora = k * PASSO
        coda.append([ora, ARRIVI * PASSO, 0, False])
        offerti[int(ora)] += ARRIVI * PASSO
        nuovi = []
        for g in coda:
            if not g[3] and ora - g[0] > PAZIENZA:
                g[3] = True
                if g[2] < ritenta:
                    nuovi.append([ora, g[1], g[2] + 1, False])
                    offerti[int(ora)] += g[1]
        coda.extend(nuovi)
        lavoro = 0.0 if GUASTO[0] <= ora < GUASTO[1] else CAPACITA * PASSO
        while lavoro > 1e-9 and coda:
            g = coda[0]
            fatto = min(g[1], lavoro)
            lavoro -= fatto
            g[1] -= fatto
            if not g[3]:
                utili[int(ora)] += fatto
            if g[1] <= 1e-9:
                coda.popleft()
    return utili, offerti


def costruisci() -> Figura:
    senza, _ = simula(0)
    con, offerti = simula(3)
    # la didascalia
    assert all(abs(v - 80) < 1e-6 for v in senza[:30] + con[:30]), "prima del guasto non sono 80"
    ripresa = next(s for s in range(40, FINE) if senza[s] > 79)
    assert ripresa - 40 <= 60, f"senza ritentativi si riprende dopo {ripresa - 40} s"
    assert all(v == 0 for v in con[40:]), "con i ritentativi le risposte utili non restano a zero"
    assert min(offerti[60:]) > 300, "il carico offerto non resta sopra i trecento"
    assert max(offerti) < 360

    r = Riquadro(x=80, y=40, larg=620, alt=260, xmin=0, xmax=FINE, ymin=0, ymax=360)
    corpo, anim = [], []
    corpo.append(f'<rect class="guasto" x="{r.sx(GUASTO[0]):.1f}" y="{r.y}" '
                 f'width="{r.sx(GUASTO[1]) - r.sx(GUASTO[0]):.1f}" height="{r.alt}"/>')
    corpo.append(f'<text class="lbs" x="{r.sx(35):.1f}" y="{r.y + r.alt + 36}" '
                 f'text-anchor="middle">guasto</text>')
    corpo.append(f'<line class="asse" x1="{r.x}" y1="{r.y + r.alt}" x2="{r.x + r.larg}" '
                 f'y2="{r.y + r.alt}"/><line class="asse" x1="{r.x}" y1="{r.y}" '
                 f'x2="{r.x}" y2="{r.y + r.alt}"/>')
    for s in (0, 30, 60, 90, 120, 150):
        corpo.append(f'<text class="lbs" x="{r.sx(s):.1f}" y="{r.y + r.alt + 18}" '
                     f'text-anchor="middle">{s}</text>')
    corpo.append(f'<text class="lbs" x="{r.x + r.larg}" y="{r.y + r.alt + 36}" '
                 f'text-anchor="end">secondi</text>')
    for v in (0, 100, 200, 300):
        corpo.append(f'<text class="lbs" x="{r.x - 8}" y="{r.sy(v) + 4:.1f}" text-anchor="end">{v}</text>')
    corpo.append(f'<text class="lbs" x="{r.x + 6}" y="{r.y + 14}">al secondo</text>')
    corpo.append(f'<line class="cap" x1="{r.x}" y1="{r.sy(CAPACITA):.1f}" x2="{r.x + r.larg}" '
                 f'y2="{r.sy(CAPACITA):.1f}"/>')
    corpo.append(f'<text class="lbc" x="{r.x + r.larg + 6}" y="{r.sy(CAPACITA) + 4:.1f}">capacità</text>')

    # una spezzata a gradini per secondo: ogni valore vale per tutto il suo secondo
    gradini = lambda serie: "M" + "".join(
        f"{r.sx(s):.1f},{r.sy(v):.1f} {r.sx(s + 1):.1f},{r.sy(v):.1f} " for s, v in enumerate(serie))
    corpo.append(f'<clipPath id="scorre"><rect id="scorre-r" style="animation:scorre var(--d) linear infinite" x="{r.x}" y="{r.y - 4}" '
                 f'width="{r.larg}" height="{r.alt + 8}"/></clipPath>')
    corpo.append(f'<g clip-path="url(#scorre)">'
                 f'<path class="offerti" d="{gradini(offerti)}"/>'
                 f'<path class="senza" d="{gradini(senza)}"/>'
                 f'<path class="con" d="{gradini(con)}"/></g>')

    quando = lambda s: 8 + 80 * s / FINE
    def accendi(nome, s, contenuto):
        t = quando(s)
        anim.append(keyframes(nome, [(0.0, "opacity:0"), (t - 0.5, "opacity:0"),
                                     (t, "opacity:1"), (100.0, "opacity:1")]))
        corpo.append(f'<g class="acceso" style="animation:{nome} var(--d) infinite">{contenuto}</g>')
    accendi("tentativi", 50,
            f'<text class="lbt" x="{r.sx(52):.1f}" y="{r.sy(offerti[60]) - 10:.1f}">'
            f'tentativi in arrivo, con tre ritentativi</text>')
    accendi("riprende", ripresa + 14,
            f'<text class="lbn" x="{r.sx(ripresa + 14):.1f}" y="{r.sy(80) + 18:.1f}">'
            f'risposte utili, senza ritentativi</text>')
    accendi("resta", 100,
            f'<text class="lbr" x="{r.sx(100):.1f}" y="{r.sy(0) - 10:.1f}">'
            f'risposte utili, con tre ritentativi</text>')
    anim.append(keyframes("scorre", [(0.0, "transform:scaleX(0)"), (8.0, "transform:scaleX(0)"),
                                     (88.0, "transform:scaleX(1)"), (100.0, "transform:scaleX(1)")]))

    return Figura(
        larghezza=780, altezza=350,
        alt="Un grafico che si scopre da sinistra a destra lungo centocinquanta "
            "secondi, con una fascia ocra fra il trentesimo e il quarantesimo, il "
            "guasto, e una linea tratteggiata nera a cento, la capacità. Prima del "
            "guasto una linea teal e una terracotta stanno sovrapposte a ottanta "
            "risposte utili al secondo. Durante il guasto scendono entrambe a zero. "
            "Dopo, la teal, senza ritentativi, resta a zero per una trentina di "
            "secondi e poi torna a ottanta; la terracotta, con tre ritentativi, resta "
            "a zero fino in fondo, mentre una linea terracotta tratteggiata, i "
            "tentativi che arrivano al modello, sale oltre i trecento al secondo e "
            "non scende più, più del triplo della capacità.",
        corpo="".join(corpo),
        stile=f"""    .asse    {{ stroke:{BORDER_STRONG}; stroke-width:1.2; }}
    .guasto  {{ fill:{OCRA}; opacity:.3; }}
    .cap     {{ stroke:{INK}; stroke-width:1.4; stroke-dasharray:6 5; }}
    .offerti {{ fill:none; stroke:{TERRACOTTA}; stroke-width:2; stroke-dasharray:5 4; }}
    .senza   {{ fill:none; stroke:{TEAL}; stroke-width:2.6; }}
    .con     {{ fill:none; stroke:{TERRACOTTA}; stroke-width:2.6; }}
    .lbc     {{ font-family:{SANS}; font-size:13px; fill:{INK}; }}
    .lbt     {{ font-family:{SANS}; font-size:13px; fill:{TERRACOTTA}; }}
    .lbr     {{ font-family:{SANS}; font-size:13px; font-weight:700; fill:{TERRACOTTA}; }}
    .lbn     {{ font-family:{SANS}; font-size:13px; font-weight:700; fill:{TEAL}; }}
    #scorre-r {{ transform-box:fill-box; transform-origin:0% 50%; }}""",
        animazioni=anim,
        durata=9.0,
        fermi="#scorre-r, .acceso",
    )
