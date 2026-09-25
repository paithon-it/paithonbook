"""I tre pezzi che tengono una JEPA lontana dal collasso, e quale a volte si toglie.

Figura ferma, un diagramma. Due rami: il contesto passa per l'encoder e per il
predictor, il bersaglio per una copia dell'encoder aggiornata come media
mobile (EMA) e mai toccata dal gradiente. La perdita confronta i due embedding.
La figura marca i tre pezzi dell'asimmetria e dice quale regge il muro:
stop-gradient e predictor insieme, mentre l'EMA a certe condizioni si può
togliere (SimSiam lo fa perdendo qualche punto, la mini-JEPA della pagina anche;
BYOL senza EMA collassa, se non si accelera il predictor).
"""

from paithon_svg import *

NOME = "jepa-tre-pezzi"
TITOLO = "i tre pezzi dell'asimmetria in una JEPA"


def scatola(x, y, w, h, testo, cls="box", sotto=None):
    s = [f'<rect class="{cls}" x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" '
         f'height="{h:.0f}" rx="8"/>',
         f'<text class="lbl" x="{x + w / 2:.0f}" y="{y + h / 2 + 5:.0f}" '
         f'text-anchor="middle">{testo}</text>']
    if sotto:
        s.append(f'<text class="lbs" x="{x + w / 2:.0f}" y="{y + h + 18:.0f}" '
                 f'text-anchor="middle">{sotto}</text>')
    return s


def freccia(x1, y1, x2, y2, cls="fl"):
    ang = 0 if x2 > x1 else 180
    return [f'<line class="{cls}" x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2 - 9:.0f}" '
            f'y2="{y2:.0f}"/>',
            f'<polygon class="pt" points="{x2 - 10:.0f},{y2 - 5:.0f} {x2:.0f},'
            f'{y2:.0f} {x2 - 10:.0f},{y2 + 5:.0f}"/>'] if ang == 0 else []


def costruisci() -> Figura:
    c = []
    ya, yb, h = 40.0, 190.0, 48.0
    # ramo del contesto
    c += scatola(20, ya, 120, h, "contesto", "dato")
    c += freccia(140, ya + h / 2, 190, ya + h / 2)
    c += scatola(190, ya, 120, h, "encoder", "box")
    c += freccia(310, ya + h / 2, 360, ya + h / 2)
    c += scatola(360, ya, 120, h, "predictor", "chiave",
                 "su un ramo solo: necessario")
    c += [f'<line class="fl" x1="480" y1="{ya + h / 2:.0f}" x2="600" '
          f'y2="{ya + h / 2:.0f}"/>']
    # ramo del bersaglio
    c += scatola(20, yb, 120, h, "bersaglio", "dato")
    c += freccia(140, yb + h / 2, 190, yb + h / 2)
    c += scatola(190, yb, 120, h, "encoder EMA", "copia",
                 "copia lenta: a volte si toglie")
    c += [f'<line class="fl sg" x1="310" y1="{yb + h / 2:.0f}" x2="600" '
          f'y2="{yb + h / 2:.0f}"/>']
    c += [f'<text class="lbs sgt" x="455" y="{yb + h / 2 - 10:.0f}" '
          f'text-anchor="middle">stop-gradient: necessario</text>',
          f'<line class="taglio" x1="445" y1="{yb + h / 2 - 8:.0f}" x2="465" '
          f'y2="{yb + h / 2 + 8:.0f}"/>',
          f'<line class="taglio" x1="465" y1="{yb + h / 2 - 8:.0f}" x2="445" '
          f'y2="{yb + h / 2 + 8:.0f}"/>']
    # la perdita
    yp = (ya + yb) / 2 - 4
    c += scatola(530, yp, 140, h + 8, "distanza", "perdita")
    c += [f'<text class="lbs" x="690" y="{yp + h / 2 + 8:.0f}">fra i due</text>',
          f'<text class="lbs" x="690" y="{yp + h / 2 + 26:.0f}">embedding</text>',
          f'<line class="fl" x1="600" y1="{ya + h / 2:.0f}" x2="600" '
          f'y2="{yp - 10:.0f}"/>',
          f'<polygon class="pt" points="{595},{yp - 10:.0f} 605,{yp - 10:.0f} '
          f'600,{yp:.0f}"/>',
          f'<line class="fl sg" x1="600" y1="{yb + h / 2:.0f}" x2="600" '
          f'y2="{yp + h + 18:.0f}"/>',
          f'<polygon class="ptsg" points="595,{yp + h + 18:.0f} 605,'
          f'{yp + h + 18:.0f} 600,{yp + h + 8:.0f}"/>']
    # la media mobile, dall'encoder alla copia
    c += [f'<line class="ema" x1="250" y1="{ya + h:.0f}" x2="250" '
          f'y2="{yb - 4:.0f}"/>',
          f'<text class="lbs" x="258" y="{(ya + yb + h) / 2 + 4:.0f}">'
          f'media mobile dei pesi</text>']
    return Figura(
        larghezza=770, altezza=yb + h + 40,
        alt="Due rami. In alto il contesto passa per l'encoder e poi per il "
            "predictor, che esiste su un ramo solo ed è necessario. In basso il "
            "bersaglio passa per una copia dell'encoder aggiornata come media "
            "mobile dei pesi, che a certe condizioni si può togliere; la sua uscita "
            "va alla perdita "
            "con uno stop-gradient, anch'esso necessario, segnato da una croce "
            "sulla freccia. La perdita misura la distanza fra i due embedding.",
        corpo="".join(c),
        stile=f"""    .box {{ fill:{CREAM}; stroke:{TEAL}; stroke-width:2; }}
    .dato {{ fill:{CREAM}; stroke:{BORDER_STRONG}; stroke-width:2; }}
    .chiave {{ fill:{CREAM}; stroke:{TERRACOTTA}; stroke-width:3; }}
    .copia {{ fill:{CREAM}; stroke:{OCRA}; stroke-width:2; stroke-dasharray:6 4; }}
    .perdita {{ fill:{CREAM}; stroke:{INK}; stroke-width:2; }}
    .fl {{ stroke:{FG_MUTED}; stroke-width:1.8; }}
    .sg {{ stroke:{TERRACOTTA}; }}
    .sgt {{ fill:{TERRACOTTA}; }}
    .taglio {{ stroke:{TERRACOTTA}; stroke-width:2.5; }}
    .pt {{ fill:{FG_MUTED}; }}
    .ptsg {{ fill:{TERRACOTTA}; }}
    .ema {{ stroke:{OCRA}; stroke-width:2; stroke-dasharray:4 4; }}""",
    )
