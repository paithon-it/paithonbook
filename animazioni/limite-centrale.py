"""Il teorema del limite centrale: la campana che si forma da sola.

Il tempo è il contenuto: la campana non viene disegnata, si accumula. Un dado
è piatto — nessuna faccia è più probabile di un'altra — eppure la somma di tre
dadi, ripetuta abbastanza volte, produce sempre la stessa forma.

I conti sono veri: i lanci si estraggono con un seme fisso e la curva
sovrapposta è la normale di media 10,5 e varianza 3·35/12, cioè i parametri
che il teorema prevede.

    python3 .claude/skills/anima-manim/driver.py render animazioni/limite-centrale.py
"""

import math
import random

from paithon_anim import *

DADI = 3
LANCI = 600
LOTTI = 5
SEME = 3

MU = DADI * 3.5                    # media di un dado = 3,5
VAR = DADI * 35 / 12               # varianza di un dado = 35/12
SIGMA = math.sqrt(VAR)
FACCE = list(range(DADI, 6 * DADI + 1))


def lanci():
    """Somme di tre dadi, un lotto per volta: la storia dell'istogramma."""
    rnd = random.Random(SEME)
    conte = {s: 0 for s in FACCE}
    storia = [dict(conte)]
    for _ in range(LOTTI):
        for _ in range(LANCI // LOTTI):
            conte[sum(rnd.randint(1, 6) for _ in range(DADI))] += 1
        storia.append(dict(conte))
    return storia


class LimiteCentrale(ScenaPaithon):
    titolo_scena = "Teorema del limite centrale"

    def costruisci(self):
        storia = lanci()
        picco = max(storia[-1].values())
        h_max, larg = 2.9, 0.34
        scala = h_max / picco

        # --- a sinistra: un dado solo, perfettamente piatto ----------------
        h_dado = 1.1
        barre_dado = VGroup(*[
            Rectangle(width=0.26, height=h_dado, stroke_width=0,
                      fill_color=SECONDARIO, fill_opacity=0.55)
            for _ in range(6)
        ]).arrange(RIGHT, buff=0.07)
        eti_dado = self.etichetta("un dado: piatto", scala=T_PICCOLO)
        eti_dado.next_to(barre_dado, DOWN, buff=0.25)
        sinistra = VGroup(barre_dado, eti_dado)

        freccia = VGroup(
            Arrow(LEFT * 0.45, RIGHT * 0.45, color=TESTO_TENUE, stroke_width=3,
                  max_tip_length_to_length_ratio=0.3, buff=0),
            self.etichetta("somma di 3", scala=T_PICCOLO),
        )
        freccia[1].next_to(freccia[0], DOWN, buff=0.18)

        # --- a destra: l'istogramma che si riempie -------------------------
        def istogramma(conte):
            g = VGroup()
            for s in FACCE:
                h = max(conte[s] * scala, 0.004)
                g.add(Rectangle(width=larg, height=h, stroke_width=0,
                                fill_color=PRIMARIO, fill_opacity=0.85))
            g.arrange(RIGHT, buff=0.06, aligned_edge=DOWN)
            return g

        barre = istogramma(storia[0])
        base = Line(LEFT, RIGHT, color=TESTO_TENUE, stroke_width=2)
        base.set_width(barre.width + 0.3).next_to(barre, DOWN, buff=0, aligned_edge=DOWN)
        eti_somma = self.etichetta("somma di tre dadi", scala=T_PICCOLO)
        eti_somma.next_to(base, DOWN, buff=0.3)
        destra = VGroup(barre, base, eti_somma)

        fila = VGroup(sinistra, freccia, destra).arrange(RIGHT, buff=0.55)
        # la formula dice ciò che l'istogramma mostra: la SOMMA, non la media
        eq = self.formula(
            r"S_n = X_1 + \dots + X_n \;\longrightarrow\; "
            r"\mathcal{N}\!\left(n\mu,\ n\sigma^2\right)",
            scala=0.8)
        self.centra(VGroup(fila, eq).arrange(DOWN, buff=0.55))

        self.play(entra(*barre_dado, run_time=RAPIDO), FadeIn(eti_dado), run_time=NORMALE)
        self.play(GrowArrow(freccia[0]), FadeIn(freccia[1]), Create(base),
                  FadeIn(eti_somma), run_time=RAPIDO)
        self.play(entra(eq), run_time=RAPIDO)

        # --- lotto dopo lotto la forma emerge ------------------------------
        for k in range(1, LOTTI + 1):
            nuove = istogramma(storia[k])
            nuove.move_to(barre, aligned_edge=DOWN)
            self.play(Transform(barre, nuove), run_time=0.55, rate_func=EASE_OUT)

        # --- la normale prevista dal teorema, non adattata ai dati ---------
        def densita(s):
            return math.exp(-((s - MU) ** 2) / (2 * VAR)) / (SIGMA * math.sqrt(2 * math.pi))

        punti = [
            [barre[i].get_x(), base.get_y() + LANCI * densita(s) * scala, 0]
            for i, s in enumerate(FACCE)
        ]
        curva = VMobject(color=SECONDARIO, stroke_width=5).set_points_smoothly(punti)
        eti_curva = self.etichetta("la normale prevista", scala=T_PICCOLO, colore=SECONDARIO)
        eti_curva.next_to(curva, UP, buff=0.12).shift(RIGHT * 1.1)

        self.play(Create(curva), run_time=NORMALE)
        self.play(FadeIn(eti_curva), run_time=RAPIDO)
        self.play(entra(self.didascalia("nessuno l'ha disegnata: la campana si accumula da sé")),
                  run_time=RAPIDO)
        self.chiusura()
