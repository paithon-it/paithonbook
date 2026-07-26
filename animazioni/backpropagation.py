"""La backpropagation: l'errore che risale la rete, un fattore per strato.

Il tempo è il contenuto: prima il segnale va avanti, poi il gradiente torna
indietro e la regola della catena si compone un pezzo alla volta.

    python3 .claude/skills/anima-manim/driver.py render animazioni/backpropagation.py
"""

from paithon_anim import *


class Backpropagation(ScenaPaithon):
    titolo_scena = "Backpropagation"

    def costruisci(self):
        # --- la rete più piccola che mostri due strati di pesi: 3 → 3 → 1 ---
        conte, xs, r = (3, 3, 1), (-4.6, -1.2, 2.4), 0.30
        strati = VGroup()
        for n, xc in zip(conte, xs):
            col = VGroup(*[
                Circle(radius=r, stroke_color=TESTO_TENUE, stroke_width=3,
                       fill_color=SFONDO, fill_opacity=1)
                for _ in range(n)
            ]).arrange(DOWN, buff=0.75)
            strati.add(col.move_to([xc, 0, 0]))

        # un gruppo di archi per ogni strato di pesi: si accendono insieme
        archi = VGroup()
        for a, b in zip(strati[:-1], strati[1:]):
            archi.add(VGroup(*[
                Line(na.get_center(), nb.get_center(), color=FILETTO,
                     stroke_width=2, buff=r + 0.03)
                for na in a for nb in b
            ]))

        eti = VGroup(
            self.etichetta("x").next_to(strati[0], DOWN, buff=0.3),
            self.etichetta("h").next_to(strati[1], DOWN, buff=0.3),
            self.etichetta("ŷ").next_to(strati[2], DOWN, buff=0.3),
        )

        lab_L = self.formula(r"\mathcal{L}=(\hat{y}-y)^2", scala=0.7, colore=PRIMARIO)
        box_L = RoundedRectangle(corner_radius=0.08, width=lab_L.width + 0.5,
                                 height=lab_L.height + 0.5, stroke_color=PRIMARIO,
                                 stroke_width=3, fill_opacity=0)
        perdita = VGroup(box_L, lab_L.move_to(box_L))
        perdita.next_to(strati[2], RIGHT, buff=0.6)

        # la regola della catena, un pezzo per volta: ogni pezzo è un arco
        catena = self.formula([
            r"\frac{\partial \mathcal{L}}{\partial W_1}", "=",
            r"\frac{\partial \mathcal{L}}{\partial \hat{y}}", r"\cdot",
            r"\frac{\partial \hat{y}}{\partial h}", r"\cdot",
            r"\frac{\partial h}{\partial W_1}",
        ], scala=0.95)

        rete = VGroup(archi, strati, eti, perdita)
        self.centra(VGroup(rete, catena).arrange(DOWN, buff=0.5))

        def scorre(gruppo, colore, indietro=False):
            """Un impulso che percorre tutti gli archi di uno strato."""
            linee = [Line(l.get_end(), l.get_start()) if indietro else l for l in gruppo]
            return LaggedStart(*[
                ShowPassingFlash(l.copy().set_stroke(color=colore, width=7), time_width=0.55)
                for l in linee
            ], lag_ratio=0.03)

        # --- andata: il segnale attraversa la rete -------------------------
        self.play(Create(archi), Create(strati), run_time=NORMALE)
        self.play(entra(*eti), run_time=RAPIDO)
        self.play(scorre(archi[0], SECONDARIO), run_time=0.55)
        self.play(scorre(archi[1], SECONDARIO), run_time=0.55)
        self.play(entra(perdita, shift=LEFT * 0.2), run_time=NORMALE)
        self.pausa(0.4)

        # --- ritorno: il gradiente risale e la catena si compone -----------
        self.play(Write(catena[0]), Write(catena[1]), run_time=RAPIDO)
        self.play(Write(catena[2]),
                  Flash(strati[2][0], color=PRIMARIO, line_length=0.22),
                  run_time=RAPIDO)
        self.play(scorre(archi[1], PRIMARIO, indietro=True), run_time=0.55)
        self.play(Write(catena[3]), Write(catena[4]), run_time=RAPIDO)
        self.play(scorre(archi[0], PRIMARIO, indietro=True), run_time=0.55)
        self.play(Write(catena[5]), Write(catena[6]), run_time=RAPIDO)

        self.play(archi[0].animate.set_stroke(color=PRIMARIO, width=3),
                  entra(self.didascalia("l'errore torna indietro: un fattore per ogni strato")),
                  run_time=NORMALE)
        self.chiusura()
