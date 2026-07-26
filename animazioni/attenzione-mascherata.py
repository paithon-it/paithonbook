"""Attenzione mascherata: perché un modello generativo guarda solo il passato.

Il tempo è il contenuto: prima tutti i punteggi, poi la maschera che spegne il
futuro, poi la softmax che ridistribuisce ciò che resta.

    python3 .claude/skills/anima-manim/driver.py render animazioni/attenzione-mascherata.py
"""

import random

from paithon_anim import *

PAROLE = ["Il", "gatto", "nero", "salta", "sul", "muro"]

# Pesi dopo la softmax: riga = query, colonna = chiave. Ogni riga somma a 1 e
# nessuna guarda oltre la diagonale — è esattamente ciò che impone la maschera.
PESI = [
    [1.00],
    [0.30, 0.70],
    [0.10, 0.60, 0.30],
    [0.05, 0.55, 0.10, 0.30],
    [0.05, 0.15, 0.05, 0.35, 0.40],
    [0.03, 0.12, 0.05, 0.30, 0.30, 0.20],
]
QUERY = 3   # la riga che mettiamo in evidenza: «salta»


class AttenzioneMascherata(ScenaPaithon):
    titolo_scena = "Attenzione mascherata"

    def costruisci(self):
        rnd = random.Random(11)
        n, lato = len(PAROLE), 0.68

        celle = VGroup(*[
            Square(side_length=lato, stroke_color=FILETTO, stroke_width=1.5,
                   fill_color=TESTO, fill_opacity=0).move_to([j * lato, -i * lato, 0])
            for i in range(n) for j in range(n)
        ])
        celle.move_to(ORIGIN)

        def cella(i, j):
            return celle[i * n + j]

        # etichette: le query a sinistra, le chiavi in alto (ruotate)
        righe = VGroup(*[
            self.etichetta(p, scala=T_PICCOLO).next_to(cella(i, 0), LEFT, buff=0.18)
            for i, p in enumerate(PAROLE)
        ])
        colonne = VGroup(*[
            self.etichetta(p, scala=T_PICCOLO).rotate(PI / 2).next_to(cella(0, j), UP, buff=0.18)
            for j, p in enumerate(PAROLE)
        ])
        q_lab = self.etichetta("query", scala=T_PICCOLO, colore=PRIMARIO)
        q_lab.rotate(PI / 2).next_to(righe, LEFT, buff=0.2)
        k_lab = self.etichetta("chiave", scala=T_PICCOLO, colore=SECONDARIO)
        k_lab.next_to(colonne, UP, buff=0.2)

        tabella = VGroup(celle, righe, colonne, q_lab, k_lab)
        eq = self.formula(
            r"\mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}} + M\right)V", scala=0.8
        )
        self.centra(VGroup(tabella, eq).arrange(DOWN, buff=0.5))

        self.play(Create(celle), run_time=NORMALE)
        self.play(entra(*righe, *colonne, q_lab, k_lab, run_time=RAPIDO), run_time=RAPIDO)

        # --- 1. i punteggi grezzi: ogni parola confrontata con tutte -------
        self.play(
            LaggedStart(*[c.animate.set_fill(TESTO, opacity=0.12 + 0.4 * rnd.random())
                          for c in celle], lag_ratio=0.012),
            run_time=LENTO,
        )
        self.play(entra(eq), run_time=RAPIDO)
        self.pausa(0.4)

        # --- 2. la maschera spegne il futuro ------------------------------
        futuro = [cella(i, j) for i in range(n) for j in range(n) if j > i]
        # il confine fra passato e futuro è una scala, non una diagonale: la
        # diagonale taglierebbe a metà le celle in cui una parola guarda sé stessa
        angoli = [cella(0, 0).get_corner(UR)]
        for i in range(n - 1):
            angoli += [cella(i, i).get_corner(DR), cella(i + 1, i + 1).get_corner(UR)]
        angoli.append(cella(n - 1, n - 1).get_corner(DR))
        diagonale = VMobject(color=SECONDARIO, stroke_width=4).set_points_as_corners(angoli)
        m_lab = self.etichetta("M = −∞", scala=T_PICCOLO, colore=SECONDARIO)
        m_lab.move_to(cella(1, 4)).shift(UP * 0.1)
        self.play(
            *[c.animate.set_fill(opacity=0).set_stroke(color=FILETTO, width=1) for c in futuro],
            Create(diagonale), FadeIn(m_lab),
            run_time=NORMALE,
        )
        self.pausa(0.4)

        # --- 3. la softmax ridistribuisce: ogni riga torna a sommare 1 ----
        passato = []
        for i in range(n):
            for j, w in enumerate(PESI[i]):
                passato.append(cella(i, j).animate.set_fill(PRIMARIO, opacity=0.10 + 0.85 * w))
        self.play(LaggedStart(*passato, lag_ratio=0.02), run_time=LENTO)

        # --- 4. una riga da leggere: «salta» guarda «gatto» ---------------
        riga = VGroup(*[cella(QUERY, j) for j in range(QUERY + 1)])
        box = SurroundingRectangle(riga, color=PRIMARIO, stroke_width=4.5, buff=0.06)
        self.play(Create(box), evidenzia(righe[QUERY]), run_time=RAPIDO)
        self.play(entra(self.didascalia("ogni parola pesa solo ciò che l'ha preceduta")),
                  run_time=RAPIDO)
        self.chiusura()
