"""Diffusione: da rumore puro a una cifra, un passo di denoising per volta.

Il tempo è il contenuto: il modello non disegna, sottrae. Ogni passo toglie
un po' di rumore e la cifra emerge dal fondo.

    python3 .claude/skills/anima-manim/driver.py render animazioni/diffusione-denoising.py
"""

import random

from paithon_anim import *

# La cifra 3 in pixel art: il target del processo inverso.
CIFRA = [
    "............",
    "...#####....",
    "..##...##...",
    ".......##...",
    "....####....",
    ".......##...",
    "........##..",
    "..##....##..",
    "...#####....",
    "............",
]

PASSI = [1000, 750, 500, 250, 0]   # i timestep mostrati


class DiffusioneDenoising(ScenaPaithon):
    titolo_scena = "Diffusione: il processo inverso"

    def costruisci(self):
        rnd = random.Random(7)          # seme fisso: la clip è riproducibile
        lato = 0.38
        righe, colonne = len(CIFRA), len(CIFRA[0])

        obiettivo = [1.0 if ch == "#" else 0.05 for r in CIFRA for ch in r]
        celle = VGroup(*[
            Square(side_length=lato, stroke_width=0, fill_color=TESTO,
                   fill_opacity=rnd.random()).move_to([j * lato, -i * lato, 0])
            for i in range(righe) for j in range(colonne)
        ])
        celle.move_to(ORIGIN)
        cornice = SurroundingRectangle(celle, color=FILETTO, stroke_width=2, buff=0.06)
        quadro = VGroup(celle, cornice)

        t = self.etichetta(f"t = {PASSI[0]}", scala=T_CORPO, colore=PRIMARIO)
        eq = self.formula(
            r"x_{t-1} = \frac{1}{\sqrt{\alpha_t}}"
            r"\Big(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\,"
            r"\epsilon_\theta(x_t, t)\Big)",
            scala=0.85,
        )
        self.centra(VGroup(quadro, t, eq).arrange(DOWN, buff=0.4))

        self.play(FadeIn(celle, run_time=NORMALE), Create(cornice), FadeIn(t))
        self.play(entra(eq), run_time=NORMALE)
        self.pausa(0.35)

        # --- ogni passo: meno rumore, più segnale --------------------------
        for k, passo in enumerate(PASSI[1:], start=1):
            peso = k / (len(PASSI) - 1)          # quanto del target è emerso
            residuo = (1 - peso) ** 1.4          # il rumore che resta
            nuovo_t = self.etichetta(f"t = {passo}", scala=T_CORPO,
                                     colore=PRIMARIO).move_to(t)
            self.play(
                *[c.animate.set_fill(TESTO,
                                     opacity=min(1.0, max(0.0,
                                                          peso * o + residuo * rnd.random())))
                  for c, o in zip(celle, obiettivo)],
                Transform(t, nuovo_t),
                run_time=0.5,
            )

        self.play(entra(self.didascalia("la rete predice il rumore e lo sottrae, passo dopo passo")),
                  run_time=RAPIDO)
        self.chiusura()
