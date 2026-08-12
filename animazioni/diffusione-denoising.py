"""Diffusione: da rumore puro a una cifra, un passo del processo inverso per volta.

Il tempo è il contenuto, ma **non** nel modo in cui si racconta di solito. Il
passo inverso non «toglie un velo di rumore»: sottrae la stima del rumore,
riscala di 1/sqrt(alfa_t) e poi **ne aggiunge di nuovo**, con una deviazione
standard che per quasi tutta la traiettoria vale sette-dieci volte il
coefficiente con cui ha sottratto. A far emergere la cifra è la riscalatura, che
amplifica ciò che sopravvive al rimescolamento; e ciò che sopravvive è la
struttura, perché è l'unica cosa che il rumore non cancella.

Per questo la formula qui sotto porta il termine sigma_t z, e per questo il
rumore nella scena cala tardi e di colpo: l'ultimo passo è l'unico in cui non se
ne aggiunge (in DDPM z = 0 per t = 1).

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

PASSI = [1000, 800, 600, 400, 200, 0]   # i timestep mostrati


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
            r"\epsilon_\theta(x_t, t)\Big) + \sigma_t z",
            scala=0.85,
        )
        self.centra(VGroup(quadro, t, eq).arrange(DOWN, buff=0.4))

        self.play(FadeIn(celle, run_time=NORMALE), Create(cornice), FadeIn(t))
        self.play(entra(eq), run_time=NORMALE)
        self.pausa(0.35)

        # --- ogni passo: meno rumore, più segnale --------------------------
        for k, passo in enumerate(PASSI[1:], start=1):
            peso = k / (len(PASSI) - 1)          # quanto del target è emerso
            # il rumore cala TARDI: a ogni passo se ne aggiunge di nuovo, e solo
            # l'ultimo ne e' privo. Con un esponente alto la scena mostrerebbe
            # una ripulitura progressiva, che e' proprio la cosa falsa.
            residuo = (1 - peso) ** 0.55         # il rumore che resta
            nuovo_t = self.etichetta(f"t = {passo}", scala=T_CORPO,
                                     colore=PRIMARIO).move_to(t)
            self.play(
                *[c.animate.set_fill(TESTO,
                                     opacity=min(1.0, max(0.0,
                                                          peso * o + residuo * rnd.random())))
                  for c, o in zip(celle, obiettivo)],
                Transform(t, nuovo_t),
                run_time=1.2,
            )

        # La formula esce di scena prima della didascalia: l'ultimo fotogramma
        # deve reggere da solo, e con tutt'e due addosso si sovrappongono.
        self.play(FadeOut(eq), run_time=RAPIDO)
        self.play(entra(self.didascalia(
                      "sottrae la stima del rumore, riscala, "
                      "e ne aggiunge di nuovo")),
                  run_time=RAPIDO)
        self.chiusura()
