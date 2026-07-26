"""Come scrive un modello linguistico: una parola per volta, rileggendo tutto.

Il tempo è il contenuto: la frase cresce, e a ogni passo il contesto da cui
si campiona è tutto ciò che è stato scritto prima.

    python3 .claude/skills/anima-manim/driver.py render animazioni/generazione-autoregressiva.py
"""

from paithon_anim import *

FRASE = ["Il", "gatto", "nero", "salta", "sul", "muro"]
DATI = 2   # «Il gatto» è il prompt: da lì in poi genera il modello

# candidati e probabilità a ogni passo; ogni distribuzione somma a 1
CANDIDATI = [
    [("nero", 0.42), ("dorme", 0.28), ("salta", 0.18), ("bianco", 0.12)],
    [("salta", 0.48), ("dorme", 0.26), ("corre", 0.16), ("miagola", 0.10)],
    [("sul", 0.51), ("su", 0.24), ("dal", 0.15), ("verso", 0.10)],
    [("muro", 0.44), ("tavolo", 0.31), ("divano", 0.15), ("letto", 0.10)],
]


class GenerazioneAutoregressiva(ScenaPaithon):
    titolo_scena = "Generazione autoregressiva"

    def costruisci(self):
        frase = VGroup(*[self.corpo(p, scala=0.42) for p in FRASE]).arrange(RIGHT, buff=0.38)
        eq = self.formula(r"p(x_t \mid x_1, \dots, x_{t-1})", scala=0.85)

        # lo spazio dei candidati: quattro barre, riservato una volta per tutte
        scala_barra = 6.5
        posto = Rectangle(width=8.0, height=2.4, stroke_opacity=0, fill_opacity=0)
        self.centra(VGroup(frase, posto, eq).arrange(DOWN, buff=0.45))

        self.play(entra(*frase[:DATI], run_time=NORMALE), run_time=NORMALE)
        for p in frase[DATI:]:
            p.set_opacity(0)
        self.play(entra(eq), run_time=RAPIDO)

        for passo, candidati in enumerate(CANDIDATI):
            i = DATI + passo

            # 1. il contesto: tutto ciò che è già stato scritto
            contesto = self.riquadro(VGroup(*frase[:i]), colore=SECONDARIO, buff=0.14)
            eti = self.etichetta("contesto", scala=T_PICCOLO, colore=SECONDARIO)
            eti.next_to(contesto, LEFT, buff=0.2)
            self.play(Create(contesto), FadeIn(eti), run_time=0.28)

            # 2. la distribuzione sul token successivo
            barre = VGroup()
            for idx, (parola, p) in enumerate(candidati):
                scelto = idx == 0                     # il primo è il più probabile
                colore = PRIMARIO if scelto else TESTO_TENUE
                nome = self.etichetta(parola, scala=T_ETICHETTA, colore=colore)
                barra = Rectangle(width=max(p * scala_barra, 0.05), height=0.34,
                                  stroke_width=0, fill_color=colore,
                                  fill_opacity=0.85 if scelto else 0.3)
                val = self.etichetta(f"{p:.2f}", scala=T_ETICHETTA, colore=colore)
                barre.add(VGroup(nome, barra, val))

            # colonna dei nomi allineata a destra, barre tutte dalla stessa x
            largo = max(r[0].width for r in barre)
            for idx, (nome, barra, val) in enumerate(barre):
                y = -idx * 0.58
                nome.move_to([-0.25 - largo / 2, y, 0])
                barra.move_to([0.05 + barra.width / 2, y, 0])
                val.move_to([0.05 + barra.width + 0.38, y, 0])
            barre.move_to(posto)

            self.play(LaggedStart(*[entra(r, run_time=0.3) for r in barre], lag_ratio=0.08),
                      run_time=0.55)

            # 3. il token scelto entra nella frase e diventa contesto
            vincitore = barre[0][0].copy()
            frase[i].set_opacity(1)
            self.play(ReplacementTransform(vincitore, frase[i]),
                      FadeOut(barre), FadeOut(contesto), FadeOut(eti),
                      run_time=0.45)

        self.play(evidenzia(frase),
                  entra(self.didascalia("ogni parola nasce dal testo che la precede")),
                  run_time=NORMALE)
        self.chiusura()
