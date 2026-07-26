"""Attenzione lineare: il passato non si rilegge, si riassume in uno stato.

Il tempo è il contenuto: a ogni token lo stato assorbe un prodotto esterno e
resta della stessa taglia. È la differenza fra rileggere tutto e ricordare.

I numeri sono calcolati davvero: la scena esegue la ricorrenza sui vettori
definiti qui sotto.

    python3 .claude/skills/anima-manim/driver.py render animazioni/stato-ricorrente.py
"""

from paithon_anim import *

# key trasformate e valori dei cinque token: entrate in {0,1} perché i prodotti
# esterni si leggano a colpo d'occhio. Notazione del capitolo (Katharopoulos):
# lo stato somma v_t phi(k_t)^T e si legge con phi(q_t)
K = [(1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1)]
V = [(1, 0, 1), (0, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
D = 3


def stati():
    """S_t = S_{t-1} + v_t phi(k_t)^T, calcolato passo per passo."""
    s = [[0] * D for _ in range(D)]
    storia = [[riga[:] for riga in s]]
    for k, v in zip(K, V):
        for i in range(D):
            for j in range(D):
                s[i][j] += v[i] * k[j]
        storia.append([riga[:] for riga in s])
    return storia


class StatoRicorrente(ScenaPaithon):
    titolo_scena = "Attenzione lineare come ricorrenza"

    def costruisci(self):
        storia = stati()
        picco = max(max(r) for r in storia[-1]) or 1
        lato = 0.90

        celle, testi = [], []
        matrice = VGroup()
        for i in range(D):
            for j in range(D):
                q = Square(side_length=lato, stroke_color=FILETTO, stroke_width=2,
                           fill_color=SECONDARIO, fill_opacity=0.0)
                q.move_to([j * lato, -i * lato, 0])
                t = self.etichetta("0", scala=T_ETICHETTA).move_to(q)
                celle.append(q)
                testi.append(t)
                matrice.add(q, t)
        nome_S = self.formula(r"S_t", scala=1.0, colore=SECONDARIO)
        nome_S.next_to(matrice, UP, buff=0.22)
        blocco_S = VGroup(matrice, nome_S)

        ric = self.formula(r"S_t = S_{t-1} + v_t\, \phi(k_t)^{\top}", scala=0.95)
        usc = self.formula(r"o_t = S_t\, \phi(q_t)", scala=0.95, colore=TESTO_TENUE)
        formule = VGroup(ric, usc).arrange(DOWN, buff=0.35)

        token = VGroup()
        for i in range(len(K)):
            lab = self.formula(rf"x_{{{i + 1}}}", scala=0.8)
            box = RoundedRectangle(corner_radius=0.08, width=1.05, height=0.72,
                                   stroke_color=TESTO_TENUE, stroke_width=2.5,
                                   fill_opacity=0)
            token.add(VGroup(box, lab.move_to(box)))
        token.arrange(RIGHT, buff=0.26)

        alto = VGroup(blocco_S, formule).arrange(RIGHT, buff=1.3)
        self.centra(VGroup(alto, token).arrange(DOWN, buff=0.8), margine_basso=1.0)

        self.play(Create(matrice), FadeIn(nome_S), run_time=NORMALE)
        self.play(entra(*token, run_time=RAPIDO), run_time=RAPIDO)
        self.play(entra(ric, usc), run_time=NORMALE)
        self.pausa(0.3)

        # --- un token per volta: lo stato assorbe, non si allunga ----------
        for t in range(len(K)):
            nuovo = storia[t + 1]
            box = self.riquadro(token[t], colore=PRIMARIO, buff=0.06)
            freccia = Arrow(token[t].get_top(), matrice.get_bottom() + DOWN * 0.05,
                            color=PRIMARIO, stroke_width=4, buff=0.12,
                            max_tip_length_to_length_ratio=0.12)
            self.play(Create(box), GrowArrow(freccia), run_time=0.3)

            aggiornamenti = []
            for i in range(D):
                for j in range(D):
                    idx = i * D + j
                    val = nuovo[i][j]
                    aggiornamenti.append(
                        celle[idx].animate.set_fill(SECONDARIO, opacity=0.7 * val / picco))
                    if val != storia[t][i][j]:
                        aggiornamenti.append(Transform(
                            testi[idx],
                            self.etichetta(str(val), scala=T_ETICHETTA).move_to(celle[idx])))
            self.play(*aggiornamenti, run_time=0.35)
            self.play(FadeOut(box), FadeOut(freccia), run_time=0.2)

        self.play(evidenzia(matrice, scala=1.08),
                  entra(self.didascalia("lo stato assorbe ogni token e non cresce mai")),
                  run_time=NORMALE)
        self.chiusura()
