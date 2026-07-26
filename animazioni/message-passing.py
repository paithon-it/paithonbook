"""Message passing: come un nodo di un grafo impara dal proprio vicinato.

Il tempo è il contenuto: a ogni giro il campo recettivo si allarga di un salto.
Dopo k giri il nodo ha visto tutto ciò che dista k archi.

    python3 .claude/skills/anima-manim/driver.py render animazioni/message-passing.py
"""

from paithon_anim import *

# posizioni scelte a mano: v al centro, il primo anello attorno, il secondo fuori
POSIZIONI = {
    "v": (0.0, 0.0),
    "a": (-1.95, 1.15), "b": (1.85, 1.0), "c": (-1.75, -1.2), "d": (2.0, -1.05),
    "e": (-3.7, 0.0), "f": (3.6, 0.2), "g": (-0.15, 2.05), "h": (0.2, -2.05),
}
ARCHI = [("v", "a"), ("v", "b"), ("v", "c"), ("v", "d"),
         ("a", "e"), ("c", "e"), ("b", "f"), ("d", "f"),
         ("a", "g"), ("b", "g"), ("c", "h"), ("d", "h")]

ANELLO_1 = ["a", "b", "c", "d"]
ANELLO_2 = ["e", "f", "g", "h"]


class MessagePassing(ScenaPaithon):
    titolo_scena = "Message passing"

    def costruisci(self):
        r = 0.30
        nodi = {
            nome: Circle(radius=r, stroke_color=TESTO_TENUE, stroke_width=3,
                         fill_color=SFONDO, fill_opacity=1).move_to([x, y, 0])
            for nome, (x, y) in POSIZIONI.items()
        }
        archi = {
            (u, w): Line(nodi[u].get_center(), nodi[w].get_center(),
                         color=FILETTO, stroke_width=2.5, buff=r + 0.02)
            for u, w in ARCHI
        }

        grafo = VGroup(*archi.values(), *nodi.values())
        eti_v = self.etichetta("v", scala=T_PICCOLO, colore=PRIMARIO).move_to(nodi["v"])

        k = self.etichetta("k = 0", scala=T_CORPO, colore=PRIMARIO)
        eq = self.formula(
            r"h_v^{(k+1)} = \phi\Big(h_v^{(k)},\ "
            r"\bigoplus_{u \in \mathcal{N}(v)} \psi\big(h_u^{(k)}, h_v^{(k)}\big)\Big)",
            scala=0.8,
        )
        piede = VGroup(k, eq).arrange(RIGHT, buff=0.7)
        self.centra(VGroup(VGroup(grafo, eti_v), piede).arrange(DOWN, buff=0.4))

        def onda(coppie, colore):
            """Un messaggio per arco: parte dal primo nodo e arriva al secondo."""
            punti, moti = VGroup(), []
            for u, w in coppie:
                # il percorso si ricostruisce dai centri: così il verso è sempre
                # quello del messaggio, non quello in cui è nato l'arco
                perc = Line(nodi[u].get_center(), nodi[w].get_center(), buff=r + 0.02)
                d = Dot(perc.get_start(), color=colore, radius=0.085)
                punti.add(d)
                moti.append(MoveAlongPath(d, perc))
            self.play(*moti, run_time=0.75, rate_func=EASE_OUT)
            self.play(FadeOut(punti), run_time=0.15)

        self.play(Create(grafo), run_time=NORMALE)
        self.play(nodi["v"].animate.set_stroke(PRIMARIO, width=4).set_fill(PRIMARIO, opacity=0.18),
                  FadeIn(eti_v), FadeIn(k), run_time=RAPIDO)
        self.play(entra(eq), run_time=NORMALE)

        # --- k = 1: arrivano i vicini diretti ------------------------------
        self.play(Transform(k, self.etichetta("k = 1", scala=T_CORPO, colore=PRIMARIO).move_to(k)),
                  *[nodi[x].animate.set_stroke(SECONDARIO, width=3.5) for x in ANELLO_1],
                  run_time=RAPIDO)
        onda([(x, "v") for x in ANELLO_1], SECONDARIO)
        self.play(evidenzia(nodi["v"], scala=1.3), run_time=RAPIDO)

        # --- k = 2: i vicini dei vicini, attraverso i vicini ---------------
        self.play(Transform(k, self.etichetta("k = 2", scala=T_CORPO, colore=PRIMARIO).move_to(k)),
                  *[nodi[x].animate.set_stroke(TERZIARIO, width=3.5) for x in ANELLO_2],
                  run_time=RAPIDO)
        onda([("e", "a"), ("e", "c"), ("f", "b"), ("f", "d"),
              ("g", "a"), ("g", "b"), ("h", "c"), ("h", "d")], TERZIARIO)
        onda([(x, "v") for x in ANELLO_1], TERZIARIO)
        self.play(evidenzia(nodi["v"], scala=1.3), run_time=RAPIDO)

        self.play(entra(self.didascalia("dopo k giri, v ha visto tutto ciò che dista k archi")),
                  run_time=RAPIDO)
        self.chiusura()
