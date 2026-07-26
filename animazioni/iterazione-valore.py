"""Iterazione di valore: il valore che si propaga a ritroso dall'obiettivo.

Il tempo è il contenuto: l'equazione di Bellman applicata in ciclo è un'onda
che parte dalla ricompensa e risale, un passo per iterazione.

I numeri non sono inventati: la scena esegue davvero l'iterazione di valore
sul mondo a griglia definito qui sotto.

    python3 .claude/skills/anima-manim/driver.py render animazioni/iterazione-valore.py
"""

from paithon_anim import *

RIGHE, COLONNE = 4, 4
MURI = {(1, 1), (2, 2)}
OBIETTIVO = (0, 3)
GAMMA = 0.9
ITERAZIONI = 6   # a k = 6 il mondo è già convergiato


def valori():
    """Iterazione di valore su un mondo a griglia deterministico.

    Ricompensa 1 solo nell'entrare nella casella obiettivo, che è terminale.
    Un'azione contro un muro o contro il bordo lascia l'agente dov'è.
    Restituisce la lista degli stati V_0 … V_k (uno per fotogramma).
    """
    v = {(i, j): 0.0 for i in range(RIGHE) for j in range(COLONNE) if (i, j) not in MURI}
    storia = [dict(v)]
    for _ in range(ITERAZIONI):
        nuovo = {}
        for s in v:
            if s == OBIETTIVO:          # terminale: non si accumula altro
                nuovo[s] = 0.0
                continue
            migliore = 0.0
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                s2 = (s[0] + di, s[1] + dj)
                if s2 not in v:          # muro o bordo: si resta fermi
                    s2 = s
                migliore = max(migliore, (1.0 if s2 == OBIETTIVO else 0.0) + GAMMA * v[s2])
            nuovo[s] = migliore
        v = nuovo
        storia.append(dict(v))
    return storia


class IterazioneValore(ScenaPaithon):
    titolo_scena = "Iterazione di valore"

    def costruisci(self):
        storia = valori()
        picco = max(max(s.values()) for s in storia) or 1.0
        lato = 1.05

        celle, testi = {}, {}
        griglia = VGroup()
        for i in range(RIGHE):
            for j in range(COLONNE):
                muro = (i, j) in MURI
                q = Square(
                    side_length=lato,
                    stroke_color=FILETTO if not muro else TESTO_TENUE,
                    stroke_width=2,
                    fill_color=TESTO_TENUE if muro else PRIMARIO,
                    fill_opacity=0.35 if muro else 0.0,
                ).move_to([j * lato, -i * lato, 0])
                griglia.add(q)
                if not muro:
                    celle[(i, j)] = q
                    if (i, j) != OBIETTIVO:   # sull'obiettivo parla la stella
                        t = self.etichetta("0.00", scala=T_PICCOLO).move_to(q)
                        testi[(i, j)] = t
                        griglia.add(t)
        griglia.move_to(ORIGIN)

        stella = Star(n=5, outer_radius=0.22, color=TERZIARIO, fill_opacity=1,
                      stroke_width=0).move_to(celle[OBIETTIVO])
        r_lab = self.etichetta("r = +1", scala=T_PICCOLO, colore=TERZIARIO)
        r_lab.next_to(celle[OBIETTIVO], UP, buff=0.12)

        k = self.etichetta("k = 0", scala=T_CORPO, colore=PRIMARIO)
        eq = self.formula(
            r"V_{k+1}(s) \;=\; \max_a \big[\, r(s,a) + \gamma\, V_k(s') \,\big]", scala=0.85
        )
        colonna_destra = VGroup(k, eq).arrange(DOWN, buff=0.5)
        blocco = VGroup(VGroup(griglia, stella, r_lab), colonna_destra).arrange(RIGHT, buff=0.9)
        self.centra(blocco)

        self.play(Create(griglia), run_time=NORMALE)
        self.play(entra(stella, r_lab), FadeIn(k), run_time=RAPIDO)
        self.play(entra(eq), run_time=NORMALE)
        self.pausa(0.35)

        # --- l'onda: a ogni giro il valore risale di una casella ----------
        for passo in range(1, ITERAZIONI + 1):
            v = storia[passo]
            nuovo_k = self.etichetta(f"k = {passo}", scala=T_CORPO, colore=PRIMARIO).move_to(k)
            animazioni = [Transform(k, nuovo_k)]
            for s, q in celle.items():
                if s == OBIETTIVO:
                    continue
                animazioni.append(q.animate.set_fill(PRIMARIO, opacity=0.75 * v[s] / picco))
                nuovo_t = self.etichetta(f"{v[s]:.2f}", scala=T_PICCOLO).move_to(q)
                animazioni.append(Transform(testi[s], nuovo_t))
            self.play(*animazioni, run_time=0.55, rate_func=EASE_OUT)

        self.play(entra(self.didascalia("il valore risale dall'obiettivo, una casella per iterazione")),
                  run_time=RAPIDO)
        self.chiusura()
