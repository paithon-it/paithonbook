"""La convoluzione: il kernel che scorre e costruisce la mappa di attivazione.

Il tempo è il contenuto: la finestra che scivola è l'operazione stessa.
L'immagine è una barra verticale spessa un pixel, il kernel un rilevatore di
bordi: in ogni posizione la barra cade sotto una sola colonna del kernel, e la
mappa risponde −3 se è la destra, +3 se è la sinistra, 0 se è quella centrale,
che ha pesi nulli.

    python3 .claude/skills/anima-manim/driver.py render animazioni/convoluzione.py
"""

from paithon_anim import *

IMG = [[0, 0, 1, 0, 0]] * 5                    # una barra verticale
KER = [[1, 0, -1], [1, 0, -1], [1, 0, -1]]     # bordi verticali (Prewitt)


def griglia(fabbrica_testo, valori, lato, colore_bordo, scala_testo):
    """Griglia di celle quadrate con il valore al centro."""
    celle, testi = VGroup(), VGroup()
    for i, riga in enumerate(valori):
        for j, v in enumerate(riga):
            c = Square(side_length=lato, stroke_color=colore_bordo, stroke_width=2,
                       fill_opacity=0)
            c.move_to([j * lato, -i * lato, 0])
            celle.add(c)
            testi.add(fabbrica_testo(f"{v}", scala=scala_testo).move_to(c))
    g = VGroup(celle, testi)
    g.move_to(ORIGIN)
    return g, celle, testi


class Convoluzione(ScenaPaithon):
    titolo_scena = "Convoluzione"

    def costruisci(self):
        lato = 0.74
        img, celle_img, _ = griglia(self.etichetta, IMG, lato, FILETTO, T_PICCOLO)
        ker, celle_ker, testi_ker = griglia(self.etichetta, KER, lato * 0.78,
                                            SECONDARIO, T_PICCOLO)
        for t in testi_ker:
            t.set_color(SECONDARIO)

        # la mappa in uscita nasce vuota: si riempie una cella per volta
        out = [[0] * 3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                out[i][j] = sum(IMG[i + m][j + n] * KER[m][n]
                                for m in range(3) for n in range(3))
        mappa, celle_out, testi_out = griglia(self.etichetta, out, lato, FILETTO, T_PICCOLO)

        seg1 = self.etichetta("∗", scala=T_CORPO, colore=TESTO)
        seg2 = self.etichetta("=", scala=T_CORPO, colore=TESTO)
        fila = VGroup(img, seg1, ker, seg2, mappa).arrange(RIGHT, buff=0.45)

        nomi = VGroup(
            self.etichetta("immagine", scala=T_PICCOLO).next_to(img, UP, buff=0.25),
            self.etichetta("kernel", scala=T_PICCOLO, colore=SECONDARIO).next_to(ker, UP, buff=0.25),
            self.etichetta("mappa", scala=T_PICCOLO, colore=PRIMARIO).next_to(mappa, UP, buff=0.25),
        )

        eq = self.formula(r"(I * K)_{ij} = \sum_{m,n} I_{i+m,\,j+n}\, K_{m,n}", scala=0.8)
        self.centra(VGroup(VGroup(fila, nomi), eq).arrange(DOWN, buff=0.55))

        self.play(entra(img, ker, seg1, seg2, run_time=NORMALE), FadeIn(nomi[0]), FadeIn(nomi[1]))
        self.play(Create(celle_out), FadeIn(nomi[2]), run_time=RAPIDO)
        self.play(entra(eq), run_time=RAPIDO)

        # --- la finestra scorre: nove posizioni, nove numeri ---------------
        def ancora(i, j):
            """Centro della finestra 3×3 con angolo alto-sinistra in (i, j)."""
            return (celle_img[i * 5 + j].get_center()
                    + celle_img[(i + 2) * 5 + j + 2].get_center()) / 2

        finestra = Square(side_length=lato * 3, stroke_color=PRIMARIO, stroke_width=4,
                          fill_color=PRIMARIO, fill_opacity=0.08).move_to(ancora(0, 0))
        self.play(Create(finestra), run_time=RAPIDO)

        for i in range(3):
            for j in range(3):
                if (i, j) != (0, 0):
                    self.play(finestra.animate.move_to(ancora(i, j)),
                              run_time=0.22, rate_func=EASE_OUT)
                v = out[i][j]
                colore = PRIMARIO if v > 0 else (SECONDARIO if v < 0 else TESTO_LIEVE)
                cella = celle_out[i * 3 + j]
                testi_out[i * 3 + j].set_color(colore)
                self.play(cella.animate.set_fill(colore, opacity=0.22 if v else 0.0),
                          FadeIn(testi_out[i * 3 + j], scale=1.2),
                          run_time=0.16)

        self.play(entra(self.didascalia("il kernel scorre: risponde ai bordi, ignora l'interno")),
                  run_time=RAPIDO)
        self.chiusura()
