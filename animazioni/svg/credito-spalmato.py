"""L'assegnazione del credito: un bit alla fine, spalmato su tutta la traiettoria.

La critica di Karpathy al reinforcement learning non si vede in una figura
ferma, perche' e' fatta di tempo: prima si spende una traiettoria lunga, poi
arriva UN numero alla fine, poi quel numero torna indietro e pesa **allo stesso
modo** ogni passo che c'e' stato, compresi i vicoli ciechi. In un fermo immagine
si vedrebbe solo il risultato, cioe' una fila di riquadri tutti uguali, e
sarebbe proprio la cosa che non si capisce.

Qui non c'e' niente di disegnato a mano. I riquadri bordati di terracotta sono
i passi in cui l'agente ha davvero sbagliato secondo la verita' di riferimento
del compito giocattolo, e il peso scritto sotto ogni riquadro e' il
coefficiente vero che lo stimatore REINFORCE mette davanti a quel passo,

    grad J = (1/M) sum_m R^(m) sum_t grad log pi(a_t | s_t),

cioe' il ritorno R della traiettoria, identico per ogni t. Le due righe di
numeri raccontano percio' la stessa cosa da due parti: il **merito** cambia da
passo a passo, il **peso** no. Gli `assert` in `simula()` sono li' perche' se un
giorno la figura smettesse di dire il vero non nascesse affatto: in particolare
verificano che i passi sbagliati ricevano un peso positivo, che e' il punto.

Lo stato di riposo e' l'ultimo istante: tutta la traiettoria percorsa, il
ritorno arrivato, e tutti i passi rinforzati con lo stesso peso.
"""

import random

from paithon_svg import *

NOME = "credito-spalmato"
TITOLO = "un bit alla fine, spalmato su tutta la traiettoria"

PASSI = 12          # lunghezza della traiettoria mostrata
SEME = 7

# Il testo alternativo dichiara dei conteggi, e li deve prendere dal disegno:
# scritti a mano si scollano al primo cambio di seme, e il lettore che ascolta
# la pagina invece di guardarla e' l'unico che non puo' accorgersene.
A_PAROLE = {1: "uno", 2: "due", 3: "tre", 4: "quattro", 5: "cinque",
            6: "sei", 7: "sette", 8: "otto", 9: "nove", 10: "dieci",
            11: "undici", 12: "dodici"}

# --- il compito giocattolo --------------------------------------------------
# A ogni passo c'e' una mossa giusta e una sbagliata (la verita' di
# riferimento). L'agente ne azzecca la maggior parte e finisce comunque bene:
# e' il caso interessante, quello in cui il ritorno e' positivo e gli errori
# lungo la strada non vengono distinti da nulla.


def simula():
    """La traiettoria, il ritorno e il coefficiente REINFORCE di ogni passo."""
    rnd = random.Random(SEME)
    giusta = [rnd.randint(0, 1) for _ in range(PASSI)]     # la mossa corretta
    azioni, sbagliati = [], []
    for t in range(PASSI):
        # l'agente sbaglia ogni tanto: sono i vicoli ciechi del racconto
        errore = rnd.random() < 0.3
        azioni.append(1 - giusta[t] if errore else giusta[t])
        if errore:
            sbagliati.append(t)

    # arriva in fondo lo stesso: il compito premia il risultato, non la strada
    ritorno = 1.0

    # il coefficiente che REINFORCE mette davanti a grad log pi(a_t | s_t):
    # e' R, lo stesso per ogni t. Non c'e' nessun altro termine.
    pesi = [ritorno for _ in range(PASSI)]

    # il merito vero, che l'algoritmo non vede: +1 se la mossa era giusta
    meriti = [1 if azioni[t] == giusta[t] else -1 for t in range(PASSI)]

    # --- guardie: la figura non nasce se smette di dire il vero ------------
    if not sbagliati:
        raise AssertionError("nessun passo sbagliato: la figura non mostra niente")
    if len(sbagliati) == PASSI:
        raise AssertionError("tutti i passi sbagliati: non c'e' contrasto")
    if len(set(pesi)) != 1:
        raise AssertionError("i pesi non sono identici: non e' piu' REINFORCE nudo")
    for t in sbagliati:
        if pesi[t] <= 0:
            raise AssertionError(f"passo {t}: sbagliato ma non rinforzato, "
                                 "la figura perde il suo punto")
        if meriti[t] >= 0:
            raise AssertionError(f"passo {t}: marcato sbagliato ma merito positivo")
    return azioni, sbagliati, meriti, pesi, ritorno


# --- geometria --------------------------------------------------------------
X0, Y0, CW, CH, GAP = 82, 92, 40, 46, 8   # X0 lascia posto a «merito»
RW = PASSI * CW + (PASSI - 1) * GAP
RX = X0 + RW + 26          # il riquadro del ritorno, in fondo alla fila


def cella(t):
    return X0 + t * (CW + GAP)


def costruisci() -> Figura:
    _, sbagliati, meriti, pesi, ritorno = simula()
    n = PASSI + 2              # i passi, l'arrivo del ritorno, la spalmata
    i_ritorno, i_spalma = PASSI, PASSI + 1
    corpo, anim = [], []

    def acceso(k, nome, resta=True):
        """`@keyframes` che accende alla fetta k e (di norma) lascia acceso."""
        t0, _ = sosta(k, n)
        tappe = [(0.0, "opacity:0")]
        if t0 > 0.5:
            tappe.append((max(t0 - 0.4, 0.01), "opacity:0"))
        tappe.append((t0, "opacity:1"))
        tappe.append((100.0, "opacity:1" if resta else "opacity:0"))
        anim.append(keyframes(nome, tappe))
        return nome

    corpo.append(f'<text class="ttl" x="{X0}" y="34">una traiettoria di '
                 f'{PASSI} passi</text>')
    corpo.append(f'<text class="lbs" x="{X0}" y="56">l\'agente sbaglia '
                 f'{len(sbagliati)} volte lungo la strada, e arriva comunque '
                 f'alla risposta giusta</text>')

    # --- la fila dei passi, sempre visibile --------------------------------
    for t in range(PASSI):
        x = cella(t)
        cls = "pas err" if t in sbagliati else "pas"
        corpo.append(f'<rect class="{cls}" x="{x}" y="{Y0}" width="{CW}" '
                     f'height="{CH}" rx="4"/>')
        corpo.append(f'<text class="num" x="{x + CW / 2}" y="{Y0 - 8}" '
                     f'text-anchor="middle">{t + 1}</text>')

    # --- il cammino: una testa che percorre la fila, un passo per fetta ----
    xf = cella(PASSI - 1) + CW / 2
    tappe = []
    for k in range(PASSI):
        t0, t1 = sosta(k, n)
        d = f"transform:translate({cella(k) + CW / 2 - xf:.1f}px,0px)"
        tappe += [(t0, d), (t1, d)]
    tappe.append((100.0, "transform:translate(0px,0px)"))
    anim.append(keyframes("cam", tappe))
    corpo.append(f'<circle class="testa" cx="{xf:.1f}" cy="{Y0 + CH / 2}" r="9" '
                 f'style="animation:cam var(--d) infinite"/>')

    # --- il ritorno: UN numero, e arriva solo alla fine --------------------
    corpo.append(f'<g style="animation:{acceso(i_ritorno, "rit")} var(--d) '
                 f'infinite" opacity="1">'
                 f'<rect class="ret" x="{RX}" y="{Y0}" width="{CW + 14}" '
                 f'height="{CH}" rx="4"/>'
                 f'<text class="retn" x="{RX + (CW + 14) / 2}" '
                 f'y="{Y0 + CH / 2 + 6}" text-anchor="middle">'
                 f'+{ritorno:.0f}</text></g>')
    corpo.append(f'<g style="animation:rit var(--d) infinite" opacity="1">'
                 f'<text class="lbs" x="{RX + (CW + 14) / 2}" y="{Y0 - 8}" '
                 f'text-anchor="middle">ritorno</text></g>')

    # --- la spalmata: dal ritorno a OGNI passo, tutti uguali ---------------
    acceso(i_spalma, "spa")
    y_arco = Y0 + CH + 26
    for t in range(PASSI):
        x = cella(t) + CW / 2
        corpo.append(f'<path class="arc" d="M {RX + (CW + 14) / 2} {Y0 + CH} '
                     f'C {RX} {y_arco + 24}, {x} {y_arco + 24}, {x} {Y0 + CH + 4}" '
                     f'style="animation:spa var(--d) infinite" opacity="1"/>')
        corpo.append(f'<rect class="rin" x="{cella(t)}" y="{Y0}" width="{CW}" '
                     f'height="{CH}" rx="4" '
                     f'style="animation:spa var(--d) infinite" opacity="1"/>')

    # --- le due righe di numeri: il peso e il merito -----------------------
    yp, ym = Y0 + CH + 62, Y0 + CH + 92
    corpo.append(f'<text class="cap" x="{X0 - 6}" y="{yp}" text-anchor="end">'
                 f'peso</text>')
    corpo.append(f'<text class="cap" x="{X0 - 6}" y="{ym}" text-anchor="end">'
                 f'merito</text>')
    for t in range(PASSI):
        x = cella(t) + CW / 2
        corpo.append(f'<text class="peso" x="{x}" y="{yp}" text-anchor="middle" '
                     f'style="animation:spa var(--d) infinite" opacity="1">'
                     f'+{pesi[t]:.0f}</text>')
        cls = "mer neg" if meriti[t] < 0 else "mer"
        corpo.append(f'<text class="{cls}" x="{x}" y="{ym}" text-anchor="middle">'
                     f'{meriti[t]:+d}</text>')

    corpo.append(f'<text class="lbs" x="{X0}" y="{ym + 30}">il peso è lo stesso '
                 f'ovunque; il merito no, e lo stimatore non lo vede</text>')

    return Figura(
        larghezza=RX + CW + 14 + 40, altezza=ym + 52,
        alt=f"Una fila di {A_PAROLE[PASSI]} riquadri, i passi di una "
            f"traiettoria; {A_PAROLE[len(sbagliati)]} sono bordati di "
            "terracotta perché in quei passi l'agente ha sbagliato. Una testa "
            "percorre la fila da sinistra a destra un passo alla volta; alla "
            "fine compare un solo riquadro con il ritorno, più uno. Da lì "
            f"partono {A_PAROLE[PASSI]} archi che tornano indietro fino a ogni "
            "riquadro e li riempiono tutti dello stesso colore. Sotto, due "
            f"righe di numeri: la riga del peso porta più uno {A_PAROLE[PASSI]} "
            "volte identiche, la riga del merito cambia fra più uno e meno uno "
            "da un passo all'altro.",
        corpo="".join(corpo),
        stile=f"""    .pas {{ fill:{CREAM}; stroke:{BORDER_STRONG}; stroke-width:1.5; }}
    .pas.err {{ stroke:{TERRACOTTA}; stroke-width:2.5; }}
    .rin {{ fill:{TEAL}; fill-opacity:0.42; stroke:none; }}
    .ret {{ fill:{OCRA}; stroke:none; }}
    .retn {{ font-family:{SANS}; font-size:19px; font-weight:700; fill:{INK}; }}
    .testa {{ fill:{TEAL}; stroke:{CREAM}; stroke-width:2; }}
    .arc {{ fill:none; stroke:{OCRA}; stroke-width:2; }}
    .num {{ font-family:{SANS}; font-size:12px; fill:{FG_MUTED}; }}
    .cap {{ font-family:{SANS}; font-size:13px; fill:{FG_MUTED}; }}
    .peso {{ font-family:{SANS}; font-size:15px; font-weight:700; fill:{TEAL}; }}
    .mer {{ font-family:{SANS}; font-size:15px; fill:{FG_MUTED}; }}
    .mer.neg {{ fill:{TERRACOTTA}; font-weight:700; }}""",
        animazioni=anim,
        durata=n * 0.9,
        fermi=".testa, .rin, .arc, .ret, .retn, .peso",
    )
