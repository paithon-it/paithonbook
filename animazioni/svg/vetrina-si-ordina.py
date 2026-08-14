"""BPR: la vetrina si ordina a forza di confronti a due a due.

Dieci libri in colonna, un cliente solo, quattro titoli che ha comprato e sei
che ha ignorato. A ogni passo si pesca una coppia (un comprato, un ignorato) e
si guarda chi sta sopra: la sezione «Imparare a ordinare: BPR» racconta
esattamente questo, e la figura la esegue invece di illustrarla.

Niente numeri scritti a mano. Il generatore fa girare BPR per davvero: i
punteggi sono prodotti scalari $\\hat{x}_{ui} = \\mathbf{p}_u^\\top\\mathbf{q}_i$,
i vettori escono da un seme dichiarato (`SEME`), l'aggiornamento è il gradiente
della loss $-\\log\\sigma(\\hat{x}_{uv} - \\hat{x}_{uw})$ e `verifica()` controlla
con degli assert che alla fine i quattro comprati siano davvero i primi quattro.

Tre scelte, e tutte e tre hanno una ragione che si vede nella figura.

**Il vettore dell'utente resta fermo**: si aggiornano solo i fattori degli item.
È la vetrina di *un* cliente, e con $\\mathbf{p}_u$ fisso si muovono soltanto i
due libri della coppia pescata; aggiornando anche $\\mathbf{p}_u$ ogni confronto
sposterebbe di un'inezia anche i libri che non stavano guardando, e chi guarda
si chiederebbe perché. Il gradiente sui $\\mathbf{q}_i$ è quello vero di BPR.

**$\\lambda = 0$**: senza il termine di regolarizzazione l'aggiornamento del
punteggio è esattamente $\\pm\\eta\\,(1 - \\sigma(\\hat{x}_{uv} - \\hat{x}_{uw}))$
(il vettore dell'utente è normalizzato), cioè lo scatto è *proporzionale* alla
spinta, che è la cosa che la figura deve far vedere.

**Ottanta confronti, dodici tappe**: si guardano uno per uno i primi dieci, poi
si salta al risultato. È il ritmo del testo, che dice «ripetuto milioni di
volte»: i primi confronti mostrano il meccanismo, il salto finale mostra che
cosa lascia.

Il seme non è scelto a caso: fra i semi che ordinano la vetrina si è preso
quello che racconta anche le due cose che il testo si impegna a dire, e che
`verifica()` pretende. La prima è la coppia già in ordine, che non muove niente
(i confronti 6 e 8: spinta minima, la vetrina resta ferma). La seconda è
l'obiezione onesta della tab Elementare, «e se il libro pescato a caso era
proprio uno che gli sarebbe piaciuto?»: al confronto 4 il negativo pescato è E,
che in questa storia gli sarebbe piaciuto davvero, scende di un posto per
sbaglio e due confronti dopo è già risalito. Che il modello non lo sappia è il
punto: E chiude comunque primo fra i non comprati.

Lo stato di riposo è l'ultimo: i quattro comprati in cima, i sei ignorati sotto,
il contatore a ottanta e la riga che tiene insieme tutto, «nessuno ha dato un
voto: solo confronti a due a due».
"""

import math
import random

from paithon_svg import *

NOME = "vetrina-si-ordina"
TITOLO = "BPR: la vetrina si ordina a forza di confronti a coppie"

# --------------------------------------------------------------------------
# Il modello, e il seme
# --------------------------------------------------------------------------
SEME = 237
N = 10                      # i libri in vetrina
D = 4                       # i fattori latenti
SIGMA = 1.0                 # scala dei vettori all'inizio
ETA = 0.7                   # passo della discesa
PASSI = 80                  # i confronti in tutto
GUARDATI = 10               # quelli che si vedono uno per uno

LETTERE = "ABCDEFGHIJ"
COMPRATI = (1, 3, 6, 8)                                    # B, D, G, I
IGNORATI = tuple(i for i in range(N) if i not in COMPRATI)

# Quello che il seme deve raccontare, e che `verifica()` non lascia scivolare.
OCRA_K = 4                  # il confronto che pesca un negativo "che sarebbe piaciuto"
OCRA_ITEM = 4               # E: il libro in questione
FERME = (6, 8)              # i confronti in cui la vetrina non si muove
SPINTA_FERMA = 0.12         # sotto questa spinta lo scatto è invisibile


def sigmoide(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def esegui() -> list[dict]:
    """BPR su un utente e dieci item. Uno stato per confronto, dal passo zero.

    L'aggiornamento è il gradiente della loss BPR sui fattori degli item:
    q_v += eta * (1 - sigma(x_uv - x_uw)) * p_u, e q_w lo stesso al contrario.
    """
    rnd = random.Random(SEME)
    p = [rnd.gauss(0, SIGMA) for _ in range(D)]
    norma = math.sqrt(sum(x * x for x in p))
    p = [x / norma for x in p]                 # ||p|| = 1: lo scatto è eta * spinta
    q = [[rnd.gauss(0, SIGMA) for _ in range(D)] for _ in range(N)]

    def punteggi():
        return [sum(p[k] * q[i][k] for k in range(D)) for i in range(N)]

    def ordine():
        x = punteggi()
        return sorted(range(N), key=lambda i: -x[i])

    def loss():
        """La loss media sulle 4 x 6 coppie possibili di questo utente."""
        x = punteggi()
        return (sum(-math.log(sigmoide(x[v] - x[w]))
                    for v in COMPRATI for w in IGNORATI)
                / (len(COMPRATI) * len(IGNORATI)))

    stati = [dict(ordine=ordine(), loss=loss(), v=None, w=None, spinta=None)]
    for _ in range(PASSI):
        v = rnd.choice(COMPRATI)
        w = rnd.choice(IGNORATI)
        x = punteggi()
        spinta = 1.0 - sigmoide(x[v] - x[w])   # quanto la coppia è sbagliata
        for k in range(D):
            q[v][k] += ETA * spinta * p[k]
            q[w][k] -= ETA * spinta * p[k]
        stati.append(dict(ordine=ordine(), loss=loss(), v=v, w=w, spinta=spinta))
    return stati


def rango(stato: dict, i: int) -> int:
    """Il posto in vetrina, da 1 (in cima) a 10."""
    return stato["ordine"].index(i) + 1


def verifica(stati: list[dict]) -> None:
    """Il seme racconta ancora la storia che la figura dichiara?"""
    finale = stati[PASSI]
    assert set(finale["ordine"][:len(COMPRATI)]) == set(COMPRATI), \
        ("dopo {} confronti i primi quattro sono {}, non i comprati {}".format(
            PASSI,
            "".join(LETTERE[i] for i in finale["ordine"][:4]),
            "".join(LETTERE[i] for i in sorted(COMPRATI))))

    assert rango(finale, OCRA_ITEM) == len(COMPRATI) + 1, \
        (f"{LETTERE[OCRA_ITEM]} chiude al posto {rango(finale, OCRA_ITEM)}, "
         f"la figura dice che è il primo dei non comprati")

    partenza = sorted(rango(stati[0], i) for i in COMPRATI)
    assert partenza[-1] >= 8, \
        f"alla partenza nessun comprato è in fondo: posti {partenza}"

    for k in FERME:
        assert stati[k]["spinta"] < SPINTA_FERMA, \
            f"confronto {k}: spinta {stati[k]['spinta']:.2f}, doveva essere minima"
        assert stati[k]["ordine"] == stati[k - 1]["ordine"], \
            f"confronto {k}: la vetrina si muove, la figura dice di no"

    assert stati[OCRA_K]["w"] == OCRA_ITEM, \
        (f"il confronto {OCRA_K} pesca {LETTERE[stati[OCRA_K]['w']]}, "
         f"non {LETTERE[OCRA_ITEM]}")
    prima = rango(stati[OCRA_K - 1], OCRA_ITEM)
    assert rango(stati[OCRA_K], OCRA_ITEM) == prima + 1, \
        f"{LETTERE[OCRA_ITEM]} doveva scendere di una tacca al confronto {OCRA_K}"
    assert rango(stati[OCRA_K + 2], OCRA_ITEM) == prima, \
        f"{LETTERE[OCRA_ITEM]} non è risalito due confronti dopo"

    assert max(rango(stati[k - 1], stati[k]["v"]) - rango(stati[k], stati[k]["v"])
               for k in range(1, GUARDATI + 1)) >= 3, \
        "nessun confronto muove la vetrina di almeno tre posti: si vedrebbe poco"


# --------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------
LARG, ALT = 700, 520

X_SC, Y_SC, W_SC, H_SC = 88, 46, 176, 360      # la cornice della vetrina
X_LIB, W_LIB, H_LIB = 100, 152, 26             # il dorso di un libro
X_ALO, W_ALO, H_ALO = 94, 164, 34              # l'alone della coppia pescata
PASSO_Y = 34                                   # una tacca
CY0 = 73                                       # centro del primo posto

X_P = 300                                      # la colonna di destra
Y_TIT, Y_R1, Y_R2, Y_R3 = 66, 92, 114, 136
W_BAR, H_BAR = 250, 16
Y_SPINTA, Y_LOSS = 190, 282
X_VAL = X_P + W_BAR + 16

COL_LIBRO = {True: TEAL, False: CREAM}


def num(v: float, cifre: int = 2) -> str:
    return f"{v:.{cifre}f}".replace(".", ",")


PAROLE = ("zero", "un", "due", "tre", "quattro", "cinque", "sei", "sette",
          "otto", "nove")


def posti(n: int) -> str:
    return "un posto" if n == 1 else f"{PAROLE[n]} posti"


def giunta(a: str, b: str) -> str:
    """«e» fra due proposizioni, «ed» davanti a una e: qui capita, i libri
    hanno per nome una lettera."""
    return f"{a} e{'d' if b[:1].lower() == 'e' else ''} {b}"


def elenco(numeri) -> str:
    """«2, 7, 8 e 9», come si scrive in italiano."""
    v = [str(x) for x in numeri]
    return v[0] if len(v) == 1 else ", ".join(v[:-1]) + " e " + v[-1]


# --------------------------------------------------------------------------
# Il racconto: ogni riga è calcolata dallo stato, nessuna è scritta a mano
# --------------------------------------------------------------------------
def righe(stati, mostrati, s):
    """(titolo, riga1, riga2, riga3-ocra) della tappa s."""
    k = mostrati[s]
    st = stati[k]
    if k == 0:
        return ("partenza",
                "punteggi presi a caso: la vetrina non sa niente",
                "i comprati stanno ai posti "
                + elenco(sorted(rango(st, i) for i in COMPRATI)),
                None, None)

    if k == PASSI:
        return (f"dopo {PASSI} confronti",
                "i quattro comprati sono i primi quattro",
                f"{LETTERE[OCRA_ITEM]}, che gli sarebbe piaciuto, chiude quinto",
                f"l'ultimo confronto spinge {num(st['spinta'])}: quasi niente",
                FG_MUTED)

    v, w = st["v"], st["w"]
    prec = stati[k - 1]
    su = rango(prec, v) - rango(st, v)
    giu = rango(st, w) - rango(prec, w)

    mosse = []
    if su:
        mosse.append(f"{LETTERE[v]} sale di {posti(su)}")
    if giu:
        mosse.append(f"{LETTERE[w]} scende di {posti(giu)}")
    esito = (giunta(*mosse) if len(mosse) == 2 else
             mosse[0] if mosse else "la vetrina non si muove")

    nota, colore = None, None
    if k == OCRA_K:
        nota = f"{LETTERE[w]} gli sarebbe piaciuto: lo spingiamo giù"
        colore = OCRA
    elif k == OCRA_K + 2:
        nota = f"{LETTERE[OCRA_ITEM]} è già risalito: uno sbaglio non conta"
        colore = OCRA

    return (f"confronto {k}",
            f"{LETTERE[v]} comprato · {LETTERE[w]} ignorato",
            esito, nota, colore)


# --------------------------------------------------------------------------
# Animazione: le tappe della timeline
# --------------------------------------------------------------------------
def scorre(valori, passo, quota=0.38):
    """(tempo, valore) per una successione che transita dentro la propria fetta.

    Il valore della tappa s arriva presto nella fetta e poi si tiene: prima lo
    scatto, poi il tempo di leggerlo.
    """
    tappe = [(0.0, valori[0])]
    for s in range(1, len(valori)):
        if valori[s] == valori[s - 1]:
            continue
        t0 = s * passo
        tappe += [(t0, valori[s - 1]), (min(t0 + passo * quota, 99.9), valori[s])]
    tappe.append((100.0, valori[-1]))
    return tappe


def accende(s, n, passo):
    """Opacità di ciò che vale solo nella tappa s: l'ultima resta accesa."""
    t0 = s * passo
    if s == 0:
        return [(0.0, "opacity:1"), (passo * 0.94, "opacity:1"),
                (passo, "opacity:0"), (100.0, "opacity:0")]
    if s == n - 1:
        return [(0.0, "opacity:0"), (t0 - passo * 0.09, "opacity:0"),
                (t0, "opacity:1"), (100.0, "opacity:1")]
    return [(0.0, "opacity:0"), (t0 - passo * 0.09, "opacity:0"),
            (t0, "opacity:1"), (t0 + passo * 0.94, "opacity:1"),
            (t0 + passo, "opacity:0"), (100.0, "opacity:0")]


# --------------------------------------------------------------------------
# La figura
# --------------------------------------------------------------------------
def costruisci() -> Figura:
    stati = esegui()
    verifica(stati)

    # dodici tappe: la partenza, i primi dieci confronti, il risultato
    mostrati = [0] + list(range(1, GUARDATI + 1)) + [PASSI]
    n = len(mostrati)
    passo = 100.0 / n
    vista = [stati[k] for k in mostrati]

    corpo, anim = [], []

    # ---- la vetrina -------------------------------------------------------
    corpo.append(f'<text class="lbs" x="{X_SC}" y="34">la vetrina, '
                 f'dall\'alto in basso</text>')
    corpo.append(f'<rect class="ax" x="{X_SC}" y="{Y_SC}" width="{W_SC}" '
                 f'height="{H_SC}" rx="4"/>')
    for posto in range(N):
        cy = CY0 + posto * PASSO_Y
        corpo.append(f'<text class="tic" x="{X_SC - 8}" y="{cy + 5}" '
                     f'text-anchor="end">{posto + 1}</text>')

    # ---- i libri: disegnati dove finiscono, l'animazione parte dall'inverso
    for i in range(N):
        comprato = i in COMPRATI
        finale = vista[-1]["ordine"].index(i)
        cy = CY0 + finale * PASSO_Y
        scarti = [f"transform:translateY({(s['ordine'].index(i) - finale) * PASSO_Y}px)"
                  for s in vista]

        dentro = []

        # l'alone della coppia: acceso solo nelle tappe in cui questo libro
        # è quello pescato, terracotta da comprato e ocra nello sbaglio
        tinte = []
        for s, st in enumerate(vista):
            if st["v"] == i:
                tinte.append((s, TERRACOTTA))
            elif st["w"] == i:
                tinte.append((s, OCRA if mostrati[s] == OCRA_K else FG_MUTED))
        # l'ultima tappa non accende niente: il riposo è una vetrina in pace
        tinte = [(s, c) for s, c in tinte if s not in (0, n - 1)]
        if tinte:
            tappe = [(0.0, "stroke-opacity:0")]
            for s, colore in tinte:
                t0 = s * passo
                acceso = f"stroke:{colore};stroke-opacity:0.95"
                tappe += [(max(t0 - passo * 0.09, 0.01), f"stroke:{colore};stroke-opacity:0"),
                          (t0, acceso),
                          (min(t0 + passo * 0.92, 99.8), acceso),
                          (min(t0 + passo, 99.9), f"stroke:{colore};stroke-opacity:0")]
            tappe.append((100.0, "stroke-opacity:0"))
            tappe.sort(key=lambda x: x[0])
            anim.append(keyframes(f"alo{i}", tappe))
            dentro.append(f'<rect class="alo" x="{X_ALO}" y="{cy - H_ALO / 2:.0f}" '
                          f'width="{W_ALO}" height="{H_ALO}" rx="7" '
                          f'style="animation:alo{i} var(--d) infinite"/>')

        dentro.append(f'<rect class="lib{"" if comprato else " ign"}" x="{X_LIB}" '
                      f'y="{cy - H_LIB / 2:.0f}" width="{W_LIB}" height="{H_LIB}" '
                      f'rx="4" fill="{COL_LIBRO[comprato]}"/>')
        dentro.append(f'<text class="let" x="{X_LIB + 14}" y="{cy + 5}" '
                      f'fill="{CREAM if comprato else INK}">{LETTERE[i]}</text>')
        dentro.append(f'<line class="rig" x1="{X_LIB + 32}" y1="{cy}" '
                      f'x2="{X_LIB + W_LIB - (26 if i == OCRA_ITEM else 12)}" '
                      f'y2="{cy}" stroke="{CREAM if comprato else BORDER_STRONG}" '
                      f'opacity="{0.45 if comprato else 1}"/>')
        if i == OCRA_ITEM:
            dentro.append(f'<circle class="seg" cx="{X_LIB + W_LIB - 14}" '
                          f'cy="{cy}" r="5"/>')

        moto = ""
        if any(s != scarti[-1] for s in scarti):
            anim.append(keyframes(f"lib{i}", scorre(scarti, passo)))
            moto = f' style="animation:lib{i} var(--d) infinite"'
        corpo.append(f'<g class="libro"{moto}>{"".join(dentro)}</g>')

    # ---- la legenda -------------------------------------------------------
    for x, colore, cls, testo in (
            (X_SC, TEAL, "lib", "i quattro comprati"),
            (X_SC + 176, CREAM, "lib ign", "i sei ignorati")):
        corpo.append(f'<rect class="{cls}" x="{x}" y="{424}" width="16" '
                     f'height="16" rx="3" fill="{colore}"/>')
        corpo.append(f'<text class="lbs" x="{x + 24}" y="{437}">{testo}</text>')
    corpo.append(f'<circle class="seg" cx="{X_SC + 8}" cy="{462}" r="5"/>')
    corpo.append(f'<text class="lbs" x="{X_SC + 24}" y="{467}">'
                 f'{LETTERE[OCRA_ITEM]} gli sarebbe piaciuto, ma il modello non lo '
                 f'sa: non l\'ha mai comprato</text>')

    # ---- la colonna di destra: le etichette fisse -------------------------
    corpo += [
        f'<text class="lbs" x="{X_P}" y="{Y_SPINTA - 10}">spinta su questa coppia</text>',
        f'<text class="min" x="{X_P}" y="{Y_SPINTA + H_BAR + 22}">'
        f'0 = già in ordine · 1 = del tutto rovesciata</text>',
        f'<text class="lbs" x="{X_P}" y="{Y_LOSS - 10}">loss media</text>',
        f'<text class="min" x="{X_P}" y="{Y_LOSS + H_BAR + 22}">'
        f'sulle 24 coppie possibili: 4 comprati × 6 ignorati</text>',
        f'<text class="lbs" x="{X_P}" y="{360}">confronti fatti</text>',
        f'<rect class="trk" x="{X_P}" y="{Y_SPINTA}" width="{W_BAR}" '
        f'height="{H_BAR}" rx="8"/>',
        f'<rect class="trk" x="{X_P}" y="{Y_LOSS}" width="{W_BAR}" '
        f'height="{H_BAR}" rx="8"/>',
    ]

    # ---- le due barre: la spinta scatta, la loss scivola ------------------
    #      La larghezza a riposo è quella finale (una lisca: dopo ottanta
    #      confronti la spinta e la loss sono quasi niente), e l'animazione
    #      ripercorre le larghezze di prima. Si anima `width` e non una
    #      scaleX perché una barra larga sette pixel scalata per trenta si
    #      porterebbe dietro gli angoli arrotondati, stirati in ellissi.
    spinte = [0.0 if s["spinta"] is None else s["spinta"] for s in vista]
    perdite = [s["loss"] / vista[0]["loss"] for s in vista]
    for cls, valori, quota, y in (("bsp", spinte, 0.04, Y_SPINTA),
                                  ("blo", perdite, 0.38, Y_LOSS)):
        larghezze = [f"{max(v, 0.0) * W_BAR:.1f}" for v in valori]
        anim.append(keyframes(cls, [(t, f"width:{v}px")
                                    for t, v in scorre(larghezze, passo, quota)]))
        corpo.append(f'<rect class="bar {cls}" x="{X_P}" y="{y}" '
                     f'width="{larghezze[-1]}" height="{H_BAR}" rx="8" '
                     f'style="animation:{cls} var(--d) infinite"/>')

    # ---- ciò che cambia a ogni tappa --------------------------------------
    for s in range(n):
        anim.append(keyframes(f"eti{s}", accende(s, n, passo)))
        fermo = ";opacity:1" if s == n - 1 else ""
        moto = f'style="animation:eti{s} var(--d) infinite{fermo}"'
        titolo, r1, r2, r3, colore = righe(stati, mostrati, s)

        corpo += [
            f'<text class="pas" x="{X_P}" y="{Y_TIT}" {moto}>{titolo}</text>',
            f'<text class="spg" x="{X_P}" y="{Y_R1}" {moto}>{r1}</text>',
            f'<text class="spg" x="{X_P}" y="{Y_R2}" {moto}>{r2}</text>',
        ]
        if r3:
            corpo.append(f'<text class="spg" x="{X_P}" y="{Y_R3}" fill="{colore}" '
                         f'{moto}>{r3}</text>')

        if vista[s]["spinta"] is not None:
            corpo.append(f'<text class="val" x="{X_VAL}" y="{Y_SPINTA + H_BAR - 2}" '
                         f'fill="{TERRACOTTA}" {moto}>{num(vista[s]["spinta"])}</text>')
        corpo.append(f'<text class="val" x="{X_VAL}" y="{Y_LOSS + H_BAR - 2}" '
                     f'fill="{TEAL}" {moto}>{num(vista[s]["loss"])}</text>')
        corpo.append(f'<text class="cnt" x="{X_P}" y="{398}" {moto}>'
                     f'{mostrati[s]}</text>')

    corpo.append(f'<text class="fin" x="{X_SC}" y="{500}">nessuno ha dato un voto: '
                 f'solo confronti a due a due</text>')

    ordine_finale = ", ".join(LETTERE[i] for i in vista[-1]["ordine"])
    in_fondo = sum(1 for i in COMPRATI if rango(vista[0], i) > N // 2)
    return Figura(
        larghezza=LARG, altezza=ALT,
        alt=f"Una vetrina di {N} libri in colonna, dal posto 1 al posto {N}. "
            f"I quattro che il cliente ha comprato ({', '.join(LETTERE[i] for i in sorted(COMPRATI))}) "
            f"partono sparsi, {'uno' if in_fondo == 1 else PAROLE[in_fondo]} di loro "
            "nella metà bassa. "
            "A ogni confronto si pesca una "
            "coppia formata da un libro comprato e da uno ignorato: se il "
            "comprato sta già sopra la spinta è quasi nulla e la vetrina non si "
            "muove, se sta sotto sale di uno o più posti e l'ignorato scende. Al "
            f"quarto confronto l'ignorato pescato è {LETTERE[OCRA_ITEM]}, che al "
            "cliente sarebbe piaciuto: scende di un posto per sbaglio e due "
            "confronti dopo è già risalito. Dopo ottanta confronti l'ordine è "
            f"{ordine_finale}: i quattro comprati sono i primi quattro, e il "
            f"primo dei non comprati è proprio {LETTERE[OCRA_ITEM]}. La loss media "
            f"scende da {num(vista[0]['loss'])} a {num(vista[-1]['loss'])}, e "
            "nessuno ha mai dato un voto.",
        corpo="".join(corpo),
        stile=f"""    .lib  {{ stroke:{TEAL}; stroke-width:1.5; }}
    .lib.ign {{ stroke:{BORDER_STRONG}; stroke-width:1.5; }}
    .alo  {{ fill:none; stroke-width:2.5; stroke-opacity:0; }}
    .rig  {{ stroke-width:2; stroke-linecap:round; }}
    .seg  {{ fill:{OCRA}; }}
    .let  {{ font-family:{SANS}; font-size:15px; font-weight:700; }}
    .tic  {{ font-family:{SANS}; font-size:12px; fill:{FG_MUTED}; }}
    .min  {{ font-family:{SANS}; font-size:12px; fill:{FG_MUTED}; }}
    .trk  {{ fill:{BORDER}; }}
    .bsp  {{ fill:{TERRACOTTA}; }}
    .blo  {{ fill:{TEAL}; }}
    .pas  {{ font-family:{SANS}; font-size:17px; font-weight:700;
            fill:{TERRACOTTA}; opacity:0; }}
    .spg  {{ font-family:{SANS}; font-size:13.5px; fill:{FG_MUTED}; opacity:0; }}
    .val  {{ font-family:{SANS}; font-size:19px; font-weight:700; opacity:0; }}
    .cnt  {{ font-family:{SANS}; font-size:34px; font-weight:700;
            fill:{TEAL}; opacity:0; }}
    .fin  {{ font-family:{SANS}; font-size:15px; fill:{INK}; }}""",
        animazioni=anim,
        durata=n * 1.5,
        fermi=".libro, .alo, .bar, .pas, .spg, .val, .cnt",
    )
