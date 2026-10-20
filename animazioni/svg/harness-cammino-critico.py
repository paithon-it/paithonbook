"""Lo stesso lavoro in tre modi: in serie, in parallelo, con le azioni speculative.

La sezione sul tempo di un agente mette in fila tre letture indipendenti e una
risposta, e ne calcola la durata in tre modi: in serie, con le letture
raggruppate in un passo, e con un modello rapido che prova a indovinare la
lettura dopo mentre il modello principale sta ancora pensando. Il programma
stampa tre totali; quello che non può stampare è **come ci si arriva**, cioè
quali pezzi si sovrappongono e quali si aspettano. Qui il tempo è il
contenuto: le tre corsie avanzano insieme sotto lo stesso cursore, e ciascuna
finisce quando finisce la sua catena più lunga.

I numeri non sono scritti a mano: questo file **esegue il blocco di codice
della sezione**, preso dal `.md`, e ne usa le costanti (PENSA, LEGGI,
INDOVINA, quali previsioni sono giuste) e la funzione `passo()`. Le tre linee
del tempo si costruiscono qui, blocco per blocco, e `controlla()` le confronta
con ciò che il programma stampa: la durata di ogni passo speculativo deve
essere quella che `passo()` restituisce, e le tre corsie devono finire sui tre
totali del blocco ```text. Le asserzioni difendono quello che didascalia e
`:alt:` promettono, e l'`:alt:` che la figura calcola deve coincidere parola per
parola con quello scritto nella pagina.

Lo stato di riposo è il lavoro concluso: le tre corsie complete, la lettura
buttata già segnata come scartata, i tre totali in vista e il cursore in fondo
alla scala. L'animazione scopre i blocchi con un ritaglio che parte schiacciato
sul bordo sinistro e finisce sull'identità, e il cursore scorre dall'inizio
della scala alla fine; tutti e due vanno con andamento lineare, perché le
tappe di un `@keyframes` hanno di default un `ease`, e cursore e ritaglio si
scollerebbero fra una tappa e l'altra.
"""

import contextlib
import io
import re
from pathlib import Path

from paithon_svg import *

NOME = "harness-cammino-critico"
TITOLO = "lo stesso lavoro in serie, in parallelo e con le azioni speculative"

QUI = Path(__file__).resolve().parent
SEZIONE = QUI.parents[1] / "book" / "Harness" / "il-conto.md"
MARCA = "def passo(giusta):"

CARDINALI = {1: "un", 2: "due", 3: "tre", 4: "quattro", 5: "cinque"}
ORDINALI_F = {1: "prima", 2: "seconda", 3: "terza", 4: "quarta"}
ORDINALI_M = {1: "primo", 2: "secondo", 3: "terzo", 4: "quarto"}


# --------------------------------------------------------------------------
# Il codice del capitolo, eseguito davvero
# --------------------------------------------------------------------------
def _testo() -> str:
    return SEZIONE.read_text(encoding="utf-8")


def _blocco(marcatore: str) -> str:
    """Il blocco con la marca (``python``) o la sua uscita (``text``)."""
    testo = _testo()
    trovati = [m for m in re.finditer(r"^```python\n(.*?)^```", testo,
                                      re.S | re.M)
               if MARCA in m.group(1)]
    if len(trovati) != 1:
        raise ValueError(f"{NOME}: {len(trovati)} blocchi ```python con "
                         f"{MARCA!r} in {SEZIONE.name}, ne serve esattamente 1")
    codice = trovati[0]
    if marcatore == "python":
        return codice.group(1)
    uscita = re.match(r"\n*```text\n(.*?)^```", testo[codice.end():],
                      re.S | re.M)
    if not uscita:
        raise ValueError(f"{NOME}: il blocco in {SEZIONE.name} non ha il suo "
                         f"```text subito dopo")
    return uscita.group(1)


def alt_della_pagina() -> str:
    """L'`:alt:` che la pagina scrive per questa figura."""
    figure = re.findall(r"```\{figure\} \.\./figures/" + re.escape(NOME)
                        + r"\.svg\n(.*?)```", _testo(), re.S)
    if len(figure) != 1:
        raise ValueError(f"{NOME}: {len(figure)} direttive figure per "
                         f"{NOME}.svg in {SEZIONE.name}, ne serve 1")
    alt = re.findall(r'^:alt: "(.*)"$', figure[0], re.M)
    if len(alt) != 1:
        raise ValueError(f"{NOME}: la direttiva non ha un :alt: fra "
                         f"virgolette su una riga sola")
    return alt[0]


def esegui_il_capitolo():
    codice = _blocco("python")
    spazio = {"__name__": "il_conto"}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(codice, str(SEZIONE), "exec"), spazio)   # noqa: S102
    return spazio, buf.getvalue().splitlines()


# --------------------------------------------------------------------------
# Le tre linee del tempo, dalle costanti del capitolo
# --------------------------------------------------------------------------
# Un blocco è (tipo, riga, inizio, fine, etichetta). I tipi: «pensa» per il
# modello principale, «prevede» per il modello rapido, «legge» per una
# lettura, «scartata» per la lettura avviata su una previsione sbagliata.
# Le righe: 0 è il modello, poi vengono le letture (e, nella corsia
# speculativa, il modello rapido in mezzo).
def in_serie(s):
    blocchi, t = [], 0
    for nome in s["LETTURE"]:
        blocchi.append(("pensa", 0, t, t + s["PENSA"], ""))
        t += s["PENSA"]
        blocchi.append(("legge", 1, t, t + s["LEGGI"], nome))
        t += s["LEGGI"]
    blocchi.append(("pensa", 0, t, t + s["PENSA"], "risposta"))
    return blocchi, t + s["PENSA"]


def in_parallelo(s):
    pensa, leggi = s["PENSA"], s["LEGGI"]
    blocchi = [("pensa", 0, 0, pensa, "")]
    for i, nome in enumerate(s["LETTURE"]):          # tre letture, un passo
        blocchi.append(("legge", 1 + i, pensa, pensa + leggi, nome))
    blocchi.append(("pensa", 0, pensa + leggi, 2 * pensa + leggi, "risposta"))
    return blocchi, 2 * pensa + leggi


def speculativo(s):
    """Un passo per lettura, come `passo()` del capitolo, più la risposta.

    Il modello rapido comincia a indovinare quando il modello principale
    comincia a pensare, e la lettura prevista parte appena ha indovinato. Se
    la previsione è giusta, il passo finisce quando finiscono tutti e due;
    se è sbagliata, la lettura prevista si butta quando il modello chiede
    l'altra, e quella vera parte allora.
    """
    pensa, leggi, indovina = s["PENSA"], s["LEGGI"], s["INDOVINA"]
    blocchi, fini, scarti, t = [], [], [], 0
    for nome, giusta in s["LETTURE"].items():
        blocchi.append(("pensa", 0, t, t + pensa, ""))
        blocchi.append(("prevede", 1, t, t + indovina, ""))
        if giusta:
            blocchi.append(("legge", 2, t + indovina, t + indovina + leggi,
                            nome))
            fine = t + max(pensa, indovina + leggi)
        else:
            blocchi.append(("scartata", 2, t + indovina, t + indovina + leggi,
                            "scartata"))
            scarti.append(t + pensa)               # il modello chiede l'altra
            blocchi.append(("legge", 2, t + pensa, t + pensa + leggi, nome))
            fine = t + pensa + leggi
        if fine - t != s["passo"](giusta):
            raise ValueError(f"{NOME}: il passo di {nome} dura {fine - t} s "
                             f"nella figura e {s['passo'](giusta)} s nel "
                             f"capitolo")
        fini.append(fine)
        t = fine
    blocchi.append(("pensa", 0, t, t + pensa, "risposta"))
    fini.append(t + pensa)
    return blocchi, t + pensa, fini, scarti


def controlla(s, stampate, corsie, fini, scarti):
    atteso = _blocco("text").splitlines()
    if stampate != atteso:
        raise ValueError(f"{NOME}: il capitolo stampa righe diverse dal suo "
                         f"blocco di uscita:\n  {stampate}\n  {atteso}")

    (serie, f_serie), (par, f_par), (spec, f_spec) = corsie
    # i tre totali: quelli del programma, e quelli della didascalia
    assert (f_serie, f_par, f_spec) == (s["in_serie"], s["in_parallelo"],
                                         s["speculativo"]), \
        f"le corsie finiscono a {f_serie}, {f_par}, {f_spec}"
    assert (f_serie, f_par, f_spec) == (25, 11, 19), \
        f"la didascalia promette 25, 11, 19: {f_serie}, {f_par}, {f_spec}"
    assert fini == [4, 11, 15, 19], f"i passi speculativi finiscono a {fini}"

    def conta(blocchi, tipo):
        return sum(b[0] == tipo for b in blocchi)

    # in serie: quattro pensieri da PENSA e tre letture da LEGGI, alternati
    assert conta(serie, "pensa") == 4 and conta(serie, "legge") == 3
    ordinati = sorted(serie, key=lambda b: b[2])
    assert [b[0] for b in ordinati] == ["pensa", "legge"] * 3 + ["pensa"]
    assert all(b[3] - b[2] == (s["PENSA"] if b[0] == "pensa" else s["LEGGI"])
               for b in serie)
    # in parallelo: le tre letture condividono lo stesso intervallo
    letture = [b for b in par if b[0] == "legge"]
    assert len(letture) == 3 and conta(par, "pensa") == 2
    assert {(b[2], b[3]) for b in letture} == {(4, 7)}, letture
    # speculativo: tre previsioni da INDOVINA, una all'inizio di ciascuno dei
    # primi tre pensieri; il quarto, la risposta, non ne ha
    pensieri = sorted((b for b in spec if b[0] == "pensa"), key=lambda b: b[2])
    previsioni = sorted((b for b in spec if b[0] == "prevede"),
                        key=lambda b: b[2])
    assert len(pensieri) == 4 and len(previsioni) == 3
    assert [p[2] for p in previsioni] == [p[2] for p in pensieri[:3]]
    assert all(p[3] - p[2] == s["INDOVINA"] == 1 for p in previsioni)
    assert pensieri[-1][4] == "risposta"
    # la seconda sbagliata e scartata, la sola; la prima e la terza giuste,
    # con la lettura che finisce insieme al pensiero
    giuste = [i for i, g in enumerate(s["LETTURE"].values(), 1) if g]
    assert giuste == [1, 3] and s["sbagliate"] == 1 == len(scarti)
    scartate = [b for b in spec if b[0] == "scartata"]
    assert len(scartate) == 1 and scartate[0][2] == previsioni[1][3]
    vere = sorted((b for b in spec if b[0] == "legge"), key=lambda b: b[2])
    for i in (0, 2):
        assert vere[i][2] == previsioni[i][3], "la lettura parte subito dopo"
        assert vere[i][3] == pensieri[i][3], "finisce insieme al pensiero"
    assert vere[1][2] == pensieri[1][3] == scarti[0], \
        "la lettura vera parte quando il pensiero finisce"
    # nella stessa riga della stessa corsia nessun blocco si sovrappone
    for blocchi, _ in corsie:
        for riga in {b[1] for b in blocchi}:
            pezzi = sorted((b[2], b[3]) for b in blocchi if b[1] == riga)
            assert all(a[1] <= b[0] for a, b in zip(pezzi, pezzi[1:])), pezzi


def alt_calcolato(s, corsie) -> str:
    (serie, f_serie), (par, f_par), (spec, f_spec) = corsie
    n_pensieri = sum(b[0] == "pensa" for b in serie)
    n_letture = len(s["LETTURE"])
    giuste = [i for i, g in enumerate(s["LETTURE"].values(), 1) if g]
    sbagliata = [i for i, g in enumerate(s["LETTURE"].values(), 1) if not g]
    n_spec = sum(b[0] == "pensa" for b in spec)
    unita = "secondo" if s["INDOVINA"] == 1 else "secondi"
    return (
        f"Tre corsie orizzontali sulla stessa scala dei tempi, da 0 a "
        f"{max(f_serie, f_par, f_spec)} secondi, attraversate da un cursore "
        f"verticale che avanza a velocità costante. Nella prima, in serie, si "
        f"alternano {CARDINALI[n_pensieri]} blocchi di pensiero del modello "
        f"da {s['PENSA']} secondi e {CARDINALI[n_letture]} letture da "
        f"{s['LEGGI']}: il lavoro finisce a {f_serie}. Nella seconda, in "
        f"parallelo, un pensiero, poi le {CARDINALI[n_letture]} letture "
        f"impilate l'una sull'altra nello stesso intervallo, poi l'ultimo "
        f"pensiero: finisce a {f_par}. Nella terza, speculativa, all'inizio "
        f"di ciascuno dei primi {CARDINALI[n_letture]} pensieri parte una "
        f"previsione da {s['INDOVINA']} {unita} e subito dopo la lettura "
        f"prevista; "
        + " e ".join(f"la {ORDINALI_F[i]}" for i in giuste)
        + f" previsione sono giuste e la loro lettura finisce insieme al "
        f"pensiero, la {ORDINALI_F[sbagliata[0]]} è sbagliata, la sua "
        f"lettura viene scartata e quella vera parte dopo il pensiero; il "
        f"{ORDINALI_M[n_spec]} pensiero, che scrive la risposta, non ha "
        f"previsione: finisce a {f_spec}.")


# --------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------
W = 800
X0, X1 = 130, 772                # la scala dei tempi
RH, PASSO_RIGA = 20, 24          # altezza di una riga e passo fra le righe
Y_LEG = 56                       # la legenda
Y_CORSIE = [100, 192, 332]       # la prima riga di ciascuna corsia
RIGHE = [2, 4, 3]                # quante righe ha ciascuna corsia
NOMI_RIGHE = [
    {0: "modello", 1: "letture"},
    {0: "modello", 2: "letture"},
    {0: "modello", 1: "modello rapido", 2: "letture"},
]
NOMI_CORSIE = ["in serie", "in parallelo", "speculativo"]
RIVELA = 0.80                    # la frazione del ciclo in cui il cursore corre


def costruisci() -> Figura:
    s, stampate = esegui_il_capitolo()
    serie, f_serie = in_serie(s)
    par, f_par = in_parallelo(s)
    spec, f_spec, fini, scarti = speculativo(s)
    corsie = [(serie, f_serie), (par, f_par), (spec, f_spec)]
    controlla(s, stampate, corsie, fini, scarti)

    alt = alt_calcolato(s, corsie)
    nella_pagina = alt_della_pagina()
    if alt != nella_pagina:
        raise ValueError(f"{NOME}: l'aria-label calcolato e l':alt: della "
                         f"pagina non coincidono:\n  {alt}\n  {nella_pagina}")

    scala = max(f_serie, f_par, f_spec)
    px_s = (X1 - X0) / scala

    def px(t: float) -> float:
        return X0 + t * px_s

    def pc(t: float) -> float:
        """L'istante t della scala, in percentuale del ciclo."""
        return t / scala * RIVELA * 100

    def y_riga(k: int, riga: int) -> float:
        return Y_CORSIE[k] + riga * PASSO_RIGA

    chiamate = len(s["LETTURE"]) + 1
    totali = [f"{f_serie} s, {chiamate} chiamate al modello",
              f"{f_par} s, 2 chiamate al modello",
              f"{f_spec} s, {chiamate} chiamate al modello, "
              f"{len(s['LETTURE'])} previsioni, {s['sbagliate']} lettura "
              f"buttata"]
    # i totali sono quelli che il programma stampa, parola per parola
    for riga, totale in zip(stampate[:3], totali):
        assert riga.split(": ", 1)[1] == totale, (riga, totale)

    corpo, anim = [], []
    anim.append(keyframes("scopri", [
        (0.0, "transform:scaleX(0)"), (RIVELA * 100, "transform:scaleX(1)"),
        (100.0, "transform:scaleX(1)")]))
    anim.append(keyframes("scorre", [
        (0.0, f"transform:translateX({X0 - X1}px)"),
        (RIVELA * 100, "transform:translateX(0px)"),
        (100.0, "transform:translateX(0px)")]))

    corpo.append('<text class="ttl" x="16" y="26">quanto dura lo stesso '
                 'lavoro: tre letture e una risposta, in tre modi</text>')

    # la legenda
    voci = [("b-pensa", "pensiero del modello"), ("b-legge", "lettura"),
            ("b-prevede", "previsione del modello rapido"),
            ("b-scartata", "lettura scartata")]
    x = 16
    for classe, testo in voci:
        corpo.append(f'<rect class="{classe}" x="{x}" y="{Y_LEG - 10}" '
                     f'width="22" height="12" rx="2"/>')
        corpo.append(f'<text class="lbs" x="{x + 30}" y="{Y_LEG}">{testo}'
                     f'</text>')
        x += 30 + 7.0 * len(testo) + 26

    # lo sfondo di ogni corsia: il nome, le righe, la griglia dei secondi
    for k in range(3):
        y_top = Y_CORSIE[k]
        y_bot = y_riga(k, RIGHE[k] - 1) + RH
        corpo.append(f'<text class="lbl nome" x="16" y="{y_top - 12}">'
                     f'{NOMI_CORSIE[k]}</text>')
        for riga, nome in NOMI_RIGHE[k].items():
            corpo.append(f'<text class="lbs" x="{X0 - 10}" '
                         f'y="{y_riga(k, riga) + 14}" text-anchor="end">'
                         f'{nome}</text>')
        for t in range(0, scala + 1, 5):
            corpo.append(f'<line class="grig" x1="{px(t):.1f}" '
                         f'y1="{y_top - 4}" x2="{px(t):.1f}" '
                         f'y2="{y_bot + 4}"/>')

    # i blocchi, dentro il ritaglio che li scopre
    corpo.append(f'<clipPath id="cscopri"><rect class="rit" x="{X0 - 2}" '
                 f'y="{Y_CORSIE[0] - 30}" width="{X1 - X0 + 4}" '
                 f'height="{y_riga(2, 2) + RH + 10 - Y_CORSIE[0] + 30}" '
                 f'style="animation:scopri var(--d) linear infinite"/>'
                 f'</clipPath>')
    corpo.append('<g clip-path="url(#cscopri)">')
    for k, (blocchi, fine) in enumerate(corsie):
        for tipo, riga, t0, t1, etichetta in blocchi:
            x0, x1 = px(t0) + 1, px(t1) - 1
            y = y_riga(k, riga)
            xm = (x0 + x1) / 2
            rett = (f'x="{x0:.1f}" y="{y}" width="{x1 - x0:.1f}" '
                    f'height="{RH}" rx="3"')
            if tipo == "scartata":
                # prima è una lettura come le altre; quando il modello chiede
                # l'altra, diventa scartata
                t_via = pc(scarti[0])
                anim.append(keyframes("prima", [
                    (0.0, "opacity:1"), (t_via, "opacity:1"),
                    (t_via + 0.6, "opacity:0"), (100.0, "opacity:0")]))
                anim.append(keyframes("dopo", [
                    (0.0, "opacity:0"), (t_via, "opacity:0"),
                    (t_via + 0.6, "opacity:1"), (100.0, "opacity:1")]))
                corpo.append(f'<g class="cambio" opacity="0" style="animation:'
                             f'prima var(--d) infinite"><rect class="b-legge" '
                             f'{rett}/></g>')
                corpo.append(f'<g class="cambio" opacity="1" style="animation:'
                             f'dopo var(--d) infinite"><rect class="b-scartata"'
                             f' {rett}/><text class="t-scartata" x="{xm:.1f}" '
                             f'y="{y + 14}" text-anchor="middle">{etichetta}'
                             f'</text></g>')
                continue
            corpo.append(f'<rect class="b-{tipo}" {rett}/>')
            if etichetta:
                classe = "t-risposta" if tipo == "pensa" else "t-legge"
                corpo.append(f'<text class="{classe}" x="{xm:.1f}" '
                             f'y="{y + 14.5}" text-anchor="middle">'
                             f'{etichetta}</text>')
        # dove la corsia finisce
        y_top = Y_CORSIE[k]
        y_bot = y_riga(k, RIGHE[k] - 1) + RH
        corpo.append(f'<line class="fine" x1="{px(fine):.1f}" '
                     f'y1="{y_top - 26}" x2="{px(fine):.1f}" '
                     f'y2="{y_bot + 5}"/>')
    corpo.append('</g>')

    # i totali, che compaiono quando il cursore arriva in fondo alla corsia
    for k, (_, fine) in enumerate(corsie):
        nome = f"tot{k}"
        anim.append(keyframes(nome, [
            (0.0, "opacity:0"), (pc(fine), "opacity:0"),
            (pc(fine) + 1.2, "opacity:1"), (100.0, "opacity:1")]))
        corpo.append(f'<g class="tot" opacity="1" style="animation:{nome} '
                     f'var(--d) infinite"><text class="t-tot" '
                     f'x="{px(fine) - 7:.1f}" y="{Y_CORSIE[k] - 12}" '
                     f'text-anchor="end">{totali[k]}</text></g>')

    # la scala dei tempi
    y_ax = y_riga(2, RIGHE[2] - 1) + RH + 16
    corpo.append(f'<line class="ax" x1="{X0}" y1="{y_ax}" x2="{X1}" '
                 f'y2="{y_ax}"/>')
    for t in range(0, scala + 1, 5):
        corpo.append(f'<line class="ax" x1="{px(t):.1f}" y1="{y_ax}" '
                     f'x2="{px(t):.1f}" y2="{y_ax + 5}"/>')
        corpo.append(f'<text class="lbs" x="{px(t):.1f}" y="{y_ax + 20}" '
                     f'text-anchor="middle">{t}</text>')
    corpo.append(f'<text class="lbs" x="{X0 - 10}" y="{y_ax + 20}" '
                 f'text-anchor="end">secondi</text>')

    # il cursore: a riposo in fondo alla scala
    y_cur = Y_CORSIE[0] - 24
    corpo.append(f'<g id="cursore"><line class="cur" x1="{X1}" '
                 f'y1="{y_cur}" x2="{X1}" y2="{y_ax}"/><polygon class="punta" '
                 f'points="{X1 - 6},{y_cur - 9} {X1 + 6},{y_cur - 9} '
                 f'{X1},{y_cur}"/></g>')

    H = y_ax + 32
    return Figura(
        larghezza=W, altezza=H,
        alt=alt,
        corpo="".join(corpo),
        stile=f"""    .rit  {{ transform-box:view-box; transform-origin:{X0 - 2}px 0px; }}
    .nome {{ font-weight:700; }}
    .grig {{ stroke:{BORDER}; stroke-width:1; stroke-dasharray:2 4; }}
    .b-pensa {{ fill:{TEAL}; }}
    .b-legge {{ fill:{OCRA}; fill-opacity:0.45; stroke:{OCRA}; stroke-width:1.6; }}
    .b-prevede {{ fill:{TERRACOTTA}; }}
    .b-scartata {{ fill:none; stroke:{TERRACOTTA}; stroke-width:1.6;
                  stroke-dasharray:4 3; }}
    .t-legge {{ font-family:{SANS}; font-size:13px; font-weight:700; fill:{INK}; }}
    .t-risposta {{ font-family:{SANS}; font-size:12px; fill:{CREAM}; }}
    .t-scartata {{ font-family:{SANS}; font-size:12px; fill:{TERRACOTTA}; }}
    .t-tot {{ font-family:{SANS}; font-size:13.5px; font-weight:700; fill:{INK}; }}
    .fine {{ stroke:{INK}; stroke-width:1.6; stroke-dasharray:5 3; }}
    .cur  {{ stroke:{INK}; stroke-width:2; }}
    .punta {{ fill:{INK}; }}
    #cursore {{ animation:scorre var(--d) linear infinite; transform-box:view-box; }}""",
        animazioni=anim,
        durata=15.0,
        fermi="#cursore, .rit, .tot, .cambio",
    )
