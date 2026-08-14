"""Il cancello che respinge: quattro tentativi, tre motivi, una memoria che cresce.

La figura ferma del capitolo (`loop-engineering-ciclo.svg`) mostra lo **schema**
del ciclo; questa mostra una **esecuzione**, che è la cosa che uno schema non
può dire: tre cose si muovono insieme e a ritmi diversi. Il candidato si
riscrive da capo a ogni giro; il motivo del rifiuto invece non si riscrive, si
accumula (una riga in più a ogni giro, e nessuna se ne va); il cancello resta
chiuso tre volte e si apre alla quarta. Chi guarda solo lo schema vede un
cerchio che gira; qui si vede che il ciclo **impara dai propri fallimenti** e
che **finisce**.

I quattro stati non sono scritti a mano: questo file **esegue il blocco di
codice della sezione**, preso dal `.md`, e disegna la traccia che ne esce.
`controlla()` la confronta tre volte (la traccia ricostruita con le funzioni
del capitolo, ciò che il programma stampa davvero, e il blocco ```text della
sezione): se un giorno il codice del libro cambiasse, la figura non si
genererebbe più invece di smentirlo in silenzio.

Perfino le regole di evidenziazione escono dal capitolo. Quali caratteri si
accendono in terracotta lo dice il **motivo del rifiuto** (le maiuscole, gli
spazi, i caratteri oltre il tetto), e il tetto del righello è il numero che
`verifica` stampa nel suo messaggio: qui non c'è nessun 20 scritto a mano.

Lo stato di riposo è il quarto tentativo: il cancello aperto, la stringa che
passa, e **la colonna dei tre motivi ancora tutta lì**. È il punto della
sezione: il ciclo non ha ritentato tre volte alla cieca, ha riletto ogni volta
perché era stato respinto.
"""

import contextlib
import io
import math
import re
from pathlib import Path

from paithon_svg import *

NOME = "cancello-che-respinge"
TITOLO = "il cancello respinge tre volte, e la memoria del fallimento cresce"

QUI = Path(__file__).resolve().parent
SEZIONE = QUI.parents[1] / "book" / "IngegneriaLLM" / "loop-engineering.md"

# Il font monospace del brand (`--pt-font-mono` di tokens.css). Il motore
# espone SANS e SERIF perché finora bastavano; qui la stringa è codice, e
# senza passo fisso i caratteri non si allineano al righello.
MONO = "JetBrains Mono, ui-monospace, monospace"


# --------------------------------------------------------------------------
# Il codice del capitolo, eseguito davvero
# --------------------------------------------------------------------------
def _blocco(marcatore: str) -> str:
    testo = SEZIONE.read_text(encoding="utf-8")
    blocchi = re.findall(rf"```{marcatore}\n(.*?)```", testo, re.S)
    if len(blocchi) != 1:
        raise ValueError(f"{NOME}: {len(blocchi)} blocchi ```{marcatore} in "
                         f"{SEZIONE.name}, ne serve esattamente 1")
    return blocchi[0]


def esegui_il_capitolo():
    """Esegue il blocco della sezione e restituisce (spazio dei nomi, righe stampate)."""
    codice = _blocco("python")
    spazio = {"__name__": "loop_engineering"}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(codice, str(SEZIONE), "exec"), spazio)   # noqa: S102
    return spazio, buf.getvalue().splitlines(), codice


def traccia(spazio, codice):
    """Rifà il loop con le funzioni del capitolo, tenendo i passi invece di stamparli."""
    richiesta = re.search(r'loop\(\s*"([^"]*)"', codice)
    richiesta = richiesta.group(1) if richiesta else ""
    massimo = spazio["loop"].__defaults__[0]

    memoria, passi = [], []
    for i in range(1, massimo + 1):
        candidata = spazio["genera"](richiesta, memoria)
        ok, motivo = spazio["verifica"](candidata)
        passi.append({"i": i, "candidata": candidata, "ok": ok, "motivo": motivo})
        if ok:
            break
        memoria.append(motivo)
    return passi, massimo


def controlla(passi, massimo, spazio, stampate):
    """Tre confronti: la traccia, ciò che il programma stampa, il blocco della sezione."""
    righe = [f"tentativo {p['i']}: {p['candidata']!r} -> {p['motivo']}" for p in passi]
    righe.append(f"accettato: {passi[-1]['candidata']}")
    if righe != stampate:
        raise ValueError(f"{NOME}: la traccia ricostruita non è quella stampata "
                         f"dal capitolo:\n  {righe}\n  {stampate}")
    atteso = _blocco("text").splitlines()
    if righe != atteso:
        raise ValueError(f"{NOME}: il capitolo stampa righe diverse da quelle "
                         f"del suo blocco di uscita:\n  {righe}\n  {atteso}")
    if len(passi) != 4 or not passi[-1]["ok"] or any(p["ok"] for p in passi[:-1]):
        raise ValueError(f"{NOME}: la figura disegna 4 stati (3 rifiuti e "
                         f"un'accettazione), il capitolo ne fa {len(passi)}")
    if massimo != 5:
        raise ValueError(f"{NOME}: max_tentativi = {massimo}, la figura ne "
                         f"mostra il tetto e il capitolo ne dichiara 5")
    if spazio.get("risultato") != passi[-1]["candidata"]:
        raise ValueError(f"{NOME}: il capitolo accetta {spazio.get('risultato')!r}")


def colpevoli(candidata, motivo, tetto):
    """Quali caratteri si accendono: lo dice il motivo del rifiuto, non una lista."""
    if "minuscolo" in motivo:
        return [i for i, c in enumerate(candidata) if c != c.lower()]
    if "spazi" in motivo:
        return [i for i, c in enumerate(candidata) if c == " "]
    if "lungo" in motivo:
        if tetto is None:
            raise ValueError(f"{NOME}: il motivo «{motivo}» non dice il tetto")
        return list(range(tetto, len(candidata)))
    raise ValueError(f"{NOME}: motivo di rifiuto sconosciuto: «{motivo}»")


def leggi_tetto(passi):
    """Il tetto del righello è il numero che `verifica` stampa nel suo messaggio."""
    for p in passi:
        m = re.search(r"\((\d+)\s*>\s*(\d+)\s*caratteri\)", p["motivo"])
        if not m:
            continue
        lungo, tetto = int(m.group(1)), int(m.group(2))
        if lungo != len(p["candidata"]):
            raise ValueError(f"{NOME}: il motivo dice {lungo} caratteri, la "
                             f"candidata ne ha {len(p['candidata'])}")
        return tetto, p["i"] - 1
    raise ValueError(f"{NOME}: nessun rifiuto per lunghezza: niente righello")


# --------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------
W, H = 800, 376

X0, CH, FS = 40.0, 9.2, 14.5        # prima cella, passo per carattere, corpo
Y_BOX, H_BOX, Y_BASE = 72, 34, 94   # la scatola della candidata

Y_RIG, Y_RIG_LBL = 116, 134         # il righello e le sue etichette

Y_GATE = 202                        # la barriera
X_MURO_A, X_MURO_B = 34, 404
X_CARD_L, X_CARD_R = 158, 280       # i cardini delle due ante
ANG = 45.0                          # quanto restano alzate, aperte
X_FLUSSO = (X_CARD_L + X_CARD_R) / 2

Y_FRECCIA = 146                     # da dove scende la candidata
Y_OK, H_OK, W_OK = 226, 46, 190     # la casella «accettato»

X_MEM, W_MEM, H_MEM = 470, 300, 34  # la colonna della memoria
Y_MEM = [84, 126, 168]
X_CONN = 432                        # il raccordo cancello -> memoria

Y_CNT = 228                         # «N motivi in memoria»
Y_TENT = 306                        # il contagiri
Y_NOTE = [338, 356]

TEN, FADE = 0.62, 0.18              # tenuta e dissolvenza, in frazioni di stato


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def costruisci() -> Figura:
    spazio, stampate, codice = esegui_il_capitolo()
    passi, massimo = traccia(spazio, codice)
    controlla(passi, massimo, spazio, stampate)
    tetto, stato_righello = leggi_tetto(passi)

    rifiuti = [p for p in passi if not p["ok"]]
    if len(rifiuti) != len(Y_MEM):
        raise ValueError(f"{NOME}: {len(rifiuti)} motivi, la colonna ne tiene "
                         f"{len(Y_MEM)}")

    n = len(passi)
    passo = 100.0 / n
    corpo, anim, nomi = [], [], {}

    # ---------------------------------------------------------------- tempo
    def visibile(a: int, b: int) -> tuple[str, int]:
        """(@keyframes, opacità di riposo) per chi si vede dallo stato a al b.

        Il valore al 100% è quello dell'ultimo stato, ed è lo stesso che
        l'elemento porta come attributo: riposo e fine dell'animazione non
        possono divergere.
        """
        finale = 1 if b == n - 1 else 0
        if (a, b) not in nomi:
            nome = f"v{a}{b}"
            tappe = [(0.0, f"opacity:{1 if a == 0 else 0}")]
            if a > 0:
                tappe.append((a * passo - passo * (FADE + 0.12), "opacity:0"))
                tappe.append((a * passo, "opacity:1"))
            tappe.append((b * passo + passo * TEN, "opacity:1"))
            if finale:
                tappe.append((100.0, "opacity:1"))
            else:
                tappe.append((b * passo + passo * (TEN + FADE), "opacity:0"))
                tappe.append((100.0, "opacity:0"))
            anim.append(keyframes(nome, tappe))
            nomi[(a, b)] = nome
        return nomi[(a, b)], finale

    def gruppo(dentro: str, classe: str, a: int, b: int) -> str:
        nome, op = visibile(a, b)
        return (f'<g class="{classe}" opacity="{op}" '
                f'style="animation:{nome} var(--d) infinite">{dentro}</g>')

    # ----------------------------------------------------------------- defs
    corpo.append(
        '<defs>'
        f'<marker id="cr-t" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
        f'markerHeight="6" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{TERRACOTTA}"/></marker>'
        f'<marker id="cr-v" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
        f'markerHeight="6" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{TEAL}"/></marker>'
        '</defs>')

    # --------------------------------------------------------- intestazioni
    corpo += [
        f'<text class="ttl" x="{X_MURO_A}" y="40">il candidato</text>',
        f'<text class="lbs" x="{X_MURO_A}" y="58">si riscrive da capo a ogni giro</text>',
        f'<text class="ttl" x="{X_MEM}" y="40">la memoria del fallimento</text>',
        f'<text class="lbs" x="{X_MEM}" y="58">cresce di una riga a ogni rifiuto</text>',
    ]

    # ------------------------------------------------- la stringa candidata
    for k, p in enumerate(passi):
        s = p["candidata"]
        accesi = set(colpevoli(s, p["motivo"], tetto)) if not p["ok"] else set()
        larg = len(s) * CH
        pezzi = [f'<rect class="cas" x="{X0 - 8:.1f}" y="{Y_BOX}" '
                 f'width="{larg + 16:.1f}" height="{H_BOX}" rx="7"/>']
        # Le caselle accese contigue diventano una fascia sola: diciotto
        # rettangoli arrotondati appaiati fanno un bordo a merletto.
        for i in sorted(accesi):
            if i - 1 in accesi:
                continue
            j = i
            while j + 1 in accesi:
                j += 1
            pezzi.append(f'<rect class="hit" x="{X0 + i * CH:.1f}" y="{Y_BOX + 3}" '
                         f'width="{(j - i + 1) * CH:.1f}" height="{H_BOX - 6}" '
                         f'rx="3"/>')
        for i, ch in enumerate(s):
            x = X0 + i * CH
            glifo = ch if ch != " " else ("·" if i in accesi else "")
            if glifo:
                cls = "ch acc" if i in accesi else "ch"
                pezzi.append(f'<text class="{cls}" x="{x + CH / 2:.1f}" '
                             f'y="{Y_BASE}" text-anchor="middle">{esc(glifo)}</text>')
        corpo.append(gruppo("".join(pezzi), "st", k, k))

    # ------------------------------------------------------------ righello
    # Solo nello stato respinto per lunghezza: è lì che la misura è il motivo.
    s_lungo = passi[stato_righello]["candidata"]
    x_tetto = X0 + tetto * CH
    x_fine = X0 + len(s_lungo) * CH
    riga = [f'<line class="tac" x1="{x_tetto:.1f}" y1="{Y_BOX - 4}" '
            f'x2="{x_tetto:.1f}" y2="{Y_RIG + 6}"/>',
            f'<line class="rig-n" x1="{X0:.1f}" y1="{Y_RIG}" '
            f'x2="{x_tetto:.1f}" y2="{Y_RIG}"/>',
            f'<line class="rig-t" x1="{x_tetto:.1f}" y1="{Y_RIG}" '
            f'x2="{x_fine:.1f}" y2="{Y_RIG}"/>']
    for c in range(0, len(s_lungo) + 1, 5):
        x = X0 + c * CH
        cls = "rig-t" if c > tetto else "rig-n"
        riga.append(f'<line class="{cls}" x1="{x:.1f}" y1="{Y_RIG - 5}" '
                    f'x2="{x:.1f}" y2="{Y_RIG}"/>')
    riga.append(f'<text class="mist" x="{x_tetto:.1f}" y="{Y_RIG_LBL}" '
                f'text-anchor="middle">{tetto}</text>')
    riga.append(f'<text class="mist" x="{(x_tetto + x_fine) / 2:.1f}" '
                f'y="{Y_RIG_LBL}" text-anchor="middle">'
                f'{len(s_lungo) - tetto} di troppo</text>')
    corpo.append(gruppo("".join(riga), "rig", stato_righello, stato_righello))

    # -------------------------------------------------------- il cancello
    corpo += [
        f'<text class="ttl" x="{X_MURO_A}" y="224">il cancello</text>',
        f'<rect class="muro" x="{X_MURO_A}" y="{Y_GATE - 4}" '
        f'width="{X_CARD_L - X_MURO_A}" height="8" rx="4"/>',
        f'<rect class="muro" x="{X_CARD_R}" y="{Y_GATE - 4}" '
        f'width="{X_MURO_B - X_CARD_R}" height="8" rx="4"/>',
    ]

    # Le ante sono disegnate **aperte**, a coordinate vere e senza transform:
    # è lo stato di riposo, quello che vedono la stampa e chi non anima.
    # L'animazione le riporta all'orizzontale (chiuse) per i primi tre stati e
    # torna sull'identità al quarto: la rotazione finisce dove il file è scritto.
    # Aperte, le ante stanno **fuori** dal passaggio, ribaltate sopra il muro:
    # chiuse lo sbarrano da parte a parte. Fra le due posizioni corre la
    # rotazione, che è quella di un cancello vero.
    lung = X_FLUSSO - X_CARD_L
    dx, dy = lung * math.cos(math.radians(ANG)), lung * math.sin(math.radians(ANG))
    rot = 180.0 - ANG
    chiuso_a, chiuso_b = 0.0, (n - 2) * passo + passo * TEN
    aperto = (n - 1) * passo

    # Rotazione e colore sono due animazioni distinte sullo stesso elemento, e
    # non una sola con due proprietà: interpolate insieme, il colore passerebbe
    # per tutta la corsa dell'anta in un bruno che non è di nessuno dei due
    # stati. Così vira quando l'anta è ormai aperta.
    anim.append(keyframes("antac", [
        (chiuso_a, f"stroke:{TERRACOTTA}"),
        (chiuso_b + 0.75 * (aperto - chiuso_b), f"stroke:{TERRACOTTA}"),
        (aperto, f"stroke:{TEAL}"), (100.0, f"stroke:{TEAL}")]))

    for lato, cardine, verso in (("l", X_CARD_L, 1), ("r", X_CARD_R, -1)):
        anim.append(keyframes(f"anta{lato}", [
            (chiuso_a, f"transform:rotate({verso * rot:.0f}deg)"),
            (chiuso_b, f"transform:rotate({verso * rot:.0f}deg)"),
            (aperto, "transform:rotate(0deg)"),
            (100.0, "transform:rotate(0deg)")]))
        corpo.append(f'<line class="anta a-{lato}" x1="{cardine}" y1="{Y_GATE}" '
                     f'x2="{cardine - verso * dx:.1f}" y2="{Y_GATE - dy:.1f}" '
                     f'style="animation:anta{lato} var(--d) infinite,'
                     f'antac var(--d) infinite"/>')
        corpo.append(f'<circle class="card" cx="{cardine}" cy="{Y_GATE}" r="4"/>')

    # ----------------------------------------- il verdetto, e cosa ne segue
    for k, p in enumerate(passi):
        if p["ok"]:
            corpo.append(gruppo(
                f'<line class="fre-v" x1="{X_FLUSSO}" y1="{Y_FRECCIA}" '
                f'x2="{X_FLUSSO}" y2="{Y_OK - 6}" marker-end="url(#cr-v)"/>'
                f'<text class="ver-v" x="{X_MURO_B}" y="224" text-anchor="end">'
                f'aperto</text>'
                f'<rect class="okb" x="{X_FLUSSO - W_OK / 2:.1f}" y="{Y_OK}" '
                f'width="{W_OK}" height="{H_OK}" rx="9"/>'
                f'<text class="okl" x="{X_FLUSSO - W_OK / 2 + 16:.1f}" '
                f'y="{Y_OK + 19}">accettato</text>'
                f'<text class="ch acc-v" x="{X_FLUSSO - W_OK / 2 + 16:.1f}" '
                f'y="{Y_OK + 39}">{esc(p["candidata"])}</text>', "esito", k, k))
        else:
            y_riga = Y_MEM[p["i"] - 1] + H_MEM / 2
            corpo.append(gruppo(
                f'<line class="fre-t" x1="{X_FLUSSO}" y1="{Y_FRECCIA}" '
                f'x2="{X_FLUSSO}" y2="{Y_GATE - 9}" marker-end="url(#cr-t)"/>'
                f'<text class="ver-t" x="{X_MURO_B}" y="224" text-anchor="end">'
                f'chiuso</text>'
                f'<path class="conn" d="M {X_MURO_B} {Y_GATE - 10} '
                f'H {X_CONN} V {y_riga:.0f} H {X_MEM - 8}" '
                f'marker-end="url(#cr-t)"/>', "esito", k, k))

    # ------------------------------------------------- la colonna dei motivi
    for j, p in enumerate(rifiuti):
        y = Y_MEM[j]

        def riga_memoria(nuova: bool) -> str:
            cls = "mem nuova" if nuova else "mem"
            tx = "mtx nuova" if nuova else "mtx"
            return (f'<rect class="{cls}" x="{X_MEM}" y="{y}" width="{W_MEM}" '
                    f'height="{H_MEM}" rx="8"/>'
                    f'<text class="mnum" x="{X_MEM + 14}" y="{y + 22}">{j + 1}.</text>'
                    f'<text class="{tx}" x="{X_MEM + 34}" y="{y + 22}">'
                    f'{esc(p["motivo"])}</text>')

        corpo.append(gruppo(riga_memoria(True), "mem-g", j, j))
        if j + 1 <= n - 1:
            corpo.append(gruppo(riga_memoria(False), "mem-g", j + 1, n - 1))

    # quante righe ha in mano il generatore, stato per stato
    for k in range(n):
        quante = min(k + 1, len(rifiuti))
        testo = ("1 motivo in memoria" if quante == 1
                 else f"{quante} motivi in memoria")
        corpo.append(gruppo(f'<text class="cnt" x="{X_MEM}" y="{Y_CNT}">'
                            f'{testo}</text>', "cnt-g", k, k))

    # ------------------------------------------------------------ il contagiri
    for k, p in enumerate(passi):
        cls = "tent ok" if p["ok"] else "tent"
        corpo.append(gruppo(f'<text class="{cls}" x="{X_MURO_A}" y="{Y_TENT}">'
                            f'tentativo {p["i"]} di {massimo}</text>',
                            "tent-g", k, k))

    corpo += [
        f'<text class="note" x="{X_MURO_A}" y="{Y_NOTE[0]}">Il cancello ha due '
        f'stati soli, aperto o chiuso: il «quasi passato» non esiste, e non lo '
        f'decide chi ha scritto la stringa.</text>',
        f'<text class="note" x="{X_MURO_A}" y="{Y_NOTE[1]}">Nessun giro riparte '
        f'da zero: il generatore rilegge l\'ultimo motivo, e il tetto ferma anche '
        f'un ciclo che non converge.</text>',
    ]

    return Figura(
        larghezza=W, altezza=H,
        alt="Una esecuzione del ciclo genera, verifica e raffina. In alto a "
            "sinistra la stringa candidata, che si riscrive a ogni tentativo; "
            "sotto, una barriera con due ante incernierate, il cancello della "
            "verifica. Al primo tentativo «Guida Introduttiva a PyTorch» ha le "
            "maiuscole accese in terracotta e il cancello resta chiuso; al "
            "secondo «guida introduttiva a pytorch» si accendono gli spazi; al "
            "terzo «guida-introduttiva-a-pytorch-per-tutti» un righello sotto "
            "la stringa segna il tetto di venti caratteri e i diciotto di "
            "troppo restano oltre la tacca, in terracotta. Ogni rifiuto manda "
            "il suo motivo nella colonna di destra, che si allunga di una riga "
            "per volta e non ne perde nessuna: deve essere tutto minuscolo, "
            "niente spazi usa il trattino, troppo lungo. Al quarto tentativo "
            "«guida-pytorch» le due ante si aprono in teal e la stringa passa "
            "verso la casella «accettato», mentre i tre motivi restano tutti "
            "in vista. In basso il contagiri arriva a tentativo 4 di 5.",
        corpo="".join(corpo),
        stile=f"""    .cas  {{ fill:{CREAM}; stroke:{BORDER_STRONG}; stroke-width:1.8; }}
    .hit  {{ fill:{TERRACOTTA}; fill-opacity:0.16; }}
    .ch   {{ font-family:{MONO}; font-size:{FS}px; fill:{INK}; }}
    .acc  {{ fill:{TERRACOTTA}; font-weight:700; }}
    .acc-v {{ fill:{TEAL}; font-weight:700; }}
    .tac  {{ stroke:{TERRACOTTA}; stroke-width:1.6; stroke-dasharray:3 3; }}
    .rig-n {{ stroke:{BORDER_STRONG}; stroke-width:2; }}
    .rig-t {{ stroke:{TERRACOTTA}; stroke-width:2.6; }}
    .mist {{ font-family:{SANS}; font-size:11px; font-weight:700; fill:{TERRACOTTA}; }}
    .muro {{ fill:{BORDER_STRONG}; }}
    .anta {{ stroke:{TEAL}; stroke-width:7; stroke-linecap:round;
            transform-box:view-box; }}
    .a-l  {{ transform-origin:{X_CARD_L}px {Y_GATE}px; }}
    .a-r  {{ transform-origin:{X_CARD_R}px {Y_GATE}px; }}
    .card {{ fill:{CREAM}; stroke:{FG_MUTED}; stroke-width:2; }}
    .fre-t {{ stroke:{TERRACOTTA}; stroke-width:3; fill:none; }}
    .fre-v {{ stroke:{TEAL}; stroke-width:3; fill:none; }}
    .conn {{ stroke:{TERRACOTTA}; stroke-width:2; fill:none; stroke-opacity:0.75; }}
    .ver-t {{ font-family:{SANS}; font-size:14px; font-weight:700; fill:{TERRACOTTA}; }}
    .ver-v {{ font-family:{SANS}; font-size:14px; font-weight:700; fill:{TEAL}; }}
    .okb  {{ fill:{CREAM}; stroke:{TEAL}; stroke-width:2.5; }}
    .okl  {{ font-family:{SANS}; font-size:11px; font-weight:700; fill:{TEAL}; }}
    .mem  {{ fill:{CREAM}; stroke:{BORDER_STRONG}; stroke-width:1.6; }}
    rect.nuova {{ stroke:{TERRACOTTA}; stroke-width:2.6; fill-opacity:0.9; }}
    .mnum {{ font-family:{SANS}; font-size:12px; font-weight:700; fill:{FG_MUTED}; }}
    .mtx  {{ font-family:{SANS}; font-size:12.5px; fill:{INK}; }}
    text.nuova {{ fill:{TERRACOTTA}; font-weight:600; }}
    .cnt  {{ font-family:{SANS}; font-size:12.5px; font-weight:700; fill:{TERRACOTTA}; }}
    .tent {{ font-family:{SANS}; font-size:18px; font-weight:700; fill:{TERRACOTTA}; }}
    .tent.ok {{ fill:{TEAL}; }}
    .note {{ font-family:{SANS}; font-size:12.5px; fill:{FG_MUTED}; }}""",
        animazioni=anim,
        durata=n * 2.4,
        fermi=".st, .rig, .anta, .esito, .mem-g, .cnt-g, .tent-g",
    )
