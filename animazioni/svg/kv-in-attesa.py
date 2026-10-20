"""La KV cache di una richiesta che aspetta uno strumento: tre strade, tre sprechi.

La sezione «Mentre lo strumento lavora» di «Servire un agente» confronta le
tre cose che un motore può fare della KV cache di una richiesta ferma ad
aspettare uno strumento (conservarla, scartarla e ricalcolarla al ritorno,
spostarla nella memoria dell'host e riportarla), e le misura con un'unità
sola, la memoria della scheda ferma moltiplicata per il tempo in cui lo resta.
Il conto ha una forma che i tre totali non mostrano: conservare tiene ferma
poca memoria per tutta la pausa, scartare e spostare ne tengono ferma molta
per un attimo, perché in quell'attimo si fermano anche tutte le altre
richieste. Qui il tempo è il contenuto: lo spreco si accumula mentre la pausa
scorre, e in ciascuna corsia in un momento diverso.

I numeri non sono scritti a mano. Il file esegue il blocco di codice della
sezione (quello con `def sprechi(`), confronta ciò che stampa con il suo
blocco ```text, e prende da lì le costanti e gli sprechi della richiesta da
2.000 token con l'attenzione multi-testa piena, per la pausa di cinque
secondi. Gli `assert` difendono i numeri che didascalia e testo alternativo
promettono (1,05 GB di cache, 56 ms di ricalcolo, 41,9 ms per viaggio; 5,24,
1,74 e 2,60 GB·s), ricontano lo spreco dalle aree tratteggiate del disegno, e
pretendono che il testo alternativo della pagina sia identico a quello che il
file calcola.

Tre scelte di disegno, dichiarate perché la scena non prometta più di quanto
contenga:

- l'asse del tempo è a tratti. Generazione e pausa sono in scala; il
  ricalcolo e i viaggi fra scheda e host, che durano decine di millisecondi,
  cadono in due finestre allargate dello stesso fattore, segnate su tutte e
  tre le corsie, così un ricalcolo resta più lungo di un viaggio e più corto
  di due, come nel conto;
- le altezze non sono in scala: la cache della richiesta, 1,05 GB contro i 30
  delle altre, sarebbe una riga sottile, e le etichette dicono i numeri;
- la richiesta alla porta, entrata nel posto liberato, non ne esce prima di
  aver finito, quindi al ritorno dello strumento la richiesta interrotta
  aspetta che il posto si liberi: è l'attesa al rientro che Continuum
  aggiunge al conto. Mentre aspetta non occupa memoria della scheda, e lo
  spreco in GB·s non cambia.

Lo stato di riposo è il conto concluso: le tre corsie complete, i tratteggi e
i tre totali. L'animazione scopre le corsie da sinistra a destra con un
ritaglio, e un cursore le percorre a velocità costante (fuori dalle finestre
allargate, al passo del tempo vero); i contatori crescono soltanto dove il
cursore attraversa memoria tratteggiata, perché sono l'area tratteggiata
contata fin lì.
"""

import contextlib
import io
import math
import re
from pathlib import Path

from paithon_svg import *

NOME = "kv-in-attesa"
TITOLO = "la KV cache di una richiesta che aspetta uno strumento"

QUI = Path(__file__).resolve().parent
SEZIONE = QUI.parents[1] / "book" / "MLOps" / "servire-un-agente.md"
MARCA = "def sprechi("
FIGURA = f"../figures/{NOME}.svg"

PAUSA = 5              # secondi: la pausa che la figura segue
VARIANTE = "MHA"       # l'attenzione multi-testa piena, la riga del conto

PAROLE = ("zero", "un", "due", "tre", "quattro", "cinque", "sei", "sette",
          "otto", "nove", "dieci")


# --------------------------------------------------------------------------
# Il codice della sezione, eseguito davvero
# --------------------------------------------------------------------------
def leggi_la_sezione() -> tuple[str, str, str]:
    """Il testo della pagina, il blocco con la marca e il suo ```text."""
    testo = SEZIONE.read_text(encoding="utf-8")
    trovati = [m for m in re.finditer(r"^```python\n(.*?)^```", testo,
                                      re.S | re.M)
               if MARCA in m.group(1)]
    if len(trovati) != 1:
        raise ValueError(f"{NOME}: {len(trovati)} blocchi ```python con "
                         f"{MARCA!r} in {SEZIONE.name}, ne serve esattamente 1")
    blocco = trovati[0]
    uscita = re.match(r"\n*```text\n(.*?)^```", testo[blocco.end():],
                      re.S | re.M)
    if not uscita:
        raise ValueError(f"{NOME}: il blocco in {SEZIONE.name} non ha il suo "
                         f"```text subito dopo")
    return testo, blocco.group(1), uscita.group(1)


def esegui_il_capitolo() -> tuple[str, str, dict, list[str]]:
    testo, codice, atteso = leggi_la_sezione()
    spazio = {"__name__": "servire_un_agente"}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(codice, str(SEZIONE), "exec"), spazio)   # noqa: S102
    stampate = buf.getvalue().splitlines()
    if stampate != atteso.splitlines():
        raise ValueError(f"{NOME}: il capitolo stampa righe diverse dal suo "
                         f"blocco di uscita:\n  {stampate}\n  "
                         f"{atteso.splitlines()}")
    return testo, codice, spazio, stampate


def conto(codice: str, spazio: dict, stampate: list[str]) -> dict:
    """Le grandezze della figura, dalle costanti e dalla funzione del capitolo.

    `m_kv` si legge dalla riga del ciclo che la definisce: dopo l'esecuzione
    la variabile vale quella dell'ultima variante (la GQA), non la nostra.
    """
    trovati = re.findall(rf'\("{VARIANTE}",\s*(\d+)\s*\*\s*KIB\)', codice)
    if len(trovati) != 1:
        raise ValueError(f"{NOME}: la variante {VARIANTE} compare "
                         f"{len(trovati)} volte nel ciclo del capitolo")
    m_kv = int(trovati[0]) * spazio["KIB"]
    n = spazio["n"]
    propria = n * m_kv
    t_ric = 2 * spazio["N_P"] * n / spazio["P_EFF"]
    t_v = propria / spazio["B_HOST"]
    c, s, p = spazio["sprechi"](n, PAUSA, m_kv)

    # La riga d'intestazione della variante e la riga della pausa, ricostruite
    # con i nostri numeri e cercate in ciò che il capitolo ha stampato: se la
    # figura e il blocco dicessero due cose diverse, ci si ferma qui.
    testa = (f"{VARIANTE}: {propria / 1e9:.2f} GB di cache, ricalcolo "
             f"{t_ric * 1000:.0f} ms, viaggio {t_v * 1000:.1f} ms")
    scelta = min([(c, "conserva"), (s, "scarta"), (p, "sposta")])[1]
    riga = (f"  pausa {PAUSA:>5} s  conserva {c:6.3f}  scarta {s:4.2f}  "
            f"sposta {p:4.2f} GB·s: {scelta}")
    assert testa in stampate, f"{NOME}: il capitolo non stampa {testa!r}"
    i = stampate.index(testa)
    seguito = []
    for r in stampate[i + 1:]:
        if not r.startswith("  "):
            break
        seguito.append(r)
    assert riga in seguito, \
        f"{NOME}: nella variante {VARIANTE} manca la riga {riga!r}"

    return dict(n=n, m_kv=m_kv, propria=propria / 1e9,
                altri=spazio["ALTRI"] / 1e9, t_ric=t_ric, t_v=t_v,
                conserva=c, scarta=s, sposta=p)


def it(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def virgola(x: float, cifre: int = 2) -> str:
    return f"{x:.{cifre}f}".replace(".", ",")


def verifica_numeri(k: dict) -> None:
    """Quello che didascalia e testo alternativo promettono, e il testo dice."""
    assert f"{k['propria']:.2f}" == "1.05", \
        f"la cache pesa {k['propria']:.4f} GB, la pagina dice 1,05"
    assert round(k["t_ric"] * 1000) == 56, \
        f"il ricalcolo dura {k['t_ric'] * 1000:.2f} ms, la pagina dice 56"
    assert f"{k['t_v'] * 1000:.1f}" == "41.9", \
        f"un viaggio dura {k['t_v'] * 1000:.3f} ms, la pagina dice 41,9"
    for nome, atteso in (("conserva", "5.24"), ("scarta", "1.74"),
                         ("sposta", "2.60")):
        assert f"{k[nome]:.2f}" == atteso, \
            f"{nome}: {k[nome]:.4f} GB·s, la pagina dice {atteso}"
    # «durano decine di millisecondi», dice la didascalia
    for nome in ("t_ric", "t_v"):
        assert 0.01 <= k[nome] < 0.1, f"{nome} non dura decine di ms"
    # Il testo: due viaggi costano più di un ricalcolo, e a cinque secondi
    # vince lo scarto, con la conservazione che spreca più di tutte.
    assert k["t_v"] < k["t_ric"] < 2 * k["t_v"], \
        "il ricalcolo non sta fra un viaggio e due"
    assert k["scarta"] < k["sposta"] < k["conserva"], \
        "a cinque secondi l'ordine degli sprechi non è quello del testo"


# --------------------------------------------------------------------------
# Il tempo, a tratti: generazione e pausa in scala, due finestre allargate
# --------------------------------------------------------------------------
S = 80.0              # px per secondo, nei tratti in scala
F = 20                # quanto si allargano le due finestre
T_GEN = 0.8           # secondi di generazione prima della chiamata
ATTESA = 1.0          # secondi fra la risposta e il posto che si libera
T_FINE = 0.5          # secondi di generazione dopo il rientro


def tratti(k: dict) -> list[tuple[float, float, float]]:
    """(da, a, fattore) in secondi, con t = 0 alla chiamata dello strumento."""
    t_b = PAUSA + ATTESA
    return [(-T_GEN, 0.0, 1),
            (0.0, k["t_v"], F),                    # l'uscita della cache
            (k["t_v"], PAUSA, 1),
            (PAUSA, t_b, 1),                       # la risposta, e l'attesa
            (t_b, t_b + k["t_ric"], F),            # il rientro
            (t_b + k["t_ric"], t_b + k["t_ric"] + T_FINE, 1)]


# --------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------
X0 = 128                         # l'inizio dell'asse del tempo
Y_TIT, Y_BRACE = 26, 54
Y_CORSIE = (78, 220, 362)        # la cima di ciascuna corsia
# righe dentro una corsia, rispetto alla sua cima
R_NOME = 14
R_PORTA, H_PORTA = 22, 16
R_SCHEDA, H_SCHEDA = 44, 50
R_POSTO, H_POSTO = 45, 16
R_ALTRI, H_ALTRI = 63, 30
R_HOST, H_HOST = 100, 16
R_NOTA = 132
R_BANDA, H_BANDA = 18, 120       # bande, linee e cursore, dentro la corsia
RIVELA = 0.85                    # la frazione del ciclo in cui si scopre
PASSO_CONTA = 12                 # px fra due letture del contatore
RAMPA = 0.05                     # punti di timeline per cambiare contatore

GB = {"posto": "propria", "altri": "altri"}


def largo(testo: str, corpo: float = 12) -> float:
    """Larghezza stimata di un'etichetta in Inter (basta per non sbordare)."""
    return len(testo) * corpo * 0.56


def corsie(k: dict) -> list[dict]:
    """Chi occupa quale riga, quando: la stessa pausa con le tre strade.

    Ogni blocco è (riga, da, a, tipo, etichetta). I tipi tratteggiati sono
    la memoria della scheda ferma, e sono gli unici che contano nello spreco.
    """
    t_v, t_ric = k["t_v"], k["t_ric"]
    t_b = PAUSA + ATTESA
    fine = t_b + t_ric + T_FINE
    gb = f"{virgola(k['propria'])} GB"
    altre = f"altre richieste: {k['altri']:.0f} GB"
    return [
        dict(nome="conserva", sotto="la cache resta al suo posto",
             strada="conserva", blocchi=[
                 ("porta", -T_GEN, fine, "Bp", "B aspetta un posto"),
                 ("posto", -T_GEN, 0.0, "A", "A"),
                 ("posto", 0.0, PAUSA, "Ah", f"A ferma: {gb}"),
                 ("posto", PAUSA, fine, "A", "A riprende"),
                 ("altri", -T_GEN, fine, "T", altre)],
             note=[], frecce=[]),
        dict(nome="scarta", sotto="la cache si butta, al ritorno si ricalcola",
             strada="scarta", blocchi=[
                 ("porta", -T_GEN, 0.0, "Bp", "B aspetta"),
                 ("porta", PAUSA, t_b, "Ap", "A aspetta"),
                 ("posto", -T_GEN, 0.0, "A", "A"),
                 ("posto", 0.0, t_b, "B", "B, nel posto liberato"),
                 ("posto", t_b, t_b + t_ric, "Ah", ""),
                 ("posto", t_b + t_ric, fine, "A", "A"),
                 ("altri", -T_GEN, t_b, "T", altre),
                 ("altri", t_b, t_b + t_ric, "Th", ""),
                 ("altri", t_b + t_ric, fine, "T", "")],
             note=[(t_b + t_ric / 2, f"ricalcolo {t_ric * 1000:.0f} ms")],
             frecce=[]),
        dict(nome="sposta", sotto="la cache va nell'host e torna",
             strada="sposta", blocchi=[
                 ("porta", -T_GEN, t_v, "Bp", "B aspetta un posto"),
                 ("porta", PAUSA, t_b, "Ap", "A aspetta"),
                 ("posto", -T_GEN, 0.0, "A", "A"),
                 ("posto", 0.0, t_v, "Ah", ""),
                 ("posto", t_v, t_b, "B", "B, nel posto liberato"),
                 ("posto", t_b, t_b + t_v, "Ah", ""),
                 ("posto", t_b + t_v, fine, "A", "A"),
                 ("altri", -T_GEN, 0.0, "T", ""),
                 ("altri", 0.0, t_v, "Th", ""),
                 ("altri", t_v, t_b, "T", altre),
                 ("altri", t_b, t_b + t_v, "Th", ""),
                 ("altri", t_b + t_v, fine, "T", ""),
                 ("host", 0.0, t_b + t_v, "H", f"A nell'host: {gb}")],
             note=[(t_v / 2, f"viaggio {virgola(t_v * 1000, 1)} ms"),
                   (t_b + t_v / 2, f"viaggio {virgola(t_v * 1000, 1)} ms")],
             frecce=[(t_v / 2, "giù"), (t_b + t_v / 2, "su")]),
    ]


def spreco(corsia: dict, k: dict, t: float) -> float:
    """GB·s di memoria della scheda tratteggiata fino all'istante t."""
    tot = 0.0
    for riga, da, a, tipo, _ in corsia["blocchi"]:
        if tipo in ("Ah", "Th"):
            tot += k[GB[riga]] * min(max(t - da, 0.0), a - da)
    return tot


def verifica_corsie(cc: list[dict], k: dict) -> None:
    fine = max(a for c in cc for _, _, a, _, _ in c["blocchi"])
    for c in cc:
        # lo spreco disegnato è quello del conto, non uno che gli somiglia
        disegnato = spreco(c, k, fine)
        assert math.isclose(disegnato, k[c["strada"]], rel_tol=1e-9), \
            (f"{c['nome']}: il tratteggio vale {disegnato:.4f} GB·s, "
             f"il conto {k[c['strada']]:.4f}")
        # nessun posto della scheda è mai occupato da due richieste insieme,
        # e A rientra solo dopo che B ha lasciato il posto
        posto = sorted((da, a) for r, da, a, _, _ in c["blocchi"]
                       if r == "posto")
        for (_, a1), (d2, _) in zip(posto, posto[1:]):
            assert a1 <= d2 + 1e-12, f"{c['nome']}: due richieste nel posto"
        assert posto[0][0] == -T_GEN and posto[-1][1] == fine, \
            f"{c['nome']}: il posto di A resta vuoto da qualche parte"
    conserva = cc[0]["blocchi"]
    assert any(t == "Bp" and a == fine for _, _, a, t, _ in conserva), \
        "nella prima corsia la richiesta alla porta entra, e non deve"


def costruisci() -> Figura:
    testo, codice, spazio, stampate = esegui_il_capitolo()
    k = conto(codice, spazio, stampate)
    verifica_numeri(k)
    cc = corsie(k)
    verifica_corsie(cc, k)

    tt = tratti(k)
    assert all(a1 == d2 for (_, a1, _), (d2, _, _) in zip(tt, tt[1:])), \
        "i tratti dell'asse non sono contigui"

    def x(t: float) -> float:
        px = X0
        for da, a, f in tt:
            px += S * f * (min(max(t, da), a) - da)
        return px

    def t_di(px: float) -> float:
        resto = px - X0
        for da, a, f in tt:
            w = S * f * (a - da)
            if resto <= w:
                return da + resto / (S * f)
            resto -= w
        return tt[-1][1]

    x1 = x(tt[-1][1])
    W = math.ceil(x1 + 22)
    t_b = PAUSA + ATTESA
    finestre = [(0.0, k["t_v"]), (t_b, t_b + k["t_ric"])]
    y_fondo = Y_CORSIE[-1] + R_NOTA + 10
    H = y_fondo + 84

    corpo, anim = [], []

    def p_anim(px: float) -> float:
        return RIVELA * 100 * (px - X0) / (x1 - X0)

    # ---- il tratteggio: diagonali, nessun transform ----------------------
    def tratteggio(nome: str, colore: str) -> str:
        return (f'<pattern id="{nome}" patternUnits="userSpaceOnUse" '
                f'width="6" height="6"><path d="M-1 1 L1 -1 M0 6 L6 0 '
                f'M5 7 L7 5" stroke="{colore}" stroke-width="1.4"/>'
                f'</pattern>')
    corpo.append(f'<defs>{tratteggio("tr-a", TERRACOTTA)}'
                 f'{tratteggio("tr-t", TEAL)}'
                 f'<clipPath id="scopri"><rect class="rit" x="{X0 - 1}" '
                 f'y="{Y_CORSIE[0]}" width="{x1 - X0 + 2:.1f}" '
                 f'height="{y_fondo - Y_CORSIE[0]}" '
                 f'style="animation:scopri var(--d) linear infinite"/>'
                 f'</clipPath></defs>')

    # ---- intestazione -----------------------------------------------------
    corpo.append(f'<text class="ttl" x="16" y="{Y_TIT}">la memoria della '
                 f'scheda mentre una richiesta aspetta uno strumento</text>')
    xa, xb = x(0.0), x(PAUSA)
    corpo.append(f'<path class="graffa" d="M {xa:.1f} {Y_BRACE + 6} L {xa:.1f} '
                 f'{Y_BRACE} L {xb:.1f} {Y_BRACE} L {xb:.1f} {Y_BRACE + 6}"/>')
    corpo.append(f'<text class="lbl" x="{(xa + xb) / 2:.1f}" y="{Y_BRACE - 6}" '
                 f'text-anchor="middle">strumento: {PAUSA} s</text>')
    corpo.append(f'<text class="lbs" x="{(X0 + xa) / 2:.1f}" '
                 f'y="{Y_BRACE - 6}" text-anchor="middle">A genera</text>')

    # ---- le due finestre allargate, e la chiamata e la risposta -----------
    for da, a in finestre:
        for y0 in Y_CORSIE:
            corpo.append(f'<rect class="banda" x="{x(da):.1f}" '
                         f'y="{y0 + R_BANDA}" width="{x(a) - x(da):.1f}" '
                         f'height="{H_BANDA}"/>')
        xm = (x(da) + x(a)) / 2
        corpo.append(f'<text class="lbs" x="{xm:.1f}" y="{y_fondo + 16}" '
                     f'text-anchor="middle">allargato</text>')
        corpo.append(f'<text class="lbs" x="{xm:.1f}" y="{y_fondo + 31}" '
                     f'text-anchor="middle">{F} volte</text>')
    for t in (0.0, PAUSA):
        for y0 in Y_CORSIE:
            corpo.append(f'<line class="soglia" x1="{x(t):.1f}" '
                         f'y1="{y0 + R_BANDA}" x2="{x(t):.1f}" '
                         f'y2="{y0 + R_BANDA + H_BANDA}"/>')

    # ---- le cornici delle corsie: si vedono anche prima del cursore -------
    for i, c in enumerate(cc):
        y0 = Y_CORSIE[i]
        corpo.append(f'<text class="lbl nome" x="16" y="{y0 + R_NOME}">'
                     f'{c["nome"]}<tspan class="sotto" dx="10">{c["sotto"]}'
                     f'</tspan></text>')
        corpo.append(f'<rect class="cornice" x="{X0 - 1}" '
                     f'y="{y0 + R_SCHEDA}" width="{x1 - X0 + 2:.1f}" '
                     f'height="{H_SCHEDA}" rx="3"/>')
        corpo.append(f'<rect class="cornice" x="{X0 - 1}" y="{y0 + R_HOST}" '
                     f'width="{x1 - X0 + 2:.1f}" height="{H_HOST}" rx="3"/>')
        for testo_g, yy in (("alla porta", y0 + R_PORTA + 12),
                            ("scheda", y0 + R_SCHEDA + H_SCHEDA / 2 + 4),
                            ("host", y0 + R_HOST + 12)):
            corpo.append(f'<text class="lbs" x="{X0 - 10}" y="{yy:.0f}" '
                         f'text-anchor="end">{testo_g}</text>')

    # ---- il contenuto delle corsie, scoperto dal ritaglio ------------------
    corpo.append('<g clip-path="url(#scopri)">')
    righe = {"porta": (R_PORTA, H_PORTA), "posto": (R_POSTO, H_POSTO),
             "altri": (R_ALTRI, H_ALTRI), "host": (R_HOST + 1, H_HOST - 2)}
    for i, c in enumerate(cc):
        y0 = Y_CORSIE[i]
        for riga, da, a, tipo, etichetta in c["blocchi"]:
            ry, rh = righe[riga]
            xa, xb = x(da), x(a)
            y, w = y0 + ry, xb - xa
            if tipo in ("Ah", "Th"):
                tinta = TERRACOTTA if tipo == "Ah" else TEAL
                nome_tr = "tr-a" if tipo == "Ah" else "tr-t"
                corpo.append(f'<rect x="{xa:.1f}" y="{y}" width="{w:.1f}" '
                             f'height="{rh}" fill="url(#{nome_tr})" '
                             f'stroke="{tinta}" stroke-width="1.2"/>')
            elif tipo in ("Ap", "Bp", "H"):
                classe = {"Ap": "ap", "Bp": "bp", "H": "hh"}[tipo]
                corpo.append(f'<rect class="{classe}" x="{xa + 1:.1f}" '
                             f'y="{y + 1}" width="{w - 2:.1f}" '
                             f'height="{rh - 2}" rx="3"/>')
            else:
                classe = {"A": "a", "B": "b", "T": "t"}[tipo]
                corpo.append(f'<rect class="{classe}" x="{xa:.1f}" y="{y}" '
                             f'width="{w:.1f}" height="{rh}"/>')
            if not etichetta:
                continue
            yt = y + rh / 2 + 4.3
            if tipo in ("Ap", "Bp"):
                # le attese si leggono da sinistra, e possono sbordare nella
                # riga della porta, che dopo di loro è vuota
                corpo.append(f'<text class="blk-i" x="{xa + 6:.1f}" '
                             f'y="{yt:.1f}">{etichetta}</text>')
                continue
            if largo(etichetta) + 8 > w:
                etichetta = etichetta.split(",")[0].split(":")[0]
                if largo(etichetta) + 8 > w:
                    continue
            xm = (xa + xb) / 2
            if tipo in ("Ah", "B"):
                # sul tratteggio un'etichetta si legge solo su un fondo pieno,
                # e sull'ocra il testo non regge in nessuno dei due temi
                lw = largo(etichetta) + 12
                corpo.append(f'<rect class="chip" x="{xm - lw / 2:.1f}" '
                             f'y="{y + 1.5}" width="{lw:.1f}" '
                             f'height="{rh - 3}" rx="2"/>')
                classe = "blk-i"
            else:
                classe = {"A": "blk-c", "T": "blk-c", "B": "blk-i",
                          "H": "blk-a"}[tipo]
            corpo.append(f'<text class="{classe}" x="{xm:.1f}" y="{yt:.1f}" '
                         f'text-anchor="middle">{etichetta}</text>')
        for t, nota in c["note"]:
            corpo.append(f'<text class="nota" x="{x(t):.1f}" '
                         f'y="{y0 + R_NOTA}" text-anchor="middle">{nota}</text>')
        for t, verso in c["frecce"]:
            xf = x(t)
            alto, basso = y0 + R_POSTO + H_POSTO + 2, y0 + R_HOST - 1
            punta, coda = (basso, alto) if verso == "giù" else (alto, basso)
            dy = 6 if verso == "giù" else -6
            d = (f"M {xf:.1f} {coda} L {xf:.1f} {punta} "
                 f"M {xf - 4:.1f} {punta - dy} L {xf:.1f} {punta} "
                 f"L {xf + 4:.1f} {punta - dy}")
            corpo.append(f'<path class="alone" d="{d}"/>'
                         f'<path class="freccia" d="{d}"/>')
    corpo.append('</g>')

    # ---- i contatori: l'area tratteggiata contata fino al cursore ---------
    contatore = [0]
    for i, c in enumerate(cc):
        y0 = Y_CORSIE[i]
        letture, px = [], X0
        while True:
            valore = virgola(spreco(c, k, t_di(min(px, x1))))
            if not letture or letture[-1][1] != valore:
                letture.append((min(px, x1), valore))
            if px >= x1:
                break
            px += PASSO_CONTA
        assert letture[-1][1] == virgola(k[c["strada"]]), \
            f"{c['nome']}: il contatore finisce su {letture[-1][1]}"
        assert letture[0][1] == virgola(0.0)
        for j, (px, valore) in enumerate(letture):
            ultima = j == len(letture) - 1
            a = p_anim(px) if j else 0.0
            b = 100.0 if ultima else p_anim(letture[j + 1][0])
            tappe = [(0.0, 1)] if a == 0 else [(0.0, 0), (a, 0),
                                               (a + RAMPA, 1)]
            tappe += [(100.0, 1)] if ultima else [(b, 1), (b + RAMPA, 0),
                                                  (100.0, 0)]
            assert all(p2 - p1 >= 0.01 for (p1, _), (p2, _)
                       in zip(tappe, tappe[1:])), "tappe del contatore fuse"
            nome = f"c{contatore[0]}"
            contatore[0] += 1
            anim.append(keyframes(nome, [(p, f"opacity:{v}")
                                         for p, v in tappe]))
            corpo.append(f'<text class="cnt" x="{x1:.1f}" y="{y0 + R_NOME}" '
                         f'text-anchor="end" opacity="{1 if ultima else 0}" '
                         f'style="animation:{nome} var(--d) linear infinite">'
                         f'spreco: {valore} GB·s</text>')

    # ---- il cursore: a riposo non c'è, perché il conto è finito -----------
    corpo.append(f'<g class="cur" opacity="0" '
                 f'style="animation:cursore var(--d) linear infinite">'
                 + "".join(f'<line x1="{x1:.1f}" y1="{y0 + R_BANDA}" '
                           f'x2="{x1:.1f}" y2="{y0 + R_BANDA + H_BANDA}"/>'
                           for y0 in Y_CORSIE)
                 + '</g>')
    fine_rivela = RIVELA * 100
    anim.append(keyframes("cursore", [
        (0.0, f"transform:translateX({X0 - x1:.1f}px);opacity:1"),
        (fine_rivela, "transform:translateX(0px);opacity:1"),
        (fine_rivela + 3, "transform:translateX(0px);opacity:0"),
        (100.0, "transform:translateX(0px);opacity:0")]))
    anim.append(keyframes("scopri", [
        (0.0, "transform:scaleX(0)"),
        (fine_rivela, "transform:scaleX(1)"),
        (100.0, "transform:scaleX(1)")]))

    # ---- legenda ------------------------------------------------------------
    yl = y_fondo + 62
    xl = 16
    for tipo, testo_l in (("T", "memoria che lavora"),
                          ("Th", "memoria della scheda ferma: lo spreco"),
                          ("Bp", "in attesa"),
                          (None, "altezze non in scala")):
        if tipo == "T":
            corpo.append(f'<rect class="t" x="{xl}" y="{yl - 11}" width="22" '
                         f'height="13"/>')
        elif tipo == "Th":
            corpo.append(f'<rect x="{xl}" y="{yl - 11}" width="22" height="13" '
                         f'fill="url(#tr-t)" stroke="{TEAL}" '
                         f'stroke-width="1.2"/>')
        elif tipo == "Bp":
            corpo.append(f'<rect class="ap" x="{xl + 1}" y="{yl - 10}" '
                         f'width="20" height="11" rx="3"/>')
        dx = 30 if tipo else 0
        corpo.append(f'<text class="lbs" x="{xl + dx}" y="{yl}">{testo_l}'
                     f'</text>')
        xl += dx + largo(testo_l, 13) + 26

    alt = testo_alternativo(k)
    verifica_pagina(testo, alt, k)

    durata = (x1 - X0) / S / RIVELA
    return Figura(
        larghezza=W, altezza=H, alt=alt,
        corpo="".join(corpo),
        stile=f"""    .rit  {{ transform-box:view-box; transform-origin:{X0 - 1}px 0px; }}
    .cur  {{ stroke:{INK}; stroke-width:1.6; transform-box:view-box; }}
    .banda {{ fill:{BORDER}; }}
    .soglia {{ stroke:{BORDER_STRONG}; stroke-width:1.2; stroke-dasharray:3 3; }}
    .graffa {{ fill:none; stroke:{INK}; stroke-width:1.3; }}
    .cornice {{ fill:none; stroke:{BORDER_STRONG}; stroke-width:1.4; }}
    .nome {{ font-weight:700; }}
    .sotto {{ font-weight:400; font-size:13px; fill:{FG_MUTED}; }}
    .a    {{ fill:{TERRACOTTA}; }}
    .b    {{ fill:{OCRA}; }}
    .t    {{ fill:{TEAL}; }}
    .ap   {{ fill:none; stroke:{TERRACOTTA}; stroke-width:1.5; }}
    .bp   {{ fill:none; stroke:{OCRA}; stroke-width:2; }}
    .hh   {{ fill:none; stroke:{TERRACOTTA}; stroke-width:1.5; }}
    .chip {{ fill:{CREAM}; }}
    .blk-c {{ font-family:{SANS}; font-size:12px; fill:{CREAM}; }}
    .blk-i {{ font-family:{SANS}; font-size:12px; fill:{INK}; }}
    .blk-a {{ font-family:{SANS}; font-size:12px; fill:{TERRACOTTA}; }}
    .nota {{ font-family:{SANS}; font-size:12px; fill:{FG_MUTED}; }}
    .cnt  {{ font-family:{SANS}; font-size:14px; font-weight:700; fill:{INK}; }}
    .freccia {{ fill:none; stroke:{INK}; stroke-width:1.6; stroke-linecap:round; stroke-linejoin:round; }}
    .alone {{ fill:none; stroke:{CREAM}; stroke-width:4.5; stroke-linecap:round; stroke-linejoin:round; }}""",
        animazioni=anim,
        durata=durata,
        fermi=".rit, .cur, .cnt",
    )


def testo_alternativo(k: dict) -> str:
    return (
        f"Tre corsie, una per strada, per una richiesta da {it(k['n'])} "
        f"token con l'attenzione multi-testa piena, con il tempo in "
        f"orizzontale e la memoria della scheda in verticale; sotto "
        f"ciascuna, una striscia più bassa per la memoria dell'host. In ogni "
        f"corsia una richiesta genera, si ferma per {PAROLE[PAUSA]} secondi "
        f"ad aspettare uno strumento e riprende, mentre altre richieste "
        f"occupano il resto della memoria e una aspetta alla porta. Nella "
        f"prima corsia la cache della richiesta ferma resta al suo posto per "
        f"tutta la pausa, tratteggiata, e la richiesta alla porta non entra: "
        f"{virgola(k['conserva'])} GB·s sprecati. Nella seconda la cache se "
        f"ne va all'inizio della pausa e la richiesta alla porta entra nel "
        f"posto liberato; al ritorno dello strumento la richiesta aspetta "
        f"che quel posto si liberi, poi il ricalcolo ferma tutte le richieste "
        f"del mazzo, tratteggiate per la sua durata: "
        f"{virgola(k['scarta'])} GB·s. Nella terza la cache scende nella "
        f"striscia dell'host, la richiesta alla porta entra allo stesso modo, "
        f"e la cache risale quando il posto si libera; ogni viaggio ferma il "
        f"mazzo: {virgola(k['sposta'])} GB·s. Le durate del ricalcolo e dei "
        f"viaggi sono allargate per renderle visibili.")


def verifica_pagina(testo: str, alt: str, k: dict) -> None:
    """Il `:alt:` della pagina è questo, e la didascalia dice questi numeri."""
    blocchi = re.findall(r"```\{figure\} " + re.escape(FIGURA)
                         + r"\n(.*?)\n```", testo, re.S)
    assert len(blocchi) == 1, \
        f"{NOME}: {len(blocchi)} figure {FIGURA} in {SEZIONE.name}"
    alt_pagina = re.findall(r'^:alt: "(.*)"$', blocchi[0], re.M)
    assert alt_pagina == [alt], \
        (f"{NOME}: il :alt: di {SEZIONE.name} non è quello che il generatore "
         f"calcola; la versione buona è questa:\n{alt}")
    didascalia = " ".join(blocchi[0].split("\n\n", 1)[1].split())
    for atteso in (f"{it(k['n'])} token", f"per {PAROLE[PAUSA]} secondi",
                   "decine di millisecondi", "allargati"):
        assert atteso in didascalia, \
            f"{NOME}: la didascalia non dice più {atteso!r}"
