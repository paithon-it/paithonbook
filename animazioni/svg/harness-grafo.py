"""Un lavoro come grafo: i passi, il guasto, la ripresa, la firma.

La sezione sul grafo esegue un motore minimo: quattro nodi (pianifica,
costruisci, collauda, rilascia), un ciclo fra collaudo e costruzione, un
guasto al primo collaudo, la ripresa dall'ultimo punto salvato e una firma
prima del rilascio. Il programma stampa le tre corse; la figura fa vedere il
filo che si muove sul grafo, ed è il tempo il contenuto: dove si spegne, da
dove riparte, dove aspetta.

I numeri e i passi non sono scritti a mano: questo file **esegue il blocco di
codice della sezione**, preso dal `.md`, e ricava il percorso da ciò che il
programma stampa. `controlla()` confronta la stampa con il blocco ```text della
sezione, e difende quello che la didascalia promette.

Lo stato di riposo è il lavoro concluso: il gettone sul rilascio, il guasto
segnato sul collaudo, la firma data e tutti i passi salvati in fila. Il
gettone si sposta con andamento lineare fra una sosta e l'altra.
"""

import contextlib
import io
import re
from pathlib import Path

from paithon_svg import *

NOME = "harness-grafo"
TITOLO = "un lavoro come grafo: i passi, il guasto, la ripresa, la firma"

QUI = Path(__file__).resolve().parent
SEZIONE = QUI.parents[1] / "book" / "Harness" / "dal-ciclo-al-grafo.md"
MARCA = "def esegui("


# --------------------------------------------------------------------------
# Il codice del capitolo, eseguito davvero
# --------------------------------------------------------------------------
def _blocco(marcatore: str) -> str:
    """Il blocco con la marca (``python``) o la sua uscita (``text``)."""
    testo = SEZIONE.read_text(encoding="utf-8")
    trovati = [m for m in re.finditer(r"```python\n(.*?)```", testo, re.S)
               if MARCA in m.group(1)]
    if len(trovati) != 1:
        raise ValueError(f"{NOME}: {len(trovati)} blocchi ```python con "
                         f"{MARCA!r} in {SEZIONE.name}, ne serve esattamente 1")
    codice = trovati[0]
    if marcatore == "python":
        return codice.group(1)
    uscita = re.match(r"\n*```text\n(.*?)```", testo[codice.end():], re.S)
    if not uscita:
        raise ValueError(f"{NOME}: il blocco in {SEZIONE.name} non ha il suo "
                         f"```text subito dopo")
    return uscita.group(1)


def esegui_il_capitolo():
    codice = _blocco("python")
    spazio = {"__name__": "dal_ciclo_al_grafo"}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(codice, str(SEZIONE), "exec"), spazio)   # noqa: S102
    return spazio, buf.getvalue().splitlines()


def percorso(stampate):
    """I passi nell'ordine, ricavati dalle tre corse stampate.

    Ogni passo è (nodo, tipo): «fatto» per un nodo eseguito e salvato,
    «guasto» per quello interrotto, «attesa» per la firma.
    """
    passi = []
    for riga in stampate:
        corsa = re.match(r"(prima corsa|ripresa|dopo l'approvazione): (.*)$",
                         riga)
        if corsa:
            for voce in corsa.group(2).split(" -> "):
                nodo = voce.split(" (")[0]
                esito = re.search(r"\((\w+)\)", voce)
                passi.append((nodo, esito.group(1) if esito else "fatto"))
            continue
        guasto = re.match(r"\s+guasto in «(\w+)»", riga)
        if guasto:
            passi.append((guasto.group(1), "guasto"))
        attesa = re.match(r"\s+fermo prima di «(\w+)»", riga)
        if attesa:
            passi.append((attesa.group(1), "attesa"))
    return passi


def controlla(spazio, stampate, passi):
    atteso = _blocco("text").splitlines()
    if stampate != atteso:
        raise ValueError(f"{NOME}: il capitolo stampa righe diverse dal suo "
                         f"blocco di uscita:\n  {stampate}\n  {atteso}")
    nodi = [n for n, _ in passi]
    tipi = [t for _, t in passi]
    if set(nodi) - set(spazio["NODI"]):
        raise ValueError(f"{NOME}: passi su nodi che il grafo non ha: {nodi}")
    # la didascalia: dopo il guasto riparte dall'ultimo punto salvato
    g = tipi.index("guasto")
    if nodi[g + 1] != nodi[g] or nodi.count("pianifica") != 1:
        raise ValueError(f"{NOME}: la ripresa non parte dal nodo interrotto, "
                         f"o la pianificazione si ripete: {passi}")
    # la didascalia: torna una volta alla costruzione, poi si ferma alla firma
    if tipi.count("fallito") != 1 or tipi.count("passato") != 1:
        raise ValueError(f"{NOME}: serve un collaudo fallito e uno passato")
    a = tipi.index("attesa")  # il gettone, lì, aspetta sulla firma
    if nodi[a] != "rilascia" or passi[a + 1:] != [("rilascia", "fatto")]:
        raise ValueError(f"{NOME}: il rilascio deve venire subito dopo la "
                         f"firma, ed essere l'ultimo passo: {passi}")


# --------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------
W, H = 800, 345
Y_NODI = 150
X_NODI = {"pianifica": 105, "costruisci": 300, "collauda": 495,
          "rilascia": 690}
L_NODO, A_NODO = 130, 48
Y_FILA = 300                      # la fila dei passi salvati
RIVELA = 0.90                     # la frazione della durata del percorso

ETICHETTA = {"fatto": "salvato", "fallito": "fallito", "passato": "passato",
             "guasto": "guasto", "attesa": "firma"}


def costruisci() -> Figura:
    spazio, stampate = esegui_il_capitolo()
    passi = percorso(stampate)
    controlla(spazio, stampate, passi)

    m = len(passi)
    fetta = RIVELA * 100 / m

    def inizio(i: int) -> float:
        return i * fetta

    corpo, anim = [], []
    corpo.append(f'<text class="ttl" x="40" y="30">un filo di lavoro che '
                 f'attraversa il grafo</text>')

    # gli archi: la fila in avanti, il ritorno dal collaudo, la firma
    nomi = list(X_NODI)
    for a, b in zip(nomi, nomi[1:]):
        x1, x2 = X_NODI[a] + L_NODO / 2, X_NODI[b] - L_NODO / 2
        corpo.append(f'<line class="arco" x1="{x1:.0f}" y1="{Y_NODI}" '
                     f'x2="{x2 - 8:.0f}" y2="{Y_NODI}"/>')
        corpo.append(f'<path class="punta" d="M {x2 - 10:.0f} {Y_NODI - 5} '
                     f'L {x2:.0f} {Y_NODI} L {x2 - 10:.0f} {Y_NODI + 5} Z"/>')
    xc, xk = X_NODI["collauda"], X_NODI["costruisci"]
    yt = Y_NODI - A_NODO / 2
    corpo.append(f'<path class="arco" d="M {xc - 20} {yt} C {xc - 40} '
                 f'{yt - 62} {xk + 40} {yt - 62} {xk + 22} {yt - 6}"/>')
    corpo.append(f'<path class="punta" d="M {xk + 16} {yt - 12} L {xk + 20} '
                 f'{yt} L {xk + 28} {yt - 10} Z"/>')
    corpo.append(f'<text class="lbs" x="{(xc + xk) / 2:.0f}" y="{yt - 52}" '
                 f'text-anchor="middle">fallito: si torna, al massimo '
                 f'{spazio["TETTO"]} tentativi</text>')
    xf = (X_NODI["collauda"] + X_NODI["rilascia"]) / 2
    corpo.append(f'<line class="firma" x1="{xf:.0f}" y1="{Y_NODI - 16}" '
                 f'x2="{xf:.0f}" y2="{Y_NODI + 16}"/>')
    corpo.append(f'<text class="lbs" x="{xf:.0f}" y="{Y_NODI + 36}" '
                 f'text-anchor="middle">firma</text>')

    # i nodi; il rilascio è irreversibile
    for nome, x in X_NODI.items():
        classe = "nodo irr" if nome in spazio["IRREVERSIBILI"] else "nodo"
        corpo.append(f'<rect class="{classe}" x="{x - L_NODO / 2:.0f}" '
                     f'y="{Y_NODI - A_NODO / 2:.0f}" width="{L_NODO}" '
                     f'height="{A_NODO}" rx="8"/>')
        corpo.append(f'<text class="lbl" x="{x}" y="{Y_NODI + 5}" '
                     f'text-anchor="middle">{nome}</text>')
    corpo.append(f'<text class="lbs" x="{X_NODI["rilascia"]}" '
                 f'y="{Y_NODI + A_NODO / 2 + 18:.0f}" text-anchor="middle">'
                 f'irreversibile</text>')

    # il segno del guasto sul collaudo, che compare quando succede
    g = [t for _, t in passi].index("guasto")
    xg, yg = X_NODI["collauda"] + L_NODO / 2 - 4, Y_NODI - A_NODO / 2 + 4
    anim.append(keyframes("guasto", [
        (0.0, "opacity:0"), (inizio(g), "opacity:0"),
        (inizio(g) + 0.5, "opacity:1"), (100.0, "opacity:1")]))
    corpo.append(f'<g class="comp" style="animation:guasto var(--d) linear '
                 f'infinite"><path class="croce" d="M {xg - 7} {yg - 7} L '
                 f'{xg + 7} {yg + 7} M {xg + 7} {yg - 7} L {xg - 7} {yg + 7}"/>'
                 f'</g>')

    # la fila dei passi, che si riempie nell'ordine
    passo_x = (W - 80) / m
    corpo.append(f'<text class="lbs" x="40" y="{Y_FILA - 34}">i passi, '
                 f'nell\'ordine in cui il filo li ha fatti</text>')
    for i, (nodo, tipo) in enumerate(passi):
        x = 40 + i * passo_x
        anim.append(keyframes(f"p{i}", [
            (0.0, "opacity:0"), (inizio(i), "opacity:0"),
            (inizio(i) + 0.5, "opacity:1"), (100.0, "opacity:1")]))
        classe = {"guasto": "casella rotta", "attesa": "casella attesa"}.get(
            tipo, "casella")
        corpo.append(f'<g class="comp" style="animation:p{i} var(--d) linear '
                     f'infinite"><rect class="{classe}" x="{x:.1f}" '
                     f'y="{Y_FILA - 22}" width="{passo_x - 8:.1f}" height="44" '
                     f'rx="6"/><text class="lbs" x="{x + (passo_x - 8) / 2:.1f}" '
                     f'y="{Y_FILA - 4}" text-anchor="middle">{nodo}</text>'
                     f'<text class="lbs nota" x="{x + (passo_x - 8) / 2:.1f}" '
                     f'y="{Y_FILA + 13}" text-anchor="middle">'
                     f'{ETICHETTA[tipo]}</text></g>')

    # il gettone: a riposo sul rilascio, prima scorre le soste del percorso
    x_fine = X_NODI[passi[-1][0]]
    tappe = []
    xf = (X_NODI["collauda"] + X_NODI["rilascia"]) / 2   # la firma
    for i, (nodo, tipo) in enumerate(passi):
        dx = (xf if tipo == "attesa" else X_NODI[nodo]) - x_fine
        tappe.append((inizio(i), f"transform:translateX({dx}px);"
                      f"opacity:{0.3 if tipo == 'guasto' else 1}"))
        tappe.append((inizio(i) + 0.6 * fetta,
                      f"transform:translateX({dx}px);"
                      f"opacity:{0.3 if tipo == 'guasto' else 1}"))
    tappe.append((RIVELA * 100, "transform:translateX(0px);opacity:1"))
    tappe.append((100.0, "transform:translateX(0px);opacity:1"))
    anim.append(keyframes("gettone", tappe))
    corpo.append(f'<circle class="tok" cx="{x_fine}" '
                 f'cy="{Y_NODI - A_NODO / 2 - 12}" r="8" '
                 f'style="animation:gettone var(--d) linear infinite"/>')

    salvati = sum(t not in ("guasto", "attesa") for _, t in passi)
    return Figura(
        larghezza=W, altezza=H,
        alt=f"Un grafo di quattro nodi in fila, pianifica, costruisci, "
            f"collauda e rilascia, con un arco che dal collaudo torna alla "
            f"costruzione, al massimo {spazio['TETTO']} tentativi, e una "
            f"firma sull'arco che porta al rilascio, segnato come "
            f"irreversibile. Un gettone percorre il grafo: pianifica, "
            f"costruisci, poi al collaudo si spegne (una croce resta sul "
            f"nodo); riparte dal collaudo, che fallisce, torna alla "
            f"costruzione, il secondo collaudo passa, si ferma alla firma e "
            f"infine arriva al rilascio. Sotto, una fila di {m} caselle "
            f"riporta i passi nell'ordine: {salvati} passi salvati, il "
            f"guasto e la firma.",
        corpo="".join(corpo),
        stile=f"""    .tok  {{ fill:{TERRACOTTA}; transform-box:view-box; }}
    .nodo {{ fill:none; stroke:{TEAL}; stroke-width:2; }}
    .irr  {{ stroke:{TERRACOTTA}; stroke-dasharray:5 3; }}
    .arco {{ fill:none; stroke:{BORDER_STRONG}; stroke-width:2; }}
    .punta {{ fill:{BORDER_STRONG}; }}
    .firma {{ stroke:{OCRA}; stroke-width:4; stroke-linecap:round; }}
    .croce {{ stroke:{TERRACOTTA}; stroke-width:3; stroke-linecap:round; }}
    .casella {{ fill:none; stroke:{TEAL}; stroke-width:1.5; }}
    .rotta {{ stroke:{TERRACOTTA}; stroke-dasharray:4 3; }}
    .attesa {{ stroke:{OCRA}; }}
    .nota {{ font-size:11px; }}""",
        animazioni=anim,
        durata=12.0,
        fermi=".tok, .comp",
    )
