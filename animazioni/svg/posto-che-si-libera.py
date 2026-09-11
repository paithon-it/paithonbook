"""Continuous batching: il posto che si libera viene ripreso all'iterazione dopo.

La sezione «Servire un LLM» dice due cose che una figura ferma può solo
riassumere. La prima è che lo scheduler del continuous batching
lavora **a livello di singola iterazione**: non appena una sequenza emette il
suo token di fine, un'altra richiesta ne prende il posto nel batch. La seconda
è che il batching statico tiene insieme il mazzo fino all'ultimo, quindi una
risposta lunga inchioda i posti che le altre hanno lasciato. Qui le due cose
si vedono succedere, sullo stesso orologio: a sinistra i posti si svuotano e
restano vuoti, a destra si riempiono subito.

Niente numeri scritti a mano. `esegui()` fa girare le due politiche sulle
stesse dieci richieste, con le stesse lunghezze, e conta i posti-iterazione
usati; `verifica()` pretende con degli assert i numeri che la didascalia
promette, cioè che nell'orizzonte mostrato lo statico ne usi 28 su 48 e il
continuo 48 su 48, e che le richieste concluse siano tre contro cinque.

Due semplificazioni, dichiarate perché la scena non prometta più di quanto
contenga. Le richieste sono tutte in coda dall'inizio (non ne arrivano di
nuove), e il costo di caricare una richiesta nel batch è zero: è la stessa
idealizzazione della figura ferma che questa sostituisce, e serve a isolare
l'unico meccanismo in gioco, cioè chi occupa il posto e quando. Il guadagno
vero, misurato, la pagina lo dà con il numero degli autori di vLLM.

Lo stato di riposo è l'ultima iterazione mostrata, ed è il fotogramma che
riassume tutto: a sinistra un posto pieno e tre fermi con la coda ancora
intera, a destra quattro posti pieni e la coda quasi finita.
"""

from paithon_svg import *

NOME = "posto-che-si-libera"
TITOLO = "continuous batching: il posto che si libera viene ripreso subito"

# --------------------------------------------------------------------------
# Le richieste e le due politiche
# --------------------------------------------------------------------------
POSTI = 4                       # quante sequenze stanno nel batch
ORIZZONTE = 12                  # le iterazioni che la scena mostra
LETTERE = "ABCDEFGHIJ"
# Quante iterazioni dura ciascuna risposta. La prima è la lunga che nel
# batching statico tiene fermi gli altri tre posti fino alla fine.
DURATE = (12, 3, 5, 8, 6, 4, 7, 5, 9, 6)

# Quello che la didascalia promette, e che `verifica()` non lascia scivolare.
USATI_STATICO = 28
USATI_CONTINUO = POSTI * ORIZZONTE
CONCLUSE_STATICO = 3
CONCLUSE_CONTINUO = 5


def esegui() -> dict:
    """Le due politiche sulle stesse richieste: chi occupa quale posto, quando.

    Restituisce, per ciascuna, la lista dei tratti (posto, richiesta, prima e
    ultima iterazione) e lo stato della coda iterazione per iterazione.
    """
    fuori = {}
    for politica in ("statico", "continuo"):
        tratti, coda = [], list(range(len(DURATE)))
        # quando il posto si libera, cioè la prima iterazione in cui è di nuovo
        # disponibile; all'inizio sono tutti liberi dalla prima.
        libero_da = [1] * POSTI
        in_coda = []
        for it in range(1, ORIZZONTE + 1):
            if politica == "statico":
                # il mazzo si rinnova tutto insieme: si carica solo quando
                # l'ultima sequenza del batch precedente ha finito.
                pronto = all(libero_da[p] <= it for p in range(POSTI))
                posti_da_riempire = range(POSTI) if pronto else ()
            else:
                # a ogni iterazione, ogni posto libero prende il primo in coda.
                posti_da_riempire = [p for p in range(POSTI) if libero_da[p] <= it]
            for p in posti_da_riempire:
                if not coda:
                    continue
                r = coda.pop(0)
                tratti.append(dict(posto=p, richiesta=r, da=it,
                                   a=it + DURATE[r] - 1))
                libero_da[p] = it + DURATE[r]
            in_coda.append(len(coda))
        fuori[politica] = dict(tratti=tratti, coda=in_coda)
    return fuori


def occupazione(tratti: list[dict], it: int) -> dict:
    """Chi sta in quale posto all'iterazione `it`; None se il posto è fermo."""
    dentro = {p: None for p in range(POSTI)}
    for t in tratti:
        if t["da"] <= it <= t["a"]:
            dentro[t["posto"]] = t["richiesta"]
    return dentro


def usati(tratti: list[dict], fino: int) -> int:
    """Posti-iterazione occupati dalla prima iterazione a `fino` compresa."""
    return sum(min(t["a"], fino) - t["da"] + 1
               for t in tratti if t["da"] <= fino)


def concluse(tratti: list[dict], it: int) -> int:
    """Richieste che hanno finito prima che l'iterazione `it` cominciasse."""
    return sum(1 for t in tratti if t["a"] < it)


def verifica(dati: dict) -> None:
    """La scena racconta ancora i numeri che la didascalia promette?"""
    st, co = dati["statico"]["tratti"], dati["continuo"]["tratti"]

    assert usati(st, ORIZZONTE) == USATI_STATICO, \
        (f"lo statico usa {usati(st, ORIZZONTE)} posti-iterazione su "
         f"{POSTI * ORIZZONTE}, la didascalia dice {USATI_STATICO}")
    assert usati(co, ORIZZONTE) == USATI_CONTINUO, \
        (f"il continuo ne usa {usati(co, ORIZZONTE)} su {POSTI * ORIZZONTE}, "
         f"la didascalia dice che non ne lascia fermo nessuno")

    assert concluse(st, ORIZZONTE) == CONCLUSE_STATICO, \
        f"lo statico ne conclude {concluse(st, ORIZZONTE)}, non {CONCLUSE_STATICO}"
    assert concluse(co, ORIZZONTE) == CONCLUSE_CONTINUO, \
        f"il continuo ne conclude {concluse(co, ORIZZONTE)}, non {CONCLUSE_CONTINUO}"

    # Non solo i totali: che nel continuo un posto liberato sia ripreso
    # all'iterazione successiva, e nello statico nessuno. Sull'orizzonte di
    # questa scena, dove la richiesta piu' lunga dura quanto l'orizzonte, lo
    # statico non ricarica mai: e' una proprieta' della scena, non della
    # politica, ed e' quella che la didascalia promette.
    for politica, tratti, deve in (("continuo", co, True), ("statico", st, False)):
        riprese = 0
        for t in tratti:
            if t["a"] < ORIZZONTE:
                dopo = [u for u in tratti
                        if u["posto"] == t["posto"] and u["da"] == t["a"] + 1]
                riprese += len(dopo)
        assert (riprese > 0) == deve, \
            (f"{politica}: {riprese} posti ripresi subito, "
             f"la scena ne promette {'almeno uno' if deve else 'nessuno'}")

    # La coda: ferma da una parte, che si svuota dall'altra.
    assert dati["statico"]["coda"][-1] == dati["statico"]["coda"][0], \
        "nello statico la coda si muove, la scena dice che resta ferma"
    assert dati["continuo"]["coda"][-1] < dati["continuo"]["coda"][0], \
        "nel continuo la coda non si svuota"

    # Nella prima iterazione le due politiche sono indistinguibili: se non lo
    # fossero, la scena starebbe confrontando due cose diverse fin dall'inizio.
    assert occupazione(st, 1) == occupazione(co, 1), \
        "alla prima iterazione le due sale non partono uguali"


# --------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------
LARG, ALT = 720, 442
Y_ITER = 26
Y_TIT = 58
X_PAN = (46, 374)               # il bordo sinistro dei due pannelli
W_PAN = 300
Y_SALA, H_SALA = 70, 182        # la cornice della GPU
X_SLOT, W_SLOT = 14, 272        # lo slot, rispetto al bordo del pannello
Y_SLOT0, H_SLOT, PASSO = 82, 32, 42
Y_CODA = 288                    # la riga della coda
W_PILL, H_PILL, GAP = 22, 16, 6
X_PILL = 86
Y_SERVITE, Y_USATI = 322, 346
Y_CHIUSA = 400

RAMPA = 1.1                     # la dissolvenza, in punti di timeline
PASSO_T = 100.0 / ORIZZONTE

PAROLE = ("nessuna", "una", "due", "tre", "quattro", "cinque", "sei", "sette",
          "otto", "nove", "dieci")


def perc(n: int, d: int) -> str:
    return f"{round(100 * n / d)}%"


def avvolgi(testo: str, colonne: int) -> list[str]:
    """A capo sulle parole: tagliare a caratteri spezza «58%» in «5» e «8%»."""
    righe, riga = [], ""
    for parola in testo.split():
        prova = f"{riga} {parola}".strip()
        if len(prova) > colonne and riga:
            righe.append(riga)
            riga = parola
        else:
            riga = prova
    return righe + [riga] if riga else righe


def visibile(nome: str, da: int, a: int) -> tuple[str, int]:
    """`@keyframes` che accende l'elemento nelle iterazioni da..a comprese.

    Torna anche l'opacita' di riposo: quella dell'ultima iterazione mostrata,
    perche' il disegno fermo e' lo stato finale.
    """
    t_in, t_out = (da - 1) * PASSO_T, min(a * PASSO_T, 100.0)
    fermo = 1 if a >= ORIZZONTE else 0
    tappe = {0.0: 1 if da <= 1 else 0, 100.0: fermo}
    if da > 1:
        tappe[max(t_in - RAMPA, 0.01)] = 0
        tappe[t_in] = 1
    if a < ORIZZONTE:
        tappe[t_out] = 1
        tappe[min(t_out + RAMPA, 99.99)] = 0
    righe = [(p, f"opacity:{v}") for p, v in sorted(tappe.items())]
    return keyframes(nome, righe), fermo


def costruisci() -> Figura:
    dati = esegui()
    verifica(dati)

    corpo, anim = [], []
    contatore = [0]

    def acceso(da: int, a: int) -> str:
        """Registra i keyframes e torna lo stile (con l'opacita' di riposo)."""
        nome = f"v{contatore[0]}"
        contatore[0] += 1
        kf, fermo = visibile(nome, da, a)
        anim.append(kf)
        return f'opacity="{fermo}" style="animation:{nome} var(--d) infinite"'

    # il contatore delle iterazioni, in comune fra i due pannelli
    for it in range(1, ORIZZONTE + 1):
        corpo.append(
            f'<text class="ite" x="{LARG / 2:.0f}" y="{Y_ITER}" '
            f'text-anchor="middle" {acceso(it, it)}>'
            f'iterazione {it} di {ORIZZONTE}</text>')

    for k, (politica, titolo) in enumerate(
            (("statico", "Batching statico"),
             ("continuo", "Continuous batching"))):
        x0 = X_PAN[k]
        tratti = dati[politica]["tratti"]
        coda = dati[politica]["coda"]

        corpo.append(f'<text class="pan" x="{x0}" y="{Y_TIT}">{titolo}</text>')
        corpo.append(
            f'<rect class="sala" x="{x0}" y="{Y_SALA}" width="{W_PAN}" '
            f'height="{H_SALA}" rx="6"/>')

        # i posti fermi: un tratteggio per ogni buco lasciato aperto
        for p in range(POSTI):
            it = 1
            while it <= ORIZZONTE:
                if occupazione(tratti, it)[p] is None:
                    fine = it
                    while (fine + 1 <= ORIZZONTE
                           and occupazione(tratti, fine + 1)[p] is None):
                        fine += 1
                    y = Y_SLOT0 + p * PASSO
                    corpo.append(
                        f'<rect class="fermo" x="{x0 + X_SLOT}" y="{y}" '
                        f'width="{W_SLOT}" height="{H_SLOT}" rx="4" '
                        f'{acceso(it, fine)}/>')
                    corpo.append(
                        f'<text class="fer" x="{x0 + X_SLOT + W_SLOT / 2:.0f}" '
                        f'y="{y + 21}" text-anchor="middle" '
                        f'{acceso(it, fine)}>posto fermo</text>')
                    it = fine + 1
                else:
                    it += 1

        # le richieste in corso
        for t in tratti:
            y = Y_SLOT0 + t["posto"] * PASSO
            da, a = t["da"], min(t["a"], ORIZZONTE)
            corpo.append(
                f'<rect class="occ" x="{x0 + X_SLOT}" y="{y}" '
                f'width="{W_SLOT}" height="{H_SLOT}" rx="4" {acceso(da, a)}/>')
            corpo.append(
                f'<text class="ric" x="{x0 + X_SLOT + W_SLOT / 2:.0f}" '
                f'y="{y + 21}" text-anchor="middle" {acceso(da, a)}>'
                f'richiesta {LETTERE[t["richiesta"]]} '
                f'({DURATE[t["richiesta"]]} iterazioni)</text>')

        # la coda: una pastiglia per ogni richiesta che aspetta
        corpo.append(f'<text class="lbs" x="{x0}" y="{Y_CODA + 13}">in coda</text>')
        for i in range(max(coda)):
            ultima = max((it for it, n in enumerate(coda, 1) if n > i),
                         default=0)
            corpo.append(
                f'<rect class="pill" x="{x0 + X_PILL + i * (W_PILL + GAP)}" '
                f'y="{Y_CODA}" width="{W_PILL}" height="{H_PILL}" rx="3" '
                f'{acceso(1, ultima)}/>')

        # i due contatori, uno stato per iterazione
        for it in range(1, ORIZZONTE + 1):
            n = concluse(tratti, it)
            u = usati(tratti, it)
            corpo.append(
                f'<text class="lbl" x="{x0}" y="{Y_SERVITE}" {acceso(it, it)}>'
                f'concluse: {PAROLE[n]}</text>')
            corpo.append(
                f'<text class="lbl" x="{x0}" y="{Y_USATI}" {acceso(it, it)}>'
                f'posti-iterazione usati: {u} su {POSTI * it}</text>')

    st, co = dati["statico"]["tratti"], dati["continuo"]["tratti"]
    totale = POSTI * ORIZZONTE
    chiusa = (f'Stesso orologio e stessa coda: lo statico rinnova il mazzo '
              f'solo quando ha finito anche la più lunga; il continuo riprende '
              f'ogni posto all’iterazione dopo che si è liberato.')
    righe = avvolgi(chiusa, 86)
    assert len(righe) <= 2 and Y_CHIUSA + (len(righe) - 1) * 22 + 8 <= ALT, \
        f"la riga di chiusura esce dal disegno: {len(righe)} righe"
    for i, riga in enumerate(righe):
        corpo.append(f'<text class="lbl" x="{X_PAN[0]}" y="{Y_CHIUSA + i * 22}">'
                     f'{riga}</text>')

    return Figura(
        larghezza=LARG, altezza=ALT,
        alt=(f"Due sale a confronto sullo stesso orologio, {PAROLE[POSTI]} "
             f"posti ciascuna e una coda di {PAROLE[len(DURATE)]} richieste. Nel batching "
             f"statico le richieste partono insieme e chi finisce lascia il "
             f"posto vuoto fino alla fine della più lunga: all'ultima "
             f"iterazione mostrata resta un posto pieno e tre fermi, con la "
             f"coda ancora intera e {PAROLE[CONCLUSE_STATICO]} richieste "
             f"concluse. Nel continuous batching ogni posto che si libera "
             f"viene ripreso all'iterazione dopo: i quattro posti sono sempre "
             f"pieni, la coda si è quasi svuotata e le richieste concluse sono "
             f"{PAROLE[CONCLUSE_CONTINUO]}. In fondo il conto dei "
             f"posti-iterazione occupati, {usati(st, ORIZZONTE)} su {totale} "
             f"contro {usati(co, ORIZZONTE)} su {totale}."),
        corpo="".join(corpo),
        stile=f"""    .sala {{ fill:none; stroke:{BORDER_STRONG}; stroke-width:2; }}
    .pan  {{ font-family:{SANS}; font-size:15px; font-weight:700; fill:{INK}; }}
    .ite  {{ font-family:{SANS}; font-size:15px; font-weight:700; fill:{TERRACOTTA}; }}
    .occ  {{ fill:{TEAL}; stroke:{INK}; stroke-width:1.5; }}
    .ric  {{ font-family:{SANS}; font-size:13px; fill:{CREAM}; }}
    .fermo {{ fill:none; stroke:{TERRACOTTA}; stroke-width:1.5; stroke-dasharray:5 4; }}
    .fer  {{ font-family:{SANS}; font-size:13px; fill:{TERRACOTTA}; }}
    .pill {{ fill:{OCRA}; stroke:{INK}; stroke-width:1.2; }}""",
        animazioni=anim,
        durata=ORIZZONTE * 0.9,
        fermi=".occ, .ric, .fermo, .fer, .pill, .ite, .lbl",
    )
