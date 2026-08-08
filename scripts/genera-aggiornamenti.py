#!/usr/bin/env python3
"""Il registro degli aggiornamenti: che cosa e' cambiato nel libro, e quando.

    python3 scripts/genera-aggiornamenti.py             # scrive book/aggiornamenti.md
    python3 scripts/genera-aggiornamenti.py --verifica  # e' allineata alla fonte?
    python3 scripts/genera-aggiornamenti.py --versione  # stampa il numero e basta

## Perche' non e' un changelog di commit

Un elenco di commit e' la storia dei *file*, e per chi legge il libro non
significa niente: dice `fix(book):` e nomina un percorso. Qui si registra
l'altra cosa, che e' quella che interessa a un lettore tornato dopo un mese:
quali sezioni sono nuove, che cosa e' stato corretto, e su quale pagina si
clicca per andarci. Il dettaglio tecnico resta dov'e' gia', nella storia
pubblica del repository.

## La fonte e' il YAML, non questa pagina

`book/_dati/aggiornamenti.yml` tiene le voci; qui c'e' solo la cornice (la
prosa che non cambia) e il modo di disegnarle. La ragione di separarli e' che
il numero di versione serve anche altrove (la landing, la scheda in cima
all'indice, l'oggetto del commit di pubblicazione), e un numero scritto in
quattro posti e' un numero che fra tre mesi ne dice due diversi. Chi lo legge:

  - `book/_ext/pt_conteggi.py`, che lo passa al testo come `{{ versione }}` e
    al tema come `pt_versione`;
  - `strumenti/pubblica.py`, che ci intesta il commit pubblico.

## Il titolo dei link non si scrive nel YAML

Lo prende il `_toc.yml`, che e' gia' la fonte dei titoli brevi del libro: una
voce del registro dice il *file*, e il nome con cui compare segue la pagina
anche se la pagina viene ribattezzata. Se il file non e' nel toc lo script si
ferma e lo dice, perche' un link morto in una pagina di servizio nessuno lo
segnala.

## Rigenerare, e quando

La pagina e' TRACCIATA (la build ha bisogno di un `.md`, non di un `.yml`),
quindi come i notebook compagni puo' restare indietro in silenzio: si tocca il
YAML e ci si dimentica. `--verifica` lo dice, e la sequenza giusta e' quella
della pubblicazione: si aggiunge la voce, si rigenera, si committa, si
pubblica.
"""

from __future__ import annotations

import argparse
import datetime
import re
import sys
from pathlib import Path

import yaml

RADICE = Path(__file__).resolve().parent.parent
LIBRO = RADICE / "book"
FONTE = LIBRO / "_dati" / "aggiornamenti.yml"
PAGINA = LIBRO / "aggiornamenti.md"

MESI = ("gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio",
        "agosto", "settembre", "ottobre", "novembre", "dicembre")

# I quattro tipi di voce, nell'ordine in cui compaiono dentro una versione, e
# l'intestazione con cui si presentano. L'ordine non e' casuale: chi torna sul
# libro cerca prima quello che non aveva letto, poi quello che aveva letto
# sbagliato.
TIPI = {
    "nuovo": "Sezioni nuove",
    "ampliato": "Pagine ampliate",
    "corretto": "Correzioni",
    "impianto": "Impianto",
}

TESTA = """\
<!-- GENERATO da scripts/genera-aggiornamenti.py: non modificare a mano.
     La fonte e' book/_dati/aggiornamenti.yml, e la prossima rigenerazione
     cancella qualunque cosa venga scritta qui dentro. -->

(aggiornamenti)=

# Che cosa è cambiato, e quando

Questo libro non esce e finisce: cambia. Una sezione si aggiunge, un conto
sbagliato si corregge, una spiegazione che non ha funzionato si riscrive.
Questa pagina è il registro di quei cambiamenti, dal più recente al più
vecchio, con il link alla pagina toccata. Se hai letto un capitolo un mese fa,
di qui vedi in un minuto se nel frattempo è cambiato qualcosa, e dove.

Non è la storia dei commit, che parla di file e serve a chi scrive il libro:
quella è pubblica e sta [su
GitHub](https://github.com/paithon-it/paithonbook/commits/main). Qui si parla
di quello che legge chi legge.

Se in una pagina trovi un errore, [segnalalo](https://github.com/paithon-it/paithonbook/issues):
le correzioni che arrivano da fuori entrano in questo registro come tutte le
altre, e chi le ha segnalate è citato nel commit che le porta online.

## Come si legge il numero

Il numero della versione è fatto di tre cifre, `impianto.sezioni.correzioni`:

- la **prima** sale quando cambia l'impianto: una parte nuova nell'indice, un
  riordino che cambia il percorso di lettura, la licenza;
- la **seconda** quando arriva un capitolo o una sezione;
- la **terza** quando si corregge o si rifinisce quello che c'è già.

Quando una cifra sale, quelle alla sua destra tornano a zero. Se una
pubblicazione porta sia sezioni nuove sia correzioni, il numero racconta la
cosa più grossa che è successa e l'elenco racconta tutto il resto.

Una versione corrisponde a una **pubblicazione**, non a una giornata di
lavoro: il libro si scrive tutti i giorni e si pubblica quando un pezzo sta in
piedi.
"""

CODA = """\
## Prima della 1.0

Il libro nasce nel 2019, scritto per uscire su carta con un editore. Non è
successo, e il manoscritto è rimasto in un cassetto: la prima forma in cui
questo testo è arrivato a qualcuno è quella che stai leggendo. Il repository è
più giovane di cinque anni, quindi le date qui sotto non sono la storia del
*libro*: sono la storia della sua **versione online**, cioè i giorni in cui il
testo è arrivato qui dentro.

Il 13 giugno 2024 nasce l'impianto Jupyter Book, con la licenza CC BY-NC-ND e
le prime pagine: l'introduzione, il capitolo su Python e, quattro giorni dopo,
i due livelli di lettura, che sono poi diventati la regola di tutto il resto.
Nell'ottobre del 2025 si allarga il capitolo sui Transformer. Poi si ferma.

Riparte nel luglio del 2026, e in tre settimane diventa un'altra cosa: prima
l'ossatura del manoscritto (matematica, machine learning, reti neurali, deep
learning, visione artificiale, NLP, GAN, reinforcement learning, speech
recognition), poi, uno dietro l'altro, i capitoli che nel 2019 non potevano
esserci, da PyTorch e le GPU fino agli agenti, ai modelli di diffusione, agli
state space model, ai sistemi multi-agente.

Agosto 2026 è il mese della rilettura: ogni capitolo ripassato sui fatti, sui
conti e sul codice, i notebook compagni riallineati alle pagine, le figure che
nessuna pagina richiamava messe da parte, e le sezioni nuove scritte per
chiudere i buchi che la rilettura aveva trovato.

Nessuna di queste tappe ha un numero di versione, e non gliene diamo uno
adesso: non erano pubblicazioni, erano lavoro. Il registro comincia dalla
1.0.0.
"""


# --------------------------------------------------------------------------
# la fonte
# --------------------------------------------------------------------------

class LettoreSevero(yaml.SafeLoader):
    """Un YAML con due chiavi uguali nella stessa mappa e' un errore, non un
    aggiornamento silenzioso.

    PyYAML tiene l'ultima e non dice niente: una modifica che perde per strada
    un `- versione:` fonde due voci in una, e quella sopra sparisce dal
    registro senza che nulla fallisca. E' successo, e da qui non si vedeva.
    """


def _mappa_senza_doppioni(lettore, nodo, deep=False):
    viste = set()
    for chiave, _ in nodo.value:
        nome = lettore.construct_object(chiave, deep=True)
        if nome in viste:
            raise yaml.constructor.ConstructorError(
                None, None, f"chiave ripetuta nella stessa voce: {nome!r}",
                chiave.start_mark)
        viste.add(nome)
    return yaml.SafeLoader.construct_mapping(lettore, nodo, deep=deep)


LettoreSevero.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mappa_senza_doppioni)


def carica() -> list[dict]:
    try:
        dati = yaml.load(FONTE.read_text(encoding="utf-8"), LettoreSevero) or {}
    except yaml.YAMLError as errore:
        raise SystemExit(f"{FONTE.relative_to(RADICE)}: {errore}")
    versioni = dati.get("versioni") or []
    if not versioni:
        raise SystemExit(f"{FONTE.relative_to(RADICE)}: nessuna versione.")

    numeri = [str(v.get("versione", "")) for v in versioni]
    if len(set(numeri)) != len(numeri):
        raise SystemExit("numeri di versione ripetuti nel registro: "
                         + ", ".join(sorted(n for n in set(numeri)
                                            if numeri.count(n) > 1)))
    return versioni


def titoli() -> dict[str, tuple[str, str]]:
    """Per ogni file del toc: (titolo breve della pagina, titolo del capitolo).

    Per l'`overview.md` di un capitolo i due coincidono, ed e' il segnale che
    la voce riguarda il capitolo intero e non una sua sezione.
    """
    dati = yaml.safe_load((LIBRO / "_toc.yml").read_text(encoding="utf-8"))
    fuori: dict[str, tuple[str, str]] = {}
    for parte in dati.get("parts", []) or []:
        for cap in parte.get("chapters", []) or []:
            nome = cap.get("title", "")
            fuori[cap["file"]] = (nome, nome)
            for sez in cap.get("sections", []) or []:
                fuori[sez["file"]] = (sez.get("title", ""), nome)
    return fuori


def data_estesa(giorno: datetime.date) -> str:
    """`2026-08-08` -> `8 agosto 2026`."""
    return f"{giorno.day} {MESI[giorno.month - 1]} {giorno.year}"


def versione_corrente() -> tuple[str, datetime.date]:
    prima = carica()[0]
    return str(prima["versione"]), prima["data"]


# --------------------------------------------------------------------------
# il disegno
# --------------------------------------------------------------------------

def ancora(numero: str) -> str:
    """`1.0.0` -> `v1-0-0`, un bersaglio stabile a cui linkare da fuori."""
    return "v" + re.sub(r"[^0-9a-z]+", "-", numero.lower())


def riga(voce: dict, mappa: dict[str, tuple[str, str]]) -> str:
    """Una voce dell'elenco: il link alla pagina, il capitolo, il testo.

    Il link non e' in grassetto: un ruolo MyST dentro `**...**` si annida bene
    in teoria, ma questa pagina si costruisce solo in remoto e un link colorato
    si distingue gia' da se'. Il testo resta una frase a se' (punto, non due
    punti) perche' comincia spesso con un nome proprio.
    """
    testo = " ".join((voce.get("testo") or "").split())
    pagina = voce.get("pagina")
    if not pagina:
        return f"- {testo}"

    percorso = "/" + pagina.rsplit(".", 1)[0]
    titolo, capitolo = mappa.get(pagina, ("", ""))
    if not titolo:                          # fuori dal toc: succede solo se
        titolo = Path(pagina).stem          # qualcuno rinomina senza guardare
    marca = f"{{doc}}`{titolo} <{percorso}>`"
    if capitolo and capitolo != titolo:
        return f"- {marca} ({capitolo}). {testo}"
    return f"- {marca}. {testo}"


def blocco(versione: dict, mappa: dict[str, tuple[str, str]]) -> str:
    numero = str(versione["versione"])
    righe = [f"({ancora(numero)})=",
             "",
             f"## {numero} · {data_estesa(versione['data'])}",
             ""]
    if titolo := versione.get("titolo"):
        righe += [f"*{titolo}*", ""]
    if nota := versione.get("nota"):
        righe += [" ".join(nota.split()), ""]

    voci = versione.get("voci") or []
    ignoti = {v.get("tipo") for v in voci} - set(TIPI)
    if ignoti:
        raise SystemExit(f"versione {numero}: tipo sconosciuto {sorted(ignoti)}; "
                         f"i tipi sono {', '.join(TIPI)}.")
    for tipo, intestazione in TIPI.items():
        gruppo = [v for v in voci if v.get("tipo") == tipo]
        if not gruppo:
            continue
        righe += [f"### {intestazione}", ""]
        righe += [riga(v, mappa) for v in gruppo]
        righe += [""]
    return "\n".join(righe)


def genera() -> str:
    versioni = carica()
    mappa = titoli()

    manca = [v["pagina"] for ver in versioni for v in ver.get("voci") or []
             if v.get("pagina") and v["pagina"] not in mappa]
    if manca:
        raise SystemExit("pagine che il _toc.yml non conosce: "
                         + ", ".join(sorted(set(manca))))

    pezzi = [TESTA] + [blocco(v, mappa) for v in versioni] + [CODA]
    return "\n".join(pezzi).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Genera book/aggiornamenti.md dal registro delle versioni.")
    ap.add_argument("--verifica", action="store_true",
                    help="esce con 1 se la pagina sul disco non e' quella attesa")
    ap.add_argument("--versione", action="store_true",
                    help="stampa il numero della versione corrente e basta")
    argomenti = ap.parse_args()

    if argomenti.versione:
        numero, giorno = versione_corrente()
        print(numero)
        return 0

    atteso = genera()
    numero, giorno = versione_corrente()

    if argomenti.verifica:
        if not PAGINA.is_file():
            print(f"manca {PAGINA.relative_to(RADICE)}")
            print("  python3 scripts/genera-aggiornamenti.py")
            return 1
        if PAGINA.read_text(encoding="utf-8") != atteso:
            print(f"{PAGINA.relative_to(RADICE)} non e' allineata al registro")
            print("  python3 scripts/genera-aggiornamenti.py")
            return 1
        print(f"allineata: versione {numero} del {data_estesa(giorno)}")
        return 0

    PAGINA.write_text(atteso, encoding="utf-8")
    quante = len(carica())
    print(f"  {PAGINA.relative_to(RADICE)}  {len(atteso.encode()) / 1000:.1f} kB")
    print(f"versione {numero} del {data_estesa(giorno)}, "
          f"{quante} version{'e' if quante == 1 else 'i'} nel registro")
    return 0


if __name__ == "__main__":
    sys.exit(main())
