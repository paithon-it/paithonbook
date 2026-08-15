#!/usr/bin/env python3
"""Il file con cui il libro si lascia citare.

    python3 scripts/genera-citazione.py             # scrive CITATION.cff
    python3 scripts/genera-citazione.py --verifica  # e' allineato al registro?

## A che cosa serve, visto che nessuno lo apre

`CITATION.cff` e' un formato che GitHub legge: se il file c'e', nella colonna
di destra del repository compare **«Cite this repository»**, con la citazione
gia' pronta in APA e in BibTeX. Se non c'e', quel riquadro non esiste, e il
DOI del libro resta una cosa che sa solo chi e' passato da Zenodo. E' esatta-
mente il pezzo che mancava: il deposito era fatto, ma su GitHub non si vedeva.

Lo leggono anche Zenodo (quando archivia una release ne prende autori e
titolo) e Zotero, quindi val la pena che dica la verita' su tutte e tre le
cose che di solito si sbagliano: l'ORCID, la licenza e la versione.

## Perche' si genera invece di scriverlo

Perche' dentro c'e' il numero di versione, e quello sta in un posto solo: la
voce in cima a `book/_dati/aggiornamenti.yml`. Scritto a mano qui, dopo due
pubblicazioni direbbe un numero e la pagina degli aggiornamenti un altro, e
nessuno se ne accorgerebbe: un file di metadati non lo rilegge nessuno. Vale
la stessa regola dei notebook compagni e della pagina del registro, il file e'
TRACCIATO e puo' restare indietro in silenzio, e per questo c'e' `--verifica`.

## Il DOI che c'e' qui e' quello «di tutte le versioni»

Zenodo ne conia due: uno per ogni deposito (`…21947220`, la 1.5.6) e uno che
li raccoglie tutti (`…21947219`) e che porta sempre all'ultimo. Qui va il
secondo, ed e' l'unico dei due che si puo' scrivere in un file: il primo
cambia a ogni pubblicazione, e sarebbe di nuovo un numero copiato a mano che
invecchia. Chi cita un passaggio che potrebbe cambiare usa quello della sua
versione, che sta sulla scheda di deposito.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

RADICE = Path(__file__).resolve().parent.parent
USCITA = RADICE / "CITATION.cff"

# Il DOI «concept» del libro su Zenodo: raccoglie tutte le versioni e apre
# sempre l'ultima. Vedi il docstring per perche' non e' quello della 1.5.6.
DOI = "10.5281/zenodo.21947219"
ORCID = "https://orcid.org/0009-0008-8586-9018"


def versione() -> tuple[str, str]:
    """(numero, data ISO) dalla voce in cima al registro.

    Si carica il file per percorso invece di importarlo: il nome ha un
    trattino, che in un `import` non si puo' scrivere.
    """
    percorso = RADICE / "scripts" / "genera-aggiornamenti.py"
    spec = importlib.util.spec_from_file_location("pt_registro", percorso)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    numero, giorno = modulo.versione_corrente()
    return numero, giorno.isoformat()


def genera() -> str:
    numero, giorno = versione()
    anno, mese, _ = giorno.split("-")
    return f"""# Come si cita questo libro.
#
# NON si scrive a mano: lo genera `python3 scripts/genera-citazione.py` dal
# registro delle versioni (`book/_dati/aggiornamenti.yml`), che e' l'unico
# posto in cui il numero di versione esiste. `--verifica` dice se e' rimasto
# indietro.
#
# La forma e' quella che vuole lo standard CFF 1.2.0, e non e' quella che
# verrebbe spontanea: al primo livello `type` puo' valere solo `software` o
# `dataset`, quindi il *libro* si descrive dentro `preferred-citation`, che e'
# poi il blocco che GitHub preferisce quando c'e'.
cff-version: 1.2.0
message: "Se citi questo libro, usa il deposito qui sotto."
title: "Paithon Book"
abstract: >-
  Libro di intelligenza artificiale in italiano, gratuito e in aggiornamento
  continuo. Ogni concetto e' spiegato su due livelli: Elementare, con analogie
  quotidiane e senza prerequisiti, e Superiore, con la trattazione formale.
  Copre machine learning, deep learning e reinforcement learning con Python e
  PyTorch, dall'algebra lineare ai Transformer.
type: software
authors:
  - family-names: "Messina"
    given-names: "Francesco"
    orcid: "{ORCID}"
    website: "https://www.paithon.it"
version: "{numero}"
date-released: "{giorno}"
url: "https://book.paithon.it/main/"
repository-code: "https://github.com/paithon-it/paithonbook"
license:
  - CC-BY-NC-ND-4.0
  - Apache-2.0
identifiers:
  - type: doi
    value: "{DOI}"
    description: >-
      Il DOI di tutte le versioni del libro: apre sempre l'ultima depositata.
keywords:
  - intelligenza artificiale
  - machine learning
  - deep learning
  - reinforcement learning
  - Python
  - PyTorch
  - divulgazione scientifica
preferred-citation:
  type: book
  title: "Paithon Book. Il Libro di Intelligenza Artificiale che spiega due volte"
  authors:
    - family-names: "Messina"
      given-names: "Francesco"
      orcid: "{ORCID}"
  version: "{numero}"
  year: {int(anno)}
  month: {int(mese)}
  doi: "{DOI}"
  url: "https://book.paithon.it/main/"
  publisher:
    name: "paithon.it"
  languages:
    - ita
"""


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Genera CITATION.cff dal registro delle versioni.")
    ap.add_argument("--verifica", action="store_true",
                    help="esce con 1 se il file sul disco non e' quello atteso")
    argomenti = ap.parse_args()

    atteso = genera()

    if argomenti.verifica:
        if not USCITA.is_file():
            print(f"manca {USCITA.relative_to(RADICE)}")
            print("  python3 scripts/genera-citazione.py")
            return 1
        if USCITA.read_text(encoding="utf-8") != atteso:
            print(f"{USCITA.relative_to(RADICE)} non e' allineato al registro")
            print("  python3 scripts/genera-citazione.py")
            return 1
        numero, _ = versione()
        print(f"allineato: versione {numero}, DOI {DOI}")
        return 0

    USCITA.write_text(atteso, encoding="utf-8")
    numero, giorno = versione()
    print(f"  {USCITA.relative_to(RADICE)}  versione {numero} del {giorno}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
