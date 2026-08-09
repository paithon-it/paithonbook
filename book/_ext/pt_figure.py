"""Le figure si numerano per capitolo: «Fig. 3.2», non «Fig. 169».

Un libro numera le figure dentro il capitolo, e il lettore che incontra
«Fig. 3.2» sa già dove sta guardando. Qui il contatore era unico per tutto il
libro e arrivava a Fig. 302: un numero che non dice niente a nessuno e che
cambia tutto quando si aggiunge una figura a metà.

## Perché non basta `numbered: true` nel toc

È la via che Sphinx offre: si numerano le sezioni e `numfig_secnum_depth`
attacca il numero di sezione davanti a quello della figura. Provata, dà tre
problemi, e il terzo è quello che la esclude:

  - i titoli si portano dietro il numero (`<h1>4.3. Algebra lineare`), e
    l'indice pure, dove il libro ha già i suoi numeri;
  - con `use_multitoc_numbering: false` la numerazione RIPARTE a ogni parte
    del toc, quindi Matematica è il 4 e Machine Learning di nuovo l'1;
  - Sphinx conta come capitolo ogni voce del toc, prefazione compresa, mentre
    il libro no: `conta_capitoli()` chiama capitolo una voce di primo livello
    che sta dentro una cartella, ed è lo stesso criterio con cui la landing
    disegna le schede e il contatore CSS numera l'indice. Le figure
    direbbero 4 dove l'indice dice 03, e due numerazioni che si contraddicono
    sono peggio di una brutta.

Quindi il numero di capitolo si prende dalla stessa fonte di tutti gli altri
numeri del libro, il `_toc.yml`, e si riscrivono i contatori a valle.

## Come

Sphinx assegna i suoi numeri (`env.toc_fignumbers`) e li risolve nei
`{numref}` quando scrive le pagine. Fra le due cose questa estensione
riscrive la tabella, quindi i `{numref}` del testo leggono i numeri nuovi
senza sapere che sono cambiati e restano corretti per costruzione.

Il momento giusto è `env-get-updated`, con priorità più alta del numero di
default perché deve girare DOPO il collector che assegna i numeri. Non
`env-updated`, che il nome farebbe pensare: quello lo emette `Builder.read()`
e arriva PRIMA: l'assegnazione è in `TocTreeCollector.get_updated_docs`, che
pende da `env-get-updated`, emesso più tardi da `env.check_dependents()`.
Agganciata a `env-updated`, la funzione trovava la tabella ancora vuota e
riscriveva zero numeri senza lamentarsi di niente.

L'ordine dentro il capitolo non lo si ricostruisce: le figure sono GIÀ
numerate in ordine di lettura da 1 a N, quindi basta ordinarle per il numero
che avevano. Vale per ogni tipo numerato separatamente (figure, tabelle,
blocchi di codice), ciascuno col suo contatore, come si aspetta chi legge.

Le pagine che non stanno in un capitolo (la prefazione, la bibliografia, gli
aggiornamenti, la copertina) non hanno un numero da anteporre e tengono la
numerazione progressiva che Sphinx ha dato loro.

Si registra da `_config.yml`, insieme alle altre:

    sphinx:
      local_extensions:
        pt_figure: _ext
"""

import pathlib

import yaml
from sphinx.util import logging

logger = logging.getLogger(__name__)


def capitoli_dal_toc(percorso_toc: pathlib.Path) -> dict[str, int]:
    """Cartella del capitolo -> suo numero, nell'ordine del `_toc.yml`.

    Stesso criterio di `conta_capitoli()` in `pt_conteggi.py`: è capitolo una
    voce di primo livello che sta dentro una cartella. Se i due divergono, i
    numeri delle figure e quelli dell'indice divergono con loro.
    """
    dati = yaml.safe_load(percorso_toc.read_text(encoding="utf-8"))
    voci = [c for parte in dati.get("parts", []) or []
            for c in parte.get("chapters", []) or []]
    if not voci:                       # forma piatta, senza parti
        voci = dati.get("chapters", []) or []

    numeri: dict[str, int] = {}
    for voce in voci:
        file = str(voce.get("file", ""))
        if "/" not in file:            # prefazione, references, aggiornamenti
            continue
        cartella = file.split("/", 1)[0]
        if cartella not in numeri:
            numeri[cartella] = len(numeri) + 1
    return numeri


def rinumera(app, env) -> list[str]:
    """Riscrive `env.toc_fignumbers` come (capitolo, progressivo).

    Torna la lista (vuota) dei documenti da rileggere: `env-get-updated`
    raccoglie i valori di ritorno con un `extend`, e un `None` lo farebbe
    esplodere.
    """
    percorso_toc = pathlib.Path(app.srcdir) / "_toc.yml"
    if not percorso_toc.is_file():
        logger.warning("pt_figure: _toc.yml non trovato, numeri invariati")
        return []

    numeri = capitoli_dal_toc(percorso_toc)
    if not numeri:
        logger.warning("pt_figure: nessun capitolo nel toc, numeri invariati")
        return []

    # (capitolo, tipo) -> [(numero vecchio, docname, id)], per rimetterli in fila
    raccolta: dict[tuple[int, str], list] = {}
    for docname, tipi in env.toc_fignumbers.items():
        cartella = docname.split("/", 1)[0] if "/" in docname else ""
        capitolo = numeri.get(cartella)
        if capitolo is None:
            continue                   # fuori dai capitoli: si lascia com'è
        for tipo, figure in tipi.items():
            for figid, numero in figure.items():
                raccolta.setdefault((capitolo, tipo), []).append(
                    (tuple(numero), docname, figid))

    for (capitolo, tipo), voci in raccolta.items():
        voci.sort()                    # l'ordine di lettura è quello che avevano
        for progressivo, (_, docname, figid) in enumerate(voci, start=1):
            env.toc_fignumbers[docname][tipo][figid] = (capitolo, progressivo)

    quante = sum(len(v) for v in raccolta.values())
    logger.info("pt_figure: %d numeri riscritti per capitolo, in %d capitoli",
                quante, len({c for c, _ in raccolta}))
    return []


def setup(app):
    # priorità 900: dopo il TocTreeCollector, che è registrato col default
    # (500) ed è quello che i numeri li assegna. Prima, non ci sarebbe ancora
    # niente da riscrivere.
    app.connect("env-get-updated", rinumera, priority=900)
    return {"version": "1.0",
            "parallel_read_safe": True,
            "parallel_write_safe": True}
