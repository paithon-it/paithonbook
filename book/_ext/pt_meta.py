"""Una descrizione diversa per ogni pagina, presa dal testo della pagina.

Prima di questa estensione tutte le 160 pagine del libro dichiaravano lo stesso
`<meta name="description">`, lo stesso `og:title` e lo stesso `og:url` (quello
della landing). Per un motore di ricerca, per un assistente che cita e per
l'anteprima di un link condiviso quella riga *e'* la pagina: identica su tutte
significa nessuna informazione su nessuna.

La descrizione non si scrive a mano una per pagina (160 righe da tenere in
pari): si prende il **primo paragrafo**, che nel libro e' sempre l'aggancio
della sezione, ed e' scritto per spiegare di cosa si parla. Il valore finisce
nel contesto del template come `pt_descrizione`, e lo usa
`_static/head_custom.html`.

La landing non passa da qui: apre con l'istruzione per l'interruttore dei
livelli, che come descrizione non direbbe niente, e il template ha per lei la
riga curata del libro.

Regola di questo file: **non deve poter rompere la build**. Una descrizione
mancante e' un peccato veniale (il template ha il ripiego), un `raise` dentro
`html-page-context` e' un libro che non si pubblica. Per questo tutto sta in un
try/except che al massimo perde la descrizione di una pagina.
"""

from __future__ import annotations

import logging

from docutils import nodes

logger = logging.getLogger(__name__)

LIMITE = 165          # oltre, Google taglia comunque
MINIMO = 40           # sotto, non e' un paragrafo: e' una didascalia o un'etichetta


def taglia(testo: str) -> str:
    testo = " ".join(testo.split())
    if len(testo) <= LIMITE:
        return testo
    fine = testo.rfind(". ", 0, LIMITE)
    if fine > LIMITE // 2:
        return testo[:fine + 1]
    return testo[:LIMITE - 1].rsplit(" ", 1)[0] + "…"


def primo_paragrafo(doctree) -> str:
    """Il primo paragrafo abbastanza lungo da essere prosa.

    Le figure e le tabelle sono `caption`/`title`, non `paragraph`, quindi non
    vengono scambiate per l'apertura; le tab dei due livelli vengono dopo
    l'aggancio, che nel libro sta sempre fuori.
    """
    trova = getattr(doctree, "findall", None) or doctree.traverse
    for paragrafo in trova(nodes.paragraph):
        if isinstance(paragrafo.parent, (nodes.caption, nodes.entry)):
            continue
        testo = " ".join(paragrafo.astext().split())
        if len(testo) >= MINIMO:
            return taglia(testo)
    return ""


def aggiungi_descrizione(app, pagename, templatename, context, doctree):
    try:
        context["pt_descrizione"] = primo_paragrafo(doctree) if doctree else ""
    except Exception as errore:                      # mai fermare la build
        logger.warning("pt_meta: niente descrizione per %s (%s)", pagename, errore)
        context["pt_descrizione"] = ""


def setup(app):
    app.connect("html-page-context", aggiungi_descrizione)
    return {"version": "1.0", "parallel_read_safe": True,
            "parallel_write_safe": True}
