"""Una descrizione diversa per ogni pagina, presa dal testo della pagina.

Senza, tutte le pagine del libro dichiarerebbero lo stesso
`<meta name="description">`, lo stesso `og:title` e lo stesso `og:url` (quello
della landing). Per un motore di ricerca, per un assistente che cita e per
l'anteprima di un link condiviso quella riga *e'* la pagina: identica su tutte
significa nessuna informazione su nessuna.

La descrizione non si scrive a mano una per pagina (sarebbe una riga per
pagina da tenere in pari): si prende il **primo paragrafo**, che nel libro e'
sempre l'aggancio della sezione, ed e' scritto per spiegare di cosa si parla.
Il valore finisce nel contesto del template come `pt_descrizione`, e lo usa
`_static/head_custom.html`.

La landing non passa da qui: apre con l'istruzione per l'interruttore dei
livelli, che come descrizione non direbbe niente, e il template ha per lei la
riga curata del libro.

Una pagina che si cerca molto puo' dire di se' qualcosa di piu' preciso, nel
front matter:

    ---
    titolo_ricerca: "Machine learning: cos'e' e come funziona"
    descrizione: "Che cos'e' il machine learning, ..."
    ---

`descrizione` prende il posto del primo paragrafo; `titolo_ricerca` finisce
nel contesto come `pt_titolo_ricerca` e lo usano `<title>` (in
`_templates/layout.html`) e `og:title`. Il titolo nel testo, l'indice e il
PDF restano quelli della pagina: e' il modo in cui la pagina si presenta in un
elenco di risultati, non un altro nome del capitolo.

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


def dal_front_matter(context, chiave: str) -> str:
    """Il valore di `chiave` nel front matter della pagina, su una riga."""
    meta = context.get("meta") or {}
    valore = meta.get(chiave, "") if hasattr(meta, "get") else ""
    return " ".join(str(valore).split()) if valore else ""


def aggiungi_descrizione(app, pagename, templatename, context, doctree):
    context["pt_titolo_ricerca"] = ""
    try:
        context["pt_titolo_ricerca"] = dal_front_matter(context, "titolo_ricerca")
        scritta = dal_front_matter(context, "descrizione")
        if scritta:
            context["pt_descrizione"] = taglia(scritta)
        else:
            context["pt_descrizione"] = primo_paragrafo(doctree) if doctree else ""
    except Exception as errore:                      # mai fermare la build
        logger.warning("pt_meta: niente descrizione per %s (%s)", pagename, errore)
        context["pt_descrizione"] = ""


def setup(app):
    app.connect("html-page-context", aggiungi_descrizione)
    return {"version": "1.1", "parallel_read_safe": True,
            "parallel_write_safe": True}
