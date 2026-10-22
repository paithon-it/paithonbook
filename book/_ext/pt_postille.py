r"""Le postille: una riga accanto a un capoverso del testo comune.

La postilla dice di che cosa parla il capoverso che accompagna. Sta in un file
accanto alla pagina (`pagina.postille.yml`; il formato e la definizione di
capoverso sono in `pt_postille_testo.py`) e si aggancia al capoverso che ha la
stessa impronta: se il capoverso cambia, l'impronta non combacia piu' e la
postilla non compare, ne' sul sito ne' in stampa, finche' qualcuno non la
rilegge e la riconferma.

Sul sito la postilla diventa un `<aside>` nascosto subito dopo il capoverso, e
la apre `_static/pt-postille.js`. Il testo sta in un attributo del nodo e non
in un nodo di testo: l'indice della ricerca legge i nodi di testo, e anche
l'HTML grezzo tolti i tag, mentre una ricerca deve portare a un testo che si
vede.

In stampa la postilla diventa `\ptPostilla{impronta}{testo}` in testa al
capoverso e `\ptFineCapoverso{impronta}` in fondo. La nota a margine la compone
`_stampa/ptpostille.sty`, che scrive anche dove l'ha messa, perche' la build
del PDF possa controllare che stia accanto al suo capoverso.

Il capoverso giusto lo dice il rilevatore sul sorgente, non il doctree: quando
questa estensione gira, i blocchi `{only}` sono gia' stati sciolti e i loro
capoversi sembrano testo comune. Il doctree serve a trovare il nodo alla riga
che il rilevatore indica, e a controllare che stia davvero nel testo comune.
La sorgente si prende dal capoverso e non dal documento: per la stampa Sphinx
fonde tutte le pagine in un albero solo, il documento ha la sorgente della
prima, e due pagine con un capoverso alla stessa riga si scambierebbero le
postille.

Il file delle postille e' una dipendenza della sua pagina: cambiato o
cancellato, la pagina si ricostruisce. Il primo file di una pagina gia'
costruita non lo dichiara nessuno, e lo cerca `nuove_postille` a ogni build.
"""

from __future__ import annotations

import pathlib

from docutils import nodes
from sphinx.transforms.post_transforms import SphinxPostTransform
from sphinx.util import logging
from sphinx.util.osutil import relative_uri
from sphinx.util.texescape import escape

import pt_postille_testo as testo_postille

logger = logging.getLogger(__name__)

# Il segno che apre la postilla: il marchio del brand, servito da `_static`.
SEGNO = "_static/brand/logos/paithon-mark.svg"

# I genitori di un capoverso del testo comune, nel doctree.
COMUNE = (nodes.section, nodes.document)


class postilla(nodes.General, nodes.Element):
    """La postilla sul sito, col testo negli attributi."""


def visita_postilla_html(self, node: postilla) -> None:
    pagina = self.builder.get_target_uri(self.builder.current_docname)
    segno = relative_uri(pagina, SEGNO)
    self.body.append(
        f'<aside class="pt-postilla" id="pt-postilla-{node["impronta"]}" '
        f'data-segno="{self.attval(segno)}" hidden>'
        f'{self.encode(node["testo"])}</aside>\n')
    raise nodes.SkipNode


def postille_accese(sorgente: pathlib.Path) -> dict[int, testo_postille.Voce]:
    """Le postille accese di una pagina, per riga del loro capoverso."""
    file = testo_postille.file_delle_postille(sorgente)
    if not file.is_file():
        return {}
    try:
        voci = testo_postille.leggi_postille(file)
    except ValueError as errore:
        logger.warning("pt_postille: %s: %s", file.name, errore)
        return {}
    capoversi = testo_postille.capoversi(sorgente.read_text(encoding="utf-8"))
    accese, spente = testo_postille.abbina(capoversi, voci)
    if spente:
        logger.info("pt_postille: %s, %d postille spente", file.name, len(spente))
    return accese


def capoversi_con_postilla(document: nodes.document):
    """[(nodo del capoverso, voce)] per ogni postilla accesa nel documento."""
    per_sorgente: dict[str, dict[int, nodes.paragraph]] = {}
    for capoverso in document.findall(nodes.paragraph):
        if capoverso.line and capoverso.source and isinstance(capoverso.parent, COMUNE):
            per_sorgente.setdefault(capoverso.source, {})[capoverso.line] = capoverso
    coppie = []
    for sorgente, nodi in sorted(per_sorgente.items()):
        for riga, voce in sorted(postille_accese(pathlib.Path(sorgente)).items()):
            capoverso = nodi.get(riga)
            if capoverso is None:
                logger.warning("pt_postille: %s, riga %d: il doctree non ha qui "
                               "un capoverso del testo comune", sorgente, riga)
                continue
            coppie.append((capoverso, voce))
    return coppie


class PostilleSulSito(SphinxPostTransform):
    """Un `<aside>` nascosto dopo ogni capoverso con una postilla accesa."""

    default_priority = 800
    formats = ("html",)

    def run(self, **kwargs) -> None:
        for capoverso, voce in capoversi_con_postilla(self.document):
            capoverso["classes"].append("pt-con-postilla")
            capoverso.parent.insert(
                capoverso.parent.index(capoverso) + 1,
                postilla(testo=testo_postille.tipografia(voce.postilla),
                         impronta=voce.impronta))


class PostilleInStampa(SphinxPostTransform):
    r"""`\ptPostilla` in testa al capoverso e `\ptFineCapoverso` in fondo."""

    default_priority = 800
    formats = ("latex",)

    def run(self, **kwargs) -> None:
        motore = self.config.latex_engine
        for capoverso, voce in capoversi_con_postilla(self.document):
            testo = escape(testo_postille.tipografia(voce.postilla), motore)
            capoverso.insert(0, nodes.raw(
                "", f"\\ptPostilla{{{voce.impronta}}}{{{testo}}}", format="latex"))
            capoverso.append(nodes.raw(
                "", f"\\ptFineCapoverso{{{voce.impronta}}}", format="latex"))


def dichiara_il_file(app, doctree) -> None:
    """Il file delle postille e' una dipendenza della sua pagina."""
    sorgente = pathlib.Path(app.env.doc2path(app.env.docname))
    file = testo_postille.file_delle_postille(sorgente)
    if file.is_file():
        app.env.note_dependency(str(file))


def nuove_postille(app, env, added, changed, removed) -> list[str]:
    """Le pagine il cui file di postille e' nato dopo l'ultima lettura."""
    nuove = []
    for docname in sorted(env.found_docs):
        if docname in added or docname in changed:
            continue
        file = testo_postille.file_delle_postille(pathlib.Path(env.doc2path(docname)))
        if file.is_file() and str(file) not in env.dependencies.get(docname, ()):
            nuove.append(docname)
    return nuove


def setup(app):
    app.add_node(postilla, html=(visita_postilla_html, None))
    app.add_post_transform(PostilleSulSito)
    app.add_post_transform(PostilleInStampa)
    app.connect("doctree-read", dichiara_il_file)
    app.connect("env-get-outdated", nuove_postille)
    return {"version": "1.0",
            "parallel_read_safe": True,
            "parallel_write_safe": True}
