"""La configurazione di Sphinx del libro, ricavata da `_config.yml`.

Il libro si costruisce con `sphinx-build` diretto. La traduzione da
`_config.yml` a configurazione di Sphinx e' quella di jupyter-book 1, ridotta
alle chiavi che il libro usa: cosi' `_config.yml` resta la fonte, e gli
attrezzi che lo leggono o lo scrivono (`scripts/genera-notebook.py`,
`scripts/genera-radice.py`, `strumenti/compila_skill.py`) non cambiano.

La traduzione ha quattro parti, nell'ordine in cui jupyter-book le applicava:

1. i valori di partenza, che valevano per ogni libro;
2. le chiavi di primo livello di `_config.yml` (titolo, autore, `html:`,
   `execute:`, la bibliografia);
3. `sphinx.config`, che **sostituisce** i valori con lo stesso nome invece di
   fonderli (`recursive_update: false`): `html_theme_options` di
   `_config.yml` prende il posto dell'intero dizionario del tema;
4. le estensioni (`extra_extensions`, poi quelle locali di `_ext`, poi la
   bibliografia e `sphinx_jupyterbook_latex`), e i fogli di stile e gli
   script di `_static`, che jupyter-book aggiungeva **tutti** da se',
   sottocartelle comprese: e' cosi' che arrivano nella pagina
   `brand/tokens.css`, `brand/fonts.css`, `brand/aurora.css`,
   `brand/motion/motion.css` e `brand/print.js`, che `_config.yml` non nomina.
"""
import sys
from pathlib import Path

import yaml

QUI = Path(__file__).resolve().parent
_LIBRO = yaml.safe_load((QUI / "_config.yml").read_text(encoding="utf-8"))

# 1. I valori di partenza.
extensions = [
    "sphinx_togglebutton",
    "sphinx_copybutton",
    "myst_nb",
    "sphinx_thebe",
    "sphinx_comments",
    "sphinx_external_toc",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "sphinx_book_theme",
]
pygments_style = "sphinx"
html_theme = "sphinx_book_theme"
html_sourcelink_suffix = ""
numfig = True
suppress_warnings = ["myst.domains"]
exclude_patterns = ["**.ipynb_checkpoints", ".DS_Store", "Thumbs.db", "_build"]
myst_enable_extensions = ["colon_fence", "dollarmath", "linkify", "substitution",
                          "tasklist"]
myst_url_schemes = ["mailto", "http", "https"]
comments_config = {"hypothesis": False, "utterances": False}
nb_execution_allow_errors = False
nb_execution_cache_path = ""
nb_execution_excludepatterns = []
nb_execution_in_temp = False
nb_execution_mode = "auto"
nb_execution_timeout = 30
nb_output_stderr = "show"
external_toc_path = "_toc.yml"
html_static_path = ["_static"]
latex_engine = "pdflatex"
use_jupyterbook_latex = True
use_multitoc_numbering = True

# 2. Le chiavi di primo livello.
html_title = _LIBRO.get("title") or ""
author = _LIBRO.get("author") or ""
copyright = _LIBRO.get("copyright") or ""
html_logo = _LIBRO.get("logo") or ""
external_toc_exclude_missing = _LIBRO.get("only_build_toc_files", False)
_html = _LIBRO.get("html") or {}
html_favicon = _html.get("favicon", "")
html_baseurl = _html.get("baseurl", "")
use_multitoc_numbering = _html.get("use_multitoc_numbering",
                                   use_multitoc_numbering)
_esegui = _LIBRO.get("execute") or {}
nb_execution_mode = _esegui.get("execute_notebooks", nb_execution_mode)
if nb_execution_mode is False:            # in YAML `off` e' un booleano
    nb_execution_mode = "off"
if "bibtex_bibfiles" in _LIBRO:
    bibtex_bibfiles = _LIBRO["bibtex_bibfiles"]

# 3. sphinx.config: sostituisce, non fonde.
_sphinx = _LIBRO.get("sphinx") or {}
globals().update(_sphinx.get("config") or {})

# 4. Le estensioni, e i fogli di stile e gli script di _static.
extensions += list(_sphinx.get("extra_extensions") or [])
for _nome, _cartella in (_sphinx.get("local_extensions") or {}).items():
    _percorso = (QUI / _cartella).resolve().as_posix()
    if _percorso not in sys.path:
        sys.path.append(_percorso)
    extensions.append(_nome)
if "bibtex_bibfiles" in _LIBRO:
    extensions.append("sphinxcontrib.bibtex")
if use_jupyterbook_latex:
    extensions.append("sphinx_jupyterbook_latex")
if use_multitoc_numbering:
    extensions.append("sphinx_multitoc_numbering")

html_css_files = list(globals().get("html_css_files", []))
html_js_files = list(globals().get("html_js_files", []))
for _statica in html_static_path:
    _radice = QUI / _statica
    html_css_files += [p.relative_to(_radice).as_posix()
                       for p in _radice.rglob("*.css")]
    html_js_files += [p.relative_to(_radice).as_posix()
                      for p in _radice.rglob("*.js")]

# sphinx-thebe, nella versione lite, a fine build copia nel sito i file della
# sorgente che i suoi `exclude_patterns` non escludono (servono al Python che
# gira nel browser): questo file non e' fra quelli che servono.
if "thebe_config" in globals():
    thebe_config = dict(thebe_config)
    thebe_config["exclude_patterns"] = (
        list(thebe_config.get("exclude_patterns", [])) + ["conf.py"])


def setup(app):
    """Le parole fatte di sole cifre restano nell'indice della ricerca.

    Sphinx 9 le scarta (`word.isdigit()` in `SearchLanguage.word_filter`;
    la 7.4 le teneva), e nel libro sono anni e numeri che il lettore cerca
    davvero: «1952», «2017», «16000». Qui c'e' la lingua italiana di Sphinx
    con il filtro che tiene le cifre e scarta solo le parole vuote.
    """
    from sphinx.search.it import SearchItalian

    class Italiano(SearchItalian):
        def word_filter(self, word: str) -> bool:
            return word not in self.stopwords

    app.add_search_language(Italiano)
