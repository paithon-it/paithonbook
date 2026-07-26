"""Quanti capitoli ha il libro: si conta una volta sola, e la conta il `_toc.yml`.

Il numero dei capitoli era scritto a lettere in due punti di prosa — la landing
(«Trentuno capitoli, dall'alfabeto…») e le conclusioni («Dopo trenta
capitoli…») — e a ogni capitolo nuovo restava indietro in silenzio: un numero
in mezzo a una frase non lo controlla nessuno, né la build né il validatore.
Qui la fonte diventa una sola, l'indice, e il testo la interpola con le
sostituzioni MyST (attive per default in Jupyter Book, `parse.
myst_enable_extensions`):

    {{ n_capitoli }}                   ->  31
    {{ n_capitoli_lettere }}           ->  trentuno
    {{ n_capitoli_meno_uno_lettere }}  ->  trenta

L'ultima è quella che serve alle conclusioni, che sono l'ultimo capitolo e
quindi ne hanno davanti uno in meno.

Si registra da `_config.yml`:

    sphinx:
      local_extensions:
        pt_conteggi: _ext

Conta le voci di primo livello del toc che stanno in una cartella
(`Python/overview.md`): `references.md` non è un capitolo e resta fuori,
esattamente come nella griglia della landing.
"""

import pathlib

import yaml
from sphinx.util import logging

logger = logging.getLogger(__name__)

UNITA = ("zero", "uno", "due", "tre", "quattro",
         "cinque", "sei", "sette", "otto", "nove")
DIECI = ("dieci", "undici", "dodici", "tredici", "quattordici",
         "quindici", "sedici", "diciassette", "diciotto", "diciannove")
DECINE = {2: "venti", 3: "trenta", 4: "quaranta", 5: "cinquanta",
          6: "sessanta", 7: "settanta", 8: "ottanta", 9: "novanta"}


def in_lettere(n: int) -> str:
    """Il numero scritto in italiano: 31 -> «trentuno», 23 -> «ventitré».

    Le decine perdono la vocale finale davanti a «uno» e «otto» (ventuno,
    trentotto) e «tre» prende l'accento quando non sta da solo (ventitré).
    """
    if n < 0:
        return str(n)
    if n < 10:
        return UNITA[n]
    if n < 20:
        return DIECI[n - 10]
    if n < 100:
        decina, unita = divmod(n, 10)
        parola = DECINE[decina]
        if unita == 0:
            return parola
        if unita in (1, 8):          # ventuno, trentotto
            parola = parola[:-1]
        return parola + ("tré" if unita == 3 else UNITA[unita])
    return str(n)                    # oltre il centinaio, meglio la cifra


def conta_capitoli(percorso_toc: pathlib.Path) -> int:
    """I capitoli del `_toc.yml`: le voci di primo livello dentro una cartella."""
    dati = yaml.safe_load(percorso_toc.read_text(encoding="utf-8"))
    voci = [c for parte in dati.get("parts", []) or []
            for c in parte.get("chapters", []) or []]
    if not voci:                     # forma piatta, senza parti
        voci = dati.get("chapters", []) or []
    return sum(1 for c in voci if "/" in str(c.get("file", "")))


def aggiungi_sostituzioni(app, config):
    percorso_toc = pathlib.Path(app.srcdir) / "_toc.yml"
    try:
        n = conta_capitoli(percorso_toc)
    except Exception as errore:      # noqa: BLE001 — un conteggio non fa cadere una build
        logger.warning("pt_conteggi: non riesco a contare i capitoli in %s (%s)",
                       percorso_toc, errore)
        return
    if not n:
        logger.warning("pt_conteggi: zero capitoli letti da %s", percorso_toc)
        return

    valori = {
        "n_capitoli": n,
        "n_capitoli_lettere": in_lettere(n),
        "n_capitoli_meno_uno": n - 1,
        "n_capitoli_meno_uno_lettere": in_lettere(n - 1),
    }
    # Le sostituzioni scritte a mano nel `_config.yml`, se un giorno ce ne
    # saranno, vincono: qui si aggiunge, non si sovrascrive il file.
    esistenti = dict(getattr(config, "myst_substitutions", None) or {})
    valori.update(esistenti)
    config.myst_substitutions = valori
    logger.info("pt_conteggi: %d capitoli (%s)", n, valori["n_capitoli_lettere"])


def setup(app):
    # priorità alta: dopo che myst_parser ha registrato `myst_substitutions`.
    app.connect("config-inited", aggiungi_sostituzioni, priority=800)
    return {"version": "1.0",
            "parallel_read_safe": True,
            "parallel_write_safe": True}
