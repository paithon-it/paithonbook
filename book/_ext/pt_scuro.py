"""In tema scuro ogni figura mostra la propria resa scura, e non una cartolina.

Le figure del libro stanno sulla palette chiara della marca. Il tema
(pydata-sphinx-theme) in scuro tratta ogni `img` del contenuto come una
fotografia:

    html[data-theme=dark] img:not(.only-dark,.dark-light)
        { filter: brightness(.8) contrast(1.2) }
    html[data-theme=dark] .bd-content img:not(.only-dark,.dark-light)
        { background-color: #fff; border-radius: .25rem }

cioe' un rettangolo bianco con i colori spenti in mezzo a una pagina nera. E'
lo stesso guasto che il segno della prefazione aveva, ed e' scritto nella
regola del libro: «le SVG devono leggersi anche in dark mode: niente fondo
bianco pieno».

## Perche' qui e non nelle pagine

La coppia chiaro/scuro il tema la sa gia' fare: due `<img>`, una `only-light`
e una `only-dark`, e mostra quella giusta. Scriverla a mano vorrebbe dire
riaprire trecentoquarantatre direttive `{figure}` nei capitoli, cioe' mettere
una decisione di presentazione dentro il testo, e farla dimenticare alla prima
figura nuova. Qui la coppia si forma in build, e una figura nuova ce l'ha per
il fatto di esistere.

La resa scura non si inventa, e viene da due mani diverse perche' le figure
del libro sono di due materie. Gli SVG li ricolora
`scripts/genera-figure-scure.py`, che sostituisce i colori uno per uno; le
GIF di Manim sono raster e non si ricolorano, quindi si **ri-renderizzano**
dallo stesso sorgente con `PAITHON_TEMA=scuro`, ed e' quello che fa
`animazioni/scure.py`. Le due scrivono nella stessa cartella,
`book/figures/scure/`, e ciascuna sa dire se la propria e' rimasta indietro.

Quando il file non c'e', questa estensione non fa niente e la figura resta
com'era: e' il caso del segno della marca, che e' palette-locked, e di
qualunque figura la cui resa scura non sia stata ancora scritta.

## Perche' in LETTURA, e non fra i post-transform come le altre

Provata prima come post-transform, che sarebbe il posto naturale, e non
funziona: l'immagine finiva in pagina con l'indirizzo del sorgente
(`figures/scure/nome.svg` invece di `../_images/nome.svg`) e il file non
veniva nemmeno copiato. La ragione sta in `Builder.post_process_images`, che
riscrive gli indirizzi e raccoglie i file da copiare **solo** per le immagini
gia' presenti in `env.images`, e quel registro lo riempie `ImageCollector` in
lettura. Un nodo nato dopo non c'e', e viene saltato in silenzio: nessun
avviso, build a codice 0, figura rotta in pagina.

Quindi la coppia si forma in lettura, prima che i collector guardino
l'albero, e la seconda immagine viene raccolta, copiata e riscritta come
tutte le altre.

## E allora la stampa

Formandosi in lettura, la coppia sta anche nell'albero che legge il builder
LaTeX. `ViaLaScuraInStampa` la toglie prima di tutti gli altri post-transform
della stampa, quindi il PDF non vede mai la seconda immagine: ne' il
convertitore di `pt_stampa.py`, che altrimenti la trasformerebbe in PDF per
niente, ne' la pagina.

Si registra da `_config.yml`, insieme alle altre:

    sphinx:
      local_extensions:
        pt_scuro: _ext
"""

import pathlib
import posixpath

from docutils import nodes
from sphinx.transforms import SphinxTransform
from sphinx.transforms.post_transforms import SphinxPostTransform
from sphinx.util import logging

logger = logging.getLogger(__name__)

# La cartella delle rese scure, dentro `book/figures/`.
CARTELLA = "scure"

# Le estensioni che possono avere una gemella. Non e' l'elenco di cio' che
# Sphinx sa mostrare: e' l'elenco di cio' per cui esiste un generatore della
# resa scura. Una `.png` scritta a mano non ce l'ha, e accoppiarla vorrebbe
# dire cercare un file che nessuno scrive mai.
ACCOPPIABILI = (".svg", ".gif")

# Le classi con cui una figura dichiara di sapersela gia' cavare da sola: il
# segno della marca porta `dark-light` (palette-locked), la copertina della
# landing ha gia' la sua coppia scritta a mano. Chi le porta non si tocca.
GIA_SUE = frozenset({"only-light", "only-dark", "dark-light"})


def indirizzo_scuro(uri: str) -> str:
    """`../figures/nome.svg` -> `../figures/scure/nome.svg`.

    Si costruisce sul posto invece di scrivere un percorso assoluto: l'indirizzo
    di una figura e' relativo al file che la richiama, e le pagine del libro
    stanno a profondita' diverse.
    """
    cartella, nome = posixpath.split(uri)
    return posixpath.join(cartella, CARTELLA, nome)


class AccoppiaLaScura(SphinxTransform):
    """Affianca a ogni figura la sua resa scura, quando sul disco c'e'."""

    # Prima che `doctree-read` chiami i collector: e' l'unico momento in cui
    # aggiungere un'immagine serve a qualcosa.
    default_priority = 500

    def apply(self, **kwargs):
        accoppiate = 0
        for immagine in list(self.document.findall(nodes.image)):
            uri = immagine.get("uri", "")
            if not uri.endswith(ACCOPPIABILI):
                continue
            if GIA_SUE & set(immagine.get("classes") or []):
                continue

            scuro = indirizzo_scuro(uri)
            _, assoluto = self.env.relfn2path(scuro, self.env.docname)
            if not pathlib.Path(assoluto).is_file():
                continue

            genitore = immagine.parent
            if genitore is None:
                continue

            copia = nodes.image(uri=scuro, alt=immagine.get("alt", ""),
                                classes=["only-dark"])
            # La larghezza dichiarata nella direttiva: senza, la resa scura
            # entrerebbe con la sua misura naturale e la pagina ballerebbe
            # cambiando tema.
            for attributo in ("width", "height", "scale", "align"):
                if attributo in immagine:
                    copia[attributo] = immagine[attributo]

            immagine["classes"] = list(immagine.get("classes") or [])
            immagine["classes"].append("only-light")
            genitore.insert(genitore.index(immagine) + 1, copia)
            accoppiate += 1

        if accoppiate:
            logger.debug("pt_scuro: %d figure accoppiate in %s",
                         accoppiate, self.env.docname)


class ViaLaScuraInStampa(SphinxPostTransform):
    """In stampa la seconda immagine non esiste: la carta non ha temi."""

    # prima di ogni altro post-transform della stampa
    default_priority = 150
    formats = ("latex",)

    def run(self, **kwargs):
        tolte = 0
        for immagine in list(self.document.findall(nodes.image)):
            classi = set(immagine.get("classes") or [])
            if "only-dark" in classi:
                immagine.parent.remove(immagine)
                tolte += 1
            elif "only-light" in classi:
                # la chiara resta, ma senza la classe: in stampa non c'e'
                # nessun tema da cui distinguersi.
                immagine["classes"] = [c for c in immagine["classes"]
                                       if c != "only-light"]
        if tolte:
            logger.debug("pt_scuro: %d rese scure tolte dalla stampa", tolte)
        return


def setup(app):
    app.add_transform(AccoppiaLaScura)
    app.add_post_transform(ViaLaScuraInStampa)
    return {"version": "1.0",
            "parallel_read_safe": True,
            "parallel_write_safe": True}
