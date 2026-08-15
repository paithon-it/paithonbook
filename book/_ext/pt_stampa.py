"""Quello che serve al libro per diventare un PDF, e solo a quello.

Ogni pezzo qui dentro si accende soltanto quando si costruisce con il builder
LaTeX: online non cambia niente, e non e' una promessa, e' come funziona
`ImageConverter`, che Sphinx interroga solo per le immagini che il builder in
uso non sa mangiare (l'HTML l'SVG lo sa mangiare benissimo).

## Le figure

Sphinx scrive `\\sphinxincludegraphics{nome.svg}` e LaTeX un SVG non lo sa
aprire. La via ufficiale e' un `ImageConverter`: si dichiara la regola
`image/svg+xml -> application/pdf` e Sphinx chiama `convert()` per ogni figura
che serve, gia' sapendo quali il builder digerisce.

Il convertitore e' **Chromium**, non CairoSVG e non ImageMagick, e la ragione
sta nelle figure stesse: 73 dei 311 SVG del libro portano un `<style>` con
selettori di classe, e tutti usano i font del brand. Un browser vero e'
l'unica cosa che li disegna come li vede chi legge online. Stampando in PDF
invece che in PNG il disegno resta **vettoriale**: pesa meno e non sgrana.

Il browser si apre una volta sola per tutta la build e si chiude alla fine.
Aprirne uno per figura sarebbe mezzo secondo per trecentoundici figure, ogni
volta che si ricostruisce.

## La cache

Le figure cambiano poco e la build si rilancia spesso. Ogni PDF convertito
resta in `_build/stampa/figure/` sotto l'impronta del contenuto dell'SVG: chi
ricostruisce dopo aver toccato una pagina paga una figura, non trecento. Per
buttarla via basta cancellare la cartella.
"""

import hashlib
import pathlib

from docutils import nodes
from sphinx.transforms.post_transforms import SphinxPostTransform
from sphinx.transforms.post_transforms.images import ImageConverter
from sphinx.util import logging

logger = logging.getLogger(__name__)

# Come l'etichetta di una tab diventa il nome di un ambiente LaTeX. Le due che
# contano sono i livelli di lettura; qualunque altra tab, se un giorno ce ne
# sara' una, finisce in un riquadro neutro.
LIVELLI = {"elementare": "pt-elementare", "superiore": "pt-superiore"}

# Le tre immagini con cui si sostituisce un'animazione, e quanto larga esce
# ciascuna. Tre a fianco su una riga: il resto e' aria fra l'una e l'altra.
LARGHEZZA_FERMO = "31%"

# I font del brand, per le etichette dentro le figure. Vanno dichiarate TUTTE
# le facce, non una per famiglia: le figure del libro chiedono il peso 600
# settantuno volte, il 700 sessantacinque e il corsivo quarantatre, e una
# faccia sola non li copre. Con una sola, Chromium ripiega su Liberation e
# DejaVu, e in una figura su tre le etichette escono con un altro carattere.
#
# Il peso 700 non c'e' e non e' una dimenticanza: il brand si ferma a 600, e
# l'URL di Google Fonts che usa il sito chiede 400, 500 e 600. Chiedendo 700,
# Chromium sceglie il 600, che e' esattamente quello che gia' succede online.
# Le figure devono somigliare a se stesse, non essere piu' belle in stampa.
#
# (famiglia, file, peso, stile)
FONT = [
    ("Fraunces", "fraunces-400.ttf", 400, "normal"),
    ("Fraunces", "fraunces-600.ttf", 600, "normal"),
    ("Fraunces", "fraunces-800.ttf", 800, "normal"),
    ("Fraunces", "fraunces-400italic.ttf", 400, "italic"),
    ("Fraunces", "fraunces-600italic.ttf", 600, "italic"),
    ("Inter", "inter-400.ttf", 400, "normal"),
    ("Inter", "inter-500.ttf", 500, "normal"),
    ("Inter", "inter-600.ttf", 600, "normal"),
    ("Inter", "inter-400italic.ttf", 400, "italic"),
    ("Inter", "inter-600italic.ttf", 600, "italic"),
    ("JetBrains Mono", "jetbrains-mono-400.ttf", 400, "normal"),
    ("JetBrains Mono", "jetbrains-mono-600.ttf", 600, "normal"),
    ("JetBrains Mono", "jetbrains-mono-400italic.ttf", 400, "italic"),
]

PAGINA = """<!doctype html><meta charset="utf-8">
<style>
{facce}
html,body{{margin:0;padding:0;background:transparent}}
svg{{display:block}}
@page{{size:{w}px {h}px;margin:0}}
</style>
{svg}
"""


class Browser:
    """Un Chromium solo, aperto pigramente e chiuso a fine build."""

    def __init__(self):
        self._playwright = None
        self._browser = None

    def pagina(self):
        if self._browser is None:
            from playwright.sync_api import sync_playwright
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch()
        return self._browser.new_page()

    def chiudi(self):
        if self._browser is not None:
            self._browser.close()
            self._playwright.stop()
            self._browser = self._playwright = None


BROWSER = Browser()


def facce_font(brand: pathlib.Path) -> str:
    """Le regole `@font-face` che puntano ai TTF del brand."""
    righe = []
    for famiglia, file, peso, stile in FONT:
        percorso = brand / "stampa" / "fonts" / file
        if percorso.exists():
            righe.append(f"@font-face{{font-family:'{famiglia}';"
                         f"font-weight:{peso};font-style:{stile};"
                         f"src:url('{percorso.as_uri()}')}}")
        else:
            logger.warning("pt_stampa: manca il font %s (submodule brand "
                           "aggiornato?)", percorso)
    return "\n".join(righe)


def misura(svg: str) -> tuple[float, float]:
    """Larghezza e altezza in px, dal viewBox o dagli attributi.

    Serve perche' la pagina che stampiamo deve essere della misura esatta del
    disegno: un foglio piu' grande lascerebbe un margine bianco attaccato alla
    figura, che in pagina si vede come una figura scentrata.
    """
    import re
    viewbox = re.search(r'viewBox="([-\d.eE\s]+)"', svg)
    if viewbox:
        pezzi = viewbox.group(1).split()
        if len(pezzi) == 4:
            return float(pezzi[2]), float(pezzi[3])
    larghezza = re.search(r'\swidth="([\d.]+)', svg)
    altezza = re.search(r'\sheight="([\d.]+)', svg)
    return (float(larghezza.group(1)) if larghezza else 800.0,
            float(altezza.group(1)) if altezza else 600.0)


class ConvertitoreChromium(ImageConverter):
    """SVG verso PDF vettoriale, con cache sul contenuto."""

    default_priority = 200
    conversion_rules = [("image/svg+xml", "application/pdf")]

    def is_available(self) -> bool:
        try:
            import playwright  # noqa: F401
        except ImportError:
            logger.warning("pt_stampa: playwright non c'e', le figure "
                           "resteranno SVG e LuaLaTeX non le aprira'. "
                           "pip3 install --user playwright && "
                           "python3 -m playwright install chromium")
            return False
        return True

    def convert(self, _from: str, _to: str) -> bool:
        origine = pathlib.Path(_from)
        svg = origine.read_text(encoding="utf-8")

        cache = pathlib.Path(self.app.outdir).parent / "figure"
        cache.mkdir(parents=True, exist_ok=True)
        impronta = hashlib.sha256(svg.encode("utf-8")).hexdigest()[:16]
        pronta = cache / f"{origine.stem}-{impronta}.pdf"

        if not pronta.exists():
            larghezza, altezza = misura(svg)
            brand = pathlib.Path(self.app.srcdir) / "_static" / "brand"
            temporanea = cache / f"{origine.stem}-{impronta}.html"
            temporanea.write_text(
                PAGINA.format(facce=facce_font(brand), svg=svg,
                              w=larghezza, h=altezza), encoding="utf-8")
            pagina = BROWSER.pagina()
            try:
                pagina.goto(temporanea.as_uri())
                pagina.pdf(path=str(pronta),
                           width=f"{larghezza}px", height=f"{altezza}px",
                           print_background=True,
                           margin={"top": "0", "bottom": "0",
                                   "left": "0", "right": "0"})
            finally:
                pagina.close()
                temporanea.unlink(missing_ok=True)

        pathlib.Path(_to).write_bytes(pronta.read_bytes())
        return True


class TabInLaTeX(SphinxPostTransform):
    """Da TabContainer a container con un nome che il .sty sa vestire.

    `sphinx-inline-tabs` non ha uno scrittore LaTeX, ma il suo
    `TabHtmlTransform` dichiara `formats = ["html"]`: in stampa l'albero resta
    nella forma originale, `TabContainer > [label, container.tab-content]`, e
    il contenuto ci arriva lo stesso, perche' `TabContainer` deriva da
    `nodes.container` e Sphinx risale la MRO cercando un visitatore. Manca
    l'etichetta, che vive in un nodo che in LaTeX nessuno sa scrivere.

    Poi fa tutto LaTeX: per ogni classe di un container Sphinx emette
    `\\begin{sphinxuseclass}{nome}`, che a sua volta apre l'ambiente
    `sphinxclassnome` **se qualcuno l'ha definito**
    (`sphinxlatexcontainers.sty`). Quindi qui si mette solo il nome, e tutto
    lo stile dei due livelli vive in `_stampa/paithon.sty`, dove si guarda e
    si cambia senza toccare il Python.
    """

    default_priority = 190          # prima del convertitore delle immagini
    formats = ("latex",)

    def run(self, **kwargs):
        try:
            from sphinx_inline_tabs._impl import TabContainer
        except ImportError:
            return

        for tab in list(self.document.findall(TabContainer)):
            etichetta, corpo = "", None
            for figlio in tab.children:
                if isinstance(figlio, nodes.label) and not etichetta:
                    etichetta = figlio.astext().strip()
                elif isinstance(figlio, nodes.container) and corpo is None:
                    corpo = figlio
            classe = LIVELLI.get(etichetta.lower(), "pt-tab")
            nuovo = nodes.container(
                "", *(corpo.children if corpo is not None else []),
                classes=[classe])
            tab.replace_self(nuovo)


# Pagine che online hanno senso e in un PDF no. Il registro delle versioni
# racconta la storia di un sito che cambia; in mano a chi ha scaricato un
# file, la versione e' una sola ed e' scritta nel colophon.
FUORI_STAMPA = ("aggiornamenti",)


class FuoriStampa(SphinxPostTransform):
    """Toglie dal PDF le pagine che in un PDF non dicono niente.

    Si svuota la sezione, titolo compreso: lasciarne il titolo darebbe un
    capitolo fantasma nell'indice, con dentro il nulla.
    """

    default_priority = 180
    formats = ("latex",)

    def run(self, **kwargs):
        for sezione in list(self.document.findall(nodes.section)):
            if sezione.parent is None or not sezione.source:
                continue
            # Il documento si chiede al file da cui il nodo viene, non a
            # `env.docname`: il builder LaTeX assembla un albero solo, e
            # quel campo dice il documento radice per tutto quanto.
            if (self.env.path2doc(sezione.source) or "") in FUORI_STAMPA:
                sezione.parent.remove(sezione)


# Il titolo che prende, in stampa, il materiale della landing. Online non ha
# bisogno di un titolo perche' e' la pagina di apertura del sito e si capisce
# da se'; in un libro un blocco di testo senza intestazione, prima di tutto e
# sotto la testatina del capitolo sbagliato, e' solo un blocco che galleggia.
TITOLO_APERTURA = "Come si legge questo libro"


class ApreLaPrefazione(SphinxPostTransform):
    """Mette la Prefazione per prima e il materiale della landing dopo.

    Il documento radice del `_toc.yml` e' `intro.md`, cioe' la pagina di
    apertura del sito, e il builder LaTeX emette il suo contenuto **prima di
    tutto**, senza titolo e sotto la testatina della prima parte. In un libro
    stampato apre il volume una prefazione, non una landing.

    Qui il contenuto della landing si stacca dalla testa del documento, si
    avvolge in una sezione con un titolo suo, e si riattacca subito dopo il
    capitolo della prefazione. Sono due gesti su un albero gia' assemblato,
    non una riscrittura del `_toc.yml`: online l'ordine e' quello giusto e non
    va toccato.
    """

    # DOPO `LatexRootDocPostTransforms` di `sphinx-jupyterbook-latex`, che
    # gira a 700 ed e' quella che sistema il documento radice. Con una
    # priorita' piu' bassa si sposterebbero nodi su un albero che quella poi
    # rimescola, e non succede niente: e' andata cosi' al primo tentativo.
    default_priority = 750
    formats = ("latex",)

    def _pagina(self, nodo) -> str:
        return (self.env.path2doc(nodo.source) or "") if nodo.source else ""

    def run(self, **kwargs):
        # Due trappole, tutte e due scoperte solo guardando il PDF.
        #
        # La prima: la prefazione non sta a una profondita' fissa. L'albero e'
        # `document > compound > start_of_file > section`, e indovinare il
        # livello falliva in silenzio. Si cerca ovunque sia.
        #
        # La seconda, peggiore: **la sezione di `intro` non si puo' togliere
        # dal documento**. E' lei a fare il livello del titolo, e portandola
        # via tutti gli altri salgono di un grado: i capitoli diventano parti.
        # Quindi si sposta il CONTENUTO e si lascia il contenitore vuoto.
        landing = next(
            (n for n in self.document.children
             if isinstance(n, nodes.section) and self._pagina(n) == "intro"),
            None)
        prefazione = next(
            (n for n in self.document.findall(nodes.section)
             if self._pagina(n) == "prefazione"), None)
        if landing is None or prefazione is None or prefazione.parent is None:
            return

        contenuto = [n for n in landing.children if not isinstance(n, nodes.title)]
        if not contenuto:
            return
        for nodo in contenuto:
            landing.remove(nodo)

        sezione = nodes.section(ids=["pt-come-si-legge"])
        sezione["names"] = ["come si legge questo libro"]
        sezione += nodes.title("", TITOLO_APERTURA)
        sezione.extend(contenuto)
        sezione.source, sezione.line = contenuto[0].source, contenuto[0].line

        genitore = prefazione.parent
        genitore.insert(genitore.index(prefazione) + 1, sezione)
        logger.info("pt_stampa: «%s» e' passata dopo la prefazione",
                    TITOLO_APERTURA)


class ViaISeparatori(SphinxPostTransform):
    """Toglie le righe di separazione, che in stampa non separano niente.

    Sono quattro, tutte nella landing, e online fanno il loro mestiere:
    spezzano una pagina lunga in blocchi. In stampa i blocchi che separavano
    sono per meta' HTML che sparisce, e restano quattro filetti impilati uno
    sull'altro con niente in mezzo.
    """

    default_priority = 183
    formats = ("latex",)

    def run(self, **kwargs):
        for riga in list(self.document.findall(nodes.transition)):
            riga.parent.remove(riga)


def si_muove(file: pathlib.Path) -> bool:
    """Una GIF, o un SVG che porta un @keyframes.

    Il controllo e' sul contenuto e non sul nome perche' degli SVG del libro
    se ne muovono 26 su 311, e nel nome non c'e' niente che lo dica.
    """
    if file.suffix == ".gif":
        return True
    if file.suffix != ".svg" or not file.exists():
        return False
    return "@keyframes" in file.read_text(encoding="utf-8", errors="ignore")


class AnimazioniInLaTeX(SphinxPostTransform):
    """Una figura che si muove diventa tre fermi immagine e un rimando.

    L'indirizzo non si scrive a mano: si ricava dal `docname` e dal `:name:`
    della figura, cosi' non puo' puntare alla pagina sbagliata. La radice e'
    la stessa `html_baseurl` che il sito mette nel `<link rel="canonical">`.

    Si sostituisce **l'immagine**, non la figura: didascalia e numero restano
    quelli che Sphinx ha gia' calcolato, e i `{numref}` sparsi nel testo
    continuano a puntare al numero giusto.
    """

    default_priority = 190          # prima del convertitore delle immagini
    formats = ("latex",)

    def dove_sta(self, figura) -> tuple[str, str]:
        """(ancora, pagina) di una figura, per costruirne l'indirizzo.

        Non si usa `env.docname` e non si usa il `source` del documento: il
        builder LaTeX **assembla un albero solo** a partire dal documento
        radice e poi ci fa girare sopra i post-transform, quindi tutti e due
        dicono `intro` per qualunque figura del libro. Si e' visto perche' i
        rimandi puntavano tutti alla stessa pagina.

        La fonte che sa la verita' e' il dominio `std`, dove ogni `:name:`
        e' registrato insieme al documento che lo contiene. E' la stessa
        tabella da cui `{numref}` prende i suoi numeri.
        """
        nomi = figura.get("names") or []
        etichette = self.env.domaindata.get("std", {})
        for nome in nomi:
            for tabella in ("anonlabels", "labels"):
                voce = etichette.get(tabella, {}).get(nome)
                if voce:
                    return nome, voce[0]
        # Ripiego: l'`ids` mangiato da Sphinx, e il file da cui il nodo viene.
        ancora = figura["ids"][0] if figura.get("ids") else ""
        pagina = ""
        if figura.source:
            pagina = self.env.path2doc(figura.source) or ""
        return ancora, pagina

    def run(self, **kwargs):
        radice = (self.app.config.html_baseurl or "").rstrip("/")
        sorgenti = pathlib.Path(self.app.srcdir)
        fermi = sorgenti / "figures" / "fermi"

        for figura in list(self.document.findall(nodes.figure)):
            immagine = next(figura.findall(nodes.image), None)
            if immagine is None:
                continue
            file = sorgenti / immagine["uri"].lstrip("/")
            if not si_muove(file):
                continue

            tre = [fermi / f"{file.stem}-{n}.png" for n in (1, 2, 3)]
            se_manca = [f.name for f in tre if not f.exists()]
            if se_manca:
                logger.warning("pt_stampa: %s si muove ma i fermi non ci "
                               "sono (%s). python3 animazioni/fermi.py",
                               file.name, ", ".join(se_manca))
                continue

            # Le tre immagini stanno in UN paragrafo, non in tre: il
            # traduttore LaTeX separa i nodi di pari livello con una riga
            # vuota, che e' un `\par`, e la terzina usciva impilata in
            # colonna invece che affiancata.
            blocco = nodes.container(classes=["pt-animazione"])
            riga = nodes.paragraph(classes=["pt-terzina"])
            for numero, fermo in enumerate(tre):
                if numero:
                    riga += nodes.raw("", "\\hfill", format="latex")
                riga += nodes.image(
                    uri=str(fermo.relative_to(sorgenti)),
                    candidates={"*": str(fermo.relative_to(sorgenti))},
                    width=LARGHEZZA_FERMO, alt=immagine.get("alt", ""))
            blocco += riga

            ancora, pagina = self.dove_sta(figura)
            if radice and ancora and pagina:
                url = f"{radice}/{pagina}.html#{ancora}"
                rimando = nodes.paragraph(classes=["pt-rimando"])
                rimando += nodes.Text("L'animazione si muove su ")
                rimando += nodes.reference("", url, refuri=url)
                blocco += rimando

            immagine.replace_self(blocco)


# I font del libro composto. Sta qui e non in `_config.yml` perche' `fontspec`
# vuole un percorso assoluto, e l'unico che lo conosce e' chi sa dov'e' la
# sorgente. Sostituisce il default di Sphinx per LuaLaTeX, che e' FreeSerif e
# non e' installato (e comunque non e' un font del brand).
#
# `Renderer=Basic` non e' un vezzo: senza, `fontspec` chiede a HarfBuzz un
# rendering completo che con queste istanze statiche non serve, e ogni riga
# costa di piu' su mille pagine.
FONTPKG = r"""
\usepackage{fontspec}
% Un font di RISERVA per i simboli. Inter, Fraunces e JetBrains Mono coprono
% il latino e poco altro: su 1753 pagine restavano ventidue glifi bianchi
% (l'insieme vuoto del capitolo sull'ASR, le parentesi angolari della
% fonetica, un epsilon, il simbolo di spazio, una T ad apice). Un buco in
% pagina non si vede nel log di nessuno se non lo si va a cercare.
%
% DejaVu Sans li ha tutti tranne una stella emoji, che comunque sta in un
% paragrafo che vale solo online. La riserva la gestisce luaotfload: quando
% il font principale non ha un carattere, lo prende di qui.
\directlua{luaotfload.add_fallback("ptSimboli", {"DejaVuSans:mode=node;"})}
\defaultfontfeatures{RawFeature={fallback=ptSimboli}}
\setmainfont{Inter}[Path=\ptFontDir,
  UprightFont=inter-400.ttf, ItalicFont=inter-400italic.ttf,
  BoldFont=inter-600.ttf, BoldItalicFont=inter-600italic.ttf]
\setsansfont{Inter}[Path=\ptFontDir,
  UprightFont=inter-500.ttf, ItalicFont=inter-400italic.ttf,
  BoldFont=inter-600.ttf, BoldItalicFont=inter-600italic.ttf]
\setmonofont{JetBrains Mono}[Path=\ptFontDir, Scale=0.86,
  UprightFont=jetbrains-mono-400.ttf,
  ItalicFont=jetbrains-mono-400italic.ttf,
  BoldFont=jetbrains-mono-600.ttf]
% Il carattere dei titoli. Non e' \setmainfont perche' il testo lungo e Inter,
% e Fraunces qui fa il mestiere che fa online: le intestazioni.
\newfontfamily\ptDisplay{Fraunces 72pt}[Path=\ptFontDir,
  UprightFont=fraunces-600.ttf, ItalicFont=fraunces-600italic.ttf,
  BoldFont=fraunces-800.ttf, BoldItalicFont=fraunces-600italic.ttf]
\newfontfamily\ptDisplayLeggero{Fraunces 72pt}[Path=\ptFontDir,
  UprightFont=fraunces-400.ttf, ItalicFont=fraunces-400italic.ttf,
  BoldFont=fraunces-600.ttf, BoldItalicFont=fraunces-600italic.ttf]
"""


# Il frontespizio e il colophon. Si generano invece di essere un file .tex
# scritto a mano per una ragione sola: dentro c'e' il numero di versione, e
# quello sta in un posto solo, la voce in cima a `_dati/aggiornamenti.yml`.
# Scritto a mano qui, un giorno direbbe una versione e la pagina degli
# aggiornamenti un'altra.
FRONTESPIZIO = r"""
\begin{titlepage}
\thispagestyle{empty}
\pagecolor{ptCrema}
\vspace*{\stretch{1}}
\begin{flushleft}
  % Il logo del libro, quello vero: bollo tribar piu' il lockup «paithon
  % book». Non il solo bollo con la parola ricomposta a mano, che diceva
  % «paithon» e non «Paithon Book», cioe' il nome di un'altra cosa.
  % LA COPERTINA STA IN UNA PAGINA PER POCHI MILLIMETRI, e le misure qui
  % sotto sono tarate su quello: il fregio da solo prende 111mm dei 297 del
  % foglio, e quando il claim si e' aggiunto sopra, il fregio e la riga della
  % versione sono finiti su una seconda pagina senza che niente lo dicesse
  % (LuaLaTeX non se ne lamenta: la copertina «riesce» lo stesso). Chi tocca
  % il corpo del claim, la larghezza del marchio o uno di questi \vspace
  % rifaccia il conto: `genera-pdf.py` lo verifica da se' a fine build, e con
  % questi valori il margine e' una decina di millimetri: tolto il sottotesto
  % la copertina ha ritrovato spazio, e una parte e' tornata al marchio e
  % all'aria sopra la firma invece di restare margine inutilizzato.
  \IfFileExists{marchio.pdf}{%
    \includegraphics[width=96mm]{marchio.pdf}\par\vspace{5mm}}{%
    {\ptDisplay\fontsize{50}{54}\selectfont Paithon Book\par}\vspace{4mm}}%
  {\color{ptTerracotta}\rule{42mm}{2.4pt}\par}
  \vspace{5mm}
  % Il claim: dice che cosa e' il libro, ed e' la riga che il lettore trova
  % identica sulla landing, nella card social e nel README. Qui mancava, e la
  % copertina apriva sulla postilla senza aver detto la cosa principale.
  % Va a capo dove va a capo sul sito (`.pt-claim` si chiude a 34ch) e sulla
  % card: e' la stessa riga e deve avere lo stesso respiro, quindi il ritorno
  % e' scritto invece di essere lasciato alla giustezza della pagina.
  {\ptDisplay\fontsize{22}{27}\selectfont
   Il Libro di Intelligenza Artificiale\\
   che spiega \textcolor{ptTeal}{due} \textcolor{ptTerracotta}{volte.}\par}
  % QUI SOTTO NON VA IL SOTTOTESTO, e per un po' c'e' stato. «l'AI che spiega
  % se stessa... due volte» e' una riga bella, ma sotto al claim diventa un'eco:
  % finiscono tutte e due su «due volte», a quattro centimetri di distanza, e
  % la seconda sembra la prima detta peggio. Una copertina dice una cosa. Il
  % sottotesto resta dov'e' una postilla e non un secondo titolo, cioe' sulla
  % landing, dove la Prefazione poi lo spiega per esteso.
  % Il sottotitolo lungo («Machine Learning, Deep Learning e Reinforcement
  % Learning con Python») non e' in copertina: dice di che cosa parla il
  % libro, e il sottotesto dice gia' che cos'e'. Resta nel colophon, dove
  % serve per citarlo: li' e' un dato bibliografico, non un richiamo.
  \vspace{12mm}
  {\ptDisplay\Large Francesco Messina\par}
\end{flushleft}
\vspace*{\stretch{1}}
% Il fregio: una conca e la discesa che la trova, con la traiettoria
% calcolata davvero (`book/_stampa/copertina.py`). Prima qui c'erano due
% terzi di pagina vuota, e la prima immagine del libro era il vuoto.
% Se il file non c'e' la copertina resta quella di prima: nessuno se ne
% accorge tranne chi la guarda.
% Va a tutta larghezza della CARTA, non del testo: dentro i margini
% sembrerebbe una figura incollata in copertina, e la texture di punti si
% interromperebbe su due bordi verticali che non c'entrano niente. Lo
% spostamento e' quello che riporta al bordo sinistro del foglio: un pollice
% (l'origine di TeX) piu' il margine della pagina dispari, che e' quella su
% cui cade sempre la copertina.
\IfFileExists{fregio.pdf}{%
  \noindent\hspace*{-\dimexpr 1in+\oddsidemargin\relax}%
  \includegraphics[width=\paperwidth]{fregio.pdf}\par
  \vspace*{\stretch{1}}}{\vspace*{\stretch{1}}}
\begin{flushleft}
  {\color{ptTeal}\normalsize VERSIONE_E_DATA\par}
  \vspace{2mm}
  {\color{ptTeal}\normalsize book.paithon.it\par}
\end{flushleft}
\end{titlepage}
\nopagecolor

\thispagestyle{empty}
\vspace*{\stretch{1}}
\begin{flushleft}\footnotesize
{\ptDisplay\large Paithon Book\par}
\vspace{1mm}
di Francesco Messina
\vspace{7mm}

{\ptDisplayLeggero\normalsize Questa è un'istantanea di un libro vivo.\par}
\vspace{1mm}
Versione VERSIONE_E_DATA. Il libro cambia di continuo: la versione
aggiornata, navigabile, con il codice da eseguire e le figure che si
muovono, sta su \textbf{book.paithon.it/main}. Se questo file ha
qualche mese, là dentro c'è di più.
\vspace{7mm}

{\ptDisplayLeggero\normalsize Come si può usare\par}
\vspace{1mm}
Copyright \textcopyright{} 2019ANNO_CORRENTE Francesco Messina, paithon.it.
Il testo e le figure sono
distribuiti sotto licenza \textbf{Creative Commons BY-NC-ND 4.0}
Internazionale. In breve: si può scaricare, leggere, stampare e
condividere liberamente, citando l'autore; non si può usare per scopi
commerciali, e non se ne possono distribuire versioni modificate. Il
testo completo della licenza:
\texttt{creativecommons.org/licenses/by-nc-nd/4.0/deed.it}.
\vspace{3mm}

Il \textbf{codice} degli esempi (blocchi, notebook, script) ha una licenza
sua, più permissiva, perché il codice serve a essere copiato e riusato:
\textbf{Apache License 2.0}. Il testo completo sta in \texttt{LICENSE-CODE}
nel repository e su \texttt{apache.org/licenses/LICENSE-2.0}.
\vspace{7mm}

{\ptDisplayLeggero\normalsize Come citarlo\par}
\vspace{1mm}
Francesco Messina, \textit{Paithon Book}, versione VERSIONE_SOLA,
\texttt{doi.org/10.5281/zenodo.21947219}.
\vspace{3mm}

Quel DOI è l'identificativo permanente del libro, ed è quello
\emph{di tutte le versioni}: chi lo apre arriva sempre all'ultima
depositata. Ogni versione ne ha poi uno suo, che si legge sulla scheda
di deposito, e va usato quando si cita un passaggio che potrebbe
cambiare. L'indirizzo di casa resta \texttt{book.paithon.it}.
\vspace{7mm}

{\ptDisplayLeggero\normalsize Se trovi un errore\par}
\vspace{1mm}
Segnalalo su \texttt{github.com/paithon-it/paithonbook/issues}: chi
segnala un errore viene citato nel commit che lo corregge.
\vspace{7mm}

{\ptDisplayLeggero\normalsize Com'è fatto\par}
\vspace{1mm}
Composto con LuaLaTeX. Caratteri Fraunces, Inter e JetBrains Mono
(SIL Open Font License). Tutte le figure sono originali, disegnate in
SVG; dove il libro online mostra un'animazione, qui trovi tre fermi
immagine e l'indirizzo per vederla muoversi.
\vspace{3mm}

Il segno al posto della \textbf{a} è un \textbf{triangolo di Penrose}: tre
lati che si reggono l'un l'altro in cerchio, come i tre passaggi da cui
questo libro esce. Ed è una figura impossibile, che è il modo esatto in
cui una macchina sbaglia su una materia tecnica: copri un angolo
qualunque e quel che resta è corretto, ma le tre soluzioni locali non si
possono avere tutte insieme. Ogni frase regge da sola, e il montaggio è
falso. La Prefazione lo racconta per esteso.
\end{flushleft}
\vspace*{\stretch{2}}
\clearpage

%% La dedica. Solo qui: e' una pagina che appartiene al libro stampato, dove
%% la carta bianca attorno a una riga sola e' parte di quello che dice. Sul
%% sito la stessa riga sarebbe un paragrafo qualsiasi in cima a una pagina.
\thispagestyle{empty}
\vspace*{\stretch{1}}
\begin{flushright}
  {\ptDisplayLeggero\itshape\large A Mareluna,\\[2mm]
   la mia ReLU.\par}
\end{flushright}
\vspace*{\stretch{3}}
\clearpage
"""


def frontespizio(app) -> str:
    """Frontespizio e colophon, col numero di versione preso dal registro."""
    import datetime
    import sys

    sys.path.insert(0, str(pathlib.Path(app.srcdir) / "_ext"))
    import pt_conteggi

    registro = pathlib.Path(app.srcdir) / "_dati" / "aggiornamenti.yml"
    marca = pt_conteggi.versione_corrente(registro) if registro.exists() else {}
    numero = marca.get("versione", "")
    data = marca.get("data_versione", "")

    testo = FRONTESPIZIO
    testo = testo.replace("VERSIONE_E_DATA",
                          f"{numero} del {data}" if data else numero)
    testo = testo.replace("VERSIONE_SOLA", numero or "in linea")
    # L'anno del copyright non si scrive: fra due anni sarebbe sbagliato.
    testo = testo.replace("ANNO_CORRENTE",
                          f"–{datetime.date.today().year}")
    return testo


def prepara_frontespizio(app, config):
    config.latex_elements["maketitle"] = frontespizio(app)


def porta_le_bande(app, config):
    """Aggiunge le bande di apertura ai file che LaTeX si porta dietro.

    Non stanno in `_config.yml` per due ragioni. La prima e' che sono una per
    capitolo, e un elenco scritto a mano resterebbe indietro al primo capitolo
    nuovo. La seconda e' che `latex_additional_files` **non prende cartelle**:
    passargliene una fa fallire la build con `IsADirectoryError`, che non
    dice affatto questo.
    """
    bande = pathlib.Path(app.srcdir) / "_stampa" / "bande"
    if not bande.is_dir():
        logger.info("pt_stampa: nessuna banda di capitolo "
                    "(python3 book/_stampa/copertina.py). Le aperture "
                    "restano la pillola col numero.")
        return
    trovate = sorted(bande.glob("capitolo-*.pdf"))
    config.latex_additional_files = list(config.latex_additional_files) + [
        str(f.relative_to(app.srcdir)) for f in trovate]
    logger.info("pt_stampa: %d bande di capitolo", len(trovate))


# Il blocco che `sphinx-jupyterbook-latex` infila nel preambolo. Si toglie
# per intero: dentro c'e' `ucharclasses`, che funziona solo con XeTeX, e
# `\contentsname` messo a "Contents".
SUO_PREAMBOLO = "% Start of preamble defined in sphinx-jupyterbook-latex %"
SUO_PREAMBOLO_FINE = "% End of preamble defined in sphinx-jupyterbook-latex %"


def disfa_jupyterbook_latex(app, config):
    """Rimette a posto quello che `sphinx-jupyterbook-latex` ha riscritto.

    Quella estensione la teniamo accesa per una ragione sola, e vale la pena:
    nel `_toc.yml` la radice e' `intro.md`, e senza di lei le sezioni della
    landing diventano i capitoli 1, 2 e 3, con i capitoli veri che slittano
    dietro. Nessuna build lo segnala; si vede aprendo il PDF.

    Ma a `config-inited` riscrive anche il motore (xelatex, che qui non c'e'),
    il tema, e il preambolo, dove infila `ucharclasses`, che **funziona solo
    con XeTeX**. Questo gancio gira dopo il suo (priorita' piu' alta) e gli
    disfa le mani.
    """
    config.latex_engine = "lualatex"
    config.latex_theme = "manual"

    preambolo = config.latex_elements.get("preamble", "")
    inizio = preambolo.find(SUO_PREAMBOLO)
    fine = preambolo.find(SUO_PREAMBOLO_FINE)
    if inizio != -1 and fine != -1:
        preambolo = preambolo[:inizio] + preambolo[fine + len(SUO_PREAMBOLO_FINE):]
    if "usepackage{paithon}" not in preambolo:
        preambolo += "\n\\usepackage{paithon}\n"
    # I metadati del PDF: il titolo e' il nome del libro, e la riga lunga
    # («Machine Learning, Deep Learning...») vive qui invece che in copertina.
    # E' quello che mostrano un lettore di PDF e un motore di ricerca: dice
    # di che cosa parla il libro dove serve saperlo, senza occupare la
    # copertina, dove il sottotesto dice gia' che cos'e'.
    preambolo += ("\n\\hypersetup{pdftitle={Paithon Book},"
                  "pdfauthor={Francesco Messina},"
                  "pdfsubject={Machine Learning, Deep Learning e "
                  "Reinforcement Learning con Python},"
                  "pdfkeywords={intelligenza artificiale, machine learning, "
                  "deep learning, reinforcement learning, Python, PyTorch}}\n")
    config.latex_elements["preamble"] = preambolo


def prepara_font(app, config):
    """Mette i font del brand al posto del default di Sphinx."""
    cartella = pathlib.Path(app.srcdir) / "_static" / "brand" / "stampa" / "fonts"
    if not cartella.is_dir():
        logger.warning("pt_stampa: la cartella dei font non c'e' (%s). "
                       "git submodule update --remote book/_static/brand",
                       cartella)
        return
    # La cartella si incolla, non si formatta: `FONTPKG` e' pieno di commenti
    # LaTeX, che cominciano per `%`, e con `%` di mezzo Python li legge come
    # segnaposto. La prima versione di questa riga faceva proprio cosi' e
    # moriva con «not enough arguments for format string».
    config.latex_elements["fontpkg"] = (
        "\\def\\ptFontDir{" + str(cartella) + "/}\n" + FONTPKG)


def chiudi_browser(app, exception):
    BROWSER.chiudi()


def setup(app):
    # La priorita' alta serve: questi ganci devono girare DOPO quello di
    # `sphinx-jupyterbook-latex`, che gira al valore di default.
    app.connect("config-inited", disfa_jupyterbook_latex, priority=900)
    app.connect("config-inited", prepara_font, priority=901)
    app.connect("config-inited", prepara_frontespizio, priority=902)
    app.connect("config-inited", porta_le_bande, priority=903)
    app.add_post_transform(FuoriStampa)
    app.add_post_transform(ApreLaPrefazione)
    app.add_post_transform(ViaISeparatori)
    app.add_post_transform(TabInLaTeX)
    app.add_post_transform(AnimazioniInLaTeX)
    app.add_post_transform(ConvertitoreChromium)
    app.connect("build-finished", chiudi_browser)
    return {"version": "1.0",
            "parallel_read_safe": True,
            # Il browser e' uno solo e non si divide fra processi.
            "parallel_write_safe": False}
