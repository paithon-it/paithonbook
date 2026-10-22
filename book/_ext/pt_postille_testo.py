"""Che cosa e' un capoverso del testo comune, e l'impronta che lo riconosce.

Una postilla e' una riga accanto a un capoverso del testo comune, cioe' fuori
dalle schede, dai riquadri, dagli elenchi e dai blocchi `{only}`, che dice di
che cosa parla il capoverso. Sta in un file accanto alla pagina
(`pagina.postille.yml`) insieme all'impronta del capoverso che accompagna, e
vale finche' l'impronta combacia: quando il capoverso cambia, la postilla si
spegne, e torna quando qualcuno la rilegge e la riconferma.

Qui sta la parte che non dipende da Sphinx, perche' la usano tre attrezzi:
l'estensione `pt_postille.py`, che aggancia le postille durante la build; il
controllo di coerenza, che gira senza Sphinx; e chi scrive le postille. La
definizione e' una sola, perche' due definizioni di capoverso che divergono
spegnerebbero o accenderebbero postille senza che nessuno se ne accorga.

Il capoverso si riconosce sul sorgente, con le stesse regole dei recinti che
segue la maschera della prosa di `scripts/coerenza.py`:

- i recinti stanno su una pila, perche' una scheda a cinque backtick contiene
  riquadri e blocchi di codice a tre;
- dentro un recinto di codice non si apre niente: la riga `~~~~^^^^` che un
  traceback stampa sotto il punto che ha ceduto e' contenuto, e l'unica riga
  che conta e' la chiusura;
- i recinti che non sono codice (schede, riquadri, figure, `{only}`) si
  attraversano: i loro capoversi esistono, ma non sono testo comune, e lo dice
  il campo `zona`.

Un capoverso finisce a una riga vuota, o dove comincia un blocco che per
CommonMark lo interrompe: un titolo, una voce d'elenco, una citazione, una
formula esposta, un recinto. Non sono capoversi i titoli (anche quelli
sottolineati), gli elenchi, le tabelle, le citazioni, l'HTML, i commenti, le
note, le formule esposte e i blocchi rientrati, che sono codice o il seguito di
una voce d'elenco.

L'impronta sono i primi dodici caratteri esadecimali dello SHA-256 del testo
normalizzato: forma NFC, apostrofi e virgolette tipografici resi dritti, ogni
sequenza di spazi e a capo ridotta a uno spazio. Contano le parole e la
punteggiatura, non gli a capo: riavvolgere un capoverso a un'altra misura non
spegne la sua postilla.

Il file delle postille si scrive con tutti i valori fra virgolette doppie: in
YAML un'impronta fatta di sole cifre senza virgolette diventa un numero, una
postilla che dice soltanto «no» diventa un valore booleano, e in un valore
senza virgolette « #» comincia un commento, che tronca la postilla senza
errore. Per questo il file si rifiuta anche quando un valore senza virgolette
si legge come un testo qualunque.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

ESTENSIONE = ".postille.yml"
CAMPI = ("inizio", "impronta", "postilla")
INTESTAZIONE = (
    "# Le postille di questa pagina, una per capoverso del testo comune.\n"
    "# Ognuna vale finché la sua impronta combacia col testo del capoverso.\n")
COMUNE = "comune"

_DRITTI = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})

_RX_RECINTO = re.compile(r"^(\s*)(`{3,}|~{3,}|:{3,})(.*)$")

# Le etichette dei recinti che non contengono prosa: si saltano per intero.
_RX_OPACO = re.compile(
    r"^\{?(?:code-block|code-cell|code|literalinclude|eval-rst|raw|math"
    r"|toctree|bibliography|glossary)\}?(?![\w-])"
    r"|^(?:python|py|ipython3?|text|bash|sh|shell|console|json|ya?ml|toml"
    r"|latex|tex|html|css|js|javascript|c|cpp|sql|diff|ini|makefile"
    r"|dockerfile|xml|markdown|md|r|julia|rust|go)\b", re.I)

# Le righe che sono un blocco da sole: subito sotto, anche senza una riga
# vuota, puo' cominciare un capoverso. Un commento `%` che si prendesse le
# righe fino alla riga vuota si mangerebbe il capoverso che lo segue.
_RX_RIGA_SOLA = re.compile(
    r"#{1,6}(?:\s|$)"                         # titolo
    r"|%"                                     # commento
    r"|\(\S*\)=\s*$"                          # bersaglio
    r"|\+\+\+"                                # separatore di blocchi
    r"|(?:[*_-][ \t]*){3,}$")                 # filetto

# I blocchi che non sono capoversi e durano fino alla riga vuota.
_RX_ALTRO_BLOCCO = re.compile(
    r"[-*+](?:\s|$)|\d{1,9}[.)](?:\s|$)"      # voce d'elenco
    r"|>"                                     # citazione
    r"|\|"                                    # tabella
    r"|<"                                     # HTML
    r"|\[[^\]]+\]:")                          # nota o link definiti a parte

# Le righe che interrompono un capoverso aperto (CommonMark: un elenco
# numerato lo interrompe solo se comincia da 1, e solo l'HTML di blocco).
_RX_INTERROMPE = re.compile(
    r"#{1,6}(?:\s|$)"
    r"|[-*+]\s+\S|1[.)]\s+\S"
    r"|>"
    r"|\$\$"
    r"|%"
    r"|\(\S*\)=\s*$"
    r"|\+\+\+"
    r"|(?:[*_][ \t]*){3,}$"
    r"|</?(?:address|article|aside|blockquote|details|div|dl|fieldset|figure"
    r"|footer|form|h[1-6]|header|hr|li|main|nav|ol|p|pre|script|section|style"
    r"|table|ul)\b|<!--", re.I)

# Una riga di soli `=` o `-` sotto un capoverso ne fa un titolo.
_RX_SOTTOLINEATURA = re.compile(r"(?:=+|-+)[ \t]*$")


@dataclass(frozen=True)
class Capoverso:
    riga: int       # la prima riga nel sorgente, contata da 1
    testo: str      # le sue righe, unite da "\n"
    impronta: str
    inizio: str     # le prime parole, per riconoscerlo a occhio
    zona: str       # COMUNE, o l'etichetta del recinto che lo contiene


@dataclass(frozen=True)
class Voce:
    inizio: str
    impronta: str
    postilla: str


def normalizza(testo: str) -> str:
    testo = unicodedata.normalize("NFC", testo).translate(_DRITTI)
    return re.sub(r"\s+", " ", testo).strip()


def impronta(testo: str) -> str:
    return hashlib.sha256(normalizza(testo).encode("utf-8")).hexdigest()[:12]


def inizio_di(testo: str, parole: int = 6) -> str:
    return " ".join(normalizza(testo).split()[:parole])


def _chiude(segno: str, etichetta: str, aperto: str) -> bool:
    return not etichetta and segno[0] == aperto[0] and len(segno) >= len(aperto)


def capoversi(sorgente: str) -> list[Capoverso]:
    """I capoversi della pagina, nell'ordine, ciascuno con la sua zona."""
    righe = sorgente.split("\n")
    trovati: list[Capoverso] = []
    pila: list[tuple[str, str, bool]] = []     # (segno, etichetta, opaco)
    blocco: list[str] = []
    prima = 0          # la riga in cui comincia il blocco aperto
    zona = COMUNE      # la zona del blocco aperto
    altro = False      # un blocco che non e' un capoverso, fino alla riga vuota
    formula = False    # dentro una formula esposta `$$ ... $$`

    def chiudi() -> None:
        nonlocal blocco
        if blocco:
            testo = "\n".join(blocco)
            trovati.append(Capoverso(prima, testo, impronta(testo),
                                     inizio_di(testo), zona))
        blocco = []

    inizio = 0
    if righe and righe[0].strip() == "---":             # front matter
        for j in range(1, len(righe)):
            if righe[j].strip() in ("---", "..."):
                inizio = j + 1
                break

    for numero in range(inizio + 1, len(righe) + 1):
        riga = righe[numero - 1]
        m = _RX_RECINTO.match(riga)
        if m and pila and pila[-1][2] and not _chiude(
                m.group(2), m.group(3).strip(), pila[-1][0]):
            m = None                       # dentro il codice: contenuto e basta
        if m:
            chiudi()
            altro = formula = False
            segno, etichetta = m.group(2), m.group(3).strip()
            if pila and _chiude(segno, etichetta, pila[-1][0]):
                pila.pop()
            else:
                pila.append((segno, etichetta,
                             not etichetta or bool(_RX_OPACO.match(etichetta))))
            continue
        if any(opaco for _, _, opaco in pila):
            continue
        if formula:
            if "$$" in riga:
                formula = False
            continue
        if not riga.strip():
            chiudi()
            altro = False
            continue
        if altro:
            continue
        if blocco:
            if _RX_SOTTOLINEATURA.match(riga):
                blocco = []                # era un titolo sottolineato
                continue
            if not _RX_INTERROMPE.match(riga):
                blocco.append(riga)
                continue
            chiudi()
        # qui comincia un blocco nuovo
        if riga.lstrip().startswith("$$"):
            formula = "$$" not in riga.lstrip()[2:]
            continue
        if _RX_RIGA_SOLA.match(riga):
            continue
        if riga[:1] in " \t" or _RX_ALTRO_BLOCCO.match(riga):
            altro = True
            continue
        blocco = [riga]
        prima = numero
        zona = pila[-1][1] if pila else COMUNE
    chiudi()
    return trovati


def file_delle_postille(pagina: Path) -> Path:
    return pagina.with_name(pagina.stem + ESTENSIONE)


def leggi_postille(percorso: Path) -> list[Voce]:
    """Le voci del file, o `ValueError` col motivo per cui non si legge."""
    import yaml

    testo = percorso.read_text(encoding="utf-8")
    try:
        dati = yaml.safe_load(testo)
        albero = yaml.compose(testo, Loader=yaml.SafeLoader)
    except yaml.YAMLError as errore:
        raise ValueError(f"non è YAML valido ({str(errore).splitlines()[0]})") from None
    if dati is None:
        return []
    if not isinstance(dati, list):
        raise ValueError("deve essere un elenco di voci")
    voci = []
    for numero, (dato, nodo) in enumerate(zip(dati, albero.value), 1):
        if not isinstance(dato, dict) or set(dato) != set(CAMPI):
            raise ValueError(f"la voce {numero} deve avere esattamente i campi "
                             f"{', '.join(CAMPI)}")
        for campo in CAMPI:
            if not isinstance(dato[campo], str):
                raise ValueError(f"la voce {numero}: «{campo}» deve essere un "
                                 f"testo fra virgolette")
        for chiave, valore in nodo.value:
            if not isinstance(valore, yaml.ScalarNode) or valore.style != '"':
                raise ValueError(f"la voce {numero}: «{chiave.value}» va scritto "
                                 f"fra virgolette doppie")
        if not re.fullmatch(r"[0-9a-f]{12}", dato["impronta"]):
            raise ValueError(f"la voce {numero}: l'impronta deve avere 12 cifre "
                             f"esadecimali")
        voci.append(Voce(dato["inizio"], dato["impronta"], dato["postilla"]))
    return voci


def _tra_virgolette(testo: str) -> str:
    return '"' + testo.replace("\\", "\\\\").replace('"', '\\"') + '"'


def scrivi_postille(percorso: Path, voci: list[Voce]) -> None:
    righe = [INTESTAZIONE.rstrip("\n")]
    for voce in voci:
        righe += [f"- inizio: {_tra_virgolette(voce.inizio)}",
                  f"  impronta: {_tra_virgolette(voce.impronta)}",
                  f"  postilla: {_tra_virgolette(voce.postilla)}"]
    percorso.write_text("\n".join(righe) + "\n", encoding="utf-8")


def abbina(capoversi: list[Capoverso],
           voci: list[Voce]) -> tuple[dict[int, Voce], list[Voce]]:
    """Le voci accese, per riga del loro capoverso, e quelle spente.

    E' accesa la voce la cui impronta corrisponde a un capoverso solo, e quel
    capoverso sta nel testo comune. Un'impronta che compare in due capoversi
    della stessa pagina non si aggancia a nessuno dei due: scegliere vorrebbe
    dire indovinare.
    """
    per_impronta: dict[str, list[Capoverso]] = {}
    for capoverso in capoversi:
        per_impronta.setdefault(capoverso.impronta, []).append(capoverso)
    accese: dict[int, Voce] = {}
    spente: list[Voce] = []
    for voce in voci:
        trovati = per_impronta.get(voce.impronta, [])
        if len(trovati) == 1 and trovati[0].zona == COMUNE:
            accese[trovati[0].riga] = voce
        else:
            spente.append(voce)
    return accese, spente


def tipografia(testo: str) -> str:
    """Virgolette e apostrofi della postilla come li compone il libro.

    Il testo della pagina passa dalla sostituzione tipografica di docutils;
    quello della postilla no, perche' arriva da un attributo, quindi la si fa
    qui. Una postilla non contiene codice ne' formule, quindi un apostrofo
    dritto e' sempre un apostrofo.
    """
    return re.sub(r'"([^"]*)"', r"«\1»", testo).replace("'", "’")
