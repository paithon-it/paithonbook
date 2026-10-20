#!/usr/bin/env python3
"""La resa scura di ogni figura del libro, derivata e non disegnata.

Le figure del libro sono SVG flat sulla palette chiara della marca: inchiostro
`#1A1A1A` su carta `#F8F5EE`, con i tre accenti. In tema scuro il tema Sphinx
le trattava come fotografie e le appoggiava su una **cartolina bianca** con
`filter: brightness(.8) contrast(1.2)`: colori spenti dentro un rettangolo che
non c'entra niente con la pagina.

La cura non e' un filtro, che sui colori della marca non sa dove mettere le
mani: e' una **seconda copia** di ogni figura con la palette scura del brand,
che `book/_ext/pt_scuro.py` affianca alla prima in pagina e il tema mostra
quando serve. Questo file scrive quelle copie in `book/figures/scure/`.

    python3 scripts/genera-figure-scure.py             # riscrive le copie
    python3 scripts/genera-figure-scure.py --verifica  # sono allineate?
    python3 scripts/genera-figure-scure.py --solo nome-figura [altra...]

## La mappa, e da dove viene

Non sta qui: e' quella del brand, `book/_static/brand/figure/palette.json`,
letta con il suo `palette.py`. E' la stessa con cui il sito ricolora le sue
figure e con cui il motore di Manim disegna le clip in tema scuro, quindi una
figura statica e una clip sulla stessa pagina hanno lo stesso fondo e gli
stessi colori. Chi vuole cambiare un colore scuro lo cambia nel brand, e qui
basta rigenerare.

I **grigi** sono il punto delicato. Il libro ne usa parecchi, e una mappa che
li schiacciasse tutti su due token ne fonderebbe due in una figura sola: coppie
come `#5E5852` e `#8B847C`, o `#C5BEAA` e `#E2DCC9`, convivono in molte figure.
La regola e' che **nessuna coppia che convive finisce sullo stesso valore**, e
siccome la mappa sta nel brand e cambia di la', il conto non si fa a mano: lo
fa `fusioni()` su ogni figura, e una coppia nuova ferma tutto, come un colore
sconosciuto. Le fusioni volute sono elencate in `FUSIONI_VOLUTE`, ciascuna con
la sua ragione.

Il verso e' quello dell'inversione: il grigio piu' scuro sul chiaro (quello
che si vede di piu') diventa il piu' chiaro sullo scuro.

## Che cosa NON si ricolora

- `triangolo-di-penrose.svg`, che e' il segno della marca: `logos/README.md`
  lo dichiara **palette-locked**, i suoi tre colori non cambiano in nessun
  tema. La sua pagina lo marca gia' `dark-light` e il tema lo lascia stare.
- le figure che nessuna pagina del `_toc.yml` richiama: una resa scura
  che non guarda nessuno peserebbe e basta.
- le animazioni di Manim, che sono GIF. Un raster non si ricolora: i colori
  della marca sono gia' fusi nei pixel, insieme all'antialiasing dei bordi e
  dei glifi, e qualunque filtro sposterebbe anche i tre accenti, che nella
  palette scura non sono l'inverso di quelli chiari ma tre colori scelti
  apposta. La loro resa scura si **ri-renderizza** dallo stesso sorgente con
  `PAITHON_TEMA=scuro`, e la scrive `animazioni/scure.py` nella stessa
  cartella di queste.

## Perche' rifiuta invece di indovinare

Un colore che non sta nella mappa ferma tutto. Indovinarlo (schiarirlo di
tanto, invertirlo) darebbe una figura plausibile e sbagliata in un punto solo,
che nessuno riaprirebbe mai. Chi aggiunge un colore al libro lo aggiunge nel
brand, in `figure/palette.json`, guardando i token.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

QUI = pathlib.Path(__file__).resolve().parent.parent
LIBRO = QUI / "book"
FIGURE = LIBRO / "figures"
SCURE = FIGURE / "scure"

# Il perimetro del libro sta in `coerenza.py`, in una copia sola: due copie
# divergono, e quella che stava qui leggeva il toc con un lettore suo. Il
# `sys.path` e' esplicito perche' lanciando questo file come script la
# cartella c'e' gia', ma importandolo da altrove no.
sys.path.insert(0, str(QUI / "scripts"))
from coerenza import pagine_del_toc                        # noqa: E402

# La mappa chiaro -> scuro del brand: i colori della palette delle figure piu'
# i loro sinonimi (grigi dell'interfaccia rimasti nelle figure). Il percorso e'
# esplicito per la stessa ragione di quello qui sopra.
sys.path.insert(0, str(LIBRO / "_static" / "brand" / "figure"))
import palette                                             # noqa: E402

MAPPA: dict[str, str] = palette.MAPPA_SCURA

# Due colori chiari che finiscono sullo stesso scuro e possono convivere nella
# stessa figura, perche' lo ha deciso il brand e qui c'e' il perche'. Ogni
# altra coppia che convive e collide ferma la generazione.
FUSIONI_VOLUTE = {
    # l'ocra cupa esiste per fare da TESTO sulla carta chiara, dove l'ocra
    # piena non si legge. Sul nero l'ocra piena regge da sola come testo, e
    # quindi il mestiere e' lo stesso dall'altra parte: un colore, non due.
    frozenset({"#C9A961", "#9A7B2E"}),
}

# `currentColor` dentro un SVG caricato con <img> non eredita niente dalla
# pagina: vale il `color` iniziale, cioe' nero. Sulla carta chiara passa per
# inchiostro; sul fondo scuro sparirebbe, quindi diventa l'inchiostro scuro.
CORRENTE = palette.scuro("#1A1A1A")

# Palette-locked: il segno della marca non cambia colore in nessun tema.
FERME = {"triangolo-di-penrose.svg", "favicon.svg"}

RX_ESADECIMALE = re.compile(r"#[0-9A-Fa-f]{6}\b")
RX_CORRENTE = re.compile(r"\bcurrentColor\b")

INTESTAZIONE = """<!--
  GENERATO da scripts/genera-figure-scure.py: non modificare a mano.
  La prossima rigenerazione cancella qualunque cosa venga scritta
  qui dentro. La mappa dei colori e' quella del brand,
  book/_static/brand/figure/palette.json.
  fonte: ../{nome}
-->
"""


def figure_richiamate() -> set[str]:
    """I nomi degli SVG che una pagina del libro richiama davvero."""
    import json

    usate: set[str] = set()
    pagine = pagine_del_toc()
    assert pagine, "nessuna pagina nel _toc.yml: perimetro sbagliato"
    for pagina in pagine:
        testo = pagina.read_text(encoding="utf-8", errors="replace")
        if pagina.suffix == ".ipynb":
            # in un .ipynb il `source` di una cella e' una lista di righe
            # OPPURE una stringa: lo stesso ciclo, senza questo, conta le
            # righe in un file e i caratteri nell'altro senza protestare.
            celle = json.loads(testo).get("cells", [])
            pezzi = []
            for cella in celle:
                sorgente = cella.get("source", "")
                pezzi.append("".join(sorgente) if isinstance(sorgente, list)
                             else str(sorgente))
            testo = "\n".join(pezzi)
        usate |= {pathlib.Path(m).name
                  for m in re.findall(r"[\w./-]+\.svg", testo)}
    return usate


def da_ricolorare(solo: list[str] | None = None) -> list[pathlib.Path]:
    """Le figure che hanno diritto a una resa scura, in ordine."""
    usate = figure_richiamate()
    file = [p for p in sorted(FIGURE.glob("*.svg"))
            if p.name in usate and p.name not in FERME]
    if solo:
        voluti = {n if n.endswith(".svg") else f"{n}.svg" for n in solo}
        mancanti = voluti - {p.name for p in file}
        if mancanti:
            raise SystemExit(
                f"non sono figure ricolorabili: {sorted(mancanti)}")
        file = [p for p in file if p.name in voluti]
    return file


def fusioni(testo: str) -> list[tuple[str, ...]]:
    """Le coppie di colori della figura che finirebbero sullo stesso scuro.

    Solo quelle che convivono davvero nella figura, e senza le volute: in
    tema scuro sarebbero due cose diverse disegnate dello stesso colore.
    """
    per_scuro: dict[str, set[str]] = {}
    for m in RX_ESADECIMALE.finditer(testo):
        chiaro = m.group(0).upper()
        if chiaro in MAPPA:
            per_scuro.setdefault(MAPPA[chiaro], set()).add(chiaro)
    fuse = []
    for gruppo in per_scuro.values():
        ordinati = sorted(gruppo)
        for i, uno in enumerate(ordinati):
            for altro in ordinati[i + 1:]:
                if frozenset({uno, altro}) not in FUSIONI_VOLUTE:
                    fuse.append((uno, altro))
    return sorted(fuse)


def scurisci(testo: str, nome: str) -> str:
    """La stessa figura con la palette scura.

    Rifiuta un colore che non conosce invece di indovinarlo, e due colori che
    convivono e diventerebbero uno solo.
    """
    ignoti = sorted({m.group(0).upper() for m in RX_ESADECIMALE.finditer(testo)}
                    - set(MAPPA))
    if ignoti:
        raise SystemExit(
            f"{nome}: colori che la mappa non conosce: {', '.join(ignoti)}.\n"
            f"  Si aggiungono nel brand, in figure/palette.json (qui "
            f"book/_static/brand), prendendo il valore scuro dai token.")

    fuse = fusioni(testo)
    if fuse:
        raise SystemExit(
            f"{nome}: colori che convivono e sullo scuro diventerebbero "
            f"uno solo: {', '.join(' e '.join(c) for c in fuse)}.\n"
            f"  O la figura smette di usarne uno, o la mappa del brand li "
            f"separa; se la fusione e' voluta, va in FUSIONI_VOLUTE con la "
            f"sua ragione.")

    # Prima gli esadecimali, POI `currentColor`: al contrario, il valore che
    # `currentColor` lascia al suo posto e' esso stesso un esadecimale, e la
    # passata dopo lo ritrova e non lo riconosce.
    fuori = RX_ESADECIMALE.sub(lambda m: MAPPA[m.group(0).upper()], testo)
    fuori = RX_CORRENTE.sub(CORRENTE, fuori)

    # Il collaudo del proprio attrezzo: dopo la sostituzione non deve
    # sopravvivere nessun colore della palette CHIARA. Se ne resta uno, la
    # regex ha guardato il posto sbagliato e la figura sarebbe uscita a meta'.
    resti = sorted({c for c in MAPPA if c in fuori.upper()}
                   - {v.upper() for v in MAPPA.values()})
    assert not resti, f"{nome}: colori chiari sopravvissuti: {resti}"

    # L'intestazione va DOPO l'eventuale dichiarazione XML, che deve restare
    # la prima riga del file.
    if fuori.lstrip().startswith("<?xml"):
        taglio = fuori.index("?>") + 2
        return (fuori[:taglio] + "\n"
                + INTESTAZIONE.format(nome=nome) + fuori[taglio:].lstrip("\n"))
    return INTESTAZIONE.format(nome=nome) + fuori


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verifica", action="store_true",
                    help="non scrive: esce 1 se una resa scura e'\n"
                         "rimasta indietro")
    ap.add_argument("--solo", nargs="+", metavar="FIGURA",
                    help="solo queste figure (con o senza .svg)")
    argomenti = ap.parse_args()

    file = da_ricolorare(argomenti.solo)
    assert file, ("elenco vuoto: nessuna figura da ricolorare, "
                  "perimetro sbagliato")

    atteso = {p.name: scurisci(p.read_text(encoding="utf-8"), p.name)
              for p in file}

    # Le rese scure rimaste senza originale: una figura rinominata o tolta.
    orfane = []
    if SCURE.is_dir() and not argomenti.solo:
        vivi = {p.name for p in FIGURE.glob("*.svg")}
        orfane = sorted(p for p in SCURE.glob("*.svg") if p.name not in vivi)

    if argomenti.verifica:
        indietro = [n for n, t in atteso.items()
                    if not (SCURE / n).is_file()
                    or (SCURE / n).read_text(encoding="utf-8") != t]
        print(f"  lette {len(file)} figure chiare, "
              f"{len(list(SCURE.glob('*.svg'))) if SCURE.is_dir() else 0}"
              f" scure")
        if indietro:
            print(f"\n  {len(indietro)} rese scure rimaste indietro:")
            for nome in indietro[:20]:
                print(f"    {nome}")
            if len(indietro) > 20:
                print(f"    ... e altre {len(indietro) - 20}")
            print("\n  python3 scripts/genera-figure-scure.py")
        if orfane:
            print(f"\n  {len(orfane)} rese scure senza originale: "
                  f"{', '.join(p.name for p in orfane[:5])}")
        if indietro or orfane:
            return 1
        print("  allineate")
        return 0

    SCURE.mkdir(parents=True, exist_ok=True)
    scritti = 0
    for nome, testo in atteso.items():
        destinazione = SCURE / nome
        if (destinazione.is_file()
                and destinazione.read_text(encoding="utf-8") == testo):
            continue
        destinazione.write_text(testo, encoding="utf-8")
        scritti += 1
    for orfana in orfane:
        orfana.unlink()

    print(f"  lette {len(file)} figure chiare in {FIGURE.relative_to(QUI)}")
    print(f"  {scritti} rese scure scritte in {SCURE.relative_to(QUI)}"
          f" ({len(atteso) - scritti} gia' allineate)")
    if orfane:
        print(f"  {len(orfane)} senza originale, tolte: "
              f"{', '.join(p.name for p in orfane)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
