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

Non e' una scelta di gusto: sono i token del blocco `[data-theme="dark"]` di
`book/_static/brand/tokens.css`, cioe' gli stessi colori con cui il sito e il
libro dipingono tutto il resto della pagina scura, piu' la tabella
chiaro/scuro del segno in `book/_static/brand/logos/README.md`.

I **grigi** sono il punto delicato. Il libro ne usa nove, e una mappa che li
schiacciasse tutti su due token ne fonderebbe due in una figura sola: succede
davvero, `#5E5852` e `#8B847C` convivono in 59 figure, `#C5BEAA` e `#E2DCC9`
in 43. Le coppie che convivono sono state contate una per una, e la mappa qui
sotto e' scelta perche' **nessuna coppia che convive finisce sullo stesso
valore**. Chi la tocca rifaccia quel conto.

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
che nessuno riaprirebbe mai. Chi aggiunge un colore al libro aggiunge una riga
qui, e la aggiunge guardando i token del brand.
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

# chiaro -> scuro, e il token del brand da cui viene. L'ordine e' quello della
# scala: prima gli ancoraggi, poi i grigi dal piu' scuro al piu' chiaro.
MAPPA = {
    "#1A1A1A": ("#F4ECDD", "--pt-fg, l'osso"),
    "#F8F5EE": ("#0E0C0A", "--pt-bg, la carta scura"),
    "#B5532C": ("#E27B52", "--pt-accent, la terracotta scura"),
    "#2D5A5C": ("#5BA39C", "--pt-accent-2, il teal scuro"),
    "#C9A961": ("#DDB874", "--pt-accent-3, l'ocra scura"),
    # ocra scurita apposta per fare da TESTO sulla carta chiara (dove l'ocra
    # piena non si legge). Sul nero il testo lo regge l'ocra scura, e quindi
    # va li': non e' un colore in piu', e' lo stesso mestiere dall'altra parte.
    "#9A7B2E": ("#DDB874", "--pt-accent-3, l'ocra scura"),
    "#5A524A": ("#BCB3A1", "--pt-fg-muted"),
    "#5E5852": ("#BCB3A1", "--pt-fg-muted"),
    "#8B847C": ("#8E8676", "--pt-fg-subtle"),
    "#A89F92": ("#8E8676", "--pt-fg-subtle"),
    "#C5BEAA": ("#34302A", "--pt-border-strong"),
    "#C9C2B4": ("#34302A", "--pt-border-strong"),
    "#E2DCC9": ("#221F1A", "--pt-border"),
    "#E8E2D5": ("#221F1A", "--pt-border"),
    "#F2EDE2": ("#16130F", "--pt-bg-elev"),
}

# `currentColor` dentro un SVG caricato con <img> non eredita niente dalla
# pagina: vale il `color` iniziale, cioe' nero. Sulla carta chiara passa per
# inchiostro; sul fondo scuro sparirebbe.
CORRENTE = "#F4ECDD"

# Palette-locked: il segno della marca non cambia colore in nessun tema.
FERME = {"triangolo-di-penrose.svg", "favicon.svg"}

RX_ESADECIMALE = re.compile(r"#[0-9A-Fa-f]{6}\b")
RX_CORRENTE = re.compile(r"\bcurrentColor\b")

INTESTAZIONE = """<!--
  GENERATO da scripts/genera-figure-scure.py: non modificare a mano.
  La prossima rigenerazione cancella qualunque cosa venga scritta
  qui dentro. La palette e' quella del blocco [data-theme=dark]
  di book/_static/brand/tokens.css.
  fonte: ../{nome}
-->
"""


def pagine_del_toc() -> list[pathlib.Path]:
    """Le pagine del libro: dal `_toc.yml`, e con i `.ipynb` dentro.

    Non `book/**/*.md`: quel glob salta i notebook e prende i file fuori dal
    toc, cioe' misura un libro che non esiste.
    """
    import yaml

    dati = yaml.safe_load((LIBRO / "_toc.yml").read_text(encoding="utf-8"))
    nomi: list[str] = []

    def scava(nodo):
        if isinstance(nodo, dict):
            if "file" in nodo:
                nomi.append(str(nodo["file"]))
            for valore in nodo.values():
                scava(valore)
        elif isinstance(nodo, list):
            for valore in nodo:
                scava(valore)

    scava(dati)
    if "root" in dati:
        nomi.append(str(dati["root"]))

    pagine = []
    for nome in nomi:
        for estensione in (".md", ".ipynb", ""):
            percorso = LIBRO / (nome + estensione)
            if percorso.is_file():
                pagine.append(percorso)
                break
    return sorted(set(pagine))


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


def scurisci(testo: str, nome: str) -> str:
    """La stessa figura con la palette scura.

    Rifiuta un colore che non conosce invece di indovinarlo.
    """
    ignoti = sorted({m.group(0).upper() for m in RX_ESADECIMALE.finditer(testo)}
                    - set(MAPPA))
    if ignoti:
        raise SystemExit(
            f"{nome}: colori che la mappa non conosce: {', '.join(ignoti)}.\n"
            f"  Aggiungili in testa a {pathlib.Path(__file__).name}, "
            f"prendendo il valore dal blocco [data-theme=\"dark\"] di "
            f"book/_static/brand/tokens.css.")

    # Prima gli esadecimali, POI `currentColor`: al contrario, il valore che
    # `currentColor` lascia al suo posto e' esso stesso un esadecimale, e la
    # passata dopo lo ritrova e non lo riconosce.
    fuori = RX_ESADECIMALE.sub(lambda m: MAPPA[m.group(0).upper()][0], testo)
    fuori = RX_CORRENTE.sub(CORRENTE, fuori)

    # Il collaudo del proprio attrezzo: dopo la sostituzione non deve
    # sopravvivere nessun colore della palette CHIARA. Se ne resta uno, la
    # regex ha guardato il posto sbagliato e la figura sarebbe uscita a meta'.
    resti = sorted({c for c in MAPPA if c in fuori.upper()}
                   - {v[0].upper() for v in MAPPA.values()})
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
