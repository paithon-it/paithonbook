#!/usr/bin/env python3
"""
Cerca incoerenze nel libro: riferimenti che non portano da nessuna parte.

Nasce da un'osservazione fatta assorbendo il corpus del magazine: quasi tutte
le sezioni che mancavano davvero non erano temi dimenticati, erano **buchi di
coerenza**. Il libro usava i decoratori senza averli spiegati, rimandava LoRA a
una sezione che non esisteva, insegnava la U bias-varianza e altrove descriveva
modelli da miliardi di parametri che la infrangono.

Quelli si trovavano solo leggendo. Questi si trovano contando, e vale la pena
toglierli di mezzo prima di rileggere:

  numref     riferimenti a figure/tabelle senza un :name: corrispondente
  cite       chiavi bibliografiche assenti dal .bib
  ref/doc    riferimenti interni a target inesistenti
  figure     file in book/figures/ che nessuno richiama
  toc        file sotto book/ non elencati in _toc.yml
  avanti     rimandi in avanti ("vedremo", "prossima sezione") — solo elenco,
             la verifica resta umana

  scripts/coerenza.py            # tutto
  scripts/coerenza.py --solo numref,cite

Un asse PROVATO E SCARTATO, per non rifarlo: "termini usati prima di essere
definiti", ricostruendo l'ordine di lettura dal _toc.yml e usando il grassetto
come marca di introduzione (e' la convenzione del libro). Ha prodotto 474
risultati quasi tutti rumore — parole comuni che capita siano in grassetto
("ottimizzazione", "stessa", "a mano") e la front matter che anticipa tutto il
libro per mestiere. Filtrando su intro/Introduzione e richiedendo che il
termine sia anche un titolo di sezione si scende a 11, e di quegli 11 **nessuno
e' un difetto reale**: l'unico plausibile, `transfer learning`, e' usato in
corsivo tre volte prima del suo capitolo ma gia' glossato in una riga in
DeepLearning/overview.

La differenza col controllo sul codice e' il vocabolario: i costrutti Python
sono un insieme chiuso e riconoscibile, i termini tecnici no. Su questo libro
l'asse dell'ordine di introduzione e' esaurito.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

QUI = Path(__file__).resolve().parent.parent
LIBRO = QUI / "book"

RX = {
    "numref": re.compile(r"\{numref\}`([^`<]+?)`"),
    "ref": re.compile(r"\{ref\}`(?:[^<`]*<)?([^`>]+)>?`"),
    "doc": re.compile(r"\{doc\}`(?:[^<`]*<)?([^`>]+)>?`"),
    "cite": re.compile(r"\{cite\}`([^`]+)`"),
    "name": re.compile(r"^\s*:name:\s*(\S+)", re.M),
    "label": re.compile(r"^\(([^)]+)\)=\s*$", re.M),
    # \S* GREEDY fino all'ultimo slash: con il non-greedy si catturava
    # "figures/nome.svg" invece di "nome.svg" e ogni figura risultava orfana.
    "fig_uso": re.compile(r"\{figure\}\s*\S*/([^\s`/]+)"),
    "img_uso": re.compile(r"!\[[^\]]*\]\(\S*/([^\s)/]+)\)"),
    "avanti": re.compile(
        r"[^.\n]*\b(vedremo|vedrai|prossima sezione|prossimo capitolo|"
        r"piu[' ]avanti|torneremo|approfondiremo)\b[^.\n]*\.", re.I),
}


def sorgenti() -> dict[str, str]:
    # _static/ contiene il submodule `brand` con i suoi README: non e' testo del
    # libro e non va nel _toc.yml.
    return {str(p.relative_to(LIBRO)): p.read_text(encoding="utf-8", errors="ignore")
            for p in sorted(list(LIBRO.rglob("*.md")) + list(LIBRO.rglob("*.ipynb")))
            if "_static" not in p.parts}


def chiavi_bib() -> set[str]:
    chiavi = set()
    for b in LIBRO.rglob("*.bib"):
        chiavi |= set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", b.read_text(
            encoding="utf-8", errors="ignore")))
    return chiavi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo", help="controlli da eseguire, separati da virgola")
    args = ap.parse_args()
    attivi = set(args.solo.split(",")) if args.solo else {
        "numref", "cite", "ref", "figure", "toc", "avanti"}

    testi = sorgenti()
    problemi = defaultdict(list)

    # target definiti ovunque nel libro (i riferimenti sono globali)
    nomi = set()
    for t in testi.values():
        nomi |= set(RX["name"].findall(t))
        nomi |= set(RX["label"].findall(t))

    if "numref" in attivi:
        for f, t in testi.items():
            for r in RX["numref"].findall(t):
                if r.strip() not in nomi:
                    problemi["numref senza :name:"].append(f"{f}  ->  {r}")

    if "ref" in attivi:
        for f, t in testi.items():
            for r in RX["ref"].findall(t) + RX["doc"].findall(t):
                r = r.strip()
                if r in nomi:
                    continue
                # {doc} punta a un file: accetta se esiste
                if any(k.startswith(r.lstrip("/")) or Path(k).stem == Path(r).stem
                       for k in testi):
                    continue
                problemi["ref/doc senza target"].append(f"{f}  ->  {r}")

    if "cite" in attivi:
        bib = chiavi_bib()
        if bib:
            for f, t in testi.items():
                for gruppo in RX["cite"].findall(t):
                    for c in (x.strip() for x in gruppo.split(",")):
                        if c and c not in bib:
                            problemi["cite senza voce nel .bib"].append(f"{f}  ->  {c}")
        else:
            problemi["cite senza voce nel .bib"].append("(nessun .bib trovato)")

    if "figure" in attivi:
        usate = set()
        for t in testi.values():
            usate |= set(RX["fig_uso"].findall(t)) | set(RX["img_uso"].findall(t))
        for p in sorted((LIBRO / "figures").glob("*")):
            if p.is_file() and p.name not in usate:
                problemi["figure mai richiamate"].append(p.name)

    if "toc" in attivi:
        toc = (LIBRO / "_toc.yml")
        if toc.is_file():
            testo_toc = toc.read_text(encoding="utf-8")
            elencati = set(re.findall(r"file:\s*(\S+)", testo_toc))
            elencati |= set(re.findall(r"^root:\s*(\S+)", testo_toc, re.M))
            elencati = {e if "." in Path(e).name else e for e in elencati}
            for f in testi:
                stem = re.sub(r"\.(md|ipynb)$", "", f)
                if stem not in elencati and f not in elencati:
                    problemi["file non nel _toc.yml"].append(f)

    if "avanti" in attivi:
        for f, t in testi.items():
            for frase in RX["avanti"].findall(t):
                pass  # findall coi gruppi non serve: conta il testo intero
            for m in RX["avanti"].finditer(t):
                problemi["rimandi in avanti (da leggere)"].append(
                    f"{f}: …{m.group(0).strip()[:110]}")

    ordine = ["numref senza :name:", "cite senza voce nel .bib",
              "ref/doc senza target", "figure mai richiamate",
              "file non nel _toc.yml", "rimandi in avanti (da leggere)"]
    totale = 0
    for k in ordine:
        v = problemi.get(k)
        if not v:
            continue
        print(f"\n=== {k} ({len(v)})")
        mostra = v if k != "rimandi in avanti (da leggere)" else v[:12]
        for riga in mostra:
            print(f"   {riga}")
        if len(v) > len(mostra):
            print(f"   … e altri {len(v) - len(mostra)}")
        if k != "rimandi in avanti (da leggere)":
            totale += len(v)

    print(f"\n{totale} problemi da correggere"
          f" ({len(problemi.get('rimandi in avanti (da leggere)', []))} rimandi da leggere a mano)")
    return 1 if totale else 0


if __name__ == "__main__":
    sys.exit(main())
