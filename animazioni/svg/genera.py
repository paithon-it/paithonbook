"""Rigenera le figure animate in SVG.

    python3 animazioni/svg/genera.py                  # tutte
    python3 animazioni/svg/genera.py percettrone-impara xor-non-separabile
    python3 animazioni/svg/genera.py --verifica       # sono allineate?

Ogni generatore è un file `<nome>.py` accanto a questo, che espone `NOME`,
`TITOLO` e `costruisci() -> Figura`. Il nome col trattino non è importabile
come modulo, quindi si carica dal percorso.

Oltre al `.svg` in `book/figures/`, scrive un **provino** PNG in
`~/.cache/paithon-svg/`: è la rasterizzazione dello stato di riposo, cioè
esattamente ciò che vedrà la stampa. Va guardato prima di pubblicare.

`--verifica` esiste perché queste figure sono **prodotti di generazione
tracciati**, come i notebook e come `aggiornamenti.md`, e quindi possono
restare indietro in silenzio: un giro di correzioni tocca i sorgenti, nessuno
rigenera, e la figura pubblicata continua a dire la cosa vecchia. È già
successo (`caffda4` ha tolto le lineette dai sorgenti e per settimane gli SVG
online ne hanno portate tredici). Rigenera in un temporaneo e confronta:
non scrive niente e torna 1 se qualcosa è disallineato.
"""

import argparse
import importlib.util
import sys
import tempfile
from pathlib import Path

QUI = Path(__file__).resolve().parent
RADICE = QUI.parents[1]
FIGURE = RADICE / "book" / "figures"

# Il motore vive nel design system, accanto al tema Manim: è la stessa
# cartella che monta il sito, così le figure delle due superfici non possono
# divergere. Qui resta l'unica cosa che sa del libro: dove vanno le figure.
MOTORE = RADICE / "book" / "_static" / "brand" / "motion"
sys.path.insert(0, str(MOTORE))
sys.path.insert(0, str(QUI))          # i generatori, che stanno accanto a qui

from paithon_svg import PROVINI, scrivi   # noqa: E402


def carica(percorso: Path):
    """Carica un generatore dal percorso, compilando il sorgente ogni volta.

    Il `.pyc` qui non si usa, e non è pignoleria: la cache si invalida su
    (mtime, dimensione) del sorgente, quindi una modifica che non cambia la
    lunghezza e cade nello stesso secondo della compilazione precedente resta
    invisibile. Preso in castagna una volta sola per fortuna, provando
    `--verifica`: `mini-batch` sostituito con `mini-lotto` (dieci caratteri
    entrambi) e Python continuava a eseguire la versione vecchia, con
    `--verifica` che segnalava un disallineamento inesistente. Un controllo
    che si fa ingannare dalla propria cache non serve a niente.
    """
    spec = importlib.util.spec_from_file_location(percorso.stem.replace("-", "_"), percorso)
    mod = importlib.util.module_from_spec(spec)
    codice = compile(percorso.read_text(encoding="utf-8"), str(percorso), "exec")
    exec(codice, mod.__dict__)          # noqa: S102 - sorgente nostro, nel repo
    return mod


def verifica(files) -> int:
    """Rigenera in un temporaneo e confronta, senza toccare book/figures/."""
    disallineate, mancanti, errori = [], [], 0

    with tempfile.TemporaryDirectory(prefix="paithon-svg-verifica-") as tmp:
        prova = Path(tmp)
        for f in files:
            try:
                mod = carica(f)
                # provino=False: qui non si guarda niente, si confronta
                atteso = scrivi(mod.NOME, mod.TITOLO, mod.costruisci(), prova, provino=False)
            except Exception as e:
                errori += 1
                print(f"✗ {f.stem}: {type(e).__name__}: {e}", file=sys.stderr)
                continue

            committata = FIGURE / f"{mod.NOME}.svg"
            if not committata.is_file():
                mancanti.append(mod.NOME)
            elif committata.read_bytes() != atteso.read_bytes():
                disallineate.append(mod.NOME)

    if errori:
        return 1
    if mancanti or disallineate:
        if mancanti:
            print("mai generate: " + ", ".join(sorted(mancanti)))
        if disallineate:
            print("da rigenerare: " + ", ".join(sorted(disallineate)))
        print("  python3 animazioni/svg/genera.py")
        return 1

    print(f"allineate ai generatori: {len(files)} figure")
    return 0


def main(argv):
    ap = argparse.ArgumentParser(description="Genera le figure animate in SVG.")
    ap.add_argument("nomi", nargs="*", metavar="NOME",
                    help="i generatori da eseguire (default: tutti)")
    ap.add_argument("--verifica", action="store_true",
                    help="esce con 1 se le figure sul disco non sono quelle attese")
    argomenti = ap.parse_args(argv)

    scelti = set(argomenti.nomi)
    files = sorted(p for p in QUI.glob("*.py")
                   if p.name != "genera.py"
                   and (not scelti or p.stem in scelti))
    if scelti and not files:
        print(f"nessun generatore per: {', '.join(sorted(scelti))}", file=sys.stderr)
        return 1

    if argomenti.verifica:
        return verifica(files)

    errori = 0
    for f in files:
        try:
            mod = carica(f)
            out = scrivi(mod.NOME, mod.TITOLO, mod.costruisci(), FIGURE)
            peso = out.stat().st_size
            print(f"✓ {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out}"
                  f"  ({peso:,} byte)".replace(",", "."))
        except Exception as e:
            errori += 1
            print(f"✗ {f.stem}: {type(e).__name__}: {e}", file=sys.stderr)

    if files and not errori:
        print(f"\nprovini (aprili con Read, è lo stato che finirà in stampa):\n  {PROVINI}")
    return 1 if errori else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
