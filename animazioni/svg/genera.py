"""Rigenera le figure animate in SVG.

    python3 animazioni/svg/genera.py                  # tutte
    python3 animazioni/svg/genera.py percettrone-impara xor-non-separabile

Ogni generatore è un file `<nome>.py` accanto a questo, che espone `NOME`,
`TITOLO` e `costruisci() -> Figura`. Il nome col trattino non è importabile
come modulo, quindi si carica dal percorso.

Oltre al `.svg` in `book/figures/`, scrive un **provino** PNG in
`~/.cache/paithon-svg/`: è la rasterizzazione dello stato di riposo, cioè
esattamente ciò che vedrà la stampa. Va guardato prima di pubblicare.
"""

import importlib.util
import sys
from pathlib import Path

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))

from paithon_svg import PROVINI, scrivi   # noqa: E402


def carica(percorso: Path):
    spec = importlib.util.spec_from_file_location(percorso.stem.replace("-", "_"), percorso)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv):
    scelti = set(argv)
    files = sorted(p for p in QUI.glob("*.py")
                   if p.name not in ("genera.py", "paithon_svg.py")
                   and (not scelti or p.stem in scelti))
    if scelti and not files:
        print(f"nessun generatore per: {', '.join(sorted(scelti))}", file=sys.stderr)
        return 1

    errori = 0
    for f in files:
        try:
            mod = carica(f)
            out = scrivi(mod.NOME, mod.TITOLO, mod.costruisci())
            peso = out.stat().st_size
            print(f"✓ {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out}"
                  f"  ({peso:,} byte)".replace(",", "."))
        except Exception as e:
            errori += 1
            print(f"✗ {f.stem}: {type(e).__name__}: {e}", file=sys.stderr)

    if files and not errori:
        print(f"\nprovini (aprili con Read — è lo stato che finirà in stampa):\n  {PROVINI}")
    return 1 if errori else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
