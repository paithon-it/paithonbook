"""Rigenera le figure animate in SVG.

    python3 animazioni/svg/genera.py                  # tutte
    python3 animazioni/svg/genera.py percettrone-impara xor-non-separabile
    python3 animazioni/svg/genera.py --verifica       # sono allineate?
    python3 animazioni/svg/genera.py --misura NOME    # rimisura il dato di NOME

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

`--misura` serve alle figure che disegnano un **esperimento**, non un calcolo:
là il dato si misura una volta, si committa in `animazioni/dati/` e il disegno
lo legge, perché due CPU con ordini di riduzione BLAS diversi danno numeri
diversi e `--verifica` non potrebbe mai tornare verde altrove. Il generatore
che fa così espone una `misura()`, che riesegue l'esperimento, lo collauda e
riscrive il proprio json; le altre figure non ce l'hanno e `--misura` le salta
dicendolo. È l'unico comando di questo file che tocca qualcosa fuori da
`book/figures/`, e va lanciato apposta: rimisurare cambia la figura.

E perché la separazione non duri finché qualcuno se la ricorda, `--verifica`
guarda anche i sorgenti: un generatore che importa torch deve esporre
`misura()`, e non deve importarlo al livello del modulo. È il cancello che
chiude la classe, e senza di lui il prossimo generatore che addestra nasce con
l'esperimento dentro `costruisci()`, come sono nati questi.
"""

import argparse
import importlib.util
import re
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


def misura(files) -> int:
    """Riesegue gli esperimenti delle figure che ne hanno uno, e ne
    riscrive il dato.

    Non ridisegna niente: dopo, `genera.py <nome>` rifà l'SVG dal dato nuovo.
    Sono due passi apposta, perché sono due decisioni diverse (rimisurare, e
    accettare quello che ne è uscito), e la seconda si prende guardando il
    provino.
    """
    moduli = [(f, carica(f)) for f in files]
    con_esperimento = [(f, m) for f, m in moduli if hasattr(m, "misura")]
    if not con_esperimento:
        nomi = ", ".join(sorted(f.stem for f in files))
        print(f"nessun esperimento da rimisurare in: {nomi}", file=sys.stderr)
        print("  (una figura che disegna un esperimento espone `misura()`)",
              file=sys.stderr)
        return 1

    errori = 0
    for f, mod in con_esperimento:
        try:
            out = mod.misura()
            print(f"✓ {out.relative_to(RADICE)}  ({out.stat().st_size:,} byte)"
                  .replace(",", "."))
        except Exception as e:
            errori += 1
            print(f"✗ {f.stem}: {type(e).__name__}: {e}", file=sys.stderr)

    if not errori:
        nomi = " ".join(f.stem for f, _ in con_esperimento)
        print(f"\nora ridisegna, e guarda il provino:\n"
              f"  python3 animazioni/svg/genera.py {nomi}")
    return 1 if errori else 0


# Una riga che importa torch, al livello di indentazione a cui sta.
IMPORTA_TORCH = re.compile(r"^(\s*)(?:import torch|from torch\b)", re.M)


def impuri(files) -> list[str]:
    """I generatori che addestrano senza aver separato il dato dal disegno.

    È il cancello che chiude la classe, e senza di lui la separazione dura
    finché qualcuno se la ricorda: il prossimo generatore che addestra nasce
    come sono nati questi tre, cioè con `costruisci()` che esegue la rete, e
    la cosa si scopre mesi dopo su un runner che cambia CPU. Due difetti, e
    stanno tutti e due nel sorgente, quindi si vedono senza eseguire niente:

    - importa torch e non espone `misura()`: l'esperimento non è stato
      separato affatto;
    - importa torch **al livello del modulo**: allora l'import lo paga anche
      chi disegna, e `costruisci()` può usarlo senza che si veda.

    Il perimetro sono i generatori passati, e il chiamante ne stampa il
    numero: uno zero che viene da una glob vuota assomiglia troppo a uno zero
    che viene da un libro pulito.
    """
    guasti = []
    for f in files:
        sorgente = f.read_text(encoding="utf-8")
        righe = IMPORTA_TORCH.findall(sorgente)
        if not righe:
            continue
        if "\ndef misura(" not in sorgente:
            guasti.append(
                f"{f.stem}: addestra e non espone `misura()`. Il disegno deve "
                f"essere una funzione pura del dato: l'esperimento si esegue "
                f"una volta, il suo risultato si committa in "
                f"animazioni/dati/{f.stem}.json e `costruisci()` legge quello. "
                f"Il procedimento sta in animazioni/README.md")
        if any(indentazione == "" for indentazione in righe):
            guasti.append(
                f"{f.stem}: importa torch al livello del modulo, quindi lo "
                f"importa anche chi disegna. L'import va dentro "
                f"`esperimento()`, dove serve")
    return guasti


def verifica(files) -> int:
    """Rigenera in un temporaneo e confronta, senza toccare book/figures/."""
    disallineate, mancanti, errori = [], [], 0

    guasti = impuri(files)
    for g in guasti:
        print(f"✗ {g}", file=sys.stderr)

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

    if errori or guasti:
        return 1
    if mancanti or disallineate:
        if mancanti:
            print("mai generate: " + ", ".join(sorted(mancanti)))
        if disallineate:
            print("da rigenerare: " + ", ".join(sorted(disallineate)))
        print("  python3 animazioni/svg/genera.py")
        return 1

    print(f"allineate ai generatori: {len(files)} figure, "
          f"e chi addestra disegna dal proprio dato")
    return 0


def main(argv):
    ap = argparse.ArgumentParser(description="Genera le figure animate in SVG.")
    ap.add_argument("nomi", nargs="*", metavar="NOME",
                    help="i generatori da eseguire (default: tutti)")
    ap.add_argument("--verifica", action="store_true",
                    help="esce con 1 se le figure sul disco non sono quelle attese")
    ap.add_argument("--misura", action="store_true",
                    help="riesegue l'esperimento delle figure che ne hanno uno "
                         "e riscrive il loro dato in animazioni/dati/")
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
    if argomenti.misura:
        return misura(files)

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
