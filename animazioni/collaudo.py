#!/usr/bin/env python3
"""Collaudo delle clip Manim: girano ancora tutte, con questo ambiente?

    python3 animazioni/collaudo.py            # dice cosa farebbe
    python3 animazioni/collaudo.py --esegui   # renderizza tutte le scene
    python3 animazioni/collaudo.py --verifica # l'ambiente e' quello collaudato?

## Perche' esiste

Le clip Manim sono l'unica cosa del libro che **non si puo' verificare
confrontando il prodotto con il sorgente**: sono video, rigenerarli costa
minuti, e due render della stessa scena non danno file identici. Per il resto
del libro il controllo c'e' (`genera-notebook.py --verifica`, `genera.py
--verifica`, `genera-aggiornamenti.py --verifica`); qui no.

`coerenza.py --solo clip` copre meta' del problema: si accorge se un **sorgente**
e' cambiato dopo il suo `.gif`. Questo copre l'altra meta', che e' peggiore
perche' non lascia tracce in git: **l'ambiente** cambia sotto le clip. Una
versione nuova di Manim rinomina una classe, un LaTeX nuovo compone una formula
in un altro modo, e le scene smettono di partire senza che nessun file del
repository sia stato toccato. Non e' teorico: `manim` in questa macchina non
c'era affatto, e il driver in Docker falliva per un disallineamento di uid.

## Come funziona

`ambiente.json` accanto a questo file e' il **manifesto**: le versioni con cui
le scene sono state viste girare, e l'esito di ciascuna. `--verifica` legge le
versioni dall'immagine e le confronta col manifesto: se combaciano non c'e'
niente da fare, se no dice **che cosa** e' cambiato e chiede il collaudo. E' la
stessa idea del `lastmod` della sitemap: non si confronta il contenuto, si
confronta l'unica cosa che si puo' confrontare a costo zero.

Il render del collaudo e' a bassa qualita' e non tocca `book/figures/`: la
domanda e' «parte?», non «e' bella?». Le clip buone le fa il driver.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

QUI = Path(__file__).resolve().parent
REPO = QUI.parent
MANIFESTO = QUI / "ambiente.json"
IMMAGINE = "paithon-manim"


def scene() -> list[tuple[Path, str]]:
    """(file, nome della classe) per ogni scena. Una scena per file, come da CLAUDE.md."""
    fuori = []
    for f in sorted(QUI.glob("*.py")):
        if f.name in ("collaudo.py",):
            continue
        m = re.search(r"^class\s+(\w+)\s*\(\s*ScenaPaithon\s*\)", f.read_text(), re.M)
        if m:
            fuori.append((f, m.group(1)))
    return fuori


def versioni() -> dict[str, str]:
    """Le versioni che contano, lette dall'immagine e non da questa macchina."""
    cmd = ["docker", "run", "--rm", IMMAGINE, "sh", "-c",
           "manim --version; python -c 'import sys;print(sys.version.split()[0])'; "
           "latex --version | head -1; ffmpeg -version | head -1"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return {}
    righe = [x.strip() for x in r.stdout.splitlines() if x.strip()]
    def prendi(rx, testo, default="?"):
        m = re.search(rx, testo)
        return m.group(1) if m else default
    tutto = "\n".join(righe)
    return {
        "manim": prendi(r"Manim Community v([\d.]+)", tutto),
        "python": prendi(r"^(\d+\.\d+\.\d+)$", tutto, righe[1] if len(righe) > 1 else "?"),
        "latex": prendi(r"(TeX Live \d+)", tutto),
        "ffmpeg": prendi(r"ffmpeg version (\S+)", tutto),
        "immagine": IMMAGINE,
    }


def render(f: Path, classe: str, tmp: Path) -> tuple[bool, float, str]:
    cmd = [
        "docker", "run", "--rm",
        "-u", f"{__import__('os').getuid()}:{__import__('os').getgid()}",
        "-v", f"{REPO}:/work",
        "-v", f"{REPO / 'book/_static/brand'}:/brand:ro",
        "-v", f"{tmp}:/media",
        "-w", "/work", "-e", "HOME=/tmp", "-e", "PYTHONPATH=/brand/motion",
        "--entrypoint", "manim", IMMAGINE, "render",
        "-ql", "--media_dir", "/media",
        f"animazioni/{f.name}", classe,
    ]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode == 0:
        return True, dt, ""
    coda = [x for x in (r.stderr or r.stdout).splitlines() if x.strip()][-3:]
    return False, dt, " / ".join(x.strip()[:120] for x in coda)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--esegui", action="store_true", help="renderizza tutte le scene")
    ap.add_argument("--verifica", action="store_true",
                    help="l'ambiente e' quello del manifesto?")
    args = ap.parse_args()

    vecchio = json.loads(MANIFESTO.read_text()) if MANIFESTO.is_file() else {}
    v = versioni()
    if not v:
        print("!! l'immagine non risponde. Costruiscila:")
        print("   docker build -t paithon-manim animazioni/")
        return 1

    if args.verifica:
        atteso = vecchio.get("versioni", {})
        cambiate = {k: (atteso.get(k), v[k]) for k in v if atteso.get(k) != v[k]}
        if not atteso:
            print("nessun manifesto: lancia --esegui")
            return 1
        if not cambiate:
            print(f"ambiente invariato ({v['manim']}, {v['latex']}): "
                  f"{len(vecchio.get('scene', {}))} scene collaudate il "
                  f"{vecchio.get('data', '?')}")
            return 0
        print("l'ambiente e' cambiato, e le clip non sono state riprovate:")
        for k, (a, b) in cambiate.items():
            print(f"   {k}: {a}  ->  {b}")
        print("\n   python3 animazioni/collaudo.py --esegui")
        return 1

    elenco = scene()
    if not args.esegui:
        print(f"{len(elenco)} scene, ambiente {v['manim']} / {v['latex']}")
        for f, c in elenco:
            print(f"   {f.name}  ->  {c}")
        print("\nrilancia con --esegui")
        return 0

    import tempfile
    esiti, rotte = {}, 0
    with tempfile.TemporaryDirectory() as tmp:
        for f, c in elenco:
            ok, dt, err = render(f, c, Path(tmp))
            esiti[f.name] = {"classe": c, "ok": ok, "secondi": round(dt, 1)}
            if not ok:
                esiti[f.name]["errore"] = err
                rotte += 1
            print(f"  {'OK ' if ok else '!! '}{f.name:34s} {dt:6.1f}s"
                  + ("" if ok else f"\n       {err}"))

    MANIFESTO.write_text(json.dumps({
        "_": "Manifesto dell'ambiente di render. Lo scrive animazioni/collaudo.py.",
        "data": time.strftime("%Y-%m-%d"),
        "versioni": v,
        "scene": esiti,
    }, indent=2, ensure_ascii=False) + "\n")
    print(f"\n{len(elenco)} scene, {rotte} rotte. Manifesto aggiornato.")
    return 1 if rotte else 0


if __name__ == "__main__":
    sys.exit(main())
