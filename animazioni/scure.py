#!/usr/bin/env python3
"""La resa scura delle clip di Manim, resa e non filtrata.

    python3 animazioni/scure.py             # quelle che mancano o sono indietro
    python3 animazioni/scure.py --verifica  # ne manca una? e' indietro una?
    python3 animazioni/scure.py --solo message-passing
    python3 animazioni/scure.py --tutte      # rende anche quelle gia' allineate

In tema scuro il libro mostra di ogni figura la propria resa scura, e non la
figura chiara appoggiata su una cartolina bianca: la coppia la forma
`book/_ext/pt_scuro.py`, che cerca in `book/figures/scure/` un file con lo
stesso nome. Per gli SVG quel file lo scrive `scripts/genera-figure-scure.py`,
sostituendo i colori uno per uno. Qui ci sono le altre.

## Perche' un secondo render e non un filtro

Perche' una GIF e' un raster, e in un raster non c'e' piu' niente da
sostituire: i colori della marca sono gia' stati fusi nei pixel, e insieme
all'antialiasing dei bordi e dei glifi. Qualunque filtro (invertire, ruotare
la tinta, schiarire) sposta anche i tre accenti, che nella palette scura non
sono l'inverso di quelli chiari ma tre colori scelti apposta. La sola strada
onesta e' **rifare la clip dallo stesso sorgente** con `PAITHON_TEMA=scuro`,
che e' la variabile con cui il tema del brand sceglie palette e sfondo. Il
disegno e' identico; cambiano i colori, e li sceglie chi li ha scelti.

## Perche' l'impronta del sorgente, e non i byte dei due render

Perche' i due file non hanno niente da confrontare: sono due render diversi
della stessa scena, e confrontarli byte a byte non dice se la resa scura e'
vecchia, dice solo che i colori sono diversi. Quello che si vuole sapere e'
**da quale versione della scena** viene, ed e' la stessa domanda a cui
risponde `fermi.py`: al render si registra in `impronte-scure.json` l'impronta
del `.py` da cui la clip e' uscita, e `--verifica` confronta quella.

E il registro copre anche il caso che nessun altro controllo vede: **la clip
chiara non ha un cancello suo**. `fermi.py` confronta i fermi immagine con la
GIF, non la GIF con la scena, quindi chi tocca `animazioni/<nome>.py` e non
ri-renderizza lascia in pagina una clip vecchia senza che niente diventi
rosso. Da qui in avanti quel cambiamento accende almeno questo, e chi lo
ripara ri-renderizza tutt'e due.

## Che cosa NON basta guardare

Che il file esista. Una GIF piu' corta della sua gemella, o piu' stretta, e' un
guasto che in pagina si vede eccome: la figura balla cambiando tema e
l'animazione si taglia. Quindi si confrontano anche **misura e numero di
fotogrammi** delle due; e per lo stesso motivo larghezza e fps del render non
sono scritti qui dentro ma **letti dalla clip chiara**, perche' una del libro
sta a 10 fps e 720 px invece che a 15 e 800, e un valore scritto a mano
l'avrebbe silenziosamente rifatta come le altre.

Sul numero di fotogrammi, pero', il confronto lascia passare **uno**, e non e'
indulgenza. Due render della stessa scena a colori diversi sono due flussi
video diversi, e la durata che il contenitore mp4 si scrive dentro cade a
mezzo millesimo di distanza: 8,366016 s la chiara di `backpropagation`,
8,366667 s la scura, su 251 fotogrammi identici a 30 fps. Ricampionando a 15
quel mezzo millesimo decide fra 125 e 126. Un cancello che si impunta li'
rifiuta un lavoro buono, e un cancello che rifiuta un lavoro buono viene
aggirato alla prima occasione.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys

QUI = pathlib.Path(__file__).resolve().parent
RADICE = QUI.parent
FIGURE = RADICE / "book" / "figures"
SCURE = FIGURE / "scure"
DRIVER = RADICE / ".claude" / "skills" / "anima-manim" / "driver.py"
IMPRONTE = QUI / "impronte-scure.json"

# L'immagine di render. `animazioni/Dockerfile` costruisce `paithon-manim`, che
# e' l'ambiente **del libro** (le versioni con cui le clip sono state fatte);
# il driver, che serve anche il sito, ha come default quella ufficiale. Si
# preferisce la propria quando c'e', si lascia decidere al driver quando non
# c'e', e in tutti e due i casi si stampa quale si e' usata: un render fatto
# con un'altra immagine puo' venire diverso, e non deve poter succedere in
# silenzio.
IMMAGINE_DEL_LIBRO = "paithon-manim"


def impronta(file: pathlib.Path) -> str:
    """L'impronta del contenuto di un sorgente, non la sua data."""
    return hashlib.sha256(file.read_bytes()).hexdigest()[:16]


def registrate() -> dict[str, dict]:
    if not IMPRONTE.is_file():
        return {}
    return json.loads(IMPRONTE.read_text(encoding="utf-8"))


def registra(nome: str, voce: dict) -> None:
    """Aggiorna il registro senza perdere le altre voci."""
    tutte = registrate() | {nome: voce}
    IMPRONTE.write_text(
        json.dumps(dict(sorted(tutte.items())), indent=1, ensure_ascii=False)
        + "\n", encoding="utf-8")


def clip() -> list[tuple[str, pathlib.Path, pathlib.Path]]:
    """(nome, sorgente della scena, clip chiara), per ogni GIF del libro."""
    fuori = []
    for chiara in sorted(FIGURE.glob("*.gif")):
        sorgente = QUI / f"{chiara.stem}.py"
        fuori.append((chiara.stem, sorgente, chiara))
    assert fuori, f"nessuna GIF in {FIGURE}: perimetro sbagliato"
    return fuori


def misura(gif: pathlib.Path) -> tuple[int, int, int]:
    """(larghezza, altezza, fotogrammi) di una GIF, senza ffprobe.

    Senza ffprobe perche' questa e' la meta' che gira nei cancelli, dove
    l'unica dipendenza gia' pagata e' Pillow (la usa `fermi.py`). Il numero di
    fotogrammi al secondo non si ricava di qui in modo affidabile, e infatti
    sta nel registro: la GIF conserva i ritardi in centesimi di secondo, quindi
    15 fps ci diventa 7 cs, che riletto fa 14,3.
    """
    from PIL import Image

    with Image.open(gif) as im:
        return im.width, im.height, getattr(im, "n_frames", 1)


def gemelle(una: tuple[int, int, int],
            altra: tuple[int, int, int]) -> bool:
    """Due clip sono la stessa clip? Misura identica, durata a meno di un
    fotogramma (il perche' del fotogramma sta nel docstring in cima)."""
    return una[:2] == altra[:2] and abs(una[2] - altra[2]) <= 1


def fps_di(gif: pathlib.Path) -> str:
    """Gli fps con cui la clip chiara e' stata scritta, da ffprobe.

    Serve solo al render, dove ffmpeg c'e' per forza: la resa scura deve
    durare quanto la chiara, e una clip del libro non e' a 15 fps.
    """
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(gif)],
        capture_output=True, text=True)
    grezzo = r.stdout.strip()
    numeratore, _, denominatore = grezzo.partition("/")
    valore = int(numeratore) // int(denominatore or 1)
    assert valore > 0, f"{gif.name}: ffprobe non da' gli fps ({grezzo!r})"
    return str(valore)


def immagine_scelta() -> str | None:
    """Il nome dell'immagine Docker, o None per lasciar decidere il driver."""
    if os.environ.get("PAITHON_MANIM_IMAGE"):
        return os.environ["PAITHON_MANIM_IMAGE"]
    presente = subprocess.run(
        ["docker", "image", "inspect", IMMAGINE_DEL_LIBRO],
        capture_output=True, text=True).returncode == 0
    return IMMAGINE_DEL_LIBRO if presente else None


def rendi(nome: str, sorgente: pathlib.Path,
          chiara: pathlib.Path) -> list[str]:
    """Un render in tema scuro, e il collaudo che sia gemella della chiara."""
    larghezza, altezza, fotogrammi = misura(chiara)
    fps = fps_di(chiara)

    ambiente = dict(os.environ)
    scelta = immagine_scelta()
    if scelta:
        ambiente["PAITHON_MANIM_IMAGE"] = scelta

    SCURE.mkdir(parents=True, exist_ok=True)
    comando = [sys.executable, str(DRIVER), "render", str(sorgente),
               "--tema", "scuro", "-o", nome, "--dir", str(SCURE),
               "--larghezza", str(larghezza), "--fps-gif", fps]
    print(f"  {nome}: render a {larghezza} px, {fps} fps "
          f"(immagine {scelta or 'del driver'})")
    esito = subprocess.run(comando, cwd=RADICE, env=ambiente,
                           capture_output=True, text=True)
    if esito.returncode != 0:
        print(esito.stdout[-2000:] + esito.stderr[-2000:])
        return [f"{nome}: il render e' fallito"]

    scura = SCURE / f"{nome}.gif"
    if not scura.is_file():
        return [f"{nome}: il render e' passato ma {scura.name} non c'e'"]

    # Il collaudo, prima di registrare: una resa scura che dura meno della
    # chiara non e' la sua gemella, ed e' meglio lasciare il cancello rosso
    # che scrivere in un registro una cosa che non e' vera.
    sua = misura(scura)
    if not gemelle(sua, (larghezza, altezza, fotogrammi)):
        return [f"{nome}: la resa scura misura {sua} e la chiara "
                f"{(larghezza, altezza, fotogrammi)}"]

    registra(nome, {"impronta": impronta(sorgente),
                    "fotogrammi": fotogrammi,
                    "larghezza": larghezza,
                    "fps": int(fps)})
    print(f"    {scura.relative_to(RADICE)} "
          f"({scura.stat().st_size / 1e6:.2f} MB, {sua[2]} fotogrammi, "
          f"la chiara {fotogrammi})")
    return []


def verifica() -> list[str]:
    """C'e', viene da questa versione della scena, ed e' gemella di lei."""
    problemi = []
    note = registrate()
    viste = set()

    tutte = clip()
    for nome, sorgente, chiara in tutte:
        scura = SCURE / f"{nome}.gif"
        viste.add(scura.name)
        if not sorgente.is_file():
            problemi.append(f"{nome}: la clip c'e' ma il sorgente della scena "
                            f"no ({sorgente.relative_to(RADICE)}): "
                            f"la resa scura non si puo' rifare")
            continue
        if not scura.is_file():
            problemi.append(f"{nome}: manca la resa scura "
                            f"(python3 animazioni/scure.py --solo {nome})")
            continue
        voce = note.get(nome)
        if voce is None:
            problemi.append(f"{nome}: la resa scura non dichiara da quale "
                            f"versione della scena viene (rilancia scure.py)")
        elif voce.get("impronta") != impronta(sorgente):
            problemi.append(f"{nome}: la resa scura viene da una versione "
                            f"precedente di {sorgente.name}")
        sue = misura(chiara)
        if not gemelle(misura(scura), sue):
            problemi.append(
                f"{nome}: la resa scura misura {misura(scura)} e la chiara "
                f"{sue}")

    if SCURE.is_dir():
        for orfana in sorted(SCURE.glob("*.gif")):
            if orfana.name not in viste:
                problemi.append(f"{orfana.name}: resa scura orfana, "
                                f"la clip chiara non c'e' piu'")

    print(f"  lette {len(tutte)} clip in {FIGURE.relative_to(RADICE)}, "
          f"{len(list(SCURE.glob('*.gif'))) if SCURE.is_dir() else 0} scure")
    return problemi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verifica", action="store_true",
                    help="non rende: esce 1 se una resa scura manca o e'"
                         " rimasta indietro")
    ap.add_argument("--tutte", action="store_true",
                    help="rende anche le clip gia' allineate")
    ap.add_argument("--solo", metavar="NOME",
                    help="una sola clip, per nome")
    scelte = ap.parse_args()

    if scelte.verifica:
        problemi = verifica()
        for p in problemi:
            print(f"  {p}")
        if problemi:
            print(f"\n  {len(problemi)} problemi")
            return 1
        print("  allineate")
        return 0

    note = registrate()
    da_fare = []
    for nome, sorgente, chiara in clip():
        if scelte.solo and scelte.solo != nome:
            continue
        if not sorgente.is_file():
            print(f"  {nome}: salto, il sorgente della scena non c'e'")
            continue
        gia = ((SCURE / f"{nome}.gif").is_file()
               and note.get(nome, {}).get("impronta") == impronta(sorgente))
        if gia and not scelte.tutte:
            continue
        da_fare.append((nome, sorgente, chiara))

    if scelte.solo and not da_fare and not scelte.tutte:
        print(f"  {scelte.solo}: gia' allineata (--tutte per rifarla)")
        return 0
    if not da_fare:
        print("  tutte allineate")
        return 0

    print(f"{len(da_fare)} clip da rendere in scuro:")
    problemi = []
    for nome, sorgente, chiara in da_fare:
        problemi += rendi(nome, sorgente, chiara)

    for p in problemi:
        print(f"  {p}")
    return 1 if problemi else 0


if __name__ == "__main__":
    sys.exit(main())
