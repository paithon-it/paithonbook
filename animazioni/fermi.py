#!/usr/bin/env python3
"""I tre fermi immagine di ogni animazione, per la stampa.

    python3 animazioni/fermi.py             # scrive i PNG
    python3 animazioni/fermi.py --verifica  # sono allineati alle animazioni?
    python3 animazioni/fermi.py --provino   # i fogli a contatto, da guardare
    python3 animazioni/fermi.py --solo dropout

Trentacinque figure del libro si muovono: 26 SVG con `@keyframes` e 9 GIF di
Manim. In un PDF il movimento non c'e', e senza far niente il lettore vede lo
stato finale (per l'SVG e' una regola del generatore) o un buco (per la GIF).
Tre fotogrammi affiancati non sono l'animazione, ma dicono che **c'era un
prima e un dopo**, che e' l'informazione che il fermo immagine da' solo perde.

## Perche' non a 0% e 100%

Perche' verrebbero due fotogrammi uguali su tre. Le animazioni SVG del libro
sono costruite in modo che **il disegno fermo sia lo stato finale**: partono
dall'inverso e finiscono sull'identita', cosi' chi non anima (la stampa, chi
ha chiesto meno movimento al sistema) vede la figura conclusa. Quindi al 100%
si e' allo stato di riposo, e allo 0% si e' nell'inverso esatto, che spesso
vuol dire "tutto invisibile". I default (10%, 50%, 90%) prendono un inizio che
si e' gia' mosso, una meta' e una fine appena prima che si richiuda il ciclo.

Dove non bastano c'e' `fermi.toml`, che si scrive **dopo aver guardato il
provino**, non prima.

## Perche' sono tracciati

Perche' quale istante rappresenta bene un'animazione e' una decisione, e le
decisioni si guardano. Come per i notebook compagni, il rischio e' che restino
indietro in silenzio: per questo c'e' `--verifica`, e per questo controlla
anche che i tre fotogrammi **siano diversi fra loro**. Tre immagini identiche
passerebbero qualunque controllo sull'esistenza dei file e non
racconterebbero niente.

## Perche' un'impronta e non una data

Perche' le date, qui, mentono in tutti e due i modi in cui si possono leggere.

La **data di modifica** del file non sopravvive a un `git checkout`: in CI
l'albero viene scritto tutto nello stesso istante, quindi ogni file risulta
coetaneo di ogni altro e il confronto diventa un sorteggio. In locale il
controllo passava e sul runner dichiarava «22 fermi immagine piu' vecchi
dell'animazione» su un albero che nessuno aveva toccato.

La **data del commit** e' meglio ma non basta: se i PNG e la loro animazione
finiscono in due commit diversi (succede quando un `git add` largo di un'altra
sessione porta dentro i file a meta' lavoro), i fermi possono risultare
committati *prima* della figura da cui sono stati generati, e il controllo
grida al lupo su una terzina perfettamente allineata.

Quello che si vuole sapere non e' quando, e' **da quale versione**: per questo
`genera()` registra in `impronte-fermi.json` l'impronta del contenuto
dell'animazione che ha appena fotografato, e `--verifica` confronta quella. Un
controllo che segnala cose che non sono difetti viene disattivato, ed e' peggio
di non averlo.
"""

import argparse
import hashlib
import json
import pathlib
import sys
import tomllib

RADICE = pathlib.Path(__file__).resolve().parent.parent
FIGURE = RADICE / "book" / "figures"
FERMI = FIGURE / "fermi"
CONFIGURAZIONE = pathlib.Path(__file__).resolve().parent / "fermi.toml"

# Le frazioni del ciclo a cui si scatta. Vedi il docstring: non 0 e 1.
ISTANTI = (0.10, 0.50, 0.90)

# Quanto grandi escono i PNG: 3 volte la misura del disegno, che a una figura
# larga 700 px stampata su mezza pagina A4 fa circa 300 dpi.
SCALA = 3

sys.path.insert(0, str(RADICE / "book" / "_ext"))


def eccezioni() -> dict[str, list[float]]:
    """Gli istanti scelti a mano, da `fermi.toml`."""
    if not CONFIGURAZIONE.exists():
        return {}
    dati = tomllib.loads(CONFIGURAZIONE.read_text(encoding="utf-8"))
    return {nome: voce["istanti"] for nome, voce in dati.items()
            if "istanti" in voce}


def animazioni() -> list[pathlib.Path]:
    """Tutto cio' che si muove: le GIF, e gli SVG che portano un @keyframes.

    Il controllo e' sul contenuto e non sul nome perche' degli SVG del libro
    se ne muovono 26 su 311, e nel nome non c'e' niente che lo dica.
    """
    trovate = sorted(FIGURE.glob("*.gif"))
    for svg in sorted(FIGURE.glob("*.svg")):
        if "@keyframes" in svg.read_text(encoding="utf-8", errors="ignore"):
            trovate.append(svg)
    return trovate


def istanti_di(nome: str, scelte: dict[str, list[float]]) -> tuple[float, ...]:
    return tuple(scelte.get(nome, ISTANTI))


def fermi_gif(file: pathlib.Path, istanti, dove: pathlib.Path) -> list[pathlib.Path]:
    """Tre PNG dai fotogrammi di una GIF."""
    from PIL import Image

    usciti = []
    with Image.open(file) as immagine:
        totale = getattr(immagine, "n_frames", 1)
        for numero, frazione in enumerate(istanti, start=1):
            immagine.seek(min(int(totale * frazione), totale - 1))
            uscita = dove / f"{file.stem}-{numero}.png"
            immagine.convert("RGB").save(uscita)
            usciti.append(uscita)
    return usciti


def fermi_svg(pagina, file: pathlib.Path, istanti,
              dove: pathlib.Path) -> list[pathlib.Path]:
    """Tre PNG da un SVG animato in CSS.

    `document.getAnimations()` da' le animazioni vive: si mettono in pausa e
    si sposta `currentTime`. E' l'unico modo di fermarne una su un fotogramma
    preciso; uno screenshot e basta prende l'istante in cui capita.
    """
    from pt_stampa import PAGINA, facce_font, misura

    svg = file.read_text(encoding="utf-8")
    larghezza, altezza = misura(svg)
    brand = RADICE / "book" / "_static" / "brand"

    temporanea = dove / f".{file.stem}.html"
    temporanea.write_text(
        PAGINA.format(facce=facce_font(brand), svg=svg,
                      w=larghezza, h=altezza), encoding="utf-8")

    pagina.set_viewport_size({"width": int(larghezza), "height": int(altezza)})
    pagina.goto(temporanea.as_uri())
    durata = pagina.evaluate("""() => {
        const vive = document.getAnimations();
        if (!vive.length) return 0;
        const t = vive[0].effect.getComputedTiming().duration;
        return typeof t === 'number' ? t : 0;
    }""")

    usciti = []
    for numero, frazione in enumerate(istanti, start=1):
        pagina.evaluate("""(t) => {
            for (const a of document.getAnimations()) {
                a.pause();
                a.currentTime = t;
            }
        }""", durata * frazione)
        uscita = dove / f"{file.stem}-{numero}.png"
        pagina.screenshot(path=str(uscita), omit_background=True)
        usciti.append(uscita)

    temporanea.unlink(missing_ok=True)
    return usciti


def uguali(uno: pathlib.Path, due: pathlib.Path) -> bool:
    """Due fermi sono la stessa immagine?"""
    from PIL import Image, ImageChops

    with Image.open(uno) as a, Image.open(due) as b:
        if a.size != b.size:
            return False
        return ImageChops.difference(a.convert("RGB"),
                                     b.convert("RGB")).getbbox() is None


IMPRONTE = pathlib.Path(__file__).resolve().parent / "impronte-fermi.json"


def impronta(file: pathlib.Path) -> str:
    """L'impronta del contenuto di un'animazione, non la sua data."""
    return hashlib.sha256(file.read_bytes()).hexdigest()[:16]


def impronte_registrate() -> dict[str, str]:
    if not IMPRONTE.is_file():
        return {}
    try:
        return json.loads(IMPRONTE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def registra(nomi: dict[str, str]) -> None:
    """Aggiorna il registro delle impronte, senza perdere le altre voci."""
    tutte = impronte_registrate() | nomi
    IMPRONTE.write_text(
        json.dumps(dict(sorted(tutte.items())), indent=1, ensure_ascii=False)
        + "\n", encoding="utf-8")


def genera(solo: str | None = None) -> list[str]:
    from playwright.sync_api import sync_playwright

    FERMI.mkdir(parents=True, exist_ok=True)
    scelte = eccezioni()
    da_fare = [f for f in animazioni() if not solo or solo in f.stem]
    problemi = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        pagina = browser.new_page(device_scale_factor=SCALA)
        for file in da_fare:
            istanti = istanti_di(file.stem, scelte)
            if file.suffix == ".gif":
                usciti = fermi_gif(file, istanti, FERMI)
            else:
                usciti = fermi_svg(pagina, file, istanti, FERMI)
            gemelli = [f"{a.name} e {b.name}"
                       for a, b in ((usciti[0], usciti[1]),
                                    (usciti[1], usciti[2]),
                                    (usciti[0], usciti[2]))
                       if uguali(a, b)]
            marca = "  <- due fotogrammi uguali" if gemelli else ""
            print(f"  {file.stem:38} {' '.join(f'{i:.0%}' for i in istanti)}{marca}")
            problemi += [f"{file.stem}: {g} sono identici" for g in gemelli]
            registra({file.name: impronta(file)})
        pagina.close()
        browser.close()

    return problemi


def verifica() -> list[str]:
    """I fermi ci sono, sono aggiornati, e non sono tre volte la stessa cosa."""
    problemi = []
    viste = set()
    scelte = eccezioni()

    for file in animazioni():
        istanti = istanti_di(file.stem, scelte)
        tre = [FERMI / f"{file.stem}-{n}.png" for n in (1, 2, 3)]
        viste.update(f.name for f in tre)
        mancanti = [f.name for f in tre if not f.exists()]
        if mancanti:
            problemi.append(f"{file.stem}: mancano {', '.join(mancanti)}")
            continue
        registrata = impronte_registrate().get(file.name)
        if registrata is None:
            problemi.append(f"{file.stem}: i fermi non dichiarano da quale "
                            f"animazione vengono (rilancia fermi.py)")
        elif registrata != impronta(file):
            problemi.append(f"{file.stem}: i fermi vengono da una versione "
                            f"precedente dell'animazione")
        for a, b in ((tre[0], tre[1]), (tre[1], tre[2]), (tre[0], tre[2])):
            if uguali(a, b):
                problemi.append(f"{file.stem}: {a.name} e {b.name} sono "
                                f"identici, gli istanti non dicono niente")
        del istanti

    if FERMI.exists():
        for orfano in sorted(FERMI.glob("*.png")):
            if orfano.name not in viste:
                problemi.append(f"{orfano.name}: fermo orfano, l'animazione "
                                f"non c'e' piu'")
    return problemi


def provino(per_foglio: int = 6) -> list[pathlib.Path]:
    """I fogli a contatto: le terzine incolonnate, da guardare davvero."""
    from PIL import Image, ImageDraw

    alta, margine, testo = 200, 18, 26
    fogli = []
    tutte = animazioni()

    for indice in range(0, len(tutte), per_foglio):
        gruppo = tutte[indice:indice + per_foglio]
        righe = []
        for file in gruppo:
            tre = [FERMI / f"{file.stem}-{n}.png" for n in (1, 2, 3)]
            if not all(f.exists() for f in tre):
                continue
            miniature = []
            for f in tre:
                with Image.open(f) as im:
                    im = im.convert("RGB")
                    larga = max(1, int(im.width * alta / im.height))
                    miniature.append(im.resize((larga, alta), Image.LANCZOS))
            righe.append((file.stem, miniature))

        if not righe:
            continue
        larghezza = max(sum(m.width for m in mm) + margine * 4
                        for _, mm in righe)
        altezza = sum(alta + testo + margine for _ in righe) + margine
        foglio = Image.new("RGB", (larghezza, altezza), "#F8F5EE")
        penna = ImageDraw.Draw(foglio)
        y = margine
        for nome, miniature in righe:
            penna.text((margine, y), nome, fill="#1A1A1A")
            x = margine
            for m in miniature:
                foglio.paste(m, (x, y + testo))
                x += m.width + margine
            y += alta + testo + margine
        dove = FERMI / f".provino-{indice // per_foglio + 1}.png"
        foglio.save(dove)
        fogli.append(dove)
    return fogli


def main() -> None:
    argomenti = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    argomenti.add_argument("--verifica", action="store_true")
    argomenti.add_argument("--provino", action="store_true")
    argomenti.add_argument("--solo", help="una sola animazione, per nome")
    scelte = argomenti.parse_args()

    if scelte.verifica:
        problemi = verifica()
        for p in problemi:
            print(f"  {p}")
        print(f"\n{len(animazioni())} animazioni, "
              f"{len(problemi)} problemi")
        sys.exit(1 if problemi else 0)

    if scelte.provino:
        for foglio in provino():
            print(f"  {foglio}")
        return

    print(f"{len(animazioni())} animazioni:")
    problemi = genera(solo=scelte.solo)
    if problemi:
        print(f"\n{len(problemi)} terzine da guardare "
              f"(si aggiustano in animazioni/fermi.toml):")
        for p in problemi:
            print(f"  {p}")


if __name__ == "__main__":
    main()
