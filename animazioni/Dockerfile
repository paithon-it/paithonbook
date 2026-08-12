# Ambiente di render delle clip Manim del libro.
#
# Perche' esiste, visto che c'e' gia' l'immagine ufficiale `manimcommunity/manim`.
# Perche' quella e' un ambiente generico e questo e' l'ambiente **del libro**:
# porta dentro le versioni con cui le clip sono state fatte, e soprattutto porta
# dentro la conoscenza che altrimenti va riscoperta ogni volta. In questa
# macchina `manim` non c'era, il pip diretto falliva in compilazione, e per
# arrivare a un render funzionante sono serviti quattro pacchetti di sviluppo e
# un LaTeX: mezz'ora buttata, e la prossima volta sarebbe stata la stessa
# mezz'ora. Qui quella mezz'ora e' scritta.
#
#   docker build -t paithon-manim animazioni/
#   PAITHON_MANIM_IMAGE=paithon-manim \
#       python3 .claude/skills/anima-manim/driver.py render animazioni/<scena>.py
#
# Oppure a mano, che e' utile sapere perche' il driver non e' sempre disponibile:
#
#   docker run --rm -u $(id -u):$(id -g) \
#       -v "$PWD:/work" -v "$PWD/book/_static/brand:/brand:ro" -w /work \
#       -e PYTHONPATH=/brand/motion -e HOME=/tmp \
#       paithon-manim manim render -qm --media_dir /tmp/media \
#       animazioni/diffusione-denoising.py DiffusioneDenoising
#
# Il `-u` non e' un vezzo: senza, i file escono di root e il repo si sporca; e
# con l'immagine ufficiale succede il contrario (il container gira come uid
# 1000 e non riesce a scrivere in una cache creata da root). E' il difetto per
# cui il driver falliva con «/manim/media: permesso negato».

FROM python:3.12-slim-bookworm

# Le dipendenze di sistema, in tre gruppi e non in uno, perche' i tre gruppi
# rispondono a tre domande diverse quando qualcosa si rompe.
#
#  1. compilare le estensioni di manim (pycairo e pangocffi vogliono gli header,
#     non le sole librerie: e' qui che `pip install manim` fallisce da nudo);
#  2. comporre le formule: manim scrive LaTeX, lo compila e converte il dvi in
#     svg con dvisvgm. `texlive-latex-extra` serve per la classe `standalone`,
#     che e' quella che manim usa;
#  3. scrivere i video: ffmpeg. Manim >= 0.19 scrive gli mp4 con PyAV, ma GIF e
#     striscia per la stampa le fa ffmpeg, e le ricette stanno in `driver.py`.
RUN apt-get update && apt-get install --no-install-recommends -y \
        build-essential pkg-config \
        libcairo2-dev libpango1.0-dev libgirepository1.0-dev \
        texlive-latex-base texlive-latex-extra texlive-fonts-recommended \
        dvisvgm \
        ffmpeg \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# La versione si passa da fuori: `--build-arg MANIM=0.20.1` ricostruisce
# l'ambiente con cui e' stata fatta la prima ondata di clip. Il default e' la
# versione con cui il render e' stato verificato da capo a fondo in questo
# repository.
ARG MANIM=0.21.0
RUN pip install --no-cache-dir "manim==${MANIM}"

# Il motore del design system (tema, font, bollo) NON entra nell'immagine: e'
# un submodule che vive di vita propria e si monta a `/brand`. Se finisse qui
# dentro, l'immagine invecchierebbe a ogni commit del brand e le clip
# smetterebbero di essere identiche a quelle del sito, che e' esattamente la
# cosa che il submodule esiste per impedire.
ENV PYTHONPATH=/brand/motion \
    MPLBACKEND=Agg

WORKDIR /work

# Nessun ENTRYPOINT: il driver passa `--entrypoint manim`, e a mano si vuole
# poter lanciare anche ffmpeg o una shell senza combattere con l'immagine.
CMD ["manim", "--version"]
