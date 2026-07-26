import pathlib

import torch
from PIL import Image

# --- un dataset di immagini in miniatura -------------------------------------
# La pagina "Dati su misura" parte da una cartella di fotografie proprie, con
# una sottocartella per classe. Qui quelle cartelle si creano: sei immagini
# minuscole generate a colori pieni, quel tanto che basta perche' ImageFolder,
# le trasformazioni e il DataLoader abbiano qualcosa da masticare.
COLORI = {"pizza": (200, 80, 40), "bistecca": (120, 40, 40), "sushi": (230, 220, 200)}
for parte in ("addestramento", "test"):
    for classe, colore in COLORI.items():
        cartella = pathlib.Path("dati") / parte / classe
        cartella.mkdir(parents=True, exist_ok=True)
        for n in range(3):
            Image.new("RGB", (64, 64), colore).save(cartella / f"img_{n}.jpg")

# --- i nomi che le pagine del capitolo danno per esistenti -------------------
# `dati` e `dati_test` compaiono nel testo senza essere costruiti: nel libro il
# punto e' l'API del DataLoader, non da dove arrivano i file.
from torchvision import datasets, transforms  # noqa: E402  (dopo la creazione dei file)

_preparazione = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
dati_train = datasets.ImageFolder(root="dati/addestramento", transform=_preparazione)
dati_test = datasets.ImageFolder(root="dati/test", transform=_preparazione)
dati = dati_train

# Un modello e un ingresso minimi, per i blocchi che li usano di passaggio.
modello = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, 3))
x = torch.randn(2, 3, 32, 32)
