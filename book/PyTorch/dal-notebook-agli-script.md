# Dal notebook agli script

Un **notebook** è il quaderno interattivo con cui si lavora quasi sempre
quando si sperimenta: una pagina divisa in **celle**, ciascuna con dentro un
pezzo di codice, che si eseguono una alla volta premendo un tasto e che
lasciano il risultato stampato lì sotto. È lo stesso oggetto che si apre
premendo «Esegui il codice» in cima a queste pagine. La comodità è enorme: si
prova una riga, si guarda il numero, si cambia. E il difetto nasce esattamente
da lì, perché le celle si possono eseguire in qualunque ordine, anche in uno
diverso da quello in cui sono scritte.

C'è un esperimento che ogni tanto conviene fare sul proprio notebook preferito:
premere *Restart & Run All*, cioè «ricomincia da zero ed esegui tutto in
ordine», e guardare che cosa succede. Molto spesso non
succede niente di buono. La cella 43 usa una variabile definita nella cella 12,
che nel frattempo è stata cancellata; la funzione buona è la terza versione,
ma le prime due sono ancora lì sotto; il modello che ha dato il risultato
migliore è stato addestrato con un learning rate che nessuno ha annotato, e
che ora non è più nel codice. Il notebook non è rotto: è che era un
**laboratorio**, e a un certo punto il laboratorio va trasformato in un
prodotto.

Questa sezione mostra come, restando dentro PyTorch e senza aggiungere alcuno
strumento: cinque file di Python semplice, e un comando che si lancia dal
terminale. Il che comincia a fare venire in mente la produzione: argomento del
[capitolo sull'MLOps](../MLOps/dal-notebook-alla-produzione.md), che riprende
il discorso da qui in poi.

## Il laboratorio e il prodotto

`````{tab} Elementare
Un notebook è una cucina di prova: assaggi, aggiungi, ributti, tieni tre
pentole sul fuoco. È lo strumento giusto per capire se un'idea funziona,
proprio perché non ti obbliga a essere ordinato. Uno script è invece la ricetta
scritta: chiunque la legga ottiene lo stesso piatto, nello stesso ordine,
senza doverti chiedere niente.

Il segnale che è arrivato il momento di passare dall'uno all'altro è sempre lo
stesso: **quando cominci a rilanciare la stessa cosa cambiando un numero**.
Cinque prove con cinque learning rate diversi, fatte modificando a mano una
cella, sono cinque esecuzioni di cui domani non ricorderai la differenza. Le
stesse cinque prove lanciate da terminale con `--lr 0.01`, `--lr 0.001` e così
via restano scritte nella cronologia del terminale: sono un esperimento, non
un ricordo.
`````

`````{tab} Superiore
La differenza sostanziale è tra **stato implicito** e **stato esplicito**. Nel
notebook lo stato vive nel kernel: l'ordine di esecuzione delle celle è
invisibile nel documento salvato, quindi il codice non determina il risultato
(condizione che rende impossibile la riproducibilità). Uno script ha un unico
punto d'ingresso, un ordine totale delle istruzioni, e tutto ciò che varia
passa dagli argomenti della riga di comando: input, output e parametri sono
dichiarati.

Ne discendono tre proprietà che il notebook non può avere: si mette sotto
controllo di versione in modo leggibile (il `.ipynb` è un JSON con dentro gli
output, e un `diff` è illeggibile); si mette in una pipeline di CI o in uno
scheduler; e si testa, perché ogni funzione è importabile da un test senza
eseguire tutto il resto.
`````

## Cinque file, cinque responsabilità

La divisione che segue è quella che si ritrova, con nomi diversi, nella
maggior parte dei progetti PyTorch. Non c'è nulla di magico: è la stessa
struttura del capitolo, resa file.

```text
progetto/
├── data_setup.py     # dai file ai DataLoader
├── model_builder.py  # la definizione del modello
├── engine.py         # un'epoca di addestramento, un'epoca di valutazione
├── utils.py          # salvataggio, semi, funzioni di servizio
└── train.py          # il punto d'ingresso: mette insieme gli altri quattro
```

Il pezzo centrale è `engine.py`: il training loop della sezione
[sull'addestramento](addestramento.md), estratto in due funzioni che non sanno
nulla del problema specifico e che quindi si riusano ovunque.

```python
# engine.py
import torch

def passo_addestramento(modello, loader, criterio, ottimizzatore, device):
    """Una epoca di addestramento. Restituisce (perdita media, accuratezza)."""
    modello.train()
    perdita_tot, corretti, totale = 0.0, 0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logit = modello(X)
        perdita = criterio(logit, y)

        ottimizzatore.zero_grad()
        perdita.backward()
        ottimizzatore.step()

        perdita_tot += perdita.item() * X.size(0)   # .item(): niente grafo trattenuto
        corretti += (logit.argmax(dim=1) == y).sum().item()
        totale += X.size(0)

    return perdita_tot / totale, corretti / totale


@torch.no_grad()                                    # decoratore: niente gradienti qui dentro
def passo_valutazione(modello, loader, criterio, device):
    """Una epoca di valutazione. Stessa firma, nessun aggiornamento dei pesi."""
    modello.eval()
    perdita_tot, corretti, totale = 0.0, 0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logit = modello(X)
        perdita_tot += criterio(logit, y).item() * X.size(0)
        corretti += (logit.argmax(dim=1) == y).sum().item()
        totale += X.size(0)

    return perdita_tot / totale, corretti / totale
```

Due dettagli che pagano subito. La moltiplicazione `* X.size(0)` serve perché
la loss restituita da PyTorch è già una **media sul batch**, e la media delle
medie non è la media. Con i numeri: due vassoi, il primo con dieci esempi che
sbagliano in media di $1$, il secondo con due esempi che sbagliano in media di
$4$. La media vera sui dodici esempi è $(10 \cdot 1 + 2 \cdot 4)/12 = 1{,}5$;
la media delle due medie è $(1 + 4)/2 = 2{,}5$, cioè quasi il doppio, perché
conta i due esempi del secondo vassoio come se fossero dieci. Moltiplicare
ciascuna media per il numero di esempi del suo vassoio, sommare, e dividere
alla fine per il totale rimette le cose a posto. Non è un caso di scuola:
`drop_last=False` fa sì che l'ultimo batch sia quasi sempre più piccolo degli
altri. E `@torch.no_grad()` scritto sopra la funzione con la chiocciola (in
Python si chiama **decoratore**: una riga che avvolge la funzione e ne cambia
il comportamento senza toccarne il corpo) evita di dover ricordare il blocco
`with` a ogni chiamata: la funzione *è* una valutazione, non può essere altro.

## Il punto d'ingresso

`train.py` è l'unico file che si lancia, e l'unico che conosce i valori
concreti. Tutto ciò che potrebbe cambiare da un esperimento all'altro diventa
un argomento della riga di comando.

```{code-block} python
:class: pt-non-eseguibile

# train.py
import argparse
import torch
from torch import nn

import data_setup, engine, model_builder, utils

def main() -> None:
    p = argparse.ArgumentParser(description="Addestra un classificatore di immagini.")
    p.add_argument("--dati", type=str, required=True, help="cartella con train/ e test/")
    p.add_argument("--epoche", type=int, default=10)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--unita-nascoste", type=int, default=128)
    p.add_argument("--seme", type=int, default=42)
    p.add_argument("--uscita", type=str, default="modelli/modello.pt")
    args = p.parse_args()

    utils.fissa_seme(args.seme)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, test_loader, classi = data_setup.crea_dataloader(
        radice=args.dati, batch_size=args.batch)

    modello = model_builder.CNNSemplice(
        unita_nascoste=args.unita_nascoste, n_classi=len(classi)).to(device)

    criterio = nn.CrossEntropyLoss()
    ottimizzatore = torch.optim.Adam(modello.parameters(), lr=args.lr)

    for epoca in range(args.epoche):
        pt, at = engine.passo_addestramento(modello, train_loader, criterio,
                                            ottimizzatore, device)
        pv, av = engine.passo_valutazione(modello, test_loader, criterio, device)
        print(f"epoca {epoca+1:>2}/{args.epoche} | "
              f"train perdita {pt:.4f} acc {at:.3f} | "
              f"test perdita {pv:.4f} acc {av:.3f}")

    utils.salva_modello(modello, ottimizzatore, args.epoche, args.uscita,
                        classi=classi, argomenti=vars(args))

if __name__ == "__main__":      # eseguito solo se si lancia questo file
    main()
```

Da terminale, un esperimento diventa una riga:

```bash
python train.py --dati dati/ --epoche 20 --lr 0.001
python train.py --dati dati/ --epoche 20 --lr 0.0001 --unita-nascoste 256
```

`````{tab} Elementare
La riga `if __name__ == "__main__":` è la più misteriosa del blocco e ha una
spiegazione semplice: dice "esegui `main()` **solo** se qualcuno ha lanciato
questo file direttamente, non se un altro file mi ha importato". Senza,
importare `train.py` da un test o da un altro script farebbe partire un intero
addestramento come effetto collaterale. E c'è un motivo più concreto: su
Windows e su macOS il `DataLoader` con più *worker* riesegue il modulo in ogni
processo figlio, e senza quella riga ogni figlio farebbe ripartire
l'addestramento da capo; Python se ne accorge e blocca tutto con un errore.

L'ultima riga di `main()` salva, insieme ai pesi, anche i nomi delle classi,
gli argomenti usati e la memoria dell'ottimizzatore. È il gesto che distingue
un modello utile da un file misterioso: fra sei mesi quel `.pt` da solo non
dirà né che cosa predice, né come è stato ottenuto, né da dove ripartire.
`````

`````{tab} Superiore
`argparse` fa parte della libreria standard e per un progetto singolo basta.
Quando la configurazione cresce (decine di parametri, combinazioni annidate,
varianti per ambiente), si passa a un sistema di configurazione a file (`YAML`
più `dataclass`, oppure Hydra), che rende l'intera configurazione un artefatto
versionabile invece di una stringa nella cronologia della shell. È esattamente
la nozione di *tre artefatti da versionare* (codice, dati, configurazione)
discussa in [dal notebook alla
produzione](../MLOps/dal-notebook-alla-produzione.md).

Sul salvataggio conviene essere espliciti: un `state_dict` nudo non è
autosufficiente. Il minimo utile è un dizionario che contenga i pesi, lo stato
dell'ottimizzatore, l'epoca raggiunta, la mappatura classe → indice e la
configurazione:

```python
# utils.py
import pathlib, torch

def salva_modello(modello, ottimizzatore, epoca, percorso, classi, argomenti):
    """Un checkpoint completo: per *usare* il modello e per *riprendere* il lavoro."""
    percorso = pathlib.Path(percorso)
    percorso.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"pesi": modello.state_dict(),
                "ottimizzatore": ottimizzatore.state_dict(),
                "epoca": epoca,
                "classi": classi,
                "config": argomenti}, percorso)
```

Le due righe centrali sono quelle che di solito mancano, ed è la distinzione
già vista nella sezione [sul training loop](addestramento.md): senza lo stato
dell'ottimizzatore il file serve a **ripartire da capo**, non a **riprendere**.
Con Adam la differenza si misura: il primo passo dopo la ripresa vale
$\approx \eta$ pieno, contro i pochi centesimi di $\eta$ della traiettoria non
interrotta. Vale per tutto ciò che ha uno `state_dict` e che il ciclo tocca,
scheduler e `GradScaler` compresi.

Da PyTorch 2.6 `torch.load` usa `weights_only=True` come default, quindi un
file che contiene oggetti Python arbitrari va ricaricato con
`weights_only=False`, o, meglio, salvato con dentro solo tipi elementari, come
qui: `vars(args)` è un dizionario di stringhe e numeri, non un `Namespace`,
proprio per questo.
`````

## Riproducibilità: fissare il caso

Uno script che dà un risultato diverso a ogni esecuzione non è un esperimento.
Il minimo indispensabile sta in poche righe, che vanno chiamate **prima** di
creare modello e `DataLoader`.

```{figure} ../figures/salvare-ricaricare-confrontare-modelli.svg
:name: fig-serializzazione
:alt: "Ciclo di serializzazione: dal modello addestrato si salva lo state_dict su disco, insieme alla configurazione e al seme; per ricaricarlo si ricostruisce prima l'architettura e poi vi si caricano i pesi. Un ramo laterale mostra il confronto fra due checkpoint diversi sulla stessa metrica."
:width: 96%

Salvare i pesi non basta. Lo `state_dict` è solo un dizionario di numeri: per
rimetterlo al suo posto serve un'architettura identica, e quindi la
configurazione va salvata insieme.
```

L'asimmetria di {numref}`fig-serializzazione` è la fonte del più comune errore
di ricaricamento. Il salvataggio parte da un oggetto vivo e produce numeri; il
ricaricamento deve fare il contrario, e i numeri da soli non sanno dire in che
forma andavano. È per questo che configurazione e seme viaggiano nello stesso
checkpoint.

```python
# utils.py
import random
import numpy as np
import torch

def fissa_seme(seme: int = 42) -> None:
    random.seed(seme)             # librerie standard
    np.random.seed(seme)          # NumPy (usato dalle trasformazioni)
    torch.manual_seed(seme)       # PyTorch, CPU
    torch.cuda.manual_seed_all(seme)   # PyTorch, tutte le GPU
```

`````{tab} Elementare
Il caso, in un computer, non è mai davvero casuale: è una lunga sequenza
prestabilita di numeri che sembrano casuali, e il **seme** è il punto da cui
si comincia a leggerla. Stesso seme, stessa sequenza, stessi pesi iniziali,
stesso ordine di mescolamento dei dati, quindi stesso risultato.

Attenzione a che cosa significa e a che cosa non significa. Fissare il seme
serve a **confrontare**: se cambio il learning rate e il risultato migliora, con
il seme fisso so che il merito è del learning rate. Non serve a dire che il
modello è buono: un risultato ottenuto con un solo seme fortunato non è un
risultato. Per quello si ripete l'esperimento con tre o cinque semi diversi e si
riporta la media, e magari anche quanto ballano i valori.
`````

`````{tab} Superiore
Fissare i semi rende riproducibile la sequenza pseudocasuale, ma non basta a
garantire risultati bit-identici su GPU: molti kernel CUDA usano riduzioni
atomiche il cui ordine di somma varia tra esecuzioni, e in virgola mobile
l'addizione non è associativa. Il determinismo completo si chiede
esplicitamente, e si paga:

```python
torch.use_deterministic_algorithms(True)   # errore se un'op non ha versione deterministica
torch.backends.cudnn.benchmark = False     # niente autotuning degli algoritmi
# e, per cuBLAS, la variabile d'ambiente CUBLAS_WORKSPACE_CONFIG=:4096:8
```

`cudnn.benchmark = True` (il default consigliato per le prestazioni) prova più
algoritmi di convoluzione e sceglie il più veloce per quella forma di input: è
ottimo quando le forme sono costanti, controproducente quando cambiano di
continuo, e non deterministico in entrambi i casi. Anche i `DataLoader` con
più worker richiedono attenzione: si passa un `generator` con seme fisso e si
definisce `worker_init_fn` per fissare il seme di ciascun processo. In
pratica, nella ricerca si punta alla riproducibilità *statistica* (stessa
distribuzione di risultati su più semi) e si riserva il determinismo bit-a-bit
ai casi in cui serve davvero, come il debugging di una regressione.
`````

## Quando *non* modularizzare

Vale la pena dirlo, perché il consiglio opposto è più comune: si può
modularizzare troppo presto. Un'idea che non si sa ancora se funzioni non ha
bisogno di cinque file, di un parser degli argomenti e di una gerarchia di
classi; ha bisogno di essere provata in venti minuti. La divisione in moduli è
un investimento che si ripaga quando qualcosa si ripete, e non prima.

Il criterio pratico è quello delle **tre volte**: la prima volta si scrive nel
notebook; la seconda si copia e incolla, borbottando; la terza si estrae una
funzione. E il passaggio non è mai tutto-o-niente: si può tenere il notebook
come interfaccia di esplorazione e importarvi `engine.py`, ottenendo il meglio
delle due cose (grafici e assaggi nel notebook, logica stabile e testabile nei
file).

Cinque file, un comando, un seme fissato: è tutto quello che serve perché un
esperimento smetta di essere un ricordo.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Il notebook è una **cucina di prova**, lo script è la **ricetta scritta**. Il
  segnale che è ora di passare dall'uno all'altro è sempre lo stesso: "sto
  rilanciando la stessa cosa cambiando un numero".
- La divisione standard è in cinque file, uno per mestiere: i dati, il
  modello, il giro di addestramento, le funzioni di servizio, e il file che si
  lancia. Il terzo non sa nulla del problema, e per questo si riusa ovunque.
- Quando si sommano gli errori di più vassoi bisogna pesarli per quanti esempi
  contengono: la media delle medie non è la media.
- Tutto ciò che cambia da un esperimento all'altro si passa **da terminale**,
  non modificando il codice: così resta scritto nella cronologia.
- Nel file salvato vanno **i pesi, i nomi delle classi, la configurazione** e
  la memoria dell'ottimizzatore: senza, fra sei mesi quel file non dice né che
  cosa predice, né come è stato ottenuto, né da dove ripartire.
- Fissare il **seme** del caso serve a confrontare due esperimenti fra loro.
  Non serve a dire che il modello è buono: per quello si ripete con tre o
  cinque semi diversi.
- Non dividere in file troppo presto: la regola delle **tre volte**.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Il notebook è un **laboratorio** (stato implicito, ordine invisibile), lo
  script è un **prodotto** (un punto d'ingresso, tutto dichiarato). Il segnale
  del passaggio è: "sto rilanciando la stessa cosa cambiando un numero".
- La divisione standard è in cinque file: `data_setup`, `model_builder`,
  `engine`, `utils`, `train` (dove `engine` contiene il loop, indipendente dal
  problema).
- Nell'accumulo di metriche: `perdita.item() * X.size(0)`, perché la loss di
  PyTorch è già una media sul batch.
- `argparse` più `if __name__ == "__main__":` (quest'ultimo indispensabile
  anche per i worker del `DataLoader` su Windows e macOS).
- Si salvano **pesi, classi, configurazione e stato dell'ottimizzatore**
  insieme: uno `state_dict` nudo fra sei mesi non dice che cosa predice, e da
  solo non permette di riprendere.
- Fissare i semi serve a **confrontare** gli esperimenti; il determinismo
  bit-a-bit su GPU si chiede a parte e si paga in prestazioni.
- Non modularizzare troppo presto: la regola delle **tre volte**.
```
`````
