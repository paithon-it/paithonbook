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
**terminale**, cioè quella finestra in cui, invece di cliccare, si scrivono
comandi e il computer risponde. È già la soglia di quello che nel mestiere si
chiama «mandare un modello in produzione», cioè metterlo al lavoro sul serio
per qualcuno che non sia chi l'ha scritto: il
[capitolo sull'MLOps](../MLOps/dal-notebook-alla-produzione.md) riprende il
discorso da qui in poi.

## Il laboratorio e il prodotto

`````{tab} Elementare
Un notebook è una cucina di prova: assaggi, aggiungi, ributti, tieni tre
pentole sul fuoco. È lo strumento giusto per capire se un'idea funziona,
proprio perché non ti obbliga a essere ordinato. Quello che non resta è la
pentola. Nella pagina salvata ci sono i piatti riusciti e i numerini che dicono
in che ordine hai acceso i fornelli l'ultima volta; non che cosa bolliva
dentro, e nemmeno le prove che nel frattempo hai buttato via. Uno script è
invece la ricetta scritta: chiunque la legga ottiene lo stesso piatto, nello
stesso ordine, senza doverti chiedere niente.

Il segnale che è arrivato il momento di passare dall'uno all'altro è sempre lo
stesso: quando cominci a rilanciare la stessa cosa cambiando un numero.
Cinque prove con cinque learning rate diversi, fatte modificando a mano una
cella, sono cinque esecuzioni di cui domani non ricorderai la differenza. Le
stesse cinque prove lanciate da terminale con `--lr 0.01`, `--lr 0.001` e così
via restano scritte nella cronologia del terminale: sono un esperimento, non
un ricordo.

E una ricetta scritta si maneggia in modi che una cucina non permette. Metti
accanto quella di ieri e quella di oggi, e in una riga vedi che cosa è
cambiato, il sale da cinque grammi a otto; due fotografie della cucina a fine
serata non te lo direbbero mai, e il file del notebook, che si porta dentro
anche tutto quello che è uscito dal forno, somiglia più alle fotografie che
alla ricetta. La ricetta puoi consegnarla a qualcuno che la esegue ogni mattina
alle sei senza che tu sia lì. E puoi provare un passaggio solo, la salsa, senza
cucinare tutta la cena, perché quel passaggio sta scritto per conto suo.
`````

`````{tab} Superiore
La differenza sostanziale è tra **stato implicito** e **stato esplicito**. Nel
notebook lo stato vive nel kernel: il documento salvato registra per ogni cella
il numero della sua ultima esecuzione, non la storia delle esecuzioni né le
celle nel frattempo cancellate, quindi il codice non determina il risultato
(condizione che rende impossibile la riproducibilità). Uno script ha un unico
punto d'ingresso, un ordine totale delle istruzioni, e tutto ciò che varia
passa dagli argomenti della riga di comando: input, output e parametri sono
dichiarati.

Ne discendono tre proprietà che nello script vengono gratis: si mette sotto
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
la media delle due medie è $(1 + 4)/2 = 2{,}5$, cioè due terzi più alta del
vero, perché conta i due esempi del secondo vassoio come se fossero dieci. Moltiplicare
ciascuna media per il numero di esempi del suo vassoio, sommare, e dividere
alla fine per il totale rimette le cose a posto. Non è un caso di scuola: a
meno di chiedere il contrario, il `DataLoader` l'ultimo vassoio lo serve anche
se è mezzo vuoto, quindi c'è quasi sempre un batch più piccolo degli altri.

Il secondo dettaglio è la riga `@torch.no_grad()` scritta sopra la seconda
funzione. Quella chiocciola in Python si chiama **decoratore**: è una riga che
avvolge la funzione e ne cambia il comportamento senza toccarne il corpo. Qui
dice «tutto quello che succede qui dentro succede a registratore spento», ed
evita di dover ricordare il blocco `with` a ogni chiamata. La funzione *è* una
valutazione, e non può essere altro.

## Il punto d'ingresso

`train.py` è l'unico file che si lancia, e l'unico che conosce i valori
concreti. Tutto ciò che potrebbe cambiare da un esperimento all'altro diventa
un argomento della riga di comando.

Leggendolo si incontrano dei nomi che qui non sono scritti da nessuna parte,
`data_setup.crea_dataloader` e `model_builder.CNNSemplice`: sono le funzioni
che stanno negli altri due file, quelli che costruiscono i `DataLoader` e il
modello, e che non riportiamo perché il capitolo li ha già scritti pagina per
pagina. È esattamente il punto della divisione in file: `train.py` non ha
bisogno di sapere come sono fatti dentro, gli basta chiamarli per nome.

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

A trasportare i valori da fuori a dentro il programma è `argparse`, la parte di
Python che legge quello che si è scritto nel terminale dopo il nome del file e
lo consegna al codice sotto forma di numeri e di parole. Ogni `add_argument`
dichiara una manopola: come si chiama da fuori (`--lr`), di che tipo è il
valore, e quanto vale se nessuno la tocca. In quest'ultimo campo si incontra
`1e-3`, che è il modo in cui i programmi scrivono $0{,}001$: si legge «uno per
dieci alla meno tre», cioè uno diviso mille. Da terminale, un esperimento
diventa quindi una riga:

```bash
python train.py --dati dati/ --epoche 20 --lr 0.001
python train.py --dati dati/ --epoche 20 --lr 0.0001 --unita-nascoste 256
```

`````{tab} Elementare
La riga `if __name__ == "__main__":` è la più misteriosa del blocco e ha una
spiegazione semplice: dice «esegui `main()` soltanto se qualcuno ha lanciato
questo file direttamente, non se un altro file è venuto a prendersi qualcosa da
qui». Senza, aprire `train.py` da un altro programma per riusarne una funzione
farebbe partire un intero addestramento senza che nessuno l'abbia chiesto.

Ce n'è anche un motivo più concreto, che riguarda gli aiutanti del
`DataLoader`. Su Windows e su macOS ciascuno di loro è un programma nuovo, che
per sapere che cosa deve fare rilegge da capo il file da cui è nato: senza
quella riga, ognuno rileggendolo farebbe ripartire l'addestramento, e ogni
addestramento farebbe nascere altri aiutanti, all'infinito. Python se ne accorge
e blocca tutto con un errore.

Le manopole da terminale bastano finché sono una manciata. Quando diventano
quaranta, con degli incastri (se il modello è questo, quell'altra manopola non
vuol dire niente) e con valori che cambiano da un computer all'altro, la riga
da scrivere diventa lunga un metro. Allora si scrivono tutte su un foglio, un
file di configurazione, che si tiene insieme al codice. E il foglio ricorda
meglio. La cronologia del terminale è di quella macchina e di quell'utente, e
con l'uso si accorcia da sé, mentre il foglio resta lì e chiunque lo può
rileggere.

L'ultima riga di `main()` salva. E salva più dei soli pesi: insieme a quelli
finiscono nel file i nomi delle classi, tutte le manopole con cui è stato
lanciato l'esperimento, e la memoria dell'ottimizzatore vista nella sezione sul
[training loop](addestramento.md). È il gesto che distingue un modello utile da
un file misterioso: fra sei mesi quel `.pt`, da solo, non direbbe né che cosa
predice, né come è stato ottenuto, né da dove ripartire.

Nel file, però, ci vanno numeri e parole e nient'altro. Un biglietto scritto in
chiaro lo legge chiunque lo trovi. Un congegno, per dire quello che sa, deve
prima essere messo in funzione, e chi lo riceve deve fidarsi di chi gliel'ha
spedito. PyTorch, da qualche versione, quando apre un file di questi
accetta i numeri e le parole e si ferma davanti al resto, a meno che tu non gli
dichiari per iscritto che di quel file ti fidi. Le manopole quindi finiscono lì
dentro come un semplice elenco di nomi e valori.
`````

`````{tab} Superiore
`argparse` fa parte della libreria standard e per un progetto singolo basta.
Quando la configurazione cresce (decine di parametri, combinazioni annidate,
varianti per ambiente), si passa a un sistema di configurazione a file (`YAML`
più `dataclass`, oppure Hydra), che rende l'intera configurazione un artefatto
versionabile invece di una stringa nella cronologia della shell, da conservare
accanto a codice, dati e modello come in [dal notebook alla
produzione](../MLOps/dal-notebook-alla-produzione.md).

Da PyTorch 2.6 `torch.load` usa `weights_only=True` come default, quindi un
file che contiene oggetti Python arbitrari va ricaricato con
`weights_only=False`, o, meglio, salvato con dentro solo tipi elementari, come
qui: `vars(args)` è un dizionario di stringhe e numeri, non un `Namespace`,
proprio per questo.
`````

Ed ecco la funzione che salva, che è il posto in cui il capitolo mette in fila
tutto quello che ha detto sui checkpoint: i pesi, lo stato
dell'ottimizzatore, l'epoca raggiunta, i nomi delle classi e la configurazione.

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

Delle cinque voci che finiscono nel file, quelle che di solito mancano sono
`"ottimizzatore"` ed `"epoca"`, ed è la distinzione già vista nella sezione
[sul training loop](addestramento.md): senza lo stato dell'ottimizzatore il
file serve a **ripartire da capo**, non a **riprendere**.

```{figure} ../figures/salvare-ricaricare-confrontare-modelli.svg
:name: fig-serializzazione
:alt: "Ciclo di serializzazione: dal modello addestrato si salva lo state_dict su disco, insieme alla configurazione e al seme; per ricaricarlo si ricostruisce prima l'architettura e poi vi si caricano i pesi. Un ramo laterale mostra il confronto fra due checkpoint diversi sulla stessa metrica."
:width: 96%

Salvare i pesi non basta. Nel file finisce solo un elenco di numeri: per
rimetterli al loro posto serve un modello fatto esattamente come quello di
partenza, e quindi la configurazione va salvata insieme.
```

L'asimmetria disegnata in {numref}`fig-serializzazione` è la fonte del più
comune errore di ricaricamento. Il salvataggio parte da un modello vivo e
produce numeri, e sembra facile; il ricaricamento deve fare il contrario, e i
numeri da soli non sanno dire in che forma andavano rimessi. È per questo che
configurazione e seme viaggiano nello stesso file dei pesi.

## Riproducibilità: fissare il caso

Uno script che dà un risultato diverso a ogni esecuzione non è un esperimento.
Il minimo indispensabile sta in poche righe, ed è la funzione che `train.py`
chiama per prima, prima ancora di costruire il modello e i `DataLoader`: se il
caso lo si fissa dopo che i pesi sono già stati sorteggiati, non si è fissato
niente.

Prima però conviene sapere che cosa sia un seme, perché il codice qui sotto
senza quello è indecifrabile. Il caso, in un computer, non esiste: quello che
c'è è una lunghissima sequenza di numeri prestabilita, calcolata con una
formula, che *sembra* casuale. Il **seme** è il punto da cui si comincia a
leggerla. Stesso seme, stesso punto di partenza, stessa sequenza, e quindi
stessi pesi iniziali e stesso ordine di mescolamento dei dati: stesso
risultato, oggi e fra un anno.

Le righe sono quattro perché ogni libreria ha la sua sequenza, e vanno
avvisate tutte: quella di Python, quella di NumPy (che le trasformazioni delle
immagini usano), e quelle di PyTorch, una per il processore e una per le schede
grafiche.

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
Attenzione a che cosa significa e a che cosa non significa. Fissare il seme
serve a **confrontare**: se cambio il learning rate e il risultato migliora, con
il seme fisso so che il merito è del learning rate. Non serve a dire che il
modello è buono: un risultato ottenuto con un solo seme fortunato non è un
risultato. Per quello si ripete l'esperimento con tre o cinque semi diversi e si
riporta la media, e magari anche quanto ballano i valori.

C'è poi una cosa che il seme non compra: le ultime cifre. Una somma lunga,
fatta in ordini diversi, dà totali diversi, e non serve sbagliare niente,
basta che a ogni passaggio si arrotondi. Un conto in banca tiene i centesimi, e
matura quattro decimi di centesimo di interessi al giorno. Accreditati giorno
per giorno, spariscono ogni volta nell'arrotondamento, e dopo un anno il saldo è
quello di partenza; sommati prima fra loro fanno un euro e quarantasei, e il
saldo si muove. Stessi numeri, ordine diverso, totale diverso.

Una scheda grafica lavora proprio così, spezzando la somma fra migliaia di
calcoli che corrono nello stesso momento e consegnano appena hanno finito, e
chi consegna per primo non è sempre lo stesso. Le differenze sono nelle ultime
cifre, molto più piccole di quelle del conto in banca; ma le somme di un
addestramento sono milioni, e alla fine due esecuzioni dello stesso codice, con
lo stesso seme e sulla stessa macchina, non danno più lo stesso numero fino
all'ultima cifra.

Si può pretendere che le somme si facciano sempre nello stesso ordine, e si può
proibire alla macchina di provare ogni volta due modi di fare la stessa
moltiplicazione per tenersi il più veloce (quale dei due vinca dipende da com'è
messa la scheda quella sera, e provarli rende soltanto se poi lo stesso calcolo
si ripete mille volte identico). Allora i numeri tornano uguali fino
all'ultima cifra, e si paga. Di qualche operazione la versione ordinata non
esiste e il programma si ferma dicendolo; il resto va più piano. È un prezzo
che si accetta quando si dà la caccia a un errore e serve sapere che fra due
esecuzioni è cambiata soltanto la cosa che si è cambiata.

Un ultimo avviso agli aiutanti che preparano i vassoi. Ognuno ha bisogno di un
punto da cui leggere la sequenza, e a ciascuno va detto quale. Senza, possono
ritrovarsi tutti sulla stessa riga e servire vassoi con le stesse identiche
variazioni, oppure ripartire ogni sera da un punto diverso.
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

`cudnn.benchmark = True` (spento di default, si accende per le prestazioni)
prova più algoritmi di convoluzione e sceglie il più veloce per quella forma
di input: è ottimo quando le forme sono costanti, controproducente quando
cambiano di continuo, e non deterministico in entrambi i casi. Anche i
`DataLoader` con più worker richiedono attenzione: si passa un `generator` con
seme fisso e si definisce `worker_init_fn` per fissare il seme di ciascun
processo. In pratica, nella ricerca si punta alla riproducibilità *statistica*
(stessa distribuzione di risultati su più semi) e si riserva il determinismo
bit-a-bit ai casi in cui serve davvero, come il debugging di una regressione.
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
