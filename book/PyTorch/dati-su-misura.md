# Dati su misura: `Dataset`, `DataLoader` e trasformazioni

Nei manuali il dataset arriva sempre pronto: una riga di codice, MNIST si
scarica da solo, le immagini sono già quadrate, già etichettate, già divise in
addestramento e test. Nella vita reale il primo giorno di un progetto
assomiglia piuttosto a questo: una cartella con quattromila fotografie, i nomi
dei file scritti da tre persone diverse, due immagini corrotte, una classe con
dodici esempi e un'altra con duemila. Prima di poter scrivere un `nn.Module`
bisogna costruire il tubo che porta quei file dentro la rete, e in PyTorch
quel tubo si costruisce con due pezzi soltanto, sempre gli stessi.

Il capitolo li ha già incontrati di sfuggita nella sezione
[sull'addestramento](addestramento.md): `Dataset` sa consegnare l'esempio
numero $i$, `DataLoader` li impila in mini-batch. Qui li costruiamo noi, sui
nostri dati.

## La convenzione delle cartelle

Prima di scrivere codice, conviene sapere che per le immagini esiste una
convenzione che risparmia il lavoro: una cartella per classe, e il nome della
cartella è l'etichetta.

```text
dati/
├── addestramento/
│   ├── pizza/      img_001.jpg  img_002.jpg  ...
│   ├── bistecca/   img_331.jpg  ...
│   └── sushi/      img_780.jpg  ...
└── test/
    ├── pizza/      ...
    ├── bistecca/   ...
    └── sushi/      ...
```

Con questa disposizione, `torchvision` fa tutto da sé:

```python
from torchvision import datasets, transforms

preparazione = transforms.Compose([
    transforms.Resize((224, 224)),   # tutte le immagini della stessa misura
    transforms.ToTensor(),           # da immagine a tensore (canali, altezza,
                                     # larghezza) con i valori portati fra 0 e 1
])

dati_train = datasets.ImageFolder(root="dati/addestramento", transform=preparazione)
dati_test = datasets.ImageFolder(root="dati/test", transform=preparazione)

print(dati_train.classes)         # ['bistecca', 'pizza', 'sushi']  (ordine alfabetico)
print(dati_train.class_to_idx)    # {'bistecca': 0, 'pizza': 1, 'sushi': 2}
print(len(dati_train))            # quante immagini in tutto
immagine, etichetta = dati_train[0]
print(immagine.shape, etichetta)  # torch.Size([3, 224, 224]) 0
```

Un dettaglio che sembra burocratico e non lo è: l'associazione classe → numero
segue l'**ordine alfabetico** delle cartelle, non quello in cui le abbiamo in
testa. Quando poi si legge una predizione, `dati_train.classes[indice]` è
l'unico modo corretto di tradurla in una parola. Scrivere a mano una lista di
nomi in un altro ordine è un classico modo di ottenere un modello che sembra
sbagliare tutto mentre invece funziona benissimo.

## Scrivere un `Dataset` a mano

`ImageFolder` copre il caso fortunato. Appena i dati stanno in un CSV, in un
database, in file audio con le etichette in un foglio a parte (o appena
servono più informazioni della sola classe), si scrive la propria classe. È
meno lavoro di quanto sembri: il contratto è di **tre metodi**.

```{figure} ../figures/ereditarieta-polimorfismo.svg
:name: fig-ereditarieta-dataset
:alt: "Gerarchia di classi: in cima una classe base che definisce un metodo di validazione comune; sotto, tre sottoclassi che la ereditano e ridefiniscono ciascuna il proprio metodo di caricamento, uno per le immagini, uno per il CSV, uno per l'audio. Chi le usa chiama sempre gli stessi metodi, senza sapere quale sottoclasse ha davanti."
:width: 92%

Uno stampo di partenza, tre versioni specializzate. Chi consuma i dati non sa
da dove vengano: fa sempre le stesse tre domande, e ognuna delle tre versioni
risponde a modo suo, leggendo immagini, un foglio di calcolo o dei file audio.
```

La {numref}`fig-ereditarieta-dataset` mostra il meccanismo che permette al
`DataLoader` di funzionare con qualunque `Dataset` senza saperne nulla, ed è
l'ereditarietà incontrata nella sezione sui
[moduli](moduli.md), usata qui per un altro scopo. In cima c'è `Dataset`, lo
stampo di PyTorch, che non contiene quasi niente: dichiara soltanto quali tre
domande gli si possono fare. Sotto ci sono le classi che noi scriviamo, una per
tipo di dato, e ciascuna risponde a quelle tre domande a modo proprio. Il
guadagno è che il `DataLoader` non deve conoscerle: gli basta sapere che
qualunque cosa erediti da `Dataset` sa rispondere. Finché la vostra classe
rispetta il contratto dei tre metodi, per il resto di PyTorch è
indistinguibile da `ImageFolder`, anche se legge un tipo di dato che chi ha
scritto la libreria non aveva previsto.

```python
import pathlib
import torch
from torch.utils.data import Dataset
from PIL import Image

class DatasetImmagini(Dataset):
    """Legge le immagini da cartelle-classe, come ImageFolder, ma è nostro."""

    def __init__(self, radice: str, transform=None):
        self.percorsi = sorted(pathlib.Path(radice).glob("*/*.jpg"))
        self.classi = sorted({p.parent.name for p in self.percorsi})
        self.classe_a_indice = {c: i for i, c in enumerate(self.classi)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.percorsi)

    def __getitem__(self, indice: int):
        percorso = self.percorsi[indice]
        immagine = Image.open(percorso).convert("RGB")     # anche i PNG a 4 canali
        etichetta = self.classe_a_indice[percorso.parent.name]
        if self.transform is not None:
            immagine = self.transform(immagine)
        return immagine, etichetta
```

`````{tab} Elementare
Sono tre risposte a tre domande che il `DataLoader` continuerà a fare per
tutto l'addestramento. *Come ti prepari?* (`__init__`: qui si fa il lavoro
lento una volta sola, elencare i file, leggere il CSV degli indici.) *Quanti
esempi hai?* (`__len__`.) *Dammi il numero 137* (`__getitem__`: qui si fa il
lavoro veloce, ed è la parte che verrà eseguita milioni di volte).

La regola pratica sta tutta in questa divisione del lavoro: **in `__init__` le
cose pesanti, in `__getitem__` le cose leggere**. Se in `__init__` carichi in
memoria tutte le immagini, un dataset da 200 GB non parte nemmeno; se in
`__getitem__` riapri un file CSV di 300 MB per leggere una riga,
l'addestramento diventa lentissimo, e la GPU, che aspetta i dati, resterà
ferma a girarsi i pollici.
`````

`````{tab} Superiore
È il protocollo *map-style*: una mappa da indice a esempio, che consente
campionamento casuale e quindi `shuffle`. L'alternativa è `IterableDataset`
(`__iter__`), pensata per gli stream (file compressi letti in sequenza, code
di messaggi, dataset che non stanno su disco), dove il campionamento casuale
non è possibile e lo shuffling si approssima con un buffer.

Due cose vanno sapute su `__getitem__`. La prima: con `num_workers > 0` viene
eseguito nei **processi worker**, non in quello principale (col default,
`num_workers=0`, tutto resta nel processo principale). Ai worker viene passato
l'oggetto `Dataset` stesso, che sulle piattaforme ad avvio *spawn* deve quindi
essere serializzabile (`pickle`); e un handle già aperto in `__init__` e usato
nel `__getitem__` (un connettore a database, un file HDF5) è la causa classica
dei crash con `num_workers > 0`, qualunque sia il modo di avvio. Si apre
*pigramente*, al primo accesso, dentro il worker. La seconda: deve restituire
tensori (o tipi che il *collate* di default sa impilare); il default gestisce
tensori, numeri, stringhe, dizionari e tuple annidate, ma pretende che tutti
gli elementi del batch abbiano la **stessa forma**.
`````

## Le trasformazioni: preparare, e moltiplicare

Una `transform` è una funzione che riceve un esempio e ne restituisce una
versione modificata. Serve a due scopi diversi, che è bene non confondere:
**preparare** (portare tutto alla stessa misura, allo stesso intervallo di
valori) e **moltiplicare** (generare varianti plausibili per rendere il
modello più robusto; la *data augmentation*, trattata in profondità nel
[capitolo sulla visione](../VisioneArtificiale/data-augmentation.md)).

```python
from torchvision import transforms

# ADDESTRAMENTO: prepara e moltiplica
train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),                    # ritaglio casuale
    transforms.RandomHorizontalFlip(p=0.5),        # specchiatura casuale
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # statistiche di ImageNet
                         std=[0.229, 0.224, 0.225]),
])

# VALUTAZIONE: solo prepara. Nessuna casualità.
test_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),                    # ritaglio deterministico
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

`````{tab} Elementare
La differenza tra le due liste è la regola d'oro di tutta la faccenda:
**si moltiplica solo in addestramento**. Ruotare, specchiare, schiarire le
foto di allenamento serve a insegnare al modello che un gatto capovolto è
ancora un gatto. Farlo durante l'esame significherebbe invece dare a ogni
studente domande diverse e a caso: il voto non sarebbe più confrontabile, né
con quello di ieri né con quello di un altro modello.

E la `Normalize`? Sposta i numeri in modo che abbiano media attorno a zero:
alle reti riesce molto più facile imparare quando i numeri in ingresso sono
piccoli e centrati, per lo stesso motivo per cui è più facile trovare la
strada in una città con l'origine al centro anziché in un angolo lontano. I sei
numeri della riga (tre medie e tre deviazioni standard, una per colore) sono le
statistiche di **ImageNet**, la grande raccolta pubblica di fotografie
etichettate su cui, dal 2012 in poi, si è misurata la visione artificiale e su
cui è addestrata la maggior parte dei modelli che si scaricano già pronti. Si
usano quelli quando si parte da uno di quei modelli, perché il modello se li
aspetta: sono i numeri con cui gli sono state date le immagini quando ha
imparato.
`````

`````{tab} Superiore
`ToTensor()` converte una `PIL.Image` in un tensore `float32` con layout
$(C, H, W)$ e valori riscalati in $[0,1]$; `Normalize`, date media $\mu$ e
deviazione standard $\sigma$, applica $x' = (x - \mu)/\sigma$ canale per
canale. L'ordine conta: la
normalizzazione lavora su tensori, quindi va **dopo** `ToTensor()`, mentre le
trasformazioni geometriche e fotometriche lavorano tradizionalmente su PIL e
vanno prima.

Le statistiche giuste sono quelle del dataset su cui il modello è stato
addestrato: se si fa transfer learning da pesi ImageNet si usano quelle di
ImageNet, e la scorciatoia più sicura è chiederle direttamente ai pesi;
`torchvision.models.EfficientNet_B0_Weights.DEFAULT.transforms()` restituisce
la pipeline esatta con cui quei pesi sono stati prodotti. Da `torchvision`
0.15 esiste `torchvision.transforms.v2`, che accetta anche box, maschere e
video insieme all'immagine (necessario per detection e segmentazione, dove la
trasformazione geometrica va applicata *coerentemente* a immagine ed
etichetta) ed è più veloce sui batch; l'API è retrocompatibile.
`````

## Il `DataLoader` sul serio

Con un `Dataset` in mano, il `DataLoader` aggiunge il resto: batch,
mescolamento, parallelismo.

```python
import os
from torch.utils.data import DataLoader

train_loader = DataLoader(
    dati_train,
    batch_size=32,
    shuffle=True,             # rimescola a ogni epoca: solo in addestramento
    num_workers=os.cpu_count(),   # processi che preparano i batch in parallelo
    pin_memory=True,          # memoria "bloccata": trasferimento più rapido alla GPU
    drop_last=True,           # scarta l'ultimo batch se incompleto
    persistent_workers=True,  # non li ricrea a ogni epoca
)

test_loader = DataLoader(dati_test, batch_size=64, shuffle=False,
                         num_workers=os.cpu_count(), pin_memory=True)
```

Vale la pena capire ognuno di questi argomenti, perché sono la differenza tra
un addestramento che dura un'ora e uno che ne dura sei.

`````{tab} Elementare
`num_workers` è il numero di aiutanti che preparano i vassoi mentre la cucina
cucina. Con zero aiutanti, il processo principale alterna: prepara un vassoio,
lo dà alla GPU, aspetta, prepara il prossimo, e la GPU, che è la parte cara
della macchina, resta ferma metà del tempo. Con quattro o otto aiutanti i
vassoi successivi sono già pronti quando servono.

`pin_memory` è il piano d'appoggio accanto al passavivande: i dati vengono
messi in una zona di memoria da cui la GPU può prenderli senza passaggi
intermedi. `drop_last` butta via l'ultimo vassoio se è mezzo vuoto, con
duemila esempi e batch da 32, l'ultimo ne ha 16, e nei modelli con la batch
normalization un batch anomalo può dare statistiche strane.
`persistent_workers` dice di non licenziare gli aiutanti alla fine di ogni
giro per riassumerli subito dopo: se prepararsi costa loro qualche secondo,
quei secondi si pagano una volta invece che a ogni epoca. E `shuffle=True`
mescola il mazzo prima di ogni giro, così la rete non impara l'ordine.

Un avvertimento: `num_workers` alto **non è sempre meglio**. Ogni aiutante è
un processo vero, con la sua memoria; oltre il numero di core della macchina
si litiga soltanto. Il modo di scegliere è misurare, non indovinare.
`````

`````{tab} Superiore
`num_workers=k` avvia $k$ processi (non thread: il GIL, spiegato nel capitolo
su Python, serializzerebbe proprio il codice Python puro del *preprocessing*,
che è il lavoro da parallelizzare qui) che eseguono
`__getitem__` e il *collate* in parallelo, riempiendo una coda da cui il
processo principale preleva. `prefetch_factor` (default 2) regola quanti batch
ogni worker tiene pronti in anticipo: la memoria occupata cresce come
$k \times \text{prefetch\_factor} \times \text{dimensione batch}$, e su
macchine con poca RAM è la prima causa di *out of memory* che non riguarda la
GPU. `persistent_workers=True` evita il costo di riavviarli a ogni epoca:
significativo quando `__init__` è pesante.

`pin_memory=True` alloca i batch in memoria *page-locked*, che consente il
trasferimento DMA asincrono verso la GPU; combinato con
`tensore.to(device, non_blocking=True)` permette di sovrapporre copia e
calcolo. Su Windows e macOS, dove i worker nascono per *spawn* e non per
*fork*, il codice che li avvia deve stare sotto
`if __name__ == "__main__":`, altrimenti si ottiene una ricorsione di
processi.

Infine `shuffle=True` e l'argomento `sampler` sono **mutuamente esclusivi**:
`shuffle` è di fatto una scorciatoia per `RandomSampler`. Chi passa un sampler
personalizzato deve togliere `shuffle`.
`````

## Quando gli esempi non hanno la stessa forma: `collate_fn`

Il meccanismo che impila gli esempi in un batch pretende che abbiano tutti la
stessa forma. Con le immagini ridimensionate è vero per costruzione; con il
testo, l'audio o le serie temporali non lo è quasi mai: una frase è lunga
sette parole, la successiva quarantatré. La soluzione è sostituire quel
meccanismo con il proprio.

```python
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class DatasetSequenze(Dataset):
    """Frasi gia' tradotte in numeri, di lunghezza diversa fra loro."""

    def __init__(self, n=100):
        lunghezze = torch.randint(5, 40, (n,))
        self.esempi = [(torch.randint(1, 50, (int(l),)), int(l) % 2)
                       for l in lunghezze]

    def __len__(self):
        return len(self.esempi)

    def __getitem__(self, indice):
        return self.esempi[indice]

def raggruppa(batch):
    """Riceve una lista di (sequenza, etichetta); restituisce un batch imbottito."""
    sequenze, etichette = zip(*batch)
    lunghezze = torch.tensor([len(s) for s in sequenze])          # (B,)
    imbottite = pad_sequence(sequenze, batch_first=True,          # (B, L_max)
                             padding_value=0)
    return imbottite, lunghezze, torch.tensor(etichette)

dati = DatasetSequenze()
loader = DataLoader(dati, batch_size=32, shuffle=True, collate_fn=raggruppa)

imbottite, lunghezze, etichette = next(iter(loader))
print(imbottite.shape, lunghezze.shape, etichette.shape)
# le lunghezze vere sono tutte diverse, la larghezza del batch e' la massima:
print(lunghezze[:8].tolist(), "-> larghezza", imbottite.shape[1])
```

Le **lunghezze** vanno restituite insieme ai dati e non sono un dettaglio.
Dopo l'imbottitura tutte le frasi del vassoio hanno la stessa larghezza, e gli
zeri aggiunti in coda sono indistinguibili da parole vere: senza sapere dove
finisce la frase, il modello imparerebbe che lo zero è una parola come le
altre, e passerebbe metà del suo tempo a studiare l'imbottitura. Le lunghezze
sono l'informazione che permette di dire «da qui in poi non guardare». Nei
capitoli sul [natural language
processing](../NaturalLanguageProcessing/modelli-sequenza.md) e sui
[Transformer](../Transformers/architettura.md) quella riga di "fin qui sì, da
qui no" prenderà un nome (si chiama *maschera*) e diventerà un ingrediente
dell'architettura; per ora basta averla in mano.

## Classi sbilanciate: pescare con criterio

Se una classe ha duemila esempi e un'altra dodici, il mescolamento uniforme
mostrerà la classe rara una volta ogni tanto e il modello imparerà, molto
razionalmente, a non nominarla mai. Un rimedio è cambiare il modo di pescare.

```python
from torch.utils.data import WeightedRandomSampler

# Le etichette si leggono dall'indice, senza aprire una sola immagine:
# ImageFolder le tiene in .targets. Iterare il dataset le otterrebbe
# ugualmente, ma caricando tutti i file da disco, inutilmente.
etichette = torch.tensor(dati_train.targets)               # (N,)
conteggi = torch.bincount(etichette)                       # esempi per classe
peso_per_classe = 1.0 / conteggi.float()                   # la classe rara pesa di più
# indicizzare con un elenco: per ogni etichetta va a prendere il peso della sua
# classe, quindi da 3 pesi (uno per classe) si ottengono N pesi (uno per esempio)
pesi = peso_per_classe[etichette]                          # un peso per esempio

campionatore = WeightedRandomSampler(weights=pesi,
                                     num_samples=len(pesi),
                                     replacement=True)

# Attenzione: con un sampler NON si passa shuffle.
loader = DataLoader(dati_train, batch_size=32, sampler=campionatore)
```

Il campionamento pesato è una delle tre leve possibili: le altre sono pesare
la *loss* (`weight` in `CrossEntropyLoss`) e generare esempi sintetici della
classe rara. Il capitolo sul machine learning discute quando conviene
ciascuna, e soprattutto perché in questi casi l'accuratezza smette di essere
una metrica onesta: si guardano [precisione, richiamo e
F1](../MachineLearning/metriche.md).

## Dividere i dati senza barare

Il gesto più innocuo del progetto (dividere in addestramento e test) è anche
quello dove si commettono i danni più difficili da scoprire.

```python
import torch
from torch.utils.data import random_split

n_val = int(0.1 * len(dati_train))
n_train = len(dati_train) - n_val
generatore = torch.Generator().manual_seed(42)    # divisione riproducibile
sotto_train, sotto_val = random_split(dati_train, [n_train, n_val],
                                      generator=generatore)
```

`````{tab} Elementare
La divisione a caso funziona solo se gli esempi sono davvero indipendenti. Non
lo sono, per esempio, se il dataset contiene **dieci fotografie dello stesso
paziente**, o dieci fotogrammi consecutivi dello stesso video: dividendo a
caso, alcune finiscono nell'addestramento e altre nel test, il modello
riconosce il paziente invece della malattia, e il voto d'esame risulta
splendido (fino al giorno in cui arriva un paziente nuovo).

La regola è: **si divide per gruppo, non per esempio**. Tutti i dati di un
paziente stanno o di qua o di là. E se i dati hanno una data, si divide per
data: si addestra sul passato e si valuta sul futuro, perché è così che
funzionerà davvero.
`````

`````{tab} Superiore
È la *data leakage* da correlazione di gruppo: la divisione casuale assume
esempi i.i.d., ipotesi violata da qualunque struttura gerarchica (paziente,
sessione, utente, documento). La contromisura è una divisione per gruppi:
l'equivalente PyTorch di `GroupShuffleSplit` di scikit-learn si scrive
raccogliendo gli indici per gruppo e passandoli a `torch.utils.data.Subset`.
Per dati temporali vale l'analogo temporale (*forward chaining*), trattato in
[serie temporali](../SerieTemporali/validazione-e-feature.md).

Un secondo tranello, più sottile: le statistiche di normalizzazione e ogni
altro parametro di preprocessing vanno calcolati **solo sul training set** e
poi applicati agli altri. Calcolare media e deviazione standard su tutto il
dataset prima di dividere lascia filtrare informazione dal test: un errore che
gonfia i risultati di poco, ma abbastanza da falsare un confronto.
`````

## Il collo di bottiglia è quasi sempre il disco

Un'ultima cosa, la meno intuitiva, ed è forse la più utile della sezione.
Quando un addestramento è lento, l'istinto dice che la colpa è del modello.
Nella maggior parte dei progetti che non riguardano i modelli giganti, la colpa
è invece del **caricamento dei dati**: la cucina finisce il vassoio e resta
ferma ad aspettare il successivo. Vale la pena tenerlo a mente, perché è la
diagnosi che quasi nessuno prova per prima e quasi sempre è quella giusta.

Come ci si accorge: si guarda quanto è occupata la scheda grafica mentre
l'addestramento gira. Se sta al cento per cento in modo stabile, il collo di
bottiglia è il calcolo; se invece salta dal cento a zero e ritorno, la scheda
sta aspettando i dati, e ogni ottimizzazione del modello sarà tempo perso.

`````{tab} Elementare
I rimedi, dal più efficace al meno:

**Più aiutanti.** Alzare `num_workers`: se il problema è che nessuno prepara i
vassoi mentre la cucina cucina, è la prima cosa da provare.

**Ritagliare le foto una volta sola.** Se ogni epoca ridimensiona quattromila
fotografie da dodici megapixel a 224 pixel per lato, quel lavoro lo si sta
rifacendo identico decine di volte. Farlo una volta e salvare le immagini già
piccole su disco è un pomeriggio che si ripaga in un'ora.

**Meno file, più grandi.** Questo è il rimedio che stupisce, perché la ragione
non è quella che si immagina: il costo grosso non è *leggere* le foto, è
**aprirle**. Aprire un file è come chiedere al bibliotecario di andare a
prendere un volume: il tempo lo fa il tragitto, non la lettura, e per un
milione di volumi si fa un milione di tragitti. Impacchettare le immagini in
pochi archivi grandi, letti di seguito, è chiedere al bibliotecario uno
scaffale intero in una volta. La differenza è enorme, e diventa drammatica
quando i file non stanno sul computer ma su un disco raggiunto attraverso la
rete.

**Spostare le trasformazioni pesanti sulla scheda grafica**, che le fa più in
fretta della CPU.
`````

`````{tab} Superiore
La diagnosi si fa con `nvidia-smi` a occhio o, meglio, con il profiler
`torch.profiler`, che separa il tempo speso in `DataLoader` da quello speso nei
kernel.

I rimedi, in ordine di efficacia: alzare `num_workers`; ridimensionare le
immagini **una volta** su disco invece che a ogni epoca; usare formati che si
leggono in blocco (`.npy`, WebDataset, LMDB) invece di milioni di piccoli file;
spostare le trasformazioni pesanti sulla GPU (`torchvision.transforms.v2`
lavora su batch di tensori, quindi anche su device).

Sul terzo vale la pena essere precisi. Su una collezione grande il costo
dominante non è decodificare i file, è **aprirli**: ogni `open()` è una
chiamata di sistema e un accesso ai metadati del filesystem, e un milione di
file piccoli produce un milione di accessi minuscoli e sparsi, che è lo schema
peggiore per qualunque disco e disastroso su uno storage di rete, dove ogni
accesso paga anche la latenza. Impacchettarli in pochi archivi letti in
sequenza sposta il lavoro dove l'hardware è veloce.
`````

Il capitolo sulle [prestazioni](prestazioni.md) riprende il discorso dal lato
del calcolo.

E c'è un secondo motivo per fare quel passaggio, che si paga una volta e serve
per sempre: mentre si scorre il dataset per impacchettarlo, si calcolano
**media e deviazione standard per canale**, i due vettori di tre numeri che
servono a `Normalize`. Sono statistiche del *training set* e vanno calcolate
solo su quello (calcolarle su tutto è la stessa forma di *leakage* dello scaler
tarato prima dello split, vista nel capitolo sul Machine Learning). Con un
modello pre-addestrato si usano invece quelle del dataset originale, ed è il
motivo per cui `EfficientNet_B0_Weights.DEFAULT.transforms()` porta con sé i
numeri di ImageNet: le feature apprese si aspettano ingressi centrati come lo
erano allora.

Il tubo che porta i file dentro la rete è fatto: da qui in avanti si può
tornare a occuparsi del modello, sapendo che i dati arrivano.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Un `Dataset` risponde a **tre domande**: come ti prepari (una volta sola, e
  lì si mette il lavoro lento), quanti esempi hai, dammi il numero 137 (ed è
  la risposta che verrà chiesta milioni di volte, quindi dev'essere veloce).
- Se le foto stanno in una cartella per classe, `ImageFolder` fa tutto da sé.
  I nomi delle classi li assegna in **ordine alfabetico**: vanno riletti da
  lui, mai riscritti a mano in un altro ordine.
- Le trasformazioni servono a due cose: **preparare** (stessa misura, stessa
  scala di numeri) e **moltiplicare** (girare, specchiare, schiarire). Si
  moltiplica solo in addestramento, mai durante l'esame.
- Il `DataLoader` ha una manopola che conta più delle altre, il numero di
  **aiutanti** che preparano i vassoi in parallelo; e una regola: o si mescola
  a caso, o si passa un modo di pescare proprio, non tutti e due.
- Se gli esempi hanno lunghezze diverse (frasi, suoni) si allungano tutti alla
  stessa misura con degli zeri, e si restituiscono anche le **lunghezze vere**,
  altrimenti il modello studia l'imbottitura.
- Si divide **per gruppo** (tutte le foto dello stesso paziente di qua o di
  là), o per data, mai a caso su esempi che si assomigliano: è il modo più
  comune di darsi un bel voto senza meritarlo.
- Se l'addestramento è lento, **sospetta i dati prima del modello**.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Un `Dataset` è un contratto di **tre metodi**: `__init__` (lavoro pesante,
  una volta), `__len__`, `__getitem__` (lavoro leggero, milioni di volte).
- `ImageFolder` copre il caso "una cartella per classe"; l'indice delle classi
  segue l'**ordine alfabetico**, e va riletto da `.classes`, mai riscritto a
  mano.
- Le trasformazioni **preparano** (resize, `ToTensor`, `Normalize`) e
  **moltiplicano** (augmentation): moltiplicare solo in addestramento, mai in
  valutazione.
- Nel `DataLoader` contano `num_workers`, `pin_memory`, `drop_last`,
  `persistent_workers`; `shuffle` e `sampler` si escludono a vicenda.
- Con esempi di lunghezza diversa serve un **`collate_fn`** che imbottisce e
  restituisce le lunghezze vere.
- Si divide **per gruppo** (paziente, video, utente) o per data, mai a caso su
  esempi correlati: è la forma più comune di *data leakage*.
- Se l'addestramento è lento, sospetta il **caricamento dei dati** prima del
  modello.
```
`````
