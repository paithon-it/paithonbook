# Dati su misura: `Dataset`, `DataLoader` e trasformazioni

Nei manuali il dataset arriva sempre pronto: una riga di codice e MNIST, la
raccolta di cifre scritte a mano su cui abbiamo addestrato il primo modello, si
scarica da solo, con le immagini già quadrate, già etichettate, già divise in
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
segue l’**ordine alfabetico** delle cartelle, non quello in cui le abbiamo in
testa. Quando poi si legge una predizione, `dati_train.classes[indice]` è
l'unico modo corretto di tradurla in una parola. Scrivere a mano una lista di
nomi in un altro ordine è un classico modo di ottenere un modello che sembra
sbagliare tutto mentre invece funziona benissimo.

## Scrivere un `Dataset` a mano

`ImageFolder` copre il caso fortunato. Appena i dati stanno in un CSV, in un
database, in file audio con le etichette in un foglio a parte (o appena
servono più informazioni della sola classe), si scrive la propria classe. È
meno lavoro di quanto sembri: si scrivono **tre metodi**, e due soli di quelli
sono il contratto vero, cioè le domande che PyTorch verrà davvero a farci; il
terzo è il costruttore, che serve a noi per prepararci.

```{figure} ../figures/ereditarieta-polimorfismo.svg
:name: fig-ereditarieta-dataset
:alt: "Gerarchia di classi: in cima Dataset, con i due metodi __len__ e __getitem__; sotto, tre sottoclassi che li scrivono ciascuna a modo proprio, una che apre file .jpg, una che legge una riga di CSV, una che apre file .wav. In basso il DataLoader, collegato a tutte e tre, che chiede sempre le stesse due cose senza sapere quale delle tre ha davanti."
:width: 92%

Uno stampo di partenza, tre versioni specializzate. Chi consuma i dati non sa
da dove vengano: chiede sempre le stesse due cose, e ognuna delle tre versioni
risponde a modo suo, leggendo immagini, un foglio di calcolo o dei file audio.
```

La {numref}`fig-ereditarieta-dataset` mostra il meccanismo che permette al
`DataLoader` di funzionare con qualunque `Dataset` senza saperne nulla, ed è
l'ereditarietà incontrata nella sezione sui [moduli](moduli.md), usata qui per
un altro scopo. In cima c'è `Dataset`, lo stampo di PyTorch, che non contiene
quasi niente: dice soltanto quali due domande gli si possono fare. Sotto ci
sono le classi che scriviamo noi, una per tipo di dato, e ciascuna risponde a
quelle due domande a modo proprio. Il guadagno è che il `DataLoader` non deve conoscerle:
gli basta sapere che qualunque cosa erediti da `Dataset` sa rispondere. Finché
la nostra classe rispetta il contratto, per il resto di PyTorch è
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
Sono tre metodi, ma le domande sono due, ed è la distinzione che la
{numref}`fig-ereditarieta-dataset` disegna. Le due domande che il `DataLoader`
farà per tutto l'addestramento sono *quanti esempi hai?* (`__len__`) e *dammi
il numero 137* (`__getitem__`: qui si fa il lavoro veloce, ed è la parte che
verrà eseguita milioni di volte). Il terzo metodo, `__init__`, nessuno ce lo
chiede: è la nostra preparazione, quella che avviene una volta sola prima di
cominciare, dove si elencano i file o si legge il foglio con le etichette.

La seconda domanda chiede un numero preciso, e questo apre una possibilità. Un
magazzino con gli scaffali numerati consegna il 137 senza toccare i
centotrentasei che vengono prima, così i pezzi si possono chiedere nell'ordine
che si vuole, per esempio in un ordine sorteggiato daccapo a ogni giro: è così
che i dati vengono mescolati.

Certi dati però non stanno su uno scaffale, arrivano come un nastro che
scorre, e da un nastro si prende quello che passa: chiedere il 137 non
significa niente, perché per arrivarci bisogna aver lasciato passare tutti
quelli davanti. Chi lavora così mescola come può, tenendo da parte un cesto di
qualche centinaio di pezzi e pescando lì dentro, e l'ordine si rompe almeno
dentro il cesto.

La regola pratica sta tutta in questa divisione del lavoro: **in `__init__` le
cose pesanti, in `__getitem__` le cose leggere**. Se in `__init__` carichi in
memoria tutte le immagini, un dataset da 200 GB non parte nemmeno; se in
`__getitem__` riapri un file CSV di 300 MB per leggere una riga,
l'addestramento diventa lentissimo, e la GPU, che aspetta i dati, resterà
ferma a girarsi i pollici.

Una cosa in `__init__` non ci va comunque: un archivio già aperto, o un
collegamento a una banca dati già stabilito. La preparazione la fa una persona
sola, e quando le richieste vengono poi smistate a più aiutanti, ognuno si
ritrova in mano la copia di una chiave che apparteneva a un altro: la porta
non si apre, e il lavoro muore con un errore che sembra venire da tutt'altra
parte. Il collegamento si stabilisce alla prima richiesta, e lo stabilisce chi
quella richiesta la sta servendo.
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
All'esame le domande sono uguali per tutti. Specchiare, ritagliare e schiarire
le fotografie serve mentre si studia, e insegna al modello che un gatto
capovolto è ancora un gatto. Farlo durante la prova vorrebbe dire sorteggiare
domande diverse per ogni studente, e un voto così non si confronta con niente,
né con quello di ieri né con quello di un altro.

Un professore che racconta com'è andata la verifica non elenca ventidue voti,
dice «due sopra la media» e «uno sotto». I numeri diventano piccoli, e la
differenza fra un compagno e l'altro si vede a colpo d'occhio invece di restare
nascosta dentro cifre che si somigliano tutte. `Normalize` fa lo stesso ai
colori, e le reti da numeri raccolti attorno allo zero imparano meglio.

La media da sola non basta, e si vede con due materie. In italiano i voti
stanno quasi tutti fra 5 e 7, in matematica vanno dal 2 al 10, e un 8 nella
prima non è la stessa impresa di un 8 nella seconda. Allora lo scarto dalla
media si divide per quanto quei voti si sparpagliano di solito, e lo
sparpagliamento si chiama **deviazione standard**. Con media 6 e sparpagliamento
1 quell'8 diventa 2, con media 6 e sparpagliamento 4 diventa 0,5.

Le materie di `Normalize` sono i colori. Per il rosso, il verde e il blu tiene
una media e uno sparpagliamento a testa, ed ecco perché i numeri della riga sono
sei, due per colore. Al rosso più acceso, che dopo la conversione vale 1, toglie
0,485 e divide per 0,229: viene circa 2,2. Al nero, che vale 0, viene circa 2,1
sotto zero. Il rosso prima andava da 0 a 1, adesso si distende da 2,1 sotto zero
a 2,2 sopra.

La media di classe si fa sui voti, non sui compiti. Finché sono fogli si possono
ricopiare, accorciare, riscrivere, ma nessuno ne fa la media. Ruotare,
ritagliare e schiarire sono cose che si fanno a una fotografia; togliere una
media e dividere per un numero sono cose che si fanno a dei numeri. Prima la
foto, poi la conversione in numeri, e solo dopo la sottrazione, perché le ultime
due al contrario non funzionano proprio.

I sei numeri vengono da **ImageNet**, la grande raccolta pubblica di fotografie
etichettate su cui, dal 2012 in poi, si è misurata la visione artificiale.
Perché la scala di ImageNet su delle foto di pizza? Perché quasi nessuno parte
da zero: si prende un modello che ha già studiato là, e lui quella scala se
l'aspetta, come uno studente che ha imparato in decimi e si trova davanti i
giudizi a lettere. Chi parte davvero da zero i sei numeri se li calcola sulle
proprie foto, la prima volta che le scorre tutte.
`````

`````{tab} Superiore
`ToTensor()` converte una `PIL.Image` in un tensore `float32` con layout
$(C, H, W)$ e valori riscalati in $[0,1]$; `Normalize`, date media $\mu$ e
deviazione standard $\sigma$, applica $x' = (x - \mu)/\sigma$ canale per
canale. L'ordine conta: la
normalizzazione lavora su tensori, quindi va dopo `ToTensor()`, mentre le
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

Conviene capire ognuno di questi argomenti, perché sono la differenza tra un
addestramento che dura un'ora e uno che ne dura sei.

`````{tab} Elementare
`num_workers` è il numero di aiutanti che preparano i vassoi mentre la cucina
cucina. Con zero aiutanti, il processo principale alterna: prepara un vassoio,
lo dà alla GPU, aspetta, prepara il prossimo, e la GPU, che è la parte cara
della macchina, resta ferma metà del tempo. Con quattro o otto aiutanti i
vassoi successivi sono già pronti quando servono.

Quanti ne tengono pronti ciascuno? Due, se non si dice altro. Con otto
aiutanti sono sedici vassoi apparecchiati in giro per la cucina, più quello in
uso: se i vassoi sono grandi, il piano di lavoro si riempie prima che la
cucina abbia fame, e la macchina resta senza memoria per una ragione che con
la scheda grafica non c'entra niente.

`pin_memory` è il piano d'appoggio accanto al passavivande: i dati vengono
messi in una zona di memoria da cui la GPU può prenderli senza passaggi
intermedi, e mentre li sta prendendo la cucina può già lavorare al resto.

`drop_last` butta via l'ultimo vassoio se è mezzo vuoto: con duemila esempi e
vassoi da 32, l'ultimo ne ha 16, e ci sono tipi di strato che dai numeri del
vassoio ricavano delle statistiche, e su un vassoio grande la metà quelle
statistiche vengono storte. Il capitolo sul deep learning dirà quali.

`persistent_workers` dice di non licenziare gli aiutanti alla fine di ogni
giro per riassumerli subito dopo: se prepararsi costa loro qualche secondo,
quei secondi si pagano una volta invece che a ogni epoca. E `shuffle=True`
mescola il mazzo prima di ogni giro, così la rete non impara l'ordine. Chi
preferisce pescare a modo suo (dando più probabilità agli esempi rari, per
esempio) passa il proprio modo di pescare e toglie `shuffle`: o l'uno o
l'altro, e chiedendoli tutti e due si ottiene un errore subito.

Un avvertimento sul numero: alzare `num_workers` non paga oltre un certo
punto. Ogni aiutante è un processo vero, con la sua memoria; oltre il numero
di core della macchina si litiga soltanto. Il modo di scegliere è misurare,
non indovinare.

E un tranello che colpisce su Windows e macOS, dove ogni aiutante che si
presenta al lavoro rilegge da capo il foglio delle istruzioni. Se sul foglio,
in mezzo alle altre righe, c'è scritto «assumi otto aiutanti», ognuno
proverebbe ad assumerne altri otto, e la catena non finirebbe più: Python se
ne accorge sulla porta e ferma tutto con un errore. Il rimedio è mettere le
righe che avviano il lavoro sotto
`if __name__ == "__main__":`, che è il modo di dire «questo pezzo lo esegue
soltanto chi ha lanciato il programma, non chi arriva dopo».
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
`if __name__ == "__main__":`: senza, ogni worker rilegge il modulo e prova a
far ripartire il programma, e Python lo ferma sul nascere con un
`RuntimeError`.

Infine `shuffle=True` e l'argomento `sampler` sono **mutuamente esclusivi**:
`shuffle` è di fatto una scorciatoia per `RandomSampler`. Chi passa un sampler
personalizzato deve togliere `shuffle`.
`````

## Quando gli esempi non hanno la stessa forma: `collate_fn`

Il pezzo che impila gli esempi in un batch (si chiama *collate*, che in inglese
vuol dire proprio «mettere in ordine dei fogli sciolti», e nel codice compare
come `collate_fn`) pretende che abbiano tutti la stessa forma. Con le immagini ridimensionate è vero per costruzione; con il
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

I numeri cambiano a ogni esecuzione, perché le frasi sono sorteggiate, e
cambia con loro la larghezza del vassoio, che è la lunghezza della frase più
lunga capitata dentro. Quello che non cambia sono le tre forme: trentadue
righe, trentadue lunghezze, trentadue etichette.

```text
torch.Size([32, 39]) torch.Size([32]) torch.Size([32])
[20, 14, 14, 32, 36, 15, 39, 9] -> larghezza 39
```

Trentadue frasi portate tutte alla larghezza della più lunga del vassoio, qui
trentanove; e accanto le trentadue lunghezze vere, tutte diverse. La frase che
ne aveva nove è arrivata a trentanove con trenta zeri in coda.

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

Il campionamento pesato tocca i dati, che è l'ultima delle quattro leve contro
lo sbilanciamento: prima vengono la metrica, la soglia e il peso delle classi
(`weight` in `CrossEntropyLoss`), e la {doc}`sezione sulle classi
sbilanciate </MachineLearning/metriche>` le ordina dalla più economica alla
più invasiva.

Quella stessa sezione spiega perché, quando le classi sono sbilanciate così,
l'accuratezza smette di dire la verità. Basta un conto: se su duemila foto
millenovecento sono pizza, un modello che risponde «pizza» a occhi chiusi,
sempre, prende novantacinque su cento e non ha imparato niente. Servono misure
che guardino anche le classi rare, e sono precisione, richiamo e F1.

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

C'è un secondo modo di sbirciare, più difficile da vedere perché non sposta
nemmeno una fotografia. Prima di dare i numeri alla rete si guarda com'è fatta
la collezione (quanto è chiara in media, quanto variano i colori) per rimettere
tutto sulla stessa scala. Se per calcolare quelle misure si guarda anche il
mucchio d'esame, un pezzetto delle foto d'esame è già entrato nelle decisioni
prese prima dell'esame. Le misure si prendono sul mucchio d'addestramento e si
applicano tali e quali all'altro. Il regalo che ci si fa è piccolo, e basta a
far sembrare vincente un metodo che non lo è.
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
Nella maggior parte dei progetti che non riguardano i modelli giganti, la
colpa è invece del **caricamento dei dati**: la cucina finisce il vassoio e
resta ferma ad aspettare il successivo. Conviene tenerlo a mente, perché è la
diagnosi che quasi nessuno prova per prima e quasi sempre è quella giusta.

Come ci si accorge, se una scheda grafica c'è: si guarda quanto è occupata
mentre l'addestramento gira, con il comando `nvidia-smi` scritto in un'altra
finestra del terminale. Se sta al cento per cento in modo stabile, il collo di
bottiglia è il calcolo; se invece salta dal cento a zero e ritorno, la scheda
sta aspettando i dati, e ogni ottimizzazione del modello sarà tempo perso. Chi
lavora sulla sola CPU non ha quel termometro, e allora si cronometra a mano:
un'epoca intera, poi un'epoca in cui il modello non fa niente e si scorrono
soltanto i dati. Se i due tempi si somigliano, il modello non c'entra.

`````{tab} Elementare
I rimedi, dal più efficace al meno:

**Più aiutanti.** Alzare `num_workers`: se il problema è che nessuno prepara i
vassoi mentre la cucina cucina, è la prima cosa da provare.

**Ritagliare le foto una volta sola.** Se ogni epoca ridimensiona quattromila
fotografie da dodici megapixel a 224 pixel per lato, quel lavoro lo si sta
rifacendo identico decine di volte. Farlo una volta e salvare le immagini già
piccole su disco è un pomeriggio che si ripaga in un'ora.

**Meno file, più grandi.** Questo è il rimedio che stupisce, perché la ragione
non è quella che si immagina: il costo grosso sta nell’**aprirle**, più che nel
*leggerle*. Aprire un file è come chiedere al bibliotecario di andare a
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

Il rimedio dei file impacchettati guadagna per una ragione precisa. Su una
collezione grande il costo
dominante sta nell’**aprirli**, più che nel decodificarli: ogni `open()` è una
chiamata di sistema e un accesso ai metadati del filesystem, e un milione di
file piccoli produce un milione di accessi minuscoli e sparsi, che è lo schema
peggiore per qualunque disco e disastroso su uno storage di rete, dove ogni
accesso paga anche la latenza. Impacchettarli in pochi archivi letti in
sequenza sposta il lavoro dove l'hardware è veloce.
`````

C'è poi un secondo motivo per impacchettare i file, che si paga una volta e
serve per sempre. Mentre si scorre tutta la collezione per riscriverla, la si sta già
leggendo: costa zero calcolare intanto **media e deviazione standard di ogni
colore**, cioè i sei numeri che servono a `Normalize` e che qualche pagina fa
avevamo preso in prestito da ImageNet. Sui propri dati si calcolano, e vengono
meglio.

Due avvertenze, e sono le stesse di sempre. La prima: quei sei numeri si
calcolano **solo sulle foto di addestramento**, mai su tutte. Calcolarli su
tutte vuol dire far entrare nelle mie decisioni anche le foto d'esame, e il
voto smette di essere onesto, per la stessa perdita di informazione della
divisione fatta a caso su esempi che si assomigliano, e la {doc}`sezione su
overfitting e validazione </MachineLearning/overfitting-validazione>` la
tratta per esteso. La seconda: se si parte da un modello già addestrato da
altri, i sei numeri non si
calcolano affatto, si prendono quelli con cui è stato addestrato lui. Le
librerie li tengono insieme ai pesi proprio per questo, e un modello a cui si
danno immagini centrate diversamente da come le ricorda risponde peggio senza
dire niente.

Il tubo che porta i file dentro la rete è fatto: da qui in avanti si può
tornare a occuparsi del modello, sapendo che i dati arrivano. La sezione sulle
[prestazioni](prestazioni.md) riprenderà il discorso dall'altro lato, quello
del calcolo.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Un `Dataset` si scrive con **tre metodi**: la preparazione, che avviene una
  volta sola e dove va messo il lavoro lento; e le due domande che il
  `DataLoader` gli farà davvero, quanti esempi hai e dammi il numero 137 (la
  seconda gli verrà chiesta milioni di volte, quindi dev'essere veloce).
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
  segue l’**ordine alfabetico**, e va riletto da `.classes`, mai riscritto a
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
