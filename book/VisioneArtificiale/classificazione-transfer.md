# Classificazione e transfer learning

Nell'autunno del 2012 una rete neurale chiamata **AlexNet** vinse la
competizione ImageNet portando l'errore top-5 dal 26% al 15% circa — un salto
che nessun metodo precedente aveva nemmeno avvicinato — e la visione
artificiale non fu più la stessa {cite}`krizhevsky2012imagenet`. Ma dietro
quel
risultato c'erano 1,2 milioni di immagini etichettate e giorni di addestramento
su GPU. La buona notizia è che quasi nessuno di noi deve ripetere quella fatica:
possiamo *prendere in prestito* ciò che quelle reti hanno già imparato. Si
chiama **transfer learning** ed è, oggi, il modo normale di costruire un
classificatore di immagini.

## Dalla foto all'etichetta: la pipeline

Prima di riusarla, capiamo cosa fa una rete convoluzionale (CNN) quando
classifica un'immagine: la *pipeline*, cioè la catena di passaggi che porta
dalla foto all'etichetta.

`````{tab} Elementare

L'immagine entra come una griglia di pixel. La rete la fa passare attraverso
una pila di **strati convoluzionali**: i primi riconoscono cose semplici —
bordi, angoli, macchie di colore — quelli più profondi combinano questi
pezzetti in forme via via più complesse: una texture, un occhio, un muso.
Alla fine tutte queste "prove raccolte" vengono riassunte in una lista di
numeri (le *caratteristiche*), e un ultimo strato le trasforma in
probabilità: `cane 0.82`, `gatto 0.11`, e così via. Vince l'etichetta con il
numero più alto.

`````

`````{tab} Superiore

L'input è un tensore $X \in \mathbb{R}^{c\times h\times w}$ (canali, altezza,
larghezza: l'ordine *channels-first* di PyTorch). Una successione di blocchi convoluzione + non linearità + pooling lo
trasforma in una *feature map* sempre più piccola nello spazio ma più ricca in
profondità. Un *global average pooling* la riduce a un vettore
$\mathbf{z}\in\mathbb{R}^d$, che uno strato *fully-connected* con **softmax**
mappa in una distribuzione sulle $K$ classi:

$$
\hat{y}_k = \frac{e^{\mathbf{w}_k^\top \mathbf{z}+b_k}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^\top \mathbf{z}+b_j}} .
$$

Qui $\hat{y}_k$ è la probabilità stimata della classe $k$ e $\mathbf{w}_k$ è la
riga di pesi ad essa associata. L'addestramento minimizza la *cross-entropy*
$\mathcal{L} = -\sum_k y_k \log \hat{y}_k$ ottimizzando tutti i parametri
$\theta$ della rete.

`````

## Perché partire da zero costa caro

Una CNN moderna ha da qualche milione a decine di milioni di parametri. Per
stimarli senza andare in **overfitting** servono moltissimi esempi etichettati
e molta potenza di calcolo. Con le poche migliaia di foto di un progetto reale
— le lastre di un ambulatorio, i difetti su una linea di produzione, le specie
di una guida botanica — una rete addestrata da zero impara a memoria il
training set e fallisce sul resto. Il collo di bottiglia, quasi sempre, non è
l'algoritmo: sono i **dati** e il **tempo**.

## Prendere in prestito: transfer learning

L'idea del transfer learning è semplice: invece di ripartire da zero,
prendiamo una rete già addestrata su un grande dataset — quasi sempre ImageNet
— e la adattiamo al nostro compito. Funziona per una ragione precisa, che vale
la pena capire.

`````{tab} Elementare

Pensa a un cuoco che ha passato anni a imparare le tecniche di base: tagliare,
soffriggere, montare, impastare. Se domani deve preparare un piatto che non ha
mai fatto, non ricomincia da capo — quelle tecniche gli servono comunque, deve
solo imparare la ricetta nuova. Una CNN funziona allo stesso modo. I suoi
primi strati imparano le "tecniche di base" della visione — riconoscere bordi,
angoli, texture, colori — che valgono per qualunque immagine. Solo gli ultimi
strati imparano la "ricetta" specifica di ImageNet, cioè distinguere un
pastore tedesco da un labrador. Riusiamo le tecniche di base e riscriviamo
soltanto la ricetta.

`````

`````{tab} Superiore

Che i primi strati di una CNN imparino filtri generici non è un'ipotesi, ma un
fatto osservato. Visualizzando i pesi degli strati iniziali
{cite}`zeiler2014visualizing` si trovano rilevatori di bordi orientati e di
macchie di colore, molto
simili ai filtri di Gabor della corteccia visiva; salendo in profondità le
unità rispondono a motivi via via più astratti e specifici del compito.
Yosinski e colleghi {cite}`yosinski2014transferable` hanno quantificato
questa *transferibilità*: le
caratteristiche dei primi strati si trasferiscono quasi senza perdita a compiti
diversi, mentre quelle degli ultimi strati sono tanto più specializzate — e
meno riusabili — quanto più ci si avvicina all'uscita. Da qui la strategia:
congelare gli strati bassi (generici) e riaddestrare quelli alti (specifici)
sul nuovo dominio.

`````

Riusiamo dunque la parte generica già addestrata e sostituiamo solo la punta.

```{figure} ../figures/transfer-learning.svg
:name: fig-transfer
:alt: Un'immagine entra in una base convoluzionale pre-addestrata su ImageNet con i pesi congelati, seguita da una testa di classificazione nuova e addestrabile che produce le probabilità delle classi.
:width: 90%

La rete pre-addestrata (in teal) fa da estrattore di caratteristiche; sopra
di essa montiamo una testa nuova (in terracotta) per il nostro compito.
```

Come mostra {numref}`fig-transfer`, teniamo la **base convoluzionale**
addestrata su ImageNet e ci attacchiamo sopra una **testa** nuova, dimensionata
sulle nostre classi. Restano due modi di procedere.

## Congelare o rifinire: feature extraction vs fine-tuning

`````{tab} Elementare

Sono due manopole. Con la **feature extraction** blocchi (congeli) tutta la
base: la usi solo per trasformare le immagini in liste di numeri, e alleni da
zero soltanto la testa. È veloce e basta poco: bene con pochi dati. Con il
**fine-tuning** sblocchi anche gli *ultimi* strati della base e li riaddestri
insieme alla testa, così la rete si adatta meglio al tuo dominio. Rende di
più, ma vuole più dati e va fatto con cautela per non "sciupare" ciò che la
rete già sapeva.

`````

`````{tab} Superiore

Nella **feature extraction** congeliamo l'intera base con
`requires_grad_(False)`: autograd smette di calcolarne i gradienti, i
parametri $\theta_{\text{base}}$ restano fissi e all'ottimizzatore
consegniamo solo la testa $\theta_{\text{head}}$. Nel **fine-tuning**
riattiviamo il gradiente sugli strati alti della base e riprendiamo
l'ottimizzazione con un learning rate **molto basso** (tipicamente $10^{-5}$
contro $10^{-3}$): passi grandi sovrascriverebbero le rappresentazioni utili.
Due accortezze: si scongelano solo gli strati alti (i bassi sono i più
generici) e i layer di *Batch Normalization* si tengono in modalità
valutazione (`.eval()`), per non destabilizzare con i piccoli batch del
fine-tuning le statistiche apprese su ImageNet.

`````

## In pratica, con PyTorch

`torchvision.models` include decine di reti già addestrate. Usiamo
**ResNet-18** {cite}`he2016deep`, compatta e collaudata, per un compito a 5
classi. Prima la feature extraction.

```python
import torch
from torch import nn, optim
from torchvision import models

# 1. Rete pre-addestrata su ImageNet, con le sue trasformazioni
pesi = models.ResNet18_Weights.IMAGENET1K_V1
model = models.resnet18(weights=pesi)
preprocess = pesi.transforms()   # resize+crop a 224x224, normalizzazione ImageNet

# 2. Feature extraction: si congela tutta la base...
for p in model.parameters():
    p.requires_grad_(False)

# ...e si sostituisce la testa: dalle 1000 classi ImageNet alle nostre 5
model.fc = nn.Linear(model.fc.in_features, 5)   # nuova, addestrabile

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
```

L'addestramento è il solito training loop del capitolo su PyTorch, sui batch
del nostro dataset trasformati da `preprocess`. Quando la testa ha smesso di
migliorare, passiamo al fine-tuning degli ultimi strati con un learning rate
ridotto.

```python
# 3. Si scongela solo l'ultimo blocco della base
for p in model.layer4.parameters():
    p.requires_grad_(True)

optimizer = optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-5,   # lr basso: cautela
)
```

Con poche centinaia di immagini per classe, questa ricetta raggiunge in pochi
minuti accuratezze che una rete addestrata da zero non vedrebbe nemmeno con
dieci volte i dati. Sostituendo `resnet18` con `resnet50` o con
`efficientnet_b0` {cite}`tan2019efficientnet` la struttura del codice non
cambia: è il
bello del transfer learning.

```{admonition} Da ricordare
:class: important
- Una CNN classifica estraendo caratteristiche via via più astratte e
  chiudendo con **softmax** sulle classi.
- Addestrare da zero è spesso proibitivo: servono troppi **dati** e troppo
  **tempo**. Il transfer learning riusa una base già addestrata su ImageNet.
- **Feature extraction**: base congelata, si allena solo la testa — veloce,
  pochi dati. **Fine-tuning**: si scongelano gli strati alti con learning rate
  piccolo — più preciso, più dati.
```
