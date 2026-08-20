# Classificazione e transfer learning

Nell'autunno del 2012 una rete neurale chiamata **AlexNet** vinse la
competizione ImageNet portando l'errore top-5 dal 26% al 15% circa (un salto
che nessun metodo precedente aveva nemmeno avvicinato). *Top-5* dice come si
contano gli errori: il modello dà le cinque etichette che ritiene più
probabili, e si segna un errore solo se quella giusta non è fra quelle cinque.
La visione artificiale non fu più la stessa
{cite}`krizhevsky2012imagenet`. Ma dietro
quel risultato c'erano 1,2 milioni di immagini etichettate e giorni di
addestramento su GPU, le schede grafiche che macinano molti conti in parallelo
e sono il motore di tutto il deep learning. La buona notizia è che quasi
nessuno di noi deve
ripetere quella fatica: possiamo *prendere in prestito* ciò che quelle reti
hanno già imparato. Si chiama **transfer learning** ed è, oggi, il modo
normale di costruire un classificatore di immagini.

## Dalla foto all'etichetta: la pipeline

Prima di riusarla, capiamo cosa fa una rete convoluzionale (CNN) quando
classifica un'immagine: la *pipeline*, cioè la catena di passaggi che porta
dalla foto all'etichetta.

`````{tab} Elementare

L'immagine entra come una griglia di pixel. La rete la fa passare attraverso
una pila di **strati convoluzionali**, quelli del capitolo precedente: ognuno
passa sull'immagine una lente piccola, sempre la stessa, e segna dove trova il
disegno che quella lente cerca. I primi riconoscono cose semplici (bordi,
angoli, macchie di colore), quelli più profondi combinano questi pezzetti in
forme via via più complesse: la trama di un pelo, un occhio, un muso. Alla fine
tutte queste "prove raccolte" vengono riassunte in una lista di numeri, e un
ultimo strato le trasforma in probabilità: `cane 0.82`, `gatto 0.11`, e così
via. Sono percentuali scritte come frazioni di uno, quindi `0.82` si legge
«82 per cento»; l'elenco continua con tutte le altre categorie, e sommando
tutta la lista si ottiene esattamente $1$. Vince l'etichetta con il numero più
alto.

`````

`````{tab} Superiore

L'input è un tensore $\mathbf{X} \in \mathbb{R}^{C\times H\times W}$ (canali,
altezza, larghezza: l'ordine *channels-first* di PyTorch, lo stesso dell'inizio
del capitolo). Una successione di blocchi convoluzione + non linearità + pooling
lo trasforma in una *feature map* sempre più piccola nello spazio ma più ricca
in profondità. Nelle architetture moderne un *global average pooling* la riduce
a un vettore $\mathbf{z}\in\mathbb{R}^d$ (le reti della prima generazione,
AlexNet e VGG, appiattivano invece la mappa e la mandavano in tre strati densi
pesantissimi: in AlexNet sono il 96% dei parametri, ed è il motivo per cui
«sostituire la testa» vuol dire due cose molto diverse sulle due famiglie), che
uno strato *fully-connected* con **softmax**
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

Una CNN moderna ha da qualche milione a decine di milioni di **parametri**: i
numeri interni che la rete regola mentre impara, un po’ come le manopole di un
impianto che si tarano una a una finché il suono non è giusto. Più manopole ci
sono, più esempi servono per trovare la posizione giusta di tutte. Per
regolarle senza andare in **overfitting**, cioè senza che la rete impari a
memoria gli esempi mostrati invece della regola che li spiega, servono
moltissimi esempi etichettati e molta potenza di calcolo. Con le poche
migliaia di foto di un progetto reale
(le lastre di un ambulatorio, i difetti su una linea di produzione, le specie
di una guida botanica), una rete addestrata da zero fa esattamente così: sulle
foto di addestramento risponde benissimo, su tutte le altre sbaglia. Il collo di bottiglia, quasi sempre, non è
l'algoritmo: sono i **dati** e il **tempo**.

## Prendere in prestito: transfer learning

L'idea del transfer learning è semplice: invece di ripartire da zero,
prendiamo una rete già addestrata su un grande dataset (quasi sempre ImageNet)
e la adattiamo al nostro compito. Funziona per una ragione precisa, da capire.

`````{tab} Elementare

Pensa a un cuoco che ha passato anni a imparare le tecniche di base: tagliare,
soffriggere, montare, impastare. Se domani deve preparare un piatto che non ha
mai fatto, non ricomincia da capo: quelle tecniche gli servono comunque, deve
solo imparare la ricetta nuova. Una CNN funziona allo stesso modo. I suoi
primi strati imparano le "tecniche di base" della visione (riconoscere bordi,
angoli, trame, colori) che valgono per qualunque immagine. Solo gli ultimi
strati imparano la "ricetta" specifica di ImageNet, cioè distinguere un
pastore tedesco da un labrador. Riusiamo le tecniche di base e riscriviamo
soltanto la ricetta.

E non è un modo di dire: qualcuno è andato a guardare. Si possono disegnare i
filtri che una rete si è costruita da sola, e nei primi strati escono sempre
le stesse cose, bordi orientati in tutte le direzioni e macchie di colore, in
qualunque rete e per qualunque compito sia stata addestrata. Andando in
profondità, invece, quello che ogni strato cerca diventa sempre più legato al
problema per cui è stata addestrata, e quindi sempre meno riusabile altrove.

`````

`````{tab} Superiore

Che i primi strati di una CNN imparino filtri generici non è un'ipotesi, ma un
fatto osservato. Visualizzando i pesi degli strati iniziali
{cite}`zeiler2014visualizing` si trovano rilevatori di bordi orientati e di
macchie di colore, molto simili ai filtri di Gabor della corteccia visiva;
salendo in profondità le unità rispondono a motivi via via più astratti e
specifici del compito. Yosinski e colleghi {cite}`yosinski2014transferable`
hanno quantificato questa *transferibilità*: le caratteristiche dei primi
strati si trasferiscono quasi senza perdita a compiti diversi, mentre quelle
degli ultimi strati sono tanto più specializzate (e meno riusabili) quanto più
ci si avvicina all'uscita. Da qui la strategia: congelare gli strati bassi
(generici) e riaddestrare quelli alti (specifici) sul nuovo dominio.

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
addestrata su ImageNet (nei diagrammi in inglese la troverete chiamata
*backbone*, la «spina dorsale») e ci attacchiamo sopra una **testa** nuova, con
tante uscite quante sono le nostre categorie: se le nostre foto vanno divise in
cinque gruppi, cinque uscite invece delle mille di ImageNet. Restano due modi
di procedere.

```{figure} ../figures/pooling-e-gerarchie.svg
:name: fig-gerarchia-pooling
:alt: "Una rete convoluzionale attraversata da sinistra a destra: le griglie delle attivazioni si rimpiccioliscono a ogni stadio, da quattro per quattro a due per due a una casella sola. In parallelo, ciò che gli strati rilevano passa dai bordi e dalle linee orientate alle forme geometriche composte, fino all'oggetto intero, riconosciuto come «gatto»."
:width: 100%

Le griglie si rimpiccioliscono, il significato cresce. Perdere risoluzione non
è un effetto collaterale del **pooling** (il passaggio che riassume ogni
quadratino di griglia in un numero solo, e così la rimpicciolisce): è il modo
in cui la rete smette di guardare i pixel e comincia a guardare le cose.
```

I due movimenti di {numref}`fig-gerarchia-pooling` vanno in verso opposto e
sono la ragione per cui il transfer learning funziona: la griglia si
rimpicciolisce, il significato cresce. Quello che si perde per strada, la
posizione esatta dei pixel, è la parte che vale in qualunque fotografia del
mondo; quello che si guadagna in fondo, l'oggetto riconosciuto, è la parte che
dipende dal compito. Ecco perché si riusa la prima e si rifà la seconda.

## Congelare o rifinire: feature extraction vs fine-tuning

`````{tab} Elementare

Sono due strade. Con la **feature extraction** blocchi (congeli) tutta la base:
la usi solo per trasformare le immagini in liste di numeri, e alleni da zero
soltanto la testa. Siccome i numeri da regolare sono pochissimi (quelli della
sola testa), bastano poche foto e pochi minuti di calcolo. Con il
**fine-tuning** sblocchi anche gli *ultimi* strati della base e li riaddestri
insieme alla testa, così la rete si adatta meglio al tuo **dominio**, cioè al
tipo di immagini di cui ti occupi tu: le tue lastre, le tue foglie, i tuoi
pezzi meccanici. Rende di più, ma vuole più foto e va fatto con cautela.

La cautela serve perché una rete non ha un cassetto dei ricordi separato: tutto
quello che sa sta in quegli stessi numeri, e riaddestrarla vuol dire
spostarli. Se li si sposta di poco, si aggiusta; se li si sposta di molto, si
cancella quello che c'era prima e si ricomincia da capo senza accorgersene.

`````

`````{tab} Superiore

Nella **feature extraction** congeliamo l'intera base con
`requires_grad_(False)`: autograd smette di calcolarne i gradienti, i
parametri $\theta_{\text{base}}$ restano fissi e all'ottimizzatore
consegniamo solo la testa $\theta_{\text{head}}$. Attenzione però ai layer di
*Batch Normalization*: le loro statistiche correnti (media e varianza) sono
**buffer**, non parametri, e in modalità `train()` si aggiornano a ogni
forward, con o senza autograd. Una base «congelata» solo con
`requires_grad_(False)` deriva quindi comunque verso il nuovo dominio: per
fermarla davvero, i moduli BatchNorm vanno messi in modalità valutazione
(`.eval()`).

Nel **fine-tuning** riattiviamo il gradiente sugli strati alti della base e
riprendiamo l'ottimizzazione con un learning rate **molto basso** (tipicamente
$10^{-5}$ contro $10^{-3}$): passi grandi sovrascriverebbero le
rappresentazioni utili. Due accortezze: si scongelano solo gli strati alti (i
bassi sono i più generici) e i BatchNorm restano anche qui in `.eval()`, per
non destabilizzare con i piccoli batch del fine-tuning le statistiche apprese
su ImageNet.

`````

## In pratica, con PyTorch

`torchvision.models`, la libreria di visione che accompagna PyTorch, include
decine di reti già addestrate su ImageNet, pronte da scaricare. Usiamo
**ResNet-18** {cite}`he2016deep`, dove il numero è semplicemente il conto degli
strati: compatta, collaudata, e il modello più leggero della sua famiglia.
Il compito è dividere le foto in cinque categorie. Prima la feature extraction.

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

# ...e anche i BatchNorm, che altrimenti continuerebbero a cambiare da soli:
# il perche' e' nella tab Superiore. Da ripetere dopo ogni model.train().
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.eval()

# ...e si sostituisce la testa: dalle 1000 classi ImageNet alle nostre 5
model.fc = nn.Linear(model.fc.in_features, 5)   # nuova, addestrabile

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
```

L'addestramento è il **training loop** del {doc}`capitolo su PyTorch </PyTorch/overview>`, il ciclo che
mostra alla rete un mucchietto di immagini alla volta, guarda quanto ha
sbagliato e sposta i suoi numeri un pochino nella direzione giusta. «Un
pochino» quanto, lo decide un valore che si chiama *learning rate*, alla
lettera «velocità di apprendimento»: è la lunghezza del passo. Quando la testa
ha smesso di migliorare, passiamo al fine-tuning degli ultimi strati
accorciando molto quel passo.

```python
# 3. Si scongela solo l'ultimo blocco della base
for p in model.layer4.parameters():
    p.requires_grad_(True)

optimizer = optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-5,   # lr basso: cautela
)
```

Con poche centinaia di immagini per classe questa ricetta arriva, in pochi
minuti di calcolo, dove una rete addestrata da zero sugli stessi dati non
arriva affatto: con così pochi esempi quella rete impara a memoria prima di
aver imparato a vedere. Quanti dati le servirebbero per rifarsi da sola le
tecniche di base non è una domanda con una risposta sola. Dipende da quanto le
nostre immagini somigliano a quelle su cui la base è stata addestrata, ed è la
stessa cosa da cui dipende quanto rende il transfer learning: più il dominio è
lontano da ImageNet (le radiografie, le immagini satellitari, il microscopio),
meno c'è da riusare e più c'è da riaddestrare. Sostituendo `resnet18` con
`resnet50` o con
`efficientnet_b0` {cite}`tan2019efficientnet` la struttura del codice non
cambia: è il
bello del transfer learning.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Una rete che classifica lavora a strati: i primi vedono bordi e macchie di
  colore, gli ultimi forme e oggetti interi. Alla fine tutto si riduce a una
  lista di numeri, e l'ultimo passaggio la trasforma in percentuali, una per
  classe: vince la più alta.
- Costruire una rete del genere da zero costa **troppe foto e troppo tempo**.
  Il **transfer learning** è la scorciatoia: si prende una rete che qualcun
  altro ha già addestrato su milioni di immagini e le si cambia solo la punta.
- Il cuoco che sa già tagliare e soffriggere deve imparare solo la ricetta
  nuova: i primi strati (le tecniche di base) valgono per qualunque
  fotografia, gli ultimi (la ricetta) no, e sono quelli da rifare.
- Due modi di procedere. **Bloccare** tutta la base e allenare solo la punta:
  veloce, e basta poco materiale. Oppure **sbloccare anche gli ultimi strati**
  della base e ritoccarli a passi piccolissimi: rende di più, ma vuole più
  foto e più cautela, perché a passi grandi la rete dimentica quello che
  sapeva.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Una CNN classifica estraendo caratteristiche via via più astratte e
  chiudendo con **softmax** sulle classi.
- Addestrare da zero è spesso proibitivo: servono troppi **dati** e troppo
  **tempo**. Il transfer learning riusa una base già addestrata su ImageNet.
- **Feature extraction**: base congelata, si allena solo la testa (veloce,
  pochi dati). Congelarla davvero vuol dire due cose, non una: togliere i
  gradienti *e* mettere i BatchNorm in `.eval()`, perché le loro statistiche
  sono buffer e in `train()` deriverebbero comunque verso il nuovo dominio.
  **Fine-tuning**: si scongelano gli strati alti con learning rate piccolo
  (più preciso, più dati), BatchNorm sempre in `.eval()`.
```

`````
