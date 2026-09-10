# Classificazione e transfer learning

Nell'autunno del 2012 una rete neurale chiamata AlexNet vinse la
competizione ImageNet portando l'errore top-5 dal 26% al 15% circa (un salto
che nessun metodo precedente aveva nemmeno avvicinato). *Top-5* dice come si
contano gli errori: il modello dà le cinque etichette che ritiene più
probabili, e si segna un errore solo se quella giusta non è fra quelle cinque.
La visione artificiale non fu più la stessa
{cite}`krizhevsky2012imagenet`. Ma dietro
quel risultato c'erano 1,2 milioni di immagini etichettate in mille categorie e
giorni di addestramento su GPU, le schede grafiche che macinano molti conti in
parallelo e sono il motore di tutto il deep learning. La buona notizia è che
quasi nessuno di noi deve
ripetere quella fatica: possiamo *prendere in prestito* ciò che quelle reti
hanno già imparato. Si chiama **transfer learning** ed è, oggi, il modo
normale di costruire un classificatore di immagini.

## Dalla foto all'etichetta: la pipeline

Prima di riusarla, capiamo cosa fa una rete convoluzionale (CNN) quando
classifica un'immagine. La catena di passaggi che porta dalla foto
all'etichetta si chiama **pipeline**, che in inglese è la conduttura: una fila
di stazioni in cui ognuna prende quello che le consegna la precedente e passa
avanti il proprio risultato. Il nome vale per qualunque catena fatta così, non
solo per questa.

`````{tab} Elementare

L'immagine entra come una griglia di pixel. La rete la fa passare attraverso
una pila di strati convoluzionali, quelli della {doc}`sezione sulle reti
convoluzionali </DeepLearning/reti-convoluzionali>`: ognuno passa
sull'immagine una lente piccola, sempre la stessa (il suo nome tecnico è
*filtro*), e segna dove trova il disegno che quella lente cerca. I primi
riconoscono cose semplici (bordi, angoli, macchie di colore), quelli più
profondi combinano questi pezzetti in
forme via via più complesse: la trama di un pelo, un occhio, un muso. Alla fine
tutte queste "prove raccolte" vengono riassunte in una lista di numeri, e un
ultimo strato le trasforma in probabilità: `cane 0.82`, `gatto 0.11`, e così
via. Sono percentuali scritte come frazioni di uno, quindi `0.82` si legge
«82 per cento»; l'elenco continua con tutte le altre categorie, e sommando
tutta la lista si ottiene esattamente $1$. Vince l'etichetta con il numero più
alto.

Come fa la rete ad avere le lenti giuste? Le si mostrano foto di cui
l'etichetta si conosce già, e di tutta la lista delle probabilità se ne guarda
una voce sola: quella dell'etichetta vera. Un `0.82` lì sopra costa poco, un
`0.11` costa molto, e a ogni foto la rete corregge di pochissimo tutto quello
che ha dentro, dalla prima lente all'ultimo strato, per far salire quella
voce. Ripetuto su milioni di foto, questo è l'addestramento.

Quello che la rete ha dentro non è distribuito alla pari. La punta, cioè gli
strati che vengono dopo le lenti, in una rete moderna tiene pochi numeri; nelle
prime reti di questo genere era il contrario. In AlexNet novantasei numeri su
cento stavano nei suoi ultimi tre strati, e rifare quella punta voleva dire
rifare quasi tutta la rete: è il motivo per cui, su una rete di allora e su una
di oggi, cambiare la punta è un lavoro di due misure diverse.

`````

`````{tab} Superiore

L'input è un tensore $\mathbf{X} \in \mathbb{R}^{C\times H\times W}$ (canali,
altezza, larghezza: l'ordine *channels-first* di PyTorch). Una successione di
blocchi convoluzione + batch normalization + non linearità
lo trasforma in una *feature map* sempre più piccola nello spazio ma più ricca
in profondità: a rimpicciolirla è il pooling nelle architetture della prima
generazione e il passo di convoluzione maggiore di uno in quelle di oggi (in
ResNet-18, due soli strati di pooling contro sette convoluzioni a passo due).
Alla fine un *global average pooling* riduce la mappa
a un vettore $\mathbf{z}\in\mathbb{R}^d$, con $d = 512$ in ResNet-18 (le reti
della prima generazione, AlexNet e VGG, appiattivano invece la mappa e la
mandavano in tre strati densi pesantissimi: in AlexNet sono il 96% dei
$60\,965\,224$ parametri della versione dell'articolo, ed è il motivo per cui
«sostituire la testa» vuol dire due cose molto diverse sulle due famiglie), che
uno strato *fully-connected* seguito dalla softmax
mappa in una distribuzione sulle $K$ classi:

$$
\hat{y}_k = \frac{e^{\mathbf{w}_k^\top \mathbf{z}+b_k}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^\top \mathbf{z}+b_j}} .
$$

Qui $\hat{y}_k$ è la probabilità stimata della classe $k$, $\mathbf{w}_k$ è la
riga di pesi ad essa associata e $b_k$ il suo termine costante.
Su un esempio il costo è la *cross-entropy*
$\ell = -\sum_k y_k \log \hat{y}_k$, dove $y_k$ vale 1 sulla classe vera
e 0 sulle altre (resta quindi il solo termine $-\log \hat{y}_{\text{vera}}$);
l'addestramento minimizza la sua media $\mathcal{L}$ sull'insieme, ottimizzando
tutti i parametri $\theta$ della rete. La softmax sta nella formula e non nel
modulo: `nn.CrossEntropyLoss` vuole i logit e applica la softmax al proprio
interno, quindi la testa che si scrive in codice è una `nn.Linear` nuda
({doc}`Moduli, strati, loss </PyTorch/moduli>`).

`````

## Perché partire da zero costa caro

Una CNN moderna ha da qualche milione a decine di milioni di parametri: i
numeri interni che la rete regola mentre impara, un po’ come le manopole di un
impianto che si tarano una a una finché il suono non è giusto. Più manopole ci
sono, più esempi servono per trovare la posizione giusta di tutte. Per
regolarle senza andare in overfitting, cioè senza che la rete impari a
memoria gli esempi mostrati invece della regola che li spiega, servono
moltissimi esempi etichettati e molta potenza di calcolo. Con le poche
migliaia di foto di un progetto reale
(le lastre di un ambulatorio, i difetti su una linea di produzione, le specie
di una guida botanica), una rete addestrata da zero fa esattamente così: sulle
foto di addestramento risponde benissimo, su tutte le altre sbaglia. Il collo
di bottiglia, quasi sempre, sono i dati e il tempo, non l'algoritmo.

## Prendere in prestito: transfer learning

L'idea del transfer learning è semplice: invece di ripartire da zero,
prendiamo una rete già addestrata su un grande dataset (quasi sempre ImageNet)
e la adattiamo al nostro compito. Funziona per una ragione precisa, da capire.

`````{tab} Elementare

Un cuoco ha passato anni a imparare le tecniche di base: tagliare, soffriggere,
montare, impastare. Se domani deve preparare un piatto che non ha mai fatto,
non ricomincia da capo: quelle tecniche gli servono comunque, deve solo
imparare la ricetta nuova. Una CNN funziona allo stesso modo. I suoi primi
strati imparano le "tecniche di base" della visione (riconoscere bordi, angoli,
trame, colori) che valgono per qualunque immagine. Solo gli ultimi strati
imparano la "ricetta" specifica di ImageNet, cioè distinguere un pastore
tedesco da un labrador. Riusiamo le tecniche di base e riscriviamo soltanto la
ricetta.

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
ci si avvicina all'uscita. A far scendere il rendimento del trasferimento,
però, non è solo la specializzazione: pesa anche il taglio in sé, perché separa
neuroni che si erano adattati l'uno all'altro, e il conto si paga pure
trasferendo un compito su sé stesso. Da qui la strategia: congelare gli strati
bassi (generici) e riaddestrare quelli alti (specifici) sul nuovo dominio,
sapendo che partire da pesi trasferiti conviene quasi sempre, anche quando poi
si rifinisce tutto.

`````

Riusiamo dunque la parte generica già addestrata e sostituiamo solo la punta,
quella che d'ora in avanti chiameremo la testa.

```{figure} ../figures/transfer-learning.svg
:name: fig-transfer
:alt: Un'immagine entra in una base convoluzionale pre-addestrata su ImageNet con i pesi congelati, seguita da una testa di classificazione nuova e addestrabile che produce le probabilità delle classi.
:width: 90%

La rete pre-addestrata (in teal, il verde-azzurro) fa da estrattore di
caratteristiche: riduce l'immagine alla lista di numeri che la descrivono.
Sopra di essa montiamo una testa nuova (in terracotta) per il nostro compito.
```

Come mostra {numref}`fig-transfer`, teniamo la **base convoluzionale**
addestrata su ImageNet (nei diagrammi in inglese la troverete chiamata
*backbone*, la «spina dorsale») e ci attacchiamo sopra una testa nuova, con
tante uscite quante sono le nostre categorie: se le nostre foto vanno divise in
cinque gruppi, cinque uscite invece delle mille di ImageNet. Restano due modi
di procedere.

```{figure} ../figures/pooling-e-gerarchie.svg
:name: fig-gerarchia-pooling
:alt: "Una rete convoluzionale attraversata da sinistra a destra: le griglie delle attivazioni si rimpiccioliscono a ogni stadio, da quattro per quattro a due per due a una casella sola. In parallelo, ciò che gli strati rilevano passa dai bordi e dalle linee orientate alle forme geometriche composte, fino all'oggetto intero, riconosciuto come «gatto»."
:width: 100%

Le griglie si rimpiccioliscono, il significato cresce. Perdere risoluzione non
è un effetto collaterale del pooling (il passaggio che riassume ogni
quadratino di griglia in un numero solo, e così la rimpicciolisce): è il modo
in cui la rete smette di guardare i pixel e comincia a guardare le cose.
```

I due movimenti di {numref}`fig-gerarchia-pooling` non si riusano allo stesso
modo. Buttare via la posizione
esatta dei pixel per costruire forme sempre più grandi serve in qualunque
fotografia del mondo; riconoscere in fondo alla pila proprio le mille categorie
di ImageNet serve solo lì. Ecco perché la pila si riusa quasi tutta, e a rifarsi
è la testa.

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
cancella quello che c'era prima e si ricomincia da capo senza accorgersene. Per
questo il fine-tuning si fa con spostamenti cento volte più piccoli di quelli
con cui si allena la testa.

E la base congelata riserva una sorpresa: continua a cambiare da sola. Dentro
ci sono gli strati di *batch normalization*, quelli che prima di passare i
numeri allo strato dopo li rimettono in scala su quanto sono chiare e variegate
le foto che la rete sta guardando in quel momento, come la macchina fotografica
che regola da sé la luce sulla media di quello che ha inquadrato. Di quelle
medie la rete tiene anche un appunto, che non è una delle sue manopole e si
aggiorna da sé ogni volta che una foto la attraversa, anche dopo che le
manopole sono state bloccate tutte. Chi congela le sole manopole manda dentro
le proprie lastre e si ritrova, senza essersene accorto, una base tarata sulle
lastre: la testa impara inseguendo un bersaglio che si sposta, e le liste di
numeri su cui si allena adesso non sono più quelle di prima. Perché la base
resti ferma davvero bisogna dire anche a quegli strati di smettere di prendere
appunti; lo si dice pure quando si sbloccano gli ultimi strati, perché una
media presa su un gruppetto di foto alla volta salta da un gruppetto all'altro.

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
riprendiamo l'ottimizzazione con un learning rate molto basso (tipicamente
$10^{-5}$ contro $10^{-3}$): passi grandi sovrascriverebbero le
rappresentazioni utili. Due accortezze: si scongelano solo gli strati alti (i
bassi sono i più generici) e i BatchNorm restano anche qui in `.eval()`, per
non destabilizzare con i piccoli batch del fine-tuning le statistiche apprese
su ImageNet.

`````

## In pratica, con PyTorch

`torchvision.models`, la libreria di visione che accompagna PyTorch, include
decine di reti già addestrate su ImageNet, pronte da scaricare. Usiamo
ResNet-18 {cite}`he2016deep`, dove il numero conta gli strati con pesi sul
percorso principale (diciassette convoluzioni più la testa, e restano fuori le
tre convoluzioni di raccordo): compatta, collaudata, e il modello più leggero
della sua famiglia.
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
# le statistiche sono buffer, non parametri. Da ripetere dopo model.train().
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.eval()

# ...e si sostituisce la testa: dalle 1000 classi ImageNet alle nostre 5
model.fc = nn.Linear(model.fc.in_features, 5)   # nuova, addestrabile

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
```

L'addestramento è il {doc}`training loop </PyTorch/addestramento>`, il ciclo
che mostra alla rete un mucchietto di immagini alla volta, guarda quanto ha
sbagliato e sposta i suoi numeri un pochino nella direzione giusta. «Un
pochino» quanto, lo decide un valore che si chiama *learning rate*, alla
lettera «velocità di apprendimento»: è la lunghezza del passo. Quando la testa
ha smesso di migliorare, passiamo al fine-tuning degli ultimi strati
accorciando molto quel passo.

```python
# 3. Si scongela solo l'ultimo blocco della base
for p in model.layer4.parameters():
    p.requires_grad_(True)

# I BatchNorm vanno rimessi in valutazione: il training loop ha chiamato
# model.train(), che li ha riportati tutti in modalità di addestramento.
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.eval()

optimizer = optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-5,   # lr basso: cautela
)

addestrabili = sum(p.numel() for p in model.parameters() if p.requires_grad)
totali = sum(p.numel() for p in model.parameters())
print(f"addestrabili {addestrabili} su {totali}")
```

Quel «solo l'ultimo blocco» è meno modesto di come suona: in una ResNet-18
`layer4` tiene da solo tre quarti dei pesi, perché in una rete residua i
parametri si addensano verso l'uscita, e la cautela sul passo non è prudenza
di maniera.

Con poche centinaia di immagini per classe questa ricetta arriva dove una rete
addestrata da zero sugli stessi dati resta molto indietro: con così pochi
esempi quella rete impara a memoria prima di
aver imparato a vedere. Quanti dati le servirebbero per rifarsi da sola le
tecniche di base non è una domanda con una risposta sola. Dipende da quanto le
nostre immagini somigliano a quelle su cui la base è stata addestrata, ed è la
stessa cosa da cui dipende quanto rende il transfer learning: più il dominio è
lontano da ImageNet (le radiografie, le immagini satellitari, il microscopio),
meno c'è da riusare e più c'è da riaddestrare. Sostituendo `resnet18` con
`resnet50` la struttura del codice non cambia. Cambiando famiglia cambiano i
nomi, e vanno guardati: in `efficientnet_b0` {cite}`tan2019efficientnet` la
testa si chiama `classifier[1]` e non `fc`, e gli strati alti stanno dentro
`features` e non in un `layer4`. E su una famiglia che al posto della batch
normalization usa la LayerNorm, come i Vision Transformer, di BatchNorm non ce
n'è nessuno: quel ciclo gira a vuoto senza protestare, e la deriva delle
statistiche non c'è, perché la LayerNorm statistiche non ne conserva.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Una rete che classifica lavora a strati: i primi vedono bordi e macchie di
  colore, gli ultimi forme e oggetti interi. Alla fine tutto si riduce a una
  lista di numeri, e l'ultimo passaggio la trasforma in percentuali, una per
  classe: vince la più alta.
- Costruire una rete del genere da zero costa troppe foto e troppo tempo.
  Il transfer learning è la scorciatoia: si prende una rete che qualcun
  altro ha già addestrato su milioni di immagini e le si cambia solo la punta.
- Il cuoco che sa già tagliare e soffriggere deve imparare solo la ricetta
  nuova: i primi strati (le tecniche di base) valgono per qualunque
  fotografia, gli ultimi (la ricetta) no, e sono quelli da rifare.
- Due modi di procedere. Bloccare tutta la base e allenare solo la punta:
  veloce, e basta poco materiale; bloccarla davvero però vuol dire due cose e
  non una, fermare le manopole e dire anche agli strati di *batch
  normalization* di smettere di prendere appunti. Oppure sbloccare anche gli
  ultimi strati della base e ritoccarli a passi piccolissimi: rende di più,
  ma vuole più foto e più cautela, perché a passi grandi la rete dimentica
  quello che sapeva.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Una CNN classifica estraendo caratteristiche via via più astratte e
  chiudendo con softmax sulle classi.
- Addestrare da zero è spesso proibitivo: servono troppi dati e troppo
  tempo. Il transfer learning riusa una base già addestrata su ImageNet.
- Feature extraction: base congelata, si allena solo la testa (veloce,
  pochi dati). Congelarla davvero vuol dire due cose, non una: togliere i
  gradienti *e* mettere i BatchNorm in `.eval()`, perché le loro statistiche
  sono buffer e in `train()` deriverebbero comunque verso il nuovo dominio.
  Fine-tuning: si scongelano gli strati alti con learning rate piccolo
  (più preciso, più dati), BatchNorm sempre in `.eval()`.
```

`````
