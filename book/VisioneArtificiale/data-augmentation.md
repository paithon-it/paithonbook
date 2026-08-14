# Data augmentation: moltiplicare i dati senza raccoglierli

Per insegnare a un bambino che cos'è una tazza non servono diecimila tazze. Ne
basta una: la gira tra le mani, la guarda dall'alto e di lato, la vede al sole
e in penombra, mezza nascosta dietro la caffettiera. Ogni sguardo è
un'immagine diversa dello stesso oggetto, e da quella manciata di occhiate il
concetto di "tazza" esce solidissimo. Una rete neurale non ha mani. Ma
possiamo girare noi l'oggetto al posto suo: prendere ogni fotografia del
training set e mostrargliela specchiata, ritagliata, un po' ruotata, più
chiara o più scura. Si chiama **data augmentation** (letteralmente "aumento
dei dati") ed è il modo più economico che esista per moltiplicare gli esempi
senza raccoglierne di nuovi. Non è un'idea recente: nel capitolo sul deep
learning abbiamo visto che già AlexNet, nel 2012, la usava in modo aggressivo
(ritagli casuali, riflessioni, perturbazioni di colore). Bastano i primi due,
calcolano gli autori, per ricavare da ogni immagine oltre duemila varianti
possibili {cite}`krizhevsky2012imagenet`, ed è una moltiplicazione, non una
magia. Il conto si rifà a mano. Da un'immagine di 256 pixel di lato se ne
ritaglia una di 224: il bordo sinistro del ritaglio può quindi scivolare di
$256 - 224 = 32$ colonne, e il bordo alto di altrettante righe, il che fa
$32 \times 32 = 1024$ ritagli diversi. Lo specchio li raddoppia, e si arriva a
$2048$. (A essere pignoli le posizioni sono $33 \times 33$, perché una la
occupa il ritaglio tutto a sinistra e le altre trentadue sono gli scivolamenti:
gli autori arrotondano alla potenza di due, che è il numero tondo per un
computer, e la sostanza non cambia.)

## Cambiare i pixel, non l'etichetta

Tutto il trucco sta in un vincolo, che va capito bene: una trasformazione è
ammessa solo se cambia l'immagine ma **non cambia l'etichetta**. Le
trasformazioni buone catturano le *invarianze* del compito: i modi in cui il
mondo può variare senza che la risposta giusta vari. E qui serve onestà:
nessuna trasformazione è innocente in assoluto.

`````{tab} Elementare
Specchia la foto di un gatto: è ancora, senza alcun dubbio, un gatto. La
natura non distingue tra gatti "che guardano a destra" e gatti "che guardano a
sinistra", quindi lo specchio è un modo gratuito di raddoppiare le foto di
gatti. Ma prova a specchiare un cartello stradale: l'obbligo di svolta a
destra diventa un obbligo di svolta a *sinistra* (stessa grafica, significato
opposto). O prendi la cifra 6 e capovolgila: ottieni un 9. Una scritta ruotata
di novanta gradi smette di essere leggibile. La stessa mossa che per i gatti è
un regalo, per i cartelli o per le cifre è un sabotaggio: la trasformazione
giusta dipende dal *compito*, non dall'immagine.
`````

`````{tab} Superiore
Sia $(\mathbf{x}, y)$ una coppia immagine–etichetta. Una trasformazione $T$ è
ammessa per il compito se la coppia $(T(\mathbf{x}), y)$ è ancora un esempio
plausibile della stessa distribuzione: l'etichetta resta valida e l'immagine trasformata
somiglia a qualcosa che il modello potrà davvero incontrare. L'insieme delle
trasformazioni ammesse è **conoscenza a priori sul dominio** che iniettiamo
nel modello: il flip orizzontale appartiene alle invarianze di "gatto contro
cane", non a quelle del riconoscimento di cifre (un 3 specchiato non è
nessuna cifra, e nemmeno le rotazioni ampie sono innocue: il 6 capovolto è
un 9) né a quelle dei cartelli stradali, dove lo specchio inverte il
significato. Conta
anche l'intensità: una rotazione di 5 gradi su una foto di strada è
realistica, una di 90 gradi produce pedoni sdraiati che in produzione non
esistono. Scegliere le trasformazioni è progettazione, non un dettaglio
tecnico.
`````

```{figure} ../figures/data-augmentation.svg
:name: fig-data-augmentation
:alt: La sagoma stilizzata di una foglia e cinque varianti, ottenute specchiandola, ruotandola, ritagliandola più da vicino, abbassandone la luminosità e spargendoci sopra dei grani di colore; da tutte e cinque parte una linea che converge su un'unica etichetta, «foglia sana».
:width: 95%

Da una sola immagine, cinque esempi "nuovi": i pixel cambiano, l'etichetta no.
```

Come mostra {numref}`fig-data-augmentation`, da una fotografia ne ricaviamo
molte: tutte diverse per la rete, tutte identiche per l'etichettatore.

## Ogni epoca un'immagine nuova, ma mai all'esame

In pratica le trasformazioni non si applicano una volta per tutte: si
estraggono **a caso, al volo**, ogni volta che un'immagine viene caricata. La
raccolta di foto sul disco non cresce di un byte, ma la rete non rivede mai due
volte la stessa identica immagine. Un giro completo su tutte le foto si chiama
**epoca**, e a ogni epoca le stesse foto tornano deformate in modo diverso.

E c'è una regola d'oro che non ammette eccezioni disinvolte. Le foto sono
divise in tre mucchi: quelle su cui la rete si allena (il **training set**),
quelle su cui controlliamo strada facendo come sta andando per aggiustare le
nostre scelte (il **validation set**) e quelle che restano chiuse in un
cassetto fino alla fine, per il giudizio conclusivo (il **test set**).
L'augmentation si applica **solo al primo mucchio**. Sugli altri due si fanno
soltanto le operazioni che danno sempre lo stesso risultato: ridimensionare,
ritagliare al centro e **normalizzare**, che vuol dire riportare i numeri dei
pixel su una scala fissa, la stessa per tutte le immagini, così che una foto
scattata in controluce e una scattata al sole partano dallo stesso metro.

`````{tab} Elementare
È la differenza tra i compiti a casa e il compito in classe. A casa
l'insegnante ti dà ogni giorno lo stesso tipo di esercizio ma con i numeri
cambiati: così impari il *metodo*, non il risultato a memoria. Il compito in
classe, invece, dev'essere uguale per tutti e ripetibile: se ogni studente
ricevesse una versione deformata a caso, il voto non misurerebbe più niente.
Il test set è il compito in classe del modello: deve dirci come andrà sulle
foto vere, così come sono, e deve dare lo stesso risultato ogni volta che lo
ripetiamo. Il validation set, in mezzo ai due, è la simulazione che si fa la
settimana prima: serve a noi per decidere che cosa cambiare, quindi anche lui
va lasciato uguale a sé stesso, altrimenti non si capisce se a migliorare sia
stato il modello o il caso.
`````

`````{tab} Superiore
A ogni epoca, per ogni esempio, si campiona una trasformazione $T$ da una
distribuzione fissata (un flip con probabilità $0{,}5$, un ritaglio con scala
casuale, una perturbazione di colore) e si addestra su $(T(\mathbf{x}), y)$: il numero
di varianti potenziali è di fatto illimitato, a costo zero di memoria. In
valutazione la pipeline dev'essere deterministica, per due ragioni: la metrica
deve riflettere la distribuzione di *deployment* (le foto arrivano intere, non
ritagliate a caso) e dev'essere riproducibile tra un'esecuzione e l'altra.
Esiste un'eccezione consapevole, la *test-time augmentation*: si media la
predizione su più copie trasformate della stessa immagine per guadagnare
qualche decimo di punto. È una scelta dichiarata di inferenza, non
un'augmentation "dimenticata accesa", e il confronto con altri modelli va
fatto a parità di questa scelta.
`````

## In pratica, con torchvision

Nella sezione sul transfer learning la catena di operazioni da fare a ogni
immagine prima di darla alla rete ce la dava la rete stessa
(`pesi.transforms()`). Per aggiungere l'augmentation la scriviamo a mano con
`torchvision.transforms`, tenendo rigorosamente separate le due catene, quella
dell'allenamento e quella dell'esame.

```{code-block} python
:class: pt-non-eseguibile

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

media = (0.485, 0.456, 0.406)   # statistiche di ImageNet: le stesse
dev   = (0.229, 0.224, 0.225)   # usate dalla rete pre-addestrata

# Pipeline di TRAINING: trasformazioni casuali, diverse a ogni epoca
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # ritaglio casuale
    transforms.RandomHorizontalFlip(p=0.5),               # specchio nel 50% dei casi
    transforms.ColorJitter(brightness=0.2, contrast=0.2,  # luce, contrasto,
                           saturation=0.2),               # saturazione
    transforms.ToTensor(),
    transforms.Normalize(media, dev),
])

# Pipeline di TEST: deterministica. Niente casualità, mai.
test_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(media, dev),
])

train_ds = datasets.ImageFolder("dati/train", transform=train_tf)
test_ds  = datasets.ImageFolder("dati/test",  transform=test_tf)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
test_dl  = DataLoader(test_ds,  batch_size=32)
```

Tutto il resto, la rete e il ciclo che la addestra, è identico a quello della
sezione precedente: l'augmentation vive tutta dentro il `Dataset`, cioè nel
punto in cui le immagini vengono lette. E la scelta delle trasformazioni segue
il compito: per un classificatore di cifre o di cartelli stradali, il
`RandomHorizontalFlip` va tolto.

## Perché funziona: un altro modo di mettere il freno

Impedire a una rete di imparare a memoria è un problema vecchio, e il libro ci
ha già messo mano due volte: nel capitolo sul machine learning penalizzando i
modelli che si affidano troppo a pochi numeri grossi (la regolarizzazione
$\ell_2$), in quello sul deep learning spegnendo a caso una parte della rete a
ogni passo, il dropout {cite}`srivastava2014dropout`. Tutti questi freni si
chiamano **regolarizzazioni**, e l'augmentation è uno di loro, con una
differenza: agisce sui dati invece che sulla rete.

`````{tab} Elementare
Uno studente che rifà cento volte lo stesso identico esercizio finisce per
ricordare il risultato, non il metodo. Se invece i numeri cambiano un po' ogni
volta, memorizzare non serve più a niente: l'unico modo di rispondere bene è
capire la regola. L'augmentation fa questo alla rete: le rende impossibile
"fotografare" il training set, perché il training set non è mai due volte lo
stesso. Ciò che sopravvive a specchi, ritagli e cambi di luce è proprio quello
che vogliamo: l'idea di gatto, non i pixel di *quel* gatto.
`````

`````{tab} Superiore
L'addestramento standard minimizza il rischio empirico
$\hat{R}(\theta) = \frac{1}{n}\sum_{i=1}^{n} \mathcal{L}\big(f_\theta(\mathbf{x}_i), y_i\big)$,
dove la distribuzione empirica concentra tutta la massa sugli $n$ punti
osservati. L'augmentation sostituisce ogni punto con una nuvola di varianti:

$$
\hat{R}_{\text{aug}}(\theta) \;=\; \frac{1}{n}\sum_{i=1}^{n}
\mathbb{E}_{T\sim\tau}\Big[\mathcal{L}\big(f_\theta(T(\mathbf{x}_i)),\, y_i\big)\Big],
$$

dove $\tau$ è la distribuzione sulle trasformazioni ammesse. In altre parole
**allarga il supporto della distribuzione empirica**: invece di esigere la
risposta giusta in $n$ punti isolati, la esige su interi intorni, e questo
spinge $f_\theta$ verso funzioni *invarianti* alle trasformazioni scelte (un
vincolo che riduce l'overfitting esattamente come farebbe un termine di
regolarizzazione). È l'idea del *vicinal risk minimization*, formulata da
Chapelle, Weston, Bottou e Vapnik {cite}`chapelle2000vicinal` (apprendere non
dai punti, ma dai loro dintorni), che tra poco vedremo portata alle estreme
conseguenze da mixup {cite}`zhang2018mixup`.
`````

## Mescolare, cancellare, imparare la ricetta

Le trasformazioni geometriche e di colore non esauriscono il repertorio. Le
varianti moderne sono più spregiudicate: producono immagini che nessuna
macchina fotografica scatterebbe, eppure aiutano.

`````{tab} Elementare
**Mixup** è come proiettare due diapositive sullo stesso schermo, una al 70%
e una al 30% di luminosità: un'immagine che è per sette decimi un gatto e per
tre decimi un cane. Anche la risposta richiesta si mescola nelle stesse
proporzioni: "70% gatto, 30% cane". Sembra assurdo, e serve a curare un vizio
preciso. Una rete addestrata solo su risposte secche («questo è un gatto,
punto») impara a essere sicurissima sempre, anche quando non ha capito niente,
perché il gioco premia soltanto chi si sbilancia. Chiedendole ogni tanto una
risposta a metà la si costringe a essere sicura solo dove ha davvero visto
qualcosa. **Cutout** è ancora più semplice: si copre un rettangolo a caso
della foto, come con un post-it. Se la rete riconosceva i gatti solo dalle
orecchie, con il post-it sulle orecchie dovrà imparare anche zampe e coda.
Infine, invece di scegliere a mano le trasformazioni, si può lasciare che sia
un algoritmo a cercare la combinazione migliore per il nostro archivio di
foto: ne prova tante per davvero, addestra ogni volta un modello, guarda quale
combinazione gli fa prendere il voto più alto a un esame di prova, e tiene
quella. È l'idea delle *policy apprese* (una *policy*, qui, è semplicemente la
lista delle trasformazioni scelte, con quanto forte applicarle). Costa
carissimo, ed è per questo che quasi nessuno la ricerca da sé: si scaricano le
combinazioni già trovate da chi aveva le macchine per cercarle.
`````

`````{tab} Superiore
**Mixup** {cite}`zhang2018mixup` costruisce esempi virtuali per
interpolazione convessa di coppie del training set:

$$
\tilde{\mathbf{x}} = \lambda \mathbf{x}_i + (1-\lambda)\, \mathbf{x}_j,
\qquad
\tilde{y} = \lambda y_i + (1-\lambda)\, y_j,
$$

dove $\mathbf{x}_i, \mathbf{x}_j$ sono due immagini, $y_i, y_j$ le rispettive
etichette in codifica one-hot e $\lambda \in [0,1]$ è estratto da una distribuzione
$\mathrm{Beta}(\alpha, \alpha)$, dove l'iperparametro $\alpha$ regola
l'intensità della mescolanza: con $\alpha$ piccolo (il paper usa valori tra
$0{,}1$ e $0{,}4$) $\lambda$ si concentra vicino a $0$ o a $1$, e le miscele
restano leggere. Il modello viene addestrato a produrre predizioni che
interpolano linearmente tra le classi, il che regolarizza il comportamento
*tra* gli esempi, dove il rischio empirico tace. **Cutout**
{cite}`devries2017improved` azzera un riquadro casuale dell'immagine (l'idea
quasi identica del *random erasing* lo sostituisce con valori casuali;
torchvision la implementa in `transforms.RandomErasing`, da applicare dopo
`ToTensor`), impedendo alla rete di dipendere da una singola regione
discriminante: un dropout applicato allo spazio dei pixel. Infine
**AutoAugment** {cite}`cubuk2019autoaugment` tratta la scelta delle
trasformazioni come un problema di ricerca: un controllore addestrato per
rinforzo compone la *policy* di augmentation che massimizza l'accuratezza in
validazione; il successore RandAugment ottiene risultati simili riducendo la
ricerca a due soli iperparametri (numero e intensità delle trasformazioni).
`````

## Quando aiuta, e quando no

L'augmentation rende di più dove i dati scarseggiano: con poche centinaia di
immagini per categoria può far salire di parecchi punti percentuali la quota di
foto indovinate, ed è la prima cosa da provare quando il modello impara a
memoria invece di capire (insieme, non in alternativa, al transfer learning
visto nella sezione precedente). Ma non è
una moltiplicazione miracolosa: le varianti di una foto portano *meno*
informazione di altrettante foto nuove, perché raccontano sempre la stessa
scena.

Soprattutto, l'augmentation **non cura lo shift di dominio**. Se il modello è
addestrato su foto diurne e in produzione arrivano riprese notturne, se il
nuovo ospedale usa uno scanner diverso da quello del training set, nessuno
specchio e nessun ritaglio colmerà quella distanza: è il problema dei *dati
che cambiano* che abbiamo discusso nel capitolo sul machine learning, e la
risposta è raccogliere dati rappresentativi del dominio reale, non deformare
quelli vecchi. Anzi, un'augmentation scelta male può peggiorare le cose,
perché iniettare l'invarianza sbagliata significa insegnare alla rete una cosa
falsa sul mondo: una radiografia del torace specchiata mette il cuore a
destra, un'anatomia rarissima che il modello imparerebbe a considerare
normale. Le trasformazioni codificano le nostre ipotesi sul problema, e delle
ipotesi, come sempre, si risponde.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- L'augmentation fabbrica varianti di ogni foto (specchiata, ritagliata,
  ruotata, più chiara) e le conta come esempi nuovi. La regola è una sola:
  **la risposta giusta non deve cambiare**. Lo specchio va bene per i gatti,
  rovina i cartelli stradali e trasforma un 6 in un 9.
- Si applica **solo alle foto su cui la rete si allena**, e cambia a ogni
  passaggio. Sulle foto d'esame si fanno solo operazioni che danno sempre lo
  stesso risultato, altrimenti il voto non misura più niente.
- Serve a impedire di **imparare a memoria**: se il compito non è mai due volte
  identico, memorizzarlo non conviene più, e l'unica strada che resta è capire.
- Ci sono modi più spregiudicati: **mescolare due foto** (e mescolare nelle
  stesse proporzioni la risposta), **coprire un rettangolo a caso** come con un
  post-it, oppure lasciare che sia un algoritmo a cercare la combinazione
  migliore.
- Non è una moltiplicazione miracolosa, e soprattutto **non serve a niente se
  le foto vere sono di un altro tipo**: se ci si è allenati di giorno e in
  produzione arrivano riprese notturne, servono foto nuove, non deformazioni
  di quelle vecchie.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- L'augmentation genera varianti di ogni immagine con trasformazioni che
  **preservano l'etichetta**: quali siano dipende dal compito (lo specchio va
  bene per i gatti, non per cartelli o cifre).
- Si applica **solo al training set**, al volo, a ogni epoca; su validation e
  test solo operazioni deterministiche (resize, crop centrale, normalize).
- È una **regolarizzazione**: allarga il supporto della distribuzione
  empirica e contrasta l'overfitting, come il dropout e la penalità $\ell_2$.
- Varianti moderne: **mixup** (interpolazione di immagini ed etichette),
  **cutout/random erasing** (occlusioni casuali), **policy apprese**
  (AutoAugment, RandAugment).
- Non risolve lo **shift di dominio**: se i dati di produzione sono diversi
  da quelli di training, servono dati nuovi, non trasformazioni.
```

`````
