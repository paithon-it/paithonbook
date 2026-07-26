# Deep Learning: perche la profondita conta

Nel 1959 due neuroscienziati, David Hubel e Torsten Wiesel
{cite}`hubel1959receptive`, infilarono un sottile elettrodo nella corteccia
visiva di un gatto e proiettarono forme luminose davanti ai suoi occhi. Scoprirono che certi neuroni si accendevano
solo quando sullo schermo compariva una linea con una precisa inclinazione:
rivelatori di bordi in carne e ossa. Altri neuroni, più a valle, combinavano
quelle risposte in configurazioni più complesse. Quel lavoro valse loro il
premio Nobel nel 1981 e, senza che nessuno potesse immaginarlo, descriveva già
il funzionamento delle reti neurali profonde. Il deep learning, in fondo,
riscopre la stessa idea: costruire la percezione a strati, dal semplice al
complesso.

## Feature apprese, non ingegnerizzate a mano

La differenza tra machine learning classico e deep learning non sta in una
matematica esoterica: sta in *chi* decide quali caratteristiche dei dati
contano.

`````{tab} Elementare

Nel machine learning classico un esperto umano deve prima trasformare i dati
grezzi in "ingredienti" utili. Per riconoscere un gatto in una foto, qualcuno
scrive a mano il codice che conta i bordi, misura quanta parte dell'immagine
è arancione o grigia, individua gli angoli: sono le *feature ingegnerizzate*,
le caratteristiche scelte da un umano. Solo dopo il modello impara a
distinguere i gatti dai cani.

Il deep learning fa un patto diverso: gli dai i pixel grezzi e lascia che sia
la rete, addestrandosi, a inventarsi da sola gli ingredienti giusti. È come la
differenza tra un cuoco a cui porti le verdure già tagliate e uno a cui porti
la verdura intera: il secondo impara a tagliarla esattamente nel modo che serve
al piatto.

`````

`````{tab} Superiore

In una pipeline classica di visione artificiale un estrattore fisso e
progettato a mano — SIFT, HOG, filtri di Gabor — mappa l'immagine in un vettore
di feature $\phi(X)$, su cui un classificatore $g$ viene addestrato. L'estrattore
$\phi$ è congelato: non impara nulla dai dati.

Una rete profonda è invece una composizione di trasformazioni parametriche

$$
f_\theta = f_L \circ f_{L-1} \circ \dots \circ f_1 ,
$$

dove ogni $f_\ell(Z) = \sigma(W_\ell Z + b_\ell)$ ha i suoi pesi $W_\ell$, e
tutti i parametri $\theta$ vengono ottimizzati insieme minimizzando la loss
$\mathcal{L}$ per retropropagazione (*end-to-end*). L'estrazione delle feature
non è più a monte e fissa: è parte del modello e viene appresa. È ciò che si
chiama *representation learning*.

`````

## Rappresentazioni gerarchiche: dai bordi agli oggetti

Cosa impara davvero uno strato? Quando nel 2014 Zeiler e Fergus riuscirono a
"visualizzare" ciò a cui rispondono i neuroni di una rete convoluzionale, il
risultato somigliava in modo sorprendente alla corteccia di Hubel e Wiesel: i
primi strati reagiscono a bordi e linee orientate; gli strati intermedi a
texture e a parti (un occhio, una ruota, la trama di un tessuto); gli ultimi
strati a interi oggetti. La profondità costruisce astrazione, un livello alla
volta ({numref}`fig-gerarchia-feature`).

```{figure} ../figures/gerarchia-feature.svg
:name: fig-gerarchia-feature
:alt: Quattro riquadri in fila mostrano come una rete profonda costruisce la percezione a strati, dai pixel dell'immagine ai bordi, alle parti, fino all'oggetto intero riconosciuto.
:width: 92%

Come una rete profonda "vede" un gatto. Da sinistra a destra: i pixel grezzi,
i bordi e le linee dei primi strati, le texture e le parti degli strati
intermedi, l'oggetto intero riconosciuto dagli ultimi strati.
```

`````{tab} Elementare

Immagina di costruire con i mattoncini. Prima metti insieme i pezzi più piccoli
— tratti dritti, curve, angoli. Poi combini quei tratti in parti riconoscibili:
due cerchi diventano occhi, due triangoli diventano orecchie. Alla fine le parti
si assemblano in un gatto intero.

La rete fa esattamente questo, ma al contrario di come lo racconteremmo noi:
nessuno le dice "questo è un occhio". Impara da sola che, per riconoscere i
gatti nelle foto, conviene prima trovare i bordi, poi comporli in parti, poi
comporre le parti in animali.

`````

`````{tab} Superiore

La chiave è la **composizionalità**. Ogni mappa di feature dello strato $\ell$
è una funzione non lineare delle mappe dello strato precedente, quindi la
complessità delle forme rilevabili cresce con la profondità. In una rete
convoluzionale cresce anche il **campo recettivo** (*receptive field*): un
neurone dello strato $\ell$ "vede" una porzione di immagine tanto più ampia
quanto più $\ell$ è profondo, perché aggrega l'output di neuroni che a loro
volta aggregano porzioni più piccole.

Formalmente, la rappresentazione allo strato $\ell$ è
$Z^{(\ell)} = f_\ell(Z^{(\ell-1)})$, con $Z^{(0)} = X$ l'immagine di ingresso.
Le $Z^{(\ell)}$ superficiali codificano feature locali e generiche (bordi,
condivisi tra compiti diversi); le $Z^{(\ell)}$ profonde codificano feature
astratte e specifiche del compito (categorie di oggetti). È questa gerarchia a
rendere così efficace il *transfer learning*: i primi strati, appresi su un
dataset enorme, si riusano quasi invariati su problemi nuovi.

`````

## Perché proprio adesso

C'è un dettaglio che spiazza chi arriva al deep learning da profano: la
retropropagazione, l'algoritmo che addestra queste reti, è del 1986 (Rumelhart,
Hinton e Williams). Se l'algoritmo esisteva da decenni, perché il deep learning
è esploso solo dopo il 2012? Perché servivano tre ingredienti tutti insieme:
**dati**, **potenza di calcolo** e **algoritmi maturi**. Il momento-simbolo è
l'autunno del 2012, la competizione ImageNet.

`````{tab} Elementare

Pensa a un fuoco. Ti serve la legna, l'ossigeno e una scintilla: se manca anche
uno solo dei tre, non parte. Il deep learning è rimasto "spento" per anni non
perché mancasse l'idea, ma perché mancava la combustione completa.

La legna sono i **dati**: milioni di immagini etichettate, che prima di
Internet semplicemente non esistevano. L'ossigeno è il **calcolo**: le schede
grafiche (GPU), nate per i videogiochi, si sono rivelate perfette per i conti
delle reti. La scintilla sono gli **algoritmi** giusti al momento giusto. Nel
2012, per la prima volta, i tre elementi c'erano tutti.

`````

`````{tab} Superiore

Nel 2009 Fei-Fei Li e collaboratori pubblicano **ImageNet**, un dataset di
milioni di immagini etichettate da persone; la sua competizione annuale, la
**ILSVRC**, ne usa un sottoinsieme di circa 1,2 milioni di immagini di
addestramento distribuite su 1000 categorie. Nel 2012 **AlexNet** (Krizhevsky,
Sutskever, Hinton) vince proprio la ILSVRC portando l'errore *top-5* dal 26,2%
del miglior metodo classico al 15,3% — un salto senza precedenti.

I tre ingredienti, in numeri:

- **Dati**: un dataset abbastanza grande da addestrare una rete con circa 60
  milioni di parametri senza overfittare in modo catastrofico.
- **Calcolo**: l'addestramento gira su due GPU NVIDIA GTX 580, sfruttando la
  parallelizzazione dei prodotti tra matrici (`cuda`).
- **Algoritmi**: attivazione ReLU $\sigma(x)=\max(0,x)$ per attenuare il
  *vanishing gradient*, *dropout* come regolarizzazione, *data augmentation*.

Nessuna di queste idee era nuova in senso stretto; nuova era la loro
combinazione, alla scala giusta.

`````

## Profondo, non solo largo

Resta un'obiezione teorica seria. Il **teorema di approssimazione universale**
({cite}`cybenko1989approximation`; {cite}`hornik1991approximation`) dimostra
che basta *un solo* strato nascosto,
purché abbastanza ampio, per approssimare qualunque funzione continua — cioè
per imitare, con la precisione voluta, qualunque regola che leghi ingressi e
uscite senza salti bruschi. Se una rete "piatta" e larga è già universale,
perché impilare tanti strati?

`````{tab} Elementare

Il teorema dice che *in teoria* un solo strato basta. Ma "in teoria" nasconde
una fregatura: quel singolo strato potrebbe aver bisogno di un numero enorme,
impraticabile, di neuroni.

La profondità permette di **riusare** il lavoro. Se ho già imparato a
riconoscere un occhio, lo uso sia per il gatto sia per il cane senza reimpararlo
da capo. Una rete larga e piatta deve invece reinventare ogni forma da zero,
ogni volta. Andare in profondità è come scrivere funzioni che chiamano altre
funzioni, invece di riscrivere sempre tutto il codice a mano.

`````

`````{tab} Superiore

Il teorema garantisce che per ogni funzione continua $f:[0,1]^n\to\mathbb{R}$ e
ogni $\varepsilon>0$ esiste una rete a un solo strato nascosto

$$
g(X) = \sum_{i=1}^{N} v_i\,\sigma\!\big(W_i^\top X + b_i\big)
\qquad\text{tale che}\qquad
\sup_{X}\,\lvert f(X) - g(X)\rvert < \varepsilon .
$$

Il punto è che $N$, il numero di neuroni, può crescere in modo **esponenziale**.
Esistono famiglie di funzioni rappresentabili da reti profonde con un numero di
neuroni *polinomiale* nella profondità, ma che richiedono larghezza
*esponenziale* se ci si limita a un solo strato {cite}`telgarsky2016benefits`.
Montúfar e
colleghi (2014) mostrano che il numero di regioni lineari che una rete ReLU può
generare cresce esponenzialmente con la profondità e solo polinomialmente con
la larghezza. Tradotto: la profondità compra efficienza espressiva. È più
economico comporre trasformazioni che allargarne una sola.

`````

Questa gerarchia si legge anche nel codice. Una piccola rete convoluzionale in
PyTorch è, letteralmente, una pila di strati che vanno dal semplice al
complesso:

```python
from torch import nn

# input: un batch di immagini RGB, shape (N, 3, 128, 128) — canali prima
model = nn.Sequential(
    nn.Conv2d(3, 32, 3), nn.ReLU(),    # primi strati: bordi e linee
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3), nn.ReLU(),   # strati intermedi: texture e parti
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3), nn.ReLU(),  # parti piu grandi, oggetti
    nn.AdaptiveAvgPool2d(1),           # media globale di ogni feature map
    nn.Flatten(),
    nn.Linear(128, 10),                # la classe finale (un logit per classe)
)
```

Ogni `nn.Conv2d` più in basso nella pila costruisce feature più astratte a
partire da quelle dello strato precedente: la stessa scala dai bordi agli
oggetti della {numref}`fig-gerarchia-feature`, resa in poche righe.

```{admonition} Da ricordare
:class: important
- Il deep learning **apprende** le feature dai dati grezzi, invece di
  richiederle ingegnerizzate a mano come il ML classico.
- Le rappresentazioni sono **gerarchiche**: bordi → texture e parti → oggetti,
  un livello di astrazione per strato.
- È esploso dopo il 2012 (ImageNet, AlexNet) grazie alla triade **dati + GPU +
  algoritmi**, non a una singola idea nuova.
- Una rete larga e piatta è universale in teoria, ma la **profondità** ottiene
  la stessa espressività con molti meno neuroni: comporre conviene.
```
