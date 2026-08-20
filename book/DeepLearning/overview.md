# Deep Learning: perché la profondità conta

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

## Le feature si imparano, non si scrivono a mano

La differenza tra machine learning classico e deep learning non sta in una
matematica esoterica: sta in *chi* decide quali caratteristiche dei dati
contano. Quelle caratteristiche, in gergo, si chiamano **feature**: sono i
numeri che descrivono un dato e su cui il modello, cioè il programma che impara
dagli esempi, fa i suoi conti.

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
progettato a mano (SIFT, HOG, filtri di Gabor) mappa l'immagine in un vettore
di feature $\phi(\mathbf{X})$, su cui un classificatore $g$ viene addestrato.
L'estrattore $\phi$ è congelato: non impara nulla dai dati.

Una rete profonda è invece una composizione di trasformazioni parametriche

$$
f_\theta = f_L \circ f_{L-1} \circ \dots \circ f_1 ,
$$

dove tipicamente $f_\ell(\mathbf{Z}) = \sigma(\mathbf{W}_\ell \mathbf{Z} + \mathbf{b}_\ell)$ ha i suoi pesi
$\mathbf{W}_\ell$ (ma incontreremo anche strati di altra forma, come il pooling), e
tutti i parametri $\theta$ vengono ottimizzati insieme minimizzando la loss
$\mathcal{L}$ per retropropagazione (*end-to-end*). L'estrazione delle feature
non è più a monte e fissa: è parte del modello e viene appresa. È ciò che si
chiama *representation learning*.

`````

## Rappresentazioni gerarchiche: dai bordi agli oggetti

Una rete profonda è fatta di **strati**: gruppi di neuroni messi in fila, dove
il primo riceve i numeri dell'immagine e ognuno dei successivi riceve quello che
ha prodotto quello prima di lui. (Un neurone, qui, non è una cellula: è un
pezzetto di conto, che prende dei numeri, li somma dopo averli pesati e ne
restituisce uno.) Cosa impara davvero uno strato?

Nel 2014 Zeiler e Fergus {cite}`zeiler2014visualizing` trovarono il modo di
"visualizzarlo", cioè di risalire, per ogni neurone di una **rete
convoluzionale** (il tipo di rete fatto apposta per le immagini, di cui parla
la sezione che segue), alla forma che lo fa accendere. Il risultato somigliava
in modo sorprendente alla corteccia di Hubel e Wiesel: i primi strati reagiscono
a bordi e linee orientate; gli strati intermedi a parti riconoscibili (un
occhio, una ruota) e a *texture*, cioè trame che si ripetono, come la stoffa di
un tessuto o il pelo di un animale; gli ultimi strati a interi oggetti. La
profondità costruisce astrazione, un livello alla volta
({numref}`fig-gerarchia-feature`).

```{figure} ../figures/gerarchia-feature.svg
:name: fig-gerarchia-feature
:alt: Quattro riquadri in fila mostrano come una rete profonda costruisce la percezione a strati, dai pixel dell'immagine ai bordi, alle parti, fino all'oggetto intero riconosciuto.
:width: 92%

Come una rete profonda "vede" un gatto. Da sinistra a destra: i pixel grezzi,
i bordi e le linee dei primi strati, le texture e le parti degli strati
intermedi, l'oggetto intero riconosciuto dagli ultimi strati.
```

`````{tab} Elementare

Davanti a una scatola di mattoncini, cominci dai pezzi più piccoli: tratti
dritti, curve, angoli. Poi combini quei tratti in parti riconoscibili: due
cerchi diventano occhi, due triangoli diventano orecchie. Alla fine le parti si
assemblano in un gatto intero.

La rete fa esattamente questo, ma senza che nessuno gliel'abbia insegnato:
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
$\mathbf{Z}^{[\ell]} = f_\ell(\mathbf{Z}^{[\ell-1]})$, con $\mathbf{Z}^{[0]} = \mathbf{X}$ l'immagine di ingresso.
Le $\mathbf{Z}^{[\ell]}$ superficiali codificano feature locali e generiche (bordi,
condivisi tra compiti diversi); le $\mathbf{Z}^{[\ell]}$ profonde codificano feature
astratte e specifiche del compito (categorie di oggetti). È questa gerarchia a
rendere così efficace il *transfer learning*: i primi strati, appresi su un
dataset enorme, si riusano quasi invariati su problemi nuovi.

`````

## Perché proprio adesso

C'è un dettaglio che spiazza chi arriva al deep learning da profano: la
retropropagazione, l'algoritmo che addestra queste reti, fu resa celebre nel
1986 da Rumelhart, Hinton e Williams {cite}`rumelhart1986learning`, e l'idea è
ancora più vecchia[^backprop-storia].
Se l'algoritmo esisteva da decenni, perché il deep learning
è esploso solo dopo il 2012? Perché servivano tre ingredienti tutti insieme:
**dati**, **potenza di calcolo** e **algoritmi maturi**. Il momento-simbolo è
l'autunno del 2012, la competizione **ImageNet**: una gara annuale in cui i
gruppi di ricerca di mezzo mondo provavano i propri programmi sulle stesse
fotografie, più di un milione, con lo stesso compito (dire che cosa
c'è dentro, scegliendo fra mille categorie) e una classifica finale. Quell'anno
la vince, con un margine mai visto prima, una rete profonda: **AlexNet**, dal
nome del primo dei suoi autori, Alex Krizhevsky.

`````{tab} Elementare

Per accendere un fuoco servono la legna, l'aria e una scintilla: se manca anche
uno solo dei tre, non parte. Il deep learning è rimasto "spento" per anni non
perché mancasse l'idea, ma perché mancava la combustione completa.

La legna sono i **dati**: milioni di immagini **etichettate**, cioè fotografie
accanto a cui qualcuno ha scritto a mano che cosa c'è dentro. Prima di Internet
non esistevano, perché raccoglierle e descriverle una per una era un lavoro
fuori portata. Attenzione a una cosa, però, perché l'Introduzione ci ha
insistito: quella legna nessuno è andato a tagliarla apposta. Erano le foto che
la gente caricava per farle vedere agli amici, e qualcuno si è accorto che
potevano servire ad altro. L'aria è il **calcolo**: le schede grafiche (GPU), nate per i
videogiochi, si sono rivelate perfette per i conti delle reti.

La scintilla sono gli **algoritmi**: tre accorgimenti precisi, ciascuno contro
un guaio preciso, che fanno la differenza fra la rete del 2012 e quelle che
non imparavano.

Il primo riguarda la funzione che ogni neurone applica al numero che gli è
uscito dai conti. Prima si usava una curva che schiaccia qualunque numero in un
intervallo stretto: grande o piccolo che fosse, quello che ne usciva era sempre
più o meno uguale, e la rete non aveva più modo di accorgersi della differenza.
Al suo posto si adotta una regola molto più semplice: i numeri positivi passano
come sono, i negativi diventano zero. Se entra 5 esce 5, se entra $-3$ esce 0.
Si chiama **ReLU**.

Il secondo combatte il guaio peggiore di una rete: impararsi a memoria le
fotografie dell'addestramento. Sembrerebbe un pregio, e invece è il modo più
sicuro di fallire, perché una rete che le ricorda a mente va benissimo su quelle
e male su tutte le altre, che sono poi le uniche che conteranno. Il rimedio è
spegnere a caso, a ogni passata sulle fotografie, una parte dei neuroni: se un
neurone potrebbe non esserci al giro dopo, gli altri non possono appoggiarsi
solo a lui, e la rete è costretta a distribuire quello che sa invece di
depositarlo tutto in un punto (**dropout**).

Il terzo è moltiplicare gli esempi ritagliando e specchiando le fotografie che
già si hanno, per dare alla rete più materiale senza doverne etichettare altre
(**data augmentation**). Nel 2012, per la prima volta, i tre elementi c'erano
tutti.

`````

`````{tab} Superiore

Nel 2009 Fei-Fei Li e collaboratori pubblicano **ImageNet**, un dataset di
milioni di immagini etichettate da persone; la sua competizione annuale, la
**ILSVRC**, ne usa un sottoinsieme di circa 1,2 milioni di immagini di
addestramento distribuite su 1000 categorie. Nel 2012 **AlexNet** (Krizhevsky,
Sutskever, Hinton) vince proprio la ILSVRC portando l'errore *top-5* dal 26,2%
del miglior metodo classico al 15,3%. Il confronto va letto per quello che è:
il 15,3% è il punteggio della sottomissione, che media le predizioni di sette
reti; la singola rete descritta nell'articolo, quella di cui parliamo qui, si
ferma al 18,2%, e anche così il salto è senza precedenti.

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

Fin qui abbiamo dato per buono che impilare strati serva a qualcosa. C'è però un
risultato matematico che sembra dire il contrario, e conviene affrontarlo
subito.

Si chiama **teorema di approssimazione universale**, e dice questo: basta *un
solo* strato in mezzo, fra l'ingresso e l'uscita. Purché lo si faccia
abbastanza largo, cioè con abbastanza neuroni, quella rete a un piano solo sa
imitare con la precisione che si vuole qualunque regola che leghi ingressi e
uscite senza salti bruschi.[^universalita]

Una condizione c'è, ed è essenziale. La funzione che ogni neurone applica al
proprio risultato, l’**attivazione** (la ReLU, per esempio, che azzera i numeri
negativi e lascia passare i positivi), non deve essere un *polinomio*, cioè una
somma di potenze come $3x^2-x+5$. Il motivo è che sommare e moltiplicare
polinomi dà sempre e solo altri polinomi: se l'attivazione fosse uno di quelli,
tutto ciò che la rete sa produrre sarebbe un polinomio, cioè una curva liscia e
regolare, e le curve lisce non bastano a descrivere qualunque legame fra
ingresso e uscita. La ReLU non è un polinomio proprio per questo: ha uno
spigolo, nel punto in cui smette di azzerare e comincia a lasciar passare, e
nessuna somma di potenze fa un angolo.

Se una rete "piatta" e larga sa già imitare tutto, perché impilare tanti strati?

`````{tab} Elementare

Il teorema dice che *in teoria* un solo strato basta. Ma "in teoria" nasconde
una fregatura: quel singolo strato potrebbe aver bisogno di un numero enorme,
impraticabile, di neuroni.

Quello che manca alla rete piatta è la possibilità di **costruire sopra**. In
una rete a un solo strato ogni neurone guarda i pixel grezzi e nient'altro:
nessuno può partire da una forma che un altro ha già trovato per comporla con
una seconda. Un occhio va descritto ogni volta a partire dai pixel, e le
combinazioni di pixel che fanno un occhio sono innumerevoli: cambia la luce,
cambia l'inclinazione, cambia la taglia, e ogni variante va prevista da capo.

Con più strati il secondo lavora su ciò che ha trovato il primo, il terzo su
ciò che ha trovato il secondo. È la differenza tra una ricetta scritta tutta
come "prendi la farina, prendi l'uovo, prendi il burro" e una che a un certo
punto dice "prepara la besciamella", dando per fatto un pezzo di strada già
percorso: la seconda arriva allo stesso piatto con molte parole in meno.

`````

`````{tab} Superiore

Il teorema garantisce che, se l'attivazione $\sigma$ è continua e non
polinomiale (l'ipotesi è essenziale: la soddisfano la sigmoide e la ReLU
{cite}`leshno1993multilayer`, non un $\sigma$ polinomiale, che produrrebbe
solo polinomi), per ogni funzione continua $f:[0,1]^n\to\mathbb{R}$ e
ogni $\varepsilon>0$ esiste una rete a un solo strato nascosto

$$
g(\mathbf{x}) = \sum_{i=1}^{N} v_i\,\sigma\!\big(\mathbf{w}_i^\top \mathbf{x} + b_i\big)
\qquad\text{tale che}\qquad
\sup_{\mathbf{x} \in [0,1]^n}\,\lvert f(\mathbf{x}) - g(\mathbf{x})\rvert < \varepsilon ,
$$

dove $\mathbf{x} \in [0,1]^n$ è il singolo esempio in ingresso (un vettore,
non una matrice di dati), $\mathbf{w}_i \in \mathbb{R}^n$ è il vettore dei
pesi del neurone $i$-esimo, $b_i$ il suo bias e $v_i$ il peso con cui
contribuisce all'uscita.

Due avvertenze sul quantificatore, che è dove il teorema promette meno di quanto
sembri. La prima: è un risultato di **esistenza**, cioè di densità. Dice che la
rete c'è, non che la discesa del gradiente la trovi, né quanti esempi servano
per impararla. La seconda: il teorema da solo non dà **nessun limite** su $N$,
il numero di neuroni. Quel limite dipende dalla classe di funzioni che si vuole
approssimare, e conviene non generalizzare la frase che si sente più spesso.
Per le classi definite dalla sola regolarità (derivate limitate fino a un certo
ordine) il numero di neuroni cresce esponenzialmente nella **dimensione
dell'ingresso**, ed è la maledizione della dimensionalità, che colpisce
qualunque schema di approssimazione lineare e non le reti in particolare. Ma
non è una legge universale: Barron {cite}`barron1993universal` individua una
classe più ristretta, definita da una condizione sulla trasformata di Fourier,
per cui l'errore quadratico scende come $O(1/N)$ **senza** dipendere dalla
dimensione. La crescita esponenziale è una proprietà della classe di funzioni,
non delle reti a uno strato in quanto tali.

Sulla profondità, invece, le separazioni sono nette e dimostrate. Esistono
famiglie di funzioni rappresentabili da reti profonde con un numero di neuroni
*polinomiale* nella profondità, ma che richiedono larghezza *esponenziale* se
ci si limita a un solo strato {cite}`telgarsky2016benefits`; Eldan e Shamir
{cite}`eldan2016power` esibiscono una funzione che una rete con **due** strati
nascosti rappresenta con un numero di neuroni polinomiale nella dimensione
dell'ingresso, e che una rete con **uno solo** non riesce ad approssimare oltre
una certa soglia a meno di renderla esponenzialmente larga. Montúfar e colleghi
{cite}`montufar2014number` mostrano che il numero di regioni lineari che una
rete ReLU può generare cresce esponenzialmente con la profondità e solo
polinomialmente con la larghezza. Tradotto: la profondità compra efficienza
espressiva. È più economico comporre trasformazioni che allargarne una sola.

`````

Questa gerarchia si legge anche nel codice. Una piccola rete convoluzionale scritta in {doc}`PyTorch </PyTorch/overview>`
è letteralmente una pila di strati che vanno dal semplice al complesso. Di che cosa faccia ciascuno
di quegli strati parla la sezione seguente: qui conta solo vedere la pila.

```python
from torch import nn

# in ingresso: un gruppo di immagini a colori, alte e larghe 128 pixel
model = nn.Sequential(
    nn.Conv2d(3, 32, 3), nn.ReLU(),    # primi strati: bordi e linee
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3), nn.ReLU(),   # strati intermedi: texture e parti
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3), nn.ReLU(),  # parti più grandi, oggetti
    nn.AdaptiveAvgPool2d(1),           # media di ogni foglio di risultati
    nn.Flatten(),
    nn.Linear(128, 10),                # la classe finale (un logit per classe)
)
```

I tre numeri dentro `nn.Conv2d(3, 32, 3)` si leggono così: quanti "fogli" di
numeri arrivano (3, cioè rosso, verde e blu), quanti filtri diversi applicare
(32) e quanto è larga la finestra che ciascun filtro guarda per volta (3 per 3
pixel). Ognuno di quei filtri produce un foglio di risultati, e quel foglio ha
un nome che accompagnerà tutto il capitolo: si chiama **feature map**, ed è la
mappa che segna punto per punto dove nell'immagine il filtro ha trovato ciò che
cerca (la sezione seguente mostra come nasce). La riga
`nn.AdaptiveAvgPool2d(1)` riduce ciascuna di quelle mappe a un numero solo, la
sua media. L'ultima riga produce dieci numeri, uno per classe: si chiamano
**logit**, sono punteggi grezzi, e vince il più alto (per trasformarli in
probabilità serve un ultimo passaggio, la *softmax*, che li schiaccia tutti fra
zero e uno facendo in modo che sommati diano uno). Ogni `nn.Conv2d` più avanti
nella pila costruisce feature più astratte a partire da quelle dello strato
precedente: la stessa scala dai bordi agli oggetti della
{numref}`fig-gerarchia-feature`, resa in poche righe.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Nel machine learning classico è una persona a decidere quali caratteristiche
  dei dati contano, e a scriverle nel codice. Una rete profonda se le
  **inventa da sola**, partendo dai dati grezzi.
- Le costruisce **a strati**, dal semplice al complesso: prima bordi e linee,
  poi pezzi riconoscibili (un occhio, una ruota), infine l'oggetto intero.
- Non è esplosa prima del 2012 perché servivano tre cose insieme, come la
  legna, l'aria e la scintilla: milioni di **fotografie già etichettate**,
  **schede grafiche** abbastanza veloci e tre accorgimenti precisi (la **ReLU**
  al posto delle funzioni che spegnevano il segnale, il **dropout** contro
  l'imparare a memoria, il moltiplicare le foto con ritagli e specchiature).
  Nel 2012 c'erano tutte e tre, e alla gara di ImageNet vinse AlexNet.
  (La ReLU è la regola più semplice possibile: i numeri positivi passano come
  sono, i negativi diventano zero.)
- Uno strato solo, se lo si facesse enorme, in teoria basterebbe. Ma la
  profondità arriva allo stesso risultato con molti meno neuroni, perché ogni
  strato può **costruire sopra** quello che ha trovato il precedente, invece di
  descrivere ogni forma a partire dai pixel.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Il deep learning **apprende** le feature dai dati grezzi, invece di
  richiederle ingegnerizzate a mano come il ML classico.
- Le rappresentazioni sono **gerarchiche**: bordi → texture e parti → oggetti,
  un livello di astrazione per strato.
- È esploso dopo il 2012 (ImageNet, AlexNet) grazie alla triade **dati + GPU +
  algoritmi**, non a una singola idea nuova.
- Una rete larga e piatta è universale in teoria (con un'attivazione non
  polinomiale), ma la **profondità** ottiene la stessa espressività con molti
  meno neuroni: comporre conviene. L'universalità però è un'esistenza, non
  un'apprendibilità.
```
`````

[^backprop-storia]: Il conto che sta sotto la retropropagazione (partire
    dall'errore in fondo alla rete e risalire all'indietro, strato per strato,
    per sapere quanto ciascun **peso**, cioè ciascuno dei numeri regolabili
    della rete, ha contribuito) era già noto ai matematici dal 1970, quando
    Seppo Linnainmaa lo descrisse in tutta generalità: è quella che oggi si
    chiama *differenziazione automatica in modalità inversa*. Paul Werbos la
    applicò alle reti neurali in un lavoro presentato nel 1981 e pubblicato
    l'anno dopo, e a metà anni Ottanta fu riscoperta per conto proprio da LeCun
    e da Parker. Il merito del 1986 è averla mostrata al mondo, con esperimenti
    convincenti, nel posto giusto: un articolo su *Nature*.

[^universalita]: Il teorema arriva in tre tempi. Cybenko lo dimostra nel 1989
    {cite}`cybenko1989approximation` per le attivazioni a forma di S, cioè per
    quelle curve che schiacciano i numeri in un intervallo stretto e che si
    chiamano *sigmoidi*; Hornik lo estende nel 1991
    {cite}`hornik1991approximation` a
    tutte quelle che restano confinate fra due valori; la forma generale,
    quella che copre anche la ReLU, arriva nel 1993 con Leshno e colleghi
    {cite}`leshno1993multilayer`.
