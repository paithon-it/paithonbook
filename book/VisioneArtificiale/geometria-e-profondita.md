# Dove sono le cose: geometria, corrispondenze e profondità

Attorno al 1413, sulla piazza del Duomo di Firenze, Filippo Brunelleschi fece
una cosa che nessuno aveva mai fatto. Aveva dipinto su una tavoletta di mezzo
braccio di lato (una trentina di centimetri) il Battistero di San Giovanni,
intarsi di marmo compresi, e nella tavoletta aveva praticato un foro. Chi
voleva vedere l'esperimento doveva mettersi dentro il portale del Duomo,
tenere la tavoletta col dipinto rivolto **all'indietro**, guardare attraverso
il foro e reggere con l'altra mano uno specchio. Nello specchio compariva il
Battistero dipinto; togliendo lo specchio compariva il Battistero vero. Erano
identici. La parte alta della tavoletta, dove ci sarebbe stato il cielo, era
coperta d'argento brunito, così che nel dipinto si riflettessero le nuvole
vere che passavano.

Quella tavoletta è il primo dispositivo della storia costruito per dimostrare
una legge geometrica: **un punto di vista, un foro, e il mondo tridimensionale
si schiaccia su una superficie piatta in un modo prevedibile e calcolabile**.
Vent'anni dopo Leon Battista Alberti ne scrisse le regole nel *De pictura*, e
la prospettiva lineare diventò una tecnica insegnabile.

Il foro di Brunelleschi è, letteralmente, il modello di fotocamera che usiamo
ancora oggi. E porta con sé il problema che occupa tutta questa sezione. Il
mondo ha tre dimensioni, l'immagine ne ha due: **proiettare significa buttare
via un numero per ogni punto**, la distanza. Il resto del capitolo si è
occupato di che *cosa* c'è in un'immagine; qui ci occupiamo di **dove**, e la
domanda è più difficile di quanto sembri, perché l'informazione che serve non
è nascosta nell'immagine: è stata proprio cancellata.

## La proiezione perde una dimensione

Cominciamo da come un punto del mondo diventa un pixel.

`````{tab} Elementare

Immagina una scatola chiusa con un foro piccolissimo su una faccia e un foglio
sulla faccia opposta. La luce che parte da un punto della scena può entrare
solo passando per il foro, e da lì colpisce il foglio in **un solo posto**.
Ogni punto del mondo, quindi, ha il suo pixel: è per questo che la foto
assomiglia alla scena.

Il guaio è che la freccia funziona in un senso solo. Un punto del mondo dà un
pixel, ma un pixel non dà un punto del mondo: tutti i punti allineati con il
foro, il vicino e quello dieci metri più in là, colpiscono lo stesso identico
posto sul foglio. È il motivo per cui nelle foto ricordo si può "reggere" la
Torre di Pisa con una mano: la mano è vicina, la torre è lontana, ma dal punto
di vista dell'obiettivo sono allineate, e la foto non conosce la differenza.

Un oggetto piccolo e vicino e uno grande e lontano fanno lo stesso pixel. La
distanza non è nascosta nell'immagine, è **andata perduta** nel momento dello
scatto. Riconquistarla è tutto il mestiere di questa sezione.

`````

`````{tab} Superiore

Nel modello **stenopeico** (in inglese *pinhole*), un punto $\mathbf{p} = (X,
Y, Z)$ espresso nel sistema di riferimento della fotocamera si proietta sul
piano immagine in

$$
u = f\,\frac{X}{Z} + c_x, \qquad v = f\,\frac{Y}{Z} + c_y,
$$

dove $f$ è la lunghezza focale espressa in pixel e $(c_x, c_y)$ è il punto
principale, cioè dove l'asse ottico buca il sensore. In coordinate omogenee la
stessa cosa è lineare, ed è questo il motivo per cui si usano:

$$
\tilde{\mathbf{u}} \sim \mathbf{K}\,[\mathbf{R} \mid \mathbf{t}]\,
\tilde{\mathbf{p}}, \qquad
\mathbf{K} = \begin{pmatrix} f & 0 & c_x \\ 0 & f & c_y \\ 0 & 0 & 1
\end{pmatrix},
$$

dove $\mathbf{K}$ raccoglie i **parametri intrinseci** (com'è fatta la
fotocamera) e $[\mathbf{R} \mid \mathbf{t}]$ gli **estrinseci** (dov'è e come
è orientata). Il simbolo $\sim$ ricorda che l'uguaglianza vale a meno di un
fattore di scala: per tornare ai pixel si divide per la terza componente.

**È quella divisione a distruggere l'informazione.** La mappa $(X,Y,Z) \mapsto
(u,v)$ non è invertibile: l'insieme dei punti che finiscono nello stesso pixel
è l'intera semiretta uscente dal centro ottico, e $\mathbf{p}$ e $\lambda
\mathbf{p}$ con $\lambda > 0$ sono indistinguibili. Un'immagine determina una
**direzione** per ogni pixel, mai una posizione.

Le fotocamere vere aggiungono due complicazioni che in pratica si misurano e
si tolgono. La prima è la **distorsione dell'obiettivo**, che curva le rette
soprattutto verso i bordi e si modella con pochi coefficienti radiali e
tangenziali. La seconda riguarda i valori dei pixel: quello che il file
contiene non è proporzionale alla luce entrata, perché in mezzo c'è una
codifica **gamma** (grossolanamente $I_{\text{file}} \approx I_{\text{luce}}
^{1/2{,}2}$ per sRGB). Ha una conseguenza pratica che vale oltre questo
capitolo: mediare, sfocare o comporre immagini sui valori codificati è
sbagliato in senso fisico, e va fatto dopo averli riportati in scala lineare
{cite}`szeliski2022computer`.

`````

Stimare $\mathbf{K}$ e i coefficienti di distorsione si chiama
**calibrazione**, e si fa mostrando alla fotocamera un oggetto di geometria
nota, tipicamente una scacchiera. Non è un dettaglio da laboratorio: senza
$\mathbf{K}$ i pixel restano numeri senza scala, e nessuna delle ricostruzioni
che seguono è possibile in unità metriche.

## Trovare la stessa cosa in due foto

Se una foto sola non basta, se ne prendono due. Ma perché due foto aiutino
bisogna prima risolvere un problema che sembra banale e non lo è: **capire
quale punto della prima corrisponde a quale punto della seconda**.

`````{tab} Elementare

Metti due fotografie della stessa piazza, scattate da due posizioni diverse, e
prova a indicare in entrambe lo stesso spigolo della stessa finestra. Tu lo
fai in un secondo. Un programma no: per lui le due immagini sono due griglie
di numeri, e i numeri sono cambiati tutti, perché è cambiato il punto di
vista, la luce, forse l'ora del giorno.

La soluzione trovata negli anni Ottanta e Novanta è in due tempi. Primo:
scegliere solo i punti **facili da ritrovare**. Una zona di cielo azzurro è
inutile, perché tutte le sue parti si somigliano; uno spigolo è ottimo, perché
è diverso da tutto ciò che ha intorno in ogni direzione. Secondo: descrivere
ogni punto scelto con una piccola scheda segnaletica, costruita in modo da non
cambiare se l'immagine viene ingrandita, ruotata o schiarita. Poi si
confrontano le schede.

Rimane un problema: qualche accoppiamento sarà sbagliato, e basta un errore
grosso per rovinare un calcolo fatto sulla media. Il rimedio si chiama
**consenso**: invece di usare tutti i dati, si prendono a caso pochissimi
punti, si calcola la risposta che darebbero, e si contano quanti altri sono
d'accordo. Si ripete centinaia di volte e si tiene l'ipotesi con più
sostenitori. Chi sbaglia non ha compagni, e resta fuori da solo.

`````

`````{tab} Superiore

I tre pezzi hanno nomi precisi ed età ben definita.

Il **rilevatore**: il rivelatore di angoli di Harris e Stephens (1988) valuta
la matrice di autocorrelazione dei gradienti in un intorno,

$$
\mathbf{M} = \sum_{w} \begin{pmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2
\end{pmatrix},
$$

e cerca i punti dove **entrambi** gli autovalori sono grandi (variazione in
ogni direzione, cioè un angolo), distinguendoli da quelli dove uno solo lo è
(un bordo, ambiguo lungo la sua direzione) {cite}`harris1988combined`.

Il **descrittore**: SIFT di David Lowe cerca gli estremi di una differenza di
gaussiane su più scale, il che dà la posizione **e** la scala, assegna
un'orientazione dominante e costruisce un vettore di 128 numeri fatto di
istogrammi di gradienti orientati, normalizzato. Invarianza a scala e
rotazione per costruzione, robustezza all'illuminazione per normalizzazione
{cite}`lowe2004distinctive`.

La **stima robusta**: RANSAC di Fischler e Bolles (1981) campiona il minimo
numero di corrispondenze necessarie a determinare il modello (quattro per
un'omografia, otto per la geometria di due viste), conta gli *inlier* entro
una soglia, e itera. Con una frazione $w$ di corrispondenze buone e un modello
che ne richiede $s$, la probabilità di aver estratto almeno un campione
interamente pulito in $N$ tentativi è $1 - (1 - w^s)^N$: si sceglie $N$ per
portarla dove serve {cite}`fischler1981random`.

`````

Vale la pena fermarsi un istante su questo passaggio, perché dice qualcosa sul
resto del libro. Trovare punti ripetibili, descriverli in modo invariante,
scartare le corrispondenze sbagliate: sono tre pezzi di ingegneria umana
raffinatissima, frutto di vent'anni di lavoro, e sono esattamente ciò che la
rivoluzione delle reti convoluzionali ha reso in gran parte superfluo. Quando
diciamo che le feature si sono smesse di disegnare e si sono cominciate a
imparare, **questo** è ciò che si è smesso di disegnare. Il confronto
funziona anche al contrario: qui la geometria resta, ed è ancora quella di
Longuet-Higgins. Le reti hanno sostituito la parte fragile, non quella
dimostrata.

## Il vincolo epipolare: da un piano a una retta

Trovare le corrispondenze sarebbe un problema quadratico (ogni pixel della
prima immagine contro tutti i pixel della seconda) se non fosse per una
proprietà geometrica che lo riduce di un'intera dimensione.

```{figure} ../figures/vincolo-epipolare.svg
:name: fig-vincolo-epipolare
:alt: "In alto due immagini affiancate: nella sinistra un solo punto evidenziato, nella destra una retta tratteggiata con tre punti candidati sopra. In basso lo schema che lo spiega: dal centro ottico della prima fotocamera parte un raggio, e tre punti a profondità crescente lungo quel raggio proiettano nella seconda fotocamera in tre posizioni diverse ma allineate."
:width: 92%

Un punto nella prima immagine non corrisponde a un punto nella seconda, ma a
una **retta**. Tutti i punti del raggio uscente dal primo centro ottico danno
lo stesso pixel a sinistra; a destra cadono in posti diversi, e allineati.
```

`````{tab} Elementare

Guarda {numref}`fig-vincolo-epipolare`. Hai scelto un pixel nella foto di
sinistra. Sappiamo che il punto del mondo che l'ha prodotto sta da qualche
parte lungo un raggio: potrebbe essere a due metri o a venti, la foto non lo
dice. Adesso però immagina di guardare quel raggio dalla seconda fotocamera.
Un raggio è una retta nello spazio, e la foto di una retta è una retta.

Quindi: il punto che cerchi nella seconda foto, qualunque sia la profondità
vera, **sta su una retta ben precisa**, che si può calcolare in anticipo
conoscendo solo la posizione reciproca delle due fotocamere. Non devi cercare
in tutta l'immagine, devi cercare lungo una riga.

È il passaggio che rende praticabile tutto il resto. Milioni di candidati
diventano qualche centinaio, e a lavorare non è un modello di come sono fatte
le cose: è una legge geometrica che vale sempre, per qualsiasi scena.

`````

`````{tab} Superiore

Il piano che contiene i due centri ottici e il punto $\mathbf{p}$ si chiama
**piano epipolare**; la sua intersezione con ciascun piano immagine è la
**retta epipolare** corrispondente. Tutte le rette epipolari di un'immagine
passano per uno stesso punto, l'**epipolo**, che è la proiezione dell'altro
centro ottico (e può cadere fuori dall'immagine, o all'infinito quando gli
assi ottici sono paralleli).

In coordinate normalizzate, con fotocamere calibrate, il vincolo si scrive con
la **matrice essenziale** $\mathbf{E} = [\mathbf{t}]_\times \mathbf{R}$,
introdotta da Longuet-Higgins su *Nature* nel 1981 insieme all'algoritmo a
otto punti che la stima {cite}`longuethiggins1981computer`:

$$
\tilde{\mathbf{x}}_R^\top \, \mathbf{E} \, \tilde{\mathbf{x}}_L = 0 .
$$

In pixel, con fotocamere non calibrate, lo stesso vincolo passa per la
**matrice fondamentale** $\mathbf{F} = \mathbf{K}_R^{-\top}\,
[\mathbf{t}]_\times \mathbf{R}\, \mathbf{K}_L^{-1}$, e

$$
\tilde{\mathbf{x}}_R^\top \, \mathbf{F} \, \tilde{\mathbf{x}}_L = 0 ,
$$

dove $[\mathbf{t}]_\times$ è la matrice antisimmetrica associata al prodotto
vettoriale per $\mathbf{t}$. Il prodotto $\mathbf{F}\tilde{\mathbf{x}}_L$ **è**
la retta epipolare nella seconda immagine, scritta come terna di coefficienti
$(a,b,c)$ dell'equazione $au + bv + c = 0$: il vincolo dice semplicemente che
$\tilde{\mathbf{x}}_R$ le appartiene. $\mathbf{F}$ ha rango 2 e sette gradi di
libertà, ed è definita a meno di scala {cite}`hartley2004multiple`.

In pratica si **rettificano** le due immagini, cioè si applica a ciascuna
un'omografia che porta gli epipoli all'infinito. Dopo la rettificazione le
rette epipolari sono orizzontali e allineate fra le due immagini, e la ricerca
della corrispondenza diventa uno scorrimento lungo la stessa riga di pixel:
è la ragione per cui le telecamere stereo si montano affiancate e allineate.

`````

## Dalla disparità alla profondità

Nel caso rettificato la geometria si riduce a una formula sola, ed è quella
che il nostro sistema visivo usa da sempre.

`````{tab} Elementare

Tieni un dito davanti al naso e chiudi alternativamente un occhio: il dito
salta. Ora allontanalo il più possibile: salta molto meno. Quel salto si
chiama **disparità**, e la sua misura è la misura della distanza. Vicino,
salto grande; lontano, salto piccolo; infinitamente lontano, nessun salto.

La relazione è un'inversa, non una proporzione: raddoppiando la distanza il
salto si dimezza. Ha due conseguenze che si toccano con mano. La prima è che
la stereo è **precisa da vicino e vaga da lontano**: a due metri qualche
pixel di disparità in più o in meno cambia poco, a cinquanta metri cambia
tutto. La seconda è che allontanare le due telecamere aumenta i salti e quindi
la precisione, ma restringe la zona che entrambe vedono. È il compromesso che
decide come si costruisce una telecamera stereo, e il motivo per cui i nostri
occhi distano sei centimetri e non uno o trenta.

E c'è un caso in cui il metodo fallisce del tutto: un muro bianco. Se lungo la
riga da esplorare tutti i pixel si somigliano, non c'è modo di dire quale
corrisponda a quale. La geometria ha fatto il suo dovere riducendo la ricerca
a una retta; su quella retta, però, ci vuole qualcosa da riconoscere.

`````

`````{tab} Superiore

Con due fotocamere identiche, assi ottici paralleli e base $B$ (la distanza fra
i centri ottici), un punto a profondità $Z$ si proietta a

$$
u_L = f\frac{X}{Z} + c_x, \qquad u_R = f\frac{X - B}{Z} + c_x ,
$$

da cui la **disparità** $d = u_L - u_R = fB/Z$ e quindi

$$
Z = \frac{f B}{d}.
$$

Le due conseguenze si leggono derivando: $\partial Z / \partial d = -fB/d^2 =
-Z^2/(fB)$. L'errore in profondità cresce con il **quadrato** della
profondità, e si riduce allargando la base o allungando la focale. È il motivo
per cui la disparità (che è proporzionale a $1/Z$) è spesso una
parametrizzazione numericamente migliore della profondità stessa: campionarla
uniformemente significa campionare finemente il vicino e grossolanamente il
lontano, che è esattamente dove serve precisione.

Il calcolo denso della disparità è un problema di ottimizzazione, e la sua
tassonomia classica {cite}`scharstein2002taxonomy` distingue quattro pezzi:
una misura di somiglianza fra finestre (differenze assolute, correlazione
normalizzata, *census transform*), un'aggregazione spaziale, un'ottimizzazione
globale con un termine di regolarità che favorisce disparità localmente
costanti, e un raffinamento sub-pixel. Il termine di regolarità è ciò che
tappa i buchi nelle zone senza tessitura: non è geometria, è un **prior** sulla
forma delle superfici, e le reti moderne lo sostituiscono con un prior appreso
da grandi collezioni di scene.

`````

## Quando a muoversi è la scena: il flusso ottico

La stereo confronta due immagini prese nello stesso istante da posizioni
diverse. Il **flusso ottico** confronta due immagini prese dalla stessa
posizione in istanti diversi, e stima per ogni pixel di quanto si è spostato.
È lo stesso problema di corrispondenza, senza il regalo della retta epipolare.

`````{tab} Elementare

Guarda un palo attraverso il finestrino di un treno che parte: scorre in
fretta. Guarda una montagna all'orizzonte: sta quasi ferma. Anche il movimento,
come il salto fra i due occhi, dice qualcosa sulla distanza, e infatti si
chiama **parallasse**. È così che stimiamo la profondità di una scena
muovendo la testa, con un occhio solo.

C'è un limite curioso, e si chiama problema dell'apertura. Guarda un palo che
si muove attraverso un tubo di cartone stretto: vedi solo un bordo verticale
che scivola, e **non puoi dire** se il palo va di lato o anche in diagonale,
perché scivolando lungo sé stesso non produce alcun cambiamento visibile.
Guardando un pezzo di bordo, il movimento lungo il bordo è invisibile. Serve
uno spigolo, o serve mettere insieme quello che dicono le zone vicine.

`````

`````{tab} Superiore

L'ipotesi di partenza è la **costanza della luminosità**: lo stesso punto
della scena ha lo stesso valore in due fotogrammi consecutivi,
$I(x, y, t) = I(x + \delta x,\, y + \delta y,\, t + \delta t)$. Sviluppando al
primo ordine si ottiene l'equazione del flusso ottico

$$
I_x u + I_y v + I_t = 0,
$$

con $(u, v)$ il flusso incognito. È **un'equazione in due incognite** per ogni
pixel: da qui il problema dell'apertura, in forma algebrica. Solo la
componente del flusso parallela al gradiente è determinata.

Le due soluzioni storiche, entrambe del 1981, chiudono il sistema in modi
diversi. **Lucas e Kanade** assumono flusso costante in una finestra e
risolvono ai minimi quadrati il sistema sovradeterminato che ne risulta: la
matrice da invertire è la stessa $\mathbf{M}$ di Harris, e il flusso è ben
determinato esattamente dove c'è un angolo {cite}`lucas1981iterative`. **Horn
e Schunck** aggiungono invece un termine globale di regolarità che penalizza
le variazioni del flusso, ottenendo una stima densa
{cite}`horn1981determining`. Dato locale più prior di regolarità: è la stessa
struttura della stereo densa, e la stessa che ritroveremo nei metodi appresi.

Oggi lo stato dell'arte è una rete che conserva quella struttura invece di
buttarla. **RAFT** costruisce esplicitamente il volume di correlazione fra
tutte le coppie di pixel (il "dato") e poi lo interroga con un aggiornamento
ricorrente che raffina il campo di flusso un passo alla volta (il "prior"),
riducendo l'errore su Sintel del 30% rispetto al metodo migliore precedente
{cite}`teed2020raft`. L'architettura è nuova, l'anatomia del problema è quella
del 1981.

`````

## Molte viste: structure from motion e SLAM

Con due immagini si ricava la forma della scena a meno di un fattore di scala
globale (le due viste non possono sapere se stanno guardando un palazzo da
lontano o un plastico da vicino: serve una misura esterna, un oggetto di
dimensione nota, un'unità inerziale). Con molte immagini si può fare di più, e
il problema prende il nome di **structure from motion**: stimare insieme
*dove erano le fotocamere* e *dov'erano i punti*, avendo solo le foto.

`````{tab} Elementare

È un problema dell'uovo e della gallina. Se sapessi dove sono le fotocamere,
calcolare i punti 3D sarebbe geometria elementare. Se sapessi dove sono i
punti, calcolare le posizioni delle fotocamere lo sarebbe altrettanto. Non
sai né l'uno né l'altro, e devi trovare entrambi insieme.

Si fa a tentativi organizzati: si parte da due foto, si stima una soluzione
approssimata, si aggiungono le altre una alla volta, e ogni tanto si aggiusta
tutto insieme minimizzando un unico errore, la distanza fra dove ogni punto
**appare** nelle foto e dove il modello attuale dice che dovrebbe apparire.
Quando quella distanza è piccola per tutti i punti in tutte le foto, la
ricostruzione è coerente.

Quando lo stesso calcolo si fa in tempo reale, mentre il dispositivo si muove,
si chiama **SLAM**: localizzarsi e costruire la mappa contemporaneamente. È
quello che fa un robot aspirapolvere, un visore per la realtà aumentata, un
drone che rientra da solo.

`````

`````{tab} Superiore

Il criterio è l'**errore di riproiezione**: dati i parametri delle fotocamere
$\{\mathbf{K}_j, \mathbf{R}_j, \mathbf{t}_j\}$ e i punti $\{\mathbf{p}_i\}$,

$$
\min \; \sum_{i,j} v_{ij} \,\big\| \, \mathbf{u}_{ij} - \pi(\mathbf{K}_j,
\mathbf{R}_j, \mathbf{t}_j, \mathbf{p}_i) \, \big\|^2 ,
$$

dove $\pi$ è la proiezione, $\mathbf{u}_{ij}$ è dove il punto $i$ è stato
*osservato* nell'immagine $j$, e $v_{ij}$ vale 1 se quell'osservazione esiste.
La minimizzazione congiunta si chiama **bundle adjustment** e si risolve con
Levenberg-Marquardt sfruttando la struttura sparsa del problema: ogni punto
compare in poche immagini, quindi la matrice normale è a blocchi e si può
eliminare per complemento di Schur.

Due cose meritano di essere notate. La prima è che il problema è **non
convesso** e serve una buona inizializzazione, che è il motivo per cui le
pipeline procedono in modo incrementale invece di ottimizzare tutto da subito
{cite}`schoenberger2016structure`. La seconda è che la soluzione è definita a
meno di una similarità (sette gradi di libertà: rotazione, traslazione,
scala), e la scala globale resta **indeterminata** senza informazione esterna.

`````

Questa sezione, apparentemente la più tecnica, è quella che serve di più
subito dopo: le pose delle fotocamere che escono da una ricostruzione
*structure from motion* sono l'ingresso obbligatorio dei metodi di rendering
neurale della sezione seguente. Chi ha provato a costruire un NeRF da un video
e ha ottenuto una nuvola confusa, nove volte su dieci non ha sbagliato la
rete: ha sbagliato le pose.

## Profondità da una sola immagine

Restava un fatto: da un'immagine sola la profondità è **matematicamente**
indeterminata. Eppure noi la vediamo, guardando una fotografia con un occhio
solo, e da qualche anno la vedono anche le reti. Vale la pena capire perché
non è una contraddizione.

`````{tab} Elementare

Guardando una foto di una strada sai benissimo che le case in fondo sono
lontane. Non lo sai per geometria: lo sai perché conosci il mondo. Sai quanto
è grande una porta, sai che le righe della carreggiata sono parallele e quindi
se convergono è per prospettiva, sai che il cielo è dietro, sai che se un
oggetto ne copre un altro sta davanti, sai che la foschia sbianca il lontano.

Una rete addestrata su milioni di foto con la profondità misurata impara la
stessa cosa: non calcola, **riconosce**. Ha visto abbastanza scene da sapere
come si dispongono di solito. È un'ottima cosa e ha un limite preciso, che
conviene tenere a mente: essendo un ricordo statistico e non una misura, si
può ingannare. Una fotografia di una fotografia viene letta come una scena
vera, e un plastico ben fatto viene letto come un palazzo.

C'è poi un limite che non è un difetto ma una legge: la **scala** resta
sconosciuta. La rete può dire con precisione che quell'auto è due volte più
lontana di quell'albero, e non può dire se siano a dieci metri o a cento.
Nessun indizio nell'immagine lo contiene.

`````

`````{tab} Superiore

Il problema è mal posto in senso stretto: infinite scene generano la stessa
immagine, e la classe di ambiguità include almeno la scala globale. Un modello
monoculare non risolve la geometria, apprende un **prior** $p(\text{scena})$
e restituisce il massimo a posteriori dato quel prior. La distinzione
terminologica che ne discende è utile: si parla di profondità **relativa**
(ordinamento, o profondità a meno di scala e offset) contro profondità
**metrica** (in metri), e la seconda richiede vincoli aggiuntivi, per esempio
intrinseci noti o un sensore inerziale.

Il progresso decisivo è stato di dati, non di architettura. **MiDaS** mostra
che mescolando dataset molto diversi (scansioni 3D, stereo da film, ricostruzioni
*structure from motion*, sintetici), ognuno con la sua nozione di profondità e
la sua scala, si ottiene un modello che generalizza a scene mai viste. La
chiave è una **loss invariante a scala e offset**, che permette di sommare
gradienti provenienti da sorgenti non commensurabili
{cite}`ranftl2022towards`. **Depth Anything** spinge la stessa strategia sul
non etichettato, usando un modello maestro per pseudo-etichettare milioni di
immagini {cite}`yang2024depth`.

Il quadro utile per il libro è questo: la geometria dà **vincoli esatti ma
insufficienti**, l'apprendimento dà **un prior sufficiente ma fallibile**, e i
sistemi che funzionano davvero usano entrambi. Uno stereo appreso mantiene la
ricerca lungo la retta epipolare e impara solo la parte ambigua; una pipeline
di ricostruzione usa la profondità monoculare per inizializzare e la geometria
multi-vista per correggerla.

`````

## In pratica: la geometria si può verificare

Il bello di questa sezione è che tutto ciò che afferma si può controllare con
un calcolo, senza addestrare niente. Costruiamo una scena finta di cui
conosciamo le risposte, proiettiamola in due fotocamere e verifichiamo due
cose: che la formula della disparità restituisce la profondità vera, e che il
vincolo epipolare vale davvero.

```python
import numpy as np

# --- una fotocamera: matrice degli intrinseci ---
f, cx, cy = 700.0, 320.0, 240.0          # focale e centro, in pixel
K = np.array([[f, 0, cx],
              [0, f, cy],
              [0, 0,  1]])

def proietta(K, R, t, P):
    """Porta i punti 3D P (N,3) nel sistema della fotocamera e li proietta."""
    Pc = P @ R.T + t                      # rototraslazione nel sistema camera
    x = Pc @ K.T                          # proiezione prospettica (omogenea)
    return x[:, :2] / x[:, 2:3]           # la divisione per Z: qui si perde la profondità

rng = np.random.default_rng(0)
P = np.column_stack([rng.uniform(-1.5, 1.5, 8),
                     rng.uniform(-1.0, 1.0, 8),
                     rng.uniform( 4.0, 9.0, 8)])   # otto punti davanti alle camere

# --- caso rettificato: seconda camera traslata di B lungo x, stessa orientazione ---
B = 0.30                                   # base, in metri
R1, t1 = np.eye(3), np.zeros(3)
R2, t2 = np.eye(3), np.array([-B, 0.0, 0.0])

u1 = proietta(K, R1, t1, P)
u2 = proietta(K, R2, t2, P)

disparita = u1[:, 0] - u2[:, 0]
Z_stimata = f * B / disparita

print("profondità vera   :", np.round(P[:, 2], 3))
print("profondità stimata:", np.round(Z_stimata, 3))
print("errore massimo    :", np.abs(Z_stimata - P[:, 2]).max())
print("righe uguali (rettificato):", np.allclose(u1[:, 1], u2[:, 1]))
```

Le due colonne coincidono e l'errore massimo è dell'ordine di $10^{-15}$, cioè
la precisione della macchina: $Z = fB/d$ non è un'approssimazione, è
un'identità. E `righe uguali` conferma che nel caso rettificato la
corrispondenza si cerca sulla **stessa riga**.

Ora il caso generale, con la seconda fotocamera ruotata di otto gradi. Qui non
c'è più nessuna riga comoda, ma il vincolo epipolare vale lo stesso.

```python
ang = np.deg2rad(8.0)
Rz = np.array([[ np.cos(ang), 0, np.sin(ang)],
               [ 0,           1, 0          ],
               [-np.sin(ang), 0, np.cos(ang)]])
t3 = np.array([-B, 0.02, 0.0])
u3 = proietta(K, Rz, t3, P)

def antisimmetrica(v):
    """La matrice che realizza il prodotto vettoriale: [v]_x @ w == np.cross(v, w)."""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

F = np.linalg.inv(K).T @ antisimmetrica(t3) @ Rz @ np.linalg.inv(K)

def omogenee(u):
    return np.column_stack([u, np.ones(len(u))])

residuo = np.einsum('ij,jk,ik->i', omogenee(u3), F, omogenee(u1))
print("residuo epipolare :", np.abs(residuo).max())
```

Il residuo massimo è dell'ordine di $10^{-17}$. Per ognuno degli otto punti,
il pixel nella seconda immagine sta **esattamente** sulla retta
$\mathbf{F}\tilde{\mathbf{x}}_L$ calcolata dalla prima. Nessuna rete, nessun
dato: è un'identità algebrica che dipende solo da come è fatta la proiezione,
ed è la ragione per cui questa parte della visione artificiale non è
invecchiata.

```{admonition} Da ricordare
:class: important
- **Proiettare perde una dimensione.** Un pixel determina una *direzione*, mai
  una posizione: tutti i punti di una semiretta uscente dal centro ottico
  danno lo stesso pixel. La profondità non è nascosta nell'immagine, è stata
  cancellata dalla divisione per $Z$.
- Il modello **stenopeico** con la matrice degli intrinseci $\mathbf{K}$
  descrive la fotocamera; stimarla si chiama **calibrazione** e senza di essa
  i pixel non hanno scala.
- Il **vincolo epipolare** riduce la ricerca della corrispondenza da un piano
  a una retta: $\tilde{\mathbf{x}}_R^\top \mathbf{F} \tilde{\mathbf{x}}_L = 0$.
  Rettificando le immagini quelle rette diventano orizzontali.
- Nel caso rettificato, $Z = fB/d$: la profondità è **inversamente**
  proporzionale alla disparità, quindi la stereo è precisa da vicino e vaga da
  lontano, e l'errore cresce come $Z^2$.
- Il **flusso ottico** è lo stesso problema nel tempo; la sua equazione ha due
  incognite per pixel (problema dell'apertura), e si chiude con un'ipotesi
  locale (Lucas-Kanade) o globale (Horn-Schunck).
- **Structure from motion** stima insieme pose e punti minimizzando l'errore
  di riproiezione (*bundle adjustment*); la scala globale resta indeterminata.
  Le sue pose sono l'ingresso obbligatorio del rendering neurale.
- La profondità **da una sola immagine** è matematicamente indeterminata: le
  reti non la calcolano, applicano un prior appreso. Ottimo in pratica,
  ingannabile per costruzione, e cieco alla scala assoluta.
```
