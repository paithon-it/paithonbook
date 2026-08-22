# Dove sono le cose: geometria, corrispondenze e profondità

Attorno al 1413, sulla piazza del Duomo di Firenze, Filippo Brunelleschi fece
una cosa che nessuno aveva mai fatto. Aveva dipinto su una tavoletta di mezzo
braccio di lato (una trentina di centimetri) il Battistero di San Giovanni,
intarsi di marmo compresi, e nella tavoletta aveva praticato un foro passante.

L'esperimento andava fatto così. Ti metti dentro il portale del Duomo, nello
stesso punto esatto da cui Brunelleschi aveva dipinto, con il Battistero
davanti. Impugni la tavoletta a un palmo dall'occhio, ma girata: il dipinto
guarda dalla parte opposta alla tua, quindi tu vedi solo il retro. Accosti
l'occhio al foro e guardi attraverso: dall'altra parte c'è il Battistero vero.
Con l'altra mano alzi uno specchietto a un braccio di distanza, davanti alla
tavoletta, e adesso attraverso il foro non vedi più il Battistero vero, ma il
riflesso del dipinto che gli sta di fronte. Togli lo specchio, torna il
Battistero vero; lo rimetti, torna il dipinto. Erano identici. La parte alta
della tavoletta, dove ci sarebbe stato il cielo, era coperta d'argento brunito,
così che nel riflesso si vedessero le nuvole vere che passavano.

Quella tavoletta è il primo esperimento noto costruito per dimostrare
una legge geometrica: **un punto di vista, un foro, e il mondo tridimensionale
si schiaccia su una superficie piatta in un modo prevedibile e calcolabile**.
Vent'anni dopo Leon Battista Alberti ne scrisse le regole nel *De pictura*, e
la prospettiva lineare diventò una tecnica insegnabile.

Il foro di Brunelleschi è, letteralmente, il modello di fotocamera che usiamo
ancora oggi. L'obiettivo di vetro del telefono non cambia le carte in tavola:
serve a far entrare più luce di quanta ne passi da un buco di spillo, ma i
raggi li fa convergere in un punto solo, e quel punto fa la parte del foro.

E il foro porta con sé il problema che occupa tutta questa sezione. Un punto
del mondo, per dire dov'è, ha bisogno di tre numeri: quanto a destra, quanto in
alto, quanto lontano. Un punto sulla foto ne ha due, la riga e la colonna del
suo pixel. **Proiettare significa quindi buttare via un numero per ogni
punto**, e quel numero è proprio la distanza. Il resto del capitolo si è
occupato di che *cosa* c'è in un'immagine; qui ci occupiamo di **dove**, e la
domanda è più difficile di quanto sembri, perché l'informazione che serve non
è nascosta nell'immagine: è stata proprio cancellata.

## La proiezione perde una dimensione

Cominciamo da come un punto del mondo diventa un pixel.

`````{tab} Elementare

Una scatola chiusa ha un foro piccolissimo su una faccia e un foglio sulla
faccia opposta. La luce che parte da un punto della scena può entrare solo
passando per il foro, e da lì colpisce il foglio in **un solo posto**. Ogni
punto del mondo, quindi, ha il suo pixel: è per questo che la foto assomiglia
alla scena.

Il guaio è che il legame funziona in un senso solo. Un punto del mondo dà un
pixel, ma un pixel non dà un punto del mondo: tutti i punti allineati con il
foro, il vicino e quello dieci metri più in là, colpiscono lo stesso identico
posto sul foglio. È il motivo per cui nelle foto ricordo si può "reggere" la
Torre di Pisa con una mano: la mano è vicina, la torre è lontana, ma dal punto
di vista dell'obiettivo sono allineate, e la foto non conosce la differenza.

Un oggetto piccolo e vicino e uno grande e lontano fanno lo stesso pixel. La
distanza non è nascosta nell'immagine, è **andata perduta** nel momento dello
scatto. Per riaverla bisogna rimetterla da fuori, e ci sono tre modi: guardare
la scena da un secondo punto, muoversi e guardare come le cose scorrono, o
mettere in campo quello che già si sa di com'è fatto il mondo.

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
^{1/2{,}2}$ per sRGB). Ha una conseguenza pratica che va ben oltre la
geometria: mediare, sfocare o comporre immagini sui valori codificati è
sbagliato in senso fisico, e va fatto dopo averli riportati in scala lineare
{cite}`szeliski2022computer`.

`````

Prima di fare qualunque conto, insomma, la fotocamera va **misurata**. Tre cose
servono. Quanto ingrandisce, cioè di quanti pixel si sposta l'immagine di un
oggetto quando l'oggetto si sposta di un centimetro: si chiama **focale**, e
dipende dall'obiettivo. Dove cade il punto in cui l'asse dell'obiettivo buca il
sensore, che uno immaginerebbe al centro esatto della foto e nella pratica non
lo è mai, perché sensore e lente vengono incollati da una macchina con una
tolleranza di qualche pixel. E di quanto quell'obiettivo incurva le linee
rette, cosa che si vede soprattutto ai bordi. Quei numeri, presi insieme,
descrivono com'è fatta la fotocamera e si chiamano i suoi **parametri
intrinseci**; ricavarli si chiama **calibrazione**, e si fa mostrando alla
fotocamera un oggetto di cui si conoscono le misure, tipicamente una scacchiera
stampata, e guardando come viene deformata. Non è un dettaglio da laboratorio:
senza quelle misure i numeri dei pixel non si possono convertire in
centimetri, e nessuna delle ricostruzioni che seguono può dare una risposta in
metri.

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

La soluzione trovata negli anni Ottanta e Novanta mette insieme tre mestieri.
Il primo è scegliere i punti **facili da ritrovare**, e non tutti lo sono allo
stesso modo. Un pezzetto di cielo azzurro non serve a niente, perché tutte le
sue parti si somigliano, e nella seconda foto un pezzetto di cielo vale
l'altro. Il bordo di un tetto va meglio, ma solo a metà: se il pezzetto che
stai guardando scivola in su o in giù te ne accorgi subito, perché il bordo
esce dall'inquadratura; se scivola lungo il tetto non te ne accorgi affatto,
perché in quella direzione il bordo è uguale a sé stesso. Uno spigolo, dove due
bordi si incontrano, è quello buono: da qualunque parte lo si sposti, qualcosa
cambia.

Il secondo mestiere è descrivere ogni punto scelto con una piccola scheda
segnaletica, costruita in modo da non cambiare se l'immagine viene ingrandita,
ruotata o schiarita. Poi si confrontano le schede.

Il terzo è buttare via gli accoppiamenti sbagliati, che ci saranno di sicuro;
basta un errore grosso per rovinare un calcolo fatto sulla media. Il rimedio si
chiama **consenso**: invece di usare tutti i dati, si prende a caso il numero
minimo di punti che basta a tirar fuori una risposta, si guarda che risposta
danno, e si contano quanti altri punti sono d'accordo, entro uno scarto deciso
prima. Si ripete centinaia di volte e si tiene l'ipotesi con più sostenitori.
Chi sbaglia non ha compagni, e resta fuori da solo.

Quel «numero minimo» va preso alla lettera, perché un punto in più da estrarre
si paga caro. Se metà degli accoppiamenti è sbagliata, un gruppetto di sette
punti presi a caso è tutto buono una volta su centoventotto; uno di otto, una
volta su duecentocinquantasei. Stessa fiducia nel risultato, il doppio dei
tentativi.

`````

`````{tab} Superiore

I tre pezzi hanno nomi precisi ed età ben definita.

Il **rilevatore**: quello di angoli proposto da Harris e Stephens (1988) valuta
la matrice di autocorrelazione dei gradienti in un intorno,

$$
\mathbf{M} = \sum_{w} \begin{pmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2
\end{pmatrix},
$$

e cerca i punti dove **entrambi** gli autovalori sono grandi (variazione in
ogni direzione, cioè un angolo), distinguendoli da quelli dove uno solo lo è
(un bordo, ambiguo lungo la sua direzione) {cite}`harris1988combined`. Senza
però calcolarli, ed è il contributo del paper: la risposta
$R = \det(\mathbf{M}) - k\,\mathrm{tr}(\mathbf{M})^2$ (con $k \approx 0{,}04$)
dice la stessa cosa a costo di quattro moltiplicazioni e di nessuna radice
quadrata, perché
$\det(\mathbf{M}) = \lambda_1\lambda_2$ e $\mathrm{tr}(\mathbf{M}) = \lambda_1
+ \lambda_2$. Guardare direttamente $\min(\lambda_1, \lambda_2)$ è invece la
variante di Shi e Tomasi (1994), che è un rilevatore diverso.

Il **descrittore**: SIFT di David Lowe cerca gli estremi di una differenza di
gaussiane su più scale, il che dà la posizione **e** la scala, assegna
un'orientazione dominante e costruisce un vettore di 128 numeri fatto di
istogrammi di gradienti orientati, normalizzato. Invarianza a scala e
rotazione per costruzione, robustezza all'illuminazione per normalizzazione
{cite}`lowe2004distinctive`.

La **stima robusta**: RANSAC di Fischler e Bolles (1981) campiona il minimo
numero di corrispondenze necessarie a determinare il modello (quattro per
un'omografia; **sette** per la matrice fondamentale, che di gradi di libertà
ne ha sette, o cinque per l'essenziale se le fotocamere sono calibrate), conta
gli *inlier* entro
una soglia, e itera. Con una frazione $w$ di corrispondenze buone e un modello
che ne richiede $s$, la probabilità di aver estratto almeno un campione
interamente pulito in $N$ tentativi è $1 - (1 - w^s)^N$: si sceglie $N$ per
portarla dove serve {cite}`fischler1981random`. Ed è qui che il minimo conta
davvero, perché $s$ sta all'esponente: con $w = 0{,}5$ e obiettivo $0{,}99$
servono circa $1177$ tentativi con $s = 8$, $588$ con $s = 7$ e $146$ con
$s = 5$. Usare l'algoritmo a otto punti dentro RANSAC, come si legge spesso,
costa il doppio dei campioni necessari.

`````

Questo passaggio dice qualcosa che va oltre la geometria. Trovare punti facili
da ritrovare, descriverli con una scheda che non cambia se l'immagine cambia,
buttare via gli accoppiamenti sbagliati: sono tre pezzi di ingegneria umana
raffinatissima, frutto di vent'anni di lavoro, e sono esattamente quelli che le
reti hanno reso in gran parte superflui. All'inizio del capitolo si diceva che
le regole per riconoscere si sono smesse di scrivere a mano e si sono
cominciate a far imparare: ecco, **questo** è ciò che si è smesso di scrivere a
mano.

Ma il confronto va guardato anche dall'altro lato, ed è la parte che sorprende.
La geometria è rimasta dov'era. Le formule che legano due fotografie della
stessa scena furono dimostrate fra la fine degli anni Settanta e l'inizio degli
anni Ottanta, e si usano oggi identiche, perché non sono ricette che funzionano
più o meno bene: sono teoremi. Le reti hanno sostituito la parte fragile, non
quella dimostrata.

## Il vincolo epipolare: da un piano a una retta

Trovare le corrispondenze costerebbe carissimo: ogni pixel della prima immagine
andrebbe confrontato con **tutti** i pixel della seconda, e siccome i pixel
sono un milione per parte i confronti sarebbero mille miliardi (per questo si
dice che il costo è *quadratico*: raddoppiando i pixel, il lavoro quadruplica).
Non è così, grazie a una proprietà geometrica che invece di farci cercare in
tutta la superficie della seconda immagine ci fa cercare lungo una linea sola,
e questo si chiama, in linguaggio tecnico, «ridurre la ricerca di una
dimensione»: da un piano a una retta.

```{figure} ../figures/vincolo-epipolare.svg
:name: fig-vincolo-epipolare
:alt: "In alto due immagini affiancate: nella sinistra un solo punto evidenziato, nella destra una retta tratteggiata con tre punti candidati sopra. In basso lo schema che lo spiega: dal centro ottico della prima fotocamera parte un raggio, e tre punti a profondità crescente lungo quel raggio proiettano nella seconda fotocamera in tre posizioni diverse ma allineate."
:width: 92%

Un punto nella prima immagine non corrisponde a un punto nella seconda, ma a
una **retta**. Tutti i punti del raggio che esce dal foro della prima
fotocamera (il suo **centro ottico**) danno lo stesso pixel a sinistra; a
destra cadono in posti diversi, e allineati.
```

`````{tab} Elementare

In {numref}`fig-vincolo-epipolare` hai scelto un pixel nella foto di sinistra.
Sappiamo che il punto del mondo che l'ha prodotto sta da qualche parte lungo un
raggio: potrebbe essere a due metri o a venti, la foto non lo dice. Adesso però
immagina di guardare quel raggio dalla seconda fotocamera.

Un raggio è una retta nello spazio, e la foto di una retta è sempre una retta.
Non è ovvio: la prospettiva rimpicciolisce le cose lontane e fa
convergere i binari, quindi qualcuno si aspetterebbe che
incurvi anche questa. Non lo fa, perché tutti i punti della retta e il foro
della seconda fotocamera stanno su uno stesso piano, e un piano taglia il
piano della pellicola lungo una retta. La prospettiva schiaccia le distanze,
non piega le rette.

Quindi: il punto che cerchi nella seconda foto, qualunque sia la profondità
vera, sta su una retta ben precisa, che si può calcolare in anticipo
conoscendo solo la posizione reciproca delle due fotocamere. Quella retta si
chiama **retta epipolare**, e l’*epipolo* da cui prende il nome è il punto in
cui ciascuna fotocamera vedrebbe l'altra. Non devi cercare in
tutta l'immagine, devi cercare lungo una riga.

È il passaggio che rende praticabile tutto il resto. Milioni di candidati
diventano qualche centinaio, e a lavorare non è un modello di come sono fatte
le cose: è una legge geometrica che vale sempre, per qualsiasi scena.

`````

`````{tab} Superiore

Il piano che contiene i due centri ottici e il punto $\mathbf{p}$ si chiama
**piano epipolare**; la sua intersezione con ciascun piano immagine è la
**retta epipolare** corrispondente. Tutte le rette epipolari di un'immagine
passano per uno stesso punto, l’**epipolo**, che è la proiezione dell'altro
centro ottico e può cadere fuori dall'immagine. Va all'infinito quando è la
**base**, cioè il segmento che unisce i due centri ottici, a essere parallela
al piano immagine: è il caso della coppia stereo affiancata, ed è il motivo per
cui rettificare una coppia consiste proprio nel mandare gli
epipoli all'infinito. Non c'entra il parallelismo fra gli assi ottici: due
fotocamere con assi perfettamente paralleli che si muovono *in avanti* hanno
l'epipolo dentro l'immagine, nel punto da cui la scena sembra espandersi (è il
caso di ogni telecamera montata su un'auto). La condizione va poi verificata su
un'immagine alla volta, perché i piani immagine sono due: ruotando verso
l'interno una sola delle due fotocamere, con la base laterale di prima, il suo
piano smette di essere parallelo alla base e il suo epipolo torna a distanza
finita, mentre l'altro resta all'infinito.

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

Il caso più comodo è quello in cui tutte le rette epipolari sono orizzontali e
la riga numero cento della prima immagine corrisponde alla riga numero cento
della seconda: allora cercare il gemello vuol dire scorrere una riga di pixel,
e basta. Quel caso lo si ottiene in due modi, montando le due fotocamere
affiancate e ben allineate, oppure raddrizzando le immagini dopo, con un
calcolo che le storce quel tanto che basta a metterle in quella posizione. In
tutti e due i casi si dice che le immagini sono **rettificate**, e da lì la
geometria si riduce a una formula sola, quella che il nostro sistema visivo usa
da sempre.

`````{tab} Elementare

Tieni un dito davanti al naso e chiudi alternativamente un occhio: il dito
salta. Ora allontanalo il più possibile: salta molto meno. Quel salto si
chiama **disparità**, e la sua misura è la misura della distanza. Vicino,
salto grande; lontano, salto piccolo; infinitamente lontano, nessun salto.

La relazione è un'inversa, non una proporzione: raddoppiando la distanza il
salto si dimezza. Su una telecamera con trenta centimetri fra i due obiettivi,
un oggetto a un metro salta $210$ pixel, a due metri $105$, a quattro metri
$52{,}5$, a otto metri poco più di $26$. Il primo di questi numeri dipende
anche da quanto l'obiettivo ingrandisce; il dimezzarsi a ogni raddoppio, invece,
vale sempre.

Ha due conseguenze che si toccano con mano. La prima è che
la stereo è precisa da vicino e vaga da lontano: a due metri qualche
pixel di disparità in più o in meno cambia poco, a cinquanta metri cambia
tutto. La seconda riguarda le leve su cui si può agire, e sono due: allontanare
le telecamere fra loro ingrandisce i salti e quindi la precisione, ma restringe
la zona che entrambe vedono; e lo stesso fa lo zoom, che ingrandisce i salti e
insieme rimpicciolisce la porzione di mondo inquadrata. È il compromesso che
decide come si costruisce una telecamera stereo, e il motivo per cui i nostri
occhi distano sei centimetri e non uno o trenta.

E c'è un caso in cui il metodo fallisce del tutto: un muro bianco. Se lungo la
riga da esplorare tutti i pixel si somigliano, non c'è modo di dire quale
corrisponda a quale. La geometria ha fatto il suo dovere riducendo la ricerca
a una retta; su quella retta, però, ci vuole qualcosa da riconoscere.

Chi vuole la distanza di ogni singolo pixel se la cava aggiungendo qualcosa che
la geometria non contiene: la scommessa che le superfici siano per lo più
lisce, cioè che due pixel vicini stiano quasi sempre più o meno alla stessa
distanza. Il muro bianco viene riempito così, tirando dentro le distanze
misurate sui suoi bordi, dove qualcosa da riconoscere c'era. È una scommessa
che di solito paga, e che a volte fa danni: davanti a una vetrata, o al bordo di
un tavolo che si affaccia sul vuoto, smussa in una rampa dolce un salto che
nella realtà è netto.

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

## Quando di mezzo c'è il tempo: il flusso ottico

La stereo confronta due immagini prese nello stesso istante da posizioni
diverse. Il **flusso ottico** confronta due immagini prese in istanti diversi e
stima per ogni pixel di quanto si è spostato. Non si chiede nemmeno se a
muoversi sia la scena o la fotocamera, perché dall'immagine sola le due cose
sono indistinguibili: un albero che scorre verso sinistra e una telecamera che
si sposta verso destra danno la stessa identica ripresa.

È lo stesso problema di corrispondenza di prima, ma il regalo di poco fa qui in
generale non c'è. La riga su cui cercare si poteva calcolare perché sapevamo
dove stavano le due fotocamere, mentre adesso non sappiamo di quanto si è
mossa la nostra, e per giunta la scena può cambiare forma da sola: una
bandiera, un viso, dell'acqua. Quando invece la scena è **rigida**, cioè si
muove tutta d'un pezzo, e il movimento della fotocamera è noto, la retta
epipolare torna eccome, ed è così che si ricava la profondità da un video preso
camminando.

`````{tab} Elementare

Guarda un palo attraverso il finestrino di un treno che parte: scorre in
fretta. Guarda una montagna all'orizzonte: sta quasi ferma. Anche il movimento,
come il salto fra i due occhi, dice qualcosa sulla distanza, e infatti si
chiama **parallasse**. È così che stimiamo la profondità di una scena
muovendo la testa, con un occhio solo.

Per misurare quello scorrimento bisogna ritrovare, nel fotogramma dopo, il
pezzo di scena che nel primo aveva un certo colore. Sotto c'è una promessa: che
lo stesso pezzo di scena si presenti con lo stesso colore da un fotogramma al
successivo. Il mondo di solito la mantiene; quando la rompe, il conto va a
sbattere. Una nuvola copre il sole: la piazza si scurisce tutta insieme, non si
è mosso niente, e il calcolo vede movimento dappertutto. Fa lo stesso effetto
l'ombra di una persona che scivola su un muro fermo.

C'è poi un limite curioso, e si chiama problema dell'apertura. Punta un tubo di
cartone stretto verso un palo che si muove: dentro il tubo si vede solo un
bordo che scivola, e da lì non si può dire se il palo va di lato o anche in
diagonale, perché scivolando lungo sé stesso non produce alcun cambiamento
visibile. Guardando un pezzo di bordo, il movimento lungo il bordo è
invisibile. Serve uno spigolo, o serve mettere insieme quello che dicono le
zone vicine.

`````

`````{tab} Superiore

L'ipotesi di partenza è la **costanza della luminosità**: lo stesso punto
della scena ha lo stesso valore in due fotogrammi consecutivi,
$I(x, y, t) = I(x + \delta x,\, y + \delta y,\, t + \delta t)$. Sviluppando al
primo ordine si ottiene l'equazione del flusso ottico

$$
I_x \dot{x} + I_y \dot{y} + I_t = 0,
$$

dove $(\dot{x}, \dot{y})$ è il flusso incognito, cioè la velocità con cui
l'immagine del punto scorre sul sensore, e $I_x, I_y, I_t$ sono le derivate
dell'immagine nelle due direzioni e nel tempo. È **un'equazione in due
incognite** per ogni
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

I metodi appresi conservano quella struttura invece di buttarla. **RAFT**
costruisce esplicitamente il volume di correlazione fra tutte le coppie di
pixel (il "dato") e poi lo interroga con un aggiornamento ricorrente che
raffina il campo di flusso un passo alla volta (il "prior")
{cite}`teed2020raft`: i due pezzi si riconoscono uno per uno nella tassonomia
di quarant'anni prima. L'architettura è nuova, l'anatomia del problema è
quella del 1981.

`````

Il tubo di cartone e l'equazione a due incognite sono la stessa cosa, e
{numref}`fig-apertura-flusso` è il punto del capitolo in cui conviene guardarla
muoversi invece che leggerla: il palo si sposta, e la finestrella di sinistra
non riesce a dire di quanto.

```{figure} ../figures/apertura-flusso.svg
:name: fig-apertura-flusso
:alt: "Due finestrelle rotonde affiancate, e in ciascuna scorre lo stesso palo inclinato che si muove verso destra. A sinistra si vede solo un tratto di bordo, e la freccia di ciò che si misura punta in diagonale, più corta della freccia del movimento vero, che è orizzontale. A destra si vede la punta del palo, e la freccia del misurato coincide con quella del movimento vero."
:width: 94%

Lo stesso palo e lo stesso identico movimento, guardati da due finestrelle
diverse. A sinistra si vede solo un tratto di bordo, e di quel movimento si
recupera la sola componente perpendicolare al bordo. A destra la finestrella
contiene la punta, e non manca più niente.
```

I due numeri sotto le finestrelle sono un conto vero, non un'illustrazione, e
conviene leggerli piano perché nascondono una trappola. Il palo si sposta di
**96 pixel in orizzontale**; la finestrella di sinistra, che vede solo un
tratto di bordo, ne misura **85**, e per giunta in una direzione sbagliata,
obliqua invece che orizzontale.

Verrebbe da dire: pazienza, sbaglia di undici pixel su novantasei, poco più del
dieci per cento. È qui la trappola, perché quegli undici non sono l'errore. La
freccia misurata e la freccia che manca per arrivare al movimento vero stanno
**ad angolo retto** fra loro, e due frecce ad angolo retto non si sommano come
si sommano i numeri: si sommano come i cateti di un triangolo rettangolo, con
Pitagora. Quella che manca è lunga **45 pixel**, e infatti
$\sqrt{85^2 + 45^2} = \sqrt{9250} = 96{,}2$, cioè i novantasei di partenza a
meno degli arrotondamenti. L'informazione persa non è un decimo, è quasi la
metà, e sfugge perché si nasconde in una direzione diversa da quella che si sta
guardando.

Quei 45 pixel sono scivolati **lungo** il bordo, ed è esattamente lì che il
palo, muovendosi, non cambia niente di ciò che si vede: la finestrella non li
può recuperare per quanto la si guardi bene, perché non hanno lasciato traccia.
Ed è la ragione per cui la finestra di Lucas e Kanade va messa dove c'è uno
spigolo: non perché lì l'immagine sia più nitida, ma perché uno spigolo ha due
bordi in due direzioni diverse, e quello che scivola lungo il primo si vede
scivolare attraverso il secondo.

`````{tab} Elementare

Da dove escono l'85 e il 45, se il movimento è di 96? Dall'inclinazione del
palo, che nel disegno è di 62 gradi. Sono le proporzioni di un triangolo
rettangolo: il pezzo che sopravvive è
$96 \times 0{,}883$, quasi $85$; il pezzo che si perde è
$96 \times 0{,}469$, poco più di $45$; e i
due fattori dipendono soltanto da quanto il palo è inclinato. Un palo verticale
li avrebbe $1$ e $0$ (il movimento orizzontale si misurerebbe tutto); un palo
orizzontale $0$ e $1$ (non se ne misurerebbe niente).

`````

`````{tab} Superiore

I due fattori sono $\sin 62^\circ = 0{,}883$ e $\cos 62^\circ = 0{,}469$: la
componente normale al bordo vale $96 \sin 62^\circ = 84{,}8$ pixel e quella
tangenziale $96 \cos 62^\circ = 45{,}1$. Solo la prima è determinata
dall'equazione del flusso, perché è la proiezione del moto sul gradiente
dell'immagine, e il gradiente è per costruzione ortogonale al bordo. La seconda
sta nel nucleo dell'equazione, dove il vincolo non dice nulla.

`````

## Molte viste: structure from motion e SLAM

Con due immagini, e con le fotocamere misurate come si diceva all'inizio, si
ricava la forma della scena ma non la sua taglia. Attenzione a non confonderlo
con quello che si è appena visto: nel caso dei due occhi la distanza fra loro
la conoscevamo, ed è quella a dare i metri. Se invece le due foto le ha scattate
una persona camminando, di quanto abbia camminato non lo sa nessuno, e allora
un palazzo fotografato da lontano e un plastico fotografato da vicino danno
esattamente le stesse due immagini. A rompere il pareggio dev'essere qualcosa
che viene da fuori: un oggetto di misura nota inquadrato nella scena, oppure
una **centralina inerziale**, il sensorino che nel telefono si accorge degli
spostamenti e delle rotazioni e che, integrando, sa dire di quanti centimetri
ci si è mossi. Con molte immagini si può fare di più, e
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
Un punto, quasi sempre, in molte delle foto non si vede proprio, e il conto
tiene in considerazione soltanto quelle in cui c'è. Quando la distanza è
piccola per ogni punto, in ognuna delle foto che lo contengono, la
ricostruzione è coerente.

L'ordine in cui si procede ha una ragione. L'aggiustamento sa solo scendere:
prende la sistemazione che ha fra le mani e la ritocca finché ogni altro
ritocco peggiora le cose. Partendo da una disposizione a caso si finisce in una
sistemazione tutta sbagliata da cui nessun ritocco fa uscire, con le fotocamere
messe dove non erano e i punti piazzati di conseguenza, a dar loro ragione. Le
due foto iniziali servono a partire già abbastanza vicini alla risposta giusta.

Quando lo stesso calcolo si fa in tempo reale, mentre il dispositivo si muove,
si chiama **SLAM**, sigla di *simultaneous localization and mapping*, cioè
«localizzarsi e disegnare la mappa nello stesso momento», che è poi l'uovo e la
gallina di prima con l'aggravante della fretta. È quello che fa un robot
aspirapolvere, un visore per la realtà aumentata, un drone che rientra da solo.

`````

`````{tab} Superiore

Il criterio è l’**errore di riproiezione**: dati i parametri delle fotocamere
$\{\mathbf{K}_j, \mathbf{R}_j, \mathbf{t}_j\}$ e i punti $\{\mathbf{p}_i\}$,

$$
\min \; \sum_{i,j} m_{ij} \,\big\| \, \mathbf{u}_{ij} - \pi(\mathbf{K}_j,
\mathbf{R}_j, \mathbf{t}_j, \mathbf{p}_i) \, \big\|^2 ,
$$

dove $\pi$ è la proiezione, $\mathbf{u}_{ij}$ è dove il punto $i$ è stato
*osservato* nell'immagine $j$, e $m_{ij}$ vale 1 se quell'osservazione esiste e
0 altrimenti.
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

Quel «a meno di una similarità» vale però solo con gli intrinseci noti. Senza
calibrazione l'ambiguità è molto più larga: si ottiene una ricostruzione
*proiettiva*, che non conserva né gli angoli né i rapporti fra lunghezze, ed è
come guardare il palazzo attraverso una deformazione prospettica arbitraria.
Per riportarla a una forma metrica bisogna stimare gli intrinseci a
posteriori, ed è quello che si chiama **auto-calibrazione**: un'altra ragione
per calibrare prima.

`````

Questa sezione, apparentemente la più tecnica, è quella che serve di più
subito dopo. Le **pose** delle fotocamere (dove stava e da che parte guardava
ognuna, che è quello che una ricostruzione *structure from motion* calcola
insieme ai punti) sono l'ingresso obbligatorio dei metodi di rendering
neurale della sezione seguente, quelli che ricostruiscono una scena
addestrando una piccola rete a rispondere «da qui, guardando di là, che colore
si vede?». Chi ha provato a costruirne uno da un video e ha ottenuto una nuvola
confusa, nove volte su dieci non ha sbagliato la rete: ha sbagliato le pose.

## Profondità da una sola immagine

Restava un fatto: da un'immagine sola la profondità è **matematicamente**
indeterminata. Eppure noi la vediamo, guardando una fotografia con un occhio
solo, e da qualche anno la vedono anche le reti. Conviene capire perché non è
una contraddizione.

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
sconosciuta. La rete mette la scena in fila dal vicino al lontano, e dice senza
esitare che l'auto sta più indietro dell'albero; se siano a dieci metri o a
cento non lo dice, e nemmeno di quante volte l'una sia più lontana dell'altro.
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

Il quadro che ne esce è questo: la geometria dà **vincoli esatti ma
insufficienti**, l'apprendimento dà **un prior sufficiente ma fallibile**, e i
sistemi che funzionano davvero usano entrambi. Uno stereo appreso mantiene la
ricerca lungo la retta epipolare e impara solo la parte ambigua; una pipeline
di ricostruzione usa la profondità monoculare per inizializzare e la geometria
multi-vista per correggerla.

`````

## In pratica: la geometria si può verificare

Tutto quello che si è detto fin qui si può controllare con
un calcolo, senza addestrare niente. Costruiamo una scena finta di cui
conosciamo le risposte, proiettiamola in due fotocamere e verifichiamo le due
affermazioni centrali. La prima è quella del dito davanti al naso: se misuro di
quanto un punto **salta** fra la prima e la seconda immagine, posso risalire a
quanto è lontano. La seconda è quella della retta: il punto che nella prima
immagine sta in un certo posto, nella seconda deve cadere su una riga precisa,
calcolabile in anticipo. Se sono vere davvero, il computer deve ridarci i
numeri esatti da cui siamo partiti.

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

Le due colonne coincidono, e l'errore massimo è dell'ordine di $10^{-15}$: cioè
zero, a meno degli arrotondamenti che il computer fa quando scrive un numero
con la virgola. La regola dice che la profondità si ottiene moltiplicando la
**focale** (quanto la fotocamera ingrandisce, qui $700$) per la **base** (la
distanza fra le due fotocamere, qui $30$ centimetri) e dividendo per il salto
misurato: $Z = fB/d$. Non è un'approssimazione che funziona più o meno bene: è
un'identità, e o si applica alla lettera o non si applica affatto. E `righe
uguali` conferma l'altra cosa che il testo aveva promesso: con le due
fotocamere affiancate e allineate, il gemello di un pixel sta sulla **stessa
riga** dell'altra immagine, così che a cercarlo basta scorrere quella.

Ora il caso generale, con la seconda fotocamera ruotata di otto gradi attorno
alla verticale, come due telecamere puntate un po’ l'una verso l'altra. Qui non
c'è più nessuna riga comoda, ma la retta esiste ancora: il calcolo qui sotto la
scrive a partire da come sono messe le due fotocamere (è la matrice `F`), e poi
verifica, punto per punto, di quanto il pixel della seconda immagine cade
fuori da quella retta. Quello scarto si chiama **residuo**.

```python
ang = np.deg2rad(8.0)
Ry = np.array([[ np.cos(ang), 0, np.sin(ang)],   # rotazione attorno all'asse y
               [ 0,           1, 0          ],   # (la verticale): la seconda riga
               [-np.sin(ang), 0, np.cos(ang)]])  # e colonna restano identiche
t3 = np.array([-B, 0.02, 0.0])
u3 = proietta(K, Ry, t3, P)

def antisimmetrica(v):
    """La matrice che realizza il prodotto vettoriale: [v]_x @ w == np.cross(v, w)."""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

F = np.linalg.inv(K).T @ antisimmetrica(t3) @ Ry @ np.linalg.inv(K)

def omogenee(u):
    return np.column_stack([u, np.ones(len(u))])

residuo = np.einsum('ij,jk,ik->i', omogenee(u3), F, omogenee(u1))
print("residuo epipolare :", np.abs(residuo).max())
```

Il residuo massimo è dell'ordine di $10^{-17}$, cioè ancora una volta zero. Se
valesse $0{,}3$ vorrebbe dire che il pixel cade a fianco della riga prevista, e
qui invece per ognuno degli otto punti il pixel nella seconda immagine sta
**esattamente** sulla retta calcolata dalla prima. Nessuna rete, nessun
dato: è un'identità algebrica che dipende solo da come è fatta la proiezione,
ed è la ragione per cui questa parte della visione artificiale non è
invecchiata.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- **Scattare una foto butta via la distanza.** Un pixel dice da che parte
  guardare, non quanto lontano: la mano vicina e la Torre di Pisa lontana, se
  sono allineate con l'obiettivo, finiscono nello stesso punto della foto. Non
  è un'informazione nascosta da scovare, è un'informazione cancellata da
  ricostruire.
- Prima di qualunque conto la fotocamera va **misurata** (quanto ingrandisce,
  dove cade il centro dell'immagine, quanto l'obiettivo incurva le rette), e
  si fa fotografando una scacchiera. Senza quelle misure nessuna risposta può
  essere in metri.
- Con due foto la ricerca del punto corrispondente non è una caccia in tutta
  l'immagine: **è una caccia lungo una riga**, e quale riga si calcola in
  anticipo dalla posizione reciproca delle due fotocamere. Se le si monta
  affiancate e allineate, quella riga è la stessa riga di pixel.
- La distanza si legge nel **salto** di un punto fra le due immagini, ed è una
  relazione inversa: raddoppiando la distanza il salto si dimezza. Perciò la
  visione a due occhi è precisa da vicino e vaga da lontano, e allontanare le
  telecamere aumenta la precisione ma restringe la zona che entrambe vedono.
- Lo stesso ragionamento vale nel tempo invece che nello spazio (guardare un
  palo dal finestrino di un treno), con due limiti da tenere a mente: su una
  superficie senza disegni non c'è niente da riconoscere, e guardando solo un
  pezzo di bordo non si vede il movimento lungo il bordo stesso.
- Con molte foto si stima tutto insieme, **dov'erano le fotocamere e dov'erano
  i punti**, aggiustando finché ogni punto non cade dove il modello dice che
  dovrebbe cadere. Una cosa resta comunque indeterminata: quanto è grande la
  scena, cioè se sia un palazzo o un plastico.
- Da una foto sola la distanza è indeterminata per legge, eppure le reti la
  stimano bene: non la calcolano, la **riconoscono**, perché hanno visto
  milioni di scene. Ottimo in pratica, ingannabile per costruzione (una
  fotografia di una fotografia le inganna), e cieco alla scala.
```

`````

`````{tab} Superiore

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
  Rettificando le immagini quelle rette diventano orizzontali, il che equivale
  a mandare gli epipoli all'infinito. Dentro RANSAC, il minimo di
  corrispondenze è sette per $\mathbf{F}$ e cinque per $\mathbf{E}$, non otto.
- Nel caso rettificato, $Z = fB/d$: la profondità è **inversamente**
  proporzionale alla disparità, quindi la stereo è precisa da vicino e vaga da
  lontano, e l'errore cresce come $Z^2$.
- Il **flusso ottico** è lo stesso problema nel tempo, indifferente a chi si
  muova fra scena e fotocamera; la sua equazione ha due
  incognite per pixel (problema dell'apertura), e si chiude con un'ipotesi
  locale (Lucas-Kanade) o globale (Horn-Schunck).
- **Structure from motion** stima insieme pose e punti minimizzando l'errore
  di riproiezione (*bundle adjustment*); con intrinseci noti la scala globale
  resta indeterminata, senza calibrazione l'ambiguità è l'intero gruppo
  proiettivo. Le sue pose sono l'ingresso obbligatorio del rendering neurale.
- La profondità **da una sola immagine** è matematicamente indeterminata: le
  reti non la calcolano, applicano un prior appreso. Ottimo in pratica,
  ingannabile per costruzione, e cieco alla scala assoluta.
```

`````
