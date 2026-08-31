# Apprendimento supervisionato: regressione e classificazione

Immagina di affiancare per una settimana un agente immobiliare esperto. Non ti
spiega nessuna formula: ti mostra centinaia di case già vendute (metri quadri,
numero di stanze, quartiere) e accanto a ciascuna il prezzo finale. Dopo un
po’, davanti a un appartamento mai visto, sai già sparare una cifra
ragionevole. Hai imparato **dagli esempi etichettati**. È, in una frase, ciò
che fa l'apprendimento supervisionato: mostragli abbastanza coppie
*domanda–risposta* e imparerà a rispondere da solo.

## Imparare una funzione dagli esempi

I dati di partenza sono quasi sempre una **tabella**: una riga per esempio (un
appartamento, un'email, un paziente) e una colonna per caratteristica. Ma non
tutte le colonne sono fatte della stessa pasta, e la differenza sta in che cosa
ha senso farci sopra:

- una colonna **numerica** contiene numeri veri, su cui somme, differenze e
  medie hanno un senso: i metri quadri di una casa, la sua età;
- una colonna **categorica** contiene nomi, senza nessun ordine: il quartiere,
  il colore, la marca. Milano e Roma non sono uno più dell'altro, e la loro
  media non esiste;
- una colonna **ordinale** sta in mezzo: i valori sono in fila (classe
  energetica A, B, C; «lieve», «moderato», «grave») ma non sappiamo di quanto
  disti uno dall'altro.

```{figure} ../figures/feature-label-tipi-dati-ml.svg
:name: fig-tipi-di-feature
:alt: "I tre tipi di feature affiancati con un esempio ciascuno: numerica, come i metri quadri, su cui hanno senso somme e differenze; categorica, come il quartiere, dove i valori sono nomi senza ordine; ordinale, come la classe energetica, dove esiste un ordine ma non una distanza."
:width: 96%

Le tre colonne appena elencate, una accanto all'altra, con sotto le operazioni
che ciascuna ammette. Confonderle è
l'errore che porta un modello a calcolare la media fra «Milano» e «Roma».
```

Decidere di quale dei tre tipi è ciascuna colonna, come in
{numref}`fig-tipi-di-feature`, è la prima decisione di ogni progetto, e non la
prende il modello: la
prende chi prepara i dati. Se al quartiere «Milano» assegniamo il numero 1 e a
«Roma» il 2 per poterli dare in pasto a un programma (si dice **codificare**
una colonna), quella colonna per il modello è numerica a tutti gli effetti: ci
farà sopra medie e differenze, e crederà che Roma sia il doppio di Milano e che
fra le due ci sia qualcosa a 1,5. È un ordine, e sono delle distanze, che
nessuno intendeva metterci.

Di ogni appartamento teniamo tre numeri in fila (metri quadri, stanze, piano):
un elenco ordinato di numeri si chiama **vettore**, ed è lo stesso oggetto della
{doc}`sezione sull'algebra lineare </Matematica/algebra-lineare>`. Lo scriviamo
$\mathbf{x}$, in grassetto minuscolo, proprio per ricordare che non è un numero
solo. A ciascun appartamento associamo poi un'etichetta $y$ (il prezzo). Il
"supervisore" è proprio quella $y$ nota: qualcuno, in passato, ha già
registrato la risposta giusta.

```{admonition} Una colonna è una direzione, un esempio è un punto
:class: tip


Prendi una tabella con due sole colonne: metri quadri e prezzo. Puoi disegnarla
su un foglio a quadretti, con i metri quadri sull'asse orizzontale e il prezzo
su quello verticale: ogni appartamento diventa **un punto**, e la tabella
diventa una nuvola di punti. Con tre colonne servirebbe una scatola invece di
un foglio, e i punti starebbero sospesi in aria. Con quattro colonne non
riusciamo più a disegnarla, e tuttavia i conti si fanno lo stesso, identici a
prima: si continua a parlare di punti, di distanze fra punti, di rette che li
separano.

Quindi: **ogni colonna della tabella è una direzione dello spazio, ogni riga è
un punto in quello spazio.** Una tabella con cento colonne descrive punti in
uno spazio a cento dimensioni, e «dimensione» vuol dire esattamente questo,
niente di più misterioso. Ridurre le dimensioni vorrà dire togliere direzioni;
«spazio delle caratteristiche» sarà il nome di quello spazio lì; e frasi come
«due esempi vicini» vorranno dire «due punti vicini», cioè due appartamenti
simili in tutte le colonne insieme.
```

`````{tab} Elementare

Abbiamo tante coppie *(descrizione, risposta)*: la descrizione è la nostra
$\mathbf{x}$, la risposta è la $y$. L'obiettivo è trovare una **regola** che,
data una nuova descrizione, indovini la risposta. Chiamiamo questa regola $f$:

$$
\hat{y} = f(\mathbf{x})
$$

Si legge così: dài la descrizione $\mathbf{x}$ alla regola $f$, e lei ti
restituisce una risposta. È la stessa scrittura dei tasti di una calcolatrice
(dài un numero a «radice quadrata» e ottieni un risultato), solo che qui quello
che entra è un elenco di numeri e la regola è tutta da trovare. Il cappello su
$\hat{y}$ ricorda che è una *previsione*, non la verità: è la migliore ipotesi
del modello.

Per dire quanto vale una regola servono due conti, e non sono lo stesso conto.
Il primo guarda un esempio alla volta e dice quanto quella singola risposta è
sbagliata. Il secondo mette insieme tutti i primi e ne fa la media, come il
voto di un compito che nasce dai punteggi delle sue domande. Con dieci esempi
ci sono dieci errori singoli e un solo numero riassuntivo, con diecimila
esempi diecimila e uno. I due servono in momenti diversi: l'errore singolo
dice dove la regola sta sbagliando, la media dice se nel suo insieme sta
migliorando. E capiterà, più avanti, di calcolare quella media su una manciata
di esempi per volta invece che su tutti. Imparare significa scegliere la $f$
che rende quella media più piccola che si può sugli esempi già noti, sperando
che se la cavi bene anche su quelli nuovi.

`````

`````{tab} Superiore

Partiamo da un insieme di addestramento di $m$ esempi etichettati,

$$
\mathcal{D} = \{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^{m},
\qquad \mathbf{x}^{(i)}\in\mathbb{R}^n,
$$

e cerchiamo una funzione $f:\mathcal{X}\to\mathcal{Y}$ che approssimi la
relazione ignota tra ingressi e uscite, con $\hat{y}=f(\mathbf{x})$. La qualità
di $f$ si misura con una **funzione di costo** (o *loss*), e i due oggetti che
portano quel nome vanno tenuti distinti: $\ell$ è il costo di **una**
predizione, $\mathcal{L}$ è quello sull'intero insieme, cioè la media dei
primi. L'addestramento è il problema di ottimizzazione

$$
\theta^\star = \arg\min_{\theta}\ \mathcal{L}(\theta),
\qquad
\mathcal{L}(\theta) = \frac{1}{m}\sum_{i=1}^{m}
\ell\big(f_\theta(\mathbf{x}^{(i)}),\, y^{(i)}\big),
$$

dove $\theta$ sono i parametri del modello. La distinzione fra $\ell$ e
$\mathcal{L}$ tornerà utile più avanti, quando il gradiente si calcolerà su un
sottoinsieme di esempi invece che su tutti. La natura di $\mathcal{Y}$
distingue i due problemi cardine: continuo per la regressione, discreto per la
classificazione.

`````

## Due domande, due problemi

Ciò che cambia tutto è il *tipo* di risposta. "Quanto costa questa casa?"
chiede un numero su una scala continua: è **regressione**. "Questa email è
spam, sì o no?" chiede un'etichetta da un insieme finito: è
**classificazione**. Stesso impianto, imparare $f$ da coppie
$(\mathbf{x}, y)$, due
geometrie diverse, come mostra {numref}`fig-regr-classif`: a sinistra
cerchiamo una linea che *segua* i punti, a destra una linea che li *separi*.

```{figure} ../figures/regressione-vs-classificazione.svg
:name: fig-regr-classif
:alt: Due pannelli affiancati. A sinistra, uno scatter di punti attraversato da una retta di regressione che ne segue l'andamento crescente. A destra, due nuvole di punti di colore diverso separate da una retta tratteggiata che funge da confine di decisione.
:width: 95%

Due volti dello stesso problema. Nella regressione (sinistra) la retta
*approssima* i dati; nella classificazione (destra) la retta *separa* le classi.
```

## La regressione lineare: la retta di best fit

Il modello più semplice, e sorprendentemente utile, ipotizza che ogni
caratteristica spinga la risposta in proporzione: raddoppia i metri quadri e,
grosso modo, il prezzo raddoppia; una stanza in più vale sempre lo stesso
tanto, che sia la seconda o la quinta.

Il conto allora è di quelli che si fanno a mano. A ogni caratteristica si
attacca un numero che dice quanto quella caratteristica conta: si chiama
**peso**, e non ha niente a che vedere con i chili. Poi si moltiplica ogni
caratteristica per il suo peso, si sommano i risultati, e la somma è la
risposta. Nient'altro: niente potenze, niente caratteristiche moltiplicate fra
loro. Una risposta ottenuta così, moltiplicando e sommando e basta, in gergo si
chiama **combinazione lineare** delle caratteristiche, e la parola «lineare»
tornerà spessissimo con questo significato.

`````{tab} Elementare

Guardando i soli metri quadri, l'agente se la cava con una regola sola:
duemila euro al metro quadro più cinquantamila di partenza. Un appartamento di
$80$ m² viene $2\,000 \cdot 80 + 50\,000 = 210\,000$ €. Con le lettere, quella
regola è una **retta**:

$$
\hat{y} = w\,x + b
$$

dove $w$ è la pendenza (quanto sale il prezzo per ogni metro quadro in più) e
$b$ il punto di partenza. Di rette ce ne sono infinite, e vogliamo quella che
passa più in mezzo alle case già vendute, la *retta di best fit*
(l'espressione inglese vuol dire «che si adatta meglio», e in italiano si dice
anche retta di regressione).

Se questa valga lo dice l'archivio. Gli $80$ m² sono andati proprio a
$210\,000$ €, scarto zero; un $60$ m² è stato venduto a $160\,000$ € e la
regola ne chiede $170\,000$, diecimila di troppo; un $100$ m² è andato a
$260\,000$ € e la regola ne chiede $250\,000$, diecimila in meno. Sommati come
stanno, il $+10\,000$ e il $-10\,000$ si cancellano, il totale viene zero e la
regola sembra perfetta dopo aver mancato due case su tre. Allora gli scarti si
elevano al quadrato, che li rende tutti positivi, e se ne fa la media: in
migliaia di euro sono $0$, $+10$ e $-10$, al quadrato $0$, $100$ e $100$, e la
media è $(0 + 100 + 100)/3 \approx 67$. Avendo elevato al quadrato, quel $67$
non è in euro ma in migliaia di euro al quadrato, e serve solo a dire quale
retta batte quale, e più è piccolo, migliore è la retta. Il quadrato fa anche
un secondo mestiere, voluto: uno scarto doppio pesa quattro volte tanto ($10$
al quadrato fa $100$, $20$ ne fa $400$), e quel conto preferisce la retta che
sbaglia poco su molte case a quella che le azzecca quasi tutte e prende un
abbaglio su una.

Provarle tutte non si può, e allora se ne prende una qualsiasi e la si
aggiusta a piccoli passi, come chi scende da un fianco di collina nella
nebbia: il fondovalle non si vede, ma il piede sente da che parte cala il
terreno, si fa un passo di là, poi un altro da lì, finché non c'è più discesa.
La collina è l'errore, la coppia $(w, b)$ è la posizione sul fianco, e
l'altezza del terreno è quanto quella retta sbaglia sulle case dell'archivio.
Camminare vuol dire cambiare $w$ e $b$, scendere vuol dire sbagliare meno. Il
piede, qui, si risparmia: l'errore è scritto in una formula, e da una formula
la pendenza si calcola stando fermi, come l'inclinazione di una rampa dalle
sue misure. La direzione di massima discesa si chiama **gradiente**, e la
camminata che ripete il passo **discesa del gradiente**. La lunghezza del
passo la decidiamo noi, e cambia tutto, perché una corta ci mette un'eternità
e una lunga scavalca il fondovalle e rimbalza da un fianco all'altro. Si
chiama **learning rate** (il *tasso di apprendimento*), e a come si sceglie è
dedicata una sezione intera.

Nella nebbia ci si ferma nella prima conca, senza sapere mai che dietro il
crinale ce n'era una più profonda. Con la media degli scarti al quadrato non
capita, perché quella collina ha la forma di una scodella, con un fondo solo,
e da qualunque retta si parta si finisce lì. La garanzia riguarda la retta e
non ogni modello, e le reti neurali camminano su terreni
molto più accidentati. Per una retta, poi, si può anche non camminare, perché
un conto diretto dà i pesi migliori in un colpo solo, e finché i dati stanno
comodi si usa quello. Con milioni di case e centinaia di colonne quel conto
costa più della passeggiata, e si torna a scendere a piccoli passi.

Un guasto, però, la nebbia non lo spiega. In archivio la superficie è finita
due volte, in metri quadri da una fonte e in centimetri quadrati da un'altra,
e il fondo della scodella si allunga in un fondovalle piatto. Duemila euro al
metro quadro e niente all'altra colonna, oppure mille euro al metro quadro e
dieci centesimi al centimetro quadrato, su ogni casa danno lo stesso identico
prezzo (su una casa di $80$ m², $160\,000$ € in tutti e due i modi). Chi
cammina si ferma dove capita lungo quel fondo, due colleghi partiti da punti
diversi arrivano a due regole diverse, e il conto diretto si inceppa, perché
gli chiediamo il punto più basso e di punti più bassi ce n'è una fila intera.
I prezzi restano buoni, ma i pesi non si possono più leggere, e la frase «i
metri quadri contano duemila euro» perde senso. Stessa storia quando le
colonne sono più delle case in archivio, con tante manopole e poche vendite da
rispettare.

Quella passeggiata a piccoli passi è il motore di quasi tutto l'apprendimento
automatico, reti neurali comprese, ed è la ragione per cui la parola
«errore» torna a ogni passaggio: non dà solo un voto alla regola, le dice da
che parte andare.

`````

`````{tab} Superiore

Con $n$ caratteristiche il modello diventa un prodotto scalare più un bias:

$$
\hat{y} = \mathbf{w}^\top \mathbf{x} + b .
$$

I parametri $\mathbf{w}\in\mathbb{R}^n$ e $b\in\mathbb{R}$ si stimano
minimizzando l’**errore quadratico medio** (*Mean Squared Error*):

$$
\mathcal{L}(\mathbf{w}, b)
= \frac{1}{m}\sum_{i=1}^{m}\big(\hat{y}^{(i)} - y^{(i)}\big)^2
= \frac{1}{m}\sum_{i=1}^{m}
\big(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\big)^2 .
$$

$\mathcal{L}$ è convessa in $(\mathbf{w},b)$: niente minimi locali in cui restare
intrappolati. Se le colonne della matrice dei dati, insieme alla colonna
costante del bias, sono linearmente indipendenti, il minimo è anche unico e si
raggiunge in forma chiusa con le equazioni normali; con feature collineari (una
feature costante basta, perché replica la colonna del bias), o con meno esempi
che feature, i punti di minimo diventano infiniti (un intero sottospazio, tutti
con lo stesso valore della loss) e le equazioni normali degenerano. Su grandi
dataset, in ogni caso, si preferisce la discesa del gradiente. Elevare al
quadrato penalizza fortemente gli errori grossi e rende la loss differenziabile
ovunque: due proprietà che tornano comode.

`````

## La regressione logistica: dal numero alla probabilità

Per la classificazione la retta da sola non basta. Cominciamo col mettere in
numeri la risposta: siccome un modello lavora solo su numeri, decidiamo per
convenzione che «no» si scrive $0$ e «sì» si scrive $1$ (è una nostra scelta di
scrittura, non una proprietà del mondo, e nulla cambierebbe scambiandole).
Fatto questo, la differenza salta all'occhio: un prezzo può valere
$310\,000$, mentre «spam sì/no» sta solo fra $0$ e $1$. La **regressione
logistica** (che, malgrado il nome, classifica: il nome le è rimasto addosso
perché il conto che fa dentro è ancora quello della regressione) risolve il
problema in due mosse.

`````{tab} Elementare

**Prima mossa: un punteggio.** Si moltiplica ogni caratteristica per il suo
peso, si somma tutto e si aggiunge il numero di partenza, esattamente come per
la retta di prima. Ne esce un numero solo, che può essere qualsiasi cosa,
$-7$ o $+412$.

**Seconda mossa: schiacciarlo.** Quel numero passa dentro una funzione a forma
di «S», la **sigmoide**, che qualunque cosa le si dia restituisce un valore
compreso fra $0$ e $1$. Un punteggio molto positivo esce vicino a $1$ («quasi
certo spam»), uno molto negativo vicino a $0$, e lo zero cade esattamente a
metà, $0{,}5$.

Quel numero fra zero e uno lo leggiamo come una **probabilità**: non la
probabilità del lancio di un dado, ma la sicurezza del modello. $0{,}9$ vuol
dire «ci scommetterei»; $0{,}52$ vuol dire «non ne ho idea, ma se proprio devo
dico sì».

Restano da trovare i pesi, e si cerca ancora il punto più basso di una
collina, ma l'altezza del terreno adesso si misura in un altro modo: conta
quanto il modello è sicuro della risposta sbagliata. Dare $0{,}6$ a un'email
che spam non era costa poco, darle $0{,}99$ costa cinque volte tanto, e chi
dicesse «sicuro al cento per cento» sbagliando pagherebbe un prezzo senza
fondo. Il cambio serve anche a chi cammina nella nebbia. Misurare l'errore come
si faceva per la retta, con la media degli scarti al quadrato, farebbe prendere
a questa collina gobbe e ripiani, e la discesa si fermerebbe dove il terreno è
piatto per caso; con la sicurezza sbagliata la scodella dal fondo solo torna, e
si arriva sempre in fondo.

E la risposta secca, quando serve? La si ottiene con una terza mossa che
facciamo noi, non il modello: si fissa un valore di taglio, per abitudine
$0{,}5$, e si risponde «sì» sopra e «no» sotto. Dove mettere quel taglio è una
decisione con conseguenze, meno innocente di quanto sembri.

`````

`````{tab} Superiore

Sia $z = \mathbf{w}^\top \mathbf{x} + b$ il punteggio lineare. La sigmoide (o
logistica) è

$$
\sigma(z) = \frac{1}{1 + e^{-z}} \in (0, 1),
$$

e interpretiamo $\hat{y} = \sigma(z)$ come $P(y=1 \mid \mathbf{x})$. La
previsione di classe si ottiene con una soglia a $0{,}5$, che equivale a
$z = 0$: l'insieme

$$
\{\mathbf{x} : \mathbf{w}^\top \mathbf{x} + b = 0\}
$$

è il **confine di decisione**, un iperpiano che divide lo spazio in due regioni.
I parametri si stimano minimizzando la *cross-entropy* invece dell'MSE, perché
si accorda con l'interpretazione probabilistica e mantiene la loss convessa.

`````

```{figure} ../figures/regressione-logistica.svg
:name: fig-sigmoide-soglia
:alt: "La curva sigmoide che sale da zero a uno, con una linea orizzontale tratteggiata alla soglia di 0,5. I punti che cadono sotto la soglia sono assegnati alla classe 0, quelli sopra alla classe 1; vicino alla soglia la curva è ripida, agli estremi si appiattisce."
:width: 84%

Dalla retta alla probabilità. La sigmoide non decide: produce un numero fra
zero e uno, e la decisione arriva dopo, quando si sceglie dove tagliare.
```

I due gesti che {numref}`fig-sigmoide-soglia` tiene separati (produrre un
numero, e poi decidere) contano più di quanto sembri, e torneranno nella
sezione sulle metriche. Il punto in cui si taglia si chiama **soglia**, e
quel $0{,}5$ è una convenzione, non un risultato: possiamo spostarlo.
Abbassandolo si segnalano più email come spam, quindi meno spam passa ma più
messaggi buoni finiscono nel cestino; alzandolo succede l'opposto. Nessuno dei
due errori sparisce, si scambiano l'uno con l'altro. La cosa notevole è che
per farlo non serve riaddestrare niente: il modello resta quello, cambia solo
dove mettiamo il taglio.

C'è poi una cosa da sapere prima di scrivere la prima riga di codice, perché è
una piccola sorpresa.

`````{tab} Elementare

Quando in scikit-learn si scrive `LogisticRegression()` e basta, non si ottiene
la regressione logistica «pura», quella fatta di punteggio, schiacciamento e
nient'altro. La libreria ci aggiunge di
suo un **freno**, cioè quel prezzo alla complessità che vedremo nella prossima
sezione: senza dire niente, tiene i pesi più piccoli di quanto sarebbero.

È quasi sempre un bene. Con dati così
facili che una linea li separa alla perfezione, il modello, per prendere pieni
voti, non deve solo azzeccare le risposte, deve anche essere *sicuro*, e per
essere più sicuro gli basta ingigantire i pesi. Un punteggio di $10$ dà una
probabilità del $99{,}99\%$, uno di $100$ ne dà una ancora più vicina a $1$: non
c'è mai un motivo per fermarsi, e senza freno i pesi crescono all'infinito. Il
freno è ciò che dice «basta così».

Il freno si vede nei numeri. Su quattro punti messi in modo che una linea li
separi senza incertezze, il peso che esce con le impostazioni di fabbrica vale
circa $1$, e chiedendo di togliere il freno diventa quasi nove volte tanto,
mentre il confine fra le due classi resta esattamente dov'era. La manopola si
chiama `C`, e ha un verso che confonde, perché il numero da scrivere dice
quanto freno si toglie: più è grande, meno freno c'è. È comunque il tipo di
dettaglio che va saputo, perché il modello che gira non è quello della
definizione, e chi
confronta quei due pesi senza saperlo pensa di aver sbagliato i conti.

`````

`````{tab} Superiore

Quello che gira davvero quando si scrive `LogisticRegression()` non è la stima
di massima verosimiglianza: scikit-learn aggiunge di suo una
penalità $\ell_2$ sui
pesi, di intensità `C=1.0` (in quella parametrizzazione $C$ è l’*inverso* della
forza del freno, come nelle SVM). Il modello che esce, quindi, minimizza la
cross-entropy **più** quella penalità, e la differenza non è cosmetica: sui
quattro punti $x = -2, -1, 1, 2$ con etichette $0, 0, 1, 1$ (una dimensione,
linearmente separabili) il coefficiente
stimato vale $1{,}01$ con i default e $8{,}85$ chiedendo `C=np.inf`. Il secondo
non è il numero «giusto»: sotto separazione perfetta il massimo di
verosimiglianza
**non esiste**, i pesi vorrebbero andare all'infinito, e ciò che li ferma è
proprio il freno. Chi vuole la stima non regolarizzata deve chiederla
sapendo che cosa sta chiedendo.

`````

## Lineare, logistica, Poisson: una famiglia sola

I due modelli visti finora rispondono a due domande diverse: la retta dice
*quanto* (un prezzo, una temperatura, un numero qualsiasi), la logistica dice
*sì o no*. Ce n'è una terza, altrettanto comune, alla quale nessuna delle due
risponde: quante volte. Quanti clienti entrano in farmacia fra le nove e le
dieci, quanti guasti registra una linea in un turno, quante volte un utente
apre l'applicazione in una settimana. Sono conteggi: numeri interi, mai
negativi, e con una particolarità che la retta non sa gestire.

Provare comunque con la retta è istruttivo, perché fallisce in due modi
visibili. Su duemila giornate (inventate al calcolatore, così che la risposta
giusta si conosca in anticipo e si possa controllare se il modello la ritrova)
la retta migliore prevede un numero negativo di clienti in
duecentoquarantasette casi, il che è una risposta che nessuno può usare; e i
suoi scarti si aprono a ventaglio, piccoli dove i clienti sono pochi e grandi
dove sono tanti, mentre tutto l'impianto della retta di prima suppone che siano
più o meno gli stessi dappertutto.

La riparazione non butta via niente di quello che si è imparato: tiene il
punteggio lineare, cambia il modo di leggerlo, e cambia la regola con cui si
misura lo scarto. Fatto per la terza volta, il gesto si riconosce come uno
solo, e il suo nome è **modelli lineari generalizzati**, che gliel'hanno dato
John Nelder e Robert Wedderburn nel 1972 {cite}`nelder1972generalized`.

`````{tab} Elementare

Il farmacista vuole sapere quanti clienti aspettarsi alle nove del mattino, e
ha le colonne di sempre: che giorno è, che tempo fa, se c'è una promozione in
corso, quanto siamo dentro la stagione dell'influenza. La prima mossa è quella
già vista: si moltiplica ogni colonna per il suo peso, si somma tutto, e ne
esce un punteggio, che come sempre può valere qualsiasi cosa, meno sette o più
quattrocento.

La seconda mossa è dove le tre risposte si separano. Per un prezzo il punteggio
si legge così com'è. Per un sì o no lo si schiaccia fra zero e uno con la curva
a esse. Per un conteggio si fa una terza cosa: si prende quel punteggio come
**logaritmo** del numero di clienti, e per tornare al numero si fa il conto
all'incontrario, cioè si eleva a quel punteggio il numero $e$ (che vale circa
$2{,}718$, e che la {doc}`sezione su analisi e ottimizzazione
</Matematica/analisi-ottimizzazione>` racconta insieme al logaritmo). Il
vantaggio è immediato: un numero positivo elevato a qualunque cosa resta
positivo, anche a un esponente negativo, quindi la previsione non può più
essere meno tre clienti.

Il prezzo di questa scelta va detto subito, perché cambia il senso dei pesi.
Nel punteggio i pesi si sommano, come sempre; ma disfare un logaritmo
trasforma le somme in prodotti, e quindi sui clienti quei pesi
**moltiplicano**. Un peso che vale un mezzo non aggiunge mezzo cliente:
moltiplica per la radice quadrata di $2{,}718$, cioè per $1{,}65$. Il sabato
non aggiunge dodici clienti: il sabato raddoppia.

C'è anche una cosa da sistemare prima di cominciare, e riguarda il tempo. Un
conteggio non vuol dire niente se non si sa su quanto tempo è stato fatto:
dodici clienti in un'ora e dodici in una giornata sono due fatti diversi. Se le
finestre non sono tutte uguali, la durata entra nel conto come un ingrediente
con il peso già deciso e non da imparare, e quello che il modello stima diventa
il ritmo, cioè i clienti per ora.

Resta il terzo pezzo, quello sull'irregolarità, e cambia la terza cosa: la
regola con cui si misura lo scarto. Un conteggio non si sparpaglia come un
prezzo. Se in media entrano due clienti l'ora, i giorni oscillano fra zero e
cinque; se ne entrano cento, oscillano fra ottanta e centoventi. In proporzione
l'oscillazione si stringe, e in valore assoluto cresce come la **radice** della
media: da due a cento la media si moltiplica per cinquanta e l'oscillazione per
poco più di sette, che è la radice di cinquanta. È la regola dei conteggi che
capitano ciascuno per conto proprio, quella che porta il nome di Poisson, e
prendere quella invece della curva a campana chiude la terza differenza.

Le tre risposte, allora, sono la stessa macchina con tre impostazioni: il
punteggio si costruisce sempre allo stesso modo, e cambiano solo come lo si
legge e con che regola si misura lo scarto. Cambia poco anche il modo di
imparare: si guarda la differenza fra quello che è successo e quello che il
modello si aspettava, e si spingono i pesi in quella direzione. Uguale per
tutte e tre.

E i due modi di sbagliare. Il primo: leggere il punteggio come un logaritmo è
una scelta di chi costruisce il modello, e dice che gli effetti si
moltiplicano. Se nel negozio in questione il sabato aggiunge davvero dodici
clienti invece di raddoppiarli, quella lettura è la lettura sbagliata, e i pesi
che ne escono raccontano una storia che non c'è. Il secondo morde più spesso: i
conteggi veri sono quasi sempre più irregolari di quanto la regola prometta.
Basta che i clienti arrivino a gruppetti, perché scende un autobus o perché è
finita la messa, e l'oscillazione diventa il doppio di quella prevista. I pesi
non se ne vanno lontano dal vero, ma la fiducia che il modello dichiara sì, ed
è troppa: chi la prende per buona prepara i turni su un'oscillazione che non
esiste. La cura è una versione con una manopola in più, che regola
l'irregolarità separatamente dalla media, e si chiama binomiale negativa.


`````

`````{tab} Superiore

Un **modello lineare generalizzato** (GLM) si specifica con tre pezzi
{cite}`nelder1972generalized`. Una **distribuzione** per $y \mid \mathbf{x}$
scelta nella famiglia esponenziale; un **predittore lineare** $\eta =
\mathbf{w}^\top\mathbf{x} + b$; e una **funzione di legame** $g$ che li unisce,
con $g(\mu) = \eta$ e $\mu = \mathbb{E}[y \mid \mathbf{x}]$.

La **famiglia esponenziale** in forma canonica raccoglie le densità

$$
p(y \mid \theta) = h(y)\,\exp\!\big(\theta\, y - A(\theta)\big),
$$

dove $h$ dipende dalla sola $y$ (la misura di base: vale $1$ per la Bernoulli,
$1/y!$ per la Poisson), $\theta$ è il **parametro naturale** (una lettera che
qui non indica i parametri del modello, che restano $\mathbf{w}$ e $b$) e
$A(\theta) = \log\int h(y)e^{\theta y}\,dy$ è la **log-partizione**, cioè il
logaritmo di ciò che serve a far tornare l'integrale a uno, e che dipende da
$\theta$. Da $A$ discende tutto, derivando sotto il segno di integrale (lecito
all'interno dell'insieme dei $\theta$ per cui quell'integrale converge):

$$
A'(\theta) = \mathbb{E}[y], \qquad A''(\theta) = \operatorname{Var}[y] .
$$

Tre casi bastano qui. La gaussiana a **varianza unitaria** ha $\theta = \mu$ e
$A(\theta) = \theta^2/2$, quindi media $\theta$ e varianza $1$; con una
varianza nota diversa da uno servono $\theta = \mu/\sigma^2$ e
$A(\theta) = \sigma^2\theta^2/2$, oppure la forma con un parametro di
dispersione a parte. La Bernoulli ha $\theta = \log\frac{\mu}{1-\mu}$ (il
*logit*) e $A(\theta) = \log(1+e^{\theta})$, quindi
$A'(\theta) = \sigma(\theta)$: la sigmoide, dunque, è la derivata della
log-partizione. La distribuzione di **Poisson**, che descrive un conteggio di
eventi indipendenti in una finestra fissa con $P(y=k) = e^{-\mu}\mu^k/k!$, ha
$\theta = \log\mu$ e $A(\theta) = e^{\theta}$, da cui
$A'(\theta) = A''(\theta) = \mu$: media e varianza coincidono, e quindi
l'ampiezza tipica dell'oscillazione va come $\sqrt{\mu}$.

Il **legame canonico** è quello che pone $\theta = \eta$, cioè $g = (A')^{-1}$:
identità per la gaussiana, logit per la Bernoulli, logaritmo per la Poisson.
Con quella scelta, e assumendo le $y_i$ indipendenti date le $\mathbf{x}_i$, la
log-verosimiglianza di $m$ osservazioni è

$$
\ell(\mathbf{w}) = \sum_{i=1}^{m}
\big(\eta_i\, y_i - A(\eta_i)\big) + \text{cost.},
\qquad
\nabla_{\mathbf{w}}\,\ell = \sum_{i=1}^{m}
\big(y_i - \mu_i\big)\,\mathbf{x}_i ,
$$

perché $A'(\eta_i) = \mu_i$. Il gradiente ha la stessa forma per tutti e tre i
modelli: residuo per feature, sommato sugli esempi. All'ottimo si annulla, cioè
i residui risultano ortogonali a ogni colonna, compresa la colonna di uno
dell'intercetta, da cui la loro somma nulla (senza intercetta quella condizione
non c'è). È la stessa condizione del primo ordine dei minimi quadrati vista
nella {doc}`sezione su ortogonalità e proiezioni
</Matematica/ortogonalita-proiezioni>`, ma non la stessa geometria, perché
$\mu_i = A'(\eta_i)$ non appartiene allo span delle colonne e non c'è nessun
teorema di Pitagora da invocare.

Due garanzie discendono da $A''>0$. La log-verosimiglianza è concava in
$\mathbf{w}$, quindi non ci sono ottimi locali e il massimo, **quando esiste**,
è unico a meno di colonne collineari: che esista non è garantito, ed è di nuovo
il caso della separazione perfetta appena visto per la logistica. E l'Hessiana
$-\sum_i A''(\eta_i)\mathbf{x}_i\mathbf{x}_i^\top$ si scrive come una matrice
di pesi, il che rende il passo di Newton una regressione ai minimi quadrati
pesata, rifatta a ogni iterazione: è l'algoritmo IRLS del lavoro del 1972, che
ancora oggi gira in `glm()` di R e in statsmodels, mentre scikit-learn
preferisce un ottimizzatore generico.

I punti di rottura sono tre. Il legame è un'ipotesi di modello e non un fatto:
con il logaritmo gli effetti sono moltiplicativi, e un fenomeno additivo va
modellato con il legame identità e distribuzione di Poisson, che è legittimo ma
non canonico. L’esposizione: se le finestre di osservazione hanno durate
diverse $t_i$, i conteggi non sono confrontabili, e si aggiunge un *offset*
$\log t_i$ al predittore lineare, cioè un termine con coefficiente fissato a
uno. E soprattutto la sovradispersione: la Poisson impone
$\operatorname{Var} = \mu$, e i conteggi reali quasi sempre oscillano di più,
perché gli eventi si raggruppano o perché resta eterogeneità non osservata.
Purché la media sia specificata bene (legame e predittore giusti) i
coefficienti restano consistenti, e a uscire sbagliati sono gli errori
standard, troppo piccoli di un fattore $\sqrt{\hat\phi}$, con $\hat\phi$ la
dispersione stimata. Il rimedio più diretto è quindi correggere quelli, per
quasi-verosimiglianza o con uno stimatore sandwich, senza toccare le stime; il
rimedio che cambia modello è la **binomiale negativa**, che aggiunge un
parametro di dispersione (e a dispersione libera non è più un GLM nel senso
appena definito). Torna come distribuzione di uscita di una rete nella
{doc}`sezione sul forecasting neurale </SerieTemporali/forecasting-neurale>`.

`````

I tre difetti della retta sui conteggi si vedono in un blocco solo, e con essi
il modo in cui il legame logaritmico li ripara.

```python
import numpy as np
from sklearn.linear_model import LinearRegression, PoissonRegressor

rng = np.random.default_rng(0)
n_giorni = 2000
# l'indice di stagione: zero d'estate, due al picco influenzale
stagione = rng.uniform(0, 2, n_giorni)
log_media = 0.4 + 1.5 * stagione         # gli effetti si moltiplicano
clienti = rng.poisson(np.exp(log_media))  # un conteggio, mai negativo
X = stagione.reshape(-1, 1)

# 1. la retta: prevede clienti negativi, e gli scarti si aprono a ventaglio
retta = LinearRegression().fit(X, clienti)
print(f"retta:  {retta.intercept_:.4f} + {retta.coef_[0]:.4f} * stagione")
negative = (retta.predict(X) < 0).sum()
print(f"        {negative} previsioni negative su {n_giorni}")
scarti = clienti - retta.predict(X)
for lo, hi in ((0.0, 0.5), (0.75, 1.25), (1.5, 2.0)):
    m = (stagione >= lo) & (stagione < hi)
    print(f"        stagione {lo}-{hi}: scarto tipico {scarti[m].std():.2f}")

# 2. lo stesso punteggio, letto come logaritmo della media
glm = PoissonRegressor(alpha=0.0, max_iter=10000, tol=1e-10).fit(X, clienti)
media = glm.predict(X)
print(f"legame log:  {glm.intercept_:.4f} + {glm.coef_[0]:.4f} * stagione")
print(f"        un punto di stagione moltiplica per"
      f" {np.exp(glm.coef_[0]):.4f}")
print(f"        scarti sommati, e per colonna: "
      f"{np.round([(clienti - media).sum(), stagione @ (clienti - media)], 3)}"
      f"  su {clienti.sum()} clienti")

# 3. il limite: conteggi piu' irregolari di quanto Poisson ammetta
dispersione = lambda y, mu: float((((y - mu) / np.sqrt(mu)) ** 2).mean())
print(f"dispersione sui dati di Poisson:  {dispersione(clienti, media):.4f}")

sovra = rng.negative_binomial(3, 3 / (3 + np.exp(log_media)))
g2 = PoissonRegressor(alpha=0.0, max_iter=10000, tol=1e-10).fit(X, sovra)
m2 = g2.predict(X)
d = dispersione(sovra, m2)
print(f"su conteggi sovradispersi:  {g2.intercept_:.4f}"
      f" + {g2.coef_[0]:.4f} * stagione")
print(f"        dispersione {d:.4f}, radice {np.sqrt(d):.4f}")
```

```text
retta:  -3.0126 + 12.5509 * stagione
        247 previsioni negative su 2000
        stagione 0.0-0.5: scarto tipico 2.06
        stagione 0.75-1.25: scarto tipico 2.57
        stagione 1.5-2.0: scarto tipico 5.92
legame log:  0.3784 + 1.5158 * stagione
        un punto di stagione moltiplica per 4.5531
        scarti sommati, e per colonna: [-0. -0.]  su 19022 clienti
dispersione sui dati di Poisson:  1.0224
su conteggi sovradispersi:  0.4086 + 1.4990 * stagione
        dispersione 3.8411, radice 1.9599
```

La retta prevede $-3{,}01$ clienti quando l'indice di stagione vale zero, cioè
d'estate, e finisce sotto zero in $247$ giornate su duemila; i suoi scarti
tipici passano da $2{,}06$ a $5{,}92$ attraversando l'intervallo, che è il
ventaglio annunciato. Il modello che legge il punteggio come logaritmo ritrova
invece i due numeri con cui i dati erano stati inventati, $0{,}4$ e $1{,}5$, e
il suo peso si legge come moltiplicatore: un punto in più di stagione
moltiplica i clienti per $4{,}55$. La riga sugli scarti è un controllo:
moltiplicandoli per ciascuna colonna e sommandoli si ottiene zero al millesimo
su quasi ventimila clienti, cioè quello che il modello non è riuscito a
spiegare non ha più niente in comune con le colonne che ha usato, ed è lo
stesso segno di «ho finito» della retta di prima.

Le ultime due righe sono il limite, e il conto è questo: si prende ogni scarto,
lo si divide per la radice del numero atteso, e si guarda quanto quei rapporti
si sparpagliano. Se l'oscillazione cresce davvero come la radice della media,
come Poisson promette, quel valore deve venire circa uno, e sui dati generati
da una Poisson viene $1{,}0224$. Rifacendo tutto su conteggi generati con la
versione a due manopole, la binomiale negativa, i pesi restano quelli
($0{,}4086$ e $1{,}499$) ma lo stesso valore sale a $3{,}84$. Il numero che
conta per chi deve decidere è la sua radice, $1{,}96$: le oscillazioni vere
sono quasi il doppio di quelle che il modello annuncia, e il farmacista che gli
crede prepara i turni per un sabato tranquillo che non arriverà.


## k-NN: chiedi ai vicini

Non tutti i modelli imparano dei parametri. Alcuni si limitano a *ricordare*, e
il capostipite è il **k-NN** (dall'inglese *k-nearest neighbors*, i $k$ vicini
più prossimi): tiene da parte tutti gli esempi che ha visto e, davanti a un
caso nuovo, va a cercare i $k$ che gli somigliano di più e li fa votare. Quel
$k$, cioè quanti vicini interpellare, è la scelta che decide tutto, e va fatta
prima di cominciare.

```{figure} ../figures/knn-classificare-per-somiglianza.svg
:name: fig-knn
:alt: "Un piano con punti di due classi già etichettati. Un punto nuovo, di classe ignota, è al centro di un cerchio che racchiude i suoi cinque vicini più prossimi: tre appartengono a una classe e due all'altra, e il punto nuovo riceve l'etichetta della maggioranza."
:width: 80%

Nessun addestramento, solo un conteggio. La classe del punto nuovo è quella
che vince fra i suoi $k$ vicini, e cambiare $k$ può cambiare il verdetto.
```

Il cerchio disegnato in {numref}`fig-knn` è tutta la scelta: allargandolo si
interpellano vicini via via più lontani, e la risposta diventa più stabile
(pochi voti strani non la ribaltano) ma anche più grossolana, perché smette di
accorgersi delle particolarità di quel pezzetto di quartiere. Con $k=1$ il
modello ripete pari pari il vicino più prossimo, **rumore** compreso. Conviene
fermarsi su questa parola, perché da qui in avanti torna in ogni sezione: il
rumore non ha niente a che fare con il suono, è tutto ciò che nei dati è
accidente invece che regola. L'errore di chi ha misurato, la casa venduta a
poco perché il proprietario aveva fretta, la giornata storta: cose che sono
successe davvero e che non si ripeteranno. Con $k$ pari al numero di esempi,
all'estremo opposto, votano tutti e il modello risponde sempre la stessa cosa,
cioè la **classe** più frequente (d'ora in avanti «classe» è il nome tecnico
di quelle che finora abbiamo chiamato categorie: spam e non spam sono due
classi).

`````{tab} Elementare

Per classificare una casa nuova si cercano le $k$ case più simili fra quelle
che già si conoscono, e le si lascia *votare*. Se i $5$ vicini più prossimi
stanno per lo più in "quartiere costoso", ci starà anche lei. Se la domanda
invece è un prezzo, al posto del voto si fa la media dei prezzi di quei $5$,
che è la stessa mossa con una risposta di tipo diverso.

Non c'è addestramento vero e proprio:
il modello tiene in memoria tutti gli esempi e decide solo al momento della
domanda. Per questo si dice **non parametrico**: non riassume i dati in pochi
numeri, li usa tutti. Il lavoro non sparisce, si sposta. Ogni volta che arriva
una casa nuova bisogna confrontarla con tutte quelle in archivio, e ogni
confronto va fatto colonna per colonna. Con centomila case e venti colonne sono
due milioni di operazioni per una sola risposta, da rifare da capo alla domanda
successiva.

E «simili» va deciso con attenzione, perché la somiglianza si calcola sommando
gli scarti di tutte le colonne, e le colonne non hanno la stessa taglia. Una
casa da $100$ m² con tre stanze, e due case che si contendono il posto di
vicina. La prima ha $108$ m² e una stanza sola, la seconda $115$ m² e tre
stanze. Sommando gli
scarti così come sono, $8$ metri quadri contro $2$ stanze, vince la prima, e il
numero di stanze non conta quasi niente, perché la superficie si muove fra $40$
e $200$ mentre le stanze stanno fra $1$ e $5$. Rimettendo le due colonne sulla
stessa taglia, cioè contando ogni scarto in rapporto all'intervallo che la sua
colonna copre ($8$ su $160$ fa un ventesimo, $2$ su $4$ fa metà), l'ordine si
rovescia, e la vicina diventa la seconda, quella con lo stesso numero di
stanze. Chi fa votare i vicini senza questa precauzione lascia decidere tutto
alla colonna con i numeri più grandi.

`````

`````{tab} Superiore

Dato un punto $\mathbf{x}$, si ordinano gli esempi di addestramento per
distanza, tipicamente euclidea,
$\lVert \mathbf{x} - \mathbf{x}^{(i)}\rVert_2$, e si prendono i $k$ più
vicini. In classificazione si assegna la classe di maggioranza; in regressione
si fa la media dei loro $y^{(i)}$. Non esiste una fase di ottimizzazione: il
costo si sposta interamente sulla previsione, ed è $O(mn)$ per query nella
versione ingenua, non $O(m)$: le distanze da calcolare sono $m$, una per
esempio, ma ciascuna costa $n$ operazioni, una per colonna. Quel fattore $n$
conta due volte, perché il numero di colonne pesa sul costo e decide anche se
il metodo funziona, e la sezione su riduzione e
clustering lo mette al centro. Il valore di $k$ regola il compromesso: $k$
piccolo segue il rumore,
$k$ grande liscia troppo. La distanza euclidea, inoltre, impone di
normalizzare le feature, altrimenti quella con la scala più ampia domina il
conto.

Due raffinamenti sono già in scikit-learn. Il **voto pesato**
(`weights="distance"`) fa contare di più i vicini più prossimi invece di dare
a tutti e $k$ lo stesso peso. Le **strutture di indicizzazione** (KD-tree,
ball-tree) partizionano lo spazio in anticipo e abbattono il numero di distanze
da calcolare, da $m$ a circa $\log m$. Proprio quegli indici, però, smettono di
essere utili oltre poche decine di dimensioni, dove le distanze fra i punti si
assomigliano tutte e il partizionamento non riesce più a escludere nessuna
regione.

`````

```{warning}
**k-NN ha un nemico naturale: le troppe dimensioni.** Tutto il metodo poggia
sull'idea che «vicino» voglia dire «simile». Ricordando che ogni colonna della
tabella è una direzione e ogni esempio un punto, «tante dimensioni» vuol dire
semplicemente «tante colonne»: cento misure per ogni paziente, mille parole
contate per ogni email. Lassù quell'idea si sgretola, e la ragione si capisce
coi dadi.

La distanza fra due punti si ottiene **sommando** gli scarti su *tutte* le
colonne. Con una colonna sola quella somma ha un addendo, e due esempi possono
essere identici o lontanissimi: il caso decide tutto. Con mille colonne gli
addendi sono mille, e succede quello che succede lanciando mille dadi: il
totale cade quasi sempre attorno a $3\,500$, perché i lanci alti e quelli bassi
si compensano a vicenda, e vedere mille sei di fila non capita mai. Allo stesso
modo, due esempi qualsiasi saranno un po’ diversi su certe colonne e un po’
simili su altre, e la somma finisce quasi sempre attorno allo stesso valore. Le
distanze fra tutte le coppie si assomigliano, e lo scarto fra il vicino più
prossimo e il più lontano si assottiglia fino a sparire. È
come chiedere a qualcuno di indicare il migliore amico in una folla dove tutti
stanno esattamente alla stessa distanza: la domanda perde senso, e il voto dei
$k$ vicini diventa un voto casuale.

Per questo k-NN va quasi sempre preceduto da un lavoro che riduca le colonne,
o tenendo solo quelle che servono o riassumendole in poche. Il fenomeno, con i
conti, è la **maledizione della dimensionalità**, ed è il punto di partenza
della sezione su riduzione e clustering.
```

## Un'ombra all'orizzonte: l'overfitting

C'è un tranello in agguato. Un modello abbastanza **flessibile** (cioè capace
di piegarsi a qualsiasi forma: una curva contorta lo è, una retta no) può
imparare *a memoria* gli esempi di addestramento, rumore compreso, e poi
fallire su dati nuovi. Attenzione a non confonderlo con lo studente
dell'apertura del capitolo, quello che studia con le soluzioni a fianco: là
guardare le soluzioni era il metodo giusto, qui il guaio è **ricopiarle senza
averle capite**, e accorgersene è possibile solo interrogandolo su un esercizio
che non ha mai visto.
È l’**overfitting**, il problema centrale del machine learning applicato: lo
affrontiamo nella sezione dedicata, insieme all'idea di tenere sempre da parte
dati che il modello non ha mai visto per misurarne l'onestà.

## In pratica, con scikit-learn

In Python i tre modelli sono tre righe, con la stessa interfaccia `fit`/`predict`:

```python
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Regressione: prevede un valore continuo (es. il prezzo)
reg = LinearRegression().fit(X_train, y_prezzo)
prezzo_stimato = reg.predict(X_nuovo)

# Classificazione lineare: prevede una probabilità, poi una classe
clf = LogisticRegression().fit(X_train, y_spam)      # y_spam vale 0 oppure 1
# predict_proba dà due colonne, la probabilità del no e quella del sì:
# [:, 1] vuol dire «tieni la seconda», cioè quanto è probabile lo spam
prob_spam = clf.predict_proba(X_nuovo)[:, 1]

# k-NN: niente da stimare, "vota" con i 5 vicini più simili
knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_spam)
etichetta = knn.predict(X_nuovo)
```

La stessa forma (`fit` per imparare, `predict` per rispondere) vale per quasi
tutti i modelli della libreria: è la grammatica comune che ci porteremo dietro
per tutto il resto del libro.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- **Supervisionato** vuol dire imparare da esempi che portano già con sé la
  risposta giusta: tante coppie *(descrizione, risposta)*, e una regola da
  trovare che leghi le une alle altre.
- Ogni **colonna** della tabella è una direzione, ogni **riga** un punto in
  quello spazio. Con due colonne il disegno sta su un foglio; con cento no, ma
  i conti sono gli stessi, e «vicini» continua a voler dire «simili».
- Se la risposta è un **numero** si cerca una retta che passi *in mezzo* ai
  punti; se è un **sì o no** si cerca una linea che li *separi*, dopo aver
  trasformato il punteggio in una probabilità e aver scelto dove tagliare.
- E se la risposta è un **conteggio** («quante volte») non va bene nessuna
  delle due: il punteggio si legge come il logaritmo del numero atteso, così la
  previsione non può venire negativa e i pesi moltiplicano invece di sommare.
  Le tre risposte sono la stessa macchina con tre impostazioni. Attenzione però
  ai conteggi veri, che quasi sempre oscillano più di quanto quel modello
  ammetta: i pesi restano giusti, la fiducia dichiarata no.
- La retta buona si può trovare con un conto diretto, che però con tanti dati
  costa più della passeggiata: allora si parte da una qualsiasi e la si sposta
  a piccoli passi nella direzione in cui l'errore cala (la **discesa del
  gradiente**), decidendo quanto lunghi sono i passi.
- Il **k-NN** non impara niente: tiene in memoria tutti gli esempi e, alla
  domanda, fa votare i $k$ più simili. Semplicissimo, ma va in crisi quando le
  colonne sono troppe, perché allora tutti i punti sono lontani uguale.
- L'insidia di tutto il capitolo è imparare **a memoria** invece che capire
  (l’*overfitting*): è la prossima sezione.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Supervisionato significa imparare $f:\mathcal{X}\to\mathcal{Y}$ da **esempi
  già etichettati**, minimizzando una loss $\mathcal{L}$ su $m$ coppie
  $(\mathbf{x}^{(i)}, y^{(i)})$.
- **Regressione** = uscita continua (MSE, retta di best fit); **classificazione**
  = uscita discreta (sigmoide, confine di decisione $\mathbf{w}^\top\mathbf{x}+b=0$).
- I **modelli lineari generalizzati** {cite}`nelder1972generalized` mettono i
  due casi (e la **Poisson** per i conteggi) sotto un solo impianto:
  distribuzione nella famiglia esponenziale, predittore lineare
  $\eta=\mathbf{w}^\top\mathbf{x}+b$, legame $g(\mu)=\eta$. Con il legame
  canonico $g=(A')^{-1}$ la log-verosimiglianza è concava e il gradiente vale
  $\sum_i (y_i-\mu_i)\mathbf{x}_i$ per tutti e tre, da cui l'IRLS. La Poisson
  impone $\operatorname{Var}=\mu$: contro la sovradispersione, binomiale
  negativa.
- Il tipo di ogni colonna (numerica, categorica, ordinale) è una decisione di
  chi prepara i dati: una categorica codificata come intero acquista un ordine
  e delle distanze che nessuno intendeva metterci.
- **k-NN** è non parametrico: non stima parametri, ricorda i dati e li fa votare.
  Costo $O(mn)$ per query ($m$ distanze da $n$ coordinate ciascuna), che gli
  indici spaziali (KD-tree, ball-tree) portano a circa $O(n\log m)$ sotto le
  poche decine di dimensioni; sopra, la concentrazione delle distanze affossa
  gli indici e il metodo insieme.
- Attenzione all’**overfitting**: imparare a memoria non è capire. Ne parliamo
  nella sezione dedicata.
```

`````
