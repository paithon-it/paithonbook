# Teoria dell'apprendimento: perché gli esempi bastano

```{image} ../figures/aperture/teoria-apprendimento.png
:class: pt-apertura only-light
:width: 100%
:alt: Un pollo di profilo che becca tranquillo un mucchietto di grano.
```

```{image} ../figures/aperture/teoria-apprendimento-scura.png
:class: pt-apertura only-dark
:width: 100%
:alt: Un pollo di profilo che becca tranquillo un mucchietto di grano.
```

Per tutta la vita, ogni mattina, un pollo ha visto arrivare l'uomo che lo
accudisce con il mangime; una mattina l'uomo arriva e gli tira il collo. La
storia è di Bertrand Russell, che nel 1912 la racconta per spiegare che cosa non
torna nel ragionare per induzione, cioè nel passare dai casi visti a quelli che
verranno. Idee più raffinate sull'uniformità della natura (sul fatto che domani
vada come ieri), commenta Russell, al pollo sarebbero state utili
{cite}`russell1912problems`.

Ogni modello che impara dagli esempi sta nella posizione di quel pollo. Ha
visto mille casi e scommette sul milleunesimo, e nessuna quantità di casi
visti dimostra il prossimo: David Hume lo aveva messo per iscritto già nel
Settecento {cite}`hume1739treatise`. Russell però, poche righe più in là, dice
anche che cosa resta da fare: se la certezza non si può avere, la probabilità
è tutto ciò che si deve cercare.

Settantadue anni dopo Leslie Valiant diede a quell'idea una forma che si può
calcolare {cite}`valiant1984theory`. Rinuncia alla certezza in due punti
precisi. Il modello imparato non deve essere perfetto ma **approssimativamente
corretto**, con un errore non più grande di una tolleranza fissata prima; e non
deve riuscirci sempre ma **probabilmente**, cioè su quasi tutti i campioni di
esempi che il caso può fornire. In cambio chiede di sapere quanti esempi servono
perché le due promesse valgano insieme. È l'apprendimento *PAC*, dalle iniziali
inglesi di *probably approximately correct*, e da lì nasce la **teoria
dell'apprendimento**: la parte del machine learning che studia perché e quando i
modelli imparati funzionano, e che da quelle domande ha tratto anche algoritmi
nuovi (il {doc}`boosting </MachineLearning/alberi-ensemble>` nasce da una
domanda posta nell'apprendimento PAC). Per questo lavoro, e per altri sulla
complessità del calcolo, Valiant ha ricevuto il premio Turing del 2010, il
riconoscimento più alto dell'informatica.

La promessa ha una condizione, e il pollo di Russell la mostra meglio di
qualunque formula. Tutto il ragionamento presuppone che gli esempi futuri
vengano dalla stessa sorgente di quelli passati, estratti allo stesso modo e
ciascuno indipendente dagli altri. Le mattine del pollo non lo erano: l'ultima
non era un'estrazione come le altre, era una data segnata su un calendario che
il pollo non vedeva. Nessuna garanzia del tipo di quella di Valiant salva chi
viola quella condizione, che è la stessa incontrata guardando {doc}`i dati che
cambiano </MachineLearning/dati-che-cambiano>`. L'apprendimento online di quella
sezione ne dà una che non la chiede, e proprio per questo promette meno: non
dice quanto il modello sbaglierà su un esempio nuovo, ma di quanto, col senno di
poi, avrà fatto peggio del miglior modello fisso.

## Due errori, e la distanza fra loro

Per parlarne con precisione servono due numeri. Il **rischio empirico**
$\hat{R}_S(h)$ è la frazione di errori che una regola $h$ commette sugli esempi
del campione $S$ (la lettera viene da *ipotesi*, il nome che la teoria dà a ogni
regola candidata); il **rischio** $R(h)$ è la probabilità che sbagli su un
esempio nuovo estratto dalla stessa sorgente, cioè il suo errore medio su tutto
quello che la sorgente può produrre, pesato con la frequenza con cui lo produce.
Il nome viene dalla teoria statistica delle decisioni, dove il rischio è la
perdita che ci si aspetta, ed «empirico» vuol dire misurato sui dati. Il primo
si calcola, il secondo no, e la teoria dell'apprendimento è lo studio della
distanza fra i due.

`````{tab} Elementare

Un'urna contiene milioni di biglietti, e su ognuno c'è una domanda con la sua
risposta giusta: una foto e il nome dell'animale che mostra, un'email e la
parola «spam» oppure «buona». Chi costruisce un modello l'urna non la vede.
Ne pesca mille biglietti, uno alla volta, rimettendo ogni volta dentro quello
pescato e rimescolando, così che nessuna pescata dipenda dalle precedenti, e
prova le sue regole su quei mille.

Di ogni regola si possono dire due percentuali. Quella degli errori sui mille
biglietti pescati si conta, ed è il rischio empirico. Quella degli errori su
tutti i biglietti dell'urna è la cosa che interessa davvero, il rischio, e non
si conta mai. Se la regola fosse stata scelta prima di pescare, la prima sarebbe
una buona stima della seconda, come un sondaggio fatto bene lo è del voto, con
il margine che la {doc}`sezione sulla concentrazione
</Matematica/concentrazione>` sa calcolare.

Ma la regola si sceglie dopo aver guardato i biglietti, prendendo quella che ne
sbaglia di meno, e fra tante regole quella che va meglio sui mille è anche
quella che con quei mille ha avuto più fortuna. È il vincitore di un concorso
in cui conta anche la sorte: il suo punteggio dice un po' di bravura e un po'
di buona sorte, e non dice quanta dell'una e quanta dell'altra. È la storia
delle mille persone che lanciano una moneta dieci volte, incontrata con lo
{doc}`spazio delle ipotesi </MachineLearning/apprendimento-supervisionato>`:
qualcuna fa dieci teste e sembra bravissima, e alla prova successiva fa come
tutti. Le persone sono le regole in gara.

Per questo non basta che il punteggio sui biglietti pescati sia vicino a quello
sull'urna per una regola sola. Siccome non si sa in anticipo chi farà dieci
teste, deve esserlo per tutte le regole fra cui si sceglie, tutte insieme:
allora, qualunque sia la vincitrice, il suo punteggio è onesto. E la garanzia
deve valere qualunque cosa ci sia scritto sui biglietti, anche per l'urna
peggiore, riempita apposta per ingannare chi pesca, perché l'urna non la vede
nessuno. Quello che c'è dentro, e in che proporzioni, è ciò che i matematici
chiamano la distribuzione dei dati. Quante regole si possono mettere in gara
prima che questa garanzia salti, e quanti biglietti servono per tenerla in
piedi, è la domanda da cui nasce tutto il resto.

L'urna poi deve restare la stessa. Se il biglietto che conta, quello di
domani, viene pescato da un'altra urna, i mille di prima non dicono più
niente: è quello che è successo al pollo.

`````

`````{tab} Superiore

Sia $\mathcal{D}$ una distribuzione ignota su $\mathcal{X}\times\mathcal{Y}$ e
$S = \big((x_1,y_1),\dots,(x_m,y_m)\big)\sim\mathcal{D}^m$ un campione i.i.d.
La notazione è quella dei testi di teoria dell'apprendimento
{cite}`mohri2018foundations`, e si scosta in tre punti da quella del
{doc}`capitolo sul machine learning
</MachineLearning/apprendimento-supervisionato>`: $\mathcal{D}$ qui è la
distribuzione, e l'insieme di addestramento, che là si chiamava $\mathcal{D}$,
qui si chiama $S$; le $x_i$ stanno in un insieme qualunque $\mathcal{X}$, non
per forza di vettori, e per questo restano in tondo; la regola è un'ipotesi
$h$, presa dallo spazio delle ipotesi $\mathcal{H}$, invece di $f_\theta$. Per
$h:\mathcal{X}\to\mathcal{Y}$ e la perdita 0-1, il rischio (o errore di
generalizzazione) e il rischio empirico sono

$$
R(h) = \Pr_{(x,y)\sim\mathcal{D}}\big[h(x)\neq y\big],
\qquad
\hat{R}_S(h) = \frac{1}{m}\sum_{i=1}^{m}\mathbb{1}\big[h(x_i)\neq y_i\big].
$$

Il secondo è la perdita media $\mathcal{L}$ di quel capitolo, calcolata con la
perdita 0-1. Per $h$ fissata prima di vedere $S$, $\hat{R}_S(h)$ è la media di
$m$ Bernoulli indipendenti di parametro $R(h)$, e la disuguaglianza di
Hoeffding dà $|R(h)-\hat{R}_S(h)|\le\sqrt{\log(2/\delta)/(2m)}$ con probabilità
almeno $1-\delta$. L'ipotesi restituita dall'algoritmo, $h_S$, dipende però da
$S$, e per lei quella disuguaglianza non vale. Se poi l'algoritmo sceglie
l'ipotesi che sbaglia meno sul campione (la *minimizzazione del rischio
empirico*), $\hat{R}_S(h_S)$ è anche distorta verso il basso: detta $h^*$ la
migliore ipotesi di $\mathcal{H}$,
$\mathbb{E}\hat{R}_S(h_S)\le\mathbb{E}\hat{R}_S(h^*)=R(h^*)\le\mathbb{E}R(h_S)$,
ed è lo stesso meccanismo della maledizione del vincitore della
{doc}`sezione sulla concentrazione </Matematica/concentrazione>`. Il rimedio è
chiedere la **convergenza uniforme**,

$$
\Pr_{S\sim\mathcal{D}^m}\Big[\sup_{h\in\mathcal{H}}
\big|R(h)-\hat{R}_S(h)\big| > \varepsilon\Big] \le \delta,
$$

che copre qualunque $h$ scelta in $\mathcal{H}$ a partire dai dati, $h_S$
compresa. I risultati che seguono dicono quanto grande debba essere $m$ perché
questo accada, in funzione della ricchezza di $\mathcal{H}$, e valgono per
ogni $\mathcal{D}$: sono garanzie *distribution-free*, ed è questo che le
rende utili, perché la distribuzione nessuno la conosce. L'unica ipotesi sulla
sorgente è che il campione sia i.i.d. e che la distribuzione su cui si valuta
il modello sia la stessa su cui lo si è addestrato. Quando cade, cadono tutte,
e il pollo di Russell ne è il controesempio.

`````

Le misure che seguono rispondono ciascuna al limite di quella prima. Quando le
regole in gara sono un numero finito basta contarle, e il conto dice che gli
esempi necessari crescono pochissimo con il loro numero (raddoppiare le regole
costa sempre la stessa manciata di esempi in più): lo mostra la sezione su
{doc}`che cosa vuol dire imparare in senso PAC <pac>`. Quando sono infinite
(tutte le rette del piano, tutti i valori di una soglia, cioè di una regola che
risponde sì sopra un certo valore e no sotto) il conto non si può fare, e al
posto delle regole si contano le classificazioni diverse che producono su $m$
punti; il numero di punti fino al quale le producono tutte, cioè riescono a
dividerli fra le due classi in ogni modo possibile, è la {doc}`dimensione VC
<dimensione-vc>`. Quando anche questa misura è troppo
pessimista, perché guarda il caso peggiore su tutte le distribuzioni possibili,
si misura la ricchezza della famiglia di regole sui dati che si hanno davvero,
chiedendole di inseguire monete lanciate a caso: è la {doc}`complessità di
Rademacher <rademacher-margine>`, e con lei il margine della SVM trova la sua
giustificazione. Le garanzie che ne escono hanno tutte la stessa forma: l'errore
vero sta sotto quello sugli esempi più un termine che cresce con la ricchezza
della famiglia e cala con il numero degli esempi. Resta infine l'esperimento
che mette in crisi tutte e tre le
misure: le reti neurali imparano senza fatica anche etichette tirate a sorte, e
quello che le fa generalizzare, quando ci riescono, va cercato {doc}`fuori
dalla sola ricchezza del modello <garanzie-e-reti>`.
