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
storia è di Bertrand Russell, che nel 1912 la racconta per spiegare che cosa
non torna nel ragionare per **induzione**, cioè nel passare dai casi visti a
quelli che verranno. Idee più raffinate sull'uniformità della natura (sul
fatto che domani vada come ieri) sarebbero state utili al pollo, commenta
Russell {cite}`russell1912problems`.

Ogni modello che impara dagli esempi sta nella posizione di quel pollo: ha
visto mille casi e scommette sul milleunesimo. Nessuna quantità di casi visti,
però, dimostra il prossimo, e David Hume lo aveva messo per iscritto già nel
Settecento {cite}`hume1739treatise`. Qualche pagina dopo il pollo, Russell
conclude che, quando la certezza non si può avere, la probabilità è tutto ciò
che si deve cercare.

Settantadue anni dopo, Leslie Valiant diede a quell'idea una forma che si può
calcolare {cite}`valiant1984theory`. La sua definizione rinuncia alla certezza
in due punti precisi. Il modello imparato non deve essere perfetto ma
**approssimativamente corretto**, con un errore non più grande di una
tolleranza fissata prima; e non deve riuscirci sempre ma **probabilmente**,
cioè su quasi tutti i campioni di esempi che il caso può fornire. In cambio,
la definizione chiede di sapere quanti esempi servono perché le due promesse
valgano insieme. È l'apprendimento *PAC*, dalle iniziali inglesi di *probably
approximately correct*.

Da lì nasce la **teoria dell'apprendimento**, cioè la parte del machine
learning che studia perché e quando i modelli imparati funzionano, e che da
quelle domande ha tratto anche algoritmi nuovi. Il {doc}`boosting
</MachineLearning/alberi-ensemble>`, ad esempio, nasce da una domanda posta
nell'apprendimento PAC. Per il lavoro sull'apprendimento PAC, e per altri sulla
complessità del calcolo, Valiant ha ricevuto il premio Turing del 2010, il
riconoscimento più alto dell'informatica.

La promessa ha una condizione, e il pollo di Russell la mostra meglio di
qualunque formula. Tutto il ragionamento presuppone che gli esempi futuri
vengano dalla stessa sorgente di quelli passati, estratti allo stesso modo e
ciascuno indipendente dagli altri. Le mattine del pollo non lo erano: l'ultima
non era un'estrazione come le altre, ma una data segnata su un calendario che
il pollo non vedeva.

Di conseguenza, nessuna garanzia del tipo di quella di Valiant salva chi viola
questa condizione, che è la stessa incontrata guardando {doc}`i dati che
cambiano </MachineLearning/dati-che-cambiano>`. L'apprendimento online di quella
sezione, invece, dà una garanzia che non la richiede, e per questo promette
meno: dice di quanto il modello, col senno di poi, avrà fatto peggio del
miglior modello fisso, ma non dice quanto sbaglierà su un esempio nuovo.

## Due errori, e la distanza fra loro

Per parlarne con precisione servono due numeri. Il **rischio empirico**
$\hat{R}_S(h)$ è la frazione di errori che una regola $h$ commette sugli esempi
del campione $S$ (la lettera viene da *ipotesi*, il nome che la teoria dà a ogni
regola candidata). Il **rischio** $R(h)$ è la probabilità che la regola sbagli
su un esempio nuovo estratto dalla stessa sorgente, cioè il suo errore medio
su tutti gli esempi che la sorgente può produrre, ciascuno pesato con la sua
frequenza.

Il nome viene dalla teoria statistica delle decisioni, dove il rischio è la
perdita che ci si aspetta, ed «empirico» vuol dire misurato sui dati. Il primo
dei due numeri si calcola, il secondo no, e la teoria dell'apprendimento è lo
studio della distanza fra i due.

`````{tab} Elementare

Un'urna contiene milioni di biglietti, e su ognuno c'è una domanda con la sua
risposta giusta: ad esempio una foto e il nome dell'animale che mostra, oppure
un'email e la parola «spam» o «buona». Chi costruisce un modello non vede
l'urna. Pesca mille biglietti, uno alla volta, rimettendo ogni volta nell'urna
quello estratto e rimescolando, così che nessuna pescata dipenda dalle
precedenti. Poi prova le sue regole su quei mille.

Di ogni regola si possono dire due percentuali. Quella degli errori sui mille
biglietti pescati si conta, e si chiama rischio empirico. Quella degli errori
su tutti i biglietti dell'urna, il rischio, è quella che interessa, ma non si
conta mai.

Se la regola fosse stata scelta prima di pescare, la prima percentuale sarebbe
una buona stima della seconda, come un sondaggio fatto bene lo è del voto, con
il margine che la {doc}`sezione sulla concentrazione
</Matematica/concentrazione>` sa calcolare.

La regola, però, si sceglie dopo aver guardato i biglietti, prendendo quella
che ne sbaglia di meno. Così, fra tante regole, quella che va meglio sui mille
è anche quella che con quei mille ha avuto più fortuna. È il vincitore di un
concorso in cui conta anche la sorte: il suo punteggio dice un po' di bravura e
un po' di fortuna, ma non dice quanta dell'una e quanta dell'altra.

Lo mostra la storia delle mille persone che lanciano una moneta dieci volte,
incontrata con lo {doc}`spazio delle ipotesi
</MachineLearning/apprendimento-supervisionato>`: qualcuna fa dieci teste e
sembra bravissima, ma alla prova successiva fa come tutti. Le persone sono le
regole in gara.

Per questo non basta che il punteggio sui biglietti pescati sia vicino a quello
sull'urna per una regola sola. Siccome non si sa in anticipo chi farà dieci
teste, i due punteggi devono essere vicini per tutte le regole fra cui si
sceglie, tutte insieme: allora, qualunque sia la vincitrice, il suo punteggio è
onesto.

Inoltre, la garanzia deve valere qualunque cosa ci sia scritto sui biglietti,
anche per l'urna peggiore, riempita apposta per ingannare chi pesca, perché
l'urna non la vede nessuno. Il contenuto dell'urna, con le proporzioni dei vari
biglietti, si chiama **distribuzione dei dati**. Quante regole si possono
mettere in gara prima che questa garanzia salti, e quanti biglietti servono per
tenerla in piedi, sono le domande da cui nasce tutto il resto.

L'urna poi deve restare la stessa. Se il biglietto che conta, quello di
domani, viene pescato da un'altra urna, i mille di prima non dicono più
niente, come è successo al pollo.

`````

`````{tab} Superiore

Sia $\mathcal{D}$ una distribuzione ignota su $\mathcal{X}\times\mathcal{Y}$ e
$S = \big((x_1,y_1),\dots,(x_m,y_m)\big)\sim\mathcal{D}^m$ un campione i.i.d.
La notazione è quella dei testi di teoria dell'apprendimento
{cite}`mohri2018foundations`, e si scosta in tre punti da quella della
{doc}`sezione sull'apprendimento supervisionato
</MachineLearning/apprendimento-supervisionato>`. In primo luogo, $\mathcal{D}$
qui è la distribuzione, mentre l'insieme di addestramento, che là si chiamava
$\mathcal{D}$, qui si chiama $S$. In secondo luogo, le $x_i$ stanno in un
insieme qualunque $\mathcal{X}$, non per forza di vettori, e per questo restano
in tondo. Infine, la regola è un'ipotesi $h$, presa dallo spazio delle ipotesi
$\mathcal{H}$, invece di $f_\theta$.

Per $h:\mathcal{X}\to\mathcal{Y}$ e la perdita 0-1, il rischio (o errore di
generalizzazione) e il rischio empirico sono

$$
R(h) = \Pr_{(x,y)\sim\mathcal{D}}\big[h(x)\neq y\big],
\qquad
\hat{R}_S(h) = \frac{1}{m}\sum_{i=1}^{m}\mathbb{1}\big[h(x_i)\neq y_i\big].
$$

Il secondo è la perdita media $\mathcal{L}$ di quel capitolo, calcolata con la
perdita 0-1. Consideriamo un'ipotesi $h$ fissata prima di vedere $S$. Allora
$\hat{R}_S(h)$ è la media di $m$ Bernoulli indipendenti di parametro $R(h)$, e
la disuguaglianza di Hoeffding dà
$|R(h)-\hat{R}_S(h)|\le\sqrt{\log(2/\delta)/(2m)}$ con probabilità almeno
$1-\delta$.

L'ipotesi restituita dall'algoritmo, $h_S$, dipende però da $S$, e per lei
quella disuguaglianza non vale. Se poi l'algoritmo sceglie
l'ipotesi che sbaglia meno sul campione (la **minimizzazione del rischio
empirico**), $\hat{R}_S(h_S)$ è anche distorta verso il basso. Infatti, detta
$h^*$ la migliore ipotesi di $\mathcal{H}$, vale
$\mathbb{E}\hat{R}_S(h_S)\le\mathbb{E}\hat{R}_S(h^*)=R(h^*)\le\mathbb{E}R(h_S)$,
per lo stesso meccanismo della maledizione del vincitore della
{doc}`sezione sulla concentrazione </Matematica/concentrazione>`. Il rimedio è
chiedere la **convergenza uniforme**,

$$
\Pr_{S\sim\mathcal{D}^m}\Big[\sup_{h\in\mathcal{H}}
\big|R(h)-\hat{R}_S(h)\big| > \varepsilon\Big] \le \delta,
$$

che copre qualunque $h$ scelta in $\mathcal{H}$ a partire dai dati, $h_S$
compresa. I risultati che seguono dicono quanto grande debba essere $m$ perché
questo accada, in funzione della ricchezza di $\mathcal{H}$, e valgono per ogni
$\mathcal{D}$: sono garanzie *distribution-free*, utili perché nessuno conosce
la distribuzione.

L'unica ipotesi sulla sorgente è che il campione sia i.i.d. e che la
distribuzione su cui si valuta il modello sia la stessa su cui lo si è
addestrato. Quando questa ipotesi cade, le garanzie non valgono più, a meno di
sapere di quanto le due distribuzioni differiscono; il pollo di Russell ne è il
caso estremo.

`````

In questo capitolo vediamo diverse misure della ricchezza di una famiglia di
regole, e ciascuna risponde al limite di quella prima. Quando le regole
candidate sono un numero finito basta contarle, e il calcolo dice che gli
esempi necessari crescono pochissimo con il loro numero (raddoppiare le regole
richiede sempre la stessa manciata di esempi in più). Lo mostra la sezione su
{doc}`che cosa vuol dire imparare in senso PAC <pac>`.

Quando invece le regole sono infinite, come tutte le rette del piano o tutti i
valori di una soglia (cioè di una regola che risponde sì sopra un certo valore
e no sotto), non si possono contare. Al posto delle regole si contano le
classificazioni diverse che producono su $m$ punti, e il numero di punti fino
al quale le producono tutte, cioè riescono a dividerli fra le due classi in
ogni modo possibile, è la {doc}`dimensione VC <dimensione-vc>`.

Quando anche questa misura è troppo pessimista, perché guarda il caso peggiore
su tutte le distribuzioni possibili, si misura la ricchezza della famiglia di
regole sui dati disponibili, chiedendole di inseguire monete lanciate a caso.
È la {doc}`complessità di Rademacher <rademacher-margine>`, e con lei il
margine della SVM trova la sua giustificazione.

Le garanzie che si ottengono con queste misure hanno tutte la stessa forma:
l'errore vero sta sotto quello sugli esempi più un termine che cresce con la
ricchezza della famiglia e cala con il numero degli esempi.

Resta infine l'esperimento che mette in crisi tutte e tre le misure: le reti
neurali imparano senza fatica anche etichette casuali, e il motivo per cui
generalizzano, quando ci riescono, va quindi cercato {doc}`fuori dalla sola
ricchezza del modello <garanzie-e-reti>`.
