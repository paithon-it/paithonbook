# Conclusioni

Siamo partiti, nell'introduzione, da una frase di Joseph Weizenbaum: «si dice
che spiegare significhi dissolvere» {cite}`weizenbaum1966eliza`. Voleva dire
che quando capisci come è fatto un trucco, il trucco smette di essere magia.
Adesso possiamo dire com'è andata. Sono passati
{{ n_capitoli_meno_uno_lettere }} capitoli, dall'algebra lineare alle reti che
generano immagini, dagli alberi di decisione agli agenti che usano strumenti,
dal codice che fa correre una scheda grafica alle domande su chi risponde
quando un modello sbaglia. Dei meccanismi non è rimasto quasi niente di
prodigioso: guardati da vicino, sono conti. Ed è proprio per questo che
l'intelligenza artificiale continua a non lasciarsi definire: ogni cosa che si
riesce a spiegare smette di sembrare intelligenza e diventa "solo" un
algoritmo.

Ma adesso abbiamo qualcosa che all'inizio non avevamo: gli strumenti per capire
*come* queste macchine funzionano davvero, e per distinguere ciò che sanno fare
da ciò che sembrano fare. Voltiamoci a guardare la strada percorsa, perché da
qui si vede un disegno che i singoli capitoli, da vicino, non lasciavano
intravedere.

## Il percorso, guardato dall'alto

**Il primo tratto** sono i mattoni: vettori, matrici, derivate, probabilità.
Non erano un rito d'iniziazione, ma il vocabolario di tutto il resto. Un
neurone è un prodotto scalare: moltiplica ogni ingresso per il suo peso e somma
i risultati, come alla cassa si moltiplica ogni quantità per il suo prezzo e si
fa il totale. L'apprendimento è una discesa lungo un gradiente, cioè lungo la
pendenza dell'errore, verso il punto in cui l'errore è più basso. Una
previsione è una distribuzione di probabilità: non una risposta secca, ma un
elenco di risposte possibili, ciascuna con la fiducia che il modello le
assegna.

Da lì il machine learning classico ci ha insegnato la disciplina fondamentale:
tenere separati i dati su cui si impara da quelli su cui si misura, temere
l'overfitting (imparare a memoria gli esempi visti, e restare poi senza
risposta davanti a uno nuovo), misurare con onestà. Una disciplina che vale
identica per un modello con dieci parametri (le manopole che l'addestramento
regola) e per uno con mille miliardi.

**Il secondo tratto** sono le reti neurali, che hanno cambiato la scala e ci
hanno costretti a guardare anche sotto il cofano. Con PyTorch abbiamo impilato
strati e scritto a mano il ciclo di addestramento (il *training loop*); col
capitolo sulla [GPU](../GPU/overview.md) abbiamo visto perché una
moltiplicazione di matrici è veloce solo se i numeri arrivano dalla memoria
abbastanza in fretta da tenere occupato il processore. Le reti convoluzionali
hanno insegnato alle macchine a vedere, e lo spartiacque è AlexNet, che nel
2012 vince la gara di riconoscimento di immagini ImageNet portando l'errore dal
26,2% al 15,3%. Quel 26,2% era del miglior metodo costruito a mano, cioè con i
dettagli da guardare scelti da un esperto invece che imparati: è il salto che
nessuno si aspettava. I modelli di sequenza, e i Transformer dopo di loro,
hanno insegnato alle macchine a leggere e a scrivere
{cite}`vaswani2017attention`; e poi a tenere insieme [visione e
linguaggio](../VisioneLinguaggio/overview.md) nello stesso modello. Sono venuti
poi il suono, la voce, i grafi (i dati fatti di puntini collegati da linee), i
[sistemi che ti raccomandano cosa
guardare](../SistemiRaccomandazione/overview.md), le serie temporali, le
equazioni della fisica: i capitoli in cui la stessa matematica cambia mestiere
a seconda della forma dei dati.

**Il terzo tratto** è quello in cui cambia la domanda: non più solo riconoscere
quello che c'è (questa foto è un gatto), ma generare quello che non c'è (una
foto di un gatto che non esiste) e agire per ottenere qualcosa. Sono i modelli
generativi: le [GAN](../GAN/overview.md), dove due reti si sfidano e quella che
inventa impara a non farsi smascherare dall'altra; la
[diffusione](../ModelliDiffusione/overview.md), che parte dal rumore e lo
ripulisce un poco alla volta; e i [modelli a
energia](../ModelliEnergia/overview.md), la più antica delle tre famiglie, di
cui la diffusione, guardata da vicino, è un caso particolare. Poi il
[reinforcement learning](../ReinforcementLearning/overview.md), dove l'agente
non riceve le risposte giuste ma le scopre agendo: è così che AlphaGo, nel
2016, è diventato più forte dei giocatori umani dalle cui partite aveva
imparato le prime mosse. E infine le architetture nate da un conto che non si
regge: per capire un testo un Transformer confronta ogni parola con tutte le
altre, e su un testo lungo quel confronto diventa proibitivo. Da lì
l’[attenzione
lineare](../AttenzioneLineare/overview.md) e i [modelli a spazio di
stati](../StateSpaceModel/overview.md). E gli [agenti](../Agenti/overview.md)
che usano strumenti, il modo in cui li si [programma a
parole](../IngegneriaLLM/overview.md), quello che succede [quando
diventano molti](../SistemiMultiAgente/overview.md) e i [modelli del
mondo](../WorldModels/overview.md) che provano a immaginare le conseguenze di
un'azione prima di compierla.

`````{tab} Elementare
Attenzione a un equivoco: il terzo tratto non è il più recente. Alcune di
queste idee sono molto più vecchie di quanto la loro posizione nel libro lasci
pensare, e più vecchie
delle reti che hanno reso famoso il deep learning. I modelli a energia nascono
dalla fisica dei primi anni Ottanta; l'impalcatura matematica del reinforcement
learning è degli anni Cinquanta, cioè di quando i calcolatori occupavano una
stanza. Quello che è successo dopo il 2012 non è che qualcuno abbia inventato
quelle idee: è che finalmente c'erano i dati e le macchine per farle
funzionare.
`````

`````{tab} Superiore
La cronologia va disaccoppiata dall'ordine dell'esposizione, perché il gruppo è
tutt'altro che coetaneo. Alcuni pezzi sono vecchissimi: la rete di Hopfield è
del 1982 {cite}`hopfield1982neural` e l'algoritmo di apprendimento della
macchina di Boltzmann del 1985 {cite}`ackley1985learning`, mentre l'impalcatura
del reinforcement learning è degli anni Cinquanta: Bellman comincia a
pubblicarla nel 1952 e la raccoglie nel libro del 1957
{cite}`bellman1957dynamic`. Altri nascono a metà degli anni Dieci (le GAN nel
2014, la diffusione fra il 2015 e il 2020, AlphaGo nel 2016) e altri ancora
prendono velocità dal 2020 in poi (attenzione lineare e modelli a spazio di
stati, gli agenti, i modelli del mondo).

Nemmeno l'ordine dell'indice è cronologico, e vale la pena dirlo perché
l'indice è la mappa che il lettore ha sotto gli occhi: il reinforcement
learning è una parte a sé, collocata prima del linguaggio e dei Transformer.
L'indice segue le domande, non le date. Gli ordini in gioco, quindi, sono tre e
non due: quello dell'indice, quello di questo ripasso (che raggruppa per
domanda e non per parte) e quello della storia.
`````

E poi gli ultimi capitoli, che non parlano di architetture ma di **mestiere**:
portare un modello in produzione e tenerlo in vita
([MLOps](../MLOps/overview.md)), aprirlo per capire perché ha deciso così
([interpretabilità](../Interpretabilita/overview.md)), rispondere delle sue
conseguenze ([AI responsabile](../AIResponsabile/overview.md)). Non sono
appendici morali messe in fondo per buona educazione: sono la parte del lavoro
che decide se quello che hai costruito serve a qualcuno o fa danni.

Argomenti diversissimi, e in mezzo quasi settant'anni di storia: dal
percettrone di Rosenblatt del 1958 ai modelli di oggi. Eppure, sotto, tornano
sempre le stesse tre idee. Vale la pena nominarle.

## Tre fili, un solo tessuto

Se dovessimo comprimere l'intero libro in tre parole, sarebbero **dati**,
**rappresentazioni** e **ottimizzazione**. Sono i fili che attraversano ogni
capitolo, dal più elementare al più avanzato ({numref}`fig-fili-conduttori`).

```{figure} ../figures/fili-conduttori.svg
:name: fig-fili-conduttori
:alt: "Un asse orizzontale mostra l'arco del libro con cinque tappe (Matematica, Machine learning, Reti neurali, Deep learning, Generativi e RL); sotto, tre linee parallele colorate rappresentano i fili conduttori: Dati, Rappresentazioni, Ottimizzazione."
:width: 90%

I modelli cambiano da sinistra a destra, via via capaci di rappresentare cose
più complicate, ma sotto di loro scorrono sempre gli stessi tre fili. I
capitoli sul mestiere (produzione, interpretabilità, responsabilità) non stanno
su questo asse: stanno attorno a tutto.
```

I **dati** vengono prima di tutto: nessun modello sa più di ciò che ha visto.
L'introduzione li ha paragonati all'ossigeno, e adesso che il percorso è finito
quel paragone si può stringere. L'ossigeno è l'avanzo di una vita che non era
la nostra, e noi siamo diventati quello che siamo imparando a respirarlo; i
dati sono l'avanzo del nostro passaggio nel mondo digitale, e ogni macchina di
questo libro è un modo diverso di respirare quello. Le reti convoluzionali
respirano immagini, i Transformer testo, le reti su grafo relazioni fra cose.
Cambia il polmone, l'aria è sempre quella.

Ed è il filo che spiega perché il libro insista tanto su cose che sembrano
noiose accanto alle architetture: come si dividono i dati, che cosa succede
quando cambiano sotto i piedi, chi li ha lasciati e se era d'accordo. Un
modello non è più intelligente dell'aria che gli hai dato da respirare.

Le **rappresentazioni apprese** sono il cuore, cioè la parte che il deep
learning ha cambiato più di ogni altra.

`````{tab} Elementare
Per decenni, per far riconoscere un gatto a un computer, un esperto doveva
spiegargli a mano cosa guardare: i baffi, le orecchie a punta, la forma degli
occhi. Il salto del deep learning è stato smettere di suggerire. Diamo alla
rete milioni di foto e la lasciamo *scoprire da sola* quali dettagli contano.
Impara a vedere prima i bordi, poi le forme, poi interi oggetti: una gerarchia
che nessuno le ha imposto. Questa capacità di costruirsi le proprie "lenti"
per guardare i dati è ciò che chiamiamo rappresentazione appresa.
`````

`````{tab} Superiore
Una rete profonda è una funzione composta
$f_\theta = f^{(L)} \circ \dots \circ f^{(1)}$ che trasforma l'input grezzo
$\mathbf{x}$ in una sequenza di rappresentazioni intermedie sempre più
astratte. Gli strati nascosti non sono altro che *feature apprese*: coordinate
in uno spazio latente dove esempi semanticamente simili finiscono vicini. È il
principio degli *embedding*, e la ragione per cui una sola rete pre-addestrata
si riadatta a molti compiti.

Il feature engineering manuale del machine learning classico non è sparito, ma
va detto dove è finito, perché le destinazioni sono due e il libro le insegna
in capitoli diversi. Una parte è stata assorbita dentro $\theta$ e delegata
all'ottimizzazione, ed è la parte che si racconta di solito. L'altra si è
spostata nell’**architettura**, come bias induttivo: la convoluzione dichiara
che un motivo va riconosciuto ovunque compaia, una rete su grafo che l'ordine
con cui si elencano i nodi non cambia la risposta. Sono proprietà della forma
della funzione, vere per ogni valore di $\theta$ e anche a rete non addestrata:
non sono state imparate, le ha scritte a mano il progettista prima che
l'ottimizzazione cominciasse.
`````

E l’**ottimizzazione** è il motore che rende tutto questo possibile: apprendere
significa cercare i parametri che minimizzano un errore.

`````{tab} Elementare
Immagina di regolare le manopole di un vecchio mixer audio per far suonare bene
una canzone. Giri un po’ una manopola, ascolti se è migliorato, correggi.
Addestrare un modello è la stessa cosa, con milioni di manopole: a ogni passo
il modello guarda quanto ha sbagliato e sposta ciascuna manopola nella
direzione che riduce l'errore, un pochino. Ripetuto abbastanza volte, funziona.

Con un mixer vero la direzione giusta la scopri provando, e con milioni di
manopole non finiresti mai. Il modello non prova: la **calcola**. Per ogni
manopola si chiede "se la giro di un pelo in su, l'errore sale o scende, e di
quanto?", e ottiene tutte le risposte insieme con un conto solo, invece che con
milioni di tentativi. Quel conto è il gradiente, e il modo di ottenerlo in una
passata sola è la [retropropagazione](../RetiNeurali/backpropagation.md), che
sta nel capitolo sulle reti neurali. Senza, niente di tutto questo sarebbe
possibile.
`````

`````{tab} Superiore
La forma che copre il grosso del libro è la minimizzazione del **rischio
empirico** su un campione etichettato:

$$
\theta^\star = \arg\min_{\theta}\ \mathcal{L}(\theta)
= \arg\min_{\theta}\ \frac{1}{m}\sum_{i=1}^{m} \ell\big(f_\theta(\mathbf{x}^{(i)}),\, y^{(i)}\big),
$$

dove $\theta$ sono i parametri, $\mathcal{L}$ la funzione di costo totale,
$\ell$ la perdita sul singolo esempio, $f_\theta$ il modello, e
$\mathbf{x}^{(i)}$, $y^{(i)}$ l'input e il target dell’$i$-esimo degli $m$
esempi di addestramento. Due avvertenze sulla scrittura, prima di usarla. Nel
capitolo sul machine learning la stessa lettera $\mathcal{L}$ indica, a seconda
del punto, ora il singolo esempio ora il totale; qui la teniamo ferma sul
totale e diamo al singolo esempio la sua, $\ell$. E «etichettato» va inteso in
senso largo: nel pre-addestramento auto-supervisionato l'etichetta esiste, solo
che non la scrive nessuno, è il token successivo.

Cambia $f_\theta$ e, per i modelli differenziabili (dalla regressione lineare
al Transformer), la macchina che ci si avvicina è la discesa del gradiente
stocastica, che su una rete profonda raggiunge un punto stazionario e non il
minimo globale.

Il perimetro di quella scrittura, però, va dichiarato, perché è più stretto del
libro, e le eccezioni sono istruttive. **Le [GAN](../GAN/overview.md)** non ci
stanno: l'ottimizzazione simultanea di un gioco minimax non equivale a
minimizzare una singola funzione, ed è la radice della loro instabilità. **Il
[reinforcement
learning](../ReinforcementLearning/overview.md)** non ci sta: non esiste un
campione fisso di $m$ coppie, la distribuzione dei dati dipende dalla politica
che si sta cercando, e l'obiettivo è massimizzare un ritorno atteso. **I
[modelli a energia](../ModelliEnergia/overview.md)** non ci stanno: la
verosimiglianza che vorrebbero massimizzare contiene una funzione di partizione
che non si sa calcolare, e si ripiega su surrogati come lo score matching. E
nemmeno sul versante classico la copertura è totale: $k$-means almeno un
obiettivo ce l'ha, l'inerzia, che l'algoritmo di Lloyd però minimizza solo
localmente; DBSCAN è una procedura sulla densità e il clustering gerarchico una
fusione greedy, e nessuno dei due è il minimo di un obiettivo globale; anche un
albero di decisione cresce con split localmente ottimi, non minimizzando una
funzione sull'albero finito.

Resta vero che sono tutti problemi di **ottimizzazione**, ed è questo che tiene
insieme il libro. Quello che cambia da una famiglia all'altra è la natura
dell'obiettivo (una somma su un campione, un equilibrio fra due giocatori, un
ritorno atteso lungo traiettorie che il modello stesso genera, una
verosimiglianza inaccessibile), non il fatto che ce ne sia uno. Tre idee,
infinite architetture.
`````

## Dove sta andando

Per capire dove va un campo, la domanda utile non è quale modello sia il più
bravo adesso: quella risposta scade in pochi mesi. La domanda utile è che cosa
succede quando si dà a un modello più risorse.

I tre grafici qui sotto rispondono proprio a quella: quanto migliora un modello
se gli diamo più potenza di calcolo, più dati o più parametri? Ogni retta mostra
l'errore che il modello commette (la *loss*: più è bassa, meglio è) al crescere
di una di quelle tre risorse, tenute le altre due così abbondanti da non essere
loro a frenare. E in nessuno dei tre la retta si piega verso l'orizzontale: non
c'è, cioè, un punto oltre il quale aggiungere risorse smette di servire. Quella
piega ha un nome, il "ginocchio", e in questi grafici non compare.

```{figure} ../figures/scaling-laws-2020.svg
:name: fig-leggi-di-scala-tre
:alt: "Tre grafici affiancati, tutti in scala logaritmica su entrambi gli assi e senza numeri sugli assi. In ciascuno la loss cala come una retta discendente al crescere rispettivamente del calcolo, della quantità di dati e del numero di parametri; accanto a ogni retta l'esponente della legge di potenza: meno 0,050 per il calcolo, meno 0,095 per i dati, meno 0,076 per i parametri. Nessuna delle tre rette mostra un ginocchio o un punto di arresto."
:width: 100%

Tre risorse, tre rette. Gli assi sono in scala logaritmica: un passo lungo
l'asse non aggiunge una quantità, la moltiplica per dieci. Una retta che scende
vuol dire quindi che per guadagnare ancora un poco bisogna moltiplicare la
risorsa, non aggiungerne un pezzetto. Il numeretto scritto accanto a ogni retta
è la sua pendenza vera, e dice quanto rende quella moltiplicazione: prendendo
dieci volte tanto, la loss resta a 0,89 di quanto era, poco meno di nove
decimi, per il calcolo (primo pannello, numeretto 0,050); a 0,80, poco più di
quattro quinti, per i dati (secondo pannello, 0,095); a 0,84, circa cinque
sesti, per i parametri (terzo, 0,076). Più il numeretto è grande, più quella
risorsa rende, e le tre rette sono disegnate con inclinazioni proporzionali ai
tre numeretti: quella dei dati è la più ripida perché 0,095 è il più grande dei
tre. Schema ridisegnato sugli esponenti misurati da Kaplan e colleghi nel 2020;
la $C$ del primo pannello è il budget di calcolo **allocato al meglio** (il
$C_{\min}$ del paper), cioè speso nella combinazione di taglia del modello e
durata dell'addestramento che con quel budget rende di più. Non è il calcolo
comunque impiegato, che il paper misura a parte e con un altro esponente.
```

L'assenza di un ginocchio in {numref}`fig-leggi-di-scala-tre` è ciò che ha
orientato gli anni che sono seguiti, ed è anche il suo limite: quelle rette
raccontano soltanto il tratto che qualcuno ha davvero misurato. Che prima o poi
la discesa debba fermarsi lo sappiamo per principio, e lo scrivono gli autori
stessi di quelle misure: in un testo scritto da esseri umani c'è un tanto di
imprevedibilità che nessun modello potrà mai togliere, e quello è il pavimento.
Quello che nessuno sa è a che altezza sia. È bene tenerlo a mente in quel che
segue.

Questo capitolo, nella sua prima stesura, indicava come "direzioni future" tre
cose: i modelli di fondazione (in inglese *foundation model*: uno solo, enorme,
riadattato a mille compiti), la multimodalità (un modello solo che tratta testo,
immagini e suono) e gli agenti. Nel frattempo sono entrati nel libro, anche se
non nella forma prevista. La multimodalità ha un capitolo suo ([visione e
linguaggio](../VisioneLinguaggio/overview.md)); gli
[agenti](../Agenti/overview.md) ne hanno uno, e altri due sono cresciuti
accanto a quello ([prompt, contesto e loop](../IngegneriaLLM/overview.md) e i
[sistemi multi-agente](../SistemiMultiAgente/overview.md)).

I modelli di fondazione, in compenso, hanno fatto una fine più curiosa: non
sono diventati né un capitolo né una sezione. Si sono sciolti dentro quello sui
Transformer, dove [addestrare un modello su tutto il testo del
web](../Transformers/llm.md) e [adattarlo poi a quello che deve
fare](../Transformers/post-training.md) sono due sezioni separate. La cosa c'è
ancora, e conta più che mai; è il nome che ha smesso di servire, perché quando
tutti i modelli si costruiscono così non c'è più niente da distinguere.

Tre previsioni su tre, quindi, hanno indovinato l'argomento; nessuna delle tre
la forma che avrebbe preso, e una si è dissolta perfino come nome. È il motivo
per cui qui sotto non troverai profezie ma i fronti su cui si lavora davvero:
dire su che cosa si sta lavorando è un'affermazione molto più piccola che dire
come andrà a finire, e si può controllare.

Con un avvertimento, perché questa è la sezione più deperibile del libro: è
scritta al presente, e il presente a cui si riferisce è quello della versione
che stai leggendo. Si rilegge a ogni pubblicazione, ed è normale che invecchi
prima del resto.

`````{tab} Elementare
Ecco la cosa che quel nome teneva insieme, e che ormai fanno tutti: non si
addestra più un modello nuovo per ogni problema. Se ne addestra *uno solo*,
enorme, su una montagna di testo o immagini, e poi lo si adatta a mille compiti
diversi con poco sforzo, o riaddestrandolo un altro po’ su qualche migliaio di
esempi del compito nuovo, o semplicemente spiegandogli a parole che cosa
vogliamo. Una base unica su cui si costruisce tutto, un po’ come una persona
con una solida cultura generale che, con una breve formazione, impara mestieri
molto diversi.

Le domande aperte, oggi, sono più concrete di quelle di una volta, quando ci si
chiedeva ancora se una macchina potesse riconoscere un gatto in una foto: a
quella si è risposto, a queste no. Sono quattro, e hanno una cosa in comune:
in nessuna delle quattro basta fare più grande.

**Quanto costa.** Per capire un testo, un Transformer confronta fra loro tutte
le sue parole, e quel confronto costa quanto il quadrato della lunghezza:
raddoppia il testo e quel pezzo di conto si moltiplica per quattro. Su un testo
corto è una spesa fra le tante; su un testo lunghissimo diventa la più grossa
di tutte, in tempo di calcolo e quindi in bolletta. C'è una gara in corso per
pagare meno.

**Se capisce o indovina.** Un modello che risponde bene non è per forza un
modello che ha capito, e per saperlo bisogna aprirlo e guardarci dentro. Non è
facile: quello che ha imparato non sta scritto in chiaro da nessuna parte, è
spalmato su miliardi di numeri, e un singolo pezzo di rete si accende per cose
che fra loro non c'entrano niente.

**Se ci si può fidare.** Un agente che agisce da solo per venti passi sbaglia
in modi che un programma che risponde a una domanda per volta non poteva
permettersi. Il conto lo puoi fare tu con una calcolatrice. Se ogni singolo
passo va bene 95 volte su 100, cioè con probabilità 0,95, e se i venti passi
non si influenzano fra loro, la probabilità che vadano bene *tutti e venti* è
0,95 elevato alla ventesima, cioè venti fattori 0,95 moltiplicati fra loro: si
moltiplica perché ogni passo aggiunge una condizione da soddisfare. Viene
0,3585, appena 36 volte su 100. Nella pratica va meglio, perché un agente può
accorgersi di uno sbaglio e tornare sui suoi passi, ma l'ordine di grandezza è
quello, ed è la ragione per cui i compiti lunghi restano difficili.

**Quanto consuma.** Addestrare e far girare questi modelli costa corrente,
acqua per raffreddare i calcolatori e chip che sanno fabbricare pochissime
aziende al mondo. Non è un conto che si chiude in laboratorio: tocca le reti
elettriche, le riserve d'acqua e quelle poche fabbriche. E sotto sotto non è
una questione tecnica: è una questione di chi ha i mezzi per pagarlo.

E una scommessa, una sola: i [modelli del mondo](../WorldModels/overview.md).
Un modello normale impara che cosa viene di solito dopo che cosa; un modello
del mondo prova a imparare le regole con cui una cosa ne fa succedere un'altra,
e allora può immaginare come andrebbe a finire una mossa che non ha mai visto
fare. Se funzionasse su larga scala, un libro come questo cambierebbe ordine:
i modelli del mondo diventerebbero il capitolo da cui si parte, non l'ultimo
arrivato.
`````

`````{tab} Superiore
I **foundation model** funzionano così: pre-addestramento auto-supervisionato
su corpora enormi, poi adattamento via fine-tuning o prompting
{cite}`bommasani2021opportunities`. Le *scaling laws* hanno mostrato che la
**cross-entropy loss** cala in modo prevedibile con parametri, dati e calcolo
{cite}`kaplan2020scaling`, e Hoffmann e colleghi ne hanno poi corretto la
conclusione operativa sull'allocazione fra parametri e dati
{cite}`hoffmann2022training`. Che a una loss più bassa corrispondano *capacità*
nuove è un'affermazione diversa, e più fragile: è la faccenda delle abilità
emergenti, che il capitolo sui
[Transformer](../Transformers/llm.md) discute con il dubbio, motivato, che
siano in buona parte un artefatto della metrica scelta. In ogni caso le leggi
di scala non promettono che *scalare* basti a risolvere tutto, e i quattro
fronti aperti sono, non per caso, quelli in cui scalare non basta.

**L'efficienza dell'attenzione.** Sui contesti lunghi il costo quadratico
dell'attenzione diventa il vincolo economico dominante. "Sui contesti lunghi" è
un'ipotesi con una soglia, e conviene calcolarla, perché la scrittura
asintotica la fa sembrare più vicina di quanto sia. Detta $n$ la lunghezza del
contesto in token e $d$ la dimensione del modello, per strato l'attenzione
costa $2n^2d$ moltiplicazioni: $n^2 d$ per i punteggi
$\mathbf{Q}\mathbf{K}^\top$ e altrettante per combinare i valori. Tutto il
resto ne costa $12nd^2$: $4nd^2$ per le quattro proiezioni (query, chiavi,
valori, uscita) e $8nd^2$ per il feedforward, la cui dimensione interna è per
convenzione $4d$. I due termini si pareggiano dove $2n^2d = 12nd^2$, cioè a
$n = 6d$; e in un decoder, dove la maschera causale rende inutile metà dei
punteggi, la soglia raddoppia a $n = 12d$, che è la regola scritta anche in
{cite}`kaplan2020scaling`. Con un $d$ di qualche migliaio siamo comunque a
decine di migliaia di token, e sotto quella soglia il collo di bottiglia
aritmetico sta altrove. Quello di memoria no: la cache di chiavi e valori, che
in inferenza cresce con la lunghezza, stringe molto prima, ed è un vincolo
diverso che conviene non confondere con questo. Da qui l'attenzione lineare e i
modelli a spazio di
stati, che sostituiscono l'attenzione softmax con calcoli in forma ricorrente,
il cui stato occupa una memoria costante nella lunghezza. Le architetture
ibride alternano i due tipi di strato: il costo resta quadratico, perché basta
una frazione costante di strati con attenzione piena, ma la costante davanti è
più piccola.

**La comprensione del modello.** L'interpretabilità meccanicistica prova a
leggere i circuiti dentro i pesi, ed è lo strumento più diretto per distinguere
una risposta corretta da una risposta corretta *per il motivo giusto*, perché
guarda dentro il modello invece di fermarsi al comportamento.

**L'affidabilità degli agenti.** Componendo più passi gli errori si accumulano:
detta $p$ la probabilità di sbagliare un singolo passo e $T$ la lunghezza della
traiettoria, se i passi sono indipendenti e ogni errore è fatale la probabilità
di arrivare in fondo senza inciampi è $(1-p)^T$, che precipita. Le due ipotesi
vanno dichiarate, perché nessuna delle due vale per un agente vero: i passi
sono correlati, e riflessione e re-planning recuperano una parte degli errori.
E attenzione a come si chiama quel numero: non è un caso peggiore. Con le
probabilità di ogni singolo passo fissate, la correlazione può portare l'esito
ovunque fra $\max(0,\,1-Tp)$, se i fallimenti si escludono a vicenda, e $1-p$,
se cadono tutti insieme: con $p = 0{,}05$ e $T = 20$ è l'intervallo
$[0;\ 0{,}95]$, e il valore indipendente $0{,}3585$ sta comodamente in mezzo.
L'indipendenza è il caso di riferimento, non il peggiore. Resta però la morale,
e non basta essere bravi a un passo. Difficile è anche misurarlo: valutare un
agente vuol dire giudicare una traiettoria e non una risposta,
distinguendo il successo raggiunto per la strada giusta da quello arrivato per
caso.

**Il conto fisico.** Energia, acqua, silicio e la concentrazione di tutto
questo in pochi attori: un problema di politica industriale travestito da
problema tecnico.

E una direzione che è più una scommessa che una tendenza, una sola e dichiarata
come tale: i **modelli del mondo**, cioè imparare la dinamica dell'ambiente
invece delle sole correlazioni nei dati. Se funzionasse su larga scala,
cambierebbe l'ordine dei capitoli di un libro come questo.
`````

## Il campo di casa

Prima o poi la domanda arriva, di solito a cena: è più intelligente di noi?

Messa così non ha risposta, e non per prudenza: manca il *dove*. È come
chiedere se un pesce si muove meglio di un uomo. In acqua vince lui senza
sforzo; su un prato perde senza appello. Stessi due, risposta rovesciata.

Il posto dove queste macchine nuotano ha un nome preciso, ed è il **mondo
digitale**. Lì tutto è già nella forma che serve a loro. Ogni cosa è già un
numero, e non c'è da andare a misurarla. Ogni azione costa quasi niente e si
può ripetere un miliardo di volte. E il giudizio non bisogna chiederlo a
nessuno, perché sta già dentro il materiale: una partita finita dice chi ha
vinto, un programma o gira o non gira, e in una frase basta coprire una parola
e chiedere di indovinarla, che è il trucco visto nell'introduzione (con un
pezzo di immagine funziona uguale). Infine, sbagliare mentre si impara non
rompe niente: si ricomincia. Gli scacchi, il go, il codice dei programmi, il
testo, le immagini sono tutti mondi fatti di quella sostanza, ed è lì che sono
arrivati per primi i risultati che hanno fatto notizia. Torna il discorso
dell'aria: il mondo digitale è il posto dove quell'aria è più densa, e lì un
polmone respira a pieno.

Non è una gara alla pari, è una **partita in casa**. E la cosa da portarsi via
è che buona parte del vantaggio non viene dall'intelligenza, viene dal terreno.

Questa però è una lettura, non un risultato, e conviene dirlo: c'è chi la mette
in fila al contrario. Le curve della sezione «Dove sta andando», qui sopra, non
si piegano; chi le guarda risponde che il terreno spiega soltanto dove si è
potuto misurare per primi, e che una capacità cresciuta lì dentro poi esce e
serve anche fuori. Non si stabilisce chi ha ragione discutendone: si guarda che
cosa succede quando la partita si sposta all'aperto, ed è il resto di questa
sezione.

Fuori, il conto si rovescia, ed è un'osservazione vecchia di decenni. Un
computer batte il campione del mondo di scacchi dal 1997; costruire il braccio
che sposta i pezzi sulla scacchiera è rimasta la parte difficile. Nel 1988 Hans
Moravec lo mise così: dare a un computer le prestazioni di un adulto in un test
di intelligenza, o a dama, è relativamente facile; dargli le capacità di un
bambino di un anno nel percepire e nel muoversi è difficile o impossibile
{cite}`moravec1988mind`.

L'osservazione ha retto. La spiegazione che ne diede lui è un'altra cosa, e va
tenuta separata, perché nessuno l'ha mai messa alla prova. Moravec la
attribuiva ai tempi dell'evoluzione: nel vedere e nel muoverci abbiamo dietro
un miliardo di anni di mestiere, mentre il pensiero astratto è un trucco
recente, forse di meno di centomila anni. È un racconto che convince, ed è per
questo che gira. Arvind Narayanan, nel 2026, ha obiettato che quel paradosso
dice più su quali problemi la ricerca trovi interessanti che su quali siano
difficili davvero: i casi facili per tutti, e quelli difficili per tutti, non
li racconta nessuno {cite}`narayanan2026moravec`.

Questo libro quella differenza la spiega in un altro modo, ed è il terreno.
Dove il giudizio è già lì si impara in fretta; dove bisogna andarselo a
prendere nel mondo, no.
Per dire perché un braccio robotico faccia più fatica di un programma che gioca
a scacchi non serve tirare in ballo l'evoluzione: basta notare che al programma
la partita dice subito com'è andata, e al braccio no.

Da qui viene la tentazione di rilassarsi, perché a noi resterebbe il mondo
vero. È giusto a metà, e la metà che manca conta.

La prima cosa che manca è che **il campo si allarga**. Ogni sensore, ogni
telecamera, ogni pagamento tracciato prende un pezzo di mondo vero e lo
trasforma in numeri, cioè lo porta dentro casa loro. È l'altra faccia di quello
che l'introduzione chiamava scarto: ciò che lasciamo dietro non è solo l'aria
che respirano, è anche il terreno su cui giocano. La robotica è il tentativo di
portare la partita all'aperto, e lì il conto si vede a occhio nudo: una prova
dura il tempo vero che ci vuole, il braccio si consuma, e una caduta non si
annulla premendo un tasto. Non è una profezia sul fatto che la robotica resterà
indietro, perché le profezie scadono: è il motivo per cui lì ogni tentativo
costa più che al chiuso, e se un giorno costerà meno sarà perché uno di quei
tre pezzi è cambiato.

La seconda è che quello che resta nostro non è un **territorio**, è un
**mestiere**: un territorio si perde, e ce ne si accorge quando qualcun altro
ci sta già giocando.

Quel mestiere è fatto di due cose. La prima è **rispondere** di una scelta, e
in italiano quella parola ne vuol dire due: dare una risposta, e assumersi le
conseguenze. Una macchina fa la prima cosa benissimo e la seconda no. Non
perché le manchi qualcosa di misterioso: quando una decisione fa un danno,
davanti a chi l'ha subìto deve andarci qualcuno che possa scusarsi, risarcire e
cambiare le regole, e quel qualcuno è sempre una persona o un'organizzazione
fatta di persone. Vale anche per un'azienda, che di suo non ha faccia né
braccia: la responsabilità non si trova, si assegna, e la si assegna a chi può
portarla.

La seconda è **decidere quale partita giocare**. A un sistema si dà un
punteggio da far salire, e lui lo fa salire con una costanza che noi non
abbiamo: se il punteggio è «quanti minuti resti a guardare», diventerà bravo a
tenerti lì, e ci riuscirà. Se quello sia il numero giusto da far salire è una
domanda che sta fuori dal campo. Non è che la macchina la sbagli: è che lì
dentro non esiste.

Sapere dove si sta giocando è anche il modo migliore per leggere la prossima
notizia che ti capiterà sotto gli occhi. Prima di chiederti quanto sia brava,
chiediti se giocava in casa.

## Una nota onesta

Sarebbe disonesto chiudere con il solo entusiasmo. Questi sistemi hanno limiti
che non sono incidenti passeggeri in attesa della prossima versione: dipendono
da come sono fatti, e restano. Un modello dà per veri fatti che non esistono, e
si porta dietro i pregiudizi dei testi da cui ha imparato; qui sotto vediamo
perché nessuna delle due cose sia una sorpresa.

Prima però una domanda che si fa di rado: chi è nella posizione di
accorgersene? Dipende da come il modello viene messo a disposizione, e i modi
sono due. Di alcuni modelli si possono scaricare i **pesi**, cioè i numeri che
l'addestramento ha regolato, le manopole di prima: sono un file che chiunque
può tenersi, e quindi misurarne i difetti, sondarlo, smentirlo. Altri si
raggiungono solo attraverso un'interfaccia, cioè mandando domande al server di
chi li possiede e ricevendo risposte: quelli si verificano solo con il permesso
di quel qualcuno, e quel permesso può essere tolto. Non è una questione di
mercato: è una questione di chi può sapere cosa.

Attenzione però: quella distinzione non è una classifica di bravura
({numref}`fig-aperti-chiusi`).

```{figure} ../figures/open-weights-vs-closed.svg
:name: fig-aperti-chiusi
:alt: "Mappa a quadranti. Sull'asse orizzontale il grado di apertura di un modello: a sinistra i chiusi, raggiungibili solo da un'interfaccia; a destra gli aperti, di cui si scaricano i pesi. Sull'asse verticale la capacità, dal basso verso l'alto. I punti stanno in tutti e quattro i quadranti e nessuno dei due lati domina l'altro: ce ne sono di capaci fra i chiusi e fra gli aperti, e di contenuti in entrambi. I punti non portano nomi."
:width: 88%

Ogni punto è un modello. Due assi indipendenti: aperto non vuol dire debole e
chiuso non vuol dire potente, e in tutti e quattro i quadranti c'è qualcuno. I
punti non hanno un nome di proposito: i nomi cambiano ogni pochi mesi, la forma
della nuvola no.
```

`````{tab} Elementare
Un modello linguistico non "sa" le cose: prevede la parola più probabile dopo
le precedenti. Funziona perché nei testi da cui ha imparato, il più delle
volte, le parole che seguono sono anche quelle giuste. Ma non sempre, e per
questo a volte inventa con perfetta sicurezza fatti falsi: le chiamiamo
*allucinazioni*. E siccome impara da testi scritti da noi, assorbe anche i
nostri pregiudizi: se i dati riflettono discriminazioni, il modello le ripete, e
a volte le rafforza, perché puntare sempre sulla risposta più frequente fa
sparire le eccezioni. Uno strumento potente non è uno strumento neutrale.
`````

`````{tab} Superiore
Un modello linguistico è pre-addestrato a massimizzare la verosimiglianza del
testo, non la verità: la fluidità di una frase non ne garantisce la
correttezza. A peggiorare le cose, la sicurezza con cui una risposta è
formulata non è un indicatore affidabile della sua correttezza; da qui le
allucinazioni sicure di sé.

I *bias* non sono un bug ma una conseguenza attesa: un modello addestrato su
corpora enormi e non curati ne assorbe la visione dominante e la amplifica
{cite}`bender2021dangers`. Conviene però non ridurli a una sola causa. In parte
i dati sotto-rappresentano qualcuno, e allora raccoglierne altri aiuta. In
parte, e peggio, i dati rappresentano fedelmente un mondo già iniquo: lì la
regolarità che il modello apprende *è* la disuguaglianza, e nessuna quantità di
dati aggiuntivi la corregge, perché il difetto sta nella definizione stessa
dell'obiettivo. La sezione sull’[equità e i
bias](../AIResponsabile/equita-e-bias.md) distingue quattro sorgenti proprio
perché richiedono rimedi diversi {cite}`mehrabi2021survey`, e due di quelle
quattro non si aggiustano con i dati.

A valle restano questioni aperte: impatto ambientale dell'addestramento,
concentrazione di potere in pochi attori, effetti sul lavoro e
sull'informazione. L'AI Act europeo, entrato in vigore nel 2024, è un primo
tentativo di regolarle. Il fact-check umano, per noi, non è opzionale.
`````

## Come continuare a imparare

Questo libro è una mappa, non il territorio. Per proseguire: leggi i **paper**
originali, cioè gli articoli scientifici in cui ciascuna di queste idee è stata
proposta per la prima volta. Quelli degli ultimi quindici anni stanno quasi
tutti su **arXiv**, l'archivio pubblico e gratuito dove i ricercatori
depositano i propri lavori. I più vecchi no: Weizenbaum, Rosenblatt, Hopfield
sono su riviste, e in biblioteca. E quello che sta su arXiv non è tutto uguale:
alcuni sono la versione definitiva di un articolo già passato da una revisione
fra pari, altri sono *preprint* che nessuno ha ancora letto oltre a chi li ha
scritti, e da fuori i due si distinguono male. Vanno letti come va letto un
modello: senza prendere per buono niente solo perché è scritto bene.

E soprattutto *riproduci il codice*, perché un modello lo capisci quando lo fai
girare e lo rompi. Il metodo per farlo sta nel capitolo su PyTorch, nella
sezione su [come si replica un paper](../PyTorch/replicare-un-paper.md):
quattro mosse, e i controlli che contano si fanno senza nemmeno addestrare.

:::{only} html
Se è la prima volta e un paper ti sembra un altro pianeta, comincia da qui:
apri il notebook di un capitolo con il pulsante "Esegui il codice", cambia un
numero e guarda che cosa si rompe. È lo stesso mestiere, a un decimo della
fatica, e insegna più di una lettura.
:::

:::{only} latex
Se è la prima volta e un paper ti sembra un altro pianeta, comincia da qui:
prendi il codice di un capitolo, mandalo in esecuzione, cambia un numero e
guarda che cosa si rompe. È lo stesso mestiere, a un decimo della fatica, e
insegna più di una lettura.
:::

Tieni i classici a portata: Géron {cite}`geron2022hands` per la pratica,
Chollet {cite}`chollet2021deep` per l'intuizione, Goodfellow, Bengio e
Courville {cite}`goodfellow2016deep` per la teoria, la documentazione di
scikit-learn e PyTorch come compagne quotidiane. Un'avvertenza sui primi due:
quando arrivano al deep learning il codice è in Keras e TensorFlow, non in
PyTorch. Quello che insegnano non dipende dal framework, ma è meglio saperlo
prima di aprirli.

E, capitolo per capitolo, questo libro ha già in bibliografia i manuali di
riferimento, che sui rispettivi argomenti dicono molto più di un generalista:
Sutton e Barto {cite}`sutton2018reinforcement` per il reinforcement learning,
Jurafsky e Martin {cite}`jurafsky2026speech` per il linguaggio e la voce,
Szeliski {cite}`szeliski2022computer` per la visione, Hamilton
{cite}`hamilton2020graph` per i grafi, Hyndman e Athanasopoulos
{cite}`hyndman2021forecasting` per le serie temporali, Huyen
{cite}`huyen2022designing` per la messa in produzione, Molnar
{cite}`molnar2022interpretable` per l'interpretabilità, Barocas, Hardt e
Narayanan {cite}`barocas2023fairness` per l'equità. Diversi si leggono
integralmente e gratuitamente online. Sono tutti in inglese, come quasi tutta
la letteratura di questo campo: è una delle ragioni per cui questo libro esiste
in italiano.

Poi mettiti alla prova. **Kaggle** è il sito dove chiunque può misurarsi su un
problema di dati vero, con una classifica e il codice degli altri partecipanti
sotto gli occhi. Partecipa a una competizione: è il modo più rapido per vedere
la differenza fra un modello che gira sul tuo computer e un modello che regge
dati che non hai scelto tu. Contribuisci a un progetto **open source**, cioè a
un programma il cui codice è pubblico e chiunque può migliorarlo. E tieni un
quaderno degli esperimenti falliti, che insegnano più dei successi.

:::{only} html
E torna qui: questa versione del libro si aggiorna, e i capitoli nascono anche
dalle segnalazioni di chi legge. Se un passaggio non ti torna, selezionalo e
mandacelo: i pulsanti che compaiono servono esattamente a questo.
:::

:::{only} latex
E torna alla versione online: si aggiorna, e i capitoli nascono anche dalle
segnalazioni di chi legge. Se un passaggio non ti torna, selezionalo lì e
mandacelo: i pulsanti che compaiono servono esattamente a questo.
:::

## Un ultimo messaggio

Abbiamo scelto di raccontare l'intelligenza artificiale in italiano, su due
livelli, senza mai barare sulla difficoltà. Non perché l'inglese non basti,
ma perché crediamo che capire davvero una cosa significhi poterla spiegare
nella propria lingua: a un collega, a uno studente, a te stesso alle due di
notte davanti a un errore che non torna.

Se c'è un'eredità che vorremmo lasciarti, non è una libreria né
un'architettura: quelle invecchiano in fretta. È un modo di stare davanti a
questi strumenti: curiosità senza reverenza, entusiasmo senza fede, e
quell'onestà intellettuale che fa dire "non lo so,
verifichiamo" invece di "l'ha detto il modello".

E resterà vero quello che Weizenbaum aveva notato nello stesso paragrafo del
1966 da cui siamo partiti: quando il funzionamento di un programma viene
spiegato in modo abbastanza chiaro «l'incanto si sgretola», e chi guarda lo
sposta «dallo scaffale marcato *intelligente* a quello riservato alle
curiosità». È successo con ELIZA, e succede ancora: ecco perché l'intelligenza
artificiale non si lascia definire, visto che ogni pezzo capito smette di
sembrare intelligenza. Smettere di sembrare intelligenza, però, non è smettere
di funzionare, e nemmeno di contare. Per te queste macchine non sono più una
scatola nera: sai di che cosa sono fatte, dati, rappresentazioni,
ottimizzazione. Quello che resta chiuso non è più la macchina, è quello che ha
imparato, e adesso sai anche perché leggerlo sia un problema aperto. Il resto è
pratica.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Tutto il libro poggia su tre idee ricorrenti: i **dati** (nessun modello sa
  più di ciò che ha visto), le **rappresentazioni apprese** (le "lenti" che la
  rete si costruisce da sola, invece di farsele suggerire) e
  l’**ottimizzazione** (le manopole del mixer, regolate un pochino alla volta).
- Addestrare vuol dire quasi sempre la stessa cosa: misurare quanto il modello
  ha sbagliato e spostare ogni manopola nella direzione che riduce l'errore,
  finché non c'è più molto da guadagnare. Qualche famiglia ci arriva per
  un'altra strada (due reti che si sfidano, oppure imparare agendo invece che
  da esempi già pronti), ma anche lì si tratta di migliorare qualcosa, un passo
  alla volta.
- Le "direzioni future" invecchiano in fretta. Il modello di fondazione, i
  modelli che capiscono testo, immagini e suono tutti insieme e gli agenti
  erano previsioni scritte in questo stesso capitolo: oggi due sono capitoli
  del libro, e il modello di fondazione è diventato così normale da non avere
  più bisogno di un nome. Restano varianti delle stesse tre idee, non magia.
- I fronti davvero aperti sono quelli in cui **fare più grande non basta**: il
  costo dei testi lunghissimi, il capire *perché* un modello ha risposto così,
  la fiducia in un agente che lavora da solo per una ventina di passi, e il
  conto di corrente, acqua e chip che tutto questo presenta a qualcuno.
- Potenza e responsabilità crescono insieme: i fatti inventati con sicurezza e
  i pregiudizi ereditati dai dati sono limiti strutturali, e verificare a mano
  quello che il modello dice non è opzionale.
- Gli ultimi capitoli non parlano di architetture ma di **mestiere**: portare
  un modello in produzione, aprirlo per capire, rispondere delle sue
  conseguenze. È la parte che decide se quello che hai costruito serve o fa
  danni.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Tutto il libro poggia su tre idee ricorrenti: **dati**, **rappresentazioni
  apprese** e **ottimizzazione**.
- L'apprendimento è, per il grosso del libro, la minimizzazione di un rischio
  empirico, $\theta^\star = \arg\min_\theta \mathcal{L}(\theta)$. Le eccezioni
  sono istruttive e vanno tenute a mente: GAN (gioco minimax), reinforcement
  learning (dati che dipendono dalla politica) e modelli a energia
  (verosimiglianza intrattabile) restano problemi di ottimizzazione, con
  obiettivi di natura diversa.
- Le "direzioni future" invecchiano in fretta: foundation model, multimodalità
  e agenti erano previsioni scritte in questo capitolo, e oggi sono testo del
  libro (due capitoli, più altri due cresciuti attorno agli agenti; i foundation
  model non hanno nemmeno una sezione propria, si sono sciolti dentro quello sui
  Transformer). Restano varianti delle stesse tre idee, non magia.
- I fronti davvero aperti sono quelli in cui **scalare non basta**: il costo
  dell'attenzione sui contesti lunghi, l'interpretabilità, l'affidabilità degli
  agenti, il conto energetico e industriale.
- Potenza e responsabilità crescono insieme: allucinazioni e bias sono limiti
  strutturali, e il fact-check umano non è opzionale.
- Gli ultimi capitoli non parlano di architetture ma di **mestiere**:
  produzione, interpretabilità, responsabilità. È la parte che decide se
  quello che hai costruito serve o fa danni.
```
`````

Buon lavoro. E, come si dice in Italia, *in bocca al lupo*.
