# Conclusioni

Siamo partiti, nell'introduzione, da una frase di Joseph Weizenbaum: "si dice
che spiegare significhi dissolvere" {cite}`weizenbaum1966eliza`. Dopo
{{ n_capitoli_meno_uno_lettere }} capitoli, dall'algebra lineare alle reti che
generano immagini, dagli alberi di decisione agli agenti che usano strumenti,
dal codice che fa correre una scheda grafica alle domande su chi risponde
quando un modello sbaglia, possiamo dire com'è andata: dei meccanismi non è
rimasto quasi niente di prodigioso, e l'intelligenza artificiale continua a non
lasciarsi definire, perché ogni volta che una cosa si spiega smette di sembrare
intelligenza e diventa "solo" un algoritmo. Ma ora abbiamo qualcosa che
all'inizio non avevamo: gli strumenti per capire *come* queste macchine
funzionano davvero, e per distinguere ciò che sanno fare da ciò che sembrano
fare. Vale la pena ripercorrere la strada all'indietro, per vedere il disegno
che i singoli capitoli, da vicino, non lasciavano intravedere.

## Il percorso, guardato dall'alto

**Il primo tratto** sono i mattoni: vettori, matrici, derivate, probabilità.
Non erano un rito d'iniziazione, ma il vocabolario di tutto il resto. Un
neurone è un prodotto scalare, cioè pesa i suoi ingressi come i prezzi pesano
le quantità nel carrello della spesa e ne fa un totale; l'apprendimento è una
discesa lungo un gradiente, cioè lungo una pendenza; una previsione è una
distribuzione di probabilità, cioè non una risposta secca ma un elenco di
risposte possibili, ciascuna con la fiducia che il modello le assegna. Da lì il
machine learning classico ci ha insegnato la disciplina fondamentale: separare
*training* e *test*, temere l'overfitting (cioè l'imparare a memoria), misurare
con onestà. Una disciplina che vale identica per un modello con dieci parametri
(le manopole che l'addestramento regola) e per uno con mille miliardi.

**Il secondo tratto** sono le reti neurali, che hanno cambiato la scala e ci
hanno costretti a guardare anche sotto il cofano: con PyTorch abbiamo impilato
strati e scritto a mano il ciclo di addestramento (il *training loop*), e col
capitolo sulla [GPU](../GPU/overview.md) abbiamo visto perché una
moltiplicazione di matrici è veloce solo se la memoria collabora. Le reti
convoluzionali hanno insegnato alle macchine a vedere (lo spartiacque è
AlexNet, che nel 2012 vince la gara di riconoscimento di immagini ImageNet con
un margine che nessuno si aspettava); i modelli di sequenza e i Transformer
{cite}`vaswani2017attention` a leggere e scrivere, e poi a tenere insieme
[visione e linguaggio](../VisioneLinguaggio/overview.md) nello stesso modello;
e poi il suono, la voce, i grafi (i dati fatti di puntini collegati da linee, e
i [sistemi che ti raccomandano cosa
guardare](../SistemiRaccomandazione/overview.md)), le serie temporali, le
equazioni della fisica. Sono i capitoli in cui la stessa matematica cambia
mestiere a seconda della forma dei dati.

**Il terzo tratto** viene dopo il secondo non in ordine di tempo, ma perché
cambia la domanda: non più solo riconoscere quello che c'è (questa foto è un
gatto), ma generare quello che non c'è (una foto di un gatto che non esiste) e
decidere cosa farne. I modelli generativi: dalle GAN alla diffusione, e i
[modelli a energia](../ModelliEnergia/overview.md) che, guardati da vicino,
sono la diffusione scritta in un'altra lingua. Il [reinforcement
learning](../ReinforcementLearning/overview.md), dove l'agente non riceve le
risposte giuste ma le scopre agendo, come AlphaGo nel 2016. Le architetture
nate per non pagare il costo quadratico dell'attenzione (leggere un testo, per
un Transformer, costa quanto il quadrato della sua lunghezza), dall'[attenzione
lineare](../AttenzioneLineare/overview.md) ai [modelli a spazio di
stati](../StateSpaceModel/overview.md). E gli [agenti](../Agenti/overview.md)
che usano strumenti, il modo in cui li si [programma a
parole](../IngegneriaLLM/overview.md), quello che succede [quando diventano
molti](../SistemiMultiAgente/overview.md) e i [modelli del
mondo](../WorldModels/overview.md) che provano a immaginare le conseguenze di
un'azione prima di compierla.

`````{tab} Elementare
Attenzione a una cosa, però: "terzo tratto" non vuol dire "il più recente".
Diverse di queste idee sono vecchie quanto il libro le racconta, e più vecchie
delle reti che hanno reso famoso il deep learning. I modelli a energia nascono
dalla fisica dei primi anni Ottanta; l'impalcatura matematica del
reinforcement learning è degli anni Cinquanta, cioè di quando i calcolatori
occupavano una stanza. Quello che è successo a metà degli anni Dieci non è che
qualcuno abbia inventato queste idee: è che finalmente c'erano i dati e le
macchine per farle funzionare.
`````

`````{tab} Superiore
La cronologia va disaccoppiata dall'ordine dell'esposizione, perché il gruppo è
tutt'altro che coetaneo. Alcuni pezzi sono vecchissimi: la rete di Hopfield è
del 1982 {cite}`hopfield1982neural` e la macchina di Boltzmann del 1985
{cite}`ackley1985learning`, mentre l'equazione di Bellman, su cui poggia tutto
il reinforcement learning, è del 1957. Altri nascono a metà degli anni Dieci
(le GAN nel 2014, la diffusione fra il 2015 e il 2020, AlphaGo nel 2016) e
altri ancora prendono velocità dal 2020 in poi (attenzione lineare e modelli a
spazio di stati, gli agenti, i modelli del mondo).

Nemmeno l'ordine dell'indice è cronologico, e vale la pena dirlo perché
l'indice è la mappa che il lettore ha sotto gli occhi: il reinforcement
learning è una parte a sé, collocata prima del linguaggio e dei Transformer.
L'indice segue le domande, non le date, ed è la ragione per cui questo ripasso
ha un ordine e la storia ne ha un altro.
`````

E poi gli ultimi capitoli, che non parlano di architetture ma di **mestiere**: portare un modello in produzione e tenerlo in vita ([MLOps](../MLOps/overview.md)), aprirlo per capire perché ha deciso così ([interpretabilità](../Interpretabilita/overview.md)), e rispondere delle sue conseguenze ([AI responsabile](../AIResponsabile/overview.md)). Non sono appendici morali messe in fondo per buona educazione: sono la parte del lavoro che decide se quello che hai costruito serve a qualcuno o fa danni.

Argomenti diversissimi, e in mezzo quasi settant'anni di storia: dal
percettrone di Rosenblatt del 1958 ai modelli di oggi. Eppure, sotto, sempre le
stesse tre idee.

## Tre fili, un solo tessuto

```{figure} ../figures/fili-conduttori.svg
:name: fig-fili-conduttori
:alt: "Un asse orizzontale mostra l'arco del libro con cinque tappe (Matematica, Machine learning, Reti neurali, Deep learning, Generativi e RL); sotto, tre linee parallele colorate rappresentano i fili conduttori: Dati, Rappresentazioni, Ottimizzazione."
:width: 90%

L'arco dei modelli cambia da sinistra a destra (via via capaci di
rappresentare cose più complicate) ma sotto scorrono sempre gli stessi tre
fili. I capitoli sul mestiere (produzione, interpretabilità, responsabilità)
non stanno su questo asse: stanno attorno a tutto.
```

Se dovessimo comprimere l'intero libro in tre parole, sarebbero **dati**, **rappresentazioni** e **ottimizzazione** ({numref}`fig-fili-conduttori`). Sono i fili che attraversano ogni capitolo, dal più elementare al più avanzato.

I **dati** sono il carburante: nessun modello sa più di ciò che ha visto. Le **rappresentazioni apprese** sono il cuore della rivoluzione del deep learning.

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
Una rete profonda è una funzione composta $f_\theta = f^{(L)} \circ \dots \circ f^{(1)}$ che trasforma l'input grezzo $X$ in una sequenza di rappresentazioni intermedie sempre più astratte. Gli strati nascosti non sono altro che *feature apprese*: coordinate in uno spazio latente dove esempi semanticamente simili finiscono vicini. È il principio degli *embedding*, e la ragione per cui una sola rete pre-addestrata si riadatta a molti compiti.

Il feature engineering manuale del machine learning classico non è sparito, ma
va detto dove è finito, perché le destinazioni sono due e il libro le insegna
in capitoli diversi. Una parte è stata assorbita dentro $\theta$ e delegata
all'ottimizzazione, ed è la parte che si racconta di solito. L'altra si è
spostata nell'**architettura**, come bias induttivo: la convoluzione dichiara
che la posizione assoluta non conta, una rete su grafo che l'ordine dei nodi
non conta, e sono proprietà della forma della funzione, vere per ogni valore di
$\theta$ e anche a rete non addestrata. Non sono state imparate: sono state
scritte a mano dal progettista, prima che l'ottimizzazione cominciasse.
`````

E l'**ottimizzazione** è il motore che rende tutto questo possibile: apprendere significa cercare i parametri che minimizzano un errore.

`````{tab} Elementare
Immagina di regolare le manopole di un vecchio mixer per far suonare bene una canzone. Giri un po' una manopola, ascolti se è migliorato, correggi. Addestrare un modello è la stessa cosa, con milioni di manopole: a ogni passo il modello guarda quanto ha sbagliato e sposta ciascuna manopola nella direzione che riduce l'errore, un pochino. Ripetuto abbastanza volte, funziona.

Con un mixer vero la direzione giusta la scopri provando, e con milioni di
manopole non finiresti mai. Il modello non prova: la **calcola**. Per ogni
manopola si chiede "se la giro di un pelo in su, l'errore sale o scende, e di
quanto?", e ottiene tutte le risposte insieme con un conto solo, invece che
con milioni di tentativi. Quel conto è il gradiente, e senza di esso niente di
tutto questo sarebbe possibile.
`````

`````{tab} Superiore
La forma che copre il grosso del libro è la minimizzazione del **rischio
empirico** su un campione etichettato:

$$
\theta^\star = \arg\min_{\theta}\ \mathcal{L}(\theta)
= \arg\min_{\theta}\ \frac{1}{m}\sum_{i=1}^{m} \ell\big(f_\theta(X^{(i)}),\, y^{(i)}\big),
$$

dove $\theta$ sono i parametri, $\mathcal{L}$ la funzione di costo, $\ell$ la
perdita sul singolo esempio (nel capitolo sul machine learning quel ruolo lo
aveva $\mathcal{L}$, che qui passa a indicare il totale), $f_\theta$ il
modello, e $X^{(i)}$, $y^{(i)}$ l'input e il target dell'$i$-esimo degli $m$
esempi di addestramento. Cambia $f_\theta$ e, per i modelli differenziabili
(dalla regressione lineare al Transformer), la macchina che risolve è la
discesa del gradiente stocastica.

Il perimetro di quella scrittura, però, va dichiarato, perché è più stretto del
libro, e le eccezioni sono istruttive. **Le [GAN](../GAN/overview.md)** non ci
stanno: l'ottimizzazione simultanea di un gioco minimax non equivale a
minimizzare una singola funzione, ed è precisamente la ragione per cui sono
instabili. **Il [reinforcement
learning](../ReinforcementLearning/overview.md)** non ci sta: non esiste un
campione fisso di $m$ coppie, la distribuzione dei dati dipende dalla politica
che si sta cercando, e l'obiettivo è massimizzare un ritorno atteso. **I
[modelli a energia](../ModelliEnergia/overview.md)** non ci stanno: la
verosimiglianza che vorrebbero massimizzare contiene una funzione di partizione
che non si sa calcolare, e si ripiega su surrogati come lo score matching. E
nemmeno sul versante classico la copertura è totale: $k$-means minimizza
l'inerzia, ma DBSCAN è una procedura sulla densità e il clustering gerarchico
una fusione greedy, e nessuno dei due è il minimo di un obiettivo globale;
anche un albero di decisione cresce con split localmente ottimi, non
minimizzando una funzione sull'albero finito.

Resta vero che sono tutti problemi di **ottimizzazione**, ed è questo che tiene
insieme il libro. Quello che cambia da una famiglia all'altra è la natura
dell'obiettivo (una somma su un campione, un equilibrio fra due giocatori, un
ritorno atteso lungo traiettorie che il modello stesso genera, una
verosimiglianza inaccessibile), non il fatto che ce ne sia uno. Tre idee,
infinite architetture.
`````

## Dove sta andando

I tre grafici qui sotto rispondono a una domanda semplice: quanto migliora un
modello se gli diamo più potenza di calcolo, più dati o più parametri, cioè
le manopole da regolare? Ogni retta mostra l'errore che il modello commette (la
*loss*: più è bassa, meglio è) al crescere di una delle tre risorse, mentre le
altre due abbondano. E in nessuno dei tre compare un "ginocchio", cioè un punto
in cui il miglioramento si ferma, almeno fin dove le misure arrivano.

```{figure} ../figures/scaling-laws-2020.svg
:name: fig-leggi-di-scala-tre
:alt: "Tre grafici affiancati, tutti in scala logaritmica su entrambi gli assi. In ciascuno la loss cala come una retta discendente al crescere rispettivamente della potenza di calcolo, della quantità di dati e del numero di parametri: nessuna delle tre curve mostra un ginocchio o un punto di arresto nell'intervallo misurato."
:width: 100%

Tre risorse, tre rette. Gli assi sono in scala logaritmica: ogni tacca vale
dieci volte la precedente, non una in più. Una retta che scende vuol dire
quindi che per guadagnare ancora un poco bisogna moltiplicare la risorsa, non
aggiungerne un pezzetto. E ogni retta è misurata mentre le altre due risorse
abbondano: se ne manca una, la discesa si ferma lì, per colpa di quella.
Ridisegnata dai dati di {cite}`kaplan2020scaling`; il primo pannello riguarda
il calcolo speso nel modo migliore possibile, non il calcolo grezzo.
```

L'assenza di un ginocchio in {numref}`fig-leggi-di-scala-tre` è ciò che ha
orientato gli anni che sono seguiti, ed è anche il suo limite: quelle rette
raccontano solo il tratto che qualcuno ha davvero provato, e più in là nessuno
sa. È bene tenerlo a mente leggendo le pagine che seguono.

Questo capitolo, nella sua prima stesura, indicava come "direzioni future" i
modelli di fondazione (in inglese *foundation model*: uno solo, enorme,
riadattato a mille compiti), la multimodalità e gli agenti. Nel frattempo sono
entrati nel libro, anche se non nella forma prevista: la multimodalità ha un
capitolo suo ([visione e linguaggio](../VisioneLinguaggio/overview.md)), gli
[agenti](../Agenti/overview.md) ne hanno uno e altri due sono cresciuti
accanto a quello ([prompt, contesto e loop](../IngegneriaLLM/overview.md) e i
[sistemi multi-agente](../SistemiMultiAgente/overview.md)); i modelli di
fondazione, invece, non sono diventati un capitolo ma una sezione, dentro
quello sui [Transformer](../Transformers/llm.md). È il modo più onesto di dire
quanto corre il campo, e la ragione per cui qui non troverai profezie, ma i
fronti su cui si lavora davvero.

Con un avvertimento, perché questa è la sezione più deperibile del libro: è
scritta al presente, e il presente a cui si riferisce è il numero di versione
che trovi in cima all'indice. Si rilegge a ogni pubblicazione, ed è normale che
invecchi prima del resto.

`````{tab} Elementare
La novità che ha cambiato tutto resta questa: non addestriamo più un modello
nuovo per ogni problema. Ne addestriamo *uno solo*, enorme, su una montagna di
testo o immagini, e poi lo adattiamo a mille compiti diversi con poco sforzo:
o riaddestrandolo un altro po' su qualche migliaio di esempi del compito nuovo,
o semplicemente spiegandogli a parole che cosa vogliamo. È il modello "di
fondazione": un'unica base su cui si costruisce tutto, un po' come una persona
con una solida cultura generale che, con una breve formazione, impara mestieri
molto diversi.

Le domande aperte sono più concrete di quelle di una volta, quando ci si
chiedeva ancora se una macchina potesse riconoscere un gatto in una foto: a
quella si è risposto, a queste no. Sono quattro.

**Quanto costa.** Leggere un testo lunghissimo, per un Transformer, costa
quanto il quadrato della sua lunghezza: raddoppia il testo e il conto si
moltiplica per quattro, in tempo di calcolo e quindi in bolletta. C'è una gara
in corso per pagare meno.

**Se capisce o indovina.** Un modello che risponde bene non è per forza un
modello che ha capito, e per saperlo bisogna aprirlo e guardarci dentro.

**Se ci si può fidare.** Un agente che agisce da solo per venti passi sbaglia
in modi che un chatbot non poteva permettersi, e il conto lo puoi fare tu: se
ogni singolo passo va bene 95 volte su 100, venti passi di fila vanno tutti e
venti bene solo 36 volte su 100. Nella pratica va un po' meglio, perché un
agente può accorgersi di uno sbaglio e tornare sui suoi passi, ma l'ordine di
grandezza è quello, ed è la ragione per cui i compiti lunghi restano difficili.

**Quanto consuma.** Addestrare e far girare questi modelli costa corrente,
acqua per raffreddare i calcolatori e chip che sanno fabbricare pochissime
aziende al mondo. È un conto che non si paga in laboratorio, e sotto sotto non
è una questione tecnica: è una questione di chi ha i mezzi per farlo.

E una scommessa, una sola: i [modelli del mondo](../WorldModels/overview.md),
cioè macchine che invece di imparare come vanno di solito le cose imparano come
*funzionano*, e possono quindi immaginare che cosa succederebbe se. Se
funzionasse su larga scala, l'ordine dei capitoli di un libro come questo
cambierebbe.
`````

`````{tab} Superiore
I **foundation model** {cite}`bommasani2021opportunities` funzionano così:
pre-addestramento auto-supervisionato su corpora enormi, poi adattamento via
fine-tuning o prompting. Le *scaling laws* {cite}`kaplan2020scaling` hanno
mostrato che la **cross-entropy loss** cala in modo prevedibile con parametri,
dati e calcolo, e {cite}`hoffmann2022training` ne ha poi corretto la
conclusione operativa sull'allocazione fra parametri e dati. Che a una loss più
bassa corrispondano *capacità* nuove è un'affermazione diversa, e più fragile:
è la faccenda delle abilità emergenti, che il capitolo sui
[Transformer](../Transformers/llm.md) discute con il dubbio, motivato, che
siano in buona parte un artefatto della metrica scelta. In ogni caso le leggi
di scala non promettono che *scalare* basti a risolvere tutto, e i quattro
fronti aperti sono, non per caso, quelli in cui scalare non basta.

**L'efficienza dell'attenzione.** Sui contesti lunghi il costo quadratico
dell'attenzione (e la cache di chiavi e valori, che in inferenza cresce con la
lunghezza) diventa il vincolo economico dominante. "Sui contesti lunghi" è
un'ipotesi con una soglia, e vale la pena dirla: il termine $O(n^2 d)$
dell'attenzione supera quello delle proiezioni e del feedforward, che è
$O(n d^2)$, solo quando $n$ si avvicina a $d$; sotto quella soglia il collo di
bottiglia sta altrove. Da qui l'attenzione lineare e i modelli a spazio di
stati, che sostituiscono l'attenzione softmax con calcoli in forma ricorrente,
il cui stato occupa una memoria costante nella lunghezza. Le architetture
ibride alternano i due tipi di strato, e restano quindi quadratiche a tratti.

**La comprensione del modello.** L'interpretabilità meccanicistica prova a
leggere i circuiti dentro i pesi, ed è lo strumento più diretto per distinguere
una risposta corretta da una risposta corretta *per il motivo giusto*, perché
guarda dentro il modello invece di fermarsi al comportamento.

**L'affidabilità degli agenti.** Componendo più passi gli errori si accumulano:
nel caso peggiore, con passi indipendenti e ogni errore fatale, la probabilità
di una traiettoria pulita è $(1-p)^n$, che precipita. Le due ipotesi vanno
dichiarate, perché nessuna delle due vale per un agente vero (i passi sono
correlati, e riflessione e re-planning ne recuperano una parte), quindi quel
conto è un caso peggiore e non una previsione; ma la morale regge, e non basta
essere bravi a un passo. Difficile è anche misurarlo: valutare un agente vuol
dire giudicare una traiettoria e non una risposta, distinguendo il successo
raggiunto per la strada giusta da quello arrivato per caso.

**Il conto fisico.** Energia, acqua, silicio e la concentrazione di tutto
questo in pochi attori: un problema di politica industriale travestito da
problema tecnico.

E una direzione che è più una scommessa che una tendenza, una sola e dichiarata
come tale: i **modelli del mondo**, cioè imparare la dinamica dell'ambiente
invece delle sole correlazioni nei dati. Se funzionasse su larga scala,
cambierebbe l'ordine dei capitoli di un libro come questo.
`````

## Una nota onesta

Sarebbe disonesto chiudere con il solo entusiasmo. Questi sistemi hanno limiti strutturali, cioè che dipendono da come sono fatti, non incidenti temporanei.

```{figure} ../figures/open-weights-vs-closed.svg
:name: fig-aperti-chiusi
:alt: "Mappa a quadranti. Sull'asse orizzontale il grado di apertura di un modello: a sinistra i chiusi, raggiungibili solo da un'interfaccia; a destra gli aperti, di cui si scaricano i pesi. Sull'asse verticale la capacità, dal basso verso l'alto. I punti stanno in tutti e quattro i quadranti e nessuno dei due lati domina l'altro: ce ne sono di capaci fra i chiusi e fra gli aperti, e di contenuti in entrambi. I punti non portano nomi."
:width: 88%

Due assi indipendenti. Aperto non vuol dire debole e chiuso non vuol dire
potente: sono scelte di distribuzione, e cambiano chi può verificare cosa. I
punti non hanno un nome di proposito: i nomi cambiano ogni pochi mesi, la forma
della nuvola no.
```

Che i due assi siano indipendenti dice una cosa, ed è quella della didascalia.
L'asse orizzontale, da solo, ne dice un'altra, che riguarda direttamente i
limiti appena elencati. Un modello di cui si possono scaricare i pesi è un file
che chiunque può tenersi: chiunque può misurarne i difetti, sondarlo e
smentirlo. Un modello che si raggiunge solo attraverso un'interfaccia, cioè
mandando domande al server di chi lo possiede e ricevendo risposte, si può
verificare solo con il permesso di quel qualcuno, e quel permesso può essere
tolto. Non è una questione di mercato: è una questione di chi può sapere cosa.

`````{tab} Elementare
Un modello linguistico non "sa" le cose: prevede la parola più probabile dopo
le precedenti. Funziona perché nei testi da cui ha imparato, il più delle
volte, le parole che seguono sono anche quelle giuste. Ma non sempre, e per
questo a volte inventa con perfetta sicurezza fatti falsi: le chiamiamo
*allucinazioni*. E siccome impara da testi scritti da noi, assorbe anche i
nostri pregiudizi: se i dati storici riflettono discriminazioni, il modello le
ripete, a volte amplificandole. Uno strumento potente non è uno strumento
neutrale.
`````

`````{tab} Superiore
Un modello linguistico è pre-addestrato a massimizzare la verosimiglianza del testo, non la verità: la fluidità di una frase non ne garantisce la correttezza. A peggiorare le cose, la confidenza che il modello esprime è spesso mal calibrata; da qui le allucinazioni sicure di sé.

I *bias* non sono un bug ma una proprietà attesa dell'apprendimento statistico
{cite}`bender2021dangers`, e conviene non ridurli a una sola causa. In parte i
dati sotto-rappresentano qualcuno, e allora raccoglierne altri aiuta. In parte,
e peggio, i dati rappresentano fedelmente un mondo già iniquo: lì la regolarità
che il modello apprende *è* la disuguaglianza, e nessuna quantità di dati
aggiuntivi la corregge, perché il difetto sta nella definizione stessa
dell'obiettivo. Il capitolo sull'[AI
responsabile](../AIResponsabile/overview.md) distingue quattro sorgenti proprio
perché richiedono rimedi diversi, e due di quelle quattro non si aggiustano con
i dati.

A valle restano questioni aperte: impatto ambientale dell'addestramento, concentrazione di potere in pochi attori, effetti sul lavoro e sull'informazione. L'AI Act europeo (2024) è un primo tentativo di regolazione. Il fact-check umano, per noi, non è opzionale.
`````

## Come continuare a imparare

Questo libro è una mappa, non il territorio. Per proseguire: leggi i paper
originali. Quasi tutti stanno su **arXiv**, l'archivio pubblico e gratuito dove
i ricercatori depositano i propri lavori, spesso prima che una rivista li abbia
esaminati. Sono *preprint*: molti sono ottimi, e nessuno è ancora stato
controllato da nessuno. Vanno letti con la stessa onestà che questo libro
chiede di usare davanti a un modello. E soprattutto *riproduci il codice* (un
modello lo capisci quando lo fai girare e lo rompi). Il metodo per farlo sta
nel capitolo su PyTorch, nella sezione su [come si replica un
paper](../PyTorch/replicare-un-paper.md): quattro mosse e tre verifiche che si
fanno senza nemmeno addestrare.

Se è la prima volta e un paper ti sembra un altro pianeta, comincia da qui: apri
il notebook di un capitolo con il pulsante "Esegui il codice", cambia un numero
e guarda che cosa si rompe. È lo stesso mestiere, a un decimo della fatica, e
insegna più di una lettura.

Tieni i classici a portata: Géron {cite}`geron2022hands` per la pratica,
Chollet {cite}`chollet2021deep` per l'intuizione, Goodfellow, Bengio e
Courville {cite}`goodfellow2016deep` per la teoria, la documentazione di
scikit-learn e PyTorch come compagne quotidiane. Un'avvertenza sui primi due:
il loro codice è in Keras, non in PyTorch. Quello che insegnano non dipende dal
framework, ma è meglio saperlo prima di aprirli.

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

Poi mettiti alla prova: partecipa a una competizione su Kaggle (il sito dove
chiunque può misurarsi su un problema di dati vero, con una classifica e il
codice degli altri partecipanti sotto gli occhi), contribuisci a un progetto
open source, tieni un quaderno degli esperimenti falliti. Insegnano più dei
successi.

E torna qui: questa versione del libro si aggiorna, e i capitoli nascono anche
dalle segnalazioni di chi legge. Se un passaggio non ti torna, selezionalo e
mandacelo: il pulsante che compare serve esattamente a questo.

## Un ultimo messaggio

Abbiamo scelto di raccontare l'intelligenza artificiale in italiano, con due
registri, senza mai barare sulla difficoltà. Non perché l'inglese non basti,
ma perché crediamo che capire davvero una cosa significhi poterla spiegare
nella propria lingua: a un collega, a uno studente, a te stesso alle due di
notte davanti a un errore che non torna.

Se c'è un'eredità che vorremmo lasciarti, non è una libreria né
un'architettura: quelle invecchiano in fretta. È un modo di stare davanti a
questi strumenti: curiosità senza reverenza, entusiasmo senza fede, e
quell'onestà intellettuale che fa dire "non lo so,
verifichiamo" invece di "l'ha detto il modello".

E resterà vero quello che Weizenbaum aveva notato per primo, nella stessa
pagina del 1966 da cui siamo partiti: quando il funzionamento di un programma
viene spiegato in modo abbastanza chiaro, "la sua magia si sbriciola", e chi
guarda lo sposta "dallo scaffale marcato *intelligente* a quello riservato alle
curiosità". È successo con ELIZA, e succede ancora: è la ragione per cui
l'intelligenza artificiale non si lascia definire, perché ogni volta che ne
capiamo un pezzo quel pezzo smette di contare. Ma per te queste macchine non
sono più una scatola nera: sai di che cosa sono fatte, dati, rappresentazioni e
ottimizzazione, e sai anche perché leggere davvero dentro un modello addestrato
resti un problema aperto. Il resto è pratica.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Tutto il libro poggia su tre idee ricorrenti: i **dati** (nessun modello sa
  più di ciò che ha visto), le **rappresentazioni apprese** (le "lenti" che la
  rete si costruisce da sola, invece di farsele suggerire) e
  l'**ottimizzazione** (le manopole del mixer, regolate un pochino alla volta).
- Addestrare vuol dire quasi sempre la stessa cosa: misurare quanto il modello
  ha sbagliato e spostare ogni manopola nella direzione che riduce l'errore,
  finché non c'è più molto da guadagnare. Qualche famiglia di modelli fa
  diversamente (chi mette due reti a giocare una contro l'altra, chi impara
  agendo invece che da esempi già pronti), ma anche lì si tratta di migliorare
  qualcosa, un passo alla volta.
- Le "direzioni future" invecchiano in fretta: il modello di fondazione (uno
  solo, enorme, riadattato a mille compiti), i modelli che capiscono testo,
  immagini e suono tutti insieme e gli agenti erano previsioni scritte in
  questo stesso capitolo, e oggi sono testo del libro. Restano varianti delle
  stesse tre idee, non magia.
- I fronti davvero aperti sono quelli in cui **fare più grande non basta**: il
  costo dei testi lunghissimi, il capire *perché* un modello ha risposto così,
  la fiducia in un agente che lavora da solo per venti passi, e il conto di
  corrente, acqua e chip che tutto questo presenta a qualcuno.
- Potenza e responsabilità crescono insieme: i fatti inventati con sicurezza e
  i pregiudizi ereditati dai dati sono limiti strutturali, e verificare a mano
  quello che il modello dice non è opzionale.
- Gli ultimi capitoli non parlano di architetture ma di **mestiere**, portare
  un modello in produzione, aprirlo per capire, rispondere delle sue
  conseguenze: è la parte che decide se quello che hai costruito serve o fa
  danni.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Tutto il libro poggia su tre idee ricorrenti: **dati**, **rappresentazioni apprese** e **ottimizzazione**.
- L'apprendimento è, per il grosso del libro, la minimizzazione di un rischio
  empirico, $\theta^\star = \arg\min_\theta \mathcal{L}(\theta)$. Le eccezioni
  sono istruttive e vanno tenute a mente: GAN (gioco minimax), reinforcement
  learning (dati che dipendono dalla politica) e modelli a energia
  (verosimiglianza intrattabile) restano problemi di ottimizzazione, con
  obiettivi di natura diversa.
- Le "direzioni future" invecchiano in fretta: foundation model, multimodalità
  e agenti erano previsioni scritte in questo capitolo, e oggi sono testo del
  libro (due capitoli, più altri due cresciuti attorno agli agenti, e una
  sezione sui modelli di fondazione). Restano varianti delle stesse tre idee,
  non magia.
- I fronti davvero aperti sono quelli in cui **scalare non basta**: il costo
  dell'attenzione sui contesti lunghi, l'interpretabilità, l'affidabilità degli
  agenti, il conto energetico e industriale.
- Potenza e responsabilità crescono insieme: allucinazioni e bias sono limiti
  strutturali, e il fact-check umano non è opzionale.
- Gli ultimi capitoli non parlano di architetture ma di **mestiere**,
  produzione, interpretabilità, responsabilità: è la parte che decide se
  quello che hai costruito serve o fa danni.
```
`````

Buon lavoro. E, come si dice qui, *in bocca al lupo*.
