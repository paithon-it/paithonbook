# Conclusioni

Siamo partiti, nell'introduzione, da una frase di Joseph Weizenbaum:
l'intelligenza artificiale è "straordinariamente resistente al tentativo di
una precisa definizione". Dopo {{ n_capitoli_meno_uno_lettere }} capitoli,
dall'algebra lineare alle reti che generano immagini, dagli alberi di
decisione agli agenti che usano strumenti, dai kernel CUDA alle domande su chi
risponde quando un modello sbaglia: quella resistenza non è scomparsa. Ma ora
abbiamo qualcosa che all'inizio non avevamo: gli strumenti per capire *come*
queste macchine funzionano davvero, e per distinguere ciò che sanno fare da
ciò che sembrano fare. Vale la pena ripercorrere la strada all'indietro, per
vedere il disegno che i singoli capitoli, da vicino, non lasciavano
intravedere.

## Il percorso, guardato dall'alto

Abbiamo cominciato dai mattoni: vettori, matrici, derivate, probabilità. Non
erano un rito d'iniziazione, ma il vocabolario di tutto il resto: un neurone è
un prodotto scalare, l'apprendimento è una discesa lungo un gradiente, una
previsione è una distribuzione di probabilità. Da lì il machine learning
classico ci ha insegnato la disciplina fondamentale: separare *training* e
*test*, temere l'overfitting, misurare con onestà. Una disciplina che vale
identica per un modello con dieci parametri e per uno con mille miliardi.

Poi le reti neurali hanno cambiato la scala, e abbiamo dovuto guardare anche sotto il cofano: con PyTorch abbiamo impilato strati e scritto a mano il training loop, e col capitolo sulla [GPU](../GPU/overview.md) abbiamo visto perché una moltiplicazione di matrici è veloce solo se la memoria collabora. Le reti convoluzionali hanno insegnato alle macchine a vedere (ImageNet, 2012, è lo spartiacque); i modelli di sequenza e i Transformer {cite}`vaswani2017attention` a leggere e scrivere; e poi il suono, la voce, i grafi, le serie temporali, le equazioni della fisica. Sono i capitoli in cui la stessa matematica cambia mestiere a seconda della forma dei dati.

Il terzo tratto è quello che, dal 2020 in poi, ha spostato il campo. I modelli
generativi: dalle GAN alla diffusione, e i [modelli a
energia](../ModelliEnergia/overview.md) che, guardati da vicino, sono la
stessa cosa scritta in un'altra lingua, e il reinforcement learning, dove
l'agente non riceve le risposte giuste ma le scopre agendo, come AlphaGo nel
2016. Le architetture nate per non pagare il costo quadratico dell'attenzione,
dall'[attenzione lineare](../AttenzioneLineare/overview.md) ai [modelli a
spazio di stati](../StateSpaceModel/overview.md). Gli
[agenti](../Agenti/overview.md) che usano strumenti, e i [modelli del
mondo](../WorldModels/overview.md) che provano a immaginarne le conseguenze.

E poi gli ultimi capitoli, che non parlano di architetture ma di **mestiere**: portare un modello in produzione e tenerlo in vita ([MLOps](../MLOps/overview.md)), aprirlo per capire perché ha deciso così ([interpretabilità](../Interpretabilita/overview.md)), e rispondere delle sue conseguenze ([AI responsabile](../AIResponsabile/overview.md)). Non sono appendici morali messe in fondo per buona educazione: sono la parte del lavoro che decide se quello che hai costruito serve a qualcuno o fa danni.

Argomenti diversissimi, e in mezzo settant'anni di storia: dal percettrone di
Rosenblatt del 1957 ai modelli di oggi. Eppure, sotto, sempre le stesse tre
idee.

## Tre fili, un solo tessuto

```{figure} ../figures/fili-conduttori.svg
:name: fig-fili-conduttori
:alt: "Un asse orizzontale mostra l'arco del libro con cinque tappe (Matematica, Machine learning, Reti neurali, Deep learning, Generativi e RL); sotto, tre linee parallele colorate rappresentano i fili conduttori: Dati, Rappresentazioni, Ottimizzazione."
:width: 90%

L'arco dei modelli cambia da sinistra a destra (via via più espressivi) ma
sotto scorrono sempre gli stessi tre fili. I capitoli sul mestiere
(produzione, interpretabilità, responsabilità) non stanno su questo asse:
stanno attorno a tutto.
```

Se dovessimo comprimere l'intero libro in tre parole, sarebbero **dati**, **rappresentazioni** e **ottimizzazione** ({numref}`fig-fili-conduttori`). Sono i fili che attraversano ogni capitolo, dal più elementare al più avanzato.

I **dati** sono il carburante: nessun modello sa più di ciò che ha visto. Le **rappresentazioni apprese** sono il cuore della rivoluzione degli ultimi quindici anni.

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
Una rete profonda è una funzione composta $f_\theta = f^{(L)} \circ \dots \circ f^{(1)}$ che trasforma l'input grezzo $X$ in una sequenza di rappresentazioni intermedie sempre più astratte. Gli strati nascosti non sono altro che *feature apprese*: coordinate in uno spazio latente dove esempi semanticamente simili finiscono vicini. È il principio degli *embedding*, e la ragione per cui una sola rete pre-addestrata si riadatta a molti compiti. Il feature engineering manuale del machine learning classico non è sparito: è stato assorbito dentro $\theta$ e delegato all'ottimizzazione.
`````

E l'**ottimizzazione** è il motore che rende tutto questo possibile: apprendere significa cercare i parametri che minimizzano un errore.

`````{tab} Elementare
Immagina di regolare le manopole di un vecchio mixer per far suonare bene una canzone. Giri un po' una manopola, ascolti se è migliorato, correggi. Addestrare un modello è la stessa cosa, con milioni di manopole: a ogni passo il modello guarda quanto ha sbagliato e sposta ciascuna manopola nella direzione che riduce l'errore, un pochino. Ripetuto abbastanza volte, funziona.
`````

`````{tab} Superiore
Quasi ogni modello del libro si riduce a un problema di ottimizzazione:

$$
\theta^\star = \arg\min_{\theta}\ \mathcal{L}(\theta)
= \arg\min_{\theta}\ \frac{1}{m}\sum_{i=1}^{m} \ell\big(f_\theta(X^{(i)}),\, y^{(i)}\big),
$$

dove $\theta$ sono i parametri, $\mathcal{L}$ la funzione di costo, $\ell$ la
perdita sul singolo esempio, $f_\theta$ il modello, e $X^{(i)}$, $y^{(i)}$
l'input e il target dell'$i$-esimo degli $m$ esempi di addestramento. Cambia
$f_\theta$ (regressione lineare, rete convoluzionale, Transformer) ma la
macchina che risolve resta la discesa del gradiente stocastica. Tre idee,
infinite architetture.
`````

## Dove sta andando

Questo capitolo, nella sua prima stesura, indicava come "direzioni future" i
foundation model, la multimodalità e gli agenti. Nel frattempo sono diventati
tre capitoli di questo libro. È il modo più onesto di dire quanto corre il
campo, e la ragione per cui qui non troverai profezie, ma i fronti su cui oggi
si lavora davvero.

`````{tab} Elementare
La novità che ha cambiato tutto resta questa: non addestriamo più un modello nuovo per ogni problema. Ne addestriamo *uno solo*, enorme, su una montagna di testo o immagini, e poi lo adattiamo a mille compiti diversi con poco sforzo. È il modello "di fondazione": un'unica base su cui si costruisce tutto, un po' come una persona con una solida cultura generale che, con una breve formazione, impara mestieri molto diversi.

Le domande aperte, oggi, sono più concrete di quelle di ieri. **Quanto costa**: leggere un testo lunghissimo, per un Transformer, costa quanto il quadrato della sua lunghezza, e c'è una gara in corso per pagare meno. **Se capisce o indovina**: un modello che risponde bene non è per forza un modello che ha capito, e per saperlo bisogna aprirlo. **Se ci si può fidare**: un agente che agisce da solo per venti passi sbaglia in modi che un chatbot non poteva permettersi.
`````

`````{tab} Superiore
I **foundation model** {cite}`bommasani2021opportunities` restano il
paradigma: pre-addestramento auto-supervisionato su corpora enormi, poi
adattamento via fine-tuning o prompting. Le *scaling laws*
({cite}`kaplan2020scaling`; {cite}`hoffmann2022training`) hanno mostrato che
le prestazioni migliorano in modo prevedibile con dati e calcolo, ma non
promettono che *scalare* basti a risolvere tutto, e i quattro fronti aperti
sono, non per caso, quelli in cui scalare non basta.

**L'efficienza dell'attenzione.** Il costo quadratico in lunghezza è il
vincolo economico dell'intero settore: da qui l'attenzione lineare, i modelli
a spazio di stati e le architetture ibride, che riscrivono lo stesso calcolo
come una ricorrenza a memoria costante. **La comprensione del modello**:
l'interpretabilità meccanicistica prova a leggere i circuiti dentro i pesi, e
resta l'unico strumento che distingue una risposta corretta da una risposta
corretta *per il motivo giusto*. **L'affidabilità degli agenti**: comporre più
passi moltiplica le probabilità di errore, e la ricerca sulla valutazione
degli agenti è ancora giovane quanto gli agenti stessi. **Il conto fisico**:
energia, acqua, silicio e la concentrazione di tutto questo in pochi attori
(un problema di politica industriale travestito da problema tecnico).

E una direzione che è più una scommessa che una tendenza: i **modelli del mondo**, cioè imparare la dinamica dell'ambiente invece delle sole correlazioni nei dati. Se funzionasse su larga scala, cambierebbe l'ordine dei capitoli di un libro come questo.
`````

## Una nota onesta

Sarebbe disonesto chiudere con il solo entusiasmo. Questi sistemi hanno limiti strutturali, non incidenti temporanei.

`````{tab} Elementare
Un modello linguistico non "sa" le cose: prevede la parola più probabile dopo
le precedenti. Per questo a volte inventa con perfetta sicurezza fatti falsi:
le chiamiamo *allucinazioni*. E siccome impara da testi scritti da noi,
assorbe anche i nostri pregiudizi: se i dati storici riflettono
discriminazioni, il modello le ripete, a volte amplificandole. Uno strumento
potente non è uno strumento neutrale.
`````

`````{tab} Superiore
Un modello generativo massimizza la verosimiglianza dei dati, non la verità: la fluidità del testo è scorrelata dalla sua correttezza, da cui i problemi di calibrazione e le allucinazioni. I *bias* non sono un bug ma una proprietà attesa dell'apprendimento statistico da dati non rappresentativi {cite}`bender2021dangers`. A valle restano questioni aperte: impatto ambientale dell'addestramento, concentrazione di potere in pochi attori, effetti sul lavoro e sull'informazione. L'AI Act europeo (2024) è un primo tentativo di regolazione. Il fact-check umano, per noi, non è opzionale.
`````

## Come continuare a imparare

Questo libro è una mappa, non il territorio. Per proseguire: leggi i paper
originali (arXiv è gratuito e sorprendentemente accessibile, una volta preso
il ritmo), ma soprattutto *riproduci il codice* (un modello lo capisci quando
lo fai girare e lo rompi). Il metodo per farlo sta nel capitolo su PyTorch,
nella sezione su [come si replica un paper](../PyTorch/replicare-un-paper.md):
quattro mosse e tre verifiche che si fanno senza nemmeno addestrare.

Tieni i classici a portata: Géron {cite}`geron2019hands` per la pratica, Chollet {cite}`chollet2017deep` per l'intuizione, Goodfellow, Bengio e Courville {cite}`goodfellow2016deep` per la teoria, la documentazione di scikit-learn e PyTorch come compagne quotidiane. Partecipa a una competizione Kaggle, contribuisci a un progetto open source, tieni un quaderno degli esperimenti falliti: insegnano più dei successi.

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
quell'onestà intellettuale che fa dire "non lo so, verifichiamo" invece di
"l'ha detto il modello". L'AI resterà, come voleva Weizenbaum, resistente a
una definizione precisa. Ma non è più una scatola nera: è fatta di dati,
rappresentazioni e ottimizzazione (cose che ora sai guardare da vicino). Il
resto è pratica.

Buon lavoro. E, come si dice qui, *in bocca al lupo*.

```{admonition} Da ricordare
:class: important
- Tutto il libro poggia su tre idee ricorrenti: **dati**, **rappresentazioni apprese** e **ottimizzazione**.
- L'apprendimento è, quasi sempre, la minimizzazione di una loss: $\theta^\star = \arg\min_\theta \mathcal{L}(\theta)$.
- Quelle che in un libro come questo si chiamano "frontiere" invecchiano in fretta: foundation model, multimodalità e agenti erano previsioni, oggi sono capitoli. Restano varianti delle stesse tre idee, non magia.
- I fronti davvero aperti sono quelli in cui **scalare non basta**: il costo dell'attenzione, l'interpretabilità, l'affidabilità degli agenti, il conto energetico.
- Potenza e responsabilità crescono insieme: allucinazioni e bias sono limiti strutturali, e il fact-check umano non è opzionale.
- Gli ultimi capitoli non parlano di architetture ma di **mestiere**,
  produzione, interpretabilità, responsabilità: è la parte che decide se
  quello che hai costruito serve o fa danni.
```
