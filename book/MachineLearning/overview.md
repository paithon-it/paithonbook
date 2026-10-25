---
titolo_ricerca: "Machine learning: come una macchina impara dai dati"
descrizione: "Che cos'è il machine learning: la definizione di Mitchell, i tre
  modi in cui una macchina impara e le tappe di un progetto, dai dati al
  modello."
---
# Machine Learning: imparare dai dati

```{image} ../figures/aperture/machine-learning.png
:class: pt-apertura only-light
:width: 100%
:alt: Due giocatori identici, uno di fronte all'altro, giocano a dama: la stessa figura che gioca contro sé stessa. Sopra la scacchiera un arco di punti accenna alle mosse guardate in anticipo.
```

```{image} ../figures/aperture/machine-learning-scura.png
:class: pt-apertura only-dark
:width: 100%
:alt: Due giocatori identici, uno di fronte all'altro, giocano a dama: la stessa figura che gioca contro sé stessa. Sopra la scacchiera un arco di punti accenna alle mosse guardate in anticipo.
```

Nel 1959 Arthur Samuel, un ingegnere dell'IBM, pubblicò un articolo dal titolo
modesto, *Some Studies in Machine Learning Using the Game of Checkers*
{cite}`samuel1959some`, che oggi suona profetico. Samuel aveva scritto un
programma per giocare a dama che, giocando contro sé stesso, arrivò a giocare
meglio del suo autore. Nessuno gli aveva insegnato le mosse giuste una per
una: il programma le ricavava dall'esperienza.

Giocando contro sé stesso, il programma non aveva un maestro che gli indicasse
la mossa buona. Per questo motivo assegnava un voto alla posizione che aveva
davanti, guardava qualche mossa più avanti e correggeva il voto attuale,
avvicinandolo a quello della posizione futura. In altri termini, il maestro
era la sua stessa valutazione, calcolata un passo più avanti, dove la partita
si vede meglio. Un'informazione esterna, tuttavia, entrava comunque: Samuel
fissava a mano una delle voci del voto (i pezzi in più o in meno rispetto
all'avversario), che il programma non modificava, e guardando avanti il
programma vedeva le catture. Ripetuta per tutta la partita e per tutte le
partite, questa correzione trasforma il voto in un giudizio affidabile sulle
posizioni favorevoli. È l'idea da cui nascerà l’**apprendimento per differenze
temporali**, trattato per intero nella sezione sul {doc}`Q-learning
</ReinforcementLearning/q-learning>`.

In quell'articolo del 1959 compare, per una delle prime volte, l'espressione
*machine learning*, cioè la capacità di un calcolatore di migliorare in un
compito senza essere riprogrammato a mano. In questo capitolo vediamo che cosa
significa, in termini precisi, che un programma «impara». Partiamo dal
confronto con la programmazione tradizionale, diamo la definizione di
Mitchell, distinguiamo i tre modi principali di apprendere e descriviamo il
ciclo di lavoro di un progetto; chiudiamo con i dati in tabella, dove i metodi
classici restano competitivi con le reti neurali. Il {doc}`capitolo di
matematica </Matematica/overview>` si era chiuso su un punto d'arrivo, la
matematica di un modello linguistico; qui torniamo all'origine dell'idea.

## Scrivere le regole, o farle emergere

Il modo più semplice per capire la novità del machine learning è confrontarlo
con la programmazione tradizionale.

`````{tab} Elementare

Nella programmazione tradizionale le regole le scrivi tu. Per un filtro
antispam, ad esempio, potresti stabilire che «se l'email contiene la parola
*vincita*, va segnalata come spam» e che «se il mittente è sconosciuto, è
sospetta». Ogni regola va pensata, scritta e corretta a mano. Il sistema
funziona finché gli spammer non cambiano trucco; a quel punto, però, bisogna
ricominciare da capo.

Il machine learning procede in senso inverso. Raccogli migliaia di email già
**etichettate** come «spam» o «non spam» e le fornisci al programma, che cerca
nei dati le regolarità: quali parole, quali mittenti, quali combinazioni
ricorrono nello spam. In altre parole, le regole *emergono* dagli esempi,
senza che nessuno le scriva.

Si noti che le regole così ottenute devono valere anche fuori dalla raccolta.
Le email che hai dato al programma sono già state smistate a mano, quindi
smistarle di nuovo non serve a nessuno; [ciò che serve è che il programma
classifichi bene la prossima email, quella che nella raccolta non
c'era]{.pt-matita}.

`````

`````{tab} Superiore

Nella programmazione tradizionale il programmatore conosce la funzione $f$ che
trasforma un input in un output, e la codifica esplicitamente:
$\text{output} = f(\text{input})$. Le regole, in questo caso, sono note *a
priori*.

Nel machine learning, invece, $f$ è ignota. Disponiamo di una collezione di
coppie input-output osservate, e cerchiamo una funzione $f_\theta$, presa da
una famiglia parametrizzata da $\theta$, che le riproduca bene e che,
soprattutto, **generalizzi** a input mai visti. Il compito, pertanto, non è
più *scrivere* $f$, ma *stimare* i parametri $\theta$ a partire dai dati: il
[codice resta fisso, e con l'esperienza cambiano i numeri contenuti in
$\theta$]{.pt-matita}.

`````

Il risultato dell'addestramento è il **modello**, cioè la funzione $f_\theta$:
la regola insieme ai suoi **parametri** $\theta$ (i numeri regolabili), fissati
dai dati. Nel filtro antispam, ad esempio, $\theta$ contiene i numeri che
stabiliscono quanto pesano la parola «vincita» o un mittente sconosciuto,
mentre il codice che li usa resta lo stesso. Le tecniche per trovare i valori
di $\theta$, e le caratteristiche dell'input da esaminare, variano da un
metodo all'altro; in nessuno, tuttavia, la regola viene scritta a mano.

## Che cosa vuol dire "imparare": la definizione di Mitchell

La definizione operativa più citata è quella di Tom Mitchell, nel manuale
*Machine Learning* del 1997 {cite}`mitchell1997machine`. Il suo pregio è di
essere verificabile: permette di stabilire quando un programma impara e
quando no.

`````{tab} Elementare

Un programma impara se, facendo pratica, diventa più bravo in un compito, e se
questo «più bravo» si può misurare. Servono quindi tre ingredienti:

- il **compito**, cioè che cosa deve fare (giocare a dama);
- l’**esperienza**, cioè su che cosa fa pratica (le partite giocate);
- la **misura**, cioè come si contano i progressi (la percentuale di partite
  vinte).

Il programma di Samuel, ad esempio, diventava più bravo (vinceva di più) man
mano che accumulava partite. Se dopo mille partite ne avesse vinte la stessa
quota di prima, non avrebbe imparato niente, per quante ne avesse giocate:
[conta che la misura migliori, non il tempo passato al tavolo]{.pt-matita}.

Si noti che i tre ingredienti non parlano di dama. Al posto del compito metti
il riconoscimento di una firma falsa, al posto dell'esperienza le firme già
controllate una per una, al posto della misura gli errori su cento firme. La
domanda resta la stessa: gli errori diminuiscono man mano che aumentano le
firme controllate? Il criterio, pertanto, si applica a un programma
qualunque, senza bisogno di sapere come è fatto dentro.

`````

`````{tab} Superiore

Mitchell la formula così: un programma apprende da un'esperienza $E$ rispetto
a una classe di compiti $T$ e a una misura di prestazione $P$ se la sua
prestazione sui compiti in $T$, misurata da $P$, migliora con l'esperienza
$E$.

- $T$ (*task*): il problema, ad esempio classificare email.
- $E$ (*experience*): i dati da cui il programma apprende, ad esempio $m$
  email etichettate.
- $P$ (*performance*): una metrica scalare, ad esempio l'accuratezza sul test.

La formulazione è volutamente astratta: non nomina reti neurali né alberi di
decisione, e vale quindi per qualunque algoritmo di apprendimento. Se
all'aumentare di $E$ la $P$ non migliora, non c'è apprendimento. Affinché la
condizione sia vincolante, tuttavia, [$P$ va misurata su esempi che non fanno
parte di $E$]{.pt-matita}. Una tabella che memorizza le coppie viste, infatti,
migliora con l'esperienza, se la si valuta su quelle stesse coppie, senza aver
imparato nulla.

`````

## Tre modi di imparare

A seconda del tipo di esperienza disponibile, il machine learning si divide in
tre grandi famiglie, che nella letteratura di ricerca prendono il nome di
*paradigmi*.

**Apprendimento supervisionato.** Ogni esempio arriva con la sua risposta
giusta, detta **etichetta**, come nel filtro antispam, e il modello impara a
prevedere l’*output* a partire dall’*input*. Se l'output è una categoria (spam
o non spam, gatto o cane) si parla di **classificazione**; se invece è un
numero che può assumere qualunque valore intermedio, come il prezzo di una
casa in funzione dei metri quadri, si parla di **regressione**. Sono i valori
intermedi a fare la differenza: 2,5 metri quadri esistono, 2,5 stanze no.

`````{tab} Elementare

A fianco di ogni esercizio c'è la soluzione. Chi studia così prova a
rispondere da solo, poi scopre il riquadro accanto e vede di quanto ha
sbagliato. Per ogni esercizio, quindi, gli resta un numero: lo **scarto** fra
la sua risposta e quella giusta.

Un singolo scarto, però, dice poco; conta la media su tutta la raccolta. Ad
esempio, cinque esercizi sbagliati di 2, 0, 1, 0 e 2 punti fanno 5 punti in
tutto, cioè 1 punto a esercizio, e chi studia cerca di far scendere questa
media. Un modo di rispondere che azzecca un solo esercizio e sbaglia di molto
gli altri quattro alza la media, e vale quindi di meno, per quanto sia
perfetto in quell'esercizio. In definitiva, fra due modi di rispondere si
sceglie quello con la media più bassa.

Le «soluzioni a fianco» sono le etichette. Senza di esse non c'è niente con
cui misurare lo scarto, e questo tipo di apprendimento non funziona. Si noti,
infine, che la raccolta si studia in vista degli esercizi che ancora non ci
sono, quelli senza soluzione a fianco: è su quelli che si vede se qualcuno ha
imparato.

`````

`````{tab} Superiore

Dato un insieme di addestramento
$\{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^{m}$, dove $\mathbf{x}^{(i)}$ è il
vettore delle feature dell'esempio $i$-esimo e $y^{(i)}$ la sua etichetta, si
cerca la funzione $f_\theta$ che minimizza una **funzione di costo** (o
*loss*), cioè una funzione che penalizza le previsioni sbagliate:

$$
\theta^\star = \arg\min_{\theta}\ \frac{1}{m}\sum_{i=1}^{m}
\ell\!\left(f_\theta(\mathbf{x}^{(i)}),\, y^{(i)}\right).
$$

Qui $f_\theta(\mathbf{x}^{(i)}) = \hat{y}^{(i)}$ è la predizione del modello,
e $\ell$ ne misura la distanza dal valore vero $y^{(i)}$; la media di tutti
gli $\ell$ è la loss sull'intero insieme, $\mathcal{L}$. L'addestramento
supervisionato consiste, in definitiva, in questo problema di minimizzazione.
La regolarizzazione vi aggiunge una penalità, mentre la validazione stima
l'errore su dati nuovi, che è la grandezza di interesse.

`````

**Apprendimento non supervisionato.** In questo caso le etichette non ci sono:
il modello riceve soltanto gli input e deve scoprire da sé una struttura
nascosta. L'esempio classico è il **clustering**, che consiste nel raggruppare
i clienti di un negozio in segmenti simili senza sapere in anticipo quali
segmenti esistano. Rientrano in questa famiglia anche la **riduzione della
dimensionalità**, ovvero la descrizione di ogni esempio con meno numeri,
conservandone il più possibile l'informazione, e il **rilevamento di
anomalie** senza esempi etichettati, ad esempio una transazione che non
somiglia alle altre. (Le «dimensioni» sono le colonne della tabella, una per
caratteristica; perché si chiamino così lo spiega la {doc}`sezione
sull'apprendimento supervisionato
</MachineLearning/apprendimento-supervisionato>`.)

**Apprendimento per rinforzo.** Qui non ci sono etichette, e non c'è nemmeno
una raccolta di esempi fissata in partenza. C'è un **agente**, cioè un
programma che agisce (non una persona), che compie azioni in un **ambiente** e
riceve, di tanto in tanto, una **ricompensa**. L'agente impara per tentativi
la strategia che massimizza la ricompensa nel tempo.

È in questo modo che AlphaGo, il programma del laboratorio DeepMind, imparò
nel 2016 a battere i campioni del Go, un antico gioco da tavolo orientale. Per
la precisione, quella prima versione studiò anche partite umane etichettate,
mentre la versione dell'anno successivo imparò soltanto giocando contro sé
stessa. Il programma di dama di Samuel ne è un antenato: giocava contro una
copia di sé stesso e, dopo ogni mossa, spostava la valutazione della posizione
verso quella che otteneva guardando qualche mossa più avanti. Non c'erano,
però, né una ricompensa esplicita né un trattamento speciale della vittoria.

Si noti che i confini fra le tre famiglie non sono netti.
Nell'apprendimento auto-supervisionato, con cui si addestrano i modelli
linguistici, le etichette si ricavano dagli stessi dati (ad esempio la parola
successiva di un testo), e la loss è quella di un problema supervisionato. Lo
tratta per esteso il {doc}`capitolo sull'auto-supervisione
</AutoSupervisione/overview>`.

## Dall'idea al modello: il flusso di un progetto

Addestrare un modello è soltanto uno dei passaggi di un progetto di machine
learning. I passaggi formano un ciclo, perché i risultati della valutazione
indicano dove tornare per migliorare ({numref}`fig-workflow-ml`).

```{figure} ../figures/workflow-ml.svg
:name: fig-workflow-ml
:alt: Cinque blocchi in fila (Dati, Feature, Modello, Valutazione, Deploy) collegati da frecce; una freccia di feedback torna dalla Valutazione alle Feature.
:width: 95%

Il flusso tipico di un progetto ML. Dopo la valutazione, fatta su dati tenuti da
parte, si torna quasi sempre indietro a rivedere le feature, e si ricomincia con
un'altra iterazione.
```

I passaggi, nell'ordine, sono i seguenti.

1. **Dati**: si raccolgono gli esempi e li si ripulisce da valori mancanti,
   duplicati ed errori. È spesso la fase più lunga e ingrata dell'intero
   progetto. Di solito i dati si organizzano in una tabella, con una riga per
   ogni esempio e una colonna per ogni grandezza misurata.
2. **Feature**: un modello non legge un'email, calcola su numeri. Le
   **feature** (in italiano le *caratteristiche*, cioè le colonne della
   tabella) sono i numeri con cui si descrive ogni esempio. Di un'email, ad
   esempio, si possono prendere il numero di parole, quello dei punti
   esclamativi, quante volte compare «vincita» e se il mittente è in rubrica:
   quattro numeri, che per il modello diventano l'email stessa. Nel machine
   learning classico le feature le sceglie chi progetta il sistema, e
   cambiandole cambia la risposta; le reti profonde, tema del {doc}`capitolo
   sulle reti neurali </RetiNeurali/overview>`, le ricavano invece in parte
   dai dati grezzi.
3. **Modello**: si decide innanzitutto quale *forma* dare al modello (una
   retta? un albero di domande? una rete?), e poi lo si addestra sui dati.
   Dentro un modello ci sono dei numeri regolabili, i **parametri**, indicati
   tutti insieme con $\theta$, e **addestrare** significa modificarli finché
   il modello sbaglia il meno possibile sui dati. Anche l'errore si misura con
   un numero, la **loss** (in italiano *perdita*), definita come la media, su
   tutti gli esempi, della penalità assegnata a ciascuna risposta sbagliata.
   Poiché la loss dipende dai parametri, si scrive $\mathcal{L}(\theta)$. In
   sintesi, la forma si sceglie prima, i numeri li trova l'addestramento, e il
   «modello» in senso stretto è il risultato dei due insieme.
4. **Valutazione**: si misurano le prestazioni su dati mai visti durante
   l'addestramento. Gli esempi già usati, infatti, non servono allo scopo: su
   di essi un modello può ottenere un errore bassissimo limitandosi a
   riprodurli, e quell'errore non dice nulla di come si comporterà su
   un'email nuova. Interessa l'errore su dati mai visti, il cosiddetto
   **errore di generalizzazione**.
5. **Deploy**: se i risultati sono convincenti, si mette il modello in
   produzione, cioè lo si lascia lavorare con utenti veri e con dati che
   arrivano ogni giorno. Il modello va comunque sorvegliato, perché i dati del
   mondo cambiano nel tempo.

Quasi mai il primo tentativo è quello buono, e per questo motivo [la freccia di
ritorno conta più delle altre]{.pt-matita}. Si osserva dove il modello sbaglia
sui dati tenuti da parte per il giudizio, si ritoccano le feature o il modello
e si ricomincia. Su quali dati si debba giudicare lo stabilisce la
{doc}`sezione su overfitting e
validazione </MachineLearning/overfitting-validazione>`.

`````{tab} Elementare

In pratica l’**addestramento** (in inglese *training*, la parola che si sente
più spesso) richiede pochissime righe di codice. Con una **libreria**, cioè
una raccolta di strumenti già pronti scritti da altri, addestrare un modello e
usarlo sono due sole istruzioni: `.fit()` per imparare dai dati, `.predict()`
per prevedere su casi nuovi. La libreria dell'esempio si chiama scikit-learn.

`````

`````{tab} Superiore

Il metodo `fit` cerca i parametri che rendono piccola la loss sugli esempi di
addestramento, in modo esatto o approssimato a seconda del modello. L'albero di
decisione dell'esempio procede in modo approssimato: costruisce le domande una
alla volta, senza tornare indietro, e a ogni passo sceglie quella che riduce di
più una misura di impurità (per default l'indice di Gini), che svolge il ruolo
della loss. Il metodo `predict` applica poi la $f_{\theta^\star}$ appresa. Si
noti che la separazione tra dati di addestramento e dati di test serve a
stimare l'errore di generalizzazione, non quanto bene il modello ricordi gli
esempi già visti.

`````

```python
from sklearn.tree import DecisionTreeClassifier   # un albero di decisione,
                                                  # cioè una catena di domande
                                                  # sì/no: lo vediamo fra poco

# X_train: le feature di ogni esempio, y_train: l'etichetta da prevedere.
# Per convenzione le X sono maiuscole (una tabella) e le y minuscole (una
# sola colonna di risposte); X_test sono gli esempi tenuti da parte.
modello = DecisionTreeClassifier()
modello.fit(X_train, y_train)       # training: il modello impara dai dati
y_pred = modello.predict(X_test)    # previsione su dati mai visti in training
```

## Dati tabellari: dove il machine learning classico resta competitivo

Nei gruppi di lavoro alle prime armi capita spesso questa scena. Arriva un
problema, ad esempio prevedere quali clienti abbandoneranno il servizio a
partire da una tabella con età, contratti, consumi e reclami, e qualcuno
propone subito una rete neurale profonda, perché è quella di cui parlano
tutti. Dopo due settimane di messa a punto la rete arriva a stento a pareggiare
un **gradient boosting**, cioè una serie di piccoli modelli semplici messi in
fila, ciascuno dei quali corregge gli errori del precedente. Un collega
scettico aveva addestrato quel gradient boosting in dieci minuti, senza
modificare nemmeno un'impostazione.

```{figure} ../figures/ml-classico-batte-deep-learning.svg
:name: fig-tabellari-vs-non-strutturati
:alt: "Due domini affiancati. A sinistra i dati tabellari, righe e colonne con significati eterogenei, dove i metodi classici basati su alberi restano competitivi. A destra i dati non strutturati, immagini, audio e testo, dove il deep learning domina perché le caratteristiche utili vanno costruite e non sono già nelle colonne."
:width: 100%

I metodi migliori cambiano da un dominio all'altro. A sinistra i dati già in
tabella, dove le colonne *sono già* le caratteristiche buone; a destra le foto,
il suono e il testo, dove le caratteristiche vanno ricavate dai puntini di
un'immagine o dall'onda di un suono, e le reti profonde sono oggi il metodo di
riferimento.
```

Il confine che quella squadra non ha rispettato è disegnato nella
{numref}`fig-tabellari-vs-non-strutturati`.

`````{tab} Elementare

Sullo schermo c'è la tabella dei clienti: una riga per persona e, in colonna,
l'età, il codice postale, il reddito annuo, i giga consumati il mese scorso, i
reclami aperti. Fra la colonna del codice postale e quella del reddito non c'è
alcun rapporto, perché hanno unità, scale e significati diversi. Le colonne,
infatti, si possono scambiare di posto senza che la tabella dica qualcosa di
diverso. In altri termini, la loro disposizione non conta: sono colonne
«senza geografia».

In una fotografia, al contrario, la geografia c'è: due puntini vicini
appartengono allo stesso occhio, e scambiarli sfigura il volto. Allo stesso
modo, in una frase c'è un ordine, e spostare una parola cambia chi fa che
cosa. Le reti profonde hanno ridefinito ciò che si può fare con immagini,
suoni e linguaggio proprio perché sono costruite su questa geografia, con
componenti fatti apposta per i puntini vicini e altri per capire quali parole
si riferiscono a quali. Davanti alla tabella dei clienti, però, quei
componenti non hanno niente da guardare, e il vantaggio svanisce.

I modelli semplici che il collega scettico ha messo in fila lavorano
una domanda alla volta: «i giga consumati sono più di ottanta?», poi «il
contratto è ancora attivo?», poi «il quartiere è Milano?», e così via fino a
una risposta. Una catena di domande con risposta sì o no, ciascuna su una sola
colonna, si chiama **albero di decisione**. Che le colonne abbiano unità e
scale diverse non gli crea alcun problema, poiché non le mescola mai fra loro.

Inoltre, dentro quella tabella le cose cambiano di colpo. Supponiamo che la
promozione parta da cinquanta euro di ricarica: chi si ferma a quarantanove e
novanta non riceve i giga in regalo, chi arriva a cinquanta e un centesimo sì,
e di due clienti quasi identici uno rinnova e l'altro se ne va. Una rete
neurale tende a tracciare curve morbide, e attraversa un salto del genere
arrotondandolo; di conseguenza sbaglia soprattutto sui clienti a ridosso della
soglia. L'albero, al contrario, fa quel salto in modo netto, perché la sua
domanda è già «la ricarica supera i cinquanta euro?», e di soglie così ne mette
una dietro l'altra.

Nella stessa tabella, poi, parecchie colonne non dicono niente sul problema:
il codice interno del cliente, la data in cui la riga è stata inserita, un
campo che qualcuno ha smesso di compilare due anni fa. La rete le usa tutte, e
da quel rumore ricava qualcosa che scambia per un segnale. L'albero, a
ogni passo sceglie una sola colonna, quella che separa meglio chi resta da chi
se ne va; una colonna muta, che non separa bene niente, di solito perde contro
le colonne che contano. Con poche righe e moltissime colonne mute può capitare
che qualcuna vinca per caso, ma l'albero ne risente comunque meno di una rete,
che le usa tutte.

Questo confronto è stato ripetuto nel 2022 su decine di tabelle reali,
concedendo ai due contendenti lo stesso tempo di messa a punto, e gli alberi
hanno vinto su tabelle come quella dell'esempio, intorno alle diecimila righe.
Su tabelle più grandi il divario si riduceva, mentre su quelle più piccole il
confronto non era stato misurato; da allora, inoltre, sono arrivati modelli
pre-addestrati che dichiarano di vincere sulle tabelle piccole. Una rete
profonda, d'altra parte, sa costruirsi da sola le caratteristiche utili a
partire dai dati grezzi, cosa che un albero non sa fare; per riuscirci, però,
ha bisogno di moltissimi esempi.

C'è infine il costo di esercizio. Il modello del collega si è addestrato sul
suo portatile e va in servizio senza macchine speciali. Per questo motivo,
prima di adottare una rete profonda [conviene mettere in piedi il modello
semplice e regolarlo bene]{.pt-matita}: molto spesso basta già quello.

`````

`````{tab} Superiore

L'osservazione è stata verificata su decine di dataset. Grinsztajn, Oyallon e
Varoquaux (NeurIPS 2022) {cite}`grinsztajn2022why` hanno confrontato modelli
ad albero e reti neurali su 45 dataset tabulari: sui dati di taglia media
(intorno ai diecimila esempi) i modelli ad albero restavano superiori anche a
parità di ricerca degli **iperparametri**, cioè delle scelte che si fissano a
mano prima dell'addestramento. Il confronto, tuttavia, risale a quell'anno: da
allora alcuni modelli pre-addestrati su dati sintetici, come TabPFN
{cite}`hollmann2025tabpfn`, dichiarano di superare sotto i diecimila esempi i
metodi precedenti, boosting compresi. Le ragioni che seguono spiegano il
risultato del 2022, e restano utili per capire dove una rete incontra
difficoltà su una tabella.

Le ragioni individuate sono di natura strutturale.

1. Le reti hanno un *bias induttivo* verso funzioni regolari, mentre i target
   tabulari sono spesso irregolari a tratti, cioè proprio il tipo di funzione
   che una serie di split assiali approssima bene.
2. Le reti sono più sensibili degli alberi alle feature non informative, di
   cui una tabella reale abbonda. Un albero, infatti, sceglie a ogni nodo la
   divisione che migliora di più un'impurità, e le colonne senza relazione con
   il bersaglio vincono di rado nei nodi alti, anche se con pochi esempi per
   nodo qualcuna vince per caso.
3. L'addestramento di un MLP con la discesa del gradiente, partendo da pesi a
   simmetria sferica (ad esempio gaussiani indipendenti), è invariante per
   rotazione. Se si ruotano le colonne con una matrice ortogonale, le
   previsioni ottenute restano invariate in distribuzione, e pertanto la
   procedura non può sfruttare la base in cui la tabella è scritta (con Adam,
   che riscala ogni coordinata separatamente, l'invarianza non è più esatta).
   Una tabella, al contrario, possiede una sua base naturale, composta da
   colonne con significati diversi, che una mescolanza cancellerebbe. Si
   osserva infatti che, dopo una rotazione casuale delle colonne, l'ordine fra
   alberi e reti si inverte, e fra le reti confrontate solo il ResNet (un MLP
   con connessioni residue) risulta insensibile alla rotazione
   {cite}`grinsztajn2022why`. Questo aspetto si lega al punto precedente: una
   procedura invariante per rotazione richiede, nel caso peggiore, un numero
   di esempi che cresce almeno linearmente con il numero di feature
   irrilevanti {cite}`ng2004feature`.

Ne segue un'indicazione pratica. [Un gradient boosting si addestra in minuti su
CPU e va in produzione senza GPU]{.pt-matita}; pertanto, prima di adottare il deep
learning, conviene disporre di una **linea di base** classica ben calibrata
(in inglese *baseline*), che spesso basta.

`````

Il machine learning classico viene trattato per primo perché i suoi quattro
concetti fondamentali (modello, feature, parametri, loss) e le sue due
operazioni (addestrare e valutare su dati mai visti) valgono anche per le
famiglie di modelli che seguono, nelle quali cambia, semmai, la grandezza
misurata dalla loss. Il primo passo è l’{doc}`apprendimento supervisionato
</MachineLearning/apprendimento-supervisionato>`.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Nel machine learning non si scrivono le regole: si danno migliaia di
  esempi già etichettati (le email marchiate «spam» e «non spam») e le regole
  emergono da sole dai dati.
- Nei confronti fatti fino al 2022 su tabelle intorno alle diecimila righe gli
  alberi battevano le reti profonde: fra le colonne di una tabella non c'è
  quella vicinanza che le reti sanno sfruttare fra i puntini di una foto o fra
  le parole di una frase. Modelli pre-addestrati più recenti dichiarano di aver
  ribaltato il risultato sulle tabelle piccole.
- Quattro concetti valgono per tutti i modelli (modello, feature, parametri,
  loss), e due operazioni: addestrare, e valutare su dati mai visti.
- Un programma impara (Mitchell) se, facendo pratica, diventa più bravo in
  un compito e questo «più bravo» si può misurare: servono il compito,
  l’esperienza e la misura.
- Tre modi di imparare: con le soluzioni a fianco (supervisionato), senza
  etichette, cercando una struttura nascosta (non supervisionato), per
  tentativi e ricompense (per rinforzo).
- Il flusso (dati, feature, modello, valutazione, deploy) è un ciclo: la
  valutazione rimanda indietro, e si ricomincia.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Nel machine learning non si scrivono le regole: si forniscono esempi e le
  regole emergono dai dati, cioè si stimano i parametri $\theta$ minimizzando
  una loss $\mathcal{L}$ sugli esempi osservati.
- Nel confronto di Grinsztajn et al. (2022), su dati tabulari di taglia media,
  dell'ordine dei diecimila esempi, i modelli ad albero restavano superiori alle
  reti anche a parità di ricerca degli iperparametri {cite}`grinsztajn2022why`.
  Modelli pre-addestrati più recenti dichiarano di aver ribaltato il risultato
  sotto i diecimila esempi.
- Le ragioni sono strutturali: il bias induttivo delle reti verso funzioni
  regolari, contro target irregolari a tratti; la loro maggiore sensibilità alle
  feature non informative; l'invarianza per rotazione del loro addestramento,
  dannosa dove mescolare linearmente le colonne cancella il significato di
  ciascuna.
- Un programma impara (Mitchell) se la sua performance $P$ su un compito $T$,
  misurata su esempi esterni a $E$, migliora con l'esperienza $E$.
- Tre paradigmi: supervisionato (dati etichettati), non supervisionato
  (struttura nascosta, senza etichette), per rinforzo (agente e ricompense).
- Il flusso (dati, feature, modello, valutazione, deploy) è un ciclo: la
  valutazione rimanda indietro, e si itera.
```

`````
