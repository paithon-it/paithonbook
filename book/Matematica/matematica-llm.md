# La matematica di un modello linguistico

Joseph Breeden costruisce modelli statistici per il rischio di credito, dopo
decenni passati fra sistemi non lineari, statistica e linguistica. Quando si
siede a leggere la letteratura sui modelli linguistici è ragionevolmente
convinto che gli basterà. Non gli basta, e non per la matematica: per i nomi.
«Un token non è una *query*. I linguisti non parlano di *chiavi*. Che cos'è
una *testa*?». Il documento che scrive per rimediare
{cite}`breeden2026simple` ricostruisce gli stessi modelli usando soltanto
parole di matematica e di statistica, e alla fine osserva che il muro non era
il contenuto: era il vocabolario preso in prestito da mestieri diversi (le
basi di dati, le scienze cognitive, l'elettrotecnica) e appiccicato sopra a
operazioni che avevano già un nome.

Dei quattro nomi di cui si lamenta, uno serve subito e va tolto di mezzo: un
**token** è il pezzetto di testo su cui il modello lavora, non proprio una
parola ma un frammento (una parola corta per intero, la radice di una lunga, un
segno di punteggiatura), e una frase, per il modello, è la fila dei suoi token.
Gli altri tre (*query*, *chiave*, *testa*) sono etichette appiccicate a tre
oggetti che qui nasceranno con un nome italiano; il nome inglese arriverà dopo,
in una tabella, quando ci sarà qualcosa da tradurre.

Questa sezione fa lo stesso percorso, ed è il punto in cui la cassetta degli
attrezzi si richiude. Non spiega l'architettura Transformer, che il capitolo
dedicato smonterà pezzo per pezzo con i suoi nomi standard; risponde a una
domanda più stretta e, per chi ha appena finito cinque sezioni di
matematica, più urgente: *di che cosa è fatto un modello linguistico, se lo si
guarda con gli strumenti visti fin qui?*

La risposta breve è che non serve nient'altro. Prodotti scalari e prodotti fra
matrici dall'algebra lineare, la regola della catena e il valore atteso dalla
probabilità, la cross-entropia dalla teoria dell'informazione, la derivata
delle funzioni composte e la discesa del gradiente dall'analisi, il
*log-sum-exp* dall'analisi numerica. Sono tutti qui dentro. La
sofisticazione non sta nei pezzi, sta in come vengono composti e in quante
volte vengono ripetuti.

## Il compito, scritto in una riga di probabilità

Prima di ogni architettura c'è un obiettivo statistico, ed è
sorprendentemente modesto: assegnare probabilità alla parola successiva.

`````{tab} Elementare

Un modello linguistico è una macchina che, davanti a un pezzo di testo, dice
quanto è probabile ogni possibile continuazione. Davanti a «Il gatto nero
salta sul» darà molta probabilità a «muro», «tetto», «divano», pochissima a
«pigiama» e praticamente zero a «democrazia».

Sembra poco, e invece è tutto: se so scommettere bene sulla parola dopo, so
anche scrivere. Basta scegliere una parola fra quelle probabili, aggiungerla
al testo e rifare la domanda. La frase esce una parola alla volta, come una
lunga catena di scommesse in cui ogni puntata tiene conto di tutte quelle
già fatte.

E se so scommettere bene sulla parola dopo *in ogni situazione*, allora ho
imparato qualcosa di più di una lista di frasi fatte, perché per indovinare
la parola che segue «Il paziente è stato dimesso con una diagnosi di» bisogna
sapere che lì ci va il nome di una malattia.

`````

`````{tab} Superiore

Il testo è una sequenza di simboli discreti $(w_1, w_2, \dots, w_n)$, con
ciascun $w_i$ preso da un vocabolario finito $\mathcal{V}$. Modellare la
probabilità dell'intera sequenza sembra un problema in molte dimensioni, ma
la **regola della catena** della probabilità lo fattorizza in modo esatto:

$$
P(w_1, w_2, \dots, w_n) = P(w_1)\, P(w_2 \mid w_1)\, P(w_3 \mid w_1, w_2)
\cdots P(w_n \mid w_1, \dots, w_{n-1}) .
$$

Non è un'approssimazione né un'ipotesi di indipendenza: è un'identità, vera
per qualunque distribuzione congiunta. Ogni fattore è una distribuzione
condizionata sulla parola successiva dato tutto ciò che la precede, e un
modello linguistico è esattamente un sistema che calcola quei fattori.

Da qui discende anche il modo di generare testo: campionare $w_{n+1}$ da
$P(\cdot \mid w_1,\dots,w_n)$, accodarlo e ripetere, è campionare dalla
congiunta un fattore alla volta.

`````

## Perché una tabella non basta

La formulazione è innocua, la sua realizzazione no. Il problema è che il
numero di contesti possibili non cresce: esplode.

`````{tab} Elementare

L'idea più ovvia sarebbe un enorme quaderno di conteggi: per ogni possibile
inizio di frase, la lista di quali parole lo hanno seguito e quante volte.
Funziona per contesti cortissimi, e per un secolo è stato il modo di fare
statistica sul linguaggio.

Fa i conti con l'aritmetica, però. Con un vocabolario da 50 000 voci, i
contesti lunghi appena dieci parole sono $50\,000^{10}$, un numero con 47
cifre: circa il doppio delle molecole d'acqua contenute in tutti gli oceani
della Terra. Nessun quaderno, nessun disco, nessun datacenter.

E anche potendo scriverlo, quel quaderno sarebbe quasi tutto vuoto: la
stragrande maggioranza delle frasi sensate non è mai stata scritta da
nessuno, quindi avrebbe conteggio zero, e un conteggio zero non è una
previsione. Serve un oggetto diverso: non una tabella da consultare ma una
**funzione** da calcolare, che dato un contesto qualsiasi (anche mai visto)
produca una distribuzione sensata. Una funzione che abbia imparato da «Il
gatto nero salta sul muro» qualcosa di utile anche per «Il cane bianco corre
sul prato».

`````

`````{tab} Superiore

Il numero di contesti di lunghezza $m$ su un vocabolario di cardinalità
$|\mathcal{V}|$ è $|\mathcal{V}|^m$: con $|\mathcal{V}| = 5 \cdot 10^4$ e
$m = 10$ si ottiene $9{,}8 \cdot 10^{46}$. Una stima tabellare
(la massima verosimiglianza per conteggi, cioè i modelli $n$-gram del
capitolo sul NLP) è impraticabile ben prima: il numero di parametri cresce
esponenzialmente in $m$, e la matrice dei conteggi diventa quasi ovunque
nulla.

La via d'uscita è sostituire la tabella con una funzione parametrica
$f_\theta$ che mappa contesti in distribuzioni, con $|\theta|$ fissato e
indipendente dal numero di contesti. Il costo dell'approssimazione è pagato
in *bias*, il guadagno è la **generalizzazione**: contesti diversi ma vicini
nello spazio delle rappresentazioni ricevono distribuzioni vicine, e questo
è, in sostanza, uno smoothing implicito e appreso al posto di quelli
progettati a mano (Laplace, Good–Turing, Kneser–Ney).

Vale la pena notare che cosa *non* cambia. Un modello linguistico ha una
lunghezza massima di contesto fissata a progetto, e tutto ciò che precede
quella soglia è invisibile: formalmente resta quindi una catena di
**Markov di ordine finito**, per quanto altissimo
{cite}`shalizi2023attention`. Cambia il modo di stimare le probabilità di
transizione, non la classe del modello.

`````

## Da simboli a punti in uno spazio

Una funzione ha bisogno di numeri in ingresso, e i token sono simboli. La
mossa standard è già stata vista nella sezione di algebra lineare, in
un'altra veste: rappresentare ogni token con un vettore, cioè con una lista di
numeri.

`````{tab} Elementare

Il modo più ovvio è anche il peggiore: si numerano le voci del vocabolario da
$1$ a cinquantamila e si dà a ciascuna una lista lunga cinquantamila, tutta di
zeri tranne un $1$ nella sua casella. È la codifica che si chiama *one-hot*,
«uno acceso». Funziona, nel senso che ogni parola ha la sua lista e nessuna si
confonde con un'altra, ma butta via l'unica cosa che ci interessava. Prese due
liste qualsiasi, la loro differenza è sempre la stessa: due caselle diverse, e
basta. Il libro dichiara così che «gatto» e «cane» sono lontani fra loro
esattamente quanto «gatto» e «sebbene». Tutto quello che il modello impara sul
primo va reimparato da capo sul secondo.

L'alternativa è dare a ogni token una lista molto più corta (da qualche
centinaio a qualche migliaio di numeri) e, soprattutto, **non deciderla noi**:
quei numeri sono manopole come tutte le altre, e l'addestramento li regola
insieme al resto. È l'**embedding**, e il capitolo sul Natural Language
Processing lo tratta per esteso, insieme alla ragione per cui funziona: una
parola si conosce dalla compagnia che frequenta, quindi parole che compaiono
negli stessi contesti finiscono per prendere liste simili.

Il guadagno è tutto qui: adesso le parole sono punti, e sui punti si possono
fare le domande di poco fa, quelle dell'algebra lineare. Quanto sono lontani?
Puntano dalla stessa parte? Qual è il punto che sta in mezzo? Su una fila di
caselle spente e una accesa nessuna di queste domande aveva una risposta
interessante.

`````

`````{tab} Superiore

Il modo ingenuo è dare a ogni voce del vocabolario un indice e trasformarlo
in un vettore lungo $|\mathcal{V}|$ con un solo 1 (la codifica *one-hot*). Ha
un difetto fatale: rende ogni coppia di token equidistante ($\lVert
\mathbf{x}-\mathbf{y}\rVert_2 = \sqrt{2}$ per ogni coppia distinta), cioè
dichiara che «gatto» e «cane» sono diversi fra loro esattamente quanto
«gatto» e «sebbene». Ogni cosa imparata sul primo va reimparata da capo sul
secondo.

L'alternativa è una **matrice di embedding** $\mathbf{E} \in
\mathbb{R}^{|\mathcal{V}| \times d}$, di cui la riga $\mathbf{e}_w \in
\mathbb{R}^d$ è la rappresentazione del token $w$; la dimensione $d$ è moderata
(nei modelli attuali da qualche centinaio a qualche migliaio) e tutte le
coordinate sono parametri appresi. Formalmente è il prodotto
$\mathbf{E}^\top \mathbf{x}_w$ con $\mathbf{x}_w$ il vettore one-hot, cioè una
selezione di riga: la codifica ingenua non viene sostituita, viene composta con
una mappa lineare appresa. Il capitolo sul Natural Language Processing tratta
l'embedding per esteso, compresa la ragione linguistica che lo giustifica: il
principio di Firth, «una parola la conosci dalla compagnia che frequenta»
{cite}`firth1957synopsis`, e le regolarità aritmetiche che ne emergono
{cite}`mikolov2013distributed`. Qui basta la conseguenza matematica, che è
tutta nel prodotto scalare: una volta che i token sono punti, si possono
sommare, mediare, confrontare per angolo e per distanza. Diventano oggetti su
cui l'algebra lineare ha qualcosa da dire.

`````

## Il significato dipende dal contesto

C'è però un limite strutturale in un vettore per token, e si vede a occhio
nudo in italiano con la parola «piano»:

- «Suonava il **piano** a orecchio» (lo strumento);
- «Abita al terzo **piano**» (il livello di un edificio);
- «Cammina **piano**, che dorme» (l'avverbio);
- «Non avevano un **piano**» (il progetto).

Un solo punto nello spazio non può stare vicino a «pianoforte», a «scala», a
«lentamente» e a «strategia» contemporaneamente. La polisemia non è
un'eccezione da manuale, è la norma: pochi vocaboli frequenti hanno un solo
significato.

Da qui l'obiettivo di calcolo, che è la richiesta precisa da cui nasce tutto
il resto: entrano tanti vettori quante sono le parole, uno per parola presa da
sola, e ne devono uscire altrettanti, uno per posizione, ciascuno dei quali
tenga conto di tutta la frase in cui quella parola si trova. La stessa parola,
in due frasi diverse, deve uscire con due vettori diversi. In simboli: data la
sequenza $(\mathbf{e}_1, \dots, \mathbf{e}_n)$, vogliamo produrre una nuova
sequenza $(\mathbf{h}_1, \dots, \mathbf{h}_n)$, della stessa lunghezza e della
stessa dimensione $d$, in cui ogni $\mathbf{h}_i$ non dipenda solo dal token in
posizione $i$.

## Il motore: una media pesata con pesi appresi

Il meccanismo che risolve il problema è, come operazione, fra le più modeste
di questo libro: una media pesata. Tutto il resto è il modo in cui i pesi
vengono decisi.

`````{tab} Elementare

Per capire «salta» in «Il gatto nero salta sul muro» conviene mescolare al
suo vettore un po' di ciò che sanno le altre parole: parecchio di «gatto»
(è chi salta), un po' di «muro» (è dove), quasi niente di «il» e di «sul».
Mescolare in proporzioni diverse è fare una **media pesata**: si moltiplica
ogni vettore per un numero e si somma tutto (sono i due gesti della sezione di
algebra lineare, moltiplicare una lista per un numero e sommarne due voce per
voce), e si pretende che i numeri siano non negativi e sommino a uno,
altrimenti non è una media ma una combinazione qualsiasi.

Quel numero, il peso che la posizione $i$ mette sulla posizione $j$, lo
chiamiamo **peso di influenza** e lo scriviamo $\alpha_{ij}$: la lettera greca
$\alpha$ si legge «alfa», e i due numerini servono perché il peso dipende da
due posizioni, non da una (quanto la parola numero $i$ si appoggia alla parola
numero $j$). Due osservazioni lo rendono meno banale di quanto sembri.

La prima: l'influenza è a senso unico. Quanto «gatto» conta per capire
«salta» non è quanto «salta» conta per capire «gatto». Sono due domande
diverse e vogliono due risposte diverse.

La seconda: quel peso non lo scrive nessuno. Non c'è una regola grammaticale
programmata da qualche parte che dica «il soggetto pesa molto sul verbo». Il
peso viene calcolato dai vettori stessi, con una formula che ha dentro dei
parametri, e sono quei parametri a essere appresi dai dati. Il modello non
riceve la grammatica: riceve la struttura del calcolo, e la grammatica (o
qualcosa che le somiglia) è ciò che deve venir fuori perché aiuta a
indovinare la parola dopo.

Per misurare quanto una parola riguardi un'altra si usa lo strumento
della sezione di algebra lineare: il prodotto scalare, che è grande quando due
vettori puntano dalla stessa parte. Ma non fra i vettori originali, per due
ragioni. Il prodotto scalare è simmetrico, e abbiamo appena detto che
l'influenza non lo è; e poi *quanto* una parola conta e *che cosa* quella
parola porta con sé sono informazioni diverse. Che «gatto» sia un nome in
funzione di soggetto è ciò che lo rende rilevante per il verbo; che sia un
animale piccolo e domestico è ciò che ha da dire. Servono tre estratti
diversi dello stesso vettore, non uno.

`````

`````{tab} Superiore

Siano $\mathbf{e}_1,\dots,\mathbf{e}_n \in \mathbb{R}^d$ i vettori in
ingresso. L'uscita dell'aggregazione, per la posizione $i$, è

$$
\mathbf{o}_i = \sum_{j=1}^{n} \alpha_{ij}\,\mathbf{c}_j ,
\qquad \alpha_{ij} \ge 0, \qquad \sum_j \alpha_{ij} = 1,
$$

dove $\alpha_{ij}$ è il **peso di influenza** della posizione $j$ su $i$ e
$\mathbf{c}_j$ è il contenuto che $j$ mette a disposizione. Restano da
definire due cose: da dove vengono i pesi e da dove viene il contenuto.

(Il simbolo è $\mathbf{o}_i$ e non $\mathbf{h}_i$ perché questa è l'uscita di
**una** aggregazione, e vive in $\mathbb{R}^k$ con $k < d$. La
rappresentazione contestuale $\mathbf{h}_i \in \mathbb{R}^d$ promessa poco fa
si ottiene solo più avanti, ricomponendo le $H$ aggregazioni parallele.)

Entrambi da **tre trasformazioni lineari apprese** applicate ai vettori
stessi. Con $\mathbf{W}^A, \mathbf{W}^B, \mathbf{W}^C \in \mathbb{R}^{k
\times d}$:

$$
\mathbf{a}_i = \mathbf{W}^A \mathbf{e}_i, \qquad
\mathbf{b}_j = \mathbf{W}^B \mathbf{e}_j, \qquad
\mathbf{c}_j = \mathbf{W}^C \mathbf{e}_j .
$$

Ogni riga di queste matrici è un rivelatore di caratteristiche: il suo
prodotto scalare con $\mathbf{e}$ misura quanto quel vettore esibisce la
caratteristica che la riga rappresenta, e l'uscita in $\mathbb{R}^k$ è la
lista di $k$ misure così ottenute. I tre ruoli sono distinti:
$\mathbf{W}^A$ estrae, dalla posizione che si sta aggiornando, ciò a cui è
ricettiva; $\mathbf{W}^B$ estrae, da ogni posizione di contesto, ciò che la
rende rilevante; $\mathbf{W}^C$ estrae ciò che quella posizione trasmette
una volta che il peso è stato deciso.

Il **punteggio di influenza** è il prodotto scalare fra i primi due,
riscalato:

$$
r_{ij} = \frac{\mathbf{a}_i^\top \mathbf{b}_j}{\sqrt{k}} .
$$

La divisione per $\sqrt{k}$ è igiene numerica: se le componenti dei due
vettori sono all'incirca indipendenti, a media nulla e varianza unitaria, il
loro prodotto scalare ha varianza $k$, e punteggi che crescono con la
dimensione saturerebbero il passo successivo azzerandone i gradienti.
Poiché $\mathbf{W}^A \neq \mathbf{W}^B$, il punteggio è **asimmetrico**:
$r_{ij} \neq r_{ji}$ in generale, che è precisamente ciò che serviva.

I punteggi sono numeri reali di segno qualunque, e vanno trasformati in pesi.
La trasformazione è quella che i manuali di reti neurali chiamano *softmax* e
che in statistica ha un nome più informativo, **logit multinomiale inverso**:

$$
\alpha_{ij} = \frac{e^{r_{ij}}}{\sum_{m} e^{r_{im}}} .
$$

È l'inversa della trasformazione log-odds generalizzata a più di due
categorie: nella regressione logistica multinomiale, se il log-odds della
categoria $m$ rispetto a una di riferimento è $z_m$, la probabilità di quella
categoria è $e^{z_m}/\sum_{m'} e^{z_{m'}}$, dove la somma comprende anche la
categoria di riferimento, che entra con $z = 0$. Qui il punteggio $r_{ij}$
gioca il ruolo del log-odds «di essere influenzati dalla posizione $j$», ma di
categoria di riferimento non ce n'è nessuna: i punteggi sono definiti solo a
meno di una costante additiva comune, ed è precisamente l'invarianza che il
*log-sum-exp* sfrutta quando sottrae il massimo. È
differenziabile ovunque, monotona nei punteggi e assegna peso positivo a
tutte le posizioni, per quanto trascurabile. In pratica si calcola sempre
sottraendo prima il massimo, con il *log-sum-exp* della sezione di analisi
numerica, altrimenti l'esponenziale trabocca.

L'operazione completa, per la posizione $i$, è quindi: proietta con le tre
matrici, calcola i punteggi, normalizzali in pesi, e restituisci la media
pesata dei contenuti.

`````

Questo è tutto, ed è qui che si annidano i nomi. Nella letteratura le tre
proiezioni si chiamano *query*, *key* e *value*, in prestito dalle basi di
dati: si interroga una base con una chiave per ottenere un valore. La
metafora è imperfetta, perché non c'è nessuna base di dati e nessuna
interrogazione: c'è un vettore confrontato con altri vettori. Il libro
continua a usare i nomi standard, perché sono quelli dei paper e del codice,
ma vale la pena tenere accanto la traduzione. La tabella che segue serve da
dizionario per quando quei nomi si incontreranno altrove: per seguire questa
sezione non serve, ed è la ragione per cui arriva adesso e non all'inizio.

| il nome che si incontra | qui | che cosa fa davvero |
|---|---|---|
| *query*, $\mathbf{W}^Q$ | $\mathbf{W}^A$ | estrae ciò a cui la posizione da aggiornare è ricettiva |
| *key*, $\mathbf{W}^K$ | $\mathbf{W}^B$ | estrae ciò che una posizione di contesto offre, per decidere quanto pesa |
| *value*, $\mathbf{W}^V$ | $\mathbf{W}^C$ | estrae ciò che quella posizione trasmette, deciso il peso |
| *attention score* | punteggio di influenza $r_{ij}$ | il prodotto scalare fra i primi due, riscalato |
| *attention weight* | peso di influenza $\alpha_{ij}$ | il punteggio normalizzato: non negativo, a somma 1 |
| *softmax* | logit multinomiale inverso | la trasformazione della regressione logistica multinomiale |
| *head* | relazione | una delle $H$ copie parallele dello stesso meccanismo |

## Una forma bilineare di rango basso

C'è un modo più compatto di guardare il punteggio di influenza, e riconoscerlo
chiarisce in un colpo solo quanti parametri servono e perché. Il titolo di
questa sezione anticipa due parole tecniche, e conviene scioglierle subito.
Una **forma bilineare** è un modo di misurare l'accordo fra due liste di
numeri passando per una tabella: si prende la prima lista, la si fa attraversare
dalla tabella e si fa il prodotto scalare con la seconda. Il **rango** di una
tabella è quante righe davvero indipendenti contiene, cioè quante ne servono
per ricostruire tutte le altre combinandole: una tabella grande di rango basso
è grande solo all'apparenza.

```{figure} ../figures/forma-bilineare-rango-basso.svg
:name: fig-forma-bilineare
:alt: "A sinistra un quadrato grande etichettato M, di lato d, che rappresenta la matrice di tutte le affinità possibili fra caratteristiche; a destra lo stesso quadrato è mostrato come prodotto di due rettangoli sottili, alti k e larghi d, le trasposte delle due matrici di proiezione. Sotto, il conteggio dei numeri da imparare: circa 151 milioni per il quadrato pieno, circa 3,1 milioni per i due rettangoli."
:width: 88%

Il quadrato pieno e la sua versione fattorizzata. Imparare il quadrato per
intero significherebbe un numero da regolare per ogni coppia di coordinate;
ricavandolo dal prodotto di due tabelle sottili se ne regolano quarantotto
volte meno, e in cambio si rinuncia a tutti i quadrati che quel prodotto non
sa produrre.
```

`````{tab} Elementare

Immagina una tabella gigantesca che, per ogni coppia di caratteristiche
possibili, dica quanto quella dell'una si accorda con quella dell'altra: la
riga «è un verbo di movimento» incrocia la colonna «è un nome animato» con un
numero alto, e così via per ogni coppia. Sarebbe il modo più generale di
misurare l'affinità fra due parole, e sarebbe anche insostenibile. Prendiamo i
numeri veri di GPT-3, il modello di cui torneremo a fare i conti più avanti:
$12\,288$ coordinate per vettore. Quella tabella avrebbe $12\,288$ righe e
$12\,288$ colonne, cioè 151 milioni di caselle, e ce ne vorrebbe una per ogni
relazione e per ogni strato.

La scorciatoia è non imparare mai la tabella intera, ma due tabelle sottili
il cui prodotto la ricostruisce approssimativamente. Con $128$ righe ciascuna
(sempre GPT-3: è la stessa scelta di progetto) sono 3,1 milioni di numeri
invece di 151 milioni, quarantotto volte meno
({numref}`fig-forma-bilineare`).

Si perde qualcosa, ovviamente. Le tabelle ottenute in questo modo non sono
tutte quelle possibili, sono una famiglia ristretta. Ma il risparmio non è
solo di memoria: meno parametri liberi significa anche meno modi di imparare
a memoria il rumore dei dati, ed è la stessa idea di *regolarizzazione* che il
capitolo sul machine learning discute a proposito dell'overfitting. La
scorciatoia, spesso, è anche il motivo per cui il modello generalizza.

`````

`````{tab} Superiore

Sostituendo le definizioni, il punteggio di influenza si riscrive senza mai
nominare $\mathbf{a}$ e $\mathbf{b}$:

$$
r_{ij} \;\propto\; \mathbf{a}_i^\top \mathbf{b}_j
= (\mathbf{W}^A \mathbf{e}_i)^\top (\mathbf{W}^B \mathbf{e}_j)
= \mathbf{e}_i^\top \underbrace{(\mathbf{W}^A)^\top \mathbf{W}^B}_{\textstyle \mathbf{M}} \,\mathbf{e}_j .
$$

È una **forma bilineare** in $\mathbf{e}_i$ e $\mathbf{e}_j$: una funzione
$\beta(\mathbf{x}, \mathbf{y})$ lineare in ciascun argomento quando l'altro è
tenuto fisso. La forma bilineare più generale su $\mathbb{R}^d$ si scrive
$\mathbf{x}^\top \mathbf{M} \mathbf{y}$ con $\mathbf{M}$ di dimensione $d
\times d$, e qui $\mathbf{M} = (\mathbf{W}^A)^\top \mathbf{W}^B$.

La differenza sta nel rango. Se $\mathbf{W}^A$ e $\mathbf{W}^B$ sono $k
\times d$ con $k < d$, allora $\operatorname{rank}(\mathbf{M}) \le k$: non si
sta imparando una forma bilineare qualsiasi, ma una **parametrizzazione a
rango basso**, cioè un elemento di una famiglia ristretta. I conti, con i
valori di GPT-3 ($d = 12\,288$, $k = d/H = 128$): la matrice piena avrebbe
$d^2 = 150\,994\,944$ parametri, la coppia di fattori ne ha $2kd =
3\,145\,728$, quarantotto volte meno. Il vincolo di rango agisce come
regolarizzatore, nello stesso senso in cui lo fa in una fattorizzazione di
matrice per i sistemi di raccomandazione.

La lettura in termini di $\mathbf{M}$ non è solo cosmesi: è il punto di
partenza dell'analisi meccanicistica dei circuiti nei Transformer
{cite}`elhage2021mathematical`, dove il prodotto $(\mathbf{W}^A)^\top
\mathbf{W}^B$ (la matrice *QK*) e l'analogo prodotto sul lato del contenuto
(la matrice *OV*) sono gli oggetti da studiare, mentre le singole proiezioni
non lo sono. Il perché è nella prossima sezione.

`````

## Due matrici che nessuno può identificare

Da questa fattorizzazione discende un fatto che vale la pena sapere prima di
provare a interpretare i pesi di un modello.

`````{tab} Elementare

Prendi due mappe della stessa città, identiche in tutto tranne che la seconda
è ruotata di trenta gradi rispetto alla prima. Ogni distanza misurata sulla
prima si ritrova uguale sulla seconda: nessuna misura fatta sulle mappe può
dire quale delle due sia «quella giusta», perché le due mappe descrivono
esattamente la stessa città.

Alle due tabelle che calcolano l'influenza succede lo stesso, e per la stessa
ragione. Ruotare una mappa vuol dire cambiare la coppia di numeri con cui si
indica ogni luogo, lasciando i luoghi dove sono: ruotare una tabella di numeri
è esattamente questo, riscrivere le stesse quantità rispetto a un'altra scelta
di assi. Le due tabelle si possono ruotare insieme, in infiniti modi, senza
che nemmeno un numero cambi nell'uscita del modello: quello che una gira, per
così dire, l'altra lo gira indietro. Ne segue che chiedersi «che cosa
significa la riga 7 di quella tabella?» è una domanda mal posta: la riga 7 non
è determinata dai dati, lo è solo il modo in cui le righe lavorano insieme.

Questo è un guaio per chi vuole capire cosa un modello ha imparato, ed è
invece una fortuna per chi lo addestra: quando tantissime configurazioni di
parametri diversi danno esattamente lo stesso risultato, trovarne una è molto
più facile che trovare un ago in un pagliaio.

`````

`````{tab} Superiore

Sia $\mathbf{O} \in \mathbb{R}^{k \times k}$ ortogonale, cioè $\mathbf{O}^\top
\mathbf{O} = \mathbf{I}$. Sostituendo $\mathbf{W}^A \to \mathbf{O}\mathbf{W}^A$
e $\mathbf{W}^B \to \mathbf{O}\mathbf{W}^B$:

$$
(\mathbf{O}\mathbf{W}^A \mathbf{e}_i)^\top (\mathbf{O}\mathbf{W}^B \mathbf{e}_j)
= \mathbf{e}_i^\top (\mathbf{W}^A)^\top \mathbf{O}^\top \mathbf{O}
\mathbf{W}^B \mathbf{e}_j
= \mathbf{e}_i^\top (\mathbf{W}^A)^\top \mathbf{W}^B \mathbf{e}_j ,
$$

cioè i punteggi non cambiano. Le due matrici non sono **identificabili**
separatamente: lo è solo il loro prodotto $\mathbf{M}$
{cite}`shalizi2023attention`. L'ortogonalità, per giunta, è più di quanto
serva: per ogni $\mathbf{S} \in \mathbb{R}^{k\times k}$ invertibile,
$\mathbf{W}^A \to \mathbf{S}^{-\top}\mathbf{W}^A$ e $\mathbf{W}^B \to
\mathbf{S}\mathbf{W}^B$ lasciano $\mathbf{M}$ invariata, e la varietà delle
soluzioni equivalenti è quindi ancora più grande.

E poiché anche gli embedding sono appresi, l'indeterminazione si estende a
loro: per ogni $\mathbf{R} \in \mathbb{R}^{d \times d}$ invertibile,
sostituire $\mathbf{e} \to \mathbf{R}\mathbf{e}$ insieme a
$\mathbf{W}^A \to \mathbf{W}^A \mathbf{R}^{-1}$, $\mathbf{W}^B \to
\mathbf{W}^B \mathbf{R}^{-1}$ **e** $\mathbf{W}^C \to \mathbf{W}^C
\mathbf{R}^{-1}$ lascia il modello identico. La terza sostituzione non è
facoltativa: senza di essa restano invariati i punteggi $r_{ij}$, ma i
contenuti $\mathbf{c}_j = \mathbf{W}^C\mathbf{e}_j$ diventano
$\mathbf{W}^C\mathbf{R}\mathbf{e}_j$ e l'uscita cambia. (Sul modello intero
l'invarianza si estende a tutto ciò che legge o scrive quel flusso: anche
$\mathbf{W}^O$ e le matrici della parte non lineare vanno composte con
$\mathbf{R}$ dal lato giusto, e i vettori d'uscita $\mathbf{u}_v$ con
$\mathbf{R}^{-\top}$.)

La situazione è la stessa della **rotazione dei fattori** nell'analisi
fattoriale, dove i fattori estratti sono determinati solo a meno di una
trasformazione ortogonale, ed è il motivo per cui esistono i criteri di
rotazione (varimax e simili) e le liti su quale usare. Due conseguenze,
opposte di segno. Per l'interpretabilità è una cattiva notizia: le quantità
che hanno senso studiare sono quelle invarianti (i prodotti, i pesi
$\alpha_{ij}$, le direzioni nello spazio delle rappresentazioni), non le
singole righe delle matrici, e il capitolo sull'interpretabilità ci torna
sopra. Per l'ottimizzazione è una buona notizia: l'insieme delle soluzioni
equivalenti è una varietà continua, non un punto isolato, e un paesaggio con
tanti minimi equivalenti è molto più facile da scendere di uno con un solo
minimo stretto.

`````

## Molte relazioni, non molte teste

Un solo terzetto di matrici impara un solo schema di influenza. Ma le parole
si legano fra loro in modi diversi: il soggetto al verbo, l'aggettivo al
nome, il pronome al suo antecedente («Marta disse che era stanca»: chi era
stanca?), e poi la semplice vicinanza. La scelta architetturale è replicare
il meccanismo un certo numero di volte in parallelo, con altrettanti terzetti
di matrici indipendenti, e ricomporne le uscite. Quel numero si scrive $H$,
dall'inglese *head*: è la stessa lettera che nella sezione sulla teoria
dell'informazione indicava l'entropia, e qui non ha niente a che vedere con
quella, è solo un conteggio.

`````{tab} Elementare

È come far leggere la stessa frase a più lettori, ciascuno con il suo
evidenziatore. Alla fine i fogli si sovrappongono e la frase risulta letta da
più punti di vista insieme.

La parte importante è quello che *non* succede. Nessuno assegna i mestieri:
non c'è una copia incaricata dei pronomi e una della concordanza. Le $H$
copie sono identiche nella struttura, partono da numeri casuali e vengono
addestrate tutte con lo stesso identico obiettivo, indovinare la parola dopo.

Si specializzano lo stesso, e la ragione è economica: se due copie
imparassero la stessa cosa, una delle due sarebbe sprecata, e sprecarla
costa in accuratezza. C'è quindi una pressione implicita a differenziarsi,
perché ogni copia contribuisce di più quando cattura qualcosa che le altre si
perdono. È lo stesso motivo per cui, nell'analisi fattoriale, i fattori
estratti finiscono per essere diversi fra loro: non perché qualcuno abbia
distribuito i ruoli, ma perché catturare fonti di variazione distinte è il
modo efficiente di spiegare i dati.

Andando a guardare dentro modelli addestrati si trovano davvero copie che
seguono i legami sintattici, altre le catene di riferimento, altre la
vicinanza. Ma l'allineamento è parziale: molte non corrispondono a nessuna
categoria che i linguisti abbiano mai avuto bisogno di nominare, e altre ne
mescolano più d'una. Il modello non ha alcun concetto di «sintassi»: ha
trovato regolarità che riducono l'errore di previsione, e quelle regolarità
si sovrappongono in parte, non del tutto, alla struttura che i linguisti
hanno descritto.

`````

`````{tab} Superiore

Con $H$ terzetti $(\mathbf{W}^A_h, \mathbf{W}^B_h, \mathbf{W}^C_h)$, ciascuno
produce la propria uscita $\mathbf{o}^{(h)}_i \in \mathbb{R}^k$; le $H$
uscite vengono concatenate e riproiettate in $\mathbb{R}^d$ da una matrice
appresa $\mathbf{W}^O \in \mathbb{R}^{d \times Hk}$:

$$
\mathbf{h}_i = \mathbf{W}^O
\big[\, \mathbf{o}^{(1)}_i;\ \mathbf{o}^{(2)}_i;\ \dots;\ \mathbf{o}^{(H)}_i \,\big] .
$$

Ponendo $k = d/H$ il costo totale resta quello di una singola aggregazione a
dimensione piena. Nei modelli attuali $H$ va tipicamente da 12 a 96.

Conviene tenere separato ciò che è progettato da ciò che è scoperto. La
scelta di avere $H$ copie è architetturale e crea *capacità* di
specializzazione; quali relazioni emergano è deciso dall'ottimizzazione, che
è identica per tutte le copie. La differenziazione è un effetto della
ridondanza penalizzata implicitamente dalla loss, non di un vincolo esplicito.
L'analisi post-hoc dei modelli addestrati conferma un allineamento parziale
con le categorie linguistiche {cite}`clark2019what`: alcune copie tracciano
dipendenze sintattiche, altre la coreferenza, altre l'adiacenza, molte niente
di nominabile.

Vista in termini di algebra lineare, ogni copia definisce un modo diverso di
costruire medie pesate, e le $H$ copie insieme formano una base rispetto a
cui la rappresentazione contestuale viene costruita. Il termine standard per
una copia è *head*, «testa»: un nome che suggerisce un componente progettato
per una funzione (come le testine di un disco rigido) proprio dove la
funzione, se c'è, è emersa da sola.

`````

## Profondità: perché gli strati non collassano

Un giro di media pesata mescola ogni parola con il suo contesto immediato.
Ma alcune dipendenze sono lunghe: in «Lo scienziato che scoprì il
superconduttore ad alta temperatura ricevette il Nobel per la ...», per
arrivare a «fisica» bisogna collegare «Nobel» a «superconduttore» e a
«scienziato» scavalcando una subordinata intera. La risposta è impilare: l'uscita
di uno strato diventa l'ingresso del successivo, $L$ volte.

`````{tab} Elementare

Impilare da solo non basterebbe. Tutto quello che uno strato fa finora è
moltiplicare per dei numeri e sommare: in gergo si dice che l'operazione è
**lineare**. E fare due volte di fila un'operazione del genere non porta più
lontano che farne una sola, perché il risultato resta pur sempre una somma di
multipli dei numeri di partenza, con altri coefficienti. È lo stesso motivo
per cui raddoppiare e poi triplicare equivale a sestuplicare: due passaggi, un
solo passaggio possibile. Cento strati tutti lineari equivarrebbero quindi a
uno, e tutta la pila collasserebbe in un'unica tabella. Serve, fra
uno strato e l'altro, una funzione che *pieghi* i numeri, e la più usata è la
più semplice che si possa immaginare: azzerare i valori negativi e lasciar
passare i positivi. È la non linearità del capitolo sulle reti neurali, ed è
la ragione per cui la profondità aggiunge davvero qualcosa.

Due accorgimenti rendono la pila addestrabile. Il primo è che ogni strato non
riscrive la rappresentazione: le somma una correzione. Il vettore attraversa
la pila e ogni strato lo ritocca un po', come una bozza che passa fra molte
mani invece di essere riscritta da capo ogni volta. Il secondo è rimettere i
numeri su una scala standard dopo ogni passaggio, come azzerare la bilancia
fra una pesata e l'altra.

Nessun parametro è condiviso fra gli strati: ogni strato ha la sua copia
completa del meccanismo, moltiplicata per il numero di relazioni. Ed è da qui
che vengono i numeri da capogiro nei nomi dei modelli.

`````

`````{tab} Superiore

Posto $\mathbf{h}^{(0)}_i = \mathbf{e}_i$, ogni strato $\ell = 1, \dots, L$
calcola $\mathbf{h}^{(\ell)}_i$ dall'intera sequenza precedente
$\mathbf{h}^{(\ell-1)}_1, \dots, \mathbf{h}^{(\ell-1)}_n$. Tre ingredienti
oltre all'aggregazione contestuale.

**Non linearità.** Fra un'aggregazione e l'altra si applica a ogni posizione,
indipendentemente, una trasformazione del tipo $\mathbf{W}_2\,\phi(\mathbf{W}_1
\mathbf{h}_i + \mathbf{b}_1) + \mathbf{b}_2$ con $\phi$ non lineare
elemento per elemento, tipicamente $\phi(x) = \max(0, x)$. Senza $\phi$ la
composizione di $L$ mappe lineari sarebbe una mappa lineare: la profondità non
aggiungerebbe potenza espressiva.

**Aggiornamento additivo.** Ogni trasformazione è applicata nella forma
$\mathbf{h}_i \leftarrow \mathbf{h}_i + f(\mathbf{h}_i)$
{cite}`he2016deep`. La ragione è nel gradiente: la regola della catena
moltiplica le derivate strato per strato, e prodotti di molti fattori minori
di uno svaniscono esponenzialmente con la profondità. La struttura additiva
mette in ogni derivata un termine dell'identità, cioè una via diretta per il
gradiente accanto a quella che passa per $f$.

Quanto quella via resti davvero libera dipende però da dove si mette la
normalizzazione, e qui il testo non può promettere più di quanto la formula
mantenga. Nella forma del 2017 (*post-LN*, quella scritta qui sotto) la
normalizzazione sta **sopra** la somma, quindi il gradiente la attraversa a
ogni strato e viene moltiplicato per la sua Jacobiana, che identità non è: il
termine si attenua, tanto più quanto più il residuo cresce in norma, ed è il
motivo per cui il post-LN richiede un riscaldamento del tasso di
apprendimento. I modelli recenti spostano la normalizzazione a monte del
sotto-strato (*pre-LN*) e solo lì la scorciatoia resta pulita. Il capitolo sui
Transformer entra nel merito con i numeri.

**Normalizzazione.** Dopo ciascuno dei due sotto-strati i vettori vengono
standardizzati (media nulla e varianza unitaria sulle coordinate): un
accorgimento che tiene le grandezze in un intervallo trattabile lungo tutta
la pila.

I parametri non sono condivisi fra strati, e questo spiega gli ordini di
grandezza. Contiamoli per GPT-3 {cite}`brown2020language`, con $d = 12\,288$,
$L = 96$, $H = 96$. Per strato: quattro matrici $d \times d$ per la parte
contestuale (le tre proiezioni, che sommate su tutte le relazioni valgono
$Hkd = d^2$ ciascuna, più la riproiezione $\mathbf{W}^O$) e due matrici per
la parte non lineare, che internamente si allarga a $4d$, cioè $8d^2$. In
totale $12d^2 = 1{,}81 \cdot 10^9$ parametri per strato, che per $96$ strati
fanno $1{,}74 \cdot 10^{11}$; gli embedding di un vocabolario da $50\,257$
token ne aggiungono $0{,}6 \cdot 10^{9}$. Sono i 175 miliardi di parametri
di GPT-3.

`````

## L'ordine delle parole non è nella formula

C'è una cosa che la media pesata, così com'è scritta, non sa fare, ed è
istruttivo che il difetto si veda direttamente dalla formula.

Nel conto del punteggio fra due parole entrano i loro vettori e nient'altro:
non entra mai il **posto** che occupano nella frase, cioè se una è la seconda
parola e l'altra la quinta. Detto in simboli: i punteggi $r_{ij}$ dipendono dai
vettori che stanno nelle posizioni $i$ e $j$, non dai numeri $i$ e $j$. Se si
mescolano le parole della frase, i punteggi sono gli stessi, solo riordinati:
l'operazione tratta la sequenza come un **insieme**, un sacchetto di parole
senza un ordine. Ma
«il cane morde l'uomo» e «l'uomo morde il cane» non sono la stessa frase, e
un modello che le confonde non è un modello del linguaggio.

L'informazione di posizione va quindi immessa a mano. La via storica è
sommare a ogni embedding un vettore che codifica la posizione,
$\mathbf{e}_i \leftarrow \mathbf{e}_i + \mathbf{p}_i$
{cite}`vaswani2017attention`. Le soluzioni oggi prevalenti codificano invece
la posizione **relativa**, cioè la distanza $i - j$ e non gli indici presi
singolarmente, perché è la distanza fra due parole a governarne il legame, e
perché un modello che ha imparato la posizione assoluta $10\,000$ non l'ha
mai vista se le frasi di addestramento erano più corte. Il capitolo sui
Transformer entra nel merito degli schemi; qui interessava il punto
matematico, cioè che il meccanismo di base è invariante rispetto all'ordine e
che l'ordine va aggiunto da fuori.

Un secondo intervento riguarda invece la direzione del tempo. Poiché il
compito è prevedere la parola successiva, la somma sui contesti non può
correre su tutte le posizioni ma solo su quelle già viste, $j \le i$:
altrimenti il modello, per indovinare la parola dopo, potrebbe leggersela.

## Dall'ultimo vettore alla parola dopo

Dopo tutti gli strati resta un vettore, quello dell'ultima posizione, che
riassume tutto ciò che il modello ha estratto dal contesto. Convertirlo in una
distribuzione di probabilità sulle parole possibili richiede un ultimo passo,
e non è un passo nuovo: è lo stesso di prima.

`````{tab} Elementare

A ogni parola del vocabolario è associata una lista di numeri, sua e appresa
come tutte le altre. Per sapere quanto quella parola è plausibile qui, si
confronta la sua lista con il vettore finale: è di nuovo il prodotto scalare,
il conto dello scontrino, alto quando le due liste puntano dalla stessa parte.
Vengono fuori cinquantamila punteggi, uno per parola, e si trasformano in
probabilità con la stessa ricetta usata per i pesi di influenza: si elevano a
potenza e si dividono per il totale, così sommano a uno.

Vale la pena fermarsi su che cosa è appena successo, perché è la conclusione
di tutta la sezione. Prese le ultime due righe da sole, quello che il modello
fa alla fine è la cosa più ordinaria della statistica: ha una lista di numeri
che descrivono la situazione, la confronta con una lista per ogni risposta
possibile e ne ricava delle probabilità. È lo stesso schema con cui si stima
se un cliente restituirà un prestito, dati il suo reddito e la sua età. La
differenza, e non è piccola, è che lì i numeri che descrivono la situazione li
sceglie una persona (reddito, età, anzianità di lavoro), mentre qui sono
calcolati dalle decine di strati che li precedono. Tutta la sofisticazione
dell'architettura serve a **costruire i numeri giusti da dare in pasto a un
metodo vecchio**.

`````

`````{tab} Superiore

A ogni token $v$ del vocabolario è associato un vettore appreso
$\mathbf{u}_v \in \mathbb{R}^d$, e il punteggio del token è il prodotto
scalare $z_v = \mathbf{u}_v^\top \mathbf{h}^{(L)}_n$: alto quando la
direzione di quel token si allinea con la direzione della rappresentazione
finale. I punteggi diventano probabilità con lo stesso logit multinomiale
inverso di prima:

$$
P(w_{n+1} = v \mid w_1, \dots, w_n)
= \frac{e^{z_v}}{\sum_{v' \in \mathcal{V}} e^{z_{v'}}} .
$$

Detto in una riga: **l'ultimo strato di un modello linguistico è una
regressione logistica multinomiale** con $|\mathcal{V}|$ categorie, in cui il
vettore dei regressori non è stato scelto da un analista ma calcolato dagli
$L$ strati precedenti. Tutta la sofisticazione dell'architettura serve a
costruire delle buone covariate (cioè le variabili esplicative, i «numeri che
descrivono il caso») per un modello statistico fra i più antichi e più
studiati che ci siano.

I vettori $\mathbf{u}_v$, per inciso, in molti modelli non sono parametri
nuovi: sono le righe della matrice di embedding $\mathbf{E}$, riusate al
contrario (*weight tying*). È il motivo per cui, nel conteggio dei parametri di
poco fa, gli embedding sono stati contati una volta sola.

`````

## Addestrare: massima verosimiglianza, ancora

Restano le manopole: le liste di numeri di ogni token, le tabelle di
proiezione di ogni relazione di ogni strato, le riproiezioni, le tabelle della
parte non lineare. Miliardi di numeri, e un solo principio per fissarli, quello
della sezione su probabilità e statistica: scegliere i valori che rendono i
dati osservati i più plausibili.

`````{tab} Elementare

Il principio è quello della moneta lanciata dieci volte. Lì c'era una manopola
sola, la probabilità di far testa, e si cercava il valore che rendeva più
probabile ciò che si era visto: sette teste su dieci portavano a $0{,}7$. Qui
le manopole sono miliardi e ciò che si è visto è un pezzo di internet, ma la
domanda è identica: quali regolazioni rendono meno sorprendente il testo che è
stato scritto davvero?

C'è una semplificazione che rende il conto praticabile, e vale la pena
notarla. Per ogni posizione del testo il modello produce una distribuzione su
tutto il vocabolario, ma la realtà, lì, non è una distribuzione: è una parola
sola, quella che è effettivamente occorsa. Del sacco di probabilità che il
modello ha prodotto interessa quindi un numero solo, quello assegnato alla
parola giusta, e l'obiettivo si riduce a: fai in modo che quel numero sia il
più alto possibile, in ogni punto del testo. È esattamente la sorpresa media
della sezione sulla teoria dell'informazione, misurata sulla realtà: minimizzare
quella e massimizzare la plausibilità dei dati sono la stessa operazione,
scritta due volte.

Le manopole si girano come nell'escursionista nella nebbia: si guarda in che
direzione l'errore cala più in fretta e si fa un passo di lì. Con una
scorciatoia obbligata, però: calcolare la pendenza su tutto il testo del mondo
a ogni passo è impensabile, quindi la si calcola ogni volta su un pugno di
brani presi a caso. Il risultato è una direzione un po' storta, ma storta per
caso, e su tanti passi gli errori si compensano invece di accumularsi.

`````

`````{tab} Superiore

$$
\max_{\theta} \ \sum_{\text{sequenze}} \sum_{i=1}^{n}
\log P_{\theta}(w_i \mid w_1, \dots, w_{i-1}) .
$$

È **massima verosimiglianza**, con il logaritmo che trasforma il prodotto
della regola della catena in una somma. Cambiando segno e mediando si
ottiene la log-verosimiglianza negativa, che è poi la **cross-entropia** fra
la distribuzione vera e quella del modello: la sezione sulla teoria
dell'informazione ha già mostrato che sono la stessa cosa, e vale la pena
vedere perché in questo caso specifico. La distribuzione «vera» su ogni
posizione è *degenere*, cioè mette tutta la probabilità su un solo esito, la
parola che è effettivamente occorsa; la sua entropia è nulla, e la
cross-entropia si riduce a $-\log q(\text{parola occorsa})$. Minimizzare la
sorpresa media del modello davanti alla realtà e massimizzare la
verosimiglianza sono, letteralmente, la stessa formula. (È anche il caso in cui
il «pavimento» della loss è zero: la sezione sulla teoria dell'informazione
osservava che il minimo teorico è $H(p)$, ed è vero della distribuzione
condizionata vera del processo, non di questo bersaglio empirico.)

La massimizzazione avviene per **discesa del gradiente stocastica**: si
calcola il gradiente su un sottoinsieme casuale di sequenze invece che
sull'intero corpus, il che dà una stima rumorosa ma **non distorta**, cioè
sbagliata in media di zero: gli errori dei singoli passi si compensano invece
di sommarsi in una direzione. I parametri si aggiornano nella direzione che
migliora l'obiettivo. Che il gradiente sia
calcolabile attraverso decine di strati non è un miracolo, è la regola della
catena: il modello, dagli embedding fino alle probabilità, è una composizione
di funzioni differenziabili, e il capitolo su PyTorch mostra la macchina che
lo fa in automatico.

`````

Se qualcosa in tutto questo somiglia alla magia, non è la stima simultanea di
miliardi di parametri: è che funzioni. L'apparenza di un'unica gigantesca
ottimizzazione è comunque fuorviante, perché l'addestramento di un modello di
frontiera è organizzato in fasi, con un pre-addestramento sul testo grezzo,
un affinamento su esempi curati e una fase di allineamento alle preferenze
umane {cite}`ouyang2022training`, il tutto con programmi di riscaldamento e
decadimento del tasso di apprendimento. Ne parla il capitolo sui Transformer;
la matematica dell'obiettivo, però, resta questa.

## Il modello, in una pagina

`````{tab} Elementare

Il modello intero, in sette righe di italiano:

1. il testo viene spezzato in token, e ogni token diventa un punto in uno
   spazio di qualche migliaio di dimensioni;
2. a ogni punto si aggiunge l'informazione di dove si trova nella frase;
3. ogni posizione guarda tutte quelle che la precedono e assegna a ciascuna
   un peso, calcolato confrontando due estratti diversi dei rispettivi
   vettori;
4. i pesi, resi positivi e a somma uno, servono a fare una media di un terzo
   estratto: è la nuova rappresentazione di quella posizione;
5. la stessa cosa avviene in parallelo in molte copie indipendenti, i cui
   risultati vengono rimescolati insieme;
6. il risultato viene sommato al vettore di partenza, piegato da una funzione
   non lineare, rimesso in scala, e il tutto ricomincia da capo per decine di
   strati;
7. l'ultimo vettore viene confrontato con un vettore per ogni parola del
   vocabolario, e i confronti diventano le probabilità della parola dopo.

Addestrare significa girare tutte le manopole (miliardi) finché le parole che
sono davvero occorse nel testo di addestramento risultano le meno
sorprendenti possibile.

`````

`````{tab} Superiore

Ingresso: una sequenza $(w_1, \dots, w_n)$ con $w_i \in \mathcal{V}$.

Rappresentazione iniziale, con $\mathbf{e}_{w} \in \mathbb{R}^d$ la riga della
matrice di embedding relativa al token $w$ e $\mathbf{p}_i \in \mathbb{R}^d$
l'informazione di posizione:

$$
\mathbf{h}^{(0)}_i = \mathbf{e}_{w_i} + \mathbf{p}_i .
$$

Aggregazione contestuale, allo strato $\ell$ e nella relazione $h$:

$$
\mathbf{a}_i = \mathbf{W}^A_{\ell,h}\,\mathbf{h}^{(\ell-1)}_i, \qquad
\mathbf{b}_j = \mathbf{W}^B_{\ell,h}\,\mathbf{h}^{(\ell-1)}_j, \qquad
\mathbf{c}_j = \mathbf{W}^C_{\ell,h}\,\mathbf{h}^{(\ell-1)}_j ,
$$

$$
\alpha_{ij} = \frac{\exp\!\big(\mathbf{a}_i^\top \mathbf{b}_j / \sqrt{k}\big)}
{\sum_{m \le i} \exp\!\big(\mathbf{a}_i^\top \mathbf{b}_m / \sqrt{k}\big)} ,
\qquad
\mathbf{o}^{(h)}_i = \sum_{j \le i} \alpha_{ij}\,\mathbf{c}_j .
$$

Ricomposizione delle $H$ relazioni, aggiornamento additivo e non linearità
(con $\mathbf{z}_i$ il vettore intermedio fra i due sotto-strati; la
normalizzazione è scritta in coda a ciascuno dei due, come nell'articolo del
2017, mentre i modelli recenti la spostano a monte):

$$
\tilde{\mathbf{h}}_i = \mathbf{W}^O_\ell
\big[\mathbf{o}^{(1)}_i; \dots; \mathbf{o}^{(H)}_i\big],
\qquad
\mathbf{z}_i = \operatorname{Norm}\!\big(
\mathbf{h}^{(\ell-1)}_i + \tilde{\mathbf{h}}_i\big) ,
$$

$$
\mathbf{h}^{(\ell)}_i = \operatorname{Norm}\!\big(
\mathbf{z}_i + \mathbf{W}_2 \max(0, \mathbf{W}_1 \mathbf{z}_i)\big) .
$$

Uscita, con $\mathbf{u}_v \in \mathbb{R}^d$ il vettore appreso del token $v$:

$$
P(w_{n+1} = v \mid w_1, \dots, w_n) =
\frac{\exp\!\big(\mathbf{u}_v^\top \mathbf{h}^{(L)}_n\big)}
{\sum_{v' \in \mathcal{V}} \exp\!\big(\mathbf{u}_{v'}^\top \mathbf{h}^{(L)}_n\big)} .
$$

Addestramento, su tutti i parametri $\theta$ raccolti insieme:

$$
\max_{\theta} \ \sum_{\text{sequenze}} \sum_{i=1}^{n}
\log P_{\theta}(w_i \mid w_1, \dots, w_{i-1}) .
$$

`````

## Quello che la matematica non dice

Aver ridotto un modello linguistico a queste formule chiarisce anche i suoi
limiti, che non sono difetti di implementazione ma conseguenze della forma
dell'oggetto.

Il modello impara da testo e basta: non ha modo di verificare un'affermazione
contro il mondo, e ottimizza la verosimiglianza, non la verità. Genera
volentieri frasi plausibili e false, perché plausibile è esattamente ciò che
l'obiettivo premia. Non esegue deduzioni: riproduce schemi di ragionamento
ben rappresentati nei dati, il che è utilissimo e non è la stessa cosa,
come si vede cambiando i nomi delle variabili in un problema di logica o
immergendolo in una storia insolita. E vede solo dentro una finestra di
contesto fissata a progetto: quello che sta prima, semplicemente, non c'è.

Resta la domanda aperta, che è anche la più interessante. Nessuna delle
formule qui sopra menziona la sintassi, i concetti o le relazioni fra
concetti, eppure al crescere di dati e parametri la qualità delle previsioni
migliora in modo regolare {cite}`kaplan2020scaling` e compaiono comportamenti
che è naturale descrivere proprio in quei termini {cite}`brown2020language`.
Non è la
prima volta che succede in una scienza: la termodinamica emerge dalla
meccanica statistica, la fluidodinamica dalle interazioni fra molecole, lo
stormo dal comportamento del singolo storno. In ogni caso il fenomeno
macroscopico non si legge nelle regole microscopiche, ma non per questo è
meno reale. Perché la sola previsione della parola successiva porti così
lontano è, onestamente, ancora una questione aperta, e la sezione sui grandi
modelli linguistici, nel capitolo sui Transformer, discute anche i motivi per
dubitare che quelle «abilità emergenti» siano tutte quel che sembrano.

## In pratica, con NumPy

Trenta righe bastano per l'intero meccanismo su una frase di quattro parole:
punteggi asimmetrici, pesi normalizzati, media pesata, e le due verifiche
delle sezioni precedenti (la forma bilineare di rango basso e l'invarianza
per rotazione).

```python
import numpy as np

rng = np.random.default_rng(0)

parole = ["il", "gatto", "nero", "salta"]
d, k = 6, 3                                  # dimensione embedding, dimensione proiezioni
E = rng.normal(size=(len(parole), d))        # una riga per parola: gli embedding

W_A = rng.normal(size=(k, d)) / np.sqrt(d)   # cosa la posizione i e' disposta a ricevere
W_B = rng.normal(size=(k, d)) / np.sqrt(d)   # cosa la posizione j offre
W_C = rng.normal(size=(k, d)) / np.sqrt(d)   # cosa la posizione j trasmette

A, B, C = E @ W_A.T, E @ W_B.T, E @ W_C.T    # tre proiezioni, (n, k) ciascuna

R = A @ B.T / np.sqrt(k)                     # punteggi di influenza r_ij

# asimmetria: l'influenza di "gatto" su "salta" non e' quella di "salta" su "gatto"
i, j = parole.index("salta"), parole.index("gatto")
print(round(R[i, j], 3), round(R[j, i], 3))  # -0.503  -0.272

def logit_multinomiale_inverso(z):           # la "softmax", riga per riga
    z = z - z.max(axis=-1, keepdims=True)    # log-sum-exp: evita l'overflow
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

alpha = logit_multinomiale_inverso(R)
print(np.round(alpha[i], 3), alpha[i].sum())  # [0.208 0.109 0.425 0.258]  1.0

contestuali = alpha @ C                      # media pesata: le nuove rappresentazioni

# gli stessi punteggi come forma bilineare, con una sola matrice di rango <= k
M = W_A.T @ W_B                              # (d, d)
print(M.shape, np.linalg.matrix_rank(M))     # (6, 6) 3
print(np.allclose(E @ M @ E.T / np.sqrt(k), R))   # True

# non identificabilita': ruotare insieme W_A e W_B non cambia nulla
O, _ = np.linalg.qr(rng.normal(size=(k, k)))
R_ruotato = (E @ (O @ W_A).T) @ (E @ (O @ W_B).T).T / np.sqrt(k)
print(np.allclose(R, R_ruotato))             # True
```

Le ultime due verifiche sono il contenuto di due sezioni intere: la matrice
`M` è $6 \times 6$ ma ha rango $3$, e le due proiezioni ruotate producono
punteggi identici alle originali fino all'ultima cifra.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Un modello linguistico è una macchina che assegna probabilità alla parola
  successiva. Sembra poco ed è tutto: scrivere è scegliere una parola fra
  quelle probabili, aggiungerla e rifare la domanda.
- Una tabella di tutti i contesti possibili non si può costruire (dieci parole
  su un vocabolario da 50 000 fanno un numero con 47 cifre, il doppio delle
  molecole d'acqua degli oceani), quindi al posto di un quaderno da consultare
  serve una funzione da calcolare, che generalizzi anche ai contesti mai visti.
- Il motore è una **media pesata**: per capire una parola si mescola un po' di
  ciò che sanno le altre, in proporzioni decise dai vettori stessi. Le
  proporzioni non le scrive nessuno: sono il risultato di parametri appresi,
  ed è per questo che il modello non riceve la grammatica ma può arrivarci.
- Servono tre estratti diversi di ogni vettore, perché *quanto* una parola
  conta e *che cosa* ha da dire sono informazioni diverse, e perché
  l'influenza è a senso unico: quanto «gatto» conta per «salta» non è quanto
  «salta» conta per «gatto».
- I nomi che si trovano in giro (*query*, *key*, *value*, *testa*) vengono da
  mestieri diversi e nascondono più di quanto spieghino: sotto ci sono tre
  proiezioni, un prodotto scalare e una media.
- Nessuno assegna i ruoli alle copie parallele del meccanismo: partono
  identiche e casuali, e si specializzano perché due copie che imparano la
  stessa cosa sprecano capacità.
- Gli strati non collassano l'uno nell'altro solo grazie a una funzione non
  lineare in mezzo, e restano addestrabili perché ognuno somma una correzione
  invece di riscrivere tutto da capo.
- Addestrare è girare miliardi di manopole finché le parole realmente occorse
  risultano le meno sorprendenti possibile: la stessa massima verosimiglianza
  della sezione sulla probabilità.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Un modello linguistico calcola i fattori $P(w_i \mid w_1,\dots,w_{i-1})$
  della regola della catena. La fattorizzazione è esatta; ciò che si
  approssima è ogni singolo fattore, con una funzione parametrica al posto di
  una tabella di conteggi ($|\mathcal{V}|^m$ contesti sono impraticabili ben
  prima di $m = 10$). Resta una catena di Markov di ordine finito, pari alla
  finestra di contesto.
- L'aggregazione contestuale è
  $\mathbf{o}_i = \sum_j \alpha_{ij}\mathbf{c}_j \in \mathbb{R}^k$ con
  $\alpha_{ij} = \operatorname{softmax}_j(\mathbf{a}_i^\top\mathbf{b}_j/\sqrt{k})$
  e $\mathbf{a}_i = \mathbf{W}^A\mathbf{e}_i$,
  $\mathbf{b}_j = \mathbf{W}^B\mathbf{e}_j$,
  $\mathbf{c}_j = \mathbf{W}^C\mathbf{e}_j$: le tre proiezioni sono *query*,
  *key* e *value*, la softmax è il logit multinomiale inverso. La
  rappresentazione $\mathbf{h}_i \in \mathbb{R}^d$ è la ricomposizione delle
  $H$ uscite, $\mathbf{h}_i = \mathbf{W}^O[\mathbf{o}^{(1)}_i;\dots;
  \mathbf{o}^{(H)}_i]$.
- Il punteggio è una **forma bilineare**
  $\mathbf{e}_i^\top \mathbf{M} \mathbf{e}_j$ con
  $\mathbf{M} = (\mathbf{W}^A)^\top\mathbf{W}^B$ di rango $\le k$: una
  parametrizzazione a rango basso che riduce i parametri (per GPT-3, $2kd$
  contro $d^2$: quarantotto volte meno) e regolarizza.
- $\mathbf{W}^A$ e $\mathbf{W}^B$ **non sono identificabili** separatamente:
  per ogni $\mathbf{O}$ ortogonale la coppia
  $(\mathbf{O}\mathbf{W}^A, \mathbf{O}\mathbf{W}^B)$ dà gli stessi punteggi, e
  con una $\mathbf{S}$ invertibile qualsiasi resta invariata $\mathbf{M}$.
  È il problema della rotazione dei fattori: male per l'interpretazione,
  bene per l'ottimizzazione. Riparametrizzare gli embedding richiede di
  trasformare **anche** $\mathbf{W}^C$: senza, i punteggi restano ma l'uscita
  cambia.
- Le $H$ copie parallele (*head*) sono capacità progettata, non ruoli
  assegnati: si differenziano perché la ridondanza costa accuratezza, e
  l'allineamento con le categorie linguistiche è parziale.
- La profondità richiede una non linearità (altrimenti $L$ mappe lineari
  collassano in una) e un aggiornamento additivo, che mette in ogni derivata un
  termine dell'identità; quanto quel cammino resti libero dipende però da dove
  sta la normalizzazione (in post-LN il gradiente la attraversa e il termine si
  attenua). Nulla è condiviso fra strati: $12d^2$ parametri per
  strato, che per GPT-3 ($d=12\,288$, $L=96$) fanno i 175 miliardi
  complessivi.
- L'ultimo strato è una **regressione logistica multinomiale** su
  $|\mathcal{V}|$ categorie, $P(v) \propto \exp(\mathbf{u}_v^\top
  \mathbf{h}^{(L)}_n)$, con covariate calcolate dai $L$ strati precedenti; i
  $\mathbf{u}_v$ di solito non sono parametri nuovi, ma le righe di
  $\mathbf{E}$ riusate al contrario (*weight tying*).
- L'obiettivo è la massima verosimiglianza
  $\max_\theta \sum \log P_\theta(w_i \mid w_{<i})$, che con target degenere
  coincide con la cross-entropia, ottimizzata per discesa del gradiente
  stocastica.
```
`````
