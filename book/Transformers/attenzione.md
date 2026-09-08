# Il meccanismo di attenzione

Quando leggi la frase "il gatto, che aveva dormito tutto il giorno sul
davanzale, saltò", e arrivi a "saltò", il tuo cervello non ripassa tutte le
parole in fila: torna dritto a "gatto". Sai *a che cosa prestare attenzione*.
Il meccanismo di attenzione dà alle reti neurali questa capacità: davanti
a una parola, guardare tutte le altre e pesare quanto ciascuna conta per
capirla.

Non è nato per fare il protagonista. Nel settembre del 2014 era un rattoppo,
inventato per migliorare le traduzioni delle reti che leggevano il testo una
parola alla volta (le reti ricorrenti che la {doc}`sezione sui modelli di
sequenza </NaturalLanguageProcessing/modelli-sequenza>` monta pezzo per pezzo)
{cite}`bahdanau2015neural`. Tre anni dopo il Transformer ci avrebbe costruito
sopra tutto il resto.

## L'idea: un'aggregazione pesata che dipende dal contenuto

Una cosa va detta prima di tutte, perché senza quella il resto sembra magia:
dentro una rete le parole non sono parole. Ognuna diventa una lista di
numeri, qualche centinaio, che la rete si è costruita imparando; nel gergo del
libro una lista del genere si chiama vettore. Il motivo per cui la cosa
conta è aritmetico: fare la media fra «gatto» e «muro» non vuol dire niente,
fare la media fra due liste di numeri sì, si sommano numero per numero.

Ed è esattamente quello che l'attenzione fa. Per ogni parola da elaborare
guarda tutte le altre parole della frase, decide quanto ciascuna conta per
capire quella, e ne mescola le liste in quella proporzione. Il risultato è una
versione della parola arricchita dal contesto: non più «salta» in astratto, ma
«salta» in *questa* frase. L'operazione ha un nome ordinario, media pesata,
e una proprietà che la rende diversa da tutte le medie pesate che si incontrano
altrove: i pesi non stanno scritti in nessun regolamento, li produce la frase
stessa, parola per parola. Di qui il nome tecnico dell'oggetto,
aggregazione pesata dipendente dal contenuto.

Il problema che questa idea viene a risolvere si vede bene guardando com'era
fatto un traduttore automatico prima. Erano due macchine attaccate: la prima
leggeva la frase di partenza e ne faceva un riassunto, la seconda leggeva solo
quel riassunto e da lì scriveva la traduzione. Le due metà hanno un nome che
torna in tutto il capitolo: l’encoder è la parte che legge, il decoder
quella che scrive, e la {doc}`sezione sulla struttura del Transformer
<architettura>` li smonta pezzo per pezzo. E il
riassunto era una sola lista di numeri, sempre lunga uguale: la stessa per una
frase di cinque parole e per una di cinquanta.

```{figure} ../figures/seq2seq-collo-di-bottiglia.svg
:name: fig-collo-di-bottiglia
:alt: "Schema di un seq2seq senza attenzione: le parole della frase in ingresso entrano una alla volta nell'encoder e vengono compresse in un unico vettore di contesto, disegnato come una strozzatura; da quel solo vettore il decoder deve generare tutta la traduzione, parola dopo parola."
:width: 92%

Il collo di bottiglia che l'attenzione viene a sciogliere. Tutta la frase
d'origine deve passare per un'unica lista di numeri, sempre lunga uguale: più
la frase è lunga, più quella lista è costretta a dimenticare.
```

{numref}`fig-collo-di-bottiglia` è il problema da cui nasce tutto, e con due
numeri si tocca con mano. Se il riassunto è lungo cinquecento numeri e la frase
è lunga cinquanta parole, a ogni parola tocca in media una decina di numeri per
raccontarsi: la prima e l'ultima si contendono lo stesso spazio, e a rimetterci
sono di solito quelle dell'inizio, viste per prime e sovrascritte da tutte
quelle che vengono dopo. L'attenzione toglie la strozzatura in un modo
sbrigativo: mentre scrive, il decoder smette di guardare il riassunto e va a
rileggersi *tutte* le parole d'origine, pesandole di volta in volta.

`````{tab} Elementare
Prendi la frase "Il gatto nero salta sul muro". Il modello sta elaborando la
parola "salta" e si chiede: chi salta? Come un lettore con l'evidenziatore,
ripassa la frase e assegna a ogni parola un'intensità di colore: "gatto"
fluorescente (è il soggetto), "muro" un colore medio (è la destinazione), "il"
e "sul" quasi trasparenti. Poi costruisce il significato di "salta" *in questa
frase* mescolando le informazioni di tutte le parole, ma in proporzione
all'evidenziatura: tanta parte di "gatto", un po’ di "muro", pochissimo del
resto.

Le intensità sono numeri veri, e per "salta" potrebbero venire così: gatto
0,52, muro 0,24, salta 0,10, nero 0,06, sul 0,05, il 0,03. Sono sei numeri, uno
per ogni parola della frase, e c'è anche "salta" stessa, perché ogni parola
guarda anche sé. Sommano a 1, ed è una regola fissa: l'attenzione distribuisce
sempre esattamente una unità di colore, quindi dare di più a "gatto" vuol dire
togliere a qualcun altro. «Mescolare in quella proporzione» significa allora
prendere il 52% della lista di numeri di "gatto", il 24% di quella di "muro", e
così via, e sommare il tutto: quello che ne esce è "salta" in questa frase e in
nessun'altra.

Due mestieri diversi, e conviene tenerli separati fin da subito. L'evidenziatore
decide quanto ciascuna parola conta; quello che finisce nel miscuglio è
invece l'informazione che ciascuna porta con sé, e le due cose la rete se le
tiene in due posti distinti. Un vantaggio si vede subito: "gatto" può
essere facile da trovare per una ragione (è un soggetto animato) e
consegnare tutt'altro (che è un felino, che è nero, che in questa frase è
il protagonista).

Quei numeri non stanno scritti da nessuna parte, e nessun programmatore li
ha battuti a tastiera: escono dalla frase, e da un'altra frase ne uscirebbero
altri. Quello che la rete impara durante l'addestramento è il criterio con cui
il colore va assegnato; le intensità le ricalcola daccapo ogni volta che le
arriva una frase nuova. E il criterio lo impara provando e correggendosi su
miliardi di frasi: ogni volta che il risultato non è quello giusto (una
traduzione sbagliata, la parola successiva sbagliata), viene ritoccato un
pochino nella direzione che avrebbe fatto sbagliare di meno, con lo stesso
provare-e-correggere del
{doc}`capitolo sulle reti neurali </RetiNeurali/overview>`.
`````

`````{tab} Superiore
Ogni parola (più precisamente ogni *token*, l'unità in cui la
{doc}`sezione sui tokenizzatori </NaturalLanguageProcessing/tokenizzatori>` ha
spezzato il testo) è rappresentata da un vettore. Una funzione di attenzione
prende un vettore query $\mathbf{q}$ e un insieme di coppie
chiave-valore $(\mathbf{k}_j, \mathbf{v}_j)$, e restituisce una
combinazione dei valori. Per una sola query:

$$
s_j = \frac{\mathbf{q}^\top \mathbf{k}_j}{\sqrt{d_k}}, \qquad
a_j = \frac{e^{s_j}}{\sum_{r} e^{s_r}}, \qquad
\mathbf{o} = \sum_j a_j \mathbf{v}_j ,
$$

dove $s_j$ è il punteggio di compatibilità fra la query e la $j$-esima chiave,
$d_k$ è la dimensione di query e chiavi, $a_j$ è il peso che la softmax ricava
dai punteggi, e $\mathbf{o}$ è l'uscita, un vettore nello stesso spazio dei
$\mathbf{v}_j$. La somma corre sulle sole chiavi permesse.

Tre osservazioni tengono in piedi tutto il resto. La prima: quella
compatibilità è appresa, perché $\mathbf{q}$ e $\mathbf{k}_j$ non sono i
vettori grezzi dei token ma proiezioni di quei vettori con matrici che
l'addestramento aggiusta. La seconda: la softmax rende i pesi non negativi e li
normalizza a somma unitaria, quindi $\mathbf{o}$ è una media pesata in senso
proprio e resta nell'inviluppo convesso dei valori. La terza, che è la ragione
per cui l'operazione ha bisogno tanto di $\mathbf{K}$ quanto di $\mathbf{V}$:
la chiave decide il peso, il valore fornisce il contenuto che viene
mescolato. Sono due mestieri distinti affidati a due spazi distinti, e nulla
obbliga i due a coincidere.

Un punto di vocabolario, su cui poggia tutto il resto. Chiamare
$\mathbf{q}$ «la domanda» e $\mathbf{k}_j$ «la risposta» aiuta a ricordare,
ed è un uso corrente; ma il meccanismo è il
prodotto scalare fra due proiezioni apprese, non un dialogo.
`````

## Il tabellone: quali sono le forme in gioco

Fin qui la domanda era una sola. In un Transformer le domande sono tante, una
per posizione, e i conti si fanno tutti insieme impilandole per righe, in
notazione matriciale. Le posizioni
che fanno una domanda sono $L$, quelle che si offrono come chiavi sono $S$, e
i due numeri non sono lo stesso numero. La matrice dei punteggi ha forma
$L \times S$, e il caso quadrato è soltanto il più frequente.

`````{tab} Elementare
Un tabellone appeso al muro, una riga per ogni parola che fa una domanda e una
colonna per ogni parola che si offre come risposta. Nella casella dove la riga
"salta" incrocia la colonna "gatto" c'è un numero solo: quanto quella coppia va
d'accordo. Riempito il tabellone, ogni riga viene guardata per conto suo, e le
sue caselle diventano le intensità dell'evidenziatore per quella parola lì.

Due cose sulle taglie. Domanda ed etichetta vanno confrontate, quindi devono
essere scritte con lo stesso numero di caselle; l'informazione consegnata no,
quella può essere lunga a piacere, perché con nessuno si confronta. E quello
che esce da una riga ha sempre la stessa taglia, che le colonne siano dieci o
diecimila: il tabellone si allarga, il risultato per ogni parola no. È il
motivo per cui la stessa macchina digerisce una frase e un romanzo.

Il tabellone è quadrato quando le due liste sono la stessa, cioè quando una
frase interroga sé stessa. Ma non deve esserlo. Un traduttore ha davanti due
liste diverse: le parole italiane che sta scrivendo fanno le domande, le parole
inglesi che ha letto fanno le risposte, e dodici righe possono incrociare
diciassette colonne senza che niente sia sbagliato. E quando un modello scrive
una parola per volta, di righe ce n'è una sola, lunga quanto tutto quello
che ha letto finora.

Del quadrato conviene diffidare, perché è la scorciatoia che costa di più. Chi
si abitua a immaginarlo quadrato scrive regole che valgono solo per quel caso,
e poi le applica agli altri: la traduzione e la scrittura parola per parola
sono esattamente i due in cui quelle regole falliscono, e falliscono in
silenzio, perché un tabellone rettangolare si riempie lo stesso.
`````

`````{tab} Superiore
Sia $\mathbf{X} \in \mathbb{R}^{n \times d_{\text{model}}}$ la matrice delle
rappresentazioni in ingresso, una riga per token. Dopo le proiezioni (di
$\mathbf{X}$ sola, o nell'attenzione incrociata di due matrici diverse, una per
le query e una per chiavi e valori), e per una sola testa,

$$
\mathbf{Q} \in \mathbb{R}^{L \times d_k}, \qquad
\mathbf{K} \in \mathbb{R}^{S \times d_k}, \qquad
\mathbf{V} \in \mathbb{R}^{S \times d_v},
$$

dove $L$ è il numero di posizioni di query (non il numero di strati, che porta
la stessa lettera nel resto del libro e nella {doc}`sezione sulla struttura del
Transformer <architettura>`: qui $L$ conta righe di una matrice, là piani di
una pila), $S$ il numero di posizioni di chiave e valore, $d_k$ la dimensione
per testa di query e chiavi, $d_v$ quella dei valori. Query e chiavi devono
condividere $d_k$, perché fra loro si fa un prodotto scalare; i valori no, e
$d_v$ può essere diverso. Da qui le forme di
tutto il resto:

$$
\mathbf{Q}\mathbf{K}^\top \in \mathbb{R}^{L \times S}, \qquad
\mathbf{A} \in \mathbb{R}^{L \times S}, \qquad
\mathbf{A}\mathbf{V} \in \mathbb{R}^{L \times d_v},
$$

con $\mathbf{A}$ la matrice dei pesi dopo la softmax. L'uscita ha una riga per
query e $d_v$ colonne: la lunghezza della sequenza di chiavi è sparita nel
prodotto, ed è il motivo per cui uno strato di attenzione accetta contesti di
lunghezza qualsiasi senza cambiare forma in uscita.

$L = S$ vale nella self-attention su una sequenza intera, e le due lunghezze
divergono in due casi tutt'altro che marginali: l'attenzione incrociata, dove
la sequenza di query e quella di memoria sono diverse; e la decodifica con la
cache, dove un blocco corto di query (spesso una riga sola) attende su un
prefisso lungo. La documentazione corrente di PyTorch tiene infatti $L$ e $S$
distinte nella firma di `scaled_dot_product_attention`. Assumere $n \times n$
come forma universale dei pesi di attenzione è un errore che non si manifesta
finché non si esce dal caso simmetrico, e allora si manifesta come un
disallineamento della maschera.
`````

Un batch e più teste aggiungono due dimensioni davanti, e non cambiano niente
del ragionamento: gli stessi conti, ripetuti per ogni esempio e per ogni testa.
Nel resto del capitolo restano sottintese.

## Da dove nascono query, chiavi e valori

Resta da dire *come* si decide l'intensità dell'evidenziatore. Ogni parola,
per partecipare al gioco,
fa tre mestieri diversi, e la rete se ne costruisce tre versioni diverse:
la **query**, la **key** e il **value**. In italiano sarebbero *domanda*,
*etichetta* e *contenuto*, ma i nomi inglesi sono ormai quelli che si trovano
scritti ovunque, e li useremo anche noi. Tutte e tre nascono dalla stessa lista
di partenza, moltiplicata per tre matrici diverse e apprese, e il percorso che
ne segue, dalla proiezione fino alla miscela dei value, è quello che
{numref}`fig-qkv` disegna per una parola sola.

`````{tab} Elementare
Tre versioni della stessa parola, una per mestiere. La prima dice che cosa
quella parola sta cercando nelle altre: "salta" cerca chi compie l'azione. La
seconda è l'etichetta con cui si fa trovare da chi la sta cercando:
"gatto" si presenta come qualcosa di animato, che può compiere azioni. La terza
è l'informazione che consegna a chi l'ha scelta: di "gatto", il fatto che
sia un felino, che sia nero, che nella frase sia il protagonista.

Le tre versioni escono dall'unica lista di partenza passandola attraverso tre
tabelle di numeri: una tabella moltiplica una lista e ne restituisce
un'altra, e siccome i numeri nelle tre tabelle sono diversi (e imparati durante
l'addestramento), le tre liste che ne escono sono diverse fra loro.

Attenzione a non prendere le tre versioni per tre etichette appiccicate addosso
alla parola una volta per tutte. Le tabelle cambiano da un piano all'altro del
modello e da un lettore all'altro, quindi "gatto" cerca una cosa al primo piano
e un'altra al ventesimo, si presenta in un modo a un lettore e in un altro modo
a quello accanto. La prova sta nel guasto che eviti sapendolo: chi si aspetta
un'etichetta fissa si aspetta anche che "gatto" venga scelto sempre dalle stesse
parole, e poi trova due piani in cui succede il contrario, senza che nessuno dei
due sia rotto.

E la separazione dei tre mestieri sembra un lusso, mentre è il punto di tutta
la faccenda. Se ogni parola avesse una sola versione di sé, cercare ed essere
trovati sarebbero la stessa operazione, e una parola potrebbe attirare soltanto
le parole che le somigliano. Con la ricerca e l'etichetta distinte può invece
cercare qualcosa di molto diverso da ciò che offre: "salta" offre un'azione e
cerca un soggetto, cioè esattamente quello che non è.
`````

`````{tab} Superiore
Uno strato di self-attention applica tre proiezioni lineari apprese alla stessa
matrice di ingresso:

$$
\mathbf{Q} = \mathbf{X}\mathbf{W}^Q, \qquad
\mathbf{K} = \mathbf{X}\mathbf{W}^K, \qquad
\mathbf{V} = \mathbf{X}\mathbf{W}^V,
$$

con $\mathbf{W}^Q, \mathbf{W}^K \in
\mathbb{R}^{d_{\text{model}} \times d_k}$ e
$\mathbf{W}^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$. Nell'attenzione
incrociata cambia una cosa sola: le query vengono da un flusso,
$\mathbf{Q} = \mathbf{Y}\mathbf{W}^Q$ con $\mathbf{Y}$ le rappresentazioni di
quell'altra sequenza, e chiavi e valori dall'altro. Tutto il
resto dell'operazione è identico.

$\mathbf{Q}$, $\mathbf{K}$ e $\mathbf{V}$ sono dunque **viste apprese** delle
rappresentazioni correnti, e non ruoli semantici attaccati ai token. Lo
stesso token ha query, chiavi e valori diversi in strati diversi e in teste
diverse dello stesso strato, perché diverse sono le matrici che li producono.
Il modello impara due spazi con cui decidere la compatibilità ($\mathbf{Q}$ e
$\mathbf{K}$) e uno spazio per l'informazione da aggregare ($\mathbf{V}$).

La libertà di avere $\mathbf{W}^Q \neq \mathbf{W}^K$ ha una conseguenza
strutturale che si perde di vista: la matrice dei punteggi
$\mathbf{X}\mathbf{W}^Q(\mathbf{X}\mathbf{W}^K)^\top =
\mathbf{X}\,\mathbf{W}^Q\mathbf{W}^{K\top}\mathbf{X}^\top$ è governata da
$\mathbf{W}^Q\mathbf{W}^{K\top}$, che in generale non è simmetrica. Che
$i$ attenda a $j$ non implica quindi che $j$ attenda a $i$, ed è da questa
asimmetria che viene la capacità di rappresentare relazioni orientate come
«chi è il soggetto di», invece della sola somiglianza. Le varianti che legano
le due proiezioni rinunciano a quella libertà, e quanto costi è una domanda
aperta: il {doc}`confronto coi modelli precedenti <confronti>` riporta l'unica
misura pubblicata, quella del Reformer, che un costo non lo trova.
`````

```{figure} ../figures/attention-is-all-you-need.svg
:name: fig-qkv
:alt: "Un token in ingresso viene proiettato in tre vettori distinti: Query, Key e Value. Il prodotto scalare fra la Query e le Key di tutti i token produce i punteggi di rilevanza, che una softmax trasforma in pesi; i pesi moltiplicano i rispettivi Value e la loro somma è l'uscita per quel token."
:width: 92%

I tre ruoli di ogni parola. La Query è la domanda che pone, la Key l'etichetta
con cui si fa trovare, il Value ciò che offre a chi la seleziona: la stessa
parola li ricopre tutti e tre insieme.
```

## La matrice dei punteggi, elemento per elemento

Presa la riga $i$ di $\mathbf{Q}$ e la riga $j$ di $\mathbf{K}$, l'elemento di
posto $(i, j)$ della matrice dei punteggi è il loro prodotto scalare:

$$
S_{ij} = \mathbf{q}_i^\top \mathbf{k}_j .
$$

Il prodotto fra matrici $\mathbf{Q}\mathbf{K}^\top$ calcola in un colpo solo
tutti i prodotti scalari query-chiave: ogni riga corrisponde a una posizione di
query, ogni colonna a una posizione di chiave.

E il confronto fra una ricerca e un'etichetta, una volta ricordato che sono due
liste di numeri, è la cosa più semplice del mondo: si moltiplicano numero per
numero e si sommano i risultati. Con due listine da tre: $(2, 0, 1)$ contro
$(3, 1, 0)$ fa $2\cdot3 + 0\cdot1 + 1\cdot0 = 6$, mentre contro $(0, 4, 0)$ fa
$0 + 0 + 0 = 0$. Se le due liste hanno numeri grandi negli stessi posti la
somma viene grande, e vuol dire che quell'etichetta risponde a quella ricerca;
se i numeri grandi stanno in posti diversi la somma viene piccola. È
l'operazione che la
{doc}`sezione sull'algebra lineare </Matematica/algebra-lineare>` chiama
*prodotto scalare*, ed è l'unico conto che l'attenzione fa davvero.

Due avvertenze sul contenuto di quelle caselle. Un punteggio grezzo non è una
probabilità: può essere negativo, e la sua grandezza dipende dalla scala e
dalla dimensione dei vettori, cioè da due cose che con la frase non c'entrano
niente. Ed è comodo chiamarlo «somiglianza», ma la parola promette più di
quanto ci sia: l'addestramento sceglie le proiezioni perché il punteggio serva
al compito, e il compito può chiedere che due parole diversissime vadano
d'accordo.

## Perché si divide per la radice di $d_k$

Prima di trasformare i punteggi in intensità, il Transformer li rimpicciolisce
tutti dividendoli per $\sqrt{d_k}$, la radice quadrata della dimensione di
query e chiavi. Il fattore ha una giustificazione precisa, e guardarla da
vicino dice anche che cosa quel fattore non promette.

`````{tab} Elementare
Un punteggio nasce come somma di tanti pezzetti, uno per ogni numero delle due
liste. I pezzetti sono un po’ positivi e un po’ negativi e in buona parte si
compensano fra loro, quindi la somma non cresce in proporzione a quanti sono:
cresce come la loro radice quadrata. Liste quattro volte più lunghe,
punteggi grandi il doppio. Dividere per quella radice riporta i punteggi alla
taglia che avevano con le liste corte, e il conto torna qualunque lunghezza si
scelga.

Il guaio da cui questo difende si vede meglio sapendo che cosa succede a un
evidenziatore quando i punteggi diventano enormi: il più alto si prende tutto
il colore e agli altri resta zero. L'evidenziatore smette di sfumare e diventa
un **interruttore**, acceso su una parola e spento su tutte le altre. E un
interruttore non si corregge. Se qualcuno ti dice che avresti dovuto colorare
"gatto" un pochino meno, con l'evidenziatore sai che cosa fare; con
l'interruttore non c'è nessun «un pochino», e il consiglio non ti serve a
niente. Siccome imparare, per una rete, è esattamente ricevere consigli di
quel genere e seguirli un pochino per volta, un modello saturo smette di
imparare in quel punto.

Due riserve, e sono la parte che si dimentica. La prima: il conto sulla radice
vale finché i numeri delle liste sono presi alla rinfusa, cioè all'inizio,
quando la rete non ha ancora imparato niente. Dopo un po’ di addestramento le
liste smettono di essere alla rinfusa e la radice non descrive più bene la loro
taglia. La seconda, che è la stessa cosa vista dall'altro lato: quella divisione
non è una promessa che l'interruttore non scatti. Con numeri abbastanza grandi
scatta lo stesso, a qualunque lunghezza delle liste; quello che la divisione
toglie è che scatti *per colpa* della lunghezza. E si divide tutto per lo
stesso numero, senza guardare quanto sia grande ciascun punteggio: le
proporzioni fra le caselle di una riga restano quelle di prima, cambia solo
quanto sono distanti.
`````

`````{tab} Superiore
La scala si applica ai punteggi grezzi prima della softmax:

$$
\tilde{\mathbf{S}} = \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} .
$$

L'analisi dimensionale dice subito che il fattore è ammissibile: $S_{ij}$ è uno
scalare, $\sqrt{d_k}$ è un numero puro, e la softmax vuole in ingresso degli
scalari; la divisione cambia la scala dei logit e non il tipo dell'oggetto.
Resta da capire perché *quel* numero.

Supponiamo che le componenti $q_i$ e $k_i$ siano a media nulla, varianza
unitaria, indipendenti fra loro e indipendenti al variare di $i$. Allora ogni
addendo del prodotto scalare ha varianza
$\mathbb{E}[q_i^2 k_i^2] - (\mathbb{E}[q_i]\,\mathbb{E}[k_i])^2 =
\mathbb{E}[q_i^2]\,\mathbb{E}[k_i^2] = 1$, dove la fattorizzazione
dell'aspettazione usa l'indipendenza fra $q_i$ e $k_i$; e siccome gli
addendi sono scorrelati al variare di $i$, le varianze si sommano:

$$
\operatorname{Var}\!\left(\sum_{i=1}^{d_k} q_i k_i\right) = d_k .
$$

Dividere per $\sqrt{d_k}$ la riporta a 1. Servono dunque due indipendenze
diverse, una per ciascun passaggio, e una terza ipotesi che di solito non si
dice: tutto questo vale all'inizializzazione, perché appena
$\mathbf{W}^Q$ e $\mathbf{W}^K$ cominciano ad allenarsi smettono di produrre
componenti a varianza unitaria. Sono ipotesi di illustrazione, non una
descrizione di che cosa faccia un modello addestrato, e l'articolo del 2017 le
presenta così {cite}`vaswani2017attention`.

Che cosa il fattore evita, allora. Con logit di grande modulo la softmax entra
in regime saturo: un peso vicino a 1 e gli altri vicini a 0. Lì lo jacobiano
della softmax, $\partial a_i / \partial s_j = a_i(\delta_{ij} - a_j)$, ha tutti
i termini che tendono a zero, quindi il gradiente che arriva ai punteggi
svanisce, e con esso quello che arriva a $\mathbf{W}^Q$ e $\mathbf{W}^K$. Il
fattore non impedisce la saturazione: con componenti di varianza diversa
da 1 la softmax satura lo stesso, a qualunque dimensione. Ne toglie la
dipendenza da $d_k$, cioè permette di allargare le teste senza che il
regime cambi per
quel solo motivo. E una precisazione che evita una confusione frequente: quel
fattore non normalizza $\mathbf{Q}$ e $\mathbf{K}$ a vettori unitari, che
sarebbe un'altra operazione e cambierebbe i punteggi in modo diverso da riga
a riga.
`````

## Le maschere: quali collegamenti sono permessi

C'è un passaggio in mezzo che finora si è dato per scontato. Prima della
softmax, all'implementazione è permesso sommare alla matrice dei punteggi una
**maschera**, cioè una matrice della stessa forma che vale $0$ dove il
collegamento è lecito e $-\infty$ dove è vietato. Il posto in cui la somma
avviene fa parte della definizione.

`````{tab} Elementare
Il regolamento si scrive sul tabellone prima di cominciare a colorare, e su
certe caselle mette una croce: quella parola lì, per questa domanda, non si può
guardare. Il modo di scriverlo è brutale e funziona benissimo: alla casella
vietata si dà un punteggio di meno infinito, cioè così basso che nessun altro
può scendere sotto. Quando poi si distribuisce il colore, a quella casella ne
tocca esattamente zero.

I divieti che si incontrano si assomigliano poco. C'è quello di guardare
avanti: chi scrive una parola alla volta non può sbirciare le parole che non ha
ancora scritto, e la metà destra del tabellone è tutta crociata. C'è quello di
guardare il riempitivo: per elaborare insieme frasi di lunghezza diversa le si
allunga tutte con parole finte fino alla più lunga, e quelle parole finte non
devono contare niente. E c'è il caso del traduttore, dove il tabellone è già
rettangolare di suo e il divieto riguarda quali parole lette siano ancora
disponibili.

Cancellare a colorazione finita sembra equivalente a mettere le croci prima,
ed è l'errore che si fa più spesso. Se distribuisci l'unità di colore su
tutte le caselle e poi cancelli quelle vietate, quello che resta sulla riga non
somma più a uno: somma a quel che è rimasto. Il miscuglio esce sbiadito in
proporzione a quanto colore hai buttato via, e se le caselle vietate si erano
prese quasi tutto, esce quasi bianco. La riga non se ne lamenta e il conto
prosegue.

E sul primo divieto c'è una finezza che morde proprio dove il tabellone non è
quadrato. «Non guardare avanti» si traduce in «cancella tutto quello che sta
sopra la diagonale» solo se righe e colonne sono in corrispondenza uno a uno.
Con una riga sola e cento colonne la diagonale non vuol dire niente, e bisogna
dire da che parte le due liste sono allineate: quella riga è l'ultima parola, e
le cento colonne sono tutto quello che c'è prima.
`````

`````{tab} Superiore
Sia $\mathbf{M} \in \{0, -\infty\}^{L \times S}$ la maschera additiva. Allora

$$
\mathbf{A} = \operatorname{softmax}\!\left(
\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M} \right),
$$

e poiché $e^{-\infty} = 0$ le posizioni vietate ricevono probabilità nulla,
mentre le altre restano normalizzate fra loro: la riga somma a 1 sulle sole
posizioni permesse. Il decoder del Transformer originale realizza così il
mascheramento autoregressivo, ponendo a $-\infty$ i collegamenti illeciti prima
della softmax {cite}`vaswani2017attention`.

Le maschere non si esauriscono nella causalità. Una maschera di padding
impedisce di attendere ai token di riempimento con cui si allineano le sequenze
di un batch; una maschera strutturata limita la connettività a finestre
locali o a blocchi, ed è la famiglia che la
{doc}`sezione sul confronto coi modelli precedenti <confronti>` percorre.
L'API corrente di PyTorch accetta maschere booleane o additive in virgola
mobile, e tratta a parte la modalità causale.

Perché mascherare dopo sia un errore si vede in una riga di algebra. Azzerare
$a_j$ dopo la softmax lascia $\sum_j a_j = 1 - \sum_{j \in \mathcal{F}} a_j$,
con $\mathcal{F}$ l'insieme vietato: l'uscita $\sum_j a_j \mathbf{v}_j$ è
allora un multiplo arbitrario, e minore di uno, della media pesata voluta.
Rinormalizzare a valle recupera i valori giusti finché i conti restano in
precisione piena, perché la softmax ristretta a un sottoinsieme coincide con la
softmax dell'intero rinormalizzata; ma i logit vietati continuano a entrare nel
massimo e nella somma, e basta che uno sia abbastanza grande perché i termini
permessi vadano in underflow. A quel punto il denominatore è zero e la
rinormalizzazione produce `nan`.

Resta la finezza dell'allineamento, che il caso quadrato nasconde. Con
$L \neq S$ l'espressione «triangolare inferiore» è ambigua finché non si
dichiara quale colonna corrisponde a quale riga, e le convenzioni possibili
sono due. La decodifica con la cache vuole quella allineata a destra:
l'ultima riga di query vede tutte le $S$ colonne, e la riga $i$ vede le prime
$S - L + i$. PyTorch le distingue per nome, `LOWER_RIGHT` e `UPPER_LEFT`, e il
suo `is_causal=True` prende la seconda: con una riga di query e $S$ chiavi
quella riga vede la sola posizione iniziale, cioè il contrario di quello che
serve a generare. Costruire un triangolo $L \times S$, o accendere una comodità
dell'API, senza guardare quale delle due si sta prendendo è uno dei modi più
efficaci di far attendere un modello al proprio futuro.
`````

## La softmax, riga per riga, e la miscela dei valori

Restano gli ultimi due gesti, e sono quelli che trasformano un tabellone di
numeri in una rappresentazione nuova.

`````{tab} Elementare
Ogni riga del tabellone viene guardata da sola, e da sola diventa una
distribuzione di colore. La ricetta che la produce si chiama softmax, ed è
una divisione con un passaggio in più: si prende il numero
$e = 2{,}718\ldots$, lo si eleva a ciascun punteggio della riga, e si divide
ciascun risultato per la somma di tutti quelli della stessa riga. Su tre
punteggi $2$, $1$ e $-1$: $e^2 = 7{,}39$, $e^1 = 2{,}72$, $e^{-1} = 0{,}37$,
che sommati fanno $10{,}48$; le tre intensità sono allora $0{,}71$, $0{,}26$ e
$0{,}04$, che sommano a uno a meno degli arrotondamenti. L'elevamento a potenza
serve a due cose: non far uscire mai numeri negativi (una parola non può
contribuire in negativo) e allargare le differenze, così che un punto di
vantaggio si veda davvero.

Che il conto si faccia per riga e non per colonna cambia il gioco, e si vede
provando a immaginarlo al contrario. Per riga, ogni parola che fa una domanda si
divide una sua unità di colore fra le parole che può guardare, e le domande non
si tolgono niente a vicenda. Per colonna sarebbe l'inverso: ogni parola avrebbe
un'unità di attenzione *ricevuta* da spartire fra chi la cerca, e allora due
domande che vogliono la stessa parola dovrebbero contendersela. È
un'operazione che si può definire, e descrive un'altra cosa.

Il gesto finale è la miscela vera e propria. Le intensità dicono quanto, e
quello che si mescola è l'informazione che ogni parola consegna: prendi il 52%
della lista di "gatto", il 24% di quella di "muro", e somma. Quello che ne esce
è fatto della stessa stoffa delle informazioni consegnate, non dei punteggi:
i punteggi hanno deciso le proporzioni e sono spariti. Da qui un limite
importante: nessun dosaggio può tirare fuori qualcosa che nelle informazioni
non c'era, e si mescola soltanto quello che c'è. È l'unico punto di
tutta la faccenda in cui l'informazione si sposta davvero da una parola
all'altra; tutto il resto serve a decidere quanta.
`````

`````{tab} Superiore
La softmax si applica lungo la dimensione delle chiavi, per ogni riga di query:

$$
A_{ij} = \frac{\exp(\tilde{S}_{ij})}{\sum_{r=1}^{S} \exp(\tilde{S}_{ir})},
\qquad A_{ij} \ge 0, \qquad \sum_{j=1}^{S} A_{ij} = 1 ,
$$

dove $\tilde{S}_{ij}$ è il punteggio scalato e mascherato. Ogni riga di
$\mathbf{A}$ è quindi una distribuzione di probabilità sulle posizioni di
chiave permesse per quella query, e le sue entrate si trovano scritte anche
$\alpha_{ij}$, che è la forma usata in {numref}`fig-qkv` e nella letteratura
a partire da Bahdanau. Una softmax lungo le colonne normalizzerebbe
fra query invece che fra chiavi, definendo un'operazione diversa: le righe non
sommerebbero più a 1 e l'uscita non sarebbe una media pesata.

L'aggregazione è il prodotto per i valori:

$$
\mathbf{o}_i = \sum_{j=1}^{S} A_{ij} \mathbf{v}_j , \qquad
\mathbf{O} = \mathbf{A}\mathbf{V} \in \mathbb{R}^{L \times d_v} .
$$

Questo prodotto è facile da trascurare, e porta con sé una proprietà da
enunciare per esteso: $\mathbf{o}_i$ vive nello spazio dei valori, mai in
quello dei punteggi, ed essendo una combinazione convessa dei
$\mathbf{v}_j$ sta nel loro inviluppo convesso. L'attenzione non può fabbricare
direzioni che i valori non contengono già; quello che può fare è sceglierne il
mescolamento in funzione del contenuto, e cambiarlo a ogni posizione. Dati
i coefficienti, la combinazione dei valori è lineare: la non-linearità viene
da come i coefficienti dipendono dall'ingresso, e dal fatto che li si
moltiplica per valori che dall'ingresso dipendono anche loro.

I pesi $\mathbf{A}$ restano una quantità intermedia. Scambiarli per l'uscita
dello strato è un errore ricorrente, e la
{doc}`sezione sull'attenzione in pratica <attenzione-in-pratica>` lo riprende
insieme agli altri della stessa famiglia.
`````

## L'attenzione con i numeri: tre token a mano

Tutto il meccanismo sta in una manciata di conti, e su una frase di tre parole
li si può rifare a mano. Le liste hanno due soli numeri ciascuna
($d_k = d_v = 2$), e il divieto è quello autoregressivo: ogni parola guarda sé
stessa e quelle prima. Le tre parole sono "il", "gatto", "salta"; le query e le
chiavi sono scelte in modo che "salta" cerchi sull'asse su cui "gatto" si fa
trovare.

```python
import torch

Q = torch.tensor([[0., 1.],      # il
                  [0., 1.],      # gatto
                  [2., 0.]])     # salta: cerca sul primo asse
K = torch.tensor([[1., 0.],      # il
                  [2., 0.],      # gatto: si fa trovare sul primo asse
                  [0., 1.]])     # salta
V = torch.tensor([[1., 0.],      # l'informazione che ciascuna consegna
                  [0., 2.],
                  [1., 1.]])
d_k = Q.shape[1]

punteggi = Q @ K.T                                  # 1) la matrice L x S
scalati = punteggi / d_k**0.5                       # 2) la scala
vietato = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
mascherati = scalati.masked_fill(vietato, float("-inf"))   # 3) la maschera
pesi = torch.softmax(mascherati, dim=-1)            # 4) softmax per riga
uscita = pesi @ V                                   # 5) la miscela

print("punteggi grezzi\n", punteggi)
print("dopo la scala\n", scalati.round(decimals=4))
print("pesi\n", pesi.round(decimals=4))
print("somme di riga", pesi.sum(dim=-1))
print("uscita\n", uscita.round(decimals=4))
```

```text
punteggi grezzi
 tensor([[0., 0., 1.],
        [0., 0., 1.],
        [2., 4., 0.]])
dopo la scala
 tensor([[0.0000, 0.0000, 0.7071],
        [0.0000, 0.0000, 0.7071],
        [1.4142, 2.8284, 0.0000]])
pesi
 tensor([[1.0000, 0.0000, 0.0000],
        [0.5000, 0.5000, 0.0000],
        [0.1867, 0.7679, 0.0454]])
somme di riga tensor([1., 1., 1.])
uscita
 tensor([[1.0000, 0.0000],
        [0.5000, 1.0000],
        [0.2321, 1.5812]])
```

Le tre righe si leggono una per una, e nessuna richiede la macchina. La prima
parola può guardare solo sé stessa: un peso di 1 su una casella sola, e la sua
uscita $(1{,}00,\ 0{,}00)$ è il proprio valore tale e quale. La seconda ha
davanti due caselle con lo stesso punteggio, quindi mezzo e mezzo, e la sua
uscita è la media esatta dei due valori,
$\tfrac{1}{2}(1, 0) + \tfrac{1}{2}(0, 2) = (0{,}50,\ 1{,}00)$. La terza è
l'unica interessante: i punteggi $2$ e $4$ diventano $1{,}414$ e $2{,}828$
dopo la scala, e la softmax li trasforma in $0{,}19$ e $0{,}77$; il poco che
resta, cinque centesimi, va a "salta" su sé stessa. "Salta" mette
tre quarti del suo colore su "gatto", e la sua uscita
$(0{,}23,\ 1{,}58)$ pende dalla parte del valore di "gatto", che era $(0, 2)$.

Due cose che quel tabellone dice e che sono più facili da vedere qui che in
una formula. La prima è dove sono finiti i punteggi: nell'uscita non ce n'è
traccia, hanno deciso le proporzioni e sono usciti di scena. La seconda è che
la terza riga resta una miscela, dove "gatto" pesa molto senza esserne una
copia: le altre due parole ci sono ancora, con il loro pezzetto.

E adesso i due guasti annunciati poco fa, misurati sugli stessi numeri.
Togliendo la scala i punteggi restano $2$ e $4$ invece di $1{,}414$ e
$2{,}828$, e la distribuzione si stringe; spostando la maschera dopo la softmax
la riga smette di sommare a uno.

```python
# senza la scala: la stessa riga, più concentrata
senza_scala = torch.softmax(punteggi.masked_fill(vietato, float("-inf")), -1)
print("terza riga con la scala   ", pesi[2].round(decimals=4))
print("terza riga senza la scala ", senza_scala[2].round(decimals=4))

# la maschera dopo la softmax, su una riga con un punteggio vietato alto
riga = torch.tensor([[1., 2., 60.]])
fuori = torch.tensor([[False, False, True]])
prima = torch.softmax(riga.masked_fill(fuori, float("-inf")), dim=-1)
dopo = torch.softmax(riga, dim=-1).masked_fill(fuori, 0.)
print("maschera prima:", prima.round(decimals=4), "somma", float(prima.sum()))
print("maschera dopo: ", dopo.round(decimals=4), "somma", float(dopo.sum()))
```

```text
terza riga con la scala    tensor([0.1867, 0.7679, 0.0454])
terza riga senza la scala  tensor([0.1173, 0.8668, 0.0159])
maschera prima: tensor([[0.2689, 0.7311, 0.0000]]) somma 1.0
maschera dopo:  tensor([[0., 0., 0.]]) somma 8.850501050313119e-26
```

La riga mascherata dopo la softmax somma a $8{,}9 \cdot 10^{-26}$ invece che a
uno, e l'uscita che ne segue è di fatto azzerata. Rinormalizzarla recupera i
valori giusti, con questi numeri; ma il punteggio vietato continua a entrare nel
conto, e portandolo da $60$ a $200$ i termini permessi finiscono sotto il più
piccolo numero rappresentabile, la somma diventa esattamente zero e la
rinormalizzazione restituisce `nan`. Il divieto scritto prima della softmax non
ha nessuno di questi due problemi.

## Auto-attenzione e attenzione incrociata

La formula non chiede da nessuna parte che $\mathbf{Q}$, $\mathbf{K}$ e
$\mathbf{V}$ vengano dalla stessa sequenza, e da questa libertà nascono i due
usi che si incontrano in ogni Transformer. Nella **self-attention** le tre
proiezioni vengono tutte dallo stesso flusso: ogni parola pesa tutte le altre
della propria frase, e anche sé stessa. Nella **cross-attention** le query
vengono da un flusso e chiavi e valori dall'altro, ed è il traduttore che,
mentre scrive in italiano, torna a rileggersi l'inglese.

| | query da | chiavi e valori da |
|---|---|---|
| self-attention | la sequenza stessa | la sequenza stessa |
| cross-attention | il flusso che scrive | il flusso che è stato letto |
| self-attention causale | la sequenza stessa | la sequenza stessa |

Due parole che il gergo confonde volentieri, e che la tabella tiene separate.
«Self-attention» dice da dove vengono le tre proiezioni; «causale» dice
quali collegamenti sono permessi, cioè è una proprietà della maschera. Una
self-attention può essere bidirezionale (l'encoder) oppure causale (il
decoder), e restano tutte e due self-attention. Chiamare causale ogni
self-attention porta a cercare una maschera dove non c'è, e a non vederla dove
c'è.

## Multi-Head Attention: più letture in parallelo

Una sola passata di evidenziatore costringe la rete a comprimere in un unico
schema tutti i tipi di relazione fra parole. La soluzione del Transformer è
farne parecchie in parallelo.

`````{tab} Elementare
Sulla stessa frase lavorano più lettori, ognuno con un evidenziatore di
colore diverso e una fissazione diversa: uno segna chi fa l'azione, un altro le
parentele di significato ("nero" e "gatto" vanno insieme perché uno è il colore
dell'altro), un altro ancora chi sta vicino a chi nella frase. Ognuno si
costruisce le sue tre versioni di ogni parola, la ricerca, l'etichetta e
l'informazione da consegnare, con tabelle di numeri tutte sue: è da lì che
nasce la differenza fra un lettore e l'altro, perché su tabelle diverse la
stessa frase si evidenzia in modo diverso.

Ogni lettore consegna la sua versione arricchita della parola, e a questo punto
di liste ce ne sono otto invece di una. Come si torna a una sola? Il trucco è
che ogni lettore lavora fin dall'inizio su liste corte, un ottavo di quelle
intere: attaccandole una in coda all'altra si ottiene di nuovo una lista lunga
quanto quella di partenza, perché otto ottavi fanno uno. Resta un ultimo
passaggio, una tabella che la lunghezza non la cambia ma mescola fra loro i
contributi degli otto, così che quello che ciascuno ha visto arrivi in tutte
le caselle e non solo nel proprio ottavo. Alla fine il conto costa quanto un
lettore solo a lista piena.

Ogni lettore si chiama, per ragioni che nessuno ricorda più, una **testa** di
attenzione, e il Transformer originale ne usa otto. Perché otto e non nove?
Perché funzionava: è una scelta provata sul campo, non una legge di natura, e i
modelli che sono venuti dopo usano numeri diversi.

Le fissazioni, poi, vengono fuori dall'addestramento come tutto il resto:
nessuno assegna un compito a un lettore piuttosto che a un altro. Chi è andato
a guardare dentro le teste di un modello già addestrato ne ha trovate alcune
con un mestiere riconoscibile e altre senza niente di preciso, e la divisione
dei compiti si vede solo in parte.
`````

`````{tab} Superiore
La **Multi-Head Attention** esegue $h$ attenzioni indipendenti in sottospazi
distinti e ne ricompone gli esiti:

$$
\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) =
\text{Concat}(\text{head}_1, \ldots, \text{head}_h)\,\mathbf{W}^O
$$

dove $\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_i^Q,
\mathbf{K}\mathbf{W}_i^K, \mathbf{V}\mathbf{W}_i^V)$ e
$\mathbf{W}_i^Q, \mathbf{W}_i^K, \mathbf{W}_i^V, \mathbf{W}^O$ sono matrici
apprese. Nel Transformer originale
$h = 8$ e, con $d_{\text{model}} = 512$, ogni testa lavora in dimensione
$d_k = d_{\text{model}}/h = 64$: tenendo $d_{\text{model}}$ fisso, aumentare
il numero di teste non moltiplica per $h$ il costo di un'attenzione a
dimensione piena, perché ogni testa è più stretta. Il costo complessivo resta
paragonabile a quello di una singola attenzione a dimensione piena, e il
modello può dedicare teste diverse a relazioni diverse (sintattiche,
semantiche, posizionali), cosa che l'analisi empirica delle teste addestrate
conferma almeno in parte.

Su quest'ultimo punto la prudenza è d'obbligo, e riguarda il modo in cui si
leggono le teste di un modello vero. Le semantiche delle teste sono
emergenti: non c'è nessun vincolo che spinga una testa a corrispondere a un
concetto pulito e nominabile, e trovarne alcune che lo fanno non autorizza a
cercare un'etichetta per tutte.
`````

## L'attenzione, da sola, non ha ordine

Un fatto che a prima lettura sorprende: nella formula che abbiamo montato non
compare mai la posizione dei token. Ogni riga di $\mathbf{Q}$ viene confrontata
con ogni riga di $\mathbf{K}$ senza che nulla dica quale venga prima. La
conseguenza è precisa, e si dimostra in una riga: rimescolare le righe di
$\mathbf{X}$ con una permutazione $\mathbf{P}$ rimescola allo stesso modo
quelle di $\mathbf{Q}$, $\mathbf{K}$ e $\mathbf{V}$, quindi i punteggi
diventano $\mathbf{P}\tilde{\mathbf{S}}\mathbf{P}^\top$; la softmax lavora riga
per riga e il rimescolamento la attraversa intatto; e il prodotto finale lo
riporta fuori tale e quale, perché
$(\mathbf{P}\mathbf{A}\mathbf{P}^\top)(\mathbf{P}\mathbf{V}) =
\mathbf{P}\mathbf{A}\mathbf{V}$. Le righe dell'uscita si permutano come quelle
dell'ingresso e nient'altro cambia: la self-attention senza maschera è
equivariante rispetto alle permutazioni. Per l'attenzione, «Il gatto morde
il cane» e «Il cane morde il gatto» sono lo stesso insieme di parole.

L'ordine va quindi reintrodotto da fuori, e il Transformer del 2017 lo fa
sommando alle rappresentazioni un segnale posizionale. Chi lo produce, come
lo si è calcolato nel 2017 e come lo si calcola oggi (con la rotazione di query
e chiavi che va sotto il nome di RoPE) è materia della
{doc}`sezione sulla struttura del Transformer <architettura>`, che monta il
blocco per intero.

Due precisazioni, perché sono i due modi in cui questo punto si fraintende. La
codifica posizionale non fa parte dell'attenzione: modifica le rappresentazioni,
o l'interazione fra query e chiavi, in modo che l'attenzione possa usare la
posizione. E la maschera causale, che pure introduce una direzione, non basta:
dice che cosa si può guardare, non quanto sia distante.

## Dove va a finire l'attenzione: encoder e decoder

L'attenzione, da sola, è un pezzo, e va montato. Il
pezzo si chiama **blocco**, e un blocco è quello che si ripete sempre uguale a
sé stesso lungo la macchina, come un piano di un palazzo. L'encoder, la
torre che legge, è una pila di questi blocchi; il decoder, quella che
scrive, è un'altra pila fatta allo stesso modo, che produce l'uscita (una
traduzione, una risposta) un pezzo alla volta. In mezzo, ancora attenzione:
mentre genera, il decoder pesa le parti rilevanti di ciò che l'encoder
ha letto, ed è la cross-attention di poco fa.

Una pila di blocchi è quella che si chiama una rete profonda, ed è profonda
proprio in questo senso: tanti passaggi uno sopra l'altro, decine o centinaia.
Impilarli, però, non è gratis, e ogni blocco porta con sé due accorgimenti che
servono soltanto a rendere la pila addestrabile.

`````{tab} Elementare
A ogni piano del palazzo la lista di numeri entra nelle stanze, passa per i
conti e ne esce cambiata. Accanto alle stanze corre una scala dritta, con un
corrimano che va da cima a fondo: lungo quella scala la stessa lista sale
intatta, senza entrare da nessuna parte, e in cima al piano si somma numero per
numero a quella uscita dalle stanze. La strada di lato è la **scorciatoia**.

Chi sale la usa per arrivare in alto senza sfilacciarsi per via. Serve però
soprattutto a chi scende. Quando la rete scopre di aver sbagliato, dall'ultimo
piano parte un messaggio che dice di quanto e in che direzione ritoccare i
conti, e quel messaggio deve arrivare fino ai primi piani. Se passa per le
stanze, a ogni piano viene moltiplicato per i numeri di quel piano, che di
solito sono un po’ minori di uno. Nove decimi a ogni piano: dopo cinquanta piani
ne resta lo $0{,}5\%$, cioè quasi niente, e i piani bassi smettono di imparare.
Sulla scala il messaggio scende senza toccare i conti, e in fondo arriva ancora
leggibile.

Su ogni pianerottolo c'è una bilancia, ed è il secondo accorgimento: la
**taratura**. Piano dopo piano i numeri scappano via, qui tutti enormi, là tutti
minuscoli, e una rete con addosso valori fuori misura non impara più. Allora la
lista viene rimessa in riga: si sottrae a tutti la loro media, così il centro
cade sullo zero, e poi si dividono tutti per quanto sono sparpagliati, così la
larghezza è sempre quella. La bilancia non cambia che cosa si sta pesando: mette
solo il numero letto sulla stessa scala di tutti gli altri. Accanto c'è una
manopola, imparata durante l'addestramento, con cui la rete riallarga o
restringe la scala dove le conviene.

Nel palazzo del 2017 la bilancia sta sul pianerottolo della scala, subito dopo
il punto in cui le due liste si sommano: chi scende deve attraversarla a ogni
piano, e il corrimano non è sgombro fino in fondo. Lo si vedeva
dall'addestramento, che partiva storto: bisognava cominciare con ritocchi
piccolissimi e allargarli pian piano, altrimenti la pila andava fuori giri alle
prime correzioni. I modelli venuti dopo hanno portato la bilancia all'ingresso
delle stanze, la scala è tornata libera da cima a fondo, e quella partenza in
punta di piedi si è potuta togliere.

La pesata, intanto, si è fatta più spiccia. Nei modelli linguistici di oggi la
media non la si toglie nemmeno: si divide e basta per la grandezza tipica dei
numeri della lista, e poi si gira la manopola. Un conto in meno a ogni piano, e
la pila sta ferma lo stesso.
`````

`````{tab} Superiore
Sono le **residual connection** e la **layer normalization**, combinate in

$$
\text{LayerNorm}\big(\mathbf{x} + \text{SubLayer}(\mathbf{x})\big)
$$

attorno a ogni sotto-strato (attenzione o feed-forward). La connessione
residuale (la stessa idea delle ResNet che abbiamo visto fra le
{doc}`architetture storiche del deep learning
</DeepLearning/architetture-storiche>`) offre al gradiente un cammino quasi
diretto verso gli strati
iniziali, contrastando il gradiente che svanisce; la layer normalization
stabilizza media e varianza delle attivazioni a ogni posizione, rendendo
l'addestramento meno sensibile a learning rate e inizializzazione. «Quasi»,
perché in questa formulazione (detta *Post-LN*, quella del 2017) la
normalizzazione sta proprio sul ramo della scorciatoia, e il gradiente la
attraversa a ogni strato: i modelli successivi la spostano prima del
sotto-strato, $\mathbf{x} + \text{SubLayer}(\text{LayerNorm}(\mathbf{x}))$, il
cosiddetto
*Pre-LN*, ed è lì che il cammino identità diventa davvero pulito (Xiong e
colleghi {cite}`xiong2020layer` mostrano che senza questo spostamento serve un
riscaldamento graduale del learning rate per addestrare stabilmente).

Nei modelli linguistici recenti anche la normalizzazione stessa si è
alleggerita: al posto della LayerNorm c'è quasi sempre la **RMSNorm**
{cite}`zhang2019root`, che non sottrae la media e non ha bias. Divide il
vettore per la sua radice quadratica media e lo riscala con un guadagno
appreso,
$\mathbf{x} \mapsto \boldsymbol{\gamma} \odot \mathbf{x}/\mathrm{RMS}(\mathbf{x})$
con $\mathrm{RMS}(\mathbf{x}) = \sqrt{\tfrac{1}{d}\sum_i x_i^2}$: meno conti
per strato, e in pratica la stessa stabilità.
`````

Scorciatoia e taratura sono la parte che nessuno racconta mai, e senza la quale
niente di tutto il resto starebbe in piedi: l'attenzione è l'idea, ma un'idea
impilata sessanta volte (tanti sono i blocchi di un modello grande di oggi) si
sfalda, e questi due accorgimenti sono ciò che la tiene insieme. Con il
meccanismo in mano, la {doc}`sezione sulla struttura del Transformer
<architettura>` prende questi pezzi e li monta nelle due torri di una macchina
vera.

```{admonition} Un cantiere parallelo: le reti a memoria
:class: note
Interrogare un archivio con una domanda, pesare quanto ciascun elemento le
risponde, e restituire la miscela pesata di ciò che quegli elementi contengono:
questa struttura è stata costruita prima dei Transformer, e per un altro scopo.

Nel 2014 le **memory network** {cite}`weston2015memory` affrontavano il
problema di far ragionare una rete su un elenco di fatti. L'esempio degli
autori è costruito apposta perché una frase sola non basti: «Giovanni è andato
in ufficio. Giovanni ha posato il latte. Giovanni è andato in bagno. Dov'è il
latte?», dove per rispondere «in ufficio» servono i primi due e nessuno dei due
basta. La rete teneva i fatti in un archivio a parte, separato dai numeri che
aveva imparato, e ci pescava dentro due volte di fila, la seconda con in mano
il fatto trovato per primo: erano gli *hop*, i salti di ragionamento. In quella
prima versione però la pesca era secca (si sceglieva *un* fatto, il più
somigliante) e per addestrarla bisognava dire alla rete, esempio per esempio,
quali fossero i fatti giusti da usare.

Il passo che ci interessa arriva l'anno dopo, con le *end-to-end memory
network* {cite}`sukhbaatar2015end`: al posto della scelta secca si mette una
graduatoria, cioè la domanda viene confrontata con tutti i fatti, il
confronto produce un'intensità di evidenziatore per ciascuno, e l'archivio
viene letto mescolando i fatti in quelle proporzioni. I salti restano; a
cambiare è che adesso ogni fatto contribuisce un po’, quindi la correzione
degli errori attraversa anche la pesca, e la rete impara da sola quali fatti
contano senza che glielo si dica.

Due cose da portarsi via. La prima è che quella graduatoria sui fatti *è*
l'attenzione, con la sola differenza che qui l'archivio è un magazzino a parte
invece della frase stessa. La seconda è che la struttura
domanda-contro-archivio, con i fatti tenuti fuori dalla rete e consultati al
momento, è esattamente la forma dei sistemi che cercano documenti prima di
rispondere: si chiamano RAG, e li costruisce per intero la
{doc}`sezione sul retrieval <rag>`.

Sulle date conviene però essere precisi. L'attenzione per la traduzione è del
settembre 2014 e le memory network dell'ottobre dello stesso anno: due strade
partite quasi insieme, da due problemi diversi, e arrivate alla stessa
operazione, tanto che la seconda cita la prima fra i lavori affini e non fra le
proprie basi. La versione a graduatoria è invece del marzo 2015, e lì le due
strade si incrociano per davvero: gli autori presentano il proprio modello come
un'estensione di quello di Bahdanau, con più passi di lettura per ogni parola
prodotta. Quello che le reti a memoria hanno di proprio è dunque
l’archivio tenuto fuori dai numeri imparati e consultato al momento della
domanda, più che l'attenzione: ed è quel pezzo lì, messo da parte perché la
sua epoca
non aveva né i dati né l'hardware, a tornare cinque anni dopo con un altro nome.
```

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- L’attenzione rilegge la frase con un evidenziatore: per capire una
  parola, guarda tutte le altre, dà a ciascuna un'intensità di colore e ne
  mescola le informazioni in quella proporzione. L'evidenziatore decide quanto;
  quello che si mescola è l'informazione, e sono due cose tenute separate.
- Le intensità le decide la frase, non un programmatore: quello che la rete
  impara, provando e correggendosi su miliardi di esempi, è il criterio con cui
  il colore va assegnato, e le intensità le ricalcola su ogni frase nuova.
  Quando è ogni parola a guardare tutte le altre, si chiama self-attention;
  quando a guardare è un testo che si sta scrivendo verso un testo che è stato
  letto, si chiama cross-attention.
- Per giocare, ogni parola si presenta in tre versioni: la query (la
  domanda che fa), la key (l'etichetta con cui si fa trovare) e il
  value (l'informazione che consegna). Le tre versioni cambiano da un piano
  all'altro del modello: non sono etichette appiccicate addosso una volta per
  tutte.
- I conti si tengono su un tabellone, una riga per chi chiede e una colonna per
  chi risponde, e il tabellone non è per forza quadrato: nella traduzione, e
  quando il modello scrive una parola alla volta, le due liste hanno lunghezze
  diverse.
- I divieti (non guardare avanti, non guardare il riempitivo) si scrivono
  prima di distribuire il colore. Colorare e poi cancellare lascia una riga
  che non somma più a uno, e un miscuglio sbiadito.
- I punteggi si rimpiccioliscono prima di diventare colore, dividendoli per la
  radice della lunghezza delle liste. Senza, con liste lunghe l'evidenziatore
  diventa un interruttore, e un interruttore non si corregge un pochino per
  volta, cioè non impara più. La divisione toglie però una causa sola: con
  numeri abbastanza grandi l'interruttore scatta comunque, a qualunque
  lunghezza delle liste.
- Di evidenziatori se ne passano parecchi in parallelo, ognuno attento a un
  tipo di legame diverso: sono le teste di attenzione, e nel Transformer del
  2017 erano otto.
- L'attenzione da sola non sa che cosa viene prima e che cosa dopo: l'ordine
  glielo si aggiunge da fuori.
- Attorno a ogni blocco ci sono una scorciatoia (l'informazione passa anche
  di lato, intatta, e la correzione degli errori trova una presa per tornare
  indietro) e una taratura (i numeri riportati su una scala standard).
  Senza di loro le torri alte non si addestrano; e conta dove la taratura si
  infila, perché nel montaggio del 2017 stava sul percorso della scorciatoia, e
  i modelli venuti dopo l'hanno spostata all'ingresso del blocco.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- L’attenzione costruisce, per ogni posizione di query, una
  rappresentazione contestuale:
  $\operatorname{softmax}(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k} +
  \mathbf{M})\,\mathbf{V}$, cioè media dei valori pesata dalle affinità
  query-chiave. La chiave decide il peso, il valore fornisce il contenuto, e
  l'uscita di ogni testa resta nell'inviluppo convesso dei suoi valori.
- Le forme: $\mathbf{Q} \in \mathbb{R}^{L \times d_k}$,
  $\mathbf{K} \in \mathbb{R}^{S \times d_k}$,
  $\mathbf{V} \in \mathbb{R}^{S \times d_v}$, punteggi e pesi
  $L \times S$, uscita $L \times d_v$. $L = S$ vale nella self-attention su una
  sequenza intera e non oltre: assumere una matrice quadrata rompe l'attenzione
  incrociata e la decodifica con la cache.
- $\mathbf{Q}$, $\mathbf{K}$ e $\mathbf{V}$ sono viste apprese
  ($\mathbf{X}\mathbf{W}^Q$ e le altre due), diverse per strato e per testa, non
  ruoli semantici fissi. Poiché $\mathbf{W}^Q\mathbf{W}^{K\top}$ in generale non
  è simmetrica, l'attenzione può rappresentare relazioni orientate.
- Il fattore $1/\sqrt{d_k}$ neutralizza la dipendenza da $d_k$ della varianza
  dei punteggi, sotto l'ipotesi di componenti indipendenti a media nulla e
  varianza unitaria, che vale all'inizializzazione. Non impedisce la
  saturazione della softmax: ne toglie una causa.
- La maschera si somma ai punteggi prima della softmax, con $-\infty$ sulle
  posizioni vietate. Azzerare i pesi dopo lascia righe che non sommano a 1; e
  con $L \neq S$ la causalità richiede una convenzione di allineamento
  esplicita.
- La softmax normalizza lungo le chiavi, per riga: ogni query produce la
  propria distribuzione. I pesi $\mathbf{A}$ sono un intermedio, non l'uscita
  dello strato.
- La Multi-Head Attention esegue più attenzioni in sottospazi distinti
  ($h = 8$ nel modello originale) e ricompone con $\mathbf{W}^O$; le semantiche
  delle teste sono emergenti e non garantite.
- La self-attention senza maschera è equivariante alle permutazioni: la
  posizione va aggiunta da fuori, e la maschera causale dà una direzione ma non
  una distanza.
- Residual connection e layer normalization tengono addestrabili le
  pile profonde di blocchi. L'articolo del 2017 le combina come
  $\text{LayerNorm}(\mathbf{x} + \text{SubLayer}(\mathbf{x}))$ (*Post-LN*); i
  modelli successivi normalizzano prima del sotto-strato,
  $\mathbf{x} + \text{SubLayer}(\text{LayerNorm}(\mathbf{x}))$ (*Pre-LN*), ed
  è così che la
  scorciatoia resta davvero libera.
```
`````

Il meccanismo, adesso, è tutto qui: cinque passaggi che si rifanno a mano su
tre parole, e una manciata di scelte che spiegano perché la formula ha proprio
quella forma. Quello che manca è la macchina che gli sta attorno, e le due
sezioni che seguono la costruiscono da due lati diversi: prima le torri in cui
il blocco si impila, poi il confronto con le reti che leggevano in fila.
