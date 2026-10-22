# Confronto con i modelli precedenti

Ogni nuova architettura va giudicata rispetto a ciò che sostituisce; prima del
2017 il testo era trattato dalle reti ricorrenti del {doc}`capitolo sul Natural
Language Processing </NaturalLanguageProcessing/overview>`: la RNN
(*recurrent neural network*) e le due varianti con porte che l'hanno
soppiantata, LSTM e GRU. Il confronto serve a capire *perché* il Transformer
ha vinto, e anche *dove* non vince affatto.

## Le reti ricorrenti: tre varianti, un limite

Le tre varianti hanno un limite in comune: l'informazione sul passato attraversa
uno stato che si aggiorna un passo alla volta. Il limite si vede già nella RNN
più semplice.

```{figure} ../figures/rnn-reti-con-memoria.svg
:name: fig-rnn-srotolamento
:alt: "A sinistra una cella ricorrente con una freccia che rientra su sé stessa, il cappio dello stato. A destra la stessa cella srotolata nel tempo in tre copie identiche, ciascuna che riceve una parola e passa lo stato alla successiva: sono la stessa cella, con gli stessi pesi, applicata a istanti diversi."
:width: 96%

Il cappio e il suo srotolamento. Le tre copie a destra sono la stessa rete,
riusata a ogni passo, e per questo una RNN funziona su
sequenze di lunghezza qualsiasi.
```

L'equivalenza di {numref}`fig-rnn-srotolamento` è anche la radice del problema.
Lo stato, il «riassunto» che la rete si porta dietro, è un vettore
$\mathbf{h}_t = f(\mathbf{h}_{t-1}, \mathbf{x}_t)$, aggiornato a ogni parola
dalla stessa funzione con gli stessi parametri. Il contributo di una parola
lontana, e la correzione che durante l'addestramento torna verso di lei,
attraversano quindi una catena di passaggi simili, uno per parola, e a ogni
passaggio si moltiplicano per un fattore simile.

Ripetere una moltiplicazione svuota presto un numero. Ad
esempio, con un fattore di $0{,}9$ per passo, dopo cento parole ne resta
$0{,}9^{100} \approx 0{,}000027$, meno di un trentamillesimo (la
{doc}`sezione sui modelli di sequenza
</NaturalLanguageProcessing/modelli-sequenza>` ha svolto il calcolo per
esteso).

Anche $0{,}99$, che è quasi non perdere niente, dopo cento parole è sceso a
$0{,}37$. In generale, qualunque fattore minore di uno finisce nello stesso
posto, e per questo l'inizio di un testo lungo si attenua; un fattore maggiore
di uno fa il contrario, e il segnale esplode.

`````{tab} Elementare
Le RNN leggono una parola alla volta portandosi dietro un riassunto
mentale, che però sbiadisce in fretta: alla fine di un paragrafo lungo,
l'inizio è quasi svanito. Le LSTM aggiungono un taccuino con delle regole:
cosa annotare, cosa cancellare, cosa rileggere (la memoria dura molto di più,
con un meccanismo più complicato). Le GRU sono il taccuino
semplificato: regole più snelle, quasi la stessa resa. Taccuino o no, però, si
legge sempre una parola alla volta, e la parola dopo aspetta che sia finita
quella prima. Il Transformer cambia tutto: niente riassunto da tenere
aggiornato, il testo resta tutto sott'occhio e ogni parola può andare a
rileggersi qualunque altra. La memoria non sbiadisce perché non c'è nulla da
ricordare: basta guardare.

Due precisazioni, perché il «qualunque altra» va preso con le pinze. La prima:
vale per la torre che legge (l'encoder); quella che scrive (il decoder)
ha una regola ferrea, non si sbircia avanti, e guarda solo all'indietro.

La seconda riguarda la velocità. Il ricordo che non sbiadisce, da solo, non
spiega la vittoria, perché il taccuino delle LSTM allungava già la memoria, ma
nessuna macchina di prima sapeva mettere tante mani sullo stesso testo. Se le
parole si guardano tutte insieme invece che una dopo l'altra, il lavoro si può
spartire fra migliaia di processori che macinano in parallelo: sono i «cento
amici», quelli che con un libro da leggere pagina
dopo pagina non servivano a niente e qui invece servono eccome.

Questo però succede solo mentre il modello studia, cioè quando ha davanti
tutto il testo e può lavorarci in una volta sola. Quando poi scrive, le parole
gli escono comunque una alla volta, perché per scegliere la prossima deve
sapere quale ha appena scritto: lì i cento amici tornano a girarsi i pollici, e
infatti generare resta lento.
`````

`````{tab} Superiore
Le RNN mantengono uno stato
$\mathbf{h}_t = f(\mathbf{h}_{t-1}, \mathbf{x}_t)$: la dipendenza tra
posizioni distanti $m$ passi attraversa $m$ applicazioni di $f$, e il
gradiente retropropagato è un prodotto di $m$ jacobiane
$\partial\mathbf{h}_s/\partial\mathbf{h}_{s-1}$. Se il massimo valore singolare
della matrice ricorrente è inferiore a $1/\gamma$, con $\gamma$ il massimo della
derivata della non linearità ($1$ per la tanh, $1/4$ per la sigmoide), la
norma del prodotto decade in modo esponenziale; perché esploda è necessario che
lo superi {cite}`pascanu2013difficulty`. La derivazione sta nella
{doc}`sezione sui modelli di sequenza
</NaturalLanguageProcessing/modelli-sequenza>`, che costruisce anche le LSTM
{cite}`hochreiter1997long` e le GRU {cite}`cho2014learning`. Entrambe, grazie
ai gate, aprono un cammino quasi lineare per il gradiente e allungano
l'orizzonte della memoria, ma restano sequenziali: il passo $t$ attende il
passo $t-1$, sia in addestramento sia in inferenza.

La Tabella 1 dell'articolo del 2017 mette a confronto quattro tipi di strato,
con $n$ la lunghezza della sequenza, $d$ la dimensione delle rappresentazioni,
$k$ l'ampiezza del filtro convoluzionale e $r$ quella del vicinato
nell'attenzione ristretta {cite}`vaswani2017attention`:

| tipo di strato | costo per strato | operazioni in sequenza | cammino massimo |
|---|---|---|---|
| self-attention | $O(n^2 d)$ | $O(1)$ | $O(1)$ |
| ricorrente | $O(n d^2)$ | $O(n)$ | $O(n)$ |
| convoluzionale | $O(k n d^2)$ | $O(1)$ | $O(\log_k n)$ |
| self-attention ristretta | $O(r n d)$ | $O(1)$ | $O(n/r)$ |

Si noti che il costo della self-attention è quello della sola operazione,
senza le proiezioni, e che il cammino $O(\log_k n)$ della convoluzione vale per
i filtri dilatati (con filtri contigui servono $O(n/k)$ strati).

Il Transformer porta la lunghezza del cammino tra due posizioni qualsiasi a
$O(1)$ (ogni coppia è collegata direttamente dalla self-attention) e rende
l'addestramento parallelo sull'intera sequenza. È questa combinazione
(dipendenze lunghe *e* parallelismo) che le architetture ricorrenti non
potevano offrire insieme. Il parallelismo, però, è un vantaggio soprattutto
in addestramento, perché in inferenza la generazione autoregressiva resta
sequenziale, un token alla volta.
`````

## Il costo quadratico: l’attenzione cresce col quadrato

Il Transformer ha un prezzo: il suo punto debole sta nell'operazione
che lo definisce, far guardare ogni parola a tutte le altre.

`````{tab} Elementare
Il conto è lo stesso dei cento sguardi, con delle persone al posto delle
parole. Quattro persone in una stanza, e ognuno deve parlare con
ognuno: io con te, io con lui, io con lei, tu con lui, tu con lei, lui con lei.
Sei coppie, contate
con le dita. Senza dita: $4 \times 3 = 12$, ognuno con tutti tranne sé, e
$12 : 2 = 6$ perché ogni coppia è stata contata due volte. In otto,
$8 \times 7 : 2 = 28$; in mille, $1000 \times 999 : 2 = 499\,500$, quasi mezzo
milione. Raddoppiando i presenti le chiacchiere quadruplicano, o quasi. Questa è
la crescita al quadrato.

Per un Transformer i presenti sono le parole, con due usanze: si parla anche da
soli, e ascoltare non conta come farsi ascoltare (che "salta" guardi "gatto" è
una cosa, il rovescio un'altra). In quattro i confronti diventano
$4 \times 4 = 16$, con la stessa crescita di prima. Una frase è una riunione
svelta, dove pesa di più il lavoro che ognuno fa da solo, prima e dopo aver
parlato; un libro è un'assemblea oceanica che nessun computer affronta
volentieri.

Le reti ricorrenti non hanno questo problema: un presente in più è un turno in
più. Un regolamento scritto in anticipo, con quanto ogni
posto deve ascoltare ogni altro, varrebbe per un solo numero di presenti e
sarebbe lunghissimo; l'attenzione calcola le coppie al momento, con le stesse
poche regole per qualunque riunione.

Poi c'è il tabellone, una casella per ogni scambio: sedici in quattro, un
milione in mille. Il tempo alla peggio lo si aspetta; il tabellone, se lo si
appende tutto intero, o sta nella stanza o non ci sta, e a lungo ha deciso lui
quanto testo un modello si tiene davanti. Lo si può anche riempire un
pezzo alla volta, usarlo e cancellarlo, senza appenderlo mai intero: le
chiacchiere restano tutte, ma la stanza basta. Gli altri rimedi tagliano invece
le chiacchiere, e fanno parlare tutti senza convocare la plenaria.

Il primo fissa il programma prima di entrare: ognuno con i vicini di posto
(poniamo i tre a destra e i tre a sinistra), qualche coppia sorteggiata per
accorciare le distanze, e due o tre persone che parlano con tutti e fanno da
ponte. In mille, invece di mezzo milione di chiacchiere ne serve qualche
migliaio. Si arriva ancora dove arrivava la plenaria, e lo si dimostra; la
dimostrazione dipende dai ponti, non dai vicini.

La riunione, però, si tiene a giri, uno per piano del modello, e in un giro i
pochi ponti non ripetono tutto a tutti: c'è un compito che la plenaria sbriga
in un giro solo, mentre il programma fisso richiede tanti più giri quanti più
sono i presenti (se vale un'ipotesi che nessuno ha ancora dimostrato). Si
risparmia sulle chiacchiere di ogni giro e si paga in numero di giri.

Il secondo lascia decidere alla sala. Ognuno ha da dire qualcosa a pochissimi,
e le altre conversazioni si tengono lo stesso a vuoto. All'ingresso, allora, i
presenti vanno a tavoli per affinità e parlano con chi si ritrovano accanto. Il
tavolo giusto si trova solo se chi cerca e chi va trovato portano lo stesso
cartellino, e questa è una rinuncia, perché in plenaria cercare e farsi trovare
erano due ruoli distinti; chi l'ha sperimentato, su due compiti, non ha visto
la riunione riuscire peggio.

I tavoli, però, sbagliano: due che avevano da dirsi qualcosa finiscono
separati. Si rifanno i tavoli più volte con criteri diversi, così l'occasione
persa in una disposizione si recupera nella successiva, ma nessuno promette che
non ne resti fuori una.

Gli stessi organizzatori hanno anche un accorgimento che non riguarda chi parla
con chi. Di ogni giro si tiene un verbale, perché a riunione finita bisogna
ripercorrerla all'indietro per capire che cosa correggere, e con più giri
ci sono più verbali da conservare. Se però ogni giro si può ricostruire
partendo da quello dopo, i verbali si buttano e si riscrivono quando servono,
rifacendo i conti: spazio risparmiato, fatica in più, un baratto che nel deep
learning torna di continuo.

Il terzo butta la lista degli invitati invece di sfoltirla: le operazioni si
riordinano perché le coppie non si formino mai una per una, invece di
calcolarle tutte e scartarne poi quasi tutte. Costa una rinuncia (il modo in
cui i punteggi diventano intensità va cambiato) e ha un capitolo suo, quello
sull’{doc}`attenzione lineare </AttenzioneLineare/overview>`.
`````

`````{tab} Superiore
La matrice di attenzione ha $n \times n$ elementi. Contando la sola
operazione di attenzione (proiezioni escluse), il costo in tempo è
$O(n^2 \cdot d)$ nella lunghezza $n$ della sequenza, con $d = d_{\text{model}}$,
contro l’$O(n \cdot d^2)$ di uno strato ricorrente; la memoria per i punteggi è
$O(n^2)$, contro l’$O(n \cdot d)$ delle attivazioni ricorrenti. Dal confronto
fra i due termini l'articolo ricava che l'operazione di attenzione costa meno
dello strato ricorrente finché $n < d$, il caso tipico della traduzione del
2017; le quattro proiezioni dello strato, che la sua tabella non conta, costano
però $4nd^2$, quanto lo strato ricorrente: lo strato intero non costa quindi
meno, ma si parallelizza meglio. Dentro il Transformer, poi, il termine
quadratico va messo accanto a quello delle proiezioni e della FFN: per strato
il totale è $12nd^2 + 2n^2d$, e il secondo supera il primo solo per $n > 6d$,
come ricava la {doc}`matematica di un modello linguistico
</Matematica/matematica-llm>`.

Un termine di paragone assente nella tabella è uno strato denso sull'intera
sequenza, che mappi gli $nd$ numeri d'ingresso negli $nd$ d'uscita: avrebbe
$O(n^2 d^2)$ parametri e una lunghezza fissata una volta per tutte.
L'attenzione ne ha $O(d^2)$, condivisi fra le posizioni, e accetta qualunque
$n$; in cambio l'interazione fra le posizioni non è parametrizzata ma
calcolata, con $O(n^2 d)$ operazioni.

Sotto il vincolo della memoria, più ancora che del tempo, sono nate le
finestre di contesto limitate dei grandi modelli, e una vasta letteratura di
rimedi: attenzione sparsa o a finestre
locali (Longformer, BigBird), approssimazioni a rango basso o kernel
(Linformer, Performer {cite}`choromanski2021performer`), e ottimizzazioni
esatte come FlashAttention {cite}`dao2022flashattention`, che lascia invariato
il costo $O(n^2 d)$ ma non scrive mai la matrice $n \times n$ in memoria
globale e porta lo spazio aggiuntivo da $O(n^2)$ a $O(n)$, come spiegato nella
{doc}`sezione sulla FlashAttention </GPU/flash-attention>`.

L'attenzione sparsa nasce da un cambio di punto di vista, più che da un trucco
di calcolo. La matrice di attenzione è la matrice di adiacenza di un
**grafo completo** sui token, e ridurne il costo è un problema
di sparsificazione di grafi. Longformer {cite}`beltagy2020longformer`
toglie archi tenendo una finestra scorrevole attorno a ogni token, qualche
finestra dilatata per allargare la portata, e alcuni **token globali**
collegati a tutti (nel question answering, quelli della domanda): il costo
scende da $O(n^2)$ a $O(n w)$, dove $w$ è l'ampiezza della finestra, un numero
fissato in anticipo e molto minore di $n$. BigBird {cite}`zaheer2020big`
prende la stessa
strada, formalizzandola: combina una finestra ad anello (un grafo «piccolo
mondo» alla Watts-Strogatz), archi casuali alla Erdős-Rényi e token
globali, e dimostra che il modello risultante resta un approssimatore
universale di funzioni su sequenze.

Il teorema richiede tuttavia ipotesi precise: vale per le funzioni continue su
un dominio limitato e vale per qualunque schema sparso che contenga i token
globali (sono questi ultimi a garantire la validità del risultato, non la
finestra). Nello stesso lavoro gli autori mostrano il limite di
questo schema: ammettendo la congettura dei vettori ortogonali, non ancora
dimostrata, esiste un compito che l'attenzione piena risolve in un numero
costante di strati, mentre qualunque attenzione sparsa con un numero di archi
proporzionale a $n$ richiede una profondità crescente con $n$. L'universalità
garantisce che la funzione si possa approssimare, non che ci si arrivi alla
stessa profondità: la sparsificazione baratta ampiezza con altezza.

La {doc}`sezione sui Graph Transformer
</GraphNeuralNetwork/architetture-applicazioni>` riprende questa lettura
dall'altro capo, dove la fase in cui ogni nodo raccoglie i vettori dei suoi
vicini e ne fonde la rappresentazione con la propria è il *message passing*, e
la self-attention è il caso in cui quel grafo è completo. Ne segue che i
**Graph Transformer** applicano questo modello a un grafo qualunque, e per non
perdere la topologia devono reintrodurla come codifica posizionale. Quella
codifica nasce dallo stesso ragionamento delle sinusoidi del 2017, ma non ne è
la generalizzazione matematica: là il confronto è svolto nel dettaglio, e le
due famiglie si somigliano senza coincidere.

Longformer e BigBird decidono in anticipo quali archi tenere, in base alla
posizione. Il **Reformer** {cite}`kitaev2020reformer` prende la strada
opposta, e cioè non deciderlo affatto: lascia che siano i dati a dire quali
coppie contano. L'osservazione di partenza è che dopo la softmax quasi
tutta la
massa di attenzione va su pochissime chiavi, quindi calcolare l'intera matrice
è sprecare lavoro su valori destinati a essere quasi zero; e le chiavi che
contano sono quelle con prodotto scalare grande, cioè quelle *vicine* alla
query. Trovare i vicini senza confrontarli tutti è un problema classico, e la
risposta classica è l’**hashing sensibile alla località** (LSH): una funzione
$g$ che manda vettori simili nello stesso secchiello con alta probabilità.

Perché l'hashing funzioni, però, query e chiavi devono coincidere: se
$g(\mathbf{q}_j) \neq g(\mathbf{k}_j)$ una query può finire in un secchiello
dove la sua stessa
chiave non c'è. Il Reformer usa quindi la stessa proiezione per entrambe
(*shared-QK*), rinunciando alla distinzione fra il cercare e l'essere trovati
su cui si basa la {numref}`fig-qkv`; gli autori non trovano perdite
sui due compiti su cui la provano (testo carattere per carattere e immagini di
$64 \times 64$ pixel).

Fatto questo, si raggruppano query e chiavi per secchiello, si calcola
l'attenzione piena solo dentro ciascun secchiello, e il costo scende da
$O(n^2)$ a $O(n \log n)$. L'hashing sbaglia: si ripete con più
funzioni indipendenti per ridurre la probabilità di perdere una coppia
importante, e la sparsità non è più garantita ma probabilistica.

Il secondo ingrediente del Reformer non riguarda l'attenzione ma la memoria:
gli **strati reversibili**. In una rete ordinaria la retropropagazione ha
bisogno delle attivazioni di ogni strato, quindi la memoria cresce con la
profondità. Se però ogni strato è costruito in modo da poter essere invertito
(dalle uscite si ricalcolano gli ingressi), non occorre conservarle: si
ricostruiscono all'indietro quando servono. È il baratto **memoria contro
calcolo** che l'ingegneria del deep learning ripropone a ogni scala, dalla
ricomputazione delle attivazioni al modo in cui FlashAttention evita di
materializzare la matrice di attenzione.
`````

## Quando conviene, e quando no

Il Transformer rende al meglio quando ci sono molti dati di addestramento,
hardware che esegue molti calcoli in parallelo invece che in sequenza (come GPU
e TPU) e legami fra parole molto distanti da catturare: sono le condizioni in
cui si addestrano i grandi modelli linguistici. Le architetture ricorrenti
restano ragionevoli dove la memoria e il calcolo per ogni parola devono restare
costanti, con uno stato di dimensione fissa invece di una memoria che cresce
con il testo: su un dispositivo piccolo, o davanti a un flusso che non finisce
mai, come i sottotitoli in diretta o un traduttore che lavora mentre l'altro
parla.

Come idea, però, non sono tramontate: due linee di ricerca recenti
riportano in gioco un riassunto che si aggiorna passo per passo, come facevano
le RNN, ma costruito in modo da non pagare il costo quadratico dell'attenzione
globale. Si chiamano *attenzioni lineari* e *state space model* (in italiano
«modelli a spazio di stato», dove lo stato è appunto il riassunto che si
aggiorna; il più noto si chiama Mamba), e a ciascuno è dedicato un capitolo: il
{doc}`capitolo sull'attenzione lineare </AttenzioneLineare/overview>` e quello
sugli {doc}`state space model </StateSpaceModel/overview>`.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Le RNN leggono in fila con un riassunto mentale che sbiadisce;
  LSTM e GRU aggiungono un taccuino con delle regole e la memoria dura
  di più, ma si legge sempre una parola alla volta.
- Il Transformer tiene tutto il testo sott'occhio: ogni parola può andare a
  rileggersi qualunque altra, e il lavoro si divide fra tanti processori. Vale
  però soprattutto mentre studia: quando scrive, le parole gli escono lo
  stesso una alla volta.
- Il costo è la riunione dove ognuno parla con ognuno: raddoppiando i
  partecipanti le chiacchiere quadruplicano. Ogni conversazione va segnata su
  un foglio, con una casella per ciascuna: è stato a lungo lo spazio di quel
  foglio, più ancora del tempo, a decidere quanto testo un modello riesce a
  tenere davanti.
- Per spendere meno si tolgono conversazioni, e i modi sono tre: decidere
  in anticipo chi parla con chi (ognuno con i vicini, più qualche
  partecipante che parla con tutti), lasciare che siano i dati a dire quali
  coppie contano, oppure cambiare del tutto il modo di calcolare (il
  {doc}`capitolo sull'attenzione lineare </AttenzioneLineare/overview>`).
- Nessuna architettura vince per sempre: i due capitoli dopo quello sui
  Transformer riportano
  in gioco l'idea del riassunto che si aggiorna, dove la riunione
  plenaria è troppo cara.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- RNN: memoria che sbiadisce e calcolo sequenziale. LSTM/GRU: gate
  che allungano la memoria (tre le prime, contando il *forget* aggiunto nel
  2000; due le seconde), ma sempre in sequenza.
- Il Transformer collega ogni coppia di posizioni in un passo e si
  addestra in parallelo: dipendenze lunghe *e* velocità. In inferenza, però, la
  generazione resta sequenziale.
- Il costo è quadratico nella lunghezza della sequenza, e i due termini
  vanno tenuti distinti: la memoria per i punteggi è $O(n^2)$ se la si
  scrive tutta insieme, e a lungo ha fissato il tetto al contesto;
  il tempo è $O(n^2 d)$, che supera il costo delle matrici dense solo oltre
  $n \approx 6d$, cioè, per i modelli attuali, oltre qualche decina di migliaia
  di token (il calcolo è nella {doc}`matematica di un modello linguistico
  </Matematica/matematica-llm>`).
- Per ridurre quel costo si tolgono archi da un grafo completo, e le
  strade sono tre: uno schema fisso deciso in anticipo (Longformer,
  BigBird), una scelta guidata dai dati con l'hashing sensibile alla
  località (Reformer, da $O(n^2)$ a $O(n\log n)$, con query e
  chiavi condivise), oppure rinunciare del tutto alla softmax e
  fattorizzarla (il capitolo sull'attenzione lineare).
- Nessuna architettura vince per sempre: attenzione lineare e *state space
  model* rimettono in gioco formulazioni
  ricorrenti
  dove l'attenzione è troppo costosa.
```
`````

Prima di quei capitoli resta da vedere che cosa costa l'attenzione mentre il
modello genera, una parola per volta: lo misura la {doc}`sezione
sull'attenzione in pratica <attenzione-in-pratica>`.
