# Confronto con i modelli precedenti

Ogni nuova architettura va giudicata contro ciò che sostituisce. Prima del
2017 il testo lo trattavano le **reti ricorrenti**, quelle che leggono una
parola alla volta portandosi dietro un riassunto di quel che è venuto prima:
la capostipite si chiama RNN (*recurrent neural network*), e le due varianti
raffinate che l'hanno soppiantata si chiamano LSTM e GRU. Le abbiamo
incontrate nel {doc}`capitolo sul Natural Language Processing </NaturalLanguageProcessing/overview>`. Conviene metterle
accanto al Transformer con onestà: capire *perché* ha vinto, e anche *dove*
non vince affatto.

## Tre generazioni di memoria

Le tre generazioni sono le tre ricorrenti, RNN, LSTM e GRU; il Transformer non
è la quarta, è quello che chiude la serie cambiando gioco. E per capire perché
la serie sia finita conviene guardare da vicino il difetto della prima.

```{figure} ../figures/rnn-reti-con-memoria.svg
:name: fig-rnn-srotolamento
:alt: "A sinistra una cella ricorrente con una freccia che rientra su sé stessa, il cappio dello stato. A destra la stessa cella srotolata nel tempo in tre copie identiche, ciascuna che riceve una parola e passa lo stato alla successiva: sono la stessa cella, con gli stessi pesi, applicata a istanti diversi."
:width: 96%

Il cappio e il suo srotolamento. Le tre copie a destra sono la stessa rete,
riusata a ogni passo, ed è per questo che una RNN funziona su
sequenze di lunghezza qualsiasi.
```

L'equivalenza di {numref}`fig-rnn-srotolamento` è anche la radice del problema
che questa sezione racconta. Quel «riassunto» che la rete si porta dietro è,
come sempre in questo capitolo, una lista di numeri, e la scatola che lo
aggiorna a ogni parola (nel disegno, la **cella**) è sempre la stessa, con
dentro sempre gli stessi numeri: aggiornare il riassunto vuol dire dunque
moltiplicarlo, parola dopo parola, per quegli stessi numeri.

Ed è lì che casca tutto, perché una moltiplicazione ripetuta cento volte non
perdona. Diciamo che il fattore sia $0{,}9$, cioè che a ogni passo il ricordo
conservi nove decimi di sé: non sembra una perdita grave. Dopo cento parole
resta $0{,}9$ moltiplicato per sé stesso cento volte, cioè $0{,}000027$, che di
quel ricordo è meno di un trentamillesimo. E anche partendo da $0{,}99$, che è
quasi non perdere niente, dopo cento parole si è scesi a $0{,}37$: più di metà
del ricordo è svanito senza che nessuno l'abbia buttato via. Il numero $0{,}9$
qui è messo per far vedere il meccanismo, non è misurato su una rete vera; ma
qualunque fattore minore di uno finisce nello stesso posto, ed è tutta lì la
ragione per cui l'inizio di un testo lungo sbiadisce.

`````{tab} Elementare
Le **RNN** leggono una parola alla volta portandosi dietro un riassunto
mentale, che però sbiadisce in fretta: alla fine di un paragrafo lungo,
l'inizio è quasi svanito. Le **LSTM** aggiungono un taccuino con delle regole:
cosa annotare, cosa cancellare, cosa rileggere (la memoria dura molto di più,
al prezzo di un meccanismo più complicato). Le **GRU** sono il taccuino
semplificato: regole più snelle, quasi la stessa resa. Taccuino o no, però, si
legge sempre una parola alla volta, e la parola dopo aspetta che sia finita
quella prima. Il **Transformer** cambia gioco: niente riassunto e niente
taccuino, il testo resta tutto sott'occhio e ogni parola può andare a
rileggersi qualunque altra. La memoria non sbiadisce perché non c'è nulla da
ricordare: basta guardare.

Due precisazioni, perché il «qualunque altra» va preso con le pinze. La prima:
vale per la torre che **legge** (l'encoder); quella che **scrive** (il decoder)
ha una regola ferrea, non si sbircia avanti, e guarda solo all'indietro.

La seconda riguarda la velocità. Il ricordo che non sbiadisce, da solo, non
spiega la vittoria, perché il taccuino delle LSTM allungava già la memoria:
quello che nessuna macchina di prima sapeva fare era mettere tante mani sullo
stesso testo. Se le parole
si guardano tutte insieme invece che in fila, il lavoro si può spartire fra
migliaia di processori che macinano in parallelo: sono i «cento amici» di
prima, quelli che con un libro da leggere in fila non servivano a niente e qui
invece servono eccome. Attenzione però a quando: succede mentre il modello
**studia**, cioè quando ha davanti tutto il
testo e può lavorarci sopra in una volta sola. Quando poi scrive, le parole
gli escono comunque una alla volta, perché per scegliere la prossima deve
sapere quale ha appena scritto: lì i cento amici tornano a girarsi i pollici,
e infatti generare resta lento.
`````

`````{tab} Superiore
Le **RNN** mantengono uno stato
$\mathbf{h}_t = f(\mathbf{h}_{t-1}, \mathbf{x}_t)$: la dipendenza tra
posizioni distanti $m$ passi attraversa $m$ applicazioni di $f$, e il
gradiente retropropagato si attenua o esplode esponenzialmente (il *vanishing
/ exploding gradient* del
{doc}`capitolo sulle reti neurali </RetiNeurali/overview>`). Le **LSTM**
{cite}`hochreiter1997long` introducono una cella di memoria regolata da gate,
che creano un cammino quasi lineare per il gradiente: nell'architettura del
1997 sono due, *input* e *output*, e diventano tre quando Gers, Schmidhuber e
Cummins aggiungono il *forget gate* {cite}`gers2000learning`; le **GRU**
{cite}`cho2014learning` ottengono un effetto simile con due soli gate, *reset*
e *update*, e meno parametri. Entrambe allungano
l'orizzonte della memoria ma restano **sequenziali**: il passo $t$ attende il
passo $t-1$, in addestramento come in inferenza.

Il Transformer porta la lunghezza del cammino tra due posizioni qualsiasi a
$O(1)$ (ogni coppia è collegata direttamente dalla self-attention) e rende
l'addestramento parallelo sull'intera sequenza. È questa combinazione
(dipendenze lunghe *e* parallelismo) che le architetture ricorrenti non
potevano offrire insieme. Il parallelismo, però, è un vantaggio soprattutto
**in addestramento**, perché in inferenza la generazione autoregressiva resta
sequenziale, un token alla volta.
`````

## Il conto da pagare: l'attenzione cresce col quadrato

Il Transformer non è gratis, e il suo tallone d'Achille è proprio il gesto
che lo definisce: far guardare ogni parola a tutte le altre.

`````{tab} Elementare
Quattro persone in una stanza, e ognuno deve parlare con ognuno: io con te, io
con lui, io con lei, tu con lui, tu con lei, lui con lei. Sei coppie, contate
con le dita. Senza dita: $4 \times 3 = 12$, ognuno con tutti tranne sé, e
$12 : 2 = 6$ perché ogni coppia è finita nel conto due volte. In otto,
$8 \times 7 : 2 = 28$; in mille, $1000 \times 999 : 2 = 499\,500$, quasi mezzo
milione. Raddoppiando i presenti le chiacchiere quadruplicano, o quasi, ed è
la crescita al quadrato.

Per un Transformer i presenti sono le parole, con due usanze: si parla anche da
soli, e ascoltare non conta come farsi ascoltare (che "salta" guardi "gatto" è
un conto, il rovescio un altro). In quattro i confronti diventano
$4 \times 4 = 16$, con la stessa crescita di prima. Una frase è una riunione
svelta, un libro un'assemblea oceanica che nessun computer regge volentieri. Le
reti ricorrenti, che leggono in fila, il problema non ce l'hanno: un presente
in più è un turno in più.

Poi c'è il tabellone, una casella per ogni scambio: sedici in quattro, un
milione in mille. Il tempo alla peggio lo si aspetta; il tabellone o sta nella
stanza o non ci sta, ed è lui a decidere quanto testo un modello si tiene
davanti. Di qui i tre modi di far parlare tutti senza convocare la plenaria.

Il primo fissa il programma prima di entrare: ognuno con i vicini di posto,
qualche coppia sorteggiata per accorciare le distanze, pochi che parlano con
tutti e fanno da ponte. Si arriva ancora dove arrivava la plenaria, e lo si
dimostra; a reggere la dimostrazione sono i ponti, non i vicini. Ma i ponti
sono pochi e in un giro non ripetono tutto a tutti: quello che la plenaria
sbrigava in una volta vuole più giri, e i giri crescono con i presenti. Si
risparmia in larghezza e si paga in altezza.

Il secondo lascia decidere alla sala. Ognuno ha da dire qualcosa a pochissimi,
e le altre conversazioni si tengono lo stesso a vuoto: lì il lavoro si spreca.
All'ingresso, allora, i presenti vanno a tavoli per affinità e parlano con chi
si ritrovano accanto. Il tavolo giusto si trova solo se chi cerca e chi va
trovato portano lo stesso cartellino, ed è una rinuncia, perché in plenaria
cercare e farsi trovare erano due mestieri distinti; chi ci ha provato non ha
visto la riunione riuscire peggio. E i tavoli sbagliano: due che avevano da
dirsi qualcosa finiscono separati, e nessuno se ne accorge. Si rimescola più
volte con criteri diversi, così l'occasione persa a un giro si recupera al
successivo, ma nessuno promette che non ne resti fuori una.

Gli stessi organizzatori hanno un secondo accorgimento, e non è un altro modo
di sfoltire: non riguarda chi parla con chi, riguarda i verbali. Di ogni giro
se ne tiene uno, perché a riunione finita bisogna tornarci sopra per capire che
cosa ha funzionato, e più giri più verbali. Se un giro si può ripercorrere a
ritroso, il verbale si butta e si riscrive rifacendo i conti: spazio
risparmiato, fatica in più, un baratto che in questo mestiere torna di
continuo.

Il terzo butta la lista degli invitati invece di sfoltirla: i conti si
riordinano perché le coppie non si formino mai una per una, invece di
calcolarle tutte e scartarne poi quasi tutte. Costa una rinuncia (il modo in
cui i punteggi diventano intensità va cambiato) e ha un capitolo suo, quello
sull'attenzione lineare.
`````

`````{tab} Superiore
La matrice di attenzione ha $n \times n$ elementi. Contando la sola
operazione di attenzione (proiezioni escluse), il costo in tempo è
$O(n^2 \cdot d)$ nella lunghezza $n$ della sequenza, contro l’$O(n \cdot
d^2)$ delle ricorrenti; la memoria per i punteggi è $O(n^2)$, contro
l’$O(n \cdot d)$ delle attivazioni ricorrenti. Sotto questo vincolo sono
nate le finestre di contesto limitate dei grandi modelli, e una vasta
letteratura di rimedi:
attenzione **sparsa** o a finestre locali (Longformer, BigBird),
approssimazioni a rango basso o kernel (Linformer, Performer), e
ottimizzazioni esatte ma efficienti in memoria come FlashAttention, che
riorganizza il calcolo per sfruttare la gerarchia di memoria delle GPU.

L'attenzione sparsa nasce da un cambio di punto di vista, più che da un trucco
di calcolo. La matrice di attenzione è la matrice di adiacenza di un
**grafo completo** sui token, e ridurne il costo è un problema
di **sparsificazione di grafi**. Longformer {cite}`beltagy2020longformer`
toglie archi tenendo una finestra scorrevole attorno a ogni token, qualche
finestra dilatata per allargare la portata, e un pugno di **token globali**
collegati a tutti (nel question answering, quelli della domanda): il costo
scende da $O(n^2)$ a $O(n w)$, dove $w$ è l'ampiezza della finestra, un numero
fissato in anticipo e molto minore di $n$. BigBird {cite}`zaheer2020big`
prende la stessa
strada dichiarando la cosa: combina una finestra ad anello (un grafo «piccolo
mondo» alla Watts-Strogatz), archi **casuali** alla Erdős-Rényi e token
globali, e dimostra che il modello risultante resta un approssimatore
universale di funzioni su sequenze.

Quel teorema va letto con le sue ipotesi accanto, che è facile dimenticare
proprio davanti ai risultati che fanno comodo. Vale per le funzioni
**continue su un dominio limitato**; e vale per
qualunque schema sparso **che contenga i token globali**, cioè sono loro a
portare il teorema, non la finestra. Soprattutto, gli stessi autori mostrano
il rovescio nello stesso lavoro, e anche quel rovescio ha la sua ipotesi:
esiste un compito che l'attenzione piena risolve in un numero costante di
strati e che **qualunque** attenzione sparsa con un numero di archi
proporzionale a $n$ costringe a una profondità che cresce con $n$, «under
standard complexity theoretic assumptions», cioè ammettendo la congettura dei
vettori ortogonali, che nessuno ha dimostrato. «Universale» vuol dire che ci
si arriva, non che ci si arriva alla stessa profondità: la sparsificazione non
è gratis, baratta ampiezza con altezza.

Il capitolo sulle reti neurali su grafo
riprende questa lettura dall'altro capo, mostrando che la self-attention è
message passing su un grafo completo, e ne trae la conseguenza: i **Graph
Transformer** applicano questo modello a un grafo qualunque, e per non perdere
la topologia devono reintrodurla come codifica posizionale. Quella codifica
nasce dallo stesso ragionamento delle sinusoidi viste qui, ma non ne è la
stessa cosa scritta in generale: là il confronto è fatto numero alla mano, e
le due famiglie si somigliano senza coincidere.

Longformer e BigBird decidono **in anticipo** quali archi tenere, in base alla
posizione. Il **Reformer** {cite}`kitaev2020reformer` prende la strada
opposta, e cioè non deciderlo affatto: **lascia che siano i dati a dire quali
coppie contano**. L'osservazione di partenza è che dopo la softmax quasi
tutta la
massa di attenzione va su pochissime chiavi, quindi calcolare l'intera matrice
è sprecare lavoro su valori destinati a essere quasi zero; e le chiavi che
contano sono quelle con prodotto scalare grande, cioè quelle *vicine* alla
query. Trovare i vicini senza confrontarli tutti è un problema classico, e la
risposta classica è l’**hashing sensibile alla località** (LSH): una funzione
$g$ che manda vettori simili nello stesso secchiello con alta probabilità.
Perché l'hashing funzioni, però, query e chiavi devono **coincidere**: se
$g(\mathbf{q}_j) \neq g(\mathbf{k}_j)$ una query può finire in un secchiello
dove la sua stessa
chiave non c'è. Il Reformer usa quindi la stessa proiezione per entrambe
(*shared-QK*), rinunciando alla distinzione fra il cercare e l'essere trovati
su cui si regge la {numref}`fig-qkv`; gli autori misurano che questa rinuncia
non costa prestazioni, il che è di per sé un'informazione interessante. Fatto
questo, si raggruppano query e chiavi per secchiello, si calcola l'attenzione
piena solo dentro ciascun secchiello, e il costo scende da $O(n^2)$ a
$O(n \log n)$. Il prezzo ulteriore è che l'hashing sbaglia: si ripete con più
funzioni indipendenti per ridurre la probabilità di perdere una coppia
importante, e la sparsità non è più garantita ma probabilistica.

Il secondo ingrediente del Reformer non riguarda l'attenzione ma la memoria, e
merita di essere ricordato perché è trasversale: gli **strati reversibili**.
In una rete ordinaria la retropropagazione ha bisogno delle attivazioni di
ogni strato, quindi la memoria cresce con la profondità. Se però ogni strato è
costruito in modo da poter essere **invertito** (dalle uscite si ricalcolano
gli ingressi), quelle attivazioni non serve tenerle: si buttano e si
ricostruiscono all'indietro quando servono. È il baratto **memoria contro
calcolo** che l'ingegneria del deep learning ripropone a ogni scala, dalla
ricomputazione delle attivazioni al modo in cui FlashAttention evita di
materializzare la matrice di attenzione.
`````

## Un bilancio onesto

Messi su una bilancia, il Transformer vince quando ricorrono tre condizioni
insieme: c'è tantissimo testo su cui studiare, c'è una macchina che sa fare
molti conti insieme invece che uno dopo l'altro, e conta capire legami fra
parole molto distanti fra loro. Sono esattamente le condizioni in cui vivono i
sistemi tipo ChatGPT. Le architetture ricorrenti restano sensate quando le
risorse sono poche e il testo è lunghissimo, e nei sistemi che devono
rispondere mentre le parole arrivano, una alla volta, senza poter aspettare la
fine (i sottotitoli in diretta, per dire, o un traduttore che lavora mentre
l'altro parla). E come idea non sono affatto morte: due linee di ricerca
recenti rimettono in mezzo un riassunto che si aggiorna passo per passo,
proprio come facevano le RNN, ma costruito in modo da non pagare il costo
della riunione plenaria. Si chiamano *attenzioni lineari* e *state space
model* (in italiano «modelli a spazio di stato», dove lo stato è appunto il
riassunto che si aggiorna; il più noto si chiama Mamba), e hanno un capitolo
ciascuna subito dopo questo. In altre parole: il Transformer ha vinto la
partita del decennio, non necessariamente il campionato eterno.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Le **RNN** leggono in fila con un riassunto mentale che sbiadisce;
  **LSTM** e **GRU** aggiungono un taccuino con delle regole e la memoria dura
  di più, ma si legge sempre una parola alla volta.
- Il **Transformer** tiene tutto il testo sott'occhio: ogni parola può andare a
  rileggersi qualunque altra, e il lavoro si divide fra tanti processori. Vale
  però soprattutto **mentre studia**: quando scrive, le parole gli escono lo
  stesso una alla volta.
- Il prezzo è la riunione dove ognuno parla con ognuno: raddoppiando i
  partecipanti le chiacchiere quadruplicano. E ogni conversazione va segnata su
  un foglio, con una casella per ciascuna: è lo spazio di quel foglio, più
  ancora del tempo, a decidere quanto testo un modello riesce a tenere davanti.
- Per spendere meno si tolgono conversazioni, e i modi sono tre: decidere
  **in anticipo** chi parla con chi (ognuno con i vicini, più qualche
  partecipante che parla con tutti), lasciare che siano **i dati** a dire quali
  coppie contano, oppure cambiare del tutto il modo di fare i conti (il
  {doc}`capitolo sull'attenzione lineare </AttenzioneLineare/overview>`).
- Nessuna architettura vince per sempre: i due capitoli che seguono riportano
  in gioco l'idea del riassunto che si aggiorna, proprio dove la riunione
  plenaria costa troppo.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- **RNN**: memoria che sbiadisce e calcolo sequenziale. **LSTM/GRU**: gate
  che allungano la memoria (tre le prime, contando il *forget* aggiunto nel
  2000; due le seconde), ma sempre in fila.
- Il **Transformer** collega ogni coppia di posizioni in un passo e si
  addestra in parallelo: dipendenze lunghe *e* velocità. In inferenza, però, la
  generazione resta sequenziale.
- Il prezzo è **quadratico** nella lunghezza della sequenza, e i due termini
  vanno tenuti distinti: la **memoria** per i punteggi è $O(n^2)$, ed è questo
  il tetto al contesto; il **tempo** è $O(n^2 d)$, che però nei modelli
  attuali non è il termine dominante alle lunghezze correnti (i conti stanno
  nella sezione sui grandi modelli linguistici).
- Ridurre quel costo vuol dire **togliere archi** da un grafo completo, e le
  strade sono tre: uno schema **fisso** deciso in anticipo (Longformer,
  BigBird), una scelta **guidata dai dati** con l'hashing sensibile alla
  località (**Reformer**, da $O(n^2)$ a $O(n\log n)$, al prezzo di query e
  chiavi condivise), oppure rinunciare del tutto alla softmax e
  **fattorizzarla** (il capitolo sull'attenzione lineare).
- Nessuna architettura vince per sempre: attenzione lineare e *state space
  model* (i due capitoli che seguono) rimettono in gioco idee ricorrenti
  proprio dove l'attenzione costa troppo.
```
`````
