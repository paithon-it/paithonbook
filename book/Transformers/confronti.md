# Confronto con i modelli precedenti

Ogni nuova architettura va giudicata contro ciò che sostituisce. Prima del
2017 il trattamento delle sequenze era il regno delle reti ricorrenti: RNN e
le loro varianti raffinate, LSTM e GRU, che abbiamo incontrato nel capitolo
sul Natural Language Processing. Vale la pena metterle accanto al Transformer
con onestà: capire *perché* ha vinto, e anche *dove* non vince affatto.

## Tre generazioni di memoria

```{figure} ../figures/rnn-reti-con-memoria.svg
:name: fig-rnn-srotolamento
:alt: "A sinistra una cella ricorrente con una freccia che rientra su sé stessa, il cappio dello stato. A destra la stessa cella srotolata nel tempo in tre copie identiche, ciascuna che riceve una parola e passa lo stato alla successiva: sono la stessa cella, con gli stessi pesi, applicata a istanti diversi."
:width: 96%

Il cappio e il suo srotolamento. Le tre copie a destra non sono tre reti: sono
la stessa, riusata a ogni passo, ed è per questo che una RNN funziona su
sequenze di lunghezza qualsiasi.
```

L'equivalenza di {numref}`fig-rnn-srotolamento` è anche la radice del problema
che questa sezione racconta. Aggiornare il riassunto, a ogni passo, vuol dire
moltiplicarlo per gli stessi numeri, perché la cella è sempre quella: se il
riassunto attraversa
la stessa moltiplicazione a ogni passo, dopo cento passi quel
fattore è stato applicato cento volte, e un numero poco più piccolo di uno
diventa quasi zero: $0{,}9$ moltiplicato per sé stesso cento volte fa
$0{,}000027$, e perfino $0{,}99$ scende a $0{,}37$. È un conto da calcolatrice,
e vale la pena farlo, perché è tutta lì la ragione per cui l'inizio di un testo
lungo sbiadisce.

`````{tab} Elementare
Le **RNN** leggono una parola alla volta portandosi dietro un riassunto
mentale, che però sbiadisce in fretta: alla fine di un paragrafo lungo,
l'inizio è quasi svanito. Le **LSTM** aggiungono un taccuino con delle regole:
cosa annotare, cosa cancellare, cosa rileggere (la memoria dura molto di più,
al prezzo di un meccanismo più complicato). Le **GRU** sono il taccuino
semplificato: regole più snelle, quasi la stessa resa. Il **Transformer**
cambia gioco: niente riassunto e niente taccuino, il testo resta tutto
sott'occhio e ogni parola può andare a rileggersi qualunque altra. La memoria
non sbiadisce perché non c'è nulla da ricordare: basta guardare.

Due precisazioni, perché altrimenti sembrano contraddizioni. La prima: «tutte
le altre» vale per la torre che **legge**; quella che **scrive** ha la regola
ferrea della sezione precedente, non si sbircia avanti, e guarda solo
all'indietro. La seconda: leggere tutto insieme fa risparmiare tempo
soprattutto **mentre il modello studia**, che è quando i cento amici possono
dividersi il lavoro. Quando poi scrive, le parole gli escono comunque una alla
volta, perché per scegliere la prossima deve sapere quale ha appena scritto: lì
i cento amici non servono a niente, e infatti generare resta lento.
`````

`````{tab} Superiore
Le **RNN** mantengono uno stato
$\mathbf{h}_t = f(\mathbf{h}_{t-1}, \mathbf{x}_t)$: la dipendenza tra
posizioni distanti $n$ passi attraversa $n$ applicazioni di $f$, e il
gradiente retropropagato si attenua o esplode esponenzialmente (il *vanishing
/ exploding gradient* del capitolo sulle reti neurali). Le **LSTM**
{cite}`hochreiter1997long` introducono una cella di memoria regolata da tre
gate (*input*, *forget*, *output*) che creano un cammino quasi lineare per il
gradiente; le **GRU** {cite}`cho2014learning` ottengono un effetto simile con
due soli gate, *reset* e *update*, e meno parametri. Entrambe allungano
l'orizzonte della memoria ma restano **sequenziali**: il passo $t$ attende il
passo $t-1$, in addestramento come in inferenza.

Il Transformer porta la lunghezza del cammino tra due posizioni qualsiasi a
$O(1)$ (ogni coppia è collegata direttamente dalla self-attention) e rende
l'addestramento parallelo sull'intera sequenza. È questa combinazione
(dipendenze lunghe *e* parallelismo) che le architetture ricorrenti non
potevano offrire insieme. Con un'avvertenza che vale la pena mettere subito: il
parallelismo è un vantaggio soprattutto **in addestramento**, perché in
inferenza la generazione autoregressiva resta sequenziale, un token alla volta.
`````

## Il conto da pagare: l'attenzione costa quadratica

Il Transformer non è gratis, e il suo tallone d'Achille è proprio il gesto
che lo definisce: far guardare ogni parola a tutte le altre.

`````{tab} Elementare
Pensa a una riunione dove ognuno deve parlare con ognuno. In quattro sono 6
conversazioni; in otto, 28; in mille, quasi mezzo milione. **Raddoppiare i
partecipanti quadruplica circa le chiacchiere**: è la crescita "al quadrato".
Per il Transformer le parole sono i partecipanti, con due differenze da poco:
ogni parola guarda anche sé stessa, e chi ascolta chi conta separatamente nei
due versi, quindi in quattro i confronti sono sedici invece di sei. La crescita
è la stessa, ed è quella che interessa: una frase è una riunione
veloce, un libro intero è un'assemblea oceanica che nessun computer regge
volentieri. Le reti ricorrenti, che leggono in fila, non hanno questo
problema: il loro costo cresce di pari passo con la lunghezza, non al
quadrato.

Ecco perché tanta ricerca lavora su come far parlare le parole senza convocare
sempre l'assemblea plenaria, e i modi sono tre. Il primo è **decidere in
anticipo chi parla con chi**: ognuno con i vicini di posto, più qualche
partecipante scelto che parla con tutti e fa da ponte. Il secondo è **lasciare
che siano i dati a dirlo**: si osserva che quasi tutta l'attenzione di una
parola finisce comunque su pochissime altre, e allora si cerca un modo di
trovare in fretta quelle poche invece di provarle tutte. Il terzo è più
radicale, e cambia l'aritmetica invece della lista degli invitati: ha un
capitolo suo, quello sull'attenzione lineare.
`````

`````{tab} Superiore
La matrice di attenzione ha $n \times n$ elementi. Contando la sola
operazione di attenzione (proiezioni escluse), il costo in tempo è
$O(n^2 \cdot d)$ nella lunghezza $n$ della sequenza, contro l'$O(n \cdot
d^2)$ delle ricorrenti; la memoria per i punteggi è $O(n^2)$, contro
l'$O(n \cdot d)$ delle attivazioni ricorrenti. Sotto questo vincolo sono
nate le finestre di contesto limitate dei grandi modelli, e una vasta
letteratura di rimedi:
attenzione **sparsa** o a finestre locali (Longformer, BigBird),
approssimazioni a rango basso o kernel (Linformer, Performer), e
ottimizzazioni esatte ma efficienti in memoria come FlashAttention, che
riorganizza il calcolo per sfruttare la gerarchia di memoria delle GPU.

Vale la pena dire da dove viene il primo di quei rimedi, perché è un cambio di
punto di vista più che un trucco. La matrice di attenzione è la matrice di
adiacenza di un **grafo completo** sui token, e ridurne il costo è un problema
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

Su quel teorema conviene mettere le ipotesi accanto, che è la regola che questo
libro si dà e che è facile dimenticare proprio davanti ai risultati che fanno
comodo. Vale per le funzioni **continue su un dominio limitato**; e vale per
qualunque schema sparso **che contenga i token globali**, cioè sono loro a
portare il teorema, non la finestra. Soprattutto, gli stessi autori dimostrano
il rovescio nella stessa pagina: esiste un compito che l'attenzione piena
risolve in un numero costante di strati e che **qualunque** attenzione sparsa
con un numero di archi proporzionale a $n$ costringe a una profondità che
cresce con $n$. «Universale» vuol dire che ci si arriva, non che ci si arriva
alla stessa profondità: la sparsificazione non è gratis, baratta ampiezza con
altezza.

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
risposta classica è l'**hashing sensibile alla località** (LSH): una funzione
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

Messi su una bilancia: il Transformer domina quando i dati sono tanti,
l'hardware è parallelo e le dipendenze sono lunghe (esattamente il regime dei
grandi modelli linguistici). Le architetture ricorrenti restano sensate su
sequenze molto lunghe a risorse limitate e nei sistemi che devono rispondere
mentre i dati arrivano, uno alla volta, senza poter aspettare la fine del
testo. E come idea non sono affatto morte: due linee di ricerca recenti
rimettono in mezzo un riassunto che si aggiorna passo per passo, proprio come
facevano le RNN, ma costruito in modo da non pagare il costo della riunione
plenaria. Si chiamano *attenzioni lineari* e *state space model* (il più noto
si chiama Mamba), e hanno un capitolo ciascuna subito dopo questo. In altre
parole: il Transformer ha vinto la partita del decennio, non necessariamente il
campionato eterno.

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
  partecipanti le chiacchiere quadruplicano. Da qui il limite alla lunghezza
  del testo che un modello riesce a tenere davanti.
- Per spendere meno si tolgono conversazioni, e i modi sono tre: decidere
  **in anticipo** chi parla con chi (ognuno con i vicini, più qualche
  partecipante che parla con tutti), lasciare che siano **i dati** a dire quali
  coppie contano, oppure cambiare del tutto il modo di fare i conti (il
  capitolo sull'attenzione lineare).
- Nessuna architettura vince per sempre: i due capitoli che seguono riportano
  in gioco l'idea del riassunto che si aggiorna, proprio dove la riunione
  plenaria costa troppo.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- **RNN**: memoria che sbiadisce e calcolo sequenziale. **LSTM/GRU**: gate
  che allungano la memoria (3 gate le prime, 2 le seconde), ma sempre in
  fila.
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
