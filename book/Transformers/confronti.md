# Confronto con i modelli precedenti

Ogni nuova architettura va giudicata contro ciò che sostituisce. Prima del
2017 il trattamento delle sequenze era il regno delle reti ricorrenti — RNN e
le loro varianti raffinate, LSTM e GRU, che abbiamo incontrato nel capitolo
sul Natural Language Processing. Vale la pena metterle accanto al Transformer
con onestà: capire *perché* ha vinto, e anche *dove* non vince affatto.

## Tre generazioni di memoria

`````{tab} Elementare
Le **RNN** leggono una parola alla volta portandosi dietro un riassunto
mentale, che però sbiadisce in fretta: alla fine di un paragrafo lungo,
l'inizio è quasi svanito. Le **LSTM** aggiungono un taccuino con delle
regole: cosa annotare, cosa cancellare, cosa rileggere — la memoria dura
molto di più, al prezzo di un meccanismo più complicato. Le **GRU** sono il
taccuino semplificato: regole più snelle, quasi la stessa resa. Il
**Transformer** cambia gioco: niente riassunto e niente taccuino, il testo
resta tutto sott'occhio e ogni parola può andare a rileggersi qualunque
altra, in qualsiasi momento. La memoria non sbiadisce perché non c'è nulla da
ricordare: basta guardare.
`````

`````{tab} Superiore
Le **RNN** mantengono uno stato $h_t = f(h_{t-1}, x_t)$: la dipendenza tra
posizioni distanti $n$ passi attraversa $n$ applicazioni di $f$, e il
gradiente retropropagato si attenua o esplode esponenzialmente (il *vanishing
/ exploding gradient* del capitolo sulle reti neurali). Le **LSTM**
{cite}`hochreiter1997long` introducono una cella di memoria regolata da
tre gate — *input*, *forget*, *output* — che creano un cammino quasi lineare
per il gradiente; le **GRU** {cite}`cho2014learning` ottengono un effetto
simile
con due soli gate, *reset* e *update*, e meno parametri. Entrambe allungano
l'orizzonte della memoria ma restano **sequenziali**: il passo $t$ attende il
passo $t-1$, in addestramento come in inferenza.

Il Transformer porta la lunghezza del cammino tra due posizioni qualsiasi a
$O(1)$ (ogni coppia è collegata direttamente dalla self-attention) e rende
l'addestramento parallelo sull'intera sequenza. È questa combinazione —
dipendenze lunghe *e* parallelismo — che le architetture ricorrenti non
potevano offrire insieme.
`````

## Il conto da pagare: l'attenzione costa quadratica

Il Transformer non è gratis, e il suo tallone d'Achille è proprio il gesto
che lo definisce: far guardare ogni parola a tutte le altre.

`````{tab} Elementare
Pensa a una riunione dove ognuno deve parlare con ognuno. In quattro sono 6
conversazioni; in otto, 28; in mille, quasi mezzo milione. **Raddoppiare i
partecipanti quadruplica circa le chiacchiere**: è la crescita "al quadrato".
Per il Transformer le parole sono i partecipanti: una frase è una riunione
veloce, un libro intero è un'assemblea oceanica che nessun computer regge
volentieri. Le reti ricorrenti, che leggono in fila, non hanno questo
problema: il loro costo cresce di pari passo con la lunghezza, non al
quadrato. Ecco perché tanta ricerca di oggi lavora su come far "parlare" le
parole senza convocare sempre l'assemblea plenaria.
`````

`````{tab} Superiore
La matrice di attenzione ha $n \times n$ elementi: costo in tempo e memoria
$O(n^2 \cdot d)$ nella lunghezza $n$ della sequenza, contro l'$O(n \cdot
d^2)$ delle ricorrenti. Sotto questo vincolo sono nate le finestre di
contesto limitate dei grandi modelli, e una vasta letteratura di rimedi:
attenzione **sparsa** o a finestre locali (Longformer, BigBird),
approssimazioni a rango basso o kernel (Linformer, Performer), e
ottimizzazioni esatte ma efficienti in memoria come FlashAttention, che
riorganizza il calcolo per sfruttare la gerarchia di memoria delle GPU. In
inferenza, inoltre, la generazione autoregressiva resta sequenziale token
per token: il parallelismo del Transformer è un vantaggio soprattutto in
addestramento.
`````

## Un bilancio onesto

Messi su una bilancia: il Transformer domina quando i dati sono tanti,
l'hardware è parallelo e le dipendenze sono lunghe — esattamente il regime
dei grandi modelli linguistici. Le architetture ricorrenti restano sensate su
sequenze molto lunghe a risorse limitate, nei sistemi in tempo reale dove i
dati arrivano in flusso, e come idea non è affatto morta: linee di ricerca
recenti — le *attenzioni lineari* e i cosiddetti *state space model* come
Mamba — riportano meccanismi di tipo ricorrente proprio per aggirare il costo
quadratico dell'attenzione, e sono il tema dei due capitoli che seguono. In
altre parole: il Transformer ha vinto la partita del decennio, non
necessariamente il campionato eterno.

```{admonition} Da ricordare
:class: important
- **RNN**: memoria che sbiadisce e calcolo sequenziale. **LSTM/GRU**: gate
  che allungano la memoria (3 gate le prime, 2 le seconde), ma sempre in
  fila.
- Il **Transformer** collega ogni coppia di posizioni in un passo e si
  addestra in parallelo: dipendenze lunghe *e* velocità.
- Il prezzo è il costo **quadratico** $O(n^2)$ nella lunghezza della
  sequenza: da qui finestre di contesto limitate e la ricerca su attenzioni
  efficienti.
- Nessuna architettura vince per sempre: attenzione lineare e *state space
  model* — i due capitoli che seguono — rimettono in gioco idee ricorrenti
  proprio dove l'attenzione costa troppo.
```
