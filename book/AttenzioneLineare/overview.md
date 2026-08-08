# Attenzione lineare

Nel 2020, mentre il mondo dell'intelligenza artificiale celebrava i
Transformer come la rottura definitiva con il passato ricorrente, quattro
ricercatori (Katharopoulos, Vyas, Pappas e Fleuret, tra l'Idiap, l'EPFL e la
University of Washington)
pubblicano un articolo dal titolo che suona come una provocazione:
*Transformers are RNNs* {cite}`katharopoulos2020transformers`. La tesi è tanto
semplice quanto spiazzante. Togliete al meccanismo di attenzione la sua
funzione softmax (quella normalizzazione che, davanti a una parola, pesa tutte
le altre l'una contro l'altra) e il Transformer, il modello che aveva appena
spodestato le reti ricorrenti, ricade esattamente in una **rete ricorrente**.
Il re, sotto il mantello, era un vecchio parente.

Non è un gioco di prestigio: è una porta. Nel capitolo sui Transformer abbiamo
visto il prezzo dell'attenzione: un costo **quadratico** nella lunghezza della
sequenza, perché ogni parola guarda tutte le altre, e in generazione una
*cache* di chiavi e valori che **cresce** parola dopo parola. È il muro contro
cui sbattono i contesti lunghi. La provocazione di Katharopoulos indica una
via per aggirarlo: se l'attenzione, spogliata della softmax, è una RNN, allora
possiamo riscriverla come una ricorrenza a **stato di dimensione fissa**
(costo lineare nella lunghezza, memoria costante in generazione) senza
rinunciare del tutto a ciò che l'aveva resa vincente.

```{figure} ../figures/kv-cache-generazione.svg
:name: fig-kv-cache-cresce
:alt: "Sequenza di passi di generazione affiancati. A ogni token prodotto, la cache delle chiavi e dei valori si allunga di una colonna, e il blocco di memoria occupato cresce di passo in passo senza mai liberarsi, fino a dominare l'occupazione."
:width: 96%

La cache che non smette di crescere. Ogni token generato ne aggiunge un pezzo,
e quel pezzo resta: la memoria occupata non dipende dal modello ma da quanto
si è scritto finora.
```

{numref}`fig-kv-cache-cresce` è il muro in una figura, ed è il motivo per cui
tutto questo capitolo esiste. Una ricorrenza a stato fisso non fa crescere
niente: comprime il passato in una memoria di taglia costante, e la domanda
diventa quanto si perde nel comprimerlo.

## Il compromesso che tutti inseguono

`````{tab} Elementare

Le due grandi famiglie di modelli per sequenze hanno ciascuna un pregio e un
difetto speculari. I Transformer si addestrano in fretta, perché guardano
tutta la frase in una volta e sfruttano a pieno le schede grafiche; ma per
farlo devono tenere tutto sott'occhio, e più il testo è lungo più questo
costa: in fretta diventa insostenibile. Le vecchie reti ricorrenti fanno il
contrario: leggono una parola alla volta portandosi dietro un riassunto di
dimensione fissa, quindi in lettura costano poco e non si spaventano davanti a
testi lunghissimi; ma proprio perché procedono in fila, addestrarle è lento.

Il sogno è avere le due cose insieme: la **velocità di addestramento** dei
Transformer e il **basso costo in lettura** delle reti ricorrenti. È esattamente
ciò che promette l'attenzione lineare, e con lei tutta la famiglia di modelli di
questo capitolo.

`````

`````{tab} Superiore

Formalizziamo il compromesso. L'attenzione softmax costa $O(n^2 d)$ nella
lunghezza $n$ della sequenza (la matrice di affinità è $n \times n$) e in
generazione autoregressiva conserva tutte le chiavi e i valori passati:
memoria che cresce linearmente con il contesto. Una rete ricorrente costa
$O(n d^2)$ (lineare in $n$) e mantiene uno stato di dimensione fissa, ma il
passo $t$ dipende dal passo $t-1$: niente parallelismo lungo la sequenza.

L'attenzione lineare vive nel punto d'incontro: espone **due forme
equivalenti** dello stesso calcolo. Una forma *parallela*, per addestrare
sull'intera sequenza sfruttando le GPU (in pratica, come vedremo, spezzata a
blocchi per tenere il costo lineare); e una forma *ricorrente*, per generare
a costo e memoria costanti per token: nessuna *cache* che si gonfia. È la
proprietà che inseguono, con ingredienti diversi, tutte le architetture di
questi due capitoli.

`````

## Una sola idea, molte incarnazioni

Se c'è una tesi che tiene insieme questo capitolo e il successivo, è questa:
tutti questi modelli sono **reti ricorrenti lineari** con uno stato di dimensione
fissa, aggiornato a ogni token da una ricorrenza della forma

$$
S_t = S_{t-1}\,(\text{transizione}_t) + (\text{scrittura}_t).
$$

Lo stato $S_t$ è una piccola matrice (una memoria che associa chiavi a valori)
e ciò che distingue un modello dall'altro è, essenzialmente, **come si
aggiorna quella memoria**: se ci si limita ad accumulare, se si impara a
dimenticare, se si corregge ciò che è già scritto. Cambiare la regola di
transizione significa passare dall'attenzione lineare a RetNet, a Mamba, a
DeltaNet: nomi che incontreremo come varianti di uno stesso scheletro, non
come invenzioni scollegate.

## Come è organizzato il capitolo

Tre tappe, dal meccanismo alle architetture concrete.

Si parte da **come l'attenzione diventa lineare**: il "trucco del kernel" che
spezza la softmax e trasforma l'attenzione in una ricorrenza a stato-matrice,
con la sua doppia natura parallelo/ricorrente, e il limite dell'accumulo puro,
una memoria che si riempie e non dimentica. Seconda tappa, **scrivere meglio
nella memoria**: i due strumenti che risolvono quel limite (i *gate* per
dimenticare e la *delta rule* per correggere) fino alla tabella che unifica
attenzione lineare, GLA, DeltaNet e Gated DeltaNet come casi di una stessa
regressione online. Terza tappa, **le architetture lineari** che oggi si
misurano con i Transformer (RetNet, RWKV, xLSTM) ciascuna un'istanza dello
stesso principio. Chiude un breve **notebook** in cui verifichiamo con NumPy,
in poche righe, che la forma parallela e quella ricorrente calcolano davvero
la stessa funzione.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Tolta la softmax, quella normalizzazione che pesa ogni parola contro tutte le
  altre, il Transformer si riscopre una **rete ricorrente**
  {cite}`katharopoulos2020transformers`: legge una parola alla volta portandosi
  dietro un riassunto di dimensione sempre uguale.
- È la via per aggirare i due conti che i Transformer pagano sui testi lunghi:
  il lavoro che cresce a valanga con la lunghezza, e la memoria di appoggio che
  si allunga a ogni parola generata.
- Il compromesso che tutta la famiglia insegue: la **velocità di addestramento**
  dei Transformer *e* il **basso costo in lettura** delle vecchie reti
  ricorrenti, perché lo stesso calcolo si può fare in due modi equivalenti,
  tutto insieme oppure una parola alla volta.
- Tesi unificante dei due capitoli: sono tutti modelli che tengono un riassunto
  di taglia fissa e lo aggiornano a ogni parola; a cambiare, dall'uno all'altro,
  è **come si aggiorna quel riassunto**: chi si limita ad aggiungere, chi impara
  a dimenticare, chi corregge ciò che è già scritto.
- Il percorso: come l'attenzione diventa economica, poi come si scrive meglio nel
  riassunto (dimenticare e correggere), infine le architetture concrete (RetNet,
  RWKV, xLSTM).
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Togliendo la softmax, l'attenzione diventa una **rete ricorrente** a stato
  fisso {cite}`katharopoulos2020transformers`: è la chiave per aggirare il costo
  quadratico e la *cache* crescente dei Transformer.
- Il compromesso inseguito da tutta la famiglia: **addestramento parallelo** come
  i Transformer *e* **inferenza a memoria costante** come le RNN, grazie a due
  forme equivalenti (parallela e ricorrente) dello stesso calcolo.
- Tesi unificante dei due capitoli: sono tutte **RNN lineari** con stato di
  dimensione fissa,
  $S_t = S_{t-1}\,(\text{transizione}_t) + (\text{scrittura}_t)$;
  cambia solo la **transizione di stato**.
- Il percorso: dall'attenzione lineare (kernel e ricorrenza) → a come scrivere
  meglio nella memoria (gate e delta rule) → alle architetture concrete (RetNet,
  RWKV, xLSTM).
```

`````
