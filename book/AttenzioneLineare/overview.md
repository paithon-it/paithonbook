# Attenzione lineare

Nel 2020, mentre il mondo dell'intelligenza artificiale celebrava i
Transformer come la rottura definitiva con il passato ricorrente, quattro
ricercatori (Katharopoulos, Vyas, Pappas e Fleuret, tra l'Idiap, l'EPFL e la
University of Washington)
pubblicano un articolo dal titolo che suona come una provocazione:
*Transformers are RNNs* {cite}`katharopoulos2020transformers`. La tesi è tanto
semplice quanto spiazzante. Togliete al meccanismo di attenzione la sua
funzione softmax (il passaggio che, davanti a una parola, spartisce l'attenzione
fra tutte le altre come le fette di una torta, in modo che i pesi si sommino a
uno) e il Transformer, il modello che aveva appena spodestato le reti
ricorrenti, ricade esattamente in una **rete ricorrente**.
Il re, sotto il mantello, era un vecchio parente.

Non è un gioco di prestigio: è una porta. Nel capitolo sui Transformer abbiamo
visto che l'attenzione si paga due volte. Il primo conto è il lavoro: ogni
parola guarda tutte le altre, quindi raddoppiando la lunghezza del testo il
lavoro **quadruplica**, ed è ciò che si chiama costo *quadratico*. Il secondo
si paga mentre il modello scrive: per non rifare ogni volta gli stessi calcoli
tiene da parte un archivio di appunti (la *cache*) dove ogni token prodotto (il
token è il pezzetto di testo che il modello tratta come unità: quasi sempre una
parola o un frammento di parola, e qui per comodità li chiameremo parole)
lascia la propria **chiave**, cioè l'etichetta con cui lo si ritrova, e il
proprio **valore**, cioè l'informazione che porta. Quell'archivio **cresce**
parola dopo parola. È il muro contro cui sbattono i contesti lunghi. La
provocazione di Katharopoulos indica una via per aggirarlo: se l'attenzione,
spogliata della softmax, è una rete ricorrente, allora possiamo riscriverla
come una ricorrenza a **stato di dimensione fissa**, dove il lavoro cresce in
modo *lineare*, cioè semplicemente proporzionale (il doppio di testo, il doppio
di lavoro) e la memoria non cresce affatto, senza rinunciare del tutto a ciò
che l'aveva resa vincente. È da lì che viene il nome del capitolo: da
quadratico a lineare.

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

Un modello di questo tipo fa due mestieri, in due momenti diversi della sua
vita, e conviene tenerli separati fin da subito. C'è l'**addestramento**, che
si fa una volta sola: il testo esiste già tutto, e il modello lo attraversa per
imparare. E c'è la **generazione**, quando il modello è in uso e scrive parola
per parola un testo che non esiste ancora (nel libro la troverete chiamata
anche *inferenza*, che è il nome tecnico dello stesso momento). Le due cose
costano in modo diverso, e un modello può essere bravo in una e disastroso
nell'altra.

Le due grandi famiglie di modelli per sequenze hanno infatti un pregio e un
difetto speculari. I Transformer si addestrano in fretta, perché guardano
tutta la frase in una volta e sfruttano a pieno le schede grafiche; ma per
farlo devono tenere tutto sott'occhio, e più il testo è lungo più questo
costa: in fretta diventa insostenibile. Le vecchie reti ricorrenti fanno il
contrario: leggono una parola alla volta portandosi dietro un riassunto di
dimensione fissa, quindi quando generano costano poco e non si spaventano
davanti a testi lunghissimi; ma proprio perché procedono in fila, addestrarle
è lento.

Il sogno è avere le due cose insieme: la **velocità di addestramento** dei
Transformer e il **basso costo in generazione** delle reti ricorrenti. È
esattamente ciò che promette l'attenzione lineare, e con lei tutta la famiglia
di modelli di questo capitolo.

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
tutti questi modelli tengono un **riassunto di taglia fissa** e a ogni parola
lo riscrivono nello stesso modo, cioè *quel che resta del riassunto di prima,
più quel che si scrive adesso*. Ciò che distingue un modello dall'altro è,
essenzialmente, **come si aggiorna quel riassunto**: se ci si limita ad
accumulare, se si impara a dimenticare, se si corregge ciò che è già scritto.
Cambiare quella regola significa passare dall'attenzione lineare a RetNet, a
Mamba, a DeltaNet: nomi che incontreremo come varianti di uno stesso
scheletro, non come invenzioni scollegate.

`````{tab} Elementare

Il riassunto non è un testo: è una **tabella di numeri**, righe e colonne,
sempre della stessa taglia (in matematica una tabella così si chiama
*matrice*, e la parola tornerà spesso). Ogni parola che passa vi deposita
un'associazione, «a questa etichetta corrisponde questa informazione», che si
somma a quello che c'è già scritto invece di aggiungere una riga nuova. Ecco
perché la memoria non cresce: a cambiare sono i numeri dentro le caselle, non
il numero di caselle.

`````

`````{tab} Superiore

Sono **reti ricorrenti lineari** con uno stato di dimensione fissa, aggiornato
a ogni token da una ricorrenza della forma

$$
\mathbf{S}_t = \mathbf{S}_{t-1}\,(\text{transizione}_t) + (\text{scrittura}_t).
$$

Lo stato $\mathbf{S}_t$ è una piccola matrice (una memoria che associa chiavi a valori)
e il pedice $t$ è il passo, cioè il token appena letto: $\mathbf{S}_{t-1}$ è la memoria
al passo precedente, la *transizione* decide che cosa ne sopravvive, la
*scrittura* è ciò che il token corrente aggiunge. Cambiare il fattore di
transizione è, letteralmente, cambiare modello.

`````

## Come è organizzato il capitolo

Tre tappe, dal meccanismo alle architetture concrete.

Si parte da **come l'attenzione diventa economica**: che cosa bisogna cambiare
nel conto perché il lavoro smetta di esplodere, come quel conto si trasformi in
un riassunto di taglia fissa aggiornato parola per parola, e qual è il difetto
di un riassunto che sa soltanto sommare, cioè che prima o poi si satura.
Seconda tappa, **scrivere meglio nel riassunto**: i due rimedi a quel difetto,
lasciar sbiadire ciò che è vecchio e correggere ciò che è già scritto invece di
aggiungerci sopra, fino a una tabella che mostra come i modelli di questa
famiglia siano lo stesso meccanismo con una manopola girata in modo diverso.
Terza tappa, **le architetture concrete** che oggi si misurano con i
Transformer (RetNet, RWKV, xLSTM). Chiude un breve **notebook** in cui
verifichiamo con i numeri veri che i due modi di fare il conto danno davvero
lo stesso risultato.

`````{tab} Elementare

Non serve portarsi dietro niente: ogni parola nuova viene spiegata dove
compare, e le formule stanno tutte nell'altro livello. Se leggi solo questo, il
capitolo si legge di fila. Le figure del capitolo sono quattro e le loro
didascalie raccontano da sole la storia: se un passaggio si complica, guarda
la figura più vicina.

`````

`````{tab} Superiore

Con i nomi che troverete nei paper: il *trucco del kernel* che spezza la
softmax e trasforma l'attenzione in una ricorrenza a stato-matrice, con la sua
doppia natura parallelo/ricorrente, e il limite di capacità dell'accumulo puro;
poi i *gate* per dimenticare (RetNet, Mamba-2, GLA) e la *delta rule* per
correggere (DeltaNet, Gated DeltaNet), unificati dalla tabella finale come casi
di una stessa **regressione online**; infine RetNet, RWKV e xLSTM come istanze
dello stesso scheletro, e un notebook NumPy che verifica l'equivalenza fra
forma parallela e forma ricorrente.

`````

Si comincia dal problema, cioè dal punto in cui l'attenzione dei Transformer
smette di essere sostenibile, e dall'osservazione algebrica che permette di
aggirarlo.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Tolta la softmax, il passaggio che spartisce l'attenzione di una parola fra
  tutte le altre come le fette di una torta, il Transformer si riscopre una
  **rete ricorrente** {cite}`katharopoulos2020transformers`: legge una parola
  alla volta portandosi dietro un riassunto di dimensione sempre uguale.
- È la via per aggirare i due conti che i Transformer pagano sui testi lunghi:
  il lavoro che cresce a valanga con la lunghezza (raddoppiando il testo
  quadruplica) e la memoria di appoggio che si allunga a ogni parola generata.
  **Lineare** vuol dire proprio questo: il lavoro torna a essere semplicemente
  proporzionale, il doppio di testo per il doppio di lavoro, e la memoria non
  si allunga affatto.
- Il compromesso che tutta la famiglia insegue: la **velocità di addestramento**
  dei Transformer (quando il modello impara, e il testo c'è già tutto) *e* il
  **basso costo in generazione** delle vecchie reti ricorrenti (quando il
  modello scrive, una parola alla volta), perché lo stesso calcolo si può fare
  in due modi equivalenti, tutto insieme oppure una parola alla volta.
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
  $\mathbf{S}_t = \mathbf{S}_{t-1}\,(\text{transizione}_t) + (\text{scrittura}_t)$;
  cambia solo la **transizione di stato**.
- Il percorso: dall'attenzione lineare (kernel e ricorrenza) → a come scrivere
  meglio nella memoria (gate e delta rule) → alle architetture concrete (RetNet,
  RWKV, xLSTM).
```

`````
