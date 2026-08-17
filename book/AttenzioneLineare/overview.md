# Attenzione lineare

Nel 2020, mentre il mondo dell'intelligenza artificiale celebrava i
Transformer come la rottura definitiva con il passato ricorrente, quattro
ricercatori fra la Svizzera e gli Stati Uniti (Katharopoulos, Vyas, Pappas e
Fleuret) pubblicano un articolo dal titolo che suona come una provocazione:
*Transformers are RNNs* {cite}`katharopoulos2020transformers`, dove RNN è la
sigla inglese delle reti ricorrenti, quelle che leggono una parola alla volta.
La tesi è tanto semplice quanto spiazzante. Togli al meccanismo di attenzione la
sua funzione softmax (il passaggio che, davanti a una parola, spartisce l'attenzione
fra tutte le altre come le fette di una torta), metti al suo posto un modo
più rozzo di misurare quanto due parole si somigliano, e il Transformer, il
modello che aveva appena spodestato le reti ricorrenti, ricade esattamente in
una **rete ricorrente**. Il re, sotto il mantello, era un vecchio parente.

Non è un gioco di prestigio: è una porta. Nel capitolo sui Transformer abbiamo
visto che l'attenzione si paga due volte. Il primo conto è il lavoro: ogni
parola guarda tutte le altre, quindi raddoppiando la lunghezza del testo il
lavoro **quadruplica**, ed è ciò che si chiama costo *quadratico*: dieci volte
il testo, cento volte il lavoro. Un numero per sentirne il peso: su centomila
parole, la lunghezza di un romanzo, sono centomila per centomila confronti da
fare, cioè dieci miliardi, e non una volta sola, ma in ognuno degli strati
della rete, che sono decine.

Il secondo conto si paga mentre il modello scrive. Per non rifare ogni volta
gli stessi calcoli, il modello tiene da parte un archivio di appunti, la
*cache*. Ogni pezzetto di testo che produce (si chiama **token**: quasi sempre
una parola o un frammento di parola, e qui per comodità li chiameremo parole)
vi lascia la propria **chiave**, cioè l'etichetta con cui lo si ritrova, e il
proprio **valore**, cioè l'informazione che porta. Etichetta e informazione non
le sceglie nessuno a mano: le calcola il modello dalla parola stessa, che al
suo interno è già diventata una fila di numeri. Quell'archivio **cresce**
parola dopo parola. È il muro contro cui sbattono i contesti lunghi.

La provocazione di Katharopoulos indica una via per aggirare tutti e due i
conti. Se l'attenzione, spogliata della softmax, è una rete ricorrente, allora
la si può riscrivere come una ricorrenza a **stato di dimensione fissa**: un
riassunto grande sempre uguale (lo chiameremo anche *stato*) che si aggiorna
una volta per parola. Il lavoro torna a crescere in modo *lineare*, cioè
semplicemente proporzionale (il doppio di testo, il doppio di lavoro), e la
memoria non cresce affatto. È da lì che viene il nome del capitolo: da
quadratico a lineare.

```{figure} ../figures/kv-cache-generazione.svg
:name: fig-kv-cache-cresce
:alt: "Quattro passi di generazione, uno sotto l'altro, sulla frase «Il gatto dorme sul». A ogni passo la fila dei riquadri si allunga di uno: il token appena prodotto è l'unico calcolato in quel passo, quelli di prima restano in cache e non si ricalcolano. A destra di ogni fila il conteggio delle coppie chiave-valore conservate, da una a quattro."
:width: 96%

La cache che non smette di crescere. Ogni token generato ne aggiunge un pezzo,
e quel pezzo resta: la memoria occupata cresce con quanto si è scritto finora,
e nessun passo la libera.
```

{numref}`fig-kv-cache-cresce` mostra il secondo dei due conti, quello che si
paga mentre il modello scrive: è il muro in una figura, ed è il motivo per cui
tutto questo capitolo esiste. Una ricorrenza a stato fisso non fa crescere
niente: comprime il passato in una memoria di taglia costante, e la domanda
diventa quanto si perde nel comprimerlo.

## Il compromesso che tutti inseguono

Conviene separare subito due momenti della vita di un modello, perché costano
in modo diverso ed è su quella differenza che gira l'intero capitolo.

`````{tab} Elementare

Il primo è l’**addestramento**, che
si fa una volta sola: il testo esiste già tutto, e il modello lo attraversa per
imparare. Il secondo è la **generazione**, quando il modello è in uso e scrive parola
per parola un testo che non esiste ancora (nel libro la trovi chiamata
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
*matrice*, e la parola tornerà spesso).

Che una tabella di numeri possa contenere delle parole suona strano, e vale la
pena chiarirlo qui perché regge tutto il resto: dentro un modello una parola
non è una parola, è una fila di qualche centinaio di numeri (le posizioni di
quella fila si chiamano *canali*). Etichetta e informazione sono due file di
numeri anche loro, ricavate dalla parola. Quindi «scrivere nel riassunto» vuol
dire sommare dei numeri alle caselle, e «rileggerlo» vuol dire rifare dei
conti.

Ogni parola che passa deposita così un'associazione, «a questa etichetta
corrisponde questa informazione», che si somma a quello che c'è già scritto
invece di aggiungere una riga nuova. Ecco perché la memoria non cresce: a
cambiare sono i numeri dentro le caselle, non il numero di caselle.

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
di un riassunto che sa soltanto sommare: le scritte si sovrappongono fin da
subito, e bastano poche informazioni perché non ci si legga più niente di
preciso.
Seconda tappa, **scrivere meglio nel riassunto**: i due rimedi a quel difetto,
lasciar sbiadire ciò che è vecchio e correggere ciò che è già scritto invece di
aggiungerci sopra, fino a una tabella che mostra come i modelli di questa
famiglia siano lo stesso meccanismo con una manopola girata in modo diverso.
Terza tappa, **le architetture concrete** che oggi si misurano con i
Transformer (RetNet, RWKV, xLSTM). Chiude un breve **notebook**, cioè una
pagina di codice che si può far girare, in cui verifichiamo con i numeri veri
che i due modi di fare il conto danno davvero lo stesso risultato.

`````{tab} Elementare

Non serve portarsi dietro niente: ogni parola nuova viene spiegata dove
compare, e le formule stanno tutte nell'altro livello (qui restano solo dei
conti con i numeri, tenuti il più semplici possibile). Se leggi solo questo, il
capitolo si legge di fila. Le figure del capitolo sono quattro e le loro
didascalie raccontano da sole la storia: se un passaggio si complica, guarda
la figura più vicina.

`````

`````{tab} Superiore

Con i nomi che si trovano nei paper: il *trucco del kernel* che spezza la
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
- Tolta la softmax (il passaggio che spartisce l'attenzione di una parola fra
  tutte le altre come le fette di una torta) e messo al suo posto un modo più
  rozzo di misurare quanto due parole si somigliano, il Transformer si riscopre
  una **rete ricorrente** {cite}`katharopoulos2020transformers`: legge una
  parola alla volta portandosi dietro un riassunto di dimensione sempre uguale.
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
