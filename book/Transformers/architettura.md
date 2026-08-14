# La struttura del Transformer

Il meccanismo di attenzione è il motore; adesso montiamo l'automobile. Il
Transformer del paper originale è una macchina per tradurre: da un lato entra
una frase ("The cat jumps on the wall"), dall'altro esce la traduzione ("Il
gatto salta sul muro"). Per farlo combina due torri di blocchi identici
(l'**encoder** che legge, il **decoder** che scrive), più un ingrediente
facile da sottovalutare: un modo per dire alla rete *in che ordine* stanno le
parole.

```{figure} ../figures/architettura-transformer.svg
:name: fig-blocco-transformer
:alt: "Schema annotato di un blocco Transformer: l'ingresso attraversa la multi-head attention, si somma a sé stesso attraverso una connessione residua e passa per una normalizzazione; il risultato attraversa la rete feed-forward, con una seconda connessione residua e una seconda normalizzazione, prima di uscire verso il blocco successivo."
:width: 62%

Il blocco che si ripete, sempre uguale a sé stesso. I due mestieri sono la
riunione (l'attenzione, dove le parole si scambiano informazioni) e il lavoro
individuale (la **rete feed-forward**, dove ogni parola rielabora per conto
suo); attorno a entrambi c'è l'impalcatura della sezione precedente, cioè la
scorciatoia (**connessione residua**) e la taratura (**normalizzazione**), che
permette di impilarne decine senza che l'addestramento si rompa.
```

Conviene fissare {numref}`fig-blocco-transformer` prima di scendere nei
dettagli, perché tutto il capitolo gira attorno a questa figura: encoder e
decoder non sono due macchine diverse, sono due pile dello stesso blocco,
montate in modo leggermente diverso.

## L'encoder: la torre che legge

`````{tab} Elementare
L'encoder è una squadra di lettori disposti in colonna, nel modello originale
sei piani. Al primo piano la frase arriva "grezza"; ogni piano la rilegge con
il meccanismo di attenzione (ogni parola guarda tutte le altre) e poi ciascuna
parola viene rielaborata per conto suo da una piccola rete di neuroni, prima
di passare il risultato al piano di sopra. Piano dopo piano, la
rappresentazione di ogni parola si arricchisce: "nero" al sesto piano non è
più solo un colore, è *il colore di quel gatto in quella frase*. Alla fine
della salita, l'encoder consegna una versione della frase in cui ogni parola
porta scritto addosso il proprio contesto.
`````

`````{tab} Superiore
L'encoder è una pila di $L = 6$ strati identici (nel modello base,
$d_{\text{model}} = 512$; l'articolo del 2017 chiama $N$ questo numero, ma qui
si usa $L$ come nel resto del libro, dove $N$ serve ad altro), ciascuno con due
sotto-strati:

1. **Multi-Head Self-Attention**: ogni posizione attende a tutte le posizioni
   dell'input, catturando le relazioni a coppie in un solo passo;
2. **Feed-Forward Network (FFN)**: una rete completamente connessa applicata
   *indipendentemente e identicamente* a ogni posizione.

Ogni sotto-strato è avvolto da residual connection e layer normalization nella
forma Post-LN dell'articolo originale,
$\text{LayerNorm}(\mathbf{x} + \text{SubLayer}(\mathbf{x}))$, come visto nella
sezione
precedente (dove si è detto anche perché i modelli successivi preferiscono il
Pre-LN). Si noti la divisione dei ruoli: l'attenzione *mescola*
informazione tra le posizioni, la FFN la *trasforma* posizione per posizione;
è l'alternanza dei due movimenti, ripetuta per $L$ strati, a costruire
rappresentazioni via via più astratte.
`````

## Il decoder: la torre che scrive

`````{tab} Elementare
Il decoder genera la traduzione una parola alla volta, e mentre lo fa consulta
due fonti: quello che ha *già scritto* (per non contraddirsi) e quello che
l'encoder *ha letto* (per restare fedele all'originale). C'è però una regola
ferrea, la stessa dei compiti in classe: **non si sbircia avanti**. Quando il
decoder impara a produrre la quarta parola, può guardare solo le prime tre: se
durante l'addestramento potesse leggere la risposta intera, "imparerebbe" a
copiare, e al momento di generare davvero, senza soluzione da copiare, non
saprebbe fare nulla.
`````

`````{tab} Superiore
Anche il decoder ha $L = 6$ strati, ma con tre sotto-strati ciascuno:

1. **Masked Multi-Head Self-Attention**: come la self-attention dell'encoder,
   ma con una maschera che azzera (pone a $-\infty$ prima della softmax) le
   affinità verso le posizioni future: la posizione $t$ vede solo
   $1, \dots, t$. È ciò che rende il modello **autoregressivo** e coerente
   tra addestramento e generazione;
2. **Cross-Attention**: le query vengono dal decoder, key e value dall'output
   dell'encoder, è qui che la generazione "consulta" la frase di partenza;
3. **Feed-Forward Network**, identica a quella dell'encoder.

In generazione il decoder produce un token alla volta: a valle della pila, una
proiezione lineare sul vocabolario (nel paper con i pesi legati a quelli
dell'embedding, §3.4) e una softmax danno la distribuzione del token
successivo. Da quella distribuzione si sceglie **un** token, ed è il suo
embedding a rientrare come input al passo dopo: non la distribuzione, che è un
vettore di $|\mathcal{V}|$ probabilità e non ha modo di entrare in un ingresso
fatto per un token. Come si sceglie (il più probabile, uno estratto a sorte, o
il token
vero durante l'addestramento, che è il *teacher forcing*) è una questione a sé,
e la sezione sui grandi modelli linguistici la affronta per intero.
`````

```{figure} ../figures/attenzione-mascherata.gif
:name: fig-attenzione-mascherata
:alt: Animazione di una matrice di attenzione 6x6 sulla frase «Il gatto nero salta sul muro». Prima tutte le celle si riempiono di punteggi grigi; poi una scala separa il triangolo superiore, che si spegne perché posto a meno infinito; infine il triangolo inferiore si ricolora con i pesi normalizzati dalla softmax.
:width: 85%

La maschera causale al lavoro. Ogni riga della griglia è una parola che guarda
tutte le altre; le caselle verso il futuro si spengono, e ogni riga
ridistribuisce tutto il colore dell'evidenziatore su ciò che precede.
```

Il dettaglio da non perdere nella {numref}`fig-attenzione-mascherata` è che le
caselle si spengono **prima** che i punteggi diventino intensità, non dopo:
cancellare le intensità già calcolate lascerebbe righe che non sommano più a
uno, cioè evidenziature con un pezzo di colore mancante. Spegnere i punteggi a
monte (tecnicamente: ponendoli a $-\infty$ prima della softmax, cioè prima del
passaggio che trasforma i punteggi in intensità) li fa uscire come zeri esatti,
e la ridistribuzione resta corretta.

## Positional encoding: dare un ordine alle parole

C'è un problema nascosto. L'attenzione tratta la frase come un *sacchetto* di
parole: se mescolassi "il gatto morde il cane" in "il cane morde il gatto", i
confronti sarebbero gli stessi fra le stesse parole, quindi gli stessi
punteggi e la stessa evidenziatura. Per l'attenzione le due frasi sono
identiche; per chi legge sono opposte. Le RNN l'ordine ce l'avevano gratis
(leggevano in fila); il Transformer deve aggiungerlo esplicitamente.

```{figure} ../figures/positional-encoding.svg
:name: fig-positional-encoding
:alt: "Più onde sinusoidali sovrapposte, di frequenza decrescente: le prime oscillano rapidamente, le ultime lentamente. Letta in verticale a una data posizione, la combinazione dei valori delle diverse onde forma la firma numerica di quella posizione."
:width: 88%

Le frequenze del positional encoding. Nessuna onda da sola dice dove siamo;
lette insieme in una colonna danno a ogni posizione una firma diversa da tutte
le altre.
```

La firma della posizione c'è, ma non è scritta come un semplice contatore (1,
2, 3, …), ed è quello che mostra {numref}`fig-positional-encoding`: è fatta di
più orologi che girano a velocità diverse, come le lancette delle ore, dei
minuti e dei secondi. Le lancette lente dicono in quale parte della frase
siamo, quelle veloci distinguono i vicini immediati, e lette tutte insieme
danno a ogni posizione una combinazione che non si ripete. Un contatore solo
avrebbe una scala sola, e con una scala sola o si distinguono i lontani o si
distinguono i vicini, non entrambi.

`````{tab} Elementare
La soluzione è dare a ogni parola un "posto numerato", come a teatro: prima di
entrare nella rete, alla rappresentazione di ogni parola viene sommata una
piccola firma che dice "sono la parola in prima posizione", "in seconda", e
così via. Il numero del posto non è scritto in cifre, è scritto con le
lancette di cui parla la figura qui sopra, ma il senso è quello: "gatto" in
prima posizione e "gatto" in quinta non sono più identici. Stessa parola,
poltrona diversa, e la rete può accorgersi che l'ordine conta.
`````

`````{tab} Superiore
Il **positional encoding** del paper originale è deterministico, fatto di
sinusoidi a frequenze diverse:

$$
\text{PE}_{(\text{pos},\, 2i)} = \sin(\text{pos}\;\omega_i)
\qquad
\text{PE}_{(\text{pos},\, 2i+1)} = \cos(\text{pos}\;\omega_i) ,
\qquad
\omega_i = 10000^{-2i/d_{\text{model}}}
$$

dove $\text{pos}$ è la posizione del token e $i$ **non** indicizza le
coordinate ma le **coppie** di coordinate, cioè le frequenze: $i$ va da $0$ a
$d_{\text{model}}/2 - 1$, e ogni $i$ riempie le due coordinate $2i$ e $2i+1$
con un seno e un coseno della stessa frequenza $\omega_i$. È il punto in cui si
sbaglia scrivendo il codice, perché il paper scrive «$i$ is the dimension» e
lascia credere che arrivi a $d_{\text{model}}$.

Ogni posizione riceve così una firma unica, sommata all'embedding del token; e
la scelta sinusoidale fa sì che la firma della posizione $\text{pos} + k$ sia
una trasformazione lineare di quella di $\text{pos}$ **con una matrice che
dipende solo da $k$**. La clausola in grassetto è tutto il contenuto: che due
vettori siano legati da *qualche* matrice è vero sempre e non dice niente; che
la matrice sia la stessa per ogni $\text{pos}$ è ciò che rende la distanza
relativa una cosa rappresentabile. Ed è una riga di trigonometria, quindi vale
la pena scriverla: dalle formule di addizione,

$$
\begin{pmatrix} \sin((\text{pos}+k)\,\omega_i) \\ \cos((\text{pos}+k)\,\omega_i) \end{pmatrix}
=
\begin{pmatrix} \cos k\omega_i & \sin k\omega_i \\ -\sin k\omega_i & \cos k\omega_i \end{pmatrix}
\begin{pmatrix} \sin(\text{pos}\,\omega_i) \\ \cos(\text{pos}\,\omega_i) \end{pmatrix},
$$

cioè una rotazione di angolo $k\omega_i$ su ciascuna coppia, e $\text{pos}$ è
sparito dalla matrice. Da lì gli autori
*ipotizzarono* che la rete potesse rappresentare facilmente le distanze
*relative*: è una congettura, e va detto che la loro stessa ablazione la
indebolisce, perché con positional embedding **appresi** i risultati sono
«quasi identici» (Tabella 3, riga E, dell'articolo). Molti modelli successivi
usano invece encoding **appresi** (BERT) o codifiche relative più sofisticate
(RoPE nei modelli recenti): il principio (iniettare l'ordine, perché
l'attenzione da sola è permutation-invariant) resta lo stesso.
`````

## La feed-forward network: il lavoro individuale

`````{tab} Elementare
Dopo ogni "riunione" di attenzione (dove le parole si scambiano informazioni),
c'è un momento di lavoro individuale: ogni parola, per conto suo, rielabora
quello che ha appena sentito. È una piccola rete di neuroni come quelle del
capitolo sulle reti neurali, la stessa per tutte le parole, che prende la
rappresentazione di una parola e la trasforma in una versione più lavorata.
Riunione, lavoro individuale, riunione, lavoro individuale: la torre del
Transformer è tutta qui.

Una cosa sorprendente, che tornerà utile più avanti: è il momento di lavoro
individuale, non la riunione, a contenere quasi tutto quello che il modello ha
imparato. Contando i numeri che la rete regola mentre impara (si chiamano
**parametri**, ed è quello che si conta quando si dice «un modello da sette
miliardi»), due terzi buoni di ogni piano stanno nel lavoro individuale, e solo
un terzo nell'attenzione. La parte concettualmente più semplice è anche quella
dove il modello tiene la roba.
`````

`````{tab} Superiore
La FFN applica a ogni posizione, separatamente e con gli stessi pesi, due
trasformazioni lineari con una ReLU in mezzo:

$$
\text{FFN}(\mathbf{x}) = \max(0,\, \mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\,
\mathbf{W}_2 + \mathbf{b}_2
$$

Nel modello base la dimensione interna è $d_{\text{ff}} = 2048$, quattro volte
$d_{\text{model}} = 512$: la FFN espande, applica la non linearità,
ricomprime. Pur essendo la parte concettualmente più semplice, contiene circa
due terzi dei parametri di uno strato di **encoder**
($2\,d\,d_{\text{ff}} = 8d^2$ contro i $4d^2$ delle quattro proiezioni
dell'attenzione); negli strati di
decoder, che hanno una seconda attenzione, la quota scende a metà. Nei grandi
modelli linguistici, che sono decoder-only e quindi senza cross-attention, si
torna ai due terzi, ed è lì che risiede gran parte della capacità. Una linea di
ricerca interpreta la FFN come una memoria associativa di conoscenze apprese
durante l'addestramento.

Questa è però la FFN del **paper originale**. I modelli successivi ne hanno
cambiato la non linearità: prima la **GELU**, una ReLU ammorbidita (BERT,
GPT-2), poi le varianti *gated* e in particolare **SwiGLU**, che è la scelta
prevalente nei modelli recenti:

$$
\text{FFN}_{\text{SwiGLU}}(\mathbf{x}) =
\big(\mathrm{Swish}(\mathbf{x}\mathbf{W}_1) \odot \mathbf{x}\mathbf{W}_3\big)\,
\mathbf{W}_2 ,
\qquad \mathrm{Swish}(z) = z\,\sigma(z).
$$

dove $\mathbf{W}_1, \mathbf{W}_3 \in \mathbb{R}^{d \times d_{\text{ff}}}$ e
$\mathbf{W}_2 \in \mathbb{R}^{d_{\text{ff}} \times d}$, e $\odot$ è il prodotto
elemento per elemento. (Shazeer chiama $\mathbf{V}$ la matrice del cancello;
qui la lettera è già impegnata dai *value* dell'attenzione, e riusarla sarebbe
una trappola.)

Il ramo $\mathbf{x}\mathbf{W}_3$ fa da **cancello**: moltiplicando elemento per
elemento, decide
quanto lasciar passare di ciascuna unità del ramo principale. Le matrici
diventano tre invece di due, e per non gonfiare il conteggio dei parametri si
riduce la dimensione interna da $4d$ a circa $\tfrac{8}{3}d$: così
$3 \cdot d \cdot \tfrac{8}{3}d = 8d^2$, esattamente quanto $2 \cdot d \cdot 4d$
della versione classica. Stessi parametri, risultati migliori a parità di
addestramento.
`````

Con la feed-forward il giro è completo, e vale la pena guardare indietro un
momento. In questa pagina non è comparso nessun ingrediente che non fosse già
sul tavolo: c'è l'attenzione della sezione precedente, c'è una piccola rete di
neuroni come quelle del capitolo sulle reti neurali, ci sono una scorciatoia e
una taratura attorno a ciascuna delle due. Il Transformer non è un pezzo nuovo,
è un modo di impilare quei quattro, sempre nello stesso ordine, per sei piani
e poi per sessanta. È il motivo per cui l'architettura ha retto senza cambiare
forma ingrandimenti di tre ordini di grandezza (dai 65 milioni di parametri del
modello base del 2017 ai 175 miliardi di GPT-3, tre anni dopo): non c'era una
forma da cambiare, c'era una sequenza da ripetere.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Il Transformer originale è fatto di **due torri**: una legge la frase di
  partenza, l'altra scrive la traduzione. Sei piani ciascuna, tutti uguali.
- Ogni piano alterna una **riunione** (con l'attenzione, ogni parola ascolta
  tutte le altre; a condurla sono otto lettori in parallelo, ognuno attento a
  un tipo di legame) e un **lavoro individuale** (ogni parola rielabora per
  conto suo quello che ha sentito). Attorno a entrambi i momenti c'è
  l'impalcatura che permette di impilare tanti piani senza che
  l'addestramento si rompa.
- La torre che scrive ha una regola ferrea, **non si sbircia avanti**: mentre
  produce la quarta parola può guardare solo le prime tre. E a ogni passo
  consulta quello che l'altra torre ha capito della frase originale.
- L'attenzione da sola non sa in che ordine stanno le parole: a ciascuna viene
  sommato prima un **posto numerato**, la firma della sua posizione (calcolata
  a tavolino nel paper del 2017; i modelli successivi la fanno in altri modi,
  ma il posto numerato serve a tutti).
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Il Transformer originale è **encoder–decoder**: $L = 6$ strati per torre,
  $d_{\text{model}} = 512$, 8 teste di attenzione.
- Ogni strato alterna **attenzione** (le posizioni si scambiano informazione)
  e **FFN** (ogni posizione rielabora per conto suo), con residual e layer
  norm attorno a ogni sotto-strato.
- Il decoder usa la **maschera causale** (vietato guardare il futuro) e la
  **cross-attention** verso l'encoder.
- L'attenzione ignora l'ordine: il **positional encoding** (sinusoidale nel
  paper, appreso o relativo nei modelli successivi) lo reintroduce.
```
`````
