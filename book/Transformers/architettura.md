# La struttura del Transformer

Il meccanismo di attenzione è il motore; adesso montiamo l'automobile. Il
Transformer del paper originale è una macchina per tradurre: da un lato entra
una frase ("The cat jumps on the wall"), dall'altro esce la traduzione ("Il
gatto salta sul muro"). Per farlo combina due torri di blocchi identici —
l'**encoder** che legge, il **decoder** che scrive — più un ingrediente
facile da sottovalutare: un modo per dire alla rete *in che ordine* stanno le
parole.

## L'encoder: la torre che legge

`````{tab} Elementare
L'encoder è una squadra di lettori disposti in colonna, nel modello originale
sei piani. Al primo piano la frase arriva "grezza"; ogni piano la rilegge con
il meccanismo di attenzione — ogni parola guarda tutte le altre — e poi
ciascuna parola viene rielaborata per conto suo da una piccola rete di
neuroni, prima di passare il risultato al piano di sopra. Piano dopo piano,
la rappresentazione di ogni parola si arricchisce: "nero" al sesto piano non
è più solo un colore, è *il colore di quel gatto in quella frase*. Alla fine
della salita, l'encoder consegna una versione della frase in cui ogni parola
porta scritto addosso il proprio contesto.
`````

`````{tab} Superiore
L'encoder è una pila di $N = 6$ strati identici (nel modello base,
$d_{\text{model}} = 512$), ciascuno con due sotto-strati:

1. **Multi-Head Self-Attention**: ogni posizione attende a tutte le posizioni
   dell'input, catturando le relazioni a coppie in un solo passo;
2. **Feed-Forward Network (FFN)**: una rete completamente connessa applicata
   *indipendentemente e identicamente* a ogni posizione.

Ogni sotto-strato è avvolto da residual connection e layer normalization,
$\text{LayerNorm}(x + \text{SubLayer}(x))$, come visto nella sezione
precedente. Si noti la divisione dei ruoli: l'attenzione *mescola*
informazione tra le posizioni, la FFN la *trasforma* posizione per posizione;
è l'alternanza dei due movimenti, ripetuta per $N$ strati, a costruire
rappresentazioni via via più astratte.
`````

## Il decoder: la torre che scrive

`````{tab} Elementare
Il decoder genera la traduzione una parola alla volta, e mentre lo fa
consulta due fonti: quello che ha *già scritto* (per non contraddirsi) e
quello che l'encoder *ha letto* (per restare fedele all'originale). C'è però
una regola ferrea, la stessa dei compiti in classe: **non si sbircia
avanti**. Quando il decoder impara a produrre la quarta parola, può guardare
solo le prime tre — se durante l'addestramento potesse leggere la risposta
intera, "imparerebbe" a copiare, e al momento di generare davvero, senza
soluzione da copiare, non saprebbe fare nulla.
`````

`````{tab} Superiore
Anche il decoder ha $N = 6$ strati, ma con tre sotto-strati ciascuno:

1. **Masked Multi-Head Self-Attention**: come la self-attention dell'encoder,
   ma con una maschera che azzera (pone a $-\infty$ prima della softmax) le
   affinità verso le posizioni future: la posizione $t$ vede solo
   $1, \dots, t$. È ciò che rende il modello **autoregressivo** e coerente
   tra addestramento e generazione;
2. **Cross-Attention**: le query vengono dal decoder, key e value
   dall'output dell'encoder — è qui che la generazione "consulta" la frase
   di partenza;
3. **Feed-Forward Network**, identica a quella dell'encoder.

In generazione il decoder produce un token alla volta: a ogni passo
l'ultimo strato proietta sulla dimensione del vocabolario e una softmax dà la
distribuzione del token successivo, che rientra come input al passo dopo.
`````

```{figure} ../figures/attenzione-mascherata.gif
:name: fig-attenzione-mascherata
:alt: Animazione di una matrice di attenzione 6x6 sulla frase «Il gatto nero salta sul muro». Prima tutte le celle si riempiono di punteggi grigi; poi una scala separa il triangolo superiore, che si spegne perché posto a meno infinito; infine il triangolo inferiore si ricolora con i pesi normalizzati dalla softmax.
:width: 85%

La maschera causale al lavoro sulla matrice $QK^\top/\sqrt{d_k}$: i punteggi
verso il futuro vengono posti a $-\infty$, così la softmax li annulla e ogni
riga ridistribuisce tutto il peso su ciò che precede.
```

Il dettaglio da non perdere nella {numref}`fig-attenzione-mascherata` è che la
maschera agisce **prima** della softmax, non dopo: azzerare i pesi a valle
lascerebbe righe che non sommano a uno, mentre un $-\infty$ a monte esce dalla
softmax come uno zero esatto e la normalizzazione resta corretta.

## Positional encoding: dare un ordine alle parole

C'è un problema nascosto. L'attenzione tratta la frase come un *insieme* di
parole: se mescolassi "il gatto morde il cane" in "il cane morde il gatto",
i prodotti scalari tra le parole non cambierebbero — e il significato sì. Le
RNN l'ordine ce l'avevano gratis (leggevano in fila); il Transformer deve
aggiungerlo esplicitamente.

`````{tab} Elementare
La soluzione è dare a ogni parola un "posto numerato", come a teatro: prima
di entrare nella rete, alla rappresentazione di ogni parola viene sommata una
piccola firma numerica che dice "sono la parola in posizione 1", "in
posizione 2", e così via. Così "gatto" in prima posizione e "gatto" in quinta
posizione non sono più identici: stessa parola, poltrona diversa — e la rete
può accorgersi che l'ordine conta.
`````

`````{tab} Superiore
Il **positional encoding** del paper originale è deterministico, fatto di
sinusoidi a frequenze diverse:

$$
PE_{(pos,\, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
\qquad
PE_{(pos,\, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

dove $pos$ è la posizione del token e $i$ indicizza le coordinate del
vettore. Ogni posizione riceve così una firma unica, sommata all'embedding
del token; la scelta sinusoidale fa sì che la firma della posizione
$pos + k$ sia una trasformazione lineare di quella di $pos$, il che permette
alla rete di rappresentare facilmente le distanze *relative*. Molti modelli
successivi usano invece encoding **appresi** (BERT) o codifiche relative
più sofisticate (RoPE nei modelli recenti): il principio — iniettare
l'ordine, perché l'attenzione da sola è permutation-invariant — resta lo
stesso.
`````

## La feed-forward network: il lavoro individuale

`````{tab} Elementare
Dopo ogni "riunione" di attenzione — dove le parole si scambiano informazioni
— c'è un momento di lavoro individuale: ogni parola, per conto suo, rielabora
quello che ha appena sentito. È una piccola rete di neuroni come quelle del
capitolo sulle reti neurali, la stessa per tutte le parole, che prende la
rappresentazione di una parola e la trasforma in una versione più lavorata.
Riunione, lavoro individuale, riunione, lavoro individuale: la torre del
Transformer è tutta qui.
`````

`````{tab} Superiore
La FFN applica a ogni posizione, separatamente e con gli stessi pesi, due
trasformazioni lineari con una ReLU in mezzo:

$$
\text{FFN}(x) = \max(0,\, xW_1 + b_1)\,W_2 + b_2
$$

Nel modello base la dimensione interna è $d_{ff} = 2048$, quattro volte
$d_{\text{model}} = 512$: la FFN espande, applica la non linearità,
ricomprime. Pur essendo la parte concettualmente più semplice, contiene circa
due terzi dei parametri di ogni strato — nei grandi modelli linguistici è
dove risiede gran parte della capacità — e una linea di ricerca la interpreta
come una memoria associativa di conoscenze apprese durante l'addestramento.

Questa è però la FFN del **paper originale**. I modelli successivi ne hanno
cambiato la non linearità: prima la **GELU**, una ReLU ammorbidita (BERT,
GPT-2), poi le varianti *gated* e in particolare **SwiGLU**, oggi lo standard
di fatto:

$$
\text{FFN}_{\text{SwiGLU}}(x) = \big(\mathrm{Swish}(xW_1) \odot xV\big)\,W_2 ,
\qquad \mathrm{Swish}(z) = z\,\sigma(z).
$$

Il ramo $xV$ fa da **cancello**: moltiplicando elemento per elemento, decide
quanto lasciar passare di ciascuna unità del ramo principale. Le matrici
diventano tre invece di due, e per non gonfiare il conteggio dei parametri si
riduce la dimensione interna da $4d$ a circa $\tfrac{8}{3}d$: così
$3 \cdot d \cdot \tfrac{8}{3}d = 8d^2$, esattamente quanto $2 \cdot d \cdot 4d$
della versione classica. Stessi parametri, risultati migliori a parità di
addestramento.
`````

```{admonition} Da ricordare
:class: important
- Il Transformer originale è **encoder–decoder**: $N = 6$ strati per torre,
  $d_{\text{model}} = 512$, 8 teste di attenzione.
- Ogni strato alterna **attenzione** (le posizioni si scambiano informazione)
  e **FFN** (ogni posizione rielabora per conto suo), con residual e layer
  norm attorno a ogni sotto-strato.
- Il decoder usa la **maschera causale** — vietato guardare il futuro — e la
  **cross-attention** verso l'encoder.
- L'attenzione ignora l'ordine: il **positional encoding** (sinusoidale nel
  paper, appreso o relativo nei modelli successivi) lo reintroduce.
```
