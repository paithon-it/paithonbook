# Il meccanismo di attenzione

Quando leggi la frase "il gatto, che aveva dormito tutto il giorno sul
davanzale, saltò", e arrivi a "saltò", il tuo cervello non ripassa tutte le
parole in fila: torna dritto a "gatto". Sai *a che cosa prestare attenzione*.
Il meccanismo di **attenzione** dà alle reti neurali esattamente questa
capacità: davanti a una parola, guardare tutte le altre e pesare quanto
ciascuna conta per capirla. Nato come rattoppo per migliorare le traduzioni
delle reti ricorrenti, con il Transformer è passato da comprimario a
protagonista assoluto.

## L'idea: pesare le parole

Per ogni parola da elaborare, l'attenzione produce una versione "arricchita
dal contesto": una media pesata delle informazioni di tutte le parole della
frase, dove i pesi dicono quanto ognuna è rilevante.

```{figure} ../figures/seq2seq-collo-di-bottiglia.svg
:name: fig-collo-di-bottiglia
:alt: "Schema di un seq2seq senza attenzione: le parole della frase in ingresso entrano una alla volta nell'encoder e vengono compresse in un unico vettore di contesto, disegnato come una strozzatura; da quel solo vettore il decoder deve generare tutta la traduzione, parola dopo parola."
:width: 92%

Il collo di bottiglia che l'attenzione viene a sciogliere. Tutta la frase
d'origine deve passare per un vettore solo, e più la frase è lunga più quel
vettore deve dimenticare.
```

{numref}`fig-collo-di-bottiglia` è il problema da cui nasce tutto. Se il
decoder può guardare solo un riassunto, la prima parola della frase e
l'ultima competono per lo stesso spazio; l'attenzione toglie la strozzatura
lasciando che ogni passo della generazione vada a rileggersi *tutte* le
parole d'origine, pesandole di volta in volta.

`````{tab} Elementare
Prendi la frase del libro: "Il gatto nero salta sul muro". Il modello sta
elaborando la parola "salta" e si chiede: chi salta? Come un lettore con
l'evidenziatore, ripassa la frase e assegna a ogni parola un'intensità di
colore: "gatto" fluorescente (è il soggetto!), "muro" un colore medio (è la
destinazione), "il" e "sul" quasi trasparenti. Poi costruisce il significato
di "salta" *in questa frase* mescolando le informazioni di tutte le parole,
ma in proporzione all'evidenziatura: tanta parte di "gatto", un po' di
"muro", pochissimo del resto.

I numeri dell'evidenziatore non li decide un programmatore: li impara la rete
durante l'addestramento, esattamente come impara ogni altro peso. E quando
questo gioco lo fa ogni parola verso tutte le altre (non solo "salta"), si
parla di **self-attention**, attenzione della frase su sé stessa.
`````

`````{tab} Superiore
Ogni parola (più precisamente ogni *token*, come vedremo) è rappresentata da
un vettore. Da ciascun vettore la rete ricava tre proiezioni con matrici
apprese: una **query** $Q$ ("che cosa sto cercando?"), una **key** $K$ ("che
cosa offro come etichetta?") e un **value** $V$ ("che informazione porto?").
L'affinità tra la parola che elabora e ogni altra è il prodotto scalare
query·key (la stessa misura di somiglianza tra vettori del capitolo di algebra
lineare) e la **Scaled Dot-Product Attention** la trasforma in pesi:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

dove $Q, K, V$ raccolgono per righe le proiezioni di tutti i token e $d_k$ è
la dimensione delle key. La softmax (già incontrata nel capitolo sulle reti
neurali) normalizza le affinità in pesi che sommano a 1; la divisione per
$\sqrt{d_k}$ evita che, al crescere della dimensione, i prodotti scalari
diventino così grandi da saturare la softmax e azzerarne i gradienti. L'output
è, per ogni token, la combinazione dei value pesata dall'attenzione: una
rappresentazione contestuale calcolata in un unico prodotto tra matrici, per
tutte le posizioni insieme.
`````

```{figure} ../figures/attention-is-all-you-need.svg
:name: fig-qkv
:alt: "Un token in ingresso viene proiettato in tre vettori distinti: Query, Key e Value. Il prodotto scalare fra la Query e le Key di tutti i token produce i punteggi di rilevanza, che una softmax trasforma in pesi; i pesi moltiplicano i rispettivi Value e la loro somma è l'uscita per quel token."
:width: 92%

I tre ruoli di ogni parola. La Query è la domanda che pone, la Key l'etichetta
con cui si fa trovare, il Value ciò che offre a chi la seleziona: la stessa
parola li ricopre tutti e tre insieme.
```

La separazione dei tre ruoli in {numref}`fig-qkv` è ciò che rende
l'attenzione più di una semplice somiglianza. Se ci fosse un solo vettore per
parola, «cercare» ed «essere trovati» sarebbero la stessa operazione; con
Query e Key distinte, una parola può cercare qualcosa di molto diverso da ciò
che offre.

## Multi-Head Attention: più letture in parallelo

Una sola "passata di evidenziatore" costringe la rete a comprimere in un
unico schema tutti i tipi di relazione tra parole. La soluzione del
Transformer è farne parecchie in parallelo.

`````{tab} Elementare
Immagina più lettori della stessa frase, ognuno con un evidenziatore di
colore diverso e una fissazione diversa: uno segna i rapporti grammaticali
(chi fa l'azione?), un altro le vicinanze di significato (nero → colore →
gatto), un altro ancora i legami di posizione. Alla fine i fogli evidenziati
si sovrappongono, e la frase risulta letta da più punti di vista
contemporaneamente. Ogni lettore è una "testa" di attenzione; il Transformer
originale ne usa otto.
`````

`````{tab} Superiore
La **Multi-Head Attention** esegue $h$ attenzioni indipendenti in sottospazi
distinti e ne ricompone gli esiti:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\,W^O
$$

dove $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$ e
$W_i^Q, W_i^K, W_i^V, W^O$ sono matrici apprese. Nel Transformer originale
$h = 8$ e ogni testa lavora in dimensione $d_k = d_{\text{model}}/h = 64$: il
costo complessivo resta paragonabile a una singola attenzione a dimensione
piena, ma il modello può dedicare teste diverse a relazioni diverse
(sintattiche, semantiche, posizionali) (cosa che l'analisi empirica delle
teste addestrate conferma almeno in parte).
`````

## Dove va a finire l'attenzione: encoder e decoder

Il blocco di attenzione non vive da solo: è il cuore di due componenti che il
prossimo capitolo smonta pezzo per pezzo. L'**encoder** legge la frase di
partenza e ne costruisce una rappresentazione ricca; il **decoder** la usa per
generare l'uscita (una traduzione, una risposta) un pezzo alla volta. In
mezzo, ancora attenzione: mentre genera, il decoder "evidenzia" le parti
rilevanti di ciò che l'encoder ha letto.

Ogni blocco è completato da due accorgimenti che rendono addestrabili anche
reti molto profonde.

`````{tab} Elementare
Il primo è una **scorciatoia**: l'informazione che entra in un blocco viene
anche fatta passare *intatta* accanto al blocco, e sommata all'uscita. Come un
corrimano lungo una scala ripida: anche se un gradino è scivoloso,
l'informazione (e la correzione degli errori durante l'apprendimento) ha
sempre una presa solida per risalire. Il secondo è una **taratura**: dopo ogni
blocco, i numeri vengono riportati su una scala standard, come rimettere a
zero la bilancia tra una pesata e l'altra, così nessuno strato lavora con
valori fuori misura.
`````

`````{tab} Superiore
Sono le **residual connection** e la **layer normalization**, combinate in

$$
\text{LayerNorm}\big(x + \text{SubLayer}(x)\big)
$$

attorno a ogni sotto-strato (attenzione o feed-forward). La connessione
residuale (la stessa idea delle ResNet che abbiamo visto nel capitolo sul deep
learning) offre al gradiente un cammino quasi diretto verso gli strati
iniziali, contrastando il gradiente che svanisce; la layer normalization
stabilizza media e varianza delle attivazioni a ogni posizione, rendendo
l'addestramento meno sensibile a learning rate e inizializzazione. «Quasi»,
perché in questa formulazione (detta *Post-LN*, quella del 2017) la
normalizzazione sta proprio sul ramo della scorciatoia, e il gradiente la
attraversa a ogni strato: i modelli successivi la spostano prima del
sotto-strato, $x + \text{SubLayer}(\text{LayerNorm}(x))$, il cosiddetto
*Pre-LN*, ed è lì che il cammino identità diventa davvero pulito (Xiong e
colleghi, 2020, mostrano che senza questo spostamento serve un riscaldamento
graduale del learning rate per addestrare stabilmente).
`````

```{admonition} Da ricordare
:class: important
- L'**attenzione** costruisce, per ogni parola, una rappresentazione
  contestuale: media dei *value* pesata dalle affinità *query*·*key*,
  normalizzate con softmax e scalate di $\sqrt{d_k}$.
- Nella **self-attention** ogni parola guarda tutte le altre; i pesi non sono
  fissati a mano ma appresi.
- La **Multi-Head Attention** esegue più attenzioni in parallelo ($h = 8$ nel
  modello originale), ciascuna libera di specializzarsi su relazioni diverse.
- **Residual connection** e **layer normalization** tengono addestrabili le
  pile profonde di blocchi. L'articolo del 2017 le combina come
  $\text{LayerNorm}(x + \text{SubLayer}(x))$ (*Post-LN*); i modelli successivi
  normalizzano prima del sotto-strato,
  $x + \text{SubLayer}(\text{LayerNorm}(x))$ (*Pre-LN*), ed è così che la
  scorciatoia resta davvero libera.
```
