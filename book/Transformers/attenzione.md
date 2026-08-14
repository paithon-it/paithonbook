# Il meccanismo di attenzione

Quando leggi la frase "il gatto, che aveva dormito tutto il giorno sul
davanzale, saltò", e arrivi a "saltò", il tuo cervello non ripassa tutte le
parole in fila: torna dritto a "gatto". Sai *a che cosa prestare attenzione*.
Il meccanismo di **attenzione** dà alle reti neurali esattamente questa
capacità: davanti a una parola, guardare tutte le altre e pesare quanto
ciascuna conta per capirla. Nato nel settembre del 2014 come rattoppo per
migliorare le traduzioni delle reti ricorrenti {cite}`bahdanau2015neural`, con
il Transformer è passato da comprimario a protagonista assoluto.

## L'idea: pesare le parole

Per ogni parola da elaborare, l'attenzione produce una versione "arricchita
dal contesto": guarda tutte le altre parole della frase, decide quanto ciascuna
conta per capire quella, e mescola le informazioni in quella proporzione. È una
media, come la media dei voti, con una differenza: i voti non pesano tutti
uguale, e a decidere quanto pesano è la frase stessa.

```{figure} ../figures/seq2seq-collo-di-bottiglia.svg
:name: fig-collo-di-bottiglia
:alt: "Schema di un seq2seq senza attenzione: le parole della frase in ingresso entrano una alla volta nell'encoder e vengono compresse in un unico vettore di contesto, disegnato come una strozzatura; da quel solo vettore il decoder deve generare tutta la traduzione, parola dopo parola."
:width: 92%

Il collo di bottiglia che l'attenzione viene a sciogliere. Tutta la frase
d'origine deve passare per un'unica lista di numeri, sempre lunga uguale: più
la frase è lunga, più quella lista è costretta a dimenticare.
```

{numref}`fig-collo-di-bottiglia` è il problema da cui nasce tutto. Nel gergo
del libro quella lista di numeri si chiama **vettore**, e le due metà del
sistema hanno un nome anche loro: l'**encoder** è la parte che legge la frase
di partenza, il **decoder** quella che scrive la frase d'arrivo (li rivediamo
per bene in fondo a questa pagina). Se il decoder può guardare solo un
riassunto, la prima parola della frase e
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
durante l'addestramento, cioè provando e correggendosi su miliardi di frasi.
Ogni volta che sbaglia a indovinare la parola che viene dopo, i numeri vengono
ritoccati un pochino nella direzione che avrebbe fatto sbagliare di meno: è lo
stesso meccanismo del capitolo sulle reti neurali. E quando
questo gioco lo fa ogni parola verso tutte le altre (non solo "salta"), si
parla di **self-attention**, attenzione della frase su sé stessa.
`````

`````{tab} Superiore
Ogni parola (più precisamente ogni *token*, come vedremo) è rappresentata da
un vettore. Da ciascun vettore la rete ricava tre proiezioni con matrici
apprese: una **query** $\mathbf{Q}$ ("che cosa sto cercando?"), una **key**
$\mathbf{K}$ ("che cosa offro come etichetta?") e un **value** $\mathbf{V}$
("che informazione porto?").
L'affinità tra la parola che elabora e ogni altra è il prodotto scalare
query·key (la stessa misura di somiglianza tra vettori vista in *Algebra
lineare*) e la **Scaled Dot-Product Attention** la trasforma in pesi:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) =
\text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}
$$

dove $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ raccolgono per righe le proiezioni
di tutti i token e $d_k$ è
la dimensione delle key. La softmax (già incontrata nel capitolo sulle reti
neurali) normalizza le affinità in pesi che sommano a 1; la divisione per
$\sqrt{d_k}$ evita che, al crescere della dimensione, i prodotti scalari
diventino così grandi da saturare la softmax e azzerarne i gradienti. Il conto
sta in due righe, e vale la pena farle perché è lì che stanno le ipotesi.
Supponiamo che le componenti $q_i$ e $k_i$ siano a media nulla, varianza
unitaria, indipendenti fra loro e indipendenti al variare di $i$. Allora ogni
addendo del prodotto scalare ha varianza
$\mathbb{E}[q_i^2 k_i^2] - (\mathbb{E}[q_i]\,\mathbb{E}[k_i])^2 =
\mathbb{E}[q_i^2]\,\mathbb{E}[k_i^2] = 1$, dove la fattorizzazione
dell'aspettazione usa l'indipendenza **fra** $q_i$ e $k_i$; e siccome gli
addendi sono scorrelati **al variare di $i$**, le varianze si sommano:

$$
\operatorname{Var}\!\left(\sum_{i=1}^{d_k} q_i k_i\right) = d_k .
$$

Dividere per $\sqrt{d_k}$ la riporta a 1. Servono dunque due indipendenze
diverse, una per ciascun passaggio, e una terza ipotesi che di solito non si
dice: tutto questo vale **all'inizializzazione**, perché appena
$\mathbf{W}^Q$ e $\mathbf{W}^K$ cominciano ad allenarsi smettono di produrre
componenti a varianza unitaria. Il fattore infatti non *impedisce* la
saturazione, ne toglie la dipendenza da $d_k$: con componenti di varianza
diversa da 1 la softmax satura lo stesso, a qualunque dimensione. L'output
è, per ogni token, la combinazione dei value pesata dall'attenzione: una
rappresentazione contestuale calcolata in un unico prodotto tra matrici, per
tutte le posizioni insieme.
`````

Resta da dire *come* si decide l'intensità dell'evidenziatore, ed è qui che
compaiono tre parole che poi tornano in tutto il capitolo. Ogni parola, per
partecipare al gioco, si presenta in tre versioni diverse di sé stessa, che la
rete si costruisce da sola: una **query**, cioè la domanda che quella parola
sta facendo alle altre («chi è che salta?»); una **key**, cioè l'etichetta con
cui quella stessa parola si fa trovare da chi la cerca («io sono un soggetto,
sono un animale»); e un **value**, cioè l'informazione che consegna a chi la
seleziona. Il confronto fra la query di una parola e la key di un'altra dà il
punteggio, i punteggi diventano le intensità dell'evidenziatore, e le
informazioni (i value) si mescolano in quelle proporzioni. In italiano
sarebbero *domanda*, *etichetta* e *contenuto*, ma i nomi inglesi sono ormai
quelli che si trovano scritti ovunque, e li useremo anche noi.

```{figure} ../figures/attention-is-all-you-need.svg
:name: fig-qkv
:alt: "Un token in ingresso viene proiettato in tre vettori distinti: Query, Key e Value. Il prodotto scalare fra la Query e le Key di tutti i token produce i punteggi di rilevanza, che una softmax trasforma in pesi; i pesi moltiplicano i rispettivi Value e la loro somma è l'uscita per quel token."
:width: 92%

I tre ruoli di ogni parola. La Query è la domanda che pone, la Key l'etichetta
con cui si fa trovare, il Value ciò che offre a chi la seleziona: la stessa
parola li ricopre tutti e tre insieme.
```

La separazione dei tre ruoli in {numref}`fig-qkv` è ciò che rende
l'attenzione più di una semplice somiglianza. Se ogni parola avesse una sola
versione di sé, «cercare» ed «essere trovati» sarebbero la stessa operazione;
con query e key distinte, una parola può cercare qualcosa di molto diverso da
ciò che offre.

```{admonition} Un cantiere parallelo: le reti a memoria
:class: note
Interrogare un archivio con una domanda, pesare quanto ciascun elemento le
risponde, e restituire la miscela pesata di ciò che quegli elementi contengono:
questa struttura è stata costruita prima dei Transformer, e per un altro scopo.

Nel 2014 le **memory network** {cite}`weston2015memory` affrontavano il
problema di far ragionare una rete su un elenco di fatti («Maria è andata in
cucina. Giovanni ha preso il latte. Dov'è il latte?»). La rete teneva i fatti
in un archivio a parte, separato dai numeri che aveva imparato, e per
rispondere andava a pescarci dentro. In quella prima versione però la pesca era
secca (si sceglieva *un* fatto, il più somigliante) e per addestrarla bisognava
dire alla rete, esempio per esempio, quali fossero i fatti giusti da usare.

Il passo che ci interessa arriva l'anno dopo, con le *end-to-end memory
network* {cite}`sukhbaatar2015end`: al posto della scelta secca si mette una
graduatoria: la domanda viene confrontata con **tutti** i fatti, il confronto
produce un'intensità di evidenziatore per ciascuno, e l'archivio viene letto
mescolando i fatti in quelle proporzioni. Poi si ripete, usando il risultato
come nuova domanda: erano i *hop*, cioè i salti di ragionamento, che
permettevano di concatenare due fatti per rispondere a una domanda che nessuno
dei due risolveva da solo. E siccome adesso ogni fatto contribuisce un po', la
rete può imparare da sola quali contano, senza che glielo si dica.

Due cose da portarsi via. La prima è che quella graduatoria sui fatti **è**
l'attenzione, con la sola differenza che qui l'archivio è un magazzino a parte
invece della frase stessa. La seconda è che la struttura
domanda-contro-archivio, con i fatti tenuti fuori dalla rete e consultati al
momento, è esattamente la forma dei sistemi che nella sezione su RAG recuperano
documenti prima di rispondere.

Sulle date conviene però essere precisi, perché la tentazione di raccontarla
come una discendenza è forte e sarebbe falsa: l'attenzione per la traduzione è
del settembre 2014, le memory network dell'ottobre dello stesso anno, la
versione a graduatoria del marzo 2015. Sono due strade partite quasi insieme,
da due problemi diversi, e arrivate alla stessa operazione; nessuna delle due
nasce dall'altra. Quello che le reti a memoria hanno di proprio non è dunque
l'attenzione, è l'**archivio tenuto fuori dai pesi** e consultato al momento
della domanda: ed è quel pezzo lì, messo da parte perché la sua epoca non aveva
né i dati né l'hardware, a tornare cinque anni dopo con un altro nome.
```

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
originale ne usa otto. Perché otto e non nove? Perché funzionava: è una scelta
provata sul campo, non una legge di natura, e i modelli che sono venuti dopo
usano numeri diversi.
`````

`````{tab} Superiore
La **Multi-Head Attention** esegue $h$ attenzioni indipendenti in sottospazi
distinti e ne ricompone gli esiti:

$$
\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) =
\text{Concat}(\text{head}_1, \ldots, \text{head}_h)\,\mathbf{W}^O
$$

dove $\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_i^Q,
\mathbf{K}\mathbf{W}_i^K, \mathbf{V}\mathbf{W}_i^V)$ e
$\mathbf{W}_i^Q, \mathbf{W}_i^K, \mathbf{W}_i^V, \mathbf{W}^O$ sono matrici
apprese. Nel Transformer originale
$h = 8$ e ogni testa lavora in dimensione $d_k = d_{\text{model}}/h = 64$: il
costo complessivo resta paragonabile a una singola attenzione a dimensione
piena, ma il modello può dedicare teste diverse a relazioni diverse
(sintattiche, semantiche, posizionali), cosa che l'analisi empirica delle
teste addestrate conferma almeno in parte.
`````

## Dove va a finire l'attenzione: encoder e decoder

Il blocco di attenzione non vive da solo: è il cuore di due componenti che la
prossima sezione smonta pezzo per pezzo. L'**encoder** legge la frase di
partenza e ne costruisce una rappresentazione ricca; il **decoder** la usa per
generare l'uscita (una traduzione, una risposta) un pezzo alla volta. In
mezzo, ancora attenzione: mentre genera, il decoder "evidenzia" le parti
rilevanti di ciò che l'encoder ha letto.

Ogni blocco è completato da due accorgimenti che rendono addestrabili anche
reti molto profonde.

`````{tab} Elementare
Il primo è una **scorciatoia**: l'informazione che entra in un blocco viene
anche fatta passare *intatta* accanto al blocco, e sommata all'uscita. Serve a
due cose, e la seconda è meno ovvia. All'andata, tiene aperta una strada
diretta perché l'informazione arrivi in cima senza sfilacciarsi in mezzo a
decine di blocchi. Al ritorno, serve alla correzione: quando la rete scopre di
aver sbagliato, il segnale che dice «di quanto e in che direzione ritoccare»
deve tornare indietro fino ai primi blocchi, e senza scorciatoia si spegne per
strada. Come un corrimano lungo una scala ripida: anche se un gradino è
scivoloso, chi sale e chi scende hanno sempre una presa solida. Il secondo
accorgimento è una **taratura**: le parole, dentro la rete, sono liste di
numeri, e quei numeri dopo ogni blocco vengono riportati su una scala standard,
come rimettere a zero la bilancia tra una pesata e l'altra, così nessuno strato
lavora con valori fuori misura.
`````

`````{tab} Superiore
Sono le **residual connection** e la **layer normalization**, combinate in

$$
\text{LayerNorm}\big(\mathbf{x} + \text{SubLayer}(\mathbf{x})\big)
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
sotto-strato, $\mathbf{x} + \text{SubLayer}(\text{LayerNorm}(\mathbf{x}))$, il
cosiddetto
*Pre-LN*, ed è lì che il cammino identità diventa davvero pulito (Xiong e
colleghi {cite}`xiong2020layer` mostrano che senza questo spostamento serve un
riscaldamento graduale del learning rate per addestrare stabilmente).
`````

Scorciatoia e taratura sono la parte che nessuno racconta mai, e senza la quale
niente di tutto il resto starebbe in piedi: l'attenzione è l'idea, ma un'idea
impilata sessanta volte si sfalda, e questi due accorgimenti sono ciò che la
tiene insieme. Ci si può fermare qui con il meccanismo in mano; la sezione
successiva prende questi pezzi e li monta nelle due torri di una macchina vera.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- L'**attenzione** rilegge la frase con un evidenziatore: per capire una
  parola, guarda tutte le altre, dà a ciascuna un'intensità di colore e ne
  mescola le informazioni in quella proporzione.
- Le intensità le decide la frase, non un programmatore: la rete le impara
  provando e correggendosi su miliardi di esempi. Quando è ogni parola a
  guardare tutte le altre, si chiama **self-attention**.
- Per giocare, ogni parola si presenta in tre versioni: la **query** (la
  domanda che fa), la **key** (l'etichetta con cui si fa trovare) e il
  **value** (l'informazione che consegna).
- Di evidenziatori se ne passano otto in parallelo, ognuno attento a un tipo di
  legame diverso: sono le **teste** di attenzione.
- Attorno a ogni blocco ci sono una **scorciatoia** (l'informazione passa anche
  di lato, intatta, e la correzione degli errori trova sempre una presa per
  tornare indietro) e una **taratura** (i numeri riportati su una scala
  standard). Senza di loro le torri alte non si addestrano.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- L'**attenzione** costruisce, per ogni parola, una rappresentazione
  contestuale: media dei *value* pesata dalle affinità *query*·*key*,
  normalizzate con softmax e scalate di $\sqrt{d_k}$ (il fattore neutralizza la
  dipendenza della varianza da $d_k$, sotto l'ipotesi di componenti
  indipendenti a media nulla e varianza unitaria).
- Nella **self-attention** ogni parola guarda tutte le altre; i pesi non sono
  fissati a mano ma appresi.
- La **Multi-Head Attention** esegue più attenzioni in parallelo ($h = 8$ nel
  modello originale), ciascuna libera di specializzarsi su relazioni diverse.
- **Residual connection** e **layer normalization** tengono addestrabili le
  pile profonde di blocchi. L'articolo del 2017 le combina come
  $\text{LayerNorm}(\mathbf{x} + \text{SubLayer}(\mathbf{x}))$ (*Post-LN*); i
  modelli successivi normalizzano prima del sotto-strato,
  $\mathbf{x} + \text{SubLayer}(\text{LayerNorm}(\mathbf{x}))$ (*Pre-LN*), ed
  è così che la
  scorciatoia resta davvero libera.
```
`````
