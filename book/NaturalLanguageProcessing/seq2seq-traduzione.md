# Da frase a frase: tradurre con le reti

Prova a tradurre parola per parola: «Il gatto nero salta sul muro»
diventerebbe *«The cat black jumps on the wall»*, e un inglese storcerebbe il
naso, perché l'aggettivo va prima del nome: *«The black cat jumps on the
wall»*. Sei parole sono diventate sette, e due si sono scambiate di posto. La
traduzione non è una sostituzione uno-a-uno: è prendere il *senso* di una
sequenza e riscriverlo in un'altra sequenza, di lunghezza diversa e con un
ordine diverso.

Nella sezione precedente abbiamo costruito gli attrezzi: RNN, LSTM, GRU. In
questa li mettiamo alla prova sul compito che più di ogni altro ha spinto
avanti l'NLP: la traduzione automatica. È una storia che vale la pena seguire
da vicino, perché è proprio qui, tra il 2014 e il 2017, che nasce il
meccanismo di **attenzione**: il ponte diretto verso il capitolo sui
Transformer.

## Scommettere sulla prossima parola

Prima di tradurre, un modello deve saper *parlare* la lingua d'arrivo. Lo
strumento è il **modello di linguaggio** che conosciamo dalla sezione sugli
*n-gram*: un sistema che, data una sequenza di parole, assegna una probabilità
alla parola successiva (la tastiera che dopo «a domani e buona» suggerisce
«serata» e quasi mai «carburatore»). La novità è *chi* fa la scommessa: non
più una tabella di conteggi, ma una rete ricorrente con la sua memoria.

`````{tab} Elementare

Facciamo il gioco a voce: «Il gatto nero salta sul…». La maggior parte delle
persone completa con «muro», qualcuno con «tetto» o «divano», nessuno con
«marmellata». Un modello di linguaggio fa esattamente questa scommessa, ma con
i numeri: «muro» 35%, «tetto» 25%, «divano» 10%, e giù fino a briciole di
probabilità per le parole assurde. Come misurare se scommette bene? Con la
**perplessità**, che abbiamo incontrato nei richiami di matematica e già
usata come pagella per gli *n-gram*: dice *come se* il modello, a ogni
parola, tirasse un dado con un
certo numero di facce. Perplessità 20 = incerto come un dado a 20 facce;
perplessità 5 = quasi sicuro, il dado ha solo 5 facce. Più bassa, meglio è: il
modello ha capito la lingua abbastanza da restringere le alternative.

`````

`````{tab} Superiore

Un modello di linguaggio stima la probabilità di un'intera sequenza
scomponendola, con la regola della catena, in predizioni della parola
successiva:

$$
P(w_1, \dots, w_n) = \prod_{t=1}^{n} P(w_t \mid w_1, \dots, w_{t-1}),
$$

dove $w_t$ è la parola al passo $t$. Una RNN implementa ciascun fattore in modo
naturale: lo stato nascosto $h_{t-1}$ riassume il prefisso letto fin lì, e una
softmax sul vocabolario produce la distribuzione
$P(w_t \mid w_{<t}) = \mathrm{softmax}(W_{hy}\,h_{t-1})$. L'addestramento è
la cross-entropia sulla parola successiva, e la qualità si misura con la
**perplessità per parola**, vista nella sezione di teoria dell'informazione e
già usata per valutare i modelli *n-gram*:

$$
\mathrm{PP} = 2^{H},
\qquad
H = -\frac{1}{n} \sum_{t=1}^{n} \log_2 P(w_t \mid w_{<t}),
$$

dove $H$ è la cross-entropia media sul testo di test. La perplessità è il
numero di alternative equiprobabili tra cui il modello «esita» a ogni passo:
un modello perfetto avrebbe $\mathrm{PP}=1$, uno che tira a caso su un
vocabolario di $50\,000$ parole avrebbe $\mathrm{PP}=50\,000$.

`````

## Leggere in due direzioni

Le RNN della sezione precedente leggono da sinistra a destra. Ma il senso di
una parola dipende anche da ciò che viene *dopo*: in «La pesca era la sua
passione» e «La pesca era matura», al momento in cui leggi «pesca» non puoi
ancora sapere se si parla di ami o di frutta. Lo scopri solo alla fine. Da qui
un'idea degli anni Novanta {cite}`schuster1997bidirectional`: le **RNN
bidirezionali**.

`````{tab} Elementare

È come rileggere un giallo conoscendo il colpevole: alla seconda lettura ogni
indizio va al suo posto, perché sai già come va a finire. Una rete
bidirezionale fa le due letture insieme: una cella percorre la frase da
sinistra a destra, un'altra da destra a sinistra, e per ogni parola si
incollano i due riassunti (quello di ciò che precede e quello di ciò che
segue). Attenzione però: questo trucco vale solo per **capire** un testo che
esiste già tutto intero. Per **generare** una frase non funziona: mentre
scrivi, le parole future non esistono ancora; nessun giallista può rileggere
il capitolo che deve ancora scrivere. Per questo, come vedremo, chi *legge* la
frase può essere bidirezionale, ma chi la *scrive* procede sempre in avanti.

`````

`````{tab} Superiore

Una RNN bidirezionale mantiene due catene di stati indipendenti: una in
avanti, $\overrightarrow{h}_t = f(\overrightarrow{h}_{t-1}, x_t)$, e una
all'indietro, $\overleftarrow{h}_t = f'(\overleftarrow{h}_{t+1}, x_t)$. La
rappresentazione della posizione $t$ è la concatenazione

$$
h_t = [\overrightarrow{h}_t ; \overleftarrow{h}_t],
$$

che condensa l'intera frase *vista da quella posizione*: prefisso e suffisso.
È lo standard per i compiti di comprensione (classificazione, NER, encoding),
ma è inapplicabile alla generazione autoregressiva: al passo $t$ la catena
all'indietro richiederebbe $x_{t+1}, \dots, x_n$, che non sono ancora stati
generati. Il decoder resta quindi unidirezionale per costruzione: un vincolo
di causalità che ritroveremo, sotto forma di *maschera*, anche nei
Transformer. Ortogonale a questo è l'**impilamento** (*stacked RNN*): più
strati ricorrenti sovrapposti, dove la sequenza di stati dello strato $\ell-1$
fa da input allo strato $\ell$, per rappresentazioni via via più astratte.

`````

In PyTorch entrambe le varianti sono un argomento del costruttore:

```python
import torch
from torch import nn

lstm = nn.LSTM(
    input_size=64, hidden_size=128,
    num_layers=2,        # due strati impilati
    bidirectional=True,  # lettura in entrambe le direzioni
    batch_first=True,
)

x = torch.randn(1, 6, 64)  # 1 frase, 6 parole ("Il gatto nero salta sul muro")
out, (h, c) = lstm(x)
print(out.shape)  # torch.Size([1, 6, 256]): 2 direzioni x 128 per ogni parola
print(h.shape)    # torch.Size([4, 1, 128]): 2 strati x 2 direzioni
```

## Comprimere una frase in un vettore

```{figure} ../figures/seq2seq-2014.svg
:name: fig-encoder-decoder
:alt: "Schema encoder-decoder: l'encoder legge una a una le parole della frase inglese e le comprime in un unico vettore di contesto; da quel vettore il decoder genera le parole italiane una dopo l'altra, riusando a ogni passo la parola appena prodotta."
:width: 100%

Due reti e un vettore in mezzo. L'encoder finisce di leggere prima che il
decoder cominci a scrivere: fra i due passa solo quel vettore, e nient'altro.
```

Il «nient'altro» di {numref}`fig-encoder-decoder` è il fatto architetturale
da cui discendono le due sezioni successive. Finché l'unico canale fra le due
reti è un vettore di dimensione fissa, la lunghezza della frase da tradurre
non cambia lo spazio disponibile per rappresentarla.

Torniamo alla traduzione. Nel 2014 due gruppi, Cho e colleghi a Montréal
{cite}`cho2014learning` (lo stesso paper che introduce la GRU) e Sutskever,
Vinyals e Le a Google {cite}`sutskever2014sequence`, arrivano alla stessa
architettura, oggi nota come **encoder–decoder** o **seq2seq**. L'idea è di
una semplicità disarmante: una prima rete ricorrente (l'*encoder*) legge tutta
la frase sorgente e la comprime nel suo stato finale, un unico **vettore di
contesto**; una seconda rete (il *decoder*) parte da quel vettore e genera la
traduzione parola per parola, come un modello di linguaggio «condizionato»
dalla frase di partenza.

`````{tab} Elementare

Immagina un interprete a un convegno che non può prendere appunti: ascolta
l'intervento intero, lo tiene tutto a memoria e solo alla fine lo ripete in
italiano. L'encoder è l'ascolto, il vettore di contesto è ciò che gli resta in
testa, il decoder è la resa in italiano. Nel paper di Google c'è un dettaglio
curioso: dare all'encoder la frase sorgente **al contrario** («muro sul salta
nero gatto Il») migliorava nettamente le traduzioni. Perché? Così l'*inizio*
della frase (la prima cosa che il decoder deve tradurre) viene letto per
*ultimo*, ed è il ricordo più fresco. Un trucco che rivela il difetto di
fondo: se la qualità dipende da quale parola è stata ascoltata più di recente,
la memoria unica è troppo stretta. Sulle frasi brevi regge; su un discorso
lungo l'interprete arranca, perché tutto non entra in un solo ricordo.

`````

`````{tab} Superiore

Il modello fattorizza la probabilità della frase di arrivo
$y = (y_1, \dots, y_m)$ data quella di partenza $x = (x_1, \dots, x_n)$ come

$$
P(y \mid x) = \prod_{t=1}^{m} P(y_t \mid y_1, \dots, y_{t-1}, c),
\qquad
c = h_n,
$$

dove $c$ è il vettore di contesto (lo stato finale dell'encoder) e ogni
fattore è calcolato dal decoder, una RNN inizializzata da $c$ con softmax sul
vocabolario di arrivo. Sutskever et al. usano LSTM a 4 strati con stati da
1000 dimensioni e riportano, sul benchmark WMT'14 inglese→francese, un BLEU di
$34{,}8$ (con un ensemble di cinque modelli) contro il $33{,}3$ del sistema
statistico a frasi di riferimento. L'aneddoto dell'inversione è documentato
nei numeri: invertire l'ordine delle parole sorgente fa scendere la
perplessità di test da $5{,}8$ a $4{,}7$ e salire il BLEU da $25{,}9$ a
$30{,}6$, perché accorcia le dipendenze tra le prime parole di $x$ e le prime
di $y$, semplificando l'ottimizzazione. Ma il limite strutturale resta:
qualunque sia $n$, tutta l'informazione su $x$ deve passare per un vettore di
dimensione fissa. È un **collo di bottiglia**, e le prestazioni degradano
visibilmente al crescere della lunghezza della frase.

`````

## Tornare a guardare: la nascita dell'attenzione

La soluzione arriva nel giro di pochi mesi, da Dzmitry Bahdanau, Kyunghyun Cho
e Yoshua Bengio {cite}`bahdanau2015neural`.

```{figure} ../figures/attention-prima-dei-transformer.svg
:name: fig-allineamento-traduzione
:alt: "Una griglia di allineamento fra le parole della frase inglese, sulle colonne, e le parole italiane generate, sulle righe. Le celle più scure indicano dove il decoder ha guardato di più a ogni passo: la diagonale è marcata ma non perfetta, e in un punto due parole italiane si collegano a una sola inglese."
:width: 88%

L'allineamento che nessuno ha annotato. La griglia non è stata insegnata al
modello: è il sottoprodotto dei pesi di attenzione, che si possono leggere
dopo l'addestramento.
```

Il fatto che la diagonale di {numref}`fig-allineamento-traduzione` sia
imperfetta è la notizia, non un difetto. Dove le due lingue ordinano le parole
in modo diverso l'allineamento si spezza e attraversa la griglia, ed è
esattamente il caso che un decoder costretto a leggere in ordine non poteva
gestire. La domanda giusta è: perché
costringere il decoder a lavorare a memoria? La frase sorgente è ancora lì,
con tutti gli stati che l'encoder ha prodotto leggendo ogni parola. Basta
lasciare che il decoder, a ogni passo, **torni a guardarli tutti**: dando a
ciascuno un peso diverso a seconda di quanto serve *adesso*. Questi pesi sono
l'**attenzione**.

```{figure} ../figures/seq2seq-attenzione.svg
:name: fig-seq2seq-attenzione
:alt: "Encoder-decoder con attenzione: in basso sei stati dell'encoder bidirezionale per «Il gatto nero salta sul muro», in alto il decoder che genera «The black cat»; mentre produce «cat», frecce di spessore diverso collegano ogni parola sorgente al decoder, e la più spessa parte da «gatto»."
:width: 100%

Mentre genera «cat», il decoder consulta *tutti* gli stati dell'encoder: lo
spessore di ogni freccia è il peso di attenzione, massimo su «gatto».
```

Come mostra {numref}`fig-seq2seq-attenzione`, mentre produce *«cat»* il
decoder concentra il peso su «gatto», tiene d'occhio «nero» e quasi ignora il
resto. E a ogni passo la mappa cambia: per *«wall»* il peso si sposterà su
«muro». Il collo di bottiglia sparisce, perché nessun vettore fisso deve più
contenere l'intera frase.

`````{tab} Elementare

Il nostro interprete adesso ha il testo dell'intervento sul tavolo. Mentre
traduce non recita più a memoria: per ogni parola che pronuncia dà un'occhiata
al foglio, e l'occhio cade sul punto che serve in quel momento, come un
evidenziatore che si sposta man mano che la traduzione avanza. Non c'è nessuna
regola scritta a mano che dica dove guardare: la rete impara da sola, durante
l'addestramento, che quando sta per dire *cat* conviene pesare molto «gatto».
Sorpresa in regalo: disegnando dove cade l'evidenziatore si ottiene, gratis,
l'allineamento tra le parole delle due lingue («cat» ↔ «gatto», «wall» ↔
«muro») che nessuno aveva chiesto al modello di imparare.

`````

`````{tab} Superiore

L'encoder (bidirezionale, così che ogni $h_j$ rappresenti la parola $j$ con
tutto il suo contesto) produce gli stati $h_1, \dots, h_n$. Al passo $i$ il
decoder, con stato $s_{i-1}$, calcola un punteggio di allineamento verso ogni
posizione sorgente con una piccola rete a un solo strato nascosto:

$$
e_{ij} = v_a^{\top} \tanh\!\left(W_a\, s_{i-1} + U_a\, h_j\right),
$$

dove $W_a$, $U_a$ e $v_a$ sono parametri appresi (è la cosiddetta attenzione
**additiva**). I punteggi diventano pesi con una softmax, e i pesi definiscono
un vettore di contesto *diverso a ogni passo*:

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{n} \exp(e_{ik})},
\qquad
c_i = \sum_{j=1}^{n} \alpha_{ij}\, h_j,
$$

dove $\alpha_{ij}$ è quanto il passo di decodifica $i$ «guarda» la parola
sorgente $j$ (i pesi sommano a 1) e $c_i$ è la media pesata degli stati
dell'encoder, che entra nel calcolo di $s_i$ e della parola successiva. La
matrice dei pesi $\alpha_{ij}$, visualizzata, è una mappa di allineamento tra
le due frasi: appresa senza alcuna supervisione esplicita.

`````

Vale la pena dirlo in modo esplicito: questa è **la stessa attenzione** che
ritroveremo nel capitolo sui Transformer. Cambierà solo il modo di calcolare i
punteggi (non più una piccola rete con la $\tanh$, ma un semplice prodotto
scalare riscalato, la *scaled dot-product attention*) e cadrà l'impalcatura
ricorrente attorno. L'idea di fondo, «una media pesata di tutti gli stati, con
pesi softmax appresi», nasce qui, come rattoppo per la traduzione.

## Generare la frase: greedy e beam search

Resta un problema che finora abbiamo dato per scontato: a ogni passo il
decoder produce una *distribuzione* di probabilità sul vocabolario. Come si
sceglie la parola? L'istinto dice: prendi la più probabile e vai avanti
(strategia **greedy**, «ingorda»). Ma la parola migliore *adesso* non porta
sempre alla frase migliore *alla fine*.

`````{tab} Elementare

Facciamo i conti su un esempio piccolo. Il decoder deve iniziare la
traduzione e propone: «A» con probabilità 0,50, «The» con 0,40. La strategia
greedy sceglie «A» e non torna più indietro. Ma guardiamo un passo più in là:
dopo «A», la parola «black» ha probabilità 0,30, quindi la coppia «A black»
vale 0,50 × 0,30 = 0,15; dopo «The», invece, «black» ha probabilità 0,60, e
«The black» vale 0,40 × 0,60 = 0,24. La strada che partiva peggio è arrivata
meglio! La **beam search** («ricerca a fascio») rimedia tenendo aperte le $k$
strade più promettenti invece di una sola: con $k=2$ conserva sia «A» sia
«The», scopre al passo dopo che «The black» è in testa, e prosegue fino a
«The black cat…». È come sciogliere un dubbio al bivio non scegliendo subito,
ma facendo qualche passo lungo entrambe le strade prima di decidere.

`````

`````{tab} Superiore

Formalmente cerchiamo $\hat{y} = \arg\max_y P(y \mid x)$, ma le sequenze
possibili sono $|V|^m$ (vocabolario $V$, lunghezza $m$): la ricerca esaustiva
è intrattabile e la scelta greedy è solo l'approssimazione con orizzonte 1.
La beam search è una via di mezzo: a ogni passo estende le $k$ ipotesi
correnti con tutte le parole del vocabolario, ordina le sequenze per
punteggio cumulato

$$
\mathrm{score}(y_{1:t}) = \sum_{i=1}^{t} \log P(y_i \mid y_{<i}, x)
$$

e trattiene le migliori $k$ (con $k=1$ si torna alla greedy). Non è una
ricerca esatta (l'ottimo globale può comunque sfuggire al fascio) ma in
traduzione valori di $k$ tra 4 e 10 bastano quasi sempre. Un dettaglio
pratico: essendo una somma di logaritmi negativi, il punteggio penalizza le
frasi lunghe, e il decoder tenderebbe a traduzioni troppo corte. Si corregge
con una **length penalty**, per esempio dividendo il punteggio per
$|y|^{\alpha}$ con $\alpha \approx 0{,}6$–$0{,}7$: è la scelta, in una
variante appena più elaborata, del sistema di traduzione di Google
{cite}`wu2016google`.

`````

```{figure} ../figures/beam-search.svg
:name: fig-beam-search
:alt: "Albero di beam search con larghezza due su tre passi: al primo passo restano nel fascio «A» (0,50) e «The» (0,40); al secondo «The black» (0,24) supera «A black» (0,15); al terzo l'ipotesi migliore è «The black cat» (0,19). I rami scartati sono in grigio tratteggiato."
:width: 100%

Beam search con $k=2$ sull'esempio del testo: i numeri sono le probabilità
cumulate. La greedy si sarebbe fermata su «A» al primo passo; il fascio
recupera «The black cat».
```

In {numref}`fig-beam-search` i rami in terracotta sono le ipotesi nel fascio,
quelli grigi le potature: il ramo spesso è la traduzione che la strategia
greedy non avrebbe mai trovato.

## 2016: la traduzione neurale entra in produzione

```{figure} ../figures/traduzione-automatica-da-regole-a-llm.svg
:name: fig-paradigmi-traduzione
:alt: "Linea del tempo con i quattro paradigmi della traduzione automatica: i sistemi a regole scritte da linguisti, i metodi statistici basati su corpora paralleli, la traduzione neurale con encoder-decoder e attenzione, e infine i modelli linguistici generalisti che traducono senza essere stati costruiti per farlo."
:width: 100%

Quattro modi di tradurre, in settant'anni. A ogni passaggio si sposta chi
fornisce la conoscenza della lingua: prima il linguista, poi il corpus, poi il
modello addestrato apposta, infine un modello che non era stato pensato per
questo.
```

L'ultimo passaggio di {numref}`fig-paradigmi-traduzione` è il più singolare, e
il libro lo incontrerà nel capitolo sui Transformer: la traduzione ha smesso di
essere un compito con un'architettura propria ed è diventata una delle cose
che un modello generalista sa fare. Qui però siamo alla terza tappa, ed è
quella che ha portato la traduzione neurale in produzione.

Questa storia ha una data di consegna. Nel settembre 2016 Google annuncia GNMT
(*Google Neural Machine Translation*) {cite}`wu2016google`: un encoder–decoder
di LSTM a 8 strati con attenzione (esattamente la ricetta di questa sezione,
in grande) che sostituisce in produzione il sistema statistico a frasi usato
per un decennio. Si parte dalla coppia cinese→inglese, circa 18 milioni di
traduzioni al giorno; nelle valutazioni umane fianco a fianco gli errori di
traduzione calano in media del 60% sulle principali coppie di lingue. Per la
prima volta le reti ricorrenti che abbiamo studiato traducono, ogni giorno,
per centinaia di milioni di persone.

Questa storia ha anche un seguito che il capitolo sui Transformer riprende per
intero. Pochi mesi dopo, invece di un modello per coppia di lingue, lo stesso
gruppo ne addestra **uno solo** su tutte le coppie insieme, e scopre che
traduce anche fra due lingue che non ha mai visto appaiate
{cite}`johnson2017google`: è il primo indizio che dentro una rete addestrata su
molte lingue si formi qualcosa di simile a una lingua franca interna.

Ma il rattoppo si stava già mangiando il vestito. Se l'attenzione permette a
ogni parola generata di guardare direttamente tutte le parole sorgente, a cosa
serve ancora far scorrere uno stato passo dopo passo? Nel 2017 un gruppo di
ricercatori di Google si fa esattamente questa domanda, e il titolo della
risposta è tutto un programma: *Attention Is All You Need*
{cite}`vaswani2017attention`; l'attenzione è tutto ciò che serve. Via la
ricorrenza, resta solo il meccanismo che avete appena visto nascere, promosso
da comprimario a protagonista. Come, di preciso, è il tema del capitolo sui
**Transformer**.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Un **modello di linguaggio** scommette sulla parola successiva, e la sua
  pagella è la **perplessità**: quante facce ha il dado con cui esita a ogni
  passo. Più è bassa, più il modello ha ristretto le alternative.
- Le **RNN bidirezionali** rileggono la frase nei due sensi insieme, come un
  giallo di cui si conosce già il colpevole: preziose per *capire* un testo
  che esiste tutto intero, inutilizzabili per *scriverne* uno, perché mentre
  si scrive le parole future non ci sono ancora.
- **Seq2seq** è l'interprete senza appunti: una rete (l'encoder) ascolta
  l'intera frase di partenza e la tiene in un unico ricordo, una seconda (il
  decoder) la ridice nell'altra lingua partendo da lì. Se la frase è lunga,
  in quel ricordo non ci sta tutto: è il collo di bottiglia.
- L'**attenzione di Bahdanau** mette il testo sul tavolo dell'interprete: per
  ogni parola che pronuncia, un'occhiata al punto che serve adesso, con
  un'attenzione che si sposta a ogni passo e che nessuno gli ha insegnato dove
  posare. In regalo si ottiene l'allineamento fra le parole delle due lingue,
  ed è la stessa idea che nei Transformer diventerà protagonista.
- Prendere ogni volta la parola più probabile è **miope**, perché la strada
  che parte peggio può arrivare meglio: la **beam search** tiene aperte le
  poche strade più promettenti e decide qualche passo più avanti (con una
  correzione che le impedisce di preferire sempre le frasi corte).
- Nel 2016 la traduzione neurale entra in produzione con GNMT di Google; nel
  2017 *Attention Is All You Need* manda in soffitta la lettura passo dopo
  passo e tiene solo l'attenzione.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Un **modello di linguaggio** assegna probabilità alla parola successiva; la
  sua qualità si misura con la **perplessità** $2^H$: il numero di alternative
  equiprobabili tra cui esita a ogni passo.
- Le **RNN bidirezionali** leggono la frase nei due sensi: preziose per
  *capire*, inutilizzabili per *generare* (il futuro non esiste ancora: il
  decoder è unidirezionale per costruzione).
- **Seq2seq**: un encoder comprime la frase sorgente in un vettore di
  contesto, un decoder la riscrive nell'altra lingua. Il vettore fisso è un
  **collo di bottiglia** sulle frasi lunghe.
- L'**attenzione di Bahdanau** lo elimina: a ogni passo il decoder rivede
  *tutti* gli stati dell'encoder con pesi $\alpha_{ij}$ appresi; è il
  precursore diretto della *scaled dot-product attention* dei Transformer.
- In generazione la scelta **greedy** è miope; la **beam search** tiene
  aperte le $k$ ipotesi migliori (con una *length penalty* per non penalizzare
  le frasi lunghe).
- Nel 2016 la traduzione neurale entra in produzione con GNMT; nel 2017
  *Attention Is All You Need* fa cadere la ricorrenza.
```
`````
