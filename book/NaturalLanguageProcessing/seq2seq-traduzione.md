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
probabilità per le parole assurde. (Le tre percentuali sono inventate qui per
far vedere l'idea: in un modello vero escono dai conteggi, come nella sezione
sugli *n-gram*.) Come misurare se scommette bene? Con la
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
naturale: lo stato nascosto $\mathbf{h}_{t-1}$ riassume il prefisso letto fin
lì, e una softmax sul vocabolario produce la distribuzione
$P(w_t \mid w_{<t}) = \mathrm{softmax}(\mathbf{W}_{hy}\,\mathbf{h}_{t-1})$.
L'addestramento è la cross-entropia sulla parola successiva, e la qualità si
misura con la **perplessità per parola**, vista nella sezione di teoria
dell'informazione e già usata per valutare i modelli *n-gram*:

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

Una RNN bidirezionale mantiene due catene di stati indipendenti: una in avanti,
$\overrightarrow{\mathbf{h}}_t = f(\overrightarrow{\mathbf{h}}_{t-1}, x_t)$, e
una all'indietro,
$\overleftarrow{\mathbf{h}}_t = f'(\overleftarrow{\mathbf{h}}_{t+1}, x_t)$. La
rappresentazione della posizione $t$ è la concatenazione

$$
\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t ; \overleftarrow{\mathbf{h}}_t],
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

In PyTorch la lettura nei due sensi si chiede scrivendo `bidirectional=True`
quando si costruisce la rete, e non c'è altro da fare. Nella stessa riga
compare anche `num_layers=2`, che chiede due celle impilate una sopra l'altra:
la seconda non legge le parole, legge quello che ha capito la prima. Sono i due
modi in cui una rete ricorrente si può far crescere, in larghezza (le due
direzioni) e in altezza (gli strati):

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

Torniamo alla traduzione. Nel 2014 due gruppi di ricerca arrivano, per strade
loro, alla stessa idea, e l'idea è di una semplicità disarmante: **due reti
ricorrenti, una di fronte all'altra**. La prima legge la frase da tradurre e
non scrive niente; la seconda scrive la traduzione e non legge l'originale. Fra
le due passa una fila di numeri, e basta.

La prima si chiama *encoder*, «chi codifica», e il suo mestiere è ridurre la
frase di partenza a quel pacchetto di numeri; la seconda si chiama *decoder*,
«chi decodifica», e il suo mestiere è srotolare il pacchetto in una frase
dell'altra lingua, una parola per volta. Il pacchetto ha un nome, **vettore di
contesto**, e non è altro che l'ultimo riassunto scritto dall'encoder: quello
che gli resta in mano dopo aver letto l'ultima parola. L'architettura si chiama
**encoder–decoder**, o **seq2seq**, e i due lavori che la propongono nello
stesso anno sono di Kyunghyun Cho e colleghi a Montréal
{cite}`cho2014learning`, che è lo stesso articolo in cui nasce la GRU, e di
Ilya Sutskever, Oriol Vinyals e Quoc Le a Google
{cite}`sutskever2014sequence`.

Il decoder, mentre scrive, fa esattamente quello che fa un modello di
linguaggio: scommette sulla parola successiva. Con una differenza: la sua
scommessa non parte dal nulla, parte dal pacchetto ricevuto. Si dice allora che
è **condizionato** dalla frase di partenza, che è il modo tecnico di dire «gli
è stato detto di che cosa deve parlare».

```{figure} ../figures/seq2seq-2014.svg
:name: fig-encoder-decoder
:alt: "Schema encoder-decoder: l'encoder legge una a una le tre parole della frase inglese «the cat sleeps» e le comprime in un unico vettore di contesto; da quel vettore il decoder genera la traduzione francese «le chat dort», una parola dopo l'altra."
:width: 100%

Due reti e un vettore in mezzo. L'encoder finisce di leggere prima che il
decoder cominci a scrivere: fra i due passa solo quel vettore, e nient'altro.
```

Il «nient'altro» di {numref}`fig-encoder-decoder` è il fatto da cui discende
tutto il resto della sezione, e conviene fissarlo bene. Quel vettore di
contesto è una fila di numeri **di lunghezza decisa in anticipo**, mille per
esempio, e resta di mille numeri sia che la frase da tradurre abbia cinque
parole sia che ne abbia cinquanta: nessuno spazio in più per le frasi lunghe,
per quanto ce ne sarebbe bisogno.

E c'è un dettaglio che tornerà utile fra due pagine, quindi mettiamolo a fuoco
adesso. L'encoder, mentre legge, produce **un riassunto dopo ogni parola**: uno
dopo «il», uno dopo «il gatto», uno dopo «il gatto nero», e così via fino in
fondo. Sono tanti riassunti quante sono le parole, ciascuno con la sua fila di
numeri. Di tutti questi, però, ne viene passato al decoder uno solo, l'ultimo.
Gli altri esistono, sono già stati calcolati, e vengono buttati via.

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
P(y \mid x) = \prod_{t=1}^{m} P(y_t \mid y_1, \dots, y_{t-1}, \mathbf{c}),
\qquad
\mathbf{c} = \mathbf{h}_n,
$$

dove $\mathbf{c}$ è il vettore di contesto (lo stato finale dell'encoder) e
ogni fattore è calcolato dal decoder, una RNN inizializzata da $\mathbf{c}$ con
softmax sul vocabolario di arrivo.

I risultati che seguono si misurano in **BLEU** {cite}`papineni2002bleu`, il
metro con cui la traduzione automatica si è confrontata per vent'anni, e vale
la pena dire com'è fatto, perché tornerà anche nella sezione sul dialogo. BLEU
confronta la traduzione candidata con uno o più riferimenti umani contando
quanti $n$-grammi hanno in comune, per $n$ da 1 a 4. Due accorgimenti fanno
tutto il lavoro. Il primo è il **clipping**: un $n$-gramma del candidato conta
al massimo il numero di volte che compare nel riferimento, altrimenti «il il il
il» otterrebbe precisione $1$. Il secondo è la **brevity penalty**,
$\mathrm{BP} = \min\!\left(1,\, e^{1 - r/c}\right)$ con $c$ e $r$ le lunghezze
in token del candidato e del riferimento, e serve perché BLEU è fatto di sole
precisioni: un termine di *recall* non c'è (non esiste un modo ovvio di
calcolarlo su più riferimenti insieme) e senza freno la traduzione più corta
sarebbe sempre la migliore. Il punteggio è

$$
\mathrm{BLEU} = \mathrm{BP} \cdot
\exp\!\left(\sum_{n=1}^{4} w_n \log p_n\right),
$$

dove $p_n$ è la precisione clippata degli $n$-grammi e $w_n = 1/4$ il peso
uniforme dei quattro ordini. I limiti vanno detti subito, perché servono a
leggere i numeri qui sotto: BLEU è definito **sul corpus** e non sulla singola
frase (le $p_n$ si accumulano su tutto il test set, e su una frase sola un
4-gramma mancante manda il punteggio a zero); dipende dalla tokenizzazione e
dal numero di riferimenti, tanto che due punteggi si confrontano solo a
protocollo identico, ed è la ragione per cui esiste `sacrebleu`; ed è cieco
alla parafrasi corretta. Un punto di differenza è un segnale, non una
sentenza.

Sutskever et al. usano LSTM a 4 strati con stati da
1000 dimensioni e riportano, sul benchmark WMT'14 inglese→francese, un BLEU di
$34{,}8$ (con un ensemble di cinque modelli) contro il $33{,}3$ del sistema
statistico a frasi di riferimento. Il confronto va letto per quello che è: il
$33{,}3$ è il sistema di *riferimento*, non lo stato dell'arte, che su quel
compito stava a $37{,}0$; la rete pura non lo raggiunge, e ci si avvicina
($36{,}5$) solo quando la si usa per riordinare le mille ipotesi prodotte dal
sistema statistico. Nel 2014 il neurale non ha ancora vinto: la data del
sorpasso è il 2016, ed è la storia con cui si chiude questa sezione. L'aneddoto
dell'inversione è invece documentato
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
e Yoshua Bengio {cite}`bahdanau2015neural`, e nasce da una domanda tanto ovvia
quanto ben posta: perché costringere il decoder a lavorare a memoria, se i
riassunti intermedi ci sono già?

Ricordate: l'encoder ne aveva prodotto uno per parola, e li avevamo buttati via
tutti tranne l'ultimo. Smettiamo di buttarli. Teniamoli lì tutti, in fila, e
lasciamo che il decoder, ogni volta che deve scrivere una parola, **li guardi
tutti quanti** e decida da sé a quali dare retta adesso. Non «guarda solo il
quinto»: dà a ciascuno un voto, alto per quelli che gli servono, basso per gli
altri, e poi li mescola in proporzione ai voti. Quei voti sono l'**attenzione**:
un numero per ogni parola della frase di partenza, ricalcolato da capo a ogni
parola prodotta.

```{figure} ../figures/attention-prima-dei-transformer.svg
:name: fig-allineamento-traduzione
:alt: "Due file di parole, una sopra l'altra: in alto la frase inglese «the black cat sleeps», in basso la traduzione italiana «il gatto nero dorme». Delle linee collegano le parole delle due file, e il loro spessore è il peso dell'attenzione. Le linee spesse legano «the» a «il» e «sleeps» a «dorme» senza incrociarsi, mentre al centro si incrociano: «black» va a «nero» e «cat» a «gatto». Due linee sottili, fra le stesse parole del centro, sono i pesi piccoli rimasti."
:width: 88%

L'allineamento che nessuno ha annotato. Le linee dicono, per ogni parola
prodotta, dove il modello ha guardato: nessuno gliel'ha insegnato, si leggono
a posteriori dai pesi che si è dato da solo.
```

Il fatto che le due linee centrali di {numref}`fig-allineamento-traduzione` si
incrocino è la notizia, non un difetto: in inglese l'aggettivo precede il nome,
in italiano lo segue, e il modello va a prendersi le parole fuori ordine, terza
prima e seconda dopo. È la cosa che con il vettore unico non si poteva fare, e
non perché fosse difficile: perché nel vettore unico l'informazione su dove
stava ciascuna parola era già stata schiacciata via.

```{figure} ../figures/seq2seq-attenzione.svg
:name: fig-seq2seq-attenzione
:alt: "Encoder-decoder con attenzione: in basso sei stati dell'encoder bidirezionale per «Il gatto nero salta sul muro», in alto il decoder che genera «The black cat»; mentre produce «cat», frecce di spessore diverso collegano ogni parola sorgente al decoder, e la più spessa parte da «gatto»."
:width: 100%

Mentre genera «cat», il decoder consulta *tutti* gli stati dell'encoder: lo
spessore di ogni freccia è il peso di attenzione, massimo su «gatto».
```

Come mostra {numref}`fig-seq2seq-attenzione`, mentre produce *«cat»* il
decoder dà il voto più alto a «gatto» (0,62), tiene d'occhio «nero» (0,20) e
quasi ignora il resto. E a ogni passo la mappa cambia: per *«wall»* il voto
grosso si sposterà su «muro».

(Nella figura quei voti sono chiamati **pesi**, ed è il termine che si usa
ovunque, ma attenzione a non confonderli con i pesi della sezione precedente,
quelli che una rete impara e si tiene: questi cambiano a ogni parola prodotta e
non sono roba che il modello possiede, sono roba che il modello *decide sul
momento*.)

Con questo, il collo di bottiglia del vettore unico sparisce: nessuna fila di
numeri di lunghezza fissa deve più contenere l'intera frase. Resta invece
intatto l'altro collo di bottiglia, quello della sezione precedente, cioè il
fatto che tutto questo si legge e si scrive in fila, un passo dopo l'altro. Due
strozzature diverse: l'attenzione ne toglie una, e sarà il capitolo dopo a
togliere l'altra.

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

L'encoder (bidirezionale, così che ogni $\mathbf{h}_j$ rappresenti la parola
$j$ con tutto il suo contesto) produce gli stati
$\mathbf{h}_1, \dots, \mathbf{h}_n$. Al passo $i$ il decoder, con stato
$\mathbf{s}_{i-1}$, calcola un punteggio di allineamento verso ogni posizione
sorgente con una piccola rete a un solo strato nascosto:

$$
e_{ij} = \mathbf{v}_a^{\top} \tanh\!\left(\mathbf{W}_a\, \mathbf{s}_{i-1} + \mathbf{U}_a\, \mathbf{h}_j\right),
$$

dove $\mathbf{W}_a$, $\mathbf{U}_a$ e $\mathbf{v}_a$ sono parametri appresi (è
la cosiddetta attenzione **additiva**). I punteggi diventano pesi con una
softmax, e i pesi definiscono un vettore di contesto *diverso a ogni passo*:

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{n} \exp(e_{ik})},
\qquad
\mathbf{c}_i = \sum_{j=1}^{n} \alpha_{ij}\, \mathbf{h}_j,
$$

dove $\alpha_{ij}$ è quanto il passo di decodifica $i$ «guarda» la parola
sorgente $j$ (i pesi sommano a 1) e $\mathbf{c}_i$ è la media pesata degli
stati dell'encoder, che entra nel calcolo di $\mathbf{s}_i$ e della parola
successiva. La matrice dei pesi $\alpha_{ij}$, visualizzata, è una mappa di
allineamento tra le due frasi: appresa senza alcuna supervisione esplicita.

`````

Vale la pena dirlo in modo esplicito, perché è il ponte verso il capitolo
successivo: questa è **la stessa attenzione** dei Transformer. E siccome è
l'idea che di là diventa tutto, conviene guardarla una volta con i numeri sotto
gli occhi.

Immaginiamo che i riassunti siano corti, tre numeri l'uno, e che ce ne siano
tre soltanto:

| riassunto | numeri | voto |
|---|---|---|
| dopo «il» | `2, 0, 1` | 0,10 |
| dopo «il gatto» | `0, 4, 2` | 0,70 |
| dopo «il gatto nero» | `1, 1, 0` | 0,20 |

I voti li ha decisi il decoder, guardando che cosa gli serve *adesso*, e sono
tre numeri positivi che sommano a uno: si spartiscono un totale fisso, proprio
come si spartisce l'attenzione di una persona. Se ne do di più a uno, ne resta
di meno per gli altri, esattamente come quando in classe ascolto il professore
e allora non sento chi mi parla da dietro.

Adesso si mescola. Per la prima casella: $0{,}10 \times 2 + 0{,}70 \times 0 +
0{,}20 \times 1 = 0{,}4$. Per la seconda: $0{,}10 \times 0 + 0{,}70 \times 4 +
0{,}20 \times 1 = 3{,}0$. Per la terza: $0{,}10 \times 1 + 0{,}70 \times 2 +
0{,}20 \times 0 = 1{,}5$. Il risultato è una fila di numeri nuova, `0,4 · 3,0 ·
1,5`, e si vede a occhio che somiglia molto al secondo riassunto e poco agli
altri: è il riassunto che il decoder aveva votato di più. Questa operazione,
mescolare più file di numeri dando a ciascuna un peso, si chiama **media
pesata**, ed è tutta l'attenzione. La fila che ne esce va dritta al decoder, che
la usa per scrivere la parola successiva: al posto del solito pacchetto sempre
uguale, adesso ne riceve uno confezionato apposta per il passo che sta facendo.

Cambieranno due cose, nel capitolo dopo. La prima è **come si decidono i
voti**. Qui li calcola un pezzo di rete apposito, addestrato insieme al resto:
gli si danno il riassunto e lo stato del decoder, e sputa fuori un numero. Là
il conto sarà diretto e molto più economico, fatto sulle due file di numeri
senza nessun pezzo in mezzo. La seconda è che cadrà tutta l'impalcatura
ricorrente attorno: resterà solo l'attenzione, che qui nasce come rattoppo per
la traduzione e là diventa l'architettura intera.

## Generare la frase: greedy e beam search

Resta un problema che finora abbiamo dato per scontato. Il decoder, a ogni
passo, non sceglie una parola: assegna una percentuale a **tutte** le parole
che conosce, decine di migliaia di numeri che sommano a uno. Come si passa da
quell'elenco a una parola sola? L'istinto dice: prendi la più alta e vai
avanti. Si chiama strategia **greedy**, «ingorda», ed è quello che fa chiunque
abbia fretta. Ma la parola migliore *adesso* non porta sempre alla frase
migliore *alla fine*.

(E chi gli dice di smettere? Fra le voci del suo elenco ce n'è una che non è
una parola: è il segnale di fine frase, lo stesso `</s>` incontrato con gli
n-gram. Quando il decoder scommette su quello, ha deciso che la traduzione è
finita, e ci si ferma. È una scommessa come le altre, e come le altre può
sbagliare: un modello che lo tira fuori troppo presto tronca la frase, uno che
non lo tira fuori mai continua a scrivere finché qualcuno non lo interrompe.)

`````{tab} Elementare

Facciamo i conti su un esempio piccolo, in cui per comodità facciamo finta che
le parole in gioco siano pochissime. Il decoder deve iniziare la traduzione e
propone: «A» con probabilità 0,50, «The» con 0,40. La strategia greedy sceglie
«A» e non torna più indietro.

Ma guardiamo un passo più in là. Il punteggio di una frase intera è il
**prodotto** delle probabilità incontrate lungo la strada, per la stessa
ragione per cui si moltiplicavano i voti del filtro antispam: sono cose che
devono capitare tutte insieme, e due cose che devono capitare insieme si
moltiplicano. Allora: dopo «A», la parola «black» ha probabilità 0,30, quindi
la coppia «A black» vale 0,50 × 0,30 = 0,15; dopo «The», invece, «black» ha
probabilità 0,60, e «The black» vale 0,40 × 0,60 = 0,24. La strada che partiva
peggio è arrivata
meglio! La **beam search** («ricerca a fascio») rimedia tenendo aperte le
poche strade più promettenti invece di una sola. Quante, lo si decide prima, e
quel numero lo si chiama $k$: con $k=2$ si conservano sia «A» sia «The», al
passo dopo si scopre che «The black» è in testa, e si prosegue fino a «The
black cat…». È come sciogliere un dubbio al bivio non scegliendo subito, ma
facendo qualche passo lungo entrambe le strade prima di decidere. Le strade
scartate lungo il cammino si dicono **potate**, come i rami di un albero, ed è
il motivo per cui questi disegni si fanno a forma di albero: dal tronco
partono tutte le continuazioni possibili, e a ogni passo se ne tagliano quasi
tutte.

C'è però un difetto da correggere, e si vede proprio dai numeri dell'esempio.
Le probabilità sono numeri minori di uno, e moltiplicandone due si ottiene
sempre qualcosa di più piccolo di ciascuna: 0,50 diventa 0,15 al secondo passo
e 0,12 al terzo. Quindi ogni parola in più fa scendere il punteggio, sempre,
anche quando la frase sta andando benissimo. Lasciata a sé, la ricerca
preferirebbe sistematicamente le traduzioni corte, e finirebbe per troncare le
frasi a metà.

Si rimedia mettendo tutte le strade sullo stesso metro prima di confrontarle:
invece del punteggio complessivo si guarda **quanto vale in media una singola
parola** di quella strada. Una frase di dieci parole e una di tre diventano
così paragonabili, perché di ciascuna si guarda la qualità per parola e non il
totale. La correzione si chiama **penalità di lunghezza**, e serve a togliere
alle frasi brevi un vantaggio che non si sono guadagnate.

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
$|y|^{\alpha}$ con $\alpha \approx 0{,}6$–$0{,}7$. Il sistema di traduzione di
Google {cite}`wu2016google` parte proprio da questa euristica e la sostituisce
poi con una variante appena più elaborata,
$lp(y) = \frac{(5+|y|)^{\alpha}}{6^{\alpha}}$, dove l'esponente agisce su
$(5+|y|)$ e non su $|y|$: per questo il valore che gli autori usano,
$\alpha = 0{,}2$, non è confrontabile con lo $0{,}65$ dell'euristica di
partenza, ed è un dettaglio che vale la pena tenere a mente prima di citare
«l'$\alpha$ di GNMT».

`````

```{figure} ../figures/beam-search.svg
:name: fig-beam-search
:alt: "Albero di beam search con larghezza due su tre passi: al primo passo restano nel fascio «A» (0,50) e «The» (0,40); al secondo «The black» (0,24) supera «A black» (0,15); al terzo l'ipotesi migliore è «The black cat» (0,19). I rami scartati sono in grigio tratteggiato."
:width: 100%

Beam search con $k=2$ sull'esempio del testo. Ogni numero è il punteggio della
strada intera fino a lì, cioè il prodotto di tutte le probabilità incontrate
lungo il cammino. La greedy si sarebbe fermata su «A» al primo passo; il
fascio recupera «The black cat».
```

In {numref}`fig-beam-search` i rami in terracotta sono le due strade tenute
aperte, quelli grigi tratteggiati i rami potati: il ramo spesso è la traduzione
che la strategia ingorda non avrebbe mai trovato.

## 2016: la traduzione neurale entra in produzione

```{figure} ../figures/traduzione-automatica-da-regole-a-llm.svg
:name: fig-paradigmi-traduzione
:alt: "Linea del tempo con i quattro paradigmi della traduzione automatica: i sistemi a regole scritte da linguisti, i metodi statistici che imparano da testi già tradotti da esseri umani, la traduzione neurale con encoder-decoder e attenzione, e infine i modelli linguistici generalisti che traducono senza essere stati costruiti per farlo."
:width: 100%

Quattro modi di tradurre, in settant'anni. A ogni passaggio si sposta chi
fornisce la conoscenza della lingua: prima il linguista che scrive le regole,
poi una montagna di testi già tradotti da esseri umani (un romanzo e la sua
traduzione, gli atti di un parlamento in due lingue), poi un modello addestrato
apposta, infine un modello che non era stato pensato per questo.
```

L'ultimo passaggio di {numref}`fig-paradigmi-traduzione` è il più singolare, e
il libro lo incontrerà nel capitolo sui Transformer: la traduzione ha smesso di
essere un compito con un'architettura propria ed è diventata una delle cose
che un modello generalista sa fare. Qui però siamo alla terza tappa, ed è
quella che ha portato la traduzione neurale in produzione.

Questa storia ha una data di consegna. Nel settembre 2016 Google annuncia GNMT
(*Google Neural Machine Translation*) {cite}`wu2016google`: un encoder–decoder
con l'attenzione, esattamente la ricetta di questa sezione, ma in grande: otto
**strati** di celle impilate per l'encoder e altrettanti per il decoder. L'idea
dell'impilamento è che il primo strato legge le parole, il secondo legge quello
che ha capito il primo, e così via, ogni piano un po' più astratto del
precedente.

Questa rete prende il posto del sistema che Google usava davvero, quello dietro
al bottone «traduci» che chiunque poteva premere: un sistema statistico che
lavorava a pezzi di frase, imparando da montagne di testi già tradotti quali
gruppi di parole si scambiano con quali. Aveva retto per un decennio. Si parte
dalla coppia cinese-inglese, circa 18 milioni di traduzioni al giorno.

E il miglioramento non è misurato con un punteggio calcolato da un programma,
ma da esseri umani. Google mette delle persone bilingui davanti alla stessa
frase tradotta dal vecchio sistema e dal nuovo, senza dire quale sia quale, e
chiede di dare un voto a ciascuna; poi confronta i voti. Nel mucchio ci mette
anche una traduzione fatta da un traduttore umano, che prende il voto più alto
di tutti ed è il metro di riferimento. Il risultato: della distanza che
separava il vecchio sistema dal traduttore umano, il nuovo ne recupera in media
il 60 per cento sulle principali coppie di lingue. Per la prima volta le reti
ricorrenti che abbiamo studiato traducono, ogni giorno, per centinaia di
milioni di persone.

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
- **BLEU** {cite}`papineni2002bleu` è precisione di $n$-grammi con *clipping*,
  frenata dalla *brevity penalty*: definito sul corpus, dipendente dal
  protocollo, cieco alla parafrasi. Nel 2014 la rete pura ($34{,}8$) supera il
  sistema statistico di riferimento ($33{,}3$) ma non lo stato dell'arte
  ($37{,}0$).
- Nel 2016 la traduzione neurale entra in produzione con GNMT; nel 2017
  *Attention Is All You Need* fa cadere la ricorrenza.
```
`````
