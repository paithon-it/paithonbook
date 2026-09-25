# Modelli di sequenza: da RNN ai Transformer

«Il gatto nero salta sul muro, e dopo un attimo lo scavalca.» Leggila da
sinistra a destra, una parola alla volta: quando arrivi a «lo», sai già che
cosa viene scavalcato, perché hai tenuto in memoria ciò che è venuto prima. Il
linguaggio funziona così: ogni parola prende senso dalla scia di quelle che la
precedono. «Il gatto nero salta sul
muro» non è un sacchetto di parole mescolabili a piacere: l'ordine *è* il
significato.

Le reti viste finora non sanno farlo. Sono fatte a strati, e i numeri le
attraversano da un capo all'altro senza tornare mai indietro: entra un blocco
di dati, esce una risposta, fine. Per questo si chiamano *feed-forward*, «che
vanno solo in avanti». Vogliono in ingresso sempre la stessa quantità di roba,
e fra una risposta e l'altra non si ricordano niente.

Un modello che vuole capire o generare testo deve invece fare due cose in più:
accettare una sequenza di lunghezza qualsiasi (le frasi non hanno tutte lo
stesso numero di parole) e portarsi dietro una memoria di quello che ha già
letto man mano che avanza. Tra gli anni Ottanta e il 2017 la risposta sono
state le reti ricorrenti, dalle prime reti con memoria fino alla vigilia dei
Transformer, l'architettura su cui oggi si costruiscono i grandi modelli
linguistici e a cui è dedicato il {doc}`capitolo che segue
</Transformers/overview>`.

## Le reti ricorrenti: una memoria che scorre nel tempo

L'idea delle **reti neurali ricorrenti** (RNN, *Recurrent Neural Network*) è
elegante: invece di guardare tutta la frase in un colpo solo, la rete la
percorre una parola alla volta e tiene da parte una fila di numeri, sempre
della stessa lunghezza, che aggiorna a ogni passo. Quella fila si chiama
**stato nascosto** ed è la sua memoria di lavoro: riassume «tutto ciò che ho
letto finora». Nascosto perché non è la risposta della rete, non si vede da
fuori: è un appunto che la rete tiene per sé.

Prima del disegno, due parole che ricorrono da qui in avanti. Il blocchetto di
conti che si ripete, quello che legge una parola e aggiorna l'appunto, si chiama
**cella**. Dentro la cella ci sono dei numeri regolabili, ed è con quelli che si
moltiplica tutto ciò che entra: si chiamano pesi, e sono ciò che la rete
impara. Sono anche tutto ciò che la rete possiede: quando si dice che un
modello «ha sette miliardi di parametri» si sta contando quei numeri lì.

La cella è anche l'unico pezzo di rete che esiste davvero, e per vederlo il modo
migliore è «srotolarla», come nel disegno che segue: si disegna una copia della
cella per ogni istante e si guarda lo stato passare di mano in mano. Le copie
sono un disegno, non delle reti: ce n'è una sola, riusata a ogni parola, con gli
stessi pesi al passo 3 e al passo tremila. Ecco perché una rete ricorrente resta
piccola anche su testi lunghissimi: allungare il testo allunga il disegno, non
la rete.

```{figure} ../figures/rnn-srotolata.svg
:name: fig-rnn-srotolata
:alt: La stessa cella ricorrente ripetuta ai passi t-1, t e t+1, con lo stato nascosto passato in avanti da una cella all'altra; a ogni passo entra una parola ed esce la scommessa sulla parola successiva.
:width: 95%

Una RNN «srotolata» nel tempo. È
sempre la stessa cella, applicata a ogni passo: riceve la parola di turno e
il riassunto di tutto quello che è venuto prima, e produce il riassunto
aggiornato più la sua scommessa ($\hat{y}$, col cappello che segna
ogni previsione: a seconda del
compito sarà la parola successiva, o l'etichetta di quella corrente). Le tre
copie del disegno sono tre momenti diversi, non tre pezzi diversi di rete.
```

`````{tab} Elementare

Un foglietto accanto al libro, su cui scrivi riga dopo riga un riassunto di ciò
che è successo finora. Per ogni nuova frase fai sempre lo stesso gesto: guardi
la frase, guardi il foglietto, e riscrivi il foglietto aggiornato. Il foglietto
è lo stato nascosto; il gesto che ripeti è la cella della RNN. È «la stessa
mano» che lavora a ogni riga, per questo la rete ha bisogno di pochi parametri
anche per testi lunghissimi: non impara un gesto diverso per ogni parola, ne
impara uno solo e lo riusa.

Quello che consegni però è un'altra cosa. A ogni riga, appena hai aggiornato il
foglietto, ci butti sopra un occhio e dici la tua, quale parola verrà o di che
cosa parla la frase. Quella è la risposta, e la ricavi dal foglietto con un
secondo gesto, più corto del primo. Il foglietto resta un appunto privato.

E la mano come impara a fare meglio? Ogni tanto ti fermi, confronti le risposte
che hai dato con quello che il libro diceva davvero, e ripercorri le righe
all'indietro per capire in quale punto il gesto ti ha portato fuori strada.
Ripercorrerle tutte vorrebbe dire tenere mille righe sotto gli occhi insieme, e
sul tavolo non ci stanno. Allora si lavora a blocchi, per esempio di trenta
righe. Correggi
la mano guardando quelle trenta, poi riparti dal foglietto così com'è, senza
più tornare su come ci sei arrivato.

Il prezzo di questa scorciatoia è chiaro, ed è meglio saperlo. Il foglietto non
viene azzerato al confine fra un blocco e il successivo: passa di là com'è, e
in avanti la lettura non si interrompe mai. A fermarsi al confine è la
correzione. Così la
mano non impara mai a legare una cosa di riga cinque con una di riga sessanta,
perché quando c'è da correggere quel legame riga cinque non è più sul tavolo.

`````

`````{tab} Superiore

A ogni passo temporale $t$ la cella combina l'input corrente $\mathbf{x}_t$ con
lo stato precedente $\mathbf{h}_{t-1}$ tramite una trasformazione lineare
seguita da una non linearità. È lo schema che Jeffrey Elman descrive nel 1990
{cite}`elman1990finding`, e che da allora si chiama **rete ricorrente
semplice**, scritto nella forma di oggi:

$$
\mathbf{h}_t = \tanh\!\left(\mathbf{W}_{hh}\,\mathbf{h}_{t-1} + \mathbf{W}_{xh}\,\mathbf{x}_t + \mathbf{b}_h\right),
\qquad
\hat{\mathbf{y}}_t = \mathrm{softmax}\!\left(\mathbf{W}_{hy}\,\mathbf{h}_t +
\mathbf{b}_y\right).
$$

Qui $\mathbf{h}_t \in \mathbb{R}^d$ è lo stato nascosto, con $d$ scelto da chi
costruisce la rete, $\mathbf{x}_t$ è l'input
al passo $t$ e $\hat{\mathbf{y}}_t$ la distribuzione sul vocabolario, mentre
$\mathbf{W}_{hh}, \mathbf{W}_{xh}, \mathbf{W}_{hy}$ sono
matrici di pesi e $\mathbf{b}_h, \mathbf{b}_y$ i bias. Senza la softmax quello
che esce sono i *logit*, i punteggi grezzi: è la forma che le librerie chiedono
in ingresso alla cross-entropia, ed è per questo che in PyTorch l'ultimo strato
di un classificatore si ferma un passo prima e la softmax non si scrive.
Il punto cruciale è che queste
matrici non dipendono da $t$: sono *condivise* su tutta la sequenza (*weight
sharing*). L'addestramento avviene con la *backpropagation through time*, cioè
la retropropagazione applicata alla rete srotolata.

Srotolare, però, ha un costo: la rete srotolata su una sequenza di mille passi
è una rete profonda mille strati, e per retropropagare bisogna tenere in
memoria tutte le attivazioni intermedie. Su un testo lungo, o su un flusso che
non finisce mai, la cosa non sta in piedi. Il rimedio si chiama **BPTT
troncato**: si spezza la sequenza in blocchi di lunghezza fissa (tipicamente
qualche decina di passi), si retropropaga dentro un blocco e si stacca lo
stato nascosto al confine, passandolo al blocco successivo come un valore
qualunque, senza la sua storia. In PyTorch è letteralmente una chiamata,
`h = h.detach()`, e per la LSTM, il cui stato è una coppia, la stessa cosa
scritta su tutti e due i pezzi: `h = tuple(s.detach() for s in h)`.

Il prezzo è dichiarato: il gradiente non attraversa mai il confine, quindi la
rete non può imparare dipendenze più lunghe del blocco. Lo stato in avanti
sì, continua a propagarsi e a portare informazione; è il segnale di
apprendimento che si ferma. Quando si legge che una ricorrente «fatica sulle
dipendenze lunghe», una parte del problema è matematica, ed è il gradiente che
svanisce; una parte è questa, cioè una scelta di ingegneria presa per far
entrare l'addestramento in memoria.

`````

## Quando la memoria si dissolve

Sulla carta una RNN potrebbe collegare la prima parola all'ultima. Nella
pratica fatica. Nella frase «Le chiavi che ho lasciato ieri sul tavolo della
cucina di mia nonna… sono sparite», per accordare il verbo la rete deve
ricordare «chiavi» attraverso una dozzina di parole. Più cresce la distanza,
più la memoria si sbiadisce.

`````{tab} Elementare

Torniamo al foglietto dei riassunti. A ogni riga lo riscrivi, e ogni riscrittura
perde un pochino dei dettagli vecchi per far posto a quelli nuovi. Dopo cento
riscritture, di cosa succedeva a pagina uno non resta quasi nulla.

Alla correzione va anche peggio. Per capire dove la mano ha sbagliato devi
risalire le righe una per una, e a ogni riga la correzione ripassa attraverso
la stessa riscrittura che all'andata aveva già smorzato i dettagli vecchi; si
accorcia un pochino, poi ancora, poi ancora. Bastano poche decine di righe
risalite e quando arriva in cima non sposta più niente. La mano si aggiusta
benissimo su quello che è appena successo, mentre su quello che è lontano non
riceve nessuna indicazione.

Le RNN soffrono esattamente di questo doppio smorzamento, e il modello finisce
per «dimenticare» ciò di cui avrebbe ancora bisogno. È il problema delle
**dipendenze a lungo termine**.

Ogni tanto capita il rovescio. Un gesto che a ogni riga ingrandisce invece di
smorzare fa crescere la correzione mentre risale, finché quello che arriva in
cima è uno strattone che scompone la mano invece di aggiustarla. Che il gesto
ingrandisca è necessario ma non basta: ingrandimenti in direzioni diverse,
passati uno dopo l'altro, possono anche compensarsi, e per sapere se lo
strattone arriva bisogna seguirli nell'ordine in cui si susseguono. Per lo
strattone il rimedio è semplice: se la correzione arriva più lunga di una
misura fissata, la si accorcia a quella misura prima di usarla, senza
cambiarne la direzione. Per lo smorzamento no: serve un foglio diverso.

`````

`````{tab} Superiore

Durante la *backpropagation through time*, il gradiente che risale da un passo
lontano si ottiene moltiplicando molte matrici jacobiane in cascata. Il termine
critico ha la forma

$$
\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-k}}
= \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}
\cdot \frac{\partial \mathbf{h}_{t-1}}{\partial \mathbf{h}_{t-2}}
\cdots \frac{\partial \mathbf{h}_{t-k+1}}{\partial \mathbf{h}_{t-k}} ,
$$

un prodotto di $k$ fattori, e l'ordine conta perché le jacobiane non commutano.
Per la rete di Elman ciascun fattore si scrive per esteso,
$\partial \mathbf{h}_s / \partial \mathbf{h}_{s-1} = \mathrm{diag}\big(1 - \mathbf{h}_s \odot \mathbf{h}_s\big)\,\mathbf{W}_{hh}$,
e poiché la derivata della $\tanh$ non supera $1$ vale
$\big\|\partial \mathbf{h}_t / \partial \mathbf{h}_{t-k}\big\| \le \|\mathbf{W}_{hh}\|^{k}$,
con $\|\cdot\|$ la norma spettrale: se il massimo valore singolare di
$\mathbf{W}_{hh}$ è minore di $1$ il gradiente svanisce in modo esponenziale
(*vanishing gradient*), e la diagonale, che tende a zero quando le unità
saturano, lo spinge ancora più giù. Il maggiorante non dice invece quando il
gradiente esplode: un valore singolare maggiore di $1$ è condizione necessaria,
non sufficiente {cite}`pascanu2013difficulty`. Per l'esplosione lo stesso lavoro
propone di tagliare la norma del gradiente (*gradient clipping*; Mikolov lo
faceva già, componente per componente): detto $\mathbf{g}$ il gradiente di tutti
i parametri messi in fila, se $\|\mathbf{g}\| > \tau$ si riscala
$\mathbf{g} \leftarrow \tau\,\mathbf{g}/\|\mathbf{g}\|$ (in PyTorch
`torch.nn.utils.clip_grad_norm_`), che cura l'esplosione e non tocca la
scomparsa. Che una rete ricorrente «semplice» faccia fatica su molti passi è un
risultato del 1994 di Yoshua Bengio, Patrice Simard e Paolo Frasconi
{cite}`bengio1994learning`, e va enunciato per quello che è: un compromesso, non
un divieto. Se la rete conserva l'informazione in modo robusto, cioè in modo che
un disturbo non la cancelli, allora il gradiente svanisce in modo esponenziale;
e quindi è la discesa del gradiente a non riuscire a trovare quei legami, non la
rete a non poterli rappresentare.

`````

## LSTM e GRU: cancelli per la memoria

La soluzione la propongono nel 1997 Sepp Hochreiter e Jürgen Schmidhuber
{cite}`hochreiter1997long`, e si chiama **LSTM** (*Long Short-Term Memory*,
memoria a breve termine lunga: il nome è un gioco di parole, ed è appunto una
memoria di lavoro che però dura).

L'intuizione è di dare alla cella due memorie invece di una. La prima è quella
che già c'era, il riassunto riscritto da capo a ogni passo. La seconda è un
taccuino protetto, che a ogni passo non viene riscritto: viene ritoccato,
con piccole aggiunte e piccole cancellature. Quello che ci si scrive resta lì
finché qualcuno non decide di toglierlo, e proprio per questo un'informazione
può sopravvivere a cento parole di distanza. A decidere che cosa scriverci e
che cosa leggerne sono due **cancelli**. Un cancello è un numero fra zero e uno
per cui si moltiplica ciò che passa: a uno lascia passare tutto, a zero niente,
nel mezzo una parte. Quel numero non è fissato una volta per tutte: lo calcola
la rete a ogni passo, dalla parola nuova e dallo stato precedente, con pesi che
si correggono come tutti gli altri.

Il terzo cancello, quello che decide che cosa cancellare, nel lavoro del
1997 non c'era: arriva tre anni dopo, con Felix Gers, Jürgen Schmidhuber e Fred
Cummins {cite}`gers2000learning`, e nasce da un difetto scoperto all'uso. Se il
taccuino non si azzera mai, su una sequenza che non finisce i numeri che ci
stanno scritti crescono senza fermarsi. Il guaio non è la carta che si esaurisce
ma la lettura: il valore che si legge dal taccuino passa per una funzione che
schiaccia i numeri grandi verso il suo estremo, e a quel punto non distingue più
un contenuto dall'altro. La cella smette di ricordare e torna a comportarsi come
una ricorrente qualunque. Su un testo che
finisce va bene lo stesso; su un flusso che non finisce mai (una trasmissione,
un sensore, una conversazione senza fine) è fatale. La forma con tre cancelli è
quella che oggi si chiama LSTM senz'altra specificazione, ed è quella che
segue.

```{figure} ../figures/lstm-gru-cancelli-memoria.svg
:name: fig-cella-lstm
:alt: "Schema interno di una cella LSTM: lo stato della cella attraversa il disegno da sinistra a destra come una linea quasi diretta; tre cancelli regolati da sigmoidi intervengono su di essa, il gate di dimenticanza che cancella, quello di ingresso che scrive e quello di uscita che decide cosa leggere verso lo stato nascosto."
:width: 88%

I tre cancelli della LSTM. La linea che attraversa la cella da parte a parte è
la memoria protetta: i cancelli la modificano poco per volta, invece di
riscriverla da capo a ogni passo.
```

Il dettaglio decisivo di {numref}`fig-cella-lstm` è la linea orizzontale che
passa da sinistra a destra quasi indisturbata: è lo stato di cella
$\mathbf{c}_t$, che da un passo all'altro viene soltanto moltiplicato per il
cancello che dimentica e sommato a una correzione, invece di passare ogni volta
per i pesi $\mathbf{W}_{hh}$ e per una $\tanh$ come lo stato della rete
semplice. Su quella strada il gradiente all'indietro attraversa prodotti per
numeri che la rete stessa tiene vicini a uno: cento moltiplicazioni per un
fattore fisso di $0{,}9$ lasciano $0{,}9^{100} \approx 0{,}000027$ del segnale,
meno di un trentamillesimo, mentre cento passaggi per un cancello spalancato ne
lasciano $1^{100} = 1$.

`````{tab} Elementare

Sul tavolo adesso ci sono due fogli. Il foglietto dei riassunti è quello di
prima, e a ogni riga lo riscrivi da capo. Accanto c'è il taccuino, e sul
taccuino non si riscrive niente. Ci aggiungi una riga, ne cancelli una, e tutto
il resto resta dov'è.

A regolare il traffico ci sono tre manopole, che si chiamano **gate**
(cancelli). La prima decide quanto di ciò che sta sul taccuino tenere e quanto
dimenticarne. Per la seconda prepari prima l'appunto per esteso, la frase che
scriveresti se dovessi scrivere tutto, e poi la manopola decide quanta parte
annotarne davvero. La terza decide quanto del taccuino mostrare sul foglietto.
Il taccuino fa da archivio, il foglietto è quello che si tiene sott'occhio per
rispondere e per leggere la riga dopo.

Le manopole si girano poco per volta, come il rubinetto dell'acqua. Di quanto
girarle lo decidi riga per riga, guardando la riga nuova e il foglietto, e a
deciderlo bene la rete ci arriva da sé. Se una cosa serve ancora («stiamo
parlando di *chiavi*, plurale»), la prima manopola resta spalancata, dal
taccuino non si cancella niente, e quella riga arriva intatta cento righe più
in là.

Ed è per la stessa ragione che la correzione, risalendo, arriva in cima. Sul
foglietto ogni riga ripassa per la stessa riscrittura, e cento riscritture che
conservano nove decimi lasciano meno di un trentamillesimo: $0{,}9$
moltiplicato per sé stesso cento volte fa $0{,}000027$. Sul taccuino la riga
vecchia non si riscrive, le si scrive accanto. Se scrivo $b = a + c$ e cambio
$a$ di un centesimo, $b$ cambia di un centesimo esatto; così la correzione che
risale lungo il taccuino attraversa cento aggiunte e arriva com'era partita.
L'unica moltiplicazione è la prima manopola, e finché resta spalancata
moltiplica per uno.

Una curiosità che dice qualcosa su come si fa ricerca: nella prima versione, del
1997, le manopole erano due, annota e mostra. Quella che dimentica sembrava
superflua (perché mai insegnare a una memoria a cancellarsi?) e fu aggiunta solo
tre anni dopo, quando ci si accorse che su un testo che non finisce mai il
taccuino si carica di segni sempre più calcati, uno sopra l'altro, finché a
leggerlo è tutto nero e una riga non si distingue più dall'altra. Saper
dimenticare,
si scoprì, è parte del saper ricordare.

Esiste anche una versione più snella della stessa idea, proposta nel 2014 da
Kyunghyun Cho e colleghi e chiamata **GRU** (*Gated Recurrent Unit*, «unità
ricorrente con i cancelli»: il nome descrive esattamente quello che è). Le
manopole sono due invece di tre, e il taccuino è lo stesso foglio del
promemoria, invece di essere separato. Una sola manopola decide insieme quanto
tenere del vecchio e quanto scrivere del nuovo, perché lo spazio è uno: quello
che lasci al vecchio lo togli al nuovo. L'altra decide quanto del vecchio
guardare mentre prepari l'appunto. Meno pezzi, meno numeri da imparare, e
spesso risultati altrettanto buoni. Le due sigle compaiono quasi sempre
appaiate, LSTM e GRU: sono due tagli dello stesso vestito.

`````

`````{tab} Superiore

La LSTM affianca allo stato nascosto $\mathbf{h}_t$ uno **stato di cella**
$\mathbf{c}_t$, la memoria a lungo termine. I gate sono vettori in $[0,1]$
prodotti da una sigmoide $\sigma$; nella formulazione del 1997 erano due,
*input* $\mathbf{i}_t$ e *output* $\mathbf{o}_t$, e la memoria si aggiornava
per pura addizione, senza poter mai essere svuotata: $\mathbf{c}_t =
\mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$, con
$\tilde{\mathbf{c}}_t$ la memoria candidata, definita poco più avanti. La
versione con il *forget gate* $\mathbf{f}_t$ {cite}`gers2000learning`, quella
che segue, è la forma canonica di oggi:

$$
\mathbf{f}_t = \sigma(\mathbf{W}_f[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_f), \quad
\mathbf{i}_t = \sigma(\mathbf{W}_i[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_i), \quad
\mathbf{o}_t = \sigma(\mathbf{W}_o[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_o).
$$

L'aggiornamento della memoria è quasi additivo, ed è questo a tenere vivo il
gradiente:

$$
\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t,
\qquad
\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t),
$$

dove
$\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_c)$
è la memoria candidata e $\odot$ è il prodotto elemento per elemento. Il perché
si legge nella derivata lungo la strada della memoria: trascurando la dipendenza
da $\mathbf{h}_{t-1}$ dei gate e della candidata,
$\partial \mathbf{c}_t / \partial \mathbf{c}_{t-1} = \mathrm{diag}(\mathbf{f}_t)$,
una diagonale che la rete porta vicino a $1$ quando deve ricordare, al posto del
fattore $\mathrm{diag}(1 - \mathbf{h}_s \odot \mathbf{h}_s)\,\mathbf{W}_{hh}$
della rete semplice, dove la matrice è la stessa a ogni passo e nessun cancello
tiene la diagonale vicina a $1$. Nel 1997 quel fattore era l'identità per
costruzione, il *constant error carousel* di Hochreiter e Schmidhuber; per la
stessa ragione conviene inizializzare a $1$ il bias di $\mathbf{f}_t$, che con i
pesi piccoli di partenza terrebbe il cancello intorno a $0{,}5$ e dimezzerebbe
il gradiente a ogni passo: Jozefowicz e colleghi lo raccomandano notando che
pochi lo facevano {cite}`jozefowicz2015empirical`, e `nn.LSTM` di PyTorch non lo
fa da sé. La **GRU** (*Gated Recurrent Unit*, {cite}`cho2014learning`) fonde
stato e memoria in un solo vettore e usa due gate, *update* $\mathbf{z}_t$ e
*reset* $\mathbf{r}_t$:

$$
\mathbf{z}_t = \sigma(\mathbf{W}_z[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_z), \quad
\mathbf{r}_t = \sigma(\mathbf{W}_r[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_r),
$$

$$
\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_h[\mathbf{r}_t \odot \mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_h), \qquad
\mathbf{h}_t = \mathbf{z}_t \odot \mathbf{h}_{t-1} + (1-\mathbf{z}_t) \odot \tilde{\mathbf{h}}_t .
$$

L'update gate fa insieme il lavoro del forget e dell'input della LSTM, legati in
modo da sommare a uno; il reset decide quanta parte dello stato passato entra
nella candidata. Con tre blocchi di pesi invece di quattro la GRU ha tre quarti
dei parametri della LSTM a parità di dimensione. I confronti sistematici non
danno un vincitore netto: il verdetto fra le due cambia con il compito e con
l'inizializzazione del forget gate {cite}`jozefowicz2015empirical`, e legare
forget e input come fa la GRU non peggiora la LSTM in modo significativo
{cite}`greff2017lstm`.

`````

## In pratica, con PyTorch

Tutta questa storia (cella, cancelli, stato nascosto) in PyTorch si condensa in
poche righe. Il compito che scegliamo per l'esempio è quello della {doc}`sezione
sulla
classificazione <classificazione-testo>`: leggere una recensione e dire se è
entusiasta o stroncatoria.

I tre pezzi del programma hanno i nomi delle cose di cui abbiamo appena
parlato: `nn.Embedding` è la tabella che trasforma ogni parola nella sua fila
di numeri, `nn.LSTM` è la cella con i suoi cancelli, `nn.Linear` è la bilancia
finale che dall'ultimo riassunto ricava il verdetto. E siccome le tre celle
disponibili, `nn.RNN`, `nn.LSTM` e `nn.GRU`, si usano tutte allo stesso modo,
per cambiarne una basta cambiare quella parola lì e rilanciare.

```python
import torch
from torch import nn

class ClassificatoreSentiment(nn.Module):
    def __init__(self, vocab=10000, dim=64, hidden=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim)  # parola -> vettore
        self.rnn = nn.LSTM(dim, hidden, batch_first=True)  # prova nn.RNN o nn.GRU
        self.out = nn.Linear(hidden, 2)            # 2 classi: negativo/positivo

    def forward(self, x):          # x: (batch, lunghezza), indici di parole
        e = self.embedding(x)      # (batch, lunghezza, dim)
        h, _ = self.rnn(e)         # stato nascosto a ogni passo
        return self.out(h[:, -1])  # ultimo passo -> logit per CrossEntropyLoss
```

Una cosa il blocco la dà per buona, e in un programma vero non lo è: `h[:, -1]`
prende l'ultimo posto della fila. Quando le frasi di un lotto hanno lunghezze
diverse le si allunga tutte alla stessa misura con dei riempitivi, e allora
l'ultimo posto è l'ultimo riempitivo, non l'ultima parola: lo stato che si legge
dipende da quanti se ne sono aggiunti. Con il riempimento si passa per
`pack_padded_sequence` (con `enforce_sorted=False` se il lotto non è ordinato
per lunghezza decrescente), e allora lo stato da leggere è il secondo valore che
la rete restituisce, `h_n` (per la LSTM la coppia `h_n, c_n`), che per ciascuna
frase si ferma alla sua ultima parola vera; va preso come `h_n[-1]`, perché la
sua prima dimensione sono gli strati anche con `batch_first=True`. `h[:, -1]`
sulla sequenza srotolata con `pad_packed_sequence` darebbe invece un vettore di
zeri per ogni frase più corta del lotto. Senza impacchettare, lo stato giusto si
prende all'indice della lunghezza vera meno uno di ciascuna frase.

Il ciclo di addestramento è quello che conosciamo dal {doc}`capitolo su PyTorch
</PyTorch/overview>`. E provare, come si è detto, costa una parola: si scambia
`nn.LSTM` con `nn.RNN` o con `nn.GRU`, e il resto del programma non cambia di
una riga. Su frasi lunghe LSTM e GRU battono quasi sempre la RNN semplice, e il
perché lo abbiamo appena visto: il taccuino protetto lascia arrivare il segnale
di ritorno anche da lontano, il foglietto riscritto da capo no.

## Il collo di bottiglia sequenziale

LSTM e GRU hanno dominato l'NLP per quasi un decennio: traduzione automatica,
riconoscimento vocale, generazione di testo. Ma restava un limite strutturale,
questa volta nel calcolo.

`````{tab} Elementare

Una RNN legge in ordine, come una persona: per calcolare il passo 100 deve
prima aver fatto il 99, che dipende dal 98, e così via. Non puoi «saltare
avanti». Su una frase lunga significa cento passi obbligatoriamente in fila,
uno dopo l'altro.

Perché è un guaio? Perché le macchine su cui girano queste reti sono fatte
apposta per il contrario. Una scheda grafica (la stessa che nel computer di
casa disegna i videogiochi, e che in gergo si chiama GPU) è brava a fare
*migliaia di conti facili tutti insieme*, più che un conto difficile.
Metterle davanti una rete ricorrente è come una catena di montaggio con una
postazione sola: per quanti operai tu abbia, devono aspettare il proprio turno,
e la fila non si accorcia.

La fila lunga si paga anche in un altro modo. Per legare quello che c'è alla
riga cento con quello che c'era alla riga uno, la correzione deve risalire
tutte le righe di mezzo, una per una. Le manopole tengono la strada aperta
molto meglio di un foglio riscritto da capo, ma cento passaggi restano cento
passaggi, e imparare un legame così lontano resta difficile.

La catena si potrebbe spezzare solo se ogni postazione facesse un gesto
semplicissimo e sempre dello stesso tipo, moltiplicare e aggiungere, senza
guardare il foglietto per decidere come: allora pezzi di catena diversi si
potrebbero montare in parallelo e poi attaccare in pochi passaggi. Con le
manopole che dipendono dal foglietto non si può, ed è la strada che
riprenderanno, molti anni dopo, i modelli a spazio di stati.

`````

`````{tab} Superiore

La ricorrenza $\mathbf{h}_t = f(\mathbf{h}_{t-1}, \mathbf{x}_t)$ è
intrinsecamente sequenziale: il calcolo su una sequenza di lunghezza $n$
richiede $O(n)$ passi che non possono essere parallelizzati lungo l'asse
temporale. Questo mal si sposa con le GPU, progettate per eseguire in parallelo
enormi moltiplicazioni tra matrici. Inoltre il segnale tra due token distanti
deve attraversare $O(n)$ celle, il che rende ancora arduo (pur mitigato dai
gate) l'apprendimento di dipendenze molto lunghe. Con stato di dimensione $d$ il
conto è $O(n\,d^2)$ operazioni in $O(n)$ passi sequenziali; l'autoattenzione
paga $O(n^2 d)$ operazioni, ma in $O(1)$ passi sequenziali e con un cammino di
lunghezza $O(1)$ fra due posizioni qualsiasi, e costa meno finché $n$ resta
sotto $d$ {cite}`vaswani2017attention`. La sequenzialità, del resto, è un
vincolo delle ricorrenze non lineari. Se l'aggiornamento è lineare in
$\mathbf{h}_{t-1}$, $\mathbf{h}_t = \mathbf{A}_t\mathbf{h}_{t-1} + \mathbf{b}_t$
con $\mathbf{A}_t$ e $\mathbf{b}_t$ che dipendono solo dall'ingresso, due passi
consecutivi si fondono in un passo della stessa forma,
$(\mathbf{A}_2\mathbf{A}_1,\ \mathbf{A}_2\mathbf{b}_1 + \mathbf{b}_2)$; la
fusione è associativa, e si calcola con una scansione parallela in $O(\log n)$
passi. Perché convenga, $\mathbf{A}_t$ si prende diagonale: con una matrice
piena ogni fusione costa $O(d^3)$. È la strada presa dai {doc}`modelli a spazio
di stati </StateSpaceModel/mamba>`.

`````

## Dove ci porta tutto questo

Le celle ricorrenti che abbiamo costruito qui sono i mattoni del passo
successivo: mettere due RNN una di fronte all'altra (una che legge, una che
scrive) e farle tradurre una frase intera. È la storia della {doc}`traduzione
con le reti <seq2seq-traduzione>`, ed è proprio lì, per rimediare ai limiti di
questa architettura, che
nascerà il meccanismo di attenzione: la possibilità, per ogni parola in
uscita, di tornare a guardare tutte le parole in ingresso e pesare da sola
quali contano.

Quell'idea si rivelerà così potente da fare, nel 2017, un passo ulteriore:
eliminare del tutto la ricorrenza e tenere solo l'attenzione; è la tesi di
*«Attention Is All You Need»* {cite}`vaswani2017attention`, il salto che ha
reso possibili i grandi modelli linguistici di oggi e a cui è dedicato un
intero capitolo. Le RNN, LSTM e GRU restano però fondamentali: sono il modo
più limpido per capire cosa significhi «memoria del contesto», e sopravvivono
dove i dati arrivano un pezzo alla volta e non finiscono mai (il segnale di un
sensore, l'audio di un microfono acceso) o dove il computer è piccolo e la
corrente poca.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Il testo è una sequenza: l'ordine è significato, e per capirlo serve
  ricordare quello che si è già letto.
- Il foglietto dei riassunti: una rete ricorrente legge una parola alla
  volta e a ogni parola riscrive un foglietto che riassume tutto il pregresso.
  È sempre la stessa mano a riscriverlo, e per questo la rete resta piccola
  anche su testi lunghissimi.
- Ogni riscrittura però perde un pochino del vecchio, e dopo cento righe di
  pagina uno non resta quasi niente; e la correzione, mentre risale le righe
  all'indietro, si accorcia a ogni passaggio finché in cima non sposta più
  niente. Sono le dipendenze lontane che si dissolvono.
- La LSTM affianca al foglietto un taccuino protetto, che non viene
  riscritto da capo a ogni passo ma solo ritoccato, e tre manopole che decidono
  quanto del taccuino dimenticare, quanta parte dell'appunto annotarci e quanto
  mostrarne sul foglietto: così un'informazione può restare intatta finché
  serve. Si girano poco per volta, come il rubinetto dell'acqua.
  Curiosamente quella che *dimentica* è arrivata tre anni dopo le altre due, ed
  è la più importante quando il testo non finisce mai. La GRU è la stessa
  idea in versione più snella, due manopole invece di tre e un foglio solo
  invece di due.
- Il limite che resta è di tempo: una rete ricorrente legge in fila, e per
  fare il passo cento deve aver fatto il novantanove. È una catena di montaggio
  con una postazione sola, e finché ogni gesto guarda il foglietto per decidere
  come riscriverlo non c'è computer che la possa mandare più veloce.
  La fila lunga si paga due volte, perché per legare la riga cento con la riga
  uno la correzione deve risalire tutte quelle di mezzo. È il collo di
  bottiglia che i Transformer toglieranno di mezzo.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Il testo è una sequenza: l'ordine è significato, e serve memoria del
  contesto.
- Una RNN riusa la stessa cella a ogni passo, facendo scorrere lo stato
  nascosto $\mathbf{h}_t$ nel tempo.
- Le RNN semplici perdono le dipendenze a lungo termine per due ragioni
  distinte: il gradiente che svanisce, che è matematica, e il BPTT
  troncato, che è ingegneria. Staccando lo stato al confine del blocco
  (`h.detach()`) il gradiente non lo attraversa, quindi un legame più lungo
  del blocco la rete non lo impara mai.
- LSTM e GRU introducono i gate, che decidono cosa ricordare e cosa
  dimenticare, proteggendo la memoria. L'architettura del 1997
  {cite}`hochreiter1997long` ne aveva due; il *forget gate* è del 2000
  {cite}`gers2000learning`.
- Il limite residuo è la sequenzialità delle ricorrenze non lineari (poca
  parallelizzazione): i Transformer la superano con l'attenzione, i modelli a
  spazio di stati rendendo lineare la ricorrenza.
```
`````
