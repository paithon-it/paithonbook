# Modelli di sequenza: da RNN ai Transformer

Leggi questa frase da sinistra a destra, una parola alla volta. Quando arrivi
a «lo» o a «quello», la tua mente sa già a cosa si riferisce, perché ha tenuto
in memoria ciò che è venuto prima. Il linguaggio funziona così: ogni parola
prende senso dalla scia di quelle che la precedono. «Il gatto nero salta sul
muro» non è un sacchetto di parole mescolabili a piacere: l'ordine *è* il
significato.

Le reti viste finora non sanno farlo, e conviene dire perché. Sono fatte a
strati, e i numeri le attraversano da un capo all'altro senza tornare mai
indietro: entra un blocco di dati, esce una risposta, fine. Per questo si
chiamano *feed-forward*, «che vanno solo in avanti». Vogliono in ingresso
sempre la stessa quantità di roba, e fra una risposta e l'altra non si ricordano
niente.

Un modello che vuole capire o generare testo deve invece fare due cose in più:
accettare una **sequenza** di lunghezza qualsiasi (le frasi non hanno tutte lo
stesso numero di parole) e portarsi dietro una **memoria di quello che ha già
letto** man mano che avanza. Questa sezione racconta come, tra gli anni Ottanta
e il 2017, si è passati dalle prime reti con memoria fino alla vigilia dei
Transformer, che sono l'architettura su cui oggi si costruiscono i grandi
modelli linguistici, e che hanno un capitolo tutto loro subito dopo questo.

## Le reti ricorrenti: una memoria che scorre nel tempo

L'idea delle **reti neurali ricorrenti** (RNN, *Recurrent Neural Network*) è
elegante: invece di guardare tutta la frase in un colpo solo, la rete la
percorre una parola alla volta e tiene da parte una fila di numeri, sempre
della stessa lunghezza, che aggiorna a ogni passo. Quella fila si chiama
**stato nascosto** ed è la sua memoria di lavoro: riassume «tutto ciò che ho
letto finora». Nascosto perché non è la risposta della rete, non si vede da
fuori: è un appunto che la rete tiene per sé.

Prima del disegno, due parole che ricorrono da qui in avanti. Il blocchetto di
conti che si ripete si chiama **cella**, ed è l'unico pezzo di rete che esiste
davvero. Dentro la cella ci sono dei numeri regolabili, ed è con quelli che si
moltiplica tutto ciò che entra: si chiamano **pesi**, e sono ciò che la rete
impara. Sono anche tutto ciò che la rete possiede: quando si dice che un
modello «ha sette miliardi di parametri» si sta contando quei numeri lì.

Ecco allora perché una rete ricorrente resta piccola anche su testi
lunghissimi: la cella è una sola, e i suoi pesi sono sempre gli stessi al passo
3 e al passo tremila. Il modo migliore per vederlo è «srotolarla», come nel
disegno qui sotto: si disegna una copia della cella per ogni istante e si
guarda lo stato passare di mano in mano.

```{figure} ../figures/rnn-srotolata.svg
:name: fig-rnn-srotolata
:alt: La stessa cella ricorrente ripetuta ai passi t-1, t e t+1, con lo stato nascosto passato in avanti da una cella all'altra; a ogni passo entra una parola ed esce la scommessa sulla parola successiva.
:width: 95%

Una RNN «srotolata» nel tempo. È
**sempre la stessa cella**, applicata a ogni passo: riceve la parola di turno e
il riassunto di tutto quello che è venuto prima, e produce il riassunto
aggiornato più la sua scommessa (qui indicata con $\hat{y}$: a seconda del
compito sarà la parola successiva, o l'etichetta di quella corrente). Le tre
copie del disegno sono tre momenti diversi, non tre pezzi diversi di rete.
```

`````{tab} Elementare

Immagina di leggere un libro tenendo accanto un foglietto su cui scrivi, riga
dopo riga, un riassunto di ciò che è successo finora. Per ogni nuova frase fai
sempre lo stesso gesto: guardi la frase, guardi il foglietto, e riscrivi il
foglietto aggiornato. Il foglietto è lo **stato nascosto**; il gesto che
ripeti è la cella della RNN. È «la stessa mano» che lavora a ogni riga, per
questo la rete ha bisogno di pochi parametri anche per testi lunghissimi: non
impara un gesto diverso per ogni parola, ne impara uno solo e lo riusa.

`````

`````{tab} Superiore

A ogni passo temporale $t$ la cella combina l'input corrente $\mathbf{x}_t$ con
lo stato precedente $\mathbf{h}_{t-1}$ tramite una trasformazione lineare
seguita da una non linearità:

$$
\mathbf{h}_t = \tanh\!\left(\mathbf{W}_{hh}\,\mathbf{h}_{t-1} + \mathbf{W}_{xh}\,\mathbf{x}_t + \mathbf{b}_h\right),
\qquad
\hat{\mathbf{y}}_t = \mathbf{W}_{hy}\,\mathbf{h}_t .
$$

Qui $\mathbf{h}_t \in \mathbb{R}^d$ è lo stato nascosto, $\mathbf{x}_t$ l'input
al passo $t$, mentre $\mathbf{W}_{hh}, \mathbf{W}_{xh}, \mathbf{W}_{hy}$ sono
matrici di pesi e $\mathbf{b}_h$ il bias. Il punto cruciale è che **queste
matrici non dipendono da $t$**: sono *condivise* su tutta la sequenza (*weight
sharing*). L'addestramento avviene con la *backpropagation through time*, cioè
la retropropagazione applicata alla rete srotolata.

Srotolare, però, ha un costo: la rete srotolata su una sequenza di mille passi
è una rete profonda mille strati, e per retropropagare bisogna tenere in
memoria tutte le attivazioni intermedie. Su un testo lungo, o su un flusso che
non finisce mai, la cosa non sta in piedi. Il rimedio si chiama **BPTT
troncato**: si spezza la sequenza in blocchi di lunghezza fissa (tipicamente
qualche decina di passi), si retropropaga dentro un blocco e **si stacca lo
stato nascosto** al confine, passandolo al blocco successivo come un valore
qualunque, senza la sua storia. In PyTorch è letteralmente una chiamata,
`h = h.detach()`.

Il prezzo è dichiarato ed è la ragione per cui vale la pena conoscerlo: il
gradiente non attraversa mai il confine, quindi **la rete non può imparare
dipendenze più lunghe del blocco**. Lo stato in avanti sì, continua a
propagarsi e a portare informazione; è il segnale di apprendimento che si
ferma. Quando si legge che una ricorrente «fatica sulle dipendenze lunghe»,
una parte del problema è matematica (il gradiente che svanisce, di cui parliamo
qui sotto) e una parte è questa, cioè una scelta di ingegneria presa per far
entrare l'addestramento in memoria.

`````

## Quando la memoria si dissolve

Sulla carta una RNN potrebbe collegare la prima parola all'ultima. Nella
pratica fatica. Nella frase «Le chiavi che ho lasciato ieri sul tavolo della
cucina di mia nonna… **sono** sparite», per accordare il verbo la rete deve
ricordare «chiavi» attraverso una dozzina di parole. Più cresce la distanza,
più la memoria si sbiadisce.

`````{tab} Elementare

Torniamo al foglietto dei riassunti. A ogni riga lo riscrivi, e ogni riscrittura
perde un pochino dei dettagli vecchi per far posto a quelli nuovi. Dopo cento
riscritture, di cosa succedeva a pagina uno non resta quasi nulla. Le RNN
soffrono esattamente di questo: le informazioni lontane nel tempo svaniscono,
e il modello «dimentica» ciò di cui avrebbe ancora bisogno. È il problema delle
**dipendenze a lungo termine**.

`````

`````{tab} Superiore

Durante la *backpropagation through time*, il gradiente che risale da un passo
lontano si ottiene moltiplicando molte matrici jacobiane in cascata. Il termine
critico ha la forma

$$
\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-k}}
= \prod_{i=t-k+1}^{t} \frac{\partial \mathbf{h}_i}{\partial \mathbf{h}_{i-1}} ,
$$

un prodotto di $k$ fattori. Se questi fattori hanno norma tipicamente minore di
1, il prodotto tende a $0$ in modo esponenziale (**gradiente che svanisce**,
*vanishing gradient*); se maggiore di 1, il gradiente *può* crescere fino a
esplodere. La norma dei fattori, infatti, dà solo un maggiorante del prodotto:
che sia maggiore di 1 è condizione necessaria perché il gradiente esploda, non
sufficiente {cite}`pascanu2013difficulty`. Che le RNN «semplici» non riescano ad apprendere dipendenze su
molti passi è un risultato dimostrato formalmente già nel 1994 da Yoshua
Bengio, Patrice Simard e Paolo Frasconi {cite}`bengio1994learning`.

`````

## LSTM e GRU: cancelli per la memoria

La soluzione la propongono nel 1997 Sepp Hochreiter e Jürgen Schmidhuber
{cite}`hochreiter1997long`, e si chiama **LSTM** (*Long Short-Term Memory*,
memoria a breve termine lunga: il nome è un gioco di parole, ed è appunto una
memoria di lavoro che però dura).

L'intuizione è di dare alla cella due memorie invece di una. La prima è quella
che già c'era, il riassunto riscritto da capo a ogni passo. La seconda è un
**taccuino protetto**, che a ogni passo non viene riscritto: viene ritoccato,
con piccole aggiunte e piccole cancellature. Quello che ci si scrive resta lì
finché qualcuno non decide di toglierlo, e proprio per questo un'informazione
può sopravvivere a cento parole di distanza. A decidere che cosa scriverci e
che cosa leggerne sono due **cancelli**, cioè due manopole che la rete impara
ad aprire e chiudere da sé.

Il terzo cancello, quello che decide che cosa **cancellare**, nel lavoro del
1997 non c'era: arriva tre anni dopo, con Felix Gers, Jürgen Schmidhuber e Fred
Cummins {cite}`gers2000learning`, e nasce da un difetto scoperto all'uso. Se il
taccuino si può solo scrivere e mai cancellare, prima o poi si riempie, e da
quel momento nessuna informazione nuova ci trova più posto. Su un testo che
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
passa da sinistra a destra quasi indisturbata, ed è quella il taccuino. Perché
sia decisiva richiede tre passaggi, e vale la pena farli.

**Primo: che cosa vuol dire «riscrivere il riassunto».** Fin qui l'abbiamo
detto a parole, ma dentro il computer quel riassunto è una fila di numeri. E
anche la parola nuova è una fila di numeri: prima di entrare nella rete, ogni
parola viene sostituita dalle sue coordinate sulla mappa dei significati, quelle
della sezione sugli embedding. Riscrivere il riassunto vuol dire allora
moltiplicarlo per i pesi della cella e sommarci la fila di numeri della parola
nuova. Non è una metafora: a ogni passo i numeri del riassunto vengono
letteralmente moltiplicati per gli stessi numeri, quelli della cella, che è
sempre la stessa.

**Secondo: perché ripetere una moltiplicazione fa danni.** Una rete impara
correggendo i propri pesi, e per correggerli deve poter risalire all'indietro
fino al punto in cui l'errore è nato. Quel segnale di ritorno si chiama
**gradiente**, e dice a ogni pezzo della rete quanto e in che verso spostarsi.
Tornando indietro di cento passi, però, il gradiente attraversa cento volte la
stessa moltiplicazione, e ripetere cento volte una moltiplicazione porta o a
zero o all'infinito: $0{,}9$ elevato a cento fa $0{,}000027$, e $1{,}1$ elevato
a cento fa quasi quattordicimila. Il segnale di ritorno o si spegne o esplode,
e la rete non impara più niente sulle cose lontane.

**Terzo: perché le somme salvano.** Nella LSTM la strada principale, quella del
taccuino, funziona per **aggiunte**: al passo dopo il taccuino è quello di
prima più una piccola correzione.

E su una somma il segnale di ritorno passa intero. Il perché si tocca con mano
con due numeri. Se scrivo $b = 0{,}9 \times a$ e poi cambio $a$ di un
centesimo, $b$ cambia di nove millesimi: il cambiamento è arrivato attenuato, e
ripetendo cento volte non arriva più niente, come abbiamo appena visto. Se
invece scrivo $b = a + c$ e cambio $a$ di un centesimo, $b$ cambia di **un
centesimo esatto**: la somma non tocca la parte che le passa attraverso, si
limita ad aggiungerle qualcosa accanto. Il segnale di ritorno funziona allo
stesso modo, all'incontrario: attraversando cento somme arriva com'era partito,
attraversando cento moltiplicazioni per $0{,}9$ arriva ridotto a meno di un
trentamillesimo.

Guardando la figura si vede che una moltiplicazione c'è anche lì, sulla
sinistra: è il cancello che dimentica, e serve appunto a cancellare. Ma è una
sola per passo, e soprattutto è **una manopola che la rete controlla**: se
quello che c'è scritto sul taccuino serve ancora, la rete tiene il cancello
spalancato, quella moltiplicazione è per uno, e non attenua niente. Nella RNN
semplice invece la moltiplicazione è obbligatoria e sempre la stessa, e nessuno
può disattivarla. È tutta qui la differenza fra riscrivere un foglio da capo e
annotare a margine.

`````{tab} Elementare

Immagina che il foglietto dei riassunti abbia ora tre interruttori. Il primo
decide quanto del vecchio riassunto **dimenticare**; il secondo quanto della
nuova frase **annotare**; il terzo quanto del riassunto **mostrare** in uscita
al passo successivo. Sono i **gate** (cancelli). Grazie a loro un'informazione
importante («stiamo parlando di *chiavi*, plurale») può restare intatta per
molte righe, finché serve, senza essere sovrascritta. La rete impara da sola
quando aprire e chiudere ogni interruttore.

Una curiosità che dice qualcosa su come si fa ricerca: nella prima versione, del
1997, gli interruttori erano due, annota e mostra. Il terzo, quello che
dimentica, sembrava superfluo (perché mai insegnare a una memoria a
cancellarsi?) e fu aggiunto solo tre anni dopo, quando ci si accorse che su un
testo che non finisce mai il foglietto si riempie e non c'è più spazio per
niente di nuovo. Saper dimenticare, si scoprì, è parte del saper ricordare.

Esiste anche una versione più snella della stessa idea, proposta nel 2014 da
Kyunghyun Cho e colleghi e chiamata **GRU** (*Gated Recurrent Unit*, «unità
ricorrente con i cancelli»: il nome descrive esattamente quello che è). Gli
interruttori sono due invece di tre, e il taccuino protetto non è separato dal
foglietto dei riassunti, è lo stesso foglio. Meno pezzi, meno numeri da
imparare, e nella maggior parte dei casi risultati paragonabili. Nel resto
della sezione le due sigle compaiono spesso appaiate, LSTM e GRU: sono due
tagli dello stesso vestito.

`````

`````{tab} Superiore

La LSTM affianca allo stato nascosto $\mathbf{h}_t$ uno **stato di cella**
$\mathbf{c}_t$, la memoria a lungo termine. I gate sono vettori in $[0,1]$
prodotti da una sigmoide $\sigma$; nella formulazione del 1997 erano due,
*input* $\mathbf{i}_t$ e *output* $\mathbf{o}_t$, e la memoria si aggiornava
per pura addizione, senza poter mai essere svuotata:
$\mathbf{c}_t = \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$,
con $\tilde{\mathbf{c}}_t$ la memoria candidata definita qui sotto. La versione
con il *forget gate* $\mathbf{f}_t$ {cite}`gers2000learning`, quella che segue,
è la forma canonica di oggi:

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
è la memoria candidata e $\odot$ è il prodotto elemento per elemento. La
**GRU** (*Gated Recurrent Unit*, {cite}`cho2014learning`) è una variante più
snella con due soli gate (*update* e *reset*) e nessuno stato di cella
separato: spesso rende quanto la LSTM con meno parametri.

`````

## In pratica, con PyTorch

Tutta questa storia (cella, cancelli, stato nascosto) in PyTorch si condensa in
poche righe. Il compito che scegliamo per l'esempio è quello della sezione sulla
classificazione: leggere una recensione e dire se è entusiasta o stroncatoria.

Chi non programma può leggere il blocco come si legge una ricetta, perché i tre
pezzi hanno i nomi delle cose di cui abbiamo appena parlato: `nn.Embedding` è la
tabella che trasforma ogni parola nella sua fila di numeri, `nn.LSTM` è la cella
con i suoi cancelli, `nn.Linear` è la bilancia finale che dall'ultimo riassunto
ricava il verdetto. E siccome le tre celle disponibili, `nn.RNN`, `nn.LSTM` e
`nn.GRU`, si usano tutte allo stesso modo, per cambiarne una basta cambiare
quella parola lì e rilanciare.

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

Il ciclo di addestramento è quello che conosciamo dal capitolo su PyTorch. E
provare, come si è detto, costa una parola: si scambia `nn.LSTM` con `nn.RNN` o
con `nn.GRU` e si guarda quante risposte esatte escono. Su frasi lunghe LSTM e
GRU battono quasi sempre la RNN semplice, e il perché lo abbiamo appena visto:
il taccuino protetto lascia arrivare il segnale di ritorno anche da lontano,
il foglietto riscritto da capo no.

## Il collo di bottiglia sequenziale

LSTM e GRU hanno dominato l'NLP per quasi un decennio: traduzione automatica,
riconoscimento vocale, generazione di testo. Ma restava un limite strutturale,
non di memoria ma di **calcolo**.

`````{tab} Elementare

Una RNN legge in ordine, come una persona: per calcolare il passo 100 deve
prima aver fatto il 99, che dipende dal 98, e così via. Non puoi «saltare
avanti». Su una frase lunga significa cento passi obbligatoriamente in fila,
uno dopo l'altro.

Perché è un guaio? Perché le macchine su cui girano queste reti sono fatte
apposta per il contrario. Una **scheda grafica** (la stessa che nel computer di
casa disegna i videogiochi, e che in gergo si chiama GPU) non è brava a fare un
conto difficile: è brava a fare *migliaia di conti facili tutti insieme*.
Metterle davanti una rete ricorrente è come una catena di montaggio con una
postazione sola: per quanti operai tu abbia, devono aspettare il proprio turno,
e la fila non si accorcia.

`````

`````{tab} Superiore

La ricorrenza $\mathbf{h}_t = f(\mathbf{h}_{t-1}, \mathbf{x}_t)$ è
intrinsecamente **sequenziale**: il calcolo su una sequenza di lunghezza $n$
richiede $O(n)$ passi che non possono essere parallelizzati lungo l'asse
temporale. Questo mal si sposa con le GPU, progettate per eseguire in parallelo
enormi moltiplicazioni tra matrici. Inoltre il segnale tra due token distanti
deve attraversare $O(n)$ celle, il che rende ancora arduo (pur mitigato dai
gate) l'apprendimento di dipendenze molto lunghe.

`````

## Dove ci porta tutto questo

Le celle ricorrenti che abbiamo costruito qui sono i mattoni del passo
successivo: mettere due RNN una di fronte all'altra (una che legge, una che
scrive) e fargli **tradurre una frase intera**. È la storia della prossima
sezione, ed è proprio lì, per rimediare ai limiti di questa architettura, che
nascerà il meccanismo di **attenzione**: la possibilità, per ogni parola in
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
- Il testo è una **sequenza**: l'ordine è significato, e per capirlo serve
  ricordare quello che si è già letto.
- Il **foglietto dei riassunti**: una rete ricorrente legge una parola alla
  volta e a ogni parola riscrive un foglietto che riassume tutto il pregresso.
  È sempre la stessa mano a riscriverlo, e per questo la rete resta piccola
  anche su testi lunghissimi.
- Ogni riscrittura però perde un pochino del vecchio, e dopo cento righe di
  pagina uno non resta quasi niente: sono le **dipendenze lontane** che si
  dissolvono.
- La **LSTM** aggiunge al foglietto tre interruttori (dimentica, annota,
  mostra) e un taccuino protetto che non viene riscritto da capo a ogni passo,
  solo ritoccato: così un'informazione può restare intatta finché serve.
  Curiosamente l'interruttore che *dimentica* è arrivato tre anni dopo gli
  altri due, ed è il più importante quando il testo non finisce mai. La **GRU**
  è la stessa idea in versione più snella, due interruttori invece di tre e un
  foglio solo invece di due.
- Il limite che resta non è di memoria ma di **tempo**: una rete ricorrente
  legge in fila, e per fare il passo cento deve aver fatto il novantanove. È
  una catena di montaggio con una postazione sola, e non c'è computer che la
  possa mandare più veloce. È il collo di bottiglia che i **Transformer**
  toglieranno di mezzo.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Il testo è una **sequenza**: l'ordine è significato, e serve **memoria del
  contesto**.
- Una **RNN** riusa la stessa cella a ogni passo, facendo scorrere lo stato
  nascosto $\mathbf{h}_t$ nel tempo.
- Le RNN semplici soffrono il **gradiente che svanisce**: dimenticano le
  dipendenze a lungo termine.
- **LSTM** e **GRU** introducono i **gate**, che decidono cosa ricordare e cosa
  dimenticare, proteggendo la memoria. L'architettura del 1997
  {cite}`hochreiter1997long` ne aveva due; il *forget gate* è del 2000
  {cite}`gers2000learning`.
- Il limite residuo è la **sequenzialità** (poca parallelizzazione): è ciò che
  i **Transformer**, con l'attenzione, superano.
```
`````
