# Modelli di sequenza: da RNN ai Transformer

Leggi questa frase da sinistra a destra, una parola alla volta. Quando arrivi
a «lo» o a «quello», la tua mente sa già a cosa si riferisce, perché ha tenuto
in memoria ciò che è venuto prima. Il linguaggio funziona così: ogni parola
prende senso dalla scia di quelle che la precedono. «Il gatto nero salta sul
muro» non è un sacchetto di parole mescolabili a piacere: l'ordine *è* il
significato.

Un modello che vuole capire o generare testo deve quindi fare due cose che una
rete come quelle viste finora non sa fare (le si chiama *feed-forward*, «che va
solo in avanti», perché il segnale le attraversa da un capo all'altro senza
tornare mai indietro): trattare l'input come una **sequenza** di lunghezza
variabile, e portarsi dietro una **memoria del contesto** man mano che avanza.
Questa sezione racconta come, tra gli anni Ottanta e il 2017, si è passati
dalle prime reti con memoria fino alla vigilia dei Transformer.

## Le reti ricorrenti: una memoria che scorre nel tempo

L'idea delle **reti neurali ricorrenti** (RNN, *Recurrent Neural Network*) è
elegante: invece di guardare tutta la frase in un colpo solo, la rete la
percorre un elemento alla volta e mantiene uno **stato nascosto** (un vettore
di numeri) che aggiorna a ogni passo. Quello stato è la sua memoria di lavoro:
riassume «tutto ciò che ho letto finora».

```{figure} ../figures/rnn-srotolata.svg
:name: fig-rnn-srotolata
:alt: La stessa cella ricorrente ripetuta ai passi t-1, t e t+1, con lo stato nascosto passato in avanti da una cella all'altra; a ogni passo entra un input ed esce una predizione.
:width: 95%

Una RNN «srotolata» nel tempo. È **sempre la stessa cella** (stessi pesi)
applicata a ogni passo: riceve la parola di turno e il riassunto di tutto
quello che è venuto prima, e produce il riassunto aggiornato più una
predizione. Nella notazione delle formule: entrano $\mathbf{x}_t$ e
$\mathbf{h}_{t-1}$, escono $\mathbf{h}_t$ e $\hat{\mathbf{y}}_t$.
```

Il modo migliore per capirla è «srotolarla», come in {numref}`fig-rnn-srotolata`:
disegniamo una copia della cella per ogni istante di tempo e vediamo lo stato
passare di mano in mano.

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
una parte del problema è matematica (il gradiente che svanisce, argomento della
prossima sezione) e una parte è questa, cioè una scelta di ingegneria presa per
far entrare l'addestramento in memoria.

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

La soluzione, proposta da Sepp Hochreiter e Jürgen Schmidhuber
{cite}`hochreiter1997long`, è la **LSTM** (*Long Short-Term Memory*).
L'intuizione: dare alla cella una memoria protetta (un anello in cui il segnale
gira senza attenuarsi, che gli autori chiamano *constant error carousel*) e due
cancelli che decidono che cosa scriverci dentro e che cosa leggerne.

Il terzo cancello, quello che decide che cosa **cancellare**, non c'era nel
lavoro del 1997 e arriva tre anni dopo, con Felix Gers, Jürgen Schmidhuber e
Fred Cummins {cite}`gers2000learning`. Nasce da un difetto scoperto all'uso, ed
è il rovescio esatto del problema di partenza: su un flusso che non finisce mai
(un testo che continua, un segnale che arriva senza sosta) una memoria che non
si azzera mai satura, e la cella smette di essere sensibile a qualunque novità.
La forma con tre cancelli è quella che oggi si chiama LSTM senz'altra
specificazione, ed è quella che segue.

```{figure} ../figures/lstm-gru-cancelli-memoria.svg
:name: fig-cella-lstm
:alt: "Schema interno di una cella LSTM: lo stato della cella attraversa il disegno da sinistra a destra come una linea quasi diretta; tre cancelli regolati da sigmoidi intervengono su di essa, il gate di dimenticanza che cancella, quello di ingresso che scrive e quello di uscita che decide cosa leggere verso lo stato nascosto."
:width: 88%

I tre cancelli della LSTM. La linea che attraversa la cella da parte a parte è
la memoria protetta: i cancelli la modificano poco per volta, invece di
riscriverla da capo a ogni passo.
```

Il dettaglio decisivo di {numref}`fig-cella-lstm` è la linea orizzontale che
passa da sinistra a destra quasi indisturbata, e conviene dire perché sia
decisiva. Una rete impara correggendo i propri numeri, e per correggerli deve
poter risalire all'indietro fino al punto in cui l'errore è nato: quel segnale
di ritorno si chiama **gradiente**, ed è il segnale che dice a ogni pezzo della
rete quanto e in che verso spostarsi. In una RNN semplice il riassunto viene
rimoltiplicato per gli stessi numeri a ogni passo, e ripetere una
moltiplicazione cento volte porta o a zero o all'infinito, come succede a
$0{,}9$ elevato a cento (quasi zero) e a $1{,}1$ elevato a cento (un numero
enorme): il segnale di ritorno o si spegne o esplode, e la rete non impara più
niente sulle cose lontane. Nella LSTM la strada principale è fatta invece di
**somme**, e su una somma il segnale di ritorno passa senza attenuarsi: può
viaggiare all'indietro per molti passi restando leggibile.

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

Tutta questa storia (cella ricorrente, gate, stato nascosto) in PyTorch si
condensa in un piccolo `nn.Module`. Le tre celle (`nn.RNN`, `nn.LSTM`,
`nn.GRU`) espongono la stessa interfaccia: si sostituiscono l'una all'altra
cambiando una sola parola.

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

Il training loop è quello che conosciamo dal capitolo su PyTorch, con
`nn.CrossEntropyLoss` sui logit. E passare da `nn.LSTM` a `nn.RNN` o `nn.GRU`
basta per confrontare le tre architetture sullo stesso problema: quasi sempre
LSTM e GRU battono la RNN semplice, proprio perché non «dimenticano» le
dipendenze lontane.

## Il collo di bottiglia sequenziale

LSTM e GRU hanno dominato l'NLP per quasi un decennio: traduzione automatica,
riconoscimento vocale, generazione di testo. Ma restava un limite strutturale,
non di memoria ma di **calcolo**.

`````{tab} Elementare

Una RNN legge in ordine, come una persona: per calcolare il passo 100 deve
prima aver fatto il 99, che dipende dal 98, e così via. Non puoi «saltare
avanti». Su una frase lunga significa cento passi obbligatoriamente in fila,
uno dopo l'altro. È come una catena di montaggio con una sola postazione:
per quanti operai (o schede grafiche) tu abbia, devono aspettare il proprio
turno.

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
là dove i dati arrivano in streaming o le risorse sono scarse.

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
  mostra) e una memoria protetta che non viene riscritta da capo a ogni passo:
  così un'informazione può restare intatta finché serve. Curiosamente
  l'interruttore che *dimentica* è arrivato tre anni dopo gli altri due, ed è
  il più importante quando il testo non finisce mai.
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
