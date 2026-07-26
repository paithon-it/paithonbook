# Modelli di sequenza: da RNN ai Transformer

Leggi questa frase da sinistra a destra, una parola alla volta. Quando arrivi
a «lo» o a «quello», la tua mente sa già a cosa si riferisce, perché ha tenuto
in memoria ciò che è venuto prima. Il linguaggio funziona così: ogni parola
prende senso dalla scia di quelle che la precedono. «Il gatto nero salta sul
muro» non è un sacchetto di parole mescolabili a piacere: l'ordine *è* il
significato.

Un modello che vuole capire o generare testo deve quindi fare due cose che una
comune rete *feed-forward* non sa fare: trattare l'input come una **sequenza**
di lunghezza variabile, e portarsi dietro una **memoria del contesto** man mano
che avanza. Questo capitolo racconta come, tra gli anni Ottanta e il 2017, si è
passati dalle prime reti con memoria fino alla vigilia dei Transformer.

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
applicata a ogni passo: riceve l'input $x_t$ e lo stato precedente $h_{t-1}$,
produce il nuovo stato $h_t$ e una predizione $\hat{y}_t$.
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

A ogni passo temporale $t$ la cella combina l'input corrente $x_t$ con lo stato
precedente $h_{t-1}$ tramite una trasformazione lineare seguita da una non
linearità:

$$
h_t = \tanh\!\left(W_{hh}\,h_{t-1} + W_{xh}\,x_t + b_h\right),
\qquad
\hat{y}_t = W_{hy}\,h_t .
$$

Qui $h_t \in \mathbb{R}^d$ è lo stato nascosto, $x_t$ l'input al passo $t$,
mentre $W_{hh}, W_{xh}, W_{hy}$ sono matrici di pesi e $b_h$ il bias. Il punto
cruciale è che **queste matrici non dipendono da $t$**: sono *condivise* su
tutta la sequenza (*weight sharing*). L'addestramento avviene con la
*backpropagation through time*, cioè la retropropagazione applicata alla rete
srotolata.

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
\frac{\partial h_t}{\partial h_{t-k}}
= \prod_{i=t-k+1}^{t} \frac{\partial h_i}{\partial h_{i-1}} ,
$$

un prodotto di $k$ fattori. Se questi fattori hanno norma tipicamente minore di
1, il prodotto tende a $0$ in modo esponenziale (**gradiente che svanisce**,
*vanishing gradient*); se maggiore di 1, esplode. Bengio, Simard e Frasconi lo
dimostrarono formalmente nel 1994: le RNN «semplici» non riescono ad apprendere
dipendenze su molti passi.

`````

## LSTM e GRU: cancelli per la memoria

La soluzione, proposta da Sepp Hochreiter e Jürgen Schmidhuber
{cite}`hochreiter1997long`, è la **LSTM** (*Long Short-Term Memory*).
L'intuizione: dare alla cella una memoria protetta, e insegnarle a decidere
(con dei «cancelli») cosa scrivere, cosa cancellare e cosa leggere.

`````{tab} Elementare

Immagina che il foglietto dei riassunti abbia ora tre interruttori. Il primo
decide quanto del vecchio riassunto **dimenticare**; il secondo quanto della
nuova frase **annotare**; il terzo quanto del riassunto **mostrare** in uscita
al passo successivo. Sono i **gate** (cancelli). Grazie a loro un'informazione
importante («stiamo parlando di *chiavi*, plurale») può restare intatta per
molte righe, finché serve, senza essere sovrascritta. La rete impara da sola
quando aprire e chiudere ogni interruttore.

`````

`````{tab} Superiore

La LSTM affianca allo stato nascosto $h_t$ uno **stato di cella** $c_t$, la
memoria a lungo termine. Tre gate (*forget* $f_t$, *input* $i_t$, *output*
$o_t$) sono vettori in $[0,1]$ prodotti da una sigmoide $\sigma$:

$$
f_t = \sigma(W_f[h_{t-1},x_t]+b_f), \quad
i_t = \sigma(W_i[h_{t-1},x_t]+b_i), \quad
o_t = \sigma(W_o[h_{t-1},x_t]+b_o).
$$

L'aggiornamento della memoria è quasi additivo, ed è questo a tenere vivo il
gradiente:

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t,
\qquad
h_t = o_t \odot \tanh(c_t),
$$

dove $\tilde{c}_t = \tanh(W_c[h_{t-1},x_t]+b_c)$ è la memoria candidata e
$\odot$ è il prodotto elemento per elemento. La **GRU** (*Gated Recurrent
Unit*, {cite}`cho2014learning`) è una variante più snella con due soli gate
(*update*
e *reset*) e nessuno stato di cella separato: spesso rende quanto la LSTM con
meno parametri.

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

La ricorrenza $h_t = f(h_{t-1}, x_t)$ è intrinsecamente **sequenziale**: il
calcolo su una sequenza di lunghezza $n$ richiede $O(n)$ passi che non possono
essere parallelizzati lungo l'asse temporale. Questo mal si sposa con le GPU,
progettate per eseguire in parallelo enormi moltiplicazioni tra matrici.
Inoltre il segnale tra due token distanti deve attraversare $O(n)$ celle, il
che rende ancora arduo (pur mitigato dai gate) l'apprendimento di dipendenze
molto lunghe.

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

```{admonition} Da ricordare
:class: important
- Il testo è una **sequenza**: l'ordine è significato, e serve **memoria del
  contesto**.
- Una **RNN** riusa la stessa cella a ogni passo, facendo scorrere lo stato
  nascosto $h_t$ nel tempo.
- Le RNN semplici soffrono il **gradiente che svanisce**: dimenticano le
  dipendenze a lungo termine.
- **LSTM** e **GRU** introducono i **gate**, che decidono cosa ricordare e cosa
  dimenticare, proteggendo la memoria.
- Il limite residuo è la **sequenzialità** (poca parallelizzazione): è ciò che
  i **Transformer**, con l'attenzione, superano.
```
