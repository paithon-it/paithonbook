# Message passing: il cuore delle GNN

C'è un gioco che tutti conosciamo: il *telefono senza fili*. Una persona
sussurra una frase all'orecchio del vicino, quello la ripete al suo, e così
via. Cambiamo le regole. Invece di ascoltare *un* vicino, ognuno ascolta
*tutti* i propri vicini in una volta sola, riassume ciò che ha sentito e si fa
un'idea aggiornata; poi si ricomincia. Dopo un giro, ogni persona sa qualcosa
dei suoi amici diretti. Dopo due giri, anche degli amici degli amici. Dopo
$K$ giri, la voce partita da un capo della rete è arrivata a chi sta a $K$
strette di mano di distanza.

Questo passaparola a giri è, alla lettera, il modo in cui una rete neurale su
grafo elabora l'informazione. La sezione «Il mondo come grafo» ha messo in
forma il dato — la matrice di adiacenza $A$, la matrice delle feature dei nodi
$X$, la matrice diagonale dei gradi $D$, e la versione con i cappi
$\tilde{A} = A + I$. L'introduzione al capitolo gli ha dato un nome:
**message passing**, «scambio di messaggi». Qui lo apriamo: prima nella sua
forma generale, poi nella sua incarnazione più usata, la *Graph Convolutional
Network*, che ricaveremo passo dopo passo e faremo girare, coi numeri alla
mano, su un grafo minuscolo.

## Un nodo, i suoi vicini, tre mosse

L'idea si regge su un'operazione sola, ripetuta a ogni giro e uguale per ogni
nodo: **guarda i vicini, riassumili, aggiornati**. La {numref}`fig-message-passing`
la mostra tutta in un colpo d'occhio.

```{figure} ../figures/message-passing.svg
:name: fig-message-passing
:alt: Un nodo centrale riceve messaggi dai vicini; i messaggi confluiscono in un blocco «aggrega» e poi in un blocco «aggiorna» che, insieme allo stato precedente del nodo, produce il nuovo stato. In basso, due pannelli mostrano il campo recettivo che passa da un salto (un strato) a due salti (due strati).
:width: 100%

Un passo di message passing su un nodo $v$: i **messaggi** dei vicini si
**aggregano** con una funzione che non dipende dal loro ordine, poi la mossa di
**aggiornamento** li fonde con lo stato precedente di $v$. In basso: impilando
due strati, il campo recettivo cresce dai vicini diretti ai vicini dei vicini.
```

`````{tab} Elementare

Pensa a ogni nodo come a una persona con una scheda su cui scrive «chi sono».
A ogni giro fa tre cose, sempre nell'ordine. **Primo**, ascolta: ogni amico gli
passa un bigliettino con scritto ciò che c'è sulla propria scheda — sono i
*messaggi*. **Secondo**, mette insieme i bigliettini in un unico riassunto. E
qui c'è un dettaglio importante: il riassunto non deve dipendere dall'ordine in
cui arrivano i bigliettini, perché tra amici non c'è un «primo» e un «ultimo».
Un riassunto che va bene è la **somma**, o la **media**: cambi l'ordine degli
addendi e il totale non cambia. **Terzo**, aggiorna la propria scheda mettendo
insieme il riassunto degli amici e quello che già sapeva di sé.

Fatto questo per tutti i nodi, il giro è finito e se ne può fare un altro. È lo
stesso identico meccanismo per ogni persona della rete: nessuno ha una regola
speciale. Proprio come nella convoluzione delle immagini, dove lo stesso
piccolo filtro scorre su tutti i pixel — solo che qui i «vicini» non sono i
quattro pixel accanto, ma gli amici sul grafo, che possono essere due o dieci.

`````

`````{tab} Superiore

Il quadro generale è la *Message Passing Neural Network* (MPNN) di Gilmer e
colleghi {cite}`gilmer2017neural`, che unifica sotto un'unica notazione quasi
tutte le GNN. Sia $h_v^{(k)}$ il vettore di stato del nodo $v$ dopo $k$ giri,
con $h_v^{(0)} = x_v$ (la sua feature iniziale). Un passo si scrive in due mosse:

$$
m_v^{(k)} = \bigoplus_{u \in \mathcal{N}(v)}
   M_k\!\big(h_v^{(k-1)},\, h_u^{(k-1)},\, e_{vu}\big),
\qquad
h_v^{(k)} = U_k\!\big(h_v^{(k-1)},\, m_v^{(k)}\big).
$$

Qui $\mathcal{N}(v)$ è l'insieme dei vicini di $v$; $M_k$ è la **funzione
messaggio** (una rete, che può usare anche la feature dell'arco $e_{vu}$); il
simbolo $\bigoplus$ è l'**aggregazione**, un'operazione *invariante alla
permutazione* dei vicini — tipicamente $\sum$, la media o il massimo — che
produce il messaggio aggregato $m_v^{(k)}$; e $U_k$ è la **funzione di
aggiornamento** che fonde lo stato precedente con $m_v^{(k)}$. Dopo $K$ passi,
per un compito sull'intero grafo si applica una funzione di lettura
($\mathrm{READOUT}$), anch'essa invariante alla permutazione,
$\hat{y}_G = R\big(\{\, h_v^{(K)} : v \in V \,\}\big)$.

L'invarianza di $\bigoplus$ è ciò che garantisce l'**equivarianza alla
permutazione** anticipata nell'introduzione: rinumerare i nodi non cambia i
messaggi, perché una somma non ha un primo addendo. Ed è la stessa forma
astratta — «aggrega dai vicini, poi aggiorna» — dello schema
$\mathrm{AGGREGATE}$/$\mathrm{UPDATE}$ visto nell'overview del capitolo, qui
resa esplicita nelle sue tre componenti apprendibili.

`````

## Dalla formula alla matrice: la GCN

La MPNN è un telaio: per avere un modello concreto bisogna scegliere le tre
mosse. La scelta più celebre — semplice, veloce, e ancora oggi il primo modello
che si prova su un grafo — è la **Graph Convolutional Network** (GCN),
presentata nel 2017 da Thomas Kipf e Max Welling {cite}`kipf2017semi`. La sua
regola di propagazione, da uno strato al successivo, sta in una riga:

$$
H^{(l+1)} = \sigma\!\left( \hat{A}\, H^{(l)}\, W^{(l)} \right),
\qquad
\hat{A} = \tilde{D}^{-1/2}\, \tilde{A}\, \tilde{D}^{-1/2}.
$$

Ogni simbolo ha un ruolo preciso:

- $H^{(l)} \in \mathbb{R}^{n \times d_l}$ raccoglie, riga per riga, gli stati
  di tutti i nodi allo strato $l$; si parte da $H^{(0)} = X$, le feature
  d'ingresso.
- $\tilde{A} = A + I$ è l'adiacenza con i **cappi** (*self-loop*): aggiungere
  la matrice identità $I$ mette ogni nodo tra i propri vicini, così che
  nell'aggregazione un nodo tenga conto anche di sé stesso e non dimentichi la
  propria feature.
- $\tilde{D}$ è la matrice diagonale dei gradi di $\tilde{A}$, cioè
  $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$ (il numero di vicini del nodo $i$,
  più uno per il cappio).
- $\hat{A} = \tilde{D}^{-1/2}\,\tilde{A}\,\tilde{D}^{-1/2}$ è l'adiacenza
  **normalizzata in modo simmetrico**: il pezzo che pesa i messaggi.
- $W^{(l)}$ è la matrice dei pesi appresi dello strato — la stessa per tutti i
  nodi, come il filtro di una CNN — e $\sigma$ una non-linearità (di solito la
  ReLU).

Letta nodo per nodo, la riga matriciale dice esattamente «aggrega, poi
aggiorna»:

$$
h_v^{(l+1)} = \sigma\!\left(
   \sum_{u \in \mathcal{N}(v)\cup\{v\}}
   \frac{1}{\sqrt{\tilde{d}_v\,\tilde{d}_u}}\; W^{(l)\top} h_u^{(l)}
\right),
$$

dove $\tilde{d}_v$ è il grado di $v$ in $\tilde{A}$. Il messaggio del vicino
$u$ è la sua feature trasformata da $W^{(l)}$; l'aggregazione è una **somma
pesata**, con pesi fissi $1/\sqrt{\tilde{d}_v\,\tilde{d}_u}$; l'aggiornamento è
la non-linearità $\sigma$. È un caso particolare della MPNN in cui la funzione
messaggio è lineare e l'aggregazione è la somma normalizzata.

### Perché normalizzare così

Resta la domanda che dà sostanza a tutta la formula: perché non sommare e basta
i vicini, ma dividere per $\sqrt{\tilde{d}_v\,\tilde{d}_u}$?

`````{tab} Elementare

Immagina un'aggregazione che somma e basta, senza dividere. Un nodo con dieci
amici riceve dieci bigliettini e li somma: un numerone. Un nodo con due amici
ottiene un numero piccolo. Dopo qualche giro, i nodi «popolari» hanno valori
enormi e quelli isolati valori minuscoli — non perché contino di più, ma solo
perché hanno più connessioni. La rete finirebbe per confondere «essere
importante» con «avere tanti amici».

La divisione mette tutti sulla stessa scala. È come fare una **media** invece
di una somma: dieci opinioni o due, quello che conta è il tenore, non il
numero. In più, il messaggio di un amico molto popolare pesa un po' meno,
perché la sua attenzione è «spalmata» su tanti: proprio come il consiglio di
chi conosce mezzo mondo vale un filo meno di quello dell'amico che hai solo tu.

`````

`````{tab} Superiore

Ci sono due normalizzazioni naturali. Quella per righe,
$\tilde{D}^{-1}\tilde{A}$, fa la **media** dei vicini: ogni riga somma a $1$, è
la matrice di transizione di una passeggiata aleatoria. La GCN usa invece
quella **simmetrica**, $\hat{A} = \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$,
in cui il peso dell'arco $(v,u)$ è $1/\sqrt{\tilde{d}_v\,\tilde{d}_u}$: si
sconta il grado di *entrambi* gli estremi. Il vantaggio è duplice.

Primo, la **scala**. Senza normalizzazione, moltiplicare ripetutamente per
$\tilde{A}$ amplifica i valori sui nodi ad alto grado in modo incontrollato;
gli autovalori di $\hat{A}$ stanno invece nell'intervallo $[-1, 1]$, così
impilare molti strati non fa esplodere né svanire il segnale — è il collegamento
diretto con il problema dei gradienti nelle reti profonde, discusso nella
sezione sulla backpropagation. Kipf e Welling la chiamano *renormalization
trick*: passare da $D^{-1/2}(A+I)D^{-1/2}$ con la $D$ «sbagliata» alla forma
coerente $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ con $\tilde{D}$ calcolata
su $\tilde{A}$, cappi inclusi.

Secondo, l'**origine spettrale**. La GCN nasce come approssimazione al
prim'ordine di una convoluzione definita nel dominio spettrale del grafo — i
filtri polinomiali di Čebyšëv di Defferrard, Bresson e Vandergheynst
{cite}`defferrard2016convolutional`. Troncare quel polinomio al primo grado e
riordinare i termini restituisce esattamente $\hat{A}$: è da qui che la
normalizzazione simmetrica «cade» naturalmente, non è una scelta arbitraria.
Rispetto al modello originale di Scarselli e colleghi
{cite}`scarselli2009graph`, che iterava fino a un punto fisso, la GCN fissa un
numero piccolo di strati e si addestra come una qualunque rete profonda.

`````

## Un passo di propagazione, coi numeri

Vale più di mille formule vedere i conti tornare. Prendiamo un grafo a quattro
nodi disposti in catena — $1 - 2 - 3 - 4$ — con una feature scalare per nodo,
$X = (1,\, 2,\, 3,\, 4)^\top$. Calcoliamo un passo di GCN a mano, scegliendo
$W = I$ e $\sigma = \text{identità}$ per isolare l'effetto della sola
propagazione $\hat{A}\,H$.

La matrice di adiacenza e quella con i cappi ($\tilde{A} = A + I$) sono

$$
A = \begin{bmatrix}
0 & 1 & 0 & 0 \\
1 & 0 & 1 & 0 \\
0 & 1 & 0 & 1 \\
0 & 0 & 1 & 0
\end{bmatrix},
\qquad
\tilde{A} = \begin{bmatrix}
1 & 1 & 0 & 0 \\
1 & 1 & 1 & 0 \\
0 & 1 & 1 & 1 \\
0 & 0 & 1 & 1
\end{bmatrix}.
$$

Sommando le righe di $\tilde{A}$ otteniamo i gradi con cappio,
$\tilde{d} = (2,\, 3,\, 3,\, 2)$ — i due nodi di bordo hanno un vicino, i due
interni ne hanno due, più il cappio. Dunque

$$
\tilde{D} = \mathrm{diag}(2,3,3,2),
\qquad
\tilde{D}^{-1/2} = \mathrm{diag}\!\left(
\tfrac{1}{\sqrt{2}},\, \tfrac{1}{\sqrt{3}},\,
\tfrac{1}{\sqrt{3}},\, \tfrac{1}{\sqrt{2}} \right)
\approx \mathrm{diag}(0{,}707,\ 0{,}577,\ 0{,}577,\ 0{,}707).
$$

Ogni entrata dell'adiacenza normalizzata è
$\hat{A}_{vu} = \tilde{A}_{vu} / \sqrt{\tilde{d}_v\,\tilde{d}_u}$. Per esempio
$\hat{A}_{12} = 1/\sqrt{2\cdot 3} = 1/\sqrt{6} \approx 0{,}408$ e
$\hat{A}_{22} = 1/\sqrt{3\cdot 3} = 1/3 \approx 0{,}333$. La matrice completa è

$$
\hat{A} = \tilde{D}^{-1/2}\,\tilde{A}\,\tilde{D}^{-1/2} \approx
\begin{bmatrix}
0{,}500 & 0{,}408 & 0 & 0 \\
0{,}408 & 0{,}333 & 0{,}333 & 0 \\
0 & 0{,}333 & 0{,}333 & 0{,}408 \\
0 & 0 & 0{,}408 & 0{,}500
\end{bmatrix}.
$$

È simmetrica, come dev'essere. Il passo di propagazione è
$H' = \hat{A}\,X$, cioè per ogni nodo la somma pesata di sé e dei suoi vicini:

$$
\begin{aligned}
h'_1 &= 0{,}500\cdot 1 + 0{,}408\cdot 2 = 0{,}500 + 0{,}816 = 1{,}316, \\
h'_2 &= 0{,}408\cdot 1 + 0{,}333\cdot 2 + 0{,}333\cdot 3 = 0{,}408 + 0{,}667 + 1{,}000 = 2{,}075, \\
h'_3 &= 0{,}333\cdot 2 + 0{,}333\cdot 3 + 0{,}408\cdot 4 = 0{,}667 + 1{,}000 + 1{,}633 = 3{,}300, \\
h'_4 &= 0{,}408\cdot 3 + 0{,}500\cdot 4 = 1{,}225 + 2{,}000 = 3{,}225.
\end{aligned}
$$

Il risultato è $H' \approx (1{,}316,\, 2{,}075,\, 3{,}300,\, 3{,}225)^\top$, e
racconta bene cosa fa la GCN: i valori si **lisciano** verso la media locale. Il
nodo 4, che valeva $4$, scende a $3{,}225$ perché è tirato in basso dal vicino
3; il nodo 1, che valeva $1$, sale a $1{,}316$ perché è tirato in alto dal
vicino 2. Un passo di message passing avvicina ogni nodo al proprio vicinato:
è il cuore differenziabile su cui la rete costruisce, strato dopo strato,
rappresentazioni sempre più ricche.

## Impilare gli strati: il campo recettivo a $K$ salti

Un solo strato di GCN fa vedere a ogni nodo i suoi vicini diretti. Ma il bello
comincia impilandone di più.

`````{tab} Elementare

Torniamo alla catena $1 - 2 - 3 - 4$ e mettiamoci nei panni del nodo 1. Al
primo giro parla col nodo 2, il suo unico vicino. Ma attenzione: nello stesso
giro, anche il nodo 2 ha parlato col nodo 3. Così, al **secondo** giro, quando
il nodo 1 riascolta il nodo 2, dentro il nodo 2 c'è già un pezzo di nodo 3.
Senza essersi mai «visti» direttamente, l'informazione del nodo 3 è arrivata al
nodo 1 in due passi. Al terzo giro arriverebbe anche quella del nodo 4.

È esattamente ciò che succede in una rete convoluzionale, dove impilando i
livelli ogni neurone «vede» una porzione via via più grande dell'immagine: il
suo *campo recettivo* cresce con la profondità. Sul grafo vale la stessa legge,
contata in **salti**: con $K$ strati, ogni nodo raccoglie informazione da tutto
ciò che sta entro $K$ passi da lui. La striscia in basso nella
{numref}`fig-message-passing` mostra proprio questo salto da uno a due.

`````

`````{tab} Superiore

Impilare $K$ strati di GCN corrisponde ad applicare $K$ volte l'operatore di
propagazione: lo stato finale $h_v^{(K)}$ dipende da tutti i nodi $u$ per cui
esiste un cammino di lunghezza $\le K$ fino a $v$ — il **campo recettivo** a
$K$ salti, l'analogo esatto del campo recettivo che cresce con la profondità
nelle CNN. Da qui due indicazioni pratiche. Primo, la profondità va scelta in
base al **diametro** del vicinato utile: due o tre strati bastano quasi sempre,
perché il numero di nodi raggiunti cresce in fretta col grado. Secondo, andare
troppo profondi è controproducente: applicando molte volte $\hat{A}$ le
rappresentazioni dei nodi convergono verso un unico punto e diventano
indistinguibili — il fenomeno dell'*over-smoothing*, per cui in pratica le GCN
molto profonde rendono peggio di quelle a due strati.

`````

```{figure} ../figures/message-passing.gif
:name: fig-message-passing-animato
:alt: Animazione di un grafo con un nodo centrale v, quattro vicini diretti e quattro nodi a due salti. Al primo giro i messaggi viaggiano dai vicini diretti verso v; al secondo giro partono prima dai nodi esterni verso i vicini, poi di nuovo verso v.
:width: 90%

Il campo recettivo che si allarga: al giro $k=1$ arrivano a $v$ solo i vicini
diretti; al giro $k=2$ i messaggi partono dai nodi a due salti, passano *per* i
vicini e arrivano anch'essi.
```

La {numref}`fig-message-passing-animato` rende evidente il punto che rende
delicata la profondità: l'informazione lontana non salta, **transita**. Ogni
strato in più la fa passare per un altro nodo, che la mescola con la propria —
ed è proprio questa mescolanza ripetuta a produrre, alla lunga, l'over-smoothing.

## Addestrare: classificare i nodi con poche etichette

Con lo schema in mano, addestrare una GCN è la solita storia: definire una loss,
calcolarne il gradiente con la backpropagation e scendere lungo di esso — le
stesse regole della sezione sull'addestramento delle reti. Cambia solo la
forma del dato. Il banco di prova classico è **Cora**: un grafo di circa 2700
articoli scientifici (i nodi), collegati da un arco quando uno cita l'altro
(circa 5400 archi), ciascuno descritto da una feature *bag-of-words* di 1433
parole e da etichettare in una di 7 categorie tematiche.

`````{tab} Elementare

La particolarità è che conosciamo l'argomento di **pochissimi** articoli — nel
setup standard di Cora appena 20 per categoria, 140 nodi in tutto su 2700 — e
vogliamo indovinare quello di tutti gli altri. Come si fa con così poche
risposte in mano? Sfruttando i collegamenti: un articolo tende a citare articoli
del suo stesso campo. Il message passing fa scorrere le poche etichette note
lungo le citazioni, contagiando i vicini. È come indovinare gli hobby di una
comitiva conoscendone solo alcuni: chi frequenta i patiti di scacchi,
probabilmente gioca a scacchi anche lui.

Il trucco è che, pur pagando (con la loss) solo gli errori sui 140 articoli che
conosciamo, per rispondere su di essi la rete ha dovuto far girare
l'informazione su **tutto** il grafo. Così, aggiustandosi, impara a
rappresentare bene anche i nodi che non abbiamo mai etichettato. Per questo si
chiama apprendimento **semi-supervisionato**: poche etichette, ma tanta
struttura.

`````

`````{tab} Superiore

Formalmente è *node classification* semi-supervisionata in regime
**trasduttivo**: il grafo intero, feature comprese, è visibile in
addestramento, ma solo un piccolo insieme $\mathcal{V}_{\text{train}}$ di nodi
è etichettato. Una GCN a due strati produce i logit per tutti i nodi,

$$
Z = \hat{A}\,\sigma\!\big(\hat{A}\,X\,W^{(0)}\big)\,W^{(1)},
$$

e la loss è la cross-entropia calcolata **solo** sui nodi etichettati:

$$
\mathcal{L} = -\sum_{v \in \mathcal{V}_{\text{train}}}
   \sum_{c=1}^{C} y_{vc} \, \log \hat{y}_{vc},
\qquad
\hat{y}_{v} = \mathrm{softmax}(z_v),
$$

dove $y_{vc}$ è l'etichetta one-hot del nodo $v$ per la classe $c$ e $C$ il
numero di classi. Il punto sottile è che $z_v$ dipende, tramite $\hat{A}$, dalle
feature dell'intero vicinato a due salti: il gradiente di $\mathcal{L}$ fluisce
quindi indietro anche attraverso nodi **non** etichettati, che partecipano
all'addestramento pur senza comparire nella somma. La GCN originale raggiunge
circa l'$81{,}5\%$ di accuratezza sul test di Cora — un balzo netto rispetto ai
metodi precedenti, ottenuto con appena due strati e $140$ nodi etichettati.

`````

## Un layer GCN in PyTorch

Tradurre la regola $H^{(l+1)} = \sigma(\hat{A}\,H^{(l)}\,W^{(l)})$ in codice è
sorprendentemente breve. Uno strato è una trasformazione lineare seguita dal
prodotto con l'adiacenza normalizzata, precalcolata una volta sola:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)  # la matrice W

    def forward(self, H, A_hat):
        # H: (N, in_dim) stati dei nodi; A_hat: (N, N) adiacenza normalizzata
        return A_hat @ self.lin(H)  # Â (H W)

class GCN(nn.Module):
    def __init__(self, in_dim, hid, n_classi):
        super().__init__()
        self.gc1 = GCNLayer(in_dim, hid)
        self.gc2 = GCNLayer(hid, n_classi)

    def forward(self, H, A_hat):
        H = F.relu(self.gc1(H, A_hat))  # primo strato + ReLU
        H = self.gc2(H, A_hat)          # secondo strato: logit per nodo
        return H
```

L'addestramento è un normale ciclo di discesa del gradiente, con l'unico
accorgimento di mascherare la loss sui soli nodi etichettati:

```{code-block} python
:class: pt-non-eseguibile

model = GCN(in_dim=1433, hid=16, n_classi=7)
opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

for epoca in range(200):
    model.train()
    opt.zero_grad()
    logit = model(H, A_hat)                                # tutti i nodi
    loss = F.cross_entropy(logit[mask_train], y[mask_train])  # solo etichettati
    loss.backward()                                        # backprop su tutto il grafo
    opt.step()
```

In pratica non serve scrivere lo strato a mano: la libreria **PyTorch
Geometric** offre `GCNConv`, che aggiunge i cappi e applica la normalizzazione
simmetrica al volo, prendendo il grafo nel formato compatto `edge_index`
(la lista degli archi, di forma `(2, num_archi)`) invece della matrice
$\hat{A}$ densa — indispensabile sui grafi grandi, dove $\hat{A}$ non entrerebbe
in memoria:

```python
from torch_geometric.nn import GCNConv

conv = GCNConv(in_channels=1433, out_channels=16)
# forward: conv(x, edge_index), con x di forma (N, 1433)
```

Da qui in avanti le domande diventano: e se i vicini fossero troppi per
guardarli tutti? E se alcuni contassero più di altri? Sono esattamente le
questioni che aprono la sezione successiva, «Oltre la GCN», dove
incontreremo il campionamento dei vicini di GraphSAGE e i pesi di attenzione
delle Graph Attention Network.

```{admonition} Da ricordare
:class: important
- Il **message passing** aggiorna ogni nodo in tre mosse ripetute a ogni
  strato: calcola i **messaggi** dei vicini, li **aggrega** con una funzione
  invariante all'ordine (somma, media, max) e **aggiorna** lo stato del nodo.
  È il telaio generale delle MPNN {cite}`gilmer2017neural`.
- La **GCN** {cite}`kipf2017semi` è l'istanza più usata:
  $H^{(l+1)} = \sigma(\hat{A}\,H^{(l)}\,W^{(l)})$ con
  $\hat{A} = \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ e $\tilde{A} = A+I$.
- La **normalizzazione simmetrica** impedisce ai nodi ad alto grado di dominare
  e tiene gli autovalori in $[-1,1]$, stabilizzando le scale attraverso gli
  strati; discende dai filtri spettrali di Čebyšëv
  {cite}`defferrard2016convolutional`.
- Impilare $K$ strati dà a ogni nodo un **campo recettivo a $K$ salti**, l'esatto
  analogo della profondità nelle CNN; troppa profondità causa *over-smoothing*.
- L'addestramento tipico è la **classificazione dei nodi
  semi-supervisionata** (Cora): cross-entropia sui soli nodi etichettati, ma
  gradienti che fluiscono su tutto il grafo, con la solita discesa del gradiente.
```
