# Oltre la GCN: GraphSAGE, GAT e applicazioni

La *Graph Convolutional Network* della sezione precedente è un piccolo
miracolo di semplicità: un solo strato di message passing (media normalizzata
dei vicini, trasformazione lineare, non linearità) e già classifica i nodi di
un grafo meglio di quanto facessero i cammini casuali. Ma quella eleganza si
paga con due limiti che, su un grafo vero, diventano subito ingombranti.

Il primo è che la GCN, così com'è formulata, è **transduttiva**: la
normalizzazione $\hat{A} = \tilde{D}^{-1/2}\tilde{A}\,\tilde{D}^{-1/2}$ va
calcolata sull'intera matrice di adiacenza, il grafo intero, fissato una volta
per tutte. Arriva un nodo nuovo (un utente che si iscrive oggi a un social) e
bisogna in linea di principio rifare i conti su tutto. Il secondo è di
**scala**: ad ogni strato ogni nodo somma *tutti* i suoi vicini, e i vicini
dei vicini, e così via; su un grafo di miliardi di archi, dove qualche
nodo-celebrità ha milioni di connessioni, questa somma esplode. Questa sezione
racconta le due idee che hanno tolto la GNN dal laboratorio e l'hanno messa in
produzione da Pinterest a Google Maps (*GraphSAGE* e la *Graph Attention
Network*) e poi fa il giro delle cose che oggi, con questi strumenti, si
riesce davvero a fare.

## GraphSAGE: imparare a generalizzare

La svolta arriva nel 2017 da Will Hamilton, Rex Ying e Jure Leskovec a Stanford,
con un modello dal nome che è già un programma: **GraphSAGE**, da *SAmple and
aggreGatE* {cite}`hamilton2017inductive`. Due parole, due idee.

`````{tab} Elementare

Torniamo all'immagine della sezione sul message passing: per farsi un'idea di
un nodo si guardano i suoi vicini. La GCN, per farlo, ha bisogno di avere
davanti *tutta* la rete di amicizie, e di averla vista per intero già durante
l'addestramento. GraphSAGE cambia il punto di vista con una domanda semplice:
e se, invece di imparare a memoria un vettore per ciascuna persona,
imparassimo la **ricetta** per costruirlo? Una ricetta del tipo «prendi la
persona, guarda i suoi amici, mescola nel modo giusto». Una ricetta la puoi
applicare anche a qualcuno che non hai mai visto, purché tu sappia chi sono i
suoi amici. Questo si chiama modo **induttivo**: la rete non impara *i
risultati*, impara *come si calcolano*, e quel «come» funziona pure sui nuovi
arrivati e su reti diverse da quella di addestramento (un antibiotico nuovo,
un utente iscritto stamattina).

La seconda idea combatte l'ingombro. Se una persona ha diecimila contatti,
guardarli tutti a ogni giro è impraticabile. E se ne bastasse un **campione**?
GraphSAGE, a ogni strato, non prende tutti i vicini ma ne pesca a caso un
numero fisso (diciamo venticinque) e aggrega solo quelli. È come farsi un'idea
di un quartiere non intervistando tutti gli abitanti, ma un campione a sorte:
molto più economico, e quasi altrettanto informativo. Nel pannello di sinistra
della {numref}`fig-gnn-graphsage-gat` i tre vicini pieni sono quelli
campionati; gli altri, questo giro, restano fuori.

`````

`````{tab} Superiore

GraphSAGE riscrive lo schema $\mathrm{AGGREGATE}$–$\mathrm{UPDATE}$ del message
passing in forma dichiaratamente induttiva. Al passo $k$, per ogni nodo $v$:

$$
h_{\mathcal{N}(v)}^{(k)} = \mathrm{AGGREGATE}_k\big(\{\, h_u^{(k-1)} : u \in \mathcal{S}(v) \,\}\big),
\qquad
h_v^{(k)} = \sigma\!\Big(W^{(k)} \big[\, h_v^{(k-1)} \;\|\; h_{\mathcal{N}(v)}^{(k)} \,\big]\Big),
$$

dove $\|$ è la concatenazione, $\sigma$ una non linearità, $W^{(k)}$ i pesi
condivisi dello strato $k$, e (cruciale)
$\mathcal{S}(v) \subseteq \mathcal{N}(v)$ è un **sottoinsieme campionato
uniformemente** dei vicini, di dimensione fissa. È il campionamento a rendere
il costo per nodo indipendente dal grado: con $S$ vicini campionati per strato
e $K$ strati, il sottografo che alimenta un nodo ha al più $S^K$ foglie,
comunque grande sia il grafo. E poiché $W^{(k)}$ e le funzioni di aggregazione
non dipendono da *quali* nodi si stia guardando ma solo dalle loro feature, il
modello si applica di peso a nodi e grafi mai visti: l'inferenza su un nuovo
nodo richiede solo di conoscerne il vicinato, non di riaddestrare.

La funzione $\mathrm{AGGREGATE}$ deve restare invariante all'ordine dei vicini;
Hamilton et al. ne propongono tre varianti:

- **mean**: la media (eventualmente pesata) dei vettori dei vicini. Con una
  piccola modifica (concatenare non più, ma sommare $v$ ai suoi vicini prima
  della media), si riottiene quasi esattamente la propagazione della GCN, che
  diventa così un *caso particolare* di GraphSAGE.
- **pool**: ogni vicino passa per uno stesso piccolo strato denso, poi si prende
  il massimo elemento per elemento (*max-pooling*): $\max\{\sigma(W_{\text{pool}}\,h_u + b) : u \in \mathcal{S}(v)\}$.
  Simmetrico perché il massimo non dipende dall'ordine.
- **LSTM**: più espressiva ma, di suo, sensibile all'ordine; la si rende
  utilizzabile applicandola a **permutazioni casuali** dei vicini.

`````

In PyTorch, con la libreria PyTorch Geometric, uno strato GraphSAGE è una riga;
il campionamento dei vicini è delegato a un *loader* dedicato (`NeighborLoader`),
così l'addestramento procede a mini-batch anche su grafi che non entrano in
memoria:

```python
import torch
from torch import nn
from torch_geometric.nn import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        # aggr='mean' e' l'aggregatore di default; 'max' -> pooling
        self.conv1 = SAGEConv(in_dim, hid_dim, aggr="mean")
        self.conv2 = SAGEConv(hid_dim, out_dim, aggr="mean")

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))  # primo giro di vicinato
        return self.conv2(x, edge_index)           # secondo giro
```

## GAT: non tutti i vicini contano uguale

GraphSAGE tratta i vicini campionati alla pari: nella media, ognuno pesa quanto
gli altri. Ma è ragionevole? In una molecola, non tutti i legami di un atomo
sono ugualmente informativi; in un social, l'amico stretto conta più del
contatto occasionale. L'idea di **pesare** i vicini l'abbiamo già incontrata, e
in grande stile: è l'**attenzione** del capitolo sui Transformer. La *Graph
Attention Network* (GAT), proposta nel 2018 da Petar Veličković e colleghi, la
porta di peso sui grafi {cite}`velickovic2018graph`.

`````{tab} Elementare

Ricordi l'evidenziatore del capitolo sull'attenzione? Davanti a una parola, il
modello ripassava tutte le altre e le colorava con intensità diversa, secondo
quanto contavano. La GAT fa la stessa cosa, ma l'evidenziatore lo passa sui
**vicini di un nodo nel grafo**. Quando aggiorna un nodo non fa più una media
democratica: prima decide, vicino per vicino, *quanto* pesarlo (dà a ognuno un
voto tra 0 e 1, e i voti sommano a 1) e poi fa la media *pesata* con quei
voti. Il bello è che nessuno scrive a mano questi pesi: li impara la rete,
come ogni altro parametro. Nel pannello di destra della
{numref}`fig-gnn-graphsage-gat` lo spessore di ogni arco è il peso di
attenzione: un vicino, quello con l'arco più grosso, si prende la fetta più
grande; gli altri contano meno.

C'è un ponte esplicito da tenere a mente. L'attenzione dei Transformer fa
guardare ogni parola a tutte le altre della frase: è, in fondo, attenzione su un
grafo *completo*, dove ogni parola è collegata a ogni altra. La GAT è la stessa
identica idea, ma su un grafo *qualunque*: ogni nodo guarda solo i vicini a cui
è davvero collegato. Detta al contrario: la self-attention è una GAT sul grafo
in cui tutti sono vicini di tutti.

`````

`````{tab} Superiore

In uno strato GAT ogni nodo calcola, verso ciascun vicino, un punteggio di
attenzione, poi lo normalizza con una softmax sul vicinato. Con
$h_i$ le feature del nodo $i$ e $W$ una trasformazione lineare condivisa:

$$
\alpha_{ij} = \frac{\exp\!\Big(\mathrm{LeakyReLU}\big(\mathbf{a}^{\top}[\,W h_i \,\|\, W h_j\,]\big)\Big)}
{\sum_{k \in \mathcal{N}(i)} \exp\!\Big(\mathrm{LeakyReLU}\big(\mathbf{a}^{\top}[\,W h_i \,\|\, W h_k\,]\big)\Big)},
\qquad
h_i' = \sigma\!\Big(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}\, W h_j\Big),
$$

dove $\|$ è la concatenazione, $\mathbf{a}$ è un vettore di parametri appreso,
$\mathrm{LeakyReLU}$ la non linearità usata sul punteggio (pendenza $0{,}2$ per
gli ingressi negativi) e $\sigma$ quella finale. Il coefficiente $\alpha_{ij}$
dice **quanto il nodo $i$ pesa il vicino $j$**; la softmax garantisce
$\sum_{j \in \mathcal{N}(i)} \alpha_{ij} = 1$. Rispetto alla *scaled dot-product
attention* dei Transformer cambia solo il modo di calcolare il punteggio (qui
una piccola rete con $\mathbf{a}$ e la $\mathrm{LeakyReLU}$, lì il prodotto
scalare query·key riscalato); l'ossatura «pesi softmax, media pesata dei value»
è la stessa.

Un esempio a mano. Un nodo $i$ ha tre vicini, e i punteggi (dopo la
$\mathrm{LeakyReLU}$) valgono $e_{i1}=2$, $e_{i2}=1$, $e_{i3}=0$. La softmax dà

$$
\alpha_{i1} = \frac{e^{2}}{e^{2}+e^{1}+e^{0}} = \frac{7{,}39}{11{,}11} \approx 0{,}67,
\quad
\alpha_{i2} = \frac{2{,}72}{11{,}11} \approx 0{,}24,
\quad
\alpha_{i3} = \frac{1}{11{,}11} \approx 0{,}09,
$$

e i tre pesi sommano a $1$ come devono: il primo vicino domina l'aggregazione
(due terzi del peso), il terzo è quasi ignorato. Come nei Transformer, si
usano più **teste** in parallelo (*multi-head*): $K$ meccanismi di attenzione
indipendenti, i cui risultati si concatenano negli strati intermedi e si
mediano nello strato finale; così il modello può pesare i vicini secondo
criteri diversi contemporaneamente.

`````

```{figure} ../figures/gnn-graphsage-gat.svg
:name: fig-gnn-graphsage-gat
:alt: "A sinistra GraphSAGE: un nodo centrale con sei vicini, di cui solo tre campionati (pieni, archi solidi) e tre sbiaditi (archi tratteggiati); si aggrega solo il sottoinsieme campionato. A destra GAT: lo stesso nodo con gli stessi sei vicini tutti presenti, ma gli archi hanno spessore diverso, proporzionale al peso di attenzione: un vicino conta molto più degli altri."

Due modi di guardare lo stesso vicinato. **GraphSAGE** (sinistra) ne
*campiona* un sottoinsieme e lo aggrega alla pari, per scalare a grafi enormi.
**GAT** (destra) tiene tutti i vicini ma li *pesa* con l'attenzione: lo
spessore dell'arco è il coefficiente $\alpha_{ij}$.
```

Anche la GAT, in PyTorch Geometric, è uno strato pronto all'uso; l'argomento
`heads` fissa il numero di teste di attenzione:

```{code-block} python
:class: pt-non-eseguibile

from torch_geometric.nn import GATConv

# 8 teste concatenate: l'uscita ha dimensione hid_dim * 8
conv1 = GATConv(in_dim, hid_dim, heads=8, concat=True)
# strato finale: le teste si mediano invece di concatenarsi
conv2 = GATConv(hid_dim * 8, out_dim, heads=1, concat=False)
```

## Dal nodo al grafo intero: readout e potere espressivo

Finora abbiamo prodotto un vettore *per ogni nodo*. Ma i compiti a **livello
di grafo** («questa molecola è tossica?», «questo composto uccide i batteri?»)
chiedono un solo verdetto per l'intero grafo. Serve un passo in più:
comprimere i tanti vettori dei nodi in **un** vettore del grafo. Questo passo
si chiama **readout** (o *pooling* globale).

`````{tab} Elementare

Immagina di aver dato un voto a ogni giocatore di una squadra e di volere ora
un unico numero per la squadra intera. Le strade ovvie sono tre: **sommare**
tutti i voti, farne la **media**, o prendere il **massimo** (il voto del
migliore). Sono esattamente le tre ricette del readout: somma, media, massimo
dei vettori dei nodi. Semplici, e (nota) tutte e tre indifferenti all'ordine
in cui elenchi i giocatori, che è proprio ciò che ci serve su un grafo.

Sembrano equivalenti, ma non lo sono, e la differenza è più profonda di quanto
sembri. La media e il massimo **dimenticano quanti** sono i nodi; la somma no.
Un esempio: una molecola con due gruppi ossidrili e una con un solo gruppo
ossidrilo, a parità di tutto il resto, sono molecole diverse, e possono
comportarsi in modo diverso. Se i due nodi «ossidrile» hanno lo stesso vettore
$b$, la **somma** dà $2b$ nel primo caso e $b$ nel secondo: li distingue. La
**media** dà $b$ in entrambi i casi, il **massimo** pure: confondono le due
molecole. Contare, a volte, è tutto.

`````

`````{tab} Superiore

Il readout aggrega l'insieme $\{h_v^{(K)} : v \in V\}$ in un vettore $h_G$ con
un'operazione invariante a permutazione; tipicamente $h_G = \sum_v h_v^{(K)}$,
oppure la media o il massimo. Esistono anche schemi di **pooling gerarchico**
(per esempio *DiffPool*), che alternano message passing e fusione di gruppi di
nodi in super-nodi, costruendo il vettore del grafo per livelli, come il
pooling delle CNN accorpa regioni dell'immagine.

La scelta dell'aggregatore non è un dettaglio implementativo: decide il
**potere espressivo** della rete, cioè quali grafi diversi essa riesce a
distinguere. Il risultato di riferimento è di Xu, Hu, Leskovec e Jegelka nel
2019 {cite}`xu2019powerful`, e lega le GNN a un classico test di
**isomorfismo**: il test di Weisfeiler–Lehman (WL). Il test WL colora
iterativamente i nodi impastando la propria etichetta con il *multinsieme*
delle etichette dei vicini: è esattamente la struttura del message passing. Xu
et al. dimostrano che **nessuna GNN a message passing può distinguere due
grafi che il test WL dichiara indistinguibili** (è il tetto teorico) e che una
GNN lo raggiunge solo se la sua aggregazione è **iniettiva** sul multinsieme
dei vicini. Da qui la gerarchia:

$$
\text{somma} \;\succ\; \text{media} \;\succ\; \text{massimo},
$$

dove $\succ$ significa «strettamente più espressiva». La somma conserva sia le
feature sia la **molteplicità** (quanti vicini di ciascun tipo); la media
conserva solo le proporzioni e perde il conteggio; il massimo tiene solo
l'insieme dei tipi presenti, dimenticando anche le proporzioni. Su queste basi
gli autori costruiscono la **Graph Isomorphism Network** (GIN), la cui regola di
aggiornamento è deliberatamente semplice e iniettiva:

$$
h_v^{(k)} = \mathrm{MLP}^{(k)}\!\Big( \big(1 + \epsilon^{(k)}\big)\, h_v^{(k-1)} + \sum_{u \in \mathcal{N}(v)} h_u^{(k-1)} \Big),
$$

dove $\mathrm{MLP}^{(k)}$ è un piccolo percettrone multistrato ed
$\epsilon^{(k)}$ uno scalare (appreso o fissato a 0) che dosa il peso del nodo
rispetto ai vicini. La somma sui vicini, seguita da un MLP, è quanto basta
perché GIN eguagli il potere del test WL: il massimo ottenibile da una GNN a
message passing.

`````

## A cosa servono: la GNN al lavoro

Il capitolo si è aperto su un antibiotico; è ora di mantenere la promessa e
mostrare dove le GNN, oggi, fanno la differenza.

**Chimica e farmaci.** È il terreno naturale delle GNN: una molecola *è* un
grafo (atomi nei nodi, legami negli archi) e prevederne una proprietà è un
compito a livello di grafo. L'idea di leggere le molecole con reti su grafo
risale ai *fingerprint molecolari neurali* di Duvenaud e colleghi del 2015
{cite}`duvenaud2015convolutional`, che sostituiscono i descrittori chimici
scritti a mano con un vettore appreso end-to-end. La punta di diamante è la
scoperta di **halicin**, già raccontata nell'apertura del capitolo: nel 2020
il gruppo di Jonathan Stokes e James Collins al MIT addestra una rete a
message passing a prevedere l'attività antibatterica, la passa al setaccio su
una libreria di composti e ne pesca uno che nessuno associava agli
antibiotici, efficace contro ceppi resistenti a ogni farmaco noto (pubblicato
su *Cell*).

**Raccomandazione su grafo.** Il caso industriale più celebre è **PinSage**, il
sistema che Pinterest mette in produzione nel 2018 (Ying e colleghi) per
suggerire contenuti su un grafo bipartito di miliardi di *pin* e bacheche.
PinSage è, nella sostanza, un GraphSAGE portato a scala web: campiona i vicini
con brevi cammini casuali e li aggrega, girando su un grafo di tre miliardi di
nodi. Come discusso nel capitolo sui sistemi di raccomandazione, raccomandare è
*link prediction* su un grafo utente–oggetto, ed è lì che le GNN danno il meglio.

**Rilevamento frodi.** Le transazioni finanziarie formano un grafo (conti nei
nodi, pagamenti negli archi) e le frodi vivono nelle *relazioni*: anelli di
conti che si rimpallano denaro, account che gravitano attorno a un mulo. Un
classificatore che guardi i conti uno per uno non lo vede; una GNN, che
propaga segnale lungo gli archi, sì. È oggi uno strumento standard
nell'antiriciclaggio e nella difesa dei pagamenti.

**Mappe e traffico.** Dal 2020 le stime del **tempo di percorrenza in Google
Maps** sono calcolate da una GNN sviluppata con DeepMind: la rete stradale è
il grafo (segmenti di strada nei nodi, incroci a collegarli) e il modello
prevede i tempi propagando informazione lungo il percorso, migliorando
l'accuratezza degli arrivi stimati in molte città (lavoro pubblicato da
Derrow-Pinion e colleghi nel 2021).

**Scienza e fisica.** Le GNN sono diventate *simulatori*: rappresentando un
fluido o un materiale come un grafo di particelle interagenti, reti come quelle
di Sanchez-Gonzalez e colleghi (2020) imparano a prevederne l'evoluzione nel
tempo. La stessa impalcatura muove GraphCast (DeepMind, 2023), che modella il
pianeta come un grafo di punti sulla superficie terrestre per la previsione
meteorologica, e diversi analizzatori di collisioni nella fisica delle particelle.

## I limiti, senza nasconderli

Le GNN non sono una bacchetta magica, e la letteratura è onesta sui loro punti
deboli: vale la pena conoscerli prima di innamorarsene. Una rassegna d'insieme
è la survey di Wu e colleghi {cite}`wu2021comprehensive`.

`````{tab} Elementare

Il difetto più curioso è che **impilare troppi strati peggiora le cose**. Con
uno strato ogni nodo ascolta i vicini; con due, anche i vicini dei vicini; ma
continuando così, dopo un po' *tutti* finiscono per ascoltare *tutti*, e le
rappresentazioni dei nodi si assomigliano sempre di più fino a diventare
indistinguibili, come una voce che, passando di bocca in bocca per tutto il
paese, si uniforma in un unico mormorio. Si chiama **oversmoothing**,
«levigatura eccessiva»: a furia di mediare con i vicini, si cancellano le
differenze che volevamo cogliere. Ecco perché, in pratica, le GNN restano
**basse**: due, tre, quattro strati, di rado di più. È l'opposto delle reti
per immagini, dove si arriva a centinaia di strati.

Un secondo problema è opposto e complementare: se l'informazione utile sta
**lontana** nel grafo (a molti passi di distanza) per arrivare deve passare da
imbuti sempre più stretti, e si perde per strada. È l'**over-squashing**, lo
«schiacciamento» dell'informazione lontana. Si aggiungono la fatica di girare
su grafi da miliardi di archi, e il fatto che quasi tutte le GNN danno per
scontato che i nodi collegati si somiglino (gli amici hanno gusti simili):
quando è il contrario (reti dove chi è connesso è *diverso*) rendono molto
meno.

`````

`````{tab} Superiore

- **Oversmoothing.** Li, Han e Wu (2018) mostrano che uno strato GCN è, in
  sostanza, un passo di *smoothing* laplaciano: iterandolo molte volte le feature
  dei nodi convergono verso un punto fisso che dipende dai gradi e non dai nodi,
  rendendoli indistinguibili. È la ragione teorica per cui, oltre pochi strati,
  l'accuratezza crolla; si mitiga con connessioni residuali, normalizzazioni ad
  hoc o salti tra strati, ma il fenomeno resta il vincolo pratico alla profondità.
- **Over-squashing.** Alon e Yahav (2021) osservano che il campo recettivo di un
  nodo cresce esponenzialmente con il numero di strati, mentre il vettore che lo
  riassume ha dimensione fissa: l'informazione proveniente da nodi distanti viene
  «schiacciata» attraverso colli di bottiglia topologici, penalizzando i compiti a
  lungo raggio. Profondità e portata sono così in tensione: servirebbero più
  strati per raggiungere nodi lontani, ma più strati innescano l'oversmoothing.
- **Scalabilità.** Il campionamento di GraphSAGE e PinSage attenua il costo, ma
  addestrare su grafi da miliardi di nodi resta un problema aperto di sistemi,
  non solo di modelli.
- **Eterofilia.** Molte GNN presuppongono l'**omofilia** (nodi collegati con
  etichette simili) che l'aggregazione dei vicini sfrutta implicitamente. Sui
  grafi **eterofili**, dove i nodi collegati tendono a differire, le
  architetture standard possono fare peggio di un percettrone che ignora la
  struttura, ed è un filone di ricerca attivo.

`````

## L'ecosistema, e dove andare da qui

Chi voglia mettere le mani in pasta non parte da zero: due librerie coprono
quasi tutto. **PyTorch Geometric** (PyG) (quella degli esempi di questa
sezione) e la **Deep Graph Library** (DGL) offrono, sopra PyTorch, gli strati
già pronti (`GCNConv`, `SAGEConv`, `GATConv`, `GINConv`), i *loader* per il
campionamento dei vicini e decine di dataset di riferimento. Scrivere una GNN,
oggi, è questione di poche righe: proprio come lo è diventato scrivere una
CNN.

Con questo si chiude il capitolo. Il filo, però, non si spezza: l'attenzione
che qui pesa i vicini di un nodo è la stessa dei Transformer, e i grafi
bipartiti utente–oggetto di questa sezione sono lo stesso oggetto del capitolo
sui sistemi di raccomandazione. Le reti su grafo non sono un'isola: sono il
punto in cui convoluzione, attenzione e apprendimento di rappresentazioni si
ritrovano, sotto un'unica lente (quella, geometrica, delle simmetrie del
dato).

```{admonition} Da ricordare
:class: important
- **GraphSAGE** rende la GNN **induttiva** (impara *funzioni* di aggregazione, non
  embedding fissi: generalizza a nodi e grafi mai visti) e **scalabile**
  (**campiona** un sottoinsieme di vicini a ogni strato). Aggregatori: mean, pool
  (max), LSTM.
- La **GAT** pesa i vicini con l'**attenzione**: coefficienti $\alpha_{ij}$
  appresi e normalizzati con softmax sul vicinato ($\sum_j \alpha_{ij}=1$), con
  più teste in parallelo. È la self-attention dei Transformer su un grafo
  qualunque anziché completo.
- I compiti a **livello di grafo** richiedono un **readout** (somma, media,
  massimo). La **somma** è la più espressiva perché conserva la molteplicità dei
  vicini: **GIN** la usa per eguagliare il test di isomorfismo di
  **Weisfeiler–Lehman**, il tetto del potere espressivo di una GNN a message
  passing.
- Applicazioni reali: farmaci (**halicin**, 2020), raccomandazione (**PinSage**,
  Pinterest 2018), rilevamento frodi, tempi di percorrenza in **Google Maps**
  (DeepMind, 2020–21), simulazioni fisiche e meteo.
- Limiti aperti: **oversmoothing** (troppi strati → nodi indistinguibili),
  **over-squashing** (informazione lontana schiacciata), **scalabilità**,
  **eterofilia**. In pratica le GNN restano **basse**, 2–4 strati.
```
