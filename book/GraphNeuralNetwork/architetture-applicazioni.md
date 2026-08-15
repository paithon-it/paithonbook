# Oltre la GCN: GraphSAGE, GAT e applicazioni

La *Graph Convolutional Network* della sezione sul message passing è un
piccolo miracolo di semplicità. Un solo giro di passaparola: si mettono
insieme i bigliettini dei vicini pesandoli, si riscrive il risultato con la
ricetta
appresa, si dà il ritocco finale. Tanto basta a classificare i nodi di un grafo
meglio di quanto facessero i cammini casuali. Ma quella eleganza si paga con
due limiti che, su un grafo vero, diventano subito ingombranti.

Il primo riguarda i nodi nuovi. La GCN, così come Kipf e Welling la addestrano,
si addestra su tutto il grafo in una volta sola, quel grafo lì e nessun altro,
e non dice cosa fare con chi arriva dopo: è la situazione che la sezione «Il
mondo come grafo» ha chiamato **transduttiva**. Pensa a un utente che si
iscrive oggi a un social: quando la rete è stata addestrata lui non c'era, e
per lui non esiste una risposta già pronta.

E il fastidioso è che il muro non è dove sembra. I numeri che la rete ha
imparato andrebbero benissimo anche per lui, e l'arrivo di un iscritto cambia
l'elenco dei collegamenti soltanto lì attorno, dove arrivano i suoi archi. A
non funzionare è la **procedura di addestramento**, che pretende di avere
davanti l'intero grafo fin dall'inizio.

Il secondo limite è di **scala**, e si vede seguendo a ritroso il conto. Per
calcolare un nodo servono tutti i suoi vicini; per calcolare quelli servono i
vicini dei vicini; e così via, allargandosi di un anello a ogni strato. Su un
grafo di miliardi di archi, dove qualche nodo-celebrità ha milioni di
connessioni, bastano due o tre anelli perché quella cerchia arrivi a
inghiottire mezza rete.

Questa sezione racconta le due idee che hanno tolto la GNN dal laboratorio e
l'hanno messa in produzione da Pinterest a Google Maps (*GraphSAGE* e la *Graph
Attention Network*) e poi fa il giro delle cose che oggi, con questi strumenti,
si riesce davvero a fare.

## GraphSAGE: imparare a generalizzare

La svolta arriva nel 2017 da Will Hamilton, Rex Ying e Jure Leskovec a
Stanford, con un modello che porta le sue due idee scritte nel nome:
**GraphSAGE**, dove SAGE sta per *SAmple and aggreGatE*, campiona e metti
insieme {cite}`hamilton2017inductive`. Pescare un campione di vicini invece di
guardarli tutti, e mettere insieme quel che si è pescato: due parole, due idee.

`````{tab} Elementare

Torniamo all'immagine della sezione sul message passing: per farsi un'idea di
un nodo si guardano i suoi vicini. La GCN, per farlo, ha bisogno di avere
davanti *tutta* la rete di amicizie, e di averla vista per intero già durante
l'addestramento. GraphSAGE cambia il punto di vista con una domanda semplice:
e se, invece di imparare a memoria una fila di numeri per ciascuna persona,
imparassimo la **ricetta** per costruirla? Una ricetta del tipo «prendi la
persona, guarda i suoi amici, mescola nel modo giusto». Una ricetta la puoi
applicare anche a qualcuno che non hai mai visto, purché tu sappia chi sono i
suoi amici. Questo si chiama modo **induttivo**: la rete non impara *i
risultati*, impara *come si calcolano*, e quel «come» funziona pure sui nuovi
arrivati e su reti diverse da quella di addestramento (un antibiotico nuovo,
un utente iscritto stamattina).

La seconda idea combatte l'ingombro. Se una persona ha diecimila contatti,
guardarli tutti a ogni giro è impraticabile. E se ne bastasse un **campione**?
GraphSAGE, a ogni strato, non prende tutti i vicini ma ne pesca a caso un
numero fisso (diciamo venticinque) e mescola solo quelli. È come farsi un'idea
di un quartiere non intervistando tutti gli abitanti, ma un campione a sorte:
molto più economico, e quasi altrettanto informativo. Nel pannello di sinistra
della {numref}`fig-gnn-graphsage-gat` i tre vicini pieni sono quelli
campionati; gli altri, questo giro, restano fuori.

`````

`````{tab} Superiore

GraphSAGE riscrive lo schema $\mathrm{AGGREGATE}$–$\mathrm{UPDATE}$ del message
passing in forma dichiaratamente induttiva. Al passo $k$, per ogni nodo $v$:

$$
\mathbf{h}_{\mathcal{N}(v)}^{(k)} = \mathrm{AGGREGATE}_k\big(\{\, \mathbf{h}_u^{(k-1)} : u \in \mathcal{S}(v) \,\}\big),
\qquad
\mathbf{h}_v^{(k)} = \sigma\!\Big(\mathbf{W}^{(k)} \big[\, \mathbf{h}_v^{(k-1)} \;\|\; \mathbf{h}_{\mathcal{N}(v)}^{(k)} \,\big]\Big),
$$

dove $\|$ è la concatenazione, $\sigma$ una non linearità, $\mathbf{W}^{(k)}$ i
pesi condivisi dello strato $k$, e (cruciale)
$\mathcal{S}(v) \subseteq \mathcal{N}(v)$ è un **sottoinsieme campionato
uniformemente** dei vicini, di dimensione fissa. È il campionamento a rendere
il costo per nodo indipendente dal grado: con $S$ vicini campionati per strato
e $K$ strati, il sottografo che alimenta un nodo ha al più $S^K$ foglie,
comunque grande sia il grafo. E poiché $\mathbf{W}^{(k)}$ e le funzioni di
aggregazione non dipendono da *quali* nodi si stia guardando ma solo dalle loro
feature, il modello si applica di peso a nodi e grafi mai visti: l'inferenza su
un nuovo nodo richiede solo di conoscerne il vicinato, non di riaddestrare.

La funzione $\mathrm{AGGREGATE}$ deve restare invariante all'ordine dei vicini;
Hamilton et al. ne propongono tre varianti:

- **mean**: la media (eventualmente pesata) dei vettori dei vicini. Con una
  piccola modifica (invece di concatenare, si somma $v$ ai suoi vicini prima
  della media) si ottiene la **variante convoluzionale**, che gli autori stessi
  descrivono come «un'approssimazione lineare grossolana» di una convoluzione
  spettrale localizzata. Non è un caso particolare della GCN, ed è utile capire
  perché: quella media è l'operatore
  $\tilde{\mathbf{D}}^{-1}\tilde{\mathbf{A}}$, la normalizzazione **per righe**
  che la sezione sul message passing aveva scartato in favore della simmetrica
  $\hat{\mathbf{A}}$. Sulla catena di quattro nodi di quella sezione, con
  $\mathbf{X} = (1,2,3,4)^\top$, un passo dei due operatori dà
  $(1{,}500,\, 2{,}000,\, 3{,}000,\, 3{,}500)$ contro
  $(1{,}316,\, 2{,}075,\, 3{,}300,\, 3{,}225)$: sul primo nodo la media per
  righe esce del $14\%$ più alta ($1{,}500 / 1{,}316$), e il conto si rifà a
  mente, perché la prima riga è semplicemente $(1+2)/2$.
- **pool**: ogni vicino passa per uno stesso piccolo strato denso, poi si
  prende il massimo elemento per elemento (*max-pooling*):
  $\max\{\sigma(\mathbf{W}_{\text{pool}}\,\mathbf{h}_u + \mathbf{b}) : u \in \mathcal{S}(v)\}$.
  Simmetrico perché il massimo non dipende dall'ordine.
- **LSTM**: più espressiva ma, di suo, sensibile all'ordine; la si rende
  utilizzabile applicandola a **permutazioni casuali** dei vicini.

Vale la pena aggiungere una cosa che il risparmio di costo mette in ombra: il
campionamento rende il forward **aleatorio**, e non nello stesso modo per i tre
aggregatori. La media su un campione stima la media completa **senza
distorsione**; il massimo su un campione è invece sistematicamente **più basso**
del massimo vero, e la distorsione cresce col grado; una somma su campione
sarebbe distorta di un fattore $S/\deg(v)$. Anche il caso non distorto smette di
esserlo appena attraversa la non linearità (disuguaglianza di Jensen), ed è la
ragione per cui esiste tutta la letteratura sulla riduzione della varianza nel
campionamento su grafo. In pratica, a inferenza si preferisce il vicinato pieno
quando il grado lo consente. Una precisazione minuta e utile: $\mathcal{S}(v)$
non è propriamente un sottoinsieme, perché quando la dimensione del campione
supera il grado si campiona **con reinserimento**.

`````

In PyTorch, con la libreria PyTorch Geometric, uno strato GraphSAGE si scrive
in una riga: è la `SAGEConv` che compare due volte qui sotto. Del campionamento
dei vicini, che nel codice non si vede perché non è compito dello strato, si
occupa un componente a parte (`NeighborLoader`): serve alla rete un pezzo di
grafo alla volta invece del grafo intero, ed è quello che permette di
addestrare anche quando il grafo, tutto insieme, in memoria non ci starebbe.

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

La {numref}`fig-gnn-graphsage-gat` mette a confronto il modo di guardare il
vicinato di GraphSAGE, che è quello appena descritto, con quello del modello
che arriva subito qui sotto: nel pannello di sinistra si tengono solo alcuni
vicini, scelti a sorte; in quello di destra si tengono tutti, ma pesati.

```{figure} ../figures/gnn-graphsage-gat.svg
:name: fig-gnn-graphsage-gat
:alt: "A sinistra GraphSAGE: un nodo centrale con sei vicini, di cui solo tre campionati (pieni, archi solidi) e tre sbiaditi (archi tratteggiati); si aggrega solo il sottoinsieme campionato. A destra GAT: lo stesso nodo con gli stessi sei vicini tutti presenti, ma gli archi hanno spessore diverso, proporzionale al peso di attenzione, dal più grosso in alto a sinistra al più sottile in basso a sinistra."
:width: 100%

Due modi di guardare lo stesso vicinato. **GraphSAGE** (sinistra) ne
*campiona* un sottoinsieme e lo aggrega alla pari, per scalare a grafi enormi.
**GAT** (destra) tiene tutti i vicini ma li **pesa** con l'attenzione: più
grosso è il tratto dell'arco, più quel vicino conta.
```

## GAT: non tutti i vicini contano uguale

GraphSAGE tratta i vicini campionati alla pari: nella media, ognuno pesa quanto
gli altri. Ma è ragionevole? In una molecola, non tutti i legami di un atomo
sono ugualmente informativi; in un social, l'amico stretto conta più del
contatto occasionale. L'idea di **pesare** i vicini l'abbiamo già incontrata, e
in grande stile: è l'**attenzione** del capitolo sui Transformer. La *Graph
Attention Network* (GAT), proposta nel 2018 da Petar Veličković e colleghi, la
prende pari pari e la porta sui grafi {cite}`velickovic2018graph`.

`````{tab} Elementare

Ricordi l'evidenziatore del capitolo sull'attenzione? Davanti a una parola, il
modello ripassava tutte le altre e le colorava con intensità diversa, secondo
quanto contavano. La GAT fa la stessa cosa, ma l'evidenziatore lo passa sui
**vicini di un nodo nel grafo**. Quando aggiorna un nodo non fa più una media
democratica: prima decide, vicino per vicino, *quanto* pesarlo (dà a ognuno un
voto tra 0 e 1, e i voti sommano a 1) e poi fa la media *pesata* con quei voti,
cioè moltiplica ogni bigliettino per il voto del suo mittente prima di
sommarli, così chi ha preso il voto più alto conta di più nel totale.

Un esempio con i numeri veri. Un nodo con tre vicini distribuisce quattro
voti, uno per vicino e uno per sé stesso, e vanno così:
$0{,}53$ al primo, $0{,}20$ al secondo, $0{,}07$ al terzo, e $0{,}20$ lo tiene
per sé (anche qui vale la regola dei cappi: un nodo è vicino di sé stesso, se
no si dimentica quello che sapeva). Sommali: fanno $1$. Il primo vicino si
prende poco più della metà del nuovo stato, il terzo è quasi ignorato, e un
quinto se lo tiene il nodo. Il bello è che nessuno scrive a mano questi voti:
li impara la rete, come ogni altro parametro. Nel pannello di destra della
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
attenzione, poi lo normalizza con una softmax sul vicinato. Con $\mathbf{h}_i$
le feature del nodo $i$ e $\mathbf{W}$ una trasformazione lineare condivisa:

$$
\alpha_{ij} = \frac{\exp\!\Big(\mathrm{LeakyReLU}\big(\mathbf{a}^{\top}[\,\mathbf{W} \mathbf{h}_i \,\|\, \mathbf{W} \mathbf{h}_j\,]\big)\Big)}
{\sum_{k \in \mathcal{N}(i) \cup \{i\}} \exp\!\Big(\mathrm{LeakyReLU}\big(\mathbf{a}^{\top}[\,\mathbf{W} \mathbf{h}_i \,\|\, \mathbf{W} \mathbf{h}_k\,]\big)\Big)},
\qquad
\mathbf{h}_i' = \sigma\!\Big(\sum_{j \in \mathcal{N}(i) \cup \{i\}} \alpha_{ij}\, \mathbf{W} \mathbf{h}_j\Big),
$$

dove il vicinato, come nel paper, comprende il nodo stesso ($j$ corre su
$\mathcal{N}(i) \cup \{i\}$): senza questo cappio il nodo dimenticherebbe la
propria feature, il difetto che i *self-loop* della GCN erano nati per evitare.
Qui $\|$ è la concatenazione, $\mathbf{W} \in \mathbb{R}^{F \times F'}$ è la
trasformazione lineare condivisa, $\mathbf{a} \in \mathbb{R}^{2F'}$ è un
vettore di parametri appreso (la lunghezza è $2F'$ perché deve moltiplicare due
vettori concatenati, ed è quel che rende il punteggio uno **scalare**),
$\mathrm{LeakyReLU}$ la non linearità usata sul punteggio (pendenza $0{,}2$ per
gli ingressi negativi) e $\sigma$ quella finale. Il coefficiente $\alpha_{ij}$
dice **quanto il nodo $i$ pesa il vicino $j$**; la softmax garantisce
$\sum_{j \in \mathcal{N}(i) \cup \{i\}} \alpha_{ij} = 1$.

Rispetto alla *scaled dot-product attention* dei Transformer l'ossatura «pesi
softmax, media pesata» è la stessa, ma due cose cambiano. Cambia il modo di
calcolare il punteggio (qui una piccola rete con $\mathbf{a}$ e la
$\mathrm{LeakyReLU}$, lì il prodotto scalare query·key riscalato per
$1/\sqrt{d_k}$, che è la ragione del nome *scaled*). E cambia il fatto che
nella GAT **non esiste una proiezione separata per i *value***: la stessa
$\mathbf{W}$ fa due mestieri, costruisce il punteggio e produce il vettore che
poi viene mediato, mentre il Transformer tiene $\mathbf{W}_Q$, $\mathbf{W}_K$ e
$\mathbf{W}_V$ distinte.

Un esempio a mano. Un nodo $i$ ha tre vicini, e i punteggi (dopo la
$\mathrm{LeakyReLU}$) valgono $e_{i1}=2$, $e_{i2}=1$, $e_{i3}=0$; il cappio,
cioè il punteggio che il nodo assegna a sé stesso, vale $e_{ii}=1$. La softmax
corre su tutti e quattro i termini, e il denominatore è
$e^{2}+e^{1}+e^{0}+e^{1} = 13{,}83$:

$$
\alpha_{i1} = \frac{7{,}39}{13{,}83} \approx 0{,}53,
\quad
\alpha_{i2} = \frac{2{,}72}{13{,}83} \approx 0{,}20,
\quad
\alpha_{i3} = \frac{1}{13{,}83} \approx 0{,}07,
\quad
\alpha_{ii} = \frac{2{,}72}{13{,}83} \approx 0{,}20 .
$$

I quattro pesi sommano a $1$ come devono: il primo vicino domina
l'aggregazione (poco più della metà), il terzo è quasi ignorato, e un quinto
del nuovo stato se lo prende il nodo stesso. Come nei Transformer, si
usano più **teste** in parallelo (*multi-head*): $K$ meccanismi di attenzione
indipendenti, i cui risultati si concatenano negli strati intermedi e si
mediano nello strato finale; così il modello può pesare i vicini secondo
criteri diversi contemporaneamente.

`````

Anche la GAT, in PyTorch Geometric, è uno strato pronto all'uso. L'argomento
`heads` dice quanti evidenziatori diversi passare in parallelo sugli stessi
vicini, ciascuno libero di dare voti secondo un criterio suo: nel capitolo sui
Transformer si chiamavano **teste** di attenzione, e qui sono la stessa cosa.
Negli strati intermedi i risultati delle teste si mettono in fila uno dopo
l'altro (`concat=True`, e infatti l'uscita è tanto più lunga quante sono le
teste); nell'ultimo strato si fa invece la media, perché lì serve una risposta
sola:

```{code-block} python
:class: pt-non-eseguibile

from torch_geometric.nn import GATConv

# 8 teste concatenate: l'uscita ha dimensione hid_dim * 8
conv1 = GATConv(in_dim, hid_dim, heads=8, concat=True)
# strato finale: le teste si mediano invece di concatenarsi
conv2 = GATConv(hid_dim * 8, out_dim, heads=1, concat=False)
```

## Dal nodo al grafo intero, e fin dove si riesce a distinguere

Finora abbiamo prodotto una fila di numeri *per ogni nodo*. Ma i compiti a
**livello di grafo** («questa molecola è tossica?», «questo composto uccide i
batteri?») chiedono un solo verdetto per l'intero grafo. Serve un passo in più:
comprimere le tante file di numeri dei nodi in **una** sola, quella del grafo.
Questo passo si chiama **readout**, letteralmente «lettura finale», e la scelta
di come farlo non è un dettaglio: decide quali grafi diversi la rete riuscirà a
distinguere fra loro.

`````{tab} Elementare

Immagina di aver dato un voto a ogni giocatore di una squadra e di volere ora
un unico numero per la squadra intera. Le strade ovvie sono tre: **sommare**
tutti i voti, farne la **media**, o prendere il **massimo** (il voto del
migliore). Sono esattamente le tre ricette del readout: somma, media, massimo
dei vettori dei nodi. Semplici, e (nota) tutte e tre indifferenti all'ordine
in cui elenchi i giocatori, che è proprio ciò che ci serve su un grafo.

Sembrano equivalenti, ma non lo sono, e la differenza è più profonda di quanto
sembri. La media e il massimo **dimenticano quanti** sono i nodi; la somma no.

L'esempio più pulito sono due molecole i cui atomi, agli occhi della rete,
portano tutti lo stesso valore, diciamo $7$: solo che una molecola ne ha tre e
l'altra sei. Sono molecole diverse, e possono comportarsi in modo diverso. La
**somma** dà $21$ nella prima e $42$ nella seconda, e le distingue. La **media**
dà $7$ in tutt'e due, il **massimo** pure: le confondono. Contare, a volte, è
tutto.

`````

`````{tab} Superiore

Il readout aggrega l'insieme $\{\mathbf{h}_v^{(K)} : v \in V\}$ in un unico
vettore $\mathbf{h}_G \in \mathbb{R}^{d_K}$ con un'operazione invariante a
permutazione, che quindi accetta un numero variabile di vettori e ne
restituisce sempre uno solo della stessa lunghezza; tipicamente
$\mathbf{h}_G = \sum_v \mathbf{h}_v^{(K)}$, oppure la media o il massimo.
Esistono anche schemi di **pooling gerarchico** (per esempio *DiffPool*), che
alternano message passing e fusione di gruppi di nodi in super-nodi, costruendo
il vettore del grafo per livelli, come il pooling delle CNN accorpa regioni
dell'immagine.

La scelta dell'aggregatore non è un dettaglio implementativo: decide il
**potere espressivo** della rete, cioè quali grafi diversi essa riesce a
distinguere. Il risultato di riferimento è di Xu, Hu, Leskovec e Jegelka nel
2019 {cite}`xu2019powerful`, e lega le GNN a un classico test di isomorfismo,
il **1-WL** di Weisfeiler–Lehman (noto anche come *color refinement*; le
versioni di ordine superiore, $k$-WL, che il paper non usa, distinguono grafi
che 1-WL confonde). Il test
colora iterativamente i nodi impastando la propria etichetta con il
*multinsieme* delle etichette dei vicini: è esattamente la struttura del
message passing.

Prima di enunciare il risultato conviene dire che cos'è 1-WL, perché il nome
«test di isomorfismo» promette più di quel che mantiene: è un'**euristica
incompleta**, e lo dichiara il paper stesso. Se due grafi ricevono colorazioni
diverse allora non sono isomorfi; se le ricevono uguali non si può concludere
niente. Il controesempio è elementare: un ciclo di sei nodi e due triangoli
separati sono entrambi $2$-regolari, quindi con feature iniziali costanti ogni
nodo ha lo stesso stato a ogni giro in tutti e due, e 1-WL li dichiara
indistinguibili. Lo sono di conseguenza anche per **qualunque** GNN a message
passing, con qualunque MLP: non è un limite di GIN, è il tetto.

Xu et al. dimostrano appunto che **nessuna GNN a message passing può
distinguere due grafi che 1-WL dichiara indistinguibili** (Lemma 2, il tetto
teorico) e che una GNN raggiunge quel tetto (Teorema 3) se sono iniettive tutte
e tre le funzioni in gioco: l'aggregazione sul **multinsieme** dei vicini, la
**combinazione** con lo stato del nodo stesso, e il **readout** finale; e con
abbastanza strati. Sopra tutto sta un'ipotesi che è facile perdere e che regge
il resto: le feature d'ingresso provengono da un insieme **numerabile**. Da qui
la gerarchia:

$$
\text{somma} \;\succ\; \text{media} \;\succ\; \text{massimo},
$$

dove $\succ$ va letto **nel quadro di Xu et al.**, cioè come una gerarchia fra
le *informazioni* che i tre aggregatori conservano dopo una mappa appresa: la
somma conserva il multinsieme (feature **e** molteplicità, quanti vicini di
ciascun tipo); la media lo riduce alla distribuzione, e perde il conteggio; il
massimo lo riduce all'insieme dei tipi presenti, e perde anche le proporzioni.
Non è un ordine totale sui numeri reali presi nudi, ed è utile vedere perché in
una riga: i multinsiemi $\{0, 2\}$ e $\{1, 1\}$ hanno la stessa somma e la
stessa media, e massimi diversi. È esattamente l'ipotesi di numerabilità a
rendere la catena vera nel quadro del paper.

Su queste basi gli autori costruiscono la **Graph Isomorphism Network** (GIN),
la cui regola di aggiornamento è deliberatamente semplice e iniettiva:

$$
\mathbf{h}_v^{(k)} = \mathrm{MLP}^{(k)}\!\Big( \big(1 + \epsilon^{(k)}\big)\, \mathbf{h}_v^{(k-1)} + \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{(k-1)} \Big),
$$

dove $\mathrm{MLP}^{(k)} \colon \mathbb{R}^{d_{k-1}} \to \mathbb{R}^{d_k}$ è un
piccolo percettrone multistrato ed $\epsilon^{(k)}$ uno scalare (appreso o
fissato a 0) che dosa il peso del nodo rispetto ai vicini. Quel termine
$(1+\epsilon^{(k)})\mathbf{h}_v^{(k-1)}$ è precisamente il modo in cui GIN si
compra la seconda delle tre iniettività, quella della combinazione con lo stato
proprio. La somma sui vicini, seguita da un MLP, è quanto basta perché GIN
eguagli il potere di 1-WL: il massimo ottenibile da una GNN a message passing,
con l'avvertenza detta sopra che quel massimo lascia fuori casi elementari come
il ciclo contro i due triangoli.

`````

## A cosa servono: la GNN al lavoro

Il capitolo si è aperto su un antibiotico; è ora di mantenere la promessa e
mostrare dove le GNN, oggi, fanno la differenza.

**Chimica e farmaci.** È il terreno naturale delle GNN: una molecola *è* un
grafo (atomi nei nodi, legami negli archi) e prevederne una proprietà è un
compito a livello di grafo. L'idea di leggere le molecole con reti su grafo
risale ai *fingerprint molecolari neurali* di Duvenaud e colleghi del 2015
{cite}`duvenaud2015convolutional`: prima di allora le caratteristiche di una
molecola da dare in pasto a un modello (quanti anelli, quali gruppi chimici,
che peso) le sceglieva un chimico a mano, una per una; il lavoro di Duvenaud le
fa trovare alla rete, che dalla struttura della molecola ricava da sé la fila
di numeri che la descrive. La punta di diamante è **halicin**, la molecola con
cui si è aperto il capitolo. Vale la pena aggiungere solo quello che lì non era
stato detto: la rete che l'ha pescata è una rete a message passing come quelle
di queste pagine, e la molecola non funziona su un batterio soltanto, ma su
batteri molto diversi fra loro (fra gli altri il bacillo della tubercolosi e
alcuni ceppi
intestinali che ai farmaci più recenti non rispondono più).

**Raccomandazione su grafo.** Il caso industriale più celebre è **PinSage**, il
sistema che Pinterest mette in produzione nel 2018
{cite}`ying2018graph` per suggerire contenuti. Il suo grafo ha da una parte le
immagini salvate dagli utenti e dall'altra le bacheche in cui finiscono, con un
arco ogni volta che un'immagine sta in una bacheca: è il grafo **bipartito**
dell'introduzione, i nodi divisi in due squadre e archi solo fra una squadra e
l'altra.
PinSage è, nella sostanza, un GraphSAGE portato a scala web: campiona i vicini
con brevi cammini casuali e li aggrega, girando su un grafo di tre miliardi di
nodi. Raccomandare, come si è detto all'inizio del capitolo, è *link
prediction* su un grafo utente-prodotto, ed è il capitolo seguente, dedicato ai
sistemi di raccomandazione, a riprendere il tema per intero.

**Rilevamento frodi.** Le transazioni finanziarie formano un grafo (conti nei
nodi, pagamenti negli archi) e le frodi vivono nelle *relazioni*: anelli di
conti che si rimpallano denaro, o decine di conti che confluiscono tutti sullo
stesso, prestato da qualcuno perché il denaro ci transiti (in gergo, un
«mulo»). Un
classificatore che guardi i conti uno per uno non lo vede; una GNN, che
propaga segnale lungo gli archi, sì. È il motivo per cui l'antiriciclaggio e la
difesa dei pagamenti sono fra i primi luoghi in cui queste reti sono entrate in
esercizio.

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

Un secondo problema è opposto e complementare, e riguarda l'informazione che
sta **lontana**, a molti passi di distanza. Il guaio è di capienza. Allargando
il giro di un passo, i nodi che devono farsi sentire raddoppiano, triplicano,
decuplicano; ma la fila di numeri su cui il nodo scrive quel che ha sentito ha
sempre la stessa lunghezza. E c'è di peggio: se due parti del grafo sono unite
da un solo arco, tutto quello che l'una ha da dire all'altra deve passare da
lì. Un imbuto, e più lontano si va più si stringe. Questo schiacciamento
dell'informazione lontana si chiama **over-squashing**.

Si aggiungono la fatica di girare su grafi da miliardi di archi, e il fatto che
quasi tutte le GNN danno per scontato che i nodi collegati si somiglino (gli
amici hanno gusti simili): quando è il contrario, cioè nelle reti dove chi è
connesso è *diverso*, rendono molto meno.

`````

`````{tab} Superiore

- **Oversmoothing.** Li, Han e Wu (2018) mostrano che uno strato GCN è, in
  sostanza, un passo di *smoothing* laplaciano: iterandolo molte volte le
  feature dei nodi convergono verso un punto fisso che dipende dai gradi e non
  dai nodi, rendendoli indistinguibili. La derivazione spettrale della sezione
  sul message passing lo rende meccanico: $\hat{\mathbf{A}}$ ha autovalori in
  $[-1,1]$ con il massimo pari a $1$, quindi $\hat{\mathbf{A}}^K$ spegne tutte
  le componenti tranne quella lungo l'autovettore dominante, che è
  $\tilde{\mathbf{D}}^{1/2}\mathbf{1}$ e non distingue un nodo dall'altro. È la
  ragione teorica per cui, oltre pochi strati, l'accuratezza crolla. I rimedi
  hanno nomi e forme precise, e vale la pena averli in mente perché sono tre
  risposte diverse alla stessa domanda. **Highway GCN** (Rahimi e colleghi,
  2018) mette un *gate* per strato che decide quanto del vecchio stato lasciar
  passare accanto al nuovo, e nei loro esperimenti le prestazioni smettono di
  migliorare attorno ai quattro strati. **Jumping Knowledge Network** (Xu e
  colleghi, 2018) parte da un'osservazione diversa, cioè che nodi diversi
  vogliono campi recettivi diversi (un hub satura in due salti, un nodo
  periferico no), e quindi invece di prendere l'uscita dell'ultimo strato le
  **concatena tutte**, lasciando che sia il modello a scegliere la profondità
  nodo per nodo. **DeepGCN** (Li e colleghi, 2019) importa di peso residui e
  connessioni dense da ResNet e DenseNet contro i gradienti che svaniscono, e
  aggiunge un vicinato **dilatato** (si prendono i vicini saltandone alcuni)
  contro l'oversmoothing: con questa ricetta arrivano a 56 strati su nuvole di
  punti. Restano eccezioni, però: il vincolo pratico alla profondità è ancora
  la regola.
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

## Graph Transformer: togliere il vincolo del vicinato

Dei limiti appena elencati, l'over-squashing dipende da com'è fatto il grafo: il
message passing fa parlare **solo i nodi collegati**, quindi l'informazione
lontana deve attraversare molti strati e si strozza nei passaggi obbligati.
Viene naturale chiedersi cosa succeda a togliere quel vincolo e a lasciar
parlare tutti con tutti. La risposta arriva dall'altro capo del libro, ed è
meno lieta di come la si racconta di solito.

Dei quattro limiti appena elencati ne tocca due, e ne guarisce uno solo:
conviene sapere subito quale. Guarisce
l'**over-squashing**, perché se ogni nodo parla con ogni altro non c'è più
niente da far transitare per strade strette. Non guarisce l'**oversmoothing**,
anzi. L'oversmoothing non nasce dalla distanza fra i nodi, ma dal fatto che a
ogni giro si fa una media con i vicini; e se i vicini diventano tutti, la media
cancella le differenze ancora più in fretta. Togliere il vincolo del vicinato,
insomma, sul secondo limite **peggiora** le cose.

`````{tab} Elementare

L'introduzione al capitolo ha stabilito che l'attenzione dei Transformer è, di
fatto, message passing su un **grafo completo**: ogni parola parla con tutte le
altre. Se è così, la strada per un grafo è ovvia: mettiamoci un Transformer
sopra e lasciamo che ogni nodo parli con ogni altro, senza aspettare che il
messaggio faccia tutto il giro lungo gli archi.

C'è però un guaio che si vede subito: **se tutti parlano con tutti, il grafo
non conta più niente**. Un Transformer applicato all'elenco dei nodi darebbe
lo stesso risultato se gli archi fossero altri, o se non ce ne fosse nessuno.
Abbiamo tolto il problema e con esso l'informazione.

Ed è lo stesso identico problema che i Transformer avevano con le frasi:
l'attenzione non sa in che ordine stiano le parole, e la soluzione fu dare a
ogni posizione una firma numerica costruita con delle onde. Qui serve
l'equivalente: **dare a ogni nodo una firma che dica dove sta nel grafo**, e
poi lasciar parlare tutti con tutti.

E adesso viene la parte bella, che questo capitolo ha già preparato senza
dirlo. Quelle firme esistono, e le abbiamo già incontrate: sono le
configurazioni di numeri sui nodi che nella sezione sul message passing abbiamo
chiamato le **frequenze** del grafo, dalla più liscia (tutti lo stesso numero)
alla più a scacchiera, e che là avevano preso il loro nome proprio di
**autovettori del laplaciano**. La prima firma dice grossomodo «da che parte
del grafo stai»; quelle dopo lo dicono con un dettaglio via via più fine. È
questo il modo per dare a ogni nodo una posizione senza inventarsela: gliela dà
la forma del grafo.

E su una fila di nodi, cioè sul grafo più semplice che esista, quelle
configurazioni sono **onde**, ordinate dalla più lenta alla più rapida: proprio
come le onde con cui i Transformer segnano la posizione delle parole in una
frase. Qui però conviene non tirare la corda più di quanto regga: sono onde
**imparentate, non le stesse onde**. Le due famiglie si costruiscono in modi
diversi e nessuna delle due si ottiene dall'altra. Quel che è vero, ed è già
molto, è che la stessa idea (segnare una posizione con onde di frequenza
crescente) qui non va scelta a mano, esce dalla forma del grafo, e funziona su
un grafo qualunque invece che soltanto su una fila.

`````

`````{tab} Superiore

Un **Graph Transformer** sostituisce l'aggregazione sui vicini con
un'attenzione su **tutte** le coppie di nodi. Il beneficio è strutturale:
ogni nodo raggiunge ogni altro in **un solo passo**, quindi l'over-squashing
sparisce per costruzione e non serve profondità per avere portata.

L'affermazione fatta sopra, che l'oversmoothing invece peggiora, si misura in
una riga. Su un grafo completo con i cappi l'adiacenza $\tilde{\mathbf{A}}$ è
$\mathbf{J}$, la matrice fatta di soli uno, tutti i gradi valgono $N$ e quindi
$\hat{\mathbf{A}} = \tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}
\tilde{\mathbf{D}}^{-1/2} = \mathbf{J}/N$, il cui spettro è $1$ una volta e $0$
le altre $N-1$ volte: il secondo autovalore è **esattamente zero**, e un solo
passo porta tutti i nodi allo stesso valore.
Sulla catena di quattro nodi della sezione sul message passing lo stesso
autovalore valeva $0{,}729$, cioè servivano decine di passi. Più il grafo è connesso, più
il collasso è rapido, e il grafo completo è il caso estremo.

Il costo dell'attenzione piena è altrettanto strutturale del beneficio:
$O(N^2)$ nel numero di nodi, che su un grafo da milioni di nodi
non è praticabile senza le stesse approssimazioni sparse viste nel capitolo sui
Transformer (e il cerchio si chiude, perché quelle approssimazioni erano
descritte proprio come sparsificazione di un grafo).

Il problema da risolvere è che l'attenzione piena **non prende $\mathbf{A}$ in
ingresso**: la sua uscita è funzione del solo multinsieme delle feature dei
nodi, ed è quindi la stessa qualunque siano gli archi, o se non ce ne fosse
nessuno. Senza informazione aggiuntiva il modello non distingue un anello da
una stella. Vale la pena non attribuirlo all'invarianza alle permutazioni, che
è un'altra cosa e che questo capitolo ha passato due sezioni a presentare come
la proprietà *desiderabile*: anche una GNN è equivariante alle permutazioni, e
non è affatto cieca alla topologia. Sono due simmetrie diverse. La GNN è
equivariante rispetto al gruppo che permuta $(\mathbf{A}, \mathbf{X})$
**insieme**; il Transformer nudo è invariante rispetto a un gruppo molto più
grande, che permuta $\mathbf{X}$ e ignora $\mathbf{A}$. Non è un difetto di
simmetria, è un'informazione che non entra, e la si deve reiniettare: le due
strade sono quelle che il capitolo sui Transformer già conosce.

La prima è una **codifica posizionale**: si calcolano i primi $k$ autovettori
non banali del laplaciano normalizzato
$\mathbf{L} = \mathbf{U}\boldsymbol{\Lambda} \mathbf{U}^\top$ e si prende la
riga $i$-esima di $\mathbf{U}_{:,1:k}$ come firma del nodo $i$, che chiamiamo
$\mathbf{p}_i$. Nel lavoro che ha introdotto la costruzione
{cite}`dwivedi2020generalization` quella firma **si somma** alle feature del
nodo dopo una proiezione lineare
($\mathbf{p}_i^0 = \mathbf{C}^0\mathbf{p}_i + \mathbf{c}^0$ con
$\mathbf{C}^0 \in \mathbb{R}^{d \times k}$, poi
$\mathbf{h}_i^0 = \hat{\mathbf{h}}_i^0 + \mathbf{p}_i^0$), non si concatena: la
proiezione serve proprio perché $k$ e $d$ non coincidono. È la stessa mossa che
il capitolo sui Transformer descrive per la codifica sinusoidale, dove la firma
della posizione si **somma** all'embedding del token invece di affiancarglisi.
Diverse implementazioni successive concatenano invece; e vale la pena notare
che la codifica entra **solo allo strato d'ingresso**, non negli strati
intermedi.

La giustificazione è quella già stabilita in questo capitolo: gli autovettori
sono i modi di variazione del grafo ordinati per frequenza, e su un grafo a
catena sono sinusoidi. È in questo senso che la costruzione spettrale
**generalizza a un grafo qualunque** l'idea della codifica posizionale
sinusoidale, ed è esattamente ciò che gli autori rivendicano («*naturally
generalize*»). Non è però un'identità, e conviene dire dove le due famiglie si
separano, perché sono differenze misurabili: le frequenze degli autovettori del
cammino sono $\pi k/N$, **spaziate linearmente** e legate alla lunghezza $N$,
mentre quelle di Vaswani sono $10000^{-2i/d}$, **geometriche** e indipendenti
dalla lunghezza (è la proprietà per cui il Transformer estrapola a frasi mai
viste); sul cammino gli autovettori sono soli **coseni**, mentre la codifica
sinusoidale accoppia un seno e un coseno per frequenza; e un autovettore è
definito **a meno del segno**, una colonna di codifica posizionale no. Parenti
stretti, insomma, non lo stesso oggetto.

Due avvertenze pratiche, entrambe reali. Gli autovettori sono definiti **a meno
del segno** ($-\mathbf{u}$ è altrettanto valido), e su autovalori ripetuti a
meno di una rotazione dentro l'autospazio: si rimedia campionando il segno a
caso in addestramento, così il modello impara a non dipenderne. E la
decomposizione costa $O(N^3)$, quindi si calcola una volta sola in
preprocessing e solo per i primi $k$ autovettori.

La seconda strada è il **bias di attenzione**: invece di aggiungere qualcosa
ai nodi, si modifica il punteggio di attenzione fra due nodi in funzione della
loro relazione. È la scelta di **Graphormer** {cite}`ying2021transformers`, che
somma ai logit un termine appreso dipendente dalla **distanza sul grafo** fra i
due nodi (più un termine sul grado e uno sugli archi lungo il cammino).
Formalmente è una variante di attenzione relativa, la stessa famiglia di idee
delle codifiche posizionali relative sulle sequenze, e gli autori mostrano che
con questi accorgimenti molte GNN classiche diventano **casi particolari** del
modello.

Le due strade non sono alternative: l'impostazione oggi prevalente le combina,
tenendo un ramo di message passing **accanto** all'attenzione globale, così che
il primo curi la struttura locale e la seconda la portata
{cite}`rampasek2022recipe`. È il riconoscimento onesto che il vicinato non era
un difetto da rimuovere ma un *prior* utile, e che quello che mancava era un
canale per il lontano.

`````

Niente di tutto questo va preso sulla fiducia, e non c'è bisogno di prenderlo:
si verifica su una catena di nodi, che è una sequenza travestita da grafo. Il
conto qui sotto misura, una alla volta, le due affermazioni che è facile
confondere: che quelle configurazioni **sono** onde ordinate per frequenza, e
che **non** sono le stesse onde dei Transformer.

Chi non programma può saltare il riquadro qui sotto e anche i tre paragrafi che
lo commentano: non c'è niente di nuovo, ci sono solo i numeri che reggono
quello che si è appena letto.

```python
import numpy as np

N = 16
# grafo a catena: 0-1-2-...-15. È una sequenza travestita da grafo.
A = np.diag(np.ones(N - 1), 1) + np.diag(np.ones(N - 1), -1)
d = A.sum(1)
L = np.eye(N) - A / np.sqrt(np.outer(d, d))          # laplaciano normalizzato
val, vec = np.linalg.eigh(L)

# 1. i primi autovettori non banali sono coseni, di frequenza crescente
t = np.arange(N)
for k in (1, 2, 3):
    onda = np.cos(np.pi * k * (t + 0.5) / N)
    onda /= np.linalg.norm(onda)
    print(f"autovettore {k}: |somiglianza| con cos(pi*{k}*(n+0.5)/N) = "
          f"{abs(vec[:, k] @ onda):.4f}   (autovalore {val[k]:.3f})")

print("\ngli autovalori crescono:", np.round(val[:5], 3))
# l'ambiguità di segno: -v è un autovettore altrettanto valido
print("il segno è arbitrario: -v risolve la stessa equazione ->",
      np.allclose(L @ (-vec[:, 1]), val[1] * (-vec[:, 1])))

# 2. ma non sono la codifica posizionale del Transformer: confrontiamole
d_model = 16
omega = 10000.0 ** (-2 * np.arange(d_model // 2) / d_model)
PE = np.concatenate([np.sin(np.outer(t, omega)), np.cos(np.outer(t, omega))], axis=1)
PE = PE / np.linalg.norm(PE, axis=0)
S = abs(PE.T @ vec)              # ogni colonna di PE contro ogni autovettore

print("\nfrequenze degli autovettori (pi*k/N):", np.round(np.pi * np.arange(1, 5) / N, 3))
print("frequenze di Vaswani (10000^-2i/d):  ", np.round(omega[:4], 3))
print("colonne di PE piu' simili all'autovettore banale u0:",
      int((S.argmax(1) == 0).sum()), "su", PE.shape[1])
print(f"massima somiglianza con un autovettore non banale: {S[:, 1:].max():.3f}")
```

La prima metà del conto dà $0{,}9891$, $0{,}9851$ e $0{,}9785$: sono tre misure
di somiglianza, e un $1$ vorrebbe dire «la stessa identica onda». I primi tre
autovettori del laplaciano di una catena **sono** dunque i primi tre coseni, e
gli autovalori crescono con la frequenza, esattamente come promesso dalla
lettura spettrale. Che non facciano $1{,}0000$ ha una ragione precisa e non è
rumore numerico: la versione del laplaciano usata qui pesa ogni nodo per quanti
vicini ha, e i due nodi agli estremi della catena ne hanno uno invece di due,
il che deforma leggermente l'onda ai bordi. Con la versione che non pesa
($\mathbf{L} = \mathbf{D} - \mathbf{A}$) la corrispondenza è esatta, $1{,}0000$
su tutti e tre.

La seconda metà dice dove la parentela si ferma. Le frequenze degli autovettori
sono $0{,}196$, $0{,}393$, $0{,}589$, $0{,}785$: crescono a passo costante, e
il passo è $\pi/N$, cioè dipende da quanto è lunga la catena. Quelle scelte nel
lavoro che ha introdotto i Transformer {cite}`vaswani2017attention`
sono $1{,}000$, $0{,}316$, $0{,}100$, $0{,}032$: calano geometricamente e non
sanno niente di $N$, tanto che su una finestra di sedici posizioni le più basse
sono così lente da risultare quasi piatte. La conseguenza si misura:
**dodici colonne su sedici** della codifica del Transformer somigliano più che
altro all'autovettore *banale*, quello costante, e nessuna coincide con un
autovettore vero, la migliore somiglianza fermandosi a $0{,}916$. Due basi di
onde su una linea, ordinate per frequenza, costruite in due modi diversi: la
parentela è reale e utile, l'identità no.

L'ultima riga della prima metà conferma l'ambiguità di segno: la stessa
equazione è soddisfatta da $\mathbf{u}$ e da $-\mathbf{u}$, e nessuna delle due
è «quella giusta». Chi usa queste firme come codifica posizionale deve
conviverci, ed è il motivo per cui in addestramento se ne campiona il segno a
caso.

## L'ecosistema, e dove andare da qui

Chi voglia mettere le mani in pasta non parte da zero: due librerie coprono
quasi tutto. **PyTorch Geometric** (PyG, quella degli esempi di questa sezione)
e la **Deep Graph Library** (DGL) offrono, sopra PyTorch, uno strato già pronto
per ciascuno dei modelli incontrati qui (`GCNConv` per la GCN, `SAGEConv` per
GraphSAGE, `GATConv` per la GAT, `GINConv` per la variante che somma i vicini),
gli arnesi che servono a campionare i vicini e decine di raccolte di dati su cui
provare. Scrivere una GNN, oggi, è questione di poche righe: proprio come lo è
diventato scrivere una rete convoluzionale.

Con questo si chiude il capitolo. Il filo, però, non si spezza: l'attenzione
che qui pesa i vicini di un nodo è la stessa dei Transformer, e i grafi a due
squadre utente-prodotto di questa sezione sono lo stesso oggetto del capitolo
sui sistemi di raccomandazione. Le reti su grafo non sono un'isola. Sono il
punto in cui convoluzione, attenzione e apprendimento di rappresentazioni si
ritrovano, e si ritrovano lì per una ragione precisa, quella con cui il
capitolo si era aperto: ciascuna di loro nasce dall'elenco delle cose che si
possono fare al dato senza cambiarne il significato. Spostare un'immagine,
riordinare i nodi di un grafo. Cambia l'elenco, cambia la rete.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- **GraphSAGE** impara la **ricetta** con cui si costruisce la descrizione di un
  nodo, non la descrizione già fatta: una ricetta la si applica anche a chi non
  c'era durante l'addestramento (l'utente iscritto stamattina) e a reti diverse
  da quella su cui si è imparato, ed è questo che si chiama modo **induttivo**.
  In più, invece di ascoltare tutti i vicini ne pesca a caso un numero fisso
  (venticinque, per dire) e mescola solo quelli: come ci si fa un'idea di un
  quartiere intervistandone un campione a sorte, invece di bussare a ogni porta.
- La **GAT** passa l'evidenziatore sui vicini: prima decide, vicino per vicino,
  quanto pesarlo (a ognuno un voto tra 0 e 1, e i voti sommano a 1), poi fa la
  media pesata con quei voti, che non scrive nessuno a mano ma impara la rete. È
  la stessa attenzione dei Transformer, dove però ogni parola guarda tutte le
  altre: qui ogni nodo guarda solo i vicini a cui è davvero collegato.
- Per un verdetto sull'**intero grafo** («questa molecola è tossica?») i valori
  di tutti i nodi vanno ridotti a uno solo, come si ricava il voto di una squadra
  dai voti dei giocatori: sommandoli, mediandoli o prendendo il massimo. La
  **somma** è la scelta più fine perché è l'unica che ricorda **quanti** sono i
  nodi: se gli atomi di due molecole portano tutti lo stesso valore, ma una
  molecola ne ha tre e l'altra sei, la somma dà il triplo di quel valore nella
  prima e il sestuplo nella seconda, mentre la media e il massimo danno lo stesso
  risultato per entrambe e le confondono.
- Applicazioni reali: farmaci (**halicin**, 2020), raccomandazione (**PinSage**
  di Pinterest, 2018), rilevamento frodi, tempi di percorrenza in **Google Maps**
  (DeepMind, 2020–21), simulazioni di fluidi e previsioni meteo.
- Limiti aperti: troppi strati appiattiscono i nodi fino a renderli
  indistinguibili (**oversmoothing**), l'informazione che sta lontana si perde
  passando per imbuti stretti (**over-squashing**), i grafi da miliardi di
  collegamenti restano cari da addestrare, e quasi tutte queste reti danno per
  scontato che chi è collegato si somigli (gli amici hanno gusti simili): nelle
  reti dove vale il contrario, dove chi è connesso è diverso, rendono molto meno.
  In pratica le GNN restano **basse**: due, tre, quattro strati, di rado di
  più.
- I **Graph Transformer** tolgono il vincolo del vicinato e lasciano parlare
  ogni nodo con ogni altro. Così l'informazione lontana non si perde più per
  strada; ma dei due difetti ne curano uno solo, perché appiattire i nodi fino
  a renderli indistinguibili, con tutti collegati a tutti, viene ancora più in
  fretta. E in più il grafo smette di contare, perché un modello che collega
  tutti con tutti non guarda mai chi è collegato a chi davvero. La struttura va
  ridata, assegnando a ogni nodo una **firma** che dica dove sta nel grafo:
  sono le configurazioni di numeri che qui abbiamo chiamato le frequenze del
  grafo (gli **autovettori del laplaciano**), e su una fila di nodi sono onde
  di frequenza crescente, **parenti** di quelle con cui i Transformer segnano
  la posizione delle parole, non le stesse.
```

`````

`````{tab} Superiore

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
  vicini: **GIN** la usa per eguagliare il test **1-WL** di
  **Weisfeiler–Lehman**, che è il tetto del potere espressivo di una GNN a
  message passing. Il teorema chiede però tre iniettività (aggregazione,
  combinazione, readout), abbastanza strati e feature da un insieme
  numerabile; e quel tetto è un'euristica incompleta, che non distingue per
  esempio un ciclo di sei nodi da due triangoli.
- Applicazioni reali: farmaci (**halicin**, 2020), raccomandazione (**PinSage**,
  Pinterest 2018), rilevamento frodi, tempi di percorrenza in **Google Maps**
  (DeepMind, 2020–21), simulazioni fisiche e meteo.
- Limiti aperti: **oversmoothing** (troppi strati → nodi indistinguibili),
  **over-squashing** (informazione lontana schiacciata), **scalabilità**,
  **eterofilia**. In pratica le GNN restano **basse**, 2–4 strati.
- Un **Graph Transformer** sostituisce l'aggregazione sui vicini con
  l'attenzione su tutte le coppie: ogni nodo raggiunge ogni altro in un passo
  (fine dell'over-squashing, **non** dell'oversmoothing, che sul grafo completo
  peggiora perché lì $\lambda_2 = 0$), al costo di $O(N^2)$ e della perdita
  della topologia, perché l'attenzione piena non prende $\mathbf{A}$ in ingresso. La
  struttura si reinietta come **codifica posizionale** con i primi autovettori
  del laplaciano, sommati alle feature dopo una proiezione lineare e solo allo
  strato d'ingresso, e definiti a meno del **segno**, che in addestramento si
  campiona; oppure come **bias di attenzione** dipendente dalla distanza sul
  grafo (**Graphormer**). Sulla catena quegli autovettori sono sinusoidi, e in
  questo senso la costruzione **generalizza** la codifica sinusoidale a un
  grafo qualunque; non la contiene però come caso particolare, perché le
  frequenze sono $\pi k/N$ e non $10000^{-2i/d}$. Le impostazioni attuali
  tengono i due canali insieme, message passing per il locale e attenzione per
  il lontano.
```

`````
