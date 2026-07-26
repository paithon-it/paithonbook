# Alzare la temperatura: le macchine di Boltzmann

La pallina di Hopfield ha un difetto di fabbrica: può solo scendere. Se
l'indizio la deposita sul pendio sbagliato, finisce nella valle sbagliata (o
in un ricordo fantasma) e da lì non esce più. E c'è un limite più profondo: la
rete *ricorda*, ma non *inventa*; i suoi neuroni coincidono con i pixel del
pattern, senza spazio per rappresentazioni interne. A metà anni Ottanta
Geoffrey Hinton e Terrence Sejnowski, con David Ackley, propongono la
**macchina di Boltzmann** {cite}`ackley1985learning`, che aggiunge alla rete
di Hopfield esattamente due ingredienti: la **temperatura** e i **neuroni
nascosti**. Il nome è un omaggio a Ludwig Boltzmann, uno dei padri della
meccanica statistica: come vedremo, all'equilibrio la rete visita gli stati
con le stesse probabilità con cui un sistema fisico caldo visita le proprie
configurazioni.

`````{tab} Elementare

La temperatura è una scossa. Immagina la pallina ferma in una conca che non è
la valle giusta: se il paesaggio resta immobile, non ne uscirà mai. Ora scuoti
tutto, come una biglia in una scatola da scarpe: con scossoni forti la biglia
salta fuori anche dalle valli profonde e gira dappertutto; con scossoni deboli
resta confinata nei fondovalle. Il trucco è scuotere forte all'inizio e sempre
più piano (è la mossa del fabbro, che scalda il metallo e lo lascia
raffreddare lentamente perché gli atomi trovino da soli la disposizione
migliore), così la biglia ha modo di uscire dalle conche mediocri finché può, e
di assestarsi in una valle profonda quando la calma torna.

I neuroni nascosti, invece, sono taccuini interni: neuroni che non
corrispondono a nessun pixel del dato ma servono alla rete per annotare
regolarità sue («qui c'è una riga verticale», «questi due angoli vanno
insieme»). E l'apprendimento diventa un confronto tra due modi di stare al
mondo: nella fase di *veglia* la macchina osserva i dati veri e registra
quali coppie di neuroni si accendono insieme; nella fase di *sogno* viene
lasciata libera di produrre stati per conto suo, e si registra la stessa
cosa. Poi i pesi si ritoccano per rinforzare ciò che accade da svegli più
che in sogno, e indebolire il contrario. Si smette quando i sogni sono
indistinguibili dalla veglia: a quel punto la macchina si è fatta un modello
dei dati. Il guaio, come vedremo, è che sognare «per bene» richiedeva tempi
biblici.

`````

`````{tab} Superiore

Nella macchina di Boltzmann l'aggiornamento del neurone $i$ diventa
stocastico:

$$
P(s_i = +1) = \sigma\!\left(\frac{2 h_i}{T}\right)
= \frac{1}{1 + e^{-2 h_i / T}},
$$

dove $h_i = \sum_j w_{ij} s_j$ è il campo locale, $\sigma$ la sigmoide già
incontrata nel capitolo sulle reti neurali e $T > 0$ la temperatura. Per
$T \to 0$ si ritrova l'aggiornamento deterministico di Hopfield; per $T$
grande la rete accetta spesso anche mosse che *alzano* l'energia, e può
quindi evadere dai minimi locali (abbassare $T$ gradualmente è la *ricottura
simulata*). All'equilibrio termico la rete visita gli stati secondo la
distribuzione di Boltzmann–Gibbs

$$
P(s) = \frac{e^{-E(s)/T}}{Z},
\qquad
Z = \sum_{s'} e^{-E(s')/T},
$$

dove $Z$ (la **funzione di partizione**) somma su tutti i $2^N$ stati
possibili: è lei che rende la rete un vero modello probabilistico, ed è lei
che costerà carissima. I neuroni si dividono in **visibili** (dove si
presentano i dati) e **nascosti** (variabili latenti che catturano regolarità
di ordine superiore). L'apprendimento massimizza la verosimiglianza dei dati
sui visibili, e il gradiente ha una forma di contrasto di rara eleganza:

$$
\Delta w_{ij} \;\propto\; \langle s_i s_j \rangle_{\text{dati}}
- \langle s_i s_j \rangle_{\text{modello}},
$$

dove il primo termine è la correlazione media tra i neuroni $i$ e $j$ con i
visibili bloccati sui dati (fase positiva, la «veglia») e il secondo la
stessa correlazione con la rete libera di campionare da sé (fase negativa,
il «sogno»). Il problema pratico è tutto nel secondo termine: stimarlo
richiede di portare all'equilibrio una catena di Markov su uno spazio
esponenziale, per *ogni* passo di gradiente. È questo doppio ciclo a rendere
l'algoritmo originale inutilizzabile oltre i problemi giocattolo.

`````

## Il sogno abbreviato: contrastive divergence

La via d'uscita arriva quasi vent'anni dopo, ed è di nuovo di Hinton: la
**contrastive divergence** {cite}`hinton2002training`. L'idea è rinunciare al
sogno completo: invece di far girare la catena fino all'equilibrio, la si fa
partire *dai dati* e la si ferma dopo un solo passo (o pochi), usando quel
sogno appena abbozzato come surrogato della fase negativa. Il gradiente che ne
esce è distorto, ma in pratica funziona, soprattutto sulle **macchine di
Boltzmann ristrette** (RBM), la variante in cui i collegamenti esistono solo
tra strato visibile e strato nascosto, così che ogni strato si campiona in
blocco, in parallelo.

Il compromesso ha un difetto noto: partendo sempre dai dati, la catena esplora
solo i dintorni di ciò che ha già visto, e le regioni in cui il modello mette
per sbaglio molta probabilità restano inesplorate: nessuno va a farvi salire
l'energia. Il rimedio più semplice è la **persistent contrastive divergence**
{cite}`tieleman2008training`: non far ripartire la catena dai dati a ogni
passo, ma tenerne una che prosegue da dove era arrivata, così che nel corso
dell'addestramento il «sogno» abbia il tempo di allontanarsi e visitare il
paesaggio. È un'idea che ritroveremo intatta, con un serbatoio di campioni al
posto della singola catena, negli EBM sulle immagini di quindici anni dopo.

Fu proprio la coppia RBM più contrastive divergence, impilata strato su
strato, a rimettere in moto il deep learning a metà anni Duemila, quando
addestrare reti profonde sembrava impossibile: un ruolo storico che va
riconosciuto con onestà, insieme al suo epilogo: di lì a pochi anni ReLU, GPU
e dataset più grandi avrebbero reso quel pre-training superfluo, e oggi le RBM
non si usano quasi più. Il *linguaggio* con cui erano scritte, invece, è vivo
e vegeto: nella prossima sezione si vede perché, e quanto costi davvero la $Z$
che qui è appena comparsa.

```{admonition} Da ricordare
:class: important
- La **macchina di Boltzmann** {cite}`ackley1985learning` aggiunge a Hopfield
  la **temperatura** (aggiornamenti stocastici, quindi la possibilità di
  risalire e uscire dai minimi sbagliati) e i **neuroni nascosti**
  (rappresentazioni interne, non solo pixel).
- All'equilibrio la rete campiona dalla distribuzione di Boltzmann–Gibbs
  $P(s) = e^{-E(s)/T}/Z$: da qui in avanti l'energia definisce una
  probabilità, e con essa arriva la **funzione di partizione** $Z$.
- L'apprendimento è un **contrasto** fra fase positiva (dati) e fase negativa
  (campioni del modello). La seconda richiede una catena di Markov portata
  all'equilibrio a ogni passo: è il collo di bottiglia.
- La **contrastive divergence** {cite}`hinton2002training` accorcia la catena
  a uno o pochi passi partendo dai dati; la **persistent CD**
  {cite}`tieleman2008training` la fa proseguire fra un aggiornamento e
  l'altro. RBM e CD hanno avuto un ruolo storico nel far ripartire il deep
  learning, e oggi sono quasi solo storia; il linguaggio dell'energia no.
```
