# Alzare la temperatura: le macchine di Boltzmann

La pallina di Hopfield ha un difetto di fabbrica: può solo scendere. Se
l'indizio la deposita sul pendio sbagliato, finisce nella valle sbagliata (o
in un ricordo fantasma) e da lì non esce più. E c'è un limite più profondo: la
rete *ricorda*, ma non *inventa*; i suoi neuroni coincidono uno a uno con i
pixel del ricordo, e non gliene resta nessuno libero per annotarsi qualcosa di
suo, per esempio che in quel punto c'è una riga verticale (di una rete che non
ha neuroni liberi per queste annotazioni si dice che non ha
**rappresentazioni interne**). A metà anni
Ottanta Geoffrey Hinton e Terrence Sejnowski, con David Ackley, propongono la
**macchina di Boltzmann** {cite}`ackley1985learning`, che aggiunge alla rete
di Hopfield esattamente due ingredienti: la **temperatura** e i **neuroni
nascosti**. Il nome è un omaggio a Ludwig Boltzmann, uno dei padri della
meccanica statistica (la fisica che spiega il comportamento di miliardi di
particelle contando le configurazioni possibili invece di seguirle una per
una), e non è un omaggio generico: lasciata scuotere
abbastanza a lungo, questa rete passa in ogni configurazione la stessa
frazione di tempo che la fisica prevede per un materiale caldo. E il tempo che
ci passa *è* la probabilità che le assegna: se guardi dove si trova un milione
di volte a caso e la trovi in fondo a una certa valle in trentamila occasioni,
quella valle vale il 3%. È la porta da cui un'altezza diventa una percentuale,
e il conto del pedaggio arriva subito dopo.

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
dei dati. Il guaio, come vedremo, è il tempo, e non soltanto quello del sogno:
nella macchina originale costava carissima anche la veglia, perché il conto
delle coppie che si accendono insieme andava rifatto da capo per ogni singolo
dato. E sognare «per bene» richiedeva tempi biblici.

`````

`````{tab} Superiore

Nella macchina di Boltzmann l'aggiornamento del neurone $i$ diventa
stocastico:

$$
P(s_i = +1) = \sigma\!\left(\frac{2 h_i}{T}\right)
= \frac{1}{1 + e^{-2 h_i / T}},
$$

dove $h_i = \sum_j w_{ij} s_j$ è il campo locale, $\sigma$ la sigmoide già
incontrata nel capitolo sulle reti neurali e $T > 0$ la temperatura. Il
fattore 2 non è un refuso e non è universale: viene dalla convenzione
$s_i \in \{-1,+1\}$ ereditata da Hopfield. Il conto è il rapporto di Gibbs fra
i due stati possibili del neurone, che per la sezione precedente valgono
$E(s_i = \pm 1) = \mp h_i + \text{cost}$, e quindi distano $\Delta E = 2h_i$:

$$
P(s_i = +1) = \frac{e^{h_i/T}}{e^{h_i/T} + e^{-h_i/T}}
= \frac{1}{1 + e^{-2h_i/T}} .
$$

È quel salto di $2h_i$, e non $h_i$, a produrre il 2. **Chi confronta con
altre fonti tenga d'occhio la convenzione**: l'articolo originale di Ackley,
Hinton e Sejnowski usa unità in $\{0,1\}$, dove il salto è $h_i$ e la formula
è $\sigma(h_i/T)$ senza il fattore, ed è la stessa forma che tornerà per le
RBM fra poco.

Per $T \to 0$ si ritrova l'aggiornamento deterministico di Hopfield; per $T$
grande la rete accetta spesso anche mosse che *alzano* l'energia, e può
quindi evadere dai minimi locali (abbassare $T$ gradualmente è la *ricottura
simulata*). All'equilibrio termico la rete visita gli stati secondo la
distribuzione di Boltzmann–Gibbs

$$
P(\mathbf{s}) = \frac{e^{-E(\mathbf{s})/T}}{Z},
\qquad
Z = \sum_{\mathbf{s}'} e^{-E(\mathbf{s}')/T},
$$

dove $Z$ (la **funzione di partizione**) somma su tutti i $2^N$ stati
possibili: è lei che rende la rete un vero modello probabilistico, ed è lei
che costerà carissima. Anche qui le ipotesi vanno dette, perché sono tre e
sono tutte necessarie: l'aggiornamento dev'essere **asincrono** (così è
campionamento di Gibbs, e soddisfa il bilancio dettagliato rispetto a questa
distribuzione), la temperatura dev'essere $T > 0$ (a $T = 0$ la catena si
inchioda nel primo minimo) e la scansione dei neuroni dev'essere equa. Con
l'aggiornamento **sincrono**, quello di Little, la distribuzione stazionaria
non è questa: è la stessa differenza che nella sezione precedente faceva
cadere la garanzia di discesa. I neuroni si dividono in **visibili** (dove si
presentano i dati) e **nascosti** (variabili latenti che catturano regolarità
di ordine superiore). L'apprendimento massimizza la verosimiglianza dei dati
sui visibili, e il gradiente ha una forma di contrasto di rara eleganza:

$$
\Delta w_{ij} \;\propto\; \langle s_i s_j \rangle_{\text{dati}}
- \langle s_i s_j \rangle_{\text{modello}},
$$

dove $\langle \cdot \rangle$ è il valore atteso (le parentesi angolari sono la
notazione dei fisici per l'$\mathbb{E}[\cdot]$ del resto del libro, e qui si
tengono perché è così che la formula si trova in letteratura), il primo
termine è la correlazione media tra i neuroni $i$ e $j$ con i
visibili bloccati sui dati (fase positiva, la «veglia») e il secondo la
stessa correlazione con la rete libera di campionare da sé (fase negativa,
il «sogno»). Il «$\propto$» nasconde un $1/T$: il tasso di apprendimento
effettivo dipende dalla temperatura a cui si raccolgono le statistiche. La
derivazione è quella della sezione sulla partizione, applicata due volte:
$\partial E/\partial w_{ij} = -s_i s_j$, e poiché i dati vincolano solo i
visibili bisogna passare da $p(\mathbf{v}) = \sum_{\mathbf{h}}
p(\mathbf{v},\mathbf{h})$, il che fa comparire una **seconda** media, quella
sui nascosti dati i visibili. Da lì i due termini. Nella macchina di Boltzmann originale, a connettività generale,
il problema non è solo il secondo termine: lo sono tutti e due. Con i visibili
bloccati sui dati la media $\langle s_i s_j \rangle_{\text{dati}}$ non ha
forma chiusa, perché le unità nascoste sono interconnesse fra loro, e va
stimata anch'essa portando una catena all'equilibrio, per *ogni* vettore
d'addestramento. Ackley, Hinton e Sejnowski, nell'esperimento 40-10-40 del
loro articolo, ricuociono la rete una volta con ciascuno dei quaranta vettori
bloccati e altrettante volte senza bloccare niente, per ogni passo di
gradiente: ottanta ricotture per un solo aggiornamento dei pesi, e la fase
positiva ne costa esattamente quanto la negativa, perché va rifatta dato per
dato. È questo doppio ciclo a rendere l'algoritmo originale inutilizzabile
oltre i problemi giocattolo, ed è metà esatta di quel doppio ciclo che l'RBM,
fra poco, farà sparire.

`````

## Il sogno abbreviato: contrastive divergence

La via d'uscita arriva quasi vent'anni dopo, ed è di nuovo di Hinton: la
**contrastive divergence** {cite}`hinton2002training`, che in italiano
suonerebbe «divergenza contrastiva», dove il contrasto è quello fra veglia e
sogno di cui si è appena detto. In una riga: rinunciare
al sogno completo. Invece di lasciar sognare la macchina finché il sogno non
si assesta, la si fa partire da una cosa vera e le si concede un istante solo
di fantasia.

`````{tab} Elementare

Il guaio, si diceva, era il sogno: per farlo «per bene» la macchina deve
sognare finché il sogno non si assesta, e ci mette un tempo che non abbiamo.
Allora si bara, e si bara in due modi.

Il primo è quello appena detto: invece di lasciarla partire dal nulla, le si
mette davanti una cosa vera e le si concede un istante solo di fantasia. Il
sogno che ne esce è appena abbozzato, costa un attimo invece di un'eternità, e
basta lo stesso. Il difetto è prevedibile: partendo sempre da cose vere, la
macchina non va mai a curiosare nelle regioni in cui si sbaglia di grosso, e
quelle regioni restano sbagliate perché nessuno ci va ad alzare il terreno.

Il secondo modo rimedia proprio a questo, e non costa niente di più: non far
ricominciare il sogno da capo ogni volta, ma lasciar continuare quello di
prima. Un po' per volta il sogno si allontana e finisce anche nei posti dove
la macchina si illude.

C'è un prezzo, ed è meglio saperlo che credere di aver trovato una scorciatoia
gratis: con il sogno abbreviato la macchina non sta più migliorando nessuna
misura precisa, e quel che si guadagna in velocità si perde in garanzie. In
pratica, sulle reti di allora, funzionava benissimo.

`````

`````{tab} Superiore

Invece di far girare la catena fino all'equilibrio, la si fa partire *dai
dati* e la si ferma dopo un solo passo (o pochi), usando quel sogno appena
abbozzato come surrogato della fase negativa. Funziona soprattutto sulle
**macchine di Boltzmann ristrette** (RBM), la variante in cui i collegamenti
esistono solo tra strato visibile e strato nascosto: lì i nascosti sono
indipendenti fra loro dati i visibili (e viceversa), quindi la fase positiva
ha forma chiusa e ogni strato si campiona in blocco, in parallelo. È l'RBM a
riparare la metà cara di cui sopra; la contrastive divergence accorcia
l'altra.

Sulla natura di quell'aggiornamento conviene essere precisi, perché la formula
abbreviata non è «il gradiente giusto, con un errore». Sutskever e Tieleman
ne danno due dimostrazioni: l'aggiornamento CD1 *noiseless* (cioè con le
attese calcolate esattamente, non stimate) per RBM binarie **non è il
gradiente di alcuna funzione** {cite}`sutskever2010convergence`. Non esiste
cioè un obiettivo di cui sia una stima, nemmeno distorta; e si può costruire
un termine di penalità (artificioso, ma legittimo) che lo fa ciclare
all'infinito invece di fermarsi, mentre con la penalità $L^2$ di tutti i
giorni gli stessi autori dimostrano che un punto fisso esiste.

Il perimetro dell'enunciato è stretto e istruttivo, e i due autori lo tracciano
richiamando risultati anteriori di Aapo Hyvärinen (2007): se la catena è di
Langevin lo stesso aggiornamento *diventa* il gradiente dello score matching
(la prossima sezione), e se campiona una componente a caso dalla condizionale
diventa quello della pseudo-verosimiglianza. È proprio nel caso comune, Gibbs
su RBM binarie, che non è il gradiente di niente. In pratica funzionava lo
stesso: è uno di quei casi in cui un campo ha usato per anni uno strumento
senza la proprietà che gli attribuiva.

Il compromesso ha un secondo difetto, questo intuitivo: partendo sempre dai
dati, la catena esplora solo i dintorni di ciò che ha già visto, e le regioni
in cui il modello mette per sbaglio molta probabilità restano inesplorate,
perché nessuno va a farvi salire l'energia. Il rimedio più semplice è la
**persistent contrastive divergence** {cite}`tieleman2008training`: non far
ripartire la catena dai dati a ogni passo, ma tenerne una che prosegue da dove
era arrivata, così che nel corso dell'addestramento il «sogno» abbia il tempo
di allontanarsi e di visitare il paesaggio. È un'idea che ritroveremo intatta,
con un serbatoio di campioni al posto della singola catena, nei modelli a
energia sulle immagini di una decina d'anni dopo.

`````

La macchina che ha lasciato il segno, però, non è quella piena di Ackley,
Hinton e Sejnowski: è una sua versione sfoltita, in cui i collegamenti restano
soltanto fra i neuroni dei dati e i taccuini interni, e nessuno più fra un
taccuino e l'altro. Si chiama **macchina di Boltzmann ristretta**, e tutti la
chiamano con la sigla inglese, **RBM**. È quella potatura a far cadere la
prima delle due metà care, la veglia, che con i taccuini scollegati fra loro
si calcola in un colpo solo; la contrastive divergence accorcia la seconda, il
sogno.

Fu proprio la coppia RBM più contrastive divergence, impilata strato su
strato {cite}`hinton2006fast`, a rimettere in moto il deep learning a metà
anni Duemila, quando
addestrare reti profonde sembrava impossibile. È un ruolo storico che va
riconosciuto con onestà, insieme al suo epilogo: di lì a pochi anni ReLU, GPU
e dataset più grandi avrebbero reso superfluo quel modo di partire (si sgrossava
la rete uno strato alla volta prima di addestrarla per intero, ed è il
*pre-training* di cui si parlava allora), e oggi le RBM
non si usano quasi più. Il *linguaggio* con cui erano scritte, invece, è vivo
e vegeto: nella prossima sezione si vede perché, e quanto costi davvero la
misura dell'intero paesaggio che qui è appena entrata in scena, quella che
trasforma un'altezza in una percentuale.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- La **macchina di Boltzmann** aggiunge alla rete di Hopfield due cose: la
  possibilità di risalire ogni tanto (si scuote il paesaggio, e quella scossa
  si chiama **temperatura**) e qualche neurone in più che non corrisponde a
  nessun pixel, buono per annotarsi le regolarità del dato.
- Si scuote forte all'inizio e sempre più piano, come il fabbro che scalda il
  metallo e lo lascia raffreddare adagio: così la pallina esce dalle conche
  mediocri finché può, e si assesta in una valle profonda quando la calma
  torna.
- Imparare è un confronto fra **veglia e sogno**: si guarda che cosa succede
  nella rete quando le si mostrano i dati veri, poi che cosa succede quando la
  si lascia fantasticare da sola, e si ritoccano i legami per rinforzare la
  prima e indebolire la seconda. Si smette quando i sogni sono
  indistinguibili dalla veglia.
- Il guaio è il tempo, e sono cari tutti e due i gesti: la veglia va rifatta
  da capo con ogni dato, e il sogno fatto per bene non finisce mai. La **rete
  ristretta** (in sigla **RBM**: i taccuini interni scollegati fra loro)
  sistema la veglia; la **contrastive divergence** bara sul sogno, concedendo
  alla macchina un istante solo di fantasia a partire da una cosa vera.
  Funziona, ma è una scorciatoia, non una soluzione: nessuno sa più che cosa la
  macchina stia esattamente migliorando.
- Da qui in avanti l'altezza del paesaggio diventa una percentuale, e per
  trasformarla servirebbe la misura dell'intero continente. Quel conto ha un
  nome, **funzione di partizione**, ed è il personaggio a cui è intitolata la
  prossima sezione.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- La **macchina di Boltzmann** {cite}`ackley1985learning` aggiunge a Hopfield
  la **temperatura** (aggiornamenti stocastici, quindi la possibilità di
  risalire e uscire dai minimi sbagliati) e i **neuroni nascosti**
  (rappresentazioni interne, non solo pixel).
- All'equilibrio la rete campiona dalla distribuzione di Boltzmann–Gibbs
  $P(\mathbf{s}) = e^{-E(\mathbf{s})/T}/Z$: da qui in avanti l'energia definisce una
  probabilità, e con essa arriva la **funzione di partizione** $Z$.
- L'apprendimento è un **contrasto** fra fase positiva (dati) e fase negativa
  (campioni del modello). Nella macchina originale entrambe richiedono una
  catena portata all'equilibrio, e la positiva va rifatta per ogni dato:
  l'RBM rende chiusa la prima, e resta la seconda come collo di bottiglia.
- La **contrastive divergence** {cite}`hinton2002training` accorcia la catena
  a uno o pochi passi partendo dai dati; la **persistent CD**
  {cite}`tieleman2008training` la fa proseguire fra un aggiornamento e
  l'altro. L'aggiornamento CD1 non è il gradiente di nessuna funzione
  {cite}`sutskever2010convergence`: funziona in pratica, ma non esiste un
  obiettivo che stia massimizzando. RBM e CD hanno avuto un ruolo storico nel
  far ripartire il deep learning, e oggi sono quasi solo storia; il linguaggio
  dell'energia no.
```
`````
