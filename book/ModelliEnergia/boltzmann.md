# Alzare la temperatura: le macchine di Boltzmann

La pallina di Hopfield ha un difetto di fabbrica: può solo scendere. Se
l'indizio la deposita sul pendio sbagliato, finisce nella valle sbagliata (o
in un ricordo fantasma) e da lì non esce più.

E c'è un limite più profondo: la rete *ricorda*, ma non *inventa*. Le sue
venticinque caselle coincidono una a una con le venticinque caselle del
disegno da ricordare, e non gliene resta nessuna libera per annotarsi qualcosa
di suo, per esempio che nelle tre lettere ricorre spesso una riga verticale al
centro. Di una rete che non ha caselle libere per queste annotazioni si dice
che non ha **rappresentazioni interne**.

La risposta a tutti e due i limiti si chiama **macchina di Boltzmann**, e
aggiunge alla rete di Hopfield esattamente due ingredienti: la **temperatura**
(la scossa di cui si diceva in apertura di capitolo) e i **neuroni nascosti**,
che sono quelle caselle libere per gli appunti. Il nome compare già nel 1983,
in un lavoro di Scott Fahlman, Geoffrey Hinton e Terrence Sejnowski;
l'articolo che ne fissa l'algoritmo di apprendimento, quello di cui si parla
qui, è del 1985 e porta la firma di David Ackley, Hinton e Sejnowski
{cite}`ackley1985learning`.

Il nome è un omaggio a Ludwig Boltzmann, uno dei padri della meccanica
statistica, cioè della fisica che spiega il comportamento di miliardi di
particelle contando le configurazioni possibili invece di seguirle una per
una. E non è un omaggio generico. Se si lascia scuotere questa rete abbastanza
a lungo, e poi si guarda dov'è a intervalli a caso, si scopre che passa più
tempo nelle valli profonde e pochissimo sulle cime, in proporzioni che
Boltzmann aveva calcolato un secolo prima per un gas o per un pezzo di metallo
caldo. Sono le stesse proporzioni, con la stessa formula.

E quel tempo *è* la probabilità che la rete assegna a una configurazione: se
la guardi un milione di volte e la trovi in fondo a una certa valle in
trentamila occasioni, quella valle vale il 3%. È la porta da cui un'altezza
diventa una percentuale, e il pedaggio da pagare per attraversarla è il
personaggio della sezione successiva.

`````{tab} Elementare

La temperatura è una scossa. La pallina è ferma in una conca che non è la
valle giusta, e se il paesaggio resta immobile non ne esce più. Scuoti tutto,
come faresti con una scatola da scarpe che ha dentro una pallina: con scossoni
forti salta fuori anche dalle valli profonde e gira dappertutto; con scossoni
deboli resta confinata nei fondovalle. Il trucco è scuotere forte all'inizio e
sempre più piano, così esce dalle conche mediocri finché può e si assesta in
una valle profonda quando la calma torna.

I neuroni nascosti sono taccuini interni: caselle che non corrispondono a
nessuna casella del dato e servono alla rete per annotare regolarità sue
(«qui c'è una riga verticale», «questi due angoli vanno insieme»).

Imparare diventa un confronto fra due modi di stare al mondo. Nella *veglia*
la macchina guarda i dati veri e segna quali coppie di caselle si accendono
insieme; le coppie, perché i suoi legami collegano due caselle per volta, e
quali coppie vanno d'accordo è tutto quello che può imparare. Nel *sogno* la
si lascia inventare configurazioni per conto suo, e si segna la stessa cosa.

Poi si ritoccano i legami, e ritoccare un legame vuol dire deformare il
paesaggio: sono i legami a decidere l'altezza di ogni punto. Rinforzare quello
che si vede da svegli abbassa il terreno sotto i dati veri; indebolire quello
che si vede solo in sogno lo alza sotto le fantasie. Un gesto solo, guardato
da due parti. Si smette quando i sogni sono indistinguibili dalla veglia:
quello che la macchina si immagina ha le stesse regolarità di quello che ha
visto.

Il guaio è il tempo. Sognare per bene vuol dire lasciarla scuotere finché le
proporzioni non smettono di cambiare, cioè finché altre mille occhiate non
spostano più i conteggi: è il momento in cui si è fotografato il paesaggio e
non un pezzo di passeggiata, e arriva tardissimo. Nella macchina originale
costava carissima anche la veglia: con i dati veri sotto gli occhi i taccuini
dovevano assestarsi allo stesso modo, e quell'attesa andava rifatta da capo
per ogni singolo dato dell'archivio.

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
notazione dei fisici per l’$\mathbb{E}[\cdot]$ del resto del libro, e qui si
tengono perché è così che la formula si trova in letteratura), il primo
termine è la correlazione media tra i neuroni $i$ e $j$ con i
visibili bloccati sui dati (fase positiva, la «veglia») e il secondo la
stessa correlazione con la rete libera di campionare da sé (fase negativa,
il «sogno»). Il «$\propto$» nasconde un $1/T$: il tasso di apprendimento
effettivo dipende dalla temperatura a cui si raccolgono le statistiche. La
derivazione è quella della sezione sulla partizione, applicata due volte:
$\partial E/\partial w_{ij} = -s_i s_j$, e poiché i dati vincolano solo i
visibili bisogna passare alla marginale, cioè sommare la congiunta su tutte le
configurazioni dei nascosti: compare così una **seconda** media, quella sui
nascosti dati i visibili. Da lì i due termini.

Nella macchina di Boltzmann originale, a connettività generale, il problema
non è solo il secondo termine: lo sono tutti e due. Con i visibili
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

Si bara in due modi, e il primo è quello appena detto. Il sogno che ne esce è
appena abbozzato, costa un attimo invece di un'eternità, e basta lo stesso. Il
difetto è prevedibile: partendo sempre da cose vere, la macchina non va mai a
curiosare nelle regioni in cui si sbaglia di grosso, e quelle regioni restano
sbagliate perché nessuno ci va ad alzare il terreno.

Il secondo modo rimedia proprio a questo, e non costa niente di più: non far
ricominciare il sogno da capo ogni volta, ma lasciar continuare quello di
prima. Un po’ per volta il sogno si allontana e finisce anche nei posti dove
la macchina si illude. Si chiama **contrastive divergence persistente**, dove
«persistente» è il sogno che non viene mai interrotto.

C'è un prezzo, ed è meglio saperlo che credere di aver trovato una scorciatoia
gratis. Di solito, quando una macchina impara, c'è un numero che dice quanto
sta sbagliando (non è l'energia, che è il voto dato a una singola risposta: è
un voto dato all'intera macchina, e si guarda una volta ogni tanto), e
imparare vuol dire farlo scendere: se scende si è sulla strada giusta, e
quando smette di scendere si è arrivati. Con il sogno abbreviato quel numero
non c'è, e non perché sia difficile da calcolare o perché nessuno l'abbia
ancora trovato: un numero del genere qui non esiste, e i ritocchi che la
macchina fa non stanno scendendo lungo niente. Nessuno può garantire che stia
andando verso qualcosa invece che in tondo. In pratica, sulle reti di allora,
funzionava benissimo.

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
soltanto fra le caselle dei dati e i taccuini interni, e nessuno più fra un
taccuino e l'altro. Si chiama **macchina di Boltzmann ristretta**, e tutti la
chiamano con la sigla inglese, **RBM**.

È quella potatura a far cadere il primo dei due costi, la veglia. Il motivo si
dice in una riga: se i taccuini non sono collegati fra loro, con i dati veri
davanti agli occhi ogni taccuino dipende soltanto dai dati, e nessuno deve
aspettare la decisione del vicino per prendere la sua. Non c'è niente da
assestare: si calcola tutto in un colpo solo. Il secondo costo, il sogno, è
quello che la contrastive divergence di poco fa ha già accorciato. Insieme, le
due mosse rendono praticabile ciò che nel 1985 non lo era.

Fu proprio la coppia RBM più contrastive divergence, con più reti impilate una
sopra l'altra a formare gli strati di una rete profonda
{cite}`hinton2006fast`, a rimettere in moto il deep learning a metà anni
Duemila, quando addestrare reti profonde sembrava impossibile. Si sgrossava la
rete uno strato alla volta prima di addestrarla per intero, ed è il
*pre-training* di cui si parlava allora. È un ruolo
storico che va riconosciuto con onestà, insieme al suo epilogo: di lì a pochi
anni le ReLU, le GPU e archivi di dati più grandi avrebbero reso superfluo
quel modo di partire, e oggi le RBM non si usano quasi più.

Il modo di ragionare con cui erano state costruite, invece, è vivo e vegeto:
nella prossima sezione si vede perché, e quanto costi davvero misurare il
paesaggio intero, cioè il gesto che trasforma un'altezza in una percentuale.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- La **macchina di Boltzmann** aggiunge alla rete di Hopfield due cose: la
  possibilità di risalire ogni tanto (si scuote il paesaggio, e quella scossa
  si chiama **temperatura**) e qualche neurone in più che non corrisponde a
  nessun pixel, buono per annotarsi le regolarità del dato.
- Si scuote forte all'inizio e sempre più piano: così la pallina esce dalle
  conche mediocri finché può, e si assesta in una valle profonda quando la
  calma torna. È la *ricottura simulata* nominata in apertura di capitolo.
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
  Funziona, ma è una scorciatoia, non una soluzione: il numero che dice quanto
  la macchina sta sbagliando, qui, non esiste, e i suoi ritocchi non stanno
  scendendo lungo niente.
- Da qui in avanti l'altezza del paesaggio diventa una percentuale, e per
  trasformarla bisognerebbe aver misurato il paesaggio intero, valle per
  valle. È il conto che l'apertura del capitolo chiamava **funzione di
  partizione**, ed è il personaggio a cui è intitolata la prossima sezione.
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
