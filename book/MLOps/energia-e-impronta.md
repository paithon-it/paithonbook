# Il conto in energia

Di un modello si dichiara quasi tutto: quanti parametri ha, quanti conti costa
una passata (in gergo quanti **FLOP**, cioè quante singole operazioni
aritmetiche, una moltiplicazione o una somma), quanto è accurato, quanti
millisecondi impiega a rispondere. Una cosa non si dichiara quasi mai, ed è
quanta elettricità consuma.

La domanda è entrata nel dibattito tecnico nel 2019, quando Strubell, Ganesh e
McCallum hanno provato a mettere un numero sull'addestramento di un modello di
linguaggio {cite}`strubell2019energy`. I loro numeri sono stati poi discussi e
corretti, e va detto subito: **le cifre di questa materia invecchiano in
fretta**, perché dipendono dall'hardware di quell'anno, dal centro dati e
perfino dall'ora del giorno. Quello che non invecchia è la catena che porta da
un'operazione aritmetica a un grammo di anidride carbonica, e sono i suoi
anelli che vale la pena conoscere, perché ognuno è una leva.

## Dal FLOP al joule

Il primo anello è già stato costruito nel capitolo sulle GPU, anche se lì lo
guardavamo con il cronometro invece che col contatore.

`````{tab} Elementare

Intuitivamente, l'energia se ne va nei conti: più moltiplicazioni, più
corrente. È sbagliato, ed è sbagliato di parecchio.

Fare una moltiplicazione dentro il processore costa pochissimo. Andare a
**prendere** i due numeri da moltiplicare nella memoria esterna costa
centinaia di volte tanto. Muovere un dato è caro perché significa far
commutare lunghe piste di rame dentro il silicio, e quella è fisica, non
programmazione.

Il capitolo sulle GPU aveva già detto questa cosa con il cronometro in mano:
il collo di bottiglia non sono i conti, sono i byte. Adesso la stessa frase si
rilegge con la bolletta in mano, e dice la stessa cosa. Le due ottimizzazioni
coincidono, ed è per questo che tutte le tecniche di quel capitolo (lavorare a
piccoli blocchi tenuti vicino al processore, fare più operazioni in un
passaggio solo invece di andare e tornare dalla memoria a ogni passo, far
scorrere il dato fra unità vicine invece che dentro e fuori) sono in fondo la
stessa tecnica: **ridurre i viaggi**. E ridurre i viaggi vuol dire insieme
andare più veloci e consumare meno.

`````

`````{tab} Superiore

Le misure di riferimento vengono da un'unica tabella: l'energia per operazione
in un nodo tecnologico a 45 nm, compilata da Horowitz e resa nota dalla sua
relazione sul problema energetico del calcolo {cite}`horowitz2014computing`.
Vale la pena sapere che quella tabella circola in più versioni, e che la più
citata non è quella della relazione ma la sua ripresa nella letteratura
successiva sulle reti compresse: i singoli valori differiscono un poco (la
moltiplicazione in virgola mobile a 32 bit è data ora $3{,}7$ ora $4$
picojoule, la lettura dalla DRAM ora $640$ picojoule ora qualche nanojoule),
mentre i **rapporti** non differiscono affatto, ed è quello che conta qui. Chi
rifà i conti con l'altra versione trova il pareggio a sessantacinque FLOP per
byte invece che a settanta: la morale non cambia, ma il numero sì, e conviene
sapere da dove viene il proprio.

Una moltiplicazione-accumulo in `float32` mette insieme una moltiplicazione
($3{,}7$ picojoule) e un'addizione ($0{,}9$), quindi costa **poco meno di
cinque picojoule**, mentre **leggere un dato a 32 bit dalla DRAM ne costa circa
640**:
più di due ordini di grandezza. Un accesso alla memoria che sta *dentro* il
chip costa invece quanto l'aritmetica stessa, dell'ordine dei $5$ pJ, e il
salto dei due ordini di grandezza è tutto nell'**uscita dal chip**. Finché il
dato resta nel silicio, toccarlo costa quanto calcolarci sopra; appena esce,
costa cento volte tanto. I valori assoluti dipendono dal nodo e dal progetto,
ma il rapporto è la cosa robusta, e nel tempo è peggiorato: la densità dei
transistor è migliorata più in fretta dell'energia per bit trasportato.

Da un rapporto fra costi unitari, però, non segue ancora niente sul budget
totale. Per sapere dove finisce l'energia serve sapere **quante operazioni si
fanno per ogni byte letto**, cioè l'intensità aritmetica del modello roofline,
la stessa grandezza con cui la sezione sulle metriche di servizio ha distinto
prefill e decode. Il pareggio cade **attorno ai settanta FLOP/byte**, e il
conto è breve: 640 picojoule ogni quattro byte fanno 160 pJ per byte, mentre
una moltiplicazione-accumulo da $4{,}6$ pJ vale due operazioni, cioè $2{,}3$ pJ
per FLOP.

Al di sotto di quella soglia l'energia se ne va quasi tutta in movimento di
dati, ed è il caso della generazione token per token, dove l'intensità è
dell'ordine dell'unità e la quota spesa in aritmetica è di qualche punto
percentuale. Al di sopra domina invece l'aritmetica: la lettura di un prompt
lungo, o una passata di addestramento, stanno dall'altra parte del
ginocchio. La leva del movimento
dei dati è dunque enorme dove il carico è memory-bound, che è quasi tutta
l'inferenza interattiva, e modesta dove non lo è. È la
giustificazione economica di tutta l'ingegneria del capitolo sulle GPU: il
riuso in shared memory, la fusione dei kernel, la precisione ridotta (che
dimezza i byte da muovere prima ancora di dimezzare i conti) e l'array
sistolico, la cui intera ragione d'essere è far attraversare un dato letto una
sola volta a decine di unità di calcolo.

Da qui, la prima stima grossolana ma utile: l'energia di un carico di lavoro si
approssima come potenza media dell'acceleratore per tempo di esecuzione. È
grossolana perché la potenza dipende da *cosa* si sta calcolando, ma ha il
pregio di essere misurabile con strumenti che esistono già
(`nvidia-smi` espone la potenza istantanea, i contatori RAPL fanno lo stesso
per la CPU).

`````

## Dal joule al grammo

Il secondo anello esce dal silicio ed entra nell'edificio.

`````{tab} Elementare

Un centro dati non consuma solo quello che consumano i calcolatori. Consuma
anche il condizionamento che porta via il calore, i gruppi di continuità (le
batterie che tengono tutto acceso quando manca la corrente, e che per restare
cariche consumano di continuo), le perdite negli alimentatori. La misura di
quanto pesa questo contorno si chiama
**PUE**: è il rapporto fra l'energia che entra nell'edificio e quella che
arriva davvero ai calcolatori. Un PUE di $1$ sarebbe la perfezione (impossibile);
una struttura moderna sta attorno a $1{,}1$–$1{,}3$, cioè spende il dieci o il
trenta per cento in più; una vecchia può superare il $2$, cioè per ogni watt di
calcolo ne brucia un altro per raffreddarlo.

Poi c'è il terzo passaggio, ed è quello che sfugge di più: **la stessa
elettricità non inquina uguale dappertutto**. Un kilowattora prodotto dove la
rete è idroelettrica o nucleare porta con sé pochi grammi di anidride
carbonica; lo stesso kilowattora dove si brucia carbone ne porta un ordine di
grandezza in più. E non cambia solo da paese a paese: cambia da un'ora
all'altra, perché di notte, o quando non c'è vento, la rete accende centrali
diverse.

La conseguenza è concreta e un po' sorprendente: **lo stesso addestramento,
identico, cambia impronta a seconda di dove e quando lo si fa girare**. Da qui
l'idea di spostare i lavori che possono aspettare verso le ore e i luoghi in
cui la rete è pulita.

`````

`````{tab} Superiore

La catena completa si scrive in una riga:

$$
\text{gCO}_2\text{e} \;=\; E_{\text{IT}} \;\times\; \text{PUE} \;\times\; I_{\text{rete}},
$$

dove $E_{\text{IT}}$ è l'energia consumata dai calcolatori (in kWh), il **PUE**
(*Power Usage Effectiveness*) è il rapporto fra energia totale della struttura
ed energia dei calcolatori, e $I_{\text{rete}}$ è l'**intensità di carbonio**
della rete elettrica in grammi di CO₂ equivalente per kWh.

I tre fattori si governano con leve diverse e da attori diversi. $E_{\text{IT}}$
è la leva di chi scrive il modello e il codice; il PUE è la leva di chi
progetta il centro dati (nelle strutture efficienti sta fra $1{,}1$ e $1{,}3$,
in quelle mal progettate supera $2$, e la differenza è quasi tutta
raffreddamento); $I_{\text{rete}}$ è la leva di chi sceglie **dove** e
**quando** eseguire, e varia di oltre un ordine di grandezza fra reti diverse,
e di alcune volte fra ore diverse della stessa rete.

Una precisazione sul PUE, perché la formula insegna a fare un conto ed è così
che il conto sbaglia. Il PUE è un rapporto **di struttura, annualizzato**:
riguarda tutto l'edificio su tutto l'anno. Moltiplicarlo per l'$E_{\text{IT}}$
di *un* singolo carico di lavoro assume che il contorno cresca in proporzione
al carico, mentre una quota rilevante (illuminazione, gruppi di continuità a
vuoto, ventilazione di base) è fissa: il PUE **marginale** di un lavoro
aggiuntivo è di norma più basso di quello medio, e la formula sovrastima. Nella
direzione opposta, il PUE non copre né le perdite di trasmissione della rete
elettrica né il consumo d'acqua. Va bene per l'ordine di grandezza, non per
confrontare due lavori sulla stessa macchina.

L'analisi di Patterson e colleghi {cite}`patterson2021carbon` mette in fila le
ampiezze delle quattro leve, e non sono affatto uguali fra loro. Sui modelli
che avevano sottomano, nel 2021, pesava di più la scelta del **modello**: una
rete grande ma ad attivazione **sparsa** (una in cui, per rispondere, si accende
ogni volta solo una piccola parte della rete, invece che tutta come in una
rete *densa*) poteva consumare meno di un decimo di una densa a parità di
qualità. Le stava vicina la **collocazione geografica**, cioè in quale rete
elettrica si esegue il lavoro, che sposta le emissioni di un fattore fra cinque
e dieci, anche restando dentro lo stesso paese e la stessa organizzazione. Più
sotto le altre due, che il paper tiene distinte: l'**hardware** specializzato
per il machine learning rende da due a cinque volte più di un sistema generico,
e un centro dati progettato bene è da 1,4 a 2 volte più efficiente di uno
tipico (è il PUE di poco fa).

Quei quattro numeri sono misure su architetture di quell'anno, e come tutte le
misure hanno una scadenza; quello che regge è la loro **morale**, ed è già
abbastanza forte. Le leve di progetto (quanto modello serve, e dove lo si
esegue) contano più di quelle di implementazione, e nessuna delle due sta dove
di solito si cerca: non nella micro-ottimizzazione del codice, che sposta molto
meno, e non tutte nelle mani del team che costruisce il modello, visto che la
scelta del luogo è di qualcun altro.

`````

## Addestrare una volta, servire un miliardo di volte

C'è un errore di prospettiva che quasi tutti fanno all'inizio, e nasce dal
fatto che l'addestramento fa notizia e l'inferenza no.

L'addestramento è un costo **una tantum**, grande e ben visibile: si può
misurare, si può datare, si può scrivere in un paper. L'inferenza è un costo
**minuscolo moltiplicato per un numero enorme**: una singola risposta consuma
pochissimo, ma se il modello risponde a milioni di richieste al giorno per due
anni, il totale supera facilmente l'addestramento che l'ha prodotto.

La conseguenza operativa è che **le leve dell'inferenza contano più di quelle
dell'addestramento**, e sono esattamente quelle già viste per motivi di costo e
di latenza: la quantizzazione e la potatura della sezione su LLMOps, il
*batching* che ammortizza la lettura dei pesi su molte richieste, la cache del
prefisso della sezione sulle metriche di servizio, che evita di ricalcolare ciò
che è già stato calcolato. Ci si aggiunge la **distillazione**, cioè
addestrare un modello piccolo a imitare le risposte di uno grande e poi servire
il piccolo: la incontra il capitolo sui Transformer, nella sezione *Tendenze e
limiti*, e sul conto dell'energia è la leva più radicale di tutte, perché non
alleggerisce il modello, lo sostituisce. Nella sezione su LLMOps quelle
tecniche erano presentate come modi di spendere meno e rispondere prima; sono
la stessa cosa vista da un'altra finestra.

Il punto di pareggio non è un numero universale: dipende da quanto è grande il
modello, da quante richieste riceve e da quanto a lungo resta in servizio. Ma
l'ordine delle priorità che ne discende è stabile, e vale la pena tenerlo in
mente quando si sceglie fra un modello grande e uno piccolo rifinito bene.

## Il carbonio che c'è già dentro

Resta un pezzo che non compare in nessuna bolletta, e che è facile dimenticare
proprio perché è già stato pagato.

`````{tab} Elementare

Un chip, prima di consumare il suo primo watt, è già costato energia: quella
per estrarre e purificare il silicio, per far funzionare una fabbrica che è
fra gli impianti industriali più energivori che esistano, per trasportare il
prodotto. Si chiama **carbonio incorporato**, ed è la parte dell'impronta che
un dispositivo si porta dietro dalla nascita.

Per un acceleratore in un centro dati che macina calcoli ventiquattr'ore al
giorno per cinque anni, il carbonio incorporato è una frazione piccola del
totale: quello che consuma girando è molto di più. Ma per un dispositivo che
sta acceso poco (un sensore, un telefono, una scheda che si sveglia due volte
al giorno) il rapporto si **rovescia**: quasi tutta la sua impronta è stata
fissata prima che qualcuno lo accendesse.

Da cui una conclusione poco intuitiva: per gli oggetti piccoli e poco usati, la
scelta ambientale più efficace non è renderli più efficienti, è **tenerli in
servizio più a lungo**.

`````

`````{tab} Superiore

Si distingue fra carbonio **operativo** (quello del conto di poco fa,
$E \times \text{PUE} \times I$) e carbonio **incorporato**, cioè le emissioni
di fabbricazione, trasporto e smaltimento, che si ammortizzano sulla vita utile
del dispositivo. Il conto complessivo è

$$
C_{\text{totale}} = C_{\text{operativo}}(t) + C_{\text{incorporato}} \cdot
\frac{t}{T_{\text{vita}}},
$$

e il rapporto fra i due termini dipende interamente dal **fattore di utilizzo**.
Un acceleratore da centro dati con utilizzo alto e vita di qualche anno è
dominato dall'operativo; un dispositivo *edge* con utilizzo dell'ordine
dell'uno per cento è dominato dall'incorporato.

La conseguenza progettuale è che le due categorie richiedono ottimizzazioni
opposte. Nel centro dati si ottimizza il joule per inferenza, e sostituire
l'hardware con una generazione più efficiente conviene anche
ambientalmente. Sull'*edge* si ottimizza la **longevità**: un modello che
continua a funzionare su hardware vecchio evita un ricambio, e quel ricambio
pesa più di anni di funzionamento.

`````

## Che cosa si può fare, e che cosa non funziona

Tirando le somme, le leve sono cinque, e le prime due contano più delle altre:
**quanto** modello serve davvero (un modello che si accende solo in parte, o
dieci volte più piccolo, purché basti allo scopo, batte qualunque
micro-ottimizzazione) e **dove** si esegue, cioè l'intensità di carbonio della
rete elettrica. Vengono poi, nell'ordine, **su cosa** si esegue (un
acceleratore adatto contro uno generico), **quando** si esegue per i lavori
differibili, e infine **come** è scritto il codice.

Due avvertenze finali, che valgono più di molte buone intenzioni.

La prima è che l'efficienza, da sola, non basta a ridurre i consumi totali.
Nella corsa alla scala degli ultimi anni, ogni volta che addestrare è diventato
più economico il risparmio è stato in buona parte reinvestito in modelli più
grandi anziché incassato: è l'effetto di rimbalzo che va sotto il nome di
paradosso di Jevons. Non è però una legge, ed è onesto dire che la questione è
aperta. C'è almeno un caso, e non piccolo, in cui l'efficienza la crescita l'ha
assorbita davvero: fra il 2010 e il 2018 il consumo elettrico complessivo dei
centri dati del mondo è cresciuto di circa il sei per cento, mentre il lavoro
che ci girava dentro si moltiplicava per più di sei. Otto anni, non un secolo, e
prima dell'ondata che questo libro racconta: la misura serve a dire che il
rimbalzo è un effetto documentato e frequente, non un destino. L'argomento non
è contro l'efficienza, è contro il crederla sufficiente da sola.

La seconda riguarda i numeri. Quasi tutte le cifre pubblicate su questo tema
sono **stime**, ottenute da ipotesi su hardware, utilizzo e mix energetico che
raramente sono dichiarate per intero; confrontarle fra due lavori diversi
significa quasi sempre confrontare due insiemi di ipotesi, non due sistemi. La
misura seria si fa in casa propria, con i contatori dell'hardware che si ha,
esattamente come per la latenza. Il resto è ordine di grandezza, ed è già
molto.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- L'energia non se ne va nei conti ma nel **movimento dei dati**: fare una
  moltiplicazione costa pochissimo, andare a prendere i due numeri nella
  memoria esterna costa centinaia di volte tanto. È la stessa frase del
  capitolo sulle GPU (il collo di bottiglia non sono i conti, sono i byte)
  riletta con la bolletta in mano: andare più veloci e consumare meno sono la
  stessa cosa.
- L'impronta finale è il prodotto di **tre fattori**: l'energia che consumano i
  calcolatori, il sovrapprezzo dell'edificio (raffreddamento e perdite, dal
  dieci per cento fino a più del doppio) e quanto sporca è l'elettricità di
  quella rete, che cambia di dieci volte fra un luogo e l'altro e di alcune
  volte fra un'ora e l'altra della stessa rete.
- **Rispondere costa più che addestrare**, quando il modello resta in servizio
  a lungo: rimpicciolirlo, servire più richieste in una volta sola e riusare
  ciò che è già stato calcolato sono leve ambientali oltre che economiche.
- Un chip ha già un'impronta **prima di essere acceso** (fabbricazione):
  trascurabile per un acceleratore che macina calcoli sempre, dominante per un
  oggetto che si accende di rado. Là conviene consumare meno, qui durare di
  più.
- Le cifre pubblicate sono quasi tutte stime, con ipotesi che raramente sono
  dichiarate: si misura in casa propria. E l'efficienza da sola non basta a far
  scendere i consumi, perché il risparmio tende a essere reinvestito in modelli
  più grandi invece che incassato. Tende, non deve: è successo anche il
  contrario.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- L'energia non se ne va nei conti ma nel **movimento dei dati**, a una
  condizione: che l'**intensità aritmetica** sia bassa. Leggere dalla DRAM costa
  più di due ordini di grandezza rispetto a una moltiplicazione-accumulo
  {cite}`horowitz2014computing` (mentre un accesso *on-chip* costa quanto
  l'aritmetica: il salto è nell'uscita dal chip), e il pareggio cade attorno a
  $70$ FLOP/byte: il decode sta molto sotto, il prefill e l'addestramento
  stanno sopra. È la stessa affermazione del capitolo sulle GPU, letta con il
  contatore invece che col cronometro.
- La catena completa è
  $\text{gCO}_2\text{e} = E_{\text{IT}} \times \text{PUE} \times I_{\text{rete}}$:
  il **PUE** misura il costo dell'edificio (da $1{,}1$ a oltre $2$, quasi tutto
  raffreddamento), l'**intensità di rete** varia di oltre un ordine di
  grandezza fra luoghi, e di alcune volte fra le ore della stessa rete. Il PUE
  però è una media annuale di struttura: applicato a un singolo carico
  sovrastima, perché una parte del contorno è fissa.
- **L'inferenza supera l'addestramento** quando il modello è servito a lungo:
  quantizzazione e potatura (sezione su LLMOps), *batching*, cache del prefisso
  (sezione sulle metriche di servizio) e distillazione sono leve ambientali
  oltre che economiche.
- Il **carbonio incorporato** (fabbricazione) è trascurabile per un
  acceleratore molto usato e **dominante** per un dispositivo poco usato: là si
  ottimizza il joule, qui la durata.
- Le cifre pubblicate sono stime con ipotesi spesso implicite: si misura in
  casa propria. E l'efficienza da sola non basta, perché il risparmio tende a
  essere reinvestito in scala (l'effetto di rimbalzo); ma non sempre, e fra il
  2010 e il 2018 i centri dati sono cresciuti del sei per cento a fronte di un
  carico moltiplicato per più di sei.
```
`````
