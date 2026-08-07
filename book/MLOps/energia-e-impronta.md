# Il conto in energia

Di un modello si dichiara quasi tutto: quanti parametri ha, quanti FLOP costa
una passata, quanto è accurato, quanti millisecondi impiega a rispondere. Una
cosa non si dichiara quasi mai, ed è quanta elettricità consuma.

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
coincidono: il tiling, la fusione dei kernel, FlashAttention e gli
acceleratori che fanno scorrere i dati fra unità vicine sono tutte tecniche per
ridurre i viaggi, e ridurre i viaggi vuol dire insieme andare più veloci e
consumare meno.

`````

`````{tab} Superiore

Le misure di riferimento sono quelle di Horowitz {cite}`horowitz2014computing`:
in un nodo tecnologico a 45 nm, una moltiplicazione-accumulo in virgola mobile
costa qualche picojoule, mentre **leggere un dato a 32 bit dalla DRAM ne costa
circa 640**, cioè più di due ordini di grandezza. Un accesso a memoria
*on-chip* sta nel mezzo, dell'ordine dei $5$–$10$ pJ. I valori assoluti
dipendono dal nodo e dal progetto, ma il **rapporto** è la cosa robusta, e nel
tempo è peggiorato: la densità dei transistor è migliorata più in fretta
dell'energia per bit trasportato.

Ne segue che, per una tipica inferenza densa, la maggior parte del budget
energetico se ne va in movimento di dati e non in aritmetica. Ed è la
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
anche il condizionamento che porta via il calore, i gruppi di continuità, le
perdite negli alimentatori. La misura di quanto pesa questo contorno si chiama
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
**quando** eseguire, e varia di oltre un ordine di grandezza fra reti diverse e
fra ore diverse della stessa rete.

L'analisi di Patterson e colleghi {cite}`patterson2021carbon` mostra che i
fattori non sono comparabili per ampiezza: le scelte di collocazione e di
hardware possono valere, insieme, ordini di grandezza, mentre l'ottimizzazione
del codice raramente supera il fattore due o tre. Il che ribalta l'intuizione
di chi lavora al modello, e va detto con onestà: la leva più grossa non è quasi
mai la nostra.

`````

## Addestrare una volta, servire un miliardo di volte

C'è un errore di prospettiva che quasi tutti fanno all'inizio, e nasce dal
fatto che l'addestramento fa notizia e l'inferenza no.

L'addestramento è un costo **una tantum**, grande e ben visibile: si può
misurare, si può datare, si può scrivere in un paper. L'inferenza è un costo
**minuscolo moltiplicato per un numero enorme**: una singola risposta consuma
pochissimo, ma se il modello risponde a milioni di richieste al giorno per due
anni, l'integrale supera facilmente l'addestramento che l'ha prodotto.

La conseguenza operativa è che **le leve dell'inferenza contano più di quelle
dell'addestramento**, e sono esattamente quelle già viste per motivi di costo e
di latenza: la quantizzazione, la distillazione e la potatura della sezione su
LLMOps, il *batching* che ammortizza la lettura dei pesi su molte richieste, la
cache del prefisso che evita di ricalcolare ciò che è già stato calcolato. Nel
capitolo su LLMOps quelle tecniche erano presentate come modi di spendere meno
e rispondere prima; sono la stessa cosa vista da un'altra finestra.

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

Si distingue fra carbonio **operativo** (quello della sezione precedente,
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

Tirando le somme, le leve sono cinque, in ordine decrescente di quanto in
media spostano: **dove** si esegue (l'intensità di carbonio della rete),
**quando** si esegue per i lavori differibili, **su cosa** si esegue (un
acceleratore adatto contro uno generico), **quanto** modello serve davvero
(un modello dieci volte più piccolo che basta allo scopo batte qualunque
micro-ottimizzazione), e infine **come** è scritto il codice.

Due avvertenze finali, che valgono più di molte buone intenzioni.

La prima è che l'efficienza, da sola, non riduce i consumi totali. Ogni volta
che addestrare è diventato più economico, il risparmio è stato reinvestito in
modelli più grandi anziché incassato: è il paradosso di Jevons, e non è un
argomento contro l'efficienza, è un argomento contro il crederla sufficiente.

La seconda riguarda i numeri. Quasi tutte le cifre pubblicate su questo tema
sono **stime**, ottenute da ipotesi su hardware, utilizzo e mix energetico che
raramente sono dichiarate per intero; confrontarle fra due lavori diversi
significa quasi sempre confrontare due insiemi di ipotesi, non due sistemi. La
misura seria si fa in casa propria, con i contatori dell'hardware che si ha,
esattamente come per la latenza. Il resto è ordine di grandezza, ed è già
molto.

```{admonition} Da ricordare
:class: important
- L'energia non se ne va nei conti ma nel **movimento dei dati**: leggere un
  valore dalla DRAM costa più di due ordini di grandezza rispetto a una
  moltiplicazione-accumulo {cite}`horowitz2014computing`. È la stessa
  affermazione del capitolo sulle GPU, letta con il contatore invece che col
  cronometro, e rende ottimizzazione e sobrietà la stessa cosa.
- La catena completa è
  $\text{gCO}_2\text{e} = E_{\text{IT}} \times \text{PUE} \times I_{\text{rete}}$:
  il **PUE** misura il costo dell'edificio (da $1{,}1$ a oltre $2$, quasi tutto
  raffreddamento), l'**intensità di rete** varia di oltre un ordine di
  grandezza fra luoghi e fra ore.
- **L'inferenza supera l'addestramento** quando il modello è servito a lungo:
  quantizzazione, distillazione, *batching* e cache del prefisso sono leve
  ambientali oltre che economiche.
- Il **carbonio incorporato** (fabbricazione) è trascurabile per un
  acceleratore molto usato e **dominante** per un dispositivo poco usato: là si
  ottimizza il joule, qui la durata.
- Le cifre pubblicate sono stime con ipotesi spesso implicite: si misura in
  casa propria. E l'efficienza da sola non basta, perché storicamente il
  risparmio è stato reinvestito in modelli più grandi.
```
