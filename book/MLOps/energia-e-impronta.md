# Il conto in energia

Di un modello si dichiara quasi tutto: quanti parametri ha, quanti conti costa
una passata (in gergo quanti **FLOP**, cioè quante singole operazioni
aritmetiche, una moltiplicazione o una somma), quanto è accurato, quanti
millisecondi impiega a rispondere. Una cosa non si dichiara quasi mai, ed è
quanta elettricità consuma.

La domanda è entrata nel dibattito tecnico nel 2019, quando Strubell, Ganesh e
McCallum hanno provato a mettere un numero sull'addestramento di un modello di
linguaggio {cite}`strubell2019energy`. I loro numeri sono stati poi discussi e
corretti, e **le cifre di questa materia invecchiano in fretta**, perché
dipendono dall'hardware di quell'anno, dal centro dati e perfino dall'ora del
giorno. Quello che non invecchia è la catena che porta da
un'operazione aritmetica a un grammo di anidride carbonica, e sono i suoi
anelli da conoscere, perché ognuno è una leva.

## Dal FLOP al joule

Il **joule** è l'unità con cui si misura l'energia, come il metro lo è per le
distanze: la corrente che si compra a kilowattora sono joule, e anche quello
che un chip brucia per fare un conto sono joule, solo molti di meno. Il primo
anello della catena porta dai conti ai joule, ed è già stato costruito
parlando della {doc}`memoria di una GPU </GPU/gerarchia-memoria>`, anche se lì
lo guardavamo con il cronometro invece che col contatore.

`````{tab} Elementare

Ci si aspetta che l'energia se ne vada nei conti: più moltiplicazioni, più
corrente. Il contatore, quasi sempre, dice un'altra cosa.

I due numeri li ha messi in fila un ingegnere di Stanford, Mark Horowitz, e
sono facili da tenere a mente. Fare un conto dentro il processore (una
moltiplicazione e la somma che la segue) costa **poco meno di cinque**. Andare
a prendere un numero nella memoria che sta fuori dal chip costa **circa 640**,
più di cento volte tanto. L'unità non conta (è il picojoule, troppo piccolo
perché immaginarlo abbia senso): conta il rapporto fra i due.

Il motivo è fisico, non informatico. Portare un numero da fuori a dentro il
chip vuol dire far cambiare stato a lunghissime piste di rame, e ogni
cambiamento di stato costa corrente. Un numero che sta già dentro costa quanto
farci sopra un conto: di rame ne muove pochissimo.

Dai due prezzi, però, non segue ancora chi si prende la bolletta. Il chip è un
tavolo di lavoro e la memoria di fuori l'armadio in fondo alla stanza, come nel
capitolo sulle GPU: chi attraversa la stanza per copiare un numero solo passa
la giornata in piedi, chi torna con un foglio da cui ricava trecento conti non
si accorge nemmeno del tragitto. Dividendo i due prezzi si trova la soglia, ed
è attorno ai **centoquaranta conti** per ogni numero preso da fuori: sotto, la
corrente se ne va nei viaggi; sopra, se ne va nei conti.

Generare una parola alla volta sta molto sotto la soglia: il calcolatore
rilegge tutti i pesi del modello per una parola sola. Lì tagliare i viaggi vale
tutto, ed è il mestiere delle tecniche di quel capitolo: tenere i dati vicino
al processore, fare più cose in un passaggio solo invece di andare e tornare.
Essere limitati dai byte invece che dai conti là valeva per il tempo e qui vale
per la corrente: andare più veloci e consumare meno sono la stessa cosa.
L'addestramento e la lettura di un prompt lungo stanno sopra la soglia, e lì
limare i viaggi sposta poco.

`````

`````{tab} Superiore

Le misure di riferimento vengono da un'unica tabella: l'energia per operazione
in un nodo tecnologico a 45 nm, compilata da Horowitz e resa nota dalla sua
relazione sul problema energetico del calcolo {cite}`horowitz2014computing`.
Quella tabella circola in più versioni, e la più citata non è quella della
relazione ma la sua ripresa nella letteratura successiva sulle reti compresse:
i singoli valori differiscono un poco (la moltiplicazione in virgola mobile a
32 bit è data ora $3{,}7$ ora $4$ picojoule, la lettura dalla DRAM ora $640$
picojoule ora qualche nanojoule),
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
salto dei due ordini di grandezza è tutto nell’**uscita dal chip**. Finché il
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
dell'ordine dell'unità e la quota spesa in aritmetica è poco più di un punto
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

Entra corrente in un centro dati, e nei calcolatori non finisce tutta: ne
prendono il condizionamento, le batterie che tengono acceso quando la corrente
salta, gli alimentatori. Quel contorno ha un nome, **PUE**, ed è un rapporto:
la bolletta dell'edificio diviso la corrente arrivata davvero alle macchine.
Una struttura moderna sta fra 1,1 e 1,3; una vecchia supera il 2, e per ogni
watt di calcolo ne brucia un altro per raffreddarlo.

Il rapporto, però, è la media di tutto l'edificio su tutto l'anno. Spegni il
tuo addestramento per una notte: la bolletta cala di quello che consumavano le
tue macchine e di poco altro, perché le luci, le batterie e la ventilazione
restavano accese comunque. Chi si addebita anche il venti per cento di contorno
si fa il conto più caro del vero: il rapporto serve per l'ordine di grandezza,
non per confrontare due lavori sulla stessa macchina.

La stessa elettricità, poi, non inquina uguale dappertutto. Un kilowattora
prodotto dove la rete è idroelettrica o nucleare porta con sé qualche decina di
grammi di anidride carbonica; dove si brucia carbone, qualche centinaio. E
cambia da un'ora all'altra, perché di notte, o senza vento, la rete accende
centrali diverse.

I tre pezzi (la corrente delle macchine, il contorno dell'edificio, quanto
sporca è la rete) si moltiplicano fra loro, non si sommano: dimezzare il
consumo o spostare il lavoro su una rete che sporca la metà fanno lo stesso
effetto sul conto finale.

E non pesano uguale. Un gruppo di ricerca di Google, guidato da David
Patterson, ha messo un numero accanto a quattro decisioni, contando la corrente
che serve ad **addestrare** un modello grande. Conta di più **quale modello**:
fra uno che per rispondere si accende tutto e uno a scomparti, che ne sveglia
due o tre e lascia spenti gli altri, a parità di qualità il primo
può consumare dieci volte tanto. Le sta vicino **dove** si esegue, cioè su
quale rete, che sposta le emissioni da cinque a dieci volte anche restando
nello stesso paese. Più sotto **su che macchina** (una fatta apposta fa da due
a cinque volte i conti di una generica con la stessa corrente) e **in che
edificio** (da 1,4 a 2 volte, ed è il raffreddamento di poco fa).

Sono misure del 2021 sull'addestramento, e scadono come tutte le misure.
L'ordine però regge anche dal lato del rispondere, perché in gioco ci sono le
stesse grandezze: quanto modello si accende e con quale corrente contano più
di qualunque limatura del programma.

`````

`````{tab} Superiore

La catena completa si scrive in una riga:

$$
\text{gCO}_2\text{e} \;=\; E_{\text{IT}} \;\times\; \text{PUE} \;\times\; I_{\text{rete}},
$$

dove $E_{\text{IT}}$ è l'energia consumata dai calcolatori (in kWh), il **PUE**
(*Power Usage Effectiveness*) è il rapporto fra energia totale della struttura
ed energia dei calcolatori, e $I_{\text{rete}}$ è l’**intensità di carbonio**
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
riguarda tutto l'edificio su tutto l'anno. Moltiplicarlo per l’$E_{\text{IT}}$
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
sotto le altre due, che il paper tiene distinte: l’**hardware** specializzato
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
fatto che addestrare fa notizia e rispondere no.

Addestrare è un costo che si paga **una volta sola**: grande, ben visibile, si
può misurare, si può datare, si può scrivere in un articolo scientifico.
Rispondere è un costo **minuscolo moltiplicato per un numero enorme**: una
singola risposta consuma pochissimo, ma se il modello risponde a milioni di
richieste al giorno per due anni, il totale supera facilmente l'addestramento
che l'ha prodotto. C'è quindi un momento, nella vita di un modello, in cui la
somma di tutte le risposte date fin lì raggiunge il costo di averlo costruito:
è il **punto di pareggio**. Dove cada non è un numero universale e non lo si
può scrivere qui: dipende da quanto è grande il modello, da quante richieste
riceve e da quanto a lungo resta acceso, e ciascuno se lo deve calcolare per
il proprio caso. Quello che è stabile è l'ordine di priorità che ne discende,
e conviene tenerlo in mente quando si sceglie fra un modello grande e uno
piccolo rifinito bene.

Ne segue che **le leve che contano sono quelle del rispondere, non quelle
dell'addestrare**. E la buona notizia è che sono le
stesse leve già viste per risparmiare denaro e tempo, cioè
alleggerire il modello (la quantizzazione e la potatura della sezione su
LLMOps), servire molte richieste in una volta sola, e non ricalcolare ciò che è
già stato calcolato (il riuso del prefisso della sezione sulle metriche di
servizio). Là erano modi di spendere meno e rispondere prima; sono la stessa
cosa vista da un'altra finestra.

Se ne aggiunge una, ed è la più radicale, perché non alleggerisce il modello:
lo sostituisce. Si chiama **distillazione** e consiste nell'addestrare un
modello piccolo a imitare le risposte di uno grande, per poi mandare in
servizio soltanto il piccolo. La incontra
{doc}`Tendenze e limiti </Transformers/tendenzefuture>`, nel capitolo sui
Transformer.

## Il carbonio che c'è già dentro

Resta un pezzo che non compare in nessuna bolletta, e che è facile dimenticare
proprio perché è già stato pagato.

`````{tab} Elementare

Un chip, prima di consumare il suo primo watt, è già costato energia: quella
per estrarre e purificare il silicio, per far funzionare una fabbrica che è
fra gli impianti industriali più energivori che esistano, per trasportare il
prodotto. Si chiama **carbonio incorporato**, ed è la parte dell'impronta che
un dispositivo si porta dietro dalla nascita.

Quanto pesi dipende da due cose: **per quale frazione della sua vita quel chip
lavora davvero**, e per quanti anni quella vita dura. Una scheda da centro dati
(un *acceleratore*, cioè un chip costruito apposta per fare i conti del machine
learning e nient'altro) macina calcoli ventiquattr'ore al giorno per cinque
anni: consumando così tanto e così a lungo, quello che ha speso per nascere
diventa una briciola del totale.

Un oggetto che si accende di rado sta all'estremo opposto. Un sensore che si
sveglia due volte al giorno lavora per una frazione minuscola del tempo in cui
esiste, e quindi consuma pochissimo: la parte grossa della sua impronta è stata
fissata in fabbrica, prima che qualcuno lo accendesse, e non c'è modo di
recuperarla.

Da cui due strade opposte. Nel centro dati la scheda vecchia conviene
cambiarla appena ne esce una che fa gli stessi conti con meno corrente: quasi
tutto il suo conto è la corrente che berrà da domani, e fabbricare quella nuova
si ripaga in fretta. Con il sensore va al rovescio: sostituirlo vuol dire
pagare da capo la fabbrica per risparmiare briciole, e la scelta ambientale che
conta diventa **tenerlo in servizio più a lungo**, perché un programma che
continua a girare sul dispositivo vecchio è un dispositivo nuovo che non si
costruisce.

`````

`````{tab} Superiore

Si distingue fra carbonio **operativo** (quello del conto di poco fa,
$E_{\text{IT}} \times \text{PUE} \times I_{\text{rete}}$) e carbonio
**incorporato**, cioè le emissioni di fabbricazione, trasporto e smaltimento,
che si ammortizzano sulla vita utile del dispositivo. Il conto complessivo è

$$
C_{\text{totale}} = C_{\text{operativo}}(t) + C_{\text{incorporato}} \cdot
\frac{t}{T_{\text{vita}}},
$$

dove $t$ è il tempo trascorso in servizio e $T_{\text{vita}}$ la vita utile
attesa del dispositivo, cioè su quanto tempo il carbonio di fabbricazione va
spalmato. Il rapporto fra i due termini, a utilizzo costante, non dipende da
$t$, che si semplifica: si gioca sul **fattore di utilizzo** e su quanto è
lunga quella vita utile.
Un acceleratore da centro dati con utilizzo alto e vita di qualche anno è
dominato dall'operativo; un dispositivo *edge* con utilizzo dell'ordine
dell'uno per cento è dominato dall'incorporato.

La conseguenza progettuale è che le due categorie richiedono ottimizzazioni
opposte. Nel centro dati si ottimizza il joule per inferenza, e sostituire
l'hardware con una generazione più efficiente conviene anche
ambientalmente. Sull’*edge* si ottimizza la **longevità**: un modello che
continua a funzionare su hardware vecchio evita un ricambio, e quel ricambio
pesa più di anni di funzionamento.

`````

## Che cosa si può fare, e che cosa non funziona

Tirando le somme, le leve sono cinque. Le prime tre sono le prime tre della
classifica di poco fa, nello stesso ordine: **quanto** modello serve davvero
(uno a scomparti, o uno dieci volte più piccolo purché basti allo scopo, batte
qualunque limatura del programma), **dove** si esegue, cioè quanto è pulita
l'elettricità di quella rete, e **su cosa** si esegue, cioè una macchina fatta
apposta contro una generica. Le altre due la classifica non le misurava:
**quando** si esegue, spostando ciò che può aspettare nelle ore in cui la rete
è pulita, e infine **come** è scritto il codice, che è la leva che conta meno di
tutte. Fuori dall'elenco resta la quarta voce della classifica, la qualità
dell'edificio, e non per distrazione: quella non la sceglie chi costruisce il
modello, la sceglie chi costruisce il centro dati.

Due avvertenze finali, che valgono più di molte buone intenzioni.

La prima è che l'efficienza, da sola, non basta a ridurre i consumi totali.
Nella corsa alla scala degli ultimi anni, ogni volta che addestrare è diventato
più economico il risparmio è stato in buona parte reinvestito in modelli più
grandi anziché incassato: è l'effetto di rimbalzo che va sotto il nome di
paradosso di Jevons. Non è però una legge, ed è onesto dire che la questione è
aperta. C'è almeno un caso, e non piccolo, in cui l'efficienza la crescita l'ha
assorbita davvero. Lo ha misurato un gruppo guidato da Eric Masanet, su
*Science* nel 2020, ricontando quanta elettricità consumano i centri dati del
mondo: fra il 2010 e il 2018 quel consumo è cresciuto di **circa il sei per
cento**, mentre nello stesso periodo il lavoro che ci girava dentro si è
**moltiplicato per più di sei**. Attenzione a non confondere le due cifre: la
prima è un pochino in più, la seconda è sei volte tanto. In otto anni il mondo
ha chiesto ai centri dati sei volte il lavoro, e loro hanno consumato quasi
uguale: lì l'efficienza la crescita se l'è mangiata tutta.

È un caso solo, e per giunta precedente all'ondata dei grandi modelli,
quindi non dimostra niente sul futuro. Serve a dire che il rimbalzo è un
effetto frequente e **non una legge**.
L'argomento, insomma, colpisce il crederla sufficiente da sola, non
l'efficienza.

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
- Prendere un numero dalla memoria esterna costa più di cento volte una
  moltiplicazione, ma da quel prezzo non segue ancora dove finisca la bolletta:
  dipende da **quanti conti si fanno per ogni numero preso**, e la soglia sta
  attorno ai centoquaranta. Generare una parola alla volta ci sta molto sotto, e
  lì la corrente se ne va nei viaggi: è la frase del capitolo sulle GPU a
  proposito del collo di bottiglia, riletta con la bolletta in mano, cioè
  andare più veloci e consumare meno sono la stessa cosa.
  Leggere un prompt lungo e addestrare stanno invece sopra la soglia, e lì
  limare i viaggi sposta poco.
- L'impronta finale è il prodotto di **tre fattori**: l'energia che consumano i
  calcolatori, il sovrapprezzo dell'edificio (raffreddamento e perdite: un
  edificio moderno aggiunge dal dieci al trenta per cento, uno vecchio può
  arrivare a raddoppiare il conto) e quanto sporca è l'elettricità di quella
  rete, che cambia di dieci volte fra un luogo e l'altro e di alcune volte fra
  un'ora e l'altra della stessa rete. Il sovrapprezzo però è la media di tutto
  l'edificio su tutto l'anno: addebitarlo per intero a un singolo lavoro fa il
  conto più caro del vero.
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
  condizione: che l’**intensità aritmetica** sia bassa. Leggere dalla DRAM costa
  più di due ordini di grandezza rispetto a una moltiplicazione-accumulo
  {cite}`horowitz2014computing` (mentre un accesso *on-chip* costa quanto
  l'aritmetica: il salto è nell'uscita dal chip), e il pareggio cade attorno a
  $70$ FLOP/byte: il decode sta molto sotto, il prefill e l'addestramento
  stanno sopra. È la stessa affermazione del capitolo sulle GPU, letta con il
  contatore invece che col cronometro.
- La catena completa è
  $\text{gCO}_2\text{e} = E_{\text{IT}} \times \text{PUE} \times I_{\text{rete}}$:
  il **PUE** misura il costo dell'edificio (da $1{,}1$ a oltre $2$, quasi tutto
  raffreddamento), l’**intensità di rete** varia di oltre un ordine di
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

La sorveglianza di cui parla questo capitolo dice sempre e solo che qualcosa è
cambiato. Che l'errore è salito, che i dati in arrivo non somigliano più a
quelli di prima, che la bolletta è cresciuta. Non dice mai perché, e nemmeno su
che cosa il modello si stia basando per rispondere. È la domanda del
{doc}`capitolo sull'interpretabilità </Interpretabilita/overview>`, che prova
a guardare dentro il modello invece che intorno.
