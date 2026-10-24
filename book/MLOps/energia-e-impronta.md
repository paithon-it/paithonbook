# Il conto in energia

Di un modello si dichiara quasi tutto: il numero di parametri, i **FLOP** di una
passata (le singole operazioni aritmetiche, cioè moltiplicazioni e somme),
l'accuratezza, i millisecondi che impiega a rispondere. Quasi mai si dichiara
quanta elettricità consuma.

Il costo energetico del calcolo è una domanda vecchia quanto i chip; sui
modelli di linguaggio l'hanno posta nel 2019 Strubell, Ganesh e McCallum,
provando a mettere un numero sull'addestramento {cite}`strubell2019energy`.
Alcuni di quei numeri sono stati poi corretti. In particolare, la loro stima
per la ricerca automatica dell'architettura dell'Evolved Transformer
supponeva che ogni
candidato fosse addestrato a grandezza piena, mentre la ricerca vera usava un
compito ridotto; ricalcolate sul compito ridotto, sull'hardware e sul centro
dati effettivi, le emissioni sono risultate 88 volte più basse (un fattore 18,7
per il compito, circa 5 per macchina e centro dati)
{cite}`patterson2021carbon`. Le cifre di questa materia invecchiano in fretta,
perché dipendono dall'hardware di quell'anno, dal centro dati e perfino
dall'ora del giorno. La catena che porta da un'operazione aritmetica a un
grammo di anidride carbonica non invecchia, e ogni anello è una leva.

## Dal FLOP al joule

L'energia si misura in **joule** (J). Il kilowattora della bolletta vale 3,6
milioni di joule; un'operazione aritmetica su un chip ne richiede qualche
milionesimo di milionesimo, cioè qualche picojoule (pJ). Il primo anello della
catena porta dai conti ai joule e poggia sulla stessa fisica della
{doc}`memoria di una GPU </GPU/gerarchia-memoria>`, letta qui con il contatore
invece che con il cronometro.

`````{tab} Elementare

Verrebbe da pensare che l'energia se ne vada nei conti: più moltiplicazioni,
più corrente. Quasi sempre, invece, la corrente serve ad altro.

I valori vengono da una tabella dell'ingegnere Mark Horowitz, di Stanford, e due
righe bastano. Ad esempio, una moltiplicazione con la somma che la segue,
dentro il processore, usa poco meno di cinque picojoule. Per prendere un
numero dalla memoria che sta fuori dal chip ne servono circa 640, più di cento
volte tanto.
Il picojoule è troppo piccolo per essere immaginato; conta il rapporto fra i
due valori.

Il motivo è fisico. Portare un numero dall'esterno al chip obbliga a far
cambiare stato a lunghissime piste di rame, e ogni cambiamento di stato assorbe
corrente. Un numero che sta già dentro muove pochissimo rame, e l'energia è
quella del conto che ci si fa sopra.

Da questi due valori non segue ancora dove finisca la bolletta. Il chip è un
tavolo di lavoro e la memoria di fuori l'armadio in fondo alla stanza, come
nella gerarchia di memoria di una GPU: chi attraversa la stanza per copiare un
numero solo passa la giornata in piedi, chi torna con un foglio da cui ricava
trecento conti non si accorge nemmeno del tragitto.

Dividendo 640 per poco meno di cinque si trova dunque la soglia: circa
centoquaranta conti per ogni numero preso da
fuori (il valore preciso dipende dal chip e da quante cifre ha ogni numero).
[Sotto la soglia la corrente se ne va soprattutto nei viaggi alla memoria; sopra,
nei conti]{.pt-matita}.

Generare una parola alla volta sta molto sotto la soglia, perché il
computer rilegge tutti i pesi del modello per produrne una sola. Servire
molte persone insieme divide quella lettura fra tutte, ma gli appunti di ogni
conversazione vanno riletti comunque uno per uno.

Tagliare i viaggi conta quindi
più di ogni altra cosa, e le tecniche del capitolo sulle GPU servono a questo:
tenere i dati vicino al processore, fare più operazioni in un passaggio solo
invece di andare e tornare. Lì i viaggi frenavano il tempo, qui frenano la
corrente, e le due cose vanno insieme: andare più veloci e consumare meno
coincidono. Addestramento e lettura di un prompt lungo stanno sopra la soglia, e
lì limare i viaggi sposta poco.

`````

`````{tab} Superiore

Le misure di riferimento vengono da una tabella di Horowitz del 2014
sull'energia per operazione in un nodo tecnologico a 45 nm
{cite}`horowitz2014computing`. Una moltiplicazione-accumulo in `float32` somma
una moltiplicazione ($3{,}7$ pJ) e un'addizione ($0{,}9$ pJ), per $4{,}6$ pJ,
mentre una lettura a 32 bit dalla DRAM arriva a circa 640 pJ nella versione
della tabella che circola di più: più di due ordini di grandezza. Un accesso a
una memoria interna al chip è invece dello stesso ordine dell'aritmetica, fra
$5$ e $10$ pJ, e il salto dei due ordini di grandezza sta tutto nell'uscita dal
chip. Finché il dato resta nel silicio, accedervi equivale a calcolarci sopra;
appena esce, costa cento volte tanto. I valori assoluti dipendono dal nodo e dal
progetto, mentre il rapporto è la parte robusta, e col tempo è peggiorato: la
densità dei transistor è migliorata più in fretta dell'energia per bit
trasportato.

Il rapporto fra due costi unitari non basta a stabilire dove finisce l'energia di
un carico: serve l’**intensità aritmetica** del modello roofline, cioè le
operazioni eseguite per ogni byte letto, la stessa grandezza con cui
{doc}`Misurare un servizio </MLOps/metriche-di-servizio>` ha distinto prefill e
decode. L'energia per byte e quella per operazione si pareggiano attorno ai
settanta FLOP/byte: 640 picojoule ogni quattro byte sono 160 pJ per byte,
mentre una moltiplicazione-accumulo da $4{,}6$ pJ vale due operazioni, cioè
$2{,}3$ pJ per FLOP, e $160/2{,}3 \approx 70$. Sono 280 FLOP (140
moltiplicazioni-accumulo) per ogni numero a 32 bit letto.

La tabella circola però in più versioni, e quella più citata, usata finora,
non è della relazione originale ma di una ripresa nella letteratura sulle reti
compresse. I singoli valori differiscono: la moltiplicazione in virgola mobile
a 32 bit è data ora a $3{,}7$ ora a $4$ picojoule; la lettura dalla DRAM è di
$640$ picojoule nella ripresa, riferita a 32 bit, e fra $1{,}3$ e $2{,}6$
nanojoule nella tabella originale, riferita a 64 bit. Riportati allo stesso
byte, i due valori quasi si toccano: $160$ picojoule per byte nella ripresa, fra
$162$ e $325$ nella relazione. Il salto di oltre due ordini di grandezza fra
aritmetica e DRAM compare in entrambe; il pareggio invece si sposta. Con la
moltiplicazione a 4 pJ cade a sessantacinque FLOP per byte invece che a
settanta; con i nanojoule della relazione, divisi per gli otto byte della
lettura, cade fra settanta e centoquaranta, fino al doppio.

Il formato dei dati sposta il pareggio più di ogni altra variante. Nella stessa
tabella la moltiplicazione e l'addizione a 16 bit valgono $1{,}1$ e $0{,}4$
pJ, quindi una moltiplicazione-accumulo a 16 bit usa $1{,}5$ pJ, cioè $0{,}75$
pJ per FLOP: con la DRAM della ripresa il pareggio sale a
$160/0{,}75 \approx 210$ FLOP/byte, e anche con l'interfaccia della DRAM
migliorata, che la relazione stima scendere al più a 10 pJ per bit (80 per
byte), resta attorno a cento. Il giudizio qualitativo non cambia, perché la
generazione sta a circa uno e la lettura di un prompt a qualche migliaio di
FLOP/byte; cambia il valore del pareggio, e conviene sapere da quale tabella e
da quale formato proviene il proprio.

Sotto il pareggio, qualunque sia il suo valore esatto, [l'energia se ne va
quasi tutta in movimento di dati]{.pt-matita}. È il caso della generazione
token per token, dove l'intensità è dell'ordine dell'unità e l'aritmetica è
circa l'uno per cento dell'energia. Sopra il pareggio domina invece
l'aritmetica: la lettura di un prompt lungo e una passata di addestramento
stanno dall'altra parte del ginocchio. La leva del movimento dei dati è quindi
grande dove il carico è limitato dalla memoria (*memory-bound*), che comprende
quasi tutta l'inferenza interattiva, e modesta altrove. Questo motiva, dal lato
dell'energia, l'ingegneria del {doc}`capitolo sulle GPU </GPU/overview>`: il
riuso in shared memory, la fusione dei kernel, la precisione ridotta (che
dimezza i byte da muovere prima ancora di alleggerire i conti) e l'array
sistolico, che esiste per far passare un dato letto una sola volta attraverso
decine di unità di calcolo.

La stessa scomposizione dà l'energia di un token generato. In un passo di
decode con $b$ sequenze di contesto $n_{\text{ctx}}$,

$$
e_{\text{tok}} \approx \Big(\frac{M_w}{b} + n_{\text{ctx}}\, m_{\text{kv}}\Big)\,
e_{\text{byte}} + 2N_p\, e_{\text{FLOP}},
$$

con $N_p$ i parametri, $M_w$ i byte dei pesi, $m_{\text{kv}}$ i byte di KV
cache per token, ed $e_{\text{byte}}$, $e_{\text{FLOP}}$ l'energia per byte
letto dalla DRAM e per operazione. Il termine dei pesi si divide per il batch,
quello della cache no, con la stessa struttura dell'intensità $I(b)$ di
{doc}`Quante repliche accendere </MLOps/capacita-e-costo>`.

Ad esempio, si prendano
$e_{\text{byte}} = 80$ pJ (la DRAM con interfaccia migliorata) ed
$e_{\text{FLOP}} = 0{,}75$ pJ (aritmetica a 16 bit), e un modello da $N_p =
7\cdot10^9$ parametri ($M_w = 14$ GB). A $b = 1$ la lettura dei pesi richiede
circa 1,1 J per token, contro una decina di millijoule di aritmetica: cento
volte tanto. A $b = 64$ scende a circa 17,5 mJ, e a quel punto pesa la cache,
che con contesti di duemila token e 128 KiB per token aggiunge circa 21 mJ a
ogni token di ogni sequenza. Sono valori di un nodo tecnologico vecchio, utili
per le proporzioni e non per la bolletta.

Ne escono due stime grossolane, ma utili. A consuntivo, l'energia di un carico
di lavoro è la potenza media dell'acceleratore per il tempo di esecuzione:
l'approssimazione è grossolana perché la potenza dipende da *che cosa* si
calcola, ma si misura con strumenti già disponibili (`nvidia-smi` espone la
potenza istantanea della GPU, i contatori RAPL l'energia consumata dalla CPU).

A preventivo si parte dai FLOP: un addestramento esegue circa $6N_pD$ FLOP
($N_p$ parametri e $D$ token; la
{doc}`sezione sui grandi modelli linguistici </Transformers/llm>` scrive la
stessa legge con $N$), e
$E \approx 6N_pD / (\text{MFU}\cdot \phi)$, dove $\phi$ sono i FLOP per joule
di picco della scheda (i FLOP al secondo di picco divisi per la sua potenza di
targa) e MFU la frazione di picco raggiunta. Con $N_p = 7\cdot10^9$,
$D = 10^{12}$, una scheda da circa $10^{15}$ FLOP/s a 700 W e MFU del $40\%$,
sono $4{,}2\cdot10^{22}$ FLOP, circa $7\cdot10^{10}$ J, cioè una ventina di MWh
sulle sole schede, e con PUE $1{,}1$ e $400$ g/kWh circa nove tonnellate di
CO₂e. Il calcolo attribuisce alla scheda la potenza di targa per tutto il tempo,
che è un tetto e non una media, e trascura CPU, rete e memoria dei nodi: i due
errori vanno in versi opposti. Indica comunque dove agire, perché l'energia
dipende dall'MFU quanto dall'efficienza della scheda.

`````

## Dal joule al grammo

Il secondo anello esce dal silicio ed entra nell'edificio.

`````{tab} Elementare

Entra corrente in un centro dati, e nei calcolatori non finisce tutta: ne
prendono il condizionamento, le batterie che tengono acceso quando la corrente
salta, gli alimentatori. Quel contorno si misura con il **PUE**, il rapporto fra
la bolletta dell'edificio e la corrente arrivata alle macchine. Una struttura
moderna sta fra 1,1 e 1,3; una vecchia supera 2, e per ogni watt di calcolo ne
brucia un altro per raffreddarlo.

Il PUE, però, è la media di tutto l'edificio su tutto l'anno. Se spegni il tuo
addestramento per una notte, la bolletta cala dell'energia che assorbivano le
tue macchine e di poco altro, perché le luci, le batterie e la ventilazione
restano accese comunque. Chi si addebita anche il venti per cento di contorno
sovrastima il proprio consumo. Il rapporto basta per l'ordine di grandezza, non
per confrontare due lavori sulla stessa macchina.

La stessa elettricità, poi, non inquina uguale dappertutto. Un kilowattora
prodotto dove la rete è idroelettrica o nucleare porta con sé qualche decina di
grammi di anidride carbonica; dove si brucia carbone, più di ottocento. Il
valore cambia anche da un'ora all'altra, perché di notte, o senza vento, la rete
accende centrali diverse. Oltre alla media del luogo e al valore di quell'ora
c'è un terzo modo di contare: chi compra a contratto energia eolica o solare per
il suo centro dati la sottrae dal totale anche nelle ore in cui dalla presa esce
la corrente della rete di tutti, e con quel metodo la stessa corrente risulta
molto più pulita. [Prima di confrontare due cifre bisogna sapere quale dei tre è
stato usato]{.pt-matita}.

I tre fattori (la corrente delle macchine, il contorno dell'edificio, quanto
sporca è la rete) si moltiplicano fra loro, non si sommano: dimezzare il
consumo o spostare il lavoro su una rete che sporca la metà producono lo stesso
effetto sul totale.

Le decisioni che li governano, però, non pesano allo stesso modo. Un gruppo di
ricerca di Google guidato da David Patterson ha misurato quattro scelte
nell'addestramento di un modello grande; il lavoro le elenca senza classifica,
e ordinate per effetto formano una scala. In cima c'è il modello: uno denso
attiva tutti i suoi parametri a ogni parola, uno a scomparti (*mixture of
experts*) ne sveglia solo una piccola parte, e a parità di qualità il denso può
consumare più di dieci volte tanto. Segue il luogo dove si esegue, cioè la rete
elettrica, che sposta le emissioni da cinque a dieci volte anche restando nello
stesso paese. Poi la macchina: una costruita per il machine learning rende da
due a cinque volte più di una generica, a parità di corrente. Per ultimo
l'edificio, da 1,4 a 2 volte, cioè il PUE.

Sono misure del 2021 sull'addestramento e hanno una scadenza. Dal lato del
rispondere, in gioco ci sono le stesse grandezze, quanto modello si attiva e
con quale corrente, ma le ampiezze non sono state misurate lì.

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
raffreddamento); $I_{\text{rete}}$ è la leva di chi sceglie dove e
quando eseguire, e varia di oltre un ordine di grandezza fra reti diverse,
e di alcune volte fra ore diverse della stessa rete.

Si noti che il PUE è un rapporto di struttura, annualizzato: riguarda tutto
l'edificio su tutto l'anno. Moltiplicarlo per l’$E_{\text{IT}}$ di *un* singolo
carico di lavoro presuppone che il contorno cresca in proporzione al carico,
mentre una quota rilevante (illuminazione, gruppi di continuità a vuoto,
ventilazione di base) è fissa: il PUE **marginale** di un lavoro aggiuntivo è di
norma più basso di quello medio, e la formula sovrastima. In direzione opposta,
il PUE non copre né le perdite di trasmissione della rete elettrica né il
consumo d'acqua. Va bene per l'ordine di grandezza, non per confrontare due
lavori sulla stessa macchina.

[Anche $I_{\text{rete}}$ va scelto, perché per lo stesso carico se ne possono
prendere tre valori]{.pt-matita}. C'è la media annuale della rete in cui sta il
centro dati (il metodo *location-based*); c'è l'intensità oraria, o quella
marginale, cioè della centrale che si accende per coprire il carico in più; e
c'è il valore contrattuale, che sconta l'energia rinnovabile acquistata
dall'azienda (il metodo *market-based*). Per spostare un lavoro nel tempo conta
l'intensità oraria, per decidere se un lavoro in più accende una centrale conta
la marginale. La differenza è grande: per il 2024 Google dichiara sui suoi
centri dati 345 g/kWh con il primo metodo e 94 con il terzo (345 meno 251 di
energia pulita acquistata), e con il terzo stima per il prompt di testo mediano
dell'assistente Gemini, nel maggio 2025, 0,24 Wh, 0,03 g di CO₂e (comprese le
emissioni di fabbricazione dell'hardware) e 0,26 millilitri d'acqua
{cite}`elsworth2025measuring`. È una misura datata, da leggere insieme alla
contabilità con cui è stata ottenuta.

Patterson e colleghi {cite}`patterson2021carbon` misurano quattro leve e le
elencano senza classifica; qui le ordiniamo per ampiezza. Sui modelli che
avevano sottomano, nel 2021, la più grande è la scelta del modello: una rete
grande ma ad attivazione sparsa (in cui a ogni passo si attiva solo una piccola
parte dei parametri, a differenza di una rete *densa*) può consumare meno di un
decimo dell'energia di una densa, a parità di qualità. Segue la collocazione
geografica, cioè la rete elettrica in cui gira il lavoro, che sposta le
emissioni di un fattore da cinque a dieci anche all'interno dello stesso paese
e della stessa organizzazione. Le altre due leve sono più piccole: l’hardware
specializzato per il machine learning rende da due a cinque volte più di un
sistema generico, e un centro dati progettato bene è da 1,4 a 2 volte più
efficiente di uno tipico (il PUE).

Quei quattro fattori sono misure su architetture del 2021 e hanno una scadenza.
Se ne ricava una gerarchia: le leve di progetto (quanto modello
serve, dove lo si esegue) contano più di quelle di implementazione. La scelta
del modello sposta molto più della micro-ottimizzazione del codice, mentre la
scelta del luogo sfugge spesso al team che costruisce il modello, perché spetta
a chi gestisce l'infrastruttura.

`````

## Addestrare una volta, servire un miliardo di volte

Addestrare e rispondere si confrontano sull'intera vita del modello, e il
confronto si falsa per un errore di prospettiva, poiché l'addestramento si
vede e la somma delle risposte no.

L'addestramento è un'energia che si spende una volta sola: grande, visibile,
misurabile, databile in un articolo scientifico. Una risposta ha un'energia
minima, che va moltiplicata per un numero enorme: se il modello risponde a
milioni di richieste al giorno per due anni, il totale supera facilmente
l'addestramento che l'ha prodotto. C'è quindi un momento, nella vita di un
modello, in cui la somma di tutte le risposte date fin lì raggiunge l'energia
spesa per costruirlo: è il **punto di pareggio**, $R^\ast =
E_{\text{add}}/e_{\text{inf}}$ richieste, dove $E_{\text{add}}$ è l'energia
dell'addestramento ed $e_{\text{inf}}$ quella media di una risposta; a $r$
richieste al giorno lo si raggiunge in $R^\ast/r$ giorni. Le misure dirette
dell'energia per richiesta, compito per compito, mostrano che $e_{\text{inf}}$
è molto più alta per i modelli generativi che per un classificatore dedicato:
nelle misure del 2023, valori mediani intorno a 0,04 kWh ogni mille inferenze
per generare testo contro 0,002 per classificarlo, e intorno a 1,4 contro 0,007
per le immagini {cite}`luccioni2024power`.

Un esempio con numeri di fantasia: se l'addestramento ha richiesto l'energia di
dieci milioni di risposte e il modello ne dà un milione al giorno, il pareggio
arriva al decimo giorno, e dopo due anni l'addestramento pesa poco più dell'uno
per cento del totale. Un valore universale del pareggio non esiste: dipende da
quanto è grande il modello, da quante richieste riceve e da quanto a lungo resta
in servizio, e va calcolato caso per caso. Un dato misurato però c'è, e va nella
stessa direzione. Nei centri dati di Google, in una settimana d'aprile del 2019,
del 2020 e del 2021, circa tre quinti dell'energia spesa per il machine learning
(ricerca, sviluppo e produzione compresi) andavano a rispondere e due quinti ad
addestrare {cite}`patterson2022plateau`. Ne discende un ordine di priorità, da
tenere presente quando si sceglie fra un modello grande e uno piccolo rifinito
bene.

[Le leve principali sono dunque quelle del rispondere]{.pt-matita}, e
coincidono con quelle già incontrate per risparmiare denaro e tempo. Si
alleggerisce il modello, scrivendone i pesi con meno cifre o togliendo quelli
che servono meno (la quantizzazione e la potatura di
{doc}`LLMOps </MLOps/llmops>`); si servono molte richieste in una volta sola; e
non si ricalcola ciò che è già stato calcolato (il riuso del prefisso di
{doc}`Misurare un servizio
</MLOps/metriche-di-servizio>`). Là servivano a spendere meno e rispondere
prima; qui riducono anche l'energia per richiesta.

Se ne aggiunge una, la più radicale, perché non alleggerisce il modello ma lo
sostituisce. Si chiama **distillazione** e consiste nell'addestrare un modello
piccolo a imitare le risposte di uno grande, per poi mandare in servizio
soltanto il piccolo. È descritta in {doc}`Un modello piccolo che imita
</Efficienza/un-modello-piccolo-che-imita>`, nel capitolo sull'efficienza.
L'addestramento del piccolo si aggiunge a quello del grande, quindi conviene
solo se il servizio dura abbastanza perché l'energia risparmiata a ogni
richiesta lo ripaghi: un pareggio dello stesso tipo, calcolato sulla differenza
fra le energie per richiesta dei due modelli.

## Il carbonio che c'è già dentro

Resta una voce che non compare nella bolletta elettrica: il **carbonio
incorporato** nell'hardware, cioè le emissioni della sua fabbricazione, che
avvengono prima che il dispositivo venga acceso.

`````{tab} Elementare

Un chip, prima di consumare il suo primo watt, ha già richiesto energia: quella
per estrarre e purificare il silicio, per far funzionare una fabbrica che è fra
gli impianti industriali più energivori che esistano, per trasportare il
prodotto. È il carbonio incorporato, la parte dell'impronta che un dispositivo
si porta dietro dalla nascita.

Quanto pesa sul totale dipende da tre cose: per quale frazione della sua vita
il chip lavora, per quanti anni quella vita dura, e quanto è sporca la corrente
che beve mentre lavora. Una scheda da centro dati (un *acceleratore*, cioè un
chip costruito apposta per fare i conti del machine learning e nient'altro)
macina calcoli ventiquattr'ore al giorno per cinque anni: consumando così tanto
e così a lungo, la quota spesa per nascere diventa una parte minore del totale,
tanto minore quanto più sporca è la sua corrente.

Un oggetto che si accende di rado sta all'estremo opposto. Un sensore che si
sveglia due volte al giorno lavora per una frazione minuscola del tempo in cui
esiste, e quindi consuma pochissimo: la parte grossa della sua impronta è stata
fissata in fabbrica, prima che qualcuno lo accendesse, e non c'è modo di
recuperarla.

Ne vengono due strade opposte. Nel centro dati la scheda vecchia conviene
cambiarla appena ne esce una che fa gli stessi conti con meno corrente, a una
condizione: che le emissioni della corrente risparmiata, fino alla fine della
vita della vecchia, superino quelle della fabbricazione della nuova. Dove la
rete brucia carbone succede in fretta; dove è pulita può volerci più della vita
che restava alla vecchia, e allora conviene tenerla. Con il sensore va al
rovescio: sostituirlo significa far fabbricare da capo un dispositivo per
risparmiare briciole di corrente, e [la scelta ambientale che conta diventa
tenerlo in servizio più a lungo]{.pt-matita}, perché un programma che continua
a girare sul dispositivo vecchio è un dispositivo nuovo che non si costruisce.

`````

`````{tab} Superiore

Si distingue fra carbonio **operativo** (quello della catena
$E_{\text{IT}} \times \text{PUE} \times I_{\text{rete}}$) e carbonio
incorporato, cioè le emissioni di fabbricazione, trasporto e smaltimento, che
si ammortizzano sulla vita utile del dispositivo. In funzione del tempo il
totale è

$$
C_{\text{totale}} = C_{\text{operativo}}(t) + C_{\text{incorporato}} \cdot
\frac{t}{T_{\text{vita}}},
$$

dove $t$ è il tempo trascorso in servizio e $T_{\text{vita}}$ la vita utile
attesa del dispositivo, cioè su quanto tempo il carbonio di fabbricazione va
spalmato. Il rapporto fra i due termini, a utilizzo costante, non dipende da
$t$, che si semplifica: dipende dal **fattore di utilizzo**, dalla lunghezza
della vita utile e dall'intensità $I_{\text{rete}}$ che entra nell'operativo. Un
acceleratore da centro dati con utilizzo alto e vita di qualche anno è dominato
dall'operativo; un dispositivo *edge* con utilizzo dell'ordine dell'uno per
cento è dominato dall'incorporato. Per il primo caso, l'addestramento di BLOOM
(176 miliardi di parametri, su una rete a circa 57 g/kWh) dà un ordine di
grandezza: Luccioni e colleghi stimano 24,7 tonnellate di CO₂e dal consumo
dinamico, 14,6 dal consumo delle macchine a vuoto e 11,2 di carbonio incorporato
di server e schede, il 22% di un totale di 50,5 {cite}`luccioni2023estimating`.
Su una rete così pulita l'operativo domina ancora, ma l'incorporato pesa più di
un quinto. Per il secondo caso, nei dispositivi a batteria la fabbricazione è
circa tre quarti delle emissioni dell'intero ciclo di vita
{cite}`gupta2021chasing`.

Le due categorie si ottimizzano in modi opposti. Nel centro dati si
ottimizza il joule per inferenza. Sostituire l'hardware con una generazione più
efficiente conviene anche ambientalmente se il carbonio incorporato della
scheda nuova si ripaga prima della fine della vita della vecchia: con
$\Delta E_{\text{IT}}$ l'energia risparmiata in un anno a parità di lavoro, il
tempo di ripagamento è

$$
T_{\text{rip}} = \frac{C_{\text{incorporato}}^{\text{nuova}}}
{\Delta E_{\text{IT}} \cdot \text{PUE} \cdot I_{\text{rete}}},
$$

e conviene se $T_{\text{rip}}$ è più breve della vita residua della scheda
vecchia. Il tempo è inversamente proporzionale a $I_{\text{rete}}$: su una rete
a qualche decina di grammi per kilowattora si allunga di un ordine di grandezza
rispetto a una rete a carbone, la stessa leva del «dove si esegue». Sull’*edge*
[si ottimizza la longevità]{.pt-matita}: un modello che continua a funzionare
su hardware vecchio evita un ricambio, e quel ricambio pesa più di anni di
funzionamento.

`````

## Che cosa si può fare, e che cosa non funziona

Le leve sono cinque. Le prime tre vengono da Patterson e colleghi e stanno
nell'ordine delle ampiezze di cui sopra: quanto modello serve (uno a scomparti,
o uno più piccolo purché basti allo scopo, conta più di qualunque limatura del
programma), dove si esegue, cioè quanto è pulita l'elettricità di quella rete,
e su che macchina, una fatta apposta o una generica. Le altre due non sono
nella loro misura: l'ora in cui si esegue (ciò che può aspettare si sposta nei
momenti in cui la rete è pulita) e come è scritto il codice, la cui leva
dipende dal carico: è grande dove il carico è limitato dalla memoria, piccola
dove è limitato dal calcolo. Resta fuori l'ultima voce di Patterson, la qualità
dell'edificio, perché non la sceglie chi costruisce il modello ma chi
costruisce il centro dati.

L'efficienza, da sola, non basta a ridurre i consumi totali. Quando un servizio
costa meno, se ne usa di più. È l’**effetto di rimbalzo**, detto anche
paradosso di Jevons, dall'economista che nel 1865 notò come le macchine a
vapore più efficienti avessero fatto crescere il consumo di carbone in
Inghilterra invece
di ridurlo. Nella corsa alla scala degli ultimi anni il risparmio
sull'addestramento è stato spesso reinvestito in modelli più grandi anziché
incassato. Non è però una legge, e c'è almeno un caso, e non piccolo, in cui
l'efficienza ha assorbito la crescita. Lo ha misurato un gruppo guidato da Eric
Masanet, su *Science* nel 2020 {cite}`masanet2020recalibrating`, ricontando
quanta elettricità consumano i centri dati del mondo: fra il 2010 e il 2018 quel
consumo è cresciuto di circa il sei per cento, mentre le istanze di calcolo sono
cresciute del 550 per cento, cioè sono diventate più di sei volte tante. In otto
anni il mondo ha chiesto ai centri dati più di sei volte il lavoro, e il loro
consumo è salito di poco.

È un caso solo, e precede l'ondata dei grandi modelli. Il seguito lo stima
l'Agenzia internazionale dell'energia: nel 2024 i centri dati del mondo hanno
consumato circa 415 TWh, l'1,5% dell'elettricità mondiale, e nello scenario di
base arrivano a circa 945 TWh nel 2030, più del doppio {cite}`iea2025energy`. Le
due serie insieme mostrano che [l'efficienza può assorbire la crescita e non
garantisce di farlo]{.pt-matita}.

Quasi tutte le cifre pubblicate su questo tema sono stime, ottenute da ipotesi
su hardware, utilizzo e mix energetico che raramente sono dichiarate tutte;
confrontarle fra due lavori diversi significa quasi sempre confrontare due
insiemi di ipotesi, non due sistemi. La misura seria si fa in casa propria, con
i contatori dell'hardware che si ha, come per la latenza. Il resto è ordine di
grandezza.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Prendere un numero dalla memoria esterna costa più di cento volte l'energia
  di una moltiplicazione. Dove finisca la bolletta dipende però da quanti conti
  si fanno per ogni numero preso, e la soglia sta attorno ai centoquaranta.
  Generare una parola alla volta sta molto sotto, e lì la corrente se ne va nei
  viaggi, come il tempo nella {doc}`memoria di una GPU
  </GPU/gerarchia-memoria>`: andare più veloci e consumare meno coincidono.
  Leggere un prompt lungo e addestrare stanno invece sopra la soglia, e lì
  limare i viaggi sposta poco.
- L'impronta finale è il prodotto di tre fattori: l'energia che consumano i
  calcolatori, il sovrapprezzo dell'edificio (raffreddamento e perdite: un
  edificio moderno aggiunge dal dieci al trenta per cento, uno vecchio può
  arrivare a raddoppiare il consumo) e quanto sporca è l'elettricità di quella
  rete, che cambia di dieci volte e più fra un luogo e l'altro e di alcune volte
  fra un'ora e l'altra della stessa rete, e che cambia anche a seconda di come
  la si conta. Il sovrapprezzo però è la media di tutto l'edificio su tutto
  l'anno: addebitarlo per intero a un singolo lavoro sovrastima il consumo.
- Rispondere consuma più energia che addestrare, quando il modello resta in
  servizio a lungo: rimpicciolirlo, servire più richieste in una volta sola e
  riusare ciò che è già stato calcolato sono leve ambientali oltre che
  economiche.
- Un chip ha già un'impronta prima di essere acceso (fabbricazione): minore
  per un acceleratore che macina calcoli sempre, tanto più quanto più sporca è
  la sua corrente, e dominante per un oggetto che si accende di rado. Là
  conviene consumare meno, e cambiare scheda se le emissioni della corrente
  risparmiata superano in tempo quelle della fabbrica della nuova; qui conviene
  durare di più.
- Le cifre pubblicate sono quasi tutte stime, con ipotesi che raramente sono
  dichiarate: si misura in casa propria. L'efficienza da sola, poi, non basta a
  far scendere i consumi, perché il risparmio tende a essere reinvestito in
  modelli più grandi invece che incassato. Tende, non deve: è successo anche il
  contrario.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- L'energia non se ne va nei conti ma nel movimento dei dati, a una condizione:
  che l’intensità aritmetica sia bassa. Leggere dalla DRAM costa più di due
  ordini di grandezza in più di una moltiplicazione-accumulo
  {cite}`horowitz2014computing` (un accesso *on-chip* ha l'ordine
  dell'aritmetica: il salto è nell'uscita dal chip), e il pareggio cade attorno
  a $70$ FLOP/byte con dati a 32 bit, fra cento e duecento con dati a 16 bit:
  il decode sta molto sotto, il prefill e l'addestramento stanno sopra. È il
  roofline della {doc}`memoria di una GPU
  </GPU/gerarchia-memoria>`, letto con il contatore invece che col cronometro.
  Nel decode l'energia per token è $(M_w/b +
  n_{\text{ctx}}m_{\text{kv}})\,e_{\text{byte}} + 2N_p\,e_{\text{FLOP}}$: il
  batch divide i pesi, non la cache.
- La catena completa è
  $\text{gCO}_2\text{e} = E_{\text{IT}} \times \text{PUE} \times I_{\text{rete}}$:
  il PUE misura il sovraccarico dell'edificio (da $1{,}1$ a oltre $2$, quasi
  tutto raffreddamento), l’intensità di rete varia di oltre un ordine di
  grandezza fra luoghi, e di alcune volte fra le ore della stessa rete, e va
  dichiarata (media della rete, oraria o marginale, contrattuale). Il PUE però
  è una media annuale di struttura: applicato a un singolo carico sovrastima,
  perché una parte del contorno è fissa.
- L'inferenza supera l'addestramento quando il modello è servito a lungo (a
  Google, nel 2019-2021, circa tre quinti dell'energia del machine learning
  {cite}`patterson2022plateau`): quantizzazione e potatura, *batching*, cache
  del prefisso e distillazione (se il servizio dura abbastanza da ripagare
  l'addestramento del piccolo) sono leve ambientali oltre che economiche.
- Il carbonio incorporato (fabbricazione) è minoritario per un acceleratore
  molto usato (il 22% nell'addestramento di BLOOM, su una rete pulita
  {cite}`luccioni2023estimating`) e dominante per un dispositivo poco usato: là
  si ottimizza il joule, e si sostituisce l'hardware se
  $T_{\text{rip}} = C_{\text{incorporato}}^{\text{nuova}}/(\Delta E_{\text{IT}}\,\text{PUE}\,I_{\text{rete}})$
  è più breve della vita residua; qui si ottimizza la durata.
- Le cifre pubblicate sono stime con ipotesi spesso implicite: si misura in
  casa propria. L'efficienza da sola, poi, non basta, perché il risparmio tende
  a essere reinvestito in scala (l'effetto di rimbalzo); ma non sempre, e fra il
  2010 e il 2018 il consumo elettrico dei centri dati è cresciuto del sei per
  cento a fronte di un carico moltiplicato per più di sei
  {cite}`masanet2020recalibrating`. Per il 2030 l'IEA ne stima più del doppio
  rispetto al 2024 {cite}`iea2025energy`.
```
`````

Con l'energia la sorveglianza di un servizio ha tutte le sue grandezze: dice che
cosa è cambiato (l'errore è salito, i dati in arrivo non somigliano più a quelli
di prima, il tempo è andato nella fila invece che nel prefill, la corrente per
richiesta è cresciuta) e, con le tracce del gateway, anche dove. Non dice su che
cosa il modello si basi per rispondere, e un modello può funzionare per la
ragione sbagliata finché i dati non cambiano. È la domanda del
{doc}`capitolo sull'interpretabilità </Interpretabilita/overview>`, che prova a
guardare dentro il modello invece che intorno.