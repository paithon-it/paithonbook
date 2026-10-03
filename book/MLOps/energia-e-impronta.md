# Il conto in energia

Di un modello si dichiara quasi tutto: quanti parametri ha, quanti conti costa
una passata (in gergo quanti FLOP, cioè quante singole operazioni
aritmetiche, una moltiplicazione o una somma), quanto è accurato, quanti
millisecondi impiega a rispondere. Una cosa non si dichiara quasi mai, ed è
quanta elettricità consuma.

Il costo energetico del calcolo è una domanda vecchia quanto i chip; sui
modelli di linguaggio l'hanno posta nel 2019 Strubell, Ganesh e McCallum,
provando a mettere un numero sull'addestramento {cite}`strubell2019energy`.
Alcuni di quei numeri sono stati poi corretti. La loro stima per la ricerca
automatica dell'architettura dell'Evolved Transformer supponeva che ogni
candidato fosse addestrato a grandezza piena, mentre la ricerca vera usava un
compito ridotto; rifatte sull'hardware e sul centro dati effettivi, le
emissioni sono risultate 88 volte più basse {cite}`patterson2021carbon`. Le
cifre di questa materia invecchiano in fretta, perché dipendono dall'hardware
di quell'anno, dal centro dati e perfino dall'ora del giorno. Quello che non
invecchia è la catena che porta da un'operazione aritmetica a un grammo di
anidride carbonica, e sono i suoi anelli da conoscere, perché ognuno è una
leva.

## Dal FLOP al joule

L'energia si misura in **joule** (J). Il kilowattora della bolletta vale 3,6
milioni di joule; un'operazione aritmetica su un chip ne richiede qualche
milionesimo di milionesimo, cioè qualche picojoule (pJ). Il primo anello della
catena porta dai conti ai joule, ed è già stato costruito parlando della
{doc}`memoria di una GPU </GPU/gerarchia-memoria>`, anche se lì lo guardavamo
con il cronometro invece che col contatore.

`````{tab} Elementare

Ci si aspetta che l'energia se ne vada nei conti: più moltiplicazioni, più
corrente. Il contatore, quasi sempre, dice un'altra cosa.

I due numeri li ha messi in fila un ingegnere di Stanford, Mark Horowitz, e
sono facili da tenere a mente. Fare un conto dentro il processore (una
moltiplicazione e la somma che la segue) costa poco meno di cinque picojoule.
Andare a prendere un numero nella memoria che sta fuori dal chip ne costa circa
640, più di cento volte tanto. Il picojoule è troppo piccolo perché
immaginarlo abbia senso: conta il rapporto fra i due.

Il motivo è fisico, non informatico. Portare un numero da fuori a dentro il
chip vuol dire far cambiare stato a lunghissime piste di rame, e ogni
cambiamento di stato costa corrente. Un numero che sta già dentro costa quanto
farci sopra un conto: di rame ne muove pochissimo.

Dai due prezzi, però, non segue ancora chi si prende la bolletta. Il chip è un
tavolo di lavoro e la memoria di fuori l'armadio in fondo alla stanza, come nel
capitolo sulle GPU: chi attraversa la stanza per copiare un numero solo passa
la giornata in piedi, chi torna con un foglio da cui ricava trecento conti non
si accorge nemmeno del tragitto. Dividendo i due prezzi, 640 per poco meno di
cinque, si trova la soglia: circa centoquaranta conti per ogni numero preso da
fuori. Chi ne fa meno, con quel numero, spende la corrente soprattutto nei
viaggi; chi ne fa di più, soprattutto nei conti.

Generare una parola alla volta sta molto sotto la soglia: il calcolatore
rilegge tutti i pesi del modello per una parola sola. Scrivere per molte
persone insieme divide quel viaggio fra tutte, ma gli appunti di ciascuna
conversazione vanno riletti lo stesso uno per uno. Tagliare i viaggi vale
tutto, ed è il mestiere delle tecniche di quel capitolo: tenere i dati vicino
al processore, fare più cose in un passaggio solo invece di andare e tornare.
Essere frenati dai viaggi invece che dai conti là valeva per il tempo e qui
vale per la corrente: andare più veloci e consumare meno sono la stessa cosa.
L'addestramento e la lettura di un prompt lungo stanno sopra la soglia, e lì
limare i viaggi sposta poco.

`````

`````{tab} Superiore

Le misure di riferimento vengono da un'unica tabella: l'energia per operazione
in un nodo tecnologico a 45 nm, compilata da Horowitz e resa nota dalla sua
relazione sul problema energetico del calcolo {cite}`horowitz2014computing`.
Una moltiplicazione-accumulo in `float32` mette insieme una moltiplicazione
($3{,}7$ picojoule) e un'addizione ($0{,}9$), quindi costa poco meno di
cinque picojoule, mentre leggere un dato a 32 bit dalla DRAM ne costa circa
640 nella versione della tabella che circola di più: più di due ordini di
grandezza. Un accesso alla memoria che sta *dentro* il chip costa invece quanto
l'aritmetica stessa, dell'ordine dei $5$ pJ, e il salto dei due ordini di
grandezza è tutto nell’uscita dal chip. Finché il dato resta nel silicio,
toccarlo costa quanto calcolarci sopra; appena esce, costa cento volte tanto. I
valori assoluti dipendono dal nodo e dal progetto, ma il rapporto è la cosa
robusta, e nel tempo è peggiorato: la densità dei transistor è migliorata più
in fretta dell'energia per bit trasportato.

Da un rapporto fra costi unitari, però, non segue ancora niente sul budget
totale. Per sapere dove finisce l'energia serve sapere quante operazioni si
fanno per ogni byte letto, cioè l'intensità aritmetica del modello roofline,
la stessa grandezza con cui
{doc}`Misurare un servizio </MLOps/metriche-di-servizio>` ha distinto prefill e
decode. Il pareggio cade attorno ai settanta FLOP/byte, cioè 280 FLOP (140
moltiplicazioni-accumulo) per ogni numero a 32 bit letto, e il conto è breve:
640 picojoule ogni quattro byte fanno 160 pJ per byte, mentre una
moltiplicazione-accumulo da $4{,}6$ pJ vale due operazioni, cioè $2{,}3$ pJ per
FLOP.

La tabella però circola in più versioni, e la più citata, quella appena usata,
non è quella della relazione ma la sua ripresa nella letteratura successiva
sulle reti compresse. I singoli valori differiscono (la moltiplicazione in
virgola mobile a 32 bit è data ora $3{,}7$ ora $4$ picojoule; la lettura dalla
DRAM $640$ picojoule nella ripresa, che è una lettura a 32 bit, e fra $1{,}3$ e
$2{,}6$ nanojoule nella tabella della relazione originale, che è una lettura a
64 bit). Riportati allo stesso byte, i due valori quasi si toccano: $160$
picojoule per byte nella ripresa, fra $162$ e $325$ nella relazione. Quello che
regge in tutte e due è il salto, oltre due ordini di grandezza fra l'aritmetica
e la DRAM. Il pareggio invece si sposta. Chi rifà i conti cambiando la sola
moltiplicazione lo trova a sessantacinque FLOP per byte invece che a settanta;
chi prende i nanojoule della relazione, divisi per gli otto byte della lettura,
lo trova fra settanta e centoquaranta, cioè fino al doppio.

Il formato dei dati lo sposta più di tutte le varianti. Nella stessa tabella la
moltiplicazione e l'addizione a 16 bit costano $1{,}1$ e $0{,}4$ pJ, quindi una
moltiplicazione-accumulo a 16 bit costa $1{,}5$ pJ, cioè $0{,}75$ pJ per FLOP:
con la DRAM della ripresa il pareggio sale a $160/0{,}75 \approx 210$
FLOP/byte, e anche migliorando l'interfaccia della DRAM, che la relazione stima
scendere al più a 10 pJ per bit, cioè 80 per byte, resta attorno a cento. La
morale non cambia, perché la generazione sta a uno e la lettura di un prompt a
qualche migliaio, ma il numero sì, e conviene sapere da dove viene il proprio.

Al di sotto del pareggio, qualunque sia il suo valore esatto, l'energia se ne va
quasi tutta in movimento di dati, ed è il caso della generazione token per
token, dove l'intensità è dell'ordine dell'unità e la quota spesa in aritmetica
è poco più di un punto percentuale. Al di sopra domina invece l'aritmetica: la
lettura di un prompt lungo, o una passata di addestramento, stanno dall'altra
parte del ginocchio. La leva del movimento dei dati è dunque enorme dove il
carico è memory-bound, che è quasi tutta l'inferenza interattiva, e modesta dove
non lo è. È la giustificazione economica di tutta l'ingegneria del capitolo
sulle GPU: il riuso in shared memory, la fusione dei kernel, la precisione
ridotta (che dimezza i byte da muovere prima ancora di dimezzare i conti) e
l'array sistolico, la cui intera ragione d'essere è far attraversare un dato
letto una sola volta a decine di unità di calcolo.

La stessa scomposizione dà l'energia di un token generato. In un passo di
decode con $b$ sequenze di contesto $n_{\text{ctx}}$,

$$
e_{\text{tok}} \approx \Big(\frac{M_w}{b} + n_{\text{ctx}}\, m_{\text{kv}}\Big)\,
e_{\text{byte}} + 2N_p\, e_{\text{FLOP}},
$$

con $N_p$ i parametri, $M_w$ i byte dei pesi, $m_{\text{kv}}$ i byte di KV
cache per token, ed $e_{\text{byte}}$, $e_{\text{FLOP}}$ l'energia per byte
letto dalla DRAM e per operazione. Il termine dei pesi si divide per il batch,
quello della cache no: è la stessa struttura dell'intensità $I(b)$ di
{doc}`Quante repliche accendere </MLOps/capacita-e-costo>`. Con i valori a 16
bit appena visti ($e_{\text{byte}} = 80$ pJ, $e_{\text{FLOP}} = 0{,}75$ pJ) e
un modello da $N_p = 7\cdot10^9$ parametri ($M_w = 14$ GB), a $b = 1$ la
lettura dei pesi costa circa 1,1 J per token, contro una decina di millijoule
di aritmetica: cento volte tanto. A $b = 64$ scende a circa 17,5 mJ, e a quel
punto pesa la cache, che con contesti di duemila token e 128 KiB per token
aggiunge circa 21 mJ a ogni token di ogni sequenza. Sono valori di un nodo
tecnologico vecchio, e servono per le proporzioni, non per la bolletta.

Da qui, la prima stima grossolana ma utile, in due forme. A consuntivo,
l'energia di un carico di lavoro si approssima come potenza media
dell'acceleratore per tempo di esecuzione: è grossolana perché la potenza
dipende da *cosa* si sta calcolando, ma ha il pregio di essere misurabile con
strumenti che esistono già (`nvidia-smi` espone la potenza istantanea, i
contatori RAPL fanno lo stesso per la CPU). A preventivo, si parte dai FLOP: un
addestramento costa circa $6N_pD$ FLOP ($N_p$ parametri e $D$ token; la
{doc}`sezione sui grandi modelli linguistici </Transformers/llm>` scrive la
stessa legge con $N$), e
$E \approx 6N_pD / (\text{MFU}\cdot \phi)$, dove $\phi$ sono i FLOP per joule di
picco della scheda (i FLOP al secondo di picco divisi per la sua potenza di
targa) e MFU la frazione di picco davvero sfruttata. Con $N_p = 7\cdot10^9$,
$D = 10^{12}$, una scheda da circa $10^{15}$ FLOP/s a 700 W e MFU del $40\%$,
sono $4{,}2\cdot10^{22}$ FLOP, circa $7\cdot10^{10}$ J, cioè una ventina di MWh
sulle sole schede, e con PUE $1{,}1$ e $400$ g/kWh circa nove tonnellate di
CO₂e. Il conto fa assorbire alla scheda la sua potenza di targa per tutto il
tempo, che è un tetto e non una media, e lascia fuori CPU, rete e memoria dei
nodi: i due errori vanno in versi opposti, ma il conto dice dove guardare,
perché l'MFU pesa quanto l'hardware.

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
centrali diverse. C'è poi un terzo modo di contare: chi compra a contratto
energia eolica o solare per il suo centro dati la mette nel conto anche nelle
ore in cui dalla presa esce la corrente della rete di tutti, e con quel conto
la stessa corrente risulta molto più pulita. Prima di confrontare due cifre
bisogna sapere quale dei modi è stato usato.

I tre pezzi (la corrente delle macchine, il contorno dell'edificio, quanto
sporca è la rete) si moltiplicano fra loro, non si sommano: dimezzare il
consumo o spostare il lavoro su una rete che sporca la metà fanno lo stesso
effetto sul conto finale.

E non pesano uguale. Un gruppo di ricerca di Google, guidato da David
Patterson, ha messo un numero accanto a quattro decisioni, contando la corrente
che serve ad addestrare un modello grande, e ne è uscita una classifica. In
cima c'è quale modello: ce n'è uno che per rispondere si accende tutto, e uno
a scomparti, che ne sveglia due o tre e lascia spenti gli altri (sono i
modelli a esperti, *mixture of experts*), e a parità di qualità il primo può
consumare dieci volte tanto. Subito dopo viene dove si esegue, cioè su quale
rete, che sposta le emissioni da cinque a dieci volte anche restando nello
stesso paese. Più sotto c'è su che macchina: una fatta apposta fa da due a
cinque volte i conti di una generica con la stessa corrente. In fondo, in che
edificio: da 1,4 a 2 volte, ed è il raffreddamento di poco fa.

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
raffreddamento); $I_{\text{rete}}$ è la leva di chi sceglie dove e
quando eseguire, e varia di oltre un ordine di grandezza fra reti diverse,
e di alcune volte fra ore diverse della stessa rete.

Una precisazione sul PUE, perché la formula insegna a fare un conto ed è così
che il conto sbaglia. Il PUE è un rapporto di struttura, annualizzato:
riguarda tutto l'edificio su tutto l'anno. Moltiplicarlo per l’$E_{\text{IT}}$
di *un* singolo carico di lavoro assume che il contorno cresca in proporzione
al carico, mentre una quota rilevante (illuminazione, gruppi di continuità a
vuoto, ventilazione di base) è fissa: il PUE **marginale** di un lavoro
aggiuntivo è di norma più basso di quello medio, e la formula sovrastima. Nella
direzione opposta, il PUE non copre né le perdite di trasmissione della rete
elettrica né il consumo d'acqua. Va bene per l'ordine di grandezza, non per
confrontare due lavori sulla stessa macchina.

Anche $I_{\text{rete}}$ va scelto, perché per lo stesso carico se ne possono
prendere tre valori. C'è la media annuale della rete in cui sta il centro dati
(il metodo *location-based*); c'è l'intensità oraria, o quella marginale,
cioè della centrale che si accende per coprire il carico in più; e c'è il
valore contrattuale, che sconta l'energia rinnovabile acquistata dall'azienda
(il metodo *market-based*). Per spostare un lavoro nel tempo conta l'intensità
oraria, per decidere se un lavoro in più accende una centrale conta la
marginale. La differenza è grande: per il 2024 Google dichiara sui suoi centri
dati 345 g/kWh con il primo metodo e 94 con il terzo, e con il terzo stima per
il prompt di testo mediano dell'assistente Gemini, nel maggio 2025, 0,24 Wh,
0,03 g di CO₂e (comprese le emissioni di fabbricazione dell'hardware) e 0,26
millilitri d'acqua {cite}`elsworth2025measuring`. È un ordine di grandezza
datato, e dice anche con quale contabilità va letto.

L'analisi di Patterson e colleghi {cite}`patterson2021carbon` mette in fila le
ampiezze delle quattro leve, e non sono affatto uguali fra loro. Sui modelli
che avevano sottomano, nel 2021, pesava di più la scelta del modello: una
rete grande ma ad attivazione sparsa (una in cui, per rispondere, si accende
ogni volta solo una piccola parte della rete, invece che tutta come in una
rete *densa*) poteva consumare meno di un decimo di una densa a parità di
qualità. Le stava vicina la collocazione geografica, cioè in quale rete
elettrica si esegue il lavoro, che sposta le emissioni di un fattore fra cinque
e dieci, anche restando dentro lo stesso paese e la stessa organizzazione. Più
sotto le altre due, che il paper tiene distinte: l’hardware specializzato
per il machine learning rende da due a cinque volte più di un sistema generico,
e un centro dati progettato bene è da 1,4 a 2 volte più efficiente di uno
tipico (è il PUE di poco fa).

Quei quattro numeri sono misure su architetture di quell'anno, e come tutte le
misure hanno una scadenza; quello che regge è la loro morale, ed è già
abbastanza forte. Le leve di progetto (quanto modello serve, e dove lo si
esegue) contano più di quelle di implementazione, e nessuna delle due sta dove
di solito si cerca: non nella micro-ottimizzazione del codice, che sposta molto
meno, e non tutte nelle mani del team che costruisce il modello, visto che la
scelta del luogo è di qualcun altro.

`````

## Addestrare una volta, servire un miliardo di volte

Un confronto fra addestrare e rispondere si fa sull'intera vita del modello, e
c'è un errore di prospettiva che lo falsa: addestrare fa notizia, e rispondere
no.

Addestrare è un costo che si paga una volta sola: grande, ben visibile, si può
misurare, si può datare, si può scrivere in un articolo scientifico. Rispondere
è un costo minuscolo moltiplicato per un numero enorme: una singola risposta
consuma pochissimo, ma se il modello risponde a milioni di richieste al giorno
per due anni, il totale supera facilmente l'addestramento che l'ha prodotto.
C'è quindi un momento, nella vita di un modello, in cui la somma di tutte le
risposte date fin lì raggiunge il costo di averlo costruito: è il **punto di
pareggio**, e ha una forma semplice, $R^\ast = E_{\text{add}}/e_{\text{inf}}$
richieste, dove $E_{\text{add}}$ è l'energia dell'addestramento ed
$e_{\text{inf}}$ quella media di una risposta; a $r$ richieste al giorno lo si
raggiunge in $R^\ast/r$ giorni. Le misure dirette del costo per richiesta,
compito per compito, mostrano che per i modelli generativi $e_{\text{inf}}$ sta
ordini di grandezza sopra quello di un classificatore dedicato
{cite}`luccioni2024power`.

Un esempio con numeri di fantasia: se addestrare è costato quanto dieci milioni
di risposte e il modello ne dà un milione al giorno, il pareggio arriva al
decimo giorno, e dopo due anni l'addestramento pesa poco più dell'uno per cento
del totale. Un valore universale del pareggio non esiste: dipende da quanto è
grande il modello, da quante richieste riceve e da quanto a lungo resta acceso,
e va calcolato caso per caso. Un dato misurato però c'è, e va nella stessa
direzione: nei centri dati di Google, in una settimana d'aprile del 2019, del
2020 e del 2021, circa tre quinti dell'energia spesa per il machine learning
andavano a rispondere e due quinti ad addestrare {cite}`patterson2022plateau`.
Quello che è stabile è l'ordine di priorità che ne discende, e conviene tenerlo
in mente quando si sceglie fra un modello grande e uno piccolo rifinito bene.

Ne segue che le leve che contano sono quelle del rispondere, non quelle
dell'addestrare. E la buona notizia è che sono le stesse leve già viste per
risparmiare denaro e tempo. Si alleggerisce il modello, scrivendone i pesi con
meno cifre o togliendo quelli che servono meno (la quantizzazione e la potatura
di {doc}`LLMOps </MLOps/llmops>`); si servono molte richieste in una volta sola;
e non si ricalcola ciò che è già stato calcolato (il riuso del prefisso di
{doc}`Misurare un servizio </MLOps/metriche-di-servizio>`). Là erano modi di
spendere meno e rispondere prima; sono la stessa cosa vista da un'altra
finestra.

Se ne aggiunge una, ed è la più radicale, perché non alleggerisce il modello:
lo sostituisce. Si chiama distillazione e consiste nell'addestrare un
modello piccolo a imitare le risposte di uno grande, per poi mandare in
servizio soltanto il piccolo. La costruisce
{doc}`Un modello piccolo che imita </Efficienza/un-modello-piccolo-che-imita>`,
nel capitolo sull'efficienza.

## Il carbonio che c'è già dentro

Resta una voce che non compare nella bolletta elettrica: il **carbonio
incorporato** nell'hardware, cioè le emissioni della sua fabbricazione, che
avvengono prima che il dispositivo venga acceso.

`````{tab} Elementare

Un chip, prima di consumare il suo primo watt, è già costato energia: quella
per estrarre e purificare il silicio, per far funzionare una fabbrica che è
fra gli impianti industriali più energivori che esistano, per trasportare il
prodotto. È il carbonio incorporato, la parte dell'impronta che un dispositivo
si porta dietro dalla nascita.

Quanto pesi dipende da tre cose: per quale frazione della sua vita quel chip
lavora davvero, per quanti anni quella vita dura, e quanto è sporca la
corrente che beve mentre lavora. Una scheda da centro dati (un *acceleratore*,
cioè un chip costruito apposta per fare i conti del machine learning e
nient'altro) macina calcoli ventiquattr'ore al giorno per cinque anni:
consumando così tanto e così a lungo, quello che ha speso per nascere diventa
una parte piccola del totale, tanto più piccola quanto più sporca è la sua
corrente.

Un oggetto che si accende di rado sta all'estremo opposto. Un sensore che si
sveglia due volte al giorno lavora per una frazione minuscola del tempo in cui
esiste, e quindi consuma pochissimo: la parte grossa della sua impronta è stata
fissata in fabbrica, prima che qualcuno lo accendesse, e non c'è modo di
recuperarla.

Da cui due strade opposte. Nel centro dati la scheda vecchia conviene cambiarla
appena ne esce una che fa gli stessi conti con meno corrente, a una condizione:
che la corrente risparmiata, prima che la vecchia arrivasse a fine vita, pesi
più di quanto è costato fabbricare la nuova. Dove la rete brucia carbone
succede in fretta; dove è pulita può volerci più della vita che restava alla
vecchia, e allora conviene tenerla. Con il sensore va al rovescio: sostituirlo
vuol dire pagare da capo la fabbrica per risparmiare briciole, e la scelta
ambientale che conta diventa tenerlo in servizio più a lungo, perché un
programma che continua a girare sul dispositivo vecchio è un dispositivo nuovo
che non si costruisce.

`````

`````{tab} Superiore

Si distingue fra carbonio **operativo** (quello del conto di poco fa,
$E_{\text{IT}} \times \text{PUE} \times I_{\text{rete}}$) e carbonio
incorporato, cioè le emissioni di fabbricazione, trasporto e smaltimento, che
si ammortizzano sulla vita utile del dispositivo. Il conto complessivo è

$$
C_{\text{totale}} = C_{\text{operativo}}(t) + C_{\text{incorporato}} \cdot
\frac{t}{T_{\text{vita}}},
$$

dove $t$ è il tempo trascorso in servizio e $T_{\text{vita}}$ la vita utile
attesa del dispositivo, cioè su quanto tempo il carbonio di fabbricazione va
spalmato. Il rapporto fra i due termini, a utilizzo costante, non dipende da
$t$, che si semplifica: si gioca sul **fattore di utilizzo**, su quanto è
lunga quella vita utile e sull'intensità $I_{\text{rete}}$ che entra
nell'operativo. Un acceleratore da centro dati con utilizzo alto e vita di
qualche anno è dominato dall'operativo; un dispositivo *edge* con utilizzo
dell'ordine dell'uno per cento è dominato dall'incorporato. Un ordine di
grandezza per il primo caso viene dall'addestramento di BLOOM, 176 miliardi di
parametri su una rete a circa 57 g/kWh: Luccioni e colleghi stimano 24,7
tonnellate di CO₂e dal consumo dinamico, 14,6 dal consumo delle macchine a
vuoto e 11,2 di carbonio incorporato di server e schede, il 22% di un totale di
50,5 {cite}`luccioni2023estimating`. Su una rete così pulita l'operativo domina
ancora, ma l'incorporato è tutt'altro che trascurabile. Per il secondo caso,
sui dispositivi a batteria la fabbricazione pesa circa tre quarti delle
emissioni dell'intero ciclo di vita {cite}`gupta2021chasing`.

La conseguenza progettuale è che le due categorie richiedono ottimizzazioni
opposte. Nel centro dati si ottimizza il joule per inferenza. Sostituire
l'hardware con una generazione più efficiente conviene anche ambientalmente se
il carbonio incorporato della scheda nuova si ripaga prima della fine della
vita della vecchia: con $\Delta E_{\text{IT}}$ l'energia risparmiata in un
anno a parità di lavoro, il tempo di ripagamento è

$$
T_{\text{rip}} = \frac{C_{\text{incorporato}}^{\text{nuova}}}
{\Delta E_{\text{IT}} \cdot \text{PUE} \cdot I_{\text{rete}}},
$$

e conviene se $T_{\text{rip}}$ è più breve della vita residua della scheda
vecchia. Il tempo è inversamente proporzionale a $I_{\text{rete}}$: su una rete
a qualche decina di grammi per kilowattora si allunga di un ordine di grandezza
rispetto a una rete a carbone, ed è la stessa leva del «dove si esegue».
Sull’*edge* si ottimizza la longevità: un modello che continua a funzionare su
hardware vecchio evita un ricambio, e quel ricambio pesa più di anni di
funzionamento.

`````

## Che cosa si può fare, e che cosa non funziona

Tirando le somme, le leve sono cinque. Le prime tre sono le prime tre della
classifica di Patterson e colleghi, nello stesso ordine: quanto modello serve
davvero (uno a scomparti, o uno dieci volte più piccolo purché basti allo
scopo, batte qualunque limatura del programma), dove si esegue, cioè quanto è
pulita l'elettricità di quella rete, e su cosa si esegue, cioè una macchina
fatta apposta contro una generica. Le altre due la classifica non le misurava:
quando si esegue, spostando ciò che può aspettare nelle ore in cui la rete è
pulita, e infine come è scritto il codice, che è la leva che conta meno di
tutte. Fuori dall'elenco resta la quarta voce della classifica, la qualità
dell'edificio, e non per distrazione: quella non la sceglie chi costruisce il
modello, la sceglie chi costruisce il centro dati.

Due avvertenze finali, che valgono più di molte buone intenzioni.

La prima è che l'efficienza, da sola, non basta a ridurre i consumi totali.
Quando un servizio costa meno, se ne usa di più: è l'effetto di rimbalzo, detto
anche paradosso di Jevons, dall'economista che nel 1865 notò come le macchine a
vapore più efficienti avessero fatto crescere il consumo di carbone in
Inghilterra invece di ridurlo. Nella corsa alla scala degli ultimi anni il
risparmio sull'addestramento è stato spesso reinvestito in modelli più grandi
anziché incassato. Non è però una legge, e c'è almeno un caso, e non piccolo,
in cui l'efficienza ha davvero assorbito la crescita. Lo ha misurato un gruppo
guidato da Eric Masanet, su *Science* nel 2020
{cite}`masanet2020recalibrating`, ricontando quanta elettricità consumano i
centri dati del mondo: fra il 2010 e il 2018 quel consumo è cresciuto di circa
il sei per cento, mentre nello stesso periodo il lavoro che ci girava dentro si
è moltiplicato per più di sei (le istanze di calcolo sono cresciute del 550 per
cento). Attenzione a non confondere le due cifre: la prima è un pochino in più,
la seconda è sei volte tanto. In otto anni il mondo ha chiesto ai centri dati
sei volte il lavoro, e loro hanno consumato quasi uguale: lì l'efficienza la
crescita se l'è mangiata tutta.

È un caso solo, e precede l'ondata dei grandi modelli. Il seguito lo stima
l'Agenzia internazionale dell'energia: nel 2024 i centri dati del mondo hanno
consumato circa 415 TWh, l'1,5% dell'elettricità mondiale, e nello scenario di
base arriveranno a circa 945 TWh nel 2030, più del doppio {cite}`iea2025energy`.
Il caso di Masanet mostra che l'efficienza può assorbire la crescita, e non
garantisce che lo faccia: a cadere è l'idea che basti da sola.

La seconda riguarda i numeri. Quasi tutte le cifre pubblicate su questo tema
sono stime, ottenute da ipotesi su hardware, utilizzo e mix energetico che
raramente sono dichiarate per intero; confrontarle fra due lavori diversi
significa quasi sempre confrontare due insiemi di ipotesi, non due sistemi. La
misura seria si fa in casa propria, con i contatori dell'hardware che si ha,
esattamente come per la latenza. Il resto è ordine di grandezza, ed è già
molto.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Prendere un numero dalla memoria esterna costa più di cento volte una
  moltiplicazione. Dove finisca la bolletta dipende però da quanti conti si
  fanno per ogni numero preso, e la soglia sta attorno ai centoquaranta.
  Generare una parola alla volta sta molto sotto, e lì la corrente se ne va nei
  viaggi, come il tempo nella
  {doc}`memoria di una GPU </GPU/gerarchia-memoria>`: andare più veloci e
  consumare meno sono la stessa cosa. Leggere un prompt lungo e addestrare
  stanno invece sopra la soglia, e lì limare i viaggi sposta poco.
- L'impronta finale è il prodotto di tre fattori: l'energia che consumano i
  calcolatori, il sovrapprezzo dell'edificio (raffreddamento e perdite: un
  edificio moderno aggiunge dal dieci al trenta per cento, uno vecchio può
  arrivare a raddoppiare il conto) e quanto sporca è l'elettricità di quella
  rete, che cambia di dieci volte fra un luogo e l'altro e di alcune volte fra
  un'ora e l'altra della stessa rete, e che cambia anche a seconda di come la si
  conta. Il sovrapprezzo però è la media di tutto l'edificio su tutto l'anno:
  addebitarlo per intero a un singolo lavoro fa il conto più caro del vero.
- Rispondere costa più che addestrare, quando il modello resta in servizio
  a lungo: rimpicciolirlo, servire più richieste in una volta sola e riusare
  ciò che è già stato calcolato sono leve ambientali oltre che economiche.
- Un chip ha già un'impronta prima di essere acceso (fabbricazione): piccola
  per un acceleratore che macina calcoli sempre, tanto più quanto più sporca è
  la sua corrente, e dominante per un oggetto che si accende di rado. Là
  conviene consumare meno, e cambiare scheda se la corrente risparmiata ripaga
  in tempo la fabbrica della nuova; qui conviene durare di più.
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
- L'energia non se ne va nei conti ma nel movimento dei dati, a una
  condizione: che l’intensità aritmetica sia bassa. Leggere dalla DRAM costa
  più di due ordini di grandezza rispetto a una moltiplicazione-accumulo
  {cite}`horowitz2014computing` (mentre un accesso *on-chip* costa quanto
  l'aritmetica: il salto è nell'uscita dal chip), e il pareggio cade attorno a
  $70$ FLOP/byte con dati a 32 bit, fra cento e duecento con dati a 16 bit: il
  decode sta molto sotto, il prefill e l'addestramento stanno sopra. È il
  roofline della {doc}`memoria di una GPU </GPU/gerarchia-memoria>`, letto con
  il contatore invece che col cronometro. Nel decode l'energia per token è
  $(M_w/b + n_{\text{ctx}}m_{\text{kv}})\,e_{\text{byte}} + 2N_p\,e_{\text{FLOP}}$:
  il batch divide i pesi, non la cache.
- La catena completa è
  $\text{gCO}_2\text{e} = E_{\text{IT}} \times \text{PUE} \times I_{\text{rete}}$:
  il PUE misura il costo dell'edificio (da $1{,}1$ a oltre $2$, quasi tutto
  raffreddamento), l’intensità di rete varia di oltre un ordine di
  grandezza fra luoghi, e di alcune volte fra le ore della stessa rete, e va
  dichiarata (media della rete, oraria o marginale, contrattuale). Il PUE però
  è una media annuale di struttura: applicato a un singolo carico sovrastima,
  perché una parte del contorno è fissa.
- L'inferenza supera l'addestramento quando il modello è servito a lungo (a
  Google, nel 2019-2021, circa tre quinti dell'energia del machine learning
  {cite}`patterson2022plateau`): quantizzazione e potatura, *batching*, cache
  del prefisso e distillazione sono leve ambientali oltre che economiche.
- Il carbonio incorporato (fabbricazione) è minoritario per un acceleratore
  molto usato (il 22% nell'addestramento di BLOOM, su una rete pulita
  {cite}`luccioni2023estimating`) e dominante per un dispositivo poco usato: là
  si ottimizza il joule, e si sostituisce l'hardware se
  $T_{\text{rip}} = C_{\text{incorporato}}^{\text{nuova}}/(\Delta E_{\text{IT}}\,\text{PUE}\,I_{\text{rete}})$
  è più breve della vita residua; qui si ottimizza la durata.
- Le cifre pubblicate sono stime con ipotesi spesso implicite: si misura in
  casa propria. E l'efficienza da sola non basta, perché il risparmio tende a
  essere reinvestito in scala (l'effetto di rimbalzo); ma non sempre, e fra il
  2010 e il 2018 il consumo elettrico dei centri dati è cresciuto del sei per
  cento a fronte di un carico moltiplicato per più di sei
  {cite}`masanet2020recalibrating`. Per il 2030 l'IEA ne stima più del doppio
  rispetto al 2024 {cite}`iea2025energy`.
```
`````

La sorveglianza di tutto l'anello MLOps dice che cosa è cambiato e, con le
tracce del gateway e le scomposizioni del conto, anche dove: l'errore è salito,
i dati in arrivo non somigliano più a quelli di prima, il tempo è andato nella
fila invece che nel prefill. Non dice su che cosa il modello si stia basando per
rispondere, e un modello può funzionare per la ragione sbagliata finché i dati
non cambiano. È la domanda del
{doc}`capitolo sull'interpretabilità </Interpretabilita/overview>`, che prova
a guardare dentro il modello invece che intorno.
