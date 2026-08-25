# Dove la fisica aiuta e dove no

Un neurochirurgo studia un aneurisma: una sacca gonfiata sulla parete di
un'arteria del cervello, che se cede uccide. La domanda che conta è quanto
**preme** il sangue contro quella parete, perché è la pressione a decidere se
e quando si romperà. Ma la pressione dentro un vaso non si misura senza
infilarci un tubicino, il catetere: un gesto invasivo, rischioso, che
sull'arteria malata proprio non si può fare.

Quello che si riesce ad avere, invece, è un filmato. Nel sangue si inietta una
sostanza che le macchine sanno vedere (si chiama **tracciante**), poi la si
segue con la risonanza magnetica, che di immagini ne fa una dopo l'altra: il
risultato è un film in cui si vede, punto per punto e istante per istante,
quanto tracciante c'è. Solo quello: dove sta la macchia. Non quanto va veloce
il sangue, non quanto preme.

Nel 2020 Maziar Raissi, Alireza Yazdani e George Karniadakis pubblicano su
*Science* un lavoro che chiude proprio questo divario
{cite}`raissi2020hidden`: il metodo si chiama *Hidden Fluid Mechanics*, e fa
una cosa che a prima vista sembra magia. Dà in pasto a una rete le immagini
del tracciante (nell'articolo sono immagini simulate al calcolatore, e portano
solo il tracciante: nessun sensore di pressione da nessuna parte) e le impone
di rispettare le equazioni di
Navier–Stokes, con cui si descrive il moto di un fluido. La rete, per essere
coerente con quelle equazioni *e* con il filmato, è costretta a ricostruire i
**campi** che nel filmato non ci sono: velocità e, soprattutto,
**pressione**. (Un campo, qui, è semplicemente una
grandezza che ha un valore in ogni punto dello spazio e in ogni istante: la
velocità del sangue in quel punto, la pressione in quel punto.)

La magia si scioglie appena si guarda che cosa dicono quelle leggi, ed è bene
scioglierla subito. Sono una catena di due anelli, e li percorriamo uno alla
volta.

Il primo anello lega il filmato alla velocità. Il tracciante non si muove da
solo: se ne sta lì e va dove lo porta il sangue, come una macchia di colore in
un fiume. Quindi il modo in cui la macchia si allunga e si sposta non è
compatibile con qualsiasi corrente: è compatibile solo con quella che
l'avrebbe spostata proprio così. La velocità non si vede nel filmato, ma il
filmato la restringe moltissimo.

Il secondo anello lega la velocità alla pressione, e la cosa che dice è quella
che tutti conosciamo senza chiamarla così: un fluido viene spinto verso i
punti dove la pressione è più bassa. Lascia andare l'imboccatura di un
palloncino gonfio e l'aria schizza fuori, perché dentro la pressione è alta e
fuori è più bassa; e più è ripido quel dislivello, più forte è la spinta.
Adesso la stessa frase si legge all'incontrario. Sapere la velocità in ogni
punto e in ogni istante vuol dire sapere anche come sta cambiando, cioè quanto
il sangue accelera: è la mossa della molla, dove dalla curva ricavavamo
pendenza e curvatura. Ma se il sangue accelera, qualcosa lo sta spingendo. E a
spingerlo è il dislivello di pressione. Quindi, saputa l'accelerazione, il
dislivello non è più libero: la legge lo ha già deciso.

Ecco la catena intera. Il filmato restringe la velocità; la velocità inchioda
la pressione; e le due leggi, messe nella loss come penalità esattamente come
si è fatto per la molla, costringono la rete a tirare fuori due grandezze che
nel filmato non c'erano. È il momento di dirlo: nell'esperimento
dell'articolo il flusso era simulato al calcolatore, il che sembra una
scorciatoia e invece è il punto. Solo così la risposta vera si conosce, e si
può controllare se la ricostruzione ci ha preso. Su una simulazione di
aneurisma intracranico il metodo ricostruisce la pressione senza che nessuna
misura di pressione gli sia mai stata data.

Va detto subito un limite, perché sta dentro le equazioni e non nel modo di
risolverle. Se il fluido è **incomprimibile** (l'acqua e il sangue lo sono in
pratica: per quanto li schiacci, il loro volume non cambia), nelle equazioni
la pressione non compare mai da sola, compare sempre come dislivello fra un
punto e il vicino. Il che vuol dire che si ricava di quanto la pressione
cambia da un punto all'altro, non a che livello stia: la ricostruzione può
dire «qui la pressione è più alta di cinque millimetri di mercurio che là»
senza saper dire se qui vale 105 o 205. Per fissare il livello servirebbe una
misura vera, presa da qualche parte, ed è proprio quella che dentro l'arteria
malata non si può prendere. Restano i dislivelli, ed è comunque moltissimo:
sono loro a dire dove la parete è sollecitata di più. Ed è la vera ragione per
cui conviene studiare le PINN: non tanto rifare quello che i **solutori
classici** fanno già benissimo (il conto a passettini dell'apertura del
capitolo, quello che avanza su una fitta rete di puntini), ma leggere il non
misurabile a partire dal misurabile.

## Il problema inverso, cioè il superpotere

Per capire perché quel risultato sull'aneurisma sia speciale bisogna
distinguere due modi opposti di usare un'equazione.

`````{tab} Elementare

Di un forno conosci tutto: la ricetta, la temperatura, i minuti. Da lì puoi
prevedere com'è la torta prima ancora di aprirlo: quanto sarà gonfia, quanto
dorata. Cinque gradi in più e viene appena più scura, senza sorprese. Questo è
il problema **diretto**, dalla regola completa alla conseguenza. È il caffè
che si raffredda: nota la legge, si ricostruisce la curva.

Il problema **inverso** cammina all'indietro. Non conosci la ricetta: assaggi
la torta e provi a indovinare le dosi. Quanto zucchero? Quanto lievito? Hai il
risultato e cerchi la causa che l'ha prodotto. È incomparabilmente più
difficile. Tante ricette diverse danno torte che al palato si somigliano, e in
mano hai una fetta sola, magari con il bordo un po’ bruciato. Sbagli di poco
l'assaggio e la dose che ne ricavi sbaglia di molto: un pizzico di sale che
non avevi sentito, e ti convinci che il lievito fosse il doppio.

Eppure è quasi sempre la domanda che interessa davvero: dalla curva del corpo
che si raffredda, a che ora è avvenuto il decesso? Dal filmato del tracciante,
quanto preme il sangue sulla parete? La PINN affronta l'inverso con
naturalezza disarmante: la dose ignota diventa una manopola in più da girare
durante l'addestramento, finché fisica e osservazioni non vanno d'accordo.

Indovinare le dosi da un assaggio, del resto, non l'ha inventato la rete. Al
calcolatore lo si fa da decenni con altri metodi, spesso spendendo meno. Della
PINN conta che il modo di procedere resta lo stesso, qualunque assaggio
capiti in mano.

`````

`````{tab} Superiore

Nel problema **diretto** l'equazione e i suoi dati al contorno sono noti e si
cerca la soluzione $u$: è tipicamente *ben posto* nel senso di Hadamard
(esistenza, unicità, dipendenza continua dai dati). Nel problema **inverso**
parte della soluzione è osservata (spesso su pochi punti e con rumore) e
l'incognita è un ingrediente dell'equazione stessa: un coefficiente, un
termine sorgente, una condizione al contorno. Questi problemi sono
notoriamente *mal posti*: l'unicità può cadere, e piccole perturbazioni dei
dati ne producono di enormi sulla stima.

La PINN non cambia impianto tra i due casi. Nel diretto minimizza
$\mathcal{L}(\theta)$ sui soli parametri di rete $\theta$; nell'inverso
promuove il parametro fisico ignoto (chiamiamolo $\alpha$) a variabile
addestrabile e minimizza *congiuntamente*

$$
\hat{\theta}, \hat{\alpha} = \arg\min_{\theta,\,\alpha}\ \mathcal{L}(\theta,\alpha),
$$

dove il residuo fisico dentro $\mathcal{L}$ dipende ora anche da $\alpha$, il
cui gradiente arriva dalla stessa passata di backpropagation che aggiorna i
pesi: soluzione e parametro si stimano nello stesso ciclo di discesa.

Qui però va evitata una scorciatoia retorica che si legge spesso, e che
consiste nel dipingere i metodi classici mentre provano un valore, risolvono
tutto, confrontano e riprovano. Non è così che si fa, e non lo è da
quarant'anni: il metodo dello **stato aggiunto** ottiene il gradiente della
funzione di scarto rispetto a tutti i parametri incogniti al costo di una o
due risoluzioni del problema diretto, indipendentemente da quanti siano quei
parametri, e poi scende con un quasi-Newton {cite}`plessix2006adjoint`. Anche
lì c'è un solo problema di ottimizzazione, non un anello annidato. La
differenza vera è che lo stato aggiunto richiede di derivare e scrivere, per
quella specifica equazione, sia il solutore diretto sia quello aggiunto,
mentre la PINN monta un unico problema non vincolato in $(\theta, \alpha)$ e
assorbe misure sparse, rumorose e di natura eterogenea senza cambiare
impianto. È un vantaggio di **uniformità e di costo di implementazione**, non
di complessità algoritmica.

E non è nemmeno un vantaggio della *rete*. La stessa formulazione (residuo
discreto più dati, minimizzati insieme al parametro incognito) si può montare
su un campo discretizzato su griglia anziché su una rete neurale, ed è quello
che fanno Karnakov, Litvinov e Koumoutsakos con ODIL, riportando su problemi
di riferimento un costo computazionale **da due a cinque ordini di grandezza
inferiore** a quello delle PINN: due o tre nell'inferenza di una
conducibilità, che è il caso inverso, cinque su un problema a valori iniziali
risolto con Newton, e su una CPU sola {cite}`karnakov2024discrete`. Il
problema inverso resta il terreno migliore per le PINN, ma «migliore per le
PINN» non vuol dire «senza rivali».

`````

In codice si fa come nella sezione precedente, e non importa quale sia il
numero della fisica che manca. Là era la rigidezza di una molla, qui prendiamo
la diffusività di un materiale, cioè quanto in fretta il calore ci si propaga
dentro: in tutti e due i casi quel numero diventa una manopola come le altre e
finisce nella lista di quelle che l'addestramento gira.

```{code-block} python
:class: pt-non-eseguibile

# Il parametro fisico ignoto (qui la diffusivita', cioe' quanto in fretta
# il calore si propaga nel materiale) diventa una manopola addestrabile,
# indistinguibile da un peso qualsiasi della rete. `rete` e' la candidata
# soluzione della sezione precedente.
alpha = torch.nn.Parameter(torch.tensor(0.5))          # valore iniziale di comodo
ottimizzatore = torch.optim.Adam(                      # ottimizzato insieme ai pesi
    list(rete.parameters()) + [alpha], lr=1e-3
)
```

## Cosa sanno fare, per davvero

Al di là dell'aneurisma, il filone ha prodotto applicazioni concrete. Conviene
elencarle con onestà: dove le PINN portano un vantaggio reale, e dove sono
ancora una promessa da verificare.

**Fluidodinamica ed emodinamica**, cioè il moto dei fluidi in generale e del
sangue in particolare. È il territorio d'elezione, quello di *Hidden Fluid
Mechanics*: ricostruire velocità e pressione punto per punto a partire da
immagini mediche sparse e disturbate, con Navier–Stokes imposta come penalità
{cite}`raissi2020hidden`. Il valore non è la velocità di calcolo (un solutore
maturo è più rapido) ma la capacità di tenere conto di misure reali tutte
insieme, che in gergo si dice *assimilarle*, e di ricavare da lì ciò che
quelle misure non contengono.

**Identificazione di parametri nei materiali.** Da poche misure di
deformazione o di temperatura, stimare grandezze nascoste trattandole come
incognite dell'equazione: quanto un materiale si piega sotto carico (il modulo
elastico), quanto si lascia attraversare dal calore (la conducibilità), quanto
si lascia attraversare da un fluido (la permeabilità). Funziona quando il
modello fisico è quello giusto; se l'equazione imposta è quella sbagliata, la
stima esce sbagliata ma con l'aria di essere giusta, coerente con tutto il
resto, e non c'è niente che lo segnali.

**Geofisica e sismica.** Risalire alla struttura del sottosuolo dai
sismogrammi (i tracciati registrati in superficie dai rilevatori di
vibrazioni) è un inverso da manuale, e le PINN sono state proposte per
affrontarlo. Resta un campo di ricerca attivo più che una tecnologia
consolidata: quello stesso problema ha già i suoi metodi classici, che in
inglese si chiamano *full-waveform inversion*, e sono maturi e difficili da
battere.

**Clima e meteo.** Qui serve una precisazione netta, per non confondere due
cose diverse. I grandi modelli meteorologici neurali che negli ultimi anni
hanno fatto notizia (capaci di previsioni globali a dieci giorni in pochi
secondi) **non sono PINN**: non hanno alcuna equazione nella loss. Hanno
imparato a prevedere guardando decenni di mappe del tempo passato. Quelle
mappe non sono l'archivio grezzo delle misure, che è pieno di buchi e cambia
strumento ogni pochi anni: sono il risultato di un lavoro lungo, in cui tutte
le osservazioni disponibili vengono rimesse insieme dai centri meteorologici e
rese omogenee, così che ogni punto del pianeta e ogni ora abbiano il loro
valore. Si chiamano dati di *rianalisi*. Da lì i modelli imparano la dinamica
dall'osservazione, non dalla fisica imposta. Ci torneremo fra qualche pagina,
in fondo a questa stessa sezione, perché sono la porta verso l'idea più
interessante di tutte.

## I limiti, detti con franchezza

Sarebbe disonesto fermarsi qui. Le PINN hanno modi di fallire ben documentati,
e conoscerli è parte del mestiere. Il lavoro di riferimento è di Aditi
Krishnapriyan e colleghi, e mostra una cosa scomoda: **una PINN può fallire
anche su equazioni semplici** {cite}`krishnapriyan2021characterizing`. Non
perché la rete sia troppo povera per disegnare quella curva, che la saprebbe
disegnare benissimo. Il guasto è più subdolo: la curva giusta esiste ed è a
portata della rete, ma la strada per arrivarci, quella che l'addestramento
percorre abbassando il punteggio un passo alla volta, diventa quasi
impraticabile.

Il primo motivo è che la loss è un **tiro alla fune**.

`````{tab} Elementare

Nella loss della PINN i termini sono due: uno tira verso la fisica, l'altro
verso quello che si sa già, cioè le misure e il punto di partenza. Due
squadre, una per capo della corda. E c'è una manopola che decide quanto è
forte una delle due squadre: è quel 100 che sulla molla moltiplicava il
termine della partenza.

Se la giri troppo da una parte, la fisica vince e la rete produce una curva
liscia che però ignora le misure; se la giri troppo dall'altra, la rete si
incolla alle misure sporche e se ne infischia della legge. La soluzione buona
sta dove le due forze si bilanciano, e trovare quel punto è un'arte: si prova
e si riprova, nessuna ricetta dice il valore giusto. Sulla molla l'abbiamo
fatto: con la manopola su 1 la curva finiva lontana dalla risposta, con 100
molto più vicina, e il solo modo di saperlo era che lì la risposta la
conoscevamo.

E no, non si può lasciarla girare all'addestramento come si fa con la
rigidezza della molla nel problema inverso, anche se la parola «manopola» è la
stessa. L'addestramento gira le manopole nella direzione che abbassa il
punteggio, e questa la porterebbe subito a zero, perché azzerare una delle due
squadre è il modo più rapido di far scendere il totale. Chi prende il voto non
decide come si dà il voto.

Nelle formule scritte da altri quella manopola porta per nome la lettera greca
«lambda», e la si trova davanti all'una o all'altra squadra: a contare è il
rapporto fra le due forze.

Poi c'è un guaio che nessuna posizione della manopola sistema. La squadra
della fisica tira in base a quanto la curva si piega, e il piegamento cambia
moltissimo per uno spostamento minimo. Un capello, e la sua forza raddoppia.
Con un avversario che strattona così non c'è passo che vada bene, e si avanza
al rallentatore o non si avanza affatto. Un rimedio che funziona è cominciare
da una legge addolcita e irrigidirla poco alla volta, mentre la curva si
sistema.

`````

`````{tab} Superiore

La loss composita
$\mathcal{L} = \mathcal{L}_{\text{dati}} + \lambda\,\mathcal{L}_{\text{fisica}}$
è un'ottimizzazione **multi-obiettivo** camuffata da obiettivo singolo. I
gradienti dei due termini possono puntare in direzioni discordi: minimizzare
l'uno peggiora l'altro, e il peso $\lambda$ ne stabilisce a mano il
compromesso. Peggio: il termine fisico contiene operatori differenziali di
ordine alto (derivate seconde, a volte quarte) che rendono il problema **mal
condizionato**, nel senso preciso visto nei richiami di analisi numerica, e la
discesa rallenta o si blocca. De Ryck e colleghi individuano la radice del
guasto non nell'ottimizzatore ma in un operatore preciso, che mette insieme il
**quadrato hermitiano** dell'operatore della PDE e il nucleo tangente del
modello: se quello è mal condizionato l'addestramento è lento o impraticabile
{cite}`deryck2024operator`. Nel regime in cui la rete si comporta come un
modello lineare quell'operatore coincide con l'Hessiano della loss, ed è da
questa lettura che gli autori ricavano il precondizionamento che propongono.

Krishnapriyan e colleghi guardano la stessa cosa da un altro lato e puntano il
dito sulla regolarizzazione soft, cioè sull'imporre la PDE come penalità
anziché come vincolo esatto: è quella, mostrano, a deformare il paesaggio, e
fra i guai che porta c'è proprio un problema peggio condizionato
{cite}`krishnapriyan2021characterizing`. I
rimedi che propongono sono due, e valgono fino a uno o due ordini di grandezza
sull'errore: la *curriculum regularization*, cioè partire da una versione
addolcita dell'equazione e irrigidirla gradualmente, e una decomposizione
sequenziale nel tempo.

`````

Il secondo motivo è che una rete impara in fretta gli andamenti larghi e lenti
di una curva, e fatica moltissimo sulle increspature strette e rapide. Chi
lavora in questo campo le chiama **alte frequenze**, prendendo in prestito la
parola dai suoni, dove alto di frequenza vuol dire acuto: qui vuol dire fitto,
cioè che la cosa cambia parecchie volte nello spazio di poco. Il fenomeno ha
un nome, lo **spectral bias**, ma l'intuizione è tutta lì.

`````{tab} Elementare

È il modo di lavorare di un pittore che parte dalle grandi campiture di colore
(il cielo, il prato) e solo alla fine, con pazienza, aggiunge i dettagli
minuti: le foglie, i riflessi. Una rete fa lo stesso da sola: cattura in
fretta la forma d'insieme, aggiunge i particolari fini con enorme lentezza.
Per molti problemi va benissimo. Ma certe soluzioni fisiche *sono* fatte di
increspature rapide: l'aria attorno a un aereo supersonico, che cambia di
colpo nello spazio di pochi centimetri; l'acqua di un fiume in piena, tutta
vortici piccoli; il bordo fra due masse d'aria che avanza, dove la temperatura
cambia tutta nello spazio di pochi chilometri. Lì la rete arranca proprio dove
servirebbe precisa, e le mancano esattamente i dettagli che contano.

Su un bordo tagliato col coltello succede qualcosa di peggio. Il voto che dice
al pittore se sta migliorando si dà misurando di quanto cambia il colore da un
punto a quello accanto, e su un salto netto quella domanda non ha risposta. La
lentezza non c'entra più: lì il voto non vuol dire niente, e ne servirebbe un
altro, dato su una macchia intera invece che su un punto.

C'è poi il quadro lunghissimo, una giornata intera dipinta su una parete.
Nessuno obbliga il pittore a partire da sinistra e ad andare in ordine: ritocca
un pezzo qua e uno là, ogni tratto guardato da vicino sta in piedi, ma l'alba
che gli avevano dato non arriva mai in fondo alla parete. Viene fuori una
giornata piatta e senza ore, che passa tutti i controlli da vicino ed è
sbagliata guardata intera.

`````

`````{tab} Superiore

Una rete a strati densi apprende le componenti di Fourier a bassa frequenza in
poche iterazioni e quelle ad alta frequenza in un numero di iterazioni molto
maggiore: la velocità di apprendimento decresce con la frequenza. Il fenomeno
è documentato da Rahaman e colleghi {cite}`rahaman2019spectral` e va anche
sotto il nome di *frequency principle* {cite}`xu2020frequency`. Per una PINN è
un problema strutturale, perché molte soluzioni interessanti (fronti ripidi,
strati limite, regimi turbolenti) vivono proprio nelle alte frequenze. Si
mitiga con accorgimenti: *Fourier features* in ingresso, funzioni di
attivazione periodiche, riscalamenti.

Va però tenuto distinto da un altro terreno ostile con cui viene spesso
confuso, quello delle PDE **stiff**. La rigidezza è un rapporto fra autovalori
dell'operatore, e la componente «veloce» di un sistema stiff è tipicamente un
modo fortemente *smorzato*, cioè un decadimento rapido, non un'oscillazione
rapida: esaurito il transitorio, la soluzione di un problema stiff è liscia e
a bassa frequenza, ed è precisamente per questo che i metodi impliciti possono
farci passi enormi. Si costruisce senza fatica un problema la cui rigidezza
cresce di cinque ordini di grandezza mentre il contenuto in frequenza della
soluzione non si muove di un millimetro. Alle PINN i problemi stiff danno
comunque filo da torcere, ma per la ragione vista poco sopra, non per lo
spectral bias: Wang, Teng e Perdikaris riconoscono il modo di fallire nello
squilibrio dei gradienti che la rigidezza produce
{cite}`wang2021understanding`, De Ryck e colleghi nel condizionamento
dell'operatore {cite}`deryck2024operator`. Quel che lo spectral bias spiega
davvero sono i fronti ripidi e gli strati limite, che sono *localmente* ad
alta frequenza, e fra questi il transitorio iniziale di un problema stiff.

C'è poi un caso che lo spectral bias non copre affatto, ed è il più duro:
quando la soluzione ha una **discontinuità vera**, come l'urto di una legge di
conservazione non viscosa. Lì il residuo puntuale non è nemmeno definito
sull'urto, perché le derivate non esistono, e la PINN in forma forte non ha a
che cosa aggrapparsi: non è che impari piano, è che il problema che sta
minimizzando non è quello giusto. La strada, in quel caso, è riscrivere il
vincolo in forma **debole** o integrale, che è un'altra famiglia di metodi.

Sugli **orizzonti temporali lunghi** agisce invece un secondo modo di
fallire, da tenere distinto dal primo: non è lo spectral bias detto in altre
parole, è un meccanismo indipendente. Non è nemmeno un accumulo di
errore passo dopo passo (quella è la malattia degli integratori sequenziali,
e qui di passi non ce ne sono: l'ottimizzazione è globale nel tempo). È che
la loss, sommando residui su punti sparsi in tutto il dominio, non impone
alcun **ordine causale**: nulla obbliga la rete a sistemare prima l'inizio
dell'intervallo e poi il resto, e l'informazione delle condizioni iniziali
non viene propagata in avanti nel tempo. Il residuo può così restare piccolo
mentre la rete collassa su una dinamica banale, plausibile punto per punto e
sbagliata nel complesso. È proprio il difetto che la decomposizione
sequenziale nel tempo di Krishnapriyan et al., vista poco sopra, va a
correggere.

`````

E poi c'è il confronto onesto con i solutori classici, da ripetere senza
sconti. Prendiamo un problema *standard*: equazione nota, forma regolare,
nessun dato sperimentale da tenere insieme alla legge. Lì il conto a
passettini visto in apertura di capitolo, fatto bene, **vince quasi sempre**:
è più rapido di centinaia o migliaia di volte, ed è più preciso. Ne esistono
due versioni mature, e conviene avere i nomi: le **differenze finite** mettono
i puntini in righe e colonne regolari, gli **elementi finiti** ritagliano la
regione in tanti triangolini e sanno quindi seguire una forma qualsiasi.

E il conto a passettini porta in dote qualcosa che a una rete addestrata manca
del tutto: si dimostra, sotto ipotesi che si sanno controllare prima di
partire, che infittendo i puntini l'errore scende, e pure di quanto. Di una
rete addestrata non si sa dire niente del genere. Una PINN che impiega minuti dove un solutore maturo impiega millisecondi, e
che ogni tanto fallisce senza preavviso, è un passo indietro. Le PINN convengono in tre
casi, e fuori di lì no: quando misure e leggi vanno usate insieme per
rispondere alla stessa domanda; quando la risposta dipende da così tante
grandezze che i puntini da mettere sarebbero più di quanti un calcolatore ne
possa tenere; e quando il problema è inverso.

## Oltre le PINN: imparare il mestiere, non il compito

C'è un limite più profondo di tutti quelli visti finora, e superarlo apre la
direzione più promettente. Una PINN risolve **un** problema, e uno soltanto: è
come se, per ogni caffè che parte da una temperatura diversa, si dovesse
rifare tutta la fatica dal principio.

Quando l'addestramento finisce, infatti, restano fissati per sempre quattro
ingredienti, e li si capisce tutti e quattro con la sbarra di ferro
dell'apertura. Dov'è la sbarra e quanto è lunga: è la regione in cui si cerca
la soluzione, il **dominio**. Com'era calda prima che tutto cominciasse: è la
**condizione iniziale**. Che cosa le si tiene attaccato ai due capi, una
fiamma da una parte e un blocco di ghiaccio dall'altra: sono le **condizioni
al contorno**. E che cosa la scalda dall'interno, se qualcosa la scalda: è la
**sorgente**. Cambia uno solo dei quattro, sposta la fiamma o accorcia la
sbarra, e si riaddestra da capo.

`````{tab} Elementare

Chi ha risolto un problema di fisica sa la risposta a quel problema; chi ha
imparato il *metodo* risolve anche quelli che non ha mai visto, purché
somiglino a quelli su cui si è esercitato, e senza rifare la fatica ogni
volta.

C'è una famiglia di reti che impara il metodo. Invece di imparare *la
soluzione* di un problema, imparano il procedimento che porta dalla domanda
alla risposta: dammi una temperatura di partenza qualsiasi, una forma del
contenitore qualsiasi, e ti restituisco subito la curva, senza riaddestrare
niente. È un'approssimazione, buona ma non esatta, e arriva in un istante. Hai
imparato il mestiere, non il singolo compito, ed è riusabile all'infinito.

Il mestiere ha i suoi confini, come li ha quello di una persona. Chi si è
allenato sulle sbarre di ferro che si scaldano non sa per questo come si
raffredda una stanza, e la rete portata fuori dal suo terreno risponde lo
stesso, con la stessa sicurezza, e risponde male. Una rete abbastanza grande
da imparare quel mestiere esiste di sicuro, lo dice un teorema; quanto grande
debba essere, il teorema non lo dice.

Quelle reti si chiamano **operatori neurali**, dove «operatore» è il nome che
i matematici danno appunto a un procedimento che prende una cosa intera e ne
restituisce un'altra intera.

`````

`````{tab} Superiore

Un operatore neurale approssima una mappa
$\mathcal{G}: \mathcal{A} \to \mathcal{U}$ tra **spazi di funzioni**:
l'ingresso non è un vettore ma una funzione intera (il campo delle condizioni
iniziali, dei coefficienti, della sorgente) e l'uscita è la funzione
soluzione. Due architetture hanno segnato il campo. Il **DeepONet** di Lu,
Jin, Pang, Zhang e Karniadakis poggia sul teorema di approssimazione
universale *degli operatori*, dimostrato da Tianping Chen e Hong Chen nel 1995
per operatori **continui** su un compatto di funzioni, e come tutti i teoremi
di quella famiglia dice che una rete abbastanza grande esiste, non quanto
debba essere grande {cite}`chen1995universal`: una rete *branch* codifica la
funzione d'ingresso campionata su un insieme di sensori, una rete *trunk*
codifica il punto di query, e il loro prodotto scalare dà il valore della
soluzione lì {cite}`lu2021learning`. Il **Fourier Neural Operator** di Li,
Kovachki e colleghi parametrizza il nucleo integrale direttamente nello spazio
di Fourier {cite}`li2021fourier`: ogni strato trasforma, tiene le sole
frequenze basse e le ripesa con parametri appresi, antitrasforma. Le alte
frequenze non sono perdute per sempre, altrimenti il filtro sarebbe un
passa-basso e basta: a rimetterle in gioco sono la trasformazione lineare che
scorre accanto al ramo spettrale e le nonlinearità fra uno strato e l'altro.
Il risultato è
*invariante alla risoluzione* (addestri su una griglia, valuti su un'altra) e
su Navier–Stokes gli autori dichiarano un'inferenza fino a circa **tre ordini
di grandezza** più rapida di un solutore pseudospettrale.

Quella cifra però va presa con le stesse pinze che questo capitolo pretende di
usare sulle PINN, e sarebbe scorretto non farlo. Nel corpo dello stesso
articolo il cronometro dà 5 millisecondi contro 2,2 secondi, cioè **440
volte**, non mille. Il confronto non è a parità di accuratezza: il solutore
pseudospettrale è quello che ha *generato* i dati di addestramento, e altrove
nello stesso articolo, nella tabella dell'accuratezza, l'errore relativo
dell'operatore alle due viscosità più basse che provano sta fra l'8% e il 19%
a seconda di quanti esempi gli si danno da studiare. E il confronto è già
stato rifatto da altri: McGreivy e Hakim, in una rassegna sistematica del
settore, replicano quel risultato con un metodo Discontinuous Galerkin e
trovano un vantaggio di **7 volte**, per di più mettendo l'operatore neurale
su GPU contro un portatile {cite}`mcgreivy2024weak`. E fra gli articoli che
dichiarano di battere un metodo numerico classico, la stessa rassegna ne trova
60 su 76 che si confrontano con una baseline debole. Gli operatori neurali
restano la direzione più interessante di tutte; i loro numeri di targa vanno
letti come si leggono tutti gli altri.

`````

Ed eccoci al meteo. I modelli neurali che prevedono il tempo su tutto il
pianeta in pochi secondi, là dove i centri di calcolo tradizionali macinano
equazioni per ore su un supercomputer, sono costruiti proprio così: reti che
hanno imparato *il metodo* invece della singola risposta. È lo spirito di uno
dei capostipiti della famiglia, il **Fourier Neural Operator**, che lavora
scomponendo in onde quel che vede: come un accordo si scompone nelle note che
lo formano, una mappa di temperature si scompone in ondulazioni, quelle larghe
e lente e quelle piccole e fitte, e la rete lavora su quelle invece che sui
singoli punti. Le architetture concrete dei modelli meteo poi variano parecchio
da uno all'altro, ma nessuna di esse ha una legge fisica nella loss: quello che
le fa funzionare è aver imparato dai dati il procedimento, non la fisica
imposta. Una previsione che prima richiedeva ore di supercalcolo esce ora in un
tempo brevissimo, a parità sorprendente di qualità nelle previsioni fino a
qualche giorno.

Un risultato così va maneggiato con prudenza, e non solo perché restano aperte
questioni serie: gli eventi estremi, che sono rari e quindi mal rappresentati
in quello che il modello ha studiato; la tenuta quando si spinge la previsione
lontano nel tempo; e il fatto che, non avendo nessuna legge dentro, un modello
del genere può restituire uno stato dell'atmosfera che la fisica non
ammetterebbe.

Poi c'è una prudenza diversa, che riguarda i numeri di targa, ed è una lezione
che vale ben oltre il meteo. Restiamo sul Fourier Neural Operator, ma
attenzione: quello che segue non riguarda i modelli meteo, riguarda il
problema di prova su cui quella famiglia si è fatta conoscere, un fluido in
due dimensioni. Nel riassunto dell'articolo che lo propone si legge «fino a
tre ordini di grandezza più rapido dei solutori tradizionali», cioè mille
volte; qualche pagina dopo, nello stesso articolo, il cronometro dice
quattrocentoquaranta. E in quel confronto il metodo classico stava lavorando
molto più fine, cioè stava dando una risposta più precisa. Quando altri sono
andati a rifarlo **a parità di precisione**, chiedendo cioè ai due la stessa
accuratezza e poi cronometrando, il vantaggio è sceso a **sette volte**
{cite}`mcgreivy2024weak`, e per giunta con la rete su una scheda grafica
contro un portatile. Sette volte è ancora un bel guadagno. Ma fra sette e
mille c'è la differenza fra un miglioramento e una rivoluzione, e conviene
sapere quale dei due si sta comprando.

## Congedo: far collaborare conoscenza e dati

Chiudiamo qui il capitolo, e con esso la lunga rassegna di modelli che occupa
gran parte di questo libro. Le PINN valgono, alla fine, più come *simbolo* che
come tecnica, e il simbolo è questo: non hanno chiesto di scegliere fra la
conoscenza umana e i dati. Per secoli la scienza ha scritto leggi e le ha
risolte al calcolatore; nel decennio abbondante che va dalla svolta del deep
learning, attorno al 2012, a oggi, il machine learning ha fatto l'opposto,
buttando via le leggi e fidandosi solo dei dati. Le PINN, e ancor più gli
operatori neurali, indicano una terza strada: mettere la legge scritta a mano
e il dato misurato **nella stessa funzione di costo** (che è l'altro nome
della loss, il punteggio da abbassare che ci ha accompagnati per tutto il
libro), e lasciare che si correggano a vicenda. La fisica riempie i vuoti che
i dati non coprono; i dati piegano la fisica dove il modello è incompleto.
Non una che sostituisce l'altra: una collaborazione, scritta in una loss.

È l'ultima delle tante idee che abbiamo montato pezzo per pezzo, dai vettori
dei primi capitoli fino a qui. Da qui in avanti il libro cambia domanda: non
più che cosa un modello sa fare, ma che cosa succede quando lo si mette
davanti a delle persone. I capitoli che seguono parlano di metterlo al lavoro
sul serio, con utenti veri, e di tenercelo negli anni; di farsi spiegare
perché decide quello che decide; e di che cosa dobbiamo a chi quelle decisioni
le subisce. Solo alla fine, nelle
Conclusioni, guarderemo l'intero percorso dall'alto: a cercare il disegno che,
capitolo per capitolo, era troppo vicino per vedersi.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Il vero superpotere è il **problema inverso**: misurare quello che si può
  misurare e far tirare fuori alla legge quello che non si può, come la
  pressione dentro un vaso sanguigno a partire dal filmato di un tracciante
  iniettato nel sangue {cite}`raissi2020hidden`. Funziona perché le leggi della fisica
  **legano fra loro** le grandezze: chi ne conosce una ovunque, dell'altra sa
  già qualcosa.
- Dove serve per davvero: sangue e fluidi, stima delle proprietà nascoste di
  un materiale. Sulla struttura del sottosuolo dalle onde dei terremoti si sta
  ancora provando, e i metodi vecchi per ora tengono. I grandi
  modelli che prevedono il tempo in pochi secondi, invece, **non sono PINN**:
  non hanno nessuna legge dentro il punteggio, hanno solo imparato da decenni
  di mappe del tempo passato.
- Limiti, senza sconti {cite}`krishnapriyan2021characterizing`: il metodo
  fallisce anche su problemi facili; la manopola che bilancia le due squadre
  del tiro alla fune va trovata a mano provando, e resta un guaio che nessuna
  posizione della manopola sistema, perché la squadra della fisica strattona a
  ogni passo (si rimedia partendo da una legge addolcita, da irrigidire poco
  alla volta); la rete impara in fretta le forme d'insieme e arranca sui
  dettagli fini (il pittore che lascia le foglie per ultime). E soprattutto,
  come si è visto sulla molla, **un punteggio basso non vuol dire risposta
  giusta**.
- Sui problemi ordinari il conto a passettini di sempre vince quasi sempre,
  in velocità e in garanzie. Le PINN si affiancano, non sostituiscono.
- Il passo successivo sono reti che imparano **il metodo invece del singolo
  compito**: una volta addestrate rispondono a qualunque situazione simile
  senza rifare la fatica, con una risposta approssimata ma buona, ed è il
  motivo per cui certe previsioni meteo escono in secondi anziché in ore.
  Fuori dal terreno su cui si sono allenate, però, rispondono lo stesso, con
  la stessa sicurezza, e sbagliano.
- I confronti di velocità che si leggono in giro vanno però verificati, e la
  storia più istruttiva non riguarda il meteo ma il problema di prova su cui
  quella famiglia si è fatta conoscere: dichiarato mille volte più rapido del
  metodo classico, rifacendo la corsa **a parità di precisione** è risultato
  sette volte più rapido {cite}`mcgreivy2024weak`.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Il vero superpotere delle PINN è il **problema inverso**: stimare grandezze
  non misurabili (la pressione in un aneurisma) da ciò che si misura, imponendo
  la fisica come vincolo {cite}`raissi2020hidden`. Il parametro ignoto diventa
  una variabile addestrabile, stimata *insieme* alla soluzione. Il vantaggio
  sui classici è di **uniformità**, non di complessità: lo stato aggiunto dà
  il gradiente in due risoluzioni {cite}`plessix2006adjoint`, e sugli stessi
  benchmark ODIL costa da due a cinque ordini di grandezza meno
  {cite}`karnakov2024discrete`.
- Applicazioni reali dove il vantaggio è concreto: emodinamica e fluidodinamica,
  identificazione di parametri nei materiali, inversione geofisica. I grandi
  modelli meteo neurali, invece, **non sono PINN**: sono operatori appresi dai
  dati di rianalisi, senza fisica nella loss.
- Limiti onesti {cite}`krishnapriyan2021characterizing`: le PINN falliscono
  anche su PDE semplici; la loss multi-obiettivo è un **tiro alla fune** da
  bilanciare a mano; lo **spectral bias** frena fronti ripidi e strati limite;
  le **PDE stiff** sono ostili per lo squilibrio dei gradienti e il
  condizionamento dell'operatore {cite}`wang2021understanding`,
  {cite}`deryck2024operator`, non per lo spectral bias; sugli orizzonti lunghi
  manca l'ordine causale, e **residuo piccolo non implica soluzione corretta**
  (lo si è misurato sulla molla della sezione precedente). Sui problemi
  standard i solutori classici vincono quasi sempre in velocità e garanzie.
- Gli **operatori neurali** imparano il mestiere, non il compito: la mappa
  condizioni → soluzione, riusabile senza riaddestrare (DeepONet
  {cite}`lu2021learning` e Fourier Neural Operator {cite}`li2021fourier`). Gli
  speedup dichiarati vanno però verificati a parità di accuratezza: il 1000×
  del FNO diventa 440× nel corpo del paper e 7× in replica indipendente
  {cite}`mcgreivy2024weak`.
- La direzione più promettente non sostituisce la conoscenza umana con i dati:
  li fa **collaborare nella stessa funzione di costo**.
```

`````

Le PINN chiudono la parte del libro che cambia dominio a ogni capitolo, grafi,
cataloghi, serie storiche, equazioni della fisica, cioè la stessa matematica
che si adatta di volta in volta alla forma dei dati. Da qui la
domanda cambia. Il {doc}`capitolo su MLOps </MLOps/overview>` non chiede più che cosa un modello riesca
a imparare, ma che cosa gli succede il giorno dopo, quando smette di essere un
esperimento e diventa un servizio che qualcuno usa davvero.
