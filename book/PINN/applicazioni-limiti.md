# Dove la fisica aiuta e dove no

Un neurochirurgo studia un aneurisma: una sacca gonfiata sulla parete di
un'arteria del cervello, che se cede uccide. La domanda che conta è quanto
**preme** il sangue contro quella parete, perché è la pressione a decidere se
e quando si romperà. Ma la pressione dentro un vaso non si misura senza
infilarci un catetere: un gesto invasivo, rischioso, che sull'arteria malata
proprio non si può fare. Quello che si riesce ad avere, invece, è un'immagine:
iniettando un tracciante e filmandolo con la risonanza, si vede *dove va* il
sangue, la sua concentrazione istante per istante. E siccome la macchia di
tracciante viaggia con il sangue che la porta, guardarla spostarsi è quasi
vedere la corrente: la velocità, in qualche modo, è sotto gli occhi. La
pressione no.

Nel 2020 Maziar Raissi, Alireza Yazdani e George Karniadakis pubblicano su
*Science* un lavoro che chiude proprio questo divario
{cite}`raissi2020hidden`: il metodo si chiama *Hidden Fluid Mechanics*, e fa
una cosa che a prima vista sembra magia. Dà in pasto a una rete le immagini
del tracciante (nell'articolo sono immagini simulate: la sola concentrazione,
niente sensori di pressione) e le impone di rispettare le equazioni di
Navier–Stokes, con cui si descrive il moto di un fluido. La rete, per essere
coerente con quelle equazioni *e* con il filmato, è costretta a ricostruire i
**campi** che nel filmato non ci sono: velocità e, soprattutto,
**pressione**. (Un campo, qui, è semplicemente una
grandezza che ha un valore in ogni punto dello spazio e in ogni istante: la
velocità del sangue in quel punto, la pressione in quel punto.)

La magia si scioglie appena si guarda cosa dicono quelle equazioni, ed è bene
scioglierla subito. Navier–Stokes non è un elenco di fatti separati: **lega**
la velocità alla pressione, e dice una cosa precisa, che un fluido accelera
verso i punti dove la pressione è più bassa. Le due grandezze non sono quindi
indipendenti, ed è un legame che si può percorrere all'indietro: vista
l'accelerazione del sangue in ogni punto e in ogni istante, di quanto la
pressione debba calare da un punto al vicino non si è più liberi di
decidere. Il filmato del tracciante dà la velocità; la legge, imposta come
vincolo nella loss, tira fuori la pressione. Su una simulazione di aneurisma
intracranico il metodo stima la pressione che nessuno strumento aveva
misurato.

Con un limite che va detto subito, perché è dentro l'equazione e non nel
metodo: in un fluido incomprimibile la pressione compare solo attraverso le
sue *differenze*, quindi da lì si ricava di quanto la pressione cambia da un
punto all'altro, non a che livello stia. Il valore assoluto resta indefinito
finché non lo si aggancia a una misura sola, da qualche parte. È moltissimo lo
stesso, ed è la vera ragione per cui vale la pena studiare le PINN: non tanto
rifare quello che i solutori classici fanno già benissimo, ma leggere il non
misurabile a partire dal misurabile.

## Il problema inverso, cioè il superpotere

Per capire perché quel risultato sull'aneurisma sia speciale bisogna
distinguere due modi opposti di usare un'equazione.

`````{tab} Elementare

Immagina un forno di cui conosci tutto: la ricetta, la temperatura, i minuti.
Da lì puoi prevedere com'è la torta prima ancora di aprirlo: quanto sarà
gonfia, quanto dorata. Questo è il problema **diretto**: dalla regola completa
alla conseguenza. È il caffè che si raffredda della sezione d'apertura del
capitolo: nota la legge, si ricostruisce la curva.

Il problema **inverso** cammina all'indietro. Non conosci la ricetta: assaggi
la torta e provi a indovinare le dosi. Quanto zucchero? Quanto lievito? Hai il
risultato e cerchi la causa che l'ha prodotto. È incomparabilmente più
difficile (tante ricette diverse possono dare torte simili) ma è quasi sempre
la domanda che interessa davvero: dalla curva del corpo che si raffredda, a
che ora è avvenuto il decesso? Dalle immagini del sangue, quanto preme sulla
parete? La PINN affronta l'inverso con naturalezza disarmante: la dose ignota
diventa una manopola in più da girare durante l'addestramento, finché fisica e
osservazioni non vanno d'accordo.

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

In codice, la promozione del parametro a incognita addestrabile è tre righe,
le stesse della sezione precedente con un altro coefficiente al posto della
rigidezza della molla: il parametro fisico diventa una manopola come tutte le
altre, e finisce nella lista di quelle che l'addestramento aggiorna.

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

Al di là dell'aneurisma, il filone ha prodotto applicazioni concrete. Vale la
pena elencarle con onestà: dove le PINN portano un vantaggio reale, e dove
sono ancora una promessa da verificare.

**Fluidodinamica ed emodinamica.** È il territorio d'elezione, quello di
*Hidden Fluid Mechanics*: ricostruire campi di velocità e pressione da dati di
imaging sparsi e rumorosi, imponendo Navier–Stokes come vincolo
{cite}`raissi2020hidden`. Il valore non è la velocità di calcolo (un solutore
maturo è più rapido) ma la capacità di tenere conto di misure reali tutte
insieme, che in gergo si dice *assimilarle*, e di ricavare da lì ciò che
quelle misure non contengono.

**Identificazione di parametri nei materiali.** Da poche misure di
deformazione o di temperatura, stimare grandezze nascoste trattandole come
incognite dell'equazione: quanto un materiale si piega sotto carico (il modulo
elastico), quanto si lascia attraversare dal calore (la conducibilità), quanto
si lascia attraversare da un fluido (la permeabilità). Funziona quando il
modello fisico è quello giusto; se l'equazione imposta è sbagliata, la stima
è coerentemente sbagliata, e nulla lo segnala.

**Geofisica e sismica.** L'inversione del campo d'onda (risalire alla
struttura del sottosuolo dai sismogrammi registrati in superficie) è un
inverso da manuale, e le PINN sono state proposte per affrontarlo. Resta un
campo di ricerca attivo più che una tecnologia consolidata: i metodi classici
di *full-waveform inversion* sono maturi e difficili da battere.

**Clima e meteo.** Qui serve una precisazione netta, per non confondere due
cose diverse. I grandi modelli meteorologici neurali che negli ultimi anni
hanno fatto notizia (capaci di previsioni globali a dieci giorni in pochi
secondi) **non sono PINN**: non hanno alcuna equazione nella loss. Hanno
imparato a prevedere guardando decenni di mappe del tempo passato, ricostruite
mettendo insieme tutte le osservazioni disponibili (si chiamano dati di
*rianalisi*: la storia meteorologica del pianeta, riscritta in modo uniforme e
completo). Imparano la dinamica dall'osservazione, non dalla fisica imposta.
Ci torneremo a fine sezione, perché sono la porta verso l'idea più
interessante di tutte.

## I limiti, detti con franchezza

Sarebbe disonesto fermarsi qui. Le PINN hanno modi di fallire ben documentati,
e conoscerli è parte del mestiere. Il riferimento obbligato è il lavoro di
Krishnapriyan, Gholami, Zhe, Kirby e Mahoney, che mostra una cosa scomoda:
**una PINN può fallire anche su equazioni semplici**
{cite}`krishnapriyan2021characterizing`. Non perché la rete sia troppo povera
per disegnare quella curva, che lo saprebbe fare benissimo, ma perché il modo
in cui il problema è impostato rende quasi impercorribile la discesa verso il
punteggio minimo.

Il primo motivo è che la loss è un **tiro alla fune**.

`````{tab} Elementare

Ricordi la loss della PINN: due termini sommati, uno che tira verso la fisica
e uno che tira verso quello che si sa già, cioè le misure e il punto di
partenza. È letteralmente un tiro alla fune, con due squadre alle estremità
della corda. E c'è una manopola che decide quanto è forte una delle due
squadre: è quel numero 100 che nella sezione precedente moltiplicava il
termine della partenza. Chi scrive le formule le dà per nome una lettera
greca, «lambda»; e la mette davanti all'uno o all'altro termine secondo quale
delle due squadre gli sembra debole, il che non cambia niente, perché a
contare è solo il rapporto fra le due forze.

Se la giri troppo da una parte, la fisica vince e la rete produce una curva
liscia e regolare che però ignora le misure; se la giri troppo dall'altra, la
rete si incolla ai dati rumorosi e se ne infischia della legge. La soluzione
buona sta dove le due forze si bilanciano, ma trovare quel punto è un'arte,
non una formula: nessuna ricetta universale dice quale sia il valore giusto.
Si prova, si sbaglia, si riprova. Sulla molla della sezione precedente
l'abbiamo fatto: con la manopola su 1 la curva finiva lontana dalla risposta,
con la manopola su 100 molto più vicina, e il solo modo di saperlo era che
lì, per una volta, la risposta la conoscevamo.

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

Il secondo motivo è che una rete impara prima le parti *lisce* di una
funzione, gli andamenti lenti e le tendenze d'insieme, e fatica molto di più
con le oscillazioni rapide e i dettagli fini, cioè con le **alte frequenze**.
Anche questo ha un nome tecnico, lo **spectral bias**, ma l'intuizione è tutta
lì.

`````{tab} Elementare

È il modo di lavorare di un pittore che parte dalle grandi campiture di colore
(il cielo, il prato) e solo alla fine, con pazienza, aggiunge i dettagli
minuti: le foglie, i riflessi. Una rete fa lo stesso da sola: cattura in
fretta la forma d'insieme, aggiunge i particolari fini con enorme lentezza.
Per molti problemi va benissimo. Ma se la soluzione fisica *è* fatta di
increspature rapide (un'onda d'urto, una turbolenza, un fronte che oscilla),
la rete arranca proprio dove servirebbe precisa, e le mancano esattamente i
dettagli che contano.

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

E poi c'è il confronto onesto con i solutori classici, che vale la pena
ripetere senza sconti. Su un problema *standard* (equazione nota, forma
regolare, nessun dato sperimentale da fondere) il conto a passettini visto in
apertura di capitolo, nelle sue due versioni mature (le differenze finite e
gli elementi finiti), fatto bene **vince quasi sempre**: è più veloce di
ordini di grandezza, più accurato, e porta in dote qualcosa che a una rete
addestrata manca del tutto: la garanzia, dimostrata sotto ipotesi che si sanno
controllare, che infittendo la griglia l'errore scende, e di quanto. Una PINN
che impiega minuti dove un solutore maturo impiega millisecondi, e che ogni
tanto fallisce senza preavviso, non è un progresso: è un passo indietro. Le
PINN convengono dove i classici arrancano (dati e leggi da fondere, dimensioni
troppe per qualunque griglia, problemi inversi) non altrove.

## Oltre le PINN: imparare il mestiere, non il compito

C'è un limite più profondo di tutti quelli visti finora, e superarlo apre la
direzione più promettente. Una PINN risolve **un** problema, e uno soltanto.
Quando l'addestramento finisce sono fissati per sempre la regione in cui si
cerca la soluzione (il **dominio**), il punto da cui si parte (la
**condizione iniziale**), quello che accade ai bordi (le **condizioni al
contorno**) e ciò che alimenta il fenomeno dall'esterno, per esempio una
fiamma sotto una sbarra (la **sorgente**). Cambia uno solo di questi
ingredienti e si riaddestra da capo: è come se, per ogni caffè che parte da
una temperatura diversa, si dovessero rifare tutti i conti.

`````{tab} Elementare

Pensa alla differenza tra risolvere *un* esercizio e imparare il *metodo*. Uno
studente che ha risolto un problema di fisica sa la risposta a quel problema;
uno che ha imparato il metodo li risolve tutti, anche quelli che non ha mai
visto, senza rifare la fatica ogni volta. Gli **operatori neurali** fanno la
seconda cosa. Invece di imparare *la soluzione* di un problema, imparano
l'operatore che *mappa le condizioni nella soluzione*: dammi una qualsiasi
temperatura di partenza, una qualsiasi forma del contenitore, e ti restituisco
la curva giusta, subito, senza riaddestrare nulla. Hai imparato il mestiere,
non il singolo compito, ed è riusabile all'infinito.

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
equazioni per ore su un supercomputer, sono costruiti proprio su operatori
appresi: reti che hanno imparato *il metodo* invece della singola risposta. È
lo spirito del Fourier Neural Operator, che lavora scomponendo il campo nelle
onde elementari di cui è fatto; le architetture concrete poi variano da
modello a modello (reti su grafo, Transformer, varianti spettrali). Una
previsione che prima richiedeva ore di supercalcolo esce ora in un tempo
brevissimo, a parità sorprendente di qualità sulle scale di alcuni giorni. È
un risultato che va maneggiato con prudenza (restano aperte questioni su
eventi estremi, stabilità a lungo termine, fisica non rispettata alla lettera)
ma la direzione è inequivocabile, e non passa dalla fisica messa nella loss:
passa dall'imparare l'operatore dai dati.

## Congedo: far collaborare conoscenza e dati

Chiudiamo qui il capitolo, e con esso la lunga rassegna di modelli che occupa
gran parte di questo libro. Le
PINN valgono, alla fine, più come *simbolo* che come tecnica: le migliori fra
quelle che abbiamo incontrato non hanno chiesto di scegliere tra la
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
davanti a delle persone. I capitoli che seguono parlano di portarlo in
produzione e tenercelo, di farsi spiegare perché decide quello che decide, e
di che cosa dobbiamo a chi quelle decisioni le subisce. Solo alla fine, nelle
Conclusioni, guarderemo l'intero percorso dall'alto: a cercare il disegno che,
capitolo per capitolo, era troppo vicino per vedersi.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Il vero superpotere è il **problema inverso**: misurare quello che si può
  misurare e far tirare fuori alla legge quello che non si può, come la
  pressione dentro un vaso sanguigno a partire da un filmato del sangue che
  scorre {cite}`raissi2020hidden`. Funziona perché le leggi della fisica
  **legano fra loro** le grandezze: chi ne conosce una ovunque, dell'altra sa
  già qualcosa.
- Dove serve per davvero: sangue e fluidi, stima delle proprietà nascoste di
  un materiale, struttura del sottosuolo dalle onde dei terremoti. I grandi
  modelli che prevedono il tempo in pochi secondi, invece, **non sono PINN**:
  non hanno nessuna legge dentro il punteggio, hanno solo imparato da decenni
  di mappe del tempo passato.
- Limiti, senza sconti {cite}`krishnapriyan2021characterizing`: il metodo
  fallisce anche su problemi facili; la manopola che bilancia le due squadre
  del tiro alla fune va trovata a mano provando; la rete impara in fretta le
  forme d'insieme e arranca sui dettagli fini (il pittore che lascia le foglie
  per ultime). E soprattutto, come si è visto sulla molla, **un punteggio
  basso non vuol dire risposta giusta**.
- Sui problemi ordinari il conto a passettini di sempre vince quasi sempre,
  in velocità e in garanzie. Le PINN si affiancano, non sostituiscono.
- Il passo successivo sono reti che imparano **il metodo invece del singolo
  compito**: una volta addestrate rispondono a qualunque situazione simile
  senza rifare la fatica. È il motivo per cui certe previsioni meteo escono in
  secondi anziché in ore. I confronti di velocità che si leggono in giro
  vanno però verificati: una di queste reti era stata data per mille volte più
  rapida del metodo classico, e quando altri hanno rifatto il confronto per
  bene è risultata sette volte più rapida {cite}`mcgreivy2024weak`.
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
