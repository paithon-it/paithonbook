# Il futuro nei numeri: serie temporali e forecasting

Negli anni Settanta dell'Ottocento William Thomson, professore a Glasgow (il
mondo lo avrebbe conosciuto come Lord Kelvin), fece costruire a Londra una
macchina di ottone, corde e pulegge che prevedeva le maree. Non con la magia:
osservando anni di misure del livello del mare, Kelvin lo aveva scomposto
nella somma di tante oscillazioni regolari (quella lunare, quella solare,
quelle più sottili) e la macchina, girando una manovella, sommava
meccanicamente quelle onde per *disegnare la marea di un anno intero prima che
accadesse*. È la stessa idea che regge tutto questo capitolo: il futuro si
stima dal passato, purché il passato conservi delle regolarità.

Prevedere è un mestiere antico. Il contadino che legge il cielo per decidere
quando seminare, il mercante che anticipa il prezzo del grano, il meteorologo
che stende le isobare (le linee che uniscono i punti di uguale pressione, e da
cui si legge dove sta andando il tempo): tutti fanno lo stesso gesto, guardare
la storia di un fenomeno per indovinarne il seguito. In inglese quel gesto si
chiama **forecasting**, la parola che dà il titolo a questo capitolo: vuol dire
previsione, e nel campo si usa così spesso che conviene farci subito
l'orecchio.

Come metodo sistematico la disciplina nasce nel 1970, con un libro di George
Box e Gwilym Jenkins destinato a diventare un classico dell'econometria (la
statistica applicata ai fenomeni economici) e dell'ingegneria
{cite}`box2015time`. La loro ricetta sta in tre parole: identificare che forma
ha la serie, stimare i numeri del modello, verificare che il modello non abbia
lasciato fuori niente. Non è la prima idea di previsione statistica: spiegare un
valore con quelli che lo precedono, e cioè l'**autoregressione**, la faceva già
George Udny Yule nel 1927, sulle macchie solari. È però la prima **procedura**
che si possa applicare a una serie qualunque, invece di inventare un metodo
diverso ogni volta. Mezzo secolo dopo, lo statistico
greco Spyros Makridakis mette alla prova quei metodi su larga scala con le
**competizioni M**, gare pubbliche di previsione su decine di migliaia di
serie reali {cite}`makridakis2020m4`. La lezione che ne esce è tanto tecnica
quanto morale: si prevede, sì, ma con umiltà. Nessun modello domina sempre, e
dichiarare *quanto* siamo incerti conta quanto la previsione stessa.

## Che cos'è una serie temporale

Il punto di partenza è la forma dei dati: valori che arrivano **in ordine**, uno
dopo l'altro nel tempo, e che in quell'ordine vanno letti.

`````{tab} Elementare

Immagina un diario in cui ogni riga porta una data: il peso segnato ogni
mattina, la temperatura misurata ogni ora, il numero di scontrini di un
negozio giorno per giorno. Una **serie temporale** è esattamente questo: una
fila di numeri con accanto un orologio.

La differenza con gli altri dati del libro è sottile ma decisiva. Quando
abbiamo parlato di apprendimento supervisionato (riconoscere gatti, filtrare
lo spam), gli esempi erano come palline in un sacchetto: potevi rimescolarle a
piacere senza perdere nulla, l'ordine non contava. Con una serie temporale non
puoi. La temperatura di oggi somiglia a quella di ieri; se mescoli le date,
distruggi proprio l'informazione che ti serve. L'ordine *è* il dato.

`````

`````{tab} Superiore

Una serie temporale è una sequenza di osservazioni $x_1, x_2, \dots, x_T$
indicizzate da un tempo discreto $t$, tipicamente a passo costante (orario,
giornaliero, mensile). Formalmente è la realizzazione di un **processo
stocastico** $\{X_t\}_{t \in \mathbb{Z}}$, cioè una famiglia di variabili
casuali ordinate nel tempo.

Il punto cruciale è che le $X_t$ **non sono indipendenti e identicamente
distribuite** (non i.i.d.). Tutto l'apprendimento supervisionato che abbiamo
incontrato nel capitolo sul Machine Learning poggia, esplicitamente o meno,
sull'ipotesi che gli esempi $(\mathbf{x}^{(i)}, y^{(i)})$ siano campionati in modo
indipendente da un'unica distribuzione; è ciò che rende lecito mescolarli e
separare a caso *train* e *test*. Qui l'ipotesi cade due volte: le osservazioni
sono **dipendenti** ($X_t$ è correlata con $X_{t-1}, X_{t-2}, \dots$) e la loro
distribuzione può **cambiare nel tempo**. Ogni tecnica del capitolo nasce per
convivere con questa doppia rottura.

`````

## I problemi che si pongono

«Serie temporale» è la forma dei dati. Le domande che ci si può fare sopra sono
diverse, e conviene distinguerle subito, perché ciascuna vuole i suoi strumenti
e il suo modo di dare un voto al risultato.

I compiti sono quattro, e il primo è il più importante: il **forecasting**,
stimare i valori *futuri* della serie a partire dai passati. È l'unico in cui la
cosa da indovinare è la serie stessa, più in là nel tempo. Attorno a lui ruotano
gli altri tre:

- **Classificazione di serie**, assegnare un'etichetta a un'intera sequenza:
  un elettrocardiogramma è normale o aritmico? una vibrazione del motore
  segnala un guasto imminente?
- **Rilevamento di anomalie**, individuare i punti in cui la serie si comporta
  in modo inatteso: una frode su una carta, un picco anomalo di traffico, un
  sensore che impazzisce.
- **Imputazione**: ricostruire i valori mancanti *dentro* la serie, quando un
  sensore si è spento per qualche ora e restano dei buchi da riempire.

Il forecasting stesso, poi, non è sempre lo stesso mestiere: due cose ne
cambiano la difficoltà, e conviene guardarle una per volta.

`````{tab} Elementare

Il primo asse riguarda **quante cose** guardiamo insieme. Se prevedi solo la
temperatura di domani osservando le temperature passate, la serie è
**univariata**: una sola grandezza che scorre. Ma spesso conviene guardarne
tante insieme (temperatura, umidità, pressione), perché si aiutano a vicenda:
è il caso **multivariato**.

Il secondo asse riguarda **quanto lontano** guardiamo. Prevedere solo il
prossimo valore (domani) è un *passo singolo*; prevedere l'intera settimana
che verrà è *a più passi*. E qui c'è una trappola quotidiana: le previsioni
del tempo a un giorno ci azzeccano quasi sempre, quelle a dieci giorni molto
meno. La ragione è semplice: fra oggi e il giorno che vuoi prevedere devono
succedere tutte le cose intermedie, e nessuno le ha ancora viste. Più giorni ci
metti in mezzo, più incertezza si somma.

Ma non all'infinito, e questo dipende da com'è fatto il fenomeno. Alcuni hanno
un valore di riposo verso cui tornano sempre, e la temperatura di una città è
uno di quelli: per quanto lontano tu guardi, marzo non se ne andrà mai a cento
gradi. Metti che a marzo, un anno con l'altro, si stia fra i 10 e i 18 gradi:
prevedere il marzo di fra dieci anni vuol dire dire esattamente quello, fra 10 e
18. L'incertezza è cresciuta fino a lì e poi si è fermata, perché a quel punto
si è già ignoranti quanto si può essere, e aspettare altri dieci anni non
peggiora niente.

Altri fenomeni un valore di riposo non ce l'hanno, e il prezzo di un'azione in
borsa è l'esempio classico: dopo un crollo riparte da dove è arrivato, e il
livello di prima non lo rincorre. Lì l'incertezza cresce e basta.

`````

`````{tab} Superiore

Nel caso **univariato** la serie è scalare, $x_t \in \mathbb{R}$; nel
**multivariato** è vettoriale, $\mathbf{x}_t \in \mathbb{R}^N$ con $N$ il numero
di serie osservate insieme, e si vuole sfruttare la correlazione tra le $N$
componenti. Sul secondo asse, il forecasting
**one-step** stima

$$
\hat{x}_{T+1} = f(x_1, \dots, x_T),
$$

mentre quello **multi-step**, su un orizzonte $h$, stima l'intero blocco
$\hat{x}_{T+1}, \dots, \hat{x}_{T+h}$. Le due strategie principali sono la
previsione **diretta** (un modello per ciascun orizzonte) e quella **ricorsiva**
(un modello one-step riapplicato, alimentando le proprie previsioni come input),

$$
\hat{x}_{T+k} = f(x_1, \dots, x_T, \hat{x}_{T+1}, \dots, \hat{x}_{T+k-1}),
$$

dove $h$ è l'orizzonte e $k$ il passo corrente. La ricorsiva è economica ma
soffre di **error compounding**: l'errore al passo $k$ entra nell'input del
passo $k+1$ e si propaga. Con un modello stimato, o non lineare, questo aggiunge
una **distorsione** che la strategia diretta non ha, perché reiniettare una
previsione puntuale in una ricorsione non lineare non restituisce la media
della distribuzione vera ($\mathbb{E}[f(X)] \neq f(\mathbb{E}[X])$): è la
ragione per cui i modelli probabilistici che vedremo campionano invece di
propagare la media.

Questa distorsione va tenuta separata dalla ragione per cui l'incertezza cresce
con l'orizzonte, che è un'altra e vale per **tutte** le strategie, diretta
compresa: fra $T$ e $T+h$ cadono $h$ innovazioni ancora da osservare, e le loro
varianze si sommano {cite}`hyndman2021forecasting`. Su un processo stazionario
quella somma converge a un valore finito, e la banda di previsione smette di
allargarsi; a crescere senza fermarsi è l'incertezza delle serie **non**
stazionarie.

`````

## Perché è un problema diverso (e difficile)

Se le serie temporali meritano un capitolo a sé, e non un paragrafo dentro il
Machine Learning, è perché ognuna delle loro proprietà rompe qualcosa che
altrove davamo per scontato.

La prima è l'**autocorrelazione**. Prendi la serie, fanne una copia e falla
scivolare indietro di un giorno, di due, di dodici: le due si somigliano, e
quella somiglianza è il legame della serie con il proprio passato. Il numero di
passi di cui si è spostata la copia si chiama **ritardo**; in inglese *lag*, ed
è la parola che si trova nel codice e nei manuali (qui ritardo e lag sono la
stessa cosa). La somiglianza si misura mettendo le due file di numeri a coppie,
il primo con il primo, il secondo con il secondo, e guardando se salgono e
scendono insieme: quello che ne esce è un numero solo, il **coefficiente di
autocorrelazione**, ed è il modo più diretto di vedere quanto una serie sia
lontana dai dati indipendenti del resto del libro. Il
codice qui sotto lo calcola su una serie inventata da noi, che sale piano e ha
un ciclo di dodici passi; quello che conta sono i cinque numeri che stampa, e i
paragrafi che seguono li leggono uno per uno, dopo aver detto che scala hanno.

```python
import numpy as np

rng = np.random.default_rng(0)
n = 200
t = np.arange(n)
# serie sintetica: tendenza + stagionalità (periodo 12) + rumore
serie = 0.05 * t + 2.0 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.5, n)

def autocorr(x, lag):
    x = x - x.mean()
    return np.sum(x[lag:] * x[:-lag]) / np.sum(x * x)

print(f"autocorrelazione a lag 1:   {autocorr(serie, 1):.3f}")

# la tendenza gonfia ogni confronto: la togliamo (sottraendo la retta
# che segue la salita) e rifacciamo il conto, a un passo e sul ciclo
detrend = serie - np.polyval(np.polyfit(t, serie, 1), t)
print(f"senza tendenza, a lag 1:    {autocorr(detrend, 1):.3f}")
print(f"senza tendenza, a lag 6:    {autocorr(detrend, 6):.3f}")
print(f"senza tendenza, a lag 12:   {autocorr(detrend, 12):.3f}")

# rimescolando l'ordine, la dipendenza temporale svanisce
mescolata = rng.permutation(serie)
print(f"lag 1, date rimescolate:    {autocorr(mescolata, 1):.3f}")
```

I numeri vanno letti sapendo che scala hanno. Una correlazione vale al massimo
$+1$, e allora i due profili salgono e scendono insieme, perfettamente; vale $0$
quando non c'è nessun legame; e arriva a $-1$ quando si muovono capovolti, cioè
quando a un valore alto dell'uno corrisponde regolarmente un valore basso
dell'altro. È in questo senso che una somiglianza può essere «negativa»: non
vuol dire che non c'è, vuol dire che è rovesciata.

Sulla serie ordinata l'autocorrelazione a un passo vale $0{,}94$, cioè quasi il
massimo: ogni valore anticipa quasi perfettamente il successivo. Una parte di
quel numero, per onestà, la mette la salita: in una serie che cresce sempre due
giorni consecutivi si somigliano più di due giorni presi a caso in tutta la
storia, e questo succede anche se di memoria vera non ce n'è nessuna. Per
levarla di mezzo si tira la retta che segue meglio la salita e si tiene, di ogni
punto, soltanto quanto sta sopra o sotto quella retta: sono due righe di codice,
e il codice qui sopra le ha già fatte. Su quel che resta il coefficiente scende
a $0{,}78$, che è ancora tanto: la dipendenza è vera, non è solo la salita.

La retta va tolta anche per leggere la stagionalità, e per lo stesso motivo:
altrimenti terrebbe alta l'autocorrelazione a qualunque distanza. Fatto questo,
il contrasto è netto. A sei passi di ritardo, cioè mezzo ciclo, la serie si
trova nel punto opposto del giro (dove prima c'era un picco adesso c'è un
avvallamento) e la somiglianza è fortemente **negativa**, $-0{,}87$; a dodici
passi di ritardo, cioè un ciclo intero, torna **alta**, $0{,}83$, perché il
fenomeno è tornato dov'era.

E adesso la prova che conta. Rimescoliamo le date: teniamo gli stessi duecento
numeri e li rimettiamo in fila a caso. Il coefficiente **crolla** a $0{,}14$.
Non è esattamente zero, e non poteva esserlo: rimescolando duecento numeri
qualche somiglianza per puro caso ci scappa sempre, di solito di qualche
centesimo, e questa volta è capitata un po' più grossa. Ma di quel $0{,}94$ non
è rimasto niente. Gli stessi identici valori, in un altro ordine, non prevedono
più niente: quello che rendeva prevedibile la serie non stava nei numeri, stava
nel loro ordine.

Ecco perché nel forecasting **futuro e passato non si mescolano mai**. E c'è un
posto in cui la regola è più facile da dimenticare che altrove, ed è proprio
quello in cui costa di più: quando si tratta di dare un voto al modello (in
gergo, la **validazione**). Se per giudicarlo gli si fanno indovinare dei giorni
che stanno *in mezzo* a quelli su cui si è allenato, gli si sta chiedendo di
riempire un buco avendo davanti i due bordi, che è tutt'altro mestiere che
indovinare il seguito. Il voto che ne esce è gonfiato, e non se ne accorge
nessuno finché il modello non va a lavorare sul futuro vero. La sezione sulla
validazione temporale è dedicata a questo; per ora basti la regola: ci si allena
sul prima, si verifica sul dopo, mai il contrario.

La seconda proprietà è la **non stazionarietà**, ed è quella che manda in crisi
i metodi statistici classici, quelli della prossima sezione.

`````{tab} Elementare

Pensa a un fiume. Se la sua portata oscilla sempre attorno allo stesso valore
medio, con piene e magre di ampiezza costante, il fiume è «stabile»: chi lo
studia oggi può usare le stesse regole di chi lo studiava vent'anni fa. Le cose
che devono restare ferme sono tre: il valore attorno a cui la portata balla,
l'ampiezza con cui balla, e il modo in cui due giorni si somigliano, che deve
dipendere da **quanto** distano fra loro e non da **quando** cadono nel
calendario (due giorni di fila si somigliano uguale, che siano di marzo o di
settembre). Questa stabilità è ciò che i tecnici chiamano **stazionarietà**.

Molte serie vere non sono così. Il prezzo di una casa cresce di decennio in
decennio (la media sale: c'è una **tendenza**), i consumi di gelato salgono
ogni estate e calano ogni inverno (la **stagionalità**), e ogni tanto succede
qualcosa che cambia le regole di colpo, una crisi, una pandemia, una nuova
tecnologia: un **cambio di regime**. Buona parte del lavoro consiste nel
togliere tendenza e stagionalità per riportare la serie a qualcosa di stabile,
su cui i modelli sappiano ragionare.

`````

`````{tab} Superiore

Un processo $\{X_t\}$ con momenti secondi finiti ($\mathbb{E}[X_t^2] < \infty$,
senza cui la richiesta non avrebbe senso) è **stazionario in senso debole** (o in
covarianza) se i suoi primi due momenti non dipendono dal tempo:

$$
\mathbb{E}[X_t] = \mu, \qquad
\mathrm{Var}(X_t) = \sigma^2, \qquad
\mathrm{Cov}(X_t, X_{t+k}) = \gamma(k) \;\; \forall t,
$$

cioè media e varianza costanti e autocovarianza $\gamma(k)$ funzione **solo**
del divario $k$ tra due istanti, non della loro posizione assoluta. È
l'ipotesi su cui poggia l'intera famiglia dei modelli ARMA. Le serie reali la
violano in tre modi ricorrenti: una **tendenza** rende $\mu$ variabile nel
tempo, la **stagionalità** rende $\mu$ periodica (e, quando è stocastica
anziché deterministica, fa dipendere l'autocovarianza dalla posizione $t$
oltre che dal divario $k$), un **cambio di regime**
(rottura strutturale) altera $\mu$, $\sigma^2$ o entrambi da un certo istante
in poi.

La strategia standard è ricondurre la serie alla stazionarietà prima di
modellarla, e lo strumento **dipende da quale** delle violazioni si ha davanti.
Contro una tendenza *stocastica* (una radice unitaria: ogni scossa sposta il
livello per sempre) si usa la **differenziazione**,
$\nabla x_t = x_t - x_{t-1}$, ed è la «I» (*integrated*) dell'ARIMA
{cite}`box2015time`; contro una tendenza *deterministica* (la serie oscilla
attorno a una retta) si stima la retta e si tengono i residui. La sezione
seguente mostra perché scambiare le due non è affatto neutro. Per decidere
esistono test appositi, ADF e KPSS, che hanno ipotesi nulle **opposte** e vanno
letti insieme; li usa, al suo primo passo, la procedura della sezione
seguente. Nessuno dei due, però,
«dimostra» la stazionarietà, esattamente come nessuna diagnostica dimostra che
un modello sia giusto: dicono soltanto se i dati contengono prove contro di
essa.

`````

Autocorrelazione, non stazionarietà, cambi di regime, ordine che conta: sono
modi diversi di dire una cosa sola. In una serie temporale l'**indipendenza**
tra gli esempi (la comoda finzione su cui abbiamo costruito il resto del
machine learning supervisionato) semplicemente non c'è, e ogni metodo del
capitolo è un modo diverso di prenderla sul serio.

## Come è organizzato il capitolo

Il capitolo è in tre sezioni, e si leggono in fila.

1. **Componenti e modelli classici**, come scomporre una serie in tendenza,
   stagionalità e residuo, e i due cavalli di battaglia storici: la famiglia
   **ARIMA** di Box e Jenkins e il lisciamento esponenziale nella forma
   **Holt-Winters**. Sono ancora oggi la **linea di base**, cioè l'avversario
   banale che un metodo più sofisticato deve battere per meritare la fatica che
   costa.
2. **Validazione temporale e feature** (le *feature* sono le colonne di una
   tabella, quelle che si danno in pasto a un modello). Perché la *k-fold*
   mescolata del capitolo sul Machine Learning (dividere gli esempi in fette a
   caso e provare il modello su una fetta per volta) qui è vietata, e come si
   valida sul tempo facendo scorrere in avanti il confine fra passato e futuro
   (è il *backtesting*). Poi come si trasforma una serie in una tabella di
   quelle: una colonna per il valore di ieri, una per la media degli ultimi
   giorni, una per il giorno della settimana, e a quel punto la sanno leggere
   tutti i modelli del capitolo sul Machine Learning.
3. **Forecasting neurale**: le reti che possono guardare solo all'indietro
   (**TCN**), quelle che invece di un numero prevedono un ventaglio di futuri
   possibili (**DeepAR**), i **Transformer** adattati alle serie, e infine i
   **foundation model** (come TimesFM o Chronos), addestrati una volta sola su
   collezioni sterminate di serie e poi capaci di prevedere fenomeni che non
   hanno mai visto.

L'ordine è quello della storia, e la storia del forecasting è una lunga
convivenza fra due famiglie: i metodi **statistici** classici, trasparenti e
sorprendentemente difficili da battere, e i metodi **neurali**, che hanno fame
di dati ma sanno cogliere regolarità più intricate e, guardando insieme migliaia
di serie diverse, portare a ciascuna quello che hanno imparato dalle altre. La
grande lezione delle competizioni M è che la rivalità fra le due è meno netta di
quanto sembri. Nella quarta edizione, la M4, correvano centomila serie e
sessantuno metodi, e a vincere non fu né la statistica pura né il deep learning
puro: fu un **ibrido**, cioè una rete neurale montata sopra uno dei metodi
classici della prossima sezione, in modo che ciascuno dei due facesse il pezzo
in cui era più bravo. Dietro, a fare meglio dei singoli concorrenti, c'erano le
**combinazioni**, cioè la media delle previsioni di più metodi messi insieme
{cite}`makridakis2020m4`.

Attraversa tutte e tre le sezioni un filo rosso, ed è quella stessa lezione: la
previsione seria non è un numero, è un numero *con la sua incertezza*. Un
modello che dice «domani 24 gradi» vale meno di uno che dice «domani fra 22 e 26
gradi, e sono sicuro all'80%», dove quell'80% vuol dire: otto volte su dieci il
valore vero cade dentro la forbice. Il secondo sa quanto poco sa.

Su questo, però, quasi tutti i metodi del capitolo barano un po', e senza
volerlo: quel «fra 22 e 26» tende a essere **più stretto** di quanto sarebbe
onesto, e la forbice che promette di contenere il valore vero otto volte su
dieci lo contiene un po' meno spesso. Non è una fatalità. Quanto stretta sia di
troppo si misura, e la sezione sulla validazione mostra come.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Una **serie temporale** è un diario di numeri con la data accanto. A
  differenza degli esempi del resto del libro, che erano palline in un sacchetto
  e si potevano rimescolare, qui l'ordine *è* l'informazione: se mescoli le date
  distruggi proprio quello che rendeva la serie prevedibile.
- Il compito principale è la **previsione** (*forecasting*): dire come andrà
  avanti la serie, guardando una grandezza sola o molte insieme, per il solo
  giorno dopo o per l'intera settimana. Più lontano guardi, più cose ancora da
  vedere ci sono in mezzo, e più larga è l'incertezza: fino a fermarsi, se il
  fenomeno torna sempre verso un suo valore di riposo, e senza fermarsi mai se
  non ce l'ha. Accanto ci sono altri tre
  compiti: dire che tipo di serie è, trovarci dentro i punti anomali, e
  ricostruire i valori mancanti.
- La difficoltà nasce dal fatto che i valori sono legati fra loro. Ogni valore
  somiglia a quelli vicini (l'**autocorrelazione**), le regole del gioco
  cambiano nel tempo (una **tendenza** che sale, una **stagione** che torna, o
  un cambio improvviso), e nessuna di queste cose capitava con le palline nel
  sacchetto. Una serie si dice **stabile** (i tecnici dicono *stazionaria*)
  quando balla sempre attorno allo stesso valore, con la stessa ampiezza, e
  quando due giorni si somigliano in base a **quanto** distano fra loro e non a
  **quando** cadono nel calendario.
- Nel valutare un modello **non si mescolano futuro e passato**: ci si allena
  sul prima e si verifica sul dopo, sempre.
- Il capitolo procede in tre tappe: i modelli classici, come si valuta
  onestamente una previsione e come si trasforma una serie in una tabella, e
  infine le reti neurali. Con un filo comune: prevedere vuol dire anche
  **dichiarare quanto poco si sa**.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Una **serie temporale** è una sequenza di osservazioni ordinate nel tempo,
  realizzazione di un processo stocastico $\{X_t\}$; a differenza degli esempi
  i.i.d. del resto del machine learning, i suoi valori sono **dipendenti** e
  l'ordine *è* l'informazione: rimescolarli la distrugge.
- I compiti principali sono **forecasting** (uni/multivariato, a passo singolo o
  a più passi), classificazione di serie, rilevamento di anomalie e imputazione.
  Nel multi-step l'incertezza cresce con l'orizzonte perché si sommano le
  varianze delle $h$ innovazioni non ancora osservate; l'*error compounding*
  della strategia ricorsiva è un fenomeno **distinto**, e riguarda la
  distorsione che la reiniezione introduce con modelli stimati o non lineari.
- Ciò che rende il problema difficile è la rottura dell'indipendenza:
  **autocorrelazione**, **non stazionarietà** (tendenza, stagionalità),
  **cambi di regime**. Un processo è **stazionario in senso debole** se media e
  varianza sono costanti e l'autocovarianza fra due istanti dipende **solo** dal
  loro divario $k$, non dalla loro posizione assoluta.
- Nella validazione **non si mescolano futuro e passato**: si addestra sul
  passato e si verifica sul futuro, sempre.
- Il capitolo procede in tre tappe, modelli classici (ARIMA, Holt-Winters),
  validazione temporale e feature, forecasting neurale (TCN, DeepAR,
  Transformer, foundation model), con un filo comune: prevedere significa
  anche **dichiarare la propria incertezza**.
```

`````
