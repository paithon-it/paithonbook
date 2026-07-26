# Dove la fisica aiuta e dove no

Un neurochirurgo studia un aneurisma: una sacca gonfiata sulla parete di
un'arteria del cervello, che se cede uccide. La domanda che conta è quanto
**preme** il sangue contro quella parete, perché è la pressione a decidere se
e quando si romperà. Ma la pressione dentro un vaso non si misura senza
infilarci un catetere — un gesto invasivo, rischioso, che sull'arteria
malata proprio non si può fare. Quello che si riesce ad avere, invece, è
un'immagine: iniettando un tracciante e filmandolo con la risonanza, si vede
*dove va* il sangue, la sua concentrazione istante per istante. La velocità,
in qualche modo, è sotto gli occhi. La pressione no.

Nel 2020 Maziar Raissi, Alireza Yazdani e George Karniadakis pubblicano su
*Science* un lavoro che chiude proprio questo divario
{cite}`raissi2020hidden`: il metodo si chiama *Hidden Fluid Mechanics*, e fa
una cosa che a prima vista sembra magia. Dà in pasto a una rete le immagini
del tracciante — la sola concentrazione, niente sensori di pressione — e le
impone di rispettare le equazioni di Navier–Stokes, la fisica esatta di come
un fluido si muove. La rete, per essere coerente con quelle equazioni *e* con
il filmato, è costretta a ricostruire i campi che nel filmato non ci sono:
velocità e, soprattutto, **pressione**. Su una simulazione di aneurisma
intracranico il metodo stima la pressione che nessuno strumento aveva
misurato. Questa è la vera ragione per cui vale la pena studiare le PINN: non
tanto rifare quello che i solutori classici fanno già benissimo, ma leggere
il non misurabile a partire dal misurabile.

## Il problema inverso, cioè il superpotere

Per capire perché quel risultato sull'aneurisma sia speciale bisogna
distinguere due modi opposti di usare un'equazione.

`````{tab} Elementare

Immagina un forno di cui conosci tutto: la ricetta, la temperatura, i minuti.
Da lì puoi prevedere com'è la torta prima ancora di aprirlo — quanto sarà
gonfia, quanto dorata. Questo è il problema **diretto**: dalla regola
completa alla conseguenza. È il caffè che si raffredda della sezione
d'apertura del capitolo: nota la legge, si ricostruisce la curva.

Il problema **inverso** cammina all'indietro. Non conosci la ricetta:
assaggi la torta e provi a indovinare le dosi. Quanto zucchero? Quanto
lievito? Hai il risultato e cerchi la causa che l'ha prodotto. È
incomparabilmente più difficile — tante ricette diverse possono dare torte
simili — ma è quasi sempre la domanda che interessa davvero: dalla curva del
corpo che si raffredda, a che ora è avvenuto il decesso? Dalle immagini del
sangue, quanto preme sulla parete? La PINN affronta l'inverso con
naturalezza disarmante: la dose ignota diventa una manopola in più da girare
durante l'addestramento, finché fisica e osservazioni non vanno d'accordo.

`````

`````{tab} Superiore

Nel problema **diretto** l'equazione e i suoi dati al contorno sono noti e si
cerca la soluzione $u$: è tipicamente *ben posto* nel senso di Hadamard —
esistenza, unicità, dipendenza continua dai dati. Nel problema **inverso**
parte della soluzione è osservata (spesso su pochi punti e con rumore) e
l'incognita è un ingrediente dell'equazione stessa: un coefficiente, un
termine sorgente, una condizione al contorno. Questi problemi sono
notoriamente *mal posti* — l'unicità può cadere, e piccole perturbazioni dei
dati ne producono di enormi sulla stima.

La PINN non cambia impianto tra i due casi. Nel diretto minimizza
$\mathcal{L}(\theta)$ sui soli parametri di rete $\theta$; nell'inverso
promuove il parametro fisico ignoto — chiamiamolo $\alpha$ — a variabile
addestrabile e minimizza *congiuntamente*

$$
\hat{\theta}, \hat{\alpha} = \arg\min_{\theta,\,\alpha}\ \mathcal{L}(\theta,\alpha),
$$

dove il residuo fisico dentro $\mathcal{L}$ dipende ora anche da $\alpha$, il
cui gradiente arriva dalla stessa passata di backpropagation che aggiorna i
pesi. Nessun ciclo esterno di tentativi: soluzione e parametro si stimano
nello stesso ciclo di discesa. È questa economia — un solo problema di
ottimizzazione per quello che ai solutori classici costa un anello di
tentativi annidato — il vero valore aggiunto delle PINN sugli inversi.

`````

In codice, la promozione del parametro a incognita addestrabile è tre righe,
le stesse richiamate dalla sezione precedente: il coefficiente fisico è un
tensore come un altro, e finisce nella lista che l'ottimizzatore aggiorna.

```python
import torch

# Il parametro fisico ignoto (qui la diffusività) diventa una manopola
# addestrabile, indistinguibile da un peso qualsiasi della rete.
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
*Hidden Fluid Mechanics*: ricostruire campi di velocità e pressione da dati
di imaging sparsi e rumorosi, imponendo Navier–Stokes come vincolo
{cite}`raissi2020hidden`. Il valore non è la velocità di calcolo — un
solutore maturo è più rapido — ma la capacità di *assimilare* misure reali e
inferire ciò che quelle misure non contengono.

**Identificazione di parametri nei materiali.** Da poche misure di
deformazione o di temperatura, stimare grandezze nascoste — un modulo
elastico, una conducibilità, una permeabilità — trattandole come incognite
dell'equazione. Funziona quando il modello fisico è quello giusto; se
l'equazione imposta è sbagliata, la stima è coerentemente sbagliata, e nulla
lo segnala.

**Geofisica e sismica.** L'inversione del campo d'onda — risalire alla
struttura del sottosuolo dai sismogrammi registrati in superficie — è un
inverso da manuale, e le PINN sono state proposte per affrontarlo. Resta un
campo di ricerca attivo più che una tecnologia consolidata: i metodi classici
di *full-waveform inversion* sono maturi e difficili da battere.

**Clima e meteo.** Qui serve una precisazione netta, per non confondere due
cose diverse. I grandi modelli meteorologici neurali che negli ultimi anni
hanno fatto notizia — capaci di previsioni globali a dieci giorni in pochi
secondi — **non sono PINN**: non hanno alcuna equazione nella loss, sono
*operatori appresi* direttamente da decenni di dati di rianalisi. Imparano la
dinamica dall'osservazione, non dalla fisica imposta. Ci torneremo, perché
sono la porta verso l'idea più interessante di tutte.

## I limiti, detti con franchezza

Sarebbe disonesto fermarsi qui. Le PINN hanno modi di fallire ben
documentati, e conoscerli è parte del mestiere. Il riferimento obbligato è il
lavoro di Krishnapriyan, Gholami, Zhe, Kirby e Mahoney
{cite}`krishnapriyan2021characterizing`, che mostra una cosa scomoda:
**una PINN può fallire anche su equazioni semplici**, non per un difetto della
rete — è abbastanza espressiva — ma perché l'impostazione stessa rende il
paesaggio della loss quasi impossibile da percorrere per la discesa del
gradiente.

Il primo motivo è che la loss è un **tiro alla fune**.

`````{tab} Elementare

Ricordi la loss della PINN: due termini sommati, uno che tira verso i dati e
uno che tira verso la fisica. È letteralmente un tiro alla fune, con due
squadre alle estremità della corda. C'è una manopola, $\lambda$, che decide
quanto è forte la squadra "fisica". Se la stringi troppo, la fisica vince e la
rete produce una curva liscia e regolare che però ignora le misure; se la
molli troppo, la rete si incolla ai dati rumorosi e se ne infischia della
legge. La soluzione buona sta dove le due forze si bilanciano — ma trovare
quel punto è un'arte, non una formula. Con $\lambda = 1$ la torta viene in un
modo, con $\lambda = 100$ in un altro, e non c'è una ricetta universale che
dica quale sia il valore giusto: si prova, si sbaglia, si riprova.

`````

`````{tab} Superiore

La loss composita
$\mathcal{L} = \mathcal{L}_{\text{dati}} + \lambda\,\mathcal{L}_{\text{fisica}}$
è un'ottimizzazione **multi-obiettivo** camuffata da obiettivo singolo. I
gradienti dei due termini possono puntare in direzioni discordi: minimizzare
l'uno peggiora l'altro, e il peso $\lambda$ ne stabilisce a mano il
compromesso. Peggio: il termine fisico contiene operatori differenziali di
ordine alto — derivate seconde, a volte quarte — che rendono il problema
**mal condizionato**, nel senso preciso visto nei richiami di analisi
numerica (il numero di condizionamento dell'Hessiano esplode) e la discesa
rallenta o si blocca. Krishnapriyan et al. {cite}`krishnapriyan2021characterizing`
mostrano che è proprio la regolarizzazione soft — imporre la PDE come
penalità anziché come vincolo esatto — a deformare il paesaggio, e propongono
rimedi come la *curriculum regularization* (partire da una versione
addolcita dell'equazione e irrigidirla gradualmente) e una decomposizione
sequenziale nel tempo, con guadagni fino a uno o due ordini di grandezza
sull'errore.

`````

Il secondo motivo ha un nome tecnico ma un'intuizione semplice: lo **spectral
bias**. Le reti neurali imparano prima le componenti *lisce* di una funzione —
gli andamenti lenti, le tendenze globali — e faticano molto di più con le
componenti ad **alta frequenza**, le oscillazioni rapide e i dettagli fini.

`````{tab} Elementare

È il modo di lavorare di un pittore che parte dalle grandi campiture di
colore — il cielo, il prato — e solo alla fine, con pazienza, aggiunge i
dettagli minuti: le foglie, i riflessi. Una rete fa lo stesso da sola: cattura
in fretta la forma d'insieme, aggiunge i particolari fini con enorme lentezza.
Per molti problemi va benissimo. Ma se la soluzione fisica *è* fatta di
increspature rapide — un'onda d'urto, una turbolenza, un fronte che oscilla —
la rete arranca proprio dove servirebbe precisa, e le mancano esattamente i
dettagli che contano.

`````

`````{tab} Superiore

Il fenomeno, noto anche come *frequency principle*, è che una rete a strati
densi apprende le componenti di Fourier a bassa frequenza in poche iterazioni
e quelle ad alta frequenza in un numero di iterazioni molto maggiore: la
velocità di apprendimento decresce con la frequenza. Per una PINN è un
problema strutturale, perché molte soluzioni interessanti (fronti ripidi,
strati limite, regimi turbolenti) vivono proprio nelle alte frequenze. Si
mitiga con accorgimenti — *Fourier features* in ingresso, funzioni di
attivazione periodiche, riscalamenti — ma resta una ragione di fondo per cui
le PDE **stiff** (con scale temporali molto diverse tra loro) e gli
**orizzonti temporali lunghi** mettono le PINN in seria difficoltà: l'errore
si accumula passo dopo passo e la rete non riesce a inseguire la dinamica
rapida.

`````

E poi c'è il confronto onesto con i solutori classici, che vale la pena
ripetere senza sconti. Su un problema *standard* — equazione nota, geometria
regolare, nessun dato sperimentale da fondere — le differenze finite e gli
elementi finiti, fatti bene, **vincono quasi sempre**: sono più veloci di
ordini di grandezza, più accurati, e portano in dote garanzie di convergenza
che un'ottimizzazione non convessa non potrà mai offrire. Una PINN che impiega
minuti dove un solutore maturo impiega millisecondi, e che ogni tanto fallisce
senza preavviso, non è un progresso: è un passo indietro. Le PINN convengono
dove i classici arrancano — dati e leggi da fondere, geometrie che mandano in
crisi le griglie, problemi inversi — non altrove.

## Oltre le PINN: imparare il mestiere, non il compito

C'è un limite più profondo di tutti quelli visti finora, e superarlo apre la
direzione più promettente. Una PINN risolve **un** problema: una condizione al
contorno, una sorgente, un dominio. Cambia la condizione iniziale e devi
riaddestrare da capo. È come se, per ogni caffè con una temperatura di
partenza diversa, dovessi rifare tutti i calcoli daccapo.

`````{tab} Elementare

Pensa alla differenza tra risolvere *un* esercizio e imparare il *metodo*. Uno
studente che ha risolto un problema di fisica sa la risposta a quel problema;
uno che ha imparato il metodo li risolve tutti, anche quelli che non ha mai
visto, senza rifare la fatica ogni volta. Gli **operatori neurali** fanno la
seconda cosa. Invece di imparare *la soluzione* di un problema, imparano
l'operatore che *mappa le condizioni nella soluzione*: dammi una qualsiasi
temperatura di partenza, una qualsiasi forma del contenitore, e ti restituisco
la curva giusta, subito, senza riaddestrare nulla. Hai imparato il mestiere,
non il singolo compito — ed è riusabile all'infinito.

`````

`````{tab} Superiore

Un operatore neurale approssima una mappa $\mathcal{G}: \mathcal{A} \to
\mathcal{U}$ tra **spazi di funzioni**: l'ingresso non è un vettore ma una
funzione intera (il campo delle condizioni iniziali, dei coefficienti, della
sorgente) e l'uscita è la funzione soluzione. Due architetture hanno segnato
il campo. Il **DeepONet** di Lu, Jin, Pang, Zhang e Karniadakis
{cite}`lu2021learning` poggia sul teorema di approssimazione universale *degli
operatori*: una rete *branch* codifica la funzione d'ingresso campionata su un
insieme di sensori, una rete *trunk* codifica il punto di query, e il loro
prodotto scalare dà il valore della soluzione lì. Il **Fourier Neural
Operator** di Li, Kovachki, Azizzadenesheli, Liu, Bhattacharya, Stuart e
Anandkumar {cite}`li2021fourier` parametrizza il nucleo integrale
direttamente nello spazio di Fourier: ogni strato trasforma, filtra le basse
frequenze con pesi appresi, antitrasforma. Il risultato è *invariante alla
risoluzione* (addestri su una griglia, valuti su un'altra) e su Navier–Stokes
gli autori riportano un'inferenza fino a circa **tre ordini di grandezza** più
rapida di un solutore pseudospettrale — dell'ordine dei millisecondi contro i
secondi, una volta pagato in anticipo il costo dell'addestramento.

`````

Ed eccoci al meteo. I modelli neurali che prevedono il tempo globale in pochi
secondi — là dove i centri di calcolo tradizionali macinano equazioni per ore
su un supercomputer — sono costruiti proprio su operatori appresi, nello
spirito del Fourier Neural Operator, anche se le architetture concrete variano
da modello a modello (reti su grafo, Transformer, varianti spettrali). Una
previsione che prima
richiedeva ore di supercalcolo esce ora in un tempo brevissimo, a parità
sorprendente di qualità sulle scale di alcuni giorni. È un risultato che va
maneggiato con prudenza — restano aperte questioni su eventi estremi,
stabilità a lungo termine, fisica non rispettata alla lettera — ma la
direzione è inequivocabile, e non passa dalla fisica messa nella loss: passa
dall'imparare l'operatore dai dati.

## Congedo: far collaborare conoscenza e dati

Chiudiamo qui il capitolo, e con esso la parte tecnica di questo libro. Le
PINN valgono, alla fine, più come *simbolo* che come tecnica: le migliori fra
quelle che abbiamo incontrato non hanno chiesto di scegliere tra la
conoscenza umana e i dati. Per secoli la scienza ha scritto leggi e le ha
risolte al calcolatore; per quindici anni il machine learning ha fatto
l'opposto, buttando via le leggi e fidandosi solo dei dati. Le PINN, e ancor
più gli operatori neurali, indicano una terza strada: mettere la legge scritta
a mano e il dato misurato **nella stessa funzione di costo**, e lasciare che
si correggano a vicenda. La fisica riempie i vuoti che i dati non coprono; i
dati piegano la fisica dove il modello è incompleto. Non una che sostituisce
l'altra: una collaborazione, scritta in una loss.

È l'ultima delle tante idee che abbiamo montato pezzo per pezzo, dai vettori
dei primi capitoli fino a qui. Nelle Conclusioni proviamo a guardare l'intero
percorso dall'alto — a cercare il disegno che, capitolo per capitolo, era
troppo vicino per vedersi.

```{admonition} Da ricordare
:class: important
- Il vero superpotere delle PINN è il **problema inverso**: stimare grandezze
  non misurabili (la pressione in un aneurisma) da ciò che si misura, imponendo
  la fisica come vincolo {cite}`raissi2020hidden`. Il parametro ignoto diventa
  una variabile addestrabile, stimata *insieme* alla soluzione.
- Applicazioni reali dove il vantaggio è concreto: emodinamica e fluidodinamica,
  identificazione di parametri nei materiali, inversione geofisica. I grandi
  modelli meteo neurali, invece, **non sono PINN**: sono operatori appresi dai
  dati, senza fisica nella loss.
- Limiti onesti {cite}`krishnapriyan2021characterizing`: le PINN falliscono
  anche su PDE semplici; la loss multi-obiettivo è un **tiro alla fune** da
  bilanciare a mano; lo **spectral bias** frena le alte frequenze; le PDE stiff
  e gli orizzonti lunghi restano ostici. Sui problemi standard, i solutori
  classici vincono quasi sempre in velocità e garanzie.
- Gli **operatori neurali** imparano il mestiere, non il compito: la mappa
  condizioni → soluzione, riusabile senza riaddestrare — DeepONet
  {cite}`lu2021learning` e Fourier Neural Operator {cite}`li2021fourier`. È la
  ragione per cui le previsioni meteo neurali escono in secondi anziché in ore.
- La direzione più promettente non sostituisce la conoscenza umana con i dati:
  li fa **collaborare nella stessa funzione di costo**.
```
