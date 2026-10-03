# Physics-Informed Neural Networks

```{image} ../figures/aperture/pinn.png
:class: pt-apertura only-light
:width: 100%
:alt: Un cannocchiale su un treppiede puntato verso il cielo, con un'orbita che finisce su un piccolo pianeta.
```

```{image} ../figures/aperture/pinn-scura.png
:class: pt-apertura only-dark
:width: 100%
:alt: Un cannocchiale su un treppiede puntato verso il cielo, con un'orbita che finisce su un piccolo pianeta.
```

La notte del 23 settembre 1846 l'astronomo Johann Galle punta il telescopio
dell'Osservatorio di Berlino verso un punto preciso del cielo, indicato in una
lettera arrivata quel giorno da Parigi. Il mittente, Urbain Le Verrier, quel
punto non l'ha mai osservato: l'ha *calcolato*, applicando per mesi le leggi
di Newton alle irregolarità dell'orbita di Urano, fino a concludere che a
perturbarla doveva essere un pianeta sconosciuto. Nettuno compare a meno di un
grado dalla posizione prevista. L'espressione rimasta celebre è di François
Arago: Le Verrier ha scoperto un pianeta «sulla punta della penna».

Per quasi due secoli la scienza ha funzionato così: leggi di natura scritte
come equazioni differenziali (regole compatte su come cambiano le cose) e
risolte, a mano finché si è potuto, poi al calcolatore. Previsioni del tempo,
gallerie del vento, reattori nucleari: sotto c'è sempre un'equazione.

Il machine learning visto finora ha lavorato quasi sempre dall'altra parte:
poche ipotesi sul fenomeno, molti dati, e una rete che le regolarità le trova
da sola. Nel {doc}`capitolo sulle serie
temporali </SerieTemporali/forecasting-neurale>` una sola rete ha imparato a
prevedere la domanda di milioni di prodotti, senza che nessuno le spiegasse
perché la gente compra. Funziona quando i dati abbondano e le leggi non le
conosce nessuno: nessuna equazione governa lo spam, e infatti contro lo spam si
è imparato dai dati.

In molti problemi fisici la situazione è rovesciata. Le leggi sono note con
buona approssimazione, e a scarseggiare sono i dati, perché lì misurare costa:
una sonda calata sul fondo del mare, un sensore infilato dentro un'arteria, un
prototipo messo in galleria del vento danno poche misure. E nessuna dà il
valore vero: ogni strumento aggiunge un errore casuale, un po’ sopra o un po’
sotto, e di misure fatte così si dice che sono rumorose. La domanda, allora: e
se si potessero usare *entrambi*?

## Una regola che dice come cambiano le cose

Un'equazione differenziale, si è detto, è una regola su come cambiano le cose.
Il modo in cui lo fa, però, spiazza: non dice mai *quanto vale* la cosa che ci
interessa, dice soltanto *di quanto sta cambiando*. Sembra pochissimo. Invece
basta, e conviene vedere subito perché.

`````{tab} Elementare

Posa una tazza di caffè bollente sulla scrivania. Nessuno sa dire d'un fiato
che temperatura avrà tra dieci minuti, ma la regola la conosciamo tutti: il
caffè si raffredda tanto più in fretta quanto più è caldo della stanza, di
corsa quando scotta, piano da tiepido, e da fermo quando è arrivato alla
temperatura dell'aria intorno.

La regola parla solo del *cambiamento*, eppure da lì si ricostruisce tutta la
storia. Caffè a 80 °C, stanza a 20 °C, e ogni minuto il caffè perde un decimo
della differenza: la differenza è 60, quindi scende di 6 gradi e va a 74 °C; poi
la differenza è 54, perde 5,4 e arriva a 68,6 °C; poi 63,7 °C, e così via, in
una curva ripida all'inizio e sempre più piatta. Un'equazione differenziale è
questo: una regola sul cambiamento che, partendo da una condizione iniziale (80
°C al minuto zero), inchioda tutto il futuro. Senza quel numero di partenza le
curve che obbediscono alla regola sarebbero infinite: l'80 sceglie la nostra.
Che una partenza basti sempre a scegliere una storia sola, però, lo si sa
dimostrare per regole semplici come questa, non per tutte: per certe regole
più difficili, come quelle dei fluidi, è ancora una domanda aperta. Le leggi di
Newton che Le Verrier stava applicando sono del tipo semplice, con la gravità
al posto del caffè.

Quel conto minuto per minuto è il metodo classico: dagli anni Cinquanta i
calcolatori risolvono così le equazioni differenziali, un passettino alla
volta lungo una fitta rete di puntini. La rete di puntini si chiama
**griglia**, e il programma che la macina **solutore**. Per il caffè i puntini
sono istanti, uno ogni minuto; per una sbarra di ferro scaldata a un capo sono
anche punti lungo il ferro, perché lì la temperatura cambia da posto a posto
oltre che da un momento all'altro. Più fitti stanno, più il risultato è
preciso e più conti servono: noi ne abbiamo fatti tre a mano, un calcolatore
ne fa miliardi.

Due cose però gli costano fatica. La griglia va tagliata su misura per la
forma del problema, e tagliarla è un mestiere a sé. E i puntini esplodono
quando la risposta dipende da tante cose insieme: con dieci puntini per ogni
cosa da cui dipende, una ne chiede dieci, due cento, tre mille, e ogni cosa in
più moltiplica il conto per dieci.

La sbarra chiede anche un'altra cosa. Per il caffè bastava sapere da dove si
parte; per la sbarra serve pure sapere che cosa le succede ai due capi,
perché lì il calore entra o esce: una fiamma sotto un'estremità e un blocco di
ghiaccio sull'altra danno due storie diverse. Sono, letteralmente, le
condizioni ai bordi, o al contorno, che è il nome che portano di solito.

`````

`````{tab} Superiore

Una tazza che si raffredda obbedisce alla legge del raffreddamento di Newton,
un’**equazione differenziale ordinaria** (ODE):

$$
\frac{du}{dt} = -k\,\big(u(t) - T_a\big), \qquad u(0) = u_0,
$$

dove $u(t)$ è la temperatura al tempo $t$, $T_a$ quella dell'ambiente, $k>0$
una costante che dipende da tazza e materiale, e la **condizione iniziale**
$u(0)=u_0$ seleziona, tra le infinite soluzioni, quella del nostro caffè. Che
ne selezioni una e una sola lo garantisce il teorema di Cauchy-Lipschitz,
quando il secondo membro è lipschitziano nell'incognita (qui è lineare).
"Ordinaria" perché l'incognita dipende da una sola variabile; con più
variabili indipendenti si parla di **equazione alle derivate parziali** (PDE).
Il capostipite è l'equazione del calore per una sbarra,

$$
\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2},
$$

dove $u(x,t)$ è la temperatura nel punto $x$ al tempo $t$ e $\alpha$ la
diffusività termica; oltre alla condizione iniziale (il profilo a $t=0$) servono
**condizioni al contorno**: cosa accade agli estremi della sbarra. Per le PDE un
teorema generale di esistenza e unicità non c'è: la buona posizione si dimostra
classe per classe, e per le equazioni di Navier–Stokes in tre dimensioni
l'esistenza globale di soluzioni regolari è ancora un problema aperto, uno dei
sette problemi del millennio del Clay Mathematics Institute. Pochissime
equazioni ammettono soluzioni in forma chiusa; per le altre si passa al
calcolatore, nello spirito dei {doc}`richiami di analisi numerica
</Matematica/analisi-numerica>`: là abbiamo accettato numeri a precisione
finita, qui si accetta un continuo fatto a punti (una **griglia** su $x$ e $t$,
con le derivate rimpiazzate da **differenze finite**, rapporti incrementali a
passo piccolo ma non nullo). Più fitta la griglia, migliore l'approssimazione e
più salato il conto: proibitivo quando le variabili indipendenti sono molte,
perché con $n$ nodi per variabile i nodi sono $n^d$ in $d$ variabili, e crescono
esponenzialmente con la dimensione. Su una geometria irregolare il problema è di
altra natura: elementi finiti e volumi finiti la trattano benissimo, ma la
griglia va costruita su misura, ed è un lavoro a sé.

`````

## Una rete come candidata soluzione

L'idea che salda i due mondi è usare una rete neurale $u_\theta$ come soluzione
candidata, e addestrarla su due richieste insieme: che passi vicino alle poche
misure e alle condizioni di partenza, che dicono dove la curva deve passare, e
che rispetti l'equazione, che dice come deve comportarsi *dappertutto*.
Sostituita nell'equazione, la candidata lascia uno scarto fra i due lati (quello
che la regola chiede e quello che la curva fa), il **residuo**, e per calcolarlo
servono soltanto le derivate della rete, cioè le sue pendenze rispetto alle
coordinate.

È il perno di tutto. Per correggere una risposta, di solito, serve avere
sott'occhio quella giusta, e qui la risposta giusta non ce l'ha nessuno: è
proprio quella che stiamo cercando. Una legge però permette di correggere lo
stesso, perché sa dire se una risposta è *sbagliata* anche quando nessuno sa
dire quale sia quella buona. Per questo l'equazione può prendere il posto dei
dati come supervisione.

A rilanciare questa mossa, nel 2019, sono Maziar Raissi, Paris Perdikaris e
George Karniadakis {cite}`raissi2019physics`, e il nome che le danno è quello
nel titolo del capitolo: **Physics-Informed Neural Networks**, PINN, cioè reti
neurali informate dalla fisica. L'idea però è più vecchia di loro:
una rete che si corregge sul residuo di un'equazione compare già nel 1994
{cite}`dissanayake1994neural`, e resta di nicchia per un quarto di secolo perché
le derivate della rete andavano scritte a mano, caso per caso. Diventa
praticabile quando le librerie imparano a calcolarle da sole.

`````{tab} Elementare

Tre misure di termometro, pure un po’ ballerine, e in mezzo il vuoto: da lì
bisogna tirare fuori tutta la curva di raffreddamento del caffè. Una rete
addestrata alla vecchia maniera passerebbe vicino ai tre punti e, nel resto
del grafico, inventerebbe: tra una misura e l'altra potrebbe fare gobbe
assurde, magari un caffè che si riscalda da solo. La PINN aggiunge un
secondo esaminatore. Il primo controlla col righello che la curva passi
vicino alle misure, e che al minuto zero valga gli 80 °C da cui il caffè
parte. Il secondo punta il dito su istanti scelti *a caso*, anche dove
nessuno ha misurato niente, e lì verifica la regola. Quegli istanti si
chiamano **punti di collocazione**.

È un controllo che si fa con la matita. Su uno di quegli istanti il secondo
esaminatore guarda due cose: a che altezza sta la curva lì, e quanto sta
scendendo lì. Se la curva dice 60 °C, allora la differenza con la stanza è 40
gradi, e la regola (un decimo della differenza al minuto) impone
che in quel momento stia scendendo di 4 gradi al minuto. Non uno, non otto:
quattro. Se la curva scende di uno, lo scarto fra quello che fa e quello che
dovrebbe fare vale 3, e la penalità cresce con lui. Quel «quanto sta
scendendo», in un punto di una curva, si chiama pendenza: è la ripidità
della strada misurata sotto i piedi, non su tutta la salita.

Le penalità dei due esaminatori si sommano in un punteggio unico, quella
loss che ci accompagna da inizio libro, e la rete ritocca i suoi pesi, i
numeri interni che l'addestramento regola, per farlo calare. Quanto conta
ciascuno dei due lo decidiamo noi: se il secondo esaminatore urla dieci volte
più forte, la curva si scosta dalle misure pur di non contraddire la regola, e
le tre letture del termometro non contano quasi più. Con le dosi giuste, dove
ci sono dati comanda il righello, dove non ce ne sono comanda la fisica, e la
curva ha molto meno spazio per inventare. Qualche spazio però le resta: il
secondo esaminatore guarda solo gli istanti su cui punta il dito, e in mezzo a
due di quegli istanti la curva è libera di fare una gobba che nessuno vede.

`````

`````{tab} Superiore

Sia $u_\theta(x,t)$ una rete neurale con parametri $\theta$ che riceve in
ingresso le coordinate $(x,t)$ e restituisce il valore della soluzione
candidata in quel punto. Per l'equazione del calore si definisce il residuo
fisico

$$
r_\theta(x,t) = \frac{\partial u_\theta}{\partial t}
- \alpha \frac{\partial^2 u_\theta}{\partial x^2},
$$

che vale zero esattamente dove la rete rispetta l'equazione. La loss somma due
richiami all'ordine:

$$
\mathcal{L}(\theta) =
\underbrace{\frac{1}{N_d} \sum_{i=1}^{N_d}
\big( u_\theta(x_i, t_i) - u_i \big)^2}_{\text{dati}}
\;+\;
\underbrace{\frac{\lambda}{N_c} \sum_{j=1}^{N_c}
r_\theta(x_j, t_j)^2}_{\text{fisica}},
$$

dove $(x_i, t_i, u_i)$ sono le $N_d$ misure disponibili, incluse le
condizioni iniziali e al contorno quando prescrivono il valore di $u$; i
$(x_j, t_j)$ sono $N_c$ **punti di collocazione**
estratti a caso nel dominio (nessuna griglia) e $\lambda$ bilancia i due
termini. Il tocco elegante è il calcolo delle derivate di $u_\theta$ rispetto
agli *ingressi*: le fornisce la differenziazione automatica, cioè la
regola della catena della backpropagation applicata a $x$ e $t$ anziché ai
pesi {cite}`rumelhart1986learning`. In PyTorch è una chiamata a
`torch.autograd.grad` {cite}`paszke2019pytorch`: derivate esatte a meno della
precisione di macchina, senza differenze finite e senza passo di
discretizzazione da scegliere.

Il termine di fisica è una stima Monte Carlo della media di $r_\theta^2$ su
tutto il dominio spazio-temporale, e controlla $r_\theta$ soltanto nei punti di
collocazione: fra l'uno e l'altro la rete resta libera. La forma si estende
senza cambiare procedura. Per un'equazione $\mathcal{N}[u] = 0$ con condizione
al bordo $\mathcal{B}[u] = g$ ($\mathcal{B}$ è l'identità per una condizione di
Dirichlet, la derivata normale per una di Neumann, come un flusso imposto agli
estremi della sbarra) si aggiunge un termine per ogni richiesta, e il residuo
$\mathcal{N}[u_\theta]$ resta una composizione di derivate della rete anche
quando l'operatore non è lineare: per l'equazione di Burgers è $\partial_t
u_\theta + u_\theta\,\partial_x u_\theta - \nu\,\partial_{xx} u_\theta$. Il
coefficiente $\lambda$, infine, non è un numero puro: il residuo dell'equazione
del calore si misura in $[u]/[t]$, uno scarto sui dati in $[u]$, quindi
$\lambda$ ha le dimensioni di $[t]^2$ e il suo valore buono cambia con le unità
scelte. È una delle ragioni per cui le PINN si addestrano di norma
sull'equazione adimensionalizzata, con $u$, $x$ e $t$ riportati a scale
caratteristiche di ordine uno.

`````

## Perché ci interessa

Tre proprietà rendono la ricetta interessante. La prima viene dall'assenza di
una griglia; la seconda e la terza dal fatto che misure e legge stanno nella
stessa loss. In più, alla fine dell'addestramento resta una funzione definita
in ogni punto del dominio, a cui si può chiedere il valore anche dove nessun
punto di collocazione è caduto, e con il valore le derivate. Il valore fra due
nodi, a dire il vero, lo sa dare anche un solutore classico, ricostruendolo per
interpolazione; le derivate esatte in ogni punto sono il contributo della rete.

La prima proprietà è che non c'è nessuna griglia da costruire. I punti di
collocazione si estraggono a caso anche dentro una forma complicata, il
condotto di un'aorta dove si vuole sapere come scorre il sangue, il profilo di
un'ala dove si vuole sapere come si comporta l'aria. La regione in cui si
cerca la soluzione si chiama **dominio**, e un solutore classico, prima di
lavorarci dentro, deve ricoprirla con una griglia fatta su misura per quella
forma. Quelle griglie i solutori classici le sanno costruire, e ricoprono
aorte e ali tutti i giorni; costruirle però è un lavoro lungo e da
specialisti, che la PINN non deve fare. È un vantaggio di comodità, non di
possibilità.

Con molte variabili indipendenti, invece, la griglia smette di essere
praticabile. Per il caffè ce n'è una, il tempo, e i nodi stanno in fila; per
la sbarra due, il tempo e il punto lungo la sbarra, e i nodi riempiono un
rettangolo. Ma ci sono problemi con dieci variabili, e i nodi vanno messi in
tutte le combinazioni: con dieci nodi per variabile ne servono dieci su un
segmento, cento su un quadrato, mille su un cubo e $10^{10}$ con dieci
variabili. Lì servono metodi che la griglia non la usano: la PINN è uno, la
media su cammini casuali della {doc}`sezione sulle equazioni come medie
</PINN/equazioni-come-medie>` ne è un altro. Che poi una rete ci arrivi
davvero non è automatico: quando le variabili sono tante davvero, i risultati
si ottengono con varianti costruite apposta, non con la ricetta base, ed è
ricerca ancora in corso {cite}`hu2024tackling`.

La seconda proprietà è quella dei dati scarsi. Dove il laboratorio arriva
con tre sensori, la legge riempie i vuoti: fra le infinite curve che passano
vicino a quei tre punti, la loss scarta quelle che violano l'equazione nei
punti di collocazione. Se il residuo si annullasse in tutto il dominio
resterebbero le sole soluzioni dell'equazione, e fra quelle le misure
sceglierebbero; con un numero finito di punti di controllo l'esclusione è
soltanto approssimata, e la {doc}`sezione sulla legge dentro la loss
</PINN/come-funziona>` misura quanto. La terza proprietà, la più
sorprendente, è quella dei **problemi inversi**.

`````{tab} Elementare

Il problema *diretto* è quello del caffè: conosco la regola, ricostruisco la
curva. Il problema *inverso* lo ribalta: ho osservato la curva (o qualche suo
punto) e voglio scoprire un pezzo di regola che mi manca. Di notte la casa si
raffredda: dalle temperature segnate ora per ora, quanto isolano i muri? È la
domanda del medico legale (a che ora il decesso, data la temperatura del
corpo?) ed era la domanda di Le Verrier: dai disturbi nell'orbita di Urano,
dov'è il pianeta che non vedo? Anche i metodi classici sanno rispondere, e
bene: per farlo hanno procedure loro, collaudate, che però vanno riscritte da
capo per ogni singola equazione, ed è un lavoro da specialisti. Per una PINN il
pezzo di regola che manca è semplicemente un numero in più da addestrare, dello
stesso tipo dei pesi che la rete regola già da sé: lo si aggiusta finché
fisica e misure non vanno d'accordo. Purché le misure lo dicano: se quella
notte fuori ci fossero stati gli stessi gradi di dentro, la casa non si
sarebbe raffreddata affatto, le letture sarebbero tutte uguali, e su quanto
isolano i muri non direbbero niente, a una PINN come a qualunque altro metodo.

`````

`````{tab} Superiore

Nel problema inverso un parametro dell'equazione (per esempio la diffusività
$\alpha$) è incognito. Lo si promuove a variabile addestrabile e si minimizza
la stessa loss su entrambi,
$\hat{\theta},\hat{\alpha}=\arg\min_{\theta,\alpha}\mathcal{L}(\theta,\alpha)$:
il residuo dipende ora anche da $\alpha$, il cui gradiente arriva dalla stessa
passata di backpropagation, e soluzione e parametro fisico si stimano
*insieme*, anche con misure rumorose e incomplete. Purché il parametro sia
**identificabile** dai dati: se la soluzione osservata è la stessa per più
valori di $\alpha$, la loss ha un'intera famiglia di minimi e la discesa si
ferma dove la porta l'inizializzazione. Il caso limite è un profilo
stazionario lineare, con $\partial_t u = 0$ e $\partial_{xx} u = 0$: il
residuo $\partial_t u - \alpha\,\partial_{xx} u$ si annulla per ogni $\alpha$,
e nessun metodo, classico o neurale, può stimarlo.

Attenzione però a non attribuirsi un vantaggio che non c'è: nemmeno i metodi
classici procedono per tentativi. Da quarant'anni gli inversi vincolati da una
PDE hanno il loro strumento maturo, il **metodo dello stato aggiunto**, che il
gradiente rispetto a tutti i parametri incogniti lo ottiene con una o due
risoluzioni del problema diretto {cite}`plessix2006adjoint`; va però scritto
su misura per ogni equazione, solutore diretto e aggiunto compresi, mentre la
PINN monta sempre lo stesso problema non vincolato in $(\theta, \alpha)$. La
{doc}`sezione su dove la fisica aiuta e dove no </PINN/applicazioni-limiti>`
ci torna sopra, e ci aggiunge un concorrente che costa ancora meno. È comunque
questa naturalezza sui problemi inversi ad aver fatto la fortuna delle PINN
{cite}`raissi2019physics`. Il filone che ne è nato (reti vincolate dalla
fisica, operatori neurali, scoperta di equazioni dai dati) va oggi sotto il
nome di **scientific machine learning** {cite}`karniadakis2021physics`.

`````

## Un'onestà dovuta

Le PINN non mandano in pensione i solutori classici, e il loro territorio è
più stretto di quanto le tre proprietà lascino credere. Sta dove legge e
misure vanno usate insieme per rispondere alla stessa domanda; dove le
variabili in gioco sono troppe perché una griglia stia in piedi; e sui
problemi inversi, non perché i metodi classici non li sappiano fare, ma perché
loro chiedono un programma scritto su misura per quell'equazione, e una PINN
no. Sui problemi diretti, con geometria regolare e senza dati da integrare, i
confronti sistematici con gli elementi finiti non trovano alle PINN un
vantaggio né nel tempo per ottenere la soluzione né nell'accuratezza
{cite}`grossmann2024can`.

`````{tab} Elementare

Su un problema ordinario (regola nota, forma regolare, nessuna misura da
tenere insieme alla legge) il conto a passettini vince, e non di poco: è più
rapido ed è più preciso. E porta con sé una promessa scritta prima di partire:
si sa dimostrare che accorciando i passi lo sbaglio scende, e di quanto. Per i
metodi più usati, dimezzare i passi divide lo sbaglio per quattro, o per
sedici.

La promessa ha le sue condizioni, e chi fa il conto le conosce in anticipo. La
più stretta lega fra loro i due passi della sbarra: se si dimezzano i
passettini lungo il ferro, quelli nel tempo vanno divisi per quattro. Chi non
lo fa vede uno sbaglio minuscolo crescere da solo. Un puntino con un grado di
troppo, al passo dopo ne ha uno di meno, e i suoi due vicini uno di troppo;
al passo dopo ancora ne ha tre di troppo, e i vicini due di meno. Lo sbaglio
cambia segno da un puntino all'altro e cresce a ogni passo, finché il conto
sputa temperature che nessun termometro vedrà mai. E se la storia da
ricostruire fa un salto netto invece di scorrere liscia, accorciare i passi
aiuta molto meno: lo sbaglio scende ancora, ma a ogni dimezzamento si riduce
appena.

Di una rete addestrata non si sa dire niente del genere: l'addestramento
finisce quando smette di migliorare, e nessuno può garantire quanto lontana
sia rimasta dalla risposta. Una PINN può metterci minuti dove il metodo di
sempre impiega millisecondi, e ogni tanto sbaglia senza che nulla lo segnali.

`````

`````{tab} Superiore

Su un problema standard (equazione nota, geometria regolare, nessun dato da
integrare) differenze finite ed elementi finiti restano più veloci, più
accurati e con garanzie di convergenza che un'ottimizzazione non convessa non
può offrire: per un metodo classico si dimostra un ordine di convergenza,
l'errore scende come $O(h^p)$ al raffinarsi del passo $h$. Il teorema ha le
sue ipotesi, e senza di quelle la garanzia non vale: lo schema dev'essere
consistente e stabile (il teorema di equivalenza di Lax), e la soluzione
esatta abbastanza regolare, perché su un urto l'ordine effettivo crolla a uno
o meno, qualunque sia $p$. La stabilità ha spesso una forma esplicita: lo schema
esplicito alle differenze centrate per il calore,
$u_i^{n+1} = u_i^n + \rho\,(u_{i+1}^n - 2u_i^n + u_{i-1}^n)$ con
$\rho = \alpha\,\Delta t/\Delta x^2$, è stabile solo per $\rho \le 1/2$.
Infittire in spazio senza ridurre $\Delta t$ in proporzione a $\Delta x^2$
porta $\rho$ oltre la soglia, e il modo a scacchiera, che a ogni passo è
moltiplicato per $1 - 4\rho < -1$, esplode. Ma sono
ipotesi verificabili in anticipo, mentre la discesa del gradiente su una loss
non convessa si ferma in un minimo locale qualsiasi e non promette nulla. Una
PINN può richiedere minuti di addestramento dove un
solutore maturo impiega millisecondi, e a volte fallisce senza preavviso
{cite}`karniadakis2021physics`.

`````

## La legge dentro la loss, e i suoi limiti

Dalla cornice al banco di lavoro. Nella prossima sezione costruiremo il metodo
per intero, con una PINN scritta in PyTorch, e il banco di prova sarà una molla
che oscilla e si smorza: la rete, la misura di quanto viola la regola, il
punteggio che mette insieme i due controlli, e infine il problema inverso, con
un pezzo di legge che fingeremo di non conoscere. Poi, per le equazioni che
diffondono, una seconda lettura che aggira la griglia: la soluzione come media
su cammini casuali. Chiuderemo con le applicazioni (il sangue nelle arterie, i
materiali), con una mappa dei limiti (quando le PINN convengono, quando no) e
con gli operatori neurali, reti che imparano il procedimento invece di un
singolo problema, la stessa idea che, appresa dai soli dati, fa girare i
modelli meteo neurali.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un’equazione differenziale non dice quanto vale una grandezza ma come
  cambia (il caffè si raffredda tanto più in fretta quanto più è caldo della
  stanza). Sapendo da dove si parte, e cosa succede ai bordi quando conta
  anche lo spazio (ai due capi della sbarra che si scalda), la storia è
  determinata tutta. Vale per le regole semplici che incontreremo qui, il
  caffè e la molla; per certe regole più difficili, come quelle dei fluidi, è
  ancora una domanda aperta.
- I metodi classici spezzettano il problema: riempiono il dominio di una
  fitta rete di puntini e avanzano da un puntino all'altro. Sono accurati e
  velocissimi, e se la cavano anche con forme complicate; ma quella rete va
  costruita su misura, e quando le grandezze in gioco sono molte il numero di
  puntini esplode.
- Una PINN usa una rete neurale come curva candidata e la corregge con
  due esaminatori: il righello, che la tiene vicina alle (poche) misure e al
  punto da cui si parte, e il controllo della regola in punti scelti a caso,
  i punti di collocazione,
  dove ogni violazione costa punti {cite}`raissi2019physics`. Funziona perché
  una legge sa dire che una risposta è sbagliata anche quando nessuno sa
  dire quale sia quella giusta; fra un punto di controllo e l'altro, però, la
  curva resta un po’ libera. Le pendenze che servono a controllarla gliele
  darà il meccanismo con cui la rete già si addestra, quello che a ogni giro le
  dice di quanto ritoccare ciascun peso.
- Punti di forza: dati scarsi ma legge nota, molte grandezze in gioco, domini
  dalla forma complicata senza una griglia da costruire, e i problemi
  inversi (il pezzo di regola che manca diventa un numero in più che
  l'addestramento regola da sé, purché le misure ne portino traccia).
- Onestà: sui problemi standard i metodi classici restano superiori, più
  rapidi e più precisi, e con una garanzia scritta prima di partire; le PINN
  si affiancano, non li sostituiscono {cite}`karniadakis2021physics`.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Un’equazione differenziale non dice quanto vale una grandezza ma come
  cambia; con condizioni iniziali e al contorno questo basta a determinarla
  (ODE: una variabile indipendente; PDE: più di una). Vale per le equazioni
  ben educate che risolveremo qui, il caffè e la molla (Cauchy-Lipschitz); per
  certe equazioni difficili, come Navier–Stokes in tre dimensioni, l'esistenza
  globale di soluzioni regolari è ancora aperta.
- I solutori classici discretizzano: differenze finite su griglie regolari,
  elementi e volumi finiti su maglie che seguono anche forme complicate
  (accurati e veloci, con un ordine di convergenza $O(h^p)$ dimostrato sotto
  consistenza, stabilità e regolarità; ma la griglia va costruita su misura, e
  con molte variabili i nodi sono $n^d$).
- Una PINN usa una rete $u_\theta(x,t)$ come candidata soluzione, con una
  loss doppia: aderenza ai (pochi) dati più penalità sul residuo fisico
  $r_\theta = \partial_t u_\theta - \alpha\,\partial_{xx} u_\theta$ nei punti
  di collocazione {cite}`raissi2019physics`; derivate dalla
  differenziazione automatica, esatte e senza griglia. Il termine di fisica è
  una stima Monte Carlo della media di $r_\theta^2$ sul dominio: fra i punti di
  collocazione la rete resta libera.
- Punti di forza: dati scarsi ma leggi note, molte variabili, domini dalla
  forma complicata senza dover costruire una griglia, problemi inversi
  (il parametro ignoto diventa una variabile addestrabile, purché sia
  identificabile dai dati).
- Onestà: sui problemi standard i solutori classici restano superiori; le
  PINN sono un complemento, non un rimpiazzo {cite}`karniadakis2021physics`.
```

`````
