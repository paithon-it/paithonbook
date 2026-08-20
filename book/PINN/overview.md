# Physics-Informed Neural Networks

La notte del 23 settembre 1846 l'astronomo Johann Galle punta il telescopio
dell'Osservatorio di Berlino verso un punto preciso del cielo, indicato in una
lettera arrivata quel giorno da Parigi. Il mittente, Urbain Le Verrier, quel
punto non l'ha mai osservato: l'ha *calcolato*, applicando per mesi le leggi
di Newton alle irregolarità dell'orbita di Urano, fino a concludere che a
perturbarla doveva essere un pianeta sconosciuto. Nettuno compare a meno di un
grado dalla posizione prevista. L'espressione rimasta celebre è di François
Arago: Le Verrier ha scoperto un pianeta «sulla punta della penna».

Per quasi due secoli la scienza ha funzionato così: leggi di natura scritte
come **equazioni differenziali** (regole compatte su come cambiano le cose) e
risolte, a mano finché si è potuto, poi al calcolatore. Previsioni del tempo,
gallerie del vento, reattori nucleari: sotto c'è sempre un'equazione.

Il machine learning di questo libro ha fatto finora l'esatto contrario: niente
leggi, solo dati, e una rete che scova le regolarità da sola. Ottimo quando i
dati abbondano e le leggi non le conosce nessuno: nessuna equazione governa lo
spam, e infatti contro lo spam si è imparato dai dati.

Nelle scienze fisiche è tutto rovesciato. Le leggi le conosciamo con
precisione ammirevole, e a scarseggiare sono i dati, perché lì misurare costa:
una sonda calata sul fondo del mare, un sensore infilato dentro un'arteria, un
prototipo messo in galleria del vento sono misure che si contano sulle dita.
E nessuna dà il valore vero: lo strumento sbaglia sempre un pochino, un po’
sopra o un po’ sotto, e di misure fatte così si dice che sono **rumorose**. La
domanda di questo capitolo, allora: e se si potessero usare *entrambi*?

## Una regola che dice come cambiano le cose

Un'equazione differenziale, si è detto, è una regola su come cambiano le cose.
Il modo in cui lo fa, però, spiazza: non dice mai *quanto vale* la cosa che ci
interessa, dice soltanto *di quanto sta cambiando*. Sembra pochissimo. Invece
basta, e conviene vedere subito perché.

`````{tab} Elementare

Posa una tazza di caffè bollente sulla scrivania. Nessuno sa scrivere d'un
fiato la sua temperatura tra dieci minuti, ma tutti conosciamo una regola più
semplice: **il caffè si raffredda tanto più in fretta quanto più è caldo della
stanza** (di corsa quando scotta, piano da tiepido, e da fermo quando è
arrivato alla temperatura dell'aria intorno).

La regola parla solo del *cambiamento*, eppure da lì si ricostruisce tutta la
storia. Caffè a 80 °C, stanza a 20 °C; diciamo che ogni minuto il caffè perde
un decimo della differenza con la stanza: al primo minuto la differenza è 60,
quindi perde 6 gradi e scende a 74 °C; poi la differenza è 54, perde 5,4 e va
a 68,6 °C; poi 63,7 °C, e così via fino alla curva completa (ripida
all'inizio, sempre più piatta). Un'equazione differenziale è questo: una
regola sul cambiamento che, partendo da una condizione iniziale (80 °C al
minuto zero), inchioda tutto il futuro. Le leggi di Newton che Le Verrier
stava applicando sono regole dello stesso tipo, con la gravità al posto del
caffè.

E vale la pena fermarsi su quel conto minuto per minuto, perché non è un
esempio qualsiasi: **quello che abbiamo appena fatto a mano è il metodo
classico**. Da settant'anni i calcolatori risolvono le equazioni differenziali
così, avanzando un passettino alla volta lungo una fitta rete di puntini stesa
su quello che ci interessa. Quella rete di puntini si chiama **griglia**. Per
il caffè i puntini sono istanti, uno ogni minuto. Per una sbarra di ferro
scaldata a un capo sarebbero anche punti lungo la sbarra, perché lì la
temperatura cambia da posto a posto oltre che da un momento all'altro. Più i
passettini sono piccoli, più il risultato è preciso e più conti servono; noi
ne abbiamo fatti tre a mano, un calcolatore ne fa miliardi.

Quando in questo capitolo si parla di **metodi classici**, o di **solutori**
classici (un solutore è il programma che risolve l'equazione), si parla di
lui: è maturo, accurato, velocissimo, ed è tuttora quello che si usa quasi
sempre. Due cose però gli costano fatica, e sono esattamente quelle su cui
gioca il metodo di questo capitolo. La prima: la rete di puntini va costruita
su misura per la forma del problema. La seconda: il numero di puntini esplode
quando la risposta che cerchiamo dipende da tante cose insieme. Per il caffè
dipende da una sola, il tempo, e i puntini stanno in fila; per la sbarra da
due, il tempo e il punto lungo la sbarra, e già bisogna riempire un
rettangolo.

Un'ultima cosa sulla sbarra, che tornerà utile. Per il caffè bastava sapere da
dove si parte; per la sbarra bisogna sapere anche che cosa le succede **ai due
capi**, perché lì il calore entra o esce e da lì dipende tutto il resto: una
fiamma sotto un'estremità e un blocco di ghiaccio sull'altra danno due storie
completamente diverse. Sono, letteralmente, le condizioni ai bordi.

`````

`````{tab} Superiore

La regola del caffè è la legge del raffreddamento di Newton, un’**equazione
differenziale ordinaria** (ODE):

$$
\frac{du}{dt} = -k\,\big(u(t) - T_a\big), \qquad u(0) = u_0,
$$

dove $u(t)$ è la temperatura al tempo $t$, $T_a$ quella dell'ambiente, $k>0$
una costante che dipende da tazza e materiale, e la **condizione iniziale**
$u(0)=u_0$ seleziona, tra le infinite soluzioni, quella del nostro caffè.
"Ordinaria" perché l'incognita dipende da una sola variabile; con più
variabili indipendenti si parla di **equazione alle derivate parziali** (PDE).
Il capostipite è l'equazione del calore per una sbarra,

$$
\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2},
$$

dove $u(x,t)$ è la temperatura nel punto $x$ al tempo $t$ e $\alpha$ la
diffusività termica; oltre alla condizione iniziale (il profilo a $t=0$)
servono **condizioni al contorno**: cosa accade agli estremi della sbarra.
Pochissime equazioni ammettono soluzioni in forma chiusa; per le altre si
passa al calcolatore, nello spirito dei richiami di analisi numerica: là
abbiamo accettato numeri a precisione finita, qui si accetta un continuo fatto
a punti (una **griglia** su $x$ e $t$, con le derivate rimpiazzate da
**differenze finite**, rapporti incrementali a passo piccolo ma non nullo).
Più fitta la griglia, migliore l'approssimazione e più salato il conto:
proibitivo quando le variabili indipendenti sono molte, perché il numero di
nodi cresce esponenzialmente con la dimensione. Su una geometria irregolare
il problema è di altra natura: elementi finiti e volumi finiti la trattano
benissimo, ma la griglia va costruita su misura, ed è un lavoro a sé.

`````

## Una rete come candidata soluzione

L'idea che salda i due mondi sta in una frase. Una rete neurale si propone
come risposta: disegna una curva e dice «la soluzione dell'equazione è
questa». Poi viene corretta, e riprovata, e corretta ancora, finché quella
curva non è davvero la risposta. A correggerla sono due cose insieme: le poche
misure, che dicono dove la curva deve passare, e la legge, che dice come deve
comportarsi *dappertutto*.

Vale la pena fermarsi un secondo su come sia possibile, perché è il perno di
tutto il capitolo. Per correggere un compito, di solito, serve avere
sott'occhio la risposta giusta. Qui la risposta giusta non ce l'ha nessuno: è
proprio quella che stiamo cercando. Una legge però permette di correggere lo
stesso, perché sa dire se una risposta è *sbagliata* anche quando nessuno sa
dire quale sia quella buona. Ecco perché un'equazione può fare da maestro a
una rete.

A rilanciare questa mossa, nel 2019, sono Maziar Raissi, Paris Perdikaris e
George Karniadakis {cite}`raissi2019physics`, e il nome che le danno è quello
in cima al capitolo: **Physics-Informed Neural Networks**, PINN, cioè reti
neurali informate dalla fisica. L'idea però è più vecchia di loro, e la
prossima sezione racconta da dove viene e perché ha dovuto aspettare
vent'anni; è da qui che diventa praticabile.

`````{tab} Elementare

Immagina di dover disegnare la curva di raffreddamento del caffè avendo solo
tre misure di termometro, pure un po’ ballerine. Una rete addestrata alla
vecchia maniera passerebbe vicino ai tre punti e, nel resto del grafico,
inventerebbe: tra una misura e l'altra potrebbe fare gobbe assurde, magari un
caffè che si riscalda da solo. La PINN aggiunge un secondo esaminatore. Il
primo, classico, controlla col righello che la curva passi vicino alle misure.
Il secondo punta il dito su istanti scelti *a caso*, anche dove nessuno ha
misurato niente, e lì verifica la regola. Quegli istanti scelti a caso si
chiamano **punti di collocazione**.

Vediamolo fare, perché è un controllo che si fa con la matita. Il secondo
esaminatore punta il dito su un istante qualsiasi e guarda due cose: a che
altezza sta la curva lì, e quanto sta scendendo lì. Se la curva dice 60 °C,
allora la differenza con la stanza è 40 gradi, e la regola (un decimo della
differenza al minuto) impone che in quel momento stia scendendo di 4 gradi al
minuto. Non uno, non otto: quattro. Se la curva scende di uno, lo scarto fra
quello che fa e quello che dovrebbe fare vale 3, e la penalità cresce con lui.
Quel «quanto sta scendendo», in un punto di una curva, si chiama **pendenza**:
è la ripidità della strada misurata sotto i piedi, non su tutta la salita.

Le penalità dei due esaminatori si sommano in un punteggio unico, quella loss
che ci accompagna da inizio libro, e la rete aggiusta le sue manopole interne
(i **pesi**) per farlo calare. Dove ci sono dati comanda il righello, dove non
ce ne sono comanda la fisica, e la curva non può più inventare.

`````

`````{tab} Superiore

Sia $u_\theta(x,t)$ una rete neurale con parametri $\theta$ che riceve in
ingresso le coordinate $(x,t)$ e restituisce il valore della soluzione
candidata in quel punto. Per l'equazione del calore si definisce il **residuo
fisico**

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
condizioni iniziali e al contorno quando prescrivono il valore di $u$
(quelle sulle derivate, come un flusso imposto agli estremi della sbarra,
entrano con un termine analogo costruito via differenziazione automatica,
come faremo per $u'(0)$ nella prossima sezione); i $(x_j, t_j)$ sono $N_c$
**punti di collocazione**
estratti a caso nel dominio (nessuna griglia) e $\lambda$ bilancia i due
termini. Il tocco elegante è il calcolo delle derivate di $u_\theta$ rispetto
agli *ingressi*: le fornisce la **differenziazione automatica**, cioè la
regola della catena della backpropagation applicata a $x$ e $t$ anziché ai
pesi {cite}`rumelhart1986learning`. In PyTorch è una chiamata a
`torch.autograd.grad` {cite}`paszke2019pytorch`: derivate esatte a meno della
precisione di macchina, senza differenze finite e senza passo di
discretizzazione da scegliere.

`````

## Perché ci interessa

Tre proprietà rendono la ricetta interessante, e tutte e tre nascono da una
cosa sola: **la griglia non c'è più**. Alla fine di un conto classico resta
una tabella, i valori calcolati sui puntini e nient'altro; alla fine di un
addestramento resta la rete, e a una rete si può chiedere il valore in
qualunque punto, anche in mezzo a due puntini, anche dove nessun puntino c'era.

La prima proprietà è la più diretta: **non c'è nessuna griglia da costruire**.
I punti di collocazione si spargono a pioggia anche dentro una forma
complicata, il condotto di un'aorta dove si vuole sapere come scorre il
sangue, il profilo di un'ala dove si vuole sapere come si comporta l'aria. La
regione in cui si cerca la soluzione ha un nome, si chiama **dominio**, e un
solutore classico prima di lavorarci dentro deve ricoprirla tutta di puntini,
con una griglia fatta su misura per quella forma. Attenzione a non prenderla
per più di quello che è: quelle griglie i solutori classici le sanno
costruire, e ricoprono aorte e ali tutti i giorni. Costruirle però è un lavoro
lungo e da specialisti, e la PINN se lo risparmia. È un vantaggio vero, ma di
comodità, non di possibilità.

Diverso è il caso in cui la risposta dipende da molte grandezze insieme, e qui
si passa dal risparmio all'impossibilità. Per il caffè ne basta una, il tempo,
e i puntini stanno in fila; per la sbarra ne servono due, il tempo e il punto
lungo la sbarra, e i puntini riempiono un rettangolo. Ma ci sono problemi in
cui le grandezze da cui la risposta dipende sono dieci, e i puntini vanno
messi in tutte le combinazioni possibili. Il conto è spietato e si fa in un
rigo: dieci puntini bastano a coprire un segmento, per un quadrato ne servono
cento, per un cubo mille, e per dieci grandezze un uno seguito da dieci zeri.
Lì una griglia non è costosa: è impossibile, e una strada che non ha bisogno
di griglia è l'unica che resta aperta. Con un'avvertenza onesta: che poi una
rete ci arrivi davvero non è automatico. Quando le grandezze sono tante
davvero i risultati si ottengono con varianti costruite apposta, non con la
ricetta base di questo capitolo, ed è ricerca ancora in corso
{cite}`hu2024tackling`.

La seconda proprietà è quella dei **dati scarsi**. Dove il laboratorio arriva
con tre sensori, la legge riempie i vuoti: fra le infinite curve che passano
vicino a quei tre punti restano solo quelle che l'equazione ammette, e sono
pochissime. La terza, la più sorprendente, è quella dei **problemi inversi**.

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
pezzo di regola che manca è semplicemente una manopola in più da addestrare,
dello stesso tipo di quelle che la rete gira già da sé: la si aggiusta finché
fisica e misure non vanno d'accordo.

`````

`````{tab} Superiore

Nel problema inverso un parametro dell'equazione (per esempio la diffusività
$\alpha$) è incognito. Basta promuoverlo a variabile addestrabile e
minimizzare la stessa loss su entrambi,
$\hat{\theta},\hat{\alpha}=\arg\min_{\theta,\alpha}\mathcal{L}(\theta,\alpha)$:
il residuo dipende ora anche da $\alpha$, il cui gradiente arriva dalla stessa
passata di backpropagation: soluzione e parametro fisico si stimano *insieme*,
anche con misure rumorose e incomplete.

Attenzione però a non attribuirsi un vantaggio che non c'è: nemmeno i metodi
classici procedono per tentativi. Da quarant'anni gli inversi vincolati da una
PDE si affrontano con il **metodo dello stato aggiunto**, che ricava il
gradiente della funzione di scarto rispetto a *tutti* i parametri incogniti al
costo di una o due risoluzioni del problema diretto, indipendentemente da
quanti siano, e poi scende esattamente come fa una rete
{cite}`plessix2006adjoint`. La differenza è un'altra, ed è di uniformità: lo
stato aggiunto chiede di scrivere su misura, per quell'equazione, sia il
solutore diretto sia quello aggiunto; la PINN risolve un unico problema non
vincolato in $(\theta, \alpha)$, assorbe misure sparse e rumorose senza
cambiare impianto e non discretizza né la geometria né l'operatore aggiunto. È
soprattutto questa naturalezza sui problemi inversi ad aver fatto la fortuna
delle PINN {cite}`raissi2019physics`. Il filone che ne è nato (reti vincolate
dalla fisica, operatori neurali, scoperta di equazioni dai dati) va oggi sotto
il nome di **scientific machine learning** {cite}`karniadakis2021physics`.

`````

## Un'onestà dovuta

Chiariamolo prima di innamorarcene: le PINN **non mandano in pensione i
solutori classici**, e il loro territorio è più stretto di quanto le pagine
precedenti lascino credere. Sta dove legge e misure vanno usate
insieme per rispondere alla stessa domanda; dove le grandezze in gioco sono
troppe perché una griglia stia in piedi; e sui problemi inversi, non perché i
metodi classici non li sappiano fare, ma perché loro chiedono un programma
scritto su misura per quell'equazione, e una PINN no. Altrove niente.

`````{tab} Elementare

Su un problema ordinario (regola nota, forma regolare, nessuna misura da
tenere insieme alla legge) il conto a passettini vince, e non di poco: è più
rapido ed è più preciso. E c'è di più: di quel conto si sa **dimostrare** al
massimo quanto può sbagliare, e che accorciando i passi sbaglia meno. È una
garanzia scritta prima di partire, non un risultato osservato dopo. Di una
rete addestrata questo non si sa dire: l'addestramento finisce quando smette
di migliorare, e nessuno può garantire quanto lontana sia rimasta dalla
risposta. Una PINN può metterci minuti dove il metodo di sempre impiega
millisecondi, e ogni tanto sbaglia senza che nulla lo segnali.

`````

`````{tab} Superiore

Su un problema standard (equazione nota, geometria regolare, nessun dato da
integrare) differenze finite ed elementi finiti restano più veloci, più
accurati e con garanzie di convergenza che un'ottimizzazione non convessa non
può offrire: per un metodo classico si dimostra un ordine di convergenza,
l'errore scende come $O(h^p)$ al raffinarsi del passo $h$. Il teorema ha le
sue ipotesi, e senza di quelle la garanzia non vale: lo schema dev'essere
consistente **e stabile** (per uno schema esplicito sul
calore, infittire solo in spazio fa divergere), e la soluzione esatta
abbastanza regolare, perché su un urto l'ordine $p$ crolla comunque. Ma sono
ipotesi verificabili in anticipo, mentre la discesa del gradiente su una loss
non convessa si ferma in un minimo locale qualsiasi e non promette nulla. Una
PINN può richiedere minuti di addestramento dove un
solutore maturo impiega millisecondi, e a volte fallisce senza preavviso
{cite}`karniadakis2021physics`.

`````

## La legge dentro la loss, e i suoi limiti

Dalla cornice al banco di lavoro. Nella prossima sezione costruiremo il metodo
per intero, con una PINN scritta in PyTorch, e il banco di prova sarà una
**molla che oscilla e si smorza**: la rete, la misura di quanto viola la
regola, il punteggio che mette insieme i due controlli, e infine il problema
inverso, con un pezzo di legge che fingeremo di non conoscere. Chiuderemo con
le applicazioni reali (il sangue nelle arterie, i materiali, il clima) e una
mappa onesta dei limiti: quando convengono, quando no.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un’**equazione differenziale** non dice quanto vale una grandezza ma **come
  cambia** (il caffè si raffredda tanto più in fretta quanto più è caldo della
  stanza). Sapendo da dove si parte, e cosa succede ai bordi quando conta
  anche lo spazio (ai due capi della sbarra che si scalda), la storia è
  determinata tutta. Vale per le equazioni ben educate che incontreremo qui,
  il caffè e la molla che ci aspetta nella prossima sezione. Non vale sempre:
  per le equazioni che descrivono un fluido che scorre nello spazio nessuno è
  ancora riuscito a dimostrare che, dato un inizio, la storia che ne segue sia
  una sola. È uno dei problemi aperti della matematica, e i nostri due esempi
  stanno lontani da lì.
- I metodi classici **spezzettano** il problema: riempiono il dominio di una
  fitta rete di puntini e avanzano da un puntino all'altro. Sono accurati e
  velocissimi, e se la cavano anche con forme complicate; ma quella rete va
  costruita su misura, e quando le grandezze in gioco sono molte il numero di
  puntini esplode.
- Una **PINN** usa una rete neurale come curva candidata e la corregge con
  due esaminatori: il righello, che la tiene vicina alle (poche) misure, e il
  controllo della regola in punti scelti a caso, i **punti di collocazione**,
  dove ogni violazione costa punti {cite}`raissi2019physics`. Funziona perché
  una legge sa dire che una risposta è **sbagliata** anche quando nessuno sa
  dire quale sia quella giusta. Le pendenze che servono a controllarla
  gliele darà il meccanismo con cui la rete già si addestra, quello che a ogni
  giro le dice di quanto ritoccare ciascun peso: è il colpo di scena della
  prossima sezione.
- Punti di forza: dati scarsi ma legge nota, molte grandezze in gioco, domini
  dalla forma complicata senza una griglia da costruire, e i **problemi
  inversi** (il pezzo di regola che manca diventa una manopola che
  l'addestramento regola da sé).
- Onestà: sui problemi standard i metodi classici restano superiori; le PINN
  si affiancano, non li sostituiscono {cite}`karniadakis2021physics`.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Un’**equazione differenziale** non dice quanto vale una grandezza ma **come
  cambia**; con condizioni iniziali e al contorno questo basta a determinarla
  (ODE: una variabile indipendente; PDE: più di una). Vale per le equazioni
  ben educate che risolveremo qui, il caffè e la molla; per certe equazioni
  difficili, come quelle dei fluidi in tre dimensioni, che una soluzione
  unica esista sempre nessuno l'ha ancora dimostrato.
- I solutori classici **discretizzano**: differenze finite o elementi finiti
  su una griglia (accurati, veloci e a loro agio anche su forme complicate;
  ma la griglia va costruita su misura, e con molte variabili il conto
  diventa proibitivo).
- Una **PINN** usa una rete $u_\theta(x,t)$ come candidata soluzione, con una
  loss doppia: aderenza ai (pochi) dati più penalità sul **residuo fisico**
  $r_\theta = \partial_t u_\theta - \alpha\,\partial_{xx} u_\theta$ nei punti
  di collocazione {cite}`raissi2019physics`; derivate dalla
  **differenziazione automatica**, esatte e senza griglia.
- Punti di forza: dati scarsi ma leggi note, molte variabili, domini dalla
  forma complicata senza dover costruire una griglia, **problemi inversi**
  (il parametro ignoto diventa una variabile addestrabile).
- Onestà: sui problemi standard i solutori classici restano superiori; le
  PINN sono un complemento, non un rimpiazzo {cite}`karniadakis2021physics`.
```

`````
