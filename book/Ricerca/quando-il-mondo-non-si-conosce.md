---
titolo_ricerca: "Ricerca e pianificazione quando il mondo non si conosce"
descrizione: "Le tre ipotesi nascoste della ricerca classica: una valutazione
  scrivibile, un test di arrivo e un modello del mondo, e che cosa accade se ne
  manca una."
---
# Le tre cose date per scontate

La ricerca vista finora poggia su tre ipotesi che nessuno ha mai scritto,
perché sembravano ovvie. Due stavano già nella {doc}`definizione di problema di
ricerca </Ricerca/overview>`; la terza è arrivata con l’euristica di A\* e con
la valutazione dei giochi. Scriverle è il primo passo per toglierle una alla
volta.

La prima è che la funzione di transizione si possa interrogare quante volte si
vuole: prima di muovere, il programma prova mille mosse e per
ciascuna sa dove porta, a costo nullo e senza conseguenze. Sposta la torre,
guarda, e la rimette dov’era. La seconda è che esista un test di arrivo, cioè
un controllo che dice sì o no, come le tessere in ordine, il re sotto scacco
matto o l’incrocio giusto.

La terza è che si sappia scrivere una valutazione, cioè dare a una posizione di
mezzo un numero sensato, anche se non esatto. Nei giochi quel numero dice chi
sta meglio e si chiama funzione di valutazione; nella ricerca di un percorso
dice quanto manca alla meta e si chiama euristica. [Sono la stessa idea con due
usi: un giudizio approssimato al posto di un calcolo esatto]{.pt-matita}.

Nel mondo vero ciascuna delle tre può mancare. Le togliamo nell’ordine in cui
cadono più facilmente: prima la valutazione, poi il test di arrivo, per ultima
la funzione di transizione, che fa cadere tutto il resto e porta
all’apprendimento per rinforzo.

## Quando la valutazione non si sa scrivere

La valutazione è la prima a cadere, ed è caduta su un gioco preciso.

Agli scacchi la valutazione si sa scrivere a mano: quanti pezzi ho, quanto vale
ciascuno, se il re è al riparo, come stanno i pedoni, sommati ciascuno con il
suo peso. Funziona: nel 1997 Deep Blue, un computer che univa una valutazione
di questo tipo alla ricerca alfa-beta a più di cento milioni di posizioni al
secondo, vinse per tre punti e mezzo a due e mezzo un incontro di sei partite
con Garri Kasparov, allora il giocatore più forte del mondo. Il punteggio
corrisponde a due vittorie contro una e tre patte, che valgono mezzo punto a
testa {cite}`russell2020artificial`.

Il Go è un altro gioco, poco diffuso in Italia. Si gioca su una griglia grande,
diciannove righe per diciannove, e i pezzi sono pietre tutte uguali, bianche e
nere, che si posano sugli incroci e da lì non si muovono più. Non ci sono re,
torri o alfieri, ma solo il disegno che le pietre formano, e vince chi alla
fine ha circondato più territorio.

Qui la valutazione non si sa scrivere. Non c’è materiale da contare, perché le
pietre sono tutte uguali e restano ferme; quanto valga una pietra dipende da
come stanno le sue vicine e le vicine delle vicine, cioè da un disegno che può
occupare mezza griglia. I giocatori riconoscono una posizione forte a colpo
d’occhio, ma nessuno è riuscito a dare ai programmi una regola scritta a mano
che la riconosca, nonostante decenni di tentativi.

[Senza la valutazione, quindi, tutta la macchina di minimax e potatura si
blocca]{.pt-matita}. Si guarda avanti fino al punto in cui bisogna fermarsi, e
lì non c’è niente da leggere.

La risposta che ha sbloccato il Go si chiama **ricerca ad albero Monte Carlo**
(*Monte Carlo tree search*, abbreviata in MCTS), e al posto della valutazione
scritta a mano mette una stima ottenuta per campionamento.

`````{tab} Elementare

Non sai giudicare una posizione? Non giudicarla: da lì, gioca la partita fino
in fondo tirando le mosse a caso, e guarda come finisce. Poi rifallo, mille
volte. Alla fine non hai un giudizio, hai un conteggio: da questa posizione,
tirando a caso, ho vinto (mettiamo) seicentotrenta volte su mille.

A caso, ma con un divieto: non si posa una pietra dentro un buco circondato
tutto da pietre proprie. Nessun giocatore lo farebbe, perché
riempirebbe con le proprie mani il territorio che stava tenendo. Senza quel
divieto le partite a caso rischiano di non finire più, e mille partite che non
finiscono non contano niente.

Detta così sembra assurdo, perché nessuna di quelle mille partite somiglia a
una partita vera: sono mosse a caso, giocate malissimo da tutti e due. Eppure,
se una posizione è buona per me, tende a restare buona anche in un mondo in cui
giochiamo tutti a caso, perché il vantaggio non dipende dalla mia bravura.

Il conteggio, quindi, non dice come andrebbe la partita vera ma quanto la
posizione è *comoda*, e nel Go, per scegliere una mossa, è già un buon punto di
partenza. Nei programmi veri, anzi, si tira a caso ma non troppo: si pescano
più spesso le mosse che un giocatore farebbe.

Le partite da giocare non sono infinite, e servono a scegliere fra le mosse
possibili adesso: mille per ciascuna sarebbero troppe. Se ne dà allora
qualcuna a tutte e si guarda quali rendono, e le prove successive vanno
soprattutto lì. Ne resta però qualcuna per le mosse provate poco, perché tre
partite andate male possono essere sfortuna.

La stessa regola decide ogni volta dove va la partita successiva. Per ogni
mossa si prende quanto ha reso finora e ci si aggiunge un premio, grande per le
mosse provate poco e sempre più piccolo a ogni prova; la partita va alla mossa
con il totale più alto.

Dentro la mossa che rende si ripete lo stesso schema, con le risposte che
l’avversario potrebbe darle. In questo modo, un pezzetto alla volta, cresce un
disegno di rami, fitto dalla parte che conta e quasi vuoto altrove, e ogni
partita nuova parte dal fondo del ramo già scavato invece che da qui.

Il limite è serio: il metodo funziona nei giochi in cui una posizione buona
resta buona anche giocando male. In altri giochi esiste una sola continuazione
che salva e tutte le altre perdono; lì tirare a caso dice sempre «si perde» e
non distingue più niente. Agli scacchi, per esempio, il solo gioco a caso non
funziona bene, e infatti agli scacchi non è così che si è vinto.

`````

`````{tab} Superiore

Si sostituisce la valutazione statica $\mathrm{ev}(s)$ con una stima
campionaria. Da $s$ si simulano $N$ partite fino alla fine con una politica
rapida, e si usa la frazione di vittorie come stima del valore. La politica più
semplice è uniforme sulle mosse legali, con un’eccezione: nel Go non si riempie
mai un proprio **occhio** (un incrocio vuoto circondato da pietre proprie),
perché altrimenti le partite a caso rischiano di non finire.

In generale, stimare una quantità che non si sa calcolare campionando a caso e
facendo la media si chiama **metodo Monte Carlo**, e l’idea ricorre in campi
diversi: sugli alberi di gioco, come qui; nel reinforcement learning, per
stimare il valore di uno stato dalle partite giocate; nei modelli generativi,
per stimare integrali che non hanno forma chiusa.

La politica uniforme, però, risponde alla domanda «qual è la mossa migliore se
tutti e due giocano a caso?», che coincide con quella vera soltanto nei giochi
semplici. Per questo motivo, nei programmi reali la politica di simulazione si
pesa con la conoscenza del gioco (gli schemi locali, nel Go) oppure si impara
{cite}`russell2020artificial`.

Resta da stabilire come distribuire il budget di simulazioni fra i figli della
radice. La risposta viene dal dilemma fra esplorare e sfruttare dei
{doc}`bandit a più braccia </ReinforcementLearning/banditi>`: dare più prove a
ciò che finora rende, senza smettere di provare ciò di cui si sa poco, e con
una regola che quantifichi quel «senza smettere».

Valutare una posizione giocando partite a caso non è un’idea del 2006: per i
giochi in generale la propose Bruce Abramson alla fine degli anni Ottanta
{cite}`abramson1990expected`, e nel Go ci era arrivato per primo Bernd Brügmann
{cite}`brugmann1993monte`. Nel 1993 il suo programma, al quale non aveva dato
nessuna conoscenza oltre alle regole, sul nove per nove aveva raggiunto la
forza di un principiante.

La fusione delle due cose nasce più tardi, in due tempi. Prima l’albero che
cresce una simulazione alla volta, con un modo di risalire i valori che
comincia facendo la media e finisce facendo il minimax: è di Rémi Coulom
{cite}`coulom2006efficient`, che fra i propri riferimenti mette Brügmann. Poi
la regola che decide dove spendere la simulazione successiva, cioè la regola
UCB1 dei bandit applicata a ogni nodo dell’albero, che prende il nome di
**UCT** {cite}`kocsis2006bandit`:

$$
a^\star = \arg\max_a \left[\, Q(s,a) + c \sqrt{\frac{\ln N(s)}{N(s,a)}}
\,\right],
$$

dove $Q(s,a)$ è la frazione di vittorie ottenuta finora passando per la mossa
$a$, $N(s)$ il numero di visite al nodo, $N(s,a)$ quelle al figlio e $c > 0$ il
peso dell’esplorazione: il secondo termine è grande per le mosse
provate poco e cala man mano che le si prova. Ogni iterazione scende
nell’albero con questa regola (**selezione**), aggiunge un nodo
(**espansione**), gioca da lì la partita rapida (**simulazione**) e aggiorna
$N$ e $Q$ lungo il cammino percorso (**propagazione all’indietro**).

È UCT a dare al metodo le sue garanzie, sotto ipotesi precise. In particolare,
per un orizzonte finito $D$, con ricompense in $[0,1]$ e il termine di
esplorazione moltiplicato per $D$, la distorsione della stima alla radice cala
come $O(\ln n / n)$ nel numero $n$ di simulazioni; inoltre, la probabilità di
scegliere una mossa sbagliata va a zero a velocità polinomiale. Le costanti
dipendono dalla profondità dell’albero, e su alberi costruiti
apposta il tempo prima che la garanzia cominci a valere è proibitivo
{cite}`coquelin2007bandit`; con la costante fissa che i programmi usano, poi,
il teorema non si applica alla lettera.

Il metodo ha però un limite intrinseco: la stima campionaria è tanto più
informativa quanto più il valore di una posizione è robusto rispetto alla
qualità del gioco. Di conseguenza, nei domini in cui il valore dipende da una
singola linea forzata, le simulazioni casuali sono rumore puro.

`````

«Monte Carlo» è il nome che i matematici danno da ottant’anni ai metodi che
stimano per campionamento casuale una quantità che non si sa calcolare
({doc}`la sezione sui metodi Monte Carlo </ReinforcementLearning/monte-carlo>`
racconta da dove viene), mentre l’albero del nome è quello che cresce dentro le
mosse che rendono. La {doc}`sezione su MCTS e AlphaGo
</DeepReinforcementLearning/mcts-alphago>` costruisce il metodo per intero e
mostra come una rete neurale e una ricerca si aiutino a vicenda.

In definitiva, il metodo nasce per supplire alla valutazione che manca, ed è il
motivo per cui il Go, che alla ricerca classica aveva resistito per più di
trent’anni, ha cominciato a cedere.

La ricerca ad albero Monte Carlo non è l’unica risposta. Una valutazione che
non si sa scrivere a mano si può anche imparare dai risultati delle partite.
Lo faceva già il programma di dama di Samuel con cui si apre il
{doc}`capitolo sul machine learning </MachineLearning/overview>`, lo fa la rete
di valore di AlphaGo accanto alla ricerca ad albero, e lo fanno oggi i
programmi di scacchi più forti, che hanno sostituito la valutazione scritta a
mano con una piccola rete. È il punto in cui la ricerca e l’apprendimento si
toccano.

## Quando manca il test di arrivo

L’assenza del test di arrivo passa più inosservata di quella della
valutazione, e per questo è più insidiosa.

In tutti i problemi visti fin qui c’era un test che diceva «sei arrivato».
Prova però a scriverlo per «trova una buona sistemazione dei turni del
personale». Un test esiste, ma non serve, perché dice se una sistemazione sta
in piedi (nessuno di turno due volte nello stesso momento), e di sistemazioni
che stanno in piedi ce ne sono milioni, quasi tutte pessime. Il test per
*buona* non si può scrivere: ci sono soluzioni migliori e peggiori, e nessun
punto in cui si è finito.

Il problema diventa allora di ottimizzazione. Ogni stato è già una soluzione
completa e ha un valore, il cammino per arrivarci non interessa, e la ricerca
passa da una soluzione a una vicina che vale di più. Questo modo di procedere
si chiama **ricerca locale** e non ha un punto d’arrivo: si ferma quando il
tempo a disposizione è finito, o quando i miglioramenti smettono di arrivare.

`````{tab} Elementare

Tutti abbiamo sistemato almeno una volta i mobili in una stanza. Non c’è una
disposizione «giusta» che a un certo punto scatta: c’è quella di adesso, e
quanto ti piace.

Allora provi, uno per uno, gli spostamenti di un pezzo solo: il divano contro
l’altra parete, la libreria vicino alla finestra, il tavolo al centro. Tieni
quello che rende la stanza migliore di tutti, e da lì ricominci. Funziona
finché nessuno spostamento di un pezzo solo migliora più niente, e lì ti fermi.

[Non è detto che sia la stanza più bella possibile]{.pt-matita}. Spostando
insieme il divano e il tavolo magari ne verrebbe una molto migliore, ma un
pezzo per volta non ci arrivi, perché ogni primo passo in quella direzione
peggiora. A volte, poi, ti fermi per un’altra ragione: gli spostamenti lasciano
tutti la stanza uguale, e non sai più da che parte andare. Il rimedio più
semplice è ricominciare da una stanza sistemata in un altro modo, e tenere la
migliore.

Un rimedio più fine è accettare ogni tanto un peggioramento. Invece di provare
tutti gli spostamenti ne provi uno a caso: se la stanza migliora lo tieni
sempre, se peggiora lo tieni qualche volta, e più volentieri se peggiora di
poco.

All’inizio, quando hai ancora voglia di provare, ne tieni parecchi, perché un
passo indietro può aprire la strada a qualcosa di meglio; poi, a mano a mano,
sempre meno, e alla fine soltanto quelli che migliorano. Se la voglia di
provare cala abbastanza piano, prima o poi trovi la stanza migliore; ma
«abbastanza piano» è così piano che in pratica nessuno aspetta tanto.

Si smette sempre per una ragione che con la stanza non c’entra: è ora di cena.

`````

`````{tab} Superiore

Quando manca il test di arrivo perché si cerca lo stato *migliore* e non uno
qualunque, il problema è massimizzare una funzione obiettivo $f : \mathcal{S}
\to \mathbb{R}$, o minimizzare un costo {cite}`russell2020artificial`. Gli
algoritmi di ricerca locale tengono uno stato solo, lo confrontano con i suoi
vicini $\mathcal{N}(s)$, cioè gli stati a una modifica di distanza, e si
spostano. In particolare, la **salita** (*hill climbing*) passa al vicino
migliore,

$$
s \leftarrow \arg\max_{s' \in \mathcal{N}(s)} f(s'),
$$

e [si ferma quando nessun vicino migliora, cioè in un massimo locale, che può
valere molto meno di quello globale]{.pt-matita}; si ferma anche sui plateau,
dove i vicini valgono tutti uguale, e fatica sulle creste. Il rimedio più
semplice è ripartire da stati iniziali presi a caso e tenere il migliore
(*random restart*). Con abbastanza ripartenze, su uno spazio finito, trova
l’ottimo con probabilità che tende a uno.

La **ricottura simulata** (*simulated annealing*) sceglie invece un vicino $s'$
a caso; lo accetta sempre se migliora, e se peggiora di
$\Delta f = f(s') - f(s) < 0$ lo accetta con probabilità

$$
P(\text{accetta}) = e^{\Delta f / T},
$$

dove la **temperatura** $T > 0$ scende nel tempo secondo un calendario fissato
{cite}`kirkpatrick1983optimization`.

A temperatura alta i peggioramenti passano spesso, a
temperatura bassa quasi mai, e quelli piccoli passano con probabilità maggiore
di quelli grandi; con $T \to 0$ resta una salita che non accetta
peggioramenti. Se $T$ scende abbastanza lentamente, la probabilità si concentra
sui massimi globali, che l’algoritmo trova con probabilità che tende a uno;
ma la lentezza richiesta rende la garanzia di poco aiuto in pratica
{cite}`russell2020artificial`.

Il tratto comune di questa famiglia è che non c’è un certificato di
ottimalità, salvo in casi particolari come i problemi convessi: ci si ferma
quando il calcolo a disposizione finisce o quando i miglioramenti si fermano.

`````

Questa famiglia di metodi ricorre in altri capitoli. La discesa del gradiente
dei {doc}`richiami di matematica </Matematica/analisi-ottimizzazione>`, con cui
impara una rete neurale, è ad esempio la stessa ricerca in uno spazio continuo:
si parte da una configurazione qualunque, si va nella direzione in cui il
costo scende più in fretta, ci si sposta di un passo, e nessuno dichiara mai di
essere arrivato. Più avanti, nei sistemi multi-agente, lo {doc}`sciame di
particelle </SistemiMultiAgente/sciami-e-simulazioni>` la ripete con tanti
punti di ricerca che si muovono insieme e si scambiano i risultati trovati,
senza nessuno che li coordini.

## Quando manca il modello del mondo

Il modello del mondo, cioè la funzione di transizione da interrogare a
piacere, è l’ipotesi più forte delle tre, e toglierla porta al
reinforcement learning.

`````{tab} Elementare

Ti siedi a un tavolo, davanti c’è un gioco che non hai mai visto, e nessuno ti
dà il regolamento. Non puoi provare una mossa nella tua testa, perché non sai
dove porta. Puoi solo farla, e guardare che cosa succede. Se era una mossa
disastrosa, il disastro resta: non c’è nessun «rimetto la torre dov’era».

[Di colpo sparisce l’intera costruzione fatta fin qui]{.pt-matita}. Non c’è albero
da esplorare, perché per costruire l’albero bisognerebbe sapere dove portano le
mosse. Non c’è potatura, perché non ci sono rami. Non c’è nemmeno il modo di
guardare avanti di un passo.

All’inizio resta una sola possibilità: provare, vedere com’è andata, e
ricordarlo. Chi ha fatto una mossa mille volte in situazioni simili sa, senza
conoscere le regole, che di solito finisce bene. Non ha una mappa: ha
un’esperienza.

Dopo abbastanza partite, però, il regolamento comincia a intravedersi: se ogni
volta che fai quella cosa succede quell’altra, ti sei scritto una regola tua.
Allora torni a provare le mosse nella testa come facevi prima, ma su un
regolamento indovinato al posto di quello vero. Di una mossa o due funziona
bene. Se ne incateni dieci, ogni pezzo storto si somma a quelli di prima, e il
finale che ti figuri non ha più molto a che fare con la partita vera al tavolo.

Il tavolo può essere ancora più ostile. Se ci sono dei dadi, la stessa mossa
porta ogni volta in un posto diverso, e puoi solo pesare gli esiti con le
probabilità. Se le carte dell’altro sono coperte, non sai nemmeno con
precisione dove ti trovi, e devi tenere conto di tutte le mani che lui potrebbe
avere; e lui lo sa, e può bluffare.

Contro chi bluffa non basta fare finta di vedere le sue carte: serve un modo di
giocare che non si lasci leggere, e che ogni tanto bluffi a sua volta.

`````

`````{tab} Superiore

Formalmente cade la disponibilità di $\mathrm{ris}(s,a)$ e di $c(s,a,s')$. In
altri termini, la funzione di transizione e la funzione di costo esistono ma
non sono interrogabili, se non eseguendo l’azione e osservando l’esito. È la
condizione dell’apprendimento per rinforzo, e la differenza operativa è
qualitativa: [la ricerca spende calcolo per guardare futuri immaginati, il
rinforzo spende esperienza per stimare valori da futuri accaduti]{.pt-matita}.

Ricerca e rinforzo, però, non sono alternative, e lo mostrano diverse famiglie
di metodi. Se il modello manca ma lo si può imparare, si ricade nella ricerca
in avanti usando il modello appreso al posto di quello vero. È la
{doc}`famiglia dei metodi basati su modello
</DeepReinforcementLearning/model-based>`, con il rischio che gli errori del
modello si accumulino lungo i rami immaginati. Se esiste una rete che
suggerisce dove guardare, la ricerca smette di essere cieca e diventa quella di
AlphaGo e dei suoi successori {cite}`silver2016mastering`. E se il modello non
c’è affatto, restano i metodi senza modello del reinforcement learning.

Resta un punto di contatto, anche fuori dai giochi: spendere calcolo al momento
della risposta invece che durante l’addestramento (*test-time compute*) è una
forma di ricerca. La {doc}`sezione sul post-addestramento
</Transformers/post-training>` la tratta per i modelli di linguaggio che
scrivono una lunga catena di pensiero (*chain-of-thought*) prima di
rispondere.

Le ipotesi implicite della ricerca classica, poi, sono più di tre, e due cadono
già nei giochi. Se l’esito di un’azione non è certo, come col dado del
backgammon, la transizione diventa una distribuzione $P(s' \mid s, a)$ e il
minimax diventa l’expectiminimax della {doc}`sezione sui giochi
</Ricerca/giocare-contro-qualcuno>`, con nodi di caso che fanno la media.

Se lo stato non si vede del tutto, come con le carte coperte, l’agente conosce
soltanto una distribuzione sugli stati possibili, il **belief state**, e il
problema è un POMDP, come nella {doc}`sezione sugli MDP
</ReinforcementLearning/mdp-valore>`. Lì non basta più fare la media del
minimax sulle carte possibili. Quella media sull’onniscienza (*averaging over
clairvoyance*) suppone infatti che dopo la prossima mossa tutti vedano tutto, e
quindi non sceglie mai una mossa per raccogliere informazione e non bluffa mai
{cite}`russell2020artificial`.

Serve una strategia di equilibrio, e i programmi che hanno battuto i
professionisti di poker la calcolano su una versione astratta del gioco
minimizzando il rimpianto controfattuale {cite}`zinkevich2008regret`: Libratus
a due giocatori {cite}`brown2018superhuman`, Pluribus a sei
{cite}`brown2019superhuman`.

`````

Letti insieme, i tre casi mostrano che ogni ipotesi tolta apre una direzione di
lavoro, e spesso più di una. Senza una valutazione scritta a mano la si stima
per campionamento, come nella ricerca ad albero Monte Carlo, oppure la si
impara dalle partite, come faceva Samuel e come fanno le reti di valore. Senza
il test di arrivo si finisce nell’ottimizzazione, cioè nel migliorare senza mai
arrivare, con la ricerca locale e con la discesa del gradiente. Senza un
modello da interrogare si entra nell’apprendimento per rinforzo, oppure si
impara un modello e ci si cerca dentro.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Tutta la ricerca in avanti poggia su tre ipotesi: che le regole del mondo si
  possano interrogare quante volte si vuole, per provare le mosse nella
  propria testa; che l’arrivo si sappia riconoscere con un test; e che a una
  posizione di mezzo si sappia dare un voto, cioè una valutazione.
- Se manca il voto (è il caso del Go, dove nessuno è riuscito a scriverlo a
  mano), lo si sostituisce con un conteggio: da qui, gioca mille partite a caso
  e guarda quante ne vinci. Le partite non si spartiscono in parti uguali fra
  le mosse candidate: ne va di più a quelle che stanno rendendo, e qualcuna
  resta sempre per quelle provate poco. Funziona nei giochi in cui una
  posizione comoda resta comoda anche giocando male, e non funziona dove esiste
  una sola continuazione che salva. Il voto, poi, si può anche imparare dalle
  partite giocate.
- Se manca l’arrivo, la ricerca smette di cercare una strada e si mette a
  migliorare quello che ha, come si fa sistemando i mobili in una stanza: un
  passo alla volta ci si può fermare in una sistemazione che non è la
  migliore, e accettando ogni tanto un peggioramento se ne esce. Ci si ferma
  quando scade il tempo.
- Se mancano le regole da interrogare, all’inizio cade tutto: non c’è albero,
  non c’è potatura, non c’è niente da guardare avanti. Resta provare e
  ricordarsi com’è andata, e questo si chiama apprendimento per rinforzo. Poi
  il regolamento comincia a intravedersi, e si torna a provare le mosse nella
  testa su quello indovinato: di una mossa o due funziona, di dieci incatenate
  ogni pezzo storto si somma a quelli di prima.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Fra le ipotesi implicite della ricerca classica, qui ne cadono tre: modello
  interrogabile ($\mathrm{ris}$ e $c$ disponibili a costo nullo), test di
  arrivo definito, valutazione degli stati intermedi scrivibile. Le altre due,
  esito certo delle azioni e stato pienamente osservabile, cadono già nei
  giochi: con il caso il minimax diventa expectiminimax, con le carte coperte
  si ragiona sul *belief state*, e nel poker servono strategie di equilibrio.
- Cade la valutazione: si sostituisce $\mathrm{ev}(s)$ con una stima
  campionaria ottenuta simulando partite fino in fondo, e si distribuiscono le
  simulazioni con UCB1 a ogni nodo, risolvendo un problema di esplorazione
  contro sfruttamento: è la ricerca ad albero Monte Carlo
  {cite}`coulom2006efficient,kocsis2006bandit`, costruita per intero nella
  {doc}`sezione su MCTS e AlphaGo </DeepReinforcementLearning/mcts-alphago>`.
  Oppure la valutazione si impara dai risultati delle partite.
- Cade il test di arrivo: il problema diventa di ottimizzazione di una
  funzione obiettivo, la ricerca è locale (salita, ricottura simulata) e non
  c’è un certificato di ottimalità.
- Cade il modello interrogabile: si entra nell’apprendimento per rinforzo.
  La distinzione operativa è che la ricerca spende calcolo su futuri
  immaginati e il rinforzo spende esperienza su futuri accaduti; il ponte fra
  i due sono i metodi che imparano il modello e poi ci cercano dentro.
```

`````

Il {doc}`capitolo sul reinforcement learning </ReinforcementLearning/overview>`
comincia dove manca il modello del mondo, e dà in cambio una capacità che la
ricerca, per raffinata che sia, non ha. La ricerca, infatti, non trasforma il
lavoro di una mossa in esperienza. Della mossa di prima conserva al più una
tabella di posizioni già giudicate o un pezzo d’albero, mentre la sua
valutazione e il suo modo di ordinare le mosse restano quelli scritti
all’inizio, e la partita di ieri non le ha insegnato niente. Un agente che
impara, invece, alla seconda partita parte da dove era arrivato nella prima.
