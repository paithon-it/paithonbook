# Le tre cose che davamo per scontate

Tutto quello che il capitolo ha costruito finora poggia su tre regali che
nessuno ha mai messo per iscritto, perché sembravano ovvi. Conviene scriverli
adesso, uno per riga, e poi toglierli uno alla volta: perché è togliendoli che
si arriva a tutto il resto del libro.

**Le regole**: si possono interrogare quante volte si vuole. Prima di muovere
davvero, il programma prova mille mosse nella sua testa e per ciascuna sa
esattamente dove porta, gratis e senza conseguenze. Sposta la torre, guarda, e
la rimette dov’era.

**L’arrivo**: si sa riconoscere. Le tessere in ordine, il re sotto scacco
matto, l’incrocio giusto. C’è un test che dice sì o no.

**Il voto**: si sa dare a una posizione di mezzo. Non un voto esatto, ma un
voto sensato: chi sta meglio, quanto manca.

Nel mondo vero ognuno dei tre può mancare, e le tre parti che seguono li
tolgono uno per volta: prima il voto, che è quello che si rompe più facilmente;
poi l’arrivo; per ultime le regole, che sono quelle che fanno cadere tutto il
resto. Ognuno, mancando, porta a un pezzo diverso del libro, e l’ultimo porta
al capitolo dopo questo.

## Quando nessuno sa scrivere il voto

Il voto è il primo dei tre a cadere, ed è caduto su un gioco preciso.

Agli scacchi il voto si sa scrivere: quanti pezzi ho, quanto vale ciascuno, se
il re è al riparo, come stanno i pedoni. È la somma della sezione precedente, e
funziona: nel 1997 il programma Deep Blue, che faceva esattamente questo e la
potatura vista qui, giocò sei partite contro Garri Kasparov, che era il
giocatore più forte del mondo, e ne uscì in vantaggio: tre e mezzo a due e
mezzo, cioè due vittorie contro una e tre patte.

Il Go è un altro gioco, e conviene dire com’è fatto perché in Italia lo si
vede di rado. Si gioca su una griglia grande, diciannove righe per diciannove,
e i pezzi sono sassolini tutti uguali, bianchi e neri, che si posano sugli
incroci e da lì non si muovono più. Non ci sono re, torri o alfieri: c’è solo
il disegno che i sassolini formano, e vince chi alla fine ha circondato più
territorio.

E qui il voto non si sa scrivere. Non c’è materiale da contare, perché i pezzi
sono tutti uguali e restano fermi; quanto valga un sassolino dipende da come
stanno i suoi vicini e i vicini dei vicini, cioè da un disegno che può occupare
mezza griglia. Una posizione forte i giocatori la riconoscono a colpo d’occhio,
ma la regola con cui la riconoscono nessuno è mai riuscito a scriverla in una
formula, e non per pigrizia: ci si è provato per trent’anni. E senza il voto,
tutta la macchina della sezione precedente si ferma: si guarda avanti, si
arriva al punto in cui bisogna fermarsi, e lì non c’è niente da leggere.

`````{tab} Elementare

L’idea che sblocca la faccenda è di quelle che sembrano una presa in giro, e
funzionano.

Non sai giudicare una posizione? E allora **non giudicarla**: da lì, gioca la
partita fino in fondo tirando le mosse a caso, e guarda come finisce. Poi
rifallo. Poi rifallo mille volte. Alla fine non hai un giudizio, hai un
conteggio: da questa posizione, tirando a caso, ho vinto (mettiamo)
seicentotrenta volte su mille.

Detta così sembra assurdo, perché nessuna di quelle mille partite somiglia a
una partita vera: sono mosse a caso, giocate malissimo da tutti e due. Ed è
proprio questo il punto, e vale la pena capirlo bene: giocate malissimo **da
tutti e due**. Se una posizione è davvero buona per me, resta buona anche in
un mondo in cui giochiamo tutti a caso, perché il vantaggio non dipende dalla
mia bravura. Il conteggio non misura come andrebbe la partita: misura quanto
la posizione è *comoda*, e per scegliere una mossa spesso basta.

Il punto di rottura c’è ed è serio: questo funziona nei giochi in cui una
posizione buona resta buona anche giocando male. Ci sono giochi in cui non è
così, in cui esiste una sola continuazione che salva e tutte le altre
perdono: lì tirare a caso dice sempre «si perde», e non distingue più niente.
Agli scacchi, per esempio, questo trucco da solo non funziona bene, e infatti
agli scacchi non è così che si è vinto.

`````

`````{tab} Superiore

La mossa è sostituire la valutazione statica $\mathrm{ev}(s)$ con una stima
**campionaria**: da $s$ si simulano $N$ partite fino alla fine con una politica
rapida (nella versione più semplice, uniforme sulle mosse legali, con
l’eccezione di quelle che nel Go riempirebbero un proprio occhio: senza quel
divieto le partite a caso non finiscono) e si usa la
frazione di vittorie come stima del valore. Stimare una quantità che non si sa
calcolare campionando a caso e facendo la media si chiama **metodo Monte
Carlo**, ed è un attrezzo che il libro riuserà in tre posti diversi: qui su un
albero di gioco, nel capitolo seguente per stimare il valore di uno stato dalle
partite giocate, e nei modelli generativi per stimare integrali che non hanno
forma chiusa.

Resta da dire come si distribuisce il budget di simulazioni fra i figli della
radice, e la risposta non è «in parti uguali»: è **esattamente** il dilemma fra
esplorare e sfruttare che il capitolo seguente introdurrà con i bandit a più
braccia. Dare più prove a ciò che finora rende, senza smettere di provare ciò
di cui si sa poco, e con una regola che quantifichi quel «senza smettere».

Valutare una posizione giocando partite a caso non è un’idea del 2006: il
primo era stato Bernd Brügmann {cite}`brugmann1993monte`, che nel 1993, senza
dare al programma nessuna conoscenza oltre alle regole, sul nove per nove
aveva raggiunto la forza di un principiante, ed è Coulom stesso a citarlo. Quello che nasce in quegli anni è la **fusione** delle due cose,
e nasce in due tempi. Prima l’albero che cresce **una simulazione alla volta**,
con un modo di risalire i valori che comincia facendo la media e finisce
facendo il minimax {cite}`coulom2006efficient`. Poi la regola che decide dove
spendere la simulazione successiva, cioè la stessa regola dei bandit (si chiama
UCB1, e sceglie il ramo col miglior compromesso fra quanto ha reso finora e
quanto poco lo si è provato) applicata a ogni nodo dell’albero
{cite}`kocsis2006bandit`: è la seconda a dare al metodo le garanzie di
convergenza, dimostrate però per una classe circoscritta di problemi e non in
generale.

Vale la pena registrare anche il limite, perché è quello che il metodo si porta
dietro: la stima campionaria è tanto più informativa quanto più il valore di
una posizione è **robusto rispetto alla qualità del gioco**. Nei domini in cui
il valore dipende da una singola linea forzata, le simulazioni casuali sono
rumore puro.

`````

Questa idea ha un nome, **ricerca ad albero Monte Carlo** (in inglese *Monte
Carlo tree search*, abbreviata in MCTS). «Monte Carlo» è il nome che i
matematici danno da ottant’anni ai metodi che stimano per sorteggio quello che
non si sa calcolare, e da dove venga quel nome lo racconta il capitolo
seguente, che sui metodi Monte Carlo ha una sezione sua. Ma il metodo non si
esaurisce nel contare le partite: quello che lo rende praticabile è non
spartire le simulazioni in parti uguali, perché darne mille a una mossa
palesemente perdente è tempo buttato.

Il libro la costruisce per intero due capitoli più avanti, in quello che
applica le reti profonde all’apprendimento per rinforzo (in inglese *deep
reinforcement learning*, che è il nome con cui il capitolo si presenta), dove
serve a raccontare come una rete neurale e una ricerca si aiutino a vicenda. Qui
interessa solo il suo posto in questa storia, che è preciso: nasce per
rispondere al voto che manca, ed è la ragione per cui il Go, che alla ricerca
classica aveva resistito per trent’anni, ha cominciato a cedere.

## Quando l’arrivo non si sa dire

L’arrivo cade più silenziosamente del voto, e per questo è più insidioso.

In tutti i problemi di questo capitolo c’era un test che diceva «sei
arrivato». Ma prova a scriverlo per «trova una buona sistemazione dei turni del
personale». Un test in realtà c’è, e non serve a niente: dice se una
sistemazione sta in piedi (nessuno di turno due volte nello stesso momento), e
di sistemazioni che stanno in piedi ce ne sono milioni, quasi tutte pessime.
Quello che manca è il test per **buona**, e quello non si scrive: ci sono
soluzioni migliori e peggiori, e nessun punto in cui si è finito.

Quando succede questo, la ricerca cambia natura, e diventa una cosa che tutti
abbiamo fatto almeno una volta: sistemare i mobili in una stanza. Non c’è una
disposizione «giusta» che a un certo punto scatta; c’è quella di adesso, ci
sono gli spostamenti che la migliorano un po’, e a un certo punto si smette
perché è ora di cena. Nessuno ha finito: uno ha smesso.

Questa famiglia di metodi il libro la incontra altrove sotto altri nomi. È
quello che fa la **discesa del gradiente** dei richiami di matematica, cioè il
modo in cui una rete neurale impara: si parte da una configurazione qualunque,
si guarda da che parte migliora, ci si sposta di un passo, e nessuno dice mai
che si è arrivati. Ed è quello che rifanno gli **sciami** del capitolo sui
sistemi multi-agente, dove a spostare i mobili sono in tanti insieme e nessuno
comanda.

## Quando le regole non si possono interrogare

Le regole sono il regalo più grosso dei tre, e toglierle è quello che apre il
capitolo seguente.

`````{tab} Elementare

Ti siedi a un tavolo, davanti c’è un gioco che non hai mai visto, e nessuno ti
dà il regolamento. Non puoi provare una mossa nella tua testa, perché non sai
dove porta. Puoi solo farla per davvero, e guardare che cosa succede. E se era
una mossa disastrosa, il disastro te lo tieni: non c’è nessun «rimetto la torre
dov’era».

Sparisce tutto quello che il capitolo ha costruito. Non c’è albero da
esplorare, perché per costruire l’albero bisognerebbe sapere dove portano le
mosse. Non c’è potatura, perché non ci sono rami. Non c’è nemmeno il modo di
guardare avanti di un passo.

Quello che resta è una cosa sola: provare, vedere com’è andata, e **ricordarsi**
com’è andata. Chi ha fatto una mossa mille volte in situazioni simili sa, senza
conoscere le regole, che di solito finisce bene. Non ha una mappa: ha
un’esperienza.

Ed è esattamente quello che si impara a fare da qui in poi.

`````

`````{tab} Superiore

Formalmente cade la disponibilità di $\mathrm{ris}(s,a)$ e di $c(s,a,s')$: la
funzione di transizione e la funzione di costo esistono ma non sono
interrogabili, se non eseguendo davvero l’azione e osservando l’esito. È la
condizione dell’**apprendimento per rinforzo**, e la differenza operativa non è
di grado ma di natura: la ricerca spende **calcolo** per guardare futuri
immaginati, il rinforzo spende **esperienza** per stimare valori da futuri
davvero accaduti.

Le due cose non sono alternative, e i capitoli che seguono lo mostrano in tre
modi. Se il modello manca ma lo si può **imparare**, si ricade nel caso di
questo capitolo usando il modello appreso al posto di quello vero: sono i
metodi basati su modello del capitolo sul deep reinforcement learning, con il
rischio che gli errori del modello si accumulino lungo i rami immaginati. Se
esiste una rete che **suggerisce dove guardare**, la ricerca smette di essere
cieca e diventa quella di AlphaGo e dei suoi successori
{cite}`silver2016mastering`. E se il modello non c’è affatto, restano i metodi
del capitolo seguente.

Resta un punto di contatto che vale la pena nominare adesso, perché il libro ci
tornerà alla fine: anche a modello ignoto, **pensare prima di rispondere paga**.
Il calcolo speso al momento della risposta invece che durante l’addestramento è
una forma di ricerca, e il capitolo sui Transformer lo tratterà per esteso
parlando dei modelli che scrivono una lunga brutta copia prima di rispondere.

`````

Conviene guardare i tre casi tutti insieme, perché messi in fila dicono una
cosa che presa uno per uno non si vede: **non sono tre difficoltà, sono tre
destinazioni**. Senza il voto si arriva alla ricerca ad albero Monte Carlo;
senza l’arrivo si finisce nell’ottimizzazione, quella della discesa del
gradiente dei richiami di matematica e degli sciami; senza le regole da
interrogare, all’apprendimento per rinforzo. Un metodo di intelligenza artificiale, spesso,
è il nome che diamo a un regalo che ci hanno tolto.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Tutta la ricerca di questo capitolo poggia su tre regali: **le regole**, che
  si possono interrogare quante volte si vuole per provare le mosse nella
  propria testa; **l’arrivo**, che si sa riconoscere; e **il voto**, che si sa
  dare a una posizione di mezzo.
- Se manca **il voto** (è il caso del Go, dove nessuno è mai riuscito a
  scriverlo), lo si sostituisce con un conteggio: da qui, gioca mille partite a
  caso e guarda quante ne vinci. Funziona nei giochi in cui una posizione comoda
  resta comoda anche giocando male, e non funziona dove esiste una sola
  continuazione che salva.
- Se manca **l’arrivo**, la ricerca smette di cercare una strada e si mette a
  migliorare quello che ha, fermandosi quando scade il tempo, come si fa
  sistemando i mobili in una stanza.
- Se mancano **le regole** da interrogare, casca tutto: non c’è albero, non c’è
  potatura, non c’è niente da guardare avanti. Resta solo provare per davvero e
  ricordarsi com’è andata, e questo ha un nome: **apprendimento per
  rinforzo**.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Le tre ipotesi implicite della ricerca classica sono: modello interrogabile
  ($\mathrm{ris}$ e $c$ disponibili a costo nullo), test di terminazione
  definito, valutazione degli stati intermedi scrivibile.
- Cade la **valutazione**: si sostituisce $\mathrm{ev}(s)$ con una stima
  campionaria ottenuta simulando partite fino in fondo, e si distribuiscono le
  simulazioni risolvendo un problema di esplorazione contro sfruttamento.
  È la ricerca ad albero Monte Carlo {cite}`coulom2006efficient,kocsis2006bandit`,
  che il libro costruisce nel {doc}`capitolo sul deep reinforcement learning </DeepReinforcementLearning/overview>`.
- Cade il **test di terminazione**: il problema diventa di ottimizzazione, non
  di ricerca di un cammino.
- Cade il **modello interrogabile**: si entra nell’apprendimento per rinforzo.
  La distinzione operativa è che la ricerca spende **calcolo** su futuri
  immaginati e il
  rinforzo spende **esperienza** su futuri accaduti; il ponte fra i due sono i
  metodi che imparano il modello e poi ci cercano dentro.
```

`````

Il capitolo che viene adesso toglie il regalo più grosso dei tre, quello di
poter provare le mosse nella propria testa. In cambio dà una cosa che qui non
c’è mai stata. La ricerca, per raffinata che sia, ricomincia da zero a ogni
mossa: quello che ha capito pensando alla mossa di prima lo butta via, e la
partita di ieri non le ha insegnato niente. Chi impara, invece, la seconda
volta comincia da dove era arrivato.
