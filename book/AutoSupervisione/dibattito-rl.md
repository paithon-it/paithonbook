# La ciliegina: il dibattito sul rinforzo

Nella torta di LeCun l'apprendimento per rinforzo è la ciliegina. È
un'immagine simpatica e una retrocessione severa: due capitoli interi sono
dedicati a una decorazione. Conviene prenderla sul serio, perché a dirlo non
sono degli estranei al campo, e perché l'argomento che ci sta sotto è quello di
sempre: quanto è grande la correzione che il modello riceve.

Una premessa, che qui conta più del solito: le posizioni che seguono sono di
persone vive su una questione aperta, e ciascuna arriva con il nome giusto
attaccato. Come si vedrà, non è banale nemmeno quello.

## L'argomento, in due mosse

Le critiche mosse all'apprendimento per rinforzo «puro» sono quattro, e due si
discutono qui. Vanno viste come **due mosse dello stesso ragionamento**, non
come due obiezioni separate.

Le altre due hanno già avuto il loro spazio: la ricompensa scritta male e
ottimizzata alla lettera sta nel {doc}`capitolo sul deep reinforcement
learning </DeepReinforcementLearning/overview>` e nella sezione sul
post-addestramento del
{doc}`capitolo sui Transformer </Transformers/overview>`; il costo in esperienza
dell'imparare per tentativi nel mondo, invece che dentro un modello del mondo,
è la tesi del {doc}`capitolo sui world model </WorldModels/overview>`.

`````{tab} Elementare

Torniamo alla giornata nel bosco, quella al termine della quale qualcuno ti
dice soltanto «oggi hai fatto bene».

La prima obiezione è quella che già conosciamo: è pochissimo. Una frase per una
giornata intera. Se invece qualcuno ti avesse commentato ogni singolo
avvistamento, avresti ricevuto migliaia di volte più informazione.

La seconda obiezione è più insidiosa, e non riguarda la quantità: riguarda il
fatto che quella frase, per essere utile, **va distribuita**. Nella giornata hai
fatto centinaia di cose. Hai preso il sentiero giusto, ti sei fermato troppo
presto a una radura, hai avuto la pazienza di aspettare mezz'ora sotto un faggio,
e a un certo punto hai fatto rumore e hai fatto volare via tutto. Alla fine il
giudizio è positivo. Che cosa impari? Che la giornata è andata bene: quindi, non
sapendo distinguere, tendi a ripetere tutto, compresa la sosta inutile e il
rumore.

Rimediare si può, in parte. Confronta la giornata con le tue solite, e un
«bene» come quello di sempre smette di giustificare tutto. Fatti accompagnare
da qualcuno che, dopo tante uscite, provi a dirti come sta andando mentre
ancora cammini. E da' più peso a quello che hai fatto poco prima di un
avvistamento riuscito che a quello che avevi fatto sei ore prima, perché a
quella distanza la colpa e il merito si perdono comunque. Sono rimedi veri, e
aiutano a smistare. Ma nemmeno chi ti accompagna ha mai sentito altro che la
frase della sera: nessuno di quei rimedi inventa quello che nessuno ha detto.

Il punto è che le due obiezioni si moltiplicano invece di sommarsi. Poca
informazione sarebbe già un problema; poca informazione, e da spalmare su
centinaia di decisioni che non sai distinguere, è un problema di un altro
ordine. E più lunga è la giornata, più sono le decisioni fra cui dividerla.

`````

`````{tab} Superiore

La prima mossa è quantitativa ed è già misurata: il bersaglio nel rinforzo è
uno scalare per episodio, cioè al più pochi bit,
mentre nel pre-addestramento auto-supervisionato è dell'ordine di $10^5$ bit per
esempio. È il rapporto che il programma sui bit stampa.

La seconda mossa è strutturale e riguarda **l'assegnazione del credito**
(*credit assignment*). Un ritorno osservato $R$ alla fine di una traiettoria
$\tau = (s_0, a_0, \dots, s_{T-1}, a_{T-1})$ non dice quale delle $T$ azioni
abbia contribuito. Nella forma più semplice del gradiente di policy,

$$
\nabla_\theta J(\theta) \;\approx\;
\frac{1}{M}\sum_{m=1}^{M} R^{(m)} \sum_{t=0}^{T-1}
\nabla_\theta \log \pi_\theta\!\left(a^{(m)}_t \mid s^{(m)}_t\right),
$$

dove $M$ è il numero di traiettorie campionate e $J(\theta)$ il ritorno atteso
della politica. Lo scalare $R^{(m)}$ è uno per traiettoria, e moltiplica il
gradiente di ogni azione di quella traiettoria: le azioni buone e quelle cattive
di un episodio riuscito ricevono lo stesso coefficiente, tutte spinte verso
l'alto. La sezione sui
metodi a gradiente di policy, nel capitolo sul deep reinforcement learning,
mostra come si attenua il problema (una linea di base per ridurre la varianza,
un critico che stima il vantaggio azione per azione, lo sconto che accorcia
l'orizzonte di attribuzione), e sono attenuazioni vere; ma nessuna di esse
fabbrica informazione che nel segnale non c'era.

Le due mosse si compongono: pochi bit, e per giunta da distribuire su un numero
di decisioni che cresce con la lunghezza dell'episodio.

`````

Il disegno di {numref}`fig-credito-spalmato` mostra la seconda mossa mentre
avviene, che è l'unico modo di vederla: da ferma sembrerebbe soltanto una fila
di riquadri tutti uguali, ed è proprio quell’«uguali» il problema.

```{figure} ../figures/credito-spalmato.svg
:name: fig-credito-spalmato
:alt: Una fila di dodici riquadri, i passi di una traiettoria; cinque sono bordati di terracotta perché in quei passi l'agente ha sbagliato. Una testa percorre la fila da sinistra a destra un passo alla volta; alla fine compare un solo riquadro con il ritorno, più uno. Da lì partono dodici archi che tornano indietro fino a ogni riquadro e li riempiono tutti dello stesso colore. Sotto, due righe di numeri: la riga del peso porta più uno dodici volte identiche, la riga del merito cambia fra più uno e meno uno da un passo all'altro.
:width: 100%

Un solo numero alla fine, e poi indietro su tutto. La riga del **peso** è quella
che l'algoritmo applica davvero a ciascun passo, ed è la stessa dappertutto; la
riga del **merito** è quella che l'algoritmo non vede, e cambia da passo a
passo fra più uno e meno uno. I passi bordati di terracotta sono quelli in cui
l'agente ha sbagliato, e ricevono la stessa spinta di tutti gli altri.
```

## Che cosa propone LeCun al suo posto

Per LeCun la conseguenza di questo argomento non è che il rinforzo sia inutile,
ma che sia **nel posto sbagliato**: va bene per rifinire una policy quando la
comprensione del mondo è già stata costruita altrove, e non va bene come modo
di costruirla.

Questa posizione è già comparsa, senza la sua motivazione. Nel
{doc}`capitolo sui modelli a energia </ModelliEnergia/overview>` c'è l'elenco
delle quattro rinunce che LeCun ripete nelle sue conferenze, e la quarta dice:
abbandonare l'apprendimento per
rinforzo in favore del **controllo predittivo basato su modello**, cioè
costruirsi un modello di come va il mondo, pianificare dentro quello, e ricorrere
ai tentativi soltanto per correggere il modello quando la previsione sbaglia.

Quel conto sui bit è il perché di quella riga. Se ogni interazione col
mondo vero paga pochi bit, allora un sistema che impara *soltanto* interagendo è
condannato a un numero di interazioni che nessun corpo fisico può permettersi; se
invece la comprensione del mondo si costruisce guardando, cioè con un segnale
grande quanto il dato, l'interazione serve solo là dove la previsione fallisce,
ed è pochissima. È la stessa aritmetica, letta come progetto di macchina.

Va detto che questa è una **posizione dentro un dibattito aperto**, non un
risultato: dice come andrebbe costruita una macchina, e le macchine costruite
così non sono ancora quelle che funzionano meglio. Il capitolo sui world model
la prende sul serio esattamente in questi termini.

## Come lo dice Karpathy

Andrej Karpathy, fra i fondatori di OpenAI e per anni responsabile
dell'intelligenza artificiale in Tesla, ha dato a questo argomento la
formulazione che è rimasta. In un'intervista dell'ottobre 2025
{cite}`karpathy2025dwarkesh` descrive che cosa succede quando si addestra un
modello a risolvere un problema di matematica generando centinaia di tentativi e
tenendo quelli che arrivano alla risposta giusta:

> Il modo in cui mi piace dirlo è che stai aspirando la supervisione attraverso
> una cannuccia. Hai fatto tutto questo lavoro, che poteva essere un minuto di
> svolgimento, e stai aspirando i bit di supervisione del segnale di ricompensa
> finale attraverso una cannuccia, e li stai trasmettendo per radiodiffusione a
> tutta la traiettoria, e usi quello per alzare o abbassare il peso di quella
> traiettoria.[^straw]

E sulla seconda mossa, l'assegnazione del credito, è ancora più netto:

> Assume quasi che ogni singolo pezzetto della soluzione che hai prodotto e che
> è arrivato alla risposta giusta fosse la cosa corretta da fare, il che non è
> vero. Puoi essere finito in vicoli ciechi prima di arrivare alla soluzione
> giusta. Ognuna di quelle cose sbagliate che hai fatto, purché tu sia arrivato
> alla soluzione corretta, verrà rinforzata come «fanne ancora». È terribile. È
> rumore.[^rumore]

[^straw]: «The way I like to put it is you're sucking supervision through a
straw. You've done all this work that could be a minute of rollout, and you're
sucking the bits of supervision of the final reward signal through a straw and
you're broadcasting that across the entire trajectory and using that to upweight
or downweight that trajectory» {cite}`karpathy2025dwarkesh`. La traduzione qui
sopra è nostra, come quelle che seguono.

[^rumore]: «It almost assumes that every single little piece of the solution
that you made that arrived at the right answer was the correct thing to do,
which is not true. You may have gone down the wrong alleys until you arrived at
the right solution. Every single one of those incorrect things you did, as long
as you got to the correct solution, will be upweighted as, "Do more of this."
It's terrible. It's noise» {cite}`karpathy2025dwarkesh`.

Va però riportato anche quello che dice **nella stessa risposta**, perché senza
di esso la posizione diventa un'altra:

> L'apprendimento per rinforzo è terribile. Si dà solo il caso che tutto quello
> che avevamo prima sia molto peggio, perché prima ci limitavamo a imitare le
> persone.

e, poche righe dopo, che il rinforzo permette di fare meglio della sola
imitazione, che su certi problemi si può migliorare senza avere traiettorie di
esperti da copiare, e che «il modello può anche scoprire soluzioni che una
persona non troverebbe mai. È incredibile. Eppure, è ancora stupido».

Non è dunque una richiesta di abbandonare il rinforzo. È la tesi che lo
strumento che oggi si usa sia molto peggiore di come viene percepito, e che il
campo abbia bisogno di qualcosa in più.

## Un'attribuzione da rimettere a posto

C'è una tesi che circola attaccata al nome di Karpathy e che non è sua. Merita di essere raccontata, perché l'errore che la fa circolare è di quelli
comuni: si attribuisce una posizione a chi non l'ha mai espressa, solo perché
il racconto suona plausibile.

La tesi è questa: l'apprendimento per rinforzo applicato ai modelli linguistici
non aggiungerebbe capacità nuove, si limiterebbe a **restringere e selezionare**
percorsi di ragionamento che il modello aveva già acquisito nel
pre-addestramento.

È una tesi tecnica precisa, ed è di Yang Yue e colleghi {cite}`yue2025rlvr`.
È una misura e non un'opinione, e come è fatta è la parte interessante.

`````{tab} Elementare

Come si fa a misurare se un addestramento ha aggiunto capacità o solo messo
ordine in quelle che c'erano? Gli autori usano un'idea semplice: invece di
chiedere al modello una risposta, gliene chiedono tante alla stessa domanda, e
contano se almeno una è giusta.

Se ne chiedi una sola, il modello addestrato col rinforzo vince: è più affidabile,
azzecca più spesso al primo colpo. Ma se gliene chiedi moltissime, succede il
contrario: è il modello di partenza, quello che il rinforzo non ha mai toccato,
a risolvere problemi che l'altro non risolve, e nelle misure degli autori il
sorpasso arriva già a qualche decina o qualche centinaio di tentativi, non a
numeri irraggiungibili.

La lettura degli autori è che l'addestramento non abbia quasi mai insegnato
qualcosa di nuovo: abbia reso più probabili alcune strade che il modello sapeva
già percorrere, e nel farlo abbia reso improbabili le altre. Più preciso, e più
stretto.

Attenzione a non tirare la conclusione più in là di dove arriva la misura, e
sono gli autori i primi a dirlo. Il trucco delle tante risposte non si può
spingere all'infinito: se i tentativi fossero un numero assurdo, prima o poi
anche battendo i tasti a caso salterebbe fuori la risposta giusta, e il
confronto non direbbe più niente. Quello che è stato misurato è il modo in cui
questo addestramento si fa **oggi**, non un limite di principio.

`````

`````{tab} Superiore

Il protocollo è il **pass@$k$**: la probabilità che almeno una fra $k$ risposte
campionate indipendentemente alla stessa domanda sia corretta. È una misura del
**confine** delle capacità, non della loro affidabilità: per $k = 1$ premia chi
azzecca al primo colpo, per $k$ grande premia chi *possiede* la soluzione da
qualche parte nella propria distribuzione, anche se raramente.

Il risultato, dall'abstract (RLVR è il rinforzo con ricompensa **verificabile**,
quella che un controllo automatico assegna senza il giudizio di nessuno):
«l'attuale impostazione di addestramento suscita
di rado schemi di ragionamento fondamentalmente nuovi. Mentre i modelli
addestrati con RLVR superano i loro modelli base per valori piccoli di $k$ (per
esempio $k=1$), i modelli base ottengono un punteggio pass@$k$ più alto quando
$k$ è grande» {cite}`yue2025rlvr`. Gli autori osservano inoltre che il confine
delle capacità di ragionamento **spesso si restringe** al procedere
dell'addestramento. Le curve pubblicate arrivano a $k = 1024$, e il sorpasso
non richiede di arrivare fin lì: avviene «man mano che $k$ cresce fino a decine
o centinaia», e sul banco di prova Minerva, con un modello da 32 miliardi di
parametri, a $k = 128$ il modello base risolve circa il 9% di problemi in più di
quello addestrato.

Tre avvertenze, e le pone il lavoro stesso. La prima: la misura riguarda gli
schemi di ragionamento accessibili per campionamento, quindi la conclusione è
sulla distribuzione del modello base, non sull'impossibilità in linea di
principio. La seconda: il pass@$k$ non si estrapola all'infinito, e sono gli
autori a dirlo, perché «con un $k$ astronomicamente grande perfino il
campionamento uniforme sul dizionario dei token inciamperebbe nel percorso di
ragionamento corretto». La terza: il titolo è una domanda, e la risposta che dà
è sull'impostazione di addestramento corrente, con gli algoritmi e i benchmark
esaminati. Trasformarla in «il rinforzo non serve» è un salto che il lavoro non
autorizza.

`````

Le due posizioni sono compatibili e vanno tenute distinte: Karpathy dice che il
segnale è povero e mal distribuito; Yue e colleghi misurano che, con
l'impostazione di oggi, il confine delle capacità non si allarga. Ma la seconda
non è un'affermazione della prima persona, e chi le fonde attribuisce a Karpathy
una misura che non ha fatto e una conclusione che nella stessa intervista
contraddice, quando dice che il rinforzo fa scoprire soluzioni che una persona
non troverebbe.

## E chi non è d'accordo

Un capitolo che riportasse solo le critiche darebbe un'impressione falsa, perché
mentre si discuteva di ciliegine l'apprendimento per rinforzo ha fatto due cose
grosse, e vanno raccontate entrambe.

La prima è che gli assistenti conversazionali che tutti usano sono rifiniti
così. Il {doc}`capitolo sui Transformer </Transformers/overview>`, nella sezione sul post-addestramento, mostra
il conto: un modello piccolo ma rifinito sulle preferenze umane batteva, nel
giudizio delle persone, un modello più di cento volte più grande. Se il
rinforzo è una ciliegina, è una ciliegina che ha cambiato il sapore della torta.

La seconda è più recente e più diretta al punto. I modelli cosiddetti
«ragionanti» si addestrano col rinforzo su problemi a **risposta verificabile**
(la correttezza di un risultato matematico, il superamento di una batteria di
test per il codice), dove la ricompensa non richiede il giudizio di nessuno. Lì
non solo il metodo funziona, ma fa emergere comportamenti che non erano stati
programmati, come rileggere i propri passaggi e cambiare strada, ed è un
resoconto che si legge nel rapporto tecnico di DeepSeek-R1
{cite}`guo2025deepseek`. È
un'osservazione che convive senza contraddizione con il risultato di Yue e
colleghi, e le due cose insieme dicono una terza cosa: l'addestramento sposta
massa di probabilità verso strade che pagano, e questo è utilissimo **e** non è
la stessa cosa che insegnare una strada nuova.

C'è infine un argomento di prospettiva da tenere presente. Le critiche fin qui
riguardano il rinforzo **come unico maestro**, cioè l'idea di
costruire un sistema per tentativi ed errori a partire da zero. Nessuno dei
due critici propone questo: la ciliegina è una ciliegina perché viene
**dopo**, su una torta già cotta. Criticare il rinforzo come unico maestro non
è quindi criticarlo come rifinitura, e le due frasi si somigliano tanto che
vengono scambiate di continuo.

## Cambiare la quantità, invece di arricchire il premio

Tutte queste critiche hanno la stessa forma: il segnale è povero.
La risposta più ovvia è allora arricchirlo, e infatti è quello che si fa (un
critico che dà un voto passo per passo, un premio interno per la novità, un
giudice che commenta le soluzioni parziali). C'è però una risposta diversa, che
non arricchisce il premio ma **cambia la quantità che si sta ottimizzando**, e
merita un posto qui perché è l'unica obiezione di principio, non di rimedio.

Viene da fuori dall'informatica. Nelle neuroscienze teoriche esiste un quadro,
l’**inferenza attiva**, che descrive percezione, pianificazione e azione come
un unico problema di inferenza {cite}`parr2022active`. Al posto della
ricompensa mette una grandezza da minimizzare, l’**energia libera attesa**: la
sorella rivolta al futuro dell'energia libera variazionale nominata in
apertura, che pesa le osservazioni già arrivate mentre questa pesa quelle che
un piano farebbe arrivare. Il punto che interessa qui è come è fatta, perché si
scompone in due pezzi.

`````{tab} Elementare

Uno vuole un caffè, e l'esempio è degli autori. In città ci sono due buoni bar:
uno apre dal lunedì al venerdì, l'altro solo nel fine settimana. Lui però non sa
che giorno è.

Che cosa fa per prima cosa? Non va a un bar: guarda il calendario. È un'azione
che non gli porta nessun caffè, e nemmeno un passo verso il caffè: gli porta
soltanto *informazione*. Solo dopo, sapendo che giorno è, va al bar giusto, e
quella seconda azione è quella che gli porta la cosa che voleva.

Le due azioni valgono per ragioni diverse: la prima **risolve un'incertezza**,
la seconda **realizza una preferenza**. Ed è qui il punto: un sistema che sappia
valutare solo la seconda non ha modo di scegliere il calendario, perché il
calendario non porta caffè. Può solo tirare a caso fra i due bar, e come dicono
gli autori, spesso il caffè non se lo beve.

Il rinforzo classico è in quella situazione. La ricompensa dice quanto ti è
andata bene, non quanto hai imparato; e per questo, quando l'esplorazione serve,
gliela si deve pagare a parte, con un premio aggiunto apposta. Nell'inferenza
attiva non si paga niente in più, perché il valore di sapere era già dentro la
quantità da minimizzare, accanto al valore di ottenere.

E quel premio aggiunto apposta è proprio quello che resta al nostro uomo se gli
si toglie la voglia di caffè: guarda il calendario per il gusto di sapere che
giorno è, e nient'altro. La curiosità che si paga a parte, insomma, è il caso in
cui il caffè non c'è.

`````

`````{tab} Superiore

L’**energia libera attesa** di una politica $\pi$ si scompone in

$$
G(\pi) \;=\; -\,\underbrace{I(\pi)}_{\text{valore epistemico}}
\;-\;\underbrace{\mathbb{E}_{Q(\tilde{o}\mid\pi)}\big[\ln P(\tilde{o} \mid C)\big]}_{\text{valore pragmatico}},
$$

dove $\tilde{o}$ sono le osservazioni future attese sotto $\pi$, $I(\pi)$ è il
guadagno di informazione atteso, cioè di quanto quelle osservazioni
ridurrebbero l'incertezza sugli stati, e $C$ codifica le preferenze
dell'agente, cioè quali osservazioni si aspetta di incontrare. Quel $C$ è la
lettera con cui la letteratura dell'inferenza attiva scrive le preferenze, e
non è il compressore $C$ di «Capire è accorciare». Minimizzare $G$ significa
massimizzare tutti e due i pezzi insieme.

La conseguenza è quella che gli autori enunciano esplicitamente: poiché
«l'utilità e il valore dell'informazione emergono come due componenti
dell'energia libera attesa», non c'è nessun compromesso fra esplorazione e
sfruttamento da regolare a mano, perché «entrambe sono al servizio
dell'ottimizzazione della stessa funzione» {cite}`parr2022active`.

Il quadro ha inoltre una proprietà che aiuta a collocarlo: togliendo pezzi a
$G$ si riottengono schemi già noti. Tolte le preferenze, cioè annullato il
valore pragmatico, quel che resta, cambiato di segno, «è variamente noto come
sorpresa bayesiana attesa o **motivazione intrinseca**», che è esattamente la
curiosità che la sezione sull'esplorazione, nel capitolo sul deep reinforcement
learning, costruisce come bonus aggiunto. Il che dice in che rapporto stanno le
due letture: non sono in concorrenza, una è il caso particolare dell'altra.

`````

Due avvertenze prima di lasciare l'argomento, perché è il punto in cui sarebbe
facile promettere troppo. La prima: **non è così che si addestrano oggi i
sistemi di cui si è parlato fin qui.** L'inferenza attiva nasce come teoria del
comportamento biologico, e le sue realizzazioni sono modelli di laboratorio,
non i sistemi su cui gira il mondo. La seconda la dicono gli autori
stessi in apertura, e conviene riportarla perché evita di trasformarli in
avversari di qualcuno: il quadro «non mira a rimpiazzare altri quadri di
riferimento, come la psicologia comportamentale, la teoria delle decisioni e
l'apprendimento per rinforzo», ma a comprenderli.

Serve dunque a quello per cui l'abbiamo chiamato in causa: mostrare che la
povertà del segnale non è una fatalità dell'imparare agendo, ma la conseguenza
di *quale* quantità si è scelto di ottimizzare. E l'oggetto della discordia, a
ben vedere, non è se il rinforzo serva, ma se basti a spiegare da dove venga la
comprensione. Su quella domanda si apre il {doc}`capitolo sui world model </WorldModels/overview>`.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Nella torta di LeCun l'apprendimento per rinforzo è la **ciliegina**, e la
  ragione è quella del conto sui bit: il segnale che porta è pochissimo, una
  frase per un'intera giornata.
- A quella si aggiunge una seconda obiezione, che è peggiore perché **moltiplica**
  la prima: quella frase va anche **distribuita** fra le centinaia di cose fatte
  durante la giornata, e nessuno dice quali erano quelle buone. Se la giornata è
  andata bene si tende a ripetere tutto, compresi gli sbagli.
- Karpathy lo dice così: si sta **aspirando la supervisione attraverso una
  cannuccia**, e quel poco lo si spalma su tutto quello che si è fatto. Nella
  stessa risposta però dice anche che il rinforzo è oggi il meglio che si abbia,
  e che fa scoprire soluzioni che una persona non troverebbe mai.
- Una tesi che gira col nome sbagliato: che il rinforzo **restringa** invece di
  allargare le capacità del modello non l'ha detta Karpathy, l'hanno **misurata**
  Yue e colleghi, chiedendo al modello tante risposte alla stessa domanda invece
  di una. Con una sola risposta vince il modello addestrato; con moltissime
  vince quello di partenza. Vale per come questo addestramento si fa oggi:
  spinto a un numero assurdo di tentativi il confronto non direbbe più niente,
  perché la risposta giusta salterebbe fuori anche battendo i tasti a caso.
- Il contraddittorio esiste ed è forte: gli assistenti che usiamo tutti i
  giorni sono rifiniti proprio così, e i modelli «ragionanti» si addestrano così
  sui problemi dove la risposta si può verificare. Nessuno propone di buttare il
  rinforzo: si discute se basti a spiegare da dove venga la comprensione.
- LeCun non dice che sia inutile: dice che è **nel posto sbagliato**, buono per
  rifinire e non per costruire la comprensione, che va costruita guardando. È
  una posizione dentro un dibattito aperto, e le macchine fatte così non sono
  ancora quelle che funzionano meglio.
- C'è poi una risposta di tipo diverso, che non arricchisce il premio ma
  **cambia la cosa da massimizzare**: chi vuole un caffè e non sa che giorno è
  guarda prima il calendario, e quel gesto non porta caffè, porta informazione.
  Se il valore di sapere sta già dentro la quantità da minimizzare,
  l'esplorazione non si paga a parte. Non è però così che i sistemi di oggi si
  addestrano.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Le critiche al rinforzo «puro» sono quattro; due sono trattate altrove (la
  ricompensa scritta male e ottimizzata alla lettera, e il costo in esperienza
  dell'imparare per tentativi nel mondo), e le due discusse qui sono due mosse
  dello stesso argomento.
- **Quanti bit**: il bersaglio è uno scalare per episodio, cioè pochi bit,
  contro l'ordine di $10^5$ bit per esempio del pre-addestramento
  auto-supervisionato.
- **Assegnazione del credito**: nel gradiente di policy elementare lo stesso
  ritorno $R^{(m)}$ moltiplica $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ per
  ogni $t$ della traiettoria. La linea di base riduce la varianza, il critico
  stima il vantaggio azione per azione e lo sconto accorcia l'orizzonte di
  attribuzione; nessuno dei tre fabbrica informazione assente dal segnale.
- **Karpathy** {cite}`karpathy2025dwarkesh`: «sucking supervision through a
  straw», e il ritorno finale «trasmesso per radiodiffusione a tutta la
  traiettoria». Nella stessa risposta: il rinforzo resta il meglio disponibile e
  «può scoprire soluzioni che una persona non troverebbe mai».
- **Attribuzione da tenere dritta**: la tesi che l'RLVR restringa il confine
  delle capacità è di **Yue e colleghi** {cite}`yue2025rlvr`, misurata col
  **pass@$k$**: i modelli addestrati vincono a $k$ piccolo, i modelli base a $k$
  grande. Vale per l'impostazione corrente, non in linea di principio.
- Il contraddittorio: post-addestramento sulle preferenze e RLVR sui domini
  verificabili funzionano, e non contraddicono il risultato precedente. Spostare
  massa di probabilità verso strade che pagano è utile **e** non equivale a
  insegnare una strada nuova.
- **La proposta di LeCun**: il rinforzo va sostituito, come costruttore della
  comprensione, dal **controllo predittivo basato su modello**, e resta per la
  rifinitura. È una posizione dentro un dibattito aperto, non un risultato.
- **Inferenza attiva** {cite}`parr2022active`: l'energia libera attesa $G(\pi)$
  si scompone in valore epistemico e valore pragmatico, quindi non c'è nessun
  compromesso fra esplorazione e sfruttamento da regolare a mano; tolte le
  preferenze resta, cambiata di segno, la **motivazione intrinseca**, cioè il
  bonus di curiosità come caso particolare. Quadro di laboratorio, non il modo
  in cui i sistemi si addestrano oggi.
```

`````

Se ne esce con un criterio più che con una risposta: prima di
chiedersi se un modo di addestrare funzioni, conviene chiedersi quanta
informazione porta il segnale su cui si regge, e a quante scelte quel poco va
poi diviso. Nel {doc}`capitolo sui world model </WorldModels/overview>` il segnale resta la previsione, ma la
cosa da prevedere diventa quello che succede dopo nel mondo, invece della parte
coperta di un dato.
