# Reinforcement Learning: imparare per tentativi

```{image} ../figures/aperture/reinforcement-learning.png
:class: pt-apertura only-light
:width: 100%
:alt: Un bambino muove il primo passo con le braccia aperte per stare in equilibrio, e dietro di lui una fila di piccole impronte irregolari.
```

```{image} ../figures/aperture/reinforcement-learning-scura.png
:class: pt-apertura only-dark
:width: 100%
:alt: Un bambino muove il primo passo con le braccia aperte per stare in equilibrio, e dietro di lui una fila di piccole impronte irregolari.
```

Nessuno insegna a un bambino a camminare elencandogli la sequenza esatta di
contrazioni muscolari. Il bambino ci prova, oscilla, cade, si rialza, fa un
passo e cade di nuovo. A ogni tentativo il mondo gli restituisce un giudizio
implicito (un tonfo doloroso oppure un metro guadagnato verso il divano), e
settimana dopo settimana quel giudizio scolpisce un modo di muoversi che
nessun manuale ha mai descritto. Questo è, in una frase, il reinforcement
learning (apprendimento per rinforzo, spesso abbreviato in RL), cioè imparare
a comportarsi dalle conseguenze delle proprie azioni, senza esempi già
etichettati.

Il nome si è già incontrato in fondo al {doc}`capitolo sulla ricerca
</Ricerca/quando-il-mondo-non-si-conosce>`, come la strada che resta quando le
regole del mondo non si possono più interrogare. Il bambino si trova in quella
condizione: non può provare un passo nella propria testa per vedere dove lo
porta, ma deve farlo, e scopre dopo che cosa succede.

## Agente, ambiente, ricompensa

Il problema ha tre protagonisti. L'agente è chi decide, come il bambino, un
robot o un programma che gioca a scacchi. L'ambiente è tutto il resto, cioè
il mondo che l'agente non controlla ma con cui interagisce. La ricompensa è un
numero che l'ambiente restituisce dopo ogni azione, per dire quanto è andata
bene.

A ogni passo l'agente osserva la situazione in cui si trova e compie
un’azione. Questa situazione si chiama stato, ed è la descrizione del mondo su
cui l'agente basa la scelta. In risposta l'ambiente passa a un nuovo stato e
consegna all'agente una ricompensa, poi il ciclo ricomincia
({numref}`fig-rl-ciclo`). L'agente non cerca di rendere più grande possibile
la ricompensa di adesso, ma la somma di tutte quelle che verranno da qui alla
fine; questa somma si chiama **ritorno** (in inglese *return*).

Nella somma i premi lontani entrano ridotti: ogni passo di attesa li
moltiplica per uno stesso numero $\gamma$ compreso fra zero e uno, detto
**fattore di sconto**. Con $\gamma$ minore di uno, e premi che non crescono
senza limite, la somma resta un numero finito anche su una partita che non
finisce mai, come dimostra il calcolo della sezione {doc}`MDP e funzioni valore
<mdp-valore>`.

```{figure} ../figures/rl-ciclo-interazione.svg
:name: fig-rl-ciclo
:alt: Due riquadri, Agente e Ambiente, collegati da due frecce che formano un anello. La freccia superiore va dall'Agente all'Ambiente ed è etichettata Azione. La freccia inferiore torna dall'Ambiente all'Agente ed è etichettata Nuovo stato e Ricompensa.
:width: 90%

Il ciclo di interazione del reinforcement learning: l'agente compie un'azione,
l'ambiente risponde con un nuovo stato e una ricompensa, e l'anello si richiude.
Accanto a ogni etichetta il disegno mette una lettera con un numeretto in
basso: la lettera è l'iniziale ($a$ per azione, $s$ per stato, $r$ per
ricompensa) e il numeretto indica il passo. L'azione si compie al passo $t$,
lo stato nuovo e la ricompensa arrivano subito dopo, al passo $t+1$.
```

Una domanda viene prima di ogni algoritmo: chi decide la ricompensa? Non
l'agente, e nemmeno il mondo: la scrive chi imposta il problema. In un
videogioco il punteggio esiste già e si prende quello; per un robot che deve
imparare a camminare qualcuno deve stabilire che cadere vale $-5$ e che un
metro guadagnato vale $+1$. È una scelta di progetto seria, perché un agente
ottimizza i numeri che gli sono stati dati e non le intenzioni di chi glieli ha
dati. Se lo si premia per la velocità, ad esempio, può imparare a buttarsi in
avanti e cadere in fretta.

Chi imposta il problema può anche aggiungere premi intermedi, per guidare
l'agente invece di lasciarlo cercare a vuoto; la tecnica si chiama *reward
shaping*, «dare forma alla ricompensa». È però un'arma a doppio taglio, perché
un premio intermedio può cambiare quale sia il comportamento migliore senza che
nessuno se ne accorga. Premiato per ogni passo che lo avvicina all'uscita, un
robot può scoprire che gli conviene fare avanti e indietro davanti alla porta,
incassando a ogni andata. Una sola forma di premio aggiuntivo lascia intatta la
strategia migliore in qualunque problema {cite}`ng1999policy`, e la racconta la
sezione su {doc}`esplorazione e ricompensa
</DeepReinforcementLearning/esplorazione-e-ricompensa>`.

La regola con cui l'agente sceglie l'azione in ogni stato si chiama
**politica**, o all'inglese *policy*, e si indica con $\pi$. Imparare, nel
reinforcement learning, consiste quindi nel trovare una politica che renda il
più grande possibile il ritorno medio.

`````{tab} Elementare

In un videogioco vedi lo schermo, premi un tasto e il gioco risponde: schermata
nuova, e qualche punto in più o in meno. Lo schermo è lo stato, il tasto è
l'azione, i punti sono la ricompensa. Il numero da far crescere è il totale che
avrai a fine partita, e quello è il ritorno; il punteggio di questo istante
conta solo per quanto aggiunge al totale.

Qualunque cosa si chieda all'agente, quindi, deve stare in quel numero. Se lo
si vuole veloce e anche prudente, velocità e prudenza vanno messe
insieme in un punteggio solo, e quanto conti l'una rispetto all'altra lo decide
chi imposta il gioco.

Dentro il totale i punti lontani possono pesare meno di quelli vicini. Quanto
contano lo decide una manopola, il fattore di sconto, che chi imposta il
problema gira prima che l'agente cominci a giocare. Tenuta al minimo, l'agente
raccoglie tutte le monete che ha sotto il naso e ignora la chiave in fondo al
livello. Tenuta al massimo, passa oltre le monete e va a prendere la chiave,
perché dietro la porta ce n'è molto di più.

Tenerla al massimo, però, crea un problema nei giochi senza traguardo, quelli
in cui si corre, si raccolgono monete e, finché non si sbaglia, si va avanti
per sempre. Se un punto fra mille schermate contasse quanto un punto adesso,
"quanti ne prendo in tutto" resterebbe senza risposta per chi gioca bene,
perché la sua partita non finisce e il totale non si ferma su nessun numero.
Nelle partite che a un certo punto finiscono il problema non si pone: i punti
da sommare sono contati, e la manopola si può tenere al massimo senza rischi.

Con l'esperienza l'agente si costruisce un voto: per ogni schermata e per ogni
tasto, quanti punti gli frutterà in tutto premerlo lì e poi tirare avanti con
le solite abitudini. Quelle abitudini ("in questa schermata salto sempre") sono
la politica, e si migliorano a turno con i voti.

Si può cominciare da abitudini qualsiasi, anche tirate a sorte. I voti dicono
quanto rendono, le abitudini si spostano verso i tasti coi voti più alti, e per
le abitudini nuove si rifanno i voti. A ogni giro migliorano gli uni e le
altre.

`````

`````{tab} Superiore

Formalmente l'interazione è un processo decisionale di Markov (*Markov Decision
Process*, MDP). A ogni passo $t$ l'agente osserva lo stato $S_t$, sceglie
un'azione $A_t$ secondo la sua politica $\pi(a \mid s)$, e l'ambiente transita
in $S_{t+1}$ restituendo una ricompensa scalare $R_{t+1}$. Le maiuscole
indicano le variabili aleatorie, le minuscole $s$, $a$, $r$ i valori che
assumono. L'obiettivo è massimizzare non la ricompensa immediata ma il ritorno
(*return*) scontato:

$$
G_t = \sum_{k=0}^{\infty} \gamma^{k}\, R_{t+1+k},
\qquad \gamma \in [0,1] .
$$

Dietro la scelta c'è un'ipotesi, che Sutton e Barto chiamano *ipotesi della
ricompensa* (*reward hypothesis*): che ogni scopo si possa esprimere come
massimizzazione del valore atteso della somma di un segnale scalare
{cite}`sutton2018reinforcement`. Come ogni ipotesi, può cadere: obiettivi
multipli o vincoli da rispettare vi rientrano solo scegliendo come
combinarli in un numero solo, e quella scelta fa parte del progetto.

Nella formula del ritorno, $\gamma$ è il fattore di sconto: vicino a $1$
l'agente è lungimirante e dà peso al futuro lontano; vicino a $0$ è miope e
insegue solo il premio immediato. Nei compiti **continui**, che non terminano
mai, la strada standard è $\gamma < 1$, perché senza sconto quella somma
infinita può non convergere (basta una ricompensa costante); l'alternativa, in
quei compiti, è cambiare criterio e massimizzare la ricompensa media per passo.
Nei compiti **episodici**, che prima o poi terminano, la somma ha un numero
finito di termini e $\gamma = 1$ è ammesso, e molto usato.

Cercare la politica ottima $\pi^{*}$ equivale a massimizzare
$\mathbb{E}_\pi[G_t]$. Molti algoritmi lo fanno stimando funzioni di valore
come $Q^\pi(s,a) = \mathbb{E}_\pi[G_t \mid S_t=s,\, A_t=a]$, cioè il ritorno
atteso partendo da $s$, giocando $a$ e poi seguendo $\pi$. A partire da queste
stime migliorano la politica, scegliendo in ogni stato l'azione di valore
stimato più alto. Altri metodi, come il gradiente di policy, ottimizzano invece
$\pi$ in modo diretto.

`````

## Cosa lo distingue dall'apprendimento supervisionato

Nell'apprendimento supervisionato ogni esempio arriva con la sua risposta
corretta, la sua etichetta: questa foto è un gatto, quella casa vale
300 000 euro. Il programma deve solo imparare a imitare quelle risposte. Nel
reinforcement learning, invece, le risposte corrette non esistono: nessuno le
conosce.

`````{tab} Elementare

Nessuno dice mai all'agente "la mossa giusta era questa". Riceve solo una
ricompensa che dice *quanto è andata bene*, non *cosa avrebbe dovuto fare*. Il
punto più difficile è il ritardo. Negli scacchi, ad esempio, il punteggio
resta a zero per tutta la partita, e che una mossa fosse un errore lo capisci
trenta mosse dopo, quando perdi. Quale ringraziare per la vittoria? Quale
incolpare per la sconfitta? È il problema dell’**assegnazione del merito**
(*credit assignment*), la difficoltà centrale del campo.

C'è una seconda differenza, meno vistosa e non meno seria. Chi impara a
riconoscere i gatti riceve un mazzo di foto già pronto, e quel mazzo resta lo
stesso sia che risponda bene sia che risponda male. Chi impara a giocare a
scacchi si fabbrica il suo mazzo muovendo: se apre sempre allo stesso modo,
per anni vedrà le stesse posizioni e delle altre non saprà niente. In altre
parole, le partite su cui ci si allena dipendono dal modo in cui si gioca, e
appena quel modo cambia cambiano anche loro.

Aspettare la fine della partita per sapere chi ringraziare è troppo lento. Si
può invece dare a ogni mossa, in ogni posizione, un voto provvisorio, anche
grossolano: quanti punti mi aspetto di raccogliere da lì in avanti se la
faccio. È il voto delle posizioni di mezzo del capitolo sulla ricerca, ma qui
nessuno lo scrive a mano, perché lo si impara giocando.

Fatta una mossa, si sommano i punti guadagnati subito e il voto migliore fra
quelli delle mosse possibili dalla posizione nuova, ridotto dal fattore di
sconto perché quei punti arriveranno un passo più tardi. Quel totale corregge
il voto della mossa appena fatta. Nessuno ha aspettato l'ultima mossa, eppure
il giudizio finale, quando arriva, risale indietro di posizione in posizione
fino all'apertura.

`````

`````{tab} Superiore

La differenza è strutturale. Nel supervisionato i dati $(\mathbf{x}, y)$ sono
indipendenti e l'etichetta $y$ è il segnale di errore diretto. Nel RL il segnale
è una ricompensa scalare, potenzialmente ritardata e sparsa, e i dati non sono
indipendenti. L'azione corrente, infatti, determina lo stato successivo, quindi
la distribuzione degli esempi dipende dalla politica stessa. L'assegnazione del
merito temporale si affronta con i metodi a differenza temporale
(*temporal-difference*, TD) e con l'equazione di Bellman, che spezza il ritorno
in premio immediato più valore scontato dello stato futuro:

$$
Q^{*}(s,a) = \mathbb{E}\!\left[\, R_{t+1} + \gamma \max_{a'} Q^{*}(S_{t+1}, a')
\;\middle|\; S_t = s,\ A_t = a \,\right] .
$$

Aggiornare $Q$ verso il lato destro di questa uguaglianza è, in sostanza, ciò
che fanno *Q-learning* e le sue versioni profonde.

`````

## Il dilemma esplorazione–sfruttamento

C'è una tensione che nessun agente può ignorare e che riconosciamo tutti.

`````{tab} Elementare

Hai un ristorante preferito che non delude mai. Provi quello nuovo appena
aperto o resti sul sicuro? Se scegli sempre il noto (**sfruttamento**), non
scoprirai mai un posto migliore. Se provi sempre cose nuove (**esplorazione**),
sprechi serate in locali mediocri. Un buon agente fa entrambe le cose: sfrutta
quasi sempre ciò che sa, ma ogni tanto azzarda, perché solo azzardando può
scoprire ricompense che non sospettava.

Questa ricetta ha un nome, **$\varepsilon$-greedy**, che si legge
"epsilon-greedy". *Greedy* è l'inglese per "avido", cioè chi sceglie sempre
l'opzione che al momento sembra la migliore; ed $\varepsilon$ (epsilon) è la
piccola probabilità con cui invece si azzarda, per esempio una volta su dieci.

Quanto grande tenere quella probabilità dipende da quanto conosci la città.
Appena trasferito non hai un preferito da difendere, e i giudizi che ti sei
fatto valgono poco, quindi tanto vale provare quasi ogni sera un posto diverso.
Dopo un anno di cene i giudizi sono solidi, e continuare a tirare a caso una
sera su dieci diventa uno spreco di serate. Per questo la quota di azzardo
parte alta e si abbassa man mano che si impara.

L'azzardo, poi, si può dosare meglio di un sorteggio. Fra due posti che finora
sono sembrati ugualmente buoni si torna volentieri in quello provato meno
volte, perché lì una sorpresa è ancora possibile; e fra tutti gli altri
si torna più spesso in quelli che promettono di più, invece di trattarli tutti
allo stesso modo.

`````

`````{tab} Superiore

Il compromesso si formalizza con strategie come $\varepsilon$-greedy: con
probabilità $1-\varepsilon$ l'agente sceglie l'azione stimata migliore, con
probabilità $\varepsilon$ ne prende una a caso.

$$
A_t =
\begin{cases}
\arg\max_a Q(S_t, a) & \text{con probabilità } 1-\varepsilon, \\
\text{azione casuale} & \text{con probabilità } \varepsilon .
\end{cases}
$$

In pratica $\varepsilon$ parte alto e decresce nel tempo; in questo modo si
esplora molto all'inizio, quando le stime di $Q$ sono grezze, e si sfrutta
sempre di più man mano che diventano affidabili. Approcci più raffinati non
affidano l'esplorazione a una moneta: *softmax* distribuisce la probabilità in
base ai valori stimati, dando più peso alle azioni che promettono di più,
mentre *Upper Confidence Bound* e i bonus di curiosità privilegiano le azioni
su cui la stima è più incerta.

`````

## Da TD-Gammon in poi

Il RL non è un'idea nuova, ma ha avuto pochi momenti che ne hanno mostrato la
potenza. Nei primi anni Novanta, all'IBM, Gerald Tesauro costruì **TD-Gammon**
{cite}`tesauro1995temporal`, un programma che imparò a giocare a backgammon (il
gioco da tavolo con le pedine e i dadi) quasi al livello dei campioni umani.

Una rete neurale stimava, da ogni posizione, la probabilità di vincere, e dopo
ogni mossa la stima della posizione appena lasciata veniva avvicinata a quella
della posizione nuova. È l'apprendimento per differenze temporali, nella
variante TD($\lambda$) che la sezione sul {doc}`Q-learning e le differenze
temporali <q-learning>` ricostruisce.

Il programma giocava le partite contro sé stesso, muovendo da tutte e due le
parti, e l'unico segnale esterno era l'esito. Nessuno gli diceva quale mossa
fosse buona, ma alla fine uno dei due lati aveva vinto, e quel giudizio bastava
per capire quali stime rivedere. Le versioni migliori si allenarono su un
milione e mezzo di partite, e ne uscirono aperture che i maestri poi
adottarono.

Le due tappe successive sono il DQN, che nel 2015 ha imparato i videogiochi
Atari dai soli pixel dello schermo, e AlphaGo, che nel 2016 ha battuto Lee
Sedol al Go. Tutte e due richiedono reti profonde al posto delle tabelle, e
aprono il {doc}`capitolo sul Deep Reinforcement Learning
</DeepReinforcementLearning/overview>`.

## Dai bandit ai valori degli stati

Cominciamo da una versione del problema spogliata di tutto tranne il dilemma
fra esplorazione e sfruttamento: i bandit a più braccia, dove la situazione non
cambia mai e l'unica domanda è quale leva tirare. Poi rimettiamo al suo posto
la situazione che cambia, e con essa la domanda che tiene insieme tutto il
resto: quanto vale trovarsi in una certa situazione, cioè quanta ricompensa ci
si può aspettare di raccogliere da lì in avanti.

Dopo i bandit, tre metodi rispondono a quella domanda, in ordine di quanto
pretendono di sapere. Il primo pretende la mappa del mondo, cioè dove porta
ogni azione e quale ricompensa dà, e con la mappa in mano calcola quei valori
senza giocare nemmeno una partita. È quindi ancora pianificazione, come nel
capitolo sulla ricerca, e fissa i calcoli che gli altri due rifanno senza
mappa.

Il secondo non pretende niente: gioca partite intere e fa la media di com'è
andata. Il terzo non aspetta nemmeno la fine della partita e corregge a
ogni passo; è la strada dell'algoritmo più famoso del campo, il Q-learning.
Tutti e tre riempiono la stessa cosa, cioè una grande tabella con una casella
per ogni situazione in cui scrivere quanto vale.

Il passo verso il deep reinforcement learning, dove reti neurali stimano quei
valori o direttamente la politica, è una necessità e non un ampliamento
facoltativo. La tabella, infatti, diventa impraticabile appena le situazioni
possibili sono tante, e va sostituita da qualcosa che sappia indovinare il
valore di situazioni mai viste prima. Lo affronta il capitolo sul {doc}`deep
reinforcement learning </DeepReinforcementLearning/overview>`, che ricostruisce
proprio l'agente che ha imparato a giocare a partire dai pixel. Sotto tutte le
sigle c'è una domanda sola, guardata da angolazioni diverse: come si impara a
decidere quando l'unico maestro è l'esperienza.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Ci sono tre protagonisti: qualcuno che decide (l'agente), un mondo che
  risponde (l'ambiente) e un punteggio che dice quanto è andata bene (la
  ricompensa). Si agisce, il mondo cambia e paga, si ricomincia.
- Nessuno dice mai qual era la mossa giusta: il punteggio dice *quanto* è
  andata bene, non *cosa* si doveva fare. Spesso arriva tardi, molte mosse
  dopo quella che lo ha meritato: capire chi ringraziare è la difficoltà
  centrale.
- Quel che si vuole rendere grande non è il punteggio del momento ma il totale
  da qui alla fine (il ritorno). Di solito i premi lontani contano meno di
  quelli vicini, e nei giochi che non finiscono mai è la scelta normale: senza
  sconto il totale può non fermarsi su nessun numero.
- Chi decide i punti è chi imposta il problema, non il mondo: numeri scelti
  male insegnano il comportamento sbagliato.
- Bisogna sempre scegliere fra tornare dove si sa che si sta bene e provare
  qualcosa di nuovo (il dilemma del ristorante): quasi sempre il noto, ogni
  tanto una prova a caso.
- Tre risultati che hanno fatto la storia: TD-Gammon a backgammon, DQN
  sui vecchi videogiochi Atari, AlphaGo al Go nel 2016.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Il RL si regge su tre elementi: un agente che decide, un ambiente che
  risponde, una ricompensa che valuta. Il ciclo azione → nuovo stato +
  ricompensa si ripete.
- A differenza del supervisionato non ci sono etichette, ma un segnale scalare
  spesso ritardato: da qui il problema dell'assegnazione del merito.
- L'obiettivo è massimizzare il ritorno scontato $G_t$, non la ricompensa
  immediata; il fattore $\gamma$ regola quanto conta il futuro, e nei compiti
  continui si prende di norma $\gamma < 1$, perché senza sconto la somma può
  non convergere (l'alternativa è massimizzare la ricompensa media per passo).
- Ogni agente deve bilanciare esplorazione e sfruttamento (per esempio
  con $\varepsilon$-greedy).
- La funzione di ricompensa è una scelta di progetto, non un dato
  dell'ambiente: l'agente ottimizza ciò che è scritto, non ciò che si intendeva,
  e che ogni scopo stia in un segnale scalare è un'ipotesi (la *reward
  hypothesis*).
- Tappe simbolo: TD-Gammon (backgammon), DQN sui giochi Atari,
  AlphaGo (Go, 2016).
```

`````
