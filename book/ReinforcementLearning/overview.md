# Reinforcement Learning: imparare per tentativi

Nessuno insegna a un bambino a camminare elencandogli la sequenza esatta di
contrazioni muscolari. Il bambino ci prova, oscilla, cade, si rialza, fa un
passo e cade di nuovo. Ogni tentativo il mondo gli restituisce un giudizio
implicito (un tonfo doloroso oppure un metro guadagnato verso il divano) e
settimana dopo settimana quel giudizio scolpisce un modo di muoversi che
nessun manuale ha mai descritto. Questo è, in una frase, il **reinforcement
learning** (apprendimento per rinforzo, spesso abbreviato in RL): imparare a
comportarsi non da esempi già etichettati, ma dalle conseguenze delle proprie
azioni.

## Agente, ambiente, ricompensa

Tre ingredienti bastano a definire l'intero paradigma. C'è un **agente**, cioè
chi decide (il bambino, un robot, un programma che gioca a scacchi). C'è un
**ambiente**, cioè tutto il resto: il mondo che l'agente non controlla ma con
cui interagisce. E c'è una **ricompensa**, un numero che l'ambiente restituisce
per dire "bene" o "male". L'agente osserva lo stato del mondo, sceglie
un'azione, l'ambiente cambia stato e gli passa una ricompensa; poi il ciclo
ricomincia ({numref}`fig-rl-ciclo`).

```{figure} ../figures/rl-ciclo-interazione.svg
:name: fig-rl-ciclo
:alt: Due riquadri, Agente e Ambiente, collegati da due frecce che formano un anello. La freccia superiore va dall'Agente all'Ambiente ed è etichettata Azione. La freccia inferiore torna dall'Ambiente all'Agente ed è etichettata Nuovo stato e Ricompensa.
:width: 90%

Il ciclo di interazione del reinforcement learning: l'agente compie un'azione,
l'ambiente risponde con un nuovo stato e una ricompensa, e l'anello si richiude.
```

`````{tab} Elementare

Immagina un videogioco. A ogni istante vedi lo schermo (lo **stato**),
premi un tasto (l'**azione**) e il gioco reagisce: nuova schermata e magari
qualche punto in più o in meno (la **ricompensa**). L'obiettivo non è indovinare
il tasto "giusto" in questo istante, ma accumulare più punti possibile fino alla
fine della partita. La strategia con cui scegli i tasti, momento per momento, si
chiama **politica** (*policy*): è tutto ciò che l'agente cerca di migliorare.

`````

`````{tab} Superiore

Formalmente l'interazione è un **processo decisionale di Markov** (*Markov
Decision Process*, MDP). A ogni passo $t$ l'agente osserva lo stato $S_t$,
sceglie un'azione $A_t$ secondo la sua politica $\pi(a \mid s)$, e l'ambiente
transita in $S_{t+1}$ restituendo una ricompensa scalare $R_{t+1}$ (maiuscole
per le variabili aleatorie, minuscole $s$, $a$, $r$ per i loro valori: è la
convenzione che terremo in tutto il capitolo). L'obiettivo è massimizzare non
la ricompensa immediata ma il **ritorno** (*return*) scontato:

$$
G_t = \sum_{k=0}^{\infty} \gamma^{k}\, R_{t+1+k},
\qquad \gamma \in [0,1) .
$$

Qui $\gamma$ è il **fattore di sconto**: vicino a $1$ l'agente è lungimirante e
dà peso al futuro lontano; vicino a $0$ è miope e insegue solo il premio
immediato. Cercare la politica ottima $\pi^{*}$ significa massimizzare
$\mathbb{E}_\pi[G_t]$, e quasi tutti gli algoritmi lo fanno stimando funzioni di
valore come $Q^\pi(s,a) = \mathbb{E}_\pi[G_t \mid S_t=s,\, A_t=a]$, il ritorno
atteso partendo da $s$, giocando $a$ e poi seguendo $\pi$.

`````

## Cosa lo distingue dall'apprendimento supervisionato

Nell'apprendimento supervisionato ogni esempio arriva con la sua risposta
corretta: questa foto è un gatto, quella casa vale 300 000 euro. Il modello deve
solo imitare le etichette. Nel reinforcement learning quelle etichette non
esistono.

`````{tab} Elementare

Nessuno dice mai all'agente "la mossa giusta era questa". Riceve solo una
ricompensa che dice *quanto è andata bene*, non *cosa avrebbe dovuto fare*. E il
punto più difficile è che la ricompensa spesso arriva **tardi**: negli scacchi
capisci di aver sbagliato solo trenta mosse dopo, quando perdi. Quale mossa
ringraziare per la vittoria? Quale incolpare per la sconfitta? Questo si chiama
problema dell'**assegnazione del merito** (*credit assignment*), ed è il cuore
di tutta la difficoltà.

`````

`````{tab} Superiore

La differenza è strutturale. Nel supervisionato i dati $(x, y)$ sono
indipendenti e l'etichetta $y$ è il segnale di errore diretto. Nel RL il segnale
è una ricompensa scalare, potenzialmente **ritardata** e **sparsa**, e i dati
non sono indipendenti: l'azione di adesso determina lo stato successivo, quindi
la distribuzione degli esempi dipende dalla politica stessa. L'assegnazione del
merito temporale si affronta con i metodi a **differenza temporale**
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

C'è una tensione che nessun agente può ignorare, e che riconosciamo tutti.

`````{tab} Elementare

Hai un ristorante preferito che non delude mai. Provi quello nuovo appena aperto
o resti sul sicuro? Se scegli sempre il noto (**sfruttamento**), non scoprirai
mai un posto migliore. Se provi sempre cose nuove (**esplorazione**), sprechi
serate in locali mediocri. Un buon agente fa entrambe le cose: sfrutta ciò che
sa quasi sempre, ma ogni tanto azzarda, perché solo azzardando può scoprire
ricompense che non sospettava.

`````

`````{tab} Superiore

Il compromesso si formalizza con strategie come **$\varepsilon$-greedy**: con
probabilità $1-\varepsilon$ l'agente sceglie l'azione stimata migliore, con
probabilità $\varepsilon$ ne prende una a caso.

$$
A_t =
\begin{cases}
\arg\max_a Q(S_t, a) & \text{con probabilità } 1-\varepsilon, \\
\text{azione casuale} & \text{con probabilità } \varepsilon .
\end{cases}
$$

In pratica $\varepsilon$ parte alto e decresce nel tempo: si esplora molto
all'inizio, quando le stime di $Q$ sono grezze, e si sfrutta sempre di più man
mano che diventano affidabili. Approcci più raffinati (*softmax*, *Upper
Confidence Bound*, bonus di curiosità) dosano l'esplorazione in base
all'incertezza invece che a caso.

`````

## Tre tappe che hanno fatto la storia

Il RL non è un'idea nuova, ma ha avuto pochi momenti che ne hanno mostrato la
potenza. Nei primi anni Novanta, all'IBM, Gerald Tesauro costruì
**TD-Gammon**: una rete neurale addestrata con la *differenza temporale* (il
metodo, che vedremo, del correggere le proprie previsioni un passo alla volta)
che imparò a giocare a backgammon quasi al livello dei campioni umani,
giocando oltre un milione di partite contro sé stessa e scoprendo aperture che
i maestri poi adottarono. Nel 2015 DeepMind pubblicò su *Nature* il **DQN**,
un agente che imparava a giocare a decine di videogiochi **Atari** partendo
dai soli pixel dello schermo e dal punteggio, senza sapere nulla delle regole.
E nel marzo 2016 **AlphaGo** batté 4-1 il campione Lee Sedol al Go, un gioco
considerato fuori portata per le macchine a causa del suo spazio combinatorio
immenso: un risultato che molti si aspettavano lontano un decennio.

## Come è organizzato questo capitolo

Cominciamo da una versione del problema spogliata di tutto tranne il dilemma
appena descritto: i **bandit a più braccia**, dove non c'è nessuno stato e
l'unica domanda è quale leva tirare. Poi rimettiamo lo stato al suo posto con
le fondamenta appena introdotte (MDP, ritorno, funzioni di valore) e le
rendiamo algoritmo con i metodi classici (programmazione dinamica, Monte
Carlo, differenza temporale, Q-learning). Da lì il passo verso il **deep
reinforcement learning**, dove reti neurali stimano le funzioni di valore o
direttamente la politica, è naturale: lo affrontiamo nel capitolo successivo,
ricostruendo proprio il tipo di agente che ha imparato a giocare a partire dai
pixel. L'obiettivo non è collezionare sigle, ma capire un'unica idea da tutte
le angolazioni: come si impara a decidere quando l'unico maestro è
l'esperienza.

```{admonition} Da ricordare
:class: important
- Il RL si regge su tre elementi: un **agente** che decide, un **ambiente** che
  risponde, una **ricompensa** che valuta. Il ciclo azione → nuovo stato +
  ricompensa si ripete.
- A differenza del supervisionato non ci sono etichette, ma un segnale scalare
  spesso **ritardato**: da qui il problema dell'assegnazione del merito.
- L'obiettivo è massimizzare il **ritorno** scontato $G_t$, non la ricompensa
  immediata; il fattore $\gamma$ regola quanto conta il futuro.
- Ogni agente deve bilanciare **esplorazione** e **sfruttamento** (per esempio
  con $\varepsilon$-greedy).
- Tappe simbolo: **TD-Gammon** (backgammon), **DQN** sui giochi **Atari**,
  **AlphaGo** (Go, 2016).
```
