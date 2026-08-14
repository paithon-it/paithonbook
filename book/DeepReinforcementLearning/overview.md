# Deep Reinforcement Learning

Nel 2013 un laboratorio londinese ancora poco noto, DeepMind, pubblica un
risultato che sembra un giochino e invece è uno spartiacque: una rete neurale
impara a giocare a *Breakout*, il classico dei mattoncini dell'Atari 2600,
guardando **solo i pixel dello schermo** e il punteggio. Nessuno le ha spiegato
le regole, cosa sia la pallina, cosa sia la racchetta. Alla fine di un
addestramento che vale settimane di gioco non si limita a giocare bene: scopre
da sola la tattica di scavare un tunnel sul lato del muro per far rimbalzare la
pallina dietro i mattoni. Nessuno gliel'ha insegnata, e nel punteggio non c'era
scritta.

Nel capitolo precedente abbiamo visto il *reinforcement learning* classico:
un agente, degli stati, delle azioni, delle ricompense, e algoritmi come il
Q-learning che imparano *quanto vale* ogni mossa. Ma quel Q-learning teneva i
suoi conti in una **tabella**: una casella per ogni coppia stato-azione. Ed è
proprio la tabella a rompersi non appena il mondo diventa grande. Il *deep*
reinforcement learning nasce per sostituirla.

## Quando la tabella diventa impossibile

Il Q-learning tabellare funziona benissimo in un labirinto con cento caselle.
Ma proviamo a giocare guardando lo schermo, come farebbe una persona.

`````{tab} Elementare

Immagina di dover scrivere su un quaderno un giudizio ("questa mossa è buona"
oppure "è pessima") per **ogni possibile immagine** che può comparire sullo
schermo. Ma le immagini possibili sono praticamente infinite: basta che la
pallina si sposti di un pixel e la figura è già diversa, e quindi ti servirebbe
una nuova riga sul quaderno. Non ci sono abbastanza quaderni al mondo. La
tabella, che nel labirinto bastava, qui è semplicemente impossibile da
riempire e da conservare.

`````

`````{tab} Superiore

Un fotogramma di gioco, ridotto come nell'esperimento originale a $84\times84$
pixel in $256$ livelli di grigio, ha

$$
|\mathcal{S}| = 256^{\,84\times 84} = 256^{7056}
$$

stati possibili: un numero con migliaia di cifre, incommensurabilmente più
grande degli atomi dell'universo osservabile ($\sim 10^{80}$). Una $Q$-table
richiederebbe una cella per ciascuno stato $s$ e azione $a$: né la memoria né i
dati per visitarli tutti esisteranno mai. Il problema non è l'algoritmo, è la
**rappresentazione**: enumerare gli stati non scala.

`````

## L'idea: una rete al posto della tabella

La svolta concettuale è semplice da enunciare. Se non possiamo *elencare* il
valore di ogni schermata, proviamo a **calcolarlo sul momento** con qualcosa che
sappia **generalizzare**, cioè rispondere anche su un caso mai visto perché
somiglia a casi già visti: schermate simili dovrebbero ricevere giudizi simili.
E quale strumento sappiamo essere bravissimo a leggere immagini ed estrarne una
risposta? Una rete neurale, e in particolare una rete **convoluzionale**, il
tipo di rete costruito apposta per guardare immagini
({numref}`fig-drl-pixel-to-q`).

```{figure} ../figures/drl-pixel-to-q.svg
:name: fig-drl-pixel-to-q
:alt: I pixel dello schermo di gioco entrano in una rete neurale convoluzionale che produce un valore Q per ogni azione possibile; l'azione col valore più alto viene scelta.
:width: 90%

Dal pixel alla decisione. La rete riceve lo schermo grezzo e restituisce, in
un colpo solo, un voto per ciascuna mossa disponibile: si sceglie quella col
voto più alto. Nel disegno quel voto è scritto $Q(s,a)$, che si legge «quanto
vale la mossa $a$ nella situazione $s$».
```

`````{tab} Elementare

Invece del quaderno con una riga per ogni immagine, addestriamo un "occhio
esperto" che guarda lo schermo e, sul momento, dà un voto a ciascuna mossa
possibile ("vai a sinistra: 3", "resta fermo: 1", "vai a destra: 8"). Poi si
sceglie la mossa col voto più alto. Il bello è che questo occhio, avendo visto
tante partite, sa dare un voto sensato anche a una schermata **mai vista
prima**, perché assomiglia ad altre che conosce. È la differenza fra imparare a
memoria e capire.

`````

`````{tab} Superiore

Si parametrizza la funzione valore con una rete di pesi $\theta$, scrivendo
$Q(s,a;\theta)$, e la si allena minimizzando lo scarto dal *target* di Bellman:

$$
\mathcal{L}(\theta) = \mathbb{E}\Big[\big(r + \gamma \max_{a'} Q(s',a';\theta^{-})
- Q(s,a;\theta)\big)^2\Big] ,
$$

dove $r$ è la ricompensa ottenuta, $\gamma \in [0,1)$ il fattore di sconto,
$s'$ lo stato successivo e $\theta^{-}$ i pesi di una copia "congelata" della
rete. È l'algoritmo **Deep Q-Network (DQN)**. In alternativa si può
parametrizzare direttamente la *policy* $\pi_\theta(a\mid s)$ e ottimizzarla per
salita del gradiente sulla ricompensa attesa (*policy gradient*): due famiglie
che il capitolo affronterà entrambe.

`````

## La svolta: da Atari a AlphaGo

```{figure} ../figures/dqn-atari-2015.svg
:name: fig-dqn-atari
:alt: "Ciclo di DQN: i pixel dello schermo del gioco entrano in una rete che stima il valore Q di ciascuna azione possibile; si sceglie l'azione col valore più alto, l'emulatore la esegue e restituisce la ricompensa. La transizione appena vissuta non viene usata subito: un percorso tratteggiato la porta in un blocco a parte, la memoria delle esperienze, da cui si ripescano a caso i ricordi con cui si addestra la rete."
:width: 100%

Dai pixel all'azione, senza niente in mezzo. Alla rete non si dice cosa sia
una navicella o un mattoncino: riceve lo schermo grezzo, come lo riceve una
persona.
```

Il blocco laterale di {numref}`fig-dqn-atari`, la memoria delle esperienze, è
uno dei due accorgimenti che rendono stabile tutto il resto, e la sezione sul
prezzo da pagare ci tornerà sopra. Addestrare sui fotogrammi nell'ordine in cui
arrivano significa dare alla rete esempi consecutivi, e quindi quasi identici
fra loro; ripescarli a caso dalla memoria li rimescola, e la rete torna a vedere
situazioni diverse una dall'altra.

Quel primo lavoro, *Playing Atari with Deep Reinforcement Learning* (Mnih e
colleghi, 2013), diventa nel 2015 un articolo su *Nature*, *Human-level
control through deep reinforcement learning*: **un'unica architettura**, senza
ritocchi specifici per gioco, regge il confronto con un collaudatore umano
professionista su 49 titoli Atari, e in ventinove di essi ne raggiunge almeno
il 75% del punteggio. È la prova che pixel grezzi e ricompensa scarna bastano.

L'anno dopo arriva il colpo che raggiunge il grande pubblico. L'articolo su
*Nature* del gennaio 2016 (Silver e colleghi) presenta **AlphaGo** e la
vittoria per 5 a 0 sul campione europeo Fan Hui; due mesi più tardi, a Seul,
una versione più forte dello stesso programma batte per 4 a 1 Lee Sedol, fra i
più forti giocatori al mondo. Il Go, un gioco con più configurazioni che atomi
nell'universo, era a lungo considerato fuori portata per le macchine. Deep
reinforcement learning e ricerca ad albero, insieme. Il deep RL smette di
essere una curiosità da laboratorio.

## Il prezzo: instabilità e campioni costosi

L'entusiasmo non deve nascondere il conto da pagare. Il deep RL (l'abbreviazione
di *deep reinforcement learning*, e da qui in avanti si userà spesso) è
notoriamente capriccioso.

`````{tab} Elementare

Due difficoltà su tutte. La prima: l'allenamento è **instabile**, come cercare
di colpire la propria ombra, che si sposta ogni volta che ti muovi tu; piccole
modifiche possono far crollare tutto. La seconda: serve
**una quantità enorme di partite**. L'agente impara per tentativi, e di
tentativi ne vuole milioni: settimane di gioco. In un videogioco simulato va
bene; con un robot vero che si può rompere, molto meno.

`````

`````{tab} Superiore

I campioni sono **fortemente correlati** (fotogrammi consecutivi) e il *target*
$r + \gamma \max_{a'} Q(s',a';\theta)$ **si muove** insieme ai pesi che stiamo
aggiornando: la combinazione di approssimazione, bootstrapping e
apprendimento off-policy è la celebre *deadly triad* che può divergere. DQN la
addomestica con due trucchi: l'**experience replay** (campionare a caso da un
buffer di transizioni passate, decorrelandole) e la **rete target** $\theta^{-}$
aggiornata di rado, che stabilizza il bersaglio. Resta il costo campionario:
la versione di *Nature* usava circa $50$ milioni di fotogrammi per titolo. La
*sample efficiency* è tuttora un problema di ricerca aperto.

`````

## Come è organizzato il capitolo

Il percorso segue le domande, non gli acronimi: di ogni pezzo interessa *perché*
esiste, cioè quale fragilità del pezzo precedente è venuto a curare.

Si comincia dal **DQN**, la rete che prende il posto della tabella, e dai due
accorgimenti che le impediscono di esplodere: la memoria delle esperienze e la
copia congelata. Poi si cambia famiglia. Invece di dare un voto a ogni mossa e
scegliere la migliore, si può imparare **direttamente a decidere**: è la strada
dei metodi a *gradiente di policy*, e passa per l'idea di affiancare al
giocatore un giudice, per l'algoritmo che oggi si prova per primo (PPO), per la
ricerca ad albero che sta dietro ad AlphaGo, e per il modo in cui si addestrano
oggi gli assistenti conversazionali. Quella strada serve subito, perché nel
**controllo continuo** (un braccio robotico, uno sterzo) le mosse non sono un
menu di poche voci e la ricetta del DQN non si applica più.

Le tre sezioni che seguono attaccano tutte lo stesso problema, cioè che
l'esperienza costa. Il RL **basato su modello** fa provare all'agente le mosse
nella propria testa prima che nel mondo. L'**imitazione** salta i tentativi ed
errori: si guarda qualcuno che il compito lo sa già fare, e si scopre perché non
basta. L'**offline RL** impara da un archivio di esperienze altrui senza mai
agire, che è l'unica strada quando sbagliare è pericoloso, in terapia intensiva
come al volante.

Si chiude sull'**esplorazione**: cosa fare quando la ricompensa arriva così di
rado che non c'è nulla da inseguire, e cosa succede quando l'agente ottimizza
*troppo* bene una ricompensa scritta male. Quest'ultima insidia, il *reward
hacking*, è il ponte verso il capitolo sull'AI responsabile.
