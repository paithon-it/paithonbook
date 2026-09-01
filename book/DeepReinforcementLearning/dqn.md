# Deep Q-Network (DQN)

Torniamo al risultato del 2013 da cui si è aperto il capitolo, stavolta per
guardarci dentro. Quel piccolo gruppo di ricercatori londinesi di una startup
chiamata DeepMind aveva mostrato un unico programma che imparava a giocare a
sette videogiochi Atari, da *Pong* a *Space Invaders*, senza che nessuno gli
avesse spiegato le regole. L'algoritmo
riceveva solo ciò che vedrebbe un ragazzino davanti al cabinato: i pixel dello
schermo e il punteggio. Da lì, per tentativi, in tre di quei sette
(*Breakout*, *Enduro* e *Pong*) arrivava a superare un umano esperto
{cite}`mnih2013playing`. Due anni dopo il
risultato finì sulla
copertina di *Nature* {cite}`mnih2015human`. Quel programma si chiama **Deep
Q-Network**, DQN.

Nel capitolo precedente abbiamo incontrato il *Q-learning*: un agente impara una
funzione $Q(s,a)$ che stima quanto è conveniente, nel lungo periodo, compiere
l'azione $a$ trovandosi nello stato $s$. Lì la $Q$ viveva in una tabella: una
riga per ogni stato, una colonna per ogni azione. Funziona con pochi stati.

Ma quanti stati ha una schermata Atari? Lo schermo è di $210\times160$ punti,
cioè 33.600 in tutto, e ciascuno può avere un colore fra molti. Le possibilità
non si sommano fra loro, si moltiplicano: il numero che ne esce ha decine di
migliaia di cifre, mentre per contare tutti gli atomi dell'universo osservabile
ne bastano ottantuno. La tabella non basta più: non basterebbe nemmeno se una
casella pesasse un atomo.

## Dalla tabella alla rete

La mossa di DQN è tanto semplice quanto radicale: **buttiamo via la tabella e
mettiamo al suo posto una rete neurale**.

`````{tab} Elementare

Uno schedario, un cartellino per ogni possibile schermata di gioco, e su
ciascun cartellino quanto vale ciascuna mossa. Compilarlo non si può: di
schermate ce ne sono più di quante se ne riesca a contare, e finiremmo i
cartellini molto prima. Al posto dello schedario si mette allora un *esperto*
che guarda la schermata e dice il valore di tutte le mosse insieme, in un colpo
solo, senza passarle in rassegna una per volta. E lo fa anche per schermate che
non ha mai visto prima, perché ha imparato a riconoscere le somiglianze.
Quell'esperto è la rete neurale.

Dentro la rete ci sono dei numeri, qualche milione, che decidono come una
schermata si trasforma in un voto: si chiamano **pesi**, e sono le uniche cose
che cambiano mentre la rete impara. Addestrare la rete vuol dire ritoccarli, un
pochino alla volta, finché i voti non diventano sensati.

`````

`````{tab} Superiore

Approssimiamo la funzione azione-valore ottima con una rete parametrizzata da
$\theta$:

$$
Q(s, a; \theta) \approx Q^{*}(s, a).
$$

La rete prende in ingresso lo stato $s$ (i pixel) e restituisce in uscita un
vettore con un valore $Q$ per ciascuna azione ammissibile: non serve una
passata per azione. È un *function approximator*: generalizza a stati mai
visti, sfruttando la struttura condivisa delle immagini invece di memorizzare
ogni caso singolarmente.

`````

Lo stato entra da un lato, una rete convoluzionale ne estrae le
caratteristiche visive, e dall'altro lato escono i valori delle azioni
({numref}`fig-dqn`).

```{figure} ../figures/schema-dqn.svg
:name: fig-dqn
:alt: Lo schermo di un gioco stile Breakout entra in una rete convoluzionale con volumi decrescenti, seguita da uno strato denso, e produce una barra di valore Q per ciascuna delle quattro azioni possibili.
:width: 95%

Lo schema di DQN. Lo schermo di gioco attraversa gli strati convoluzionali
e uno strato denso; l'uscita è un valore $Q$ per ogni azione, cioè un voto per
ogni mossa. L'agente sceglie l'azione con il valore più alto.
```

## Perché divergeva: la triade fatale

Mettere una rete al posto della tabella non era, di per sé, un'idea nuova.
**TD-Gammon** lo faceva dal 1992, con una rete addestrata a suon di partite di
backgammon giocate contro se stessa; nella versione descritta tre anni dopo
giocava quasi come i più forti campioni del mondo {cite}`tesauro1995temporal`.
Eppure il Q-learning con una rete, per anni, **divergeva**: i valori stimati
crescevano senza fermarsi, invece di assestarsi.

Fra i due c'è una differenza sola, e conviene guardarla da vicino perché torna
in tutto il capitolo. TD-Gammon si allenava sulle partite che stava giocando
davvero. Il Q-learning fa una cosa più furba, e più rischiosa: gioca in un modo
e impara un altro modo. Ogni tanto, apposta, tira una mossa a caso per vedere
che succede; ma quando poi si segna il voto di quella situazione non ci scrive
quanto vale la mossa a caso, ci scrive quanto vale la mossa *migliore* fra
quelle disponibili lì. Gioca da esploratore e prende appunti da campione. Si
chiama **off-policy**, ed è comodissimo, perché permette di imparare da
qualunque partita: anche da una giocata male, anche da una giocata da un altro
molto tempo prima.

E non divergeva per sfortuna, né perché qualcuno avesse sbagliato a tarare il
passo di apprendimento (il *learning rate*: di quanto si spostano i pesi a
ogni correzione). Prima dei due trucchi che lo hanno reso praticabile conviene
capire da che cosa lo hanno salvato, perché è un risultato preciso e
sorprendentemente pulito.

`````{tab} Elementare

Al cabinato c'è un tale col taccuino: guarda giocare e per ogni schermata
segna quanto promette bene. Ha tre abitudini.

Non tiene un foglio per ogni schermata, giudica a somiglianza: ci rimette in
precisione sulla singola, e in cambio ha un voto anche per quelle che non ha
mai visto. È l’**approssimazione**.

Non aspetta la fine della partita. Una mossa gli frutta $1$ punto, valuta $7$
la schermata in cui si ritrova, e scrive subito i due numeri messi insieme,
il secondo contato per nove decimi: $7{,}3$, ed è il **bersaglio** verso cui
correggerà. Aggiustare un voto
con un altro voto si chiama **bootstrapping**, e separa le differenze
temporali del capitolo precedente, il TD, dai metodi Monte Carlo, che
aspettano il fischio finale per tirare le somme.

Guarda partite giocate a casaccio e scrive i voti come se al posto di quel
giocatore ci fosse un campione: l'off-policy di poco fa.

Ognuna di queste abitudini, da sola, è utile, e anche a coppie il taccuino
resta sensato. Tutte e tre insieme no: i voti possono crescere senza fermarsi.
Richard Sutton e Andrew Barto, che hanno scritto il manuale classico della
materia, la chiamano **triade fatale**.

Per vederla non serve un gioco difficile, serve il contrario. Il più facile
del mondo lo costruì Baird, e da lui si chiama **controesempio di Baird**:
sette schermate, e non si guadagna mai un punto. Zero dappertutto è la
risposta giusta, e il taccuino la scriverebbe alla perfezione, con una
manciata di numeri e il modo più elementare di darli. Quei numeri, invece di
posarsi sullo zero, crescono e non smettono più. Se il metodo sbaglia il
problema più semplice del mondo, il guasto non è nel problema.

Il tale ritocca il voto di una schermata, e per somiglianza si spostano da sé
anche le vicine: di solito è il suo vantaggio. Ma il bersaglio da cui era
partito è il voto di una vicina, uno di quelli che ha appena mosso. Ogni
ritocco sposta il bersaglio che l'aveva deciso, e il seguente parte da un
bersaglio già mosso.

Restava una protezione, e la toglie la terza abitudine. Il tale non corregge
una schermata per volta: rivede un mucchio di schermate insieme, e quelle che
nel mucchio tornano spesso tirano il taccuino più delle altre. Dalle partite
che giocherebbe lui il mucchio uscirebbe nelle proporzioni vere, e siccome sono
le schermate frequenti a decidere come va a finire, il rimpallo si smorzerebbe
da sé. Ma è dimostrato solo per il modo più elementare di dare i voti,
moltiplicare per un numero ogni cosa che si vede e sommare. Con una rete a
molti strati nessuno c'è riuscito, e si conoscono casi in cui i voti scappano
perfino quando le partite se le gioca lui. Il mucchio del tale, poi, viene da
partite giocate in un altro modo: certe schermate gli passano davanti molto più
spesso di quanto capiterebbero, altre quasi mai. Corregge con forza dove non
gli serve, e i voti salgono invece di posarsi.

`````

`````{tab} Superiore

La **triade fatale** {cite}`sutton2018reinforcement` è la coesistenza di:

1. **approssimazione di funzione**, cioè una rappresentazione parametrica che
   generalizza fra stati invece di trattarli come voci indipendenti;
2. **bootstrapping**, cioè bersagli che contengono stime correnti (TD,
   programmazione dinamica) invece dei soli ritorni osservati;
3. **addestramento off-policy**, cioè una distribuzione degli aggiornamenti
   diversa da quella indotta dalla policy che si sta valutando.

Con due soli dei tre l'instabilità si può evitare; con tutti e tre no,
e la divergenza si osserva già nel caso della sola **predizione**, senza
controllo né miglioramento della policy. Non dipende nemmeno
dall'incertezza sull'ambiente: si manifesta identica nella programmazione
dinamica, dove il modello è noto per intero.

Il **controesempio di Baird** lo esibisce in forma minima: sette stati, due
azioni, ricompensa sempre nulla, $\gamma = 0{,}99$, e una policy di
comportamento che visita gli stati in modo uniforme mentre la policy bersaglio
ne concentra tutta la massa su uno solo. La funzione valore vera è
identicamente zero ed è **esattamente rappresentabile** dai parametri
disponibili; il TD semi-gradiente, ciononostante, fa divergere i pesi. Il
fattore decisivo è la **distribuzione degli aggiornamenti**: uniforme sugli
stati, mentre la policy bersaglio li visiterebbe in proporzioni tutte diverse.
Non basta osservare che l'aggiornamento semi-gradiente non è il gradiente di
nessuna funzione obiettivo (si deriva rispetto alla stima ma non rispetto al
bersaglio, che pure dipende dai parametri): lo stesso aggiornamento, con
approssimatore lineare e sotto la distribuzione on-policy, converge, come
dimostrarono Tsitsiklis e Van Roy nel 1997 {cite}`tsitsiklis1997analysis`. Ed
è a quel caso, lineare e
on-policy, che si fermano le garanzie di convergenza note: con approssimatori
non lineari come le reti si conoscono controesempi di divergenza perfino
on-policy.

Sutton e Barto passano poi in rassegna i tre elementi chiedendosi a quale si
possa rinunciare, ed è la lettura più utile per chi progetta. All’**approssimazione**
no: senza, non si scala. Al **bootstrapping** si può, usando Monte Carlo, e si
paga in efficienza computazionale (bisogna conservare tutto fino alla fine
dell'episodio) e in efficienza di dati. All’**off-policy** si può, sostituendo
il Q-learning con Sarsa, e si perde la possibilità di imparare da un archivio
di esperienze altrui, che è però proprio la premessa del replay buffer.

`````

## Due accorgimenti per non far esplodere l'addestramento

DQN non rinuncia a nessuno dei tre ingredienti. Ne addolcisce due, uno per
accorgimento, e sono gli accorgimenti che seguono.

### Experience replay

`````{tab} Elementare

Un agente che impara sui fotogrammi *nell'ordine in cui li vive* è come uno
studente che rilegge cento volte la stessa pagina di seguito: fotogrammi
consecutivi si somigliano troppo e la rete finisce per "fissarsi". La memoria
di replay è un grande quaderno degli appunti: ogni esperienza vissuta viene
annotata e, per allenarsi, l'agente pesca **a caso** vecchie esperienze dal
quaderno. Così mescola situazioni lontane nel tempo e impara in modo più
equilibrato, e riutilizza ogni esperienza molte volte, non una sola.

`````

`````{tab} Superiore

Ogni transizione $(s, a, r, s')$ viene salvata in un buffer $\mathcal{D}$
(tipicamente un milione di transizioni). L'aggiornamento dei pesi avviene su
*minibatch* campionati uniformemente da $\mathcal{D}$, e non sull'ultima
transizione. Questo rompe la correlazione temporale tra campioni consecutivi
(che violerebbe l'ipotesi di indipendenza della discesa del gradiente
stocastica) e aumenta enormemente l'efficienza nell'uso dei dati, riutilizzando
ogni transizione in molti aggiornamenti.

`````

### Rete-target

`````{tab} Elementare

C'è un secondo problema: la rete deve inseguire un bersaglio che lei stessa
sposta a ogni passo, come cercare di colpire la propria ombra. La soluzione è
tenere **due copie** della rete: una che impara di continuo e una "congelata"
che fornisce il bersaglio e viene aggiornata solo ogni tanto. Il bersaglio
resta fermo abbastanza a lungo perché la rete che apprende riesca a
raggiungerlo.

`````

`````{tab} Superiore

Si mantiene una rete-target con parametri $\theta^{-}$, copia periodica dei
$\theta$ ogni $C$ passi. La rete si allena minimizzando l'errore quadratico
sull'equazione di Bellman:

$$
\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s')\sim U(\mathcal{D})}
\left[\big(\, r + \gamma \max_{a'} Q(s', a'; \theta^{-}) - Q(s, a;
\theta)\,\big)^2\right].
$$

Qui $r$ è la ricompensa immediata, $\gamma\in[0,1)$ il fattore di sconto, e il
termine $r + \gamma \max_{a'} Q(s', a'; \theta^{-})$ è il **bersaglio**,
calcolato con i pesi congelati $\theta^{-}$. Congelarli evita il *feedback*
instabile in cui il bersaglio si muove insieme alla stima.

`````

Sono due modi di rompere lo stesso legame, quello che tiene attaccate fra loro
cose che si susseguono nel tempo. La memoria di replay lo spezza fra
un'esperienza e la successiva, pescando a caso invece che nell'ordine in cui le
cose sono state vissute; la rete-target lo spezza fra la stima e il bersaglio
che la guida, tenendo fermo il secondo mentre la prima si muove
({numref}`fig-dqn-stabilita`).

```{figure} ../figures/dqn-stabilita.svg
:name: fig-dqn-stabilita
:alt: "Animazione: a sinistra un buffer di ventiquattro celle disposte in ordine di arrivo, di cui a ogni passo se ne accendono quattro pescate a caso, sparpagliate nel buffer invece che una di fila all'altra: sono il minibatch. A destra il valore Q di una stessa coppia stato-azione: la curva della rete che impara sale a ogni passo, mentre la scaletta della copia congelata resta ferma per tre passi e poi scatta a raggiungerla."
:width: 95%

I due accorgimenti al lavoro sullo stesso addestramento. A sinistra la memoria
di replay: le esperienze entrano in ordine, e la manciata su cui si studia (il
*minibatch*) le pesca a caso. A destra lo stesso valore $Q$ calcolato dalla rete
che impara, che si muove a ogni passo, e dalla copia congelata, che resta ferma
per un numero fisso di passi (nel disegno tre) e poi scatta a raggiungerla. La
figura è in scala ridotta: in un addestramento vero le esperienze in memoria
sono un milione e quelle pescate a ogni giro sono trentadue.
```

Si vede anche quale dei tre ingredienti ciascun accorgimento addolcisce. La
copia congelata addolcisce il bootstrapping, cioè il correggere una stima
guardandone un'altra: quell'altra adesso sta ferma per un po’ e si fa
raggiungere. La memoria di replay addolcisce l'off-policy, cioè l'imparare da
partite giocate in un altro modo: pescando a caso da un milione di ricordi
l'agente si allena su un miscuglio largo, invece che sulla manciata di
situazioni che sta attraversando in questo momento. Il primo ingrediente, la
rete al posto della tabella, resta intatto: è quello per cui si è fatto tutto
il resto.

La rete, in PyTorch, si costruisce in poche righe. Un paio di numeri prima di
leggerla. I fotogrammi arrivano ridotti a $84\times84$ punti in scala di grigi
e impilati a quattro a quattro, perché da una sola immagine ferma non si
capisce dove stia andando la pallina. Poi ciascuno dei tre strati
convoluzionali passa sull'immagine con una finestrella che avanza a salti, e
più lungo è il salto più piccolo è ciò che restituisce. Le finestrelle sono da
otto, quattro e tre punti: il primo strato salta di quattro punti alla volta e
riduce $84$ a $20$, il secondo salta di due e porta $20$ a $9$, il terzo salta
di uno e lascia $7$. Alla fine restano $7\times7$ caselle per ciascuno dei $64$
**filtri**, cioè dei rivelatori che quello strato ha imparato (uno reagisce ai
bordi verticali, un altro alla pallina, e così via). Da lì esce il `64 * 7 * 7`
del primo **strato denso**, quello in cui ogni numero in entrata parla con ogni
numero in uscita, senza più finestrelle.

```python
from torch import nn

def crea_q_network(n_azioni):
    # ingresso (4, 84, 84): 4 fotogrammi impilati (per cogliere il movimento)
    return nn.Sequential(
        nn.Conv2d(4, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 512),
        nn.ReLU(),
        nn.Linear(512, n_azioni),  # un valore Q per azione, nessuna attivazione
    )
```

Il bersaglio si calcola con la rete-target, e quando la partita finisce lì (uno
stato *terminale*: nessun seguito, quindi niente futuro da scontare) si tiene la
sola ricompensa:

```{code-block} python
:class: pt-non-eseguibile

import torch

# minibatch: 32 esperienze pescate a caso dalla memoria di replay
s, a, r, s_next, fine = replay.campiona(batch=32)

with torch.no_grad():                              # il bersaglio non si corregge
    # il voto migliore nella situazione seguente, secondo la COPIA CONGELATA
    q_next = target_net(s_next).max(dim=1).values
# gamma (fra 0 e 1) dice quanto conta il futuro rispetto al presente
bersaglio = r + gamma * q_next * (1 - fine)        # se finisce qui, resta solo r
```

## Atari: giocare partendo dai pixel

Il dettaglio storicamente rilevante è cosa vede la rete: nient'altro che
l'immagine. DeepMind impilava quattro fotogrammi consecutivi in scala di
grigi, ridotti a $84\times84$, per dare alla rete un senso del movimento (dove
va la pallina?). Nessuna informazione sulle regole, e nessuna misura scelta e
calcolata a mano da un programmatore (in gergo, nessuna *feature*: niente
«distanza fra pallina e racchetta», niente «numero di mattoni rimasti»). Lo
**stesso** algoritmo, con le **stesse** manopole di regolazione (gli
*iperparametri*: quelli che si decidono prima e non si imparano), fu addestrato
su 49 giochi diversi: raggiunse un livello comparabile a quello di
un tester umano professionista, ottenendo almeno il 75% del suo punteggio in
29 giochi su 49. In *Breakout* scoprì da solo la strategia del "tunnel", che
nessuno gli aveva insegnato. Era la prima volta che un singolo sistema
imparava una gamma così ampia di compiti partendo da input sensoriali grezzi.

## Il difetto che il massimo si porta dietro

Nel bersaglio di DQN c'è un'operazione che sembra innocua e non lo è:
**prendere il valore più alto**. Conviene capire perché gonfia le stime, sia
perché è controintuitivo (prendere il massimo è proprio quello che si vuole
fare), sia perché lo stesso difetto e la stessa cura torneranno, identici,
nella {doc}`sezione sul controllo continuo <controllo-continuo>`.

`````{tab} Elementare

Otto mosse da cui scegliere, e per ciascuna un voto approssimativo: giusto *in
media*, ma sporcato ogni volta da un errore in più o in meno. Tu prendi sempre
il voto più alto. Ora, il voto più alto degli otto è di
solito quello della mossa a cui l'errore ha dato la spinta verso l'alto più
grande, non quello della mossa davvero migliore. Fra otto misure sbagliate a
caso, la più alta è quasi sempre una misura fortunata.

Prendere il massimo di stime rumorose, insomma, non restituisce il massimo dei
valori veri: restituisce qualcosa di sistematicamente più grande. Il conto si
può anche fare. Le otto mosse valgono tutte esattamente $5$, e ogni voto
sbaglia di una quantità qualsiasi fra $-1$ e $+1$, in su come in giù, senza
preferenze. Fra otto errori pescati così, il più grande sta quasi sempre vicino
al bordo alto: in media vale $(8-1)/(8+1)$, cioè $+0{,}78$ invece di $0$.
Quindi il voto più alto degli otto, in media, non vale $5$: vale $5{,}78$.
Quanto si gonfia dipende da due cose: da quante sono le mosse fra cui si
sceglie, e da quanto sono sballati i voti. Con due mosse sole, e gli stessi
errori di prima, la gonfiatura scende a $1/3$; con otto mosse ma errori larghi
il doppio, tutto raddoppia e sale a $1{,}56$. E il guaio è che il voto gonfiato
diventa il bersaglio dell'aggiornamento successivo, quindi la gonfiatura non
resta dov'era: si tramanda.

Il rimedio si chiama **Double DQN**, e divide in due un lavoro che prima faceva
una rete sola. Prima: la stessa rete decide qual è la mossa migliore *e* dice
quanto vale. Dopo: **la rete che sta imparando dice quale mossa**, e la **copia
congelata dice quanto vale quella mossa lì**. Che i due ruoli stiano su reti
**diverse** è tutta la sostanza: l'errore che ha fatto *sembrare* buona quella
mossa non è lo stesso errore che poi ne *misura* il valore, così la fortuna non
viene contata due volte. E non si assegnano a caso: a dire *quanto vale* deve
essere la copia congelata, altrimenti si perde il bersaglio fermo che serviva a
non far esplodere l'addestramento.

Attenua, però, non guarisce. Le due reti non sono estranee fra loro: una è la
copia dell'altra di qualche passo prima, e quella parentela lascia passare buona
parte della gonfiatura. E quando le mosse non valgono davvero tutte uguale, il
correttivo tende a esagerare dalla parte opposta: i voti escono un filo bassi
invece che alti.

`````

`````{tab} Superiore

La causa non è che le singole stime siano distorte, perché non lo sono: è
l'incontro fra il **rumore** e la **convessità del massimo**, che la
disuguaglianza di Jensen mette in conto. Per stime $\hat Q$ non distorte,

$$
\mathbb{E}\big[\max_a \hat Q(s,a)\big] \;\ge\; \max_a \mathbb{E}\big[\hat Q(s,a)\big]
= \max_a Q(s,a),
$$

e il divario cresce con il numero di azioni e con la varianza dell'errore: la
disuguaglianza è **stretta** ogni volta che il rumore può cambiare quale azione
risulti la migliore, e con stime esatte si ridurrebbe a un'uguaglianza. Basta
quindi un errore di stima a media nulla
perché il bersaglio sia sistematicamente gonfio, e il bootstrapping lo propaga
all'indietro.

Il **Double DQN** {cite}`vanhasselt2016deep` disaccoppia i due ruoli. Il
bersaglio di DQN usa $\theta^{-}$ sia per scegliere sia per valutare; quello di
Double DQN fa scegliere alla rete **online** $\theta$ e valutare alla rete
**target** $\theta^{-}$:

$$
y^{\text{Double}} = r + \gamma\, Q\Big(s',\;
\arg\max_{a'} Q(s', a'; \theta);\ \theta^{-}\Big).
$$

Da confrontare con $y = r + \gamma \max_{a'} Q(s',a';\theta^{-})$: la differenza
sta tutta in quali parametri compaiono dentro l’$\arg\max$, ed è una riga sola
di codice. L'ordine dei due ruoli non è però scambiabile a piacere: far scegliere
a $\theta^{-}$ e valutare a $\theta$ conserverebbe il disaccoppiamento, e quindi
una parte della correzione, ma rimetterebbe i pesi in aggiornamento dentro il
bersaglio, buttando via il congelamento che era servito a stabilizzarlo.

Gli autori sono prudenti sul risultato, e conviene esserlo con loro: l'algoritmo
«riduce le sovrastime osservate», non le elimina. Il disaccoppiamento
annullerebbe il bias solo con due stimatori **indipendenti**, e qui il secondo
non lo è: la rete target è una copia ritardata del primo, scelta perché è il
candidato che c'era già, e il paper stesso avverte che il disaccoppiamento non è
completo. Resta inoltre che il correttivo non è neutro: quando i valori veri
delle azioni non sono tutti uguali, lo stimatore doppio tende a sostituire la
sovrastima con una lieve **sottostima**.

`````

## Ripassare ciò che sorprende, e giudicare la situazione prima delle mosse

Sul telaio di DQN la ricerca ha montato una famiglia intera di migliorie. Due
sono entrate nella pratica quasi quanto il Double DQN, e portano un'idea
ciascuna: una cambia **che cosa si ripassa**, l'altra **come si scompone il
voto**.

`````{tab} Elementare

Il quaderno degli appunti di prima si ripassa pescando a caso, e a caso vuol
dire che un'esperienza banale vale quanto una sorprendente. Uno studente vero
non fa così: ripassa più spesso le pagine dove l'ultimo compito è andato
peggio. È il **replay con priorità**: ogni esperienza porta un segnalibro
grande quanto l'errore che la rete ci ha fatto sopra l'ultima volta, e la pesca
premia i segnalibri grandi. Quanto li premi è una manopola: portata a zero, si
torna alla pesca a caso di prima. Le esperienze appena vissute entrano col
segnalibro al massimo, così nessuna finisce nel dimenticatoio prima di un
primo ripasso. Il prezzo c'è: chi ripassa quasi soltanto le pagine dove
sbaglia si fa un'idea storta del libro intero, e il rimedio è contare i
ripassi pescati apposta un po’ meno di quelli che sarebbero usciti a caso.

L'altra idea spezza il voto in due domande: quanto è buona la situazione, e
quanto aggiunge ciascuna mossa. Su un rettilineo vuoto guidare bene non
dipende dalla piccola correzione che dai al volante: conta che il rettilineo è
tranquillo, e lo sarà per chiunque. La rete **a due rami** impara le due cose
separatamente e le rimette insieme alla fine, sommandole. Sommare, però, lascia
una libertà di troppo: "la strada vale 10 e la sterzata non aggiunge niente" e
"la strada vale 7 e la sterzata aggiunge 3" fanno lo stesso voto, e niente dice
quale delle due divisioni sia quella buona. Serve un patto, e glielo si impone: i
contributi delle mosse devono compensarsi fra loro, tanto in su quanto in giù,
e quello che avanza è il giudizio sulla situazione. Con i due rami separati, quel
giudizio si affina a ogni passaggio, anche quando sulle singole mosse non c'è
niente da imparare. Così è già pronto quando arriva la curva in cui le mosse
tornano a contare.

`````

`````{tab} Superiore

Il **prioritized experience replay** {cite}`schaul2016prioritized` sostituisce
il campionamento uniforme dal buffer con

$$
P(i) \;\propto\; |\delta_i|^{\alpha},
$$

dove $\delta_i$ è l'ultimo errore TD misurato sulla transizione $i$, a cui il
lavoro somma un $\epsilon$ piccolo perché un errore sceso a zero non escluda
per sempre quella transizione, e $\alpha \ge 0$ dosa quanto la priorità morde
($\alpha = 0$ riporta all'uniforme); le transizioni nuove entrano con priorità
massima. Il campionamento non uniforme distorce però la distribuzione degli
aggiornamenti, e la correzione è un peso di *importance sampling* $w_i =
\big(N\, P(i)\big)^{-\beta}$, dove $N$ è il numero di transizioni in memoria;
il peso si normalizza sul massimo del minibatch, e $\beta$ viene portato verso
$1$ nel corso dell'addestramento, quando la correzione conta di più.

La **dueling network** {cite}`wang2016dueling` spezza la testa della rete in
due rami, il valore dello stato $V(s)$ e il vantaggio delle azioni $A(s,a)$,
e li ricompone in

$$
Q(s,a) \;=\; V(s) + \Big(A(s,a) - \tfrac{1}{|\mathcal{A}|}
\sum_{a'} A(s,a')\Big),
$$

dove $|\mathcal{A}|$ è il numero di azioni e la sottrazione della media rende
identificabile la scomposizione: senza, una costante potrebbe passare da $V$ ad
$A$ lasciando $Q$ identica. Il
guadagno è che ogni aggiornamento allena $V$, qualunque azione contenga il
minibatch: negli stati in cui le azioni più o meno si equivalgono, e in molti
giochi sono tanti, la rete impara comunque qualcosa che servirà altrove.

Con il Double DQN e altre tre migliorie (i ritorni a più passi, la stima di
un'intera distribuzione di ritorni al posto della media, l'esplorazione
tramite rumore nei pesi), questi due pezzi confluiscono in **Rainbow**
{cite}`hessel2018rainbow`; l'ablazione di quel lavoro indica il replay con
priorità e i ritorni a più passi come i componenti la cui rimozione costa di
più all'insieme.

`````

## I limiti

Molti confini di questo approccio hanno guidato la ricerca successiva, e
conviene metterli in fila. Oltre alla sovrastima del massimo, che il Double
DQN attenua e basta, ne restano tre.

- **Fame di dati.** Servono decine di milioni di fotogrammi per gioco:
  l'equivalente di settimane di gioco ininterrotto. Un umano impara in pochi
  minuti. DQN è potente ma spaventosamente inefficiente.
- **Le mosse devono essere poche e distinte** (in gergo *discrete*, cioè
  contabili una per una, come le voci di un menu). Prendere il valore più alto
  vuol dire scorrerle tutte: va bene per un joystick a poche direzioni, non per
  uno sterzo o un braccio robotico, dove la mossa è una quantità da dosare e le
  possibilità sono infinite. Da lì nascono gli algoritmi **attore-critico**
  (*actor-critic*), dove uno propone la mossa e l'altro la giudica, che
  incontreremo nel gradiente di policy e nel controllo continuo.
- **Ricompense rade.** In certi giochi il punteggio arriva solo dopo lunghe
  sequenze di mosse esatte: in *Montezuma's Revenge*, per esempio, bisogna
  scendere una scala, saltare una fune e schivare un teschio prima di prendere
  la chiave che vale il primo punto. Lì DQN sostanzialmente fallisce: senza
  segnale, non c'è nulla da inseguire.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- **DQN** butta via lo schedario che aveva un cartellino per ogni schermata e
  ci mette una rete neurale: guarda i pixel e dice a colpo d'occhio quanto
  vale ciascuna mossa, anche su schermate mai viste prima.
- Per anni un'idea così divergeva, e per una ragione precisa, la **triade
  fatale**: una rete al posto della tabella, stime aggiornate a partire da
  altre stime, e una strategia imparata mentre se ne gioca un'altra. Due
  qualunque dei tre convivono senza danni, tutti e tre insieme no: i valori
  possono crescere senza fermarsi. Il **controesempio di Baird** lo mostra su
  sette stati in cui non si guadagna mai nulla e la risposta giusta ("tutto
  vale zero") il sistema saprebbe rappresentarla alla perfezione: i numeri
  crescono lo stesso, e non si fermano.
- Due accorgimenti lo rendono stabile: il **quaderno degli appunti**, che si
  chiama **memoria di replay** (ogni esperienza viene annotata e ripescata a
  caso, così l'agente mescola situazioni lontane invece di rileggere cento volte
  la stessa pagina) e la
  **copia congelata** della rete, che tiene fermo il bersaglio abbastanza a
  lungo perché lo si possa raggiungere. Nessuno dei tre ingredienti sparisce: il
  quaderno addolcisce l'off-policy (imparare da partite giocate in un altro
  modo) e la copia congelata il bootstrapping (correggere una stima guardandone
  un'altra); l'approssimazione, la rete al posto della tabella, resta intatta.
- Prendere sempre il **voto più alto** gonfia i voti: fra tante stime sporcate
  da un errore, la più alta è quasi sempre una stima fortunata, non la mossa
  migliore. Il **Double DQN** attenua il difetto facendo dire *quale mossa*
  alla rete che impara e *quanto vale* alla copia congelata; non lo elimina,
  perché le due reti sono parenti strette.
- Due migliorie con un'idea ciascuna: si ripassa più spesso ciò che ha
  **sorpreso** (contando un po’ meno i ripassi pescati apposta, per non farsi
  un'idea storta), e si giudica la **situazione** separatamente dalle mosse,
  così si impara anche dove le mosse non contano.
- Il risultato storico del 2015: un solo programma, con le stesse manopole di
  regolazione, arriva al livello di un tester umano professionista su molti
  giochi Atari partendo dai soli pixel. Restano i limiti: servono quantità
  enormi di partite, le mosse devono essere poche e distinte, e dove il
  punteggio arriva di rado l'agente resta senza nulla da inseguire.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- **DQN** sostituisce la tabella $Q$ con una rete neurale $Q(s,a;\theta)$ che
  mappa i pixel dello stato ai valori delle azioni.
- Divergeva per una ragione precisa, la **triade fatale**: approssimazione,
  bootstrapping e off-policy insieme possono far esplodere i valori. Con due
  soli dei tre l'instabilità si può evitare, con tutti e tre no, e il **controesempio di
  Baird** lo mostra su sette stati con ricompense tutte nulle, dove la
  soluzione esatta è rappresentabile e i pesi divergono lo stesso.
- Due accorgimenti lo rendono stabile: l’**experience replay** (memoria di
  transizioni campionate a caso) e la **rete-target** (bersaglio congelato).
  Non rinunciano a nessuno dei tre ingredienti: ne attenuano due.
- Il $\max$ nel bersaglio **sovrastima** perché il rumore incontra una funzione
  convessa, non perché le stime siano distorte (Jensen; stretta solo se il
  rumore può cambiare quale azione risulta la migliore, non su tutte le stime
  aleatorie):
  $\mathbb{E}[\max_a \hat Q] \ge \max_a \mathbb{E}[\hat Q]$. Il **Double DQN**
  fa scegliere l'azione a $\theta$ e valutarla a $\theta^{-}$: *riduce* il bias,
  non lo annulla, perché i due stimatori non sono indipendenti.
- Il **prioritized replay** campiona con $P(i)\propto|\delta_i|^{\alpha}$ e
  corregge il bias con pesi di importance sampling; la **dueling network**
  ricompone $Q = V + (A - \bar A)$ e allena $V$ a ogni aggiornamento. Con
  Double DQN e altre tre migliorie confluiscono in Rainbow.
- Il risultato storico (Mnih et al., 2015): livello umano su molti giochi
  Atari partendo dai soli pixel. Restano limiti di efficienza, azioni discrete
  e ricompense rade.
```
`````
