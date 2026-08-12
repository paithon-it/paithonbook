# Imparare insieme: quando l'ambiente impara anche lui

Negli anni Venti il biologo marino Umberto D'Ancona mise in fila i registri dei
mercati ittici dell'alto Adriatico (Venezia, Trieste e Fiume) per gli anni dal
1905 al 1923, e trovò un fatto che non sapeva spiegare. Durante gli anni della
Grande Guerra, quando la pesca si era quasi fermata, la quota dei pesci
**predatori** (squali, razze) sul totale del pescato non era rimasta uguale: era
cresciuta parecchio, e finita la guerra era tornata a scendere. Meno pesca per
tutti, e a guadagnarci erano stati solo quelli che mangiano gli altri.

D'Ancona portò il problema a Vito Volterra, di cui nel luglio del 1926 sarebbe
diventato genero. Volterra rispose quello stesso anno, per i Lincei, con
*Variazioni e fluttuazioni del numero d'individui in specie animali
conviventi*: due equazioni differenziali, una per le prede e una per i
predatori, ciascuna con un termine che dipende dall'altra popolazione. La
risposta al quesito di
D'Ancona c'era, ma la cosa che vale la pena tenere è un'altra: quelle due
popolazioni **non convergono** a un equilibrio, ci girano attorno all'infinito.
Ognuna insegue l'altra, e l'altra nel frattempo si è spostata.

Tenete a mente quelle orbite, perché torneranno alla fine della sezione con
altri nomi. Le sezioni precedenti hanno *progettato* il coordinamento:
chi parla con chi, con quali messaggi, con quale regola di decisione. Qui il
coordinamento non lo progetta nessuno: gli agenti devono **impararlo**, per
tentativi e ricompense, come il bambino in bicicletta del capitolo sul
reinforcement learning. E il primo fatto da mettere in chiaro è che quasi tutto
quello che sappiamo da quel capitolo, in compagnia, smette di valere.

## Il terreno si muove sotto i piedi

Il reinforcement learning classico poggia su un'impalcatura precisa, e tutta
l'impalcatura sta appesa a una parola: **fisse**. Fisse sono le regole del mondo
(se faccio questa mossa in questa situazione, quello che succede dopo obbedisce
sempre alla stessa legge) e fissi sono i premi (la stessa cosa vale sempre
altrettanto). Su quell'impalcatura si dimostra che imparare per tentativi
funziona davvero, cioè che provando abbastanza a lungo si arriva alla strategia
migliore e poi ci si resta. Togliete la parola «fisse» e non crolla un
dettaglio: crolla la dimostrazione.

`````{tab} Elementare

Vai al lavoro e hai due strade: il viale grande o una scorciatoia fra le case.
Provi la scorciatoia per un mese intero e tieni il conto: risparmia dieci
minuti. Non è un'impressione, l'hai misurata trenta volte.

Il mese dopo la stessa scorciatoia ne risparmia cinque. Il terzo mese, nessuno.
Il quarto ci metti di più che sul viale. Non è cambiata la strada, non ci sono
lavori in corso, e non è sfortuna: è che anche gli altri automobilisti hanno un
navigatore, e anche loro hanno imparato la stessa cosa che avevi imparato tu.
Quando eri l'unico a saperlo la scorciatoia era vuota; adesso che lo sanno
tutti, la scorciatoia è la coda.

Il punto è più fondo di quanto sembri. Hai misurato bene, hai fatto trenta
prove, e il risultato è comunque scaduto. Un agente che impara da solo può
fidarsi di quello che ha misurato ieri, perché il mondo di ieri è quello di
oggi. Quando anche gli altri imparano, la misura di ieri parlava di un mondo
che non c'è più: non perché il mondo sia capriccioso, ma perché il mondo, in
buona parte, sono loro.

`````

`````{tab} Superiore

Il quadro formale è il **gioco stocastico** enunciato nell'apertura del
capitolo: $N$ agenti, uno spazio di stati $\mathcal{S}$, uno spazio di azioni
per ciascun agente, una transizione $P(s' \mid s, a)$ che dipende dall'azione
**congiunta** $a = (a^1, \dots, a^N)$ e una ricompensa $r^i$ per ciascuno. Dal
punto di vista del solo agente $i$, il processo osservato ha transizione

$$
P^i_t(s' \mid s, a^i) \;=\; \sum_{a^{-i}}
\Big(\textstyle\prod_{j \ne i} \pi^j_t(a^j \mid s)\Big)\, P(s' \mid s, a),
$$

dove $a^{-i}$ è l'azione congiunta di tutti tranne $i$, $\pi^j_t$ è la policy
dell'agente $j$ al passo di addestramento $t$ e $a = (a^i, a^{-i})$. La
transizione vera $P$ è fissa; quella **indotta** $P^i_t$ no, e lo si legge dal
pedice: cambia mentre gli altri aggiornano i propri parametri. L'agente $i$ non
vive in un MDP, e sul suo stato la proprietà di Markov non regge, perché per
prevedere $s'$ non bastano $(s, a^i)$: servirebbe sapere a che punto
dell'addestramento sono gli altri.

Cade con essa la garanzia di convergenza del Q-learning tabellare, che si
dimostra come approssimazione stocastica su un operatore di Bellman **fisso**:
contrazione di fattore $\gamma$, punto fisso unico, convergenza da qualunque
inizializzazione purché ogni coppia stato-azione sia visitata infinite volte e
i passi soddisfino le condizioni di Robbins-Monro. Se $P^i_t$ si muove,
l'operatore cambia a ogni passo e il punto fisso inseguito non sta fermo. In
pratica la stessa coppia $(s, a^i)$ restituisce ritorni diversi non per la
stocasticità di $P$, ma perché l'avversario di oggi non è quello di ieri; e il
campione pescato dal buffer di *experience replay* descrive una partita che non
si sta più giocando. Il capitolo sul deep reinforcement learning aveva già
elencato le fragilità di quella famiglia di algoritmi (campioni correlati,
bersaglio mobile, la *deadly triad*): il caso multi-agente le aggrava tutte,
perché il bersaglio si muove per una ragione in più.

`````

C'è un secondo strato di difficoltà, e nei sistemi reali non è evitabile:
quasi mai un agente vede lo stato globale. Il difensore non vede l'attaccante
alle sue spalle, il drone non vede l'altro lato del capannone, l'agente che
scrive il codice non vede la conversazione dell'agente che ha letto la
specifica. Ognuno decide su un'osservazione parziale e deve dedurre il resto,
compreso quello che gli altri stanno per fare.

`````{tab} Elementare

Immagina una squadra di soccorso in un capannone pieno di fumo, senza radio.
Ciascuno vede tre metri davanti a sé e nient'altro, e non può chiedere niente a
nessuno. Tutto quello che la squadra farà deve essere deciso **prima** di
entrare, e deve essere un piano completo: non «vai a destra», ma «se davanti a
te vedi una porta chiusa vai a destra, se vedi una scala scendi, se non vedi
niente prosegui», per ogni cosa che ciascuno potrebbe trovarsi davanti.

Contiamo quanti piani diversi si potrebbero scrivere, nel caso più misero
immaginabile: due soccorritori, due sole cose che ciascuno può vedere, due sole
cose che può fare, e tre passi in tutto. Un piano per una persona deve dire
cosa fare al primo passo (1 caso), poi cosa fare per ciascuna delle 2 cose che
può aver visto, poi per ciascuna delle 4 combinazioni di due osservazioni (le 2
cose che poteva vedere al primo passo, moltiplicate per le 2 del secondo: due
per due, quattro): in tutto $1 + 2 + 4 = 7$ decisioni, ognuna fra 2 azioni, cioè
$2^7 = 128$ piani possibili. Per due persone insieme, $128 \times 128 = 16.384$ coppie di piani
da confrontare per trovare la migliore.

Sembra poco. Portiamo i passi da tre a cinque, e le decisioni per persona
diventano $1+2+4+8+16 = 31$: i piani individuali sono $2^{31}$, poco più di due
miliardi, e le coppie da confrontare oltre quattro miliardi di miliardi. Due
soccorritori, due cose da vedere, due da fare, cinque passi. Non c'è nessun
computer, e non ci sarà.

`````

`````{tab} Superiore

Il modello che raccoglie i due strati (più agenti che imparano, ciascuno con
osservazione parziale, ricompensa comune) è il **Dec-POMDP**, la tupla
$(N, \mathcal{S}, \{\mathcal{A}^i\}, P, R, \{\Omega^i\}, O, \gamma)$: gli
agenti, gli stati, le azioni di ciascuno, la transizione $P(s' \mid s, a)$
sull'azione congiunta, una ricompensa **unica** $R(s,a)$ per tutta la squadra,
gli insiemi di osservazioni $\Omega^i$ e la funzione $O(o \mid a, s')$ che dà
la probabilità di ricevere le osservazioni $o = (o^1, \dots, o^N)$
{cite}`oliehoek2016concise`. La policy dell'agente $i$ non è una funzione dello
stato ma della propria **storia** di osservazioni,
$\pi^i(a^i \mid \bar{o}^i)$, perché nessuna osservazione locale è markoviana; e
non esiste alcun canale di comunicazione implicito, per cui il coordinamento
dev'essere interamente contenuto nelle policy fissate in anticipo.

Il prezzo di questa generalità conviene dirlo subito e per intero. Risolvere
esattamente un Dec-POMDP a orizzonte finito è **NEXP-completo**, e lo è già con
due soli agenti; per confronto lo stesso problema è P-completo per un MDP e
PSPACE-completo per un POMDP, quindi la sola decentralizzazione fa saltare due
gradini di complessità. A differenza di quanto succede con NP, qui non c'è
nessuna congettura di mezzo: dal teorema di gerarchia temporale segue
$\mathrm{P} \subsetneq \mathrm{NEXP}$, e un problema NEXP-completo non può
stare in P. Un algoritmo polinomiale esatto non esiste, ed è dimostrato. Né
serve accontentarsi: anche calcolare una soluzione approssimata entro un errore
assoluto fissato resta **NEXP-difficile**, quindi la rinuncia all'ottimo non
compra un algoritmo trattabile {cite}`oliehoek2016concise`.

`````

Da qui in avanti, quindi, nessuno risolve: tutti approssimano. Il resto della
sezione è il catalogo delle approssimazioni che reggono e delle ragioni per cui
reggono, che è più interessante dell'elenco dei loro nomi.

## Chi è stato bravo?

Il libro ha già incontrato l'assegnazione del merito, ma su un altro asse. Nel
reinforcement learning il problema era **temporale**: il premio arriva alla fine
della partita, e bisogna capire quale mossa se lo sia guadagnato. La prima
risposta era spalmarlo all'indietro su tutte le mosse, contando meno quelle più
lontane nel tempo; poi si è affinata affiancando a chi decide un secondo pezzo
di programma che tiene il conto di quanto ci si aspettava di guadagnare in
quella situazione, così da poter dire non «hai preso otto» ma «hai preso due più
di quanto ci si aspettasse». Chi decide si chiama **attore**, chi tiene il conto
si chiama **critico**, e i due nomi tornano per tutta la sezione. Con più agenti
al merito temporale se ne aggiunge uno **strutturale**, e la domanda cambia di
natura: la squadra ha vinto, chi è stato bravo?

`````{tab} Elementare

È il lavoro di gruppo a scuola, con un voto solo per tutti. Cinque studenti,
una relazione, un otto. Ognuno dei cinque porta a casa lo stesso otto: quello
che ha scritto metà del testo, quello che ha corretto le note, e quello che ha
mandato tre messaggi e poi è sparito.

Sul quaderno di un agente che impara succede una cosa precisa, e non è che
diventi furbo. L'agente sparito registra «quello che ho fatto ha fruttato otto»,
ma lo registra **per qualunque cosa abbia fatto**: stare fermo, scrivere una
riga a caso, andarsene. Tutte le sue azioni ricevono lo stesso voto, e un voto
uguale per tutte le risposte non insegna niente, esattamente come un professore
che dà otto a chiunque. Non impara a essere pigro: non impara e basta.

E c'è la seconda metà del problema, che riguarda anche chi lavora sul serio.
Chi ha scritto metà del testo vorrebbe capire quali sue scelte hanno alzato il
voto, ma il voto si muove anche per merito (o per colpa) degli altri quattro,
che nel frattempo cambiano anche loro. Il suo contributo c'è, ma è coperto dal
rumore di quello che fanno i compagni: più sono, meno si sente.

`````

`````{tab} Superiore

Con ricompensa comune, $r^i = r$ per ogni $i$, il gradiente di policy
dell'agente $i$ è

$$
\nabla_{\theta^i} J \;=\; \mathbb{E}\Big[\textstyle\sum_t
\nabla_{\theta^i} \log \pi^i_{\theta}(a^i_t \mid o^i_t)\; G_t\Big],
$$

dove $\theta^i$ sono i parametri dell'agente $i$, $\pi^i_\theta$ la sua policy,
$o^i_t$ la sua osservazione al passo $t$ e $G_t$ il ritorno scontato **della
squadra**. Il fattore di sinistra riguarda solo $i$; quello di destra riguarda
tutti. L'agente $i$ vede il proprio ritocco moltiplicato per un numero a cui
hanno contribuito anche gli altri $N-1$.

Quantifichiamo, su un modello dichiaratamente di comodo: lo stimatore così
com'è, senza alcuna baseline, e un ritorno che si scompone in contributi
indipendenti $G = \sum_j g^j$ con $\mathrm{Var}(g^j) = \sigma^2$ per ogni $j$.
Il segnale che interessa a $i$ è $g^i$ mentre il rumore che gli arriva addosso
è la somma degli altri, di varianza $(N-1)\sigma^2$. Il rapporto
segnale-rumore, misurato in deviazioni standard, vale

$$
\frac{\sigma}{\sigma\sqrt{N-1}} \;=\; \frac{1}{\sqrt{N-1}},
$$

e sotto quelle ipotesi si degrada come la radice del numero di compagni: con
nove compagni (dieci agenti in tutto) il segnale utile è **un terzo** di quello
che sarebbe stato da soli, e con cento compagni un decimo. Il modello additivo
è, si noti, il caso *facile*, quello in cui il credito sarebbe in linea di
principio separabile: già lì lo stimatore ingenuo affoga, e nei casi in cui i
contributi si intrecciano non va meglio. È il **passeggero a scrocco** in
forma di gradiente, e la cosa da notare è che nessuno bara: il problema non è
la disonestà di un agente, è che l'informazione che distinguerebbe l'utile dal
passivo non arriva a destinazione.

Quel rumore ha un antidoto parziale ma diretto: una **baseline
controfattuale**. Invece di moltiplicare il gradiente per il ritorno di tutti,
si sottrae al valore dell'azione congiunta quello che la squadra avrebbe
ottenuto se $i$ avesse giocato una mossa media, a mosse degli altri
**fissate**:

$$
A^i\big(s, a^1,\dots,a^N\big) \;=\; Q\big(s, a^1,\dots,a^N\big)
\;-\; \sum_{b}\pi^i\big(b \mid \bar{o}^i\big)\,
Q\big(s, (a^{-i}, b)\big),
$$

dove $s$ è lo stato globale, disponibile solo in addestramento, $a^{-i}$ le
azioni di tutti tranne $i$, $\bar{o}^i$ la storia locale di $i$ e la somma
corre sulle sue azioni possibili $b$. È l'idea di COMA
{cite}`foerster2018counterfactual`, ed è il vantaggio dell'architettura
actor-critic ricalcolato sull'asse strutturale invece che su quello temporale.

Conviene però dire con precisione che cosa si guadagna, perché la formula
promette meno di quanto sembri. Il termine sottratto non dipende da $a^i$,
quindi è una baseline legittima (non altera il valore atteso del gradiente) e
toglie di mezzo la parte di ritorno che $i$ incasserebbe comunque, cioè
esattamente il rumore contato sopra. Ma $A^i$ resta in generale una funzione
anche delle azioni $a^{-i}$: il valore congiunto non si scompone, ed è solo nel
modello additivo di comodo di poco fa che la differenza collassa sul solo
$g^i$. Quello che la baseline controfattuale garantisce è la **riduzione della
varianza**, non l'isolamento del contributo di $i$.

`````

La contromisura ovvia, dare a ciascuno una ricompensa sua, sposta il problema
invece di risolverlo: ricompense individuali scritte a mano sono il terreno di
coltura del *reward hacking* già visto nel deep reinforcement learning, e un
agente che ottimizza la propria può danneggiare la squadra in perfetta buona
fede. La strada che ha funzionato è l'opposta: tenere una ricompensa sola e
ricavarne il merito di ciascuno, invece di dichiararlo in anticipo. Le vie sono
due. La prima misura **per differenza**: si confronta com'è andata con come
sarebbe andata se quel membro, al posto della mossa che ha fatto, ne avesse
fatta una qualsiasi fra le sue solite, lasciando ferme quelle degli altri;
sulla relazione di gruppo, è chiedersi che voto avrebbe preso lo stesso lavoro
se uno dei cinque avesse scritto la sua parte come gli capita. La seconda impara a
**scomporre il valore**, cioè a stimare quanto ciascuno ha contribuito
partendo dal solo risultato di squadra: è la via che il resto della sezione
segue, perché si porta dietro anche il modo di addestrare insieme e poi giocare
ognuno per conto proprio. Con questo la sezione salda anche un debito lasciato
aperto dalle topologie, dove attribuire una colpa lungo una gerarchia era
rimasto un problema senza rimedio.

## Barare in allenamento, non in partita

L'asimmetria che salva è questa: durante l'addestramento il simulatore è
nostro, e possiamo leggere lo stato globale, le azioni di tutti, perfino
quantità che nessun agente potrà mai osservare; in esecuzione no, e ciascuno ha
soltanto la propria osservazione. La ricetta che ne discende si chiama
**CTDE**, addestramento centralizzato ed esecuzione decentralizzata, e la sua
regola d'oro si enuncia in una riga: l'informazione privilegiata si usa solo
dove **non servirà** durante la partita. Un critico serve ad addestrare
l'attore e poi si butta via; una funzione valore di squadra serve a distribuire
il merito e poi non serve più. L'unica cosa che deve sopravvivere alla fine
dell'addestramento è la regola con cui ciascun agente sceglie la propria mossa a
partire da quello che vede lui, e soltanto da quello. Se sopravvivesse qualcosa
che per funzionare ha bisogno di sapere anche che cosa vedono gli altri, in
partita non funzionerebbe.

`````{tab} Elementare

Il calcio ha già inventato tutto. In allenamento l'allenatore ha la ripresa
dall'alto, vede tutti e ventidue i giocatori insieme e può dire al terzino:
«quel passaggio era sbagliato, il tuo compagno stava arrivando alle tue
spalle». La domenica il terzino ha soltanto i suoi occhi, e l'allenatore resta
a bordo campo. Nessuno grida allo scandalo: la ripresa dall'alto serviva
*prima*.

La prima ricetta è esattamente questa. Ogni agente ha un **attore**, che guarda
solo il proprio pezzo di campo e decide, e in allenamento un **critico** (il
giudice di cui sopra) che guarda tutto: la situazione completa e la mossa di
ciascuno. Così il terreno si muove molto meno, perché un giudizio dato sapendo
che cosa hanno fatto *tutti* non scade appena gli altri cambiano abitudini: le
loro abitudini non gli servivano, gli servivano le loro mosse, e quelle gliele
abbiamo messe sul tavolo.

Molto meno, però, non vuol dire fermo, e la differenza va detta perché è quella
che spiega tutti i puntelli che vengono dopo. Il giudice sa che cosa hanno fatto
tutti **adesso**, ma il voto che deve dare riguarda come andrà a finire, e come
andrà a finire dipende da come giocheranno gli altri da qui in poi. Se quelli
migliorano, lo stesso identico istante di gioco merita un voto diverso. Il
terreno resta molto più fermo di prima, il che basta a far funzionare la cosa in
pratica e non basta a garantirla.

La seconda ricetta serve quando la ricompensa è una sola per tutta la squadra.
Si impara un voto per ogni giocatore e una regola per comporli nel voto di
squadra, con un vincolo che sembra innocuo: **se un giocatore alza il proprio
voto, il voto di squadra non può scendere**. È quel vincolo, e solo quello, a
permettere a ciascuno di scegliere da solo la mossa migliore: se il tuo voto
sale e quello di squadra non può scendere, allora fare del proprio meglio è
fare il meglio per tutti, e nessuno ha bisogno di consultarsi.

Il vincolo però costa, e il conto si vede su quattro numeri. Due agenti, due
mosse a testa (A e B), e questo punteggio di squadra:

|  | l'altro fa A | l'altro fa B |
|:--|:--:|:--:|
| **io faccio A** | $2$ | $0$ |
| **io faccio B** | $0$ | $1$ |

Il massimo è che vadano tutti e due su A, e vale $2$. Perché ciascuno ci arrivi
da solo, A deve avere il voto individuale più alto per entrambi. Ma se per il
primo agente A vale più di B, allora, per la regola, la squadra con il primo su
A non può fare peggio della squadra con il primo su B, a parità di mossa del
secondo. Mettiamo il secondo su B: la casella $(A, B)$ dovrebbe valere almeno
quanto la casella $(B, B)$. La tabella dice $0$ contro $1$. Non torna, e non
tornerà mai. Le situazioni che questa ricetta perde sono proprio quelle in cui
bisogna accordarsi su una **convenzione arbitraria**, come guidare tutti a
destra o tutti a sinistra: lì la mossa giusta per me dipende da quella
dell'altro in un modo che nessun voto individuale può contenere.

`````

`````{tab} Superiore

**MADDPG** {cite}`lowe2017multi` è la forma canonica del principio. Ogni agente
$i$ ha un attore deterministico $\mu^i_{\theta^i}(o^i)$, che riceve solo la
propria osservazione, e un critico

$$
Q^i_{\phi^i}\big(x,\, a^1, \dots, a^N\big),
$$

dove $x$ raccoglie l'informazione di stato disponibile in addestramento (nel
caso più semplice la concatenazione delle osservazioni di tutti) e
$a^1, \dots, a^N$ sono le azioni di **tutti** gli agenti. L'attore è
decentralizzato, il critico no, e a fine addestramento il critico si getta.

La ragione per cui questo **attenua** la non stazionarietà si scrive in una
riga:

$$
P\big(s' \mid s, a^1, \dots, a^N, \pi^1, \dots, \pi^N\big)
\;=\; P\big(s' \mid s, a^1, \dots, a^N\big),
$$

cioè **condizionando sulle azioni di tutti la transizione non dipende più dalle
policy**, e quindi non cambia quando le policy cambiano; è il critico
decentralizzato, che vede solo $a^i$ e dovrebbe marginalizzare sulle altre
azioni usando le policy correnti, a vedersi muovere il terreno sotto i piedi.

Attenzione a non chiedere a quella riga più di quanto dica, perché è un passo
che si sbaglia facilmente. Ciò che è stazionario per costruzione è il **nucleo
di transizione** condizionato all'azione congiunta. Ciò che il critico deve
regredire non è la transizione: è $Q^i(x, a^1, \dots, a^N)$, un valore **atteso
lungo la traiettoria futura**, e quella traiettoria la generano le policy
$\pi^{-i}$ dal passo successivo in poi. Tenendo $P$ e $r$ identiche e cambiando
soltanto la policy dell'avversario, il numero da regredire a parità di ingresso
cambia: basta un gioco stocastico minimo (due stati, due agenti, due azioni) e
una valutazione esatta di $Q^1$ sotto due policy diverse dell'agente $2$ per
vedere le due tabelle separarsi di una frazione ben visibile del valore medio.
La transizione è ferma; il bersaglio no.

È il motivo per cui MADDPG tiene comunque le reti target e, nel lavoro
originale, gli *ensemble* di policy: sono contromisure alla non stazionarietà
**residua**, e non esisterebbero se questa fosse stata eliminata «per
costruzione». Da cui la conseguenza già enunciata in apertura di sezione, che
qui va ribadita perché è facile crederla revocata: **le garanzie di convergenza
del capitolo precedente, qui, non valgono**. Il Q-learning converge perché itera
un operatore di Bellman fisso; qui l'operatore si muove insieme alle policy
altrui, e per gli algoritmi di questa sezione una dimostrazione analoga non
c'è. Funzionano in pratica, che è un'affermazione diversa e va tenuta distinta.

I costi sono due e vanno detti: l'ingresso del critico cresce linearmente in $N$
(e con esso il numero di campioni necessari a coprirlo), e serve un critico per
agente non appena le ricompense non coincidono, cioè in ogni scenario
competitivo o misto.

**QMIX** {cite}`rashid2018qmix` affronta l'altro pezzo, l'assegnazione
strutturale del merito, nel caso puramente cooperativo. Ogni agente stima
un'utilità $Q^i(\bar{o}^i, a^i)$ sulla sola storia locale, e una rete di
miscelazione le compone nel valore di squadra $Q_{tot}$ sotto il vincolo

$$
\frac{\partial Q_{tot}}{\partial Q^i} \;\ge\; 0 \qquad \forall i,
$$

cioè $Q_{tot}$ è **monotona non decrescente** in ciascuna utilità individuale;
in pratica i pesi della rete di miscelazione sono vincolati a essere non
negativi, e li produce una *hypernetwork* che riceve lo stato globale $s$,
disponibile solo in addestramento. La monotonia è esattamente ciò che serve per
decentralizzare l'esecuzione, perché implica

$$
\arg\max_{a} Q_{tot}(\bar{o}, a) \;=\;
\Big(\arg\max_{a^1} Q^1(\bar{o}^1, a^1), \;\dots,\;
\arg\max_{a^N} Q^N(\bar{o}^N, a^N)\Big),
$$

dove $a = (a^1, \dots, a^N)$ è l'azione congiunta: **massimizzare
individualmente equivale a massimizzare globalmente**, e il $\max$ sull'azione
congiunta, che costerebbe $|\mathcal{A}|^N$, si calcola in $N|\mathcal{A}|$
operazioni. In esecuzione nessuno consulta nessuno.

Ciò di cui il vincolo priva è altrettanto preciso, e per dimostrarlo basta un
gioco a due agenti con due azioni a testa. Poniamo $Q_{tot}(A,A)=2$,
$Q_{tot}(A,B)=Q_{tot}(B,A)=0$, $Q_{tot}(B,B)=1$, e supponiamo esista una $f$
monotona con $Q_{tot}(a^1,a^2) = f\big(Q^1(a^1), Q^2(a^2)\big)$. Se
$Q^1(A) \ge Q^1(B)$, la monotonia nel primo argomento dà
$Q_{tot}(A,B) \ge Q_{tot}(B,B)$, cioè $0 \ge 1$: falso, quindi
$Q^1(A) < Q^1(B)$. Per simmetria $Q^2(A) < Q^2(B)$. Ma allora la monotonia nei
due argomenti insieme dà $Q_{tot}(A,A) \le Q_{tot}(B,B)$, cioè $2 \le 1$: falso
di nuovo. Nessuna $f$ monotona rappresenta quella matrice. La classe di giochi
che le sfugge è quella in cui la mossa migliore per uno dipende in modo **non
monotono** da ciò che fanno gli altri, e il coordinamento su una convenzione
arbitraria ne è il caso da manuale.

`````

C'è una terza ricetta, e il suo interesse è metodologico prima che tecnico.
**MAPPO** {cite}`yu2022surprising` non inventa nulla: prende il PPO del
capitolo sul deep reinforcement learning, quello che limita di quanto la policy
può spostarsi a ogni aggiornamento, lascia a ogni agente il suo attore
decentralizzato e gli affianca **una sola funzione valore centralizzata**, che
in addestramento vede lo stato globale. Il titolo del lavoro dichiara la
sorpresa: un metodo semplice, con le manopole girate per bene, regge il
confronto con architetture costruite apposta per il caso multi-agente. E sono
manopole noiose: rimettere i punteggi su una scala comune, non insistere troppe
volte sugli stessi dati prima di buttarli, spostarsi poco per volta. Vale la
pena tenerlo accanto alla regola prudente del «Costo del coordinamento»: prima
di credere che serva la macchina complicata, conviene misurare fin dove arriva
quella semplice messa a punto per bene.

## L'avversario sei tu di ieri

Nel 1959 Arthur Samuel, ingegnere IBM, pubblica sull'*IBM Journal of Research
and Development* uno studio su un programma che gioca a dama. Dentro c'è
un'idea che regge ancora oggi: per allenarlo, Samuel ne teneva due copie, che
chiamava alpha e beta. Alpha aggiustava i propri coefficienti **dopo ogni
mossa**; beta teneva gli stessi per tutta la durata di una partita. Il contrasto
è lì, ed è tutto: uno dei due si muove *dentro* la partita, l'altro sta fermo
finché la partita non è finita, ed è proprio questo a farne un avversario
stabile contro cui misurarsi. Quando alpha vinceva, il suo sistema di punteggio
passava a beta e si ricominciava. Quando invece perdeva, alpha si prendeva un
segno nero, e al terzo il suo polinomio veniva scombinato di forza azzerando il
coefficiente del termine principale: una ripartenza casuale, mezzo secolo prima
che si chiamasse così. Il programma non aveva un maestro: aveva se stesso, un
passo indietro.

Il meccanismo prezioso del **self-play** non è il risparmio di partite umane:
è che il **curriculum si genera da solo**. Un avversario troppo forte non
insegna niente (perdi comunque, e non sai per che cosa); uno troppo debole
nemmeno (vinci comunque, e qualunque cosa tu faccia va bene). L'avversario che
insegna è quello che ti sta appena sopra, e deve cambiare man mano che migliori.
Progettare quella scala a mano, per un gioco complesso, è fuori portata:
nessuno sa scrivere l'esercizio giusto per un giocatore di Go di livello
intermedio. Nel self-play la scala non si progetta, si ottiene per costruzione,
perché l'avversario è forte quanto te, sempre, essendo te. Nessuno dei due vince
troppo spesso, e le partite restano informative.

È la linea che porta ad **AlphaGo** {cite}`silver2016mastering`, già raccontato
nel capitolo sul deep reinforcement learning: una rete di policy addestrata
prima sulle mosse dei giocatori umani e poi affinata giocando contro copie di
sé, e una rete di valore addestrata proprio sulle partite così generate, il
tutto dentro una ricerca ad albero Monte Carlo. L'anno dopo la stessa squadra
toglie di mezzo anche il punto di partenza umano: **AlphaGo Zero**
{cite}`silver2017mastering` parte dalle sole regole del gioco e da pesi
casuali, e tutto il suo addestramento è self-play. È la prova più netta del
punto: le partite umane erano un acceleratore, non un ingrediente necessario.

## Quando la scala non esiste

Arriva qui la parte più istruttiva, ed è un fallimento. Il self-play ingenuo,
cioè allenarsi sempre contro l'ultima versione di sé, funziona benissimo quando
«essere più bravi» è una relazione d'ordine: se A batte B e B batte C, allora A
batte C. In molti giochi non lo è, e allora l'idea stessa di una classifica
lineare della bravura non ha un referente.

`````{tab} Elementare

Il caso più piccolo lo conoscono tutti: sasso, carta, forbici. Il sasso batte
le forbici, le forbici battono la carta, la carta batte il sasso. Chi è il più
forte? La domanda non ha risposta, e non perché manchino i dati: non c'è
proprio, un più forte.

Guarda cosa succede se ti alleni contro l'ultima versione di te stesso.
Cominci giocando sasso. La versione 2 impara a batterlo e gioca carta. La
versione 3 impara a battere la carta e gioca forbici. La versione 4 impara a
battere le forbici e gioca... sasso, cioè la versione 1. Sei tornato al punto di
partenza dopo tre giri, e continuerai a girare per sempre.

Adesso la parte che dovrebbe far paura. Se misuri i progressi come si fa di
solito, cioè guardando quanto la versione nuova batte quella precedente, vedi
**il cento per cento di vittorie a ogni generazione**, per sempre: una curva che
sale e non scende mai, il grafico più rassicurante che esista. Se invece
metti la versione nuova contro quella di due generazioni prima, perde sempre. E
se la metti contro il mucchio di tutte le versioni passate, in media non guadagna
niente. Il progresso era un'illusione ottica prodotta dal metro di misura.

`````

`````{tab} Superiore

Il fenomeno si chiama **non transitività**. In un gioco simmetrico a somma zero
con matrice di payoff antisimmetrica $\mathbf{A}$ ($A_{ij} = -A_{ji}$, il
guadagno di chi gioca $i$ contro chi gioca $j$), la relazione «$i$ batte $j$»
può contenere cicli, e quando li contiene non esiste alcuna funzione $f$ su una
scala reale, nessun punteggio di tipo Elo, tale che
$A_{ij} > 0 \iff f(i) > f(j)$: un ordine totale semplicemente non c'è.

Il self-play ingenuo è, in questa notazione, l'iterazione della miglior risposta
alla strategia corrente: si sceglie l'indice
$i_{t+1} \in \arg\max_i (\mathbf{A}\,\pi_t)_i$ e si pone
$\pi_{t+1} = \mathbf{e}_{i_{t+1}}$, il versore della strategia pura
corrispondente. In un gioco
ciclico l'orbita di quell'iterazione è un ciclo, e la quantità che si sta
massimizzando (il guadagno contro $\pi_t$) resta massima a ogni passo mentre la
quantità che interessa davvero, la **sfruttabilità**
$\varepsilon(\pi) = \max_i (\mathbf{A}\,\pi)_i$, cioè quanto ricava contro $\pi$
il miglior avversario possibile, resta al valore peggiore. Una strategia pura
in sasso-carta-forbici ha $\varepsilon = 1$ qualunque essa sia; l'unico
equilibrio è la miscela uniforme, che ha $\varepsilon = 0$ e che **nessuna
strategia pura realizza**, né è il limite della successione qui sopra, la quale
di vertici è fatta e sui vertici resta. A convergerci è la **frequenza empirica**
di una successione di strategie pure, ed è precisamente quello che fa il rimedio
qui sotto.

Il rimedio è cambiare l'avversario: non l'**ultima** versione, ma la
**popolazione** di tutte quelle passate. La miglior risposta alla media
empirica delle versioni precedenti è il **gioco fittizio** di Brown e Robinson,
di cui è noto che in un gioco a due giocatori e somma zero la frequenza
empirica converge a un equilibrio di Nash. Sono i due regimi che il codice qui
sotto mette in colonna.

`````

```python
import numpy as np

# Gioco ciclico a somma zero. A[i, j] e' il guadagno di chi gioca i contro
# chi gioca j: +1 vittoria, -1 sconfitta, 0 pareggio.
#             sasso  carta  forbici
A = np.array([[  0,   -1,    +1],    # sasso
              [ +1,    0,    -1],    # carta
              [ -1,   +1,     0]])   # forbici
NOMI = ["sasso", "carta", "forbici"]


def miglior_risposta(q):
    """L'azione che rende di piu' contro un avversario distribuito come q."""
    return int(np.argmax(A @ q))


def pura(i):
    """La distribuzione concentrata su una sola azione."""
    e = np.zeros(3)
    e[i] = 1.0
    return e


# Self-play ingenuo: ogni versione e' la miglior risposta all'ULTIMA versione.
storia = [0, 1]                      # gen 1 gioca sasso, gen 2 e' la sua risposta
print("gen  gioca     vs gen-1  vs gen-2  vs le passate  sfruttabilita'")
for t in range(3, 9):
    nuova = miglior_risposta(pura(storia[-1]))
    popolazione = np.mean([pura(p) for p in storia], axis=0)
    print(f"{t:3d}  {NOMI[nuova]:9s} {A[nuova, storia[-1]]:+7.2f} "
          f"{A[nuova, storia[-2]]:+8.2f} {(A @ popolazione)[nuova]:+13.2f} "
          f"{np.max(A @ pura(nuova)):+14.2f}")
    storia.append(nuova)

# Contro una POPOLAZIONE: miglior risposta alla media di tutte le versioni.
freq = np.array([1.0, 0.0, 0.0])
for _ in range(2000):
    freq[miglior_risposta(freq / freq.sum())] += 1
p = freq / freq.sum()
print("\npopolazione dopo 2000 generazioni: "
      + " ".join(f"{n}={v:.3f}" for n, v in zip(NOMI, p)))
print(f"sfruttabilita' della popolazione:  {np.max(A @ p):+.3f}")
```

```text
gen  gioca     vs gen-1  vs gen-2  vs le passate  sfruttabilita'
  3  forbici     +1.00    -1.00         +0.00          +1.00
  4  sasso       +1.00    -1.00         +0.00          +1.00
  5  carta       +1.00    -1.00         +0.25          +1.00
  6  forbici     +1.00    -1.00         +0.00          +1.00
  7  sasso       +1.00    -1.00         +0.00          +1.00
  8  carta       +1.00    -1.00         +0.14          +1.00

popolazione dopo 2000 generazioni: sasso=0.326 carta=0.335 forbici=0.339
sfruttabilita' della popolazione:  +0.008
```

Vale la pena leggere le quattro colonne una per una, perché ciascuna è una
lezione. La prima è la metrica che tutti guardano, ed è una linea piatta di
vittorie: ogni generazione batte la precedente, sempre, per sempre. La seconda è
la stessa storia dal lato scomodo: contro la versione di **due** generazioni
prima si perde, sempre. La terza dice che contro l'insieme delle versioni
passate il guadagno resta a zero, con qualche sussulto verso l'alto nelle
generazioni in cui quell'insieme è sbilanciato. La quarta è la più severa: la
**sfruttabilità** dell'agente corrente, cioè quanto ci ricava contro di lui il
miglior avversario possibile (più è alta, più l'agente è facile da battere),
resta $1$, il massimo, a ogni generazione. Dopo sei generazioni il campione è
fragile esattamente quanto il primo giorno, e la curva dei progressi non lo
dice.

Le ultime due righe mostrano l'alternativa. Allenandosi contro la media di
tutte le versioni passate, invece che contro l'ultima, la popolazione converge
in duemila generazioni a $(0{,}33; 0{,}33; 0{,}34)$, cioè all'equilibrio, e la
sua sfruttabilità scende a $0{,}008$: praticamente zero. Notate però il
soggetto della frase, perché è tutta la differenza: a essere imbattibile non è
un agente, è la **popolazione**. Il campione da schierare non è l'ultimo nato,
è il mucchio.

Su scala industriale, questa è la *league* di AlphaStar
{cite}`vinyals2019grandmaster`. Ci convivono gli agenti **principali**, che
devono battere tutti e si allenano contro la lega intera pescando gli avversari
con probabilità che dipende dal tasso di vittoria (una versione prioritaria del
gioco fittizio); gli **sfruttatori**, di due tipi, addestrati non a essere forti
in generale ma a trovare il punto debole di qualcun altro, gli uni degli agenti
principali del momento, gli altri della lega nel suo insieme; e le versioni
**congelate** di tutti costoro, che una volta entrate nella lega ci restano per
sempre. Gli sfruttatori sono la parte controintuitiva: sono agenti pagati per
non essere bravi, cioè per specializzarsi in una singola strategia che umilia il
campione del momento, ed esistono per esibire una debolezza che il campione, da
solo, non incontrerebbe mai. La lega però non è gratis: ogni avversario in più è
memoria, partite e calcolo, e il conto del «Costo del coordinamento» torna a
presentarsi qui, sul lato dell'addestramento.

E torniamo ai pesci dell'Adriatico. Le orbite chiuse che Volterra trovò nelle
sue due equazioni non sono una curiosità zoologica: la dinamica con cui una
popolazione di strategie evolve in un gioco ciclico ha la stessa forma
matematica, e gira attorno all'equilibrio senza mai caderci dentro. Prede e
predatori, sasso e carta: quando ciascuno insegue l'altro e l'altro nel
frattempo si sposta, il sistema non converge, orbita.

## Una GAN è un sistema multi-agente a due

Conviene chiudere il cerchio su una cosa che il libro racconterà per esteso più
avanti, nel capitolo sulle **GAN**, e che da qui si riconosce a colpo d'occhio.
Non serve averlo già letto: quello che serve sta in due righe. Una GAN è una
coppia di reti che si allenano l'una contro l'altra. La prima, il
**generatore**, fabbrica esemplari falsi (di solito immagini) partendo dal caso;
la seconda, il **discriminatore**, guarda un esemplare e dice se è vero o falso.
Il generatore vince quando inganna, il discriminatore quando smaschera.
Nient'altro.

Detta così, una GAN è un sistema multi-agente con $N = 2$, e riconoscerlo non è
un gioco di parole: spiega i suoi guasti tipici meglio di quanto li spieghi la
teoria delle reti.

`````{tab} Elementare

Il traguardo, per cominciare, non è un traguardo. In tutto il resto del libro
addestrare vuol dire scendere: c'è una valle, si cerca il fondo, e quando ci si
è arrivati si è finito. Qui un fondo non c'è, perché la discesa di uno è la
salita dell'altro. Quello che si può sperare è un **pareggio**, cioè la
situazione in cui a nessuno dei due conviene più cambiare mossa da solo.

Da lì si capiscono i due modi in cui l'addestramento di una GAN va storto, e
sono i due che questa sezione ha già raccontato con altri nomi.

Il primo è l'**oscillazione**: le due reti girano in tondo invece di
avvicinarsi, come i pesci dell'Adriatico e come sasso, carta e forbici. Il
falsario impara a battere il poliziotto di adesso; il poliziotto impara a
battere quel falsario lì; il falsario cambia di nuovo, e si ricomincia. Ognuno
insegue l'altro, e l'altro nel frattempo si è spostato.

Il secondo si chiama **collasso dei modi**, e guardato da qui è il self-play
ingenuo visto dall'altro lato. Il falsario scopre l'unica cosa che riesce a far
passare per buona al poliziotto del momento (mettiamo: un certo tipo di volto) e
si mette a produrre soltanto quella. Contro l'avversario corrente vince quasi
sempre, quindi il suo punteggio è ottimo; ma di tutto il resto ha smesso di
saper fare qualunque cosa. È, parola per parola, il «batte l'ultima versione e
perde contro quella di due generazioni fa».

E allora non stupisce che fra le contromisure ce ne sia una che qui si riconosce
subito, anche se il capitolo sulle GAN non la elenca fra le proprie: far vedere
al poliziotto anche i falsi prodotti dalle **versioni vecchie** del falsario. È
la lega di AlphaStar in miniatura, con la stessa identica motivazione.

`````

`````{tab} Superiore

Nella formulazione **minimax** generatore e discriminatore condividono un'unica
funzione di valore: uno la minimizza, l'altro la massimizza, e il gioco è a due
giocatori e somma zero **esattamente**. Il traguardo non è un minimo di
$\mathcal{L}$ ma un **equilibrio di Nash**, il punto in cui a nessuno dei due
conviene più deviare da solo.

Una precisazione che il capitolo sulle GAN paga per intero e che qui non va
persa: quella formulazione non è quella che si usa. Con la *loss*
non-saturante, cioè la funzione obiettivo con cui le GAN si addestrano davvero,
il gioco **non è più a somma zero** e non si lascia più scrivere con un'unica
funzione di valore. La lettura come sistema a due giocatori regge comunque; la
somma zero è una proprietà della sola versione minimax, e va attribuita a
quella.

Riletti da qui, i guasti classici dell'addestramento avversario smettono di
sembrare capricci di quella famiglia di modelli. L'**oscillazione** è la stessa
orbita di poche righe fa: il generatore fa la miglior risposta al discriminatore
corrente, il discriminatore la miglior risposta a quello, e la coppia percorre
un ciclo invece di avvicinarsi all'equilibrio. Il **collasso dei modi** è il
self-play ingenuo visto dall'altro lato: il generatore si specializza sull'unica
regione dello spazio che inganna l'avversario corrente, ottiene contro di lui un
tasso di successo altissimo, e perde tutto il resto del supporto (che è, parola
per parola, il «batte l'ultima versione, perde contro quella di due generazioni
fa»). Non stupisce allora che fra le contromisure note ce ne sia una che qui si
riconosce a colpo d'occhio, e che il capitolo sulle GAN non elenca fra le
proprie: mostrare al discriminatore anche campioni prodotti da versioni
**passate** del generatore. È una lega in miniatura, con la stessa motivazione.

`````

## Il critico che vede tutto

Resta da vedere quanto poco codice serva a scrivere la struttura di CTDE.
L'esempio qui sotto è lo scheletro di MADDPG: attori decentralizzati, uno per
agente, ciascuno con la sola osservazione che avrà anche in esecuzione; e
critici centralizzati che ricevono la concatenazione delle osservazioni **e**
delle azioni di tutti. Manca tutto il resto (l'archivio delle partite passate da
cui si ripescano gli esempi, le copie congelate delle reti che servono a tenere
fermi i conti mentre si impara, i passi di ottimizzazione veri e propri), perché
qui il punto è soltanto chi vede che cosa.

```python
import torch
from torch import nn


class Attore(nn.Module):
    """Decentralizzato: vede solo la propria osservazione, in addestramento
    come in esecuzione. E' l'unica parte che sopravvive alla fine."""

    def __init__(self, dim_oss, dim_azione):
        super().__init__()
        self.rete = nn.Sequential(
            nn.Linear(dim_oss, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, dim_azione), nn.Tanh(),   # azioni continue in [-1, 1]
        )

    def forward(self, oss):
        return self.rete(oss)


class CriticoCentralizzato(nn.Module):
    """Esiste solo in addestramento: riceve osservazioni e azioni di TUTTI,
    cosi' la transizione che vede non dipende piu' dalle policy altrui.
    (Il bersaglio da regredire, invece, dipende ancora dal futuro: per
    quello servono comunque le reti target.)"""

    def __init__(self, dim_oss, dim_azione, n_agenti):
        super().__init__()
        ingresso = n_agenti * (dim_oss + dim_azione)
        self.rete = nn.Sequential(
            nn.Linear(ingresso, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1),                      # un solo numero: Q(x, a1..aN)
        )

    def forward(self, osservazioni, azioni):
        # due liste di N tensori (lotto, dim): si concatenano sull'asse features
        return self.rete(torch.cat(osservazioni + azioni, dim=1))


N, DIM_OSS, DIM_AZ, LOTTO = 3, 10, 2, 4

attori = nn.ModuleList([Attore(DIM_OSS, DIM_AZ) for _ in range(N)])
# un critico per agente: serve appena le ricompense non coincidono
critici = nn.ModuleList([CriticoCentralizzato(DIM_OSS, DIM_AZ, N) for _ in range(N)])

oss = [torch.randn(LOTTO, DIM_OSS) for _ in range(N)]   # cosa vede ciascuno
azioni = [attori[i](oss[i]) for i in range(N)]          # ognuno decide da solo

print(azioni[0].shape)             # torch.Size([4, 2])
print(critici[0](oss, azioni).shape)  # torch.Size([4, 1])
```

Due righe raccontano tutta l'architettura. La penultima è l'esecuzione:
`attori[i](oss[i])`, ogni agente con la propria osservazione e nient'altro, ed è
ciò che girerà sul robot o dentro il processo. L'ultima è l'addestramento:
`critici[0](oss, azioni)`, dove il primo argomento è la lista di tutte le
osservazioni e il secondo la lista di tutte le azioni. Quel critico non esiste
la domenica.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Quando più agenti imparano insieme, per ciascuno il mondo **non sta fermo**:
  è la scorciatoia che risparmiava dieci minuti finché eri l'unico a
  conoscerla, e che adesso è la coda. La stessa mossa nella stessa situazione
  rende diversamente non per sfortuna ma perché gli altri sono migliorati, e le
  garanzie di convergenza viste nel reinforcement learning, che presuppongono un
  mondo fisso, non valgono più.
- Se poi ciascuno vede solo il proprio pezzo (la squadra di soccorso nel fumo,
  senza radio) tutto il coordinamento va deciso **prima** di entrare, e i piani
  da confrontare esplodono: bastano due soccorritori, due cose da vedere, due da
  fare e cinque passi per superare i quattro miliardi di miliardi di coppie di
  piani. Non esiste, ed è dimostrato, un modo di risolvere davvero questo
  problema in tempo utile, e nemmeno di approssimarlo bene
  {cite}`oliehoek2016concise`: nessuno risolve, tutti approssimano.
- Al merito **nel tempo** (quale mossa della sequenza ha prodotto il premio) se
  ne aggiunge uno **fra compagni**: con un voto solo per tutta la squadra, come
  nel lavoro di gruppo a scuola, chi è sparito registra lo stesso otto degli
  altri e quindi non impara niente, e chi ha lavorato non distingue il proprio
  contributo dal rumore dei compagni; più sono, meno si sente. Il rimedio è
  misurare ciascuno **per differenza**: che voto avrebbe preso lo stesso lavoro
  se lui, al posto della mossa che ha fatto, ne avesse fatta una qualsiasi fra
  le sue solite e gli altri no {cite}`foerster2018counterfactual`. Abbassa il
  rumore, non isola il merito del singolo.
- La ricetta che funziona è **CTDE**: informazione privilegiata in allenamento,
  occhi veri in partita. L'allenatore ha la ripresa dall'alto e il terzino, la
  domenica, ha solo il proprio sguardo. Un giudice che conosce le mosse di tutti
  dà giudizi che non scadono quando i compagni cambiano abitudini
  {cite}`lowe2017multi`; e se la ricompensa è una sola, si impara un voto per
  giocatore più una regola per comporli, con il vincolo che alzare il proprio
  voto non possa far scendere quello di squadra {cite}`rashid2018qmix`, così
  ciascuno sceglie da solo la mossa migliore. Il prezzo è che restano fuori le
  situazioni in cui bisogna accordarsi su una convenzione arbitraria (tutti a
  destra o tutti a sinistra). Un metodo semplice, regolato con cura, va misurato
  prima di sostituirlo con uno complicato {cite}`yu2022surprising`.
- Il **self-play** vale perché l'esercizio giusto se lo costruisce da solo:
  l'avversario è forte quanto te, sempre, essendo te. È la linea che va da
  AlphaGo {cite}`silver2016mastering` ad AlphaGo Zero
  {cite}`silver2017mastering`, che parte dalle sole regole del gioco.
- Ma dove non esiste un più forte in assoluto (sasso, carta, forbici)
  allenarsi contro l'ultima versione di sé **gira in tondo**: si vince sempre
  contro la versione precedente, si perde sempre contro quella di due
  generazioni prima, e la curva dei progressi è un'illusione ottica prodotta dal
  metro di misura. Il rimedio è allenarsi contro il **mucchio** di tutte le
  versioni passate: è la *league* di AlphaStar {cite}`vinyals2019grandmaster`,
  con i campioni, gli specialisti pagati per trovare il punto debole di
  qualcuno, e tutte le versioni congelate del passato.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Con più agenti che imparano insieme, per ciascuno l'ambiente **non è
  stazionario**: la transizione indotta $P^i_t$ dipende dalle policy degli
  altri, che cambiano. La stessa coppia $(s, a^i)$ rende diversamente non per
  rumore ma perché l'avversario è migliorato, e la garanzia di convergenza del
  Q-learning, che presuppone un operatore di Bellman fisso, decade.
- Il quadro formale è il **gioco stocastico**, e con osservazione parziale il
  **Dec-POMDP** {cite}`oliehoek2016concise`: risolverlo esattamente a orizzonte
  finito è **NEXP-completo** già con due agenti (contro P-completo per un MDP e
  PSPACE-completo per un POMDP), e approssimarlo resta NEXP-difficile. Nessuno
  risolve: tutti approssimano.
- All'assegnazione **temporale** del merito se ne aggiunge una
  **strutturale**: con ricompensa comune il gradiente del singolo è moltiplicato
  per il ritorno di tutti, e quel che ha fatto lui resta sepolto sotto quel che
  hanno fatto i compagni. Nel caso di comodo in cui i contributi semplicemente
  si sommano e sono indipendenti, il rapporto fra segnale e rumore scende come
  $1/\sqrt{N-1}$ (un terzo con dieci agenti). È il **passeggero a scrocco** in
  forma di gradiente. Il rimedio è misurare ciascuno **per differenza**:
  confrontare com'è andata con come sarebbe andata se lui, al posto della sua
  mossa, ne avesse fatta una qualsiasi fra le solite, e gli altri no. È la
  mossa di COMA {cite}`foerster2018counterfactual`, che abbatte quel rumore
  senza però isolare il contributo del singolo.
- La ricetta che funziona è **CTDE**: informazione privilegiata in addestramento,
  osservazione vera in esecuzione. **MADDPG** {cite}`lowe2017multi` dà a ogni
  agente un critico che vede le azioni di tutti, e condizionando su quelle la
  **transizione** non dipende più dalle policy; il **bersaglio** di regressione
  invece sì, perché è un valore atteso sul futuro, ed è per questo che restano
  necessarie le reti target. Le garanzie di convergenza del caso a un agente
  solo, qui, non si trasferiscono. **QMIX** {cite}`rashid2018qmix`
  fattorizza $Q_{tot}$ in modo **monotono** ($\partial Q_{tot}/\partial Q^i \ge 0$),
  così l'argmax individuale coincide con quello congiunto; il prezzo è che i
  giochi non monotoni (accordarsi su una convenzione arbitraria) non si
  rappresentano. **MAPPO** {cite}`yu2022surprising` ricorda che un metodo
  semplice ben regolato va misurato prima di sostituirlo.
- Il **self-play** vale perché il curriculum si genera da solo, restando sempre
  al limite delle proprie capacità: da AlphaGo {cite}`silver2016mastering` ad
  AlphaGo Zero {cite}`silver2017mastering`, che parte dalle sole regole.
- Ma nei giochi **non transitivi** (sasso-carta-forbici) non esiste un ordine
  della bravura, e il self-play ingenuo **cicla**: vince sempre contro la
  versione precedente, perde sempre contro quella di due generazioni prima, e la
  sua sfruttabilità resta al massimo. Il rimedio è allenarsi contro una
  **popolazione**, non contro l'ultimo: è la *league* di AlphaStar
  {cite}`vinyals2019grandmaster`, con agenti principali, sfruttatori e tutte le
  versioni congelate del passato.
```

`````
