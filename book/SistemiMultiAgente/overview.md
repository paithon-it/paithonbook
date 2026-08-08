# Sistemi multi-agente: molti che decidono

Chi attraversa Roma in un pomeriggio d'inverno lo ha visto almeno una volta:
poco prima del tramonto migliaia di storni arrivano dalla campagna e sopra gli
alberi del posatoio disegnano forme che si allungano, si piegano e tornano
compatte. Per decenni quello spettacolo è stato materia di stupore, non di
misura: nessuno sapeva dire *che cosa* guardi un singolo storno mentre vira,
perché nessuno era riuscito a seguire un uccello alla volta dentro una nuvola
di migliaia.

A provarci, fra il dicembre 2005 e il febbraio 2006, è stato un gruppo di
fisici dei sistemi complessi (CNR-INFM, Sapienza, Istituto Superiore di Sanità)
dentro il progetto europeo **StarFlag**, coordinato da Giorgio Parisi, che nel
2021 avrebbe ricevuto il Nobel per la fisica per i suoi contributi alla
comprensione dei sistemi fisici complessi; il lavoro sul campo lo guidavano
Andrea Cavagna e Irene Giardina, e fra i dodici autori dell'articolo compare
anche Nicola Cabibbo. Il laboratorio era la terrazza di Palazzo Massimo, al
Museo Nazionale Romano, che guarda gli alberi del posatoio nella piazza davanti
alla stazione Termini: due postazioni fotografiche sincronizzate, a venticinque
metri l'una dall'altra e trenta sopra la strada, per dieci fotogrammi al secondo
(che nessuna macchina reggeva da sola: su ogni postazione ne stavano due, che
scattavano a turno cinque volte al secondo). Da ogni coppia di immagini, con la
stessa geometria con cui due occhi ricavano la profondità, si risale alla
posizione di ciascun uccello nello spazio: dieci eventi di volo ricostruiti in
tre dimensioni, in stormi che arrivavano a 2600 individui.

Il risultato, pubblicato su *PNAS* nel 2008 {cite}`ballerini2008interaction`, è
di quelli che cambiano la domanda. Fino ad allora si dava per scontato che ogni
uccello reagisse ai compagni entro un certo raggio, mettiamo due metri: una
regola **metrica**, la stessa che in fisica lega due particelle vicine. I dati
di Roma dicono altro. Ogni storno tiene d'occhio un numero circa **fisso** di
vicini più prossimi, in media $6{,}5$, cioè sei o sette, e quanto siano
distanti in metri non conta (dentro lo stesso raggio in metri, nello stormo più
fitto dei dieci ci stanno dieci uccelli e nel più rado uno solo): la distanza
che governa lo stormo non si misura in metri ma in *posizioni in classifica*, è
**topologica**. Vent'anni prima Craig Reynolds aveva mostrato che tre regole
locali (stare distanti, allinearsi, restare uniti) bastavano a far volare uno
stormo credibile in computer grafica, i celebri **boids**
{cite}`reynolds1987flocks`; erano regole metriche, e la natura, si scopre, ne
usa una diversa.

## Il collettivo è nella regola

La differenza fra le due regole sembra un dettaglio da pignoli, e invece è
tutto. Le due coincidono finché lo stormo mantiene la stessa densità, e
divergono quando la densità cambia, cioè quando lo stormo si dirada per
sfuggire a un falco: il momento in cui la coesione serve di più.

`````{tab} Elementare

Facciamo il conto su uno stormo che si dirada fino a occupare uno spazio
doppio, con lo stesso numero di uccelli.

Con la regola **metrica** ogni storno guarda dentro una sfera di raggio fisso,
diciamo due metri, dove prima c'erano sette compagni; adesso che lo spazio è
raddoppiato ne restano la metà, e a ogni ulteriore diradamento se ne perdono
altri, finché ciascuno resta solo e lo stormo si sfalda nell'istante peggiore.

Con la regola **topologica** ogni storno guarda i suoi sette vicini più
prossimi, dovunque siano, e sette restano: stanno solo un po' più lontano, e
per ritrovarli in uno spazio doppio basta allargare lo sguardo di un quarto, da
due metri a due metri e mezzo. Il legame regge perché la regola non parla di
metri: parla di *quanti*, e quanti restano quanti anche quando il gruppo si
allarga.

`````

`````{tab} Superiore

Sia $\rho$ la densità locale (uccelli per unità di volume). Con una regola
metrica di raggio $r$ il numero di vicini con cui un individuo interagisce è
$n(r) = \tfrac{4}{3}\pi r^{3} \rho$, proporzionale a $\rho$: il grado di
interazione collassa quando lo stormo si dirada. Con una regola topologica si
fissa invece $n_c$ e a variare è il raggio implicito,

$$
r_c \simeq \left(\frac{3\,n_c}{4\pi\rho}\right)^{1/3} \propto \rho^{-1/3},
$$

che dipende dalla densità solo come l'inverso della sua radice cubica:
dimezzare $\rho$ allunga $r_c$ di $2^{1/3} \approx 1{,}26$, il 26%. Il grado
resta costante per costruzione, ed è questa invarianza a garantire la coesione
sotto grandi variazioni di densità.

Il test empirico usa proprio questa differenza. Su dieci stormi di rarefazione
molto diversa ($r_1$, la distanza media dal primo vicino, va da $0{,}68$ a
$1{,}51$ m) il raggio di interazione $r_c$ cresce con $r_1$ in modo netto
($R^2 = 0{,}78$), mentre il numero di vicini interagenti non mostra alcuna
correlazione con la rarefazione ($n_c^{-1/3}$ contro $r_1$: $R^2 = 0{,}0002$).
Il raggio segue la densità, il grado no: in media
$n_c = 6{,}5 \pm 0{,}9$ (errore standard) {cite}`ballerini2008interaction`. Nel
linguaggio delle Graph Neural Network, la regola metrica costruisce un grafo a
raggio fisso e la topologica il grafo **diretto** dei $k$ vicini più prossimi,
in cui ogni nodo sceglie i propri $k$ archi uscenti: il grado uscente è
costante per costruzione (quello entrante no, perché la scelta non va
ricambiata: io guardo te senza che tu debba guardare me, che è la natura stessa
dell'interazione fra storni) e il grafo non cambia affatto se tutte le distanze
vengono riscalate.

`````

Da qui la tesi che percorre tutto il capitolo, e che vale ben oltre gli
uccelli: **il comportamento di un gruppo è una proprietà della regola di
interazione, non della bravura dei singoli**. Gli storni sono gli stessi; cambia
la regola con cui ciascuno guarda i vicini, e cambia lo stormo: uno tiene,
l'altro si sbriciola. Ogni sezione che segue non fa altro che declinare questa
frase su agenti software: dieci agenti identici, collegati in modi diversi, sono
dieci sistemi diversi.

## Perché adesso

Il campo non è nuovo. Gli agenti sono un filone dell'intelligenza artificiale
dagli anni Settanta, e l'intelligenza artificiale *distribuita* ha passato
trent'anni su come far cooperare programmi separati: linguaggi di
comunicazione, protocolli di negoziazione, meccanismi di consenso. Quel lavoro
esiste, e conviene non riscoprirlo male.

Quello che è cambiato è il costo di partenza. Con un modello di linguaggio,
istanziare dieci agenti costa una riga di codice: lo stesso motore, dieci
prompt di sistema diversi, e la squadra è in piedi. Il capitolo sugli
**Agenti** lo ha già mostrato con i ruoli specializzati (un pianificatore, un
esecutore, un critico) e con l'idea di modellare il sistema come una
conversazione fra agenti, come fa il framework AutoGen {cite}`wu2024autogen`.
Ma se creare i partecipanti è gratis, tutta la difficoltà si sposta altrove: su
**chi parla con chi**, su **chi decide** quando le proposte sono in
disaccordo, e su una domanda che con un agente solo non si pone, **come ci si
accorge che il gruppo nel suo insieme ha sbagliato**.

L'ultima è la meno ovvia. Quando sbaglia un gruppo ogni singolo pezzo sembra a
posto: ciascuno ha fatto il proprio turno, i messaggi sono ben scritti, e il
risultato è sbagliato perché una specifica si è deformata lungo la catena o
perché una risposta plausibile e falsa non è stata contestata da nessuno. Il
guasto sta nella conversazione, non nei turni, e su come misurarlo la
letteratura è ancora sottile {cite}`xi2023rise`.

## Molti battono uno solo se sbagliano in modo diverso

Prima di progettare squadre, va chiarita l'ipotesi nascosta sotto l'idea stessa
che «più teste ragionino meglio di una». Non è sempre vera, e la condizione che
la rende vera è una sola, precisa, e facilissima da violare quando gli agenti
sono costruiti tutti allo stesso modo.

`````{tab} Elementare

Tre colleghi rispondono a una domanda difficile e ognuno, da solo, ci prende
sette volte su dieci. Decidendo a maggioranza il gruppo ci prende quasi otto
volte su dieci (78%), e con nove colleghi il 90%: perché il gruppo sbagli
servono almeno due errori insieme, che sono più rari di uno.

Il conto però vale solo se i tre sbagliano in modo *diverso*. Se hanno studiato
sugli stessi appunti sbagliati sbagliano insieme, la maggioranza conferma
l'errore invece di correggerlo e il gruppo resta al 70% del singolo: tre
stipendi per il risultato di uno. E se ciascuno ci prende quattro volte su
dieci, votare *peggiora* le cose: tre danno il 35%, nove il 27%. Il voto
amplifica la tendenza di fondo, qualunque sia.

`````

`````{tab} Superiore

È il **teorema della giuria di Condorcet** (1785). Con $n$ votanti indipendenti,
ciascuno corretto con probabilità $p$, e decisione a maggioranza semplice, la
probabilità che il gruppo abbia ragione è

$$
P_n = \sum_{k=\lceil (n+1)/2 \rceil}^{n} \binom{n}{k}\, p^{k} (1-p)^{\,n-k},
$$

dove $\binom{n}{k}$ è il numero di modi in cui $k$ votanti su $n$ possono
azzeccare. L'andamento asintotico è una dicotomia: al crescere di $n$,
$P_n \to 1$ se $p > 1/2$ e $P_n \to 0$ se $p < 1/2$. Con $p = 0{,}7$ si ha
$P_3 = 0{,}784$, $P_5 = 0{,}837$, $P_9 = 0{,}901$; con $p = 0{,}4$,
$P_3 = 0{,}352$ e $P_9 = 0{,}267$.

L'ipotesi vincolante è l'**indipendenza degli errori**, ed è la più fragile che
ci sia fra agenti che condividono il modello di base, i dati di
pre-addestramento e spesso metà del prompt. Nel limite di correlazione perfetta
$P_n = p$ per ogni $n$: la maggioranza di $n$ agenti vale un agente,
moltiplicandone il costo. È la lezione degli **ensemble** del capitolo sul
machine learning, dove il guadagno non viene dal numero di modelli ma dalla
loro decorrelazione. Diversità prima di quantità; quanto costi ottenerla è il
tema della prossima sezione.

`````

## Quello che sappiamo già

Questo capitolo poggia su tre capitoli precedenti. Dagli **Agenti** vengono il
ciclo osserva-ragiona-agisci e i ruoli specializzati: qui diamo per acquisito
il singolo agente e studiamo ciò che nasce quando sono molti. Dal
**Reinforcement Learning** vengono l'MDP, la policy $\pi$ e l'assegnazione del
merito, che con più agenti si sdoppia: non solo *quale mossa* ha prodotto il
risultato, ma *quale agente*. Dalle **GAN**, infine, viene qualcosa che si fa
fatica a riconoscere come multi-agente e invece lo è in pieno: due giocatori
con obiettivi opposti, che non convergono a un minimo ma a un equilibrio.

`````{tab} Elementare

Nella GAN c'erano due reti in gara, un falsario e un detective, e la fine della
partita non era un traguardo ma un pareggio: il punto in cui a nessuno dei due
conviene più cambiare mossa. Un sistema multi-agente allarga quella struttura:
i giocatori possono essere dieci, e non sono per forza nemici. E imparare
diventa più difficile, perché un agente solo studia in un mondo fermo mentre
dieci agenti studiano in un mondo che si muove: il «mondo» di ciascuno contiene
gli altri nove, che stanno imparando anche loro. È come preparare un esame in
cui il programma cambia perché i tuoi compagni studiano.

`````

`````{tab} Superiore

Il quadro formale generalizza l'MDP a un **gioco stocastico**: $N$ agenti, uno
spazio di stati $S$, spazi di azione $A^1, \dots, A^N$, una transizione
$P(s' \mid s, a)$ che dipende dall'azione **congiunta**
$a = (a^1, \dots, a^N)$ e una ricompensa $r^i$ per ciascun agente. Se
$r^i = r$ per ogni $i$ il gioco è cooperativo; se $N = 2$ e $r^1 + r^2 = 0$ si
ricade nel caso a somma zero della GAN, dove l'obiettivo non è un minimo di
$\mathcal{L}$ ma un **equilibrio di Nash**: un profilo $(\pi^1, \dots, \pi^N)$
in cui nessun agente migliora il proprio ritorno atteso cambiando policy da
solo. Ne segue che il caso multi-agente non è quello singolo ripetuto $N$
volte: per l'agente $i$ l'ambiente comprende le policy $\pi^{-i}$ degli altri,
che cambiano durante l'addestramento, quindi il processo che $i$ osserva **non
è stazionario** e le garanzie di convergenza del Q-learning, che presuppongono
un MDP fisso, decadono. Lo affronta la sezione «Imparare insieme».

`````

## Tre domande, cinque sezioni

Il capitolo risponde a tre domande, in quest'ordine. **Conviene davvero più di
un agente**, e a che prezzo? **Come si organizzano**, cioè chi parla con chi,
con quali messaggi e con quale regola di decisione? E infine: **possono
imparare a coordinarsi** invece di essere programmati per farlo, come gli
storni, a cui la regola dei sei o sette vicini non l'ha insegnata nessuno?

- **Il costo del coordinamento**: quando più agenti battono un singolo agente
  ben progettato, con i conti in mano (chiamate, token, giri di conversazione).
- **Topologie: chi parla con chi**: le forme del grafo di comunicazione
  (catena, stella con un coordinatore, dibattito, gerarchia) e cosa ciascuna fa
  a costo, latenza e qualità.
- **Protocolli e consenso**: come si parla (messaggi tipizzati, atti
  linguistici) e come si decide (voto, quorum), fino al caso duro in cui un
  partecipante si guasta o mente: i generali bizantini.
- **Imparare insieme**: l'apprendimento per rinforzo multi-agente,
  l'addestramento centralizzato con esecuzione decentralizzata e il *self-play*
  incontrato dietro AlphaGo.
- **Sciami e simulazioni**: regole locali elementari che risolvono
  problemi globali (colonie di formiche, sciami di particelle) e le società
  simulate, dove l'oggetto di studio è il collettivo stesso.

```{admonition} Da ricordare
:class: important
- Gli stormi di storni sopra Termini, ricostruiti in 3D dal progetto
  **StarFlag** {cite}`ballerini2008interaction`, seguono una regola
  **topologica**: ogni uccello guarda un numero fisso di vicini più prossimi
  (sei o sette), non tutti quelli entro un raggio in metri.
- Così lo stormo resta unito anche quando si dirada, perché il grado di
  interazione non dipende dalla densità ($r_c \propto \rho^{-1/3}$); una regola
  metrica, come quella dei **boids** {cite}`reynolds1987flocks`, tiene molto
  meno, come mostrano le simulazioni dello stesso lavoro.
- La tesi del capitolo: **il comportamento del gruppo è una proprietà della
  regola di interazione, non della bravura dei singoli**. Stessi individui,
  regola diversa, collettivo diverso.
- Istanziare dieci agenti costa una riga di codice {cite}`wu2024autogen`; il
  difficile è chi parla con chi, chi decide, e accorgersi che ha sbagliato il
  *gruppo* e non un turno {cite}`xi2023rise`.
- «Più teste» aiuta solo se sbagliano in modo **indipendente** (Condorcet: con
  $p = 0{,}7$, tre votanti danno $0{,}784$); con errori perfettamente correlati
  $n$ agenti valgono quanto uno, al costo di $n$.
- Il caso multi-agente non è quello singolo ripetuto: per ciascun agente
  l'ambiente contiene gli altri, che cambiano, e quindi **non è stazionario**;
  l'obiettivo diventa un **equilibrio**, non un minimo.
```
