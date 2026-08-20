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
dentro il progetto europeo **StarFlag**. Il lavoro sul campo lo guidava Andrea
Cavagna; l'algoritmo con cui gli uccelli vennero riconosciuti da un'immagine
all'altra lo disegnarono lui e Irene Giardina, e a scriverne il codice fu
Massimiliano Viale; fra i dodici autori dell'articolo
compaiono anche Nicola Cabibbo e Giorgio Parisi, che nel 2021 avrebbe ricevuto
il Nobel per la fisica per i suoi contributi alla comprensione dei sistemi
fisici complessi.

Il laboratorio era la terrazza di Palazzo Massimo, al Museo Nazionale Romano,
che guarda gli alberi del posatoio nella piazza davanti alla stazione Termini.
Lassù, trenta metri sopra la strada, montarono tre postazioni fotografiche che
scattavano tutte nello stesso istante: due lontane venticinque metri l'una
dall'altra, la terza a soli due metri e mezzo da una delle prime due. Servivano
dieci fotogrammi al secondo, e nessuna macchina di allora ci arrivava da sola:
su ogni postazione ne misero due, che si alternavano cinque scatti per uno.

Due fotografie scattate nello stesso istante bastano a dire dove sta un uccello
nello spazio, con la stessa geometria con cui due occhi ricavano la profondità.
La **terza** postazione serve ad altro, e serve alla cosa che finora aveva reso
lo spettacolo impossibile da misurare: capire quale puntino di una foto sia lo
stesso uccello di quale puntino dell'altra, quando i puntini sono migliaia e si
somigliano tutti. Con due punti di vista le coppie plausibili sono troppe; il
terzo le taglia, perché una coppia sbagliata cade nel posto sbagliato sulla
terza immagine. Vengono così ricostruiti in tre dimensioni **dieci stormi**,
ciascuno ripreso per qualche secondo, alcuni dei quali arrivavano a 2600
individui.

Il risultato, pubblicato su *PNAS* nel 2008 {cite}`ballerini2008interaction`, è
di quelli che cambiano la domanda. Fino ad allora si dava per scontato che ogni
uccello reagisse ai compagni entro un certo raggio, mettiamo due metri: chi sta
dentro conta, chi sta fuori no. È una regola **metrica**, cioè fatta di metri.
I dati di Roma dicono altro. Ogni storno tiene d'occhio un numero circa
**fisso** di vicini più prossimi, sei o sette (la media sui dieci stormi è sei
e mezzo), e quanto quei vicini siano lontani non conta. La differenza fra le due
regole non è sottile, e si vede confrontando gli stormi fra loro: a parità di
raggio, nel più fitto dei dieci ci stanno **dieci volte** gli uccelli che ci
stanno nel più rado (nel confronto che fanno gli autori, dieci contro uno).
Eppure la regola che ciascuno segue è la stessa. La distanza che governa lo
stormo, insomma, non si misura in metri ma in *posizioni in classifica*, e per
dirlo si usa la parola **topologica**, che qui vuol dire soltanto questo: conta
il posto in graduatoria, non quanti metri ci sono di mezzo.

Vent'anni prima Craig Reynolds aveva mostrato che tre regole locali (stare
distanti, allinearsi, restare uniti) bastavano a far volare uno stormo credibile
in computer grafica, i celebri **boids** {cite}`reynolds1987flocks`; erano regole
metriche, e la natura, si scopre, ne usa una diversa.

## Il collettivo è nella regola

La differenza fra le due regole sembra un dettaglio da pignoli, e invece è
tutto. Finché lo stormo resta fitto sempre allo stesso modo le due danno lo
stesso risultato, e nessuno saprebbe distinguerle. Si separano quando lo stormo
si allarga, cioè quando si dirada per sfuggire a un falco: proprio il momento in
cui restare insieme conta di più.

`````{tab} Elementare

Facciamo il conto su uno stormo che si dirada fino a occupare uno spazio
doppio, con lo stesso numero di uccelli.

Con la regola **metrica** ogni storno guarda dentro una sfera di raggio fisso,
diciamo due metri, dove prima c'erano sette compagni; adesso che lo spazio è
raddoppiato gliene restano tre o quattro, e a ogni ulteriore diradamento se ne
perdono altri, finché ciascuno resta solo e lo stormo si sfalda nell'istante
peggiore.

Con la regola **topologica** ogni storno guarda i suoi sette vicini più
prossimi, dovunque siano, e sette restano: stanno solo un po’ più lontano.

Quanto più lontano? Meno di quanto verrebbe da dire, e vale la pena farlo
vedere con dei cubetti, perché è il punto su cui poggia tutto il resto. Una
scatola larga due metri, per due, per due contiene otto metri cubi. Per
contenerne il doppio, sedici, **non** serve una scatola larga il doppio: quella
ne conterrebbe sessantaquattro, cioè otto volte tanto, perché allargando il lato
si allargano insieme le tre direzioni. Proviamo allora ad allargare di poco.
Portiamo il lato da due metri a due e mezzo, un quarto in più: due e mezzo per
due e mezzo per due e mezzo fa quindici metri cubi e sei, e i sedici che
cercavamo sono lì a un soffio. Ecco perché per ritrovare i suoi sette compagni
in uno spazio doppio a uno storno basta allargare lo sguardo di un quarto: lo
spazio cresce molto più in fretta della distanza.

Il legame regge perché la regola non parla di metri: parla di *quanti*, e
quanti restano quanti anche quando il gruppo si allarga.

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
correlazione con la rarefazione ($n_c^{-1/3}$ contro $r_1$: $R^2 = 0{,}00021$).
Il raggio segue la densità, il grado no: in media
$n_c = 6{,}5 \pm 0{,}9$ (errore standard) {cite}`ballerini2008interaction`. Nel
linguaggio dei grafi, che il capitolo sulle Graph Neural Network riprenderà per
esteso, la regola metrica costruisce un grafo a
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
l'altro si sbriciola. Ogni sezione che segue non fa altro che ripetere questa
frase su programmi invece che su uccelli. Un **agente**, qui, è un programma a
cui si affida un compito e che lo porta avanti da sé, decidendo un passo alla
volta che cosa fare; e dieci agenti identici, a seconda di chi può scrivere a
chi, sono dieci sistemi diversi.

## Perché adesso

Il campo non è nuovo. Gli agenti sono un filone dell'intelligenza artificiale
dagli anni Settanta, e l'intelligenza artificiale *distribuita* ha passato
trent'anni su come far cooperare programmi separati: che lingua parlano fra
loro, come si accordano su chi fa che cosa, come decidono quando le loro
risposte non coincidono. Sono le domande di questo capitolo, e hanno già
trent'anni di risposte: conviene non riscoprirle male.

Quello che è cambiato è il costo di partenza. Un **modello di linguaggio** è un
programma che, dato un testo, ne scrive il seguito, e lo fa abbastanza bene da
poter ricevere le istruzioni a parole invece che in codice. Farne nascere dieci
copie non costa quasi niente, perché il programma è sempre lo stesso: a
distinguerle è soltanto il foglio di istruzioni che ciascuna si trova davanti
prima di cominciare (il *prompt di sistema*: «tu scrivi codice e non discuti le
scelte altrui», «tu cerchi errori e non ne proponi la correzione»). Dieci fogli
diversi, e la squadra è in piedi.

Conviene fissare subito un esempio, perché nelle prossime pagine si parlerà a
lungo di quanto costa una squadra e di che forma darle, e il prezzo di una cosa
non dice niente finché non si sa che cosa sia. Prendiamo la richiesta: «apri
questo file di vendite e dimmi quali negozi stanno peggiorando». Un agente la
riceve e tiene le fila. Un secondo scrive il programma che apre il file
e fa i conti. Un terzo lo legge e dice soltanto una cosa: se è sicuro
eseguirlo, cioè se non cancella niente e non combina danni. A quel punto tocca
di nuovo al primo, perché è l'unico dei tre che può toccare la macchina vera:
gli altri due scrivono e leggono testo, lui esegue. Il risultato torna al
secondo, che lo interpreta e scrive la risposta. È il sistema di programmazione
presentato insieme ad AutoGen {cite}`wu2024autogen`, uno dei programmi con cui
queste squadre si mettono in piedi, ed è la squadra a cui pensare ogni volta che
in questo capitolo si parla di agenti che si passano messaggi. Il capitolo sugli **Agenti**
ha già descritto quei ruoli uno per uno (un pianificatore, un esecutore, un
critico); qui si studia che cosa succede quando sono insieme.

Ma se creare i partecipanti è gratis, tutta la difficoltà si sposta altrove: su
**chi parla con chi**, su **chi decide** quando le proposte sono in
disaccordo, e su una domanda che con un agente solo non si pone, **come ci si
accorge che il gruppo nel suo insieme ha sbagliato**.

L'ultima è la meno ovvia. Quando sbaglia un gruppo ogni singolo pezzo sembra a
posto: ciascuno ha fatto il proprio turno, i messaggi sono ben scritti, e il
risultato è sbagliato per una di due ragioni. O la richiesta di partenza si è
deformata passando di mano in mano, e l'ultimo ha risposto benissimo a una
domanda diversa da quella iniziale; oppure qualcuno ha detto una cosa falsa
detta bene, e nessuno l'ha contestata. Il guasto sta nella conversazione, non
nei singoli turni, e su come misurarlo si è ancora scritto poco
{cite}`xi2023rise`.

## Molti battono uno solo se sbagliano in modo diverso

Prima di progettare squadre, va chiarita l'ipotesi nascosta sotto l'idea stessa
che «più teste ragionino meglio di una». Non è sempre vera, e la condizione che
la rende vera è una sola, precisa, e facilissima da violare quando gli agenti
sono costruiti tutti allo stesso modo.

`````{tab} Elementare

Tre colleghi rispondono a una domanda difficile e ognuno, da solo, ci prende
sette volte su dieci. Decidendo a maggioranza il gruppo ci prende quasi otto
volte su dieci (78%), e con nove colleghi il 90%: perché il gruppo sbagli
servono almeno due errori insieme, che sono più rari di uno. Da dove esca
esattamente quel 78 lo vedremo elencando i casi uno per uno nella
sezione «Protocolli e consenso» (il 90 esce allo stesso modo, con molti più
casi da elencare): è un conto che si fa a mano e vale la pena
rifarlo, ma per adesso basta il senso.

Il conto però vale solo se i tre sbagliano in modo *diverso*. Se hanno studiato
sugli stessi appunti sbagliati sbagliano insieme, la maggioranza conferma
l'errore invece di correggerlo e il gruppo resta al 70% del singolo: tre
stipendi per il risultato di uno. E se ciascuno ci prende quattro volte su
dieci, votare *peggiora* le cose: tre danno il 35%, nove il 27%. Il voto
amplifica la tendenza di fondo, qualunque sia. (Anche questi due numeri escono
dall'elenco dei casi di cui sopra: è lo stesso conto, fatto partendo da quattro
volte su dieci invece che da sette.)

`````

`````{tab} Superiore

È il **teorema della giuria di Condorcet** (1785). Con $n$ votanti indipendenti
che scelgono fra **due** alternative, ciascuno corretto con probabilità $p$, e
decisione a maggioranza semplice, la probabilità che il gruppo abbia ragione è

$$
P_n = \sum_{k=\lfloor n/2 \rfloor + 1}^{n} \binom{n}{k}\, p^{k} (1-p)^{\,n-k},
$$

dove $\binom{n}{k}$ è il numero di modi in cui $k$ votanti su $n$ possono
azzeccare e la somma parte dalla più piccola maggioranza stretta. L'andamento
asintotico è una dicotomia: al crescere di $n$,
$P_n \to 1$ se $p > 1/2$ e $P_n \to 0$ se $p < 1/2$. Con $p = 0{,}7$ si ha
$P_3 = 0{,}784$, $P_5 = 0{,}837$, $P_9 = 0{,}901$; con $p = 0{,}4$,
$P_3 = 0{,}352$ e $P_9 = 0{,}267$.

L'ipotesi vincolante è l’**indipendenza degli errori**, ed è la più fragile che
ci sia fra agenti che condividono il modello di base, i dati di
pre-addestramento e spesso metà del prompt. Nel limite di correlazione perfetta
$P_n = p$ per ogni $n$: la maggioranza di $n$ agenti vale un agente,
moltiplicandone il costo. È la lezione degli **ensemble** del capitolo sul
machine learning, dove il guadagno non viene dal numero di modelli ma dalla
loro decorrelazione. Diversità prima di quantità; quanto costi ottenerla è il
tema della prossima sezione.

`````

## Quello che sappiamo già, e una cosa che sapremo dopo

Questo capitolo poggia su due capitoli precedenti, e ne anticipa uno. Dagli
**Agenti** vengono il ciclo osserva-ragiona-agisci e i ruoli specializzati: qui
diamo per acquisito il singolo agente e studiamo ciò che nasce quando sono
molti. Dal **Reinforcement Learning** viene il processo decisionale di Markov
(il modo di descrivere un mondo in cui si osserva una situazione, si sceglie una
mossa, si incassa un premio e si finisce nella situazione successiva),
insieme alla *policy* (la regola con cui un agente sceglie che mossa fare in una
data situazione) e
all’**assegnazione del merito**: la ricompensa arriva alla fine di una partita e
bisogna capire quale delle mosse se la sia guadagnata. Con più agenti quella
domanda si sdoppia, e non chiede più soltanto *quale mossa* ha prodotto il
risultato, ma anche *quale agente*.

Il terzo arriva più avanti nel libro: sono le **GAN**, l'esempio più puro
della cosa che questo capitolo studia, cioè due parti che si spingono a
vicenda a migliorare. Ogni volta che serviranno diremo per esteso quel che c'è
da saperne.

`````{tab} Elementare

Più avanti nel libro incontrerai due reti che si allenano l'una contro l'altra:
una fabbrica immagini false, l'altra cerca di smascherarle. Si chiamano GAN, e
di loro qui basta sapere come finisce la partita. Addestrare un programma, di
solito, somiglia a cercare il punto più basso di una valle nella nebbia: si
scende, e quando non si scende più si è arrivati. Qui no, perché la discesa di
uno è la salita dell'altro e un fondo non c'è. Quello che si può sperare è un
**pareggio**, cioè il momento in cui a nessuno dei due conviene più cambiare
mossa, perché a qualunque mossa l'altro saprebbe rispondere.

Un sistema multi-agente allarga quella struttura: i giocatori possono essere
dieci, e non sono per forza nemici. E imparare diventa più difficile, perché un
agente solo studia in un mondo fermo mentre dieci agenti studiano in un mondo
che si muove: il «mondo» di ciascuno contiene gli altri nove, che stanno
imparando anche loro. È come preparare un esame in cui il programma cambia
perché i tuoi compagni studiano.

`````

`````{tab} Superiore

Il quadro formale generalizza l'MDP a un **gioco stocastico** (qui e altrove
$\pi$ è la policy, e non ha niente a che vedere con il $3{,}14$ della
circonferenza): $N$ agenti, uno
spazio di stati $\mathcal{S}$, spazi di azione
$\mathcal{A}^1, \dots, \mathcal{A}^N$, una transizione
$P(s' \mid s, a)$ che dipende dall'azione **congiunta**
$a = (a^1, \dots, a^N)$ e una ricompensa $r^i$ per ciascun agente. Se
$r^i = r$ per ogni $i$ il gioco è cooperativo; se $N = 2$ e $r^1 + r^2 = 0$ si
ricade nel caso a somma zero, che è la forma **minimax** della GAN (con la
*loss* non-saturante che si usa in pratica la somma non è più zero, e il
capitolo sulle GAN spiega perché), dove l'obiettivo non è un minimo di
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
  ben progettato, con i conti in mano (quante volte si interroga il modello,
  quanto testo gli si fa rileggere, quanti giri di conversazione servono).
- **Chi parla con chi**: le forme che può prendere lo schema di chi scrive a chi
  (catena, stella con un coordinatore, dibattito, gerarchia) e che cosa ciascuna
  fa al costo, al tempo di attesa e alla qualità.
- **Protocolli e consenso**: come si parla, cioè come si scrive su un messaggio
  che cosa quel messaggio fa; e come si decide (voto di maggioranza, dibattito),
  fino al caso duro in cui un partecipante si guasta o mente.
- **Imparare insieme**: l'apprendimento per rinforzo multi-agente,
  l'addestramento centralizzato con esecuzione decentralizzata e il *self-play*
  incontrato dietro AlphaGo.
- **Sciami e simulazioni**: regole locali elementari che risolvono
  problemi globali (colonie di formiche, sciami di particelle) e le società
  simulate, dove l'oggetto di studio è il collettivo stesso.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Gli stormi di storni sopra Termini, ricostruiti in tre dimensioni dal progetto
  **StarFlag** {cite}`ballerini2008interaction`, seguono una regola
  **topologica**: ogni uccello tiene d'occhio un numero fisso di vicini, i sei o
  sette più prossimi, e non tutti quelli che gli stanno entro due metri.
- È questo che tiene insieme lo stormo quando si dirada. Con una regola a metri
  i compagni dentro il raggio si dimezzano appena lo spazio raddoppia; contando
  i vicini invece che misurandoli, sette restano sette, e per ritrovarli basta
  allargare lo sguardo di un quarto. Le regole a metri, come quella dei **boids**
  {cite}`reynolds1987flocks`, tengono molto meno.
- La tesi del capitolo: **il comportamento del gruppo è una proprietà della
  regola di interazione, non della bravura dei singoli**. Stessi individui,
  regola diversa, collettivo diverso.
- Far nascere dieci agenti costa una riga di codice: stesso modello, dieci fogli
  di istruzioni diversi {cite}`wu2024autogen`. Il difficile viene dopo: chi parla
  con chi, chi decide quando le risposte non coincidono, e come ci si accorge che
  a sbagliare è stato il *gruppo* e non un singolo turno {cite}`xi2023rise`.
- «Più teste» aiuta solo se sbagliano in modo **diverso**. Tre persone che ci
  prendono sette volte su dieci, votando, ci prendono quasi otto volte su dieci;
  ma se hanno studiato sugli stessi appunti sbagliati sbagliano insieme, e dieci
  agenti valgono quanto uno, al costo di dieci.
- Il caso con molti agenti non è quello singolo ripetuto tante volte: per
  ciascuno il mondo contiene gli altri, che nel frattempo cambiano. Il traguardo
  non è più il fondo di una valle, è un **pareggio**: la situazione in cui a
  nessuno conviene più muoversi da solo.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Gli stormi di storni sopra Termini, ricostruiti in 3D dal progetto
  **StarFlag** {cite}`ballerini2008interaction`, seguono una regola
  **topologica**: ogni uccello guarda un numero fisso di vicini più prossimi
  ($n_c = 6{,}5 \pm 0{,}9$), non tutti quelli entro un raggio in metri.
- Così lo stormo resta unito anche quando si dirada, perché il grado di
  interazione non dipende dalla densità ($r_c \propto \rho^{-1/3}$); una regola
  metrica, come quella dei **boids** {cite}`reynolds1987flocks`, tiene molto
  meno, e lo mostrano le simulazioni dello stesso lavoro, dove uno stormo a
  regola metrica si spezza in più tronconi molto più spesso di uno topologico.
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
  l'obiettivo diventa un **equilibrio di Nash**, non un minimo di $\mathcal{L}$.
```

`````
