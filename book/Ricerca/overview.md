# Ricerca e pianificazione: l’albero dei futuri

Una cassa di ingranaggi e contatti elettrici, in una sala di Parigi, gioca a
scacchi da sola davanti al pubblico. Dentro non c’è nessuno, a differenza del
famoso Turco che aveva girato l’Europa più di un secolo prima: l’ha costruita
l’ingegnere spagnolo Leonardo Torres Quevedo, funziona dal 1912, e quel giorno
del 1914 riconosce da sé dove sono i pezzi, muove i propri e non ha bisogno di
nessuno che le dica che cosa fare.

Gioca una situazione sola, e non è l’inizio della partita ma la sua coda,
quella che negli scacchi si chiama **finale**: da una parte re e torre,
dall’altra il re avversario e nient’altro. Ma lo gioca contro chiunque, da
qualunque posizione, e il matto **lo dà sempre**. Non lo dà in fretta: ci mette
più mosse del necessario, a volte tante da sforare le cinquanta oltre le quali
il regolamento degli scacchi dichiara patta. E non è questo il punto. Il punto
è come fa.

E come fa è la cosa che a noi serve: **non pensa avanti**. Non immagina le
mosse dell’avversario, non prova continuazioni, non valuta niente. Guarda dove
sono i tre pezzi e applica una regola fissa, scritta a mano dentro gli
ingranaggi, del tipo «se il re nemico sta in questa fascia, porta la torre una
riga più in là». Quella regola esiste perché re e torre contro re è un
problema abbastanza piccolo da avere una ricetta: qualcuno l’ha trovata, l’ha
scritta, e la macchina la esegue.

Per la partita intera, di ricetta non ce n’è. Nessuno sa scrivere una regola
che, guardando una posizione qualunque di scacchi, dica quale mossa fare. E
allora bisogna fare l’altra cosa, quella che fa un giocatore umano davanti alla
scacchiera: **immaginare**. Se muovo qui lui risponde là, e allora io potrei…
Questo capitolo è su come si immagina in modo ordinato, che è la prima
grande idea dell’intelligenza artificiale e ha un nome asciutto: **ricerca**.
Guardata dal lato di quello che restituisce, cioè una **sequenza di mosse da
eseguire poi nell’ordine**, la stessa faccenda si chiama **pianificazione**: è
la parola che sta nel titolo, e in questo capitolo le due si possono leggere
come sinonimi.

## Gli stati, le mosse, e l’albero che ne esce

Per immaginare in modo ordinato servono tre cose sole, e sono le stesse per gli
scacchi, per il navigatore satellitare e per quel rompicapo di plastica in cui
si fanno scivolare delle tessere numerate dentro una cornice.

Serve dire **in che situazione ci si trova**, e quella descrizione si chiama
**stato**: la posizione di tutti i pezzi sulla scacchiera, l’incrocio in cui
sono adesso, la disposizione delle tessere. Serve dire **che cosa si può
fare**, cioè quali mosse sono ammesse in quello stato e in quale stato portano.
E serve sapere **quando si è arrivati**, cioè riconoscere lo stato di fine.

Da queste tre cose l’oggetto che nasce è sempre lo stesso. Dallo stato di
partenza si dipartono tante linee quante sono le mosse possibili; da ciascuno
degli stati che ne escono, altrettante; e così via. È un **albero**, con la
radice in alto e i rami che si moltiplicano scendendo. È un albero capovolto
rispetto a quelli veri: la radice sta in cima, e in fondo, alla punta di ogni
ramo, ci sono le **foglie**, cioè le situazioni in cui non si va più avanti
perché la partita è finita. Ogni cammino dalla radice a una foglia è un futuro
possibile.

`````{tab} Elementare

Prendi il rompicapo con le tessere numerate che scorrono in una cornice, quello
in cui c’è una casella vuota e bisogna rimettere i numeri in ordine facendo
scivolare una tessera per volta nel buco.

Lo **stato** è come stanno adesso le tessere. Le **mosse** sono le tessere che
in questo momento confinano con la casella vuota, e sono due, tre o quattro a
seconda di dove il buco si trova. Lo **stato di fine** è i numeri in ordine.

Adesso disegna. In cima metti la situazione di partenza. Sotto, tre caselle,
una per ciascuna mossa che puoi fare. Sotto ciascuna di quelle, altre due o
tre. Dopo quattro righe hai già un centinaio di disegnini, e non sei arrivato
da nessuna parte: la soluzione, per questo rompicapo, sta venti mosse più in
basso.

Quel disegno è l’albero, e la cosa da portarsi via è che **non lo si costruisce
mai tutto**. Si costruisce un pezzetto, si guarda, si decide da che parte
continuare a costruirlo. Tutto il capitolo è su come si sceglie quel pezzetto.

`````

`````{tab} Superiore

Un **problema di ricerca** è definito da cinque componenti: lo spazio degli
stati $\mathcal{S}$; lo stato iniziale $s_0 \in \mathcal{S}$; l’insieme delle
azioni ammesse $\mathcal{A}(s)$ per ogni stato; una funzione di transizione
$\mathrm{ris}(s, a)$ che dice in quale stato si finisce; e un test di
terminazione. Se le azioni hanno costi diversi si aggiunge $c(s, a, s')$, il
costo del passo.

Da questa definizione discende l’**albero di ricerca**, che non va confuso con
lo spazio degli stati. Lo spazio degli stati è un **grafo**: stati diversi si
possono raggiungere per strade diverse, e la stessa posizione può ripresentarsi
(la cosa ha un nome, **trasposizione**, e la sezione sui giochi ci torna).
L’albero di ricerca è invece l’oggetto che l’algoritmo srotola, e in cui lo
stesso stato può comparire in mille punti diversi, uno per ogni cammino che ci
arriva. È l’albero, non il grafo, a esplodere.

Le due misure che governano tutto sono il **fattore di ramificazione** $b$,
cioè quante mosse ci sono in media in uno stato, e la **profondità**, che
conviene distinguere in due: $d$ è quella a cui sta la soluzione, $m$ quella
massima dell’albero. Un albero completo fino a profondità $d$ ha circa $b^d$
nodi, e
questo è l’unico conto del capitolo che valga davvero la pena ricordare: il
costo non cresce con la profondità, **si moltiplica** a ogni livello.

`````

## Perché l’albero esplode, e perché è il problema

Vale la pena mettere dei numeri, perché è la differenza fra un problema
difficile e un problema che non si affronta affatto.

Negli scacchi, in una posizione tipica, le mosse legali sono circa
trentacinque, e una partita dura in media una ottantina di mosse contando
quelle di tutti e due i giocatori. L’albero completo di una partita ha quindi
qualcosa come trentacinque elevato a ottanta nodi: un 3 seguito da altre
centoventitré cifre {cite}`russell2020artificial`.

Un numero così non è «tanto». È un numero che non ha riscontro fisico: gli
atomi dell’universo osservabile si stimano attorno a un 1 seguito da ottanta
zeri, cioè con più di quaranta cifre in meno. Se ogni atomo fosse un
calcolatore che esamina una posizione al secondo dal Big Bang a oggi, l’albero
degli scacchi non sarebbe stato sfiorato.

Questa non è una curiosità da mettere in una didascalia: **è il problema del
capitolo**. Tutto quello che segue esiste per una ragione sola, e cioè per
guardare pochissimo di quell’albero e decidere bene lo stesso. Le due strade
sono quelle che danno il nome alle due sezioni centrali: **guardare nel posto
giusto** (e per farlo serve un fiuto, cioè una stima di quanto manca), e
**smettere di guardare dove non serve** (e per farlo serve accorgersi che un
ramo è già peggio di uno che si conosce).

```{figure} ../figures/albero-dei-futuri.svg
:name: fig-albero-futuri
:alt: "Un albero disegnato con pallini e linee, la radice in alto e i rami che scendono. A sinistra, una etichetta per ciascuna riga: «adesso» accanto al pallino solo in cima, «dopo una mossa» accanto ai tre della riga sotto, «dopo due mosse» accanto ai nove della riga seguente, «dopo tre mosse» accanto ai ventisette dell’ultima, che sono più piccoli e collegati con linee tratteggiate. Sotto, la scritta «e così via». A destra una colonna intestata «quanti sono» riporta 1, 3, 9, 27 e la nota «per tre a ogni riga»; in fondo, in terracotta, «dopo venti mosse 3.486.784.401, dieci cifre, con tre mosse sole»."
:width: 92%

Tre mosse per stato sono poche, e bastano. Il numero a destra non cresce
scendendo: si moltiplica per tre a ogni riga, e dopo venti righe è un numero
con dieci cifre. Con le trentacinque mosse degli scacchi e ottanta righe, le
cifre diventano centoventiquattro.
```

Il numero di destra in {numref}`fig-albero-futuri` è il motivo per cui questo
capitolo non parla mai di costruire l’albero: parla sempre di **quale pezzetto
costruire**.

## Dove sta questo capitolo, e dove finisce

Il libro sta per entrare nella parte in cui una macchina non riconosce e non
scrive: **decide**. E le cose da decidere si dividono secondo che cosa si sa
del mondo, in un modo che vale la pena avere in testa fin da adesso, perché
spiega l’ordine dei prossimi capitoli.

**Il mondo si conosce, ed è piccolo.** Si può passare in rassegna ogni
situazione possibile e calcolare, per ciascuna, quanto vale: cioè segnare
accanto a ogni casella del labirinto quanto conviene trovarcisi. È quello che
fa il capitolo seguente nella sua prima metà, e un labirinto ha poche caselle,
quindi si possono guardare tutte.

**Il mondo si conosce, ed è enorme.** Le situazioni sono più di quante se ne
possano guardare, e allora non si guardano tutte: si guarda in avanti dalla
situazione in cui ci si trova adesso, lungo pochi rami scelti bene, e si decide
solo la mossa da fare subito. È questo capitolo.

**Il mondo non si conosce.** Nessuno ci dice dove porta una mossa né quanto
paga: bisogna provare e vedere come va. È l’apprendimento per rinforzo, cioè il
punto in cui quel capitolo seguente va a finire, ed è la ragione per cui i due
stanno uno accanto all’altro.

## Come è organizzato il capitolo

Tre sezioni. La prima è la ricerca in un mondo che non ha avversari, dove
l’unico nemico è la dimensione: si comincia dal cercare a tentoni, si misura
quanto costa, e si introduce l’unica cosa che cambia davvero le proporzioni,
cioè una **stima di quanto manca** alla fine. Ne esce un algoritmo del 1968 che
si chiama A\* (si legge «a stella»), quello con cui ancora oggi si cercano i
percorsi più brevi, dai navigatori ai videogiochi, e la condizione precisa che
deve valere perché quella stima non faccia sbagliare strada.

La seconda mette un avversario dall’altra parte del tavolo, e cambia tutto:
metà dei rami li sceglie qualcuno che vuole il contrario di quello che
vogliamo noi. Ne escono il modo di ragionare sui giochi a due, la potatura che
permette di ignorare interi rami senza guardarli, e il difetto che tutte le
ricerche a profondità limitata si portano dietro: il disastro che sta un passo
oltre l’ultimo che si è guardato.

La terza fa una cosa sola, e la fa alla fine perché prima non si poteva: mette
in chiaro i tre regali di cui le prime due hanno approfittato senza dirlo (le
**regole**, che si possono interrogare quante volte si vuole; l’**arrivo**, che
si sa riconoscere; il **voto**, che si sa dare a una posizione di mezzo) e li
toglie uno per volta. Ognuno, mancando, porta a un pezzo diverso del libro, e
il più grosso dei tre porta esattamente al capitolo dopo questo.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- La prima macchina che ha giocato a scacchi da sola non pensava affatto:
  seguiva una regola scritta a mano, e poteva farlo perché il finale che
  giocava era abbastanza piccolo da avere una regola.
- Per i problemi che una regola non ce l’hanno bisogna **immaginare i futuri**:
  da dove sono adesso, che cosa succede se faccio questa mossa, e poi quella, e
  poi quell’altra. I futuri immaginati formano un **albero**.
- L’albero **esplode**, e non per poco: il numero di futuri non cresce mano a
  mano che si guarda più avanti, si **moltiplica** a ogni passo. Per gli
  scacchi si arriva a un numero di centoventiquattro cifre, cioè più di quaranta
  cifre in più di quante ne servano per contare gli atomi dell’universo.
- Quindi l’albero non si costruisce mai tutto. Tutto il capitolo è su come si
  sceglie il pezzetto da costruire: **guardare nel posto giusto**, e **smettere
  di guardare dove non serve**.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Un problema di ricerca è (stati, stato iniziale, azioni, transizione, test di
  fine), più eventualmente i costi. Da lì discende l’**albero di ricerca**, che
  è quello che l’algoritmo srotola, distinto dal **grafo** dello spazio degli
  stati in cui la stessa situazione si raggiunge per strade diverse.
- Le due grandezze che governano il costo sono il fattore di ramificazione $b$
  e la profondità: un albero completo fino a $d$ ha $O(b^d)$ nodi. Per gli
  scacchi $b \approx 35$ e la partita è profonda $m \approx 80$, cioè
  $35^{80} \approx 3 \cdot 10^{123}$, un numero di centoventiquattro cifre
  {cite}`russell2020artificial`.
- La ricerca è **pianificazione a modello noto**: si assume di poter
  interrogare la funzione di transizione quante volte si vuole, senza pagare
  pegno. È l’ipotesi che il capitolo sul reinforcement learning toglierà.
- Rispetto alla programmazione dinamica del capitolo seguente, che calcola il
  valore di **tutti** gli stati, qui si guarda in avanti dal solo stato
  corrente e si decide una mossa sola: è ricerca **locale nel tempo**, e si
  paga rifacendola da capo a ogni mossa.
```

`````

Il pezzetto più piccolo che valga la pena costruire è quello che comincia
adesso: un albero senza avversari, in cui l’unica difficoltà è che è grande.
