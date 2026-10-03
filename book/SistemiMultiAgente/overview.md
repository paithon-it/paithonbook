# Sistemi multi-agente: molti che decidono

```{image} ../figures/aperture/sistemi-multi-agente.png
:class: pt-apertura only-light
:width: 100%
:alt: Un grande stormo di storni disegna una forma fluida nel cielo, sopra i pini a ombrello e la sagoma di una cupola.
```

```{image} ../figures/aperture/sistemi-multi-agente-scura.png
:class: pt-apertura only-dark
:width: 100%
:alt: Un grande stormo di storni disegna una forma fluida nel cielo, sopra i pini a ombrello e la sagoma di una cupola.
```

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
Massimiliano Viale; fra i dodici che firmarono il risultato
compaiono anche Nicola Cabibbo e Giorgio Parisi, che nel 2021 avrebbe ricevuto
metà del Nobel per la fisica per avere scoperto come il disordine e le
fluttuazioni (le piccole variazioni casuali attorno al valore medio) si
intreccino nei sistemi fisici, dalla scala atomica a quella planetaria.

Il laboratorio era la terrazza di Palazzo Massimo, al Museo Nazionale Romano,
che guarda gli alberi del posatoio nella piazza davanti alla stazione Termini.
Lassù, trenta metri sopra la strada, montarono tre postazioni fotografiche che
scattavano tutte nello stesso istante: due lontane venticinque metri l'una
dall'altra, la terza a soli due metri e mezzo da una delle prime due. Volevano
dieci fotogrammi al secondo, ma oltre i cinque le loro macchine non riuscivano
più a scattare tutte nello stesso istante. Su ogni postazione ne misero allora
due, sei macchine in tutto, ciascuna a cinque scatti al secondo, e gli scatti
dell'una cadevano esattamente a metà fra quelli dell'altra.

Due fotografie scattate nello stesso istante bastano a dire dove sta un uccello
nello spazio, con la stessa geometria con cui due occhi ricavano la profondità.
La terza postazione serve ad altro, e serve alla cosa che finora aveva reso lo
spettacolo impossibile da misurare: capire quale puntino di una foto sia lo
stesso uccello di quale puntino dell'altra, quando i puntini sono migliaia e si
somigliano tutti. Fra le due postazioni lontane l'algoritmo riusciva ad
abbinare un uccello su cinque, troppo pochi; fra due vicine, che vedono quasi
la stessa scena, nove su dieci. Ma due macchine vicine sono come due occhi
quasi sovrapposti, e della profondità misurano poco: a misurarla bene serve la
coppia lontana. Ecco allora a che cosa serve la terza: gli abbinamenti si fanno
fra le due vicine, dove sono facili, e la geometria delle tre viste li
trasferisce sulla coppia lontana. Vengono così ricostruiti in tre dimensioni
dieci stormi, ciascuno ripreso per qualche secondo, e il più numeroso arrivava a
2600 individui.

Il risultato, pubblicato nel 2008 sulla rivista dell'Accademia delle scienze
statunitense, *PNAS* {cite}`ballerini2008interaction`, è di quelli che
cambiano la domanda. Quasi tutti i modelli davano per scontato che ogni uccello
reagisse ai compagni entro un certo raggio, mettiamo due metri: chi sta dentro
conta, chi sta fuori no. È una regola **metrica**, cioè fatta di metri. Anche i
modelli che contavano i vicini invece di misurarli continuavano a pesare di più
i vicini e di meno i lontani, e così la distanza in metri rientrava nella
regola.

I dati di Roma dicono altro. Ogni storno interagisce con un numero circa fisso
di vicini, sei o sette (la media sui dieci stormi è sei e mezzo), comunque
siano lontani. Lo si vede confrontando gli stormi fra loro: dentro uno stesso
raggio il più fitto dei dieci contiene dieci volte gli uccelli del più rado, e
se la regola fosse in metri, nel primo ogni storno avrebbe dieci volte i
compagni con cui interagire; invece ne ha sei o sette nell'uno e nell'altro. La
distanza che governa lo stormo è quindi **topologica**, e la parola qui vuol
dire soltanto questo: conta il numero d'ordine del vicino (il primo, il
secondo, fino al settimo più vicino), non quanti metri ci sono di mezzo.

Vent'anni prima Craig Reynolds aveva mostrato che tre regole locali (stare
distanti, allinearsi, restare uniti) bastavano a far volare uno stormo credibile
in computer grafica: i **boids** {cite}`reynolds1987flocks`, come lui chiamò i
suoi uccelli simulati, da *bird-oid*, «simile a uccello». Erano regole
metriche, e la natura, si scopre, ne usa una diversa. Al posto degli uccelli,
fra poco, ci saranno programmi che si scrivono messaggi, e la domanda
resterà la stessa: quanto del comportamento di un gruppo dipende da come
ciascuno guarda gli altri.

## Il collettivo è nella regola

Finché la densità dello stormo resta la stessa, le due regole prevedono lo
stesso comportamento e nessuna osservazione saprebbe distinguerle. Si separano
quando la densità cambia, cioè quando lo stormo si dirada per sfuggire a un
falco: proprio il momento in cui restare insieme conta di più. La
{numref}`fig-regola-metrica-topologica` mette le due regole una accanto
all'altra sullo stesso stormo.

```{figure} ../figures/regola-metrica-e-topologica.svg
:name: fig-regola-metrica-topologica
:alt: "Quattro riquadri, due colonne per due righe, con lo stesso stormo di novanta uccelli disegnato su un piano. In alto lo stormo è fitto, in basso lo stesso stormo diradato fino a occupare una superficie doppia, disegnato con lo stesso metro. La colonna di sinistra applica la regola metrica: un cerchio pieno di raggio fisso attorno all'uccello in ocra al centro, con un segmento verso ogni compagno che ci sta dentro. Da fitto i compagni collegati sono 7, da rado sono 3: il cerchio è lo stesso e dentro ne restano meno della metà. La colonna di destra applica la regola topologica: un cerchio tratteggiato che arriva fino al settimo compagno più vicino. Da fitto i compagni collegati sono 7, da rado sono ancora 7, e sono gli stessi uccelli: a cambiare è solo il raggio del cerchio, che si allarga del 41 per cento."
:width: 88%

Lo stesso stormo, disegnato su un piano per poterlo guardare, prima fitto e poi
diradato fino a occupare una superficie doppia. Il raggio fisso è sempre quello
e dentro restano tre compagni invece di sette; i sette più vicini sono ancora
quei sette, e a cambiare è soltanto quanto sta lontano il settimo, qui del 41
per cento. Nell'aria, dove lo stormo vive davvero, si allarga meno, perché c'è
una dimensione in più su cui distribuirsi.
```

`````{tab} Elementare

Lo stormo si dirada fino a occupare uno spazio doppio, con lo stesso numero di
uccelli. Con la regola a metri lo storno guarda dentro una palla di due metri
che a ogni diradamento si svuota: i sette compagni di prima diventano tre o
quattro, poi meno ancora, finché ciascuno resta solo nell'istante peggiore. Con
la regola dei sette vicini i compagni restano sette, e la sola domanda è quanto
siano finiti lontano.

Meno di quanto verrebbe da dire. Il conto su una palla è scomodo e su una
scatola no, e di quanto vada allargato lo sguardo viene identico, quindi
facciamolo lì. Una scatola larga due metri, per due, per due contiene otto metri
cubi. Per contenerne il doppio, sedici, non serve una scatola larga il doppio:
quella ne conterrebbe sessantaquattro, cioè otto volte tanto, perché allargando
il lato si allargano insieme le tre direzioni. Proviamo allora ad allargare di
poco. Portiamo il lato da due metri a due e mezzo, un quarto in più: due e mezzo
per due e mezzo per due e mezzo fa quindici metri cubi e sei decimi, e i sedici
che cercavamo sono lì a un soffio. Ecco perché per ritrovare i suoi sette
compagni in uno spazio doppio a uno storno basta allargare lo sguardo di un
quarto: lo spazio cresce molto più in fretta della distanza.

Lo stormo resta unito perché la regola non parla di metri: parla di *quanti*, e
quanti restano quanti anche quando il gruppo si allarga.

`````

`````{tab} Superiore

Sia $\rho$ la densità locale (uccelli per unità di volume; nel capitolo questa
lettera farà altri due mestieri, la correlazione fra votanti e l'evaporazione
del feromone, e ognuno è la convenzione del proprio campo). Con una regola
metrica di raggio $r$ il numero di vicini con cui un individuo interagisce è
$n(r) = \tfrac{4}{3}\pi r^{3} \rho$, proporzionale a $\rho$: il grado di
interazione collassa quando lo stormo si dirada. Con una regola topologica si
fissa invece $n_c$ e a variare è il raggio implicito,

$$
r_c \simeq \left(\frac{3\,n_c}{4\pi\rho}\right)^{1/3} \propto \rho^{-1/3},
$$

che dipende dalla densità solo come l'inverso della sua radice cubica:
dimezzare $\rho$ allunga $r_c$ di $2^{1/3} \approx 1{,}26$, il 26%. Il grado
resta costante per costruzione, ed è questa invarianza a sostenere la coesione
sotto grandi variazioni di densità: gli autori la argomentano e la mostrano su
un modello a particelle auto-propulse in due dimensioni, non la dimostrano.
Le due formule valgono in un intorno abbastanza omogeneo e lontano dal bordo
dello stormo, dove il conteggio dei vicini va corretto e senza correzione dà
risultati sbagliati.

Il test empirico usa proprio questa differenza. Su dieci stormi di rarefazione
molto diversa ($r_1$, la distanza media dal primo vicino, va da $0{,}68$ a
$1{,}51$ m) il raggio di interazione $r_c$ cresce con $r_1$ in modo netto
($R^2 = 0{,}78$), mentre il numero di vicini interagenti non mostra alcuna
correlazione con la rarefazione ($n_c^{-1/3}$ contro $r_1$: $R^2 = 0{,}00021$).
Il raggio segue la densità, il grado no: in media
$n_c = 6{,}5 \pm 0{,}9$ (errore standard) {cite}`ballerini2008interaction`. Nel
linguaggio dei grafi, quello dei {doc}`dati a grafo
</GraphNeuralNetwork/dati-a-grafo>`, la regola metrica costruisce un grafo a
raggio fisso e la topologica il grafo diretto degli $n_c$ vicini più prossimi,
in cui ogni nodo sceglie i propri $n_c$ archi uscenti: il grado uscente è
costante per costruzione (quello entrante no, perché la scelta non va
ricambiata: io guardo te senza che tu debba guardare me, che è la natura stessa
dell'interazione fra storni) e il grafo non cambia affatto se tutte le distanze
vengono riscalate.

`````

Da qui la tesi che percorre tutto il capitolo, e che vale ben oltre gli
uccelli: a parità di individui, il comportamento di un gruppo lo decide la
regola di interazione. Nelle simulazioni con cui Ballerini e colleghi
accompagnano le misure gli individui sono gli stessi, e cambia soltanto la
regola con cui ciascuno guarda i vicini: con quella topologica lo stormo
attaccato da un predatore resta quasi sempre intero, con quella metrica si
spezza molto più spesso in più tronconi. La bravura dei singoli conta ancora,
ma non basta a prevedere il gruppo. Ogni sezione che segue riprende questa tesi
su programmi invece che su uccelli. Un agente, qui, è un programma a cui si
affida un compito e che lo porta avanti da sé, decidendo un passo alla volta
che cosa fare; e gli stessi dieci agenti, a seconda di chi può scrivere a chi,
formano sistemi diversi.

## Perché adesso

Il campo non è nuovo. Gli agenti sono un filone dell'intelligenza artificiale
dagli anni Settanta, e l'intelligenza artificiale *distribuita* ha passato
trent'anni su come far cooperare programmi separati: che lingua parlano fra
loro, come si accordano su chi fa che cosa, come decidono quando le loro
risposte non coincidono. Sono le domande dei sistemi multi-agente, e hanno già
trent'anni di risposte: conviene non riscoprirle male.

Quello che è cambiato è il costo di partenza. Un modello di linguaggio è un
programma che, dato un testo, ne scrive il seguito, e lo fa abbastanza bene da
poter ricevere le istruzioni a parole invece che in codice. Definirne dieci
copie costa una riga di configurazione, perché il programma è sempre lo stesso:
a distinguerle è soltanto il foglio di istruzioni che ciascuna si trova davanti
prima di cominciare (il *prompt di sistema*: «tu scrivi codice e non discuti le
scelte altrui», «tu cerchi errori e non ne proponi la correzione»). Dieci fogli
diversi, e la squadra c'è. Il costo vero comincia quando la squadra lavora,
perché ogni messaggio che un agente scrive è una chiamata al modello, e ogni
messaggio che legge si paga anche lui.

Conviene fissare subito un esempio, perché nelle prossime pagine si parlerà a
lungo di quanto costa una squadra e di che forma darle, e il prezzo di una cosa
non dice niente finché non si sa che cosa sia. Prendiamo la richiesta: «apri
questo file di vendite e dimmi quali negozi stanno peggiorando». Il
coordinatore la riceve e tiene le fila. Il programmatore scrive il codice che
apre il file e fa i conti. Il revisore lo legge e dice soltanto una cosa: se è
sicuro eseguirlo, cioè se non cancella niente e non combina danni. A quel punto
tocca di nuovo al coordinatore, perché è l'unico dei tre che può toccare la
macchina vera: gli altri due scrivono e leggono testo, lui esegue. Il risultato
torna al programmatore, che lo interpreta, e la risposta al mittente la
consegna di nuovo il coordinatore.

La squadra viene dall'articolo che presenta AutoGen {cite}`wu2024autogen`, un
framework, cioè una libreria di programmi pronti, con cui conversazioni di
questo genere fra agenti si scrivono in poche righe: là i tre si chiamano
Commander, Writer e Safeguard, e interrogano un modello che ottimizza le
consegne di una catena di fornitura (l'applicazione si chiama OptiGuide). Qui la
domanda è un'altra, i tre mestieri sono quelli, ed è la squadra a cui pensare
ogni volta che si parla di agenti che si passano messaggi. La {doc}`sezione
sulle architetture degli agenti </Agenti/architetture-e-valutazione>` ha già
descritto ruoli come questi uno per uno: il pianificatore, l'esecutore che
svolge il lavoro e il critico che lo controlla. Qui il programmatore è
l'esecutore, il revisore è il critico, e il coordinatore pianifica e in più è
l'unico che fa girare il codice sulla macchina; quello che si studia è che cosa
succede quando i tre sono insieme.

Se definire i partecipanti costa poco, la difficoltà sta altrove: in chi parla
con chi, in chi decide quando le proposte sono in disaccordo, e in una domanda
che con un agente solo non si pone, come ci si accorge che il gruppo nel suo
insieme ha sbagliato.

L'ultima è la meno ovvia. Quando sbaglia un gruppo ogni singolo pezzo sembra a
posto: ciascuno ha fatto il proprio turno, i messaggi sono ben scritti, e il
risultato è sbagliato lo stesso. Due esempi: la richiesta di partenza si
deforma passando di mano in mano, e l'ultimo risponde benissimo a una domanda
diversa da quella iniziale (la *deriva del compito*); oppure qualcuno afferma
con sicurezza una cosa falsa e nessuno la controlla (la *verifica assente o
sbagliata*, fra i modi di fallire più frequenti). Il guasto sta nella
conversazione, non nei singoli turni, e catalogarne le forme è un lavoro
cominciato da poco {cite}`cemri2025why`: la {doc}`sezione sul costo del
coordinamento </SistemiMultiAgente/costo-del-coordinamento>` riprende quel
catalogo, dove i modi sono quattordici e la famiglia più numerosa non è nessuna
di queste due.

## Molti battono uno solo se sbagliano in modo diverso

Prima di progettare squadre, va chiarita l'ipotesi nascosta sotto l'idea stessa
che «più teste ragionino meglio di una». Non è sempre vera, e la condizione che
la rende vera è una sola, precisa, e facilissima da violare quando gli agenti
sono costruiti tutti allo stesso modo.

`````{tab} Elementare

Tre colleghi devono rispondere sì o no a una domanda difficile, e ognuno, da
solo, ci prende sette volte su dieci. Decidendo a maggioranza il gruppo ci
prende quasi otto volte su dieci (78%), e con nove colleghi il 90%: perché il
gruppo sbagli servono almeno due errori insieme, che sono più rari di uno. Da
dove esca esattamente quel 78 lo vedremo elencando i casi uno per uno nella
{doc}`sezione sui protocolli e il consenso <protocolli-e-consenso>` (il 90 esce
allo stesso modo, con molti più
casi da elencare): è un conto da foglio e matita, ma per adesso basta il senso.

Il conto però vale solo se i tre sbagliano in modo *diverso*. Se hanno studiato
sugli stessi appunti sbagliati sbagliano insieme, la maggioranza conferma
l'errore invece di correggerlo e il gruppo ci prende sette volte su dieci come
ciascuno di loro: tre stipendi per il risultato di uno. E se ciascuno ci prende
quattro volte su dieci, votare *peggiora* le cose: tre danno il 35%, nove il
27%. Il voto amplifica la tendenza di fondo, qualunque sia. (Anche questi due
numeri escono dallo stesso elenco di casi, rifatto partendo da quattro volte su
dieci invece che da sette.)

`````

`````{tab} Superiore

È il **teorema della giuria di Condorcet** (1785). Con $n$ votanti indipendenti
che scelgono fra due alternative, ciascuno corretto con la stessa probabilità
$p$, e decisione a maggioranza semplice, la probabilità che il gruppo abbia
ragione è (con $n$ dispari, così che un pareggio non si possa dare)

$$
P_n = \sum_{k=\lfloor n/2 \rfloor + 1}^{n} \binom{n}{k}\, p^{k} (1-p)^{\,n-k},
$$

dove $\binom{n}{k}$ conta i modi in cui $k$ votanti su $n$ possono azzeccare.
Al crescere di $n$, $P_n \to 1$ se $p > 1/2$ e $P_n \to 0$ se $p < 1/2$; con
$p = 0{,}7$ si ha $P_3 = 0{,}784$ e $P_9 = 0{,}901$, con $p = 0{,}4$
$P_3 = 0{,}352$. La velocità la dà la disuguaglianza di Hoeffding: detto $S_n$
il numero di voti corretti, per $p > 1/2$

$$
P_n \;=\; \Pr\!\big(S_n > n/2\big) \;\ge\; 1 - e^{-2n\,(p - 1/2)^2},
$$

cioè l'errore del gruppo decade esponenzialmente in $n$. Il limite è largo
($p = 0{,}7$ e $n = 9$ danno soltanto $P_9 \ge 0{,}513$, contro il $0{,}901$
esatto) ma ha la forma giusta. Le ipotesi che lo reggono sono tre: voti
indipendenti, la stessa $p$ per tutti, e voto sincero, cioè ciascuno vota per
l'alternativa che ritiene giusta. La derivazione, il caso a più alternative e
un modello di errori correlati stanno nella {doc}`sezione sui protocolli e il
consenso <protocolli-e-consenso>`.

L'ipotesi vincolante è l’**indipendenza degli errori**, ed è la più fragile che
ci sia fra agenti che condividono il modello di base, i dati di
pre-addestramento e spesso metà del prompt. Nel limite di correlazione perfetta
$P_n = p$ per ogni $n$: la maggioranza di $n$ agenti vale un agente,
moltiplicandone il costo. È la lezione degli ensemble del capitolo sul
machine learning, dove il guadagno non viene dal numero di modelli ma dalla
loro decorrelazione. Diversità prima di quantità; i modi di comprarla, ciascuno
con il suo prezzo, li elenca la stessa sezione sui protocolli e il consenso.

`````

## Dal singolo agente al gioco: i prerequisiti

I sistemi multi-agente poggiano su due capitoli precedenti, e ne anticipano
uno. Dagli Agenti vengono il ciclo osserva-ragiona-agisci e i ruoli
specializzati: qui diamo per acquisito il singolo agente e studiamo ciò che
nasce quando sono molti. Dal Reinforcement Learning vengono il processo
decisionale di Markov (MDP: si osserva uno stato, si sceglie un'azione, si
riceve una ricompensa e si passa allo stato successivo), la *policy* $\pi$, cioè
la regola con cui l'agente sceglie l'azione in ciascuno stato, e il problema
dell'assegnazione del merito (*credit assignment*): la ricompensa arriva alla
fine di una partita e bisogna stabilire quale mossa se la sia guadagnata. Con
più agenti la domanda si sdoppia, e non chiede più soltanto *quale mossa* ha
prodotto il risultato, ma anche *quale agente*.

Il terzo arriva più avanti: sono le {doc}`GAN </GAN/overview>`, l'esempio più
puro di due parti che si spingono a vicenda a migliorare. Ogni volta che
serviranno diremo per esteso quel che c'è da saperne.

`````{tab} Elementare

Più avanti nel libro incontrerai due reti che si allenano l'una contro l'altra:
una fabbrica immagini false, l'altra cerca di smascherarle. Si chiamano GAN, e
di loro qui basta sapere come finisce la partita. Addestrare un programma, di
solito, somiglia a cercare il punto più basso di una valle nella nebbia: si
scende, e quando non si scende più si è arrivati. Qui no, perché ogni passo
avanti di uno rende più difficile il mestiere dell'altro, e una valle sola non
c'è. Quello che si può sperare è un **equilibrio**, cioè il momento in cui a
nessuno dei due conviene più cambiare mossa da solo: se l'altro resta dov'è,
cambiare non gli fa guadagnare niente. Non è per forza un buon risultato. Può
essere quello in cui tutti stanno peggio di come starebbero fidandosi l'uno
dell'altro, e nessuno può uscirne da solo.

Un sistema multi-agente allarga quella struttura: i giocatori possono essere
dieci, e non sono per forza nemici. E imparare diventa più difficile, perché un
agente solo studia in un mondo fermo mentre dieci agenti studiano in un mondo
che si muove: il «mondo» di ciascuno contiene gli altri nove, che stanno
imparando anche loro. È come allenarsi per una gara in cui anche gli avversari
si allenano: quello che bastava a vincere il mese scorso, il mese prossimo non
basta più.

`````

`````{tab} Superiore

Il quadro formale generalizza l'MDP a un **gioco stocastico** (qui e altrove
$\pi$ è la policy, e non ha niente a che vedere con il $3{,}14$ della
circonferenza), la tupla
$(N, \mathcal{S}, \{\mathcal{A}^i\}, P, \{r^i\}, \gamma)$: $N$ agenti, uno
spazio di stati $\mathcal{S}$, spazi di azione
$\mathcal{A}^1, \dots, \mathcal{A}^N$, una transizione $P(s' \mid s, a)$ che
dipende dall'azione **congiunta** $a = (a^1, \dots, a^N)$, una ricompensa $r^i$
per ciascun agente e un fattore di sconto $\gamma \in [0, 1)$. Se ogni agente
osserva soltanto una parte $o^i$ dello stato, e la ricompensa è comune, il
modello diventa il Dec-POMDP (processo decisionale di Markov decentralizzato e
parzialmente osservabile) {cite}`oliehoek2016concise`, quello della sezione su
{doc}`imparare insieme <imparare-insieme>`, ed è il caso degli agenti
linguistici, ciascuno dei quali vede soltanto il proprio contesto. Se $r^i = r$
per ogni $i$ il gioco è cooperativo; se $N = 2$ e $r^1 + r^2 = 0$ si ricade nel
caso a somma zero, che è la forma minimax della GAN (con la *loss* non-saturante
che si usa in pratica la somma non è più zero, e la {doc}`sezione su come
funziona una GAN </GAN/come-funziona>` spiega perché), dove l'obiettivo non è un
minimo di $\mathcal{L}$ ma un **equilibrio di Nash**: un profilo
$(\pi^1, \dots, \pi^N)$ tale che per ogni agente $i$, ogni policy alternativa
$\tilde{\pi}^i$ e ogni stato $s$ valga
$V^i_{(\pi^i,\pi^{-i})}(s) \ge V^i_{(\tilde{\pi}^i,\pi^{-i})}(s)$, dove $V^i$ è
il ritorno atteso di $i$, scontato con $\gamma$, e $\pi^{-i}$ le policy degli
altri. Con stati e azioni finiti e ritorno scontato un equilibrio in policy
stazionarie esiste (per la somma zero {cite}`shapley1953stochastic`, in generale
{cite}`fink1964equilibrium`), ma le policy sono in generale stocastiche: in
sasso, carta e forbici l'unico equilibrio gioca ogni mossa un terzo delle volte,
mentre con un agente solo una policy ottima deterministica c'è sempre. Di
solito, poi, l'equilibrio non è unico e non è per forza un buon esito. Nel
**dilemma del prigioniero** (ricompense $(3,3)$ se cooperano entrambi, $(5,0)$ o
$(0,5)$ se defeziona uno solo, $(1,1)$ se defezionano entrambi) defezionare è la
miglior risposta a qualunque mossa dell'altro: l'unico equilibrio è $(1,1)$, ed
è anche l'unico esito non **Pareto-ottimo**, cioè migliorabile per qualcuno
senza peggiorarlo per nessuno. Nel caso cooperativo il profilo ottimo è un
equilibrio, ma accanto a equilibri peggiori, e scegliere fra loro è un problema
a sé. Ne segue che il caso multi-agente non è quello singolo ripetuto $N$ volte:
per l'agente $i$ l'ambiente comprende le policy $\pi^{-i}$ degli altri, che
cambiano durante l'addestramento, quindi il processo che $i$ osserva non è
stazionario e le garanzie di convergenza del Q-learning, che presuppongono un
MDP fisso, decadono. Lo affronta la stessa sezione su imparare insieme.

`````

## Tre domande

Il capitolo risponde a tre domande, in quest'ordine. Conviene davvero più di
un agente, e a che prezzo? Come si organizzano, cioè chi parla con chi,
con quali messaggi e con quale regola di decisione? E infine: possono
imparare a coordinarsi invece di essere programmati per farlo, come gli
storni, a cui la regola dei sei o sette vicini non l'ha insegnata nessuno?

- La prima domanda è quella del costo del coordinamento: quando più agenti
  battono un singolo agente ben progettato, con i conti in mano (quante volte si
  interroga il modello, quanto testo gli si fa rileggere, quanti giri di
  conversazione servono).
- La seconda ha due metà. Chi parla con chi: le forme che può prendere lo
  schema di chi scrive a chi (un coordinatore con i suoi lavoratori, la catena,
  la gerarchia, la lavagna condivisa, il mercato) e che cosa ciascuna fa al
  numero di passaggi, al lavoro del più carico e a quello che resta in
  piedi se si ferma. Poi protocolli e consenso: messaggi che dichiarano che cosa
  fanno (una richiesta, un impegno, un rifiuto) e regole per decidere (il voto
  di maggioranza, il dibattito), fino al caso duro in cui un partecipante si
  guasta o mente.
- La terza è imparare insieme: l'apprendimento per rinforzo multi-agente, la
  ricetta di allenarsi guardando tutto e poi giocare guardando solo il proprio
  pezzo (l'addestramento centralizzato con esecuzione decentralizzata) e il
  *self-play* incontrato dietro AlphaGo.
- In fondo, sciami e simulazioni: regole locali elementari che risolvono
  problemi globali (colonie di formiche, sciami di particelle) e le società
  simulate, dove l'oggetto di studio è il collettivo stesso.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Gli stormi di storni sopra Termini, ricostruiti in tre dimensioni dal progetto
  StarFlag {cite}`ballerini2008interaction`, seguono una regola
  topologica: ogni uccello tiene d'occhio un numero fisso di vicini, i sei o
  sette più prossimi, e non tutti quelli che gli stanno entro due metri.
- È questo che tiene insieme lo stormo quando si dirada. Con una regola a metri
  i compagni dentro il raggio si diradano insieme allo stormo; contando
  i vicini invece che misurandoli, sette restano sette, e sono gli stessi sette,
  che si trovano soltanto un quarto più in là. Le regole a metri, come quella
  dei boids {cite}`reynolds1987flocks`, tengono molto meno.
- La tesi del capitolo: a parità di individui, il comportamento del gruppo lo
  decide la regola di interazione. Stessi individui, regola diversa,
  collettivo diverso.
- Definire dieci agenti costa poco: stesso modello, dieci fogli di istruzioni
  diversi {cite}`wu2024autogen`. Il conto arriva quando lavorano, perché ogni
  messaggio scritto e ogni messaggio letto si paga. La squadra da tenere in
  mente per tutto il capitolo ne ha tre: chi tiene le fila ed esegue, chi
  scrive il codice, chi controlla che si possa eseguire. Il difficile viene
  dopo: chi parla con chi, chi decide quando le risposte non coincidono, e come
  ci si accorge che a sbagliare è il *gruppo* e non un turno
  {cite}`cemri2025why`.
- «Più teste» aiuta solo se sbagliano in modo diverso. Tre persone che ci
  prendono sette volte su dieci, votando, ci prendono quasi otto volte su dieci;
  ma se hanno studiato sugli stessi appunti sbagliati sbagliano insieme, e tre
  agenti valgono quanto uno, al costo di tre.
- Il caso con molti agenti non è quello singolo ripetuto tante volte: per
  ciascuno il mondo contiene gli altri, che nel frattempo cambiano. Il traguardo
  non è più il fondo di una valle ma un equilibrio: la situazione in cui a
  nessuno conviene più muoversi da solo, e che non è per forza un buon
  risultato per tutti.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Gli stormi di storni sopra Termini, ricostruiti in 3D dal progetto
  StarFlag {cite}`ballerini2008interaction`, seguono una regola
  topologica: ogni uccello guarda un numero fisso di vicini più prossimi
  ($n_c = 6{,}5 \pm 0{,}9$), non tutti quelli entro un raggio in metri.
- Così lo stormo resta unito anche quando si dirada, perché il grado di
  interazione non dipende dalla densità ($n_c$ è fissato, e a seguire la densità
  è semmai il raggio implicito, $r_c \propto \rho^{-1/3}$); una regola
  metrica, come quella dei boids {cite}`reynolds1987flocks`, tiene molto
  meno, e lo mostrano le simulazioni dello stesso lavoro, fatte su un modello a
  particelle auto-propulse in due dimensioni, dove uno stormo a regola metrica
  si spezza in più tronconi molto più spesso di uno topologico.
- La tesi del capitolo: a individui fissati, il comportamento del gruppo lo
  decide la regola di interazione. Stessi individui, regola diversa,
  collettivo diverso.
- Istanziare dieci agenti costa una riga di codice {cite}`wu2024autogen`, farli
  lavorare no: ogni messaggio è una chiamata al modello. Il difficile è chi
  parla con chi, chi decide, e accorgersi che ha sbagliato il *gruppo* e non un
  turno {cite}`cemri2025why`.
- «Più teste» aiuta solo se sbagliano in modo indipendente (Condorcet: con
  $p = 0{,}7$, tre votanti danno $0{,}784$); con errori perfettamente correlati
  $n$ agenti valgono quanto uno, al costo di $n$.
- Il caso multi-agente non è quello singolo ripetuto: per ciascun agente
  l'ambiente contiene gli altri, che cambiano, e quindi non è stazionario;
  l'obiettivo diventa un equilibrio di Nash, non un minimo di $\mathcal{L}$.
```

`````
