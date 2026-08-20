# Generative Adversarial Networks

L'idea di cui parla questo capitolo è nata, dice la leggenda (perché ormai è
una leggenda), in un bar di Montréal, i *Trois Brasseurs*, nel 2014. Ian
Goodfellow, dottorando nel laboratorio di Yoshua Bengio, festeggia con alcuni
colleghi. Uno di loro racconta a che cosa sta lavorando: vuole una rete che
generi fotografie realistiche, e il suo metodo è misurare a una a una le
regolarità delle immagini vere (quanto spesso due pixel vicini hanno lo stesso
colore, quali sfumature si accompagnano a quali) per poi costruire un'immagine
nuova che le rispetti tutte. Il lavoro non finisce mai, perché le regolarità
sono troppe, e il collega si è arenato lì. Goodfellow obietta che così non
funzionerà, ma tornando a casa gli viene un'idea diversa: e se invece di una
rete sola ne mettessi *due*, una a fabbricare immagini e una a smascherarle, e
le facessi combattere? Quella notte scrive il codice. Funziona quasi al primo
colpo. Ne esce l'articolo *Generative Adversarial Nets*
{cite}`goodfellow2014generative`, uno dei più citati del decennio:
*generative adversarial networks*, alla lettera reti generative avversarie,
cioè reti che fabbricano qualcosa e che imparano a farlo sfidandosi.

## Generare, non classificare

Il libro ha già incontrato reti che *producono* qualcosa, e parecchie: un
modello linguistico scrive la parola dopo, un sintetizzatore legge un testo ad
alta voce, un vocoder trasforma la descrizione di un suono nell'onda sonora
vera e propria. Tutte quelle, però, mentre imparavano avevano sotto gli occhi
la risposta giusta: la parola che veniva davvero dopo, l'onda che quella frase
aveva davvero. Chi deve disegnare un gatto mai esistito non ce l'ha e non può
averla: non c'è nessun originale da mettere accanto al risultato per vedere,
punto per punto, di quanto ci si è allontanati. La domanda di questo capitolo è
proprio questa: come si insegna a una rete a **fabbricare dati nuovi e
plausibili** quando non c'è niente con cui confrontarli.

Una risposta il libro l'ha appena data, ed è quella del {doc}`capitolo sui modelli
latenti </ModelliLatenti/overview>`: si scrive una formula per la probabilità di un dato, si rinuncia a
calcolarla esattamente e si ottimizza quello che si riesce a calcolare. Questo
capitolo prende la strada opposta, e conviene tenerlo a mente perché è la
scelta che spiega tutto il resto: qui la probabilità non si scrive affatto, e
al suo posto si mette qualcuno che guarda il risultato e dice se ci crede.

`````{tab} Elementare

Un classificatore guarda la foto di un gatto e dice "gatto". Un modello
generativo, partendo da un pugno di numeri casuali, *disegna* la foto di un
gatto che non è mai esistito: un gatto che nessuna macchina fotografica ha mai
ripreso. Non ha imparato a mettere un'etichetta: ha imparato la "ricetta" di
che aspetto ha una foto di gatto, e può cucinarne di nuove all'infinito.

I numeri casuali sono la sua materia prima. Sono una manciata (un centinaio, di
solito), e glieli diamo noi tirandoli a sorte come i numeri di una tombola. Che
debbano essere **diversi** ogni volta si capisce: ad addestramento finito la
rete non cambia più, resta quella, e quei numeri sono l'unica cosa che la
distingue da una richiesta all'altra. È da lì che viene la varietà: numeri
diversi in ingresso, gatti diversi in uscita.

Che debbano essere **a sorte**, e non scelti da noi in fila (1, 2, 3…), è una
faccenda diversa. La ragione è che durante l'addestramento la rete si allena su
numeri tirati a sorte, e impara a cavarsela in tutte le zone da cui possono
uscire. Se poi, a cose fatte, le dessimo numeri scelti da noi, la porteremmo
sempre nelle stesse zone e ci ritroveremmo sempre gli stessi gatti: il sorteggio
è ciò che tiene la promessa fatta in addestramento.

`````

`````{tab} Superiore

Un modello discriminativo apprende la probabilità condizionata $p(y \mid \mathbf{x})$
di un'etichetta $y$ dato l'input $\mathbf{x}$. Un modello **generativo** apprende,
esplicitamente o implicitamente, la distribuzione dei dati $p_{\text{dati}}(\mathbf{x})$,
così da poterne campionare esempi nuovi. Una GAN la apprende in modo
*implicito*: non stima una densità in forma chiusa, ma costruisce un
**campionatore** $G(\mathbf{z})$ che trasforma un rumore semplice $\mathbf{z} \sim p_z$
(tipicamente gaussiano) in campioni che l'addestramento spinge a diventare
indistinguibili da quelli reali; la distribuzione da cui questi campioni
provengono si indica con $p_G$.

`````

## Due reti in competizione

L'intuizione di Goodfellow è tutta in una metafora da tenere a mente per
l'intero capitolo.

`````{tab} Elementare

Un **falsario** dipinge quadri contraffatti; un **esperto
d'arte** deve dire quali sono autentici e quali falsi. All'inizio il
falsario è maldestro e l'esperto lo smaschera senza sforzo. Ma ogni volta che
viene scoperto, il falsario impara qualcosa e migliora; e l'esperto, di fronte
a falsi sempre più raffinati, affina il proprio occhio. È una corsa agli
armamenti: i due si perfezionano a vicenda. Alla fine i falsi sono così buoni
che nemmeno l'esperto sa più distinguerli. Il **generatore** è il falsario, il
**discriminatore** è l'esperto.

Qui c'è però una domanda da fare subito, perché è il cuore di tutto il
capitolo: il falsario impara *che cosa*? Se l'esperto si limitasse a dire
"falso", il falsario saprebbe di aver sbagliato ma non saprebbe **dove**, ed è
la stessa differenza che passa fra un professore che scrive "no" in fondo al
compito e uno che sottolinea le righe da rifare. L'esperto di questa storia
appartiene al secondo tipo: non dice "falso", dice "falso, e soprattutto per
via di *questo* qui", indicando col dito, punto per punto del quadro, da che
parte tirare. Vedremo nella prossima sezione come fa, e perché per riuscirci
debba essere una rete e non una persona.

`````

`````{tab} Superiore

Le due reti hanno ruoli antagonisti. Il **generatore** $G$ mappa un vettore di
rumore $\mathbf{z}$ in un campione sintetico $G(\mathbf{z})$. Il **discriminatore** $D$ riceve
un'immagine e restituisce $D(\cdot) \in [0,1]$, la probabilità stimata che sia
reale. Si addestrano *insieme* ma con obiettivi opposti: $D$ vuole assegnare
$1$ ai dati veri e $0$ ai falsi; $G$ vuole che $D$ assegni $1$ ai propri falsi.
Il segnale che smaschera il falso, propagato all'indietro attraverso $D$, è lo
stesso che insegna a $G$ come migliorarlo: è questa condivisione a saldare
l'addestramento delle due reti in un unico ciclo di feedback.

Conviene dire subito che cosa sia, quel segnale, perché è il perno
dell'intero capitolo e non è il verdetto. Ciò che risale da $D$ verso $G$ è il
gradiente della loss del generatore rispetto al **dato generato**
$\tilde{\mathbf{x}} = G(\mathbf{z})$, cioè $\partial \mathcal{L}_G /
\partial \tilde{\mathbf{x}}$: non un numero ma un vettore, con una componente per ogni
numero del dato, che dice in che verso spostare ciascuna di quelle componenti
perché il verdetto cambi. È una direzione, non un voto, e la sua esistenza
richiede che $D$ sia **derivabile rispetto al proprio ingresso**: la sezione
seguente riprende il punto con la regola della catena.

`````

Lo schema complessivo del gioco è quello di {numref}`fig-gan-gioco`.

```{figure} ../figures/gan-gioco-avversario.svg
:name: fig-gan-gioco
:alt: Del rumore casuale entra nel generatore che produce un'immagine falsa; questa e un'immagine reale dal dataset entrano nel discriminatore che emette un verdetto vero o falso; una freccia di feedback in basso torna indietro e addestra sia il generatore sia il discriminatore.
:width: 90%

Il gioco avversario. Il generatore trasforma numeri casuali in un'immagine
falsa; il discriminatore riceve immagini di tutti e due i tipi, una per volta,
e su ciascuna emette un verdetto. Da come è arrivato al verdetto si ricava una
correzione, che torna indietro a tutte e due le reti: all'esperto serve per
sbagliare di meno, al falsario per farlo sbagliare di più. La correzione non è
il verdetto ed è molto più ricca di quello, ma per capire perché bisogna
arrivare alla sezione seguente.
```

## Il gioco a somma zero

Falsario ed esperto giocano l'uno *contro* l'altro: ciò che è un guadagno per
il primo è una perdita per il secondo. In teoria dei giochi si chiama gioco a
somma zero, e si tiene con un punteggio solo: c'è un tabellone unico, uno dei
due lo vuole più alto possibile e l'altro più basso possibile, e nessuno dei
due ha un tabellone suo su cui segnare punti per conto proprio.

`````{tab} Elementare

È un tiro alla fune: il discriminatore tira da una parte (vuole avere
sempre ragione), il generatore tira dall'altra (vuole ingannarlo), e la corda è
una sola, cioè il terreno che guadagna uno lo perde l'altro. L'immagine serve
per questo, e conviene fermarla qui: in un tiro alla fune, quando nessuno dei
due si sposta più vuol dire che non sta succedendo niente, mentre qui il
pareggio è il traguardo.

Perché il pareggio arriva quando il falsario è diventato bravissimo. Se i suoi
quadri sono indistinguibili da quelli veri, l'esperto non ha più niente su cui
appoggiarsi: qualunque cosa gli passi davanti, può solo tirare a indovinare,
come a testa o croce, e indovina una volta su due. Quel 50% non è un esperto
che si è arreso: è il massimo che chiunque possa ottenere quando non c'è più
niente da vedere.

Ma perché dovrebbe finire *così*, e non con l'esperto che vince sempre e il
falsario che resta scarso per sempre? La risposta sta in un dettaglio
dell'allenamento dell'esperto, e la sezione seguente ci torna sopra per esteso:
ogni volta che tocca a lui, l'esperto non guarda soltanto dei falsi, guarda
anche dei quadri autentici, ed è su quelli che viene corretto. È lui il punto
in cui la realtà entra nel gioco, e finché quel punto c'è, l'unico modo che il
falsario ha di ingannarlo stabilmente è somigliare davvero ai quadri veri.

Resta il caso opposto, quello in cui l'esperto prende troppo vantaggio e il
falsario non riesce più a stargli dietro. È un problema concreto, e anche
quello lo affronta la sezione seguente.

`````

`````{tab} Superiore

Goodfellow formula l'addestramento come un problema **minimax** su una funzione
valore $V(D,G)$:

$$
\min_{G}\ \max_{D}\ V(D,G) =
\mathbb{E}_{\mathbf{x}\sim p_{\text{dati}}}\!\big[\log D(\mathbf{x})\big]
+ \mathbb{E}_{\mathbf{z}\sim p_z}\!\big[\log\big(1 - D(G(\mathbf{z}))\big)\big].
$$

Qui $\mathbf{x}\sim p_{\text{dati}}$ è un campione reale, $\mathbf{z}\sim p_z$ è il rumore in
ingresso a $G$, $D(\mathbf{x})$ è la probabilità stimata che l'input sia autentico. Il
discriminatore *massimizza* $V$ (assegna probabilità alta ai veri, bassa ai
falsi $G(\mathbf{z})$); il generatore *minimizza* il secondo termine, cioè spinge
$D(G(\mathbf{z}))$ verso $1$. All'ottimo teorico si ha
$p_G=p_{\text{dati}}$ e $D(\mathbf{x})=\tfrac{1}{2}$ sul supporto dei dati: l'esperto
non sa più decidere. In pratica l'equilibrio è delicato: instabilità
dell'addestramento e *mode collapse* (il generatore che produce sempre la
stessa immagine vincente) sono i due grattacapi ricorrenti, che affronteremo
più avanti.

`````

## Perché ce ne importa

Le GAN hanno spostato il confine di ciò che una macchina può *fabbricare*. Da
questa idea nascono i **volti fotorealistici** di persone inesistenti: la
famiglia StyleGAN di NVIDIA {cite}`karras2019style` alimenta siti come *This
Person Does Not Exist*, dove ogni ricarica mostra un volto sintetico che a un
primo sguardo non si distingue da una fotografia. Da qui arrivano anche i
**deepfake** (volti sostituiti nei video) con tutto il loro carico di rischi
per disinformazione e consenso. E arriva l’**arte generata**, con un ritratto prodotto da una GAN battuto
all'asta da Christie's nel 2018: l'episodio, e la questione di chi ne sia
l'autore, sono raccontati in chiusura di capitolo.

Uno strumento potente e ambivalente, insomma: capace di fabbricare dataset
(le raccolte di esempi su cui si addestrano le altre reti),
restaurare immagini e perfino di proporre molecole nuove (anche una molecola,
per una macchina, è un disegno con le sue regolarità), ma anche di fabbricare
falsi convincenti. Ragione in più per capirne bene il funzionamento.

## Il duello, e quello che ne è nato

Dall'intuizione passiamo alla pratica. La sezione seguente smonta il
meccanismo: che cosa entra e che cosa esce da ciascuna delle due reti, come si
collegano, che cosa esattamente l'una restituisce all'altra, e come da
quell'unico punteggio ciascuna ricavi la propria **loss**, cioè il conto del
proprio errore. Poi il ciclo di addestramento a turni, scritto riga per riga in
PyTorch, con le sue insidie: il duello che non si stabilizza, e il *mode
collapse*, cioè il falsario che scopre un solo quadro capace di ingannare
l'esperto e si limita a rifare sempre quello. Da lì una domanda tutt'altro che
ovvia, come si faccia a **misurare** se una GAN sta funzionando, visto che la
sua loss non lo dice.

L'ultima sezione racconta le **varianti** che hanno fatto la storia, dalla
DCGAN alle GAN condizionali fino a StyleGAN, e chiude sul passaggio di
testimone ai **modelli di diffusione**. Sono un altro modo di far disegnare le
macchine: invece di mettere due reti l'una contro l'altra, insegnano a una rete
sola a partire da una macchia di puntini casuali e a ripulirla un poco alla
volta finché non ne esce un'immagine. È la famiglia che dal 2021 ha tolto alle
GAN il primato, e ha un capitolo tutto suo.
