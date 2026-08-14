# Generative Adversarial Networks

La leggenda (perché ormai è una leggenda) la colloca in un bar di Montréal, i
*Trois Brasseurs*, nel 2014. Ian Goodfellow, dottorando nel laboratorio di
Yoshua Bengio, festeggia con alcuni colleghi. Qualcuno racconta di stare
provando a costruire una rete che generi fotografie realistiche misurando una
per una le regolarità delle immagini vere (quanto spesso due pixel vicini
hanno lo stesso colore, quali sfumature si accompagnano a quali) per poi
rimetterle insieme, e si arena. Goodfellow obietta che così non funzionerà,
ma tornando a casa gli viene un'idea diversa: e se invece di una rete sola ne
mettessi *due*, una a fabbricare immagini e una a smascherarle, e le facessi
combattere? Quella notte scrive il codice. Funziona quasi al primo colpo. Ne
esce *Generative Adversarial Nets* {cite}`goodfellow2014generative`, uno dei
paper più citati del decennio: *generative adversarial networks*, alla lettera
reti generative avversarie, cioè reti che fabbricano qualcosa e che imparano a
farlo sfidandosi.

## Generare, non classificare

Il libro ha già incontrato reti che *producono* qualcosa, e parecchie: un
modello linguistico scrive la parola dopo, un vocoder ricostruisce l'onda
sonora di una frase, un sintetizzatore legge un testo ad alta voce. Tutte
quelle, però, mentre imparavano avevano sotto gli occhi la risposta giusta: la
parola che veniva davvero dopo, l'onda che quella frase aveva davvero. Chi deve
disegnare un gatto mai esistito non ce l'ha e non può averla, perché non esiste
un originale da cui misurare la distanza. La domanda di questo capitolo è
proprio questa: come si insegna a una rete a **fabbricare dati nuovi e
plausibili** quando non c'è niente con cui confrontarli.

`````{tab} Elementare

Un classificatore guarda la foto di un gatto e dice "gatto". Un modello
generativo, partendo da un pugno di numeri casuali, *disegna* la foto di un
gatto che non è mai esistito: un gatto che nessuna macchina fotografica ha mai
ripreso. Non ha imparato a mettere un'etichetta: ha imparato la "ricetta" di
che aspetto ha una foto di gatto, e può cucinarne di nuove all'infinito.

I numeri casuali sono la sua materia prima, e sono casuali per una ragione
precisa: sono l'unica cosa che cambia da un gatto all'altro. Glieli diamo noi,
tirandoli a sorte come i numeri di una tombola, ed è da lì che viene la
varietà: numeri diversi in ingresso, gatti diversi in uscita.

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

L'intuizione di Goodfellow è tutta in una metafora che vale la pena tenere a
mente per l'intero capitolo.

`````{tab} Elementare

Immagina un **falsario** che dipinge quadri contraffatti e un **esperto
d'arte** che deve dire quali sono autentici e quali falsi. All'inizio il
falsario è maldestro e l'esperto lo smaschera senza sforzo. Ma ogni volta che
viene scoperto, il falsario impara qualcosa e migliora; e l'esperto, di fronte
a falsi sempre più raffinati, affina il proprio occhio. È una corsa agli
armamenti: i due si perfezionano a vicenda. Alla fine i falsi sono così buoni
che nemmeno l'esperto sa più distinguerli. Il **Generatore** è il falsario, il
**Discriminatore** è l'esperto.

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

Le due reti hanno ruoli antagonisti. Il **Generatore** $G$ mappa un vettore di
rumore $\mathbf{z}$ in un campione sintetico $G(\mathbf{z})$. Il **Discriminatore** $D$ riceve
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
:alt: Del rumore casuale entra nel Generatore che produce un'immagine falsa; questa e un'immagine reale dal dataset entrano nel Discriminatore che emette un verdetto vero o falso; una freccia di feedback in basso torna indietro e addestra sia il Generatore sia il Discriminatore.
:width: 90%

Il gioco avversario. Il Generatore trasforma numeri casuali in un'immagine
falsa; il Discriminatore confronta falsi e immagini reali ed emette un
verdetto. Dal verdetto si ricava una correzione, che torna indietro a tutte e
due le reti: la stessa correzione, letta al contrario dall'una e dall'altra,
perché quello che per l'esperto è un errore da ridurre è per il falsario un
risultato da cercare.
```

## Il gioco a somma zero

Falsario ed esperto giocano l'uno *contro* l'altro: ciò che è un guadagno per
il primo è una perdita per il secondo. In teoria dei giochi si chiama gioco a
somma zero, e si tiene con un punteggio solo: uno dei due lo vuole più alto
possibile, l'altro più basso possibile, e non c'è un terzo tabellone.

`````{tab} Elementare

Pensa a un tiro alla fune. Il Discriminatore tira da una parte (vuole avere
sempre ragione), il Generatore tira dall'altra (vuole ingannarlo). Non esiste
una vittoria definitiva: se il gioco è equilibrato i due finiscono per
bilanciarsi, il Generatore produce falsi perfetti e l'esperto è ridotto a
tirare a indovinare, "testa o croce", con una probabilità del 50%.

Ma perché dovrebbe finire *così*, e non con l'esperto che vince sempre e il
falsario che resta scarso per sempre? La condizione è una sola, e conviene
tenerla a mente da subito: a ogni turno l'esperto rivede anche dei quadri
autentici, e su quelli viene corretto. È lui l'ancora della storia. Senza
quella parte del suo allenamento i due potrebbero mettersi d'accordo su una
schifezza, come due che non hanno mai visto un gatto e passano la vita a
disegnarsi macchie trovandole bellissime; con quella, l'unico modo che il
falsario ha di ingannarlo stabilmente è somigliare davvero ai quadri veri.
Resta il caso in cui l'esperto prende troppo vantaggio e il falsario non
riesce più a stargli dietro: è un problema concreto, e la sezione seguente lo
affronta.

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
Discriminatore *massimizza* $V$ (assegna probabilità alta ai veri, bassa ai
falsi $G(\mathbf{z})$); il Generatore *minimizza* il secondo termine, cioè spinge
$D(G(\mathbf{z}))$ verso $1$. All'ottimo teorico si ha
$p_G=p_{\text{dati}}$ e $D(\mathbf{x})=\tfrac{1}{2}$ sul supporto dei dati: l'esperto
non sa più decidere. In pratica l'equilibrio è delicato: instabilità
dell'addestramento e *mode collapse* (il Generatore che produce sempre la
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
per disinformazione e consenso, un tema su cui questo libro sceglie l'onestà
più che l'entusiasmo. E
arriva l'**arte generata**, con un ritratto prodotto da una GAN battuto
all'asta da Christie's nel 2018: l'episodio, e la questione di chi ne sia
l'autore, sono raccontati in chiusura di capitolo.

Uno strumento potente e ambivalente, insomma: capace di creare dataset,
restaurare immagini e progettare molecole, ma anche di fabbricare falsi
convincenti. Ragione in più per capirne bene il funzionamento.

## Come è organizzato il capitolo

Dall'intuizione passiamo alla pratica. Nelle prossime sezioni vedremo
**l'architettura** del Generatore e del Discriminatore e come collegarli; il
punteggio del gioco, e come da quel punteggio ciascuna rete ricavi la propria
**loss**, cioè il conto del proprio errore; il ciclo di addestramento
alternato, scritto riga per riga in PyTorch, con le sue insidie (instabilità,
*mode collapse*) e i trucchi per domarle; il problema, tutt'altro che ovvio, di
**misurare** se una GAN sta funzionando, visto che la sua loss non lo dice; e
le **varianti** che
hanno fatto la storia, DCGAN, le GAN condizionali, fino a StyleGAN, per
chiudere sul passaggio di testimone ai **modelli di diffusione**: un altro modo
di far disegnare le macchine, che invece di far competere due reti insegna a
una sola rete a ricavare l'immagine da una macchia di puro rumore,
ripulendola un poco alla volta. È la famiglia che dal 2021 ha tolto alle GAN il
primato, e ha un capitolo tutto suo.
