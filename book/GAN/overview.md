# Generative Adversarial Networks

La leggenda (perché ormai è una leggenda) la colloca in un bar di Montréal, i
*Trois Brasseurs*, nel 2014. Ian Goodfellow, dottorando nel laboratorio di
Yoshua Bengio, festeggia con alcuni colleghi. Qualcuno racconta di stare
provando a costruire una rete che generi fotografie realistiche stimandone
tutte le statistiche, e si arena. Goodfellow obietta che così non funzionerà,
ma tornando a casa gli viene un'idea diversa: e se invece di una rete sola ne
mettessi *due*, una a fabbricare immagini e una a smascherarle, e le facessi
combattere? Quella notte scrive il codice. Funziona quasi al primo colpo. Ne
esce *Generative Adversarial Nets* (Goodfellow et al., 2014), uno dei paper
più citati del decennio.

## Generare, non classificare

Fin qui, in questo libro, i modelli hanno soprattutto *risposto a domande su
dati esistenti*: questa email è spam? questa cifra è un 7? Un modello
generativo fa il movimento opposto: **produce dati nuovi e plausibili** che
non esistevano prima.

`````{tab} Elementare

Un classificatore guarda la foto di un gatto e dice "gatto". Un modello
generativo, partendo da un pugno di numeri casuali, *disegna* la foto di un
gatto che non è mai esistito: un gatto che nessuna macchina fotografica ha mai
ripreso. Non ha imparato a mettere un'etichetta: ha imparato la "ricetta" di
che aspetto ha una foto di gatto, e può cucinarne di nuove all'infinito.

`````

`````{tab} Superiore

Un modello discriminativo apprende la probabilità condizionata $p(y \mid X)$
di un'etichetta $y$ dato l'input $X$. Un modello **generativo** apprende,
esplicitamente o implicitamente, la distribuzione dei dati $p_{\text{data}}(X)$,
così da poterne campionare nuovi esempi $X \sim p_{\text{model}}$. Una GAN la
apprende in modo *implicito*: non stima una densità in forma chiusa, ma
costruisce un **campionatore** $G(z)$ che trasforma un rumore semplice
$z \sim p_z$ (tipicamente gaussiano) in campioni indistinguibili da quelli
reali.

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

`````

`````{tab} Superiore

Le due reti hanno ruoli antagonisti. Il **Generatore** $G$ mappa un vettore di
rumore $z$ in un campione sintetico $G(z)$. Il **Discriminatore** $D$ riceve
un'immagine e restituisce $D(\cdot) \in [0,1]$, la probabilità stimata che sia
reale. Si addestrano *insieme* ma con obiettivi opposti: $D$ vuole assegnare
$1$ ai dati veri e $0$ ai falsi; $G$ vuole che $D$ assegni $1$ ai propri falsi.
Il segnale che smaschera il falso, propagato all'indietro attraverso $D$, è lo
stesso che insegna a $G$ come migliorarlo: è questa condivisione a saldare
l'addestramento delle due reti in un unico ciclo di feedback.

`````

Lo schema complessivo del gioco è quello di {numref}`fig-gan-gioco`.

```{figure} ../figures/gan-gioco-avversario.svg
:name: fig-gan-gioco
:alt: Del rumore casuale entra nel Generatore che produce un'immagine falsa; questa e un'immagine reale dal dataset entrano nel Discriminatore che emette un verdetto vero o falso; una freccia di feedback in basso torna indietro e addestra sia il Generatore sia il Discriminatore.
:width: 90%

Il gioco avversario. Il Generatore trasforma rumore casuale in un'immagine
falsa; il Discriminatore confronta falsi e immagini reali ed emette un
verdetto; quel verdetto genera il gradiente che addestra entrambe le reti.
```

## Il gioco a somma zero

Falsario ed esperto giocano l'uno *contro* l'altro: ciò che è un guadagno per
il primo è una perdita per il secondo. In teoria dei giochi si chiama gioco a
somma zero, e si scrive con un unico obiettivo che uno vuole minimizzare e
l'altro massimizzare.

`````{tab} Elementare

Pensa a un tiro alla fune. Il Discriminatore tira da una parte (vuole avere
sempre ragione), il Generatore tira dall'altra (vuole ingannarlo). Non esiste
una vittoria definitiva: quando il gioco è "giusto" i due si equilibrano, il
Generatore produce falsi perfetti e l'esperto è ridotto a tirare a indovinare,
"testa o croce", con una probabilità del 50%.

`````

`````{tab} Superiore

Goodfellow formula l'addestramento come un problema **minimax** su una funzione
valore $V(D,G)$:

$$
\min_{G}\ \max_{D}\ V(D,G) =
\mathbb{E}_{X\sim p_{\text{data}}}\!\big[\log D(X)\big]
+ \mathbb{E}_{z\sim p_z}\!\big[\log\big(1 - D(G(z))\big)\big].
$$

Qui $X\sim p_{\text{data}}$ è un campione reale, $z\sim p_z$ è il rumore in
ingresso a $G$, $D(X)$ è la probabilità stimata che l'input sia autentico. Il
Discriminatore *massimizza* $V$ (assegna probabilità alta ai veri, bassa ai
falsi $G(z)$); il Generatore *minimizza* il secondo termine, cioè spinge
$D(G(z))$ verso $1$. All'ottimo teorico si ha
$p_{\text{model}}=p_{\text{data}}$ e $D(X)=\tfrac{1}{2}$ ovunque: l'esperto
non sa più decidere. In pratica l'equilibrio è delicato: instabilità
dell'addestramento e *mode collapse* (il Generatore che produce sempre la
stessa immagine vincente) sono i due grattacapi ricorrenti, che affronteremo
più avanti.

`````

## Perché ce ne importa

Le GAN hanno spostato il confine di ciò che una macchina può *fabbricare*. Da
questa idea nascono i **volti fotorealistici** di persone inesistenti: la
famiglia StyleGAN di NVIDIA (Karras et al., 2019) alimenta siti come *This
Person Does Not Exist*, dove ogni ricarica mostra un volto sintetico
indistinguibile da una fotografia. Da qui arrivano anche i **deepfake** (volti
sostituiti nei video) con tutto il loro carico di rischi per disinformazione e
consenso, un tema su cui questo libro sceglie l'onestà più che l'entusiasmo. E
arriva l'**arte generata**: nel 2018 il ritratto *Edmond de Belamy*, prodotto
con una GAN dal collettivo Obvious, è stato battuto da Christie's per circa
432.500 dollari, la prima opera così venduta da una grande casa d'aste.

Uno strumento potente e ambivalente, insomma: capace di creare dataset,
restaurare immagini e progettare molecole, ma anche di fabbricare falsi
convincenti. Ragione in più per capirne bene il funzionamento.

## Come è organizzato il capitolo

Dall'intuizione passiamo alla pratica. Nelle prossime sezioni vedremo
**l'architettura** del Generatore e del Discriminatore e come collegarli; la
**funzione di perdita** e il ciclo di addestramento alternato, con le sue
insidie (instabilità, *mode collapse*) e i trucchi per domarle; il problema,
tutt'altro che ovvio, di **misurare** se una GAN sta funzionando, visto che la
sua loss non lo dice; le
**varianti** che hanno fatto la storia, DCGAN, le GAN condizionali, fino a
StyleGAN; e infine un'**implementazione in PyTorch**, dove costruiremo una GAN
che impara a generare cifre manoscritte partendo, letteralmente, dal rumore.
