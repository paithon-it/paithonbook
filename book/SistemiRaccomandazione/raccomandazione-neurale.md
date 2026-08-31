# La raccomandazione neurale

Intorno al 2016 il deep learning aveva già conquistato il riconoscimento delle
immagini e stava conquistando il linguaggio, e la domanda era nell'aria: una
rete al posto del confronto voce per voce farebbe meglio? Quel confronto, il
prodotto scalare della sezione precedente, è pur sempre una regola di calcolo
fissa, decisa a tavolino da chi ha scritto il modello, mentre una rete la
regola se la cerca da sé. E può cercarla molto lontano: il {doc}`capitolo sul Deep
Learning </DeepLearning/overview>` racconta che una rete abbastanza grande sa imitare, con la precisione
che si vuole, quasi qualunque legame fra un ingresso e un'uscita. Il paper che
diede forma alla domanda è *Neural Collaborative Filtering*
{cite}`he2017neural`, e la risposta è più interessante di un semplice «sì»: è
un piccolo caso di studio su cosa significa davvero «più potente» in machine
learning.

## Dal prodotto scalare alla rete

L'idea del Neural Collaborative Filtering (NCF) è chirurgica. Si tiene tutto
l'impianto della fattorizzazione, una scheda di numeri per ogni utente e una
per ogni film, e si cambia solo l'ultimo passo: al posto del confronto voce per
voce va una rete, che *impara* da sé come leggere insieme le due schede
({numref}`fig-ncf-architettura`).

```{figure} ../figures/ncf-architettura.svg
:name: fig-ncf-architettura
:alt: Gli identificativi di utente e film passano da due tabelle di embedding, i due vettori vengono concatenati e un percettrone multistrato produce il punteggio di affinità.
:width: 95%

L'architettura NCF: le schede di numeri dell'utente e del film vengono
incollate una sotto l'altra (*concatenate*) e passate a una piccola rete a più
strati (un percettrone multistrato, in sigla MLP), che al posto del confronto
voce per voce produce il punteggio di affinità. Il disegno si ferma un passo
prima della fine: nel modello quel punteggio passa ancora per una funzione che
lo schiaccia fra zero e uno, la sigmoide, perché si legga come una
probabilità.
```

`````{tab} Elementare

Nella fattorizzazione, il confronto tra la scheda dell'utente e quella del
film è una regola fissa: si moltiplicano le voci corrispondenti e si somma (la
manopola "commedia" dell'utente incontra solo la manopola "commedia" del film,
mai le altre). È come giudicare una coppia sommando i punti in comune, voce
per voce.

Il NCF cambia il giudice. Le due schede vengono incollate una sotto l'altra e
consegnate a una piccola rete neurale, che durante l'addestramento impara *da
sola* come leggerle insieme. In teoria può cogliere combinazioni che la somma
voce per voce non vede (l'equivalente di «ama i documentari, *ma solo se*
durano meno di un'ora»), perché nessuno le impone di trattare le voci a
coppie.

Per trovarsi la regola da sé, però, il giudice nuovo ha bisogno di vedere
tantissime coppie già giudicate, e ne ha viste pochissime: la tabella da cui
impara è quasi tutta vuota. C'è anche una versione che tiene
tutti e due i modi di leggere le schede, il confronto voce per voce e la rete,
e li fa decidere insieme: ognuno però si porta le sue schede, così ogni utente
e ogni film ne hanno due invece di una, e il paragone con la fattorizzazione
non è più alla pari, perché da quella parte i numeri da imparare sono il
doppio.

`````

`````{tab} Superiore

Con gli embedding $\mathbf{p}_u, \mathbf{q}_i \in \mathbb{R}^k$ della sezione
precedente, il NCF sostituisce il prodotto scalare con un percettrone
multistrato applicato alla concatenazione:

$$
\hat{y}_{ui} \;=\; \sigma\!\Big( f_{\theta}\big([\,\mathbf{p}_u \,;\, \mathbf{q}_i\,]\big) \Big) ,
$$

dove $[\,\cdot\,;\,\cdot\,]$ è la concatenazione dei due vettori,
$f_{\theta}$ un MLP con attivazioni ReLU e $\sigma$ la sigmoide, che
schiaccia l'uscita in $(0,1)$: il modello è pensato per feedback implicito,
e $\hat{y}_{ui}$ si legge come probabilità di interazione.

Una parola sui simboli, prima di andare avanti. Il cappello indica sempre la
predizione, ma la lettera sotto cambia con il compito, e in questo capitolo
seguiamo quella dei paper d'origine: $\hat{r}_{ui}$ per un voto da prevedere
(fattorizzazione), $\hat{y}_{ui}$ per una probabilità di interazione (NCF),
$\hat{x}_{ui}$ per un punteggio di ranking (BPR), dove conta solo l'ordine e
non il valore assoluto.

Il paper propone anche una variante che affianca i due mondi (*NeuMF*): un
ramo con il prodotto elemento per elemento e un ramo MLP, **ciascuno con la
propria coppia di tabelle di embedding**, fusi concatenando l'ultimo strato
nascosto. Che le tabelle siano separate conta, ed è il paper stesso ad
argomentarlo: condividerle costringerebbe i due rami alla
stessa dimensione degli embedding, e gli autori scrivono che questo potrebbe
limitare le prestazioni del modello fuso. La
conseguenza da tenere a mente è che NeuMF ha il doppio dei parametri di
embedding di una fattorizzazione a parità di $k$, quindi il confronto fra i due
non è a parità di capacità: cosa che rende il risultato di Rendle, fra poco,
ancora più netto. In linea di principio l'MLP, per il teorema di
approssimazione universale {cite}`hornik1991approximation` (nella versione di
Leshno et al. {cite}`leshno1993multilayer`, che copre attivazioni illimitate
come la ReLU), può approssimare con precisione arbitraria, su un compatto,
qualunque interazione continua tra i fattori; ma approssimare non è
rappresentare esattamente, e se poi la *impari* davvero da dati sparsi è
un'altra faccenda ancora.

`````

Qui serve una dose di onestà intellettuale, e non riguarda un paper solo. Nel
2019 due ricercatori del Politecnico di Milano e un collega dell'università di
Klagenfurt hanno provato a rifare i conti di diciotto metodi neurali per la
raccomandazione, presentati alle conferenze principali {cite}`dacrema2019are`.
Solo sette si sono lasciati riprodurre con uno sforzo ragionevole, e di quei
sette **sei venivano spesso battuti da metodi molto più semplici**: i vicini
della sezione precedente, o tecniche su grafo. Il settimo batteva i termini di
paragone scelti dai suoi autori, ma perdeva contro un metodo lineare, cioè
senza reti neurali, quando qualcuno si prendeva la briga di tararlo bene. Il
lavoro
ha vinto il premio per il miglior articolo lungo di RecSys, la conferenza del
settore, e ha spostato la domanda che si fa a un risultato nuovo: non
«funziona?» ma «meglio di che cosa, tarato da chi?».

L'anno dopo Steffen Rendle e colleghi hanno rifatto la stessa operazione
proprio sul NCF, sugli stessi banchi di prova del paper originale
{cite}`rendle2020neural`. Hanno trovato due cose: che il vecchio confronto voce
per voce, tarato con cura, batte la rete; e che per una rete imparare a
riprodurre un confronto voce per voce, partendo da dati sparsi, è
sorprendentemente difficile.

La morale non è «le reti non servono», ma qualcosa di più fine, e conviene
dirla in italiano prima che in gergo: **più libertà non è gratis**. Lasciare al
modello la libertà di scoprire da sé come confrontare due schede sembra un
regalo, e invece gli toglie l'unica cosa che sapeva già di sicuro, cioè che le
voci vanno confrontate a coppie. Il confronto voce per voce non è una rigidità
arbitraria: è un'ipotesi giusta sul problema, e su dati scarsi un'ipotesi
giusta vale più di mille parametri in più.

E il confronto voce per voce ha un secondo pregio, che con la qualità non
c'entra: è l'unica forma di punteggio che permette di **non** calcolarne uno
per ogni titolo del catalogo. Su cataloghi da milioni di titoli è questo, più
della qualità, a decidere se il sistema sta in piedi. Il motivo in breve: le
schede dei titoli si possono preparare tutte
in anticipo e mettere in uno scaffale ordinato, dove trovare le più vicine alla
tua non richiede di guardarle tutte; una rete, invece, va fatta girare una
volta per ogni titolo, e nessuno può farla girare un milione di volte per ogni
persona che apre l'app (ci torniamo in fondo al capitolo). Su dati fitti, cioè
il contrario
della tabella quasi vuota di prima, le reti ripagano; e ripagano ancora di più
quando accanto alle interazioni c'è dell'altro da guardare, l'ora, il
dispositivo, il prezzo, il genere, che nel gergo del mestiere si chiamano
*feature*. Sul filtraggio collaborativo puro, invece, il vecchio prodotto
scalare ben tarato resta un avversario durissimo.

## La matrice è un grafo

C'è un secondo modo di andare oltre il confronto voce per voce, e non consiste
nel rendere più furba la regola che confronta le due schede: consiste nel darle
più cose da guardare. Per vederlo basta riscrivere lo stesso dato in un'altra
forma. (Le due parole del titolo, in breve: la tabella dei voti in matematica
si chiama **matrice**, e il disegno di pallini e linee che stiamo per fare si
chiama **grafo**.)

`````{tab} Elementare

La tabella utenti per film si può disegnare invece che tabulare. Metti tutti
gli utenti in una colonna di pallini a sinistra, tutti i film in una colonna a
destra, e tira una linea ogni volta che qualcuno ha visto qualcosa. I pallini
sono i *nodi* e le linee gli *archi*, le parole del capitolo sulle reti
neurali su grafo; qui continueremo a dire pallini e linee, che si vedono
meglio. Non hai
aggiunto né tolto niente: è lo stesso dato, disegnato. Ma adesso si vede una
cosa che nella tabella era nascosta, e cioè che **raccomandare vuol dire
indovinare le linee che ancora non ci sono**.

Vista così, la fattorizzazione guarda vicino: la scheda di ognuno riassume le
linee che partono dal suo pallino, e per giudicare una coppia si confrontano
quelle due schede. Non è poco (le schede sono proprio la mossa che permette di
confrontare due persone senza film in comune) ma è **un passo solo** di
distanza. Il metodo dei vicini della sezione precedente arriva più in là: da
te, ai film che hai visto, alle persone che li hanno visti, e da lì ai film che
loro hanno visto e tu no. Sono tre passi. E poi? Perché fermarsi lì? Un film
può interessarti perché piace a persone che a loro volta somigliano a chi
somiglia a te, e per scoprire un legame così bisogna camminare più a lungo. Il
grafo permette di raccogliere quel segnale lontano; la tabella no, perché lì i
passi non si vedono.

Camminare così ha un nome, **propagazione**: a ogni passo ogni pallino si
riscrive mescolando ciò che gli arriva dai pallini a cui è collegato, ma non
tutto allo stesso volume, perché chi è collegato a mezzo catalogo parla più
piano e un film visto da tutti conta poco per ognuno dei suoi spettatori. Dopo
tre o quattro passi ogni pallino ha in pancia anche notizie che vengono da
lontano. E c'è un
modello del 2020, **LightGCN**, famoso proprio perché non fa altro: niente rete
neurale sopra, solo il camminare, ripetuto qualche volta e rimesso insieme alla
fine. È nato per sottrazione: qualcuno ha preso un modello che faceva di più e
gli ha tolto dei pezzi, scoprendo che così andava meglio.

`````

`````{tab} Superiore

La matrice di interazione $\mathbf{R} \in \{0,1\}^{n \times m}$, dove $n$ è il
numero degli utenti e $m$ quello degli oggetti (le stesse lettere della figura
della sezione precedente), è la matrice di adiacenza di un grafo **bipartito**
utente-oggetto, a meno di riscriverla in forma simmetrica:

$$
\mathbf{A} = \begin{pmatrix} \mathbf{0} & \mathbf{R} \\ \mathbf{R}^\top & \mathbf{0} \end{pmatrix} .
$$

Su un grafo si può propagare, ed è esattamente il *message passing* del
capitolo sulle reti neurali su grafo. Nella forma più nuda, l'embedding di un
utente al passo $\ell+1$ è una somma pesata degli embedding degli oggetti con
cui ha interagito, e viceversa:

$$
\mathbf{e}_u^{(\ell+1)} = \sum_{i \in \mathcal{N}(u)}
\frac{1}{\sqrt{|\mathcal{N}(u)|\,|\mathcal{N}(i)|}}\; \mathbf{e}_i^{(\ell)},
\qquad
\mathbf{e}_i^{(\ell+1)} = \sum_{u \in \mathcal{N}(i)}
\frac{1}{\sqrt{|\mathcal{N}(i)|\,|\mathcal{N}(u)|}}\; \mathbf{e}_u^{(\ell)} .
$$

Il peso è la stessa normalizzazione simmetrica dei gradi vista per la GCN
(non una media: i coefficienti non sommano a uno), con una differenza da non
scavalcare: la GCN del capitolo sui grafi normalizza
$\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}$, cioè con i cappi, e qui la somma
corre sui soli $\mathcal{N}(u)$, senza cappio. Non è una svista dei paper:
LightGCN scarta le *self-connection* per scelta dichiarata, mostrando che la
combinazione finale degli strati (che include lo strato $0$) ne cattura già
l'effetto. La lettura per il resto è la stessa: un utente che ha visto tutto, o
un film visto da tutti, contano meno per singolo arco. Impilare $L$ strati
significa raccogliere segnale da $L$ salti di distanza.

L'idea è nell'aria dal 2017, quando GC-MC formulò il completamento della
matrice come convoluzione sul grafo bipartito {cite}`vandenberg2017graph`. La
tappa canonica è **NGCF** {cite}`wang2019neural`, che ricalca la GCN completa:
trasformazione lineare, non linearità, propagazione.
**LightGCN** toglie i primi due e tiene solo il terzo {cite}`he2020lightgcn`,
combinando poi gli strati con pesi uniformi
$\mathbf{e}_u = \sum_{\ell=0}^{L} \frac{1}{L+1} \mathbf{e}_u^{(\ell)}$ e
tornando al prodotto scalare per il punteggio. Solo embedding e propagazione:
nessun peso da imparare oltre alla tabella iniziale. Costa molto meno, e
funziona meglio **di NGCF**: circa il 16% di miglioramento relativo medio, a
parità di protocollo sperimentale. È il termine di paragone da tenere a mente,
perché è interno alla famiglia dei metodi a grafo: il paper non sta dicendo che
LightGCN batte una fattorizzazione ben tarata, sta dicendo che togliere pezzi a
NGCF lo migliora. Detto altrimenti, è il prodotto scalare della sezione
precedente ad avere resistito anche qui.

`````

La morale somiglia a quella del paragrafo su Rendle, e conviene metterle in
fila. Una precisazione però conta, e di solito si salta: le due storie non
pesano allo stesso modo come prova. La prima, il riesame del NCF, l'hanno
fatta persone diverse da chi il metodo l'aveva proposto, ritarando con cura
gli avversari e facendoli correre di nuovo. La seconda è il paper di LightGCN,
cioè i suoi autori che riportano la propria vittoria: è quello che fa chiunque
pubblichi, ed è proprio per questo che da sola pesa meno. Detto questo, la
direzione è la stessa, ed è quella già incontrata: **più libertà non è
gratis**. NCF mette una rete al posto del confronto voce per voce e non
guadagna niente; LightGCN toglie la rete, tiene solo il camminare, e batte il
modello più complicato da cui è stato ricavato. Camminare sul grafo, in fondo,
è un modo di dire al modello una cosa che il confronto voce per voce non sa:
*chi ha visto cose simili alle tue va ascoltato, anche a più di un passo di
distanza*. È un'ipotesi migliore su come è fatto il problema, non più potenza
di calcolo.

Il disegno dà anche una risposta parziale alla partenza a freddo, il muro
contro cui la sezione precedente si era fermata, e quella risposta conviene
guardarla bene: è una delle idee più eleganti del capitolo.

Nella tabella, un film appena uscito è una riga vuota, e da una riga vuota non
si estrae niente: fine del discorso. Nel disegno no, perché nel disegno nulla
obbliga i pallini a essere solo persone e film. Si possono aggiungere pallini
di un altro colore: il regista, il genere, l'attore protagonista, l'etichetta
messa da qualcuno. Un film uscito ieri non ha ancora nessuna linea verso gli
spettatori, ma ha già le sue linee verso il regista e verso gli attori, e
quelle bastano: camminando su di esse il film raccoglie qualcosa da tutti gli
altri film dello stesso regista, e si presenta al sistema con una scheda
sensata prima ancora che qualcuno lo guardi. Un film senza spettatori è un film
di cui sappiamo tutto tranne la cosa che ci interessa. Un grafo con più tipi di
pallini e di linee si dice **eterogeneo**.

Riformulare la raccomandazione come *link prediction*, cioè come il compito di
prevedere gli archi che mancano, è la lettura che rende disponibile tutto
l'armamentario delle reti su grafo, e non un gioco di parole. Il caso più
noto, **PinSage** {cite}`ying2018graph`, è raccontato nel capitolo sulle reti
neurali su grafo, insieme al campionamento dei vicini che lo rende praticabile
a scala web. Leggere la raccomandazione come link prediction non è però *la
definizione* del problema, ed è bene non prenderla per tale: un disegno di
pallini e linee non ha un orologio, e i sistemi che girano davvero restano
organizzati intorno al confronto fra due schede.

## Imparare a ordinare: BPR

Il vero salto della raccomandazione moderna non sta nel disegno del modello:
sta in che cosa gli si chiede di indovinare. Quando non ci sono voti, ma solo
la traccia di quello che uno ha guardato (il feedback implicito della prima
pagina del capitolo), non c'è nessun numero da prevedere. C'è l'elenco di ciò
che hai guardato e l'oceano di ciò che non hai guardato; e quell'oceano, lo
sappiamo dalla panoramica, non è un elenco di bocciature. La **Bayesian
Personalized Ranking** (BPR) prende sul serio questa asimmetria: smette di
prevedere valori e impara direttamente a *ordinare*
{cite}`rendle2009bpr`. Delle tre parole del nome quella che conta è l'ultima,
*ranking*, che vuol dire mettere in fila. Le altre due dicono da dove viene la
formula: *personalized* perché la fila è diversa per ogni persona, *bayesian*
perché la si ricava partendo da un'ipotesi su come sono fatti i numeri del
modello, dichiarata prima ancora di guardare i dati.

`````{tab} Elementare

Sistemi la vetrina di una libreria per un cliente
abituale. Non conosci i suoi voti, ma sai cosa ha comprato. La regola di BPR è
tutta qui: *ciò che ha scelto deve stare più in alto di ciò che ha ignorato*.
A ogni passo peschi una coppia (un libro che ha comprato, uno a caso tra i
mille che non ha mai toccato) e controlli la tua vetrina: se il libro comprato
sta già ben sopra, va bene così, quasi nessuna correzione; se sta sotto, sistemi
la vetrina spostandolo su. Ripetuto milioni di volte, questo gioco di
confronti a coppie produce una classifica personale senza che nessuno abbia
mai dato un voto. Nota la finezza: non serve decidere *quanto* gli piace ogni
libro; serve solo che l'ordine sia giusto.

Una domanda onesta, a questo punto: e se il
libro pescato a caso era proprio uno che gli sarebbe piaciuto, e che non ha
comprato solo perché non l'ha mai visto? Succede, e per un istante lo stiamo
spingendo giù per sbaglio. Il gioco regge lo stesso, per due motivi. Il primo è
che su un catalogo grande capita di rado, e più il catalogo è grande più capita
di rado (un libro che quel cliente ha già comprato non finisce mai fra gli
ignorati, quello si riconosce; uno che gli sarebbe piaciuto e che non ha mai
visto sì, e non c'è modo di accorgersene). Il secondo, che conta di più:
ogni singolo
confronto sposta la vetrina di pochissimo, quindi dopo milioni di confronti
resta impressa la regolarità, non lo sbaglio di uno di essi. È il motivo per
cui questo metodo vuole tantissimi confronti approssimativi e non pochi giudizi
precisi.

E c'è un momento in cui il gioco smette di insegnare: a vetrina quasi a posto i
libri pescati a caso stanno già tutti sotto, e da un confronto già vinto non si
impara niente. Da lì in poi i confronti bisogna sceglierli, non pescarli.

`````

`````{tab} Superiore

Sia $\hat{x}_{ui}$ il punteggio che un modello qualunque assegna alla coppia
$(u,i)$: nel paper è una fattorizzazione ridotta al solo prodotto scalare,
$\hat{x}_{ui} = \mathbf{p}_u^\top \mathbf{q}_i$, senza nessuno dei termini
additivi della sezione precedente. Due dei tre spariscono da sé, perché BPR
confronta sempre due item *dello stesso* utente e nella differenza $\mu$ e $b_u$
si elidono. Il bias di item no: sopravvive come $b_v - b_w$, e nel paper
semplicemente non c'è. Un modo naturale di leggere quell'assenza (il paper non
la discute) è che quel termine è la stessa quantità per tutti gli utenti, cioè
la parte *non* personalizzata dell'ordinamento: la P di *Personalized*.
Rimetterlo è legittimo e varie implementazioni lo fanno, al prezzo di una
classifica che per un utente senza storia collassa su quella dei titoli più
popolari, che è però anche il meglio che si possa fare quando di quell'utente
non si sa nulla.

BPR costruisce triple $(u, v, w)$: un utente $u$, un item $v$ con cui ha
interagito, un item $w$ campionato tra quelli mai toccati. La loss chiede che
$v$ superi $w$:

$$
\mathcal{L}_{\text{BPR}} \;=\;
-\sum_{(u,v,w)} \log \sigma\big(\hat{x}_{uv} - \hat{x}_{uw}\big)
\;+\; \lambda\,\lVert\theta\rVert^2 ,
$$

dove $\sigma$ è la sigmoide e $\theta$ raccoglie tutti i parametri del
modello. La lettura probabilistica è elegante:
$\sigma(\hat{x}_{uv} - \hat{x}_{uw})$ è la probabilità, secondo il modello,
che $u$ preferisca $v$ a $w$; la loss è la log-verosimiglianza negativa di
aver ordinato bene tutte le coppie, assunte indipendenti tra loro (senza
questa ipotesi il prodotto delle sigmoidi non sarebbe una verosimiglianza). E
il termine $\lambda\,\lVert\theta\rVert^2$ non è una regolarizzazione
qualsiasi: nel paper nasce come prior gaussiano sui parametri di una stima
MAP, ed è lì la "B" di *Bayesian*. Conta solo la *differenza* dei punteggi,
non il loro valore assoluto. E il gradiente ha il comportamento giusto: coppie
già ben ordinate con margine ampio contribuiscono quasi zero, coppie invertite
spingono forte. In pratica i negativi $w$ si campionano a caso a ogni passo,
con l'accortezza che, a modello maturo, i negativi "facili" non insegnano più
nulla e il campionamento intelligente dei negativi difficili diventa metà del
mestiere.

`````

```{figure} ../figures/vetrina-si-ordina.svg
:name: fig-vetrina-si-ordina
:alt: "Una vetrina di dieci libri in colonna, dal posto 1 al posto 10. I quattro che il cliente ha comprato partono sparsi, tre di loro nella metà bassa. A ogni confronto si pesca una coppia formata da un libro comprato e da uno ignorato: se il comprato sta già sopra la spinta è quasi nulla e la vetrina non si muove, se sta sotto sale di uno o più posti e l'ignorato scende. A un certo punto l'ignorato pescato è un libro che al cliente sarebbe piaciuto: scende di un posto per sbaglio e due confronti dopo è già risalito. Dopo ottanta confronti i quattro comprati sono i primi quattro, e la loss media è scesa da 1,17 a 0,05."
:width: 95%

Ottanta confronti a coppie su una vetrina di dieci libri; l'animazione mostra
uno per uno i primi dieci, poi salta al risultato. Ogni confronto pesca un
libro comprato e uno ignorato: se il comprato sta già sopra la spinta è quasi
nulla e la vetrina resta ferma, se sta sotto risale di uno o più posti. A un
certo punto l'ignorato pescato è il libro $E$ del disegno, che a quel cliente
sarebbe piaciuto davvero: scende di un posto per sbaglio, e due confronti dopo
è già risalito. Alla fine i quattro comprati sono i primi quattro, e nessuno ha
mai dato un voto.
```

In PyTorch la misura di quanto il modello sta sbagliando (la **loss**) è una
riga, e si innesta sul modello di fattorizzazione della sezione precedente
senza toccarlo. È un frammento, non un programma completo: `modello`, `u` e
`n_film` sono quelli di là, e `positivi` è l'elenco delle coppie (utente, film)
che si conoscono.

```{code-block} python
:class: pt-non-eseguibile

import random
import torch
import torch.nn.functional as F

def loss_bpr(x_uv, x_uw):
    # x_uv: punteggi (utente, item visto) · x_uw: (utente, item ignorato)
    return -F.logsigmoid(x_uv - x_uw).mean()

def pesca_negativo(utente, positivi, n_film):
    """Un item che `utente` non ha mai toccato: si pesca finché non ne esce uno."""
    w = random.randrange(n_film)
    while (utente, w) in positivi:   # `positivi`: il set delle coppie note
        w = random.randrange(n_film)
    return w

# nel ciclo di addestramento: v = item con cui l'utente ha interagito, w = item
# mai toccato da quell'utente. Pescandolo a caso in tutto il catalogo ogni
# tanto uscirebbe un positivo: su una matrice piena al 5-10%, come MovieLens
# 100K, una volta su dieci o venti; sui cataloghi veri, molto piu' di rado.
# Costa poco ripescare.
w = torch.tensor([pesca_negativo(int(x), positivi, n_film) for x in u])
loss = loss_bpr(modello(u, v), modello(u, w))  # stesso modello di prima
```

Quella riga, detta in italiano: guarda di quanto il libro comprato sta sopra a
quello ignorato, e trasforma quel margine in una spinta. Se il comprato sta già
molto sopra, la spinta è quasi zero e la vetrina non si muove; se sta sotto, la
spinta cresce, e cresce tanto più quanto è sotto.

`F.logsigmoid` fa in un passaggio solo due conti che si potrebbero anche fare
separati, e non è un vezzo. Il primo schiaccia il margine fra zero e uno, ed è
il mestiere della sigmoide: così si legge come una probabilità, «quanto il
modello è convinto di aver messo i due libri nell'ordine giusto». Il secondo
trasforma quella probabilità nel punteggio da minimizzare, e per farlo ne
prende il logaritmo cambiato di segno (è il meno davanti a `F.logsigmoid` nel
codice): quel punteggio vale zero quando la probabilità è uno e cresce senza
limite quando la probabilità si avvicina a zero, che è esattamente il
comportamento che serve a una misura di errore.

Il guaio, se i due conti si fanno separati, è che «senza limite» il computer
non lo regge. Quando il libro comprato sta molto sotto quello ignorato, diciamo
cento posizioni di punteggio, la sigmoide restituisce un numero con più di
quaranta zeri dopo la virgola, e la macchina non riesce più a distinguerlo
dallo zero: scrive proprio zero. Sullo zero il logaritmo non ha una risposta
finita, il calcolatore stampa `-inf`, e da lì in poi ogni conto che ci passa
sopra è rovinato. Fatti insieme, invece, i due passaggi si semplificano a
vicenda e il punteggio resta un numero: esattamente $100$.

## Misurare una classifica

Se il compito è mettere in ordine, anche il metro deve cambiare. Contare di
quanto si sbaglia sui voti non serve a niente qui: una vetrina è buona o
cattiva per l'ordine in cui ci stanno i titoli, e di voti non ce n'è nemmeno
uno. I metri buoni per una classifica guardano solo la lista dei primi dieci o
venti suggerimenti, perché è l'unica cosa che l'utente vedrà.

Prima del metro, però, c'è una domanda che si salta quasi sempre e che pesa più
del metro: **su che cosa si misura**. Nessuno può dire se ti sarebbe piaciuto
un titolo che non hai mai visto, quindi si procede per finta: si prende la
storia di un utente, si **nasconde** una parte di ciò che ha davvero guardato,
si addestra il modello su quel che resta, e poi si guarda quanti dei titoli
nascosti il modello rimette in cima alla lista. I titoli nascosti sono il metro
di verità: si sa che gli interessavano, perché li ha guardati, e si sa che il
modello non li ha visti, perché glieli abbiamo tolti noi. È un trucco, e come
tutti i trucchi funziona finché si ricorda che è un trucco.

`````{tab} Elementare

I titoli nascosti sono 6, il sistema ne mostra 10, e 3 di quei 10 stanno fra i
nascosti. La **precision@10** (la chiocciola si legge «sui primi dieci») è la
frazione di consigli azzeccati: $3/10 = 0{,}3$. Il **recall@10** misura invece
quanti dei 6 nascosti ne ha ritrovati: $3/6 = 0{,}5$. Le due metriche tirano in
direzioni opposte: sparare consigli a raffica alza il recall e affonda la
precision.

C'è però un dettaglio che entrambe ignorano: *dove* stanno i colpi
azzeccati. Un successo al primo posto vale più di uno al decimo, perché al
decimo posto forse non arrivi mai. La **NDCG** è la metrica che ne tiene
conto: premia le classifiche che mettono i titoli giusti in cima, come un
giornale che sceglie bene la prima pagina. Quanto premia, in cifre: un titolo
giusto al primo posto
vale $1$, lo stesso titolo al secondo posto vale $0{,}63$, al decimo $0{,}29$.
Lo sconto cala sempre più piano man mano che si scende: fra il primo e il
secondo posto c'è più differenza ($0{,}37$) che fra il quinto e il decimo
($0{,}10$).

I punti si sommano, e poi si dividono per il punteggio della classifica
perfetta, quella che avrebbe messo i titoli giusti tutti in testa: così il
risultato sta sempre fra $0$ e $1$ ed è confrontabile fra persone diverse, che
altrimenti chi ha sei titoli nascosti raccoglierebbe più punti di chi ne ha due
solo perché ne ha di più.

Finiamo l'esempio di prima. I 3 titoli azzeccati stiano ai posti 1, 4 e 7:
valgono $1 + 0{,}43 + 0{,}33 = 1{,}76$. La classifica perfetta avrebbe messo
tutti e 6 i nascosti in cima, dal primo al sesto posto, per un totale di
$3{,}30$. La **NDCG@10** è $1{,}76 / 3{,}30 = 0{,}53$.

E nascondere si può fare in più modi, che non sono equivalenti. Togliere un
pezzo di storia **a caso** è comodo e imbroglia: il modello si addestra anche
su cose successe *dopo* quelle su cui viene interrogato, e nella vita vera il
futuro non è disponibile. Nascondere **l'ultima cosa** che ciascuno ha guardato
è più onesto. **Tagliare a una data** è il più severo, e l'unico che somiglia
alla situazione vera: fa comparire anche chi a quella data era appena arrivato,
cioè proprio le persone su cui si sbaglia di più. Cambiando modo di nascondere,
la classifica dei metodi può ribaltarsi. E c'è una seconda decisione che
nessuno dichiara: contro quanti titoli deve farsi largo quello nascosto.
Batterne cento presi a caso è tutt'altra impresa che batterne un milione, e i
due risultati si chiamano allo stesso modo. Contare su cento gonfia il
punteggio, e non lo gonfia allo stesso modo per tutti: anche qui l'ordine fra
due sistemi può rovesciarsi.

`````

`````{tab} Superiore

Detto $\mathrm{Ril}_u$ l'insieme degli item rilevanti per $u$ (nel test:
le interazioni nascoste) e $\mathrm{Top}_k(u)$ i primi $k$ raccomandati:

$$
\text{precision@}k = \frac{\lvert \mathrm{Ril}_u \cap \mathrm{Top}_k(u)\rvert}{k},
\qquad
\text{recall@}k = \frac{\lvert \mathrm{Ril}_u \cap \mathrm{Top}_k(u)\rvert}{\lvert \mathrm{Ril}_u \rvert}.
$$

Per pesare le posizioni si usa la *Discounted Cumulative Gain*:

$$
\mathrm{DCG@}k \;=\; \sum_{j=1}^{k} \frac{\mathrm{rel}_j}{\log_2(j+1)},
\qquad
\mathrm{NDCG@}k \;=\; \frac{\mathrm{DCG@}k}{\mathrm{IDCG@}k} \in [0,1],
$$

dove $\mathrm{rel}_j$ è la rilevanza dell'item in posizione $j$ (binaria qui;
per la rilevanza graduata la convenzione prevalente in *information retrieval*
mette $2^{\mathrm{rel}_j} - 1$ al numeratore, e le due forme coincidono solo
nel caso binario) e $\mathrm{IDCG@}k$ è la DCG della classifica ideale, che
normalizza il punteggio tra utenti con numeri diversi di item rilevanti. Lo
sconto logaritmico penalizza dolcemente: la posizione 2 vale
$1/\log_2 3 \approx 0{,}63$ della posizione 1. Quando il test ha **un solo**
item rilevante per utente, che è il caso del protocollo *leave-one-out* con cui
è valutato NCF, la recall@k degenera nella *hit rate* HR@k ed è con quel nome
che la si trova nei paper; la metrica naturale diventa allora la **MRR**
(*mean reciprocal rank*), $\frac{1}{|\mathcal{U}|}\sum_u 1/\mathrm{rank}_u$,
cioè la media dell'inverso della posizione in cui è finito l'unico item
giusto.

Tutte queste metriche si mediano sugli utenti; e tutte ereditano il difetto
della valutazione offline: misurano il recupero di interazioni passate,
avvenute sotto l'esposizione del vecchio sistema, non il gradimento futuro.

**Come si nasconde.** «Nascondere una parte delle interazioni» sotto-specifica
la decisione che sposta i risultati più di qualunque scelta di modello, e le
opzioni in uso sono tre, con costi diversi. *Split casuale sulle interazioni*:
comodo, ed è la scelta maggioritaria in letteratura, ma mette
nell'addestramento interazioni **successive** a quelle di test, cioè addestra
il modello su un futuro che al momento della predizione non esisteva; è una
fuga di informazione, e gonfia i numeri {cite}`ji2023critical`.
*Leave-one-out* sull'ultima interazione di ciascun utente: rispetta la
cronologia del singolo utente, non quella globale, perché il modello vede
comunque il futuro degli altri. *Taglio a un istante globale*: l'unico che
riproduce la situazione di produzione, ed è di gran lunga il più severo, perché
fa emergere gli utenti che al momento della predizione non avevano ancora
storia. La differenza non è di livello ma di ordine: sui benchmark classici,
passando da split casuale a temporale, l'ordine fra una fattorizzazione e la
banale classifica dei più popolari può rovesciarsi, in buona parte proprio per
via di quegli utenti freddi.

**Su quanti candidati.** Seconda decisione tacita, e stessa morale. La
precision, la recall e la NDCG, così definite, suppongono di ordinare l'intero
catalogo non interagito, e ordinarlo tutto costa: molti lavori mettono in
classifica l'item di test contro poche decine o centinaia di negativi
campionati (è, alla lettera, il protocollo con cui sono prodotti i numeri di
NCF: 100 negativi per utente). Le due
quantità portano lo stesso nome e non sono confrontabili, perché battere cento
concorrenti è molto più facile che batterne un milione. Il guaio peggiore però
è un altro: il gonfiamento **non è uguale per tutti i modelli**,
quindi la metrica campionata può invertire l'ordine fra due sistemi
{cite}`krichene2020sampled`. Leggendo un Recall@10 in un paper, conviene sempre
cercare prima su quanti candidati è stato calcolato.

`````

## La storia recente conta

Un limite silenzioso di tutto ciò che abbiamo visto: la matrice dei voti non
ha orologio. Per la fattorizzazione, il film visto ieri sera e quello di dieci
anni fa pesano uguale. Ma chi ha appena comprato una tenda da campeggio è, per
qualche giorno, una persona diversa: sacco a pelo e fornelletto sono consigli
d'oro oggi e rumore tra un mese. La **raccomandazione sequenziale** tratta la
storia dell'utente come una frase da continuare: prevedere la prossima
interazione come si prevede la prossima parola. Le cose da portarsi via sono
due: la storia recente pesa più di quella vecchia, e i modelli del linguaggio
sanno già trattare le sequenze. Gli strumenti li avete
già visti nei {doc}`capitoli sul NLP </NaturalLanguageProcessing/overview>` e sui Transformer; qui cambia solo che al posto
delle parole ci sono i titoli del catalogo.

Non a caso il settore ha seguito la stessa parabola del NLP: prima le reti
ricorrenti (GRU4Rec {cite}`hidasi2016session`), poi l'auto-attenzione (SASRec
{cite}`kang2018self` e BERT4Rec {cite}`sun2019bert4rec`, che sono Transformer
in tutto e per tutto). Quei tre nomi sono lì per essere riconosciuti se li
incontrate, non per essere imparati. E vale
l'avvertenza di poche pagine fa: che i modelli si susseguano in ordine di
pubblicazione non vuol dire che si susseguano in ordine di qualità, e per
saperlo servono le riprove indipendenti, non gli annunci.

## Come lo fa l'industria

Un'ultima dose di realismo, ed è la sezione che racconta cosa succede davvero
nell'attimo fra il momento in cui apri l'app e il momento in cui compare la
prima riga di suggerimenti. Nessuna piattaforma calcola un punteggio
raffinato per milioni di titoli a ogni visita: i sistemi reali lavorano **a due
stadi**, e li descrissero pubblicamente gli ingegneri di YouTube nel 2016
{cite}`covington2016deep`.

`````{tab} Elementare

Un concorso ha un milione di iscritti e una giuria di dieci persone.
Nessuna giuria può ascoltare un milione di candidati: si fa una scrematura
rapidissima e grossolana, che da un milione ne tiene qualche centinaio, e poi
la giuria vera ascolta solo quelli. Chi consiglia i video fa la stessa cosa, e
la fa da capo ogni volta che apri l'app, in una frazione di secondo.

**Il primo tempo** (nel gergo, il primo *stadio*) è la scrematura, e deve
essere velocissima, quindi il lavoro grosso è già stato fatto la notte prima:
per ogni titolo del catalogo la scheda di numeri è già lì, calcolata e messa in
uno scaffale ordinato. Conta da che cosa è fatta, quella scheda: se la si
ricava guardando il titolo (di che parla, chi l'ha girato) allora ce l'ha anche
un film uscito stanotte; se invece è solo un numero cresciuto a forza di
visioni, un film che nessuno ha ancora guardato resta senza. Quando arrivi tu,
si calcola solo la *tua* scheda, tenendo conto anche di quello che hai
guardato oggi, e poi si cerca sullo scaffale quali schede di titoli le
somigliano di più. Questa
ricerca è **approssimata** nel senso che non le guarda tutte: sullo scaffale le
schede che si somigliano stanno vicine, e questo permette di scartare interi
ripiani senza aprirli. Ogni tanto ci si perde per strada un titolo buono, e in
cambio si va enormemente più veloci: a questo punto è un baratto che conviene
sempre.

**Il secondo tempo** è la giuria: sulle poche centinaia di superstiti si può
finalmente spendere del calcolo, e lì entra tutto ciò che il primo tempo non
poteva guardare, cioè che ore sono, da che dispositivo stai guardando, quante
volte quel titolo ti è già stato messo davanti senza che tu lo aprissi. È qui
che si decide l'ordine di quello che vedi.

`````

`````{tab} Superiore

Il primo stadio, il *retrieval*, screma il catalogo da milioni a qualche
centinaio di candidati con un modello volutamente semplice. Lo schema che si è
imposto è la **two-tower**: due reti separate producono l'una l'embedding
dell'utente, l'altra quello dell'item a partire dalle sue feature, e il
punteggio è il loro prodotto scalare. La separazione è il punto: gli embedding
degli item si precalcolano tutti offline, e a richiesta basta una ricerca dei
vicini più prossimi approssimata (ANN) nello spazio degli embedding, cioè il
prodotto scalare "rigido" della sezione precedente, riabilitato
dall'efficienza. Il secondo stadio, il *ranking*, applica ai soli sopravvissuti
un modello ricco quanto si vuole, con centinaia di feature di contesto (ora,
dispositivo, storia recente).

Le attribuzioni vanno separate, perché la letteratura le confonde
spesso. Covington et al. 2016 descrivono i **due stadi** e il recupero per
prodotto scalare con vicini approssimati, e in quel paper il modello di
candidate generation è una rete sola, sull'utente: i vettori dei video sono i
**pesi dello strato softmax di uscita**, non l'uscita di una seconda torre. La
two-tower propriamente detta, con una rete anche sul lato item e la correzione
del bias di campionamento che la rende addestrabile su cataloghi enormi, si
afferma negli anni successivi {cite}`yi2019sampling`. La differenza non è
terminologica: una torre che legge le *feature* dell'item sa dare un embedding
anche a un item mai visto, un peso appreso per identificativo no, ed è
esattamente la partenza a freddo.

`````

E una cosa che sorprende sempre, in un sistema del genere: il lavoro difficile
sta quasi sempre nell'impianto che tiene tutto aggiornato, più che nel modello. Uno
scaffale di schede vecchie di una settimana consiglia benissimo la settimana
scorsa.

## Suggerire o pilotare?

Chiudiamo con la domanda che questo capitolo si porta dietro fin dalla matrice
vuota, la stessa annunciata nella prima pagina: un sistema che decide cosa
vedi, e impara da ciò che vedi, ti sta *servendo* o ti sta *plasmando*? Nel
2011 l'attivista Eli Pariser ha dato un nome alla paura
{cite}`pariser2011filter`: *filter bubble*, la bolla in cui l'algoritmo,
inseguendo i tuoi click, ti mostra sempre più di ciò che già pensi. Gli studi
empirici hanno poi restituito un quadro meno netto, e per certi versi
sorprendente. Prendiamo le notizie online. Immagina di dare a ogni giornale un
posto su una riga che va da sinistra a destra, e a ogni lettore il posto medio
dei giornali che legge: si può allora misurare quanto due lettori sono lontani.
Chi arriva agli articoli passando da un motore di ricerca o dai social risulta,
in media, più lontano dagli altri lettori di chi va dritto sul sito del
giornale: quei canali dividono di più, e fin qui la bolla c'è. Ma le stesse
persone, proprio passando di lì, finiscono **più spesso** anche su articoli
della parte politica che gradiscono meno {cite}`flaxman2016filter`. Le due cose
non si contraddicono, perché non misurano la stessa cosa: la prima dice quanto i
lettori sono distanti fra loro, la seconda quanto ciascuno di loro incontra
l'altra campana. Ed è la seconda a sorprendere. Resta però vero il meccanismo
da cui è nata la paura di Pariser, il feedback loop già incontrato: il modello
impara da dati che il modello stesso ha filtrato, come discusso nella sezione
{doc}`Quando i dati cambiano </MachineLearning/dati-che-cambiano>`.

Il punto critico non è la tecnica, è la metrica. Un sistema addestrato a
massimizzare i minuti di visione imparerà, con perfetta onestà matematica, a
mostrare tutto ciò che ci tiene incollati allo schermo: l'indignazione e il
sensazionalismo compresi, se funzionano. Chi sceglie il metro su cui il sistema
viene premiato (la **funzione obiettivo**, come si dice in gergo) sceglie, in
ultima analisi, il
comportamento che il sistema coltiverà nei suoi utenti: è qui che passa il
confine tra suggerire e pilotare. Le contromisure esistono e sono concrete. Si
può misurare, accanto a quanto il sistema ci azzecca, anche quanto la lista è
varia e quanto spesso fa incontrare qualcosa di buono che non si stava
cercando, e quest'ultima si chiama *serendipità*. Si possono mettere controlli
espliciti nelle mani di chi il sistema lo usa. E da qualche anno c'è anche la
legge: in Europa il Digital Services Act impone alle grandi piattaforme di
offrire almeno una versione del loro sistema di raccomandazione che non si basi
sulla **profilazione**, cioè sulla ricostruzione dei gusti di ciascuno a
partire da quello che ha fatto.
Nessuna di queste è una soluzione definitiva. Ma un ingegnere che sa *come*
funziona la macchina (e ora lo sapete) è esattamente la persona nella
posizione giusta per pretendere che funzioni bene. E lo è, per un'altra via,
anche chi la macchina la subisce e basta: sapere che il consiglio nasce da un
confronto fra schede di numeri a cui nessuno ha dato un nome, che la lista è
già stata scremata
prima che tu arrivassi, e che il sistema insegue il metro su cui è stato
premiato, è ciò che trasforma «me l'ha consigliato l'app» in una frase che si
può discutere. Da questa parte dello schermo non si progetta niente, ma si può
smettere di prendere la vetrina per il catalogo.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Il **NCF** cambia il giudice: invece di confrontare le due schede voce per
  voce con una regola fissa, le incolla una sotto l'altra e lascia decidere a
  una piccola rete. In teoria vede combinazioni che il confronto voce per voce
  non coglie; alla prova dei fatti il vecchio confronto, tarato con cura, resta
  un avversario durissimo: più libertà non è gratis. E il caso non è isolato:
  quando si è provato a rifare i conti di diciotto metodi neurali, sette si
  sono lasciati riprodurre e sei di quei sette perdevano contro metodi molto
  più semplici.
- La tabella dei voti si può disegnare: utenti da una parte, film dall'altra,
  una linea per ogni visione. Raccomandare vuol dire **indovinare le linee che
  ancora non ci sono**. Camminando sul disegno per più passi si raccoglie anche
  il segnale lontano, e **LightGCN** mostra che per farlo non serve una rete
  sopra: basta camminare.
- Quando non ci sono voti ma solo ciò che l'utente ha guardato, non si prevede
  un numero, si sistema una vetrina: ciò che ha scelto deve stare più in alto
  di un titolo preso a caso fra i mille che ha ignorato (**BPR**). Conta
  l'ordine, non quanto gli piace ogni titolo.
- Una classifica si misura su quanti dei consigli mostrati sono azzeccati
  (**precision**), su quanti dei titoli buoni ha ritrovato (**recall**) e su
  quanto in alto li ha messi (**NDCG**), come un giornale che sceglie bene la
  prima pagina. Ma prima ancora conta **su che cosa** si misura: si nasconde
  una parte di ciò che l'utente ha davvero guardato e si controlla se il
  modello la ritrova. Ribaltano la classifica dei metodi sia il modo di
  nascondere, sia il numero di titoli contro cui il nascosto deve farsi largo,
  ed è la parte più fragile del mestiere.
- I sistemi veri lavorano in **due tempi**: un primo filtro rapido e grossolano
  che da milioni di titoli ne tiene qualche centinaio, poi un giudizio accurato
  sui soli superstiti.
- Il metro che scegli plasma il sistema, e chi lo usa: premiato sui minuti di
  visione, imparerà tutto ciò che trattiene. Il confine tra suggerire e
  pilotare passa da lì.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Il **NCF** sostituisce il prodotto scalare con un MLP sulla concatenazione
  degli embedding: più espressivo in teoria, ma un prodotto scalare ben tarato
  resta un avversario durissimo. Il prodotto scalare è un ottimo *bias
  induttivo* per questo problema, e su dati sparsi un buon bias induttivo vale
  più di parametri in più. Che il fenomeno sia sistemico, e non aneddotico, lo
  documenta la riproducibilità mancata di 11 metodi neurali su 18.
- La matrice di interazione **è** il grafo bipartito utente-oggetto, e leggerla
  come **link prediction** (prevedere gli archi mancanti) è una riformulazione
  feconda, non la definizione del problema. Propagare
  sul grafo raccoglie segnale a più salti; **LightGCN** mostra che basta la
  propagazione, senza rete sopra, e il suo +16% è misurato su NGCF, non su una
  fattorizzazione ritarata.
- Con feedback implicito si impara a **ordinare**, non a prevedere voti:
  la loss **BPR** $-\log\sigma(\hat{x}_{uv}-\hat{x}_{uw})$ chiede solo che
  l'item scelto superi quello ignorato (in codice, `-F.logsigmoid(·)`, che non
  esplode).
- Le classifiche si misurano con **precision@k**, **recall@k** e **NDCG**,
  che premia i successi in cima alla lista. Due decisioni tacite le governano:
  **come** si costruisce il test (casuale, leave-one-out, taglio temporale) e
  **su quanti candidati** si ordina (catalogo intero o negativi campionati).
  Entrambe possono invertire l'ordine fra due modelli.
- I sistemi reali sono a **due stadi**: retrieval con vicini approssimati su
  embedding precalcolati, poi ranking fine sui candidati superstiti. I due
  stadi sono di Covington et al. 2016; la **two-tower** con una rete anche sul
  lato item, che è quella che dà un embedding a un item mai visto, viene dopo.
- La metrica scelta plasma il comportamento del sistema, e degli utenti: il
  confine tra suggerire e pilotare passa dalla funzione obiettivo.
```

`````

Due cose da portarsi dietro, e non valgono solo per le classifiche di film. La
prima è che il metro scelto plasma il sistema che si costruisce. La seconda è
che il modo di dividere i dati per misurare può ribaltare la graduatoria dei
metodi. Quella divisione, nel capitolo sulle serie temporali, smette di essere
una scelta, perché i dati hanno un verso e misurare vuol dire non lasciare che
il modello sbirci il futuro che gli si sta chiedendo di prevedere.
