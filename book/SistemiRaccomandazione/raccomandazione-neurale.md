# La raccomandazione neurale

Intorno al 2016 il deep learning aveva già conquistato la visione e stava
conquistando il linguaggio, e la domanda era nell'aria: perché la
raccomandazione dovrebbe accontentarsi di un confronto voce per voce? Quel
confronto, il prodotto scalare della sezione precedente, è pur sempre una
regola di calcolo fissa, decisa a tavolino da chi ha scritto il modello. Una
rete neurale, invece, sa imitare con la precisione che si vuole qualunque
regola leghi ingressi e uscite senza salti bruschi, come racconta la panoramica
del capitolo sul Deep Learning. Il paper che diede forma alla domanda è *Neural
Collaborative Filtering* {cite}`he2017neural`, e la risposta è più
interessante di un semplice «sì»: è un piccolo caso di studio su cosa
significa davvero «più potente» in machine learning.

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
nascosto. Che le tabelle siano separate non è un dettaglio implementativo, ed è
il paper stesso ad argomentarlo: condividerle costringerebbe i due rami alla
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
della sezione precedente, o tecniche su grafo. Il settimo batteva i metodi di
riferimento, ma non un metodo lineare, senza reti, tarato con cura. Il lavoro
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
giusta vale più di mille parametri in più. Ed è anche l'unica forma che permette
di **non** calcolare un punteggio per ogni titolo del catalogo, e su cataloghi
da milioni di titoli è questo, più della qualità, a decidere se il sistema sta
in piedi. Il motivo in breve: le schede dei titoli si possono preparare tutte
in anticipo e mettere in uno scaffale ordinato, dove trovare le più vicine alla
tua non richiede di guardarle tutte; una rete, invece, va fatta girare una
volta per ogni titolo, e un milione di volte a testa non le fa nessuno (ci
torniamo in fondo al capitolo). Su dati fitti, cioè il contrario
della tabella quasi vuota di prima, le reti ripagano; e ripagano ancora di più
quando accanto alle interazioni c'è dell'altro da guardare, l'ora, il
dispositivo, il prezzo, il genere, che nel gergo del mestiere si chiamano
*feature*. Sul filtraggio collaborativo puro, invece, il vecchio prodotto
scalare ben tarato resta un avversario durissimo.

## La matrice è un grafo

C'è un secondo modo di andare oltre il confronto voce per voce, e non consiste
nel rendere più furba la regola che confronta le due schede: consiste nel darle
più cose da guardare. Per vederlo basta riscrivere lo stesso dato in un'altra
forma.

`````{tab} Elementare

La tabella utenti per film si può disegnare invece che tabulare. Metti tutti
gli utenti in una colonna di pallini a sinistra, tutti i film in una colonna a
destra, e tira una linea ogni volta che qualcuno ha visto qualcosa. I pallini
sono i *nodi* e le linee gli *archi*: sono le parole del capitolo precedente,
quello sui grafi, e da qui in avanti il testo userà quelle. Non hai
aggiunto né tolto niente: è lo stesso dato, disegnato. Ma adesso si vede una
cosa che nella tabella era nascosta, e cioè che **raccomandare vuol dire
indovinare le linee che ancora non ci sono**.

Vista così, la fattorizzazione guarda vicino: la scheda di ognuno riassume le
linee che partono dal suo pallino, e per giudicare una coppia si confrontano
quelle due schede. Non è poco (le schede sono proprio la mossa che permette di
confrontare due persone senza film in comune) ma è **un passo solo** di
distanza. Il filtraggio per vicinato della sezione precedente arriva un passo
più in là: da te, ai film che hai visto, alle persone che li hanno visti. E
poi? Perché fermarsi a due passi? Un film può somigliarti perché piace a gente
che ha gusti simili ai
tuoi, e quella somiglianza si scopre camminando sul disegno per tre, quattro
passi. Il grafo permette di raccogliere quel segnale lontano; la tabella no,
perché lì i passi non si vedono.

Camminare così ha un nome, **propagazione**: a ogni passo ogni pallino si
riscrive mescolando ciò che gli arriva dai pallini a cui è collegato, e dopo
tre o quattro passi ha in pancia anche notizie che vengono da lontano. E c'è un
modello del 2020, **LightGCN**, famoso proprio perché non fa altro: niente rete
neurale sopra, solo il camminare, ripetuto qualche volta e rimesso insieme alla
fine. È nato togliendo pezzi a un modello che ne aveva di più, e quel modello
lo batte: togliere, qui, è servito.

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

La morale somiglia a quella del paragrafo su Rendle, e vale la pena metterle in
fila. Una precisazione però conta, e di solito si salta: i due episodi non
valgono come prova allo stesso modo. Il primo è una rivalutazione fatta da
altri, che ha ritarato i concorrenti e li ha fatti correre di nuovo; il secondo
è un metodo che riporta i propri risultati, e i propri risultati li riportano
tutti.
Detto questo, la direzione è la stessa, ed è quella già incontrata: **più
libertà non è gratis**. NCF aggiunge una rete al posto del confronto voce per
voce e non guadagna niente; LightGCN toglie la rete, tiene solo il camminare, e
batte il modello più carico da cui è stato ricavato. Camminare sul grafo, in
fondo, è un modo di dire al modello una cosa
che il confronto voce per voce non sa: *chi ha visto cose simili alle tue va
ascoltato, anche a più di un passo di distanza*. Non è più potenza di calcolo:
è un'ipotesi migliore su come è fatto il problema.

Il disegno dà anche una risposta parziale alla partenza a freddo, il muro
contro cui la sezione precedente si era fermata, e vale la pena guardarla bene
perché è una delle idee più eleganti del capitolo.

Nella tabella, un film appena uscito è una riga vuota, e da una riga vuota non
si estrae niente: fine del discorso. Nel disegno no, perché nel disegno nulla
obbliga i pallini a essere solo persone e film. Si possono aggiungere pallini
di un altro colore: il regista, il genere, l'attore protagonista, l'etichetta
messa da qualcuno. Un film uscito ieri non ha ancora nessuna linea verso gli
spettatori, ma ha già le sue linee verso il regista e verso gli attori, e
quelle bastano: camminando su di esse il film raccoglie qualcosa da tutti gli
altri film dello stesso regista, e si presenta al sistema con una scheda
sensata prima ancora che qualcuno lo guardi. Un film senza spettatori non è un
film sconosciuto: è un film di cui sappiamo tutto tranne la cosa che ci
interessa. Un grafo con più tipi di pallini e di linee si dice **eterogeneo**.

Riformulare la raccomandazione come *link prediction*, cioè come il compito di
prevedere gli archi che mancano, non è un gioco di parole: è la lettura che
rende disponibile tutto l'armamentario delle reti su grafo. Il caso più
noto, **PinSage** {cite}`ying2018graph`, è raccontato nel capitolo sulle reti
neurali su grafo, insieme al campionamento dei vicini che lo rende praticabile
a scala web. Non è però *la definizione* del problema, ed è bene non prenderla
per tale: più avanti in questa pagina due paragrafi ne mostrano i limiti da due
lati diversi, perché un grafo statico non ha un orologio, e i sistemi che
girano davvero restano organizzati intorno al confronto fra due schede.

## Imparare a ordinare: BPR

Il vero salto concettuale della raccomandazione moderna non è
nell'architettura: è nell'obiettivo. Con il feedback implicito non ci sono voti
da prevedere. C'è l'elenco di ciò che hai guardato e l'oceano di ciò che non
hai guardato; e quell'oceano, lo sappiamo dalla panoramica, non è un elenco di
bocciature. La **Bayesian Personalized Ranking** (BPR) prende sul serio questa
asimmetria: smette di prevedere valori e impara direttamente a *ordinare*
{cite}`rendle2009bpr`. Delle tre parole del nome quella che conta è l'ultima,
*ranking*, che vuol dire mettere in fila; le altre due dicono come è stata
ricavata la formula, e le spiega la versione formale qui sotto.

`````{tab} Elementare

Immagina di dover sistemare la vetrina di una libreria per un cliente
abituale. Non conosci i suoi voti, ma sai cosa ha comprato. La regola di BPR è
tutta qui: *ciò che ha scelto deve stare più in alto di ciò che ha ignorato*.
A ogni passo peschi una coppia (un libro che ha comprato, uno a caso tra i
mille che non ha mai toccato) e controlli la tua vetrina: se il libro comprato
sta già sopra, va bene così, quasi nessuna correzione; se sta sotto, sistemi
la vetrina spostandolo su. Ripetuto milioni di volte, questo gioco di
confronti a coppie produce una classifica personale senza che nessuno abbia
mai dato un voto. Nota la finezza: non serve decidere *quanto* gli piace ogni
libro; serve solo che l'ordine sia giusto.

Una domanda onesta, a questo punto, ed è la prima che viene in mente: e se il
libro pescato a caso era proprio uno che gli sarebbe piaciuto, e che non ha
comprato solo perché non l'ha mai visto? Succede, e per un istante lo stiamo
spingendo giù per sbaglio. Il gioco regge lo stesso, per due motivi. Il primo è
che su un catalogo grande capita di rado, e più il catalogo è grande più capita
di rado (nel codice qui sotto lo evitiamo del tutto, almeno per i libri che
quel cliente ha già preso). Il secondo, che conta di più: ogni singolo
confronto sposta la vetrina di pochissimo, quindi dopo milioni di confronti
resta impressa la regolarità, non lo sbaglio di uno di essi. È il motivo per
cui questo metodo vuole tantissimi confronti approssimativi e non pochi giudizi
precisi.

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
certo punto
l'ignorato pescato è $E$, che a quel cliente sarebbe piaciuto davvero: scende
di un posto per sbaglio, e due confronti dopo è già risalito. Alla fine i
quattro comprati sono i primi quattro, e nessuno ha mai dato un voto.
```

In PyTorch la misura di quanto il modello sta sbagliando (la **loss**) è una
riga, e si innesta sul modello di fattorizzazione della sezione precedente
senza toccarlo:

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
spinta cresce, e cresce tanto più quanto è sotto. Il nome della funzione,
`F.logsigmoid`, tiene insieme in un passaggio solo due conti che si potrebbero
fare separati, e non è un vezzo: fatti separati, quando il comprato sta molto
sotto, il computer arrotonda a zero il risultato intermedio e il conto finale
esce infinito. Tenuti insieme, resta un numero.

## Misurare una classifica

Se l'obiettivo è ordinare, anche il metro deve cambiare: l'errore quadratico
sui voti non dice nulla sulla qualità di una vetrina. Le metriche di ranking
guardano la lista dei primi $k$ suggerimenti, con $k$ piccolo, dieci o venti,
perché è l'unica cosa che l'utente vedrà.

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

Supponi che i titoli nascosti fossero 6, che il sistema te ne mostri 10, e che
3 dei 10 fossero fra i nascosti. La **precision@10** è la frazione di consigli
azzeccati: $3/10 = 0{,}3$. Il **recall@10** misura invece quanti dei 6 ne ha
ritrovati: $3/6 = 0{,}5$. Le
due metriche tirano in direzioni opposte: sparare consigli a raffica alza il
recall e affonda la precision.

C'è però un dettaglio che entrambe ignorano: *dove* stanno i colpi
azzeccati. Un successo al primo posto vale più di uno al decimo, perché al
decimo posto forse non arrivi mai. La **NDCG** è la metrica che ne tiene
conto: premia le classifiche che mettono i titoli giusti in cima, come un
giornale che sceglie bene la prima pagina. Il nome non vuol dire niente in
italiano, sono le iniziali di quattro parole inglesi: è un'etichetta, non una
sigla da decifrare. Quanto premia, in cifre: un titolo giusto al primo posto
vale $1$, lo stesso titolo al secondo posto vale $0{,}63$, al decimo $0{,}29$.
I punti si sommano e poi si dividono per il punteggio della classifica
perfetta, quella che avrebbe messo i titoli giusti tutti in testa, così il
risultato sta sempre fra $0$ e $1$ e le persone si possono confrontare fra
loro.

E nascondere si può fare in più modi, che non sono affatto equivalenti.
Si può togliere un pezzo di storia **a caso**, che è comodo e bara: il
modello finisce per addestrarsi anche su cose successe *dopo* quelle su cui
viene interrogato, e nella vita vera il futuro non è disponibile. Si può
nascondere **l'ultima cosa** che ciascuno ha guardato, che è più onesto. Oppure
si può **tagliare a una data**: tutto quello che è successo prima serve per
imparare, tutto quello che viene dopo per giudicare. L'ultimo è il più severo,
ed è l'unico che somiglia alla situazione vera, perché fa comparire anche le
persone che a quella data erano appena arrivate e di cui non si sapeva nulla,
che sono proprio quelle su cui si sbaglia di più. Cambiando modo di nascondere,
la classifica dei metodi può ribaltarsi: ecco perché «su che cosa si misura»
viene prima del metro.

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
che il capitolo non usa ma che il lettore incontrerà.

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
definizione data qui suppone di ordinare l'intero catalogo non interagito, e
ordinarlo tutto costa: molti lavori mettono in classifica l'item di test contro
poche decine o centinaia di negativi campionati (è, alla lettera, il protocollo
con cui sono prodotti i numeri di NCF: 100 negativi per utente). Le due
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
interazione come si prevede la prossima parola. La cosa da portarsi via è
tutta qui, e sono due: la storia recente pesa più di quella vecchia, e i
modelli del linguaggio sanno già trattare le sequenze. Gli strumenti li avete
già visti nei capitoli sul NLP e sui Transformer; qui cambia solo cosa c'è al
posto delle parole, e al posto delle parole c'è il catalogo.

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
nei due secondi fra il momento in cui apri l'app e il momento in cui compare la
prima riga di suggerimenti. Nessuna piattaforma calcola un punteggio
raffinato per milioni di titoli a ogni visita: i sistemi reali lavorano **a due
stadi**, e li descrissero pubblicamente gli ingegneri di YouTube nel 2016
{cite}`covington2016deep`.

`````{tab} Elementare

Immagina un concorso con un milione di iscritti e una giuria di dieci persone.
Nessuna giuria può ascoltare un milione di candidati: si fa una scrematura
rapidissima e grossolana, che da un milione ne tiene qualche centinaio, e poi
la giuria vera ascolta solo quelli. Chi consiglia i video fa la stessa cosa, e
la fa da capo ogni volta che apri l'app, in una frazione di secondo.

**Il primo tempo** è la scrematura, e deve essere velocissima, quindi il lavoro
grosso è già stato fatto la notte prima: per ogni titolo del catalogo la scheda
di numeri è già lì, calcolata e messa in cassetto. Quando arrivi tu, si calcola
solo la *tua* scheda, che è l'unica che può essere cambiata da quello che hai
fatto dieci minuti fa, e poi si cerca nel cassetto quali schede di titoli le
somigliano di più. Questa ricerca è **approssimata** nel senso che non le
guarda tutte: nel cassetto le schede che si somigliano stanno vicine, e questo
permette di scartare interi scomparti senza aprirli.
Ogni tanto ci si perde per strada un titolo buono, e in cambio si va
enormemente più veloci: a questo stadio è un baratto che conviene sempre. È
anche il momento in cui il
vecchio confronto voce per voce si prende la rivincita: è l'unica forma di
punteggio che permette di preparare tutto in anticipo così.

**Il secondo tempo** è la giuria: sulle poche centinaia di superstiti si può
finalmente spendere, e lì entra tutto ciò che il primo tempo non poteva
guardare, cioè che ore sono, da che dispositivo stai guardando, cosa hai visto
dieci minuti fa. È qui che si decide l'ordine di quello che vedi.

E una cosa che sorprende sempre: il lavoro difficile, in un sistema del genere,
non è quasi mai il modello. È l'impianto che tiene tutto aggiornato, perché un
cassetto di schede vecchie di una settimana consiglia benissimo la settimana
scorsa.

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

Vale la pena separare le attribuzioni, perché la letteratura le confonde
spesso. Covington et al. 2016 descrivono i **due stadi** e il recupero per
prodotto scalare con vicini approssimati, e in quel paper il modello di
candidate generation è una rete sola, sull'utente: i vettori dei video sono i
**pesi dello strato softmax di uscita**, non l'uscita di una seconda torre. La
two-tower propriamente detta, con una rete anche sul lato item e la correzione
del bias di campionamento che la rende addestrabile su cataloghi enormi, si
afferma negli anni successivi {cite}`yi2019sampling`. La differenza non è
terminologica: una torre che legge le *feature* dell'item sa dare un embedding
anche a un item mai visto, un peso appreso per identificativo no, ed è
esattamente la partenza a freddo di cui il capitolo si occupa a lungo.

`````

Vale per entrambi i livelli la stessa chiusa: gran parte del lavoro vero, in un
sistema di raccomandazione industriale, non è nel modello ma nell'infrastruttura
che lo tiene fresco.

## Suggerire o pilotare?

Chiudiamo con la domanda che questo capitolo si porta dietro fin dalla matrice
vuota, la stessa annunciata nella prima pagina: un sistema che decide cosa
vedi, e impara da ciò che vedi, ti sta *servendo* o ti sta *plasmando*? Nel
2011 l'attivista Eli Pariser ha dato un nome alla paura
{cite}`pariser2011filter`: *filter bubble*, la bolla in cui l'algoritmo,
inseguendo i tuoi click, ti mostra sempre più di ciò che già pensi. Gli studi
empirici hanno poi restituito un quadro meno netto, e per certi versi
sorprendente. Prendiamo le notizie online. Chi ci arriva passando da un motore
di ricerca o dai social legge, in media, cose più lontane dalle opinioni degli
altri lettori rispetto a chi va dritto sul sito del giornale; ma quelle stesse
persone, insieme, incontrano **più spesso** anche articoli del lato politico
che preferiscono meno {cite}`flaxman2016filter`. Le due cose valgono
contemporaneamente, e la seconda è quella che non ci si aspetta. Il meccanismo
di fondo però è reale,
ed è il feedback loop già incontrato: il modello impara da dati che il modello
stesso ha filtrato, come discusso nella sezione *Quando i dati cambiano* del
capitolo di Machine Learning.

Il punto critico non è la tecnica, è la metrica. Un sistema addestrato a
massimizzare i minuti di visione imparerà, con perfetta onestà matematica,
tutto ciò che trattiene: inclusi l'indignazione e il sensazionalismo, se
trattengono. Chi sceglie il metro su cui il sistema viene premiato (la
**funzione obiettivo**, come si dice in gergo) sceglie, in ultima analisi, il
comportamento che il sistema coltiverà nei suoi utenti: è qui che passa il
confine tra suggerire e pilotare. Le contromisure esistono e sono concrete:
misurare accanto all'accuratezza anche quanto la lista è varia e quanto spesso
fa incontrare qualcosa di buono che non si stava cercando (la *serendipità*),
mettere controlli espliciti nelle mani di chi legge, e da qualche anno anche la
legge (in Europa
il Digital Services Act impone alle grandi piattaforme di offrire almeno una
versione del loro sistema di raccomandazione non basata sulla profilazione).
Nessuna di queste è una soluzione definitiva. Ma un ingegnere che sa *come*
funziona la macchina (e ora lo sapete) è esattamente la persona nella
posizione giusta per pretendere che funzioni bene. E lo è, per un'altra via,
anche chi la macchina la subisce e basta: sapere che il consiglio nasce da un
confronto fra schede di numeri senza nome, che la lista è già stata scremata
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
  un avversario durissimo: più libertà non è gratis. E non è un caso isolato:
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
  modello la ritrova. Cambiare il modo di nascondere può ribaltare la
  classifica dei metodi, ed è la parte più fragile del mestiere.
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
