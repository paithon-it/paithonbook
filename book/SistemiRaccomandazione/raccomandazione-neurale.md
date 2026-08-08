# La raccomandazione neurale

Intorno al 2016 il deep learning aveva già conquistato la visione e stava
conquistando il linguaggio, e la domanda era nell'aria: perché la
raccomandazione dovrebbe accontentarsi di un prodotto scalare? Un prodotto
scalare è pur sempre una funzione fissa e piuttosto rigida; le reti neurali,
lo sappiamo dal capitolo sulle reti, possono approssimare funzioni molto più
generali. Il paper che diede forma alla domanda è *Neural Collaborative
Filtering* {cite}`he2017neural`. La risposta, come vedremo, è più
interessante di un semplice "sì": è un piccolo caso di studio su cosa
significa davvero "più potente" in machine learning.

## Dal prodotto scalare alla rete

L'idea del Neural Collaborative Filtering (NCF) è chirurgica: tenere tutto
l'impianto della fattorizzazione (un embedding per utente, un embedding per
film) e sostituire solo l'ultimo passo, il prodotto scalare, con una rete che
*impara* come combinare i due vettori ({numref}`fig-ncf-architettura`).

```{figure} ../figures/ncf-architettura.svg
:name: fig-ncf-architettura
:alt: Gli identificativi di utente e film passano da due tabelle di embedding, i due vettori vengono concatenati e un percettrone multistrato produce il punteggio di affinità.
:width: 95%

L'architettura NCF: gli embedding di utente e film vengono concatenati e un
MLP, al posto del prodotto scalare, produce il punteggio di affinità.
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

Con gli embedding $P_u, Q_i \in \mathbb{R}^k$ della sezione precedente, il
NCF sostituisce il prodotto scalare con un percettrone multistrato applicato
alla concatenazione:

$$
\hat{y}_{ui} \;=\; \sigma\!\Big( f_{\theta}\big([\,P_u \,;\, Q_i\,]\big) \Big) ,
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
ramo con il prodotto elemento per elemento degli embedding e un ramo MLP, fusi
nell'ultimo strato. In linea di principio l'MLP, per il teorema di
approssimazione universale {cite}`hornik1991approximation` (nella versione di
Leshno et al. {cite}`leshno1993multilayer`, che copre attivazioni illimitate
come la ReLU), può approssimare con precisione arbitraria, su un compatto,
qualunque interazione continua tra i fattori; ma approssimare non è
rappresentare esattamente, e se poi la *impari* davvero da dati sparsi è
un'altra faccenda ancora.

`````

Qui serve una dose di onestà intellettuale. Nel 2020 Steffen Rendle e colleghi
{cite}`rendle2020neural` hanno rifatto i conti sugli stessi benchmark del
paper originale, e hanno trovato che un semplice prodotto scalare, con
iperparametri scelti con cura, batte l'MLP, e che per una rete imparare a
riprodurre un prodotto scalare da dati sparsi è sorprendentemente difficile.
La morale non è "le reti non servono", ma qualcosa di più fine: più capacità
espressiva non è gratis, e il prodotto scalare non è una rigidità arbitraria
bensì un ottimo *bias induttivo* per questo problema, oltre a essere migliaia
di volte più economico da calcolare su un catalogo di milioni di titoli. Su
dati densi e ricchi di feature le reti ripagano; sul filtraggio collaborativo
puro, il vecchio prodotto scalare ben tarato resta un avversario durissimo.

## La matrice è un grafo

C'è un secondo modo di superare il prodotto scalare, e non passa dal rendere
più furbo il giudice: passa dal dargli più cose da guardare. Per vederlo basta
riscrivere lo stesso dato in un'altra forma.

`````{tab} Elementare

La tabella utenti per film si può disegnare invece che tabulare. Metti tutti
gli utenti in una colonna di pallini a sinistra, tutti i film in una colonna a
destra, e tira una linea ogni volta che qualcuno ha visto qualcosa. Non hai
aggiunto né tolto niente: è lo stesso dato, disegnato. Ma adesso si vede una
cosa che nella tabella era nascosta, e cioè che **raccomandare vuol dire
indovinare le linee che ancora non ci sono**.

Vista così, la fattorizzazione guarda pochissimo: per giudicare una coppia
utente-film usa solo le linee che partono da quei due pallini. Il filtraggio
per vicinato del capitolo precedente arriva un passo più in là (da te, ai film
che hai visto, alle persone che li hanno visti). E poi? Perché fermarsi a due
passi? Un film può somigliarti perché piace a gente che ha gusti simili ai
tuoi, e quella somiglianza si scopre camminando sul disegno per tre, quattro
passi. Il grafo permette di raccogliere quel segnale lontano; la tabella no,
perché lì i passi non si vedono.

`````

`````{tab} Superiore

La matrice di interazione $R \in \{0,1\}^{m \times n}$ è la matrice di
adiacenza di un grafo **bipartito** utente-oggetto, a meno di riscriverla in
forma simmetrica:

$$
A = \begin{pmatrix} 0 & R \\ R^\top & 0 \end{pmatrix} .
$$

Su un grafo si può propagare, ed è esattamente il *message passing* del
capitolo sulle reti neurali su grafo. Nella forma più nuda, l'embedding di un
utente al passo $k+1$ è una somma pesata degli embedding degli oggetti con
cui ha interagito, e viceversa:

$$
e_u^{(k+1)} = \sum_{i \in \mathcal{N}(u)}
\frac{1}{\sqrt{|\mathcal{N}(u)|\,|\mathcal{N}(i)|}}\; e_i^{(k)},
\qquad
e_i^{(k+1)} = \sum_{u \in \mathcal{N}(i)}
\frac{1}{\sqrt{|\mathcal{N}(i)|\,|\mathcal{N}(u)|}}\; e_u^{(k)} .
$$

Il peso è la stessa normalizzazione simmetrica dei gradi vista per la GCN
(non una media: i coefficienti non sommano a uno), e la stessa lettura vale
qui: un utente che ha visto tutto, o un film visto da tutti, contano meno per
singolo arco. Impilare $K$ strati significa raccogliere segnale da $K$ salti
di distanza.

L'idea è nell'aria dal 2017, quando GC-MC {cite}`vandenberg2017graph` formulò
il completamento della matrice come convoluzione sul grafo bipartito; la tappa
canonica è **NGCF** {cite}`wang2019neural`, che ricalca la GCN completa:
trasformazione lineare, non linearità, propagazione.
**LightGCN** {cite}`he2020lightgcn` toglie i primi due e tiene solo il terzo,
combinando poi gli strati con pesi uniformi
$e_u = \sum_{k=0}^{K} \frac{1}{K+1} e_u^{(k)}$ e tornando al prodotto scalare
per il punteggio. Solo embedding e propagazione: nessun peso da imparare oltre
alla tabella iniziale. Funziona meglio, e costa molto meno.

`````

La morale è la stessa del paragrafo su Rendle, e vale la pena metterla in
fila, perché due volte di seguito in questo capitolo la stessa cosa si è
rivelata vera: **quel che serviva non era più capacità espressiva, era il
giusto bias induttivo**. NCF aggiunge una rete al posto del prodotto scalare e
non guadagna; LightGCN toglie la rete e tiene la propagazione, e guadagna. La
propagazione sul grafo, in fondo, è un modo di dire al modello una cosa che il
prodotto scalare non sa: *chi ha visto cose simili alle tue va ascoltato,
anche a più di un passo di distanza*.

Il grafo dà anche una risposta parziale al problema dell'avvio a freddo visto
alla fine della sezione precedente. Un oggetto nuovo, nella matrice, è una
riga vuota, e da una riga vuota non si estrae niente. In un grafo, invece,
nulla vieta di aggiungere nodi che non siano né utenti né oggetti: il regista,
il genere, l'attore protagonista, il tag. Un film appena uscito ha zero archi
verso gli utenti ma già i suoi archi verso gli attributi, e la propagazione gli
consegna un embedding sensato prima ancora che qualcuno lo guardi. Il grafo
smette di essere bipartito e diventa **eterogeneo**, con tipi diversi di nodo e
di arco.

Che questo sia il modo giusto di vedere il problema non è una tesi di questo
capitolo: è la definizione. Raccomandare **è** *link prediction* su un grafo
utente-oggetto, cioè prevedere gli archi mancanti, ed è il motivo per cui i
sistemi industriali su cataloghi enormi sono oggi costruiti così. Il caso più
noto, **PinSage** {cite}`ying2018graph`, è raccontato nel capitolo sulle reti
neurali su grafo insieme al campionamento dei vicini che lo rende praticabile
a scala web.

## Imparare a ordinare: BPR

Il vero salto concettuale della raccomandazione moderna non è architetturale
ma di *obiettivo*. Con il feedback implicito non ci sono voti da prevedere:
c'è solo l'elenco di ciò che hai guardato, e l'oceano di ciò che non hai
guardato, che, lo sappiamo dalla panoramica, non è un elenco di bocciature. La
**Bayesian Personalized Ranking** (BPR) {cite}`rendle2009bpr` prende sul serio
questa asimmetria: smette di prevedere valori e impara direttamente a
*ordinare*.

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

`````

`````{tab} Superiore

Sia $\hat{x}_{ui}$ il punteggio che un modello qualunque assegna alla coppia
$(u,i)$: nel paper è una fattorizzazione ridotta al solo prodotto scalare,
$\hat{x}_{ui} = P_u^\top Q_i$, senza nessuno dei termini additivi della
sezione precedente. Due dei tre spariscono da sé, perché BPR confronta sempre
due item *dello stesso* utente e nella differenza $\mu$ e $b_u$ si elidono. Il
bias di item no: sopravvive come $b_v - b_w$, e viene tolto per scelta, perché
è la stessa quantità per tutti gli utenti, cioè la parte *non* personalizzata
dell'ordinamento (la P di *Personalized*). Rimetterlo è legittimo e varie
implementazioni lo fanno, al prezzo di una classifica che per un utente senza
storia collassa su quella dei titoli più popolari.

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

In PyTorch la loss è una riga, e si innesta sul modello di fattorizzazione
della sezione precedente senza toccarlo:

```{code-block} python
:class: pt-non-eseguibile

import torch

def loss_bpr(x_uv, x_uw):
    # x_uv: punteggi (utente, item visto) · x_uw: (utente, item ignorato)
    return -torch.log(torch.sigmoid(x_uv - x_uw)).mean()

# nel ciclo di addestramento: v = item con cui l'utente ha interagito,
# w = item pescato a caso in tutto il catalogo. Ogni tanto capiterà un item
# che l'utente aveva già preso: succede così di rado che lo lasciamo passare
w = torch.randint(0, n_film, v.shape)          # campionamento dei negativi
loss = loss_bpr(modello(u, v), modello(u, w))  # stesso modello di prima
```

## Misurare una classifica

Se l'obiettivo è ordinare, anche il metro deve cambiare: l'errore quadratico
sui voti non dice nulla sulla qualità di una vetrina. Le metriche di ranking
guardano la lista dei primi $k$ suggerimenti, perché è l'unica cosa che
l'utente vedrà.

`````{tab} Elementare

Supponi che il sistema ti mostri 10 titoli, e che 3 ti sarebbero davvero
piaciuti. La **precision@10** è la frazione di consigli azzeccati:
$3/10 = 0{,}3$. Se i titoli che ti sarebbero piaciuti erano 6 in tutto nel
catalogo, il **recall@10** misura quanti ne ha ritrovati: $3/6 = 0{,}5$. Le
due metriche tirano in direzioni opposte: sparare consigli a raffica alza il
recall e affonda la precision.

C'è però un dettaglio che entrambe ignorano: *dove* stanno i colpi
azzeccati. Un successo al primo posto vale più di uno al decimo, perché al
decimo posto forse non arrivi mai. La **NDCG** è la metrica che ne tiene
conto: premia le classifiche che mettono i titoli giusti in cima, come un
giornale che sceglie bene la prima pagina.

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

dove $\mathrm{rel}_j$ è la rilevanza dell'item in posizione $j$ (binaria o
graduata) e $\mathrm{IDCG@}k$ è la DCG della classifica ideale, che normalizza
il punteggio tra utenti con numeri diversi di item rilevanti. Lo sconto
logaritmico penalizza dolcemente: la posizione 2 vale
$1/\log_2 3 \approx 0{,}63$ della posizione 1. Tutte queste metriche si
mediano sugli utenti; e tutte ereditano il difetto della valutazione offline:
misurano il recupero di interazioni passate, avvenute sotto l'esposizione del
vecchio sistema, non il gradimento futuro.

`````

## La storia recente conta

Un limite silenzioso di tutto ciò che abbiamo visto: la matrice dei voti non
ha orologio. Per la fattorizzazione, il film visto ieri sera e quello di dieci
anni fa pesano uguale. Ma chi ha appena comprato una tenda da campeggio è, per
qualche giorno, una persona diversa: sacco a pelo e fornelletto sono consigli
d'oro oggi e rumore tra un mese. La **raccomandazione sequenziale** tratta la
storia dell'utente come una frase da continuare: prevedere la prossima
interazione come si prevede la prossima parola. Non a caso il settore ha
seguito la stessa parabola del NLP: prima le reti ricorrenti, poi
l'auto-attenzione, con modelli come SASRec {cite}`kang2018self` e BERT4Rec
{cite}`sun2019bert4rec` che sono Transformer in tutto e per tutto, dove il
"vocabolario" è il catalogo. Gli strumenti li avete già visti nei capitoli sul
NLP e sui Transformer; qui cambia solo cosa c'è al posto delle parole.

## Come lo fa l'industria

Un'ultima dose di realismo. Nessuna piattaforma calcola un punteggio raffinato
per milioni di titoli a ogni visita: i sistemi reali lavorano **a due stadi**.
Il primo, il *retrieval*, screma il catalogo da milioni a qualche centinaio di
candidati con un modello volutamente semplice, lo schema dominante è la
**two-tower**: due reti separate producono l'una l'embedding dell'utente,
l'altra quello dell'item, e il punteggio è il loro prodotto scalare. La
separazione è il punto: gli embedding degli item si precalcolano tutti
offline, e a richiesta basta una ricerca dei vicini più prossimi
(approssimata) nello spazio degli embedding (il prodotto scalare "rigido"
della sezione precedente, riabilitato dall'efficienza). Il secondo stadio, il
*ranking*, applica ai soli sopravvissuti un modello ricco quanto si vuole, con
centinaia di feature di contesto (ora, dispositivo, storia recente). Questa
architettura, descritta pubblicamente dagli ingegneri di YouTube nel 2016
{cite}`covington2016deep`, è oggi lo standard di fatto, e gran parte del
lavoro vero, va detto, non è nel modello ma nell'infrastruttura che lo tiene
fresco.

## Suggerire o pilotare?

Chiudiamo con la domanda che questo capitolo si porta dietro fin dalla matrice
vuota: un sistema che decide cosa vedi, e impara da ciò che vedi, ti sta
*servendo* o ti sta *plasmando*? Nel 2011 l'attivista Eli Pariser ha dato un
nome alla paura: *filter bubble*, la bolla in cui l'algoritmo, inseguendo i
tuoi click, ti mostra sempre più di ciò che già pensi. La ricerca empirica
successiva ha restituito un quadro più sfumato (per molti utenti la
raccomandazione *allarga* il consumo rispetto al fai-da-te, e le bolle più
ermetiche spesso ce le costruiamo da soli) ma il meccanismo di fondo è reale,
ed è il feedback loop già incontrato: il modello impara da dati che il modello
stesso ha filtrato, come discusso nella sezione *Quando i dati cambiano* del
capitolo di Machine Learning.

Il punto critico non è la tecnica, è la metrica. Un sistema addestrato a
massimizzare i minuti di visione imparerà, con perfetta onestà matematica,
tutto ciò che trattiene: inclusi l'indignazione e il sensazionalismo, se
trattengono. Chi sceglie la funzione obiettivo sceglie, in ultima analisi, il
comportamento che il sistema coltiverà nei suoi utenti: è qui che passa il
confine tra suggerire e pilotare. Le contromisure esistono e sono concrete:
metriche di diversità e serendipità accanto all'accuratezza, controlli
espliciti nelle mani dell'utente, e da qualche anno anche la legge (in Europa
il Digital Services Act impone alle grandi piattaforme di offrire almeno una
versione del loro sistema di raccomandazione non basata sulla profilazione).
Nessuna di queste è una soluzione definitiva. Ma un ingegnere che sa *come*
funziona la macchina (e ora lo sapete) è esattamente la persona nella
posizione giusta per pretendere che funzioni bene.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Il **NCF** cambia il giudice: invece di confrontare le due schede voce per
  voce con una regola fissa, le incolla una sotto l'altra e lascia decidere a
  una piccola rete. In teoria vede combinazioni che il confronto voce per voce
  non coglie; alla prova dei fatti il vecchio confronto, tarato con cura, resta
  un avversario durissimo: più libertà non è gratis.
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
  prima pagina.
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
  resta un avversario durissimo (l'espressività non è gratis).
- La matrice di interazione **è** il grafo bipartito utente-oggetto, e
  raccomandare è **link prediction**: prevedere gli archi mancanti. Propagare
  sul grafo raccoglie segnale a più salti; **LightGCN** mostra che basta la
  propagazione, senza rete sopra.
- Con feedback implicito si impara a **ordinare**, non a prevedere voti:
  la loss **BPR** $-\log\sigma(\hat{x}_{uv}-\hat{x}_{uw})$ chiede solo che
  l'item scelto superi quello ignorato.
- Le classifiche si misurano con **precision@k**, **recall@k** e **NDCG**,
  che premia i successi in cima alla lista.
- I sistemi reali sono a **due stadi**: retrieval con two-tower e vicini
  approssimati, poi ranking fine sui candidati superstiti.
- La metrica scelta plasma il comportamento del sistema, e degli utenti: il
  confine tra suggerire e pilotare passa dalla funzione obiettivo.
```

`````
