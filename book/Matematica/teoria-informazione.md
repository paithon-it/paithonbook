# Teoria dell'informazione: misurare la sorpresa

Nel luglio del 1948, sulla rivista tecnica dei Bell Labs, un ingegnere
trentaduenne di nome Claude Shannon pubblica un articolo dal titolo
volutamente sobrio: *A Mathematical Theory of Communication*
{cite}`shannon1948mathematical`. Dentro c'è un'idea che cambierà il mondo più
di quanto il titolo lasci intuire: l'informazione si può **misurare**, con la
stessa oggettività con cui si misurano metri e chilogrammi. L'unità di misura
è il **bit**: contrazione di *binary digit*, parola che Shannon attribuisce al
collega John Tukey e che proprio in quell'articolo compare a stampa per la
prima volta.

In quell'articolo la grandezza centrale della teoria prende il nome che ha
ancora oggi, **entropia**, lo stesso della termodinamica: è il nome
dell'aneddoto raccontato in apertura di capitolo, quello che von Neumann
avrebbe suggerito a Shannon. Qui lo riempiamo di contenuto, perché ci riguarda
da vicino: il punteggio d'errore con cui addestreremo quasi tutti
i **classificatori** (cioè i modelli che devono scegliere fra
alternative: gatto o cane, spam o no) discende in linea diretta da
quell'articolo del 1948, e si chiama *cross-entropy*.

## La sorpresa di un evento

Il punto di partenza di Shannon è un'osservazione quasi banale: un messaggio
porta tanta più informazione quanto più è *improbabile*.

`````{tab} Elementare

Il telegiornale non apre mai con "domani il sole sorgerà": è certo, quindi non
è una notizia. Apre con la nevicata a Palermo, proprio perché è rara.
L'informazione è **sorpresa**: un evento scontato ne porta poca, un evento raro
ne porta molta.

Shannon trasformò l'intuizione in un numero, e il gioco delle venti domande fa
vedere come. Uno pensa a un oggetto, l'altro può chiedere solo cose con
risposta sì o no. Chi gioca bene sceglie domande che dimezzano ogni volta le
possibilità rimaste, e in venti mosse arriva a distinguere più di un milione di
oggetti, perché $2^{20} \approx 1{,}05$ milioni. Chi gioca male ("è un
carciofo?", "è un trapano?") non esaurisce nemmeno il contenuto di un cassetto.
Una domanda ben posta è l'unità di misura della sorpresa, e si chiama bit.

Il bit è l'esponente del due. Due alternative fanno $1$ bit perché $2^1 = 2$;
quattro ne fanno $2$ perché $2^2 = 4$; otto ne fanno $3$. L'esito di una moneta
equa (testa o croce, 50 e 50) vale quindi esattamente 1 bit: una domanda secca,
e la risposta è trovata.

Due monete lanciate insieme danno quattro esiti, cioè due domande: le
possibilità si moltiplicano, le domande si sommano. Un bit di sorpresa più un
bit di sorpresa fanno due bit, e questo vale per qualunque coppia di eventi che
non si influenzano a vicenda.

Un dado onesto ha sei facce, e sei sta in mezzo fra quattro e otto: fra $2$ e
$3$ bit. Il numero esatto è l'esponente che elevando $2$ dà $6$, cioè circa
$2{,}585$. In una partita singola nessuno fa due domande e mezzo; su tante
partite, invece, una strategia ottima ne consuma *in media* poco più di due e
mezzo, e sono i «2,6 bit» del dado.

Con una moneta sbilanciata il conto è lo stesso, con un'avvertenza: al posto
del numero di alternative si mette uno diviso la probabilità. Una moneta
truccata dà testa $9$ volte su $10$. La testa è "un'alternativa su
$1/0{,}9 = 1{,}11$", cioè quasi nessuna domanda, $0{,}15$ bit, e ce lo
aspettavamo. La croce è "una su $1/0{,}1 = 10$", cioè quanto un dado a dieci
facce, e vale $3{,}32$ bit: rara, e perciò molto informativa.

Quei due esponenti li dà una calcolatrice, ma controllarli si può a mano,
andando nel verso facile, cioè elevando il due. Se
$0{,}15$ è giusto, $2^{0{,}15}$ deve fare $1{,}11$, e infatti fa $1{,}11$. Se
$3{,}32$ è giusto, $2^{3{,}32}$ deve fare $10$: sta fra $2^3 = 8$ e
$2^4 = 16$, e viene $9{,}99$.

`````

`````{tab} Superiore

L’**autoinformazione** (o sorpresa) di un esito $x$ con probabilità $p(x)$ è

$$
I(x) = -\log_2 p(x),
$$

dove il segno meno rende la quantità positiva (i logaritmi di numeri fra $0$ e
$1$ sono negativi) e la base $2$ fissa l'unità di misura in bit. La forma
logaritmica non è un vezzo: è l'unica forma **continua** (a meno della base)
che rende la sorpresa **additiva** per eventi indipendenti; se
$p(x,y)=p(x)\,p(y)$, allora $I(x,y)=I(x)+I(y)$, perché il logaritmo trasforma i
prodotti in somme. L'aggettivo serve: l'equazione funzionale $f(pq)=f(p)+f(q)$
ha anche soluzioni patologiche, che però si costruiscono solo rinunciando a
ogni regolarità (continuità, monotonia o misurabilità).

Esempi: moneta equa, $I=-\log_2 0{,}5 = 1$ bit; una faccia del dado,
$I=-\log_2 \tfrac{1}{6} \approx 2{,}585$ bit; testa con la moneta truccata,
$I=-\log_2 0{,}9 \approx 0{,}152$ bit; croce con la stessa moneta,
$I=-\log_2 0{,}1 \approx 3{,}322$ bit. Con il logaritmo naturale al posto di
$\log_2$ l'unità si chiama *nat*: è la convenzione usata dalle loss di PyTorch.

`````

## L'entropia: la sorpresa media

Un singolo esito ha una sorpresa; una *sorgente* di esiti (una moneta, un
dado, una lingua) ha una sorpresa **media**. È l'entropia: quanto ci
aspettiamo di essere sorpresi, in media, a ogni estrazione
({numref}`fig-entropia-monete`).

```{figure} ../figures/entropia-monete.svg
:name: fig-entropia-monete
:alt: "Due monete identiche all'aspetto con le barre delle probabilità di testa e croce: la moneta equa, 50 e 50, ha entropia di 1 bit; quella truccata, 90 e 10, di circa 0,47 bit."
:width: 85%

Due monete identiche all'aspetto, entropie diverse: l'equa produce in media 1
bit di sorpresa per lancio, la truccata meno della metà. (La lettera $H$ che
compare nel disegno è il simbolo con cui si indica l'entropia, così come
$\pi$ indica il pi greco: «$H = 1$ bit» si legge «l'entropia di questa moneta
è di un bit».)
```

`````{tab} Elementare

Riprendiamo la moneta truccata: testa 9 volte su 10. Nove lanci su dieci la
sorpresa è quasi nulla (0,15 bit), una volta su dieci è grande (3,32 bit). La
media pesata fa $0{,}9 \times 0{,}15 + 0{,}1 \times 3{,}32 \approx 0{,}47$ bit
per lancio: meno della metà del bit pieno della moneta equa. Ha senso: una
moneta prevedibile ci sorprende poco, e infatti "produce" poca informazione.

La regola generale: l'entropia è **massima quando tutto è ugualmente
possibile** (massima incertezza: 1 bit per la moneta equa, circa 2,6 bit per il
dado onesto) e **scende verso zero** man mano che un esito diventa dominante.
Una moneta con due teste ha entropia zero: nessuna sorpresa, mai.

`````

`````{tab} Superiore

Per una distribuzione discreta $p=(p_1,\dots,p_n)$, l’**entropia** è il valore
atteso dell'autoinformazione:

$$
H(p) = -\sum_{i=1}^{n} p_i \log_2 p_i ,
$$

dove i termini con $p_i=0$ valgono $0$ per convenzione (coerente col limite
$p\log p \to 0$). Verifiche: moneta equa,
$H = -(0{,}5\log_2 0{,}5 + 0{,}5\log_2 0{,}5) = 1$ bit; moneta truccata con
$p=0{,}9$, $H \approx 0{,}9\cdot 0{,}152 + 0{,}1\cdot 3{,}322 \approx 0{,}469$
bit; dado equo, $H = \log_2 6 \approx 2{,}585$ bit.

Due proprietà strutturali: $H(p)\ge 0$, con uguaglianza solo per distribuzioni
degeneri (un esito certo); e $H(p)\le \log_2 n$, con uguaglianza solo per la
distribuzione uniforme. L'entropia è quindi una misura di *incertezza*: nulla
quando l'esito è scritto, massima quando le $n$ alternative sono equiprobabili.

Entrambe valgono nel **discreto**, ed è bene dirlo perché nel continuo la
prima cade. L'analogo per una densità, l’*entropia differenziale*
$h(f) = -\int f\log_2 f$, può essere negativo appena la densità si concentra:
$h(\mathcal{N}(0,1)) = +2{,}05$ bit, ma $h(\mathcal{N}(0,\,0{,}1^2)) =
-1{,}27$. La divergenza KL, invece, resta $\ge 0$ in entrambi i casi, ed è una
delle ragioni per cui è lei l'oggetto su cui si costruisce.

`````

## Confrontare distribuzioni: cross-entropia e divergenza KL

Fin qui una sola sorgente e una sola tabella di probabilità. Ma nel machine
learning ce ne sono sempre *due*, e conviene dire di quali si tratta. La prima
descrive come vanno le cose davvero: quanto spesso, nel mondo, esce ciascuna
risposta. Nessuno la conosce per intero (con la moneta truccata sì, perché
l'abbiamo truccata noi; con le foto di gatti no), ma esiste, e la chiamiamo
$p$. La seconda è quello che il **modello** crede: le probabilità che assegna
lui, e che sono sbagliate finché non impara. La chiamiamo $q$. Serve un modo
per misurare quanto la seconda sbaglia rispetto alla prima.

```{figure} ../figures/cross-entropy-kl-divergence.svg
:name: fig-cross-entropia-kl
:alt: "Due distribuzioni disegnate sugli stessi assi, sull'asse orizzontale le parole del vocabolario e sul verticale la probabilità: p, la realtà, e q, il modello, che le somiglia ma è spostata a destra e di forma diversa. Una parentesi in alto misura lo scarto fra i due picchi. Sulla coda sinistra un punto evidenziato segnala il caso peggiore, quello in cui la realtà assegna probabilità e il modello quasi nessuna. In basso la formula che lega le due misure."
:width: 88%

Le due curve e il fatto che non combaciano. La cross-entropia misura il costo
totale di descrivere $p$ usando $q$; la divergenza KL misura solo il
sovrapprezzo, cioè quanto si paga in più rispetto a conoscere $p$, ed è la
sottrazione scritta in fondo al disegno. Due avvertenze sul resto. Lo spazio
bianco fra le due curve è un promemoria visivo e non la misura: le due misure
vere sono definite fra poche righe. E il punto segnato sulla coda di sinistra è
il caso che costa di più, quello in cui la realtà ogni tanto produce una parola
e il modello le aveva dato quasi zero: essere colti di sorpresa lì è la cosa
più cara che possa capitare, ed è per questo che la scritta accanto dice che il
conto «esplode».
```

La distinzione che {numref}`fig-cross-entropia-kl` rende visiva spiega perché
in pratica si minimizzi la cross-entropia e non la KL. Le due quantità
differiscono per una sola cosa, la sorpresa media della realtà $p$, che
dipende dai dati e non da chi li prevede: è la stessa qualunque modello si
usi. Spingere in basso l'una o l'altra porta quindi esattamente allo stesso
modello, e la cross-entropia ha il vantaggio di potersi **stimare dai soli
esempi**, senza mai scrivere $p$.

A prima vista sembra impossibile, visto che nella definizione la $p$ c'è. Il
punto è che non serve la tabella completa delle probabilità vere: bastano gli
esiti veri, uno alla volta. Ogni foto etichettata «gatto» è la realtà che si
presenta e dice «stavolta è toccato a me», e facendo la media della sorpresa
del modello su tutte le foto che si hanno, la $p$ entra nel conto da sé, senza
che nessuno l'abbia mai scritta.

La KL, invece, quella tabella la vorrebbe davvero, perché al suo interno c'è la
sorpresa media della realtà, che dagli esempi non si ricava. Ed è la ragione
per cui, dovendo sceglierne una, si minimizza la cross-entropia. È poi la
situazione in cui ci si trova sempre: gli esempi si hanno, la legge che li ha
prodotti no.

`````{tab} Elementare

Il codice Morse assegna il segnale più corto (un punto) alla E, che in inglese
è la lettera più frequente: le scorciatoie migliori vanno alle cose più
comuni, così i messaggi restano brevi. Ora immagina di telegrafare in italiano
usando il Morse *tarato sull'inglese*: funziona, ogni lettera ha il suo
codice, ma le frequenze delle lettere sono diverse e ogni tanto una lettera
comune da noi si porta dietro un codice lungo. In media, **sprechi**.

La **cross-entropia** è la lunghezza media dei tuoi messaggi quando usi il
codice pensato per la lingua sbagliata: le lettere arrivano secondo la realtà
($p$), le scorciatoie sono ottimizzate per la convinzione del modello ($q$).
La **divergenza di Kullback–Leibler** è lo spreco puro: i bit pagati in più
rispetto al codice giusto. È zero solo se $q$ indovina esattamente $p$, e non
è simmetrica: sbagliare codice in un verso non costa quanto sbagliarlo
nell'altro.

`````

`````{tab} Superiore

La **cross-entropia** fra la distribuzione vera $p$ e quella del modello $q$ è

$$
H(p,q) = -\sum_i p_i \log_2 q_i ,
$$

cioè la sorpresa media che *proviamo* usando le probabilità sbagliate $q$
mentre gli esiti escono secondo $p$. La **divergenza di Kullback–Leibler**
{cite}`kullback1951information` è l'eccesso rispetto al minimo possibile:

$$
D_{KL}(p\,\|\,q) = H(p,q) - H(p) = \sum_i p_i \log_2 \frac{p_i}{q_i} \;\ge\; 0,
$$

dove la disuguaglianza (di Gibbs) vale sempre, con uguaglianza se e solo se
$p=q$. Esempio con le nostre monete: se la realtà è la moneta truccata
($p = (0{,}9;\, 0{,}1)$) e il modello la crede equa, $H(p,q)=1$ bit e
$D_{KL} = 1 - 0{,}469 \approx 0{,}531$ bit. Nel verso opposto (realtà equa,
modello convinto del trucco) $H(p,q) \approx 1{,}737$ bit e
$D_{KL} \approx 0{,}737$ bit. I due valori differiscono: la KL è
**asimmetrica**, $D_{KL}(p\,\|\,q) \ne D_{KL}(q\,\|\,p)$ in generale, e non
soddisfa la disuguaglianza triangolare. Non è una distanza in senso
matematico, per quanto la si usi come misura di dissimilarità.

`````

## Il ponte con l'apprendimento

Ed ecco il motivo per cui l'entropia sta in un libro di machine learning.

`````{tab} Elementare

Quando una rete neurale impara a classificare, a ogni esempio le si fa una
sola domanda: *quanto ti sorprende la risposta giusta?* Se il modello dava al
gatto il 90% di probabilità e l'immagine era davvero un gatto, la sorpresa è
piccola e la correzione minima; se gli dava il 2%, la sorpresa è enorme e la
correzione energica. Addestrare significa girare le manopole dei parametri per
rendere la risposta giusta sempre meno sorprendente. La "punizione" media è
esattamente la cross-entropia dell'analogia del Morse: il modello smette di
sprecare quando il suo codice (le sue probabilità) combacia con la realtà.

Smettere di sprecare, però, non vuol dire arrivare a costo zero: anche col
codice giusto i telegrammi hanno una lunghezza. Se una foto sfocata può essere
gatto o cane, nessuna manopola rende certa la risposta; quella sorpresa che
resta è l'incertezza dei dati stessi, e la paga anche il modello perfetto. La
stessa manovra ha infine un terzo nome: scegliere i parametri sotto i quali
gli esempi raccolti risultano i più plausibili. Sorprendersi poco della
risposta giusta e trovare plausibile quello che è successo sono la stessa
regolazione delle manopole, vista da due lati.

`````

`````{tab} Superiore

Minimizzare la cross-entropia rispetto ai parametri $\theta$ del modello
equivale a minimizzare la divergenza KL, perché

$$
H(p, q_\theta) = H(p) + D_{KL}(p\,\|\,q_\theta)
$$

e $H(p)$ non dipende da $\theta$: il minimo teorico della loss non è zero ma
l'entropia dei dati, la loro incertezza irriducibile.

Attenzione però a **quale** $p$, perché lo stesso simbolo (qui come
dappertutto) copre due cose diverse. Se $p$ è la distribuzione condizionata vera del
processo che genera i dati, il pavimento è $H(p) > 0$ e nessun modello scende
sotto. Se invece $p$ è il bersaglio empirico di un singolo esempio, cioè
«questa immagine è un gatto» con probabilità $1$ e tutto il resto a zero,
allora $H(p) = 0$ e il pavimento è zero. Le due affermazioni convivono e
spiegano una cosa che si osserva addestrando: la loss di *training* può
scendere quasi a zero, quella di *validazione* no, perché la prima misura la
distanza da bersagli certi e la seconda da una distribuzione che certa non è.

Inoltre, sulla
distribuzione empirica del training set la cross-entropia coincide con la
log-verosimiglianza negativa media: minimizzarla *è* la stima di massima
verosimiglianza vista nella sezione su probabilità e statistica. Le tre
prospettive (minimizzare la cross-entropia, avvicinare $q_\theta$ a $p$ nel
senso della KL, massimizzare la verosimiglianza) sono la stessa operazione. È
ciò che fa `nn.CrossEntropyLoss`, la loss $\mathcal{L}=-\log \hat{y}_c$ (dove
$\hat{y}_c$ è la probabilità che il modello assegna alla classe corretta $c$)
che useremo nei capitoli sulle reti neurali e su PyTorch.

`````

## La perplessità: quante facce ha il dado

Dall'entropia si ricava una misura più parlante, cara a chi costruisce modelli
di linguaggio.

```{figure} ../figures/perplessita-righello.svg
:name: fig-perplessita-righello
:alt: "Due righelli paralleli e allineati: in alto le facce del dado equivalente, con le tacche a 1, 2, 4, 8 e 16; in basso i bit di sorpresa media, con le tacche a 0, 1, 2, 3 e 4. Le tacche cadono negli stessi punti perché raddoppiare le facce costa un bit. Quattro linee verticali tratteggiate collegano i due righelli e segnano quattro sorgenti: la moneta truccata a 0,47 bit e 1,4 facce, la moneta equa a 1 bit e 2 facce, il dado onesto a 2,59 bit e 6 facce, un modello di linguaggio di perplessità 20 a 4,32 bit e 20 facce."
:width: 96%

Un righello solo, con due scritte diverse sui due bordi. La perplessità non
aggiunge niente all'entropia: la dice in facce invece che in bit, e le facce
raddoppiano dove i bit crescono di uno.
```

Il salto alla perplessità è una riscrittura, non un concetto nuovo: si torna
indietro dall'esponente al numero di alternative. Per la moneta equa $2^1 = 2$
facce, per quella truccata $2^{0{,}47} \approx 1{,}4$, e in
{numref}`fig-perplessita-righello` sono lo stesso punto letto sui due bordi.

Quel secondo conto merita una riga, perché «due elevato a zero virgola
quarantasette» non è più «due moltiplicato per sé stesso un certo numero di
volte»: l'elevamento a potenza si estende agli esponenti con la virgola in
modo che continui a valere la regola di sempre, cioè che sommando gli
esponenti si moltiplichino i risultati. Con quella regola $2^{0{,}5}$ deve
essere il numero che moltiplicato per sé stesso dà $2$, cioè $\sqrt 2 \approx
1{,}41$; e $2^{0{,}47}$, di pochissimo più piccolo, vale circa $1{,}4$. Non
esiste un dado con $1{,}4$ facce, e non serve: il numero dice «meno di due
alternative vere», cioè che quella moneta è poco più che decisa.

`````{tab} Elementare

Dire "entropia 2,585 bit" non è intuitivo; dire "è incerto come un dado a sei
facce" sì. La **perplessità** fa proprio questa traduzione: riconverte
l'entropia nel *numero di alternative ugualmente probabili* che darebbero la
stessa incertezza. Moneta equa: perplessità 2. Dado onesto: 6. La moneta
truccata: circa 1,4 (quasi nessun dubbio, poco più di un'alternativa secca).
Quando leggerai che un modello di linguaggio "ha perplessità 20", ora sai cosa
significa: a ogni parola è incerto *come se* tirasse un dado a 20 facce.

`````

`````{tab} Superiore

La **perplessità** di una distribuzione è

$$
\mathrm{PP}(p) = 2^{H(p)},
$$

l'esponenziale dell'entropia nella stessa base del logaritmo. Per la
distribuzione uniforme su $n$ esiti, $\mathrm{PP} = 2^{\log_2 n} = n$: il
numero di alternative, appunto. Per le nostre sorgenti:
$2^{1}=2$ (moneta equa), $2^{\log_2 6}=6$ (dado),
$2^{0{,}469}\approx 1{,}38$ (moneta truccata). Nei modelli di linguaggio si usa
la perplessità *per parola*, calcolata sulla cross-entropia media del modello
su un testo di test: la riprenderemo, numeri alla mano, nel capitolo sul
Natural Language Processing.

`````

## Il limite della compressione

Chiudiamo con la conseguenza più concreta del lavoro di Shannon: l'entropia è
un **limite alla compressione**. Comprimere un file, come fa un programma tipo
`zip` o `gzip`, vuol dire riscriverlo più corto in modo da poterlo poi
ricostruire identico. Shannon dimostrò che quel «più corto» ha un fondo:
nessun programma, per quanto ingegnoso, può scendere sotto l’**entropia per
simbolo** del messaggio, cioè sotto la sorpresa media che ogni carattere porta
con sé. In media, sotto quella soglia non si scende.

L'aggettivo «per simbolo» non è un dettaglio, ed è il punto in cui la frase
detta male diventa falsa. La sorpresa media $H$ calcolata sulle sole frequenze
delle lettere descrive una sorgente **senza memoria**, una che estrae ogni
lettera indipendentemente dalle precedenti. Una lingua non è così: dopo una
«q» arriva quasi sempre una «u», dopo «il gatto ne» le continuazioni plausibili
sono poche. Per una sorgente con memoria il limite vero è più basso, ed è la
sorpresa media di ogni lettera **dato tutto ciò che la precede**.

La differenza si tocca con mano, e chiunque può rifare il conto su un testo
che ha già. Prendendo i file di
testo con cui questo libro è scritto (sei megabyte abbondanti) e contando
soltanto quanto è frequente ciascun carattere, la sorpresa media viene circa
$4{,}7$ bit a carattere. Poi si passa il tutto a `gzip`, che è il compressore
più ordinario che ci sia, e il file esce a circa $2{,}9$ bit a carattere, cioè
a poco più del $60\%$ di quel presunto limite invalicabile. Non ha violato
nessun teorema: sta sfruttando proprio la ridondanza fra un carattere e il
successivo, che quel conto ignorava.

È la stessa quantità che Shannon stimò nel 1951 per l'inglese scritto in circa
**un bit per lettera** {cite}`shannon1951prediction`, e va confrontata con i
quasi $5$ bit che darebbero ventisei lettere equiprobabili tenendo conto solo
di quante sono. Uno zip morde bene un testo perché quella ridondanza c'è tutta;
non morde più niente su un file già compresso, dove è già stata spremuta via.

Comprimere è l'arte del Morse portata al suo limite matematico: scorciatoie a
ciò che è frequente. E qui si chiude il cerchio con il machine learning, in un
passaggio da fare per esteso. Un compressore ha bisogno di sapere che cosa è
frequente, per dare a quello le scorciatoie. Un modello che predice bene sa
esattamente questo, anzi qualcosa di più: sa che cosa è frequente *proprio
lì*, dopo le parole appena lette. Chi ha un modello così può scrivere il
messaggio in un modo diverso e più corto: invece del testo, le sorprese, e
dove il modello indovina la sorpresa è quasi zero, quindi non c'è quasi niente
da scrivere. Predire e comprimere, ci dice Shannon, sono in fondo la stessa
cosa.

## In pratica, con NumPy

```python
import numpy as np

def entropia(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]                        # convenzione: 0·log 0 = 0
    return -(p * np.log2(p)).sum()

equa     = [0.5, 0.5]
truccata = [0.9, 0.1]
dado     = np.full(6, 1/6)

print(entropia(equa))                   # 1.0
print(entropia(truccata))               # ~0.4690
print(entropia(dado))                   # ~2.5850

def cross_entropia(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    m = p > 0
    return -(p[m] * np.log2(q[m])).sum()

# la realta' e' truccata, il modello crede la moneta equa
H_pq = cross_entropia(truccata, equa)   # 1.0
kl   = H_pq - entropia(truccata)        # ~0.5310: lo "spreco" in bit
print(H_pq, kl)

# perplessita': 2^H, il numero di alternative equiprobabili
print(2**entropia(equa), 2**entropia(dado))   # 2.0  6.0
```

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- L'informazione è **sorpresa**: una notizia scontata (domani sorge il sole)
  non informa, una rara sì. Si misura in **bit**, e un bit è una domanda ben
  posta, con risposta sì o no.
- L’**entropia** è la sorpresa media di una sorgente: 1 bit a lancio per la
  moneta equa, circa 0,47 per quella truccata che dà testa nove volte su dieci,
  circa 2,585 per il dado a sei facce. Massima quando tutti gli esiti sono
  ugualmente possibili, nulla quando l'esito è già deciso in partenza.
- La **cross-entropia** è quanto costa scrivere i messaggi con il codice
  sbagliato (il Morse tarato sull'inglese, usato per l'italiano); i bit pagati
  in più rispetto al codice giusto, cioè lo spreco puro, sono la **divergenza
  di Kullback–Leibler**: mai negativa, zero solo se il modello indovina la
  realtà, e diversa a seconda del verso in cui si sbaglia (perciò non è una
  distanza).
- Addestrare un classificatore rendendo la risposta giusta sempre meno
  sorprendente, avvicinare le credenze del modello alla realtà e scegliere i
  parametri che rendono i dati più plausibili sono tre nomi per la stessa
  operazione.
- La **perplessità** traduce l'entropia in facce del dado: quante alternative
  ugualmente probabili darebbero la stessa incertezza (2 per la moneta equa, 6
  per il dado). La ritroveremo nei modelli di linguaggio.
- Comprimere senza perdere niente ha un limite, ed è l'entropia **per
  simbolo** della sorgente: quanta sorpresa porta in media ogni pezzo di
  messaggio, tenuto conto di tutto quello che lo precede. È molto meno di
  quanto direbbero le sole frequenze delle lettere, ed è la ragione per cui
  uno zip su un testo fa meglio di quel conto ingenuo, e non fa niente su un
  file già compresso.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- L'informazione è **sorpresa**: un esito di probabilità $p$ vale $-\log_2 p$
  bit; tanto più, quanto più è raro.
- L’**entropia** $H(p)=-\sum_i p_i \log_2 p_i$ è la sorpresa media: 1 bit per
  la moneta equa, 0,47 per quella truccata, 2,585 per il dado. Massima
  sull'uniforme, nulla sul certo.
- La **cross-entropia** $H(p,q)$ è il costo di usare il "codice" sbagliato; la
  **divergenza KL** $D_{KL}(p\,\|\,q)=H(p,q)-H(p)\ge 0$ è lo spreco puro.
  Asimmetrica: non è una distanza.
- Minimizzare la cross-entropy come loss = minimizzare la KL fra dati e
  modello = massima verosimiglianza: tre nomi per la stessa operazione.
- La **perplessità** $2^{H}$ traduce l'entropia in "facce del dado": la
  ritroveremo nei modelli di linguaggio.
- Il limite della compressione senza perdite è l’**entropia per simbolo**
  (*entropy rate*) $\lim_n H(X_1,\dots,X_n)/n$, non la $H$ di ordine zero
  calcolata sulle frequenze marginali: per una sorgente con memoria la seconda
  sovrastima largamente, e un compressore generico la scavalca senza
  contraddire Shannon.
```
`````
