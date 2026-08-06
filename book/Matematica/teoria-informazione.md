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

Alla grandezza centrale della teoria Shannon diede il nome **entropia**, lo
stesso della termodinamica. Si racconta (la fonte è lo stesso Shannon, in una
conversazione riferita molti anni dopo, quindi va presa con la cautela che
meritano gli aneddoti) che a suggerirglielo fu John von Neumann: "chiamala
entropia: nessuno sa davvero cosa sia, e in ogni discussione partirai in
vantaggio". Vera o abbellita che sia la battuta, il nome è rimasto. E ci
riguarda da vicino: la funzione di costo con cui addestreremo quasi tutti i
classificatori di questo libro (la *cross-entropy*) discende in linea diretta
da quell'articolo del 1948.

## La sorpresa di un evento

Il punto di partenza di Shannon è un'osservazione quasi banale: un messaggio
porta tanta più informazione quanto più è *improbabile*.

`````{tab} Elementare

Il telegiornale non apre mai con "domani il sole sorgerà": è certo, quindi non
è una notizia. Apre con la nevicata a Palermo, proprio perché è rara.
L'informazione, insomma, è **sorpresa**: un evento scontato ne porta poca, un
evento raro ne porta molta.

Shannon trasformò l'intuizione in un numero. L'esito di una moneta equa (testa
o croce, 50 e 50) vale esattamente **1 bit**: è la sorpresa di una domanda
secca con due risposte ugualmente possibili. L'esito di un dado a sei facce
sorprende di più (le alternative erano sei, non due): circa 2,6 bit. E sapere
che una moneta truccata, che dà testa 9 volte su 10, ha dato testa? Quasi
niente: 0,15 bit. Ce lo aspettavamo già.

Se l'idea di "misurare in domande" sembra astratta, pensa al gioco delle
**venti domande**: uno pensa a un oggetto, l'altro può chiedere solo cose con
risposta sì o no. Giocando bene (ogni domanda dimezza le possibilità rimaste)
venti domande bastano a distinguere fra più di un milione di oggetti, perché
$2^{20} \approx 1{,}05$ milioni. Giocando male ("è un carciofo?", "è un
trapano?") non bastano nemmeno per il contenuto di un cassetto. Un bit è
esattamente una domanda ben posta, e l'entropia conterà quante ne servono in
media.

`````

`````{tab} Superiore

L'**autoinformazione** (o sorpresa) di un esito $x$ con probabilità $p(x)$ è

$$
I(x) = -\log_2 p(x),
$$

dove il segno meno rende la quantità positiva (i logaritmi di numeri fra $0$ e
$1$ sono negativi) e la base $2$ fissa l'unità di misura in bit. La forma
logaritmica non è un vezzo: è l'unica (a meno della base) che rende la
sorpresa **additiva** per eventi indipendenti; se $p(x,y)=p(x)\,p(y)$, allora
$I(x,y)=I(x)+I(y)$, perché il logaritmo trasforma i prodotti in somme.

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
bit di sorpresa per lancio, la truccata meno della metà.
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

Per una distribuzione discreta $p=(p_1,\dots,p_n)$, l'**entropia** è il valore
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

`````

## Confrontare distribuzioni: cross-entropia e divergenza KL

Fin qui una sola distribuzione. Ma nel machine learning ce ne sono sempre
*due*: la distribuzione **vera** dei dati, che chiamiamo $p$, e quella che il
**modello** crede vera, che chiamiamo $q$. Serve un modo per misurare quanto la
seconda sbaglia rispetto alla prima.

```{figure} ../figures/cross-entropy-kl-divergence.svg
:name: fig-cross-entropia-kl
:alt: "Due distribuzioni disegnate sugli stessi assi: p, la realtà, e q, il modello, che le somiglia ma è spostata e di forma diversa. L'area fra le due curve rappresenta lo scarto, cioè quanto costa in più descrivere i dati usando le credenze del modello invece della distribuzione vera."
:width: 88%

Le due curve e ciò che le separa. La cross-entropia misura il costo totale di
descrivere $p$ usando $q$; la divergenza KL misura solo il sovrapprezzo, cioè
quanto si paga in più rispetto a conoscere $p$.
```

La distinzione che {numref}`fig-cross-entropia-kl` rende visiva spiega perché
in pratica si minimizzi la cross-entropia e non la KL. Le due differiscono per
l'entropia di $p$, che è una proprietà dei dati e non dipende dal modello:
minimizzare l'una o l'altra porta agli stessi pesi, ma la prima non richiede
di conoscere $p$.

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

Ed ecco il motivo per cui questa sezione sta in un libro di machine learning.

`````{tab} Elementare

Quando una rete neurale impara a classificare, a ogni esempio le si fa una
sola domanda: *quanto ti sorprende la risposta giusta?* Se il modello dava al
gatto il 90% di probabilità e l'immagine era davvero un gatto, la sorpresa è
piccola e la correzione minima; se gli dava il 2%, la sorpresa è enorme e la
correzione energica. Addestrare significa girare le manopole dei parametri per
rendere la risposta giusta sempre meno sorprendente. La "punizione" media è
esattamente la cross-entropia dell'analogia del Morse: il modello smette di
sprecare quando il suo codice (le sue probabilità) combacia con la realtà.

`````

`````{tab} Superiore

Minimizzare la cross-entropia rispetto ai parametri $\theta$ del modello
equivale a minimizzare la divergenza KL, perché

$$
H(p, q_\theta) = H(p) + D_{KL}(p\,\|\,q_\theta)
$$

e $H(p)$ non dipende da $\theta$: il minimo teorico della loss non è zero ma
l'entropia dei dati, la loro incertezza irriducibile. Inoltre, sulla
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

```{figure} ../figures/entropia-di-shannon.svg
:name: fig-entropia-due-monete
:alt: "Due monete a confronto. Quella equa, con probabilità 50 e 50, ha entropia pari a 1 bit: il massimo possibile per due esiti. Quella truccata, 90 contro 10, ha entropia 0,47 bit: sapendo che esce quasi sempre testa, ogni lancio informa meno della metà."
:width: 92%

L'entropia misura quanto c'è da imparare. Una moneta prevedibile porta poca
informazione a ogni lancio, e la perplessità traduce quel numero in «quante
facce ha il dado equivalente».
```

Il salto da {numref}`fig-entropia-due-monete` alla perplessità è una
riscrittura, non un concetto nuovo: $2^1 = 2$ facce per la moneta equa, $2^{0,47}
\approx 1{,}4$ per quella truccata. Il secondo numero dice, in modo più
intuitivo del primo, che quella moneta è poco più che decisa.

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
un **limite alla compressione**. Nessun programma, per quanto ingegnoso, può
comprimere senza perdite una sorgente sotto la sua entropia: in media, sotto
$H$ bit per simbolo non si scende. È il motivo per cui uno zip morde bene un
file di testo, le lingue sono ridondanti: nel 1951 lo stesso Shannon stimò per
l'inglese scritto circa un bit per lettera, contro i quasi $5$ di lettere
equiprobabili {cite}`shannon1951prediction`, e non riesce a comprimere un file
già compresso, dove la ridondanza è già stata spremuta. Comprimere è l'arte
del Morse portata al suo limite matematico: scorciatoie a ciò che è frequente.
E un modello che predice bene, assegnando poca sorpresa alla realtà, è per ciò
stesso un buon compressore: predire e comprimere, ci dice Shannon, sono in
fondo la stessa cosa.

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

```{admonition} Da ricordare
:class: important
- L'informazione è **sorpresa**: un esito di probabilità $p$ vale $-\log_2 p$
  bit; tanto più, quanto più è raro.
- L'**entropia** $H(p)=-\sum_i p_i \log_2 p_i$ è la sorpresa media: 1 bit per
  la moneta equa, 0,47 per quella truccata, 2,585 per il dado. Massima
  sull'uniforme, nulla sul certo.
- La **cross-entropia** $H(p,q)$ è il costo di usare il "codice" sbagliato; la
  **divergenza KL** $D_{KL}(p\,\|\,q)=H(p,q)-H(p)\ge 0$ è lo spreco puro.
  Asimmetrica: non è una distanza.
- Minimizzare la cross-entropy come loss = minimizzare la KL fra dati e
  modello = massima verosimiglianza: tre nomi per la stessa operazione.
- La **perplessità** $2^{H}$ traduce l'entropia in "facce del dado": la
  ritroveremo nei modelli di linguaggio. E l'entropia è il limite invalicabile
  della compressione senza perdite.
```
