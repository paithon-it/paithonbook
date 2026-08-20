# Analisi e ottimizzazione: derivate e discesa del gradiente

Nebbia fitta, nessuna mappa, e non si vede a un metro di distanza. Un'informazione, però, ce l'hai sempre: la
pendenza del terreno sotto i piedi. Ti basta sentire da che parte scende, fare
un passo in quella direzione, rimisurare e ripetere. Addestrare un modello è
esattamente questo. La collina da scendere è la funzione che misura *quanto il
modello sbaglia* (il **costo** o **loss**, che indichiamo con $\mathcal{L}$) e
lo strumento che sente la pendenza sotto i piedi è la **derivata**. 

## La derivata: la pendenza istante per istante

Una **funzione** è una regola che, dato un numero in ingresso, ne restituisce
uno in uscita, sempre lo stesso a parità di ingresso: «raddoppia» è una
funzione, «il prezzo del biglietto per un viaggio di tanti chilometri» è una
funzione, e anche «di quanto sbaglia questo modello, se le sue manopole sono
regolate così» è una funzione. Disegnarla si può: si mette l'ingresso
sull'asse orizzontale e l'uscita su quello verticale, e l'insieme dei punti
che ne viene fuori è il **grafico**, di solito una curva che sale e scende. La
derivata risponde a una domanda sola: *se muovo l'ingresso di un pelo, di
quanto cambia l'uscita?*

`````{tab} Elementare

La posizione di un'auto cambia nel tempo, e la velocità è "quanto in fretta"
cambia. Il tachimetro, quindi, sta già mostrando una derivata: la derivata
della posizione.

Sul grafico si vede ancora meglio. Appoggia un righello alla curva in un
punto e giralo finché, lì attorno, righello e curva si sovrappongono: quella è
la retta **tangente**, e la sua inclinazione (di quanto sale ogni volta che si
avanza di un passo verso destra) è la derivata lì. Dove la curva sale ripida il
righello è ripido e la derivata è grande e positiva; dove scende, il righello
punta in giù e la derivata è negativa; in cima a una gobba o in fondo a una
conca, dove per un istante il terreno è piatto, il righello è orizzontale e la
derivata vale **zero**.

`````

`````{tab} Superiore

La derivata di $f$ in $x$ è il limite del rapporto incrementale:

$$
f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}.
$$

Misura la pendenza della retta tangente al grafico in $x$. I punti in cui
$f'(x) = 0$ si dicono **stazionari**: massimi, minimi o flessi a tangente
orizzontale (in più variabili, punti di sella). Sono esattamente i candidati
che cerchiamo quando vogliamo minimizzare una loss.

`````

## Le derivate che tornano di continuo

Nessuno, in pratica, va a misurare la pendenza punto per punto: per le funzioni
che si incontrano di solito la derivata si ricava da poche regole, imparate una
volta e riusate sempre, come le tabelline. E tre famiglie di funzioni compaiono
ovunque nel machine learning.

`````{tab} Elementare

Le tre "solite sospette" sono le **potenze**, l’**esponenziale** e il
**logaritmo**. Le prime le conosci già: la parabola $x^2$, cioè «il numero
moltiplicato per sé stesso», è la forma dell'errore quadratico, quello che si
minimizza quando il modello deve prevedere un numero. Le altre due meritano
una presentazione, perché ritornano in tre sezioni di questo capitolo e poi in
mezzo libro.

**Il logaritmo è la domanda inversa dell'elevamento a potenza.** Se
l'elevamento chiede «quanto fa $2^3$, cioè $2\times2\times2$?» (risposta:
$8$), il logaritmo chiede la stessa cosa dall'altro capo: «quanti $2$ devo
moltiplicare fra loro per arrivare a $8$?», e la risposta è $3$. Si scrive
$\log_2 8 = 3$, e quel $2$ in basso si chiama *base*. Con la base $10$ è
ancora più immediato: $\log_{10} 1000 = 3$, perché $1000$ è
$10\times10\times10$. Due
cose lo rendono prezioso, e sono le sole che servono qui. La prima: **schiaccia
i numeri enormi**. Fra $1000$ e $1\,000\,000$ ci sono novecentonovantanovemila
unità di differenza, ma fra i loro logaritmi ce ne sono tre: è il motivo per
cui le scale dei terremoti e del volume sonoro sono logaritmiche, e per cui
sono comode quando i numeri in gioco vanno da un milionesimo a un miliardo. La
seconda: **trasforma le moltiplicazioni in somme**, perché moltiplicare due
potenze vuol dire sommarne gli esponenti. Moltiplicare fra loro mille
probabilità piccolissime dà un numero che nessun calcolatore riesce a
scrivere; sommare i loro logaritmi no, e il risultato è lo stesso a meno di
tradurlo indietro.

**L'esponenziale è il logaritmo letto al contrario**, e $e^x$ è il caso
particolare che si usa quasi sempre. Quel simbolo $e$ non è una variabile: è
un numero fisso, $e \approx 2{,}718$, come $\pi$ vale $3{,}14$. Vale la pena
sapere da dove salta fuori, perché non è arbitrario: fra tutte le curve di
crescita, $e^x$ è quella che in ogni punto **cresce esattamente quanto vale**.
Se in un certo istante vale $5$, sta salendo con pendenza $5$; quando vale
$10$, sale il doppio più in fretta. Da qui viene la proprietà per cui la si
usa: in intervalli di tempo uguali una crescita così non aggiunge ogni volta
la stessa quantità, la **moltiplica** ogni volta per lo stesso fattore. È il
comportamento degli interessi composti, della crescita di una popolazione e,
letto al contrario, del decadimento di una sostanza. Nel libro compare dentro due funzioni che incontrerai presto, la
**sigmoide** e la **softmax**: due ricette che prendono i punteggi grezzi
sputati da un modello (numeri qualsiasi, anche negativi) e li rimettono in
riga come probabilità, cioè numeri fra zero e uno che sommati fanno uno. Il
logaritmo, dal canto suo, compare nella **cross-entropy**, la funzione di
costo con cui si addestrano i classificatori, di cui parla per esteso la
sezione sulla teoria dell'informazione.

Restano le derivate. Non si dimostrano qui, si prendono da una tabella, come
si prende la formula dell'area del cerchio: la pendenza di $x^2$ è $2x$ (nel
punto $x=3$ la parabola sale con pendenza $6$), e quella di $e^x$ è di nuovo
$e^x$, che è poi la proprietà appena raccontata detta in simboli. È anche il
motivo per cui l'esponenziale rende i conti sopportabili: derivandola, resta
identica a sé stessa.

`````

`````{tab} Superiore

Le tre regole che useremo senza più pensarci:

$$
\frac{d}{dx}\,x^n = n\,x^{n-1}, \qquad
\frac{d}{dx}\,e^{x} = e^{x}, \qquad
\frac{d}{dx}\,\ln x = \frac{1}{x}.
$$

Non è un caso che ricorrano proprio queste: l'errore quadratico medio è una
potenza, la sigmoide $\sigma(x)=1/(1+e^{-x})$ e la softmax sono costruite
sull'esponenziale, la log-verosimiglianza e la cross-entropy sul logaritmo.
La stabilità di $e^x$ sotto derivazione è ciò che rende quei conti trattabili.

`````

## Dal singolo numero al gradiente

Un modello reale non ha un parametro solo, ne ha milioni, e la loss dipende da
tutti insieme. Con due parametri il costo non è più una curva ma una
superficie, un paesaggio di colline e conche in cui ogni punto del terreno è
una coppia di regolazioni e la quota è l'errore che ne viene fuori. Un modo
comodo di disegnare un paesaggio su un foglio è quello delle carte
escursionistiche: guardarlo dall'alto e tracciare le **curve di livello**,
cioè le linee che uniscono i punti alla stessa quota. Dove le linee sono
fitte, il terreno è ripido; dove sono larghe, è pianeggiante.

```{figure} ../figures/derivate-gradiente.svg
:name: fig-curve-di-livello
:alt: "Una superficie di loss vista dall'alto, disegnata come curve di livello concentriche attorno a un minimo. Sopra di essa il percorso della discesa del gradiente: una successione di passi che in ogni punto imbocca la direzione perpendicolare alla curva di livello, cioè quella di massima pendenza, avvicinandosi progressivamente al centro."
:width: 88%

Il paesaggio del costo visto dall'alto, come in una carta escursionistica:
ogni anello unisce i punti in cui il modello sbaglia allo stesso modo, e il
centro è il fondo della conca. Il **gradiente** disegnato sopra è la freccia
che in ogni punto indica dove il terreno sale più ripido (nel disegno compare
col segno meno, $-\nabla\mathcal{L}$, perché per scendere si va nel verso
opposto: il triangolino capovolto $\nabla$ è il simbolo che lo indica, si legge
«nabla»). Quella freccia taglia sempre ad angolo retto l'anello su cui si
trova.
```

Quell'angolo retto di {numref}`fig-curve-di-livello` non è un vezzo del
disegnatore, ed è utile capire da dove viene. Camminare lungo un anello vuol
dire, per definizione dell'anello, restare alla stessa quota: in quella
direzione il costo non cambia di niente. La direzione in cui il costo cambia
*di più* è dunque la più lontana possibile da quella, e sul piano la più
lontana possibile da una direzione è quella a novanta gradi. Ecco perché la
freccia esce sempre perpendicolare all'anello (chi preferisce la parola tecnica
la trova come «ortogonale»: vuol dire la stessa cosa).

È lo stesso angolo retto a spiegare un fastidio che si incontra sempre. Se la
conca invece di essere tonda si allunga in una valle stretta, gli anelli
diventano ovali schiacciati, e la perpendicolare a un ovale schiacciato punta
verso il fianco vicino, non verso il fondo lontano. Il percorso allora
zigzaga da una parete all'altra e avanza poco.

`````{tab} Elementare

La **derivata parziale** è semplice: tieni fermi tutti i parametri tranne uno
e misura la pendenza rispetto a quello, come chiudere gli occhi su tutte le
manopole di un mixer tranne una e ascoltare l'effetto di quella sola. Metti in
fila tutte queste pendenze e ottieni il **gradiente**: un vettore che punta
nella direzione in cui il costo cresce più in fretta (la salita più ripida).
Per *scendere*, ci basta andare nel verso opposto.

`````

`````{tab} Superiore

Per una loss $\mathcal{L}(\theta)$ che dipende dai parametri
$\theta = (\theta_1, \dots, \theta_n)$, il gradiente è il vettore delle
derivate parziali:

$$
\nabla \mathcal{L}(\theta) =
\begin{bmatrix} \dfrac{\partial \mathcal{L}}{\partial \theta_1} \\[4pt]
\vdots \\[2pt]
\dfrac{\partial \mathcal{L}}{\partial \theta_n} \end{bmatrix}
\in \mathbb{R}^n .
$$

```{admonition} Una convenzione, dichiarata una volta per tutte
:class: note
Il libro adotta il **layout al denominatore**: la derivata di uno scalare
rispetto a un oggetto ha **sempre la stessa forma di quell'oggetto**. Il
gradiente rispetto a un vettore è quindi un vettore colonna, e
$\partial\mathcal{L}/\partial\mathbf{W}$ è una matrice $m\times n$ come
$\mathbf{W}$. Non è pedanteria: è la differenza fra
$\boldsymbol{\delta}\mathbf{a}^\top$ e $\mathbf{a}\boldsymbol{\delta}^\top$,
cioè fra un aggiornamento dei pesi che ha le dimensioni giuste e uno che non
si può nemmeno scrivere. Il capitolo sulle reti neurali e quello su PyTorch
compongono catene di derivate: è la stessa convenzione, e il lettore che le
rifà a mano deve poterle attaccare senza trasposte a sorpresa.
```

Vale un fatto centrale: $\nabla\mathcal{L}$ indica la direzione di **massima
crescita** di $\mathcal{L}$, quindi $-\nabla\mathcal{L}$ è la direzione di
massima discesa. È il verso in cui muoveremo i parametri, e si dimostra in due
righe. La variazione di $\mathcal{L}$ nella direzione di un versore
$\mathbf{u}$ (la **derivata direzionale**) è
$D_\mathbf{u}\mathcal{L} = \nabla\mathcal{L}^\top\mathbf{u}$; per la
disuguaglianza di Cauchy–Schwarz vale
$|\nabla\mathcal{L}^\top\mathbf{u}| \le \lVert\nabla\mathcal{L}\rVert$, con
uguaglianza se e solo se $\mathbf{u}$ è parallelo a $\nabla\mathcal{L}$. Da
la perpendicolarità alle curve di livello si ottiene invece con la regola
della catena. Se $\gamma(t)$ è una curva che resta su una curva di livello,
allora $\mathcal{L}(\gamma(t))$ è costante, quindi
$\frac{d}{dt}\mathcal{L}(\gamma(t)) = \nabla\mathcal{L}^\top \gamma'(t) = 0$:
il gradiente è ortogonale a ogni direzione tangente all'insieme di livello.
Lungo una curva di
livello $D_\mathbf{u}\mathcal{L}=0$, cioè
$\nabla\mathcal{L}^\top\mathbf{u}=0$.

Un'avvertenza che tornerà utile fra poche righe: quel primato è relativo alla
**norma euclidea**. «Il passo di lunghezza fissata che fa scendere di più»
dipende da come si misura la lunghezza di un passo, e cambiando metrica cambia
la direzione più ripida. Non è un cavillo: è precisamente ciò che fanno Adam,
riscalando ogni coordinata, e i metodi del secondo ordine, misurando i passi
con la curvatura. Il gradiente non è *la* direzione migliore in assoluto, è la
migliore secondo un metro particolare.

`````

## La regola della catena: il motore del backpropagation

Una rete neurale è una funzione dentro una funzione dentro una funzione: strati
impilati, ognuno che riceve l'uscita del precedente. Per sapere come un peso
del primo strato influenza il costo finale servono le derivate delle funzioni
composte.

```{figure} ../figures/chain-rule.svg
:name: fig-regola-catena
:alt: "In alto una catena di funzioni annidate: x entra in f, l'uscita di f entra in g, quella di g entra in h e si arriva alla loss. In basso il percorso inverso: a ogni anello è associata la sua derivata locale, f primo uguale 2, g primo uguale 3, h primo uguale 0,5, e le tre si moltiplicano fra loro dando una derivata totale di 3. Una nota chiude: ogni anello conosce solo il proprio tratto, il prodotto conosce la catena intera."
:width: 92%

La regola della catena in un numero. Le tre funzioni si chiamano $f$, $g$ e
$h$, e l'apostrofo accanto al nome (si legge «f primo») è il modo consueto di
indicare la derivata di quella funzione: $f' = 2$ vuol dire che quel tratto
di catena amplifica per due. Nessun anello sa dove porti la catena: conosce
solo quanto amplifica ciò che gli arriva, e il prodotto di quegli effetti
locali dà l'effetto complessivo.
```

L'ultima riga di {numref}`fig-regola-catena` riassume tutto in una frase.
Il calcolo si può fare localmente, anello per anello, senza che nessuno abbia
in testa la funzione intera, ed è questo che rende derivabile una rete da
milioni di parametri con lo stesso sforzo per ogni peso. La procedura che lo fa
ha un nome che si incontra ovunque, ed è quello che compare fra due righe: si
chiama *backpropagation*, cioè «propagazione all'indietro».

`````{tab} Elementare

Tre ingranaggi in fila: A muove B, B muove C. Se B gira due volte più in fretta
di A, e C una volta e mezza più in fretta di B, allora C gira rispetto ad A di
$2 \times 1{,}5 = 3$ volte. Gli effetti lungo la catena si
**moltiplicano**.

E un ingranaggio è una derivata travestita, perché «quanti giri fa B per ogni
giro di A» è esattamente la domanda della derivata: *se muovo un po’
l'ingresso, di quanto si muove l'uscita?* Sostituendo agli ingranaggi gli
strati di una rete, la conclusione è la stessa: le pendenze si moltiplicano una
dopo l'altra. Il *backpropagation* è questo e nient'altro: moltiplicare le
pendenze strato per strato, partendo dall'uscita e risalendo verso l'ingresso.

`````

`````{tab} Superiore

Per una funzione composta $\mathcal{L} = f\big(g(w)\big)$ la regola della
catena dà

$$
\frac{d\mathcal{L}}{dw} =
\frac{df}{dg} \cdot \frac{dg}{dw},
$$

il prodotto tra la pendenza della funzione esterna $f$ (valutata in $g(w)$) e
quella della funzione interna $g$. In una rete profonda la catena si allunga
di un anello per strato, e le derivate si moltiplicano una dopo l'altra. Il
**backpropagation** applica questa regola in
ordine inverso (dall'uscita agli ingressi) riutilizzando i fattori condivisi
tra i cammini. È ciò che permette di calcolare il gradiente rispetto a milioni
di parametri in un'unica passata all'indietro, invece di derivare ogni peso da
capo {cite}`rumelhart1986learning`.

`````

## La discesa del gradiente

Ora abbiamo tutto: uno strumento che dice da che parte si scende (il gradiente)
e un posto dove si vuole arrivare, il fondo della valle, che è per l'appunto il
punto in cui il costo è più piccolo. La ricetta è quella dell'escursionista
nella nebbia: un passo in discesa, ricalcola, ripeti
({numref}`fig-discesa-gradiente`).

```{figure} ../figures/discesa-gradiente.svg
:name: fig-discesa-gradiente
:alt: Curva di costo a forma di scodella con quattro punti che scendono lungo il fianco verso il minimo, collegati da frecce; i passi si accorciano avvicinandosi al fondo.
:width: 85%

La funzione di costo $\mathcal{L}(\theta)$ come una scodella. Sull'asse
orizzontale c'è il parametro da regolare, che si scrive con la lettera greca
$\theta$ (si legge «theta»); il numerino in basso conta i passi, quindi
$\theta_0$ è la regolazione di partenza, quella scelta a caso prima che
l'addestramento cominci. Da lì ogni passo va nel verso opposto al gradiente, e
i passi si accorciano avvicinandosi al minimo, dove la pendenza (e quindi il
passo) tende a zero.
```

La scodella è il caso gentile. Appena la valle si allunga in una direzione, la
ricetta «vai dove è più ripido» smette di puntare verso il fondo.

```{figure} ../figures/sgd-momentum.svg
:name: fig-valle-allungata
:alt: "Curve di livello di una valle stretta e allungata, percorsa da due traiettorie. Senza momentum il percorso zigzaga da una parete all'altra e avanza poco lungo l'asse della valle. Con il momentum le oscillazioni laterali si cancellano fra loro e la traiettoria scorre diritta verso il minimo."
:width: 96%

Due modi di scendere nella stessa valle stretta. Il percorso etichettato «SGD
puro» è la ricetta di base, un passo alla volta nella direzione più ripida del
momento, e rimbalza da una parete all'altra. Quello etichettato «SGD +
momentum» tiene conto anche dei passi precedenti e scorre diritto verso il
fondo.
```

Il meccanismo di {numref}`fig-valle-allungata` si chiama **momento**
(*momentum*), e sotto il nome fisico c'è una ricetta più semplice
dell'immagine della pallina che rotola: invece di muoversi lungo la pendenza
sentita adesso, ci si muove lungo una **media delle ultime pendenze sentite**.
Il perché funzioni si vede senza formule. In una valle stretta la pendenza ha
due parti: quella che attraversa la valle, che a ogni passo cambia verso
perché si sbatte prima contro una parete e poi contro l'altra, e quella che
scende lungo la valle, che punta sempre dalla stessa parte. Facendo la media,
la prima si cancella da sé (una volta è più uno, la volta dopo è meno uno) e
la seconda si somma. Restano meno rimbalzi e più avanzamento, che è appunto
quel che mostra il disegno.

La sigla del disegno, **SGD**, sta per *stochastic gradient descent*, discesa
stocastica del gradiente: è la discesa raccontata qui, con l'accorgimento che
a ogni passo la pendenza non si misura su tutti i dati ma su un pugno di
esempi presi a caso, il che la rende più sbrigativa e un po’ traballante. È la
variante che si usa in pratica, e su di essa il momento è quasi sempre
attivo.

`````{tab} Elementare

Cammini verso il basso e a ogni passo scegli la direzione di discesa. Quanto
lungo sia il passo lo decidono due cose insieme: quanto è ripido lì dove sei
(più ripido, passo più lungo) e una manopola che moltiplica tutto, il
**learning rate** (tasso di apprendimento). La manopola è la sola che scegli
tu, ed è
un compromesso delicato: un passo troppo lungo scavalca il fondo e ti fa
rimbalzare da una parete all'altra senza mai fermarti; un passo troppo corto
arriva, ma dopo un'eternità. Trovare la lunghezza giusta è metà del mestiere
di chi addestra modelli.

`````

`````{tab} Superiore

L'aggiornamento è una sola riga, ripetuta:

$$
\theta \leftarrow \theta - \eta \, \nabla \mathcal{L}(\theta).
$$

Qui $\theta$ sono i parametri, $\nabla\mathcal{L}(\theta)$ il gradiente della
loss e $\eta > 0$ il **learning rate**, che dosa l'ampiezza del passo. Nella
pratica il gradiente non si calcola su tutti i dati a ogni passo, ma su un
piccolo lotto (*mini-batch*) di esempi: è la **discesa stocastica del
gradiente** (SGD), più rumorosa ma molto più veloce, e base di ottimizzatori
moderni come Adam.

`````

## Minimi locali e globali: perché al deep learning basta così

La discesa del gradiente scende sempre. Ma "in fondo a cosa", esattamente?

```{figure} ../figures/minimi-locali-plateau-sella.svg
:name: fig-paesaggio-loss
:alt: "Profilo stilizzato di una superficie di loss percorsa da sinistra a destra: un tratto quasi orizzontale segnato come plateau, una conca poco profonda segnata come minimo locale, un tratto in cui la discesa si appiattisce per poi riprendere (in una dimensione l'analogo del punto di sella) e infine la conca più profonda, il minimo globale. Una nota avverte che nei punti piatti, plateau e sella, il segnale di discesa quasi scompare."
:width: 92%

I quattro luoghi dove una pallina che segue il gradiente può fermarsi o
rallentare, con i nomi che si trovano nel disegno. Un **plateau** è un
altopiano, un tratto in cui il terreno è quasi orizzontale su una distanza
lunga. Un **minimo locale** è una conca vera, ma non la più profonda del
paesaggio. Un **punto di sella** è un passo di montagna: sceso da una parte, si
sale dall'altra, quindi non è né una cima né un fondo (visto su un profilo a
una dimensione sola, com'è qui, si presenta come un tratto che si appiattisce e
poi riprende a scendere). L'ultimo è il **minimo globale**, il fondo più basso
di tutti, ed è l'unico che vorremmo: nulla, nel gradiente, dice alla pallina in
quale dei quattro si trova.
```

Il guaio, guardando {numref}`fig-paesaggio-loss`, non sono tanto i minimi
locali quanto le zone piatte. In un minimo locale il gradiente è nullo e
l'ottimizzazione si ferma, il che almeno si nota; su un plateau o vicino a una
sella il gradiente è quasi nullo, l'addestramento procede lentissimo e non
c'è modo di distinguerlo, dall'esterno, da un problema difficile.

`````{tab} Elementare

Dipende dal paesaggio. Se è una scodella liscia, con un'unica valle, non ci
sono conche secondarie in cui restare intrappolati: da qualunque punto si
parta si scende verso quell'unico fondo, purché il passo non sia troppo lungo.
È il caso **convesso**, il più comodo. Se invece è una catena montuosa piena
di conche, si può finire intrappolati in una conca che non è la più profonda:
un **minimo locale**, un
buon posto ma non il migliore.

`````

`````{tab} Superiore

Una funzione è **convessa** se il segmento che unisce due punti qualsiasi del
suo grafico non sta mai sotto la curva (la formulazione con il «non sotto»
invece che con il «sopra» serve a non escludere le rette, che sono convesse e
per cui il segmento sta *sulla* curva). Per una funzione convessa ogni minimo
locale è anche globale: nessuna conca secondaria in cui restare intrappolati.

La convergenza della discesa del gradiente, però, richiede due ipotesi in più,
e vale la pena enunciarle perché senza di esse l'affermazione è falsa. La
prima è che il gradiente sia **lipschitziano** di costante $L$ (cioè che la
curvatura sia limitata da $L$), e allora ogni passo fisso $\eta < 2/L$ va
bene. La seconda è che **il minimo esista**. Nessuna delle due è gratis:
$f(x)=x^4$ è convessa e liscia, ma $f''(x)=12x^2$ è illimitata, e per *ogni*
$\eta$ fissato la discesa diverge se si parte abbastanza lontano (la soglia è
$|x_0| > 1/\sqrt{2\eta}$: con $\eta=10^{-9}$ basta partire da $x_0 = 23\,000$);
e $f(x)=e^x$ è convessa con gradiente sempre positivo e nessun minimo, quindi
la discesa scende per sempre senza convergere a niente. Anche restando dentro
le ipotesi, un passo troppo lungo diverge in una scodella perfetta, come
mostrerà tra poco l'esperimento con `eta = 1.1`.

Le loss del deep learning, poi, sono quasi sempre **non convesse**: nessuna
garanzia. La buona
notizia empirica è che per reti molto grandi i minimi locali "buoni" sono
tantissimi e quasi equivalenti al globale; gli ostacoli veri sono più i punti
di sella che le conche profonde {cite}`dauphin2014identifying`. Ci si
accontenta (con ottimi risultati) di un minimo *abbastanza buono*.

`````

## In pratica, con NumPy

Un esempio giocattolo rende tutto concreto: minimizziamo $\mathcal{L}(\theta) =
(\theta - 3)^2$, la cui derivata è $2(\theta - 3)$ e il cui minimo è ovviamente
in $\theta = 3$.

```python
import numpy as np

# Costo da minimizzare: L(theta) = (theta - 3)^2, minimo in theta = 3
def grad(theta):
    return 2 * (theta - 3)          # derivata della loss

theta = -4.0                        # punto di partenza sul fianco della scodella
eta = 0.1                           # learning rate

for _ in range(20):
    theta = theta - eta * grad(theta)   # un passo di discesa del gradiente

print(round(theta, 3))              # -> 2.919, ormai vicino al minimo 3
```

Cambia `eta` e osserva: con un valore piccolo (`0.01`) l'avvicinamento al
minimo, che in gergo si chiama **convergenza**, rallenta
(dopo venti passi $\theta$ è a $-1{,}67$, ancora lontano); con uno troppo
grande (`1.1`) $\theta$ **diverge** oscillando, cioè scappa via invece di
avvicinarsi, saltando a ogni passo da una parte all'altra del minimo e sempre
più lontano (dopo venti passi vale $-265$). È la stessa dinamica, in scala
minima, che governa l'addestramento di una rete con miliardi di pesi.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- La **derivata** misura la pendenza: di quanto cambia l'uscita se muovo di
  poco l'ingresso, come il tachimetro dice quanto in fretta cambia la
  posizione. Dove il terreno è piatto (in cima a una gobba, in fondo a una
  conca) vale zero.
- Il **gradiente** mette in fila la pendenza rispetto a ogni parametro preso da
  solo, una manopola alla volta: indica la direzione in cui il costo cresce più
  in fretta, e per scendere si va nel verso opposto.
- La **discesa del gradiente** è l'escursionista nella nebbia: un passo verso il
  basso, si risente la pendenza, si ripete. Il passo è tanto più lungo quanto
  più è ripido, moltiplicato per una manopola che si chiama **learning rate**:
  quella è la scelta delicata, perché con la manopola troppo alta si rimbalza da
  una parete all'altra e con quella troppo bassa si arriva dopo un'eternità.
- La **regola della catena** moltiplica fra loro le pendenze anello per anello,
  come ingranaggi che si trascinano: è così che la correzione risale
  dall'uscita fino ai primi strati (il *backpropagation*).
- Il paesaggio di una rete profonda non è una scodella liscia ma una catena
  montuosa: nessuno garantisce che si arrivi al fondo più basso, e in pratica
  una conca abbastanza profonda basta quasi sempre.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- La **derivata** misura la pendenza: di quanto cambia l'uscita se muovo di
  poco l'ingresso. È zero nei punti stazionari.
- Il **gradiente** $\nabla\mathcal{L}$ è il vettore delle derivate parziali:
  punta verso la massima crescita del costo, e noi andiamo nel verso opposto.
- La **discesa del gradiente** aggiorna i parametri con
  $\theta \leftarrow \theta - \eta\,\nabla\mathcal{L}(\theta)$; il **learning
  rate** $\eta$ dosa la lunghezza del passo.
- La **regola della catena** propaga le derivate lungo gli strati: è il cuore
  del *backpropagation*.
- In deep learning la loss non è convessa, ma un minimo "abbastanza buono"
  basta quasi sempre.
```
`````
