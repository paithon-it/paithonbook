# Analisi e ottimizzazione: derivate e discesa del gradiente

Immagina di dover scendere a valle nella nebbia più fitta, senza mappa e senza
vedere a un metro di distanza. Un'informazione, però, ce l'hai sempre: la
pendenza del terreno sotto i piedi. Ti basta sentire da che parte scende, fare
un passo in quella direzione, rimisurare e ripetere. Addestrare un modello è
esattamente questo. La collina da scendere è la funzione che misura *quanto il
modello sbaglia* (il **costo** o **loss**, che indichiamo con $\mathcal{L}$) e
lo strumento che sente la pendenza sotto i piedi è la **derivata**. Questa
sezione è la bussola promessa all'inizio del capitolo.

## La derivata: la pendenza istante per istante

Una funzione lega un ingresso a un'uscita. La derivata risponde a una domanda
sola: *se muovo l'ingresso di un pelo, di quanto cambia l'uscita?*

`````{tab} Elementare

Pensa al tachimetro di un'auto. La posizione cambia nel tempo, e la velocità
è "quanto in fretta" cambia: è la derivata della posizione. Su un grafico,
la derivata in un punto è la **pendenza della tangente** lì. Dove la curva
sale ripida la derivata è grande e positiva; dove scende è negativa; in cima
a una gobba o in fondo a una conca, dove per un istante il terreno è piatto,
la derivata vale **zero**.

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

In pratica non calcoliamo mai il limite a mano: bastano poche regole. E tre
famiglie di funzioni compaiono ovunque nel machine learning.

`````{tab} Elementare

Le tre "solite sospette" sono: le **potenze** (la parabola $x^2$ è la forma
dell'errore quadratico, quello che si minimizza quando il modello deve
prevedere un numero), l'**esponenziale** $e^x$ (dentro la sigmoide e la
softmax, che trasformano punteggi in probabilità) e il **logaritmo** (nella
cross-entropy, la loss della classificazione). Per derivarle si va a memoria:
la pendenza di $x^2$ è $2x$ (nel punto $x=3$ la parabola sale con pendenza
$6$), quella di $e^x$ è di nuovo $e^x$: derivandola, resta identica a sé
stessa.

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
tutti insieme. Ci serve la derivata "una direzione alla volta".

```{figure} ../figures/derivate-gradiente.svg
:name: fig-curve-di-livello
:alt: "Una superficie di loss vista dall'alto, disegnata come curve di livello concentriche attorno a un minimo. Sopra di essa il percorso della discesa del gradiente: una successione di passi che in ogni punto imbocca la direzione perpendicolare alla curva di livello, cioè quella di massima pendenza, avvicinandosi progressivamente al centro."
:width: 88%

La stessa superficie vista dall'alto. Il gradiente in ogni punto è
perpendicolare alla curva di livello che ci passa: è la direzione in cui la
loss cambia più in fretta.
```

La perpendicolarità visibile in {numref}`fig-curve-di-livello` non è un
dettaglio grafico ma la definizione stessa del gradiente, e spiega perché il
percorso zigzaghi in una valle allungata: la direzione più ripida punta verso
il fianco più vicino, non verso il fondo.

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
\left( \frac{\partial \mathcal{L}}{\partial \theta_1}, \;
\dots, \; \frac{\partial \mathcal{L}}{\partial \theta_n} \right).
$$

Vale un fatto centrale: $\nabla\mathcal{L}$ indica la direzione di **massima
crescita** di $\mathcal{L}$, quindi $-\nabla\mathcal{L}$ è la direzione di
massima discesa. È il verso in cui muoveremo i parametri.

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

La regola della catena in un numero. Nessun anello sa dove porti la catena:
conosce solo quanto amplifica ciò che gli arriva, e il prodotto di quegli
effetti locali dà l'effetto complessivo.
```

L'ultima riga di {numref}`fig-regola-catena` è tutto il backpropagation in
una frase. Il calcolo si può fare localmente, anello per anello, senza che
nessuno abbia in testa la funzione intera: è questo che rende derivabile una
rete da milioni di parametri con lo stesso sforzo per ogni peso.

`````{tab} Elementare

Immagina tre ingranaggi: A muove B, B muove C. Se A gira due volte più in
fretta di B, e B una volta e mezza più in fretta di C, allora A gira rispetto
a C di $2 \times 1{,}5 = 3$ volte. Le pendenze lungo la catena si
**moltiplicano**. Il *backpropagation* è esattamente questo: moltiplicare le
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
**backpropagation** {cite}`rumelhart1986learning` applica questa regola in
ordine inverso (dall'uscita agli ingressi) riutilizzando i fattori condivisi
tra i cammini. È ciò che permette di calcolare il gradiente rispetto a milioni
di parametri in un'unica passata all'indietro, invece di derivare ogni peso da
capo.

`````

## La discesa del gradiente

Ora abbiamo tutto: la pendenza (il gradiente) e la certezza che il fondo della
valle è dove il costo è minimo. La ricetta è quella dell'escursionista nella
nebbia: un passo in discesa, ricalcola, ripeti ({numref}`fig-discesa-gradiente`).

```{figure} ../figures/discesa-gradiente.svg
:name: fig-discesa-gradiente
:alt: Curva di costo a forma di scodella con quattro punti che scendono lungo il fianco verso il minimo, collegati da frecce; i passi si accorciano avvicinandosi al fondo.
:width: 85%

La funzione di costo $\mathcal{L}(\theta)$ come una scodella. Partendo da
$\theta_0$ sul fianco, ogni passo va nel verso opposto al gradiente. I passi
si accorciano avvicinandosi al minimo, dove la pendenza (e quindi il passo)
tende a zero.
```

La scodella è il caso gentile. Appena la valle si allunga in una direzione, la
ricetta «vai dove è più ripido» smette di puntare verso il fondo.

```{figure} ../figures/sgd-momentum.svg
:name: fig-valle-allungata
:alt: "Curve di livello di una valle stretta e allungata, percorsa da due traiettorie. Senza momentum il percorso zigzaga da una parete all'altra e avanza poco lungo l'asse della valle. Con il momentum le oscillazioni laterali si cancellano fra loro e la traiettoria scorre diritta verso il minimo."
:width: 96%

Perché il momento aiuta. Le componenti che cambiano segno a ogni passo si
elidono sommandosi; quella che punta sempre nella stessa direzione si accumula.
```

Il meccanismo di {numref}`fig-valle-allungata` è più semplice dell'analogia fisica
con cui si racconta di solito. Non serve immaginare una pallina con
un'inerzia: basta osservare che mediare gli ultimi gradienti cancella ciò che
oscilla e conserva ciò che è costante.

`````{tab} Elementare

Cammini verso il basso e a ogni passo scegli la direzione di discesa. La
lunghezza del passo si chiama **learning rate** (tasso di apprendimento). È
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

## Minimi locali, globali e la resa del deep learning

La discesa del gradiente scende sempre. Ma "in fondo a cosa", esattamente?

```{figure} ../figures/minimi-locali-plateau-sella.svg
:name: fig-paesaggio-loss
:alt: "Profilo stilizzato di una superficie di loss percorsa da sinistra a destra: un tratto quasi orizzontale segnato come plateau, una conca poco profonda segnata come minimo locale, un punto di flesso segnato come punto di sella e infine la conca più profonda, il minimo globale. Una nota avverte che nei punti piatti, plateau e sella, il segnale di discesa quasi scompare."
:width: 92%

I quattro luoghi dove una pallina che segue il gradiente può fermarsi o
rallentare. Solo l'ultimo è quello che vorremmo, e nulla nel gradiente dice
alla pallina in quale dei quattro si trova.
```

Il guaio, guardando {numref}`fig-paesaggio-loss`, non sono tanto i minimi
locali quanto le zone piatte. In un minimo locale il gradiente è nullo e
l'ottimizzazione si ferma, il che almeno si nota; su un plateau o vicino a una
sella il gradiente è quasi nullo, l'addestramento procede lentissimo e non
c'è modo di distinguerlo, dall'esterno, da un problema difficile.

`````{tab} Elementare

Dipende dal paesaggio. Se è una scodella liscia, con un'unica valle, qualsiasi
punto di partenza porta all'unico fondo: è il caso **convesso**, il più
comodo. Se invece è una catena montuosa piena di conche, si può finire
intrappolati in una conca che non è la più profonda: un **minimo locale**, un
buon posto ma non il migliore.

`````

`````{tab} Superiore

Una funzione è **convessa** se il segmento che unisce due punti qualsiasi del
suo grafico sta sopra la curva; per una funzione convessa ogni minimo locale è
anche globale, e la discesa del gradiente converge all'ottimo, purché il
learning rate sia abbastanza piccolo rispetto alla curvatura: con un passo
troppo lungo si diverge anche in una scodella perfetta, come mostrerà tra poco
l'esperimento con `eta = 1.1`. Le loss del deep
learning, però, sono quasi sempre **non convesse**: nessuna garanzia. La buona
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

Cambia `eta` e osserva: con un valore piccolo (`0.01`) la convergenza rallenta,
con uno troppo grande (`1.1`) $\theta$ diverge oscillando. È la stessa dinamica,
in scala minima, che governa l'addestramento di una rete con miliardi di pesi.

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
  basso, si risente la pendenza, si ripete. La lunghezza del passo si chiama
  **learning rate** ed è la scelta delicata: troppo lungo e si rimbalza da una
  parete all'altra, troppo corto e si arriva dopo un'eternità.
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
