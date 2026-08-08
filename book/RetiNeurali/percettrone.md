# Il percettrone e la sua regola di apprendimento

Nel 1958 uno psicologo di Cornell, Frank Rosenblatt, presenta alla stampa il
*percettrone*, un modello di neurone artificiale che impara a riconoscere
forme dagli esempi. Il *New York Times* scrive che è l'embrione di un cervello
elettronico capace, un giorno, di camminare, parlare e riprodursi. Due anni
dopo l'idea prende corpo in una macchina grande come un armadio, il *Mark I
Perceptron*: una griglia di quattrocento fotocellule collegate da fili a pesi
realizzati con potenziometri motorizzati. L'hype era smisurato, e lo pagheremo
caro qualche pagina più avanti. Ma dietro il clamore c'è un'idea sobria e
duratura, che ancora oggi è il mattone di ogni rete neurale: un neurone
artificiale non è altro che un pezzo di aritmetica.

## Un neurone fatto di aritmetica

Un neurone biologico riceve segnali da altri neuroni, li combina e "scarica" un
impulso se lo stimolo complessivo supera una soglia. Rosenblatt cattura questa
idea con tre gesti: pesare gli ingressi, sommarli, decidere.

```{figure} ../figures/neurone-artificiale.svg
:name: fig-neurone
:alt: Gli ingressi x1, x2, fino a xn sono moltiplicati per i pesi w1, w2, wn e sommati in un nodo sigma che aggiunge il bias b; la somma passa in una funzione di attivazione a gradino e produce l'uscita y.
:width: 90%

Il neurone artificiale: ogni ingresso ha un peso, il sommatore $\Sigma$ combina
tutto (più il bias $b$), la funzione di attivazione decide l'uscita.
```

Ogni ingresso $x_i$ arriva con un **peso** $w_i$ che ne misura l'importanza. Il
neurone li combina in una somma pesata e vi aggiunge un termine costante, il
**bias** $b$ ({numref}`fig-neurone`).

`````{tab} Elementare

Immagina di decidere se uscire di casa con l'ombrello. Guardi alcuni indizi:
quanto è nuvoloso, l'umidità, cosa dice l'app del meteo. Dai a ciascun indizio
un peso (l'app conta più del colore del cielo) e fai una somma: indizio per il
suo peso, il tutto sommato. Il bias è la tua indole di partenza: un pessimista
parte già orientato verso il "sì, prendilo". Se il totale supera una soglia,
esci con l'ombrello.

Con due ingressi la somma è semplicemente

$$
z = w_1 x_1 + w_2 x_2 + b .
$$

Proviamo con i numeri. Nuvolosità $x_1 = 7$ (su una scala da 0 a 10) con peso
$w_1 = 0{,}3$; l'app che dice pioggia $x_2 = 1$ con peso $w_2 = 2$ (ti fidi
molto dell'app); indole pessimista $b = 0{,}5$. Totale:
$z = 0{,}3 \cdot 7 + 2 \cdot 1 + 0{,}5 = 4{,}6$. Se la tua soglia è "esco con
l'ombrello sopra il 3", oggi l'ombrello lo prendi.

`````

`````{tab} Superiore

Raccogliamo gli ingressi in un vettore $\mathbf{x}\in\mathbb{R}^n$ e i pesi in
$\mathbf{w}\in\mathbb{R}^n$. La somma pesata è un **prodotto scalare** più il
bias:

$$
z = \mathbf{w}^\top\mathbf{x} + b = \sum_{i=1}^{n} w_i x_i + b .
$$

È la stessa operazione vista nel capitolo di algebra lineare: $z$ è grande e
positivo quando $\mathbf{x}$ "punta nella direzione" di $\mathbf{w}$. Il luogo
dei punti in cui $z = 0$, cioè $\mathbf{w}^\top\mathbf{x} + b = 0$, è un
**iperpiano**: la frontiera che il neurone traccia per separare lo spazio degli
ingressi in due regioni.

`````

## La funzione a gradino: decidere sì o no

La somma pesata $z$ è un numero qualsiasi. Per trasformarla in una decisione
serve un ultimo passo, la **funzione di attivazione**. Nel percettrone classico
è la più netta possibile: la funzione a **gradino** (o di Heaviside).

`````{tab} Elementare

Il gradino è un interruttore: se la somma raggiunge la soglia, l'uscita è $1$
("sì"); altrimenti è $0$ ("no"). Niente sfumature, solo acceso o spento. Tornando
all'ombrello: sommati gli indizi, o esci con l'ombrello o non lo prendi.

`````

`````{tab} Superiore

L'uscita del neurone è $\hat{y} = g(z)$, con $g$ la funzione a gradino:

$$
g(z) = \begin{cases} 1 & \text{se } z \ge 0, \\ 0 & \text{altrimenti.}\end{cases}
$$

La decisione è dunque binaria: $\hat{y}\in\{0,1\}$. Il neurone assegna la
classe $1$ ai punti da un lato dell'iperpiano $\mathbf{w}^\top\mathbf{x}+b=0$ e
la classe $0$ a quelli dall'altro. È un **classificatore lineare**: sposta il
bias $b$ e trasli la frontiera; ruota $\mathbf{w}$ e la inclini.

`````

## Imparare dagli errori: la regola del percettrone

Fin qui il neurone sa *calcolare*, ma non ancora *imparare*: chi sceglie i
pesi? L'intuizione di Rosenblatt è di lasciarli trovare alla macchina, un
esempio alla volta. Le si mostra un input di cui conosciamo la risposta giusta
$y$; se sbaglia, si correggono i pesi nella direzione che riduce l'errore.

```{figure} ../figures/perceptron-adaline-da-zero.svg
:name: fig-neurone-con-retroazione
:alt: "Schema di un neurone artificiale percorso in due sensi: in avanti gli ingressi vengono pesati, sommati e passati alla funzione a gradino che produce l'uscita; all'indietro, il confronto fra uscita e risposta attesa genera una correzione che torna sui pesi."
:width: 88%

Lo stesso neurone, con la freccia di ritorno. È quella a fare la differenza
fra un circuito che calcola e un modello che impara.
```

La freccia di ritorno in {numref}`fig-neurone-con-retroazione` è, in miniatura,
tutto ciò che questo libro chiamerà addestramento. Cambierà la funzione al
posto del gradino, cambierà il modo di calcolare la correzione, ma lo schema
resta: si misura lo scarto dalla risposta attesa e lo si rimanda sui pesi.

```{figure} ../figures/percettrone-impara.svg
:name: fig-percettrone-impara
:alt: "Animazione: una retta di separazione parte con l'inclinazione sbagliata e, a ogni punto classificato male, ruota; dopo quattro correzioni i punti terracotta e quelli teal sono da parti opposte."
:width: 85%

La regola all'opera: la retta parte sbagliata e a ogni punto classificato male
ruota un po'. Dopo quattro correzioni le due classi sono separate, e da lì in
poi nessun esempio provoca più un aggiornamento.
```

Nella {numref}`fig-percettrone-impara` si vede la proprietà che rese famoso
l'algoritmo: **quando una retta separatrice esiste, il percettrone la trova in
un numero finito di correzioni**. È il teorema di convergenza di Rosenblatt, e
il "quando esiste" è la clausola che fra poco presenterà il conto.

`````{tab} Elementare

La ricetta è quasi banale. Per ogni esempio:

- se il neurone azzecca la risposta, non tocchi nulla;
- se dice $0$ e doveva dire $1$, alzi un po' i pesi degli ingressi attivi;
- se dice $1$ e doveva dire $0$, li abbassi.

Ripeti su tutti gli esempi, più volte. Ogni correzione è piccola (la governa un
passo $\eta$), così la macchina non "salta" ma si assesta gradualmente.

`````

`````{tab} Superiore

Sia $\eta > 0$ il **tasso di apprendimento**. Per ogni esempio
$(\mathbf{x}, y)$ si calcola la predizione $\hat{y}$ e si aggiornano pesi e bias:

$$
w_i \leftarrow w_i + \eta\,(y - \hat{y})\,x_i, \qquad
b \leftarrow b + \eta\,(y - \hat{y}).
$$

Il fattore $(y-\hat{y})$ vale $0$ quando la predizione è corretta (nessun
aggiornamento), $+1$ o $-1$ altrimenti. Rosenblatt dimostrò il **teorema di
convergenza del percettrone**: se i dati sono linearmente separabili,
l'algoritmo trova in un numero finito di passi un iperpiano che li separa. Il
seme dell'apprendimento moderno è già qui, anche se la discesa del gradiente su
loss differenziabili verrà dopo.

`````

Tradotta in NumPy, la ricetta è quasi identica a come l'abbiamo raccontata:

```python
import numpy as np

def gradino(z):
    return np.where(z >= 0, 1, 0)          # decisione binaria 0/1

def addestra(X, y, eta=0.1, epoche=10):
    w = np.zeros(X.shape[1])               # pesi iniziali a zero
    b = 0.0
    for _ in range(epoche):
        for xi, target in zip(X, y):
            pred = gradino(w @ xi + b)      # prodotto scalare + bias
            errore = target - pred
            w += eta * errore * xi          # aggiorna i pesi
            b += eta * errore               # aggiorna il bias
    return w, b

# La porta logica AND: separabile con una retta
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])
w, b = addestra(X, y_and)
print(gradino(X @ w + b))                   # -> [0 0 0 1]: ha imparato la AND
```

## Il muro dello XOR

```{figure} ../figures/perceptron-primo-inverno-ai.svg
:name: fig-inverni-ai
:alt: "In alto lo schema del neurone artificiale di Rosenblatt; sotto, una linea del tempo che segna l'entusiasmo iniziale degli anni Cinquanta e Sessanta, il crollo dei finanziamenti dopo il 1969 e il secondo inverno degli anni Ottanta, con in mezzo i periodi di ripresa."
:width: 100%

Un limite matematico e le sue conseguenze storiche. La dimostrazione di
Minsky e Papert riguardava un modello a uno strato; il gelo che ne seguì
colpì l'intero campo.
```

La sproporzione visibile in {numref}`fig-inverni-ai` fra la portata del
risultato e l'ampiezza della reazione è una lezione che vale oltre questa
storia. Il teorema era corretto e limitato; la sua lettura pubblica fu che le
reti neurali non funzionavano, e servirono quindici anni per rimediare.

E qui torna il conto lasciato in sospeso. Nel 1969 Marvin Minsky e Seymour
Papert, nel libro *Perceptrons*, mostrano un limite che sembra insormontabile.
Prendi la funzione logica **XOR** ("o esclusivo"): vale $1$ quando i due
ingressi sono diversi, $0$ quando sono uguali.

```{figure} ../figures/xor-non-separabile.svg
:name: fig-xor-non-separabile
:alt: "Animazione: una retta ruota su quattro punti disposti a XOR; a ogni orientamento due punti restano dalla parte sbagliata e vengono cerchiati."
:width: 85%

La stessa retta della figura precedente, sugli stessi quattro punti dello XOR:
per quanto la si giri, restano sempre **due** punti dalla parte sbagliata.
```

Il contrasto con la {numref}`fig-percettrone-impara` è tutto il punto: là la
rotazione finiva, qui non finisce mai. La {numref}`fig-xor-non-separabile` non
prova il teorema (mostra solo alcuni orientamenti) ma rende evidente da dove
viene l'ostacolo: le due classi occupano angoli **opposti** del quadrato.

`````{tab} Elementare

Disegna i quattro casi su un foglio. I punti $(0,0)$ e $(1,1)$ vogliono uscita
$0$; i punti $(0,1)$ e $(1,0)$ vogliono uscita $1$. Prova a separarli con una
sola riga dritta: è impossibile. Le due classi stanno negli angoli opposti del
quadrato, "incrociate". Un singolo percettrone traccia solo una linea, quindi
sullo XOR è condannato a sbagliare almeno un caso.

`````

`````{tab} Superiore

Lo XOR **non è linearmente separabile**: non esiste alcun $(\mathbf{w}, b)$ tale
che $g(\mathbf{w}^\top\mathbf{x}+b)$ riproduca la tabella. Le classi $\{(0,0),
(1,1)\}$ e $\{(0,1),(1,0)\}$ non sono divisibili da un iperpiano in
$\mathbb{R}^2$. Se lanci `addestra` su `y_xor = np.array([0, 1, 1, 0])`,
l'algoritmo non converge: continua a oscillare a ogni epoca. Non è un difetto
dell'ottimizzatore, è un limite di *capacità* del modello.

`````

Il libro di Minsky e Papert raffreddò gli entusiasmi e contribuì al primo
"inverno" dell'AI. Ma la critica indicava anche la via d'uscita.

## Oltre la linea: strati nascosti e non linearità

Se un neurone traccia una sola linea, mettiamone di più. Impilando i neuroni
in uno **strato nascosto** e componendo gli strati, la rete può piegare la
frontiera fino a separare anche lo XOR: un primo strato costruisce
rappresentazioni intermedie, un secondo le combina. C'è però una condizione
non negoziabile: tra uno strato e l'altro serve una **non linearità**.
Comporre due trasformazioni lineari dà ancora una trasformazione lineare:
cento strati senza attivazioni collasserebbero in un unico iperpiano, di nuovo
incapace di XOR. È qui che entrano funzioni come la ReLU o la sigmoide, e con
esse il percettrone multistrato (MLP) e l'algoritmo che lo addestra, la
*backpropagation*: il tema dei prossimi capitoli.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un neurone artificiale è un pezzo di aritmetica: dà a ogni indizio in
  ingresso un **peso**, somma tutto, aggiunge la propria indole di partenza
  (il **bias**) e passa il totale a un interruttore che risponde sì o no.
- Il percettrone **impara sbagliando**: quando azzecca non tocca niente, quando
  dice "no" e doveva dire "sì" alza un po' i pesi degli ingressi attivi, e
  viceversa. Le correzioni sono piccole e si ripetono su tutti gli esempi.
- Un neurone solo sa tracciare **una riga dritta** fra le due classi: se quella
  riga esiste la trova, ma sullo **XOR** non esiste, perché i casi da separare
  stanno negli angoli opposti del quadrato.
- Per piegare la frontiera servono più neuroni impilati in **strati** e, fra
  uno strato e l'altro, un passaggio che non sia una semplice riga: è il ponte
  verso le reti neurali profonde.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Un neurone artificiale calcola una **somma pesata** degli ingressi più un
  bias, $\mathbf{w}^\top\mathbf{x}+b$, e la fa passare in una funzione di
  attivazione.
- Il percettrone **impara** correggendo i pesi in proporzione all'errore:
  $w_i \leftarrow w_i + \eta\,(y-\hat{y})\,x_i$.
- Un solo neurone è un **classificatore lineare**: separa lo spazio con un
  iperpiano e fallisce su problemi non separabili come lo **XOR**.
- Servono **strati nascosti** e **non linearità** per superare quel limite: è
  il ponte verso le reti neurali profonde.
```

`````
