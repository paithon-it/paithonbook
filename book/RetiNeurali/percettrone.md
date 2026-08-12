# Il percettrone e la sua regola di apprendimento

Nel 1958 uno psicologo di Cornell, Frank Rosenblatt, presenta alla stampa il
*percettrone* {cite}`rosenblatt1958perceptron`, un modello di neurone
artificiale che impara a riconoscere forme dagli esempi. Il *New York Times*
scrive che è l'embrione di un computer elettronico capace, un giorno, di
camminare, parlare e riprodursi. Due anni dopo l'idea prende corpo in una
macchina grande come un armadio, il *Mark I Perceptron*: una griglia di
quattrocento fotocellule collegate da fili a pesi realizzati con manopole che
un motorino gira da sé. Il clamore era smisurato, e lo pagheremo caro qualche
pagina più avanti. Ma sotto c'è un'idea sobria e duratura, che ancora oggi è il
mattone di ogni rete neurale: un neurone artificiale non è altro che un pezzo
di aritmetica.

## Un neurone fatto di aritmetica

Un neurone biologico riceve segnali da altri neuroni, li combina e "scarica" un
impulso se lo stimolo complessivo supera una soglia. Rosenblatt cattura questa
idea con tre gesti: pesare gli ingressi, sommarli, decidere.

```{figure} ../figures/neurone-artificiale.svg
:name: fig-neurone
:alt: Gli ingressi x1, x2, fino a xn sono moltiplicati per i pesi w1, w2, wn e sommati in un nodo sigma che aggiunge il bias b; la somma passa in una funzione di attivazione a gradino e produce l'uscita, indicata con y col cappello.
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

Le manopole a questo punto sembrano tre (i pesi, il bias, la soglia), e invece
sono due: chiedere che il totale superi $3$ è la stessa identica cosa che
togliere $3$ dal totale e chiedere che superi lo zero. La soglia si può sempre
nascondere dentro il bias, e da qui in avanti è quello che faremo: la soglia è
sempre lo zero, e a spostare il punto in cui il neurone cambia idea è il bias.

E adesso la cosa che vale la pena vedere con gli occhi, perché tutto il resto
del capitolo ci si appoggia. Prendi un foglio a quadretti e mettici i due
indizi dell'ombrello, la nuvolosità in orizzontale e l'app in verticale: ogni
giornata diventa un puntino. Con i numeri di prima, il totale pareggia la
soglia quando la nuvolosità vale $8{,}3$ e l'app dice sereno, oppure quando
vale $1{,}7$ e l'app dice pioggia. Segna quei due punti, tira una riga fra
loro: da una parte della riga il neurone risponde sempre sì, dall'altra sempre
no, e non esiste una terza possibilità. **Quella riga è tutto ciò che un
neurone sa disegnare.** Cambiare i pesi la inclina, cambiare il bias la sposta
avanti e indietro, ma resta una riga dritta. Fra qualche pagina questo dettaglio
diventerà un muro.

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

Lo stesso neurone, con la freccia di ritorno: è quella a fare la differenza fra
un circuito che calcola e un modello che impara. Un dettaglio del disegno da
non lasciarsi sfuggire: qui il bias non è tenuto da parte come nella figura
precedente, ma compare come un quarto ingresso sempre pari a $1$, con il suo
peso $w_0$. È una scrittura equivalente, e comoda, perché così anche il bias si
corregge con la stessa identica regola degli altri pesi.
```

La freccia di ritorno in {numref}`fig-neurone-con-retroazione` è, in miniatura,
tutto ciò che questo libro chiamerà addestramento. Cambierà la funzione al
posto del gradino, cambierà il modo di calcolare la correzione, ma lo schema
resta: si misura lo scarto dalla risposta attesa e lo si rimanda sui pesi.

```{figure} ../figures/percettrone-impara.svg
:name: fig-percettrone-impara
:alt: "Animazione: otto punti già etichettati, quattro di classe 1 in terracotta e quattro di classe 0 in teal. Una retta di separazione parte con l'inclinazione sbagliata e, a ogni punto classificato male, ruota; dopo quattro correzioni i punti terracotta e quelli teal sono da parti opposte."
:width: 85%

La regola all'opera su otto esempi di cui conosciamo già la risposta giusta:
quattro di classe $1$ (terracotta) e quattro di classe $0$ (teal). La retta
parte sbagliata e a ogni punto classificato male ruota un po'. Dopo quattro
correzioni le due classi sono separate, e da lì in poi nessun esempio provoca
più un aggiornamento.
```

Nella {numref}`fig-percettrone-impara` si vede la proprietà che rese famoso
l'algoritmo: **quando una retta separatrice esiste, il percettrone la trova in
un numero finito di correzioni**. È il teorema di convergenza di Rosenblatt, e
il "quando esiste" è la clausola che fra poco presenterà il conto.

`````{tab} Elementare

La ricetta è quasi banale. Per ogni esempio:

- se il neurone azzecca la risposta, non tocchi nulla;
- se dice $0$ e doveva dire $1$, alzi i pesi;
- se dice $1$ e doveva dire $0$, li abbassi.

Di quanto? In proporzione a quanto valeva quell'ingresso nell'esempio appena
sbagliato: chi valeva zero non si muove per niente (non aveva colpa, non ha
detto la sua), chi valeva molto si muove molto. È il criterio più naturale del
mondo: si corregge chi ha parlato più forte.

Ripeti su tutti gli esempi, più volte. Ogni correzione è piccola, perché la si
moltiplica per un numeretto scelto da noi, il **passo di apprendimento** (in
inglese *learning rate*, e nel codice qui sotto si chiama `eta`): così la
macchina non "salta" da una parte all'altra, ma si assesta gradualmente.

`````

`````{tab} Superiore

Sia $\eta > 0$ il **tasso di apprendimento**. Per ogni esempio
$(\mathbf{x}, y)$ si calcola la predizione $\hat{y}$ e si aggiornano pesi e bias:

$$
w_i \leftarrow w_i + \eta\,(y - \hat{y})\,x_i, \qquad
b \leftarrow b + \eta\,(y - \hat{y}).
$$

Il fattore $(y-\hat{y})$ vale $0$ quando la predizione è corretta (nessun
aggiornamento), $+1$ o $-1$ altrimenti. Vale la pena notare che qui $\eta$ è
cosmetico: partendo da $\mathbf{w}=\mathbf{0}$ e $b=0$ ogni aggiornamento è
proporzionale a $\eta$, quindi cambiarlo riscala $\mathbf{w}$ e $b$ dello stesso
fattore, e la decisione dipende solo dal **segno** di
$\mathbf{w}^\top\mathbf{x}+b$, che un riscalamento positivo non tocca. Con
$\eta=0{,}1$, $\eta=1$ o $\eta=7{,}3$ non cambia nemmeno una predizione.
Diventerà una scelta vera nel capitolo sulla backpropagation, dove la
correzione non sarà più proporzionale all'errore ma al gradiente di una loss.

Rosenblatt dimostrò il **teorema di convergenza del percettrone**: se i dati
sono linearmente separabili, l'algoritmo trova in un numero finito di passi un
iperpiano che li separa. Il teorema dice di più di quanto sembri, nella forma
che si deve a Novikoff {cite}`novikoff1962convergence`: il numero di correzioni
è al più $(R/\gamma)^2$, dove $R = \max_i \lVert\mathbf{x}_i\rVert$ è la norma
massima degli esempi e $\gamma$ il margine del miglior separatore. In quel
limite **non compaiono né il numero di esempi né la dimensione**: quel che conta
è quanto è sottile il corridoio fra le due classi, e raddoppiare il dataset non
raddoppia il lavoro. E non dice l'altra metà: l'iperpiano trovato è uno
qualunque fra quelli che separano, senza alcuna garanzia di margine, che è
esattamente la differenza con le SVM del capitolo precedente. Il seme
dell'apprendimento moderno è già qui, anche se la discesa del gradiente su loss
differenziabili verrà dopo.

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

# La porta logica AND (vale 1 solo se entrambi gli ingressi valgono 1):
# i suoi quattro casi si separano con una retta
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])
w, b = addestra(X, y_and)
print(gradino(X @ w + b))                   # -> [0 0 0 1]: ha imparato la AND
```

## Il muro dello XOR

```{figure} ../figures/perceptron-primo-inverno-ai.svg
:name: fig-inverni-ai
:alt: "In alto lo schema del neurone artificiale di Rosenblatt, con i tre ingressi pesati, il sommatore, la soglia e l'uscita 0 o 1. Sotto, una linea del tempo dal 1958 al 1980 con cinque tappe: il percettrone nel 1958, il Mark I nel 1960, il libro Perceptrons nel 1969, il rapporto Lighthill nel 1973, e il disgelo attorno al 1980; una banda colorata copre il primo inverno dell'AI, fra il 1974 e il 1980."
:width: 100%

Un limite matematico e le sue conseguenze storiche. I teoremi di Minsky e
Papert riguardavano un modello a uno strato; il gelo che ne seguì, e che la
linea del tempo colloca a metà anni Settanta, colpì l'intero campo. In mezzo
c'è il rapporto Lighthill del 1973, la stroncatura commissionata dal governo
britannico che porta ai tagli veri.
```

La sproporzione visibile in {numref}`fig-inverni-ai` fra la portata dei
risultati e l'ampiezza della reazione è una lezione che vale oltre questa
storia. I teoremi erano corretti e limitati; la loro lettura pubblica fu che le
reti neurali non funzionavano, e servì quasi un ventennio per rimediare.

E qui torna il conto lasciato in sospeso, che merita di essere raccontato per
quello che è, perché la versione corrente è comoda e sbagliata. Nel 1969 Marvin
Minsky e Seymour Papert pubblicano *Perceptrons*
{cite}`minsky1969perceptrons`. Il libro è ricordato per lo **XOR** ("o
esclusivo", che vale $1$ quando i due ingressi sono diversi e $0$ quando sono
uguali), e cioè per l'osservazione che un neurone solo non ce la fa. Quella
osservazione però era già nota, e i due autori la danno per nota: che una rete
di elementi a soglia possa calcolare qualunque funzione logica è in McCulloch e
Pitts, ventisei anni prima. Vediamo prima l'ostacolo come lo si racconta di
solito, poi che cosa dimostra davvero quel libro.

```{figure} ../figures/xor-non-separabile.svg
:name: fig-xor-non-separabile
:alt: "Animazione: i quattro casi dello XOR agli angoli di un quadrato, con gli assi segnati 0 e 1. In terracotta i due casi che vogliono uscita 1, in teal i due che vogliono uscita 0. Una retta ruota su di essi e a ogni orientamento due punti restano dalla parte sbagliata e vengono cerchiati."
:width: 85%

La stessa retta della figura precedente, questa volta sui quattro casi dello
XOR: gli assi portano i due ingressi, $0$ e $1$, e ogni angolo del quadrato è
uno dei quattro casi. Come là, il terracotta è la classe con uscita $1$. Per
quanto la si giri, la retta lascia sempre **due** punti dalla parte sbagliata.
```

Il contrasto con la {numref}`fig-percettrone-impara` è tutto il punto: là la
rotazione finiva, qui non finisce mai. La {numref}`fig-xor-non-separabile` non
prova il teorema (mostra solo alcuni orientamenti) ma rende evidente da dove
viene l'ostacolo: le due classi occupano angoli **opposti** del quadrato.

`````{tab} Elementare

Disegna i quattro casi sullo stesso foglio a quadretti di prima, il primo
ingresso in orizzontale e il secondo in verticale: vengono i quattro angoli di
un quadrato di lato $1$. I punti $(0,0)$ e $(1,1)$ vogliono uscita $0$; i punti
$(0,1)$ e $(1,0)$ vogliono uscita $1$. Prova a separarli con una sola riga
dritta: è impossibile. Le due classi stanno negli angoli opposti, "incrociate",
e una riga non sa scavalcare il centro per andarle a prendere tutte e due. Un
singolo percettrone traccia solo quella riga, quindi sullo XOR è condannato a
sbagliare almeno un caso.

`````

`````{tab} Superiore

Lo XOR **non è linearmente separabile**: non esiste alcun $(\mathbf{w}, b)$ tale
che $g(\mathbf{w}^\top\mathbf{x}+b)$ riproduca la tabella. Le classi $\{(0,0),
(1,1)\}$ e $\{(0,1),(1,0)\}$ non sono divisibili da un iperpiano in
$\mathbb{R}^2$. Non è un difetto dell'ottimizzatore, è un limite di *capacità*
del modello.

Vale la pena lanciare `addestra` su `y_xor = np.array([0, 1, 1, 0])` e guardare
che cosa succede, perché quello che si vede non è quello che ci si aspetta. Non
converge, e fin qui è ovvio; ma nemmeno diverge, e nemmeno vaga. Dalla seconda
epoca in poi tutti e quattro gli esempi vengono sbagliati, tutti e quattro
producono una correzione, e le quattro correzioni **si annullano fra loro**:
alla fine di ogni epoca i parametri sono tornati esattamente dov'erano
($\mathbf{w} = (-0{,}1,\ 0)$, $b = 0$), e l'uscita stampata è `[1 1 0 0]` alla
decima epoca come alla centesima. Da fuori una risposta immobile e sbagliata
(due casi su quattro), da dentro un ciclo. Non è un caso fortunato: è il
*perceptron cycling theorem*, enunciato nello stesso *Perceptrons* e dimostrato
per intero da Block e Levin {cite}`block1970boundedness`, che su dati non
separabili garantisce almeno che i pesi restino limitati.

`````

### Che cosa dimostra davvero *Perceptrons*

Lo XOR è il ricordo che è rimasto, ma non è il risultato. Nel libro il
percettrone non è il neurone di poco fa: è una somma pesata di **predicati**
qualsiasi, ciascuno dei quali però può guardare solo un pezzetto dell'immagine
in ingresso, e il numero di punti che il predicato più affamato deve guardare si
chiama **ordine**. I teoremi sono su quello. Il più celebre dice che per
calcolare la **parità** di $n$ bit (rispondere "quanti sono accesi, pari o
dispari?") serve ordine $n$: qualche predicato deve guardare **tutti** gli
ingressi in una volta sola, e non c'è modo di cavarsela con pezzetti. Un altro
dice che per decidere se una figura disegnata è tutta d'un pezzo o spezzata in
due, il numero di punti da guardare insieme cresce con la figura. Lo XOR è la
parità a due bit, cioè il caso più piccolo di una famiglia: non un impossibile,
ma il primo gradino di un costo che esplode. Il messaggio non era "una retta non
basta", era "questo modo di costruire le caratteristiche non scala".

E sulle reti a più strati, cioè sulla cosa per cui il libro è stato usato come
condanna, *Perceptrons* non dimostra niente, e lo dichiara: parla di un
"giudizio intuitivo" che l'estensione al multistrato sia sterile, e chiede
esplicitamente a qualcuno di confermarlo o smentirlo. È una congettura
dichiarata come tale, letta per vent'anni come un teorema. Il libro raffreddò
gli entusiasmi e contribuì a spostare risorse verso l'AI simbolica; ma la
ragione tecnica per cui le reti restarono ferme la indicarono gli stessi autori,
ed è quella che il libro non prova a nascondere: nessuno sapeva come correggere
i neuroni in mezzo. Ed è esattamente la via d'uscita.

## Oltre la linea: strati nascosti e non linearità

Se un neurone traccia una sola linea, mettiamone di più. Impilando i neuroni in
uno **strato nascosto** e componendo gli strati, la rete può piegare la
frontiera fino a separare anche lo XOR: un primo strato costruisce
rappresentazioni intermedie, un secondo le combina. C'è però una condizione non
negoziabile: fra uno strato e l'altro serve una **non linearità**. Il motivo
è che mettere in fila due passaggi che si limitano a moltiplicare e sommare dà
ancora un solo passaggio dello stesso tipo: cento strati senza attivazioni si
schiaccerebbero in un unico strato, con la sua unica riga dritta, di nuovo
incapace di XOR. È qui che entrano funzioni come la ReLU o la sigmoide, e con
esse il percettrone multistrato (MLP) e l'algoritmo che lo addestra, la
*backpropagation*: il tema delle prossime due sezioni.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un neurone artificiale è un pezzo di aritmetica: dà a ogni indizio in
  ingresso un **peso**, somma tutto, aggiunge la propria indole di partenza
  (il **bias**) e passa il totale a un interruttore che risponde sì o no.
- Il percettrone **impara sbagliando**: quando azzecca non tocca niente, quando
  dice "no" e doveva dire "sì" alza i pesi, e viceversa. Ogni peso si sposta in
  proporzione a quanto valeva il suo ingresso in quell'esempio: si corregge chi
  ha parlato più forte. Le correzioni sono piccole e si ripetono su tutti gli
  esempi.
- Un neurone solo sa tracciare **una riga dritta** fra le due classi, e il
  motivo è che somma e confronta con una soglia: i casi in cui il totale
  pareggia la soglia stanno tutti su una riga. Se quella riga esiste, la trova;
  ma sullo **XOR** non esiste, perché i casi da separare stanno negli angoli
  opposti del quadrato.
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
  $w_i \leftarrow w_i + \eta\,(y-\hat{y})\,x_i$. Se i dati sono separabili
  converge in al più $(R/\gamma)^2$ correzioni, un limite che **non dipende dal
  numero di esempi** ma dal margine $\gamma$; se non lo sono, i pesi non
  divergono, entrano in un ciclo.
- Un solo neurone è un **classificatore lineare**: separa lo spazio con un
  iperpiano e fallisce su problemi non separabili come lo **XOR**. Quel che
  *Perceptrons* dimostra però è sull'**ordine** dei predicati (la parità su $n$
  bit lo richiede pari a $n$); sul multistrato il libro avanza una congettura e
  lo dichiara.
- Servono **strati nascosti** e **non linearità** per superare quel limite: è
  il ponte verso le reti neurali profonde.
```

`````
