# Il percettrone e la sua regola di apprendimento

Nel 1958 uno psicologo di Cornell, Frank Rosenblatt, presenta alla stampa il
*percettrone* {cite}`rosenblatt1958perceptron`, un modello di neurone
artificiale che impara a riconoscere forme dagli esempi. Il *New York Times*
scrive che è l'embrione di un computer elettronico capace, un giorno, di
camminare, parlare e riprodursi. Due anni dopo l'idea prende corpo in una
macchina grande come un armadio, il *Mark I Perceptron*. Davanti c'è una
griglia di quattrocento fotocellule, che è l'occhio. Dietro, il filo di ogni
fotocellula finisce su una manopola, e girare quella manopola vuol dire
cambiare quanto conta ciò che quella fotocellula vede. A girarle, ogni volta
che la macchina sbaglia, è un motorino: l'apprendimento, lì, era fatto di
ferro. Il clamore era smisurato, e lo pagheremo caro qualche
pagina più avanti. Ma sotto c'è un'idea sobria e duratura, che ancora oggi è il
mattone di ogni rete neurale: un neurone artificiale non è altro che un pezzo
di aritmetica.

## Un neurone fatto di aritmetica

Un neurone biologico riceve segnali da altri neuroni, li combina e "scarica" un
impulso se lo stimolo complessivo supera una soglia. Rosenblatt cattura questa
idea con tre gesti: pesare gli ingressi, sommarli, decidere.

Ogni ingresso arriva con un **peso** che ne misura l'importanza: il primo
ingresso lo chiamiamo $x_1$ e il suo peso $w_1$, il secondo $x_2$ e $w_2$, e
avanti così (scrivere $x_i$ e $w_i$, con una lettera al posto del numero, è il
modo di dire «uno qualunque di loro»). Il neurone li combina in una somma
pesata e vi aggiunge un termine costante, il **bias** $b$, che è la sua
inclinazione di partenza. Poi il totale passa a un ultimo gesto, che decide sì
o no, e quel gesto si chiama **funzione di attivazione**
({numref}`fig-neurone`).

```{figure} ../figures/neurone-artificiale.svg
:name: fig-neurone
:alt: Gli ingressi x1, x2, fino a xn sono moltiplicati per i pesi w1, w2, wn e sommati in un nodo sigma che aggiunge il bias b; la somma passa in una funzione di attivazione a gradino e produce l'uscita, indicata con y col cappello.
:width: 90%

Il neurone artificiale: ogni ingresso ha il suo peso, il sommatore (la lettera
greca $\Sigma$, che in matematica vuol dire «somma tutto») mette insieme i
contributi e aggiunge il bias $b$, la funzione di attivazione decide l'uscita.
```

`````{tab} Elementare

Immagina di decidere se uscire di casa con l'ombrello. Guardi due indizi:
quanto è nuvoloso e cosa dice l'app del meteo. Dai a ciascun indizio un peso
(l'app conta più del colore del cielo) e fai una somma: indizio per il suo
peso, il tutto sommato. Il bias è la tua indole di partenza: un pessimista
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
è la più netta possibile: la funzione a **gradino** (detta anche di Heaviside,
dal nome del fisico inglese che la mise in uso).

`````{tab} Elementare

Il gradino è un interruttore: se la somma raggiunge la soglia, l'uscita è $1$
("sì"); altrimenti è $0$ ("no"). Niente sfumature, solo acceso o spento. Tornando
all'ombrello: sommati gli indizi, o esci con l'ombrello o non lo prendi.

Le manopole a questo punto sembrano tre (i pesi, il bias, la soglia), e invece
sono due: chiedere che il totale superi $3$ è la stessa identica cosa che
togliere $3$ dal totale e chiedere che superi lo zero. La soglia si può sempre
nascondere dentro il bias, e da qui in avanti è quello che faremo: la soglia è
sempre lo zero, e a spostare il punto in cui il neurone cambia idea è il bias.

Facciamolo subito, sui numeri dell'ombrello. Il bias era $0{,}5$ e la soglia
$3$, quindi il bias nuovo è $0{,}5 - 3 = -2{,}5$, e la regola diventa: «esci con
l'ombrello se $0{,}3 \cdot x_1 + 2 \cdot x_2 - 2{,}5$ è sopra lo zero». Rifai il
conto della giornata di prima e vedi che non è cambiato niente:
$0{,}3 \cdot 7 + 2 \cdot 1 - 2{,}5 = 1{,}6$, che è sopra zero, e l'ombrello lo
prendi come prima.

E adesso la cosa che vale la pena vedere con gli occhi, perché tutto il resto
del capitolo ci si appoggia. Prendi un foglio a quadretti e mettici i due
indizi dell'ombrello, la nuvolosità in orizzontale e l'app in verticale: ogni
giornata diventa un puntino. L'app ha due sole risposte, e anche quelle
diventano numeri: $x_2 = 1$ se dice pioggia, $x_2 = 0$ se dice sereno.

Cerchiamo adesso le giornate in bilico, quelle in cui il totale fa esattamente
zero. Con i numeri appena sistemati sono due conti di seconda media.

- App **sereno**, cioè $x_2 = 0$. Resta $0{,}3 \cdot x_1 - 2{,}5 = 0$, quindi
  $0{,}3 \cdot x_1 = 2{,}5$ e la nuvolosità in bilico è
  $x_1 = 2{,}5 : 0{,}3 = 8{,}33\ldots$, in pratica $8{,}3$: ci vuole quasi
  tutto il cielo grigio.
- App **pioggia**, cioè $x_2 = 1$. Il suo contributo è $2 \cdot 1 = 2$, e resta
  $0{,}3 \cdot x_1 + 2 - 2{,}5 = 0$, quindi $0{,}3 \cdot x_1 = 0{,}5$ e la
  nuvolosità in bilico è $x_1 = 0{,}5 : 0{,}3 = 1{,}66\ldots$, in pratica
  $1{,}7$: se l'app promette pioggia, basta molto meno.

Segna quei due punti sul foglio, uno a $8{,}3$ in basso e uno a $1{,}7$ in
alto, e tira una riga fra loro: da una parte della riga il neurone risponde
sempre sì, dall'altra sempre no, e non esiste una terza possibilità. **Quella
riga è tutto ciò che un neurone sa disegnare.** Cambiare i pesi la inclina,
cambiare il bias la sposta avanti e indietro, ma resta una riga dritta.

(Un dubbio legittimo, se hai davvero preso il foglio: l'app dice solo pioggia o
sereno, quindi tutte le giornate vere finiscono su due sole righe orizzontali,
e la riga di confine attraversa una fascia dove non c'è nessun puntino. Va
bene lo stesso: se al posto dell'app ci fosse un indizio che può valere
qualunque numero, come l'umidità, i puntini riempirebbero il foglio e il
confine sarebbe sempre quella riga lì.)

Fra qualche pagina questo dettaglio diventerà un muro.

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
un circuito che calcola e un modello che impara. Un dettaglio del disegno, per
chi lo nota: il bias qui non è tenuto da parte come nella figura precedente, ma
è disegnato come un ingresso in più che vale sempre $1$, con il suo peso $w_0$.
È la stessa cosa scritta in un altro modo, perché un peso moltiplicato per $1$
dà il peso stesso, e quindi $w_0$ fa esattamente il mestiere del bias. Comodo,
perché così anche il bias si corregge con la regola degli altri pesi, che è
quella della prossima pagina.
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
quattro vogliono uscita $1$ (i punti color terracotta, arancione di vaso) e
quattro uscita $0$ (i punti color teal, il verde-azzurro). Ciascuno dei due
gruppi si chiama **classe**. La retta parte sbagliata e a ogni punto messo dal
lato sbagliato ruota un po’. Dopo quattro correzioni le due classi sono
separate, e da lì in poi nessun esempio provoca più un aggiornamento.
```

Nella {numref}`fig-percettrone-impara` si vede la proprietà che rese famoso
l'algoritmo: **quando una retta separatrice esiste, il percettrone la trova in
un numero finito di correzioni**. È il **teorema di convergenza** di
Rosenblatt, dove convergere vuol dire che a un certo punto la ricerca si ferma,
invece di andare avanti per sempre. Attenzione però a quel "quando esiste":
fra poco diventerà il problema principale.

`````{tab} Elementare

La ricetta è quasi banale. Per ogni esempio:

- se il neurone azzecca la risposta, non tocchi nulla;
- se dice $0$ e doveva dire $1$, alzi i pesi;
- se dice $1$ e doveva dire $0$, li abbassi.

Di quanto? In proporzione a quanto valeva quell'ingresso nell'esempio appena
sbagliato: chi valeva zero non si muove per niente (non aveva colpa, non ha
detto la sua), chi valeva molto si muove molto. È il criterio più naturale del
mondo: si corregge chi ha parlato più forte.

Ogni correzione poi è piccola, perché la si moltiplica per un numeretto scelto
da noi, il **passo di apprendimento** (in inglese *learning rate*, e nel codice
qui sotto si chiama `eta`): così la macchina non "salta" da una parte
all'altra, ma si assesta gradualmente.

Vediamola su numeri veri. Passo di apprendimento $0{,}1$; l'esempio ha due
ingressi, $x_1 = 1$ e $x_2 = 0$; entrambi i pesi partono da zero; il neurone ha
detto $0$ e doveva dire $1$. Allora il primo peso sale di $0{,}1 \cdot 1 =
0{,}1$ e il secondo di $0{,}1 \cdot 0 = 0$, cioè non si muove affatto: i due
pesi diventano $0{,}1$ e $0$. Il secondo ingresso valeva zero, non ha detto
niente, e non paga niente.

E il bias? Si corregge anche lui, con la stessa regola, comportandosi come il
peso di un ingresso che vale sempre $1$: quindi si sposta ogni volta del passo
intero, in su quando la risposta era troppo bassa e in giù quando era troppo
alta. Nel codice qui sotto è la riga `b += eta * errore`.

Poi si ripete su tutti gli esempi, più volte: un giro completo su tutti gli
esempi si chiama **epoca**, ed è la parola che nel codice qui sotto dà il nome
a `epoche`.

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
Diventerà una scelta vera nella sezione sulla backpropagation, dove la
correzione non sarà più proporzionale all'errore ma al gradiente di una loss.

C'è poi il **teorema di convergenza del percettrone**: se i dati sono
linearmente separabili, l'algoritmo trova in un numero finito di passi un
iperpiano che li separa. Rosenblatt lo dimostra in *Principles of
Neurodynamics* (1962) {cite}`rosenblatt1962principles`, non nell'articolo del
1958 che presenta il modello. Il teorema dice di più di quanto sembri, nella
forma che si deve a Novikoff {cite}`novikoff1962convergence`: il numero di
correzioni è al più $(R/\gamma)^2$, dove $R = \max_i \lVert\mathbf{x}_i\rVert$ è
la norma massima degli esempi e $\gamma$ il margine **geometrico** del miglior
separatore, cioè la distanza fra quell'iperpiano e il punto più vicino
($\gamma = \min_i |\mathbf{w}^{*\top}\mathbf{x}_i|$ con
$\lVert\mathbf{w}^*\rVert = 1$: senza quel vincolo il rapporto non sarebbe
nemmeno un numero puro, perché basterebbe raddoppiare $\mathbf{w}^*$ per
raddoppiare $\gamma$). Il limite vale per la versione **senza bias**, o con il
bias assorbito come ingresso costante: è il trucco della
{numref}`fig-neurone-con-retroazione`, l'ingresso sempre pari a $1$, e serviva
proprio qui (assorbito il bias, quella coordinata in più entra anche in $R$). In
quel limite **non compaiono né il numero di esempi né la dimensione**: quel che
conta è quanto è sottile il corridoio fra le due classi, e raddoppiare il
dataset non raddoppia il lavoro. E non dice l'altra metà: l'iperpiano trovato è uno
qualunque fra quelli che separano, senza alcuna garanzia di margine, che è
esattamente la differenza con le SVM del capitolo precedente. Il seme
dell'apprendimento moderno è già qui, anche se la discesa del gradiente su loss
differenziabili verrà dopo.

`````

Tradotta in NumPy (la libreria di calcolo del capitolo su Python), la ricetta è
quasi identica a come l'abbiamo raccontata:

```python
import numpy as np

def gradino(z):
    return np.where(z >= 0, 1, 0)          # decisione binaria 0/1

def addestra(X, y, eta=0.1, epoche=10):
    w = np.zeros(X.shape[1])               # pesi iniziali a zero
    b = 0.0
    for _ in range(epoche):
        for xi, target in zip(X, y):
            # la chiocciola @ è la somma pesata: w1*x1 + w2*x2 + ...
            pred = gradino(w @ xi + b)
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

Ed eccolo, il muro annunciato qualche pagina fa. Merita di essere raccontato
per quello che è, perché la versione che si sente di solito è comoda e
sbagliata.

Nel 1969 Marvin Minsky e Seymour Papert pubblicano *Perceptrons*
{cite}`minsky1969perceptrons`. Il libro è ricordato per lo **XOR** ("o
esclusivo", che vale $1$ quando i due ingressi sono diversi e $0$ quando sono
uguali), e cioè per l'osservazione che un neurone solo non ce la fa. Ma quella
osservazione era già nota, e i due autori la danno per nota: che un elemento a
soglia da solo tracci una riga e nient'altro si sapeva da decenni, ed è il
motivo per cui già nel 1943, in McCulloch e Pitts, per calcolare le funzioni
logiche gli elementi si **collegavano fra loro** invece di usarne uno. Nel 1969
era materia da manuale. Vediamo prima l'ostacolo come lo si racconta di solito,
poi che cosa dimostra davvero quel libro.

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
è una dimostrazione, perché mostra alcune inclinazioni e non tutte quelle
possibili; ma rende evidente da dove viene l'ostacolo: le due classi occupano
angoli **opposti** del quadrato.

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
converge, e fin qui è ovvio; ma nemmeno diverge, e nemmeno vaga. Le prime due
epoche sbagliano tre esempi su quattro; dalla **terza** in poi li sbagliano
tutti e quattro, tutti e quattro producono una correzione, e le quattro
correzioni **si annullano fra loro**: alla fine di ogni epoca i parametri sono
tornati esattamente dov'erano ($\mathbf{w} = (-0{,}1,\ 0)$, $b = 0$), e l'uscita
stampata è `[1 1 0 0]` alla decima epoca come alla centesima.

Le quattro correzioni si seguono sul foglio, e conviene farlo perché è il modo
più rapido per convincersi che il ciclo non è un caso fortunato. Si parte da
$\mathbf{w} = (-0{,}1,\ 0)$ e $b = 0$. Il primo esempio, $(0,0)$, riceve $1$ e
doveva ricevere $0$: la correzione tocca **solo il bias**, perché ogni peso si
muove in proporzione al proprio ingresso e qui gli ingressi valgono zero, e $b$
scende a $-0{,}1$. Il secondo, $(0,1)$, e il terzo, $(1,0)$, ricevono $0$ e
dovevano ricevere $1$: ciascuno alza di $0{,}1$ il peso del proprio ingresso
acceso, e ciascuno rialza il bias. Il quarto, $(1,1)$, riceve $1$ e doveva
ricevere $0$: riabbassa entrambi i pesi e riporta il bias dov'era. Fine
dell'epoca, e siamo al punto di partenza. Da fuori una risposta immobile e
sbagliata (due casi su quattro), da dentro un ciclo. Non è un caso fortunato: è
il *perceptron cycling theorem*, enunciato nello stesso *Perceptrons* e
dimostrato per intero da Block e Levin {cite}`block1970boundedness`, che su dati
non separabili garantisce almeno che i pesi restino limitati.

`````

### Che cosa dimostra davvero *Perceptrons*

Lo XOR è il ricordo che è rimasto, ma non è il risultato.

`````{tab} Elementare

Prima di tutto, una parola che cambia significato. Nel loro libro «percettrone»
non indica il neurone di poco fa, ma una macchina più generale, e conviene
saperlo, altrimenti i loro risultati sembrano parlare di una cosa che non è.

Immagina una fotografia e una squadra di ispettori: ciascuno può guardare
**solo qualche punto** dell'immagine e risponde sì o no, poi un capo raccoglie
le risposte, dà a ognuna un peso, somma e decide. Il neurone di poco fa è il
caso più semplice di questa macchina, quello in cui ogni ispettore guarda un
punto solo. La domanda dei teoremi non è se la squadra ce la fa, ma **quanti
punti deve guardare in una volta sola l'ispettore più affamato**: quel numero
si chiama **ordine**, e misura quanto il problema si lascia dividere in
pezzetti.

C'è un compito in cui va malissimo: dire se i punti accesi sono in numero pari
o dispari (i matematici lo chiamano la **parità**). Qui nessun ispettore può
accontentarsi della propria zona, perché accendere o spegnere un punto
qualunque, in un angolo qualunque, ribalta la risposta.

Verrebbe da obiettare: e se ogni ispettore dicesse «nella mia zona gli accesi
sono pari», lasciando al capo il compito di mettere insieme? Non funziona,
perché il capo non ragiona: sa fare una cosa sola, sommare le risposte con dei
pesi e confrontare il totale con una soglia. Combinare «pari» e «pari» e
«dispari» per sapere com'è il totale è di nuovo lo stesso problema di partenza,
e una somma pesata non lo risolve. Non resta che guardare l'immagine intera in
un colpo solo, e l'ordine è grande quanto l'immagine. Un altro compito
difficile è dire se una figura disegnata è tutta d'un pezzo o spezzata in due:
più la figura è grande, più punti bisogna guardare insieme.

E lo XOR è il caso più piccolo della parità, quello con due punti soli. Provaci:
nessuno acceso fa zero, che è pari, e la risposta è no; uno acceso solo fa uno,
dispari, risposta sì; tutti e due accesi fa due, di nuovo pari, di nuovo no. È
esattamente la tabella dello XOR. Ed è anche per questo che è rimasto nella
memoria di tutti: si disegna su un foglio in un secondo.

Ma è la punta di una famiglia, e la conclusione vera è più interessante di un
semplice «non si può». Non «una riga non basta», bensì: **mettere insieme la risposta a
partire da ispettori che guardano ciascuno il proprio pezzetto funziona sui
casi piccoli e diventa impraticabile appena il problema cresce.** Che è un
difetto peggiore, perché non si vede finché non si prova a ingrandire.

`````

`````{tab} Superiore

Nel libro il percettrone non è il neurone di poco fa: è una somma pesata di
**predicati**
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

`````

E sulle reti a più strati, cioè sulla cosa per cui il libro è stato usato come
condanna, *Perceptrons* non dimostra niente, e lo dichiara: parla di un
"giudizio intuitivo" che l'estensione al multistrato sia sterile, e chiede
esplicitamente a qualcuno di confermarlo o smentirlo. È una **congettura**,
cioè un sospetto che nessuno ha ancora dimostrato, dichiarata come tale e letta
per vent'anni come se fosse un teorema.

Quello che venne dopo lo riassume la {numref}`fig-inverni-ai`.

```{figure} ../figures/perceptron-primo-inverno-ai.svg
:name: fig-inverni-ai
:alt: "In alto lo schema del neurone artificiale di Rosenblatt, con i tre ingressi pesati, il sommatore, la soglia e l'uscita 0 o 1. Sotto, una linea del tempo dal 1958 al 1980 con cinque tappe: il percettrone nel 1958, il Mark I nel 1960, il libro Perceptrons nel 1969, il rapporto Lighthill nel 1973, e il disgelo attorno al 1980; una banda colorata copre il primo inverno dell'AI, fra il 1974 e il 1980."
:width: 100%

In alto il neurone di Rosenblatt; sotto, vent'anni di storia in cinque tappe.
La banda colorata è il primo **inverno dell'AI**: gli anni in cui i
finanziamenti si ritirarono e il campo quasi si fermò. In mezzo c'è il rapporto
Lighthill del 1973, la stroncatura commissionata dal governo britannico che
porta ai tagli veri.
```

Seguirono anni magri, che oggi si chiamano il primo **inverno dell'AI**: i
finanziamenti si ritirarono, i gruppi di ricerca si svuotarono e il campo quasi
si fermò. A far scattare i tagli veri, però, non fu questo libro, e non fu
subito: fu il **rapporto Lighthill** del 1973, una stroncatura commissionata dal
governo britannico che riguardava l'intelligenza artificiale tutta intera, reti
neurali o no.

La sproporzione fra la portata dei risultati e l'ampiezza della reazione è una
lezione che vale oltre questa storia. I teoremi erano corretti e limitati, e
riguardavano macchine a uno strato; la loro lettura pubblica fu che le reti
neurali non funzionavano, e servì quasi un ventennio per rimediare. Il libro
contribuì a spostare risorse verso l'altro modo di fare intelligenza
artificiale, quello dei programmi a regole scritte a mano (l’**AI simbolica**);
ma la ragione tecnica per cui le reti restarono ferme la indicarono gli stessi
autori, ed è quella che il libro non prova a nascondere: nessuno sapeva come
correggere i neuroni in mezzo. È da lì che sarebbe arrivata la via d'uscita.

## Oltre la linea: strati nascosti e non linearità

Se un neurone traccia una sola riga, mettiamone di più. Un primo strato di
neuroni traccia più righe insieme, e si chiama **strato nascosto** perché non
si affaccia né sull'ingresso né sull'uscita: lavora in mezzo. Un secondo strato
poi lavora sulle risposte del primo, e il confine che ne esce non è più una
riga sola. Tanto basta per lo XOR, e fra poco lo vediamo disegnato.

C'è però una condizione non negoziabile, la stessa già annunciata
nell'introduzione del capitolo: fra uno strato e l'altro deve succedere
qualcosa che non sia moltiplicare e sommare. Il motivo è che due passaggi di
sola moltiplicazione e somma, messi in fila, danno ancora un passaggio dello
stesso tipo: se il primo strato moltiplica per $2$ e il secondo per $3$,
insieme moltiplicano per $6$, e un neurone che moltiplica per $6$ sa fare
esattamente quello che sapeva fare prima, cioè una riga dritta. Cento strati
così si schiaccerebbero in uno solo, di nuovo incapace di XOR. Quel qualcosa da
mettere in mezzo si chiama **non linearità**, e sono funzioni come la ReLU o la
sigmoide: sono loro, insieme al percettrone multistrato (MLP) e all'algoritmo
che lo addestra, la *backpropagation*, il tema delle prossime due sezioni.

La {numref}`fig-xor-si-piega` fa vedere il passaggio per intero, ed è la
risposta che aspettavamo da tre pagine: lo XOR risolto. Vale la pena capire il
trucco, perché è lo stesso di tutto il deep learning.

I due neuroni del primo strato tracciano due righe parallele, e lasciano fra
loro una fascia. Dentro la fascia stanno i due casi che vogliono risposta $1$;
fuori, uno da una parte e uno dall'altra, i due che vogliono risposta $0$.

Una precisazione prima dei numeri: questi due neuroni non usano l'interruttore
secco di prima. Usano la ReLU, che lascia passare il totale quando è positivo e
dà zero quando è negativo, quindi la loro risposta non è solo «sì o no», è
«quanto». Le due regole si leggono in fondo alla figura, e sono
$h_1 = \mathrm{ReLU}(x_1 + x_2 - 0{,}5)$ e
$h_2 = \mathrm{ReLU}(-x_1 - x_2 + 1{,}5)$. Applicate ai quattro casi danno
questo:

| ingresso | risposta voluta | $h_1$ | $h_2$ |
|---|---|---|---|
| $(0,0)$ | $0$ | $0$ | $1{,}5$ |
| $(0,1)$ | $1$ | $0{,}5$ | $0{,}5$ |
| $(1,0)$ | $1$ | $0{,}5$ | $0{,}5$ |
| $(1,1)$ | $0$ | $1{,}5$ | $0$ |

Guarda le ultime due colonne, perché è lì che succede tutto. I due casi con
risposta $1$ danno la stessa identica coppia, $(0{,}5;\ 0{,}5)$: erano due punti
diversi e adesso sono lo stesso punto. I due casi con risposta $0$ danno
$(0;\ 1{,}5)$ e $(1{,}5;\ 0)$, cioè due punti lontani e da parti opposte.
Adesso prendi un foglio nuovo, mettici $h_1$ in orizzontale e $h_2$ in
verticale, segna i tre punti: una riga sola li separa, e a tracciarla è il
neurone di uscita. Il primo strato non ha risolto il problema: lo ha
**spostato** in un posto dove era facile, ed è questo il mestiere degli strati
nascosti in tutto il libro.

```{figure} ../figures/xor-si-piega.svg
:name: fig-xor-si-piega
:alt: "Due pannelli affiancati. A sinistra i quattro casi dello XOR agli angoli del quadrato unitario, in terracotta i due con uscita 1 e in teal i due con uscita 0, tagliati da due rette parallele, i due neuroni nascosti, che lasciano in mezzo una fascia con i soli punti terracotta. A destra gli stessi quattro punti si spostano nelle coordinate calcolate da quei neuroni: i due terracotta finiscono esattamente nello stesso posto e i due teal ai lati opposti, e a quel punto una sola retta li separa."
:width: 95%

Il seguito della {numref}`fig-xor-non-separabile`, cioè lo XOR risolto. A
sinistra il piano di partenza: il primo strato sono due neuroni, quindi **due**
rette invece di una, e la fascia che lasciano in mezzo contiene i due casi con
uscita $1$. A destra gli stessi quattro punti ridisegnati nelle coordinate
$(h_1, h_2)$ che quei due neuroni calcolano: i due casi con uscita $1$ sono
finiti nello stesso posto, i due con uscita $0$ ai lati opposti, e lì il
neurone di uscita li separa con una retta sola. I due neuroni nascosti sono
scelti a mano, e le loro formule si leggono in fondo alla figura; i pesi del
neurone d'uscita no, li trova la discesa del gradiente, il metodo di
aggiustamento automatico di cui parla la sezione sulla backpropagation. Il
programma che
disegna la figura esegue poi la rete sui quattro ingressi e controlla che
risponda $0, 1, 1, 0$.
```

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
  *Perceptrons* dimostra però è sull’**ordine** dei predicati (la parità su $n$
  bit lo richiede pari a $n$); sul multistrato il libro avanza una congettura e
  lo dichiara.
- Servono **strati nascosti** e **non linearità** per superare quel limite: è
  il ponte verso le reti neurali profonde.
```

`````
