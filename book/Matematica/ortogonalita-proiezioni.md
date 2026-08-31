# Ortogonalità e proiezioni: la risposta migliore quando non ce n'è una esatta

Dall'osservatorio di Palermo, nella prima notte dell'Ottocento, Giuseppe
Piazzi punta il telescopio verso il Toro e nota un puntino che nelle carte non
c'è. Lo segue per una quarantina di notti, prende ventiquattro misure di
posizione, poi il corpo celeste passa dietro il Sole e sparisce. Per ritrovarlo
bisogna prevedere dove riemergerà, e per prevederlo bisogna ricavarne l'orbita
da quelle ventiquattro misure.

Il problema è mal messo. Un'orbita si descrive con sei numeri, e le misure sono
ventiquattro: ci sono quattro volte più equazioni che incognite, e nessuna
orbita passa esattamente per tutti i punti osservati, perché ogni misura porta
con sé il suo errore. Il ventiquattrenne Carl Friedrich Gauss risolve il
problema cambiando la domanda: se nessuna orbita azzera gli scarti, si prende
quella che rende **minima la somma dei loro quadrati**. Cerere viene ritrovata
in dicembre, quasi dove il conto diceva.

Il metodo, pubblicato da Adrien-Marie Legendre nel 1805 e da Gauss nel 1809,
si chiama dei **minimi quadrati**, ed è il primo attrezzo statistico della
storia a diventare di uso quotidiano. Quello che nessuno dei due scrisse, e che
oggi si insegna per primo, è che quel minimo ha una forma geometrica semplice:
è un'ombra.

## Perpendicolare vuol dire che non si dicono niente

Il prodotto scalare della sezione sull'algebra lineare misura quanto due
vettori vanno d'accordo, e ha un valore speciale che merita un nome.

`````{tab} Elementare

Se il prodotto scalare di due liste di numeri fa **zero**, le due frecce
corrispondenti sono perpendicolari. Il nome tecnico è **ortogonali**, dal greco
per «ad angolo retto», e la cosa importante è quello che significa in pratica:
spostarsi lungo una delle due non ti sposta di un millimetro lungo l'altra.

Un esempio con due frecce sul foglio: $(3, 0)$ punta a destra, $(0, 5)$ punta
in alto, il loro prodotto scalare fa $3\cdot 0 + 0\cdot 5 = 0$. Quanto vai a est
e quanto vai a nord sono due conti separati, e ciascuno si fa senza guardare
l'altro. Senza tirare la conclusione più grossa: due direzioni perpendicolari
non si intralciano, e questo non vuol dire che due grandezze misurate lungo di
esse non abbiano niente da dirsi.

Quando in un gruppo le direzioni sono tutte perpendicolari a due a due, e
lunghe uno, il gruppo si dice **ortonormale** e diventa comodissimo. Il motivo
è che ciascuna direzione si può interrogare da sola. Per sapere quanto di una
certa direzione c'è dentro un vettore basta fare il prodotto scalare con quella
direzione, e non serve sapere niente delle altre: gli assi di una carta
stradale funzionano così, «quanto a est» si legge senza guardare «quanto a
nord».

Con direzioni oblique il gioco si rompe. Se due assi formano un angolo di
trenta gradi, chiedere «quanto ce n'è del primo?» dà una risposta che dipende
da quanto ce n'è del secondo, e per districarli tocca risolvere un sistema.
Ecco perché, quando si può scegliere, si scelgono assi perpendicolari: fanno
risparmiare tutto il lavoro della sezione precedente.

E c'è un secondo motivo, meno ovvio e più importante quando i conti li fa una
macchina. Girare la carta stradale, tenendo i due assi perpendicolari fra loro,
non cambia nessuna distanza e nessun angolo: le città restano lontane
esattamente quanto prima, cambia solo come si chiamano le loro coordinate. Un
cambio di assi perpendicolari non gonfia niente e non schiaccia niente, quindi
non introduce errori di suo, ed è il motivo per cui gli algoritmi numerici
seri sono costruiti a partire da rotazioni.

`````

`````{tab} Superiore

Due vettori $\mathbf{u},\mathbf{v}\in\mathbb{R}^n$ sono **ortogonali** se
$\mathbf{u}^\top\mathbf{v}=0$. Un insieme $\{\mathbf{q}_1,\dots,\mathbf{q}_k\}$
è **ortonormale** se

$$
\mathbf{q}_i^\top \mathbf{q}_j = \delta_{ij}
= \begin{cases} 1 & i = j\\ 0 & i \neq j,\end{cases}
$$

dove $\delta_{ij}$ è il simbolo di Kronecker. Un insieme ortonormale è sempre
linearmente indipendente: prendendo il prodotto scalare di una combinazione
nulla con $\mathbf{q}_i$ resta $c_i = 0$.

La comodità operativa sta tutta in una formula. Se $\{\mathbf{q}_i\}$ è una
base ortonormale di un sottospazio e $\mathbf{v}$ appartiene a quel
sottospazio, allora

$$
\mathbf{v} = \sum_{i=1}^{k} (\mathbf{q}_i^\top\mathbf{v})\,\mathbf{q}_i ,
$$

cioè i coefficienti si **leggono** con un prodotto scalare ciascuno, invece di
risolvere un sistema. Con una base qualunque $\{\mathbf{a}_i\}$ i coefficienti
sono la soluzione di $\mathbf{A}\mathbf{c}=\mathbf{v}$, e costano
$\Theta(k^3)$.

Una matrice quadrata $\mathbf{Q}$ le cui colonne sono ortonormali si dice
**ortogonale** e soddisfa $\mathbf{Q}^\top\mathbf{Q}=\mathbf{I}$, quindi
$\mathbf{Q}^{-1}=\mathbf{Q}^\top$: l'inversa si ottiene trasponendo, senza
calcoli. Le trasformazioni ortogonali conservano prodotti scalari, lunghezze e
angoli ($\lVert\mathbf{Q}\mathbf{x}\rVert_2 = \lVert\mathbf{x}\rVert_2$), cioè
sono rotazioni e riflessioni. Sono anche perfettamente condizionate: detto
$\kappa_2(\mathbf{A})=\sigma_{\max}/\sigma_{\min}$ il rapporto fra il massimo e
il minimo allungamento, che si chiama **numero di condizionamento**, per una
matrice ortogonale tutti i valori singolari valgono uno e quindi
$\kappa_2(\mathbf{Q})=1$. Ed è la ragione per cui gli algoritmi numerici seri
sono costruiti a partire da esse.

`````

## L'ombra di un vettore su una direzione

L'immagine della lampada della sezione precedente torna qui, con una sola
differenza: il sole a picco, cioè i raggi tutti perpendicolari al muro. Questa
è la **proiezione ortogonale**, ed è l'ombra che tutti disegnano quando dicono
«ombra».

`````{tab} Elementare

Tieni una matita obliqua sopra un tavolo, con il sole esattamente a picco.
L'ombra sul tavolo è più corta della matita, e sta tutta sul tavolo: è la parte
della matita che il tavolo «vede». La parte che sparisce è quella verticale,
cioè quella nella direzione perpendicolare al tavolo.

Questa divisione in due pezzi è la cosa da portare via. Ogni vettore si spacca
in modo unico in **quello che sta nella direzione scelta** più **quello che le
è perpendicolare**, e i due pezzi non si parlano. Il primo si chiama
proiezione, il secondo scarto (o residuo).

Un caso di questa mossa lo usi già tutti i giorni senza saperlo. Hai misurato
cinque volte la stessa cosa, e ti servono cinque numeri riassunti in uno solo.
La lista delle cinque misure è un vettore in uno spazio a cinque dimensioni; i
vettori «tutte e cinque le misure uguali fra loro» formano una direzione sola,
quella di $(1,1,1,1,1)$. Cercare il numero unico che rappresenta meglio le
cinque misure significa cercare, su quella direzione, il punto più vicino alla
lista vera: cioè farne l'ombra.

Il conto dà esattamente la **media aritmetica**, e la scena spiega perché: la
media è l'ombra delle misure sulla direzione in cui tutte le misure sono
uguali. Lo
scarto, cioè quello che l'ombra non registra, è la lista delle differenze fra
ciascuna misura e la media, ed essendo perpendicolare a $(1,1,1,1,1)$ ha somma
zero. È la ragione per cui gli scarti dalla media si compensano sempre, cosa
che di solito si impara come una curiosità e che qui diventa geometria.

E c'è un terzo pezzo di regalo. La lunghezza al quadrato dello scarto è la
somma dei quadrati delle differenze dalla media, cioè quella che in statistica
si chiama variabilità dei dati. Media e variabilità, che sembrano due nozioni
separate, dicono l'una dove cade l'ombra e l'altra di quanto i dati ne
sporgono.

`````

`````{tab} Superiore

Sia $\mathbf{a}\neq\mathbf{0}$. La **proiezione ortogonale** di $\mathbf{b}$
sulla retta generata da $\mathbf{a}$ è

$$
\operatorname{proj}_{\mathbf{a}}(\mathbf{b})
= \frac{\mathbf{a}^\top\mathbf{b}}{\mathbf{a}^\top\mathbf{a}}\,\mathbf{a},
$$

dove il coefficiente $c=(\mathbf{a}^\top\mathbf{b})/(\mathbf{a}^\top\mathbf{a})$
si ricava imponendo che il residuo $\mathbf{r}=\mathbf{b}-c\,\mathbf{a}$ sia
ortogonale ad $\mathbf{a}$:

$$
\mathbf{a}^\top(\mathbf{b}-c\,\mathbf{a}) = 0
\;\Longleftrightarrow\;
c = \frac{\mathbf{a}^\top\mathbf{b}}{\mathbf{a}^\top\mathbf{a}} .
$$

La decomposizione $\mathbf{b}=\operatorname{proj}_{\mathbf{a}}(\mathbf{b}) +
\mathbf{r}$ è unica, e i due addendi sono ortogonali, quindi vale il teorema di
Pitagora: $\lVert\mathbf{b}\rVert^2 =
\lVert\operatorname{proj}_{\mathbf{a}}(\mathbf{b})\rVert^2 +
\lVert\mathbf{r}\rVert^2$.

Con $\mathbf{a}=\mathbf{1}=(1,\dots,1)^\top\in\mathbb{R}^n$ si ottiene

$$
c = \frac{\mathbf{1}^\top\mathbf{y}}{\mathbf{1}^\top\mathbf{1}}
= \frac{1}{n}\sum_{i=1}^{n} y_i = \bar{y},
$$

cioè la media campionaria, e il residuo $\mathbf{y}-\bar{y}\mathbf{1}$ ha per
costruzione somma nulla. La sua norma al quadrato è
$\sum_i (y_i-\bar{y})^2 = n\,\hat{\sigma}^2$, con $\hat{\sigma}^2$ la varianza
calcolata dividendo per $n$. La media è quindi il **coefficiente** dell'ombra
lungo $\mathbf{1}$, e la varianza è la lunghezza al quadrato di ciò che resta
divisa per $n$: la
{doc}`sezione su probabilità e statistica </Matematica/probabilita-statistica>`
le ritroverà per la strada probabilistica, e i due conti danno gli stessi
numeri.

`````

```python
import numpy as np

# cinque misure della stessa grandezza, in metri
y = np.array([2.03, 1.98, 2.05, 1.99, 2.00])
uno = np.ones(5)

c = (uno @ y) / (uno @ uno)      # la proiezione su (1,1,1,1,1)
print(c, y.mean())               # -> 2.01 2.01

r = y - c * uno                       # lo scarto, perpendicolare a uno
print(round(r.sum(), 12), round(uno @ r, 12))   # -> 0.0 0.0
print(round(r @ r, 10), round(5 * y.var(), 10))  # -> 0.0034 0.0034
```

## Proiettare su un piano, e perché quella è la risposta migliore

Una direzione sola basta di rado. Il caso che interessa è quello in cui le
direzioni disponibili sono parecchie, e insieme formano un sottospazio: nel
linguaggio della sezione precedente, l’**immagine** di una matrice.

```{figure} ../figures/proiezione-minimi-quadrati.svg
:name: fig-proiezione
:alt: "Un piano disegnato in prospettiva rappresenta l'immagine della matrice, cioè tutte le combinazioni delle sue colonne, disegnate come due frecce che giacciono sul piano. Una freccia b parte dall'origine e punta fuori dal piano. La sua ombra sul piano è la freccia p, e il segmento verticale che congiunge la punta di b alla punta di p, marcato con un angolo retto, è il residuo. Un altro punto del piano, diverso da p, è collegato alla punta di b da un segmento visibilmente più lungo."
:width: 85%

Il vettore dei dati $\mathbf{b}$ sporge fuori dal piano delle combinazioni
possibili, e nessuna scelta dei coefficienti lo raggiunge. Il punto del piano
più vicino è la sua ombra $\mathbf{p}$: qualunque altro punto del piano forma
con $\mathbf{b}$ l'ipotenusa di un triangolo rettangolo, e quindi dista di più.
```

`````{tab} Elementare

Torniamo al colorificio della sezione precedente, con una differenza: il colore
del cliente **non si può fare**. I barattoli disponibili producono un
repertorio di tinte, e quella richiesta ne sta fuori. Mandare via il cliente
sarebbe scortese, e la mossa sensata è un'altra: fra tutte le tinte che si
possono davvero preparare, scegliere quella che gli somiglia di più.

{numref}`fig-proiezione` disegna la scena. Il piano è il repertorio, la freccia
che sporge è il colore chiesto, e la sua ombra a picco sul piano è la risposta.
Che sia proprio l'ombra il punto più vicino, e non un altro punto del piano, si
vede con il teorema di Pitagora: prendendo un punto qualsiasi del repertorio
diverso dall'ombra, la distanza dal colore chiesto è l'ipotenusa di un
triangolo rettangolo che ha per cateto quella distanza minima. E l'ipotenusa è
sempre più lunga del cateto.

Da qui viene una regola pratica che si usa senza pensarci. **Quando la
soluzione migliore è stata trovata, l'errore che resta è perpendicolare a tutto
ciò che si poteva fare.** Se non lo fosse, l'errore avrebbe una componente
lungo una direzione disponibile, e spostandosi un po' in quella direzione si
farebbe meglio: segno che non si era ancora al minimo. Perpendicolarità
dell'errore ed essere arrivati al minimo sono la stessa cosa detta in due modi.

Da questa geometria discendono due conseguenze. La prima:
rifare l'ombra di un'ombra non cambia niente, perché l'ombra sta già distesa
sul piano e la sua ombra è sé stessa. La seconda riguarda una scelta che finora
è passata liscia. Prendere la somma dei **quadrati** degli scarti, invece della
somma dei loro valori assoluti, è una decisione, e ha un prezzo: un errore
doppio pesa quattro volte, quindi un solo dato molto sbagliato tira la risposta
verso di sé molto più di dieci dati sbagliati di poco. In compenso i quadrati
danno l'ombra, cioè una risposta con una formula chiusa e una geometria
limpida, mentre i valori assoluti non hanno né l'una né l'altra.

Ed è per questa strada che il problema dell'orbita di Cerere si chiude. Le
misure di Piazzi sono la freccia che sporge, le orbite possibili sono il piano,
e l'orbita di Gauss è l'ombra: quella che lascia uno scarto perpendicolare a
tutte le orbite disponibili, cioè uno scarto che nessun ritocco può ridurre.

`````

`````{tab} Superiore

Sia $\mathbf{A}\in\mathbb{R}^{m\times n}$ con $m>n$ e
$\mathbf{b}\in\mathbb{R}^m$. Il sistema $\mathbf{A}\mathbf{x}=\mathbf{b}$ è in
generale incompatibile, e si sostituisce il problema di minimo

$$
\hat{\mathbf{x}} = \arg\min_{\mathbf{x}\in\mathbb{R}^n}
\lVert \mathbf{A}\mathbf{x} - \mathbf{b}\rVert_2^2 .
$$

**Teorema della proiezione.** Il minimo si raggiunge quando il residuo
$\mathbf{r}=\mathbf{b}-\mathbf{A}\hat{\mathbf{x}}$ è ortogonale a
$\operatorname{im}(\mathbf{A})$, cioè a tutte le colonne insieme:
$\mathbf{A}^\top\mathbf{r}=\mathbf{0}$. Sviluppando si ottengono le
**equazioni normali**

$$
\mathbf{A}^\top\mathbf{A}\,\hat{\mathbf{x}} = \mathbf{A}^\top\mathbf{b},
$$

e se $\mathbf{A}$ ha rango pieno per colonne, $\mathbf{A}^\top\mathbf{A}$ è
invertibile e la soluzione è unica:
$\hat{\mathbf{x}} = (\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top\mathbf{b}$.
La proiezione stessa si scrive $\mathbf{p}=\mathbf{P}\mathbf{b}$ con

$$
\mathbf{P} = \mathbf{A}(\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top ,
$$

matrice che soddisfa $\mathbf{P}^2=\mathbf{P}$ (proiettare due volte è come
proiettare una volta) e $\mathbf{P}^\top=\mathbf{P}$, le due proprietà che
caratterizzano i proiettori ortogonali.

La stessa condizione si ricava per via analitica annullando il gradiente di
$\lVert\mathbf{A}\mathbf{x}-\mathbf{b}\rVert^2$, ed è utile vedere che le due
strade coincidono: $\nabla_{\mathbf{x}} =
2\mathbf{A}^\top(\mathbf{A}\mathbf{x}-\mathbf{b}) = \mathbf{0}$ dà
letteralmente l'ortogonalità del residuo. Il gradiente arriva nella
{doc}`sezione su analisi e ottimizzazione
</Matematica/analisi-ottimizzazione>`; l'identità fra «gradiente nullo» e
«residuo perpendicolare a tutto ciò che è raggiungibile» è una delle
corrispondenze più usate del mestiere.

Due avvertenze. Se il rango non è pieno, la soluzione di minimo esiste ancora
ma non è unica: differisce per elementi di $\ker(\mathbf{A})$, e la convenzione
è scegliere quella di norma minima (ciò che restituisce
`numpy.linalg.lstsq`). E il minimo dei quadrati è una **scelta**, non un
obbligo: minimizzare la somma dei valori assoluti dà un'altra retta, meno
sensibile ai valori anomali, e la {doc}`sezione sulle metriche
</MachineLearning/metriche>` mostra quanto le due risposte possano divergere.

`````

Con questa lettura la regressione lineare smette di essere una formula da
ricordare. Ecco lo stesso conto su cinque appartamenti di cui si conoscono i
metri quadri e il prezzo, fatto una volta risolvendo le equazioni normali e una
volta con la funzione di libreria.

```python
mq     = np.array([55., 70., 85., 100., 120.])
prezzo = np.array([148., 182., 210., 250., 290.])   # migliaia di euro

# due colonne: la costante e i metri quadri. Le loro combinazioni sono
# tutte e sole le rette del piano (mq, prezzo).
A = np.column_stack([np.ones(5), mq])

coef = np.linalg.solve(A.T @ A, A.T @ prezzo)       # equazioni normali
print(np.round(coef, 4))                            # -> [26.9339  2.1984]

residuo = prezzo - A @ coef
print(np.round(A.T @ residuo, 10))   # perpendicolare alle colonne -> [0. 0.]
print(np.round(residuo, 3))          # -> [ 0.152  1.175 -3.802  3.222 -0.747]
```

La retta trovata dice che un appartamento parte da circa ventisettemila euro e
ne guadagna circa duemiladuecento per metro quadro. Il conto che conta però è
la riga in mezzo: il residuo, moltiplicato per le colonne, dà zero in tutte e
due le direzioni. Sembra una formalità e invece è la definizione di «migliore»
messa alla prova, perché quello zero dice che nessun ritocco dei due
coefficienti potrebbe ridurre l'errore.

## Rendere perpendicolari le colonne: Gram-Schmidt e QR

Le equazioni normali sono la formula giusta per capire e la strada sbagliata
per calcolare, e conviene sapere perché.

`````{tab} Elementare

Il problema è quello degli assi obliqui di poco fa, portato all'estremo. Se due
colonne puntano quasi nella stessa direzione (i metri quadri e il numero di
stanze di un appartamento, che crescono insieme), distinguere il contributo
dell'una da quello dell'altra diventa un esercizio di equilibrismo: basta uno
spostamento minimo dei dati perché le due dosi cambino tantissimo pur dando
quasi lo stesso risultato.

C'è una grandezza che misura questo equilibrismo, e più è grande più i conti
sono delicati. Il guaio delle equazioni normali è che, nel passaggio da
$\mathbf{A}$ al prodotto per la sua trasposta, quella grandezza viene
**elevata al quadrato**: un problema un po' delicato diventa molto delicato, e
uno molto delicato diventa impossibile. Sull'esempio degli appartamenti si
passa da circa trecentocinquanta a più di centoventimila.

Il rimedio ha due facce. La prima è una precauzione che costa niente: togliere
a ogni colonna di dati la propria media, e se serve dividerla per la propria
scala. La colonna delle costanti si lascia stare, che centrata diventerebbe
tutta di zeri; le altre sì, e in questo esempio ce n'è una sola. La colonna dei
metri quadri vale attorno a ottanta, quella delle costanti vale uno, e già solo
questa sproporzione fa danni: centrare i metri quadri porta il numero da
trecentocinquanta a ventidue.

La seconda faccia è più profonda e consiste nel **raddrizzare le colonne**
prima di usarle. Si prende la prima direzione così com'è; alla seconda si toglie
la sua ombra sulla prima, e quel che resta è perpendicolare alla prima; alla
terza si tolgono le ombre sulle prime due, e così via. Alla fine si hanno
direzioni perpendicolari che descrivono esattamente lo stesso repertorio di
prima, e su assi perpendicolari nessuno fa equilibrismo. È il procedimento che
porta il nome di Gram e Schmidt, e le librerie serie lo usano in una versione
più robusta chiamata scomposizione QR.

`````

`````{tab} Superiore

Il **procedimento di Gram-Schmidt** trasforma una base
$\{\mathbf{a}_1,\dots,\mathbf{a}_n\}$ in una base ortonormale
$\{\mathbf{q}_1,\dots,\mathbf{q}_n\}$ dello stesso sottospazio:

$$
\tilde{\mathbf{q}}_k = \mathbf{a}_k
- \sum_{j<k} (\mathbf{q}_j^\top \mathbf{a}_k)\,\mathbf{q}_j,
\qquad
\mathbf{q}_k = \frac{\tilde{\mathbf{q}}_k}
{\lVert\tilde{\mathbf{q}}_k\rVert_2},
$$

dove la sommatoria toglie ad $\mathbf{a}_k$ le sue proiezioni su tutte le
direzioni già ortonormalizzate. Raccogliendo i coefficienti si ottiene la
**fattorizzazione QR**

$$
\mathbf{A} = \mathbf{Q}\mathbf{R},
$$

con $\mathbf{Q}\in\mathbb{R}^{m\times n}$ a colonne ortonormali e
$\mathbf{R}\in\mathbb{R}^{n\times n}$ triangolare superiore. Sostituendo nelle
equazioni normali, $\mathbf{Q}^\top\mathbf{Q}=\mathbf{I}$ fa collassare tutto
in

$$
\mathbf{R}\,\hat{\mathbf{x}} = \mathbf{Q}^\top\mathbf{b},
$$

un sistema triangolare che si risolve per sostituzione all'indietro in
$\Theta(n^2)$.

Il motivo per preferirlo sta nel condizionamento. Per una matrice a rango pieno
per colonne

$$
\kappa_2(\mathbf{A}^\top\mathbf{A}) = \kappa_2(\mathbf{A})^2 ,
$$

perché i valori singolari di $\mathbf{A}^\top\mathbf{A}$ sono i quadrati di
quelli di $\mathbf{A}$. Formare esplicitamente il prodotto raddoppia quindi le
cifre perse, mentre la via QR lavora su $\mathbf{A}$ e conserva
$\kappa_2(\mathbf{A})$. La {doc}`sezione di analisi numerica
</Matematica/analisi-numerica>` spiega che cosa quella grandezza misura e
perché le cifre si perdono. In pratica: `numpy.linalg.lstsq` non forma mai
$\mathbf{A}^\top\mathbf{A}$, e usa la decomposizione ai valori singolari, che è
ancora più cauta.

Nota storica sulla versione da usare: il Gram-Schmidt scritto sopra (detto
*classico*) è instabile in aritmetica finita, perché le proiezioni si
sottraggono tutte dallo stesso vettore di partenza e gli errori si sommano. La
variante *modificata*, che sottrae una proiezione per volta aggiornando ogni
volta il vettore, è algebricamente identica e numericamente molto migliore; le
librerie usano riflessioni di Householder, che sono meglio di entrambe.

`````

```python
# lo stesso conto per tre strade, e il prezzo delle equazioni normali
Q, R = np.linalg.qr(A)
print(np.round(np.linalg.solve(R, Q.T @ prezzo), 4))    # -> [26.9339  2.1984]
print(np.round(np.linalg.lstsq(A, prezzo, rcond=None)[0], 4))  # idem

print(round(np.linalg.cond(A), 1))            # -> 348.9
print(round(np.linalg.cond(A.T @ A), 1))      # -> 121756.6, cioe' il quadrato

# centrare la colonna dei metri quadri costa una riga e cambia tutto
A_centrata = np.column_stack([np.ones(5), mq - mq.mean()])
print(round(np.linalg.cond(A_centrata), 3))   # -> 22.672
```

## Tenere le prime direzioni, e sapere quanto si perde

Nel 1936 Carl Eckart e Gale Young pubblicano su *Psychometrika*, una rivista di
misurazione psicologica, un teorema che a loro serviva per l'analisi fattoriale
{cite}`eckart1936approximation`, e che oggi sta sotto la riduzione delle
dimensioni, sotto il modo in cui si decide quante direzioni di una tabella
contano davvero, e sotto ogni conto che dica quanto costa comprimere. Non
sapevano che Erhard Schmidt lo avesse dimostrato ventinove anni prima, in
tedesco e per le equazioni integrali {cite}`schmidt1907theorie`. Il teorema
dice che, quando di una tabella si può tenere solo un pezzo, esiste un pezzo
migliore di tutti gli altri; e dice, prima ancora di costruirlo, quanto si
sbaglia a tenere quello.

Il problema è quello lasciato aperto dalla
{doc}`sezione sui sistemi lineari </Matematica/sistemi-lineari>`, dove una
tabella di rango basso si riscriveva come il passaggio attraverso una strettoia
a poche corsie. Nei dati veri la strettoia esatta non c'è quasi mai: c'è una
tabella in cui poche direzioni contano molto e tutte le altre quasi niente.
Fissato allora un budget di $k$ direzioni, quali conviene tenere, e quanto si
sbaglia a tenere quelle?

Alla prima domanda si risponde d'istinto, a patto di avere un procedimento che
ordini le direzioni per importanza. Quel procedimento esiste ed è la
**decomposizione ai valori singolari**, cioè la riscrittura di una tabella come
somma di pezzi elementari, ciascuno con il proprio peso: si tengono i primi $k$
pezzi e si buttano gli altri. L'istinto ha ragione, e non è ovvio che ce
l'abbia, perché le tabelle di rango $k$ sono infinite e nessuna legge di natura
promette che la migliore sia fatta di pezzi presi da un'altra riscrittura. Alla
seconda domanda, quella sull'errore, la risposta è più precisa di quanto ci si
aspetti: l'errore non si stima, si legge.

`````{tab} Elementare

Il colorificio ha allargato il catalogo: sul cartoncino ci sono quaranta tinte,
e ognuna è una ricetta scritta sui dodici barattoli del magazzino. È una
tabella con una riga per tinta e una colonna per barattolo, ed è un catalogo
inventato apposta per la prova, perché di un catalogo inventato si sa in
anticipo com'è fatto dentro.

Poi arriva il vincolo: sullo scaffale del laboratorio ci stanno due barattoli
soli. Non due dei dodici, che sarebbe una scelta povera, ma due **premiscele**,
preparate una volta per tutte, ciascuna con la sua dose fissa dei dodici. Ogni
tinta del catalogo andrà poi ottenuta dosando quelle due e basta. Nessuna delle
quaranta verrà più esatta, e la domanda del titolare è secca: quali due
premiscele, e quanto verranno sbagliate le tinte?

Il procedimento risponde a tutte e due le domande insieme, e le premiscele le
inventa lui. Ne propone cinque, tante quante sono le direzioni che questo
catalogo ha davvero, e le consegna in ordine di importanza, ognuna con un peso
accanto. Il peso dice due cose in una: quanta parte del catalogo quella
premiscela porta sulle spalle, e quanto ogni litro versato sposta il colore.
Nell'esempio i pesi sono $10$, $6$, $2$, $0{,}5$ e $0{,}2$: la prima premiscela
muove le tinte cinquanta volte più dell'ultima.

Il vincolo dice due, quindi si tengono le prime due e si buttano le altre tre.
Il teorema garantisce che questa sia davvero la scelta migliore: nessun'altra
coppia di premiscele, per quanto astutamente mescolata, avvicina il catalogo
più di questa. Quanto si sbaglia, poi, si sa prima di aprire un barattolo,
perché sta scritto nei pesi buttati via, che sono $2$, $0{,}5$ e $0{,}2$. Il
conto è quello del teorema di Pitagora, esteso a più di due pezzi: si fa il
quadrato di ciascuno ($4$, $0{,}25$ e $0{,}04$), si sommano, si prende la
radice. Vengono $2{,}07$. Lo stesso conto sui cinque pesi insieme dà la
grandezza dell'intero catalogo, $11{,}84$, e il rapporto fra i due dice che con
due barattoli invece di cinque si sbaglia del diciassette e mezzo per cento.
Dallo stesso conto segue anche una garanzia sul caso peggiore: nessuna singola
tinta sbaglia più del peso più grosso fra quelli buttati, cioè più di $2$.

Che l'ordine dei pesi conti si vede tenendo la prima e la terza premiscela
invece delle prime due: si butta il $6$, il conto di Pitagora dà poco più di
$6$, e lo sbaglio quasi triplica. Sullo sbaglio complessivo l'unico caso in cui
l'ordine lascia qualcosa in sospeso è quello di due pesi identici: allora quale
delle due premiscele tenere non lo decide nessuno, e vanno bene tutte e due.
Guardando invece la sola tinta peggiore le coppie ottime sono tante, perché
quel conto non vede la differenza fra due scelte che sbagliano ugualmente sul
caso più difficile.

Due avvertenze, e sono le situazioni in cui questo mestiere non si può fare. La
prima è un catalogo con i pesi tutti uguali, mettiamo tutti $3$: buttarne tre
lascia uno sbaglio di più del settantasette per cento, e la risposta onesta al
titolare è che il catalogo ha bisogno di cinque barattoli davvero. La seconda è
più insidiosa, perché il conto viene lo stesso e il risultato non serve. «Lo
sbaglio complessivo del catalogo» è un criterio, e va bene finché tutte le
tinte contano uguale; se una sola di esse è il colore del marchio di un cliente
e deve venire perfetta, la coppia migliore è un'altra, e questa non la trova.
Lo stesso vale quando del catalogo si conosce solo un pezzo, con delle caselle
lasciate in bianco: lì i pesi si dovrebbero calcolare su qualcosa che non si
ha.

`````

`````{tab} Superiore

Sia $\mathbf{A}\in\mathbb{R}^{m\times n}$ di rango $r$, con decomposizione ai
valori singolari $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$.
Scrivendo $\mathbf{u}_i$ e $\mathbf{v}_i$ per le colonne di $\mathbf{U}$ e
$\mathbf{V}$, cioè per i vettori singolari sinistri e destri (ortonormali,
perché le due matrici sono ortogonali), la stessa decomposizione si legge come
somma di $r$ pezzi di rango uno,

$$
\mathbf{A} = \sum_{i=1}^{r} \sigma_i\,\mathbf{u}_i\mathbf{v}_i^\top ,
\qquad \sigma_1\ge\sigma_2\ge\dots\ge\sigma_r>0 ,
$$

ed è questa la scrittura che serve qui. Per $k<r$ si chiami **troncamento di
rango $k$** la somma parziale $\mathbf{A}_k = \sum_{i\le k}
\sigma_i\,\mathbf{u}_i\mathbf{v}_i^\top$, che ha rango esattamente $k$ perché
$\mathbf{A}_k\mathbf{v}_j = \sigma_j\mathbf{u}_j$ per $j\le k$, cioè la sua
immagine contiene $k$ vettori indipendenti.

**Teorema di Schmidt-Eckart-Young-Mirsky.** Per ogni $\mathbf{B}$ con
$\operatorname{rank}(\mathbf{B})\le k$,

$$
\lVert\mathbf{A}-\mathbf{B}\rVert_F \ \ge\
\lVert\mathbf{A}-\mathbf{A}_k\rVert_F = \sqrt{\sum_{i>k}\sigma_i^2},
\qquad
\lVert\mathbf{A}-\mathbf{B}\rVert_2 \ \ge\
\lVert\mathbf{A}-\mathbf{A}_k\rVert_2 = \sigma_{k+1},
$$

dove $\lVert\cdot\rVert_F$ è la norma di Frobenius (la radice della somma dei
quadrati di tutti gli elementi) e $\lVert\cdot\rVert_2$ la norma spettrale,
cioè l'allungamento massimo già incontrato. Eckart e Young dimostrano il caso
di Frobenius {cite}`eckart1936approximation`; Mirsky, nel 1960, mostra che lo
stesso $\mathbf{A}_k$ minimizza qualunque norma invariante per trasformazioni
ortogonali {cite}`mirsky1960symmetric`, cioè che il troncamento non è una
risposta tarata su un modo particolare di misurare l'errore.

L'unicità invece **dipende dalla norma**, e vale la pena non confonderle. In
norma di Frobenius il minimo è unico se e solo se $\sigma_k > \sigma_{k+1}$: a
valori singolari pari il sottospazio da tenere non è determinato. In norma
spettrale di minimi ce ne sono quasi sempre infiniti, perché conta solo il
residuo più grande: per $\mathbf{A} = \operatorname{diag}(2,1)$ e $k=1$ ogni
$\mathbf{B} = \operatorname{diag}(c,0)$ con $c\in[1,3]$ ha rango uno e lascia
$\lVert\mathbf{A}-\mathbf{B}\rVert_2 = 1 = \sigma_2$, mentre in Frobenius solo
$c=2$ è ottimo.

Per la norma spettrale la dimostrazione è un conteggio di dimensioni, e vale la
pena vederla perché mostra da dove venga $\sigma_{k+1}$. Se
$\operatorname{rank}(\mathbf{B})\le k$ allora $\dim\ker(\mathbf{B})\ge n-k$;
lo span di $\mathbf{v}_1,\dots,\mathbf{v}_{k+1}$ ha dimensione $k+1$; siccome
$(n-k)+(k+1)>n$, la formula di Grassmann
($\dim(U\cap W)\ge\dim U+\dim W-n$) dà un'intersezione non banale, e se ne
prende un vettore unitario $\mathbf{w}$. Allora $(\mathbf{A}-\mathbf{B})
\mathbf{w} = \mathbf{A}\mathbf{w}$, e siccome $\mathbf{w}$ vive nello span dei
primi $k+1$ vettori singolari destri, che sono ortonormali, per l'identità di
Parseval $\sum_{i\le k+1}(\mathbf{v}_i^\top\mathbf{w})^2 =
\lVert\mathbf{w}\rVert_2^2 = 1$, e quindi

$$
\lVert\mathbf{A}\mathbf{w}\rVert_2^2
= \sum_{i\le k+1}\sigma_i^2\,(\mathbf{v}_i^\top\mathbf{w})^2
\ \ge\ \sigma_{k+1}^2 \sum_{i\le k+1}(\mathbf{v}_i^\top\mathbf{w})^2
= \sigma_{k+1}^2 .
$$

Siccome $\lVert\mathbf{A}-\mathbf{B}\rVert_2 \ge \lVert(\mathbf{A}-\mathbf{B})
\mathbf{w}\rVert_2$ per definizione di norma spettrale, nessuna matrice di
rango $k$ scende sotto $\sigma_{k+1}$, e il troncamento ci arriva. Da lì
discende anche il controllo riga per riga, che è quello che interessa quando le
righe sono i casi di una tabella: la riga $j$-esima di
$\mathbf{A}-\mathbf{A}_k$ ha norma
$\lVert(\mathbf{A}-\mathbf{A}_k)^\top\mathbf{c}_j\rVert_2 \le
\lVert\mathbf{A}-\mathbf{A}_k\rVert_2 = \sigma_{k+1}$, dove $\mathbf{c}_j$ è
il vettore che vale uno nella posizione $j$ e zero altrove.

Da qui discendono tre fatti che tornano di continuo. La **PCA** è il
troncamento della decomposizione della matrice dei dati centrata per colonne:
i vettori singolari destri di $\mathbf{X}$ sono gli autovettori di
$\mathbf{X}^\top\mathbf{X}$, cioè della covarianza campionaria a meno di un
fattore costante (che non cambia gli autovettori), quindi la strada spettrale e
quella dei valori singolari danno gli stessi assi, e la {doc}`sezione sulla
riduzione di dimensionalità </MachineLearning/riduzione-clustering>` percorre
la prima; senza il centraggio si ottiene invece il migliore sottospazio
passante per l'origine, che in generale è un altro. Il rango numerico è la
stessa mossa letta al contrario: azzerare i $\sigma_i$ sotto una soglia $\tau$
significa sostituire $\mathbf{A}$ con il troncamento che dista da essa meno di
$\tau$ **in norma spettrale** (in Frobenius la distanza è
$\sqrt{\sum_{i>k}\sigma_i^2}$ e può superare $\tau$ quanto si vuole). E un
collo di bottiglia lineare addestrato su un errore quadratico ha in
$\mathbf{A}_k$ il proprio ottimo globale; che la discesa del gradiente ci
arrivi è un fatto in più, dovuto a Baldi e Hornik, che mostrano come su questa
superficie non esistano minimi locali spuri, e come la soluzione sia
determinata solo **a meno di un cambio di base** nel latente
($\mathbf{W}_2 = \mathbf{U}_k\mathbf{C}$ e
$\mathbf{W}_1 = \mathbf{C}^{-1}\mathbf{U}_k^\top$, con $\mathbf{C}$
invertibile qualunque): un autoencoder lineare ritrova il sottospazio della
PCA, non i suoi assi {cite}`baldi1989neural`.

I due limiti sono precisi, e il secondo è quello che si dimentica. Se lo
spettro non decade il teorema resta vero e diventa inutile: con tutti i
$\sigma_i$ uguali l'errore relativo del troncamento è $\sqrt{(r-k)/r}$,
cioè quasi tutto. E il teorema vale per la norma di Frobenius su tutte le
celle: se la somma corre su un sottoinsieme di celle (le sole osservate) o pesa
le celle in modo diverso, il minimo smette di essere il troncamento e il
problema perde la soluzione in forma chiusa. È esattamente la ragione per cui
la fattorizzazione delle matrici di valutazioni si stima per discesa del
gradiente invece che con una decomposizione, come racconta la
{doc}`sezione sul filtraggio collaborativo
</SistemiRaccomandazione/filtraggio-collaborativo>`.

`````

Il teorema si mette alla prova costruendo una tabella di cui si conoscono i
pesi in anticipo, e guardando se l'errore del troncamento cade dove promesso.

```python
rng = np.random.default_rng(0)

# quaranta tinte, dodici pigmenti, cinque premiscele di peso deciso
pesi = np.array([10.0, 6.0, 2.0, 0.5, 0.2])
U0, _ = np.linalg.qr(rng.normal(size=(40, 5)))
V0, _ = np.linalg.qr(rng.normal(size=(12, 5)))
C = U0 @ np.diag(pesi) @ V0.T

U, s, Vt = np.linalg.svd(C, full_matrices=False)
print(np.round(s[:6], 6))            # -> [10.  6.  2.  0.5 0.2 0. ]

k = 2
C_k = (U[:, :k] * s[:k]) @ Vt[:k]
print(round(np.linalg.norm(C - C_k, 'fro'), 4),
      round(np.sqrt((s[k:] ** 2).sum()), 4))          # -> 2.0712 2.0712
print(round(np.linalg.norm(C - C_k, 2), 4), round(s[k], 4))   # -> 2.0 2.0
print(round(np.linalg.norm(C - C_k, axis=1).max(), 4))        # -> 0.8463
print(round(np.linalg.norm(C, 'fro'), 4))                     # -> 11.8444

# tenere la prima e la terza invece delle prime due
sel = [0, 2]
print(round(np.linalg.norm(C - (U[:, sel] * s[sel]) @ Vt[sel], 'fro'), 4))

# duemila vicine di rango 2, sempre piu' vicine: nessuna sotto 2.0712
B0, D0 = U[:, :k] * s[:k], Vt[:k]
for eps in (0.05, 0.01, 0.001):
    print(eps, round(min(np.linalg.norm(
        C - (B0 + eps * rng.normal(size=B0.shape))
          @ (D0 + eps * rng.normal(size=D0.shape)), 'fro')
        for _ in range(2000)), 4))

# lo stesso troncamento su uno spettro piatto
Z = U0 @ np.diag(np.full(5, 3.0)) @ V0.T
Uz, sz, Vtz = np.linalg.svd(Z, full_matrices=False)
Z_k = (Uz[:, :k] * sz[:k]) @ Vtz[:k]
print(round(np.linalg.norm(Z - Z_k, 'fro') / np.linalg.norm(Z, 'fro'), 4),
      round(np.linalg.norm(C - C_k, 'fro') / np.linalg.norm(C, 'fro'), 4))
```

Il conto conferma le due uguaglianze. Lo sbaglio complessivo vale $2{,}0712$,
cioè la radice della somma dei quadrati dei pesi scartati, e la grandezza
dell'intera tabella misurata allo stesso modo vale $11{,}8444$; il caso
peggiore vale $2{,}0$, cioè il primo peso buttato, e la riga della tabella che
sbaglia più di tutte si ferma a $0{,}8463$, dentro quella garanzia con
margine. Tenere la prima e la terza direzione invece delle prime due porta lo
sbaglio a $6{,}0241$; e di duemila alternative costruite sgarrando dalla coppia
migliore nessuna scende sotto $2{,}0712$, con la meno peggio a $2{,}3703$
sgarrando di cinque centesimi, a $2{,}0830$ di un centesimo e a $2{,}0714$ di
un millesimo, cioè il limite si vede stringere. Sulle due tabelle messe a
confronto si vede invece il caso in cui il teorema non serve: quella dei pesi
decrescenti perde il $17{,}5\%$ fermandosi a due direzioni, quella dei pesi
tutti uguali il $77{,}5\%$. Stesso algoritmo, stesso rango, e nel secondo
caso non c'è niente da comprimere.

## Quando le soluzioni ugualmente buone sono più di una

Finora ogni domanda ha avuto una risposta sola. Ce n'è una che ne ha infinite,
e capita ogni volta che due colonne dicono la stessa cosa: una grandezza
misurata due volte in unità diverse, un totale registrato accanto ai suoi
addendi. Nei dati veri è la regola più che l'eccezione. Un metodo che davanti a
infinite risposte si ferma non serve a niente; ne serve uno che ne scelga una
e dica quale.

`````{tab} Elementare

Che cosa si risponde a un cliente quando le ricette che vanno bene sono più di
una? Al bancone succede per un guaio di magazzino: due dei barattoli sono in
realtà lo stesso colore, comprati da due fornitori diversi. Allora le dosi che
danno la miscela migliore non sono più una sola. Un litro dal primo e niente
dal secondo, niente dal primo e un litro dal secondo, mezzo e mezzo, tre quarti
e un quarto: viene sempre la stessa tinta, e lo stesso identico scarto dal
colore chiesto.

Fra risposte che valgono uguale bisogna sceglierne una, e la regola è quella
che il bancone suggerisce da sé: si spalma invece di concentrare. Si
sommano i quadrati delle dosi e si tiene la ricetta che dà il totale più
piccolo. Mezzo e mezzo dà $0{,}25 + 0{,}25$, cioè $0{,}50$; il litro secco dà
$1$, il doppio. È lo stesso motivo per cui i quadrati erano comparsi parlando
di ombre: fanno pagare caro chi esagera in un punto solo. È una convenzione, e
nessuno vieta al titolare di preferirne un'altra; ma è quella che le librerie
di calcolo del libro restituiscono senza chiedere, ed è bene sapere di averla
ricevuta.

Il modo di ottenerla è quello del catalogo, letto al contrario. Là ogni
premiscela aveva un peso, che diceva quanto un litro di quella premiscela
sposta il colore; qui si parte dallo spostamento che serve e si cerca il litro,
quindi si divide per il peso invece di moltiplicare. Una premiscela pesante
sposta molto, e ne bastano poche gocce; una leggera ne vuole litri. E una
premiscela di peso zero non sposta niente: qualunque dose se ne metta il colore
non cambia, e allora la si lascia fuori dalla ricetta. Se poi i barattoli sono
tutti diversi la ricetta buona è una sola, e questa procedura restituisce
quella, senza cambiare niente di quello che si sapeva già.

C'è anche una seconda strada, e vale la pena conoscerla perché mostra che la
regola di poco fa non è arbitraria. Si chiede al conto di non esagerare con le
dosi, mettendo una penale sui litri usati, e poi si allenta la penale un po'
alla volta. Man mano che si alleggerisce, la ricetta si avvicina proprio a
quella spalmata, e continuando ad alleggerirla ci arriva.

Il perché la regola convenga si vede quando i due barattoli sono quasi uguali e
non del tutto. Allora, in senso stretto, la ricetta migliore torna a essere una
sola, e prescrive decine di migliaia di litri dal primo barattolo e, dal
secondo, una dose negativa: vernice da togliere. Una dose negativa, in bottega,
è un gesto che non esiste, e il segnale sta lì più che nella grandezza del
numero. La differenza fra i
due barattoli è più piccola dell'imprecisione con cui li si è misurati, e il
conto sta rispondendo a una domanda che nessuno ha davvero posto. La cura è
dichiararlo: si fissa una soglia sotto la quale due barattoli si considerano lo
stesso barattolo, e la ricetta torna quella di bottega. Ed è la mossa del
catalogo, la stessa: buttare via le premiscele di peso trascurabile.

`````

`````{tab} Superiore

Data $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$, la
**pseudoinversa di Moore-Penrose** è

$$
\mathbf{A}^{+} = \mathbf{V}\boldsymbol{\Sigma}^{+}\mathbf{U}^\top ,
$$

dove $\boldsymbol{\Sigma}^{+}$ si ottiene trasponendo $\boldsymbol{\Sigma}$ e
sostituendo ogni $\sigma_i>0$ con $1/\sigma_i$, lasciando a zero gli altri.
Penrose la caratterizza nel 1955 come l'unica matrice che soddisfa quattro
identità {cite}`penrose1955generalized`: $\mathbf{A}\mathbf{A}^{+}\mathbf{A} =
\mathbf{A}$, $\mathbf{A}^{+}\mathbf{A}\mathbf{A}^{+} = \mathbf{A}^{+}$, e la
simmetria dei due prodotti $\mathbf{A}\mathbf{A}^{+}$ e
$\mathbf{A}^{+}\mathbf{A}$, che sono i proiettori ortogonali sull'immagine di
$\mathbf{A}$ e sul complemento ortogonale del suo nucleo. Se $\mathbf{A}$ è
invertibile allora $\mathbf{A}^{+}=\mathbf{A}^{-1}$; se ha rango pieno per
colonne allora $\mathbf{A}^{+} =
(\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top$, cioè la formula delle
equazioni normali.

Il suo mestiere sta in una riga: $\hat{\mathbf{x}} =
\mathbf{A}^{+}\mathbf{b}$ è, fra tutti i minimizzatori di
$\lVert\mathbf{A}\mathbf{x}-\mathbf{b}\rVert_2$, quello di norma
$\lVert\mathbf{x}\rVert_2$ minima. Le due situazioni si leggono insieme: con
più righe che colonne il minimo dell'errore in generale non vale zero (vale
zero solo se $\mathbf{b}$ sta nell'immagine) e la pseudoinversa dà la
proiezione; con un nucleo non banale i minimizzatori formano
$\hat{\mathbf{x}}+\ker(\mathbf{A})$ e la pseudoinversa sceglie il
rappresentante ortogonale al nucleo.

C'è anche un ponte con la regolarizzazione:

$$
\lim_{\lambda\to 0^{+}}
(\mathbf{A}^\top\mathbf{A}+\lambda\mathbf{I})^{-1}\mathbf{A}^\top
= \mathbf{A}^{+} .
$$

Il termine $\lambda\mathbf{I}$ è la penalità **ridge**, cioè il prezzo messo
sulla grandezza dei coefficienti che la
{doc}`sezione su overfitting e validazione
</MachineLearning/overfitting-validazione>` tratta per esteso: farla svanire
porta alla soluzione di norma minima, il che dà alla pseudoinversa una lettura
statistica, quella di limite di un modello regolarizzato invece che di ripiego
algebrico.

Il punto di rottura è che $\mathbf{A}\mapsto\mathbf{A}^{+}$ perde la continuità
dove il rango cambia. Se un valore singolare vale $\varepsilon$ invece di zero
il suo reciproco vale $1/\varepsilon$, e la soluzione di norma minima esplode;
il limite per $\varepsilon\to 0$ di quelle soluzioni non è la soluzione della
matrice di rango carente. Per questo tutte le implementazioni troncano:
`numpy.linalg.pinv` azzera i $\sigma_i$ sotto `rcond` volte $\sigma_{\max}$,
cioè calcola la pseudoinversa di un troncamento $\mathbf{A}_k$ invece che di
$\mathbf{A}$. Il legame con il teorema di Schmidt-Eckart-Young-Mirsky è
letterale: la sola pseudoinversa numericamente sensata è quella di un
troncamento. E il nome del parametro dice che cosa si sta dichiarando, perché
`rcond` è il reciproco del condizionamento che si accetta: fissarlo a $10^{-4}$
vuol dire rifiutare ogni direzione con $\kappa_2$ oltre $10^{4}$.

`````

All'esempio degli appartamenti basta aggiungere una colonna che ne ripete
un'altra, il numero di stanze contato come un venticinquesimo dei metri
quadri, per avere un sistema con infinite risposte ugualmente buone.

```python
A2 = np.column_stack([np.ones(5), mq, mq / 25.0])
print(np.linalg.matrix_rank(A2))                      # -> 2, non 3

x = np.linalg.pinv(A2) @ prezzo
print(np.round(x, 4), round(np.linalg.norm(x), 4))
# -> [26.9339  2.1949  0.0878] 27.0233

alt = x + 200.0 * np.array([0.0, 1 / 25.0, -1.0])     # nel nucleo di A2
print(round(np.linalg.norm(A2 @ x - prezzo), 5),
      round(np.linalg.norm(A2 @ alt - prezzo), 5),
      round(np.linalg.norm(alt), 4))      # -> 5.17627 5.17627 201.9759

for lam in (1.0, 1e-2, 1e-4, 1e-6):                   # il limite ridge
    xl = np.linalg.solve(A2.T @ A2 + lam * np.eye(3), A2.T @ prezzo)
    print(lam, round(np.linalg.norm(xl - x), 6))
# -> 1.0 20.310268 | 0.01 0.803555 | 0.0001 0.00828 | 1e-06 8.3e-05

# la colonna quasi doppione, che il rango numerico non vede piu'
A3 = A2.copy()
A3[:, 2] += 1e-6 * np.array([1., -1., 1., -1., 1.])
print(np.linalg.matrix_rank(A3))                      # -> 3
print(np.round(np.linalg.pinv(A3) @ prezzo, 1))
# -> [ 2.7100000e+01  7.3378800e+04 -1.8344156e+06]
print(np.round(np.linalg.pinv(A3, rcond=1e-4) @ prezzo, 4))
# -> [26.9339  2.1949  0.0878]
```

I due vettori `x` e `alt` lasciano lo stesso errore, $5{,}17627$ tutti e due, e
la misura che li distingue, la radice della somma dei quadrati dei
coefficienti, vale $27{,}02$ per il primo e $201{,}98$ per il secondo: la
pseudoinversa restituisce il primo. La penalità ridge ci arriva da fuori, e la
distanza dalla
soluzione parsimoniosa scende da $20{,}31$ a $8{,}3\cdot 10^{-5}$ mentre la
penalità si alleggerisce di sei ordini di grandezza. E la matrice `A3`, che
differisce dalla precedente per un milionesimo, viene contata come se le sue
tre colonne portassero informazioni davvero distinte, e la soluzione più
parsimoniosa passa a coefficienti dell'ordine del milione; dichiarando una
soglia si torna esattamente a quella di prima. È il motivo per cui `lstsq`
accetta un `rcond`, e il valore che quel parametro ha di suo è comunque una
soglia, scelta da chi ha scritto la libreria e non da chi la usa.

## In molte dimensioni quasi tutto è perpendicolare

C'è un fatto sull'ortogonalità che nello spazio di tutti i giorni non si vede,
e che regge buona parte di come funzionano le rappresentazioni interne dei
modelli.

`````{tab} Elementare

Sul foglio, due frecce prese a caso formano spesso angoli piccoli: capita di
continuo che puntino più o meno dalla stessa parte. Salendo di dimensione la
faccenda cambia, e cambia in fretta. In uno spazio a mille dimensioni due
direzioni prese a caso sono quasi sempre **quasi perpendicolari**: il loro
prodotto scalare, misurato su vettori di lunghezza uno, si aggira intorno a
due centesimi e mezzo invece che intorno a mezzo.

La ragione si intuisce contando le occasioni di essere diversi. Perché due
frecce puntino nella stessa direzione devono andare d'accordo su **tutte** le
coordinate contemporaneamente; con mille coordinate sorteggiate a caso, che
vadano d'accordo su tutte è un colpo di fortuna astronomico, e la somma dei
mille contributi (metà positivi, metà negativi) si compensa quasi
perfettamente.

La conseguenza pratica cambia il modo di pensare agli spazi grandi. In mille
dimensioni si possono sistemare **molte più di mille** direzioni tutte quasi
perpendicolari fra loro: rinunciando alla perpendicolarità esatta e
accontentandosi di «quasi», la capienza esplode. È il motivo per cui una rete
può tenere in uno spazio di poche migliaia di dimensioni un numero di concetti
molto più grande, dando a ciascuno una direzione propria e sperando che si
disturbino poco. La {doc}`sezione sull'interpretabilità meccanicistica
</Interpretabilita/attribuzione-e-meccanicistica>` racconta che cosa succede
quando invece si disturbano.

Il rovescio della medaglia va detto subito, perché è il punto in cui questa
materia inganna di più: **l'intuizione costruita su tre dimensioni sbaglia in
modo sistematico**, e non di poco. Cose che nel piano sembrano rare (due
direzioni quasi perpendicolari) là dentro sono la norma, e cose che nel piano
sembrano normali (due direzioni quasi allineate) là dentro non capitano mai per
caso. Quando un ragionamento su uno spazio grande poggia su un disegno fatto
sul foglio, conviene rifarlo con i numeri.

`````

`````{tab} Superiore

Siano $\mathbf{u},\mathbf{v}$ due vettori unitari indipendenti e uniformi sulla
sfera $S^{d-1}$. Il loro coseno $\mathbf{u}^\top\mathbf{v}$ ha media nulla e
varianza $1/d$, quindi la sua deviazione standard è $1/\sqrt{d}$; per $d$
grande il valore atteso del modulo tende a $\sqrt{2/(\pi d)}\approx
0{,}8/\sqrt{d}$, mentre in tre dimensioni vale esattamente $1/2$. La concentrazione
è esponenziale: per ogni $\varepsilon>0$

$$
\Pr\big[\,|\mathbf{u}^\top\mathbf{v}| > \varepsilon\,\big]
\le 2\,e^{-d\varepsilon^2/2},
$$

e da questa stima, con l'unione sulle coppie, segue che si possono collocare
$N$ direzioni con coseni tutti sotto $\varepsilon$ finché
$N \lesssim e^{d\varepsilon^2/4}$, cioè un numero **esponenziale** nella
dimensione. È lo stesso conto che regge il lemma di Johnson-Lindenstrauss sulle
proiezioni casuali che quasi conservano le distanze.

Le conseguenze per il deep learning sono due, e vanno tenute insieme. La prima
è che uno spazio di rappresentazione a $d$ dimensioni ospita molte più di $d$
feature distinguibili, purché si accetti una sovrapposizione piccola: è
l'ipotesi della **sovrapposizione**, che la {doc}`sezione
sull'interpretabilità meccanicistica
</Interpretabilita/attribuzione-e-meccanicistica>` discute per esteso. La
seconda è un avvertimento sulla geometria dell'alta dimensione: l'intuizione
tridimensionale sbaglia sistematicamente, e le quantità che sembrano tipiche
(distanze, angoli, volumi vicino alla superficie) si concentrano attorno a
valori che nel piano non si osservano mai.

`````

```python
rng = np.random.default_rng(0)

for d in (3, 100, 1000):
    V = rng.normal(size=(3000, d))
    V /= np.linalg.norm(V, axis=1, keepdims=True)   # sulla sfera unitaria
    G = V @ V.T
    np.fill_diagonal(G, 0.0)                        # via i coseni di un
                                                    # vettore con se' stesso
    print(d, round(abs(G).mean(), 4), round(abs(G).max(), 3),
          round(1 / np.sqrt(d), 4))
# -> 3    0.4998 1.0   0.5774
# -> 100  0.0799 0.477 0.1
# -> 1000 0.0252 0.166 0.0316
```

Tremila direzioni sorteggiate in mille dimensioni, cioè tre volte più direzioni
che dimensioni, e fra i quattro milioni e mezzo di coppie possibili la più
allineata di tutte ha coseno $0{,}166$, che è un angolo di poco più di ottanta
gradi. Le stesse tremila direzioni in tre dimensioni danno invece coseno
massimo $1{,}0$: là dentro non c'è posto, e due finiscono per sovrapporsi.

I due arrotondamenti a dodici cifre del conto sulle cinque misure meritano una
riga: la somma degli scarti non viene esattamente zero ma un numero
dell'ordine di $10^{-16}$, cioè il gradino più piccolo che la macchina sa
scrivere accanto a $2$, e il motivo lo spiega la
{doc}`sezione di analisi numerica </Matematica/analisi-numerica>`.

## In pratica, con NumPy

```python
import numpy as np

A = np.column_stack([np.ones(5), np.array([55., 70., 85., 100., 120.])])
b = np.array([148., 182., 210., 250., 290.])

np.linalg.lstsq(A, b, rcond=None)[0]   # la via consigliata: minimi quadrati
np.linalg.qr(A)                        # Q ortonormale, R triangolare
np.linalg.cond(A)                      # quanto il problema e' delicato

P = A @ np.linalg.inv(A.T @ A) @ A.T   # il proiettore, da guardare non da usare
np.allclose(P @ P, P)                  # proiettare due volte -> True

np.linalg.svd(A, full_matrices=False)  # U, valori singolari, V trasposta
np.linalg.pinv(A)                      # la pseudoinversa, troncata a rcond
```

Il proiettore costruito per esteso vale come esercizio e mai come codice di
produzione: costa $\Theta(m^2 n)$, e l'inversa che compare nella sua formula è
quella di $\mathbf{A}^\top\mathbf{A}$, cioè proprio il passaggio che raddoppia
il condizionamento. Nel lavoro vero si usa `lstsq`,
e se serve la proiezione la si ottiene come `A @ coefficienti`.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Due direzioni sono **perpendicolari** quando il loro prodotto scalare fa
  zero, e allora sapere quanto vale l'una non dice niente sull'altra. Con
  direzioni perpendicolari e lunghe uno ogni coefficiente si legge con un
  prodotto scalare, senza risolvere nessun sistema.
- La **proiezione** è l'ombra a picco: ogni vettore si spacca in modo unico
  nella parte che sta nella direzione scelta più la parte perpendicolare. La
  **media aritmetica** è esattamente l'ombra delle misure sulla direzione in
  cui sono tutte uguali, e questo spiega perché gli scarti dalla media si
  compensano.
- Quando il colore chiesto non si può fare, si prende quello più vicino fra
  quelli possibili, cioè l'ombra. Il segno di aver finito è che **l'errore
  rimasto è perpendicolare a tutto ciò che si poteva fare**: se non lo fosse,
  spostandosi si migliorerebbe ancora.
- Colonne quasi allineate rendono i conti equilibristici, e la formula più
  immediata peggiora le cose elevando al quadrato quella delicatezza. Le due
  cure sono centrare le colonne dei dati, non quella delle costanti (una riga
  di codice), e **raddrizzarle**
  togliendo a ciascuna l'ombra sulle precedenti, che è quello che fanno
  Gram-Schmidt e la scomposizione QR.
- Quando di una tabella si può tenere solo un pezzo, il pezzo migliore si
  ottiene ordinando le direzioni per importanza e tenendo le prime: nessun'altra
  scelta le si avvicina di più. Lo sbaglio si conosce **prima**, perché sta
  scritto nei pesi scartati: quadrato di ciascuno, somma, radice. Vale anche
  caso per caso, perché nessuna riga sbaglia più del peso più grosso fra quelli
  buttati. Se i pesi sono tutti simili non c'è niente da guadagnare, e se
  alcune caselle mancano il conto non vale più.
- Quando le ricette ugualmente buone sono più di una, si tiene quella che
  spalma le dosi invece di concentrarle, cioè quella con la somma dei quadrati
  più piccola. Due direzioni quasi identiche vanno contate come una sola, sotto
  una soglia che qualcuno deve dichiarare.
- In uno spazio con tante dimensioni due direzioni a caso sono quasi sempre
  **quasi perpendicolari**, e ce ne stanno molte più di quante siano le
  dimensioni. È così che un modello dà una direzione propria a un numero
  enorme di concetti.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- $\mathbf{u}^\top\mathbf{v}=0$ definisce l'ortogonalità; una base
  ortonormale dà i coefficienti come prodotti scalari, e una matrice ortogonale
  soddisfa $\mathbf{Q}^{-1}=\mathbf{Q}^\top$, conserva le norme e ha
  $\kappa_2=1$.
- $\operatorname{proj}_{\mathbf{a}}(\mathbf{b}) =
  \frac{\mathbf{a}^\top\mathbf{b}}{\mathbf{a}^\top\mathbf{a}}\mathbf{a}$; con
  $\mathbf{a}=\mathbf{1}$ si ottiene la media campionaria, e la norma al
  quadrato del residuo è $n$ volte la varianza.
- **Teorema della proiezione**: $\hat{\mathbf{x}}$ minimizza
  $\lVert\mathbf{A}\mathbf{x}-\mathbf{b}\rVert_2$ se e solo se
  $\mathbf{A}^\top(\mathbf{b}-\mathbf{A}\hat{\mathbf{x}})=\mathbf{0}$, da cui
  le **equazioni normali** $\mathbf{A}^\top\mathbf{A}\hat{\mathbf{x}} =
  \mathbf{A}^\top\mathbf{b}$ e il proiettore
  $\mathbf{P}=\mathbf{A}(\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top$, con
  $\mathbf{P}^2=\mathbf{P}=\mathbf{P}^\top$. La stessa condizione è
  l'annullamento del gradiente.
- **Gram-Schmidt** ortonormalizza togliendo a ogni colonna le proiezioni sulle
  precedenti e dà $\mathbf{A}=\mathbf{Q}\mathbf{R}$; si preferisce alle
  equazioni normali perché
  $\kappa_2(\mathbf{A}^\top\mathbf{A})=\kappa_2(\mathbf{A})^2$. Usare la
  variante modificata, o meglio Householder.
- **Schmidt-Eckart-Young-Mirsky**: fra le matrici di rango $\le k$, il
  troncamento $\mathbf{A}_k=\sum_{i\le k}\sigma_i\mathbf{u}_i\mathbf{v}_i^\top$
  minimizza ogni norma invariante per trasformazioni ortogonali, con
  $\lVert\mathbf{A}-\mathbf{A}_k\rVert_F=\sqrt{\sum_{i>k}\sigma_i^2}$ e
  $\lVert\mathbf{A}-\mathbf{A}_k\rVert_2=\sigma_{k+1}$. In Frobenius il
  minimo è unico se e solo se $\sigma_k>\sigma_{k+1}$; in norma spettrale i
  minimi sono in genere infiniti. Diventa inutile se lo spettro non decade, e
  cade se la norma è pesata o ristretta alle celle osservate.
- La **pseudoinversa**
  $\mathbf{A}^{+}=\mathbf{V}\boldsymbol{\Sigma}^{+}\mathbf{U}^\top$ dà il
  minimizzatore di $\lVert\mathbf{A}\mathbf{x}-\mathbf{b}\rVert_2$ di norma
  minima, coincide con $(\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top$ a
  rango pieno per colonne ed è il limite della penalità ridge per
  $\lambda\to 0^{+}$. Non è continua dove il rango cambia: da qui il
  troncamento a `rcond`.
- Su $S^{d-1}$ il coseno fra due direzioni casuali ha deviazione standard
  $1/\sqrt{d}$ e code sub-gaussiane, quindi in $\mathbb{R}^d$ si collocano
  esponenzialmente molte direzioni quasi ortogonali: è la base della
  sovrapposizione e delle proiezioni casuali.
```
`````

L'ombra risolve il caso in cui i vincoli sono troppi, e la
{doc}`sezione sui sistemi lineari </Matematica/sistemi-lineari>` aveva risolto
quello in cui sono troppo pochi; i valori singolari, ordinati, hanno poi detto
che cosa conviene tenere quando non si può tenere tutto. Resta una terza
domanda, che finora è rimasta sullo sfondo: che cosa una trasformazione fa alle
grandezze che si misurano. La risposta sta in un numero solo, che dice quanto
una trasformazione gonfia o comprime lo spazio su cui lavora.
