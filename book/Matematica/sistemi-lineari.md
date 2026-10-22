# Sistemi lineari: quando i vincoli bastano, e quando no

Un manuale cinese compilato intorno al primo secolo, i *Nove capitoli
sull'arte matematica*, dedica il capitolo ottavo a una famiglia di problemi
che chiama *fangcheng*, «disposizione rettangolare». Il primo suona così: tre
covoni di grano di prima qualità, due di seconda e uno di terza rendono
trentanove misure; due di prima, tre di seconda e uno di terza ne rendono
trentaquattro; uno di prima, due di seconda e tre di terza ne rendono
ventisei. Quanto rende un covone di ciascuna qualità?

Il procedimento che il manuale insegna è disporre i numeri in colonne su un
tavoliere e sottrarre ripetutamente una colonna dall'altra, finché in fondo non
resta una qualità sola. Quasi diciotto secoli dopo, all'inizio dell'Ottocento,
Gauss lo sistemò lavorando sulle orbite dei pianetini, e da lui il procedimento
prende il nome di **eliminazione di Gauss**. Per governare quelle sottrazioni
il testo cinese si trova costretto a maneggiare quantità negative, che in
Europa avrebbero atteso il Rinascimento {cite}`shen1999nine`.

Oggi una libreria esegue il calcolo in pochi microsecondi, ma guardarci dentro
serve a un'altra cosa: lo stesso procedimento che trova la risposta dice anche
quando la risposta non c'è e quando invece ce ne sono infinite. Sono i due casi
tipici del machine learning. Con più equazioni che incognite, come in una
regressione, di norma non c'è una soluzione esatta; con più incognite che
equazioni, come in una rete che ha più parametri che esempi, ce ne sono
infinite.

## Le due letture di uno stesso sistema

Consideriamo le tre rese dei *Nove capitoli* e scriviamole con dei simboli al
posto delle parole. Chiamando $x$, $y$, $z$ la resa di un covone di prima,
seconda e terza qualità, si ottiene il sistema

$$
\begin{cases}
3x + 2y + \phantom{1}z = 39\\
2x + 3y + \phantom{1}z = 34\\
\phantom{1}x + 2y + 3z = 26
\end{cases}
$$

Ogni riga è una somma di prodotti fra un numero noto e un'incognita, senza
incognite moltiplicate fra loro, senza quadrati e senza radici. Per questo
le equazioni si dicono **lineari**, e si trattano con le matrici della
{doc}`sezione sull'algebra lineare <algebra-lineare>`.

Con le matrici il sistema diventa $\mathbf{A}\mathbf{x} = \mathbf{b}$, dove la
matrice $\mathbf{A}$ raccoglie i coefficienti, il vettore $\mathbf{x}$ le
incognite e il vettore $\mathbf{b}$ i termini noti. Nel problema del grano
$\mathbf{A}=\begin{pmatrix}3&2&1\\2&3&1\\1&2&3\end{pmatrix}$,
$\mathbf{x}=(x,y,z)^\top$ e $\mathbf{b}=(39,34,26)^\top$, e il prodotto
$\mathbf{A}\mathbf{x}$ dà, una riga dopo l'altra, i tre membri sinistri. Il
sistema si legge in due modi, per righe e per colonne, e le due letture
rispondono a domande diverse.

`````{tab} Elementare

In un colorificio, sul bancone ci sono tre barattoli di tinta base e un cliente
porta un colore da riprodurre. La domanda è: quante parti di ciascun barattolo?

Prima lettura, una riga per volta. Ogni riga del sistema è una promessa da
mantenere. Il colore del cliente contiene una certa quantità di rosso, e la
miscela che prepari dovrà contenerne altrettanto: ecco la prima equazione. Poi
c'è il giallo, e viene la seconda. Poi il blu. Tre pigmenti, tre promesse,
tutte da mantenere insieme. Risolvere vuol dire quindi trovare le dosi che le
tengono tutte e tre in piedi contemporaneamente.

Seconda lettura, una colonna per volta. Ogni barattolo ha una sua composizione
fissa, che è una colonna della tabella: tanto rosso, tanto giallo, tanto blu.
Versare mezzo litro del primo barattolo significa prendere metà di quella
colonna. La miscela finale è la somma dei tre versamenti, una mescolanza
delle colonne, e le dosi sono i numeri per cui le moltiplichi. La domanda
diventa: con quali dosi la mescolanza dei tre barattoli dà il colore
del cliente?

Le due letture descrivono lo stesso bancone. La prima guarda i pigmenti e chiede
se i conti tornano voce per voce; la seconda guarda i barattoli e chiede se,
dosandoli, ci si arriva. La seconda porta più lontano, perché rende
visibile un caso che la prima nasconde.

Se il terzo barattolo fosse una miscela dei primi due (un blu e un giallo, per
esempio, e un verde fatto mescolando quei due), non aggiungerebbe nessun colore
nuovo. Con tre barattoli, infatti, si farebbero le stesse tinte che si fanno
con due, e un colore fuori da quel repertorio non uscirebbe con nessuna dose.
Il colore chiesto sarebbe fuori portata, e il fallimento non dipenderebbe dalla
bravura di chi mescola ma da che cosa c'è sul bancone.

Torniamo al grano. I «barattoli» sono le tre qualità, e ciascuna ha una colonna
che dice quanti covoni di quella qualità entrano in ciascuna delle tre
combinazioni. Il «colore da riprodurre» è la lista delle tre rese osservate,
$39$, $34$ e $26$. Le dosi che cerchi sono le rese per covone.

`````

`````{tab} Superiore

Impilando i coefficienti in una matrice, le incognite in un vettore e i termini
noti in un altro, il sistema si scrive in forma compatta come

$$
\mathbf{A}\mathbf{x} = \mathbf{b},
\qquad
\mathbf{A} = \begin{pmatrix} 3 & 2 & 1\\ 2 & 3 & 1\\ 1 & 2 & 3\end{pmatrix},
\quad
\mathbf{x} = \begin{pmatrix} x\\ y\\ z\end{pmatrix},
\quad
\mathbf{b} = \begin{pmatrix} 39\\ 34\\ 26\end{pmatrix},
$$

dove $\mathbf{A}\in\mathbb{R}^{m\times n}$ raccoglie i coefficienti ($m$
equazioni, $n$ incognite), $\mathbf{x}\in\mathbb{R}^{n}$ le incognite e
$\mathbf{b}\in\mathbb{R}^{m}$ i termini noti.

**Lettura per righe.** Detta $\mathbf{a}_i^\top$ la riga $i$-esima, il sistema
è la collezione di $m$ prodotti scalari

$$
\mathbf{a}_i^\top \mathbf{x} = b_i, \qquad i = 1,\dots,m .
$$

Ciascuna equazione descrive un **iperpiano** di $\mathbb{R}^n$ (una retta se
$n=2$, un piano se $n=3$), e l'insieme delle soluzioni è la loro intersezione.

**Lettura per colonne.** Dette $\mathbf{a}_1,\dots,\mathbf{a}_n$ le colonne, lo
stesso sistema si riscrive

$$
x_1\mathbf{a}_1 + x_2\mathbf{a}_2 + \dots + x_n\mathbf{a}_n = \mathbf{b},
$$

cioè $\mathbf{b}$ è una **combinazione lineare** delle colonne di
$\mathbf{A}$, e le incognite sono i coefficienti della combinazione.

Le due letture sono algebricamente equivalenti, ma non guidano l'intuizione
allo stesso modo. La prima si visualizza bene per $n\le 3$ e smette di aiutare
subito dopo; la seconda resta utile in qualunque dimensione e basta a rispondere
alla domanda sull'esistenza: il sistema ammette soluzione se e solo se
$\mathbf{b}$ appartiene all'insieme delle combinazioni lineari delle colonne.
Quell'insieme è l’immagine di $\mathbf{A}$.

`````

## L'eliminazione: tre operazioni elementari di riga che non cambiano le risposte

Il metodo del tavoliere cinese è ancora quello che si insegna, e consiste in tre
operazioni.

`````{tab} Elementare

Le mosse lecite sono tre, e ciascuna si giustifica in poche parole.

*Scambiare due equazioni.* L'ordine in cui le promesse sono scritte non
cambia chi le mantiene.

*Moltiplicare un'equazione per un numero diverso da zero.* «Tre covoni rendono
trentanove» e «sei covoni rendono settantotto» dicono la stessa cosa. Lo zero
va escluso, perché moltiplicando per zero resta $0 = 0$, che è vero sempre e non
vincola più niente.

*Sommare a un'equazione un multiplo di un'altra.* È l'operazione che fa
tutto il lavoro. Chi soddisfa la prima promessa e la seconda soddisfa anche
la loro somma, e poiché l'operazione si può disfare (basta
risottrarre), le soluzioni restano le stesse.

Il gioco consiste nello scegliere i multipli in modo da far sparire le
incognite una alla volta. Nel problema del grano, ad esempio, conviene partire
dalla terza promessa, che ha un solo covone di prima qualità. Tolta tre volte
dalla prima e due volte dalla seconda, la $x$ sparisce da entrambe, e restano
$-4y - 8z = -39$ e $-y - 5z = -18$. Tolta quattro volte la seconda di queste
dalla prima, sparisce anche la $y$, e resta $12z = 33$, cioè $z = 2{,}75$.

Da lì si risale: la $z$ nota dà $y = 4{,}25$, e le due insieme danno $x =
9{,}25$. Il tavoliere alla fine ha la forma di una scaletta: la prima riga con
tre incognite, la seconda con due, l'ultima con una.

Il costo si stima a occhio. Per far sparire un'incognita da una riga servono
tanti prodotti quante sono le colonne; le righe da ripulire sono quasi tutte, e
le incognite da far sparire pure. Il lavoro cresce quindi come il cubo del
numero di incognite: triplicando le incognite si lavora ventisette volte tanto.

Per tre covoni è un attimo, ma per un sistema con un milione di incognite il
cubo diventa proibitivo; per questo su quelle taglie si usano metodi
che sfruttano una tabella quasi vuota o si accontentano di una soluzione
approssimata.

C'è infine un'astuzia pratica che nasce dall'aritmetica della macchina. Per far
sparire un'incognita bisogna dividere per il numero che le sta davanti, e se
quel numero è minuscolo la divisione gonfia tutto ciò che incontra, errori
compresi. Si sceglie perciò la riga: fra quelle disponibili, si porta in cima
quella in cui l'incognita da eliminare ha il coefficiente più grosso.

`````

`````{tab} Superiore

Le tre **operazioni elementari di riga** (scambio $R_i\leftrightarrow R_j$,
scalatura $R_i \leftarrow \alpha R_i$ con $\alpha\neq 0$, combinazione $R_i
\leftarrow R_i + \alpha R_j$) sono invertibili, quindi preservano l'insieme
delle soluzioni. Si applicano alla **matrice aumentata**
$[\,\mathbf{A}\mid\mathbf{b}\,]\in\mathbb{R}^{m\times(n+1)}$ e la portano in
**forma a scala** per righe, nella quale in ogni riga non nulla il primo
coefficiente diverso da zero (il **pivot**) sta a destra del pivot della riga
sopra. Nella forma ridotta, a cui si arriva proseguendo l'eliminazione verso
l'alto (Gauss-Jordan), ogni pivot vale $1$ ed è l'unico elemento non nullo
della sua colonna.

Il numero di pivot è un invariante della matrice, e prende il nome di
**rango**. Le colonne senza pivot corrispondono alle **variabili libere**: sono
i gradi di libertà che restano dopo aver imposto tutti i vincoli.

Il costo è $\Theta(n^3)$ operazioni per una matrice quadrata $n\times n$
(circa $n^3/3$ moltiplicazioni per la sola triangolarizzazione, cioè
$\tfrac{2}{3}n^3$ operazioni contando anche le somme), e la sostituzione
all'indietro ne aggiunge $\Theta(n^2)$.

Due avvertenze di pratica. In primo luogo, per risolvere
$\mathbf{A}\mathbf{x} = \mathbf{b}$ non si calcola $\mathbf{A}^{-1}$ per poi
moltiplicarla per $\mathbf{b}$. Invertire costa circa tre volte
l'eliminazione, e per giunta amplifica gli errori di arrotondamento più della
fattorizzazione diretta; `numpy.linalg.solve` fattorizza, e
`numpy.linalg.inv` serve solo quando l'inversa stessa è ciò che si cerca.

In secondo luogo, si adotta il **pivoting parziale**, cioè a ogni passo si
scambia in cima la riga con il coefficiente di modulo massimo nella colonna
corrente. Dividere per un pivot piccolo, infatti, moltiplica per un numero
enorme l'errore già presente nei dati, e la {doc}`sezione di analisi numerica
</Matematica/analisi-numerica>` mostra con quale grandezza si misura questo
rischio.

`````

La macchina ritrova i numeri che i *Nove capitoli* davano già duemila anni fa:
la prima qualità rende $9\tfrac{1}{4}$ misure per covone, la seconda
$4\tfrac{1}{4}$, la terza $2\tfrac{3}{4}$.

```python
import numpy as np

# I "Nove capitoli", problema 8.1. Una riga per resa, una colonna per
# qualità di grano; a destra le rese osservate.
A = np.array([[3.0, 2.0, 1.0],
              [2.0, 3.0, 1.0],
              [1.0, 2.0, 3.0]])
b = np.array([39.0, 34.0, 26.0])

x = np.linalg.solve(A, b)
print(x)                          # -> [9.25 4.25 2.75]
print(np.allclose(A @ x, b))      # rimessa dentro, la soluzione torna -> True
```

## Una soluzione, nessuna, infinite

Un sistema lineare può comportarsi in tre modi soltanto: o c'è esattamente una
soluzione, o non ce n'è nessuna, o ce ne sono infinite. Non capita mai che ce
ne siano solo due, perché $\mathbf{A}$ è lineare. Se
$\mathbf{A}\mathbf{u}=\mathbf{b}$ e $\mathbf{A}\mathbf{v}=\mathbf{b}$, ogni
punto $t\mathbf{u}+(1-t)\mathbf{v}$ della retta che li congiunge dà
$t\mathbf{b}+(1-t)\mathbf{b}=\mathbf{b}$.

`````{tab} Elementare

**Nessuna soluzione.** È il caso di due promesse che si contraddicono, ad
esempio «il primo barattolo e il secondo, insieme, devono fare due litri» e «il
primo barattolo e il secondo, insieme, devono fare tre litri». Chiunque si
presenti al bancone tornerà a casa a mani vuote, e non per colpa sua.

**Infinite soluzioni.** È il caso di una promessa che ne ripete un'altra con
parole diverse. Se la terza riga dice quello che dicono la prima e la seconda
messe insieme, allora sono due vincoli travestiti da tre, e resta un margine di
manovra.

Questo secondo caso capita più spesso di quanto ci si aspetti, e capita anche
quando le equazioni sembrano indipendentissime.

Un piccolo editore, ad esempio, ha due titoli in catalogo e due librerie che li
vendono. A fine mese sa quante copie ha venduto di ciascun titolo, novanta del
primo e sessanta del secondo, e quante ne ha vendute in ciascuna libreria,
settanta quella di centro e ottanta quella di periferia. Vuole ricostruire la
tabella completa: quante copie del primo titolo ha venduto la libreria di
centro, e così per le altre tre caselle.

Le caselle da trovare sono quattro, e quattro sono i numeri noti. Il problema
sembra determinato, ma non lo è. Il totale generale, infatti, si può leggere in
due modi, sommando i titoli ($90+60=150$) oppure le librerie ($70+80=150$), e i
due modi danno per forza lo stesso numero. Il quarto dato non aggiunge quindi
nulla ai primi tre, perché è già contenuto in essi. I vincoli veri sono tre, le
caselle quattro, e resta una direzione libera lungo cui muoversi.

Muoversi lungo quella direzione significa spostare una copia dalla casella in
alto a sinistra a quella in alto a destra, e compensare spostandone una dal
basso a destra al basso a sinistra; in questo modo i totali di riga e di
colonna restano identici. Ogni tabella così ottenuta rispetta tutti e quattro i
dati, che non bastano a stabilire quale sia quella giusta. Per sceglierne una
serve un'informazione in più che dai totali non arriva.

Un dettaglio dell'esempio ritorna spesso in machine learning. Le soluzioni
possibili sono infinite in senso matematico, ma non tutte hanno senso: una
libreria non può vendere meno di zero copie. Aggiungere questa richiesta
ritaglia dall'insieme infinito un tratto finito. In generale, quando i dati non
bastano, si aggiungono richieste ragionevoli finché la risposta non si
restringe.

`````

`````{tab} Superiore

La classificazione è governata dal confronto fra due ranghi. Secondo il teorema
di **Rouché–Capelli**, il sistema $\mathbf{A}\mathbf{x}=\mathbf{b}$ ammette
soluzioni se e solo se

$$
\operatorname{rank}(\mathbf{A})
= \operatorname{rank}([\,\mathbf{A}\mid\mathbf{b}\,]),
$$

dove $\operatorname{rank}$ è il numero di pivot dell'eliminazione e
$[\,\mathbf{A}\mid\mathbf{b}\,]$ la matrice aumentata. Quando la condizione
vale, detti $r$ il rango comune e $n$ il numero di incognite, l'insieme delle
soluzioni è un sottospazio affine di dimensione $n-r$: un solo punto per
$r=n$, infiniti punti altrimenti, con $n-r$ variabili libere.

Il senso della condizione si vede nella lettura per colonne. Un pivot in più
nella colonna dei termini noti significa una riga della forma $0 = 1$, cioè un
$\mathbf{b}$ che non appartiene all'insieme delle combinazioni delle colonne.

Consideriamo ora la tabella dell'editore. Ordinando le incognite come $(x_{11},
x_{12}, x_{21}, x_{22})$ e mettendo prima i due vincoli di riga e poi i due di
colonna, la matrice dei coefficienti e il vettore dei termini noti sono

$$
\mathbf{M} =
\begin{pmatrix}
1&1&0&0\\
0&0&1&1\\
1&0&1&0\\
0&1&0&1
\end{pmatrix},
\qquad
\mathbf{t} = \begin{pmatrix}90\\60\\70\\80\end{pmatrix}.
$$

Le quattro righe di $\mathbf{M}$ soddisfano
$\mathbf{r}_1+\mathbf{r}_2 = \mathbf{r}_3+\mathbf{r}_4$, quindi il rango è $3$
e non $4$; poiché $90+60 = 70+80$, anche la matrice aumentata ha rango $3$, il
sistema è compatibile e le soluzioni formano una retta. La direzione libera è
$(1,-1,-1,1)$, che $\mathbf{M}$ manda nel vettore nullo.

Aggiungendo i vincoli $x_{ij}\ge 0$, che i dati non impongono ma la realtà
sì, la retta si accorcia in un segmento limitato. È il passaggio che porta da
un problema di algebra lineare a uno di programmazione lineare, e da lì al
trasporto fra distribuzioni.

Questo schema, in cui si osservano i marginali di una tabella e se ne vorrebbe
ricostruire l'interno, ricompare in molti altri campi. In particolare, è la
ragione per cui conoscere due distribuzioni separate non determina la loro
congiunta, e la {doc}`sezione sul flow matching
</ModelliDiffusione/flow-matching>` ritroverà lo stesso schema nella forma del
trasporto fra due distribuzioni.

`````

```python
import numpy as np
# La tabella dell'editore: quattro caselle, quattro vincoli.
# Ordine delle incognite: x11, x12, x21, x22.
M = np.array([[1.0, 1.0, 0.0, 0.0],    # titolo 1: x11 + x12 = 90
              [0.0, 0.0, 1.0, 1.0],    # titolo 2: x21 + x22 = 60
              [1.0, 0.0, 1.0, 0.0],    # centro:   x11 + x21 = 70
              [0.0, 1.0, 0.0, 1.0]])   # periferia:x12 + x22 = 80
t = np.array([90.0, 60.0, 70.0, 80.0])

print(np.linalg.matrix_rank(M))                          # -> 3, su 4 righe
print(np.linalg.matrix_rank(np.column_stack([M, t])))    # -> 3: compatibile

# due tabelle diverse che rispettano tutti e quattro i totali
for s in (40.0, 55.0):
    v = np.array([s, 90 - s, 70 - s, s - 10])
    print(v, np.allclose(M @ v, t))
# -> [40. 50. 30. 30.] True
# -> [55. 35. 15. 45.] True
```

## Dove la matrice arriva, e che cosa lascia indietro

Le due domande poste a un sistema lineare (esiste una soluzione? è una sola?)
dipendono dalla matrice $\mathbf{A}\in\mathbb{R}^{m\times n}$ attraverso due
sottospazi. La sua immagine, in $\mathbb{R}^m$, dice quali termini noti
$\mathbf{b}$ ammettono almeno una soluzione; il suo nucleo, in $\mathbb{R}^n$,
dice quante ne ammettono quando ne ammettono una.

```{figure} ../figures/immagine-e-nucleo.svg
:name: fig-immagine-nucleo
:alt: "Il sole in alto a sinistra, con i suoi raggi paralleli; al centro un cubo di filo che rappresenta lo spazio di partenza con le sue tre direzioni, a destra il muro dello spazio di arrivo. Un raggio di luce attraversa il cubo: due punti distinti che stanno sullo stesso raggio cadono in un unico punto del muro, e la freccia che li unisce porta l'etichetta «nucleo». Il muro intero, in verde petrolio, è marcato «immagine: il piano del muro», mentre davanti al muro, fuori dal suo piano, una crocetta segna un punto che non è l'ombra di niente. In fondo il conto: tre direzioni di partenza uguale due che danno il muro, cioè l'immagine, più una che si perde nel nucleo."
:width: 88%

Immagine e nucleo di una trasformazione lineare dello spazio in sé, di rango
due, disegnati come l'ombra di un punto su un muro. Il sole proietta ogni punto
sul piano del muro lungo raggi paralleli. L'immagine è l'insieme dei punti che
stanno sul muro, un piano dentro lo spazio a tre dimensioni: un punto che sta
davanti al muro non è l'ombra di niente. Il nucleo è la direzione dei raggi: due
punti sullo stesso raggio hanno la stessa ombra, e spostarsi lungo il raggio non
la muove.
```

`````{tab} Elementare

Al sole, metti un oggetto davanti a un muro: sul muro compare un'ombra. La
trasformazione che una matrice compie assomiglia molto a questo, e
{numref}`fig-immagine-nucleo` la disegna così.

Che cosa si può ottenere sul muro. Ogni punto dello spazio ha la sua ombra, e le
ombre stanno tutte sul piano del muro: un punto che sta davanti o dietro il muro
non è l'ombra di niente. Il piano del muro è l’**immagine** della
trasformazione e risponde alla prima domanda: il sistema ha soluzione
soltanto se il termine noto sta lì sopra.

Che cosa il muro non registra. Fissa un punto dell'oggetto e fallo scorrere
lungo il raggio di sole che lo illumina: la sua ombra resta inchiodata
dov'era. Il muro non vede quello spostamento, e poiché i raggi del sole
arrivano tutti paralleli, lo spostamento invisibile è lo stesso per ogni punto
dell'oggetto.

L'insieme degli spostamenti invisibili si chiama **nucleo**, e risponde alla
seconda domanda: se esiste uno spostamento invisibile diverso dallo stare
fermi, allora ogni ombra ottenibile si ottiene in infiniti modi, perché a una
posizione buona se ne possono sempre aggiungere altre scorrendo lungo quella
direzione.

Con una lampada al posto del sole i raggi divergerebbero, ogni punto avrebbe
la sua direzione invisibile, e l'ombra smetterebbe di essere una
trasformazione di questo tipo.

Immagine e nucleo stanno in un rapporto stretto. Un punto può scorrere in tre
direzioni indipendenti (avanti, di lato, in alto); se una di queste è
invisibile, sul muro se ne vedono due, e sono le due direzioni del piano del
muro. Le direzioni di partenza si ripartiscono quindi fra quelle che si perdono
e quelle che arrivano, e la somma torna sempre: due più uno fa tre.

Se una direzione finisce nel nucleo, nessuna quantità di osservazioni la potrà
mai recuperare, perché all'arrivo quell'informazione non c'è, per quanto meglio
o più a lungo si misuri. Per questo certi parametri di un modello restano
indeterminati, e la tabella dell'editore non si ricostruisce dai soli totali.

`````

`````{tab} Superiore

Data $\mathbf{A}\in\mathbb{R}^{m\times n}$, si definiscono due sottospazi.

L’**immagine** (o spazio delle colonne) vive in $\mathbb{R}^m$ ed è definita
come

$$
\operatorname{im}(\mathbf{A})
= \{\mathbf{A}\mathbf{x} : \mathbf{x}\in\mathbb{R}^n\}
= \operatorname{span}\{\mathbf{a}_1,\dots,\mathbf{a}_n\},
$$

dove $\mathbf{a}_j$ è la $j$-esima colonna e $\operatorname{span}$ indica
l'insieme di tutte le loro combinazioni lineari. Il **nucleo** vive in
$\mathbb{R}^n$ ed è definito come

$$
\ker(\mathbf{A}) = \{\mathbf{x}\in\mathbb{R}^n : \mathbf{A}\mathbf{x}
= \mathbf{0}\} .
$$

Entrambi contengono l'origine e sono chiusi rispetto a somma e moltiplicazione
per uno scalare, cioè sono sottospazi vettoriali a pieno titolo. Nel disegno di
{numref}`fig-immagine-nucleo` il primo è il piano del muro, il secondo la
direzione lungo il raggio di luce.

Con questi due oggetti le domande sull'esistenza e sull'unicità diventano
enunciati secchi. Il sistema è compatibile se e solo se
$\mathbf{b}\in\operatorname{im}(\mathbf{A})$. Se $\mathbf{x}_p$ è una
soluzione particolare, l'insieme completo è

$$
\{\mathbf{x} : \mathbf{A}\mathbf{x} = \mathbf{b}\}
= \mathbf{x}_p + \ker(\mathbf{A}),
$$

perché la differenza di due soluzioni sta nel nucleo e viceversa. Nella tabella
dell'editore, ad esempio, $\mathbf{x}_p = (40,50,30,30)$ e il nucleo è generato
da $(1,-1,-1,1)$.

Le dimensioni dei due sottospazi non sono indipendenti. Il teorema di
nullità più rango afferma

$$
\dim \operatorname{im}(\mathbf{A}) + \dim \ker(\mathbf{A}) = n ,
$$

con $n$ il numero di colonne, cioè di incognite. La dimostrazione è
l'eliminazione stessa: le colonne di $\mathbf{A}$ nelle posizioni dei pivot
danno una base dell'immagine (non quelle della forma ridotta, poiché le
operazioni di riga alterano lo spazio delle colonne), le colonne senza pivot
danno una variabile libera ciascuna, e ogni colonna è dell'uno o dell'altro
tipo.

La conseguenza per i modelli prende il nome di **non identificabilità**. Se
$\ker(\mathbf{A})\neq\{\mathbf{0}\}$, due insiemi di parametri che differiscono
per un elemento del nucleo producono osservazioni identiche, e nessuna quantità
di dati potrà distinguerli. La {doc}`sezione sulla matematica di un modello
linguistico </Matematica/matematica-llm>` mostra questo stesso fenomeno sulle
matrici di query e chiave, che sono determinate solo a meno di una
trasformazione invertibile; e ogni volta che una libreria restituisce «la»
soluzione di un problema indeterminato, quella scelta è un'assunzione
aggiuntiva, non un risultato.

`````

## Il rango: quante direzioni davvero diverse

Il rango di una matrice è il massimo numero di colonne linearmente
indipendenti, cioè quante ne restano scartando, una alla volta, quelle che si
ottengono combinando le precedenti. Dice in quante direzioni indipendenti varia
una tabella di dati, e per questo torna ogni volta che una tabella va
compressa.

`````{tab} Elementare

Una tabella con cento righe sembra contenere cento informazioni. Può darsi però
che la terza riga sia la prima più la seconda, la quarta il doppio della prima,
e così via. In questo caso le righe che portano qualcosa di nuovo sono due, e
le altre novantotto si ricostruiscono da quelle. Quel conteggio, quante righe
portano qualcosa di nuovo, è il **rango**.

Lo stesso vale per le colonne, e non è ovvio: il numero di righe autonome e il
numero di colonne autonome coincidono sempre. Di conseguenza, una tabella di
mille righe e tre colonne ha rango al massimo tre, comunque siano fatti i
numeri.

Nel linguaggio delle ombre il rango è quante direzioni sopravvivono al
passaggio. Se l'oggetto può muoversi in tre direzioni e sul muro se ne vedono
due, il rango è due, e la direzione perduta è il nucleo.

Quando una trasformazione ha rango basso, la tabella che la descrive è grande
soltanto all'apparenza: si può riscrivere come il passaggio attraverso una
strettoia con poche corsie. Prima si comprime in poche direzioni, poi si
riespande. I numeri da regolare crollano di conseguenza: una tabella di
duecento righe per trecento colonne ne contiene sessantamila, ma
se la si ottiene facendo passare i dati per una strettoia a quattro corsie i
numeri diventano duemila, trenta volte di meno.

È l'idea su cui si basa LoRA, uno dei modi di adattare un modello linguistico a
un compito nuovo senza riaddestrarlo tutto: si lasciano fermi i numeri già
imparati e si affianca loro una correzione che passa per una strettoia. La
{doc}`sezione su ciò che viene dopo il pre-addestramento
</Transformers/post-training>` la racconta per esteso; il calcolo che la rende
possibile è tutto qui.

Nei dati veri, però, quasi nessuna riga è esattamente la somma di altre due:
c'è sempre un pulviscolo di rumore che la rende autonoma per un pelo. Se
si conta con l'aritmetica alla lettera, una tabella che in sostanza ha tre
direzioni ne dichiara duecento, e la risposta è vera e inutile. Conviene allora
chiedersi quante direzioni contano. Per rispondere bisogna ordinarle dalla più
importante alla meno importante, e lo fa la {doc}`sezione su ortogonalità e
proiezioni </Matematica/ortogonalita-proiezioni>`, che da quell'ordine ricava
anche quanto si perde tenendo solo le prime.

`````

`````{tab} Superiore

Un insieme di vettori $\{\mathbf{v}_1,\dots,\mathbf{v}_k\}$ è **linearmente
indipendente** se l'unica combinazione lineare che dà il vettore nullo è
quella a coefficienti tutti nulli. Una **base** di un sottospazio $V$ è un
insieme indipendente che lo genera; tutte le basi di $V$ hanno la stessa
cardinalità, e quel numero è la dimensione di $V$.

Il rango di $\mathbf{A}$ ammette quattro caratterizzazioni equivalenti:

- il numero di pivot nella forma a scala ridotta;
- $\dim\operatorname{im}(\mathbf{A})$, cioè il massimo numero di colonne
  linearmente indipendenti;
- il massimo numero di righe linearmente indipendenti (il rango per righe e
  quello per colonne coincidono);
- il numero di valori singolari non nulli nella decomposizione
  $\mathbf{A}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ della {doc}`sezione
  sull'algebra lineare <algebra-lineare>`.

Da qui seguono i limiti che si usano di continuo:
$\operatorname{rank}(\mathbf{A})\le\min(m,n)$, e soprattutto

$$
\operatorname{rank}(\mathbf{A}\mathbf{B})
\le \min\!\big(\operatorname{rank}(\mathbf{A}),
\operatorname{rank}(\mathbf{B})\big),
$$

che è la disuguaglianza dietro ogni collo di bottiglia. Fattorizzando, ad
esempio, $\mathbf{W}\in\mathbb{R}^{m\times n}$ come $\mathbf{B}\mathbf{A}$ con
$\mathbf{B}\in\mathbb{R}^{m\times r}$ e $\mathbf{A}\in\mathbb{R}^{r\times n}$,
si passa da $mn$ parametri a $r(m+n)$, e in cambio si rinuncia a tutte le
matrici di rango maggiore di $r$. È l'aritmetica di LoRA {cite}`hu2022lora`,
delle raccomandazioni per fattorizzazione e degli autoencoder lineari.

Nell'aritmetica in virgola mobile il rango così definito è una quantità
fragile: una perturbazione generica, anche minima, rende di rango pieno una
matrice singolare. Si usa quindi il **rango numerico**, cioè il numero di
valori singolari sopra una soglia proporzionale a $\sigma_{\max}$ e alla
precisione di macchina, che è ciò che `numpy.linalg.matrix_rank` calcola. La
domanda sensata su dati reali riguarda il decadimento dello spettro, non
l'annullamento; e il teorema dell’{doc}`approssimazione di rango basso
</Matematica/ortogonalita-proiezioni>` dice quanto costi fermarsi alle prime
$r$ direzioni.

`````

```python
import numpy as np
# rango pieno, rango carente, e il collo di bottiglia
rng = np.random.default_rng(0)

W_stretta = rng.normal(size=(200, 4)) @ rng.normal(size=(4, 300))
print(W_stretta.shape, np.linalg.matrix_rank(W_stretta))   # -> (200, 300) 4

print(200 * 300, 200 * 4 + 4 * 300)   # numeri da regolare -> 60000 2000

# la terza riga è la somma delle prime due: tre righe, due direzioni
T = np.array([[1.0, 2.0, 3.0],
              [0.0, 1.0, 1.0],
              [1.0, 3.0, 4.0]])
print(np.linalg.matrix_rank(T))       # -> 2

# e le righe autonome sono due anche contando per colonne
print(np.linalg.matrix_rank(T.T))     # -> 2
```

## Quando le equazioni sono più delle incognite

Nel problema del grano e nella tabella dell'editore i vincoli erano pochi o
giusti. Nel machine learning classico (regressione, modelli lineari e additivi)
il caso frequente è l'opposto: le equazioni sono migliaia (una per ogni esempio
osservato) e le incognite poche (i parametri del modello). Un sistema del
genere non ha quasi mai soluzione, perché i dati portano rumore e nessuna retta
passa esattamente per mille punti sparsi. In una rete che ha più parametri che
esempi accade il contrario: le soluzioni esatte sono infinite.

Rinunciare sarebbe assurdo, e la domanda si cambia: se non si può azzerare
l'errore, si cerca la scelta che lo rende più piccolo possibile. La
{doc}`sezione su ortogonalità e proiezioni
</Matematica/ortogonalita-proiezioni>` risponde a questa domanda con una
costruzione geometrica: il punto dell'immagine di $\mathbf{A}$ più vicino a
$\mathbf{b}$ è la sua proiezione ortogonale, l'ombra che darebbe un sole a
picco.

## In pratica, con NumPy

```python
import numpy as np

A = np.array([[3.0, 2.0, 1.0],
              [2.0, 3.0, 1.0],
              [1.0, 2.0, 3.0]])
b = np.array([39.0, 34.0, 26.0])

np.linalg.solve(A, b)        # sistema quadrato non singolare
np.linalg.matrix_rank(A)     # rango numerico -> 3
np.linalg.lstsq(A, b, rcond=None)[0]   # funziona anche se A è singolare
                                       # o rettangolare: minimi quadrati
```

`solve` pretende una matrice quadrata e invertibile, cioè con il nucleo ridotto
al solo zero (una matrice che non lo è si dice *singolare*). Solleva l'errore
`LinAlgError` solo se durante la fattorizzazione incontra un pivot esattamente
nullo: su una matrice singolare a meno degli arrotondamenti (la situazione
normale con i dati reali) non dice niente e restituisce un vettore enorme.

`lstsq` invece non si ferma mai: restituisce una soluzione anche quando ce ne
sono infinite (sceglie quella di norma minima) e anche quando non ce n'è
nessuna (sceglie quella che minimizza l'errore). È comodo, ma insidioso: una
risposta arriva comunque, e devi sapere a quale delle tre situazioni
corrisponde. Due colonne uguali a meno di un'unità di macchina bastano a
vederlo:

```python
import numpy as np
u = 2.0 ** -52                                  # un'unità di macchina
A = np.array([[1.0, 1.0], [1.0, 1.0 + u]])      # due colonne quasi uguali
b = np.array([1.0, 2.0])
print(np.linalg.matrix_rank(A))                 # rango numerico
print(np.linalg.solve(A, b))                    # nessun errore, numeri enormi
x = np.linalg.lstsq(A, b, rcond=None)[0]
print(np.round(x, 2), round(np.linalg.norm(A @ x - b), 2))  # e il suo residuo
```

```text
1
[-4.50359963e+15  4.50359963e+15]
[0.75 0.75] 0.71
```

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Un sistema lineare si legge in due modi. Per righe è una lista di
  promesse da mantenere tutte insieme; per colonne è la domanda «con quali
  dosi di questi barattoli ottengo quel colore?». La seconda lettura è quella
  che dice subito quando la richiesta è fuori portata.
- L’eliminazione è il metodo per risolverlo, e sta in tre mosse che non
  cambiano le risposte: scambiare due equazioni, moltiplicarne una per un
  numero diverso da zero, sommare a una il multiplo di un'altra. Si fanno
  sparire le incognite una alla volta finché non resta una scaletta.
- Gli esiti sono tre e solo tre: una soluzione, nessuna, oppure infinite. Quando
  le equazioni si ripetono travestite (i totali di riga e di colonna di una
  tabella sono un caso classico) restano delle scelte libere, e i dati da soli
  non bastano a scegliere.
- Quando le equazioni sono più delle incognite, di solito una soluzione esatta
  non c'è, e si cerca quella che sbaglia di meno.
- L’immagine è il piano delle ombre ottenibili, e dice se una richiesta si può
  soddisfare; il nucleo è la direzione lungo cui l'oggetto si sposta senza che
  l'ombra cambi, e dice che quell'informazione all'arrivo è perduta per sempre.
  Le direzioni di partenza si dividono fra le due, e la somma torna.
- Il rango conta quante righe (o colonne, che è lo stesso) portano
  qualcosa di nuovo. Una tabella grande di rango basso è grande solo
  all'apparenza: si riscrive come un passaggio attraverso una strettoia, e i
  numeri da regolare crollano.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- $\mathbf{A}\mathbf{x}=\mathbf{b}$ si legge per righe ($\mathbf{a}_i^\top
  \mathbf{x} = b_i$, intersezione di iperpiani) o per colonne ($\mathbf{b}$
  come combinazione lineare delle colonne). La seconda lettura risponde da
  sola all'esistenza.
- L’eliminazione di Gauss applica operazioni di riga invertibili alla
  matrice aumentata e la porta in forma a scala; costa
  $\Theta(n^3)$, richiede pivoting parziale per stabilità, e va preferita al
  calcolo esplicito di $\mathbf{A}^{-1}$.
- Rouché–Capelli: il sistema è compatibile se e solo se
  $\operatorname{rank}(\mathbf{A}) =
  \operatorname{rank}([\,\mathbf{A}\mid\mathbf{b}\,])$, e in tal caso le
  soluzioni formano un sottospazio affine di dimensione $n-r$.
- $\operatorname{im}(\mathbf{A})$ è lo span delle colonne,
  $\ker(\mathbf{A})$ ciò che va a zero, e l'insieme delle soluzioni è
  $\mathbf{x}_p + \ker(\mathbf{A})$. Vale
  $\dim\operatorname{im}+\dim\ker = n$: un nucleo non banale significa
  parametri non identificabili, cioè indistinguibili da qualunque dato.
- Il rango è il numero di pivot, la dimensione dell'immagine, il massimo
  numero di righe o colonne indipendenti e il numero di valori singolari non
  nulli. Da $\operatorname{rank}(\mathbf{A}\mathbf{B}) \le
  \min(\operatorname{rank}\mathbf{A}, \operatorname{rank}\mathbf{B})$ discende
  ogni fattorizzazione a collo di bottiglia. Su dati reali si guarda il
  decadimento dei valori singolari, non l'annullamento.
```
`````

I sistemi lineari hanno dato tre domande da porre a una trasformazione: che
cosa produce (l'immagine), che cosa perde (il nucleo), quante direzioni
indipendenti conserva (il rango). Resta il caso più comune nei dati, quello con
più equazioni che incognite, dove di norma una soluzione esatta non esiste. In quel
caso, invece di cercare il punto che soddisfa
$\mathbf{A}\mathbf{x}=\mathbf{b}$, si cerca il punto dell'immagine di
$\mathbf{A}$ più vicino a $\mathbf{b}$, e lo trovano le proiezioni.
