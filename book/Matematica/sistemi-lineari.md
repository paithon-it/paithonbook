# Sistemi lineari: quando i vincoli bastano, e quando no

Un manuale cinese compilato intorno al primo secolo, i *Nove capitoli
sull'arte matematica*, dedica il capitolo ottavo a una famiglia di problemi
che chiama *fangcheng*, «disposizione rettangolare». Il primo suona così: tre
covoni di grano di prima qualità, due di seconda e uno di terza rendono
trentanove misure; due di prima, tre di seconda e uno di terza ne rendono
trentaquattro; uno di prima, due di seconda e tre di terza ne rendono
ventisei. Quanto rende un covone di ciascuna qualità?

Il procedimento che il libro insegna è disporre i numeri in colonne su un
tavoliere e sottrarre ripetutamente una colonna dall'altra, finché in fondo
non resta una qualità sola. Duemila anni dopo si chiama **eliminazione**, dal
nome di Carl Friedrich Gauss che la sistemò all'inizio dell'Ottocento
lavorando sulle orbite dei pianetini; e per governare quelle sottrazioni il
testo cinese si trova costretto a maneggiare quantità negative, che in Europa
avrebbero atteso il Rinascimento {cite}`shen1999nine`.

Il conto in sé oggi lo fa una libreria in un microsecondo. Guardarci dentro
serve a un'altra cosa: lo stesso procedimento che trova la
risposta dice anche **quando la risposta non c'è**, e quando invece ce ne sono
infinite. In machine learning si vive quasi sempre in uno di quei due casi, e
riconoscerli è metà del mestiere.

## Le due letture di uno stesso sistema

Scrivere le tre pesate dei *Nove capitoli* con dei simboli al posto delle
parole dà tre equazioni. Chiamando $x$, $y$, $z$ la resa di un covone di prima,
seconda e terza qualità:

$$
\begin{cases}
3x + 2y + \phantom{1}z = 39\\
2x + 3y + \phantom{1}z = 34\\
\phantom{1}x + 2y + 3z = 26
\end{cases}
$$

Ogni riga somma dei prodotti fra un numero noto e un'incognita, e nient'altro:
niente incognite moltiplicate fra loro, niente quadrati, niente seni. È questo
che le rende **lineari**, ed è la ragione per cui si maneggiano con le matrici
della sezione precedente.

Lo stesso mucchio di numeri si può leggere in due modi, e conviene averli
tutti e due in testa, perché rispondono a domande diverse.

`````{tab} Elementare

Immagina un colorificio. Sul bancone ci sono tre barattoli di tinta base, e un
cliente porta un campione di colore da riprodurre. La domanda è: quante parti
di ciascun barattolo?

**Prima lettura, una riga per volta.** Ogni riga del sistema è una promessa da
mantenere. Il campione contiene una certa quantità di rosso, e la miscela che
prepari dovrà contenerne esattamente altrettanto: ecco la prima equazione. Poi
c'è il giallo, e viene la seconda. Poi il blu. Tre pigmenti, tre promesse,
tutte da mantenere insieme. Risolvere vuol dire trovare le dosi che le tengono
tutte e tre in piedi contemporaneamente.

**Seconda lettura, una colonna per volta.** Ogni barattolo ha una sua
composizione fissa, che è una colonna della tabella: tanto rosso, tanto
giallo, tanto blu. Versare mezzo litro del primo barattolo significa prendere
metà di quella colonna. La miscela finale è la somma dei tre versamenti, cioè
una **mescolanza delle colonne**, e le dosi sono i numeri per cui le
moltiplichi. La domanda diventa: con quali dosi la mescolanza dei tre
barattoli fa esattamente il colore del campione?

Le due letture descrivono lo stesso bancone. La prima guarda i pigmenti e
chiede se i conti tornano voce per voce; la seconda guarda i barattoli e
chiede se, dosandoli, ci si arriva. La seconda è quella che porta più lontano,
perché rende visibile un caso che la prima nasconde: se i tre barattoli fossero
tutti sfumature di verde, nessuna dose, per quanto astuta, tirerebbe fuori un
rosso. Il colore chiesto sarebbe **fuori portata**, e il fallimento non
dipenderebbe dalla bravura di chi mescola ma da che cosa c'è sul bancone.

Torniamo al grano. I «barattoli» sono le tre qualità, e ciascuna ha una
colonna che dice quanti covoni di quella qualità entrano in ciascuna delle tre
pesate. Il «campione da riprodurre» è la lista delle tre rese osservate, $39$,
$34$ e $26$. Le dosi che cerchi sono le rese per covone.

`````

`````{tab} Superiore

Impilando i coefficienti in una matrice, le incognite in un vettore e i
termini noti in un altro, il sistema si scrive in una riga:

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

cioè: $\mathbf{b}$ è una **combinazione lineare** delle colonne di
$\mathbf{A}$, e le incognite sono i coefficienti della combinazione.

Le due letture sono algebricamente equivalenti e cognitivamente no. La prima
si visualizza bene per $n\le 3$ e smette di aiutare subito dopo. La seconda
resta utile in qualunque dimensione e risponde da sola alla domanda
sull'esistenza: il sistema ammette soluzione se e solo se $\mathbf{b}$
appartiene all'insieme delle combinazioni lineari delle colonne. Quell'insieme
ha un nome, e più avanti si chiamerà l’**immagine** di $\mathbf{A}$.

`````

## L'eliminazione: tre mosse che non cambiano le risposte

Il metodo del tavoliere cinese è ancora quello che si insegna, e sta in tre
gesti.

`````{tab} Elementare

Le mosse lecite sono tre, e ciascuna si giustifica in mezza riga.

*Scambiare due equazioni.* L'ordine in cui le promesse sono scritte non
cambia chi le mantiene.

*Moltiplicare un'equazione per un numero diverso da zero.* «Tre covoni rendono
trentanove» e «sei covoni rendono settantotto» dicono la stessa cosa. Lo zero
va escluso: moltiplicando per zero resta $0 = 0$, che è vero sempre e non
vincola più niente.

*Sommare a un'equazione un multiplo di un'altra.* Questa è la mossa che fa
tutto il lavoro. Chi soddisfa la prima promessa e la seconda soddisfa anche
la loro somma; e siccome la mossa si può disfare (basta
risottrarre), non si è perso niente per strada.

Il gioco consiste nello scegliere i multipli in modo da far **sparire** le
incognite una alla volta. Nel problema del grano si toglie la $x$ dalla
seconda e dalla terza riga, poi la $y$ dalla terza: a quel punto l'ultima riga
parla di una sola incognita, la si ricava, la si porta su nella penultima, e
si risale fino in cima. Il tavoliere alla fine ha la forma di una scaletta:
la prima riga con tre incognite, la seconda con due, l'ultima con una.

Il costo si conta a occhio. Per far sparire un'incognita da una riga servono
tanti prodotti quante sono le colonne; le righe da ripulire sono quasi tutte,
e le incognite da far sparire pure. Il lavoro cresce quindi come il cubo del
numero di incognite: triplicando le incognite si lavora ventisette volte
tanto. Per tre covoni è un attimo, per un sistema con un milione di incognite
il cubo diventa proibitivo, ed è la ragione per cui su quelle taglie si usano
metodi che si accontentano di una soluzione approssimata.

C'è infine un'astuzia pratica che nasce dall'aritmetica della macchina. Per
far sparire un'incognita bisogna dividere per il numero che le sta davanti,
e se quel numero è minuscolo la divisione gonfia tutto quello che tocca,
errori compresi. Il rimedio è scegliersi la riga: fra quelle disponibili, si
porta in cima quella in cui l'incognita da eliminare ha il coefficiente più
grosso.

`````

`````{tab} Superiore

Le tre **operazioni elementari di riga** (scambio $R_i\leftrightarrow R_j$,
scalatura $R_i \leftarrow \alpha R_i$ con $\alpha\neq 0$, combinazione
$R_i \leftarrow R_i + \alpha R_j$) sono invertibili, quindi preservano
l'insieme delle soluzioni. Si applicano alla **matrice aumentata**
$[\,\mathbf{A}\mid\mathbf{b}\,]\in\mathbb{R}^{m\times(n+1)}$ e la portano in
**forma a scala ridotta** per righe: in ogni riga non nulla il primo
coefficiente diverso da zero vale $1$ (si chiama **pivot**), sta a destra del
pivot della riga sopra, ed è l'unico elemento non nullo della sua colonna.

Il numero di pivot è un invariante della matrice, e prende il nome di
**rango**. Le colonne senza pivot corrispondono alle **variabili libere**:
sono i gradi di libertà che restano dopo aver imposto tutti i vincoli.

Il costo è $\Theta(n^3)$ operazioni per una matrice quadrata $n\times n$
(circa $n^3/3$ moltiplicazioni per la sola triangolarizzazione, cioè
$\tfrac{2}{3}n^3$ operazioni contando anche le somme), e la sostituzione
all'indietro ne aggiunge $\Theta(n^2)$.

Due avvertenze di pratica. La prima: per risolvere $\mathbf{A}\mathbf{x} =
\mathbf{b}$ non si calcola $\mathbf{A}^{-1}$ e poi la si moltiplica per
$\mathbf{b}$. Invertire costa circa tre volte l'eliminazione, e per giunta
amplifica gli errori di arrotondamento più della fattorizzazione diretta;
`numpy.linalg.solve` fattorizza, e `numpy.linalg.inv` compare quasi solo nei
testi. La seconda: si adotta il **pivoting parziale**, cioè a ogni passo si
scambia in cima la riga con il coefficiente di modulo massimo nella colonna
corrente. Dividere per un pivot piccolo moltiplica per un numero enorme
l'errore già presente nei dati, e la {doc}`sezione di analisi numerica
</Matematica/analisi-numerica>` mostra con quale grandezza si misura questo
rischio.

`````

Il conto del grano, fatto dalla macchina, torna a quello che i *Nove capitoli*
davano già duemila anni fa: la prima qualità rende $9\tfrac{1}{4}$ misure per
covone, la seconda $4\tfrac{1}{4}$, la terza $2\tfrac{3}{4}$.

```python
import numpy as np

# I "Nove capitoli", problema 8.1. Una riga per pesata, una colonna per
# qualita' di grano; a destra le rese osservate.
A = np.array([[3.0, 2.0, 1.0],
              [2.0, 3.0, 1.0],
              [1.0, 2.0, 3.0]])
b = np.array([39.0, 34.0, 26.0])

x = np.linalg.solve(A, b)
print(x)                          # -> [9.25 4.25 2.75]
print(np.allclose(A @ x, b))      # rimessa dentro, la soluzione torna -> True
```

## Tre esiti, e quello di mezzo è il più interessante

Un sistema lineare può comportarsi in tre modi soltanto, e un quarto non
esiste: o c'è esattamente una soluzione, o non ce n'è nessuna, o ce ne sono
infinite. Non capita mai che ce ne siano due e basta, e la ragione si vede
bene: se $\mathbf{u}$ e $\mathbf{v}$ risolvono, risolve anche ogni punto della
retta che le congiunge.

`````{tab} Elementare

**Nessuna soluzione.** Due promesse che si contraddicono. «Il primo barattolo
e il secondo, insieme, devono fare due litri» e «il primo barattolo e il
secondo, insieme, devono fare tre litri»: chiunque si presenti al bancone
tornerà a casa a mani vuote, e non per colpa sua.

**Infinite soluzioni.** Una promessa che ripete un'altra con parole diverse.
Se la terza riga dice esattamente quello che dicono la prima e la seconda
messe insieme, allora sono due vincoli travestiti da tre, e resta un margine
di manovra.

Questo secondo caso capita più spesso di quanto ci si aspetti, e capita anche
quando le equazioni sembrano indipendentissime. Ecco una scena vera. Un
piccolo editore ha due titoli in catalogo e due librerie che li vendono. A
fine mese sa quanto ha venduto ciascun titolo, novanta copie il primo e
sessanta il secondo, e quante ne ha vendute ciascuna libreria, settanta quella
di centro e ottanta quella di periferia. Vuole ricostruire la
tabella completa: quante copie del primo titolo ha venduto la libreria di
centro, e così per le altre tre caselle.

Quattro caselle da trovare, quattro numeri noti. Sembra chiuso, e non lo è. Il
totale generale si può leggere in due modi, sommando i titoli ($90+60=150$)
oppure le librerie ($70+80=150$), e i due modi danno per forza lo stesso
numero. Quindi il quarto dato non aggiunge niente ai primi tre: è già
contenuto in essi. I vincoli veri sono tre, le caselle quattro, e resta una
direzione libera lungo cui muoversi.

Muoversi lungo quella direzione significa spostare una copia dalla casella in
alto a sinistra a quella in alto a destra, e compensare spostandone una dal
basso a destra al basso a sinistra: i totali di riga e di colonna restano
identici. Ogni tabella così ottenuta rispetta tutti e quattro i dati, e i dati
da soli non sanno dire quale sia quella giusta. Per sceglierne una serve
un'informazione in più che dai totali non arriva.

Un dettaglio della scena ritorna spesso in machine learning. Le soluzioni
possibili sono infinite in senso matematico, ma non tutte hanno senso: una
libreria non può vendere meno di zero copie. Aggiungere questa richiesta
ritaglia dall'insieme infinito un tratto finito, ed è il primo esempio di una
mossa che si rivedrà spesso: quando i dati non bastano, si aggiungono richieste
ragionevoli finché la risposta non si restringe.

`````

`````{tab} Superiore

La classificazione è governata dal confronto fra due ranghi, ed è il teorema
di **Rouché–Capelli**: il sistema $\mathbf{A}\mathbf{x}=\mathbf{b}$ ammette
soluzioni se e solo se

$$
\operatorname{rank}(\mathbf{A})
= \operatorname{rank}([\,\mathbf{A}\mid\mathbf{b}\,]),
$$

dove $\operatorname{rank}$ è il numero di pivot introdotto sopra e
$[\,\mathbf{A}\mid\mathbf{b}\,]$ la matrice aumentata. Quando la condizione
vale, detta $r$ il rango comune e $n$ il numero di incognite, l'insieme delle
soluzioni è un sottospazio affine di dimensione $n-r$: unica per $r=n$,
infinita altrimenti, con $n-r$ variabili libere.

Il senso della condizione si legge nella lettura per colonne: un pivot in più
nella colonna dei termini noti significa una riga della forma
$0 = 1$, cioè $\mathbf{b}$ che sporge fuori dalle combinazioni delle colonne.

Sulla tabella dell'editore, ordinando le incognite come
$(x_{11}, x_{12}, x_{21}, x_{22})$ e mettendo prima i due vincoli di riga e
poi i due di colonna:

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
un problema di algebra lineare a uno di **programmazione lineare**, e da lì al
trasporto fra distribuzioni.

Questo schema, dove si osservano i **marginali** di una tabella e si vorrebbe
ricostruire la tabella, ricompare in tutto il libro sotto altri nomi: è la
ragione per cui conoscere due distribuzioni separate non determina la loro
congiunta, ed è la struttura del problema che la {doc}`sezione sul flow
matching </ModelliDiffusione/flow-matching>` incontrerà nella forma del
trasporto fra due distribuzioni.

`````

```python
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

Le due domande di poco fa (esiste una soluzione? è una sola?) si decidono
guardando la sola matrice: è lei a dire quali termini noti ammettono una
risposta, e quante ne ammettono quando ne ammettono. Sono due insiemi, e
stanno da parti opposte: uno vive nello spazio di arrivo, l'altro in quello di
partenza.

```{figure} ../figures/immagine-e-nucleo.svg
:name: fig-immagine-nucleo
:alt: "Una lampada a sinistra, al centro un cubo di filo che rappresenta lo spazio di partenza con le sue tre direzioni, a destra il muro dello spazio di arrivo. Un raggio di luce attraversa il cubo: due punti distinti che stanno sullo stesso raggio cadono in un unico punto del muro, e la freccia che li unisce porta l'etichetta «nucleo». Sul muro una regione ovale in verde petrolio è marcata «immagine, le ombre che si possono fare», mentre più in basso una crocetta segna una sagoma fuori portata. In fondo il conto: tre direzioni di partenza uguale due che arrivano sul muro più una che si perde nel nucleo."
:width: 88%

L'immagine e il nucleo di una trasformazione, disegnati come l'ombra di un
oggetto su un muro. Sul muro esiste una regione che si può raggiungere e una
che resta fuori portata; nello spazio dell'oggetto esiste una direzione lungo
cui ci si può spostare senza che l'ombra cambi di un millimetro.
```

`````{tab} Elementare

Accendi una lampada e metti un oggetto fra la lampada e il muro: sul muro
compare un'ombra. La trasformazione che una matrice compie assomiglia molto a
questo, e {numref}`fig-immagine-nucleo` la disegna così.

**Che cosa si può ottenere sul muro.** Muovendo l'oggetto in tutti i modi
possibili si ottengono tante ombre diverse, ma non tutte le sagome
immaginabili: la lampada e la forma dell'oggetto decidono un repertorio, e
fuori da quello non si va. Se il cliente porta una sagoma che nel repertorio
non c'è, non esiste posizione dell'oggetto che la produca. Questo repertorio
è l’**immagine** della trasformazione, ed è la risposta alla prima domanda:
il sistema ha soluzione soltanto se il termine noto sta lì dentro.

**Che cosa il muro non registra.** Fissa un punto dell'oggetto e fallo scorrere
lungo il raggio che lo illumina, avvicinandolo o allontanandolo dalla lampada:
la sua ombra resta inchiodata dov'era. Quello spostamento il muro non lo vede.
L'insieme degli spostamenti invisibili si chiama **nucleo**, e risponde alla
seconda domanda: se esiste uno spostamento invisibile diverso dallo stare
fermi, allora ogni ombra ottenibile si ottiene in infiniti modi, perché a una
posizione buona se ne possono sempre aggiungere altre scorrendo lungo quel
raggio.

Le due cose stanno in un rapporto stretto, e il conto è il seguente. Un punto
può scorrere in tre direzioni indipendenti (avanti, di lato, in alto); se una
di queste è invisibile, sul muro se ne vedono due, e le ombre che si possono
ottenere coprono una superficie piatta e non di più. Le direzioni di partenza si
ripartiscono fra quelle che si perdono e quelle che arrivano, e la somma torna
sempre: due più uno fa tre.

Dalla scena si porta via una morale che vale ben oltre le ombre. **Se una
direzione finisce nel nucleo, nessuna quantità di osservazioni la potrà mai
recuperare**: non è questione di misurare meglio o più a lungo, quella
informazione all'arrivo non c'è. È il motivo per cui certi parametri di un
modello restano indeterminati per sempre, e perché la tabella dell'editore non
si può ricostruire dai soli totali per quanto a lungo li si guardi.

`````

`````{tab} Superiore

Data $\mathbf{A}\in\mathbb{R}^{m\times n}$, si definiscono due sottospazi.

L’**immagine** (o spazio delle colonne) vive in $\mathbb{R}^m$:

$$
\operatorname{im}(\mathbf{A})
= \{\mathbf{A}\mathbf{x} : \mathbf{x}\in\mathbb{R}^n\}
= \operatorname{span}\{\mathbf{a}_1,\dots,\mathbf{a}_n\},
$$

dove $\mathbf{a}_j$ è la $j$-esima colonna e $\operatorname{span}$ indica
l'insieme di tutte le loro combinazioni lineari. Il **nucleo** vive in
$\mathbb{R}^n$:

$$
\ker(\mathbf{A}) = \{\mathbf{x}\in\mathbb{R}^n : \mathbf{A}\mathbf{x}
= \mathbf{0}\} .
$$

Entrambi contengono l'origine e sono chiusi rispetto a somma e moltiplicazione
per uno scalare, cioè sono sottospazi vettoriali a pieno titolo. Nel disegno di
{numref}`fig-immagine-nucleo` il primo è la regione raggiungibile del muro, il
secondo la direzione lungo il raggio di luce.

Con questi due oggetti le domande di prima diventano enunciati secchi. Il
sistema è compatibile se e solo se $\mathbf{b}\in\operatorname{im}(\mathbf{A})$.
E se $\mathbf{x}_p$ è una soluzione particolare, l'insieme completo è

$$
\{\mathbf{x} : \mathbf{A}\mathbf{x} = \mathbf{b}\}
= \mathbf{x}_p + \ker(\mathbf{A}),
$$

perché la differenza di due soluzioni sta nel nucleo e viceversa. Nella
tabella dell'editore $\mathbf{x}_p = (40,50,30,30)$ e il nucleo è generato da
$(1,-1,-1,1)$.

Le dimensioni dei due sottospazi non sono indipendenti. Il **teorema di
nullità più rango** afferma

$$
\dim \operatorname{im}(\mathbf{A}) + \dim \ker(\mathbf{A}) = n ,
$$

con $n$ il numero di colonne, cioè di incognite. La dimostrazione è
l'eliminazione stessa: le colonne con pivot danno una base dell'immagine, le
colonne senza pivot danno una variabile libera ciascuna, e ogni colonna è
dell'uno o dell'altro tipo.

La conseguenza modellistica ha un nome: **non identificabilità**. Se
$\ker(\mathbf{A})\neq\{\mathbf{0}\}$, due insiemi di parametri che
differiscono per un elemento del nucleo producono osservazioni identiche, e
nessuna quantità di dati potrà distinguerli. La {doc}`sezione sulla matematica
di un modello linguistico </Matematica/matematica-llm>` mostra questo stesso
fenomeno sulle matrici di query e chiave, che sono determinate solo a meno di
una trasformazione invertibile; e ogni volta che una libreria restituisce «la»
soluzione di un problema indeterminato, quella scelta è un'assunzione
aggiuntiva, non un risultato.

`````

## Il rango: quante direzioni davvero diverse

Sotto tutti i ragionamenti fatti finora c'è un numero solo, e adesso merita di
essere guardato per sé: è la grandezza che il resto del libro nomina più
spesso.

`````{tab} Elementare

Una tabella con cento righe sembra contenere cento informazioni. Può darsi
però che la terza riga sia la prima più la seconda, la quarta il doppio della
prima, e così via: allora le righe che portano qualcosa di nuovo sono due, e
le altre novantotto si ricostruiscono da quelle. Quel conteggio, quante righe
portano qualcosa di nuovo, è il **rango**.

Vale la stessa cosa per le colonne, ed è un fatto tutt'altro che ovvio: il
numero di righe autonome e il numero di colonne autonome coincidono sempre.
Una tabella di mille righe e tre colonne ha rango al massimo tre, comunque
siano fatti i numeri.

Nel linguaggio delle ombre il rango è quante direzioni sopravvivono al
passaggio. Se l'oggetto può muoversi in tre direzioni e sul muro se ne vedono
due, il rango è due, e la direzione perduta è il nucleo.

Ecco perché la parola torna così spesso. Quando una trasformazione ha rango
basso, la tabella che la descrive è **grande soltanto all'apparenza**: si può
riscrivere come il passaggio attraverso una strettoia con poche corsie. Prima
si comprime in poche direzioni, poi si riespande. I numeri da regolare crollano
di conseguenza: una tabella di duecento righe per trecento colonne ne contiene
sessantamila, ma se la si ottiene facendo passare i dati per una strettoia a
quattro corsie i numeri diventano duemila, trenta volte di meno.

È esattamente la mossa con cui oggi si adattano i modelli linguistici a un
compito nuovo senza riaddestrarli: si lasciano fermi i numeri già imparati e
si affianca loro una correzione che passa per una strettoia stretta. La
{doc}`sezione su ciò che viene dopo il pre-addestramento
</Transformers/post-training>` la racconta per esteso; qui interessa il conto
che la rende possibile, e il conto è tutto qui.

E c'è un modo di sbagliarlo, che conviene conoscere. Nei dati veri quasi
nessuna riga è **esattamente** la somma di altre due: c'è sempre un pulviscolo
di rumore che la rende autonoma per un pelo. Se si conta con l'aritmetica alla
lettera, una tabella che in sostanza ha tre direzioni ne dichiara duecento, e
la risposta è vera e inutile. La domanda sensata non è quante direzioni ci
sono, ma quante contano davvero: la sezione precedente ne ha già dato la
misura, ordinando le direzioni dalla più importante alla meno.

`````

`````{tab} Superiore

Un insieme di vettori $\{\mathbf{v}_1,\dots,\mathbf{v}_k\}$ è **linearmente
indipendente** se l'unica combinazione lineare che dà il vettore nullo è
quella a coefficienti tutti nulli. Una **base** di un sottospazio $V$ è un
insieme indipendente che lo genera; tutte le basi di $V$ hanno la stessa
cardinalità, e quel numero è la **dimensione** di $V$.

Il **rango** di $\mathbf{A}$ ammette quattro caratterizzazioni equivalenti,
ed è utile averle tutte:

- il numero di pivot nella forma a scala ridotta;
- $\dim\operatorname{im}(\mathbf{A})$, cioè il massimo numero di colonne
  linearmente indipendenti;
- il massimo numero di righe linearmente indipendenti (il rango per righe e
  quello per colonne coincidono);
- il numero di valori singolari non nulli nella decomposizione
  $\mathbf{A}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ della sezione
  precedente.

Da qui seguono i limiti che si usano di continuo:
$\operatorname{rank}(\mathbf{A})\le\min(m,n)$, e soprattutto

$$
\operatorname{rank}(\mathbf{A}\mathbf{B})
\le \min\!\big(\operatorname{rank}(\mathbf{A}),
\operatorname{rank}(\mathbf{B})\big),
$$

che è la disuguaglianza dietro ogni collo di bottiglia: fattorizzando
$\mathbf{W}\in\mathbb{R}^{m\times n}$ come $\mathbf{B}\mathbf{A}$ con
$\mathbf{B}\in\mathbb{R}^{m\times r}$ e $\mathbf{A}\in\mathbb{R}^{r\times n}$
si passa da $mn$ parametri a $r(m+n)$, e in cambio si rinuncia a tutte le
matrici di rango maggiore di $r$. È l'aritmetica di LoRA, delle
raccomandazioni per fattorizzazione e degli autoencoder lineari.

Nell'aritmetica in virgola mobile il rango così definito è una quantità
fragile: una perturbazione arbitrariamente piccola rende una matrice singolare
di rango pieno. Si usa quindi il **rango numerico**, cioè il numero di valori
singolari sopra una soglia proporzionale a $\sigma_{\max}$ e alla precisione
di macchina, che è ciò che `numpy.linalg.matrix_rank` calcola. La domanda
sensata su dati reali riguarda il **decadimento** dello spettro, non
l'annullamento; e quanto costi fermarsi alle prime $r$ direzioni lo dice il
teorema dell’{doc}`approssimazione di rango basso
</Matematica/ortogonalita-proiezioni>`.

`````

```python
# rango pieno, rango carente, e il collo di bottiglia
rng = np.random.default_rng(0)

W_stretta = rng.normal(size=(200, 4)) @ rng.normal(size=(4, 300))
print(W_stretta.shape, np.linalg.matrix_rank(W_stretta))   # -> (200, 300) 4

print(200 * 300, 200 * 4 + 4 * 300)   # numeri da regolare -> 60000 2000

# la terza riga e' la somma delle prime due: tre righe, due direzioni
T = np.array([[1.0, 2.0, 3.0],
              [0.0, 1.0, 1.0],
              [1.0, 3.0, 4.0]])
print(np.linalg.matrix_rank(T))       # -> 2

# e le righe autonome sono due anche contando per colonne
print(np.linalg.matrix_rank(T.T))     # -> 2
```

## Quando le equazioni sono più delle incognite

Nei problemi visti finora i vincoli erano pochi o giusti. Nel machine
learning il caso normale è l'opposto: le equazioni sono migliaia (una per
esempio osservato) e le incognite poche (i parametri del modello). Un sistema
del genere non ha quasi mai soluzione, perché i dati portano rumore e nessuna
retta passa esattamente per mille punti sparsi.

Rinunciare sarebbe assurdo, e la domanda si cambia: se non si può azzerare
l'errore, si cerca la scelta che lo rende **più piccolo possibile**. La
{doc}`sezione su ortogonalità e proiezioni </Matematica/ortogonalita-proiezioni>`
risponde a questa domanda, e la risposta ha una forma geometrica sorprendente,
che è poi la stessa ombra della lampada e del muro.

## In pratica, con NumPy

```python
import numpy as np

A = np.array([[3.0, 2.0, 1.0],
              [2.0, 3.0, 1.0],
              [1.0, 2.0, 3.0]])
b = np.array([39.0, 34.0, 26.0])

np.linalg.solve(A, b)        # sistema quadrato non singolare
np.linalg.matrix_rank(A)     # rango numerico -> 3
np.linalg.lstsq(A, b, rcond=None)[0]   # funziona anche se A e' singolare
                                       # o rettangolare: minimi quadrati
```

`solve` pretende una matrice quadrata e invertibile, e su una matrice
singolare solleva `LinAlgError`. `lstsq` non si ferma mai: restituisce una
soluzione anche quando ce ne sono infinite (sceglie quella di norma minima) e
anche quando non ce n'è nessuna (sceglie quella che minimizza l'errore). È
comodo e va saputo, perché una risposta arriva comunque e sta a chi legge
sapere a quale delle tre situazioni corrisponde.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Un **sistema lineare** si legge in due modi. Per righe è una lista di
  promesse da mantenere tutte insieme; per colonne è la domanda «con quali
  dosi di questi barattoli ottengo quel colore?». La seconda lettura è quella
  che dice subito quando la richiesta è fuori portata.
- L’**eliminazione** è il metodo per risolverlo, e sta in tre mosse che non
  cambiano le risposte: scambiare due equazioni, moltiplicarne una per un
  numero diverso da zero, sommare a una il multiplo di un'altra. Si fanno
  sparire le incognite una alla volta finché non resta una scaletta.
- Gli esiti sono tre e solo tre: una soluzione, nessuna, oppure infinite.
  Quando le equazioni si ripetono travestite (i totali di riga e di colonna di
  una tabella sono un caso classico) restano dei gradi di libertà, e i dati da
  soli non bastano a scegliere.
- L’**immagine** è il repertorio delle ombre ottenibili, e dice se una
  richiesta si può soddisfare; il **nucleo** è la direzione lungo cui l'oggetto
  si sposta senza che l'ombra cambi, e dice che quell'informazione all'arrivo
  è perduta per sempre. Le direzioni di partenza si dividono fra le due, e la
  somma torna.
- Il **rango** conta quante righe (o colonne, che è lo stesso) portano
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
- L’**eliminazione di Gauss** applica operazioni di riga invertibili alla
  matrice aumentata e la porta in forma a scala ridotta; costa
  $\Theta(n^3)$, richiede pivoting parziale per stabilità, e va preferita al
  calcolo esplicito di $\mathbf{A}^{-1}$.
- **Rouché–Capelli**: il sistema è compatibile se e solo se
  $\operatorname{rank}(\mathbf{A}) =
  \operatorname{rank}([\,\mathbf{A}\mid\mathbf{b}\,])$, e in tal caso le
  soluzioni formano un affine di dimensione $n-r$.
- $\operatorname{im}(\mathbf{A})$ è lo span delle colonne,
  $\ker(\mathbf{A})$ ciò che va a zero, e l'insieme delle soluzioni è
  $\mathbf{x}_p + \ker(\mathbf{A})$. Vale
  $\dim\operatorname{im}+\dim\ker = n$: un nucleo non banale significa
  parametri **non identificabili**, cioè indistinguibili da qualunque dato.
- Il **rango** è il numero di pivot, la dimensione dell'immagine, il massimo
  numero di righe o colonne indipendenti e il numero di valori singolari non
  nulli. Da $\operatorname{rank}(\mathbf{A}\mathbf{B}) \le
  \min(\operatorname{rank}\mathbf{A}, \operatorname{rank}\mathbf{B})$ discende
  ogni fattorizzazione a collo di bottiglia. Su dati reali si guarda il
  decadimento dei valori singolari, non l'annullamento.
```
`````

Con i sistemi lineari l'algebra lineare smette di essere solo un modo di
impacchettare i dati e diventa un modo di interrogarli: che cosa questa
trasformazione può produrre, che cosa perde, quante informazioni davvero
distinte contiene. Restava sospesa una domanda, quella dei sistemi con troppe
equazioni, e la risposta cambia geometria: invece di cercare il punto esatto
si cerca il più vicino, che è il mestiere delle proiezioni.
