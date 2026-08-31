# Il determinante: quanto una trasformazione gonfia lo spazio

Il determinante è più vecchio della matrice. Nel 1683 il matematico giapponese
Seki Takakazu descrive un metodo per decidere se un sistema di equazioni ha
soluzione, e il metodo consiste nel calcolare un certo numero a partire dai
coefficienti; dieci anni dopo, in una lettera a l'Hôpital, Leibniz scrive
qualcosa di equivalente per la stessa ragione. Nessuno dei due ha una parola
per la tabella di numeri da cui quel valore si ricava: «matrice» la conierà
James Joseph Sylvester nel 1850, e la teoria arriverà con Arthur Cayley nel
1858.

Per un secolo e mezzo, quindi, si è calcolato il determinante di oggetti che
non avevano nome. La cronologia dice che cosa quel numero è venuto al mondo per
fare, cioè rispondere alla domanda che la {doc}`sezione sui sistemi lineari
</Matematica/sistemi-lineari>` ha lasciato aperta: una soluzione esiste, ed è
una sola? La risposta geometrica, che arriverà molto più tardi, è ancora più
netta: quel numero misura di quanto la trasformazione gonfia lo spazio, e se
vale zero lo ha schiacciato.

## Un numero che misura un'area

```{figure} ../figures/determinante-area.svg
:name: fig-determinante-area
:alt: "Tre riquadri affiancati. Nel primo un quadratino unitario su una griglia, con i due lati disegnati come frecce e l'area annotata uguale a uno. Nel secondo lo stesso quadratino è diventato un parallelogramma di area cinque, e il quadratino di partenza resta tratteggiato accanto per il confronto. Nel terzo le due frecce cadono sulla stessa retta e il quadratino si riduce a un segmento spesso: l'area è zero, e l'annotazione dice che da lì non si torna indietro."
:width: 92%

Il quadretto di partenza, lo stesso quadretto dopo la trasformazione, e il
caso in cui la trasformazione lo appiattisce. Il determinante è il rapporto fra
l'area finale e quella iniziale, e vale zero esattamente quando la figura
collassa.
```

`````{tab} Elementare

Prendi un foglio di gomma con sopra disegnato un quadretto di un centimetro di
lato, e tiralo. Il quadretto diventa un parallelogramma, e la domanda naturale
è di quante volte la sua area sia cambiata. Quel numero, per lo stiramento che
hai fatto, è il **determinante**, e {numref}`fig-determinante-area` lo disegna.

La cosa notevole è che quel numero non dipende dal quadretto che hai scelto.
Qualunque figura tu avessi disegnato sul foglio (un cerchio, una lettera, la
sagoma di un gatto) l'area sarebbe cambiata dello stesso fattore, perché lo
stiramento è lo stesso dappertutto. Un solo numero riassume l'effetto della
trasformazione su tutte le aree del piano.

Ci sono quattro cose da sapere, e sono tutte conseguenze del guardarlo come un
fattore di area.

*Può essere negativo.* Se oltre a tirare il foglio lo rovesci, l'area c'è
ancora ma il disegno è specchiato: quello che prima girava in senso orario ora
gira in senso antiorario. Il segno meno tiene il conto di questo ribaltamento,
e il valore assoluto resta il fattore di area.

*Se vale zero, hai schiacciato tutto.* Uno stiramento che appiattisce il foglio
su una retta manda ogni figura in un segmento: l'area diventa zero. Da un
segmento non si risale al quadretto di partenza, e in effetti quello è
esattamente il caso in cui la trasformazione non si può disfare. È la stessa
cosa che la sezione sui sistemi lineari chiamava avere un nucleo più grande del
solo zero, guardata da un'altra finestra.

*Due stiramenti di fila moltiplicano i fattori.* Se il primo raddoppia le aree
e il secondo le triplica, insieme le fanno sei volte. Ne segue subito che
tornare indietro divide: se una trasformazione moltiplica le aree per cinque,
la sua inversa le divide per cinque.

*In tre dimensioni si chiama volume, e non cambia nient'altro.* Il cubetto di
un centimetro di lato diventa un solido obliquo, e il determinante dice di
quante volte il volume è cambiato. Sopra le tre dimensioni non c'è più niente
da guardare, ma il conto continua a funzionare e continua a significare la
stessa cosa.

`````

`````{tab} Superiore

Per $\mathbf{A}\in\mathbb{R}^{2\times 2}$ il determinante è

$$
\det\mathbf{A} = \det\begin{pmatrix} a & b\\ c & d\end{pmatrix} = ad - bc ,
$$

ed è l’**area con segno** del parallelogramma generato dalle due colonne. In
$\mathbb{R}^n$ è il volume con segno del parallelepipedo generato dalle $n$
colonne, e la definizione che rende tutto immediato è quella assiomatica: il
determinante è l'unica funzione $\mathbb{R}^{n\times n}\to\mathbb{R}$ che sia
**multilineare** nelle colonne, **alternante** (cambia segno scambiando due
colonne) e **normalizzata** ($\det\mathbf{I}=1$).

Da questi tre assiomi discende tutto il resto, e in particolare le proprietà
che si usano davvero:

$$
\det(\mathbf{A}\mathbf{B}) = \det\mathbf{A}\,\det\mathbf{B},
\qquad
\det(\mathbf{A}^{-1}) = \frac{1}{\det\mathbf{A}},
\qquad
\det(\mathbf{A}^\top) = \det\mathbf{A},
$$

e per uno scalare $\alpha$, in dimensione $n$,
$\det(\alpha\mathbf{A}) = \alpha^n\det\mathbf{A}$, che è la prima cosa che si
sbaglia scrivendo a memoria (il fattore va all’$n$-esima potenza perché scala
tutte e $n$ le direzioni).

La caratterizzazione che lega il determinante al nucleo è

$$
\det\mathbf{A} \neq 0
\;\Longleftrightarrow\;
\mathbf{A} \text{ invertibile}
\;\Longleftrightarrow\;
\ker(\mathbf{A}) = \{\mathbf{0}\}
\;\Longleftrightarrow\;
\operatorname{rank}(\mathbf{A}) = n ,
$$

quattro modi di dire la stessa cosa. Vale infine il legame con lo spettro:
$\det\mathbf{A} = \prod_{i=1}^{n}\lambda_i$, il prodotto degli autovalori
contati con la loro molteplicità, e $|\det\mathbf{A}| = \prod_i \sigma_i$, il
prodotto dei valori singolari. La seconda scrittura rende evidente il senso
geometrico: la decomposizione ai valori singolari dice che ogni trasformazione
è una rotazione, poi una dilatazione lungo assi ortogonali di fattori
$\sigma_i$, poi un'altra rotazione; le rotazioni non toccano i volumi, quindi
il fattore di volume è il prodotto delle dilatazioni.

Una conseguenza da tenere a mente prima di usare il determinante come diagnosi:
$|\det|$ non dice niente sul condizionamento. Una matrice
$\operatorname{diag}(10^{-3},10^{3})$ ha determinante $1$, quindi conserva le
aree alla perfezione, ed è pessimamente condizionata perché schiaccia
mille volte in una direzione e stira mille volte nell'altra. Il determinante
è un prodotto, e in un prodotto uno zero quasi e un infinito quasi si
cancellano.

`````

Il fatto che il determinante si annulli esattamente quando la trasformazione
perde delle direzioni lo rende la risposta più compatta alla domanda dei
sistemi lineari. Vale però soltanto per le matrici quadrate: un sistema con più
equazioni che incognite non ha un determinante da guardare, e lì la domanda si
sposta sul rango.

```python
import numpy as np

A = np.array([[3.0, 1.0],
              [1.0, 2.0]])
print(np.linalg.det(A))              # 3*2 - 1*1 -> 5.000000000000001

specchio = np.array([[0.0, 1.0],     # scambia le due coordinate
                     [1.0, 0.0]])
print(np.linalg.det(specchio))       # ribalta senza deformare -> -1.0

# due trasformazioni di fila: i fattori si moltiplicano
print(round(np.linalg.det(A @ specchio), 6),
      round(np.linalg.det(A) * np.linalg.det(specchio), 6))   # -> -5.0 -5.0

# seconda colonna = doppio della prima: il foglio viene schiacciato
S = np.array([[2.0, 4.0],
              [1.0, 2.0]])
print(np.linalg.det(S), np.linalg.matrix_rank(S))             # -> 0.0 1
```

## Come si calcola, e come non si calcola

La regola che si impara a scuola per le matrici piccole si estende a qualunque
dimensione, ed è la strada sbagliata.

`````{tab} Elementare

Lo sviluppo che si insegna a mano funziona benissimo per due o tre righe e
diventa assurdo appena dopo. Per una tabella di venti righe per venti
chiederebbe due miliardi di miliardi di moltiplicazioni: una macchina che ne
fa un miliardo al secondo ci metterebbe settantasette anni. La strada buona è
la stessa dei sistemi lineari, l'eliminazione: si porta la tabella a scaletta,
e a quel punto il determinante è il prodotto dei numeri sulla diagonale, con un
cambio di segno per ogni scambio di righe fatto per strada. Ottomila
operazioni invece di due miliardi di miliardi, e il conto finisce prima che tu
abbia alzato gli occhi.

Resta un guaio, e riguarda la grandezza del risultato. Moltiplicare fra loro
qualche centinaio di numeri dà un valore assurdamente grande o assurdamente
piccolo: su una tabella di quattrocento righe presa a caso il determinante
esce dalla scala dei numeri che la macchina sa scrivere, e al posto della
risposta compare la parola «infinito». La cura è antica e si usa dappertutto:
invece del prodotto si sommano i logaritmi. Il logaritmo di un numero enorme è
un numero comodo, e si somma senza problemi. Le librerie offrono la funzione
apposita, che restituisce due cose separate, il segno e il logaritmo del valore
assoluto.

`````

`````{tab} Superiore

Lo **sviluppo di Laplace** lungo una riga o una colonna,

$$
\det\mathbf{A} = \sum_{j=1}^{n} (-1)^{i+j} A_{ij}\,\det \mathbf{M}_{ij},
$$

dove $\mathbf{M}_{ij}$ è il minore ottenuto cancellando riga $i$ e colonna $j$,
ha costo $\Theta(n!)$ e serve solo a dimostrare teoremi. La via praticabile è
la fattorizzazione $\mathbf{P}\mathbf{A}=\mathbf{L}\mathbf{U}$ prodotta
dall'eliminazione con pivoting: poiché $\det\mathbf{L}=1$ e il determinante di
una matrice triangolare è il prodotto della diagonale,

$$
\det\mathbf{A} = (-1)^{s}\prod_{i=1}^{n} U_{ii},
$$

con $s$ il numero di scambi di riga. Il costo scende a $\Theta(n^3)$, lo stesso
dell'eliminazione, e in effetti il determinante arriva gratis come sottoprodotto
della risoluzione di un sistema.

Il prodotto di $n$ pivot ha però una dinamica esponenziale: per
$\mathbf{M}\in\mathbb{R}^{400\times 400}$ con entrate normali standard il
valore vero è dell'ordine di $e^{998}$, cioè fuori dai numeri in virgola mobile
a doppia precisione, che si fermano attorno a $1{,}8\cdot 10^{308}$. Per questo
esiste `numpy.linalg.slogdet`, che restituisce la coppia
$(\operatorname{sign}, \log|\det|)$ sommando i logaritmi dei pivot invece di
moltiplicarli. Nelle applicazioni la quantità che serve è quasi sempre
$\log|\det|$ e non $\det$, quindi non si perde niente: è il termine che compare
nelle log-verosimiglianze, e chiamarlo direttamente evita di calcolare un
numero enorme per poi prenderne il logaritmo.

`````

```python
rng = np.random.default_rng(0)
M = rng.normal(size=(400, 400))

print(np.linalg.det(M))          # -> -inf, con un avviso di overflow

segno, log_det = np.linalg.slogdet(M)
print(segno, round(log_det, 4))  # -> -1.0 998.4819

# su una triangolare il conto e' il prodotto della diagonale, e si vede
T = np.tril(rng.normal(size=(5, 5)))
np.fill_diagonal(T, [1.2, 0.7, 2.0, 0.5, 1.5])
print(round(np.linalg.det(T), 8), round(np.prod(np.diag(T)), 8))   # -> 1.26 1.26
```

Il determinante che va in `inf` è il motivo per cui, nel codice vero, si
incontra quasi sempre sotto forma di logaritmo. Il conto sulla triangolare è
invece la mossa su cui è costruita un'intera famiglia di modelli generativi, e
va guardata da vicino.

## Quando lo spazio si deforma, quello che è spalmato sopra si assottiglia

Fin qui il determinante ha misurato aree e volumi. La ragione per cui torna di
continuo nel machine learning è un'altra, e discende da questa.

`````{tab} Elementare

Hai un litro di vernice steso su una parete di dieci metri quadri, quindi un
decimo di litro per metro quadro. Se la parete si allarga fino a venti metri
quadri e la vernice resta quella, lo strato si dimezza: la quantità totale non
cambia, quindi quanta ce n'è per metro quadro deve scendere esattamente di
quanto la parete si è allargata.

Questa è tutta l'idea. Ogni volta che una trasformazione deforma lo spazio,
qualunque cosa sia distribuita su quello spazio si concentra dove lo spazio si
è ristretto e si dirada dove si è allargato, e il fattore è precisamente il
determinante. Dove la trasformazione non è la stessa dappertutto (un foglio
tirato più da una parte che dall'altra) si guarda punto per punto: attorno a
ogni punto, per uno spostamento piccolo, ogni deformazione somiglia a uno
stiramento uniforme, e il determinante di quello stiramento locale dice quanto
la vernice si assottiglia lì.

Questo conto è quello che permette a certi generatori di immagini di dire
**quanto è probabile** ciò che hanno prodotto, invece di limitarsi a produrlo.
Funzionano così: partono da una nuvola di punti semplicissima e la deformano,
con una trasformazione che si può disfare, finché non somiglia ai dati veri.
Per sapere quanto è denso il risultato in un punto bisogna sapere quanto la
deformazione ha allargato lo spazio proprio lì, e cioè fare il determinante,
per ogni punto e a ogni passo.

Ed ecco il problema pratico, con la sua soluzione, che è la parte istruttiva.
Quel conto costa il cubo del numero di coordinate, e su un'immagine di
centomila pixel sarebbe fuori discussione. Allora si costruisce la
trasformazione apposta perché il conto venga facile: si impone che la prima
coordinata nuova dipenda solo dalla prima vecchia, la seconda dalle prime due,
la terza dalle prime tre. Con questo vincolo la tabella viene a scaletta, e il
determinante torna a essere il prodotto della sua diagonale: un fattore per
coordinata, quindi un costo proporzionale al numero di coordinate invece che al
suo cubo. È un buon esempio di come una
proprietà matematica finisca per decidere la forma di un'architettura: la
trasformazione è stata piegata al conto, e non il contrario.

`````

`````{tab} Superiore

Sia $\mathbf{f}:\mathbb{R}^n\to\mathbb{R}^n$ una trasformazione invertibile e
differenziabile, e $\mathbf{y}=\mathbf{f}(\mathbf{x})$. La **formula del cambio
di variabile** per una densità è

$$
p_Y(\mathbf{y}) = p_X(\mathbf{x})\,
\left|\det \mathbf{J}_{\mathbf{f}}(\mathbf{x})\right|^{-1},
\qquad
\big(\mathbf{J}_{\mathbf{f}}\big)_{ij}
= \frac{\partial f_i}{\partial x_j},
$$

dove $\mathbf{J}_{\mathbf{f}}$ è la matrice **jacobiana**, cioè la migliore
approssimazione lineare di $\mathbf{f}$ attorno a $\mathbf{x}$, e le derivate
parziali che la compongono arrivano nella {doc}`sezione su analisi e
ottimizzazione </Matematica/analisi-ottimizzazione>`. La lettura è quella della
vernice: la massa totale si conserva, quindi la densità va divisa per il
fattore di volume locale.

In forma logaritmica, che è quella che si usa,

$$
\log p_Y(\mathbf{y}) = \log p_X(\mathbf{x})
- \log\left|\det \mathbf{J}_{\mathbf{f}}(\mathbf{x})\right| ,
$$

e componendo $K$ trasformazioni i termini si sommano, perché il determinante di
un prodotto è il prodotto dei determinanti. È l'ossatura dei **flussi
normalizzanti**, che il {doc}`capitolo sulla verosimiglianza esatta
</VerosimiglianzaEsatta/overview>` sviluppa per intero.

Il vincolo di progetto discende dal costo. Un determinante generico costa
$\Theta(n^3)$ per passo, insostenibile per $n$ dell'ordine di $10^5$. Le
architetture si costruiscono quindi perché la jacobiana sia **triangolare**:
imponendo che $f_i$ dipenda solo da $x_1,\dots,x_i$, tutti gli elementi sopra
la diagonale sono nulli e

$$
\log\left|\det \mathbf{J}\right|
= \sum_{i=1}^{n} \log\left|\frac{\partial f_i}{\partial x_i}\right| ,
$$

cioè $\Theta(n)$. Lo stesso determinante compare in altri due posti che
conviene riconoscere come lo stesso conto: nella densità della gaussiana
multivariata, dove il fattore di normalizzazione contiene
$(\det\boldsymbol{\Sigma})^{-1/2}$ e misura il volume dell'ellissoide di
covarianza, e nella verosimiglianza dei processi gaussiani, dove il termine
$\log\det$ della matrice di covarianza è precisamente il pezzo che rende il
metodo cubico nel numero di osservazioni.

`````

Il caso opposto chiude il discorso. Una rete
neurale ordinaria non è invertibile: la funzione di attivazione più diffusa
manda a zero tutti i valori negativi, e da uno zero non si risale al numero di
partenza. Il determinante della sua jacobiana è nullo su intere regioni, e
quindi la formula del cambio di variabile non si applica. È il prezzo che
separa i modelli capaci di dire quanto è probabile ciò che generano da quelli
che sanno soltanto generare, ed è una scelta di progetto, non una svista.

## In pratica, con NumPy

```python
import numpy as np

A = np.array([[3.0, 1.0], [1.0, 2.0]])

np.linalg.det(A)          # il fattore di area, con segno
np.linalg.slogdet(A)      # (segno, log|det|): la forma da usare quasi sempre
np.linalg.matrix_rank(A)  # la domanda giusta se la matrice non e' quadrata

np.linalg.det(2 * A)      # -> 4 volte det(A): il fattore va alla potenza n
```

Il determinante della matrice raddoppiata merita un secondo di attenzione,
perché è la svista più comune: raddoppiare una matrice $2\times 2$ non raddoppia il determinante, lo
quadruplica, e su una matrice $n\times n$ lo moltiplica per $2^n$. Il motivo è
geometrico: raddoppiando la matrice si raddoppia ogni lato del
parallelepipedo, e i lati sono $n$.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Il **determinante** è di quante volte una trasformazione cambia le aree (o i
  volumi), ed è lo stesso numero per qualunque figura, perché lo stiramento è
  lo stesso dappertutto.
- Il **segno** dice se la trasformazione ha anche ribaltato il foglio. Il
  valore **zero** dice che ha schiacciato tutto su una retta: da lì non si
  torna indietro, ed è lo stesso caso in cui un sistema non ha una sola
  soluzione.
- Due trasformazioni di fila **moltiplicano** i loro fattori, e disfarne una
  divide per il suo.
- La regola che si impara a scuola su venti righe chiederebbe settantasette
  anni di conti: si calcola con l'eliminazione, portando la tabella a scaletta
  e moltiplicando la diagonale. E poiché quel prodotto diventa subito enorme,
  nel codice si usa il **logaritmo** del determinante invece del determinante.
- Quello che è distribuito nello spazio si **assottiglia** dove lo spazio si
  allarga, esattamente del fattore dato dal determinante. È il conto che
  permette a certi generatori di dire quanto è probabile ciò che producono, e
  li obbliga a una forma in cui quel conto costa poco.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- $\det\mathbf{A}$ è il volume con segno del parallelepipedo generato dalle
  colonne; è l'unica funzione multilineare, alternante e con
  $\det\mathbf{I}=1$. Valgono
  $\det(\mathbf{A}\mathbf{B})=\det\mathbf{A}\det\mathbf{B}$,
  $\det(\mathbf{A}^{-1})=1/\det\mathbf{A}$ e
  $\det(\alpha\mathbf{A})=\alpha^n\det\mathbf{A}$.
- $\det\mathbf{A}\neq 0$ equivale a invertibilità, a nucleo banale e a rango
  pieno. Inoltre $\det\mathbf{A}=\prod_i\lambda_i$ e
  $|\det\mathbf{A}|=\prod_i\sigma_i$. Un determinante grande non dice niente
  sul condizionamento: $\operatorname{diag}(10^{-3},10^{3})$ ha determinante $1$.
- Laplace costa $\Theta(n!)$ e serve per le dimostrazioni; si calcola con
  $\mathbf{P}\mathbf{A}=\mathbf{L}\mathbf{U}$ in $\Theta(n^3)$, e nel codice si
  usa `slogdet` perché il prodotto dei pivot va in overflow già a poche
  centinaia di righe.
- Cambio di variabile: $\log p_Y(\mathbf{y}) = \log p_X(\mathbf{x}) -
  \log|\det\mathbf{J}_{\mathbf{f}}(\mathbf{x})|$, e componendo trasformazioni i
  termini si sommano. È l'ossatura dei flussi normalizzanti, e la ragione per
  cui le loro jacobiane si costruiscono **triangolari**, portando il costo da
  $\Theta(n^3)$ a $\Theta(n)$.
- Lo stesso $\log\det$ compare nella normalizzazione della gaussiana
  multivariata e nella verosimiglianza dei processi gaussiani, dove è il
  termine che rende il metodo cubico nel numero di osservazioni.
```
`````

Con il determinante l'algebra lineare del libro è completa: si sa mettere i
dati in fila, trasformarli, chiedersi che cosa una trasformazione può produrre
e che cosa perde, trovare la risposta migliore quando quella esatta non esiste,
e misurare di quanto lo spazio si è deformato. Manca l'altra metà degli
attrezzi, quella che non guarda una trasformazione fissa ma il modo in cui una
quantità cambia quando se ne muove un'altra: sono le derivate, ed è con esse
che si impara a migliorare.
