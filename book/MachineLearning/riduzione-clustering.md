# Ridurre le dimensioni e trovare gruppi: l'apprendimento non supervisionato

Fino a qui abbiamo sempre avuto un maestro alle spalle. La regressione, la
classificazione, gli alberi e gli ensemble: ogni esempio arrivava con la sua
risposta giusta accanto (il prezzo della casa, l'etichetta «spam» o «non
spam»). Era l'apprendimento *supervisionato*, il primo dei «tre modi di
imparare» che abbiamo distinto all'inizio del capitolo. Ma se ci pensi, quelle
etichette sono un lusso: qualcuno le ha dovute scrivere, una per una. La
stragrande maggioranza dei dati che il mondo produce (foto, transazioni,
segnali di sensori, log di navigazione) arriva **muta**, senza nessuna
risposta allegata.

Questo è il territorio dell'**apprendimento non supervisionato**: si danno al
modello solo gli input, e gli si chiede di scoprire da sé una struttura
nascosta. Due domande, soprattutto, si possono porre a dati senza etichette.
La prima: *questi dati hanno davvero bisogno di tutte queste dimensioni, o si
possono comprimere senza perdere l'essenziale?* È la **riduzione della
dimensionalità**. La seconda: *ci sono gruppi naturali là dentro, famiglie di
esempi che si somigliano tra loro?* È il **clustering**. Sono i due pilastri
di questa sezione.

## Quando avere troppe dimensioni è un problema

Verrebbe da pensare che più informazioni abbiamo su ogni esempio (più colonne,
più misure, più *feature*) meglio è. Sorprendentemente, oltre una certa soglia
è vero il contrario. Lo spazio ad alta dimensione si comporta in modi che la
nostra intuizione, allenata a due o tre dimensioni, non prevede. Il fenomeno
ha un nome quasi teatrale: la **maledizione della dimensionalità**.

`````{tab} Elementare

Immagina di cercare un amico. In una **strada** (una dimensione) è facile:
sarà a pochi metri da te. In una **piazza** (due dimensioni) devi guardarti
attorno un po' di più. In un **grattacielo** (tre dimensioni) devi anche
scegliere il piano. Aggiungi una dimensione e lo spazio da esplorare si gonfia
ogni volta: in dieci, cento, mille dimensioni, tutto finisce per essere
lontanissimo da tutto il resto; non ci sono più «vicini», perché lo spazio è
troppo vuoto.

C'è un secondo effetto, ancora più controintuitivo: in tante dimensioni, quasi
tutto lo spazio si accalca **sui bordi**. Prendi una scatola e considera il
guscio sottile vicino alla superficie, spesso un decimo del lato. In una
dimensione (un segmento) quel guscio è il 20% del totale. In dieci dimensioni
è già l'89%. In cento, praticamente il 100%: quasi nessun punto sta «nel
mezzo», stanno tutti appiccicati alle pareti. In un mondo così svuotato e
spinto ai margini, gli algoritmi che si fidano delle distanze («chi è vicino a
chi») vanno in crisi. Da qui l'idea di **ridurre le dimensioni**: buttare via
quelle che non servono e tenere solo ciò che conta davvero.

`````

`````{tab} Superiore

Consideriamo l'ipercubo unitario $[0,1]^d$ e il guscio dei punti che distano
meno di $\varepsilon = 0{,}1$ da almeno una faccia. Il «cuore» interno è un
cubo di lato $1 - 2\varepsilon = 0{,}8$, di volume $0{,}8^{\,d}$; la frazione
di volume nel guscio è quindi

$$
1 - (1 - 2\varepsilon)^{d} = 1 - 0{,}8^{\,d}.
$$

Per $d=1$ vale $0{,}2$; per $d=10$ vale $1 - 0{,}8^{10} \approx 0{,}89$; per
$d=100$ vale $1 - 0{,}8^{100} \approx 1 - 2\cdot 10^{-10}$, ossia
praticamente $1$. All'aumentare di $d$ il volume fugge verso la superficie.

Parallelamente, le distanze si **concentrano**: per molte distribuzioni si
dimostra che il rapporto tra la distanza massima e minima tra $n$ punti
casuali tende a $1$ al crescere di $d$, cioè

$$
\frac{\operatorname{dist}_{\max} - \operatorname{dist}_{\min}}{\operatorname{dist}_{\min}} \xrightarrow[d\to\infty]{} 0 .
$$

Il punto più vicino e il più lontano finiscono per essere quasi equidistanti:
la nozione stessa di «vicinanza» perde di significato, e con essa vacillano
$k$-NN, il clustering per distanza e la stima di densità {cite}`geron2019hands`.
La risposta è cercare un sottospazio di dimensione $k \ll d$ che conservi
l'informazione utile: è la **riduzione della dimensionalità**.

`````

## PCA: le direzioni in cui i dati si muovono di più

Il metodo più antico e più usato per ridurre le dimensioni è l'**analisi delle
componenti principali** (*Principal Component Analysis*, PCA), le cui radici
risalgono a Karl Pearson {cite}`pearson1901lines` nel 1901 e a Harold Hotelling
{cite}`hotelling1933analysis` nel 1933. L'idea è di una semplicità elegante:
tra tutte le direzioni possibili nello spazio dei dati, alcune sono quelle
lungo cui i punti si sparpagliano molto, altre quelle lungo cui restano quasi
fermi. Le prime portano informazione, le seconde quasi nessuna. La PCA le
trova e tiene solo le prime.

`````{tab} Elementare

Pensa a uno stormo di uccelli fotografato da lontano. Se lo stormo è disteso
in lunghezza, la fotografia più informativa la scatti di lato, cogliendo la
direzione in cui gli uccelli sono più sparpagliati: da quella prospettiva
distingui bene chi è avanti e chi è indietro. Fotografarlo di punta, invece, li
schiaccerebbe tutti in un mucchietto indistinto.

La PCA fa esattamente questo con i dati. Cerca la direzione lungo cui i punti
sono **più dispersi** (la chiama *prima componente principale*), perché è lì
che si nasconde la maggior parte delle differenze tra un esempio e l'altro.
Poi cerca la direzione più dispersa tra quelle rimaste (perpendicolare alla
prima), e così via. Se le prime due o tre direzioni catturano quasi tutta la
dispersione, possiamo buttare le altre e rappresentare ogni dato con due o tre
numeri soltanto, quasi senza perdite. La «dispersione» ha un nome tecnico che
già conosciamo: **varianza**.

`````

`````{tab} Superiore

Sia $X \in \mathbb{R}^{m \times d}$ la matrice dei dati, con le feature già
**centrate** (media di colonna nulla) e, di norma, **standardizzate** (varianza
unitaria, così che una feature misurata in metri non domini una misurata in
chilometri). La dispersione dei dati è riassunta dalla **matrice di
covarianza**

$$
C = \frac{1}{m}\, X^{\top} X \in \mathbb{R}^{d \times d},
$$

dove $C_{jk}$ è la covarianza tra la feature $j$ e la feature $k$. La PCA
cerca il versore $u$ che massimizza la varianza dei dati proiettati,
$\operatorname{Var}(Xu) = u^{\top} C\, u$, con il vincolo $\lVert u \rVert = 1$.
Con i moltiplicatori di Lagrange il problema diventa

$$
C\, u = \lambda\, u ,
$$

cioè le direzioni cercate sono gli **autovettori** di $C$, e la varianza
catturata da ciascuna è il corrispondente **autovalore** $\lambda$. Ordinando
gli autovalori $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_d \ge 0$, la
prima componente principale è l'autovettore di $\lambda_1$, la seconda quello
di $\lambda_2$, e così via: sono ortogonali tra loro perché $C$ è simmetrica.
Proiettare su $u_1, \dots, u_k$ (con $k \ll d$) dà la rappresentazione ridotta
$Z = X\,U_k$, dove $U_k$ raccoglie i primi $k$ autovettori in colonna.

`````

Vale la pena vedere i conti su un esempio minuscolo, in due dimensioni, per
poi proiettare su una sola.

Prendiamo quattro punti già centrati (media nulla), così da saltare la
sottrazione della media:

$$
X = \{(2,2),\ (1,-1),\ (-2,-2),\ (-1,1)\}.
$$

La matrice di covarianza, dividendo per $m = 4$, ha elementi

$$
C_{xx} = \tfrac{2^2 + 1^2 + 2^2 + 1^2}{4} = \tfrac{10}{4} = 2{,}5,
\qquad
C_{yy} = \tfrac{2^2 + 1^2 + 2^2 + 1^2}{4} = 2{,}5,
$$
$$
C_{xy} = \tfrac{(2)(2) + (1)(-1) + (-2)(-2) + (-1)(1)}{4}
       = \tfrac{4 - 1 + 4 - 1}{4} = \tfrac{6}{4} = 1{,}5,
$$

quindi

$$
C = \begin{pmatrix} 2{,}5 & 1{,}5 \\ 1{,}5 & 2{,}5 \end{pmatrix}.
$$

Gli autovalori risolvono $\det(C - \lambda I) = 0$, cioè
$(2{,}5-\lambda)^2 - 1{,}5^2 = 0$, da cui $2{,}5 - \lambda = \pm 1{,}5$ e

$$
\lambda_1 = 4, \qquad \lambda_2 = 1 .
$$

Gli autovettori corrispondenti sono $u_1 = \tfrac{1}{\sqrt2}(1, 1)$ (la
diagonale che sale) e $u_2 = \tfrac{1}{\sqrt2}(1, -1)$. La **varianza
spiegata** dalla prima componente è

$$
\frac{\lambda_1}{\lambda_1 + \lambda_2} = \frac{4}{4 + 1} = 0{,}8,
$$

l'80%: proiettando su $u_1$ soltanto, buttiamo via una dimensione ma
conserviamo i quattro quinti della dispersione. La proiezione di ogni punto è
il prodotto scalare con $u_1$:

$$
(2,2)\!\cdot\! u_1 = \tfrac{4}{\sqrt2} = 2\sqrt2,\quad
(1,-1)\!\cdot\! u_1 = 0,\quad
(-2,-2)\!\cdot\! u_1 = -2\sqrt2,\quad
(-1,1)\!\cdot\! u_1 = 0.
$$

I due punti «fuori diagonale» collassano a $0$ (stavano interamente lungo la
direzione scartata $u_2$), mentre i due «in diagonale» conservano tutta la
loro distanza. La {numref}`fig-pca-proiezione` mostra la stessa idea su una
nuvola più fitta: l'asse lungo è $u_1$, quello corto $u_2$, e proiettare
significa lasciar cadere ogni punto perpendicolarmente sull'asse lungo.

```{figure} ../figures/pca-proiezione.svg
:name: fig-pca-proiezione
:alt: Una nuvola di punti in teal allungata in diagonale; sovrapposti l'asse PC1 (terracotta, lungo) nella direzione di massima varianza e l'asse PC2 (ocra, corto) ortogonale. Tre punti sono collegati da tratteggi alla loro proiezione perpendicolare su PC1.
:width: 80%

La prima componente principale (PC1) segue la direzione di massima varianza;
la seconda (PC2), ortogonale, ne raccoglie molta meno. Proiettare i punti su
PC1 comprime i dati da due dimensioni a una, perdendo poco.
```

A cosa serve, in concreto? A **comprimere** i dati (meno numeri da salvare e
da dare in pasto ai modelli); a **visualizzare** in due o tre dimensioni
dataset con centinaia di feature; a **denoising**, perché il rumore casuale
tende a finire nelle ultime componenti a bassa varianza, che scartando
ripuliamo il segnale. Il limite, cruciale, è che la PCA è **lineare**: sa solo
ruotare e proiettare lungo assi dritti. Se la struttura interessante dei dati
è curva (un rotolo arrotolato su sé stesso, due spirali intrecciate), la PCA
la appiattisce e la rovina. Per quei casi servono metodi non lineari.

## t-SNE e UMAP: vedere in due dimensioni ciò che vive in mille

Quando l'obiettivo è soltanto **guardare** dati ad alta dimensione (non
comprimerli per un modello a valle, ma disegnarli su uno schermo per capirli
con gli occhi), la PCA spesso non basta. Sono nate tecniche pensate apposta
per la visualizzazione, che rinunciano alla linearità pur di rendere leggibile
una mappa in 2D: le più note sono **t-SNE** (van der Maaten e Hinton, 2008)
{cite}`maaten2008visualizing` e **UMAP** (McInnes, Healy e Melville, 2018)
{cite}`mcinnes2018umap`.

`````{tab} Elementare

Immagina di dover disegnare su un foglio la mappa delle amicizie di una
scuola. Non ti interessa la distanza reale tra le case: ti interessa che chi è
amico finisca **vicino** sul foglio, e chi non si conosce finisca lontano.
t-SNE e UMAP fanno questo con i dati: prendono punti che vivono in uno spazio
con tante dimensioni e li dispongono in due, cercando di mettere accanto i
punti che erano vicini in origine. Il risultato sono mappe bellissime, dove
categorie diverse (cifre scritte a mano, tipi di cellule, generi musicali) si
separano in isole ben distinte.

Ma qui serve un'avvertenza onesta, perché è la causa di molti errori.
Su queste mappe **non fidarti delle distanze grandi**: due isole lontane sul
foglio non sono necessariamente più diverse di due isole vicine, l'algoritmo
non lo garantisce. E non fidarti della **grandezza** delle isole né di
**quanto sono fitte**: un gruppo disegnato grande e sparso può in realtà
essere compatto quanto uno disegnato piccolo. Sono strumenti per *vedere* se
esistono dei gruppi, non per *misurare* quanto distano o quanto sono grandi.
Ottimi per l'occhio, pessimi per il righello.

`````

`````{tab} Superiore

t-SNE modella le vicinanze come probabilità. Nello spazio originale la
somiglianza tra i punti $i$ e $j$ è una gaussiana sulla loro distanza,
$p_{j\mid i} \propto \exp(-\lVert x_i - x_j\rVert^2 / 2\sigma_i^2)$, con
$\sigma_i$ tarato localmente da un iperparametro, la *perplexity*. Nello
spazio ridotto usa invece una $t$ di Student a un grado di libertà (con code
pesanti, che evitano l'affollamento al centro),
$q_{ij} \propto (1 + \lVert y_i - y_j\rVert^2)^{-1}$, e dispone i punti $y_i$
minimizzando la divergenza di Kullback–Leibler $\mathrm{KL}(P \Vert Q)$ tra le
due distribuzioni. Poiché la KL pesa molto le vicinanze e poco le lontananze,
t-SNE **preserva la struttura locale** ma distorce quella globale: distanze
tra cluster, densità e dimensioni apparenti sui grafici **non sono
quantitativamente affidabili**.

UMAP parte da fondamenta diverse (una costruzione su grafi e topologia) ma
persegue un obiettivo simile; in pratica è **più veloce**, scala meglio a
milioni di punti e tende a **preservare meglio la struttura globale**, pur
condividendo lo stesso monito: sono strumenti di visualizzazione, non di
analisi metrica. Entrambi vanno usati per *esplorare*, mai per concludere che
«questo gruppo è il doppio più lontano di quell'altro».

`````

```{admonition} Quando usarli, e quando no
:class: tip
**Sì**: guardare se un dataset ha una struttura a gruppi prima di modellare;
ispezionare gli *embedding* di una rete per capire cosa ha imparato; presentare
a un pubblico la forma di dati che nessuno può visualizzare in $784$ dimensioni.

**No**: come passo di preprocessing prima di un classificatore, per quello
serve la PCA, che è lineare, invertibile e si applica a dati nuovi. Né t-SNE
né UMAP producono una trasformazione riusabile in modo affidabile su punti mai
visti, e la mappa cambia a ogni esecuzione con seed diversi.

**Mai**: fare clustering *sulle coordinate 2D* prodotte da t-SNE. I gruppi che
vedi possono essere artefatti della proiezione, e la loro separazione apparente
non corrisponde a una separazione reale. Il clustering si fa nello spazio
originale; la mappa serve solo a guardarne il risultato.
```

## Clustering con k-means: assegna, ricalcola, ripeti

Cambiamo domanda. Non più «come comprimo i dati» ma «ci sono gruppi naturali
là dentro». Il **clustering** cerca di partizionare gli esempi in famiglie
omogenee (clienti simili, documenti sullo stesso tema, pixel dello stesso
oggetto) senza che nessuno abbia mai detto quali famiglie esistano. È il
gemello non supervisionato della classificazione: stessi «insiemi di
etichette» in uscita, ma le etichette qui **non le conosciamo**, le inventa
l'algoritmo. Il metodo più celebre è **k-means**, il cui algoritmo iterativo è
dovuto a Stuart Lloyd (formulato ai Bell Labs nel 1957, pubblicato nel 1982)
{cite}`lloyd1982least`; il nome «$k$-means» compare in James MacQueen nel 1967
{cite}`macqueen1967some`.

`````{tab} Elementare

Devi sistemare un mucchio di persone sparse in un parco attorno a due punti di
ritrovo, in modo che ciascuno vada al ritrovo più vicino. Ma non sai ancora
*dove* mettere i due punti di ritrovo. k-means risolve il dilemma con un
tira-e-molla, ripetuto finché tutto si stabilizza:

1. **Piazza** i due punti di ritrovo a caso.
2. **Assegna** ogni persona al ritrovo più vicino: si formano due gruppi.
3. **Sposta** ogni ritrovo esattamente al centro del suo gruppo (la media
   delle posizioni).
4. Torna al passo 2. Con i ritrovi spostati, qualcuno cambierà gruppo; si
   ricalcolano i centri; e si continua.

Passo dopo passo i centri smettono di muoversi e nessuno cambia più gruppo:
l'algoritmo si è fermato. Il numero di ritrovi (qui due) è il famoso $k$, che
devi decidere tu in anticipo.

`````

`````{tab} Superiore

Dato un numero $k$ di cluster, k-means cerca i centroidi
$\mu_1, \dots, \mu_k$ e l'assegnazione dei punti che minimizzano l'**inerzia**
(somma delle distanze quadrate dai rispettivi centroidi):

$$
\mathcal{L}(\mu, c) = \sum_{i=1}^{m} \bigl\lVert x_i - \mu_{c_i} \bigr\rVert^2,
$$

dove $c_i \in \{1, \dots, k\}$ è il cluster assegnato al punto $x_i$. L'algoritmo
di Lloyd minimizza $\mathcal{L}$ alternando due passi di coordinate:

$$
\textbf{assegnazione:}\quad
c_i = \arg\min_{j} \lVert x_i - \mu_j \rVert^2,
\qquad
\textbf{aggiornamento:}\quad
\mu_j = \frac{1}{|C_j|} \sum_{i \in C_j} x_i .
$$

Ciascun passo non aumenta mai $\mathcal{L}$, quindi la procedura converge, ma
solo a un **minimo locale**, che dipende dall'inizializzazione. Il costo è
$O(m\,k\,d)$ per iterazione.

`````

Seguiamo una manciata di iterazioni a mano. Sei punti in due dimensioni,
$k = 2$:

$$
A(1,1),\ B(1,2),\ C(2,1),\ D(8,8),\ E(9,8),\ F(8,9).
$$

Partiamo (di proposito male) con i centroidi $\mu_1 = (1,1)$ e
$\mu_2 = (2,1)$, entrambi in mezzo al gruppo di sinistra.

**Iterazione 1: assegnazione.** Ogni punto va al centroide più vicino. $A$ e
$B$ finiscono in $\mu_1$; $C$ (che coincide con $\mu_2$), $D$, $E$, $F$
finiscono in $\mu_2$, le distanze di $D, E, F$ da $\mu_2 = (2,1)$ (circa
$9{,}2$; $9{,}9$; $10{,}0$) sono appena minori di quelle da $\mu_1 = (1,1)$
(circa $9{,}9$; $10{,}6$; $10{,}6$). Cluster: $C_1 = \{A, B\}$,
$C_2 = \{C, D, E, F\}$.

**Iterazione 1, aggiornamento.** Ricalcoliamo i centri come media:

$$
\mu_1 = \Bigl(\tfrac{1+1}{2}, \tfrac{1+2}{2}\Bigr) = (1;\ 1{,}5),
\qquad
\mu_2 = \Bigl(\tfrac{2+8+9+8}{4}, \tfrac{1+8+8+9}{4}\Bigr) = (6{,}75;\ 6{,}5).
$$

**Iterazione 2, assegnazione.** Ora $C(2,1)$ dista
$\sqrt{1{,}25} \approx 1{,}12$ da $\mu_1 = (1;\,1{,}5)$ ma ben $\approx 7{,}3$
da $\mu_2 = (6{,}75;\,6{,}5)$: **cambia gruppo** e passa a $C_1$. I punti
$D, E, F$ restano in $C_2$. Adesso $C_1 = \{A, B, C\}$, $C_2 = \{D, E, F\}$: i
due gruppi «veri».

**Iterazione 2, aggiornamento.**
$\mu_1 = (\tfrac{4}{3}; \tfrac{4}{3}) \approx (1{,}33; 1{,}33)$ e
$\mu_2 = (\tfrac{25}{3}; \tfrac{25}{3}) \approx (8{,}33; 8{,}33)$. Alla terza
iterazione nessun punto cambia più gruppo: l'algoritmo è **a convergenza**.
Nota come una partenza sbagliata si sia corretta da sola in due passi.

### Quante famiglie? Scegliere k

Il tallone d'Achille di k-means è che $k$ va deciso prima. Due strumenti
aiutano a sceglierlo.

`````{tab} Elementare

Il **metodo del gomito** (*elbow*): provi diversi valori di $k$ e, per
ciascuno, misuri quanto sono «strette» le famiglie (la somma delle distanze
dei punti dal proprio centro). All'aumentare di $k$ questo numero cala sempre
(con tanti centri ognuno è vicinissimo al suo) ma a un certo punto il guadagno
si appiattisce di colpo: il grafico forma un «gomito». Quel valore di $k$ è di
solito una buona scelta: aggiungere altri gruppi non compra quasi più niente.

La **silhouette**: per ogni punto confronta quanto è vicino ai compagni del suo
gruppo con quanto è vicino al gruppo estraneo più prossimo. Se il punteggio è
alto (vicino a $+1$) il punto è ben piazzato; se è basso o negativo è nel
gruppo sbagliato. La media su tutti i punti dice quanto è «pulita» la
partizione: si sceglie il $k$ che la massimizza.

`````

`````{tab} Superiore

Il metodo del gomito osserva l'**inerzia** $\mathcal{L}(k)$ in funzione di $k$
e cerca il punto di rendimento decrescente (la curvatura massima), un criterio
utile ma soggettivo. La **silhouette** lo rende quantitativo: per il punto $i$,
detta $a_i$ la distanza media dai punti del suo cluster e $b_i$ la distanza
media minima verso un altro cluster,

$$
s_i = \frac{b_i - a_i}{\max(a_i, b_i)} \in [-1, 1],
$$

dove $s_i \to 1$ indica un punto ben separato, $s_i \approx 0$ un punto al
confine, $s_i < 0$ un punto probabilmente mal assegnato. Il coefficiente medio
$\bar{s}$ si massimizza su $k$ per una scelta più oggettiva.

`````

I limiti di k-means, però, non si esauriscono nella scelta di $k$. L'algoritmo
assume che i cluster siano **sferici** e di dimensione simile (minimizza
distanze quadrate attorno a un centro), quindi inciampa su forme allungate o
concentriche. Ed è **sensibile all'inizializzazione**: partenze diverse
possono portare a minimi locali diversi. Il rimedio standard a quest'ultimo
punto è **k-means++**, che sceglie i centroidi iniziali lontani tra loro
invece che a caso: è oggi l'inizializzazione predefinita in scikit-learn.

## DBSCAN: seguire la densità, non i centri

E se i gruppi non fossero pallini tondi ma serpenti, anelli, spirali? Serve
un'idea diversa da «un centro e tutto ciò che gli sta attorno». **DBSCAN**
(Ester, Kriegel, Sander e Xu, 1996) {cite}`ester1996density` cambia
prospettiva: un cluster è una **regione densa** di punti, e le regioni dense
sono separate da zone quasi vuote.

`````{tab} Elementare

Guarda le luci di una città dall'aereo di notte. Non ti servono dei «centri»
per riconoscere i quartieri: li vedi come **zone fitte** di luci, separate da
buio. Un lampione isolato in campagna non è un quartiere, è solo un puntino
sperduto. DBSCAN ragiona così. Ha due manopole: un **raggio di vicinato**
(quanto vicini devono stare due punti per dirsi «vicini») e un **numero minimo
di vicini** perché una zona conti come densa. Con queste, parte da un punto in
una zona affollata e «cresce» il cluster contagiando i vicini, e i vicini dei
vicini, finché la densità regge. Quando i punti si diradano, il cluster
finisce.

Due regali rispetto a k-means. Primo: **non devi dire quanti gruppi cerchi**
(li scopre lui, contando le zone dense). Secondo: i punti isolati, quelli in
mezzo al buio, non vengono forzati dentro a nessun gruppo: DBSCAN li marca
come **rumore**. E poiché segue la forma della densità, riconosce famiglie di
qualunque sagoma: anche due lune intrecciate, dove k-means fallisce
miseramente ({numref}`fig-clustering-metodi`).

`````

`````{tab} Superiore

DBSCAN è governato da due parametri: il raggio $\varepsilon$ e la soglia
$\mathrm{minPts}$. Un punto è **core** se nel suo intorno di raggio
$\varepsilon$ cadono almeno $\mathrm{minPts}$ punti (sé stesso incluso). Un
cluster è un insieme massimale di punti *connessi per densità*: due punti core
appartengono allo stesso cluster se raggiungibili tramite una catena di punti
core a distanza $\le \varepsilon$; i punti non-core nell'intorno di un core
sono di **bordo** e vi si aggregano; tutti gli altri sono **rumore**, e non
appartengono ad alcun cluster. Il numero di cluster $k$ **non è un
parametro**: emerge dai dati. In compenso la scelta di $\varepsilon$ è
delicata (una regola pratica è ispezionare il grafico delle distanze al
$\mathrm{minPts}$-esimo vicino e cercarne il gomito), e DBSCAN soffre quando i
cluster hanno densità molto diverse tra loro: un $\varepsilon$ unico non può
adattarsi a tutte.

`````

```{figure} ../figures/clustering-metodi.svg
:name: fig-clustering-metodi
:alt: Confronto su dati a due lune intrecciate. A sinistra k-means separa i punti con un confine rettilineo verticale che taglia a metà entrambe le lune, colorandole in modo misto e scorretto; due centroidi sono segnati con una X. A destra DBSCAN colora ciascuna luna in modo uniforme e corretto (una teal, una terracotta) seguendo la densità, e marca due punti isolati come rumore in grigio.
:width: 100%

Due lune intrecciate. A sinistra k-means, che cerca cluster sferici attorno a
due centroidi, taglia le lune con un confine rettilineo e sbaglia. A destra
DBSCAN segue la densità, ricostruisce le due forme curve e isola il rumore.
```

Come mostra la {numref}`fig-clustering-metodi`, sui «due lune» k-means è
costretto a un confine dritto (impossibile separare due forme così con un
taglio netto attorno a due centri), mentre DBSCAN segue il filo della densità.
Non è che un metodo sia sempre migliore: k-means è veloce, scala benissimo e
va bene quando i gruppi sono blob compatti; DBSCAN brilla su forme irregolari
e in presenza di rumore, ma teme le densità disomogenee.

Una terza via, utile quando si vuole *esplorare* la struttura a diversi
livelli di granularità, è il **clustering gerarchico**: invece di fissare i
gruppi in un colpo solo, costruisce un albero di fusioni progressive; si parte
da ogni punto come cluster a sé e si fondono via via i più vicini. L'albero
risultante, il **dendrogramma**, si può «tagliare» a qualunque altezza per
ottenere il numero di cluster desiderato, decidendolo *dopo* aver visto la
struttura invece che prima. L'immagine giusta è un albero genealogico letto al
contrario: dai singoli individui alle famiglie, ai ceppi, alle popolazioni.
Nessun livello è "quello giusto" in assoluto: dipende dalla domanda.

Resta un dettaglio che cambia tutto: quando due cluster contengono molti punti,
cosa vuol dire che sono "vicini"? La risposta si chiama **linkage**, e la
scelta produce alberi molto diversi:

- **single**: distanza fra i due punti più vicini dei due gruppi. Segue bene
  le forme allungate, ma soffre di *concatenamento*, basta un ponte di punti
  sparsi per fondere due gruppi che a occhio sono separati;
- **complete**: distanza fra i due punti più lontani. Dà cluster compatti e di
  dimensioni simili, e spezza le forme allungate;
- **average**: la media delle distanze fra tutte le coppie, un compromesso;
- **Ward**: fonde la coppia che aumenta meno la varianza interna. È il default
  di scikit-learn e tende a produrre gruppi bilanciati: vicino nello spirito a
  k-means, con cui condivide il pregio e il difetto di preferire forme
  sferiche.

In pratica: `single` se ti aspetti strutture filiformi, `Ward` come punto di
partenza ragionevole in tutti gli altri casi.

## In pratica, con scikit-learn

Come per l'apprendimento supervisionato, in scikit-learn ogni tecnica è poche
righe, con la solita interfaccia `fit` (qui spesso `fit_transform` per chi
trasforma i dati, o `fit_predict` per chi assegna etichette di cluster):

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

# Standardizzare prima: PCA e le distanze sono sensibili alla scala
X_std = StandardScaler().fit_transform(X)

# --- Riduzione della dimensionalità ---
pca = PCA(n_components=2)          # tieni le prime 2 componenti
Z = pca.fit_transform(X_std)      # dati proiettati: (m, 2)
print(pca.explained_variance_ratio_)  # varianza spiegata da ogni componente

# Visualizzazione non lineare (solo per guardare, non per misurare)
Z_tsne = TSNE(n_components=2, perplexity=30).fit_transform(X_std)

# --- Clustering ---
km = KMeans(n_clusters=3, init="k-means++", n_init=10)
etichette_km = km.fit_predict(X_std)   # un intero per punto: 0, 1, 2

db = DBSCAN(eps=0.5, min_samples=5)
etichette_db = db.fit_predict(X_std)   # -1 marca il rumore
```

Due dettagli che fanno la differenza in pratica. La **standardizzazione**
prima di PCA o di qualunque clustering per distanza non è opzionale: senza, la
feature con la scala numerica più ampia domina il conto e falsa tutto. E le
etichette restituite dal clustering sono **arbitrarie**: il «cluster 0» di
k-means non ha alcun significato intrinseco, è solo un nome; due esecuzioni
possono scambiare i numeri senza che nulla sia cambiato.

```{admonition} Da ricordare
:class: important
- L'apprendimento **non supervisionato** lavora su dati **senza etichette**
  (la maggioranza dei dati reali) per scoprire una struttura nascosta.
- In alte dimensioni scatta la **maledizione della dimensionalità**: i volumi
  si concentrano sui bordi e le distanze si appiattiscono, mandando in crisi i
  metodi basati sulla vicinanza.
- La **PCA** trova le direzioni di **massima varianza** (autovettori della
  matrice di covarianza) e vi proietta i dati; è lineare, ottima per
  compressione, visualizzazione e denoising.
- **t-SNE** e **UMAP** visualizzano dati ad alta dimensione preservando la
  vicinanza **locale**: sulle loro mappe distanze globali, densità e dimensioni
  dei cluster **non sono affidabili**.
- **k-means** alterna assegnazione ai centroidi e ricalcolo delle medie
  (algoritmo di Lloyd); richiede $k$ a priori (gomito, silhouette), assume
  cluster **sferici** ed è sensibile all'inizializzazione (**k-means++**).
- **DBSCAN** raggruppa per **densità** ($\varepsilon$, $\mathrm{minPts}$):
  trova cluster di forma arbitraria, marca il **rumore** e non richiede $k$;
  il **clustering gerarchico** offre un dendrogramma da tagliare a piacere.
```
