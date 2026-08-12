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

```{figure} ../figures/maledizione-dimensionalita.svg
:name: fig-maledizione-dimensionalita
:alt: "Tre pannelli mostrano quanto volume occupa la sfera inscritta nel cubo al crescere delle dimensioni: il 78,5% in due dimensioni, il 52,4% in tre, e una frazione minuscola in dieci. All'aumentare delle dimensioni quasi tutto il volume del cubo si concentra negli angoli, lontano dal centro."
:width: 100%

Dove finisce lo spazio. La sfera è il centro, gli angoli sono la periferia: in
dieci dimensioni la periferia è praticamente tutto, e i punti si trovano
quasi sempre lì, lontani gli uni dagli altri.
```

Il conto di {numref}`fig-maledizione-dimensionalita` ha una conseguenza che
tocca ogni algoritmo basato sulle distanze. Se i punti finiscono tutti lontani
fra loro e a distanze simili, «il vicino più vicino» smette di voler dire
qualcosa: la differenza fra il primo e il centesimo vicino si assottiglia fino
a sparire nel rumore.

`````{tab} Elementare

Prima di tutto, di quali dimensioni stiamo parlando? Delle **colonne della
tabella**: nella sezione sull'apprendimento supervisionato abbiamo visto che
ogni colonna è una direzione dello spazio e ogni riga un punto. Se di ogni
cliente registriamo età, reddito, spese mensili e altre novantasette misure,
quel cliente è un punto in uno spazio a cento dimensioni. Non lo possiamo
disegnare, ma i conti (distanze comprese) si fanno identici a quelli su un
foglio.

Immagina allora di cercare un amico. In una **strada** (una dimensione) è
facile: sarà a pochi metri da te. In una **piazza** (due dimensioni) devi
guardarti attorno un po' di più. In un **grattacielo** (tre dimensioni) devi
anche scegliere il piano. Aggiungi una dimensione, cioè una colonna, e lo
spazio da esplorare si gonfia ogni volta: in dieci, cento, mille dimensioni,
tutto finisce per essere lontanissimo da tutto il resto; non ci sono più
«vicini», perché lo spazio è troppo vuoto.

C'è un secondo effetto, ancora più controintuitivo: in tante dimensioni, quasi
tutto lo spazio si accalca **sui bordi**. Prendi una scatola e considera il
guscio sottile vicino alla superficie, spesso un decimo del lato. Il conto lo
puoi fare da solo, perché è una potenza. Il «cuore» interno, quello che resta
tolto il guscio, è una scatola di lato $0{,}8$ (un decimo tolto di qua e uno di
là): in una dimensione occupa $0{,}8$ del totale, in due $0{,}8 \times 0{,}8 =
0{,}64$, in dieci $0{,}8$ moltiplicato per sé stesso dieci volte, cioè $0{,}11$.
Quindi il guscio è il 20% in una dimensione, il 36% in due, e già l'**89%** in
dieci. In cento dimensioni il cuore è praticamente zero e il guscio è quasi il
100%: nessun punto sta «nel mezzo», stanno tutti appiccicati alle pareti. In un
mondo così svuotato e spinto ai margini, gli algoritmi che si fidano delle
distanze («chi è vicino a chi») vanno in crisi. Da qui l'idea di **ridurre le
dimensioni**: togliere direzioni tenendo solo ciò che conta davvero.

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
$k$-NN, il clustering per distanza e la stima di densità {cite}`geron2022hands`.
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
numeri soltanto, quasi senza perdite.

La «dispersione» ha un nome tecnico, **varianza**, ed è la stessa parola che
abbiamo già usato parlando del compromesso bias-varianza. **Ma non è la stessa
cosa, e conviene tenerle separate.** Là la varianza era l'irrequietezza di un
*modello*: quanto cambiano le sue risposte se lo riaddestriamo su un campione
diverso. Qui è una proprietà dei *dati*, e non c'è nessun modello in giro:
quanto sono sparpagliati i punti lungo una direzione. Stessa parola perché il
conto che si fa è lo stesso (quanto le cose si scostano dalla loro media), ma
la cosa misurata è diversa: là un modello, qui un mucchio di numeri.

`````

`````{tab} Superiore

Sia $\mathbf{X} \in \mathbb{R}^{m \times d}$ la matrice dei dati, con le feature già
**centrate** (media di colonna nulla) e, di norma, **standardizzate** (varianza
unitaria, così che una feature misurata in metri non domini una misurata in
chilometri). La dispersione dei dati è riassunta dalla **matrice di
covarianza**

$$
\mathbf{C} = \frac{1}{m}\, \mathbf{X}^{\top} \mathbf{X} \in \mathbb{R}^{d \times d},
$$

dove $C_{jk}$ è la covarianza tra la feature $j$ e la feature $k$. La PCA
cerca il versore $\mathbf{u}$ che massimizza la varianza dei dati proiettati,
$\operatorname{Var}(\mathbf{X}\mathbf{u}) =
\mathbf{u}^{\top}\mathbf{C}\,\mathbf{u}$, con il vincolo
$\lVert \mathbf{u} \rVert = 1$.
Con i moltiplicatori di Lagrange il problema diventa

$$
\mathbf{C}\, \mathbf{u} = \lambda\, \mathbf{u} ,
$$

cioè le direzioni cercate sono gli **autovettori** di $\mathbf{C}$, e la varianza
catturata da ciascuna è il corrispondente **autovalore** $\lambda$. Ordinando
gli autovalori $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_d \ge 0$, la
prima componente principale è l'autovettore di $\lambda_1$, la seconda quello
di $\lambda_2$, e così via; e si possono sempre **scegliere ortogonali** fra
loro, perché $\mathbf{C}$ è simmetrica (con autovalori distinti lo sono per
forza; se un autovalore è ripetuto, il teorema spettrale garantisce che una
base ortonormale del suo autospazio esiste, ed è quella che ogni
implementazione restituisce).
Proiettare su $\mathbf{u}_1, \dots, \mathbf{u}_k$ (con $k \ll d$) dà la
rappresentazione ridotta
$\mathbf{Z} = \mathbf{X}\,\mathbf{U}_k$, dove $\mathbf{U}_k$ raccoglie i
primi $k$ autovettori in colonna.

`````

Vale la pena vedere lo stesso meccanismo su un esempio minuscolo: quattro
punti in due dimensioni, da proiettare su una sola.

`````{tab} Elementare

Disegna quattro punti su un foglio a quadretti: $(2,2)$ e $(-2,-2)$ sulla
diagonale che sale, $(1,-1)$ e $(-1,1)$ su quella che scende. Lo «stormo» è
chiaramente più allungato lungo la diagonale che sale: i due punti che ci
stanno sopra sono i più lontani dal centro.

Misuriamo la dispersione nelle due direzioni, e i conti li puoi rifare con
Pitagora. Lungo la diagonale che sale, i due punti che ci stanno sopra distano
dal centro $\sqrt{2^2+2^2} = \sqrt{8}$, mentre gli altri due la attraversano
esattamente al centro e distano $0$: la dispersione, cioè la media dei quadrati
di quelle quattro distanze, vale $(8 + 0 + 8 + 0)/4 = 4$. Lungo la diagonale
che scende succede il contrario, con distanze $\sqrt{1^2+1^2} = \sqrt2$ per due
punti e $0$ per gli altri due: $(0+2+0+2)/4 = 1$. Quindi la prima direzione, da
sola, cattura $4$ su $5$, cioè l'$80\%$ del totale.

Proiettare significa far cadere ogni punto, come un'ombra, sulla diagonale
che sale. I due punti che già ci stavano sopra conservano tutta la loro
distanza; gli altri due cadono entrambi sul centro, e la loro (piccola)
differenza va persa. Risultato: ogni punto è descritto da un numero solo
invece che da due, e abbiamo tenuto i quattro quinti dell'informazione.

`````

`````{tab} Superiore

Prendiamo quattro punti già centrati (media nulla), così da saltare la
sottrazione della media:

$$
\mathbf{X} = \{(2,2),\ (1,-1),\ (-2,-2),\ (-1,1)\}.
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
\mathbf{C} = \begin{pmatrix} 2{,}5 & 1{,}5 \\ 1{,}5 & 2{,}5 \end{pmatrix}.
$$

Gli autovalori risolvono $\det(\mathbf{C} - \lambda \mathbf{I}) = 0$, cioè
$(2{,}5-\lambda)^2 - 1{,}5^2 = 0$, da cui $2{,}5 - \lambda = \pm 1{,}5$ e

$$
\lambda_1 = 4, \qquad \lambda_2 = 1 .
$$

Gli autovettori corrispondenti sono $\mathbf{u}_1 = \tfrac{1}{\sqrt2}(1, 1)$
(la
diagonale che sale) e $\mathbf{u}_2 = \tfrac{1}{\sqrt2}(1, -1)$. La **varianza
spiegata** dalla prima componente è

$$
\frac{\lambda_1}{\lambda_1 + \lambda_2} = \frac{4}{4 + 1} = 0{,}8,
$$

l'80%: proiettando su $\mathbf{u}_1$ soltanto, buttiamo via una dimensione ma
conserviamo i quattro quinti della dispersione. La proiezione di ogni punto è
il prodotto scalare con $\mathbf{u}_1$:

$$
(2,2)\!\cdot\! \mathbf{u}_1 = \tfrac{4}{\sqrt2} = 2\sqrt2,\quad
(1,-1)\!\cdot\! \mathbf{u}_1 = 0,\quad
(-2,-2)\!\cdot\! \mathbf{u}_1 = -2\sqrt2,\quad
(-1,1)\!\cdot\! \mathbf{u}_1 = 0.
$$

I due punti «fuori diagonale» collassano a $0$ (stavano interamente lungo la
direzione scartata $\mathbf{u}_2$), mentre i due «in diagonale» conservano tutta la
loro distanza.

`````

La {numref}`fig-pca-proiezione` mostra la stessa idea su una
nuvola più fitta: l'asse lungo è la prima componente, quello corto la
seconda, e proiettare significa lasciar cadere ogni punto perpendicolarmente
sull'asse lungo.

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
dataset con centinaia di feature; a **ripulirli dal rumore** (*denoising*): le
differenze vere tra un esempio e l'altro si concentrano nelle prime componenti,
mentre nelle ultime resta quasi soltanto confusione, e buttare via quelle
direzioni lascia un dato più pulito. La confusione che si era mescolata alle
prime componenti, però, resta lì: la PCA non la sa distinguere. Il limite
cruciale è che la PCA è **lineare**, sa solo ruotare e proiettare lungo assi
dritti. Se la struttura interessante dei dati è curva (un rotolo arrotolato su
sé stesso, due spirali intrecciate), la PCA la appiattisce e la rovina. Per
quei casi servono metodi non lineari.

## t-SNE e UMAP: vedere in due dimensioni ciò che vive in mille

Quando l'obiettivo è soltanto **guardare** dati ad alta dimensione (non
comprimerli per un modello a valle, ma disegnarli su uno schermo per capirli
con gli occhi), la PCA spesso non basta. Sono nate tecniche pensate apposta
per la visualizzazione, che rinunciano alla linearità pur di rendere leggibile
una mappa in 2D: le più note sono **t-SNE** (van der Maaten e Hinton, 2008)
{cite}`maaten2008visualizing` e **UMAP** (McInnes, Healy e Melville, 2018)
{cite}`mcinnes2018umap`.

```{figure} ../figures/tsne-umap.svg
:name: fig-tsne-umap
:alt: "A sinistra un groviglio di punti in 784 dimensioni, con le classi intrecciate e le distanze illeggibili. Una freccia di proiezione, etichettata t-SNE e UMAP, porta a destra, dove la stessa informazione appare come una mappa piatta in due dimensioni con tre gruppi ben separati; una nota precisa che sono i vicinati a essere preservati."
:width: 96%

Da un groviglio a una mappa. Ciò che queste tecniche promettono di conservare
sono i *vicinati*, cioè chi sta vicino a chi, non le distanze assolute né la
posizione dei gruppi fra loro.
```

Quella promessa limitata di {numref}`fig-tsne-umap` è anche l'avvertenza
d'uso: su una mappa t-SNE la dimensione di un gruppo e la distanza fra due
gruppi non vogliono dire quasi niente, e leggerle come se fossero misure è
l'errore più comune che si fa con queste figure.

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
$p_{j\mid i} \propto \exp(-\lVert \mathbf{x}_i - \mathbf{x}_j\rVert^2 / 2\sigma_i^2)$, con
$\sigma_i$ tarato localmente da un iperparametro, la *perplexity*. Queste
condizionate non sono simmetriche ($p_{j\mid i} \neq p_{i\mid j}$, perché
$\sigma_i$ e $\sigma_j$ differiscono) e vengono simmetrizzate in una congiunta,
$p_{ij} = (p_{j\mid i} + p_{i\mid j})/2n$: è il passaggio che distingue t-SNE
dal SNE originale di Hinton e Roweis. Nello
spazio ridotto usa invece una $t$ di Student a un grado di libertà (con code
pesanti, che evitano l'affollamento al centro),
$q_{ij} \propto (1 + \lVert \mathbf{z}_i - \mathbf{z}_j\rVert^2)^{-1}$, e
dispone i punti $\mathbf{z}_i$
minimizzando la divergenza di Kullback–Leibler $\mathrm{KL}(P \Vert Q)$ tra le
due distribuzioni **congiunte**. Poiché la KL pesa molto le vicinanze e poco le
lontananze,
t-SNE **preserva la struttura locale** ma distorce quella globale: distanze
tra cluster, densità e dimensioni apparenti sui grafici **non sono
quantitativamente affidabili**.

UMAP parte da fondamenta diverse (una costruzione su grafi e topologia) ma
persegue un obiettivo simile; in pratica è **più veloce**, scala meglio a
milioni di punti e tende a **preservare meglio la struttura globale**. Su
quest'ultimo punto vale però una precisazione che ridimensiona il confronto:
Kobak e Linderman {cite}`kobak2021initialization` hanno mostrato che il divario
si annulla inizializzando t-SNE con la PCA invece che a caso, ed è quindi
l'**inizializzazione**, più dell'algoritmo, a decidere quanto sopravvive della
struttura globale (in scikit-learn, `init="pca"`). Resta comunque lo stesso
monito per entrambi: sono strumenti di visualizzazione, non di
analisi metrica. Entrambi vanno usati per *esplorare*, mai per concludere che
«questo gruppo è il doppio più lontano di quell'altro».

`````

```{admonition} Quando usarli, e quando no
:class: tip
**Sì**: guardare se un dataset ha una struttura a gruppi prima di modellare;
ispezionare gli *embedding* di una rete per capire cosa ha imparato; presentare
a un pubblico la forma di dati che nessuno può visualizzare in $784$ dimensioni.

**No**: come passo di preprocessing prima di un classificatore, per quello
serve la PCA, che è lineare, si applica a **dati nuovi** con la stessa matrice
e permette di tornare nello spazio di partenza con una ricostruzione
approssimata (approssimata e non esatta: proiettare su meno direzioni butta via
informazione per sempre, ed è lo stesso $20\%$ dell'esempio di poco fa; solo
tenendo tutte le componenti la ricostruzione è esatta). Né t-SNE
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

```{figure} ../figures/k-means-raggruppare-senza-etichette.svg
:name: fig-kmeans-migrazione
:alt: "Tre pannelli in sequenza sulla stessa nube di punti. Nel primo i centroidi, segnati con una x, sono in posizione casuale e le assegnazioni sono sbagliate. Nel secondo i centroidi si sono spostati verso il centro dei rispettivi gruppi. Nel terzo hanno raggiunto la convergenza e i gruppi sono corretti. Una nota spiega che a ogni iterazione il centroide si sposta nella media dei punti che gli sono stati assegnati."
:width: 100%

Le due mosse di k-means, ripetute. Ogni punto va al centroide più vicino, poi
ogni centroide va nel mezzo dei punti che gli sono toccati: da un inizio a
caso si arriva ai gruppi in poche iterazioni.
```

Guardando {numref}`fig-kmeans-migrazione` si capisce anche la fragilità
dell'algoritmo: il primo pannello è casuale, e da inizi diversi si può
convergere a soluzioni diverse. È il motivo per cui in pratica k-means si fa
ripartire più volte, tenendo la soluzione migliore.

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
\mathcal{L}(\mu, c) = \sum_{i=1}^{m} \bigl\lVert \mathbf{x}_i - \mu_{c_i} \bigr\rVert^2,
$$

dove $c_i \in \{1, \dots, k\}$ è il cluster assegnato al punto $\mathbf{x}_i$. L'algoritmo
di Lloyd minimizza $\mathcal{L}$ alternando due passi di coordinate:

$$
\textbf{assegnazione:}\quad
c_i = \arg\min_{j} \lVert \mathbf{x}_i - \mu_j \rVert^2,
\qquad
\textbf{aggiornamento:}\quad
\mu_j = \frac{1}{|C_j|} \sum_{i \in C_j} \mathbf{x}_i .
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

Il centro di un gruppo si chiama **centroide** e nelle formule si scrive con la
lettera greca $\mu$ (*mi*), che in statistica indica da sempre una media: un
centroide, in fondo, è la media delle posizioni dei suoi punti. Le distanze fra
due punti le calcoliamo con Pitagora, come sul foglio a quadretti: differenza
delle ascisse e delle ordinate, ciascuna al quadrato, sommate, e radice.

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
Nota come una partenza sbagliata si sia corretta da sola in due passi
({numref}`fig-kmeans-converge`): l'unico momento in cui succede qualcosa di non
ovvio è quando $C$ cambia gruppo, e da lì in poi non si muove più niente.

```{figure} ../figures/kmeans-converge.svg
:name: fig-kmeans-converge
:alt: I sei punti A, B, C, D, E, F dell'esempio su un piano, e due centroidi segnati con una x che partono tutti e due nel gruppo di sinistra. Poi si alternano due mosse: nell'assegnazione ogni punto prende il colore del centroide più vicino, nell'aggiornamento ogni centroide si sposta nella media dei suoi punti lasciando una scia tratteggiata. Alla seconda assegnazione il punto C passa al gruppo di sinistra; dopo il secondo aggiornamento nessun punto cambia più colore e l'algoritmo si ferma.
:width: 90%

Le due mosse sui sei punti dell'esempio: ogni punto prende il colore del
centroide più vicino, poi ogni centroide si sposta nella media dei suoi punti.
Al secondo giro $C$ cambia gruppo, e da lì non si muove più niente.
```

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
possono portare a soluzioni diverse, ciascuna delle quali è un **minimo
locale**, cioè un assetto che non si può migliorare con una mossa piccola pur
non essendo il migliore possibile (come una pallina che si ferma in una
conchetta a mezza costa invece di arrivare a valle: da lì, in qualunque
direzione si guardi, si sale). Il rimedio standard è
**k-means++**, che sceglie i centroidi iniziali lontani tra loro
invece che a caso: è oggi l'inizializzazione predefinita in scikit-learn. Il
secondo rimedio è farlo ripartire più volte e tenere la soluzione migliore.

## DBSCAN: seguire la densità, non i centri

E se i gruppi non fossero pallini tondi ma serpenti, anelli, spirali? Serve
un'idea diversa da «un centro e tutto ciò che gli sta attorno». **DBSCAN**
(Ester, Kriegel, Sander e Xu, 1996) {cite}`ester1996density` cambia
prospettiva: un cluster è una **regione densa** di punti, e le regioni dense
sono separate da zone quasi vuote.

```{figure} ../figures/dbscan-clustering-gerarchico.svg
:name: fig-dbscan-dendrogramma
:alt: "A sinistra DBSCAN su due cluster di forma irregolare: i punti sono distinti in core, che hanno abbastanza vicini, border, che stanno ai margini di un cluster, e rumore, isolati e non assegnati a nessun gruppo. A destra un dendrogramma del clustering gerarchico, con una linea di taglio orizzontale che determina quanti gruppi si ottengono."
:width: 100%

Due modi di non dover dire quanti gruppi cercare. DBSCAN lo deduce dalla
densità e ammette che alcuni punti non appartengano a nessuno; il gerarchico
li ordina tutti e lascia la scelta all'altezza del taglio.
```

La categoria «rumore» in {numref}`fig-dbscan-dendrogramma` è la differenza
pratica più importante rispetto a k-means. Là ogni punto finisce per forza in
un gruppo, anche quello isolato in mezzo al nulla, che trascina il centroide;
qui un punto può restare fuori, e i cluster non vengono deformati da chi non
c'entra.

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

Immagina due comitive in gita e chiediti quanto distano fra loro: la risposta
cambia secondo chi guardi.

- **single** («legame singolo»): distanza fra i due membri più vicini, uno per
  comitiva. Due gruppi sono vicini se anche solo due persone si sfiorano.
  Segue bene le forme allungate (un serpente di punti resta un serpente), ma
  soffre di *concatenamento*: basta una fila di passanti sparsi a fare da ponte
  perché due comitive lontanissime vengano dichiarate una sola;
- **complete** («legame completo»): distanza fra i due membri più lontani. Due
  gruppi si fondono solo se stanno stretti *tutti quanti*: ne escono cluster
  compatti e di dimensioni simili, al prezzo di spezzare le forme allungate;
- **average**: la media delle distanze fra tutte le coppie, un compromesso fra
  i due;
- **Ward**: fonde la coppia che, una volta unita, resta la più raccolta (in
  termini tecnici, quella che aumenta meno la varianza interna). È il default
  di scikit-learn e tende a produrre gruppi bilanciati: vicino nello spirito a
  k-means, con cui condivide il pregio e il difetto di preferire forme
  sferiche.

In pratica: `single` se ti aspetti strutture filiformi, `Ward` come punto di
partenza ragionevole in tutti gli altri casi.

## Misture gaussiane: dal gruppo alla distribuzione

Tutti i metodi visti finora rispondono alla stessa domanda («a quale gruppo
appartiene questo punto?») dando la stessa forma di risposta: un nome, secco.
C'è un'altra famiglia che risponde con una **probabilità**, e nel farlo cambia
anche cosa impara di ciascun gruppo.

`````{tab} Elementare

Torna sul difetto di k-means che abbiamo già incontrato: preferisce i gruppi
tondi e della stessa taglia, perché tutto ciò che sa di un gruppo è **dove sta
il suo centro**. Se un gruppo è una nuvola allungata e uno è una pallina
stretta, il centro da solo non basta a distinguerli, e i punti sul confine
finiscono dalla parte sbagliata.

Le **misture gaussiane** cambiano due cose insieme. (Il nome viene da Carl
Friedrich Gauss: una *gaussiana* è la curva a campana, quella che descrive
l'altezza delle persone o gli errori di una misura, con tanti valori
addensati attorno a una media e sempre meno man mano che ci si allontana. Una
«mistura» di gaussiane è semplicemente più campane sovrapposte, una per
gruppo.)

La prima: di ogni gruppo non imparano solo il centro, ma anche la **forma**.
Quanto è largo, quanto è allungato, in che direzione è orientato. Un gruppo
non è più un puntino, è una macchia con un suo profilo.

La seconda: l'appartenenza smette di essere un sì o un no. Ogni punto riceve
una risposta come «sono al 90% del gruppo A e al 10% del gruppo B». Per i
punti nel cuore di una nuvola la risposta sarà quasi certa; per quelli sul
confine sarà divisa, ed è giusto che lo sia, perché sul confine l'incertezza
c'è davvero. Chi decide una strategia commerciale su quei clienti farebbe bene
a saperlo, invece di ricevere un'etichetta che finge una sicurezza inesistente.

Come si impara tutto questo? Con un ragionamento circolare che si scioglie
girandolo. *Se* sapessi a quale gruppo appartiene ogni punto, calcolare centro
e forma di ogni gruppo sarebbe una media. *Se* conoscessi centri e forme,
calcolare l'appartenenza di ogni punto sarebbe un confronto. Non sai né l'una
né l'altra cosa, quindi tiri a indovinare e poi alterni: aggiorni le
appartenenze usando le forme attuali, aggiorni le forme usando le appartenenze
attuali, e ripeti. Ogni giro la spiegazione dei dati migliora un pochino,
finché smette di migliorare.

Quel ciclo si chiama **algoritmo EM**, dalle due mosse che alterna:
*expectation* (l'attesa, cioè indovinare a quale gruppo appartiene ogni punto,
date le forme attuali) e *maximization* (la massimizzazione, cioè ricalcolare
centri e forme al meglio, date quelle appartenenze). È una delle idee più
riusate di tutta la statistica: lo stesso schema, sotto altri nomi, fa
funzionare i vecchi sistemi di riconoscimento vocale e uno dei tokenizzatori
che il libro incontrerà più avanti.

`````

`````{tab} Superiore

Un **modello di mistura gaussiana** (*Gaussian Mixture Model*, GMM) è un
modello **generativo**: assume che ogni punto sia stato prodotto scegliendo
prima una componente e poi campionando dalla sua gaussiana. La densità è

$$
p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \,
\mathcal{N}(\mathbf{x} \mid \mu_k, \Sigma_k),
\qquad \pi_k \geq 0, \;\; \sum_k \pi_k = 1,
$$

con $\pi_k$ i pesi di mistura, $\mu_k$ le medie e
$\Sigma_k$ le covarianze, che sono ciò che k-means non ha. Il
parametro da stimare è $\theta = \{\pi_k, \mu_k,
\Sigma_k\}$, per massima verosimiglianza:

$$
\log p(\mathbf{X} \mid \theta) = \sum_{i=1}^{m} \log \sum_{k=1}^{K} \pi_k\,
\mathcal{N}(\mathbf{x}^{(i)} \mid \mu_k, \Sigma_k).
$$

Il logaritmo di una somma non si separa, e annullando il gradiente non si
ottiene una forma chiusa. Il rimedio è introdurre una **variabile latente**
$z^{(i)} \in \{1,\dots,K\}$, l'identità della componente che ha generato il
punto: se le $z^{(i)}$ fossero note, la stima sarebbe immediata.

L'algoritmo **EM**, formalizzato da Dempster, Laird e Rubin nel 1977
{cite}`dempster1977maximum`, alterna due passi.

**Passo E** (*expectation*): a parametri fissi, calcola le
**responsabilità**, cioè la posteriore di ogni componente su ogni punto,

$$
\gamma_{ik} = p(z^{(i)} = k \mid \mathbf{x}^{(i)}) =
\frac{\pi_k \, \mathcal{N}(\mathbf{x}^{(i)} \mid \mu_k,
\Sigma_k)}
{\sum_{j} \pi_j \, \mathcal{N}(\mathbf{x}^{(i)} \mid \mu_j,
\Sigma_j)} .
$$

**Passo M** (*maximization*): a responsabilità fisse, ristima i parametri con
medie pesate, dove il peso è la responsabilità e $m_k = \sum_i \gamma_{ik}$ è
la massa della componente:

$$
\mu_k = \frac{1}{m_k}\sum_i \gamma_{ik}\,\mathbf{x}^{(i)},
\qquad
\Sigma_k = \frac{1}{m_k}\sum_i \gamma_{ik}
(\mathbf{x}^{(i)} - \mu_k)(\mathbf{x}^{(i)} -
\mu_k)^\top,
\qquad
\pi_k = \frac{m_k}{m}.
$$

La proprietà che rende EM un algoritmo e non un'euristica: **la verosimiglianza
non decresce mai**, perché ogni iterazione massimizza una funzione che sta
sotto di essa e la tocca nel punto corrente. Non garantisce l'ottimo globale
(la verosimiglianza è multimodale, e da inizializzazioni diverse si arriva a
soluzioni diverse: per questo si inizializza tipicamente con k-means e si
riparte più volte), garantisce la monotonia.

Due letture che pagano nel resto del libro. La prima: **k-means è il caso
limite** di EM su una mistura con covarianze $\sigma^2\mathbf{I}$ e
$\sigma^2 \to 0$, dove le responsabilità collassano su 0 e 1. L'assegnazione
dura non è un metodo diverso, è la versione degenere di quella morbida, e la
preferenza di k-means per gruppi sferici è scritta in quella $\mathbf{I}$. La
seconda: essendo generativo, un GMM restituisce una **densità**, quindi serve
anche a quello che il clustering non fa, cioè segnalare i punti improbabili.
È uno dei rilevatori di anomalie di riferimento, e riaggancia il capitolo sui
dati che cambiano.

Poiché il modello ha una verosimiglianza, il numero di componenti si sceglie
con un criterio di informazione invece che a occhio. **BIC** e **AIC** sommano
a $-2$ volte la log-verosimiglianza una penalità sul numero di parametri
($p\log m$ per il BIC, $2p$ per l'AIC), e si prende il $K$ che li **minimizza**:
il primo termine premia chi spiega bene i dati (cambiato di segno, quindi
minimizzarlo vuol dire massimizzare la verosimiglianza), il secondo fa pagare i
parametri usati per farlo. È la convenzione di `GaussianMixture.bic`, quella
usata dal codice più sotto, e una risposta più difendibile del gomito o
della silhouette, che sono diagnostiche geometriche senza un modello sotto.

`````

L'algoritmo EM merita di essere riconosciuto quando ricompare, perché ricompare
spesso e sotto altri nomi. Il capitolo sul riconoscimento vocale (in inglese
*Automatic Speech Recognition*, da cui la sigla **ASR**) racconta i sistemi che
per trent'anni hanno trascritto il parlato accoppiando una catena di stati
nascosti alle misture gaussiane di questa sezione: le seconde sono proprio
queste, e il loro addestramento è EM. Il capitolo sui tokenizzatori descrive
il modello **Unigram** di SentencePiece, che pota il vocabolario stimando le
probabilità dei pezzi: anche lì il motore è EM, con la segmentazione al posto
dell'identità della componente. Lo schema è sempre lo stesso, e conviene
tenerlo come sagoma: *quando la cosa che renderebbe facile la stima è proprio
quella che non osservi, stimala e alterna*.

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

La differenza fra assegnazione dura e morbida non è teorica: si vede su due
gruppi allungati e vicini, esattamente il caso in cui il centro da solo non
basta.

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

rng = np.random.default_rng(1)
# due nuvole allungate nella stessa direzione, vicine fra loro
forma = [[4.0, 0.0], [0.0, 0.15]]
X = np.vstack([rng.multivariate_normal([0.0, 0.0], forma, 300),
               rng.multivariate_normal([1.0, 2.2], forma, 300)])
vero = np.r_[np.zeros(300), np.ones(300)]

def concordanza(a, b):
    """Quota di punti d'accordo, a meno di uno scambio dei nomi dei cluster."""
    return max((a == b).mean(), (a != b).mean())

km = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
gm = GaussianMixture(n_components=2, covariance_type="full",
                     random_state=0).fit(X)

print(f"k-means           : {concordanza(km, vero):.3f}")
print(f"mistura gaussiana : {concordanza(gm.predict(X), vero):.3f}")

# l'assegnazione morbida: quanto ogni punto appartiene a ciascun gruppo
incerti = (gm.predict_proba(X).max(axis=1) < 0.9).sum()
print(f"punti su cui il modello resta incerto: {incerti} su {len(X)}")

# quanti gruppi? con una verosimiglianza sotto, lo dice il BIC
for k in range(1, 6):
    bic = GaussianMixture(n_components=k, covariance_type="full",
                          random_state=0).fit(X).bic(X)
    print(f"  k={k}  BIC={bic:9.1f}")
```

k-means si ferma a $0{,}663$, appena sopra il tirare a caso: le due nuvole sono
allungate, e la frontiera a metà strada fra i due centri le taglia di
traverso. La mistura arriva a $0{,}997$, perché ha imparato che i gruppi sono
larghi in una direzione e stretti nell'altra. Restano **cinque punti** su
seicento su cui il modello non si sbilancia oltre il 90%, e sono quelli in
mezzo: non è un difetto, è l'unica risposta onesta lì.

Resta l'ultima stampa, quella del **BIC**: un punteggio che mette insieme due
cose opposte, quanto bene il modello spiega i dati e quanti parametri ha
speso per farlo, e che quindi si può confrontare fra modelli di taglia diversa.
Più basso è, meglio è. Qui tocca il minimo esattamente a $k=2$: avendo un
modello probabilistico sotto, il numero dei gruppi si sceglie con un criterio
invece che con un giudizio a occhio sul grafico.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Qui i dati arrivano **muti**, senza risposta giusta accanto, ed è il caso
  della stragrande maggioranza dei dati del mondo. Si può chiedere loro due
  cose: se si possono descrivere con meno colonne, e se contengono gruppi
  naturali.
- Troppe colonne sono un problema, non una ricchezza: in uno spazio con tante
  direzioni i punti finiscono tutti lontani e tutti alla stessa distanza, e
  «chi somiglia a chi» perde senso. È la **maledizione della dimensionalità**.
- La **PCA** cerca le direzioni lungo cui i punti sono più sparpagliati e butta
  le altre: è la fotografia dello stormo scattata dal lato giusto. Serve a
  comprimere, a disegnare in due dimensioni ciò che ne ha cento, e a ripulire
  dal rumore.
- **t-SNE** e **UMAP** disegnano mappe bellissime, dove chi si somigliava
  finisce vicino. Ma sono ottime per l'occhio e pessime per il righello: la
  distanza fra due isole, la loro grandezza e quanto sono fitte non vogliono
  dire quasi niente.
- **k-means** raggruppa alternando due mosse: ognuno va al punto di ritrovo più
  vicino, poi ogni ritrovo si sposta in mezzo ai suoi. Bisogna dirgli quanti
  gruppi cercare, e preferisce i gruppi tondi e della stessa taglia.
- **DBSCAN** guarda invece le zone fitte, come le luci di una città viste
  dall'aereo: scopre da solo quanti gruppi ci sono, riconosce forme di qualunque
  sagoma, e ha il buon senso di lasciare fuori i puntini isolati.
- Le **misture gaussiane** imparano di ogni gruppo anche la forma, non solo il
  centro, e invece di un'etichetta secca rispondono «al 90% di qua e al 10% di
  là»: sul confine l'incertezza c'è davvero, ed è onesto dirlo.
```

`````

`````{tab} Superiore

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
- Le **misture gaussiane** imparano di ogni gruppo non solo il centro ma la
  **forma** (la covarianza), e assegnano una **probabilità** invece di
  un'etichetta secca. Si stimano con l'**algoritmo EM**, che alterna il calcolo
  delle responsabilità (passo E) e la ristima dei parametri (passo M) e
  garantisce che la verosimiglianza non decresca. **k-means è il caso limite**
  di questo schema con covarianze sferiche che tendono a zero.
- Avendo un modello probabilistico sotto, una mistura dà una **densità**
  (quindi serve anche per le anomalie) e permette di scegliere il numero di
  componenti con **BIC** o **AIC** invece che a occhio: penalità sommata a
  $-2\log L$, e si minimizza.
```

`````
