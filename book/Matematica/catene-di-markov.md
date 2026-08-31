# Catene di Markov: la matrice che si applica per sempre

Nel gennaio del 1990 il *New York Times* dedicò un articolo a un risultato di
matematica pura: per mescolare davvero un mazzo di cinquantadue carte servono
sette mescolate, e la settima non è una scelta di stile. Dave Bayer e Persi
Diaconis avevano studiato che cosa succede a un mazzo mescolato alla maniera
dei croupier, dividendolo in due e facendo cadere le carte a intreccio, e
avevano scoperto un fenomeno inatteso: fino a quattro o cinque mescolate il
mazzo conserva quasi tutto l'ordine di partenza, alla settima la distanza da un
mazzo qualsiasi scende per la prima volta sotto la metà, e dalla nona in poi
mescolare ancora non serve quasi a niente {cite}`bayer1992trailing`. Fra il
mazzo che ricorda ancora l'ordine di prima e il mazzo che l'ha dimenticato non
c'è una salita graduale, c'è una soglia, e sta fra la quinta mescolata e la
settima.

Quel risultato riguarda un mazzo di carte e la matematica che lo produce è
tutta qui dentro. Un mazzo mescolato è uno **stato**; la mescolata è una regola
che porta da uno stato al successivo, con un po' di caso; e la domanda «dopo
quante volte non si distingue più da un mazzo qualsiasi?» ha una risposta che,
sorprendentemente, si legge in un autovalore.

## Il patto: domani dipende solo da oggi

`````{tab} Elementare

Nel gioco dell'oca, dove finisci al prossimo turno dipende da due cose: la
casella su cui ti trovi adesso e il numero che esce dal dado. Non conta come ci
sei arrivato, se hai fatto un percorso fortunato o disastroso, se hai già perso
tre turni. Tutta la tua storia è riassunta da un solo dato, la casella.

Questo è il **patto di Markov**, e i sistemi che lo rispettano si chiamano
catene di Markov. È una restrizione seria, ed è proprio per questo che porta
lontano: un sistema che ricorda tutto è impossibile da studiare, uno che
ricorda solo dove si trova si maneggia con l'algebra lineare.

C'è una mossa da imparare qui, perché serve continuamente. Se il gioco avesse
una regola del tipo «chi è già finito in prigione una volta, la seconda volta
esce subito», il patto sarebbe rotto: la casella da sola non basterebbe più a
decidere il futuro. Ma basta cambiare che cosa si chiama stato. Invece della
sola casella si prende la coppia (casella, sei già stato in prigione?), e il
patto torna valido, con il doppio degli stati. **Quasi ogni sistema si può
rendere markoviano allargando la definizione di stato**, e il prezzo è che gli
stati si moltiplicano.

Descritto il gioco, quello che si vuole sapere è dove si finisce. Non in che
casella cadrai tu stasera, che dipende dai dadi, ma come si distribuiscono i
giocatori sulle caselle dopo molti turni: quale frazione sta qui, quale lì. Il
gioco parte da una distribuzione (all'inizio tutti sulla casella di partenza) e
a ogni turno la rimescola; la domanda è dove quella distribuzione va a finire.

`````

`````{tab} Superiore

Un processo $\{S_t\}_{t\ge 0}$ a valori in un insieme finito
$\{1,\dots,n\}$ è una **catena di Markov** se

$$
\Pr[S_{t+1} = i \mid S_t = j, S_{t-1}, \dots, S_0]
= \Pr[S_{t+1} = i \mid S_t = j] \;=\; P_{ij},
$$

cioè se il futuro è condizionatamente indipendente dal passato dato il
presente. I numeri $P_{ij}$ formano la **matrice di transizione**
$\mathbf{P}\in\mathbb{R}^{n\times n}$.

Una convenzione va fissata subito perché i testi si dividono. Qui
$\mathbf{P}$ è **stocastica per colonne**: $P_{ij}$ è la probabilità di andare
in $i$ **partendo da** $j$, quindi ogni colonna è una distribuzione di
probabilità e somma a $1$. Con questa scelta lo stato del sistema è un vettore
colonna $\mathbf{x}_t\in\mathbb{R}^n$ di probabilità, e l'evoluzione è la
moltiplicazione a sinistra che il resto del capitolo usa dappertutto:

$$
\mathbf{x}_{t+1} = \mathbf{P}\,\mathbf{x}_t,
\qquad\text{quindi}\qquad
\mathbf{x}_t = \mathbf{P}^t\,\mathbf{x}_0 .
$$

Molti testi di probabilità adottano la convenzione trasposta (righe
stocastiche, vettori riga, $\boldsymbol{\pi}_{t+1} = \boldsymbol{\pi}_t
\mathbf{P}$): confrontando una formula con la sua fonte conviene guardare da
che parte sta la somma a uno.

L'ipotesi di Markov è meno restrittiva di quanto sembri, per via
dell’**allargamento dello stato**: un processo che dipende dagli ultimi $k$
istanti diventa markoviano prendendo come stato la $k$-upla degli ultimi $k$
valori, al prezzo di $n^k$ stati. È esattamente la costruzione dei modelli
$n$-gram, e la stessa che porta dai processi decisionali di Markov alle loro
versioni con memoria.

`````

## Dove si finisce: un autovettore di autovalore uno

```{figure} ../figures/catena-si-assesta.svg
:name: fig-catena-si-assesta
:alt: "Quattro colonne, una per pagina del web in miniatura, alte quanto la probabilità di trovarsi su quella pagina. Si parte da quattro colonne uguali, alte 0,25 ciascuna; dopo un passo la colonna C scavalca il proprio bersaglio salendo a 0,463 e la B scende sotto il suo a 0,144; nei passi successivi A, B e C continuano a passare sopra e sotto il proprio bersaglio oscillando sempre meno, mentre D, che nessuno linka, è già arrivata al suo dopo un passo solo e non si muove più. Le quattro si assestano sulla distribuzione stazionaria 0,380, 0,199, 0,384, 0,038, marcata da una riga tratteggiata sopra ciascuna colonna."
:width: 92%

La distribuzione dimentica da dove è partita. Le quattro colonne sono quattro
pagine del web, alte quanto la probabilità di trovarsi lì; partendo da «tutte
uguali», tre di esse non salgono dritte verso la propria riga tratteggiata: la
scavalcano, ci passano sotto, e ogni volta di meno. La quarta, che nessuno
linka, ci arriva in un passo solo e poi non si muove più.
```

`````{tab} Elementare

Applicare la regola una volta cambia la distribuzione. Applicarla di nuovo la
cambia ancora, ma di meno. Andando avanti si arriva quasi sempre a una
distribuzione che la regola **lascia com'è**: applicandola, non succede più
niente. Quella si chiama distribuzione **stazionaria**, e
{numref}`fig-catena-si-assesta` la mostra come la riga tratteggiata verso cui
le colonne si assestano.

Va sottolineata una cosa che il disegno rende evidente e che a parole si perde:
il sistema continua a muoversi. I singoli giocatori continuano a cambiare
casella a ogni turno; è la **fotografia d'insieme** che smette di cambiare,
perché quanti entrano in una casella e quanti ne escono si pareggiano. Nessuno
sta fermo, e nondimeno il quadro è immobile.

Il punto notevole è che la distribuzione stazionaria si può trovare
risolvendo, invece che aspettando. Cercare una distribuzione che la regola lascia
identica a sé stessa è esattamente cercare una direzione che la matrice non
devia, cioè un autovettore, con fattore di allungamento uguale a uno. Le
venature del legno viste parlando di autovettori sono la stessa cosa vista da
lontano: qui la venatura che conta è quella che non si allunga né si accorcia,
e il sistema ci scivola sopra e ci resta.

`````

`````{tab} Superiore

Una distribuzione $\boldsymbol{\pi}\in\mathbb{R}^n$, con $\pi_i\ge 0$ e
$\sum_i\pi_i=1$, si dice **stazionaria** se

$$
\mathbf{P}\,\boldsymbol{\pi} = \boldsymbol{\pi} ,
$$

cioè se è un autovettore di $\mathbf{P}$ relativo all'autovalore $1$. Che
l'autovalore $1$ esista sempre si vede subito: le colonne di $\mathbf{P}$
sommano a $1$, quindi $\mathbf{1}^\top\mathbf{P} = \mathbf{1}^\top$, e
$\mathbf{1}$ è autovettore sinistro con autovalore $1$; poiché $\mathbf{P}$ e
$\mathbf{P}^\top$ hanno lo stesso spettro, esiste anche l'autovettore destro.
Che quell'autovettore si possa poi scegliere a componenti non negative, e
quindi normalizzare a una distribuzione, da questo conto non discende: è la
parte del teorema di Perron-Frobenius che vale per ogni matrice a entrate non
negative, e senza di essa l'autovettore potrebbe avere segni misti.

Vale inoltre $|\lambda|\le 1$ per ogni autovalore, e il motivo va enunciato sul
vettore giusto, perché gli autovettori non sono distribuzioni: una matrice
stocastica non allunga la norma $\ell_1$ di **nessun** vettore, dato che
$\sum_i \big|\sum_j P_{ij} x_j\big| \le \sum_j |x_j| \sum_i P_{ij} = \sum_j |x_j|$,
e un autovettore con $|\lambda|>1$ la violerebbe. Lo spettro sta quindi nel
disco unitario, con almeno un punto sul bordo.

Il calcolo si può fare in tre modi, tutti usati:

- **decomposizione spettrale**, prendendo l'autovettore di autovalore $1$ e
  normalizzandolo a somma uno;
- **sistema lineare** $(\mathbf{P}-\mathbf{I})\boldsymbol{\pi}=\mathbf{0}$ con
  il vincolo $\mathbf{1}^\top\boldsymbol{\pi}=1$, che è un sistema
  indeterminato reso determinato dalla normalizzazione, esattamente nella forma
  della {doc}`sezione sui sistemi lineari </Matematica/sistemi-lineari>`;
- **iterazione della potenza**, applicando $\mathbf{P}$ ripetutamente a una
  distribuzione qualsiasi. È il metodo che si usa sulle matrici enormi, dove
  fattorizzare è fuori discussione e moltiplicare per una matrice sparsa costa
  poco.

L'equilibrio è **dinamico**: la catena continua a saltare da uno stato
all'altro, ed è il flusso complessivo a pareggiarsi. Una condizione più forte,
il **bilancio dettagliato** $P_{ij}\pi_j = P_{ji}\pi_i$, chiede che si pareggi
ogni singola coppia di stati; non è necessaria perché $\boldsymbol{\pi}$ sia
stazionaria, ma è la proprietà su cui si costruiscono i campionatori usati nei
{doc}`modelli a energia </ModelliEnergia/overview>`.

`````

## Quando la meta è una sola

L'esistenza di una distribuzione stazionaria è garantita sempre. Che ce ne sia
una sola chiede una condizione; che la catena ci arrivi davvero da qualunque
partenza ne chiede due, e si capiscono meglio da come si rompono.

`````{tab} Elementare

**Prima condizione: da ogni stato si deve poter arrivare a ogni altro.** Se il
tabellone è fatto di due zone senza passaggi fra loro, chi parte nella prima ci
resta per sempre, e le distribuzioni di equilibrio sono almeno due: una per
zona. Chiedere «dove finisce il sistema» non ha una risposta sola, perché
dipende da dove è partito. È il caso di un web fatto di due gruppi di pagine
che non si citano a vicenda, ed è più comune di quanto si creda.

**Seconda condizione: non deve esserci un ritmo fisso.** Immagina due stanze
collegate da una porta girevole che a ogni passo ti obbliga a cambiare stanza.
Partendo dalla prima sarai nella seconda a ogni turno dispari e nella prima a
ogni turno pari, per sempre: la distribuzione oscilla fra due valori e non si
assesta mai, anche se la media sul lungo periodo è metà e metà. Basta una sola
possibilità di restare fermi, o un giro che torni al punto di partenza in un
numero dispari di passi, per rompere il ritmo e far convergere tutto.

Con tutte e due le condizioni la garanzia è forte, e va enunciata con
precisione perché è quella su cui si conta: esiste **una sola** distribuzione
di equilibrio, e la catena ci arriva **da qualunque punto di partenza**. La
memoria dell'inizio si consuma, e quello che resta dipende soltanto dalla
regola. È il motivo per cui un mazzo mescolato abbastanza a lungo non ricorda
più com'era ordinato all'inizio.

`````

`````{tab} Superiore

Una catena è **irriducibile** se per ogni coppia $(i,j)$ esiste $t$ con
$(\mathbf{P}^t)_{ij}>0$ (il grafo delle transizioni è fortemente connesso), ed
è **aperiodica** se il massimo comun divisore dei tempi a cui un ritorno è
possibile, cioè degli $t\ge 1$ con $(\mathbf{P}^t)_{ii}>0$, vale $1$.

**Teorema di Perron-Frobenius**, nella forma che serve qui: se $\mathbf{P}$ è
stocastica, irriducibile e aperiodica, allora l'autovalore $1$ è semplice,
tutti gli altri hanno modulo strettamente minore di $1$, l'autovettore
associato si può scegliere a componenti tutte positive, ed è unico a meno della
normalizzazione. Di conseguenza

$$
\lim_{t\to\infty}\mathbf{P}^t\mathbf{x}_0 = \boldsymbol{\pi}
\qquad\text{per ogni distribuzione iniziale } \mathbf{x}_0 .
$$

I due controesempi mostrano che nessuna delle due ipotesi è di comodo. Con
$\mathbf{P}=\mathbf{I}$ (riducibile all'estremo) ogni distribuzione è
stazionaria e non c'è unicità. Con
$\mathbf{P}=\begin{pmatrix}0&1\\1&0\end{pmatrix}$ (irriducibile ma periodica di
periodo $2$) lo spettro è $\{1,-1\}$: la stazionaria $(\tfrac12,\tfrac12)$
esiste ed è unica, ma $\mathbf{P}^t\mathbf{x}_0$ oscilla e non converge, perché
l'autovalore $-1$ sta anch'esso sul bordo del disco.

Un'ipotesi più debole basta per l'esistenza e l'unicità senza la convergenza:
l'irriducibilità da sola garantisce una stazionaria unica e strettamente
positiva. L'aperiodicità serve solo a far convergere le distribuzioni, ed è
questa la ragione per cui gli algoritmi che sfruttano le catene aggiungono
quasi sempre una probabilità di restare fermi.

`````

## Quanto ci si mette: il secondo autovalore

Sapere che si arriva non basta, perché in pratica si applica la regola un
numero finito di volte. La velocità ha un nome preciso ed è il secondo
autovalore.

`````{tab} Elementare

Lo scarto fra la distribuzione di adesso e quella di equilibrio si accorcia a
ogni passo, e a lungo andare lo fa sempre della **stessa frazione**: cento
passi la applicano cento volte, come l'interesse composto al contrario. Fra due
passi vicini il rapporto oscilla, e a essere costante è la media su molti
passi: è quella la velocità della catena.

Quella frazione è un numero che si legge nella matrice, ed è il secondo dei
suoi fattori di allungamento, il più grande dopo l'uno che tiene ferma la
distribuzione di equilibrio. Piccolo vuol dire che si arriva in fretta, vicino
a uno vuol dire che ci si mette moltissimo. E poiché lo scarto si moltiplica
per quella frazione, il numero di passi che serve non cresce in proporzione
alla precisione voluta: ogni cifra decimale in più costa sempre lo stesso
numero di passi, quale che sia la cifra.

Ecco che cosa rende speciale il risultato sul mazzo di carte, ed è l'eccezione
al ritmo regolare appena descritto. Lì il passaggio dal mescolato male al
mescolato bene non è graduale: per quattro o cinque mescolate lo scarto resta
grande, poi crolla in un paio di passaggi, poi non c'è quasi più niente da
guadagnare. Le catene enormi si comportano spesso così, con
una soglia netta, ed è una fortuna pratica: vuol dire che esiste un numero
giusto di passi, e che farne il doppio è spreco.

`````

`````{tab} Superiore

Siano $1 = |\lambda_1| > |\lambda_2| \ge \dots \ge |\lambda_n|$ gli autovalori
di $\mathbf{P}$. Scomponendo $\mathbf{x}_0 = \boldsymbol{\pi} + \mathbf{r}_0$
con $\mathbf{r}_0$ nel complemento dell'autospazio dominante,

$$
\mathbf{P}^t\mathbf{x}_0 - \boldsymbol{\pi}
= \mathbf{P}^t\mathbf{r}_0
= O\!\left(|\lambda_2|^{\,t}\right) ,
$$

cioè lo scarto decade geometricamente con ragione $|\lambda_2|$. La quantità

$$
\text{gap spettrale} = 1 - |\lambda_2|
$$

governa tutto: il **tempo di mescolamento**, definito come il numero di passi
oltre il quale la distanza in variazione totale dalla stazionaria resta sotto
una soglia $\varepsilon$, si comporta come

$$
t_{\text{mix}}(\varepsilon) = O\!\left(
\frac{1}{1-|\lambda_2|}\,\log\frac{1}{\varepsilon}\right) .
$$

Il logaritmo dice che guadagnare una cifra decimale costa sempre lo stesso
numero di passi, e il fattore davanti dice quanti. Attenzione però a come si
misura: $\lambda_2$ può essere **complesso**, e allora il decadimento è
geometrico soltanto in modulo, con una rotazione sovrapposta. Il rapporto fra
due passi consecutivi oscilla, e la stima onesta di $|\lambda_2|$ si prende
come media geometrica su molti passi. Un gap piccolo (una catena
quasi riducibile, con due gruppi di stati collegati da pochi passaggi) rende il
mescolamento lentissimo, ed è la difficoltà tipica dei campionatori usati in
statistica bayesiana.

Il fenomeno del mazzo di carte porta un nome suo, **cutoff**: per certe
famiglie di catene la distanza dalla stazionaria resta vicina al massimo per un
tempo, poi cala a zero in una finestra molto più stretta del tempo stesso. La
stima asintotica $\tfrac{3}{2}\log_2 n$ per il mescolamento a intreccio di $n$
carte dà circa $8{,}5$ per $n=52$, e l'analisi fine dei valori esatti individua
in sette il punto in cui il crollo è avvenuto. Il decadimento geometrico
descritto sopra vale sempre asintoticamente, ma nella finestra del cutoff
descrive male ciò che si osserva, ed è la ragione per cui la sola conoscenza di
$|\lambda_2|$ a volte non basta.

`````

## Il web come catena: PageRank

L'esempio che ha reso queste idee note fuori dalla matematica è il modo in cui
si ordinano i risultati di una ricerca.

`````{tab} Elementare

Immagina qualcuno che naviga a caso: apre una pagina, sceglie a caso uno dei
link che ci trova, ci va, e ricomincia. Dopo moltissimi salti, la frazione di
tempo che passa su ciascuna pagina è la sua importanza. Una pagina è importante
se molte pagine importanti la linkano, che sembra una definizione circolare e
invece è la definizione di una distribuzione stazionaria: si morde la coda
apposta, e il morso ha una soluzione sola.

Il navigatore così com'è si incastra però in due modi, e le due condizioni
viste sopra dicono esattamente come. Può finire su una pagina senza link uscenti
e restarci; e può finire in un gruppo di pagine che si linkano solo fra loro e
non uscirne più, accumulando tutta l'importanza. La toppa è la stessa per
entrambi ed è semplicissima: ogni tanto, con una probabilità piccola, il
navigatore si annoia e salta su una pagina qualsiasi del web, scelta a caso.
Nella formulazione originale succede circa una volta ogni sette clic.

Questo salto fa tre cose insieme, e conviene contarle. Rende il web
attraversabile da qualunque punto (prima condizione), toglie ogni ritmo fisso
(seconda condizione), e per giunta **accelera** la convergenza: più spesso il
navigatore si annoia, prima il conto si stabilizza. Un difetto di
modellazione riparato con una toppa che risulta essere anche la ragione per cui
il calcolo è possibile su miliardi di pagine.

`````

`````{tab} Superiore

Sia $\mathbf{M}$ la matrice del web resa stocastica per colonne
($M_{ij}=1/\ell_j$ se la pagina $j$ linka la $i$, con $\ell_j$ il numero di
link uscenti da $j$). La **matrice di Google** è

$$
\mathbf{G} = d\,\mathbf{M} + \frac{1-d}{n}\,\mathbf{1}\mathbf{1}^\top ,
$$

con $d\approx 0{,}85$ il **fattore di smorzamento** e $n$ il numero di pagine.
Il secondo termine è il teletrasporto uniforme; le pagine senza link uscenti si
trattano a parte, sostituendo la loro colonna nulla con la distribuzione
uniforme.

$\mathbf{G}$ ha tutte le entrate strettamente positive, quindi è irriducibile e
aperiodica, e Perron-Frobenius garantisce che il PageRank
$\mathbf{G}\boldsymbol{\pi}=\boldsymbol{\pi}$ esista, sia unico e abbia
componenti positive. Il risultato che rende la cosa calcolabile su scala reale
è però un altro, ed è una stima sul secondo autovalore:

$$
|\lambda_2(\mathbf{G})| \;\le\; d ,
$$

indipendentemente da come è fatto il grafo dei link. Lo smorzamento, che
sembrava una toppa modellistica, si rivela così una garanzia di convergenza a
velocità nota. Con $d=0{,}85$ ogni iterazione taglia almeno il $15\%$ dello
scarto misurato come somma degli scostamenti (in altre misure il singolo passo
può anche peggiorare, ed è la media su molti passi a rispettare il limite),
quindi bastano poche decine di prodotti matrice-vettore sparsi per arrivare
alla precisione utile, e il costo di ciascuno è proporzionale al numero di
link.

`````

```python
import numpy as np

# quattro pagine, e chi punta a chi
link = {0: [1, 2], 1: [2], 2: [0], 3: [0, 2]}
nomi, n, d = ["A", "B", "C", "D"], 4, 0.85

M = np.zeros((n, n))
for da, verso in link.items():
    for a in verso:
        M[a, da] = 1 / len(verso)        # colonne stocastiche
print(M.sum(axis=0))                     # ogni colonna somma a uno -> [1. 1. 1. 1.]

G = d * M + (1 - d) / n * np.ones((n, n))

# prima strada: l'autovettore di autovalore uno
autoval, autovet = np.linalg.eig(G)
i = np.argmin(np.abs(autoval - 1))
pi = np.real(autovet[:, i]); pi = pi / pi.sum()
print(np.round(pi, 4))                   # -> [0.3797 0.1989 0.3839 0.0375]

# seconda strada: applicare la regola finche' non si muove piu'
x = np.ones(n) / n
for passo in range(1, 41):
    x = G @ x
print(np.round(x, 4), np.abs(x - pi).max() < 1e-9)   # stesso risultato -> True
```

Le due strade danno lo stesso vettore, ed è il controllo che conta: la prima
chiede a NumPy di fattorizzare la matrice, la seconda la applica e basta,
quaranta volte di fila, e non hanno in comune nemmeno una riga di conto. La
pagina $D$ prende il $3{,}75\%$ perché nessuno la linka, e quel poco che ha le
arriva soltanto dal teletrasporto.

Il secondo autovalore si legge nella stessa decomposizione, e dice quanto in
fretta la seconda strada arriva.

```python
print(np.round(autoval, 3))
# -> [ 1.   +0.j    -0.425+0.425j -0.425-0.425j -0.   +0.j   ]

secondo = sorted(np.abs(autoval))[-2]
print(round(secondo, 4), d)              # -> 0.601 0.85, e vale sempre <= d

# lo scarto si riduce davvero di quel fattore a ogni passo?
x, scarti = np.ones(n) / n, []
for passo in range(40):
    x = G @ x
    scarti.append(np.abs(x - pi).max())

print([float(round(scarti[k + 1] / scarti[k], 3))
       for k in range(14, 19)])
# -> [0.425, 0.5, 0.85, 0.723, 0.425]

print(round((scarti[34] / scarti[14]) ** (1 / 20), 4))   # -> 0.601
```

La prima riga smentisce quello che ci si aspettava: il rapporto fra due passi
consecutivi non è costante, oscilla, e con un ritmo che si ripete ogni quattro
passi. La ragione sta nei due autovalori appena stampati, quelli che portano
il pezzo scritto «$+0{,}425\mathrm{j}$»: quel pezzo in più li rende una coppia
che non si limita ad accorciare lo scarto, lo fa anche **ruotare**, di tre
ottavi di giro a ogni passo. Dopo quattro passi ha girato di un giro e mezzo,
cioè punta esattamente al contrario; e siccome uno scarto misurato in valore
assoluto non distingue una direzione dalla sua opposta, il ritmo che si vede è
di quattro. Lo scarto quindi non scivola verso zero, ci gira attorno
stringendo, e chi misurasse il rapporto fra due passi qualsiasi concluderebbe
$0{,}425$ oppure $0{,}85$ a seconda di dove guarda.

Sull'arco di un numero intero di giri la rotazione si chiude e resta solo il
restringimento: la media geometrica su venti passi dà $0{,}601$, cioè
esattamente il modulo del secondo autovalore. La velocità di convergenza è
quella, e si misura sull'inviluppo, mai su un passo solo.

## Dove le catene tornano, nel resto del libro

Sotto nomi diversi, è sempre la stessa struttura.
I {doc}`modelli n-gram </NaturalLanguageProcessing/modelli-ngram>` sono catene
di Markov sulle parole, con lo stato allargato alle ultime $n-1$. I
{doc}`processi decisionali di Markov </ReinforcementLearning/mdp-valore>` del
reinforcement learning sono catene in cui a ogni passo qualcuno sceglie, e
l'equazione di Bellman che vi si risolve è un sistema lineare come quelli
visti. I campionatori dei {doc}`modelli a energia </ModelliEnergia/overview>`
costruiscono una catena apposta perché la sua stazionaria sia la distribuzione
che si vuole campionare, e ne aspettano il mescolamento. E il processo di
andata dei {doc}`modelli di diffusione </ModelliDiffusione/overview>`, che
aggiunge rumore un passo alla volta, è una catena di Markov la cui stazionaria
è il rumore puro: tutta la difficoltà di quei modelli sta nel percorrerla al
contrario.

## In pratica, con NumPy

```python
import numpy as np

P = np.array([[0.9, 0.5],      # colonne stocastiche: P[i, j] = da j vai in i
              [0.1, 0.5]])

# dove si finisce, applicando la regola 50 volte: le due colonne coincidono,
# cioe' la partenza non conta piu'
print(np.round(np.linalg.matrix_power(P, 50), 4))
# -> [[0.8333 0.8333]
#     [0.1667 0.1667]]

# fra gli autovalori ce n'e' sempre uno che vale 1, ma in virgola mobile puo'
# uscire 0.9999999999999998: si cerca il piu' vicino a uno, non l'uguale
w, v = np.linalg.eig(P)
print(np.round(w, 4), np.argmin(np.abs(w - 1)))   # -> [1.  0.4] 0

# la stazionaria come sistema lineare: (P - I) pi = 0 con somma uno
A = np.vstack([P - np.eye(2), np.ones(2)])
b = np.array([0.0, 0.0, 1.0])
print(np.round(np.linalg.lstsq(A, b, rcond=None)[0], 4))   # -> [0.8333 0.1667]
```

Il sistema è sovradeterminato (tre equazioni, due incognite) e si risolve con i
minimi quadrati della sezione sulle proiezioni: la riga di normalizzazione
aggiunta in fondo è ciò che sceglie, fra gli infiniti multipli
dell'autovettore, quello che è davvero una distribuzione.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Una **catena di Markov** è un sistema in cui il prossimo passo dipende solo
  da dove ti trovi adesso, e non da come ci sei arrivato. Quando la regola
  sembra ricordare qualcosa di più, quasi sempre basta **allargare la
  definizione di stato** e il patto torna valido.
- Applicando la regola molte volte la distribuzione si assesta su una
  configurazione che la regola lascia identica: la distribuzione
  **stazionaria**. Il sistema continua a muoversi, è la fotografia d'insieme a
  fermarsi.
- Trovarla vuol dire cercare una direzione che la matrice non devia, cioè un
  **autovettore** con fattore di allungamento uguale a uno.
- La meta è **una sola** se da ogni stato si può raggiungere ogni altro; e ci
  si arriva da ovunque solo se in più non c'è un ritmo fisso. Due stanze e una
  porta girevole che obbliga a cambiare stanza hanno una meta sola, metà e
  metà, e non ci si assestano mai.
- **Quanto ci si mette** lo dice il secondo fattore di allungamento della
  matrice: lo scarto dall'equilibrio si riduce, in media, di quella frazione a
  ogni passo.
  Nel PageRank quel numero è tenuto sotto controllo dal salto casuale del
  navigatore, che serve a far tornare i conti e per giunta accelera.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- $\Pr[S_{t+1}\mid S_t,\dots,S_0]=\Pr[S_{t+1}\mid S_t]$ definisce la catena; con
  $\mathbf{P}$ stocastica **per colonne** l'evoluzione è
  $\mathbf{x}_t=\mathbf{P}^t\mathbf{x}_0$. Un processo con memoria $k$ diventa
  markoviano su $n^k$ stati.
- La stazionaria è l'autovettore di autovalore $1$:
  $\mathbf{P}\boldsymbol{\pi}=\boldsymbol{\pi}$. Esiste sempre (le colonne
  sommano a uno) e tutti gli autovalori stanno nel disco unitario.
- **Perron-Frobenius**: se $\mathbf{P}$ è irriducibile e aperiodica,
  l'autovalore $1$ è semplice, $\boldsymbol{\pi}$ è unica e positiva, e
  $\mathbf{P}^t\mathbf{x}_0\to\boldsymbol{\pi}$ da ogni partenza.
  $\mathbf{P}=\mathbf{I}$ rompe la prima ipotesi,
  $\begin{pmatrix}0&1\\1&0\end{pmatrix}$ la seconda.
- La velocità è il **gap spettrale** $1-|\lambda_2|$: lo scarto decade come
  $|\lambda_2|^t$ e $t_{\text{mix}}(\varepsilon)=O\big(\log(1/\varepsilon)/
  (1-|\lambda_2|)\big)$. Alcune famiglie mostrano **cutoff**, cioè un crollo in
  una finestra stretta, e lì $|\lambda_2|$ da solo descrive male il transitorio.
- **PageRank**: $\mathbf{G}=d\mathbf{M}+\frac{1-d}{n}\mathbf{1}\mathbf{1}^\top$
  è positiva, quindi la stazionaria esiste ed è unica; e
  $|\lambda_2(\mathbf{G})|\le d$ indipendentemente dal grafo, il che rende
  l'iterazione della potenza praticabile su miliardi di pagine.
```
`````

Con le catene il capitolo ha messo insieme le sue due metà: una matrice
applicata per sempre è algebra lineare, e la cosa su cui la si applica è una
distribuzione di probabilità. Resta un attrezzo, e serve a rispondere a una
domanda che finora è rimasta senza risposta: quanto vale, in numero, una cosa
che non sapevamo e adesso sappiamo.
