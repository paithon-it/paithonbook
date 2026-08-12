# Reti neurali: dal neurone al percettrone multistrato

C'è un'immagine che accompagna le reti neurali fin dal loro battesimo: quella
del cervello. Miliardi di cellule che si scambiano segnali, e da quel
brulichio emergono la memoria, il linguaggio, il riconoscere il volto di un
amico in mezzo alla folla. L'idea, vecchia quasi quanto i computer, è
seducente: se il cervello *è* fatto di neuroni, forse per costruire una
macchina che impara basta costruire neuroni artificiali e collegarli fra loro.
Da questa intuizione nasce tutto il deep learning. Ma conviene sgombrare
subito il campo da un equivoco.

## Un'ispirazione, non una copia

Il neurone biologico riceve segnali dalle sue diramazioni d'ingresso, i
**dendriti**, li somma nel corpo cellulare e, se il totale supera una soglia,
"spara" un impulso lungo il proprio cavo d'uscita, l'assone, fino ai punti di
contatto con gli altri neuroni. È da questa immagine che nasce il neurone
artificiale. Attenzione, però: è una metafora, non una fotografia. Un neurone
reale è una cellula viva,
governata da una chimica di una complessità che non riproduciamo. Il neurone
artificiale ne prende in prestito una sola idea (*sommare tanti segnali e
decidere*) e la trasforma in aritmetica.

`````{tab} Elementare

Immagina una giuria che deve dire "sì" o "no". Ogni giurato porta
un'opinione, e non tutte pesano uguale: quella dell'esperto conta più di
quella del distratto in fondo alla sala. Il presidente somma i voti, ciascuno
moltiplicato per quanto ci fidiamo di chi lo esprime. Se il totale supera una
certa soglia, il verdetto è "sì"; altrimenti "no". Un neurone artificiale è
esattamente questo: prende dei numeri in ingresso, li pesa, li somma e decide.

`````

`````{tab} Superiore

Un neurone artificiale calcola una combinazione lineare degli ingressi seguita
da una funzione non lineare. Dato un vettore di input $\mathbf{x}\in\mathbb{R}^n$:

$$
z = \mathbf{w}^\top \mathbf{x} + b, \qquad a = \sigma(z).
$$

Qui $\mathbf{w}$ è il vettore dei **pesi** (una "importanza" per ciascun
ingresso), $b$ è il **bias** (che sposta la soglia), $z$ è la
**pre-attivazione** (la somma pesata) e $\sigma$ è la **funzione di
attivazione** che introduce la non linearità: un gradino, una sigmoide o,
oggi, quasi sempre la ReLU $\max(0,z)$. Il cuore del calcolo,
$\mathbf{w}^\top\mathbf{x}$, è il prodotto scalare che abbiamo incontrato nel
capitolo di algebra lineare.

`````

## 1943–1958: il neurone formale e il percettrone

La storia comincia in piena Seconda guerra mondiale. Nel 1943 il neurofisiologo
Warren McCulloch e il logico Walter Pitts pubblicano *A Logical Calculus of the
Ideas Immanent in Nervous Activity*: dimostrano che una rete di neuroni
"tutto-o-niente" può calcolare qualunque **funzione logica**, cioè qualunque
regola che, ricevuti in ingresso dei sì e dei no, risponda sì o no (l'*e*,
l'*o*, il *non*, e tutto ciò che si ottiene combinandoli). Era una macchina di
carta, senza apprendimento: i pesi li fissava a mano il progettista.

Il salto arriva nel 1958 con Frank Rosenblatt e il suo **percettrone**. Non
solo un neurone che decide, ma un neurone che *impara*: aggiusta i propri pesi
guardando gli errori che commette. La stampa dell'epoca si entusiasma fino
all'iperbole: il *New York Times* scrisse di una macchina che un giorno
avrebbe "camminato, parlato e avuto coscienza di sé".

`````{tab} Elementare

Il percettrone impara come un allievo con un maestro severo. Riceve un
esempio, prova a rispondere, e il maestro gli dice se ha sbagliato. Se la
risposta era troppo bassa, il percettrone alza i pesi degli ingressi che
spingevano verso il "sì"; se era troppo alta, li abbassa. Un esempio dopo
l'altro, i pesi si assestano finché gli errori diventano rari. Nessuno gli ha
scritto la regola: l'ha ricavata dai dati.

`````

`````{tab} Superiore

Il percettrone produce una decisione binaria

$$
\hat{y} = g(\mathbf{w}^\top \mathbf{x} + b),
$$

dove $g$ è la funzione a gradino ($g(z)=1$ se $z\ge 0$, $0$ altrimenti), e
aggiorna i parametri con la **regola di apprendimento del percettrone**, per
ogni esempio $(\mathbf{x}, y)$ classificato male:

$$
\mathbf{w} \leftarrow \mathbf{w} + \eta\,(y - \hat{y})\,\mathbf{x},
\qquad
b \leftarrow b + \eta\,(y - \hat{y}),
$$

dove $y\in\{0,1\}$ è l'etichetta corretta, $\hat{y}$ la predizione ed
$\eta>0$ il **tasso di apprendimento**. Rosenblatt dimostrò il *teorema di
convergenza*: se i dati sono linearmente separabili, questa regola trova in un
numero finito di passi un iperpiano che li separa. Il guaio è tutto in quel
"se".

`````

## L'inverno: lo scandalo dello XOR (1969)

Nel 1969 Marvin Minsky e Seymour Papert pubblicano *Perceptrons*
{cite}`minsky1969perceptrons`, un libro di matematica rigorosa che è passato
alla storia per una cosa che non contiene. Che un neurone solo sappia tracciare
una sola linea di separazione, e che quindi certe funzioni elementari gli
sfuggano, si sapeva già: è un'osservazione di poche righe, e i due autori la
danno per nota. Il controesempio simbolo, quello che tutti ricordano, è lo
**XOR**, l'"o esclusivo". I loro teoremi veri sono un'altra cosa, più forte, e
li vediamo nella sezione dedicata al percettrone.

`````{tab} Elementare

Lo XOR risponde "vero" quando i due ingressi sono diversi (uno acceso e uno
spento) e "falso" quando sono uguali. Disegna i quattro casi su un foglio a
quadretti: il primo ingresso in orizzontale, il secondo in verticale, e i
quattro casi diventano i quattro angoli di un quadrato. I due "veri" finiscono
su angoli opposti, i due "falsi" sugli altri due. Ora prova a separare i veri
dai falsi con **una sola retta**: è impossibile, qualunque riga tu tracci ne
lascia sempre uno dalla parte sbagliata.

E perché mai un neurone dovrebbe essere legato a una retta? Perché fa una cosa
sola: moltiplica ogni ingresso per il suo peso, somma e confronta il totale con
una soglia. Su quel foglio a quadretti i punti in cui il totale raggiunge la
soglia stanno tutti allineati, e sono proprio una riga dritta: da una parte il
neurone dice sì, dall'altra no. Cambiare i pesi inclina quella riga, cambiare
il bias la sposta, ma una riga resta. Un percettrone singolo sa disegnare solo
quello, e quindi lo XOR non lo imparerà mai.

`````

`````{tab} Superiore

Le quattro coppie sono $(0,0)\to 0$, $(0,1)\to 1$, $(1,0)\to 1$, $(1,1)\to 0$.
Un percettrone realizza un separatore lineare $\mathbf{w}^\top\mathbf{x}+b=0$,
cioè un iperpiano; può risolvere solo problemi **linearmente separabili**. Lo
XOR non lo è: non esistono $\mathbf{w}$ e $b$ tali che $w_1 x_1 + w_2 x_2 + b$
risulti positivo esattamente sui due punti con etichetta $1$ e non positivo
sugli altri due. La soluzione (impilare più neuroni in **strati**) era nota già
allora, ma mancava un modo efficiente per addestrarla, ed è questa la ragione
che gli stessi Minsky e Papert indicheranno per la lunga pausa che seguì. Il
libro contribuì a raffreddare gli entusiasmi e a spostare risorse verso l'AI
simbolica; nel decennio successivo, con il rapporto Lighthill del 1973, i
tagli colpirono l'intero campo. È il primo "inverno" delle reti neurali, e la
storiografia recente invita a non attribuirlo a un libro solo
{cite}`olazaran1996sociological`.

`````

## La rinascita: la backpropagation (1986)

La chiave era là da vedere: se un neurone solo non basta, se ne mettono di
più, in **strati**. Ma come si addestrano i neuroni intermedi, che non hanno
un'etichetta che dica loro "la risposta giusta era questa"? La risposta è la
**backpropagation** (retropropagazione dell'errore), resa celebre nel 1986 da
David Rumelhart, Geoffrey Hinton e Ronald Williams su *Nature*. L'algoritmo
aveva precursori (Paul Werbos lo aveva formulato nella sua tesi del 1974) ma è
quel lavoro a farne lo standard.

`````{tab} Elementare

Immagina una catena di montaggio dove il prodotto finale esce difettoso.
Backpropagation è il modo di distribuire la colpa all'indietro: parte dal
difetto finale e risale la catena, assegnando a ogni stazione una quota di
responsabilità. Chi ha contribuito di più all'errore riceve la correzione più
grande. Ripetuto su migliaia di esempi, questo "attribuire la colpa e
correggere" fa sì che anche gli operai in mezzo alla catena (i neuroni
nascosti) imparino il loro mestiere.

`````

`````{tab} Superiore

Si definisce una funzione di **loss** $\mathcal{L}(\hat{\mathbf{y}},
\mathbf{y})$ derivabile e si aggiornano tutti i parametri $\theta$ (pesi e
bias di ogni strato) scendendo lungo il gradiente:

$$
\theta \leftarrow \theta - \eta\,\nabla_\theta \mathcal{L}.
$$

Il gradiente rispetto ai pesi degli strati profondi si calcola applicando la
**regola della catena** strato per strato, propagando l'errore dall'uscita
verso l'ingresso. È efficiente perché riusa i calcoli condivisi: il costo di
una passata all'indietro è dello stesso ordine di una in avanti.

`````

## L'anatomia di un percettrone multistrato

Il modello che nasce da questa storia è il **percettrone multistrato** (MLP,
*multilayer perceptron*): neuroni organizzati in strati successivi, dove
l'uscita di uno strato è l'ingresso del prossimo
({numref}`fig-percettrone-multistrato`).

```{figure} ../figures/percettrone-multistrato.svg
:name: fig-percettrone-multistrato
:alt: Diagramma di un percettrone multistrato con tre nodi di input, quattro nodi nello strato nascosto e due nodi di output, tutti collegati tra strati adiacenti.
:width: 85%

Un percettrone multistrato: lo strato di input (3 nodi) passa i dati a uno
strato nascosto (4 nodi), che a sua volta alimenta lo strato di output
(2 nodi). Ogni connessione porta un peso; l'informazione fluisce da sinistra
a destra.
```

Quanti neuroni mettere in mezzo, e quanti strati, non lo dice nessuna formula:
il 3-4-2 della figura è un esempio, e nella pratica quelle misure si scelgono
provando. Ciò che cambia davvero, aggiungendo strati, è la **forma** delle
frontiere che la rete sa tracciare ({numref}`fig-confini-multistrato`).

```{figure} ../figures/reti-multistrato.svg
:name: fig-confini-multistrato
:alt: "Tre pannelli sulla stessa nube di quattro punti, due pieni e due vuoti. Con un solo neurone il confine di decisione è una retta. Con uno strato nascosto diventa una regione triangolare, ottenuta combinando tre rette. Con più strati le regioni si compongono in una figura chiusa a venti lati, che da lontano sembra una curva ma conserva gli spigoli."
:width: 100%

Gli stessi quattro punti, tre confini. Un neurone solo li separa con una retta;
uno strato nascosto mette insieme più rette e ritaglia una regione chiusa;
aggiungendo strati le regioni si combinano fra loro e il contorno segue la
forma dei dati sempre più da vicino. Attenzione a cosa cambia davvero: non è
che il terzo confine sia irraggiungibile per gli altri due (con abbastanza
neuroni in un solo strato ci si arriva lo stesso), è che con più strati lo si
ottiene con molti meno neuroni.
```

La progressione di {numref}`fig-confini-multistrato` è la risposta visiva allo
scandalo dello XOR raccontato poco sopra. I quattro punti disegnati qui non
sono lo XOR, anzi: sono un problema che una retta risolve, mostrato per
confronto. Il problema del percettrone non era che imparasse male, era che una
retta sola non può separare **i quattro casi dello XOR**, per quanto bene la si
posizioni. Serviva un secondo strato, non un addestramento migliore.

`````{tab} Elementare

Tre strati, tre ruoli. Lo **strato di input** non calcola nulla: è la porta da
cui entrano i dati (i tre numeri che descrivono l'esempio). Gli **strati
nascosti** sono la fabbrica: ogni neurone combina ciò che riceve e passa avanti
qualcosa di un po' più grosso di quello che ha ricevuto. In una rete che guarda
fotografie, per dire, i neuroni del primo strato reagiscono a cose minime, un
bordo chiaro-scuro, una macchia di colore; quelli del secondo mettono insieme
quei bordi e reagiscono a un angolo, a un cerchietto; più avanti si arriva a un
occhio, a un muso, a una faccia intera. Nessuno glielo ha insegnato: viene
fuori così dall'addestramento. Lo **strato di output** tira le somme e
produce la risposta: qui due numeri, per esempio quanto la rete è convinta di
ciascuna delle due risposte possibili ("gatto" o "cane"). "Nascosto" vuol dire
solo che non lo vediamo né in ingresso né in uscita: lavora in mezzo.

`````

`````{tab} Superiore

Con un solo strato nascosto, l'MLP calcola

$$
\mathbf{h} = \sigma\!\left(W^{(1)}\mathbf{x} + \mathbf{b}^{(1)}\right),
\qquad
\hat{\mathbf{y}} = \varphi\!\left(W^{(2)}\mathbf{h} + \mathbf{b}^{(2)}\right).
$$

Qui $W^{(1)}\in\mathbb{R}^{4\times 3}$ e $W^{(2)}\in\mathbb{R}^{2\times 4}$ sono
le matrici dei pesi, $\mathbf{b}^{(1)}, \mathbf{b}^{(2)}$ i bias, $\sigma$ la
non linearità nascosta (tipicamente ReLU) e $\varphi$ l'attivazione d'uscita
(per esempio softmax). È l'impilamento di trasformazioni lineari e non lineari
a dare la potenza: il *teorema di approssimazione universale* garantisce che
una rete con un solo strato nascosto abbastanza ampio può approssimare, con
errore arbitrariamente piccolo, qualunque funzione continua su un insieme
compatto. Dimostrato prima per attivazioni limitate, come la sigmoide
({cite}`cybenko1989approximation`; {cite}`hornik1991approximation`), vale per
ogni $\sigma$ non polinomiale, ReLU compresa ({cite}`leshno1993multilayer`).
È però un teorema di esistenza: dice che i pesi giusti ci sono, non che la
discesa del gradiente li trovi. La non linearità $\sigma$ è essenziale: senza
di essa, due strati lineari collasserebbero in uno solo.

Il teorema chiarisce anche come leggere {numref}`fig-confini-multistrato`: la
profondità non sblocca confini altrimenti irraggiungibili, riduce il numero di
neuroni che servono per ottenerli. Il terzo pannello è disegnato con gli
spigoli per una ragione: con la ReLU la rete è lineare a tratti e il confine
resta un poligono, con sempre più lati man mano che gli strati si accumulano,
finché da lontano sembra una curva. Curvo alla lettera lo è solo con
attivazioni derivabili come la sigmoide o la tangente iperbolica.

`````

## Come è organizzato questo capitolo

Da qui in avanti smontiamo l'MLP pezzo per pezzo, in tre sezioni. La prima
torna sul **percettrone**, il neurone singolo, e sulla regola con cui impara: è
il mattone, e il suo limite è la ragione di tutto il resto. La seconda sono le
**funzioni di attivazione**, il piccolo gesto non lineare senza cui la
profondità non servirebbe a niente. La terza è la **backpropagation**, cioè il
meccanismo con cui l'errore risale la rete e corregge ogni peso, e con essa la
**discesa del gradiente**, il modo in cui quelle correzioni si fanno in
pratica. Chi ha le derivate nello zaino ci riconoscerà la regola della catena
applicata con ordine, ma la parte principale è raccontata anche senza. Restano
fuori due scelte pratiche che meritano una sezione a sé, e l'avranno nel
capitolo sul deep learning: da dove far partire i pesi, e come tenere a bada una
rete che impara troppo bene i dati che ha visto. È la cerniera del libro: tutto
ciò che segue (visione artificiale, NLP, modelli generativi) sono percettroni
multistrato cresciuti, specializzati e resi profondi.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Il neurone artificiale prende dal neurone vero *una sola idea*: sommare
  segnali con un'importanza diversa ciascuno, e decidere. Il resto è metafora,
  non copia.
- McCulloch e Pitts, nel 1943, scrivono il neurone come una regoletta di
  calcolo; Rosenblatt, nel 1958, lo fa *imparare*: gli mostri esempi e lui
  aggiusta da solo le importanze.
- Un neurone da solo sa dividere i casi con **una riga dritta**. Sullo
  **XOR** quella riga non esiste, e nel 1969 il libro di Minsky e Papert lo
  ricorda a tutti nel momento peggiore: seguono anni di disinteresse e di
  fondi tagliati.
- La **backpropagation** (1986) risolve la domanda che teneva ferma la
  faccenda: come si correggono i neuroni in mezzo, quelli a cui nessuno dice
  quale fosse la risposta giusta.
- Una rete è una pila: i dati entrano, attraversano uno o più strati che li
  rimescolano, e dall'ultimo esce la risposta. Fra uno strato e l'altro ci
  vuole sempre un passaggio che non sia una semplice proporzione.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Il neurone artificiale prende dal biologico *una sola idea*: sommare
  ingressi pesati e decidere. Il resto è metafora, non copia.
- McCulloch e Pitts {cite}`mcculloch1943logical` formalizzano il neurone;
  Rosenblatt {cite}`rosenblatt1958perceptron` lo fa *imparare* con il
  percettrone.
- Un percettrone singolo è un **classificatore lineare** e non risolve lo
  **XOR**. *Perceptrons* {cite}`minsky1969perceptrons` non dimostra questo (era
  noto): dimostra che nel suo modello il costo di certi predicati cresce con la
  taglia dell'ingresso, e sul multistrato avanza una congettura, non un
  teorema.
- La **backpropagation** (Rumelhart, Hinton, Williams, 1986) rende
  addestrabili gli strati nascosti: è la rinascita del campo.
- Un **MLP** impila input → strati nascosti → output, alternando
  trasformazioni lineari e non linearità. Il teorema di approssimazione
  universale garantisce l'esistenza dei pesi giusti per ogni attivazione **non
  polinomiale**, non che la discesa del gradiente li trovi.
```

`````
