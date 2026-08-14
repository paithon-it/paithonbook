# Reti neurali: dal neurone al percettrone multistrato

C'è un'immagine che accompagna le reti neurali fin dal loro battesimo: quella
del cervello. Miliardi di cellule che si scambiano segnali, e da quel
brulichio emergono la memoria, il linguaggio, il riconoscere il volto di un
amico in mezzo alla folla. L'idea, vecchia quasi quanto i computer, è
seducente: se il cervello *è* fatto di neuroni, forse per costruire una
macchina che impara basta costruire neuroni artificiali e collegarli fra loro.
Da questa intuizione nasce tutto il **deep learning**, che è poi il nome che si
dà all'apprendimento automatico quando i neuroni artificiali si impilano in
molte file: «profondo» vuol dire questo, e nient'altro. Ma conviene sgombrare
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

Due nomi, perché torneranno a ogni pagina. Quel "quanto ci fidiamo" di
ciascun giurato è un numero, e si chiama **peso**. E il presidente ha una sua
inclinazione ancora prima di sentire la giuria, chi parte prevenuto verso il
sì e chi verso il no: anche quella è un numero, e si chiama **bias**.

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
guardando gli errori che commette. La stampa dell'epoca si entusiasma oltre
ogni misura: il *New York Times* scrisse di una macchina che un giorno
avrebbe "camminato, parlato e avuto coscienza di sé".

`````{tab} Elementare

Il percettrone impara come un allievo con un maestro severo. Riceve un
esempio, prova a rispondere sì o no, e il maestro gli dice se ha sbagliato. Se
doveva dire sì e ha detto no, alza i pesi degli indizi che in quell'esempio
erano accesi, così la prossima volta il totale verrà più alto e supererà la
soglia; se doveva dire no e ha detto sì, li abbassa. Un esempio dopo l'altro,
i pesi si assestano finché gli errori diventano rari. Nessuno gli ha scritto la
regola: l'ha ricavata dai dati.

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
$\eta>0$ il **tasso di apprendimento**. C'è poi il *teorema di convergenza*: se
i dati sono linearmente separabili, questa regola trova in un numero finito di
passi un iperpiano che li separa. Non è nell'articolo del 1958, che presenta il
modello: la dimostrazione arriva con *Principles of Neurodynamics* del 1962
{cite}`rosenblatt1962principles`, e le forme rigorose che si citano oggi si
devono a Novikoff {cite}`novikoff1962convergence` e a Block. Il guaio è tutto in
quel "se".

`````

## L'inverno: lo scandalo dello XOR (1969)

Nel 1969 Marvin Minsky e Seymour Papert pubblicano *Perceptrons*
{cite}`minsky1969perceptrons`, un libro di matematica rigorosa, passato alla
storia per una cosa che non contiene.

Il fatto è questo. Un neurone solo, dovendo dividere i casi in due gruppi, sa
tracciare una riga dritta e nient'altro. Esistono allora regole semplicissime
che gli restano precluse, perché per separarne i casi una riga non basta.
Nel 1969 questo si sapeva benissimo: è un'osservazione di poche righe, e i due
autori la danno per nota. L'esempio che tutti ricordano è lo **XOR**, l'"o
esclusivo". I teoremi veri del libro sono un'altra cosa, più forte, e li
vediamo nella sezione dedicata al percettrone.

E qui il titolo di questa sezione. Al 1969 seguirono anni in cui i soldi per
l'intelligenza artificiale si ritirarono e molti gruppi di ricerca chiusero:
sono l'**inverno dell'AI**, e sono la ragione per cui fra il libro di Minsky e
Papert e la ripresa passano quasi vent'anni. Quanta parte di colpa tocchi
davvero a quel libro è una questione aperta, e la riprendiamo nella sezione sul
percettrone: meno di quanta gliene attribuisca il racconto corrente.

`````{tab} Elementare

Lo XOR risponde "vero" quando i due ingressi sono diversi (uno acceso e uno
spento) e "falso" quando sono uguali. Scriviamo acceso come $1$ e spento come
$0$: le combinazioni possibili sono quattro, e sono $(0,0)$, $(0,1)$, $(1,0)$ e
$(1,1)$. Le prime due cifre le mettiamo in orizzontale su un foglio a
quadretti, le seconde in verticale, e i quattro casi diventano i quattro angoli
di un quadrato. I due "veri", $(0,1)$ e $(1,0)$, finiscono su angoli opposti; i
due "falsi", $(0,0)$ e $(1,1)$, sugli altri due, di nuovo opposti fra loro. Ora
prova a separare i veri dai falsi con **una sola retta**: è impossibile,
qualunque riga tu tracci ne lascia sempre uno dalla parte sbagliata.

E perché mai un neurone dovrebbe essere legato a una retta? Perché fa una cosa
sola: moltiplica ogni ingresso per il suo peso, somma e confronta il totale con
una soglia. Su quel foglio a quadretti i punti in cui il totale pareggia la
soglia stanno tutti allineati, e formano proprio una riga dritta: da una parte
il neurone dice sì, dall'altra no. Cambiare i pesi inclina quella riga,
cambiare il bias la sposta, ma una riga resta. Un percettrone singolo sa
disegnare solo quello, e quindi lo XOR non lo imparerà mai. (Che quei punti
siano allineati non è una cosa da bersi sulla fiducia: nella prossima sezione
li calcoliamo, due punti veri con due numeri veri, e la riga si vede.)

`````

`````{tab} Superiore

Le quattro coppie sono $(0,0)\to 0$, $(0,1)\to 1$, $(1,0)\to 1$, $(1,1)\to 0$.
Un percettrone realizza un separatore lineare $\mathbf{w}^\top\mathbf{x}+b=0$,
cioè un iperpiano; può risolvere solo problemi **linearmente separabili**. Lo
XOR non lo è: poiché il gradino risponde $1$ quando $z\ge 0$, servirebbero
$\mathbf{w}$ e $b$ tali che $w_1 x_1 + w_2 x_2 + b$ risulti **non negativo** sui
due punti con etichetta $1$ e **negativo** sugli altri due, e non esistono. La
soluzione (impilare più neuroni in **strati**) era nota già
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
più, in **strati**. Resta però una domanda: che risposta dovrebbero dare i
neuroni in mezzo? Per quelli in fondo lo sappiamo, perché ogni esempio di
addestramento si porta dietro la risposta giusta (si chiama **etichetta**:
«questa foto è un gatto»). Per quelli in mezzo non l'ha scritta nessuno. La
risposta è la
**backpropagation** (retropropagazione dell'errore), resa celebre nel 1986 da
David Rumelhart, Geoffrey Hinton e Ronald Williams su *Nature*. L'algoritmo
aveva precursori (Paul Werbos lo aveva formulato nella tesi di dottorato del
1974) ma è quel lavoro a farne lo standard.

`````{tab} Elementare

Immagina una catena di montaggio dove il prodotto finale esce difettoso.
Backpropagation è il modo di distribuire la colpa all'indietro: parte dal
difetto finale e risale la catena, assegnando a ogni stazione una quota di
responsabilità. Chi ha contribuito di più all'errore riceve la correzione più
grande. Ripetuto su migliaia di esempi, questo "attribuire la colpa e
correggere" fa sì che anche gli operai in mezzo alla catena imparino il loro
mestiere. Quegli operai in mezzo hanno un nome: si chiamano neuroni
**nascosti**, e "nascosto" vuol dire soltanto che non si affacciano né
sull'ingresso né sull'uscita.

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
*multilayer perceptron*). I neuroni si mettono in fila, e ogni fila è uno
**strato**: l'uscita di uno strato è l'ingresso del prossimo. Il primo strato è
la porta da cui entrano i dati e si chiama **strato di input**, l'ultimo
produce la risposta e si chiama **strato di output**, e quelli in mezzo sono i
nascosti di poco fa ({numref}`fig-percettrone-multistrato`). Nel disegno ogni
pallino è un neurone e ogni filo un peso.

```{figure} ../figures/percettrone-multistrato.svg
:name: fig-percettrone-multistrato
:alt: Diagramma di un percettrone multistrato con tre nodi di input, quattro nodi nello strato nascosto e due nodi di output, tutti collegati tra strati adiacenti.
:width: 85%

Un percettrone multistrato: lo strato di input (3 neuroni) passa i dati a uno
strato nascosto (4 neuroni), che a sua volta alimenta lo strato di output
(2 neuroni). Ogni collegamento porta un peso; l'informazione va da sinistra
a destra.
```

Quanti neuroni mettere in mezzo, e quanti strati, non lo dice nessuna formula:
il 3-4-2 della figura è un esempio, e nella pratica quelle misure si scelgono
provando. Ciò che cambia davvero, aggiungendo strati, è la **forma** del
confine con cui la rete separa i casi ({numref}`fig-confini-multistrato`).

```{figure} ../figures/reti-multistrato.svg
:name: fig-confini-multistrato
:alt: "Tre riquadri sugli stessi quattro punti, due pieni e due vuoti. Con un solo neurone il confine fra i due gruppi è una riga dritta. Con uno strato nascosto diventa una regione triangolare, ottenuta combinando tre righe. Con più strati le regioni si compongono in una figura chiusa a venti lati, che da lontano sembra una curva ma conserva gli spigoli."
:width: 100%

Gli stessi quattro punti, tre confini. Un neurone solo li separa con una riga
dritta; uno strato nascosto mette insieme più righe e ritaglia una regione
chiusa; aggiungendo strati le regioni si combinano fra loro e il bordo segue
la forma dei dati sempre più da vicino.
```

I quattro punti del disegno non sono lo XOR: sono un caso più facile, che una
riga sola risolve benissimo, e stanno lì per far vedere che cosa si guadagna
mano a mano che gli strati aumentano. Lo XOR risolto arriva nella prossima
sezione. Il senso però è lo stesso: il guaio del percettrone non era che
imparasse male, era che una riga sola non può separare **i quattro casi dello
XOR** per quanto bene la si giri. Serviva un secondo strato, non un
addestramento migliore.

E i due passaggi del disegno non sono lo stesso passaggio, il che è la cosa
meno ovvia della figura.

Dal primo riquadro al secondo si guadagna qualcosa che prima non c'era. Un
neurone solo disegna una riga e nient'altro, quindi un confine chiuso gli è
precluso comunque lo si addestri; due righe che si incrociano ritagliano invece
uno spicchio, tre un triangolo, ed è il triangolo del secondo riquadro.

Dal secondo al terzo, invece, non si guadagna niente di nuovo: si risparmia. Lo
dice un teorema, che nella sua forma esatta sta qui sotto nel livello
Superiore: a un contorno come quello del terzo riquadro ci si arriverebbe anche
con un solo strato nascosto, purché di neuroni ce ne sia un numero abbastanza
grande. Con più strati ne bastano molti meno, ed è il vero motivo per cui le
reti si fanno profonde.

`````{tab} Elementare

Tre strati, tre ruoli. Lo **strato di input** non calcola nulla: è la porta da
cui entrano i dati (i tre numeri che descrivono l'esempio). Gli **strati
nascosti** sono la fabbrica: ogni neurone combina ciò che riceve e passa avanti
qualcosa di più elaborato di quello che ha ricevuto. In una rete che guarda
fotografie, per dire, i neuroni del primo strato reagiscono a cose minime, un
bordo chiaro-scuro, una macchia di colore; quelli del secondo mettono insieme
quei bordi e reagiscono a un angolo, a un cerchietto; più avanti si arriva a un
occhio, a un muso, a una faccia intera. Nessuno glielo ha insegnato: viene
fuori così dall'addestramento. Lo **strato di output** tira le somme e
produce la risposta: qui due numeri, per esempio quanto la rete è convinta di
ciascuna delle due risposte possibili ("gatto" o "cane").

C'è però una condizione, ed è il motivo per cui esiste una delle sezioni che
seguono. Fra uno strato e l'altro deve succedere qualcosa che non sia
soltanto moltiplicare e sommare. Se ogni strato si limitasse a quello, dieci
strati in fila darebbero lo stesso risultato di uno solo: moltiplicare per $2$
e poi per $3$ è come moltiplicare per $6$, e tutta la pila non servirebbe a
niente.

`````

`````{tab} Superiore

Con un solo strato nascosto, l'MLP calcola

$$
\mathbf{h} = \sigma\!\left(\mathbf{W}^{[1]}\mathbf{x} + \mathbf{b}^{[1]}\right),
\qquad
\hat{\mathbf{y}} = \varphi\!\left(\mathbf{W}^{[2]}\mathbf{h} + \mathbf{b}^{[2]}\right).
$$

Qui $\mathbf{W}^{[1]}\in\mathbb{R}^{4\times 3}$ e
$\mathbf{W}^{[2]}\in\mathbb{R}^{2\times 4}$ sono
le matrici dei pesi, $\mathbf{b}^{[1]}, \mathbf{b}^{[2]}$ i bias, $\sigma$ la
non linearità nascosta (tipicamente ReLU) e $\varphi$ l'attivazione d'uscita
(per esempio softmax). L'indice fra parentesi quadre in alto è il numero dello
strato, e resta questo per tutto il libro: la parentesi tonda serve per
distinguere gli **esempi**, come in $\hat{y}^{(i)}$.

È l'impilamento di trasformazioni lineari e non lineari
a dare la potenza: il *teorema di approssimazione universale* garantisce che
una rete con un solo strato nascosto abbastanza ampio può approssimare, con
errore arbitrariamente piccolo, qualunque funzione continua su un insieme
compatto. Dimostrato prima per attivazioni limitate, come la sigmoide
({cite}`cybenko1989approximation`; {cite}`hornik1991approximation`), vale per
ogni $\sigma$ non polinomiale, ReLU compresa ({cite}`leshno1993multilayer`).
È però un teorema di esistenza: dice che i pesi giusti ci sono, non che la
discesa del gradiente li trovi. E c'è una seconda cosa che non dice, altrettanto
importante: **quanto ampio**. Nell'enunciato "abbastanza ampio" non è
quantificato, e per una funzione qualsiasi di $d$ variabili il numero di neuroni
necessari cresce esponenzialmente in $d$: la garanzia c'è, il conto è fuori
portata già per un'immagine piccola. È il vero motivo per cui il teorema
consola meno di quanto suoni. La non linearità $\sigma$ è essenziale: senza
di essa, due strati lineari collasserebbero in uno solo.

Come leggere allora {numref}`fig-confini-multistrato`? Non con il teorema di
approssimazione universale, che sull'efficienza non dice niente: dice solo che
uno strato solo, se ampio a piacere, basta. A dire qualcosa sono altri
risultati, i **teoremi di separazione**, che funzionano per esibizione: mostrano
una funzione che una rete profonda calcola con pochi neuroni e che una rete più
piatta non sa approssimare se non pagando una larghezza esponenziale. Telgarsky
{cite}`telgarsky2016benefits` costruisce funzioni di una variabile, lineari a
tratti (una sega dai molti denti), che una rete profonda disegna con pochi
neuroni per strato e che una rete di pochi strati richiederebbe un numero di
neuroni esponenziale nella profondità risparmiata; Eldan e Shamir
{cite}`eldan2016power` fanno lo stesso in $d$ dimensioni sul salto più piccolo
che ci sia, da due strati nascosti a uno: esiste una funzione che due strati
nascosti realizzano con larghezza polinomiale in $d$ e che uno strato solo non
approssima senza larghezza esponenziale in $d$. Sono funzioni costruite apposta,
non un teorema su tutte le funzioni, ed è già abbastanza: dicono che la
profondità può comprare qualcosa che la larghezza paga carissimo.

Il terzo pannello è disegnato con gli
spigoli per una ragione: con la ReLU la rete è lineare a tratti e il confine
resta un poligono, con sempre più lati man mano che gli strati si accumulano,
finché da lontano sembra una curva. Curvo alla lettera lo è solo con
attivazioni derivabili come la sigmoide o la tangente iperbolica.

`````

## Come è organizzato questo capitolo

Da qui in avanti smontiamo l'MLP pezzo per pezzo, in tre sezioni.

La prima torna sul **percettrone**, il neurone singolo, e sulla regola con cui
impara: è il mattone, e il suo limite è la ragione di tutto il resto. La
seconda sono le **funzioni di attivazione**, cioè proprio quel qualcosa che
deve succedere fra uno strato e l'altro perché impilarne dieci non equivalga a
impilarne uno. La terza è la **backpropagation**, il meccanismo con cui
l'errore risale la rete e dice a ogni peso di quanto muoversi; e con essa la
**discesa del gradiente**, il modo in cui quelle correzioni si fanno davvero,
un passettino alla volta, sempre nella direzione che fa scendere l'errore
(«gradiente» è il nome che prende, tutto insieme, l'elenco di quelle
direzioni). Chi ha le derivate nello zaino ci riconoscerà la
regola della catena applicata con ordine, ma la parte principale è raccontata
anche senza.

Restano fuori due scelte pratiche, che riprende il capitolo sul deep learning:
da dove far partire i pesi, e come impedire a una rete di imparare a memoria
gli esempi che ha visto invece della regola che li governa (quando succede, sui
casi nuovi sbaglia). È la cerniera del libro: tutto ciò che viene dopo (la
visione artificiale, il linguaggio, i modelli che generano immagini e testo)
sono percettroni multistrato cresciuti, specializzati e resi profondi.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Il neurone artificiale prende dal neurone vero *una sola idea*: sommare
  segnali con un'importanza diversa ciascuno, e decidere. Il resto è metafora,
  non copia.
- McCulloch e Pitts, nel 1943, scrivono il neurone come una regoletta di
  calcolo; Rosenblatt, nel 1958, lo fa *imparare*: gli mostri esempi e lui
  aggiusta da solo le importanze.
- Un neurone da solo sa dividere i casi con **una riga dritta**, e sullo
  **XOR** quella riga non esiste: nel 1969 il libro di Minsky e Papert lo
  ricorda a tutti. Agli anni di disinteresse e di fondi tagliati che seguirono
  quel libro contribuì, ma non li decise: a tenere ferme le reti era un
  problema tecnico (nessuno sapeva correggere i neuroni in mezzo), e i tagli
  veri arrivarono qualche anno dopo e colpirono l'intelligenza artificiale
  tutta intera.
- La **backpropagation** (1986) risolve la domanda che teneva ferma la
  faccenda: come si correggono i neuroni in mezzo, quelli a cui nessuno dice
  quale fosse la risposta giusta.
- Una rete è una pila: i dati entrano, attraversano uno o più strati che li
  rimescolano, e dall'ultimo esce la risposta. Fra uno strato e l'altro deve
  succedere qualcosa che non sia moltiplicare e sommare, altrimenti dieci
  strati in fila valgono quanto uno solo.
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
