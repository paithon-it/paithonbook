# Privacy e robustezza: dati protetti e attacchi avversari

Nel 2021 un gruppo di ricercatori guidato da Nicholas Carlini pose a GPT-2 (il
modello linguistico di OpenAI addestrato su un'enorme raccolta di testi del
web) un gran numero di inneschi, e si mise a leggere le risposte. In mezzo al
mare di frasi plausibili ne trovarono alcune che *non erano* plausibili: erano
*vere*. Il modello sputava, parola per parola, il nome completo di una persona
reale, il suo indirizzo, un numero di telefono, un'email: informazioni
comparse una manciata di volte nei dati di addestramento, e da lì
*memorizzate*. Nessuno aveva chiesto al modello di ricordarle: l'aveva fatto
da solo, come effetto collaterale dell'imparare.

Questo episodio apre il secondo dei tre assi annunciati nell'introduzione del
capitolo. Dopo l'equità, affrontiamo insieme **privacy** e **robustezza**: due
facce della stessa domanda scomoda; quanto è discreto, e quanto è fragile, un
modello una volta messo nel mondo. Il filo conduttore è che entrambe le
proprietà non si aggiungono alla fine come una vernice, ma vanno costruite
dentro l'addestramento, con un costo in accuratezza da mettere in conto
onestamente.

## I modelli ricordano più di quanto vorremmo

Un modello di machine learning è, in fondo, una compressione dei suoi dati di
addestramento. E come ogni compressione con perdite, a volte conserva un
dettaglio per intero invece di riassumerlo. Quando quel dettaglio è
un'informazione personale, la memorizzazione diventa una falla di privacy.

```{figure} ../figures/gdpr-e-llm.svg
:name: fig-flusso-dati-personali
:alt: "Il percorso di un dato personale dentro un sistema basato su LLM: raccolta, inserimento nel corpus di addestramento, presenza nei pesi del modello, comparsa nel prompt a uso dell'utente e infine nei log delle conversazioni. A ogni passaggio è indicata la base giuridica che lo dovrebbe giustificare."
:width: 100%

Un dato personale attraversa più stazioni di quante se ne contino a occhio.
Ciascuna ha bisogno di una giustificazione propria: il consenso raccolto per
la prima non copre automaticamente l'ultima.
```

Il punto che {numref}`fig-flusso-dati-personali` rende difficile da aggirare è
la stazione centrale, i pesi. Nelle altre il dato si può cancellare; da lì,
una volta che l'addestramento è finito, non si toglie senza riaddestrare, ed è
il motivo per cui il «diritto alla cancellazione» è tecnicamente scomodo
proprio dove sarebbe più necessario.

`````{tab} Elementare

Immagina uno studente che, invece di capire la materia, impari il libro a
memoria. All'esame, se gli capita una domanda vista in aula, non ragiona:
recita la pagina. Molti modelli fanno qualcosa di simile con gli esempi rari o
ripetuti: non ne colgono la regola, li imparano di sbieco così come sono. Due
guai ne seguono. Il primo: dando al modello l'inizio di una frase che c'era
nei dati, questo può completarla *identica*; se in quei dati c'era il tuo
indirizzo, può ripeterlo. Il secondo, più sottile: anche senza fargli sputare
nulla, si può spesso indovinare *se una certa persona era nei dati* di
addestramento, osservando che il modello è stranamente sicuro proprio sui suoi
esempi. È come capire che uno studente ha già visto un compito perché lo
svolge troppo in fretta e senza esitazioni. Sapere «Tizio era nel dataset
dell'ospedale» può essere di per sé un'informazione sensibile.

`````

`````{tab} Superiore

I due attacchi hanno nomi precisi. Il **membership inference attack**,
formalizzato da Shokri e colleghi (2017), decide se un dato campione $X$
apparteneva o meno all'insieme di addestramento, sfruttando il divario di
comportamento del modello tra ciò che ha visto e ciò che non ha visto:
tipicamente una loss più bassa, o una confidenza più alta, sugli esempi di
training. È l'evidenza empirica dell'*overfitting* discusso nel capitolo di
Machine Learning, qui riletto come vulnerabilità: più un modello si adatta ai
singoli esempi, più li lascia riconoscere. L'**estrazione di dati di
addestramento** è più aggressiva: Carlini e colleghi (2021) mostrarono che da
GPT-2 si potevano recuperare *verbatim* sequenze memorizzate (nomi, recapiti,
frammenti di codice) presenti anche una sola manciata di volte nel corpus. La
memorizzazione cresce con la dimensione del modello e con la ripetizione del
dato: un problema strutturale dei grandi modelli linguistici, non un bug
isolato. Serve quindi una nozione di privacy che sia una *garanzia
matematica*, non un rammendo a posteriori.

`````

## Privacy differenziale: rumore calibrato al singolo

La risposta più solida a questa domanda nasce nel 2006 nella comunità
crittografica, con la **privacy differenziale** di Cynthia Dwork e colleghi
{cite}`dwork2006calibrating`. L'idea, elegante, ribalta la prospettiva: invece
di chiedersi «questo output rivela qualcosa?», si chiede «l'output cambierebbe se
un singolo individuo entrasse o uscisse dai dati?». Se la risposta è «quasi per
niente», allora nessun individuo può essere in pericolo, perché la sua presenza
non lascia traccia rilevabile.

`````{tab} Elementare

C'è un vecchio trucco per fare sondaggi su domande imbarazzanti («hai mai
evaso le tasse?») senza mettere in imbarazzo nessuno. Prima di rispondere,
ognuno lancia in segreto una moneta: se esce testa dice la verità, se esce
croce risponde a caso. Ora, se qualcuno ha detto «sì», tu non puoi accusarlo:
forse è solo la moneta. Eppure, su mille persone, il rumore delle monete si
media e la percentuale vera di evasori salta comunque fuori con buona
precisione. Ogni individuo ha la sua *negabilità plausibile*; la statistica
collettiva sopravvive.

La privacy differenziale è questa idea resa una garanzia numerica: al risultato
di un calcolo sui dati si aggiunge un pizzico di caso, *calibrato* in modo che
la presenza o assenza di una singola persona non sposti quasi nulla. Un solo
manopola, chiamata $\varepsilon$ (epsilon), regola il compromesso: piccola vuol
dire più rumore e più privacy, grande vuol dire meno rumore e più precisione ma
meno protezione.

`````

`````{tab} Superiore

Un meccanismo randomizzato $\mathcal{M}$ soddisfa la **$\varepsilon$-privacy
differenziale** se, per ogni coppia di dataset $D$ e $D'$ che differiscono per un
solo individuo e per ogni insieme di esiti $S$,

$$
\Pr[\mathcal{M}(D) \in S] \;\le\; e^{\varepsilon}\,\Pr[\mathcal{M}(D') \in S].
$$

Qui $\mathcal{M}$ è la procedura (randomizzata) che produce l'output; $D$ e
$D'$ sono *dataset vicini*, identici a meno di una riga; $\varepsilon \ge 0$ è
il **budget di privacy**. La disuguaglianza dice che aggiungere o togliere una
persona può moltiplicare la probabilità di *qualunque* esito al più per
$e^{\varepsilon}$: con $\varepsilon = 0{,}5$ il fattore è
$e^{0{,}5}\approx 1{,}65$, uno scarto modesto. Una versione rilassata, la
**$(\varepsilon,\delta)$-DP**, ammette un termine additivo $+\,\delta$ con
$\delta$ piccolissimo (la probabilità che la garanzia salti) ed è quella che
serve per i meccanismi gaussiani usati nel deep learning.

Come si ottiene? Con il **meccanismo di Laplace**. Data una funzione numerica
$f$, se ne misura la *sensibilità* $\Delta f = \max_{D,D'} \lVert f(D)-f(D')\rVert_1$,
cioè quanto al massimo un singolo individuo può farne variare il valore; poi si
restituisce

$$
\mathcal{M}(D) = f(D) + \mathrm{Lap}\!\left(\frac{\Delta f}{\varepsilon}\right),
$$

rumore estratto da una distribuzione di Laplace di scala $b = \Delta f/\varepsilon$.
Più il calcolo è sensibile al singolo, o più $\varepsilon$ è piccolo, più rumore
va aggiunto. Il risultato garantisce esattamente $\varepsilon$-DP.

`````

Un esempio concreto vale la definizione. Vogliamo pubblicare **quanti**
dipendenti di un'azienda guadagnano oltre una certa soglia: un conteggio.
Aggiungere o togliere una persona cambia il conteggio al massimo di $1$,
dunque la sensibilità è $\Delta f = 1$. Scegliamo $\varepsilon = 0{,}5$: la
scala del rumore è $b = \Delta f / \varepsilon = 1/0{,}5 = 2$. Se il conteggio
vero è $42$, pubblichiamo $42$ più un numero estratto da $\mathrm{Lap}(0, 2)$:
il più delle volte cade entro $\pm 3$, così l'utente legge $43$ o $40$ invece
di $42$. La statistica resta utile, ma nessuno può dedurre dal risultato se
*una specifica persona* fosse o meno nel conteggio: entrambi i mondi (con lei,
senza di lei) producono numeri quasi indistinguibili.

In `numpy` il meccanismo sta in tre righe, ed è eseguibile così com'è:

```python
import numpy as np
rng = np.random.default_rng(0)

def conteggio_privato(conteggio_vero, epsilon):
    sensibilita = 1.0                       # un individuo cambia il conteggio di 1
    b = sensibilita / epsilon               # scala del rumore di Laplace
    return conteggio_vero + rng.laplace(0.0, b)

vero = 42
stime = [conteggio_privato(vero, epsilon=0.5) for _ in range(5)]
print("vero:", vero, " privati:", np.round(stime, 1))
# vero: 42  privati: [42.6 40.8 37.  35.2 44. ]
```

### Dalla statistica al deep learning: DP-SGD

Un conteggio è facile; una rete neurale con milioni di parametri, addestrata per
discesa del gradiente, è un'altra storia. La ricetta che ha reso praticabile la
privacy differenziale nel deep learning è la **DP-SGD** di Abadi e colleghi
{cite}`abadi2016deep`, e modifica la solita discesa del gradiente in due punti.

`````{tab} Elementare

Nell'addestramento normale ogni esempio spinge i pesi del modello nella
direzione che riduce il suo errore. Il problema di privacy è che un esempio
*insolito* può dare una spinta enorme e riconoscibile: la sua impronta resta
nei pesi. DP-SGD fa due cose per cancellare quell'impronta. Primo, mette un
**tetto** alla spinta di ogni singolo esempio: per quanto strano sia, non può
spingere più di tanto. Secondo, alla spinta complessiva del gruppo aggiunge un
po' di **rumore casuale**, così da confondere il contributo dei singoli. Il
modello impara comunque la tendenza generale (la spingono tutti nella stessa
direzione) ma il segno particolare di ciascuno si perde nel rumore. Si paga in
accuratezza, com'è giusto: la privacy non è mai gratis.

`````

`````{tab} Superiore

Ad ogni passo, su un minibatch, DP-SGD calcola il gradiente della loss **per
ogni esempio separatamente**, $g_i = \nabla_\theta \mathcal{L}(\theta, X^{(i)}, y^{(i)})$,
e lo sottopone a due operazioni. Il **clipping per-esempio** limita la norma di
ciascun gradiente a una soglia $C$,

$$
\bar{g}_i = g_i \,/\, \max\!\left(1,\ \frac{\lVert g_i \rVert_2}{C}\right),
$$

così nessun campione può influire oltre $C$; poi si aggiunge **rumore gaussiano**
alla somma e si media,

$$
\tilde{g} = \frac{1}{B}\left( \sum_{i} \bar{g}_i
   + \mathcal{N}\!\big(0,\ \sigma^2 C^2 I\big)\right),
\qquad
\theta \leftarrow \theta - \eta\,\tilde{g}.
$$

Qui $B$ è la dimensione del batch, $\sigma$ il *moltiplicatore di rumore*,
$\eta$ il passo di apprendimento e $I$ l'identità. Il clipping fissa la
sensibilità del passo (nessun esempio la fa esplodere), il rumore gaussiano
fornisce la garanzia; componendo i molti passi con il *moments accountant*
introdotto nello stesso lavoro si ottiene un budget $(\varepsilon,\delta)$
complessivo. Il **compromesso privacy/utilità** è concreto: Abadi e colleghi
addestrano su MNIST con un budget dell'ordine di $\varepsilon \approx 8$
arrivando attorno al $97\%$ di accuratezza (qualche punto sotto un modello non
privato) e la qualità cala via via che si stringe $\varepsilon$. Più privacy,
meno accuratezza: la manopola è sempre la stessa.

`````

## Federated learning: portare il modello ai dati

C'è una via complementare alla privacy: non proteggere l'output di un modello
addestrato su dati raccolti in un unico posto, ma **non raccoglierli affatto**.
È l'idea del *federated learning*, proposta da McMahan e colleghi
{cite}`mcmahan2017communication` per addestrare la tastiera predittiva di milioni
di telefoni senza spedire a un server ciò che le persone digitano.

`````{tab} Elementare

Il modo ovvio di addestrare un modello su dati di tanti ospedali sarebbe
raccogliere tutte le cartelle cliniche in un unico grande archivio. Ma quelle
cartelle non devono uscire dall'ospedale. Il *federated learning* rovescia il
verso del viaggio: invece di portare i dati al modello, porta il **modello ai
dati**. Il server manda a ogni ospedale una copia del modello; ognuno lo
allena un po' sui propri pazienti, in casa; poi rispedisce indietro non i
dati, ma solo il modello aggiornato: cosa ha *imparato*, non cosa ha *visto*.
Il server fonde insieme tutte le versioni in un modello migliore e ricomincia.
Le cartelle non lasciano mai l'ospedale.

`````

`````{tab} Superiore

L'algoritmo di riferimento è **FedAvg**. A ogni round, il server invia i pesi
correnti $\theta_t$ a un sottoinsieme di $K$ client; ciascun client $k$ esegue
alcune epoche di discesa del gradiente sui propri $n_k$ dati locali, ottenendo
$\theta_{t+1}^{k}$; il server li ricompone con una **media pesata** dalla
numerosità locale,

$$
\theta_{t+1} = \sum_{k=1}^{K} \frac{n_k}{n}\,\theta_{t+1}^{k},
\qquad n = \sum_{k} n_k.
$$

Il vantaggio è duplice: i dati grezzi restano sul dispositivo e comunicare i
pesi ogni tanto costa molto meno che spedire i dati ad ogni passo. Ma
attenzione a non dichiarare vittoria troppo presto: **i gradienti perdono
informazione**. Lavori successivi (Zhu e colleghi, 2019) hanno mostrato che da
un aggiornamento condiviso si possono talvolta *ricostruire* gli esempi che
l'hanno prodotto. Il federated learning va perciò combinato con la privacy
differenziale (rumore sugli aggiornamenti) e con l'aggregazione sicura, che
lascia vedere al server solo la somma dei contributi, mai il singolo.
Decentrare i dati riduce il rischio, non lo azzera.

`````

## Esempi avversari: ingannare la rete a comando

Passiamo dalla discrezione alla fragilità. Nel 2013 Szegedy e colleghi
scoprirono una proprietà sconcertante delle reti neurali: si può prendere
un'immagine classificata correttamente, aggiungerle una perturbazione così
piccola da essere **invisibile all'occhio**, e far cambiare idea alla rete con
altissima sicurezza. Due anni dopo Goodfellow, Shlens e Szegedy spiegarono il
fenomeno e ne diedero la ricetta più semplice {cite}`goodfellow2015explaining`.
Il loro esempio è diventato un'icona, e lo riproduce schematicamente la
{numref}`fig-esempio-avversario`.

```{figure} ../figures/esempio-avversario.svg
:name: fig-esempio-avversario
:alt: Tre riquadri in fila collegati da un piu e da un uguale. Nel primo una sagoma stilizzata di panda con etichetta panda 58 per cento. Nel secondo una griglia di rumore impercettibile etichettata epsilon per il segno del gradiente. Nel terzo la stessa identica sagoma di panda con l'etichetta errata gibbone 99 per cento in terracotta.
:width: 100%

La ricetta di un esempio avversario. All'immagine di un panda, riconosciuta
con il $57{,}7\%$ di confidenza, si somma un rumore impercettibile
($\varepsilon$ per il segno del gradiente) e la *stessa* immagine viene
classificata «gibbone» con il $99{,}3\%$ di confidenza. A occhio nudo le due
immagini sono identiche.
```

`````{tab} Elementare

La cosa controintuitiva è che la perturbazione non è casuale: è costruita *su
misura* per quel modello. Un rumore a caso non farebbe quasi nulla; questo,
invece, spinge ogni singolo pixel nella direzione precisa che aumenta l'errore
della rete, tutti d'accordo nello stesso verso. Presi uno a uno, gli
spostamenti sono minuscoli: non li vedi. Ma sommati su centinaia di migliaia
di pixel, formano una spinta abbastanza forte da scavallare il confine di
decisione. È come far cadere una persona non con una spinta, ma con mille dita
che premono tutte dallo stesso lato di un soffio ciascuna: singolarmente
impercettibili, insieme irresistibili. Ed è specifico della macchina: a noi il
panda resta un panda.

`````

`````{tab} Superiore

Il metodo si chiama **Fast Gradient Sign Method** (FGSM). Fissati i pesi
$\theta$, invece di derivare la loss rispetto ai parametri (come
nell'addestramento) la si deriva rispetto all'**input**, e ci si muove nella
direzione che la *aumenta*:

$$
X_{\text{adv}} = X + \varepsilon \cdot \operatorname{sign}\!\big(\nabla_X \mathcal{L}(\theta, X, y)\big).
$$

Qui $X$ è l'input, $y$ l'etichetta vera, $\mathcal{L}$ la loss, $\theta$ i
pesi (congelati), e $\nabla_X \mathcal{L}$ il gradiente della loss *rispetto
all'input*; $\operatorname{sign}(\cdot)$ ne prende il segno componente per
componente e $\varepsilon$ è il budget di perturbazione, cioè la massima
variazione ammessa per singola componente (una norma $\ell_\infty$). Prendere
il solo segno assegna a ogni componente lo stesso spostamento
$\pm\varepsilon$: la perturbazione è impercettibile per pixel, ma allineata al
gradiente e quindi massimamente dannosa. Nell'esempio originale bastava
$\varepsilon = 0{,}007$ (uno scarto sotto la soglia di quantizzazione a 8 bit)
per far passare il panda ($57{,}7\%$) a gibbone ($99{,}3\%$).

FGSM è un unico passo, ed è per questo un attacco *debole*. La sua versione
iterativa è la **Projected Gradient Descent** (PGD) di Madry e colleghi
{cite}`madry2018towards`: si ripete il passo più volte con ampiezza $\alpha$
piccola, riproiettando ogni volta dentro la palla di raggio $\varepsilon$ attorno
all'input originale,

$$
X^{t+1} = \Pi_{\mathcal{B}(X,\varepsilon)}\!\Big( X^{t} + \alpha \operatorname{sign}\!\big(\nabla_X \mathcal{L}(\theta, X^{t}, y)\big) \Big),
$$

dove $\Pi_{\mathcal{B}(X,\varepsilon)}$ è la proiezione sull'insieme delle
perturbazioni ammesse (la palla $\ell_\infty$ di raggio $\varepsilon$ centrata in
$X$). PGD è considerato l'attacco «di primo ordine» più forte e, soprattutto, la
base della difesa: Madry inquadra la robustezza come un problema **min-max**,
$\min_\theta \mathbb{E}_{(X,y)}\big[\max_{\delta \in \mathcal{B}(0,\varepsilon)} \mathcal{L}(\theta, X+\delta, y)\big]$,
in cui l'attaccante (il $\max$ interno, risolto da PGD) e il difensore (il
$\min$ esterno, l'addestramento) giocano l'uno contro l'altro.

`````

## Difese e la corsa agli armamenti

Se il $\max$ interno è l'attacco, il $\min$ esterno è la difesa. La strategia
più efficace e concettualmente pulita è l'**adversarial training**: durante
l'addestramento non si mostrano alla rete solo gli esempi puliti, ma anche le
loro versioni avversarie, generate con PGD ad ogni passo. La rete impara così
a classificare correttamente anche gli input perturbati. Funziona, ma ha un
prezzo: è molto più costoso (ogni passo di addestramento contiene un piccolo
attacco al suo interno) e migliora la robustezza a una certa soglia
$\varepsilon$ spesso a scapito dell'accuratezza sugli esempi puliti.

`````{tab} Elementare

Difendersi dagli esempi avversari somiglia a una rincorsa continua. Si propone
una difesa, sembra reggere, e poco dopo qualcuno trova un attacco nuovo che la
aggira. Molte protezioni annunciate negli anni si sono rivelate illusorie: non
bloccavano davvero l'avversario, gli nascondevano solo le tracce (il
gradiente) che usava per orientarsi, e bastava recuperarle per bucarle di
nuovo. È una *corsa agli armamenti*, e va detto con onestà: al momento non
esiste una difesa definitiva. L'unica garanzia solida viene dalla **robustezza
certificata**, che non promette «nessuno passerà» ma dimostra, con un teorema,
che *dentro un raggio preciso* attorno all'input nessuna perturbazione può
cambiare la risposta: un perimetro piccolo ma sicuro.

`````

`````{tab} Superiore

L'onestà impone di ricordare che molte difese euristiche proposte dopo il 2015
sono state poi aggirate: Athalye e colleghi (2018) mostrarono che davano una
falsa sicurezza per *gradient masking* (offuscavano il gradiente invece di
rimuovere la vulnerabilità) e cadevano appena l'attaccante lo ricostruiva.
L'adversarial training con PGD è tra i pochi ad aver retto. In parallelo si è
sviluppata la **robustezza certificata**, che fornisce garanzie dimostrabili:
il *randomized smoothing* (Cohen e colleghi, 2019), per esempio, costruisce da
qualsiasi classificatore una versione «lisciata» per cui si prova un raggio
$\ell_2$ entro cui la predizione è invariante. Le certificazioni coprono raggi
ancora modesti, ma spostano il terreno da «non sono riuscito a romperla» a «si
dimostra che non si rompe».

Gli esempi avversari agiscono in fase di *inferenza*, su un modello già
addestrato. Esiste una minaccia gemella che agisce in fase di *addestramento*:
il **data poisoning**, in cui l'attaccante inietta esempi malevoli nel dataset
per degradare il modello o piazzarvi una **backdoor**; un innesco segreto (un
piccolo adesivo su un segnale stradale, una parola-chiave in un testo) che, se
presente, fa scattare a comando una risposta scelta dall'attaccante, mentre su
tutti gli altri input il modello si comporta normalmente (Gu e colleghi,
2017). Chi controlla i dati, controlla il modello: un'altra ragione per
prendere sul serio la provenienza dei dati di addestramento.

`````

## Marchiare il sintetico: watermarking e provenienza

La provenienza vale anche dall'altro capo del tubo. Se un modello genera testo,
immagini o voce indistinguibili dal vero, come si riconosce a posteriori che
sono stati generati?

```{figure} ../figures/deepfake-watermarking.svg
:name: fig-watermarking-testo
:alt: "Confronto fra due righe di testo in cui ogni parola è colorata secondo l'appartenenza a una lista verde o a una lista rossa. Nel testo naturale le parole verdi sono circa la metà e non lasciano traccia. Nel testo con watermark le parole verdi sono circa il settanta per cento: un eccesso statistico rilevabile."
:width: 90%

La filigrana su un testo è uno sbilanciamento. Nessuna parola, presa da sola,
è sospetta: è la proporzione sull'intero brano a non essere quella del caso.
```

{numref}`fig-watermarking-testo` mostra anche il limite del metodo, oltre al
suo funzionamento. Se la firma è statistica, serve una quantità di testo
sufficiente perché lo sbilanciamento si distingua dal caso: su una frase corta
non c'è niente da misurare, e riscrivere il brano con parole proprie diluisce
l'eccesso fino a cancellarlo.

`````{tab} Elementare

Due strade, complementari.

La **filigrana nascosta** (*watermarking*) altera il contenuto in modo
impercettibile ma riconoscibile da chi possiede la chiave. Su un'immagine
modifica di pochissimo migliaia di pixel, o le componenti in frequenza, secondo
uno schema segreto: l'occhio non nota nulla, un rilevatore che conosce lo
schema ritrova il pattern anche dopo una compressione moderata. SynthID di
Google DeepMind applica questa idea a immagini, audio e video.

La **provenienza dichiarata** fa l'opposto: invece di nascondere, allega. Lo
standard **C2PA** attacca al file una scheda firmata crittograficamente: chi
l'ha creato, con quale strumento, come è stato modificato.

La differenza pratica è netta e vale la pena tenerla a mente: **il watermark
sopravvive a uno screenshot, i metadati C2PA no**; ma i metadati raccontano una
storia ricca, mentre il watermark dice solo "sono artificiale".

`````

`````{tab} Superiore

Sul testo il meccanismo è diverso e istruttivo. A ogni passo di generazione si
partiziona pseudo-casualmente il vocabolario in una lista "verde" e una
"rossa", con un seme derivato dai token precedenti, e si aggiunge un piccolo
bias ai logit dei verdi. Il testo resta fluido perché le alternative plausibili
sono molte; ma su una sequenza lunga la frazione di token verdi si scosta dal
$50\%$ atteso in modo statisticamente rilevabile. Il rilevatore non deve
conoscere il testo originale: gli basta ricalcolare le liste e fare un test
d'ipotesi (Kirchenbauer e colleghi, 2023).

Ed è anche il punto debole: **una parafrasi distrugge la marca**. Basta far
riscrivere il testo a un altro modello e la partizione verde/rossa si dissolve.
Sulle immagini, ridimensionamento, ritaglio, ricompressione o una foto dello
schermo erodono il segnale; i metadati C2PA li cancella uno screenshot.

C'è poi un limite di natura teorica, non di implementazione: oltre una certa
qualità del contenuto, nessun watermark può essere insieme impercettibile e
robusto contro un avversario determinato. Se il falso è indistinguibile dal
vero per un essere umano, esiste sempre una trasformazione che ne preserva il
significato e cancella la marca.

La conclusione onesta è la stessa della crittografia applicata: il watermarking
non stabilisce cosa è vero, **alza il costo di far passare il sintetico per
autentico**. Non esiste il lucchetto inviolabile, esiste il lucchetto che costa
più della refurtiva.

`````

## FGSM in pratica, con NumPy

Per toccare con mano il fenomeno non serve una rete profonda: basta un
classificatore lineare, ed è anzi il caso in cui la matematica è più
trasparente. Addestriamo una regressione logistica giocattolo, poi costruiamo la
perturbazione FGSM su un esempio che il modello classifica *bene*, e osserviamo
la predizione ribaltarsi. Il gradiente della cross-entropia rispetto all'input,
per la logistica, è semplicemente $(\hat{y}-y)\,W$.

```python
import numpy as np
rng = np.random.default_rng(0)

# --- dataset giocattolo in dimensione d, da un vero modello logistico ---
d, n = 30, 500
w_true = rng.normal(size=d)
X = rng.normal(size=(n, d))
prob = 1.0 / (1.0 + np.exp(-(X @ w_true)))
y = (rng.random(n) < prob).astype(float)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# --- regressione logistica addestrata con la discesa del gradiente ---
w, b = np.zeros(d), 0.0
for _ in range(3000):
    p = sigmoid(X @ w + b)
    w -= 0.2 * (X.T @ (p - y) / n)
    b -= 0.2 * np.mean(p - y)

# --- un esempio classificato correttamente e con buona confidenza ---
i = 1
x, yt = X[i].copy(), y[i]
p0 = sigmoid(x @ w + b)

# --- FGSM: un passo lungo il segno del gradiente della loss rispetto a x ---
grad_x = (p0 - yt) * w                      # dL/dx per la cross-entropy logistica
eps = 0.15
x_adv = x + eps * np.sign(grad_x)
p1 = sigmoid(x_adv @ w + b)

print(f"vera etichetta y = {int(yt)}")
print(f"originale:  p(classe 1) = {p0:.3f}  ->  predice {int(p0 > 0.5)}  (corretto)")
print(f"avversario: p(classe 1) = {p1:.3f}  ->  predice {int(p1 > 0.5)}  (sbagliato)")
print(f"perturbazione: {eps} per feature; norma L2 = {np.linalg.norm(x_adv - x):.2f}"
      f" contro {np.linalg.norm(x):.2f} dell'input")
```

L'output mostra il ribaltamento:

```text
vera etichetta y = 1
originale:  p(classe 1) = 0.890  ->  predice 1  (corretto)
avversario: p(classe 1) = 0.190  ->  predice 0  (sbagliato)
perturbazione: 0.15 per feature; norma L2 = 0.82 contro 6.00 dell'input
```

Il modello passa da una confidenza dell'$89\%$ nella classe corretta al
$19\%$, sbagliando, con una perturbazione la cui norma è meno di un settimo di
quella dell'input. E qui si vede la spiegazione *lineare* di Goodfellow: lungo
la direzione $\operatorname{sign}(W)$, il punteggio si sposta di
$\varepsilon\,\lVert W\rVert_1$, una quantità che cresce con il numero di
dimensioni. In alta dimensione (dove vivono immagini e testi) bastano tante
piccole spinte concordi per scavallare il confine. La stessa formula in
PyTorch si scriverebbe con `x.requires_grad_(True)`, un passaggio
`loss.backward()` e `x + eps * x.grad.sign()`: identica idea, gradiente
rispetto all'input calcolato in automatico.

```{admonition} Da ricordare
:class: important
- I modelli **memorizzano** i dati rari o ripetuti: da qui i *membership
  inference* (capire se un individuo era nel training) e l'**estrazione**
  verbatim di dati sensibili dagli LLM. La memorizzazione è overfitting visto
  come falla di privacy.
- La **privacy differenziale** {cite}`dwork2006calibrating` garantisce che
  l'output cambi al più di un fattore $e^{\varepsilon}$ se un individuo entra o
  esce dai dati, aggiungendo rumore (meccanismo di Laplace) calibrato alla
  sensibilità. **DP-SGD** {cite}`abadi2016deep` la porta nel deep learning con
  clipping per-esempio + rumore gaussiano, al prezzo di un po' di accuratezza.
- Il **federated learning** {cite}`mcmahan2017communication` porta il modello ai
  dati invece del contrario (FedAvg); ma i gradienti condivisi perdono
  informazione, e vanno protetti con DP e aggregazione sicura.
- Gli **esempi avversari** {cite}`goodfellow2015explaining` ingannano una rete
  con perturbazioni impercettibili: **FGSM** somma $\varepsilon$ per il segno del
  gradiente della loss rispetto all'input; **PGD** {cite}`madry2018towards` ne è
  la versione iterativa e la base dell'*adversarial training*.
- Non esiste difesa definitiva: è una **corsa agli armamenti**. La robustezza
  certificata offre garanzie provate ma su raggi piccoli; *data poisoning* e
  *backdoor* attaccano invece in fase di addestramento.
```
