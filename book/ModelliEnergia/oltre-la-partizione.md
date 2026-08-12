# Oltre la partizione: tre modi di aggirare $Z$

Nella sezione precedente è comparso, quasi di sfuggita, il personaggio che
domina questo capitolo: la **funzione di partizione**, che il titolo qui sopra
chiama con la sua iniziale, $Z$. È il conto di cui si diceva: la somma su
*tutte* le configurazioni possibili, quella che trasforma un'altezza in una
percentuale. Vale la pena guardarla in faccia, perché è lei a dettare tutto ciò
che segue, e perché la sua intrattabilità non è una difficoltà tecnica fra le
tante, è un muro.

Una rete di venticinque neuroni accesi o spenti, come quella della memoria
associativa, ha trentatré milioni di configurazioni: un computer le passa in
rassegna in una frazione di secondo, e $Z$ si calcola davvero
($2^{25} = 33\,554\,432$). Aggiungiamone settantacinque. Con cento neuroni le
configurazioni diventano un numero lungo trentuno cifre, e a un miliardo al
secondo servirebbero quasi **tremila volte l'età dell'universo** per contarle
tutte ($2^{100} \approx 1{,}27 \times 10^{30}$, cioè circa
$4 \times 10^{13}$ anni). E cento neuroni accesi o spenti sono un'immagine in
bianco e nero di dieci pixel per dieci: nemmeno una figurina. Nessun trucco di
ingegneria recupera trenta ordini di grandezza: se una strada passa da $Z$,
quella strada è chiusa.

`````{tab} Elementare

Torniamo al paesaggio. Per dire quanto è *alta* una valle rispetto alle
altre non serve nulla di speciale: si guardano le due altezze e si
confrontano. Per dire invece che una valle contiene «il 30% di tutta la
pioggia che cade sul continente» bisogna aver misurato l'intero continente,
valle per valle. La funzione di partizione è la misura dell'intero
continente: è ciò che trasforma un'altezza in una percentuale.

E qui verrebbe da chiedersi perché mai dovremmo misurarlo. La risposta è che
imparare, per un modello a energia, sono **due** gesti e non uno: abbassare il
paesaggio dove stanno i dati veri, e alzarlo dove il modello si immagina roba
che non esiste. Il primo è facile, i dati ce li abbiamo in mano. Il secondo
no: per sapere che cosa il modello si immagina bisogna prima fargli produrre
qualcosa, e per farlo per bene bisognerebbe conoscere il paesaggio intero. Se
ci si limitasse ad abbassare, il modello troverebbe subito la scorciatoia:
abbassare tutto, dappertutto, e dire di sì a qualunque cosa gli si presenti.
L'alzare è il gesto che costa, ed è il motivo per cui la misura del continente
continua a ripresentarsi.

Il problema è che il continente, qui, è grande quanto tutte le immagini
possibili. Non lo si percorre. E allora si può fare una di tre cose.
*Campionarlo* (mandare esploratori a caso e accontentarsi di quello che
riportano). *Evitarlo* (accorgersi che per molte domande la percentuale non
serve: basta la pendenza sotto i piedi). Oppure *aggirarlo con un trucco*:
sostituire la domanda «quanto è probabile questo?» con «questo viene dai dati
o l'ho inventato io?», che è una domanda da rispondere sì o no, e per le
domande sì o no sappiamo addestrare un classificatore da trent'anni.

Le tre strade esistono tutte e tre, hanno tutte e tre un nome, e la seconda
(la meno intuitiva) è quella che nel giro di un decennio ha prodotto i modelli
di diffusione del capitolo precedente.

`````

`````{tab} Superiore

Il punto di attrito è il gradiente della log-verosimiglianza. Da
$p_\theta(\mathbf{x}) = e^{-E_\theta(\mathbf{x})}/Z(\theta)$ segue
$\log p_\theta(\mathbf{x}) = -E_\theta(\mathbf{x}) - \log Z(\theta)$, e derivando rispetto ai
parametri:

$$
\nabla_\theta \log p_\theta(\mathbf{x})
= -\nabla_\theta E_\theta(\mathbf{x})
+ \mathbb{E}_{\mathbf{x}' \sim p_\theta}\!\left[\nabla_\theta E_\theta(\mathbf{x}')\right],
$$

dove il primo termine (**fase positiva**) abbassa l'energia sul dato
osservato e il secondo (**fase negativa**) la rialza sui campioni *del
modello*. Il termine che dà problemi non è $Z$ in sé, ma quel valore atteso:
per calcolarlo bisogna saper campionare da $p_\theta$, cioè dal modello che
stiamo ancora addestrando.

Il tutorial di LeCun {cite}`lecun2006tutorial` legge la stessa formula in
chiave energetica, e la lettura è illuminante: il termine contrastivo «solleva
l'energia di ogni risposta con una forza proporzionale alla sua
verosimiglianza sotto il modello», e tutte le tecniche di approssimazione
(Monte Carlo, metodi variazionali) si possono vedere come **strategie diverse
per scegliere quali risposte tirare su**. Le tre sezioni che seguono sono, in
questa luce, tre risposte alla stessa domanda: chi solleviamo, e come?

`````

## Prima via: campionare il paesaggio

Se il conto esatto su tutto il paesaggio non si può fare, lo si può stimare
visitandone dei pezzi: non calcolare, campionare. Serve però un modo di
produrre risposte davvero pescate dal modello, cioè che escano fuori con la
frequenza che il paesaggio prescrive, e il modo classico è lasciar vagare
qualcosa sul paesaggio abbastanza a lungo perché passi in ogni punto il tempo
che gli spetta. Nel caso continuo la ricetta più usata porta il nome del
fisico francese Paul Langevin, che nel 1908 la scrisse per il moto browniano,
ed è quasi uno slogan: **scendere lungo la pendenza dell'energia, con addosso
un po' di rumore**.

`````{tab} Elementare

Una pallina che rotola in discesa finisce nel fondovalle più vicino e lì si
ferma: è la dinamica di Hopfield, e produce sempre la stessa risposta. Ora
immagina la stessa pallina su un tavolo che vibra: continua a scendere,
perché la pendenza c'è ancora, ma i sussulti la fanno anche risalire un po',
uscire dalle conche, passare da una valle all'altra. Se la guardi per molto
tempo e segni dove si trova, scoprirai che passa **più tempo dove il
paesaggio è basso** e pochissimo sulle cime: la frequenza con cui visita
ogni punto *è* la probabilità che il paesaggio definisce.

Questo è il punto elegante della faccenda: per far vibrare e scendere la
pallina serve solo la pendenza locale, quella sotto i suoi piedi. La misura
dell'intero continente (la costante che non sappiamo calcolare) non entra mai
nella discesa, perché una costante non ha pendenza. Campionare non richiede di
normalizzare.

Il prezzo è il tempo. Se due valli sono separate da una montagna alta, la
pallina può restare intrappolata a lungo da una parte, e la fotografia che
ne ricavi è sbilanciata.

E qui va spiegata una parola che tornerà spesso da adesso in poi: **alta
dimensione**. Un'immagine di dieci pixel per dieci è fatta di cento numeri, e
ognuno di quei numeri è una direzione in cui la si può cambiare: il suo
paesaggio non è una collina con un davanti e un dietro, ha cento direzioni
indipendenti, e quello di una fotografia vera ne ha milioni. In un posto così
le valli sono separate da creste lunghissime e le vie per passare da una
all'altra sono rarissime: una pallina può vagare per un tempo lunghissimo
senza trovarne una. È il tallone d'Achille di tutta la famiglia.

`````

`````{tab} Superiore

La **dinamica di Langevin** genera una sequenza di stati

$$
\mathbf{x}_{k+1} = \mathbf{x}_k - \frac{\epsilon}{2}\, \nabla_{\mathbf{x}} E_\theta(\mathbf{x}_k) + \sqrt{\epsilon}\, \mathbf{z}_k,
\qquad \mathbf{z}_k \sim \mathcal{N}(0, \mathbf{I}),
$$

dove $\epsilon > 0$ è il passo e $\mathbf{z}_k$ il rumore gaussiano. Per
$k \to \infty$, con $\epsilon \to 0$ e $k\epsilon \to \infty$ (il passo si
accorcia, ma il tempo totale percorso dalla catena deve crescere senza
limite), la distribuzione di $\mathbf{x}_k$ converge a
$p_\theta \propto e^{-E_\theta}$. A passo fissato, com'è nel codice qui sotto
e nella pratica degli EBM, la catena si assesta invece su una distribuzione
leggermente distorta, con un errore dell'ordine di $\epsilon$: lo
eliminerebbe un test di accettazione alla Metropolis (la variante MALA), a
cui di solito si rinuncia in cambio della semplicità. Si noti che compare
**solo** $\nabla_{\mathbf{x}} E_\theta$: la costante $\log Z(\theta)$, non dipendendo da
$\mathbf{x}$, ha gradiente nullo. Il campionamento non ha mai bisogno della
normalizzazione: è l'osservazione su cui poggia tutto il resto della sezione.

La versione stocastica su minibatch, che sostituisce il gradiente esatto con
quello stimato, è la *stochastic gradient Langevin dynamics*
{cite}`welling2011bayesian`: un passo di discesa dimezzato ($\epsilon/2$) più
un rumore di ampiezza $\sqrt{\epsilon}$, che per $\epsilon$ piccolo è molto
più grande della deriva (a $\epsilon = 0{,}01$ vale $0{,}1$ contro $0{,}005$),
con il passo che decresce. Nella pratica degli EBM la catena si tronca dopo poche
decine di passi (*short-run MCMC*) e si conservano i campioni in un serbatoio
da cui ripartire, l'erede diretto della persistent contrastive divergence
della sezione precedente.

`````

Il codice che segue costruisce il paesaggio più semplice in cui la faccenda si
vede: due valli e una collinetta in mezzo (in formula, l'energia a doppia buca
$E(x) = (x^2 - 1)^2$, con i minimi in $x = \pm 1$). Ci mette sopra ventimila
palline, le fa vibrare con la ricetta di Langevin e alla fine guarda dove si
sono distribuite, senza aver mai calcolato $Z$. Poi, per pura verifica, $Z$ la
calcola: in un paesaggio a una sola dimensione si può, ed è l'unico modo per
sapere se il campionamento ha detto il vero. Chi non programma può saltare
alla tabella: la colonna «campioni» dice dove sono finite le palline, la
colonna «esatto» dove sarebbero dovute finire.

```python
import numpy as np

rng = np.random.default_rng(0)

# Energia a doppia buca: minimi in x = -1 e x = +1, barriera in x = 0.
def energia(x):
    return (x**2 - 1.0)**2

def gradiente(x):            # dE/dx = 4x(x^2 - 1); lo score e' -gradiente
    return 4.0 * x * (x**2 - 1.0)

# Dinamica di Langevin: ventimila catene in parallelo, passi piccoli.
eps, passi, catene = 0.01, 2000, 20000
x = rng.normal(0.0, 2.0, size=catene)        # partenza qualsiasi
for _ in range(passi):
    x = x - 0.5 * eps * gradiente(x) + np.sqrt(eps) * rng.normal(size=catene)

# Verifica: p(x) = e^{-E(x)}/Z per quadratura numerica (si puo' fare in 1D).
griglia = np.linspace(-3, 3, 60001)
peso = np.exp(-energia(griglia))
Z = np.trapezoid(peso, griglia)
p_esatta = peso / Z

print(f"Z (quadratura)          = {Z:.4f}")
print(f"campioni |x| medio      = {np.abs(x).mean():.3f}")
print(f"esatto   |x| medio      = {np.trapezoid(np.abs(griglia)*p_esatta, griglia):.3f}")
print(f"frazione x>0 (campioni) = {(x > 0).mean():.3f}   (esatto 0.500)")
print()
print(" intervallo   campioni   esatto")
for a, b in [(-2.0, -1.5), (-1.5, -0.5), (-0.5, 0.5), (0.5, 1.5), (1.5, 2.0)]:
    emp = ((x >= a) & (x < b)).mean()
    m = (griglia >= a) & (griglia < b)
    ex = np.trapezoid(p_esatta[m], griglia[m])
    print(f" [{a:+.1f},{b:+.1f})    {emp:6.3f}   {ex:6.3f}")
```

```text
Z (quadratura)          = 1.9737
campioni |x| medio      = 0.822
esatto   |x| medio      = 0.827
frazione x>0 (campioni) = 0.497   (esatto 0.500)

 intervallo   campioni   esatto
 [-2.0,-1.5)     0.011    0.011
 [-1.5,-0.5)     0.380    0.379
 [-0.5,+0.5)     0.225    0.219
 [+0.5,+1.5)     0.373    0.379
 [+1.5,+2.0)     0.012    0.011
```

Le due colonne coincidono entro pochi millesimi (lo scarto più largo è di
0,006, nel bin centrale): le catene hanno ricostruito la distribuzione senza
che $Z$ sia mai entrata nel ciclo. Quell'ultimo millesimo, però, non è tutto
rumore statistico: dentro c'è il bias di passo finito di cui sopra. Per
vederlo, una sola esecuzione non basta, e vale la pena spiegare perché. Con
ventimila catene l'incertezza statistica su un bin vale circa 0,003, cioè
quanto l'effetto che vogliamo misurare: il numero stampato qui sopra, da solo,
non sa distinguere le due cose. Ripetendo l'esperimento su sei semi diversi, a
tempo totale costante $k\epsilon = 20$, lo scarto **medio** sul bin
$[-0{,}5;\,+0{,}5)$ passa da $+0{,}0032 \pm 0{,}0008$ con $\epsilon = 0{,}01$
a $+0{,}0024 \pm 0{,}0011$ con $\epsilon = 0{,}002$ e a
$-0{,}0002 \pm 0{,}0009$ con $\epsilon = 0{,}0005$: in media la barriera è
sovrappesata, e la sovrappesatura si riduce accorciando il passo, che è
esattamente ciò che correggerebbe il test di accettazione di Metropolis. Sul
singolo seme, invece, il segno cambia: con $\epsilon = 0{,}0005$ tre
esecuzioni su sei danno uno scarto negativo. La lezione vale ben oltre questo
esempio: un effetto sistematico grande quanto il rumore si vede solo
ripetendo, e un numero solo, per quanto stampato con quattro cifre, non
dimostra niente.

Vale la pena notare anche *perché* qui funziona così bene, per non trarne una
lezione sbagliata: la barriera fra le due buche è alta un'unità di energia
(bassa) e le catene sono ventimila e indipendenti. Alzando la barriera, o
passando a mille dimensioni dove le valli sono separate da creste
lunghissime, la stessa procedura darebbe una fotografia sbilanciata, e nessuno
se ne accorgerebbe: in alta dimensione la colonna «esatto» non si può
stampare.

## Seconda via: imparare la pendenza, non la probabilità

Se il campionamento è costoso perché insegue le percentuali, si può cambiare
bersaglio. Aapo Hyvärinen, nel 2005, propone di smettere di confrontare
*quanta* probabilità il modello mette in ogni punto, e di confrontare invece
la **pendenza** del paesaggio in quel punto {cite}`hyvarinen2005estimation`.
Sembra un dettaglio ed è una liberazione, per una ragione che si dice in una
riga: la misura dell'intero continente è un numero solo, lo stesso
dappertutto, e un numero uguale dappertutto non ha pendenza. Cambiando
bersaglio, sparisce.

`````{tab} Elementare

Immagina di dover descrivere un paesaggio a qualcuno che non lo vedrà mai.
Puoi dirgli, per ogni punto, «qui c'è il 3% della pioggia», e per farlo devi
aver misurato tutto il continente. Oppure puoi dirgli, per ogni punto, «da qui
si scende verso nord-est, con questa pendenza». La seconda descrizione non
richiede di conoscere il continente: è tutta locale. Eppure basta a
ricostruire la forma del paesaggio, a meno di quanto sta in alto o in basso in
assoluto, che per generare non serve.

Attenzione a non immaginare due paesaggi: è sempre lo stesso. L'altezza è
l'energia, e dove l'energia è bassa la probabilità è alta, quindi parlare del
paesaggio dell'una o dell'altra è parlare della stessa carta, una capovolta
rispetto all'altra. La sua pendenza ha un nome tecnico, **score**, ed è la
stessa parola che compare nel capitolo sui modelli di diffusione. Non è una
coincidenza: è la stessa cosa. Insegnare a una rete la pendenza in ogni punto,
invece della percentuale, è ciò che rende addestrabile un generatore di
immagini, e ciò che ha tolto di mezzo, per quella strada, il problema della
normalizzazione.

`````

`````{tab} Superiore

Lo **score** di una densità è $s(\mathbf{x}) = \nabla_{\mathbf{x}} \log p(\mathbf{x})$ (la lettera $s$ qui
non ha niente a che vedere con lo stato della rete delle sezioni precedenti:
cambia mestiere, e in questa sezione vale questo). Per un modello a energia,

$$
\nabla_{\mathbf{x}} \log p_\theta(\mathbf{x}) = -\nabla_{\mathbf{x}} E_\theta(\mathbf{x}),
$$

perché $\log Z(\theta)$ non dipende da $\mathbf{x}$. Lo **score matching** minimizza
la distanza attesa fra lo score del modello e quello dei dati,

$$
J(\theta) = \frac{1}{2}\,
\mathbb{E}_{\mathbf{x} \sim p_{\text{dati}}}
\left\lVert \nabla_{\mathbf{x}} \log p_\theta(\mathbf{x}) - \nabla_{\mathbf{x}} \log p_{\text{dati}}(\mathbf{x})
\right\rVert^2,
$$

che a prima vista è inservibile (lo score dei dati non lo conosciamo) ma che
un'integrazione per parti, lecita sotto condizioni di regolarità e di
decadimento all'infinito delle densità in gioco (qui sempre assunte, come in
{cite}`hyvarinen2005estimation`), trasforma in una quantità calcolabile su un
campione:

$$
J(\theta) = \mathbb{E}_{\mathbf{x} \sim p_{\text{dati}}}
\left[ \operatorname{tr}\!\big(\nabla_{\mathbf{x}}^2 \log p_\theta(\mathbf{x})\big)
+ \tfrac{1}{2} \left\lVert \nabla_{\mathbf{x}} \log p_\theta(\mathbf{x}) \right\rVert^2 \right]
+ \text{cost.},
$$

dove $\nabla_{\mathbf{x}}^2$ è la matrice hessiana rispetto a $\mathbf{x}$ e la costante non
dipende da $\theta$ {cite}`hyvarinen2005estimation`. Niente $Z$, niente
catene di Markov: solo derivate del modello. Il costo si è spostato sulla
traccia dell'hessiana, che in alta dimensione è cara.

Il colpo di scena arriva nel 2011: Pascal Vincent dimostra che lo score
matching su dati **perturbati con rumore gaussiano** equivale, a meno di
costanti, ad addestrare un *denoising autoencoder*
{cite}`vincent2011connection`. Con $\tilde{\mathbf{x}} = \mathbf{x} + \sigma \boldsymbol{\varepsilon}$ e
$\boldsymbol{\varepsilon} \sim \mathcal{N}(0, \mathbf{I})$, dove $\sigma$ è qui la deviazione
standard del rumore e non la sigmoide di poco fa, il bersaglio dello score sul
dato perturbato è noto in forma chiusa,
$\nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) = -(\tilde{\mathbf{x}} - \mathbf{x})/\sigma^2 = -\boldsymbol{\varepsilon}/\sigma$,
e l'obiettivo diventa una regressione: predire il rumore iniettato. È il
**denoising score matching**, niente hessiana e niente MCMC, ed è, riga per
riga, la loss dei modelli di diffusione del capitolo precedente
{cite}`song2021score`.

Il prezzo c'è, e non è quello che si direbbe: ciò che si impara non è lo score
dei dati, è lo score dei dati **sporcati**, cioè della densità $q_\sigma$
convoluta con la gaussiana. I due coincidono solo nel limite
$\sigma \to 0$, e a $\sigma$ finito resta un errore sistematico che nessuna
quantità di dati riduce. È esattamente il motivo per cui i modelli di
diffusione non usano un solo livello di rumore ma un'intera scala di livelli,
e il campionamento deve attraversarli in fila.

`````

Qui conviene fermarsi un istante, perché il cerchio che si chiude è largo. Il
compito con cui si addestrano i modelli di diffusione, «indovina il rumore che
ho aggiunto a questa immagine», nasce nel capitolo precedente come una scelta
pratica e felice. Vista da questo capitolo è la soluzione di un problema
vecchio di vent'anni: come dare forma a un paesaggio senza mai misurare il
continente. I modelli di diffusione sono, in questa luce, modelli a energia
addestrati sulla pendenza.

Con una differenza tecnica che vale la pena dire per onestà. Un modello a
energia impara l'altezza del paesaggio, e la pendenza si ricava da quella; un
modello di diffusione impara direttamente la pendenza, una freccia per ogni
punto, e non si preoccupa che esista davvero una superficie di cui quelle
frecce siano la discesa. Sono due cose diverse, e a rigore niente garantisce
che le frecce imparate siano la pendenza di qualcosa. Ci si guadagna in
stabilità dell'addestramento, e a chi genera immagini l'altra cosa non è mai
importata.

## Terza via: trasformare la densità in una domanda sì o no

La terza strada è la più obliqua e ha il fascino delle idee che spostano il
problema invece di risolverlo. Michael Gutmann e Aapo Hyvärinen, nel 2010,
osservano che stimare una densità è difficile, ma **distinguere** i dati veri
da rumore fabbricato da noi è un problema di classificazione, e a classificare
siamo bravi {cite}`gutmann2010noise`.

`````{tab} Elementare

Invece di chiedere al modello «quanto è probabile questa immagine?», gli si
chiede: «questa l'ho presa dal mondo o l'ho fabbricata io?». Si mescolano
esempi veri e finti (questi ultimi generati da una sorgente di rumore di cui
sappiamo tutto) e si addestra il modello a smistarli. Per riuscirci, il
modello deve implicitamente sapere quanto ogni esempio è tipico dei dati: la
conoscenza che serviva sta tutta lì dentro, ma è arrivata rispondendo a una
domanda facile.

Se suona familiare, è perché lo è. È la stessa mossa di una delle due reti
delle GAN, quella a cui tocca dire se l'immagine che ha davanti viene dal
mondo o l'ha fabbricata l'altra rete. Ed è la stessa mossa con cui si insegna
a un computer a rappresentare con dei numeri le parole di una lingua o i nodi
di un grafo: gli si mostrano accostamenti veri e accostamenti inventati, e gli
si chiede di distinguerli (nel capitolo sulle Graph Neural Network la si trova
col suo nome inglese, *negative sampling*). La famiglia è più larga di quanto
il nome lasci pensare.

`````

`````{tab} Superiore

La **noise-contrastive estimation** (NCE) affianca ai dati un rumore di
riferimento $p_n$ noto e campionabile, e addestra un classificatore logistico
a distinguere le due sorgenti. Il lavoro del 2010 {cite}`gutmann2010noise`
tratta il caso con tanti campioni di rumore quanti dati ($\nu = 1$); nella
formulazione generale, che gli stessi autori danno due anni dopo sul *Journal
of Machine Learning Research* {cite}`gutmann2012noise`, con $\nu$ campioni di
rumore per ogni dato la probabilità a posteriori che $\mathbf{x}$ venga dai dati è

$$
P(\text{dati} \mid \mathbf{x})
= \frac{p_\theta(\mathbf{x})}{p_\theta(\mathbf{x}) + \nu\, p_n(\mathbf{x})}
= \sigma\!\left(\log p_\theta(\mathbf{x}) - \log p_n(\mathbf{x}) - \log \nu\right),
$$

e si massimizza la log-verosimiglianza di questa classificazione binaria. La
mossa decisiva è che $\log Z$ viene trattata come un **parametro in più**,
stimato insieme agli altri: il modello non normalizzato
$\log p_\theta(\mathbf{x}) = -E_\theta(\mathbf{x}) - c$ impara anche $c$, perché al
classificatore la costante *serve* per calibrarsi, al contrario della massima
verosimiglianza, dove sarebbe stata assorbita e persa
{cite}`gutmann2010noise`.

Il *negative sampling* di word2vec {cite}`mikolov2013distributed` (il secondo
dei due articoli word2vec: il primo usava la softmax gerarchica) è una
semplificazione dichiarata di questa idea, e il
discriminatore delle GAN ne è cugino stretto: in tutti e tre i casi si impara
un rapporto fra densità, non una densità.

`````

## Le tre strade a confronto

{numref}`tab-tre-vie` le mette a confronto.

```{list-table} Tre modi di non pagare il conto del continente, e il prezzo di ciascuno.
:header-rows: 1
:name: tab-tre-vie
:widths: 22 34 44

* - Via
  - Che cosa fa con la misura del continente ($Z$)
  - Che cosa costa
* - **Campionamento** (Langevin e parenti)
  - La ignora: è un numero uguale dappertutto, e un numero uguale dappertutto
    non ha pendenza
  - Tempo. Le palline restano intrappolate da una parte quando le montagne
    sono alte, e in alta dimensione lo sono; e da dentro non si vede
* - **Score matching** (e la sua forma denoising)
  - La elimina, perché sparisce nel passaggio dall'altezza alla pendenza
  - Nella forma originale un conto sulle derivate seconde (l'hessiana), caro
    in alta dimensione; nella forma denoising, il fatto che la pendenza
    imparata è quella dei dati sporcati di rumore, non dei dati
* - **NCE** e parenti
  - La stima come un numero qualunque, insieme a tutto il resto
  - Dipende dal rumore che si sceglie: se è troppo diverso dai dati,
    distinguere diventa banale e non si impara nulla
```

Tre modi di non pagare il conto, e nessuno dei tre gratis. Resta una quarta
possibilità, la più radicale, che è anche la tesi del prossimo paragrafo del
capitolo: **non chiedere mai la probabilità**. Se ciò che serve è decidere,
ordinare, pianificare (non stampare percentuali), l'energia basta da sola, e il
conto non si apre nemmeno.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Misurare l'intero continente non è caro, è impossibile. Cento interruttori
  accesi o spenti danno un numero di configurazioni lungo trentuno cifre: a un
  miliardo di configurazioni al secondo servirebbero quasi tremila volte
  l'età dell'universo, e cento interruttori sono un'immagine in bianco e nero
  di dieci pixel per dieci.
- Imparare vuol dire abbassare il paesaggio dove stanno i dati veri e alzarlo
  dove il modello immagina male. Il primo gesto è facile, i dati ce li
  abbiamo; il secondo no, perché per sapere che cosa il modello immagina
  bisogna prima fargli produrre qualcosa.
- **Prima via, campionare.** La pallina su un tavolo che vibra scende ma ogni
  tanto risale, cambia valle e alla lunga passa più tempo in basso che in
  cima: le serve soltanto la pendenza sotto i piedi, mai la misura del
  continente. Nell'esempio a due valli ricostruisce le proporzioni giuste
  entro pochi millesimi; il prezzo è il tempo, e le montagne alte che la
  tengono prigioniera da una parte sola.
- **Seconda via, la pendenza.** Invece di dire quanta pioggia tocca a ogni
  punto, si dice da che parte si scende e quanto ripido: una descrizione tutta
  locale, che basta a ricostruire la forma del paesaggio. Insegnata su dati
  sporcati apposta, diventa il compito «indovina il rumore che ti ho aggiunto»,
  cioè quello che imparano i modelli di diffusione.
- **Terza via, la domanda sì o no.** Al posto di «quanto è probabile questo?»
  si chiede «viene dal mondo o l'ho fabbricato io?», e si addestra il modello
  a smistare i veri dai finti. Funziona, ma dipende dal rumore che gli si
  mette davanti: se è troppo diverso dai dati, il gioco diventa facile e non
  si impara niente.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- $Z$ non è cara: è impossibile. Con $N = 100$ variabili binarie gli stati
  sono $\approx 1{,}27 \times 10^{30}$, quasi tremila volte l'età
  dell'universo a un miliardo di stati al secondo.
- Il gradiente della log-verosimiglianza ha una **fase positiva** (abbassa
  l'energia sui dati) e una **fase negativa** (la rialza sui campioni del
  modello): è la seconda a richiedere di saper campionare da $p_\theta$.
- **Langevin**:
  $\mathbf{x}_{k+1} = \mathbf{x}_k - \frac{\epsilon}{2}\nabla_{\mathbf{x}} E_\theta(\mathbf{x}_k) + \sqrt{\epsilon}\, \mathbf{z}_k$.
  Usa solo $\nabla_{\mathbf{x}} E$, mai $Z$: nell'esempio a doppia buca
  ricostruisce la distribuzione esatta entro pochi millesimi.
- **Score matching** {cite}`hyvarinen2005estimation` confronta i gradienti
  invece delle densità; la forma **denoising** {cite}`vincent2011connection`
  la riduce a una regressione sul rumore ed è la loss dei modelli di
  diffusione.
- **NCE** {cite}`gutmann2010noise` trasforma la stima di densità in una
  classificazione dati contro rumore, con $\log Z$ come parametro. Il
  *negative sampling* di word2vec è suo discendente.
```
`````
