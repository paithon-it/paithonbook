# Oltre la partizione: tre modi di aggirare $Z$

Nella sezione precedente è comparso, quasi di sfuggita, il personaggio che
domina questo capitolo: la funzione di partizione. Vale la pena guardarlo in
faccia, perché è lui a dettare tutto ciò che segue, e perché la sua
intrattabilità non è una difficoltà tecnica fra le tante, è un muro.

Una rete di 25 neuroni binari, come quella della memoria associativa, ha
$2^{25} = 33\,554\,432$ stati: un computer li enumera in una frazione di
secondo, e $Z$ si calcola davvero. Aggiungiamo settantacinque neuroni. Con
$N = 100$ gli stati diventano $2^{100} \approx 1{,}27 \times 10^{30}$: a un
miliardo di configurazioni al secondo servirebbero circa
$4 \times 10^{13}$ anni, quasi **tremila volte l'età dell'universo**. E cento
neuroni binari sono un'immagine in bianco e nero di dieci pixel per dieci.
Nessun trucco di ingegneria recupera trenta ordini di grandezza: se una
strada passa da $Z$, quella strada è chiusa.

`````{tab} Elementare

Torniamo al paesaggio. Per dire quanto è *alta* una valle rispetto alle
altre non serve nulla di speciale: si guardano le due altezze e si
confrontano. Per dire invece che una valle contiene «il 30% di tutta la
pioggia che cade sul continente» bisogna aver misurato l'intero continente,
valle per valle. La funzione di partizione è la misura dell'intero
continente: è ciò che trasforma un'altezza in una percentuale.

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
$p_\theta(x) = e^{-E_\theta(x)}/Z(\theta)$ segue
$\log p_\theta(x) = -E_\theta(x) - \log Z(\theta)$, e derivando rispetto ai
parametri:

$$
\nabla_\theta \log p_\theta(x)
= -\nabla_\theta E_\theta(x)
+ \mathbb{E}_{x' \sim p_\theta}\!\left[\nabla_\theta E_\theta(x')\right],
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

Se il problema è calcolare un valore atteso sotto $p_\theta$, la risposta
classica è: non calcolarlo, stimalo. Servono campioni del modello, e per
ottenerli si costruisce una catena di Markov che, lasciata correre, li
visita con la frequenza giusta. Nel caso continuo la ricetta più usata porta
il nome del fisico francese Paul Langevin, che nel 1908 la scrisse per il
moto browniano: **scendere lungo il gradiente dell'energia, con addosso un
po' di rumore**.

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
ne ricavi è sbilanciata. In alta dimensione le montagne sono tante, e questo
è il tallone d'Achille di tutta la famiglia.

`````

`````{tab} Superiore

La **dinamica di Langevin** genera una sequenza di stati

$$
x_{k+1} = x_k - \frac{\epsilon}{2}\, \nabla_x E_\theta(x_k) + \sqrt{\epsilon}\, z_k,
\qquad z_k \sim \mathcal{N}(0, I),
$$

dove $\epsilon > 0$ è il passo e $z_k$ il rumore gaussiano. Per
$\epsilon \to 0$ e $k \to \infty$ la distribuzione di $x_k$ converge a
$p_\theta \propto e^{-E_\theta}$. Si noti che compare **solo**
$\nabla_x E_\theta$: la costante $\log Z(\theta)$, non dipendendo da $x$, ha
gradiente nullo. Il campionamento non ha mai bisogno della normalizzazione: è
l'osservazione su cui poggia tutto il resto della sezione.

La versione stocastica su minibatch, che sostituisce il gradiente esatto con
quello stimato, è la *stochastic gradient Langevin dynamics*
{cite}`welling2011bayesian`: mezzo passo di discesa e mezzo di rumore, con il
passo che decresce. Nella pratica degli EBM la catena si tronca dopo poche
decine di passi (*short-run MCMC*) e si conservano i campioni in un serbatoio
da cui ripartire, l'erede diretto della persistent contrastive divergence
della sezione precedente.

`````

Il codice che segue costruisce l'energia a doppia buca $E(x) = (x^2 - 1)^2$
(due minimi in $x = \pm 1$, una barriera in mezzo) e ne campiona con Langevin,
senza mai calcolare $Z$. Alla fine, per pura verifica, $Z$ viene calcolata per
quadratura numerica: in una dimensione si può, ed è l'unico modo per sapere se
il campionamento ha detto il vero.

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

Le due colonne coincidono alla terza cifra: le catene hanno ricostruito la
distribuzione senza che $Z$ sia mai entrata nel ciclo. Vale la pena notare
*perché* qui funziona così bene, per non trarne una lezione sbagliata: la
barriera fra le due buche è alta un'unità di energia (bassa) e le catene sono
ventimila e indipendenti. Alzando la barriera, o passando a mille dimensioni
dove le valli sono separate da creste lunghissime, la stessa procedura darebbe
una fotografia sbilanciata, e nessuno se ne accorgerebbe: in alta dimensione
la colonna «esatto» non si può stampare.

## Seconda via: imparare la pendenza, non la probabilità

Se il campionamento è costoso perché insegue $p_\theta$, si può cambiare
bersaglio. Aki Hyvärinen, nel 2005, propone di non confrontare più le
*densità* ma i loro **gradienti rispetto ai dati**
{cite}`hyvarinen2005estimation`. È una mossa che sembra un dettaglio ed è una
liberazione: la costante di normalizzazione, che non dipende da $x$, sparisce
nella derivata.

`````{tab} Elementare

Immagina di dover descrivere un paesaggio a qualcuno che non lo vedrà mai.
Puoi dirgli, per ogni punto, «qui c'è il 3% della pioggia», e per farlo devi
aver misurato tutto il continente. Oppure puoi dirgli, per ogni punto, «da qui
si scende verso nord-est, con questa pendenza». La seconda descrizione non
richiede di conoscere il continente: è tutta locale. Eppure basta a
ricostruire la forma del paesaggio, a meno di quanto sta in alto o in basso in
assoluto, che per generare non serve.

La pendenza del paesaggio di probabilità ha un nome tecnico, **score**, ed è
la stessa parola che compare nel capitolo sui modelli di diffusione. Non è una
coincidenza: è la stessa cosa. Insegnare a una rete la pendenza in ogni punto,
invece della percentuale, è ciò che rende addestrabile un generatore di
immagini, e ciò che ha tolto di mezzo, per quella strada, il problema della
normalizzazione.

`````

`````{tab} Superiore

Lo **score** di una densità è $s(x) = \nabla_x \log p(x)$. Per un modello a
energia,

$$
\nabla_x \log p_\theta(x) = -\nabla_x E_\theta(x),
$$

perché $\log Z(\theta)$ non dipende da $x$. Lo **score matching** minimizza
la distanza attesa fra lo score del modello e quello dei dati,

$$
J(\theta) = \frac{1}{2}\,
\mathbb{E}_{x \sim p_{\text{dati}}}
\left\lVert \nabla_x \log p_\theta(x) - \nabla_x \log p_{\text{dati}}(x)
\right\rVert^2,
$$

che a prima vista è inservibile (lo score dei dati non lo conosciamo) ma che
un'integrazione per parti trasforma in una quantità calcolabile su un
campione:

$$
J(\theta) = \mathbb{E}_{x \sim p_{\text{dati}}}
\left[ \operatorname{tr}\!\big(\nabla_x^2 \log p_\theta(x)\big)
+ \tfrac{1}{2} \left\lVert \nabla_x \log p_\theta(x) \right\rVert^2 \right]
+ \text{cost.},
$$

dove $\nabla_x^2$ è la matrice hessiana rispetto a $x$ e la costante non
dipende da $\theta$ {cite}`hyvarinen2005estimation`. Niente $Z$, niente
catene di Markov: solo derivate del modello. Il costo si è spostato sulla
traccia dell'hessiana, che in alta dimensione è cara.

Il colpo di scena arriva nel 2011: Pascal Vincent dimostra che lo score
matching su dati **perturbati con rumore gaussiano** equivale, a meno di
costanti, ad addestrare un *denoising autoencoder*
{cite}`vincent2011connection`. Con $\tilde{x} = x + \sigma \varepsilon$ e
$\varepsilon \sim \mathcal{N}(0, I)$, il bersaglio dello score sul dato
perturbato è noto in forma chiusa,
$\nabla_{\tilde{x}} \log q_\sigma(\tilde{x} \mid x) = -(\tilde{x} - x)/\sigma^2 = -\varepsilon/\sigma$,
e l'obiettivo diventa una regressione: predire il rumore iniettato. È il
**denoising score matching**, niente hessiana e niente MCMC, ed è, riga per
riga, la loss dei modelli di diffusione del capitolo precedente
{cite}`song2021score`.

`````

Qui conviene fermarsi un istante, perché il cerchio che si chiude è largo. La
loss di DDPM (predire il rumore aggiunto a un'immagine) nasce nel capitolo
sulla diffusione come una scelta pratica e felice. Vista da questo capitolo è
la soluzione di un problema vecchio di vent'anni: come stimare un modello non
normalizzato senza mai calcolare $Z$. I modelli di diffusione sono, in questa
luce, modelli a energia addestrati per score matching, con una differenza
tecnica che vale la pena dire per onestà: non imparano una $E_\theta$ scalare
per poi derivarla, imparano **direttamente** il campo vettoriale dello score.
Guadagnano in stabilità e perdono la garanzia che quel campo sia il gradiente
di qualcosa; a chi genera immagini, non è mai importato.

## Terza via: trasformare la densità in una domanda sì o no

La terza strada è la più obliqua e ha il fascino delle idee che spostano il
problema invece di risolverlo. Michael Gutmann e Aki Hyvärinen, nel 2010,
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

Se suona familiare, è perché lo è: è la stessa mossa del discriminatore delle
GAN, ed è la stessa del *negative sampling* con cui si addestrano gli
embedding di parole del capitolo sull'NLP. La famiglia è più larga di quanto
il nome lasci pensare.

`````

`````{tab} Superiore

La **noise-contrastive estimation** (NCE) affianca ai dati un rumore di
riferimento $p_n$ noto e campionabile, e addestra un classificatore logistico
a distinguere le due sorgenti. Con un campione di dati e $\nu$ campioni di
rumore per ogni dato, la probabilità a posteriori che $x$ venga dai dati è

$$
P(\text{dati} \mid x)
= \frac{p_\theta(x)}{p_\theta(x) + \nu\, p_n(x)}
= \sigma\!\left(\log p_\theta(x) - \log p_n(x) - \log \nu\right),
$$

e si massimizza la log-verosimiglianza di questa classificazione binaria. La
mossa decisiva è che $\log Z$ viene trattata come un **parametro in più**,
stimato insieme agli altri: il modello non normalizzato
$\log p_\theta(x) = -E_\theta(x) - c$ impara anche $c$, perché al
classificatore la costante *serve* per calibrarsi, al contrario della massima
verosimiglianza, dove sarebbe stata assorbita e persa
{cite}`gutmann2010noise`.

Il *negative sampling* di word2vec {cite}`mikolov2013efficient` è una
semplificazione di questa idea, e il discriminatore delle GAN ne è cugino
stretto: in tutti e tre i casi si impara un rapporto fra densità, non una
densità.

`````

## Le tre strade a confronto

```{list-table}
:header-rows: 1
:name: tab-tre-vie
:widths: 22 34 44

* - Via
  - Che cosa fa con $Z$
  - Che cosa costa
* - **Campionamento** (MCMC, Langevin)
  - La ignora: la costante non ha gradiente in $x$
  - Catene lente, che in alta dimensione restano intrappolate; il difetto non
    si vede da dentro
* - **Score matching** (e la sua forma denoising)
  - La elimina derivando rispetto a $x$
  - L'hessiana nella forma originale; nella forma denoising, un modello che
    conosce la pendenza ma non i livelli
* - **NCE** e parenti
  - La stima come un parametro qualsiasi
  - Dipende dal rumore scelto: se è troppo diverso dai dati, il problema
    diventa banale e non si impara nulla
```

Tre modi di non pagare il conto, e nessuno dei tre gratis. Resta una quarta
possibilità, la più radicale, che è anche la tesi del prossimo paragrafo del
capitolo: **non chiedere mai la probabilità**. Se ciò che serve è decidere,
ordinare, pianificare (non stampare percentuali), l'energia basta da sola, e il
conto non si apre nemmeno.

```{admonition} Da ricordare
:class: important
- $Z$ non è cara: è impossibile. Con $N = 100$ variabili binarie gli stati
  sono $\approx 1{,}27 \times 10^{30}$, quasi tremila volte l'età
  dell'universo a un miliardo di stati al secondo.
- Il gradiente della log-verosimiglianza ha una **fase positiva** (abbassa
  l'energia sui dati) e una **fase negativa** (la rialza sui campioni del
  modello): è la seconda a richiedere di saper campionare da $p_\theta$.
- **Langevin**:
  $x_{k+1} = x_k - \frac{\epsilon}{2}\nabla_x E_\theta(x_k) + \sqrt{\epsilon} z_k$.
  Usa solo $\nabla_x E$, mai $Z$: nell'esempio a doppia buca ricostruisce la
  distribuzione esatta alla terza cifra.
- **Score matching** {cite}`hyvarinen2005estimation` confronta i gradienti
  invece delle densità; la forma **denoising** {cite}`vincent2011connection`
  la riduce a una regressione sul rumore ed è la loss dei modelli di
  diffusione.
- **NCE** {cite}`gutmann2010noise` trasforma la stima di densità in una
  classificazione dati contro rumore, con $\log Z$ come parametro. Il
  *negative sampling* di word2vec è suo discendente.
```
