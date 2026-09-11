# Da dove viene la loss

I minimi quadrati hanno una seconda lettura, oltre a quella geometrica di
{doc}`Ortogonalità e proiezioni </Matematica/ortogonalita-proiezioni>`, dove
sono un'ombra proiettata su un sottospazio. Gauss la percorse in un verso che
oggi si fa al contrario. Nel *Theoria motus* del 1809 si chiese quale
legge degli errori rendesse la media aritmetica il valore più probabile di una
serie di misure: la risposta fu la curva a campana, e da lì la somma dei
quadrati scendeva come conseguenza. Oggi si parte dall'altro capo. Si sceglie
la campana, e i quadrati arrivano.

Il cambio di verso sembra un dettaglio di racconto e vale molto di più, perché
la strada che scende dalla distribuzione alla loss si può percorrere con
qualunque distribuzione, non solo con la campana. Diventa una procedura: dato
un problema nuovo, la sua loss si ottiene invece di cercarla a occhio fra
quelle già viste. Le tre parole che servono sono la verosimiglianza, la
log-verosimiglianza negativa e la mossa che le collega alla rete: la rete
emette i parametri di una distribuzione sull'uscita, al posto del valore.

## La rete dichiara una distribuzione

Finora la rete ha risposto con un numero: questa casa vale $210.000$ euro,
questa foto è un gatto. La mossa che apre tutto il discorso sulle loss è
cambiare l'oggetto della risposta. La rete riceve $\mathbf{x}$ e restituisce
i parametri $\boldsymbol{\lambda}$ di una distribuzione di probabilità
$p(y \mid \boldsymbol{\lambda})$ definita sul dominio delle risposte
possibili. (Questa $\boldsymbol{\lambda}$ raccoglie i parametri di una
distribuzione, ed è in grassetto perché di solito ha più di una componente;
altrove nel libro la stessa lettera, tonda, è il coefficiente di una
penalità, e le due cose non hanno niente in comune.)

Il valore singolo non sparisce: si recupera quando serve, prendendo il punto
in cui la distribuzione dichiarata è più alta. Quello che si guadagna è tutto
il resto della dichiarazione, cioè quanto la rete crede a ciascuna delle altre
risposte.

## La ricetta in quattro mosse

Le mosse sono quattro, e hanno un nome ciascuna perché ognuna tornerà da sola
più avanti.

- **la famiglia**: si prende una distribuzione definita sul dominio delle
  risposte, una gaussiana se la risposta è un numero reale qualunque, una
  Bernoulli se è un sì o un no, una categorica se è uno fra $K$ nomi;
- **la consegna alla rete**: i parametri di quella distribuzione diventano
  l'uscita della rete, $\boldsymbol{\lambda} = f_\theta(\mathbf{x})$;
- **la verosimiglianza**: si cercano i pesi $\theta$ che rendono massima la
  probabilità congiunta delle risposte osservate nei dati di addestramento;
- **il segno e il logaritmo**: per convenienza numerica si passa alla
  log-verosimiglianza cambiata di segno, e quella somma è la loss
  $\mathcal{L}(\theta)$ da minimizzare.

La presentazione in questa forma di procedura è di Prince
{cite}`prince2023understanding`.

`````{tab} Elementare

C'è un gioco che si fa con dieci gettoni. A ogni turno viene fuori qualcosa (il
numero di panini venduti oggi, se domani piove, quale delle cinque squadre
vince) e tu non devi dire la risposta: devi distribuire i gettoni sulle
risposte possibili, prima di sapere com'è andata. Poi si scopre la verità, e
tu vieni pagato in base a quanti gettoni avevi messo proprio lì.

Le regole sono quattro, e la prima è disegnare il tavolo. Se la risposta è un
sì o un no bastano due caselle; se è una fra cinque squadre, cinque caselle; se
è un numero qualunque non ci sono caselle affatto, c'è un righello lungo, e
allora i gettoni si spargono a mucchietto. Su un righello non ha senso chiedere
quanti gettoni stanno esattamente sul punto giusto (di punti ce n'è un'infinità
e su ciascuno ne cadrebbe zero): quello che conta è quanto è spesso il
mucchio lì, cioè quanti gettoni per centimetro. Schiacciandoli tutti in un
centimetro, lì lo strato è altissimo.

La seconda regola è che a distribuire i gettoni non sei tu a mano: lo fa la
rete, e lo fa dopo aver guardato l'indizio di quel turno. A ogni turno
l'indizio cambia, e cambia anche il modo in cui i gettoni finiscono sul tavolo.

La terza è il punteggio. Alla fine di cento turni il punteggio è il prodotto
di tutte le cento vincite: chi ha messo tanti gettoni su ciò che poi è
successo, ogni volta, moltiplica numeri grandi. Moltiplicarle così vuol dire
avere dato per buone due cose: che il tavolo è sempre lo stesso, turno dopo
turno, e che ogni turno si conta per sé, senza che com'è andata ieri cambi il
punteggio di oggi.

La quarta è una comodità di conti. Cento numeri minori di uno moltiplicati fra
loro danno una cifra così piccola che il calcolatore la confonde con lo zero,
quindi si smette di moltiplicare vincite e si comincia a sommare penalità: a
ogni turno una multa, piccola se avevi messo tanti gettoni sulla
risposta uscita, grande se ne avevi messi pochi. Il logaritmo è esattamente la
funzione che trasforma un prodotto in una somma, e il segno meno trasforma la
vincita in una multa. Il totale delle multe è quello che la rete deve far
scendere.

E c'è un vincolo, uno solo, senza il quale il gioco non starebbe in piedi: i
gettoni sono dieci e non uno di più. Per metterne di più su una risposta
bisogna toglierli a un'altra, ed è da questo vincolo che escono, una dopo
l'altra, tutte le penalità che seguono.

`````

`````{tab} Superiore

Sia $\mathcal{D}=\{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^{m}$ l'insieme di
addestramento. Si sceglie una famiglia $p(y \mid \boldsymbol{\lambda})$
definita sul dominio di $y$, si pone
$\boldsymbol{\lambda}^{(i)} = f_\theta(\mathbf{x}^{(i)})$ e si scrive la
verosimiglianza dei parametri della rete:

$$
L(\theta) = \prod_{i=1}^{m}
p\big(y^{(i)} \mid f_\theta(\mathbf{x}^{(i)})\big) .
$$

Il prodotto poggia su due ipotesi che conviene enunciare, perché non sono
gratis. La prima: la famiglia è la stessa per ogni esempio, e a cambiare
da un esempio all'altro sono solo i parametri. La seconda: le risposte sono
**condizionatamente indipendenti** dati gli ingressi, cioè sapere com'è andato
l'esempio $j$ non dice niente in più sull'esempio $i$ una volta noto
$\mathbf{x}^{(i)}$. Su dati raccolti in serie (misure consecutive dello stesso
sensore, parole della stessa frase) la seconda è quasi sempre falsa, e ciò che
ne risente è la stima dell'incertezza, non tanto il centro.

Il logaritmo è strettamente crescente, quindi non sposta il punto di massimo,
e il segno meno scambia il massimo con un minimo:

$$
\mathcal{L}(\theta) = -\sum_{i=1}^{m}
\log p\big(y^{(i)} \mid f_\theta(\mathbf{x}^{(i)})\big) ,
\qquad
\hat\theta = \arg\max_{\theta} L(\theta)
= \arg\min_{\theta} \mathcal{L}(\theta) .
$$

Quella somma è la **log-verosimiglianza negativa**, ed è la loss. Il vantaggio
numerico è che si sommano logaritmi invece di moltiplicare $m$ numeri minori
di uno, che in virgola mobile finirebbero a zero; il vantaggio concettuale è
che ogni esempio contribuisce con un addendo proprio, cioè con il costo
$\ell$ di una singola predizione.

Il vincolo che rende il gioco non banale è la normalizzazione: per ogni
$\boldsymbol{\lambda}$ vale $\sum_y p(y \mid \boldsymbol{\lambda}) = 1$, o
$\int p(y \mid \boldsymbol{\lambda})\,\mathrm{d}y = 1$ nel caso continuo.
Alzare la probabilità di una risposta obbliga ad abbassarla altrove, e senza
questo vincolo il criterio si potrebbe soddisfare dichiarando tutto
probabilissimo.

Una nota sul caso continuo: lì $p$ è una densità, non una probabilità, e
può superare $1$. Ne segue che $-\log p$ può essere negativa, e che il suo
valore assoluto dipende dall'unità di misura scelta per $y$ (cambiare metri in
centimetri sposta la loss di una costante additiva). A essere confrontabili
sono le differenze fra due modelli sugli stessi dati, non il numero da solo.

`````

## Il pagamento: le loss che già si usano escono da qui

La ricetta si giudica da quello che restituisce nei casi noti, e restituisce
proprio le due loss che la
{doc}`backpropagation </RetiNeurali/backpropagation>` userà: l'errore
quadratico per la regressione, la cross-entropia per la classificazione.
Le tira fuori la stessa procedura, cambiando una riga sola.

### Un numero reale: la gaussiana dà l'errore quadratico

`````{tab} Elementare

Il tavolo è il righello, e sul righello i gettoni si spargono con uno stampo
fisso, sempre della stessa larghezza: la campana. Alla rete resta una cosa sola
da decidere, dove mettere il centro. E il centro può cadere in qualunque punto
del righello, quindi alla rete non si chiede nessuna acrobazia per farcelo
stare: qualunque numero le esca va bene così com'è.

Adesso guarda che multa esce. Lo stampo è più spesso al centro e si assottiglia
allontanandosi, e si assottiglia in un modo particolare: lo spessore cala come
l'esponenziale della distanza al quadrato. La multa è il logaritmo di quello
spessore, cambiato di segno, e il logaritmo di un esponenziale è l'esponente:
resta la distanza al quadrato, divisa per un numero fisso che dipende solo
dalla larghezza dello stampo.

Ecco il pagamento. Siccome lo stampo non cambia mai larghezza, quel numero
fisso è lo stesso a ogni turno, e lo è anche la parte di multa che si paga
comunque, perché anche l'altezza del mucchio al centro è sempre la stessa.
Quindi confrontare due reti guardando le loro multe totali equivale a
confrontarle guardando la somma delle distanze al quadrato.
Cioè: chi minimizza l'errore quadratico sta giocando questo gioco con la
campana, che lo sappia o no. Il quadrato viene di lì, dalla forma dello
stampo, e punire di più gli sbagli grossi ne è la conseguenza.

E si vede subito che cosa lo renderebbe la scelta sbagliata. Se ogni tanto
capita un turno stravagante, un numero lontanissimo da tutti gli altri, quel
turno da solo tira il centro verso di sé, perché la multa cresce col quadrato
e una distanza doppia ne pesa quattro. E se i panini
venduti sono sempre o pochissimi o moltissimi, e quasi mai una via di mezzo, un
mucchietto solo, per quanto ben centrato, mette il grosso dei gettoni proprio
nella zona dove non succede mai niente. La campana è simmetrica e ha una sola
gobba: dove i dati ne hanno due, questo tavolo è mal disegnato, e nessun modo
di centrarla rimedia.

`````

`````{tab} Superiore

Per $y \in \mathbb{R}$ la famiglia naturale è la gaussiana
$\mathcal{N}(\mu, \sigma^2)$ della
{doc}`sezione su probabilità e statistica </Matematica/probabilita-statistica>`,
con densità

$$
p(y \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\,
\exp\!\left(-\frac{(y-\mu)^2}{2\sigma^2}\right) .
$$

La rete predice il solo centro, $\mu^{(i)} = f_\theta(\mathbf{x}^{(i)})$, e
$\sigma^2$ resta una costante non appresa. Il dominio del parametro $\mu$ è
tutta la retta reale, quindi sull'uscita non serve nessuna funzione di
schiacciamento: ecco perché in regressione l'ultimo strato è lineare. Passando
al logaritmo cambiato di segno,

$$
\mathcal{L}(\theta) = \sum_{i=1}^{m}
\left[\frac{1}{2}\log\!\big(2\pi\sigma^2\big)
+ \frac{\big(y^{(i)} - f_\theta(\mathbf{x}^{(i)})\big)^2}{2\sigma^2}\right] .
$$

Il primo addendo non dipende da $\theta$ e il denominatore $2\sigma^2$ è una
costante positiva: nessuno dei due sposta il punto di minimo. Quello che resta
è $\sum_i (y^{(i)} - \hat y^{(i)})^2$, la somma dei quadrati dei residui. I
minimi quadrati sono la log-verosimiglianza negativa di una gaussiana a
varianza fissa, e le ipotesi che li giustificano sono esattamente tre:
gaussianità, indipendenza condizionata, varianza costante.

Quali sono i punti di rottura si legge dalle ipotesi, una per una. Se la
distribuzione condizionata vera ha code più pesanti di una gaussiana, un
singolo valore anomalo entra nella somma col suo quadrato e sposta la stima:
la cura è cambiare famiglia, e la Laplace (che dà l'errore assoluto e stima la
mediana invece della media) è la scelta usuale. Se è multimodale nessuna
scelta di $\mu$ rimedia, e la cura è ammettere più gobbe, cioè una
{doc}`mistura di gaussiane </MachineLearning/riduzione-clustering>` come
famiglia di uscita. Se la varianza cambia con l'ingresso, il rimedio è la
regressione eteroschedastica, che la ricetta produce cambiando una mossa sola,
la seconda.

`````

### Una risposta fra tante: Bernoulli e categorica danno la cross-entropia

Quando le risposte possibili si contano, il dominio su cui va definita la
distribuzione è un insieme finito, e le famiglie che ci vivono sono la
Bernoulli con due esiti e la categorica con $K$. Cambia la famiglia, il resto
della procedura no.

`````{tab} Elementare

Il tavolo adesso ha delle caselle, e i gettoni ci vanno dentro davvero. Due
caselle se la domanda è «piove sì o no», cinque se è «quale squadra vince».

Qui compare un problema pratico che la campana non aveva. La rete, dentro,
produce punteggi grezzi, numeri che possono valere qualunque cosa, anche $-4$
o $37$. Ma i gettoni sono dieci e devono stare tutti sul tavolo: nessuna
casella può riceverne un numero negativo, e la somma deve fare dieci esatti.
Serve quindi un passaggio che prenda i punteggi grezzi e li trasformi in una
distribuzione legittima. Con due caselle è la sigmoide, con molte è la
softmax, e sono proprio quelle che la sezione sulle funzioni di attivazione
mette in fondo a un classificatore. Adesso si vede perché stanno lì e perché
sono quelle: sono il modo di rendere valida una distribuzione di gettoni. E si
fa una volta sola: chi lo rifà una seconda volta ridistribuisce gettoni già
distribuiti, e il tavolo che ne esce non è più quello che voleva.

La multa, poi, si legge da sola. Alla fine del turno esce una casella, e la
multa dipende solo da quanti gettoni c'erano lì dentro: nessuno se ne
avevi messi dieci, tantissimi se ne avevi messi quasi zero. Il nome tecnico è
cross-entropia, e la lettura è questa: la rete paga la sicurezza sbagliata
molto più cara dell'incertezza. Dichiarare novantanove su cento e prendere
l'altra costa un'enormità; dichiarare cinquanta e cinquanta costa poco, e costa
poco sempre, anche quando si indovina.

Ed è il vincolo dei dieci gettoni a impedire la furbizia ovvia. Se si potessero
riempire tutte le caselle, la strategia migliore sarebbe metterne tante
dappertutto e non sbagliare mai. Non si può: alzarne una vuol dire abbassarne
un'altra, quindi dichiarare significa esporsi, sempre.

`````

`````{tab} Superiore

Per $y \in \{0,1\}$ la famiglia è la Bernoulli, che ha un solo parametro,
la probabilità della classe $1$. Quel parametro è la predizione stessa,
$\hat y \in [0,1]$, e

$$
p(y \mid \hat y) = \hat y^{\,y}\,(1-\hat y)^{1-y} .
$$

Il dominio del parametro è $[0,1]$ e l'uscita grezza della rete vive su tutto
$\mathbb{R}$: la si schiaccia con la sigmoide, $\hat y = \sigma(z)$ con
$z = f_\theta(\mathbf{x})$. La log-verosimiglianza negativa è

$$
\mathcal{L}(\theta) = -\sum_{i=1}^{m}
\Big[y^{(i)} \log \hat y^{(i)}
+ \big(1-y^{(i)}\big) \log\big(1-\hat y^{(i)}\big)\Big] ,
$$

cioè la **cross-entropia binaria**. Per $y \in \{1,\dots,K\}$ la famiglia è la
**categorica**, i parametri sono $K$ probabilità che sommano a $1$, il
simplesso è il loro dominio, e a portarci i $K$ logit è la softmax; la loss
diventa $\mathcal{L}(\theta) = -\sum_i \log \hat y^{(i)}_{c_i}$, con $c_i$ la
classe vera dell'esempio $i$.

Dietro i tre casi c'è una regolarità sola, ed è la parte da portarsi via: la
funzione sull'ultimo strato la detta il dominio del parametro. La retta
reale della media gaussiana non chiede niente, l'intervallo $[0,1]$ della
Bernoulli chiede la sigmoide, il simplesso della categorica chiede la softmax.
La tabella di {doc}`Funzioni di attivazione </RetiNeurali/funzioni-attivazione>`
diceva che sull'uscita la funzione la detta il problema; qui si vede
attraverso che cosa la detta.

Da qui anche il motivo per cui `nn.CrossEntropyLoss` di PyTorch vuole i logit
e non le probabilità: la softmax e il logaritmo si compongono in
$\log\hat y_c = z_c - \log\sum_k e^{z_k}$, che si calcola in modo stabile con
il *log-sum-exp*, mentre applicare prima la softmax e poi il logaritmo perde
cifre proprio dove la loss è grande, cioè dove $\hat y_c$ è così piccola da
sparire in virgola mobile.

Un ultimo ponte, verso la
{doc}`teoria dell'informazione </Matematica/teoria-informazione>`. Sulla
distribuzione empirica dei dati la cross-entropia fra bersaglio e modello
coincide con la log-verosimiglianza negativa media: minimizzare la
cross-entropia, avvicinare il modello ai dati nel senso della divergenza KL e
massimizzare la verosimiglianza sono tre nomi della stessa operazione, e la
ricetta ne ha appena percorso il terzo.

`````

### Le risposte di altra forma: il dominio è un catalogo

Errore quadratico e cross-entropia sono due voci di un elenco più lungo, e chi
ha la ricetta può leggerlo tutto. A cambiare da una voce all'altra è una cosa
sola: l'insieme in cui la risposta ha il permesso di cadere. Una durata non è
mai negativa, una proporzione sta fra zero e uno, una direzione torna su sé
stessa dopo un giro, un conteggio salta di uno in uno. Ognuna di quelle forme
ha almeno una distribuzione che ci vive sopra, e sceglierla è tutta la mossa;
il resto della procedura non cambia di una riga.

```{figure} ../figures/il-dominio-sceglie-la-distribuzione.svg
:name: fig-dominio-distribuzione
:alt: Otto riquadri, in ciascuno la forma dell'insieme in cui vive la risposta disegnata a sinistra e a destra il nome della distribuzione che ci vive e che cosa si predice. Tutta la retta: gaussiana, un numero qualsiasi. I numeri positivi, cioè una semiretta che parte da zero: esponenziale o gamma, una durata o una grandezza. Un segmento fra zero e uno: beta, una proporzione. Un cerchio: von Mises, una direzione. Due caselle: Bernoulli, sì o no. K caselle: categorica, una classe fra K. I numeri interi a partire da zero, disegnati come punti staccati: Poisson, quante volte. Tre rette parallele: normale multivariata, più risposte insieme. In fondo, la riga che chiude: il dominio del parametro sceglie la funzione dell'ultimo strato, nessuna sulla retta, la sigmoide sul segmento, la softmax sulle caselle.
:width: 90%

Le due loss di sempre sono due voci di questo elenco, quella della retta
intera e quella delle caselle. Le altre si ricavano con la stessa procedura,
cambiando la sola riga in cui si sceglie la famiglia.
```

Il catalogo di {numref}`fig-dominio-distribuzione`, dominio per dominio, è di
Prince {cite}`prince2023understanding`, e va letto con una cautela: dice quali
distribuzioni *possono* stare su quel dominio, e fra quelle non sceglie. Sulla
retta intera ci vive anche la Laplace, che dà l'errore assoluto invece del
quadrato, e ci vive una mistura di gaussiane; a decidere fra le tre sono i
dati che si hanno.

Lo stesso gesto, senza reti, è quello dei {doc}`modelli lineari generalizzati
</MachineLearning/apprendimento-supervisionato>`: scelgono anche loro una
distribuzione, e un punteggio lineare gliene fornisce il parametro passando per
una funzione di legame. Qui il punteggio lineare
diventa una rete, e la funzione dell'ultimo strato fa il mestiere del legame
percorso all'incontrario, dal punteggio grezzo al parametro. Lineare,
logistica e Poisson erano già la stessa macchina con tre impostazioni: il
resto della ricetta non è cambiato.

## Quando l'incertezza cambia da punto a punto

Fin qui la larghezza della gaussiana è rimasta una costante, ed è
un'assunzione forte con un nome: **omoschedasticità**, cioè incertezza uguale
dappertutto. La ricetta però non obbliga a fissarla. Se i parametri della
distribuzione sono l'uscita della rete, e la varianza è un parametro, allora
anche la varianza può essere un'uscita della rete: è la **regressione
eteroschedastica**, vecchia quanto la statistica, e a farla predire a una rete
sono Nix e Weigend nel 1994 {cite}`nix1994estimating`.

`````{tab} Elementare

Torniamo al righello, e cambiamo una regola sola: adesso lo stampo non ha più
una larghezza fissa. A ogni turno la rete dichiara due cose invece di una, dove
sta il centro e quanto largo spargere.

Sembra un regalo, ed è soprattutto una domanda in più a cui rispondere,
perché sbagliare la larghezza si paga in tutti e due i versi. Sparso stretto, i
gettoni stanno tutti in un dito di righello: se la verità cade lì la vincita è
enorme, se cade appena fuori il mucchio è già finito e la multa è pesantissima.
Sparso largo, il mucchio copre mezzo tavolo e la verità ci cade dentro di
sicuro, ma è così basso che anche il centro pieno frutta pochissimo. I dieci
gettoni sono sempre dieci: allargare vuol dire assottigliare.

Quindi al giocatore conviene una cosa sola, dichiarare davvero quanto è
sicuro. Nei turni facili stringe e incassa; in quelli difficili allarga e si
protegge, pagando il prezzo del mucchio basso. E il conto lo mostra: la
larghezza che frutta di più è esattamente lo sbaglio tipico del giocatore,
quello che si ottiene facendo la media degli sbagli al quadrato e poi la
radice, né un dito di meno né uno di più.

C'è però un modo in cui questo gioco si guasta, e chi lo usa lo incontra. Un
giocatore pigro può accorgersi che dichiararsi incerto costa poco e permette
di smettere di mirare: sui turni difficili allarga il mucchio, la multa
diventa sopportabile comunque, e la fatica di trovare il centro giusto lì non
la fa più. A rimediare è il modo di allenarsi, non una regola in più: le due
manopole non si girano mai insieme, e si aggiusta il centro a larghezza ferma,
poi la larghezza a centro fermo.

E c'è un limite che il gioco non copre affatto. La larghezza dichiarata dice
quanto ballano i risultati nei turni che il giocatore ha già visto, e non dice
niente su quanto lui sia in alto mare davanti a un turno di un genere mai
capitato prima. Lì una larghezza la dichiara lo stesso, con la stessa faccia
sicura di sempre, e non c'è nel gioco niente che lo smentisca.

`````

`````{tab} Superiore

La rete $f_\theta$ ha due uscite, $f_1$ e $f_2$. La prima è il centro,
$\mu = f_1(\mathbf{x})$. La seconda deve dare una larghezza, che è positiva per
definizione mentre l'uscita di uno strato lineare non lo è, quindi le si fa
attraversare una funzione che la rende positiva; la scelta comune è leggerla
come il logaritmo dello scarto tipico, $\sigma = \exp\big(f_2(\mathbf{x})\big)$,
che copre tutti gli ordini di grandezza senza chiedere alla rete uscite
grandi.
(Questa $\sigma$ è la larghezza della gaussiana, e non ha niente a che vedere
con la $\sigma(\cdot)$ della classificazione binaria, che è la sigmoide: la
prima è un numero, la seconda una funzione.) La loss diventa

$$
\mathcal{L}(\theta) = \sum_{i=1}^{m}
\left[\log \sigma_i
+ \frac{\big(y^{(i)} - \mu_i\big)^2}{2\sigma_i^{2}}\right]
+ \frac{m}{2}\log 2\pi ,
$$

e i due addendi si leggono uno contro l'altro. Il secondo è l'errore
quadratico pesato: un punto su cui la rete si è dichiarata incerta pesa
meno, ed è la cosa che con la varianza fissa non si poteva fare. Il primo è il
prezzo di dichiararsi incerti, e senza di esso il minimo sarebbe
$\sigma_i \to \infty$ per ogni $i$, cioè la resa incondizionata.

Che il prezzo sia tarato bene si vede fissando $\mu$ e annullando la derivata
rispetto a $\sigma$: si ottiene $\hat\sigma^2 = (y-\mu)^2$ sul singolo punto e,
su un gruppo di punti che condividono la stessa larghezza, la media dei
quadrati dei residui. La larghezza ottima è lo scarto quadratico medio,
cioè la loss chiede di dichiarare l'incertezza che si ha davvero.

Due punti di rottura. Il primo: il gradiente rispetto a $\mu_i$
porta un fattore $1/\sigma_i^2$, quindi la rete può abbassare la loss
gonfiando $\sigma$ dove il centro è difficile, e da lì in poi su quei punti
smette di correggere il centro; l'addestramento congiunto di media e varianza
è instabile per questa ragione, e il rimedio è non aggiornarle mai insieme,
ottimizzando l'una a parametri dell'altra fermi
{cite}`detlefsen2019reliable`. Il secondo: la varianza predetta descrive il
rumore dei dati, non l'ignoranza del modello. Su una zona dell'ingresso senza
nemmeno un esempio di addestramento la rete dichiarerà una larghezza qualsiasi,
e con la stessa faccia sicura che ha altrove.

`````

## In pratica, con NumPy

Tre affermazioni si controllano in un blocco solo. Che a varianza fissa la
log-verosimiglianza negativa sia l'errore quadratico più una costante; che la
larghezza scelta dalla verosimiglianza sia lo scarto quadratico medio, con un
costo che sale allargando e stringendo; e che due modelli con lo stesso centro,
quindi con lo stesso errore quadratico, si distinguano appena si guarda
l'incertezza dichiarata.

```python
import numpy as np

rng = np.random.default_rng(0)

# Duecento misure. La media vera e' una retta; il rumore no: nella meta'
# destra e' sei volte piu' largo che nella sinistra.
n = 200
x = np.sort(rng.uniform(0, 1, n))
sigma_vero = np.where(x < 0.5, 0.1, 0.6)
y = 2 * x + 1 + rng.normal(0, sigma_vero)

# Un solo centro per tutti e due i modelli: la retta ai minimi quadrati.
a, b = np.polyfit(x, y, 1)
residui = y - (a * x + b)

def nll(residui, sigma):
    "log-verosimiglianza gaussiana cambiata di segno, sommata sugli esempi"
    return np.sum(0.5 * np.log(2 * np.pi * sigma**2)
                  + residui**2 / (2 * sigma**2))

# 1. A larghezza fissa la NLL e' l'errore quadratico piu' una costante.
sse = np.sum(residui**2)
print(f"NLL(sigma=1) = {nll(residui, 1.0):.3f}")
print(f"0,5 * SSE    = {0.5 * sse:.3f}   differenza = "
      f"{nll(residui, 1.0) - 0.5 * sse:.3f}")
print(f"n/2 * log(2 pi) = {n / 2 * np.log(2 * np.pi):.3f}")

# 2. Se la larghezza la sceglie la verosimiglianza, il minimo cade sullo
#    scarto quadratico medio: sia stringere sia allargare costa.
rmse = np.sqrt(np.mean(residui**2))
print(f"\nscarto quadratico medio: {rmse:.3f}")
for s in [0.5 * rmse, rmse, 2 * rmse, 10 * rmse]:
    print(f"  sigma = {s:.3f}  ->  NLL = {nll(residui, s):.1f}")

# 3. Stesso centro, stesso errore quadratico, due modi di dichiarare
#    l'incertezza: uno solo per tutti, oppure uno per meta'.
sinistra = x < 0.5
s_sx = np.sqrt(np.mean(residui[sinistra]**2))
s_dx = np.sqrt(np.mean(residui[~sinistra]**2))
print(f"\nerrore quadratico medio: {np.mean(residui**2):.4f} (uguale)")
print(f"una larghezza sola:  sigma = {rmse:.3f}          "
      f"NLL = {nll(residui, rmse):.1f}")
etero = nll(residui[sinistra], s_sx) + nll(residui[~sinistra], s_dx)
print(f"una per meta':       sigma = {s_sx:.3f} e {s_dx:.3f}  NLL = {etero:.1f}")
```

```text
NLL(sigma=1) = 201.416
0,5 * SSE    = 17.628   differenza = 183.788
n/2 * log(2 pi) = 183.788

scarto quadratico medio: 0.420
  sigma = 0.210  ->  NLL = 271.6
  sigma = 0.420  ->  NLL = 110.2
  sigma = 0.840  ->  NLL = 173.8
  sigma = 4.199  ->  NLL = 471.7

errore quadratico medio: 0.1763 (uguale)
una larghezza sola:  sigma = 0.420          NLL = 110.2
una per meta':       sigma = 0.115 e 0.552  NLL = 26.8
```

La differenza fra le prime due righe è $183{,}788$, ed è la terza riga: la
costante vale $\frac{n}{2}\log 2\pi$ e non dipende dai residui, quindi
minimizzare la log-verosimiglianza negativa a larghezza fissa e minimizzare la
somma dei quadrati portano nello stesso punto. Il secondo gruppo dice che la
larghezza non è gratis in nessuno dei due versi: dimezzarla porta la loss da
$110{,}2$ a $271{,}6$, decuplicarla a $471{,}7$, e il minimo cade su
$0{,}420$, che è appunto lo scarto quadratico medio. Il terzo è il confronto
che conta: l'errore quadratico medio vale $0{,}1763$ per tutti e due i modelli,
perché il centro è lo stesso, e non ha modo di preferirne uno; la loss
costruita con la ricetta scende invece da $110{,}2$ a $26{,}8$, e le due
larghezze che sceglie, $0{,}115$ e $0{,}552$, sono vicine a quelle con cui i
dati sono stati generati, $0{,}1$ e $0{,}6$.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- La penalità con cui si addestra una rete si ricava, invece di cercarla a
  occhio in un elenco, e la domanda da cui si ricava è una sola: che forma ha
  la risposta che stiamo chiedendo.
- La mossa è che la rete smette di dire un numero e comincia a dire come si
  spargono dieci gettoni sulle risposte possibili. I gettoni sono dieci:
  metterne di più su una risposta vuol dire toglierli a un'altra, e dichiarare
  significa esporsi.
- Da lì la penalità viene da sé. Con i gettoni sparsi a campana su un righello
  viene la distanza al quadrato, cioè l'errore quadratico di sempre. Con i
  gettoni nelle caselle viene la cross-entropia, che punisce la sicurezza
  sbagliata molto più dell'incertezza.
- La funzione che sta in fondo alla rete (la sigmoide con due risposte, la
  softmax con molte) serve a rendere valida la distribuzione dei gettoni:
  niente quantità negative, e il totale che torna.
- Se la rete dichiara anche quanto largo spargere, impara a dire quando non
  sa. Allargare protegge e frutta poco, stringere frutta molto e rischia
  grosso, e il conto migliore è dichiarare l'incertezza che si ha davvero.
  Ma è l'incertezza dei casi già visti: davanti a un caso di un genere mai
  capitato una larghezza la dichiara lo stesso.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- La ricetta è in quattro mosse: si sceglie una famiglia
  $p(y \mid \boldsymbol{\lambda})$ sul dominio delle risposte, si pone
  $\boldsymbol{\lambda} = f_\theta(\mathbf{x})$, si massimizza la
  verosimiglianza dei dati e si minimizza la log-verosimiglianza negativa, che
  è la loss.
- Le loss note ne sono i casi particolari. Gaussiana a varianza fissa →
  minimi quadrati; Bernoulli → cross-entropia binaria; categorica →
  cross-entropia multiclasse. Le ipotesi dei minimi quadrati sono quindi tre e
  vanno dichiarate: gaussianità, indipendenza condizionata, varianza costante.
- La funzione sull'ultimo strato la detta il dominio del parametro: nessuna
  per $\mu \in \mathbb{R}$, sigmoide per $[0,1]$, softmax per il simplesso.
- La regressione eteroschedastica fa predire anche $\sigma$, e la loss
  diventa $\sum_i [\log\sigma_i + (y^{(i)}-\mu_i)^2 / 2\sigma_i^2]$: errore
  pesato più il prezzo di dichiararsi incerti. Senza il termine $\log\sigma_i$
  il minimo sarebbe $\sigma \to \infty$.
- Due limiti restano. Il gradiente su $\mu_i$ va come $1/\sigma_i^2$, quindi
  gonfiare $\sigma$ spegne l'apprendimento del centro; e la varianza predetta
  descrive il rumore dei dati, non l'ignoranza del modello fuori dai dati
  visti.
```

`````

Con questo la loss ha smesso di essere un ingrediente che qualcuno ci passa, e
si può costruire per un problema nuovo, anche uno che non compare fra questi
capitoli: si guarda che forma ha la risposta, si trova una distribuzione
definita là sopra, e si scrive il logaritmo cambiato di segno. Resta la
domanda a cui serviva: adesso che $\mathcal{L}(\theta)$ c'è, come si trovano i
pesi che la rendono minima. È il mestiere della backpropagation, che ricava la
pendenza della loss rispetto a ciascuno dei parametri, e della discesa del
gradiente, che con quelle pendenze si muove.
