# Backpropagation: come impara una rete

Nel 1986 tre ricercatori (David Rumelhart, Geoffrey Hinton e Ronald Williams)
pubblicano su *Nature* un articolo di poche pagine, *"Learning representations
by back-propagating errors"* {cite}`rumelhart1986learning`. L'algoritmo che
descrivono non era del tutto nuovo. Le radici stanno nella **differenziazione
automatica**, cioè l'arte di far calcolare a un programma non solo il risultato
di un conto ma anche di quanto quel risultato cambierebbe muovendo ciascuno dei
suoi ingressi: il modo che serve qui lo pubblica il finlandese Seppo Linnainmaa
nel 1970, e Paul Werbos lo porta sulle reti neurali nella sua tesi del 1974. Ma
è quel testo del 1986 a mostrare al mondo come una rete neurale possa
correggersi da sola, un errore alla volta. È la stessa ricetta con cui, ancora
oggi, imparano modelli da miliardi di parametri.

L'idea sta in due movimenti, come un respiro. **In avanti** la rete produce una
risposta; **all'indietro** misura di quanto ha sbagliato e distribuisce la
"colpa" a ogni peso. Vediamo i due movimenti uno per uno.

## Il forward pass: dai dati all'uscita

Un esempio entra da sinistra e attraversa gli strati uno dopo l'altro, finché
l'ultimo strato non emette una previsione. Ogni strato prende ciò che riceve, lo
combina con i propri parametri e lo passa avanti.

`````{tab} Elementare

Immagina una catena di montaggio. Alla prima postazione arrivano i dati grezzi
(per esempio i pixel di una foto). Ogni postazione ha una fila di "manopole"
(i **pesi**) con cui mescola ciò che riceve, applica un piccolo filtro e
consegna il risultato alla postazione successiva. L'ultima postazione affaccia
il prodotto finito: la previsione della rete, per esempio "gatto: 0,92".

Nessuna postazione vede l'intero problema: ognuna trasforma solo un pezzetto e
lo passa avanti. Questo scorrere in avanti, dai dati alla risposta, è il
**forward pass**.

`````

`````{tab} Superiore

Indichiamo con $a^{[0]} = x$ l'input. Per ogni strato $l = 1, \dots, L$ il
forward pass calcola una combinazione lineare seguita da una non linearità:

$$
z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}, \qquad
a^{[l]} = \sigma\!\left(z^{[l]}\right) \;\; (l < L), \qquad
a^{[L]} = \varphi\!\left(z^{[L]}\right).
$$

Qui $W^{[l]}$ è la matrice dei pesi dello strato $l$, $b^{[l]}$ il vettore di
bias, $z^{[l]}$ la pre-attivazione e $a^{[l]}$ l'attivazione. Come
nell'overview, $\sigma$ è l'attivazione degli strati nascosti (per esempio la
ReLU, $\sigma(z)=\max(0,z)$) e $\varphi$ quella dello strato d'uscita, che di
norma è un'altra: softmax per la classificazione, identità per la regressione.
L'uscita finale è la previsione $\hat{y} = a^{[L]}$. Ogni strato non è che il
prodotto matrice-vettore già incontrato in algebra lineare, "avvolto" in una
non linearità.

`````

## Quanto abbiamo sbagliato: la funzione di loss

La previsione da sola non basta: serve un numero che dica *quanto* la rete ha
sbagliato rispetto alla risposta giusta. Quel numero è la **loss**, e imparare
significa renderlo il più piccolo possibile.

`````{tab} Elementare

La loss è una distanza tra la risposta della rete e la verità. Mettiamo che la
rete debba stimare il prezzo di una casa: la casa vale davvero 200.000 € e lei
ne prevede 170.000, quindi l'errore è di 30.000. Poi quell'errore si eleva al
quadrato: $30.000 \times 30.000 = 900$ milioni. Perché al quadrato? Per punire
di più gli sbagli grossi, e si vede subito confrontando due casi: sbagliare di
60.000, cioè il doppio, dà $3.600$ milioni, cioè **quattro** volte tanto.
Raddoppiare l'errore ne quadruplica il costo, e la rete impara a evitare le
cantonate prima delle imprecisioni. Più la previsione è vicina al vero, più la
loss è piccola; se fossero identiche, la loss sarebbe zero. Tutto
l'addestramento è una caccia a quel numero più basso.

`````

`````{tab} Superiore

Per la regressione si usa spesso l'**errore quadratico medio** su $m$ esempi:

$$
\mathcal{L} = \frac{1}{m} \sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right)^2 .
$$

Per la classificazione si preferisce la **cross-entropia**, che confronta la
distribuzione prevista $\hat{y}$ con l'etichetta $y$:
$\mathcal{L} = -\sum_{k} y_k \log \hat{y}_k$. In entrambi i casi $\mathcal{L}$ è
una funzione dei parametri $\theta = \{W^{[l]}, b^{[l]}\}$: cambiando i pesi
cambia la loss, e il nostro obiettivo è trovare i $\theta$ che la minimizzano.

Che per la classificazione si «preferisca» la cross-entropia merita una
ragione, e non è solo che si accorda con l'interpretazione probabilistica. È
meccanica, e riguarda proprio il gradiente. Con un'uscita sigmoide
$\hat{y} = \sigma(z)$ e la MSE, la derivata rispetto a $z$ porta un fattore
$\sigma'(z)$:

$$
\frac{\partial}{\partial z}\,\tfrac{1}{2}(\sigma(z)-y)^2
= (\sigma(z)-y)\,\sigma'(z).
$$

Ma $\sigma'(z) = \sigma(z)(1-\sigma(z))$ vale quasi zero agli estremi, cioè
**proprio quando il neurone è sicuro e sbagliato**: il modello che ha torto
marcio è quello che impara più lentamente, che è l'esatto contrario di quel che
serve. Con la cross-entropia quel fattore si semplifica:

$$
\frac{\partial}{\partial z}\Big[-y\log\sigma(z) - (1-y)\log(1-\sigma(z))\Big]
= \sigma(z) - y,
$$

e il gradiente diventa **proporzionale all'errore**: più si sbaglia, più si
corregge. È lo stesso fenomeno di saturazione che nella sezione sulle funzioni
di attivazione motivava l'abbandono della sigmoide, visto però dal lato della
loss invece che da quello dell'attivazione: la scelta della funzione di costo
non è una convenzione, è ciò che decide se il gradiente sopravvive.

`````

## L'idea della backpropagation

Sappiamo di quanto la rete ha sbagliato. La domanda vera è: *di chi è la colpa?*
Ogni peso, in mezzo alla catena, ha contribuito un po' all'errore finale. La
backpropagation calcola con precisione quel contributo, ripercorrendo la rete a
ritroso.

```{figure} ../figures/rete-forward-backward.svg
:name: fig-forward-backward
:alt: Una rete a quattro strati con una freccia in alto verso destra (forward, i dati che avanzano) e una freccia in basso verso sinistra (backward, il gradiente dell'errore che si propaga verso gli strati iniziali).
:width: 90%

I due movimenti dell'addestramento. In avanti (in alto) i dati diventano una
previsione; all'indietro (in basso) il gradiente dell'errore risale la rete e
raggiunge anche i primi strati.
```

`````{tab} Elementare

Pensa a una catena di responsabilità. L'errore nasce all'uscita, ma non è
"colpa" solo dell'ultimo strato: viene ereditato da quello prima, e da quello
prima ancora, fino all'inizio. La backpropagation parte dal fondo e chiede a
ogni strato: "quanto hai contribuito tu a questo errore?". La risposta di uno
strato serve a calcolare quella dello strato precedente, come un rimprovero
che si passa all'indietro lungo la fila ({numref}`fig-forward-backward`).

Un esempio in piccolo, con i numeri di prima: la rete prevede 170.000 € per la
casa che ne vale 200.000. I 30.000 € di errore vengono ripartiti tra i neuroni
dell'ultimo strato in proporzione a quanto ciascuno ha pesato sulla risposta:
chi ha contribuito con un peso grande eredita una colpa grande, chi ha
contribuito poco quasi niente. Poi ogni neurone gira la propria quota di colpa
ai neuroni dello strato prima, con lo stesso criterio, fino all'ingresso.

Alla fine ogni singolo peso ha in mano la sua quota di colpa, e quella quota
non dice solo *quanto*, dice anche *da che parte*, perché ha un segno. Nel
nostro caso la rete ha previsto **troppo poco**: allora un peso che spingeva la
risposta verso l'alto va alzato, uno che la tirava verso il basso va abbassato.
Se la rete avesse previsto troppo, tutto al contrario. È l'ultimo anello, e da
qui in poi la correzione è meccanica.

`````

`````{tab} Superiore

Il meccanismo è la **regola della catena** applicata a ritroso (→ la deriviamo
nel capitolo di Matematica). Scriviamo tutto per un **singolo esempio**: il
gradiente della loss media di un mini-batch è la media di questi contributi,
uno per esempio. Definiamo il segnale d'errore $\delta^{[l]}$ e lo
propaghiamo dallo strato d'uscita verso l'input:

$$
\delta^{[L]} = \nabla_{a^{[L]}} \mathcal{L} \;\odot\; \varphi'\!\left(z^{[L]}\right),
\qquad
\delta^{[l]} = \left(W^{[l+1]}\right)^{\!\top} \delta^{[l+1]} \;\odot\; \sigma'\!\left(z^{[l]}\right).
$$

Da $\delta^{[l]}$ ricaviamo i gradienti che ci servono per aggiornare i parametri:

$$
\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \delta^{[l]} \left(a^{[l-1]}\right)^{\!\top},
\qquad
\frac{\partial \mathcal{L}}{\partial b^{[l]}} = \delta^{[l]} .
$$

Il simbolo $\odot$ è il prodotto elemento per elemento (Hadamard), $\sigma'$ e
$\varphi'$ le derivate delle due attivazioni: la scrittura con $\odot$
presuppone quindi un'attivazione applicata componente per componente. La
softmax, cioè la $\varphi$ tipica della classificazione, non lo è (ogni
uscita dipende da tutti i logit), ma accoppiata alla cross-entropia il conto
si semplifica e il termine d'uscita diventa $\delta^{[L]} = \hat{y} - y$: è la
combinazione che i framework implementano. Il punto cruciale: ogni $\delta^{[l]}$ riusa
$\delta^{[l+1]}$, così un solo passaggio all'indietro basta a calcolare tutti i
gradienti. È questo che rende l'addestramento praticabile su reti enormi.

Che il verso giusto sia questo non è un caso, ed è il contenuto della
**differenziazione automatica**. Derivare automaticamente si può in due modi.
Nel **modo diretto** si propaga in avanti, insieme al calcolo, la derivata
rispetto a una direzione fissata dei parametri: una passata dà la derivata
lungo *quella* direzione, e per il gradiente completo servono $n$ passate, una
per parametro. Nel **modo inverso** si propaga all'indietro dall'uscita, e una
passata sola le dà tutte quante. Quando l'uscita è **una** (la loss è uno
scalare) e gli ingressi sono milioni, il verso conveniente è ovviamente il
secondo, e il gradiente finisce per costare un multiplo costante della
funzione, qualunque sia il numero di parametri: è il *cheap gradient
principle*. La backpropagation è il modo inverso applicato a una rete: quello
generale è di Linnainmaa (1970), Werbos (1974) è chi lo porta qui.

Il conto però non è gratis, e il prezzo è in **memoria**. Per calcolare
$\partial\mathcal{L}/\partial W^{[l]} = \delta^{[l]}(a^{[l-1]})^{\!\top}$ serve
l'attivazione $a^{[l-1]}$: il forward deve quindi **conservare** le attivazioni
di tutti gli strati finché il gradiente non torna indietro a prenderle. È
l'unica voce che cresce con la profondità **e** con la dimensione del batch
mentre i pesi restano gli stessi. Misurata su una rete di venti strati da $512$
unità, con i byte davvero trattenuti dal grafo: a batch $32$ le attivazioni
pesano un quindicesimo del modello, a batch $2048$ quattro volte il modello,
sessantaquattro volte tanto per un batch sessantaquattro volte più grande. Da
qui il *gradient checkpointing*, che ne butta via una parte e la ricalcola in
avanti quando serve: memoria contro tempo.

`````

```{figure} ../figures/backpropagation.gif
:name: fig-backpropagation-animata
:alt: Animazione di una rete a tre strati. Prima un impulso percorre gli archi da sinistra a destra fino a un riquadro che misura l'errore; poi un impulso torna indietro e, strato dopo strato, si allunga di un pezzo la formula che dice quanto ciascun peso ha contribuito all'errore.
:width: 85%

I due movimenti, uno dopo l'altro: il segnale va avanti fino all'errore, poi la
colpa torna indietro. Tornando, a ogni strato che attraversa **si moltiplica
per un pezzo in più**: è il prodotto che si allunga sullo schermo, e ogni suo
fattore è il contributo di uno strato.
```

La {numref}`fig-backpropagation-animata` mostra perché il passaggio all'indietro
costa quanto quello in avanti e non di più: nessun peso viene ricalcolato da
capo, a ogni strato si aggiunge un pezzo a un prodotto che esiste già. Chi ha le
derivate nello zaino ci riconosce la regola della catena; chi non le ha può
tenersi l'immagine del prodotto che si allunga, che è la stessa cosa.

## Aggiornare i pesi: discesa del gradiente e learning rate

Prima di procedere conviene saldare due parole che finora sono corse su binari
separati. La "colpa" di un peso e la sua **pendenza** sono la stessa cosa detta
in due lingue: la quota di colpa che un peso si porta a casa è di quanto
cambierebbe l'errore se muovessi quel peso di pochissimo, ed è esattamente ciò
che si intende per pendenza dell'errore rispetto a quel peso. Il **gradiente**
non è che l'elenco completo di queste pendenze, una per peso. Quindi la
backpropagation, che distribuisce le colpe, e la discesa in cui stiamo per
entrare, che segue le pendenze, non sono due meccanismi: sono la prima metà e
la seconda metà dello stesso gesto.

Il gradiente indica, per ogni peso, la direzione in cui la loss *cresce*. Per
farla calare basta muoversi nel verso opposto, a piccoli passi.

```{figure} ../figures/discesa-gradiente-da-zero.svg
:name: fig-discesa-passi
:alt: "Una curva a forma di valle percorsa da una successione di punti: in orizzontale il valore di un peso, in verticale l'errore che ne risulta. Partendo in alto su un fianco, ogni passo scende verso il fondo, e i passi si accorciano man mano che la pendenza diminuisce, addensandosi vicino al minimo."
:width: 88%

In orizzontale il valore di un peso, in verticale l'errore che ne risulta:
muovere il peso vuol dire spostarsi lungo questa valle, e imparare vuol dire
scendere. I passi si accorciano da soli, e nessuno li rimpicciolisce: la
lunghezza è proporzionale alla pendenza, che vicino al fondo è quasi nulla.
```

L'addensarsi dei punti in {numref}`fig-discesa-passi` è una proprietà comoda e
insieme un problema. Comoda perché l'algoritmo rallenta da sé arrivando a
destinazione, senza che nessuno glielo dica; problema perché rallenta
altrettanto sui tratti piatti che *non* sono il fondo, quelli in cui il terreno
si stende senza che ci sia niente sotto, o in cui scende da una parte e sale
dall'altra come una sella da cavallo. Sono i plateau e le selle già incontrati
nel capitolo di matematica.

`````{tab} Elementare

Immagina di essere su una collina, nella nebbia, e di voler scendere a valle.
Non vedi lontano, ma puoi sentire la pendenza sotto i piedi e fare un passo
nella direzione più ripida verso il basso. Ripeti, passo dopo passo.

Quanto è lungo il passo lo decidono due cose insieme: la pendenza che senti
sotto i piedi, e un moltiplicatore fisso che scegli tu, il **learning rate**.
La pendenza è quella che accorcia i passi da sola vicino al fondo, come nella
figura qui sopra; il moltiplicatore è la manopola che hai in mano. Con un
moltiplicatore troppo grande scavalchi la valle e rimbalzi avanti e indietro;
con uno troppo piccolo scendi lentissimo. Trovare un buon moltiplicatore è metà
del mestiere.

`````

`````{tab} Superiore

L'aggiornamento è la **discesa del gradiente**:

$$
\theta \leftarrow \theta - \eta \, \nabla_{\theta} \mathcal{L},
$$

dove $\theta$ sono i parametri, $\nabla_{\theta}\mathcal{L}$ il gradiente
calcolato dalla backpropagation ed $\eta > 0$ il **learning rate** (o tasso di
apprendimento). Un $\eta$ troppo grande fa divergere la loss; troppo piccolo
rende la convergenza lentissima o la blocca in un minimo mediocre. Gli
ottimizzatori moderni aggiungono memoria delle direzioni già prese (Momentum,
che però conserva un solo $\eta$ per tutti i parametri) oppure un passo diverso
per ciascun parametro (RMSProp, **Adam** {cite}`kingma2015adam`), ma il cuore
resta questo.

`````

## Mini-batch, epoche e SGD

Calcolare il gradiente su *tutti* i dati a ogni passo sarebbe accuratissimo ma
lentissimo. In pratica si divide il dataset in **mini-batch** (per esempio 32 o
64 esempi): per ciascuno si fa un forward, una backpropagation e un aggiornamento
dei pesi. Un giro completo su tutti i mini-batch è un'**epoca**; l'addestramento
ne conta decine o centinaia. Poiché ogni batch è un campione casuale dei dati,
la pendenza che si misura non è quella vera ma una sua stima un po' storta, e
storta in modo diverso a ogni gruppetto: per questo il metodo si chiama
**discesa del gradiente stocastica** (SGD, *Stochastic Gradient Descent*).

Quel tremolio, che sembrerebbe un difetto, è utile, ma non per la ragione che
si racconta più spesso. Non serve tanto a scavalcare i minimi locali, che nelle
reti profonde sono rari; serve a staccarsi dai tratti piatti e dalle selle di
poco fa, dove la pendenza vera è quasi zero e un algoritmo perfettamente preciso
resterebbe fermo. Un po' di imprecisione dà la spinta per uscirne
{cite}`dauphin2014identifying`. E c'è evidenza che porti a fermarsi in valli
larghe invece che in fessure strette, che è un bene: una soluzione che regge
anche se la si sposta un po' regge meglio anche sui dati nuovi.

## Reti profonde: attenzione ai gradienti

Più la rete è profonda, più il gradiente deve viaggiare lontano per
raggiungere i primi strati, e lungo il tragitto può degradarsi.

```{figure} ../figures/vanishing-exploding-gradients.svg
:name: fig-gradienti-svaniscono
:alt: "Cinque strati affiancati e, dentro ciascuno, una barra che misura l'ampiezza del gradiente che torna indietro dall'uscita verso l'ingresso: alta al quinto strato, si riduce a ogni passaggio fino a essere un trattino quasi invisibile al primo. Quattro frecce indicano la direzione della retropropagazione, da destra verso sinistra."
:width: 92%

Il gradiente si spegne tornando indietro. Gli strati vicini all'uscita
ricevono un segnale forte e imparano; i primi, che dovrebbero costruire le
rappresentazioni di base, quasi non lo sentono.
```

C'è un dettaglio crudele in {numref}`fig-gradienti-svaniscono`: la rete non
smette di addestrarsi, e la loss continua a calare. A imparare sono gli ultimi
strati, che si arrangiano su rappresentazioni iniziali rimaste quasi a caso.
Dal di fuori sembra addestramento; dal di dentro, metà della rete è ferma.

"A caso" è da prendere alla lettera, ed è l'occasione per dire da dove parte una
rete: i pesi si estraggono a sorte, piccoli, non si mettono a zero. Il
percettrone di due sezioni fa poteva permetterselo perché aveva un neurone solo;
in uno strato di cento, con tutti i pesi a zero i cento neuroni calcolerebbero
lo stesso identico numero, riceverebbero la stessa identica correzione e
resterebbero uguali fra loro per sempre. Il caso iniziale serve a rompere quella
simmetria. Quanto piccoli, e con quale regola, è una scelta che pesa parecchio
e ha una sezione tutta sua nel capitolo sul deep learning.

`````{tab} Elementare

È il gioco del "telefono senza fili". Il messaggio (il gradiente) parte
dall'uscita e viene sussurrato all'indietro di strato in strato. Se a ogni
passaggio si affievolisce, arriva ai primi strati talmente flebile da non
insegnare loro nulla: sono i **gradienti che svaniscono** e la rete non impara.
Può succedere anche il contrario, e qui il paragone col telefono senza fili si
rompe, perché di sussurri che diventano urla passando di bocca in bocca non se
ne sono mai visti: se a ogni strato il messaggio viene moltiplicato per un
numero maggiore di uno, dopo cinquanta strati arriva assordante e manda tutto in
tilt. Sono i **gradienti che esplodono**.

Le reti profonde vanno quindi progettate perché il messaggio arrivi integro fino
in fondo, e un rimedio lo conosci già: è la **ReLU** della sezione precedente,
che dal lato positivo non appiattisce il segnale e quindi non lo indebolisce a
ogni passaggio. Gli altri (da dove far partire i pesi, le scorciatoie che
saltano gli strati) sono il mestiere del capitolo sul deep learning.

`````

`````{tab} Superiore

Il gradiente verso i primi strati è un prodotto di molti fattori: le Jacobiane
strato per strato, la cui "grandezza" si misura con i **valori singolari** (e
non con gli autovalori, perché sono matrici diverse l'una dall'altra e non c'è
nessuna potenza di una matrice sola da diagonalizzare: è l'avvertimento del
capitolo di algebra lineare, ed è qui che serviva).

Se i valori singolari **massimi** restano sistematicamente sotto $1$, il
prodotto tende a zero esponenzialmente con la profondità (*vanishing
gradient*), e la garanzia è immediata: la norma di un prodotto non supera il
prodotto delle norme, $\lVert\prod_l J_l\rVert \le \prod_l \sigma_{\max}(J_l)
\le c^{\,L}$.

Nell'altro verso, però, **non c'è simmetria**, ed è l'errore che si fa a
scrivere la frase di getto. Che ogni fattore allunghi qualche direzione non
basta a far esplodere niente, perché lo strato successivo può accorciare proprio
quella: il prodotto di trenta matrici con $\sigma_{\max}=2$ ciascuna può avere
norma $3\cdot10^{-11}$ (basta alternare $\mathrm{diag}(2;\,0{,}1)$ e
$\mathrm{diag}(0{,}1;\,2)$). A garantire l'esplosione è il valore singolare
**minimo** sopra $1$, che è una condizione molto più forte: allora
$\lVert\prod_l J_l\,\mathbf{v}\rVert \ge \prod_l \sigma_{\min}(J_l)\,
\lVert\mathbf{v}\rVert$ e non c'è scampo. È per questo che lo svanire si
prevede in anticipo e l'esplodere si tampona a valle, con il *gradient
clipping*.

L'analisi è quella resa celebre da Hochreiter
{cite}`hochreiter1991untersuchungen` e da Bengio {cite}`bengio1994learning`
sulle reti ricorrenti. I rimedi standard: attivazioni **ReLU** al posto della
sigmoide {cite}`glorot2011deep`, **inizializzazione** accorta dei pesi
({cite}`glorot2010understanding`, {cite}`he2015delving`), **batch
normalization** {cite}`ioffe2015batch`, **connessioni residue** delle ResNet
{cite}`he2016deep` e, appunto, il *gradient clipping*. Sono queste tecniche ad
aver reso addestrabili reti da centinaia di strati.

`````

## In pratica, con PyTorch

Nella pratica non implementiamo la backpropagation a mano: i framework la
eseguono per noi con la differenziazione automatica di poco fa (in PyTorch si
chiama *autograd*). A noi resta da dichiarare l'architettura, la loss e
l'ottimizzatore, e da scrivere il ciclo di addestramento, che in PyTorch
ricalca passo per passo il respiro descritto in questa sezione.

Il problema qui è il riconoscimento delle cifre scritte a mano, il classico
esercizio di prima prova: ogni immagine è un quadrato di $28\times 28$ pixel in
scala di grigio, che disteso in fila fa $784$ numeri in ingresso, e le risposte
possibili sono $10$, le cifre da $0$ a $9$.

```{code-block} python
:class: pt-non-eseguibile

import torch
from torch import nn, optim

model = nn.Sequential(
    nn.Linear(784, 64), nn.ReLU(),   # strato nascosto
    nn.Linear(64, 10),               # uscita: un punteggio per classe
)

criterion = nn.CrossEntropyLoss()                   # la loss
optimizer = optim.SGD(model.parameters(), lr=0.01)  # discesa del gradiente

for epoca in range(20):
    for X_batch, y_batch in train_loader:  # mini-batch di 32 esempi
        y_pred = model(X_batch)            # forward: la previsione
        loss = criterion(y_pred, y_batch)  # quanto abbiamo sbagliato
        optimizer.zero_grad()              # azzera i gradienti vecchi
        loss.backward()                    # backpropagation automatica
        optimizer.step()                   # aggiornamento dei pesi
```

Le quattro righe dentro il ciclo sono esattamente i movimenti che abbiamo
descritto: i dati avanzano, la loss misura l'errore, `loss.backward()` fa
tornare indietro il gradiente, `optimizer.step()` aggiorna i pesi.

Il `20` delle epoche non è un numero magico, ed è anzi la domanda che il codice
lascia aperta: quand'è che si smette? Non quando la loss sui dati di
addestramento smette di calare, perché quella può calare anche mentre il
modello sta imparando a memoria gli esempi che ha visto invece della regola che
li governa. È l'*overfitting* incontrato nel capitolo di machine learning, e si
riconosce nello stesso modo: tenendo da parte dei dati che la rete non vede
mai, e fermandosi quando è su **quelli** che i risultati smettono di migliorare.
Il prossimo capitolo è dedicato proprio a questo codice: lo riprenderemo riga
per riga.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Il **forward pass** è la catena di montaggio: i dati passano di postazione in
  postazione, ognuna li mescola con le proprie manopole, e all'ultima esce la
  previsione. La **loss** dice quanto quella previsione è lontana dalla verità.
- La **backpropagation** riparte dal fondo e chiede a ogni strato quanto ha
  contribuito all'errore: la risposta di uno serve a calcolare quella dello
  strato prima, e un solo giro all'indietro basta per tutte le manopole.
- La quota di colpa di una manopola **è** la sua pendenza, cioè di quanto
  cambierebbe l'errore muovendola di pochissimo. Sono la stessa cosa in due
  parole diverse, e l'elenco di tutte queste pendenze si chiama **gradiente**.
- Poi ogni manopola si sposta di poco nel verso che fa calare la loss, ed è la
  **discesa del gradiente**. Il passo è lungo quanto la pendenza moltiplicata
  per il **learning rate**, che è l'unica manopola in mano a noi: decide se si
  scende a valle, se si rimbalza o se non si arriva mai.
- Si procede a piccoli gruppi di esempi (i **mini-batch**), ripassando più volte
  su tutti i dati (le **epoche**). Nelle reti molto profonde il messaggio che
  torna indietro è un telefono senza fili: può affievolirsi fino a non insegnare
  più niente ai primi strati, oppure amplificarsi fino a diventare assordante e
  mandare tutto in tilt. Per questo una rete profonda va progettata apposta per
  far arrivare il messaggio integro fino in fondo, e un rimedio è già noto: la
  ReLU, che non appiattisce il segnale.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Il **forward pass** trasforma i dati in una previsione, strato per strato; la
  **loss** misura di quanto quella previsione sbaglia.
- La **backpropagation** è la regola della catena applicata all'indietro, cioè
  il **modo inverso** della differenziazione automatica: con una sola passata dà
  tutte le derivate, perché l'uscita è una sola e gli ingressi sono milioni. In
  tempo costa quanto l'andata; in **memoria** costa le attivazioni di tutti gli
  strati, che crescono con la profondità e con il batch.
- I pesi si aggiornano con la **discesa del gradiente**,
  $\theta \leftarrow \theta - \eta\,\nabla_\theta\mathcal{L}$; il **learning
  rate** $\eta$ dosa il passo, la cui lunghezza è
  $\eta\,\lVert\nabla_\theta\mathcal{L}\rVert$.
- Si lavora a **mini-batch** ed **epoche** (SGD), e il rumore del
  campionamento aiuta a staccarsi da selle e altipiani, non tanto dai minimi
  locali.
- Nelle reti profonde i gradienti possono **svanire o esplodere**, ma le due
  cose non sono simmetriche: $\sigma_{\max} < 1$ su ogni Jacobiana basta a
  garantire lo svanire, mentre per l'esplodere serve $\sigma_{\min} > 1$. Da qui
  ReLU, inizializzazioni accorte, batch norm, connessioni residue e, a valle,
  il *gradient clipping*.
```

`````
