# Backpropagation: come impara una rete

Nel 1986 tre ricercatori — David Rumelhart, Geoffrey Hinton e Ronald Williams —
pubblicano su *Nature* un articolo di poche pagine, *"Learning representations
by back-propagating errors"* {cite}`rumelhart1986learning`. L'algoritmo che descrivono non era del tutto nuovo
(Paul Werbos lo aveva già formulato nel 1974, e le radici affondano nella
differenziazione automatica degli anni '70), ma quel testo mostra al mondo come
una rete neurale possa correggersi da sola, un errore alla volta. È la stessa
ricetta con cui, ancora oggi, imparano modelli da miliardi di parametri.

L'idea sta in due movimenti, come un respiro. **In avanti** la rete produce una
risposta; **all'indietro** misura di quanto ha sbagliato e distribuisce la
"colpa" a ogni peso. Vediamo i due movimenti uno per uno.

## Il forward pass: dai dati all'uscita

Un esempio entra da sinistra e attraversa gli strati uno dopo l'altro, finché
l'ultimo strato non emette una previsione. Ogni strato prende ciò che riceve, lo
combina con i propri parametri e lo passa avanti.

`````{tab} Elementare

Immagina una catena di montaggio. Alla prima postazione arrivano i dati grezzi
(per esempio i pixel di una foto). Ogni postazione ha una fila di "manopole" —
i **pesi** — con cui mescola ciò che riceve, applica un piccolo filtro e
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
z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}, \qquad a^{[l]} = \sigma\!\left(z^{[l]}\right).
$$

Qui $W^{[l]}$ è la matrice dei pesi dello strato $l$, $b^{[l]}$ il vettore di
bias, $\sigma$ la funzione di attivazione (per esempio la ReLU, $\sigma(z)=\max(0,z)$),
$z^{[l]}$ la pre-attivazione e $a^{[l]}$ l'attivazione. L'uscita finale è la
previsione $\hat{y} = a^{[L]}$. Ogni strato non è che il prodotto matrice-vettore
già incontrato in algebra lineare, "avvolto" in una non linearità.

`````

## Quanto abbiamo sbagliato: la funzione di loss

La previsione da sola non basta: serve un numero che dica *quanto* la rete ha
sbagliato rispetto alla risposta giusta. Quel numero è la **loss**, e imparare
significa renderlo il più piccolo possibile.

`````{tab} Elementare

La loss è una distanza tra la risposta della rete e la verità. Se la casa vale
davvero 200.000 € e la rete ne prevede 170.000, l'errore è di 30.000: al
quadrato, per punire di più gli sbagli grossi, fa $0{,}9$ miliardi. Più la
previsione è vicina al vero, più la loss è piccola; se fossero identiche, la
loss sarebbe zero. Tutto l'addestramento è una caccia a quel numero più basso.

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

Pensa a una catena di responsabilità. L'errore nasce all'uscita, ma non è "colpa"
solo dell'ultimo strato: viene ereditato da quello prima, e da quello prima
ancora, fino all'inizio. La backpropagation parte dal fondo e chiede a ogni
strato: "quanto hai contribuito tu a questo errore?". La risposta di uno strato
serve a calcolare quella dello strato precedente — come un rimprovero che si
passa all'indietro lungo la fila ({numref}`fig-forward-backward`).

Un esempio in piccolo, con i numeri di prima: la rete prevede 170.000 € per la
casa che ne vale 200.000. I 30.000 € di errore vengono ripartiti tra i neuroni
dell'ultimo strato in proporzione a quanto ciascuno ha pesato sulla risposta:
chi ha contribuito con un peso grande eredita una colpa grande, chi ha
contribuito poco quasi niente. Poi ogni neurone gira la propria quota di colpa
ai neuroni dello strato prima, con lo stesso criterio, fino all'ingresso. Alla
fine ogni singolo peso sa in che direzione muoversi per far calare la loss.

`````

`````{tab} Superiore

Il meccanismo è la **regola della catena** applicata a ritroso (→ la deriviamo
nel capitolo di Matematica). Definiamo il segnale d'errore $\delta^{[l]}$ e lo
propaghiamo dallo strato d'uscita verso l'input:

$$
\delta^{[L]} = \nabla_{a^{[L]}} \mathcal{L} \;\odot\; \sigma'\!\left(z^{[L]}\right),
\qquad
\delta^{[l]} = \left(W^{[l+1]}\right)^{\!\top} \delta^{[l+1]} \;\odot\; \sigma'\!\left(z^{[l]}\right).
$$

Da $\delta^{[l]}$ ricaviamo i gradienti che ci servono per aggiornare i parametri:

$$
\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \delta^{[l]} \left(a^{[l-1]}\right)^{\!\top},
\qquad
\frac{\partial \mathcal{L}}{\partial b^{[l]}} = \delta^{[l]} .
$$

Il simbolo $\odot$ è il prodotto elemento per elemento (Hadamard) e $\sigma'$ la
derivata dell'attivazione. Il punto cruciale: ogni $\delta^{[l]}$ riusa
$\delta^{[l+1]}$, così un solo passaggio all'indietro basta a calcolare tutti i
gradienti. È questo che rende l'addestramento praticabile su reti enormi.

`````

```{figure} ../figures/backpropagation.gif
:name: fig-backpropagation-animata
:alt: Animazione di una rete 3-3-1. Prima un impulso percorre gli archi da sinistra a destra fino a un riquadro con la loss; poi un impulso torna indietro e, strato dopo strato, si compone il prodotto dei tre fattori della regola della catena.
:width: 85%

I due movimenti, uno dopo l'altro: il segnale va avanti fino alla loss, poi il
gradiente torna indietro e la regola della catena si compone **un fattore per
strato**.
```

La {numref}`fig-backpropagation-animata` mostra perché il costo del passaggio
all'indietro è confrontabile con quello in avanti: non si ricomincia da capo per
ogni peso, si aggiunge un fattore alla volta a un prodotto già calcolato.

## Aggiornare i pesi: discesa del gradiente e learning rate

Il gradiente indica, per ogni peso, la direzione in cui la loss *cresce*. Per
farla calare basta muoversi nel verso opposto, a piccoli passi.

`````{tab} Elementare

Immagina di essere su una collina, nella nebbia, e di voler scendere a valle.
Non vedi lontano, ma puoi sentire la pendenza sotto i piedi e fare un passo nella
direzione più ripida verso il basso. Ripeti, passo dopo passo. La lunghezza del
passo è il **learning rate**: se è troppo lungo scavalchi la valle e rimbalzi
avanti e indietro; se è troppo corto scendi lentissimo. Trovare un buon passo è
metà del mestiere.

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
ottimizzatori moderni (Momentum, RMSProp, **Adam** {cite}`kingma2015adam`)
adattano di fatto il passo per ciascun parametro, ma il cuore resta questo.

`````

## Mini-batch, epoche e SGD

Calcolare il gradiente su *tutti* i dati a ogni passo sarebbe accuratissimo ma
lentissimo. In pratica si divide il dataset in **mini-batch** (per esempio 32 o
64 esempi): per ciascuno si fa un forward, una backpropagation e un aggiornamento
dei pesi. Un giro completo su tutti i mini-batch è un'**epoca**; l'addestramento
ne conta decine o centinaia. Poiché ogni batch è un campione casuale dei dati, il
gradiente stimato è "rumoroso": per questo il metodo si chiama **discesa del
gradiente stocastica** (SGD, *Stochastic Gradient Descent*). Quel rumore, che
sembrerebbe un difetto, aiuta la rete a scavalcare i minimi poco profondi.

## Reti profonde: attenzione ai gradienti

Più la rete è profonda, più il gradiente deve viaggiare lontano per raggiungere i
primi strati — e lungo il tragitto può degradarsi.

`````{tab} Elementare

È il gioco del "telefono senza fili". Il messaggio (il gradiente) parte
dall'uscita e viene sussurrato all'indietro di strato in strato. Se a ogni
passaggio si affievolisce, arriva ai primi strati talmente flebile da non
insegnare loro nulla: sono i **gradienti che svaniscono** e la rete non impara.
Se invece a ogni passaggio si amplifica, arriva assordante e manda tutto in tilt:
i **gradienti che esplodono**. Le reti profonde vanno progettate per far arrivare
il messaggio integro fino in fondo.

`````

`````{tab} Superiore

Il gradiente verso i primi strati è un prodotto di molti fattori (le Jacobiane
strato per strato). Se questi fattori sono in media $<1$, il prodotto tende a
zero esponenzialmente con la profondità (*vanishing gradient*); se sono $>1$,
diverge (*exploding gradient*) — analisi resa celebre da Hochreiter (1991) e da
Bengio et al. (1994) sulle reti ricorrenti. I rimedi standard: attivazioni
**ReLU** al posto della sigmoide (Glorot et al., 2011), **inizializzazione**
accorta dei pesi (Xavier/He), **batch normalization**
{cite}`ioffe2015batch`, **connessioni residue** delle ResNet
{cite}`he2016deep` e, per l'esplosione, il
*gradient clipping*. Sono queste tecniche ad aver reso addestrabili reti da
centinaia di strati.

`````

## In pratica, con PyTorch

Nella pratica non implementiamo la backpropagation a mano: i framework la
eseguono per noi tramite differenziazione automatica (in PyTorch si chiama
*autograd*). A noi resta da dichiarare l'architettura, la loss e
l'ottimizzatore — e da scrivere il ciclo di addestramento, che in PyTorch
ricalca passo per passo il respiro descritto in questo capitolo.

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
tornare indietro il gradiente, `optimizer.step()` aggiorna i pesi. Venti
epoche, e la rete ha imparato. Il prossimo capitolo è dedicato proprio a
questo codice: lo riprenderemo riga per riga.

```{admonition} Da ricordare
:class: important
- Il **forward pass** trasforma i dati in una previsione, strato per strato; la
  **loss** misura di quanto quella previsione sbaglia.
- La **backpropagation** è la regola della catena applicata all'indietro:
  calcola in un solo passaggio quanto ogni peso contribuisce all'errore.
- I pesi si aggiornano con la **discesa del gradiente**,
  $\theta \leftarrow \theta - \eta\,\nabla_\theta\mathcal{L}$; il **learning
  rate** $\eta$ decide la dimensione del passo.
- Si lavora a **mini-batch** ed **epoche** (SGD); nelle reti profonde i gradienti
  possono **svanire o esplodere**, ed è per questo che esistono ReLU,
  inizializzazioni accorte, batch norm e connessioni residue.
```
