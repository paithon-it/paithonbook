# Come funziona l'addestramento avversario

Nel 2014, in un pub di Montréal, Ian Goodfellow discute con alcuni colleghi un problema che sembra senza uscita: come far *generare* a una rete neurale immagini nuove e credibili. Le idee sul tavolo gli paiono troppo macchinose. Quella sera torna a casa e prova un'intuizione opposta: e se, invece di dire a una rete quanto è "giusto" ogni pixel, la mettessimo semplicemente in competizione con un'altra rete che ha il compito di smascherarla? Il codice funziona quasi al primo tentativo. Nasce così l'idea delle **Generative Adversarial Networks** (GAN), descritta nel paper *Generative Adversarial Nets* {cite}`goodfellow2014generative`.

Il cuore dell'idea è un duello. Due reti neurali giocano una contro l'altra, e migliorano proprio perché si ostacolano a vicenda.

## Il falsario e il detective

Goodfellow usa un'immagine che vale la pena tenere a mente per tutto il capitolo. Immagina un **falsario** che stampa banconote contraffatte e un **detective** che deve distinguere le false dalle vere. All'inizio il falsario è maldestro e il detective lo smaschera subito. Ma ogni volta che viene scoperto, il falsario impara qualcosa e produce banconote un po' migliori; e ogni volta il detective affina il suo occhio. Se il gioco procede a lungo e ad armi pari, alla fine le banconote false diventano indistinguibili dalle vere.

Nelle GAN il falsario è il **generatore** $G$, il detective è il **discriminatore** $D$.

## Il generatore: dal rumore al dato

Il generatore parte dal nulla (letteralmente da un pugno di numeri casuali) e
deve costruire un dato che sembri autentico.

`````{tab} Elementare

Pensa a $G$ come a un artigiano bendato. Gli consegni una manciata di numeri estratti a caso (il "rumore"): è la sua materia prima, sempre diversa. Da quei numeri deve modellare qualcosa di sensato, per esempio l'immagine di un volto. All'inizio produce macchie informi. Con l'allenamento impara a trasformare quei numeri casuali in volti sempre più plausibili. Il punto è che numeri casuali diversi in ingresso danno volti diversi in uscita: è così che $G$ genera *varietà*, non una sola immagine ripetuta.

`````

`````{tab} Superiore

Il generatore è una funzione $G(z;\theta_G)$ parametrizzata da una rete neurale, che mappa un vettore di rumore $z\in\mathbb{R}^k$ nello spazio dei dati:

$$
z \sim p_z(z) \quad\longmapsto\quad \tilde{x} = G(z) .
$$

Il vettore $z$ è campionato da un *prior* semplice, tipicamente $p_z=\mathcal{N}(0, I)$ o uniforme. $G$ definisce implicitamente una distribuzione $p_G$ sullo spazio dei dati: spingendo campioni di $z$ attraverso la rete, otteniamo campioni di dati sintetici. L'obiettivo dell'addestramento è far convergere $p_G$ verso la distribuzione reale $p_{\text{dati}}$, senza mai scrivere esplicitamente la densità: da qui il nome di modello *generativo implicito*.

`````

## Il discriminatore: dal dato alla probabilità

Il discriminatore fa il mestiere opposto, e più familiare: è un classificatore binario.

`````{tab} Elementare

$D$ è il detective. Riceve un dato, a volte vero (pescato dal dataset), a
volte falso (sfornato da $G$), e deve rispondere a una sola domanda: *è
autentico?* La sua risposta è un numero tra $0$ e $1$, una specie di livello
di fiducia: vicino a $1$ significa "sono quasi certo che sia reale", vicino a
$0$ significa "quasi certo che sia un falso". Il suo mestiere è non farsi
ingannare.

`````

`````{tab} Superiore

Il discriminatore è una funzione $D(x;\theta_D)\in[0,1]$ che stima la probabilità che $x$ provenga dai dati reali anziché da $G$:

$$
D(x) \approx \Pr[\,x \sim p_{\text{dati}}\,] .
$$

È un classificatore addestrato con la consueta *cross-entropy* binaria: vuole assegnare $D(x)\to 1$ agli esempi reali e $D(G(z))\to 0$ a quelli sintetici. L'uscita in $[0,1]$ si ottiene applicando una sigmoide al punteggio grezzo (il *logit*) dell'ultimo strato; nell'implementazione, come d'abitudine in PyTorch, la sigmoide sarà assorbita dentro la loss (`nn.BCEWithLogitsLoss`) per stabilità numerica.

`````

Messi uno di fronte all'altro, $G$ e $D$ compongono l'architettura completa della GAN ({numref}`fig-gan-architettura`).

```{figure} ../figures/gan-architettura.svg
:name: fig-gan-architettura
:alt: "Schema di una GAN: un vettore di rumore entra nel generatore che produce un dato falso; dati falsi e dati reali entrano nel discriminatore che restituisce una probabilità reale/falso."
:width: 90%

Architettura di una GAN. Il generatore $G$ trasforma il rumore $z$ in un dato sintetico; il discriminatore $D$ riceve dati reali e sintetici e stima la probabilità che siano autentici. I gradienti dell'errore di $D$ retroagiscono su entrambe le reti.
```

## Il gioco minimax

Le due reti non ottimizzano due funzioni scollegate: condividono un'**unica** funzione di valore, che uno vuole massimizzare e l'altro minimizzare.

`````{tab} Elementare

Immagina un punteggio unico del gioco. Il detective guadagna punti ogni volta
che indovina; il falsario "vince" ogni volta che gli fa perdere punti. Quello
che è un bene per uno è un male per l'altro: è un gioco a somma (quasi) nulla.
Non esiste un traguardo fisso da raggiungere: esiste un *equilibrio*, il punto
in cui nessuno dei due riesce più a migliorare a spese dell'altro. Lì il
falsario è così bravo che il detective, per quanto si sforzi, può solo tirare
a indovinare.

`````

`````{tab} Superiore

$G$ e $D$ giocano un gioco minimax sulla funzione di valore

$$
\min_{G}\ \max_{D}\ V(D,G) =
\mathbb{E}_{x\sim p_{\text{dati}}}\big[\log D(x)\big]
+ \mathbb{E}_{z\sim p_z}\big[\log\big(1 - D(G(z))\big)\big] .
$$

Qui $p_{\text{dati}}$ è la distribuzione dei dati reali, $p_z$ il prior del rumore, $D(x)$ la probabilità stimata di autenticità e $G(z)$ il campione generato. $D$ massimizza $V$ (vuole $D(x)$ grande sui reali e $1-D(G(z))$ grande sui falsi); $G$ minimizza il secondo termine (vuole $D(G(z))\to 1$). Goodfellow dimostra che, con capacità e dati sufficienti, l'ottimo globale si raggiunge quando $p_G = p_{\text{dati}}$, e in quel punto $D(x)=\tfrac{1}{2}$ ovunque: il detective non sa più distinguere.

`````

## L'addestramento alternato

Poiché i due obiettivi sono in conflitto, non si può ottimizzarli in un colpo solo. Si procede **a turni**, con discesa del gradiente stocastica: un passo per $D$, un passo per $G$, e così via. Mentre si aggiorna una rete, i parametri dell'altra restano congelati.

```python
import torch
from torch import nn

# G e D sono due nn.Module, ciascuno con il proprio ottimizzatore
# (opt_G e opt_D): aggiornare l'uno non tocca i pesi dell'altro
criterio = nn.BCEWithLogitsLoss()        # sigmoide inclusa nella loss
uni  = torch.ones(batch_size, 1)         # etichette "reale"
zeri = torch.zeros(batch_size, 1)        # etichette "falso"

for epoca in range(n_epoche):
    for batch_reale in loader:
        # 1) Passo del discriminatore: distinguere reale da falso
        z = torch.randn(batch_size, dim_rumore)  # rumore
        falsi = G(z).detach()            # campioni sintetici, staccati da G
        loss_D = (criterio(D(batch_reale), uni)   # spinge D(x) -> 1
                  + criterio(D(falsi), zeri))     # spinge D(G(z)) -> 0
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # 2) Passo del generatore: ingannare D (si aggiorna solo G)
        z = torch.randn(batch_size, dim_rumore)
        loss_G = criterio(D(G(z)), uni)  # vuole D(G(z)) -> 1
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()
```

Due dettagli del codice contengono tutta la logica del gioco: il modo in cui i due allenamenti restano separati, e una piccola astuzia sulla lezione impartita al generatore.

`````{tab} Elementare

Primo dettaglio: quando si allena il detective sui falsi, quei falsi vengono
"staccati" dal falsario (è la parola `.detach()` nel codice): il giudizio
serve a correggere solo chi giudica, non chi ha dipinto. E quando è il turno
del falsario, l'aggiornamento tocca solo i pesi suoi: ognuno impara nel
proprio turno, come da regolamento; è il "congelamento" descritto sopra, che
in PyTorch si scrive in una parola.

Secondo dettaglio: nel suo turno, il falsario chiede al detective di trattare
i propri falsi come "reali" e impara da quanto il verdetto se ne discosta. È
una versione più *generosa* del gioco: dà al falsario lezioni chiare proprio
all'inizio, quando i suoi quadri sono pessimi e il detective li respinge con
totale sicurezza. Senza questo trucco (già suggerito nel paper del 2014), il
principiante non riceverebbe quasi nessun insegnamento e resterebbe maldestro
per sempre.

`````

`````{tab} Superiore

Nel passo di $D$, il `.detach()` stacca i campioni sintetici dal grafo di $G$:
il giudizio su di essi corregge solo il discriminatore, e nessun gradiente
risale fino al generatore. Nel passo di $G$ accade il contrario: il gradiente
attraversa $D$, ma `opt_G` aggiorna solo i pesi del generatore. E c'è una
scelta nascosta nella riga `criterio(D(G(z)), uni)`: chiedere che i falsi
siano etichettati "reale" equivale a **massimizzare** $\log D(G(z))$, invece
di minimizzare $\log(1-D(G(z)))$ come nella formula minimax. Le due
formulazioni hanno lo stesso ottimo, ma la prima fornisce gradienti più forti
proprio all'inizio, quando $G$ è pessimo e $D(G(z)) \approx 0$ farebbe
saturare l'obiettivo originale: è il *non-saturating loss* già suggerito nel
paper del 2014.

`````

## Quando il duello si inceppa

L'eleganza teorica delle GAN convive con una fama meritata di addestramento capriccioso. Tre problemi ricorrono.

`````{tab} Elementare

- **Instabilità.** I due giocatori si rincorrono senza mai fermarsi: migliora uno, l'altro peggiora, e il punteggio oscilla invece di stabilizzarsi. È come due lottatori troppo forti che si sbilanciano a vicenda.
- **Mode collapse.** Il falsario scopre *un solo* falso che inganna sempre il detective e si limita a rifarlo. Risultato: $G$ genera sempre la stessa immagine (o pochissime varianti), buttando via tutta la varietà dei dati reali.
- **Mancata convergenza.** A volte il gioco non trova mai un equilibrio: le immagini oscillano, degenerano, o non migliorano più.

`````

`````{tab} Superiore

- **Instabilità.** L'ottimizzazione simultanea di un gioco minimax non equivale a minimizzare una singola funzione: la dinamica può divergere o entrare in cicli limite. Se $D$ diventa troppo accurato, $D(G(z))\to 0$ e i gradienti verso $G$ si annullano (*vanishing gradients*); se è troppo debole, non fornisce segnale utile.
- **Mode collapse.** $G$ mappa molti $z$ diversi su una stessa uscita $\tilde{x}$: $p_G$ collassa su pochi modi di $p_{\text{dati}}$. Formalmente minimizza il proprio obiettivo locale ignorando la copertura dell'intera distribuzione.
- **Mancata convergenza.** L'equilibrio di Nash del gioco non è garantito raggiungibile con la sola discesa del gradiente; i parametri possono orbitare indefinitamente attorno all'ottimo senza stabilizzarsi.

`````

## Accorgimenti pratici (cenni)

La ricerca successiva ha prodotto una cassetta degli attrezzi per domare l'addestramento. Solo alcuni titoli, che approfondiremo: adottare architetture convoluzionali disciplinate come nelle **DCGAN** {cite}`radford2016unsupervised`; cambiare la funzione di costo con la **Wasserstein GAN** {cite}`arjovsky2017wasserstein`, che sostituisce la probabilità con una distanza dal comportamento più regolare, spesso accoppiata al *gradient penalty*; usare *label smoothing*, *minibatch discrimination* e aggiornamenti bilanciati per tenere in equilibrio i due giocatori. Nessuno di questi trucchi è una bacchetta magica: l'addestramento avversario resta un'arte oltre che una scienza, ma è proprio da questa tensione che nascono i risultati più sorprendenti del deep learning generativo.

```{admonition} Da ricordare
:class: important
- Una GAN è un **duello** tra due reti: il generatore $G$ trasforma rumore in dati sintetici, il discriminatore $D$ stima la probabilità che un dato sia reale.
- Condividono un'unica **funzione di valore minimax**: $G$ la minimizza, $D$ la massimizza; l'equilibrio si ha quando $p_G = p_{\text{dati}}$ e $D(x)=\tfrac12$.
- L'addestramento è **alternato** e notoriamente instabile: attenzione a *mode collapse*, gradienti che svaniscono e mancata convergenza.
```
