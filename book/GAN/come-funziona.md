# Come funziona l'addestramento avversario

Dell'idea nata quella sera a Montréal si è già detto in apertura di capitolo; qui la smontiamo pezzo per pezzo.

Il modo consueto di insegnare a una rete a produrre un'immagine è dirle, punto per punto, quanto la sua uscita si discosta da un'immagine vera che le mettiamo accanto: questo puntino doveva essere più scuro, quest'altro più chiaro. Il metodo funziona finché l'originale ce l'abbiamo sotto mano; ma per *inventare* un'immagine nuova l'originale non esiste, e non c'è niente con cui fare il confronto. Le **Generative Adversarial Networks** (GAN), descritte nel paper *Generative Adversarial Nets* {cite}`goodfellow2014generative`, cambiano giudice: al posto del confronto punto per punto mettono una seconda rete, il cui unico mestiere è smascherare la prima.

## Il falsario e l'esperto, tradotti in reti

La metafora dell'apertura si traduce così in due reti con un nome: il falsario che dipinge quadri contraffatti è il **generatore** $G$, l'esperto d'arte che deve smascherarlo è il **discriminatore** $D$. Da qui in avanti li chiamiamo con questi nomi, e ci occupiamo di come sono fatti dentro e di come si allenano a vicenda.

## Il generatore: dal rumore al dato

Il generatore parte dal nulla, letteralmente da un pugno di numeri estratti a
caso, che d'ora in poi chiamiamo **rumore** (è il termine tecnico, e non ha
niente a che vedere con il suono: dice solo che quei numeri non significano
niente in partenza), e deve costruire un dato che sembri autentico.

`````{tab} Elementare

Pensa a $G$ come a un artigiano bendato: bendato perché i quadri autentici non
li vedrà mai, nemmeno uno, e fra poco vedremo che è una scelta di progetto e
non una dimenticanza. Gli consegni una manciata di numeri estratti a caso (il
rumore): è la sua materia prima, sempre diversa. Da quei numeri deve modellare
qualcosa di sensato, per esempio l'immagine di un volto. All'inizio produce
macchie informi. Con l'allenamento impara a trasformare quei numeri casuali in
volti sempre più plausibili. Il punto è che numeri casuali diversi in ingresso
danno volti diversi in uscita: è così che $G$ genera *varietà*, non una sola
immagine ripetuta.

`````

`````{tab} Superiore

Il generatore è una funzione $G(\mathbf{z};\theta_G)$ parametrizzata da una rete neurale, che mappa un vettore di rumore $\mathbf{z}\in\mathbb{R}^k$ nello spazio dei dati:

$$
\mathbf{z} \sim p_z(\mathbf{z}) \quad\longmapsto\quad \tilde{\mathbf{x}} = G(\mathbf{z}) .
$$

Il vettore $\mathbf{z}$ è campionato da un *prior* semplice, tipicamente $p_z=\mathcal{N}(0, \mathbf{I})$ o uniforme. $G$ definisce implicitamente una distribuzione $p_G$ sullo spazio dei dati: spingendo campioni di $\mathbf{z}$ attraverso la rete, otteniamo campioni di dati sintetici. L'obiettivo dell'addestramento è far convergere $p_G$ verso la distribuzione reale $p_{\text{dati}}$, senza mai scrivere esplicitamente la densità: da qui il nome di modello *generativo implicito*.

`````

## Il discriminatore: dal dato alla probabilità

Il discriminatore fa il mestiere opposto, e più familiare: è un classificatore binario.

`````{tab} Elementare

$D$ è l'esperto d'arte. Riceve un dato, a volte vero (pescato dal dataset), a
volte falso (sfornato da $G$), e deve rispondere a una sola domanda: *è
autentico?* La sua risposta è un numero tra $0$ e $1$, una specie di livello
di fiducia: vicino a $1$ significa "sono quasi certo che sia reale", vicino a
$0$ significa "quasi certo che sia un falso". Il suo mestiere è non farsi
ingannare.

`````

`````{tab} Superiore

Il discriminatore è una funzione $D(\mathbf{x};\theta_D)\in[0,1]$ che stima la probabilità che $\mathbf{x}$ provenga dai dati reali anziché da $G$. La formalizzazione passa da una mistura: il campione arriva metà delle volte dal dataset e metà dal generatore, e $D$ stima la probabilità *a posteriori* che la sorgente sia quella reale, visto il campione:

$$
D(\mathbf{x}) \approx P(\text{reale} \mid \mathbf{x}) .
$$

È un classificatore addestrato con la consueta *cross-entropy* binaria: vuole assegnare $D(\mathbf{x})\to 1$ agli esempi reali e $D(G(\mathbf{z}))\to 0$ a quelli sintetici. L'uscita in $[0,1]$ si ottiene applicando una sigmoide al punteggio grezzo (il *logit*) dell'ultimo strato; nell'implementazione, come d'abitudine in PyTorch, la sigmoide sarà assorbita dentro la loss (`nn.BCEWithLogitsLoss`) per stabilità numerica.

`````

Messi uno di fronte all'altro, $G$ e $D$ compongono l'architettura completa della GAN ({numref}`fig-gan-architettura`).

```{figure} ../figures/gan-architettura.svg
:name: fig-gan-architettura
:alt: "Schema di una GAN: un vettore di rumore entra nel generatore che produce un dato falso; dati falsi e dati reali entrano nel discriminatore che restituisce una probabilità reale/falso."
:width: 90%

Architettura di una GAN. Il generatore trasforma il rumore in un dato
sintetico; il discriminatore riceve dati reali e dati sintetici e stima, per
ciascuno, quanto è probabile che sia autentico. La freccia tratteggiata in
basso è la correzione che dal giudizio torna indietro verso il generatore.
```

Le due figure disegnano lo stesso circuito, ma servono a guardare due cose
diverse: {numref}`fig-gan-architettura` mette i nomi ai pezzi e alle grandezze
che si scambiano, {numref}`fig-gan-circuito` ripercorre il giro completo, ed è
la figura da tenere sott'occhio per la sezione che segue.

```{figure} ../figures/gan-2014.svg
:name: fig-gan-circuito
:alt: "Circuito di una GAN: il rumore casuale z entra nel generatore G, che produce un campione falso; il discriminatore D riceve sia campioni reali dal dataset sia il falso, e per ciascuno decide se è vero o finto; il suo verdetto torna indietro come segnale di apprendimento sia a D sia a G."
:width: 96%

Il circuito completo. La freccia di ritorno verso il generatore è il punto:
G non vede mai i dati reali, impara soltanto da quanto bene ha ingannato D.
```

Vale la pena fermarsi su cosa {numref}`fig-gan-circuito` *non* collega. Fra i
dati reali e il generatore non passa nessuna freccia: G non li copia né li
confronta, e tutto ciò che sa del mondo gli arriva filtrato dal giudizio del
discriminatore. È una scelta di progetto elegante e, come vedremo, fragile.

Elegante per una ragione che conviene dire ad alta voce, perché è la
mossa più furba del disegno: se il falsario vedesse i quadri autentici, la
strategia migliore per ingannare l'esperto sarebbe **ricopiarne uno**, e
avremmo costruito una macchina che restituisce ciò che le abbiamo dato. Tenerlo
bendato è ciò che lo costringe a inventare.

E allora da dove entra la verità, se metà del sistema il mondo non lo vede
mai? Entra dall'altra metà. L'esperto, a ogni turno, guarda anche dei quadri
autentici presi dal dataset e viene corretto su quelli: è lui l'ancora, ed è
l'unica cosa che impedisce ai due di mettersi d'accordo su una schifezza.
Immaginiamo per un attimo di togliergliela, cioè di allenare l'esperto solo sui
falsi: gli basterebbe rispondere "falso" a tutto, il falsario non avrebbe
nessuna strada per accontentarlo, e il duello girerebbe a vuoto. Il senso della
freccia che manca è tutto qui: la realtà entra nel sistema una volta sola, dal
lato di $D$, e da lì raggiunge $G$ di rimbalzo.

## Il gioco minimax

Le due reti non ottimizzano due funzioni scollegate: condividono un'**unica
funzione di valore**, cioè un punteggio solo per tutta la partita, che uno vuole
tirare più in alto possibile e l'altro più in basso possibile. È il *minimax*
del titolo, contrazione di *minimo* e *massimo*: il più piccolo dei risultati
più grandi, che è ciò a cui punta chi gioca contro qualcuno che farà del suo
meglio per rovinargli la mossa.

`````{tab} Elementare

Immagina un punteggio unico del gioco. L'esperto guadagna punti ogni volta
che indovina; il falsario "vince" ogni volta che gli fa perdere punti. Quello
che è un bene per uno è un male per l'altro: è un gioco a somma zero.
Non esiste un traguardo fisso da raggiungere: esiste un *equilibrio*, il punto
in cui nessuno dei due riesce più a migliorare a spese dell'altro. Lì il
falsario è così bravo che l'esperto, per quanto si sforzi, può solo tirare
a indovinare.

`````

`````{tab} Superiore

$G$ e $D$ giocano un gioco minimax sulla funzione di valore

$$
\min_{G}\ \max_{D}\ V(D,G) =
\mathbb{E}_{\mathbf{x}\sim p_{\text{dati}}}\big[\log D(\mathbf{x})\big]
+ \mathbb{E}_{\mathbf{z}\sim p_z}\big[\log\big(1 - D(G(\mathbf{z}))\big)\big] .
$$

Qui $p_{\text{dati}}$ è la distribuzione dei dati reali, $p_z$ il prior del rumore, $D(\mathbf{x})$ la probabilità stimata di autenticità e $G(\mathbf{z})$ il campione generato. $D$ massimizza $V$ (vuole $D(\mathbf{x})$ grande sui reali e $1-D(G(\mathbf{z}))$ grande sui falsi); $G$ minimizza il secondo termine (vuole $D(G(\mathbf{z}))\to 1$). La dimostrazione di Goodfellow sta in due passaggi che vale la pena avere sott'occhio. Primo: per $G$ fissato, il discriminatore che massimizza $V$ è

$$
D^*(\mathbf{x}) = \frac{p_{\text{dati}}(\mathbf{x})}{p_{\text{dati}}(\mathbf{x}) + p_G(\mathbf{x})},
$$

cioè proprio l'ottimo bayesiano della mistura descritta sopra. Secondo: sostituendo $D^*$ in $V$ si ottiene $V(D^*,G) = -\log 4 + 2\,\mathrm{JSD}\big(p_{\text{dati}} \,\|\, p_G\big)$, dove $\mathrm{JSD}$ è la divergenza di Jensen-Shannon, non negativa e nulla se e solo se le due distribuzioni coincidono. L'obiettivo ideale ha dunque minimo globale esattamente in $p_G = p_{\text{dati}}$, e lì $D^*(\mathbf{x})=\tfrac{1}{2}$ sul supporto dei dati: l'esperto non sa più distinguere.

Questa però è una *caratterizzazione* dell'ottimo, non una promessa di arrivarci, e le due cose vanno tenute separate. La prova di convergenza del paper suppone che a ogni passo $D$ raggiunga il proprio ottimo dato $G$, e soprattutto che a muoversi sia la densità $p_G$, dove $V$ è convessa; nell'addestramento vero si muovono i parametri $\theta_G$ di una rete, e lì la convessità che regge la dimostrazione non c'è più. Lo scrivono gli autori stessi, subito dopo la dimostrazione: usare un percettrone multistrato per definire $G$ introduce molti punti critici nello spazio dei parametri, e le reti funzionano bene in pratica «despite their lack of theoretical guarantees».

`````

Di questo gioco non esiste un fotogramma che lo racconti: quello che conta è
il **movimento**, il falso che si avvicina al vero e l'esperto che perde terreno
mentre succede. {numref}`fig-gan-inseguimento` lo mette in scena su un caso
minuscolo, una sola dimensione e una campana per parte.

```{figure} ../figures/gan-inseguimento.svg
:name: fig-gan-inseguimento
:alt: "Due pannelli sovrapposti. Sopra, la campana dei dati veri sta ferma al centro mentre quella del generatore, all'inizio spostata a sinistra e più larga, si sposta e si stringe fino a coprirla. Sotto, la curva del verdetto parte a gobba, alta dove prevalgono i dati veri e bassa dove prevale il generatore, e si appiattisce fino a diventare la retta orizzontale a un mezzo; è disegnata solo nel tratto in cui almeno una delle due campane ha densità apprezzabile."
:width: 92%

Sopra: i dati veri stanno fermi, il generatore li insegue. Sotto: il verdetto
migliore che l'esperto possa dare contro quel generatore. Finché le due
campane sono separate il verdetto è netto; quando si sovrappongono diventa un
mezzo dappertutto, cioè una moneta lanciata in aria.
```

Il pannello di sotto non è disegnato a mano: è la formula di $D^*$ valutata
sulle due curve del pannello di sopra, punto per punto. E la formula dice una
cosa che il disegno rende evidente: il verdetto guarda il **rapporto** fra le
due altezze, non la loro distanza. Dove le due curve sono alte uguali il
verdetto sta a un mezzo, anche se lì di roba ce n'è pochissima; dove una
prevale sull'altra si allontana da un mezzo, anche se le due sono vicine.

Due cose si muovono insieme, ed è utile guardarle separatamente. La prima è il
punto in cui il verdetto vale esattamente un mezzo, cioè il confine oltre il
quale l'esperto cambia idea: parte da circa $-0{,}8$ e scivola verso destra
mentre il generatore avanza. La seconda è l'altezza della gobba, cioè quanto
l'esperto è sicuro nel suo terreno migliore: parte da $0{,}93$ e scende a
$0{,}74$, a $0{,}64$, fino a $0{,}50$. Il confine si sposta e nel frattempo si
sgonfia, e quando la gobba tocca il mezzo il confine non c'è più: non è che
l'esperto abbia sbagliato posto, è che non c'è più un posto giusto.

C'è anche una ragione per cui la curva di sotto **non attraversa tutto il
riquadro**. Il rapporto fra due densità è definito ovunque, ma dove non c'è né
vero né falso non c'è niente da giudicare, e disegnarlo lì direbbe al lettore
«certamente falso» in una regione vuota. È lo stesso perimetro che la
dimostrazione si dà quando conclude $D^*(\mathbf{x}) = \tfrac{1}{2}$ **sul supporto dei
dati**.

## L'addestramento alternato

Poiché i due obiettivi sono in conflitto, non si può ottimizzarli in un colpo
solo. Si procede **a turni**: un passo per $D$, un passo per $G$, e così via,
con la discesa del gradiente stocastica già incontrata nei capitoli precedenti
(*stocastica* vuol dire che a ogni passo si guarda un pugno di esempi presi a
caso, non tutti insieme). Mentre si aggiorna una rete, i parametri dell'altra
restano fermi.

Nel codice, il punteggio unico del gioco si spezza in due conti dell'errore,
uno per rete: sono le due **loss** che si vedono qui sotto, `loss_D` e
`loss_G`. Non sono due giochi diversi, sono le due facce dello stesso
punteggio, ciascuna scritta dal punto di vista di chi la deve far scendere; e
d'ora in avanti, quando parleremo di "loss", parleremo di queste.

Il ciclo completo sta in una ventina di righe. Chi non legge Python può
passare oltre senza rimetterci: le tre cose che contano sono spiegate subito
sotto, e per capirle il codice non serve.

```{code-block} python
:class: pt-non-eseguibile

import torch
from torch import nn

# G e D sono due nn.Module, ciascuno con il proprio ottimizzatore
# (opt_G e opt_D): aggiornare l'uno non tocca i pesi dell'altro
criterio = nn.BCEWithLogitsLoss()        # sigmoide inclusa nella loss

for epoca in range(n_epoche):
    for batch_reale in loader:
        n = batch_reale.size(0)          # quanti esempi ci sono in questo gruppo
        uni  = torch.ones(n, 1)          # etichette "reale"
        zeri = torch.zeros(n, 1)         # etichette "falso"

        # 1) Passo del discriminatore: distinguere reale da falso
        z = torch.randn(n, dim_rumore)   # rumore
        falsi = G(z).detach()            # campioni sintetici, staccati da G
        loss_D = (criterio(D(batch_reale), uni)   # spinge D(x) -> 1
                  + criterio(D(falsi), zeri))     # spinge D(G(z)) -> 0
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # 2) Passo del generatore: ingannare D (si aggiorna solo G)
        z = torch.randn(n, dim_rumore)
        loss_G = criterio(D(G(z)), uni)  # vuole D(G(z)) -> 1
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()
```

Tre punti di questo ciclo meritano di essere guardati da vicino: che cosa
esattamente torna indietro dall'esperto al falsario, come mai i due
allenamenti non si mescolano, e una piccola astuzia sulla lezione impartita al
generatore. Il primo è il meccanismo centrale del capitolo, e conviene
partire da lì.

### Che cosa torna indietro

`````{tab} Elementare

Riprendiamo la domanda lasciata in sospeso: quando l'esperto boccia un quadro,
che cosa impara il falsario? Se ciò che torna indietro fosse il verdetto
("falso"), non imparerebbe niente di utile: saprebbe di aver sbagliato, e
basta. Ma il ritorno non è il verdetto.

L'esperto è fatto in modo che gli si possa chiedere qualcosa di più fine di un
giudizio, e cioè, per ogni singolo puntino del quadro: *se questo puntino
fosse un po' più chiaro, il tuo giudizio salirebbe o scenderebbe, e di quanto?*
La risposta a quella domanda, posta per tutti i puntini insieme, è una lista
lunga quanto il quadro: per ciascun puntino, da che parte tirare e con quanta
forza. È questo che torna indietro. Non un voto, ma una correzione con un
verso, punto per punto: l'esperto non dice "falso", dice "falso, e soprattutto
per via di *questo* qui".

Ed è anche la risposta alla domanda gemella: perché l'esperto dev'essere una
rete, e non una persona o un elenco di regole? Una persona darebbe lo stesso
verdetto, e magari un consiglio a parole ("la firma non convince"); quello che
non può dare è la lista. A un critico d'arte non si può chiedere di quanto
spostare ciascuno dei due milioni di puntini di una fotografia. A una rete sì,
perché una rete è una formula, e a una formula si può sempre domandare come
cambia il risultato se si muove un ingresso. Il falsario, dal canto suo, sa
come i propri numeri di partenza diventano puntini: attaccando le due cose una
all'altra ottiene ciò che gli serve davvero, cioè come ritoccare sé stesso.

`````

`````{tab} Superiore

La quantità che il passo di $G$ propaga si scrive con la regola della catena,
spezzata nel punto in cui le due reti si toccano:

$$
\frac{\partial \mathcal{L}_G}{\partial \theta_G} =
\frac{\partial \mathcal{L}_G}{\partial \tilde{\mathbf{x}}} \cdot
\frac{\partial \tilde{\mathbf{x}}}{\partial \theta_G},
\qquad \tilde{\mathbf{x}} = G(\mathbf{z}) .
$$

Il primo fattore è il gradiente della loss del generatore rispetto al **dato
generato**, e vive nello spazio dei dati: ha una componente per ogni numero di
$\tilde{\mathbf{x}}$ (per un'immagine, una per pixel e per canale). È lì che sta la
differenza fra un'informazione utile e un'informazione inutile: il verdetto
$D(\tilde{\mathbf{x}})$ è uno scalare, mentre $\partial \mathcal{L}_G / \partial
\tilde{\mathbf{x}}$ è un vettore che indica, componente per componente, in che verso
spostare il dato perché il verdetto salga. Il secondo fattore è la jacobiana
del generatore rispetto ai propri parametri, e con $D$ non ha niente a che
vedere: è la parte che il falsario conosce già di sé.

Da qui discende un **requisito di progetto**, non un dettaglio di
implementazione: $D$ dev'essere derivabile rispetto al proprio ingresso. Un
giudice umano, o un programma a regole, darebbe lo stesso verdetto e nessun
vettore; la catena si spezzerebbe nel primo fattore e a $G$ non arriverebbe
niente. È anche la ragione per cui le GAN sui dati discreti (il testo, prima di
tutto) sono sempre state faticose: se $\tilde{\mathbf{x}}$ è una sequenza di simboli
campionati, il secondo fattore non esiste, e per aggirare la rottura servono
stimatori a punteggio in stile REINFORCE o rilassamenti continui come
Gumbel-softmax.

Un esempio minimo dà la misura della differenza fra le due informazioni. Con un
$D$ giocattolo su un dato di quattro numeri, il verdetto è
$D(\tilde{\mathbf{x}}) = 0{,}427161$: un solo numero, che dice "propendo per il falso" e
nient'altro. Il gradiente sullo stesso dato è invece

$$
\frac{\partial \mathcal{L}_G}{\partial \tilde{\mathbf{x}}} =
[\,-0{,}02442,\ +0{,}07015,\ -0{,}02293,\ -0{,}02082\,],
$$

quattro numeri che dicono di alzare la prima, la terza e la quarta componente e
di abbassare nettamente la seconda. Muovendo il dato di mezzo passo in senso
opposto al gradiente il verdetto sale a $0{,}428544$: piccolo, perché il passo è
piccolo, ma nella direzione voluta, e ottenuto senza che nessuno abbia mai
mostrato al generatore un dato autentico.

`````

Le due sezioni che seguono nel capitolo dipendono da questo passaggio più di
quanto sembri. I "gradienti che svaniscono" di cui si parlerà fra poco sono
gradienti *di questo tipo*, e il *gradient penalty* citato in fondo alla
sezione è una penalità sulla ripidità di $D$ rispetto al proprio ingresso:
entrambe le cose riguardano il vettore che torna indietro, non il verdetto.

### Gli altri due dettagli

`````{tab} Elementare

Prima dei due dettagli, una parola sulla riga con la `n`: i dati non si danno
in pasto alla rete uno per volta ma a gruppetti, e l'ultimo gruppo di ogni giro
può risultare più corto degli altri (se gli esempi sono $1000$ e i gruppi da
$64$, l'ultimo ne contiene $40$). La `n` conta quanti esempi ci sono davvero nel
gruppo di turno, e serve a preparare esattamente altrettante etichette "vero" e
"falso".

Primo dettaglio: ciascuno dei due impara solo nel proprio turno, e questo è
garantito dal fatto che il falsario e l'esperto hanno **due allenatori
distinti** (`opt_G` e `opt_D` nel codice), ciascuno dei quali conosce soltanto i
pesi del proprio allievo. È il "congelamento" descritto sopra, e non è un
trucco: è il modo in cui i due allenatori sono stati messi su. La parola
`.detach()`, che
compare quando si allena l'esperto sui falsi, fa una cosa più modesta di quanto
si legga di solito: dice di non calcolare nemmeno la correzione per il
falsario, dato che in quel turno verrebbe comunque buttata via. Non serve a
tenerli separati (a quello bastano i due allenatori), serve a non sprecare
lavoro; su reti grandi il risparmio è però notevole.

Secondo dettaglio: nel suo turno, il falsario chiede all'esperto di trattare
i propri falsi come "reali" e impara da quanto il verdetto se ne discosta. È
una versione più *generosa* del gioco, e il motivo per cui serve è quello
appena visto. La correzione non è il voto, è **di quanto il voto cambierebbe**:
quando l'esperto è sicurissimo che il quadro sia falso, il suo giudizio è
schiacciato contro il fondo della scala, e un piccolo miglioramento del quadro
non lo sposta di una virgola. Un principiante corretto così è come uno studente
che prende zero a ogni compito senza mai sapere quale zero fosse meno grave:
non ha modo di capire se l'ultimo ritocco andava nella direzione giusta.
Chiedendo invece all'esperto quanto manca perché quel quadro passi per vero, si
ottiene una correzione che resta forte anche all'inizio. Il trucco è già
suggerito nel paper del 2014; il prezzo lo vedremo fra poco.

`````

`````{tab} Superiore

La separazione fra i due allenamenti la garantiscono i **due ottimizzatori**:
`opt_D` non conosce i parametri di $G$ e viceversa, quindi nessuno dei due passi
può toccare i pesi dell'altra rete. Il `.detach()` nel passo di $D$ aggiunge un
risparmio: stacca i campioni sintetici dal grafo di $G$, così il gradiente
attraverso il generatore non viene nemmeno calcolato. In questo ciclo, senza
`.detach()`, quel gradiente verrebbe calcolato, si depositerebbe in `.grad` e
sarebbe poi azzerato da `opt_G.zero_grad()` prima di essere usato: il risultato
numerico è identico (verificato: pesi finali di $G$ uguali bit a bit), ma su
una rete grande si paga un passaggio all'indietro intero per niente. Attenzione
però che l'innocuità dipende dall'ordine delle righe: in una variante che
azzeri i gradienti in cima all'iterazione, o che legga `.grad` fra i due passi,
`.detach()` torna necessario.

C'è poi una scelta nascosta nella riga `criterio(D(G(z)), uni)`: chiedere che i
falsi siano etichettati "reale" equivale a **massimizzare** $\log D(G(\mathbf{z}))$,
invece di minimizzare $\log(1-D(G(\mathbf{z})))$ come nella formula minimax. Le due
formulazioni hanno lo stesso punto fisso, e questa fornisce gradienti più
forti proprio all'inizio, quando $G$ è pessimo e $D(G(\mathbf{z})) \approx 0$ farebbe
saturare l'obiettivo originale: è il *non-saturating loss* già suggerito nel
paper del 2014.

Non sono però lo stesso gioco, ed è meglio dirlo esplicitamente perché il
capitolo ha appena costruito due sezioni sull'idea di un punteggio unico: con
questa formulazione il gioco **non è più a somma zero** e non si lascia più
scrivere con un'unica funzione di valore, come nota Goodfellow stesso nel
proprio tutorial NIPS {cite}`goodfellow2016nips`. Arjovsky e Bottou
{cite}`arjovsky2017towards` mostrano
che il gradiente ricevuto qui dal generatore è quello di
$\mathrm{KL}\big(p_G \,\|\, p_{\text{dati}}\big) - 2\,\mathrm{JSD}\big(p_G \,\|\,
p_{\text{dati}}\big)$: una divergenza di Kullback-Leibler rovesciata, più un
termine che spinge le due distribuzioni ad allontanarsi. Conviene tenerlo a
mente fra poco, quando parleremo di *mode collapse*.

`````

## Quando il duello si inceppa

L'eleganza teorica delle GAN convive con una fama meritata di addestramento capriccioso. Tre problemi ricorrono.

`````{tab} Elementare

- **Instabilità.** I due giocatori si rincorrono senza mai fermarsi: migliora
  uno, l'altro peggiora, e il punteggio oscilla invece di stabilizzarsi.
  All'inizio del capitolo avevamo detto che i due "si perfezionano a vicenda",
  ed è ancora vero: la differenza sta nel passo. Finché ciascuno insegue
  l'altro con ritocchi piccoli, ogni miglioramento resta acquisito e
  l'equilibrio si sposta un poco alla volta; se i passi sono troppo grandi,
  ogni correzione disfa la precedente e nessuno dei due consolida niente. Due
  lottatori che si sbilanciano a vicenda invece di allenarsi.
- **Mode collapse.** Il falsario scopre *un solo* falso che inganna sempre
  l'esperto e si limita a rifarlo. Risultato: $G$ genera sempre la stessa
  immagine (o pochissime varianti), buttando via tutta la varietà dei dati
  reali. Verrebbe da chiedersi come mai l'esperto non si insospettisca nel
  vedere sempre lo stesso quadro: il fatto è che li guarda **uno per volta**, e
  uno per volta quel falso è convincente. Per smascherare la ripetizione
  bisognerebbe fargli guardare un gruppo intero in blocco, ed è uno degli
  accorgimenti che vedremo in fondo alla sezione.
- **Mancata convergenza.** A volte il gioco non trova mai un equilibrio: le immagini oscillano, degenerano, o non migliorano più.

`````

`````{tab} Superiore

- **Instabilità.** L'ottimizzazione simultanea di un gioco minimax non equivale a minimizzare una singola funzione: la dinamica può divergere o entrare in cicli limite. Se $D$ diventa troppo accurato si ha $D(G(\mathbf{z}))\to 0$, e con l'obiettivo minimax originale questo annulla i gradienti verso $G$ (*vanishing gradients*); la non-saturating loss vista sopra scongiura l'annullamento, ma con un discriminatore quasi ottimo lo paga in aggiornamenti instabili e ad alta varianza {cite}`arjovsky2017towards`. Se invece $D$ è troppo debole, non fornisce segnale utile.

  Che $D$ diventi "troppo accurato" non è però un incidente di dosaggio, ed è
  un punto che cambia il rimedio. Arjovsky e Bottou mostrano che $p_G$, essendo
  l'immagine di uno spazio di rumore a poche decine o centinaia di dimensioni,
  vive su una varietà di dimensione bassa immersa nello spazio dei dati: con
  $\mathbf{z} \in \mathbb{R}^{100}$ e immagini $1024\times1024$ a colori, il supporto di
  $p_G$ ha dimensione al più $100$ dentro $\mathbb{R}^{3\,145\,728}$. Due
  varietà così hanno supporti quasi certamente disgiunti (o intersecantisi in
  un insieme di misura nulla), un discriminatore perfetto esiste, e su supporti
  disgiunti la $\mathrm{JSD}$ vale $\log 2$ **qualunque** sia la distanza fra le
  due distribuzioni. Il gradiente non è piccolo: è nullo, e resta nullo mentre
  $G$ si avvicina. Alternare meglio i turni non lo risolve, ed è da qui che
  nasce l'idea di cambiare misura, cioè la Wasserstein GAN.
- **Mode collapse.** $G$ mappa molti $\mathbf{z}$ diversi su una stessa uscita
  $\tilde{\mathbf{x}}$: $p_G$ collassa su pochi modi di $p_{\text{dati}}$. Sembra un
  paradosso, visto che l'obiettivo ideale ha minimo solo in $p_G =
  p_{\text{dati}}$ e la $\mathrm{JSD}$ i modi mancanti li paga eccome; la
  spiegazione è che l'addestramento non sta ottimizzando quell'obiettivo. Da un
  lato conta l'ordine dei quantificatori {cite}`goodfellow2016nips`: la
  soluzione di $\max_D \min_G$ è
  *esattamente* il generatore che manda ogni $\mathbf{z}$ sul punto che $D$ crede più
  reale, e la discesa alternata non privilegia $\min_G \max_D$ sull'altro
  ordine. Dall'altro c'è la loss non-saturating che stiamo usando, il cui
  termine $\mathrm{KL}(p_G \,\|\, p_{\text{dati}})$ addebita un costo enorme a un
  campione implausibile ($p_G > 0$ dove $p_{\text{dati}} \approx 0$) e un costo
  che tende a zero a un modo abbandonato ($p_{\text{dati}} > 0$ dove $p_G
  \approx 0$): il collasso non lo previene, lo premia.
- **Mancata convergenza.** L'equilibrio di Nash del gioco non è garantito raggiungibile con la sola discesa del gradiente; i parametri possono orbitare indefinitamente attorno all'ottimo senza stabilizzarsi.

`````

## La loss non dice niente: come si misura una GAN

C'è una domanda che a questo punto è inevitabile, e la risposta non è affatto
ovvia: **come si fa a sapere se sta funzionando?**

In tutto il resto del libro la risposta è la stessa: si guarda la loss su un
insieme di validazione, e se scende va bene. Qui non funziona, per un motivo
strutturale. Le due loss non misurano la qualità: misurano **chi dei due sta
vincendo in questo momento**. Se la loss del generatore scende può voler dire
che genera meglio, oppure soltanto che il discriminatore si è indebolito. Al
punto di equilibrio teorico, quando i falsi sono perfetti, il
discriminatore tira a indovinare e le loss si assestano su valori che non
distinguono un capolavoro da un disastro. Guardare le immagini a occhio, per
contro, non scala e soprattutto **non vede il mode collapse**: mille immagini
bellissime e tutte uguali sembrano un successo, una per volta.

Non è una cautela retorica, ed è una di quelle affermazioni che conviene
misurare invece di ripetere. Si prende il ciclo scritto qui sopra, riga per
riga, e gli si dà un compito minuscolo di cui conosciamo già la risposta: al
posto delle immagini, dei punti sparsi attorno a otto mucchietti disposti in
cerchio. Il vantaggio di un compito così è che permette di controllare quello
che sulle facce non si potrebbe controllare, cioè quanti mucchietti il
generatore ha davvero imparato e quanti dei suoi punti finiscono su uno di
essi.

Quattro addestramenti identici in tutto tranne il numero da cui parte il
sorteggio. Alla fine, le due loss dei quattro sono in pratica lo stesso numero:
`loss_D` fra $1{,}09$ e $1{,}16$, `loss_G` fra $1{,}01$ e $1{,}15$. La qualità
no: la frazione di punti generati che cade davvero su un mucchietto va dal
$26\%$ al $62\%$, e fra il migliore e il peggiore c'è un fattore due e mezzo.
Il peggiore dei quattro, per giunta, ha la seconda `loss_G` più bassa: la
classifica secondo la loss e quella secondo la qualità non si somigliano.

Dentro un singolo addestramento la cosa è ancora più netta, perché **la loss
del generatore sale mentre il generatore migliora**. All'inizio, quando non ha
imparato niente (nessun mucchietto coperto, nessun punto a segno), la sua loss
vale $0{,}59$. Quattrocento giri dopo, con tutti e otto i mucchietti coperti e
più della metà dei punti a segno, vale $1{,}09$: quasi il doppio. Un criterio
di arresto che aspettasse la loss più bassa avrebbe fermato tutto al primo
giro, con un generatore buono a niente.

Serve una misura che giudichi un **insieme** di immagini invece di una sola:
in gergo, la loro *distribuzione*, cioè come si spartiscono fra i vari tipi
possibili, quanti gatti e quanti cani e in quali pose, non soltanto se ciascuna
presa da sé è venuta bene. La strada che si è imposta è obliqua: usare una rete
già addestrata a riconoscere immagini (storicamente Inception, addestrata su
ImageNet) come strumento di misura.

`````{tab} Elementare

Il primo tentativo, l'**Inception Score**, chiede due cose insieme a un
giudice esterno che sa riconoscere gli oggetti. E qui attenzione, perché
entra in scena un personaggio nuovo: non è l'esperto d'arte del duello, è un
giudice terzo, una rete addestrata altrove a riconoscere cani, automobili e
divani, che con la nostra partita non c'entra niente e non ha nessun interesse
a farla finire in un modo o nell'altro. Il falsario e l'esperto restano dove
sono; questo signore arriva a cose fatte e guarda i risultati.

Primo: guardando una singola
immagine generata, il giudice deve saper dire con sicurezza cos'è («questo è
un cane», non «forse un cane, forse un divano»); se esita, l'immagine è
informe. Secondo: guardando tutte le immagini generate insieme, deve trovarci
soggetti diversi; se sono tutti cani, c'è mode collapse. Un punteggio alto
significa immagini nitide e varie.

Il difetto salta all'occhio appena lo si dice: in questa misura **le immagini
vere non entrano mai**. Un generatore potrebbe produrre cani nitidi e assortiti
che non somigliano a nessun cane esistente, e prendere un bel voto.

Il **FID** ripara proprio questo. Invece di interrogare il giudice sul nome
dell'oggetto, gli si sbircia dentro: si prendono i numeri che la rete calcola
a metà strada, quelli che descrivono l'immagine senza ancora nominarla.

Da lì al disegno di una nuvola il passo è breve, e vale la pena farlo piano.
Immagina che quei numeri siano due soltanto: allora ogni immagine diventa un
punto su un foglio, come una città su una cartina, e mille immagini fanno
mille punti. Immagini che si somigliano finiscono vicine, immagini diverse
lontane, e l'insieme dei punti forma una macchia con una sua posizione e una
sua forma: la **nuvola**. I numeri veri sono più di due (duemila e passa), il
foglio quindi non si può disegnare, ma i conti si fanno lo stesso e la nuvola
c'è.

Una nuvola per le immagini vere, una per
quelle generate. Se le due nuvole si sovrappongono, il generatore ha imparato;
se stanno in due posti diversi, no; e se quella generata è molto più stretta
dell'altra, il generatore sta ripetendo poche cose. Il FID è la distanza fra
le due nuvole, e più è **basso**, meglio è.

Con un'avvertenza che conviene mettere subito accanto all'ultima frase: due
nuvole possono avere lo stesso centro e la stessa larghezza pur essendo fatte
in modo diverso, e in quel caso il FID le trova uguali quando non lo sono. Un
generatore che perde per strada metà dei soggetti, ma li perde in modo
bilanciato, può ottenere un ottimo voto: il conto vede dove sta la nuvola e
quanto è larga, non quali buchi ha dentro.

`````

`````{tab} Superiore

L'**Inception Score** {cite}`salimans2016improved` combina le due richieste in
un'unica quantità:

$$
\text{IS} = \exp\Big( \mathbb{E}_{\mathbf{x} \sim p_G}\big[\, D_{\text{KL}}
\big( p(y \mid \mathbf{x})\,\|\,p(y) \big) \,\big] \Big),
$$

dove $p(y\mid \mathbf{x})$ è la distribuzione sulle classi che il classificatore assegna
al campione $\mathbf{x}$ e $p(y) = \mathbb{E}_{\mathbf{x}\sim p_G}[p(y\mid \mathbf{x})]$ è la marginale
sull'intero insieme generato. La divergenza KL è grande quando la prima è
concentrata (campione riconoscibile) e la seconda è piatta (insieme vario): le
due richieste della tab precedente, in una formula. Si valuta tipicamente su
decine di migliaia di campioni. I limiti sono noti: non usa mai $p_{\text{dati}}$,
è cieco alla varietà *dentro* una classe, e dipende dalle mille classi di
ImageNet, il che lo rende poco sensato fuori dalle immagini naturali.

La **Fréchet Inception Distance** {cite}`heusel2017gans` abbandona le classi e
lavora sulle attivazioni di uno strato intermedio (il vettore da $2048$
componenti del *pooling* finale di Inception). Si approssimano le due
popolazioni di attivazioni, reali e generate, con due gaussiane
$\mathcal{N}(\mu_r, \Sigma_r)$ e $\mathcal{N}(\mu_g, \Sigma_g)$, e si misura la
distanza di Fréchet fra le due, che per gaussiane ha forma chiusa:

$$
\text{FID} = \lVert \mu_r - \mu_g \rVert_2^2
+ \operatorname{Tr}\!\Big( \Sigma_r + \Sigma_g
- 2\big(\Sigma_r \Sigma_g\big)^{1/2} \Big).
$$

Il primo termine confronta i centri delle due nuvole, il secondo la loro forma:
è quest'ultimo a far pagare il collasso *di varianza*, perché un generatore che
ripete sempre la stessa uscita ha covarianza nulla e paga
$\operatorname{Tr}(\Sigma_r)$ anche col centro azzeccato. Il FID correla meglio
dell'IS con il giudizio umano ed è oggi lo standard di fatto.

Restano quattro avvertenze da tenere a mente quando si leggono due FID a
confronto. È **distorto verso l'alto con pochi campioni**, quindi due valori
calcolati su numerosità diverse non si confrontano. Dipende dai dettagli
implementativi (come si ridimensionano le immagini, quale versione di Inception,
quale interpolazione), al punto che numeri presi da paper diversi vanno
maneggiati con prudenza. Resta un giudizio dato da un classificatore
addestrato su fotografie: su volti, radiografie o disegni misura qualcosa,
ma non esattamente ciò che dice di misurare.

E soprattutto: il FID vede **solo i primi due momenti**. Approssimare due
popolazioni di attivazioni con due gaussiane significa non poterle distinguere
quando media e covarianza coincidono, per quanto diverse siano davvero. Un
esempio costruito apposta lo mostra bene: se i dati reali sono una mistura di
due gaussiane separate e il generatore emette un'unica gaussiana con la stessa
media e la stessa covarianza dell'insieme, il FID scende sotto $10^{-3}$, cioè
al livello che in letteratura si legge come «indistinguibile dal vero», mentre
il generatore ha buttato via un modo su due e riempie di campioni proprio la
voragine che li separa (un quarto delle sue uscite cade dove i dati reali ne
mettono il due per cento). Il termine sulle covarianze smaschera il
collasso su un punto; la perdita di modi a momenti invariati, no.

`````

Vale la pena fissare un punto che tornerà: queste due misure non giudicano
un'immagine, giudicano un **insieme** di immagini contro un altro insieme. Non
esiste il FID di una foto. È la conseguenza tecnica di ciò che una GAN cerca
di fare, cioè avvicinare $p_G$ a $p_{\text{dati}}$: si valuta l'obiettivo
dichiarato, non il singolo prodotto. Il FID sarà anche l'unità di misura con
cui, nel capitolo sui modelli di diffusione, la nuova famiglia dimostrerà di
aver superato le GAN.

## Accorgimenti pratici (cenni)

La ricerca successiva ha prodotto una cassetta degli attrezzi per domare
l'addestramento. Qui ne diamo solo i titoli, e sono cenni (della sola DCGAN
riparleremo nella prossima sezione); il filo che li unisce è che si può
intervenire su tre cose diverse.

Si può cambiare **com'è fatta** ciascuna delle due reti, adottando
l'architettura convoluzionale disciplinata delle **DCGAN**
{cite}`radford2016unsupervised`.

Si può cambiare **come si misura** la distanza fra i falsi e i veri, dove
"distanza" non è fra due immagini ma fra i due mucchi: quello delle immagini
vere e quello delle generate. È la strada della **Wasserstein GAN**
{cite}`arjovsky2017wasserstein`, che al posto della probabilità "è autentico o
no" adotta una misura dal comportamento più regolare, che cala e cresce con
dolcezza mentre i due mucchi si avvicinano invece di saltare da un estremo
all'altro; ed è esattamente il rimedio al vicolo cieco visto sopra, dove la
misura restava piatta e non indicava nessuna direzione. Il prezzo è un vincolo:
perché quella distanza sia calcolabile, l'esperto deve essere una funzione che
non reagisce mai a uno scatto più di quanto lo scatto valga (in gergo,
1-Lipschitziana). La WGAN lo imponeva ritagliando i pesi dentro una scatola;
Gulrajani e colleghi {cite}`gulrajani2017improved` hanno poi mostrato che quel
ritaglio spreca la capacità dell'esperto e fa esplodere o svanire i gradienti,
e hanno proposto il rimedio che si è imposto, il *gradient penalty*, che invece
di ritagliare i pesi penalizza direttamente la ripidità della sua risposta.

E si può cambiare **il regolamento del duello**: chiedere all'esperto di non
essere mai sicuro al cento per cento, ma di fermarsi a "reale al novanta"
(*label smoothing*), perché un giudice mai del tutto certo dà lezioni più
utili; fargli guardare i falsi a gruppi invece che uno per volta
(*minibatch discrimination*), così che un falsario che ripete sempre lo stesso
quadro venga smascherato proprio per la ripetizione; dosare i turni delle due
reti perché nessuna delle due prenda troppo vantaggio sull'altra.

Nessuno di questi trucchi è una bacchetta magica: l'addestramento avversario
resta un'arte oltre che una scienza, ma è proprio da questa tensione che
nascono i risultati più sorprendenti del deep learning generativo.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Una GAN è un **duello** fra due reti: il falsario parte da una manciata di
  numeri casuali e ne ricava un dato che sembri autentico, l'esperto guarda un
  dato e dice quanto lo crede vero.
- Quello che torna indietro dall'esperto al falsario **non è il verdetto**: è
  una lista lunga quanto il quadro, che per ogni puntino dice da che parte
  tirare e con quanta forza. Per questo l'esperto dev'essere una rete: una
  persona darebbe lo stesso giudizio e nessuna lista. E la realtà entra nel
  gioco da un lato solo, perché è l'esperto (mai il falsario) a vedere i quadri
  autentici, e a essere corretto su quelli.
- Giocano un **punteggio unico**: quello che è un bene per uno è un male per
  l'altro. L'equilibrio arriva quando i falsi non si distinguono più dai veri,
  e lì l'esperto può soltanto tirare a indovinare.
- Si allenano **a turni**, un passo ciascuno, ed è un addestramento
  capriccioso: attenzione al *mode collapse* (il falsario trova un solo quadro
  che inganna sempre e si limita a rifarlo) e alla mancata convergenza. Quando
  l'esperto è troppo bravo, il suo giudizio è talmente schiacciato sul "falso"
  che non si muove più, e senza movimento non c'è correzione: si rimedia
  chiedendo al falsario, nel suo turno, di far passare i propri quadri per
  autentici; il prezzo sono correzioni più sbalzate.
- **La loss, cioè il conto dell'errore, non misura la qualità**: dice solo chi
  dei due sta vincendo. Si giudica confrontando *insiemi* di immagini, mai una
  alla volta: con l'**Inception Score** (nitidezza e varietà secondo un giudice
  esterno, che però le immagini vere non le guarda mai) e soprattutto con il
  **FID**, la distanza fra la nuvola delle immagini vere e quella delle
  generate: più è basso, meglio è. Neanche il FID però è infallibile: vede dove
  sta la nuvola e quanto è larga, quindi smaschera il falsario che ripete
  sempre lo stesso quadro, non quello che perde per strada interi soggetti
  lasciando la nuvola dov'era.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Una GAN è un **duello** tra due reti: il generatore $G$ trasforma rumore in dati sintetici, il discriminatore $D$ stima la probabilità che un dato sia reale.
- Ciò che $D$ restituisce a $G$ è $\partial \mathcal{L}_G / \partial \tilde{\mathbf{x}}$,
  un **vettore** nello spazio dei dati, non il verdetto scalare; la regola della
  catena lo compone con $\partial \tilde{\mathbf{x}} / \partial \theta_G$. Ne segue un
  requisito di progetto: $D$ dev'essere derivabile rispetto al proprio ingresso,
  ed è la ragione per cui sui dati discreti la catena si spezza.
- Condividono un'unica **funzione di valore minimax**: $G$ la minimizza, $D$ la
  massimizza; l'obiettivo ideale ha minimo in $p_G = p_{\text{dati}}$, e lì il
  discriminatore ottimo vale $D^*(\mathbf{x})=\tfrac12$ sul supporto dei dati. La
  *caratterizzazione* dell'ottimo non è però una garanzia di convergenza: la
  prova vive nello spazio delle densità, l'addestramento in quello dei
  parametri.
- L'addestramento è **alternato** e notoriamente instabile: attenzione al
  *mode collapse* e alla mancata convergenza. I gradienti che svaniscono, invece,
  riguardano l'obiettivo minimax originale: la *non-saturating loss* usata nel
  codice li evita, al prezzo di aggiornamenti ad alta varianza quando $D$ è
  quasi ottimo, e di un gioco che non è più a somma zero.
- **La loss non misura la qualità**: dice solo chi sta vincendo. Si valuta
  confrontando *distribuzioni*, con l'**Inception Score** (nitidezza e varietà
  secondo un classificatore, ma senza mai guardare i dati veri) e soprattutto
  con il **FID**, la distanza fra la nuvola delle attivazioni reali e quella
  delle generate: più basso è meglio. Il FID però guarda solo i primi due
  momenti: il termine sulle covarianze smaschera il collasso di varianza, non
  la perdita di modi a media e covarianza invariate.
```

`````
