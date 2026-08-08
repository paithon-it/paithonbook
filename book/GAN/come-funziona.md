# Come funziona l'addestramento avversario

Dell'idea nata quella sera a Montréal si è già detto in apertura di capitolo; qui la smontiamo pezzo per pezzo.

Il modo consueto di insegnare a una rete a produrre un'immagine è dirle, punto per punto, quanto la sua uscita si discosta da un'immagine vera che le mettiamo accanto: questo puntino doveva essere più scuro, quest'altro più chiaro. Il metodo funziona finché l'originale ce l'abbiamo sotto mano; ma per *inventare* un'immagine nuova l'originale non esiste, e non c'è niente con cui fare il confronto. Le **Generative Adversarial Networks** (GAN), descritte nel paper *Generative Adversarial Nets* {cite}`goodfellow2014generative`, cambiano giudice: al posto del confronto punto per punto mettono una seconda rete, il cui unico mestiere è smascherare la prima.

## Il falsario e l'esperto, tradotti in reti

La metafora dell'apertura si traduce così in due reti con un nome: il falsario che dipinge quadri contraffatti è il **generatore** $G$, l'esperto d'arte che deve smascherarlo è il **discriminatore** $D$. Da qui in avanti li chiamiamo con questi nomi, e ci occupiamo di come sono fatti dentro e di come si allenano a vicenda.

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

$D$ è l'esperto d'arte. Riceve un dato, a volte vero (pescato dal dataset), a
volte falso (sfornato da $G$), e deve rispondere a una sola domanda: *è
autentico?* La sua risposta è un numero tra $0$ e $1$, una specie di livello
di fiducia: vicino a $1$ significa "sono quasi certo che sia reale", vicino a
$0$ significa "quasi certo che sia un falso". Il suo mestiere è non farsi
ingannare.

`````

`````{tab} Superiore

Il discriminatore è una funzione $D(x;\theta_D)\in[0,1]$ che stima la probabilità che $x$ provenga dai dati reali anziché da $G$. La formalizzazione passa da una mistura: il campione arriva metà delle volte dal dataset e metà dal generatore, e $D$ stima la probabilità *a posteriori* che la sorgente sia quella reale, visto il campione:

$$
D(x) \approx P(\text{reale} \mid x) .
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

## Il gioco minimax

Le due reti non ottimizzano due funzioni scollegate: condividono un'**unica** funzione di valore, che uno vuole massimizzare e l'altro minimizzare.

`````{tab} Elementare

Immagina un punteggio unico del gioco. L'esperto guadagna punti ogni volta
che indovina; il falsario "vince" ogni volta che gli fa perdere punti. Quello
che è un bene per uno è un male per l'altro: è un gioco a somma (quasi) nulla.
Non esiste un traguardo fisso da raggiungere: esiste un *equilibrio*, il punto
in cui nessuno dei due riesce più a migliorare a spese dell'altro. Lì il
falsario è così bravo che l'esperto, per quanto si sforzi, può solo tirare
a indovinare.

`````

`````{tab} Superiore

$G$ e $D$ giocano un gioco minimax sulla funzione di valore

$$
\min_{G}\ \max_{D}\ V(D,G) =
\mathbb{E}_{x\sim p_{\text{dati}}}\big[\log D(x)\big]
+ \mathbb{E}_{z\sim p_z}\big[\log\big(1 - D(G(z))\big)\big] .
$$

Qui $p_{\text{dati}}$ è la distribuzione dei dati reali, $p_z$ il prior del rumore, $D(x)$ la probabilità stimata di autenticità e $G(z)$ il campione generato. $D$ massimizza $V$ (vuole $D(x)$ grande sui reali e $1-D(G(z))$ grande sui falsi); $G$ minimizza il secondo termine (vuole $D(G(z))\to 1$). La dimostrazione di Goodfellow sta in due passaggi che vale la pena avere sott'occhio. Primo: per $G$ fissato, il discriminatore che massimizza $V$ è

$$
D^*(x) = \frac{p_{\text{dati}}(x)}{p_{\text{dati}}(x) + p_G(x)},
$$

cioè proprio l'ottimo bayesiano della mistura descritta sopra. Secondo: sostituendo $D^*$ in $V$ si ottiene $V(D^*,G) = -\log 4 + 2\,\mathrm{JSD}\big(p_{\text{dati}} \,\|\, p_G\big)$, dove $\mathrm{JSD}$ è la divergenza di Jensen-Shannon, non negativa e nulla se e solo se le due distribuzioni coincidono. Con capacità e dati sufficienti, l'ottimo globale si raggiunge dunque se e solo se $p_G = p_{\text{dati}}$, e in quel punto $D^*(x)=\tfrac{1}{2}$ sul supporto dei dati: l'esperto non sa più distinguere.

`````

## L'addestramento alternato

Poiché i due obiettivi sono in conflitto, non si può ottimizzarli in un colpo solo. Si procede **a turni**, con discesa del gradiente stocastica: un passo per $D$, un passo per $G$, e così via. Mentre si aggiorna una rete, i parametri dell'altra restano congelati.

```python
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

Due dettagli del codice contengono tutta la logica del gioco: il modo in cui i due allenamenti restano separati, e una piccola astuzia sulla lezione impartita al generatore.

`````{tab} Elementare

Prima dei due dettagli, una parola sulla riga con la `n`. I dati non si danno
in pasto alla rete uno per volta ma a gruppetti, e l'ultimo gruppo di ogni giro
può risultare più corto degli altri (se gli esempi sono $1000$ e i gruppi da
$64$, l'ultimo ne contiene $40$). La `n` conta quanti esempi ci sono davvero nel
gruppo di turno, e serve a preparare esattamente altrettante etichette "vero" e
"falso": una in meno o una in più e il conto dell'errore non torna.

Primo dettaglio: quando si allena l'esperto sui falsi, quei falsi vengono
"staccati" dal falsario (è la parola `.detach()` nel codice): il giudizio
serve a correggere solo chi giudica, non chi ha dipinto. E quando è il turno
del falsario, l'aggiornamento tocca solo i pesi suoi: ognuno impara nel
proprio turno, come da regolamento; è il "congelamento" descritto sopra, che
in PyTorch si scrive in una parola.

Secondo dettaglio: nel suo turno, il falsario chiede all'esperto di trattare
i propri falsi come "reali" e impara da quanto il verdetto se ne discosta. È
una versione più *generosa* del gioco: dà al falsario lezioni chiare proprio
all'inizio, quando i suoi quadri sono pessimi e l'esperto li respinge con
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
- **Mode collapse.** Il falsario scopre *un solo* falso che inganna sempre l'esperto e si limita a rifarlo. Risultato: $G$ genera sempre la stessa immagine (o pochissime varianti), buttando via tutta la varietà dei dati reali.
- **Mancata convergenza.** A volte il gioco non trova mai un equilibrio: le immagini oscillano, degenerano, o non migliorano più.

`````

`````{tab} Superiore

- **Instabilità.** L'ottimizzazione simultanea di un gioco minimax non equivale a minimizzare una singola funzione: la dinamica può divergere o entrare in cicli limite. Se $D$ diventa troppo accurato si ha $D(G(z))\to 0$, e con l'obiettivo minimax originale questo annulla i gradienti verso $G$ (*vanishing gradients*); la non-saturating loss vista sopra scongiura l'annullamento, ma con un discriminatore quasi ottimo lo paga in aggiornamenti instabili e ad alta varianza {cite}`arjovsky2017towards`. Se invece $D$ è troppo debole, non fornisce segnale utile.
- **Mode collapse.** $G$ mappa molti $z$ diversi su una stessa uscita $\tilde{x}$: $p_G$ collassa su pochi modi di $p_{\text{dati}}$. Formalmente minimizza il proprio obiettivo locale ignorando la copertura dell'intera distribuzione.
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
punto di equilibrio teorico, quando i campioni sono perfetti, il
discriminatore tira a indovinare e le loss si assestano su valori che non
distinguono un capolavoro da un disastro. Guardare le immagini a occhio, per
contro, non scala e soprattutto **non vede il mode collapse**: mille immagini
bellissime e tutte uguali sembrano un successo, una per volta.

Servono metriche che giudichino una **distribuzione** invece di un campione, e
la strada che si è imposta è obliqua: usare una rete già addestrata a
riconoscere immagini (storicamente Inception, addestrata su ImageNet) come
strumento di misura.

`````{tab} Elementare

Il primo tentativo, l'**Inception Score**, chiede due cose insieme a un
giudice esterno che sa riconoscere gli oggetti. Primo: guardando una singola
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
a metà strada, quelli che descrivono l'immagine senza ancora nominarla, e si
guarda la **nuvola** che formano. Una nuvola per le immagini vere, una per
quelle generate. Se le due nuvole si sovrappongono, il generatore ha imparato;
se stanno in due posti diversi, no; e se quella generata è molto più stretta
dell'altra, il generatore sta ripetendo poche cose. Il FID è la distanza fra
le due nuvole, e più è **basso**, meglio è.

`````

`````{tab} Superiore

L'**Inception Score** {cite}`salimans2016improved` combina le due richieste in
un'unica quantità:

$$
\text{IS} = \exp\Big( \mathbb{E}_{x \sim p_G}\big[\, D_{\text{KL}}
\big( p(y \mid x)\,\|\,p(y) \big) \,\big] \Big),
$$

dove $p(y\mid x)$ è la distribuzione sulle classi che il classificatore assegna
al campione $x$ e $p(y) = \mathbb{E}_{x\sim p_G}[p(y\mid x)]$ è la marginale
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
è quest'ultimo a cogliere il mode collapse, perché una distribuzione collassata
ha covarianza troppo piccola e paga anche se il centro è azzeccato. Il FID
correla meglio dell'IS con il giudizio umano ed è oggi lo standard di fatto.

Restano tre avvertenze da tenere a mente quando si leggono due FID a
confronto. È **distorto verso l'alto con pochi campioni**, quindi due valori
calcolati su numerosità diverse non si confrontano. Dipende dai dettagli
implementativi (come si ridimensionano le immagini, quale versione di Inception,
quale interpolazione), al punto che numeri presi da paper diversi vanno
maneggiati con prudenza. E resta un giudizio dato da un classificatore
addestrato su fotografie: su volti, radiografie o disegni misura qualcosa,
ma non esattamente ciò che dice di misurare.

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

Si può cambiare **come si misura** la distanza fra i falsi e i veri: è la
strada della **Wasserstein GAN** {cite}`arjovsky2017wasserstein`, che al posto
della probabilità "è autentico o no" usa una misura di distanza dal
comportamento più regolare, cioè che cala e cresce con dolcezza invece di
saltare da un estremo all'altro (spesso accompagnata da un termine, il
*gradient penalty*, che impedisce all'esperto di reagire in modo troppo brusco
a piccole differenze).

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
- Giocano un **punteggio unico**: quello che è un bene per uno è un male per
  l'altro. L'equilibrio arriva quando i falsi non si distinguono più dai veri,
  e lì l'esperto può soltanto tirare a indovinare.
- Si allenano **a turni**, un passo ciascuno, ed è un addestramento
  capriccioso: attenzione al *mode collapse* (il falsario trova un solo quadro
  che inganna sempre e si limita a rifarlo) e alla mancata convergenza. Che il
  principiante resti senza lezioni quando l'esperto è troppo bravo si evita
  chiedendogli, nel suo turno, di far passare i propri falsi per veri; il
  prezzo sono correzioni più sbalzate.
- **Il conto dell'errore non misura la qualità**: dice solo chi dei due sta
  vincendo. Si giudica confrontando *insiemi* di immagini, mai una alla volta:
  con l'**Inception Score** (nitidezza e varietà secondo un giudice esterno,
  che però le immagini vere non le guarda mai) e soprattutto con il **FID**,
  la distanza fra la nuvola delle immagini vere e quella delle generate: più è
  basso, meglio è, ed è la parte che confronta la forma delle due nuvole a
  smascherare il mode collapse.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Una GAN è un **duello** tra due reti: il generatore $G$ trasforma rumore in dati sintetici, il discriminatore $D$ stima la probabilità che un dato sia reale.
- Condividono un'unica **funzione di valore minimax**: $G$ la minimizza, $D$ la massimizza; l'equilibrio si ha quando $p_G = p_{\text{dati}}$, e lì il discriminatore ottimo vale $D^*(x)=\tfrac12$ sul supporto dei dati.
- L'addestramento è **alternato** e notoriamente instabile: attenzione al
  *mode collapse* e alla mancata convergenza. I gradienti che svaniscono, invece,
  riguardano l'obiettivo minimax originale: la *non-saturating loss* usata nel
  codice li evita, al prezzo di aggiornamenti ad alta varianza quando $D$ è
  quasi ottimo.
- **La loss non misura la qualità**: dice solo chi sta vincendo. Si valuta
  confrontando *distribuzioni*, con l'**Inception Score** (nitidezza e varietà
  secondo un classificatore, ma senza mai guardare i dati veri) e soprattutto
  con il **FID**, la distanza fra la nuvola delle attivazioni reali e quella
  delle generate: più basso è meglio, e la parte sulle covarianze è quella che
  smaschera il mode collapse.
```

`````
