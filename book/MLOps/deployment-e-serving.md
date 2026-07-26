# Servire un modello: dal file all'API

Alla fine di tutto, un modello addestrato è un file. Qualche centinaio di
megabyte di numeri su un disco — i pesi che nella sezione precedente abbiamo
imparato a versionare e archiviare — e nient'altro. Da solo non fa niente: è
inerte come uno spartito senza orchestra. «Metterlo in produzione» significa
esattamente dargli l'orchestra: un modo di ricevere richieste dal mondo e di
rispondere, in fretta e in modo affidabile, migliaia di volte al minuto.

La sezione precedente si è fermata sul punto in cui l'artefatto — dati, codice,
pesi — è tracciabile e riproducibile. Questa affronta il passo successivo, il
nodo *Deploy* dell'anello: come si mette un modello **in ascolto**. È una
questione tanto ingegneristica quanto di aspettative — decidere *che cosa
promettere* a chi userà il servizio — e per una volta il grosso del lavoro non
riguarda la rete neurale, ma tutto ciò che le sta attorno
{cite}`kreuzberger2023machine`.

## Batch, online, streaming

Prima di scrivere una riga di codice va scelto il **regime di inferenza**, cioè
il modo in cui il modello incontra le richieste. Non è un dettaglio: cambia
l'architettura, le priorità, perfino il conto della bolletta. I regimi
fondamentali sono tre.

`````{tab} Elementare

Pensa a un forno.

C'è il pane **in blocco**: di notte, quando il negozio è chiuso, il fornaio
prepara in un colpo solo tutto il pane che servirà l'indomani. Nessuno aspetta
al bancone, quindi non importa se ci mette due ore: conta solo sfornarne tanto.
Questo è il regime *batch*: si accumula un mucchio di richieste e le si smaltisce
tutte insieme, quando fa comodo — di notte, offline.

C'è poi il **panino al momento**: un cliente entra, ordina, e vuole il suo
panino *adesso*, non domani. Qui conta la fretta: ogni singola richiesta deve
avere risposta in pochi secondi, mentre la persona aspetta. Questo è il regime
*online*: una richiesta, una risposta, subito.

E c'è il **nastro trasportatore** del sushi: i piatti scorrono senza sosta e tu
prendi al volo quello che passa. Nessuno «ordina» e nessuno «finisce»: è un
flusso continuo che non si ferma mai. Questo è lo *streaming*: eventi che
arrivano ininterrottamente — clic, transazioni, sensori — e il modello li lavora
al volo, mentre passano.

`````

`````{tab} Superiore

I tre regimi si distinguono lungo due assi: **latenza** (quanto tempo passa tra
una richiesta e la sua risposta) e **throughput** (quante richieste al secondo
il sistema smaltisce). Sono in tensione, e ogni regime ottimizza uno sacrificando
l'altro.

- **Batch (offline)**: si accumula un grande insieme di input e li si elabora in
  un'unica passata pianificata (tipicamente notturna). La latenza per singolo
  esempio è irrilevante — ore vanno benissimo — mentre il throughput si
  massimizza sfruttando batch enormi. Caso d'uso: assegnare un punteggio a tutti
  i clienti di un database per una campagna, o pre-calcolare le raccomandazioni
  della home page.
- **Online (sincrono)**: la richiesta arriva e attende la risposta in linea, con
  un budget di latenza stretto (spesso decine di millisecondi). Il throughput si
  ottiene con la concorrenza, non con batch grandi. Caso d'uso: la
  raccomandazione calcolata al clic dell'utente.
- **Streaming (event-driven)**: il modello consuma un flusso continuo di eventi,
  spesso su finestre temporali scorrevoli, e produce output man mano. Caso d'uso:
  rilevare frodi su transazioni mentre avvengono.

La scelta non è di gusto: è dettata dal prodotto {cite}`huyen2022designing`. Se
la previsione può essere pronta *prima* che serva, il batch è più semplice ed
economico; se dipende da un input che esiste solo al momento della richiesta,
serve l'online.

`````

Batch e online sono i due estremi che si incontrano più spesso, ed è utile
vederli affiancati ({numref}`fig-mlops-serving`): stessa scatola «modello» al
centro, priorità opposte ai due lati.

```{figure} ../figures/mlops-serving.svg
:name: fig-mlops-serving
:alt: "Due schemi affiancati. A sinistra il regime batch: una pila di molte richieste entra nel modello e ne esce una pila di molte previsioni, con priorità al throughput e latenza non critica. A destra il regime online: una singola richiesta entra nel modello e ne esce una singola risposta, con priorità alla latenza bassa. In entrambi il modello carica i pesi una sola volta."
:width: 90%

Batch contro online: lo stesso modello, priorità opposte. A sinistra si smaltisce
un mucchio di richieste in blocco (conta il *throughput*); a destra si risponde a
una richiesta per volta, subito (conta la *latenza*).
```

## Il modello dietro un'API

Nel regime online — il più comune e il più esigente — il modello vive dietro un
**endpoint**: un indirizzo a cui altri programmi mandano una richiesta (di solito
in JSON, via HTTP) e da cui ricevono la risposta. Chi chiama non sa e non deve
sapere che dentro c'è una rete neurale: vede solo un servizio che, dati certi
ingressi, restituisce una previsione. Il programma che tiene il modello in
memoria e traduce le richieste in chiamate al `forward` si chiama **model
server**.

`````{tab} Elementare

È lo sportello di un ufficio. Dietro il vetro c'è l'impiegato — il modello — che
sa fare una cosa sola ma la sa fare bene. Tu non entri nel retro a rovistare tra
le pratiche: passi il tuo modulo dalla fessura e ti torna indietro la risposta
compilata. Lo sportello (l'*endpoint*) nasconde tutto il resto. E c'è una regola
di buon senso che vale oro: l'impiegato arriva la mattina, si siede *una volta
sola* e resta lì tutto il giorno. Sarebbe assurdo se andasse a casa e tornasse a
ogni singolo cliente. Con i modelli è identico: i pesi si caricano in memoria una
volta all'avvio del servizio, non a ogni richiesta — caricarli costa secondi, e a
ogni richiesta li si pagherebbe di nuovo.

`````

`````{tab} Superiore

Il model server carica i pesi una volta all'avvio e li tiene in memoria; a ogni
richiesta esegue solo il *forward*, in modalità inferenza. Sopra questa logica si
appoggia un livello di trasporto — REST/JSON per semplicità, o gRPC per la bassa
latenza e i payload binari — che però è, in sostanza, contorno.

Il problema serio non è esporre l'endpoint, è renderlo **riproducibile**. Un
modello dipende da una versione precisa di PyTorch, delle librerie di
pre-processing, perfino di CUDA: la stessa trappola del «sul mio computer
funzionava» vista nella sezione precedente, spostata dall'addestramento al
servizio. La risposta standard è la **containerizzazione**: un'immagine Docker
che congela sistema, interprete Python, dipendenze e pesi in un artefatto unico e
avviabile ovunque allo stesso modo. Il container è ciò che si versiona e si
distribuisce; l'orchestrazione di più container (scalare le repliche sotto
carico) è il livello successivo, di competenza dell'infrastruttura.

`````

Lo scheletro di un servizio d'inferenza, spogliato del framework web, sta in
poche righe. La sostanza è tutta in tre gesti: **caricare una volta**, mettere il
modello in **modalità inferenza**, disattivare l'**autograd**.

```{code-block} python
:class: pt-non-eseguibile

import torch

# Caricamento UNA VOLTA all'avvio del servizio, non a ogni richiesta
modello = MiaRete()                        # la classe nn.Module usata in addestramento
modello.load_state_dict(torch.load("pesi.pt", map_location="cpu"))
modello.eval()                             # modalità inferenza: niente dropout, BatchNorm congelata

@torch.no_grad()                           # niente autograd: meno memoria, più veloce
def predici(richiesta: dict) -> dict:
    x = preprocessa(richiesta)             # dal JSON al tensore d'ingresso (batch di 1)
    logit = modello(x)                     # forward: logit grezzi (cfr. capitolo PyTorch)
    prob = torch.softmax(logit, dim=1)     # logit -> probabilità
    return {
        "classe": int(prob.argmax(dim=1).item()),
        "confidenza": float(prob.max().item()),
    }
```

Due dettagli fanno la differenza tra un giocattolo e un servizio corretto.
`modello.eval()` commuta il comportamento degli strati che si comportano in modo
diverso in addestramento e in inferenza — *dropout* e *BatchNorm* su tutti —
senza il quale le previsioni sarebbero silenziosamente sbagliate.
`torch.no_grad()` (o l'equivalente più aggressivo `torch.inference_mode()`) dice
ad autograd di non costruire il grafo per il *backward*: in inferenza non serve, e
ometterlo spreca memoria e tempo a ogni richiesta. Il resto — la produzione dei
*logit*, la `softmax` che li trasforma in probabilità — è esattamente il modello
del capitolo PyTorch, ora chiamato a rispondere invece che a imparare.

## Ottimizzare l'inferenza

Un servizio corretto può ancora essere troppo lento o troppo costoso. Le leve per
accelerare l'inferenza sono diverse da quelle dell'addestramento, ma una radice è
comune con il capitolo PyTorch: meno byte da spostare, più velocità.

La prima leva è il **batching dinamico**. Come abbiamo visto parlando di
prestazioni, la GPU rende al massimo su tanti conti identici in parallelo: servire
le richieste una per una la lascia mezza vuota. Il server allora accumula per
qualche millisecondo le richieste che arrivano, le impacchetta in un unico batch e
le passa al modello in un colpo solo — un filo di latenza in più in cambio di molto
più throughput. La seconda leva è **ridurre la precisione** dei numeri: la sezione
sulle prestazioni ha introdotto la precisione mista e il `float16` per
l'addestramento; in inferenza lo stesso passaggio a 16 bit dimezza memoria e banda
quasi gratis. La terza leva spinge oltre, fino agli **interi**: la
**quantizzazione** a `int8` {cite}`jacob2018quantization`.

`````{tab} Elementare

È il trucco di quando mandi una foto su una chat: l'app la rimpicciolisce prima di
spedirla. Perdi un filo di nitidezza — se ci fai molto caso —, ma il file pesa un
quarto e parte in un lampo. Quantizzare un modello è la stessa idea applicata ai
suoi numeri. I pesi, di norma, sono decimali finissimi (tante cifre dopo la
virgola); la quantizzazione li riscrive come **numeri interi grossolani**, da 0 a
255 livelli. Ne guadagni quattro volte in leggerezza e spesso un bel taglio di
velocità; ne perdi un pizzico di precisione. Il patto conviene quasi sempre: nella
maggior parte dei modelli l'accuratezza cala di una frazione di punto percentuale,
un prezzo minuscolo per un modello quattro volte più piccolo che gira anche su un
telefono.

`````

`````{tab} Superiore

La forma più economica è la **quantizzazione post-training** (*PTQ*): si prende un
modello già addestrato in `float32` e se ne convertono i pesi in interi, senza
riaddestrare. Il legame tra il numero reale $r$ e l'intero $q$ è una **mappa affine**
{cite}`jacob2018quantization`:

$$
r = S\,(q - Z),
$$

dove $q$ è l'intero a 8 bit (in $[-128, 127]$), $S > 0$ è la **scala** (un numero
reale, l'ampiezza di un gradino), e $Z$ è lo **zero-point**, l'intero che
rappresenta il valore reale $0$. La quantizzazione è l'inversa,
$q = \mathrm{round}(r / S) + Z$, troncata all'intervallo.

Facciamo i conti a mano su un vettore di pesi
$w = [-1{,}00,\ -0{,}352,\ 0{,}20,\ 1{,}55,\ 0{,}073]$. Da minimo e massimo
($r_{\min} = -1{,}00$, $r_{\max} = 1{,}55$) si ricava la scala
$S = (r_{\max} - r_{\min}) / (127 - (-128)) = 2{,}55 / 255 = 0{,}01$ e lo
zero-point $Z = -128 - \mathrm{round}(r_{\min}/S) = -128 - (-100) = -28$.
Quantizzando ($q = \mathrm{round}(r/S) + Z$) si ottiene
$q = [-128,\ -63,\ -8,\ 127,\ -21]$; ricostruendo ($\hat r = S(q - Z)$) si torna a
$\hat r = [-1{,}00,\ -0{,}35,\ 0{,}20,\ 1{,}55,\ 0{,}07]$. L'**errore** è al più
$0{,}003$: non può superare mezzo gradino, $S/2 = 0{,}005$. Sul piano dei byte, i
cinque `float32` (20 byte) diventano cinque `int8` (5 byte) più i due parametri di
calibrazione $S$ e $Z$ condivisi da tutto il tensore: per uno strato con milioni di
pesi, quei due numeri sono trascurabili e la compressione è di circa **4×**.

In PyTorch la quantizzazione dinamica post-training è una riga, e l'esportazione
verso un runtime dedicato (indipendente da Python) è un'altra:

```{code-block} python
:class: pt-non-eseguibile

import torch
import torch.nn as nn

# Pesi in int8, attivazioni quantizzate al volo (dynamic quantization)
modello_int8 = torch.quantization.quantize_dynamic(
    modello, {nn.Linear}, dtype=torch.qint8,
)

esempio = torch.randn(1, 784)
torch.onnx.export(modello, esempio, "modello.onnx")   # export verso il runtime ONNX
programma = torch.export.export(modello, (esempio,))  # cattura del grafo nativa (PyTorch 2.x)
```

La `quantize_dynamic` è la variante più indolore (nessun dato di calibrazione
richiesto, adatta agli strati lineari); l'export ONNX o `torch.export` staccano il
modello dal codice Python di addestramento e lo consegnano a un motore d'inferenza
ottimizzato. Vale la stessa disciplina della precisione mista: la quantizzazione va
sempre **misurata** su un insieme di validazione, perché il calo di accuratezza
dipende dal modello e non è mai garantito trascurabile a priori.

`````

## Latenza e throughput: cosa promettere

Ottimizzato il servizio, resta la domanda più scomoda: che cosa **promettere** a
chi lo userà? La promessa si scrive in un **SLA** (*Service Level Agreement*), e il
punto delicato è che va misurata con l'onestà giusta — la media, qui, è una
bugia gentile.

`````{tab} Elementare

Immagina la coda alla posta. Se dico che «in media» si aspetta cinque minuti, ho
detto poco: magari novanta persone su cento passano in un minuto e dieci restano
impantanate mezz'ora, e la media di cinque minuti non la vive quasi nessuno. Quello
che conta davvero è la promessa sul *caso quasi peggiore*: «il 95% dei clienti è
servito entro dieci minuti». Con i modelli è identico. Non si promette la latenza
media, si promette che la *stragrande maggioranza* delle risposte arriva entro un
tempo dato. Perché il cliente scontento non è quello medio: è quello finito nella
coda lenta.

`````

`````{tab} Superiore

Si descrive la latenza con i suoi **percentili**, non con la media. La p50
(mediana) è il tempo entro cui risponde metà delle richieste; la **p95** e la
**p99** i tempi entro cui ne risponde il 95% e il 99%. La *coda* della
distribuzione — la p99, la p99.9 — è ciò che governa l'esperienza reale sotto
carico, perché in un sistema che compone più servizi anche una piccola frazione di
richieste lente si propaga e degrada l'insieme. Un SLA serio si scrive sui
percentili alti: «p99 sotto i 200 ms», non «latenza media 80 ms», che nasconde la
coda.

Il secondo numero da promettere è il **throughput** sostenibile (richieste al
secondo), che con la latenza forma il classico compromesso: più batch grandi
alzano il throughput ma allungano la coda della latenza. Il terzo è economico, il
**costo per richiesta** — tempo di calcolo moltiplicato per il prezzo
dell'hardware — che spesso è il vero vincolo di progetto: un modello che rispetta
lo SLA ma costa dieci volte troppo per richiesta non è dispiegabile
{cite}`huyen2022designing`.

`````

C'è infine una cautela che riguarda *come* si sostituisce un modello con uno
nuovo, senza rompere niente. Non si spegne la versione vecchia e si accende la
nuova sperando bene: si procede per gradi. In un rilascio *canary* la nuova
versione riceve dapprima una piccola frazione del traffico, che si allarga solo se
le metriche tengono; in modalità *shadow* la nuova versione riceve una copia delle
richieste reali ma le sue risposte non vengono servite, solo confrontate con la
vecchia; un test *A/B* misura su traffico reale quale delle due converte meglio. È
il lato «serving» della stessa prudenza che l'anello MLOps chiede a ogni tappa:
misurare prima di fidarsi.

```{admonition} Da ricordare
:class: important
- Un modello in produzione va scelto per **regime di inferenza**: *batch* (in
  blocco, offline, massimizza il throughput), *online* (sincrono, budget di
  latenza stretto), *streaming* (flusso continuo di eventi)
  {cite}`huyen2022designing`.
- Nel serving online il modello vive dietro un **endpoint** gestito da un **model
  server** che carica i pesi **una volta sola**; il **container** Docker congela
  l'ambiente e lo rende riproducibile {cite}`kreuzberger2023machine`.
- Lo scheletro d'inferenza corretto in PyTorch è: `load_state_dict` all'avvio,
  `model.eval()` (per *dropout* e *BatchNorm*), `torch.no_grad()` per non costruire
  il grafo del backward.
- Le leve per accelerare sono il **batching dinamico**, la **riduzione di
  precisione** (`float16`, come nel capitolo PyTorch) e la **quantizzazione a
  `int8`** con la mappa affine $r = S(q - Z)$ {cite}`jacob2018quantization`: circa
  4× di memoria in meno al prezzo di un piccolo calo di accuratezza, **da
  misurare** sempre.
- Un **SLA** si scrive sui **percentili alti** della latenza (p95, p99), non sulla
  media, e va bilanciato con **throughput** e **costo per richiesta**.
- Le nuove versioni si rilasciano per gradi — *canary*, *shadow*, A/B — per non
  rompere niente in produzione.
```
