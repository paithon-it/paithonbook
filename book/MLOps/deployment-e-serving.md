# Servire un modello: dal file all'API

Alla fine di tutto, un modello addestrato è un file. Qualche centinaio di
megabyte di numeri su un disco (i pesi che nella sezione *Dal notebook alla
produzione* abbiamo imparato a versionare e archiviare) e nient'altro. Da solo
non fa niente: è inerte come uno spartito senza orchestra. «Metterlo in
produzione» significa esattamente dargli l'orchestra: un modo di ricevere
richieste dal mondo e di rispondere, in fretta e in modo affidabile, migliaia
di volte al minuto.

Le due sezioni precedenti si sono fermate sul punto in cui l'artefatto (dati,
codice, pesi) è tracciabile e riproducibile. Questa affronta il passo
successivo, il nodo *Deploy* dell'anello: come si mette un modello **in
ascolto**. È una
questione tanto ingegneristica quanto di aspettative (decidere *che cosa
promettere* a chi userà il servizio) e per una volta il grosso del lavoro non
riguarda la rete neurale, ma tutto ciò che le sta attorno
{cite}`kreuzberger2023machine`.

## Batch, online, streaming

Prima di scrivere una riga di codice va scelto il **regime di inferenza**.
*Inferenza* è il momento in cui il modello non impara più ma risponde: gli si
dà un caso, restituisce la sua previsione; il regime è il modo in cui quel
momento è organizzato, cioè come il modello incontra le richieste. Non è un
dettaglio: cambia l'architettura, le priorità, perfino il conto della
bolletta. I regimi fondamentali sono tre.

`````{tab} Elementare

Pensa a un forno.

C'è il pane **in blocco**: di notte, quando il negozio è chiuso, il fornaio
prepara in un colpo solo tutto il pane che servirà l'indomani. Nessuno aspetta
al bancone, quindi non importa se ci mette due ore: conta solo sfornarne
tanto. Questo è il regime *batch*: si accumula un mucchio di richieste e le si
smaltisce tutte insieme, quando fa comodo (di notte, offline).

C'è poi il **panino al momento**: un cliente entra, ordina, e vuole il suo
panino *adesso*, non domani. Qui conta la fretta: ogni singola richiesta deve
avere risposta in pochi secondi, mentre la persona aspetta. Questo è il regime
*online*: una richiesta, una risposta, subito.

E c'è il **nastro trasportatore** del sushi: i piatti scorrono senza sosta e
tu prendi al volo quello che passa. Nessuno «ordina» e nessuno «finisce»: è un
flusso continuo che non si ferma mai. Questo è lo *streaming*: eventi che
arrivano ininterrottamente (clic, transazioni, sensori) e il modello li lavora
al volo, mentre passano.

`````

`````{tab} Superiore

I tre regimi si distinguono lungo due assi: **latenza** (quanto tempo passa tra
una richiesta e la sua risposta) e **throughput** (quante richieste al secondo
il sistema smaltisce). Sono in tensione, e ogni regime ottimizza uno sacrificando
l'altro.

- **Batch (offline)**: si accumula un grande insieme di input e li si elabora
  in un'unica passata pianificata (tipicamente notturna). La latenza per
  singolo esempio è irrilevante (ore vanno benissimo), mentre il throughput si
  massimizza sfruttando batch enormi. Caso d'uso: assegnare un punteggio a
  tutti i clienti di un database per una campagna, o pre-calcolare le
  raccomandazioni della home page.
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
un mucchio di richieste in blocco, e conta il *throughput*, cioè quante se ne
servono al secondo; a destra si risponde a una richiesta per volta, subito, e
conta la *latenza*, cioè quanto tempo passa fra la domanda e la risposta.
```

## Il modello dietro un'API

Nel regime online (il più comune e il più esigente) il modello vive dietro un
**endpoint**: un indirizzo a cui altri programmi mandano una richiesta (di
solito in JSON, via HTTP) e da cui ricevono la risposta. È lo sportello di
un'API, quello di cui parlava *Dal notebook alla produzione*.

Solo che l'indirizzo, da solo, non basta: dietro ci dev'essere una macchina su
cui il modello gira, e quella macchina deve comportarsi allo stesso modo
ovunque, altrimenti si torna al «sul mio computer funzionava». La risposta del
mestiere è chiudere il modello dentro una scatola che contiene anche tutto ciò
che gli serve per funzionare: il sistema, l'interprete Python, la versione
esatta di ogni libreria, i pesi. Sigillata la scatola, quella scatola si
comporta uguale su qualunque computer. Le tre parole che servono per parlarne
sono in {numref}`fig-immagine-container-volume`: la scatola sigillata, che
nessuno modifica più, si chiama **immagine**; una sua copia in funzione si
chiama **container**, ed è usa e getta, perché per averne un'altra basta
riaprire l'immagine; e siccome tutto ciò che il container scrive muore con
lui, quello che deve sopravvivere (i dati, i risultati) si tiene fuori, in uno
spazio agganciato dall'esterno che si chiama **volume**.

```{figure} ../figures/docker-per-data-scientist.svg
:name: fig-immagine-container-volume
:alt: "Tre oggetti distinti e il loro rapporto: l'immagine, una fotografia immutabile dell'ambiente con il codice e le dipendenze; il container, un'istanza in esecuzione di quell'immagine, che si può creare e distruggere; il volume, uno spazio di memoria persistente montato nel container, che sopravvive alla sua distruzione."
:width: 96%

Tre cose che si confondono di continuo. L'immagine non cambia, il container è
usa e getta, e tutto ciò che deve sopravvivere sta nel volume.
```

Questa distinzione è ciò che rende un
servizio riproducibile: se l'immagine contiene l'ambiente per intero, la stessa
identica versione del modello gira sul portatile di chi sviluppa e sul
**server** di produzione, cioè sul computer sempre acceso che risponde alle
richieste del mondo. Il container si può buttare e ricreare senza pensarci, ed è
esattamente il presupposto di ogni strategia di rilascio. Chi chiama non sa e
non deve sapere che dentro c'è una rete neurale: vede solo un servizio che,
dati certi ingressi, restituisce una previsione. Il programma che tiene il
modello in memoria e traduce le richieste in chiamate al `forward` si chiama
**model server**.

`````{tab} Elementare

È lo sportello di un ufficio. Dietro il vetro c'è l'impiegato (il modello) che
sa fare una cosa sola ma la sa fare bene. Tu non entri nel retro a rovistare
tra le pratiche: passi il tuo modulo dalla fessura e ti torna indietro la
risposta compilata. Lo sportello (l'*endpoint*) nasconde tutto il resto. E c'è
una regola di buon senso che vale oro: l'impiegato arriva la mattina, si siede
*una volta sola* e resta lì tutto il giorno. Sarebbe assurdo se andasse a casa
e tornasse a ogni singolo cliente. Con i modelli è identico: i pesi si
caricano in memoria una volta all'avvio del servizio, non a ogni richiesta;
caricarli costa secondi, e a ogni richiesta li si pagherebbe di nuovo.

`````

`````{tab} Superiore

Il model server carica i pesi una volta all'avvio e li tiene in memoria; a
ogni richiesta esegue solo il *forward*, in modalità inferenza. Sopra questa
logica si appoggia un livello di trasporto (REST/JSON per semplicità, o gRPC
per la bassa latenza e i payload binari) che però è, in sostanza, contorno.

Il problema serio non è esporre l'endpoint, è renderlo **riproducibile**. Un
modello dipende da una versione precisa di PyTorch, delle librerie di
pre-processing, perfino di CUDA: la stessa trappola del «sul mio computer
funzionava» vista in *Dal notebook alla produzione*, spostata
dall'addestramento al servizio. La risposta standard è la
**containerizzazione**: un'immagine Docker
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

Due righe fanno la differenza fra un giocattolo e un servizio corretto, e sono
le due che si dimenticano più spesso. La prima, `modello.eval()`, dice alla
rete che ha finito di studiare: alcuni suoi pezzi si comportano in un modo
mentre imparano e in un altro mentre rispondono (il *dropout*, che durante
l'addestramento spegne a caso una parte della rete per non farle imparare a
memoria, e la *BatchNorm*, che si tara sul gruppo di esempi che ha davanti), e
se nessuno glielo dice continuano a comportarsi da studenti: le previsioni
escono sbagliate senza che niente segnali l'errore. La seconda,
`torch.no_grad()`, spegne il meccanismo che durante l'addestramento annota ogni
operazione per poter poi tornare indietro e correggere i pesi. Qui non si
corregge più niente, e tenerne traccia costa memoria e tempo a ogni richiesta;
`torch.inference_mode()` è la versione ancora più drastica della stessa
rinuncia. Il resto (i punteggi grezzi che la rete produce, i *logit*, e la
`softmax` che li trasforma in probabilità) è esattamente il modello del
capitolo PyTorch, ora chiamato a rispondere invece che a imparare.

## Ottimizzare l'inferenza

Un servizio corretto può ancora essere troppo lento o troppo costoso. Le leve per
accelerare l'inferenza sono diverse da quelle dell'addestramento, ma una radice è
comune con il capitolo PyTorch: meno byte da spostare, più velocità.

La prima leva è il **batching dinamico**. Come abbiamo visto parlando di
prestazioni, la GPU rende al massimo su tanti conti identici in parallelo:
servire le richieste una per una la lascia mezza vuota. Il server allora
accumula per qualche millisecondo le richieste che arrivano, le impacchetta in
un unico batch e le passa al modello in un colpo solo: un filo di latenza in
più in cambio di molto più throughput. La seconda leva è **ridurre la
precisione** dei numeri, cioè scriverli con meno cifre. Dentro un calcolatore
un numero occupa un certo numero di caselle elementari, i *bit*: di solito
trentadue, e scendere a sedici (il `float16` della precisione mista,
introdotta nel capitolo PyTorch per l'addestramento) dimezza sia lo spazio
occupato sia la quantità di numeri da trasportare al secondo fra memoria e
processore, che è quasi sempre il vero collo di bottiglia. Si perde qualche
cifra decimale, ed è quasi gratis. La terza leva spinge
oltre, fino agli **interi**: la **quantizzazione** a `int8`
{cite}`jacob2018quantization`.

`````{tab} Elementare

È il trucco di quando mandi una foto su una chat: l'app la rimpicciolisce
prima di spedirla. Perdi un filo di nitidezza (se ci fai molto caso), ma il
file pesa un quarto e parte in un lampo. Quantizzare un modello è la stessa
idea applicata ai suoi numeri. I pesi, di norma, sono decimali finissimi
(tante cifre dopo la virgola): possono valere qualunque cosa. Quantizzare vuol
dire smettere di ammettere qualunque valore e tenerne pronti soltanto 256, come
i gradini di una scala (256 perché tanti sono i valori diversi che stanno in
una casella di memoria da otto cifre binarie, la più piccola che le macchine
maneggino comodamente); ogni peso viene arrotondato al gradino più vicino, e al
suo posto si scrive il **numero del gradino**, che è un intero piccolo. Ne
guadagni quattro volte in leggerezza, perché ogni numero prima occupava quattro
di quelle caselle e adesso ne occupa una, e spesso un bel taglio di velocità;
ne perdi un pizzico di precisione. Il patto conviene quasi sempre: spesso
l'accuratezza cala di una frazione di punto percentuale, un prezzo minuscolo
per un modello quattro volte più piccolo che gira anche su un telefono. Ma
«spesso» non è «sempre», e di quanto cali lo si scopre soltanto misurandolo su
quel modello lì.

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
$\mathbf{w} = [-1{,}00,\ -0{,}352,\ 0{,}20,\ 1{,}55,\ 0{,}073]$. Da minimo e massimo
($r_{\min} = -1{,}00$, $r_{\max} = 1{,}55$) si ricava la scala
$S = (r_{\max} - r_{\min}) / (127 - (-128)) = 2{,}55 / 255 = 0{,}01$ e lo
zero-point $Z = -128 - \mathrm{round}(r_{\min}/S) = -128 - (-100) = -28$.
Quantizzando ($q = \mathrm{round}(r/S) + Z$) si ottiene
$\mathbf{q} = [-128,\ -63,\ -8,\ 127,\ -21]$; ricostruendo ($\hat r = S(q - Z)$) si torna a
$\hat{\mathbf{r}} = [-1{,}00,\ -0{,}35,\ 0{,}20,\ 1{,}55,\ 0{,}07]$. L'**errore** è al più
$0{,}003$: non può superare mezzo gradino, $S/2 = 0{,}005$. Sul piano dei byte, i
cinque `float32` (20 byte) diventano cinque `int8` (5 byte) più i due parametri di
calibrazione $S$ e $Z$ condivisi da tutto il tensore: per uno strato con milioni di
pesi, quei due numeri sono trascurabili e la compressione è di circa **4×**.

Quel «condivisi da tutto il tensore» è però la forma più semplice, ed è anche
la più fragile. Il limite non sta nel mezzo gradino, sta in **quanto è largo il
gradino**: la scala $S$ la dettano gli estremi, quindi basta un solo canale con
pesi anomali per allargare il gradino di tutti gli altri e schiacciarli in poche
decine di livelli. Il rimedio costa una manciata di scalari per strato e si
chiama quantizzazione **per canale**: una coppia $S, Z$ per ogni riga della
matrice dei pesi. Su una matrice $256 \times 512$ con un singolo canale fuori
scala, l'errore relativo sull'uscita della moltiplicazione passa da circa il
$12\%$ (per tensore) allo $0{,}8\%$ (per canale), a parità di bit. È anche il
ponte verso i metodi per i grandi modelli linguistici della sezione su LLMOps,
che di quell'idea sono lo sviluppo.

In PyTorch la quantizzazione dinamica post-training è una riga, e l'esportazione
verso un runtime dedicato (indipendente da Python) è un'altra:

```{code-block} python
:class: pt-non-eseguibile

import torch
import torch.nn as nn

# Pesi in int8, attivazioni quantizzate al volo (dynamic quantization).
# API storica, oggi in via di sostituzione: avvisa che si migri a torchao.
modello_int8 = torch.quantization.quantize_dynamic(
    modello, {nn.Linear}, dtype=torch.qint8,
)

esempio = torch.randn(1, 784)
programma = torch.export.export(modello, (esempio,))    # il grafo, catturato
torch.onnx.export(modello, (esempio,), "modello.onnx")  # lo stesso, in ONNX
```

La `quantize_dynamic` è la variante più indolore (nessun dato di calibrazione
richiesto, adatta agli strati lineari), e va usata sapendo due cose. La prima è
che qui PyTorch adotta lo schema **simmetrico**, con $Z = 0$: nel tensore che ne
esce lo zero-point del conto fatto a mano qui sopra non si ritrova, perché quel
conto mostra il caso generale e la libreria ne implementa un caso particolare.
La seconda è che quell'API è dichiarata in uscita (il messaggio di deprecazione
rimanda a `torchao` e alla sua `quantize_`), quindi è materia da ricontrollare a
ogni aggiornamento invece che da imparare a memoria.

Sull'esportazione, invece, è cambiato il rapporto fra i due strumenti.
`torch.export` cattura il grafo del modello staccandolo dal codice Python che
l'ha addestrato; ONNX è un formato aperto con cui quel grafo si consegna a un
motore d'inferenza di terzi. Non sono due strade alternative, e chi le ha
imparate ai tempi di PyTorch 1.x se lo ricorda al contrario: da PyTorch 2.9
l'esportatore ONNX **passa** per `torch.export` (il parametro `dynamo` vale
`True` di serie) e delega a lui la cattura del grafo. Ne discende una
conseguenza pratica che coglie tutti di sorpresa: la traduzione in ONNX vive in
un pacchetto a parte, `onnxscript`, che non viene più installato insieme a
`torch`. Su un ambiente appena preparato quella riga solleva un
`ModuleNotFoundError`, e la cura è installarlo (`pip install onnxscript`), non
tornare all'esportatore vecchio.

Vale infine la stessa disciplina della precisione mista: la quantizzazione va
sempre **misurata** su un insieme di validazione, perché il calo di accuratezza
dipende dal modello e non è mai garantito trascurabile a priori.

`````

## Latenza e throughput: cosa promettere

Le due grandezze da promettere hanno nomi inglesi e significati semplici: la
**latenza** è quanto si aspetta una risposta, il tempo fra la domanda e la
risposta; il **throughput** è quante richieste il sistema smaltisce al secondo.
Tirano in direzioni opposte, ed è per questo che vanno promesse insieme.

Ottimizzato il servizio, resta la domanda più scomoda: che cosa **promettere**
a chi lo userà? I termini in gioco sono tre, e il gergo li confonde di
continuo. C'è la grandezza che si **misura** (in gergo l'**SLI**, *Service
Level Indicator*: per esempio il tempo entro cui risponde il 99% delle
richieste). C'è la promessa che i tecnici si danno da soli su quella grandezza,
il bersaglio che si impegnano a centrare (lo **SLO**, *Service Level
Objective*: «quel tempo sta sotto i 200 millisecondi»). E c'è il contratto
firmato con il cliente, che su quel bersaglio si appoggia e stabilisce le
conseguenze se la promessa salta (quello è lo **SLA**, *Service Level
Agreement*). Qui parliamo del secondo, del bersaglio che il team si dà, e il
punto delicato è come si sceglie il primo: la media, come indicatore, è una
bugia gentile.

```{figure} ../figures/latency-vs-throughput.svg
:name: fig-latenza-throughput
:alt: "Curva che lega throughput e latenza al variare della dimensione del batch. Ingrandendo il batch il throughput cresce rapidamente e poi si appiattisce, mentre il tempo di servizio della singola richiesta continua a salire: oltre il punto in cui la capacità copre il carico non c'è più una scelta che migliori entrambi, solo un tratto in cui il guadagno di throughput vale l'attesa aggiunta."
:width: 90%

Le due grandezze tirano in direzioni opposte, a una condizione: che il sistema
stia già stando dietro alle richieste che arrivano. Batch grandi servono più
utenti al secondo, e ciascuno di loro aspetta di più.
```

Il tratto piatto a destra in {numref}`fig-latenza-throughput` è quello da
riconoscere: oltre quel punto si continua a pagare in attesa senza più
guadagnare in capacità. Dove fermarsi non lo decide la curva ma il bersaglio
che il team si è dato (lo SLO), ed è questo il senso di sceglierlo prima.

La condizione posta nella didascalia merita il suo paragrafo, perché la curva
da sola inganna. Quella che sale è la latenza *di servizio*, cioè quanto ci
mette il modello a rispondere una volta che alla richiesta è arrivato il turno;
ma ciò che l'utente vive è l'attesa in coda **più** il servizio. Se il sistema
non sta dietro alle richieste che arrivano, la coda cresce senza fermarsi e
l'attesa esplode: in quel tratto ingrandire il batch **abbassa** la latenza e
alza il throughput insieme, perché aumenta la capacità e la coda si smaltisce.
Il compromesso comincia dopo, quando la capacità ha superato il carico. È anche
la ragione per cui in un servizio molto sollecitato il batching dinamico non è
un lusso: spesso è l'unica configurazione stabile.

`````{tab} Elementare

Immagina la coda alla posta. Se dico che «in media» si aspetta cinque minuti, ho
detto poco: magari novanta persone su cento passano in un minuto e dieci restano
impantanate mezz'ora, e la media di cinque minuti non la vive quasi nessuno. Quello
che conta davvero è la promessa sul *caso quasi peggiore*: «il 95% dei clienti è
servito entro dieci minuti». Con i modelli è identico. Non si promette la latenza
media, si promette che la *stragrande maggioranza* delle risposte arriva entro un
tempo dato. Perché il cliente scontento non è quello medio: è quello finito nella
coda lenta.

Quel «il 95% entro dieci minuti» ha un nome, e conviene impararlo perché torna
in ogni pagina che segue: si chiama **percentile**. La **p95** è il tempo entro
cui è servito il 95% delle richieste, cioè il caso peggiore su venti; la **p99**
è il caso peggiore su cento. La media, che è il numero che si guarda per
abitudine, non dice niente né dell'uno né dell'altro.

`````

`````{tab} Superiore

Si descrive la latenza con i suoi **percentili**, non con la media. La p50
(mediana) è il tempo entro cui risponde metà delle richieste; la **p95** e la
**p99** i tempi entro cui ne risponde il 95% e il 99%. La *coda* della
distribuzione, la p99, la p99.9: è ciò che governa l'esperienza reale sotto
carico, perché in un sistema che compone più servizi anche una piccola
frazione di richieste lente si propaga e degrada l'insieme. Uno SLO serio si
scrive sui percentili alti: «p99 sotto i 200 ms», non «latenza media 80 ms»,
che nasconde la coda.

Il secondo numero da promettere è il **throughput** sostenibile (richieste al
secondo), che con la latenza forma il classico compromesso: più batch grandi
alzano il throughput ma allungano la coda della latenza. Il terzo è economico,
il **costo per richiesta** (tempo di calcolo moltiplicato per il prezzo
dell'hardware) che spesso è il vero vincolo di progetto: un modello che
rispetta lo SLO ma costa dieci volte troppo per richiesta non è dispiegabile
{cite}`huyen2022designing`.

`````

C'è infine una cautela che riguarda *come* si sostituisce un modello con uno
nuovo, senza rompere niente. Non si spegne la versione vecchia e si accende la
nuova sperando bene: si procede per gradi. In un rilascio *canary* (dal
canarino che i minatori portavano sottoterra per accorgersi del gas prima degli
uomini) la nuova versione riceve dapprima una piccola frazione delle richieste
in arrivo, e la quota si allarga solo se le metriche tengono; in modalità
*shadow*, cioè «in ombra», la nuova versione riceve una copia delle richieste
reali ma le sue risposte non vengono servite a nessuno, solo confrontate con
quelle della vecchia; un test *A/B* divide gli utenti in due gruppi e misura su
richieste vere quale delle due versioni funziona meglio. Le tre tornano per
esteso, ciascuna con la sua analogia, nella sezione sul monitoraggio, che è
dove si decide quando usarle. È il lato «serving» della stessa prudenza che
l'anello MLOps chiede a ogni tappa: misurare prima di fidarsi.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- **Inferenza** è il modello che risponde, non che impara, e ci sono tre modi
  di organizzarla: tutto insieme quando fa comodo (il pane sfornato di notte),
  una richiesta per volta mentre qualcuno aspetta (il panino al momento), o su
  un flusso che non si ferma mai (il nastro del sushi).
- Il modello sta dietro uno **sportello**: chi lo interroga non sa e non deve
  sapere che cosa c'è dietro. E l'impiegato allo sportello si siede una volta
  sola la mattina: i pesi si caricano all'avvio del servizio, non a ogni
  richiesta.
- Perché lo sportello funzioni uguale ovunque, il modello si chiude in una
  scatola sigillata con dentro tutto quello che gli serve (l'**immagine**); una
  sua copia in funzione (il **container**) è usa e getta, e ciò che deve
  sopravvivere si tiene fuori.
- Per andare più veloci: servire più richieste in un colpo solo, scrivere i
  numeri con meno cifre, e al limite arrotondarli ai 256 gradini di una scala
  (la **quantizzazione**, come l'app che rimpicciolisce la foto prima di
  mandarla). Quattro volte più leggero, un pizzico meno preciso, **da misurare
  ogni volta**.
- Non si promette il tempo **medio** di risposta, che non lo vive quasi
  nessuno: si promette il caso quasi peggiore, «il 95% entro dieci minuti».
  Quel numero si chiama percentile, e il cliente scontento non è quello medio,
  è quello finito nella coda lenta.
- Una versione nuova non si accende di colpo per tutti: la si fa provare a
  pochi, o la si fa girare in ombra senza servirla, o si divide la sala in due
  e si confronta.
```
`````

`````{tab} Superiore
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
  misurare** sempre. La scala **per canale** costa una manciata di scalari per
  strato ed evita che un solo canale anomalo allarghi il gradino di tutti.
- L'esportazione stacca il modello dal codice che l'ha addestrato:
  `torch.export` cattura il grafo, e da PyTorch 2.9 **l'esportatore ONNX passa
  di lì** invece di essere la sua alternativa (serve `onnxscript`, che non è più
  una dipendenza di `torch`).
- I termini sono **tre**: l'**SLI** è ciò che si misura (la p99 della latenza),
  lo **SLO** la soglia che il team si impone su quell'indicatore, lo **SLA** il
  contratto con le penali. Uno SLO serio si scrive sui **percentili alti**
  (p95, p99), non sulla media, e va bilanciato con **throughput** e **costo per
  richiesta**.
- Il compromesso fra latenza e throughput vale **oltre il punto in cui la
  capacità copre il carico**: sotto quel punto la coda è instabile, e batch più
  grandi migliorano tutte e due le grandezze insieme.
- Le nuove versioni si rilasciano per gradi (*canary*, *shadow*, A/B) per non
  rompere niente in produzione.
```
`````
