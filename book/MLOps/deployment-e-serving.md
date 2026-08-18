# Servire un modello: dal file all'API

Alla fine di tutto, un modello addestrato è un file. Qualche centinaio di
megabyte di numeri su un disco (i pesi che nella sezione *Dal notebook alla
produzione* abbiamo imparato a versionare e archiviare) e nient'altro. Da solo
non fa niente: è inerte come uno spartito senza orchestra. «Metterlo in
produzione» significa esattamente dargli l'orchestra: un modo di ricevere
richieste dal mondo e di rispondere, in fretta e in modo affidabile, migliaia
di volte al minuto.

Le due sezioni precedenti (*Dal notebook alla produzione* e *Dati e pipeline*)
si sono fermate sul punto in cui i tre pezzi (dati, codice, pesi) sono
tracciabili e riproducibili. Questa affronta il passo successivo, il quarto
nodo dell'anello disegnato nella pagina d'apertura: **consegnare** il modello
al mondo, cioè metterlo in un posto dove chi ne ha il diritto possa
interrogarlo. In inglese consegnarlo si dice *deployment*; tenere acceso quel
posto, giorno dopo giorno, si dice *serving*. Sono le due parole che nel gergo
del mestiere coprono tutto quello che segue, e conviene averle in mano prima di
cominciare.

È una questione tanto ingegneristica quanto di aspettative, perché metà del
lavoro è decidere *che cosa promettere* a chi userà il servizio. E per una
volta il grosso non riguarda la rete neurale, ma tutto ciò che le sta attorno
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
centro, priorità opposte ai due lati. Le due priorità hanno un nome, e sono le
grandezze di cui questa sezione parlerà fino alla fine. La **latenza** è quanto
si aspetta una risposta, il tempo che passa fra la domanda e la risposta; il
**throughput** è quante risposte il sistema riesce a sfornare in un secondo.

```{figure} ../figures/mlops-serving.svg
:name: fig-mlops-serving
:alt: "Due schemi affiancati. A sinistra il regime batch: una pila di molte richieste entra nel modello e ne esce una pila di molte previsioni, con priorità al throughput e latenza non critica. A destra il regime online: una singola richiesta entra nel modello e ne esce una singola risposta, con priorità alla latenza bassa. In entrambi il modello carica i pesi una sola volta."
:width: 90%

Batch contro online: lo stesso modello, priorità opposte. A sinistra si
smaltisce un mucchio di richieste in blocco e conta quante se ne servono al
secondo; a destra si risponde a una richiesta per volta, subito, e conta quanto
si aspetta.
```

## Il modello dietro un'API

Nel regime online (il più comune e il più esigente) il modello vive dietro
l’**API** di cui parlava *Dal notebook alla produzione*: lo sportello elettronico
a cui un altro programma manda la domanda e da cui riceve la risposta, senza
sapere né dover sapere che cosa c'è dietro.

Serve una parola in più, perché uno sportello ha un indirizzo. L'indirizzo
preciso a cui si bussa si chiama **endpoint**, che alla lettera è «il capo
della linea»: l'API è lo sportello, l'endpoint è la targa con il numero civico.

Solo che l'indirizzo, da solo, non basta: dietro ci dev'essere una macchina su
cui il modello gira, e quella macchina deve comportarsi allo stesso modo
ovunque, altrimenti si torna al «sul mio computer funzionava». La risposta del
mestiere è chiudere il modello dentro una scatola che contiene anche tutto ciò
che gli serve per funzionare: il sistema, l'interprete Python, la versione
esatta di ogni libreria, i pesi. Sigillata la scatola, quella scatola si
comporta uguale su qualunque computer.

Le tre parole che servono per parlarne sono in
{numref}`fig-immagine-container-volume`. La scatola sigillata, che nessuno
modifica più, si chiama **immagine**. Una sua copia in funzione si chiama
**container**, ed è usa e getta: per averne un'altra basta riaprire
l'immagine. E siccome tutto ciò che il container scrive muore con lui, quello
che deve sopravvivere (i dati, i risultati) si tiene fuori, in uno spazio
agganciato dall'esterno che si chiama **volume**.

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
richieste del mondo. E il container si può buttare e ricreare senza pensarci,
il che tornerà utile in fondo a questa sezione: è quello che permette di tenere
in piedi due versioni del modello nello stesso momento, o di spegnere in un
istante quella nuova se si comporta male.

Chi chiama non sa e non deve sapere che dentro c'è una rete neurale: vede solo
un servizio che, dati certi ingressi, restituisce una previsione. Dietro lo
sportello c'è un programma che tiene il modello acceso in memoria e gli passa
le domande man mano che arrivano: si chiama **model server**.

`````{tab} Elementare

È lo sportello di un ufficio. Dietro il vetro c'è l'impiegato (il modello) che
sa fare una cosa sola ma la sa fare bene. Tu non entri nel retro a rovistare
tra le pratiche: passi il tuo modulo dalla fessura e ti torna indietro la
risposta compilata. Lo sportello (l’*endpoint*) nasconde tutto il resto. E c'è
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

Lo scheletro di un servizio d'inferenza, tolto tutto ciò che riguarda il
traffico in arrivo, sta in poche righe. La sostanza è tutta in tre gesti:
caricare i pesi **una volta sola**, dire alla rete che ha finito di studiare, e
spegnere il meccanismo che le serviva solo per imparare.

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

Il primo gesto, caricare i pesi all'avvio, l'abbiamo già visto con l'impiegato
che si siede una volta sola. Gli altri due sono una riga di codice ciascuno, e
sono i due che si dimenticano più spesso.

`````{tab} Elementare

Il secondo gesto è dire alla rete che **ha finito di studiare**. Sembra strano
doverglielo dire, e invece serve, perché alcuni suoi pezzi si comportano in due
modi diversi a seconda che stiano imparando o rispondendo.

Uno di questi pezzi, mentre la rete studia, ne spegne a caso dei pezzetti a
ogni ripetizione: è un trucco d'allenamento, serve a non farle imparare le
risposte a memoria, come un insegnante che copre a caso qualche riga del testo.
Un altro si regola guardando il gruppo di esempi che ha davanti, e quindi con
un esempio solo non saprebbe che pesci pigliare. Se nessuno gli dice che
l'allenamento è finito, quei pezzi continuano a comportarsi da studenti, e le
risposte escono sbagliate **senza che niente segnali l'errore**.

Il terzo gesto è spegnere il taccuino. Mentre impara, la rete annota ogni
singola operazione che fa, perché le servirà per tornare indietro e correggersi.
Quando risponde non si corregge più niente, e continuare a prendere appunti
costa memoria e tempo a ogni richiesta. Si spegne, e si va più veloci.

`````

`````{tab} Superiore

La prima riga, `modello.eval()`, commuta i moduli che hanno un comportamento
distinto fra addestramento e inferenza: il *dropout*, che in addestramento
azzera a caso una frazione delle attivazioni e in inferenza deve lasciarle
passare tutte, e la *BatchNorm*, che in addestramento normalizza sulle
statistiche del batch corrente e in inferenza deve usare le medie mobili
accumulate (con un batch di uno, le statistiche del batch non sono nemmeno
definite). Dimenticarla non solleva alcuna eccezione: produce solo predizioni
sbagliate.

La seconda, `torch.no_grad()`, disattiva la costruzione del grafo delle
operazioni che l’*autograd* userebbe per la retropropagazione. In inferenza
quel grafo non serve, e costruirlo costa memoria e tempo a ogni richiesta;
`torch.inference_mode()` è la variante più aggressiva della stessa rinuncia, che
disattiva anche il version counter dei tensori.

Il resto è il modello del capitolo PyTorch chiamato a rispondere invece che a
imparare: i *logit* grezzi in uscita dal `forward` e la `softmax` che li porta
su una distribuzione di probabilità.

`````

## Ottimizzare l'inferenza

Un servizio corretto può ancora essere troppo lento o troppo costoso. Le leve per
accelerare l'inferenza sono diverse da quelle dell'addestramento, ma una radice è
comune con il capitolo PyTorch: meno numeri da spostare, più velocità.

La prima leva è il **batching dinamico**, e qui la parola *batch* torna con un
significato diverso da quello di poco fa: non è più il regime di chi macina
tutto di notte, è solo il mazzetto di richieste che il server mette insieme
prima di passarle al modello. Il motivo è che la scheda grafica che fa i conti
(la **GPU**) è costruita per eseguire migliaia di operazioni identiche nello
stesso istante, e a servirle una richiesta per volta la si tiene quasi ferma: è
il punto su cui è costruito tutto il capitolo che le è dedicato. Il server
allora accumula per qualche millisecondo le richieste che arrivano, ne fa un
mazzetto e lo passa al modello in un colpo solo. Chi era arrivato per primo
aspetta quei pochi millisecondi in più; in cambio, nello stesso secondo, il
sistema ne serve molte di più.

La seconda leva è **ridurre la precisione** dei numeri, cioè scriverli con meno
cifre. Dentro un calcolatore ogni informazione è fatta di cifre binarie, i
*bit*, che valgono zero o uno, e un numero con la virgola di solito ne occupa
trentadue. Scendendo a sedici (è la scrittura che il capitolo PyTorch chiamava
`float16`, e usarla per una parte dei conti e non per tutti è la *precisione
mista* che là serviva ad addestrare più in fretta) si dimezza lo spazio
occupato, e con esso la quantità di byte da far scorrere fra memoria e
processore: i numeri restano tanti quanti erano, sono più corti. Ed è proprio
lo scorrere dei byte, quasi sempre, il vero collo di bottiglia. Si perde
qualche cifra dopo la virgola, ed è quasi gratis.

La terza leva spinge oltre, fino agli **interi**: la **quantizzazione** a
`int8` {cite}`jacob2018quantization`. Le due leve non sono alternative, sono un
seguito, e il conto si fa sempre rispetto ai trentadue bit di partenza: sedici
bit sono due volte più leggeri, otto bit quattro volte.

`````{tab} Elementare

È il trucco di quando mandi una foto su una chat: l'app la spedisce un po’
sgranata. Non la ritaglia e non la rimpicciolisce, i pixel restano tutti al
loro posto: sono i **colori** a diventare più grossolani, e a occhio quasi non
si vede. In cambio il file pesa un quarto e parte in un lampo. Quantizzare un
modello è la stessa idea applicata ai suoi numeri: i numeri restano tutti, ma
ciascuno è scritto peggio. I pesi, di norma, sono decimali finissimi
(tante cifre dopo la virgola): possono valere qualunque cosa. Quantizzare vuol
dire smettere di ammettere qualunque valore e tenerne pronti soltanto 256, come
i gradini di una scala. Il 256 non è scelto a caso: le macchine maneggiano i
bit a gruppi di otto, e con otto bit si scrivono $2^8 = 256$ valori diversi.
Ogni peso viene arrotondato al gradino più vicino, e al suo posto si scrive il
**numero del gradino**, che è un intero piccolo. Ne guadagni quattro volte in
leggerezza, perché prima ogni numero occupava trentadue bit e adesso ne occupa
otto, e spesso un bel taglio di velocità;
ne perdi un pizzico di precisione. Il patto conviene quasi sempre: spesso
l'accuratezza cala di una frazione di punto percentuale, un prezzo minuscolo
per un modello quattro volte più piccolo che gira anche su un telefono. Ma
«spesso» non è «sempre», e di quanto cali lo si scopre soltanto misurandolo su
quel modello lì.

`````

`````{tab} Superiore

La forma più economica è la **quantizzazione post-training** (*PTQ*): si prende
un modello già addestrato in `float32` e se ne convertono i pesi in interi,
senza riaddestrare. Il meccanismo (la mappa affine $r = S(q - Z)$ fra il numero
reale $r$ e l'intero $q$, con $S$ la larghezza di un gradino e $Z$ il livello
che rappresenta lo zero; l'errore limitato a mezzo gradino; e soprattutto il
fatto che
la larghezza del gradino la detti l'elemento più grande fra quelli che
condividono la scala) è costruito per esteso nel capitolo sull'efficienza
{cite}`jacob2018quantization`, insieme alle due conseguenze che qui si danno
per acquisite: che la **granularità** della scala sia la leva più economica, e
che nei modelli linguistici poche componenti anomale la dettino per tutte le
altre. Qui interessa la parte di servizio, cioè che cosa si scrive e che cosa
si misura.

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
che qui PyTorch adotta lo schema **simmetrico**, cioè quello con lo zero-point
fissato a zero: è il caso particolare della mappa affine, quello in cui la
scala basta da sola, ed è anche la forma su cui il capitolo sull’efficienza
costruisce tutto il meccanismo.
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

Ottimizzato il servizio, resta la domanda più scomoda: che cosa **promettere** a
chi lo userà? Le due grandezze in ballo sono quelle introdotte in cima alla
pagina: la **latenza**, quanto si aspetta una risposta, e il **throughput**,
quante risposte il sistema sforna in un secondo. Tirano in direzioni opposte, ed
è per questo che vanno promesse insieme. Ma prima di promettere qualcosa bisogna
decidere *quale numero* guardare, e qui il gergo confonde tre cose diverse.

Le tre cose sono quelle di qualunque promessa: **che cosa si guarda**, **che
cosa ci si impegna a fare** e **che cosa succede se non lo si fa**. Un treno le
ha tutte e tre: si guarda il ritardo all'arrivo, ci si impegna a stare sotto i
cinque minuti, e se si sfora il biglietto viene rimborsato.

La prima è la grandezza che si **misura**. Non è la latenza media, per una
ragione che fra poco vedremo con la coda alla posta: è il tempo entro cui
risponde la stragrande maggioranza delle richieste, per esempio il 99%. In
gergo si chiama **SLI**, *Service Level Indicator*.

La seconda è il **bersaglio** che i tecnici si danno da soli su quella
grandezza: «il tempo entro cui risponde il 99% delle richieste sta sotto i 200
millisecondi». È lo **SLO**, *Service Level Objective*.

La terza è il **contratto** firmato con il cliente, che su quel bersaglio si
appoggia e stabilisce il rimborso se la promessa salta: lo **SLA**, *Service
Level Agreement*.

Qui parliamo dello SLO, del bersaglio che il team si dà, e il punto delicato è
proprio quale grandezza mettere nel mirino: la media, come indicatore, è una
bugia gentile.

```{figure} ../figures/latency-vs-throughput.svg
:name: fig-latenza-throughput
:alt: "Curva che lega throughput e latenza al variare della dimensione del batch. Ingrandendo il batch il throughput cresce rapidamente e poi si appiattisce, mentre il tempo di servizio della singola richiesta continua a salire: oltre il punto in cui la capacità copre il carico non c'è più una scelta che migliori entrambi, solo un tratto in cui il guadagno di throughput vale l'attesa aggiunta."
:width: 90%

Le due grandezze tirano in direzioni opposte, a una condizione: che il sistema
stia già smaltendo le richieste alla velocità con cui arrivano. Batch grandi
servono più utenti al secondo, e ciascuno di loro aspetta di più. (Sull'asse
verticale il throughput è contato in token al secondo invece che in risposte al
secondo, perché la curva è disegnata su un modello che genera testo: la forma
della curva è la stessa in tutti e due i casi.)
```

Il tratto piatto a destra in {numref}`fig-latenza-throughput` è quello da
riconoscere: oltre quel punto si continua a pagare in attesa senza più
guadagnare in capacità. Dove fermarsi non lo decide la curva ma il bersaglio
che il team si è dato (lo SLO), ed è questo il senso di sceglierlo prima.

La condizione posta nella didascalia merita il suo paragrafo, perché la curva
da sola inganna. Il tempo segnato sull'asse orizzontale è la latenza *di
servizio*: quanto ci mette il modello a rispondere una volta che alla richiesta
è arrivato il turno. Ma ciò che l'utente vive è l'attesa in fila **più** il
servizio, e la fila non compare nella curva.

Torniamo al forno del fornaio, ma stavolta di giorno, col negozio aperto e la
gente in fila davanti al bancone: all'inizio della sezione il forno lavorava di
notte apposta perché non ci fosse nessuno ad aspettare, e adesso invece
l'attesa è tutto il problema. E con dei numeri. Un'infornata da una pagnotta
sola richiede mezz'ora, quindi il forno ne sforna due all'ora; una da cento
richiede
quaranta minuti, un po’ di più, ma di pagnotte ne consegna centocinquanta
all'ora. Adesso mettiamo che i clienti che entrano nel negozio siano sessanta
all'ora.

Con il forno da una pagnotta la fila **non smette mai di allungarsi**: entrano
sessanta persone e ne escono due, quindi ogni ora ne restano dentro
cinquantotto in più, e chi arriva alle undici aspetta più di chi è arrivato
alle dieci, per sempre. Con il forno da cento, che ne fa centocinquanta contro
sessanta, la fila si smaltisce e nessuno aspetta più di un'infornata. La singola
infornata è più lenta, e ciononostante **tutti aspettano meno**.

Ci sono quindi tre situazioni, non due, e vale la pena tenerle distinte. Finché
il forno non sta dietro ai clienti, ingrandire l'infornata migliora tutto:
sforna di più *e* fa aspettare meno. Quando il forno ha superato la richiesta,
comincia il vero compromesso: allargare ancora fa sfornare qualcosa in più e fa
aspettare qualcosa in più, e sta a chi decide capire se lo scambio conviene.
Ancora oltre, nel tratto piatto della curva, il forno non sforna più niente in
più e si continua solo ad aspettare: lì non si scambia niente, si perde e basta.

È anche la ragione per cui in un servizio molto sollecitato il batching dinamico
non è un lusso: spesso è l'unica configurazione stabile.

`````{tab} Elementare

Immagina la coda alla posta, e conta davvero. Su cento clienti, ottanta escono
dall'ufficio due minuti dopo esserci entrati, quindici ci mettono dieci minuti
e cinque restano impantanati quaranta minuti: sono tempi porta a porta, fila
compresa, che è esattamente quello che vive chi aspetta. L'attesa media è
$(80 \times 2 + 15 \times 10 + 5 \times 40)/100 = 5{,}1$ minuti, cioè circa
cinque. Ma **cinque minuti non li aspetta nessuno**: chi entra alla posta
aspetta due minuti, o dieci, o quaranta. La media è un numero che non descrive
l'esperienza di nessuno dei presenti.

Quello che conta davvero è la promessa sul *caso quasi peggiore*. Con questi
stessi numeri si può dire: «novantacinque clienti su cento sono serviti entro
dieci minuti», e stavolta è una frase vera e verificabile (i primi ottanta in
due minuti più i quindici in dieci fanno novantacinque). Restano fuori i cinque
sfortunati, e sono loro il problema del direttore dell'ufficio.

Con i modelli è identico. Non si promette il tempo medio, si promette che la
*stragrande maggioranza* delle risposte arriva entro un tempo dato. Perché il
cliente scontento non è quello medio: è quello finito nel gruppetto lento.

Quel «novantacinque su cento entro dieci minuti» è la stessa idea che poche
righe più su avevamo chiamato «il tempo entro cui risponde il 99% delle
richieste», e il nome tecnico è **percentile**. La **p95** è il tempo entro cui
è servito il 95% delle richieste, cioè il caso peggiore su venti; la **p99** è
il caso peggiore su cento. Quale dei due mettere nel mirino lo decide chi
promette, ed è una scelta di severità: la p99 è più difficile da rispettare
della p95, perché lascia fuori dieci volte meno gente. La media, che è il numero
che si guarda per abitudine, non dice niente né dell'una né dell'altra.

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
nuova sperando bene: si procede per gradi, e i gradi sono tre, in ordine di
quanto si sta esponendo il pubblico.

Il primo grado non espone nessuno. In modalità *shadow*, cioè «in ombra», la
nuova versione riceve una copia delle richieste vere e produce le sue risposte,
ma quelle risposte non vengono date a nessuno: si mettono da parte e si
confrontano con quelle della vecchia.

Il secondo grado espone pochi. In un rilascio *canary* la nuova versione
risponde davvero, ma solo a una piccola frazione delle richieste, e la quota si
allarga solo se i numeri tengono; il nome viene dal canarino che i minatori
portavano sottoterra per accorgersi del gas prima degli uomini.

Il terzo grado espone metà. Un test *A/B* divide gli utenti in due gruppi e
misura su richieste vere quale delle due versioni funziona meglio.

Le tre tornano in «Monitoraggio e drift», ciascuna con la sua analogia e con la
domanda a cui risponde. È il lato «serving» della stessa
prudenza che l'anello MLOps chiede a ogni tappa: misurare prima di fidarsi.

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
  scatola sigillata con dentro tutto quello che gli serve (l’**immagine**); una
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
- Una versione nuova non si accende di colpo per tutti, ma per gradi: prima la
  si fa girare **in ombra**, senza servirne le risposte a nessuno; poi la si fa
  provare a **pochi**; e infine si dividono gli utenti in due gruppi, si dà a
  ciascun gruppo una versione diversa, e si guarda quale va meglio.
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
  `int8`** con la mappa affine $r = S(q - Z)$, cioè scala e livello dello zero
  {cite}`jacob2018quantization`: circa
  4× di memoria in meno al prezzo di un piccolo calo di accuratezza, **da
  misurare** sempre. La scala **per canale** costa una manciata di scalari per
  strato ed evita che un solo canale anomalo allarghi il gradino di tutti.
- L'esportazione stacca il modello dal codice che l'ha addestrato:
  `torch.export` cattura il grafo, e da PyTorch 2.9 **l'esportatore ONNX passa
  di lì** invece di essere la sua alternativa (serve `onnxscript`, che non è più
  una dipendenza di `torch`).
- I termini sono **tre**: l’**SLI** è ciò che si misura (la p99 della latenza),
  lo **SLO** la soglia che il team si impone su quell'indicatore, lo **SLA** il
  contratto con le penali. Uno SLO serio si scrive sui **percentili alti**
  (p95, p99), non sulla media, e va bilanciato con **throughput** e **costo per
  richiesta**.
- Il compromesso fra latenza e throughput vale **oltre il punto in cui la
  capacità copre il carico**: sotto quel punto la coda è instabile, e batch più
  grandi migliorano tutte e due le grandezze insieme.
- Le nuove versioni si rilasciano **per gradi di esposizione** (*shadow*, che
  non serve nessuno; *canary*, che serve pochi; A/B, che divide il traffico a
  metà) per non rompere niente in produzione.
```
`````
