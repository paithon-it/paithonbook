# A che serve saperlo, e dove sbaglia

Abbiamo speso due sezioni per ottenere un numero. Adesso la domanda che le
giustifica: a che cosa serve. Sono tre mestieri (comprimere, scegliere fra
ipotesi, riconoscere ciò che è fuori posto), e sul terzo questa famiglia ha
preso una lezione che vale più degli altri due.

## Primo: comprimere

Sapere quanto un dato è probabile e saperlo comprimere sono la stessa cosa
detta in due modi, e la {doc}`sezione sulla teoria dell'informazione
</Matematica/teoria-informazione>`, nei richiami di matematica, l'ha già
stabilito: il numero di bit che servono per scrivere un messaggio con il codice
migliore possibile è $-\log_2 p$ del messaggio, cioè quante volte bisogna
dimezzare 1 per arrivare alla sua probabilità. Un messaggio che ha una
probabilità su otto costa 3 bit, perché $2 \times 2 \times 2 = 8$; uno che ne ha
una su mille ne costa circa dieci, perché $2^{10} = 1024$. E il codice migliore
quel numero lo raggiunge davvero, non solo come ordine di grandezza: lo supera
di un paio di bit su tutto il file, che è quanto costa scrivere in bit interi
una quantità che intera non è.

Un modello che sa dire $p(\mathbf{x})$ *è* un compressore, e non per
analogia. Gli si dà un file, lui dà una probabilità, e un codificatore
aritmetico, il programma che trasforma una sequenza di probabilità in una
sequenza di zeri e di uni lunga appunto $-\log_2 p$ (lo stesso del limite della
compressione nei richiami), la trasforma in bit; il file si ricostruisce
esattamente. L'equivalenza ha due condizioni. La prima è che $p$ sia
normalizzata: per la disuguaglianza di Kraft un codice con quelle lunghezze
esiste solo se le probabilità sommano al più a uno, e un punteggio che non fa
uno, come quello dei modelli a energia, non si traduce in un codice. La
seconda è che chi decomprime sappia rifare le stesse probabilità nello stesso
ordine: con un modello autoregressivo comprimere costa un passaggio della rete,
perché il file è tutto lì, ma decomprimere ne costa uno per valore, come
generare. È la ragione per cui in questa letteratura la qualità non si misura
in punti su cento ma in **bit per dimensione**: quanti bit costa, in media, ogni
numero dell'immagine. Un numero più basso vuol dire un modello migliore e un
file più piccolo, ed è la stessa frase.

`````{tab} Elementare

Una figurina a colori di 32 pixel per lato è fatta di $32 \times 32 \times 3 =
3.072$ numeri, ognuno fra 0 e 255. Chi non sa niente di come sono fatte le
immagini deve spendere 8 bit per ciascuno, cioè 3.072 byte: è il file grezzo.
Un modello autoregressivo ben fatto costa 2,92 bit per numero, e
$3.072 \times 2{,}92$ fa 8.970 bit, cioè meno di 1.130 byte. Stesso contenuto,
ricostruibile senza perdere niente, in poco più di un terzo dello spazio.
L'unica cosa che il modello ha in più è sapere che cosa aspettarsi.

Un'avvertenza sulla parola «esatto», perché vale per una delle due strade e
non per l'altra. Chi mette i pixel in fila lavora sui numeri interi che
l'immagine ha davvero, e il conto è quello e basta. Un flusso invece dà
un'altezza di curva, non una probabilità (è la differenza fra altezza e area
della sezione precedente), e su numeri interi questo apre un imbroglio: la
curva può farsi altissima e sottilissima proprio sopra ciascuno dei 256 valori
ammessi, con l'area totale sempre uguale a uno e un'altezza che sale quanto si
vuole. Il modello dichiarerebbe allora densità enormi e un file che non costa
quasi niente. Il rimedio è spalmare ogni pixel su tutta la sua casella: invece
di 128 si usa un numero preso a caso fra 128 e 129, così che un picco sottile
non serve più a niente. Il numero che esce misura i dati spalmati, e rispetto
al file vero è prudente: il file vero non costa mai di più, semmai un po’ meno.
Va benissimo per confrontare due modelli misurati allo stesso modo, ma non è
la stessa cosa.

`````

`````{tab} Superiore

Il codice della sezione precedente stampa nat di una densità continua su dati
standardizzati, la letteratura riporta bit per dimensione su pixel interi, e il
ponte fra i due è una divisione, purché $p$ sia misurata nella scala giusta:

$$
\text{bit/dim} = \frac{-\log p(\mathbf{x})}{D \ln 2},
$$

con $D$ il numero di componenti del dato ($D = 32 \times 32 \times 3 = 3.072$
per un'immagine di CIFAR-10) e $p$ una probabilità sui valori interi, oppure
una densità sui pixel dequantizzati nella loro scala originale $[0, 256)^D$. Se
il modello lavora sui pixel riscalati in $[0, 1]$, al numeratore si aggiunge
$D \ln 256$, cioè otto bit per dimensione: è il logaritmo del fattore $256^D$
di cui il riscalamento restringe il volume, e senza quel termine i numeri
escono irrisori o negativi. Per la stessa ragione la log-densità delle due lune,
misurata su coordinate standardizzate, non ha un equivalente in bit. Per un
modello linguistico la grandezza corrispondente sono i bit per simbolo, che di
solito si riportano come perplessità, $2^{\text{bit per simbolo}}$, quella dei
richiami di teoria dell'informazione. Su CIFAR-10, che è il banco di prova
storico della famiglia, in bit per dimensione sul test (meno è meglio): NICE
$4{,}48$; RealNVP $3{,}49$; Glow $3{,}35$; PixelCNN $3{,}14$; Gated PixelCNN
$3{,}03$; PixelRNN $3{,}00$; PixelCNN++ $2{,}92$. E in cima, a $8{,}00$, il
modello che non sa niente. I valori degli autoregressivi sono esatti, quelli
dei flussi limiti superiori, per la dequantizzazione del capoverso che segue.

L'elenco è quello storico, fermo al 2017, e si è mosso: nella tabella di VDM
{cite}`kingma2021variational` un autoregressivo a Transformer scende a
$2{,}80$, e VDM stesso, un modello di diffusione addestrato su un limite
variazionale, a $2{,}65$, meglio di tutti i modelli esatti con un numero che è
soltanto un limite superiore del costo. La contraddizione è apparente. Per
comprimere un limite basta, con lo schema *bits-back* già incontrato con i
{doc}`VAE </ModelliLatenti/il-salto-probabilistico>`, e VDM lo usa per una
compressione senza perdita vicina all'ottimo teorico. L'esattezza serve quando
si vuole la probabilità di un singolo dato, come nel riconoscere un dato
anomalo, e non solo la media su un insieme di prova.

Una precisazione che la parola «esatta» rischia di far perdere. Un modello
autoregressivo sui 256 livelli dà la probabilità di *quell'immagine lì*, e il
conto è esatto senza aggiunte. Un flusso invece è continuo, e una densità
continua su valori interi è mal posta (l'entropia differenziale di una
distribuzione discreta è $-\infty$, e la verosimiglianza si può gonfiare a
piacere): si aggiunge allora rumore uniforme ai pixel, cioè si
**dequantizza**, e quel che si ottiene è la verosimiglianza esatta dei dati
dequantizzati. Il legame con il numero che interessa lo dà una disuguaglianza
di Jensen {cite}`theis2016note`. Se $q$ è la densità del flusso sui pixel
dequantizzati, la probabilità che assegna all'immagine intera $\mathbf{x}$ è
$P(\mathbf{x}) = \int_{[0,1)^D} q(\mathbf{x} + \mathbf{u})\,\mathrm{d}\mathbf{u}$,
e per la concavità del logaritmo
$\mathbb{E}_{\mathbf{u}}\big[\log q(\mathbf{x} + \mathbf{u})\big] \le \log
P(\mathbf{x})$: la log-verosimiglianza media del modello continuo sta sotto
quella del corrispondente modello discreto, quindi i bit per dimensione
riportati per un flusso sono un limite *superiore* al costo di codifica vero.
Conservativo, il che va benissimo per confrontare due flussi nella stessa
convenzione, ma non è la stessa cosa. La convenzione conta quanto il modello:
la scala dei pixel sposta il numero di otto bit per dimensione, e Glow riporta
per CIFAR-10 $3{,}35$ bit per dimensione sulle immagini a 8 bit e $1{,}67$
sulle stesse immagini ridotte a 5 bit per canale {cite}`kingma2018glow`. Due
righe di una tabella si confrontano solo se dati, scala e dequantizzazione sono
gli stessi. Il problema, poi, è più largo dei soli valori interi: dati che
stanno vicino a una varietà di dimensione minore di $D$ non hanno una densità
rispetto al volume, e un flusso, che per costruzione ne definisce una positiva
su tutto $\mathbb{R}^D$, può solo inseguire una densità degenere. Il rumore
aggiunto, uniforme o gaussiano come in TarFlow {cite}`zhai2025tarflow`, dà al
bersaglio una densità vera.

Che verosimiglianza e qualità dei campioni si possano separare lo mostrano due
costruzioni discusse da Theis e colleghi {cite}`theis2016note`. La prima, che
riprendono da un argomento di van den Oord e Dambre, mescola un buon modello
$p$ con un generatore di rumore $r$, $q = 0{,}01\,p + 0{,}99\,r$: poiché
$q \ge 0{,}01\,p$ ovunque,
$\log_2 q(\mathbf{x}) \ge \log_2 p(\mathbf{x}) - \log_2 100$, cioè $q$ perde
al più $6{,}6$ bit per immagine, $0{,}002$ bit per dimensione su CIFAR-10,
mentre novantanove campioni su cento sono rumore. La seconda è loro: un modello
che ripete le immagini di addestramento con un nucleo strettissimo attorno a
ciascuna dà campioni perfetti e una verosimiglianza pessima sui dati di prova.
Una differenza di qualità che salta all'occhio può quindi valere, in bit per
dimensione, meno dell'ultima cifra che le tabelle riportano.

`````

C'è anche un motivo per cui questo modo di misurare piace, e la
{doc}`sezione sull'addestramento avversario </GAN/come-funziona>` lo rende
evidente per contrasto. Confrontare due GAN richiede il FID, che richiede una
terza rete addestrata da qualcun altro, che a sua volta ha le sue idee su che
cosa sia una fotografia. Confrontare due modelli a verosimiglianza esatta
richiede un numero solo, misurato su dati mai visti e senza una terza rete che
faccia da giudice, a patto che i due numeri siano misurati allo stesso modo:
stessi dati, stessa scala dei pixel, stessa dequantizzazione. Le stesse
immagini a 8 bit e a 5 bit per canale, o in scala $[0, 1]$ invece che
$[0, 256)$, danno numeri che non si possono mettere in fila.

```{admonition} Un numero solo, ma su che cosa
:class: caution
Quel numero non misura quanto le immagini generate sono belle, ed è un
malinteso frequente su questa famiglia: i due giudizi possono divergere fino a
diventare quasi indipendenti. Lucas Theis, Aäron van den Oord e Matthias Bethge
{cite}`theis2016note` lo mostrano già nel 2016: la verosimiglianza media, le
stime di Parzen (che giudicano un modello mettendo una piccola campana attorno a
ciascuno dei suoi campioni e guardando quanto i dati veri ci cadono vicino) e
la qualità visiva dei campioni sono, in alta dimensione, criteri largamente
slegati, e «ottenere buoni risultati su un criterio non implica necessariamente
ottenerne sugli altri». Un modello può avere una verosimiglianza eccellente e
produrre campioni mediocri, e viceversa. La loro conclusione è la regola
pratica da tenere: un modello generativo va valutato rispetto all'uso per cui
lo si vuole, non rispetto al numero più comodo da stampare.
```

## Secondo: ordinare, confrontare, scegliere

Il secondo mestiere è meno vistoso e più usato di quanto sembri. Avere
$p(\mathbf{x})$ vuol dire poter mettere in ordine delle ipotesi e sceglierne
una. Un riconoscitore del parlato che esita fra «ho visto un gatto» e «o visto
un gatto», che all'orecchio suonano uguali, chiede a un modello del linguaggio
quale delle due frasi sia più probabile, e riordina le sue ipotesi secondo la
risposta (la {doc}`sezione sui modelli di riconoscimento
</SpeechRecognition/modelli-asr>` lo fa con le prime $n$ trascrizioni); un
programma che deve ricostruire i pixel mancanti di una fotografia confronta le
ricostruzioni candidate con lo stesso criterio. In questi ruoli il modello di
densità non genera niente: sta dentro un sistema più grande e dà voti
confrontabili. È anche il ruolo da cui i flussi sono nati
{cite}`rezende2015variational`, l'inferenza variazionale, il mestiere di
approssimare una distribuzione difficile con una che si sa maneggiare: nei VAE
del capitolo sui modelli latenti l'encoder propone per il latente una
distribuzione semplice, e un flusso la rende più flessibile lasciandone
calcolabile la densità, che è proprio ciò che serve all'ELBO.

## Terzo: riconoscere ciò che è fuori posto, e dove la verosimiglianza sbaglia

L'applicazione più ovvia di tutte è questa. Ho addestrato il modello sulle mie
fotografie; adesso me ne arriva una nuova; se il modello le dà una probabilità
bassissima, vuol dire che è diversa da quelle che ho visto. Rilevamento di
anomalie, controllo qualità, allarme quando il mondo cambia sotto ai piedi del
sistema (è il tema della {doc}`sezione sul sorvegliare un modello vivo
</MLOps/monitoring-e-drift>`). Sembra il criterio più solido che ci sia.

Usato così, il criterio fallisce, e il modo in cui fallisce è così netto da
essere diventato un caso di scuola.

Eric Nalisnick e colleghi {cite}`nalisnick2019do` addestrano flussi, VAE e
PixelCNN su CIFAR-10, una raccolta di fotografie di cani, camion, cavalli e
altre cose comuni. Poi mostrano a quei modelli SVHN, fotografie di numeri
civici: un'altra raccolta, con un altro contenuto, che il modello non ha mai
visto. E misurano la verosimiglianza. Il risultato, nelle loro parole, è che
«la densità appresa» da quei modelli «non riesce a distinguere immagini di
oggetti comuni come cani, camion e cavalli da quelle di numeri civici,
assegnando una verosimiglianza più alta a queste ultime quando il modello è
stato addestrato sulle prime».

Non «faticano a distinguere»: sbagliano nella direzione sbagliata, e con
sicurezza. Il modello dichiara più probabili le immagini che non ha mai visto.
Gli autori ne traggono un avvertimento più che una condanna: non usare la
densità di questi modelli per riconoscere i dati estranei finché il suo
comportamento fuori dalla distribuzione di addestramento non è capito meglio.

`````{tab} Elementare

Una spiegazione plausibile è tanto semplice quanto scomoda, e si vede meglio
lontano dalle immagini.

Un modello addestrato su testi italiani riceve una pagina e dice quanto la
trova probabile. Gli arriva un foglio bianco, con scritto in mezzo
«aaaaaaaaaaaa». Non è italiano, non l'ha mai visto, è fuori posto sotto ogni
criterio. Ma è anche facilissima: ogni carattere è identico al precedente,
quindi il modello ci azzecca ogni volta, e la probabilità che ne esce è
altissima, più alta di quella di una pagina di italiano vero, piena di scelte
difficili.

Le fotografie dei numeri civici fanno la stessa cosa: sono immagini più lisce,
più povere di dettaglio, più prevedibili di quelle di un bosco o del pelo di un
cane. Il modello le trova facili, e «facile» per lui vuol dire «probabile». Al
contrario lo sbaglio non succede: un modello addestrato sui numeri civici trova
le fotografie di cani poco probabili, come ci si aspetta, perché per lui sono
difficili.

C'è poi un secondo motivo, ancora più strano, e varrebbe anche per un modello
perfetto. In una vita di letture, una pagina fatta di «a» in fila non capita
mai, anche se ciascuna di quelle pagine è più probabile di qualunque pagina
vera: le pagine vere sono tantissime, ognuna poco probabile, e tutte insieme si
prendono quasi tutta la probabilità. Stanno in una fascia di mezzo, mai
facilissime e mai impossibili, e nessuna arriva vicino alla pagina più
probabile di tutte, che è il record. «L'ho già visto» abita quella fascia, e il
record ne sta fuori.

Ed è qui la lezione, che vale ben oltre questa famiglia: «quanto è probabile» e
«l'ho già visto» sono due domande diverse. Le abbiamo confuse perché nella
nostra testa vanno insieme, e per un modello no. Per sapere se un dato viene
dalla distribuzione su cui il modello è stato addestrato, una strada proposta è
guardare se quel dato cade nella fascia di mezzo invece che se ha battuto un
record; se basti, è ancora discusso.

C'è un lato lieto. La convinzione che una probabilità bassa bastasse a scovare
un intruso girava da anni, ed è bastato un esperimento piccolo e riproducibile a
smontarla: tre modelli costruiti in modi diversi, addestrati sulla stessa
raccolta di fotografie e misurati su un'altra, e tutti e tre sbagliano nello
stesso verso. Uno sbaglio condiviso da macchine così diverse sta nella domanda
che si fa loro, più che in una di loro.

`````

`````{tab} Superiore

Il fenomeno si presenta su modelli ben addestrati e con buona verosimiglianza
sui dati di prova, ed è stabile fra famiglie diverse (flussi, VAE, PixelCNN),
il che rende implausibile che sia un difetto dell'addestramento o di una
singola architettura. I numeri, per una versione ridotta di Glow addestrata su
CIFAR-10: $3{,}46$ bit per dimensione sul test di CIFAR-10 (contro i $3{,}35$
del modello originale), $2{,}39$ sul test di SVHN, cioè più di un bit in meno
per ogni numero dell'immagine. Il fenomeno non è simmetrico: un Glow
addestrato su SVHN non assegna a CIFAR-10 una verosimiglianza più alta
{cite}`nalisnick2019do`, un'asimmetria coerente con la lettura per complessità
che segue.

Una spiegazione la danno gli autori stessi, restringendo i flussi a
trasformazioni a volume costante, che si prestano a un conto in forma chiusa:
la differenza di verosimiglianza si spiega con la posizione e la varianza dei
dati e con la curvatura del modello. Una lettura molto discussa, che Serrà e
colleghi {cite}`serra2020input` propongono come ipotesi e sostengono con una
serie di esperimenti, è che la densità in alta dimensione sia dominata dalla
**complessità** dell'input più che dalla sua appartenenza alla distribuzione:
dati più lisci ricevono log-densità più alte per ragioni che hanno poco a che
vedere con il supporto della distribuzione di addestramento, e SVHN, con
sfondi uniformi e poche texture, è esattamente questo rispetto a CIFAR-10. Da
una stima della complessità dell'input ricavano un punteggio, leggibile come
un rapporto di verosimiglianze, che regge il confronto con i metodi dedicati.
Le altre letture non la escludono. Kirichenko e colleghi
{cite}`kirichenko2020why` mostrano che i flussi imparano soprattutto
correlazioni locali fra pixel vicini, comuni a tutte le immagini naturali e non
specifiche della raccolta. Le Lan e Dinh {cite}`lelan2021perfect` mostrano che
nemmeno un modello perfetto basterebbe: con il cambio di variabile della
sezione precedente si costruisce una riparametrizzazione invertibile dei dati
in cui ogni punto riceve la densità che si vuole, e il punteggio di densità si
ribalta senza che i dati abbiano perso un'informazione. Ne discende comunque la
stessa diagnosi: la densità non è una statistica di appartenenza, e usarla come
tale confonde $p_\theta(\mathbf{x})$ alto con
$\mathbf{x} \in \operatorname{supp} p_{\text{dati}}$.

Va aggiunta una precisazione di geometria in alta dimensione, che rende il
risultato meno paradossale di quanto sembri: la massa di
probabilità di una gaussiana non sta nel punto di densità massima ma in un
guscio a distanza $\approx \sqrt{D}$ dall'origine (su CIFAR-10, dove
$D = 3.072$, la norma di un campione gaussiano standard vale in media
$55{,}4$). Un campione tipico non
è quindi un campione ad alta densità, e i due concetti divergono tanto più
quanto $D$ cresce. Cercare l'atipico guardando la densità è, letteralmente,
guardare l'asse sbagliato, ed è la strada che lo stesso gruppo prende subito
dopo, nel 2019, proponendo al posto della densità un test di **tipicità**
{cite}`nalisnick2019typicality`. È una proposta, e non chiude la questione:
la costruzione di Le Lan e Dinh ribalta anche i test di tipicità applicati a un
dato solo.

`````

## Il ponte: dai flussi al *flow matching*

Resta da dire da dove viene la parola «flusso» che il capitolo precedente usa
per il *flusso rettificato* (*rectified flow*). La {doc}`sezione sul flow
matching </ModelliDiffusione/flow-matching>` chiama flusso la mappa con cui un
campo di velocità trasporta i punti dello spazio, senza dire che cosa abbia a
che fare con i flussi normalizzanti costruiti fin qui. La parentela passa per
un limite.

Un flusso normalizzante è una composizione di passi invertibili, ciascuno
vincolato ad avere un determinante calcolabile. Se i passi diventano
infinitamente piccoli, la composizione diventa un'equazione differenziale
ordinaria, $\mathrm{d}\mathbf{x}_t / \mathrm{d}t = \mathbf{v}_t(\mathbf{x}_t)$:
una rete dichiara una velocità $\mathbf{v}_t$ in ogni punto e in ogni istante,
e il dato si lascia trasportare fino alla gaussiana, con un movimento continuo
al posto di una scala di gradini. È il flusso normalizzante continuo delle
*neural ODE* di Ricky Chen e colleghi {cite}`chen2018neural`. Nel limite il
logaritmo del determinante diventa l'integrale nel tempo della traccia della
jacobiana del campo, cioè della sua divergenza; FFJORD
{cite}`grathwohl2019ffjord` stima quella traccia a caso con lo stimatore di
Hutchinson, invece di calcolarla. Cadono i vincoli sull'architettura del
campo, purché resti abbastanza regolare da tenere la trasformazione
invertibile, e il prezzo si sposta sul calcolo: per avere la verosimiglianza, e
per generare, bisogna integrare l'equazione, ogni volta e per ogni esempio.

`````{tab} Elementare

Torniamo all'acqua sul tavolo, ma questa volta il tavolo non si stira a
scatti: l'acqua scorre. Ogni goccia ha in ogni istante una velocità, come una
foglia in un torrente, e il viaggio dura un tempo fissato, alla fine del quale
l'acqua è arrivata sulla nuvola semplice. Per sapere quanto si è alzata o
abbassata non serve più rifare il conto del determinante su tutta la tabella:
basta guardare, istante per istante, se attorno a una goccia la corrente
allarga o stringe, cioè se le gocce vicine si allontanano o si avvicinano, e
sommare quei piccoli allargamenti lungo il viaggio. Quell'allargamento istante
per istante si chiama divergenza, e per misurarlo basta la diagonale della
tabella: quanto ogni coordinata stira sé stessa.

Anche la diagonale costa, quando le coordinate sono un milione. Il trucco è non
misurarla tutta: si sceglie una direzione a caso, si guarda di quanto la
corrente la allunga, e si ripete con un'altra direzione a ogni nuova
misura. Ogni prova sbaglia un po', ma in media fa giusto, e una prova costa
quanto un passaggio della rete.

Una regola però resta, ed è quella che tiene in piedi tutto: la corrente deve
essere dolce, senza strappi, perché due gocce partite da posti diversi non
finiscano mai nello stesso punto. Se succedesse, al ritorno non si saprebbe più
da quale delle due ripartire, e la macchina non si potrebbe usare al contrario.
Con una corrente dolce le traiettorie non si incrociano, ogni goccia arriva in
un posto suo, e il numero di gocce non cambia: tanti numeri entrano, tanti ne
escono.

`````

`````{tab} Superiore

Per un campo $\mathbf{v}_t$ continuo nel tempo e lipschitziano in
$\mathbf{x}$, l'equazione $\mathrm{d}\mathbf{x}_t / \mathrm{d}t =
\mathbf{v}_t(\mathbf{x}_t)$ ha soluzione unica (teorema di Picard-Lindelöf), e
la mappa $\Phi_T$ che porta $\mathbf{x}_0$ in $\mathbf{x}_T$ è un
diffeomorfismo. L'invertibilità e la conservazione della dimensione restano, e
la regolarità del campo prende il posto dei vincoli sull'architettura: FFJORD
chiede che $\mathbf{v}_t$ e le sue derivate prime siano lipschitziane, il che
in pratica vuol dire attivazioni lisce e non la ReLU. La densità lungo la
traiettoria segue il cambio di variabile istantaneo {cite}`chen2018neural`,

$$
\begin{gathered}
\frac{\mathrm{d}\log p_t(\mathbf{x}_t)}{\mathrm{d}t}
= -\operatorname{tr}\!\left(\frac{\partial \mathbf{v}_t}{\partial
\mathbf{x}}(\mathbf{x}_t)\right)
= -\nabla\!\cdot\!\mathbf{v}_t(\mathbf{x}_t),
\\
\log p_0(\mathbf{x}_0) = \log p_T(\mathbf{x}_T)
+ \int_0^T \nabla\!\cdot\!\mathbf{v}_t(\mathbf{x}_t)\,\mathrm{d}t ,
\end{gathered}
$$

che è la forma in cui la {doc}`sezione su SDE e ODE
</ModelliDiffusione/sde-e-ode>` scrive la verosimiglianza della PF-ODE: nel
limite il logaritmo del determinante della jacobiana di $\Phi_T$ diventa
l'integrale della traccia della jacobiana del campo. La traccia esatta richiede
$D$ prodotti jacobiana-vettore, cioè $\mathcal{O}(D^2)$ per una rete di costo
$\mathcal{O}(D)$, contro $\mathcal{O}(D^3)$ del determinante. FFJORD
{cite}`grathwohl2019ffjord` la sostituisce con lo stimatore di Hutchinson
{cite}`hutchinson1989stochastic`,

$$
\operatorname{tr}(\mathbf{A}) = \mathbb{E}_{\boldsymbol{\epsilon}}
\big[\boldsymbol{\epsilon}^\top \mathbf{A}\,\boldsymbol{\epsilon}\big],
\qquad \mathbb{E}[\boldsymbol{\epsilon}] = \mathbf{0}, \quad
\operatorname{Cov}(\boldsymbol{\epsilon}) = \mathbf{I},
$$

con $\boldsymbol{\epsilon}$ gaussiano o di Rademacher (componenti $\pm 1$
equiprobabili). Il prodotto $\boldsymbol{\epsilon}^\top \mathbf{A}$, con
$\mathbf{A}$ la jacobiana del campo, è un solo prodotto vettore-jacobiana, che
la differenziazione automatica all'indietro calcola al costo di una valutazione
di $\mathbf{v}_t$: il costo scende a $\mathcal{O}(D)$. Lo stesso
$\boldsymbol{\epsilon}$ si tiene per tutta l'integrazione, così la dinamica
resta deterministica dentro il risolutore e la stima della log-densità resta
non distorta; la sua varianza cresce come $\lVert\mathbf{A}\rVert_F^2$. Il
costo totale è $\mathcal{O}(DH\hat{L})$, con $H$ la larghezza dello strato
nascosto e $\hat{L}$ il numero di valutazioni del campo scelto dal risolutore
adattivo, che negli esperimenti di FFJORD cresce durante l'addestramento ma non
con $D$. Ogni verosimiglianza, e ogni campione, richiede di integrare
numericamente l'equazione.

`````

E qui arriva la mossa che ha preso piede, ed è una rinuncia. Il flow matching
{cite}`lipman2023flow` osserva che, se quello che si vuole è generare, la
verosimiglianza durante l'addestramento non serve affatto: basta che la
velocità sia quella giusta. E la velocità giusta si può insegnare per
regressione, mostrando alla rete coppie (punto, velocità) prese da traiettorie
costruite a tavolino, senza mai integrare niente e senza mai calcolare una
traccia. Il flusso rettificato {cite}`liu2023rectified` sceglie come
traiettorie le linee dritte, ed è quello che Stable Diffusion 3 usa, come
racconta il capitolo precedente.

La famiglia di questo capitolo esiste per una proprietà sola, la
verosimiglianza esatta, e il suo discendente più usato per generare immagini la
mette da parte in addestramento, tenendo solo la parte geometrica, il
movimento. La proprietà però resta lì: un modello a flow matching, se qualcuno
vuole pagare il conto dell'equazione differenziale e della divergenza, la
verosimiglianza la sa ancora dare. La rinuncia riguarda l'addestramento e
lascia intatta la struttura, e la parola «flusso» dice ancora da dove viene.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Sapere quanto è probabile un dato è la stessa cosa che saperlo
  comprimere. Il numero di bit che serve per scriverlo con il codice migliore
  è $-\log_2$ della sua probabilità, e non per analogia: è quella grandezza lì,
  a un paio di bit su tutto il file. Per questo qui la qualità si misura in
  bit: su una figurina di CIFAR-10 un buon modello costa meno di 3 bit per
  numero, contro gli 8 di chi non sa niente.
- Quel numero però non dice se le immagini generate sono belle. I due
  giudizi possono andare per conto loro, ed è documentato dal 2016: un modello
  può avere una verosimiglianza ottima e campioni mediocri.
- L'uso più ovvio, «se la probabilità è bassa allora è una cosa che non ho
  mai visto», non funziona come ci si aspetta. Modelli addestrati su fotografie
  di cani e camion danno una probabilità *più alta* a fotografie di numeri
  civici, che non hanno mai visto. Una spiegazione plausibile: quelle immagini
  sono più lisce, e per un modello «facile» vuol dire «probabile». «Quanto è
  probabile» e «l'ho già visto» sono due domande diverse.
- Il seguito della storia: se invece di comporre tanti passi grossi si lascia
  scorrere un movimento continuo, i vincoli sulla forma della macchina cadono,
  ma la corrente deve restare dolce, la macchina resta invertibile e conserva
  il numero di coordinate, e calcolare la probabilità diventa caro. Un metodo
  con cui oggi si disegnano le immagini (il *flow matching*, quello di Stable
  Diffusion 3) rinuncia a calcolarla durante l'addestramento e tiene solo il
  movimento. Ecco da dove viene quella parola, «flusso».
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- $-\log_2 p(\mathbf{x})$ è la lunghezza di codice ottima, purché $p$ sia
  normalizzata (Kraft): verosimiglianza e compressione sono la stessa quantità,
  ed è il motivo per cui la metrica standard della famiglia sono i bit per
  dimensione, $-\log p(\mathbf{x}) / (D \ln 2)$, confrontabili solo a parità
  di dati, scala dei pixel e dequantizzazione. Su CIFAR-10 l'elenco storico va
  da $4{,}48$ (NICE) a $2{,}92$ (PixelCNN++), contro $8{,}00$ del modello che
  non sa niente; per un flusso quel numero è un limite superiore, perché la
  densità è misurata su dati dequantizzati. Un limite variazionale fa anche
  meglio (VDM, $2{,}65$), perché per comprimere basta un limite (*bits-back*):
  l'esattezza serve alla probabilità del singolo dato.
- Verosimiglianza e qualità dei campioni sono criteri largamente indipendenti
  in alta dimensione {cite}`theis2016note`: vanno scelti in funzione
  dell'applicazione, non estrapolati l'uno dall'altro.
- Fallimento OOD {cite}`nalisnick2019do`: flussi, VAE e PixelCNN addestrati
  su CIFAR-10 assegnano log-densità più alta a SVHN, e non viceversa. Una
  lettura discussa è che la densità in alta dimensione sia dominata dalla
  complessità dell'input più che dall'appartenenza al supporto
  {cite}`serra2020input`; in $\mathbb{R}^D$ l'insieme tipico non coincide con
  la regione ad alta densità (il guscio a $\approx\sqrt{D}$), e la densità non
  è invariante per riparametrizzazioni invertibili {cite}`lelan2021perfect`.
  $p_\theta$ alto non implica $\mathbf{x} \in \operatorname{supp}
  p_{\text{dati}}$.
- Limite continuo (*neural ODE* {cite}`chen2018neural`): la composizione di
  passi discreti diventa una ODE sul campo di velocità e
  $\log\lvert\det\mathbf{J}\rvert$ si riduce a
  $\int_0^T \operatorname{tr} (\partial \mathbf{v}_t / \partial \mathbf{x}_t)\,
  \mathrm{d}t$, con $\mathbf{v}_t$ il campo di velocità:
  $\operatorname{tr}(\partial \mathbf{v}_t / \partial \mathbf{x}_t) =
  \nabla\!\cdot\!\mathbf{v}_t$ è la sua divergenza, ed è la forma in cui la
  sezione su SDE e ODE scrive lo stesso integrale. La traccia esatta costa
  $\mathcal{O}(D^2)$; lo stimatore di Hutchinson,
  $\operatorname{tr}(\mathbf{A}) =
  \mathbb{E}[\boldsymbol{\epsilon}^\top\mathbf{A}\boldsymbol{\epsilon}]$, usato
  da FFJORD {cite}`grathwohl2019ffjord`, la porta a $\mathcal{O}(D)$ per una
  rete di costo $\mathcal{O}(D)$ e toglie i vincoli sull'architettura del
  campo. Restano l'invertibilità e la conservazione della dimensione, garantite
  se $\mathbf{v}_t$ è lipschitziano in $\mathbf{x}$; il prezzo è
  l'integrazione numerica.
- Flow matching {cite}`lipman2023flow` rinuncia alla verosimiglianza in
  addestramento e regredisce direttamente il campo di velocità su cammini
  prescritti; il flusso rettificato {cite}`liu2023rectified` sceglie cammini
  rettilinei ed è la scelta di Stable Diffusion 3. La verosimiglianza resta
  ottenibile a posteriori risolvendo l'ODE e integrando la divergenza: la
  rinuncia riguarda l'addestramento, e la struttura resta.
```

`````

Il filo da tenere è il prezzo. La verosimiglianza esatta non è gratis, si paga
in vincoli su come la rete può essere fatta, e ogni modello a verosimiglianza
esatta è un modo diverso di pagarla, dalla generazione un pezzo alla volta ai
determinanti facili dei flussi. Quel prezzo si può anche rifiutare, ed è la
strada dei {doc}`modelli a energia </ModelliEnergia/overview>`: rinunciare a
normalizzare, cioè al conto che trasforma i punteggi in probabilità vere, e
tenersi soltanto il confronto fra un dato e l'altro.
