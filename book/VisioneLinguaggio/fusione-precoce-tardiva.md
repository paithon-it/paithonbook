# Un vocabolario solo: fusione tardiva e fusione precoce

Chiedi a un sistema come quello della sezione precedente di *disegnare* un
gatto nero che salta su un muro, e ti risponderà con delle parole. Magari
ottime parole: quella scena descritta benissimo. A deciderlo è la forma
dell'ultimo strato, e non una pigrizia o un rifiuto.
Qualunque cosa quel modello abbia capito guardando, per uscire deve passare da
un unico collo di bottiglia: scegliere una voce da un elenco chiuso, il
**vocabolario**. E in quell'elenco ci sono soltanto parole.

L'asimmetria è strutturale, e detta in una riga suona così: il sistema ha un
occhio in ingresso e una bocca in uscita, e nessuna mano. L'immagine entra, e
serve a scegliere le parole; ma fra le cose che il modello sa produrre ci sono
solo parole, perché solo di parole è fatto l'elenco da cui pesca.

`````{tab} Elementare

Al momento di scrivere la parola successiva, il modello ha davanti due cose di
natura diversa. Da una parte quello che ha già scritto e la fotografia, che gli
servono per decidere; dall'altra l'elenco da cui deve pescare, e in
quell'elenco ci sono soltanto parole. La fotografia sta dalla parte di chi
decide, non da quella delle cose che si possono pescare: è una **condizione**,
non una voce dell'elenco.

`````

`````{tab} Superiore

La stessa cosa, in formula, si vede a colpo d'occhio. Un modello costruito
innestando un encoder visivo su un modello di linguaggio, come quelli della
sezione precedente {cite}`liu2023visual`, calcola

$$
p_\theta(y_t \mid y_{<t},\, E(\mathbf{I})),
\qquad y_t \in V_{\text{testo}},
$$

dove $\mathbf{I}$ è l'immagine, $E$ l'encoder visivo con il suo connettore, $y_t$ il
token prodotto al passo $t$ e $V_{\text{testo}}$ il vocabolario di uscita. La
barra verticale si legge «dato che», e separa due mestieri: a sinistra quel che
il modello produce, a destra quel che gli è stato messo davanti per produrlo.
L'immagine sta a destra: è una **condizione**, non un valore che $y_t$ possa
assumere.

`````

Da qui la domanda ingenua che è anche quella giusta: e se anche l'immagine
avesse i suoi token? Se nell'elenco ci fossero, accanto alle parole, dei simboli
che stanno per pezzi di immagine, l'ultimo strato potrebbe pescare anche quelli,
e produrre un disegno diventerebbe *lo stesso gesto* che produrre una frase. Il
resto della sezione è il prezzo di questa idea.

## Un alfabeto anche per i pixel

L'attrezzo per costruire quei simboli serve ogni volta che una grandezza
**continua**, cioè che può assumere qualunque valore, tutte le cifre dopo la
virgola comprese, deve entrare in un modello che mangia simboli, cioè voci di un
elenco finito.

Come funziona è presto detto. Si prepara un catalogo finito di file di numeri
campione (il *codebook*), e ogni pezzetto di segnale, che l'encoder ha già
ridotto a una fila di numeri, viene sostituito dal campione del catalogo che gli
somiglia di più. Di quel pezzetto non si conserva la fila: si conserva il suo
**numero di catalogo**, e quel numero è il token. Il gesto si chiama
**quantizzazione vettoriale** (in inglese *vector quantization*), ed è quello
del VQ-VAE {cite}`oord2017neural`, le cui prime due lettere stanno proprio per
questo.

Non è un attrezzo dei soli pixel: il {doc}`capitolo sull'audio
</Audio/overview>` lo rimonterà tal quale per il suono, nella sezione sui
**codec neurali**, perché là il problema avrà la stessa forma. Un'onda è
continua, e un alfabeto per il suono in natura non esiste: va costruito.

Sui pixel il gesto è tutto qui: comprimere, arrotondare al prototipo più
vicino, tenere l'indice.

```{figure} ../figures/vq-vae-rappresentazioni-discrete.svg
:name: fig-vq-vae
:alt: "L'immagine entra in un encoder che produce una griglia di vettori continui. Accanto, il codebook appreso elenca un numero finito di vettori-prototipo. Ogni vettore della griglia viene sostituito dal prototipo più vicino, e di ciascuno resta soltanto l'indice: la griglia di vettori diventa così una griglia di numeri interi."
:width: 100%

Da grandezza continua ad alfabeto. L'arrotondamento al prototipo più vicino è
tutto il trucco: perde qualcosa, ma trasforma un'immagine in una sequenza di
numeri interi, cioè in qualcosa che un modello di linguaggio sa leggere. La
freccia tratteggiata è il modo con cui l'addestramento aggira l'arrotondamento,
facendo finta che non ci sia.
```

Il passaggio che conta in {numref}`fig-vq-vae` è l'ultimo, quando del vettore
resta solo l'indice. Da lì in poi un'immagine e una frase sono la stessa cosa
per il modello, sequenze di simboli da un vocabolario finito, ed è questo che
rende pensabile trattarle nello stesso Transformer.

`````{tab} Elementare

Un catalogo di 8.192 tessere diverse, ognuna con il suo numero: è tutto quello
che il mosaicista ha in magazzino, e non sono tutti i colori del mondo. Per
riprodurre una fotografia la divide in quadratini di 16 pixel per lato, e per
ogni quadratino sceglie dal catalogo la tessera che gli somiglia di più. Alla
fine, invece della fotografia, ha una lista di numeri di catalogo: uno per
quadratino, letti riga per riga come si legge una pagina.

Facciamo il conto su una foto di 512 pixel per lato. I quadratini sono 32 per
riga e 32 per colonna, in tutto 1.024, e altrettanti sono i numeri della lista.
Per scrivere un numero fra 1 e 8.192 bastano 13 cifre di quelle che usa un
calcolatore, che sono soltanto 0 e 1: con tredici di quelle cifre si contano
$2^{13}$, cioè 8.192 cose diverse. Tredici cifre per 1.024 quadratini fanno
13.312 cifre in tutto; e siccome otto di quelle cifre fanno un byte, sono poco
più di 1,6 kilobyte. La fotografia grezza, invece, ha 512 per 512 puntini e
ciascuno porta tre numeri (rosso, verde e blu) da un byte l'uno:
$512 \times 512 \times 3 = 786.432$ byte, 786 kilobyte. Circa 470 volte meno. E
la lista è lunga uguale per qualunque foto: 1.024 numeri per un muro bianco
come per una folla in piazza, quanto diverse centinaia di parole in fila. Il
risparmio però conta poco (per quello esistono già i formati di compressione):
quel che conta è che adesso l'immagine è una **lista di simboli presi da un
elenco fisso**, esattamente come una frase è una lista di parole prese da un
dizionario.

Due cose sono andate perse per strada. La tessera scelta è quasi sempre la più
vicina che c'era in magazzino e non il quadratino originale, e la
differenza è sparita per sempre. E per fare della griglia una lista abbiamo
dovuto decidere un ordine di lettura, riga per riga, come per il testo: ma una
fotografia non ha un verso di lettura, quell'ordine ce lo siamo inventato noi.

`````

`````{tab} Superiore

Sia $\mathcal{C} = \{\mathbf{e}_1, \dots, \mathbf{e}_K\}$ il codebook appreso e $\mathbf{z}$ la fila di numeri
prodotta dall'encoder per una porzione di immagine. La quantizzazione ha la
forma che il capitolo sull'audio riprenderà per il suono,

$$
k^\star = \arg\min_{k \in \{1, \dots, K\}} \lVert \mathbf{z} - \mathbf{e}_k \rVert^2,
$$

con $k^\star$ token della porzione. L’$\arg\min$ non è differenziabile, e il
gradiente non attraverserebbe la quantizzazione: lo si aggira con lo
*straight-through estimator*, cioè copiando il gradiente del decoder tal quale
sull'uscita dell'encoder, come se l'arrotondamento fosse l'identità. Quel che
cambia rispetto al suono è la forma del dominio: non una
sequenza monodimensionale di frame, ma un reticolo bidimensionale di patch, che
va **linearizzato** (di norma in ordine raster) per diventare una sequenza.

Con i parametri del tokenizzatore di Chameleon {cite}`chameleon2024mixed`,
$K = 8192$ e un'immagine $512 \times 512$ ridotta a $1024$ token: ogni token
copre $512^2/1024 = 256$ pixel, cioè una patch di $16 \times 16$, e vale
$\log_2 8192 = 13$ bit. L'immagine intera occupa $1024 \cdot 13 = 13312$ bit
contro i $512 \cdot 512 \cdot 3 \cdot 8 = 6\,291\,456$ bit del formato grezzo,
un fattore di compressione di circa $472$. Il tasso è irrigidito per
costruzione: qualunque sia il contenuto, l'immagine costa sempre 1.024 token,
né uno di più per una scena complessa né uno di meno per un muro bianco.

Il vocabolario del modello diventa allora l'unione
$V = V_{\text{testo}} \cup V_{\text{img}}$; in Chameleon un vocabolario BPE da
65.536 voci che include le 8.192 dell'immagine. Da notare l'ordine di grandezza
del contesto: un'immagine occupa quanto un migliaio di token di testo, cioè
diverse centinaia di parole, e con un'attenzione quadratica nella lunghezza
della sequenza questo è il vincolo che domina tutto il resto (è il tema della
prossima sezione, sulla risoluzione).

`````

## Dove si incontrano i due flussi

Con i token visivi in mano possiamo dare alle due parole del titolo un
significato preciso. La differenza fra fusione **tardiva** e fusione
**precoce** non sta in quanta informazione si scambiano immagine e testo, ma in
quanto presto cominciano a scambiarsela, e se a maneggiarle sia un pezzo solo
di rete o due pezzi diversi, cresciuti separati.

`````{tab} Elementare

In una prima redazione il fotografo e chi scrive lavorano in stanze separate:
il fotografo guarda le sue immagini, passa a chi scrive quello che ha visto,
ed esce di scena. Che gli passi un foglietto riassunto o il fascicolo
intero della sezione precedente, qui non cambia niente: quel che conta è che chi
riceve fa un mestiere solo, mettere in fila delle parole. Il prodotto è sempre e
soltanto un testo: da quella redazione non esce mai una fotografia, perché chi
tiene la penna non ha mai avuto in mano una macchina fotografica.

Nella seconda redazione c'è una persona sola, e davanti a sé ha una cassa
tipografica come quelle dei caratteri mobili: nelle caselle ci sono le lettere,
e nelle caselle accanto ci sono le tessere del mosaico di cui parlavamo prima.
Sono nella stessa cassa, si prendono con lo stesso gesto. Chi compone può
mettere in riga tre parole, poi mille tessere, poi altre due parole, e
rileggendo la riga non distingue fra le due cose: sono tutti pezzi di piombo.
Questa persona può produrre un'immagine per la stessa ragione per cui può
produrre una frase, e cioè perché i pezzi stanno tutti nella stessa cassa.

Il prezzo si intuisce già: la prima redazione la metti in piedi assumendo un
fotografo che sa già fotografare e uno scrittore che sa già scrivere; la
seconda devi formarla da capo, e chi la compone dovrà imparare due mestieri
insieme, con la stessa testa.

`````

`````{tab} Superiore

Le profondità di fusione sono tre, ma qui ne bastano due: i connettori, la via
*intermedia*, ricadono dal lato tardivo, perché i pesi che elaborano le due
modalità restano quelli di due modelli pre-addestrati per conto proprio, e lo
strato di uscita produce soltanto token di testo.

**Fusione tardiva.** Due encoder addestrati separatamente producono
rappresentazioni che si incontrano vicino all'uscita. Nel caso estremo,
l'addestramento contrastivo della sezione su CLIP, l'unica interazione fra le
modalità è un prodotto scalare fra due vettori. Nel caso dei connettori della
sezione precedente, l'interazione è più profonda (le patch entrano come
prefisso o via cross-attention) ma i pesi restano in gran parte quelli di due
modelli pre-addestrati per conto proprio, e soprattutto la testa di uscita è
una softmax su $V_{\text{testo}}$: la capacità generativa è asimmetrica per
costruzione.

**Fusione precoce.** Le due modalità diventano token dello stesso vocabolario
$V$ **all'ingresso**, e da lì in avanti non esistono più due flussi: c'è una
sequenza sola, $\mathbf{s} = (s_1, \dots, s_n)$ con $s_t \in V$, che un unico
Transformer attraversa dal primo strato all'ultimo con la stessa attenzione e
gli stessi pesi. L'obiettivo è quello della {doc}`sezione sui grandi modelli
linguistici </Transformers/llm>`, senza aggiunte:

$$
\mathcal{L}(\theta) = -\sum_{t=1}^{n} \log p_\theta(s_t \mid s_{<t}),
$$

dove $\theta$ sono i parametri dell'unico modello e la somma corre su tutti i
token della sequenza, visivi e testuali indifferentemente. È l'impostazione di
Chameleon {cite}`chameleon2024mixed`: sequenze **miste**, in cui un'immagine è
un blocco di 1.024 token delimitato da due simboli speciali di apertura e
chiusura, e sequenze di addestramento che alternano i due tipi in ordine
arbitrario (testo con immagini in mezzo, immagini con didascalie, pagine web
intere).

La conseguenza è quella che cercavamo: siccome la softmax finale copre tutto
$V$, il modello può emettere un token visivo dove prima poteva emettere solo
una parola. Generare un'immagine è campionare 1.024 volte dalla stessa softmax
da cui si campionano le parole, e passare i risultati al decoder del
tokenizzatore. Nessuna testa aggiuntiva, nessun secondo modello: la
simmetria è un effetto della scelta del vocabolario, più che una funzionalità
in più.

`````

La stessa ricetta si estende oltre le immagini ferme. Emu3 {cite}`wang2024emu3`
riduce a numeri di catalogo, allo stesso modo, testo, immagini **e video**. Qui
il catalogo comprime anche nel tempo: quattro fotogrammi consecutivi diventano
un solo gruppo di token, altrimenti un secondo di ripresa costerebbe quanto le
ventiquattro o trenta fotografie che lo compongono. Sopra ci sta un unico
Transformer addestrato da zero, con l'unico obiettivo di predire il simbolo
successivo. Lo stesso
modello genera e capisce: nella stessa sequenza di simboli sta la richiesta di
produrre un'immagine e la domanda su un'immagine già data. Che il video entri
quasi senza modifiche è un'indicazione: quel che rende la fusione precoce
interessante è che il formato «sequenza di simboli» le assorbe tutte.

## Perché non l'hanno fatto subito

A leggerla così, la fusione precoce sembra la cosa ovvia da fare, e viene da
chiedersi perché la strada battuta sia stata a lungo l'altra. Le ragioni sono
due, ed entrambe dicono qualcosa sui modelli in generale.

La prima è economica ed è la più banale: la fusione tardiva riusa. Un encoder
visivo pre-addestrato e un modello di linguaggio pre-addestrato esistono già,
sono costati a qualcun altro, e il connettore che li unisce si addestra con
risorse alla portata di un laboratorio universitario. La fusione precoce non
riusa niente, perché il suo vocabolario non è quello di nessun modello
esistente: il pre-addestramento va rifatto da zero, su migliaia di miliardi di
token.

La seconda è più interessante, ed è che il modello che ne esce è **più difficile
da addestrare**. Non «più lento»: instabile. Qui «costo» smette di voler dire
soldi e torna a essere il punteggio dell'errore, quello che l'addestramento deve
far scendere. La sua curva, che dovrebbe calare piano piano fino alla fine, a un
certo punto schizza verso l'alto e non torna più; e lo fa tardi, quando una
fetta importante del calcolo è già stata spesa.

`````{tab} Elementare

Due cantanti si dividono un microfono e un amplificatore, con una manopola del
volume sola. Quel che arriva in fondo alla sala è chi dei due sta sopra
all'altro: alzarli tutti e due insieme non cambia niente. Il primo canta piano,
il secondo forte. Per farsi sentire, il primo alza un po’ la voce; allora il
secondo, per non essere coperto, alza la sua; e il primo di nuovo. Nessuno dei
due sta facendo niente di sbagliato, ciascuno cerca solo di farsi sentire, ma
il livello sale e sale, e a un certo punto l'amplificatore non ce la fa più:
quel che esce dall'altoparlante si impasta e gracchia, e la canzone non si
riconosce.

Nel modello a fusione precoce i due cantanti sono le due **modalità**, cioè
l'immagine e il testo, e l'amplificatore condiviso sono i pesi. Testo e immagini
sono fatti in modo molto diverso: indovinare quale sarà la prossima tessera di
mosaico è molto più difficile che indovinare un articolo determinativo, e i
numeri che le due cose fanno girare dentro la rete non hanno la stessa taglia.
Passano però per gli stessi pesi, e ciascuna, per contare qualcosa nel
risultato, tende a farsi un po’ più grossa. I numeri interni crescono, lentamente, per milioni
di passi; e poiché sono memorizzati con una precisione finita, prima o poi si
esce dall'intervallo in cui quei numeri hanno ancora un senso, e
l'addestramento salta.

La cura è quella di un fonico: dei limitatori, in tre punti diversi. Uno dove
il segnale entra nell'amplificatore. Uno a ogni passaggio della catena, prima
che quel che il passaggio aggiunge si sommi a quel che gli è arrivato. E uno in
fondo, sul livello che va all'altoparlante, che tiene fermo il totale senza
toccare l'equilibrio fra le due voci. Nessuno perde il diritto di parola, si
toglie solo la possibilità di urlare.

`````

`````{tab} Superiore

Il meccanismo documentato nel lavoro su Chameleon {cite}`chameleon2024mixed`
parte da una proprietà innocua della softmax: è invariante per traslazione,
$\mathrm{softmax}(\boldsymbol{\ell}) = \mathrm{softmax}(\boldsymbol{\ell} + c)$,
dove $\boldsymbol{\ell}$ è il vettore dei logit e $c$ uno stesso numero sommato
a tutti; il suo risultato non dice nulla sul loro livello assoluto. Con pesi
condivisi fra modalità dalle statistiche diverse, ciascuna può allora «competere» con l'altra alzando
un po’ la norma delle proprie attivazioni, senza che la funzione di perdita se
ne accorga. Le norme di query e chiavi crescono, con esse i logit
$\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k}$ che entrano nella softmax dell'attenzione, e quando quei
valori escono dall'intervallo in cui l'aritmetica a precisione ridotta (bf16)
rappresenta ancora qualcosa, l'addestramento diverge. La crescita è lenta, ed è
questo a renderla insidiosa: nel lavoro citato il problema compare sopra gli 8
miliardi di parametri e i mille miliardi di token, e le divergenze arrivano
quando è già stato percorso un buon 20–30% dell'addestramento. È la ragione per
cui non lo si scopre addestrando modelli piccoli.

Due accorgimenti, entrambi di normalizzazione, lo tengono a bada. Il primo è la
**query-key normalization**: si normalizzano $\mathbf{Q}$ e $\mathbf{K}$ *prima* del prodotto
scalare,

$$
\mathrm{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \mathrm{softmax}\!\left(
\frac{\mathrm{LN}(\mathbf{Q})\,\mathrm{LN}(\mathbf{K})^{\top}}{\sqrt{d_k}}\right) \mathbf{V},
$$

dove $\mathrm{LN}$ è la layer normalization, $d_k$ la dimensione delle chiavi e
$\mathbf{V}$ sono qui i valori dell'attenzione (il grassetto li distingue dal
vocabolario $V$ del modello):
i logit dell'attenzione smettono di dipendere dalla scala delle attivazioni, e
la loro crescita è limitata alla sorgente. Il secondo riguarda **dove** stanno
le normalizzazioni nel blocco. Nella disposizione *pre-norm* usuale il flusso
residuo riceve l'uscita non normalizzata del sotto-strato,
$\mathbf{x} \leftarrow \mathbf{x} + F(\mathrm{LN}(\mathbf{x}))$, e nulla impedisce alla norma di $\mathbf{x}$ di
crescere strato dopo strato; spostando la normalizzazione a valle,
$\mathbf{x} \leftarrow \mathbf{x} + \mathrm{LN}(F(\mathbf{x}))$, quel che si somma al flusso residuo è
sempre di scala controllata. Nel lavoro citato la normalizzazione di query e
chiavi serve a entrambe le taglie di modello: a quello da 7 miliardi di
parametri basta affiancarle il dropout, mentre quello da 34 miliardi chiede la
riorganizzazione delle norme, che con il dropout non va d'accordo e finisce per
sostituirlo.

Resta l'ultima softmax, quella sul vocabolario, dove la stessa invarianza per
traslazione lascia scivolare i logit tutti insieme verso l'alto (in un modello
di solo testo il fenomeno ha già un nome, *logit drift*). Lì si aggiunge alla
perdita un termine $10^{-5}\log^2 Z$, dove $Z$ è la funzione di partizione
della softmax, cioè la somma degli esponenziali dei logit: non tocca le
probabilità, che dal livello assoluto non dipendono, ma toglie al modello la
libertà di farlo crescere.

La lezione generale va oltre la multimodalità: quando
più sorgenti eterogenee condividono gli stessi parametri, la competizione fra
di esse si scarica sulle **norme**, e la stabilità va difesa esattamente dove
quelle norme entrano in una softmax, che della loro crescita non si accorge
finché non è troppo tardi.

`````

C'è poi un costo di natura diversa, e non si cura con nessuno degli accorgimenti
di prima: **arrotondare butta via**. L'encoder della fusione tardiva descrive
quello che ha visto con file di numeri con la virgola, tutte le cifre che
servono; il mosaicista, cioè il tokenizzatore, arrotonda ciascuna alla voce di
catalogo più vicina, e la differenza non è recuperabile da nessuna parte a
valle. Con i numeri di prima, ogni quadratino di $16 \times 16$ pixel (768 byte
di colori) diventa uno fra 8.192 simboli: quel che sopravvive è la
struttura grossa, non il tratto sottile. Il caso peggiore è il testo dentro
l'immagine, perché un carattere di un documento fotografato sta proprio nella
scala di dettaglio che l'arrotondamento cancella, ed è una limitazione che i
lavori su questi modelli riconoscono apertamente. La sezione successiva, sulla
risoluzione, mostrerà che il problema si sposta ma non sparisce nemmeno con gli
encoder continui.

## La terza via: due obiettivi, un modello solo

A questo punto la domanda diventa più precisa. La fusione precoce ci serviva
per un motivo solo: rendere l'immagine qualcosa che il modello possa
**emettere**, non solo leggere. Ma quel motivo richiede davvero che l'immagine
sia fatta di simboli discreti? Di modi per generare un'immagine ne esiste un
altro, che con i vocabolari non ha niente a che fare, e il {doc}`capitolo sui modelli
di diffusione </ModelliDiffusione/overview>` lo costruirà per intero: si parte da un quadrato di puro rumore e
se ne toglie un velo alla volta, finché sotto i veli compare la figura.

Transfusion {cite}`zhou2024transfusion` prende sul serio l'ipotesi: un solo
Transformer, un solo insieme di parametri, ma **due obiettivi diversi** a
seconda del tipo di token che sta trattando. Sul testo, la predizione del token
successivo di sempre. Sull'immagine, la diffusione: niente catalogo, niente
arrotondamento, le tessere restano file di numeri, e quel che il modello impara
è a indovinare il disturbo da togliere. (Per l'esattezza non sono le tessere
grezze, ma una loro versione più compatta preparata a parte, come in quasi
tutti i generatori di immagini.)

`````{tab} Elementare

Una persona che sa scrivere e dipingere non usa la stessa tecnica per le due
cose, e non c'è ragione di obbligarla. Il testo lo scrive da sinistra a destra,
una parola dopo l'altra, e mentre sceglie una parola può rileggere solo quel
che ha già scritto: quello che verrà dopo non esiste ancora. Il quadro no: il
quadro lo abbozza tutto insieme e poi ci ripassa sopra, e mentre ritocca
l'angolo in alto a sinistra guarda anche l'angolo in basso a destra, perché
sono già lì tutti e due e devono stare insieme.

È esattamente questa la differenza che il modello si porta dentro. Quando
elabora le parole, ogni posizione guarda solo all'indietro. Quando elabora i
pezzi di un'immagine, dentro quel blocco tutti guardano tutti, avanti e
indietro, perché non c'è un verso di lettura da rispettare: è la stessa
obiezione che avevamo lasciato in sospeso quando abbiamo dovuto decidere,
arbitrariamente, di leggere il mosaico riga per riga.

Quando nel foglio arriva il punto in cui ci vuole un disegno, chi scrive posa
la penna, apre un riquadro, lo abbozza e lo rifinisce guardando quel che ha
scritto fin lì, poi lo chiude e riprende a scrivere dalla riga dopo. Le due
tecniche non stanno in due teste separate: a scrivere e a dipingere si allena
la stessa mano, e a essere diversi sono soltanto gli attrezzi, la penna e il
pennello.

Il vantaggio pratico si vede subito: siccome l'immagine non viene mai
arrotondata a un catalogo di tessere, il dettaglio fine non viene buttato via
in partenza. Il vantaggio di principio è che si smette di fingere che un
disegno sia una frase.

`````

`````{tab} Superiore

Due meccanismi reggono la costruzione.

**L'attenzione mista.** La maschera è causale fra i token di testo, come in
qualunque decoder, ma **bidirezionale all'interno di ciascun blocco immagine**:
le patch della stessa immagine si vedono tutte a vicenda, mentre continuano a
vedere solo il passato per quel che riguarda il testo che le precede. La
motivazione è che l'ordine raster è una finzione: gli elementi di un'immagine
sono co-presenti, non successivi, e imporre loro una causalità è un vincolo
gratuito.

**I due obiettivi.** La perdita è la somma di due termini che nel resto del
libro vivono separati, uno per il testo e uno per l'immagine,

$$
\mathcal{L} = \mathcal{L}_{\text{LM}} + \lambda\, \mathcal{L}_{\text{DDPM}},
$$

dove $\mathcal{L}_{\text{LM}}$ è la cross-entropia sul token successivo
calcolata sulle sole posizioni testuali, $\mathcal{L}_{\text{DDPM}}$ è
l'errore quadratico sul rumore stimato calcolato sulle sole posizioni visive
(la $\mathbb{E}\lVert \epsilon - \epsilon_\theta(\mathbf{x}_t, t) \rVert^2$ che
il capitolo sulla diffusione deriverà per esteso: $\epsilon$ è il rumore davvero
aggiunto, $\epsilon_\theta$ quello che il modello stima di dover togliere, e
questo $t$ è il livello di rumore, non la posizione nella sequenza)
e $\lambda$ pesa il secondo rispetto al primo (nel lavoro
originale $\lambda = 5$). Le due perdite non riguardano parametri diversi:
attraversano gli stessi strati, e ogni peso del Transformer riceve gradienti da
entrambe. Restano specifici della modalità soltanto gli innesti agli estremi
(l'embedding dei token di testo da una parte, i pochi strati che impacchettano
e spacchettano le patch latenti dall'altra), che è il minimo indispensabile per
far entrare due tipi di dato nella stessa sequenza.

In generazione il modello lavora in due modalità e passa dall'una all'altra da
sé: campiona parole finché non emette il simbolo speciale che apre
un'immagine, a quel punto accoda un blocco di patch di puro rumore, esegue su
quel blocco il ciclo di denoising del capitolo sulla diffusione (con tutto il
testo precedente che fa da condizionamento, tramite l'attenzione), chiude il
blocco e torna a scrivere parole.

`````

La regola su chi può guardare chi (in gergo, la **maschera** di attenzione) è
l'unico pezzo davvero nuovo, e sta in poche righe.
Costruiamola per una sequenza di nove posizioni: tre token di testo, un blocco
immagine di quattro patch, altri due token di testo.

```python
import numpy as np

def maschera_mista(segmenti):
    """Maschera di attenzione per una sequenza mista di testo e immagini.

    segmenti: lista di coppie (tipo, lunghezza), con tipo "testo" o "immagine".
    Restituisce una matrice booleana: consentito[i, j] e' True se la posizione
    i puo' guardare la posizione j."""
    tipi, blocchi = [], []
    for indice, (tipo, lunghezza) in enumerate(segmenti):
        tipi.extend([tipo] * lunghezza)        # tipo di ogni posizione
        blocchi.extend([indice] * lunghezza)   # a quale segmento appartiene
    n = len(tipi)
    blocchi = np.array(blocchi)
    e_immagine = np.array([t == "immagine" for t in tipi])

    consentito = np.tril(np.ones((n, n), dtype=bool))     # causale: j <= i
    stesso_blocco = blocchi[:, None] == blocchi[None, :]
    dentro_immagine = stesso_blocco & e_immagine[:, None] & e_immagine[None, :]
    return consentito | dentro_immagine    # nel blocco immagine, anche avanti


m = maschera_mista([("testo", 3), ("immagine", 4), ("testo", 2)])
for riga in m:
    print("".join("X" if v else "." for v in riga))
```

L'uscita disegna la struttura meglio di qualunque descrizione (`X` dove
l'attenzione è permessa, `.` dove è vietata):

```text
X........
XX.......
XXX......
XXXXXXX..
XXXXXXX..
XXXXXXX..
XXXXXXX..
XXXXXXXX.
XXXXXXXXX
```

Le prime tre righe e le ultime due sono la regola di sempre per le parole:
ognuna guarda solo all'indietro, ed è la maschera **causale**. Le quattro righe
centrali sono il blocco immagine: ciascuna vede tutte e quattro
le patch, comprese quelle che vengono dopo, e vede tutto il testo che precede.
Nessuna riga di testo, invece, guarda avanti. In PyTorch questa matrice si
passa a `nn.MultiheadAttention` come `attn_mask`, con l'avvertenza che lì la
convenzione è rovesciata (per una maschera booleana, `True` significa
*vietato*): si passa `torch.from_numpy(~m)`.

## Quando serve un vocabolario comune

Le tre strade non si superano a vicenda, e la scelta si fa sul meccanismo.

Se il compito è **capire** (descrivere una foto, rispondere a domande su un
grafico, leggere un documento), la fusione tardiva è la scelta ragionevole e lo
resta: riusa due modelli già addestrati, non paga il dazio dell'arrotondamento a
catalogo, e chiede solo un connettore. Pagare per la mano quando basta la bocca
è cattiva ingegneria.

Se il compito è **produrre immagini e testo dentro lo stesso sistema**, o
peggio alternarli (una risposta che contiene un diagramma, un disegno corretto
alla luce di quanto detto due paragrafi prima), allora la strada del connettore
chiede due modelli, uno che capisce e uno che disegna, e un passaggio di
consegne fra i due; e quel passaggio, che è fatto di parole, è il punto in cui
l'informazione si perde. Un vocabolario comune, o un modello con due obiettivi,
tiene tutto in un contesto solo.

Onestà d'obbligo, per chiudere: il campo non ha deciso, e i sistemi reali sono
spesso ibridi (un encoder continuo per capire, un decoder generativo per
produrre, dentro lo stesso prodotto). L'argomento a favore della fusione
precoce non è che generi meglio, perché su questo confrontare i sistemi ha
poco senso e invecchia in fretta: è che la condivisione dei parametri
*dovrebbe* produrre un **trasferimento** fra le due direzioni. Imparare a
disegnare un gatto dovrebbe aiutare a riconoscerlo, perché per generarlo
bisogna sapere com'è fatto, mentre per descriverlo spesso basta indovinare
quello che di solito si scrive sotto una foto del genere, cioè parlare della
fotografia senza averla guardata: il problema che l'apertura del capitolo ha
chiamato allucinazione visiva.

La prova che deciderebbe la questione ha allora questa forma: a parità di
parametri, di dati e di calcolo, la capacità di *capire* di un modello a
fusione precoce migliora quando gli si insegna anche a *generare*? Se sì, il
vocabolario comune è un modo di imparare meglio, e non una comodità
architetturale, e vale il costo del pre-addestramento da zero. Se le due
capacità si
limitano a convivere senza aiutarsi, la fusione precoce resta una scelta di
ingegneria, giustificata quando serve un solo sistema al posto di due, e il
connettore continuerà a vincere per la ragione più semplice del mondo: costa
meno.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un modello con l'occhio innestato ha **un occhio in ingresso e una bocca in
  uscita, e nessuna mano**: l'immagine può entrare, ma quel che esce viene da un
  elenco fatto di sole parole. Non disegna perché non ha simboli per dirlo.
- Il **mosaicista con il catalogo** dà all'immagine i suoi simboli: si divide la
  foto in quadratini, per ognuno si sceglie dal catalogo di 8.192 tessere quella
  che gli somiglia di più, e della foto resta una lista di numeri di catalogo.
  Da quel momento disegnare e scrivere sono lo stesso mestiere.
- **Due redazioni.** Nella prima il fotografo passa un foglietto a chi scrive ed
  esce di scena: da lì esce sempre e solo un testo. Nella seconda c'è una cassa
  tipografica dove le lettere e le tessere stanno nelle caselle accanto, e chi
  compone le prende con lo stesso gesto: quella redazione può produrre anche
  un'immagine.
- Il conto della seconda strada è doppio: bisogna rifare tutta la formazione da
  capo, e i **due cantanti con un solo amplificatore** alzano la voce a turno
  finché quel che esce si impasta e gracchia. Si cura con dei limitatori in tre
  punti della catena, cioè togliendo a tutti la possibilità di urlare senza
  toccare l'equilibrio fra le voci.
- **Arrotondare butta via**: la tessera scelta a catalogo non è mai identica al
  quadratino vero, e la differenza non torna più. Il primo a sparire è il
  dettaglio sottile, cioè il testo scritto dentro una fotografia.
- C'è una via di mezzo: una macchina sola che **scrive da sinistra a destra e
  dipinge tutto insieme**, con due tecniche diverse per le due cose e nessun
  arrotondamento. Il quadrato di `X` e di punti mostra la regola: le parole
  guardano solo indietro, le tessere della stessa immagine si guardano tutte fra
  loro, perché una fotografia non ha un verso di lettura.
- La domanda aperta non è quale disegni meglio, ma se imparare a disegnare aiuti
  a **capire**. Se sì, la lingua unica vale il suo costo; se no, vince l'innesto,
  che costa meno.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Un modello che innesta un encoder visivo su un modello di linguaggio
  {cite}`liu2023visual` è **asimmetrico**: l'immagine è una condizione, e la
  softmax finale copre solo il vocabolario del testo. Non può generare
  immagini perché non ha simboli per dirle.
- La **quantizzazione vettoriale** del VQ-VAE {cite}`oord2017neural`, la stessa
  che i codec neurali useranno per il suono, dà all'immagine i suoi simboli: un
  codebook di prototipi, l'indice del più vicino come token. Con un codebook da
  8.192 voci, un'immagine $512 \times 512$ diventa 1.024 token da 13 bit.
- **Tardiva** è la fusione di due encoder addestrati a parte che si incontrano
  vicino all'uscita; **precoce** è un vocabolario unico all'ingresso, con un
  solo Transformer e un solo obiettivo autoregressivo
  {cite}`chameleon2024mixed`, esteso al video da {cite}`wang2024emu3`. Solo la
  seconda genera, perché la simmetria è nel vocabolario.
- Il conto della fusione precoce è doppio: il pre-addestramento va rifatto da
  zero, e con pesi condivisi fra modalità dalle statistiche diverse le **norme
  crescono**, finché i logit dell'attenzione escono dall'intervallo in cui
  l'aritmetica a precisione ridotta ha ancora senso. Si difende normalizzando
  query e chiavi prima del prodotto scalare, spostando le normalizzazioni a
  valle dei sotto-strati e frenando con un termine di perdita la deriva dei
  logit finali.
- **Quantizzare butta via**: ogni patch di $16 \times 16$ pixel diventa uno fra
  8.192 simboli, e il dettaglio fine (il testo dentro una foto) è il primo a
  sparire.
- **Transfusion** {cite}`zhou2024transfusion` tiene un modello solo con due
  obiettivi (autoregressivo sul testo, diffusione sull'immagine) e
  un'attenzione **mista**: causale fra le parole, bidirezionale dentro il
  blocco immagine, perché un'immagine non ha un ordine di lettura.
- La domanda aperta non è quale generi meglio, ma se condividere i parametri
  produca **trasferimento** fra capire e generare. Se sì, il vocabolario comune
  vale il suo costo; se no, l'innesto vince perché costa meno.
```

`````
