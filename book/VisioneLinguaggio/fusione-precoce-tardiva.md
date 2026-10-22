# Un vocabolario solo: fusione tardiva e fusione precoce

Chiedi a un sistema come quelli di {doc}`Innestare gli occhi
</VisioneLinguaggio/innestare-gli-occhi>` di *disegnare* un gatto nero che
salta su un muro, e ti risponderà con delle parole: magari ottime, la scena
descritta benissimo. A deciderlo è la testa di uscita, cioè una softmax su un
vocabolario che contiene soltanto parole. Infatti, qualunque cosa il modello
abbia capito guardando, per uscire deve passare da lì.

L'asimmetria è strutturale: il sistema ha un occhio in ingresso e una bocca in
uscita, e nessuna mano.

`````{tab} Elementare

Al momento di scrivere la parola successiva, il modello ha davanti due cose di
natura diversa. Da una parte il testo già scritto e la fotografia, che gli
servono per decidere; dall'altra l'elenco da cui deve pescare, e in
quell'elenco ci sono soltanto parole.

È un quiz a crocette in cui la fotografia
sta nel testo della domanda e le risposte possibili sono tutte parole: per
quanto bene il modello guardi la foto, può soltanto crocettare una parola, e
una foto fra le risposte non c'è.

`````

`````{tab} Superiore

La stessa cosa, in formula, si vede a colpo d'occhio. Un modello costruito
innestando un encoder visivo su un modello di linguaggio
{cite}`liu2023visual` calcola

$$
p_\theta(y_t \mid y_{<t},\, E(\mathbf{I})),
\qquad y_t \in V_{\text{testo}},
$$

dove $\mathbf{I}$ è l'immagine, $E$ l'encoder visivo con il suo connettore,
$y_t$ il token prodotto al passo $t$ e $V_{\text{testo}}$ il vocabolario di
uscita. La barra verticale si legge «dato», o «condizionato a», e separa due
ruoli: a sinistra l'uscita del modello, a destra la condizione che gli è stata
messa davanti per produrla. L'immagine sta a destra: è una condizione, non un
valore che $y_t$ possa assumere.

`````

Da qui la domanda ingenua che è anche quella giusta: e se anche l'immagine
avesse i suoi token? Se nell'elenco ci fossero, accanto alle parole, dei
simboli che stanno per pezzi di immagine, l'ultimo strato potrebbe pescare
anche quelli, e quindi produrre un disegno diventerebbe *la stessa operazione*
con cui si produce una frase. L'idea funziona, ma ha dei costi.

## Un alfabeto anche per i pixel

La **quantizzazione vettoriale** (*vector quantization*) serve ogni volta che
una grandezza continua deve entrare in un modello che lavora su simboli, cioè
su voci di un elenco finito. Si fissa un insieme finito di vettori prototipo,
il **codebook**, e ogni vettore $\mathbf{z}$ prodotto dall'encoder viene
sostituito dal prototipo più vicino; di $\mathbf{z}$ si conserva soltanto
l'indice del prototipo, e quell'indice è il token.

La tecnica viene dalla
compressione dei segnali ed è molto più vecchia del VQ-VAE
{cite}`oord2017neural`, il lavoro che la rende addestrabile: un encoder e un
decoder addestrati insieme, con la quantizzazione in mezzo (le prime due
lettere stanno per *vector quantized*, le altre per *variational autoencoder*,
la famiglia che la {doc}`sezione sul latente che si usa
</ModelliLatenti/il-latente-che-si-usa>` esplorerà in dettaglio).

La quantizzazione non serve ai soli pixel: la {doc}`sezione sui codec neurali
</Audio/codec-neurali>` la riprenderà per il suono, con una variante a più
livelli (la quantizzazione vettoriale residua), perché un solo codebook è
troppo grossolano per un'onda. Un'onda è continua, e un alfabeto per il suono
in natura non esiste: va costruito.

Sui pixel l'operazione si riduce a tre passi: comprimere, arrotondare al
prototipo più vicino, tenere l'indice.

```{figure} ../figures/vq-vae-rappresentazioni-discrete.svg
:name: fig-vq-vae
:alt: "L'immagine entra in un encoder che produce una griglia di vettori continui. Accanto, il codebook appreso elenca un numero finito di vettori-prototipo. Ogni vettore della griglia viene sostituito dal prototipo più vicino, e di ciascuno resta soltanto l'indice: la griglia di vettori diventa così una griglia di numeri interi."
:width: 100%

Da grandezza continua ad alfabeto. L'arrotondamento al prototipo più vicino
perde informazione, ma trasforma un'immagine in una sequenza di numeri interi,
cioè in qualcosa che un modello di linguaggio sa leggere. La freccia
tratteggiata è il modo con cui l'addestramento aggira l'arrotondamento, facendo
finta che non ci sia.
```

Il passaggio che conta in {numref}`fig-vq-vae` è l'ultimo, quando del vettore
resta solo l'indice. Da lì in poi un'immagine e una frase sono oggetti dello
stesso tipo per il modello, cioè sequenze di simboli da un vocabolario finito;
quindi si possono trattare nello stesso Transformer, benché le loro
statistiche restino molto diverse.

`````{tab} Elementare

Un catalogo di 8.192 campioni diversi, ognuno con il suo numero: è tutto quello
che il mosaicista ha in magazzino, e non sono tutti i colori del mondo. Per
riprodurre una fotografia la divide in tessere di 16 pixel per lato, e per
ogni tessera prima la descrive con una scheda di pochi numeri, e poi sceglie
dal catalogo il campione che somiglia di più.

Il catalogo non l'ha comprato:
se l'è costruito guardando migliaia di foto, tenendo i campioni che servivano
di più. Per rifare la foto c'è un secondo artigiano, che dal numero di
catalogo ridisegna la tessera. Alla fine, invece della fotografia, ha una
lista di numeri di catalogo: uno per tessera, letti riga per riga come si legge
una pagina.

Un catalogo fatto in casa ha un rischio: molti campioni non vengono scelti mai,
e restano in magazzino a prendere polvere mentre pochi altri fanno tutto il
lavoro. Si rimedia rimettendo in gioco quelli inutilizzati, oppure facendo a
meno del catalogo e arrotondando ogni numero della scheda alla tacca più vicina
di un righello fisso.

Ad esempio, su una foto di 512 pixel per lato le tessere sono 32 per riga e 32
per colonna, in tutto 1.024, e altrettanti sono i numeri della lista, 1.024 per
un muro bianco come per una folla in piazza, quanto diverse centinaia di parole
in fila.

Ogni numero sta fra 1 e 8.192, e per scriverlo bastano 13 cifre
binarie: la lista intera occupa poco più di 1,6 kilobyte, contro i 786 della
foto grezza, circa 470 volte meno. Il risparmio però conta poco (per quello
esistono già i formati di compressione): quel che conta è che adesso l'immagine
è una lista di simboli presi da un elenco fisso, come una frase è una lista di
parole prese da un dizionario.

Due cose sono andate perse per strada. Il campione scelto è il più vicino che
c'era in magazzino, non la tessera originale, e la differenza è sparita per
sempre. Per fare della griglia una lista abbiamo dovuto decidere un
ordine di lettura, riga per riga, come per il testo: ma una fotografia non ha
un verso di lettura, quell'ordine ce lo siamo inventato noi.

`````

`````{tab} Superiore

Sia $\mathcal{C} = \{\mathbf{e}_1, \dots, \mathbf{e}_K\}$ il codebook appreso e
$\mathbf{z}$ il vettore prodotto dall'encoder per una porzione di immagine. La
quantizzazione ha la forma che il capitolo sull'audio riprenderà per il suono,

$$
k^\star = \arg\min_{k \in \{1, \dots, K\}} \lVert \mathbf{z} - \mathbf{e}_k \rVert^2,
$$

con $k^\star$ token della porzione e $\mathbf{z}_q = \mathbf{e}_{k^\star}$ il
prototipo che lo sostituisce. L’$\arg\min$ non è differenziabile, e lo si
aggira con lo **straight-through estimator** {cite}`bengio2013estimating`: il
gradiente del decoder si copia
tal quale su $\mathbf{z}$, come se l'arrotondamento fosse l'identità. Così
però ai prototipi non arriva niente, e la perdita del VQ-VAE
{cite}`oord2017neural` aggiunge quindi due termini,

$$
\mathcal{L} = -\log p(\mathbf{x} \mid \mathbf{z}_q)
+ \lVert \mathrm{sg}[\mathbf{z}] - \mathbf{e}_{k^\star} \rVert^2
+ \beta\,\lVert \mathbf{z} - \mathrm{sg}[\mathbf{e}_{k^\star}] \rVert^2,
$$

dove $\mathrm{sg}$ blocca il gradiente: il secondo termine porta il prototipo
verso l'uscita dell'encoder, il terzo (con $\beta = 0{,}25$ nel lavoro
originale) impedisce all'encoder di allontanarsi dai prototipi.

Il difetto tipico è il **collasso del codebook**, con molte voci che non
vengono mai scelte. Si contrasta aggiornando i prototipi con una media mobile
esponenziale (una variante proposta nello stesso lavoro), ricollocando quelli
inutilizzati, o eliminando del tutto il codebook appreso a favore di una
quantizzazione scalare su una griglia fissa, componente per componente (FSQ
{cite}`mentzer2024finite`, e LFQ {cite}`yu2024language`, dove ogni componente
diventa $\pm 1$).

Rispetto al suono, poi, cambia la forma del dominio: non una sequenza
monodimensionale di frame, ma un reticolo bidimensionale di patch, che va
linearizzato (di norma in ordine raster) per diventare una sequenza.

**Esempio.** Con i parametri del tokenizzatore di Chameleon
{cite}`chameleon2024mixed`, $K =
8192$ e un'immagine $512 \times 512$ ridotta a $1024$ token: ogni token copre
$512^2/1024 = 256$ pixel, cioè una patch di $16 \times 16$, e vale $\log_2 8192
= 13$ bit.

L'immagine intera occupa $1024 \cdot 13 = 13312$ bit contro i $512 \cdot 512
\cdot 3 \cdot 8 = 6\,291\,456$ bit del formato grezzo, un fattore di
compressione di circa $473$. Il tasso è fisso per costruzione: qualunque sia
il contenuto, l'immagine occupa sempre 1.024 token, né uno di più per una
scena complessa né uno di meno per un muro bianco.

Il vocabolario del modello diventa allora l'unione $V = V_{\text{testo}} \cup
V_{\text{img}}$; in Chameleon è un vocabolario BPE da 65.536 voci che include
le 8.192 dell'immagine. Nel contesto, quindi, un'immagine occupa quanto un
migliaio di token di testo, cioè diverse centinaia di parole, e con più
immagini il contesto e la KV cache si riempiono in fretta; il calcolo completo
è nella sezione {doc}`Il costo del dettaglio
</VisioneLinguaggio/risoluzione-e-dettaglio>`.

`````

## Dove si incontrano i due flussi

Con i token visivi in mano si possono definire la fusione **tardiva** e la
fusione **precoce** nel senso che usa la letteratura sui modelli a token. È
tardiva ogni architettura che tiene un encoder per ciascuna modalità; è
precoce quella in cui le modalità entrano nella stessa sequenza fin
dall'ingresso e le elabora un solo Transformer, con gli stessi pesi.

Delle tre strade presentate in apertura di
capitolo ne bastano quindi due: la mappa di CLIP e l'occhio innestato stanno
dal lato tardivo, anche se nel secondo i token visivi entrano nel modello di
linguaggio fin dal primo strato, e nessuno dei due sa produrre un'immagine. La
differenza non sta in quanta informazione si scambiano immagine e testo, ma in
come ciascuna modalità arriva alla sequenza: con un proprio encoder già
addestrato, oppure con lo stesso vocabolario e gli stessi pesi.

`````{tab} Elementare

In una prima redazione il fotografo e chi scrive lavorano in stanze separate:
il fotografo guarda le sue immagini, passa a chi scrive quanto ha visto, ed
esce di scena.

Che il fotografo passi un foglietto riassuntivo o il fascicolo
intero, e che il fascicolo arrivi sulla scrivania prima della prima riga, qui
non cambia niente: il fotografo lavora con i suoi strumenti, e chi riceve ha un
solo compito, mettere in fila delle parole.

Il prodotto è dunque sempre e soltanto un
testo: da quella redazione non esce mai una fotografia, perché chi tiene la
penna non ha mai avuto in mano una macchina fotografica.

Nella seconda redazione c'è una persona sola, e davanti a sé ha una cassa
tipografica come quelle dei caratteri mobili: nelle caselle ci sono le lettere,
e nelle caselle accanto ci sono i campioni del mosaicista di cui si parlava
prima. Sono nella stessa cassa e si prendono allo stesso modo. Chi compone può
mettere in riga tre parole, poi mille campioni, poi altre due parole, e
rileggendo la riga non distingue fra le due cose: sono tutti pezzi di piombo.

Questa persona può quindi produrre un'immagine per la stessa ragione per cui può
produrre una frase, e cioè perché i pezzi stanno tutti nella stessa cassa.

La contropartita si intuisce già. La prima redazione la metti in piedi
assumendo un fotografo che sa già fotografare e uno scrittore che sa già
scrivere; la seconda, invece, devi formarla, spesso da capo, e chi la compone
dovrà imparare due competenze insieme, con la stessa testa.

`````

`````{tab} Superiore

Le profondità di fusione dell'apertura del capitolo sono tre, ma con il
criterio di Chameleon {cite}`chameleon2024mixed`, encoder separati per modalità
o no, ne bastano due: i connettori, la via *intermedia*, ricadono dal lato
tardivo, perché l'encoder visivo resta una rete a sé, i pesi che elaborano le
due modalità restano quelli di due modelli pre-addestrati a parte, e lo strato
di uscita produce soltanto token di testo.

**Fusione tardiva.** Ciascuna modalità ha il suo encoder, addestrato a parte, e
le rappresentazioni si combinano dopo che ciascuna è stata costruita. Nel caso
estremo, l'addestramento contrastivo della sezione su CLIP, l'unica interazione
fra le modalità è un prodotto scalare fra due vettori. Nel caso dei connettori
l'interazione è più profonda (le patch entrano come prefisso, o via
cross-attention a ogni quarto strato in Flamingo-9B), ma l'encoder visivo resta
una rete a sé, i pesi restano in gran parte quelli di due modelli
pre-addestrati separatamente, e soprattutto la testa di uscita è una softmax su
$V_{\text{testo}}$: ne segue che la capacità generativa è asimmetrica per
costruzione.

**Fusione precoce.** Le due modalità diventano token dello stesso vocabolario
$V$ all'ingresso, e da lì in avanti non esistono più due flussi: c'è una
sequenza sola, $\mathbf{s} = (s_1, \dots, s_n)$ con $s_t \in V$, che un unico
Transformer attraversa dal primo strato all'ultimo con la stessa attenzione e
gli stessi pesi. L'obiettivo è quello della {doc}`sezione sui grandi modelli
linguistici </Transformers/llm>`, con il solo termine di stabilizzazione che si
vedrà fra poco:

$$
\mathcal{L}(\theta) = -\sum_{t=1}^{n} \log p_\theta(s_t \mid s_{<t}),
$$

dove $\theta$ sono i parametri dell'unico modello e la somma corre su tutti i
token della sequenza, visivi e testuali indifferentemente.

È l'impostazione di
Chameleon {cite}`chameleon2024mixed`: sequenze miste, in cui un'immagine è
un blocco di 1.024 token delimitato da due simboli speciali di apertura e
chiusura, e sequenze di addestramento che alternano i due tipi in ordine
arbitrario (testo con immagini in mezzo, immagini con didascalie, pagine web
intere).

La conseguenza è quella che cercavamo: siccome la softmax finale copre tutto
$V$, il modello può emettere un token visivo dove prima poteva emettere solo
una parola. Generare un'immagine è campionare 1.024 volte dalla stessa softmax
da cui si campionano le parole, e passare i risultati al decoder del
tokenizzatore, che è l'unico pezzo in più: nessuna testa aggiuntiva, nessun
secondo Transformer. La simmetria è un effetto della scelta del vocabolario,
più che una funzionalità in più.

`````

Lo stesso schema, un codebook e un Transformer solo, si estende oltre le
immagini ferme. Emu3 {cite}`wang2024emu3`, un modello del 2024, riduce a token,
allo stesso modo, testo, immagini e video. Qui il tokenizzatore
comprime anche nel tempo: quattro fotogrammi consecutivi diventano un solo
gruppo di token, altrimenti un secondo di ripresa occuperebbe tanti token
quanti le ventiquattro o trenta fotografie che lo compongono.

Sopra ci sta un
unico Transformer addestrato da zero, con il solo obiettivo di predire il
simbolo successivo. Lo stesso modello genera e capisce: nella stessa sequenza
di simboli sta la richiesta di produrre un'immagine e la domanda su un'immagine
già data. Che il video entri quasi senza modifiche indica il punto di forza
della fusione precoce: il formato «sequenza di simboli» assorbe tutte le
modalità.

## Perché non l'hanno fatto subito

Letta così, la fusione precoce sembra la scelta ovvia, e viene da chiedersi
perché per tanto tempo si sia battuta l'altra strada. Le ragioni sono due.

L'idea in sé è vecchia quanto CLIP: DALL·E {cite}`ramesh2021zero`, nel 2021,
metteva già 256 token di testo e 1.024 token d'immagine, da un codebook di
8.192 voci, in un unico Transformer, ma solo per generare immagini dal testo.
Estenderla a un modello che legga e scriva entrambe le cose, in ordine
qualsiasi, ha incontrato quelle due ragioni.

La prima è economica: la fusione
tardiva riusa. Un encoder visivo pre-addestrato e un modello di linguaggio
pre-addestrato esistono già, sono costati a qualcun altro, e il connettore che
li unisce si addestra con risorse alla portata di un laboratorio universitario.

La fusione precoce riusa meno, perché il suo vocabolario non è quello di nessun
modello esistente: in Chameleon e in Emu3 il pre-addestramento è stato rifatto
da zero (in Chameleon su migliaia di miliardi di token).

Lavori successivi
partono invece da un modello di linguaggio già addestrato e ne estendono il
vocabolario con i token visivi (Show-o {cite}`xie2024showo`, Liquid
{cite}`wu2024liquid`), con un addestramento molto più breve.

La seconda ragione è che il modello che ne esce è più difficile da addestrare:
non più lento, ma instabile. La funzione di perdita (*loss*), che
l'addestramento deve far scendere, a un certo punto diverge invece di calare
fino alla fine; e lo fa tardi, quando una parte importante del calcolo è già
stata spesa.

`````{tab} Elementare

Due cantanti si dividono un microfono e un amplificatore, con un volume
solo. Per chi ascolta conta chi dei due sta sopra all'altro: se alzano
la voce tutti e due insieme la canzone resta la stessa, solo più forte, e in
sala nessuno protesta.

Il primo canta piano,
il secondo forte. Per farsi sentire, il primo alza un po’ la voce; allora il
secondo, per non essere coperto, alza la sua; e il primo di nuovo. Nessuno dei
due sta facendo niente di sbagliato, ciascuno cerca solo di farsi sentire, ma
il livello sale e sale, e a un certo punto l'amplificatore non ce la fa più:
quel che esce dall'altoparlante si impasta e gracchia, e la canzone non si
riconosce.

Nel modello a fusione precoce i due cantanti sono le due modalità, cioè
l'immagine e il testo, e l'amplificatore condiviso sono i pesi. Testo e immagini
sono fatti in modo molto diverso: indovinare quale sarà la prossima tessera di
mosaico è molto più difficile che indovinare un articolo determinativo, e i
numeri che le due cose fanno girare dentro la rete non hanno la stessa taglia.

Passano però per gli stessi pesi, e ciascuna, per contare qualcosa nel
risultato, tende a farsi un po’ più grossa. I numeri interni crescono,
lentamente, per milioni di passi; e poiché sono memorizzati con una precisione
finita, prima o poi si esce dall'intervallo in cui quei numeri hanno ancora un
senso, e l'addestramento salta.

La cura è quella di un fonico: dei limitatori, in tre punti diversi. Uno dove
il segnale entra nell'amplificatore. Uno a ogni passaggio della catena, prima
che quel che il passaggio aggiunge si sommi a quel che gli è arrivato. L'ultimo
in fondo, sul livello che va all'altoparlante, che tiene fermo il totale senza
toccare l'equilibrio fra le due voci.

`````

`````{tab} Superiore

Il meccanismo documentato nel lavoro su Chameleon {cite}`chameleon2024mixed`
parte da una proprietà innocua della softmax: è invariante per traslazione,
$\mathrm{softmax}(\boldsymbol{\ell}) = \mathrm{softmax}(\boldsymbol{\ell} +
c)$, dove $\boldsymbol{\ell}$ è il vettore dei logit e $c$ uno stesso numero
sommato a tutti; il suo risultato non dice nulla sul loro livello assoluto.

Con
pesi condivisi fra modalità dalle statistiche diverse, ciascuna può allora
«competere» con l'altra alzando un po’ la norma delle proprie attivazioni,
senza che la funzione di perdita se ne accorga. Le norme di query e chiavi
crescono, con esse i logit $\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k}$ che entrano
nella softmax dell'attenzione, e quando quei valori escono dall'intervallo in
cui l'aritmetica a precisione ridotta (bf16) rappresenta ancora qualcosa,
l'addestramento diverge.

La crescita è lenta, e per questo insidiosa:
nel lavoro citato il problema compare sopra gli 8 miliardi di parametri e i
mille miliardi di token, e le divergenze possono arrivare anche dopo il 20-30%
dell'addestramento (il modello da 7 miliardi senza normalizzazione di query e
chiavi diverge dopo circa il 20% di un'epoca). È la ragione per cui non lo si
scopre addestrando modelli piccoli.

Due accorgimenti, entrambi di normalizzazione, lo tengono a bada. Il primo è la
**query-key normalization**: si normalizzano $\mathbf{Q}$ e $\mathbf{K}$
*prima* del prodotto scalare,

$$
\mathrm{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \mathrm{softmax}\!\left(
\frac{\mathrm{LN}(\mathbf{Q})\,\mathrm{LN}(\mathbf{K})^{\top}}{\sqrt{d_k}}\right) \mathbf{V},
$$

dove $\mathrm{LN}$ è la layer normalization, $d_k$ la dimensione delle chiavi e
$\mathbf{V}$ sono qui i valori dell'attenzione (il grassetto li distingue dal
vocabolario $V$ del modello). Così
i logit dell'attenzione smettono di dipendere dalla scala delle attivazioni, e
la loro crescita è limitata alla sorgente.

Il secondo accorgimento riguarda dove stanno le normalizzazioni nel blocco.
Nella disposizione *pre-norm* usuale il flusso residuo riceve l'uscita non
normalizzata del sotto-strato, $\mathbf{x} \leftarrow \mathbf{x} +
F(\mathrm{LN}(\mathbf{x}))$, e nulla impedisce alla norma di $\mathbf{x}$ di
crescere strato dopo strato; spostando la normalizzazione a valle, $\mathbf{x}
\leftarrow \mathbf{x} + \mathrm{LN}(F(\mathbf{x}))$, quel che si somma al
flusso residuo è sempre di scala controllata.

Nel lavoro citato la normalizzazione di query e
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

La lezione generale va oltre la multimodalità: quando più sorgenti eterogenee
condividono gli stessi parametri, la competizione fra di esse si scarica sulle
norme, e la stabilità va difesa nei punti in cui quelle norme entrano in una
softmax, che della loro crescita non si accorge finché non è troppo tardi.

`````

C'è poi un costo di natura diversa, che nessuno degli accorgimenti precedenti
cura: la quantizzazione perde informazione. L'encoder della fusione tardiva
consegna vettori continui; il tokenizzatore della fusione precoce, invece,
sostituisce ciascun vettore con il prototipo più vicino, e la differenza non è
recuperabile in nessun punto successivo della catena.

Con i numeri di prima,
ogni patch di $16 \times 16$ pixel (768 byte di colori) diventa uno fra 8.192
simboli: sopravvive la struttura grossa, non il tratto sottile. Il caso
peggiore è il testo dentro l'immagine, perché un carattere di un documento
fotografato sta nella scala di dettaglio che la quantizzazione cancella; il
lavoro su Chameleon lo indica come la debolezza principale del proprio
tokenizzatore, che limita le prestazioni sui compiti di OCR
{cite}`chameleon2024mixed`.

## Due obiettivi, un modello solo

Il primo motivo per la fusione precoce è rendere l'immagine qualcosa che il
modello possa emettere, non solo leggere. Questo non obbliga a usare simboli
discreti. La diffusione, ad esempio, genera un'immagine in un altro modo, senza
vocabolari: parte da rumore gaussiano e ne toglie un po' a ogni passo, come
descrive il {doc}`capitolo sui modelli di diffusione
</ModelliDiffusione/overview>`.

Che precoce non voglia dire discreto, del resto, si era già visto dal lato della
comprensione: Fuyu-8B {cite}`bavishi2023fuyu`, nel 2023, è un decoder senza
encoder visivo, in cui le patch vengono proiettate linearmente nel primo strato
e lette in ordine raster, a risoluzione arbitraria; è fusione precoce senza
codebook, anche se non genera immagini.

Transfusion {cite}`zhou2024transfusion`
prende sul serio l'ipotesi per la generazione: un solo Transformer, un solo
insieme di parametri, ma due obiettivi diversi a seconda del tipo di token che
sta trattando. Sul testo, la predizione del token successivo di sempre.
Sull'immagine, la diffusione: niente codebook, niente arrotondamento, le patch
restano vettori continui, e quel che il modello impara è a stimare il
rumore da togliere. (Per l'esattezza non sono le patch grezze, ma una loro
versione più compatta preparata a parte, come in quasi tutti i generatori di
immagini.)

`````{tab} Elementare

Una persona che sa scrivere e dipingere non usa la stessa tecnica per le due
cose, e non c'è ragione di obbligarla. Il testo lo scrive da sinistra a destra,
una parola dopo l'altra, e mentre sceglie una parola può rileggere solo le
parole già scritte: le successive non esistono ancora. Il quadro no: il quadro
lo abbozza tutto insieme e poi ci ripassa sopra, e mentre ritocca l'angolo in
alto a sinistra guarda anche l'angolo in basso a destra, perché sono già lì
tutti e due e devono stare insieme.

Il modello si porta dentro questa stessa differenza. Quando elabora le parole,
ogni posizione guarda solo all'indietro. Quando elabora i pezzi di un'immagine,
dentro quel blocco tutti guardano tutti, avanti e indietro, perché non c'è un
verso di lettura da rispettare. Cade così l'obiezione sull'ordine arbitrario
in cui il mosaico veniva letto, riga per riga.

Quando nel foglio arriva il punto in cui ci vuole un disegno, chi scrive posa
la penna, apre un riquadro, lo abbozza e lo rifinisce guardando quel che ha
scritto fin lì, poi lo chiude e riprende a scrivere dalla riga dopo. Le due
tecniche non stanno in due teste separate: a scrivere e a dipingere si allena
la stessa mano, e a essere diversi sono soltanto gli strumenti, la penna e il
pennello.

Il vantaggio pratico si vede subito: siccome l'immagine non viene mai
arrotondata a un catalogo di tessere, il dettaglio fine non viene buttato via
in partenza. In linea di principio, poi, un disegno non viene più trattato come
se fosse una frase.

`````

`````{tab} Superiore

La costruzione poggia su due meccanismi.

**L'attenzione mista.** La maschera è causale fra i token di testo, come in
qualunque decoder, ma bidirezionale all'interno di ciascun blocco immagine:
le patch della stessa immagine si vedono tutte a vicenda, mentre continuano a
vedere solo il passato per quel che riguarda il testo che le precede. La
motivazione è che l'ordine raster è una finzione: gli elementi di un'immagine
sono co-presenti, non successivi, e imporre loro una causalità è un vincolo
gratuito.

**I due obiettivi.** La perdita è la somma di due termini che altrove nel libro
compaiono separati, uno per il testo e uno per l'immagine,

$$
\mathcal{L} = \mathcal{L}_{\text{LM}} + \lambda\, \mathcal{L}_{\text{DDPM}},
$$

dove $\mathcal{L}_{\text{LM}}$ è la cross-entropia sul token successivo
calcolata sulle sole posizioni testuali, $\mathcal{L}_{\text{DDPM}}$ è l'errore
quadratico sul rumore stimato calcolato sulle sole posizioni visive (la
$\mathbb{E}\lVert \epsilon - \epsilon_\theta(\mathbf{x}_t, t) \rVert^2$ che il
capitolo sulla diffusione deriverà per esteso: $\epsilon$ è il rumore
effettivamente aggiunto, $\epsilon_\theta$ il rumore che il modello stima di
dover togliere, e questo $t$ è il livello di rumore, non la posizione nella
sequenza) e $\lambda$ pesa il secondo rispetto al primo (nel lavoro originale
$\lambda = 5$).

Si noti che le due perdite non riguardano parametri diversi: attraversano
gli stessi strati, e ogni peso del Transformer riceve gradienti da entrambe.
Restano specifici della modalità soltanto gli innesti agli estremi (l'embedding
dei token di testo da una parte, i pochi strati che impacchettano e
spacchettano le patch latenti dall'altra), che è il minimo indispensabile per
far entrare due tipi di dato nella stessa sequenza.

In generazione il modello lavora in due modalità e passa dall'una all'altra
senza interventi esterni: campiona parole finché non emette il simbolo speciale
che apre un'immagine, a quel punto accoda un blocco di patch di puro rumore,
esegue su quel blocco il ciclo di denoising del capitolo sulla diffusione (con
tutto il testo precedente che fa da condizionamento, tramite l'attenzione),
chiude il blocco e torna a scrivere parole.

`````

La maschera di attenzione, cioè la regola su chi può guardare chi, è l'unico
pezzo nuovo e occupa poche righe. Costruiamola per una sequenza di nove
posizioni: tre token di testo, un blocco immagine di quattro patch, altri due
token di testo. Nell'uscita `X` vuol dire che l'attenzione è permessa e `.` che
è vietata.

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
ognuna guarda solo all'indietro, cioè la maschera causale. Le quattro righe
centrali sono il blocco immagine: ciascuna vede tutte e quattro
le patch, comprese quelle che vengono dopo, e vede tutto il testo che precede.
Nessuna riga di testo, invece, guarda avanti.

In PyTorch questa matrice si
passa a `nn.MultiheadAttention` come `attn_mask`, con l'avvertenza che lì la
convenzione è rovesciata (per una maschera booleana, `True` significa
*vietato*): si passa `torch.from_numpy(~m)`. Con
`F.scaled_dot_product_attention`, invece, `True` significa *consentito*, e si
passa `m` così com'è. Si noti che le due funzioni della stessa libreria usano
convenzioni opposte, e l'errore non dà nessun avviso.

## Quando serve un vocabolario comune

Il connettore, il vocabolario comune e il modello con due obiettivi non si
superano a vicenda, e la scelta si fa sul meccanismo.

Se il compito è capire (descrivere una foto, rispondere a domande su un
grafico, leggere un documento), la fusione tardiva è la scelta ragionevole e lo
resta, perché riusa due modelli già addestrati, evita l'arrotondamento al
codebook e chiede solo un connettore. Pagare per la mano
quando basta la bocca è cattiva ingegneria.

Se il compito è produrre immagini e testo dentro lo stesso sistema, o
peggio alternarli (una risposta che contiene un diagramma, un disegno corretto
alla luce di quanto detto due paragrafi prima), allora la strada del connettore
chiede due modelli, uno che capisce e uno che disegna, e un passaggio di
consegne fra i due; e quel passaggio, che è fatto di parole, è il punto in cui
l'informazione si perde. Un vocabolario comune, o un modello con due obiettivi,
tiene tutto in un contesto solo.

Il campo non ha deciso, e i sistemi reali sono spesso ibridi (un encoder
continuo per capire, un decoder generativo per produrre, dentro lo stesso
prodotto). L'argomento a favore della fusione precoce non è che generi meglio,
perché su questo confrontare i sistemi ha poco senso e invecchia in fretta: è
che la condivisione dei parametri *dovrebbe* produrre un trasferimento fra le
due direzioni.

Imparare a disegnare un gatto dovrebbe aiutare a riconoscerlo,
perché per generarlo bisogna sapere com'è fatto, mentre per descriverlo spesso
basta indovinare quello che di solito si scrive sotto una foto del genere, cioè
parlare della fotografia senza averla guardata: il problema che
l’{doc}`apertura del capitolo
</VisioneLinguaggio/overview>` ha chiamato allucinazione visiva.

La prova che deciderebbe la questione ha allora questa forma: a parità di
parametri, di dati e di calcolo, la capacità di *capire* di un modello a
fusione precoce migliora quando gli si insegna anche a *generare*? Se sì, il
vocabolario comune è un modo di imparare meglio, e non una comodità
architetturale, e vale il costo di estendere e riaddestrare il modello. Se le
due capacità si limitano a convivere senza aiutarsi, la fusione precoce resta
una scelta di ingegneria, giustificata quando serve un solo sistema al posto di
due, e il connettore continuerà a vincere per una ragione semplice: è più
economico.

Le prove disponibili nel 2024 vanno in direzioni diverse, e nessuna
è quel confronto a parità di tutto. Liquid {cite}`wu2024liquid` riporta che in
uno spazio di token unificato generazione e comprensione si rinforzano a
vicenda, e che il calo dovuto all'addestramento congiunto diminuisce al
crescere del modello; Janus {cite}`wu2024janus` trova invece un conflitto fra i
due compiti quando condividono lo stesso encoder visivo, e separa i due
percorsi di codifica mantenendo un solo Transformer.

Che l'immagine entri come simboli o come vettori continui, comunque, il numero
dei pezzi resta un vincolo: ogni patch occupa un posto nella sequenza, e la
sezione {doc}`Il costo del dettaglio
</VisioneLinguaggio/risoluzione-e-dettaglio>` mostra che il problema si sposta,
ma non sparisce nemmeno con gli encoder continui.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un modello con l'occhio innestato ha un occhio in ingresso e una bocca in
  uscita, e nessuna mano: l'immagine può entrare, ma quel che esce viene da un
  elenco fatto di sole parole. Non disegna perché non ha simboli per dirlo.
- Il mosaicista con il catalogo dà all'immagine i suoi simboli: si divide la
  foto in tessere, per ognuna si sceglie dal catalogo di 8.192 campioni quello
  che le somiglia di più, e della foto resta una lista di numeri di catalogo.
  Da quel momento disegnare e scrivere sono la stessa operazione.
- Due redazioni. Nella prima il fotografo passa un foglietto a chi scrive ed
  esce di scena: da lì esce sempre e solo un testo. Nella seconda c'è una cassa
  tipografica dove le lettere e i campioni stanno nelle caselle accanto, e chi
  compone le prende allo stesso modo: quella redazione può produrre anche
  un'immagine.
- La seconda strada chiede un doppio sforzo: bisogna rifare buona parte della
  formazione, spesso da capo, e i due cantanti con un solo amplificatore alzano
  la voce a turno finché quel che esce si impasta e gracchia. Si cura con dei
  limitatori in tre punti della catena, cioè togliendo a tutti la possibilità
  di urlare senza toccare l'equilibrio fra le voci.
- Arrotondare butta via: il campione scelto a catalogo non è mai identico alla
  tessera vera, e la differenza non torna più. Il primo a sparire è il
  dettaglio sottile, cioè il testo scritto dentro una fotografia.
- C'è una via di mezzo: una macchina sola che scrive da sinistra a destra e
  dipinge tutto insieme, con due tecniche diverse per le due cose e nessun
  arrotondamento. La maschera di attenzione (le `X` e i punti del codice)
  mostra la regola: le parole guardano solo indietro, le tessere della stessa
  immagine si guardano tutte fra loro, perché una fotografia non ha un verso di
  lettura.
- La domanda aperta non è quale disegni meglio, ma se imparare a disegnare
  aiuti a capire. Se sì, la lingua unica vale il suo costo; se no, vince
  l'innesto, che è più economico. Le prime prove, del 2024, non sono d'accordo
  fra loro.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Un modello che innesta un encoder visivo su un modello di linguaggio
  {cite}`liu2023visual` è asimmetrico: l'immagine è una condizione, e la
  softmax finale copre solo il vocabolario del testo. Non può generare
  immagini perché non ha simboli per dirle.
- La quantizzazione vettoriale del VQ-VAE {cite}`oord2017neural`, la stessa
  che i codec neurali useranno per il suono, dà all'immagine i suoi simboli: un
  codebook di prototipi, l'indice del più vicino come token. Con un codebook da
  8.192 voci, un'immagine $512 \times 512$ diventa 1.024 token da 13 bit.
- Tardiva è la fusione di due encoder addestrati a parte, le cui
  rappresentazioni si combinano dopo che ciascuna è stata costruita (i
  connettori compresi); precoce è un vocabolario unico all'ingresso, con un
  solo Transformer e un solo obiettivo autoregressivo
  {cite}`chameleon2024mixed`, esteso al video da {cite}`wang2024emu3`. Solo la
  seconda genera, perché la simmetria è nel vocabolario.
- Per la fusione precoce gli oneri sono due: il pre-addestramento va rifatto da
  zero, o esteso a lungo partendo da un modello di linguaggio
  {cite}`xie2024showo`, e con pesi condivisi fra modalità dalle statistiche
  diverse le norme crescono, finché i logit dell'attenzione escono
  dall'intervallo in cui l'aritmetica a precisione ridotta ha ancora senso. Si
  difende normalizzando query e chiavi prima del prodotto scalare, spostando le
  normalizzazioni a valle dei sotto-strati e frenando con un termine di perdita
  la deriva dei logit finali.
- La quantizzazione perde informazione: ogni patch di $16 \times 16$ pixel
  diventa uno fra 8.192 simboli, e il dettaglio fine (il testo dentro una foto)
  è il primo a sparire. Il collasso del codebook si contrasta con medie mobili,
  ricollocazione dei prototipi o quantizzazione scalare su griglia fissa.
- Precoce non vuol dire discreto (Fuyu-8B {cite}`bavishi2023fuyu` proietta le
  patch nel primo strato). Transfusion {cite}`zhou2024transfusion` tiene un
  modello solo con due obiettivi (autoregressivo sul testo, diffusione
  sull'immagine) e un'attenzione mista: causale fra le parole, bidirezionale
  dentro il blocco immagine, perché un'immagine non ha un ordine di lettura.
- La domanda aperta non è quale generi meglio, ma se condividere i parametri
  produca trasferimento fra capire e generare. Se sì, il vocabolario comune
  vale il suo costo; se no, l'innesto vince perché è più economico. Le prove del
  2024 vanno in direzioni opposte {cite}`wu2024liquid`, {cite}`wu2024janus`.
```

`````
