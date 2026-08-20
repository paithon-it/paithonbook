# Il costo del dettaglio

Prendi un giornale e allontanalo dagli occhi. A un metro riconosci ancora che è
un giornale: distingui le fotografie dal testo, forse leggi il titolo di
apertura. A due metri il titolo se ne va e restano quattro rettangoli grigi.
Quello che hai perso non è l'immagine, che è ancora tutta lì: è il
**dettaglio**, e con lui tutto ciò che nella pagina era *scritto* invece che
disegnato.

Le sezioni precedenti hanno dato per scontata la risoluzione: un encoder taglia
l'immagine in patch, un connettore la consegna a un modello di linguaggio, un
tokenizzatore la riduce a simboli. In tutti e tre i casi c'era un numero
nascosto sotto il tappeto, quanti pixel entrano nell'encoder, ed è il vincolo
economico che governa i sistemi reali molto più delle differenze
architetturali. Ogni pixel in più si paga in **contesto**, cioè nei posti che
l'immagine occupa nella sequenza e sottrae a tutto il resto; e non lo si paga in
proporzione.

## Il conto, in due righe

Il meccanismo è quello del Vision Transformer {cite}`dosovitskiy2021image`: una
tessera quadrata di lato fisso diventa un pezzo della sequenza (in gergo, una
*patch* e un *token*). Da qui il numero dei pezzi è aritmetica elementare, e la
sua conseguenza sul costo dell'attenzione non lo è.

`````{tab} Elementare

Prendiamo un encoder che taglia tessere da $14 \times 14$ puntini. Su
un'immagine da $224 \times 224$ ne stanno $224 : 14 = 16$ per riga e altrettante
per colonna, quindi $16 \times 16 = 256$ tessere: la nostra immagine è una
«frase» di 256 pezzi.

Adesso raddoppiamo il lato, da $224$ a $448$: le tessere diventano $32$ per riga
e $32$ per colonna, in tutto $32 \times 32 = 1024$. Sono **quadruplicate**, e la
ragione è che a raddoppiare sono due lati insieme, la larghezza e l'altezza:
l'immagine ha quattro volte l'area.

Il seguito è meno ovvio. Il modello, per capire ogni tessera, la confronta con
tutte le altre (è l’**attenzione**, il meccanismo del capitolo sui Transformer):
con 256 tessere i confronti sono $256 \times 256$, con 1024 sono
$1024 \times 1024$. Quattro volte i pezzi significa $4 \times 4 = 16$ volte i
confronti: raddoppiare il lato di una fotografia moltiplica per sedici il lavoro
dell'attenzione, e raddoppiarlo ancora (da $448$ a $896$) lo moltiplica per
altri sedici, duecentocinquantasei volte il conto di partenza.

È il motivo per cui non esiste la risposta «e allora aumentiamo la risoluzione».
La si aumenta, ma sapendo cosa si compra e a che prezzo.

`````

`````{tab} Superiore

Con immagine $H \times W$ e patch di lato $p$, il numero di token è

$$
N = \left\lfloor \frac{H}{p} \right\rfloor \cdot
    \left\lfloor \frac{W}{p} \right\rfloor,
$$

dove $H$ e $W$ sono altezza e larghezza in pixel e $p$ il lato della patch.
Con $H = W = 224$ e $p = 14$ si ha $N = 256$; con $H = W = 448$, $N = 1024$.
$N$ è lineare nell’**area**, quindi quadratico nel lato.

Il costo dell'attenzione è a sua volta quadratico in $N$, cioè $O(N^2 d)$ con
$d$ dimensione del modello: quartico nel lato dell'immagine. Da $224$ a $448$
si paga $(1024/256)^2 = 16$ volte tanto; da $224$ a $896$, $(4096/256)^2 = 256$.

Una precisazione, per non vendere il termine quadratico più caro di quanto sia.
Nei FLOP di un blocco Transformer l'attenzione vale circa $4N^2 d$ e le
proiezioni più il feed-forward (con strato nascosto a $4d$ unità) circa
$24 N d^2$: il quadratico supera il lineare solo per $N > 6d$, cioè oltre
$24\,576$ token in un modello con $d = 4096$ (nell'encoder visivo, dove $d$ vale
circa un migliaio, la soglia scende attorno ai seimila). Quello che si paga
subito è il **contesto occupato**, il tempo di *prefill* e la cache di chiavi e
valori; e la FlashAttention del capitolo sulle GPU
{cite}`dao2022flashattention` toglie dal conto la memoria $O(N^2)$, non il
calcolo: alza il tetto, non cambia l'esponente.

`````

## Perché duecentoventiquattro pixel non bastano

Se il conto è così severo, quanta risoluzione serve davvero? Non c'è una
risposta valida in generale, ed è l'osservazione che riorganizza tutta la
sezione: **la risoluzione la detta il compito, non l'architettura**.

Un gatto lo si riconosce da lontano, perché la sagoma, le orecchie e la coda
sono strutture larghe che sopravvivono a una riduzione brutale, ed è il tipo di
compito su cui sono stati costruiti i primi encoder visivi. La riga di una
fattura, no. Il numero di un grafico, la voce di una tabella, la scritta su un
cartello, il pulsante di una schermata vivono nella scala di dettaglio più fine,
quella che sparisce per prima.

`````{tab} Elementare

Facciamo il conto su un foglio A4, alto 297 millimetri. Se lo riduciamo a 224
puntini di altezza, ogni millimetro di carta diventa **tre quarti di puntino**
($224 : 297 = 0{,}75$). Una maiuscola stampata in un libro come questo è alta
circa due millimetri, quindi ne occupa uno e mezzo, e un puntino e mezzo non
contiene una lettera: contiene una macchia grigia.

Guardiamola dall'altro verso. Se un millimetro vale tre quarti di puntino, un
puntino vale un millimetro e un terzo, e la tessera del mosaico, che è larga 14
puntini, copre **quasi due centimetri di pagina**. Il modello riceve, in un
unico pezzetto di informazione, un quadratino di foglio alto quattro o cinque
righe di testo. Chiedergli cosa c'è scritto è come chiedere di leggere un libro
attraverso un vetro smerigliato.

Se invece la pagina la diamo alta 1024 puntini, ogni millimetro vale tre puntini
e mezzo: la maiuscola ne occupa sette e una tessera copre quattro millimetri di
carta, quanto è alta una riga di testo. Adesso qualcosa da leggere c'è. Il compito ha deciso la risoluzione, e nessuna
astuzia di architettura può cambiare il fatto che dove non ci sono puntini non
c'è informazione.

`````

`````{tab} Superiore

Conviene ragionare in **pixel per millimetro** e confrontarli con la scala del
segnale da leggere. Un A4 alto $297$ mm, ridotto a un lato lungo di $L$ pixel,
dà $L/297$ px/mm; una maiuscola di un corpo da 9-10 punti è alta fra $2$ e
$2{,}5$ mm, e nella tabella prendiamo l'estremo basso, $2$ mm. La scelta non
decide l'esito: anche la maiuscola più alta, $2{,}5$ mm, a $224$ pixel resta
sotto i due pixel ($1{,}9$), quindi la conclusione regge pure nel caso più
favorevole alla lettura.

| lato lungo | px/mm | maiuscola | una patch da 14 px |
|---|---|---|---|
| $224$ | $0{,}75$ | $1{,}5$ px | $18{,}6$ mm |
| $1024$ | $3{,}45$ | $6{,}9$ px | $4{,}1$ mm |
| $1792$ | $6{,}03$ | $12{,}1$ px | $2{,}3$ mm |
| $3508$ (300 dpi) | $11{,}81$ | $23{,}6$ px | $1{,}2$ mm |

La colonna che decide è l'ultima: dice quanta pagina deve stare dentro *un
solo* token. A $224$ pixel un token porta quasi due centimetri di foglio, cioè
un frammento di paragrafo; nessuna proiezione, per quanto ben addestrata, può
far uscire da un solo vettore il contenuto di sei parole scritte, e il limite è
informativo prima che statistico. A $1792$ pixel un token copre poco più di due
millimetri per lato, un paio di caratteri: la stessa architettura, con lo stesso
encoder, di colpo legge.

I compiti si dispongono quindi su una scala di **frequenza spaziale** richiesta:
riconoscere una scena sta in basso, leggere testo dentro l'immagine o agire su
una schermata stanno in alto, e nessun aumento di parametri li risolve se
l'informazione è già stata buttata nel ridimensionamento. Non a caso i sistemi
che puntano ai documenti alzano la risoluzione nativa dell'encoder in una fase
di addestramento apposita: Qwen-VL {cite}`bai2023qwenvl`, il cui encoder è un
ViT con patch di lato $14$, la porta da $224$ a $448$ nella fase di
pre-addestramento multi-compito, e la ragione dichiarata è esattamente questa,
ridurre l'informazione persa nel sotto-campionamento.

`````

Il resto della sezione è la storia di tre risposte a questo vincolo. Nessuna
lo cancella: tutte e tre lo **spostano** in un punto del sistema dove fa meno
male, e conviene tenere d'occhio dove finisce il conto ogni volta.

## Prima risposta: tagliare l'immagine a riquadri

La più semplice e la più diffusa. L'encoder sa lavorare a una risoluzione sola,
quella su cui è stato addestrato, e l'immagine è più grande. Invece di
rimpicciolire l'immagine fino a farla stare nell'encoder, la si taglia in
**riquadri** grandi esattamente quanto lui si aspetta. Ogni riquadro passa per
conto suo, e i pezzi che ne escono si mettono tutti in fila; in coda si aggiunge
una **miniatura** dell'immagine intera, che è l'unico posto in cui si vede come
i riquadri stanno insieme. Il metodo si chiama **tiling**, che in italiano vuol
dire «tagliare a piastrelle», e si trova anche sotto il nome *any-resolution*.

`````{tab} Elementare

Devi fotografare un quadro grande con una macchina che inquadra solo un
quadratino. Fai così: scatti sei foto ravvicinate, una per ogni pezzo del
quadro, poi fai un passo indietro e ne scatti una settima che prende tutto, con
molto meno dettaglio ma completa. Chi riceve le sette foto ha il dettaglio (le sei
ravvicinate) e sa anche come stanno insieme (la settima). Il pregio è che non
hai comprato una macchina nuova, ed è per questo che il taglio a riquadri ha
vinto: si aggiunge sopra un encoder già addestrato senza toccarlo.

Il difetto lo indovini pensando a una figura a cavallo fra due pezzi. Nelle sei
foto ravvicinate non c'è mai per intero: mezza faccia in una e mezza nell'altra,
e chi guarda deve rimetterle insieme senza essere sicuro che appartengano alla
stessa cosa. L'unico posto dove si vede tutta è la settima foto, quella senza dettaglio.
Il taglio è arbitrario, non segue i confini degli oggetti, e questa
arbitrarietà è il prezzo del metodo.

`````

`````{tab} Superiore

Nella variante documentata da InternVL {cite}`chen2024far` i riquadri sono da
$448 \times 448$, la griglia si sceglie fra le combinazioni ammesse (da uno a
dodici riquadri in addestramento, fino a quaranta in uso) cercando quella che
meno distorce le proporzioni dell'immagine, e la miniatura, ridotta anch'essa a
$448 \times 448$, accompagna sempre i riquadri.

Sia allora $t$ il lato del riquadro (la risoluzione nativa dell'encoder) e
$g_h \times g_w$ la griglia che minimizza la distorsione delle proporzioni
originali. L'immagine viene ridimensionata a $(g_h t) \times (g_w t)$, divisa in
$g_h g_w$ riquadri e affiancata dall'immagine intera ridotta a $t \times t$:
i token totali sono $(g_h g_w + 1) \cdot N_t$ con $N_t = (t/p)^2$.

Il guadagno computazionale è il primo argomento che viene in mente, ed è il meno
solido dei tre: conviene misurarlo bene. L'attenzione dell'encoder è quadratica
**dentro ogni riquadro** e assente fra riquadri diversi, quindi il costo passa da
$O\big((g_h g_w N_t)^2\big)$ a $O(g_h g_w N_t^2)$, cioè da quadratico a
**lineare nell'area**: asintoticamente è un guadagno vero, ed è la ragione per
cui il metodo scala. Con $896 \times 896$, $t = 448$ e
$p = 14$: monolitica sono $4096$ token e $4096^2 \approx 16{,}8$ milioni di
coppie; a riquadri sono cinque pezzi (quattro più la miniatura) da $1024$ token,
cioè $5 \cdot 1024^2 \approx 5{,}2$ milioni di coppie, $3{,}2$ volte meno.

Quel $3{,}2$, però, conta le **coppie di attenzione**, non i FLOP dell'encoder, e
a questi valori di $N$ le due cose non coincidono affatto. La soglia $N > 6d$
calcolata poco sopra dice che con $4096$ token e $d$ dell'ordine del migliaio
siamo ancora *sotto*: l'attenzione è meno della metà del blocco, e il tiling
taglia la parte piccola del conto mentre manda nel feed-forward $5120$ token
invece di $4096$, cioè paga di più sul termine che domina. Rifacendo il conto per
intero con la stessa contabilità di prima ($24Nd^2 + 4N^2d$ per strato), il
lavoro totale cala di $1{,}24$ volte a $d = 768$, di $1{,}14$ a $d = 1024$ e di
$1{,}06$ a $d = 1408$: un risparmio fra il 5% e il 20%, non tre volte, e tanto
minore quanto più l'encoder è largo. (Attenzione a non leggere il rapporto come
una percentuale: dividere per $1{,}24$ vuol dire risparmiare il 19%, non il
24%.) In cambio i token *totali* salgono da $4096$ a $5120$, perché la
miniatura è ridondante per costruzione: **il conto si è spostato, non è
sparito**, e tutti quei token finiscono nella stessa sequenza del modello di
linguaggio, dove l'attenzione è di nuovo quadratica su tutto.

Gli argomenti che reggono di più, quindi, sono gli altri due, e non sono
computazionali. Gli **embedding di posizione**
dell'encoder restano validi, quindi non vanno interpolati su una griglia più
grande, operazione che degrada e in genere chiede un riaddestramento; e il
sistema resta indifferente alle proporzioni, perché una schermata panoramica e
una pagina verticale ricevono griglie diverse invece di finire schiacciate
entrambe in un quadrato.

I limiti sono altrettanto netti. Un oggetto o una riga di testo che attraversano
il taglio finiscono in due passaggi indipendenti dell'encoder, che non si vedono
fra loro, e ricucirli tocca all'attenzione a valle, che ha la sola miniatura come
riferimento globale. E il numero di token cresce con l'area: una pagina in
griglia $3 \times 4$ con la miniatura sono tredici passaggi di encoder e, senza
altre contromisure, $13 \cdot 1024 = 13\,312$ token.

`````

## Seconda risposta: comprimere i token

Tagliando a riquadri, però, i pezzi si moltiplicano. Una pagina di documento, con
la griglia più fitta che questi sistemi usano in addestramento, può volerne
dodici, e con la miniatura fanno tredici passaggi dell'encoder: siccome ogni
riquadro da $448$ puntini di lato dà $1024$ tessere, in fila ne finiscono
$13 \times 1024 = 13\,312$. Tante non sono sostenibili, e la seconda
risposta attacca quel numero riducendo le tessere *dopo* l'encoder e *prima* del
modello di linguaggio: l'immagine viene guardata ad alta risoluzione, ma quello
che entra nel contesto è più corto. Resta da decidere **come** si comprime, e
c'è un modo che butta via subito e uno che rimanda il conto.

`````{tab} Elementare

Hai quattro barattoli di tempera, uno per colore, e devi liberare tre ripiani.

Primo modo: versi i quattro colori in un barattolo solo e mescoli. Occupi un
ripiano invece di quattro, e quel che ne esce è davvero la media dei quattro; ma
se qualcuno chiede «di che colore era il terzo barattolo?», dal marrone che hai
in mano non lo ricavi più. Questo è il **pooling**, prendere quattro tessere
vicine e sostituirle con la loro media. Semplice, efficace, irreversibile.

Secondo modo: prendi una cassetta con quattro scomparti e ci infili dentro i
quattro barattoli, ciascuno nel suo. Sempre un ripiano occupato invece di
quattro, e non hai perso un grammo di colore: la cassetta è solo quattro volte
più pesante. Questo è il **pixel shuffle** (alla lettera «rimescolamento dei
puntini»: il nome è più oscuro della cosa): quattro tessere adiacenti diventano
un pezzo solo, che porta con sé tutti e quattro i contenuti, uno di fianco
all'altro. L'informazione si è spostata dai *posti* al *contenuto di ogni posto*.

Una precisazione onesta, però. La cassetta, prima di entrare nel modello di
linguaggio, deve passare per una fessura larga sempre uguale: i posti in fila
sono tutti della stessa taglia, e in quella taglia adesso devono starci quattro
tessere invece di una. Il rimescolamento in sé non perde niente; la fessura sì.

`````

`````{tab} Superiore

Sia $\mathbf{Z} \in \mathbb{R}^{N \times d_v}$ l'uscita dell'encoder, riorganizzata sulla
griglia $\sqrt{N} \times \sqrt{N} \times d_v$ da cui proviene. Il **pixel
shuffle** con fattore $r = 2$ è la mappa

$$
\mathbb{R}^{\sqrt{N} \times \sqrt{N} \times d_v} \longrightarrow
\mathbb{R}^{\frac{\sqrt{N}}{2} \times \frac{\sqrt{N}}{2} \times 4 d_v},
$$

che raggruppa ogni blocco $2 \times 2$ di posizioni adiacenti e ne concatena i
quattro vettori lungo la dimensione dei canali. Il numero di token scende a
$N/4$ e la dimensione di ciascuno sale a $4 d_v$: è una **permutazione degli
elementi del tensore**, quindi biiettiva, e il conteggio dell'informazione non
cambia. (Il nome viene dalla super-risoluzione, dove si usa nel verso opposto;
in PyTorch le due direzioni sono `nn.PixelShuffle` e `nn.PixelUnshuffle`, e qui
serve la seconda, applicata dopo aver rimesso la sequenza di token in forma di
griglia con i canali per primi.) Con i numeri di prima, i $1024$
token di un riquadro $448 \times 448$ diventano $256$: è la scelta di InternVL
{cite}`chen2024far`, che riporta una pagina in griglia $3 \times 4$ più
miniatura da $13\,312$ a $13 \cdot 256 = 3328$ token.

Il confronto con il **pooling medio** sullo stesso blocco $2 \times 2$ si
formula in una riga. Il pooling è la mappa lineare
$\mathbb{R}^{4 d_v} \to \mathbb{R}^{d_v}$,
$(\mathbf{z}_1, \mathbf{z}_2, \mathbf{z}_3, \mathbf{z}_4) \mapsto \frac{1}{4}\sum_i \mathbf{z}_i$, il cui nucleo ha
dimensione $3 d_v$: tre quarti dei gradi di libertà finiscono
irrecuperabilmente a zero, e con essi ogni differenza *fra* le quattro patch,
cioè precisamente il segnale ad alta frequenza spaziale su cui si gioca la
lettura del testo. Il pixel shuffle ha nucleo banale.

La compressione, però, si è solo spostata sul **proiettore**, che deve comunque
portare $4 d_v$ nella dimensione $d_t$ del modello di linguaggio. Se
$d_t < 4 d_v$ è quella matrice la vera strozzatura, con la differenza
sostanziale che è **appresa** invece che imposta a priori. Resta che ogni token
deve rappresentare quattro volte più pagina con la stessa capacità: un
compromesso favorevole, non un pasto gratis.

`````

Le due risposte, insieme, dicono una cosa sola: si guarda l'immagine ad alta
risoluzione **a pezzi**, così l'encoder non esplode, e al modello di linguaggio
si consegna una versione **impacchettata** di quei pezzi, così non esplode il
contesto.

## Terza risposta: non convertire affatto

C'è una famiglia di compiti in cui tutto questo si vede a occhio nudo, ed è la
lettura dei documenti. Per decenni la sola strada praticabile è stata una
catena: la pagina a un sistema di **riconoscimento ottico dei caratteri**
(l'OCR), e il testo che ne usciva a chi doveva farci qualcosa, che oggi è un
modello di linguaggio. Ogni anello è una conversione, e ogni conversione decide qualcosa al
posto di chi verrà dopo.

`````{tab} Elementare

Un archivio ha due modi di conservare le pagine: fotocopiarle o farle ribattere
a macchina.

La trascrizione è comodissima, perché poi si cerca per parola. Ma chi ribatteva
ha dovuto decidere: in che ordine si leggono due colonne affiancate? Dove
finisce una cella della tabella? E il grafico, che non è fatto di parole, come
si ribatte? (Di solito non si ribatte: sparisce, e con lui il numero stampato di
fianco a una delle sue colonne, che nessuno saprebbe più a quale colonna
attribuire.) Decisioni prese al buio, senza
sapere che domanda arriverà, e una volta per tutte.

La fotocopia non decide niente: tiene la pagina com'è, con le colonne al loro
posto e il timbro storto in fondo. Per anni non è stata un'alternativa, perché
una macchina sapeva cercare solo fra le parole e in una fotocopia di parole
cercabili non ce ne sono. Un modello che *vede* toglie l'obbligo.

`````

`````{tab} Superiore

Le perdite della catena OCR sono strutturali, non difetti di implementazione.
L'estrazione linearizza un oggetto bidimensionale: la posizione in pagina, che è
informazione semantica (una cifra in fondo a destra di una fattura non è una
cifra qualsiasi), diventa al più una coordinata in un file a parte; l'ordine di
lettura su più colonne è una scelta euristica; la struttura di una tabella va
ricostruita da allineamenti di *bounding box*; grafici, firme e caselle barrate
non hanno rappresentazione nel testo estratto e si perdono. E gli errori si
compongono, perché quello che l'OCR sbaglia il modello a valle non può
correggere: non vede più l'originale.

Un VLM che riceve la pagina come immagine salta l'intera catena, legge il testo
*e* la sua disposizione nello stesso passaggio, e la sua unica conversione è la
patchificazione, che almeno preserva la geometria. Il prezzo va detto senza
sconti: una pagina a risoluzione leggibile costa alcune migliaia di token,
mentre la sua trascrizione ne costerebbe attorno al migliaio.

`````

Il passo successivo riguarda la ricerca. La RAG (cercare in un archivio i pezzi
che servono e passarli al modello insieme alla domanda) l'abbiamo costruita
nella {doc}`sezione «Cercare per rispondere» </Transformers/rag>` del capitolo
sui Transformer, e il capitolo sugli agenti la raffinerà
nella sezione sul RAG avanzato: non la rispieghiamo. Qui cambia una cosa sola, ma a
monte di tutto: **che cosa si mette nell'indice**. L'indice è la copia
riorganizzata dell'archivio su cui la ricerca lavora davvero. È come quello in
fondo a un libro: non è il libro, ma serve a trovarci dentro le cose, con la
differenza che qui al posto delle parole ci sono file di numeri. Nessuno cerca
frugando fra i documenti originali: si cerca lì dentro, e quel che nell'indice
non è finito, per la ricerca non esiste. In una pipeline classica si indicizza
il testo estratto, e si eredita ogni decisione dell'OCR prima ancora che una
domanda sia stata formulata. L'alternativa è indicizzare la pagina **come
immagine**, senza trascriverla: si cerca fra le pagine viste invece che fra le
pagine ribattute, ed è la strada del recupero *vision-native* alla ColPali
{cite}`faysse2025colpali`.

`````{tab} Elementare

L'idea è semplice quanto suona: invece di trascrivere ogni pagina dell'archivio
per poterla cercare, si dà ogni pagina in pasto a un modello che vede, si
tengono i numeri che ne escono, e si cerca fra quelli. Anche la domanda diventa
numeri. Nessuno ha trascritto niente, quindi nessuno ha deciso in che ordine
leggere le colonne o cosa fare del grafico: la decisione arriva insieme alla
domanda, che è il momento giusto.

C'è un dettaglio che fa la differenza: di ogni pagina non si tiene una sola fila
di numeri riassuntiva, ma **una fila per ogni tessera del mosaico**, mille
riassunti minuscoli invece di uno grande. Così ogni parola della domanda può
cercarsi il pezzo di pagina che le somiglia di più, ed è quello il pezzo che fa
punteggio. In cambio l'archivio occupa molto più spazio: è il prezzo di non aver
buttato via niente.

`````

`````{tab} Superiore

Il meccanismo monta insieme due pezzi. Il primo è un VLM intero usato come
**indicizzatore**: la pagina viene patchificata, il modello di linguaggio
contestualizza i token visivi e ogni vettore in uscita viene proiettato in una
dimensione bassa, così che la pagina diventi una matrice
$\mathbf{D} \in \mathbb{R}^{n_d \times k}$ con $n_d$ dell'ordine del migliaio di patch e
$k$ dell'ordine del centinaio (ColPali poggia su un VLM da tre miliardi di
parametri che guarda la pagina a $448 \times 448$). Il secondo è
l’**interazione tardiva** di ColBERT {cite}`khattab2020colbert`, che il capitolo
sugli agenti riprenderà in versione testuale: invece di collassare la pagina in
un vettore solo si conservano tutti i vettori e il punteggio si compone in
fondo,

$$
s(q, d) = \sum_{i \in q} \max_{j \in d} \; E(q_i)^{\top} E(d_j),
$$

dove $q_i$ è l’$i$-esimo token della domanda, $d_j$ la $j$-esima **patch** della
pagina ed $E(\cdot)$ il rispettivo embedding. La differenza rispetto al caso
testuale è tutta nel secondo indice: il massimo non corre più sui token di un
passaggio trascritto, ma sulle regioni dell'immagine, e un token della domanda
si aggancia alla zona di pagina che gli corrisponde, parola, cella di tabella o
etichetta di un asse che sia.

Il costo è la vera obiezione. Un indice multi-vettore conserva $n_d \cdot k$
numeri per pagina invece di $k$: con $n_d \approx 1024$ e $k = 128$ in mezza
precisione sono $1024 \cdot 128 \cdot 2 \approx 262$ KB per pagina, contro il
paio di centinaia di byte di un embedding singolo. Su un milione di pagine si
parla di centinaia di gigabyte, e la ricerca chiede strutture approssimate
pensate per l'interazione tardiva.

E si perde il **confronto letterale**, su cui la sezione sulla RAG aveva già
messo in guardia. La ricerca per codice esatto («errore E-52», un numero di
protocollo, un IBAN) è il terreno dove l'indice invertito resta imbattibile,
perché lì il significato *è* la stringa. Un indice puramente visivo lo perde, e
la contromisura è la solita: affiancare i due indici invece di sostituirne uno
con l'altro.

`````

## Il conto, in poche righe

Tutta l'aritmetica della sezione si scrive in poche righe eseguibili: la prima
funzione conta i token, la seconda simula il tiling con la miniatura e con
l'eventuale riduzione del pixel shuffle.

```python
import numpy as np

PATCH = 14  # lato della patch dell'encoder, in pixel

def token(lato, patch=PATCH):
    """Token di un ViT su un'immagine quadrata: una patch, un token."""
    return (lato // patch) ** 2

def a_riquadri(lato, riquadro=448, riduzione=1):
    """Tiling: riquadri alla risoluzione nativa piu' una miniatura dell'intera
    immagine. Restituisce (pezzi, token totali, coppie viste dall'encoder).

    Vale per immagini quadrate con lato multiplo del riquadro: la griglia
    rettangolare g_h x g_w del testo si ottiene sostituendo (lato // riquadro)**2
    con g_h * g_w."""
    pezzi = (lato // riquadro) ** 2 + 1                # +1: la miniatura
    per_pezzo = token(riquadro) // riduzione
    return pezzi, pezzi * per_pezzo, pezzi * per_pezzo ** 2

lati = np.array([224, 448, 896])
n = np.array([token(l) for l in lati])

print(f"{'immagine':>13} {'token':>7} {'x token':>9} {'x attenzione':>13}")
for lato, t in zip(lati, n):
    print(f"{lato:>5} x {lato:<5} {t:>7} {t / n[0]:>8.0f}x {(t / n[0]) ** 2:>12.0f}x")

lato = 896
pezzi, tot, coppie = a_riquadri(lato)
_, tot_ps, _ = a_riquadri(lato, riduzione=4)  # riduzione=4: pixel shuffle 2x2

print(f"\n{lato} x {lato} in riquadri da 448:")
print(f"  monolitica    {token(lato):>5} token   {token(lato) ** 2:>9} coppie nell'encoder")
print(f"  a riquadri    {tot:>5} token   {coppie:>9} coppie  ({pezzi} pezzi)")
print(f"  l'encoder confronta {token(lato) ** 2 / coppie:.1f} volte meno coppie")
print(f"  con pixel shuffle al modello di linguaggio arrivano {tot_ps} token")
```

```text
     immagine   token   x token  x attenzione
  224 x 224       256        1x            1x
  448 x 448      1024        4x           16x
  896 x 896      4096       16x          256x

896 x 896 in riquadri da 448:
  monolitica     4096 token    16777216 coppie nell'encoder
  a riquadri     5120 token     5242880 coppie  (5 pezzi)
  l'encoder confronta 3.2 volte meno coppie
  con pixel shuffle al modello di linguaggio arrivano 1280 token
```

L'uscita è il riassunto numerico della sezione: le tre righe della tabella
sono il vincolo, le quattro sotto sono le due
contromisure. Il tiling compra un encoder che confronta $3{,}2$ volte meno
coppie, e lo paga con mille token di ridondanza. Attenzione però a non leggere quel $3{,}2$ come un risparmio di lavoro. I
confronti fra tessere sono solo una parte di quello che l'encoder fa: c'è anche
il lavoro che spende su ogni tessera per conto suo, e quello cresce con il
numero delle tessere e basta, quindi il taglio a riquadri, che di tessere ne
aggiunge mille, lo peggiora. Messi insieme i due conti (sono gli stessi due
addendi della soglia $N > 6d$ di poco fa: i confronti fra tessere da una
parte, il lavoro su ogni tessera dall'altra), il risparmio vero sta
fra il 5% e il 20%, ed è tanto minore quanto più l'encoder è grosso. Il pixel shuffle, dal canto suo, riporta quei
$5120$ token a $1280$, meno di un terzo di quanto vedrebbe l'immagine
monolitica, cioè non tagliata a pezzi. Nessuna delle due ha toccato la prima
tabella.

## La risoluzione si decide guardando il mestiere

Il proiettore lineare della sezione sui connettori {cite}`liu2023visual`
sembrava la scelta che decide tutto, e in parte lo era. Ma se la domanda è se un
sistema saprà leggere una bolletta, il numero che conta non sta lì: sta in
quanti pixel gli si danno da guardare, e quella scelta si fa guardando **cosa il
modello dovrà leggere**.
È una manopola che si gira sapendo a che cosa servirà il prodotto finito, non un
dettaglio interno da lasciare a chi disegna l'architettura. Nel gergo del libro
è un **iperparametro**, cioè un numero che nessun addestramento sceglie per noi;
la novità è che questo non lo sceglie nemmeno chi progetta il modello, lo
sceglie chi sa che cosa il modello dovrà leggere.

Le tre risposte non eliminano il costo, lo spostano, e ognuna lo lascia in un
posto diverso. Il tiling lo toglie all'encoder e lo consegna al contesto, dove
diventa lunghezza di sequenza. La compressione lo toglie al contesto e lo carica
sui singoli token, dove diventa capacità. Il recupero *vision-native* lo toglie
alla pipeline di estrazione e lo mette nell'indice, dove diventa spazio su
disco. Chi progetta sceglie in quale dei tre posti preferisce pagare, e la risposta
dipende dal compito.

Resta un'ultima domanda. Abbiamo speso una sezione intera a far arrivare al
modello abbastanza dettaglio; ma un modello che riceve qualche migliaio di pezzi
d'immagine li sta davvero *guardando*? È il tema della prossima sezione, e la risposta non è
confortante.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Il conto è una divisione: quante tessere da 14 puntini stanno nella foto. A
  $224 \times 224$ sono **256 tessere**, a $448 \times 448$ sono **1024**, cioè
  quattro volte tante perché l'area è quadruplicata. Ma il lavoro del modello
  cresce come il **quadrato** delle tessere: raddoppiare il lato della foto
  moltiplica per sedici il lavoro di confrontare ogni tessera con tutte le altre.
- Quanta risoluzione serve **lo decide il compito**. Un gatto si riconosce anche
  da lontano; su un foglio A4 ridotto a 224 puntini una maiuscola ne occupa uno e
  mezzo e una tessera copre due centimetri di pagina, cioè cinque o sei parole in
  un pezzetto solo. Dove non ci sono puntini non c'è informazione, e nessuna
  astuzia la rimette.
- **A riquadri**: sei foto ravvicinate più una settima che prende tutto. Si monta
  sopra un encoder già addestrato senza toccarlo, e in cambio una figura a
  cavallo di due pezzi si spezza: l'unico posto dove si vede intera è la
  settima foto, quella sfocata.
- **I barattoli di tempera**: versarne quattro in uno solo libera i ripiani, ma
  dei quattro colori resta un marrone (è il *pooling*); infilarli in una
  cassetta a quattro scomparti libera gli stessi ripiani senza perdere un
  grammo (è il *pixel shuffle*). Nel secondo caso il peso si sposta dal numero
  di contenitori al peso di ciascuno.
- Per i documenti, **fotocopiare invece di ribattere**: chi ribatte decide in che
  ordine si leggono le colonne, che fare delle tabelle e dei grafici, e lo decide
  prima di sapere che domanda arriverà. Un modello che vede la pagina toglie
  l'obbligo, e si può perfino cercare fra le fotocopie invece che fra le
  trascrizioni, al prezzo di un archivio molto più grosso.
- Nessuna delle tre risposte cancella il costo: lo spostano. Chi progetta sceglie
  se pagarlo in posto occupato, in quanta roba deve stare dentro ogni tessera, o
  in spazio su disco.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Il conto è aritmetica: $N = \lfloor H/p \rfloor \cdot \lfloor W/p \rfloor$, e
  con patch $14 \times 14$ un'immagine $224 \times 224$ dà **256 token**,
  $448 \times 448$ ne dà **1024**. I token crescono con l’**area**, il costo
  dell'attenzione con il loro **quadrato**: raddoppiare il lato moltiplica per
  sedici il costo dell'attenzione.
- La risoluzione la detta **il compito, non l'architettura**. Un gatto si
  riconosce a $224$ pixel; su un A4 ridotto a $224$ pixel una maiuscola occupa
  un pixel e mezzo e una patch copre quasi due centimetri di pagina. Documenti,
  grafici, schermate e testo dentro l'immagine vivono nell'alta frequenza
  spaziale.
- Il **tiling** {cite}`chen2024far` taglia l'immagine in riquadri della
  risoluzione nativa dell'encoder e aggiunge una miniatura per il contesto
  globale: l'attenzione dell'encoder diventa lineare nell'area e non serve
  riaddestrare nulla. Attenzione a non sopravvalutare il guadagno immediato: a
  $4096$ token le coppie di attenzione calano di $3{,}2$ volte ma i FLOP solo del
  5-20% a seconda di $d$, perché a quei valori domina il feed-forward. I due
  argomenti solidi sono gli embedding di posizione che restano validi e
  l'indifferenza alle proporzioni. In cambio un oggetto a cavallo di due riquadri
  si spezza, e la miniatura è l'unico posto dove l'insieme resta visibile.
- Il **pixel shuffle** riduce i token di quattro volte concatenando quattro
  patch adiacenti lungo i canali: è una permutazione, non butta via niente, e
  sposta l'informazione dai posti al contenuto di ogni posto. Il **pooling
  medio**, sullo stesso blocco, ha nucleo di dimensione $3 d_v$: cancella
  proprio le differenze fra patch vicine, cioè il dettaglio fine.
- Per i documenti la catena **OCR poi testo poi modello** decide l'ordine di
  lettura, la struttura delle tabelle e il destino dei grafici prima di
  conoscere la domanda. Un VLM legge la pagina come immagine e salta la catena;
  il recupero *vision-native* {cite}`faysse2025colpali` indicizza le patch
  visive della pagina e le confronta con l'interazione tardiva, al prezzo di un
  indice molto più grande e della perdita del confronto letterale su codici e
  sigle.
- Nessuna delle tre risposte elimina il costo: il tiling lo sposta
  sull'encoder-contesto, la compressione sulla capacità dei singoli token, il
  recupero visivo sull'indice. La risoluzione è un **iperparametro di prodotto**.
```

`````
