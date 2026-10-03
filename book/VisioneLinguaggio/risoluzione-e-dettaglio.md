# Il costo del dettaglio

Prendi un giornale e allontanalo dagli occhi. A un metro riconosci ancora che è
un giornale: distingui le fotografie dal testo, forse leggi il titolo di
apertura. A due metri il titolo se ne va e restano quattro rettangoli grigi.
Quello che hai perso non è l'immagine, che è ancora tutta lì: è il
dettaglio, e con lui tutto ciò che nella pagina era *scritto* invece che
disegnato.

Nelle sezioni su encoder, connettori e tokenizzatori la risoluzione era un
dato: quanti pixel entrano nell'encoder. È il vincolo che governa i sistemi
reali più delle differenze di architettura: a parità di token il tipo di
connettore conta poco, mentre contano la risoluzione e il numero di token
{cite}`mckinzie2024mm1`. Ogni pixel in più si paga in contesto, cioè in
posizioni nella sequenza sottratte a tutto il resto: le posizioni crescono con
l'area, cioè con il quadrato del lato, e il confronto di ciascuna con tutte le
altre cresce ancora più in fretta.

## Il conto, in due righe

Il meccanismo è quello del Vision Transformer {cite}`dosovitskiy2021image`: una
tessera quadrata di lato fisso diventa un pezzo della sequenza (in gergo, una
*patch* e un *token*). Da qui il numero dei pezzi è aritmetica elementare, e la
sua conseguenza sul costo dell'attenzione non lo è.

`````{tab} Elementare

Prendiamo un encoder che taglia tessere da $14 \times 14$ puntini. Su
un'immagine da $224 \times 224$ ne stanno $224 : 14 = 16$ per riga e altrettante
per colonna, quindi $16 \times 16 = 256$ tessere: la nostra immagine è una
«frase» di 256 pezzi. Con tessere da $16 \times 16$ ne verrebbero 196, ed è il
conto fatto sul Vision Transformer: a decidere quanti pezzi ha la «frase» è la
taglia della tessera.

Raddoppiando il lato, da $224$ a $448$, vale il conto già fatto all'inizio del
capitolo: le tessere quadruplicano, qui da 256 a $32 \times 32 = 1024$, perché
a raddoppiare sono due lati insieme, e i confronti che il modello fa per
capire ogni tessera, cioè l’{doc}`attenzione </Transformers/attenzione>`, si
moltiplicano per sedici. Raddoppiando ancora, da $448$ a $896$, si moltiplicano
per altri sedici, duecentocinquantasei volte il conto di partenza. E vale
l'altra metà della regola: i confronti sono solo una parte del lavoro, e
prendono il comando quando le tessere diventano migliaia.

Il prezzo che si paga per primo, però, sono i posti. Un modello di linguaggio
ha un numero fissato di posti nella sequenza, per esempio 4096, e 1024 tessere
ne occupano un quarto: alle parole della domanda e della risposta restano
quelli che avanzano.

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
$N$ è lineare nell’area, quindi quadratico nel lato.

Il costo dell'attenzione è a sua volta quadratico in $N$, cioè $O(N^2 d)$ con
$d$ dimensione del modello: quartico nel lato dell'immagine. Da $224$ a $448$
si paga $(1024/256)^2 = 16$ volte tanto; da $224$ a $896$, $(4096/256)^2 = 256$.

Una precisazione, per non vendere il termine quadratico più caro di quanto sia.
Nei FLOP di un blocco Transformer l'attenzione vale circa $4N^2 d$ e le
proiezioni più il feed-forward (con strato nascosto a $4d$ unità) circa $24 N
d^2$: il quadratico supera il lineare solo per $N > 6d$, cioè oltre $24\,576$
token in un modello con $d = 4096$ (nell'encoder visivo, dove $d$ vale circa un
migliaio, la soglia scende attorno ai seimila). Quello che si paga subito è il
contesto occupato, il tempo di *prefill* e la KV cache; e la
FlashAttention della {doc}`sezione sulle GPU </GPU/flash-attention>`
{cite}`dao2022flashattention` toglie dal conto la memoria $O(N^2)$, non il
calcolo: alza il tetto, non cambia l'esponente.

`````

## Perché duecentoventiquattro pixel non bastano

Se il conto è così severo, quanta risoluzione serve davvero? Non c'è una
risposta valida in generale, ed è l'osservazione da cui dipende tutto il resto:
la risoluzione la detta il compito, non l'architettura.

Un gatto lo si riconosce da lontano, perché la sagoma, le orecchie e la coda
sono strutture larghe che sopravvivono a una riduzione brutale, ed è il tipo di
compito su cui sono stati costruiti i primi encoder visivi. La riga di una
fattura, no. Il numero di un grafico, la voce di una tabella, la scritta su un
cartello, il pulsante di una schermata vivono nella scala di dettaglio più fine,
quella che sparisce per prima.

`````{tab} Elementare

Un foglio A4 è alto 297 millimetri. Ridotto a 224 puntini di altezza, ogni
millimetro di carta diventa tre quarti di puntino
($224 : 297 = 0{,}75$). Una maiuscola stampata in un romanzo è alta circa due
millimetri, quindi ne occupa uno e mezzo, e un puntino e mezzo non contiene una
lettera: contiene una macchia grigia.

Guardiamola dall'altro verso. Se un millimetro vale tre quarti di puntino, un
puntino vale un millimetro e un terzo, e la tessera del mosaico, che è larga 14
puntini, copre quasi due centimetri di pagina. Il modello riceve, in un
unico pezzetto di informazione, un quadratino di foglio alto quattro o cinque
righe di testo. Chiedergli cosa c'è scritto è come chiedere di leggere un libro
attraverso un vetro smerigliato.

Se invece la pagina la diamo alta 1024 puntini, ogni millimetro vale tre puntini
e mezzo: la maiuscola ne occupa sette e una tessera copre quattro millimetri di
carta, quanto è alta una riga di testo. Adesso qualcosa da leggere c'è. Il
compito ha deciso la risoluzione, e nessuna astuzia di architettura può cambiare
il fatto che dove non ci sono puntini non c'è informazione.

`````

`````{tab} Superiore

Si ragiona in pixel per millimetro, confrontati con la scala del segnale da
leggere. Un A4 alto $297$ mm, ridotto a un lato lungo di $L$ pixel,
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
una decina di parole su quattro o cinque righe; ma il limite non è la capacità
del vettore, che di numeri ne ha centinaia, è il campionamento. Il passo del
pixel è $297/224 = 1{,}33$ mm, e il periodo più fine rappresentabile, due
pixel, è $2{,}65$ mm, mentre i tratti di una lettera sono larghi qualche decimo
di millimetro: l'informazione che serve a leggere, nell'immagine ridotta, non
c'è più. A $1792$ pixel il periodo minimo scende a $0{,}33$ mm e un token copre
poco più di due millimetri per lato, un paio di caratteri: la stessa
architettura, con lo stesso encoder, di colpo legge.

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

Tre risposte affrontano questo vincolo, e nessuna lo cancella: lo spostano.
Le prime due riguardano quante tessere arrivano al modello di linguaggio; la
terza, per i documenti, riguarda che cosa della pagina si conserva. Per
ciascuna si vede dove finisce il conto.

## Prima risposta: tagliare l'immagine a riquadri

La più semplice, adottata fra l'altro da LLaVA-1.5-HD {cite}`liu2024improved` e
da InternVL {cite}`chen2024far`. Un encoder come quello di CLIP lavora a una
risoluzione sola, quella su cui è stato addestrato, e l'immagine è più grande.
Invece di rimpicciolire l'immagine fino a farla stare nell'encoder, la si taglia
in riquadri grandi esattamente quanto lui si aspetta. Ogni riquadro passa per
conto suo, e i pezzi che ne escono si mettono tutti in fila; in coda si aggiunge
una miniatura dell'immagine intera, che è l'unico posto in cui si vede come i
riquadri stanno insieme. Il metodo si chiama **tiling**, «tagliare a
piastrelle», e si trova anche sotto i nomi *AnyRes* (in LLaVA-NeXT) e *dynamic
high resolution* (in InternVL).

`````{tab} Elementare

Devi fotografare un quadro grande con una macchina che inquadra solo un
quadratino. Fai così: scatti sei foto ravvicinate, una per ogni pezzo del
quadro, poi fai un passo indietro e ne scatti una settima che prende tutto, con
molto meno dettaglio ma completa. Chi le riceve ha il dettaglio nelle prime sei
e vede dalla settima come stanno insieme. Il pregio è che non hai comprato una
macchina nuova, ed è per questo che il taglio a riquadri si è diffuso: si può
aggiungere sopra un encoder già addestrato senza toccarlo. E un quadro lungo e
basso lo copri con una fila di scatti, uno alto e stretto con una colonna,
senza doverlo schiacciare in un quadrato.

Il difetto lo indovini pensando a una figura a cavallo fra due pezzi. Nelle sei
foto ravvicinate non c'è mai per intero: mezza faccia in una e mezza nell'altra,
e chi guarda deve rimetterle insieme senza essere sicuro che appartengano alla
stessa cosa. Intera si vede solo nella settima, quella senza dettaglio. Il
taglio cade dove capita, non segue i confini degli oggetti.

Poi resta la pila. Sette foto invece di una, e la settima ripete quello che le
altre hanno già preso: da guardare ce ne sono comunque sette. Il conto si è
spostato dalla macchina fotografica al mucchio di foto.

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
solido dei tre. L'attenzione dell'encoder è quadratica
dentro ogni riquadro e assente fra riquadri diversi, quindi il costo passa da
$O\big((g_h g_w N_t)^2\big)$ a $O(g_h g_w N_t^2)$, cioè da quadratico a
lineare nell'area: asintoticamente è un guadagno vero, ed è la ragione per
cui il metodo scala. Con $896 \times 896$, $t = 448$ e
$p = 14$: monolitica sono $4096$ token e $4096^2 \approx 16{,}8$ milioni di
coppie; a riquadri sono cinque pezzi (quattro più la miniatura) da $1024$ token,
cioè $5 \cdot 1024^2 \approx 5{,}2$ milioni di coppie, $3{,}2$ volte meno.

Quel $3{,}2$, però, conta le coppie di attenzione, non i FLOP dell'encoder, e a
questi valori di $N$ le due cose non coincidono affatto. La soglia $N > 6d$ fra
il termine quadratico e quello lineare dice che con $4096$ token e $d$
dell'ordine del migliaio siamo ancora *sotto*: l'attenzione è meno della metà
del blocco, e il tiling taglia la parte piccola del conto mentre manda nel
feed-forward $5120$ token invece di $4096$, cioè paga di più sul termine che
domina. Rifacendo il conto per intero con la stessa contabilità di prima
($24Nd^2 + 4N^2d$ per strato), il lavoro totale cala di $1{,}24$ volte a $d =
768$, di $1{,}14$ a $d = 1024$ e di $1{,}06$ a $d = 1408$, e con l'encoder di
InternVL, largo $d = 3200$, si rovescia: a riquadri costa l'8,5% in più. Il
guadagno immediato va da un quinto a niente, e diventa una perdita quanto più
l'encoder è largo. (Attenzione a non leggere il rapporto come una percentuale:
dividere per $1{,}24$ vuol dire risparmiare il 19%, non il 24%.) In cambio i
token *totali* salgono da $4096$ a $5120$, perché la miniatura è ridondante per
costruzione: il conto si è spostato, non è sparito, e tutti quei token
finiscono nella stessa sequenza del modello di linguaggio, dove l'attenzione è
di nuovo quadratica su tutto.

Gli argomenti che reggono di più, quindi, sono gli altri due, e non sono
computazionali. Gli embedding di posizione dell'encoder restano validi, quindi
non vanno interpolati su una griglia più grande, operazione che degrada e in
genere chiede un riaddestramento; e il sistema resta indifferente alle
proporzioni, perché una schermata panoramica e una pagina verticale ricevono
griglie diverse invece di finire schiacciate entrambe in un quadrato.

Tutti e due si possono avere anche senza tagliare, a un altro prezzo:
addestrare l'encoder a risoluzione nativa, impacchettando nello stesso batch
immagini di forme diverse {cite}`dehghani2023navit`, con codifiche di posizione
che non hanno una griglia fissa da interpolare. Qwen2-VL
{cite}`wang2024qwen2vl` usa la RoPE in due dimensioni: nessun riquadro, nessuna
miniatura, e un'immagine $224 \times 224$ che, dopo una fusione $2 \times 2$
dei token (lo stesso raggruppamento del pixel shuffle della seconda risposta,
seguito da un MLP), ne costa $64$, più i due segnaposto che la aprono e la
chiudono. Il prezzo è che l'encoder va riaddestrato.

I limiti sono altrettanto netti. Un oggetto o una riga di testo che attraversano
il taglio finiscono in due passaggi indipendenti dell'encoder, che non si vedono
fra loro, e ricucirli tocca all'attenzione a valle, che ha la sola miniatura come
riferimento globale. E il numero di token cresce con l'area, che è il problema
della seconda risposta.

`````

## Seconda risposta: comprimere i token

Il tiling moltiplica i pezzi: una pagina di documento, con la griglia più
fitta che questi sistemi usano in addestramento, può chiederne dodici, e con la
miniatura fanno tredici passaggi dell'encoder; siccome ogni riquadro da $448$
puntini di lato dà $1024$ tessere, in fila ne finiscono $13 \times 1024 =
13\,312$. A quella lunghezza il termine quadratico è ancora sotto il lineare
($N < 6d_t$), ma il contesto occupato e la KV cache pesano: circa $6{,}5$ GiB
per richiesta in un modello da sette miliardi di parametri con attenzione
multi-testa. La seconda risposta riduce le tessere *dopo* l'encoder e *prima*
del modello di linguaggio: l'immagine viene guardata ad alta risoluzione, ma
quello che entra nel contesto è più corto. Si può comprimere perdendo
informazione subito, con il pooling, o spostandola dai posti al contenuto di
ogni posto, con il **pixel shuffle**.

`````{tab} Elementare

Quattro barattoli di tempera, uno per colore, e tre ripiani da liberare.

Primo modo: versi i quattro colori in un barattolo solo e mescoli. Occupi un
ripiano invece di quattro, e quel che ne esce è davvero la media dei quattro; ma
se qualcuno chiede «di che colore era il terzo barattolo?», dal marrone che hai
in mano non lo ricavi più. Questo è il pooling, prendere quattro tessere
vicine e sostituirle con la loro media. Semplice, efficace, irreversibile.

Secondo modo: prendi una cassetta con quattro scomparti e ci infili dentro i
quattro barattoli, ciascuno nel suo. Sempre un ripiano occupato invece di
quattro, e non hai perso un grammo di colore: la cassetta è solo quattro volte
più pesante. Questo è il pixel shuffle (alla lettera «rimescolamento dei
puntini»: il nome è più oscuro della cosa): quattro tessere adiacenti diventano
un pezzo solo, che porta con sé tutti e quattro i contenuti, uno di fianco
all'altro. L'informazione si è spostata dai *posti* al *contenuto di ogni posto*.

La cassetta, però, prima di entrare nel modello di linguaggio deve stare in un
posto della misura di sempre: i posti in fila sono tutti uguali, e in uno solo
adesso devono entrarci quattro barattoli. Se il posto è abbastanza capiente ci
stanno tutti; se non lo è, qualcosa resta fuori. Il rimescolamento in sé non
perde niente: a perdere, semmai, è il farcelo stare. Con una differenza dal
barattolo mescolato: là il marrone viene deciso in partenza e sempre allo stesso
modo, qui a far stare la cassetta nel posto è il proiettore, la tabella di
conversione del connettore, che ha imparato a furia di prove che cosa conviene
tenere.

`````

`````{tab} Superiore

Sia $\mathbf{Z} \in \mathbb{R}^{N \times d_v}$ l'uscita dell'encoder, riorganizzata sulla
griglia $\sqrt{N} \times \sqrt{N} \times d_v$ da cui proviene. Il pixel
shuffle con fattore $r = 2$ è la mappa

$$
\mathbb{R}^{\sqrt{N} \times \sqrt{N} \times d_v} \longrightarrow
\mathbb{R}^{\frac{\sqrt{N}}{2} \times \frac{\sqrt{N}}{2} \times 4 d_v},
$$

che raggruppa ogni blocco $2 \times 2$ di posizioni adiacenti e ne concatena i
quattro vettori lungo la dimensione dei canali. Il numero di token scende a
$N/4$ e la dimensione di ciascuno sale a $4 d_v$: è una permutazione degli
elementi del tensore, quindi biiettiva, e il conteggio dell'informazione non
cambia. (Il nome viene dalla super-risoluzione, dove si usa nel verso opposto;
in PyTorch le due direzioni sono `nn.PixelShuffle` e `nn.PixelUnshuffle`, e qui
serve la seconda, applicata dopo aver rimesso la sequenza di token in forma di
griglia con i canali per primi.) Con i numeri di prima, i $1024$
token di un riquadro $448 \times 448$ diventano $256$: è la scelta di InternVL
{cite}`chen2024far`, dove una tessera da $448 \times 448$ vale 256 token, e una
pagina in griglia $3 \times 4$ più miniatura passa così da $13\,312$ a $13
\cdot 256 = 3328$ token.

Il confronto con il pooling medio sullo stesso blocco $2 \times 2$ si
formula in una riga. Il pooling è la mappa lineare
$\mathbb{R}^{4 d_v} \to \mathbb{R}^{d_v}$,
$(\mathbf{z}_1, \mathbf{z}_2, \mathbf{z}_3, \mathbf{z}_4) \mapsto \frac{1}{4}\sum_i \mathbf{z}_i$, il cui nucleo ha
dimensione $3 d_v$: tre quarti dei gradi di libertà finiscono
irrecuperabilmente a zero, e con essi ogni differenza *fra* le quattro patch,
cioè precisamente il segnale ad alta frequenza spaziale su cui si gioca la
lettura del testo. Il pixel shuffle ha nucleo banale.

La compressione, però, si è solo spostata sul proiettore, che deve comunque
portare $4 d_v$ nella dimensione $d_t$ del modello di linguaggio. Se
$d_t < 4 d_v$ è quella matrice la vera strozzatura, con la differenza
sostanziale che è appresa invece che imposta a priori. Resta che ogni token
deve rappresentare quattro volte più pagina con la stessa capacità: un
compromesso favorevole, non un pasto gratis.

`````

Le due risposte, insieme, dicono una cosa sola: si guarda l'immagine ad alta
risoluzione a pezzi, così l'encoder non esplode, e al modello di linguaggio
si consegna una versione impacchettata di quei pezzi, così non esplode il
contesto.

## Terza risposta: non convertire affatto

C'è una famiglia di compiti in cui tutto questo si vede a occhio nudo, ed è la
lettura dei documenti. Per decenni la sola strada praticabile è stata una
catena: la pagina a un sistema di **riconoscimento ottico dei caratteri**
(l'OCR), e il testo che ne usciva a chi doveva farci qualcosa, che oggi è un
modello di linguaggio. Ogni anello è una conversione, e ogni conversione decide
qualcosa al posto di chi verrà dopo. Qui la risposta al costo del dettaglio non
riguarda quante tessere entrano, ma che cosa si tiene della pagina: invece di
convertirla in testo, la si tiene come immagine.

`````{tab} Elementare

Un archivio ha due modi di conservare le pagine: fotocopiarle o farle ribattere
a macchina.

La trascrizione è comodissima, perché poi si cerca per parola. Ma chi ribatteva
ha dovuto decidere: in che ordine si leggono due colonne affiancate? Dove
finisce una cella della tabella? E il grafico, che non è fatto di parole, come
si ribatte? (Di solito non si ribatte: sparisce, e con lui il numero stampato di
fianco a una delle sue colonne, che nessuno saprebbe più a quale colonna
attribuire.) Decisioni prese al buio, senza sapere che domanda arriverà, e una
volta per tutte: chi legge la trascrizione l'originale non ce l'ha più, e se il
dattilografo ha battuto 8 dove c'era 3, quel 3 non torna.

La fotocopia non decide niente: tiene la pagina com'è, con le colonne al loro
posto e il timbro storto in fondo. Per anni non è stata un'alternativa, perché
una macchina sapeva cercare solo fra le parole e in una fotocopia di parole
cercabili non ce ne sono. Un modello che *vede* toglie l'obbligo. Il prezzo lo
paga lo schedario: una pagina fotocopiata prende il posto di qualche pagina
ribattuta.

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

Il passo successivo riguarda la ricerca. La RAG di {doc}`«Cercare per
rispondere» </Transformers/rag>` (cercare in un archivio i pezzi che servono e
passarli al modello insieme alla domanda) si basa su un indice di embedding
costruito dal testo dei documenti, e la sezione {doc}`RAG avanzato
</Agenti/rag-avanzato>` lo raffinerà. Qui cambia una cosa sola, ma a monte di
tutto: che cosa si indicizza. Di solito si indicizza il testo estratto, e così
si eredita ogni decisione dell'OCR prima ancora che una domanda sia stata
formulata. L'alternativa è indicizzare la pagina come immagine, senza
trascriverla: è la strada del recupero *vision-native* alla ColPali
{cite}`faysse2025colpali`.

`````{tab} Elementare

Per cercare in un archivio non si fruga fra i documenti: si cerca in un indice,
come quello in fondo a un libro, e quel che nell'indice non è finito per la
ricerca non esiste. Di solito nell'indice finiscono le pagine ribattute. L'idea
nuova è semplice quanto suona: invece di trascrivere ogni pagina dell'archivio
per poterla cercare, si dà ogni pagina in pasto a un modello che vede, si
tengono i numeri che ne escono, e si cerca fra quelli. Anche la domanda diventa
numeri. Nessuno ha trascritto niente, quindi nessuno ha deciso in che ordine
leggere le colonne o cosa fare del grafico: la decisione arriva insieme alla
domanda, che è il momento giusto.

C'è un dettaglio che fa la differenza: di ogni pagina non si tiene una sola fila
di numeri riassuntiva, ma una fila per ogni tessera del mosaico, mille
riassunti minuscoli invece di uno grande. Così ogni parola della domanda può
cercarsi il pezzo di pagina che le somiglia di più, e i voti che ciascuna
parola dà al suo pezzo migliore si sommano: vince la pagina in cui tutte le
parole della domanda hanno trovato qualcosa di loro. In cambio l'archivio occupa
molto più spazio: è il prezzo di non aver
buttato via niente.

Una cosa però si perde per strada, ed è il codice esatto. «Errore E-52», un
numero di protocollo, l'IBAN di un conto in banca: o si trovano alla lettera o
non servono a niente,
e una ricerca per somiglianza restituisce quello che somiglia. Per quelli il
vecchio elenco di parole resta imbattibile, quindi si tengono tutti e due gli
archivi invece di sostituirne uno con l'altro.

`````

`````{tab} Superiore

Il meccanismo monta insieme due pezzi. Il primo è un VLM intero usato come
indicizzatore: la pagina viene patchificata, il modello di linguaggio
contestualizza i token visivi e ogni vettore in uscita viene proiettato in una
dimensione bassa, così che la pagina diventi una matrice $\mathbf{D} \in
\mathbb{R}^{n_d \times k}$ con $n_d$ dell'ordine del migliaio di patch e $k$
dell'ordine del centinaio (ColPali poggia su un VLM da tre miliardi di
parametri che guarda la pagina a $448 \times 448$, una risoluzione più bassa di
quella che, nella tabella delle maiuscole, serve a leggere il corpo del testo).
Il secondo è l’**interazione tardiva** di ColBERT {cite}`khattab2020colbert`,
che la sezione {doc}`RAG avanzato </Agenti/rag-avanzato>` svilupperà in
versione testuale (qui «tardiva» ha un senso diverso da quello della fusione:
il punteggio si compone dopo aver calcolato tutti i vettori): invece di
collassare la pagina in un vettore solo si conservano tutti i vettori e il
punteggio si compone in fondo. Ridotta anche la domanda a una matrice
$\mathbf{Q} \in \mathbb{R}^{n_q \times k}$, una riga per token,

$$
s(\mathbf{Q}, \mathbf{D}) = \sum_{i=1}^{n_q} \; \max_{1 \le j \le n_d}
\; \mathbf{q}_i^{\top} \mathbf{d}_j,
$$

dove $\mathbf{q}_i$ è l’embedding dell’$i$-esimo token della domanda e
$\mathbf{d}_j$ quello della $j$-esima patch della pagina, cioè la riga $j$
di $\mathbf{D}$. La differenza rispetto al caso
testuale è tutta nel secondo indice: il massimo non corre più sui token di un
passaggio trascritto, ma sulle regioni dell'immagine, e un token della domanda
si aggancia alla zona di pagina che gli corrisponde, parola, cella di tabella o
etichetta di un asse che sia. Il modello si addestra con una perdita
contrastiva sul batch che oppone il punteggio della pagina giusta a quello del
negativo più alto del batch, $\mathcal{L} = \frac{1}{b}\sum_{k=1}^{b}
\operatorname{softplus}\big(s^-_k - s^+_k\big)$, con $b$ le coppie
domanda-pagina del batch, $s^+_k = s(\mathbf{Q}_k, \mathbf{D}_k)$ e $s^-_k =
\max_{l \neq k} s(\mathbf{Q}_k, \mathbf{D}_l)$: non la InfoNCE su tutti i
negativi della sezione sull'allineamento, ma la sua variante sul negativo più
difficile.

Il costo è la vera obiezione. Un indice multi-vettore conserva $n_d \cdot k$
numeri per pagina invece di $k$: con $n_d \approx 1024$ e $k = 128$ in mezza
precisione sono $1024 \cdot 128 \cdot 2 \approx 262$ KB per pagina, contro il
paio di centinaia di byte di un embedding singolo. Su un milione di pagine si
parla di centinaia di gigabyte, e la ricerca chiede strutture approssimate
pensate per l'interazione tardiva.

E si perde il confronto letterale, su cui la sezione sulla RAG aveva già
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

# il lavoro dell'encoder per strato: 24 N d^2 (proiezioni e FFN) + 4 N^2 d
def lavoro(N, d):
    return 24 * N * d**2 + 4 * N**2 * d

print()
for d in (768, 1024, 1408, 3200):
    rapporto = lavoro(4096, d) / (5 * lavoro(1024, d))
    print(f"  d = {d:>4}: lavoro monolitico / a riquadri = {rapporto:.2f}")
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

  d =  768: lavoro monolitico / a riquadri = 1.24
  d = 1024: lavoro monolitico / a riquadri = 1.14
  d = 1408: lavoro monolitico / a riquadri = 1.06
  d = 3200: lavoro monolitico / a riquadri = 0.92
```

Le tre righe della tabella sono il conto del quadrato: quattro volte i token,
sedici volte le coppie. Le quattro righe sotto «896 x 896» riguardano il taglio
a riquadri e il pixel shuffle. Il taglio fa confrontare all'encoder $3{,}2$
volte meno coppie, ma aggiunge mille token, quelli della miniatura, che ripete
quello che i riquadri hanno già visto. E quel $3{,}2$ non è un risparmio di
lavoro: vale la regola dell'inizio del capitolo, i confronti fra tessere sono
solo una parte di quello che l'encoder fa, e il lavoro che spende su ogni
tessera per conto suo cresce con il numero delle tessere, quindi i mille token
in più lo fanno crescere. Messi insieme i due conti, le ultime quattro righe
dicono che il risparmio vero va da un quinto a niente, e con un encoder molto
largo, come quello di InternVL ($d = 3200$), diventa una spesa in più. Il pixel
shuffle, dal canto suo, riporta i $5120$ token che arrivano al modello di
linguaggio a $1280$, meno di un terzo dei $4096$ dell'immagine intera non
tagliata. Nessuna delle due tecniche cambia le tre righe della tabella: il
quadrato resta lì.

## La risoluzione si decide guardando il mestiere

Se la domanda è se un sistema saprà leggere una bolletta, il numero che conta,
più del tipo di connettore, è quanti pixel gli si danno da guardare, e quella
scelta si fa guardando che cosa il modello dovrà leggere. La risoluzione è un
iperparametro di progetto, che si fissa sapendo a che cosa servirà il prodotto
finito.

Le tre risposte non eliminano il costo, lo spostano. Il tiling lo toglie
all'encoder e lo consegna al contesto, dove diventa lunghezza di sequenza; la
compressione lo toglie al contesto e lo carica sui singoli token, dove diventa
capacità, cioè quanta pagina deve portare ogni vettore; e le due si combinano,
come in InternVL. Il recupero *vision-native* risponde a un problema vicino ma
diverso, le conversioni della catena di estrazione: sposta il costo
nell'indice, dove diventa spazio su disco, e lavora a una risoluzione più bassa
di quella che serve per leggere il corpo del testo. Chi progetta sceglie dove
pagare, e la risposta dipende dal compito.

Resta un'ultima domanda. Tutto questo serve a far arrivare al modello
abbastanza dettaglio; ma un modello che riceve qualche migliaio di pezzi
d'immagine li sta davvero *guardando*? È il tema della {doc}`sezione
sull'allucinazione visiva </VisioneLinguaggio/vedere-quel-che-non-ce>`, e la
risposta non è confortante.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Il conto è una divisione: quante tessere da 14 puntini stanno nella foto. A
  $224 \times 224$ sono 256 tessere, a $448 \times 448$ sono 1024, cioè
  quattro volte tante perché l'area è quadruplicata. Ma i confronti fra le
  tessere crescono come il quadrato delle tessere: raddoppiare il lato della
  foto moltiplica per sedici il lavoro di confrontare ogni tessera con tutte le
  altre, che diventa la parte grossa del lavoro quando le tessere sono migliaia.
  Prima ancora si pagano i posti nella sequenza.
- Quanta risoluzione serve lo decide il compito. Un gatto si riconosce anche
  da lontano; su un foglio A4 ridotto a 224 puntini una maiuscola ne occupa uno e
  mezzo e una tessera copre due centimetri di pagina, cioè una decina di parole
  su quattro o cinque righe in un pezzetto solo. Dove non ci sono puntini non
  c'è informazione, e nessuna astuzia la rimette.
- A riquadri: sei foto ravvicinate più una settima che prende tutto. Si può
  montare sopra un encoder già addestrato senza toccarlo, e in cambio una figura
  a cavallo di due pezzi si spezza: l'unico posto dove si vede intera è la
  settima foto, quella sfocata.
- I barattoli di tempera: versarne quattro in uno solo libera i ripiani, ma
  dei quattro colori resta un marrone (è il *pooling*); infilarli in una
  cassetta a quattro scomparti libera gli stessi ripiani senza mescolare
  niente (è il *pixel shuffle*). La cassetta però deve stare in un posto della
  misura di sempre: a perdere qualcosa, semmai, è il farcelo stare.
- Per i documenti, fotocopiare invece di ribattere: chi ribatte decide in che
  ordine si leggono le colonne, che fare delle tabelle e dei grafici, e lo decide
  prima di sapere che domanda arriverà. Un modello che vede la pagina toglie
  l'obbligo, e si può perfino cercare fra le fotocopie invece che fra le
  trascrizioni, al prezzo di un archivio molto più grosso e del vecchio elenco di
  parole da tenere accanto, perché un codice esatto o si trova alla lettera o non
  serve a niente.
- Nessuna delle tre risposte cancella il costo: lo spostano. Le prime due, che
  si usano anche insieme, lo fanno pagare in posto occupato o in quanta pagina
  deve stare dentro ogni tessera; la terza, per i documenti, in spazio su disco,
  e guarda le pagine più da lontano di quanto serva a leggerle parola per
  parola.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Il conto è aritmetica: $N = \lfloor H/p \rfloor \cdot \lfloor W/p \rfloor$, e
  con patch $14 \times 14$ un'immagine $224 \times 224$ dà 256 token,
  $448 \times 448$ ne dà 1024. I token crescono con l’area, il costo
  dell'attenzione con il loro quadrato: raddoppiare il lato moltiplica per
  sedici il costo dell'attenzione.
- La risoluzione la detta il compito, non l'architettura. Un gatto si
  riconosce a $224$ pixel; su un A4 ridotto a $224$ pixel una maiuscola occupa
  un pixel e mezzo e una patch copre quasi due centimetri di pagina. Documenti,
  grafici, schermate e testo dentro l'immagine vivono nell'alta frequenza
  spaziale.
- Il tiling taglia l'immagine in riquadri della risoluzione nativa
  dell'encoder e aggiunge una miniatura per il contesto globale: l'attenzione
  dell'encoder diventa lineare nell'area e non serve un encoder addestrato a
  risoluzioni più alte {cite}`liu2024improved`. Attenzione a non sopravvalutare
  il guadagno immediato: a $4096$ token le coppie di attenzione calano di
  $3{,}2$ volte ma i FLOP al più di un quinto, e con un encoder largo
  aumentano, perché a quei valori domina il feed-forward. I due argomenti
  solidi sono gli embedding di posizione che restano validi e l'indifferenza
  alle proporzioni. In cambio un oggetto a cavallo di due riquadri
  si spezza, e la miniatura è l'unico posto dove l'insieme resta visibile.
- Il pixel shuffle riduce i token di quattro volte concatenando quattro
  patch adiacenti lungo i canali: è una permutazione, quindi da sola non butta
  via niente, e sposta l'informazione dai posti al contenuto di ogni posto. La
  strozzatura si sposta sul proiettore, che deve portare $4 d_v$ in $d_t$, e
  dove $d_t < 4 d_v$ è quella matrice a decidere che cosa passa, con la
  differenza che lo ha imparato. Il pooling medio, sullo stesso blocco, ha
  nucleo di dimensione $3 d_v$: cancella proprio le differenze fra patch
  vicine, cioè il dettaglio fine.
- Per i documenti la catena OCR poi testo poi modello decide l'ordine di
  lettura, la struttura delle tabelle e il destino dei grafici prima di
  conoscere la domanda. Un VLM legge la pagina come immagine e salta la catena;
  il recupero *vision-native* {cite}`faysse2025colpali` indicizza le patch
  visive della pagina e le confronta con l'interazione tardiva, al prezzo di un
  indice molto più grande e della perdita del confronto letterale su codici e
  sigle.
- Nessuna delle tre risposte elimina il costo: il tiling lo sposta
  sull'encoder-contesto, la compressione sulla capacità dei singoli token, e le
  due si combinano (InternVL {cite}`chen2024far`); il recupero visivo risponde
  alle conversioni della catena OCR, sposta il costo sull'indice e guarda la
  pagina a una risoluzione più bassa di quella che serve a leggerne il corpo.
  La risoluzione è un iperparametro di progetto.
```

`````
