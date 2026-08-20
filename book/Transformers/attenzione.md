# Il meccanismo di attenzione

Quando leggi la frase "il gatto, che aveva dormito tutto il giorno sul
davanzale, saltò", e arrivi a "saltò", il tuo cervello non ripassa tutte le
parole in fila: torna dritto a "gatto". Sai *a che cosa prestare attenzione*.
Il meccanismo di **attenzione** dà alle reti neurali esattamente questa
capacità: davanti a una parola, guardare tutte le altre e pesare quanto
ciascuna conta per capirla.

Non è nato per fare il protagonista. Nel settembre del 2014 era un rattoppo,
inventato per migliorare le traduzioni delle reti che leggevano il testo una
parola alla volta (le **reti ricorrenti** del {doc}`capitolo sul Natural Language
Processing </NaturalLanguageProcessing/overview>`) {cite}`bahdanau2015neural`. Tre anni dopo il Transformer ci avrebbe
costruito sopra tutto il resto.

## L'idea: pesare le parole

Una cosa va detta prima di tutte, perché senza quella il resto sembra magia:
**dentro una rete le parole non sono parole**. Ognuna diventa una lista di
numeri, qualche centinaio, che la rete si è costruita imparando; nel gergo del
libro una lista del genere si chiama **vettore**. Il motivo per cui la cosa
conta è aritmetico: fare la media fra «gatto» e «muro» non vuol dire niente,
fare la media fra due liste di numeri sì, si sommano numero per numero.

Ed è esattamente quello che l'attenzione fa. Per ogni parola da elaborare
guarda tutte le altre parole della frase, decide quanto ciascuna conta per
capire quella, e ne mescola le liste in quella proporzione. Il risultato è una
versione della parola «arricchita dal contesto»: non più «salta» in astratto,
ma «salta» in *questa* frase. È una media pesata, come la media dei voti di una
pagella dove però le materie non contano tutte uguale, con una differenza
importante: chi pesa quanto non è scritto in nessun regolamento, lo decide la
frase stessa, parola per parola.

Il problema che questa idea viene a risolvere si vede bene guardando com'era
fatto un traduttore automatico prima. Erano due macchine attaccate: la prima
leggeva la frase di partenza e ne faceva un riassunto, la seconda leggeva solo
quel riassunto e da lì scriveva la traduzione. Le due metà hanno un nome che
torna in tutto il capitolo: l’**encoder** è la parte che legge, il **decoder**
quella che scrive (li rivediamo per bene in fondo a questa pagina). E il
riassunto era una sola lista di numeri, sempre lunga uguale: la stessa per una
frase di cinque parole e per una di cinquanta.

```{figure} ../figures/seq2seq-collo-di-bottiglia.svg
:name: fig-collo-di-bottiglia
:alt: "Schema di un seq2seq senza attenzione: le parole della frase in ingresso entrano una alla volta nell'encoder e vengono compresse in un unico vettore di contesto, disegnato come una strozzatura; da quel solo vettore il decoder deve generare tutta la traduzione, parola dopo parola."
:width: 92%

Il collo di bottiglia che l'attenzione viene a sciogliere. Tutta la frase
d'origine deve passare per un'unica lista di numeri, sempre lunga uguale: più
la frase è lunga, più quella lista è costretta a dimenticare.
```

{numref}`fig-collo-di-bottiglia` è il problema da cui nasce tutto, e con due
numeri si tocca con mano. Se il riassunto è lungo cinquecento numeri e la frase
è lunga cinquanta parole, a ogni parola tocca in media una decina di numeri per
raccontarsi: la prima e l'ultima si contendono lo stesso spazio, e a rimetterci
sono di solito quelle dell'inizio, viste per prime e sovrascritte da tutte
quelle che vengono dopo. L'attenzione toglie la strozzatura in un modo
sbrigativo: mentre scrive, il decoder smette di guardare il riassunto e va a
rileggersi *tutte* le parole d'origine, pesandole di volta in volta.

Sono dunque due usi dello stesso gesto, e conviene distinguerli subito perché
il capitolo li alterna. Nel primo una frase guarda **sé stessa**: ogni parola
pesa tutte le altre della propria frase (e anche sé stessa), e si chiama
**self-attention**. Nel secondo la frase che si sta scrivendo guarda la frase
che è stata letta, ed è la **cross-attention**, quella del traduttore che torna
sull'originale. Il meccanismo è identico in tutto e per tutto: cambia soltanto
chi guarda chi. Nelle pagine che seguono il caso di riferimento sarà il primo,
perché è quello su cui si capiscono i conti; il secondo torna in fondo a questa
pagina, quando encoder e decoder si incontrano.

`````{tab} Elementare
Prendi la frase che accompagna tutto questo libro: "Il gatto nero salta sul
muro". Il modello sta
elaborando la parola "salta" e si chiede: chi salta? Come un lettore con
l'evidenziatore, ripassa la frase e assegna a ogni parola un'intensità di
colore: "gatto" fluorescente (è il soggetto!), "muro" un colore medio (è la
destinazione), "il" e "sul" quasi trasparenti. Poi costruisce il significato
di "salta" *in questa frase* mescolando le informazioni di tutte le parole,
ma in proporzione all'evidenziatura: tanta parte di "gatto", un po’ di
"muro", pochissimo del resto.

Le intensità sono numeri veri, e vale la pena vederli almeno una volta. Per
"salta" potrebbero venire così: gatto 0,52, muro 0,24, salta 0,10, nero 0,06,
sul 0,05, il 0,03. Sono sei numeri, uno per ogni parola della frase, e c'è
anche "salta" stessa, perché ogni parola guarda anche sé. Sommano a 1, ed è una
regola fissa: l'attenzione distribuisce sempre esattamente una unità di colore,
quindi dare di più a "gatto" vuol dire togliere a qualcun altro. «Mescolare in
quella proporzione» significa allora prendere il 52% della lista di numeri di
"gatto", il 24% di quella di "muro", e così via, e sommare il tutto: quello che
ne esce è "salta" in questa frase e in nessun'altra.

Quei numeri non li decide un programmatore: li impara la rete durante
l'addestramento, cioè provando e correggendosi su miliardi di frasi. Ogni volta
che il risultato non è quello giusto (una traduzione sbagliata, la parola
successiva sbagliata), i numeri vengono ritoccati un pochino nella direzione che
avrebbe fatto sbagliare di meno: è lo stesso provare-e-correggere del capitolo
sulle reti neurali. E quando il gioco lo fa ogni parola verso tutte le altre e
non solo "salta", siamo nella self-attention di poco fa: l'intera frase che si
rilegge da sé.
`````

`````{tab} Superiore
Ogni parola (più precisamente ogni *token*, come vedremo) è rappresentata da
un vettore. Da ciascun vettore la rete ricava tre proiezioni con matrici
apprese: una **query** $\mathbf{Q}$ ("che cosa sto cercando?"), una **key**
$\mathbf{K}$ ("che cosa offro come etichetta?") e un **value** $\mathbf{V}$
("che informazione porto?").
L'affinità tra la parola che elabora e ogni altra è il prodotto scalare
query·key (la stessa misura di somiglianza tra vettori vista in *Algebra
lineare*) e la **Scaled Dot-Product Attention** la trasforma in pesi:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) =
\text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}
$$

dove $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ raccolgono per righe le proiezioni
di tutti i token e $d_k$ è
la dimensione delle key. La softmax (già incontrata nel capitolo sulle reti
neurali) normalizza le affinità in pesi che sommano a 1; la divisione per
$\sqrt{d_k}$ evita che, al crescere della dimensione, i prodotti scalari
diventino così grandi da saturare la softmax e azzerarne i gradienti. Il conto
sta in due righe, e vale la pena farle perché è lì che stanno le ipotesi.
Supponiamo che le componenti $q_i$ e $k_i$ siano a media nulla, varianza
unitaria, indipendenti fra loro e indipendenti al variare di $i$. Allora ogni
addendo del prodotto scalare ha varianza
$\mathbb{E}[q_i^2 k_i^2] - (\mathbb{E}[q_i]\,\mathbb{E}[k_i])^2 =
\mathbb{E}[q_i^2]\,\mathbb{E}[k_i^2] = 1$, dove la fattorizzazione
dell'aspettazione usa l'indipendenza **fra** $q_i$ e $k_i$; e siccome gli
addendi sono scorrelati **al variare di $i$**, le varianze si sommano:

$$
\operatorname{Var}\!\left(\sum_{i=1}^{d_k} q_i k_i\right) = d_k .
$$

Dividere per $\sqrt{d_k}$ la riporta a 1. Servono dunque due indipendenze
diverse, una per ciascun passaggio, e una terza ipotesi che di solito non si
dice: tutto questo vale **all'inizializzazione**, perché appena
$\mathbf{W}^Q$ e $\mathbf{W}^K$ cominciano ad allenarsi smettono di produrre
componenti a varianza unitaria. Il fattore infatti non *impedisce* la
saturazione, ne toglie la dipendenza da $d_k$: con componenti di varianza
diversa da 1 la softmax satura lo stesso, a qualunque dimensione. L'output
è, per ogni token, la combinazione dei value pesata dall'attenzione: una
rappresentazione contestuale calcolata in un unico prodotto tra matrici, per
tutte le posizioni insieme.
`````

Resta da dire *come* si decide l'intensità dell'evidenziatore, ed è il debito
che questa sezione ha ancora con chi legge.

Ogni parola, per partecipare al gioco, fa tre mestieri diversi, e la rete se ne
costruisce tre versioni diverse. Tutte e tre escono dall'unica lista di
partenza, passandola attraverso tre **tabelle** di numeri: una tabella
moltiplica una lista e ne restituisce un'altra, e siccome i numeri nelle tre
tabelle sono diversi (e imparati durante l'addestramento), le tre liste che ne
escono sono diverse fra loro. La prima versione dice
**che cosa quella parola sta cercando** nelle altre: "salta" cerca chi compie
l'azione. La seconda è **l'etichetta con cui si fa trovare** da chi la sta
cercando: "gatto" si presenta come qualcosa di animato, che può compiere
azioni. La terza è **l'informazione che consegna** a chi l'ha scelta: di
"gatto", il fatto che sia un felino, che sia nero, che in questa frase sia il
protagonista.

I tre mestieri hanno tre nomi inglesi, e sono le tre parole che tornano poi in
tutto il capitolo: la ricerca è la **query**, l'etichetta è la **key**,
l'informazione consegnata è il **value**. In italiano sarebbero *domanda*,
*etichetta* e *contenuto*, ma i nomi inglesi sono ormai quelli che si trovano
scritti ovunque, e li useremo anche noi.

E il confronto fra una ricerca e un'etichetta, una volta ricordato che sono due
liste di numeri, è la cosa più semplice del mondo: si moltiplicano numero per
numero e si sommano i risultati. Con due listine da tre: $(2, 0, 1)$ contro
$(3, 1, 0)$ fa $2\cdot3 + 0\cdot1 + 1\cdot0 = 6$, mentre contro $(0, 4, 0)$ fa
$0 + 0 + 0 = 0$. Se le due liste hanno numeri grandi negli stessi posti la
somma viene grande, e vuol dire che quell'etichetta risponde a quella ricerca;
se i numeri grandi stanno in posti diversi la somma viene piccola. È
l'operazione che il {doc}`capitolo di matematica </Matematica/overview>` chiama *prodotto scalare*, ed è
l'unico conto che l'attenzione fa davvero.

Resta un ultimo passaggio, e va detto perché senza di esso i conti non
tornano. Da quel confronto escono numeri qualsiasi: 6, oppure 340, oppure $-7$.
Le intensità dell'evidenziatore, invece, devono essere positive e sommare a
uno. A trasformare gli uni nelle altre c'è una ricetta che in tutto il libro si
chiama **softmax**, ed è una divisione con un passaggio in più: si prende il
numero $e = 2{,}718\ldots$, lo si eleva a ciascun punteggio, e si divide
ciascun risultato per la somma di tutti. Su tre punteggi $2$, $1$ e $-1$:
$e^2 = 7{,}39$, $e^1 = 2{,}72$, $e^{-1} = 0{,}37$, che sommati fanno $10{,}48$;
le tre intensità sono allora $0{,}71$, $0{,}26$ e $0{,}04$, che sommano a uno a
meno degli arrotondamenti, come promesso. L'elevamento a potenza serve a due cose: non far uscire mai
numeri negativi (una parola non può contribuire in negativo), e allargare le
differenze, così che un punto di vantaggio si veda davvero. Proprio perché le
allarga, però, va tenuto d'occhio: se i punteggi grezzi sono numeri enormi, il
più alto si prende tutto il colore e agli altri resta zero, cioè l'evidenziatore
smette di sfumare e diventa un interruttore. E i punteggi crescono con la
lunghezza delle liste, perché sono somme di tanti pezzi: per questo, prima della
softmax, si rimpiccioliscono tutti dividendoli per uno stesso numero, tanto più
grande quanto più le liste sono lunghe. Fatto questo, i
value si mescolano in quelle proporzioni.

```{figure} ../figures/attention-is-all-you-need.svg
:name: fig-qkv
:alt: "Un token in ingresso viene proiettato in tre vettori distinti: Query, Key e Value. Il prodotto scalare fra la Query e le Key di tutti i token produce i punteggi di rilevanza, che una softmax trasforma in pesi; i pesi moltiplicano i rispettivi Value e la loro somma è l'uscita per quel token."
:width: 92%

I tre ruoli di ogni parola. La Query è la domanda che pone, la Key l'etichetta
con cui si fa trovare, il Value ciò che offre a chi la seleziona: la stessa
parola li ricopre tutti e tre insieme.
```

La separazione dei tre ruoli in {numref}`fig-qkv` sembra un lusso e invece è
il punto di tutta la faccenda. Se ogni parola avesse una sola versione di sé,
cercare ed essere trovati sarebbero la stessa operazione, e una parola potrebbe
attirare soltanto le parole che le somigliano. Con query e key distinte può
invece cercare qualcosa di molto diverso da ciò che offre: "salta" offre
un'azione e cerca un soggetto, cioè esattamente quello che non è.

## Multi-Head Attention: più letture in parallelo

Una sola "passata di evidenziatore" costringe la rete a comprimere in un
unico schema tutti i tipi di relazione tra parole. La soluzione del
Transformer è farne parecchie in parallelo.

`````{tab} Elementare
Sulla stessa frase lavorano più lettori, ognuno con un evidenziatore di
colore diverso e una fissazione diversa: uno segna chi fa l'azione, un altro le
parentele di significato ("nero" e "gatto" vanno insieme perché uno è il colore
dell'altro), un altro ancora chi sta vicino a chi nella frase.

Ogni lettore consegna la sua versione arricchita della parola, e a questo punto
di liste ce ne sono otto invece di una. Come si torna a una sola? Il trucco è
che ogni lettore lavora fin dall'inizio su liste corte, un ottavo di quelle
intere: attaccandole una in coda all'altra si ottiene di nuovo una lista lunga
quanto quella di partenza, perché otto ottavi fanno uno. Resta un ultimo
passaggio, una tabella che la lunghezza non la cambia ma mescola fra loro i
contributi degli otto, così che quello che ciascuno ha visto arrivi in tutte
le caselle e non solo nel proprio ottavo. Alla fine il conto costa quanto un
lettore solo a lista piena.

Ogni lettore si chiama, per ragioni che nessuno ricorda più, una "**testa**" di
attenzione, e il Transformer originale ne usa otto. Perché otto e non nove?
Perché funzionava: è una scelta provata sul campo, non una legge di natura, e i
modelli che sono venuti dopo usano numeri diversi.
`````

`````{tab} Superiore
La **Multi-Head Attention** esegue $h$ attenzioni indipendenti in sottospazi
distinti e ne ricompone gli esiti:

$$
\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) =
\text{Concat}(\text{head}_1, \ldots, \text{head}_h)\,\mathbf{W}^O
$$

dove $\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_i^Q,
\mathbf{K}\mathbf{W}_i^K, \mathbf{V}\mathbf{W}_i^V)$ e
$\mathbf{W}_i^Q, \mathbf{W}_i^K, \mathbf{W}_i^V, \mathbf{W}^O$ sono matrici
apprese. Nel Transformer originale
$h = 8$ e, con $d_{\text{model}} = 512$, ogni testa lavora in dimensione
$d_k = d_{\text{model}}/h = 64$: il
costo complessivo resta paragonabile a una singola attenzione a dimensione
piena, ma il modello può dedicare teste diverse a relazioni diverse
(sintattiche, semantiche, posizionali), cosa che l'analisi empirica delle
teste addestrate conferma almeno in parte.
`````

## Dove va a finire l'attenzione: encoder e decoder

L'attenzione, da sola, non è ancora un modello: è un pezzo, e va montato. Il
pezzo si chiama **blocco**, e un blocco è quello che si ripete sempre uguale a
sé stesso lungo la macchina, come un piano di un palazzo. L’**encoder**, la
torre che legge, è una pila di questi blocchi; il **decoder**, quella che
scrive, è un'altra pila fatta allo stesso modo, che produce l'uscita (una
traduzione, una risposta) un pezzo alla volta. In mezzo, ancora attenzione:
mentre genera, il decoder "evidenzia" le parti rilevanti di ciò che l'encoder
ha letto, ed è la cross-attention di poco fa. La prossima sezione smonta le due
torri pezzo per pezzo.

Una pila di blocchi è quella che si chiama una rete **profonda**, ed è profonda
proprio in questo senso: tanti passaggi uno sopra l'altro, decine o centinaia.
Impilarli, però, non è gratis, e ogni blocco porta con sé due accorgimenti che
servono soltanto a rendere la pila addestrabile.

`````{tab} Elementare
Il primo è una **scorciatoia**: la lista di numeri che entra in un blocco viene
anche fatta passare *intatta* accanto al blocco, e sommata numero per numero a
quella che esce. Serve a due cose, e la seconda è meno ovvia. All'andata, tiene
aperta una strada diretta perché l'informazione arrivi in cima senza
sfilacciarsi in mezzo a decine di blocchi. Al ritorno, serve alla correzione:
quando la rete scopre di aver sbagliato, il segnale che dice «di quanto e in
che direzione ritoccare» deve tornare indietro fino ai primi blocchi. Tornando
indietro, però, quel segnale attraversa a ritroso gli stessi conti dell'andata,
e a ogni blocco viene moltiplicato per i numeri di quel blocco, che di solito
sono un po’ minori di uno. Se ogni blocco lo riduce a nove decimi, dopo
cinquanta blocchi ne resta lo $0{,}5\%$: praticamente niente, e i primi blocchi
smettono di imparare. La scorciatoia è la strada che il segnale può fare senza
subire nessuna di quelle moltiplicazioni. Come un corrimano lungo una scala
ripida: anche se un gradino è scivoloso, chi sale e chi scende hanno sempre una
presa solida.

Il secondo accorgimento è una **taratura**. I numeri, blocco dopo blocco,
tendono a scappare via: qui diventano tutti enormi, là tutti minuscoli, e una
rete con addosso valori fuori misura non impara più. Allora dopo ogni blocco si
riscrive la lista in modo che i suoi numeri abbiano sempre la stessa media e la
stessa dispersione: si sottrae a tutti la loro media, così il centro cade
sullo zero, e poi si dividono tutti per quanto sono sparpagliati, così la
larghezza è sempre quella. È come tarare la bilancia prima di ogni pesata: non
cambia che cosa si sta pesando, garantisce solo che il numero letto sia sulla
stessa scala di tutti gli altri.
`````

`````{tab} Superiore
Sono le **residual connection** e la **layer normalization**, combinate in

$$
\text{LayerNorm}\big(\mathbf{x} + \text{SubLayer}(\mathbf{x})\big)
$$

attorno a ogni sotto-strato (attenzione o feed-forward). La connessione
residuale (la stessa idea delle ResNet che abbiamo visto nel capitolo sul deep
learning) offre al gradiente un cammino quasi diretto verso gli strati
iniziali, contrastando il gradiente che svanisce; la layer normalization
stabilizza media e varianza delle attivazioni a ogni posizione, rendendo
l'addestramento meno sensibile a learning rate e inizializzazione. «Quasi»,
perché in questa formulazione (detta *Post-LN*, quella del 2017) la
normalizzazione sta proprio sul ramo della scorciatoia, e il gradiente la
attraversa a ogni strato: i modelli successivi la spostano prima del
sotto-strato, $\mathbf{x} + \text{SubLayer}(\text{LayerNorm}(\mathbf{x}))$, il
cosiddetto
*Pre-LN*, ed è lì che il cammino identità diventa davvero pulito (Xiong e
colleghi {cite}`xiong2020layer` mostrano che senza questo spostamento serve un
riscaldamento graduale del learning rate per addestrare stabilmente).
`````

Scorciatoia e taratura sono la parte che nessuno racconta mai, e senza la quale
niente di tutto il resto starebbe in piedi: l'attenzione è l'idea, ma un'idea
impilata sessanta volte (tanti sono i blocchi di un modello grande di oggi) si
sfalda, e questi due accorgimenti sono ciò che la tiene insieme. Ci si può
fermare qui con il meccanismo in mano; la sezione successiva prende questi pezzi
e li monta nelle due torri di una macchina vera.

```{admonition} Un cantiere parallelo: le reti a memoria
:class: note
Interrogare un archivio con una domanda, pesare quanto ciascun elemento le
risponde, e restituire la miscela pesata di ciò che quegli elementi contengono:
questa struttura è stata costruita prima dei Transformer, e per un altro scopo.

Nel 2014 le **memory network** {cite}`weston2015memory` affrontavano il
problema di far ragionare una rete su un elenco di fatti («Maria è andata in
cucina. Giovanni ha preso il latte. Dov'è il latte?»). La rete teneva i fatti
in un archivio a parte, separato dai numeri che aveva imparato, e per
rispondere andava a pescarci dentro. In quella prima versione però la pesca era
secca (si sceglieva *un* fatto, il più somigliante) e per addestrarla bisognava
dire alla rete, esempio per esempio, quali fossero i fatti giusti da usare.

Il passo che ci interessa arriva l'anno dopo, con le *end-to-end memory
network* {cite}`sukhbaatar2015end`: al posto della scelta secca si mette una
graduatoria, cioè la domanda viene confrontata con **tutti** i fatti, il
confronto produce un'intensità di evidenziatore per ciascuno, e l'archivio
viene letto mescolando i fatti in quelle proporzioni. Poi si ripete, usando il
risultato come nuova domanda: erano gli *hop*, cioè i salti di ragionamento,
che permettevano di concatenare due fatti per rispondere a una domanda che
nessuno dei due risolveva da solo. E siccome adesso ogni fatto contribuisce un
po’, la rete può imparare da sola quali contano, senza che glielo si dica.

Due cose da portarsi via. La prima è che quella graduatoria sui fatti **è**
l'attenzione, con la sola differenza che qui l'archivio è un magazzino a parte
invece della frase stessa. La seconda è che la struttura
domanda-contro-archivio, con i fatti tenuti fuori dalla rete e consultati al
momento, è esattamente la forma dei sistemi che cercano documenti prima di
rispondere: si chiamano RAG, e li costruisce per intero la
{doc}`sezione sul retrieval <rag>`.

Sulle date conviene però essere precisi, perché la tentazione di raccontarla
come una discendenza è forte e sarebbe falsa: l'attenzione per la traduzione è
del settembre 2014, le memory network dell'ottobre dello stesso anno, la
versione a graduatoria del marzo 2015. Sono due strade partite quasi insieme,
da due problemi diversi, e arrivate alla stessa operazione; nessuna delle due
nasce dall'altra. Quello che le reti a memoria hanno di proprio non è dunque
l'attenzione, è l’**archivio tenuto fuori dai numeri imparati** e consultato al
momento della domanda: ed è quel pezzo lì, messo da parte perché la sua epoca
non aveva né i dati né l'hardware, a tornare cinque anni dopo con un altro nome.
```

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- L’**attenzione** rilegge la frase con un evidenziatore: per capire una
  parola, guarda tutte le altre, dà a ciascuna un'intensità di colore e ne
  mescola le informazioni in quella proporzione.
- Le intensità le decide la frase, non un programmatore: la rete le impara
  provando e correggendosi su miliardi di esempi. Quando è ogni parola a
  guardare tutte le altre, si chiama **self-attention**.
- Per giocare, ogni parola si presenta in tre versioni: la **query** (la
  domanda che fa), la **key** (l'etichetta con cui si fa trovare) e il
  **value** (l'informazione che consegna).
- Di evidenziatori se ne passano otto in parallelo, ognuno attento a un tipo di
  legame diverso: sono le **teste** di attenzione.
- Attorno a ogni blocco ci sono una **scorciatoia** (l'informazione passa anche
  di lato, intatta, e la correzione degli errori trova sempre una presa per
  tornare indietro) e una **taratura** (i numeri riportati su una scala
  standard). Senza di loro le torri alte non si addestrano.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- L’**attenzione** costruisce, per ogni parola, una rappresentazione
  contestuale: media dei *value* pesata dalle affinità *query*·*key*,
  normalizzate con softmax e scalate di $\sqrt{d_k}$ (il fattore neutralizza la
  dipendenza della varianza da $d_k$, sotto l'ipotesi di componenti
  indipendenti a media nulla e varianza unitaria).
- Nella **self-attention** ogni parola guarda tutte le altre; i pesi non sono
  fissati a mano ma appresi.
- La **Multi-Head Attention** esegue più attenzioni in parallelo ($h = 8$ nel
  modello originale), ciascuna libera di specializzarsi su relazioni diverse.
- **Residual connection** e **layer normalization** tengono addestrabili le
  pile profonde di blocchi. L'articolo del 2017 le combina come
  $\text{LayerNorm}(\mathbf{x} + \text{SubLayer}(\mathbf{x}))$ (*Post-LN*); i
  modelli successivi normalizzano prima del sotto-strato,
  $\mathbf{x} + \text{SubLayer}(\text{LayerNorm}(\mathbf{x}))$ (*Pre-LN*), ed
  è così che la
  scorciatoia resta davvero libera.
```
`````
