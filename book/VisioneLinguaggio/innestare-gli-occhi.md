# Il pezzo più piccolo che si può addestrare

Chi innesta una vite non pianta un albero nuovo. Prende un ceppo già radicato
(il **portainnesto**), scelto perché regge quel terreno e quel parassita, e ci
salda sopra un rametto di un'altra pianta (la **marza**), quello che dà l'uva che
gli interessa. Le due parti hanno storie separate e restano quello che sono: il
mestiere sta tutto nella saldatura, che è la più piccola delle tre cose e l'unica
che il vivaista fabbrica davvero.

L'architettura di questa sezione è un innesto in senso letterale, e l'immagine
conviene tenerla fino in fondo: fra le giunzioni che sono state provate ha
resistito la più semplice, non la più ingegnosa.

## La domanda che genera l'architettura

Nel punto in cui siamo arrivati esistono già due cose che funzionano bene,
ciascuna per conto proprio. Da un lato gli **encoder visivi**: il Vision
Transformer {cite}`dosovitskiy2021image` e la torre di immagini di un modello
contrastivo come CLIP {cite}`radford2021learning` sanno trasformare una
fotografia in una griglia di vettori che ne codificano il contenuto. Dall'altro
i **modelli di linguaggio**, che sanno scrivere, seguire un'istruzione e
argomentare. Nessuno dei due sa fare il mestiere dell'altro, e nessuno dei due è
economico: dietro ciascuno ci sono mesi di calcolo su cluster di GPU.

Addestrare da zero un unico modello che veda e parli sarebbe la soluzione
pulita, ed è fuori portata per quasi tutti. La domanda diventa allora un'altra,
e tutto il resto della sezione è la storia della risposta:

> qual è il **pezzo più piccolo** che si può addestrare perché quei due modelli
> comincino a parlarsi?

Non è solo una questione di soldi. C'è una seconda ragione per lasciare fermi i
pesi che esistono già. Un modello di linguaggio riaddestrato su qualche milione
di didascalie perde per strada una parte di quello che sapeva fare con il testo
puro: impara una cosa nuova cancellandone una vecchia. È la **dimenticanza
catastrofica**, già incontrata dal vivo nella {doc}`pagina sui modelli
multilingua </Transformers/multilingua>`. Congelare non è soltanto un risparmio: è una garanzia sul
comportamento che si vuole conservare.

## Che cosa deve fare, di preciso, il pezzo in mezzo

Il pezzo in mezzo si chiama **connettore**, e prima di elencare i modi di
costruirlo conviene fissare che cosa gli si chiede: due cose diverse, che
devono essere fatte insieme.

La prima è **cambiare formato**. Un modello di linguaggio, dentro, non riceve
parole: riceve file di numeri tutte della stessa lunghezza, che va a pescare in
una tabella dove a ogni parola corrisponde la sua fila. Anche l'encoder visivo
produce file di numeri, ma sono lunghe diversamente e, soprattutto, sono scritte
in un'altra convenzione: come un menu in una lingua straniera, dove i piatti ci
sono tutti e nessuna parola combacia. Tradurre quella convenzione nell'altra è il
primo mestiere del pezzo in mezzo.

La seconda è **decidere quante** file di numeri consegnargli. Non è una scelta di
comodo: ogni pezzetto d'immagine occupa nella sequenza lo stesso posto di una
parola, e quel posto si paga.

`````{tab} Elementare

Prendiamo numeri veri. Un encoder molto usato lavora su immagini da
$336 \times 336$ puntini e le taglia in tessere da $14 \times 14$: siccome
$336 : 14 = 24$, ne stanno 24 per riga e 24 per colonna, cioè
$24 \times 24 = 576$ tessere, e quindi 576 vettori. Allega quell'immagine a una
domanda di venti parole. La sequenza che entra nel modello di linguaggio è fatta
di 596 pezzi, e 576 su 596 sono immagine: il 97%. La tua
domanda è una briciola dentro un contesto quasi interamente occupato dalla foto.

Peggio: il costo dell'attenzione, il meccanismo con cui ogni pezzo guarda tutti
gli altri, non cresce come la lunghezza della sequenza, cresce come il suo
**quadrato** (è il conto fatto nel capitolo sui Transformer: se i pezzi
raddoppiano, le coppie da confrontare quadruplicano). Se al posto di 576 tessere
ne passassimo 32 (un numero che qualcuno ha scelto davvero), la sequenza
scenderebbe da 596 a 52 pezzi, cioè undici volte e mezzo più corta; e siccome il
costo va col quadrato, undici e mezzo per undici e mezzo fa circa centotrenta
volte meno.

Ecco perché «quanti» pesa: comprimere l'immagine in pochi vettori fa risparmiare
moltissimo. Quel risparmio si paga in quello che dell'immagine viene buttato
via.

`````

`````{tab} Superiore

L'encoder produce $\mathbf{Z} \in \mathbb{R}^{N \times d_v}$, con $N$ il numero di patch
e $d_v$ la dimensione delle sue feature; il modello di linguaggio consuma
sequenze di vettori in $\mathbb{R}^{d_t}$, dove $d_t$ è la dimensione dei suoi
embedding. Un connettore è una funzione appresa

$$
g_\theta : \mathbb{R}^{N \times d_v} \longrightarrow \mathbb{R}^{M \times d_t},
$$

dove $\theta$ sono i suoi parametri (gli unici che si aggiornano, nella
configurazione base) e $M$ il numero di vettori consegnati al modello di
linguaggio. Sono quindi due i gradi di libertà: la **mappa**, che riallinea la
geometria, e il **fattore di compressione** $N/M$.

Il secondo grado di libertà ha un prezzo esplicito. Se il prompt testuale ha $T$
token, il costo dell'attenzione per strato è $O\big((M+T)^2 d_t\big)$, e la
memoria della cache delle chiavi e dei valori cresce linearmente in $M+T$ per
ogni strato e ogni testa. Con i valori dell'esempio numerico, $N = 576$ e
$T = 20$: passare da $M = 32$ a $M = N$ moltiplica il termine quadratico per
$(596/52)^2 \approx 131$. La compressione, quando si può fare, non è un
dettaglio di ottimizzazione: è ciò che rende praticabile allegare un'immagine a
ogni richiesta.

`````

## Tre risposte, dalla più elaborata alla più povera

Le soluzioni che hanno lasciato il segno sono tre, e conviene percorrerle in
ordine di complessità **decrescente**, che qui coincide con l'ordine
cronologico: è il rovescio di come di solito vanno queste cose. Non è un caso, e
questa sezione esiste soprattutto per spiegare perché.

```{figure} ../figures/vlm-connettori.svg
:name: fig-vlm-connettori
:alt: Tre architetture affiancate con lo stesso encoder visivo congelato in basso e lo stesso modello di linguaggio congelato in alto; cambia solo il pezzo in mezzo. A sinistra Flamingo, con un Perceiver Resampler che produce 64 token e li inietta in due nuovi strati di cross-attention gated inseriti fra i blocchi congelati del modello di linguaggio. Al centro BLIP-2, con un Q-Former in cui 32 query apprese interrogano l'immagine e ne escono 32 token messi in testa al prompt. A destra LLaVA, con un proiettore che porta ogni patch nello spazio dei token e consegna 576 token in testa al prompt. Il pezzo che si addestra è in terracotta piena, quelli congelati hanno il contorno tratteggiato.
:width: 85%

Gli stessi due modelli congelati, tre saldature diverse. Il conto del connettore
scende da sinistra a destra (miliardi di parametri, poi 188 milioni, poi 21), e
i primi due comprimono a un numero fisso di vettori (64 e 32, qualunque immagine
arrivi) mentre il terzo non comprime affatto: consegna un token per ogni
tessera, ed è il più leggero dei tre.
```

Come mostra {numref}`fig-vlm-connettori`, i blocchi alle estremità non cambiano
mai. Cambia il pezzo in mezzo, e cambiano con lui due grandezze: quanti
parametri si addestrano, che calano da sinistra a destra di quasi tre ordini di
grandezza, e se ci sia o no una compressione. Le prime due colonne comprimono e
la terza no, ed è quest'ultima distinzione, non il conto dei parametri, la
ragione per cui la sezione finisce come finisce.

### Aggiungere strati nuovi dentro il modello congelato

La prima risposta, quella di Flamingo {cite}`alayrac2022flamingo` nel 2022, non
mette il connettore *prima* del modello di linguaggio: lo mette **dentro**. Fra i
blocchi congelati si inseriscono strati nuovi di **cross-attention**, dove le
query vengono dal testo e le chiavi e i valori dall'immagine (il testo chiede,
l'immagine risponde), e sono questi strati (più il pezzo che prepara i vettori
visivi) l'unica cosa che si addestra. Encoder visivo e modello di linguaggio
restano entrambi fermi; gli strati nuovi entrano ogni quarto blocco nella
versione di Flamingo da nove miliardi di parametri, e ogni settimo in quella da
ottanta.

Aprire un modello addestrato e infilarci dentro strati inizializzati a caso è
però pericoloso: al primo passo quegli strati emettono rumore, il rumore si
somma alle attivazioni costruite in mesi di addestramento, e il comportamento
che si voleva conservare va in pezzi prima ancora di cominciare. La soluzione
è più povera di quanto il problema faccia temere (sta tutta in un numero
solo), e conviene guardarla da vicino.

`````{tab} Elementare

Un secondo microfono va aggiunto a un impianto audio già tarato bene. Acceso al
volume che capita, rovina il concerto. Si collega allora con il **volume a
zero**: l'impianto suona esattamente come prima, come se il microfono non ci
fosse. Poi la manopola si alza, se e quanto serve, e ha un fondo scala: per
quanto la si giri, il microfono nuovo non arriverà mai a coprire l'orchestra.

Ogni strato nuovo è collegato così: una sua manopola, un unico numero, moltiplica
tutto quello che lo strato produce prima di sommarlo al resto, e parte da zero.
Al primo istante l'immagine non influenza nulla e il modello si comporta
identico a com'era; poi l'addestramento scopre che alzarla conviene, perché
aiuta a indovinare le parole giuste, e la alza da sé.

Resta il problema di quante file di numeri consegnare: un'immagine ne dà
centinaia, un video ne dà centinaia per ogni fotogramma, e gli strati nuovi sono
fatti per riceverne sempre lo stesso numero, altrimenti andrebbero ridisegnati a
ogni cambio di formato. Il pezzo che se ne occupa ne fa uscire sempre 64,
qualunque sia la roba che entra. Le 64 righe da riempire ci sono già, fisse, e
sono le domande che il pezzo ha imparato a fare in addestramento: per ognuna
guarda tutto quello che è entrato, prende soprattutto da dove trova la risposta e
scrive lì il riassunto. Che il materiale sia poco o tanto non cambia il numero di
righe, cambia solo dove ciascuna va a pescare: è un modulo con 64 caselle, non un
imbuto tarato su una quantità.

Tutto questo, gli strati nuovi più il pezzo che riempie le 64 righe, pesa
**miliardi** di numeri da imparare: è un modello dentro il modello.

E un modulo lo si compila prima di sapere che cosa vi verrà cercato dentro.

`````

`````{tab} Superiore

Il meccanismo si chiama **tanh gating**. Ogni sotto-strato aggiunto entra nel
flusso residuale non come $\mathbf{x} \leftarrow \mathbf{x} + \mathrm{XAttn}(\mathbf{x}, \mathbf{R})$, ma come

$$
\mathbf{x} \;\leftarrow\; \mathbf{x} + \tanh(\alpha)\, \mathrm{XAttn}\big(\mathbf{x},\, \mathbf{R}\big),
$$

dove $\mathbf{x}$ sono le attivazioni del testo che attraversano il modello
congelato, $\mathbf{R}$ i vettori visivi già preparati dal modulo a monte,
$\mathrm{XAttn}$ la cross-attention (query da $\mathbf{x}$, chiavi e valori da
$\mathbf{R}$) e $\alpha$ uno scalare appreso, **uno per strato**,
inizializzato a zero. Poiché $\tanh(0) = 0$, alla prima iterazione ogni blocco
aggiunto è esattamente l'identità: la funzione calcolata dalla rete è, token per
token, quella del modello di partenza. L'inizializzazione non è quindi
«piccola», è **esatta**, e si parte da un punto di cui si conoscono le
prestazioni; $\tanh$ dà inoltre un gate limitato in $(-1, 1)$, che non fa
esplodere il ramo nuovo quando $\alpha$ cresce. Lo stesso schema avvolge il
blocco feed-forward che accompagna la cross-attention, da cui il nome *gated
cross-attention dense*.

A monte, il **Perceiver Resampler** risolve il problema del formato variabile.
È un modulo di attenzione con $K$ latenti appresi
$\mathbf{L} \in \mathbb{R}^{K \times d}$ (nel lavoro, $K = 64$) che fanno da query;
chiavi e valori vengono dalla concatenazione $[\mathbf{Z}; \mathbf{L}]$ delle feature visive,
appiattite in un'unica sequenza e già proiettate in $\mathbb{R}^{N \times d}$
(la concatenazione avviene lungo l'asse della sequenza, quindi la dimensione di
feature dev'essere la stessa dei latenti), con i latenti stessi, che quindi
attendono anche a sé: $\mathbf{R} = \mathrm{XAttn}(\mathbf{L}, [\mathbf{Z}; \mathbf{L}]) \in \mathbb{R}^{K \times d}$,
ripetuta per qualche strato con un feed-forward dopo ciascuno. Qualunque sia $N$ (una
sola immagine, oppure le feature spazio-temporali di un video) l'uscita ha
sempre $K$ righe, e il costo a valle diventa indipendente dalla risoluzione e
dalla durata.

Il conto dei parametri, però, è severo: gli strati aggiunti sono blocchi di
attenzione a dimensione piena distribuiti lungo tutta la pila, e nella variante
più grande la differenza fra il totale dichiarato (ottanta miliardi di
parametri) e il modello di linguaggio congelato su cui poggia (settanta) è
dell'ordine dei **dieci miliardi**. Il connettore, qui, è letteralmente un
modello dentro il modello.

`````

### Un questionario con domande fisse

La seconda risposta, il **Q-Former** di BLIP-2 {cite}`li2023blip2` nel 2023,
tiene l'idea della compressione e butta via quella degli strati inseriti nel
modello di linguaggio. Il connettore torna a essere un pezzo esterno, messo in
fila fra i due modelli congelati, e il suo compito è ridurre la griglia di
feature visive a un pugno di vettori che il modello di linguaggio riceve come
un normale prefisso di token.

`````{tab} Elementare

Un assistente, davanti a qualunque fotografia, compila sempre lo stesso
questionario di 32 domande. Le domande non gliele detta nessuno: se le è scritte
da solo in addestramento, tenendo quelle le cui risposte servivano di più a chi
poi doveva parlare dell'immagine, e guardandole tutte insieme perché non
finissero a chiedere due volte la stessa cosa. Potrebbero essere «che oggetti ci
sono» o «dove stanno l'uno rispetto all'altro», e cose senza nome che a noi non
verrebbero in mente.

Le ha scritte in due tempi. Prima con le sole foto davanti, correggendo le
domande finché dalle risposte si capiva quale fotografia stava guardando; solo
dopo le risposte sono andate a chi doveva scriverci sopra. Facendo tutto insieme
si otterrebbero 32 risposte gradevoli da leggere e senza rapporto con la foto.

Davanti a ogni nuova foto pone le 32 domande, la guarda per rispondere e consegna
32 risposte: da lì in poi il modello di linguaggio ha in mano solo quelle, la
foto non la vede più.

Il guadagno si legge nei numeri. L'encoder qui produce 257 file per ogni foto:
l'immagine è da 224 puntini di lato, le tessere da 14, quindi 16 per riga e 16
per colonna fanno 256 tessere, più una fila che riassume l'intera fotografia. (Su
un'immagine da 336 puntini, e senza contare la fila di riassunto, le tessere
erano 576: il numero cambia con la foto e con la tessera, non è mai fisso.) Da
257 si scende a 32, otto volte meno, e ciascuna risposta è anche più corta, 768
numeri invece dei 1024 con cui l'encoder descrive una tessera: in tutto passano
$32 \times 768$ numeri contro $257 \times 1024$, circa undici volte meno.

Il questionario lo compila una macchina sua, che pesa 188 milioni di caselle:
poca cosa accanto ai due modelli che collega.

Quelle 32 domande sono state fissate prima che qualcuno facesse una richiesta. Se
ne arriva una a cui non rispondono, non c'è più niente da rileggere.

`````

`````{tab} Superiore

Il Q-Former è un piccolo Transformer (inizializzato dai pesi di BERT-base, per
un totale di 188 milioni di parametri) che riceve un insieme di **query
apprese** $\mathbf{Q} \in \mathbb{R}^{M \times d_q}$, con $M = 32$ e $d_q = 768$. Le
query non dipendono dall'immagine: sono parametri del modello, come una matrice
di pesi. Dentro il blocco si alternano due interazioni: una **self-attention**
fra le query (che permette loro di specializzarsi e non chiedere tutte la stessa
cosa) e una **cross-attention** verso le feature congelate dell'immagine,
inserita ogni due blocchi, in cui l'immagine fornisce chiavi e valori.

$$
\mathbf{Z}_{\text{out}} = \mathrm{QFormer}_\theta\big(\mathbf{Q},\, \mathbf{Z}\big) \in \mathbb{R}^{32 \times 768},
\qquad
\mathbf{E} = \mathbf{Z}_{\text{out}} \mathbf{W}_{\text{proj}},
$$

dove $\mathbf{Z}$ sono le feature dell'encoder visivo, $\theta$ i parametri del Q-Former
e $\mathbf{W}_{\text{proj}} \in \mathbb{R}^{768 \times d_t}$ una proiezione lineare che
porta le uscite nella dimensione degli embedding del modello di linguaggio. Il
collo di bottiglia è dimensionato apposta: con un ViT-L/14 le feature visive
sono $257 \times 1024$, l'uscita è $32 \times 768$, cioè un fattore $10{,}7$ in
meno di numeri. Le query non possono portarsi dietro tutto, e sono costrette a
selezionare.

L'addestramento avviene in **due fasi**, e la prima serve a decidere che cosa
selezionare: il Q-Former è collegato al solo encoder visivo e ottimizza tre
obiettivi congiunti (contrastivo fra immagine e testo, generazione di testo
condizionata all'immagine, classificazione binaria di appaiamento). Solo nella
seconda l'uscita viene proiettata e data al modello di linguaggio congelato, con
la sola loss di modellazione del linguaggio. Senza la prima fase le query
imparerebbero a produrre qualcosa che il modello di linguaggio accetta, non
qualcosa che descrive l'immagine.

`````

### Una matrice, e basta

La terza risposta, LLaVA {cite}`liu2023visual` nello stesso 2023, è talmente
scarna che a raccontarla sembra mancare un pezzo. Niente compressione, niente
query apprese, niente strati nuovi. C'è una **matrice** (il pezzo, d'ora in
avanti, si chiama **proiettore**): la fila di numeri di ogni tessera viene
moltiplicata per quella matrice e diventa un token, e i token si mettono in fila
davanti al prompt come fossero parole.

`````{tab} Elementare

Una matrice è una tabella di conversione, e non fa niente di più misterioso di
una ricetta a dosi fisse: ogni numero che esce è una miscela sempre uguale dei
numeri che entrano, con le dosi decise una volta per tutte durante
l'addestramento. Qui la lista che entra è la descrizione di una tessera come
l'ha prodotta
l'encoder (1024 numeri), quella che esce è un token nel formato che il modello
di linguaggio si aspetta (4096 numeri). Una tessera entra, un token esce:
nessuna selezione, nessun riassunto, nessuna domanda decisa in anticipo.

Quanto costa una tabella del genere? Ha una casella per ogni coppia
«numero in ingresso, numero in uscita»: $1024 \times 4096$, cioè poco più di
quattro milioni di caselle. Il modello di linguaggio a cui si salda ha sette
miliardi di parametri, quindi la saldatura pesa lo $0{,}06\%$ del pezzo che
collega, sei centesimi di punto percentuale. Una versione successiva dello
stesso lavoro mette due tabelle in fila invece di una, e guarda le immagini da
$336$ puntini invece che da $224$: è quella da 576 tessere. La prima
tabella resta quella di prima, da $1024 \times 4096$; la seconda parte da un
token già tradotto, quindi è da $4096 \times 4096$, quattro volte più grande, e
in tutto fanno ventuno milioni di caselle, lo $0{,}3\%$. Sempre un'inezia, ma
cinque volte l'inezia di prima.

`````

`````{tab} Superiore

$$
\mathbf{H}_v = \mathbf{Z}_v \mathbf{W},
\qquad
\mathbf{W} \in \mathbb{R}^{d_v \times d_t},
$$

dove $\mathbf{Z}_v \in \mathbb{R}^{N \times d_v}$ sono le feature dell'encoder visivo
congelato, $\mathbf{W}$ è l'unico parametro addestrato nella prima fase e
$\mathbf{H}_v \in \mathbb{R}^{N \times d_t}$ sono i token visivi, che vivono nello stesso
spazio degli embedding di parola. La mappa è lineare e applicata patch per
patch: nessuna interazione fra le righe, nessuna riduzione di $N$.

Con $d_v = 1024$ (un ViT-L/14) e $d_t = 4096$ (un modello di linguaggio da sette
miliardi di parametri), $\mathbf{W}$ ha $1024 \times 4096 \approx 4{,}2$ milioni
di parametri, cioè lo 0,06% del modello che serve. Una versione successiva del
lavoro sostituisce la mappa lineare con un percettrone a due strati (`Linear`
$\to$ GELU $\to$ `Linear`), che porta il connettore a circa 21 milioni di
parametri, cioè lo $0{,}3\%$ del totale.

`````

Contro i miliardi della prima risposta e i 188 milioni della seconda siamo a un
altro ordine di grandezza, ed è questa terza la strada che il grosso dei sistemi
ha finito per prendere.

## Perché ha vinto il più semplice

Se il questionario è più sofisticato, e gli strati con la manopola del volume
sono più eleganti e più rispettosi del modello congelato, perché la strada che
si è imposta è quella della matrice? La risposta non è estetica, e non è nemmeno «perché costa meno
addestrarla». È che gli altri due connettori fanno una cosa che sembrava un
pregio ed è un difetto: **comprimono**. E comprimere significa scegliere che
cosa dell'immagine conta, **prima** di sapere quale sarà la domanda.

`````{tab} Elementare

Un collega ti prepara le carte per una riunione. Nella versione efficiente ti
legge un fascicolo di quaranta pagine e ti consegna dieci righe di riassunto:
veloce, comodo, quasi sempre sufficiente. Nella versione inefficiente ti mette
il fascicolo intero sulla scrivania e ti lascia sfogliarlo.

Finché in riunione ti chiedono quello che il collega si aspettava, il riassunto
vince a mani basse. Il giorno in cui qualcuno chiede il numero scritto in una
nota a piè di pagina, a pagina dodici, il riassunto non solo non lo contiene:
non c'è modo di andarlo a prendere, perché il fascicolo il collega se l'è
portato via.

I connettori che comprimono sono il collega efficiente, e il riassunto lo
scrivono sempre uguale, prima di sentire la domanda. Il proiettore è il
fascicolo lasciato sulla scrivania: costa contesto (576 tessere occupano posto e
tempo di calcolo) ma non butta via niente, e la selezione la fa l'attenzione del
modello di linguaggio, quando la domanda è già arrivata.

Il fascicolo sulla scrivania, però, regge finché sono quaranta pagine. Se ne
fossero quattromila (una fotografia enorme, oppure un'ora di video, un fotogramma
dopo l'altro), sfogliarle tutte a ogni domanda non si potrebbe, e il collega che
riassume tornerebbe ad avere ragione.

Ne esce una regola generale: **quando il collo di bottiglia è l'informazione, e
non il calcolo, conviene rimandare la selezione al momento in cui si conosce la
domanda.**

`````

`````{tab} Superiore

Il punto si formula bene in termini di condizionamento. Un connettore con
$M \ll N$ è un canale a capacità fissa, e la funzione $g_\theta$ che decide che
cosa passa viene appresa **marginalizzando** sulla distribuzione dei compiti
visti in addestramento: produce il riassunto ottimo *in media*. All'inferenza
però il compito non è più una variabile aleatoria, è la domanda che l'utente ha
scritto, e il connettore non la vede: i token visivi si calcolano prima di
leggere il prompt (nel Q-Former per costruzione, dato che le query sono
parametri). L'informazione scartata è irrecuperabile, e lo è **prima** che il
condizionamento su cui conterebbe sia disponibile.

Con $M = N$ e una mappa iniettiva, invece, nessuna informazione viene scartata a
monte: la selezione è delegata all'attenzione del modello di linguaggio, che
opera con query derivate dal testo (quindi **condizionate alla domanda**) e
agisce a ogni strato e in ogni testa, non una volta sola. La compressione non
sparisce, cambia posto: da preprocessing fisso diventa attenzione dinamica. Il
principio generale è che quando il collo di bottiglia è **informativo** e non
computazionale, la selezione va rimandata al punto del sistema in cui è
disponibile il massimo condizionamento; è la stessa logica per cui il collo di
bottiglia del seq2seq classico è stato sciolto dall'attenzione di Bahdanau
invece che da un vettore di contesto più grande.

Due onestà, per non trasformare un'osservazione in un dogma. Il prezzo è
pesante: il termine quadratico calcolato sopra diventa proibitivo su immagini ad
alta risoluzione, su documenti e sui video, dove il collo di bottiglia torna a
essere *computazionale* e la compressione torna sensata (è il tema della sezione
sulla risoluzione). E le query apprese non sono un'idea sbagliata: sono un'idea
con un dominio di validità, e riappaiono proprio dove i token visivi sarebbero
troppi.

`````

## Due tempi, in quest'ordine

Resta da dire come si addestra la saldatura: in due tempi, con due obiettivi
diversi e due insiemi di pesi congelati diversi. L'ordine non è negoziabile.

**Primo tempo, allineamento.** Si congela tutto e si addestra il solo connettore
su coppie immagine-didascalia, con l'obiettivo di sempre di un modello di
linguaggio: data l'immagine, scrivi la sua didascalia, una parola dopo l'altra.
Nel primo lavoro su LLaVA questa fase usa
595 000 coppie filtrate da un grande corpus di immagini e testi del web. Non si
insegna niente di nuovo ai due modelli: si insegna al connettore dove scrivere.

**Secondo tempo, istruzioni visive.** Si scongela anche il modello di
linguaggio (l'encoder visivo di norma resta fermo) e si continua su **dialoghi
che riguardano immagini**. È l'instruction tuning descritto nel capitolo sui
Transformer, applicato a un modello che adesso ha un occhio, e non lo
rispieghiamo qui. Vale anche in questa versione la cosa che là sorprende di più:
rispetto al pre-addestramento serve pochissimo materiale.

`````{tab} Elementare

Perché due fasi e non una sola? Perché sono due lezioni diverse, e mescolarle
significa non impararne bene nessuna.

La prima è di traduzione pura: il connettore deve capire dove mettere le cose.
Se qui lasciassi libero anche il modello di linguaggio, quello si adatterebbe ai
vettori sgangherati che il connettore gli manda all'inizio; e allora il
connettore non avrebbe mai il motivo di mandarli fatti bene. È come insegnare a
un principiante lasciando che sia l'orchestra ad aggiustarsi sui suoi errori.

La seconda è di comportamento: rispondere alla domanda, e non limitarsi a
descrivere la foto. Qui il modello di linguaggio deve poter cambiare, perché il
compito non è più quello su cui era stato addestrato.

`````

`````{tab} Superiore

Le due fasi ottimizzano la stessa forma di loss, la cross-entropia
autoregressiva sui token della risposta,

$$
\mathcal{L}(\theta) = - \sum_{t} \log p_\theta\big(y_t \mid y_{<t},\, \mathbf{H}_v,\, \mathbf{x}\big),
$$

dove $y_t$ è il token da produrre, $\mathbf{H}_v$ sono i token visivi e $\mathbf{x}$ il prompt
testuale, e cambiano soltanto per (a) quali parametri stanno dentro $\theta$ e
(b) come sono fatti i dati.

Nella prima fase $\theta$ contiene i soli parametri del connettore e i dati sono
coppie immagine-didascalia; il compito è quasi geometrico, portare le feature
visive nella regione dello spazio di embedding che il modello di linguaggio sa
già leggere. Nella seconda $\theta$ include i pesi del modello di linguaggio e i
dati sono conversazioni multi-turno; la loss è mascherata sui token
dell'istruzione e su quelli visivi, cioè il modello li legge ma non viene
penalizzato per non saperli generare. Fondere le due fasi porta il modello di
linguaggio ad adattarsi a un connettore non ancora allineato, con il rischio
classico dell'ottimizzazione congiunta di due componenti mal condizionate.

`````

Un dettaglio metodologico di questa seconda fase merita di essere raccontato,
perché è interessante e perché ha un limite che si vede a occhio nudo. I dati di
istruzione visiva del primo LLaVA non sono stati scritti da persone davanti a
delle fotografie: sono stati **generati da un modello di solo testo**, a cui delle immagini si
davano soltanto due sostituti scritti: le didascalie già disponibili e le
coordinate dei riquadri degli oggetti annotati. Da quel
materiale uscivano conversazioni, descrizioni dettagliate e domande di
ragionamento, per un totale di 158 000 esempi (58 000 dialoghi, 23 000
descrizioni, 77 000 ragionamenti).

È un caso di **dati sintetici** che ha funzionato, e conviene essere precisi sul
perché: il compito non era procurarsi conoscenza nuova (quella stava già nelle
annotazioni) ma un **formato**, insegnare che a una domanda si risponde. Il
limite sta in una frase: il generatore l'immagine non l'ha mai vista. Quello che
didascalie e riquadri non dicono non finisce nei dati, e quello che il
generatore inventa dentro un ragionamento plausibile ci finisce come se fosse
vero. Chi studia quel materiale ne eredita lo stile, e con lo stile anche la
sicurezza con cui il generatore afferma cose che non poteva sapere. La sezione
sull'allucinazione visiva ci tornerà sopra; qui basti annotare che una parte del
difetto nasce in addestramento, e non nel momento in cui il modello risponde.

## Il connettore in dieci righe

Tradotto in PyTorch, il proiettore è quello che promette di essere: due strati
lineari con una non linearità in mezzo.

```python
import torch
from torch import nn


class Proiettore(nn.Module):
    """Porta le feature dell'encoder visivo nello spazio dei token del testo."""

    def __init__(self, d_visione: int, d_testo: int):
        super().__init__()
        self.rete = nn.Sequential(
            nn.Linear(d_visione, d_testo),
            nn.GELU(),
            nn.Linear(d_testo, d_testo),
        )

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        # patch: (B, N, d_visione) -> (B, N, d_testo). Una patch, un token.
        return self.rete(patch)


proiettore = Proiettore(d_visione=1024, d_testo=4096)
print(sum(p.numel() for p in proiettore.parameters()))  # 20979712
```

Il pezzo che conta davvero, però, non è la classe: è come i token visivi
raggiungono il decoder. Non passano da una porta di servizio, entrano dalla
stessa porta delle parole.

```{code-block} python
:class: pt-non-eseguibile

# encoder e llm sono pre-addestrati e congelati; si addestra solo il proiettore
for modulo in (encoder, llm):
    for p in modulo.parameters():
        p.requires_grad = False

with torch.no_grad():
    patch = encoder(immagine)          # (B, 576, 1024): la griglia di feature

token_visivi = proiettore(patch)       # (B, 576, 4096): ora sono "parole"

# llm è un modello causale della libreria transformers: espone la tabella
# degli embedding e accetta gli embedding già calcolati al posto degli id
tabella = llm.get_input_embeddings()
prefisso = tabella(id_prima)           # (B, T1, 4096) il testo che precede
suffisso = tabella(id_dopo)            # (B, T2, 4096) la domanda vera e propria

# la sequenza che entra nel decoder: testo, immagine, testo. Tutto insieme.
ingresso = torch.cat([prefisso, token_visivi, suffisso], dim=1)

uscita = llm(inputs_embeds=ingresso, attention_mask=maschera, labels=etichette)
```

Due osservazioni. La prima è che il modello di linguaggio non viene modificato in
nessun punto: gli si passa `inputs_embeds` invece degli identificativi dei token,
e da lì in poi non sa, e non ha bisogno di sapere, che 576 delle sue posizioni
vengono da una fotografia. La seconda riguarda `etichette`: nelle posizioni
visive e in quelle dell'istruzione va messo il valore che segnala «ignora» (in
PyTorch, `-100`), perché quei token il modello deve leggerli senza essere
penalizzato per non saperli generare.

## Dove porta questa strada, e dove si ferma

Il connettore risolve il problema che aveva chiuso la sezione precedente: un
modello che *legge* l'immagine token per token, invece di comprimerla in un
vettore solo, può parlare delle relazioni fra le cose e non solo del loro
elenco, e lo fa senza riaddestrare niente di grosso.

Restano due domande, e sono le prossime due sezioni. Se l'immagine entra dalla
stessa porta delle parole, perché non farne davvero delle parole, simboli di un
vocabolario, così che il modello possa anche *scriverne*? E poi: 576 token
bastano per riconoscere una scena e non per leggere una tabella stampata dentro
la fotografia, ma
moltiplicarli significa pagare il fattore quadratico calcolato all'inizio. Il
conto del dettaglio è il vero limite pratico di tutto quello che abbiamo visto
qui.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Il connettore nasce da un problema di soldi: i modelli che sanno guardare una
  foto (la tagliano in tessere e descrivono ogni tessera con una lista di
  numeri) e quelli che sanno scrivere esistono già, e dietro ciascuno ci sono
  mesi di calcolo, quindi si cerca **il pezzo più piccolo da addestrare** perché
  comincino a parlarsi.
  Tenere fermi i due modelli è anche una garanzia: riaddestrarli farebbe loro
  dimenticare per strada una parte di quello che sapevano fare.
- Al pezzo in mezzo si chiedono due cose insieme: **tradurre** la descrizione di
  una tessera d'immagine nel formato che il modello di linguaggio si aspetta, e
  **decidere quante** tessere consegnargli. La seconda pesa quanto la prima: ogni
  tessera occupa posto come una parola, e il lavoro dell'attenzione cresce con
  il quadrato dei pezzi messi in fila.
- **Strati nuovi dentro il modello congelato** {cite}`alayrac2022flamingo`: si
  inseriscono strati in cui il testo chiede e l'immagine risponde, collegati con
  una manopola del volume che parte da zero, così al primo istante il modello
  suona esattamente come prima; davanti a loro un pezzo riduce qualunque
  immagine (o video) a 64 vettori sempre, un modulo con 64 righe da compilare
  prima di sapere che cosa vi verrà cercato dentro.
- **Questionario fisso** {cite}`li2023blip2`: 32 domande scritte una volta per
  tutte in addestramento vengono poste a ogni foto, e ne escono 32 risposte: dai
  257 vettori con cui la foto era stata descritta si scende a 32, e siccome ogni
  risposta è anche un po’ più corta, nel punto più stretto passano circa undici
  volte meno numeri. **Tabella di conversione**
  {cite}`liu2023visual`: una tessera entra, un token esce, nessun riassunto
  (circa quattro milioni di caselle, poi una ventina di milioni con due tabelle
  in fila).
- Ha prevalso il più semplice, e la ragione è di principio: **riassumere vuol
  dire scegliere prima di sapere qual è la domanda**. Meglio il fascicolo intero
  lasciato sulla scrivania, finché lo si può sfogliare: si paga in posto
  occupato, ma a scegliere è il modello di linguaggio, quando la domanda è già
  arrivata.
- L'addestramento è in **due tempi**: prima il solo connettore su coppie
  immagine-didascalia (imparare dove scrivere), poi dialoghi sulle immagini, con
  il modello di linguaggio libero di cambiare. I dialoghi del primo LLaVA li ha
  scritti un modello di solo testo, che le foto non le aveva mai viste: materiale
  inventato che ha funzionato, ma che passa anche i difetti di chi l'ha scritto.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Il connettore nasce da una domanda economica: encoder visivi e modelli di
  linguaggio esistono già e costano milioni, quindi si cerca **il pezzo più
  piccolo addestrabile** che li faccia parlare. Congelare protegge anche dalla
  dimenticanza catastrofica.
- Deve fare due cose insieme: **cambiare spazio** (da $d_v$ a $d_t$) e
  **decidere quanti token visivi** entrano nel contesto. Il secondo non è un
  dettaglio: il costo dell'attenzione cresce con il quadrato della lunghezza
  della sequenza.
- **Cross-attention gated** {cite}`alayrac2022flamingo`: strati nuovi inseriti
  fra i blocchi congelati, con un gate $\tanh(\alpha)$ e $\alpha$ inizializzato
  a zero, così all'inizio il modello è *esattamente* quello di prima; un
  Perceiver Resampler porta un numero variabile di feature a 64 token fissi.
  Sono gli strati aggiunti a costare miliardi di parametri, non il resampler.
- **Q-Former** {cite}`li2023blip2`: 32 query apprese interrogano l'immagine in
  cross-attention e ne estraggono 32 vettori (188 milioni di parametri, due fasi
  di addestramento). **Proiettore** {cite}`liu2023visual`: una matrice, poi un
  MLP a due strati, e una patch resta un token (4 milioni di parametri la
  sola matrice, 21 con l'MLP).
- Ha prevalso il più semplice, e la ragione è di principio: **comprimere
  significa scegliere prima di conoscere la domanda**. Quando il collo di
  bottiglia è l'informazione e non il calcolo, conviene rimandare la selezione
  al punto in cui il condizionamento è massimo, cioè all'attenzione del modello
  di linguaggio. Il prezzo è il contesto occupato, e su immagini ad alta
  risoluzione, documenti e video quel collo di bottiglia torna a essere
  computazionale: lì la compressione ha di nuovo senso.
- L'addestramento è in **due tempi**: prima il solo connettore su coppie
  immagine-didascalia, poi instruction tuning visivo con il modello di
  linguaggio scongelato. I dati di istruzione del primo LLaVA furono generati da
  un modello di solo testo a partire da didascalie e riquadri: dati sintetici
  che funzionano, ma che trasmettono anche i difetti del generatore.
```

`````
