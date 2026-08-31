# La struttura del Transformer

Il meccanismo di attenzione è il motore; adesso montiamo l'automobile. Il
Transformer descritto in *Attention Is All You Need*, l'articolo del 2017 da
cui questo capitolo è partito, è una macchina per tradurre: da un lato entra
una frase ("The black cat jumps on the wall"), dall'altro esce la traduzione
("Il gatto nero salta sul muro"). Per farlo combina due torri di blocchi
identici (l’**encoder** che legge, il **decoder** che scrive), più un
ingrediente facile da sottovalutare: un modo per dire alla rete *in che ordine*
stanno le parole.

```{figure} ../figures/architettura-transformer.svg
:name: fig-blocco-transformer
:alt: "Schema annotato di un blocco Transformer: l'ingresso passa per una normalizzazione, entra nella multi-head attention e si somma alla copia di sé stesso arrivata dalla connessione residua; il risultato passa per una seconda normalizzazione, attraversa la rete feed-forward e si somma di nuovo a sé stesso, prima di uscire verso il blocco successivo."
:width: 62%

Il blocco che si ripete, sempre uguale a sé stesso. I due mestieri sono la
riunione (l'attenzione, dove le parole si scambiano informazioni) e il lavoro
individuale, dove ogni parola rielabora per conto suo; attorno a entrambi c'è
l'impalcatura della sezione precedente, cioè la scorciatoia e la taratura, che
permette di impilare decine di blocchi senza che i primi smettano di imparare.
Il disegno mette la taratura **all'ingresso** di ciascuno dei due mestieri, che
è il montaggio dei modelli di oggi; quello del 2017 la metteva subito dopo la
somma.
```

Conviene fissare {numref}`fig-blocco-transformer` prima di scendere nei
dettagli, perché tutto il capitolo gira attorno a questa figura: encoder e
decoder sono due pile dello stesso blocco, montate in modo leggermente
diverso.

Una parola serve prima di cominciare. Dentro la rete ogni parola è diventata
una lista di numeri, e quella lista si chiama la sua **rappresentazione**: è
quello che il modello ha capito della parola finora, non la parola scritta, e
cambia a ogni piano. Tutto il lavoro delle due torri consiste nel riscriverla.

## L'encoder: la torre che legge

Cominciamo dalla torre che legge, perché è la più semplice delle due: fa una
cosa sola, prendere la frase di partenza e capirla il meglio possibile.

`````{tab} Elementare
L'encoder è una pila di sei piani identici, e a ogni piano c'è un lettore che
rilegge tutta la frase. Al primo piano le parole arrivano "grezze", cioè con la
rappresentazione che avevano da sole, fuori da qualunque frase: "nero" vale
"nero" e basta. Ogni piano la rilegge con il meccanismo di attenzione (ogni
parola guarda tutte le altre e si arricchisce di quello che ha visto), poi
ciascuna parola viene rielaborata per conto suo da una piccola rete di neuroni,
e il risultato sale al piano di sopra. Piano dopo piano la rappresentazione di
ogni parola si specializza: "nero" al sesto piano è diventato
*il colore di quel gatto in quella frase*. Alla fine della salita, l'encoder
consegna una versione della frase in cui ogni parola porta scritto addosso il
proprio contesto.

Perché sei piani e non quattro? Come per le otto teste della sezione
precedente, perché funzionava: è un numero provato sul campo, e i modelli
venuti dopo sono arrivati a decine e centinaia di piani.
`````

`````{tab} Superiore
L'encoder è una pila di $L = 6$ strati identici (nel modello base,
$d_{\text{model}} = 512$; l'articolo del 2017 chiama $N$ questo numero, ma qui
si usa $L$ come nel resto del libro, dove $N$ serve ad altro), ciascuno con due
sotto-strati:

1. **Multi-Head Self-Attention**: ogni posizione attende a tutte le posizioni
   dell'input, catturando le relazioni a coppie in un solo passo;
2. **Feed-Forward Network (FFN)**: una rete completamente connessa applicata
   *indipendentemente e identicamente* a ogni posizione.

Ogni sotto-strato è avvolto da residual connection e layer normalization nella
forma Post-LN dell'articolo originale,
$\text{LayerNorm}(\mathbf{x} + \text{SubLayer}(\mathbf{x}))$, come visto nella
sezione
precedente (dove si è detto anche perché i modelli successivi preferiscono il
Pre-LN). Si noti la divisione dei ruoli: l'attenzione *mescola*
informazione tra le posizioni, la FFN la *trasforma* posizione per posizione;
è l'alternanza dei due movimenti, ripetuta per $L$ strati, a costruire
rappresentazioni via via più astratte.
`````

## Il decoder: la torre che scrive

Quello che esce dalla torre che legge, però, è solo una frase capita bene, non
ancora una traduzione. A trasformarla in un'altra frase ci pensa la seconda
torre, ed è la più delicata delle due, perché deve scrivere.

`````{tab} Elementare
Il decoder genera la traduzione una parola alla volta, e mentre lo fa consulta
due fonti: quello che ha *già scritto* (per non contraddirsi) e quello che
l'encoder *ha letto* (per restare fedele all'originale). C'è però una regola
ferrea, la stessa dei compiti in classe: **non si sbircia avanti**. Per capirla
serve sapere come si addestra questa macchina, che è più semplice di quanto
sembri: le si danno milioni di frasi con accanto la traduzione giusta, scritta
da una persona, e la si costringe a indovinarla parola per parola, controllando
ogni volta quanto ci è andata vicina. La traduzione giusta, insomma, durante lo
studio ce l'ha davvero sotto gli occhi. Ed è proprio per questo che le si copre:
quando impara a produrre la quarta parola può guardare solo le prime tre, altrimenti
"imparerebbe" a copiare la quarta dalla soluzione, e il giorno in cui la
soluzione non c'è (cioè sempre, una volta finito lo studio) non saprebbe fare
nulla.

Una parola alla volta vuol dire proprio una. A ogni passo la macchina
distribuisce la propria fiducia su tutte le parole che conosce, decine di
migliaia, un po’ a questa e un po’ a quella; poi da quell'elenco ne prende una
sola. È la parola presa a rientrare al passo dopo, in coda a quelle già
scritte, perché l'ingresso è fatto per parole e un elenco di fiducie non ci
passerebbe. Durante lo studio la parola che rientra è quella vera della
soluzione, e non quella che la macchina avrebbe detto: così un inciampo alla
terza parola non le fa sbagliare anche tutto il resto della frase. Il futuro,
quello, resta coperto come prima.
`````

`````{tab} Superiore
Anche il decoder ha $L = 6$ strati, ma con tre sotto-strati ciascuno:

1. **Masked Multi-Head Self-Attention**: come la self-attention dell'encoder,
   ma con una maschera che azzera (pone a $-\infty$ prima della softmax) le
   affinità verso le posizioni future: la posizione $t$ vede solo
   $1, \dots, t$. È ciò che rende il modello **autoregressivo**, e fa sì che il
   *meccanismo* di condizionamento sia lo stesso in addestramento e in
   generazione;
2. **Cross-Attention**: le query vengono dal decoder, key e value dall'output
   dell'encoder, è qui che la generazione "consulta" la frase di partenza;
3. **Feed-Forward Network**, identica a quella dell'encoder.

In generazione il decoder produce un token alla volta: a valle della pila, una
proiezione lineare sul vocabolario (nel paper con i pesi legati a quelli
dell'embedding, §3.4) e una softmax danno la distribuzione del token
successivo. Da quella distribuzione si sceglie un token, ed è il suo
embedding a rientrare come input al passo dopo: non la distribuzione, che è un
vettore di $|\mathcal{V}|$ probabilità e non ha modo di entrare in un ingresso
fatto per un token. Come si sceglie a generazione (il più probabile, oppure uno
estratto a sorte) è una questione a sé, e la sezione sui grandi modelli
linguistici la affronta per intero. In addestramento entra invece il token
**vero**, ed è il *teacher forcing*. La maschera garantisce che il meccanismo
sia lo stesso nei due casi, ma i prefissi su cui il decoder viene interrogato
no: in addestramento sono quelli del riferimento, in generazione i propri, ed è
l’*exposure bias* che la sezione
[con la soluzione accanto](../NaturalLanguageProcessing/seq2seq-traduzione.md)
ha raccontato.
`````

```{figure} ../figures/attenzione-mascherata.gif
:name: fig-attenzione-mascherata
:alt: Animazione di una matrice di attenzione 6x6 sulla frase «Il gatto nero salta sul muro». Prima tutte le celle si riempiono di punteggi grigi; poi una scala separa il triangolo superiore, che si spegne perché posto a meno infinito; infine il triangolo inferiore si ricolora con i pesi normalizzati dalla softmax.
:width: 85%

Il divieto di sbirciare avanti, al lavoro: si chiama *maschera causale*, dove
«causale» vuol dire che nessuna parola dipende da ciò che viene dopo. Le
caselle verso il futuro si spengono, e ogni riga ridistribuisce tutto il colore
dell'evidenziatore su ciò che precede.
```

La griglia di {numref}`fig-attenzione-mascherata` è l'evidenziatore della
sezione precedente messo in tabella. Righe e colonne sono le parole nell'ordine
in cui stanno nella frase: una riga per ogni parola che guarda, una colonna per
ogni parola guardata, e in ogni casella l'intensità di colore che la prima dà
alla seconda. La casella dove riga e colonna portano lo stesso nome (la
diagonale) è la parola che guarda sé stessa, e resta accesa; le caselle alla
sua destra sono le parole che vengono dopo, cioè il futuro, e sono quelle da
spegnere. Le intensità di una riga sommano sempre a uno, perché ogni parola ha
esattamente una unità di colore da distribuire.

Il dettaglio da non perdere è **quando** si spengono, ed è il passaggio in due
tempi visto nella sezione precedente: prima si calcolano i punteggi, poi la
softmax li trasforma in intensità che sommano a uno. Spegnere dopo lascerebbe
righe che non sommano più a uno, cioè evidenziature con un pezzo di colore
mancante e una parola che pesa meno delle altre senza motivo. Si spegne quindi
prima, sui punteggi, e allora è la softmax stessa a ridistribuire sul passato
tutto il colore che sarebbe andato al futuro. Il modo di spegnerli è elegante:
al posto del punteggio si mette meno infinito, e siccome $e$ elevato a meno
infinito fa zero, dalla softmax quelle caselle escono come zeri esatti.

## Positional encoding: dare un ordine alle parole

C'è un problema nascosto. L'attenzione tratta la frase come un *sacchetto* di
parole: se mescolassi "il gatto morde il cane" in "il cane morde il gatto", i
confronti sarebbero gli stessi fra le stesse parole, quindi gli stessi
punteggi e la stessa evidenziatura. Per l'attenzione le due frasi sono
identiche; per chi legge sono opposte. Le reti che leggevano in fila l'ordine
ce l'avevano gratis; il Transformer deve aggiungerlo apposta.

Il rimedio è quello che si fa a teatro: dare a ogni parola un **posto
numerato**. Prima che entri nella rete, alla sua lista di numeri se ne somma
un'altra che dice «io sono la parola in prima posizione», «in seconda», e così
via, e da lì in avanti "gatto" in prima posizione e "gatto" in quinta non sono
più identici. Resta da decidere come si scrive quel numero di posto, ed è la
parte inaspettatamente interessante.

```{figure} ../figures/positional-encoding.svg
:name: fig-positional-encoding
:alt: "Più onde sinusoidali sovrapposte, di frequenza decrescente: le prime oscillano rapidamente, le ultime lentamente. Letta in verticale a una data posizione, la combinazione dei valori delle diverse onde forma la firma numerica di quella posizione."
:width: 88%

Le frequenze del positional encoding. Nessuna onda da sola dice dove siamo: la
firma di una posizione è la fila verticale dei punti che tutte le onde toccano
lì (qui il disegno ne mostra tre, in un modello vero sono centinaia). Con tre
sole onde, e così lente, la fila tornerebbe uguale ogni dieci posizioni; nel
modello vero la più lenta compie un giro in oltre sessantamila, e due posizioni
non ricevono mai la stessa firma.
```

La firma della posizione c'è, ma non è scritta come un semplice contatore (1,
2, 3, …), ed è quello che mostra {numref}`fig-positional-encoding`. Il
contatore, in effetti, sarebbe la prima idea di chiunque, e ha due difetti
concreti. Il primo è che cresce senza fermarsi: la parola numero
diecimila porterebbe addosso il numero diecimila, e una rete davanti a un
ingresso mille volte più grande di tutti gli altri va in tilt. Il secondo è che
normalizzarlo non aiuta: se per tenerlo piccolo si divide per la lunghezza
della frase, «metà frase» diventa 0,5 sia in una frase di sei parole sia in una
di seicento, e la stessa firma finisce a significare due cose diverse.

La soluzione del 2017 tiene insieme le due esigenze con un'idea sola:
**lancette**. Immagina più orologi affiancati, uno veloce, uno medio, uno lento.
Nessuna lancetta si allontana mai, perché gira e torna: qualunque posizione
della frase, il numero che se ne legge resta sempre nella stessa fascia. E la
lancetta veloce distingue i vicini immediati, quella lenta dice in quale parte
della frase siamo: due scale insieme invece di una.

Il legame con la figura è che l'altezza della punta di una lancetta, disegnata
mano a mano che l'orologio avanza, è proprio un’**onda** che sale e scende: le
tre curve del disegno sono tre lancette a tre velocità. La firma di una
posizione è allora la fila verticale dei tre punti che le tre onde toccano lì.

Una lancetta sola, certo, si ripete: alle tre di notte e alle tre di pomeriggio
la lancetta delle ore sta nello stesso posto. Su un orologio da parete si ripete
anche la fila intera delle tre, ogni dodici ore, perché lì le velocità sono
l'una multipla dell'altra. Le velocità delle onde stanno molto più lontane fra
loro, e la più lenta impiega oltre sessantamila posizioni a compiere un giro:
nessuna frase è abbastanza lunga perché la fila dei punti torni uguale.

`````{tab} Elementare
Il posto numerato, dunque, c'è, ma il numero è scritto con le lancette invece
che in cifre. La sostanza però è quella del teatro: stessa parola, poltrona
diversa, e la rete può accorgersi che l'ordine conta. Il modo in cui la firma
viene consegnata è sbrigativo: non si aggiunge un pezzo in fondo alla lista
della parola, si **somma** numero per numero alla lista che c'è già. Parola e
posizione finiscono mescolate negli stessi numeri, e alla rete tocca imparare a
distinguerle.

Le lancette hanno poi una comodità che un contatore non ha. Andare avanti di
tre parole è sempre lo stesso gesto, in qualunque punto della frase lo si
faccia: ogni lancetta scatta di un suo angolo fisso, quella veloce di parecchio,
quella lenta di pochissimo, e quanto scatta dipende dal tre e non da dove si era
prima. Chi legge le firme ha quindi un modo di riconoscere «tre parole più in
là» che funziona uguale all'inizio e alla fine della frase; col contatore
diviso per la lunghezza, invece, tre parole più in là valgono mezzo passo in
una frase di sei parole e mezzo centesimo in una di seicento.

Nel 2017 quelle firme erano calcolate a tavolino, con una formula scritta a mano
prima di cominciare: il modello non le impara, se le trova già pronte. Che la
comodità delle lancette gli serva davvero resta però un sospetto, e sono stati
gli autori stessi a incrinarlo: hanno provato a lasciare che il modello si
costruisse le firme da solo, e le traduzioni sono venute quasi uguali. Infatti i modelli venuti dopo hanno preso strade diverse (chi le fa
imparare, chi scrive direttamente quanto due parole sono distanti invece di dove
stanno), ma il posto numerato, in una forma o nell'altra, serve a tutti.
`````

`````{tab} Superiore
Il **positional encoding** del paper originale è deterministico, fatto di
sinusoidi a frequenze diverse:

$$
\text{PE}_{(\text{pos},\, 2i)} = \sin(\text{pos}\;\omega_i)
\qquad
\text{PE}_{(\text{pos},\, 2i+1)} = \cos(\text{pos}\;\omega_i) ,
\qquad
\omega_i = 10000^{-2i/d_{\text{model}}}
$$

dove $\text{pos}$ è la posizione del token e $i$ *non* indicizza le
coordinate ma le **coppie** di coordinate, cioè le frequenze: $i$ va da $0$ a
$d_{\text{model}}/2 - 1$, e ogni $i$ riempie le due coordinate $2i$ e $2i+1$
con un seno e un coseno della stessa frequenza $\omega_i$. È il punto in cui si
sbaglia scrivendo il codice, perché il paper scrive «$i$ is the dimension» e
lascia credere che arrivi a $d_{\text{model}}$.

Ogni posizione riceve così una firma unica, sommata all'embedding del token; e
la scelta sinusoidale fa sì che la firma della posizione $\text{pos} + k$ sia
una trasformazione lineare di quella di $\text{pos}$ **con una matrice che
dipende solo da $k$**. La clausola in grassetto è tutto il contenuto: che due
vettori siano legati da *qualche* matrice è vero sempre e non dice niente; che
la matrice sia la stessa per ogni $\text{pos}$ è ciò che rende la distanza
relativa una cosa rappresentabile. Ed è una riga di trigonometria: dalle
formule di addizione,

$$
\begin{pmatrix} \sin((\text{pos}+k)\,\omega_i) \\ \cos((\text{pos}+k)\,\omega_i) \end{pmatrix}
=
\begin{pmatrix} \cos k\omega_i & \sin k\omega_i \\ -\sin k\omega_i & \cos k\omega_i \end{pmatrix}
\begin{pmatrix} \sin(\text{pos}\,\omega_i) \\ \cos(\text{pos}\,\omega_i) \end{pmatrix},
$$

cioè una rotazione di angolo $k\omega_i$ su ciascuna coppia, e $\text{pos}$ è
sparito dalla matrice. Da lì gli autori
*ipotizzarono* che la rete potesse rappresentare facilmente le distanze
*relative*: è una congettura, e la loro stessa ablazione la
indebolisce, perché con positional embedding **appresi** i risultati sono
«quasi identici» (Tabella 3, riga E, dell'articolo). Molti modelli successivi
usano infatti encoding **appresi** (BERT); i modelli linguistici recenti usano
quasi tutti la **RoPE** (*rotary position embedding*
{cite}`su2024roformer`), che prende quell'identità sul serio e la promuove da
congettura a costruzione. La RoPE non aggiunge niente all'embedding:
**ruota** query e chiave, coppia di coordinate per coppia, di un angolo
proporzionale alla posizione assoluta del token,
$\mathbf{q}_m \mapsto \mathbf{R}_{m}\,\mathbf{q}_m$ e
$\mathbf{k}_n \mapsto \mathbf{R}_{n}\,\mathbf{k}_n$; nel prodotto scalare le
due rotazioni si compongono, $\mathbf{R}_m^\top \mathbf{R}_n =
\mathbf{R}_{n-m}$, e ai punteggi di attenzione arriva soltanto la distanza
$n-m$. Ogni vettore è ruotato secondo la posizione assoluta in cui sta,
e l'attenzione vede solo le posizioni relative. Il principio, in ogni caso,
resta lo stesso: iniettare l'ordine, perché la self-attention da
sola è permutation-**equivariante**, cioè permutando i token in ingresso le
uscite escono permutate allo stesso modo e la rappresentazione di ogni parola
non dipende da dove sta.
`````

## La feed-forward network: il lavoro individuale

Manca un pezzo solo, ed è quello che nella {numref}`fig-blocco-transformer`
sta subito dopo l'attenzione. Ha un nome inglese, *feed-forward network*, che
vuol dire soltanto «rete che va in avanti», cioè senza cappi né ritorni:
i numeri entrano da una parte ed escono dall'altra.

`````{tab} Elementare
La riunione finisce e ognuno torna alla propria scrivania. Lì, da solo, rimette
in ordine quello che ha appena sentito, e lo fa in tre gesti, con due moduli
prestampati che sono gli stessi per tutti. Primo: la nota uscita dalla riunione,
512 righe di numeri, viene ricopiata su un modulo quattro volte più lungo, 2.048
righe. Ogni riga in più è una miscela diversa di quelle di partenza, e fa venire
fuori una combinazione che nella nota corta stava schiacciata insieme alle
altre. Secondo: si ripassa il foglio e si mette a zero ogni riga venuta
negativa. La riga resta dov'è, con uno zero sopra, e il foglio resta lungo
uguale; è l'unico momento in cui alla scrivania si sceglie invece di mescolare.
Terzo: un secondo modulo riporta il foglio alla lunghezza della nota di
partenza, e di tutto quel materiale largo tiene solo quello che serve al piano
di sopra. Riunione, scrivania, riunione, scrivania: la torre è tutta qui.

Quello che il modello ha imparato sta scritto nei numeri stampati sui moduli (si
chiamano **parametri**, ed è quello che si conta quando si dice «un modello da
sette miliardi»). In un piano della torre che legge, due terzi di quei numeri
stanno sui moduli della scrivania e un terzo su quelli della riunione.

Il conto è alla portata. La riunione usa quattro moduli grandi uguali: uno per
la query, uno per la key, uno per il value, uno per rimettere insieme le
risposte degli otto lettori. La scrivania ne usa due soli, ma ciascuno quattro
volte più grande, perché uno allarga e l'altro ricomprime: sono otto moduli
della prima taglia. Otto contro quattro, due terzi contro un terzo. Nei piani
della torre che scrive le riunioni sono due, la propria e la consultazione
dell'altra torre, quindi lì si va a otto contro otto e la quota scende a metà;
ma i grandi modelli linguistici di oggi tengono solo la torre che scrive e non
consultano nessuno, e tornano ai due terzi. Il gesto più semplice della torre è
anche quello dove il modello tiene quello che sa.

Tre gesti e due moduli: questo è il piano del 2017, e i modelli di oggi lo hanno
ritoccato in due punti. Le righe negative ora si scoloriscono invece di sparire
di colpo: quanto più erano negative, tanto più si avvicinano allo zero. E i
moduli sono passati da due a tre. Accanto al foglio lungo se ne compila un
secondo della stessa lunghezza, che riga per riga dice quanto quella riga conta,
una seconda scelta accanto a quella sui negativi. Perché il totale dei numeri
stampati non cresca, l'allargamento si accorcia da quattro volte a poco meno di
tre: tre moduli così fanno di nuovo otto, e il conto di prima resta in piedi.
Stessi numeri da regolare, stesso studio, e il modello riesce meglio: è la
ragione per cui alla scrivania oggi si compilano tre fogli.
`````

`````{tab} Superiore
La FFN applica a ogni posizione, separatamente e con gli stessi pesi, due
trasformazioni lineari con una ReLU in mezzo:

$$
\text{FFN}(\mathbf{x}) = \max(0,\, \mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\,
\mathbf{W}_2 + \mathbf{b}_2
$$

Nel modello base la dimensione interna è $d_{\text{ff}} = 2048$, quattro volte
$d_{\text{model}} = 512$: la FFN espande, applica la non linearità,
ricomprime. Pur essendo la parte concettualmente più semplice, contiene circa
due terzi dei parametri di uno strato di **encoder**
($2\,d\,d_{\text{ff}} = 8d^2$ contro i $4d^2$ delle quattro proiezioni
dell'attenzione); negli strati di
decoder, che hanno una seconda attenzione, la quota scende a metà. Nei grandi
modelli linguistici, che sono decoder-only e quindi senza cross-attention, si
torna ai due terzi, ed è lì che risiede gran parte della capacità. Una linea di
ricerca interpreta la FFN come una memoria associativa di conoscenze apprese
durante l'addestramento.

Questa è però la FFN del **paper originale**. I modelli successivi ne hanno
cambiato la non linearità: prima la **GELU**, una ReLU ammorbidita (BERT,
GPT-2), poi le varianti *gated* e in particolare **SwiGLU**, che è la scelta
prevalente nei modelli recenti:

$$
\text{FFN}_{\text{SwiGLU}}(\mathbf{x}) =
\big(\mathrm{Swish}(\mathbf{x}\mathbf{W}_1) \odot \mathbf{x}\mathbf{W}_3\big)\,
\mathbf{W}_2 ,
\qquad \mathrm{Swish}(z) = z\,\sigma(z).
$$

dove $\mathbf{W}_1, \mathbf{W}_3 \in \mathbb{R}^{d \times d_{\text{ff}}}$ e
$\mathbf{W}_2 \in \mathbb{R}^{d_{\text{ff}} \times d}$, $\odot$ è il prodotto
elemento per elemento, e nella Swish $z$ è un numero singolo (la funzione si
applica componente per componente) e $\sigma$ è la sigmoide, così che
$\mathrm{Swish}$ sia una ReLU ammorbidita: quasi $z$ per $z$ grande, quasi zero
per $z$ molto negativo. (Shazeer chiama $\mathbf{V}$ la matrice del cancello;
qui la lettera è già impegnata dai *value* dell'attenzione, e riusarla sarebbe
una trappola.)

Il ramo $\mathbf{x}\mathbf{W}_3$ fa da **cancello**: moltiplicando elemento per
elemento, decide
quanto lasciar passare di ciascuna unità del ramo principale. Le matrici
diventano tre invece di due, e per non gonfiare il conteggio dei parametri si
riduce la dimensione interna da $4d$ a circa $\tfrac{8}{3}d$: così
$3 \cdot d \cdot \tfrac{8}{3}d = 8d^2$, esattamente quanto $2 \cdot d \cdot 4d$
della versione classica. Stessi parametri, risultati migliori a parità di
addestramento.
`````

Con la feed-forward il giro è completo, e conviene guardare indietro un
momento. In questa pagina un solo ingrediente era nuovo davvero, il posto
numerato; tutti gli altri erano già sul tavolo. C'è l'attenzione della sezione
precedente, c'è una piccola rete di neuroni come quelle del capitolo sulle
reti neurali, ci sono una scorciatoia e una taratura attorno a ciascuna delle
due. Il Transformer è un modo di impilare quei quattro, sempre nello stesso ordine:
sei piani nel modello del 2017, qualche decina in quelli su cui si fanno i
conti oggi.

È il motivo per cui l'architettura ha retto senza cambiare forma mentre i
modelli diventavano quasi tremila volte più grandi: dai 65 milioni di parametri
del modello base del 2017 ai 175 miliardi di GPT-3, tre anni dopo. Non c'era
una forma da cambiare, c'era una sequenza da ripetere.

E il nome GPT-3 dice anche un'altra cosa, da anticipare perché altrimenti si
resta con l'idea che il Transformer sia una macchina per tradurre e basta.
Quella macchina non traduce, chiacchiera, perché tiene **solo la torre che
scrive** e butta via quella che legge: e con la torre che legge se ne va anche
il momento in cui il decoder la consultava, cioè dei tre pezzi di ogni suo
piano ne restano due. Quel che rimane, davanti a un pezzo di testo qualsiasi,
fa esattamente quello che sa fare, cioè continuarlo; e continuare un testo, se
il testo è una domanda, somiglia molto a rispondere. Le famiglie di modelli
che nascono da questa potatura sono l'argomento di una delle prossime sezioni.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Il Transformer originale è fatto di **due torri**: una legge la frase di
  partenza, l'altra scrive la traduzione. Sei piani ciascuna, tutti uguali.
- Ogni piano alterna una **riunione** (con l'attenzione, ogni parola ascolta
  tutte le altre; a condurla sono otto lettori in parallelo, ognuno attento a
  un tipo di legame) e un **lavoro individuale** (ogni parola rielabora per
  conto suo quello che ha sentito). Attorno a entrambi i momenti c'è
  l'impalcatura che permette di impilare tanti piani senza che
  l'addestramento si rompa.
- La torre che scrive ha una regola ferrea, **non si sbircia avanti**: mentre
  produce la quarta parola può guardare solo le prime tre. E a ogni passo
  consulta quello che l'altra torre ha capito della frase originale.
- L'attenzione da sola non sa in che ordine stanno le parole: a ciascuna viene
  sommato prima un **posto numerato**, la firma della sua posizione (calcolata
  a tavolino nel paper del 2017; i modelli successivi la fanno in altri modi,
  ma il posto numerato serve a tutti).
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Il Transformer originale è **encoder–decoder**: $L = 6$ strati per torre,
  $d_{\text{model}} = 512$, 8 teste di attenzione.
- Ogni strato alterna **attenzione** (le posizioni si scambiano informazione)
  e **FFN** (ogni posizione rielabora per conto suo), con residual e layer
  norm attorno a ogni sotto-strato.
- Il decoder usa la **maschera causale** (vietato guardare il futuro) e la
  **cross-attention** verso l'encoder.
- L'attenzione ignora l'ordine: il **positional encoding** (sinusoidale nel
  paper, appreso o relativo nei modelli successivi) lo reintroduce.
```
`````
