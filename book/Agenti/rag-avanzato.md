# RAG avanzato: oltre il recupero ingenuo

Ricapitoliamo in due righe di che cosa parliamo. **RAG** vuol dire questo:
prima di far rispondere il modello, si va a cercare in un archivio i pezzi di
testo che c'entrano con la domanda e glieli si mette davanti, così che risponda
leggendo invece che ricordando. Quei pezzi di testo li chiameremo
**passaggi**: sono i pezzetti in cui l'archivio è stato tagliato, lunghi una
frase o un paragrafo. Chi va a cercarli è il **cercatore** (in inglese
*retriever*), chi poi scrive la risposta è il **generatore**.

«Su cosa salta il gatto nero?». Nel capitolo sui Transformer, nella sezione
«Cercare per rispondere», il nostro cercatore in miniatura aveva risposto quasi
bene: al primo posto il passaggio giusto, «Il gatto nero salta sul muro del
giardino». Ma al secondo posto si era intrufolato un impostore, «Il gatto dorme
accanto ai fornelli»: vicino per tema, muto sulla domanda. Era il
**quasi-pertinente**, e non era un incidente di percorso.

Era il sintomo di un limite di fondo, ed è il perno di tutta la sezione. Se il
passaggio giusto non viene ripescato, il generatore non ha modo di rimediare:
può solo tornare a ricordare, che è esattamente la cosa che il recupero serviva
a evitare (ricordando, un modello inventa; leggendo, no). La quota dei passaggi
giusti che la ricerca riesce a ripescare ha un nome, il *recall*, ed è il
**tetto** di tutto il sistema: nessun accorgimento applicato dopo può
recuperare ciò che la ricerca non ha trovato.

Quanto sia duro quel tetto lo ha misurato l'articolo che ha inaugurato la RAG.
Prendiamo le domande in cui la risposta non compare in **nessuno** dei
documenti recuperati: il sistema di Lewis e colleghi le azzecca comunque poco
più di una volta su dieci, l'$11{,}8\%$ su *Natural Questions*, che è una
raccolta di domande vere rivolte a un motore di ricerca
{cite}`lewis2020retrieval`. Quel po' che resta viene dalla memoria del
modello, cioè da quello che gli era rimasto impresso in addestramento.

Il numero si legge in due modi, e vale la pena tenerli tutti e due. In un
senso è tanto: un sistema che sappia soltanto *ritagliare* la risposta dai
documenti che ha davanti, senza poterla ricordare, in quei casi prende zero per
forza. In un altro senso è pochissimo: nove volte su dieci, quando la ricerca
manca il bersaglio, la risposta è persa. È abbastanza poco da fare del recupero
il posto giusto dove intervenire.

La RAG di base, così come l'abbiamo costruita, faceva tre gesti: trasformava
domanda e passaggi in punti su una mappa del significato, prendeva i pochi
passaggi più vicini alla domanda (quanti, lo decidiamo noi: diciamo i primi
cinque) e li incollava nel foglietto di istruzioni che si dà al modello, il
*prompt*, prima di fargli scrivere la risposta {cite}`karpukhin2020dense`. È
un ottimo punto di partenza e un pessimo punto di arrivo.

Questa sezione raccoglie le tecniche che spingono quel tetto più in alto. Sono
tre, e si distinguono per dove intervengono lungo la catena di passi che porta
dalla domanda alla risposta (una catena così, in gergo, si chiama
**pipeline**): **prima** di cercare, migliorando la domanda; **dopo** aver
cercato, riordinando i candidati; e **attorno** all'intero ciclo, facendo
decidere al modello se e quando cercare. Chiudiamo con la domanda che tiene
onesto tutto il resto: come si misura se un sistema RAG funziona davvero.

```{figure} ../figures/rag-avanzato.svg
:name: fig-rag-avanzato
:alt: "Sette riquadri collegati da frecce, da sinistra a destra. La domanda dell'utente entra in un riquadro che la riscrive; da lì il percorso si sdoppia in due ricerche fatte in parallelo, una per significato e una per parole così come sono scritte; i due rami si ricongiungono in un riquadro di fusione, che produce una lista unica di una cinquantina di candidati; questi passano a un riordinatore, che ne tiene i primi cinque e li fa scendere all'ultimo riquadro, il modello di linguaggio. In fondo la scritta: recupero generoso a monte, filtro severo a valle."
:width: 100%

La catena per intero. Rispetto alla RAG di base cambiano tre cose: la domanda
non va a cercare com'è arrivata, la ricerca è doppia (per significato e per
parole così come sono scritte) e fra il recupero e il modello si interpone un
riordino.
```

Conviene tenere {numref}`fig-rag-avanzato` sott'occhio mentre si legge il
resto. Dei suoi blocchi, due li sbrighiamo qui sotto in poche righe (la doppia
ricerca e la fusione) e gli altri hanno una sezione ciascuno. La regola che
governa l'insieme è una sola: il recupero grezzo, quello all'inizio, deve
essere **generoso** (meglio cento candidati mediocri che dieci scelti male,
perché ciò che non entra lì è perduto per sempre) e ciò che viene dopo deve
essere **severo**, perché al modello arrivi poco e buono. È il senso della
scritta in fondo al disegno, dove *a monte* vuol dire all'inizio della catena
e *a valle* alla fine, come per un fiume.

Il disegno chiama «densa» la prima delle due ricerche, e la parola da sola non
dice niente. Una ricerca **densa** confronta *significati*: domanda e passaggi
diventano punti su una mappa, e si prendono i più vicini, anche quando non
condividono una sola parola. L'altra, chiamata **sparsa**, confronta invece le
parole così come sono scritte: conta quante parole della domanda compaiono nel
passaggio, dando più valore a quelle rare (se cerchi «guarnizione», trovarla
vale molto più che trovare «il»). Il modo di fare quel conto che si è imposto
si chiama **BM25**, ha trent'anni e funziona ancora benissimo.

Le due ricerche sbagliano in modi diversi, ed è per questo che conviene farle
entrambe. Quella densa capisce che «auto» e «vettura» sono la stessa cosa, ma
può perdere un codice di prodotto scritto identico; quella sparsa il codice lo
trova al primo colpo, ma davanti a un sinonimo resta muta.

Cercando due volte, però, ci si ritrova con due classifiche invece che con
una, e vanno rimesse insieme: è la **fusione** del disegno. C'è un ostacolo, e
un modo elegante di aggirarlo. L'ostacolo è che le due ricerche danno punteggi
calcolati in modi diversi, e sommarli non vorrebbe dire niente, come sommare
un voto in decimi a uno in centesimi. Il modo di aggirarlo è buttare via i
punteggi e tenere solo la **posizione** in classifica: chi sta in alto in tutte
e due le liste sale, chi sta in alto in una sola resta indietro. Basta questo,
e non serve sapere quanto valgano i due voti né come siano stati calcolati.

Un'ultima immagine, e poi si entra nel merito. Quella qui sotto guarda la
stessa catena da un'altra angolatura, non *dove* si interviene ma *quando* si
paga.

```{figure} ../figures/extra-rag-spiegato.svg
:name: fig-rag-due-fasi
:alt: "Pipeline RAG divisa in due fasi. La prima, di indicizzazione, si esegue una volta sola: i documenti vengono spezzati in blocchi, ogni blocco convertito in un vettore e depositato in un archivio di vettori. La seconda, a ogni domanda: la domanda diventa un vettore, si recuperano i blocchi più vicini, e questi entrano nel prompt del modello, che risponde citando le fonti."
:width: 100%

Due fasi con tempi diversi. Nella prima si prepara l'archivio: i documenti si
spezzano in blocchi (nel disegno, *chunk*), ogni blocco diventa un punto sulla
mappa del significato (un *embedding*) e va a finire nell'archivio di quei
punti (*vector store*). Si paga una volta sola e si riusa sempre. La seconda
fase, invece, si paga a ogni domanda: si cercano i blocchi più vicini
(*retrieval*), si incollano nel prompt e il modello risponde citando le fonti.
```

La separazione di {numref}`fig-rag-due-fasi` conviene tenerla in testa per
tutto il resto: le tecniche che seguono intervengono quasi tutte nella seconda
fase, quella che si ripaga a ogni domanda ricevuta e che decide quanti secondi
l'utente aspetta prima di vedere qualcosa (quell'attesa si chiama **latenza**,
e tornerà più volte).

## Migliorare la domanda: riscrittura ed espansione

Prima di entrare nel merito, una parola sul termine che tornerà in ogni riga
di questa sezione. Nel gergo del recupero si chiama **query** il testo con cui
si interroga l'archivio. Non è sempre la domanda dell'utente, ed è proprio
questo il punto: la domanda è quello che una persona ha scritto, la query è
quello che mandiamo davvero a cercare. Tutta la prima leva consiste nel non
farle coincidere.

Ed è una leva a cui si pensa per ultimi, perché sembra fuori dal nostro
controllo: la domanda la scrive l'utente, e noi la subiamo. La ricerca per
significato presume che la query e il passaggio giusto finiscano vicini sulla
mappa; ma la domanda che scrive una persona è quasi sempre la query
*peggiore* per cercare, perché è breve, sbrigativa, piena di sottintesi, a
volte una sola parola. Chi ha appena letto una pagina sulla manutenzione
dell'automobile e digita «e il tagliando?» non ha alcuna speranza di arrivare
al manuale d'officina, semplicemente perché quelle tre parole non dicono
abbastanza.

L'idea è interporre, tra l'utente e la ricerca, un passaggio di
**riscrittura**: un modello di linguaggio riformula la domanda in una o più
query pensate *per la ricerca*. Le varianti sono tre, di ambizione crescente.
La **riscrittura** rende esplicito il sottinteso («il tagliando» →
«manutenzione periodica dell'automobile, intervalli e operazioni»).
L'**espansione** aggiunge sinonimi e termini correlati, così da coprire più
modi di dire la stessa cosa. La **multi-query** genera diverse riformulazioni
della stessa domanda, cerca con ciascuna e fonde i risultati: se anche una
sola variante «azzecca» le parole del documento giusto, quel documento entra.

`````{tab} Elementare

Immagina di chiedere a un bibliotecario: «Ha qualcosa sul cane?». Se prende la
richiesta alla lettera cercherà la parola «cane» e ti porterà un romanzo dal
titolo *Il cane di terracotta*, che con i cani non c'entra nulla. Un
bibliotecario esperto invece ti interroga un attimo e capisce cosa cerchi
davvero («l'educazione del cucciolo») e allora fa non una, ma tre ricerche
mirate: *addestramento del cane*, *comportamento del cucciolo*, *comandi di
base*. Con tre reti gettate in punti diversi, la probabilità di tirare su il
libro giusto sale.

C'è un trucco ancora più sorprendente, che a prima vista sembra assurdo, e si
chiama **HyDE**, che sta per «documenti di risposta immaginati». Invece di
cercare con la *domanda*, cerchi con una **risposta inventata**. Chiedi al modello: «Scrivi tu, di getto, come *sarebbe* la
risposta ideale» (anche se sbaglia qualche dettaglio) e poi cerchi i documenti
veri che somigliano a quella risposta finta. Perché funziona? Perché una
domanda e la sua risposta sono scritte in modi diversi (una chiede, l'altra
afferma), mentre due risposte sullo stesso tema si somigliano molto. La
risposta inventata fa da esca: pesca meglio delle domande i documenti che
*sono* fatti di risposte.

Un avvertimento, però, perché il trucco ha il suo posto e non è dappertutto.
Un cercatore si può **addestrare** sul proprio archivio, cioè lo si può
correggere mostrandogli molte domande con accanto i passaggi che le
soddisfano, finché non impara a pescare quelli. Chi può permetterselo (e
servono migliaia di esempi, raccolti a mano o dedotti da quello che gli utenti
cliccano) fa quello, ed è la strada migliore. HyDE nasce per l'altro caso,
quello di chi quegli esempi non li ha: il primo giorno, quando l'archivio è
nuovo e nessuno ci ha ancora cercato niente.

`````

`````{tab} Superiore

Il trucco dell'ultimo paragrafo ha un nome (**HyDE**, *Hypothetical Document
Embeddings* {cite}`gao2023hyde`) e risolve un problema geometrico preciso: la
**asimmetria** tra query e documenti. Una domanda («su cosa salta il gatto?»)
e il passaggio che la soddisfa («il gatto salta sul muro») sono testi di forma
diversa, e un encoder addestrato genericamente può collocarli non così
vicini quanto vorremmo. HyDE aggira l'ostacolo: chiede a un LLM di generare
$M$ documenti *ipotetici* di risposta e interroga l'indice con la media dei
loro embedding e di quello della query,

$$
\mathbf{v}_{\text{ricerca}} = \frac{1}{M+1} \Big( \sum_{i=1}^{M} E_p(\tilde{d}_i) + E_p(q) \Big),
\qquad \tilde{d}_i \sim \mathrm{LLM}(\,\cdot \mid q\,),
$$

dove $q$ è la domanda, le $\tilde{d}_i$ sono $M$ risposte ipotetiche
**campionate indipendentemente** dal modello condizionato sulla domanda (il
campionamento è essenziale: con una decodifica deterministica le $M$
generazioni coinciderebbero e la media collasserebbe su un solo embedding),
$E_p$ l'encoder dei passaggi (lo stesso che ha indicizzato l'archivio, e che
qui codifica anche la query, trattata come un documento in più) e
$\mathbf{v}_{\text{ricerca}}$ il vettore con cui si interroga l'indice.
L'intuizione è che le $\tilde{d}_i$, pur potendo contenere errori fattuali,
vivono nello **spazio delle risposte** (la stessa regione dove abitano i
passaggi veri) e ci somigliano più di quanto ci somigli la domanda. Attenzione
però a non leggere il termine $E_p(q)$ come un'àncora: la query entra nella
media con peso $1/(M+1)$, cioè conta esattamente quanto una delle ipotesi, e
già con una manciata di documenti generati il suo contributo è minoritario. In
pratica i documenti ipotetici sono generati dall'LLM e poi **codificati** con
l'encoder dei documenti: le allucinazioni del modello non finiscono nella
risposta finale, servono solo da esca per il recupero.

Il perimetro va dichiarato, perché è la cosa più utile a chi deve decidere se
adottare il metodo, ed è scritta nel lavoro originale: HyDE nasce per il caso
in cui **non si hanno etichette di rilevanza**, con un unico encoder
contrastivo non supervisionato usato indifferentemente per query e documenti.
Gli autori sono espliciti nel dire che l'uso con un retriever messo a punto sul
proprio dominio *non è quello previsto*.

Il quadro che misurano su quel caso è più sfumato di come lo si racconta di
solito, e vale la pena riportarlo com'è, perché non dipende dalla raccolta ma
da **quanto è buono il modello che genera le ipotesi**. Con un generatore forte
HyDE alza anche un retriever addestrato sul dominio, ma in modo asimmetrico: su
TREC DL19 l'NDCG@10 passa da $62{,}1$ a $67{,}4$, su DL20 da $63{,}2$ a
$63{,}5$, cioè tre decimi, che è niente. Con generatori più deboli lo
**peggiora** su entrambe le raccolte, di poco. La lettura onesta è che HyDE
risolve il problema di chi **non ha etichette di rilevanza**; dove quelle
etichette ci sono, è una cosa da provare e misurare, non un guadagno che si
somma a occhi chiusi. Gli autori stessi lo inquadrano come una fase: HyDE il
primo giorno, quando non c'è ancora niente su cui addestrare, e via via che il
registro delle ricerche cresce il traffico passa a un retriever supervisionato,
lasciando a HyDE le domande rare e nuove.

Attenzione anche alla lettera della formula: qui c'è un
encoder solo, mentre poche pagine più avanti scriveremo la similarità come
$E_q(q)^\top E_p(d)$, con due reti distinte. Applicare la ricetta di HyDE su un
indice a due encoder significherebbe codificare la query con la rete
sbagliata. Le riscritture
multi-query si formalizzano invece come unione dei risultati, spesso fusi con
la **reciprocal rank fusion** {cite}`cormack2009reciprocal`: a ogni documento
si assegna il punteggio $\sum 1/(c + r)$, sommando sui ranghi $r$ che occupa
nelle diverse liste, dove $c$ è una costante di smorzamento ($60$
nell'articolo originale) che evita di sovrappesare i primissimi posti.

`````

## Riordinare i candidati: il reranking

Riordinare vuol dire prendere quello che la ricerca ha pescato e rimetterlo in
fila con più cura. Prima di poterlo fare, però, bisogna capire come pesca un
archivio vero, perché finora ci siamo limitati a dire «prende i più vicini» e
non abbiamo mai detto come faccia a trovarli senza guardarli tutti. Sono le
prossime venti righe, e poi si torna al riordino.

Cominciamo dal nome. I punti sulla mappa del significato di cui parliamo
dall'inizio si chiamano **vettori**, e un vettore è semplicemente una lista di
numeri: quelli del nostro mini-archivio erano lunghi quattro, e ciascuno dei
quattro misurava quanto la frase parlasse di un certo tema (gatti, muri,
automobili, cucina). In un sistema vero i numeri sono da qualche centinaio a
qualche migliaio, e nessuno sa dire che cosa misuri ciascuno: li ha scelti
l'addestramento.

Cercare, allora, vuol dire trovare i vettori più vicini a quello della
domanda, e l'archivio va guardato tutto: nessun passaggio deve essere escluso
in partenza. Il modo ovvio per farlo sarebbe confrontare la domanda con ogni
passaggio, a uno a uno: esatto, e impraticabile su milioni di documenti.

Il modo che si usa davvero copre lo stesso archivio ma senza toccarlo tutto,
procedendo per **scale successive**, come si cerca un indirizzo in una città:
prima il quartiere, poi l'isolato, poi il numero civico. Nessuna via è esclusa
in principio, però si visitano solo le poche che servono. Si rinuncia così alla
garanzia di trovare *sempre* il vicino migliore, in cambio di una ricerca
incomparabilmente più rapida; ed è un altro motivo per cui il recupero grezzo
va tenuto generoso, perché per strada qualche buon candidato si perde.

Quelle scale successive, disegnate, sono {numref}`fig-hnsw`, dove prendono la
forma di **strati** sovrapposti.

```{figure} ../figures/vector-database.svg
:name: fig-hnsw
:alt: "Tre file di punti sovrapposte, etichettate strato 2 in alto, strato 1 al centro e strato 0 in basso. In cima pochi punti, collegati da linee lunghe che scavalcano metà del disegno; scendendo, i punti si infittiscono e le linee si accorciano; in basso ci sono tutti i punti, collegati solo ai loro vicini immediati. Il percorso di ricerca, marcato più scuro, entra in alto a sinistra, scende alla fila di mezzo, la percorre verso destra e scende ancora fino al punto colorato, marcato «più vicino». In fondo la legenda: grandi balzi in alto, passi corti in basso."
:width: 92%

Come si cerca fra milioni di vettori senza confrontarli tutti. Gli strati alti
servono ad arrivare nella zona giusta con pochi salti; quelli bassi a trovare
il vicino esatto. È lo stesso mestiere del quartiere, dell'isolato e del
numero civico.
```

La seconda leva agisce a valle del recupero, e poggia su una distinzione che
vale la pena rifare per intero, perché è il cuore di tutta la sezione. Ci sono
due modi di far confrontare una domanda con un passaggio.

Il primo è quello che abbiamo usato finora: si riassume il passaggio in un
punto sulla mappa, si riassume la domanda in un altro punto, e si guarda
quanto sono vicini. Ha un nome, **bi-encoder** («due codificatori», perché i
due testi vengono letti separatamente, ciascuno per conto proprio), ed è
velocissimo per una ragione sola: i punti dei passaggi si calcolano una volta
per tutte, quando si prepara l'archivio, e poi si riusano a ogni domanda.

Il secondo modo è dare i due testi **insieme** a un unico modello, che li legge
uno accanto all'altro e dice, guardandoli entrambi, quanto il secondo risponde
al primo. Si chiama **cross-encoder** («codificatore incrociato», perché
incrocia i due testi invece di tenerli separati), ed è molto più accurato,
perché può accorgersi che una frase parla dello stesso argomento senza
rispondere alla domanda. Il prezzo è che non si può precalcolare niente: il
confronto va rifatto da capo per ogni coppia, e passare in rassegna così
milioni di documenti è fuori discussione. La soluzione è usarli **in due
stadi**.

Il primo stadio, il bi-encoder, fa il grosso: percorre l'archivio con le scale
successive di poco fa e restituisce non i pochi passaggi che finiranno sotto
gli occhi del modello, ma molti più candidati grezzi (cinquanta, cento)
tra cui, si spera, ci sono anche i migliori. Il secondo stadio, il
**reranker** cross-encoder, si applica *solo* a quella rosa ristretta e la
riordina con cura, promuovendo i passaggi che davvero rispondono e affondando
i quasi-pertinenti. È il compromesso classico dell'ingegneria del recupero:
recuperare tanto e alla svelta, poi riordinare poco e per bene.

`````{tab} Elementare

Pensa a come si assume una persona. Non si fa un colloquio di due ore a ognuno
dei mille che hanno mandato il curriculum: si fa prima una scrematura rapida
(sguardi il CV, tieni i cinquanta più promettenti) e *solo* a quei cinquanta
si fa il colloquio vero, quello che costa tempo ma capisce davvero chi hai
davanti. Il bi-encoder è la scrematura dei curriculum: veloce, superficiale,
va bene per buttare via i chiaramente fuori tema. Il cross-encoder è il
colloquio: lento, ma legge la domanda e il candidato *insieme* e non si fa
ingannare da chi «suona simile» senza rispondere. Applicarlo a tutti sarebbe
rovinoso; applicarlo ai cinquanta scremati è esattamente il punto giusto.

`````

`````{tab} Superiore

La struttura che rende possibile la ricerca «per scale» della figura è
**HNSW** {cite}`malkov2020hnsw` (*Hierarchical Navigable Small World*): un
grafo a strati in cui quelli alti tengono pochi nodi con archi lunghi, buoni
per attraversare in fretta lo spazio, e quelli bassi tutti i punti con archi
solo fra vicini. La discesa strato per strato è una ricerca *approssimata*: in
cambio della garanzia di esattezza, il numero di confronti cresce molto più
lentamente della dimensione dell'archivio, e raddoppiare i vettori costa un
pugno di confronti in più invece del doppio.

Sulla forma esatta di quella crescita conviene però essere precisi, perché è
il punto in cui si tende a promettere un teorema che non c'è. Gli autori
**argomentano** una scalabilità **logaritmica** in $N$ e la misurano in un
caso solo (vettori casuali a otto dimensioni, dieci vicini cercati, recall
tenuto fermo a $0{,}95$), dove osservano una complessità «non peggiore che
logaritmica». È evidenza empirica su dati sintetici a bassa dimensione, non
una garanzia dimostrata in generale. E in ogni caso il logaritmo è in $N$, non
nella dimensione degli embedding: il prezzo di vettori più lunghi non
sparisce, si sposta nel costo del singolo confronto.

Formalmente il primo stadio ordina l'archivio con la similarità del bi-encoder
$E_q(q)^\top E_p(d)$ e ne trattiene i primi $N$; il secondo riordina
questi $N$ con il punteggio del cross-encoder
$\mathrm{score}(q, d) = \mathrm{CrossEnc}([q; d])$, dove $[q; d]$ è la
concatenazione dei due testi data in input a un unico Transformer, e tiene i
primi $k$, cioè i passaggi che entreranno davvero nel prompt del generatore
(si sceglie $N \gg k$: tipicamente $N$ nell'ordine delle decine o centinaia,
$k$ pochi). Il costo del reranking è $N$ inferenze di cross-encoder per query:
accettabile con quei valori di $N$, proibitivo sull'intero archivio.

Tra i due estremi esiste una via di mezzo elegante: l'**interazione tardiva**
di **ColBERT** {cite}`khattab2020colbert`. Invece di collassare ogni testo in
*un* vettore (bi-encoder) o di rifare l'attenzione congiunta a ogni query
(cross-encoder), ColBERT conserva un embedding *per token* e calcola la
rilevanza con l'operatore **MaxSim**:

$$
s(q, d) = \sum_{i \,\in\, q} \max_{j \,\in\, d}\; E(q_i)^\top E(d_j),
$$

dove $q_i$ è l'$i$-esimo token della query, $d_j$ il $j$-esimo del documento,
ed $E(\cdot)$ il loro embedding contestuale. Per ogni token della domanda si
prende la migliore corrispondenza tra i token del documento e si sommano: un
confronto fine, token-a-token, ma con gli embedding dei documenti
**precalcolabili offline** come nel bi-encoder. Si guadagna gran parte della
precisione del cross-encoder senza pagarne il costo a query time, al prezzo di
un indice molto più grande (un vettore per token, non per passaggio).

`````

Vediamo il riordino in azione, estendendo il cercatore in miniatura di
«Cercare per rispondere». Teniamo lo stesso mini-archivio (ogni frase è
riassunta da quattro numeri, uno per ciascuno dei quattro temi presenti: gatti,
muri, automobili, cucina) e la stessa domanda; aggiungiamo lo **stadio di
reranking**. Quanto due frasi si somiglino lo dice un numero che qui va da $0$
(niente in comune) a $1$ (stessa direzione esatta): si chiama **coseno**,
perché si ricava dall'angolo fra i due punti sulla mappa, ed è la stessa misura
che avevamo usato per costruire il cercatore.

Il cross-encoder è finto, non è un vero modello addestrato, ma imita la cosa
che conta: legge la coppia. Il bi-encoder ha schiacciato ogni frase in un punto
solo, e da lì «gatto» e «salta» sono ormai mescolati; il nostro reranker riceve
invece *domanda e passaggio* e conta quanti concetti della domanda il passaggio
copre, dando un premio a ogni **coppia** di concetti che compaiono **insieme**.
È il punto: a rispondere non è il gatto, e nemmeno il saltare, ma un gatto *che
salta*.

La regola esatta, così il conto si può rifare a mente: ogni concetto della
domanda che il passaggio copre vale un punto, e ogni coppia di concetti coperti
insieme ne vale due. La domanda porta tre concetti (gatto, nero, saltare); il
passaggio giusto li copre tutti e tre, quindi fa $3$ punti per i concetti più
$2 \times 3 = 6$ per le tre coppie che se ne ricavano, in tutto $9$. «Il gatto
dorme accanto ai fornelli» copre il solo «gatto»: un concetto, nessuna coppia,
un punto. Il premio alle coppie ce lo siamo scelto noi, ed è la sola cosa finta
di tutto il blocco: in un cross-encoder vero questa preferenza per le
combinazioni non si scrive a mano, si impara dagli esempi.

```python
import torch
from itertools import combinations

# --- Stadio 1: recupero grezzo con il bi-encoder ---
# stesso mini-archivio della sezione «Cercare per rispondere»:
# quattro dimensioni leggibili [gatti, muri/casa, automobili, cucina].
passaggi = [
    "Il gatto nero salta sul muro del giardino.",
    "Il muro portante sostiene il solaio.",
    "La vettura elettrica si ricarica in garage.",
    "L'auto storica sfila per il centro.",
    "Il gatto dorme accanto ai fornelli.",
    "La ricetta prevede burro e salvia.",
]
E = torch.tensor([
    [0.9, 0.6, 0.0, 0.1],
    [0.1, 0.9, 0.1, 0.0],
    [0.0, 0.1, 0.9, 0.0],
    [0.1, 0.0, 0.8, 0.1],
    [0.8, 0.2, 0.0, 0.5],
    [0.0, 0.1, 0.1, 0.9],
])
E = E / E.norm(dim=1, keepdim=True)   # righe normalizzate: prodotto = coseno

domanda = "Su cosa salta il gatto nero?"
q = torch.tensor([0.9, 0.7, 0.0, 0.0])
q = q / q.norm()

# il bi-encoder e' veloce, quindi copre tutto l'archivio (qui sei passaggi, e
# li confrontiamo davvero uno a uno; in un archivio vero si userebbero gli
# strati della figura, che coprono tutto senza toccare tutto). Ma e' grezzo.
# recuperiamo di proposito piu' candidati di quanti ne serviranno: sono il
# terreno di caccia dello stadio successivo.
sim = E @ q
val, cand = torch.topk(sim, k=4)

print("Stadio 1 - bi-encoder (veloce, tutto l'archivio):")
for v, i in zip(val, cand):
    print(f"  coseno {v:.2f}  {passaggi[i]}")

# --- Stadio 2: reranking con un "cross-encoder" didattico ---
# Il bi-encoder ha collassato ogni frase in un unico vettore: cosi' un
# passaggio che condivide solo il tema "gatto" gli sembra vicino. Un
# cross-encoder legge domanda e passaggio INSIEME. Qui lo simuliamo con i
# concetti dei due testi, premiando le CO-OCCORRENZE: a rispondere non e' il
# gatto, ne' il saltare, ma un gatto CHE SALTA.
concetti = {
    domanda: {"gatto", "nero", "saltare"},
    0: {"gatto", "nero", "saltare", "muro", "giardino"},  # copre tutto: risponde
    1: {"muro", "portante", "solaio"},
    2: {"vettura", "garage"},
    3: {"auto", "centro"},
    4: {"gatto", "dormire", "fornelli"},   # solo il tema "gatto": quasi-pertinente
    5: {"ricetta", "burro"},
}

def cross_encoder(domanda, i):
    """Punteggio della COPPIA (domanda, passaggio), non del solo passaggio.
    Ogni concetto della domanda coperto vale 1; ogni coppia di concetti
    coperti INSIEME vale 2, perche' e' la combinazione a rispondere."""
    coperti = concetti[domanda] & concetti[i]
    return len(coperti) + 2 * len(list(combinations(coperti, 2)))

# il cross-encoder e' costoso: lo applichiamo SOLO ai candidati dello stadio 1
riordino = sorted(cand.tolist(), key=lambda i: cross_encoder(domanda, i),
                  reverse=True)

print("\nStadio 2 - cross-encoder (preciso, solo sui 4 candidati):")
for i in riordino:
    print(f"  pertinenza {cross_encoder(domanda, i):.1f}  {passaggi[i]}")

# ora che i punteggi sono separati, una soglia ha senso: passa solo chi si
# avvicina al migliore, invece di riempire a forza un numero fisso di posti.
migliore = cross_encoder(domanda, riordino[0])
rosa = [i for i in riordino if cross_encoder(domanda, i) >= migliore / 2]

print("\nAl generatore va solo chi supera meta' del punteggio migliore:")
for n, i in enumerate(rosa, 1):
    print(f"  [{n}] {passaggi[i]}")
```

L'output racconta la storia in due tempi:

```text
Stadio 1 - bi-encoder (veloce, tutto l'archivio):
  coseno 0.99  Il gatto nero salta sul muro del giardino.
  coseno 0.78  Il gatto dorme accanto ai fornelli.
  coseno 0.69  Il muro portante sostiene il solaio.
  coseno 0.10  L'auto storica sfila per il centro.

Stadio 2 - cross-encoder (preciso, solo sui 4 candidati):
  pertinenza 9.0  Il gatto nero salta sul muro del giardino.
  pertinenza 1.0  Il gatto dorme accanto ai fornelli.
  pertinenza 0.0  Il muro portante sostiene il solaio.
  pertinenza 0.0  L'auto storica sfila per il centro.

Al generatore va solo chi supera meta' del punteggio migliore:
  [1] Il gatto nero salta sul muro del giardino.
```

Ecco il punto, e non è quello che ci si aspetterebbe. L'ordine dei primi due
non cambia: il quasi-pertinente «Il gatto dorme accanto ai fornelli» era
secondo e resta secondo. Quello che cambia è la **distanza** fra il primo e il
secondo. Il bi-encoder li dava a $0{,}99$ contro $0{,}78$: una differenza che
non permette di decidere niente, perché il quasi-pertinente è quasi buono
quanto la risposta. Il cross-encoder li dà a $9{,}0$ contro $1{,}0$, e nove
volte non è un margine sfumato: è un verdetto.

Da lì viene il guadagno vero, che è una decisione diventata possibile. Con
punteggi indistinguibili l'unica regola disponibile è «prendine i primi
$k$», e riempiendo i posti si finisce per infilare nel prompt un passaggio
che non risponde. Con punteggi separati si può mettere una **soglia**: qui
teniamo chi supera metà del punteggio migliore, cioè $4{,}5$, e l'unico a
passare è il $9$. Al generatore arriva un passaggio solo, quello giusto, senza
compagnia fuorviante. Notiamo anche che «Il muro portante sostiene il solaio»,
che pure condivide una parola con la risposta, finisce a zero, ed è corretto:
la domanda parlava di un gatto che salta, non di solai.

Non abbiamo alzato il recall, perché il passaggio giusto era già stato
recuperato. Abbiamo alzato un'altra cosa, la **precisione**: la quota di roba
buona fra quella che consegniamo al generatore. Sono due misure gemelle e
raccontano due guai diversi: il recall dice quanto ci siamo persi per strada,
la precisione quanta spazzatura abbiamo consegnato insieme al buono. Il tetto
del sistema resta il primo, perché ciò che non si trova non si recupera più; ma
la seconda si può alzare a valle, ed è quello che abbiamo appena fatto,
guadagnando la capacità di dire di no.

## Il RAG che si corregge: Self-RAG e RAG agentico

Fin qui abbiamo migliorato una catena che resta **rigida**: cerca sempre, una
volta sola, poi genera. E cade in due modi opposti.

Cercare quando non serve è uno spreco, e non solo di tempo. «Traduci
"buongiorno" in francese» non ha bisogno di alcun archivio, ma il sistema
rigido ci va lo stesso, e quel che riporta finisce nel prompt: passaggi che
non c'entrano niente, davanti agli occhi del modello, con qualche probabilità
di distrarlo dalla cosa semplice che gli era stata chiesta.

Cercare una volta sola, all'opposto, a volte non basta. Ci sono domande la cui
risposta vive nell'incrocio di due fatti che stanno in documenti diversi, e
nessuna singola ricerca li riporta insieme: per trovare il secondo bisogna
sapere il primo. Lì il sistema rigido non fallisce per distrazione, fallisce
per costruzione.

La terza leva rompe la rigidità nello stesso modo in cui si rompe ogni
rigidità in questo capitolo: lascia che sia il **modello** a decidere se,
quando e quante volte cercare.

C'è anche una via diversa al secondo dei due problemi, e non passa dal cercare
più volte: passa dal cambiare la forma dell'archivio. Invece di paragrafi si
tengono i fatti in una rete di collegamenti, come una mappa di città unite da
strade, e allora incrociare due fatti vuol dire percorrere due strade. Il
capitolo sulle reti neurali su grafo la riprende per esteso.

`````{tab} Elementare

Torniamo allo studente all'esame a libro aperto. Lo studente ingenuo apre il
libro *a ogni* domanda, anche a «quanto fa sette per otto»: perde tempo e
rischia di copiare la pagina sbagliata. Lo studente maturo fa tre cose in più.
Primo, si chiede *se* gli serve il libro: alle domande che sa già risponde a
memoria e basta. Secondo, quando lo apre, **rilegge criticamente** ciò che ha
trovato: «questa pagina risponde davvero alla domanda, o l'ho aperta a caso?».
Terzo, se una pagina non basta, ne apre un'altra, e un'altra ancora, finché
non ha in mano tutto quello che serve. Non è più un gesto automatico (apri,
copia) ma un piccolo ciclo di decisioni: mi serve cercare? ho trovato la cosa
giusta? mi manca ancora qualcosa? È la differenza tra consultare un libro e
saperlo consultare.

Le due cose hanno un nome, e vale la pena averlo in tasca. Un modello
addestrato a chiedersi da sé se gli serve aprire il libro, e a rileggere con
occhio critico quello che ha trovato, si chiama **Self-RAG**, cioè «RAG che si
controlla da solo». Un agente che invece torna a cercare quante volte gli
serve, perché cercare non è più un passo obbligato ma un attrezzo che prende
quando vuole, fa il **RAG agentico**.

`````

`````{tab} Superiore

Il primo dei due comportamenti è **Self-RAG** {cite}`asai2024selfrag`: il
modello è addestrato a emettere, intercalati al testo, degli speciali **token
di riflessione**. Un token *Retrieve* decide, passo per passo, se in quel
momento serve recuperare (`sì`/`no`/`continua`); quando il recupero avviene,
altri token *critici* valutano ciascun passaggio (è **rilevante** per la
domanda? la frase generata è **supportata** dal passaggio, o lo travisa? la
risposta è nel complesso **utile**?) e questi giudizi diventano un punteggio
che seleziona la generazione migliore. Il modello impara così non solo a
rispondere, ma a *criticare le proprie fonti e sé stesso*, riducendo il caso
in cui un passaggio recuperato ma irrilevante trascina la risposta fuori
strada.

Il secondo comportamento è il **RAG agentico**: il recupero smette di essere
un passo obbligato della pipeline e diventa uno **strumento** che un agente
può invocare quando vuole, più volte, in un ciclo. È la stessa idea del *tool
use* di questo capitolo (un modello che ragiona, decide di chiamare uno
strumento, ne legge il risultato e decide la mossa successiva) applicata alla
ricerca: l'agente formula una query, esamina i risultati, si accorge che gli
manca un pezzo, formula una seconda query mirata, e itera finché ha raccolto
abbastanza per rispondere. È ciò che serve alle domande **multi-hop**, dove la
risposta vive nell'incrocio di fonti che nessuna singola ricerca restituisce
insieme.

`````

L'onestà, qui, è d'obbligo, ed è la stessa che abbiamo tenuto per tutto il
libro. Ogni giro in più (una riscrittura, un riordino, una seconda tornata di
ricerca, una pausa in cui il modello si chiede se quello che ha trovato serve
davvero) vuol dire far lavorare il modello un'altra volta. Ogni volta si paga:
il modello sta da qualche parte su una macchina che consuma, e chi lo usa lo
paga a consumo, un tanto per ogni pezzetto di testo che entra e che esce. Due
voci, quindi, e crescono insieme: **latenza** e **denaro**. Un RAG agentico che
fa cinque giri è cinque volte più lento e più caro di un recupero secco, e non
sempre di qualità cinque volte migliore. La domanda ingegneristica non è
«quanti giri posso fare», ma «qual è il numero minimo di giri che risolve
*questa* classe di domande»: sulle domande semplici, spesso, la risposta è
zero.

## Valutare un sistema RAG

Tutte queste leve hanno senso solo se sappiamo dire quale migliora il sistema e
quale no. Ma un RAG ha due pezzi, il cercatore e il generatore, e una risposta
sbagliata può essere colpa dell'uno o dell'altro: o la ricerca ha mancato il
documento giusto, o il documento c'era e chi ha scritto la risposta l'ha
ignorato, o l'ha letto male. Un voto solo non distingue i due casi, e chi
ripara non sa dove mettere le mani. Servono misure separate, che sappiano
**dove** guardare.

`````{tab} Elementare

Immagina di correggere il compito di uno studente che cita le fonti. Non basta
un voto solo: devi controllare tre cose diverse. Primo, la **fedeltà**: ciò che
ha scritto è davvero sostenuto dalle pagine che cita, o ha aggiunto di suo
spacciandolo per citazione? Secondo, la **pertinenza**: la risposta parla della
domanda che avevi fatto, o divaga su un tema vicino? Terzo, la **qualità della
ricerca**: le pagine che ha aperto erano quelle giuste, o ne ha aperte di
inutili e saltate di essenziali?

E qui si affaccia una tentazione che tornerà più volte nel libro: siccome
correggere a mano migliaia di risposte costa, si promuove un altro modello a
esaminatore, che
legge risposta e fonti e assegna i tre voti in un lampo. Comodissimo: a patto
di ricordare che quell'esaminatore ha i suoi pregiudizi, e che va tenuto
d'occhio esattamente come si tiene d'occhio uno studente che si autovaluta.

`````

`````{tab} Superiore

Le metriche proprie di un sistema RAG smontano il giudizio nelle sue
componenti. La **fedeltà** (o *groundedness*) misura se ogni affermazione della
risposta è deducibile dai passaggi recuperati; un modo di stimarla è
scomporre la risposta in singole *claim* e contare la frazione supportata dal
contesto:

$$
\text{fedeltà} = \frac{\#\{\text{claim supportate dal contesto}\}}{\#\{\text{claim totali della risposta}\}},
$$

dove una *claim* è una singola affermazione verificabile estratta dalla
risposta. La **pertinenza della risposta** valuta invece se la risposta
indirizza la domanda posta (non un tema adiacente), e la **precision/recall
del contesto** misurano la qualità del recupero a monte: quanti dei passaggi
recuperati sono rilevanti (precision) e quanti dei rilevanti sono stati
recuperati (recall); quest'ultimo è proprio il *tetto* da cui siamo partiti.
Il quadro operativo di riferimento è **RAGAS** {cite}`es2024ragas`, che
nell'articolo originale propone tre metriche **senza risposte di riferimento**
(*reference-free*): fedeltà, pertinenza della risposta e rilevanza del
contesto (una precision), affidate a un LLM che fa da giudice. La recall del
contesto fa eccezione: per contare i rilevanti *mancati* serve una risposta di
riferimento annotata, e infatti la libreria la calcola solo se gliene si dà
una. Il reference-free è comodo perché non richiede un dataset etichettato a
mano, ma eredita in blocco i limiti
dell'**LLM-as-a-judge** che vedremo più avanti, nel capitolo su MLOps (il
*position bias*, il *verbosity bias*, l'auto-preferenza) e va perciò calibrato
contro un campione di giudizi umani, mai preso per oracolo.

`````

Un'ultima avvertenza, che è il filo rosso di tutta la RAG. La metrica più
importante (la fedeltà) non va confusa con la verità. Un sistema può essere
**perfettamente fedele** e **fattualmente sbagliato**: se l'archivio contiene
un documento errato, la risposta più fedele possibile a quel documento sarà
errata, con tanto di citazione impeccabile. E vale anche il rovescio, già
enunciato in «Cercare per rispondere»: una citazione formalmente corretta non rende
vera una risposta che ne **travisa** il contenuto. La fedeltà dice che la
risposta non ha inventato *rispetto alle fonti*; non dice nulla sulla bontà
delle fonti, né sull'onestà con cui sono state riassunte. La RAG avanzata alza
il tetto del recupero e ripulisce la rosa dei candidati, ma non solleva mai
chi la usa dal dovere di scegliere bene cosa mettere nell'archivio.

Sei punti per ripercorrere la sezione più lunga del capitolo.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Quello che la ricerca non ripesca, la risposta quasi certamente non lo
  conterrà: la qualità del **recupero** è il tetto di tutto il sistema. Le
  tecniche di questa sezione lo alzano intervenendo *prima* di cercare
  (migliorare la domanda), *dopo* (riordinare i risultati) e *attorno*
  (decidere se e quando cercare).
- La domanda che scrive una persona è quasi sempre il testo peggiore da mandare
  a cercare: troppo corta, piena di sottintesi. Conviene riscriverla, oppure
  farne tre versioni diverse e unire i risultati. C'è perfino il trucco di
  cercare con una **risposta inventata** (si chiama **HyDE**
  {cite}`gao2023hyde`), che fa da esca perché somiglia ai documenti veri più di
  quanto ci somigli la domanda. Nasce per il caso in cui non si ha modo di
  addestrare la ricerca sul proprio archivio: dove quel modo c'è, si parte da
  lì e l'esca semmai si prova dopo.
- **Due stadi**: prima una scrematura rapida che tiene molti candidati, poi un
  esame lento e attento solo su quei pochi (la selezione dei curriculum e poi
  il colloquio). Il secondo stadio non serve tanto a cambiare l'ordine, quanto
  a **separare** i punteggi: quando il primo stacca nettamente gli altri, si
  può scartare il resto invece di riempire a forza i posti liberi.
- Un recupero che si **corregge**: il modello può decidere da sé se gli serve
  cercare e rileggere criticamente quello che ha trovato (è il **Self-RAG**),
  oppure tornare a cercare finché non gli basta, usando la ricerca come un
  attrezzo invece che come un passo obbligato (è il **RAG agentico**). È lo
  studente maturo all'esame a libro aperto. Ma ogni
  giro in più è tempo di attesa e denaro: la domanda giusta non è «quanti giri
  posso fare» ma «qual è il minimo che risolve questa domanda».
- **Dare un voto** vuol dire guardare tre cose separate: la risposta è davvero
  sostenuta dalle pagine citate? parla della domanda che era stata posta? le
  pagine trovate erano quelle giuste? Correggere a mano costa, e allora si
  promuove un altro modello a esaminatore: comodo, purché si ricordi che anche
  l'esaminatore ha i suoi pregiudizi e va tenuto d'occhio.
- **Fedele non vuol dire vero**: se l'archivio contiene un documento sbagliato,
  la risposta più fedele possibile a quel documento sarà sbagliata, con tanto
  di citazione impeccabile. Nessuna tecnica di questa sezione solleva chi la usa
  dal dovere di scegliere bene cosa mettere nell'archivio.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Il **recall del retriever è il vincolo dominante** del sistema RAG: il
  generatore recupera dalla memoria parametrica solo una frazione di ciò che il
  recupero ha mancato ($11{,}8\%$ su NQ nel lavoro originale
  {cite}`lewis2020retrieval`). La RAG avanzata interviene *prima* di cercare
  (migliorare la query), *dopo* (riordinare i candidati) e *attorno* (decidere
  se e quando cercare).
- **Query rewriting, espansione, multi-query**: la domanda dell'utente è la
  query peggiore per cercare. **HyDE** {cite}`gao2023hyde` genera $M$ risposte
  *ipotetiche* e cerca con la media dei loro embedding, perché vivono nello
  spazio dei documenti; la query pesa $1/(M+1)$, quindi non è un'àncora. È
  pensato per il regime **senza etichette di rilevanza**: sopra un retriever
  messo a punto sul dominio gli autori dichiarano che non è l'uso previsto, e
  quel che misurano dipende dal generatore (con uno forte, $62{,}1 \to 67{,}4$
  di NDCG@10 su DL19 ma appena $63{,}2 \to 63{,}5$ su DL20; con uno debole,
  peggiora entrambe).
- **Reranking in due stadi**: il **bi-encoder** recupera tanti candidati grezzi
  (veloce, embedding precalcolati), un **cross-encoder** riordina solo quella
  rosa ristretta (preciso ma costoso, $N$ inferenze per query).
  **ColBERT** {cite}`khattab2020colbert` è la via di mezzo, con MaxSim
  token-a-token ed embedding precalcolabili. Il guadagno vero del secondo stadio
  è la **separazione** dei punteggi, che rende possibile una soglia.
- **RAG che si corregge**: **Self-RAG** {cite}`asai2024selfrag` addestra il
  modello a decidere *se* recuperare e a **criticare** i passaggi con token di
  riflessione; il **RAG agentico** usa il recupero come strumento invocabile più
  volte (multi-hop). Ogni giro in più costa **latenza e denaro**.
- **Valutare**: **fedeltà**/*groundedness* (la risposta è supportata dai
  passaggi?), pertinenza della risposta, precision/recall del contesto.
  **RAGAS** {cite}`es2024ragas` stima le prime tre senza risposte di
  riferimento, con un LLM-giudice (la recall del contesto vuole una risposta
  annotata) e con i bias dell'**LLM-as-a-judge** del capitolo su MLOps.
- Fedeltà **non è** verità: una risposta fedele a un documento sbagliato è
  sbagliata, e una citazione corretta non salva una risposta che travisa la
  fonte.
```

`````
