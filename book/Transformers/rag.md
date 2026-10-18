# Cercare per rispondere: retrieval e RAG

C'è una differenza che ogni studente conosce bene: l'esame a libro chiuso e
l'esame a libro aperto. A libro chiuso conta solo ciò che hai in testa, e
quando la memoria vacilla, la tentazione di improvvisare con disinvoltura è
fortissima. A libro aperto cambia tutto: non serve ricordare ogni data, serve
saper *trovare* la pagina giusta in fretta e usarla bene.

Un modello di linguaggio, così come è stato costruito fin qui, dà sempre
l'esame a libro chiuso. Il suo sapere è compresso nei pesi durante il
pre-addestramento; se gli chiedi qualcosa che non c'è (un fatto raro, un
documento privato, una notizia successiva ai suoi dati) non tace: completa nel
modo più plausibile, con la sicurezza fluente delle **allucinazioni**, uno dei
limiti strutturali di questi modelli. C'è anche un secondo difetto, più
prosaico: il libro interno si ferma al giorno in cui è finito l'addestramento,
e per riscriverlo (riaddestrare il modello) servono settimane di calcolo su
migliaia di acceleratori.

L'idea del *retrieval* è dare al modello il libro aperto, e insegnargli a
consultarlo. Servono due operazioni diverse: cercare (è il territorio
dell’*information retrieval*, una disciplina che precede di decenni i modelli di
linguaggio neurali) e rispondere usando ciò che si è trovato. La combinazione
dei due si chiama **RAG**, *Retrieval-Augmented Generation*, e per capirla si
parte dalla ricerca, la più antica.

## Trovare l'ago: l'information retrieval

Il problema è presto detto: data una richiesta, trovare in una collezione che
può contenere milioni di documenti i pochi che servono. È il problema dei
motori di ricerca, ma è nato ben prima del web. Il termine *information
retrieval* compare nel 1950, in una relazione di Calvin Mooers a un convegno, e
negli anni Sessanta il gruppo di Gerard Salton confrontava già documenti e
domande come vettori, misurandone la somiglianza con il coseno
{cite}`sanderson2012history`.

Questa richiesta è una **query**, la stessa parola che nel
meccanismo di attenzione indica la domanda che una parola pone alle altre.
L'operazione è analoga: si confronta la domanda con ogni testo che potrebbe
rispondere, e si tengono i migliori. Cambia il modo: l'attenzione dà a tutte le
parole un peso, grande o piccolo, mentre il retrieval sceglie in modo netto i
primi $k$ fra milioni di voci e scarta il resto. Il primo problema è di
ingegneria: come si cerca in milioni di documenti *senza leggerli tutti* a ogni
richiesta?

`````{tab} Elementare

In fondo al manuale di biologia, dopo seicento pagine, c'è l’**indice
analitico**. Cerchi «fotosintesi» e non sfogli niente: leggi *fotosintesi →
pp. 91, 214, 380* e salti dritto lì. Qualcuno ha letto tutto una volta sola e ha
annotato dove compare ogni parola, perché tu non debba rifarlo a ogni ricerca.

Lo stesso schedario per un'intera biblioteca è l’**indice invertito** dei
motori di ricerca: una scheda per ogni parola, e sotto la lista di tutti i
libri che la contengono. Alla domanda «gatto muro» tiri fuori due schede,
confronti le due liste e tieni i titoli che compaiono in tutte e due. Non hai
aperto un libro.

Restano mille titoli. Quale apri per primo? Le schede lo dicono già in parte:
quella di «muro» è sottile, quella di «il» è spessa un dito e ci sta dentro
mezza biblioteca. Una parola che sta dappertutto non distingue niente: le
parole rare contano di più.

Poi apri un titolo e conti. «Gatto» ci compare dieci volte, ma il libro non è
dieci volte più pertinente di uno che lo nomina una volta sola: dopo un po’ il
tema è chiaro e le menzioni in più aggiungono briciole. Con la regolazione più
usata, e su un libro di lunghezza normale, una menzione vale un punto e dieci
ne valgono meno di due; per quanto si insista non si supera un tetto, che con
quella regolazione sta a due punti e due decimi, e si tocca presto.

Sullo scaffale accanto c'è l'enciclopedia in dodici volumi, che contiene
«gatto» e «muro» per forza di cose, come contiene quasi ogni parola. Se non si
sconta la lunghezza, l'enciclopedia finisce sempre in cima ai risultati. Chi
compila lo schedario ha allora due manopole, quanto scontare le ripetizioni e
quanto pesare la lunghezza, e le gira finché i risultati non lo convincono.

Sullo scaffale dei ricettari la rarità, calcolata alla lettera, si rompe.
Nella formula più diretta il peso di una parola nasce dal rapporto fra i libri
che non la contengono e quelli che la contengono: «muro» in cinque libri su
cento dà novantacinque contro cinque, e pesa molto. «Cucchiaio» in ottanta
ricettari su cento dà venti contro ottanta, un rapporto sotto uno, e la formula
lo trasforma in un peso negativo: contenere «cucchiaio» fa scendere il
punteggio invece di lasciarlo dov'è. Chi cerca «cucchiaio di burro» si
vedrebbe passare davanti le ricette in cui il cucchiaio non compare. Nei
sistemi veri la formula si ferma prima dello zero: una parola diffusissima vale
pochissimo, mai meno di niente.

Come si sa se lo schedario funziona? Si prepara un elenco di domande di cui
si conosce già la pagina giusta e si guardano due numeri: quante delle prime
dieci risposte sono utili (più sono, meglio è) e a che posto arriva la prima
buona (più è in alto, meglio è).

`````

`````{tab} Superiore

Un **indice invertito** associa a ogni termine $t$ del vocabolario la lista dei
documenti (e delle posizioni) in cui compare: l'elaborazione di una query si
riduce a operazioni su liste ordinate, con costo legato ai documenti *che
contengono i termini della query*, non all'intera collezione.

Per l'ordinamento, il punto di riferimento da trent'anni è la funzione di
punteggio **BM25**, nata negli anni Novanta per il sistema Okapi e
sistematizzata da Robertson e Zaragoza {cite}`robertson2009probabilistic`.
Discende dal modello probabilistico della rilevanza degli anni Settanta e
Ottanta, e ha la forma del {doc}`TF-IDF
</NaturalLanguageProcessing/rappresentare-testo>` con due differenze che
pesano: la frequenza del termine *satura* invece di crescere senza fine, e il
punteggio è normalizzato per la lunghezza del documento:

$$
\mathrm{BM25}(q, d) \;=\; \sum_{t \,\in\, q} \mathrm{idf}(t)\cdot
\frac{\mathrm{tf}(t, d)\,(k_1 + 1)}
{\mathrm{tf}(t, d) + k_1\!\left(1 - b + b\,\dfrac{|d|}{\bar{\ell}}\right)},
$$

dove $q$ è la query e $d$ il documento; $\mathrm{tf}(t,d)$ è il numero di
occorrenze del termine $t$ in $d$ e $\mathrm{idf}(t)$ la sua rarità nella
collezione, in versione levigata,
$\mathrm{idf}(t) = \log\frac{N - \mathrm{df}(t) + 0{,}5}{\mathrm{df}(t) + 0{,}5}$,
con $N$ documenti totali e $\mathrm{df}(t)$ quelli contenenti $t$; $|d|$ è la
lunghezza del documento e $\bar{\ell}$ la lunghezza media nella collezione;
$k_1$ e $b$ sono due parametri di regolazione. Così scritta,
l'idf diventa negativa per ogni termine presente in più di metà della
collezione (basta che $\mathrm{df}(t) > N/2$), e un contributo negativo
significa che contenere il termine *peggiora* il punteggio. È un'anomalia nota,
che le implementazioni correnti (Lucene, e quindi quasi tutti i BM25 in
produzione) evitano usando
$\log\!\big(1 + \frac{N - \mathrm{df}(t) + 0{,}5}{\mathrm{df}(t) + 0{,}5}\big)$,
sempre positiva e con lo stesso ordinamento nei casi utili.

La frazione è il cuore della formula, e codifica la **saturazione** della term
frequency: cresce con $\mathrm{tf}$ ma tende al tetto $k_1 + 1$. Con il valore
tipico $k_1 = 1{,}2$ e un documento di lunghezza media (la parentesi vale 1):
un'occorrenza dà punteggio $1$; dieci occorrenze danno
$\frac{10 \cdot 2{,}2}{10 + 1{,}2} \approx 1{,}96$ (non dieci volte tanto,
nemmeno il doppio, e sempre sotto il tetto). Il termine $b \in [0, 1]$
(tipicamente $b = 0{,}75$) dosa invece la **normalizzazione per lunghezza**:
penalizza i documenti più lunghi della media, che accumulano occorrenze per
pura mole.

Per misurare la qualità di un ordinamento, detti $\mathrm{Ril}(q)$ i documenti
rilevanti per la query e $\mathrm{Top}_k(q)$ i primi $k$ restituiti, la
**precision@k** è $|\mathrm{Ril}\cap\mathrm{Top}_k|/k$ e la **recall@k** è
$|\mathrm{Ril}\cap\mathrm{Top}_k|/|\mathrm{Ril}|$: la prima misura quanto è
pulita la lista, la seconda quanto del necessario è stato recuperato, e per un
sistema che poi legge i passaggi è la seconda a contare. L’**MRR** (*Mean
Reciprocal Rank*) è la media, sulle query, di $1/\mathrm{rank}$, dove
$\mathrm{rank}$ è la posizione del primo risultato corretto (vale $1$ se
compare sempre al primo posto). Quando la rilevanza ha dei gradi si usa
l’**nDCG@k**, $\sum_{i=1}^{k}(2^{g_i}-1)/\log_2(i+1)$ diviso per lo stesso
valore dell'ordinamento ideale, dove $g_i$ è il grado di rilevanza del
documento in posizione $i$. Per tutte e quattro, valori più alti indicano un
ordinamento migliore.

`````

## Quando le parole non bastano: cercare per significato

L'indice invertito ha un difetto congenito: cerca parole, non significati. Fra
le insidie della lingua ci sono i sinonimi ({doc}`capitolo sul NLP
</NaturalLanguageProcessing/overview>`): «auto», «macchina»
e «vettura» indicano lo stesso oggetto, ma per un indice invertito sono tre
chiavi diverse, e la query «manutenzione della vettura» non troverà mai il
documento che parla solo di «tagliando dell'auto», perché non condividono una
parola. Il rimedio sono gli embedding della {doc}`sezione su come si
rappresenta il testo </NaturalLanguageProcessing/rappresentare-testo>`, la
stessa mappa del significato della {doc}`sezione sui modelli multilingua
</Transformers/multilingua>`: ogni passaggio, e anche la domanda, diventa un
vettore di $\mathbb{R}^d$, e testi dal significato simile hanno vettori vicini.
Cercare, allora, diventa misurare distanze invece che confrontare parole.
Questo approccio si chiama **retrieval denso**, dove «denso» descrive il
vettore: non una coordinata per ogni parola della lingua, quasi tutte nulle,
come nell'indice invertito, ma qualche centinaio o qualche migliaio di
coordinate quasi tutte diverse da zero, nessuna delle quali ha un significato
isolata dalle altre; contano le distanze e le direzioni.

Su una mappa del genere alcune relazioni diventano direzioni, e si può
perfino fare dell'aritmetica.

```{figure} ../figures/word2vec-2013.svg
:name: fig-aritmetica-vettori
:alt: "Piano con i vettori di quattro parole, uomo, re, donna e regina, disposti ai vertici di un parallelogramma: la differenza fra re e uomo è all'incirca lo stesso spostamento che porta da donna a regina, e sommando quello spostamento a donna si arriva vicino a regina. Il disegno è idealizzato."
:width: 84%

Il significato come geometria, in un disegno idealizzato. Sulla mappa, andare da
«uomo» a «re» è all'incirca lo stesso spostamento che porta da «donna» a
«regina»: la relazione «la versione regale di» diventa una direzione. Nello
spazio vero le due frecce non coincidono, e il parallelogramma si chiude solo
per approssimazione.
```

Il disegno di {numref}`fig-aritmetica-vettori` è la versione ideale della
proprietà. Sui vettori veri, come mostra la {doc}`sezione sull'aritmetica del
significato </NaturalLanguageProcessing/rappresentare-testo>`, la parola più
vicina al punto d'arrivo resta *re*, e *regina* esce prima solo se si tolgono
dalla gara le tre parole della domanda. Per cercare basta la proprietà più
semplice, la vicinanza.

Con la mappa cambia anche l'unità di misura. Un documento intero non si lascia
ridurre a un punto solo, perché un manuale di quattrocento pagine, dovendo
riassumersi in un indirizzo, finirebbe per parlare di tutto e quindi di niente.
L'archivio si taglia prima in pezzi lunghi una frase o un paragrafo, e ogni
pezzo diventa un punto sulla mappa. Si chiamano **passaggi** (*chunk*, nel
gergo dei programmi); ognuno porta poche idee, spesso una sola, e quindi viene
trovato quando quell'idea è chiesta. Il taglio è un compromesso: se è troppo
fine, al pezzo manca il contesto che lo rende comprensibile; se è troppo ampio,
si torna al manuale che parla di tutto.

`````{tab} Elementare

Il catalogo di una biblioteca e un libraio esperto non trovano le stesse cose.
Il catalogo trova solo ciò che combacia con le parole della tua richiesta. Il
libraio ha letto tutto: gli chiedi «qualcosa sull'educazione del cucciolo» e
ti mette in mano *Come allenare il tuo cane* (nessuna parola in comune, tema
identico).

Il retrieval denso costruisce un libraio artificiale. Ogni passaggio
dell'archivio viene trasformato in un punto sulla mappa del significato: la
stessa idea degli embedding di parole, ma per frasi intere. La domanda, quando
arriva, diventa anch'essa un punto sulla stessa mappa: i passaggi pertinenti
sono semplicemente i punti più vicini, parole in comune o no.

Il catalogo, però, non va pensionato. Se cerchi «errore E-52 della
lavatrice», vuoi la sigla E-52, non «un errore simile»: sui codici, sui nomi
propri, sulle sigle, il confronto letterale resta lo strumento giusto, perché
lì il significato *è* la parola esatta. Per questo i sistemi seri fanno cercare
tutti e due, il catalogo e il libraio, e poi fondono le due liste di risultati
in una sola, tenendo in cima i risultati che compaiono in alto in entrambe.

`````

`````{tab} Superiore

L'architettura standard è il **bi-encoder**, che applica la struttura siamese
della {doc}`sezione su come si rappresenta il testo
</NaturalLanguageProcessing/rappresentare-testo>` a due ingressi distinti. Due
encoder Transformer (o uno condiviso), $E_q$ per le query ed $E_z$ per i
passaggi, producono vettori in $\mathbb{R}^d$, dove $d$ è la dimensione
dell'embedding, e non il documento che la stessa lettera indicava in BM25. La
rilevanza fra una query $q$ e un passaggio $z$ è data dal prodotto scalare
$\mathrm{sim}(q, z) = E_q(q)^\top E_z(z)$, che coincide con la similarità del
coseno, già incontrata nella {doc}`sezione sull'algebra lineare
</Matematica/algebra-lineare>`, quando i vettori sono normalizzati. Il
vantaggio computazionale è decisivo: gli embedding dei passaggi si calcolano
una volta sola, offline; al momento della ricerca restano solo una codifica e
il calcolo dei vicini più prossimi. La ricerca esatta ha costo $O(Nd)$ per
domanda, con $N$ passaggi di dimensione $d$: per i 21 milioni di passaggi di
Wikipedia usati negli esperimenti di DPR e $d = 768$, servono circa $1{,}6
\times 10^{10}$ moltiplicazioni a domanda, e i soli vettori, in `float32`,
occupano circa 65 GB. Su milioni di vettori si usano allora indici
*approssimati* dei vicini più prossimi (in sigla ANN, *approximate nearest
neighbor*, da non confondere con la sigla delle reti neurali): grafi navigabili
a più livelli come HNSW {cite}`malkov2020hnsw`, liste invertite con
quantizzazione del prodotto, librerie come FAISS {cite}`johnson2017billion`. Il
compromesso è fra recall, latenza e memoria: si accetta di perdere qualche
vicino vero in cambio di una ricerca sublineare, e di questo si occupano i
database vettoriali.

Il modello che ha mostrato come bastino coppie supervisionate è DPR (*Dense
Passage Retrieval*) {cite}`karpukhin2020dense`. Prima della sua introduzione, i
retriever densi superavano BM25 solo grazie a un costoso pre-addestramento
ausiliario, come ORQA {cite}`lee2019latent`. DPR impiega invece due encoder
BERT addestrati in modo contrastivo ad avvicinare le coppie domanda-passaggio
corrette e ad allontanare i negativi; a questo scopo ricicla come negativi gli
altri esempi del batch (*in-batch negatives*) e aggiunge a ogni domanda un
negativo selezionato tramite BM25. Il salto sopra un BM25 ben tarato è ampio su
quattro delle cinque raccolte di question answering a dominio aperto
dell'epoca, dagli $8{,}9$ ai $19{,}3$ punti di accuratezza top-20, cioè della
quota di domande per cui la risposta compare in almeno uno dei primi venti
passaggi. Sulla quinta, SQuAD, DPR resta sotto di $5{,}6$ punti, e gli autori
lo attribuiscono, come congettura, al modo in cui quella raccolta è nata: le
domande sono state scritte avendo il passaggio sotto gli occhi, quindi ne
ricalcano le parole, e vengono tutte da poco più di cinquecento articoli. Quel
che resta valido oltre i numeri è il perché: il denso recupera ciò che è detto
con altre parole, il lessicale ciò che è scritto con quelle esatte. Per
valutare la generalizzazione fuori dal dominio di addestramento si usa il
benchmark BEIR {cite}`thakur2021beir`, composto da diciotto raccolte
eterogenee: nei risultati dell'articolo, BM25 si rivela una base robusta; il
reranking e i modelli a interazione tardiva come ColBERT
{cite}`khattab2020colbert` ottengono in media prestazioni migliori ma a un
costo elevato, mentre i retriever densi, pur più efficienti, risultano spesso
inferiori. I sistemi ibridi interrogano tutti e due e fondono le liste con la
*reciprocal rank fusion*, $\mathrm{RRF}(d) = \sum_{m} 1/(c +
\mathrm{rank}_m(d))$, dove $d$ torna a essere il documento come in BM25, $m$
scorre le liste e $c$ è una costante vicina a 60 {cite}`cormack2009reciprocal`.
Conta la posizione in ciascuna lista e non il punteggio, che fra un BM25 e un
prodotto scalare non è confrontabile. La fusione e il reranking hanno una
sezione propria, {doc}`RAG avanzato
</Agenti/rag-avanzato>`.

Il bi-encoder codifica query e passaggio
*separatamente*, mentre un **cross-encoder** li concatena in un unico
Transformer, in cui l'attenzione confronta i token dei due testi uno a uno. È
un approccio più accurato ma troppo costoso per scandagliare l'archivio; per
questo si usa come reranker, per riordinare i migliori $k$ candidati proposti
dal retriever.

`````

## Rispondere: il question answering

Cercare non basta: il *retriever* restituisce passaggi, non risposte.
Trasformare un testo trovato in una risposta alla domanda è il *question
answering*, uno dei compiti classici del NLP. Ha avuto persino
il suo momento televisivo: nel febbraio 2011 Watson di IBM, un sistema
costruito su ricerca e analisi della domanda, batté i campioni umani del quiz
*Jeopardy!*.

`````{tab} Elementare

Ci sono due modi di rispondere avendo il testo sotto gli occhi. Il primo è
l’evidenziatore: la risposta è già scritta nel brano, basta evidenziarla. Alla
domanda «Su cosa salta il gatto nero?», se il brano dice «il gatto nero salta
sul muro», evidenzi «sul muro». È il QA **estrattivo**. Il secondo è la penna:
la risposta va composta con parole tue, magari cucendo insieme più punti del
testo. È il QA **generativo**, più flessibile, ma con la libertà arriva il
rischio: chi scrive di suo può anche scrivere cose che nel testo non ci sono.

C'è poi il caso in cui nessuno dà il brano: la domanda arriva nuda, e il primo
passo è cercare le pagine giuste. Trovate quelle, si torna a scegliere fra
evidenziatore e penna.

`````

`````{tab} Superiore

Il QA estrattivo è stato standardizzato da **SQuAD**
{cite}`rajpurkar2016squad`: oltre centomila domande scritte da crowdworker su
articoli di Wikipedia, dove la risposta è per costruzione uno *span* del
paragrafo dato. Il modello predice le posizioni di inizio e fine dello span
(con BERT {cite}`devlin2019bert` bastano due vettori appresi che danno a ogni
token un punteggio di inizio e uno di fine, normalizzati da una softmax) e si
valuta con *exact match* e F1 sui token. Nel 2018 i modelli neurali, BERT
compreso, hanno superato su questo benchmark le prestazioni degli annotatori
umani: un risultato significativo, da leggere però sapendo che
cosa misura (individuare uno span in un singolo paragrafo fornito a priori, in
cui la risposta è sempre presente, non equivale a rispondere a domande aperte).
Il QA generativo rimuove il vincolo dello span: un modello seq2seq (o un
decoder autoregressivo) *scrive* la risposta condizionata a domanda e contesto.
Il QA a dominio aperto rimuove anche il paragrafo dato: prima si recuperano i
passaggi da una collezione, poi si legge il materiale trovato, e su questa
catena poggia la RAG.

`````

## Il libro aperto: la RAG

Tutti i pezzi sono sul tavolo: un retriever che trova i passaggi giusti e un
modello di linguaggio che sa scrivere. La Retrieval-Augmented Generation,
«generazione aumentata dal recupero», li mette in fila nella catena di
{numref}`fig-rag-pipeline`. La domanda va al retriever, che consulta
l'archivio e restituisce i passaggi più pertinenti. Domanda e passaggi vengono
poi uniti in un unico testo, passato al modello come contesto: è il
**prompt aumentato**. Il modello scrive la risposta appoggiandosi a quei passaggi e
citandoli. Il nome viene dal lavoro di Patrick
Lewis e colleghi {cite}`lewis2020retrieval`, che nel 2020 hanno mostrato come
saldare le due metà in un unico modello addestrabile.

```{figure} ../figures/rag-pipeline.svg
:name: fig-rag-pipeline
:alt: "Pipeline RAG in due righe: la domanda entra nel retriever, collegato a un archivio di documenti indicizzato, che restituisce i passaggi più simili con i punteggi di similarità; domanda e passaggi convergono nel prompt aumentato, che alimenta il modello di linguaggio, il quale produce una risposta con la citazione della fonte."
:width: 100%

La pipeline RAG: recuperare prima di generare. La conoscenza vive
nell'archivio (aggiornabile), la competenza linguistica nel modello.
```

`````{tab} Elementare

È lo studente all'esame a libro aperto, addestrato a usarlo bene. Arriva la
domanda; lo studente non si fida della memoria: apre l'indice, trova le tre
pagine giuste, le tiene sotto gli occhi e scrive la risposta *da lì*,
annotando a margine «pag. 214». Non apre nemmeno le altre seicento pagine,
perché l'indice dice che non c'entrano. Le tre pagine stanno tutte sullo
stesso banco, ed è lo studente a decidere quale conti mentre scrive. Ogni
pagina in più è altro da leggere, e quando la pila è alta lo studente legge
meglio quelle in cima e in fondo che quelle in mezzo.

I vantaggi sono concreti. La risposta è controllabile: chi corregge può
andare a pagina 214 e verificare, cosa impossibile con una risposta recitata
a memoria. Il sapere è aggiornabile: se esce l'edizione nuova del libro basta
sostituirla sullo scaffale, e nessuno deve rimandare lo studente a scuola. Nei
sistemi reali è la differenza tra aggiornare un archivio stanotte e
riaddestrare un modello per settimane.

Il libro aperto, però, non rende infallibili: le cose vanno storte in tre punti
diversi. Lo studente può aprire la pagina *sbagliata*, perché l'indice l'ha
ingannato o perché la domanda era ambigua; scrive allora una risposta sbagliata
con la citazione in bella vista, più convincente perché ha la fonte a margine.
Può non trovare la pagina che serviva: quel contenuto non entrerà nella
risposta presa dal libro, e se risponde lo fa a memoria senza dichiararlo,
cioè è tornato di nascosto all'esame a libro chiuso. Può infine avere la pagina
giusta sotto gli occhi e scrivere altro, o piegarne il senso: «pag. 214» resta
una citazione esatta, ma la frase sopra non corrisponde al testo di pagina 214.
Spostare il problema dalla memoria alla ricerca resta un buon affare, perché
una ricerca si può ispezionare e migliorare; il problema però non sparisce, e
un pezzo resta dov'era, nella penna di chi scrive.

`````

`````{tab} Superiore

Nel modello originale {cite}`lewis2020retrieval` le due metà sono esplicite: un
retriever DPR fornisce $p_\eta(z \mid x)$, la probabilità di recuperare il
passaggio $z$ data la domanda $x$, e un generatore seq2seq (BART) fornisce
$p_\theta(y \mid x, z)$. Il prodotto scalare del bi-encoder diventa quella
probabilità nel modo consueto, passando per l'esponenziale e normalizzando su
tutto l'archivio: $p_\eta(z \mid x) \propto \exp\!\big(E_z(z)^\top
E_q(x)\big)$. (Qui $\eta$ indica i parametri del retriever, non il tasso di
apprendimento come altrove nel libro, e la domanda è $x$ invece
che $q$.) La risposta marginalizza sui passaggi recuperati, nella variante
che l'articolo chiama RAG-Sequence, quella che usa lo stesso passaggio per
tutta la risposta (nell'altra, RAG-Token, la somma viene ricalcolata per ogni
token, e parole diverse possono venire da passaggi diversi):

$$
p(y \mid x) \;\approx\; \sum_{z \,\in\, \text{top-}k}
p_\eta(z \mid x)\; p_\theta(y \mid x, z),
$$

dove $x$ è la domanda, $y$ la risposta generata, $z$ uno dei $k$ passaggi
recuperati, $\eta$ i parametri del retriever e $\theta$ quelli del generatore.
Il «circa» segnala che la somma esatta si estenderebbe a tutti i passaggi
dell'archivio, mentre qui si ferma ai primi $k$. L'articolo la chiama
approssimazione top-$k$ e non la giustifica: quanto pesi la coda esclusa
dipende dalla scala dei prodotti scalari del retriever, e non è detto sia poco.
La somma troncata viene poi rinormalizzata sui soli $k$ passaggi tenuti
(nell'implementazione di Hugging Face con una softmax sui rispettivi punteggi),
e il risultato è di nuovo una distribuzione, che coincide con quella esatta
solo se la coda scartata ha peso trascurabile.

Il modello si addestra end-to-end sulle sole coppie domanda-risposta: il
gradiente attraversa il generatore e l'encoder delle query (l'indice dei
passaggi, che nell'articolo conta circa 21 milioni di brani da cento parole di
Wikipedia, resta congelato). La lettura concettuale è la più duratura: il
modello ha una **memoria parametrica** (i pesi) e una **memoria non
parametrica** (l'indice), e la seconda si può ispezionare, correggere e
aggiornare senza toccare la prima.

Il testo recuperato può entrare nel sistema in punti diversi, ciascuno col suo
costo. In Fusion-in-Decoder {cite}`izacard2021leveraging` ogni passaggio è
codificato insieme alla domanda, separatamente dagli altri, e il decoder applica
l'attenzione alla concatenazione delle codifiche: l'encoder ha un costo lineare nel
numero di passaggi, e la qualità continua a crescere con quel numero. RETRO
{cite}`borgeaud2022improving` porta il recupero dentro il modello, con
un'attenzione incrociata su blocchi presi da una base di duemila miliardi di token;
kNN-LM {cite}`khandelwal2020generalization` lo porta all'uscita, interpolando la
distribuzione del modello con quella dei vicini più prossimi nello spazio degli
embedding, senza riaddestrare. A parte sta l'addestramento del retriever:
REALM {cite}`guu2020realm`, uscito due mesi prima di DPR, lo pre-addestra con il
linguaggio mascherato, propagando il gradiente attraverso la ricerca.

Oggi il termine RAG indica più spesso la variante leggera, che non addestra
insieme il retriever e il generatore: recupero, poi *prompt augmentation*
verso un modello già istruito col {doc}`post-training
<post-training>`; è l'instruction tuning a rendergli eseguibile una consegna
come «rispondi usando solo i passaggi e cita le fonti». In quella variante
sparisce anche la marginalizzazione: i passaggi finiscono tutti nello stesso
prompt, il generatore li vede insieme, e a pesarne la pertinenza è lui, dal
contenuto e non dal punteggio del recupero, che però decide ancora quali
entrano e in che ordine. L'ordine conta: con molti passaggi nel contesto le
prestazioni sono spesso migliori quando l'informazione rilevante sta all'inizio
o alla fine, e calano quando sta in mezzo {cite}`liu2024lost`. Ogni passaggio
in più occupa contesto: con $k$ passaggi di $n_p$ token ciascuno il prompt si
allunga di $k\,n_p$ token, e $k$ diventa il parametro che mette in tensione la
recall con il costo e il rumore. I limiti non cambiano. Il tetto delle risposte
fondate e citabili è la recall del retriever: la quota dei passaggi pertinenti
che la ricerca riesce a ripescare, definita come per un classificatore (la
{doc}`sezione sulle metriche
</MachineLearning/metriche>` la tratta per esteso), con i passaggi pertinenti
dell'archivio al posto dei positivi. Un passaggio non recuperato non può
entrare nella risposta *a partire dai documenti*; il modello può sempre
attingere ai propri pesi, ma allora è tornato all'esame a libro chiuso, e senza
dirlo. Il generatore, poi, può ignorare i passaggi o contraddirli, tanto che la
fedeltà alla fonte (*groundedness*) è oggi una metrica di valutazione a sé; e
una citazione formalmente corretta non rende vera una risposta che ne travisa
il contenuto.

`````

Il modello originale di Lewis e colleghi tiene insieme le due metà con una
somma: ogni passaggio recuperato produce una risposta indipendente, e le
risposte si sommano pesate dal punteggio che il recupero ha assegnato al
passaggio ({numref}`fig-somma-pesata-passaggi`). I passaggi non si vedono fra
loro, e il peso di ciascuno è deciso prima che il generatore scriva una parola.
La variante di oggi mette invece tutti i passaggi nello stesso prompt e lascia
al modello il compito di distinguere i passaggi utili da quelli solo
apparentemente pertinenti.

```{figure} ../figures/somma-pesata-passaggi.svg
:name: fig-somma-pesata-passaggi
:alt: "La domanda è «Dove avviene la fotosintesi?». A sinistra tre riquadri, uno per pagina recuperata: pag. 214 in terracotta con peso 0,600, pag. 380 in teal con peso 0,220, pag. 91 in ocra con peso 0,180. Dentro ogni riquadro tre barre, una per risposta possibile: pag. 214 dà 0,700 a «nelle foglie», 0,200 a «nel fusto» e 0,100 a «nelle radici»; pag. 380 dà 0,750 a «nel fusto», 0,150 a «nelle foglie» e 0,100 a «nelle radici»; pag. 91 dà 0,700 a «nel fusto», 0,200 a «nelle radici» e 0,100 a «nelle foglie». Le tre barre di ogni riquadro crescono tutte nello stesso momento, e le barre dei pesi compaiono già fatte. A destra le tre risposte sommate, ciascuna una barra composta di tre segmenti, uno per pagina: «nelle foglie» vale 0,471, «nel fusto» 0,411, «nelle radici» 0,118, e sotto la prima è scritto il conto per esteso, 0,60×0,70 più 0,22×0,15 più 0,18×0,10. Vince «nelle foglie», benché due pagine su tre preferiscano «nel fusto», perché il segmento di pag. 214 è più lungo degli altri due messi insieme; e «nel fusto», che perde, si porta via lo stesso quattro decimi."
:width: 100%

Tre passaggi recuperati (le tre pagine), tre risposte parziali che nascono
insieme e non si correggono a vicenda, e una risposta sola che le somma
pesandole. Un voto a maggioranza direbbe «nel fusto», che due passaggi su tre
preferiscono; la somma pesata dà «nelle foglie»: il primo passaggio pesa più
degli altri due messi insieme, e la sua preferenza prevale, 0,471 contro 0,411.
La risposta che perde si porta via lo stesso quattro decimi: pesare non è
filtrare. I numeri sono inventati apposta, per far vedere il meccanismo.
```

La RAG mitiga le allucinazioni: per i contenuti presenti nell'archivio il
modello non deve più inventare, e nel dialogo fondato su conoscenze i modelli
con recupero allucinano meno {cite}`shuster2021retrieval`. Non le elimina: un
recupero sbagliato produce una risposta sbagliata con le fonti in bella vista.
In cambio offre due proprietà che i soli pesi non garantiscono in modo
affidabile: la citabilità, perché una risposta con la fonte si può verificare e
una senza fonte no, e l'aggiornabilità, perché quando i documenti cambiano si
ricostruisce l'indice dell'archivio, non si riaddestra il modello.

## Un retriever denso in miniatura

Un retriever denso completo, su un archivio di sei passaggi, si costruisce in
poche righe: si calcolano gli embedding, si misura quanto ciascuno è vicino
alla domanda, si tengono i due migliori, si monta il prompt aumentato. Gli
embedding qui sono scritti a mano e hanno quattro sole coordinate, ognuna con
un significato leggibile (quanto il passaggio parla di gatti, di muri e casa, di
automobili, di cucina), al posto delle centinaia di coordinate opache che
produrrebbe un modello vero. Il resto è identico a un sistema in funzione.

La vicinanza si misura con la similarità del coseno, $\cos\theta =
\mathbf{a}^\top\mathbf{b} / (\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert)$,
che dice quanto due vettori puntano nella stessa direzione: vale $1$ per due
vettori allineati, $0$ per due perpendicolari e $-1$ per due opposti, e con i
vettori normalizzati coincide con il prodotto scalare. Qui le coordinate sono
tutte positive, quindi il coseno sta fra $0$ e $1$; con un modello vero scende
fino a $-1$, dove i valori negativi indicano direzioni opposte, non un errore.
Il coseno va inteso come una misura di direzione, non come una percentuale di
somiglianza.

Il programma segue la variante leggera e stampa i due passaggi trovati con il
loro coseno, poi il prompt aumentato completo, cioè il testo che il modello si
troverà davanti.

```python
import torch

# mini-archivio: sei passaggi e i loro embedding "didattici"
# dimensioni: [gatti, muri/casa, automobili, cucina]
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

# normalizza le righe: il prodotto scalare diventa similarita' del coseno
E = E / E.norm(dim=1, keepdim=True)

# la domanda, codificata dallo stesso "encoder": parla di gatti e muri
domanda = "Su cosa salta il gatto nero?"
q = torch.tensor([0.9, 0.7, 0.0, 0.0])
q = q / q.norm()

sim = E @ q                       # coseno con tutti i passaggi in un colpo
val, idx = torch.topk(sim, k=2)   # i due passaggi piu' vicini

for v, i in zip(val, idx):
    print(f"{v:.2f}  {passaggi[i]}")

# assemblaggio del prompt aumentato
contesto = "\n".join(f"[{n + 1}] {passaggi[i]}" for n, i in enumerate(idx))
prompt = (
    "Rispondi usando solo i passaggi seguenti e cita le fonti.\n\n"
    f"{contesto}\n\nDomanda: {domanda}\nRisposta:"
)
print(prompt)
```

```text
0.99  Il gatto nero salta sul muro del giardino.
0.78  Il gatto dorme accanto ai fornelli.
Rispondi usando solo i passaggi seguenti e cita le fonti.

[1] Il gatto nero salta sul muro del giardino.
[2] Il gatto dorme accanto ai fornelli.

Domanda: Su cosa salta il gatto nero?
Risposta:
```

Il primo passaggio è quello giusto, con una somiglianza di $0{,}99$, quasi
perfetta. Il secondo, a $0{,}78$, parla di gatti e per questo sta *vicino* alla
domanda sulla mappa, eppure non risponde: è il quasi-pertinente, l'insidia
tipica della ricerca per significato, dove la vicinanza di tema non è
pertinenza alla domanda. Nei sistemi reali interviene un secondo lettore, più
lento e più accurato, che riesamina uno per uno i pochi candidati e li rimette
in ordine: il **reranker**. Senza di lui spetta al generatore distinguerli: la
consegna «usando solo i passaggi» gli vieta di uscire dall'archivio, ma tra
[1] e [2] deve scegliere dal contenuto, e il modello può
citare il passaggio sbagliato.

I passi da questo giocattolo a un sistema vero, di quelli che servono utenti
reali, sono pochi. Al posto degli embedding scritti a mano serve un modello
addestrato a produrli: il riferimento classico è **DPR** (*Dense
Passage Retrieval*), formato da due encoder addestrati insieme a mettere vicine le
domande e i passaggi che le soddisfano. Al posto delle sei righe subentrano
milioni di passaggi, tagliati da un archivio vero e serviti da un indice che sa
trovare i punti vicini fra milioni di punti senza confrontarli tutti: si
accontenta dei quasi-vicini in cambio della velocità, e per questo si chiama
approssimato. Al posto del `print` finale c'è la chiamata a un modello
istruito. La struttura (si codifica, si misura il coseno, si tengono i primi
$k$, si monta il prompt) è quella del programma.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Un modello risponde a libro chiuso: un fatto che non ha in testa se lo
  inventa in modo plausibile, e per rifargli studiare il libro servono
  settimane. Il recupero gli apre il libro.
- Cercare per parole è il metodo antico: l’indice analitico dice
  subito in quali pagine compare una parola, senza doverle sfogliare tutte;
  poi si ordinano i risultati dando più peso alle parole rare, contando le
  ripetizioni sempre meno man mano che aumentano, e penalizzando i
  documenti-fiume.
- Cercare per significato è il metodo nuovo: ogni passaggio diventa un
  punto su una mappa dove i testi dal significato simile stanno vicini,
  e la domanda diventa un punto sulla stessa mappa. Vince sui sinonimi
  («auto» trova «vettura»), perde sui codici e sui nomi esatti, dove serve la
  parola precisa: per questo i sistemi seri usano tutti e due.
- Rispondere avendo il testo sotto gli occhi si fa in due modi:
  l'evidenziatore (la risposta è già scritta, basta evidenziarla) o la penna
  (la risposta si compone con parole proprie, più libera e più rischiosa).
- La RAG unisce le due operazioni: prima si cerca, poi si risponde con i
  passaggi sotto gli occhi, citando. I modi sono due: quello di oggi mette
  tutti i passaggi nello stesso testo e lascia scegliere al modello; quello
  originale scriveva una risposta per passaggio e poi le sommava, contando
  ciascuna per quanto la ricerca l'aveva messa in cima. Attenua le risposte
  inventate ma non le elimina (se si apre la pagina sbagliata, la risposta è
  sbagliata con tanto di fonte in bella vista); in cambio si può verificare, e
  si aggiorna cambiando l'archivio invece del modello.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Un LLM risponde a libro chiuso: un fatto assente dai pesi viene completato in
  modo plausibile (allucinazioni), e per aggiornare i pesi occorre
  riaddestrare. Il retrieval gli apre il libro.
- L’information retrieval classico cerca per parole: indice invertito per non
  leggere tutto, BM25 {cite}`robertson2009probabilistic` per ordinare (funzione
  derivata dal modello probabilistico della rilevanza, con la forma di un
  TF-IDF, saturazione della term frequency e normalizzazione per lunghezza).
  Qualità misurata con precision@$k$, recall@$k$ (il tetto per un sistema che
  poi legge i passaggi), MRR e nDCG@$k$.
- Il retrieval denso {cite}`karpukhin2020dense` cerca per significato:
  bi-encoder, embedding di passaggi, similarità del coseno, indici
  approssimati per non confrontare tutto. Vince sui sinonimi
  («auto»/«vettura»), perde su sigle, nomi esatti e domini lontani da quello di
  addestramento {cite}`thakur2021beir`: gli ibridi e il reranking con
  cross-encoder correggono il tiro.
- Il question answering (già tra i compiti del capitolo NLP) è
  estrattivo (evidenziare lo span: SQuAD {cite}`rajpurkar2016squad`) o
  generativo (scrivere la risposta).
- La RAG {cite}`lewis2020retrieval` incatena recupero e generazione, e nella
  forma originale marginalizza sui passaggi, cioè somma le risposte pesandole
  con la probabilità di recupero; la variante leggera di oggi salta la somma e
  monta invece il prompt aumentato. Mitiga le allucinazioni ma non le elimina (la
  recall del retriever è il tetto), e offre citabilità e aggiornabilità;
  l'archivio si cambia, il modello no.
```
`````

La {doc}`sezione sulle tendenze e sui limiti <tendenzefuture>` colloca il
recupero di fonti esterne fra le mitigazioni delle allucinazioni, accanto alla
verifica e alla calibrazione.
