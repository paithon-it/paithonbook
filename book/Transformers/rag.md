# Cercare per rispondere: retrieval e RAG

C'è una differenza che ogni studente conosce bene: l'esame a libro chiuso e
l'esame a libro aperto. A libro chiuso conta solo ciò che hai in testa, e
quando la memoria vacilla, la tentazione di improvvisare con disinvoltura è
fortissima. A libro aperto cambia tutto: non serve ricordare ogni data, serve
saper *trovare* la pagina giusta in fretta e usarla bene.

Un modello di linguaggio, così come l'abbiamo costruito in questo capitolo, dà
sempre l'esame a libro chiuso. Tutto quello che «sa» è compresso nei pesi
durante il pre-addestramento; e quando gli chiedi qualcosa che non c'è (un
fatto raro, un documento privato, una notizia successiva ai suoi dati) non
tace: completa nel modo più plausibile, con la sicurezza fluente delle
**allucinazioni** che abbiamo già messo tra i limiti strutturali di questi
modelli. C'è anche un secondo difetto, più prosaico: il libro interno si ferma
al giorno in cui è finito l'addestramento, e riscriverlo (riaddestrare il
modello) costa settimane di calcolo e cifre con molti zeri.

L'idea di questa sezione è dare al modello il libro aperto, e insegnargli a
consultarlo. Servono due mestieri diversi: **cercare** (ed è il territorio
dell'*information retrieval*, una disciplina che ha mezzo secolo di vantaggio
sui modelli di linguaggio) e **rispondere** usando ciò che si è trovato. La
combinazione dei due ha un nome che oggi si sente ovunque: **RAG**,
*Retrieval-Augmented Generation*. Ma per capirla davvero conviene partire dal
primo mestiere, il più antico.

## Trovare l'ago: l'information retrieval

Il problema è presto detto: data una richiesta (una **query**) trovare, in una
collezione che può contenere milioni di documenti, i pochi che servono. È il
problema dei motori di ricerca, ma è nato ben prima del web, nelle biblioteche
digitalizzate degli anni Sessanta e Settanta. E la prima domanda è di pura
ingegneria: come si cerca in milioni di documenti *senza leggerli tutti* a
ogni richiesta?

`````{tab} Elementare

La risposta ce l'hai già in casa: è l'**indice analitico** in fondo ai
manuali. Se in un testo di biologia di seicento pagine cerchi «fotosintesi»,
non sfogli il libro: vai in fondo, trovi *fotosintesi → pp. 214, 380* e salti
dritto lì. Qualcuno ha fatto il lavoro una volta sola (leggere tutto e
annotare dove compare ogni parola), perché tu non debba rifarlo a ogni
ricerca.

L'**indice invertito** dei motori di ricerca è la stessa idea in scala: per
*ogni* parola, la lista di *tutti* i documenti che la contengono. Alla query
«gatto muro» il motore prende le due liste, le interseca, e ottiene i
documenti che contengono entrambe le parole, senza aprirne nessuno.

Restano magari mille documenti: quali mostrare per primi? Qui tornano i pesi
del capitolo sul NLP: le parole **rare** contano di più («muro» dice più di
«il», è il principio del TF-IDF). E il buon senso aggiunge due correzioni.
Primo: se «gatto» compare dieci volte, il documento non è dieci volte più
pertinente di uno in cui compare una volta; dopo un po' il tema è chiaro, le
ripetizioni in più aggiungono briciole. Secondo: un documento lunghissimo
contiene quasi ogni parola per forza di cose, quindi la lunghezza va messa in
conto, altrimenti vincono sempre i documenti-fiume.

`````

`````{tab} Superiore

Un **indice invertito** mappa ogni termine $t$ del vocabolario nella lista dei
documenti (e delle posizioni) in cui compare: l'elaborazione di una query si
riduce a operazioni su liste ordinate, con costo legato ai documenti *che
contengono i termini della query*, non all'intera collezione.

Per l'ordinamento, il punto di riferimento da trent'anni è la funzione di
punteggio **BM25**, nata negli anni Novanta per il sistema Okapi e
sistematizzata da Robertson e Zaragoza {cite}`robertson2009probabilistic`.
È un'evoluzione del TF-IDF visto nel capitolo sul NLP:

$$
\mathrm{BM25}(q, d) \;=\; \sum_{t \,\in\, q} \mathrm{idf}(t)\cdot
\frac{\mathrm{tf}(t, d)\,(k_1 + 1)}
{\mathrm{tf}(t, d) + k_1\!\left(1 - b + b\,\dfrac{|d|}{\bar{\ell}}\right)},
$$

dove $q$ è la query e $d$ il documento; $\mathrm{tf}(t,d)$ è la frequenza del
termine $t$ in $d$ e $\mathrm{idf}(t)$ la sua rarità nella collezione, in
versione levigata,
$\mathrm{idf}(t) = \log\frac{N - \mathrm{df}(t) + 0{,}5}{\mathrm{df}(t) + 0{,}5}$,
con $N$ documenti totali e $\mathrm{df}(t)$ quelli contenenti $t$; $|d|$ è la
lunghezza del documento e $\bar{\ell}$ la lunghezza media nella collezione;
$k_1$ e $b$ sono due manopole.

La frazione è il cuore della formula, e codifica la **saturazione** della term
frequency: cresce con $\mathrm{tf}$ ma tende al tetto $k_1 + 1$. Con il valore
tipico $k_1 = 1{,}2$ e un documento di lunghezza media (la parentesi vale 1):
un'occorrenza dà punteggio $1$; dieci occorrenze danno
$\frac{10 \cdot 2{,}2}{10 + 1{,}2} \approx 1{,}96$ (non dieci volte tanto,
nemmeno il doppio, e sempre sotto il tetto). Il termine $b \in [0, 1]$
(tipicamente $b = 0{,}75$) dosa invece la **normalizzazione per lunghezza**:
penalizza i documenti più lunghi della media, che accumulano occorrenze per
pura mole.

Due parole sulla valutazione, perché torneranno: la **precision@k** è la
frazione di documenti rilevanti tra i primi $k$ restituiti, e l'**MRR** (*Mean
Reciprocal Rank*) è la media, sulle query, di $1/r$ dove $r$ è la posizione
del primo risultato corretto (vale $1$ se il sistema azzecca sempre il primo
posto).

`````

## Quando le parole non bastano: il retrieval denso

L'indice invertito ha un difetto congenito: cerca **parole**, non significati.
Nell'overview del capitolo sul NLP avevamo elencato i sinonimi tra le insidie
della lingua: «auto», «macchina» e «vettura» indicano lo stesso oggetto. Ma
per un indice invertito sono tre chiavi diverse: la query «manutenzione della
vettura» non troverà mai il documento che parla solo di «tagliando dell'auto»,
perché non condividono una sola parola. Il rimedio lo abbiamo già in mano dal
capitolo sul NLP: gli **embedding**, la mappa geometrica del significato.

`````{tab} Elementare

Pensa alla differenza tra il catalogo di una biblioteca e un libraio esperto.
Il catalogo trova solo ciò che combacia con le parole della tua richiesta. Il
libraio ha letto tutto: gli chiedi «qualcosa sull'educazione del cucciolo» e
ti mette in mano *Come allenare il tuo cane* (nessuna parola in comune, tema
identico).

Il **retrieval denso** costruisce un libraio artificiale. Ogni passaggio
dell'archivio viene trasformato in un punto sulla mappa del significato: la
stessa idea degli embedding di parole, ma per frasi intere. La domanda, quando
arriva, diventa anch'essa un punto sulla stessa mappa: i passaggi pertinenti
sono semplicemente **i punti più vicini**, parole in comune o no.

Attenzione però a non pensionare il catalogo. Se cerchi «errore E-52 della
lavatrice», vuoi *esattamente* E-52, non «un errore simile»: sui codici, sui
nomi propri, sulle sigle, il confronto letterale resta imbattibile, perché lì
il significato *è* la parola esatta. Per questo i sistemi seri usano spesso
entrambi gli approcci insieme, e fanno decidere ai risultati.

`````

`````{tab} Superiore

L'architettura standard è il **bi-encoder**: due encoder Transformer (o uno
condiviso), $E_q$ per le query ed $E_p$ per i passaggi, producono vettori in
$\mathbb{R}^d$, e la rilevanza è il prodotto scalare
$\mathrm{sim}(q, d) = E_q(q)^\top E_p(d)$, che coincide con la **similarità
del coseno**, già incontrata in *Algebra lineare*, quando i vettori sono
normalizzati. Il vantaggio computazionale è decisivo: gli embedding dei
passaggi si calcolano **una volta sola**, offline; a query time restano una
codifica e una ricerca di vicini più prossimi, che su milioni di vettori si fa
con indici approssimati (ANN); è il servizio che oggi vendono i **database
vettoriali**.

Il risultato che ha sdoganato l'approccio è **DPR** (*Dense Passage
Retrieval*) {cite}`karpukhin2020dense`: due BERT addestrati in modo
contrastivo, avvicinare le coppie domanda–passaggio corrette, allontanare i
negativi, riciclando come negativi gli altri esempi del batch (*in-batch
negatives*), che sui benchmark di question answering a dominio aperto supera
un solido BM25 di 9–19 punti assoluti di accuratezza top-20. Non senza limiti:
su termini rari, sigle ed entità fuori distribuzione il lessicale regge, e gli
ibridi BM25 + denso restano una scelta di buon senso.

Un raffinamento chiude il quadro: il bi-encoder codifica query e passaggio
*separatamente*, mentre un **cross-encoder** li concatena in un unico
Transformer (l'attenzione confronta i token dei due testi uno a uno) ed è più
accurato ma troppo costoso per scandagliare l'archivio, quindi si usa come
**reranker**: riordina i migliori $k$ candidati proposti dal retriever.

`````

## Rispondere: il question answering

Cercare non basta: il retriever restituisce passaggi, non risposte.
Trasformare un testo trovato in una risposta alla domanda è il compito che
nell'overview del capitolo sul NLP avevamo chiamato **question answering**:
uno dei quattro classici. Ha avuto persino il suo momento televisivo: nel
febbraio 2011 Watson di IBM, un sistema costruito proprio su ricerca più
analisi della domanda, batté i campioni umani del quiz *Jeopardy!*.

`````{tab} Elementare

Ci sono due modi di rispondere avendo il testo sotto gli occhi, e li conosci
dai compiti in classe. Il primo è l'**evidenziatore**: la risposta è già
scritta nel brano, basta sottolinearla. «Su cosa salta il gatto nero?»: il
brano dice «il gatto nero salta sul muro», evidenzi «sul muro», fine. È il QA
**estrattivo**. Il secondo è la **penna**: la risposta va composta con parole
tue, magari cucendo insieme più punti del testo. È il QA **generativo**, più
flessibile, ma con la libertà arriva il rischio: chi scrive di suo può anche
scrivere cose che nel testo non ci sono.

`````

`````{tab} Superiore

Il QA estrattivo è stato standardizzato da **SQuAD**
{cite}`rajpurkar2016squad`: oltre centomila domande scritte da crowdworker su
articoli di Wikipedia, dove la risposta è per costruzione uno *span* del
paragrafo dato. Il modello predice le posizioni di inizio e fine dello span
(con BERT bastano due teste softmax sulle rappresentazioni dei token) e si
valuta con *exact match* e F1 sui token. Nel giro di un paio d'anni i modelli
della famiglia BERT {cite}`devlin2019bert` hanno superato su questo benchmark
le prestazioni degli annotatori umani: un risultato vero, da leggere però per
quello che misura (trovare uno span in *un* paragrafo già dato, non rispondere
a domande nel mondo). Il QA **generativo** rimuove il vincolo dello span: un
modello seq2seq (o un decoder autoregressivo) *scrive* la risposta,
condizionata su domanda e contesto. E il QA **a dominio aperto** rimuove anche
il paragrafo dato: prima trova i passaggi in una collezione, poi rispondi, che
è esattamente la catena retrieval + lettura di questa sezione.

`````

## Il libro aperto: la RAG

Tutti i pezzi sono sul tavolo: un retriever che trova i passaggi giusti e un
generatore che sa scrivere. La **Retrieval-Augmented Generation** li mette in
fila, ed è la pipeline di {numref}`fig-rag-pipeline`: la domanda va al
retriever, che consulta l'archivio indicizzato e restituisce i passaggi più
pertinenti; domanda e passaggi vengono montati in un **prompt aumentato**; il
modello di linguaggio genera la risposta appoggiandosi ai passaggi, citandoli.
Il nome viene dal lavoro di Patrick Lewis e colleghi
{cite}`lewis2020retrieval`, che nel 2020 hanno mostrato come saldare le due
metà in un unico modello addestrabile.

```{figure} ../figures/rag-pipeline.svg
:name: fig-rag-pipeline
:alt: "Pipeline RAG in due righe: la domanda entra nel retriever, collegato a un archivio di documenti indicizzato, che restituisce i passaggi più simili con i punteggi di similarità; domanda e passaggi convergono nel prompt aumentato, che alimenta il modello di linguaggio, il quale produce una risposta con la citazione della fonte."
:width: 100%

La pipeline RAG: recuperare prima di generare. La conoscenza vive
nell'archivio (aggiornabile), la competenza linguistica nel modello.
```

`````{tab} Elementare

È lo studente all'esame a libro aperto, addestrato a usarlo bene. Arriva la
domanda; lo studente non si fida della memoria: apre l'indice, trova le due
pagine giuste, le tiene sotto gli occhi e scrive la risposta *da lì*,
annotando a margine «pag. 214». I vantaggi sono concreti. Primo: la risposta è
**controllabile**; chi corregge può andare a pagina 214 e verificare, cosa
impossibile con una risposta recitata a memoria. Secondo: il sapere è
**aggiornabile**, se esce l'edizione nuova del libro, basta sostituirla sullo
scaffale; nessuno deve rimandare lo studente a scuola. Nei sistemi reali è la
differenza tra aggiornare un archivio stanotte e riaddestrare un modello per
settimane.

Ma il libro aperto non rende infallibili. Se lo studente apre la pagina
*sbagliata* (perché l'indice l'ha ingannato o la domanda era ambigua),
scriverà una risposta sbagliata con tanto di citazione in bella vista, più
convincente proprio perché ha la fonte a margine. La RAG **sposta** il
problema dalla memoria alla ricerca: è un ottimo affare, perché la ricerca si
può ispezionare e migliorare, ma non è una garanzia di verità.

`````

`````{tab} Superiore

Nel modello originale {cite}`lewis2020retrieval` le due metà sono esplicite:
un retriever DPR fornisce $p_\eta(z \mid x)$, la probabilità di recuperare il
passaggio $z$ data la domanda $x$, e un generatore seq2seq (BART) fornisce
$p_\theta(y \mid x, z)$. La risposta marginalizza sui passaggi recuperati:

$$
p(y \mid x) \;\approx\; \sum_{z \,\in\, \text{top-}k}
p_\eta(z \mid x)\; p_\theta(y \mid x, z),
$$

dove $x$ è la domanda, $y$ la risposta generata, $z$ uno dei $k$ passaggi
recuperati, $\eta$ i parametri del retriever e $\theta$ quelli del
generatore. Il tutto si addestra end-to-end sulle sole coppie
domanda–risposta: il gradiente attraversa il generatore e l'encoder delle
query (l'indice dei passaggi, circa 21 milioni di blocchi da cento parole di
Wikipedia nell'articolo, resta congelato). La lettura concettuale è la più
duratura: il modello ha una **memoria parametrica** (i pesi) e una **memoria
non parametrica** (l'indice), e la seconda si può ispezionare, correggere e
aggiornare senza toccare la prima.

Oggi il termine RAG indica più spesso la variante leggera, senza addestramento
congiunto: recupero, poi *prompt augmentation* verso un modello già istruito
col post-training della sezione precedente; è proprio l'instruction tuning a
rendergli eseguibile una consegna come «rispondi usando solo i passaggi e cita
le fonti». I limiti però non cambiano: il recall del retriever è il **tetto**
dell'intero sistema (ciò che non viene recuperato non può entrare nella
risposta); il generatore può ignorare i passaggi o contraddirli, tanto che la
fedeltà alla fonte (*groundedness*) è oggi una metrica di valutazione a sé; e
una citazione formalmente corretta non rende vera una risposta che ne travisa
il contenuto.

`````

Vale la pena fissare il bilancio, senza hype. La RAG **mitiga** le
allucinazioni (su ciò che sta nell'archivio, il modello non deve più
inventare) ma **non le elimina**: un recupero sbagliato produce una risposta
sbagliata con le fonti in bella vista. In cambio offre due cose che i pesi da
soli non daranno mai: la **citabilità**, perché una risposta con la fonte si
può verificare e una senza fonte no; e l'**aggiornabilità**, perché quando i
documenti cambiano si reindicizza l'archivio, non si riaddestra il modello.
Quando, nella prossima sezione, troverai «il recupero di fonti esterne» tra le
mitigazioni del problema dell'affidabilità, saprai esattamente che cosa c'è
dietro, e perché è una mitigazione, non una cura.

## Un retriever denso in miniatura

Chiudiamo con il codice. Costruiamo un retriever denso completo (embedding,
similarità del coseno, top-$k$, prompt aumentato) su un archivio di sei
passaggi. Gli embedding sono fittizi ma didattici: quattro dimensioni
leggibili (gatti, muri e casa, automobili, cucina) al posto delle centinaia di
dimensioni opache di un encoder vero. Tutto il resto è identico a un sistema
reale.

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

L'output della ricerca merita un momento di attenzione:

```text
0.99  Il gatto nero salta sul muro del giardino.
0.78  Il gatto dorme accanto ai fornelli.
```

Il primo passaggio è quello giusto, con similarità quasi perfetta. Ma guarda
il secondo: parla di gatti, ed è per questo geometricamente *vicino* alla
domanda, eppure **non risponde**. È il quasi-pertinente, l'insidia tipica del
retrieval denso: la vicinanza di tema non è pertinenza alla domanda. Nei
sistemi reali è qui che interviene il reranker, e per questo la consegna nel
prompt dice «usando *solo* i passaggi»: il generatore deve appoggiarsi al
passaggio [1] e avere la disciplina di ignorare il [2].

Da questo giocattolo a un sistema di produzione i passi sono pochi e tutti già
nominati: al posto dei vettori scritti a mano, un bi-encoder addestrato alla
DPR; al posto delle sei righe, milioni di passaggi ottenuti spezzando i
documenti in blocchi (il *chunking*) e serviti da un indice ANN; al posto del
`print` finale, la chiamata a un modello istruito. La struttura (codifica,
coseno, top-$k$, prompt) è esattamente quella che hai appena eseguito.

```{admonition} Da ricordare
:class: important
- Un LLM da solo risponde **a libro chiuso**: ciò che non è nei pesi viene
  completato in modo plausibile (allucinazioni), e aggiornare i pesi costa un
  riaddestramento. Il retrieval gli apre il libro.
- L'**information retrieval** classico cerca per parole: indice invertito per
  non leggere tutto, **BM25** {cite}`robertson2009probabilistic` per ordinare
  (un TF-IDF evoluto con **saturazione** della term frequency e
  normalizzazione per lunghezza). Qualità misurata con precision@$k$ e MRR.
- Il **retrieval denso** {cite}`karpukhin2020dense` cerca per significato:
  bi-encoder, embedding di passaggi, similarità del coseno. Vince sui
  sinonimi («auto»/«vettura»), perde su sigle e nomi esatti: gli ibridi e il
  reranking con **cross-encoder** correggono il tiro.
- Il **question answering** (già tra i compiti del capitolo NLP) è
  **estrattivo** (evidenziare lo span: SQuAD {cite}`rajpurkar2016squad`) o
  **generativo** (scrivere la risposta).
- La **RAG** {cite}`lewis2020retrieval` incatena recupero, prompt aumentato e
  generazione con fonti: **mitiga** le allucinazioni ma non le elimina (il
  recall del retriever è il tetto), e offre citabilità e aggiornabilità;
  l'archivio si cambia, il modello no.
```
