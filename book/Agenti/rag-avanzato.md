# RAG avanzato: oltre il recupero ingenuo

«Su cosa salta il gatto nero?». Nella sezione «Cercare per rispondere» il
nostro retriever in miniatura aveva risposto quasi bene: al primo posto il
passaggio giusto («Il gatto nero salta sul muro del giardino») con similarità
quasi perfetta. Ma al secondo posto si era intrufolato un impostore, «Il gatto
dorme accanto ai fornelli»: vicino per tema, muto sulla domanda. Era il
**quasi-pertinente**, e non era un incidente di percorso. Era il sintomo di un
limite di fondo che avevamo enunciato senza troppi giri di parole: ciò che la
ricerca non porta a galla, la risposta non potrà mai contenerlo, per quanto
bravo sia il modello che poi la scrive. In una parola, il *recall* (la quota
dei passaggi giusti che la ricerca riesce a ripescare) è il **tetto**
dell'intero sistema {cite}`lewis2020retrieval`.

La RAG di base, così come l'abbiamo costruita, faceva tre gesti: trasformava
domanda e passaggi in punti su una mappa del significato, prendeva i pochi
passaggi più vicini alla domanda (quanti, lo decidiamo noi: diciamo i primi
cinque) e li incollava nel foglietto di istruzioni che si dà al modello, il
*prompt*, prima di fargli scrivere la risposta {cite}`karpukhin2020dense`. È
un ottimo punto di partenza e un pessimo punto di arrivo. Questa sezione
raccoglie le tecniche che spingono quel tetto più in alto: intervengono sulle
tre giunture della pipeline; **prima** di cercare (migliorando la domanda),
**dopo** aver cercato (riordinando i candidati), e **attorno** all'intero
ciclo (facendo decidere al modello se e quando cercare). Chiudiamo con la
domanda che tiene onesto tutto il resto: come si misura se un sistema RAG
funziona davvero.

```{figure} ../figures/rag-avanzato.svg
:name: fig-rag-avanzato
:alt: "Pipeline di RAG avanzato da sinistra a destra: la domanda dell'utente viene prima riscritta, poi cercata in parallelo con una ricerca densa (vettoriale) e una sparsa (per parole chiave); i due elenchi di risultati confluiscono in una fusione che produce una lista unica di candidati; un reranker li riordina per precisione e solo i primi passano al modello linguistico, che genera la risposta."
:width: 100%

La pipeline per intero. Rispetto alla RAG di base cambiano tre cose: la
domanda non va a cercare com'è arrivata, la ricerca è doppia (densa e per
parole chiave) e fra il recupero e il modello si interpone un riordino.
```

Conviene tenere {numref}`fig-rag-avanzato` sott'occhio mentre si legge il
resto: ogni sezione che segue è uno di questi blocchi, e il punto è sempre lo
stesso. Il recupero grezzo deve essere **generoso** (meglio cento candidati
mediocri che dieci scelti male, perché ciò che non entra qui è perduto per
sempre) e ciò che viene dopo deve essere **severo**, perché al modello arrivi
poco e buono.

## Migliorare la domanda: query rewriting ed espansione

```{figure} ../figures/extra-rag-spiegato.svg
:name: fig-rag-due-fasi
:alt: "Pipeline RAG divisa in due fasi. La prima, di indicizzazione, si esegue una volta sola: i documenti vengono spezzati in blocchi, ogni blocco convertito in un vettore e depositato in un archivio di vettori. La seconda, a ogni domanda: la domanda diventa un vettore, si recuperano i blocchi più vicini, e questi entrano nel prompt del modello, che risponde citando le fonti."
:width: 100%

Due fasi con tempi diversi. L'indicizzazione si paga una volta e si riusa
sempre; il recupero si paga a ogni domanda, ed è lì che si gioca la latenza.
```

La separazione di {numref}`fig-rag-due-fasi` è quella che conviene tenere in
testa leggendo il resto della sezione: le tecniche che seguono intervengono
quasi tutte nella seconda fase, dove il costo è ricorrente e i margini di
miglioramento si moltiplicano per ogni domanda ricevuta.

La prima leva è quella a cui si pensa per ultimi, perché sembra fuori dal
nostro controllo: la domanda. Il retrieval denso presume che la query
dell'utente e il passaggio giusto finiscano vicini sulla mappa del
significato. Ma la domanda che scrive una persona è quasi sempre la query
*peggiore* per cercare: breve, sbrigativa, piena di sottintesi, a volte una
sola parola. «E il tagliando?» non ha alcuna speranza di avvicinarsi al
manuale d'officina, semplicemente perché non dice abbastanza.

L'idea è interporre, tra l'utente e il retriever, un passaggio di
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

C'è un trucco ancora più sorprendente, che a prima vista sembra assurdo.
Invece di cercare con la *domanda*, cerchi con una **risposta inventata**.
Chiedi al modello: «Scrivi tu, di getto, come *sarebbe* la risposta ideale»
(anche se sbaglia qualche dettaglio) e poi cerchi i documenti veri che
somigliano a quella risposta finta. Perché funziona? Perché una domanda e la
sua risposta sono scritte in modi diversi (una chiede, l'altra afferma),
mentre due risposte sullo stesso tema si somigliano molto. La risposta
inventata fa da esca: pesca meglio delle domande i documenti che *sono* fatti
di risposte.

`````

`````{tab} Superiore

Il trucco dell'ultimo paragrafo ha un nome (**HyDE**, *Hypothetical Document
Embeddings* {cite}`gao2023hyde`) e risolve un problema geometrico preciso: la
**asimmetria** tra query e documenti. Una domanda («su cosa salta il gatto?»)
e il passaggio che la soddisfa («il gatto salta sul muro») sono testi di forma
diversa, e un bi-encoder addestrato genericamente può collocarli non così
vicini quanto vorremmo. HyDE aggira l'ostacolo: chiede a un LLM di generare
$N$ documenti *ipotetici* di risposta e interroga l'indice con la media dei
loro embedding e di quello della query,

$$
v_{\text{ricerca}} = \frac{1}{N+1} \Big( \sum_{i=1}^{N} E_p(\tilde{d}_i) + E_p(q) \Big),
\qquad \tilde{d}_i \sim \mathrm{LLM}(\,\cdot \mid q\,),
$$

dove $q$ è la domanda, le $\tilde{d}_i$ sono $N$ risposte ipotetiche
**campionate indipendentemente** dal modello condizionato sulla domanda (il
campionamento è essenziale: con una decodifica deterministica le $N$
generazioni coinciderebbero e la media collasserebbe su un solo embedding),
$E_p$ l'encoder dei passaggi (lo stesso che ha indicizzato l'archivio, e che
qui codifica anche la query, trattata come un documento in più) e
$v_{\text{ricerca}}$ il vettore con cui si interroga l'indice. L'intuizione è
che le $\tilde{d}_i$, pur potendo contenere errori fattuali, vivono nello
**spazio delle risposte** (la stessa regione dove abitano i passaggi veri) e
ci somigliano più di quanto ci somigli la domanda; tenere anche la query nella
media àncora comunque la ricerca a ciò che l'utente ha chiesto davvero. In
pratica i documenti ipotetici sono generati dall'LLM e poi **codificati** con
l'encoder dei documenti: le allucinazioni del modello non finiscono nella
risposta finale, servono solo da esca per il recupero. Le riscritture
multi-query si formalizzano invece come unione dei risultati, spesso fusi con
la **reciprocal rank fusion** {cite}`cormack2009reciprocal`: a ogni documento
si assegna il punteggio $\sum 1/(c + r)$, sommando sui ranghi $r$ che occupa
nelle diverse liste, dove $c$ è una costante di smorzamento ($60$
nell'articolo originale) che evita di sovrappesare i primissimi posti.

`````

## Riordinare i candidati: il reranking

```{figure} ../figures/vector-database.svg
:name: fig-hnsw
:alt: "Struttura HNSW a tre strati sovrapposti. In cima pochi nodi collegati da archi lunghi, che permettono di attraversare rapidamente lo spazio; scendendo, i nodi si infittiscono e gli archi si accorciano; in basso tutti i punti, con collegamenti solo fra vicini. La ricerca scende di strato in strato, raffinando via via."
:width: 92%

Come si cerca fra milioni di vettori senza confrontarli tutti. Gli strati alti
servono ad arrivare nella zona giusta con pochi salti; quelli bassi a trovare
il vicino esatto.
```

Prima di riordinare bisogna aver recuperato, e {numref}`fig-hnsw` mostra come
lo fa un archivio vettoriale vero: non confrontando la domanda con tutti i
passaggi a uno a uno, che sarebbe esatto e impraticabile, ma per scale
successive, come si cerca un indirizzo in una città (prima il quartiere, poi
l'isolato, poi il numero civico). Si rinuncia così alla garanzia di trovare
*sempre* il vicino migliore, in cambio di una ricerca incomparabilmente più
rapida; ed è un altro motivo per cui il recupero grezzo va tenuto generoso,
perché per strada qualche buon candidato si perde.

La seconda leva agisce a valle del recupero. Nella sezione «Cercare per
rispondere» avevamo già distinto due architetture: il **bi-encoder**, che
codifica domanda e passaggio *separatamente* e li confronta con un prodotto
scalare (velocissimo, perché gli embedding dei passaggi si calcolano una volta
sola) e il **cross-encoder**, che li dà in pasto *insieme* a un unico
Transformer, così che l'attenzione confronti i loro token uno a uno: molto più
accurato, ma troppo costoso per scandagliare milioni di documenti. La
soluzione, l'avevamo anticipata, è usarli **in due stadi**.

Il primo stadio, il bi-encoder, fa il grosso: spazza l'intero archivio e
restituisce non i pochi passaggi che finiranno sotto gli occhi del modello, ma
molti più candidati grezzi (cinquanta, cento)
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
cambio della garanzia di esattezza, gli autori misurano empiricamente un tempo
di ricerca che cresce come il **logaritmo** del numero di vettori, cioè
raddoppiare l'archivio costa un pugno di confronti in più, non il doppio.

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

Vediamo il reranking in azione, estendendo il retriever in miniatura della
sezione precedente. Teniamo lo stesso mini-archivio a quattro dimensioni e la
stessa domanda; aggiungiamo lo **stadio di reranking**. Il cross-encoder è
finto (non un vero Transformer) ma sensato: dove il bi-encoder vede solo il
*tema* di una frase, il nostro reranker pesa i **concetti portanti** della
domanda, distinguendo l'azione («saltare») e il bersaglio («muro») dal tema
generico («gatto»).

```python
import torch

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

# il bi-encoder e' veloce: puo' spazzare tutto l'archivio, ma e' grezzo.
# recuperiamo piu' candidati del necessario (over-retrieval): sono il
# terreno di caccia dello stadio successivo.
sim = E @ q
val, cand = torch.topk(sim, k=4)

print("Stadio 1 - bi-encoder (veloce, tutto l'archivio):")
for v, i in zip(val, cand):
    print(f"  coseno {v:.2f}  {passaggi[i]}")

# --- Stadio 2: reranking con un "cross-encoder" didattico ---
# Il bi-encoder ha collassato ogni frase in un unico vettore: cosi' un
# passaggio che condivide solo il tema "gatto" gli sembra vicino. Un
# cross-encoder legge domanda e passaggio INSIEME e distingue il tema
# dall'azione. Qui lo simuliamo con dei concetti pesati: il tema (gatto)
# conta poco, l'azione (saltare) e il bersaglio (muro) molto.
concetti_domanda = {"gatto": 1.0, "saltare": 3.0, "muro": 3.0}
concetti_passaggio = {
    0: {"gatto", "saltare", "muro"},      # gatto che SALTA sul MURO: risponde
    1: {"muro", "solaio"},
    2: {"vettura", "garage"},
    3: {"auto", "centro"},
    4: {"gatto", "dormire", "fornelli"},  # solo il tema "gatto": quasi-pertinente
    5: {"ricetta", "burro"},
}

def cross_encoder(i):
    # pertinenza = peso dei concetti della domanda davvero coperti dal passaggio
    coperti = concetti_passaggio[i]
    return sum(peso for c, peso in concetti_domanda.items() if c in coperti)

# il cross-encoder e' costoso: lo applichiamo SOLO ai candidati dello stadio 1
riordino = sorted(cand.tolist(), key=cross_encoder, reverse=True)

print("\nStadio 2 - cross-encoder (preciso, solo sui 4 candidati):")
for i in riordino:
    print(f"  pertinenza {cross_encoder(i):.1f}  {passaggi[i]}")

print("\nAl generatore vanno i primi due dopo il reranking:")
for n, i in enumerate(riordino[:2], 1):
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
  pertinenza 7.0  Il gatto nero salta sul muro del giardino.
  pertinenza 3.0  Il muro portante sostiene il solaio.
  pertinenza 1.0  Il gatto dorme accanto ai fornelli.
  pertinenza 0.0  L'auto storica sfila per il centro.

Al generatore vanno i primi due dopo il reranking:
  [1] Il gatto nero salta sul muro del giardino.
  [2] Il muro portante sostiene il solaio.
```

Ecco il punto. Al primo stadio il bi-encoder aveva messo il quasi-pertinente
«Il gatto dorme accanto ai fornelli» al **secondo posto** (coseno $0{,}78$),
appena sotto il passaggio giusto: quella vicinanza di tema che avevamo
diagnosticato come l'insidia tipica del retrieval denso. Il reranker, che
guarda l'azione e non solo il soggetto, lo declassa al terzo posto e lo
**scaccia dalla rosa** che arriva al generatore. Al suo posto sale «Il muro
portante…»: non risponde nemmeno lui, ma almeno condivide il concetto portante
«muro», e soprattutto il margine è ora netto ($7{,}0$ contro $3{,}0$ e
$1{,}0$) invece dello sfumato $0{,}99$ contro $0{,}78$. Il generatore riceve
un segnale più pulito su dove appoggiarsi. Non abbiamo alzato il recall (il
passaggio giusto era già stato recuperato) ma abbiamo alzato la **precisione
ai primi posti**, che è il secondo tetto del sistema.

## Il RAG che si corregge: Self-RAG e RAG agentico

Fin qui abbiamo migliorato una pipeline che resta **rigida**: cerca sempre,
una volta sola, poi genera. È uno schema che spreca e a volte danneggia.
Spreca quando la domanda non ha bisogno di recupero («traduci "buongiorno" in
francese» non richiede alcun archivio) e recuperare comunque infila nel prompt
passaggi irrilevanti che confondono il modello. Danneggia quando una sola
tornata di ricerca non basta, perché la risposta richiede di incrociare due
fatti che stanno in documenti diversi e che nessuna singola query trova
insieme. La terza leva rompe la rigidità: lascia che sia il **modello** a
decidere se, quando e quante volte cercare. (Esiste anche una quarta via a
quel problema di composizione, che non passa da più tornate di ricerca ma dal
cambiare la forma dell'archivio: se i fatti stanno su un grafo invece che in
paragrafi, incrociarne due è percorrere due archi. Il capitolo sulle reti
neurali su grafo la riprende parlando di knowledge graph.)

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
libro. Ogni giro in più (una riscrittura, un reranking, una seconda tornata di
ricerca, un token di riflessione) è una chiamata in più al modello: costa
**latenza** e **denaro**. Un RAG agentico che itera cinque volte è cinque
volte più lento e più caro di un recupero secco, e non sempre di qualità
cinque volte migliore. La domanda ingegneristica non è «quanti giri posso
fare», ma «qual è il numero minimo di giri che risolve *questa* classe di
domande»: sulle domande semplici, spesso, la risposta è zero.

## Valutare un sistema RAG

Tutte queste leve hanno senso solo se sappiamo dire quale migliora il sistema
e quale no, e valutare un RAG è più sottile che valutare un retriever o un
generatore separatamente, perché gli errori si annidano nelle giunture. La
risposta può essere sbagliata perché la ricerca ha mancato il documento giusto
(colpa del retriever), oppure perché il documento c'era ma il generatore l'ha
ignorato o travisato (colpa della generazione). Servono metriche che sappiano
**dove** guardare.

`````{tab} Elementare

Immagina di correggere il compito di uno studente che cita le fonti. Non basta
un voto solo: devi controllare tre cose diverse. Primo, la **fedeltà**: ciò che
ha scritto è davvero sostenuto dalle pagine che cita, o ha aggiunto di suo
spacciandolo per citazione? Secondo, la **pertinenza**: la risposta parla della
domanda che avevi fatto, o divaga su un tema vicino? Terzo, la **qualità della
ricerca**: le pagine che ha aperto erano quelle giuste, o ne ha aperte di
inutili e saltate di essenziali?

E qui torna la tentazione già vista altrove: siccome correggere a mano
migliaia di risposte costa, si promuove un altro modello a esaminatore, che
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
dell'**LLM-as-a-judge** che vedremo nel capitolo conclusivo sull'MLOps (il
*position bias*, il *verbosity bias*, l'auto-preferenza) e va perciò calibrato
contro un campione di giudizi umani, mai preso per oracolo.

`````

Un'ultima avvertenza, che è il filo rosso di tutta la RAG. La metrica più
importante (la fedeltà) non va confusa con la verità. Un sistema può essere
**perfettamente fedele** e **fattualmente sbagliato**: se l'archivio contiene
un documento errato, la risposta più fedele possibile a quel documento sarà
errata, con tanto di citazione impeccabile. E vale anche il rovescio, già
enunciato nella sezione base: una citazione formalmente corretta non rende
vera una risposta che ne **travisa** il contenuto. La fedeltà dice che la
risposta non ha inventato *rispetto alle fonti*; non dice nulla sulla bontà
delle fonti, né sull'onestà con cui sono state riassunte. La RAG avanzata alza
il tetto del recupero e ripulisce la rosa dei candidati, ma non solleva mai
chi la usa dal dovere di scegliere bene cosa mettere nell'archivio.

```{admonition} Da ricordare
:class: important
- Il **recall del retriever è il tetto** del sistema RAG: la RAG avanzata
  interviene *prima* di cercare (migliorare la domanda), *dopo* (riordinare i
  candidati) e *attorno* (decidere se e quando cercare).
- **Query rewriting, espansione, multi-query**: la domanda dell'utente è la
  query peggiore per cercare. **HyDE** {cite}`gao2023hyde` genera una risposta
  *ipotetica* e cerca con quella, perché vive nello spazio dei documenti veri
  più vicino della domanda.
- **Reranking in due stadi**: il **bi-encoder** recupera tanti candidati grezzi
  (veloce), un **cross-encoder** riordina solo quella rosa ristretta (preciso
  ma costoso).
  **ColBERT** {cite}`khattab2020colbert` è la via di mezzo, con MaxSim
  token-a-token ed embedding precalcolabili. Nel codice il reranking scaccia il
  quasi-pertinente dai primi posti.
- **RAG che si corregge**: **Self-RAG** {cite}`asai2024selfrag` addestra il
  modello a decidere *se* recuperare e a **criticare** i passaggi con token di
  riflessione; il **RAG agentico** usa il recupero come strumento invocabile più
  volte (multi-hop). Ogni giro in più costa **latenza e denaro**.
- **Valutare**: **fedeltà**/*groundedness* (la risposta è supportata dai
  passaggi?), pertinenza della risposta, precision/recall del contesto.
  **RAGAS** {cite}`es2024ragas` stima le prime tre senza risposte di
  riferimento, con un LLM-giudice (la recall del contesto vuole una risposta
  annotata) e con i bias dell'**LLM-as-a-judge** del capitolo sull'MLOps.
- Fedeltà **non è** verità: una risposta fedele a un documento sbagliato è
  sbagliata, e una citazione corretta non salva una risposta che travisa la
  fonte.
```
