# Dalla softmax alla ricorrenza

Il capitolo sui Transformer si chiude con un paradosso. Il meccanismo che ha
vinto la partita (l'attenzione, che lascia ogni parola libera di guardare
tutte le altre) vince *proprio perché* guarda tutte le altre; ed è esattamente
questo il suo conto da pagare. Sono i due conti di cui si è detto aprendo il
capitolo: il lavoro che cresce col quadrato della lunghezza (gli informatici lo
scrivono $O(n^2)$, che è solo un modo compatto di dirlo) e la **KV cache** che
si allunga a ogni parola generata, il "segnalibro" che nel capitolo sui
Transformer conserva le chiavi e i valori già calcolati per non rifarli a ogni
passo.

Le reti ricorrenti che abbiamo studiato prima dei Transformer non avevano
nessuno di questi due problemi: leggevano in fila, a costo lineare, con uno
stato di dimensione *fissa* che non cresceva mai. Le avevamo abbandonate per
un difetto altrettanto grave, la **sequenzialità**: ogni passo deve aspettare
quello prima, e le schede grafiche (le GPU), che sono fatte per macinare
montagne di conti tutti insieme, restano a guardare. La domanda di questo
capitolo è se si possano avere entrambe le cose: distribuire il lavoro su tanti
conti simultanei quando il modello impara, e pagare poco, sempre lo stesso,
quando scrive. La risposta parte da un'osservazione che si fa alle medie, e la
vediamo subito.

## Il trucco del kernel: spezzare la softmax

L'ostacolo ha un nome preciso, ed è la softmax. Per calcolare l'attenzione
bisogna prima misurare quanto ogni parola somiglia a ogni altra, e riempire con
quei numeri una tabella che ha una riga e una colonna per ciascuna parola:
mille parole, un milione di caselle. La tabella grande serve perché la softmax
*mescola*, cioè spartisce fra tutte: quanto la parola numero uno dà retta alla
numero due dipende anche da tutte le altre, e finché è così nessuna casella si
può calcolare per conto suo.

La via d'uscita è cambiare il modo di misurare la somiglianza. Serve una misura
che, invece di nascere dal confronto di una coppia, si **spezzi in due pezzi
indipendenti**: uno che riguarda solo chi fa la domanda, uno che riguarda solo
chi risponde. (Ogni parola fa tutte e due le parti: interroga le altre con la
propria domanda, e alle domande delle altre risponde con la propria etichetta.)
Se la somiglianza si spezza così, i conti si possono riordinare, e riordinarli
li fa crollare di numero.

`````{tab} Elementare

Il riordino è il raccoglimento a fattor comune, aritmetica di seconda media, e
si vede meglio con due conti affiancati.

Immagina tre parole che rispondono, ciascuna col proprio contributo: $2$, $5$ e
$9$. Una quarta parola fa la domanda, e quello che riceve è la somma dei tre
contributi, ciascuno moltiplicato per quanto le somiglia.

Se la somiglianza **non si spezza**, ogni peso è un numero a sé, che nasce dal
confronto di quella coppia lì: il conto è $3\times 2 + 7\times 5 + 4\times 9$.
Quei tre pesi ($3$, $7$, $4$) sono tre caselle della tabella grande, e vanno
calcolati uno per uno; con mille parole che fanno mille domande, sono un
milione di caselle.

Se invece la somiglianza **si spezza**, il peso diventa il prodotto di due
numeri, uno di chi chiede e uno di chi risponde. Chi chiede porta allora sempre
lo stesso numero, e il conto si raccoglie: $3\times 2 + 3\times 5 + 3\times 9 =
3\times(2+5+9) = 3\times 16$. Il bello è che la parte fra parentesi non dipende
da chi chiede: la parola dopo farà $7\times 16$, quella dopo ancora
$4\times 16$. Quel $16$ si calcola **una volta per tutte**, e le caselle della
tabella grande non si scrivono mai.

La stessa cosa, raccontata come una scena. Riprendiamo l'assemblea del capitolo
sui Transformer: mille parole sono mille persone, e farle parlare tutte con
tutte sono quasi mezzo milione di conversazioni (le caselle della tabella erano
un milione perché ogni coppia ci compare due volte, una per la domanda che il
primo fa al secondo e una per quella che il secondo fa al primo; le
conversazioni, contate una volta sola, sono la metà). Il costo esplode perché
ogni coppia va gestita a parte.

Invece di metterli tutti a chiacchierare fra loro, teniamo un **registro
riassuntivo**: è la somma dentro la parentesi di poco fa, fatta una volta per
tutte. Ogni persona che entra ci somma il proprio contributo («io ho questa
etichetta e porto questa informazione»), sempre nelle stesse caselle, e chi
deve farsi un'idea non interroga più tutti a uno a uno: legge il registro, già
bell'e riassunto. Le persone restano mille, ma il registro è uno solo, e lo si
aggiorna una volta per ciascuna: mille aggiornamenti invece di mezzo milione di
conversazioni.

Il trucco sta tutto nel trovare un modo di riassumere che non perda ciò che
serve, e questo è il punto delicato: un riassunto, per quanto ben fatto, tiene
meno cose di un archivio completo. Il conto lo pagheremo più avanti in questa
pagina, quando vedremo che cosa succede a un registro su cui si continua a
scrivere senza mai cancellare. In matematica quel modo di misurare la
somiglianza si chiama **kernel**, ed è il nome che dà il titolo alla sezione.

`````

`````{tab} Superiore

Nell'attenzione dei Transformer {cite}`vaswani2017attention` l'uscita (non
normalizzata) per la query $i$ è una somma dei value pesati dalla somiglianza
esponenziale $\exp(\mathbf{q}_i^\top \mathbf{k}_j)$ (il fattore di scala $1/\sqrt{d}$ lo
consideriamo assorbito in query e chiavi). Il guaio è che quell'esponenziale
*non si spezza*: non esiste una fattorizzazione esatta **a dimensione finita**
in un prodotto di una funzione della sola $\mathbf{q}_i$ per una funzione della sola
$\mathbf{k}_j$. Una feature map che lo riproduca c'è, ma con infinite componenti, e in
pratica la si può solo approssimare: è la strada delle *random features* del
Performer {cite}`choromanski2021performer`. Restando esatti, quindi,
l'esponenziale va valutato per ogni coppia $(i,j)$: la matrice $n \times n$.

L'idea di Katharopoulos e colleghi {cite}`katharopoulos2020transformers` è
sostituire la somiglianza con una che *si spezza*, cioè un prodotto scalare fra
versioni trasformate di query e key:

$$
\text{sim}(\mathbf{q}, \mathbf{k}) = \phi(\mathbf{q})^\top \phi(\mathbf{k}),
$$

dove $\phi$ è una **feature map** applicata a ciascun vettore, riga per riga
sulle matrici di query e chiavi. In generale è una
$\phi: \mathbb{R}^d \to \mathbb{R}^C$, che può cambiare dimensione: $C$ è la
dimensione dello spazio di feature ed è un parametro libero (il Performer, per
dire, ne sceglie 256 di *random features*), e da essa dipende la taglia della
memoria. Con questa scelta l'uscita per la query $i$ diventa

$$
\mathbf{o}_i = \sum_{j} \big(\phi(\mathbf{q}_i)^\top \phi(\mathbf{k}_j)\big)\, \mathbf{v}_j
    = \sum_{j} \mathbf{v}_j \big(\phi(\mathbf{k}_j)^\top \phi(\mathbf{q}_i)\big)
    = \Big(\underbrace{\sum_{j} \mathbf{v}_j\, \phi(\mathbf{k}_j)^\top}_{\mathbf{S}}\Big)\, \phi(\mathbf{q}_i)
    = \mathbf{S}\,\phi(\mathbf{q}_i),
$$

dove il passaggio chiave è la semplice **associatività** del prodotto: lo
scalare $\phi(\mathbf{k}_j)^\top \phi(\mathbf{q}_i)$ si può portare a destra di $\mathbf{v}_j$, e allora
la somma su $j$ si stacca dalla query e si condensa in un'unica matrice
$\mathbf{S} = \sum_j \mathbf{v}_j\, \phi(\mathbf{k}_j)^\top \in \mathbb{R}^{d_v\times C}$ (il «registro»
chiave→valore). Qui $\mathbf{q}_i, \mathbf{k}_j, \mathbf{v}_j$ sono i vettori query, key e value del
token e $d_v$ è la dimensione del value. Da qui in avanti, per semplicità,
assumiamo key, query e value della stessa dimensione $d$ e una $\phi$ che non
cambia dimensione ($C = d$), così che lo stato sia una $d \times d$: è il caso
della feature map che sceglieremo fra poco, ma non il caso generale.

Le due strade hanno costi diversissimi. Calcolare tutti i prodotti
$\phi(\mathbf{q}_i)^\top \phi(\mathbf{k}_j)$ è la matrice $n \times n$, costo $O(n^2 d)$;
costruire $\mathbf{S}$ una volta e applicarla a ogni query costa $O(n d^2)$: una
matrice $d \times d$ al posto di una $n \times n$. Quando $n \gg d$ (sequenze
lunghe), la seconda vince nettamente, ed è il passaggio da $O(n^2 d)$ a
$O(n d^2)$, cioè da quadratico a **lineare** nella lunghezza (lo stesso
$O(n d^2)$ delle ricorrenti che avevamo incontrato nel confronto fra
Transformer e RNN). In forma matriciale compatta, per tutte le query insieme,
è l'identità

$$
\big(\phi(\mathbf{Q})\,\phi(\mathbf{K})^\top\big)\,\mathbf{V} = \phi(\mathbf{Q})\,\big(\phi(\mathbf{K})^\top \mathbf{V}\big),
$$

la stessa $\text{softmax}(\mathbf{Q}\mathbf{K}^\top)\mathbf{V}$ del capitolo sui Transformer con la softmax
tolta di mezzo: senza di lei il prodotto si può ri-associare, e conviene fare
prima $\phi(\mathbf{K})^\top \mathbf{V}$, la matrice piccola. Un'avvertenza sulla convenzione,
che in un capitolo dedicato a *da che parte* moltiplica un fattore non è
pedanteria: qui i token stanno sulle righe di $\mathbf{Q}$, $\mathbf{K}$ e $\mathbf{V}$, quindi la matrice
piccola $\phi(\mathbf{K})^\top \mathbf{V}$ è $d \times d_v$, cioè $\mathbf{S}^\top$ e non $\mathbf{S}$.

Resta da scegliere $\phi$. Serve una feature map che dia somiglianze positive
(perché i pesi si comportino come quelli di una media), e Katharopoulos et al.
propongono la più semplice che funzioni:

$$
\phi(x) = \operatorname{elu}(x) + 1,
$$

applicata componente per componente e sempre maggiore di zero (in aritmetica
esatta: per $x<0$ vale $\exp(x)$, ma il calcolo passa da $-1 + \exp(x)$, e in
`float32` quella somma arrotonda a $-1$ appena $\exp(x)$ scende sotto
$2^{-25}$, cioè da $x = -25\ln 2 \approx -17{,}33$ in giù; da lì $\phi$ è zero esatto, e un
denominatore che si azzera è il modo tipico in cui questa implementazione si
rompe). La softmax aveva anche un denominatore che
normalizzava i pesi: lo si conserva come un secondo accumulatore
$\mathbf{z} = \sum_j \phi(\mathbf{k}_j)$, e la lettura *normalizzata* diventa
$\mathbf{o}_i = \mathbf{S}\,\phi(\mathbf{q}_i) \,/\, \big(\mathbf{z}^\top \phi(\mathbf{q}_i)\big)$.

`````

## L'attenzione lineare è una RNN

Fin qui abbiamo ragionato come se tutte le parole fossero disponibili insieme.
Ma quando il modello scrive vale una regola in più: ogni parola può guardare
solo quelle che la precedono, non quelle che verranno, e questo si dice
**causale**. Allora il registro non è più uno solo, compilato alla fine: ce
n'è una versione a ogni passo, e ogni parola nuova aggiunge il suo contributo a
quello che c'è già scritto. Attenzione a non fraintendere: a cambiare sono i
numeri scritti nelle caselle, non il numero di caselle, che resta quello.

Un registro che si aggiorna così, un passo alla volta e a partire da come era
al passo prima, ha un nome: è una *ricorrenza*. Ed è il nome che portavano le
reti che i Transformer avevano mandato in pensione.

`````{tab} Elementare

Pensa alla differenza fra due modi di prendere appunti a una riunione lunga.

Il primo è la KV cache dei Transformer: ogni volta che qualcuno parla,
aggiungi uno scontrino alla pila. Non butti via niente, e questo è comodo (hai
tutto) ma la pila cresce, e a fine giornata occupa mezzo tavolo. Ogni parola
nuova ne aggiunge un'altra.

Il secondo è un **foglio-registro di dimensione fissa**. Non aggiungi
scontrini: aggiorni le stesse caselle. Quando qualcuno parla, sommi il suo
contributo a ciò che c'è già scritto, e il foglio resta un foglio: sempre lo
stesso, che tu sia alla decima o alla decimillesima parola. Per rispondere a
una domanda leggi il foglio, non rovisti nella pila.

L'attenzione lineare tiene esattamente questo foglio. Ecco perché la memoria non
cresce: qualunque sia la lunghezza del testo, il foglio (negli articoli si
chiama *stato*) ha sempre le stesse dimensioni. È il ritorno, sotto mentite
spoglie, della vecchia idea delle reti ricorrenti.

`````

`````{tab} Superiore

Basta riscrivere $\mathbf{S}$ come somma cumulativa fino al passo $t$:

$$
\mathbf{S}_t = \sum_{i \le t} \mathbf{v}_i\, \phi(\mathbf{k}_i)^\top
    = \mathbf{S}_{t-1} + \mathbf{v}_t\, \phi(\mathbf{k}_t)^\top,
\qquad
\mathbf{z}_t = \mathbf{z}_{t-1} + \phi(\mathbf{k}_t),
$$

e la lettura al passo $t$ usa lo stato corrente:

$$
\mathbf{o}_t = \frac{\mathbf{S}_t\, \phi(\mathbf{q}_t)}{\mathbf{z}_t^\top\, \phi(\mathbf{q}_t)}.
$$

Qui $\mathbf{S}_t \in \mathbb{R}^{d\times d}$ è lo **stato**, una memoria chiave→valore
che a ogni passo incassa il prodotto esterno $\mathbf{v}_t\, \phi(\mathbf{k}_t)^\top$ (scrivi il
value $\mathbf{v}_t$ sotto l'etichetta $\phi(\mathbf{k}_t)$); $\mathbf{z}_t$ è il normalizzatore che
accumula le key trasformate; $\mathbf{o}_t$ è l'uscita. Leggere con $\phi(\mathbf{q}_t)$ significa
$\mathbf{S}_t \phi(\mathbf{q}_t) = \sum_{i\le t} \big(\phi(\mathbf{k}_i)^\top\phi(\mathbf{q}_t)\big)\, \mathbf{v}_i$: la query
ripesca dai value in proporzione a quanto la sua etichetta somiglia a ciascuna
key già scritta.

Guardiamo bene questa ricorrenza. È **esattamente una RNN**, ma con due
differenze rispetto alle celle del capitolo sull'NLP. La prima: lo stato non è
un vettore $\mathbf{h}_t$ ma una **matrice** $\mathbf{S}_t$ (una memoria molto più capiente). La
seconda, decisiva: la transizione di stato è **lineare**, anzi è l'identità
($\mathbf{S}_{t-1}$ passa intatto, gli si somma soltanto un termine nuovo). Non c'è
nessuna $\tanh$ o non-linearità *sullo stato*, come invece in
$\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \dots)$. L'aggiornamento costa $O(d^2)$ per
token e la memoria è **costante**: la matrice $d \times d$ non cambia
dimensione, che siamo al decimo o al milionesimo token.

È il senso, volutamente ironico, del titolo del paper di Katharopoulos et al.,
*Transformers are RNNs* {cite}`katharopoulos2020transformers`: sotto una certa
scelta della somiglianza, un Transformer *è* una rete ricorrente; solo che se
n'era dimenticato.

`````

```{figure} ../figures/attenzione-lineare-ricorrenza.svg
:name: fig-attenzione-lineare-ricorrenza
:alt: Due pannelli affiancati. A sinistra l'attenzione classica: a ogni passo da t=1 a t=4 la cache si allunga di una coppia chiave-valore, e la pila cresce verso l'alto passo dopo passo. A destra l'attenzione lineare: quattro riquadri identici, uno per passo, collegati da frecce, ciascuno una matrice di stato S della stessa taglia, aggiornata dal prodotto esterno fra il valore v e la chiave k.
:width: 85%

Due modi di ricordare. A sinistra la memoria dei Transformer, che *cresce* di
una coppia etichetta-informazione a ogni parola. A destra l'attenzione
lineare: un'unica tabella di numeri, sempre della stessa taglia, in cui ogni
parola somma la propria informazione sotto la propria etichetta, e che si
rilegge facendole una domanda.
```

Come mostra {numref}`fig-attenzione-lineare-ricorrenza`, i due schemi
raccolgono la stessa informazione in modi opposti: la KV cache la conserva
tutta e paga in memoria che cresce; il registro la comprime in un foglio di
taglia fissa.

Una nota di servizio prima di proseguire, che riguarda il modo di scrivere il
meccanismo, non il meccanismo.

`````{tab} Elementare

Nelle prossime sezioni ritroverai lo stesso foglio-registro raccontato in modo
un po’ più asciutto, perché i gruppi di ricerca non usano tutti gli stessi
accorgimenti di conto. Il meccanismo però non cambia di una virgola: si scrive,
si somma, si rilegge. Se leggi solo questo livello, tira dritto.

`````

`````{tab} Superiore

Katharopoulos tiene il normalizzatore $\mathbf{z}_t$: la lettura è una media pesata e
$\mathbf{z}_t^\top \phi(\mathbf{q}_t)$ ne è il denominatore. Già i lavori sui *fast weight* di
poco successivi lo abbandonano, giudicandolo instabile (quell'accumulatore può
crescere senza controllo), e normalizzano invece chiavi e query trasformate; le
varianti più recenti vi rinunciano del tutto, aggiungendo una *layer
normalization* in uscita e, dove la transizione lo richiede (DeltaNet),
riportando le key a norma unitaria. Impostazioni diverse, che è
meglio non mescolare: nelle prossime sezioni terremo distinte le due scuole e,
parlando delle architetture moderne, useremo la seconda, cioè feature map
$\phi$ posta all'identità e nessun $\mathbf{z}_t$ nelle formule.

`````

## Addestrare in parallelo, generare in ricorrenza

Fin qui il registro lo abbiamo visto riempirsi una parola alla volta. Ma quando
il testo c'è già tutto non c'è nessun bisogno di aspettare: lo stesso identico
risultato si può ottenere anche in un colpo solo, spartendo il lavoro fra tante
unità di calcolo che macinano insieme. Sono due volti dello stesso calcolo. Non è
una coincidenza tecnica: è la proprietà che rende interessante tutta questa
famiglia di modelli, e che ritroveremo (con parole diverse) nel capitolo sugli
*State Space Model*, i modelli che descrivono una sequenza come un sistema che
evolve nel tempo.

```{figure} ../figures/stato-ricorrente.gif
:name: fig-stato-ricorrente
:alt: "Animazione: cinque token entrano uno alla volta in una matrice di stato 3x3; a ogni token la matrice somma un prodotto esterno e le sue celle cambiano valore, ma la matrice resta sempre della stessa dimensione."
:width: 90%

La memoria che si aggiorna parola per parola: a ogni passo i numeri nelle
caselle cambiano, perché ci si somma sopra il contributo della parola appena
letta, ma le caselle restano quelle. Alla quinta parola la tabella è grande
esattamente come alla prima.
```

La {numref}`fig-stato-ricorrente` mostra la parte "economica" del patto: la
memoria non cresce. Ma mostra anche, senza dirlo, il prezzo: dentro ogni cella
i contributi di parole diverse finiscono sommati fra loro, e una volta sommati
non si possono più separare. È il tema di *Il limite dell'accumulo*, più avanti
in questa pagina.

`````{tab} Elementare

Immagina la differenza fra studiare un libro che hai già tutto in mano e
raccontare a voce una storia che stai inventando adesso.

Nel primo caso (è l'addestramento, quando il modello impara) il testo esiste
già per intero: puoi aprirlo a metà, dare un capitolo a testa a dieci persone e
finire in un decimo del tempo. È la forma «tutta insieme», quella che tiene
occupata tutta la scheda grafica.

Nel secondo caso (è la generazione, quando il modello scrive) procedi parola per
parola, perché il seguito non esiste ancora: lo stai inventando. Qui l'unica
cosa che ti porti dietro è il foglio-registro, che aggiorni a ogni parola e che
non cresce mai.

Il bello è che le due forme danno lo stesso risultato, e allora si usa ciascuna
dove conviene: il modello si allena con quella che sfrutta tutte le unità di
calcolo insieme, e scrive con quella che occupa sempre la stessa memoria. La KV
cache dei Transformer, al contrario, obbliga a trascinare una pila che si
allunga a ogni parola generata.

Un'avvertenza, perché il conto non sembri troppo bello: la forma «tutta
insieme», fatta nel modo più ovvio, si ritrova per le mani la tabella grande di
prima. Per evitarla si lavora **a blocchi**: si taglia il testo in pezzi, e
dentro ogni pezzo si fa tutto insieme, dove la tabella è piccola perché le
parole sono poche; poi il registro passa da un pezzo al successivo. In fila
vanno i pezzi, che sono pochi e grossi; il grosso del lavoro, quello dentro ai
pezzi, resta in parallelo. Il risultato non cambia, e i blocchi torneranno in
tutto il resto del capitolo.

`````

`````{tab} Superiore

**In addestramento** si usa la forma parallela, e qui serve un momento di
onestà sui costi. La strada più diretta è il prodotto fra matrici *mascherato*
dalla causalità, $\big(\phi(\mathbf{Q})\,\phi(\mathbf{K})^\top \odot \mathbf{M}\big)\mathbf{V}$ con $\mathbf{M}$ la
maschera triangolare: parallelo esattamente come un Transformer, ma la
maschera impedisce di ri-associare il prodotto e si torna a pagare
$O(n^2 d)$.
L'alternativa è srotolare la somma cumulativa
$\mathbf{S}_t = \sum_{i\le t} \mathbf{v}_i\,\phi(\mathbf{k}_i)^\top$ come *prefix sum* (una somma
progressiva): l'operazione è associativa, quindi si calcola con uno *scan*
parallelo a costo $O(n d^2)$, lineare. L'ostacolo qui non è il numero di
operazioni ma la memoria: uno scan pretende di materializzare tutti gli $n$
stati intermedi $d \times d$, cioè $O(n d^2)$ di memoria contro gli $O(d^2)$
della forma ricorrente, e il traffico da e verso la memoria della GPU si mangia
il guadagno del parallelismo (con $n = 8192$ e $d = 64$ per testa sono più di
33 milioni di valori per testa e per strato, contro i 4096 dello stato).
Nessuna delle due forme dà insieme le due cose; la conciliazione usata in
pratica è il calcolo **a blocchi** (*chunkwise*): parallelo dentro ogni blocco,
ricorrente fra un blocco e l'altro, costo $O(nBd + nd^2)$ con blocchi di
ampiezza $B$, cioè lineare in $n$. Lo ritroveremo, formalizzato, in RetNet e
DeltaNet.

Un'ultima onestà sul lato addestramento, perché «lineare» non vuol dire
«subito più veloce»: l'attenzione softmax ha implementazioni curatissime nel
traffico di memoria, e sotto qualche migliaio di token un calcolo a blocchi
scritto in modo ingenuo perde il confronto. Il vantaggio si raccoglie quando il
contesto cresce, ed è il motivo per cui buona parte del lavoro su queste
architetture è lavoro di implementazione, non di formule.

**In inferenza autoregressiva** si usa la forma ricorrente: si aggiorna $\mathbf{S}_t$
sul posto e si legge $\mathbf{o}_t$, con costo $O(d^2)$ per token e memoria $O(d^2)$
**costante**. Nessuna KV cache che si allunga: lo stato è sempre la stessa
matrice $d \times d$. In un Transformer, ricordiamo, la cache cresce di una
coppia $(\mathbf{k}_t, \mathbf{v}_t)$ per token e per strato, e il costo di generare l’$n$-esimo
token sale con la lunghezza del prefisso; qui resta piatto.

È da questo contrasto che nasce l'accelerazione più spettacolare riportata da
Katharopoulos et al.: fino a circa **quattromila volte** più veloce nella
generazione autoregressiva di sequenze *molto* lunghe. Va letta con onestà (è
un caso limite, riguarda solo l'inferenza e non l'addestramento) ma misura
bene il vantaggio della memoria costante quando $n$ diventa enorme.

`````

Concretamente, il passo con cui il modello genera una parola è poche righe: una
tabella di numeri che vive in un posto solo e a ogni parola viene aggiornata
sul posto.
Chi non programma può leggerci una cosa sola, ed è quella che conta: `S` entra
ed esce dalla funzione sempre della stessa taglia, e in tutto il codice non c'è
nessuna lista che si allunga.

```python
import torch
import torch.nn.functional as F

phi = lambda x: F.elu(x) + 1.0   # feature map: sempre positiva

def genera_passo(S, z, q_t, k_t, v_t):
    """Un passo di attenzione lineare, forma ricorrente.
    S: stato d x d (fisso)   z: normalizzatore d   q_t, k_t, v_t: vettori d."""
    pk = phi(k_t)
    S = S + torch.outer(v_t, pk)          # scrivi: S += v_t phi(k_t)^T
    z = z + pk                            # aggiorna il normalizzatore
    pq = phi(q_t)
    o_t = (S @ pq) / (z @ pq)             # leggi: o_t = S phi(q_t) / (z . phi(q_t))
    return o_t, S, z
```

La memoria occupata da quella tabella non dipende da quanti token abbiamo già
generato: è il cuore del vantaggio. (Accanto a `S` viaggia un secondo
accumulatore, `z`, molto più piccolo, che serve solo a tenere i numeri in
scala; anche lui è di taglia fissa.)

## Il limite dell'accumulo

Tanta eleganza ha un prezzo, ed è bene dirlo subito e senza sconti. Il registro
dell'attenzione lineare sa fare una cosa sola: **sommare**. Non cancella e non
corregge. Ogni parola che passa lascia la sua traccia sul foglio, sommata a
tutte le altre, e lì resta per sempre.

`````{tab} Elementare

Un foglio-registro su cui continui a sommare senza mai cancellare niente, prima
o poi diventa illeggibile: le scritte si sovrappongono, e quando cerchi
un'informazione precisa ti ritrovi un pasticcio di tracce che si confondono.

Se due parole diverse hanno etichette simili, i loro contributi si mescolano, e
rileggendo non capisci più bene quale informazione appartenesse a quale
etichetta. Il disturbo non comincia di colpo a una certa soglia: comincia
subito, piano, e cresce con quante cose hai scritto.

E il foglio si riempie molto prima di quanto lascino sperare le sue caselle,
perché un'informazione non occupa una casella: quando la scrivi si spalma su
tutto il foglio, e la successiva si spalma sopra di lei. Prendi un foglio
piccolo, trentadue righe per trentadue colonne (nei modelli veri una memoria di
questo tipo ne ha sessantaquattro o centoventotto): di caselle ne ha più di
mille, ma già a otto informazioni la risposta torna sbagliata di circa metà del
suo valore, e a trentadue lo sbaglio è grande quanto la risposta. Da lì in poi
ritrovare il dettaglio giusto («di che colore era il cappotto citato venti
pagine fa?») è impossibile. L'attenzione dei Transformer,
quella che tiene tutti gli scontrini, quel dettaglio ce l'ha ancora; il registro
riassuntivo può averlo perso.

E la domanda ovvia (perché non prendersi un foglio più grande?) ha una
risposta altrettanto ovvia: si può, ed è una delle manopole di chi progetta il
modello, ma si paga. Un foglio più largo vuol dire più caselle da aggiornare e
da rileggere a ogni parola, quindi più conti e più memoria: allargandolo
abbastanza si torna a spendere quanto un Transformer, e il vantaggio che
eravamo venuti a cercare svanisce. Il gioco di tutto il resto del capitolo è
un altro: tenere il foglio piccolo e imparare a **scriverci meglio**.

`````

`````{tab} Superiore

Il limite è di **capacità**, ed è una conseguenza della dimensione finita. Lo
stato $\mathbf{S}$ è una matrice $d \times d$: in uno spazio di dimensione $d$ non
esistono più di $d$ vettori mutuamente ortogonali. Se le key
$\phi(\mathbf{k}_1), \dots$ fossero esattamente ortogonali, leggere con $\phi(\mathbf{q})$
recupererebbe il value giusto pulito fino a $d$ associazioni; ma le key le
produce una proiezione lineare, non un'ortogonalizzazione, e con chiavi
casuali il **crosstalk** (le briciole degli altri value che il retrieval
raccoglie insieme a quello cercato) non compare a una soglia: cresce da subito
come $\sqrt{N/d}$ con il numero $N$ di associazioni scritte. A $N \approx d$
l'interferenza vale ormai quanto il valore cercato. In $d=32$, con chiavi
gaussiane riportate a norma unitaria e value gaussiani, l'errore relativo medio
del richiamo (media su tutte le chiavi scritte e su duemila estrazioni) è
$0{,}99$ a $N=d$ e $0{,}46$ a $N=d/4$: sono i valori attesi
$\sqrt{(N-1)/d}$, cioè $\sqrt{N/d}$ a meno del contributo della chiave che si
sta interrogando. Il richiamo pulito vuole quindi
$N$ **ben minore** di $d$, non $N \le d$. Ed essendo la transizione l'identità, non
c'è modo di **dimenticare**: una scrittura spuria fatta all'inizio resta a
disturbare per sempre. È la ragione per cui l'attenzione lineare pura, così
com'è, resta indietro rispetto all'attenzione softmax proprio sui compiti di
richiamo preciso.

`````

Da qui in avanti l'intero capitolo è un tentativo di curare questi due mali
tenendo però il dono della memoria che non cresce. Servono due ingredienti: un
modo per **dimenticare**, cioè un interruttore (in inglese *gate*) che lasci
sbiadire quello che è vecchio invece di tenerlo eterno; e un modo per
**correggere** invece di sommare alla cieca, cioè andare a vedere che cosa il
foglio risponde già a quella etichetta e scriverci sopra solo la differenza (è
la *regola delta*). Sono esattamente i due fili della prossima sezione.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Far parlare ogni parola con tutte le altre è un'esplosione di conversazioni:
  mille parole sono quasi mezzo milione di scambi. Sostituire quel confronto a
  due a due con un **registro riassuntivo**, che ciascuno aggiorna una volta
  sola, porta il conto a mille aggiornamenti: da esplosivo a proporzionale alla
  lunghezza del testo, ed è esattamente ciò che si intende per costo
  **lineare**.
- Il registro è un **foglio di dimensione fissa**: ogni parola ci scrive la
  propria informazione sotto la propria etichetta, e per rispondere a una
  domanda lo si rilegge, invece di rovistare nella pila degli scontrini.
- Letto parola per parola, quel foglio è il riassunto di una vecchia rete
  ricorrente: si aggiorna sommando, costa sempre lo stesso a ogni parola e non
  cresce mai, che si sia alla decima o alla milionesima. È l'ironia del titolo
  *Transformers are RNNs* (Katharopoulos e colleghi, 2020).
- Stesso risultato, **due modi di ottenerlo**: tutto insieme quando il testo c'è
  già (studiare un libro che si ha in mano, cioè addestrare, e in pratica si fa
  a blocchi) e una parola alla volta quando il testo si sta inventando
  (raccontarlo a voce, cioè generare), senza la pila che si allunga. La stessa
  doppia natura tornerà con gli State Space Model.
- Il difetto del registro: somma e basta, non cancella e non corregge. Le
  scritte si sovrappongono un po’ fin dalla prima riga, e il foglio si riempie
  molto prima di quanto lascino sperare le sue caselle, perché ogni
  informazione si spalma su tutto il foglio: uno da trentadue righe per
  trentadue colonne ne ha più di mille, ma già a otto informazioni la risposta
  torna sbagliata di circa metà del suo valore, e a trentadue lo sbaglio è
  grande quanto la risposta. Prendere un foglio più grande si può, ma costa conti e
  memoria a ogni parola, e a quel punto tanto vale un Transformer.
- I due rimedi delle sezioni successive: un modo per **sbiadire** ciò che è
  vecchio e un modo per **correggere** invece di sommare alla cieca.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- La softmax dei Transformer costa $O(n^2 d)$ perché non si fattorizza **in
  dimensione finita**; sostituirla con una somiglianza
  $\text{sim}(\mathbf{q},\mathbf{k})=\phi(\mathbf{q})^\top\phi(\mathbf{k})$ e
  ri-associare il prodotto porta il costo a $O(n d^2)$, cioè **lineare** nella
  lunghezza.
- Il calcolo si condensa in uno **stato-matrice**
  $\mathbf{S} = \sum_j \mathbf{v}_j\,\phi(\mathbf{k}_j)^\top$
  di dimensione fissa $d \times d$: una memoria chiave→valore che si legge con
  $\mathbf{S}\,\phi(\mathbf{q})$.
- In forma causale è una **ricorrenza**,
  $\mathbf{S}_t = \mathbf{S}_{t-1} + \mathbf{v}_t\,\phi(\mathbf{k}_t)^\top$:
  cioè una **RNN a stato matriciale**
  con transizione lineare (l'identità), aggiornamento $O(d^2)$ per token e
  memoria **costante**; da cui l'ironia di *Transformers are RNNs*
  (Katharopoulos et al., 2020).
- Stessa funzione, **due forme**: parallela per addestrare, ricorrente per
  generare a memoria costante, senza la KV cache che invece cresce. La stessa
  dualità tornerà per gli State Space Model. Lineare non vuol dire però subito
  più veloce: sotto qualche migliaio di token una implementazione ingenua perde
  contro un'attenzione softmax ben scritta.
- Il difetto dell'accumulo puro: non dimentica e non corregge. Con chiavi
  casuali l'interferenza (**crosstalk**) cresce come $\sqrt{N/d}$ fin da
  subito e a $N \approx d$ pareggia il segnale: il richiamo pulito vuole $N$
  ben minore di $d$.
- I due rimedi (un **gate** che dimentica e una **regola delta** che corregge
  invece di sommare) sono il filo delle sezioni successive.
```

`````
