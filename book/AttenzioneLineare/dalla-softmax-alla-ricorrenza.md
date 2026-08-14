# Dalla softmax alla ricorrenza

Il capitolo sui Transformer si chiude con un paradosso. Il meccanismo che ha
vinto la partita (l'attenzione, che lascia ogni parola libera di guardare
tutte le altre) vince *proprio perché* guarda tutte le altre; ed è esattamente
questo il suo conto da pagare. Far parlare ognuno con ognuno costa
**quadratico**: raddoppiando la lunghezza della frase il lavoro quadruplica
(gli informatici lo scrivono $O(n^2)$, che è solo un modo compatto di dire
«cresce come il quadrato della lunghezza $n$»). E in generazione c'è un secondo
conto. Come abbiamo visto nella sezione sui grandi modelli linguistici, la
generazione autoregressiva si appoggia alla **KV cache**, il "segnalibro" che
conserva le chiavi e i valori già calcolati (le etichette e le informazioni di
cui parlavamo aprendo il capitolo) per non rifarli a ogni passo: comodissima,
ma cresce parola dopo parola, memoria che si accumula finché, su un contesto
lungo, arriva a pesare quanto il modello stesso.

Le reti ricorrenti che abbiamo studiato prima dei Transformer non avevano
nessuno di questi due problemi: leggevano in fila, a costo lineare, con uno
stato di dimensione *fissa* che non cresceva mai. Le avevamo abbandonate per
un difetto altrettanto grave, la **sequenzialità**: ogni passo deve aspettare
quello prima, e le schede grafiche (le GPU), che sono fatte per macinare
montagne di conti tutti insieme, restano a guardare. La domanda di questo
capitolo è se si
possano avere entrambe le cose: **il parallelismo dell'attenzione in
addestramento e il costo lineare a memoria costante delle RNN in inferenza.**
La risposta, sorprendentemente, parte da un'osservazione quasi algebrica sulla
softmax.

## Il trucco del kernel: spezzare la softmax

L'ostacolo ha un nome preciso, ed è la softmax. Per calcolare l'attenzione
bisogna prima misurare quanto ogni parola somiglia a ogni altra, e riempire con
quei numeri una tabella che ha una riga e una colonna per ciascuna parola:
mille parole, un milione di caselle. La tabella grande serve perché la softmax
*mescola*, cioè spartisce fra tutte: quanto la parola numero uno dà retta alla
numero due dipende anche da tutte le altre, e finché è così nessuna casella si
può calcolare per conto suo.

La via d'uscita è cambiare il modo di misurare la somiglianza. Se al posto di
un numero che nasce dal confronto di una coppia se ne mettesse uno che si
**spezza in due pezzi indipendenti** (uno che riguarda solo chi fa la domanda,
uno che riguarda solo chi risponde), i conti si potrebbero riordinare: si
raccoglie una volta per tutte quello che riguarda chi risponde, e la tabella
grande non serve più.

`````{tab} Elementare

Torniamo all'immagine dell'assemblea del capitolo sui Transformer: mille parole
sono mille persone, e farle parlare tutte con tutte sono quasi mezzo milione di
conversazioni (mille per mille fa un milione, ma ogni coppia conta una volta
sola, quindi circa la metà). Il costo esplode perché ogni coppia va gestita a
parte.

Ma c'è un altro modo di raccogliere le stesse informazioni. Invece di mettere
tutti a chiacchierare fra loro, teniamo un **registro riassuntivo**. Ogni
persona che entra scrive una riga («io ho questa etichetta e porto questa
informazione») e chi deve farsi un'idea non interroga più tutti a uno a uno:
legge il registro, già bell'e riassunto. Le persone diventano mille, ma il
registro resta uno solo, e lo si aggiorna una volta per ciascuna: mille
aggiornamenti invece di mezzo milione di conversazioni.

Il trucco sta tutto nel trovare un modo di riassumere che non perda ciò che
serve, e questo è il punto delicato: un riassunto, per quanto ben fatto, tiene
meno cose di un archivio completo. Il conto lo pagheremo in fondo alla sezione,
quando vedremo che cosa succede a un registro su cui si continua a scrivere
senza mai cancellare. In matematica quel «modo di riassumere» si chiama
**kernel**.

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
$2^{-25}$, cioè da $x \approx -17{,}3$ in giù; da lì $\phi$ è zero esatto, e un
denominatore che si azzera è il modo tipico in cui questa implementazione si
rompe). La softmax aveva anche un denominatore che
normalizzava i pesi: lo si conserva come un secondo accumulatore
$\mathbf{z} = \sum_j \phi(\mathbf{k}_j)$, e la lettura *normalizzata* diventa
$\mathbf{o}_i = \mathbf{S}\,\phi(\mathbf{q}_i) \,/\, \big(\mathbf{z}^\top \phi(\mathbf{q}_i)\big)$.

`````

## L'attenzione lineare è una RNN

Fin qui abbiamo ragionato come se tutte le parole fossero disponibili insieme.
Ma in generazione il modello è **causale**: ogni parola può guardare solo
quelle che la precedono, non quelle che verranno. Allora il registro non è più
uno solo, compilato alla fine: ce n'è una versione a ogni passo, e ogni parola
nuova aggiunge il suo contributo a quello che c'è già scritto. Attenzione a non
fraintendere: a crescere è il contenuto, non il foglio, che resta grande
uguale. Descrivere un registro che si aggiorna così, un passo alla volta a
partire dal proprio stato precedente, significa descrivere una *ricorrenza*, ed
è qui che l'attenzione lineare rivela la sua vera natura.

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
cresce: qualunque sia la lunghezza del testo, lo stato ha sempre le stesse
dimensioni. È il ritorno, sotto mentite spoglie, della vecchia idea delle RNN.

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

Nelle prossime sezioni ritroverai lo stesso foglio-registro raccontato in
modo un po' più asciutto: chi lavora su questi modelli si porta dietro qualche
accorgimento di conto in più o in meno, e i vari gruppi di ricerca non hanno
scelto tutti lo stesso. Serve a tenere i numeri entro grandezze ragionevoli, e
non cambia di una virgola quello che hai letto qui: si scrive, si somma, si
rilegge. Se leggi solo questo livello non ti perdi nulla; se ogni tanto sbirci
le formule dell'altro, sappi che da qui in avanti sono scritte in una forma
sola.

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

Abbiamo ora due volti dello stesso calcolo. Lo stesso identico risultato si
può ottenere riempiendo in un colpo solo la tabella di tutte le somiglianze
fra le parole, oppure facendo scorrere il registro parola per parola. Non è una
coincidenza tecnica: è la proprietà che rende interessante tutta questa
famiglia di modelli, e che ritroveremo (con parole diverse) nel capitolo sugli
*State Space Model*.

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
memoria non cresce. Ma mostra anche, senza dirlo, il prezzo: le celle si
sommano l'una sull'altra, e ciò che è stato scritto non si può più separare. È
il tema di *Il limite dell'accumulo*, più avanti in questa pagina.

`````{tab} Elementare

Immagina di dover correggere un tema già scritto e di doverne scrivere uno
nuovo.

Per **correggere** (l'addestramento) hai il testo intero davanti: puoi guardare
tutte le frasi in una volta, distribuire il lavoro a più persone, finire in
fretta. È la forma «tutta insieme», parallela.

Per **scrivere** (la generazione) procedi invece parola per parola: non conosci
il seguito perché lo stai inventando adesso. Qui l'unica cosa che ti porti
dietro è il foglio-registro, che aggiorni a ogni parola e che non cresce mai. È
la forma ricorrente, a memoria costante.

Il bello è che le due forme danno lo stesso risultato: alleni il modello con
quella comoda e veloce, lo fai generare con quella economica. La KV cache dei
Transformer, al contrario, ti obbligava a trascinare una pila che si allunga a
ogni parola generata.

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
coppia $(\mathbf{k}_t, \mathbf{v}_t)$ per token e per strato, e il costo di generare l'$n$-esimo
token sale con la lunghezza del prefisso; qui resta piatto.

È da questo contrasto che nasce l'accelerazione più spettacolare riportata da
Katharopoulos et al.: fino a circa **quattromila volte** più veloce nella
generazione autoregressiva di sequenze *molto* lunghe. Va letta con onestà (è
un caso limite, riguarda solo l'inferenza e non l'addestramento) ma misura
bene il vantaggio della memoria costante quando $n$ diventa enorme.

`````

Concretamente, il ciclo con cui il modello genera è poche righe: una tabella di
numeri che vive in un posto solo e a ogni parola viene aggiornata sul posto.
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

La memoria occupata da `S` e `z` non dipende da quanti token abbiamo già
generato: è il cuore del vantaggio.

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
etichetta. Il pasticcio non comincia di colpo a una certa soglia: comincia
subito, piano, e diventa serio quando le cose da ricordare si avvicinano al
numero di caselle del foglio, perché a quel punto il rumore delle altre pesa
quanto la risposta cercata. Da lì in poi ritrovare il dettaglio giusto («di che
colore era il cappotto citato venti pagine fa?») diventa impossibile.
L'attenzione piena, che tiene tutti gli scontrini, quel dettaglio ce l'ha
ancora; il registro riassuntivo può averlo perso.

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
l'interferenza vale ormai quanto il valore cercato: con chiavi casuali di norma
unitaria in $d=32$, l'errore relativo medio del richiamo è $0{,}99$ a $N=d$ e
$0{,}46$ a $N=d/4$, cioè proprio $\sqrt{N/d}$. Il richiamo pulito vuole quindi
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
  già (correggere un tema, cioè addestrare) e una parola alla volta quando il
  testo si sta inventando (scriverlo, cioè generare), senza la pila che si
  allunga. La stessa doppia natura tornerà con gli State Space Model.
- Il difetto del registro: somma e basta, non cancella e non corregge. Le
  scritte si sovrappongono un po' fin dalla prima riga, e quando le cose da
  ricordare si avvicinano al numero di caselle del foglio il disturbo pesa
  quanto la risposta cercata: da lì in poi ritrovare il dettaglio esatto
  diventa impossibile. Prendere un foglio più grande si può, ma costa conti e
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
