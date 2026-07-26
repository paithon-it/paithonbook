# Dalla softmax alla ricorrenza

Il capitolo sui Transformer si chiude con un paradosso. Il meccanismo che ha
vinto la partita (l'attenzione, che lascia ogni parola libera di guardare
tutte le altre) vince *proprio perché* guarda tutte le altre; ed è esattamente
questo il suo conto da pagare. Far parlare ognuno con ognuno costa
**quadratico**: raddoppiando la lunghezza della frase il lavoro quadruplica,
$O(n^2)$ nella lunghezza $n$ della sequenza. E in generazione c'è un secondo
conto. Come abbiamo visto nella sezione sui grandi modelli linguistici, la
generazione autoregressiva si appoggia alla **KV cache**, il "segnalibro" che
conserva key e value già calcolati per non rifarli a ogni passo: comodissima,
ma cresce parola dopo parola, memoria che si accumula finché, su un contesto
lungo, arriva a pesare quanto il modello stesso.

Le reti ricorrenti che abbiamo studiato prima dei Transformer non avevano
nessuno di questi due problemi: leggevano in fila, a costo lineare, con uno
stato di dimensione *fissa* che non cresceva mai. Le avevamo abbandonate per
un difetto altrettanto grave, la **sequenzialità**: il passo $t$ deve
aspettare il passo $t-1$, e le GPU, fatte per moltiplicare enormi matrici
tutte insieme, restano a guardare. La domanda di questo capitolo è se si
possano avere entrambe le cose: **il parallelismo dell'attenzione in
addestramento e il costo lineare a memoria costante delle RNN in inferenza.**
La risposta, sorprendentemente, parte da un'osservazione quasi algebrica sulla
softmax.

## Il trucco del kernel: spezzare la softmax

L'ostacolo ha un nome preciso, ed è la softmax. Per calcolare l'attenzione
bisogna prima formare la matrice $QK^\top$ di tutte le affinità, e quella
matrice è $n \times n$ perché la softmax *mescola* ogni query con ogni key: il
peso che la parola $i$ dà alla parola $j$ dipende, attraverso la
normalizzazione, anche da tutte le altre. Se riuscissimo a sostituire quella
somiglianza con una che si **fattorizza** (un pezzo che dipende solo da $q$
per un pezzo che dipende solo da $k$), potremmo riordinare i conti e non
costruire più la matrice grande.

`````{tab} Elementare

Torniamo all'immagine dell'assemblea del capitolo sui Transformer: mille parole
sono mille persone, e farle parlare tutte con tutte sono quasi mezzo milione di
conversazioni. Il costo esplode perché ogni coppia va gestita a parte.

Ma c'è un altro modo di raccogliere le stesse informazioni. Invece di mettere
tutti a chiacchierare fra loro, teniamo un **registro riassuntivo**. Ogni
persona che entra scrive una riga («io ho questa etichetta e porto questa
informazione») e chi deve farsi un'idea non interroga più tutti a uno a uno:
legge il registro, già bell'e riassunto. Le persone diventano mille, ma il
registro resta uno solo, e lo si aggiorna una volta per ciascuna: mille
aggiornamenti invece di mezzo milione di conversazioni.

Il trucco sta tutto nel trovare un modo di riassumere che non perda ciò che
serve. In matematica quel «modo di riassumere» si chiama **kernel**, e il
prezzo per ottenerlo è rinunciare alla softmax così com'è.

`````

`````{tab} Superiore

Nell'attenzione dei Transformer {cite}`vaswani2017attention` l'uscita (non
normalizzata) per la query $i$ è una somma dei value pesati dalla somiglianza
esponenziale $\exp(q_i^\top k_j)$. Il guaio è che quell'esponenziale *non si
spezza*: non esiste un modo di scriverlo come prodotto di una funzione della
sola $q_i$ per una funzione della sola $k_j$, e quindi va valutato per ogni
coppia $(i,j)$ (la matrice $n \times n$).

L'idea di Katharopoulos e colleghi {cite}`katharopoulos2020transformers` è
sostituire la somiglianza con una che *si spezza*, cioè un prodotto scalare fra
versioni trasformate di query e key:

$$
\text{sim}(q, k) = \phi(q)^\top \phi(k),
$$

dove $\phi$ è una **feature map** applicata elemento per elemento. Con questa
scelta l'uscita per la query $i$ diventa

$$
o_i = \sum_{j} \big(\phi(q_i)^\top \phi(k_j)\big)\, v_j
    = \sum_{j} v_j \big(\phi(k_j)^\top \phi(q_i)\big)
    = \Big(\underbrace{\sum_{j} v_j\, \phi(k_j)^\top}_{S}\Big)\, \phi(q_i)
    = S\,\phi(q_i),
$$

dove il passaggio chiave è la semplice **associatività** del prodotto: lo
scalare $\phi(k_j)^\top \phi(q_i)$ si può portare a destra di $v_j$, e allora
la somma su $j$ si stacca dalla query e si condensa in un'unica matrice
$S = \sum_j v_j\, \phi(k_j)^\top \in \mathbb{R}^{d\times d}$ (il «registro»
chiave→valore). Qui $q_i, k_j, v_j$ sono i vettori query, key e value del
token, $d$ la loro dimensione (per semplicità assumiamo key, query e value
della stessa dimensione), $\phi$ la feature map.

Le due strade hanno costi diversissimi. Calcolare tutti i prodotti
$\phi(q_i)^\top \phi(k_j)$ è la matrice $n \times n$, costo $O(n^2 d)$;
costruire $S$ una volta e applicarla a ogni query costa $O(n d^2)$: una
matrice $d \times d$ al posto di una $n \times n$. Quando $n \gg d$ (sequenze
lunghe), la seconda vince nettamente, ed è il passaggio da $O(n^2 d)$ a
$O(n d^2)$, cioè da quadratico a **lineare** nella lunghezza (lo stesso
$O(n d^2)$ delle ricorrenti che avevamo incontrato nel confronto fra
Transformer e RNN). In forma matriciale compatta, per tutte le query insieme,
è l'identità

$$
\big(\phi(Q)\,\phi(K)^\top\big)\,V = \phi(Q)\,\big(\phi(K)^\top V\big),
$$

la stessa $\text{softmax}(QK^\top)V$ del capitolo sui Transformer con la softmax
tolta di mezzo: senza di lei il prodotto si può ri-associare, e conviene fare
prima $\phi(K)^\top V$, la matrice piccola.

Resta da scegliere $\phi$. Serve una feature map che dia somiglianze positive
(perché i pesi si comportino come quelli di una media), e Katharopoulos et al.
propongono la più semplice che funzioni:

$$
\phi(x) = \operatorname{elu}(x) + 1,
$$

sempre maggiore di zero. La softmax aveva anche un denominatore che
normalizzava i pesi: lo si conserva come un secondo accumulatore
$z = \sum_j \phi(k_j)$, e la lettura *normalizzata* diventa
$o_i = S\,\phi(q_i) \,/\, \big(z^\top \phi(q_i)\big)$.

`````

## L'attenzione lineare è una RNN

Fin qui abbiamo ragionato come se tutte le parole fossero disponibili insieme.
Ma in generazione il modello è **causale**: la parola $t$ può guardare solo il
passato, non il futuro. Allora quel registro $S$ non è più fisso: cresce man
mano che leggiamo, un pezzo per ogni parola. Scriverlo come somma che si
allunga significa scriverlo come *ricorrenza*, ed è qui che l'attenzione
lineare rivela la sua vera natura.

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

Basta riscrivere $S$ come somma cumulativa fino al passo $t$:

$$
S_t = \sum_{i \le t} v_i\, \phi(k_i)^\top
    = S_{t-1} + v_t\, \phi(k_t)^\top,
\qquad
z_t = z_{t-1} + \phi(k_t),
$$

e la lettura al passo $t$ usa lo stato corrente:

$$
o_t = \frac{S_t\, \phi(q_t)}{z_t^\top\, \phi(q_t)}.
$$

Qui $S_t \in \mathbb{R}^{d\times d}$ è lo **stato**, una memoria chiave→valore
che a ogni passo incassa il prodotto esterno $v_t\, \phi(k_t)^\top$ (scrivi il
value $v_t$ sotto l'etichetta $\phi(k_t)$); $z_t$ è il normalizzatore che
accumula le key trasformate; $o_t$ è l'uscita. Leggere con $\phi(q_t)$ significa
$S_t \phi(q_t) = \sum_{i\le t} \big(\phi(k_i)^\top\phi(q_t)\big)\, v_i$: la query
ripesca dai value in proporzione a quanto la sua etichetta somiglia a ciascuna
key già scritta.

Guardiamo bene questa ricorrenza. È **esattamente una RNN**, ma con due
differenze rispetto alle celle del capitolo sull'NLP. La prima: lo stato non è
un vettore $h_t$ ma una **matrice** $S_t$ (una memoria molto più capiente). La
seconda, decisiva: la transizione di stato è **lineare**, anzi è l'identità
($S_{t-1}$ passa intatto, gli si somma soltanto un termine nuovo). Non c'è
nessuna $\tanh$ o non-linearità *sullo stato*, come invece in
$h_t = \tanh(W_{hh} h_{t-1} + \dots)$. L'aggiornamento costa $O(d^2)$ per
token e la memoria è **costante**: la matrice $d \times d$ non cambia
dimensione, che siamo al decimo o al milionesimo token.

È il senso, volutamente ironico, del titolo del paper di Katharopoulos et al.,
*Transformers are RNNs* {cite}`katharopoulos2020transformers`: sotto una certa
scelta della somiglianza, un Transformer *è* una rete ricorrente; solo che se
n'era dimenticato.

`````

```{figure} ../figures/attenzione-lineare-ricorrenza.svg
:name: fig-attenzione-lineare-ricorrenza
:alt: A sinistra lo stato-matrice S di dimensione fissa d per d aggiornato token per token sommando il prodotto esterno tra il value v e la key trasformata phi(k), e la lettura che moltiplica S per la query trasformata phi(q); a destra, per contrasto, la KV cache dei Transformer rappresentata come una pila di coppie key-value che si allunga a ogni nuovo token.
:width: 85%

A sinistra l'attenzione lineare: uno stato-matrice $S$ di dimensione *fissa*,
aggiornato a ogni parola sommando $v_t\,\phi(k_t)^\top$ e letto con
$\phi(q_t)$. A destra la KV cache dei Transformer, che invece *cresce* di una
coppia key–value a ogni token.
```

Come mostra {numref}`fig-attenzione-lineare-ricorrenza`, i due schemi
raccolgono la stessa informazione in modi opposti: la KV cache la conserva tutta
e paga in memoria che cresce; lo stato-matrice la comprime in un foglio di
taglia fissa.

Una nota di cautela sulla notazione, che ci servirà nel resto del capitolo:
Katharopoulos (e, poco dopo, i lavori sui *fast weight*) tengono il
normalizzatore $z_t$. Diverse varianti più recenti vi rinunciano del tutto,
normalizzando invece le key (le riportano tutte a lunghezza uno) e aggiungendo
una *layer normalization* in uscita: due impostazioni diverse, che è meglio
non mescolare. Nelle prossime sezioni terremo distinte le due scuole.

## Addestrare in parallelo, generare in ricorrenza

Abbiamo ora due volti dello stesso calcolo. Lo stesso identico risultato si
può ottenere formando la matrice delle affinità su tutta la sequenza in un
colpo solo, oppure facendo scorrere lo stato $S_t$ token per token. Non è una
coincidenza tecnica: è la proprietà che rende interessante tutta questa
famiglia di modelli, e che ritroveremo (con parole diverse) nel capitolo sugli
*State Space Model*.

```{figure} ../figures/stato-ricorrente.gif
:name: fig-stato-ricorrente
:alt: "Animazione: cinque token entrano uno alla volta in una matrice di stato 3x3; a ogni token la matrice somma un prodotto esterno e le sue celle cambiano valore, ma la matrice resta sempre della stessa dimensione."
:width: 90%

La forma ricorrente, token per token: lo stato $S_t$ assorbe il prodotto
esterno $v_t\,\phi(k_t)^\top$ e resta una matrice $d \times d$; al quinto
token è grande esattamente come al primo.
```

La {numref}`fig-stato-ricorrente` mostra la parte "economica" del patto: la
memoria non cresce. Ma mostra anche, senza dirlo, il prezzo: le celle si
sommano l'una sull'altra, e ciò che è stato scritto non si può più separare. È
il tema della sezione *Il limite dell'accumulo*, più avanti.

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

**In addestramento** si usa la forma parallela. La somma cumulativa
$S_t = \sum_{i\le t} v_i\,\phi(k_i)^\top$ è un *prefix sum* (una somma
progressiva): l'operazione è associativa, quindi non serve affatto procedere
in fila token per token; la si calcola su tutta la sequenza in parallelo, come
un prodotto fra matrici mascherato dalla causalità. Si sfrutta il parallelismo
delle GPU esattamente come farebbe un Transformer.

**In inferenza autoregressiva** si usa la forma ricorrente: si aggiorna $S_t$
sul posto e si legge $o_t$, con costo $O(d^2)$ per token e memoria $O(d^2)$
**costante**. Nessuna KV cache che si allunga: lo stato è sempre la stessa
matrice $d \times d$. In un Transformer, ricordiamo, la cache cresce di una
coppia $(k_t, v_t)$ per token e per strato, e il costo di generare l'$n$-esimo
token sale con la lunghezza del prefisso; qui resta piatto.

È da questo contrasto che nasce l'accelerazione più spettacolare riportata da
Katharopoulos et al.: fino a circa **quattromila volte** più veloce nella
generazione autoregressiva di sequenze *molto* lunghe. Va letta con onestà (è
un caso limite, riguarda solo l'inferenza e non l'addestramento) ma misura
bene il vantaggio della memoria costante quando $n$ diventa enorme.

`````

Concretamente, il ciclo di inferenza è poche righe: una matrice-stato che vive
in un posto solo e si aggiorna con un prodotto esterno.

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
corregge. Ogni parola che passa lascia il suo prodotto esterno nello stato, e lì
resta per sempre.

`````{tab} Elementare

Un foglio-registro su cui continui a sommare senza mai cancellare niente, prima
o poi diventa illeggibile: le scritte si sovrappongono, e quando cerchi
un'informazione precisa ti ritrovi un pasticcio di tracce che si confondono.

Se due parole diverse hanno etichette simili, i loro contributi si mescolano,
e rileggendo non capisci più bene quale value appartenga a quale key. Finché
le cose da ricordare sono poche il foglio regge; oltre una certa soglia va in
sovraccarico, e ritrovare il dettaglio giusto («di che colore era il cappotto
citato venti pagine fa?»), diventa impossibile. L'attenzione piena, che tiene
tutti gli scontrini, quel dettaglio ce l'ha ancora; il registro riassuntivo
può averlo perso.

`````

`````{tab} Superiore

Il limite è di **capacità**, ed è una conseguenza della dimensione finita. Lo
stato $S$ è una matrice $d \times d$: in uno spazio di dimensione $d$ non
esistono più di $d$ vettori mutuamente ortogonali. Finché le key
$\phi(k_1), \dots$ restano quasi ortogonali, leggere con $\phi(q)$ recupera il
value giusto quasi pulito; ma quando le associazioni memorizzate superano
l'ordine di $d$, non possono più stare tutte «separate», e il retrieval
raccoglie insieme al value cercato anche le briciole degli altri: il
cosiddetto **crosstalk**. Oltre la soglia la memoria va in sovraccarico e il
recupero associativo esatto degrada. Ed essendo la transizione l'identità, non
c'è modo di **dimenticare**: una scrittura spuria fatta all'inizio resta a
disturbare per sempre. È la ragione per cui l'attenzione lineare pura, così
com'è, resta indietro rispetto all'attenzione softmax proprio sui compiti di
richiamo preciso.

`````

Da qui in avanti l'intero capitolo è un tentativo di curare questi due mali
tenendo però il dono della ricorrenza lineare. Servono due ingredienti: un
modo per **dimenticare** (un *gate* che sbiadisca lo stato vecchio invece di
tenerlo eterno) e un modo per **correggere** invece di sommare alla cieca
(scrivere non il value intero ma solo l'*errore* rispetto a ciò che la memoria
già prevede per quella key, la cosiddetta *regola delta*). Sono esattamente i
due fili della prossima sezione.

```{admonition} Da ricordare
:class: important
- La softmax dei Transformer costa $O(n^2)$ perché non si fattorizza;
  sostituirla con una somiglianza $\text{sim}(q,k)=\phi(q)^\top\phi(k)$ e
  ri-associare il prodotto porta il costo a $O(n d^2)$, cioè **lineare** nella
  lunghezza.
- Il calcolo si condensa in uno **stato-matrice** $S = \sum_j v_j\,\phi(k_j)^\top$
  di dimensione fissa $d \times d$: una memoria chiave→valore che si legge con
  $S\,\phi(q)$.
- In forma causale è una **ricorrenza**,
  $S_t = S_{t-1} + v_t\,\phi(k_t)^\top$: cioè una **RNN a stato matriciale**
  con transizione lineare (l'identità), aggiornamento $O(d^2)$ per token e
  memoria **costante**; da cui l'ironia di *Transformers are RNNs*
  (Katharopoulos et al., 2020).
- Stessa funzione, **due forme**: parallela per addestrare (si sfruttano le
  GPU come nei Transformer), ricorrente per generare a memoria costante, senza
  la KV cache che invece cresce. La stessa dualità tornerà per gli State Space
  Model.
- Il difetto dell'accumulo puro: non dimentica e non corregge. In dimensione
  $d$ non stanno più di $d$ chiavi ortogonali; oltre quella soglia le
  associazioni interferiscono (**crosstalk**) e il richiamo esatto degrada.
- I due rimedi (un **gate** che dimentica e una **regola delta** che corregge
  invece di sommare) sono il filo delle sezioni successive.
```
