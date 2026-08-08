# Le architetture lineari: RetNet, RWKV, xLSTM

Le due sezioni precedenti hanno smontato i *meccanismi*: il trucco del kernel
che trasforma l'attenzione in una ricorrenza a stato fisso, i gate che fanno
sbiadire la memoria, la delta rule che corregge invece di accumulare. Erano
pezzi sciolti su un tavolo. In questa sezione li vediamo montati in macchine
intere: le architetture che, tra il 2023 e il 2025, hanno provato a fare
concorrenza al Transformer sul suo stesso terreno.

Ne guardiamo tre, scelte perché raccontano tre strade diverse verso la stessa
meta: **RetNet**, nata in un laboratorio industriale attorno a un'idea di
decadimento fisso; **RWKV**, cresciuta come progetto aperto di comunità;
**xLSTM**, in cui l'inventore della LSTM torna a rimettere mano alla propria
creatura. Sotto la carrozzeria, però, il motore è sempre lo stesso: una rete
ricorrente lineare a stato di dimensione fissa, che si addestra in parallelo
come un Transformer e fa inferenza un token alla volta a costo costante come
una RNN. A cambiare, da una all'altra, è quasi soltanto la **transizione di
stato** (il fattore che decide come la memoria di ieri sopravvive a oggi) e
l'ingegneria che la rende addestrabile su larga scala.

## RetNet: la retention e le sue tre forme

La prima architettura arriva da Microsoft Research e Tsinghua nel luglio 2023,
con Sun e colleghi {cite}`sun2023retnet`. Il nome del meccanismo è
**retention**, «ritenzione», e il suo gesto è tra i più semplici possibili: al
posto della softmax dell'attenzione, che normalizza i punteggi, si mette un
**decadimento esponenziale fisso**. Ogni coppia di posizioni pesa in base a
quanto dista nel tempo, e nient'altro.

Il punto interessante di RetNet non è tanto il meccanismo quanto il fatto che
lo stesso calcolo ammette **tre forme equivalenti**: tre modi di ottenere
esattamente lo stesso risultato, ciascuno conveniente in una situazione
diversa.

`````{tab} Elementare

Immaginate di dover fare la somma dei voti di una classe, dando più peso alle
interrogazioni recenti e meno a quelle vecchie. Potete farlo in tre modi, e il
totale non cambia. **Tutto insieme**: mettete tutti i voti in tabella, accanto
a ciascuno il suo peso, e sommate in un colpo solo (comodo se avete un foglio
di calcolo che macina tante moltiplicazioni in parallelo). **Uno alla volta**:
tenete un totale corrente e, a ogni nuova interrogazione, sbiadite un po' il
totale vecchio e ci aggiungete il voto nuovo; comodo quando i voti arrivano in
diretta, uno oggi e uno domani, e non volete rifare tutto da capo ogni volta.
**A blocchi**: sommate un mese per volta con il metodo veloce, e poi collegate
i totali mensili sbiadendo l'uno nell'altro; il compromesso per quando i voti
sono tantissimi.

RetNet è esattamente questo: la stessa somma pesata, in tre versioni. La prima
serve ad addestrare in fretta, la seconda a usare il modello parola per parola,
la terza per i testi lunghissimi. Il risultato è identico; cambia solo la
convenienza pratica.

`````

`````{tab} Superiore

Manteniamo la convenzione del capitolo: lo stato $S_t \in \mathbb{R}^{d\times d}$
è una memoria «chiave $\to$ valore», $q_t, k_t, v_t$ sono query, chiave e valore
del token $t$. Le tre forme della retention sono:

**Parallela** (addestramento). Come nell'attenzione, ma senza softmax:

$$
\text{Retention}(X) = \big(Q K^\top \odot D\big)\,V,
\qquad
D_{nm} =
\begin{cases}
\gamma^{\,n-m} & n \ge m \\[2pt]
0 & n < m
\end{cases}
$$

dove $Q, K, V$ sono le matrici di query, chiavi e valori, $\odot$ è il prodotto
elemento per elemento e $D$ è una **maschera causale con decadimento**: sostituisce
la normalizzazione softmax moltiplicando la coppia di posizioni $(n,m)$ per
$\gamma^{\,n-m}$, un peso che dipende *solo* dalla distanza $n-m$ e svanisce in modo
esponenziale ($0 < \gamma < 1$). Costa $O(n^2)$ come l'attenzione, ma tutte le
posizioni si calcolano insieme.

**Ricorrente** (inferenza, $O(1)$ per token). La stessa funzione, srotolata come
una RNN a stato matriciale:

$$
S_t = \gamma\, S_{t-1} + v_t\, k_t^\top,
\qquad
o_t = S_t\, q_t,
$$

dove $\gamma$ è il fattore di decadimento e $v_t k_t^\top$ è la nuova coppia
scritta in memoria. Ogni token costa un aggiornamento a memoria fissa: niente
cache che cresce.

**Chunkwise** (contesto lungo). Un ibrido: si spezza la sequenza in blocchi,
dentro ogni blocco si usa la forma parallela, tra un blocco e l'altro quella
ricorrente. Il costo diventa lineare in $n$, tenendo la parallelizzazione dentro
i blocchi.

Un dettaglio dà a RetNet la sua firma: la retention è **multi-scala**. Ogni
testa usa un $\gamma$ diverso (chi vicino a $1$ ricorda a lungo, chi più
piccolo dimentica in fretta), così che l'insieme delle teste copra orizzonti
temporali di durata diversa, dal contesto immediato a quello lontano.

`````

Vale la pena collocare RetNet nella famiglia che abbiamo costruito nella
sezione precedente. La sua ricorrenza $S_t = \gamma\, S_{t-1} + v_t k_t^\top$
è la riga «Mamba-2 / RetNet» della tabella unificante: un **decadimento
scalare**. La differenza, rispetto a Mamba-2 e alla GLA, è che qui $\gamma$ è
**fisso e data-indipendente**: scelto a priori, uguale per ogni parola,
imparato solo nel senso che si sceglie l'insieme dei valori per le teste. È la
forma più grossolana di oblio: efficace e a costo nullo, ma cieca al
contenuto. I gate appresi che abbiamo visto (lo scalare data-dipendente di
Mamba-2, il vettore diagonale della GLA) nascono proprio per superare questa
cecità, lasciando che sia l'input a decidere, token per token, cosa tenere e
cosa lasciar andare.

## RWKV: reinventare le RNN

La seconda architettura non esce da un laboratorio ma da una comunità.
**RWKV**, l'acronimo sta per le sue quattro componenti: *Receptance*,
*Weight*, *Key*, *Value*; è un progetto aperto guidato da Bo Peng, sviluppato
in pubblico da una comunità di ricercatori indipendenti. Il suo obiettivo
dichiarato è nel titolo del primo articolo: *«Reinventing RNNs for the
Transformer Era»*, reinventare le reti ricorrenti per l'epoca dei Transformer
{cite}`peng2023rwkv`.

Strutturalmente, un blocco RWKV impila due sotto-blocchi, per analogia con il
Transformer. Il **time-mixing** è il cuore che mescola l'informazione *nel
tempo*: fa il lavoro che nel Transformer fa l'attenzione, ma con una ricorrenza
lineare invece di un confronto tutti-contro-tutti. Il **channel-mixing** è una
piccola rete che mescola l'informazione *tra i canali* di uno stesso token, il
corrispettivo del blocco *feed-forward*. Entrambi si aprono con un **token-shift**:
una semplice interpolazione tra il token corrente e quello immediatamente
precedente, che dà alla rete un accesso diretto e gratuito al passo appena
trascorso.

`````{tab} Elementare

Il tratto che rende RWKV curioso è che è **due cose insieme, a seconda di come
lo guardi**. Durante l'addestramento si comporta come un Transformer: legge
tutta la sequenza in blocco e sfrutta le schede grafiche a pieno, in
parallelo. Durante l'uso (quando genera parola per parola) si comporta come
una RNN: tiene uno stato di dimensione fissa, lo aggiorna a ogni parola e non
conserva nulla del passato se non quel piccolo riassunto. È la stessa ricetta
cucinata in due modi: in cucina lavori come una catena di montaggio veloce, a
tavola servi un piatto alla volta. Questo è esattamente il graal che insegue
tutta la famiglia: addestramento parallelo *e* inferenza a memoria costante,
senza doverne sacrificare una.

`````

`````{tab} Superiore

Nella sua prima versione, la **RWKV-4**, il cuore del time-mixing è l'operatore
**WKV**, una forma di attenzione lineare con decadimento. Per un singolo canale:

$$
\text{wkv}_t =
\frac{\displaystyle\sum_{i<t} e^{-(t-1-i)\,w + k_i}\, v_i \;+\; e^{\,u + k_t}\, v_t}
     {\displaystyle\sum_{i<t} e^{-(t-1-i)\,w + k_i} \;+\; e^{\,u + k_t}} ,
$$

dove $k_i$ e $v_i$ sono chiave e valore alla posizione $i$, $w > 0$ è il
**decadimento** del canale (il peso di un token svanisce come $e^{-(t-1-i)w}$
al crescere della distanza) e $u$ è un **bonus** riservato al token corrente,
che lo esenta dal decadimento così che il presente non venga penalizzato quanto
il passato. Numeratore e denominatore sono la classica media pesata: il
denominatore è il **normalizzatore**, la somma dei pesi, nella stessa impostazione
di Katharopoulos e Schlag e diversa dalla famiglia senza normalizzatore vista
nella sezione precedente. In RWKV-4 il decadimento $w$ è appreso ma **fisso**
(uno per canale, non dipende dall'input): la transizione di stato è dunque un
decadimento diagonale data-indipendente.

L'architettura è poi evoluta in due tappe. **RWKV-5/6**, nome in codice
*Eagle* e *Finch* {cite}`peng2024eagle`, promuove lo stato da vettore a
**matrice** (multi-testa, come qui) e rende il decadimento
**data-dipendente**: in Finch (v6) il fattore di oblio è generato dall'input,
avvicinando RWKV alla GLA. **RWKV-7**, nome in codice *Goose*
{cite}`peng2025rwkv7`, compie il salto più netto: adotta una **delta rule
generalizzata**, con l'evoluzione di stato (nella convenzione del capitolo)

$$
S_t = S_{t-1}\,\big(\operatorname{Diag}(w_t) - \hat{\kappa}_t\,(a_t \odot \hat{\kappa}_t)^\top\big) + v_t\, \tilde{k}_t^\top ,
$$

dove $w_t$ è un decadimento vettoriale, un valore per canale (l'erede del gate
diagonale di Finch), $\hat{\kappa}_t$ è una chiave di *rimozione* normalizzata
e disaccoppiata dalla chiave di scrittura $\tilde{k}_t$, e $a_t$ è un tasso di
apprendimento **appreso in contesto**, anch'esso canale per canale. La
transizione di stato è dunque un fattore diagonale più una correzione di rango
uno: un **gated-delta**, la stessa famiglia dell'ultima riga della tabella
unificante. Questo dà a RWKV-7 una
capacità di *state tracking* che le versioni precedenti non avevano: gli
autori mostrano che riconosce tutti i linguaggi regolari pur mantenendo
l'addestramento parallelo, e argomentano che, sotto le congetture standard
della teoria della complessità, ciò eccede quanto un Transformer a profondità
fissa può fare (che resta confinato nella classe $\text{TC}^0$).

`````

RWKV ha una particolarità sociologica che vale la pena notare, in un campo
dominato dai grandi laboratori: RWKV-4 è stata scalata fino a 14 miliardi di
parametri (la più grande RNN densa del suo tempo) e RWKV-7 è distribuita con
pesi aperti sotto licenza Apache 2.0, in una gamma di taglie da circa 0,19 a
2,9 miliardi di parametri. È la dimostrazione che un'architettura competitiva
può crescere fuori dai recinti industriali.

## xLSTM: il ritorno di Hochreiter

La terza architettura ha il sapore di un ritorno. Nel capitolo sull'NLP abbiamo
studiato la **LSTM** {cite}`hochreiter1997long`, la cella con uno stato di
memoria $c_t$ governato da alcuni interruttori, i *gate*: è la cella che negli
anni Novanta risolse il problema del gradiente che svanisce e che per un
decennio ha dominato l'elaborazione delle sequenze. All'inizio i gate erano
due, uno per far entrare l'informazione (*input*) e uno per farla uscire
(*output*). Il terzo, quello che lascia sbiadire la memoria vecchia
(*forget*), arriva nel 2000 con Gers, Schmidhuber e Cummins
{cite}`gers2000learning`, ed è la forma a tre gate che oggi tutti chiamano
LSTM. Nel 2024 uno dei suoi due inventori,
**Sepp Hochreiter**, torna sulla propria creatura e la aggiorna per l'era dei
Transformer. Il risultato è **xLSTM**, di Beck e colleghi, presentato a
NeurIPS 2024 {cite}`beck2024xlstm`.

La domanda di partenza è schietta: che cosa mancava alla LSTM per reggere il
confronto? Due cose, secondo gli autori. Primo, un modo di **rivedere le decisioni
di memoria** in modo più netto: da qui il *gating esponenziale*. Secondo, una
memoria più **capiente e parallelizzabile**: da qui il passaggio da una cella
scalare a una matriciale. xLSTM offre due tipi di blocco, che rispondono a queste
due esigenze.

`````{tab} Elementare

Pensate alla vecchia LSTM come a un magazziniere con un unico scaffale e tre
interruttori: uno per buttare via, uno per riporre, uno per mostrare cosa c'è.
Ha funzionato per anni, ma lo scaffale è piccolo e gli interruttori sono
delicati. xLSTM è lo stesso magazziniere che riapre bottega in grande. Nella
prima variante tiene lo scaffale singolo ma installa interruttori più decisi,
che possono spalancare o chiudere la memoria di colpo invece che a metà. Nella
seconda sostituisce lo scaffale con un intero **archivio a griglia**, dove
ogni richiesta («dammi il valore di questa chiave») pesca in una tabella molto
più grande, e, dettaglio decisivo, questo archivio si può riempire tutto in
una volta in parallelo, non una casella alla volta. È la LSTM di trent'anni
fa, rifatta con la memoria e i muscoli di oggi.

`````

`````{tab} Superiore

**sLSTM** (memoria *scalare*) conserva la struttura classica ma introduce il
**gating esponenziale**. La cella e il suo normalizzatore evolvono come

$$
c_t = f_t\, c_{t-1} + i_t\, z_t,
\qquad
n_t = f_t\, n_{t-1} + i_t,
\qquad
h_t = o_t \,\frac{c_t}{n_t},
$$

dove $z_t$ è l'input candidato, $f_t, o_t$ i gate di *forget* e *output* e
$i_t = \exp(\tilde{\imath}_t)$ è il gate di *input* reso **esponenziale**. Il
denominatore $n_t$ è un normalizzatore che accumula i gate di input, così che la
lettura $c_t/n_t$ resti una media ben scalata. La sLSTM ha una *memory mixing*
tra le teste ed è, per costruzione, **ricorrente non parallelizzabile**: si valuta
con un kernel sequenziale.

**mLSTM** (memoria *matriciale*) è la variante pensata per le GPU. Lo stato
diventa una matrice $C_t \in \mathbb{R}^{d\times d}$ aggiornata con una
**regola di covarianza**, un prodotto esterno, esattamente come
nell'attenzione lineare:

$$
C_t = f_t\, C_{t-1} + i_t\, v_t\, k_t^\top,
\qquad
n_t = f_t\, n_{t-1} + i_t\, k_t,
\qquad
h_t = o_t \odot \frac{C_t\, q_t}{\max\!\big(|n_t^\top q_t|,\, 1\big)} ,
$$

dove $q_t, k_t, v_t$ sono query, chiave e valore, $f_t$ e $i_t$ i gate di forget e
input, e il denominatore normalizza la lettura. Senza *memory mixing*, la mLSTM è
**completamente parallelizzabile**: di fatto è una *gated linear attention* con
gating esponenziale, in cui la transizione di stato è il decadimento scalare
$f_t$ moltiplicato per l'identità.

Resta un problema numerico. Un gate esponenziale $i_t = \exp(\tilde{\imath}_t)$
può esplodere. La cura è un **stabilizzatore** in scala logaritmica, uno stato

$$
m_t = \max\!\big(\log f_t + m_{t-1},\; \log i_t\big),
$$

che tiene il massimo corrente dei logaritmi dei gate e viene sottratto prima di
esponenziare: è il classico trucco *log-sum-exp*, che mantiene i conti in un
intervallo numerico sicuro senza cambiare il risultato.

`````

La mLSTM si è rivelata la variante di maggior peso pratico. Nel 2025 gli stessi
autori presentano **xLSTM-7B** {cite}`beck2025xlstm7b`, un modello da 7 miliardi
di parametri costruito su **sole celle mLSTM** e addestrato su 2,3 mila miliardi
di token: la prova che la formula regge alla scala dei grandi modelli linguistici
di produzione, con l'inferenza a memoria costante che l'architettura ricorrente
garantisce.

## Lo stesso scheletro

Tre architetture, tre storie (un laboratorio industriale, una comunità aperta,
il ritorno di un pioniere) e tre insiemi di scelte ingegneristiche. Eppure, se
si toglie la carrozzeria, sotto c'è sempre lo stesso telaio. RetNet, RWKV e
xLSTM sono tutte **reti ricorrenti lineari a stato di dimensione fissa**:
tengono una memoria che si scrive per prodotto esterno, si addestrano in
parallelo e si usano in modo ricorrente a costo costante per token. Sono,
insieme a GLA e DeltaNet della sezione precedente, variazioni sullo stesso
tema.

E il tema è quello della tabella unificante che abbiamo costruito poco fa: a
cambiare, da un'architettura all'altra, è **quasi soltanto la transizione di
stato**. RetNet la fissa a un decadimento scalare data-indipendente
($\gamma I$); la mLSTM di xLSTM la rende uno scalare data-dipendente con
gating esponenziale; RWKV la fa evolvere nel tempo, dal decadimento diagonale
fisso della v4 a quello data-dipendente della v6, fino alla transizione
gated-delta della v7: lo stesso gradino di espressività che, nella famiglia di
Yang, separa la GLA dal Gated DeltaNet. Nomi, sigle e comunità diversi
descrivono, in fondo, lo stesso zoo di matrici di transizione.

Questa unità apparecchia il prossimo capitolo. Gli **State Space Model** (S4,
Mamba e i loro discendenti) arrivano esattamente allo stesso posto, ma da
tutt'altra strada: non quella dell'attenzione da linearizzare, bensì quella
dei **sistemi dinamici continui**, discretizzati passo dopo passo. Vedremo che
il punto d'arrivo coincide: anche un SSM è una ricorrenza lineare a stato
fisso con le sue due forme, parallela e ricorrente. E vedremo che non è una
coincidenza: Mamba-2, con la sua *dualità* tra stato e attenzione, dimostrerà
che le due famiglie (le attenzioni lineari di questo capitolo e gli SSM del
prossimo) sono due viste della stessa cosa.

Un'ultima onestà, prima di proseguire. Nessuna di queste architetture ha
«ucciso» il Transformer, e nessuna lo farà a breve. Lo stato di dimensione
fissa, che è la loro forza in efficienza, è anche il loro limite: quando serve
ritrovare un dettaglio preciso in un contesto molto lungo (il *recall
associativo esatto*) l'attenzione piena, che conserva ogni token, resta
superiore. È da qui che nascono gli **ibridi**, che alternano pochi strati di
attenzione a molti strati lineari. Ma questi limiti, e il modo in cui
l'ecosistema li sta affrontando, si capiscono meglio dopo aver visto anche
l'altra metà della famiglia: li riprenderemo alla fine del prossimo capitolo.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- **RetNet** {cite}`sun2023retnet` toglie la normalizzazione dell'attenzione e la
  sostituisce con un **peso che sbiadisce con la distanza**, sempre lo stesso:
  quanto conta una parola dipende solo da quanto è lontana. È la somma dei voti
  della classe, con le interrogazioni recenti che pesano più delle vecchie, e si
  può fare nei tre modi che danno lo stesso totale: tutto insieme (per addestrare
  in fretta), uno alla volta tenendo un totale corrente (per generare, a costo
  fisso per parola), a blocchi (per i testi lunghissimi).
- Quel ritmo di sbiadimento, però, è **deciso a priori e uguale per ogni
  parola**: la forma più grossolana di oblio, cieca al contenuto, all'opposto
  dello sbiadimento che nella sezione precedente si regolava da sé, parola per
  parola e zona per zona della lavagna.
- **RWKV** {cite}`peng2023rwkv`, progetto aperto di comunità, alterna due
  blocchi: uno mescola l'informazione fra le parole (il mestiere
  dell'attenzione), l'altro rimescola fra loro i numeri con cui è scritta una
  singola parola. È la stessa ricetta cucinata in due modi: in addestramento
  lavora come una catena di montaggio in parallelo, in uso serve un piatto alla
  volta tenendo un riassunto di taglia fissa.
- Le sue versioni successive {cite}`peng2024eagle` {cite}`peng2025rwkv7` salgono
  gli stessi gradini della sezione precedente: prima uno sbiadimento fissato una
  volta per tutte, poi deciso parola per parola, infine una versione che, prima
  di scrivere, corregge quello che c'è già.
- **xLSTM** {cite}`beck2024xlstm` riapre la bottega della vecchia LSTM di
  Hochreiter {cite}`hochreiter1997long`, il magazziniere con un solo scaffale e
  tre interruttori. Rimette interruttori più decisi (che spalancano o chiudono
  di colpo, tenuti a bada da un accorgimento di calcolo perché non esplodano) e
  apre due botteghe: quella con l'unico scaffale, che si riempie una casella
  alla volta, e quella con un **archivio a griglia**, che si riempie tutto
  insieme in parallelo. Un modello da sette miliardi di parametri costruito solo
  su quest'ultima {cite}`beck2025xlstm7b` mostra che la formula regge alla scala
  dei grandi modelli.
- Il filo comune: le tre architetture di questa sezione, e quelle della
  precedente, sono la stessa cosa: un riassunto di taglia fissa aggiornato
  parola per parola. A cambiare è **solo il modo in cui la memoria di ieri
  sopravvive a oggi**. Il prossimo capitolo arriverà allo stesso motore partendo
  da tutt'altra strada.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- **RetNet** {cite}`sun2023retnet` sostituisce la softmax con un **decadimento
  esponenziale fisso** $\gamma$ e offre lo stesso calcolo in **tre forme
  equivalenti**: parallela (addestramento), ricorrente
  $S_t = \gamma S_{t-1} + \mathbf{v}_t \mathbf{k}_t^\top$
  (inferenza $O(1)$ per token), chunkwise (contesto lungo). È
  multi-scala: ogni testa usa un $\gamma$ diverso.
- Il decadimento di RetNet è **scalare, fisso e data-indipendente**: la forma più
  grossolana di oblio, all'opposto dei gate appresi di Mamba-2 e GLA.
- **RWKV** {cite}`peng2023rwkv`, progetto aperto di comunità, alterna
  *time-mixing* (attention-like) e *channel-mixing* (FFN-like) con *token-shift*:
  si addestra come un Transformer, si usa come una RNN a stato costante.
- L'evoluzione di RWKV va dall'operatore WKV a decadimento fisso (v4) allo stato
  matriciale con decadimento data-dipendente (v5/6 *Eagle/Finch*
  {cite}`peng2024eagle`) fino alla **delta rule generalizzata** di v7 *Goose*
  {cite}`peng2025rwkv7`, capace di *state tracking* e di riconoscere i linguaggi
  regolari.
- **xLSTM** {cite}`beck2024xlstm` aggiorna la LSTM di Hochreiter
  {cite}`hochreiter1997long` con **gating esponenziale** (stabilizzato in scala
  log) e due celle: **sLSTM** (memoria scalare, non parallelizzabile) e **mLSTM**
  (memoria matriciale
  $C_t = f_t C_{t-1} + i_t \mathbf{v}_t \mathbf{k}_t^\top$,
  parallelizzabile,
  di fatto una gated linear attention). **xLSTM-7B** {cite}`beck2025xlstm7b` la
  porta alla scala dei grandi modelli.
- Il filo comune: RetNet, RWKV e xLSTM (con GLA e DeltaNet) sono la stessa
  **RNN lineare a stato fisso**; cambia **solo la transizione di stato**. Gli
  State Space Model del prossimo capitolo arriveranno allo stesso punto da
  un'altra strada, e Mamba-2 dimostrerà che sono la stessa cosa.
```

`````
