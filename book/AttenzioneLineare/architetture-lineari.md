# Le architetture lineari: RetNet, RWKV, xLSTM

Le due sezioni precedenti hanno smontato i *meccanismi*: il modo di riassumere
che trasforma l'attenzione in una memoria di taglia fissa, gli interruttori che
la fanno sbiadire, la regola che corregge una voce invece di sommarci sopra.
Erano pezzi sciolti su un tavolo. In questa sezione li vediamo montati in
macchine intere: le architetture che, tra il 2023 e il 2025, hanno provato a
fare concorrenza al Transformer sul suo stesso terreno.

Ne guardiamo tre, scelte perché raccontano tre strade diverse verso la stessa
meta: **RetNet**, nata in un laboratorio industriale attorno a un'idea di
decadimento fisso; **RWKV**, cresciuta come progetto aperto di comunità;
**xLSTM**, in cui uno degli inventori della LSTM torna a rimettere mano alla
propria creatura.

Sotto la carrozzeria, però, il motore è sempre lo stesso, ed è quello delle due
sezioni precedenti: una memoria di taglia fissa, aggiornata una volta per
parola. Si riempie tutta insieme mentre il modello impara, sfruttando tutte le
unità di calcolo della scheda grafica come fa un Transformer, e si rilegge una
parola alla volta mentre il modello scrive, pagando ogni parola sempre lo
stesso, come faceva una vecchia rete ricorrente. A cambiare, da un'architettura
all'altra, è quasi soltanto la **transizione di stato** (il fattore che decide
come la memoria di ieri sopravvive a oggi) e l'ingegneria che la rende
addestrabile su larga scala.

## RetNet: la retention e le sue tre forme

La prima architettura arriva da Microsoft Research e Tsinghua nel luglio 2023,
con Sun e colleghi {cite}`sun2023retnet`. Il nome del meccanismo è
**retention**, «ritenzione», e il suo gesto è tra i più semplici possibili: al
posto della softmax dell'attenzione, che spartisce i pesi fra tutte le parole,
si mette un **decadimento esponenziale fisso**. Le parole continuano a
confrontarsi fra loro come nell'attenzione; quello che cambia è che il
punteggio del confronto non viene più spartito, ma soltanto moltiplicato per un
peso che dipende da quante parole ci sono in mezzo e svanisce con la distanza.
Quel peso, ed è il punto, non guarda mai il contenuto: sbiadisce allo stesso
modo una data e un intercalare.

Il punto interessante di RetNet non è tanto il meccanismo quanto il fatto che
lo stesso calcolo ammette **tre forme equivalenti**: tre modi di ottenere
esattamente lo stesso risultato, ciascuno conveniente in una situazione
diversa.

`````{tab} Elementare

Immagina di dover fare la somma dei voti di uno studente lungo tutto l'anno,
dando più peso alle interrogazioni recenti e meno a quelle vecchie. (Nel
modello vero ogni voto porta con sé anche quanto quella parola c'entra con la
domanda; qui teniamo solo il peso della distanza, che è la novità di RetNet.)
Puoi farlo in tre modi, e il totale non cambia. **Tutto insieme**: metti tutti i
voti in tabella, accanto a ciascuno il suo peso, e sommi in un colpo solo
(comodo se hai un foglio
di calcolo che macina tante moltiplicazioni in parallelo). **Uno alla volta**:
tieni un totale corrente e, a ogni nuova interrogazione, sbiadisci un po’ il
totale vecchio e ci aggiungi il voto nuovo; comodo quando i voti arrivano in
diretta, uno oggi e uno domani, e non vuoi rifare tutto da capo ogni volta.
**A blocchi**: sommi un mese per volta con il metodo veloce, e poi colleghi
i totali mensili sbiadendo l'uno nell'altro. Serve perché il totale finale, in
realtà, non basta: ne vuole uno **dopo ogni interrogazione** (com'era messo lo
studente a ottobre, a novembre, a dicembre), e per averli tutti in un colpo
solo il primo metodo pretende una tabella con una riga per ogni momento e una
colonna per ogni voto. Su un anno intero è di nuovo la tabella grande da cui
siamo scappati all'inizio del capitolo; dentro un mese resta piccola.

Vale la pena vederlo con i numeri, perché «il totale non cambia» è la cosa che
regge tutto. Tre voti in ordine, $6$, $7$ e $8$, e a ogni passo quello che c'è
già si dimezza. Il totale dopo il terzo voto viene così:

- Tutto insieme: l'ultimo voto vale pieno, il precedente metà, quello prima
  ancora un quarto, cioè $8 + 0{,}5\times 7 + 0{,}25\times 6 = 13$.
- Uno alla volta: parto da $6$; arriva il $7$ e faccio $0{,}5\times 6 + 7 = 10$;
  arriva l’$8$ e faccio $0{,}5\times 10 + 8 = 13$. (E per strada mi sono
  ritrovato in mano anche i totali intermedi, $6$ e $10$.)
- A blocchi, con blocchi da due: i primi due voti li peso e li sommo in un
  colpo solo, senza passare per il totale del primo, $7 + 0{,}5\times 6 = 10$;
  poi il blocco successivo riparte da quel $10$ e ci attacca il terzo,
  $0{,}5\times 10 + 8 = 13$.

Tredici tutte e tre le volte. RetNet è esattamente questo: la stessa somma
pesata, in tre versioni. La prima serve ad addestrare in fretta, la seconda a
usare il modello parola per parola, la terza per i testi lunghissimi. Il
risultato è identico; cambia solo la convenienza pratica.

`````

`````{tab} Superiore

Manteniamo la convenzione del capitolo: lo stato $\mathbf{S}_t \in \mathbb{R}^{d\times d}$
è una memoria «chiave $\to$ valore», $\mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t$ sono query, chiave e valore
del token $t$. Le tre forme della retention sono:

**Parallela** (addestramento). Come nell'attenzione, ma senza softmax:

$$
\text{Retention}(\mathbf{X}) = \big(\mathbf{Q} \mathbf{K}^\top \odot \mathbf{D}\big)\,\mathbf{V},
\qquad
D_{ij} =
\begin{cases}
\gamma^{\,i-j} & i \ge j \\[2pt]
0 & i < j
\end{cases}
$$

dove $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ sono le matrici di query, chiavi e valori, $\odot$ è il prodotto
elemento per elemento e $\mathbf{D}$ è una **maschera causale con decadimento**: sostituisce
la normalizzazione softmax moltiplicando la coppia di posizioni $(i,j)$ per
$\gamma^{\,i-j}$, un peso che dipende *solo* dalla distanza $i-j$ e svanisce in modo
esponenziale ($0 < \gamma < 1$). Costa $O(n^2 d)$ come l'attenzione, ma tutte le
posizioni si calcolano insieme.

Questa è la forma essenziale. Il paper vi affianca due cose che non cambiano il
discorso ma è onesto nominare: una rotazione di fase (*xPos*) che convive con
il decadimento e fa da codifica posizionale relativa (il decadimento completo è
$\gamma e^{i\theta}$, dove il modulo dimentica e la fase codifica la
posizione), e tre riscalature dei punteggi che servono solo a tenere i conti in
un intervallo numerico sicuro, cioè a rendere praticabile proprio
l'equivalenza fra le tre forme.

**Ricorrente** (inferenza, costo costante nella lunghezza: $O(d^2)$ per token,
come nella sezione precedente). La stessa funzione, srotolata come
una RNN a stato matriciale:

$$
\mathbf{S}_t = \gamma\, \mathbf{S}_{t-1} + \mathbf{v}_t\, \mathbf{k}_t^\top,
\qquad
\mathbf{o}_t = \mathbf{S}_t\, \mathbf{q}_t,
$$

dove $\gamma$ è il fattore di decadimento e $\mathbf{v}_t \mathbf{k}_t^\top$ è la nuova coppia
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
sezione precedente: è il primo dei tre gradini dello sbiadimento, quello in cui
il ritmo con cui la memoria si scolora è **deciso una volta per tutte** quando
il modello viene progettato, uguale per ogni parola e per ogni sua parte. È
la forma più grossolana di oblio: efficace e a costo nullo, ma cieca al
contenuto, perché sbiadisce con lo stesso ritmo una data e un intercalare. Gli
altri due gradini, che abbiamo già incontrato, nascono proprio per superare
questa cecità: **Mamba-2**, che il ritmo lo ricalcola a ogni parola guardando
cosa sta leggendo, e **GLA** (*gated linear attention*), che oltre a
ricalcolarlo lo differenzia zona per zona della memoria. Sono i tre modi di
decidere quanto dimenticare, dal più rigido al più libero.

## RWKV: reinventare le RNN

La seconda architettura non esce da un laboratorio ma da una comunità.
**RWKV** è un progetto aperto guidato da Bo Peng, sviluppato in pubblico da una
comunità di ricercatori indipendenti. Il suo obiettivo dichiarato è nel titolo
del primo articolo: *«Reinventing RNNs for the Transformer Era»*, reinventare
le reti ricorrenti per l'epoca dei Transformer {cite}`peng2023rwkv`.

Strutturalmente, un blocco RWKV impila due pezzi che si alternano, per analogia
con il Transformer: uno **allarga lo sguardo**, l'altro lo **approfondisce**.
Il primo mescola l'informazione **fra le parole**, cioè fa
il mestiere che nel Transformer fa l'attenzione, ma con una memoria che si
aggiorna parola per parola invece che con un confronto tutti-contro-tutti: si
chiama *time-mixing*, mescolamento nel tempo. Il secondo rimescola fra loro i
numeri con cui è scritta **una singola parola**, senza guardare le altre: sono
i *canali* di cui si è già detto, le posizioni della fila di numeri con cui il
modello scrive ogni parola. Questo secondo pezzo si chiama *channel-mixing*, ed
è il blocco che nel Transformer sta dopo l'attenzione, il *feed-forward*.
Entrambi si aprono con un *token-shift*, che è la mossa più semplice del mondo:
invece di guardare solo la parola corrente, si guarda una miscela fra la parola
corrente e quella appena prima, un tanto dell'una e un tanto dell'altra. Costa
niente e dà alla rete un accesso diretto al passo appena trascorso.

`````{tab} Elementare

Perché servano tutti e due si vede togliendoli. Senza il primo la frase
resterebbe un elenco di parole che non si parlano; senza il secondo il modello
saprebbe chi parla con chi ma capirebbe poco di ciascuno. Si alternano per tutta l'altezza della rete, ed è
la stessa divisione del lavoro dei Transformer: di tutto questo capitolo,
soltanto il primo dei due ci riguarda, perché è lì che sta la memoria di taglia
fissa.

Il nome, poi, è la lista dei quattro ingredienti del primo pezzo:
*Receptance*, *Weight*, *Key*, *Value*. Le ultime tre le conosciamo (il peso
che sbiadisce, l'etichetta, l'informazione); la *receptance* è un rubinetto
d'uscita, che decide quanta parte di ciò che la memoria risponde viene
effettivamente lasciata passare al resto della rete.

Un'ultima cosa da sapere, perché il seguito ci conta sopra: RWKV non è un
modello solo, è una famiglia che ha cambiato pelle più volte, e le versioni si
contano con un numero. Le tappe che ci riguardano sono quattro. La quarta
(2023) non tiene ancora un foglio a righe e colonne, ma una fila di numeri che
sbiadiscono con ritmi decisi una volta per tutte. La quinta (2024) sostituisce
quella fila con il foglio che conosciamo. La sesta quei ritmi li ricalcola a
ogni parola, zona per zona, come GLA. La settima (2025), prima di scrivere,
corregge quello che c'è già, come faceva la rubrica di Mario.

`````

`````{tab} Superiore

Nella sua prima versione, la **RWKV-4**, il cuore del time-mixing è l'operatore
**WKV**, una forma di attenzione lineare con decadimento. Per un singolo canale:

$$
\text{wkv}_t =
\frac{\displaystyle\sum_{i<t} e^{-(t-1-i)\,w + k_i}\, v_i \;+\; e^{\,u + k_t}\, v_t}
     {\displaystyle\sum_{i<t} e^{-(t-1-i)\,w + k_i} \;+\; e^{\,u + k_t}} ,
$$

dove $k_i$ e $v_i$ sono chiave e valore alla posizione $i$, $w \ge 0$ è il
**decadimento** del canale (il peso di un token svanisce come $e^{-(t-1-i)w}$
al crescere della distanza, e il caso limite $w = 0$ è il canale che non
dimentica) e $u$ è un **bonus** riservato al token corrente,
che lo esenta dal decadimento così che il presente non venga penalizzato quanto
il passato. Numeratore e denominatore sono la classica media pesata: il
denominatore è il **normalizzatore**, la somma dei pesi, nella stessa impostazione
di Katharopoulos, che il normalizzatore lo tiene, e diversa dalla famiglia
senza normalizzatore vista nella sezione precedente. In RWKV-4 il decadimento
$w$ è appreso ma **fisso** (uno per canale, non dipende dall'input): la
transizione di stato è dunque un decadimento diagonale data-indipendente.

Da notare, perché conta per il riepilogo di fine sezione: in RWKV-4 lo stato è
un vettore per canale e la scrittura è elemento per elemento, non un prodotto
esterno. Nella formula non compare nessuna query, e infatti non c'è: la
*receptance* $r$ è un gate d'uscita, non un termine di affinità. Il prodotto
esterno arriva con la v5.

L'architettura è poi evoluta in due tappe. **RWKV-5/6**, nome in codice
*Eagle* e *Finch* {cite}`peng2024eagle`, promuove lo stato da vettore a
**matrice** (multi-testa, come qui) e rende il decadimento
**data-dipendente**: in Finch (v6) il fattore di oblio è generato dall'input,
avvicinando RWKV alla GLA. **RWKV-7**, nome in codice *Goose*
{cite}`peng2025rwkv7`, compie il salto più netto: adotta una **delta rule
generalizzata**, con l'evoluzione di stato (nella convenzione del capitolo)

$$
\mathbf{S}_t = \mathbf{S}_{t-1}\,\big(\operatorname{Diag}(\mathbf{w}_t) - \hat{\boldsymbol{\kappa}}_t\,(\mathbf{a}_t \odot \hat{\boldsymbol{\kappa}}_t)^\top\big) + \mathbf{v}_t\, \tilde{\mathbf{k}}_t^\top ,
$$

dove $\mathbf{w}_t$ è un decadimento vettoriale, un valore per canale (l'erede del gate
diagonale di Finch), $\hat{\boldsymbol{\kappa}}_t$ è una chiave di *rimozione* normalizzata
e disaccoppiata dalla chiave di scrittura $\tilde{\mathbf{k}}_t$, e $\mathbf{a}_t$ è un tasso di
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
parametri, i numeri che il modello impara e che ne misurano la taglia (era la
più grande rete ricorrente *densa* del suo tempo, cioè fra quelle che usano
tutti i propri parametri a ogni parola), e il lavoro che presenta RWKV-7
rilascia quattro modelli, da 190 milioni a 2,9 miliardi di parametri, con pesi
aperti sotto licenza Apache 2.0, cioè scaricabili e riusabili da chiunque. È la dimostrazione che un'architettura
competitiva può crescere fuori dai recinti industriali.

## xLSTM: il ritorno di Hochreiter

La terza architettura ha il sapore di un ritorno. Nel capitolo sull'NLP abbiamo
studiato la **LSTM** {cite}`hochreiter1997long`: una cella che tiene una
memoria e la governa con alcuni interruttori, i *gate*. Negli anni Novanta
risolse un problema che sembrava senza uscita, quello di una rete ricorrente
che su una sequenza lunga smetteva semplicemente di imparare (il gradiente che
svanisce), e per un decennio ha dominato l'elaborazione delle sequenze.

All'inizio i gate erano due: uno per far entrare l'informazione (*input*), uno
per farla uscire (*output*). Il terzo, quello che lascia sbiadire la memoria
vecchia (*forget*), arriva nel 2000 con Gers, Schmidhuber e Cummins
{cite}`gers2000learning`, ed è la forma a tre gate che oggi tutti chiamano
LSTM.

Nel 2024 uno dei suoi due inventori, **Sepp Hochreiter**, torna sulla propria
creatura e la aggiorna per l'era dei Transformer. Il risultato è **xLSTM**, di
Beck e colleghi, presentato a NeurIPS 2024 {cite}`beck2024xlstm`.

La domanda di partenza è schietta: che cosa mancava alla LSTM per reggere il
confronto? Due cose, secondo gli autori. Primo, un modo di **rivedere le
decisioni di memoria** in modo più netto, cioè interruttori capaci di
spalancarsi davvero invece di fermarsi sempre un po’ prima (nei paper si chiama
*gating esponenziale*). Secondo, una memoria più **capiente** e che si possa
riempire tutta insieme invece che una casella per volta: da qui il passaggio da
una cella con un solo posto a una cella a griglia (da *scalare* a
*matriciale*). xLSTM offre due tipi di blocco, che rispondono a queste due
esigenze.

`````{tab} Elementare

Pensa alla vecchia LSTM come a un magazziniere con un unico scaffale e tre
interruttori: uno per buttare via, uno per riporre, uno per mostrare cosa c'è.
Ha funzionato per anni, ma lo scaffale è piccolo e gli interruttori sono
timidi. Timidi in un senso preciso: sono manopole che non arrivano mai al
fondo, si fermano sempre un po’ prima dello spalancato e un po’ prima del
chiuso. Non è un difetto di fabbrica, è come sono fatte: dentro c'è un conto che
prende qualunque numero, anche enorme, e lo riporta dentro l'intervallo fra
zero e uno. È comodo, perché così niente sfugge di mano, ma impedisce a una
manopola di dare più di «tutto». Se il
magazziniere si accorge, dopo mille articoli, che quello di oggi conta più di
tutti gli altri messi insieme, non riesce a dargli molto più spazio degli
altri: al massimo un po’ di più. xLSTM è lo stesso magazziniere che riapre
bottega in grande.

Nella prima variante (nei paper si chiama **sLSTM**) tiene lo scaffale singolo
e cambia gli interruttori: i
nuovi possono spalancarsi davvero, perché al posto di quel conto che schiaccia
ne usano uno che lascia salire senza tetto, e allora una cosa importante entra
occupando quanto merita, anche molto più di tutte le precedenti. Il prezzo lo
paga in un altro modo: quando le manopole possono arrivare a valori enormi, i
conti rischiano di andare fuori scala, e serve un accorgimento apposito per
tenerli buoni.

Nella seconda variante (nei paper si chiama **mLSTM**, e sarà quella che conta)
sostituisce lo scaffale con un intero **archivio a griglia**, dove ogni
richiesta («dammi l'informazione di questa etichetta») pesca in una tabella
molto più grande: è di nuovo il foglio-registro delle sezioni precedenti.
Quanto sbiadire lo decide guardando l'articolo che sta arrivando, quindi parola
per parola, come faceva Mamba-2.

Questa seconda bottega ha poi un vantaggio che non si vede a occhio, e non è la
capienza. Dove va un articolo dipende solo dall'articolo, non da com'è messo
l'archivio in quel momento: nessun addetto deve aspettare che il collega abbia
finito per sapere che cosa fare, e allora mille addetti possono sistemare mille
articoli **contemporaneamente**. Nella prima bottega no: per decidere quanto
aprire gli interruttori il magazziniere guarda com'è ridotto lo scaffale
adesso, e finché non ha sistemato l'articolo di oggi non sa come regolarsi
domani, quindi si procede per forza in fila. (È lo stesso motivo per cui, nella
sezione precedente, correggere era lento e sbiadire no.) È la LSTM di
trent'anni fa, rifatta con la memoria e i muscoli di oggi.

`````

`````{tab} Superiore

**sLSTM** (memoria *scalare*) conserva la struttura classica ma introduce il
**gating esponenziale**. Qui ogni cella tiene un numero solo, e le formule si
leggono per una cella alla volta: la memoria e il suo normalizzatore evolvono
come

$$
c_t = f_t\, c_{t-1} + i_t\, z_t,
\qquad
n_t = f_t\, n_{t-1} + i_t,
\qquad
h_t = o_t\, \frac{c_t}{n_t},
$$

dove $z_t$ è l'input candidato, $f_t, o_t$ i gate di *forget* e *output* e
$i_t = \exp(\tilde{\imath}_t)$ è il gate di *input* reso **esponenziale**. Il
denominatore $n_t$ è un normalizzatore che accumula i gate di input, così che la
lettura $c_t/n_t$ resti una media ben scalata. La sLSTM ha una *memory mixing*
fra le celle di una stessa testa, e non fra teste diverse: è per questo che i
suoi parametri ricorrenti sono $d^2/N_h$ invece di $d^2$, con $N_h$ il numero
di teste. Proprio quel mescolamento (i collegamenti da stato a stato) la rende,
per costruzione, **ricorrente non parallelizzabile**: si valuta con un kernel
sequenziale, che gli autori hanno però scritto in CUDA e reso veloce.

**mLSTM** (memoria *matriciale*) è la variante pensata per le GPU. Lo stato
diventa una matrice $\mathbf{C}_t \in \mathbb{R}^{d\times d}$ aggiornata con una
**regola di covarianza**, un prodotto esterno, esattamente come
nell'attenzione lineare:

$$
\mathbf{C}_t = f_t\, \mathbf{C}_{t-1} + i_t\, \mathbf{v}_t\, \mathbf{k}_t^\top,
\qquad
\mathbf{n}_t = f_t\, \mathbf{n}_{t-1} + i_t\, \mathbf{k}_t,
\qquad
\mathbf{h}_t = \mathbf{o}_t \odot \frac{\mathbf{C}_t\, \mathbf{q}_t}{\max\!\big(|\mathbf{n}_t^\top \mathbf{q}_t|,\, 1\big)} ,
$$

dove $\mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t$ sono query, chiave e valore, $f_t$ e $i_t$ i gate di forget e
input, e il denominatore normalizza la lettura. Senza *memory mixing*, la mLSTM è
**completamente parallelizzabile**: di fatto è un'attenzione lineare con gate e
gating esponenziale, in cui la transizione di stato è il decadimento scalare
$f_t$ moltiplicato per l'identità. È quindi la riga «Mamba-2 / RetNet» della
tabella unificante (con $f_t$ data-dipendente, come in Mamba-2), non quella di
GLA, che ha un gate per canale.

Resta un problema numerico. Un gate esponenziale $i_t = \exp(\tilde{\imath}_t)$
può esplodere. La cura è un **stabilizzatore** in scala logaritmica, uno stato

$$
m_t = \max\!\big(\log f_t + m_{t-1},\; \log i_t\big),
$$

che tiene il massimo corrente dei logaritmi dei gate e viene sottratto prima di
esponenziare: è il classico trucco *log-sum-exp*. Perché il risultato resti
davvero identico, però, va riscalato anche il fondo del denominatore, che
diventa $\max\big(|\mathbf{n}_t^\top \mathbf{q}_t|,\, e^{-m_t}\big)$: è la forma
stabilizzata che gli autori danno in appendice {cite}`beck2024xlstm`, e la
ragione è aritmetica. Sottrarre $m_t$ rimpicciolisce di $e^{-m_t}$ tanto
$\mathbf{C}_t$ quanto $\mathbf{n}_t$; se anche la soglia porta lo stesso fattore, esso si
semplifica e le due scritture coincidono per costruzione, mentre la soglia
fissa $1$ non si riscala con il resto e, appena il massimo entra in gioco,
l'uscita è un'altra. Nella sLSTM il problema non si pone, perché lì la lettura
è un rapporto puro.

`````

La mLSTM, cioè la variante con la memoria a griglia, si è rivelata la variante
di maggior peso pratico. Nel 2025 gli stessi
autori presentano **xLSTM-7B** {cite}`beck2025xlstm7b`, un modello da 7 miliardi
di parametri costruito su **sole celle mLSTM** e addestrato su 2,3 mila miliardi
di token: la prova che la formula regge alla scala dei grandi modelli linguistici
di produzione, con l'inferenza a memoria costante che l'architettura ricorrente
garantisce.

## Lo stesso scheletro

Tre architetture, tre storie (un laboratorio industriale, una comunità aperta,
il ritorno di un pioniere) e tre insiemi di scelte ingegneristiche. Eppure, se
si toglie la carrozzeria, sotto c'è sempre lo stesso telaio: **una memoria di
taglia fissa, che si aggiorna a ogni parola e non cresce mai**, riempita
mentre il modello impara guardando tutto il testo insieme e riletta, quando il
modello scrive, una parola alla volta a costo sempre uguale. Sono, insieme ai
modelli della sezione precedente, variazioni sullo stesso tema.

E il tema è quello che la tabella della sezione precedente metteva in fila: a
cambiare, da un'architettura all'altra, è **solo il modo in cui la memoria di ieri
sopravvive a oggi**. RetNet la sbiadisce con un ritmo deciso una volta per
tutte. La mLSTM di xLSTM la sbiadisce con un ritmo che ricalcola a
ogni parola, cioè sta sul gradino di Mamba-2. RWKV ha percorso tutta la scala
in due anni: dai ritmi fissi della v4 (2023) a quelli ricalcolati a ogni
parola, zona per zona, della v6 (2024), che è il gradino di GLA, fino alla v7
(2025), che prima di scrivere cancella la voce che sta per riscrivere, cioè
corregge invece di sommare alla cieca.

Che due modelli stiano sullo stesso gradino non vuol dire che siano lo stesso
modello: vuol dire che scelgono lo stesso modo di far sopravvivere la memoria,
e poi si distinguono per tutto il resto (come si aprono gli interruttori, che
cosa si mette attorno alla memoria, come si scrive il codice che gira sulla
scheda grafica). È il senso di questo capitolo: nomi, sigle e comunità diverse
raccontano, in fondo, la stessa storia.

`````{tab} Elementare

Una precisazione, perché le storie tutte-uguali sono sospette e questa ha
un'eccezione onesta. La prima versione di RWKV, la v4, non tiene una tabella
di etichette e informazioni come le altre: tiene una fila di numeri, uno per
canale, che sbiadiscono ciascuno per conto proprio. È attenzione lineare
anche quella, e la storia dello sbiadimento vale identica, ma il foglio a righe
e colonne, quello che risponde alle domande per etichetta, in RWKV arriva con
la versione 5.

`````

`````{tab} Superiore

Con i nomi tecnici: la transizione di stato è un decadimento scalare
data-indipendente in RetNet ($\gamma \mathbf{I}$), uno scalare data-dipendente con
gating esponenziale nella mLSTM di xLSTM, e in RWKV passa dal decadimento
diagonale fisso della v4 a quello data-dipendente della v6 fino alla
transizione gated-delta della v7, cioè lo stesso gradino di espressività che,
nella famiglia di Yang, separa la GLA dal Gated DeltaNet.

Una riserva sull'affermazione «tutte tengono una memoria che si scrive per
prodotto esterno»: vale da RWKV-5 in poi (oltre che per RetNet e per la
mLSTM), non per RWKV-4, il cui stato è un vettore per canale aggiornato
elemento per elemento e la cui formula, come si è visto, non contiene nessuna
query. Nella tassonomia dei paper è la differenza fra stato *piccolo* e stato
*grande*, ed è proprio il salto che compie Eagle.

`````

Questa unità apparecchia il prossimo capitolo. Gli **State Space Model** (S4,
Mamba e i loro discendenti) arrivano esattamente allo stesso posto, ma da
tutt'altra strada: non da un'attenzione da rendere economica, bensì dalla
matematica con cui si descrive un sistema che evolve nel tempo (un pendolo, un
circuito), presa nella sua forma continua e poi ridotta a passi discreti.
Vedremo che il punto d'arrivo coincide: anche un SSM è una ricorrenza lineare a
stato fisso con le sue due forme, parallela e ricorrente. E vedremo che non è
una
coincidenza: Mamba-2, con la sua *dualità* tra stato e attenzione, dimostrerà
che le due famiglie (le attenzioni lineari di questo capitolo e gli SSM del
prossimo) sono due viste della stessa cosa.

Un'ultima onestà, prima di proseguire. Nessuna di queste architetture ha
«ucciso» il Transformer, e nessuna lo farà a breve. Lo stato di dimensione
fissa, che è la loro forza in efficienza, è anche il loro limite: quando serve
ritrovare un dettaglio preciso in un contesto molto lungo, l'attenzione piena,
che conserva ogni parola, resta superiore. (Nei paper quel compito si chiama
*recall associativo esatto*, ed è il metro su cui queste architetture vengono
misurate.) È da qui che nascono gli **ibridi**, che alternano pochi strati di
attenzione a molti strati lineari. Ma questi limiti, e il modo in cui
l'ecosistema li sta affrontando, si capiscono meglio dopo aver visto anche
l'altra metà della famiglia: li riprenderemo alla fine del prossimo capitolo.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- **RetNet** {cite}`sun2023retnet` toglie alle parole la gara per l'attenzione e
  la sostituisce con un **peso che sbiadisce con la distanza**, sempre lo
  stesso: le parole si confrontano ancora, ma quanto il passato conti dipende
  solo da quanto è lontano. È la somma pesata dei voti di uno studente, con le
  interrogazioni recenti che pesano più delle
  vecchie, e si può fare nei tre modi che danno lo stesso risultato: tutto
  insieme (per addestrare in fretta), uno alla volta tenendo un totale corrente
  (per generare, a costo fisso per parola), a blocchi (per i testi
  lunghissimi).
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
- Le sue versioni successive salgono gli stessi gradini della sezione
  precedente: prima uno sbiadimento fissato una volta per tutte (v4, 2023), poi
  deciso parola per parola e zona per zona (v6, 2024 {cite}`peng2024eagle`),
  infine una versione che, prima di scrivere, corregge quello che c'è già (v7,
  2025 {cite}`peng2025rwkv7`).
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
  equivalenti**: parallela (addestramento, $O(n^2 d)$), ricorrente
  $\mathbf{S}_t = \gamma \mathbf{S}_{t-1} + \mathbf{v}_t \mathbf{k}_t^\top$
  (inferenza a costo costante nella lunghezza, $O(d^2)$ per token), chunkwise
  (contesto lungo, lineare in $n$). È multi-scala: ogni testa usa un $\gamma$
  diverso.
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
  $\mathbf{C}_t = f_t \mathbf{C}_{t-1} + i_t \mathbf{v}_t \mathbf{k}_t^\top$,
  parallelizzabile,
  di fatto una gated linear attention). **xLSTM-7B** {cite}`beck2025xlstm7b` la
  porta alla scala dei grandi modelli.
- Il filo comune: RetNet, RWKV e xLSTM (con GLA e DeltaNet) sono la stessa
  **RNN lineare a stato fisso**; cambia **solo la transizione di stato**. Gli
  State Space Model, che il libro racconta subito dopo, arrivano allo stesso
  punto da un'altra strada, e Mamba-2 dimostra che sono la stessa cosa.
```

`````

Da portarsi dietro c'è una cosa sola: tutti i modelli di questo capitolo sono
un riassunto di taglia fissa che si aggiorna parola per parola, e a
distinguerli è soltanto il modo in cui la memoria di ieri sopravvive a oggi.
Resta una pagina di verifica, poche righe che rifanno lo stesso conto in due
modi per vedere se torna lo stesso numero; poi si cambia strada, e non è un
gradino più su. **State Space Model** è l'altro ramo della stessa famiglia:
allo stesso motore ci arriva da lontano, senza passare dall'attenzione, e per
un buon tratto le due strade non sanno di somigliarsi.
