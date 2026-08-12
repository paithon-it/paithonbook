# Mixture of Experts: più parametri, stesso conto

Un'enciclopedia in trenta volumi non si legge tutta per rispondere a una
domanda: si guarda l'indice, si prende il volume giusto, e gli altri
ventinove restano sullo scaffale. Possedere una conoscenza e consultarla sono
due costi diversi, e nessuno si sognerebbe di confonderli.

Il Transformer che abbiamo montato nella sezione sull'architettura fa
esattamente il contrario. Ogni token attraversa **tutti** i pesi di tutti gli
strati: le matrici dell'attenzione, la rete feed-forward, piano dopo piano
fino in cima, senza saltarne uno. Da qui un'aritmetica spietata: raddoppiare i
parametri raddoppia il calcolo per token, in addestramento e in inferenza. La
sezione sui grandi modelli linguistici ha mostrato che crescere conviene (le
leggi di scala {cite}`kaplan2020scaling` {cite}`hoffmann2022training` sono
curve lisce, prevedibili, che premiano ogni raddoppio) ma ha mostrato anche il
prezzo: la bolletta cresce insieme al modello, e a un certo punto smette di
essere pagabile.

La domanda di questa sezione è se le due cose si possano separare. Esiste un
modello **grande in conoscenza** e **piccolo in calcolo per token**? Si può
comprare capacità senza comprare, nella stessa misura, aritmetica? La risposta
è sì, ha un nome (*mixture of experts*, miscela di esperti) ed è
l'architettura che oggi sta sotto buona parte dei modelli di frontiera. Come
sempre, però, non è un pasto gratis: il conto non sparisce, cambia voce.

## Molti blocchi al posto di uno

Se si vuole tagliare, conviene farlo dove c'è più stoffa. Nella sezione
sull'architettura abbiamo visto che il blocco Transformer alterna attenzione
(le posizioni si scambiano informazione) e **rete feed-forward** (ogni
posizione rielabora per conto suo), e che la seconda, pur essendo la parte
concettualmente più semplice, contiene circa due terzi dei parametri di ogni
strato: è lì che risiede gran parte della capacità del modello.

```{figure} ../figures/mixture-of-experts.svg
:name: fig-moe-layer
:alt: "Schema di uno strato Mixture of Experts: un token entra in un router, che fra otto esperti disponibili ne seleziona due; solo i due esperti scelti elaborano il token, e le loro uscite vengono combinate in un'unica uscita. I sei esperti non selezionati restano inattivi."
:width: 88%

Otto esperti, due al lavoro. Il modello contiene tutti i parametri degli otto,
ma per ogni singolo token ne attraversa soltanto due: da qui il divorzio fra
quanto un modello è grande e quanto costa farlo girare.
```

È il divorzio illustrato in {numref}`fig-moe-layer` a rendere interessante
tutto il resto della sezione. Finché capacità e calcolo crescono insieme,
l'unico modo di avere un modello più capace è pagarlo a ogni token; separarli
apre una terza via, e il prezzo di quella via (un router che può sbagliare, e
memoria per esperti che quasi sempre stanno fermi) è l'argomento delle pagine
che seguono.

L'idea sta in una riga: **sostituire l'unica rete feed-forward di ogni strato
con $N$ copie indipendenti**, gli *esperti*, e aggiungere davanti un piccolo
**router** che per ogni token ne sceglie $k$, di solito uno o due.
L'attenzione resta esattamente com'era. Il modello possiede i parametri di
tutti gli esperti; ogni token ne attraversa soltanto $k$.

`````{tab} Elementare

Immagina la redazione di un giornale. Nella versione «densa» c'è un solo
redattore, bravissimo in tutto, che rilegge e sistema ogni articolo che passa:
cronaca, sport, economia, cucina. Funziona, ma per farlo bene quel redattore
deve sapere tutto, e più cose deve sapere più tempo gli serve su ogni pezzo.

La versione a esperti assume trenta redattori, ciascuno con la sua
specialità, e mette all'ingresso uno smistatore che legge le prime righe e
passa il pezzo a quello che c'entra di più (o ai due che c'entrano di più, se
si vuole una seconda opinione). Un articolo sul mercato dei calciatori va allo
sportivo; uno sul restauro di un affresco va all'esperto d'arte. La redazione
nel suo insieme sa molte più cose di prima, perché sono trenta teste invece di
una; ma il lavoro su *un* articolo non è cambiato, perché a occuparsene è
sempre uno solo, o due. Il giornale è grande, il lavoro sul singolo pezzo
resta piccolo.

Tutta la mixture of experts è qui: separare quanto il modello **sa** da quanto
**fatica** a ogni parola. Con un avvertimento che vale la pena anticipare
subito, perché è il punto dove l'analogia dice la verità: i trenta redattori
lo stipendio lo prendono tutti, anche quelli che oggi non hanno scritto una
riga. La redazione costa trenta; il singolo articolo costa due.

`````

`````{tab} Superiore

Facciamo il conto su un modello di taglia realistica, con
$d_{\text{model}} = 4096$, dimensione interna della feed-forward
$d_{ff} = 4\,d_{\text{model}} = 16384$ e $L = 32$ strati. Per ogni strato:

$$
\underbrace{2\,d_{\text{model}}\,d_{ff}}_{\text{FFN}} = 134{,}2 \text{ M},
\qquad
\underbrace{4\,d_{\text{model}}^2}_{\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V,
\mathbf{W}_O} = 67{,}1 \text{ M},
$$

dove il primo termine sono le due matrici della rete feed-forward e il secondo
le quattro proiezioni dell'attenzione. In tutto $201{,}3$ M per strato, cioè
$6{,}44$ miliardi di parametri sui 32 strati (embedding esclusi): un modello
«da 7 miliardi», nel gergo corrente. La FFN è quella classica a due matrici
della sezione sull'architettura; con una variante *gated* come SwiGLU le
matrici diventano tre, e allora dipende da cosa si tiene fisso: a $d_{ff}$
invariato il primo termine cresce di metà, mentre riducendo $d_{ff}$ a
$\tfrac{8}{3}d_{\text{model}}$, che è la pratica corrente vista nella sezione
sull'architettura, resta identico. In nessuno dei due casi cambiano i rapporti
che seguono.

Ora sostituiamo ogni FFN con $N = 8$ esperti della stessa taglia. I parametri
**totali** diventano

$$
L\,\bigl(N \cdot 2\,d_{\text{model}}\,d_{ff} + 4\,d_{\text{model}}^2\bigr)
= 32 \times (8 \times 134{,}2 + 67{,}1)\text{ M} = 36{,}5 \text{ miliardi},
$$

mentre i parametri **attivi**, quelli che un singolo token attraversa
davvero, con $k = 1$ valgono

$$
L\,\bigl(k \cdot 2\,d_{\text{model}}\,d_{ff} + 4\,d_{\text{model}}^2\bigr)
= 32 \times (134{,}2 + 67{,}1)\text{ M} = 6{,}44 \text{ miliardi},
$$

cioè **esattamente** quanto il modello denso di partenza. Quasi sei volte i
parametri ($36{,}5 / 6{,}44 \approx 5{,}7$) a parità di aritmetica per
token. Con $k = 2$ ed esperti della stessa taglia gli attivi salgono a
$10{,}7$ miliardi, $1{,}7$ volte il denso; per pareggiare del tutto si riduce
la $d_{ff}$ di ciascun esperto, così che due esperti dimezzati costino quanto
una FFN intera.

Il router, in tutto questo, è rumore di fondo: una matrice
$\mathbf{W}_g \in \mathbb{R}^{N \times d_{\text{model}}}$ per strato, cioè
$8 \times 4096 = 32\,768$ parametri, poco più di un milione sull'intero
modello, lo $0{,}003\%$ del totale.

`````

Da qui in avanti un modello sparso non si descrive più con un numero solo. Ne
servono due, e non sono intercambiabili: i **parametri totali** dicono quanta
memoria serve per ospitarlo, i **parametri attivi** quanto costa fargli
scrivere un token. Confonderli è l'errore più comune quando si leggono le
schede tecniche di questi modelli, e conviene prendere l'abitudine di citarli
sempre in coppia.

## Il router, in formule

Il router è il pezzo più semplice di tutta l'architettura. Prende la lista di
numeri che rappresenta il token e ne ricava $N$ punteggi, uno per esperto, con
la moltiplicazione più elementare che ci sia fra una tabella di numeri e una
lista (in gergo: un singolo strato lineare, senza nemmeno il termine costante
che di solito si aggiunge). Poi si tengono i $k$ punteggi più alti, si
trasformano in proporzioni che sommano a uno, e l'uscita dello strato è la
miscela degli esperti scelti in quelle proporzioni.

`````{tab} Elementare

Vediamolo con quattro esperti e numeri veri. Arriva un token; il router lo
guarda ed emette quattro punteggi:

| esperto | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| punteggio | $2{,}0$ | $0{,}5$ | $1{,}5$ | $-1{,}0$ |

Con $k = 2$ si tengono i due migliori, l'esperto 1 e l'esperto 3, e gli altri
due si buttano via: per questo token semplicemente non esistono. Restano da
decidere le proporzioni della miscela, cioè da trasformare due punteggi
($2{,}0$ e $1{,}5$) in due percentuali che sommino a cento. La ricetta standard
si chiama **softmax**, ed è una divisione con un passaggio in più: si prende il
numero $e = 2{,}718\ldots$, lo si eleva a ciascun punteggio, e si divide
ciascun risultato per la somma di tutti. Sui nostri due, $e^{2{,}0} = 7{,}39$ e
$e^{1{,}5} = 4{,}48$, che sommati fanno $11{,}87$; quindi
$7{,}39 / 11{,}87 = 0{,}62$ e $4{,}48 / 11{,}87 = 0{,}38$. Due pesi che sommano
a uno, con il primo un po' più pesante perché il suo punteggio era più alto.
(L'elevamento a potenza serve a due cose: non far uscire mai numeri negativi,
e allargare le differenze, così che mezzo punto di vantaggio conti davvero.)

L'uscita è la media pesata delle due risposte: il 62% di quella dell'esperto 1
più il 38% di quella dell'esperto 3. Il token ha attraversato due reti su
quattro, e le altre due sono rimaste ferme: nessun calcolo, nessun costo.

Un dettaglio che sembra un cavillo e invece conta: la softmax si applica
**dopo** il taglio, non prima. Applicata a tutti e quattro (stessa ricetta, ma
dividendo per la somma di quattro numeri invece che di due) darebbe $0{,}53$,
$0{,}12$, $0{,}32$ e $0{,}03$, e i due scelti insieme farebbero solo $0{,}85$:
buttare via il resto lascerebbe l'uscita sistematicamente più piccola del
dovuto. Rinormalizzando sui soli scelti, il cento per cento viene sempre
distribuito.

`````

`````{tab} Superiore

Sia $\mathbf{x} \in \mathbb{R}^{d_{\text{model}}}$ la rappresentazione di un
token in ingresso allo strato,
$\mathbf{W}_g \in \mathbb{R}^{N \times d_{\text{model}}}$
la matrice del router e $E_1, \dots, E_N$ gli esperti (ciascuno una FFN
indipendente). I pesi di miscelazione sono

$$
G(\mathbf{x}) =
\operatorname{softmax}\bigl(\text{top-}k(\mathbf{W}_g\,\mathbf{x})\bigr),
$$

dove $\text{top-}k(\cdot)$ conserva le $k$ componenti maggiori e pone le altre
a $-\infty$ (così la softmax le manda a zero esatto e normalizza sui soli
sopravvissuti: è la formulazione di Shazeer e colleghi
{cite}`shazeer2017outrageously`). L'uscita dello strato è

$$
\mathbf{y} = \sum_{i \,\in\, \text{top-}k} G(\mathbf{x})_i \, E_i(\mathbf{x}),
$$

dove $G(\mathbf{x})_i$ è il peso assegnato all'esperto $i$ (i pesi dei
selezionati sommano a 1) ed $E_i(\mathbf{x})$ la sua risposta. Con i punteggi
$\mathbf{W}_g \mathbf{x} = (2{,}0;\ 0{,}5;\ 1{,}5;\ -1{,}0)$ e $k = 2$
sopravvivono gli
indici 1 e 3, con

$$
G(\mathbf{x})_1 = \frac{e^{2{,}0}}{e^{2{,}0} + e^{1{,}5}} = 0{,}622,
\qquad
G(\mathbf{x})_3 = \frac{e^{1{,}5}}{e^{2{,}0} + e^{1{,}5}} = 0{,}378 .
$$

Il costo del router è $O(N\,d_{\text{model}})$ per token, contro
$O(k\,d_{\text{model}}\,d_{ff})$ degli esperti: con i valori della sezione
precedente, $32\,768$ moltiplicazioni contro i $134$ milioni di un solo
esperto. Il meccanismo di selezione è, in termini di calcolo, gratis.

Una variante del lavoro del 2017 merita una riga: il **noisy top-k gating**,
che somma ai punteggi un rumore gaussiano di ampiezza appresa prima di
prendere i $k$ migliori. Serve al bilanciamento del carico, cioè a dare ogni
tanto una possibilità a un esperto che il router non avrebbe scelto: la
sezione sul collasso, qui sotto, spiega perché sia una precauzione necessaria.

`````

C'è un punto sottile e importante, e vale per entrambi i livelli di lettura.
La scelta dei $k$ esperti è **discreta**: si ordina, si taglia, e un taglio non
ha vie di mezzo, perché un esperto è dentro o è fuori, mai «dentro per il tre
per cento in più». Il modo in cui una rete impara, invece, è tutto fatto di vie
di mezzo: si guarda com'è andata e ci si chiede, per ogni numero interno, «se
lo avessi alzato di pochissimo, sarebbe andata meglio o peggio?», poi lo si
sposta di un'inezia nella direzione buona. Su un taglio quella domanda non ha
risposta: alzare di pochissimo un punteggio, quasi sempre, non cambia chi entra
e chi resta fuori. Come fa allora il router a imparare a smistare?

La risposta è che impara dall'altro pezzo, i **pesi della miscela**, i due
numeri calcolati poco fa. Quelli sì che rispondono a variazioni piccole, e
dipendono direttamente dai punteggi del router. Se l'esperto 1 ha dato una
risposta utile, conviene alzare il suo peso; per alzare il suo peso bisogna
alzare il punteggio che il router gli aveva assegnato; e allora la prossima
volta che arriverà un token simile, l'esperto 1 sarà scelto un po' più
volentieri. Il router impara di riflesso, guardando com'è andata a chi ha
mandato, senza mai essere corretto direttamente su chi avrebbe dovuto
scegliere.

E gli esperti *non* scelti? Non ricevono nulla. Nessuna correzione, nessun
modo di dimostrare che avrebbero fatto meglio: chi non lavora non sbaglia, e
chi non sbaglia non impara. Questa asimmetria è comodissima (è il motivo per
cui il calcolo si risparmia davvero) ed è anche la radice del guasto
caratteristico di tutta la famiglia.

## Il collasso del router

Un modello a esperti, lasciato a sé stesso, tende a **collassare**: dopo
qualche migliaio di passi il router manda quasi tutti i token agli stessi due
o tre esperti, e gli altri restano dei blocchi di parametri inerti. Non è un
bug di implementazione, è la dinamica naturale del sistema.

`````{tab} Elementare

Torniamo in redazione. All'inizio i trenta redattori valgono più o meno
uguale, e lo smistatore assegna i pezzi un po' a caso. Per puro effetto del
sorteggio, però, il redattore numero 7 ne riceve qualcuno in più. Scrivendo di
più migliora; migliorando, lo smistatore impara che mandare i pezzi a lui dà
buoni risultati; e allora gliene manda ancora di più. Dopo un mese il 7 e altri
due lavorano diciotto ore al giorno, ventisette colleghi non hanno mai toccato
un articolo, e siccome non ne hanno mai toccato uno non hanno imparato niente:
non potranno mai diventare bravi abbastanza da meritarsene uno.

È un circolo che si rinforza da solo, e finisce nel modo peggiore: il giornale
paga trenta stipendi per tre redattori. Nel modello è identico: si è pagata
memoria per parametri che non imparano nulla, e il vantaggio dell'architettura
svanisce.

La cura non è furba, è amministrativa: si aggiunge alla pagella del modello una
voce che **punisce lo sbilanciamento**. Oltre a chiedergli di predire bene la
parola successiva, gli si chiede di distribuire il lavoro; e siccome è una
penalità piccola, il modello la paga volentieri quando specializzarsi conviene
davvero, ma non può ignorarla del tutto.

`````

`````{tab} Superiore

La contromisura standard è una **loss ausiliaria di bilanciamento**, sommata
alla cross-entropia con un coefficiente piccolo. Nella forma di Switch
Transformer {cite}`fedus2022switch`, per un batch $\mathcal{B}$ di $T$ token e
$N$ esperti:

$$
\mathcal{L}_{\text{aux}} = \alpha\,N \sum_{i=1}^{N} f_i \, P_i,
\qquad
f_i = \frac{1}{T}\sum_{\mathbf{x} \in \mathcal{B}}
\mathbb{1}\{\arg\max_j p_j(\mathbf{x}) = i\},
\qquad
P_i = \frac{1}{T}\sum_{\mathbf{x} \in \mathcal{B}} p_i(\mathbf{x}),
$$

dove $p(\mathbf{x})$ è la distribuzione softmax del router sul token
$\mathbf{x}$, $f_i$ è la
**frazione di token effettivamente instradati** all'esperto $i$ (un
conteggio), $P_i$ la **probabilità media** che il router gli ha assegnato (una
quantità continua) e $\alpha$ il peso della penalità, $10^{-2}$ nel paper.

Perché quel prodotto spinge verso il carico uniforme? Entrambi i vettori $f$ e
$P$ stanno sul simplesso ($\sum_i f_i = \sum_i P_i = 1$) e tendono a essere
**allineati**, perché l'instradamento segue l'$\arg\max$ delle stesse
probabilità che compongono $P$: gli esperti con $P_i$ alto sono di norma
quelli con $f_i$ alto. In quel regime il prodotto scalare si comporta come
$\sum_i P_i^2$, e per la disuguaglianza di Cauchy-Schwarz

$$
\sum_{i=1}^{N} P_i^2 \;\ge\; \frac{1}{N},
$$

con uguaglianza se e solo se la distribuzione è uniforme. Moltiplicando per
$N$ si ottiene un termine che vale $1$ sul carico uniforme e cresce man mano
che il carico si concentra. È un argomento euristico, non un teorema: con
punteggi quasi in pareggio l'allineamento fra $f$ e $P$ si allenta, ed
esistono configurazioni non uniformi in cui il termine scende sotto $1$.
Fedus e colleghi, del resto, presentano la loss come un *incentivo* al
bilanciamento, non come una garanzia. Un esempio con $N = 4$ e $T = 8$
token, con cinque token al primo esperto, due al secondo, uno al terzo e
nessuno al quarto:
$f = (0{,}625;\ 0{,}25;\ 0{,}125;\ 0)$ e
$P = (0{,}55;\ 0{,}25;\ 0{,}15;\ 0{,}05)$ danno
$4 \times 0{,}425 = 1{,}70$, contro l'$1{,}00$ del caso uniforme.

Il dettaglio elegante è **dove passa il gradiente**. Il conteggio $f_i$ non è
differenziabile (è la stessa selezione discreta di prima), quindi la derivata
scorre solo attraverso $P_i$:

$$
\frac{\partial \mathcal{L}_{\text{aux}}}{\partial P_i} = \alpha\,N\,f_i ,
$$

cioè una spinta verso il basso **proporzionale al carico già ricevuto**. Gli
esperti affollati si vedono abbassare i punteggi in proporzione a quanto sono
affollati; quelli vuoti non ricevono alcuna spinta negativa e risalgono per
differenza. A instradamento fissato la penalità è lineare in $P$, e in quel
regime è ben condizionata: il gradiente non dipende da dove ci si trova sul
simplesso. È una linearità locale, però, non globale: $f$ dipende dagli stessi
parametri del router, e quando l'instradamento cambia cambia anche il
coefficiente della penalità, il che riporta il paesaggio della loss ausiliaria
fra le cose che si osservano, non fra quelle che si dimostrano.

`````

### La capacità, e i token che cadono

Il bilanciamento è una spinta statistica, non una garanzia: in un mucchietto di
token qualunque (nel gergo un **batch**, cioè il gruppo di esempi che il
modello elabora in una volta sola) un esperto può comunque ricevere più token
di quanti ne possa elaborare. Per questo l'implementazione fissa in anticipo
una **capacità**, cioè il numero massimo di token che ciascun esperto accetta
per batch. La ricetta è semplice: si conta quanti token toccherebbero a testa
in un mondo perfettamente equo, si aggiunge un margine di sicurezza, e si
arrotonda per eccesso.

$$
\text{capacità} = \left\lceil \frac{T}{N} \cdot c \right\rceil,
$$

dove $T$ è il numero di token del batch, $N$ il numero di esperti, $c$ il
*capacity factor*, cioè il margine, appena sopra 1 (valori tipici tra $1{,}0$ e
$1{,}25$), e le parentesi con gli angoli sono l'arrotondamento all'intero
superiore. Con $T = 8$ token, $N = 4$ esperti e $c = 1{,}25$ la capacità è $3$
(due token a testa più un quarto, cioè $2{,}5$, arrotondato a $3$): un esperto
che si vedesse assegnare cinque token ne elabora tre e ne lascia cadere due.

Cosa succede ai token caduti? Nulla di drammatico e nulla di visibile: l'uscita
dello strato MoE per quel token è zero, e siccome il blocco è avvolto dalla
connessione residua incontrata con le ResNet nel capitolo sul deep learning
($\mathbf{x} + \text{MoE}(\mathbf{x})$), il token attraversa lo strato
**immutato**, come se lì
non ci fosse. Nessun errore, nessun messaggio: solo un po' di qualità in meno,
distribuita in modo silenzioso. Alzare $c$ riduce i token caduti ma alloca
buffer più grandi, cioè spreca memoria per posti mai occupati. E c'è una
conseguenza più insidiosa: **il destino di un token dipende dagli altri token
del batch**, quindi lo stesso identico input, in compagnia diversa, può
ricevere un trattamento diverso. Un modello sparso non è una funzione del
singolo esempio, e chi ne debugga il comportamento farebbe bene a saperlo.

## Il conto si sposta, non sparisce

Fin qui la parte lieta. Adesso quella onesta, che è anche il motivo per cui la
mixture of experts non è la fine della storia: **si risparmia calcolo, non
memoria**. Gli esperti che nessun token attraversa non consumano aritmetica,
ma devono comunque esistere da qualche parte, caricati e pronti, perché il
token successivo potrebbe chiederli.

`````{tab} Elementare

I trenta redattori dello stipendio lo prendono tutti, e a tutti serve una
scrivania. Se stanno in un unico ufficio grande, l'ufficio deve essere trenta
volte più grande. E se non ci sta, bisogna distribuirli in edifici diversi
sparsi per la città: a quel punto lavorare su un pezzo costa poco, ma
*consegnarlo* costa tanto, perché ogni articolo deve attraversare la città per
arrivare al suo specialista e poi tornare indietro per andare in stampa.

È esattamente ciò che succede alle schede grafiche. Il calcolo risparmiato si
ritrova, in buona parte, come **traffico**: i token viaggiano verso la scheda
che ospita il loro esperto e tornano indietro, due volte per ogni strato. E
c'è un effetto collaterale del collasso di cui sopra: se il carico è
sbilanciato, la scheda affollata fa aspettare tutte le altre, che restano
ferme a guardare. Il bilanciamento, insomma, non serve solo alla qualità del
modello: serve a non pagare venti schede per farne lavorare tre.

In inferenza il problema è ancora più netto. Generare testo è un lavoro
limitato dalla **memoria**, non dal calcolo: il collo di bottiglia è leggere i
pesi a ogni parola, non moltiplicarli (l'abbiamo intravisto parlando della KV
cache, e il capitolo sull'MLOps ci tornerà sopra con i numeri). Un modello con
pochi parametri attivi ma tantissimi totali attacca il lato sbagliato del
problema, e resta pesante da servire.

`````

`````{tab} Superiore

Il conto in memoria è immediato. Il modello sparso della prima sezione, in
precisione a 16 bit, occupa $36{,}5 \times 2 \approx 73$ GB di soli pesi,
contro i $12{,}9$ GB del modello denso che gli costa la stessa aritmetica per
token. Quasi sei volte la memoria per lo stesso calcolo: il baratto è
esplicito.

In addestramento distribuito la strategia naturale è l'**expert parallelism**,
già nominato nella sezione sul parallelismo distribuito accanto agli assi
dati, tensor e pipeline: gli esperti di ciascuno strato si spartiscono fra le
schede, una manciata per GPU. Il pattern di comunicazione che ne nasce non è
l'all-reduce del parallelismo dati, ma un **all-to-all**: ogni GPU spedisce a
ogni altra i token destinati agli esperti che quella ospita, e ne riceve
indietro le uscite. Due all-to-all per strato MoE, andata e ritorno. Due
proprietà lo rendono scomodo. Primo, il volume dipende dalla distribuzione del
routing, che cambia a ogni batch: è traffico irregolare, difficile da
sovrapporre al calcolo come si fa con l'all-reduce di
`DistributedDataParallel`. Secondo, è una barriera implicita: la GPU con
l'esperto più affollato detta il ritmo a tutte le altre. Questa, e non solo la
qualità del modello, è la ragione economica della loss di bilanciamento.

In inferenza vale il quadro che la sezione su LLMOps riprenderà in dettaglio:
la generazione è **memory-bound**, e il tempo per token è dominato dalla
lettura dei pesi dalla memoria della GPU, non dall'aritmetica. La
sparsità qui aiuta in modo condizionato, e la condizione è il **batch**. Con
poche sequenze in volo si leggono davvero solo i $k$ esperti selezionati, e la
latenza per token è quella del modello piccolo: un vantaggio reale. Ma appena
il batch cresce, token diversi scelgono esperti diversi, e con qualche
centinaio di token per strato praticamente ogni esperto viene richiesto da
qualcuno: la lettura torna quasi completa, e il modello si comporta, in banda,
come i suoi $73$ GB. La sparsità del calcolo non si traduce automaticamente in
sparsità del traffico di memoria; e siccome servire in batch grandi è
esattamente ciò che rende sostenibile un LLM, i due obiettivi tirano in
direzioni opposte.

`````

Riassumendo in una frase: la mixture of experts sposta il collo di bottiglia
dal **calcolo** alla **memoria e alla comunicazione**. È un ottimo affare per
chi addestra su cluster con interconnessioni veloci e per chi serve poche
sequenze a bassa latenza; è un affare meno ovvio per chi deve stipare il
modello in una macchina sola.

## Da un'idea del 1991

L'idea era pronta trent'anni prima dell'hardware che l'ha resa utile.
Nel 1991 Robert Jacobs, Michael Jordan, Steven Nowlan e Geoffrey Hinton
pubblicano su *Neural Computation* un articolo intitolato *Adaptive Mixtures
of Local Experts* {cite}`jacobs1991adaptive`. La proposta è già tutta lì: più
reti separate, e una **gating network** (letteralmente «rete cancello»: è il
nonno del router) che impara a pesarle a seconda di quello che le arriva
davanti, così che ciascuna si specializzi su un tipo di dati invece di fare un
compromesso mediocre su tutto. Manca però il pezzo che ci interessa qui: la
miscela era **densa**, cioè si facevano lavorare tutti gli esperti e poi si
faceva la media delle loro risposte. Un buon modo di organizzare
l'apprendimento, non un modo di risparmiare conto.

Il salto è del 2017, con *Outrageously Large Neural Networks: The
Sparsely-Gated Mixture-of-Experts Layer* di Noam Shazeer e colleghi
{cite}`shazeer2017outrageously`, lo stesso anno di *Attention Is All You Need*
{cite}`vaswani2017attention` e con un autore in comune. Qui il gating diventa
**sparso**: si calcolano solo i $k$ esperti scelti, e la miscela smette di
essere un modo di combinare modelli per diventare un modo di comprare
parametri senza comprare aritmetica. Lo strato viene infilato fra strati LSTM
(l'articolo esce a gennaio, i Transformer arriveranno cinque mesi dopo) e
arriva a contenere fino a 137 miliardi di parametri, due ordini di grandezza
sopra i modelli linguistici densi dell'epoca. È anche il lavoro che mette a
fuoco lo squilibrio di carico e il collasso del router (osservato prima da
Eigen, Ranzato e Sutskever {cite}`eigen2013learning`, che lo curavano con un
tetto imposto a mano sulle assegnazioni) e che introduce le loss ausiliarie
per correggerlo durante l'addestramento.

Nel 2020 GShard {cite}`lepikhin2021gshard` (presentato a ICLR l'anno
successivo, che è la data della voce in bibliografia) porta il meccanismo dentro
il Transformer nella forma che ancora usiamo: la rete feed-forward di uno strato
ogni due sostituita da un banco di esperti, due esperti per token, il tetto alla
capacità con i token che cadono, e gli esperti sparsi su centinaia di schede che
si scambiano token in continuazione. Pochi mesi dopo Switch Transformer
{cite}`fedus2022switch` fa la mossa controintuitiva: **un solo esperto per
token**. Il ragionamento del 2017
diceva che ne servivano almeno due per avere un gradiente sensato sul router;
Fedus, Zoph e Shazeer mostrano che il gradiente arriva comunque, e che il
top-1 dimezza il traffico dell'all-to-all, semplifica il codice e permette
capacità più piccole. Con il top-1 arrivano anche gli accorgimenti che rendono
stabile l'addestramento, come calcolare il router in `float32` mentre il resto
del modello sta in mezza precisione: la softmax su punteggi vicini è sensibile
all'arrotondamento, e un esperto scelto per un errore di terza cifra è un
esperto sbagliato.

Da allora l'architettura sparsa è di uso comune nei modelli di frontiera, ed è
il motivo per cui capita di leggere due numeri di parametri per lo stesso
modello. Restano limiti documentati, che è giusto nominare: i modelli sparsi
sono più delicati da rifinire su compiti piccoli (tendono a sovradattarsi, e
servono accorgimenti come un dropout più aggressivo dentro gli esperti), e
l'ipotesi implicita che gli esperti si specializzino in qualcosa di
interpretabile (grammatica, argomento, lingua) è nella pratica molto meno vera
di quanto il nome suggerisca. «Esperti» è una metafora comoda, non una
descrizione verificata.

## In pratica: uno strato MoE in PyTorch

Il codice qui sotto implementa lo strato per intero: esperti, router, top-$k$,
softmax sui soli scelti, combinazione pesata. Non c'è nessuna delle
ottimizzazioni vere (il *dispatch* efficiente dei token, l'all-to-all, la
capacità con i buffer preallocati), e il ciclo `for` sugli esperti sarebbe
inaccettabile su scala; ma la struttura è quella, e si legge. Chi legge al
livello Elementare può saltare da qui alla fine della sezione: il conto che
chiude la pagina è raccontato senza formule nel riquadro finale.

```python
import torch
from torch import nn


class StratoMoE(nn.Module):
    """Uno strato Mixture of Experts: N esperti FFN e un router top-k."""

    def __init__(self, d_model=64, d_ff=256, n_esperti=8, k=2):
        super().__init__()
        self.k = k
        self.esperti = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model),
            )
            for _ in range(n_esperti)
        ])
        self.router = nn.Linear(d_model, n_esperti, bias=False)  # la matrice W_g

    def forward(self, x):                       # x: [batch, seq, d_model]
        forma = x.shape
        x = x.reshape(-1, forma[-1])            # i token diventano una lista piatta
        punteggi = self.router(x)               # [T, N]: un punteggio per esperto
        valori, indici = torch.topk(punteggi, self.k, dim=-1)   # i k migliori
        pesi = torch.softmax(valori, dim=-1)    # softmax SOLO sui selezionati

        y = torch.zeros_like(x)
        for i, esperto in enumerate(self.esperti):
            # quali token hanno scelto l'esperto i, e in quale delle k posizioni
            token, posto = (indici == i).nonzero(as_tuple=True)
            if token.numel() == 0:
                continue                        # esperto inutilizzato in questo batch
            contributo = esperto(x[token])      # solo i suoi token, non tutti
            y[token] = y[token] + pesi[token, posto].unsqueeze(-1) * contributo
        return y.reshape(forma)
```

Due righe meritano attenzione. La softmax si applica a `valori`, cioè ai soli
punteggi sopravvissuti al `topk`: è la rinormalizzazione discussa sopra. E
l'esperto viene chiamato su `x[token]`, un sottoinsieme delle righe: è qui che
il calcolo si risparmia davvero, perché se nessuno lo ha scelto non viene
eseguito affatto.

Un controllo dei conti, con gli iperparametri di default:

```python
strato = StratoMoE(d_model=64, d_ff=256, n_esperti=8, k=2)

x = torch.randn(2, 5, 64)          # 2 frasi da 5 token
print(strato(x).shape)             # torch.Size([2, 5, 64]): la forma non cambia

totali = sum(p.numel() for p in strato.parameters())
per_esperto = sum(p.numel() for p in strato.esperti[0].parameters())
print(totali, per_esperto * strato.k)
# 265216 66176  -> totali contro attivi per token: circa 4 volte tanto
```

Un esperto pesa $64 \times 256 + 256 + 256 \times 64 + 64 = 33\,088$
parametri (un *parametro* è uno dei numeri che la rete regola durante
l'addestramento: sono quelli che si contano quando si dice «un modello da sette
miliardi»); otto ne fanno $264\,704$, più i $512$ del router: $265\,216$ in
tutto. Ogni token ne attraversa due, cioè $66\,176$: esattamente un quarto dei
parametri degli esperti, il rapporto $N/k = 8/2$ (sul totale il rapporto è
$4{,}008$, perché il router lo pagano tutti i token). Quattro volte i
parametri, lo stesso conto.

Lo strato è intercambiabile con la FFN di un blocco Transformer: stessa forma
in ingresso, stessa forma in uscita. Ed è precisamente questa
intercambiabilità la ragione per cui la mixture of experts si è diffusa così
in fretta: non è un'architettura nuova, è un pezzo di ricambio.

Nulla di tutto questo, però, cambia *cosa* il modello ha imparato a fare.
Denso o sparso, quello che esce dal pre-addestramento resta un completatore di
testo, e per trasformarlo in un interlocutore serve la fase successiva, il
**post-training**, di cui parla la sezione che segue.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- La miscela di esperti prende il momento di **lavoro individuale** di ogni
  strato (quello dove sta la maggior parte di ciò che il modello ha imparato:
  due terzi buoni) e lo moltiplica: molti blocchi in parallelo, gli esperti,
  più uno **smistatore** che per ogni parola ne sceglie uno o due. Un modello
  così non si racconta con un numero solo: uno dice quanto **sa**, cioè quanta
  memoria occupa; l'altro quanto **fatica** su ogni parola, cioè quanto costa
  farlo scrivere.
- Lo smistatore dà un voto a ciascun esperto, tiene i migliori e mescola le
  loro risposte in proporzione ai voti (il $62\%$ di uno, il $38\%$
  dell'altro). La scelta in sé è un taglio netto, e da un taglio non si impara
  nulla: lo smistatore migliora guardando **com'è andata a chi ha mandato il
  pezzo**, cioè attraverso le proporzioni della miscela.
- Lasciato a sé, lo smistatore **collassa**: manda tutto ai soliti due o tre,
  che lavorando migliorano ancora, mentre gli altri non toccano un articolo e
  non impareranno mai. La cura è amministrativa: una voce in più nella pagella
  del modello che punisce lo sbilanciamento (un incentivo, non una garanzia).
  C'è poi un tetto ai pezzi che un esperto accetta per turno: quelli in
  eccesso attraversano lo strato **senza essere lavorati**, in silenzio.
- Si risparmia **fatica**, non **spazio**: i redattori fermi prendono lo
  stipendio e occupano una scrivania lo stesso. Quando stanno in edifici
  diversi il costo si sposta sul viavai, perché ogni articolo attraversa la
  città per arrivare al suo specialista e poi torna indietro. E quando il
  modello scrive, il tempo se ne va più ad andare a prendere quello che sa che
  a fare i conti: uno che sa moltissimo e fatica poco su ogni parola attacca
  il lato sbagliato del problema, e resta pesante da far girare.
- L'idea è del 1991 {cite}`jacobs1991adaptive`, ma allora ogni pezzo passava
  per tutti gli esperti e delle loro risposte si faceva la media: un buon modo
  di organizzare il lavoro, non di risparmiarlo. Il salto è del 2017
  {cite}`shazeer2017outrageously`, quando si calcolano davvero solo gli
  esperti scelti; poi Switch Transformer {cite}`fedus2022switch` mostra che
  **uno solo per parola** basta e semplifica tutto.
- «Esperti» è una metafora comoda: quello in cui ciascuno si specializza è
  raramente riconoscibile, e questi modelli sono più delicati da rifinire su
  compiti piccoli.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- La mixture of experts sostituisce la **rete feed-forward** di uno strato con
  $N$ esperti paralleli più un **router** che per ogni token ne sceglie $k$
  (uno o due). Un modello sparso si descrive con **due** numeri, non uno:
  **parametri totali** (la memoria) e **parametri attivi** (il calcolo per
  token).
- Il router è uno strato lineare:
  $G(\mathbf{x}) = \operatorname{softmax}(\text{top-}k(\mathbf{W}_g \mathbf{x}))$
  e $\mathbf{y} = \sum_{i \in \text{top-}k} G(\mathbf{x})_i E_i(\mathbf{x})$.
  La selezione è discreta e non differenziabile: il gradiente arriva al router
  **attraverso i pesi** $G(\mathbf{x})_i$ degli esperti scelti.
- Senza contromisure il router **collassa** su pochi esperti, in un circolo
  che si rinforza da solo. La cura è una **loss ausiliaria**
  $\alpha N \sum_i f_i P_i$ {cite}`fedus2022switch`, che resta bassa quando il
  lavoro è distribuito in parti uguali e cresce quando si concentra su pochi
  esperti (un incentivo, non una garanzia: l'argomento regge finché
  l'$\arg\max$ tiene allineati $f$ e $P$); la **capacità** limita i token per
  esperto e quelli in eccesso attraversano lo strato immutati grazie alla
  connessione residua.
- Si risparmia **calcolo**, non **memoria**: tutti gli esperti devono
  risiedere da qualche parte. In addestramento il costo si sposta sulla
  comunicazione (**expert parallelism**, all-to-all); in inferenza resta il
  limite di banda della generazione autoregressiva, tanto più stringente
  quanto più grande è il batch.
- La linea storica va dalla miscela **densa** del 1991
  {cite}`jacobs1991adaptive` allo strato **sparso** del 2017
  {cite}`shazeer2017outrageously`, fino a Switch Transformer
  {cite}`fedus2022switch`, che mostra come **un solo esperto per token** basti
  e semplifichi tutto.
- «Esperti» è una metafora comoda: la specializzazione che emerge è raramente
  interpretabile, e i modelli sparsi sono più delicati da rifinire su compiti
  piccoli.
```
`````
