# Mixture of Experts: più parametri, stesso conto

Un'enciclopedia in trenta volumi non si legge tutta per rispondere a una
domanda: si guarda l'indice, si prende il volume giusto, e gli altri
ventinove restano sullo scaffale. Possedere una conoscenza e consultarla sono
due costi diversi, e nessuno si sognerebbe di confonderli.

Il Transformer che abbiamo montato nella sezione sull'architettura fa
esattamente il contrario. Ogni **token** (uno dei mattoncini in cui il testo
viene spezzato, di solito una parola o un pezzo di parola) viene moltiplicato
per **tutti** i numeri che la rete ha imparato, a ogni piano: quelli
dell'attenzione e quelli della rete feed-forward, dal primo piano fino in cima,
senza saltarne uno. (Quei numeri stanno in tabelle di righe e colonne, che in
matematica si chiamano *matrici*, e presi tutti insieme sono i **parametri**
del modello, quelli che si contano quando si dice «un modello da sette
miliardi».) Da qui un'aritmetica spietata: raddoppiare i parametri raddoppia il
calcolo per ogni parola, sia mentre il modello studia sia quando scrive.

La sezione sui grandi modelli linguistici ha mostrato che crescere conviene: le
leggi di scala {cite}`kaplan2020scaling` {cite}`hoffmann2022training` dicono
che ogni raddoppio di parametri e di testo fa sbagliare il modello un po’ meno,
in modo regolare e prevedibile. Ma ha mostrato anche il prezzo: la bolletta di
ogni singola parola cresce insieme al modello, e a un certo punto smette di
essere pagabile.

La domanda di questa sezione è se le due cose si possano separare. Esiste un
modello **grande in conoscenza** e **piccolo in calcolo per token**? Si può
comprare capacità senza comprare, nella stessa misura, aritmetica? La risposta
è sì, ha un nome (*mixture of experts*, miscela di esperti) ed è la ragione per cui capita di leggere due numeri di parametri per lo
stesso modello. Come
sempre, però, non è un pasto gratis: il conto non sparisce, cambia voce.

## Molti blocchi al posto di uno

Se si vuole risparmiare calcolo, conviene farlo dove il calcolo è più grosso.
Nella sezione sull'architettura abbiamo visto che ogni piano del Transformer
alterna la riunione (l'attenzione, dove le parole si scambiano informazione) e
il lavoro individuale (la **rete feed-forward**, dove ogni parola rielabora per
conto suo), e che il secondo momento, pur essendo il più semplice da capire,
contiene due terzi dei numeri imparati di un piano. Il conto stava lì: quattro
tabelle di una certa taglia nell'attenzione, contro due tabelle grandi il
quadruplo nel lavoro individuale, cioè otto della stessa taglia. Otto contro
quattro, due terzi contro un terzo. È lì che risiede gran parte di ciò che il
modello sa.

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

L'idea sta in una riga: **sostituire l'unica rete feed-forward di ogni strato**
(«strato» è il nome tecnico di quelli che nella torre della sezione
sull'architettura avevamo chiamato piani) **con $N$ copie indipendenti**, gli
*esperti*, e aggiungere davanti un piccolo
**router** che per ogni token ne sceglie $k$, di solito uno o due.
L'attenzione resta esattamente com'era. Il modello possiede i parametri di
tutti gli esperti; ogni token ne attraversa soltanto $k$.

`````{tab} Elementare

In una redazione «densa» c'è un solo redattore, bravissimo in tutto, che
rilegge e sistema ogni articolo che passa: cronaca, sport, economia, cucina.
Funziona, ma per farlo bene quel redattore deve sapere tutto, e più cose deve
sapere più tempo gli serve su ogni pezzo.

La versione a esperti ne assume otto, ciascuno con la sua specialità, e mette
all'ingresso uno smistatore che legge le prime righe e passa il pezzo a quello
che c'entra di più (o ai due che c'entrano di più, se si vuole una seconda
opinione). Un articolo sul mercato dei calciatori va allo sportivo; uno sul
restauro di un affresco va all'esperto d'arte. La redazione nel suo insieme sa
molte più cose di prima, perché sono otto teste invece di una, mentre il
lavoro su *un* articolo resta quello di sempre, perché a occuparsene è uno
solo, o due. Lo smistatore, nel conto, non pesa: è una persona sola con un
elenco di nomi.

Una parte del lavoro, però, non si moltiplica. Ogni pezzo passa comunque per
la riunione del mattino, dove tutti si dicono quello che sanno, e di riunione
ce n'è una sola: si sono moltiplicati i tavoli di rilettura, non la sala.
Diamo un prezzo alle due cose, così si vede: se la sala riunioni vale uno, un
tavolo di rilettura vale due. La redazione di prima valeva tre, un tavolo più
la sala. Quella nuova ha otto tavoli, cioè sedici, più la sala: diciassette.
Quasi sei volte, e non otto come parrebbe a contare i soli tavoli.

E un singolo pezzo quanto costa? Se lo rilegge un redattore solo, un tavolo
più la sala: tre, esattamente quanto costava prima, con otto specialisti in
casa al posto di un tuttologo. Se lo rileggono in due si arriva a cinque, una
volta e mezza abbondante; per tornare a tre si assumono redattori a mezzo
servizio, così che due di loro costino quanto il tuttologo.

Separare quanto la redazione **sa** da quanto **fatica** su ogni pezzo: la
miscela di esperti è tutta qui. Il prezzo si vede in busta paga, e non lo
sconta nessuno: gli otto lo stipendio lo prendono tutti, anche quelli che oggi
non hanno scritto una riga. La redazione costa diciassette; il pezzo, quando
lo rilegge un redattore solo, costa tre.

`````

`````{tab} Superiore

Facciamo il conto su un modello di taglia realistica, con
$d_{\text{model}} = 4096$, dimensione interna della feed-forward
$d_{\text{ff}} = 4\,d_{\text{model}} = 16384$ e $L = 32$ strati. Per ogni strato:

$$
\underbrace{2\,d_{\text{model}}\,d_{\text{ff}}}_{\text{FFN}} = 134{,}2 \text{ M},
\qquad
\underbrace{4\,d_{\text{model}}^2}_{\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V,
\mathbf{W}^O} = 67{,}1 \text{ M},
$$

dove il primo termine sono le due matrici della rete feed-forward e il secondo
le quattro proiezioni dell'attenzione. In tutto $201{,}3$ M per strato, cioè
$6{,}44$ miliardi di parametri sui 32 strati (embedding esclusi): un modello
«da 7 miliardi», nel gergo corrente. La FFN è quella classica a due matrici
della sezione sull'architettura; con una variante *gated* come SwiGLU le
matrici diventano tre, e allora dipende da cosa si tiene fisso: a $d_{\text{ff}}$
invariato il primo termine cresce di metà, mentre riducendo $d_{\text{ff}}$ a
$\tfrac{8}{3}d_{\text{model}}$, che è la pratica corrente vista nella sezione
sull'architettura, resta identico. In nessuno dei due casi cambiano i rapporti
che seguono.

Ora sostituiamo ogni FFN con $N = 8$ esperti della stessa taglia. I parametri
**totali** diventano

$$
L\,\bigl(N \cdot 2\,d_{\text{model}}\,d_{\text{ff}} + 4\,d_{\text{model}}^2\bigr)
= 32 \times (8 \times 134{,}2 + 67{,}1)\text{ M} = 36{,}5 \text{ miliardi},
$$

mentre i parametri **attivi**, quelli che un singolo token attraversa
davvero, con $k = 1$ valgono

$$
L\,\bigl(k \cdot 2\,d_{\text{model}}\,d_{\text{ff}} + 4\,d_{\text{model}}^2\bigr)
= 32 \times (134{,}2 + 67{,}1)\text{ M} = 6{,}44 \text{ miliardi},
$$

cioè **esattamente** quanto il modello denso di partenza. Quasi sei volte i
parametri ($36{,}5 / 6{,}44 \approx 5{,}7$) a parità di aritmetica per
token. Con $k = 2$ ed esperti della stessa taglia gli attivi salgono a
$10{,}7$ miliardi, $1{,}7$ volte il denso; per pareggiare del tutto si riduce
la $d_{\text{ff}}$ di ciascun esperto, così che due esperti dimezzati costino quanto
una FFN intera.

Il router, in tutto questo, è rumore di fondo: una matrice
$\mathbf{W}_g \in \mathbb{R}^{N \times d_{\text{model}}}$ per strato, cioè
$8 \times 4096 = 32\,768$ parametri, poco più di un milione sull'intero
modello, lo $0{,}003\%$ del totale.

`````

Un modello fatto così si chiama **sparso**, perché per ogni token accende solo
una piccola parte di sé; quello di prima, che accendeva tutto, si chiama
**denso**. E un modello sparso non si descrive più con un numero solo: ne
servono due, che non sono intercambiabili. I **parametri totali** dicono quanta
memoria serve per ospitarlo, i **parametri attivi** quanto costa fargli
scrivere una parola.

Un esempio con numeri veri, perché la differenza sorprende. Si parte da un
modello denso normale, di quelli «da sette miliardi»: 6,44 miliardi di
parametri, contando i piani e non il vocabolario. Se ne prende il momento di
lavoro individuale, che vale 134 milioni per piano, e lo si moltiplica per
otto: il modello arriva a **36,5 miliardi** di parametri totali. Ma ogni parola
ne attraversa uno solo degli otto, quindi i parametri **attivi** restano
**6,44 miliardi**, cioè esattamente quelli del modello di partenza. Quasi sei
volte la conoscenza, la stessa bolletta a parola. Confondere i due numeri è
l'errore più comune quando si leggono le schede tecniche di questi modelli, e
conviene prendere l'abitudine di citarli sempre in coppia.

## Come sceglie lo smistatore

Il router è il pezzo più semplice di tutta l'architettura, e conviene vederlo
per intero perché è un conto solo. Prende la lista di numeri che rappresenta
il token e ne deve ricavare $N$ punteggi, uno per esperto. Come? Si tiene in
serbo, per ciascun esperto, una lista di numeri lunga uguale, e il punteggio
di quell'esperto è il confronto fra le due liste: si moltiplicano numero per
numero e si sommano i risultati, cioè lo stesso prodotto scalare con cui
l'attenzione confronta una query e una key. Le $N$ liste, messe una sotto
l'altra, formano una tabella, e in gergo l'operazione si chiama uno *strato
lineare*.

Poi si tengono i $k$ punteggi più alti, si trasformano in proporzioni che
sommano a uno, e l'uscita dello strato è la miscela degli esperti scelti in
quelle proporzioni.

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
a uno, con il primo un po’ più pesante perché il suo punteggio era più alto.
(L'elevamento a potenza serve a due cose: non far uscire mai numeri negativi,
e allargare le differenze, così che mezzo punto di vantaggio conti davvero.)

L'uscita è la media pesata delle due risposte: il 62% di quella dell'esperto 1
più il 38% di quella dell'esperto 3. Il token ha attraversato due reti su
quattro, e le altre due sono rimaste ferme: nessun calcolo, nessun costo.

Un dettaglio che sembra un cavillo e invece conta: la softmax si applica
*dopo* il taglio, non prima. Applicata a tutti e quattro (stessa ricetta, ma
dividendo per la somma di quattro numeri invece che di due) darebbe $0{,}53$,
$0{,}12$, $0{,}32$ e $0{,}03$, e i due scelti insieme farebbero solo $0{,}85$:
buttare via il resto lascerebbe l'uscita sistematicamente più piccola del
dovuto. Rinormalizzando sui soli scelti, il cento per cento viene sempre
distribuito.

Quanto costa scegliere? Poco o niente. Un punteggio è il confronto fra due
liste di numeri, e di confronti ne servono quattro, uno per esperto; la rete
di un esperto, in un modello vero, di conti ne fa milioni. Il router può
permettersi di dare un voto a tutti proprio perché il voto costa così poco.

C'è poi una variante che, prima di tagliare, aggiunge a ogni punteggio un
pizzico di casualità: ogni tanto entra un esperto che sarebbe rimasto fuori
per un soffio. Serve a non lasciare fermi sempre gli stessi.

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
$O(k\,d_{\text{model}}\,d_{\text{ff}})$ degli esperti: con $N = 8$,
$d_{\text{model}} = 4096$ e $d_{\text{ff}} = 16384$, sono $32\,768$
moltiplicazioni contro i $134$ milioni di un solo esperto. Il meccanismo di
selezione è, in termini di calcolo, gratis.

Una variante del lavoro del 2017 merita una riga: il **noisy top-k gating**,
che somma ai punteggi un rumore gaussiano di ampiezza appresa prima di
prendere i $k$ migliori. Serve al bilanciamento del carico, cioè a dare ogni
tanto una possibilità a un esperto che il router non avrebbe scelto, ed è una
precauzione contro la tendenza del router a servirsi sempre dai soliti pochi.

`````

C'è un punto sottile e importante.
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
volta che arriverà un token simile, l'esperto 1 sarà scelto un po’ più
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

Torniamo in redazione. All'inizio gli otto redattori valgono più o meno
uguale, e lo smistatore assegna i pezzi un po’ a caso. Per puro effetto del
sorteggio il numero 7 ne riceve qualcuno in più. Scrivendo di più migliora;
migliorando, lo smistatore impara che i pezzi mandati a lui vengono bene; e
allora gliene manda ancora. Dopo un mese il 7 e altri due lavorano diciotto
ore al giorno, e cinque colleghi non hanno mai toccato un articolo. Siccome
non ne hanno mai toccato uno non hanno imparato niente, e non diventeranno
mai bravi abbastanza da meritarsene uno. Il circolo si stringe da solo: il
giornale paga otto stipendi per tre redattori, e la ragione per cui li aveva
assunti in otto è svanita.

La cura è amministrativa. A fine mese il direttore guarda un numero solo,
quanto il giornale ha sbagliato, e tutto il suo mestiere è farlo scendere. Da
adesso a quel numero si aggiunge una seconda voce, che sale quando il lavoro
è sbilanciato: oltre agli articoli scritti bene, il direttore vuole che il
lavoro giri. La seconda voce pesa un centesimo della prima, il valore con cui
è stata proposta: poco abbastanza perché al giornale convenga pagarla quando
specializzarsi rende davvero, tanto abbastanza da non poterla ignorare.

Il direttore la calcola così. Per ogni redattore segna due numeri: quanti
pezzi gli sono arrivati davvero, e quanto lo smistatore lo gradisce in media,
cioè il voto che gli dà anche nei casi in cui poi il pezzo va a un altro.
Moltiplica i due numeri, redattore per redattore, e somma gli otto prodotti.
Se ciascuno riceve un ottavo dei pezzi ed è gradito un ottavo, la somma vale
un ottavo; se uno solo prende tutti i pezzi ed è l'unico gradito, vale uno,
otto volte tanto. Poi moltiplica per il numero dei redattori, così il mese
equo vale uno tanto in una redazione da otto quanto in una da cento. Più il
lavoro si concentra, più la voce sale. Spinge, però, senza garantire: ci sono
redazioni storte in cui il conto non se ne accorge.

E la stangata arriva dove può arrivare. I pezzi consegnati o sono cinque o
sono sei, e non si ritoccano di un'inezia; il gradimento sì. Allora si taglia
il favore di ciascuno in proporzione ai pezzi che ha ricevuto: chi ne ha
presi tanti se lo vede tagliare di parecchio, chi non ne ha preso nessuno non
viene toccato, e risale da solo, perché i gradimenti sono proporzioni e
devono comunque sommare a cento.

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
**allineati**, perché l'instradamento segue l’$\arg\max$ delle stesse
probabilità che compongono $P$: gli esperti con $P_i$ alto sono di norma
quelli con $f_i$ alto. In quel regime si può **sostituire** $f_i$ con $P_i$
(una sostituzione dichiarata, non una conseguenza: è lecita solo dove
l’$\arg\max$ è netto) e il prodotto scalare si comporta come $\sum_i P_i^2$.
Su quella somma vale Cauchy-Schwarz, applicata a $P$ e al vettore di tutti
uno, che insieme al vincolo $\sum_i P_i = 1$ dà

$$
1 = \Bigl(\sum_{i=1}^{N} P_i \cdot 1\Bigr)^{\!2}
\;\le\; \Bigl(\sum_{i=1}^{N} P_i^2\Bigr)\Bigl(\sum_{i=1}^{N} 1^2\Bigr)
= N \sum_{i=1}^{N} P_i^2 ,
\qquad\text{cioè}\qquad
\sum_{i=1}^{N} P_i^2 \;\ge\; \frac{1}{N},
$$

con uguaglianza se e solo se $P$ è proporzionale al vettore di tutti uno, cioè
se e solo se il carico è uniforme. Moltiplicando per
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
$4 \times 0{,}425 = 1{,}70$, contro l’$1{,}00$ del caso uniforme.

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
superiore. Il conto su un esempio minuscolo: con $T = 8$ token e $N = 4$
esperti, in un mondo perfettamente equo ne toccherebbero due a testa; il margine
$c = 1{,}25$ li porta a $2 \times 1{,}25 = 2{,}5$, che arrotondato per eccesso
fa $3$. Un esperto che si vedesse assegnare cinque token ne elabora dunque tre
e ne lascia cadere due.

Cosa succede ai token caduti? Nulla di drammatico e nulla di visibile. Lo
strato non produce niente per quel token, e qui torna comoda la **scorciatoia**
della sezione sull'attenzione, quella che porta la lista di numeri intatta
accanto al blocco e la somma all'uscita: se l'uscita è zero, resta la
scorciatoia, e il token attraversa lo strato **immutato**, come se lì non ci
fosse. Nessun errore, nessun messaggio: solo un po’ di qualità in meno,
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

Otto stipendi, otto scrivanie. Anche i redattori fermi occupano il loro
posto, e la sala riunioni resta lì per tutti: l'ufficio deve essere quasi sei
volte quello di prima. Se non ci sta, i redattori vanno distribuiti in sedi
diverse sparse per la città, e allora lavorare su un pezzo costa poco ma
*consegnarlo* costa tanto: ogni articolo attraversa la città per arrivare al
suo specialista, e la riattraversa per andare in stampa.

Le sedi sparse per la città sono vere. Un modello di questa taglia in un
computer solo non ci sta: lo si spezza fra decine o centinaia di **schede
grafiche**, i processori specializzati nel fare tanti conti insieme, e gli
esperti finiscono su schede diverse. Il lavoro risparmiato torna allora come
**traffico**, due traversate della città per ogni piano, andata e ritorno.

I furgoni, poi, non si possono caricare la sera prima. Quanti pezzi tocchino
a ciascuna sede lo decide lo smistatore la mattina stessa, un articolo alla
volta, e finché i furgoni sono per strada le sedi stanno ferme ad aspettare.
Se il lavoro è sbilanciato, la sede affollata fa aspettare tutte le altre,
che hanno finito da un pezzo. Tenere il carico pari serve a far uscire un
buon giornale, e serve anche a non pagare venti sedi per farne lavorare tre.

Quando poi il giornale scrive, il tempo se ne va in un posto che nessuno
guarda. Le scrivanie di una sede sono migliaia e vanno velocissime, perché
sono fatte apposta per quello; ma la roba con cui si scrive sta in faldoni
giù in archivio, e dall'archivio alle scrivanie c'è un montacarichi solo. Le
penne aspettano i faldoni, e la giornata se ne va tutta in quell'attesa. E un
giornale a esperti, quando scrive, sta risparmiando proprio sulle penne.

Che il risparmio si senta dipende da quanti articoli sono in lavorazione
insieme. Uno alla volta, i due faldoni degli specialisti scelti sono gli
unici da tirare su: il montacarichi fa pochi viaggi e il vantaggio si sente
per intero. Centinaia insieme, ciascuno chiama i suoi, e su per il
montacarichi finiscono per passare quasi tutti i faldoni: la fatica torna
quella dell'archivio intero, tutti e trentasei i miliardi. Siccome è proprio
lavorando a centinaia di articoli insieme che il giornale sta in piedi, le
due cose tirano in direzioni opposte, e non c'è modo di averle tutte e due.

`````

`````{tab} Superiore

Il conto in memoria è immediato. Il modello sparso da $36{,}5$ miliardi di
parametri, in precisione a 16 bit, occupa $36{,}5 \times 2 \approx 73$ GB di
soli pesi,
contro i $12{,}9$ GB del modello denso che gli costa la stessa aritmetica per
token. Quasi sei volte la memoria per lo stesso calcolo: il baratto è
esplicito.

In addestramento distribuito la strategia naturale è l’**expert parallelism**,
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

Riassumendo in una frase: la mixture of experts sposta il collo di bottiglia dal
**calcolo** alla **memoria e alla comunicazione**. È un ottimo affare per chi
addestra su un parco di macchine collegate fra loro da cavi veloci, e per chi
deve rispondere in fretta a poche richieste per volta; è un affare molto meno
ovvio per chi deve stipare il modello in una macchina sola, o per chi ne serve
tante di richieste insieme.

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
tetto imposto a mano sulle assegnazioni) e che introduce la voce in più nella
pagella del modello, la *loss ausiliaria*, per correggerlo mentre impara.

Nel 2020 GShard {cite}`lepikhin2021gshard` (presentato a ICLR l'anno
successivo, che è la data della voce in bibliografia) porta il meccanismo dentro
il Transformer nella forma che ancora usiamo: la rete feed-forward di uno strato
ogni due sostituita da un banco di esperti, due esperti per token, il tetto alla
capacità con i token che cadono, e gli esperti sparsi su centinaia di schede che
si scambiano token in continuazione. Sette mesi dopo, nel gennaio 2021, Switch
Transformer {cite}`fedus2022switch` (uscito su rivista l'anno dopo, che è la
data in bibliografia) fa la mossa controintuitiva: **un solo esperto per
token**. Il ragionamento del 2017 diceva che ne servivano almeno due, altrimenti
lo smistatore avrebbe perso il segnale con cui impara (quella domanda «se avessi
alzato di pochissimo il punteggio, sarebbe andata meglio?», che in gergo si
chiama il **gradiente**): con un solo scelto e le proporzioni rinormalizzate su
di lui, il suo peso varrebbe sempre uno, qualunque punteggio gli sia stato
assegnato, e lo smistatore non avrebbe più modo di accorgersi di niente.

Fedus, Zoph e Shazeer risolvono la cosa togliendo proprio la rinormalizzazione:
con un esperto solo, il peso resta la proporzione calcolata su tutti e otto, un
numero minore di uno che varia con il punteggio, e il segnale arriva. Non
contraddice il «rinormalizzare sempre» di poco fa, lo circoscrive: la
rinormalizzazione serve quando gli esperti scelti sono più d'uno, per non
perdere per strada una parte dell'uscita; con uno solo quella parte perduta è
tutta la stessa uscita moltiplicata per un fattore, il che è una cosa che la
rete impara a compensare da sé, e in cambio si guadagna la manopola su cui lo
smistatore impara. In più, scegliere un solo esperto dimezza il viavai fra le
schede, semplifica il codice e permette tetti di capacità più bassi. Con il top-1 arrivano anche gli accorgimenti che rendono
stabile l'addestramento. Il più istruttivo riguarda le cifre con cui si scrivono
i numeri: per andare più in fretta il modello ne usa poche (mezza precisione,
`float16`), ma i punteggi del router si calcolano con il doppio delle cifre
(`float32`), perché quando due esperti sono quasi in pareggio è la terza cifra
a decidere chi entra, e un esperto scelto per un errore di arrotondamento è un
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
inaccettabile su scala; ma la struttura è quella, e si legge. Manca però anche
una cosa che non è un'ottimizzazione, ed è bene dirlo forte perché è la sola
che riguarda la **correttezza**: non c'è la loss di bilanciamento. Chi prendesse
questo strato e lo addestrasse così com'è otterrebbe, puntualmente, il collasso
del router. Calcolarla sarebbe questione di poche righe a
partire da `indici` e `punteggi`, che il `forward` ha già in mano.

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

Il conto, in italiano. Ogni esperto è fatto di due tabelle, una che allarga la
lista da 64 numeri a 256 e una che la ricomprime a 64, più un numero di
aggiustamento per ciascuna delle uscite: $64 \times 256 + 256$ per la prima,
$256 \times 64 + 64$ per la seconda, in tutto $33\,088$ numeri da regolare (un
*parametro* è appunto uno di quei numeri). Otto esperti ne fanno $264\,704$; lo
smistatore, che tiene una lista da 64 numeri per ciascuno degli otto, ne
aggiunge $512$; totale $265\,216$.

Ogni token, però, ne attraversa soltanto due esperti, cioè $33\,088 \times 2 =
66\,176$: **un quarto**, che è poi la frazione $k/N = 2/8$ degli esperti che
lavorano. Lo strato sa quattro volte quello che gli costa lavorare una parola.
(A voler essere pignoli il rapporto non è $4$ esatto ma $4{,}008$, perché nel
totale ci sono anche i $512$ dello smistatore, che negli attivi non li abbiamo
contati; contandoli da tutte e due le parti verrebbe $3{,}98$.)

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
  l’$\arg\max$ tiene allineati $f$ e $P$); la **capacità** limita i token per
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
