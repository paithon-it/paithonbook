# Modelli trasparenti e importanza delle feature

A metà degli anni Novanta, addestrando un modello per stimare il rischio di
morte dei pazienti ricoverati per polmonite, un gruppo di ricercatori di
Pittsburgh scoprì che l'algoritmo aveva imparato una regola sorprendente: *chi
soffre d'asma ha un rischio più basso*. Preso alla lettera, un consiglio
pericoloso: gli asmatici sono pazienti fragili. La spiegazione era clinica, e
sta tutta in quello che i medici facevano con loro. Negli ospedali un asmatico
con la polmonite veniva mandato subito in terapia intensiva, proprio perché
considerato a rischio, e quelle cure aggressive gli abbassavano la mortalità
sotto quella di tutti gli altri. L'asma, di suo, non proteggeva un bel niente.
A proteggere era la corsia in cui l'asma ti faceva finire. Il modello aveva colto una correlazione vera nei dati e ne aveva
tratto una conclusione che, usata per decidere chi mandare a casa, avrebbe
ucciso. La storia (raccontata anni dopo da Rich Caruana) è diventata il
manifesto di un campo: se non possiamo *guardare dentro* un modello, non
sappiamo su quali scorciatoie si regge, e non possiamo fidarcene quando la
posta è alta.

Ci sono due strade per capire un modello. La prima è sceglierlo **trasparente
per costruzione**, così semplice che la sua logica si legge a occhio nudo. La
seconda è tenere il modello com'è, anche se dentro ha milioni di numeri e non
si legge affatto, e interrogarlo da fuori: gli si passano dei casi, si guardano
le risposte, e si deduce il resto. Un modello trattato così si dice una
**scatola nera**, perché non se ne vede l'interno; uno che si legge, per
contrasto, una scatola bianca.

Questa sezione percorre la prima strada per intero, e poi imbocca la seconda
con il primo attrezzo che vi si incontra: una classifica delle colonne dei dati,
ordinate per quanto pesano sulle risposte. Le colonne di una tabella di dati si
chiamano **feature**, e quella classifica si chiama quindi **importanza delle
feature**. Per un panorama sistematico dell'intero campo il riferimento è il
manuale di Molnar {cite}`molnar2022interpretable`.

## Modelli trasparenti per costruzione

Alcuni modelli non hanno bisogno di essere spiegati: *sono* la loro
spiegazione. L'esempio più puro è quello che risponde facendo una somma:
prende ogni colonna, la moltiplica per un numero suo, e somma tutto. Sono i due
modelli incontrati nel capitolo sul machine learning con i nomi di **regressione
lineare** (quando la risposta è una quantità, un prezzo) e **regressione
logistica** (quando è un sì o un no). Quei numeri, uno per colonna, si chiamano **pesi** (o, con la parola che
si usa più spesso in statistica, **coefficienti**: sono la stessa cosa), e una
somma fatta così si dice **pesata**. Il punto è che quei pesi *sono* la storia
che il modello racconta: non c'è altro da sapere.

```{figure} ../figures/regressione-lineare.svg
:name: fig-retta-residui
:alt: "Una nube di punti attraversata da una retta. Ogni punto è un esempio: sull'asse orizzontale la caratteristica misurata, per esempio i metri quadri di una casa, sull'asse verticale la quantità da prevedere, il prezzo. Da ciascun punto scende o sale un segmento verticale fino alla retta, che misura di quanto il modello ha sbagliato su quell'esempio."
:width: 84%

La retta e ciò che le sfugge. Ogni punto è un esempio (una casa, con i suoi
metri quadri e il suo prezzo) e il segmento verticale è di quanto il modello
sbaglia proprio su quello: si chiama **residuo**. La retta scelta è quella che
li rende complessivamente più piccoli, e li lascia tutti in bella vista.
```

C'è una qualità di {numref}`fig-retta-residui` che un modello con milioni di
numeri dentro non ha, ed è il motivo di questa sezione: la regola che ha
prodotto quella retta si legge per intero, ed è una riga sola di somma. I
residui, invece, si misurano per qualunque modello, perché basta confrontare la
risposta con la verità. Quello che con un modello opaco non si può fare è
aprire la regola e vedere quale pezzo del conto ha prodotto proprio quella
risposta. Trasparente non vuol dire accurato: vuol dire che non c'è niente da
scoprire dopo.

`````{tab} Elementare

Prendiamo un modello che stima il prezzo di una casa come somma di
contributi: tanti euro per ogni metro quadro, tanti per ogni stanza, un bonus
o un malus per il quartiere. Ogni peso è un'etichetta col prezzo appesa a una
caratteristica, «$+2\,000$ € al metro quadro», e si legge senza sforzo. Per
capire perché il modello ha risposto $210\,000$ € non serve nessuno strumento
esterno: basta leggere la ricevuta, voce per voce.

| voce | quanto | peso | contributo |
|---|---|---|---|
| metri quadri | 90 | $+2\,000$ €/m² | $+180\,000$ € |
| stanze | 3 | $+8\,000$ € a stanza | $+24\,000$ € |
| quartiere | centro | $+6\,000$ € | $+6\,000$ € |
| **totale** | | | **$210\,000$ €** |

Ecco: quella tabella *è* il modello. Non c'è un altro posto in cui guardare, e
se il prezzo non ci convince sappiamo esattamente con quale riga prendercela.

Vale lo stesso per la regressione logistica, che al posto di una quantità dà
una probabilità: non «sì» o «no» secchi, ma «questo cliente restituirà il
prestito con probabilità del 65%», e poi sta a chi la usa decidere sopra quale
soglia il sì è sì. Anche lì i pesi si leggono uno per uno: un peso positivo
spinge verso il sì, uno negativo verso il no, e più è grande più spinge. Un modello così si può stampare su
mezza pagina e discutere con chi non ha mai visto una formula. È questo che
intendiamo per *trasparente*: la regola di decisione è alla luce del sole.

`````

`````{tab} Superiore

In un modello lineare $\hat{y} = \mathbf{w}^\top \mathbf{x} + b$ ogni
coefficiente $w_j$ è
l'effetto marginale della feature $j$: a parità di tutte le altre, un aumento
unitario di $x_j$ sposta la predizione di esattamente $w_j$. Nella regressione
logistica $\hat{y} = \sigma(\mathbf{w}^\top \mathbf{x} + b)$
l'interpretazione passa alle *log-odds*: $w_j$ è la variazione del logaritmo
del rapporto di probabilità
$\log\frac{p}{1-p}$ per un incremento unitario di $x_j$, cosicché $e^{w_j}$ è
il fattore moltiplicativo sull’*odds ratio*.

Due avvertenze rendono onesta questa lettura. Primo, i coefficienti sono
confrontabili tra loro solo se le feature sono **standardizzate** (stessa
scala): un $w_j$ grande può riflettere semplicemente un'unità di misura
piccola. Secondo, l'inciso «a parità di tutte le altre» è fragile quando le
feature sono **correlate**: se due colonne si muovono insieme, il modello può
spartire il loro effetto in modo arbitrario, e i singoli coefficienti
diventano instabili (la stessa multicollinearità che rende preziosa la
regolarizzazione Ridge/Lasso vista nel capitolo di machine learning).

`````

La trasparenza non finisce con i modelli lineari. Gli **alberi di decisione**,
studiati nel capitolo sul machine learning, sono l'altro archetipo di «scatola
bianca»: si parte dalla domanda in cima (che si chiama **radice**, perché
l'albero si disegna capovolto, con le foglie in basso) e a ogni risposta si
scende di un ramo, fino a una casella finale che porta la decisione (una
**foglia**). Quel percorso *è* la spiegazione.

```{figure} ../figures/alberi-di-decisione.svg
:name: fig-albero-percorso
:alt: "Un albero di decisione con la radice in alto. Ogni nodo porta una domanda su una singola caratteristica con una soglia: alla radice «reddito maggiore di 30 mila?», e sotto «età maggiore di 40?» e «rate in corso?». Dalla radice partono due rami, etichettati «sì» e «no», e scendendo si arriva a una delle quattro foglie colorate, che portano la decisione: approva, verifica, rifiuta, verifica."
:width: 90%

La spiegazione è il percorso. Per sapere perché un esempio ha ricevuto quella
risposta si parte dalla radice e si segue, a ogni nodo, il ramo che le sue
risposte scelgono: le domande incontrate scendendo sono poche, e si leggono una
per una.
```

{numref}`fig-albero-percorso` mostra una forma di trasparenza diversa da
quella dei modelli lineari, e per certi versi più forte. Un modello lineare
spiega con dei pesi, che valgono per tutti gli esempi insieme; un albero
spiega *questo* esempio con una catena di condizioni verificabili una per una.

Sempre fra i modelli trasparenti, e sempre dalla parte della somma anziché da
quella delle domande sì/no, stanno i **modelli additivi generalizzati**, che
estendono la regressione lineare sostituendo a ogni peso una curva. Il nome
dice il meccanismo: **additivi** perché la risposta resta una somma di
contributi, uno per colonna, che non si mescolano fra loro; **generalizzati**
perché lo stesso impianto va bene sia quando la risposta è una quantità sia
quando è una probabilità. Si citano quasi sempre con la sigla inglese, **GAM**.
Come si costruiscono, e che cosa costa la loro ipotesi quando è falsa, lo
racconta la sezione sulle spline del capitolo di machine learning; qui interessa
l'altra metà, cioè perché si lasciano leggere.

`````{tab} Elementare

Nel modello lineare ogni caratteristica porta un cartellino fisso: «$+2\,000$ €
al metro quadro», sempre, dal primo metro all'ultimo, come nella ricevuta di
poco fa. Un GAM ammette che il
prezzo del metro quadro cambi lungo la scala: i primi cinquanta metri valgono
molto, i successivi meno, e oltre una certa soglia quasi niente. Al posto di un
numero c'è quindi una **curva** per ogni caratteristica, che si può guardare e
discutere («ecco come cambia il rischio al variare dell'età»). La trasparenza
resta intatta, perché le curve non si mescolano: si legge una caratteristica
alla volta, come le voci di una ricevuta.

`````

`````{tab} Superiore

Un GAM scrive

$$
g\big(\mathbb{E}[y \mid \mathbf{x}]\big) = \theta_0 + \sum_j f_j(x_j),
$$

dove ogni $f_j$ è una funzione liscia stimata dai dati (spline, smoother) e $g$
è la **funzione di legame** ereditata dai modelli lineari generalizzati:
l'identità in regressione, il logit in classificazione. È $g$ il
«generalizzato» del nome, ed è ciò che rende il modello utilizzabile fuori dal
caso di una risposta continua: senza di essa la somma additiva vivrebbe su
tutta la retta reale anche quando la quantità da prevedere è una probabilità.
Nel caso logit ogni $f_j$ si legge come contributo alle *log-odds*, esattamente
come il $w_j$ della regressione logistica, ma variabile con $x_j$ invece che
costante {cite}`hastie1986generalized`. L'additività è ciò che conserva la
leggibilità: nessun termine di interazione, quindi ogni curva si può guardare
da sola.

`````

E ci sono i **sistemi a regole**, elenchi di condizioni del tipo «SE il reddito
è sotto 20 000 E il contratto è a termine ALLORA nega il prestito», che decidono
in un modo che si può leggere riga per riga.

Aleggia però un pregiudizio diffuso: che la trasparenza si paghi in
accuratezza, che per essere bravi si debba per forza essere oscuri. È vero solo
in parte.

`````{tab} Elementare

Della sostanza si è già detto nell'apertura del capitolo, sui fiori: su tanti
problemi a righe e colonne un modello trasparente ben costruito arriva
vicinissimo, a volte alla pari, con la scatola nera, mentre su immagini, testo e
suoni le reti profonde vincono senza rivali.

Quello che qui vale la pena aggiungere è il consiglio pratico che ne segue, ed è
di buon senso: parti dal modello trasparente e **misura** quanto perdi davvero
passando a uno più complicato, invece di darlo per scontato. Se la differenza è
minima, la chiarezza è un guadagno netto, e lo è soprattutto dove una decisione
sbagliata ha un costo umano.

`````

`````{tab} Superiore

Il presunto compromesso accuratezza/interpretabilità è stato messo in
discussione, in particolare da Cynthia Rudin {cite}`rudin2019stop`, che
sostiene come su dati
**strutturati** con feature dotate di senso il divario tra un modello
interpretabile ben ingegnerizzato e una scatola nera sia spesso trascurabile o
nullo. La ragione è che il vantaggio del *deep learning* si manifesta soprattutto
là dove serve **apprendere le rappresentazioni** da dati grezzi ad alta
dimensione (pixel, forme d'onda, token); sui dati tabellari le feature sono già
significative, e modelli come gradient boosting o GAM catturano quasi tutta la
struttura utile restando ispezionabili.

Ne discende una gerarchia metodologica: preferire un modello intrinsecamente
interpretabile quando le prestazioni sono comparabili, e riservare gli
strumenti *post-hoc* (importanza delle feature, PDP, e i metodi locali che
vedremo più avanti nel capitolo) ai casi in cui la scatola nera è davvero
necessaria. Gli strumenti post-hoc, va detto subito, spiegano il modello
*dall'esterno* e sono approssimazioni: non sostituiscono la trasparenza di
progetto.

`````

## L'importanza delle feature: quali colonne contano

Passiamo agli strumenti che interrogano un modello già addestrato, quale che
sia. La prima domanda, la più naturale, è: **su quali colonne si regge?**
Vogliamo cioè una classifica delle feature, ordinate per quanto contano nelle
risposte.

Prima di costruirla, conviene togliere di mezzo un equivoco. Chi misura quanto
contano le colonne, di solito, lo fa per poi **buttarne via qualcuna**: si
misura, si tira una riga, e le colonne che restano sotto si eliminano dai dati.
Quel secondo passo si chiama **selezione delle feature**, viene subito dopo il
primo e per questo lo si confonde con lui, ma è un'altra cosa.

```{figure} ../figures/feature-selection.svg
:name: fig-feature-selection
:alt: "A sinistra un grafico a barre con il punteggio di otto feature, una barra per feature, e una riga orizzontale tratteggiata che fa da soglia: tre barre la superano, le altre cinque restano sotto. A destra restano solo le tre colonne che hanno superato la soglia, disegnate come tre rettangoli affiancati."
:width: 100%

I due passi affiancati. A sinistra si misura: una barra per colonna, e il
punteggio scritto sotto è uno dei tanti possibili. A destra si è deciso, e sono
rimaste tre colonne su otto.
```

La differenza fra i due passi è di natura, non di ordine. Una classifica è un
fatto misurabile: si misura, e viene quel che viene. La riga tratteggiata invece
non la dice nessun dato, la decide una persona, e va giustificata con qualcosa
d'altro: il costo di raccogliere una colonna, un vincolo di leggibilità, una
prova che il modello ridotto non peggiora. La selezione delle feature la
nominiamo una volta sola, qui: quello di cui parla la sezione è la classifica.

Cominciamo dal modo più generale e più solido di costruirla. È un metodo che non
guarda dentro il modello: lo tratta da scatola nera, gli passa dei casi e si
tiene solo le risposte, quindi funziona con qualunque cosa.

### L'importanza per rimescolamento

L'ha proposto Leo Breiman nel 2001, insieme alle foreste casuali, ed è di una
semplicità che quasi offende.

`````{tab} Elementare

L'idea è quasi impertinente: se una colonna conta davvero, allora
**rovinarla** deve far crollare le risposte giuste. Prendiamo un modello che
prevede se un cliente restituirà un prestito, e mettiamolo alla prova su 100
clienti mai visti: indovina 90 volte su 100. Ora prendiamo una colonna sola
(il reddito) e ne **rimescoliamo** i valori tra i 100 clienti: ognuno si
ritrova il reddito di qualcun altro. Tutto il resto è intatto, ma quella colonna
adesso contiene numeri che con la persona non c'entrano più niente: è diventata
**rumore**, che è il modo in cui si chiamano dei dati che non portano
informazione. Riproviamo il modello: ora indovina solo 72 volte.
Ha perso 18 punti *solo* perché gli abbiamo scombinato il reddito: segno che
ci si appoggiava molto. L'importanza del reddito è quel calo,
$90\% - 72\% = 18$ punti.

Rifacciamo lo stesso gioco con una colonna che non c'entra nulla, il colore
preferito: rimescolandola, il modello continua a indovinare 90 volte. Calo
zero, importanza zero. Poiché il rimescolamento è casuale, lo si ripete
qualche volta e si fa la media, per non farsi ingannare da un mescolamento
fortunato. Il bello è che questo trucco funziona con *qualsiasi* modello,
perché tutto quello che serve è potergli fare delle domande e sentire le
risposte.

Rimescolare i valori di una colonna, in matematica, si dice **permutarli**: da
qui il nome con cui il metodo si trova nelle librerie, *permutation
importance*.

`````

`````{tab} Superiore

Formalizziamo. Sia $f$ il modello addestrato e
$e_{\text{orig}} = \mathcal{L}(f, \mathcal{D})$ il suo errore (o l'opposto di uno
*score*: MSE in regressione, $1-\text{acc}$ in classificazione) su un insieme
di valutazione $\mathcal{D} = (\mathbf{X}, y)$. Per la feature $j$ si costruisce
$\mathbf{X}_{\pi_j}$,
copia di $\mathbf{X}$ in cui i valori della **sola colonna $j$** sono permutati
casualmente lungo le righe (rompendo il legame tra $x_j$ e $y$ ma
preservandone la distribuzione marginale) e si misura
$e_{\pi_j} = \mathcal{L}(f, (\mathbf{X}_{\pi_j}, y))$. L'importanza è il
peggioramento

$$
\mathrm{FI}_j = \frac{1}{K}\sum_{k=1}^{K} e_{\pi_j}^{(k)} - e_{\text{orig}},
$$

media su $K$ permutazioni indipendenti (in `scikit-learn`, `n_repeats`), che
fornisce anche una deviazione standard. Introdotta da Breiman con le foreste
casuali {cite}`breiman2001random` e in seguito formalizzata da Fisher, Rudin e
Dominici {cite}`fisher2019models` come *model reliance* (nella loro variante
il rapporto
$e_{\pi_j}/e_{\text{orig}}$ anziché la differenza) è **model-agnostic**:
richiede solo il forward del modello e un insieme etichettato.

Due accortezze. La misura va calcolata su dati **held-out**: sul *training* essa
racconta quanto il modello si è appoggiato a $x_j$ per memorizzare, non quanto
quella feature aiuti a generalizzare. E le feature **correlate** portano due
guai distinti, che conviene non confondere. Il primo: il modello recupera
l'informazione dalla colonna gemella non permutata, e l'importanza, spartita
fra le due, risulta *sottostimata*. Il secondo: la permutazione crea
combinazioni irrealistiche (un'altezza da adulto con un peso da bambino) su
cui il modello viene interrogato fuori dal supporto dei dati, e l'errore così
gonfiato può *sovrastimare* l'importanza delle feature coinvolte
{cite}`hooker2021unrestricted`: la stessa patologia di estrapolazione che
ritroveremo nel PDP. I due guasti non si possono correggere insieme, e la
ragione è quella vista in apertura di capitolo: servono due domande diverse.
Il primo va evitato da chi chiede «di che cosa ha bisogno *questo modello*»,
il secondo da chi chiede «quanta informazione porta *questa colonna*».

`````

### Importanza da impurità (e la sua distorsione)

C'è un secondo modo di fare la classifica, e viene gratis con gli alberi. Per
capirlo bisogna sapere come un albero sceglie le sue domande.

Un albero decide dove tagliare guardando quanto un taglio *ordina* le risposte.
Prima del taglio un gruppo di esempi tiene dentro risposte mescolate; il taglio
lo divide in due gruppi, e il taglio buono è quello che rende i due gruppi il
più possibile omogenei. Quanto un gruppo è mescolato si chiama **impurità**, e
si misura con formule dai nomi tecnici (l'indice di Gini, l'entropia) che non
cambiano l'idea: massima quando le risposte dentro il gruppo sono di tutti i
tipi, zero quando sono tutte uguali. Ogni taglio (in inglese **split**) fa
scendere l'impurità di un tanto, e quel tanto è il merito che si accredita alla
colonna su cui il taglio è stato fatto. Il taglio, si badi, è una domanda con un
numero dentro: «il reddito supera i 30 000?». Quel numero si chiama **soglia**,
e per una colonna con tanti valori diversi le soglie fra cui scegliere sono
tantissime.

L'albero, dunque, mentre impara tiene già il conto di questi meriti. Basta
sommarli, e la classifica è fatta senza fare nient'altro. Lo stesso vale per una **foresta casuale**, i cui alberi sono già stati
incontrati in apertura di capitolo: sono centinaia, e ciascuno cresce su un
campione diverso delle righe, estratto a sorte, e a seconda delle impostazioni
anche su un sottoinsieme diverso delle colonne. Da lì il «casuale». Le loro
risposte si mettono ai voti, e i meriti si sommano su tutti gli alberi. Questa misura
si chiama, con la sigla inglese che si trova ovunque, **MDI** (*mean decrease
in impurity*, cioè calo medio dell'impurità), ed è quella che nella sezione
sugli alberi e gli insiemi di modelli del capitolo sul machine learning si
leggeva da `feature_importances_`. È rapidissima, perché non c'è niente da
calcolare dopo, ma va letta con prudenza, per due ragioni che vale la pena
rendere esplicite.

`````{tab} Elementare

L'importanza da impurità premia le feature che l'albero *usa spesso* per
tagliare. Il problema è che una feature con tanti valori diversi (un'età
precisa al giorno, un importo in centesimi) offre all'albero un'enorme
quantità di soglie tra cui scegliere, e con così tante possibilità ne trova
quasi sempre una che, per puro caso, separa un po’ i dati. Così accumula
«meriti» anche quando non porta vera informazione. Una feature con pochi
valori (sì/no, tre categorie) parte invece svantaggiata: ha poche soglie da
provare.

Il risultato è che l'importanza da impurità tende a **gonfiare** le feature
continue o con molte categorie e a **sminuire** quelle a pochi valori: un
difetto strutturale, non del singolo insieme di dati. Lo vedremo con i nostri
occhi più avanti in questa stessa pagina, dando in pasto al modello due colonne
di puro rumore,
una con tanti valori e una con due soli: valgono zero tutte e due, e questa
misura ne premia una sette volte più dell'altra.

E c'è una seconda ragione, indipendente dalla prima, che conviene tenere a
mente perché fra poco servirà. Questi meriti l'albero se li accredita **mentre
impara**, cioè sugli stessi esempi da cui sta imparando. Ma su quegli esempi un
taglio sembra sempre utile, anche quando ha soltanto imparato a memoria una
particolarità di quei dati che non si ripeterà altrove (si dice che il modello
**sovradatta**). Il merito resta accreditato lo stesso. Il rimescolamento, che
si può misurare su esempi che il modello non ha mai visto, di questo problema
non soffre: ed è la ragione per cui, dovendo scegliere, ci si fida di quello.

`````

`````{tab} Superiore

Il bias della MDI è verso le feature ad **alta cardinalità** e quelle
**continue**, ed è stato stabilito da Strobl, Boulesteix, Zeileis e Hothorn
{cite}`strobl2007bias`, che ne identificano **due** sorgenti distinte.

La prima è combinatoria: il numero di split candidati cresce con
il numero di valori distinti, e massimizzare la riduzione d'impurità su molti
tagli equivale a un test statistico con molte comparazioni (una feature
puramente casuale ma continua ottiene, in aspettazione, un guadagno positivo
per sovradattamento locale). La seconda sta nel **campionamento bootstrap con
reimmissione**, che è il default di `RandomForestRegressor`: pescare con
ripetizione induce fra le variabili associazioni che nella popolazione non ci
sono, e l'effetto è tanto più marcato quanti più valori la variabile ha. A
queste si aggiunge il fatto, indipendente dai due, che la stessa documentazione
di `scikit-learn` ricorda: `feature_importances_` è calcolata sul *training
set*, quindi ogni colonna su cui gli alberi hanno tagliato accumula merito
anche quando quel taglio era sovradattamento.

Rispetto alla permutation importance, la MDI ha due svantaggi: è legata alla
struttura interna del modello (vale solo per gli alberi) ed è misurata sui dati
di addestramento. La permutazione, calcolata su un *hold-out*, è model-agnostic
e riflette la generalizzazione; è la stima che la sezione sugli alberi e gli
ensemble già raccomandava di preferire. Con una precisazione che il lavoro di
Strobl impone:
la permutazione **non è immune per natura** al secondo meccanismo, e la loro
soluzione completa prevede alberi a selezione non distorta *più* subsampling
senza reimmissione. Quello che mette al riparo la stima raccomandata qui è che
`sklearn.inspection.permutation_importance` si calcola su un hold-out
indipendente, non OOB sui campioni bootstrap: è la circostanza che toglie di
mezzo il meccanismo, non una proprietà della permutazione in sé. Entrambe,
comunque, restano misure di importanza **globale**: dicono quanto una feature
conta *in media su tutto il dataset*, non per la singola predizione.

`````

## Come agisce una feature, non solo quanto

Sapere *quanto* una feature conta non dice *come* agisce. Il prezzo sale o
scende con i metri quadri? Ogni metro in più vale quanto il precedente, o dopo
i primi cento non conta più niente? La prima domanda è sul segno, la seconda
sulla forma, e per rispondere serve disegnare una **curva**: sull'asse
orizzontale i valori della colonna, su quello verticale la risposta del
modello. I tre attrezzi che seguono disegnano quella curva in tre modi diversi,
e si citano tutti e tre con la sigla inglese.

`````{tab} Elementare

Il primo si chiama **PDP** (*Partial Dependence Plot*) ed è come chiedere a
tutta la popolazione: «e se aveste tutti quarant'anni?». In pratica si prende
l'elenco dei clienti, si riscrive a tutti la stessa età, quaranta, lasciando
invariato tutto il resto, si chiedono al modello le risposte e se ne fa la
media. Poi si rifà con 41 anni, con 42, e così via. Unendo i punti viene fuori
una curva, ed è l'effetto medio dell'età.

C'è un limite: la media può nascondere storie opposte. Se l'età fa salire la
risposta per metà dei clienti e scendere per l'altra metà, la curva media
resta piatta e ti fa credere che l'età non conti. Il rimedio è la curva
**ICE** (*Individual Conditional Expectation*): invece della sola media,
disegni *una curva per ogni cliente*. Un fascio di curve che vanno in direzioni
diverse rivela subito che l'effetto non è uguale per tutti.

C'è un secondo limite, più insidioso, e per quello esiste un attrezzo diverso.
Riscrivere a tutti la stessa età va bene finché l'età non è legata ad altro; ma
se due colonne vanno sempre insieme (l'altezza e il peso, per dire) riscriverne
una sola fabbrica persone che non esistono, alte due metri e pesanti cinquanta
chili. Al modello quelle persone non le ha mai viste nessuno, quindi risponde a
caso, e la curva che ne esce è la media di un mucchio di risposte a caso.

Il rimedio si chiama **ALE** (*Accumulated Local Effects*, effetti locali
accumulati), e il nome dice il metodo. Non si chiede più niente a tutta la
popolazione: si divide la colonna in fascette sottili (i quarantenni, i
quarantunenni, e così via) e dentro ciascuna fascetta si lavora **solo con chi
in quella fascetta ci sta davvero**. A quelle persone, e solo a quelle, si
chiede il modello due volte: una con l'età portata all'estremo basso della
fascetta e una all'estremo alto. La differenza fra le due risposte, mediata su
di loro, è quanto conta un anno in più *per chi ha quell'età lì*, ed è uno
scalino. Nessuno viene inventato, perché a un quarantenne stiamo chiedendo di
avere quarantun anni, non cinquanta.

Poi quegli scalini si sommano uno dopo l'altro, dal primo all'ultimo (ecco gli
«accumulati»): il primo parte da zero, il secondo si appoggia sul primo, e la
scaletta che viene fuori è la curva. Si preferisce al PDP proprio quando le
colonne si muovono insieme.

`````

`````{tab} Superiore

Per la feature $j$, la **partial dependence** è l'attesa della predizione
marginalizzando sulle altre feature $\mathbf{X}_{-j}$, stimata sul dataset come

$$
\mathrm{PD}_j(v) = \frac{1}{m}\sum_{i=1}^{m} f\!\big(v,\, \mathbf{X}_{-j}^{(i)}\big),
$$

dove si fissa $x_j = v$ e si mediano le predizioni su tutti gli esempi
{cite}`friedman2001greedy`. La curva **ICE** è la stessa quantità *prima* di
mediare: $f(v, \mathbf{X}_{-j}^{(i)})$ per il singolo esempio $i$
{cite}`goldstein2015peeking`. Il PDP
è dunque la media verticale del fascio di ICE; quando le curve ICE si
sventagliano, un effetto medio piatto maschera **interazioni** o eterogeneità.

Il difetto profondo del PDP è l’**estrapolazione con feature correlate**:
fissare $x_j = v$ mentre si tengono i valori reali di $\mathbf{X}_{-j}$ genera punti
$(v, \mathbf{X}_{-j}^{(i)})$ implausibili (altezza 2 m con peso 50 kg) su cui il
modello viene interrogato fuori dal supporto dei dati, producendo curve
fuorvianti. L’**Accumulated Local Effects** (ALE) di Apley e Zhu
{cite}`apley2020visualizing`
corregge il tiro: invece di marginalizzare su tutta la distribuzione, media le
*differenze* di predizione entro piccoli intervalli di $x_j$, usando la
distribuzione **condizionata** e restando così nelle regioni densamente
popolate. È la scelta da preferire quando le feature sono marcatamente
correlate. Vale la pena notare che la scelta fra i due non è fra un metodo
giusto e uno sbagliato, ma è di nuovo la forcella dell'apertura: il PDP
marginale risponde a «che cosa farebbe *questo modello* se gli riscrivessi una
colonna», l'ALE condizionato a «come si comporta la predizione lungo i dati che
esistono davvero».

`````

## In pratica: rimescolamento contro impurità

Torniamo alle due classifiche, quelle di due sezioni fa, e mettiamole a
confronto su dati veri. Le curve appena viste rispondevano a «come agisce una
colonna»; adesso si torna alla domanda di prima, «quanto conta», e si guarda
quale dei due modi di misurarla è affidabile. Ne useremo una raccolta che si
studia da decenni, distribuita insieme alla libreria `scikit-learn` (lo
strumentario di machine learning che il libro usa da sempre) e che si chiama
`diabetes`. È una tabella di 442 righe, una per paziente diabetico, e dieci
colonne di misure cliniche: l'età, il sesso, l'indice di massa corporea
(`bmi`), la pressione (`bp`) e sei valori del sangue, chiamati da `s1` a `s6`.
La cosa da prevedere, in ogni riga, è quanto la malattia sarà progredita dopo un
anno; la colonna da prevedere si chiama, in gergo, il **target**, ed è l'unica
che il modello non riceve in ingresso.

I 442 pazienti li dividiamo in due mucchi, come si fa sempre: circa il 70% (309
righe) serve al modello per imparare, e su quelle diremo che il modello si
**addestra**; il restante 30% (133 righe) resta da parte, e il modello lo vedrà
solo alla fine, per essere messo alla prova su casi che non ha mai incontrato.
Il primo mucchio si chiama insieme di addestramento, il secondo insieme di
prova, o *test*. La distinzione qui non è un dettaglio: è metà della morale di
questa pagina.

E poi un accorgimento, che è il vero esperimento: aggiungiamo alla tabella
**due colonne inventate**, riempite di numeri tirati a sorte e senza alcun
rapporto con la malattia. Una continua (numeri con la virgola, tutti diversi
fra loro), una binaria (soltanto 0 o 1). Sappiamo per costruzione che non
valgono niente, tutte e due allo stesso modo, e proprio per questo servono:
sono il metro con cui leggere ciò che le due misure diranno. Su questa tabella a
dodici colonne facciamo crescere una foresta casuale, e poi chiediamo a
entrambe le tecniche quali colonne contano.

```python
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

dati = load_diabetes()
X, y, nomi = dati.data, dati.target, list(dati.feature_names)

# Due colonne di puro rumore, scorrelate dal target: una continua e una binaria.
# Non valgono niente ne l'una ne l'altra: servono da metro per le due misure.
rng = np.random.default_rng(0)
X = np.column_stack([X, rng.normal(size=len(y)), rng.integers(0, 2, size=len(y))])
nomi += ["rumore_cont", "rumore_bin"]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

rf = RandomForestRegressor(n_estimators=300, random_state=0)
rf.fit(X_tr, y_tr)
print("R^2 sul test:", round(rf.score(X_te, y_te), 3))  # -> 0.315

# Importanza da permutazione, misurata sul TEST (10 mescolamenti per feature)
pi = permutation_importance(rf, X_te, y_te, n_repeats=10, random_state=0)

print("feature      (impurita)   perm-import")
for i in np.argsort(rf.feature_importances_)[::-1]:   # dalla piu alta per la MDI
    print(f"{nomi[i]:>11}   {rf.feature_importances_[i]:.3f}       "
          f"{pi.importances_mean[i]:+.3f} +/- {pi.importances_std[i]:.3f}")
```

L'output ordina le dodici colonne per importanza da impurità e affianca quella
per rimescolamento:

```text
R^2 sul test: 0.315
feature      (impurita)   perm-import
        bmi   0.297       +0.179 +/- 0.023
         s5   0.296       +0.188 +/- 0.065
         bp   0.091       +0.035 +/- 0.012
         s3   0.058       -0.015 +/- 0.015
         s6   0.051       -0.001 +/- 0.008
rumore_cont   0.047       +0.004 +/- 0.012
         s2   0.042       +0.002 +/- 0.009
        age   0.041       +0.005 +/- 0.011
         s1   0.037       +0.002 +/- 0.007
         s4   0.025       -0.009 +/- 0.008
        sex   0.007       +0.004 +/- 0.002
 rumore_bin   0.006       -0.002 +/- 0.002
```

Prima di leggere la classifica, i tre numeri della stampa, uno alla volta.

Il **primo** dice quanto è bravo il modello, ed è costruito su una scala con due
paletti. Da una parte c'è chi risponde sempre la media, senza nemmeno guardare
il paziente: quello prende **zero**. Dall'altra c'è chi indovina la progressione
esatta di ogni paziente: quello prende **uno**. (E si può anche andare sotto
zero, facendo peggio di chi risponde sempre la media.) Il nostro modello prende
$0{,}315$, cioè sta a poco meno di un terzo del cammino fra il pigro e
l'indovino. Quella misura si chiama $R^2$, e il numero va tenuto a mente per
tutta la pagina: l'importanza che stiamo per leggere descrive *questo* modello,
che non è bravissimo, non la verità clinica.

Il **secondo**, la colonna dell'impurità, è il merito accumulato dai tagli. È
distribuito su tutte le colonne come una torta: i dodici numeri sommano a 1, e
infatti si leggono come frazioni del merito totale. (Sommandoli a mano da questa
stampa viene $0{,}998$: è colpa dei tre decimali a cui la stampa arrotonda, non
del conto.)

Il **terzo**, la colonna del rimescolamento, è il calo di quel primo numero,
l’$R^2$, quando la colonna viene rimescolata. Le due colonne di numeri non sono
quindi nella stessa unità di misura: la prima è una fetta di torta, la seconda
è un danno misurato in $R^2$ perduto. Il «$\pm$» accanto dice quanto quel danno
balla fra un rimescolamento e l'altro dei dieci provati (nella stampa, dove i
simboli matematici non si possono scrivere, quel «più o meno» compare come
`+/-`).

Fatta la lettura, le due misure concordano sull'essenziale: `bmi` e `s5` (un
valore del sangue legato ai grassi che vi circolano) dominano, `bp` le segue, il
resto conta poco. Ma emergono anche le differenze attese, e le due colonne inventate le
rendono misurabili.

**La prima differenza è il segno.** Diverse colonne hanno un'importanza da
rimescolamento lievemente **negativa** (`s3`, `s4`, `s6`, e il rumore binario):
rimescolarle *migliora* di un soffio le risposte. Non è un paradosso, è il caso:
sono numeri dell'ordine del centesimo, cioè dello stesso ordine del ballerio fra
un rimescolamento e l'altro che il «$\pm$» accanto dichiara, e il modo giusto di
leggerli è «quella colonna non serviva». La
misura da impurità questo non lo può dire, perché non scende **mai sotto zero**:
un taglio o abbassa l'impurità o non viene scelto, quindi accredita sempre
merito positivo, e nel suo linguaggio la frase «questa colonna non serve»
letteralmente non esiste.

**La seconda differenza è la distorsione, e adesso si vede.** Il `rumore_cont`
prende un'impurità di $0{,}047$: più di `s2`, di `age`, di `s1` e di `s4`, che
sono indicatori clinici veri. Il `rumore_bin`, altrettanto inutile, prende
$0{,}006$. Fra due colonne che valgono entrambe esattamente zero c'è quindi un
fattore **sette**: la stampa arrotonda a tre decimali, e a occhio la divisione
darebbe quasi otto, ma sui valori pieni, $0{,}0467$ e $0{,}0065$, il rapporto è
$7{,}2$. L'unica differenza fra le due colonne è quanti valori distinti
contengono nelle 309 righe su cui gli alberi sono cresciuti: 309 la prima, cioè
un valore diverso per ogni riga; due la seconda. È la distorsione verso le
colonne con tanti valori, misurata invece che affermata.

E il rimescolamento, sulle stesse due colonne inventate, dà $+0{,}004$ e
$-0{,}002$: zero entrambe, come dev'essere. Su questo non si fa ingannare dal
numero di valori, perché non guarda le soglie: guarda soltanto se il modello
peggiora.

La riga di `sex`, invece, va letta con prudenza. È una colonna vera, non
inventata da noi, ed è ferma a $0{,}007$, cioè al livello del rumore binario, e
verrebbe voglia di dire che è la sua binarietà a penalizzarla. Può darsi, ma la
tabella non lo dimostra: anche il rimescolamento le dà quasi zero, quindi il
sesso potrebbe semplicemente contare poco per *questo* modello, e le due
spiegazioni producono lo stesso numero. È esattamente per questo che le colonne
inventate servono: di quelle sappiamo in partenza che non valgono niente, e ogni
merito che ricevono è distorsione e basta.

Resta da spiegare perché `s3` e `s6` prendano un'impurità non trascurabile
(circa $0{,}05$) benché il rimescolamento li dichiari inutili. Qui le due
ragioni agiscono **insieme**, dentro lo stesso numero. Nelle 309 righe di
addestramento `s3` e `s6` hanno 59 e 56 valori distinti: molti meno del rumore
continuo, che ne ha 309, ma moltissimi di più di `sex`, che ne ha due.
Cinquantotto soglie fra cui scegliere bastano perché una colonna senza alcun
valore si guadagni comunque un merito, e lo si può misurare: rimescolando `s3`
e `s6` su tutte e 442 le righe, cioè cancellando ogni loro legame con la
malattia e lasciandone intatta la distribuzione, l'impurità che ricevono scende
soltanto a circa $0{,}036$ e $0{,}040$. Due terzi di quel merito, dunque, non
venivano dalla malattia: venivano dal numero di soglie. Il terzo che resta lo
aggiunge la seconda ragione, cioè che i meriti sono accreditati sulle stesse
309 righe da cui gli alberi hanno imparato: là un taglio su `s3` sembrava
utile, sulle 133 righe di prova non serve più. Due meccanismi diversi, sommati
dentro un numero solo, ed è per questo che quella colonna non va letta come una
classifica.

## Che una feature conti, non come, né perché

Chiudiamo con l'avvertenza più importante, la stessa della storia degli
asmatici. L'importanza delle feature (per rimescolamento o da impurità) dice
**che** una colonna pesa sulle risposte del modello. Non dice **come** agisce
(per quello servono le curve di poco fa), non dice se l'effetto sia lo stesso
per tutti (per quello servono i metodi della sezione seguente), e soprattutto
non dice che quella colonna sia la **causa** di niente. Attenzione a questa
parola, che somiglia a un'altra usata dieci volte in questa pagina: «casuale»
vuol dire tirato a sorte, «causale» vuol dire che una cosa ne provoca un'altra,
ed è la seconda che qui stiamo negando.

Un esempio, e sta tutto nella storia degli asmatici di apertura. Là l'asma
risultava importante, e chi avesse letto quel numero come una causa avrebbe
concluso che l'asma protegge dalla polmonite. La causa vera erano le cure
intensive; l'asma era soltanto la colonna che, nei dati, viaggiava insieme a
quelle cure. Il modello ha visto quali cose vanno insieme, non che cosa provoca
che cosa, e le due sono diverse ogni volta che in mezzo c'è qualcosa che nella
tabella non compare. Confondere «colonna importante per il modello»
con «causa del fenomeno» è l'errore che trasforma uno strumento per trovare i
difetti in una fonte di decisioni sbagliate. L'interpretabilità apre la
scatola: sta a noi non leggerci dentro più di quel che c'è.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- I **modelli trasparenti** sono già la propria spiegazione. Nel modello che
  stima il prezzo di una casa ogni peso è un cartellino col prezzo appeso a una
  caratteristica, e la risposta si legge come una ricevuta, voce per voce; in un
  albero la spiegazione è il percorso di domande che porta alla risposta. Sono
  di questa famiglia anche i **modelli additivi generalizzati**, che al posto di
  un cartellino fisso mettono una curva leggibile per ogni caratteristica, e i
  sistemi a regole.
- Il presunto scambio fra accuratezza e chiarezza **non vale sempre**, e sui
  dati a righe e colonne spesso non vale affatto.
- L’**importanza per rimescolamento** (Breiman, 2001; in inglese *permutation
  importance*) rimescola i valori di una sola colonna e guarda quanto peggiora
  il modello: se rimescolando il reddito le risposte giuste scendono dal $90\%$
  al $72\%$, quella colonna vale 18 punti. Funziona con qualunque modello, va
  misurata su dati che il modello non ha mai visto in addestramento e ripetuta
  più volte, facendo la media.
- L'importanza **da impurità** degli alberi (l'impurità è quanto sono mescolate
  le risposte dentro un gruppo: l'albero taglia per fare gruppi più omogenei)
  arriva gratis con l'addestramento ma è **distorta**: premia le colonne con
  tanti valori diversi, che offrono moltissime soglie fra cui scegliere, e
  penalizza quelle con due o tre valori; in più è calcolata sui dati di
  addestramento, dove ogni taglio sembra utile. Due colonne di puro rumore
  aggiunte apposta lo fanno vedere: quella con tanti valori si prende sette
  volte l'altra ($0{,}0467$ contro $0{,}0065$), e valgono zero tutte e due.
  Meglio fidarsi del rimescolamento.
- Sapere quanto una colonna conta non dice **come** agisce. Il **PDP** riscrive
  a tutti lo stesso valore («e se aveste tutti quarant'anni?») e fa la media
  delle risposte; l’**ICE** disegna una curva per ogni esempio e rivela i
  casi in cui l'effetto è opposto da persona a persona e la media lo nasconde.
  Attenzione quando due colonne vanno sempre insieme (l'altezza e il peso, per
  dire): riscrivendone una sola, il PDP finisce per chiedere al modello cosa
  pensa di persone che non esistono, alte due metri e pesanti cinquanta chili,
  e la curva che ne esce inganna. In quel caso si usa l’**ALE**, che confronta
  solo valori vicini fra chi quei valori li ha davvero, senza inventare
  nessuno.
- L'importanza dice **che** una colonna pesa sulle risposte, non come agisce
  né che ne sia la **causa**: il reddito può contare solo perché fa da spia del
  quartiere. È l'errore della regola sugli asmatici; il panorama completo è nel
  manuale di Molnar.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- I **modelli trasparenti** (lineari/logistici, alberi, GAM, regole) sono la
  propria spiegazione: nella regressione lineare ogni coefficiente $w_j$ è
  l'effetto marginale della feature $j$. Il presunto compromesso
  accuratezza/interpretabilità **non vale sempre**, specie sui dati tabellari.
- La **permutation importance** {cite}`breiman2001random` mescola i valori di
  una sola colonna e misura il **calo** di performance ($\mathrm{FI}_j =
  e_{\pi_j} - e_{\text{orig}}$): è **model-agnostic**, va calcolata su dati
  **held-out** e mediata su più permutazioni.
- L'importanza da **impurità** (MDI) negli alberi è gratis ma **distorta**
  {cite}`strobl2007bias`: gonfia le feature continue e ad alta cardinalità (per
  via del numero di split candidati *e* del bootstrap con reimmissione), ed è
  misurata sul training. Preferire la permutazione, calcolata su un hold-out
  indipendente: due colonne di puro rumore, una continua e una binaria,
  ricevono MDI in rapporto sette a uno e permutazione nulla entrambe.
- **PDP** mostra l'effetto marginale *medio* di una feature, **ICE** una curva
  per istanza (rivela le interazioni che il PDP media via); con feature
  **correlate** il PDP estrapola e inganna: meglio **ALE**.
- L'importanza dice **che** una feature conta, non **come** né se è **causale**.
  Correlazione nel modello non è causazione nel mondo. Panoramica completa in
  Molnar {cite}`molnar2022interpretable`.
```

`````
