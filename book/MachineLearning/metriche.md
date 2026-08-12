# Valutare un modello: le metriche

Un modello di machine learning, l'abbiamo visto, "impara" solo se **migliora
con l'esperienza in un compito, e in modo misurabile**: è la definizione di
Mitchell, e la parte che qui ci interessa è l'ultima, la misura. Come la
misuriamo davvero? La domanda sembra tecnica e invece è
la più delicata dell'intero progetto: scegliere male la metrica significa
ottimizzare il modello verso l'obiettivo sbagliato.

Partiamo da un caso che mette in guardia. Immagina un modello che deve
diagnosticare una malattia rara, presente in 1 persona su 100. Un modello
pigro che risponde sempre "sano" indovina il 99% delle volte. Il 99%: un voto
quasi perfetto. Eppure quel modello non ha riconosciuto un solo malato: è del
tutto inutile. L'accuratezza, da sola, ci ha ingannati. Per capire *dove* un
modello sbaglia serve uno strumento più fine.

## La matrice di confusione

Tutto, nella valutazione di un classificatore, parte da qui: contare i quattro
esiti possibili di una previsione binaria; una previsione con due sole
risposte in gioco, sì o no ({numref}`fig-matrice-confusione`).

```{figure} ../figures/matrice-confusione.svg
:name: fig-matrice-confusione
:alt: Matrice due per due; colonne valore reale positivo/negativo, righe predizione positivo/negativo; celle VP, FP, FN, VN.
:width: 75%

La matrice di confusione confronta la predizione del modello con la verità.
Sulla diagonale (VP, VN) le risposte esatte; fuori (FP, FN) i due tipi di
errore.
```

Una parola sull'orientamento, perché è una tabella che si trova disegnata in
tutti i modi possibili e non c'è una convenzione universale. In
{numref}`fig-matrice-confusione` le **colonne** sono la verità, le **righe**
sono la predizione, e la classe positiva viene per prima: il vero positivo
finisce in alto a sinistra. Più avanti vedremo che `scikit-learn` fa
esattamente il contrario su entrambi i fronti, ed è un dettaglio che va saputo
prima di leggere quattro numeri stampati a schermo. Quello che non cambia mai è
il senso: sulla diagonale le risposte giuste, fuori gli errori.

`````{tab} Elementare

Pensa a un rilevatore di fumo. Ci sono quattro cose che può succedere:

- c'è un incendio e l'allarme suona → **vero positivo** (VP): giusto;
- non c'è incendio ma l'allarme suona → **falso positivo** (FP): falso allarme;
- c'è un incendio ma l'allarme resta muto → **falso negativo** (FN): il caso
  più pericoloso;
- non c'è incendio e l'allarme tace → **vero negativo** (VN): giusto.

La matrice di confusione non è altro che questa tabellina: quante volte il
modello ha azzeccato (VP e VN) e, soprattutto, *in che modo* ha sbagliato
(falso allarme o incendio mancato). Non è mai indifferente quale dei due.

Da questi quattro numeri si ricava tutto il resto. Il primo e più ovvio è
l'**accuratezza**: la quota di volte in cui il modello ha risposto giusto,
cioè i due casi buoni (VP e VN) divisi per il totale delle risposte. Cento
allarmi, novantanove giusti: accuratezza del 99%. Tieni d'occhio questo numero,
perché la prossima sezione serve a diffidarne.

`````

`````{tab} Superiore

Per un problema binario con classe *positiva* e *negativa*, ogni predizione
cade in una delle quattro celle: $\text{VP}$, $\text{FP}$, $\text{FN}$,
$\text{VN}$ (con $m = \text{VP}+\text{FP}+\text{FN}+\text{VN}$ esempi totali).
Da questi quattro numeri si ricava ogni metrica di classificazione. La prima,
l'**accuratezza**, è semplicemente la frazione di predizioni corrette:

$$
\text{accuratezza} = \frac{\text{VP}+\text{VN}}{\text{VP}+\text{FP}+\text{FN}+\text{VN}} .
$$

Il numeratore è la diagonale della matrice, il denominatore il totale. Per
problemi con più di due classi la matrice diventa $k \times k$ e l'accuratezza
resta la somma della diagonale sul totale; le metriche che seguono, invece,
sono definite sul caso **binario** e per estenderle a $k$ classi bisogna
scegliere come mediarle, questione tutt'altro che innocua a cui è dedicato un
paragrafo più avanti.

`````

## Perché l'accuratezza inganna

Torniamo al modello che dice sempre "sano". Su 100 pazienti, 99 sani e 1
malato, fa $\text{VN}=99$, $\text{FN}=1$, e $\text{VP}=\text{FP}=0$:
accuratezza del 99%, eppure zero malati trovati. Quando le classi sono
**sbilanciate** (una molto più frequente dell'altra), l'accuratezza premia chi
si limita a predire sempre la classe maggioritaria. È il tranello più comune,
e la ragione per cui non ci si ferma mai alla sola accuratezza. Servono
metriche che guardino separatamente ai due tipi di errore.

L'analogia che chiarisce la posta in gioco è una **guardia notturna** valutata
sul numero di notti "gestite correttamente". Se i furti avvengono una notte su
cento, la guardia che dorme sempre ottiene una valutazione del $99\%$:
identica sulla carta a quella di un collega scrupoloso, e del tutto inutile
nell'unica notte che conta. Il valore della guardia sta tutto nel caso raro;
la metrica lo pesa come un centesimo del totale.

## Precision e recall: due domande diverse

Precision e recall isolano i due errori della matrice e rispondono a due
domande distinte.

`````{tab} Elementare

- La **precision** risponde: *"quando il modello dice positivo, quanto spesso
  ci azzecca?"* Se il filtro antispam sposta 10 email nel cestino e 8 erano
  davvero spam, la precision è $8/10$.
- La **recall** (o *sensibilità*) risponde: *"di tutti i casi positivi
  esistenti, quanti ne ha trovati?"* Se ci sono 20 malati e il modello ne
  individua 15, la recall è $15/20$.

C'è quasi sempre un compromesso: alzare la recall (non lasciarsi scappare
nessun malato) tende ad abbassare la precision (più falsi allarmi), e
viceversa. La **F1** è un voto unico che riassume le due, alto solo quando
*entrambe* sono alte.

`````

`````{tab} Superiore

$$
\text{precision} = \frac{\text{VP}}{\text{VP}+\text{FP}}, \qquad
\text{recall} = \frac{\text{VP}}{\text{VP}+\text{FN}} .
$$

La precision penalizza i falsi positivi (denominatore con $\text{FP}$), la
recall i falsi negativi (denominatore con $\text{FN}$). Per combinarle si usa la
loro **media armonica**, la $F_1$:

$$
F_1 = 2\cdot\frac{\text{precision}\cdot\text{recall}}{\text{precision}+\text{recall}} .
$$

La media armonica, a differenza di quella aritmetica, resta bassa se anche solo
uno dei due valori è basso: non basta eccellere in una per avere una buona
$F_1$. La versione generale $F_\beta$ pesa la recall $\beta$ volte più della
precision, utile quando i due errori non hanno lo stesso costo.

**Con più di due classi va scelta una media, e la scelta è tutto.** Le formule
qui sopra presuppongono una classe «positiva»: con $k$ classi si calcolano
precision, recall e $F_1$ **una per classe** e poi si aggregano, e
`classification_report` ne offre tre modi che non sono intercambiabili.

- **micro**: si sommano VP, FP e FN su tutte le classi *prima* di fare il
  rapporto. Con etichetta singola (ogni esempio appartiene a esattamente una
  classe) $\text{micro-}F_1$ **coincide identicamente con l'accuratezza**, e
  ne eredita quindi tutti i difetti: l'antidoto proposto contro l'accuratezza,
  in questa variante, *è* l'accuratezza.
- **macro**: media aritmetica delle $F_1$ per classe. Dà a una classe da
  trenta esempi lo stesso peso di una da trentamila, ed è la scelta giusta
  quando il valore sta nel raro.
- **weighted**: media pesata per il numero di esempi di ciascuna classe. Sta in
  mezzo, ed è quella che somiglia di più all'accuratezza.

Quanto importi si vede su un caso minimo: tre classi in proporzione
$94/3/3$ e un modello che le due rare non le trova mai. Accuratezza $0{,}940$,
micro-$F_1$ $0{,}940$ (identica), weighted-$F_1$ $0{,}911$, macro-$F_1$
$0{,}323$. Fra la prima e l'ultima ci sono sessanta punti, ed è lo stesso
modello: cambia solo la domanda che si è deciso di fargli.

`````

Quale privilegiare dipende da **quale errore fa più male**. Nello screening di
una malattia grave un falso negativo (un malato dichiarato sano) è
inaccettabile: si punta sulla **recall**, accettando qualche falso allarme in
più. In un filtro antispam è il contrario: un falso positivo butta nel cestino
un'email importante, quindi si privilegia la **precision**, tollerando che
qualche spam passi. Stessa matrice, priorità opposte.

## La curva ROC e l'AUC

Molti classificatori non restituiscono un secco "sì/no" ma una probabilità;
siamo noi a fissare la **soglia** oltre la quale dichiarare positivo. Cambiare
soglia cambia l'equilibrio tra i due errori, e la curva ROC visualizza tutti
questi equilibri in un colpo solo.

```{figure} ../figures/roc-auc-valutare-a-ogni-soglia.svg
:name: fig-curva-roc
:alt: "Piano con il tasso di falsi positivi in ascissa e il tasso di veri positivi in ordinata. La diagonale rappresenta il caso, con area sotto la curva pari a 0,5. Sopra di essa, la curva di un buon modello con area circa 0,9 si inarca verso l'angolo in alto a sinistra; tre punti segnati lungo la curva corrispondono a soglia alta, media e bassa. L'area sotto la curva è l'AUC."
:width: 76%

Ogni punto della curva è una soglia diversa dello stesso modello. Muoversi
lungo la curva non migliora il modello: sceglie quale dei due errori si
preferisce pagare.
```

La diagonale di {numref}`fig-curva-roc` è il termine di paragone che rende
leggibile tutto il resto: è ciò che otterrebbe un modello che tira a
indovinare. L'AUC misura quanto la curva se ne stacca, ed è un numero solo
proprio perché riassume *tutte* le soglie, senza che se ne debba fissare una.

`````{tab} Elementare

Immagina di spostare lentamente la soglia da "sospettosissimo" a "fiducioso".
Per ogni posizione segni due numeri, ed entrambi sono **quote**, non conteggi:
la frazione dei malati che riesci a pescare (su tutti i malati che ci sono) e
la frazione dei sani che disturbi per niente (su tutti i sani che ci sono). Per
questo negli assi della figura si legge «tasso»: sono percentuali, e stanno
tutte e due fra $0$ e $1$. Unendo i punti ottieni la **curva ROC**. Un modello
che distingue bene le due classi disegna una curva che sale subito verso
l'angolo in alto a sinistra; uno che tira a caso segue la diagonale.

L'**AUC** è semplicemente l'area sotto quella curva. Siccome il grafico è un
quadrato di lato $1$, l'area totale disponibile vale $1$, e la diagonale lo
taglia esattamente a metà: ecco perché chi tira a caso prende $0{,}5$ e non
$0$. È un voto unico fra $0{,}5$ (a caso) e $1$ (perfetto), e il suo pregio è
che non dipende dalla soglia scelta.

`````

`````{tab} Superiore

La curva ROC (*Receiver Operating Characteristic*) traccia il **tasso di veri
positivi** contro il **tasso di falsi positivi** al variare della soglia:

$$
\text{TPR} = \frac{\text{VP}}{\text{VP}+\text{FN}} \;(=\text{recall}), \qquad
\text{FPR} = \frac{\text{FP}}{\text{FP}+\text{VN}} .
$$

L'**AUC** (*Area Under the Curve*) è l'area sottesa, in $[0,1]$. Ha una lettura
probabilistica elegante: è la probabilità che il modello assegni a un positivo
scelto a caso uno score più alto che a un negativo scelto a caso, **contando
mezzo punto quando i due score coincidono**:

$$
\text{AUC} = P(s^+ > s^-) + \tfrac{1}{2}\,P(s^+ = s^-).
$$

Il termine dei pareggi non è un cavillo, ed è la statistica di
Mann–Whitney a imporlo: gli score pari sono all'ordine del giorno appena il
modello produce pochi valori distinti (un albero poco profondo, un punteggio a
gradini, un `predict` secco). Su un classificatore che risponde solo $0$ o $1$
(quindi con pareggi a valanga) la definizione senza il mezzo punto dà
$0{,}56$ dove `roc_auc_score` restituisce $0{,}75$: chi controllasse a mano
concluderebbe che la libreria ha un errore.

$0{,}5$ equivale al caso, $1$ alla separazione perfetta. A differenza
dell'accuratezza, l'AUC è indipendente dalla soglia e meno sensibile allo
sbilanciamento, ma su dataset molto sbilanciati la curva *precision–recall* è
spesso più informativa.

`````

## Classi sbilanciate: cosa farci

Diagnosticare non basta. Un modello antifrode che non ha mai segnalato una
frode resta al $99\%$ di accuratezza finché nessuno guarda altrove.

```{figure} ../figures/classi-sbilanciate.svg
:name: fig-classi-sbilanciate
:alt: "Una griglia di cento punti rappresenta un dataset sbilanciato 99 a 1: novantanove negativi e un solo positivo, evidenziato. Una nota indica che il modello che risponde sempre «negativo» sbaglia soltanto su quell'unico punto, ottenendo il 99% di accuratezza e uno 0% di recall."
:width: 78%

Il punto solo, in mezzo agli altri novantanove. Un modello che lo ignora
sbaglia una volta su cento e sembra ottimo; eppure ha mancato l'unica cosa che
gli era stata chiesta di trovare.
```

{numref}`fig-classi-sbilanciate` rende visibile perché l'accuratezza non sia
sbagliata, ma inservibile qui: misura una media su cento casi, mentre il
valore del modello si gioca tutto su uno. Le metriche delle pagine precedenti
servivano proprio a guardare quel punto invece della media; ora la domanda
diventa che cosa farci, e le risposte sono tre, in ordine di quanto costano.

`````{tab} Elementare

Tre leve, in ordine di quanto conviene provarle.

**Cambiare metrica** è gratis ed è il primo passo: la **recall** e la **F1**
che abbiamo appena visto, e soprattutto la curva **precision-recall**, cugina
della ROC ma con la precision al posto dei falsi allarmi. Con sbilanciamenti
estremi è più informativa: la ROC, avendo a denominatore l'enorme mucchio dei
negativi, tende a dipingere quadri troppo ottimisti.

**Spostare la soglia.** Un classificatore produce una probabilità, e la soglia
di $0{,}5$ non ha niente di sacro. Abbassarla a $0{,}2$ significa "segnala
anche i casi dubbi": la recall sale, la precision scende. La soglia giusta
dipende da quanto costa un falso allarme rispetto a un caso mancato: una
decisione di business, non di statistica.

**Pesare le classi.** Quasi tutti i modelli accettano un peso per classe: dire
che un errore sulla classe rara costa cento volte tanto cambia direttamente
cosa l'ottimizzazione considera conveniente. In scikit-learn è
`class_weight="balanced"`, ed è spesso la prima cosa da provare.

Solo dopo si interviene sui **dati**. L'*oversampling* aumenta il numero di
esempi rari: o duplicandoli, o inventandone di nuovi ma verosimili. La ricetta
più nota si chiama **SMOTE** (è il nome di un metodo, non di un programma): per
fabbricare un nuovo caso raro prende due casi rari che si somigliano e ne
costruisce uno a metà strada, come se fra due pazienti di 40 e 50 anni si
inventasse un paziente di 45 con tutti i valori a metà (si dice *interpolare*).
L'*undersampling* fa il contrario: scarta esempi della classe frequente, e così
butta via informazione.

`````

`````{tab} Superiore

Sul ricampionamento vale un'avvertenza che si dimentica spesso: **va applicato
solo al training set, dentro la cross-validation**. Ricampionare prima dello
split significa che copie sintetiche dello stesso positivo finiscono sia in
train sia in validation: il modello riconosce esempi che ha già visto e la
stima delle prestazioni diventa ottimistica in modo invisibile. In pratica lo
si ottiene con la libreria **imbalanced-learn** (è sua sia l'implementazione di
SMOTE sia una `Pipeline` compatibile con scikit-learn): il campionatore va
messo dentro quella pipeline, non applicato al dataset prima; la `Pipeline` di
scikit-learn, da sola, non ammette passi che cambiano il numero di esempi.

Un secondo punto: il ricampionamento **distorce le probabilità predette**. Un
modello addestrato su dati riequilibrati stima $P(y=1\mid x)$ rispetto alla
distribuzione riequilibrata, non a quella reale. Se servono probabilità
calibrate (per una soglia basata sui costi, o per combinarle con altre stime),
occorre ricalibrare, oppure preferire i pesi di classe al ricampionamento.

Infine il criterio decisionale corretto quando i costi sono noti: non "massimizza
F1" ma minimizza il costo atteso. Con $c_{\text{FN}}$ e $c_{\text{FP}}$ i costi
dei due errori, e **posto a zero il costo delle due decisioni corrette**, la
soglia ottimale è

$$
\tau^\star = \frac{c_{\text{FP}}}{c_{\text{FP}} + c_{\text{FN}}} ,
$$

che per $c_{\text{FN}} = 100\,c_{\text{FP}}$ dà $\tau^\star \approx 0{,}01$:
molto lontano dal $0{,}5$ di default.

Quell'ipotesi va detta, perché nei domini in cui la soglia serve davvero è
quasi sempre falsa: un fido concesso a chi restituisce *rende*, uno screening
negativo servito a un sano *costa* comunque la visita. Nel caso generale
{cite}`elkan2001foundations`, detto $c_{ij}$ il costo di predire $i$ quando il
vero è $j$, la soglia è

$$
\tau^\star = \frac{c_{10}-c_{00}}{(c_{10}-c_{00}) + (c_{01}-c_{11})},
$$

e la formula precedente è il caso particolare $c_{00} = c_{11} = 0$. Detto
altrimenti: contano le *differenze* fra il costo di sbagliare e quello di
azzeccare, non i soli costi degli errori, e in un conto di business i ricavi ci
sono.

`````

## Quando il target è un numero: le metriche di regressione

Se il modello non classifica ma prevede una quantità continua (il prezzo di
una casa, la temperatura di domani), la matrice di confusione non serve:
contano gli **scarti** tra valore previsto $\hat{y}$ e valore reale $y$.

`````{tab} Elementare

Prima una parola sul nome: qui la risposta da prevedere si chiama spesso
**target** (in inglese: il bersaglio). È la stessa cosa che finora abbiamo
chiamato «etichetta» o «la risposta giusta», e nelle formule $y$: tre nomi per
un oggetto solo, e conviene riconoscerli tutti perché il libro e le librerie
li usano tutti.

Le più usate sono tre. Il **MAE** è l'errore medio "in valore assoluto": in
media, di quanti euro sbagliamo il prezzo. Il **RMSE** è simile, ma prima di
mediare eleva al quadrato gli errori, e alla fine fa la radice quadrata del
risultato (la *R* del nome sta proprio per «radice»). Il quadrato serve a
**punire di più i grandi svarioni** (sbagliare una volta di 100 pesa più che
sbagliare due volte di 50); la radice serve a rimettere il numero nell'unità di
partenza, perché senza di lei avremmo euro al quadrato. Ecco perché MAE e RMSE
si leggono entrambi in euro. L'**R²** invece è un voto: vale $1$ se la
previsione è perfetta e $0$ se il modello non fa meglio di chi risponde sempre
la media di tutti i valori; e sì, può anche scendere **sotto zero**, se fa
peggio di così.

`````

`````{tab} Superiore

Per $m$ esempi, con predizioni $\hat{y}^{(i)}$ e valori veri $y^{(i)}$:

$$
\text{MAE} = \frac{1}{m}\sum_{i=1}^{m}\bigl|\hat{y}^{(i)}-y^{(i)}\bigr|,
\qquad
\text{MSE} = \frac{1}{m}\sum_{i=1}^{m}\bigl(\hat{y}^{(i)}-y^{(i)}\bigr)^2,
\qquad
\text{RMSE} = \sqrt{\text{MSE}} .
$$

L'MSE eleva al quadrato gli scarti, quindi pesa di più gli errori grandi ed è
più sensibile agli *outlier*; la radice (RMSE) riporta il valore nell'unità del
target. Il **coefficiente di determinazione** $R^2$ confronta l'errore del
modello con quello del predittore banale "media di $y$":

$$
R^2 = 1 - \frac{\sum_i \bigl(y^{(i)}-\hat{y}^{(i)}\bigr)^2}
{\sum_i \bigl(y^{(i)}-\bar{y}\bigr)^2},
$$

dove $\bar{y}$ è la media dei valori osservati. $R^2=1$ è la previsione
perfetta, $R^2=0$ equivale a predire sempre $\bar{y}$; valori negativi sono
possibili e segnalano un modello peggiore della semplice media.

`````

## Quando il target ha un ordine ma non una distanza

Fra la categoria e il numero c'è un caso intermedio che si incontra spesso e
che quasi sempre viene trattato male: il target **ordinale**. La fascia d'età,
le stelle di una recensione, la classe energetica, la gravità di una diagnosi:
sono categorie, ma **in fila**.

`````{tab} Elementare

Nella sezione sull'apprendimento supervisionato l'**ordinalità** era comparsa
come proprietà di una colonna *in ingresso*: la classe energetica di una casa,
i cui valori stanno in fila ma senza una distanza fra loro. Qui è la cosa da
**predire** a essere ordinata, e cambia quale errore conta.

Se il modello deve stimare la fascia d'età e la risposta giusta è «30-40»,
rispondere «40-50» è uno sbaglio piccolo e rispondere «over 70» è uno sbaglio
grosso. L'accuratezza secca non lo sa: per lei sono due errori identici, e un
modello che sbaglia sempre di una fascia sembra pessimo quanto uno che spara a
caso. Peggio, se si sceglie il modello con l'accuratezza, si può finire per
preferire proprio quello che sbaglia di più.

Due rimedi, a seconda di quanto si vuole essere precisi. Il più semplice è
contare giusta anche la risposta **adiacente**, dicendo esplicitamente che si
sta misurando così. Il più solido è usare una misura che **pesa gli errori in
base a quanto sono lontani**, e che quindi punisce lo scambio fra due fasce
vicine molto meno di quello fra la prima e l'ultima.

La terza via è cambiare problema: se le classi hanno un ordine, si può
predire un numero e poi tagliarlo in fasce, trattandolo come una regressione.
Funziona bene quando le fasce sono davvero equidistanti, e male quando non lo
sono (fra «lieve» e «moderato» può esserci molta meno distanza che fra
«moderato» e «grave»).

`````

`````{tab} Superiore

Un target ordinale ha $K$ classi con un ordine totale ma **senza una metrica**
data: sappiamo che $c_1 \prec c_2 \prec c_3$ ma non che la distanza fra le
prime due sia pari a quella fra le seconde. Trattarlo come nominale butta via
l'ordine; trattarlo come numerico gli impone una distanza che non ha.

Sul fronte delle **metriche**, l'accuratezza è insensibile all'ordine. Una
correzione grossolana ma usata è l'accuratezza *one-off*, che conta corretta
anche la classe adiacente ($|\hat{y}-y| \leq 1$): utile per dichiarare quanto
un sistema è «quasi giusto», ma arbitraria, perché la soglia a uno non ha
giustificazione. La misura di riferimento è il **kappa di Cohen pesato
quadraticamente**,

$$
\kappa_w = 1 - \frac{\sum_{ij} w_{ij}\, O_{ij}}{\sum_{ij} w_{ij}\, E_{ij}},
\qquad w_{ij} = \frac{(i-j)^2}{(K-1)^2},
$$

dove $\mathbf{O}$ è la matrice di confusione osservata ed $\mathbf{E}$ quella
attesa per caso date le marginali. I pesi quadratici fanno pagare l'errore in
proporzione al **quadrato** della distanza fra le classi, ed è la ragione per
cui diverse competizioni su diagnosi a stadi l'hanno adottata.

Attenzione però a che cosa fa la normalizzazione per $\mathbf{E}$: corregge
l'accordo dovuto al caso **su quel dataset**, e proprio per questo il valore
**dipende dalle marginali**. È il cosiddetto «paradosso del kappa»
{cite}`feinstein1990high`: a parità di meccanismo d'errore, il kappa crolla
quando le classi si sbilanciano. Con lo stesso modello (sbaglia di una classe
nel 20% dei casi) valutato su tre popolazioni diverse, il kappa pesato passa da
$0{,}90$ con classi bilanciate a $0{,}80$ con proporzioni $90/5/5$ a $0{,}48$
con proporzioni $98/1/1$, mentre l'accuratezza in quegli stessi tre casi
*sale*. Va letta come misura interna a un dataset, non come voto trasportabile:
due sistemi valutati su popolazioni con prevalenze diverse non hanno kappa
confrontabili.

Sul fronte del **modello**, la soluzione elegante è la **regressione
ordinale**: invece di $K$ probabilità indipendenti si stima una variabile
latente continua e $K-1$ soglie, e la probabilità cumulata
$P(y \leq k) = \sigma(\tau_k - f(\mathbf{x}))$ è monotona per costruzione. Il
vantaggio pratico rispetto alla regressione seguita da arrotondamento è che le
soglie sono **apprese** invece che imposte equidistanti, quindi il modello può
scoprire che fra due classi c'è poco spazio e fra altre due molto.

`````

## Scegliere la metrica giusta

Non esiste "la" metrica migliore: esiste quella allineata al problema. Il
target è una categoria o un numero? Se è una categoria, le classi sono
bilanciate (e l'accuratezza può bastare) o sbilanciate, e allora servono
precision, recall, F1 o AUC? E dei due errori, quale costa di più: un falso
allarme o un caso mancato? La metrica non è un dettaglio da consultare alla
fine: è la definizione stessa di "successo" che diamo al modello prima ancora
di addestrarlo.

## In pratica, con scikit-learn

`scikit-learn` calcola tutte queste metriche in poche righe, a partire dalle
etichette vere e dalle predizioni.

```{code-block} python
:class: pt-non-eseguibile

from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, mean_absolute_error, r2_score)

# --- classificazione ---
# Attenzione all'orientamento: scikit-learn mette la VERITA' in riga e la
# PREDIZIONE in colonna, ed elenca le etichette in ordine crescente, quindi la
# classe 0 (negativa) per prima. Esce [[VN, FP], [FN, VP]]: la figura di questa
# sezione, che ha VP in alto a sinistra, e' quella stessa matrice ruotata.
print(confusion_matrix(y_test, y_pred))
# con labels=[1, 0] l'ordine torna quello della figura, VP in alto a sinistra
print(confusion_matrix(y_test, y_pred, labels=[1, 0]))

# precision, recall e F1 per classe; le righe "macro avg" e "weighted avg"
# sono le due medie, e su piu' classi la scelta fra loro cambia il verdetto
print(classification_report(y_test, y_pred))

# AUC: richiede le probabilità, non le classi secche
proba = modello.predict_proba(X_test)[:, 1]   # probabilità della classe positiva
print("AUC:", roc_auc_score(y_test, proba))

# --- regressione: altri dati e un altro modello, il target qui e' continuo ---
print("MAE:", mean_absolute_error(y_test_reg, y_pred_reg))
print("R2 :", r2_score(y_test_reg, y_pred_reg))
```

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Tutto parte dal contare i **quattro esiti** del rilevatore di fumo: allarme
  giusto, falso allarme, incendio mancato, silenzio giusto. Da quei quattro
  numeri si ricava ogni altra misura.
- La percentuale di risposte giuste (l'**accuratezza**) **inganna** quando una
  risposta è molto più frequente dell'altra: la guardia che dorme sempre prende
  99 su 100 e non ha mai fermato un ladro.
- Due domande diverse: *quando dice sì, quanto spesso ci azzecca?* (la
  **precision**) e *di tutti i casi veri, quanti ne trova?* (la **recall**).
  Alzare l'una abbassa l'altra; la **F1** è un voto unico, alto solo se lo sono
  entrambe. Nello screening medico conta di più trovarli tutti, nell'antispam
  conta di più non cestinare un'email buona.
- Spostando la soglia si cambia il compromesso senza riaddestrare niente; la
  **curva ROC** li mostra tutti insieme e l'area sotto di essa (l'**AUC**) è un
  voto fra $0{,}5$ (tira a caso) e $1$ (perfetto).
- Se la risposta è un numero: **MAE** e **RMSE** dicono di quanto sbagliamo,
  nella stessa unità del target (euro, gradi), e l'RMSE è più severo con i
  grandi svarioni; l'**R²** dice quanto siamo meglio di chi risponde sempre la
  media.
- La metrica si sceglie **prima** di addestrare, guardando quale dei due errori
  costa di più. È il modo in cui diciamo al modello che cosa significa
  «riuscire».
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- La **matrice di confusione** conta i quattro esiti (VP, FP, FN, VN): da lì
  nasce ogni metrica di classificazione. L'orientamento non è universale:
  `scikit-learn` usa verità in riga, predizione in colonna, classe $0$ per
  prima.
- L'**accuratezza inganna** con classi sbilanciate: premia chi predice sempre la
  classe maggioritaria.
- **Precision** (pochi falsi allarmi) vs **recall** (pochi casi mancati): la
  $F_1$ le riassume. Privilegia la recall nello screening medico, la precision
  nell'antispam. Con $k$ classi va dichiarata la **media**: la *micro* coincide
  con l'accuratezza, la *macro* è quella che dà voce alle classi rare.
- **AUC**: qualità del classificatore indipendente dalla soglia, tra $0{,}5$
  (a caso) e $1$ (perfetto). È $P(s^+>s^-) + \tfrac12 P(s^+=s^-)$: il mezzo
  punto sui pareggi non è opzionale.
- Con costi noti la soglia ottimale è $c_{\text{FP}}/(c_{\text{FP}} +
  c_{\text{FN}})$ **se** le decisioni corrette non costano né rendono nulla;
  altrimenti contano le differenze fra costi e guadagni.
- Per la **regressione**: MAE e RMSE nell'unità del target (RMSE punisce di più
  i grandi errori), $R^2$ come frazione di varianza spiegata, negativo se il
  modello fa peggio della media.
- Per un target **ordinale**, kappa di Cohen pesato quadraticamente, ricordando
  che dipende dalle marginali e non si confronta fra popolazioni diverse.
- La metrica va scelta *prima*, in base al problema e al costo degli errori.
```

`````
