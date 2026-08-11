# Valutare un modello: le metriche

Un modello di machine learning, l'abbiamo visto, "impara" solo se la sua
performance migliora con l'esperienza: la $P$ della definizione di Mitchell.
Ma quella $P$, come la misuriamo davvero? La domanda sembra tecnica e invece è
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
resta la somma della diagonale sul totale.

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
valore del modello si gioca tutto su uno. Le metriche delle prossime pagine
esistono per guardare quel punto invece della media.

`````{tab} Elementare

Tre leve, in ordine di quanto conviene provarle.

**Cambiare metrica** è gratis ed è il primo passo: recall, F1, e soprattutto
la curva **precision-recall**, che con sbilanciamenti estremi è più
informativa della ROC (quest'ultima tende a dipingere quadri troppo
ottimisti).

**Spostare la soglia.** Un classificatore produce una probabilità, e la soglia
di $0{,}5$ non ha niente di sacro. Abbassarla a $0{,}2$ significa "segnala
anche i casi dubbi": la recall sale, la precision scende. La soglia giusta
dipende da quanto costa un falso allarme rispetto a un caso mancato: una
decisione di business, non di statistica.

**Pesare le classi.** Quasi tutti i modelli accettano un peso per classe: dire
che un errore sulla classe rara costa cento volte tanto cambia direttamente
cosa l'ottimizzazione considera conveniente. In scikit-learn è
`class_weight="balanced"`, ed è spesso la prima cosa da provare.

Solo dopo si interviene sui **dati**: l'*oversampling* duplica o sintetizza
esempi rari (SMOTE li genera interpolando fra positivi vicini),
l'*undersampling* scarta esempi della classe frequente buttando informazione.

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
dei due errori, la soglia ottimale è

$$
\tau^\star = \frac{c_{\text{FP}}}{c_{\text{FP}} + c_{\text{FN}}} ,
$$

che per $c_{\text{FN}} = 100\,c_{\text{FP}}$ dà $\tau^\star \approx 0{,}01$:
molto lontano dal $0{,}5$ di default.

`````

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
Per ogni posizione segni due numeri: quanti veri positivi peschi e quanti falsi
allarmi generi. Unendo i punti ottieni la **curva ROC**. Un modello che
distingue bene le due classi disegna una curva che sale subito verso l'angolo
in alto a sinistra; uno che tira a caso segue la diagonale. L'**AUC** è
semplicemente l'area sotto quella curva: un voto unico tra $0{,}5$ (a caso) e
$1$ (perfetto), che non dipende dalla soglia scelta.

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
scelto a caso uno score più alto che a un negativo scelto a caso. $0{,}5$ equivale
al caso, $1$ alla separazione perfetta. A differenza dell'accuratezza, l'AUC è
indipendente dalla soglia e meno sensibile allo sbilanciamento, ma su dataset
molto sbilanciati la curva *precision–recall* è spesso più informativa.

`````

## Quando il target è un numero: le metriche di regressione

Se il modello non classifica ma prevede una quantità continua (il prezzo di
una casa, la temperatura di domani), la matrice di confusione non serve:
contano gli **scarti** tra valore previsto $\hat{y}$ e valore reale $y$.

`````{tab} Elementare

Le più usate sono tre. Il **MAE** è l'errore medio "in valore assoluto": in
media, di quanti euro sbagliamo il prezzo. Il **RMSE** è simile, ma eleva al
quadrato gli errori prima di mediarli: così **punisce di più i grandi
svarioni** (sbagliare di 100 pesa molto più di due errori da 50). Entrambi si
leggono nella stessa unità del target. L'**R²** invece è un voto da $0$ a $1$:
dice quanta parte della variabilità dei dati il modello riesce a spiegare; $1$
è perfetto, $0$ è come indovinare sempre la media.

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

Il libro ha già incontrato l'ordinalità come proprietà di una *feature*, cioè
di un dato in ingresso. Qui è la cosa da **predire** a essere ordinata, e
cambia quale errore conta.

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

dove $O$ è la matrice di confusione osservata ed $E$ quella attesa per caso
date le marginali. I pesi quadratici fanno pagare l'errore in proporzione al
**quadrato** della distanza fra le classi, e la normalizzazione per $E$
corregge per l'accordo dovuto al caso, il che la rende robusta anche a classi
molto sbilanciate. È la metrica scelta da diverse competizioni su diagnosi a
stadi, per esattamente questa ragione.

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
print(confusion_matrix(y_test, y_pred))       # la matrice VP/FP/FN/VN
print(classification_report(y_test, y_pred))  # precision, recall, F1 per classe

# AUC: richiede le probabilità, non le classi secche
proba = modello.predict_proba(X_test)[:, 1]   # probabilità della classe positiva
print("AUC:", roc_auc_score(y_test, proba))

# --- regressione ---
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 :", r2_score(y_test, y_pred))
```

```{admonition} Da ricordare
:class: important
- La **matrice di confusione** conta i quattro esiti (VP, FP, FN, VN): da lì
  nasce ogni metrica di classificazione.
- L'**accuratezza inganna** con classi sbilanciate: premia chi predice sempre la
  classe maggioritaria.
- **Precision** (pochi falsi allarmi) vs **recall** (pochi casi mancati): la
  $F_1$ le riassume. Privilegia la recall nello screening medico, la precision
  nell'antispam.
- **AUC**: qualità del classificatore indipendente dalla soglia, tra $0{,}5$
  (a caso) e $1$ (perfetto).
- Per la **regressione**: MAE e RMSE nell'unità del target (RMSE punisce di più
  i grandi errori), $R^2$ come frazione di varianza spiegata.
- La metrica va scelta *prima*, in base al problema e al costo degli errori.
```
