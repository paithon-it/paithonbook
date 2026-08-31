# Valutare un modello: le metriche

Un modello deve diagnosticare una malattia rara, presente in una persona su
cento. Il modello più pigro del mondo, quello che risponde sempre «sano»,
indovina il 99% delle volte: un voto quasi perfetto, e non ha riconosciuto un
solo malato. È il caso da tenere in mente, perché dice in due righe che cosa
può andare storto quando si sceglie male il modo di dare un voto a un modello:
si finisce per premiare l'obiettivo sbagliato. L'accuratezza, da sola, ci ha
ingannati, e per capire *dove* un modello sbaglia serve uno strumento più fine.

Come si misura davvero, allora, se un modello è bravo?

## La matrice di confusione

Tutto, nella valutazione di un classificatore, parte da qui: contare i quattro
esiti possibili di una previsione **binaria**, cioè con due sole risposte in
gioco, sì o no.

Prima però va tolta di mezzo una parola che confonde tutti. Delle due risposte,
una si chiama **positiva** e l'altra **negativa**, e non hanno niente a che
vedere con «buona» e «cattiva»: positiva è la cosa che stiamo cercando, che
quasi sempre è la cosa brutta. Un incendio è positivo, un tumore è positivo,
una frode è positiva. Il malato è il caso positivo e il sano è il negativo, ed
è il motivo per cui un referto medico «positivo» è una brutta notizia.

`````{tab} Elementare

Un rilevatore di fumo ha quattro esiti possibili, due giusti e due sbagliati:

- c'è un incendio e l'allarme suona → **vero positivo** (VP): giusto;
- non c'è incendio ma l'allarme suona → **falso positivo** (FP): falso allarme;
- c'è un incendio ma l'allarme resta muto → **falso negativo** (FN): il caso
  più pericoloso;
- non c'è incendio e l'allarme tace → **vero negativo** (VN): giusto.

La matrice di confusione non è altro che questa tabellina: quante volte il
modello ha azzeccato (VP e VN) e, soprattutto, *in che modo* ha sbagliato
(falso allarme o incendio mancato). Non è mai indifferente quale dei due.

Da questi quattro numeri si ricava tutto il resto. Il primo e più ovvio è
l’**accuratezza**: la quota di volte in cui il modello ha risposto giusto,
cioè i due casi buoni (VP e VN) divisi per il totale delle risposte. Cento
allarmi, novantanove giusti: accuratezza del 99%. Il conto regge anche quando
le risposte in gioco sono più di due (silenzio, fumo di padella, incendio
vero): la tabellina cresce, e l'accuratezza resta le risposte giuste divise per
tutte. Tieni d'occhio questo numero, perché è anche il più facile da prendere
per buono a torto.

`````

`````{tab} Superiore

Per un problema binario con classe *positiva* e *negativa*, ogni predizione
cade in una delle quattro celle: $\text{VP}$, $\text{FP}$, $\text{FN}$,
$\text{VN}$ (con $m = \text{VP}+\text{FP}+\text{FN}+\text{VN}$ esempi totali).
Da questi quattro numeri si ricava ogni metrica di classificazione. La prima,
l’**accuratezza**, è semplicemente la frazione di predizioni corrette:

$$
\text{accuratezza} = \frac{\text{VP}+\text{VN}}{\text{VP}+\text{FP}+\text{FN}+\text{VN}} .
$$

Il numeratore è la diagonale della matrice, il denominatore il totale. Per
problemi con più di due classi la matrice diventa $K \times K$ e l'accuratezza
resta la somma della diagonale sul totale; le metriche che seguono, invece,
sono definite sul caso **binario** e per estenderle a $K$ classi bisogna
scegliere come mediarle, questione tutt'altro che innocua.

`````

Messi in tabella, quei quattro numeri sono la **matrice di confusione**
({numref}`fig-matrice-confusione`).

```{figure} ../figures/matrice-confusione.svg
:name: fig-matrice-confusione
:alt: Matrice due per due; colonne valore reale positivo/negativo, righe predizione positivo/negativo; celle VP, FP, FN, VN.
:width: 75%

La matrice di confusione confronta la predizione del modello con la verità.
Sulla diagonale (i veri positivi e i veri negativi) le risposte esatte; fuori
(i falsi positivi e i falsi negativi) i due tipi di errore.
```

Una parola sull'orientamento, perché è una tabella che si trova disegnata in
tutti i modi possibili e non c'è una convenzione universale. In
{numref}`fig-matrice-confusione` le **colonne** sono la verità, le **righe**
sono la predizione, e la classe positiva viene per prima: il vero positivo
finisce in alto a sinistra. In `scikit-learn` l'orientamento è
esattamente il contrario su entrambi i fronti, ed è un dettaglio che va saputo
prima di leggere quattro numeri stampati a schermo. Quello che non cambia mai è
il senso: sulla diagonale le risposte giuste, fuori gli errori.

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

Il filtro antispam sposta dieci email nel cestino, e otto erano davvero spam.
Otto su dieci è la sua **precision**: quando dice positivo, quanto spesso ci
azzecca. La **recall** (o *sensibilità*) guarda l'altra metà della storia,
cioè quanti dei positivi che esistono riesce a pescare. Su venti malati ne
individua quindici, e la recall è $15/20$.

Il medico che non vuole lasciarsene scappare nessuno diventa sospettoso e
segnala al minimo dubbio. Ormai li prende quasi tutti e la recall sale, ma fra
i segnalati finiscono anche parecchi sani, e la precision scende. Il collega
che manda a fare gli esami solo i casi lampanti fa il percorso opposto, e ci
azzecca quasi sempre, mentre parecchi malati escono dall'ambulatorio senza che
nessuno si sia accorto di niente. Sono i due piatti di una bilancia, e a
spostarla basta un numero.

La **F1** è il voto unico che tiene conto di tutti e due i piatti, e resta
alto solo quando lo sono entrambi. I due numeri devono uscire dallo stesso
modello sulle stesse prove, e l'otto su dieci del cestino e il quindici su
venti dell'ambulatorio vengono da macchine diverse e non si mescolano.
Prendiamo un filtro prudente, che azzecca nove volte su dieci quando cestina
ma di spam ne pesca solo quattro su dieci, cioè precision $0{,}9$ e recall
$0{,}4$. La sua F1 vale $0{,}55$. Non $0{,}65$, che è la media dei due, ma
molto più vicina al piatto peggiore. Se la recall crollasse a $0{,}1$, la F1
crollerebbe con lei anche con la precision al massimo.

Il voto si può anche sbilanciare apposta, dichiarando quante volte un malato
mancato pesa più di un falso allarme. Da quel momento premia chi non se ne
lascia scappare nessuno, e qualche esame fatto per niente non basta più ad
abbassarlo.

Con più di due risposte in gioco c'è una decisione da prendere prima del voto,
e pesa più di quanto sembri. Un rilevatore a tre risposte passa cento notti in
casa: novantaquattro volte non succede niente, tre volte c'è del fumo di
padella, tre volte un incendio vero, e lui dice sempre «niente». Alla risposta
«niente» tocca un voto altissimo, $0{,}97$, perché delle novantaquattro notti
tranquille non se ne lascia sfuggire una e la dà a sproposito soltanto nelle
sei rimanenti. Alle altre due tocca zero, dato che non le nomina mai. Chi fa
la media dei tre voti ottiene $0{,}32$. Chi butta tutte le risposte in un
mucchio solo e dà un voto al mucchio ottiene $0{,}94$, che è poi la
percentuale di notti indovinate, cioè l'accuratezza, rientrata dalla finestra
proprio nella misura scelta per non farsene ingannare. Chi pesa i tre voti per
quanto spesso ciascuna risposta capita ottiene $0{,}91$, di nuovo a un soffio
dall'accuratezza. Stesso apparecchio, stesse cento notti; cambia solo come si
è deciso di fare la somma.

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
$F_1$. Quando i due errori non hanno lo stesso costo si usa la versione
generale, che sposta il peso da una parte:

$$
F_\beta = (1+\beta^2)\,\frac{\text{precision}\cdot\text{recall}}
{\beta^2\,\text{precision} + \text{recall}},
$$

dove $\beta$ è la manopola che decide a quale
delle due si tiene di più. Attenzione a quanto sposta: nella formula la recall
entra con $\beta^2$, e questo va letto sapendo di che peso si parla. Nella
media armonica il termine della recall vale $\beta^2$ volte quello della
precision: partendo da precision e recall uguali, con $\beta = 2$ un punto di
recall in più vale quattro punti di precision. La formulazione tradizionale di
van Rijsbergen dice invece che $\beta$ è il rapporto fra recall e precision al
quale i due errori pesano uguale, e in quel senso $\beta = 2$ vuol dire «recall
due volte più importante». Stesso $\beta$, due letture, e conviene dichiarare
quale si sta usando; per $\beta = 1$ coincidono e si ritrova la $F_1$.

**Con più di due classi va scelta una media, e la scelta è tutto.** Precision,
recall e $F_\beta$ presuppongono una classe «positiva»: con $K$ classi si calcolano
precision, recall e $F_1$ **una per classe** e poi si aggregano. I modi di
aggregare sono tre, e non sono intercambiabili.

- **micro**: si sommano VP, FP e FN su tutte le classi *prima* di fare il
  rapporto. Con etichetta singola (ogni esempio appartiene a esattamente una
  classe) $\text{micro-}F_1$ **coincide identicamente con l'accuratezza**, e
  ne eredita quindi tutti i difetti: l'antidoto proposto contro l'accuratezza,
  in questa variante, *è* l'accuratezza. È il motivo per cui
  `classification_report` non stampa nessuna riga «micro»: al suo posto scrive
  `accuracy`.
- **macro**: media aritmetica delle $F_1$ per classe. Dà a una classe da
  trenta esempi lo stesso peso di una da trentamila, ed è la scelta giusta
  quando il valore sta nel raro.
- **weighted**: media pesata per il numero di esempi di ciascuna classe. Sta in
  mezzo, ed è quella che somiglia di più all'accuratezza.

Quanto importi si vede su un caso minimo: tre classi in proporzione $94/3/3$ e
un modello che risponde sempre la classe frequente, quindi le due rare non le
trova mai. Accuratezza $0{,}940$, micro-$F_1$ $0{,}940$ (identica),
weighted-$F_1$ $0{,}911$, macro-$F_1$ $0{,}323$. Fra l'accuratezza e la macro
ci sono più di sessanta punti, ed è lo stesso modello: cambia solo la domanda
che si è deciso di fargli.

`````

Quale privilegiare dipende da **quale errore fa più male**. Nello screening di
una malattia grave un falso negativo (un malato dichiarato sano) è
inaccettabile: si punta sulla **recall**, accettando qualche falso allarme in
più. In un filtro antispam è il contrario: un falso positivo butta nel cestino
un'email importante, quindi si privilegia la **precision**, tollerando che
qualche spam passi. Stessa matrice, priorità opposte.

## La curva ROC e l'AUC

Ecco il numero che sposta la bilancia. Come si è visto parlando di regressione
logistica, molti classificatori non restituiscono un secco «sì/no» ma una
probabilità, un numero fra $0$ e $1$; siamo noi a fissare la **soglia** oltre
la quale dichiararlo positivo, e per abitudine si parte da $0{,}5$. Cambiare
soglia cambia l'equilibrio tra i due errori, senza toccare il modello, e la
curva ROC li mostra tutti in un colpo solo.

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
indovinare. Più la curva sale verso l'angolo in alto a sinistra, meglio va, e l’**AUC** è il
modo di ridurre quel confronto a un numero solo: è l'area che resta sotto la
curva. Il suo pregio è di riassumere *tutte* le soglie, senza che se ne debba
fissare una.

`````{tab} Elementare

Sposta lentamente la soglia dal più basso al più alto, cioè da
sospettosissimo (segnalo tutto, anche col $10\%$ di probabilità) a fiducioso
(segnalo solo se sono quasi certo).
Per ogni posizione segni due numeri, ed entrambi sono **quote**, non conteggi:
la frazione dei malati che riesci a pescare (su tutti i malati che ci sono) e
la frazione dei sani che disturbi per niente (su tutti i sani che ci sono). Per
questo negli assi della figura si legge «tasso»: sono percentuali, e stanno
tutte e due fra $0$ e $1$. Unendo i punti ottieni la **curva ROC**. Un modello
che distingue bene le due classi disegna una curva che sale subito verso
l'angolo in alto a sinistra; uno che tira a caso segue la diagonale.

L’**AUC** è semplicemente l'area sotto quella curva. Siccome il grafico è un
quadrato di lato $1$, l'area totale disponibile vale $1$, e la diagonale lo
taglia esattamente a metà: ecco perché chi tira a caso prende $0{,}5$ e non
$0$. È un voto unico fra $0$ e $1$, dove $1$ è la separazione perfetta e
$0{,}5$ non è il minimo ma il punteggio di chi risponde a caso: sotto quella
linea si può scendere, e vuol dire che il modello ha invertito le due classi
(uno che prende $0{,}2$, letto al contrario, ne vale $0{,}8$). Il pregio
dell'AUC è che non dipende dalla soglia scelta.

C'è un secondo modo di leggere quel numero, ed è il più maneggevole: pesca a
caso un malato e a caso un sano, e guarda a quale dei due il modello ha dato il
sospetto più alto. L'AUC è la quota di coppie in cui vince il malato, con una
regola in più per i pari merito, che valgono mezza vittoria. I pari merito non
sono una rarità: un modello che risponde soltanto «sì» o «no», senza sfumature,
ne produce di continuo. Uno che segnala il $70\%$ dei malati e il $20\%$ dei sani
vince le coppie in cui il malato ha il «sì» e il sano il «no», che sono
$0{,}7 \times 0{,}8 = 0{,}56$, poco più della metà; le coppie pari (tutti e due
«sì», o tutti e due «no») sono un altro $0{,}38$, e la loro metà porta il conto
a $0{,}75$. Chi si dimenticasse i pari merito si fermerebbe a $0{,}56$ e
darebbe del bugiardo al calcolatore.

`````

`````{tab} Superiore

La curva ROC (*Receiver Operating Characteristic*) traccia il **tasso di veri
positivi** contro il **tasso di falsi positivi** al variare della soglia:

$$
\text{TPR} = \frac{\text{VP}}{\text{VP}+\text{FN}} \;(=\text{recall}), \qquad
\text{FPR} = \frac{\text{FP}}{\text{FP}+\text{VN}} .
$$

L’**AUC** (*Area Under the Curve*) è l'area sottesa, in $[0,1]$. Ha una lettura
probabilistica elegante: è la probabilità che il modello assegni a un positivo
scelto a caso uno score più alto che a un negativo scelto a caso, **contando
mezzo punto quando i due score coincidono**:

$$
\text{AUC} = P(s^+ > s^-) + \tfrac{1}{2}\,P(s^+ = s^-),
$$

dove $s^+$ è lo score che il modello dà a un positivo estratto a caso e $s^-$
quello che dà a un negativo estratto a caso.

Il termine dei pareggi non è un cavillo, ed è la statistica di
Mann–Whitney a imporlo: gli score pari sono all'ordine del giorno appena il
modello produce pochi valori distinti (un albero poco profondo, un punteggio a
gradini, un `predict` secco). Su un classificatore che risponde solo $0$ o $1$
(quindi con pareggi a valanga) il conto si fa a mano. Se segnala il $70\%$ dei
positivi e il $20\%$ dei negativi, la coppia in cui il positivo prende lo score
più alto capita nel $0{,}7 \cdot 0{,}8 = 0{,}56$ dei casi, e la definizione
senza il mezzo punto si ferma lì; i pareggi sono un altro $0{,}38$, e la
loro metà porta a $0{,}75$, che è quanto restituisce `roc_auc_score`. Chi si
fermasse al primo numero concluderebbe che la libreria ha un errore.

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
diventa che cosa farci. Ci sono quattro leve, e conviene provarle in
quest'ordine, perché è l'ordine che va dalla più economica alla più invasiva.

`````{tab} Elementare

Sul tavolo di chi dà la caccia alle frodi arrivano ogni mattina le operazioni
che il modello ha segnalato, e vanno aperte a una a una.

La prima leva costa zero e cambia solo il voto: al posto dell'accuratezza, la
recall, la F1 e soprattutto la curva **precision-recall**, che si costruisce
come la ROC spostando la soglia, ma i due numeri segnati sono la recall e la
precision. Con sbilanciamenti estremi dice di più, e si capisce guardando quel
tavolo. La ROC conta i falsi allarmi in rapporto a tutti i conti tranquilli,
che sono novantanove su cento, e mille segnalazioni a vuoto su centomila le
sembrano una quisquilia; sul tavolo sono mille pratiche da aprire a mano. La
precision le conta rispetto alle sole operazioni segnalate, e il disastro si
vede.

Poi c'è la soglia. Il modello non risponde «frode» o «niente», dà una
probabilità, e quel $0{,}5$ da cui si parte per abitudine non ha niente di
sacro. Portarlo a $0{,}2$ vuol dire segnalare anche i casi dubbi, e allora il
tavolo si riempie, la recall sale e la precision scende. Dove metterlo dipende
da quanto costa una pratica aperta per niente rispetto a una frode che passa,
ed è una decisione di chi dirige la banca, non dello statistico. E il conto si
fa davvero. Se una frode mancata costa cento volte un falso allarme, la soglia
scende attorno a un sospetto su cento ($0{,}01$), lontanissima dal mezzo di
partenza. Regge finché rispondere giusto non costa e non rende niente, e allo
sportello non è mai così, perché un prestito concesso a chi poi restituisce
rende, e la visita fatta a un sano si paga comunque. Allora nel conto entra
anche la differenza fra il guadagno di chi azzecca e la perdita di chi
sbaglia.

La terza leva è il peso delle classi. Addestrare vuol dire far scendere un
numero, quello che conta quanto il modello sbaglia. Dire che una frode mancata
conta cento volte tanto cambia quel conteggio, e con esso cosa al modello
conviene fare. In scikit-learn si scrive `class_weight="balanced"`, ed è
spesso la prima cosa da provare. Il prezzo lo pagano le percentuali che il
modello annuncia, gonfiate tutte insieme: dove diceva «quattro su cento»
arriva a «venticinque su cento», e le frodi nel frattempo non sono aumentate.
Chi le legge come probabilità vere si sbaglia, e lo stesso gonfiaggio, tale e
quale, arriva anche duplicando i casi rari.

Solo come quarta mossa si toccano i dati. L’*oversampling* aumenta il numero
di esempi rari, duplicandoli oppure inventandone di nuovi ma verosimili. La
ricetta più nota si chiama **SMOTE** (è il nome di un metodo, non di un
programma): per fabbricare una frode nuova ne prende due che si somigliano e
ne costruisce una intermedia, come se fra due clienti di 40 e 50 anni ne
spuntasse uno di 45, con gli altri valori a metà strada (si dice
*interpolare*). L’*undersampling* fa il contrario, scarta operazioni della
classe frequente e butta via informazione.

Sui casi fabbricati c'è un ordine da rispettare, o si rovina tutto il resto.
Si aggiungono dopo aver messo da parte le operazioni con cui si darà il voto,
mai prima. Altrimenti le copie di una stessa frode finiscono metà fra gli
esercizi e metà fra le domande d'esame, e all'esame il modello ritrova facce
già viste e prende un voto che non ha meritato. Il guasto non lascia tracce,
perché i conti tornano, e tornano troppo belli.

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
distribuzione riequilibrata, non a quella reale. La tentazione, a questo punto,
è di cavarsela preferendo i pesi di classe al ricampionamento, e non funziona:
i pesi distorcono allo stesso modo, perché fanno la stessa cosa (contare la
classe rara più di quanto sia). Su una regressione logistica addestrata su
`make_classification(n_samples=5000, n_features=10, n_informative=5,
weights=[0.966], flip_y=0.01, random_state=0)`, che dà una prevalenza reale
del $3{,}9\%$, la probabilità media prevista vale $0{,}039$
lasciando il modello com'è (cioè esattamente la prevalenza, com'è giusto che
sia), $0{,}252$
con `class_weight="balanced"` e $0{,}253$ duplicando i positivi fino a
pareggiare le due classi: le due correzioni
sono indistinguibili. Se servono probabilità calibrate (per una soglia basata
sui costi, o per combinarle con altre stime) va ricalibrato in ogni caso; in
alternativa si lascia il modello sbilanciato com'è e si sposta la **soglia**
secondo i costi.

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

## Quando il modello dice novanta: la calibrazione

Sul tavolo dell'antifrode arriva un'operazione che il modello ha segnato al
novantaquattro per cento. L'analista la apre, e non c'è nessuna frode. Il
modello ha sbagliato? No, e la risposta secca è il punto di partenza. Dire
«novantaquattro su cento è frode» vuol dire anche «sei su cento è pulita», e
quell'operazione può essere una delle sei: una singola affermazione
probabilistica non si può smentire. L'unica cosa che si può mettere alla prova
è il gruppo. Si prendono tutte le operazioni segnate attorno al novantaquattro,
si aprono, e si contano le frodi: se sono novantaquattro su cento il numero era
una promessa mantenuta, se sono sessanta il modello si vantava.

Quella corrispondenza fra il numero detto e la frequenza osservata si chiama
**calibrazione**. È ciò che i pesi di classe e i casi rari duplicati rompono,
gonfiando tutte le probabilità insieme, ed è la ragione per cui dopo quelle
leve un modello va ricalibrato. Che cosa voglia dire, come si misura e come si
ripara, è quello che segue.

`````{tab} Elementare

Alla fine del mese sul tavolo dell'antifrode si è accumulato un mazzo di
operazioni chiuse, di cui ormai si sa com'è andata. Per giudicare i numeri che
il modello aveva dato si fa una cosa sola: si divide il mazzo in **dieci
cassetti**, uno per ogni decimo della scala. Dieci è una scelta di chi fa il
conto, non una legge: con cinque cassetti si vede meno, con cinquanta ciascuno
resta quasi vuoto e i conti ballano. Nel primo finiscono le operazioni segnate
sotto il dieci per cento, nell'ultimo quelle sopra il novanta. Poi, cassetto
per cassetto, si scrivono due numeri accanto: la media di quello che il modello
aveva detto, e la frazione di frodi che c'erano davvero. Se le due colonne si
somigliano il modello è onesto; dove si allontanano, si vede di quanto e in che
verso.

Per avere un voto unico si prende la distanza fra le due colonne in ogni
cassetto, la si moltiplica per la quota di mazzo che quel cassetto contiene, e
si sommano i dieci prodotti. Nell'ultimo cassetto, per dire, la distanza è
$0{,}937 - 0{,}613 = 0{,}324$, e quel cassetto tiene $181$ operazioni su
$7500$: il suo contributo al voto è $0{,}324 \times 181 / 7500$, cioè meno di
un centesimo. Un cassetto con duemila operazioni pesa più di uno con dieci.
Viene un numero fra zero e uno, e più è piccolo meglio è.

La riparazione non chiede di riaddestrare niente, ed è di una semplicità che
sorprende: si costruisce una **tavola di conversione**. Si prende un secondo
mazzo di operazioni chiuse, che il modello non ha mai visto (se fossero le
stesse su cui ha imparato, gli si starebbe chiedendo di correggere il proprio
compito con le risposte in mano), si guarda cassetto per cassetto quanto
sbaglia, e si scrive la regola che porta il detto sul vero: dove dice
novantaquattro, il cassetto di sopra dice che le frodi erano sessantuno, e
sessantuno si scrive. Le forme di quella tavola sono due. La prima è una
**curva liscia** con due manopole, una che allarga o stringe la scala e una che
la sposta tutta in su o in giù (e ne esiste una versione ridotta, con la sola
prima manopola); la seconda è una **scaletta** libera di salire come vuole,
purché salga sempre. La scaletta si adatta meglio, ma con poche operazioni su
cui impararla copia le loro coincidenze invece della regola, e allora conviene
la curva.

Il pregio della conversione è che non scavalca: quello che stava sopra resta
sopra, quindi l'operazione più sospetta di tutte resta la più sospetta.
Cambiano i numeri, non la graduatoria, e tutto ciò che si giudicava sulla
graduatoria (la curva ROC, l'area sotto di essa) resta identico. Con una
riserva sulla scaletta, che è fatta di gradini: due operazioni un po' diverse
possono finire sullo stesso gradino e diventare pari merito, e i pari merito
nella graduatoria contano mezzo punto ciascuno.

E una trappola, che è la ragione per cui la calibrazione da sola non è un voto
sufficiente. Un modello che a ogni operazione risponde «quattro su cento»,
sempre lo stesso numero, finisce tutto in un cassetto solo, e in quel cassetto
il conto torna: di frodi ce ne sono davvero quattro per cento. È calibrato
quasi alla perfezione ed è del tutto inutile, perché non distingue
un'operazione dall'altra. Onestà e capacità di distinguere sono due virtù
separate: la prima è il minimo che si pretende, la seconda è quella per cui il
modello viene pagato.

`````

`````{tab} Superiore

Un modello che produce $\hat{p}(\mathbf{x}) \in [0,1]$ è **calibrato** se

$$
P\big(y = 1 \;\big|\; \hat{p}(\mathbf{x}) = q\big) = q
\qquad \text{per ogni } q \in [0,1] .
$$

Quando il punteggio è continuo la condizione riguarda un evento di probabilità
nulla; e comunque, su un campione finito, per ciascun valore di $q$ ci sarebbero
pochissimi casi. Si stima quindi raggruppando: si partiziona $[0,1]$ in $M$
intervalli, e per ciascuno si confrontano la confidenza media
$\bar{p}_b$ e la frequenza osservata $\bar{y}_b$. Il grafico delle coppie
$(\bar{p}_b, \bar{y}_b)$ è il **diagramma di affidabilità**, cioè la resa
grafica della condizione di calibrazione, che DeGroot e Fienberg mettono al
centro della valutazione di un previsore accanto alla capacità di discriminare
{cite}`degroot1983comparison`. La sintesi in un numero è l’**errore atteso di
calibrazione**

$$
\mathrm{ECE} = \sum_{b=1}^{M} \frac{n_b}{n}\,\big|\bar{p}_b - \bar{y}_b\big| ,
$$

dove $n_b$ sono le osservazioni cadute nel $b$-esimo intervallo e $n$ il
totale. La formula è quella del binning bayesiano {cite}`naeini2015obtaining`,
che però partiziona a massa uguale; a fette uguali della scala, come qui e come
nella maggior parte del deep learning, la convenzione è quella di Guo e colleghi
{cite}`guo2017calibration`. Ed è una stima distorta in due versi opposti: con
pochi casi per intervallo il rumore di conteggio la fa crescere anche su un
modello impeccabile, mentre con intervalli larghi la media dentro l'intervallo
può nascondere una miscalibrazione vera. Cambiando $M$, o passando da fette
uguali a masse uguali, il numero cambia: va dichiarato con la sua partizione o
non è confrontabile.

Le tre ricalibrazioni post-hoc sono tutte trasformazioni monotone del punteggio
$f(\mathbf{x})$, stimate su un insieme indipendente da quello di
addestramento. Lo **scaling di Platt** {cite}`platt1999probabilistic` adatta
una sigmoide $\sigma(a f + b)$ per massima verosimiglianza su bersagli
leggermente ammorbiditi ($(N_++1)/(N_++2)$ al posto di $1$, e simmetricamente
per gli zeri), che sono il freno al sovradattamento incorporato nel metodo: due
parametri.
Lo **scaling di temperatura** {cite}`guo2017calibration` ne fissa uno,
$\sigma(f / T)$, e si estende al caso multiclasse dividendo per lo stesso $T$
tutti i punteggi che entrano nella softmax (i *logit*): è la ricetta oggi più
usata sulle reti profonde. Non può correggere uno spostamento sistematico,
perché $\sigma(0/T) = 1/2$ per ogni $T>0$: il centro della scala è un punto
fisso che nessuna temperatura sposta, e quella manopola stringe o allarga
soltanto. La
**regressione isotonica** {cite}`zadrozny2002transforming` cerca invece una
qualunque funzione non decrescente, con l'algoritmo *pool adjacent violators*:
più espressiva, e più incline a sovradattarsi quando i dati di calibrazione
scarseggiano.

Quale serva dipende dalla forma della distorsione, e Niculescu-Mizil e Caruana
l'hanno mappata su dieci algoritmi {cite}`niculescu2005predicting`: i metodi a
massimo margine (SVM, alberi con boosting) allontanano la massa da $0$ e da
$1$, con una distorsione a forma di sigmoide che lo scaling di Platt è fatto
apposta per raddrizzare; i modelli che assumono indipendenze irrealistiche la
schiacciano contro $0$ e $1$, e lì la sigmoide aiuta ancora ma ha la forma
sbagliata, e appena ci sono dati l'isotonica fa meglio;
regressione logistica e alberi in bagging sono già calibrati. Sotto le
duecento-mille osservazioni di calibrazione, la sigmoide batte l'isotonica su
tutti e nove i metodi entrati nell'analisi delle curve di apprendimento (gli
alberi di decisione ne restano fuori).

Due garanzie e due limiti. Una trasformazione crescente in senso stretto
lascia invariato ogni ordinamento, quindi l'AUC non cambia; l'isotonica, che è
crescente ma non in senso stretto, può creare pareggi e spostare l'AUC di poco.
E la calibrazione da sola non ordina i modelli: il predittore costante
$\hat{p} \equiv P(y=1)$, con $P(y=1)$ la prevalenza vera, è calibrato per
costruzione e ha AUC $0{,}5$. La
formulazione corretta, dovuta a Gneiting, Balabdaoui e Raftery, è
«massimizzare la finezza sotto vincolo di calibrazione»
{cite}`gneiting2007probabilistic`: la finezza è quanto le previsioni sono
concentrate, cioè quanta poca incertezza dichiarano, ed è una proprietà delle
sole previsioni; la calibrazione è il vincolo che impedisce di ottenerla
mentendo. Sotto quel vincolo la finezza coincide con la dispersione delle
previsioni attorno alla frequenza di base, ma solo sotto quel vincolo: il
modello grezzo di queste righe si allontana moltissimo dalla frequenza di
base, e quello scostamento dice soltanto quanto è sbilanciato. La
{doc}`sezione sulla validazione delle serie temporali
</SerieTemporali/validazione-e-feature>` usa la stessa
coppia sui punteggi propri per le previsioni con banda.

`````

La prova si fa sul modello riequilibrato con i pesi di classe, in tre passi:
guardare i dieci cassetti, costruire la tavola di conversione su dati mai
visti, riguardare i cassetti. Il voto unico, quello che pesa ogni cassetto per
quanto è pieno, nelle tabelle si scrive `ECE`, dall'inglese *expected
calibration error*.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

# la ricetta dei dati sbilanciati, con piu' esempi: la tavola di conversione
# vuole un mazzo suo, che il modello non abbia mai visto
X, y = make_classification(n_samples=30000, n_features=10, n_informative=5,
                           weights=[0.966], flip_y=0.01, random_state=0)
X_tr, X_resto, y_tr, y_resto = train_test_split(X, y, test_size=0.5,
                                                random_state=0)
X_cal, X_te, y_cal, y_te = train_test_split(X_resto, y_resto, test_size=0.5,
                                            random_state=0)

def cassetti(p, M=10):           # M cassetti, a fette uguali della scala
    return np.clip((p * M).astype(int), 0, M - 1)

def ece(p, y, M=10):             # distanza media fra il detto e il vero
    c = cassetti(p, M)
    return sum((c == k).mean() * abs(p[c == k].mean() - y[c == k].mean())
               for k in range(M) if (c == k).any())

def tabella(p, y, titolo):
    c = cassetti(p)
    print(f"{titolo:>12s}   quante    detto     vero")
    for k in range(10):
        m = c == k
        if m.sum():
            print(f"     {k / 10:.1f}-{(k + 1) / 10:.1f}   {m.sum():5d}    "
                  f"{p[m].mean():.3f}    {y[m].mean():.3f}")
    print(f"       totale   {len(p):5d}    {p.mean():.3f}    {y.mean():.3f}")

modello = LogisticRegression(max_iter=1000,
                             class_weight="balanced").fit(X_tr, y_tr)
p_cal = modello.predict_proba(X_cal)[:, 1]
p_te = modello.predict_proba(X_te)[:, 1]
tabella(p_te, y_te, "come esce")

# la tavola di conversione, imparata sul mazzo che il modello non ha visto
al_sicuro = lambda v: np.clip(v, 1e-12, 1 - 1e-12)
logit = lambda v: np.log(al_sicuro(v) / (1 - al_sicuro(v)))
L_cal, L_te = logit(p_cal).reshape(-1, 1), logit(p_te).reshape(-1, 1)
convertite = {nome: LogisticRegression(**opz).fit(L_cal, y_cal)
                                        .predict_proba(L_te)[:, 1]
              for nome, opz in (("Platt", {}),
                                ("temperatura", {"fit_intercept": False}))}
convertite["isotonica"] = (IsotonicRegression(out_of_bounds="clip")
                           .fit(p_cal, y_cal).predict(p_te))
print()
tabella(convertite["Platt"], y_te, "con Platt")

print()
print(f"{'grezzo':12s} ECE {ece(p_te, y_te):.4f}   "
      f"AUC {roc_auc_score(y_te, p_te):.4f}")
for nome, conv in convertite.items():
    print(f"{nome:12s} ECE {ece(conv, y_te):.4f}   "
          f"AUC {roc_auc_score(y_te, conv):.4f}")

# il modello onesto e inutile: risponde sempre la frequenza di base
costante = np.full(len(y_te), y_tr.mean())
print(f"{'costante':12s} ECE {ece(costante, y_te):.4f}   "
      f"AUC {roc_auc_score(y_te, costante):.4f}   "
      f"dice {y_tr.mean():.4f}, vere {y_te.mean():.4f}")

# e lo stesso conto con altre partizioni: il voto si muove, e l'ordine pure
print()
for M in (5, 10, 15, 30):
    print(f"con {M:2d} cassetti   Platt {ece(convertite['Platt'], y_te, M):.4f}"
          f"   isotonica {ece(convertite['isotonica'], y_te, M):.4f}")
```

```text
   come esce   quante    detto     vero
     0.0-0.1    2275    0.051    0.005
     0.1-0.2    1481    0.146    0.006
     0.2-0.3     992    0.248    0.006
     0.3-0.4     687    0.348    0.007
     0.4-0.5     548    0.447    0.011
     0.5-0.6     463    0.548    0.054
     0.6-0.7     359    0.647    0.084
     0.7-0.8     311    0.753    0.138
     0.8-0.9     203    0.850    0.345
     0.9-1.0     181    0.937    0.613
       totale    7500    0.283    0.042

   con Platt   quante    detto     vero
     0.0-0.1    6783    0.014    0.013
     0.1-0.2     357    0.144    0.148
     0.2-0.3     104    0.247    0.317
     0.3-0.4      75    0.345    0.427
     0.4-0.5      67    0.446    0.493
     0.5-0.6      41    0.551    0.634
     0.6-0.7      36    0.654    0.778
     0.7-0.8      25    0.747    0.640
     0.8-0.9      11    0.842    0.636
     0.9-1.0       1    0.928    1.000
       totale    7500    0.040    0.042

grezzo       ECE 0.2409   AUC 0.9064
Platt        ECE 0.0047   AUC 0.9064
temperatura  ECE 0.2221   AUC 0.9064
isotonica    ECE 0.0035   AUC 0.9050
costante     ECE 0.0059   AUC 0.5000   dice 0.0364, vere 0.0423

con  5 cassetti   Platt 0.0036   isotonica 0.0033
con 10 cassetti   Platt 0.0047   isotonica 0.0035
con 15 cassetti   Platt 0.0038   isotonica 0.0040
con 30 cassetti   Platt 0.0063   isotonica 0.0055
```

Nella prima tabella le due colonne non si somigliano in nessuno dei dieci
cassetti, e sbagliano tutte nello stesso verso: dove il modello annuncia il
$44{,}7\%$ le frodi sono l’$1{,}1\%$, dove annuncia il $93{,}7\%$ sono il
$61{,}3\%$. In fondo, sul mazzo intero, promette il $28{,}3\%$ e di frodi ce
n'è il $4{,}2\%$: è il gonfiaggio che i pesi di classe avevano introdotto.

Nella seconda le stesse due colonne si somigliano quasi ovunque, e il totale
scende al $4{,}0\%$ contro un $4{,}2\%$ vero. Va guardata anche la colonna
delle quantità, perché racconta l'altra metà: dopo la conversione i cassetti
alti si svuotano (undici operazioni nel penultimo, una nell'ultimo), e lo
scarto più grosso che resta è proprio nel penultimo, $0{,}842$ contro
$0{,}636$, cioè su undici casi. Il voto scende da $0{,}2409$ a $0{,}0047$ con
la curva liscia e a $0{,}0035$ con la scaletta, e con la curva l'AUC resta
$0{,}9064$, identica, perché una conversione che non scavalca non riordina
niente; la scaletta ne perde un pelo ($0{,}9050$), ed è il prezzo dei pari
merito che introduce.

Fra $0{,}0047$ e $0{,}0035$, però, non si può scegliere, e le ultime righe
stampate dicono perché: rifacendo lo stesso conto con cinque, quindici e trenta
cassetti i due voti si muovono entrambi, e a quindici l'ordine si rovescia
($0{,}0038$ contro $0{,}0040$). A questa distanza dallo zero il numero misura
anche il rumore del conteggio, non solo il modello: sotto una certa soglia le
due tavole di conversione vanno dichiarate pari, ed è la riserva che la
formula si porta dietro.

Due righe vanno lette insieme. La versione a una manopola sola, che si chiama
scaling di temperatura ed è la ricetta standard sulle reti che il libro
incontrerà a partire dal capitolo sulle reti neurali, qui non serve quasi a
niente ($0{,}2221$ contro $0{,}2409$): stringe o allarga la sicurezza, mentre
il guasto era uno spostamento di tutta la scala, e per raddrizzarlo serve la
seconda manopola. E il modello costante, che risponde $0{,}0364$ a chiunque,
prende $0{,}0059$, cioè è calibrato quaranta volte meglio del modello di
partenza, con un'AUC di $0{,}5000$: non distingue niente. Quel $0{,}0059$, poi,
è esattamente $0{,}0423 - 0{,}0364$, la distanza fra quello che dice e quello
che c'è: con un cassetto solo il voto è quella sottrazione e nient'altro. Chi
scegliesse un modello guardando la sola calibrazione sceglierebbe quello.

Le probabilità, adesso, sono numeri di cui fidarsi, e servono a decidere: la
soglia che pareggia i costi si può finalmente calcolare, perché quel conto
presuppone che le probabilità dicano la verità. Resta da
vedere che cosa cambia quando la risposta da prevedere non è un sì o un no ma
un numero.

## Quando il target è un numero: le metriche di regressione

Prima una parola sul nome che compare nel titolo: la risposta da prevedere si
chiama spesso **target** (in inglese: il bersaglio). È la stessa cosa che
finora abbiamo chiamato «etichetta» o «la risposta giusta», e nelle formule
$y$: tre nomi per un oggetto solo, e conviene riconoscerli tutti perché il
libro e le librerie li usano tutti.

Se il modello non classifica ma prevede una quantità continua (il prezzo di
una casa, la temperatura di domani), la matrice di confusione non serve:
contano gli **scarti** tra valore previsto $\hat{y}$ e valore reale $y$.

`````{tab} Elementare

Le misure più usate sono tre, e le sigle sono tutte inglesi. Il **MAE** (*mean
absolute error*, errore medio assoluto) è l'errore medio «in valore assoluto»:
in media, di quanti euro sbagliamo il prezzo. Il **RMSE** (*root mean squared
error*, radice dell'errore quadratico medio) è simile, ma prima di
mediare eleva al quadrato gli errori, e alla fine ne fa la radice quadrata.
Il quadrato serve a
**punire di più i grandi svarioni**: sbagliare una volta di $100$ costa
$100^2 = 10\,000$, mentre sbagliare due volte di $50$ costa
$50^2 + 50^2 = 5\,000$, cioè la metà, benché l'errore totale sia lo stesso. La
radice serve a rimettere il numero nell'unità di
partenza, perché senza di lei avremmo euro al quadrato. Ecco perché MAE e RMSE
si leggono entrambi in euro.

L’**R²** invece è un voto, e si legge «erre quadro». Vale $1$ se la
previsione è perfetta e $0$ se il modello non fa meglio di chi risponde sempre
la media di tutti i valori; e sì, può anche scendere **sotto zero**, se fa
peggio di così. Un esempio: se rispondendo sempre la media si sbaglia in media
di $40\,000$ € al quadrato e il modello scende a $10\,000$, ne ha risparmiati
tre quarti e l'R² vale $0{,}75$. È il vantaggio dell'R² sulle altre due: non è
in euro, quindi si può confrontare fra problemi diversi.

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

La risposta giusta è «30-40». Il modello dice «40-50» e ha sbagliato di poco;
dice «over 70» e ha scambiato una ragazza per sua nonna. Fra una fascia e
l'altra, però, non c'è una distanza scritta da nessuna parte, e si sa solo chi
viene prima e chi dopo.

L'accuratezza non vede niente di tutto questo, e conta i due sbagli come
identici. E c'è di peggio, perché chi sceglie il modello guardando lei può
finire per preferire proprio quello che sbaglia di più. Con cinque fasce
d'età, mettiamo alla prova due modelli. Il primo sbaglia sempre, ma sempre di
una fascia sola, e dice «40-50» quando è «30-40». Il secondo azzecca una volta
su cinque, e nelle altre quattro spara la fascia più lontana che c'è.
Accuratezza zero il primo, $0{,}20$ il secondo, e chi sceglie con
l'accuratezza si porta a casa il secondo. Uno scivola nella fascia accanto,
l'altro scambia i ventenni per gli ottantenni.

Ci sono due modi di rimediare, a seconda di quanto si vuole essere precisi. Il
più semplice conta giusta anche la risposta adiacente, e chi lo usa deve dirlo
in chiaro, perché quel confine a una fascia di distanza lo ha scelto lui. Il
più solido pesa ogni errore per quanto è lontano (il nome da cercare è *kappa
di Cohen pesato*), e allora lo scambio fra due fasce vicine costa molto meno
di quello fra la prima e l'ultima.

Quel voto ha un'abitudine da conoscere prima di fidarsene, e si vede portando
lo stesso modello in due sale d'attesa diverse. Il modello sposta la risposta
di una fascia una volta su cinque, in su o in giù a caso, e quando gli
toccherebbe uscire dalla scala lascia la persona dov'è. Nella prima sala le
tre fasce sono ugualmente affollate, e prende $0{,}90$. Nella seconda
novantotto persone su cento stanno nella prima fascia, e con le stesse
identiche mosse prende $0{,}47$. A cambiare è stata la sala d'attesa. Serve a
confrontare due modelli sulle stesse persone, e non due sistemi che lavorano
su popolazioni diverse.

C'è poi una strada che lascia stare la misura e cambia il problema. Se le
fasce stanno in fila si può predire un numero, gli anni, e poi tagliarlo in
fasce, trattandolo come una regressione. Funziona bene quando le fasce sono
davvero equidistanti, e male quando non lo sono, come fra «lieve» e
«moderato», dove può esserci molta meno distanza che fra «moderato» e «grave».
Il rimedio è non decidere a tavolino dove tagliare. I punti di taglio li
sceglie il modello guardando i dati, e allora può accorgersi che fra due fasce
c'è pochissimo spazio e fra altre due moltissimo.

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
quando le classi si sbilanciano. Prendiamo un modello che sbaglia sempre allo
stesso modo, cioè sposta la risposta di una classe nel $20\%$ dei casi, in su o
in giù a sorte, con l'unica avvertenza che chi sta a un estremo da una parte
non ha dove andare e in quella metà dei casi resta dov'è. Valutato in
aspettativa su tre popolazioni con tre classi, il kappa pesato passa da
$0{,}90$ con classi bilanciate a $0{,}80$ con proporzioni $90/5/5$ a $0{,}47$
con proporzioni $98/1/1$; l'accuratezza, in quegli stessi tre casi, *sale* da
$0{,}867$ a $0{,}895$ a $0{,}899$, perché sbilanciare la popolazione significa
metterci dentro sempre più esempi della classe estrema, che è quella su cui il
modello sbaglia meno.
Va letta come misura interna a un dataset, non come voto trasportabile:
due sistemi valutati su popolazioni con prevalenze diverse non hanno kappa
confrontabili.

Sul fronte del **modello**, la soluzione elegante è la **regressione
ordinale**: invece di $K$ probabilità indipendenti si stima una variabile
latente continua e $K-1$ soglie, e la probabilità cumulata
$P(y \leq k) = \sigma(\tau_k - f(\mathbf{x}))$ è monotona per costruzione, dove
$f(\mathbf{x})$ è la variabile latente stimata, $\tau_1 < \dots < \tau_{K-1}$
sono le soglie apprese (un'altra cosa dalla soglia di decisione $\tau^\star$
dei costi) e $\sigma$ è la sigmoide. Il
vantaggio pratico rispetto alla regressione seguita da arrotondamento è che le
soglie sono **apprese** invece che imposte equidistanti, quindi il modello può
scoprire che fra due classi c'è poco spazio e fra altre due molto.

`````

## Scegliere la metrica giusta

Non esiste "la" metrica migliore: esiste quella allineata al problema. Il
target è una categoria o un numero? Se è una categoria, le classi sono
bilanciate (e l'accuratezza può bastare) o sbilanciate, e allora servono
precision, recall, F1 o AUC? E dei due errori, quale costa di più: un falso
allarme o un caso mancato? La metrica è la definizione stessa di "successo" che
diamo al modello, e la si sceglie prima di addestrarlo.

## Dopo il numero: guardare dove sbaglia

Una metrica riassume in un numero il comportamento del modello su migliaia di
esempi, e per farlo li tratta tutti allo stesso modo. C'è però un ordine fra
quegli esempi che il numero butta via, ed è quello che dice **quanto** il
modello ha sbagliato su ciascuno. Recuperarlo costa due righe e cambia il
mestiere: dal misurare al capire che cosa fare.

`````{tab} Elementare

Chi corregge trenta compiti in classe non si ferma alla media della classe.
Prende i tre andati peggio e li apre, perché lì dentro c'è sempre una di due
cose, e sono tutte e due utili. O la domanda era scritta male, o la griglia di
correzione dava per giusta la risposta sbagliata: allora l'errore è del compito,
non di chi lo ha svolto. Oppure la domanda era giusta e difficile davvero, e
quello è l'argomento su cui tornare la settimana dopo.

Con un modello si fa esattamente questo. Alla fine di ogni esempio di prova c'è
un numero che dice quanto quella risposta è stata sbagliata, e basta metterli in
fila dal peggiore. In cima si trovano le due cose di prima: esempi la cui
etichetta è sbagliata (il modello ha ragione e il dato ha torto) ed esempi
genuinamente difficili, che dicono dove serve altro materiale. E si trova
qualcosa di più: sono pochissimi, e trovarli a mano sarebbe impossibile, mentre
metterli in fila li porta tutti nelle prime posizioni.

`````

`````{tab} Superiore

La quantità è la **perdita per esempio**, cioè il termine che la funzione di
costo somma o media prima di restituire un numero solo. Per un classificatore
probabilistico è la log-loss del singolo campione,
$\ell_i = -\log \hat{p}_i(y_i)$, cioè la sorpresa del modello davanti
all'etichetta vera: cresce senza limite man mano che la probabilità assegnata a
quella classe tende a zero, e per questo pone in cima alla graduatoria gli
esempi su cui il modello ha sbagliato **con convinzione**, che sono i soli
informativi.

L'ordinamento per $\ell_i$ decrescente separa due popolazioni che si
riconoscono aprendo gli esempi:

- **rumore d'etichetta**, dove $y_i$ è sbagliata: il modello ha imparato la
  regola giusta dal resto dei dati e la applica, quindi diverge dall'etichetta
  con alta confidenza. Sono i casi in cui la correzione va fatta sul dataset;
- **difficoltà genuina**, dove $y_i$ è corretta ma l'esempio sta vicino al
  confine o in una regione poco rappresentata. Sono i casi in cui la correzione
  va fatta sul modello o sulla raccolta.

La coda opposta della stessa graduatoria ha un uso suo: gli esempi con $\ell_i$
minima sono quelli su cui il modello è più sicuro, e una $\ell_i$ quasi nulla
su un esempio che *dovrebbe* essere difficile è la firma di una **perdita di
informazione dal futuro**, cioè di una feature che contiene il bersaglio.

Il costo del gesto è nullo: la perdita per esempio è già stata calcolata per
ottenere la metrica, e l'unica cosa che si aggiunge è non sommarla.

`````

### In pratica: le etichette sbagliate salgono in cima

Un modo di misurare la resa del gesto è prendere un dataset pulito, guastarne a
mano una piccola frazione delle etichette, e vedere quante di quelle guaste
finiscono nelle prime posizioni della graduatoria. La frazione guasta è nota,
quindi la domanda ha una risposta esatta.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

X, y_vero = make_classification(n_samples=2000, n_features=12, n_informative=6,
                                flip_y=0, class_sep=1.2, random_state=0)
rng = np.random.default_rng(0)
y = y_vero.copy()
guasti = rng.choice(len(y), size=60, replace=False)   # 3% di etichette sbagliate
y[guasti] = 1 - y[guasti]
sbagliata = np.zeros(len(y), bool)
sbagliata[guasti] = True

Xtr, Xva, ytr, yva, gtr, gva = train_test_split(
    X, y, sbagliata, test_size=0.4, random_state=0)
mod = LogisticRegression(max_iter=1000).fit(Xtr, ytr)

print(f"accuratezza sulla validazione: {accuracy_score(yva, mod.predict(Xva)):.3f}")
print(f"etichette sbagliate nella validazione: {gva.sum()} su {len(gva)}"
      f" ({100 * gva.mean():.1f}%)")

# la perdita di OGNI esempio, invece della loro media: e' tutto il gesto
p = mod.predict_proba(Xva)
perdita = np.array([log_loss([v], [pi], labels=[0, 1]) for v, pi in zip(yva, p)])
ordine = np.argsort(-perdita)

for k in (10, 25, 50):
    print(f"   fra i {k:3d} con la perdita piu' alta: {gva[ordine[:k]].sum():3d}"
          f" sbagliate  ({100 * gva[ordine[:k]].mean():5.1f}%)")
print(f"   in tutta la validazione ({len(gva)}):   {gva.sum():3d} sbagliate"
      f"  ({100 * gva.mean():5.1f}%)")
```

```text
accuratezza sulla validazione: 0.899
etichette sbagliate nella validazione: 22 su 800 (2.8%)
   fra i  10 con la perdita piu' alta:   8 sbagliate  ( 80.0%)
   fra i  25 con la perdita piu' alta:  14 sbagliate  ( 56.0%)
   fra i  50 con la perdita piu' alta:  20 sbagliate  ( 40.0%)
   in tutta la validazione (800):    22 sbagliate  (  2.8%)
```

Le etichette guaste sono il $2{,}8\%$ della validazione, e pescando a caso
sarebbe quella la probabilità di incontrarne una. Fra i dieci esempi con la
perdita più alta sono l’$80\%$: ventotto volte più dense. E il numero che
serve davvero a chi deve decidere quanto tempo spenderci è l'ultimo:
**aprendone cinquanta se ne ritrovano venti su ventidue**, cioè il novanta per
cento delle etichette guaste in un'ora di lavoro invece che ottocento.

L'accuratezza, intanto, dice $0{,}899$ e non dice niente di tutto questo. Non è
un difetto della metrica, che sta facendo il suo mestiere: è che il suo mestiere
finisce dove comincia questo.

## In pratica, con scikit-learn

`scikit-learn` calcola tutte queste metriche in poche righe, a partire dalle
etichette vere e dalle predizioni.

```{code-block} python
:class: pt-non-eseguibile

from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, mean_absolute_error, r2_score)

# --- classificazione ---
# Attenzione all'orientamento: scikit-learn mette la VERITÀ in riga e la
# PREDIZIONE in colonna, ed elenca le etichette in ordine crescente, quindi la
# classe 0 (negativa) per prima. Esce [[VN, FP], [FN, VP]]: la figura di questa
# sezione, che ha VP in alto a sinistra, è quella stessa matrice ruotata.
print(confusion_matrix(y_test, y_pred))
# con labels=[1, 0] l'ordine torna quello della figura, VP in alto a sinistra
print(confusion_matrix(y_test, y_pred, labels=[1, 0]))

# precision, recall e F1 per classe; le righe "macro avg" e "weighted avg"
# sono le due medie, e su più classi la scelta fra loro cambia il verdetto
print(classification_report(y_test, y_pred))

# AUC: richiede le probabilità, non le classi secche
proba = modello.predict_proba(X_test)[:, 1]   # probabilità della classe positiva
print("AUC:", roc_auc_score(y_test, proba))

# --- regressione: altri dati e un altro modello, il target qui è continuo ---
print("MAE:", mean_absolute_error(y_test_reg, y_pred_reg))
print("R2 :", r2_score(y_test_reg, y_pred_reg))
```

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Tutto parte dal contare i **quattro esiti** del rilevatore di fumo: allarme
  giusto, falso allarme, incendio mancato, silenzio giusto. Da quei quattro
  numeri si ricava ogni altra misura.
- La percentuale di risposte giuste (l’**accuratezza**) **inganna** quando una
  risposta è molto più frequente dell'altra: la guardia che dorme sempre prende
  99 su 100 e non ha mai fermato un ladro.
- Due domande diverse: *quando dice sì, quanto spesso ci azzecca?* (la
  **precision**) e *di tutti i casi veri, quanti ne trova?* (la **recall**).
  Alzare l'una abbassa l'altra; la **F1** è un voto unico, alto solo se lo sono
  entrambe. Nello screening medico conta di più trovarli tutti, nell'antispam
  conta di più non cestinare un'email buona.
- Spostando la soglia si cambia il compromesso senza riaddestrare niente; la
  **curva ROC** li mostra tutti insieme e l'area sotto di essa (l’**AUC**) è un
  voto fra $0$ e $1$: $1$ è perfetto, $0{,}5$ è quanto prende chi tira a caso,
  e sotto quel valore il modello sta scambiando le due classi.
- Un «novantaquattro per cento» non si giudica su un caso solo, si giudica sul
  gruppo: si mettono i casi in dieci cassetti secondo quello che il modello ha
  detto e si conta, cassetto per cassetto, quanti lo erano davvero. È la
  **calibrazione**, e si ripara senza riaddestrare, con una tavola di
  conversione imparata su casi mai visti, che non scavalca e quindi lascia la
  graduatoria com'era (la scaletta a gradini può però creare dei pari merito).
  Chi risponde sempre lo stesso numero è calibrato e inutile: onestà e capacità
  di distinguere sono due virtù separate.
- Se la risposta è un numero: **MAE** e **RMSE** dicono di quanto sbagliamo,
  nella stessa unità del target (euro, gradi), e l'RMSE è più severo con i
  grandi svarioni; di questi due si cerca il valore **più basso**. L’**R²**
  invece dice quanto siamo meglio di chi risponde sempre la media, e lì si
  cerca il **più alto**.
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
- L’**accuratezza inganna** con classi sbilanciate: premia chi predice sempre la
  classe maggioritaria.
- **Precision** (pochi falsi allarmi) vs **recall** (pochi casi mancati): la
  $F_1$ le riassume. Privilegia la recall nello screening medico, la precision
  nell'antispam. Con $K$ classi va dichiarata la **media**: la *micro* coincide
  con l'accuratezza, la *macro* è quella che dà voce alle classi rare.
- **AUC**: qualità del classificatore indipendente dalla soglia, in $[0,1]$,
  con $0{,}5$ come punteggio del caso e non come minimo. È
  $P(s^+>s^-) + \tfrac12 P(s^+=s^-)$: il mezzo punto sui pareggi non è
  opzionale.
- Con costi noti la soglia ottimale è $c_{\text{FP}}/(c_{\text{FP}} +
  c_{\text{FN}})$ **se** le decisioni corrette non costano né rendono nulla;
  altrimenti contano le differenze fra costi e guadagni.
- **Calibrazione**: $P(y=1\mid\hat{p}=q)=q$, stimata a intervalli con il
  diagramma di affidabilità e riassunta dall'ECE, che dipende dalla partizione
  e va dichiarato insieme a essa. Si corregge dopo, su dati indipendenti
  (Platt, temperatura, isotonica); una trasformazione crescente in senso
  stretto lascia l'AUC invariata, mentre l'isotonica può creare pari merito e
  spostarla di poco. Da sola non ordina i modelli, perché il predittore
  costante è calibrato con AUC $0{,}5$: il criterio è massimizzare la finezza
  (quanta poca incertezza le previsioni dichiarano) sotto vincolo di
  calibrazione.
- Per la **regressione**: MAE e RMSE nell'unità del target (RMSE punisce di più
  i grandi errori) **si minimizzano**; $R^2$, frazione di varianza spiegata, si
  massimizza, ed è negativo se il modello fa peggio della media.
- Per un target **ordinale**, kappa di Cohen pesato quadraticamente, ricordando
  che dipende dalle marginali e non si confronta fra popolazioni diverse.
- La metrica va scelta *prima*, in base al problema e al costo degli errori.
```

`````
