# Overfitting, bias-varianza e validazione

C'è un modo infallibile per andare male a un esame: imparare a memoria le
soluzioni dei compiti degli anni scorsi. Chi lo fa risponde alla perfezione
alle domande già viste e va nel panico davanti a un esercizio anche solo
leggermente diverso. Ha memorizzato, non capito. Un modello di machine
learning può cadere esattamente nella stessa trappola, e distinguere la
memoria dalla comprensione è, in fondo, il problema centrale di tutta la
disciplina.

Nelle sezioni precedenti abbiamo detto che la posta in gioco non è riprodurre
gli esempi già visti, ma **generalizzare** a input nuovi. Un modello che azzecca
ogni risposta sui dati di addestramento e sbaglia su quelli veri non ha imparato
nulla di utile. Questa sezione parla di come accorgersene e come porvi rimedio.

## Imparare o memorizzare: overfitting e underfitting

Ci sono due modi opposti di sbagliare, e conviene tenerli davanti agli occhi
insieme ({numref}`fig-overfitting`).

```{figure} ../figures/overfitting-tre-fit.svg
:name: fig-overfitting
:alt: Tre pannelli con la stessa nube di punti a forma di collina. A sinistra una retta quasi orizzontale la ignora (underfitting); al centro una curva morbida la segue bene (buon fit); a destra una curva contorta passa per ogni punto oscillando in mezzo (overfitting).
:width: 95%

Lo stesso insieme di punti, tre modelli. Il modello troppo semplice non coglie
l'andamento; quello troppo flessibile lo ricalca fin dentro il rumore. In mezzo,
il buon compromesso.
```

`````{tab} Elementare

Da una parte c'è chi generalizza troppo poco: un modello **troppo semplice**,
come una retta costretta a descrivere dei dati chiaramente curvi. Sbaglia già
sugli esempi che ha visto in addestramento, figurarsi sui nuovi. Si chiama
**underfitting**: il modello è troppo rigido per il problema.

Dall'altra parte c'è chi generalizza troppo: un modello **troppo flessibile**
che si contorce per passare esattamente su ogni singolo punto, rumore compreso.
Sul foglio d'esame degli esempi visti prende dieci e lode, ma ha imparato anche
gli errori di misura, gli accidenti, il caso. Su un dato nuovo crolla. Si chiama
**overfitting**: il modello ha memorizzato invece di capire.

`````

`````{tab} Superiore

Formalmente, l'errore che ci interessa è quello su dati **non** visti in
addestramento, l’*errore di generalizzazione*. Confrontarlo con l'errore
sull'insieme di training rivela il regime in cui ci troviamo:

- **Underfitting**: errore di training *alto* e vicino a quello di test. Il
  modello è troppo poco espressivo: non riesce a catturare la struttura dei
  dati (*bias* alto).
- **Overfitting**: errore di training *molto basso* ma errore di test *alto*.
  Il modello ha capacità in eccesso e adatta $f_\theta$ anche alle
  fluttuazioni casuali del campione (*varianza* alta).

Il divario tra i due errori, $\text{err}_{\text{test}} -
\text{err}_{\text{train}}$, è la spia dell'overfitting: quando si allarga,
il modello sta memorizzando.

`````

## Il compromesso bias-varianza

Underfitting e overfitting sono le due facce di un'unica tensione, che ha un nome
classico: il **compromesso bias-varianza** (*bias-variance tradeoff*).

Il modo classico di raccontarla è un bersaglio da tiro a segno, ma prima di
guardarlo serve sapere che cosa sia un colpo, perché il
modello è uno solo e i fori sul bersaglio sono tanti. Il gioco è questo:
immagina di rifare l'esperimento da capo molte volte, ogni volta raccogliendo
un campione di dati nuovo e riaddestrando il modello su quello. **Ogni foro sul
bersaglio è un addestramento**, e il centro del bersaglio è la risposta giusta.
Un modello può sbagliare in due modi indipendenti: perché il gruppo dei fori è
tutto spostato da una parte (sbaglia sempre nello stesso verso) oppure perché è
sparpagliato (cambia idea a ogni campione). Il primo difetto si chiama
**bias**, il secondo **varianza**.

```{figure} ../figures/bias-varianza.svg
:name: fig-bias-varianza
:alt: "Quattro bersagli disposti in una griglia due per due, con le colonne per varianza bassa e alta e le righe per bias basso e alto. Con bias e varianza bassi i colpi sono raccolti al centro; con varianza alta e bias basso sono sparsi ma centrati in media; con bias alto e varianza bassa sono raccolti ma spostati dal centro; con entrambi alti sono sparsi e spostati."
:width: 78%

I quattro casi sul bersaglio, con un foro per ogni addestramento. Il bias è di
quanto si è spostato il gruppo dei
colpi; la varianza è quanto il gruppo è largo. Sono due difetti diversi e si
correggono in modi opposti.
```

Il bersaglio in basso a sinistra di {numref}`fig-bias-varianza`, colpi
raccolti ma tutti fuori centro, è il più insidioso: un modello del genere è
molto *coerente*, dà quasi sempre la stessa risposta, e la coerenza si scambia
facilmente per affidabilità. Raccogliere altri dati serve a stringere il gruppo
dei fori, non a spostarlo: qui il problema non è che il gruppo sia largo, è che
è centrato nel punto sbagliato, e altri dati non lo aggiustano.

`````{tab} Elementare

Immagina di ripetere l'esperimento: raccogli molte volte un nuovo campione di
dati e riaddestri il modello ogni volta.

- Un modello **rigido** (la retta) darà sempre più o meno la stessa risposta,
  campione dopo campione: è *stabile* ma *sistematicamente storto*, perché la
  forma giusta non è una retta. Questo errore sistematico è il **bias** (si
  pronuncia *bàias*, e in inglese vuol dire proprio «inclinazione», la
  tendenza a pendere sempre dalla stessa parte).
- Un modello **flessibile** (la curva contorta) cambierà parecchio a ogni nuovo
  campione, inseguendo il rumore di turno. È *senza pregiudizi* sulla forma, ma
  *instabile*: questa irrequietezza è la **varianza**.

Semplice = molto bias, poca varianza. Complesso = poco bias, molta varianza. Il
bravo modellista cerca il punto di mezzo.

`````

`````{tab} Superiore

Per un target $y = f(x) + \varepsilon$, con rumore a media nulla
($\mathbb{E}[\varepsilon]=0$), varianza $\sigma^2$ e indipendente dal campione
di addestramento, l'errore quadratico atteso di una previsione $\hat{f}(x)$, a
$x$ fissato, mediato sui possibili insiemi di addestramento e sul rumore del
punto di test, si decompone in tre termini (sono proprio quelle ipotesi a far
sparire i doppi prodotti):

$$
\mathbb{E}\big[(y-\hat{f}(x))^2\big]
= \underbrace{\big(\mathbb{E}[\hat{f}(x)]-f(x)\big)^2}_{\text{Bias}^2}
+ \underbrace{\mathbb{E}\big[(\hat{f}(x)-\mathbb{E}[\hat{f}(x)])^2\big]}_{\text{Varianza}}
+ \underbrace{\sigma^2}_{\text{irriducibile}} .
$$

Il **bias** misura quanto la previsione media si scosta dalla verità $f(x)$; la
**varianza** quanto $\hat{f}(x)$ oscilla al variare del campione; $\sigma^2$ è il
rumore intrinseco, che nessun modello può eliminare. Aumentando la complessità il
bias cala ma la varianza cresce: l'errore di test ha la classica forma a **U**, e
il minimo è il modello ottimale.

Un'avvertenza sull'ambito di validità, perché il vocabolario viaggia più
lontano del teorema. La decomposizione è un’**identità della loss quadratica**:
per la loss 0-1 dei classificatori non esiste una scomposizione additiva
analoga, e più varianza può perfino *ridurre* l'errore quando il bias sta dalla
parte sbagliata della soglia {cite}`wood2023unified`. Da qui in avanti «bias» e
«varianza» restano utilissimi come vocabolario anche parlando di alberi e di
foreste; non come aritmetica.

`````

Il compromesso si può disegnare, e conviene tenere in mente il disegno perché
tornerà più volte. Mettiamo su un asse orizzontale la **complessità** del
modello, da sinistra (rigidissimo: una retta) a destra (flessibilissimo: la
curva che si contorce), e sull'asse verticale l'errore che il modello commette
sui **dati nuovi**. Da sinistra l'errore scende, perché il modello è troppo
rozzo e ogni pezzetto di flessibilità in più lo aiuta; a destra risale, perché
il modello comincia a imparare a memoria. In mezzo c'è un punto più basso di
tutti. La curva, insomma, ha la forma di una **U**, e il fondo della U è il
modello che conviene scegliere. (In fondo a questa pagina, parlando di *doppia
discesa*, vedremo in quali casi la storia non finisca lì.)

### Distinguerli in pratica: le curve di apprendimento

Bias e varianza, finora, sono una spiegazione. C'è un modo di **misurarli**, e
risponde alla domanda che costa di più in un progetto vero: *conviene
raccogliere altri dati, o cambiare modello?*

Si disegnano di nuovo delle curve, ma stavolta sono due e l'asse orizzontale
cambia: non più la
complessità del modello, come nella U di poco fa, bensì **la quantità di dati
usata**, da pochi esempi a tutti quelli che abbiamo. Le due curve sono
l'errore sugli esempi con cui il modello ha
studiato (l'addestramento) e l'errore su esempi tenuti da parte per giudicarlo
(la **validazione**: più avanti in questa pagina si vede come si mettono da
parte e perché sia essenziale farlo). Guardandole scendere si capisce
quale dei due mali si ha davanti.

- Le due curve **si avvicinano e si fermano in alto**: il modello sbaglia
  tanto sui dati che ha visto quanto su quelli che non ha visto, ed è già al
  suo limite. È **bias**. Altri dati non servono a niente: serve un modello
  capace di piegarsi a forme più complicate.
- Fra le due resta un **divario largo**, e quella di validazione sta ancora
  scendendo: il modello ha imparato bene ciò che ha visto e generalizza meno. È
  **varianza**, e qui altri dati aiutano davvero.

Sono poche righe di scikit-learn, e vale la pena eseguirle perché il verdetto
è netto.

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve

rng = np.random.default_rng(0)
n = 3000
X = rng.normal(size=(n, 6))
y = np.sin(2 * X[:, 0]) + X[:, 1] ** 2 - X[:, 2] + rng.normal(0, 0.3, n)  # non lineare

# i due estremi del campo di gioco, senza i quali "alto" e "basso" non dicono
# niente: l'errore di chi risponde sempre la media, e il pavimento del rumore
print(f"rispondere sempre la media: {y.var():.3f}")
print(f"pavimento del rumore:       {0.3 ** 2:.3f}")

taglie = np.linspace(0.05, 1.0, 8)
for nome, modello in [("lineare (troppo semplice)", LinearRegression()),
                      ("foresta (abbastanza ricca)",
                       RandomForestRegressor(n_estimators=120, random_state=0))]:
    # shuffle=True mescola le righe prima di ritagliare i sottoinsiemi: senza,
    # il seme non farebbe nulla (learning_curve lo usa solo se si mescola)
    m, tr, va = learning_curve(modello, X, y, train_sizes=taglie, cv=5,
                               scoring="neg_mean_squared_error",
                               shuffle=True, random_state=0)
    tr, va = -tr.mean(1), -va.mean(1)
    print(f"\n{nome}")
    print(f"  con {m[0]:>4} esempi: train {tr[0]:.3f}  validazione {va[0]:.3f}"
          f"  divario {va[0]-tr[0]:+.3f}")
    print(f"  con {m[-1]:>4} esempi: train {tr[-1]:.3f}  validazione {va[-1]:.3f}"
          f"  divario {va[-1]-tr[-1]:+.3f}")
```

Una parola su che cosa sono questi numeri. Qui gli $y$ non sono euro né gradi:
sono numeri puri, fabbricati apposta dalla riga `y = ...`. E l'errore si misura
come per la retta di best fit, cioè scarto fra vero e previsto, elevato al
quadrato e mediato. Un errore di $2{,}3$ vuol dire che, in media, il quadrato
dello scarto vale $2{,}3$: da solo non dice niente, e infatti le prime due
righe del programma servono a costruire il metro.

Il primo estremo del metro è **quanto sbaglia chi non ci prova nemmeno**, cioè
chi risponde sempre la media di tutti gli $y$: su questi dati $3{,}471$. (È la
varianza di $y$, e attenzione, non c'entra con la varianza del modello di poco
fa: qui è semplicemente quanto i valori di $y$ sono sparpagliati attorno alla
loro media.) Il secondo estremo è **quanto sbaglia chi sa tutto**. Non è zero:
la riga che fabbrica $y$ ci aggiunge un disturbo casuale di ampiezza $0{,}3$,
che nessun modello può indovinare perché non dipende da niente, e siccome
l'errore si misura al quadrato quel disturbo costa $0{,}3^2 = 0{,}09$. Fra
$3{,}471$ e $0{,}09$ si gioca tutta la partita: la strada da percorrere è lunga
$3{,}471 - 0{,}09 = 3{,}38$.

Il modello lineare, passando da 120 a 2400 esempi, chiude il divario da
$+0{,}230$ a $+0{,}013$: le due curve si sono **toccate**. Ma si sono toccate a
$2{,}3$, e $2{,}3$ è ancora quasi in cima: dai $3{,}471$ di partenza sono
scesi appena $1{,}17$ su $3{,}38$, cioè un terzo della strada. (Metà strada
sarebbe stata $0{,}09 + 3{,}38/2 = 1{,}78$, parecchio più in basso.)
L'errore di addestramento, per giunta, non è migliorato di un'unghia, anzi è
salito appena ($2{,}294$ con 120 esempi, $2{,}321$ con 2400). Il rialzo è
normale, ed è il segno che stiamo cercando: con pochi esempi una retta riesce a
passare un po’ più vicino a tutti; con tanti non ce la fa più, perché la forma
giusta non è una retta e i punti in più non fanno che ricordarglielo. Quel
modello ha dato tutto quello che aveva, e altri diecimila esempi non
sposterebbero nulla. Se serve di meglio, serve un modello diverso.

La foresta (una **foresta casuale**, un modello fatto di tanti alberi di
decisione che votano: la incontreremo nella sezione sugli alberi, e qui basta
sapere che è molto più flessibile di una retta) arriva a $0{,}031$
sull'addestramento e
$0{,}215$ in validazione, con un divario di $+0{,}184$ ancora aperto: ha
imparato benissimo ciò che ha visto e generalizza un po’ meno. Ma il suo
$0{,}215$, sul metro di prima, è a un passo dal traguardo: della strada da
$3{,}471$ a $0{,}09$ ne ha percorso il $96\%$. Diagnosi opposta e ricetta
opposta: qui i dati in più pagano.

Il valore di questa diagnostica è che si fa **prima** di spendere. Raccogliere
o etichettare dati è la voce più cara di quasi ogni progetto, e queste due
curve dicono in un pomeriggio se quella spesa avrà un effetto.

```{admonition} Due curve diverse con lo stesso nome
:class: note
Attenzione a non confonderle con le curve che si guardano durante
l'addestramento di una rete, dove sull'asse orizzontale ci sono le **epoche**
(un'epoca è una passata completa su tutti gli esempi: si addestra facendone
molte di seguito): quelle diagnosticano l'andamento di *quella* sessione (passi
della discesa del gradiente troppo lunghi o troppo corti, overfitting che
comincia, quando fermarsi) e sono trattate nel capitolo su
PyTorch. Qui l'asse orizzontale è la **quantità di dati**, e la domanda è
diversa: non «come sta andando questo addestramento» ma «questo modello, con
più dati, andrebbe meglio».
```

## Train, validation e test: perché il test non si tocca

Per accorgersi dell'overfitting bisogna misurare l'errore su dati che il modello
**non ha usato** per imparare. Da qui la regola d'oro: si divide il mucchio
degli esempi in tre parti, ciascuna con un compito distinto. Sono, nell'ordine,
**studio, prove ed esame**.

- **Training set** (lo studio): i dati su cui il modello impara i suoi parametri
  (i numeri interni, la $\theta$ dell'apertura del capitolo). È la fetta più
  grossa.
- **Validation set** (le prove): i dati su cui si scelgono gli *iperparametri*,
  cioè le
  scelte di contorno che non si imparano dai dati: quanto complesso può essere
  il modello, quanto forte il freno alla memorizzazione che vedremo tra poco.
- **Test set** (l'esame): i dati che si guardano **una sola volta**, alla fine,
  per stimare onestamente le prestazioni nel mondo reale.

Quanto grandi? Non c'è una regola sacra: proporzioni tipiche sono $60/20/20$ o
$80/10/10$, cioè in ogni caso la maggior parte degli esempi allo studio.

`````{tab} Elementare

Il test set è il compito d'esame vero. Se lo sbirci mentre studi e aggiusti le
tue scelte in base a quello, il voto finale non dice più nulla: hai imparato a
memoria *quell’* esame. Per questo il test si tiene chiuso in un cassetto e si
apre soltanto alla fine. Ogni volta che usi il test per decidere qualcosa, lo
"consumi", e il numero che ti restituisce diventa troppo ottimista.

`````

`````{tab} Superiore

Usare il test per selezionare modelli introduce una forma sottile di *data
leakage*: si finisce per fare overfitting sul test stesso, e la stima
dell'errore di generalizzazione diventa distorta verso il basso. Il validation
set esiste proprio per assorbire tutte le decisioni intermedie e preservare
l'imparzialità del test. La
selezione dei modelli avviene su training + validation; il test resta un
osservatore neutrale che entra in scena solo a giochi fatti.

Due precisazioni operative che fanno la differenza fra una stima onesta e una
che sembra tale. La prima riguarda **come** si divide: il taglio puramente
casuale è affidabile solo se il dataset è grande, e su dataset piccoli o con
categorie rare produce insiemi che non si somigliano. Il rimedio è il
**campionamento stratificato**, che preserva in ciascuna parte le proporzioni
della variabile che conta (la classe da predire, o una covariata importante):
è il `stratify=` di `train_test_split` e la `StratifiedKFold` della sezione
seguente. Nel caso estremo, un test set che non contiene un solo esempio della
classe rara non misura la cosa che interessa.

La seconda è che il test si sporca anche **soltanto guardandolo**. È il *data
snooping bias*: se si ispeziona il test per decidere quali feature costruire,
quale trasformazione applicare o quale famiglia di modelli provare, quelle
decisioni sono state prese sui dati d'esame, e il numero finale è ottimista
anche se il modello non li ha mai visti in addestramento. La disciplina
corretta è mettere da parte il test **come primo gesto**, prima ancora
dell'analisi esplorativa.

`````

C'è però una perdita d'informazione più insidiosa di tutte, perché non passa
dal modello: passa dai **preparativi**.

```{figure} ../figures/train-test-split-scaling-outlier.svg
:name: fig-split-e-scaler
:alt: "Il dataset viene diviso in una parte di training e una di test. Lo scaler viene tarato soltanto sulla parte di training, calcolandone media e deviazione standard, e poi applicato a entrambe le parti. Una freccia barrata segnala l'errore da evitare: tarare lo scaler sull'intero dataset prima della divisione."
:width: 96%

La freccia barrata è l'errore che non si vede. Se il calcolo che riscala i
numeri guarda anche il test per farsi la sua media, un pezzo di informazione
del test è già entrato nell'addestramento.
```

Quasi mai i dati si danno al modello così come sono. Prima si sistemano, e la
prima cosa che si sistema sono le scale. Nella tabella delle case i metri
quadri stanno attorno a $100$ e le stanze attorno a $3$: chiunque misuri
distanze fra esempi, o penalizzi pesi grandi, sta di fatto ascoltando quasi
solo i metri quadri, non perché contino di più ma perché i loro numeri sono più
grossi. Il rimedio è riportare tutte le colonne su una scala comune (lo
strumento che lo fa si chiama *scaler*, e per farlo deve calcolare, di ogni
colonna, il valore medio e quanto i valori se ne discostano di solito).
Poi si riempiono le
caselle vuote con un valore plausibile, per esempio la media della colonna (si
chiama **imputazione**); si scartano le colonne che non servono. Tutte queste
operazioni **imparano qualcosa dai dati**, e qui sta la trappola: se lo
imparano guardando anche il test, allora il test ha già parlato.

È la forma più insidiosa di **data leakage** (una «fuga» di informazione dal
test verso l'addestramento) perché non produce nessun errore e nessun avviso:
produce solo un punteggio un po’ più alto del vero. La regola pratica che ne
discende è secca: qualunque cosa impari dai dati va calcolata **dentro** il
training e poi applicata al resto, mai prima della divisione.

## La cross-validation

Mettere da parte un validation set fisso ha un difetto: con pochi dati, la stima
dipende troppo da *quali* esempi sono finiti nel validation. La
**k-fold cross-validation** aggira il problema riutilizzando i dati con
intelligenza.

Si badi bene: qui il test, quello dell'esame, resta chiuso nel cassetto dove
l'abbiamo messo. Quello che si divide in blocchi è **soltanto la parte di
studio**, e ciò che ruota è il blocco delle prove.

```{figure} ../figures/cross-validation-il-test-che-non-bara.svg
:name: fig-cross-validation
:alt: "Cinque righe, una per giro. In ciascuna, i dati di addestramento sono divisi in cinque blocchi: uno fa da validazione e gli altri quattro da training, e il blocco di validazione scorre di una posizione a ogni riga, dal primo al quinto. A destra di ogni riga il punteggio ottenuto in quel giro. In fondo, il risultato è la media dei cinque punteggi con la loro deviazione standard."
:width: 92%

Il blocco di validazione ruota. Alla fine ogni esempio ha fatto da giudice
esattamente una volta, e il risultato non è un numero ma un numero con la sua
variabilità.
```

`````{tab} Elementare

Dividi i dati di addestramento in $k$ blocchi uguali (di solito $k=5$ o $10$).
A turno, ogni blocco fa da giudice mentre gli altri $k-1$ addestrano il
modello. Ottieni così $k$ misure di errore, ognuna su un pezzo diverso di dati,
e ne fai la **media**. È come fare cinque compiti in classe su cinque parti
diverse del programma invece di giocarsi tutto su una sola interrogazione: il
giudizio finale è più affidabile e meno soggetto al caso.

`````

La riga finale di {numref}`fig-cross-validation` è la parte che si tende a
buttare via: non la media dei cinque punteggi, ma la loro **dispersione**, cioè
di quanto i cinque giri si discostano dalla media. In statistica la si riassume
in un numero, la *deviazione standard*: quanto, in media, un giro si scosta dal
risultato medio.

Serve a non prendere per differenze quelle che sono oscillazioni. Poniamo che
un modello faccia $0{,}81$ e un altro $0{,}83$, e che i cinque giri di ciascuno
ballino di $\pm 0{,}05$: quei due centesimi di scarto sono più piccoli del
ballo, e a rifare la divisione in blocchi la classifica potrebbe benissimo
capovolgersi. Fra quei due modelli, semplicemente, la cross-validation non sa
scegliere, e dire il contrario è dare un significato al caso.

`````{tab} Superiore

Partizionato il training in $k$ fold $D_1,\dots,D_k$, per ogni $i$ si addestra su
$D\setminus D_i$ e si valuta su $D_i$. La stima cross-validata è la media degli
errori di validazione:

$$
\text{CV}_{k} = \frac{1}{k}\sum_{i=1}^{k}
\mathcal{L}\big(f_\theta^{(-i)},\, D_i\big),
$$

dove $f_\theta^{(-i)}$ è il modello addestrato escludendo il fold $i$-esimo. Il
caso estremo $k=m$ (un fold per esempio) è la *leave-one-out*: quasi non
distorta ma costosa. Valori $k=5$ o $k=10$ offrono il miglior compromesso tra
costo computazionale e stabilità della stima.

Un'ipotesi va però dichiarata, perché è quella che regge tutto il
ragionamento: $\text{CV}_k$ stima l'errore **a patto che le righe siano
scambiabili**, cioè che partizionarle a caso produca fold indipendenti fra
loro. Se più righe descrivono lo **stesso soggetto** (più visite dello stesso
paziente, più eventi dello stesso utente, più fotogrammi dello stesso video),
il rimescolamento mette quasi-duplicati sia in training sia in validation, e il
modello ritrova in validation ciò che ha già visto. Il risultato non è una
stima un po’ ottimista: è una stima priva di significato, e non dà nessun
segnale d'allarme. Con duecento soggetti, dieci misure quasi identiche
ciascuno e un'etichetta assegnata **a caso** (quindi non c'è niente da
imparare, e la verità è $0{,}50$), la 5-fold mescolata riporta accuratezza
$1{,}000$; raggruppando per soggetto torna attorno a $0{,}5$, dove deve stare.
In questi casi i fold vanno costruiti per soggetto (`GroupKFold`,
`GroupShuffleSplit`); se invece le righe sono ordinate nel tempo vale il
discorso della sezione sui dati che cambiano, cioè `TimeSeriesSplit` e non un
rimescolamento.

`````

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge

# il test resta da parte fin dall'inizio, non lo tocchiamo più
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

modello = Ridge(alpha=1.0)              # alpha = quanto frena il modello (v. sotto)
scores = cross_val_score(modello, X_train, y_train, cv=5,
                         scoring="neg_mean_squared_error")  # 5-fold CV
print(-scores.mean())                  # errore medio di validazione
```

## Mettere un freno: la regolarizzazione

Un modo diretto per contrastare l'overfitting è impedire al modello di diventare
troppo "estremo". Il trucco è furbo, e per capirlo conviene ricordare che
addestrare vuol dire **rendere più piccolo possibile un numero**, quello che
misura quanto il modello sbaglia: girare le manopole finché quel numero scende.

La **regolarizzazione** cambia le carte in tavola aggiungendo a quel numero un
secondo addendo, una **penalità** che cresce con la grandezza dei pesi (i
numeri per cui il modello moltiplica ogni caratteristica). Da quel momento il
modello non sta più minimizzando soltanto l'errore, sta minimizzando
«errore più spesa in pesi»: alzare un peso continua a convenire se fa scendere
l'errore *più* di quanto fa salire la spesa, e smette di convenire quando serve
solo a rincorrere un punto isolato. Quanto sia caro quel prezzo lo decidiamo
noi, ed è una manopola che nelle
formule si chiama $\lambda$ (la lettera greca *lambda*): $\lambda$ a zero vuol
dire nessun freno, $\lambda$ grande vuol dire freno tirato.

Restano da scegliere le unità della spesa, cioè come si conta quanto è «grande»
un peso, e i due modi classici hanno nomi che sembrano codici da magazzino,
**L1** e **L2**. Vogliono dire poco più della cifra che portano: L1 somma i
pesi elevati alla prima (in valore assoluto), L2 li somma elevati al quadrato.
Sembra un dettaglio contabile e non lo è.

Quello che di solito si impara come una regola da mandare a memoria («la L1
azzera i pesi inutili, la L2 no») è in realtà una questione di forme, e si può
disegnare.

Immagina un piano con due soli pesi, $w_1$ e $w_2$, uno per asse. È lo stesso
gesto della collina nella nebbia: gli assi non portano più i dati, portano le
manopole del modello, e **ogni punto del piano è una scelta possibile dei due
numeri**. Dire al modello «non spendere
più di tanto in pesi» significa allora recintare una regione attorno
all'origine e obbligarlo a restare dentro. Se la spesa si conta sommando i
**valori assoluti** (la L1), il recinto è un rombo con le punte sugli assi: per
star dentro basta che $|w_1| + |w_2|$ non superi il budget, e i due estremi
sono spendere tutto su un peso solo, che sono appunto le punte. Se si conta
sommando i **quadrati** (la L2) il recinto è un cerchio.

E l'errore? Fuori dal recinto l'errore ha la forma di una **conca**, con il
punto più basso dove starebbe la soluzione senza freni. Disegniamo su questa
conca le linee che uniscono i punti di pari errore, le stesse curve di livello
di una carta topografica: sono anelli che si stringono attorno al fondo. Ora,
il modello vorrebbe scendere il più possibile, ma non può uscire dal recinto:
il meglio che può fare è fermarsi sul punto del recinto che sta più in basso, e
quel punto è dove il primo anello che si allarga dal fondo tocca il bordo.

Ed è qui che la forma decide.

```{figure} ../figures/regolarizzazione-l1-l2.svg
:name: fig-l1-l2
:alt: "Due piani con i pesi w1 e w2 sugli assi. A sinistra la L1: la regione ammessa è un rombo con i vertici sugli assi, e le curve di livello dell'errore lo toccano proprio in un vertice, dove w1 è esattamente zero. A destra la L2: la regione è un cerchio, e il punto di contatto cade in una posizione qualsiasi del bordo, dove entrambi i pesi sono piccoli ma nessuno è zero."
:width: 96%

La differenza sta negli spigoli. A sinistra il recinto della L1, un rombo con
le punte sugli assi, e gli anelli dell'errore che lo toccano proprio in una
punta: lì un peso è esattamente zero. A destra il cerchio della L2, che di
punte non ne ha e non privilegia nessuna direzione.
```

Un anello che si allarga incontra un rombo quasi
sempre in una **punta**, come mostra {numref}`fig-l1-l2`, e le punte del rombo
stanno sugli assi, cioè in punti
dove uno dei due pesi vale esattamente zero. Un cerchio invece non ha punte, e
il primo contatto cade in un posto qualunque del bordo, dove entrambi i pesi
sono piccoli ma nessuno è nullo. Ecco perché sommare i valori assoluti seleziona
le caratteristiche e sommare i quadrati no: non è una proprietà nascosta della
formula, è la forma del recinto.

`````{tab} Elementare

Pensa alla regolarizzazione come a un budget di spesa sui pesi del modello.
Senza limiti, per passare su ogni punto il modello gonfia i suoi pesi
a dismisura: è così che nasce la curva contorta. Imponendo un tetto alla spesa
totale, lo costringi a essere sobrio, e le curve sobrie sono più morbide,
quindi generalizzano meglio. Due modi di contare la spesa:

- **Ridge (L2)**: penalizza la *somma dei quadrati* dei pesi. Li rimpicciolisce
  tutti dolcemente, senza azzerarne nessuno. (Il quadrato punisce pochissimo
  chi è già piccolo: portare un peso da $0{,}1$ a $0$ fa risparmiare appena
  $0{,}01$, e quel risparmio non vale mai la pena.)
- **Lasso (L1)**: penalizza la *somma dei valori assoluti*. Qui l'ultimo
  centesimo costa quanto il primo, quindi conviene sempre azzerare del tutto un
  peso che non serve: tende a spingere a **zero netto** i pesi delle
  caratteristiche inutili, che è come cancellarle dalla tabella. È lo stesso
  fatto che il rombo con le punte sugli assi racconta con la geometria.

`````

`````{tab} Superiore

Si aggiunge alla loss $\mathcal{L}(\theta)$ un termine di penalità pesato da un
iperparametro $\lambda \ge 0$ che regola l'intensità del freno. Per la
regressione **Ridge** (norma $\ell_2$):

$$
\mathcal{L}_{\text{Ridge}}(\theta)
= \frac{1}{m}\sum_{i=1}^{m}\big(\hat{y}^{(i)}-y^{(i)}\big)^2
+ \lambda\sum_{j=1}^{n}\theta_j^{2},
$$

per il **Lasso** (norma $\ell_1$):

$$
\mathcal{L}_{\text{Lasso}}(\theta)
= \frac{1}{m}\sum_{i=1}^{m}\big(\hat{y}^{(i)}-y^{(i)}\big)^2
+ \lambda\sum_{j=1}^{n}|\theta_j| .
$$

Con $\lambda \to 0$ si torna al modello non regolarizzato (varianza alta); con
$\lambda$ grande i pesi sono schiacciati verso zero (bias alto): $\lambda$ è
la manopola del compromesso bias-varianza, e la si sceglie per
cross-validation. La geometria spigolosa della norma $\ell_1$ è ciò che rende
*sparse* le soluzioni del Lasso, annullando interi coefficienti: un selettore
automatico di feature.

Due cose le formule le dicono in silenzio, e vale la pena dirle ad alta voce.
La prima è che l'indice $j$ corre da $1$ a $n$, cioè sulle sole
caratteristiche: l’**intercetta non è penalizzata**. Se lo fosse, il modello
dipenderebbe dall'origine scelta per $y$, e sommare mille a tutte le etichette
(misurare in gradi Kelvin invece che in Celsius) cambierebbe la soluzione, il
che non ha senso. La seconda è che la penalità mette sullo stesso piano pesi
che vivono su scale diverse, e quindi **presuppone feature standardizzate**: il
peso che moltiplica un reddito in euro è piccolo per forza, e la penalità lo
lascerebbe in pace mentre schiaccia quello di una percentuale. Vale qui la
stessa avvertenza che il libro dà per il k-NN e per le SVM, con la differenza
che qui è meno visibile, perché un modello mal regolarizzato funziona
comunque, solo peggio: `Ridge` e `Lasso` non standardizzano da soli, e vanno
messi in una pipeline dietro uno `StandardScaler`.

L’**Elastic Net** somma le due penalità,
$\lambda\big(\alpha\sum_j|\theta_j| + \tfrac{1-\alpha}{2}\sum_j\theta_j^2\big)$,
e non è un compromesso pigro: rimedia a un difetto preciso del Lasso. Fra due
feature fortemente correlate il Lasso ne tiene **una sola**, scelta in modo
instabile (basta cambiare il campione perché scelga l'altra), mentre il termine
$\ell_2$ le fa entrare o uscire **insieme**, il cosiddetto *grouping effect*.
Con feature molte e correlate, che è il caso normale sui dati reali, è la
scelta di partenza più sensata delle due pure.

Un avvertimento sulla lettera $\alpha$, che in questa sezione fa due mestieri
diversi. Nella formula dell'Elastic Net è il **rapporto di miscela** fra le due
penalità ($\alpha = 1$ è Lasso puro, $\alpha = 0$ è Ridge puro) e non ha niente
a che vedere con la loro intensità, che resta $\lambda$. Nel codice, invece,
l'argomento `alpha` di `Ridge`, `Lasso` ed `ElasticNet` è proprio
l’**intensità**, cioè il nostro $\lambda$, mentre la miscela lì si chiama
`l1_ratio`. Sono due tradizioni che si sono incrociate su una lettera sola:
conviene guardare che cosa fa il parametro, non come si chiama.

`````

## Il rasoio di Occam

Sotto tutto questo c'è un principio antico. Nel XIV secolo il frate francescano
**Guglielmo di Occam** enunciò quello che oggi chiamiamo il *rasoio*, riassunto
poi nella formula *entia non sunt multiplicanda praeter necessitatem*, non
moltiplicare le entità oltre il necessario (la frase esatta, per la cronaca,
non compare nei suoi scritti: la coniò un commentatore del Seicento). Tradotto
per noi: **a parità di capacità di spiegare i dati, scegli
il modello più semplice**.

La regolarizzazione non è altro che il rasoio di Occam scritto in formule: la
manopola $\lambda$ è il prezzo che facciamo pagare alla complessità, così che il
modello la compri solo quando serve davvero. La curva morbida del pannello
centrale vince non perché sia la più elaborata, ma perché è la più semplice tra
quelle che rendono conto dei dati. La semplicità, in machine learning, non è
eleganza estetica: è ciò che permette di generalizzare.

## Quando la U non basta: la doppia discesa

C'è un punto in cui il quadro appena disegnato entra in tensione con la
pratica delle reti profonde, e vale la pena affrontarlo invece di ignorarlo. La
curva a U di poco fa dice: oltre una certa complessità l'errore sui
dati nuovi risale.

Guardiamo allora una grande rete neurale che riconosce immagini (una rete è un
modello fatto a strati, il protagonista dei prossimi capitoli). Ha milioni di
manopole regolabili e le si danno da studiare qualche decina di migliaia di
esempi: molte più manopole che esempi, il che vuol dire che le basta assegnarne
qualcuna a ciascuno per impararli tutti a memoria, rumore compreso. E infatti
lo fa, l'errore di addestramento va a zero. Secondo la U dovremmo essere
nel disastro, e invece quella rete generalizza benissimo.

```{figure} ../figures/double-descent.svg
:name: fig-double-descent
:alt: "Grafico con la capacità del modello, cioè il numero di parametri, in ascissa e l'errore in ordinata. L'errore di training scende e resta a zero. L'errore di test disegna prima la classica U del regime classico, con un minimo, poi risale fino a un picco in corrispondenza della soglia di interpolazione, e infine riscende in una seconda discesa nel regime sovraparametrizzato."
:width: 96%

La U non è sbagliata: è solo il primo tratto. Oltre il picco, dove il modello
ha appena abbastanza capacità per memorizzare tutto, la curva riscende invece
di continuare a salire.
```

Il punto interessante di {numref}`fig-double-descent` è il **picco**, non le
discese, e per capirlo serve l'immagine della curva che passa per dei punti.
Il picco sta dove il modello ha esattamente le manopole che servono per passare
per tutti i dati e nemmeno una di più: di curve così ne esiste una sola, il
modello è costretto a prendere quella, e quella è una curva che fra un punto e
l'altro impazzisce. Appena si aggiungono manopole, invece, le curve che passano
per tutti i punti tornano a essere infinite, e fra infinite ce n'è anche
qualcuna tranquilla: la parte sorprendente, di cui si parla fra poco, è che
l'addestramento tende proprio a quelle.

`````{tab} Elementare

Qualcuno ha fatto la cosa che i manuali sconsigliavano: ha continuato a
ingrandire il modello *oltre* il punto in cui impara a memoria ogni esempio. E
l'errore sul test, dopo essere risalito come previsto, **è tornato a scendere**.
Non un caso fortunato: un fenomeno riproducibile, chiamato **doppia discesa**.

La curva, insomma, non è una U ma una U seguita da una seconda discesa. Il
picco sta esattamente nel punto di **interpolazione**, cioè dove il modello
riesce per la prima volta a passare per tutti i punti (in matematica si dice
*interpolare*) e non gli avanza niente.

Un'immagine per vederlo: far passare una curva per venti punti. Con una curva
che ha esattamente venti numeri da scegliere, uno per ogni punto, non c'è
margine: la curva è obbligata, e fra un punto e l'altro impazzisce. Con molta
più libertà, invece, puoi scegliere fra le infinite curve che passano per quei
punti la meno tormentata.

E l'addestramento sceglie davvero quella? In buona parte sì, e la ragione sta
nel modo in cui procede. La discesa del gradiente
(la procedura a piccoli passi vista con la retta di best fit) parte da numeri
piccoli, sorteggiati vicino allo zero, e si muove a passettini finché i dati
non tornano; appena tornano, si ferma. Il risultato è che non va mai a cercare
lontano una soluzione strana, se ce n'è una mansueta lì vicino. Avere manopole
in eccesso non è più libertà di sbagliare, è **libertà di scegliere una
soluzione gentile**.

`````

`````{tab} Superiore

Il fenomeno è stato descritto sistematicamente da Belkin e colleghi (2019)
{cite}`belkin2019reconciling` e poi in ambito neurale da Nakkiran e colleghi
(2020) {cite}`nakkiran2020deep`. Tre precisazioni che evitano
di trarne la conclusione sbagliata.

**Non è solo la taglia del modello.** La doppia discesa si osserva anche
rispetto al *tempo di addestramento* (epoch-wise) e alla *quantità di dati*, e
in quest'ultimo caso produce l'effetto contro-intuitivo per cui, vicino al
punto di interpolazione, **aggiungere dati può peggiorare** il test error.

**Il rasoio di Occam non è confutato, è misurato male.** Il numero di
parametri è un pessimo proxy della complessità di una rete: la quantità che
conta è una misura di norma della soluzione trovata, non di capacità
dell'ipotesi. La discesa del gradiente ha un *bias implicito* verso soluzioni
a norma piccola, e in quel senso continua a scegliere la spiegazione più
semplice: solo che «semplice» non si conta in parametri.

**La regolarizzazione appiana il picco.** Con regolarizzazione adeguata la gobba
attorno all'interpolazione si attenua o sparisce: la doppia discesa è più
marcata proprio dove non si regolarizza.

Resta parecchio da capire: quali architetture e quali regimi la mostrino, e
perché il bias implicito abbia la forma che ha. Il consiglio operativo non è
cambiato, ma la sua motivazione sì: **non fermarsi al primo minimo della curva
di validazione solo perché il modello sembra troppo grande.**

`````

## Il biglietto vincente: a che serve tutta quella capacità

La doppia discesa dice *che* le reti sovradimensionate generalizzano. Resta la
domanda su *perché*, e c'è un risultato che offre una risposta diversa e
sorprendentemente concreta.

Prima però serve un'immagine di che cosa sia una **rete neurale**, perché qui
la si pota come una pianta (ci sarà un capitolo intero a raccontarla: quel che
segue basta per questa pagina). Immagina tanti nodi disposti a strati, e fra un
nodo e il successivo un filo che porta il segnale moltiplicandolo per un
numero: quel numero è il **peso** del collegamento, uno dei tanti parametri che
l'addestramento aggiusta. E i **pesi iniziali**, quelli da cui la messa a punto
parte, sono sorteggiati a caso. Sembra strano, e invece è necessario: se
partissero tutti dallo stesso valore, tutti i fili riceverebbero la stessa
correzione e resterebbero uguali per sempre, e una rete di fili identici non
serve a niente. Il sorteggio li rende diversi, e ognuno può specializzarsi.
Una rete grande ha milioni di questi
fili. Un peso quasi nullo è un filo che di fatto non trasmette niente:
tagliarlo non cambia la risposta, e **potare** vuol dire proprio tagliare i
fili più deboli; quello che resta dopo il taglio è una **sottorete**.

```{figure} ../figures/lottery-ticket-hypothesis.svg
:name: fig-biglietto-vincente
:alt: "A sinistra una rete densa con tutte le sue connessioni disegnate in grigio. A destra la stessa rete con evidenziato un sottoinsieme molto più piccolo di connessioni e nodi, il biglietto vincente, che addestrato da solo a partire dalla propria inizializzazione originale raggiunge la stessa accuratezza della rete intera."
:width: 96%

Dentro la rete grande ce n'è una piccola che basta. Il punto non è che si può
potare a posteriori: è che quella sottorete funziona solo se riparte dai *suoi*
pesi iniziali, quelli che aveva nella rete grande.
```

La condizione in coda a {numref}`fig-biglietto-vincente` è ciò che rende
questa idea, che si chiama **ipotesi del biglietto vincente**, interessante
invece che ovvia. Se si riprende la stessa sottorete e
la si inizializza da capo a caso, non impara altrettanto bene: il biglietto
non è la forma della sottorete, è la coppia fra la forma e i numeri con cui è
nata. Sovradimensionare, in questa lettura, serve a comprare molti biglietti.

`````{tab} Elementare

Il punto di partenza è un paradosso noto da tempo. Prendi una rete addestrata,
elimina i pesi più piccoli: puoi buttarne via il $90\%$ senza quasi perdere
accuratezza. Ma se poi provi a costruire da zero una rete piccola *con quella
stessa struttura* e ad addestrarla, impara peggio. La potatura funziona solo
**dopo** l'addestramento, e nessuno spiegava bene perché.

Frankle e Carbin (2019) hanno provato una cosa diversa. Dopo aver potato, invece
di ripartire con pesi casuali nuovi, hanno **riavvolto** i pesi sopravvissuti ai
valori casuali che avevano *all'inizio*, prima di qualsiasi addestramento. Quella
sottorete minuscola, riaddestrata da sola, raggiunge l'accuratezza della rete
piena.

Il dettaglio decisivo è che se reinizializzi con *altri* numeri casuali, non
funziona più. Quindi non conta solo quali connessioni sopravvivono: conta che
partano da quei valori lì. La rete grande, insomma, non serve perché serva tutta
quella capacità: serve perché, fra le sue milioni di connessioni inizializzate a
caso, ne contiene per fortuna un sottoinsieme già disposto bene per il compito.
Il resto è impalcatura. Da qui il nome: **biglietto vincente**, e la rete grande
come un mazzo di biglietti comprati tutti insieme.

`````

`````{tab} Superiore

La procedura {cite}`frankle2019lottery` è l’*iterative magnitude pruning*: si
annota l'inizializzazione
$\theta_0$, si addestra, si elimina una frazione dei pesi più piccoli, si
riportano i sopravvissuti ai valori in $\theta_0$, si riaddestra, si ripete. Le
sottoreti trovate pesavano spesso meno del $10$–$20\%$ della rete di partenza, e
raggiungevano l'accuratezza piena in un numero comparabile di iterazioni.

Due avvertenze onestà, perché il risultato è più fragile di come viene spesso
citato.

**Alla scala grande la ricetta va corretta.** Su reti profonde e dataset seri il
riavvolgimento a $\theta_0$ smette di funzionare; si riavvolge invece a un
$\theta_k$ dopo qualche iterazione di addestramento (*rewinding* tardivo). Il
biglietto, quindi, non è del tutto presente all'inizializzazione: si forma nelle
prime fasi.

**Non è un metodo di compressione pratico.** Per *trovare* il biglietto
bisogna addestrare la rete piena, più volte. Il valore è conoscitivo (dice
qualcosa su cosa fa la sovraparametrizzazione) non computazionale. Per
comprimere davvero si usano la potatura strutturata e la quantizzazione del
capitolo su MLOps.

Il filo con la sezione precedente è comunque lo stesso: il numero di parametri
misura male la complessità. La doppia discesa lo mostra dall'esterno, guardando
la curva d'errore; il biglietto vincente dall'interno, guardando cosa la rete
usa davvero.

`````

Attenzione a non trarne la conclusione sbagliata. Doppia discesa e biglietto
vincente non smontano nulla di ciò che viene prima: dicono soltanto che
«quanto è complesso un modello» non si conta in parametri. Tenere il test
chiuso, misurare su dati mai visti e far pagare un prezzo alla complessità
restano esattamente ciò che erano, e sono le cose che si portano via da questa
sezione.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Ci sono due modi opposti di sbagliare: essere **troppo rigidi** (una retta
  dove serviva una curva: si sbaglia già sugli esempi di scuola) ed essere
  **troppo flessibili** (una curva che passa per ogni punto, rumore compreso:
  dieci e lode sugli esempi di scuola, disastro sui casi nuovi). Il secondo è
  l’*overfitting*, cioè imparare a memoria.
- Immagina di riaddestrare il modello molte volte su dati sempre nuovi: se le
  risposte sono tutte spostate dalla stessa parte è un difetto di **mira**; se
  sono sparpagliate è un difetto di **stabilità**. Si correggono in modi
  opposti, e mettendo la flessibilità del modello su un asse l'errore sui dati
  nuovi disegna una **U**: si sceglie il fondo.
- Prima di spendere per raccogliere altri dati, si guardano le **curve di
  apprendimento**: si riaddestra con sempre più esempi e si guarda l'errore. Se
  quello sugli esempi di scuola e quello sui casi nuovi si sono già raggiunti e
  fermati, altri dati non servono e va cambiato modello; se fra i due resta un
  divario, i dati in più pagano.
- Per accorgersene bisogna misurare su dati che il modello non ha usato: si
  divide in tre, **studio, prove, esame**. L'esame (il *test*) si apre una sola
  volta, alla fine: ogni sbirciata lo consuma e il voto diventa più generoso
  del vero. E anche i preparativi (rimettere le colonne in scala, riempire le
  caselle vuote) vanno fatti guardando solo la parte di studio.
- Con pochi dati conviene la **cross-validation**: si divide in cinque blocchi
  e a turno uno fa da prova, come far correggere il compito a cinque professori
  invece che a uno. Contano la media dei cinque voti e quanto sono discordi.
- Per frenare la memorizzazione si mette un **prezzo alla complessità**: il
  modello può usare pesi grandi solo se ne vale la pena. Contando la spesa a
  valori assoluti alcuni pesi vanno esattamente a zero (le caratteristiche
  inutili spariscono), contandola a quadrati si rimpiccioliscono tutti.
- Il principio antico è il **rasoio di Occam**: a parità di spiegazione dei
  dati, vince la spiegazione più semplice. Con un'eccezione che vale la pena
  conoscere, la **doppia discesa**: oltre il punto in cui il modello impara
  tutto a memoria, ingrandirlo ancora torna a farlo funzionare meglio.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- **Underfitting**: modello troppo semplice, sbaglia già sul training (bias
  alto). **Overfitting**: modello troppo flessibile, memorizza il rumore ed è
  ottimo sul training ma pessimo sui dati nuovi (varianza alta).
- L'errore di test ha forma a **U** nella complessità: il minimo è il
  compromesso **bias-varianza**. Oltre il punto di interpolazione, però, la
  curva può **riscendere** (doppia discesa): il numero di parametri è un
  pessimo proxy della complessità di una rete.
- Il **biglietto vincente** guarda lo stesso fatto dall'interno: una rete grande
  contiene una sottorete piccola già ben inizializzata, e il resto è
  impalcatura.
- La decomposizione bias-varianza è un'identità della **loss quadratica**: per
  la 0-1 i due termini restano vocabolario, non aritmetica.
- Si divide in **train / validation / test**. Il **test non si tocca**: si apre
  una sola volta, alla fine, o la stima diventa ottimista. Ogni trasformazione
  che *impara* dai dati (scaler, imputazione, selezione) si tara dentro il
  training: è la forma di *leakage* che non dà avvisi.
- La **k-fold cross-validation** media $k$ validazioni su fold diversi: stima
  più stabile quando i dati sono pochi. Vale se le righe sono **scambiabili**:
  con righe raggruppate per soggetto servono `GroupKFold`, con righe ordinate
  nel tempo `TimeSeriesSplit`.
- La **regolarizzazione** (Ridge $\ell_2$, Lasso $\ell_1$) frena la complessità
  con una penalità $\lambda$ sui pesi; il Lasso azzera le feature inutili.
- Tutto obbedisce al **rasoio di Occam**: a parità di adattamento, vince il
  modello più semplice.
```

`````
