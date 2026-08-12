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
nulla di utile. Questo capitolo parla di come accorgersene e come porvi rimedio.

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
addestramento, l'*errore di generalizzazione*. Confrontarlo con l'errore
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

```{figure} ../figures/bias-varianza.svg
:name: fig-bias-varianza
:alt: "Quattro bersagli disposti in una griglia due per due, con le colonne per varianza bassa e alta e le righe per bias basso e alto. Con bias e varianza bassi i colpi sono raccolti al centro; con varianza alta e bias basso sono sparsi ma centrati in media; con bias alto e varianza bassa sono raccolti ma spostati dal centro; con entrambi alti sono sparsi e spostati."
:width: 78%

I quattro casi sul bersaglio. Il bias è di quanto si è spostato il gruppo dei
colpi; la varianza è quanto il gruppo è largo. Sono due difetti diversi e si
correggono in modi opposti.
```

Prima di leggere la figura serve sapere che cosa sia un colpo, perché il
modello è uno solo e i fori sul bersaglio sono tanti. Il gioco è questo:
immagina di rifare l'esperimento da capo molte volte, ogni volta raccogliendo
un campione di dati nuovo e riaddestrando il modello su quello. **Ogni foro sul
bersaglio è un addestramento**, e il centro del bersaglio è la risposta giusta.
Un modello può sbagliare in due modi indipendenti: perché il gruppo dei fori è
tutto spostato da una parte (sbaglia sempre nello stesso verso) oppure perché è
sparpagliato (cambia idea a ogni campione).

Il bersaglio in basso a sinistra di {numref}`fig-bias-varianza`, colpi
raccolti ma tutti fuori centro, è il più insidioso: un modello del genere è
molto *coerente*, dà quasi sempre la stessa risposta, e la coerenza si scambia
facilmente per affidabilità. Aggiungere dati non lo aggiusta, perché il
problema non è l'incertezza ma la mira.

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
lontano del teorema. La decomposizione è un'**identità della loss quadratica**:
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
modello che conviene scegliere. (Sarà una sezione più avanti, sulla *doppia
discesa*, a raccontare in quali casi la storia non finisca lì.)

### Distinguerli in pratica: le curve di apprendimento

Bias e varianza, finora, sono una spiegazione. C'è un modo di **misurarli**, e
risponde alla domanda che costa di più in un progetto vero: *conviene
raccogliere altri dati, o cambiare modello?*

Si tracciano di nuovo due curve, ma cambiando l'asse orizzontale: non più la
complessità del modello, come nella U di poco fa, bensì **la quantità di dati
usata**. Le due curve sono l'errore sugli esempi con cui il modello ha
studiato (l'addestramento) e l'errore su esempi tenuti da parte per giudicarlo
(la **validazione**: la prossima sezione spiega come si mettono da parte e
perché sia essenziale farlo). La forma che assumono dice quale dei due mali si
ha davanti.

- Le due curve **si avvicinano e si fermano in alto**: il modello sbaglia
  quanto sui dati che ha visto quanto su quelli che non ha visto, ed è già al
  suo limite. È **bias**. Altri dati non servono a niente: serve più capacità.
- Fra le due resta un **divario largo**, e quella di validazione sta ancora
  scendendo: il modello ha imparato bene ciò che ha visto e generalizza meno. È
  **varianza**, e qui altri dati aiutano davvero.

Sono cinque righe di scikit-learn, e vale la pena eseguirle perché il verdetto
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

Prima dei numeri serve un metro, altrimenti «alto» e «basso» non vogliono dire
niente. Su questi dati chi rispondesse sempre la media sbaglierebbe di
$3{,}5$; e siccome il target porta un rumore che nessun modello può prevedere,
sotto $0{,}09$ non si può scendere nemmeno sapendo tutto. Fra $3{,}5$ e
$0{,}09$ si gioca la partita.

Il modello lineare, passando da 120 a 2400 esempi, chiude il divario da
$+0{,}230$ a $+0{,}013$: le due curve si sono **toccate**, e si sono toccate a
$2{,}3$, cioè poco sotto la metà strada fra il rispondere a caso e il sapere
tutto. L'errore di addestramento, per giunta, non è affatto migliorato
($2{,}294$ con 120 esempi, $2{,}321$ con 2400): quel modello ha dato tutto
quello che aveva, e altri diecimila esempi non sposterebbero nulla. Se serve
di meglio, serve un modello diverso.

La foresta (una **foresta casuale**, un modello fatto di tanti alberi di
decisione che votano: la incontreremo fra due sezioni, e qui basta sapere che è
molto più flessibile di una retta) arriva a $0{,}031$ sull'addestramento e
$0{,}215$ in validazione, con un divario di $+0{,}184$ ancora aperto: ha
imparato benissimo ciò che ha visto e generalizza un po' meno, ma il suo
$0{,}215$ è già vicino al pavimento di $0{,}09$. Diagnosi opposta e ricetta
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
**non ha usato** per imparare. Da qui la regola d'oro: si divide il dataset in
tre parti, ciascuna con un compito distinto.

- **Training set**: i dati su cui il modello impara i suoi parametri (i numeri
  interni, la $\theta$ del capitolo introduttivo).
- **Validation set**, i dati su cui si scelgono gli *iperparametri*, cioè le
  scelte di contorno che non si imparano dai dati: quanto complesso può essere
  il modello, quanto forte il freno alla memorizzazione che vedremo tra poco.
- **Test set**: i dati che si guardano **una sola volta**, alla fine, per
  stimare onestamente le prestazioni nel mondo reale.

`````{tab} Elementare

Il test set è il compito d'esame vero. Se lo sbirci mentre studi e aggiusti le
tue scelte in base a quello, il voto finale non dice più nulla: hai imparato a
memoria *quell'* esame. Per questo il test si tiene chiuso in un cassetto e si
apre soltanto alla fine. Ogni volta che usi il test per decidere qualcosa, lo
"consumi", e il numero che ti restituisce diventa troppo ottimista.

`````

`````{tab} Superiore

Usare il test per selezionare modelli introduce una forma sottile di *data
leakage*: si finisce per fare overfitting sul test stesso, e la stima
dell'errore di generalizzazione diventa distorta verso il basso. Il validation
set esiste proprio per assorbire tutte le decisioni intermedie e preservare
l'imparzialità del test. Suddivisioni tipiche: $60/20/20$ o $80/10/10$. La
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

Quasi mai i dati si danno al modello così come sono. Prima si sistemano: si
riportano le colonne a una scala comune, perché i metri quadri e il numero di
stanze non si schiaccino a vicenda (lo strumento che lo fa si chiama *scaler*,
e per farlo deve calcolare media e ampiezza di ogni colonna); si riempiono le
caselle vuote con un valore plausibile, per esempio la media della colonna (si
chiama **imputazione**); si scartano le colonne che non servono. Tutte queste
operazioni **imparano qualcosa dai dati**, e qui sta la trappola: se lo
imparano guardando anche il test, allora il test ha già parlato.

È la forma più insidiosa di **data leakage** (una «fuga» di informazione dal
test verso l'addestramento) perché non produce nessun errore e nessun avviso:
produce solo un punteggio un po' più alto del vero. La regola pratica che ne
discende è secca: qualunque cosa impari dai dati va calcolata **dentro** il
training e poi applicata al resto, mai prima della divisione.

## La cross-validation

Mettere da parte un validation set fisso ha un difetto: con pochi dati, la stima
dipende troppo da *quali* esempi sono finiti nel validation. La
**k-fold cross-validation** aggira il problema riutilizzando i dati con
intelligenza.

```{figure} ../figures/cross-validation-il-test-che-non-bara.svg
:name: fig-cross-validation
:alt: "Cinque righe, una per giro. In ciascuna, i dati sono divisi in cinque blocchi: uno fa da test e gli altri quattro da training, e il blocco di test scorre di una posizione a ogni riga, dal primo al quinto. A destra di ogni riga il punteggio ottenuto in quel giro. In fondo, il risultato è la media dei cinque punteggi con la loro deviazione standard."
:width: 92%

Il blocco di test ruota. Alla fine ogni esempio ha fatto da test esattamente
una volta, e il risultato non è un numero ma un numero con la sua
variabilità.
```

La riga finale di {numref}`fig-cross-validation` è la parte che si tende a
buttare via: non la media dei cinque punteggi, ma la loro **dispersione**, cioè
di quanto i cinque giri si discostano dalla media (in statistica la si riassume
in un numero, la *deviazione standard*: quanto in media un giro si scosta dal
risultato medio). Se due modelli distano meno di quella, la classifica fra loro
dipende da come sono caduti i blocchi, non da quale sia migliore.

`````{tab} Elementare

Dividi i dati di addestramento in $k$ blocchi uguali (di solito $k=5$ o $10$).
A turno, ogni blocco fa da validation mentre gli altri $k-1$ addestrano il
modello. Ottieni così $k$ misure di errore, ognuna su un pezzo diverso di dati,
e ne fai la **media**. È come far correggere il compito a cinque professori
diversi invece che a uno solo: il giudizio finale è più affidabile e meno
soggetto al caso.

`````

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
stima un po' ottimista: è una stima priva di significato, e non dà nessun
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

## Mettere un freno: regolarizzazione L1 e L2

Un modo diretto per contrastare l'overfitting è impedire al modello di diventare
troppo "estremo". La **regolarizzazione** aggiunge all'errore da minimizzare un
secondo addendo, una **penalità** che cresce con la grandezza dei pesi (i
numeri per cui il modello moltiplica ogni caratteristica): così il modello paga
un prezzo ogni volta che alza troppo un peso, e lo fa solo se ne vale davvero
la pena. Quanto sia caro quel prezzo è una manopola che decidiamo noi, e nelle
formule si chiama $\lambda$ (la lettera greca *lambda*): $\lambda$ a zero vuol
dire nessun freno, $\lambda$ grande vuol dire freno tirato.

```{figure} ../figures/regolarizzazione-l1-l2.svg
:name: fig-l1-l2
:alt: "Due piani con i pesi w1 e w2 sugli assi. A sinistra la L1: la regione ammessa è un rombo con i vertici sugli assi, e le curve di livello dell'errore lo toccano proprio in un vertice, dove w1 è esattamente zero. A destra la L2: la regione è un cerchio, e il punto di contatto cade in una posizione qualsiasi del bordo, dove entrambi i pesi sono piccoli ma nessuno è zero."
:width: 96%

La differenza sta negli spigoli. Il rombo della L1 ha i vertici sugli assi, e
un vertice è il punto che una curva di livello incontra per primo: da lì i
pesi esattamente nulli. Il cerchio della L2 non ha spigoli, e non privilegia
nessuna direzione.
```

{numref}`fig-l1-l2` spiega con un disegno quello che di solito si impara come
una regola da mandare a memoria («la L1 azzera i pesi inutili, la L2 no»), e
vale la pena leggerla con calma perché la ragione è tutta lì e non è
misteriosa.

Immagina un piano con due soli pesi, $w_1$ e $w_2$, uno per asse: ogni punto
del piano è una scelta possibile dei due numeri. Dire al modello «non spendere
più di tanto in pesi» significa recintare una regione attorno all'origine e
obbligarlo a restare dentro: se la spesa si conta sommando i **valori
assoluti** il recinto è un rombo con le punte sugli assi; se si conta sommando
i **quadrati** è un cerchio. Fuori dal recinto, l'errore del modello ha la
forma di una collina con il fondo dove starebbe la soluzione senza freni: se
disegniamo le linee che uniscono i punti di pari errore (le stesse curve di
livello di una carta topografica), sono anelli che si stringono attorno a quel
fondo. La soluzione col freno è dove il primo di quegli anelli, allargandosi,
tocca il recinto.

Ed è qui che la forma decide: un anello che si allarga incontra un rombo quasi
sempre in una **punta**, e le punte del rombo stanno sugli assi, cioè in punti
dove uno dei due pesi vale esattamente zero. Un cerchio invece non ha punte, e
il primo contatto cade in un posto qualunque del bordo, dove entrambi i pesi
sono piccoli ma nessuno è nullo. Ecco perché sommare i valori assoluti seleziona
le caratteristiche e sommare i quadrati no: non è una proprietà nascosta della
formula, è la forma del recinto.

`````{tab} Elementare

Pensa alla regolarizzazione come a un budget di spesa sui pesi del modello.
Senza limiti, per passare su ogni punto il modello gonfia i suoi coefficienti
a dismisura: è così che nasce la curva contorta. Imponendo un tetto alla spesa
totale, lo costringi a essere sobrio, e le curve sobrie sono più morbide,
quindi generalizzano meglio. Due modi di contare la spesa:

- **Ridge (L2)**: penalizza la *somma dei quadrati* dei pesi. Li rimpicciolisce
  tutti dolcemente, senza azzerarne nessuno. (Il quadrato punisce pochissimo
  chi è già piccolo: portare un peso da $0{,}1$ a $0$ fa risparmiare appena
  $0{,}01$, e quel risparmio non vale mai la pena.)
- **Lasso (L1)**: penalizza la *somma dei valori assoluti*. Qui l'ultimo
  centesimo costa quanto il primo, quindi conviene sempre azzerare del tutto un
  peso che non serve: tende a spingere a **zero netto** i pesi delle feature
  inutili, di fatto selezionandole. È lo stesso fatto che il rombo con le punte
  sugli assi racconta con la geometria.

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

L'**Elastic Net** somma le due penalità,
$\lambda\big(\alpha\sum_j|\theta_j| + \tfrac{1-\alpha}{2}\sum_j\theta_j^2\big)$,
e non è un compromesso pigro: rimedia a un difetto preciso del Lasso. Fra due
feature fortemente correlate il Lasso ne tiene **una sola**, scelta in modo
instabile (basta cambiare il campione perché scelga l'altra), mentre il termine
$\ell_2$ le fa entrare o uscire **insieme**, il cosiddetto *grouping effect*.
Con feature molte e correlate, che è il caso normale sui dati reali, è la
scelta di partenza più sensata delle due pure.

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
pratica del deep learning, e vale la pena affrontarlo invece di ignorarlo. La
curva a U di poche pagine fa dice: oltre una certa complessità l'errore sui
dati nuovi risale. Eppure una
grande rete neurale che riconosce immagini (una rete è un modello fatto a
strati, il protagonista dei prossimi capitoli) ha milioni di parametri contro
le poche decine di migliaia di esempi su cui la si addestra: azzera l'errore di
training, il che significa memorizzare ogni esempio rumore compreso, e
ciononostante generalizza.

```{figure} ../figures/double-descent.svg
:name: fig-double-descent
:alt: "Grafico con la capacità del modello, cioè il numero di parametri, in ascissa e l'errore in ordinata. L'errore di training scende e resta a zero. L'errore di test disegna prima la classica U del regime classico, con un minimo, poi risale fino a un picco in corrispondenza della soglia di interpolazione, e infine riscende in una seconda discesa nel regime sovraparametrizzato."
:width: 96%

La U non è sbagliata: è solo il primo tratto. Oltre il picco, dove il modello
ha appena abbastanza capacità per memorizzare tutto, la curva riscende invece
di continuare a salire.
```

Il punto interessante di {numref}`fig-double-descent` è il **picco**, non le
discese. Sta dove il modello ha esattamente la capacità necessaria per
interpolare i dati e nessuna di più: è costretto a una sola soluzione, e
quella soluzione è pessima. Con più capacità le soluzioni tornano a essere
tante, e fra tante l'addestramento ne sceglie una regolare.

`````{tab} Elementare

Qualcuno ha fatto la cosa che i manuali sconsigliavano: ha continuato a
ingrandire il modello *oltre* il punto in cui impara a memoria ogni esempio. E
l'errore sul test, dopo essere risalito come previsto, **è tornato a scendere**.
Non un caso fortunato: un fenomeno riproducibile, chiamato **doppia discesa**.

La curva, insomma, non è una U ma una U seguita da una seconda discesa. Il
picco sta esattamente nel punto di **interpolazione**: quando il modello ha
giusto i parametri sufficienti per azzerare l'errore di training. Lì è
costretto a passare per tutti i punti, rumore compreso, nell'unico modo che
gli riesce: contorcendosi. È il momento peggiore.

Oltre quel punto, però, di soluzioni che passano per tutti i dati ce ne sono
infinite, e l'addestramento non ne sceglie una a caso: la discesa del gradiente
(la procedura a piccoli passi vista con la retta di best fit) parte da valori
piccoli e si ferma appena i dati sono spiegati, quindi tende verso le soluzioni
«più lisce» fra quelle disponibili. Avere parametri in
eccesso non è più libertà di sbagliare, è **libertà di scegliere una soluzione
gentile**.

Un'immagine: far passare una curva per venti punti. Con una curva che ha venti
numeri da scegliere, uno per ogni punto, non c'è margine: la curva è obbligata,
e fra un punto e l'altro impazzisce. Con molta più libertà puoi scegliere, fra
le infinite curve che passano per quei punti, la meno tormentata.

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

```{figure} ../figures/lottery-ticket-hypothesis.svg
:name: fig-biglietto-vincente
:alt: "A sinistra una rete densa con tutte le sue connessioni disegnate in grigio. A destra la stessa rete con evidenziato un sottoinsieme molto più piccolo di connessioni e nodi, il biglietto vincente, che addestrato da solo a partire dalla propria inizializzazione originale raggiunge la stessa accuratezza della rete intera."
:width: 96%

Dentro la rete grande ce n'è una piccola che basta. Il punto non è che si può
potare a posteriori: è che quella sottorete funziona solo se riparte dai *suoi*
pesi iniziali, quelli che aveva nella rete grande.
```

La condizione in coda a {numref}`fig-biglietto-vincente` è ciò che rende
l'ipotesi interessante invece che ovvia. Se si riprende la stessa sottorete e
la si inizializza da capo a caso, non impara altrettanto bene: il biglietto
non è la forma della sottorete, è la coppia fra la forma e i numeri con cui è
nata. Sovradimensionare, in questa lettura, serve a comprare molti biglietti.

`````{tab} Elementare

Serve prima un'immagine di che cosa sia una **rete neurale**, perché qui la si
pota come una pianta. Immagina tanti nodi disposti a strati, e fra un nodo e il
successivo un filo che porta il segnale moltiplicandolo per un numero: quel
numero è il **peso** del collegamento, ed è uno dei tanti parametri che
l'addestramento aggiusta. Una rete grande ha milioni di questi fili. Un peso
quasi nullo è un filo che di fatto non trasmette niente: tagliarlo non cambia
la risposta, e «potare» vuol dire proprio tagliare i fili più deboli.

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

La procedura {cite}`frankle2019lottery` è l'*iterative magnitude pruning*: si
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
  l'*overfitting*, cioè imparare a memoria.
- Immagina di riaddestrare il modello molte volte su dati sempre nuovi: se le
  risposte sono tutte spostate dalla stessa parte è un difetto di **mira**; se
  sono sparpagliate è un difetto di **stabilità**. Si correggono in modi
  opposti, e mettendo la flessibilità del modello su un asse l'errore sui dati
  nuovi disegna una **U**: si sceglie il fondo.
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
