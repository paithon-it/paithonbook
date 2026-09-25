# Overfitting, bias-varianza e validazione

C'è un modo infallibile per andare male a un esame: imparare a memoria le
soluzioni dei compiti degli anni scorsi. Chi lo fa risponde alla perfezione
alle domande già viste e va nel panico davanti a un esercizio anche solo
leggermente diverso. Ha memorizzato, non capito. Un modello di machine
learning può cadere esattamente nella stessa trappola, e distinguere la
memoria dalla comprensione è, in fondo, il problema centrale di tutta la
disciplina.

Nelle sezioni precedenti abbiamo detto che la posta in gioco non è riprodurre
gli esempi già visti, ma generalizzare a input nuovi. Un modello che azzecca
ogni risposta sui dati di addestramento e sbaglia su quelli veri non ha imparato
nulla di utile. Resta da vedere come accorgersene e come porvi rimedio.

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

Un modello troppo semplice, una retta a cui si chiede di descrivere dati
chiaramente curvi, sbaglia già sugli esempi su cui ha studiato. E su un esempio
nuovo sbaglia più o meno quanto sui vecchi: il suo limite è la forma che ha, e
resta quello anche dandogliene altri mille. Si chiama **underfitting**, ed è un
modello troppo rigido per il problema.

All'estremo opposto c'è il modello troppo flessibile, che si contorce per
passare esattamente su ogni singolo punto, rumore compreso. Sul foglio degli
esempi già visti prende dieci e lode, ma ha imparato anche gli errori di
misura, gli accidenti, il caso, e su un dato nuovo crolla. Si chiama
**overfitting**: ha memorizzato invece di capire.

Accorgersene è un confronto fra due numeri: quanto il modello sbaglia sugli
esempi con cui ha studiato, e quanto sbaglia su esempi che non ha mai visto.
Sbaglia parecchio in tutti e due i casi, e più o meno allo stesso modo? È
underfitting. Quasi niente sugli esempi di studio e parecchio sugli altri? È
overfitting, e la distanza fra i due numeri ne dà la misura: quando quella
distanza si allarga, il modello sta memorizzando.

`````

`````{tab} Superiore

Formalmente, l'errore che ci interessa è quello su dati mai visti in
addestramento, l’*errore di generalizzazione*. Confrontarlo con l'errore
sull'insieme di training rivela il regime in cui ci troviamo:

- Underfitting: errore di training *alto* e vicino a quello di test. Il
  modello è troppo poco espressivo: non riesce a catturare la struttura dei
  dati (*bias* alto).
- Overfitting: errore di training *molto basso* ma errore di test *alto*.
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
modello è uno solo e i fori sul bersaglio sono tanti. Il gioco è questo: si
rifà l'esperimento da capo molte volte, ogni volta raccogliendo
un campione di dati nuovo e riaddestrando il modello su quello. Ogni foro sul
bersaglio è un addestramento, e il centro del bersaglio è la risposta giusta.
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
dei fori, non a spostarlo: qui il guaio è dove il gruppo è centrato, e altri
dati non lo aggiustano.

`````{tab} Elementare

Al poligono ci sono due tiratori.

Il primo ha il mirino storto sempre nello stesso modo. I suoi colpi finiscono
tutti raccolti, e tutti a dieci centimetri dal centro. È il modello rigido, la
retta: campione dopo campione dà quasi la stessa risposta, e quasi sempre la
stessa risposta storta, perché la forma giusta non è una retta. L'errore che si
ripete uguale è il bias (si pronuncia *bàias*, e in inglese vuol dire
proprio «inclinazione», la tendenza a pendere sempre dalla stessa parte).

Il secondo ha il mirino a posto ma la mano che trema. In media è centrato,
preso colpo per colpo va un po’ dappertutto. È il modello flessibile, la curva
contorta: a ogni campione nuovo cambia parecchio, perché insegue il rumore di
turno. Quell'irrequietezza è la varianza.

Poi c'è il vento, che non dipende da nessuno dei due. Anche con il mirino a
posto e la mano ferma i colpi non cadono tutti nello stesso punto, perché
l'aria si muove. Nei dati il vento è la misura imprecisa, l'eccezione, tutto
ciò che capita e basta: quella parte di errore resta lì comunque, e nessun
modello, per quanto bravo, se la prende.

Il conto va fatto come si è fatto sul prezzo delle case: si misura di quanto il
colpo si allontana dal centro, si eleva al quadrato, e si fa la media su tutti
i colpi. Solo allora i tre pezzi si sommano davvero, e quella media si spacca
in tre addendi puliti: quello del mirino, quello della mano, quello del vento.
Sulle distanze nude la somma non torna. Il quadrato di una somma si apre in
pezzi, i quadrati dei tre scarti più i loro prodotti a due a due, e i prodotti
misti, in media su molti colpi, fanno zero: quelli con la mano perché il
tremore, misurato dal punto medio dei colpi, in media si annulla da sé, e
quelli con il vento perché il vento non sa niente né del mirino né della mano.
Resta la somma dei tre quadrati. È la stessa aritmetica che
rende il quadrato scomodo da leggere a renderlo scomponibile.

Letto così, il quadro è semplice. Un modello rigido ha molto bias e poca
varianza; uno flessibile, poco bias e molta varianza. Il bravo modellista cerca
il punto di mezzo.

Il conto vale finché a giudicare è la distanza dal centro. Se quel che conta è
soltanto finire dentro il cerchietto o fuori, il tiratore raccolto e storto non
fa un centro in tutta la giornata, mentre quello che trema ogni tanto dentro ci
finisce. Dove la domanda è secca, sano o malato, un po’ di tremore può perfino
convenire, e i tre addendi smettono di sommarsi.

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

Il bias misura quanto la previsione media si scosta dalla verità $f(x)$; la
varianza quanto $\hat{f}(x)$ oscilla al variare del campione; $\sigma^2$ è il
rumore intrinseco, che nessun modello può eliminare. Aumentando la complessità il
bias cala ma la varianza cresce: l'errore di test ha la classica forma a U, e
il minimo è il modello ottimale.

Un'avvertenza sull'ambito di validità, perché il vocabolario viaggia più
lontano del teorema. La decomposizione è un’identità della loss quadratica:
per la loss 0-1 dei classificatori manca una scomposizione additiva con
termini di segno fisso: in quella proposta da Domingos
{cite}`domingos2000unified`, per due classi, la varianza entra col segno più
dove la previsione più frequente fra i possibili addestramenti coincide con
quella ottima e col segno meno dove non coincide, e in generale non
esiste una scomposizione additiva
analoga {cite}`wood2023unified`, e più varianza può perfino *ridurre* l'errore
quando il bias sta dalla parte sbagliata della soglia. Da qui in avanti «bias» e
«varianza» restano utilissimi come vocabolario anche parlando di alberi e di
foreste; non come aritmetica.

`````

Il compromesso si può disegnare, e conviene tenere in mente il disegno perché
tornerà più volte. Mettiamo su un asse orizzontale la complessità del
modello, da sinistra (rigidissimo: una retta) a destra (flessibilissimo: la
curva che si contorce), e sull'asse verticale l'errore che il modello commette
sui dati nuovi. Da sinistra l'errore scende, perché il modello è troppo
rozzo e ogni pezzetto di flessibilità in più lo aiuta; a destra risale, perché
il modello comincia a imparare a memoria. In mezzo c'è un punto più basso di
tutti. La curva, insomma, ha la forma di una U, e il fondo della U è il
modello che conviene scegliere. (Più avanti, parlando di *doppia discesa*,
vedremo in quali casi la storia non finisca lì.)

### Distinguerli in pratica: le curve di apprendimento

Bias e varianza, finora, sono una spiegazione. C'è un modo di misurarli, e
risponde alla domanda che costa di più in un progetto vero: *conviene
raccogliere altri dati, o cambiare modello?*

Si disegnano di nuovo delle curve, ma stavolta sono due e l'asse orizzontale
cambia: non più la
complessità del modello, come nella U di poco fa, bensì la quantità di dati
usata, da pochi esempi a tutti quelli che abbiamo. Le due curve sono
l'errore sugli esempi con cui il modello ha
studiato (l'addestramento) e l'errore su esempi tenuti da parte per giudicarlo
(la validazione: come si mettono da parte, e perché sia essenziale farlo,
arriva fra poco). Guardandole scendere si capisce
quale dei due mali si ha davanti.

- Le due curve si avvicinano e si fermano in alto: il modello sbaglia
  tanto sui dati che ha visto quanto su quelli che non ha visto, ed è già al
  suo limite. È bias. Altri dati non servono a niente: serve un modello
  capace di piegarsi a forme più complicate.
- Fra le due resta un divario largo, e quella di validazione sta ancora
  scendendo: il modello ha imparato bene ciò che ha visto e generalizza meno. È
  varianza, e qui altri dati aiutano davvero.

Sono poche righe di scikit-learn, e conviene eseguirle perché il verdetto è
netto.

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve

rng = np.random.default_rng(0)
m = 3000
X = rng.normal(size=(m, 6))
y = np.sin(2 * X[:, 0]) + X[:, 1] ** 2 - X[:, 2] + rng.normal(0, 0.3, m)  # non lineare

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
    usati, tr, va = learning_curve(modello, X, y, train_sizes=taglie, cv=5,
                                   scoring="neg_mean_squared_error",
                                   shuffle=True, random_state=0)
    tr, va = -tr.mean(1), -va.mean(1)
    print(f"\n{nome}")
    print(f"  con {usati[0]:>4} esempi: train {tr[0]:.3f}  validazione {va[0]:.3f}"
          f"  divario {va[0]-tr[0]:+.3f}")
    print(f"  con {usati[-1]:>4} esempi: train {tr[-1]:.3f}  validazione {va[-1]:.3f}"
          f"  divario {va[-1]-tr[-1]:+.3f}")
```

```text
rispondere sempre la media: 3.471
pavimento del rumore:       0.090

lineare (troppo semplice)
  con  120 esempi: train 2.294  validazione 2.524  divario +0.230
  con 2400 esempi: train 2.321  validazione 2.335  divario +0.013

foresta (abbastanza ricca)
  con  120 esempi: train 0.142  validazione 0.844  divario +0.702
  con 2400 esempi: train 0.031  validazione 0.215  divario +0.184
```

Una parola su che cosa sono questi numeri. Qui gli $y$ sono numeri puri e non
euro né gradi: li fabbrica la riga `y = ...`. E l'errore si misura
come per la retta di best fit, cioè scarto fra vero e previsto, elevato al
quadrato e mediato. Un errore di $2{,}3$ vuol dire che, in media, il quadrato
dello scarto vale $2{,}3$: da solo non dice niente, e infatti le prime due
righe del programma servono a costruire il metro.

Il primo estremo del metro è quanto sbaglia chi non ci prova nemmeno, cioè
chi risponde sempre la media di tutti gli $y$: su questi dati $3{,}471$. (È la
varianza di $y$, e attenzione, non c'entra con la varianza del modello di poco
fa: qui è semplicemente quanto i valori di $y$ sono sparpagliati attorno alla
loro media.) Il secondo estremo è quanto sbaglia chi sa tutto. Non è zero:
la riga che fabbrica $y$ ci aggiunge un disturbo casuale di ampiezza $0{,}3$,
che nessun modello può indovinare perché non dipende da niente, e siccome
l'errore si misura al quadrato quel disturbo costa $0{,}3^2 = 0{,}09$. Fra
$3{,}471$ e $0{,}09$ si gioca tutta la partita: la strada da percorrere è lunga
$3{,}471 - 0{,}09 = 3{,}38$.

Il modello lineare, passando da 120 a 2400 esempi, chiude il divario da
$+0{,}230$ a $+0{,}013$: le due curve si sono toccate. Ma si sono toccate a
$2{,}335$, che è ancora quasi in cima: dai $3{,}471$ di partenza sono scesi
appena $1{,}14$ su $3{,}38$, cioè un terzo della strada. (Metà strada sarebbe
stata $0{,}09 + 3{,}38/2 = 1{,}78$, parecchio più in basso.) L'errore di
addestramento, per giunta, non è migliorato di un'unghia ($2{,}294$ con 120
esempi, $2{,}321$ con 2400: la differenza è più piccola di quanto il sorteggio
dei blocchi sposti da solo). Quello che non fa è scendere, ed è il segno che
stiamo cercando: con pochi esempi una retta riesce a passare un po’ più vicino
a tutti, con tanti non ce la fa più, perché la forma giusta non è una retta e i
punti in più non fanno che ricordarglielo. Quel modello ha dato tutto quello
che aveva, e altri diecimila esempi non sposterebbero nulla. Se serve di
meglio, serve un modello diverso.

La foresta (una foresta casuale, un modello fatto di tanti alberi di
decisione che votano: la incontreremo negli {doc}`alberi decisionali e metodi
ensemble </MachineLearning/alberi-ensemble>`, e qui basta
sapere che è molto più flessibile di una retta) arriva a $0{,}031$
sull'addestramento e
$0{,}215$ in validazione, con un divario di $+0{,}184$ ancora aperto: ha
imparato benissimo ciò che ha visto e generalizza un po’ meno. Ma il suo
$0{,}215$, sul metro di prima, è a un passo dal traguardo: della strada da
$3{,}471$ a $0{,}09$ ne ha percorso il $96\%$. Diagnosi opposta e ricetta
opposta: qui i dati in più pagano.

Il valore di questa diagnostica è che si fa prima di spendere. Raccogliere
o etichettare dati è la voce più cara di quasi ogni progetto, e queste due
curve dicono in un pomeriggio se quella spesa avrà un effetto.

```{admonition} Due curve diverse con lo stesso nome
:class: note
Attenzione a non confonderle con le curve che si guardano durante
l'addestramento di una rete, dove sull'asse orizzontale ci sono le epoche
(un'epoca è una passata completa su tutti gli esempi: si addestra facendone
molte di seguito): quelle diagnosticano l'andamento di *quella* sessione (passi
della discesa del gradiente troppo lunghi o troppo corti, overfitting che
comincia, quando fermarsi) e sono trattate nel {doc}`training loop
</PyTorch/addestramento>`. Qui l'asse orizzontale è la quantità di dati, e la
domanda è diversa: non «come sta andando questo addestramento» ma «questo
modello, con più dati, andrebbe meglio».
```

## Train, validation e test: perché il test non si tocca

Per accorgersi dell'overfitting bisogna misurare l'errore su dati che il modello
non ha usato per imparare. Da qui la regola d'oro: si divide il mucchio
degli esempi in tre parti, ciascuna con un compito distinto. Sono, nell'ordine,
studio, prove ed esame.

- **Training set** (lo studio): i dati su cui il modello impara i suoi parametri
  (i numeri interni, la $\theta$ dell'apertura del capitolo). È la fetta più
  grossa.
- **Validation set** (le prove): i dati tenuti da parte per giudicare, quelli
  che nelle curve di apprendimento davano la seconda curva, e su cui si
  scelgono gli *iperparametri*, cioè le scelte di contorno che non si imparano
  dai dati: quanto complesso può essere il modello, quanto forte il freno alla
  memorizzazione che vedremo tra poco.
- **Test set** (l'esame): i dati che si guardano una sola volta, alla fine,
  per stimare onestamente le prestazioni nel mondo reale.

Quanto grandi? Non c'è una regola sacra: proporzioni tipiche sono $60/20/20$ o
$80/10/10$, cioè in ogni caso la maggior parte degli esempi allo studio.

`````{tab} Elementare

Il test set è il compito d'esame vero. Se lo sbirci mentre studi e correggi le
tue scelte in base a quello, il voto finale non dice più nulla: hai imparato a
memoria *quell’* esame. Per questo il test si tiene chiuso in un cassetto e si
apre soltanto alla fine. Ogni volta che usi il test per decidere qualcosa, lo
"consumi", e il numero che ti restituisce diventa troppo ottimista.

E si sbircia anche senza barare. Basta aprire la busta per vedere di che cosa
parla, e le settimane di studio si organizzano da sole attorno a quel poco che
si è intravisto, anche se su quelle pagine non ci si esercita mai. Per questo
la busta si mette da parte come primo gesto, prima ancora di dare un'occhiata
ai dati: dopo, è tardi.

Le prove in vista dell'esame sono un'altra faccenda, e si possono rifare quante
volte si vuole. È il loro mestiere: assorbono tutte le decisioni prese per
strada, su che cosa insistere e per quanto, così che all'esame si arrivi
puliti.

Resta da comporre la busta, e il sorteggio da solo non basta. Con moltissime
domande in gioco il caso si compensa; con poche, o quando un argomento del
programma compare una volta sola, capita di tirare fuori un esame che su
quell'argomento non chiede niente. Il voto esce alto e misura un'altra cosa. Si
compone allora a proporzioni, da ogni argomento tante domande quanto quello
pesa nel programma, e nella busta finisce un po’ di tutto.

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
che sembra tale. La prima riguarda come si divide: il taglio puramente
casuale è affidabile solo se il dataset è grande, e su dataset piccoli o con
categorie rare produce insiemi che non si somigliano. Il rimedio è il
**campionamento stratificato**, che preserva in ciascuna parte le proporzioni
della variabile che conta (la classe da predire, o una covariata importante):
è il `stratify=` di `train_test_split` e la `StratifiedKFold` della sezione
seguente. Nel caso estremo, un test set che non contiene un solo esempio della
classe rara non misura la cosa che interessa.

La seconda è che il test si sporca anche soltanto guardandolo. È il *data
snooping bias*: se si ispeziona il test per decidere quali feature costruire,
quale trasformazione applicare o quale famiglia di modelli provare, quelle
decisioni sono state prese sui dati d'esame, e il numero finale è ottimista
anche se il modello non li ha mai visti in addestramento. La disciplina
corretta è mettere da parte il test come primo gesto, prima ancora
dell'analisi esplorativa.

`````

C'è però una perdita d'informazione più insidiosa di tutte, perché non passa
dal modello: passa dai preparativi.

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
operazioni imparano qualcosa dai dati, e qui sta la trappola: se lo
imparano guardando anche il test, allora il test ha già parlato.

È la forma più insidiosa di **data leakage** (una «fuga» di informazione dal
test verso l'addestramento) perché non produce nessun errore e nessun avviso:
produce solo un punteggio un po’ più alto del vero. La regola pratica che ne
discende è secca: qualunque cosa impari dai dati va calcolata dentro il
training e poi applicata al resto, mai prima della divisione.

## La cross-validation

Mettere da parte un validation set fisso ha un difetto: con pochi dati, la stima
dipende troppo da *quali* esempi sono finiti nel validation. La
**k-fold cross-validation** aggira il problema riutilizzando i dati con
intelligenza.

Si badi bene: qui il test, quello dell'esame, resta chiuso nel cassetto dove
l'abbiamo messo. Quello che si divide in blocchi è soltanto la parte di
studio, e ciò che ruota è il blocco delle prove.

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
e ne fai la media. È come fare cinque compiti in classe su cinque parti
diverse del programma invece di giocarsi tutto su una sola interrogazione: il
giudizio finale è più affidabile e meno soggetto al caso.

Si può spingere all'estremo: un blocco per ogni singolo esempio, cioè tanti
compiti quanti sono i dati. Sembra il giudizio più solido di tutti, e non lo è.
Quello che il modello ha studiato prima di un compito e prima del successivo
cambia di un esempio soltanto, quindi i giudizi si somigliano tutti, e la media
di tanti giudizi che si somigliano è poco più stabile di uno solo. In più il
modello va riaddestrato una volta per esempio, e con centomila esempi il conto
non sta in piedi. Cinque o dieci blocchi sono il punto in cui la spesa vale il
guadagno.

Tutto questo regge su una condizione che salta più spesso di quanto sembri: le
domande dei cinque compiti devono essere davvero diverse fra loro. Se nel
mucchio ci sono dieci varianti quasi identiche dello stesso esercizio, il
sorteggio ne manda qualcuna nel compito e qualcuna nelle pagine da studiare: lo
studente ritrova nel compito quello che ha appena letto, e prende dieci senza
sapere niente. Con le risposte giuste tirate a monetina, dove non c'è proprio
nulla da imparare e chiunque ne azzecca la metà, un giudizio costruito così
arriva a dire che il modello non sbaglia mai. Il numero che ne esce non vuol
dire niente, e a guardarlo non si vede la differenza.

Succede tutte le volte che più righe raccontano lo stesso soggetto: dieci
visite dello stesso paziente, dieci fotogrammi dello stesso video. Il rimedio è
tenere insieme la famiglia, tutte le righe di un soggetto nello stesso blocco,
così che quando un soggetto fa da giudice non sieda anche fra i banchi.

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
distorta, perché ogni modello studia su $m-1$ esempi, ma costosa e con varianza
più alta della $k$-fold con $k<m$ {cite}`james2023introduction`, perché media
$m$ modelli addestrati su insiemi quasi identici e quindi fortemente correlati
fra loro, e la media di quantità fortemente correlate si stabilizza poco: con
correlazione $\rho$ fra i termini la sua varianza non scende sotto
$\rho\,\sigma^2$, qualunque sia il numero dei termini. Valori $k=5$ o $k=10$
offrono il miglior compromesso fra distorsione, costo computazionale e
stabilità della stima. Resta da dire che cosa stimino. $\text{CV}_k$ approssima
bene l'errore atteso sui possibili insiemi di addestramento, e male l'errore
del modello addestrato proprio sui nostri dati, con cui nelle simulazioni di
ESL risulta perfino debolmente anticorrelata {cite}`hastie2009elements`; e
siccome ogni modello vede $(k-1)m/k$ esempi, la stima è pessimista per il
modello finale, addestrato su tutti gli $m$. Perché stimi qualcosa, poi, ogni
passo che guarda i dati deve stare dentro il ciclo. Il controesempio classico è
di ESL: cinquanta esempi, cinquemila colonne di puro rumore, le cento più
correlate con l'etichetta scelte guardando tutti i dati, e un $1$-NN valutato
in cross-validation; l'errore stimato è del $3\%$, quello vero del $50\%$.
Selezione delle colonne, riscalatura, imputazione e ricampionamento vanno
quindi in una `Pipeline`, che scikit-learn riaddestra da capo in ogni fold.

E quando più configurazioni risultano a pari merito dentro l'incertezza della
stima, la convenzione per decidere è la **regola dell'errore standard**: si
tiene la più semplice fra le configurazioni il cui errore sta entro un errore
standard dal minimo {cite}`james2023introduction`. L'errore standard di una
stima cross-validata vale la dispersione fra i $k$ giri divisa per $\sqrt{k}$,
quindi con cinque blocchi meno della metà: usare la dispersione al suo posto
allarga la fascia di un fattore $2{,}2$ e fa scegliere un modello più semplice
del dovuto. È il rasoio di Occam applicato a una
classifica che si sa incerta.

Un'ipotesi va però dichiarata, perché è quella che regge tutto il
ragionamento: $\text{CV}_k$ stima l'errore a patto che le righe siano
scambiabili, cioè che partizionarle a caso produca fold indipendenti fra
loro. Se più righe descrivono lo stesso soggetto (più visite dello stesso
paziente, più eventi dello stesso utente, più fotogrammi dello stesso video),
il rimescolamento mette quasi-duplicati sia in training sia in validation, e il
modello ritrova in validation ciò che ha già visto. Il risultato è una stima
priva di significato, non solo un po’ ottimista, e non dà nessun
segnale d'allarme. Il conto, con duecento soggetti, dieci misure quasi
identiche ciascuno e un'etichetta assegnata a caso a ogni soggetto (quindi non
c'è niente da imparare, e la verità è $0{,}50$), fatto con un classificatore
capace di memorizzare come il $k$-NN a un vicino:

```python
import numpy as np
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

rng = np.random.default_rng(0)
soggetto = np.repeat(np.arange(200), 10)              # dieci righe per soggetto
centro = rng.normal(size=(200, 5))
X_g = centro[soggetto] + rng.normal(0, 0.01, (2000, 5))  # quasi identiche
y_g = rng.integers(0, 2, 200)[soggetto]   # a caso: non c'è niente da imparare
uno = KNeighborsClassifier(n_neighbors=1)
mescolata = cross_val_score(uno, X_g, y_g,
                            cv=KFold(5, shuffle=True, random_state=0))
per_soggetto = cross_val_score(uno, X_g, y_g, cv=GroupKFold(5), groups=soggetto)
print(f"5-fold mescolata:    {mescolata.mean():.3f}")
print(f"5-fold per soggetto: {per_soggetto.mean():.3f}")
```

```text
5-fold mescolata:    1.000
5-fold per soggetto: 0.543
```

La validazione mescolata dichiara un classificatore perfetto; raggruppata per
soggetto torna a $0{,}543$, dentro l'oscillazione che duecento soggetti
consentono attorno a $0{,}5$. In questi casi i fold vanno costruiti per
soggetto (`GroupKFold`, `GroupShuffleSplit`); se invece le righe sono ordinate
nel tempo vale il discorso dei {doc}`dati che cambiano
</MachineLearning/dati-che-cambiano>`, cioè `TimeSeriesSplit` e non un
rimescolamento.

`````

La riga finale di {numref}`fig-cross-validation` è la parte che si tende a
buttare via: non la media dei cinque punteggi, ma la loro dispersione, cioè
di quanto i cinque giri si discostano dalla media. In statistica la si riassume
in un numero, la *deviazione standard*: quanto, in media, un giro si scosta dal
risultato medio.

Serve a non prendere per differenze quelle che sono oscillazioni. Poniamo che
un modello faccia $0{,}81$ e un altro $0{,}83$, e che i cinque giri di ciascuno
ballino di $\pm 0{,}05$: quei due centesimi di scarto sono più piccoli del
ballo, e a rifare la divisione in blocchi la classifica potrebbe benissimo
capovolgersi. Fra quei due modelli, semplicemente, la cross-validation non sa
scegliere, e dire il contrario è dare un significato al caso.

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge

# il test resta da parte fin dall'inizio, non lo tocchiamo più
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

modello = Ridge(alpha=1.0)   # alpha: quanto si fa pagare la complessità
scores = cross_val_score(modello, X_train, y_train, cv=5,
                         scoring="neg_mean_squared_error")  # 5-fold CV
print(f"errore medio di validazione: {-scores.mean():.3f}")
print(f"quanto ballano i cinque giri: {scores.std():.3f}")
```

```text
errore medio di validazione: 2.248
quanto ballano i cinque giri: 0.217
```

Il ballo è $0{,}217$ su una media di $2{,}248$, cioè poco meno di un decimo.
Su questi dati, quindi, due modelli che si scostassero di qualche centesimo di
errore la cross-validation non li saprebbe ordinare, e la sola media non lo
direbbe.

## Mettere un freno: la regolarizzazione

Un modo diretto per contrastare l'overfitting è impedire al modello di diventare
troppo "estremo". Il trucco è furbo, e per capirlo conviene ricordare che
addestrare vuol dire rendere più piccolo possibile un numero, quello che
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
manopole del modello, e ogni punto del piano è una scelta possibile dei due
numeri. Far pagare un prezzo alla spesa in pesi e mettere un tetto a quella
spesa sono due modi di dire la stessa cosa: a ogni prezzo corrisponde il tetto
che porta alla stessa soluzione, e viceversa (chi paga un prezzo per ogni unità
di spesa finisce per spendere una certa cifra, e con quella cifra come tetto
avrebbe scelto lo stesso). Il tetto conviene perché si disegna: è una regione
attorno all'origine, e il modello deve restare dentro. Se la spesa si conta
sommando i valori assoluti (la L1), il recinto è un rombo con le punte sugli
assi: per star dentro basta che $|w_1| + |w_2|$ non superi il budget, e i due
estremi sono spendere tutto su un peso solo, che sono appunto le punte. Se si
conta sommando i quadrati (la L2) il recinto è un cerchio.

E l'errore? Fuori dal recinto l'errore ha la forma di una conca, con il
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

Un anello che si allarga incontra un rombo in una punta, come mostra
{numref}`fig-l1-l2`, e le punte del rombo stanno sugli assi, cioè in punti dove
uno dei due pesi vale esattamente zero. Succede tanto più spesso quanto più
stretto è il budget, ed è la ragione per cui la L1 azzera di più quando il
freno è tirato di più. Un cerchio invece non ha punte, e il primo contatto
cade in un posto qualunque del bordo, dove entrambi i pesi sono piccoli ma
nessuno è nullo. Ecco perché sommare i valori assoluti seleziona le
caratteristiche e sommare i quadrati no: la ragione sta tutta nella forma del
recinto.

`````{tab} Elementare

La regolarizzazione è un budget di spesa sui pesi del modello. Senza limiti, per
passare su ogni punto la curva deve piegarsi di scatto, e le pieghe brusche si
fanno solo con pesi enormi di segno opposto, come $+1000$ e $-999$, che quasi si
annullano a vicenda: è così che nasce la curva contorta. Con un tetto alla spesa
totale il modello deve essere sobrio, e le curve sobrie sono più morbide. I due
modi di contare la spesa portano un nome ciascuno.

- **Ridge** (la L2): si paga la *somma dei quadrati* dei pesi. Li rimpicciolisce
  tutti dolcemente, senza azzerarne nessuno. Il quadrato punisce pochissimo chi
  è già piccolo: portare un peso da $0{,}1$ a $0$ fa risparmiare $0{,}01$, e per
  un risparmio così non si rinuncia a un peso che serve ancora un po’.
- **Lasso** (la L1): si paga la *somma dei valori assoluti*. Qui l'ultimo
  centesimo costa quanto il primo, quindi azzerare del tutto un peso che non
  serve conviene sempre: le caratteristiche inutili spariscono dalla tabella. È
  lo stesso fatto che il rombo con le punte sugli assi racconta con la
  geometria.

Quanto stringere il budget è la manopola con cui si sceglie fra i due modi di
sbagliare visti al poligono. Tirata a fondo, con la spesa quasi vietata, tutti i
pesi vanno a zero e il modello risponde sempre la stessa cosa: stabile e storto.
Lasciata andare, si torna alla curva che si contorce. Il valore giusto sta in
mezzo e non si indovina: se ne provano parecchi, e a dire quale tenere sono i
cinque compiti in classe della cross-validation.

Un budget ha senso se si paga per le cose giuste, e con la stessa moneta per
tutte. Non si paga per la quota fissa che il modello somma a ogni previsione:
chi prevede le temperature in gradi Kelvin, che sono i Celsius più 273, ha
bisogno di una quota fissa più alta di 273, e se la pagasse il modello
cambierebbe per una semplice scelta di unità. E la moneta è la stessa solo se le
colonne sono misurate sulla stessa scala. Un reddito di $30\,000$ euro porta
$30$ punti con un peso di $0{,}001$, una percentuale del $5$ ne vuole uno di $6$
per portarne altrettanti, e allo stesso prezzo per peso il freno lascerebbe in
pace il reddito e strangolerebbe la percentuale. Per questo, prima, ogni colonna
si misura in quanto si scosta dal suo solito.

Il Lasso ha anche un capriccio. Fra due colonne gemelle, che dicono quasi la
stessa cosa, ne tiene una e butta l'altra, e quale delle due è quasi un
sorteggio: cambia il campione e cambia la scelta. Con il quadrato, invece,
dividere un peso a metà fra le due gemelle costa meno che darlo tutto a una
($0{,}5^2 + 0{,}5^2$ fa $0{,}5$, contro $1$), e allora, facendo pagare un po’ in
un modo e un po’ nell'altro, le gemelle entrano o escono insieme.

A volte le colonne vanno in gruppo per costruzione. Il colore di un'auto, con
quattro valori possibili, entra nel modello come quattro colonne sì o no, una
per colore, e il Lasso le tratta una per una: può tenere «rosso» e buttare
«verde», come se il colore contasse per le auto rosse e non per le verdi. Lo
stesso colore si può scrivere anche con tre colonne sole, togliendo quella del
verde: un'auto verde ha allora tutte e tre le colonne a no, e il peso del rosso
dice quanto una rossa vale più di una verde. Il verde fa da zero, come lo zero
di un termometro, e gli altri colori si misurano da lì. I dati sono gli stessi,
eppure scritti così il Lasso può buttare colori diversi. La conclusione
dipendeva dalla scrittura, non dai dati.

Il **lasso a gruppi** fa pagare ogni gruppo come un pacchetto, secondo la sua
lunghezza, che si trova come l'ipotenusa di Pitagora, con quanti cateti servono:
si sommano i quadrati dei pesi e si prende la radice. Due pesi da $0{,}3$ e
$0{,}4$ fanno $0{,}09 + 0{,}16 = 0{,}25$, cioè un pacchetto lungo $0{,}5$. Il
conto toglie poi a ogni pacchetto un pezzo fisso di lunghezza, come il Lasso
toglie a ogni peso lo stesso tanto: il pacchetto più corto di quel pezzo
sparisce tutto, quello più lungo resta, ed è per questo che entra intero o resta
fuori intero. Dentro, i pesi si spartiscono la spesa come nel Ridge, e nessuno
viene azzerato da solo. Il pezzo tolto, però, non è uguale per tutti: cresce con
la radice di quanti pesi il pacchetto contiene, e per quattro pesi è il doppio.
Anche una colonna inutile riceve dal caso un pesetto, e quattro pesetti fanno un
pacchetto più lungo di uno solo: con lo stesso pezzo per tutti, un pacchetto di
colonne inutili entrerebbe soltanto perché è grosso.

I pacchetti li decide chi scrive il modello, e il metodo li prende per buoni.
Chi mette nello stesso pacchetto il colore e l'età dell'auto se li vede entrare
e uscire insieme, anche se conta soltanto il colore. E se il pacchetto dell'età
contiene l'età, il suo quadrato e il suo cubo, e conta solo l'età, chi vuole
poter scartare il quadrato e il cubo aggiunge un po’ del prezzo del Lasso: il
pacchetto entra, ma dentro i pesi inutili possono ancora andare a zero.

`````

`````{tab} Superiore

Si aggiunge alla loss $\mathcal{L}(\theta)$ un termine di penalità pesato da un
iperparametro $\lambda \ge 0$ che regola l'intensità del freno. Per la
regressione Ridge (norma $\ell_2$):

$$
\mathcal{L}_{\text{Ridge}}(\theta)
= \frac{1}{m}\sum_{i=1}^{m}\big(\hat{y}^{(i)}-y^{(i)}\big)^2
+ \lambda\sum_{j=1}^{n}\theta_j^{2},
$$

per il Lasso (norma $\ell_1$):

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

Con le colonne centrate, così che l'intercetta resti fuori dalla penalità,
Ridge ha soluzione in forma chiusa,

$$
\hat{\theta}_{\text{Ridge}} = \big(\mathbf{X}^\top\mathbf{X} + m\lambda\,\mathbf{I}\big)^{-1}\mathbf{X}^\top\mathbf{y},
$$

dove il fattore $m$ viene dalla media nella loss. La matrice è invertibile per
ogni $\lambda>0$, anche con colonne collineari o con più colonne che esempi,
cioè proprio dove le equazioni normali degeneravano. Con colonne ortogonali,
$\mathbf{X}^\top\mathbf{X} = m\,\mathbf{I}$, le due penalità si leggono
coefficiente per coefficiente a partire dalla stima dei minimi quadrati
$\hat{\theta}_j$: Ridge la divide, $\hat{\theta}_j/(1+\lambda)$, e non la
azzera mai; Lasso la accorcia di una quantità fissa,
$\operatorname{sign}(\hat{\theta}_j)\,\big(|\hat{\theta}_j| - \lambda/2\big)_+$
(il *soft thresholding*), e manda a zero esatto ogni coefficiente sotto
$\lambda/2$ {cite}`hastie2009elements`. Le due penalità sono anche stime MAP:
con rumore $\mathcal{N}(0,\sigma^2)$ e prior $\theta_j \sim \mathcal{N}(0,\tau^2)$
si ottiene Ridge con $\lambda = \sigma^2/(m\tau^2)$, e con un prior di Laplace
si ottiene il Lasso, che non ha forma chiusa e in scikit-learn si risolve per
discesa coordinata.

Due cose le formule le dicono in silenzio. La prima è che l'indice $j$ corre da
$1$ a $n$, cioè sulle sole caratteristiche: l’intercetta non è penalizzata. Se
lo fosse, il modello dipenderebbe dall'origine scelta per $y$, e sommare una
costante a tutte le etichette (per esempio misurarle in gradi Kelvin invece che
in Celsius, cioè aggiungere $273{,}15$) cambierebbe la soluzione, il che non ha
senso. La seconda è che la penalità mette sullo stesso piano pesi che vivono su
scale diverse, e quindi presuppone feature standardizzate: il peso che
moltiplica un reddito in euro è piccolo per forza, e la penalità lo lascerebbe
in pace mentre schiaccia quello di una percentuale. Vale qui la stessa
avvertenza del k-NN e delle SVM, con la differenza che qui è meno visibile,
perché un modello mal regolarizzato funziona comunque, solo peggio: `Ridge` e
`Lasso` non standardizzano da soli, e vanno messi dietro uno `StandardScaler`
dentro una `Pipeline`, la catena di passaggi che scikit-learn tratta come se
fosse un modello solo.

L’**Elastic Net** somma le due penalità,
$\lambda\big(\alpha\sum_j|\theta_j| + \tfrac{1-\alpha}{2}\sum_j\theta_j^2\big)$,
e non è un compromesso pigro: rimedia a un difetto preciso del Lasso. Fra due
feature fortemente correlate il Lasso ne tiene una sola, scelta in modo
instabile (basta cambiare il campione perché scelga l'altra), mentre il termine
$\ell_2$ le fa entrare o uscire insieme, il cosiddetto *grouping effect*.
Con feature molte e correlate, che è il caso normale sui dati reali, è la
scelta di partenza più sensata delle due pure.

Un avvertimento sulla lettera $\alpha$, che qui fa due mestieri
diversi. Nella formula dell'Elastic Net è il **rapporto di miscela** fra le due
penalità ($\alpha = 1$ è Lasso puro, $\alpha = 0$ è Ridge puro) e non ha niente
a che vedere con la loro intensità, che resta $\lambda$. Nel codice, invece,
l'argomento `alpha` di `Ridge`, `Lasso` ed `ElasticNet` fa il mestiere
dell’**intensità**, cioè del nostro $\lambda$, mentre la miscela lì si chiama
`l1_ratio`. Fa il mestiere, però non è lo stesso numero, e la ragione è dove
ciascuna libreria mette la divisione per il numero di esempi: la loss qui
è mediata sugli esempi e la penalità no, mentre `Ridge` non media affatto e
`Lasso` divide per $2m$.
A parità di soluzione, $\lambda = \texttt{alpha}/m$ per la prima e
$\lambda = 2\,\texttt{alpha}$ per la seconda. Sono due tradizioni che si sono incrociate su una lettera sola:
conviene guardare che cosa fa il parametro, non come si chiama.

Quando i gruppi sono noti in anticipo (le colonne indicatrici di una variabile
categorica, i termini di un polinomio nella stessa variabile) il Lasso li
spezza: sulle indicatrici tiene alcuni livelli e ne azzera altri, e quali
dipende dalla codifica, per esempio dal livello scelto come riferimento, che è
una convenzione e non un fatto dei dati. Il **group lasso**, proposto da Bakin
{cite}`bakin1999adaptive` e studiato da Yuan e Lin {cite}`yuan2006model`, impone
i gruppi, con i coefficienti divisi in blocchi $\theta_g$ di $p_g$ elementi e
$\mathbf{X}_g$ le colonne del blocco $g$:

$$
\mathcal{L}_{\text{gruppi}}(\theta)
= \frac{1}{m}\sum_{i=1}^{m}\big(\hat{y}^{(i)}-y^{(i)}\big)^2
+ \lambda\sum_{g}\sqrt{p_g}\,\lVert\theta_g\rVert_2 .
$$

La norma $\ell_2$ non elevata al quadrato ha uno spigolo nell'origine di ogni
blocco, e lo spigolo annulla il blocco intero; dentro il blocco la norma è
rotonda e non azzera nessuna componente da sola. Il fattore $\sqrt{p_g}$ tiene
alla pari i blocchi di taglia diversa. Un blocco resta a zero finché
$\tfrac{2}{m}\lVert\mathbf{X}_g^\top\mathbf{r}_{-g}\rVert_2$ non supera
$\lambda\sqrt{p_g}$, dove $\mathbf{r}_{-g}$ è il residuo lasciato dagli altri
blocchi; se le colonne del blocco non contano, il membro di sinistra cresce per
puro caso come $\sqrt{p_g}$, e senza il fattore un blocco grande entrerebbe più
spesso solo perché è grande. Con colonne ortonormali la condizione, elevata al
quadrato, confronta con una soglia la somma dei quadrati spiegata dal blocco
divisa per $p_g$, come il test $F$ dell'analisi della varianza, ed è la ragione
per cui Yuan e Lin scelgono questo fattore. E $p_g$ coefficienti tutti uguali ad
$a$ hanno norma $a\sqrt{p_g}$, quindi pagano $p_g a$, quanto il Lasso li farebbe
pagare uno per uno.

Se le colonne di ogni blocco sono ortonormali nella scala della media
($\mathbf{X}_g^\top\mathbf{X}_g = m\,\mathbf{I}$) e i blocchi sono ortogonali
fra loro, la soluzione si scrive blocco per blocco, a partire dalla stima dei
minimi quadrati $\hat{\theta}_g$, come
$\big(1 - \lambda\sqrt{p_g}/(2\lVert\hat{\theta}_g\rVert_2)\big)_+\,\hat{\theta}_g$:
è il *soft thresholding* del Lasso applicato alla lunghezza del blocco invece
che al singolo coefficiente. L'ortonormalità dentro i blocchi, che Yuan e Lin
assumono in tutto il lavoro, fa più che semplificare i conti. La norma
$\lVert\theta_g\rVert_2$ non cambia se la base del blocco ruota, ma cambia sotto
un cambio di base qualsiasi, e solo con blocchi ortonormalizzati la soluzione
dipende dallo spazio generato dalle colonne e non dai contrasti con cui si è
scritta la variabile; una variabile a quattro livelli ha tre gradi di libertà, e
il suo $p_g$ è $3$. Su blocchi non ortonormali il metodo prende ancora il blocco
intero o niente, ma il $\lambda$ a cui lo prende torna a dipendere dalla
codifica. Con la sola ortonormalità dentro i blocchi, la stessa formula
applicata a turno a ogni blocco, sul residuo lasciato dagli altri, è l'algoritmo
di Yuan e Lin; nel caso generale la si usa come passo di un metodo del gradiente
prossimale, con soglia $\lambda\sqrt{p_g}$ moltiplicata per il passo.

Il punto di rottura è la partizione, che il metodo prende per vera: un blocco
sbagliato entra o esce intero, e se in un blocco conta un solo termine del
polinomio entra lo stesso il blocco intero. Lo *sparse group lasso*
{cite}`simon2013sparse` ci rimedia mescolando le due penalità con un rapporto di
miscela, come l'Elastic Net,

$$
\lambda\Big((1-\alpha)\sum_g\sqrt{p_g}\,\lVert\theta_g\rVert_2
+ \alpha\lVert\theta\rVert_1\Big),
$$

così che dentro un blocco acceso le singole componenti si possano ancora
azzerare. Sulle indicatrici di una variabile categorica il rimedio va letto con
cautela, perché lì uno zero dentro il blocco dice che quel livello non si
distingue dal livello di base, e quale sia la base dipende di nuovo dalla
codifica.

`````

Il blocco costruisce una risposta che dipende da una variabile numerica e dal
colore (quattro valori, quindi quattro colonne sì/no), mentre un'altra variabile
numerica e la regione di provenienza (altre quattro colonne) non contano. Poi
guarda, per trenta intensità del freno, se il Lasso prende qualche gruppo a
metà, e quali colori tiene a un'intensità intermedia; e prova a tre intensità il
lasso a gruppi, scritto in poche righe come discesa del gradiente seguita
dall'accorciamento di ogni blocco.

```python
import numpy as np
from sklearn.linear_model import Lasso

rng = np.random.default_rng(0)
m = 400
def a_colonne(etichette, k):
    """Una variabile a k valori spezzata in k colonne sì/no, centrate (a media zero)."""
    D = np.eye(k)[etichette]
    return D - D.mean(axis=0)
colore = rng.integers(0, 4, m)                    # conta: rosso, verde, blu, giallo
regione = rng.integers(0, 4, m)                   # non conta
x1, x2 = rng.standard_normal(m), rng.standard_normal(m)
X = np.column_stack([x1, x2, a_colonne(colore, 4), a_colonne(regione, 4)])
gruppi = [[0], [1], [2, 3, 4, 5], [6, 7, 8, 9]]  # un gruppo per variabile
nomi = ["x1", "x2", "colore", "regione"]
y = 1.5 * x1 + np.array([0.0, 1.0, -1.0, 0.5])[colore] + rng.normal(0, 1, m)
y = y - y.mean()

def lasso_a_gruppi(X, y, lam, passi=3000):
    """Discesa del gradiente in cui, dopo ogni passo, ogni gruppo si accorcia per
    intero, e sotto una soglia sparisce (il block soft-thresholding): la penalità
    è lam per la radice della taglia del gruppo per la lunghezza dei suoi pesi."""
    theta = np.zeros(X.shape[1])
    # il passo è l'inverso della costante di Lipschitz del gradiente, 2 ||X||^2 / m:
    # con un passo così la discesa non diverge
    passo = len(y) / (2 * np.linalg.norm(X, 2) ** 2)
    for _ in range(passi):
        z = theta - passo * 2 * X.T @ (X @ theta - y) / len(y)
        for g in gruppi:
            norma = np.linalg.norm(z[g])
            soglia = passo * lam * np.sqrt(len(g))       # la penalità, scalata dallo stesso passo
            theta[g] = 0.0 if norma <= soglia else (1 - soglia / norma) * z[g]
    return theta

def a_meta(theta):
    """I gruppi di più colonne presi a metà: qualche colonna dentro, qualcuna fuori."""
    return [nomi[i] for i, g in enumerate(gruppi) if len(g) > 1
            and 0 < np.count_nonzero(theta[g]) < len(g)]

presi_a_meta = set()
for alpha in np.logspace(-3, 0, 30):              # trenta intensità, dal freno lieve al forte
    theta = Lasso(alpha=alpha, fit_intercept=False).fit(X, y).coef_
    presi_a_meta |= set(a_meta(theta))
print("Lasso: gruppi presi a metà per qualche intensità:", sorted(presi_a_meta))
theta = Lasso(alpha=0.1, fit_intercept=False).fit(X, y).coef_
colori = ["rosso", "verde", "blu", "giallo"]
print("Lasso con alpha 0.1, colori tenuti:",
      [c for c, t in zip(colori, theta[2:6]) if t != 0])
for lam in (0.01, 0.1, 1.0):
    theta = lasso_a_gruppi(X, y, lam)
    dentro = ", ".join(f"{nomi[i]} {np.linalg.norm(theta[g]):.2f}"   # la lunghezza del gruppo
                       for i, g in enumerate(gruppi) if np.any(theta[g] != 0))
    print(f"lasso a gruppi, lambda {lam}: dentro {dentro}; a metà {a_meta(theta)}")
```

```text
Lasso: gruppi presi a metà per qualche intensità: ['colore', 'regione']
Lasso con alpha 0.1, colori tenuti: ['verde', 'blu']
lasso a gruppi, lambda 0.01: dentro x1 1.56, x2 0.02, colore 1.49, regione 0.10; a metà []
lasso a gruppi, lambda 0.1: dentro x1 1.52, colore 1.13; a metà []
lasso a gruppi, lambda 1.0: dentro x1 1.08; a metà []
```

Il Lasso, per qualche intensità del freno, tiene alcune colonne del colore e
ne butta altre, e fa lo stesso con la regione, che non conta affatto. Con
`alpha` a $0{,}1$ tiene il verde e il blu, una selezione che dice «il verde e il
blu contano, il rosso e il giallo no», mentre nei dati ogni colore sposta la
risposta di un tanto suo. Il lasso a gruppi, per costruzione, non prende mai un
gruppo a metà. Con il freno lieve tiene dentro tutto, i due gruppi che non
contano con lunghezze piccole ($0{,}02$ e $0{,}10$); al crescere di $\lambda$
escono prima quei due, poi il colore, e ogni gruppo esce intero. La lunghezza di
chi resta cala a ogni stretta del freno: è il restringimento, lo stesso del
Lasso, applicato al blocco. Il blocco non ortonormalizza i gruppi, per restare
vicino alle colonne sì/no, e al colore dà quattro colonne dove ne basterebbero
tre: la regola del tutto o niente non ne risente, le intensità a cui i gruppi
escono sì.

### A salti o a poco a poco: scegliere le colonne

Prima dei freni continui le colonne si sceglievano a salti: si provano dei
sottoinsiemi e si tiene il migliore. La **selezione del sottoinsieme migliore**
(*best subset selection*) li prova tutti, e con $p$ colonne sono $2^p$ (ogni
colonna dentro o fuori, due scelte per ciascuna), più di un milione già con
venti colonne; la **selezione in avanti** (*forward stepwise selection*) ne
costruisce una catena, aggiungendo a ogni passo la colonna che riduce di più
l'errore, e quante tenerne lo decide la cross-validation. Il confronto con il
Lasso dice quando il freno continuo conviene, e quando no.

`````{tab} Elementare

Scegliere a salti è come convocare una squadra: per ogni giocatore si decide
dentro o fuori, e chi è dentro gioca tutta la partita. Il freno continuo del
Lasso concede invece a ciascuno dei minuti, e li toglie un po' alla volta.

Il difetto dei salti è che una convocazione cambia per poco. Basta qualche
esempio diverso perché fra due giocatori simili entri l'altro, e con lui cambia
di colpo il modello intero, come una formazione che per un solo cambio
ridistribuisce i ruoli di tutti: la scelta a salti è nervosa, e il nervosismo si
paga in errore sui dati nuovi. Anche il Lasso, da un campione all'altro, può
cambiare quale di due colonne gemelle tiene, ed è il suo capriccio; ma quando i
dati si spostano di poco i minuti passano dall'una all'altra poco alla volta, e
siccome le gemelle dicono quasi la stessa cosa la previsione si sposta appena.
Può ballare l'elenco dei convocati, mentre la previsione resta quasi ferma.

Ma il freno ha un prezzo suo, perché fa due mestieri con una manopola sola:
toglie minuti a tutti e decide chi resta in panchina. Tirato quanto serve per
lasciare fuori le riserve, toglie troppi minuti ai titolari; allentato per far
giocare i titolari, lascia entrare qualche riserva che non servirebbe. C'è poi
un costo nascosto nella convocazione, ed è la trappola delle mille persone che
lanciano la moneta: chi prova migliaia di squadre sugli stessi dati ne trova
sempre una che su quei dati va bene per caso, e il merito che le si attribuisce
è gonfiato dalla ricerca stessa, anche se alla fine i convocati sono pochi,
perché a pesare non è quanti giocano ma quante squadre si sono provate per
sceglierli. Quando i dati sono chiari, con poco rumore, la squadra convocata
bene è la più precisa, perché chi gioca gioca a pieno; quando il rumore è tanto
conviene il freno, perché una convocazione fatta sul rumore sbaglia di più.
Nessuno dei due vince sempre, e la via di mezzo se la cava bene dappertutto: si
convoca con il freno, e poi ai convocati lo si allenta.

`````

`````{tab} Superiore

La selezione del sottoinsieme migliore risolve
$\min_\theta \lVert \mathbf{y} - \mathbf{X}\theta\rVert^2$ con il vincolo
$\lVert\theta\rVert_0 \le k$, il numero di coefficienti non nulli: un problema
combinatorio, NP-difficile in generale, che la programmazione intera mista
risolve con ottimalità certificata quando le colonne sono centinaia e gli esempi
migliaia, in minuti per ogni $k$ secondo gli autori {cite}`bertsimas2016best`,
spesso in un'ora o più per chiudere il certificato secondo chi ha rifatto le
prove {cite}`hastie2020best`. La selezione in avanti lo approssima in modo
avido, con $O(p^2)$ adattamenti ai minimi quadrati lungo il cammino, e il Lasso
ne è il rilassamento convesso, con $\lVert\theta\rVert_1$ al posto di
$\lVert\theta\rVert_0$. Breiman ha mostrato, con un argomento euristico e con
simulazioni, che la selezione del sottoinsieme è *instabile* (cambiare pochi
esempi può cambiare il sottoinsieme scelto) mentre la regressione ridge è
stabile, e che l'instabilità si paga nella scelta della complessità: con la
sfera di cristallo, cioè scegliendo $k$ sull'errore vero, il sottoinsieme
batteva spesso la ridge, e perdeva il vantaggio quando $k$ andava scelto sui
dati {cite}`breiman1996heuristics`. Il Lasso sta dalla parte della ridge: a
$\lambda$ fissato la sua previsione $\mathbf{X}\hat{\theta}$ è una funzione
continua di $\mathbf{y}$, mentre quella del sottoinsieme migliore e della
selezione in avanti, a $k$ fissato, salta quando $\mathbf{y}$ attraversa il
confine fra due insiemi attivi {cite}`hastie2020best`.

Il confronto sistematico di Hastie, Tibshirani e Tibshirani
{cite}`hastie2020best`, nato per verificare le simulazioni di Bertsimas e
colleghi in cui il sottoinsieme migliore vinceva sempre, ha precisato il
quadro. A rapporto segnale/rumore alto la selezione del sottoinsieme migliore e
quella in avanti, che si comportano in modo simile, battono il Lasso, che per
non restringere troppo i coefficienti veri sceglie un $\lambda$ piccolo e
accetta qualche falso positivo; a rapporto basso vince il Lasso. Il sorpasso
cade attorno a $1{,}2$ con cento esempi e dieci colonne, attorno a $0{,}4$ con
cinquecento esempi e cento colonne, e gli autori avvertono che su dati
osservazionali già un rapporto di $1$, cioè un modello che spiega metà della
varianza di $y$, è raro, e uno di $6$ è inaudito. Il *relaxed lasso*, che usa
il Lasso per scegliere e poi restringe meno, è competitivo dappertutto. C'è
anche un costo nascosto nella scelta a salti. I gradi di libertà effettivi di
un modello (la somma delle covarianze fra ciascuna previsione e la sua
etichetta, divisa per $\sigma^2$, che per i minimi quadrati su $k$ colonne
fissate vale esattamente $k$) superano di molto i $k$ coefficienti che restano
quando le colonne le ha scelte una ricerca, perché la ricerca stessa ha
guardato i dati; quelli del Lasso valgono invece il numero atteso di
coefficienti non nulli {cite}`hastie2020best`.

`````

Il blocco mette alla prova le due strade su cento esempi con venti colonne, di
cui cinque contano, correlate fra loro tanto più quanto sono vicine
($0{,}5^{|i-j|}$), a tre rapporti segnale/rumore (quanto è sparpagliata la
parte di $y$ che le colonne spiegano, divisa per quanto lo è il rumore: le
varianze dei dati, non quella del modello), trenta campioni per ciascuno. La
selezione in avanti sceglie quante colonne tenere con una cross-validation a
cinque blocchi, come fa `LassoCV` per l'intensità del freno, e nessuno dei due
stima un'intercetta, che i dati non hanno. L'errore è misurato su dati nuovi e
diviso per la varianza del rumore, quindi $1$ è il meglio possibile, perché il
rumore nessun modello lo può prevedere; accanto, il blocco stampa lo scarto fra
i due con il suo errore standard, e quante colonne tiene in media ciascuno.

```python
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold

m, p = 100, 20
S = 0.5 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))   # correlazioni fra colonne vicine
beta = np.r_[np.ones(5), np.zeros(p - 5)]                          # contano le prime cinque

def dati(snr, rng):
    """m esempi per stimare, 2000 per misurare, rapporto segnale/rumore snr."""
    # Cholesky: trasforma colonne indipendenti in colonne con le correlazioni di S
    X = rng.standard_normal((m + 2000, p)) @ np.linalg.cholesky(S).T
    sigma = np.sqrt(beta @ S @ beta / snr)
    y = X @ beta + sigma * rng.standard_normal(m + 2000)
    return X[:m], y[:m], X[m:], y[m:], sigma

def minimi_quadrati(X, y, colonne):
    return np.linalg.lstsq(X[:, colonne], y, rcond=None)[0]

def in_avanti(X, y):
    """L'ordine in cui le colonne entrano: a ogni passo quella che riduce di più l'errore."""
    dentro, fuori = [], list(range(p))
    for _ in range(p):
        errori = [((y - X[:, dentro + [j]] @ minimi_quadrati(X, y, dentro + [j])) ** 2).sum()
                  for j in fuori]
        dentro.append(fuori.pop(int(np.argmin(errori))))
    return dentro

def in_avanti_cv(X, y):
    """In avanti, con il numero di colonne scelto dalla cross-validation."""
    errore = np.zeros(p)
    for tr, va in KFold(5).split(X):
        ordine = in_avanti(X[tr], y[tr])
        for k in range(1, p + 1):
            b = minimi_quadrati(X[tr], y[tr], ordine[:k])
            errore[k - 1] += ((y[va] - X[va][:, ordine[:k]] @ b) ** 2).sum()
    colonne = in_avanti(X, y)[:int(np.argmin(errore)) + 1]
    return colonne, minimi_quadrati(X, y, colonne)

print("rapporto   in avanti   Lasso   scarto           colonne tenute")
for snr in (0.25, 1.0, 6.0):
    avanti, lasso, n_avanti, n_lasso = [], [], [], []
    for r in range(30):                                  # trenta campioni per ogni rapporto
        X, y, Xt, yt, sigma = dati(snr, np.random.default_rng(r))
        colonne, b = in_avanti_cv(X, y)
        avanti.append(((yt - Xt[:, colonne] @ b) ** 2).mean() / sigma ** 2)
        # senza intercetta, come la selezione in avanti: i dati non ne hanno
        las = LassoCV(cv=5, fit_intercept=False).fit(X, y)
        lasso.append(((yt - las.predict(Xt)) ** 2).mean() / sigma ** 2)
        n_avanti.append(len(colonne))
        n_lasso.append(np.count_nonzero(las.coef_))
    d = np.array(avanti) - np.array(lasso)   # lo scarto, campione per campione
    es = d.std(ddof=1) / np.sqrt(len(d))     # e il suo errore standard
    print(f"{snr:8}{np.mean(avanti):12.2f}{np.mean(lasso):8.2f}",
          f"  {d.mean():+.3f} ± {es:.3f}",
          f"  {np.mean(n_avanti):.1f} contro {np.mean(n_lasso):.1f}")
```

```text
rapporto   in avanti   Lasso   scarto           colonne tenute
    0.25        1.13    1.09   +0.039 ± 0.010   2.2 contro 6.3
     1.0        1.19    1.12   +0.071 ± 0.011   5.1 contro 8.4
     6.0        1.09    1.12   -0.038 ± 0.011   5.6 contro 8.9
```

Con molto rumore e con rumore medio il Lasso sbaglia meno ($1{,}09$ contro
$1{,}13$, $1{,}12$ contro $1{,}19$); con i dati quasi puliti il sorpasso si
rovescia, e la selezione in avanti arriva a $1{,}09$ contro $1{,}12$. Su questi
trenta campioni ogni scarto vale più di tre volte il suo errore standard.
L'ultima colonna dice perché il Lasso perde dove i dati sono chiari: tiene in
media quasi nove colonne, quando quelle che contano sono cinque, mentre la
selezione in avanti ne tiene poco più di cinque. È, in piccolo, quello che
Hastie, Tibshirani e Tibshirani trovano su un confronto molto più ampio: il
freno continuo conviene quando il rumore rende nervosa ogni convocazione, la
scelta a salti quando i dati sono abbastanza chiari da convocare bene.

## Il rasoio di Occam

Sotto tutto questo c'è un principio antico. Nel XIV secolo il frate francescano
**Guglielmo di Occam** enunciò quello che oggi chiamiamo il *rasoio*, riassunto
poi nella formula *entia non sunt multiplicanda praeter necessitatem*, non
moltiplicare le entità oltre il necessario (la frase esatta, per la cronaca,
non compare nei suoi scritti: la coniò un commentatore del Seicento). Tradotto
per noi: a parità di capacità di spiegare i dati, scegli
il modello più semplice.

La regolarizzazione non è altro che il rasoio di Occam scritto in formule: la
manopola $\lambda$ è il prezzo che facciamo pagare alla complessità, così che
il modello la compri solo quando serve davvero. La curva morbida del pannello
centrale di {numref}`fig-overfitting` vince non perché sia la più elaborata, ma
perché è la più semplice tra quelle che rendono conto dei dati. La semplicità,
in machine learning, è ciò che permette di generalizzare.

(sec-doppia-discesa)=

## Quando la U non basta: la doppia discesa

C'è un punto in cui il quadro appena disegnato entra in tensione con la
pratica delle reti profonde, e conviene affrontarlo invece di ignorarlo. La
curva a U di poco fa dice: oltre una certa complessità l'errore sui dati nuovi
risale.

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

La U descrive solo il primo tratto. Oltre il picco, dove il modello
ha appena abbastanza capacità per memorizzare tutto, la curva riscende invece
di continuare a salire.
```

Il punto interessante di {numref}`fig-double-descent` è il picco, non le
discese, e per capirlo serve l'immagine della curva che passa per dei punti.
Il picco sta dove il modello ha esattamente le manopole che servono per
passare per tutti i dati e nemmeno una di più. Di curve così ne esiste una
sola (dieci punti e dieci manopole: il polinomio di nono grado che ci passa è
uno e uno solo), il modello è costretto a prendere quella, e per obbedire a
tutti i punti insieme quella curva fra l'uno e l'altro impazzisce. Appena si
aggiungono manopole, invece, le curve che passano per tutti i punti tornano a
essere infinite, e fra infinite ce n'è anche qualcuna tranquilla: la parte
sorprendente, di cui si parla fra poco, è che l'addestramento tende proprio a
quelle.

`````{tab} Elementare

Qualcuno ha fatto la cosa che i manuali sconsigliavano: ha continuato a
ingrandire il modello *oltre* il punto in cui impara a memoria ogni esempio. E
l'errore sul test, dopo essere risalito come previsto, è tornato a scendere.
Non un caso fortunato: un fenomeno riproducibile, chiamato **doppia discesa**.

La curva, insomma, è una U seguita da una seconda discesa. Il picco sta
esattamente nel punto di **interpolazione**, cioè dove il modello riesce per la
prima volta a passare per tutti i punti (in matematica si dice *interpolare*) e
non gli avanza niente.

E l'addestramento sceglie davvero, fra le infinite curve che passano per tutti
i punti, quella meno tormentata? In buona parte sì, e la ragione sta
nel modo in cui procede. La discesa del gradiente
(la procedura a piccoli passi vista con la retta di best fit) parte da numeri
piccoli, sorteggiati vicino allo zero, e si muove a passettini finché i dati
non tornano; appena tornano, si ferma. Il risultato è che non va mai a cercare
lontano una soluzione strana, se ce n'è una mansueta lì vicino. Avere manopole
in eccesso dà soprattutto questo: la libertà di scegliere una soluzione
gentile.

La gobba, poi, non compare soltanto ingrandendo il modello. Si vede anche
allungando l'addestramento, e perfino aumentando i dati: se il modello sta
vicino al punto di interpolazione, raccogliere altri esempi può fargli fare
peggio, perché lo spinge proprio là dove non ha margine. Ma non compare
sempre: la si vede soprattutto quando fra le risposte giuste ce n'è una quota
sbagliata, e il prezzo messo sui pesi grandi l'appiana.

Ne esce un solo consiglio pratico. Quando l'errore sui dati nuovi ha toccato il
fondo e ha ricominciato a salire, non è detto che si sia già visto il meglio:
ingrandire ancora, qualche volta, ripaga.

`````

`````{tab} Superiore

Il fenomeno è stato descritto sistematicamente da Belkin e colleghi (2019)
{cite}`belkin2019reconciling` e poi sulle reti profonde da Nakkiran e colleghi
(2020) {cite}`nakkiran2020deep`. Tre precisazioni che evitano
di trarne la conclusione sbagliata.

Non è solo la taglia del modello. La doppia discesa si osserva anche
rispetto al *tempo di addestramento* (epoch-wise) e alla *quantità di dati*, e
in quest'ultimo caso produce l'effetto contro-intuitivo per cui, vicino al
punto di interpolazione, aggiungere dati può peggiorare il test error.

Il rasoio di Occam regge, se si cambia che cosa si misura. Il numero di
parametri è un pessimo proxy della complessità di una rete, e la candidata più
studiata al suo posto è una misura di norma della soluzione trovata, la
stessa da cui partono i {doc}`bound di norma della teoria dell'apprendimento
</TeoriaApprendimento/garanzie-e-reti>`. In due casi il
legame con l'algoritmo è un teorema: sui minimi quadrati con più parametri che
esempi la discesa del gradiente partita da zero converge alla soluzione
interpolante di norma $\ell_2$ minima, un fatto classico dell'algebra lineare, e
sulla regressione logistica con dati separabili la direzione dei pesi converge
a quella di massimo margine, come hanno dimostrato Soudry e colleghi
{cite}`soudry2018implicit`. Sulle reti profonde lo stesso *bias implicito* è
un'ipotesi con buone prove sperimentali: che «semplice» non si conti in
parametri è assodato, in che cosa si conti no.

Si vede soprattutto con le etichette sporche, e la regolarizzazione appiana il
picco. La prima clausola è degli autori stessi («le osserviamo tutte con più
forza dove le etichette hanno del rumore»), che però elencano subito dopo i
casi in cui il picco c'è anche con le etichette pulite, e ne mostrano almeno
uno in cui sopravvive perfino all'arresto anticipato scelto al meglio. La
seconda viene da un lavoro successivo dello stesso gruppo
{cite}`nakkiran2021optimal`: una penalità $\ell_2$ tarata al meglio rende
monotona la curva sui modelli lineari con dati isotropi, e attenua il picco
anche sulle reti.

Resta parecchio da capire: quali architetture e quali regimi la mostrino, e
perché il bias implicito abbia la forma che ha. Il consiglio operativo non è
cambiato, ma la sua motivazione sì: non fermarsi al primo minimo della curva
di validazione solo perché il modello sembra troppo grande.

`````

## Il biglietto vincente: a che serve tutta quella capacità

La doppia discesa dice *che* le reti sovradimensionate generalizzano. Resta la
domanda su *perché*, e c'è un risultato che offre una risposta diversa e
sorprendentemente concreta.

Prima però serve un'immagine di che cosa sia una rete neurale, perché qui
la si pota come una pianta (ci sarà un capitolo intero a raccontarla: quel che
segue basta per ora). Immagina tanti nodi disposti a strati, e fra un
nodo e il successivo un filo che porta il segnale moltiplicandolo per un
numero: quel numero è il peso del collegamento, uno dei tanti parametri che
l'addestramento aggiusta. E i pesi iniziali, quelli da cui la messa a punto
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

Dentro la rete grande ce n'è una piccola che basta. Il punto sta
nell'inizializzazione: quella sottorete funziona solo se riparte dai *suoi*
pesi iniziali, quelli che aveva nella rete grande.
```

La condizione in coda a {numref}`fig-biglietto-vincente` è ciò che rende
questa idea, che si chiama **ipotesi del biglietto vincente**, interessante
invece che ovvia. Se si riprende la stessa sottorete e
la si inizializza da capo a caso, non impara altrettanto bene: il biglietto sta
nella coppia fra la forma della sottorete e i numeri con cui è
nata. Sovradimensionare, in questa lettura, serve a comprare molti biglietti.

`````{tab} Elementare

Il punto di partenza è un paradosso noto da tempo. Prendi una rete addestrata
ed elimina i pesi più piccoli: puoi buttarne via il $90\%$ senza quasi perdere
accuratezza. Ma se poi provi a costruire da zero una rete piccola *con quella
stessa struttura* e ad addestrarla, impara peggio. La potatura funziona solo
dopo l'addestramento, e nessuno spiegava bene perché.

Frankle e Carbin (2019) hanno provato una cosa diversa. Dopo aver potato, invece
di ripartire con pesi casuali nuovi, hanno riavvolto i pesi sopravvissuti ai
valori casuali che avevano *all'inizio*, prima di qualsiasi addestramento. Quella
sottorete minuscola, riaddestrata da sola, raggiunge l'accuratezza della rete
piena. E il giro si può ripetere: si pota ancora, si riavvolge ancora, e la
sottorete si stringe di volta in volta.

La rete grande, allora, non serve tutta. Serve perché, fra le sue milioni di
connessioni inizializzate a caso, ne contiene per fortuna un sottoinsieme già
disposto bene per il compito; il resto è impalcatura. Da qui il nome,
biglietto vincente, e la rete grande come un mazzo di biglietti comprati
tutti insieme.

Il seguito ha corretto due cose, e conoscerle evita di prendere l'idea per più
di quello che è. Sulle reti grandi e sui dati veri il riavvolgimento fino al
primo giorno smette di funzionare: bisogna tornare indietro un po’ meno, a dopo
qualche giro di addestramento. Il biglietto, quindi, prende forma nelle prime
ore di scuola invece di essere già stampato alla nascita. E per trovarlo la
rete intera va addestrata comunque, più di una volta: è un modo di capire a che
cosa serva tutta quella taglia, non una scorciatoia per allenare reti piccole.

`````

`````{tab} Superiore

La procedura {cite}`frankle2019lottery` è l’*iterative magnitude pruning*: si
annota l'inizializzazione
$\theta_0$, si addestra, si elimina una frazione dei pesi più piccoli, si
riportano i sopravvissuti ai valori in $\theta_0$, si riaddestra, si ripete. Le
sottoreti trovate pesavano spesso meno del $10$–$20\%$ della rete di partenza, e
raggiungevano l'accuratezza piena in un numero comparabile di iterazioni.

Due avvertenze di onestà, perché il risultato è più fragile di come viene
spesso citato.

Alla scala grande la ricetta va corretta, e la correzione non è nel lavoro del
2019 ma in uno successivo di Frankle e Carbin con Dziugaite e Roy
{cite}`frankle2020linear`: su reti profonde e dataset seri il riavvolgimento a
$\theta_0$ smette di funzionare, e si riavvolge invece a un $\theta_k$ dopo
qualche iterazione di addestramento (*rewinding* tardivo). Il biglietto,
quindi, non è del tutto presente all'inizializzazione: si forma nelle prime
fasi.

Non è un metodo di compressione pratico. Per *trovare* il biglietto
bisogna addestrare la rete piena, più volte. Il valore è conoscitivo (dice
qualcosa su cosa fa la sovraparametrizzazione) non computazionale. Per
comprimere davvero si usano la potatura strutturata e la quantizzazione, che
sono il mestiere del {doc}`capitolo sull'efficienza </Efficienza/overview>`.

Il filo con la doppia discesa è comunque lo stesso: il numero di parametri
misura male la complessità. La doppia discesa lo mostra dall'esterno, guardando
la curva d'errore; il biglietto vincente dall'interno, guardando cosa la rete
usa davvero.

`````

Attenzione a non trarne la conclusione sbagliata. Doppia discesa e biglietto
vincente non smontano nulla di ciò che viene prima: dicono soltanto che
«quanto è complesso un modello» non si conta in parametri. Tenere il test
chiuso, misurare su dati mai visti e far pagare un prezzo alla complessità
restano esattamente ciò che erano, e sono le cose da portarsi via.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Ci sono due modi opposti di sbagliare: essere troppo rigidi (una retta
  dove serviva una curva: si sbaglia già sugli esempi di scuola) ed essere
  troppo flessibili (una curva che passa per ogni punto, rumore compreso:
  dieci e lode sugli esempi di scuola, disastro sui casi nuovi). Il secondo è
  l’*overfitting*, cioè imparare a memoria.
- Immagina di riaddestrare il modello molte volte su dati sempre nuovi: se le
  risposte sono tutte spostate dalla stessa parte è un difetto di mira; se
  sono sparpagliate è un difetto di stabilità. Si correggono in modi
  opposti, e mettendo la flessibilità del modello su un asse l'errore sui dati
  nuovi disegna una U: si sceglie il fondo.
- Prima di spendere per raccogliere altri dati, si guardano le curve di
  apprendimento: si riaddestra con sempre più esempi e si guarda l'errore. Se
  quello sugli esempi di scuola e quello sui casi nuovi si sono già raggiunti e
  fermati, altri dati non servono e va cambiato modello; se fra i due resta un
  divario, i dati in più pagano.
- Per accorgersene bisogna misurare su dati che il modello non ha usato: si
  divide in tre, studio, prove, esame. L'esame (il *test*) si apre una sola
  volta, alla fine: ogni sbirciata lo consuma e il voto diventa più generoso
  del vero. E anche i preparativi (rimettere le colonne in scala, riempire le
  caselle vuote) vanno fatti guardando solo la parte di studio.
- Con pochi dati conviene la cross-validation: si divide in cinque blocchi
  e a turno uno fa da prova, come cinque compiti in classe su cinque parti
  diverse del programma invece di una sola interrogazione. Contano la media dei
  cinque voti e quanto sono discordi. Vale però solo se i cinque compiti
  chiedono cose davvero diverse: se lo stesso soggetto ricompare in più righe,
  le sue righe vanno tenute tutte nello stesso blocco.
- Per frenare la memorizzazione si mette un prezzo alla complessità: il
  modello può usare pesi grandi solo se ne conviene. Contando la spesa a
  valori assoluti alcuni pesi vanno esattamente a zero (le caratteristiche
  inutili spariscono), contandola a quadrati si rimpiccioliscono tutti. Le
  colonne che vanno in gruppo si fanno pagare come un pacchetto, che entra o
  esce intero, e i pacchetti li sceglie chi scrive il modello.
- Scegliere le colonne a salti (dentro o fuori) è nervoso: basta poco per
  cambiare la scelta. Conviene quando i dati sono chiari, mentre con tanto
  rumore vince il freno continuo, e convocare con il freno per poi allentarlo
  va bene quasi sempre.
- Il principio antico è il rasoio di Occam: a parità di spiegazione dei
  dati, vince la spiegazione più semplice. Il principio regge anche per la
  doppia discesa, dove oltre il punto in cui il modello impara tutto a
  memoria ingrandirlo ancora torna a farlo funzionare meglio: quello che conta
  è quanto sono grandi i pesi, più che quante manopole ha il modello.
- E una risposta c'è anche alla domanda a che cosa serva tutta quella taglia:
  dentro una rete grande ce n'è una piccola già disposta bene per il compito, e
  funziona solo se riparte dai numeri che aveva all'inizio, o poco dopo quando
  la rete è grande davvero. Trovarla costa più che addestrare la rete intera,
  quindi è un modo di capire, non di risparmiare.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Underfitting: modello troppo semplice, sbaglia già sul training (bias
  alto). Overfitting: modello troppo flessibile, memorizza il rumore ed è
  ottimo sul training ma pessimo sui dati nuovi (varianza alta).
- L'errore di test ha forma a U nella complessità: il minimo è il
  compromesso bias-varianza. Oltre il punto di interpolazione, però, la
  curva può riscendere (doppia discesa): il numero di parametri è un
  pessimo proxy della complessità di una rete.
- Il biglietto vincente guarda lo stesso fatto dall'interno: una rete grande
  contiene una sottorete piccola già ben inizializzata, e il resto è
  impalcatura.
- La decomposizione bias-varianza è un'identità della loss quadratica: per
  la 0-1 i due termini restano vocabolario, non aritmetica.
- Si divide in train / validation / test. Il test non si tocca: si apre
  una sola volta, alla fine, o la stima diventa ottimista. Ogni trasformazione
  che *impara* dai dati (scaler, imputazione, selezione) si tara dentro il
  training: è la forma di *leakage* che non dà avvisi.
- La k-fold cross-validation media $k$ validazioni su fold diversi: stima
  più stabile quando i dati sono pochi. Vale se le righe sono scambiabili:
  con righe raggruppate per soggetto servono `GroupKFold`, con righe ordinate
  nel tempo `TimeSeriesSplit`.
- La regolarizzazione (Ridge $\ell_2$, Lasso $\ell_1$) frena la complessità
  con una penalità $\lambda$ sui pesi; il Lasso azzera le feature inutili, il
  group lasso ($\sum_g\sqrt{p_g}\lVert\theta_g\rVert_2$) interi blocchi
  dichiarati in anticipo, indipendentemente dalla codifica solo se i blocchi
  sono ortonormalizzati.
- La selezione discreta (sottoinsieme migliore, selezione in avanti) è
  instabile, e la sua previsione salta dove quella del Lasso è continua: a
  rapporto segnale/rumore basso vince il Lasso, a rapporto alto (raro sui dati
  osservazionali) la selezione; il relaxed lasso è competitivo in tutti e due i
  regimi.
- Il rasoio di Occam regge se la complessità non si conta in parametri: sui
  modelli lineari la discesa del gradiente ha un *bias implicito* dimostrato
  verso soluzioni di norma piccola, sulle reti profonde è l'ipotesi più
  studiata.
```

`````
