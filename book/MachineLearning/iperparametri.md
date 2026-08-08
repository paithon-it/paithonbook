# Trovare gli iperparametri

Nel dicembre 2017, dal palco di NIPS (oggi NeurIPS) (la più importante
conferenza mondiale di machine learning) Ali Rahimi pronunciò una frase
destinata a far discutere per mesi: «il machine learning è diventato
alchimia». Non ce l'aveva con i modelli, che funzionano, ma con il modo in cui
li mettiamo a punto: ricette tramandate di laboratorio in laboratorio, dosi
aggiustate a occhio, risultati che arrivano senza che nessuno sappia spiegare
fino in fondo perché. Il bersaglio della provocazione erano soprattutto loro:
gli **iperparametri**.

Ogni modello ha manopole che nessun addestramento gira da solo: il passo della
discesa del gradiente (il *learning rate*), la profondità di un albero o il
numero di strati di una rete, la forza $\lambda$ del freno di
regolarizzazione. Il gradiente aggiusta i parametri $\theta$, ma le manopole
restano dove le abbiamo messe noi, e da dove le mettiamo dipende, spesso in
modo drammatico, la qualità del risultato. Nella sezione su overfitting e
validazione abbiamo già stabilito *dove* giudicare queste scelte: sul
validation set, o meglio in cross-validation, mai sul test. Resta la domanda
che occupa questa sezione: **come esplorare** lo spazio delle combinazioni.
Girare le manopole a mano finché "funziona" è l'alchimia di cui parlava
Rahimi; farlo per bene è un problema di ricerca (nel senso letterale di
*search*) con i suoi algoritmi, i suoi conti e le sue trappole.

## Provarle tutte: la grid search

L'idea più naturale è la forza bruta: per ogni manopola si sceglie una manciata
di valori candidati e si prova **ogni combinazione**, tenendo quella con il
punteggio di validazione migliore. È la *grid search*, la ricerca a griglia:
semplice, esaustiva entro la griglia, facilissima da eseguire in parallelo. E
con un difetto che non perdona.

`````{tab} Elementare

Pensa a una macchina del caffè professionale con quattro regolazioni:
macinatura, temperatura, pressione, tempo di estrazione. Cinque livelli
ciascuna. Per assaggiare tutte le combinazioni servono
$5 \times 5 \times 5 \times 5 = 625$ caffè. E siccome un solo assaggio può
ingannare (magari quella tazzina è venuta bene per caso), ne servono cinque
per ogni combinazione, la cross-validation che già conosciamo: 3 125 caffè. Se
ogni "caffè" è un addestramento da due minuti, sono più di quattro giorni di
macchina; aggiungi una quinta manopola a cinque livelli e i giorni diventano
ventuno. È l'**esplosione combinatoria**: ogni manopola in più *moltiplica* le
prove, non le somma.

`````

`````{tab} Superiore

Con $d$ iperparametri e $n$ valori candidati ciascuno, la griglia è il
prodotto cartesiano $\Lambda = \Lambda_1 \times \dots \times \Lambda_d$ con
$|\Lambda| = n^d$ configurazioni; scriveremo $\lambda \in \Lambda$ per una
singola configurazione: l'intero vettore di manopole, tra cui la forza di
regolarizzazione che per tradizione si indica con la stessa lettera. Con
$k$-fold cross-validation il costo sale a $k \cdot n^d$ addestramenti:
esponenziale in $d$. E a parità di budget la risoluzione per dimensione è
misera, $n = |\Lambda|^{1/d}$: con nove prove in due dimensioni si vedono
appena tre valori per asse. La grid search resta ragionevole per $d \le 2$, e
ha il pregio che i punti, indipendenti tra loro, si valutano in parallelo; per
i parametri di scala, come learning rate e $\lambda$, i candidati vanno
disposti in progressione geometrica ($10^{-4}, 10^{-3}, \dots$).

`````

## Il caso batte la griglia: la random search

La prima alternativa sembra una resa: invece di una griglia ordinata, estrarre
le combinazioni **a caso** dentro gli stessi intervalli. Nel 2012 James
Bergstra e Yoshua Bengio mostrarono che questa mossa apparentemente pigra è,
quasi sempre, la più efficiente {cite}`bergstra2012random`. Il motivo sta in
un fatto empirico: in quasi tutti i problemi **poche manopole contano
davvero**, e non sappiamo in anticipo quali ({numref}`fig-grid-vs-random`).

```{figure} ../figures/grid-vs-random.svg
:name: fig-grid-vs-random
:alt: Due pannelli affiancati. In ciascuno, un quadrato rappresenta lo spazio di due iperparametri e una curva a campana sopra il quadrato mostra come il punteggio dipende dal solo parametro importante, sull'asse orizzontale. A sinistra nove punti in griglia tre per tre si proiettano su appena tre posizioni dell'asse orizzontale; a destra nove punti casuali si proiettano su nove posizioni distinte, e uno cade quasi sul massimo della curva.
:width: 95%

Nove prove in griglia esplorano solo tre valori del parametro che conta davvero
(curva in alto); nove prove casuali ne esplorano nove, e una atterra a un passo
dal massimo.
```

`````{tab} Elementare

Immagina una vecchia radio con due manopole: la sintonia e il volume. La
stazione la trovi solo girando la sintonia; il volume, ai fini della ricerca,
non conta nulla. Hai diritto a nove tentativi. Se li disponi in una griglia
$3 \times 3$, di frequenze ne provi in realtà **tre**: ogni colonna ripete la
stessa sintonia a tre volumi diversi, cioè due tentativi su tre sono buttati.
Nove tentativi a caso, invece, toccano nove frequenze **tutte diverse**, e la
probabilità di cascare vicino alla stazione giusta sale parecchio. Il bello è
che non serve sapere in anticipo quale manopola conti: il caso esplora bene
*tutte* le dimensioni contemporaneamente.

`````

`````{tab} Superiore

L'argomento formale è la proiezione: se l'errore di validazione dipende in
modo apprezzabile solo da un piccolo sottoinsieme delle $d$ dimensioni (*low
effective dimensionality*), una griglia di $N$ punti ne proietta appena
$N^{1/d}$ distinti su ciascun asse, mentre $N$ punti casuali ne proiettano $N$
{cite}`bergstra2012random`. C'è anche una garanzia indipendente da $d$: se
esiste una regione "buona" che copre il 5% del volume dello spazio di ricerca,
la probabilità che $n$ estrazioni indipendenti la manchino tutte è
$(1-0{,}05)^n$; con $n = 60$ si ottiene $1 - 0{,}95^{60} \approx 0{,}95$;
sessanta prove la centrano con il 95% di confidenza, in qualunque dimensione.
(Vale *se* una tale regione esiste ed è così larga: è un'ipotesi sul problema,
non una promessa.) In pratica contano anche le distribuzioni: per i parametri
di scala si campiona in **log-uniforme**, cioè uniforme sull'esponente, così
che il learning rate cada tra $10^{-5}$ e $10^{-4}$ con la stessa probabilità
con cui cade tra $10^{-2}$ e $10^{-1}$.

`````

## Tornei a eliminazione: successive halving e Hyperband

Griglia e caso condividono uno spreco: dedicano lo **stesso budget** a ogni
candidato, anche a quelli che dopo due epoche sono già palesemente senza
speranza. Gli approcci *multi-fidelity* ribaltano la logica: valutazioni
economiche e approssimate per scremare, valutazioni costose e accurate solo per
i migliori. Come un torneo a eliminazione diretta.

`````{tab} Elementare

Un torneo di tennis non fa giocare cento partite a ogni iscritto: fa giocare a
tutti *una* partita, e solo chi vince continua. Il *successive halving* fa lo
stesso con le configurazioni: parti con 81 candidate e concedi a ciascuna una
sola epoca di addestramento; le 27 migliori ne ricevono tre; le 9 migliori
nove; le 3 migliori ventisette; la finalista arriva a 81. Ogni turno costa
all'incirca lo stesso (81 epoche in tutto) e i turni sono cinque: circa 400
epoche totali, contro le oltre seimila che servirebbero per addestrare fino in
fondo tutte le 81 candidate. C'è però un rischio: eliminare i "diesel", le
configurazioni che partono piano ma finirebbero forte. Hyperband copre il
rischio organizzando più tornei con regole diverse: alcuni spietati
(tantissimi iscritti, primo turno brevissimo), altri clementi (pochi iscritti,
tanto tempo a testa fin dall'inizio).

`````

`````{tab} Superiore

Il *successive halving* {cite}`jamieson2016non` con fattore di eliminazione
$\eta$ (tipicamente 3): date $n$ configurazioni con budget iniziale $r$
ciascuna (epoche, o frazione del dataset), a ogni round tiene le migliori
$1/\eta$ e moltiplica per $\eta$ il budget individuale. I round sono
$\lfloor \log_\eta n \rfloor + 1$ e ognuno costa circa $n \cdot r$: per
$n=81$, $r=1$, $\eta=3$, 405 epoche-modello contro le $81 \times 81 = 6\,561$
della valutazione completa. L'analisi inquadra il problema come *best-arm
identification* in un **bandit** non stocastico (la famiglia di problemi che il
capitolo sul reinforcement learning introduce per prima, dove ogni
configurazione è una leva e addestrarla per un'epoca è un tiro): basta che le
classifiche parziali
siano abbastanza indicative di quelle finali, ed è proprio questa l'ipotesi
fragile, perché una configurazione a convergenza lenta viene eliminata da
giovane. **Hyperband** {cite}`li2018hyperband` aggira il dilemma tra molte
configurazioni e molto budget per testa eseguendo $s_{\max}+1$ istanze di
successive halving (i *bracket*, $s_{\max} = \lfloor \log_\eta R \rfloor$ con
$R$ budget massimo per configurazione), dalla più aggressiva
($\sim \eta^{s_{\max}}$ configurazioni con budget iniziale
$R/\eta^{s_{\max}}$) alla più conservativa, poche configurazioni a budget
pieno; la garanzia teorica è restare entro fattori logaritmici dal bracket
migliore col senno di poi.

`````

L'*early stopping* (che vedremo all'opera nel capitolo su PyTorch) è il caso
limite di questa idea: un torneo con un solo iscritto, che si ritira quando la
validazione smette di migliorare.

## Cercare con giudizio: l'ottimizzazione bayesiana

Griglia, caso e tornei condividono un ultimo difetto, il più profondo: ogni
prova **ignora ciò che le precedenti hanno scoperto**. Se dieci esperimenti
hanno già mostrato che i learning rate alti fanno divergere il modello,
l'undicesimo estratto a caso può cascarci di nuovo. L'**ottimizzazione
bayesiana** {cite}`snoek2012practical` tratta la ricerca degli iperparametri
come un problema di apprendimento a sua volta: costruisce un modello di *come
le manopole influenzano il punteggio* e lo usa per decidere dove provare.

`````{tab} Elementare

Pensa a un geologo che cerca l'acqua potendo scavare pochi pozzi, perché ogni
trivellazione costa cara. Dopo tre pozzi non sceglie il quarto a caso: disegna
una mappa ("qui l'acqua c'era a dieci metri, là il terreno era secco"),
completa di zone d'ombra dove non sa ancora nulla. Il quarto pozzo lo piazza
dove la *promessa* è massima: un po' dove la mappa dice bene (sfruttare ciò
che sa), un po' dove la mappa è bianca (esplorare ciò che ignora).
L'ottimizzazione bayesiana funziona così: dopo ogni addestramento aggiorna la
sua mappa del punteggio e sceglie la configurazione successiva chiedendosi *di
quanto mi aspetto di battere il mio record, se provo qui?* La domanda ha un
nome, **miglioramento atteso** (*expected improvement*), e ha il pregio di
spegnersi da sola nelle zone che non promettono nulla e non nascondono più
sorprese.

`````

`````{tab} Superiore

Due ingredienti. Il **modello surrogato** è una distribuzione di probabilità
sulla funzione ignota $f(\lambda)$ (l'errore di validazione della
configurazione $\lambda$) aggiornata dopo ogni osservazione; il surrogato
standard è il **processo gaussiano** {cite}`rasmussen2006gaussian`, che per
ogni $\lambda$ fornisce una media $\mu(\lambda)$ e una deviazione standard
$\sigma(\lambda)$: la stima e la sua incertezza. (Ai processi gaussiani è
dedicata una sezione di questo capitolo.) La **funzione di acquisizione**
traduce stima e incertezza in una decisione; la più usata è l'*expected
improvement*:

$$
\mathrm{EI}(\lambda) = \mathbb{E}\big[\max\big(0,\; f_{\min} - f(\lambda)\big)\big],
$$

dove $f_{\min}$ è il miglior errore osservato finora e l'attesa è presa sulla
distribuzione del surrogato. Con surrogato gaussiano l'attesa ha forma chiusa:

$$
\mathrm{EI}(\lambda) = \sigma(\lambda)\,\big(\gamma\,\Phi(\gamma) + \varphi(\gamma)\big),
\qquad
\gamma = \frac{f_{\min} - \mu(\lambda)}{\sigma(\lambda)},
$$

dove $\Phi$ e $\varphi$ sono la funzione di ripartizione e la densità della
normale standard. La formula premia sia $\mu(\lambda)$ basso (sfruttamento)
sia $\sigma(\lambda)$ alto (esplorazione); la prossima prova è
$\lambda_{\text{next}} = \arg\max_\lambda \mathrm{EI}(\lambda)$:
un'ottimizzazione a sua volta, ma sul surrogato, che risponde in millisecondi.
Il prezzo è la natura essenzialmente **sequenziale** del metodo (ogni scelta
attende l'esito della precedente) e la dipendenza dalle ipotesi del surrogato,
a cominciare dalla scelta del kernel.

`````

## Alla prova del codice

scikit-learn offre le due strategie di base con la stessa interfaccia:
`GridSearchCV` e `RandomizedSearchCV` incapsulano il ciclo "prova una
configurazione, valutala in cross-validation, tieni la migliore". Le proviamo
su un classificatore SVC, una *support vector machine*, che qui usiamo come
scatola nera: ci basta sapere che ha due manopole delicate, `C` e `gamma`,
entrambe parametri di scala (sul piccolo dataset di cifre manoscritte incluso
in scikit-learn).

```python
from scipy.stats import loguniform
from sklearn.datasets import load_digits
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     train_test_split)
from sklearn.svm import SVC

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)   # il test resta nel cassetto

# Grid search: 4 x 4 = 16 combinazioni, x 5 fold = 80 addestramenti
griglia = {"C": [0.1, 1, 10, 100],
           "gamma": [1e-4, 1e-3, 1e-2, 1e-1]}
ricerca_griglia = GridSearchCV(SVC(), griglia, cv=5, n_jobs=-1)
ricerca_griglia.fit(X_train, y_train)
print(ricerca_griglia.best_params_, round(ricerca_griglia.best_score_, 3))

# Random search: 20 estrazioni log-uniformi (uniformi sull'esponente)
distribuzioni = {"C": loguniform(1e-2, 1e3),
                 "gamma": loguniform(1e-5, 1e0)}
ricerca_casuale = RandomizedSearchCV(SVC(), distribuzioni, n_iter=20,
                                     cv=5, random_state=42, n_jobs=-1)
ricerca_casuale.fit(X_train, y_train)
print(ricerca_casuale.best_params_, round(ricerca_casuale.best_score_, 3))

# il test si apre una sola volta, alla fine
print(ricerca_casuale.score(X_test, y_test))
```

Il trucco di `loguniform` è lo stesso che useremo per il learning rate di una
rete neurale: si campiona l'esponente, non il valore. scikit-learn implementa
anche il successive halving (`HalvingGridSearchCV` e `HalvingRandomSearchCV`,
ancora marcati come sperimentali); per l'ottimizzazione bayesiana e i tornei
in versione moderna la libreria di riferimento è **Optuna**, in cui lo spazio
di ricerca si descrive direttamente nel codice e la potatura anticipata
elimina in corsa le prove peggiori.

## Le avvertenze sul foglietto

Tre avvertenze, prima di chiudere. La prima è il **costo**: ogni punto dello
spazio di ricerca vale $k$ addestramenti completi, e nessun algoritmo elimina
questo fattore (lo spende meglio). Griglia e caso almeno si parallelizzano
senza sforzo; l'ottimizzazione bayesiana si parallelizza peggio: le varianti
batch esistono (già Snoek e colleghi ne proponevano una
{cite}`snoek2012practical`), ma pagano in efficienza per prova. La
seconda è la **riproducibilità**: una ricerca casuale senza seme fissato (il
*seed*: `random_state` nel codice sopra) cambia esito a ogni esecuzione, e un
confronto tra metodi che non dichiari spazio di ricerca e budget non è un
confronto (è aneddotica). La terza è la più subdola.

`````{tab} Elementare

Se mille persone lanciano una moneta dieci volte, qualcuna farà nove teste:
non è una maga, è la più fortunata di mille. Lo stesso vale per le
configurazioni: il punteggio di validazione della vincitrice di una ricerca
con centinaia di prove è in parte merito e in parte fortuna, e tende a essere
**troppo ottimista**. Più a lungo cerchi, più il validation set si consuma:
proprio come il test set che avevamo giurato di non sbirciare. Il rimedio è lo
stesso di sempre: il numero da raccontare al mondo si misura una sola volta,
alla fine, sul test rimasto intatto.

`````

`````{tab} Superiore

Il massimo di $N$ stime rumorose è uno stimatore distorto verso l'alto del
vero massimo: selezionando la configurazione con il miglior punteggio di
validazione si eredita anche il suo errore di stima favorevole, e la
distorsione cresce con $N$. In altre parole, una ricerca abbastanza lunga fa
overfitting *sul validation set*. Le contromisure: riservare il test a
un'unica valutazione finale; riportare media e deviazione standard sui fold,
non il solo massimo; nei confronti metodologici, usare la *nested
cross-validation* (un anello esterno per la stima onesta dell'errore, un
anello interno per la selezione degli iperparametri) accettandone il costo,
che è il prodotto dei due anelli.

`````

```{admonition} Da ricordare
:class: important
- Gli iperparametri non si imparano con il gradiente: si **cercano**, e si
  giudicano su validation o cross-validation, mai sul test.
- La **grid search** è esaustiva ma esponenziale nel numero di manopole:
  ragionevole solo per una o due dimensioni.
- La **random search** a parità di prove esplora più valori di ogni singola
  dimensione: vince quando poche manopole contano davvero
  {cite}`bergstra2012random`. Parametri di scala in log-uniforme.
- **Successive halving** e **Hyperband** sono tornei a eliminazione: poco
  budget a molti, molto budget a pochi {cite}`jamieson2016non,li2018hyperband`.
- L'**ottimizzazione bayesiana** usa un surrogato (tipicamente un processo
  gaussiano) e una funzione di acquisizione per imparare dalle prove passate
  {cite}`snoek2012practical`.
- Il punteggio del vincitore è **ottimista**: numero finale solo dal test
  intatto, seed fissati, spazio e budget dichiarati.
```
