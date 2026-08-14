# Trovare gli iperparametri

Nel dicembre 2017, dal palco di NIPS, la più importante conferenza mondiale di
machine learning (oggi si chiama NeurIPS), Ali Rahimi ritirò un premio per un
lavoro di dieci anni prima e ne approfittò per dire una cosa scomoda: «il
machine learning è diventato alchimia».

Non ce l'aveva con i risultati, che ci sono: ce l'aveva con le fondamenta.
Facciamo funzionare le cose, disse, senza sapere davvero *perché* funzionino, e
portò tre esempi. Uno era un problemino minuscolo su cui la discesa del
gradiente si pianta: non perché sia arrivata in fondo alla discesa, ma pur
avendo ancora sotto i piedi un terreno in pendenza. Un altro era un ingrediente
che a quel tempo tutti mettevano nelle reti (si chiama *batch normalization*, e
il libro la incontrerà più avanti) di cui, a suo dire, «come disciplina non
sappiamo quasi niente». Il terzo era la storia di un sistema che si era rotto
senza che nessuno capisse perché: qualcuno aveva cambiato il modo di
arrotondare i numeri dentro una libreria, e l'errore era passato da meno del
25% a quasi il 99%. La spiegazione arrivò dopo, e vale la pena darla perché
smentisce a metà l'aneddoto: quell'arrotondamento portava a uno un numero che
doveva restare appena sotto, e da lì usciva una divisione per zero. Un difetto
del programma, quindi, non del metodo.

Gli **iperparametri**, in quel discorso, non erano nominati nemmeno una volta.
Ma se c'è un posto in cui l'alchimia si vede a occhio nudo sono loro: ricette
tramandate di laboratorio in laboratorio, dosi aggiustate a occhio, risultati
che arrivano senza che nessuno sappia spiegare fino in fondo perché.

Un iperparametro è una **scelta che facciamo noi prima di cominciare e che
l'addestramento non cambia**. Sono le manopole del modello, e nessun
addestramento le gira da solo. Qualche esempio già incontrato: quanto è lungo
il passo della discesa del gradiente (il *learning rate* della sezione
sull'apprendimento supervisionato) e quanto è tirato il freno alla
memorizzazione (la $\lambda$, la lettera greca *lambda*, della sezione sulla
regolarizzazione). Qualche esempio che incontreremo: quante domande di fila può
fare un albero di decisione, quanti strati ha una rete.

Non vanno confusi con i **parametri**: quelli sono i numeri interni che
l'addestramento aggiusta da sé, girando finché il modello sbaglia il meno
possibile. Le manopole, invece, restano dove le abbiamo messe noi, e da dove le
mettiamo dipende, spesso in
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
semplice, esaustiva entro la griglia, e facilissima da spalmare su più
calcolatori, perché ogni combinazione è indipendente dalle altre e nessuno deve
aspettare nessuno (è quel che si intende con «si esegue **in parallelo**»). E
con un difetto che non perdona.

`````{tab} Elementare

Pensa a una macchina del caffè professionale con quattro regolazioni:
macinatura, temperatura, pressione, tempo di estrazione. Cinque livelli
ciascuna. Per assaggiare tutte le combinazioni servono
$5 \times 5 \times 5 \times 5 = 625$ caffè. E siccome un solo assaggio può
ingannare (magari quella tazzina è venuta bene per caso), ogni combinazione va
provata cinque volte: è la cross-validation della sezione precedente, che
divide i dati in cinque blocchi e fa girare il blocco di prova. Quindi
$625 \times 5 = 3\,125$ caffè.

Se ogni «caffè» è un addestramento da due minuti, sono
$3\,125 \times 2 = 6\,250$ minuti, cioè quattro giorni e un terzo di macchina
accesa. E se aggiungi una quinta manopola, sempre a cinque livelli, le
combinazioni si moltiplicano ancora per cinque e i giorni diventano quasi
ventidue. È
l'**esplosione combinatoria**: ogni manopola in più *moltiplica* le
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
davvero**, e non sappiamo in anticipo quali.

{numref}`fig-grid-vs-random` mostra il caso più semplice, due sole manopole di
cui una decisiva e l'altra ininfluente. Il quadrato è lo spazio delle prove
possibili, una manopola per lato, e ogni pallino è una prova. La curva a
campana disegnata sopra il quadrato dice come va il punteggio al variare della
manopola che conta: più in alto, meglio è, e il colmo della campana è il valore
che stiamo cercando.

```{figure} ../figures/grid-vs-random.svg
:name: fig-grid-vs-random
:alt: Due pannelli affiancati. In ciascuno, un quadrato rappresenta lo spazio di due iperparametri e una curva a campana sopra il quadrato mostra come il punteggio dipende dal solo iperparametro importante, sull'asse orizzontale. A sinistra nove punti in griglia tre per tre si proiettano su appena tre posizioni dell'asse orizzontale; a destra nove punti casuali si proiettano su nove posizioni distinte, e uno cade quasi sul massimo della curva.
:width: 95%

Nove prove in griglia (a sinistra) si affacciano su appena tre valori della
manopola che conta, perché tre a tre stanno in colonna. Nove prove casuali (a
destra) ne toccano nove diversi, e una finisce a un passo dal colmo della
campana.
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
$(1-0{,}05)^n = 0{,}95^n$; quindi la probabilità di centrarla **almeno una
volta** è il complementare, $1 - 0{,}95^n$, che per $n = 60$ vale $0{,}954$:
sessanta prove la centrano con il 95% di confidenza, in qualunque dimensione.
(Vale *se* una tale regione esiste ed è così larga: è un'ipotesi sul problema,
non una promessa.) In pratica contano anche le distribuzioni: per i parametri
di scala si campiona in **log-uniforme**, cioè uniforme sull'esponente, così
che il learning rate cada tra $10^{-5}$ e $10^{-4}$ con la stessa probabilità
con cui cade tra $10^{-2}$ e $10^{-1}$.

`````

## Tornei a eliminazione: successive halving e Hyperband

Griglia e caso condividono uno spreco: dedicano lo **stesso tempo** a ogni
candidato, anche a quelli che dopo pochissimo allenamento sono già palesemente
senza speranza. C'è una famiglia di metodi che ribalta la logica: prove brevi e
grossolane per scremare, prove lunghe e accurate solo per i pochi che si sono
salvati. Come un torneo a eliminazione diretta. (Il nome tecnico è
*multi-fidelity*, cioè «a più livelli di fedeltà»: la prova breve è una versione
poco fedele di quella vera.)

L'unità di misura di tutta questa sezione è l'**epoca**: una passata completa
sull'insieme di addestramento, cioè il modello che ha visto una volta ciascuno
dei suoi esempi. Un addestramento serio ne fa decine o centinaia, e il costo di
una ricerca si conta in epoche esattamente come il costo di un viaggio si conta
in litri.

`````{tab} Elementare

Un torneo di tennis non fa giocare cento partite a ogni iscritto: fa giocare a
tutti *una* partita, e solo chi vince continua. Il *successive halving* fa lo
stesso con le combinazioni di manopole: parti con 81 candidate e concedi a
ciascuna una
sola epoca di addestramento; le 27 migliori ne ricevono tre; le 9 migliori
nove; le 3 migliori ventisette; la finalista arriva a 81.

Il bello è che ogni turno costa quanto gli altri, perché a ogni giro i
sopravvissuti si riducono a un terzo e le epoche a testa si triplicano:
$81 \times 1$, poi $27 \times 3$, poi $9 \times 9$, poi $3 \times 27$, poi
$1 \times 81$. Fanno 81 epoche per turno, e i turni sono cinque: 405 epoche in
tutto. Addestrare fino in fondo tutte e 81 le candidate ne costerebbe
$81 \times 81 = 6\,561$, sedici volte tanto.

C'è però un rischio: eliminare i "diesel", le
configurazioni che partono piano ma finirebbero forte. Hyperband copre il
rischio organizzando più tornei con regole diverse: alcuni spietati
(tantissimi iscritti, primo turno brevissimo), altri clementi (pochi iscritti,
tanto tempo a testa fin dall'inizio).

`````

`````{tab} Superiore

Il *successive halving* è di Karnin, Koren e Somekh (ICML 2013)
{cite}`karnin2013almost`, che lo introdussero per il bandit stocastico a pura
esplorazione; Jamieson e Talwalkar {cite}`jamieson2016non` ne definiscono la
variante **non stocastica** e la portano agli iperparametri, ed è la forma che
si usa qui. Con fattore di eliminazione
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

C'è un dettaglio pratico che separa il torneo descritto qui da quello che gira
davvero quando le prove sono distribuite su molte macchine.

Il successive halving, così come lo abbiamo raccontato, è **sincrono**: per
decidere chi passa il turno aspetta che *tutte* le prove di quel turno abbiano
finito, come una gara in cui la premiazione si fa solo quando è arrivato anche
l'ultimo. Su una macchina sola non cambia niente, perché le prove si fanno
comunque una per volta. Su cento macchine è uno spreco: novantanove restano a
girarsi i pollici aspettando la centesima, e basta una candidata lenta a
bloccare tutto.

La versione **asincrona** (nota come ASHA) toglie la barriera, e lo fa
cambiando il criterio di promozione. Invece di chiedere «sei fra le migliori
tre di nove?», che è una domanda a cui non si può rispondere finché le nove non
sono arrivate, chiede: «rispetto a chi è già passato di qui prima di te, saresti
nel terzo migliore?». È una classifica parziale, fatta sulle candidate finite
fino a quel momento, e si può rispondere subito. Chi supera la prova viene
promosso all'istante, e la macchina che si libera prende il lavoro successivo.
Si accetta di decidere con informazione incompleta in cambio di non lasciare
nessuno fermo, ed è quasi sempre il baratto giusto.

## Cercare con giudizio: l'ottimizzazione bayesiana

Griglia, caso e tornei condividono un ultimo difetto, il più profondo: ogni
prova **ignora ciò che le precedenti hanno scoperto**. Se dieci esperimenti
hanno già mostrato che con un passo troppo lungo l'errore, invece di scendere,
schizza fuori controllo (si dice che il modello *diverge*: rimbalza da un
fianco all'altro della valle e se ne allontana),
l'undicesimo estratto a caso può cascarci di nuovo. L'**ottimizzazione
bayesiana** {cite}`snoek2012practical` tratta la ricerca degli iperparametri
come un problema di apprendimento a sua volta: impara a prevedere *quale
punteggio darà una combinazione di manopole prima di provarla*, e usa quella
previsione per decidere dove provare. Si finisce così con due modelli in
scena, uno dentro l'altro: quello che vogliamo addestrare, e questo secondo che
studia il primo dall'esterno. Il nome «bayesiana» viene dal modo in cui il
secondo aggiorna le sue convinzioni ogni volta che arriva una prova nuova, e
porta il nome del reverendo Thomas Bayes.

`````{tab} Elementare

Pensa a un geologo che cerca l'acqua potendo scavare pochi pozzi, perché ogni
trivellazione costa cara. Dopo tre pozzi non sceglie il quarto a caso: disegna
una mappa ("qui l'acqua c'era a dieci metri, là il terreno era secco"),
completa di zone d'ombra dove non sa ancora nulla. Il quarto pozzo lo piazza
dove la *promessa* è massima: un po' dove la mappa dice bene (sfruttare ciò
che sa), un po' dove la mappa è bianca (esplorare ciò che ignora).
L'ottimizzazione bayesiana funziona così: dopo ogni addestramento aggiorna la
sua mappa del punteggio e sceglie la combinazione successiva chiedendosi *di
quanto mi aspetto di battere il mio record, se provo qui?* La domanda ha un
nome, **miglioramento atteso** (*expected improvement*), ed è una domanda sola
che tiene insieme le due esigenze. La risposta è alta dove la mappa promette
bene, e anche dove la mappa è bianca, perché lì il record potrebbe essere
battuto di parecchio. Ed è bassa, cioè quasi zero, dove il terreno è stato già
scavato e si è rivelato secco: là non c'è più niente da sapere e niente da
sperare, e il metodo smette di andarci senza che nessuno glielo debba dire.

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

scikit-learn offre le due strategie di base con la stessa forma d'uso:
`GridSearchCV` e `RandomizedSearchCV` racchiudono in un solo oggetto il giro
«prova una combinazione, valutala in cross-validation, tieni la migliore», e
lo ripetono da soli. Le proviamo
su un classificatore SVC, una *support vector machine*, che qui usiamo come
scatola nera: ci basta sapere che ha due manopole delicate, `C` e `gamma`. Sono
entrambe **parametri di scala**, e vuol dire che quello che conta è il loro
ordine di grandezza, non la differenza fra un valore e l'altro: fra $0{,}001$ e
$0{,}01$ c'è lo stesso salto che fra $1$ e $10$, mentre fra $1$ e $1{,}5$ non
c'è quasi niente. È per questo che i loro valori si provano moltiplicandoli per
dieci ogni volta invece che sommando una costante. (Il dataset è quello delle
cifre manoscritte incluso in scikit-learn.)

```python
from scipy.stats import loguniform
from sklearn.datasets import load_digits
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     train_test_split)
from sklearn.svm import SVC

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)   # il test resta nel cassetto

# Grid search: 4 x 4 = 16 combinazioni, x 5 blocchi di CV = 80 addestramenti
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

Il trucco di `loguniform` merita una riga, perché tornerà ogni volta che si
sceglie un learning rate: **si sorteggia l'esponente, non il valore**. Invece
di estrarre un numero a caso fra $0{,}00001$ e $1$, che nel $99\%$ dei casi
darebbe qualcosa di grande, si estrae un numero a caso fra $-5$ e $0$, poniamo
$-3{,}2$, e si usa $10^{-3{,}2}$. Così ogni ordine di grandezza ha le stesse
probabilità degli altri.

scikit-learn implementa
anche il successive halving (`HalvingGridSearchCV` e `HalvingRandomSearchCV`,
ancora marcati come sperimentali); per l'ottimizzazione bayesiana e i tornei
in versione moderna la libreria di riferimento è **Optuna**, in cui lo spazio
di ricerca si descrive direttamente nel codice e le prove peggiori vengono
interrotte in corsa.

## Le avvertenze sul foglietto

Tre avvertenze, prima di chiudere.

La prima è il **costo**. Ogni combinazione provata non costa un addestramento
ma cinque, perché la si giudica in cross-validation su cinque blocchi, e quel
fattore cinque non lo toglie nessun algoritmo: i metodi furbi lo spendono
meglio, non lo evitano. Griglia e caso hanno almeno il vantaggio di spalmarsi
su tante macchine senza sforzo; l'ottimizzazione bayesiana no, perché è fatta
per scegliere la prossima prova dopo aver visto l'esito della precedente.
Esistono varianti che ne lanciano un gruppo alla volta (già Snoek e colleghi ne
proponevano una {cite}`snoek2012practical`), ma ogni prova, presa da sola,
rende meno.

La seconda è la **riproducibilità**. Un computer non sa tirare a caso davvero:
produce numeri che *sembrano* casuali partendo da un numero iniziale, il
**seme** (in inglese *seed*, il `random_state` del codice qui sopra). Stesso
seme, stessa sequenza di numeri «a caso», stesso risultato domani e sul
computer di un altro; seme non fissato, esito diverso a ogni esecuzione, e
allora nessuno può ripetere il tuo esperimento, nemmeno tu. Alla stessa
famiglia appartiene un'altra dimenticanza: dire «il metodo A batte il metodo B»
senza dichiarare in quale intervallo si è cercato e quante prove si sono fatte
non è un confronto, è un aneddoto, perché a parità di tempo il vincitore può
capovolgersi.

La terza avvertenza è la più subdola.

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

Le tre avvertenze hanno un'unica morale, ed è il modo migliore di chiudere il
cerchio aperto da Rahimi: **una ricerca degli iperparametri è essa stessa un
addestramento**, e come ogni addestramento può imparare a memoria. Chi la
tratta come tale, dichiarando spazio di ricerca, budget e semi, e tenendo il
test chiuso fino all'ultimo, ha già tolto dall'alchimia la parte che faceva più
danno.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un **iperparametro** è una manopola che giriamo noi prima di cominciare e che
  l'addestramento non tocca. Si sceglie provando, e si giudica sui dati di
  prova, mai su quelli d'esame.
- Provare **tutte le combinazioni** è la cosa più ovvia e la meno praticabile:
  la macchina del caffè con quattro manopole a cinque livelli chiede 3 125
  assaggi. Ogni manopola in più *moltiplica* le prove.
- Provarle **a caso** conviene quasi sempre, ed è la cosa che sorprende di più:
  se una sola manopola conta davvero (la sintonia, non il volume), nove
  tentativi a caso provano nove sintonie diverse, mentre nove disposti in
  griglia ne provano tre.
- I **tornei a eliminazione** danno a tutti un allenamento breve, poi solo ai
  migliori uno lungo: si spende dove serve. Il rischio è tagliare fuori i
  «diesel», quelli che partono piano e finirebbero forte.
- Il metodo più furbo **impara dalle prove già fatte**, come il geologo che
  sceglie dove scavare il prossimo pozzo: un po' dove la mappa promette bene,
  un po' dove la mappa è ancora bianca.
- Il punteggio del vincitore è **troppo bello**: fra mille che lanciano una
  moneta, qualcuno fa nove teste per fortuna. Il numero da raccontare al mondo
  si misura una volta sola, alla fine, sui dati d'esame rimasti intatti.
```

`````

`````{tab} Superiore

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
  budget a molti, molto budget a pochi
  {cite}`karnin2013almost,jamieson2016non,li2018hyperband`.
- L'**ottimizzazione bayesiana** usa un surrogato (tipicamente un processo
  gaussiano) e una funzione di acquisizione per imparare dalle prove passate
  {cite}`snoek2012practical`.
- Il punteggio del vincitore è **ottimista**: numero finale solo dal test
  intatto, seed fissati, spazio e budget dichiarati.
```

`````
