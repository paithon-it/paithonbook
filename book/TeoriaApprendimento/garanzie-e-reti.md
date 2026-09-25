# Quello che le garanzie non spiegano

Le garanzie viste fin qui hanno tutte la stessa forma: l'errore vero sta sotto
l'errore sugli esempi più un termine che misura la ricchezza della famiglia di
regole. Per una rete neurale quel termine si può stimare con la {doc}`prova
delle monete <rademacher-margine>`, e il risultato di quella prova, fatta sul
serio su reti vere, è l'esperimento più discusso della teoria
dell'apprendimento recente.

Nel 2017 Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht e Oriol
Vinyals presero le reti neurali più brave a classificare le immagini e le
addestrarono su etichette tirate a sorte {cite}`zhang2017understanding`. Le
reti le impararono tutte, o quasi, e senza fatica. Il lavoro si intitolava, in
italiano, «capire il deep learning chiede di ripensare la generalizzazione», e
il titolo diceva già la conclusione: le reti vanno bene su esempi nuovi per
ragioni che la teoria non aveva ancora capito.

## Uno studente che impara qualunque cosa

`````{tab} Elementare

Uno studente prepara un esame a quiz. Lo studente è la rete neurale, e tutto
quello che sarebbe capace di imparare è la sua famiglia di regole. Ha un foglio
con trecento domande, ciascuna con dieci risposte possibili, e accanto a ognuna
quella giusta, e le impara; all'esame, su domande nuove dello stesso tipo, ne
azzecca novantacinque su cento. Poi gli si dà un altro foglio con le stesse
domande e le risposte rimescolate a caso. Le impara lo stesso, tutte, e ci
mette solo cinque o sei volte il tempo di prima, quando ci si aspetterebbe
un'eternità da chi deve mandare a memoria trecento risposte senza senso.
All'esame, su domande nuove, va poco meglio di chi tira a indovinare.

Lo studente è lo stesso nei due casi, con la stessa memoria e la stessa
capacità. La prova delle monete, fatta su di lui con un terzo foglio in cui le
risposte sono un sì o un no tirati con una moneta, dà il risultato peggiore
possibile per la garanzia: impara qualunque sequenza di sì e di no gli si metta
davanti. Tutte le garanzie di prima, che guardano solo quanto è capace lo
studente, dicono quindi la stessa cosa nei due casi, cioè niente. Eppure nel
primo caso lo studente va benissimo. Quello che fa la differenza non può essere
la sua capacità: deve stare nel foglio che ha studiato e nel modo in cui lo ha
studiato.

Il foglio conta anche in un modo più sottile. Nel rimescolamento qualche
risposta è rimasta al suo posto per caso, una su dieci circa, perché dieci
erano le risposte possibili, ed è per questo che lo studente, all'esame, fa
appena un po' meglio di chi tira a indovinare. Se invece ogni risposta del
foglio è sbagliata apposta, mai quella giusta, lo studente la impara lo stesso
e poi fa peggio del caso, perché ha imparato l'unica cosa vera che quel foglio
diceva: la risposta giusta non è mai quella, e sulle domande nuove, che
somigliano a quelle del foglio, evita proprio lei.

E per andare bene sul foglio vero non ha avuto bisogno di nessun freno, cioè
di nessuna delle regole per scoraggiare la memoria che la
{doc}`regolarizzazione </MachineLearning/overfitting-validazione>` mette a un
modello. Freni del genere a volte aiutano, ma non sono loro a fare la
differenza.

`````

`````{tab} Superiore

Gli esperimenti di Zhang e colleghi, su CIFAR-10 con architetture standard e
SGD, stabiliscono tre cose, e le si legge meglio nelle loro parole. Le reti
«adattano facilmente un'etichettatura casuale dei dati di addestramento» fino
all'errore zero, e lo fanno anche sostituendo le immagini con rumore privo di
struttura; su ImageNet, con un milione di etichette casuali su mille classi,
Inception v3 arriva al $95{,}2\%$ senza nessuna taratura. L'ottimizzazione
resta facile, e il tempo di addestramento cresce «solo di un piccolo fattore
costante» rispetto alle etichette vere. E la regolarizzazione esplicita «può
migliorare la generalizzazione, ma non è né necessaria né da sola sufficiente
a controllarla». Accanto agli esperimenti, una costruzione: una rete ReLU a
due strati con $2n+d$ pesi realizza qualunque etichettatura di $n$ esempi in
$d$ dimensioni, e per i trecento esempi dell'esperimento sulle cifre ne
bastano $664$, dove la rete ne ha $38\,410$.

La conseguenza per le garanzie viste fin qui è immediata. Se l'architettura
realizza qualunque etichettatura di $S$, allora per la famiglia $\mathcal{H}$
delle funzioni che essa calcola il sup in $\hat{\mathfrak{R}}_S(\mathcal{H})$
vale $1$ per ogni $\boldsymbol\sigma$, e $\hat{\mathfrak{R}}_S(\mathcal{H})=1$:
il bound di Rademacher diventa $R(h)\le\hat{R}_S(h)+1+\dots$, vuoto. Lo stesso
vale per il bound VC, perché una famiglia che realizza tutte le etichettature
di $m$ punti li frantuma, e ha quindi dimensione VC almeno $m$. La dimensione
di un'architettura fissata resta finita per le attivazioni in uso (per le ReLU
è $\Theta(WL\log W)$, con $W$ pesi e $L$ strati
{cite}`bartlett2019nearly`; per la rete del blocco, $38\,410$ pesi su $300$
esempi, è dell'ordine del milione), e il teorema fondamentale continua a
valere: la famiglia è apprendibile, ma la garanzia chiede esempi in numero
proporzionale alla dimensione, che supera già quelli a disposizione. E il punto
decisivo non è che i bound siano larghi, ma che sono **gli stessi** per le
etichette vere e per quelle casuali, perché dipendono solo da $\mathcal{H}$ e
da $S$: non possono distinguere il caso in cui la rete generalizza da quello in
cui non lo fa. Una spiegazione deve quindi dipendere dall'ipotesi $h_S$
effettivamente trovata, cioè dall'algoritmo, o dalla distribuzione che ha
prodotto le etichette.

`````

Il blocco rifà l'esperimento in piccolo, con una rete a un solo strato
nascosto di 512 unità (il modello del {doc}`capitolo sulle reti neurali
</RetiNeurali/overview>`, qui preso già fatto dalla libreria) su trecento
cifre scritte a mano, e la addestra finché non indovina tutti gli esempi. Lo fa
con le etichette vere, con le stesse etichette rimescolate, con etichette
scelte sbagliate apposta, e infine con segni $\pm1$ tirati a sorte, che sono la
prova delle monete in senso stretto.

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

X, y = load_digits(return_X_y=True)                  # cifre scritte a mano, 8x8
X = X / 16.0
X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=300, random_state=0)
rng = np.random.default_rng(0)

def addestra_finche_impara(etichette, classi, tetto=2000):
    """Epoche su epoche finché la rete non indovina tutti gli esempi."""
    rete = MLPClassifier(hidden_layer_sizes=(512,), alpha=0.0,
                         learning_rate_init=0.01, batch_size=50, random_state=0)
    for epoca in range(1, tetto + 1):
        rete.partial_fit(X_tr, etichette, classes=classi)
        if rete.score(X_tr, etichette) == 1.0:
            break
    return rete, epoca

rimescolate = rng.permutation(y_tr)                  # le stesse etichette, in disordine
sbagliate = (y_tr + rng.integers(1, 10, size=len(y_tr))) % 10   # mai quella giusta
print(f"etichette rimaste giuste nel rimescolamento: {(rimescolate == y_tr).mean():.1%}")
for nome, etichette in (("vere", y_tr), ("rimescolate", rimescolate),
                        ("sempre sbagliate", sbagliate)):
    rete, epoche = addestra_finche_impara(etichette, np.arange(10))
    print(f"etichette {nome:>16}: addestramento {rete.score(X_tr, etichette):.1%},"
          f" prova {rete.score(X_te, y_te):.1%}, {epoche} epoche")

# la complessità di Rademacher della rete su questi 300 esempi: segni tirati a sorte
correlazioni = []
for _ in range(3):
    sigma = rng.choice([-1, 1], size=len(y_tr))
    rete, _ = addestra_finche_impara(sigma, np.array([-1, 1]))
    correlazioni.append(np.mean(sigma * rete.predict(X_tr)))
print(f"correlazione con i segni casuali, tre sorteggi: {np.round(correlazioni, 2)}")
```

```text
etichette rimaste giuste nel rimescolamento: 11.7%
etichette             vere: addestramento 100.0%, prova 95.2%, 7 epoche
etichette      rimescolate: addestramento 100.0%, prova 11.8%, 43 epoche
etichette sempre sbagliate: addestramento 100.0%, prova 1.9%, 42 epoche
correlazione con i segni casuali, tre sorteggi: [1. 1. 1.]
```

Con le etichette vere la rete impara in sette epoche (un'epoca è un passaggio
su tutti i trecento esempi) e sui dati nuovi indovina il $95{,}2\%$; con quelle
rimescolate impara lo stesso tutto, in sei volte il tempo, e sui dati nuovi
scende all’$11{,}8\%$, appena sopra il dieci per cento del caso perché
l’$11{,}7\%$ delle etichette era rimasto giusto. Con le etichette sempre
sbagliate scende all’$1{,}9\%$, sotto il caso. L'ultima riga è la complessità di
Rademacher della rete su quei trecento esempi, stimata sui tre sorteggi: vale
$1$, il massimo possibile, con le monete inseguite tutte, e a quel valore ogni
garanzia che dipenda solo dalla famiglia diventa vuota. Per molte famiglie
trovare la regola migliore su segni tirati a sorte è un conto intrattabile; per
una famiglia così ricca è facile, perché la rete trova da sola una regola
perfetta per ogni sorteggio, e il massimo è raggiunto.

## Dove cercare la spiegazione

Se la ricchezza della famiglia non spiega la differenza, la spiegazione deve
dipendere da qualcosa che cambia fra le etichette vere e quelle a caso. Le
piste più seguite guardano la soluzione trovata, la sua solidità e il modo in
cui l'algoritmo ci arriva, e nessuna è ancora una risposta completa.

`````{tab} Elementare

Una pista guarda la fatica. Per il foglio vero allo studente bastano
poche regole semplici, perché le risposte giuste seguono un criterio; per il
foglio rimescolato deve ricordare trecento risposte una per una. Misurare
quanto è complicata la soluzione che ha trovato, e non quanto avrebbe potuto
esserlo, distingue i due casi. È l'idea della strada larga delle SVM portata
dentro una rete: là contava quanto la strada fosse larga rispetto all'ingombro
dei punti, qui quanto i pesi della rete siano piccoli rispetto al margine con
cui separa le risposte, e la misura è più grande per le risposte a caso.

Un'altra guarda quanto la preparazione è salda. Se si prende lo studente e lo
si confonde un po', cambiandogli qualche ricordo a caso, e continua ad andare
bene, la sua bravura non è un colpo di fortuna: vanno bene anche tutte le copie
di lui appena diverse. Una garanzia che vale per tutte quelle copie insieme,
invece che per uno studente solo, si può calcolare, e per una rete vera dà
numeri che finalmente dicono qualcosa: promette meno di un errore su sei,
mentre la rete ne fa uno su trenta. Larga, ma non vuota.

L'ultima guarda il modo di studiare. Lo studente parte da una mente
sgombra, fa passi piccoli e si ferma alla prima spiegazione che regge su tutto
il foglio: quando si ferma ha cambiato le proprie idee il meno possibile, e
fra le infinite spiegazioni possibili quella che si scosta meno dal niente
tende a essere la più semplice. È la stessa ragione per cui un modello con
molte più manopole del necessario non impazzisce, e sta dietro alla
{ref}`doppia discesa <sec-doppia-discesa>`; per le reti profonde è una
congettura con buone prove, non una cosa dimostrata.

`````

`````{tab} Superiore

**Bound di norma e di margine.** Bartlett, Foster e Telgarsky
{cite}`bartlett2017spectrally` dimostrano un bound di generalizzazione
multiclasse che scala con la *complessità spettrale* della rete normalizzata
per il margine: il prodotto delle norme spettrali delle matrici dei pesi (una
costante di Lipschitz della rete) per un fattore correttivo, diviso per il
margine di classificazione. È la generalizzazione del bound di margine delle
funzioni lineari nella linea aperta da Bartlett nel 1998, quando mostrò che a
contare è la taglia dei pesi e non quella della rete
{cite}`bartlett1998sample`, e ha la proprietà che ai bound visti fin qui
mancava, perché dipende dai pesi trovati.
Sperimentalmente, su AlexNet addestrata con SGD su MNIST e CIFAR-10, il bound,
le costanti di Lipschitz e l'eccesso di rischio sono direttamente correlati, e
con etichette casuali crescono tutti: SGD sceglie predittori la cui
complessità scala con la difficoltà del compito. Il valore numerico del bound
resta però lontano dall'errore osservato.

**PAC-Bayes.** Invece di una sola ipotesi si considera una distribuzione $Q$
sui pesi, e il bound PAC-Bayes, proposto da McAllester alla fine degli anni
Novanta {cite}`mcallester1999some` e raffinato in molte varianti, controlla
$\mathbb{E}_{h\sim Q}[R(h)]$ con $\mathbb{E}_{h\sim Q}[\hat{R}_S(h)]$ più un
termine che cresce con la divergenza di Kullback-Leibler fra $Q$ e una
distribuzione a priori fissata prima dei dati. Una rete i cui pesi si possono
perturbare parecchio senza che l'errore empirico salga ammette una $Q$ larga e
quindi un termine piccolo. Dziugaite e Roy {cite}`dziugaite2017computing`,
ottimizzando direttamente il bound, hanno ottenuto il primo bound non vacuo per
una rete profonda stocastica con molti più parametri che esempi. Su MNIST a due
classi con etichette vere, per una rete con uno strato nascosto di seicento
unità, l'errore garantito della rete stocastica è $0{,}161$, mentre sui dati di
prova quella rete sbaglia il $3{,}4\%$ (e la rete deterministica da cui nasce
l’$1{,}8\%$): non vacuo, e ancora parecchie volte sopra la realtà. Con le
etichette casuali lo stesso bound vale $1{,}352$, cioè si rifiuta di garantire
qualunque cosa, ed è il comportamento che ci si aspetta da una misura che
dipende dall'ipotesi trovata.

**Il bias implicito dell'algoritmo.** La discesa del gradiente non esplora
$\mathcal{H}$ in modo uniforme. Sui minimi quadrati sovraparametrizzati, partita
da zero, converge alla soluzione interpolante di norma minima, un fatto
classico dell'algebra lineare; sulla regressione logistica
con dati separabili, con passo abbastanza piccolo e per ogni perdita a coda
esponenziale come la logistica, la direzione dei pesi converge a quella di
massimo margine, ed è il risultato di Soudry e colleghi
{cite}`soudry2018implicit`: lentamente, perché $\mathbf{w}(t)$ cresce come
$\log t$ lungo quella direzione. La generalizzazione diventa allora una
proprietà della coppia algoritmo più dati, e la
{ref}`doppia discesa <sec-doppia-discesa>` è il fenomeno che questa lettura
spiega: per i modelli lineari con un teorema, per le reti profonde con
un'ipotesi che ha buone prove sperimentali.

Nessuna delle tre piste dà oggi, per una rete grande, un numero che predica
l'errore di prova con la precisione con cui un intervallo di confidenza lo
stima su un insieme tenuto da parte. La teoria dell'apprendimento, per le
reti, è passata dal ruolo di garanzia a quello di spiegazione, e la garanzia
pratica resta la misura su dati mai visti.

`````

Il bilancio, per chi deve usare questi strumenti, è netto. Per le famiglie di
regole semplici la teoria dà un numero, e il numero regge; per le reti dà un
modo di ragionare e qualche misura che si muove nel verso giusto, e l'errore
lo si continua a misurare su esempi tenuti da parte.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Una rete neurale grande impara anche risposte tirate a sorte, tutte, in un
  tempo solo cinque o sei volte più lungo: la prova delle monete su di lei dà
  il massimo.
- Quindi le garanzie che guardano solo quanto è capace la famiglia di regole
  dicono lo stesso per risposte vere e per risposte a caso, e non spiegano
  perché con le vere la rete va bene.
- La spiegazione va cercata nella soluzione trovata (quanto è semplice, quanto
  è salda) e nel modo di arrivarci; la garanzia pratica resta la prova su
  esempi mai visti.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Zhang e colleghi (2017): le reti adattano etichette casuali con errore di
  addestramento nullo su CIFAR-10 (e al $95\%$ su ImageNet), in tempo solo un
  piccolo fattore più lungo, e la regolarizzazione esplicita non è né
  necessaria né sufficiente.
- Per l'architettura, $\hat{\mathfrak{R}}_S(\mathcal{H}) = 1$ e il bound
  VC è vuoto: i bound che dipendono solo da $\mathcal{H}$ e $S$ non
  distinguono etichette vere e casuali.
- Piste dipendenti da $h_S$: norme spettrali normalizzate per il margine,
  PAC-Bayes (primo bound non vacuo, $0{,}161$ contro un errore osservato di
  pochi punti), bias implicito della discesa del gradiente; nessuna predice
  ancora l'errore con precisione.
```
`````

Dal pollo di Russell in poi il filo è uno solo: un modello che ha visto degli
esempi fa una scommessa sul prossimo, e la teoria dell'apprendimento dice
quanto la scommessa è sicura. Per le famiglie di regole semplici lo dice con un
numero; per le reti neurali, per ora, dice soprattutto dove guardare. Quelle
reti, fin qui prese già fatte da una libreria, le costruisce pezzo per pezzo il
{doc}`capitolo sulle reti neurali </RetiNeurali/overview>`, a partire dal
neurone più semplice.
