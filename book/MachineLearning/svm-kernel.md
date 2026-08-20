# Il kernel trick: separare l'inseparabile

La SVM del massimo margine traccia frontiere *diritte*. E se le due classi sono
intrecciate in modo che nessuna retta le separi? Pensa a un
bersaglio, con una classe al centro e l'altra tutt'intorno ad anello: nessuna
riga tirata su quel foglio le divide. Qui
entra in gioco l'idea più affascinante di tutta la storia delle SVM, quella
che le ha rese celebri: il **kernel trick**.

`````{tab} Elementare

Prendi il bersaglio: cerchio interno di una classe, anello esterno dell'altra.
Sul foglio, piatto, nessuna retta li separa. Ma immagina di *sollevare* ogni
punto in aria di un'altezza pari a quanto è lontano dal centro: i punti del
cerchio interno, vicini al centro, restano bassi; quelli dell'anello, lontani,
salgono in alto. Ora le due classi stanno a quote diverse, e un semplice
*piano orizzontale* (una lastra di vetro infilata a mezz'aria) le separa
nettamente. Non abbiamo cambiato i punti: li abbiamo guardati in uno spazio
con una dimensione in più, e lì il problema è diventato lineare.

Il guaio è che sollevare i punti costa. Nei casi utili le dimensioni da
aggiungere non sono una ma migliaia, a volte infinite, e nessun calcolatore
può reggerle. E qui sta il trucco: il passo precedente ci ha lasciato una
frontiera che si calcola **usando soltanto le ombre a due a due**, cioè un
numero per ogni coppia di punti. Se sappiamo produrre direttamente quei numeri
*come sarebbero dopo il sollevamento*, il sollevamento non serve più farlo.

La regola che li produce si chiama **kernel**, ed è tutto ciò che serve. Si
sceglie il kernel, e lo spazio sollevato resta un'idea: non lo si costruisce
mai.

`````

`````{tab} Superiore

L'idea è mappare ogni esempio in uno spazio di dimensione maggiore con una
funzione $\phi$, e cercare l'iperpiano *lì*. Per il bersaglio basta aggiungere
la feature $r^2 = x_1^2 + x_2^2$:

$$
\phi\,(x_1, x_2) = \big(x_1,\ x_2,\ x_1^2 + x_2^2\big).
$$

Nello spazio a tre dimensioni la classe interna (piccolo $r^2$) e quella esterna
(grande $r^2$) sono separate da un piano orizzontale a un'altezza-soglia. Il
problema, così com'è, sembra però costoso: se $\phi$ manda in uno spazio a
migliaia di dimensioni, calcolare e conservare tutti quei
$\phi(\mathbf{x}_i)$ diventa proibitivo.

Il **kernel trick** è l'osservazione che salva tutto, ed è la ragione vera
per cui [la strada più larga](svm.md) ha percorso il duale passo per passo:
là dentro, e nella regola
di decisione, gli esempi compaiono *solo* attraverso prodotti scalari
$\phi(\mathbf{x})^\top\phi(\mathbf{z})$. Se esiste una funzione $k$ che
calcola quel prodotto scalare direttamente dalle coordinate originali,

$$
k(\mathbf{x}, \mathbf{z}) = \phi(\mathbf{x})^\top\phi(\mathbf{z}),
$$

allora non serve mai costruire $\phi$: si lavora nello spazio ad alta dimensione
*senza mai visitarlo*. La funzione $k$ è il **kernel**
{cite}`scholkopf2002learning`. I più usati:

$$
\begin{aligned}
&\text{lineare:} && k(\mathbf{x},\mathbf{z}) = \mathbf{x}^\top \mathbf{z}, \\
&\text{polinomiale:} && k(\mathbf{x},\mathbf{z})
   = (\mathbf{x}^\top \mathbf{z} + c)^{d}, \\
&\text{RBF / gaussiano:} && k(\mathbf{x},\mathbf{z})
   = \exp\!\big(-\gamma\,\lVert \mathbf{x} - \mathbf{z}\rVert^2\big),
\end{aligned}
$$

dove $d$ è il grado del polinomio, $c \ge 0$ un termine costante e $\gamma > 0$
il parametro di ampiezza del kernel gaussiano, che **stringe** la campana al
crescere. La larghezza è la deviazione standard equivalente $\sigma$, e vale
$\sigma = 1/\sqrt{2\gamma}$, cioè $\gamma = 1/(2\sigma^2)$: $\gamma$ è quindi
l'inverso del *quadrato* della larghezza, e per dimezzare la campana va
quadruplicato, non raddoppiato. $\gamma$ grande, campana stretta. Il kernel RBF
corrisponde a uno spazio $\phi$ di dimensione *infinita*: sarebbe impossibile
da costruire, eppure $k$ si calcola in una riga.

Un'ultima clausola, quella che fa del trucco un teorema invece che una
speranza. La frase «se esiste una funzione $k$ che calcola quel prodotto
scalare» rovescia l'ordine dei fatti: in pratica non si parte da $\phi$ per
cercare $k$, si **sceglie** $k$ e si spera che un $\phi$ esista. Esiste se e
solo se $k$ è simmetrica e **semidefinita positiva**, cioè se ogni matrice di
Gram $\mathbf{K}$, quella di elementi $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$,
ha autovalori non negativi: è il
teorema di Mercer {cite}`scholkopf2002learning`, ed è la ragione per cui i
kernel non si inventano a piacere. Se $k$ non lo è, il duale smette di essere
concavo e il solutore sta risolvendo un problema diverso da quello che si
crede. Non ogni «misura di somiglianza» è un kernel, ed è l'errore più comune
di chi prova a scriversene uno.

`````

```{figure} ../figures/svm-kernel-trick.svg
:name: fig-svm-kernel
:alt: "A sinistra, in due dimensioni, un disco di punti teal circondato da un anello di punti terracotta, non separabili da una retta; una freccia phi al centro. A destra, dopo la mappa, gli stessi punti giacciono su una parabola: gli interni in basso, gli esterni in alto, e una retta orizzontale tratteggiata li separa."
:width: 100%

Il kernel trick. A sinistra due classi che nessuna retta separa: un disco al
centro e un anello intorno. A destra gli stessi punti dopo il sollevamento, con
l'altezza pari alla distanza dal centro elevata al quadrato: i punti del disco
restano in basso, quelli dell'anello salgono, e una retta orizzontale basta a
dividerli.
```

Come illustra {numref}`fig-svm-kernel`, ciò che era un anello inseparabile
diventa, dopo il sollevamento, un problema lineare banale.

Fra i kernel c'è un preferito, e si chiama **RBF** (sono le iniziali di
*radial basis function*, «funzione a base radiale»: radiale perché guarda solo
la distanza fra due punti, in qualunque direzione). Ha una manopola sola, che
nelle formule si chiama $\gamma$, «gamma», e conviene capire che cosa fa,
perché è il **raggio d'influenza** di ogni punto.

`````{tab} Elementare

Il kernel RBF si racconta meglio con i lampioni. Pensa a ogni punto come a un
lampione acceso di notte: illumina bene chi
gli sta accanto, sempre meno chi si allontana, per niente chi è lontano, e il
numero che il kernel restituisce per due punti è quanta luce dell'uno arriva
all'altro. Sta fra $0$ e $1$, e vale $1$ solo per il lampione stesso.

Chiamiamo *portata* del lampione la distanza alla quale la
luce è scesa a poco più di un terzo, cioè a $0{,}37$: mettiamo un metro e mezzo.
Chi sta a un metro e mezzo si vede ancora. E chi sta al doppio, a tre metri?
Non riceve la metà della luce, e nemmeno un terzo. Il motivo è che a decidere
non è la distanza ma il suo **quadrato**, e raddoppiando la distanza il
quadrato si moltiplica per quattro: è come se quel lampione, per lui, fosse
lontano quattro portate invece di una. La luce che gli arriva è quindi $0{,}37$
elevato alla quarta, cioè circa $0{,}018$: meno di due
centesimi, praticamente buio. Ecco perché il raggio d'influenza di un punto
finisce così bruscamente.

La manopola $\gamma$ decide quanto è stretto il cono di luce, ed è la portata
**al contrario**: $\gamma$ grande, luce corta, e la frontiera viene frastagliata
perché ogni punto comanda solo nel suo cortile (rischio di imparare il rumore);
$\gamma$ piccolo, luce lunga, e la frontiera esce morbida.

`````

`````{tab} Superiore

Vediamolo con i numeri, scegliendo $\gamma = 0{,}5$:

- due punti *vicini*, $\mathbf{x}=(2,2)$ e $\mathbf{z}=(3,3)$, distano
  $\lVert \mathbf{x}-\mathbf{z}\rVert^2 = 1^2 + 1^2 = 2$, quindi
  $k(\mathbf{x},\mathbf{z}) = e^{-0{,}5\cdot 2} = e^{-1} \approx 0{,}37$:
  si «vedono» bene;
- due punti *lontani*, $\mathbf{x}=(2,2)$ e $\mathbf{z}=(0,0)$, distano
  $\lVert \mathbf{x}-\mathbf{z}\rVert^2 = 2^2 + 2^2 = 8$, quindi
  $k(\mathbf{x},\mathbf{z}) = e^{-0{,}5\cdot 8} = e^{-4} \approx 0{,}018$:
  quasi si ignorano.

Con $\gamma$ grande la campana si stringe, ogni punto influenza solo i vicinissimi
e la frontiera si fa frastagliata (varianza alta, rischio overfitting); con
$\gamma$ piccolo la campana si allarga, l'influenza è a lungo raggio e la
frontiera si liscia.

`````

Insieme a $C$, il parametro $\gamma$ è l'altra manopola da tarare per
validazione.

## Non solo classificare: la regressione con le SVM

Lo stesso principio si ribalta per la **regressione** (SVR, *Support Vector
Regression*). Nella classificazione la SVM vuole il corridoio più largo *tra*
le classi; nella regressione vuole un tubo che contenga *quanti più punti
possibile*.

`````{tab} Elementare

Invece di penalizzare ogni piccolo scarto tra previsione e valore vero (come
fa la regressione lineare classica) la SVR disegna un «tubo» di tolleranza
attorno alla curva: finché un punto ci sta dentro, l'errore conta *zero*.
Vengono penalizzati solo i punti che sporgono dal tubo, e solo per quanto
sporgono. È un modo indulgente di adattare i dati: non insegue le piccole
oscillazioni, si preoccupa solo degli scostamenti seri.

`````

`````{tab} Superiore

Si fissa una tolleranza $\epsilon > 0$ e si usa la **loss
$\epsilon$-insensitive**, nulla dentro il tubo e lineare fuori:

$$
L_\epsilon\big(y,\, f(\mathbf{x})\big)
= \max\!\big(0,\ |y - f(\mathbf{x})| - \epsilon\big).
$$

Gli errori entro $\pm\epsilon$ non vengono penalizzati; oltre, la penalità
cresce linearmente. Il parametro $\epsilon$ fissa l'ampiezza del tubo, mentre
$C$ regola come sempre il compromesso tra piattezza del modello e violazioni.
Anche qui vale il kernel trick, così la SVR può adattare curve non lineari
esattamente come la SVM classifica frontiere non lineari.

`````

## Una classe sola: novelty e anomaly detection

C'è un'ultima variante, e risponde a una domanda diversa: e se avessimo esempi
di *una sola* classe? Vogliamo imparare com'è fatto il «normale» (transazioni
regolari, macchinari sani, traffico di rete legittimo) per poi accorgerci di
ciò che se ne discosta. È il problema della **novelty detection** (riconoscere
il nuovo) e dell’**anomaly detection** (riconoscere il guasto), e si lega a
quel tema dei dati fuori distribuzione di cui si occupa la
{doc}`sezione sui dati che cambiano <dati-che-cambiano>`: individuare gli
input troppo lontani da ciò che il modello ha visto, invece di predire con
finta sicurezza.

`````{tab} Elementare

Immagina di aver visto migliaia di transazioni oneste con la carta di credito
e nemmeno una frode. Non puoi addestrare un classificatore «onesto contro
frode»: la seconda classe non ce l'hai. La **one-class SVM** ribalta il
problema: impara a disegnare, attorno ai dati normali, il «recinto» più
stretto che li racchiude tutti. Da quel momento, ogni nuova transazione che
cade *fuori* dal recinto è sospetta: non perché somigli a una frode nota, ma
perché non somiglia a nulla di normale. Una manopola, $\nu$, dice più o meno
quale frazione di dati ci aspettiamo che finisca fuori (le anomalie
tollerate). Serve per rilevare frodi, guasti di macchinari, intrusioni
informatiche, difetti in una linea di produzione: ovunque gli esempi «anomali»
siano rari o non ancora visti.

`````

`````{tab} Superiore

La one-class SVM di Schölkopf e colleghi {cite}`scholkopf2001estimating`
adatta l'idea del margine al caso non supervisionato: mappati i dati nello
spazio delle feature con un kernel (di solito RBF), cerca l'iperpiano che
separa i punti dall’**origine** con il massimo margine. Ricondotto allo spazio
originale, questo equivale a racchiudere i dati normali in una regione
compatta; ciò che cade fuori è novità/anomalia. Il parametro $\nu \in (0,1]$
ha un doppio significato preciso: è un limite *superiore* alla frazione di
esempi di addestramento classificati come anomali (i *margin error*) e un
limite *inferiore* alla frazione di vettori di supporto. La distingue dalla
classificazione binaria un'assenza: in addestramento **non** c'è la classe
«anomalo»: si impara solo la forma del normale. Un parente stretto è la
**Support Vector Data Description** (SVDD) di Tax e Duin, che invece della
separazione dall'origine cerca la *ipersfera* minima che racchiude i dati; e
tra le alternative non-kernel ci sono l’**Isolation Forest** (che isola le
anomalie con partizioni casuali, ereditando la scalabilità degli alberi della
sezione sugli ensemble) e il *Local Outlier Factor* basato sulla densità
locale.

`````

## In pratica, con scikit-learn

In scikit-learn la famiglia SVM vive nel modulo `sklearn.svm`: `SVC` per la
classificazione con kernel, `LinearSVC` per la versione lineare veloce, `SVR`
per la regressione, `OneClassSVM` per la novelty detection. Due avvertenze
valgono per tutte, e non sono opzionali.

**Standardizzare sempre le feature**, cioè riportare tutte le colonne alla
stessa scala prima di dare i dati al modello: si sottrae a ogni colonna la sua
media e la si divide per la sua ampiezza tipica, così che i metri quadri (che
valgono decine) e il numero di stanze (che vale unità) contino allo stesso
modo. La SVM misura distanze, e senza questa operazione una colonna con numeri
grandi domina il conto e schiaccia le altre, esattamente come
succede al k-NN. Si antepone quindi sempre uno `StandardScaler` in una `Pipeline`, come si è
fatto per il k-NN e per Ridge.

**Attenzione ai numeri grandi.** Il costo di addestramento cresce assai più in
fretta del numero di esempi: raddoppiando gli esempi il lavoro non raddoppia,
si moltiplica per quattro o per otto. Nella notazione con cui si scrivono
queste crescite (si legge «ordine di») è circa fra $O(m^2)$ e $O(m^3)$ nel
numero di esempi $m$: ottima da poche
centinaia a qualche decina di migliaia di punti, diventa proibitiva su milioni.
Per i dataset molto grandi si ripiega su modelli lineari (`LinearSVC`,
`SGDClassifier`, che scalano circa come $O(m)$) o sugli alberi in boosting
della sezione sugli ensemble {cite}`geron2022hands`.

```python
import numpy as np
from sklearn.datasets import make_moons
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC, SVR, OneClassSVM

X, y = make_moons(n_samples=200, noise=0.20, random_state=0)

# Classificazione con kernel RBF: standardizzare SEMPRE (la SVM e' sensibile alla scala)
clf = make_pipeline(StandardScaler(),
                    SVC(kernel="rbf", C=1.0, gamma="scale"))
clf.fit(X, y)

# Variante lineare, veloce su molti esempi: niente kernel trick, costo ~O(m)
lin = make_pipeline(StandardScaler(), LinearSVC(C=1.0))
lin.fit(X, y)

# Regressione: qui serve un target CONTINUO, non le classi 0/1 di sopra.
# Fabbrichiamone uno: una sinusoide della prima coordinata, con un po’ di rumore.
rng = np.random.default_rng(0)
y_reg = np.sin(3 * X[:, 0]) + rng.normal(0, 0.1, size=len(X))

# il tubo epsilon-insensitive ignora gli scarti piccoli
reg = make_pipeline(StandardScaler(),
                    SVR(kernel="rbf", C=10.0, epsilon=0.1))
reg.fit(X, y_reg)

# One-class SVM: impara la regione dei dati "normali";
# nu ~ frazione di anomalie attese
normali = X[y == 0]                      # fingiamo di avere solo la classe "normale"
det = make_pipeline(StandardScaler(),
                    OneClassSVM(kernel="rbf", nu=0.05, gamma="scale"))
det.fit(normali)
esito = det.predict(X)                   # +1 = normale, -1 = anomalia
print("anomalie segnalate:", int(np.sum(esito == -1)))
```

```text
anomalie segnalate: 104
```

La solita grammatica `fit`/`predict` regge anche qui. Per la SVM con kernel la
coppia di iperparametri da tarare per validazione è $(C, \gamma)$: una ricerca
su griglia con la cross-validation della sezione sull'overfitting è la prassi.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Il **kernel trick** rende curva la frontiera: gli stessi punti si guardano in
  uno spazio con una dimensione in più, e lì tornano separabili da un taglio
  dritto. È il bersaglio sollevato in aria, ogni punto tanto più in alto quanto
  più è lontano dal centro, finché una lastra di vetro orizzontale divide il
  centro dall'anello: i punti non sono cambiati, è cambiato il posto da cui li
  guardiamo. Il modo più usato di misurare quanto due punti si somigliano è
  quello «a lampione», dove ogni punto illumina i vicini: luce corta, frontiera
  frastagliata; luce lunga, frontiera morbida.
- La stessa idea serve anche a **prevedere numeri** (un tubo di tolleranza
  attorno alla curva: finché il punto ci sta dentro, l'errore conta zero) e a
  **riconoscere le anomalie** (un recinto attorno ai dati normali, e chi cade
  fuori è sospetto, senza aver mai visto una frode).
- In pratica: portare **sempre** tutte le caratteristiche alla stessa scala,
  perché la SVM ragiona per distanze; e ricordare che il conto cresce assai più
  in fretta del numero di esempi, tanto che oltre le decine di migliaia la SVM
  con kernel diventa impraticabile.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Il **kernel trick** rende non lineare la SVM: mappa i dati in uno spazio più
  ampio dove diventano separabili, calcolando i prodotti scalari con un
  **kernel** $k(\mathbf{x},\mathbf{z})=\phi(\mathbf{x})^\top\phi(\mathbf{z})$
  senza costruirlo: funziona perché nel **duale** gli esempi compaiono solo
  dentro prodotti scalari, e vale se e solo se $k$ è simmetrica e semidefinita
  positiva (Mercer). Kernel principali: lineare, polinomiale, RBF, dove
  $\gamma$ **stringe** la campana al crescere ($\gamma = 1/(2\sigma^2)$).
- La **SVR** regredisce con un tubo $\epsilon$-insensitive; la **one-class SVM**
  ($\nu$ = frazione di anomalie attese) impara la regione dei dati normali per
  la **novelty/anomaly detection**, senza vedere esempi anomali.
- In pratica: **standardizzare sempre** le feature; il costo $O(m^2)$–$O(m^3)$
  sconsiglia la SVM con kernel oltre le decine di migliaia di esempi.
```

`````
