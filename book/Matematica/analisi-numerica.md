# Analisi numerica: quando i numeri hanno precisione finita

Apri un terminale Python e prova la cosa più innocente del mondo:

```{code-block} python
:class: pt-non-eseguibile

>>> 0.1 + 0.2
0.30000000000000004
```

Non è un difetto di Python: è così su qualunque calcolatore. Un computer non
conserva i numeri reali, ma loro approssimazioni con un numero **finito** di
cifre. Di solito la differenza è invisibile; a volte no. Il 4 giugno 1996 il
razzo europeo Ariane 5 si autodistrusse 37 secondi dopo il lancio perché un
numero troppo grande, convertito in un formato più piccolo, "straripò": un
errore di rappresentazione da centinaia di milioni di dollari. L'analisi numerica
studia questi limiti e insegna a conviverci. Nel machine learning non è un
dettaglio accademico: è la differenza tra un addestramento che converge e uno
che produce `NaN`.

## La virgola mobile: un budget fisso di cifre

```{figure} ../figures/float16-precisione-stabilita-numerica.svg
:name: fig-formati-virgola-mobile
:alt: "Tre barre che mostrano come si spartiscono i bit in tre formati. Il float32 ha un bit di segno, 8 bit di esponente (la portata) e 23 bit di mantissa (la precisione). Il float16 ha 5 bit di esponente e 10 di mantissa, con portata ridotta a un massimo di circa 65.504. Il bfloat16 ha 8 bit di esponente, la stessa portata del float32, e solo 7 bit di mantissa, cioè una grana più grossa."
:width: 96%

Lo stesso budget di sedici bit, speso in due modi. Il float16 tiene la
precisione e perde la portata; il bfloat16 fa il contrario, e per addestrare
una rete è quasi sempre il baratto giusto.
```

La distinzione di {numref}`fig-formati-virgola-mobile` fra **portata** e
**precisione** attraversa tutta questa sezione: l'esponente dice fin dove si
può arrivare, la mantissa con quanta finezza. Sono due budget separati, e i
guai che vedremo (overflow, cancellazione) nascono dall'esaurirsi ora
dell'uno, ora dell'altro.

`````{tab} Elementare

Pensa al display di una calcolatrice tascabile: può mostrare solo una decina
di cifre. Se le chiedi $1/3$ ti risponde $0{,}3333333$ e si ferma: le altre
cifre le butta via. I computer fanno lo stesso, in binario, con un **budget**
fisso di cifre per ogni numero.

Con questo budget si scrive un numero come nella notazione scientifica ("tante
cifre significative, moltiplicate per una potenza"), così lo stesso formato
copre sia $0{,}0000001$ sia $10^{30}$. Il prezzo è che tra due numeri vicini
resta sempre un piccolo "gradino" vuoto: $0{,}1 + 0{,}2$ cade tra due gradini
e viene arrotondato, ed ecco lo `0.30000000000000004`.

`````

`````{tab} Superiore

Lo standard **IEEE 754** rappresenta un numero come

$$
x = \pm\, (1 + f)\cdot 2^{e},
$$

dove $f$ è la **mantissa** (le cifre significative, con $0 \le f < 1$) ed $e$
l'**esponente** (la scala). Il formato `float32` spende 1 bit di segno, 8 di
esponente e 23 di mantissa; il `float64` (doppia precisione) ne dà 52 alla
mantissa. La granularità relativa è l'**epsilon macchina** $\varepsilon$: la
distanza fra $1$ e il numero rappresentabile immediatamente successivo, pari a
$2^{-23}\approx 1{,}19\cdot10^{-7}$ per `float32` e
$2^{-52}\approx 2{,}22\cdot10^{-16}$ per `float64`. Ogni operazione arrotonda
al numero rappresentabile più vicino, e l'errore relativo che ne deriva è
limitato dall'**unità di arrotondamento** $u=\varepsilon/2$ (metà del gradino,
perché si arrotonda all'estremo più vicino). Le reti neurali si addestrano
spesso in precisione ridotta (`float32` o perfino `float16`) per risparmiare
memoria e tempo: più veloci, ma con meno cifre di margine.

`````

```{admonition} Quando un bit si gira da solo
:class: seealso

Il primo bit di quella rappresentazione è il **segno**. Vale la pena fermarsi
un attimo su che cosa significa, perché è un caso in cui la struttura del
formato ha una conseguenza che di solito non si associa alla matematica.

I bit in memoria non sono eterni. Una particella ionizzante, un disturbo
elettromagnetico, una cella difettosa: ogni tanto un bit si ribalta senza che
nessuno lo abbia chiesto. Su un computer da scrivania è un evento raro; su
decine di migliaia di acceleratori che macinano per settimane, diventa
un'occorrenza ordinaria, e i grandi operatori la trattano come tale.

Se il bit che si gira è l'ultimo della mantissa, l'effetto è invisibile: il
numero cambia nella settima cifra. Se è il **primo**, un peso che valeva
$0{,}5$ diventa $-0{,}5$, e il segno di un contributo si rovescia in mezzo
alla rete. Su una rete profonda l'effetto non resta locale, perché quel valore
alimenta lo strato successivo: un singolo bit ribaltato in un punto sensibile
può far crollare l'accuratezza di un classificatore su ImageNet (il $76\%$ di
una ResNet50) fino ai livelli del tiro a caso {cite}`hong2019terminal`.

La differenza rispetto al software tradizionale è che qui **non si vede**. Un
bit sbagliato in un programma normale di solito produce un crash o un risultato
palesemente assurdo; in una rete produce una risposta plausibile e sbagliata,
indistinguibile da una risposta giusta se non si conosce quella giusta. È il
motivo per cui il tema ha un nome tutto suo, *corruzione silenziosa dei dati*,
e per cui la robustezza di un sistema di ML ha due facce distinte: quella ai
**dati** (deriva, esempi avversari, discussa nel capitolo sull'AI
responsabile) e quella all'**hardware**, che non riguarda il modello ma il
silicio su cui gira.
```

## Overflow e underflow: i bordi del mondo rappresentabile

Il budget di cifre ha due confini. Oltre il più grande numero rappresentabile
si va in **overflow** (il risultato diventa $\pm\infty$); sotto il più piccolo
si va in **underflow** (il risultato collassa a $0$).

`````{tab} Elementare

È come un contachilometri con un numero fisso di caselle: superato il massimo,
il valore "sballa". Un `float32` arriva a circa $3{,}4\cdot10^{38}$: sembra
enorme, ma basta calcolare $e^{89}$ per sfondarlo. All'estremo opposto,
$e^{-120}$ è così vicino a zero che un `float32` lo registra proprio come $0$.
Il guaio arriva subito dopo: se poi *dividi* per quel numero diventato zero, o
ne fai il logaritmo, ottieni infinito o `NaN`, e l'addestramento si rompe.

`````

`````{tab} Superiore

Per `float32` l'estremo superiore è $\approx 3{,}40\cdot10^{38}$ e il più
piccolo positivo normalizzato è $\approx 1{,}18\cdot10^{-38}$. Poiché
$\exp$ compare ovunque (softmax, verosimiglianze gaussiane, funzioni di
partizione), è la sorgente tipica di overflow: $e^{z}$ supera il limite già per
$z \gtrsim 88{,}7$. L'underflow è insidioso perché *silenzioso*: un prodotto di
molte probabilità, $\prod_i p_i$ con $p_i < 1$, tende esponenzialmente a zero e
sparisce senza segnalazioni. La difesa standard è lavorare nel dominio
logaritmico, dove i prodotti diventano somme.

`````

## Il trucco log-sum-exp

Nessuna funzione soffre di questi problemi quanto la **softmax**, che chiude
quasi ogni classificatore trasformando dei punteggi grezzi $z_i$ (i *logit*) in
probabilità. La sua definizione contiene esponenziali, e gli esponenziali
straripano. La cura è un'idea semplice ed elegante.

`````{tab} Elementare

La softmax risponde alla domanda "che quota di probabilità spetta a ciascuna
classe?". Se i punteggi sono $z=(1000,\ 1001,\ 1002)$, gli $e^{z_i}$ vanno
tutti in overflow e il conto fallisce. Ma la softmax guarda solo le
*differenze* tra i punteggi: se sottrai lo stesso numero a tutti (per esempio
il massimo, $1002$) il risultato non cambia. I punteggi diventano
$(-2,\ -1,\ 0)$, gli esponenziali restano piccoli e comodi, e la probabilità è
identica. Sottrarre il massimo prima di esponenziare: tutto qui.

`````

`````{tab} Superiore

La softmax è

$$
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}} .
$$

Sia $m = \max_j z_j$. Moltiplicando numeratore e denominatore per $e^{-m}$ il
valore non cambia, ma ogni esponente diventa $\le 0$:

$$
\text{softmax}(z)_i = \frac{e^{z_i - m}}{\sum_{j} e^{z_j - m}} .
$$

La stessa mossa stabilizza il logaritmo della somma di esponenziali, l'identità
**log-sum-exp**:

$$
\log \sum_j e^{z_j} = m + \log \sum_j e^{z_j - m} .
$$

Da qui la log-probabilità della classe corretta,
$\log \hat{p}_i = z_i - \operatorname{logsumexp}(z)$, si calcola senza mai
formare $e^{z_i}$ crudo; la **cross-entropy** è semplicemente il suo opposto,
$\operatorname{logsumexp}(z) - z_i$. È il motivo per cui i framework espongono
`log_softmax` e loss che lavorano direttamente sui logit (come la
`nn.CrossEntropyLoss` di PyTorch, che incontreremo nel capitolo dedicato): non
è pigrizia d'API, è stabilità numerica.

`````

## Arrotondamento e cancellazione

Ogni operazione arrotonda, e di solito gli errori sono trascurabili. C'è però
un caso in cui esplodono: la **cancellazione**, cioè la sottrazione di due
numeri quasi uguali.

`````{tab} Elementare

Immagina di misurare il peso di un capitano *con la sua barca* ($80\,000$ kg)
e della sola barca ($79\,930$ kg), ciascuno accurato al chilo. La differenza
(il peso del capitano) è $70$ kg, ma l'incertezza di un chilo su ciascuna
misura ora pesa tantissimo *in proporzione*. Le cifre affidabili si sono
"cancellate" e resta soprattutto rumore. In pratica: evita di calcolare una
quantità piccola come differenza di due quantità grandi.

`````

`````{tab} Superiore

Il caso da manuale è la varianza con la formula "ingenua"
$\operatorname{Var}(x)=\overline{x^2}-\bar{x}^2$: con dati grandi e varianza
piccola i due termini sono quasi uguali e la sottrazione perde quasi tutte le
cifre significative (può perfino dare un valore negativo). Le librerie evitano
la formula ingenua: NumPy calcola prima la media e poi la media degli scarti
quadratici, in due passate; quando i dati arrivano in flusso e di passata se ne
può fare una sola, si usa l'algoritmo di **Welford**, numericamente stabile.
Regola generale: riformula le espressioni per non sottrarre grandezze vicine;
la stessa quantità matematica può avere condizionamenti numerici molto diversi
a seconda di *come* la si calcola.

`````

## Condizionamento: quanto un problema amplifica gli errori

Abbiamo incontrato la parola "condizionamento": vale la pena isolarne
l'intuizione. Un problema è **ben condizionato** se piccole variazioni
dell'input producono piccole variazioni dell'output; è **mal condizionato** se
le amplifica a dismisura. È una proprietà del problema, non dell'algoritmo: su
un problema mal condizionato anche il codice perfetto fatica, perché eredita
l'errore di arrotondamento già presente negli input. Per un sistema lineare
$A x = b$ lo si misura con il **numero di condizionamento** di $A$, il
rapporto tra la massima e la minima "amplificazione" che la matrice può
imprimere a un vettore (formalmente, tra il suo valore singolare più grande e
quello più piccolo): se è enorme, la soluzione è ipersensibile e poco
affidabile.

## Perché normalizzare i dati aiuta

Ed è qui che i conti si ricongiungono alla pratica quotidiana. Prima di dare i
dati a un modello quasi sempre li **standardizziamo**, sottraendo la media e
dividendo per la deviazione standard, $z = (x - \mu)/\sigma$, così ogni
caratteristica (*feature*) ha media $0$ e scala $1$. Non è solo cosmesi:
serve la stabilità.

Se una feature vale in migliaia di euro e un'altra in numero di stanze, i loro
prodotti dentro la rete stanno su scale lontanissime (invito all'overflow) e
la superficie della *loss* si allunga in una valle stretta, mal condizionata.
La discesa del gradiente vi rimbalza da una parete all'altra a zig-zag,
convergendo con lentezza esasperante. Standardizzare rende le curve di livello
molto più tonde: il gradiente punta quasi dritto verso il minimo
({numref}`fig-condizionamento`). Non è una cura completa, perché mette tutte le
feature sulla stessa scala ma non cambia il modo in cui si somigliano fra loro:
se due di esse crescono e calano quasi sempre insieme, la valle resta un po'
storta e qualche zig-zag la discesa lo fa ancora. Resta il rimedio più
economico che ci sia: due righe di codice, e il problema è molto meglio
condizionato di prima.

```{figure} ../figures/condizionamento-normalizzazione.svg
:name: fig-condizionamento
:alt: A sinistra curve di livello molto allungate, con il cammino del gradiente che rimbalza a zig-zag fra le pareti della valle; a destra le stesse curve diventate quasi tonde, con un cammino quasi dritto verso il minimo.
:width: 92%

Con dati grezzi (sinistra) la loss forma una valle stretta e il gradiente
rimbalza a zig-zag; standardizzando (destra) le curve di livello diventano
molto più tonde e la discesa punta quasi dritta al minimo.
```

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_std = scaler.fit_transform(X)   # ogni colonna: media 0, deviazione std 1
```

```{admonition} Da ricordare
:class: important
- I numeri in **virgola mobile** hanno precisione finita: `0.1 + 0.2` non fa
  esattamente `0.3`, ed esiste un limite oltre il quale si va in **overflow**
  o **underflow**.
- La softmax e le verosimiglianze si calcolano nel dominio logaritmico con il
  trucco **log-sum-exp** (sottrai il massimo) per evitare che gli esponenziali
  straripino.
- **Standardizzare** i dati non è solo buona educazione statistica: riduce il
  condizionamento del problema e fa convergere l'ottimizzazione molto più in
  fretta.
```
