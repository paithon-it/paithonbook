# Equità e bias algoritmico

Nel 2018 un'inchiesta di Reuters rivela che Amazon aveva accantonato in
silenzio, un anno prima, uno strumento sperimentale di selezione del
personale. L'idea era seducente: dare in pasto a un modello i curriculum degli
ultimi dieci anni e lasciargli imparare a riconoscere i candidati «bravi»,
quelli che in passato erano stati assunti. Il modello imparò benissimo: troppo
bene. Poiché quei dieci anni di assunzioni erano stati dominati da uomini, il
sistema dedusse che *essere uomo* fosse un buon segnale: penalizzava i
curriculum che contenevano la parola «women's» (come in «women's chess club
captain») e declassava chi aveva studiato in due college femminili. Nessuno
aveva scritto una regola contro le donne. La regola era stata *appresa*, letta
nel passato dell'azienda e riproposta come profezia.

È il tema di questa sezione. Un modello non inventa il pregiudizio: lo
eredita. E per governarlo servono due cose che affronteremo in ordine: prima
capire *da dove* entra il bias, poi imparare a *misurarlo* con precisione,
riusando la matrice di confusione già vista nel capitolo di Machine Learning
ma applicandola gruppo per gruppo. Alla fine ci imbatteremo in una sorpresa
scomoda: alcune richieste di equità, per quanto ragionevoli, non possono
valere tutte insieme.

## Da dove entra il bias

Il pregiudizio algoritmico non nasce dal codice, che è cieco e indifferente:
nasce a monte, nei dati e nelle scelte con cui li abbiamo raccolti ed
etichettati.

```{figure} ../figures/bias-nei-modelli.svg
:name: fig-ciclo-del-bias
:alt: "Catena che si chiude ad anello: i dati storici, che portano già le disparità del passato, entrano nel modello, che le amplifica; il modello produce decisioni che riproducono la disparità nel mondo; e quelle decisioni diventano i dati storici del giro successivo."
:width: 92%

Il bias non attraversa il modello e si ferma: torna indietro. Le decisioni di
oggi diventano i dati di domani, e il giro seguente parte da una disparità un
po' più marcata.
```

La freccia di ritorno in {numref}`fig-ciclo-del-bias` è la ragione per cui il
problema non si risolve una volta sola. Un sistema che seleziona candidati
genera i dati sulle assunzioni future: se ha escluso un gruppo, il prossimo
addestramento troverà davvero meno esempi di successo in quel gruppo, e avrà
ragione a diffidarne. Il pregiudizio si fabbrica le proprie prove. La letteratura (per esempio la rassegna di Mehrabi e colleghi
{cite}`mehrabi2021survey`) distingue alcune sorgenti ricorrenti.

`````{tab} Elementare

Immagina un apprendista che impara il mestiere osservando *solo* le decisioni
prese in passato dai suoi capi, senza mai chiedersi se fossero giuste. Erediterà
la loro bravura, ma anche le loro storture. Con i dati succede lo stesso, e le
storture arrivano da quattro porte.

- **Il passato è ingiusto.** Se per anni i prestiti sono andati soprattutto
  agli abitanti di certi quartieri, un modello addestrato su quello storico
  imparerà a dire di sì agli stessi e di no agli altri: non perché siano meno
  affidabili, ma perché *storicamente* hanno avuto meno occasioni.
- **Il campione non rappresenta tutti.** Se le foto per allenare un
  riconoscitore di volti ritraggono in gran parte persone dalla pelle chiara, il
  sistema funzionerà peggio su tutti gli altri: non li ha quasi mai visti.
- **Le etichette sono distorte.** Spesso la «risposta giusta» che diamo in
  pasto al modello non è la verità, ma una sua approssimazione imperfetta: «è
  stato arrestato» al posto di «ha commesso un reato», e l'arresto dipende
  anche da *dove* e *chi* la polizia controlla di più.
- **Il modello si morde la coda.** Se un sistema manda più pattuglie in un
  quartiere, lì si registreranno più reati, il che convince il sistema a
  mandarcene ancora di più. Il pregiudizio si auto-conferma.

Il riassunto sta in un adagio: *bias in, bias out*. Un modello impeccabile
allenato su dati storti produce risultati storti.

`````

`````{tab} Superiore

Conviene distinguere le sorgenti, perché richiedono rimedi diversi
{cite}`mehrabi2021survey`.

- **Bias storico.** I dati riflettono fedelmente un mondo già iniquo. Anche con
  campionamento perfetto ed etichette perfette, la regolarità che il modello
  apprende *è* la disuguaglianza. Nessuna quantità di dati aggiuntivi la corregge,
  perché non è un errore di misura: è il fenomeno stesso.
- **Bias di rappresentazione (campionamento).** La distribuzione dei dati di
  addestramento $P_{\text{train}}$ differisce dalla popolazione bersaglio
  $P_{\text{test}}$, e in particolare sotto-rappresenta alcuni gruppi. È il caso
  di *Gender Shades* citato nell'apertura del capitolo: pochi volti scuri nei
  set di addestramento, quindi errore molto più alto su quel gruppo.
- **Bias di misura (etichette).** L'etichetta osservata è un *proxy* del
  costrutto d'interesse: si misura «arresto» per «reato», «voto del manager» per
  «rendimento». Se il proxy è più rumoroso o più severo per un gruppo, il bias
  entra dalle etichette prima ancora del modello.
- **Bias di feedback (loop).** Le decisioni del modello alterano i dati futuri
  su cui il modello successivo verrà addestrato. La *polizia predittiva* è
  l'esempio da manuale: più controlli dove il modello prevede reati $\Rightarrow$
  più reati *registrati* lì $\Rightarrow$ previsioni ancora più concentrate. Il
  segnale si auto-rinforza indipendentemente dal tasso reale.

La distinzione operativa è netta: campionamento e feedback si possono attaccare
raccogliendo o correggendo i dati; il bias storico e quello di misura no,
perché il difetto è nella definizione stessa dell'obiettivo.

`````

## Misurare l'equità: le definizioni di gruppo

Per parlare di equità con rigore serve un vocabolario. Fissiamo la notazione:
$A$ è l'**attributo protetto** che identifica il gruppo (per esempio $A=a$ e
$A=b$), $Y \in \{0,1\}$ è l'**esito reale**, $\hat{Y}$ è la **decisione** del
modello e $S \in [0,1]$ il **punteggio** da cui la decisione si ottiene
fissando una soglia. Le metriche che seguono sono le stesse del capitolo di
Machine Learning, tasso di veri positivi (TPR) e di falsi positivi (FPR) letti
dalla matrice di confusione, con una differenza sola ma decisiva: si calcolano
**separatamente per ciascun gruppo** e poi si confrontano
({numref}`fig-equita-tassi`).

```{figure} ../figures/equita-tassi.svg
:name: fig-equita-tassi
:alt: Due matrici di confusione due per due affiancate, etichettate Gruppo A e Gruppo B, con le celle VP, FP, FN, VN riempite di numeri esempio; sotto ciascuna matrice il tasso di veri positivi (TPR) e il tasso di falsi positivi (FPR), con valori diversi fra i due gruppi.
:width: 100%

Lo stesso modello valutato su due gruppi. Dalle matrici di confusione si
leggono tassi diversi, Gruppo A: $\text{TPR}=0{,}80$, $\text{FPR}=0{,}10$;
Gruppo B: $\text{TPR}=0{,}60$, $\text{FPR}=0{,}30$. Quando TPR e FPR divergono
fra i gruppi, l'equalized odds è violata.
```

`````{tab} Elementare

Ci sono tre modi diversi di chiedere «il modello è equo?», e portano a tre
richieste distinte.

- **Stessa quota di sì (parità demografica).** Il modello dice «approvato»
  alla stessa percentuale di persone in ogni gruppo. Se approva il 40% degli
  uomini, deve approvare il 40% delle donne: a prescindere da tutto il resto.
- **Stessa affidabilità sugli errori (equalized odds).** Fra chi *meritava
  davvero* un sì, la quota di sì è uguale nei due gruppi; e fra chi *meritava un
  no*, la quota di sì sbagliati è uguale. In altre parole: il modello sbaglia
  allo stesso modo su tutti. La {numref}`fig-equita-tassi` mostra il caso in cui
  questa richiesta è **violata**: stesso modello, ma tasso di veri positivi e di
  falsi positivi diversi fra Gruppo A e Gruppo B.
- **Stesso significato del punteggio (calibrazione).** Un punteggio di rischio
  «70» deve voler dire la stessa probabilità reale per tutti: se fra gli uomini
  con punteggio 70 il 70% ricade nell'esito, lo stesso deve valere fra le donne.

Sembrano tre facce della stessa medaglia. Vedremo tra poco che, sorprendentemente,
non possono quasi mai brillare tutte insieme.

`````

`````{tab} Superiore

Le definizioni di equità di gruppo si organizzano attorno a tre criteri
statistici {cite}`hardt2016equality`.

**Parità demografica** (*independence*, $\hat{Y} \perp A$): la quota di esiti
positivi non dipende dal gruppo,

$$
P(\hat{Y}=1 \mid A=a) \;=\; P(\hat{Y}=1 \mid A=b).
$$

È il *selection rate* uguale fra i gruppi. Limite noto: ignora del tutto $Y$,
quindi è compatibile con l'assurdo di selezionare i candidati *giusti* in un
gruppo e a *caso* nell'altro.

**Equalized odds** (*separation*, $\hat{Y} \perp A \mid Y$), introdotta da Hardt,
Price e Srebro {cite}`hardt2016equality`: a parità di esito reale la predizione
non dipende dal gruppo,

$$
P(\hat{Y}=1 \mid Y=y,\, A=a) \;=\; P(\hat{Y}=1 \mid Y=y,\, A=b), \qquad y\in\{0,1\}.
$$

Per $y=1$ questa è l'uguaglianza dei **TPR**, per $y=0$ quella dei **FPR**: il
modello deve avere lo stesso tasso di veri positivi *e* lo stesso tasso di falsi
positivi in ogni gruppo. La {numref}`fig-equita-tassi` illustra una violazione:
$\text{TPR}_a=0{,}80 \neq \text{TPR}_b=0{,}60$ e
$\text{FPR}_a=0{,}10 \neq \text{FPR}_b=0{,}30$. La versione più debole
**equal opportunity** impone la sola uguaglianza dei TPR (solo su $y=1$),
appropriata quando il costo asimmetrico ricade su chi *meritava* l'esito
positivo e viene mancato.

**Calibrazione per gruppo** (*sufficiency*, $Y \perp A \mid S$): a parità di
punteggio la probabilità reale dell'esito è la stessa,

$$
P(Y=1 \mid S=s,\, A=a) \;=\; P(Y=1 \mid S=s,\, A=b) \qquad \forall\, s.
$$

Qui $s$ è il valore del punteggio; la condizione dice che uno stesso $s$
«significa» la stessa cosa in ogni gruppo. È il criterio che l'azienda produttrice
di COMPAS invocava a propria difesa.

`````

## Il risultato di impossibilità

Arriviamo al nodo. Le tre richieste appena viste non sono capricci in
conflitto per caso: sono **matematicamente incompatibili** ogni volta che i
gruppi partono da tassi di base diversi. Lo hanno dimostrato, in modo
indipendente e quasi simultaneo, Alexandra Chouldechova
{cite}`chouldechova2017fair` e (con un teorema gemello sui punteggi di
rischio) Jon Kleinberg, Sendhil Mullainathan e Manish Raghavan (2016).

`````{tab} Elementare

Immagina due gruppi in cui l'esito che vogliamo prevedere è, nella realtà, più
frequente in uno che nell'altro: non per colpa di nessuno, semplicemente perché
i *tassi di base* differiscono. Ora pretendi due cose ragionevoli insieme. Uno:
che un punteggio «alto» significhi lo stesso rischio reale per tutti
(calibrazione). Due: che il modello generi la stessa quota di falsi allarmi in
ogni gruppo (parte dell'equalized odds).

Il teorema dice: non puoi. Se i tassi di base sono diversi, garantire la prima
costringe la seconda a saltare, e viceversa. Non è un bug da correggere con
codice migliore o più dati: è un vincolo dell'aritmetica, come chiedere a un
rettangolo di avere area 12 e perimetro 10 con lati interi (semplicemente non
esiste). È il cuore della disputa su COMPAS: l'inchiesta di ProPublica
accusava il sistema di generare molti più falsi allarmi fra gli imputati neri;
l'azienda rispondeva che il punteggio era calibrato allo stesso modo per
tutti. Avevano ragione **entrambe**, ed è proprio questo il punto.

`````

`````{tab} Superiore

La chiave è un'identità algebrica esatta che lega, all'interno di un gruppo,
quattro grandezze: la prevalenza $p = P(Y=1)$, il valore predittivo positivo
$\text{VPP}$ (la *precision* del capitolo di Machine Learning), il tasso di falsi
negativi $\text{FNR}=1-\text{TPR}$ e il tasso di falsi positivi
{cite}`chouldechova2017fair`:

$$
\text{FPR} \;=\; \frac{p}{1-p}\cdot\frac{1-\text{VPP}}{\text{VPP}}\cdot\bigl(1-\text{FNR}\bigr).
$$

Qui $p$ è la frazione reale di positivi nel gruppo, $\text{VPP}=P(Y=1\mid\hat{Y}=1)$
è la probabilità che un positivo predetto sia davvero positivo, e $\text{FNR}$ e
$\text{FPR}$ sono i due tassi di errore. L'identità si ricava dalla sola
definizione di $\text{VPP}$ e vale sempre. La sua conseguenza è drastica:
**fissati $\text{VPP}$ e $\text{FNR}$ uguali fra due gruppi, se le prevalenze
$p_a \neq p_b$ differiscono, allora i $\text{FPR}$ sono per forza diversi.**

Un esempio numerico lo rende palpabile. Siano due gruppi con prevalenze
$p_a=0{,}50$ e $p_b=0{,}25$, e supponiamo un modello con lo *stesso* valore
predittivo $\text{VPP}=0{,}70$ e la *stessa* quota di positivi mancati
$\text{FNR}=0{,}30$ (dunque anche $\text{TPR}=0{,}70$: perfino l'equal
opportunity è rispettata). Applicando l'identità:

$$
\text{FPR}_a = \frac{0{,}50}{0{,}50}\cdot\frac{0{,}30}{0{,}70}\cdot 0{,}70 = 0{,}30,
\qquad
\text{FPR}_b = \frac{0{,}25}{0{,}75}\cdot\frac{0{,}30}{0{,}70}\cdot 0{,}70 = 0{,}10.
$$

Stesso valore predittivo, stesso tasso di veri positivi, eppure il tasso di
falsi positivi è tre volte più alto nel gruppo con prevalenza maggiore:
$0{,}30$ contro $0{,}10$. È esattamente la forma del caso COMPAS
{cite}`angwin2016machine`. Kleinberg, Mullainathan e Raghavan (2016) provano
la versione per i punteggi continui: calibrazione, bilanciamento della classe
positiva e bilanciamento della classe negativa coesistono solo nei casi
degeneri (prevalenze identiche o predizione perfetta).

`````

## Attenuare il bias: tre punti di intervento

Se una cura definitiva non esiste, restano comunque leve per ridurre il divario.
Si classificano per il *punto* della pipeline in cui agiscono.

`````{tab} Elementare

Pensa a una gara di corsa in cui un gruppo parte più indietro. Puoi intervenire
in tre momenti. **Prima** della gara, riequilibrando la linea di partenza:
correggi i dati, dando più peso agli esempi dei gruppi sotto-rappresentati o
riequilibrando le proporzioni. **Durante** la gara, cambiando le regole:
addestri il modello con un vincolo che lo obbliga a tenere i tassi vicini fra i
gruppi, come un giudice che penalizza chi taglia la strada. **Dopo** la gara,
correggendo il tempo finale: lasci il modello com'è ma usi soglie diverse per
gruppo, in modo che il tasso di errore finale coincida.

Nessuno dei tre è gratis: riequilibrare i dati può abbassare l'accuratezza
complessiva, e usare soglie diverse per gruppo è a sua volta una scelta delicata,
che qualcuno considera essa stessa una forma di disparità di trattamento.

`````

`````{tab} Superiore

- **Pre-processing.** Si trasforma il dataset prima dell'addestramento:
  *reweighting* (pesi $w_i$ per esempio, calcolati così da rendere $\hat{Y}$
  indipendente da $A$ nel campione pesato), ricampionamento dei gruppi
  sotto-rappresentati, o rimozione/decorrelazione delle feature che fungono da
  *proxy* dell'attributo protetto. Vantaggio: agnostico al modello a valle.
- **In-processing.** Si modifica l'obiettivo di addestramento aggiungendo un
  **vincolo** o un termine di **regolarizzazione** di equità, per esempio
  minimizzare $\mathcal{L}_{\text{pred}} + \lambda\,\mathcal{L}_{\text{fair}}$
  dove $\mathcal{L}_{\text{fair}}$ penalizza il divario di TPR/FPR fra i gruppi
  e $\lambda$ regola il compromesso equità–accuratezza. Adversarial debiasing e
  ottimizzazione vincolata rientrano qui.
- **Post-processing.** Si lascia intatto il modello e si aggiustano le
  **soglie**: Hardt, Price e Srebro {cite}`hardt2016equality` mostrano come
  derivare soglie per-gruppo (eventualmente randomizzate) che raggiungono
  l'equalized odds a partire da un qualsiasi punteggio già addestrato (una
  costruzione geometrica sulle curve ROC dei due gruppi).

Il risultato di impossibilità della sezione precedente resta sullo sfondo:
nessuna di queste tecniche annulla il conflitto fra calibrazione ed equalized
odds quando i tassi di base differiscono. Sposta soltanto *quale* criterio
privilegiare, e quel «quale» non è una scelta tecnica.

`````

## Equità individuale

Le definizioni viste finora guardano ai gruppi in media. Una famiglia
alternativa sposta l'obiettivo sul singolo, ed è la *fairness through awareness*
di Cynthia Dwork e colleghi {cite}`dwork2012fairness`.

`````{tab} Elementare

L'idea è intuitiva: **due persone simili devono ricevere esiti simili**. Se due
candidati hanno percorso, competenze ed esperienza quasi identici, il modello non
può approvarne uno e bocciare l'altro solo perché appartengono a gruppi diversi.
È un principio di coerenza, non di media: non dice «tratta bene i gruppi», dice
«non fare distinzioni ingiustificate fra individui vicini».

Il problema è tutto in quella parola, *simili*. Simili rispetto a cosa? Due
curriculum possono sembrare vicini per titoli di studio e lontani per
esperienza: chi decide il metro? Definire la somiglianza «giusta» è difficile
quanto il problema di equità di partenza, e spesso nasconde, dentro il metro,
le stesse distorsioni che volevamo eliminare.

`````

`````{tab} Superiore

Formalmente, dato un metro di distanza fra individui $d(x_i, x_j)$ e una
distanza fra distribuzioni di esito $D$, il classificatore (che a ogni individuo
associa una distribuzione sugli esiti) deve essere **Lipschitz**
{cite}`dwork2012fairness`:

$$
D\bigl(M(x_i),\, M(x_j)\bigr) \;\le\; d(x_i, x_j),
$$

dove $M(x)$ è la distribuzione di esito assegnata a $x$. In parole: individui
vicini secondo $d$ ricevono esiti vicini secondo $D$; il modello non può
«strappare» a piacere due punti che il metro dichiara simili. È una garanzia più
forte e più fine dell'equità di gruppo, ma sposta l'intera difficoltà su $d$: la
metrica di somiglianza specifica del compito è assunta *data*, mentre in pratica
sceglierla è precisamente il giudizio di valore che si voleva rendere oggettivo.
Per questo l'equità individuale è teoricamente elegante ma di rado applicabile
tale e quale.

`````

## Il conflitto, coi numeri

Chiudiamo il cerchio con un esperimento riproducibile. Costruiamo un punteggio
di rischio **calibrato per costruzione**, l'etichetta reale è estratta con
probabilità pari al punteggio, $P(Y=1\mid S=s)=s$, identica in entrambi i
gruppi, ma con **tassi di base diversi**, ottenuti da distribuzioni di
punteggio diverse. Poi applichiamo la stessa soglia a tutti e leggiamo i tassi
gruppo per gruppo.

```python
import numpy as np

rng = np.random.default_rng(0)

def genera_gruppo(n, alpha, beta):
    # Il punteggio è calibrato per costruzione: P(Y=1 | S=s) = s
    s = rng.beta(alpha, beta, size=n)          # punteggio in [0,1]
    y = (rng.random(n) < s).astype(int)        # etichetta vera ~ Bernoulli(s)
    return s, y

# Gruppo A: rischio di base più alto; Gruppo B: più basso
sA, yA = genera_gruppo(20000, 3.0, 3.0)        # media score ~0,50
sB, yB = genera_gruppo(20000, 2.0, 4.0)        # media score ~0,33

soglia = 0.5

def tassi(s, y, t):
    yhat = (s >= t).astype(int)
    sel = yhat.mean()                # selection rate: quota di sì
    tpr = yhat[y == 1].mean()        # veri positivi / positivi reali
    fpr = yhat[y == 0].mean()        # falsi positivi / negativi reali
    ppv = y[yhat == 1].mean()        # valore predittivo positivo (precision)
    return sel, tpr, fpr, ppv

for nome, s, y in [("A", sA, yA), ("B", sB, yB)]:
    sel, tpr, fpr, ppv = tassi(s, y, soglia)
    print(f"Gruppo {nome}: base={y.mean():.3f}  selection={sel:.3f}  "
          f"TPR={tpr:.3f}  FPR={fpr:.3f}  VPP={ppv:.3f}")

# Calibrazione per gruppo: in ogni bin di score, frazione reale di positivi
bins = np.linspace(0, 1, 6)
print("\nCalibrazione (bin di score -> frazione reale di positivi):")
for nome, s, y in [("A", sA, yA), ("B", sB, yB)]:
    idx = np.clip(np.digitize(s, bins) - 1, 0, len(bins) - 2)
    riga = [f"[{bins[b]:.1f},{bins[b+1]:.1f})->{y[idx == b].mean():.2f}"
            for b in range(len(bins) - 1)]
    print(f"  Gruppo {nome}:", "  ".join(riga))
```

L'esecuzione stampa qualcosa come:

```text
Gruppo A: base=0.500  selection=0.502  TPR=0.658  FPR=0.346  VPP=0.656
Gruppo B: base=0.329  selection=0.188  TPR=0.348  FPR=0.110  VPP=0.607

Calibrazione (bin di score -> frazione reale di positivi):
  Gruppo A: [0.0,0.2)->0.16  [0.2,0.4)->0.31  [0.4,0.6)->0.50  [0.6,0.8)->0.69  [0.8,1.0)->0.85
  Gruppo B: [0.0,0.2)->0.13  [0.2,0.4)->0.29  [0.4,0.6)->0.48  [0.6,0.8)->0.67  [0.8,1.0)->0.81
```

Le due curve di calibrazione sono **essenzialmente identiche**: in ogni fascia
di punteggio, la frazione reale di positivi è pressoché la stessa fra i gruppi
(e coincide con il punteggio medio della fascia, come impone la calibrazione
per costruzione). La calibrazione, cioè, *vale*. Eppure il tasso di falsi
positivi è tre volte più alto nel Gruppo A ($0{,}346$ contro $0{,}110$) e
anche il TPR diverge nettamente ($0{,}658$ contro $0{,}348$): l'equalized odds
è platealmente violata. Non c'è nessun errore nel codice: è l'impossibilità
della sezione precedente che si materializza in numeri. Cambiare la soglia
sposta i tassi, ma non li può allineare *tutti* insieme finché le prevalenze
restano $0{,}50$ e $0{,}33$.

## Nessuna metrica è «quella giusta»

Se c'è una lezione da portare via, è questa: la domanda «questo modello è equo?»
è mal posta finché non specifichiamo *secondo quale criterio*. Parità
demografica, equalized odds, calibrazione ed equità individuale non sono
approssimazioni successive di un'unica verità nascosta: sono definizioni
**diverse e in tensione**, ciascuna sensata in certi contesti e inaccettabile in
altri. Nello screening di una malattia grave conta non mancare i malati (uguale
TPR); nella concessione di un mutuo conta che un punteggio significhi lo stesso
per tutti (calibrazione).

La statistica fa il suo mestiere fino a un certo punto: delimita lo spazio
delle opzioni, quantifica i compromessi, smaschera le incompatibilità. Ma
*quale* criterio far valere non discende dai dati: è una scelta di valore, che
va posta in chiaro e discussa, non nascosta dentro una funzione obiettivo. Con
lo stesso spirito affronteremo, nelle sezioni successive, la privacy e la
robustezza dei modelli, e più avanti l'interpretabilità come strumento per
rendere queste scelte finalmente ispezionabili.

```{admonition} Da ricordare
:class: important
- Il bias non nasce nel codice ma **a monte**, nei dati: passato iniquo,
  campione non rappresentativo, etichette-proxy distorte e *feedback loop* che
  si auto-conferma. *Bias in, bias out* {cite}`mehrabi2021survey`.
- L'equità di gruppo si misura riusando la **matrice di confusione** del capitolo
  di Machine Learning, ma *separatamente per gruppo*. Tre criteri: **parità
  demografica** (stessa quota di sì), **equalized odds** (stessi TPR e FPR)
  {cite}`hardt2016equality`, **calibrazione** (stesso significato del punteggio).
- **Risultato di impossibilità** {cite}`chouldechova2017fair`: se i tassi di
  base differiscono, calibrazione ed equalized odds non possono valere insieme.
  È il nodo del caso COMPAS {cite}`angwin2016machine`, dove ProPublica e
  l'azienda avevano ragione entrambe.
- Si può **attenuare**, non risolvere, intervenendo in pre-processing
  (riequilibrio dei dati), in-processing (vincoli/regolarizzazione di equità) o
  post-processing (soglie per gruppo).
- L'**equità individuale** {cite}`dwork2012fairness` chiede esiti simili per
  individui simili, ma sposta la difficoltà sul definire «simile».
- Nessuna metrica è «quella giusta»: scegliere il criterio di equità è una
  **decisione di valore**, non un calcolo.
```
