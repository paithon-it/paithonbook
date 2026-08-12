# Equità e bias algoritmico

Nel 2018 un'inchiesta di Reuters rivela che Amazon aveva accantonato in
silenzio, un anno prima, uno strumento sperimentale di selezione del personale
{cite}`dastin2018amazon`. L'idea era seducente: dare in pasto a un modello i
curriculum degli ultimi dieci anni e lasciargli imparare a riconoscere i
candidati «bravi», quelli che in passato erano stati assunti. Il modello imparò
benissimo: troppo bene. Poiché quei dieci anni di assunzioni erano stati
dominati da uomini, il sistema dedusse che *essere uomo* fosse un buon segnale:
penalizzava i curriculum che contenevano la parola «women's», cioè
«femminile», come nella riga «capitana della squadra di scacchi femminile», e
declassava chi aveva studiato in due college per sole donne. Nessuno aveva
scritto una regola contro le donne. La regola era stata *appresa*, letta nel
passato dell'azienda e riproposta come profezia.

Prima di procedere, una parola sul titolo. **Bias**, qui, vuol dire
*pregiudizio*, e non è nessuna delle due cose che il libro ha già chiamato così:
non è il bias del neurone (il termine $b$ che, nel capitolo sulle reti neurali,
si somma ai pesi e sposta la soglia) e non è il bias del compromesso
bias-varianza (l'errore sistematico di un modello troppo semplice, nel capitolo
di Machine Learning). La distinzione conta, perché il secondo è vicino
abbastanza da confondere: un modello con molto bias statistico non è un modello
che discrimina, e un modello che discrimina può avere bias statistico nullo. È
la terza accezione della stessa parola inglese, e in italiano si dice
pregiudizio.

È il tema di questa sezione. Un modello non inventa il pregiudizio: lo
eredita. E per governarlo servono due cose che affronteremo in ordine: prima
capire *da dove* entra il pregiudizio, poi imparare a *misurarlo* con
precisione, riusando la tabella a quattro caselle già vista nel capitolo di
Machine Learning ma compilandola gruppo per gruppo. Alla fine ci imbatteremo in
una sorpresa scomoda: alcune richieste di equità, per quanto ragionevoli, non
possono valere tutte insieme.

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
ragione a diffidarne. Il pregiudizio si fabbrica le proprie prove. Gli studi su
questo argomento (per esempio la rassegna di Mehrabi e colleghi
{cite}`mehrabi2021survey`) distinguono alcune sorgenti ricorrenti.

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

Per parlare di equità con rigore serve un vocabolario, e quattro cose da tenere
distinte: il **gruppo** a cui una persona appartiene, l'**esito reale** (quello
che poi è successo davvero), la **decisione** del modello, e il **punteggio** da
cui quella decisione si ricava fissando una soglia. Punteggio e decisione non
sono la stessa cosa, e la differenza tornerà utile: il punteggio è un numero fra
zero e uno, la decisione è il sì o il no che si ottiene tagliandolo a una certa
altezza.

Su questi ingredienti si contano gli errori. Si usano le stesse due misure del
capitolo di Machine Learning, il **tasso di veri positivi** e il **tasso di
falsi positivi**: «tasso» vuol dire soltanto «ogni quanti su cento», cioè una
percentuale scritta come frazione, e sono rispettivamente la quota di casi veri
che il modello prende e la quota di falsi allarmi su chi non c'entrava nulla. La
differenza rispetto a quel capitolo è una sola, ed è decisiva: qui i conti si
fanno **separatamente per ciascun gruppo** e poi si confrontano
({numref}`fig-equita-tassi`).

```{figure} ../figures/equita-tassi.svg
:name: fig-equita-tassi
:alt: Due matrici di confusione due per due affiancate, etichettate Gruppo A e Gruppo B, con le celle VP, FP, FN, VN riempite di numeri esempio; sotto ciascuna matrice il tasso di veri positivi (TPR) e il tasso di falsi positivi (FPR), con valori diversi fra i due gruppi.
:width: 100%

Lo stesso modello, gli stessi conti, due gruppi. Nel Gruppo A, su cento
persone a cui l'esito è poi capitato davvero il modello ne aveva individuate
$80$: $80$ su $100$, cioè $0{,}80$. Nel Gruppo B solo $60$, cioè $0{,}60$. E i
falsi allarmi sono $10$ su $100$ nel primo gruppo contro $30$ su $100$ nel
secondo. Quando le due coppie di numeri non coincidono, il modello sbaglia in
modo diverso a seconda del gruppo.
```

`````{tab} Elementare

Una precisazione prima dell'elenco, perché altrimenti ci si perde. In tutta la
sezione «sì» vuol dire **il modello ha detto la cosa che stavamo prevedendo**, e
quella cosa non è per forza bella: nel prestito il «sì» è «te lo diamo», nel
software dei tribunali il «sì» è «questa persona è ad alto rischio». Allo stesso
modo «l'esito è accaduto» vuol dire che è successo davvero quel che il modello
prevedeva, buono o brutto che sia. Sono parole neutre per un conto che si fa
uguale nei due casi.

Ci sono tre modi diversi di chiedere «il modello è equo?», e portano a tre
richieste distinte.

- **Stessa quota di sì (parità demografica).** Il modello dice «sì» alla stessa
  percentuale di persone in ogni gruppo. Se approva il 40% degli uomini, deve
  approvare il 40% delle donne: a prescindere da tutto il resto.
- **Stessa affidabilità sugli errori (equalized odds).** Fra le persone a cui
  l'esito è poi capitato davvero, la quota di sì è uguale nei due gruppi; e fra
  quelle a cui non è capitato, la quota di sì sbagliati è uguale. In altre
  parole: il modello sbaglia allo stesso modo su tutti. La
  {numref}`fig-equita-tassi` mostra il caso in cui questa richiesta è
  **violata**: stesso modello, ma i due tassi sono diversi fra Gruppo A e
  Gruppo B.
- **Stesso significato del punteggio (calibrazione).** Qui serve sapere una cosa
  sul punteggio: non è un voto, è una **previsione di probabilità**. «70» non
  vuol dire «bravo sette su dieci», vuol dire «di persone come questa, secondo
  me, l'esito capita a settanta su cento». Calibrato vuol dire che quella
  promessa viene mantenuta, e mantenuta allo stesso modo per tutti: se fra gli
  uomini con punteggio 70 l'esito capita davvero al 70%, lo stesso deve valere
  fra le donne. (Se il punteggio fosse scritto in decimi, «7» direbbe la stessa
  identica cosa: cambia la scala, non il patto.)

Sembrano tre facce della stessa medaglia. Vedremo tra poco che, sorprendentemente,
non possono quasi mai brillare tutte insieme.

`````

`````{tab} Superiore

Fissiamo la notazione: $A$ è l'**attributo protetto** che identifica il gruppo
(per esempio $A=a$ e $A=b$), $Y \in \{0,1\}$ è l'**esito reale**, $\hat{Y}$ è la
**decisione** del modello e $S \in [0,1]$ il **punteggio** da cui la decisione
si ottiene fissando una soglia. Sono le quattro grandezze nominate sopra a
parole; la distinzione fra $S$ e $\hat{Y}$ è quella che regge la sezione
successiva, perché un criterio enunciato su $S$ e lo stesso criterio enunciato
su $\hat{Y}$ sono affermazioni diverse.

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
«significa» la stessa cosa in ogni gruppo. Si noti che è una proprietà di $S$,
non di $\hat{Y}$: cambiare la soglia non tocca la calibrazione.

Accanto va tenuta una quarta condizione, che le somiglia e non coincide: la
**parità del valore predittivo** (*predictive parity*), che riguarda la
decisione,

$$
P(Y=1 \mid \hat{Y}=1,\, A=a) \;=\; P(Y=1 \mid \hat{Y}=1,\, A=b),
$$

cioè l'uguaglianza fra i gruppi del valore predittivo positivo
$\text{VPP}=P(Y=1\mid\hat{Y}=1)$, che è la *precision* del capitolo di Machine
Learning. È questo, e non la calibrazione, il criterio che Northpointe
rivendicava a difesa di COMPAS {cite}`dieterich2016compas`, e la distinzione
non è pedanteria: è il perno della sezione successiva.

`````

## Il risultato di impossibilità

Arriviamo al nodo. Le richieste appena viste non entrano in conflitto per caso:
alcune di esse sono **matematicamente incompatibili** ogni volta che i gruppi
partono da tassi di base diversi, cioè ogni volta che l'esito, nei dati, è più
frequente in un gruppo che nell'altro.

Conviene però dire subito una cosa che il libro finora ha lasciato implicita, e
che si legge di continuo detta male. Non c'è **un** teorema di impossibilità:
ce ne sono tre, dimostrati da persone diverse, e riguardano combinazioni
diverse di criteri. Assomigliano abbastanza da essere scambiati l'uno per
l'altro, e quando si scambiano si finisce per affermare cose false. Li vediamo
in fila; chi legge il livello Elementare può fermarsi al primo, che è quello
del caso COMPAS.

`````{tab} Elementare

Immagina due gruppi in cui l'esito che vogliamo prevedere è, nei dati, più
frequente in uno che nell'altro: i *tassi di base* differiscono. Ora pretendi
tre cose ragionevoli insieme. Uno: che fra le persone a cui il modello ha detto
sì, la quota di quelle a cui l'esito è poi capitato davvero sia la stessa nei
due gruppi (in gergo, la *parità del valore predittivo*: quando il modello dice
sì, dice sì con la stessa affidabilità per tutti). Due: che il modello generi la
stessa quota di falsi allarmi in ogni gruppo. Tre: che in ogni gruppo sfugga la
stessa quota di persone a cui l'esito è poi capitato. La seconda e la terza
richiesta sono le due metà dell'equalized odds vista sopra, la richiesta di
sbagliare allo stesso modo su tutti, qui guardata da vicino nei suoi due tipi
di errore.

Il teorema dice: non puoi. Se i tassi di base sono diversi, queste tre garanzie
non stanno insieme: qualunque due tu scelga di assicurare, la terza salta. E la
richiesta rimasta fuori, la stessa quota di sì in ogni gruppo (parità
demografica), non si salva da sé: con tassi di base diversi litiga a sua volta
con entrambe le altre. Non è un bug da correggere con codice migliore o più
dati: è un vincolo dell'aritmetica, come chiedere a un rettangolo di avere area
12 e perimetro 10 con lati interi (semplicemente non esiste). È il cuore della
disputa su COMPAS: l'inchiesta di ProPublica accusava il sistema di generare
molti più falsi allarmi fra gli imputati neri; l'azienda rispondeva che, quando
il suo punteggio diceva «alto rischio», ci prendeva ugualmente spesso nei due
gruppi. Avevano ragione **entrambe**, ed è proprio questo il punto.

Ci sono altri due teoremi che gli somigliano, e che il livello Superiore mette
in fila. Qui conta sapere che dicono cose diverse, e che nessuno dei tre dice
«non si può essere equi»: dicono quali garanzie si possono comprare insieme, e
quale bisogna lasciare andare.

`````

`````{tab} Superiore

**Primo teorema: Chouldechova (2017), parità del valore predittivo.** La chiave
è un'identità algebrica esatta che lega, all'interno di un gruppo, quattro
grandezze: la prevalenza $p = P(Y=1)$, il valore predittivo positivo
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
{cite}`angwin2016machine`. Si noti che cosa **non** compare in questo enunciato:
la calibrazione. Il teorema di Chouldechova parla di $\text{VPP}$, che è una
proprietà della decisione $\hat{Y}$, non di $S$.

**Secondo teorema: Kleinberg, Mullainathan e Raghavan (2016), calibrazione.**
Il risultato gemello, indipendente e quasi simultaneo, riguarda i punteggi
continui {cite}`kleinberg2017inherent`. Le tre condizioni in gioco sono la
calibrazione e i due **bilanciamenti di classe**: che il punteggio *medio*
ricevuto dai positivi sia lo stesso nei due gruppi, e che lo stesso valga per i
negativi. Coesistono solo nei casi degeneri (prevalenze identiche o predizione
perfetta). È un enunciato sui punteggi medi, non sui tassi della matrice di
confusione: il bilanciamento della classe positiva **non** è l'uguaglianza dei
TPR, e chiamarlo «bilanciamento dei falsi negativi» significa cambiarlo.

**Terzo teorema: Pleiss e colleghi (2017), calibrazione più equalized odds.**
Resta la domanda che i due precedenti lasciano aperta: calibrazione ed
equalized odds possono valere insieme? La risposta è nel lavoro di Geoff
Pleiss e colleghi {cite}`pleiss2017fairness`, e non è né sì né no: **stanno
insieme solo con un vincolo d'errore alla volta**, o i falsi positivi o i falsi
negativi, non entrambi, salvo prevalenze uguali o predittore perfetto. Le
ipotesi contano quanto l'enunciato: un unico classificatore probabilistico
usato tale e quale, dunque **senza soglie diverse per gruppo**, e tassi d'errore
*generalizzati*, calcolati sui punteggi invece che sulle decisioni binarie.

Vale la pena vedere perché quelle ipotesi non sono un dettaglio, perché è la
parte che insegna. Se le si lascia cadere, l'incompatibilità sparisce. Si
prendano due gruppi con un punteggio calibrato **per costruzione** in entrambi
(l'etichetta estratta con probabilità pari al punteggio, quindi
$P(Y=1\mid S=s, A=a)=s$ esattamente) e prevalenze $0{,}50$ e $0{,}45$: con una
soglia per gruppo, $0{,}584$ e $0{,}537$, si ottiene
$\text{TPR}=0{,}489$ e $\text{FPR}=0{,}202$ in **tutti e due**, cioè equalized
odds pieno, con la calibrazione intatta perché le soglie non toccano $S$. A
divergere è il valore predittivo, $0{,}707$ contro $0{,}664$: esattamente come
impone l'identità di Chouldechova. Non è una smentita di Pleiss, è il suo
contrappunto: due soglie sono due classificatori, e l'ipotesi era che ce ne
fosse uno solo. Il post-processing di Hardt, Price e Srebro, che la sezione
sulle mitigazioni richiama, vive proprio in questo spiraglio.

**E la terna dei criteri statistici.** Per *independence*, *separation* e
*sufficiency* il risultato è più severo, come riassumono Barocas, Hardt e
Narayanan {cite}`barocas2023fairness`: sono incompatibili già **a due a due**,
fuori dai casi degeneri. Independence e separation coesistono solo se
$A \perp Y$ oppure $\hat{Y} \perp Y$; independence e sufficiency solo se
$A \perp Y$; separation e sufficiency solo se $A \perp Y$ o la predizione è
perfetta. Qui però le tre condizioni sono enunciate sullo **stesso** predittore:
è la ragione per cui il controesempio di poco sopra non le contraddice, dato che
lì la calibrazione riguarda $S$ e l'equalized odds riguarda un $\hat{Y}$
ottenuto con due soglie.

`````

### Quel che il teorema non dice: i tassi di base sono misure

Tutti e tre i risultati partono dallo stesso presupposto, e conviene guardarlo
in faccia perché è il punto in cui il capitolo rischia di smentire sé stesso.
Il conflitto si accende quando i tassi di base differiscono. Ma «tasso di base»
non è un dato di natura: è la frequenza di un esito **così come lo abbiamo
misurato**, ed è esattamente la grandezza che il bias di misura, definito
all'inizio di questa sezione, può distorcere. In COMPAS il tasso di base non è
la frequenza dei reati: è la frequenza dei **riarresti**. Se gli arresti
dipendono anche da dove passano le pattuglie, allora la difesa «il nostro
punteggio è affidabile allo stesso modo rispetto a quel tasso di base» è una
difesa rispetto a un numero che porta dentro il problema che si voleva
misurare.

Questo non rende falso nessuno dei tre teoremi: sono identità e restano vere
per qualunque $Y$ si scelga. Cambia però cosa se ne può concludere. Il teorema
dice che, dati quei numeri, non si possono avere tutte le garanzie; non dice
che quei numeri siano i numeri giusti, e non dice quale garanzia tenere.
Presentare la scelta come puramente normativa («è una decisione di valore») è
vero e insufficiente, perché lascia credere che almeno le premesse del conflitto
siano solide. Una di esse non lo è.

`````{tab} Elementare

È la domanda che viene naturale e che di solito non si trova scritta: *e se i
due gruppi hanno tassi di base diversi perché il mondo è ingiusto, il teorema
che cosa dimostra?*

Dimostra esattamente quello che dice, né più né meno: che con quei numeri lì
non puoi avere tutte le garanzie insieme. Non dice che quei numeri fotografino
la realtà. Se in un quartiere passano più pattuglie, il «tasso di base» di quel
gruppo è più alto perché ci sono più arresti, non necessariamente perché ci
siano più reati. Il teorema continua a valere, e continua a costringerti a
scegliere; ma il conto che ti costringe a scegliere è stato fatto su una misura
storta.

La conseguenza pratica è che ci sono due domande, non una. La prima è «quale
garanzia di equità pretendo?», ed è la domanda che il teorema mette sul tavolo.
La seconda viene prima e si dimentica quasi sempre: «l'esito che sto prevedendo
è davvero quello che mi interessa, o è solo quello che qualcuno ha registrato?».
Un capitolo che facesse solo la prima insegnerebbe a scegliere bene dentro un
problema mal posto.

`````

`````{tab} Superiore

L'osservazione non è esterna alla letteratura: è di Chouldechova stessa.
Fogliato, G'Sell e Chouldechova {cite}`fogliato2020fairness` studiano la
valutazione dell'equità quando l'etichetta osservata è una versione rumorosa e
**sistematicamente distorta** di quella d'interesse, ed è precisamente il caso
dell'arresto usato come proxy del reato. La conclusione è che anche piccole
distorsioni nell'etichetta osservata possono rovesciare le conclusioni di
un'analisi condotta su di essa.

Il meccanismo si vede con una costruzione minima. Si prendano due gruppi con
tasso di base **vero identico** e una sola differenza, la probabilità che il
fatto venga rilevato ($0{,}90$ nel primo, $0{,}50$ nel secondo). Il punteggio
resta calibrato per costruzione sull'etichetta osservata in entrambi. Le
prevalenze vere coincidono; le prevalenze *osservate* no, e a quel punto il
teorema, che legge la riga «arresto», dichiara inevitabile un divario che è
interamente prodotto dalla misura.

Da cui una regola di lettura per tutta la sezione: ogni volta che si scrive
$P(Y=1)$ conviene ricordarsi che $Y$ è un dato raccolto da qualcuno, in un
posto, con un criterio. La formalizzazione è onesta a patto di non far passare
$Y$ per il fenomeno invece che per la sua registrazione.

`````

## Attenuare il bias: tre punti di intervento

Se una cura definitiva non esiste, restano comunque leve per ridurre il divario.
Si classificano per il *momento* in cui agiscono: prima dell'addestramento, sui
dati; durante, sull'obiettivo; dopo, sulle decisioni già prodotte.

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
  *reweighting* (pesi $w_i$ per esempio, calcolati così da rendere $Y$
  indipendente da $A$ nel campione pesato; l'effetto sperato a valle è un
  classificatore più vicino alla parità demografica), ricampionamento dei gruppi
  sotto-rappresentati, o rimozione/decorrelazione delle feature che fungono da
  *proxy* dell'attributo protetto. Vantaggio: agnostico al modello a valle.
- **In-processing.** Si modifica l'obiettivo di addestramento aggiungendo un
  **vincolo** o un termine di **regolarizzazione** di equità, per esempio
  minimizzare $\mathcal{L}_{\text{pred}} + \lambda\,\mathcal{L}_{\text{fair}}$
  dove $\mathcal{L}_{\text{fair}}$ penalizza il divario di TPR/FPR fra i gruppi
  e $\lambda$ regola il compromesso equità-accuratezza. Adversarial debiasing e
  ottimizzazione vincolata rientrano qui.
- **Post-processing.** Si lascia intatto il modello e si aggiustano le
  **soglie**: Hardt, Price e Srebro {cite}`hardt2016equality` mostrano come
  derivare soglie per-gruppo (eventualmente randomizzate) che raggiungono
  l'equalized odds a partire da un qualsiasi punteggio già addestrato (una
  costruzione geometrica sulle curve ROC dei due gruppi).

Il risultato di impossibilità della sezione precedente resta sullo sfondo, e
il post-processing è il posto in cui si vede meglio che cosa comprano davvero
queste tecniche. Le soglie per gruppo raggiungono l'equalized odds senza toccare
il punteggio, quindi senza toccarne la calibrazione; ciò che non possono fare è
tenere fermo anche il valore predittivo, che si separa fra i gruppi non appena
le prevalenze differiscono. Nessuna delle tre leve annulla il conflitto: sposta
*quale* criterio privilegiare, e quel «quale» non è una scelta tecnica.

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

Formalmente, dato un metro di distanza fra individui
$d(\mathbf{x}_i, \mathbf{x}_j)$ e una distanza fra distribuzioni di esito $D$,
il classificatore (che a ogni individuo associa una distribuzione sugli esiti)
deve essere **Lipschitz** {cite}`dwork2012fairness`:

$$
D\bigl(M(\mathbf{x}_i),\, M(\mathbf{x}_j)\bigr) \;\le\; d(\mathbf{x}_i, \mathbf{x}_j),
$$

dove $M(\mathbf{x})$ è la distribuzione di esito assegnata a $\mathbf{x}$. In
parole: individui vicini secondo $d$ ricevono esiti vicini secondo $D$; il
modello non può «strappare» a piacere due punti che il metro dichiara simili. È
una garanzia più forte e più fine dell'equità di gruppo, ma sposta l'intera
difficoltà su $d$: la metrica di somiglianza specifica del compito è assunta
*data*, mentre in pratica sceglierla è precisamente il giudizio di valore che si
voleva rendere oggettivo. Per questo l'equità individuale è teoricamente
elegante ma di rado applicabile tale e quale.

`````

## Il conflitto, coi numeri

Chiudiamo il cerchio con un esperimento riproducibile. Inventiamo due gruppi
di persone e diamo a ciascuna un punteggio di rischio onesto **per
costruzione**: se il punteggio dice $0{,}7$, l'esito accade davvero sette
volte su dieci, in entrambi i gruppi allo stesso modo (è la calibrazione di
prima). L'unica differenza è che l'esito è complessivamente più frequente in
un gruppo che nell'altro: i tassi di base diversi da cui parte il teorema. Poi
applichiamo la stessa soglia a tutti e contiamo gli errori gruppo per gruppo.

`````{tab} Elementare

Nel codice qui sotto il punteggio fa da dado truccato: se una persona ha
punteggio $0{,}7$, tiriamo un dado che dice sì sette volte su dieci, e quello
che esce diventa il suo esito reale. Il dado funziona nello stesso modo per
tutti; a cambiare fra i due gruppi è soltanto quanti punteggi alti girano.

`````

`````{tab} Superiore

L'etichetta reale è estratta con probabilità pari al punteggio,
$P(Y=1\mid S=s)=s$, identica nei due gruppi: il punteggio è calibrato per
costruzione. A differire è la sola distribuzione marginale di $S$ (due Beta di
media $0{,}50$ e $0{,}33$), e con essa la prevalenza $p=\mathbb{E}[S]$.

`````

Se non programmi, il codice si può saltare: quello che conta sono i numeri
stampati sotto, ed è di quelli che parla il commento.

```python
import numpy as np

rng = np.random.default_rng(0)

def genera_gruppo(n, alpha, beta):
    # Il punteggio è calibrato per costruzione: P(Y=1 | S=s) = s
    s = rng.beta(alpha, beta, size=n)          # punteggio in [0,1]
    y = (rng.random(n) < s).astype(int)        # etichetta vera: 1 con probabilità s
    return s, y

# I due numeri decidono quanti punteggi alti girano nel gruppo: piu' il primo
# supera il secondo, piu' il gruppo e' spostato verso i punteggi alti.
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

# Seconda prova: una soglia diversa per ciascun gruppo, scelta per pareggiare
# i due tassi di errore. Il punteggio non viene toccato, quindi resta calibrato.
print("\nCon una soglia per gruppo (0,72 per A e 0,57 per B):")
for nome, s, y, t in [("A", sA, yA, 0.72), ("B", sB, yB, 0.57)]:
    sel, tpr, fpr, ppv = tassi(s, y, t)
    print(f"  Gruppo {nome}: TPR={tpr:.3f}  FPR={fpr:.3f}  VPP={ppv:.3f}")
```

L'esecuzione stampa qualcosa come:

```text
Gruppo A: base=0.500  selection=0.502  TPR=0.658  FPR=0.346  VPP=0.656
Gruppo B: base=0.329  selection=0.188  TPR=0.348  FPR=0.110  VPP=0.607

Calibrazione (bin di score -> frazione reale di positivi):
  Gruppo A: [0.0,0.2)->0.16  [0.2,0.4)->0.31  [0.4,0.6)->0.50  [0.6,0.8)->0.69  [0.8,1.0)->0.85
  Gruppo B: [0.0,0.2)->0.13  [0.2,0.4)->0.29  [0.4,0.6)->0.48  [0.6,0.8)->0.67  [0.8,1.0)->0.81

Con una soglia per gruppo (0,72 per A e 0,57 per B):
  Gruppo A: TPR=0.215  FPR=0.056  VPP=0.795
  Gruppo B: TPR=0.219  FPR=0.055  VPP=0.660
```

Le colonne dicono, per ciascun gruppo: quanto è frequente davvero l'esito
(`base`), a quante persone il modello dice sì (`selection`), la quota di sì
giusti fra le persone a cui l'esito è poi capitato (`TPR`), la quota di falsi
allarmi (`FPR`) e quanti dei sì del modello erano giusti (`VPP`).

Le due righe della calibrazione sono **essenzialmente identiche**: in ogni
fascia di punteggio, la frazione reale di positivi è pressoché la stessa fra i
gruppi (e coincide con il punteggio medio della fascia, come impone la
calibrazione per costruzione). La calibrazione, cioè, *vale*. Eppure, con la
soglia unica, la quota di falsi allarmi è tre volte più alta nel Gruppo A
($0{,}346$ contro $0{,}110$) e diverge nettamente anche il TPR ($0{,}658$
contro $0{,}348$): le due metà dell'equalized odds saltano entrambe.

Attenzione però a non trarne la conclusione sbagliata, che è quella che si
legge più spesso. Non è che quei tassi *non si possano* allineare: le ultime tre
righe stampate fanno esattamente questo, con una soglia diversa per gruppo, e
li allineano bene ($0{,}215$ contro $0{,}219$ e $0{,}056$ contro $0{,}055$),
senza toccare il punteggio e senza toccare le frequenze di base, che restano
$0{,}50$ e $0{,}33$. Quel che si sposta, e che con la soglia unica passava
inosservato, è il **valore predittivo**: $0{,}795$ nel Gruppo A contro
$0{,}660$ nel Gruppo B. Cioè: quando il modello dice sì, ci prende molto più
spesso in un gruppo che nell'altro.

È il teorema, nella sua forma esatta: tre garanzie, due alla volta. Un
esperimento non dimostra un'impossibilità (un esempio non è una prova), ma
questo mostra la forma del vincolo, e mostra soprattutto che la scelta è una
scelta vera: si può decidere di pareggiare gli errori accettando che il «sì»
valga meno in un gruppo, oppure di pareggiare l'affidabilità del sì accettando
tassi di errore diversi. Non esiste la terza opzione.

`````{tab} Elementare

E qui il cerchio si chiude sull'inchiesta da cui era partito il capitolo. I due
gruppi A e B, con i loro numeri, sono la fotografia della disputa su COMPAS: da
una parte ProPublica, che guardava i falsi allarmi e li trovava molto più alti
per gli imputati neri; dall'altra l'azienda, che guardava quanto spesso il suo
«alto rischio» ci prendeva davvero e lo trovava uguale nei due gruppi.
Guardavano due colonne diverse della stessa tabella, e nessuno dei due mentiva.

`````

`````{tab} Superiore

La configurazione a soglia unica riproduce la struttura del caso COMPAS
{cite}`angwin2016machine`: divario nei tassi di errore con punteggio calibrato.
Quella a due soglie riproduce il post-processing di Hardt, Price e Srebro
{cite}`hardt2016equality`, e mostra il prezzo che si paga a valle, la
separazione del VPP, che è il criterio rivendicato da Northpointe
{cite}`dieterich2016compas`. Resta il caveat della sezione precedente, e vale
anche qui: nella simulazione i tassi di base sono noti per costruzione, mentre
su dati veri sono stimati da etichette che possono essere a loro volta distorte.

`````

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

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Qui **bias** vuol dire *pregiudizio*: non è il numero dentro il neurone né
  l'errore di un modello troppo semplice, che nel libro portano lo stesso nome.
- Il pregiudizio non nasce nel codice, entra **prima**, da quattro porte: il
  passato è ingiusto; di qualche gruppo ci sono pochi esempi; quello che è
  scritto nei dati non è la cosa che credevamo («arrestato» al posto di «ha
  commesso un reato»); e il modello si morde la coda, perché le sue decisioni
  di oggi diventano i dati di domani.
- Per misurarlo si usa la tabella a quattro caselle del capitolo di Machine
  Learning, compilata **un gruppo alla volta**, e si confrontano due numeri:
  quanti dei casi veri il modello prende, e quanti falsi allarmi dà.
- Ci sono tre idee di equità tutte ragionevoli: stessa quota di sì per tutti;
  stessi errori per tutti; e stesso significato del punteggio, che non è un
  voto ma una previsione di probabilità. **Non si possono avere tutte
  insieme** quando l'esito è più frequente in un gruppo che nell'altro: ne
  scegli due, la terza salta.
- E quella frequenza è un numero **misurato** da qualcuno, non un dato di
  natura: se è la frequenza degli arresti invece che dei reati, il conto che ti
  costringe a scegliere è stato fatto su una misura storta.
- Si può **attenuare**, non risolvere, e si può farlo in tre momenti: prima
  della gara (sistemando i dati), durante (cambiando le regole
  dell'addestramento) o dopo (usando una soglia diversa per gruppo).
- Quale garanzia pretendere non è un calcolo: è una **decisione**, e va presa
  alla luce del sole.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Il bias non nasce nel codice ma **a monte**, nei dati: passato iniquo,
  campione non rappresentativo, etichette-proxy distorte e *feedback loop* che
  si auto-conferma. *Bias in, bias out* {cite}`mehrabi2021survey`.
- L'equità di gruppo si misura riusando la **matrice di confusione** del capitolo
  di Machine Learning, ma *separatamente per gruppo*. Tre criteri: **parità
  demografica** (stessa quota di sì), **equalized odds** (stessi TPR e FPR)
  {cite}`hardt2016equality`, **calibrazione** (stesso significato del punteggio,
  proprietà di $S$ e non di $\hat{Y}$).
- **Tre risultati di impossibilità distinti**, da non confondere:
  Chouldechova {cite}`chouldechova2017fair`, parità del **valore predittivo**
  più i due tassi d'errore, due su tre; Kleinberg, Mullainathan e Raghavan
  {cite}`kleinberg2017inherent`, calibrazione più i due **bilanciamenti di
  classe** (punteggi medi, non tassi); Pleiss e colleghi
  {cite}`pleiss2017fairness`, calibrazione più equalized odds, ma con **un solo
  vincolo d'errore alla volta** e sotto l'ipotesi di un unico classificatore
  senza soglie per gruppo. È il nodo del caso COMPAS
  {cite}`angwin2016machine`, dove ProPublica e Northpointe
  {cite}`dieterich2016compas` avevano ragione entrambe.
- I **tassi di base** che accendono il conflitto sono grandezze *misurate*: su
  COMPAS sono riarresti, non reati, e il bias di misura le tocca
  {cite}`fogliato2020fairness`. Il teorema resta vero; quel che non è dato è
  che le sue premesse siano neutrali.
- Si può **attenuare**, non risolvere, intervenendo in pre-processing
  (riequilibrio dei dati), in-processing (vincoli/regolarizzazione di equità) o
  post-processing (soglie per gruppo: raggiungono l'equalized odds lasciando
  intatta la calibrazione, al prezzo di separare il VPP).
- L'**equità individuale** {cite}`dwork2012fairness` chiede esiti simili per
  individui simili, ma sposta la difficoltà sul definire «simile».
- Nessuna metrica è «quella giusta»: scegliere il criterio di equità è una
  **decisione di valore**, non un calcolo.
```

`````
