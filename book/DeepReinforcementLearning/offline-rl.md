# Offline reinforcement learning: imparare da dati fissi

Nel 2018 un gruppo dell'Imperial College di Londra ha addestrato un sistema
(lo hanno chiamato *AI Clinician*) a suggerire i dosaggi di fluidi e
vasopressori per i pazienti con sepsi in terapia intensiva. Il modello non ha
mai provato una terapia su un malato vero: ha imparato guardando l'archivio di
ricoveri passati, con le decisioni prese dai medici e ciò che ne è seguito. È
l'unica strada possibile, del resto. Nessun comitato etico autorizzerebbe un
agente a somministrare farmaci «a caso» per esplorare, come fa il Q-learning
in un videogioco; e nessuno lo lascerebbe sterzare a caso al volante di
un'auto, o proporre acquisti assurdi a milioni di utenti per misurarne la
reazione.

Eppure è proprio così (provando, sbagliando, esplorando) che gli agenti dei
capitoli precedenti imparano. Il **reinforcement learning offline** (o *batch
RL*) affronta il caso opposto e più scomodo: imparare la migliore strategia
possibile *senza mai interagire con l'ambiente*, disponendo solo di un
registro di esperienze già accadute {cite}`sutton2018reinforcement`.

## Imparare da un archivio, non dall'esperienza

`````{tab} Elementare

Pensa alla differenza tra imparare a cucinare ai fornelli e imparare studiando
i quaderni di ricette di tua nonna. Nel primo caso assaggi, aggiusti il sale,
rifai il piatto: ogni tentativo ti dà un riscontro nuovo. Nel secondo hai solo
un archivio chiuso (quello che lei ha cucinato negli anni, con gli esiti) e da
lì devi tirar fuori la ricetta migliore, senza poter provare nulla di persona.

L'RL offline è la seconda situazione. Qualcuno, in passato, ha agito
nell'ambiente seguendo una sua strategia (un medico, un guidatore, un vecchio
algoritmo di raccomandazione) e ha lasciato un diario: «in questa situazione
ho fatto questo, ed è successo quello». L'agente riceve solo il diario. Non
può tornare indietro a chiedere «e se avessi fatto diversamente?». Deve
cavarsela con ciò che è scritto.

`````

`````{tab} Superiore

Formalmente disponiamo di un dataset fisso di transizioni

$$
\mathcal{D} = \big\{\, (s_i,\, a_i,\, r_i,\, s'_i) \,\big\}_{i=1}^{N},
$$

raccolto da una o più **policy comportamentali** $\pi_\beta$ (di cui in genere
non conosciamo la forma esplicita) e mai più ampliato: non c'è alcuna nuova
interazione con l'ambiente durante l'addestramento. L'obiettivo resta quello
consueto (trovare una policy $\pi$ che massimizzi il ritorno atteso scontato
$\mathbb{E}\big[\sum_t \gamma^t r_t\big]$) ma va raggiunto *dentro il
supporto* di $\mathcal{D}$. La differenza con il setting online non è
cosmetica: il Q-learning e i metodi a gradiente di policy dei capitoli
precedenti presuppongono di poter *provare* le proprie ipotesi, e sono le
prove a correggere le stime sbagliate. Togliere le prove cambia la natura del
problema.

`````

Verrebbe da pensare che sia facile: il Q-learning è già *off-policy*; impara
la policy ottima anche mentre ne segue un'altra, come abbiamo visto nella
sezione sul Q-learning. Basta dargli in pasto le transizioni del diario invece
di raccoglierle sul momento, e lasciarlo lavorare. Purtroppo, fatto così,
fallisce quasi sempre. Capire *perché* è il cuore di tutto l'argomento.

## Il buco nero delle azioni mai viste

Il colpevole ha un nome tecnico: **distributional shift**, lo scarto tra la
distribuzione di azioni presente nel dataset e quella che la nuova policy
vorrebbe seguire.

`````{tab} Elementare

Torniamo ai quaderni della nonna. Immagina che nei suoi decenni ai fornelli
non abbia *mai* messo il peperoncino in un piatto. Nel diario, alla voce
«peperoncino», non c'è nulla. Ora, se ti chiedessero «quanto sarebbe buono
questo risotto con tre cucchiai di peperoncino?», la risposta onesta è «non ne
ho idea». Ma un modello ingenuo, quando deve indovinare, tira comunque un
numero, e a volte, per puro caso della sua matematica interna, tira un numero
*altissimo*. «Risotto al peperoncino: buonissimo, dieci e lode.»

Ecco la trappola. L'agente cerca sempre l'azione col voto più alto. Se un voto
gonfiato per sbaglio spunta proprio su un'azione mai provata, l'agente ci si
butta a capofitto. E siccome non può cucinarla davvero per scoprire che è
immangiabile, quel voto sballato non viene mai corretto: anzi, contagia le stime
vicine e peggiora tutto. Nel gioco vero, in cui provi le mosse, l'errore si
smaschera da sé; qui no, e resta lì a fare danni.

`````

`````{tab} Superiore

Il target del Q-learning è
$y = r + \gamma \max_{a'} Q_\theta(s', a')$. Il problema è l'operatore
$\max_{a'}$: spazia su *tutte* le azioni, comprese quelle che $\pi_\beta$ non ha
mai eseguito in $s'$. Su quelle azioni *out-of-distribution* (OOD) la rete
$Q_\theta$ non ha mai visto dati e non fa che **estrapolare**; i suoi errori di
estrapolazione sono casuali, ma il $\max$ non è casuale: seleziona
sistematicamente i valori più alti, cioè proprio le sovrastime. Il target
risulta gonfiato, la policy insegue quelle azioni fantasma, e per bootstrapping
il valore inflazionato si propaga all'indietro agli altri stati.

Nel RL online questo circolo si spezza da solo: eseguendo l'azione
sopravvalutata si osserva la ricompensa vera, bassa, e la stima si corregge.
Offline il correttivo non arriva mai (non c'è nuova interazione) e l'errore
diverge. Fujimoto e colleghi {cite}`fujimoto2019off` hanno mostrato che questa
**extrapolation error** è la ragione per cui gli algoritmi off-policy standard
crollano su dati fissi, spesso *peggiorando* al crescere delle iterazioni
invece di migliorare. Tutte le tecniche che seguono attaccano lo stesso nemico
da angolazioni diverse: impedire, in un modo o nell'altro, di fidarsi delle
azioni fuori dal supporto dei dati.

`````

Prima di vedere i rimedi, tocchiamo con mano il fenomeno. Il codice qui sotto
(puro NumPy, eseguibile) stima un valore-azione da un dataset che copre solo
una fetta dello spazio, e mostra il massimo «naive» cadere su un'azione mai
osservata, con un valore stimato gonfiato ben lontano da quello vero.

```python
import numpy as np
rng = np.random.default_rng(0)

# Valore-azione "vero": una parabola con l'ottimo in a = 0.3
def Q_vero(a):
    return -(a - 0.3) ** 2

# La policy comportamentale ha esplorato solo azioni "prudenti" in [-1.0, -0.2]:
# il dataset NON contiene mai l'azione ottima.
a_dati = rng.uniform(-1.0, -0.2, size=40)
q_dati = Q_vero(a_dati) + rng.normal(0, 0.01, size=a_dati.shape)

# Stimiamo Q con un modello flessibile (polinomio di grado 5) sui soli dati.
coeff = np.polyfit(a_dati, q_dati, deg=5)
Q_stima = np.poly1d(coeff)

azioni = np.linspace(-1, 1, 201)                 # candidate su TUTTO lo spazio

# (1) max naive: nessun vincolo, come nel Q-learning off-policy classico
a_naive = azioni[np.argmax(Q_stima(azioni))]
print(f"naive     : a*={a_naive:+.2f}  Q_stimato={Q_stima(a_naive):+.2f}  "
      f"Q_vero={Q_vero(a_naive):+.2f}")

# (2) max vincolato al supporto dei dati (idea alla base di BCQ)
in_supp = (azioni >= a_dati.min()) & (azioni <= a_dati.max())
a_vinc = azioni[in_supp][np.argmax(Q_stima(azioni[in_supp]))]
print(f"vincolato : a*={a_vinc:+.2f}  Q_stimato={Q_stima(a_vinc):+.2f}  "
      f"Q_vero={Q_vero(a_vinc):+.2f}")
```

Il massimo naive sceglie un'azione al bordo dello spazio, mai osservata, a cui
il polinomio assegna un valore assurdamente alto rispetto al suo valore reale.
Vincolando la ricerca al supporto dei dati, stima e realtà tornano a coincidere:
è, in miniatura, la prima famiglia di soluzioni.

## BCQ: restare vicini a ciò che è stato visto

Il primo algoritmo pensato apposta per il RL offline profondo è **BCQ**
(*Batch-Constrained deep Q-learning*), di Scott Fujimoto, David Meger e Doina
Precup {cite}`fujimoto2019off`, presentato nel 2019. L'idea è quella suggerita
dall'esempio: se il guaio nasce dal valutare azioni fuori distribuzione, allora
non valutiamole affatto. BCQ limita il $\max$ alle sole azioni *plausibili*
secondo il dataset.

Per sapere quali azioni siano plausibili, BCQ addestra un **modello
generativo** (un *variational autoencoder*), sulle coppie $(s, a)$ del
dataset: dato uno stato, genera azioni simili a quelle che $\pi_\beta$ avrebbe
scelto in situazioni analoghe. La rete $Q$ viene poi massimizzata solo su un
pugno di azioni campionate da questo generatore (con una piccola perturbazione
appresa che concede un margine di miglioramento). L'operatore di
massimizzazione non può più cadere nelle regioni fantasma: sceglie il meglio
*tra ciò che si sarebbe davvero potuto fare*. È il modo più letterale di
rispondere al distributional shift: costruire un recinto attorno ai dati e non
uscirne.

## CQL: essere pessimisti sull'ignoto

BCQ mette un recinto *esplicito* attorno alle azioni. Un anno dopo, nel 2020,
Aviral Kumar, Aurick Zhou, George Tucker e Sergey Levine propongono un approccio
più elegante che non ha bisogno di un modello generativo separato: **CQL**
(*Conservative Q-Learning*) {cite}`kumar2020conservative`. Invece di vietare le
azioni OOD, le rende *poco appetibili* agendo direttamente sulla loss.

`````{tab} Elementare

Immagina un critico gastronomico prudente. La regola che si dà è semplice: «di
ogni piatto che non ho mai assaggiato, do per scontato che sia mediocre; di
quelli documentati nei quaderni, mi fido di ciò che c'è scritto». In questo modo
nessun piatto sconosciuto potrà mai battere, sulla carta, un piatto ben
documentato: il critico non correrà mai dietro a una fantasia.

CQL insegna esattamente questa prudenza alla rete dei voti. A ogni
aggiornamento aggiunge due spinte: **abbassa** i voti delle azioni fuori dal
dataset e **alza** quelli delle azioni davvero presenti. Il risultato è un
sistema di voti *conservativo* (pessimista su tutto ciò che non ha visto) che
difficilmente si fa abbagliare dall'ignoto. Perde forse qualche occasione
buona ma nascosta; in cambio non si getta mai in un burrone che non ha mai
esplorato.

`````

`````{tab} Superiore

CQL aggiunge alla consueta minimizzazione dell'errore di Bellman un termine
regolarizzatore che comprime i valori delle azioni fuori distribuzione:

$$
\min_{Q}\ \alpha\Big(
\underbrace{\mathbb{E}_{s\sim\mathcal{D},\, a\sim\mu(\cdot\mid s)}\big[Q(s,a)\big]}_{\text{abbassa: azioni OOD}}
-
\underbrace{\mathbb{E}_{(s,a)\sim\mathcal{D}}\big[Q(s,a)\big]}_{\text{alza: azioni nel dataset}}
\Big)
+ \tfrac{1}{2}\,\mathbb{E}_{\mathcal{D}}\Big[\big(Q - \hat{\mathcal{B}}Q\big)^2\Big].
$$

Qui $\mu(\cdot\mid s)$ è una distribuzione *ampia* di azioni candidate (nella
pratica una softmax sui $Q$ stessi, o l'uniforme), $\hat{\mathcal{B}}Q$ è il
target di Bellman e $\alpha>0$ dosa la conservatività. Il primo termine tira
*giù* i $Q$ delle azioni pescate da $\mu$ (tipicamente OOD), mentre il secondo
li tira *su* sulle azioni realmente presenti in $\mathcal{D}$. Kumar e
colleghi dimostrano che, per $\alpha$ abbastanza grande e con $\mu$ agganciata
alla policy che si sta valutando, il **valore atteso** delle azioni sotto la
$Q$ così ottenuta minora in ogni stato il vero valore della policy: un limite
inferiore *in valore*, non punto per punto (la singola stima $Q(s,a)$ può
ancora eccedere quella vera). La prudenza sull'ignoto resta comunque una
garanzia formale, non un'euristica.

`````

## IQL: non guardare mai fuori dai dati

CQL valuta ancora le azioni OOD, salvo poi penalizzarle. Nel 2022 Ilya
Kostrikov, Ashvin Nair e Sergey Levine portano l'idea alle estreme conseguenze
con **IQL** (*Implicit Q-Learning*) {cite}`kostrikov2022offline`: costruire
una policy migliore di quella che ha raccolto i dati *senza mai interrogare la
rete dei voti su un'azione che non sia nel dataset*. Se non guardi mai fuori,
non puoi essere ingannato da ciò che c'è fuori.

`````{tab} Elementare

Come si fa a scegliere bene senza mai considerare piatti mai cucinati? IQL
cambia la domanda che rivolge ai quaderni. Non chiede più «quanto varrebbe
questa ricetta ipotetica?», che è la domanda da cui nascono i voti di
fantasia: chiede «nelle serate come questa, quanto hanno reso le ricette
*migliori* fra quelle davvero provate?». È come giudicare il potenziale di una
cucina dai suoi piatti più riusciti, senza fantasticare su menù mai esistiti.

Con quel metro («il meglio di ciò che è stato fatto qui») si rileggono poi le
pagine del diario: le mosse che hanno reso più del solito nella loro
situazione vengono imitate di più, le altre di meno. Dall'inizio alla fine,
nessuna azione fuori dal diario viene mai nemmeno nominata: dove gli altri
metodi mettono recinti o penalità, IQL toglie proprio l'occasione di
sbagliare.

`````

`````{tab} Superiore

Il trucco sta nel sostituire il $\max_{a'} Q(s',a')$ (che costringe a spaziare
su tutte le azioni) con una funzione valore $V(s')$ stimata in modo obliquo,
tramite **regressione expectile**. Un expectile alto (vicino a $1$) fa sì che
$V(s)$ approssimi il *massimo* dei valori $Q(s,a)$ ma solo sulle azioni $a$
*effettivamente osservate* in quello stato: coglie «il meglio tra ciò che è
stato fatto» senza mai nominare un'azione ipotetica. Le tre reti si addestrano
per pura regressione sui dati:

$$
\mathcal{L}_V = \mathbb{E}_{(s,a)\sim\mathcal{D}}\Big[ L_2^{\tau}\big(Q(s,a) - V(s)\big)\Big],
\qquad
L_2^{\tau}(u) = \big|\,\tau - \mathbb{1}(u<0)\,\big|\; u^2,
$$

$$
\mathcal{L}_Q = \mathbb{E}_{(s,a,s')\sim\mathcal{D}}\Big[\big(r + \gamma\, V(s') - Q(s,a)\big)^2\Big].
$$

Dove $\tau \in (0,1)$ è l'**expectile** (in pratica $\tau \approx 0{,}7$–$0{,}9$):
la perdita asimmetrica $L_2^\tau$ pesa di più i residui positivi, spingendo $V$
verso l'alto della distribuzione dei $Q$ nel dataset. Il target di $\mathcal{L}_Q$
usa $V(s')$, **non** un massimo su azioni arbitrarie: ecco perché nessuna azione
OOD viene mai valutata. La policy si estrae infine per *advantage-weighted
regression*, imitando le azioni del dataset pesate per il loro vantaggio
$Q(s,a)-V(s)$.

`````

Il nucleo di IQL, in PyTorch, è una perdita volutamente sbilanciata, che tira
la stima del «meglio già fatto» verso l'alto dei voti realmente osservati:

```python
import torch

def expectile_loss(q, v, tau=0.8):
    # regressione expectile: residui positivi pesati di piu (tau > 0.5)
    diff = q - v
    peso = torch.where(diff > 0, tau, 1.0 - tau)
    return (peso * diff.pow(2)).mean()
```

## Decision Transformer: l'RL come previsione di sequenze

Fin qui abbiamo curato il Q-learning perché sopravvivesse ai dati fissi. Nel 2021
un gruppo di Berkeley (Lili Chen, Kevin Lu e colleghi) propone di cambiare
proprio domanda {cite}`chen2021decision`: e se smettessimo di stimare valori e
trattassimo l'apprendimento offline come un problema di **modellazione di
sequenze**, lo stesso su cui i Transformer eccellono nel linguaggio?

`````{tab} Elementare

Un modello linguistico impara a completare le frasi: gli dài l'inizio e lui
predice la parola dopo, avendo letto miliardi di testi. Il **Decision
Transformer** fa lo stesso, ma la «frase» è una partita: una sequenza di
situazioni e mosse. Lo alleni sui diari delle partite passate a completare la
sequenza: «dopo questa situazione, che mossa è stata fatta?».

C'è un ingrediente in più, ed è geniale nella sua semplicità. All'inizio di
ogni mossa gli si dice anche *quanto punteggio si vuole ancora totalizzare da
qui alla fine*. In addestramento questo numero è noto (basta sommare le
ricompense future). Al momento di giocare davvero, gli si mette in bocca un
obiettivo ambizioso («da qui voglio fare cento punti») e il modello, per
coerenza con tutte le partite viste, produce le mosse che *tipicamente portano
a cento punti*. Non calcola alcun valore: racconta la partita che vorresti, e
la recita.

`````

`````{tab} Superiore

Il Decision Transformer riordina la traiettoria come una sequenza di token

$$
\big(\hat{R}_1,\, s_1,\, a_1,\ \hat{R}_2,\, s_2,\, a_2,\ \dots,\ \hat{R}_T,\, s_T,\, a_T\big),
\qquad
\hat{R}_t = \sum_{t'=t}^{T} r_{t'},
$$

dove $\hat{R}_t$ è il **return-to-go**, la somma delle ricompense da $t$ in
poi. Un Transformer causale in stile GPT (la stessa architettura ad
auto-attenzione mascherata descritta nel capitolo sui Transformer) predice in
modo autoregressivo l'azione $a_t$ condizionando sui token precedenti, cioè su
return desiderato, stati e azioni fino a $s_t$. L'addestramento è puramente
**supervisionato**: minimizza l'errore (cross-entropy per azioni discrete, MSE
per continue) tra l'azione predetta e quella nel dataset. Non compaiono né
equazione di Bellman, né bootstrapping, né operatore $\max$, e quindi neppure
la sovrastima delle azioni OOD che affligge il Q-learning offline.

In fase di controllo si fissa un return-to-go iniziale $\hat{R}_1$ desiderato, si
osserva lo stato, si genera l'azione; a ogni passo si decrementa il return-to-go
della ricompensa incassata e si prosegue. È il ponte esplicito tra reinforcement
learning e *sequence modeling*: condizionare sul risultato voluto, anziché
inseguire un valore stimato.

`````

Il Decision Transformer non è sempre il migliore: su dati molto sub-ottimali, in
cui nessuna traiettoria vista raggiunge buoni ritorni, non può inventare
strategie mai osservate, mentre CQL e IQL riescono talvolta a «ricucire» pezzi di
traiettorie diverse in una migliore. Ma ha aperto una linea feconda, mostrando
che una parte del RL può essere riformulata come apprendimento supervisionato di
sequenze.

## Un filo che torna: le preferenze dell'RLHF

Questa prospettiva illumina qualcosa che abbiamo già incontrato. Nella sezione
sui metodi a gradiente di policy abbiamo visto l'RLHF, con cui si allineano i
modelli linguistici {cite}`ouyang2022training`: valutatori umani confrontano le
risposte del modello, e le loro preferenze addestrano un modello di ricompensa.

Ma quelle preferenze sono, a tutti gli effetti, un **dataset fisso**: nessuno
torna dagli annotatori a chiedere un giudizio su ogni nuova risposta generata
durante l'ottimizzazione. Non stupisce allora che l'RLHF erediti le stesse
tensioni (la policy tende ad allontanarsi dalla distribuzione dei dati e a
«sfruttare» il modello di ricompensa dove questo è poco vincolato) e le
contenga con gli stessi strumenti: un termine di penalità che tiene la nuova
policy *vicina* a quella di partenza, cugino diretto della prudenza di BCQ e
CQL. Imparare da dati fissi, dalla terapia intensiva agli assistenti
conversazionali, pone sempre la stessa domanda: quanto possiamo fidarci di ciò
che non abbiamo mai visto?

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Il **RL offline** impara da un archivio chiuso di esperienze già accadute,
  come i quaderni di ricette della nonna: qualcun altro ha agito e ha lasciato
  scritto com'è andata. Nessun assaggio nuovo, nessuna prova sul campo.
- Il metodo classico, applicato così com'è, fallisce quasi sempre. L'agente
  cerca sempre il voto più alto, e i voti più alti finiscono per capitare
  proprio sulle mosse mai provate, quelle su cui il modello può solo tirare a
  indovinare (il risotto al peperoncino da dieci e lode). Nessuna prova può
  smentire quel voto gonfiato, che anzi contagia le stime vicine.
- **BCQ** costruisce un recinto: valuta solo le mosse plausibili secondo
  l'archivio, generate imitando chi i dati li ha raccolti. **CQL** non vieta
  nulla, insegna prudenza: abbassa i voti di ciò che non è mai stato provato e
  alza quelli di ciò che è documentato, così l'ignoto non batte mai sulla
  carta il conosciuto. **IQL** toglie proprio l'occasione di sbagliare: chiede
  solo quanto hanno reso, in situazioni come questa, le mosse migliori fra
  quelle davvero fatte, e non nomina mai un'azione fuori dal diario.
- Il **Decision Transformer** cambia domanda: tratta la partita come una frase
  da completare e si allena sui diari a predire la mossa successiva. Gli si
  dice anche quanto punteggio si vuole ancora totalizzare, e lui produce le
  mosse che di solito portano lì. Nessun voto da stimare, quindi nessun voto
  gonfiato.
- Anche le preferenze umane con cui si allineano i modelli linguistici sono un
  archivio chiuso: stesso problema, e stesso rimedio, cioè restare vicini a
  ciò che l'archivio contiene davvero.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Il **RL offline** (batch RL) impara da un dataset fisso di transizioni
  $\mathcal{D}=\{(s,a,r,s')\}$ raccolto da una policy comportamentale
  $\pi_\beta$, **senza** alcuna nuova interazione con l'ambiente.
- Il Q-learning off-policy naive fallisce per **distributional shift**:
  l'operatore $\max$ sovrastima le azioni fuori distribuzione (errore di
  estrapolazione), la policy le insegue e (senza prove che smascherino
  l'errore) la sovrastima si propaga {cite}`fujimoto2019off`.
- **BCQ** vincola le azioni valutate a quelle plausibili secondo un modello
  generativo del dataset; **CQL** aggiunge un termine conservativo che abbassa i
  $Q$ delle azioni OOD e alza quelli nel dataset, ottenendo un limite inferiore
  del valore {cite}`kumar2020conservative`; **IQL** stima $V$ per regressione
  expectile e non interroga **mai** azioni fuori dai dati
  {cite}`kostrikov2022offline`.
- Il **Decision Transformer** riformula l'RL come modellazione di sequenze:
  condiziona sul *return-to-go* desiderato e predice l'azione con un Transformer,
  in modo puramente supervisionato {cite}`chen2021decision`.
- I dati di preferenza dell'**RLHF** sono anch'essi un dataset fisso: stesso
  problema, stessi rimedi (restare vicini alla distribuzione dei dati
  {cite}`ouyang2022training`).
```
`````
