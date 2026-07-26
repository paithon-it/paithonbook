# Spiegazioni locali: LIME, SHAP e controfattuali

Nel 1953 un matematico della RAND Corporation, **Lloyd Shapley**, si pose una
domanda che con l'intelligenza artificiale non c'entrava nulla: se un gruppo di
persone collabora a un'impresa e ne ricava un guadagno, come si divide il
merito «in modo equo» tra chi ha partecipato? Alcuni contano di più, alcuni
solo in combinazione con altri; non basta guardare cosa fa ciascuno da solo.
Shapley diede una risposta con quattro proprietà così ragionevoli da risultare,
in un certo senso, l'unica possibile. Mezzo secolo dopo avrebbe vinto il premio
Nobel per l'economia — non per questa formula, per la verità, ma per un altro
filone dei suoi studi di teoria dei giochi, quello sugli abbinamenti stabili. Non
poteva immaginare che la sua idea sarebbe diventata, nel 2017, lo strumento più
usato al mondo per spiegare la singola decisione di una rete neurale.

Questa è la seconda tappa del capitolo, e cambia scala. La sezione precedente,
sull'importanza delle feature, rispondeva a una domanda **globale**: «su quali
colonne si regge il modello, *in media*?». Ma chi si vede negare un prestito non
chiede una media: chiede «perché *la mia* domanda?». È una domanda **locale** —
riguarda una predizione sola, $f(x_0)$, non l'intero comportamento del modello.
Qui vediamo tre modi di risponderle, tre attrezzi diventati lo standard di
fatto: **LIME**, **SHAP** e le spiegazioni **controfattuali**.

## Perché *questa* predizione

`````{tab} Elementare

Torniamo al modello che decide i prestiti, e immaginiamo che a Maria l'abbia
rifiutato. Sapere che «in generale il reddito conta molto» non le serve: è una
statistica sul modello, non una risposta sul suo caso. Maria vuole sapere cosa,
*nella sua pratica*, ha spinto verso il no — il reddito, i debiti in corso, un
ritardo di pagamento di due anni fa? — e magari quanto ciascuna cosa ha pesato.

È la differenza tra chiedere «come giudica in media questo professore» e «perché
ha dato a *me* questo voto». La prima è una spiegazione **globale**, sul modello
intero; la seconda è **locale**, su una singola risposta. Un modello può essere
troppo intricato per capirlo tutto in un colpo, eppure restare semplice
*attorno a un punto*: come una strada di montagna piena di curve che, guardata
da vicino su pochi metri, sembra dritta. È su questa idea — «complicato ovunque,
semplice qui accanto» — che si reggono i metodi di questa sezione.

`````

`````{tab} Superiore

Formalmente, una spiegazione locale riguarda il valore del modello $f$ in un
intorno del punto $x_0$, non la funzione $f$ sull'intero dominio. La
motivazione è geometrica: la superficie decisionale di un modello complesso può
essere globalmente inintelligibile ma **localmente regolare**, cioè bene
approssimabile da un modello semplice in un intorno abbastanza piccolo di
$x_0$ — l'analogo di linearizzare una funzione differenziabile con il suo piano
tangente. Ne discende il criterio di qualità già introdotto nella prima
sezione, la **fedeltà locale**: una spiegazione è buona se il surrogato
interpretabile concorda con $f$ sui punti campionati vicino a $x_0$, senza
alcuna pretesa di valere lontano da lì. Cambia anche l'oggetto restituito: non
più una classifica di feature valida ovunque, ma un vettore di **attribuzioni**
$\phi_j$ specifiche di $x_0$, una per feature, che dicono quanto ciascuna ha
spinto *questa* predizione sopra o sotto un riferimento.

`````

## LIME: un modello semplice, ritagliato attorno al punto

Il primo attrezzo è **LIME** — *Local Interpretable Model-agnostic
Explanations* — proposto nel 2016 da Marco Túlio Ribeiro, Sameer Singh e Carlos
Guestrin {cite}`ribeiro2016why`, gli stessi dell'esperimento sugli husky e i
lupi visto in apertura di capitolo. L'idea è disarmante nella sua semplicità:
se la scatola nera è troppo complicata da capire tutta, costruiamone una copia
*facile* che le somigli **solo qui vicino**, attorno al punto da spiegare.

`````{tab} Elementare

Immagina di voler capire come il modello decide sul caso di Maria. LIME fa così:
prende la pratica di Maria e ne genera tante **varianti** leggermente modificate
— un po' più di reddito, un debito in meno, un'età diversa — e per ciascuna
chiede alla scatola nera cosa risponderebbe. Ottiene così una nuvola di
esempi-fantasma con le relative risposte del modello, tutti raccolti *intorno*
al caso di Maria.

A questo punto adatta a quella nuvola un modello **semplice e leggibile** — una
retta pesata, come la regressione lineare del capitolo di machine learning —
dando più importanza ai fantasmi più vicini alla pratica vera di Maria e meno a
quelli lontani. I coefficienti di quella retta *sono* la spiegazione: «il
reddito basso ha spinto verso il no di tanto, i debiti di tanto, l'anzianità di
lavoro ha spinto un po' verso il sì». È come tracciare la tangente a una curva:
non descrive tutta la strada, ma dice benissimo la pendenza nel punto in cui ti
trovi.

`````

`````{tab} Superiore

LIME cerca un surrogato $g$ in una classe interpretabile $G$ (tipicamente
modelli lineari sparsi) che minimizzi

$$
\xi(x_0) = \operatorname*{arg\,min}_{g \in G}\;
   \mathcal{L}\big(f, g, \pi_{x_0}\big) + \Omega(g),
$$

dove $\mathcal{L}$ è l'infedeltà di $g$ rispetto a $f$ **pesata dalla
prossimità** $\pi_{x_0}$, e $\Omega(g)$ penalizza la complessità di $g$ (per
esempio il numero di feature non nulle, per una spiegazione corta). In pratica
si campionano punti perturbati $z$ attorno a $x_0$, si valuta $f(z)$ (la sola
cosa che serve del modello: LIME è **model-agnostic**), si pesa ciascun campione
con un kernel esponenziale
$\pi_{x_0}(z) = \exp\!\big(-D(x_0, z)^2 / \sigma^2\big)$ che decade con la
distanza $D$, e si adatta una regressione lineare pesata. I coefficienti
appresi sono le attribuzioni.

Due limiti vanno dichiarati. Primo, l'**instabilità**: campionamento casuale e
scelta del kernel rendono la spiegazione sensibile ai dettagli — rieseguire LIME
sullo stesso punto può dare coefficienti diversi. Secondo, la **larghezza del
vicinato** $\sigma$ è un iperparametro senza una regola universale: un intorno
troppo ampio linearizza una zona in cui $f$ non è affatto lineare (bassa
fedeltà), uno troppo stretto lascia troppo pochi campioni informativi. La
spiegazione dipende quindi da scelte che l'utente raramente controlla — un
motivo per affiancarle un metodo dai fondamenti più solidi.

`````

## I valori di Shapley: dividere il merito in modo equo

Quel metodo esiste, ed è la formula di Shapley del 1953. Il salto concettuale è
vedere una predizione come un **gioco cooperativo**: le feature sono i
giocatori, e il «guadagno» da spartire è di quanto la predizione si scosta dal
valore medio del modello. La domanda «quanto ha contribuito il reddito a questo
rifiuto?» diventa la vecchia domanda di Shapley: «quanto merito spetta a questo
giocatore?».

`````{tab} Elementare

Il punto delicato è che i giocatori non agiscono da soli: contano anche le
**combinazioni**. Prendiamo due feature, il *reddito* e la *storia creditizia*,
e chiediamoci quanto vale ciascuna. L'idea di Shapley è: proviamo tutti gli
ordini in cui le feature possono «entrare in campo» e, per ognuna, misuriamo
quanto aggiunge nel momento in cui entra. Poi facciamo la media.

Diciamo che il modello, senza sapere niente del cliente, parte da un punteggio
base di **10**. Se conosce solo il reddito, sale a **30**; se conosce solo la
storia, sale a **20**; se le conosce entrambe, arriva a **50**. Con due feature
gli ordini possibili sono due — prima il reddito, oppure prima la storia:

- *Prima il reddito*: entra e porta il punteggio da 10 a 30, quindi aggiunge
  **20**. Poi entra la storia e lo porta da 30 a 50, aggiunge **20**.
- *Prima la storia*: entra e porta da 10 a 20, aggiunge **10**. Poi entra il
  reddito e porta da 20 a 50, aggiunge **30**.

Il merito del reddito è la media di quanto aggiunge nei due ordini:
$(20 + 30)/2 = 25$. Quello della storia: $(20 + 10)/2 = 15$. E il conto torna:
$25 + 15 = 40$, esattamente quanto separa il punteggio finale (50) da quello
base (10). Tutto il «guadagno» è stato ripartito, senza avanzi: nessun merito
inventato, nessuno perso per strada.

`````

`````{tab} Superiore

Sia $N = \{1, \dots, n\}$ l'insieme delle feature e $v(S)$ il valore della
**coalizione** $S \subseteq N$: la predizione attesa quando si conoscono solo le
feature in $S$ e si marginalizza sulle altre, con $v(\varnothing) = \mathbb{E}[f]$
il valore base e $v(N) = f(x_0)$. Il valore di Shapley della feature $i$ è la
media dei suoi **contributi marginali** su tutti gli ordini di ingresso:

$$
\phi_i = \sum_{S \subseteq N \setminus \{i\}}
   \frac{|S|!\,\big(n - |S| - 1\big)!}{n!}\,
   \big[\, v(S \cup \{i\}) - v(S) \,\big],
$$

dove $v(S \cup \{i\}) - v(S)$ è quanto aggiunge $i$ unendosi alla coalizione
$S$, e il coefficiente combinatorio conta la frazione di permutazioni in cui $i$
entra proprio dopo l'insieme $S$. Questa è l'**unica** attribuzione che
soddisfa quattro assiomi:

- **Efficienza**: $\sum_{i} \phi_i = v(N) - v(\varnothing) = f(x_0) - \mathbb{E}[f]$.
  I contributi sommano esattamente allo scarto della predizione dal valore base:
  niente si crea, niente si perde.
- **Simmetria**: se due feature danno lo stesso contributo a ogni coalizione
  ($v(S \cup \{i\}) = v(S \cup \{j\})$ per ogni $S$), allora $\phi_i = \phi_j$.
- **Giocatore nullo** (*dummy*): una feature che non cambia mai il valore
  ($v(S \cup \{i\}) = v(S)$ per ogni $S$) riceve $\phi_i = 0$.
- **Additività**: i valori di Shapley di una somma di modelli sono la somma dei
  valori — è la proprietà che rende trattabili gli ensemble.

Sull'esempio dei due giocatori, con $v(\varnothing)=10$, $v(\{1\})=30$,
$v(\{2\})=20$, $v(\{1,2\})=50$, la formula dà $\phi_1 = 25$ e $\phi_2 = 15$, in
accordo con la media sugli ordini, e $\phi_1 + \phi_2 = 40 = v(N) - v(\varnothing)$
verifica l'efficienza. Il costo è la maledizione combinatoria: la somma è su
$2^{\,n-1}$ coalizioni, impraticabile oltre poche decine di feature. È il
problema che SHAP risolve.

`````

## SHAP: i valori di Shapley, resi praticabili

Calcolare i valori di Shapley esatti richiede di provare tutte le coalizioni:
con 30 feature sono già oltre un miliardo. Nel 2017 Scott Lundberg e Su-In Lee
{cite}`lundberg2017unified` hanno mostrato come stimarli in modo efficiente,
unificando sotto un'unica teoria — **SHAP**, *SHapley Additive exPlanations* —
metodi fino ad allora scollegati (LIME incluso, come caso particolare). Da qui
il metodo è diventato lo standard pratico dell'interpretabilità post-hoc.

`````{tab} Elementare

SHAP non cambia la definizione: restituisce ancora i contributi «equi» di
Shapley. Cambia il *come* li ottiene, con due scorciatoie a seconda del modello.
Per una scatola nera qualsiasi usa **KernelSHAP**: campiona con astuzia solo
alcune coalizioni invece di tutte, e ne ricostruisce i valori con una
regressione pesata — cugino di LIME, ma con i pesi «giusti» che garantiscono di
puntare ai valori di Shapley. Per i modelli ad alberi — foreste casuali,
gradient boosting, i protagonisti del capitolo sugli alberi ed ensemble — usa
**TreeSHAP**, che sfrutta la struttura ad albero per calcolare i valori
**esatti** in tempo ragionevole, senza campionare nulla.

Il risultato più leggibile è il grafico **a cascata** (*waterfall*) della
{numref}`fig-shap-contributi`: si parte dal valore base — la predizione media
del modello — e si impilano i contributi, quelli che spingono verso l'alto in
terracotta e quelli che spingono verso il basso in teal, fino ad arrivare alla
predizione di *questo* cliente. Una singola immagine racconta, voce per voce,
perché il modello ha deciso così.

`````

`````{tab} Superiore

KernelSHAP riformula il calcolo come una regressione lineare pesata sulle
coalizioni: campionando sottoinsiemi $S$ e pesandoli con il **kernel di
Shapley**
$\pi(S) = \frac{n-1}{\binom{n}{|S|}\,|S|\,(n - |S|)}$,
la soluzione ai minimi quadrati converge ai valori di Shapley — è la scelta di
pesi che distingue SHAP da LIME, i cui pesi euristici non hanno questa garanzia.
TreeSHAP — introdotto in un lavoro successivo degli stessi autori — calcola
invece i valori **esatti** per i modelli ad albero in $O(T L D^2)$ ($T$ alberi,
$L$ foglie, $D$ profondità), propagando lungo l'albero le popolazioni delle
coalizioni.

Il vantaggio teorico è la **consistenza**: se si modifica il modello così che
una feature contribuisca di più in ogni coalizione, il suo valore SHAP non può
diminuire — una monotonìa che l'importanza da impurità della prima sezione *non*
garantisce. Oltre al waterfall (una predizione), i grafici tipici sono il *force
plot*, che dispone gli stessi contributi come forze contrapposte lungo una
retta, e soprattutto il **beeswarm**: impilando i valori SHAP di migliaia di
istanze, una riga per feature, si ricostruisce una vista **globale** — quali
feature contano e in che direzione — a partire da tante spiegazioni locali. È il
ponte tra il locale e il globale che rende SHAP così usato.

`````

```{figure} ../figures/shap-contributi.svg
:name: fig-shap-contributi
:alt: "Grafico a cascata di una predizione. A sinistra una linea verticale tratteggiata al valore base E[f(x)] uguale a 0,20. Quattro barre orizzontali si impilano: 'storia = buona' aggiunge +0,18 verso destra in terracotta, 'reddito = alto' aggiunge +0,10, 'eta = 40' aggiunge +0,04, mentre 'debito = alto' sottrae 0,08 tornando verso sinistra in teal. Si arriva all'output f(x) uguale a 0,44 segnato da una seconda linea verticale a destra."
:width: 85%

Un grafico a cascata (*waterfall*) SHAP scompone una singola predizione. Si
parte dal valore base $\mathbb{E}[f(x)]$ — la predizione media del modello — e
si sommano i contributi di ogni feature: in terracotta quelli che alzano la
predizione, in teal quelli che la abbassano. La somma porta esattamente
all'output $f(x)$ per questo cliente (assioma di efficienza).
```

## Controfattuali: cosa sarebbe dovuto cambiare

LIME e SHAP rispondono a «perché questa decisione?». Ma a chi si è visto negare
un prestito interessa spesso un'altra domanda, più pratica: «cosa devo cambiare
perché la prossima volta sia un sì?». È la spiegazione **controfattuale**,
formalizzata nel 2017 da Sandra Wachter, Brent Mittelstadt e Chris Russell
{cite}`wachter2017counterfactual` proprio pensando al «diritto alla spiegazione»
del GDPR europeo, discusso nell'introduzione del capitolo.

`````{tab} Elementare

Un controfattuale è un'affermazione del tipo: «se il tuo reddito fosse stato
30 000 € invece di 24 000, il prestito sarebbe stato approvato». Non ti dice
come funziona il modello dentro; ti dice **la modifica più piccola** che avrebbe
ribaltato la decisione. È **azionabile**: indica una via d'uscita concreta,
non una diagnosi astratta.

La qualità di un buon controfattuale sta in due cose. Primo, dev'essere
**vicino** alla tua situazione reale: «guadagna 200 000 € in più» tecnicamente
ribalta la decisione ma non serve a nessuno; «riduci di una rata i tuoi debiti»
è molto più utile. Secondo, dev'essere **plausibile e realizzabile**: suggerire
di cambiare l'età non ha senso, perché non è una leva su cui puoi agire. Il buon
controfattuale è il consiglio minimo, concreto e onesto che ti mette dalla parte
giusta della decisione.

`````

`````{tab} Superiore

Wachter e colleghi cercano un punto $x'$ che ottenga l'esito desiderato $y'$
restando il più vicino possibile all'istanza originale $x_0$, minimizzando

$$
\operatorname*{arg\,min}_{x'}\; \big(f(x') - y'\big)^2
   + \lambda\, d(x', x_0),
$$

dove il primo termine spinge la predizione verso il valore-bersaglio $y'$ (la
soglia di approvazione), $d$ è una distanza che misura quanto $x'$ si discosta
da $x_0$ — spesso una $L_1$ per ottenere modifiche **sparse**, che toccano poche
feature — e $\lambda$ bilancia i due obiettivi. Estensioni successive aggiungono
vincoli di **plausibilità** (restare sul supporto dei dati) e di
**azionabilità** (non modificare feature immutabili come l'età o l'etnia).

C'è un parallelo tecnico che vale la pena rendere esplicito. Cercare la
perturbazione minima di $x_0$ che cambia l'uscita del modello è, formalmente, lo
stesso problema degli **esempi avversari** — le impercettibili modifiche
d'input che ingannano una rete, studiate da Goodfellow, Shlens e Szegedy
{cite}`goodfellow2015explaining` e riprese nel capitolo sull'AI responsabile. La
matematica è la medesima, l'intento opposto: un esempio avversario nasconde la
perturbazione per **ingannare** il modello; un controfattuale la esibisce per
**spiegarlo** e offrire una via d'azione. Lo stesso strumento può violare o
servire l'interesse di chi subisce una decisione, a seconda di come lo si usa.

`````

## In pratica: i valori di Shapley con NumPy

La libreria `shap` calcola tutto questo in poche righe, ma per capire davvero
cosa c'è sotto conviene ricostruire i valori di Shapley **a mano**, con la
definizione per forza bruta: la media dei contributi marginali su tutti gli
ordini. Il codice qui sotto usa solo NumPy. Il modello giocattolo ha
un'**interazione** — la terza feature conta solo insieme alla prima — proprio per
vedere come Shapley se la cava con i contributi non additivi.

```python
import itertools
from math import factorial
import numpy as np

# Un modello giocattolo con un'interazione: la feature 2 "conta" solo con la 0
def f(x):
    return x[0] + 2.0 * x[1] + x[0] * x[2]

# istanza da spiegare e riferimento (baseline) su cui "spegnere" le feature assenti
x = np.array([1.0, 1.0, 1.0])
r = np.array([0.0, 0.0, 0.0])
n = len(x)

# valore della coalizione S: le feature in S prendono il valore di x, le altre di r
def v(S):
    z = r.copy()
    for i in S:
        z[i] = x[i]
    return f(z)

# valori di Shapley per forza bruta: media dei contributi marginali su TUTTI gli ordini
phi = np.zeros(n)
for perm in itertools.permutations(range(n)):
    S = []
    for i in perm:
        prima = v(S)            # coalizione prima di aggiungere i
        S = S + [i]
        dopo = v(S)             # coalizione dopo aver aggiunto i
        phi[i] += dopo - prima  # contributo marginale di i in questo ordine
phi /= factorial(n)             # media sugli n! ordini

print("valori di Shapley:", np.round(phi, 3))
print("somma dei phi:     ", round(float(phi.sum()), 3))
print("f(x) - f(base):    ", round(float(f(x) - f(r)), 3))  # assioma di efficienza
```

L'output:

```text
valori di Shapley: [1.5 2.  0.5]
somma dei phi:      4.0
f(x) - f(base):     4.0
```

Il conto racconta esattamente cosa fa Shapley. La feature 1, con coefficiente
$2$ e nessuna interazione, prende $2{,}0$: tutto suo. Il termine d'interazione
$x_0 x_2$, che vale $1$, viene **spartito equamente** tra le due feature che lo
producono: mezzo punto alla feature 0 (che così arriva a $1{,}5$: il suo $1$ più
$0{,}5$) e mezzo alla feature 2 (che da sola non farebbe nulla, e infatti prende
solo $0{,}5$). È la simmetria all'opera: nessuna delle due può rivendicare
l'interazione più dell'altra. E la somma $1{,}5 + 2{,}0 + 0{,}5 = 4{,}0$
coincide con $f(x) - f(\text{base})$: l'**efficienza** è verificata numericamente.
Nella pratica non si enumerano tutti gli ordini — sono $n!$ — ma si campionano,
o si usa TreeSHAP se il modello è un albero; la definizione, però, è questa.

```{admonition} Da ricordare
:class: important
- Una spiegazione **locale** riguarda *una* predizione $f(x_0)$, non l'intero
  modello: risponde a «perché *questo* caso?» dove l'importanza globale della
  sezione precedente rispondeva a «cosa conta in media?».
- **LIME** {cite}`ribeiro2016why` approssima il modello con un **surrogato
  lineare** adattato a punti perturbati attorno a $x_0$ e pesati per prossimità;
  è model-agnostic ma **instabile** e sensibile alla larghezza del vicinato.
- I **valori di Shapley** (teoria dei giochi, 1953) ripartiscono lo scarto
  $f(x_0) - \mathbb{E}[f]$ tra le feature come media dei **contributi marginali**
  su tutti gli ordini; sono l'unica attribuzione che soddisfa **efficienza,
  simmetria, giocatore nullo e additività**.
- **SHAP** {cite}`lundberg2017unified` li stima in modo efficiente —
  **KernelSHAP** (agnostico) e **TreeSHAP** (esatto per gli alberi) — con la
  proprietà di **consistenza**; si legge col grafico a cascata (*waterfall*),
  il *force plot* e il *beeswarm*.
- I **controfattuali** {cite}`wachter2017counterfactual` indicano la modifica
  minima e azionabile che ribalterebbe la decisione; sono lo stesso problema
  matematico degli **esempi avversari** {cite}`goodfellow2015explaining`, con
  intento opposto: spiegare invece di ingannare.
```
