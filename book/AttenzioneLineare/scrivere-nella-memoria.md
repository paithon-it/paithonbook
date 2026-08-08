# Scrivere meglio nella memoria: gate e delta rule

Nella sezione precedente abbiamo scoperto che l'attenzione lineare è, sotto
mentite spoglie, una rete ricorrente: al posto della cache di chiavi e valori
che cresce a ogni parola, un'unica memoria $S$ di dimensione fissa (una
matrice $d \times d$) che accumula ogni nuova coppia con un semplice prodotto
esterno,

$$
S_t = S_{t-1} + v_t\, k_t^\top ,
$$

e che si legge proiettandola sulla query, $o_t = S_t\, q_t$. Qui $k_t$, $v_t$ e
$q_t$ sono le chiavi, i valori e le query che conosciamo dall'attenzione dei
Transformer, e $S_t$ è la memoria dopo aver letto i primi $t$ token. Come
annunciato nella sezione precedente, la formula è scritta nella forma più
snella usata dalle architetture che stiamo per incontrare. Immaginate
$S$ come un registro che funziona da rubrica: ogni riga della somma è una nuova
voce «chiave → valore» scritta sopra le precedenti.

Il registro, però, ha due difetti che si vedono a occhio nudo. **Non
dimentica**: ogni voce resta scritta per sempre, e con abbastanza token la
pagina si satura di tracce sovrapposte finché non si legge più nulla di
preciso. E **non corregge**: se una voce era sbagliata, l'unico modo per
rimediare è scriverne un'altra sopra che la contraddica; la vecchia resta lì a
disturbare. Da qui due idee semplici e complementari per scrivere *meglio*
nella memoria: **dimenticare** (i gate) e **correggere** (la delta rule). Sono
le due mosse da cui nasce, come vedremo, l'intera famiglia delle ricorrenze
lineari moderne.

## Dimenticare: i gate

La prima idea è lasciare che le voci vecchie sbiadiscano da sole. Invece di
tramandare la memoria intatta, la si moltiplica a ogni passo per un fattore di
decadimento minore di uno: ciò che è stato scritto tempo fa pesa sempre meno,
finché svanisce. È il **gate di dimenticanza**: lo stesso *forget gate* che
Gers, Schmidhuber e Cummins aggiunsero alle LSTM nel 2000
{cite}`gers2000learning`, riportato qui nella sua forma più essenziale.

`````{tab} Elementare

Pensate al registro come a una lavagna su cui l'inchiostro sbiadisce. A ogni
passo tutte le voci si affievoliscono un po': quelle appena scritte sono nitide,
quelle vecchie quasi invisibili. Se il fattore di sbiadimento è $0{,}9$, dopo
dieci passi una voce vale $0{,}9^{10} \approx 0{,}35$ di quanto valeva: circa
un terzo. La lavagna così non si satura mai, perché fa spazio da sola buttando via
il passato lontano.

C'è però una raffinatezza. Sbiadire *tutto allo stesso ritmo* è grossolano:
magari in una parte della lavagna c'è un dettaglio che servirà ancora fra mille
parole, in un'altra solo appunti usa-e-getta. Meglio poter sbiadire **una zona
della lavagna alla volta**: tenere nitida quella con le cose che contano
(fattore $0{,}99$, quasi non svanisce) e cancellare in fretta quella degli
appunti di servizio (fattore $0{,}5$, dimezza a ogni passo). È la differenza
tra abbassare le luci di tutta la stanza e regolare ogni lampada
singolarmente.

`````

`````{tab} Superiore

Un promemoria di notazione, valido da qui a fine capitolo: la feature map
$\phi$ è posta all'identità e il normalizzatore $z_t$ non compare, secondo la
seconda delle due scuole distinte nella sezione precedente.

La forma più semplice è il **decadimento scalare**: un unico numero
$\alpha_t \in (0,1)$ moltiplica l'intera memoria,

$$
S_t = \alpha_t\, S_{t-1} + v_t\, k_t^\top ,
$$

dove $\alpha_t$ è il gate di dimenticanza al passo $t$ e $v_t k_t^\top$ è la
nuova voce scritta. È la ricorrenza di **RetNet** {cite}`sun2023retnet`, dove
$\alpha_t = \gamma$ è una costante fissata a priori (data-*indipendente*),
diversa per ciascuna testa così da coprire orizzonti temporali diversi; ed è
anche quella di **Mamba-2** {cite}`dao2024mamba2`, dove invece $\alpha_t$ è
prodotto dall'input, quindi data-*dipendente*. Srotolando la ricorrenza si vede
cosa fa il gate: il contributo scritto al passo $j$ arriva al passo $t$ pesato
per $\prod_{i=j+1}^{t}\alpha_i$, cioè decade in modo (quasi) esponenziale con la
distanza.

Uno scalare, però, applica la stessa dimenticanza a *tutte* le dimensioni della
memoria. La **Gated Linear Attention** (GLA) di Yang e colleghi, presentata
all'ICML 2024 {cite}`yang2024gla`, la rende molto più fine sostituendo lo
scalare con un **gate diagonale**:

$$
S_t = S_{t-1}\, \operatorname{Diag}(\alpha_t) + v_t\, k_t^\top ,
$$

dove ora $\alpha_t \in (0,1)^d$ è un *vettore* di gate, uno per canale di
chiave, e $\operatorname{Diag}(\alpha_t)$ è la matrice diagonale che ne fa i
coefficienti (moltiplica lo stato da destra, come la transizione della delta
rule che vedremo tra poco). Ogni colonna della memoria, cioè il canale della
chiave $i$, decade al proprio ritmo: $\alpha_{t,i}\to 1$ conserva il canale $i$
(nel limite si torna all'accumulo puro), $\alpha_{t,i}\to 0$ lo azzera. Il vettore è ricavato dall'input con una proiezione a **basso rango**
(un collo di bottiglia stretto, dimensione 16) seguita da una sigmoide, così da
generare $d$ gate distinti senza far esplodere il numero di parametri. La
gerarchia è chiara: scalare fisso (RetNet) $\to$ scalare data-dipendente
(Mamba-2) $\to$ diagonale data-dipendente (GLA), dal più grossolano al più
selettivo.

`````

Il gate risolve il primo difetto (la saturazione) ma non il secondo. Per
quanto si sbiadisca, ogni scrittura resta un'aggiunta cieca: nessuno controlla
se quella voce contraddice ciò che c'è già.

## Correggere: la delta rule

La seconda idea è più sottile e viene da lontano. Nel 2021 Schlag, Irie e
Schmidhuber {cite}`schlag2021linear` osservano che il Transformer lineare è, a
tutti gli effetti, un vecchio **fast weight programmer** degli anni Novanta di
Schmidhuber: una rete «lenta» (i pesi appresi) che a ogni passo *programma* la
memoria «veloce» $S$ scrivendoci sopra i prodotti esterni che si genera da sola.
E se è una rete che si riprogramma, allora conviene programmarla bene: prima di
scrivere, guardare cosa c'è già.

`````{tab} Elementare

Torniamo alla rubrica. L'accumulo puro è chi, ogni volta che scopre un numero
di telefono, aggiunge una riga nuova: anche se quel contatto era già in
rubrica, magari con un numero sbagliato. Le righe si accavallano e chi
consulta la rubrica trova una media confusa tra il numero vecchio e quello
nuovo.

La **delta rule** fa la cosa sensata: prima di scrivere, *cerca il contatto* e
legge il numero attualmente memorizzato. Poi scrive soltanto la
**correzione**: la differenza tra quello giusto e quello che c'era. Un esempio
con i numeri: alla chiave «Mario» la memoria oggi risponde $7$, ma il valore
giusto è $10$; l'errore è $10 - 7 = 3$. Con un «passo di correzione» pari a
$\beta = 0{,}5$ scrivo solo $0{,}5 \times 3 = 1{,}5$, e la memoria passa a
rispondere $8{,}5$: si avvicina alla verità senza cancellare tutto di colpo.
Il parametro $\beta$ dosa quanto dare retta all'errore: con $\beta = 1$
**sovrascrivo** del tutto (la memoria risponde $10$), con $\beta = 0$
**ignoro** e lascio $7$. È esattamente come si corregge un tiro: non riparti
da zero, aggiusti in proporzione a quanto hai sbagliato.

`````

`````{tab} Superiore

Al passo $t$, prima di scrivere, si interroga la memoria con la *chiave*
corrente e si ottiene il valore che essa già predice per quella chiave,

$$
\bar{v}_t = S_{t-1}\, k_t .
$$

Qui $\bar{v}_t$ è la «vecchia risposta»: ciò che la rubrica restituisce oggi
alla chiave $k_t$. La delta rule scrive allora soltanto l'**errore** $v_t -
\bar{v}_t$, scalato da un *learning-rate* $\beta_t \in (0,1)$ appreso
dinamicamente ($\beta_t = \sigma(w_\beta^\top x_t)$):

$$
S_t = S_{t-1} + \beta_t\,(v_t - \bar{v}_t)\, k_t^\top
    = S_{t-1} + \beta_t\,(v_t - S_{t-1} k_t)\, k_t^\top .
$$

È la regola di **Widrow–Hoff** (o LMS), il mattone dell'apprendimento
adattivo. Raccogliendo i termini si ottiene una forma compatta ed elegante, la
**Householder generalizzata**:

$$
S_t = S_{t-1}\,(I - \beta_t\, k_t k_t^\top) + \beta_t\, v_t\, k_t^\top ,
$$

dove $I$ è l'identità e $k_t k_t^\top$ è il prodotto esterno della chiave con se
stessa. Il fattore $(I - \beta_t k_t k_t^\top)$ agisce come una transizione di
stato che **cancella la vecchia traccia lungo la direzione $k_t$** appena prima
di scriverci quella nuova: se $\beta_t = 1$ sovrascrive del tutto quella chiave,
se $\beta_t = 0$ lascia la memoria intatta. Perché il fattore sia ben
condizionato (autovalori in $[0,1]$) le chiavi vanno **normalizzate in norma
$L_2$**, così che $k_t^\top k_t = 1$.

Una precisazione su quanto anticipato nella sezione precedente: sono proprio
Schlag e colleghi a compiere il primo passo della scuola «senza $z_t$».
Mostrano che il **normalizzatore** di Katharopoulos (la somma delle chiavi
trasformate) può crescere senza controllo e diventare instabile, e lo
**scartano**, normalizzando invece per somma le chiavi e le query trasformate.
La famiglia di ricorrenze che seguiamo qui (GLA, DeltaNet e le loro parenti)
porta la rinuncia a compimento: normalizza le chiavi in $L_2$ e applica una
LayerNorm all'uscita, ottenendo lo stesso effetto stabilizzante senza il
termine $z_t$.

`````

Resta un problema pratico, ed è serio. Il gate scalare o diagonale è una
semplice **moltiplicazione per uno stato precedente**: srotolato lungo la
sequenza diventa una somma cumulativa, e le somme cumulative si calcolano in
parallelo. La delta rule no: il termine $S_{t-1} k_t$ dipende dall'intera
memoria appena *prima* del passo $t$, quindi ogni aggiornamento aspetta il
precedente. Per anni questo l'ha confinata a un giocattolo teorico, non
addestrabile su sequenze lunghe. Nel 2024 **DeltaNet**, di Yang e colleghi
(NeurIPS 2024) {cite}`yang2024deltanet`, l'ha sbloccata: un algoritmo
**chunk-parallel** che spezza la sequenza in blocchi e, tramite una
rappresentazione a matrici di rango basso (la cosiddetta forma *WY*), calcola
il blocco in parallelo passando allo stato solo un riassunto compatto. La
delta rule diventa così addestrabile alla scala dei modelli linguistici. E si
vede dove ci si aspetta: quando il compito è ritrovare il valore giusto legato
a una chiave incontrata prima (il numero di telefono del contatto giusto, per
restare alla rubrica), DeltaNet fa meglio delle altre memorie a foglio unico di
questo capitolo e si avvicina all'attenzione dei Transformer, che su quel
terreno resta il metro di paragone perché non butta via niente.

## Unire oblio e correzione: Gated DeltaNet

A questo punto abbiamo due mosse che risolvono difetti diversi, e la domanda
si fa naturale: perché scegliere? Il gate **svuota in fretta** la memoria, ma
in modo **uniforme e indiscriminato**: non sa *cosa* sta buttando via. La
delta rule fa **correzioni mirate** su singole chiavi, ma da sola **non
svuota**: tende a lasciare la memoria piena di tracce, sia pure aggiustate.
Yang, del MIT, con Kautz e Hatamizadeh di NVIDIA (2024)
{cite}`yang2024gateddelta`, osservano che sono **complementari** e li mettono
insieme in **Gated DeltaNet**.

La ricorrenza combina i due fattori di transizione:

$$
S_t = S_{t-1}\big[\, \alpha_t\,(I - \beta_t\, k_t k_t^\top)\,\big]
      + \beta_t\, v_t\, k_t^\top ,
$$

dove i due parametri hanno ruoli distinti e leggibili a colpo d'occhio:

- $\alpha_t \in (0,1)$ è un **gate scalare** (nella parametrizzazione di
  Mamba-2): il *decadimento globale*, che alleggerisce l'intera memoria a ogni
  passo;
- $\beta_t \in (0,1)$ è la **forza di scrittura** della delta rule: quanto
  correggere la chiave corrente, come nella sezione precedente.

Con $\alpha_t \to 1$ si ritrova la pura delta rule (nessun oblio, sole
correzioni); con $\beta_t \to 0$ si ritrova il puro gate scalare (nessuna
correzione, solo oblio). Gated DeltaNet vive nel mezzo: dimentica in fretta ciò
che non serve più *e* aggiusta con precisione ciò che tiene. La
parallelizzazione lungo la sequenza si ottiene estendendo la stessa
rappresentazione WY di DeltaNet, così che anche questa forma più ricca resti
addestrabile su contesti lunghi.

## Tutto è regressione online

Fermiamoci a guardare cosa abbiamo costruito, perché qui il capitolo trova il
suo climax concettuale. Accumulo, gate, delta rule, e la loro combinazione:
sembrano trucchi diversi, ma sono lo **stesso gesto** visto da angolazioni
diverse. E quel gesto ha un nome che conosciamo bene dal capitolo sul machine
learning: è un passo di **minimi quadrati**.

`````{tab} Elementare

Ricordate la retta di best fit? Data una nuvola di punti, cercavamo la retta
che minimizza l'errore quadratico: la somma dei quadrati degli scarti tra
valori veri e previsti. Facevamo tutto in una volta, con l'intero dataset
sotto gli occhi.

Una memoria associativa fa la stessa cosa, ma **un dato alla volta, mentre
scorre**. Il suo compito è imparare a mappare ogni chiave nel suo valore: dato
$k_t$, restituire $v_t$. A ogni token arriva una nuova coppia
(chiave, valore) e la memoria fa **un piccolo passo** per predirla meglio, senza
poter rileggere il passato. È la versione «in tempo reale» della retta di best
fit: non risolvi il problema in blocco, lo aggiusti in continuazione a ogni
esempio che passa. Ecco perché la delta rule assomigliava a correggere un tiro:
è letteralmente un passo di discesa del gradiente.

`````

`````{tab} Superiore

Fissiamo, a ogni passo, l'obiettivo di **regressione online**

$$
\mathcal{L}_t(S) = \tfrac{1}{2}\,\lVert S\,k_t - v_t \rVert^2 ,
$$

cioè «mappare la chiave $k_t$ nel valore $v_t$», con $S$ la memoria da
addestrare. Il suo gradiente rispetto a $S$ è

$$
\nabla_S\,\mathcal{L}_t = (S\,k_t - v_t)\, k_t^\top ,
$$

e un singolo passo di discesa del gradiente con learning-rate $\beta_t$,
partendo da $S_{t-1}$, dà

$$
S_t = S_{t-1} - \beta_t\,(S_{t-1} k_t - v_t)\, k_t^\top
    = S_{t-1}\,(I - \beta_t\, k_t k_t^\top) + \beta_t\, v_t\, k_t^\top .
$$

È **esattamente** la delta rule di DeltaNet: il passo di gradiente *esatto*
su $\mathcal{L}_t$. Le altre ricorrenze sono parenti meno fedeli dello stesso
passo. L'accumulo puro dell'attenzione lineare scrive $v_t k_t^\top$ senza
sottrarre ciò che la memoria già predice: equivale a trascurare il termine
$S_{t-1} k_t$ nel gradiente, cioè a fare il passo esatto sull'obiettivo
*linearizzato* $-v_t^\top S\, k_t$; coincide con il passo su $\mathcal{L}_t$
solo quando la memoria non ha ancora nulla da dire sulla chiave corrente
($S_{t-1} k_t = 0$, per esempio con chiavi mutuamente ortogonali). E il gate
$\alpha_t$ è, in questa lettura, un **weight decay adattivo**: la contrazione
$S \leftarrow \alpha_t S$ che l'ottimizzazione applica per non lasciare che i
vecchi coefficienti si accumulino all'infinito, regolata token per token. Ogni
RNN lineare che abbiamo incontrato è, dunque, un passo di ottimizzazione
online sulla stessa famiglia di problemi ai minimi quadrati: a cambiare è
quanto fedelmente segue il gradiente di $\mathcal{L}_t$.

`````

La {numref}`fig-famiglia-ricorrenze-lineari` mette in fila lo «zoo» delle
transizioni di stato: si parte dall'identità $I$ dell'accumulo puro e si sale in
espressività un gradino alla volta, fino alla transizione gated-delta che le
contiene tutte come casi particolari.

```{figure} ../figures/famiglia-ricorrenze-lineari.svg
:name: fig-famiglia-ricorrenze-lineari
:alt: Cinque riquadri in fila che mostrano la matrice di transizione di stato di ciascuna ricorrenza lineare, dalla più semplice alla più espressiva. Primo riquadro, attenzione lineare, matrice identità I con la diagonale piena e il resto vuoto. Secondo riquadro, Mamba-2 e RetNet, scalare alpha per identità, la diagonale uniforme scurita da un unico fattore. Terzo riquadro, GLA, diagonale Diag(alpha) con celle di intensità diversa lungo la diagonale. Quarto riquadro, DeltaNet, identità meno beta k k trasposto, con un blocco fuori diagonale che rappresenta la correzione householder. Quinto riquadro, Gated DeltaNet, alpha per parentesi identità meno beta k k trasposto, che combina il decadimento diagonale e il blocco di correzione.
:width: 85%

Lo «zoo» delle ricorrenze lineari, ordinato per espressività crescente della
transizione di stato in $S_t = S_{t-1}\,(\text{transizione}_t) + v_t
k_t^\top$: identità, scalare, diagonale, Householder e gated-delta. È il
fattore che moltiplica $S_{t-1}$ a distinguerle (nelle due righe con la delta
rule il termine di scrittura porta anche il fattore $\beta_t$).
```

La tabella seguente è la stessa figura in forma algebrica: le cinque
ricorrenze, con l'aggiornamento completo di $S_t$ e il fattore di transizione
che moltiplica, da destra, lo stato precedente.

| Metodo | Aggiornamento di $S_t$ | Transizione di stato |
| :--- | :--- | :--- |
| Attenzione lineare | $S_t = S_{t-1} + v_t\, k_t^\top$ | $I$ (identità) |
| Mamba-2 / RetNet | $S_t = \alpha_t\, S_{t-1} + v_t\, k_t^\top$ | $\alpha_t I$ (decadimento scalare; commuta con lo stato) |
| GLA | $S_t = S_{t-1}\operatorname{Diag}(\alpha_t) + v_t\, k_t^\top$ | $\operatorname{Diag}(\alpha_t)$ (decadimento diagonale) |
| DeltaNet | $S_t = S_{t-1}(I - \beta_t\, k_t k_t^\top) + \beta_t\, v_t\, k_t^\top$ | $I - \beta_t\, k_t k_t^\top$ (Householder) |
| Gated DeltaNet | $S_t = S_{t-1}\big[\alpha_t(I - \beta_t\, k_t k_t^\top)\big] + \beta_t\, v_t\, k_t^\top$ | $\alpha_t(I - \beta_t\, k_t k_t^\top)$ (gated-delta) |

Letta dall'alto in basso, la tabella racconta una sola storia: **cambia
soltanto la transizione di stato**, il fattore che decide *come* la memoria di
ieri sopravvive a oggi. Tutto il resto (lo stato di dimensione fissa, la
scrittura per prodotto esterno, il fatto che si addestri in parallelo e si
inferisca in modo ricorrente) resta identico. Questa è la struttura profonda
che unifica l'intera famiglia, e la stessa che, nel prossimo capitolo,
ritroveremo arrivando da tutt'altra strada: quella dei sistemi dinamici degli
State Space Model. Non è un caso che Mamba-2 compaia due volte, qui tra le
attenzioni lineari e là tra gli SSM: è il ponte, e la sua «dualità» tra stato
e attenzione è la prova che le due famiglie sono due viste della stessa cosa.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Il registro che somma e basta ha due difetti: **non dimentica** (la pagina si
  satura di tracce sovrapposte) e **non corregge** (una voce sbagliata resta
  scritta). Sono mali diversi, e si curano con due mosse diverse.
- **Dimenticare**: si lascia sbiadire l'inchiostro a ogni passo, così la lavagna
  fa spazio da sola (con un fattore di $0{,}9$, dopo dieci passi una voce vale
  circa un terzo). Si può sbiadire tutto allo stesso ritmo, fissato una volta per
  tutte; oppure decidere il ritmo parola per parola, guardando cosa si sta
  leggendo; oppure sbiadire **zona per zona** della lavagna, tenendo nitida la
  zona con le cose che serviranno ancora e cancellando in fretta quella degli
  appunti di servizio. Sono i tre gradini che vanno dal più grossolano al più
  selettivo, ed è la differenza tra abbassare le luci di tutta la stanza e
  regolare ogni lampada singolarmente.
- **Correggere**: prima di scrivere si consulta la rubrica, si legge la risposta
  che dà oggi e si annota soltanto la differenza rispetto a quella giusta, dosata
  da una manopola. Girata al massimo, la voce viene sovrascritta; a zero, resta
  com'era. È una vecchia regola dell'apprendimento adattivo (Widrow e Hoff),
  tornata utile quando ci si è accorti che questa memoria è la stessa idea di
  certe reti degli anni Novanta, che riscrivevano al volo la propria memoria
  mentre leggevano. Per anni è stata impraticabile sui testi lunghi, perché va
  fatta in fila; DeltaNet ha trovato il modo di farla a blocchi, e quindi in
  parallelo.
- **Gated DeltaNet** mette insieme i due gesti, che sono complementari: sbiadire
  alleggerisce tutto in blocco ma non sa che cosa sta buttando via, correggere
  aggiusta una voce per volta ma non svuota nulla. Insieme dimenticano in fretta
  ciò che non serve *e* aggiustano con precisione ciò che tengono.
- Il filo che unisce tutto: ogni ricorrenza di questo capitolo è **un passo di
  apprendimento fatto al volo** (la «regressione online» del titolo qui sopra),
  un token alla volta, sullo stesso identico compito, «data questa chiave,
  rispondi con questo valore»: è la retta di best fit aggiustata di continuo
  invece che calcolata in blocco. Correggere è il passo fatto bene, perché prima
  guarda l'errore; sommare alla cieca va bene solo finché le etichette non si
  assomigliano fra loro; sbiadire serve a non far gonfiare la memoria
  all'infinito.
- A cambiare, da un'architettura all'altra, è **solo il modo in cui la memoria di
  ieri sopravvive a oggi**: tenerla intatta, sbiadirla tutta allo stesso ritmo,
  sbiadirla zona per zona, cancellare la vecchia traccia di una singola voce
  prima di riscriverla, o le ultime due cose insieme. Tutto il resto della
  ricorrenza resta identico.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- L'accumulo puro dell'attenzione lineare ha due difetti: **non dimentica**
  (la memoria si satura) e **non corregge** (le voci sbagliate restano). Gate e
  delta rule li risolvono separatamente.
- Il **gate di dimenticanza** moltiplica la memoria per un fattore in $(0,1)$:
  scalare e fisso in RetNet, scalare e data-dipendente in Mamba-2,
  **diagonale** e data-dipendente in GLA; dal più grossolano al più selettivo,
  canale per canale.
- La **delta rule** (Widrow–Hoff, dai fast weights di Schlag e colleghi) scrive
  solo l'**errore** $\mathbf{v}_t - S_{t-1}\mathbf{k}_t$ scalato da
  $\beta_t$: $\beta_t=1$ sovrascrive la chiave, $\beta_t=0$ la ignora.
  DeltaNet l'ha resa
  **parallelizzabile** (algoritmo chunk-parallel, rappresentazione WY).
- **Gated DeltaNet** combina i due gesti complementari: $\alpha_t$ decade in
  modo globale, $\beta_t$ corregge in modo mirato (dimentica in fretta *e*
  aggiusta con precisione).
- Il filo che unisce tutto: ogni ricorrenza di questo capitolo è **un passo di
  apprendimento fatto al volo** (la «regressione online» del titolo qui sopra),
  un token alla volta, sullo stesso identico compito, «data questa chiave,
  rispondi con questo valore». La delta rule è il
  passo fatto bene, perché prima guarda l'errore e poi corregge; l'accumulo
  puro scrive alla cieca, senza controllare cosa c'era già, e va bene solo
  finché le chiavi non si assomigliano fra loro; il gate serve a non far
  gonfiare la memoria all'infinito.
- A cambiare, da un'architettura all'altra, è **solo la transizione di stato**
  ($I \to \alpha_t I \to \operatorname{Diag}(\alpha_t) \to
  I-\beta_t \mathbf{k}_t \mathbf{k}_t^\top
  \to \alpha_t(I-\beta_t \mathbf{k}_t \mathbf{k}_t^\top)$): tutto il
  resto della ricorrenza resta identico.
```

`````
