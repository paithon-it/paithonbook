# Scrivere meglio nella memoria: gate e delta rule

Nella sezione precedente abbiamo scoperto che l'attenzione lineare è, sotto
mentite spoglie, una rete ricorrente: al posto dell'archivio di appunti che si
allunga a ogni parola, un'unica memoria di dimensione fissa. Immaginala come
un registro che funziona da **rubrica**: ogni parola che passa vi aggiunge una
voce «etichetta → informazione», sommandola a quelle che ci sono già, e per
rispondere a una domanda la rubrica si rilegge invece di ripercorrere tutto il
testo. È l'immagine che ci accompagnerà per tutta la sezione.

`````{tab} Elementare

Una precisazione sulla rubrica, perché è quella che rende conto di tutto il
resto. La rubrica non ha una riga per contatto: ha un numero fisso di caselle,
e ogni voce nuova si somma a quello che c'è già scritto. Quando le chiedi «che
cosa corrisponde a questa etichetta?», lei non pesca una riga: risponde con un
miscuglio, in cui pesa soprattutto l'informazione scritta sotto l'etichetta più
somigliante, più un po’ di tutte le altre. Finché le etichette sono ben diverse
fra loro quel «po’ di tutte le altre» è trascurabile, ed è per questo che il
trucco funziona.

`````

`````{tab} Superiore

In formule, e con la scrittura snella annunciata nella sezione precedente (che
vale da qui a fine capitolo: la feature map $\phi$ è posta all'identità e il
normalizzatore $\mathbf{z}_t$ non compare, secondo la seconda delle due scuole),

$$
\mathbf{S}_t = \mathbf{S}_{t-1} + \mathbf{v}_t\, \mathbf{k}_t^\top ,
$$

letta proiettando lo stato sulla query, $\mathbf{o}_t = \mathbf{S}_t\, \mathbf{q}_t$. Qui $\mathbf{k}_t$, $\mathbf{v}_t$ e
$\mathbf{q}_t$ sono le chiavi, i valori e le query che conosciamo dall'attenzione dei
Transformer, $\mathbf{v}_t \mathbf{k}_t^\top$ è il prodotto esterno che scrive il valore sotto la
sua chiave, e $\mathbf{S}_t$ è la memoria dopo aver letto i primi $t$ token.

`````

Il registro, però, ha due difetti che si vedono a occhio nudo. **Non
dimentica**: ogni voce resta scritta per sempre, e con abbastanza token la
pagina si satura di tracce sovrapposte finché non si legge più nulla di
preciso. E **non corregge**: se una voce era sbagliata, l'unico modo per
rimediare è scriverne un'altra sopra che la contraddica; la vecchia resta lì a
disturbare. Da qui due idee semplici e complementari per scrivere *meglio*
nella memoria: **dimenticare** (i gate) e **correggere** (la delta rule). Sono
le due mosse da cui nasce, come vedremo, l'intera famiglia di modelli di questo
capitolo e del prossimo, quelli che tengono una memoria di taglia fissa e la
riscrivono a ogni parola (nei paper si chiamano *ricorrenze lineari*).

## Dimenticare: i gate

La prima idea è lasciare che le voci vecchie sbiadiscano da sole. Invece di
tramandare la memoria intatta, la si moltiplica a ogni passo per un fattore di
decadimento minore di uno: ciò che è stato scritto tempo fa pesa sempre meno,
finché svanisce. È il **gate di dimenticanza**, e non è un'idea nuova: è lo
stesso *forget gate* che Gers, Schmidhuber e Cummins aggiunsero nel 2000 alle
LSTM, le celle ricorrenti che abbiamo incontrato nel capitolo sull'NLP
{cite}`gers2000learning`. Qui torna nella sua forma più spoglia, un numero che
moltiplica.

`````{tab} Elementare

Pensa al registro come a una lavagna su cui l'inchiostro sbiadisce. A ogni
passo tutte le voci si affievoliscono un po’: quelle appena scritte sono nitide,
quelle vecchie quasi invisibili. Se il fattore di sbiadimento è $0{,}9$, dopo
dieci passi una voce vale $0{,}9^{10} \approx 0{,}35$ di quanto valeva: circa
un terzo. La lavagna così non si satura mai, perché fa spazio da sola buttando via
il passato lontano.

Quel numero, lo $0{,}9$, non lo sceglie nessuno a mano parola per parola. Ci
sono tre modi di deciderlo, ed è la scala su cui si dispongono i modelli di
questa famiglia. Nel primo è **fissato una volta per tutte** quando il modello
viene progettato, e non cambia mai (è la strada di RetNet, che incontreremo
nella prossima sezione). Nel secondo il modello lo **calcola parola per
parola** a partire da ciò che sta leggendo: davanti a una cosa importante
sbiadisce poco, davanti a un intercalare sbiadisce molto (è la strada di
**Mamba-2**, un modello che incontreremo nel prossimo capitolo e che si
comporta, da questo lato, esattamente così). La
differenza fra i due sta tutta lì: nel primo caso il valore è deciso una volta
sola, in fase di progetto, e la rete se lo tiene così com'è per sempre; nel
secondo non c'è nessun valore da tenere, perché a essere stato appreso durante
l'addestramento è il *modo* di ricalcolarlo a ogni parola.

E c'è un terzo modo, il più fine. Sbiadire *tutta la lavagna allo stesso
ritmo* è grossolano: magari in una parte della lavagna c'è un dettaglio che
servirà ancora fra mille parole, in un'altra solo appunti usa-e-getta. Meglio
poter sbiadire **una zona della lavagna alla volta**: tenere nitida quella con
le cose che contano (fattore $0{,}99$, quasi non svanisce) e cancellare in
fretta quella degli appunti di servizio (fattore $0{,}5$, dimezza a ogni
passo). È la differenza tra abbassare le luci di tutta la stanza e regolare
ogni lampada singolarmente.

Le zone, poi, non sono zone a caso: sono i *canali*, cioè le posizioni della
fila di numeri con cui il modello scrive ogni parola, e c'è un interruttore per
ciascuna. Il modello che lo fa si chiama **GLA**, cioè attenzione lineare con
gli interruttori (in inglese *gated linear attention*): la troverai citata
anche più avanti, ed è semplicemente questo.

`````

`````{tab} Superiore

La forma più semplice è il **decadimento scalare**: un unico numero
$\alpha_t \in (0,1)$ moltiplica l'intera memoria,

$$
\mathbf{S}_t = \alpha_t\, \mathbf{S}_{t-1} + \mathbf{v}_t\, \mathbf{k}_t^\top ,
$$

dove $\alpha_t$ è il gate di dimenticanza al passo $t$ e $\mathbf{v}_t \mathbf{k}_t^\top$ è la
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
\mathbf{S}_t = \mathbf{S}_{t-1}\, \operatorname{Diag}(\boldsymbol{\alpha}_t) + \mathbf{v}_t\, \mathbf{k}_t^\top ,
$$

dove ora $\boldsymbol{\alpha}_t \in (0,1)^d$ è un *vettore* di gate, uno per
canale di chiave (il grassetto lo distingue dallo scalare $\alpha_t$ della
formula precedente: stesso ruolo, una componente sola contro $d$), e
$\operatorname{Diag}(\boldsymbol{\alpha}_t)$ è la matrice diagonale che ne fa i
coefficienti. Il **lato** da cui moltiplica non è un dettaglio di scrittura:
con la convenzione di questo capitolo ($\mathbf{S} = \sum_i \mathbf{v}_i \mathbf{k}_i^\top$,
lettura $\mathbf{o} = \mathbf{S}\mathbf{q}$) le colonne di $\mathbf{S}$ sono
indicizzate dai canali della chiave e le righe da quelli del valore, quindi
moltiplicando **da destra** ogni colonna decade al proprio ritmo, che è quel
che si vuole; da sinistra sbiadirebbero i canali del valore, che è un'altra
cosa. (Anche la transizione della delta rule, fra poco, sta da quel lato e per
la stessa ragione.) Così $\alpha_{t,i}\to 1$ conserva il canale $i$
(nel limite si torna all'accumulo puro), $\alpha_{t,i}\to 0$ lo azzera. Il
vettore è ricavato dall'input con una proiezione a **basso rango** (un collo di
bottiglia stretto, dimensione 16) seguita da una sigmoide, così da generare $d$
gate distinti senza far esplodere il numero di parametri; la sigmoide è poi
elevata a $1/\tau$ con $\tau = 16$, una *temperatura* che spinge i gate verso
1, cioè verso l'oblio lento, che è la molla di tutto il meccanismo:
$\boldsymbol{\alpha}_t = \sigma\big(\mathbf{x}_t \mathbf{W}^1_\alpha \mathbf{W}^2_\alpha + \mathbf{b}_\alpha\big)^{1/\tau}$. La
gerarchia è chiara: scalare fisso (RetNet) $\to$ scalare data-dipendente
(Mamba-2) $\to$ diagonale data-dipendente (GLA), dal più grossolano al più
selettivo.

`````

Il gate risolve il primo difetto (la saturazione) ma non il secondo. Per
quanto si sbiadisca, ogni scrittura resta un'aggiunta cieca: nessuno controlla
se quella voce contraddice ciò che c'è già.

## Correggere: la delta rule

La seconda idea si dice in una riga: prima di scrivere, guardare che cosa c'è
già scritto. Viene da lontano, e per arrivarci conviene passare da una
scoperta del 2021.

In quell'anno Schlag, Irie e Schmidhuber si accorgono che l'attenzione lineare
(nei paper la chiamano anche *Transformer lineare*: è la stessa cosa) è a tutti
gli effetti una rete che Schmidhuber aveva progettato negli anni Novanta, il
**fast weight programmer**, il «programmatore di pesi veloci»
{cite}`schlag2021linear`. Il nome dice che in queste reti convivono due memorie con due
velocità diverse: una **lenta**, i pesi appresi durante l'addestramento, che
cambiano nel corso di giorni di calcolo e poi restano fermi per sempre; e una
**veloce**, il nostro registro, che cambia a ogni parola letta. La rete lenta
non contiene le risposte: contiene le istruzioni con cui, mentre legge,
riscrive al volo la memoria veloce. E se è una rete che si riprogramma da sé,
tanto vale insegnarle a farlo con criterio.

`````{tab} Elementare

Torniamo alla rubrica. L'accumulo puro è chi, ogni volta che scopre un numero
di telefono, lo scrive sopra a quello che c'era senza nemmeno guardarlo: anche
se quel contatto era già in rubrica, magari con un numero sbagliato. Le due
scritte si sovrappongono, e chi consulta la rubrica si sente rispondere un
miscuglio del numero vecchio e di quello nuovo, in cui non si riconosce più né
l'uno né l'altro.

La **delta rule** fa la cosa sensata: prima di scrivere, *cerca il contatto* e
legge il numero attualmente memorizzato. Poi scrive soltanto la
**correzione**: la differenza tra quello giusto e quello che c'era. Un esempio
con i numeri: alla voce «Mario» la memoria oggi risponde $7$, ma il valore
giusto è $10$; l'errore è $10 - 7 = 3$. Con un «passo di correzione» pari a
metà (nelle formule questa manopola si chiama *beta*, e si scrive $\beta$)
scrivo solo $0{,}5 \times 3 = 1{,}5$, e la memoria passa a rispondere $8{,}5$:
si avvicina alla verità senza cancellare tutto di colpo. La manopola dosa
quanto dare retta all'errore: al massimo, $\beta = 1$, **sovrascrivo** del
tutto (la memoria risponde $10$); a zero **ignoro** e lascio $7$. È esattamente
come si corregge un tiro: non riparti da zero, aggiusti in proporzione a quanto
hai sbagliato.

Due modi di dire la stessa cosa, e conviene fissarlo perché più avanti la
tabella userà l'altro: scrivere la differenza, oppure cancellare la vecchia
risposta e rimetterne una nuova, danno lo stesso numero. Togliere metà del $7$
e metterci metà del $10$ fa $3{,}5 + 5 = 8{,}5$, cioè quello di prima. Quanto
si cancella e quanto si scrive li decide la stessa manopola.

Da qui una conseguenza che tornerà utile fra poco: girata a zero, la manopola
non spegne solo la correzione, spegne la scrittura. Se non scrivo la
differenza, non scrivo niente, e la memoria resta com'era.

Correggere in proporzione all'errore è una vecchia idea dell'ingegneria: la
formularono Widrow e Hoff nel 1960, per una macchina che imparava aggiustandosi
da sé. Come il fattore di sbiadimento di poco fa, la manopola non la gira una
persona: il modello la calcola da sé a ogni parola, in base a quello che sta
leggendo, e a essere stato appreso durante l'addestramento è il modo di
calcolarla.

`````

`````{tab} Superiore

Al passo $t$, prima di scrivere, si interroga la memoria con la *chiave*
corrente e si ottiene il valore che essa già predice per quella chiave,

$$
\bar{\mathbf{v}}_t = \mathbf{S}_{t-1}\, \mathbf{k}_t .
$$

Qui $\bar{\mathbf{v}}_t$ è la «vecchia risposta»: ciò che la rubrica restituisce oggi
alla chiave $\mathbf{k}_t$. La delta rule scrive allora soltanto l’**errore** $\mathbf{v}_t -
\bar{\mathbf{v}}_t$, scalato da un *learning-rate* $\beta_t \in (0,1)$ appreso
dinamicamente ($\beta_t = \sigma(\mathbf{w}_\beta^\top \mathbf{x}_t)$, dove
$\mathbf{x}_t$ è il vettore in ingresso al passo $t$, $\mathbf{w}_\beta$ un
vettore di pesi appresi e $\sigma$ la sigmoide, che tiene $\beta_t$ fra $0$ e
$1$):

$$
\mathbf{S}_t = \mathbf{S}_{t-1} + \beta_t\,(\mathbf{v}_t - \bar{\mathbf{v}}_t)\, \mathbf{k}_t^\top
    = \mathbf{S}_{t-1} + \beta_t\,(\mathbf{v}_t - \mathbf{S}_{t-1} \mathbf{k}_t)\, \mathbf{k}_t^\top .
$$

È la regola di **Widrow–Hoff** (o LMS), il mattone dell'apprendimento
adattivo. Raccogliendo i termini si ottiene una forma compatta ed elegante, la
**Householder generalizzata**:

$$
\mathbf{S}_t = \mathbf{S}_{t-1}\,(\mathbf{I} - \beta_t\, \mathbf{k}_t \mathbf{k}_t^\top) + \beta_t\, \mathbf{v}_t\, \mathbf{k}_t^\top ,
$$

dove $\mathbf{I}$ è l'identità e $\mathbf{k}_t \mathbf{k}_t^\top$ è il prodotto esterno della chiave con se
stessa. Il fattore $(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top)$ agisce come una transizione di
stato che **cancella la vecchia traccia lungo la direzione $\mathbf{k}_t$** appena prima
di scriverci quella nuova: se $\beta_t = 1$ sovrascrive del tutto quella chiave,
se $\beta_t = 0$ lascia la memoria intatta. Perché il fattore sia ben
condizionato (autovalori in $[0,1]$) le chiavi vanno **normalizzate in norma
$L_2$**, così che $\mathbf{k}_t^\top \mathbf{k}_t = 1$.

Una precisazione su quanto anticipato nella sezione precedente: sono proprio
Schlag e colleghi a compiere il primo passo della scuola «senza $\mathbf{z}_t$».
Mostrano che il **normalizzatore** di Katharopoulos (la somma delle chiavi
trasformate) può crescere senza controllo e diventare instabile, e lo
**scartano**, normalizzando invece per somma le chiavi e le query trasformate.
La famiglia di ricorrenze che seguiamo qui (GLA, DeltaNet e le loro parenti)
porta la rinuncia a compimento: applica una LayerNorm all'uscita, e ottiene lo
stesso effetto stabilizzante senza il termine $\mathbf{z}_t$. La strada, però, non è la
stessa per tutte, ed è un dettaglio che vale la pena guardare: la
normalizzazione $L_2$ delle chiavi è di **DeltaNet**, dove serve a tenere gli
autovalori della transizione in $[0,1]$, cioè a fare della transizione una
contrazione; in GLA le chiavi sono una proiezione lineare secca, e a tenere
limitato lo stato è il gate $\boldsymbol{\alpha}_t \in (0,1)^d$, cioè la
contrazione è nella memoria e non nella trasformazione. Due vie diverse per lo
stesso scopo.

`````

Resta un problema pratico, ed è serio. Sbiadire è un'operazione che si può
anticipare: quanto resterà fra dieci parole di quello che scrivo adesso si sa
già adesso, quindi il conto di tutte le parole si può distribuire su tante
unità di calcolo che lavorano insieme. Correggere no: per sapere che cosa
scrivere alla parola numero cento bisogna prima sapere che cosa la memoria
risponde alla parola numero cento, e quello dipende da tutte le novantanove
correzioni precedenti. Ogni passo aspetta quello prima, e una scheda grafica
che potrebbe fare mille conti insieme ne fa uno.

Il modello con la delta rule è dunque nato subito, ma nato lento. Lo propongono
nel 2021 gli stessi Schlag, Irie e Schmidhuber, con il nome di **DeltaNet**, e
lo addestrano davvero come modello linguistico. L'algoritmo che avevano,
però, procede in fila lungo la sequenza: spreca le schede grafiche e non regge
oltre le taglie piccole. Nel 2024 Yang e colleghi (NeurIPS 2024)
{cite}`yang2024deltanet` lo sbloccano: un algoritmo **chunk-parallel** che
spezza la sequenza in blocchi e, dentro ogni blocco, riesce a fare i conti
tutti insieme, passando al blocco successivo soltanto un riassunto compatto di
quello appena chiuso (nei paper quel riassunto si chiama forma *WY*). Da
lì in avanti la delta rule è addestrabile alla scala dei modelli linguistici. E
il guadagno si vede dove ci si aspetta: quando il compito è ritrovare il valore
giusto legato a una chiave incontrata prima (il numero di telefono del contatto
giusto, per restare alla rubrica), DeltaNet fa meglio delle altre memorie di
taglia fissa di questo capitolo e si avvicina all'attenzione dei Transformer,
che su quel terreno resta il metro di paragone perché non butta via niente.

## Unire oblio e correzione: Gated DeltaNet

A questo punto abbiamo due mosse che risolvono difetti diversi, e la domanda
si fa naturale: perché scegliere? Il gate **svuota in fretta** la memoria, ma
in modo **uniforme e indiscriminato**: non sa *cosa* sta buttando via. La
delta rule fa **correzioni mirate** su singole chiavi, ma da sola **non
svuota**: tende a lasciare la memoria piena di tracce, sia pure aggiustate. Le
due mosse sono dunque **complementari**, e conviene tenerle insieme: è quello
che fanno Yang, del MIT, con Kautz e Hatamizadeh di NVIDIA in **Gated
DeltaNet**, presentato a ICLR nel 2025 {cite}`yang2024gateddelta`.

`````{tab} Elementare

Torniamo alla lavagna e alla rubrica, che qui diventano lo stesso oggetto.
Ogni volta che arriva una parola nuova si fanno due gesti, in quest'ordine.
Primo: si passa uno straccio su tutta la lavagna, che sbiadisce quello che
c'era senza guardare cosa fosse. Secondo: si cerca la voce che riguarda la
parola appena letta, si legge cosa dice adesso e si scrive sopra soltanto la
correzione, come faceva la rubrica di Mario.

I due gesti si occupano di due problemi diversi e non si pestano i piedi. Lo
straccio fa spazio, e serve perché la lavagna non arrivi mai piena; la
correzione tiene in ordine le voci che restano, e serve perché quello che c'è
scritto sia giusto. Se si smette di passare lo straccio, resta la sola
correzione: le voci sono precise, ma la lavagna a un certo punto si riempie. Se
si smette di correggere, resta il solo straccio, e questa è la parte
sorprendente: non si torna al registro che sommava e basta, si ottiene una
lavagna che sbiadisce e non scrive più niente: è la conseguenza vista con
Mario, dove la manopola girata a zero spegneva la correzione e con lei la
scrittura, perché quel che si scrive *è* la correzione. Insieme, invece, i due
gesti fanno il mestiere per intero: buttare via in fretta ciò che non serve e
tenere in ordine ciò che si conserva.

`````

`````{tab} Superiore

La ricorrenza combina i due fattori di transizione:

$$
\mathbf{S}_t = \mathbf{S}_{t-1}\big[\, \alpha_t\,(\mathbf{I} - \beta_t\, \mathbf{k}_t \mathbf{k}_t^\top)\,\big]
      + \beta_t\, \mathbf{v}_t\, \mathbf{k}_t^\top ,
$$

dove i due parametri hanno ruoli distinti e leggibili a colpo d'occhio:

- $\alpha_t \in (0,1)$ è un **gate scalare** (nella parametrizzazione di
  Mamba-2): il *decadimento globale*, che alleggerisce l'intera memoria a ogni
  passo;
- $\beta_t \in (0,1)$ è la **forza di scrittura** della delta rule: quanto
  correggere la chiave corrente, come nella sezione precedente.

Con $\alpha_t \to 1$ si ritrova la pura delta rule (nessun oblio, sole
correzioni). Il limite $\beta_t \to 0$ va invece letto con attenzione, perché
$\beta_t$ compare **anche nel termine di scrittura**: la transizione si riduce
sì al puro gate scalare $\alpha_t \mathbf{I}$, ma con la correzione si spegne anche la
scrittura, e quel che resta è $\mathbf{S}_t = \alpha_t \mathbf{S}_{t-1}$, una memoria che decade
a zero senza registrare più nulla. La riga «Mamba-2 / RetNet» della tabella qui sotto, invece, non si ritrova
come caso particolare: là il gate scalare la scrittura la fa a piena forza,
mentre qui la forza di scrittura è la stessa manopola che comanda la
correzione. Gated DeltaNet contiene la delta rule pura, non il decadimento
scalare puro. Gated DeltaNet vive nel mezzo: dimentica in fretta
ciò che non serve più *e* aggiusta con precisione ciò che tiene. La
parallelizzazione lungo la sequenza si ottiene estendendo la stessa
rappresentazione WY di DeltaNet, così che anche questa forma più ricca resti
addestrabile su contesti lunghi.

`````

## Tutto è regressione online

Accumulo, gate, delta rule e la loro combinazione sembrano trucchi diversi, e
sono lo **stesso gesto** visto da angolazioni diverse. E quel gesto ha un nome che conosciamo bene dal {doc}`capitolo sul machine
learning </MachineLearning/overview>`: è un passo di apprendimento.

Il titolo della sezione lo dice con le parole di quel capitolo, e conviene
scioglierle. **Regressione** è il mestiere di indovinare un numero a partire da
un altro, quello della retta che si fa passare in mezzo a una nuvola di punti.
**Online** vuol dire farlo mentre i dati arrivano, uno alla volta, senza poter
tornare indietro a rileggerli. Tutto questo capitolo, dalla prima riga, ha
descritto una macchina che fa esattamente questo.

`````{tab} Elementare

Ricordi la retta di best fit? Data una nuvola di punti, cercavamo la retta
che minimizza l'errore quadratico: la somma dei quadrati degli scarti tra
valori veri e previsti. Facevamo tutto in una volta, con l'intero dataset
sotto gli occhi.

La nostra rubrica fa la stessa cosa, ma **un dato alla volta, mentre scorre**.
Il suo compito è imparare a rispondere bene: data un'etichetta, restituire
l'informazione giusta. A ogni parola arriva una nuova coppia
(etichetta, informazione) e la rubrica fa **un piccolo passo** per rispondere
meglio, senza poter rileggere il passato. È la versione «in tempo reale» della
retta di best fit: non risolvi il problema in blocco, lo aggiusti in
continuazione a ogni esempio che passa. Il passo lo abbiamo già visto in
numeri: la rubrica che rispondeva $7$, dopo la correzione risponde $8{,}5$, e
si avvicina al $10$ senza arrivarci in un colpo solo. Ecco perché la delta rule
assomigliava a correggere un tiro:
non *assomiglia* a imparare, è imparare, nell'unico modo in cui una macchina
lo fa, cioè guardare di quanto ha sbagliato e spostarsi un po’ in quella
direzione.

`````

`````{tab} Superiore

Fissiamo, a ogni passo, l'obiettivo di **regressione online**

$$
\mathcal{L}_t(\mathbf{S}) = \tfrac{1}{2}\,\lVert \mathbf{S}\,\mathbf{k}_t - \mathbf{v}_t \rVert^2 ,
$$

cioè «mappare la chiave $\mathbf{k}_t$ nel valore $\mathbf{v}_t$», con $\mathbf{S}$ la memoria da
addestrare. Il suo gradiente rispetto a $\mathbf{S}$ è

$$
\nabla_{\mathbf{S}}\,\mathcal{L}_t = (\mathbf{S}\,\mathbf{k}_t - \mathbf{v}_t)\, \mathbf{k}_t^\top ,
$$

e un singolo passo di discesa del gradiente con learning-rate $\beta_t$,
partendo da $\mathbf{S}_{t-1}$, dà

$$
\mathbf{S}_t = \mathbf{S}_{t-1} - \beta_t\,(\mathbf{S}_{t-1} \mathbf{k}_t - \mathbf{v}_t)\, \mathbf{k}_t^\top
    = \mathbf{S}_{t-1}\,(\mathbf{I} - \beta_t\, \mathbf{k}_t \mathbf{k}_t^\top) + \beta_t\, \mathbf{v}_t\, \mathbf{k}_t^\top .
$$

È **esattamente** la delta rule di DeltaNet: il passo di gradiente *esatto*
su $\mathcal{L}_t$. Le altre ricorrenze sono parenti meno fedeli dello stesso
passo. L'accumulo puro dell'attenzione lineare scrive $\mathbf{v}_t \mathbf{k}_t^\top$ senza
sottrarre ciò che la memoria già predice: equivale a trascurare il termine
$\mathbf{S}_{t-1} \mathbf{k}_t$ nel gradiente, cioè a fare il passo esatto (a learning rate
unitario) sull'obiettivo *linearizzato* $-\mathbf{v}_t^\top \mathbf{S}\, \mathbf{k}_t$. Con
$\mathcal{L}_t$ coincide solo se valgono **due** condizioni: che la memoria non
abbia ancora nulla da dire sulla chiave corrente ($\mathbf{S}_{t-1} \mathbf{k}_t = 0$, per
esempio con chiavi mutuamente ortogonali) *e* che la scrittura sia a piena
forza, $\beta_t = 1$. La prima da sola non basta, e si vede subito: con chiavi
ortonormali la delta rule scrive $\beta_t\, \mathbf{v}_t \mathbf{k}_t^\top$ dove l'accumulo
scrive $\mathbf{v}_t \mathbf{k}_t^\top$, quindi a $\beta_t = 0{,}5$ le due memorie si separano
già al primo token, di un fattore due. E il gate
$\alpha_t$ è, in questa lettura, un **weight decay adattivo**: la contrazione
$\mathbf{S} \leftarrow \alpha_t \mathbf{S}$ che l'ottimizzazione applica per non lasciare che i
vecchi coefficienti si accumulino all'infinito, regolata token per token. Ogni
RNN lineare che abbiamo incontrato è, dunque, un passo di ottimizzazione
online sulla stessa famiglia di problemi ai minimi quadrati: a cambiare è
quanto fedelmente segue il gradiente di $\mathcal{L}_t$.

`````

La {numref}`fig-famiglia-ricorrenze-lineari` mette in fila lo «zoo» delle
memorie di questo capitolo e del prossimo: si parte da quella che tiene tutto e
si sale un gradino alla volta, fino all'ultima, che tiene insieme le due mosse.
(Nella figura compaiono due nomi che appartengono al seguito: RWKV-6, che è
un'architettura della prossima sezione e sbiadisce zona per zona come GLA, e
Mamba-2, che vedremo da vicino nel prossimo capitolo.)

```{figure} ../figures/famiglia-ricorrenze-lineari.svg
:name: fig-famiglia-ricorrenze-lineari
:alt: Cinque riquadri in fila, sotto una freccia che va da «accumulo puro» a «oblio + correzione mirata», mostrano la matrice di transizione di stato di ciascuna ricorrenza lineare. Primo riquadro, attenzione lineare, matrice identità I, la sola diagonale piena e il fondo vuoto. Secondo riquadro, RetNet e Mamba-2, alpha per identità, la diagonale uniforme ma di tinta più chiara, cioè scalata da un unico fattore. Terzo riquadro, GLA e RWKV-6, Diag(alpha), la diagonale a segmenti di intensità diversa. Quarto riquadro, DeltaNet, identità meno beta k k trasposto, la diagonale piena su un fondo velato che occupa tutto il quadrato, perché la correzione di rango uno tocca anche fuori dalla diagonale. Quinto riquadro, Gated DeltaNet, alpha per parentesi identità meno beta k k trasposto, con la diagonale scalata del secondo riquadro sopra il fondo velato del quarto.
:width: 85%

Lo «zoo» delle memorie di questi due capitoli, in fila da quella che sa fare
meno a quella che sa fare di più. In ogni quadrato è disegnato che cosa resta
della memoria di ieri: la diagonale è ciò che sopravvive, e il fondo è ciò che
viene toccato oltre la diagonale. Diagonale piena vuol dire tenere tutto;
diagonale scolorita, sbiadire tutto allo stesso modo; diagonale a segmenti,
sbiadire zona per zona; fondo velato, cancellare la voce che si sta per
riscrivere. L'ultimo riquadro tiene insieme la diagonale scolorita del secondo
e il fondo velato del quarto. Il resto del meccanismo non cambia mai.
```

La stessa storia, messa in tabella: cinque modi di far sopravvivere la memoria
di ieri, dal più semplice al più raffinato.

`````{tab} Elementare

| Modello | Che cosa fa della memoria di ieri |
| :--- | :--- |
| Attenzione lineare | La tiene tutta, intatta, e ci somma sopra la voce nuova. |
| Mamba-2 / RetNet | La sbiadisce tutta allo stesso ritmo, poi ci somma sopra la voce nuova. I due si dividono proprio sul ritmo: RetNet lo fissa una volta per tutte, Mamba-2 lo ricalcola a ogni parola. |
| GLA | La sbiadisce **zona per zona**, ogni zona al suo ritmo, poi ci somma sopra la voce nuova. |
| DeltaNet | Non la sbiadisce, ma prima di scrivere **sbianchetta la vecchia voce** proprio dell'etichetta che sta per riscrivere, tanto quanto dice la manopola, e ci scrive sopra la correzione. |
| Gated DeltaNet | Le due cose insieme: sbiadisce tutto, e in più cancella e riscrive la voce di turno. |

Cinque righe, una storia sola: **cambia soltanto la prima mossa**, quella che
decide che cosa resta di ciò che si era scritto prima. Tutto il resto (la
memoria che non cresce mai, la voce che si somma sotto la sua etichetta, il
fatto che il modello si alleni tutto in una volta e poi scriva una parola alla
volta) è identico in tutte e cinque, con una sola differenza di contorno: nelle
due righe che correggono, quel che si somma sotto l'etichetta è la correzione,
e la manopola la dosa.

`````

`````{tab} Superiore

| Metodo | Aggiornamento di $\mathbf{S}_t$ | Transizione di stato |
| :--- | :--- | :--- |
| Attenzione lineare | $\mathbf{S}_t = \mathbf{S}_{t-1} + \mathbf{v}_t\, \mathbf{k}_t^\top$ | $\mathbf{I}$ (identità) |
| Mamba-2 / RetNet | $\mathbf{S}_t = \alpha_t\, \mathbf{S}_{t-1} + \mathbf{v}_t\, \mathbf{k}_t^\top$ | $\alpha_t \mathbf{I}$ (decadimento scalare; commuta con lo stato) |
| GLA | $\mathbf{S}_t = \mathbf{S}_{t-1}\operatorname{Diag}(\boldsymbol{\alpha}_t) + \mathbf{v}_t\, \mathbf{k}_t^\top$ | $\operatorname{Diag}(\boldsymbol{\alpha}_t)$ (decadimento diagonale, un gate per canale) |
| DeltaNet | $\mathbf{S}_t = \mathbf{S}_{t-1}(\mathbf{I} - \beta_t\, \mathbf{k}_t \mathbf{k}_t^\top) + \beta_t\, \mathbf{v}_t\, \mathbf{k}_t^\top$ | $\mathbf{I} - \beta_t\, \mathbf{k}_t \mathbf{k}_t^\top$ (Householder) |
| Gated DeltaNet | $\mathbf{S}_t = \mathbf{S}_{t-1}\big[\alpha_t(\mathbf{I} - \beta_t\, \mathbf{k}_t \mathbf{k}_t^\top)\big] + \beta_t\, \mathbf{v}_t\, \mathbf{k}_t^\top$ | $\alpha_t(\mathbf{I} - \beta_t\, \mathbf{k}_t \mathbf{k}_t^\top)$ (gated-delta) |

Letta dall'alto in basso, la tabella racconta una sola storia: **cambia
soltanto la transizione di stato**, il fattore che moltiplica $\mathbf{S}_{t-1}$. Lo
stato di dimensione fissa, la scrittura per prodotto esterno, l'addestramento
parallelo e l'inferenza ricorrente restano identici (unica differenza di
contorno: nelle due righe con la delta rule il termine di scrittura porta anche
il fattore $\beta_t$).

`````

Questa è la struttura profonda che unifica l'intera famiglia, ed è la stessa
che, nel prossimo capitolo, ritroveremo arrivando da tutt'altra strada: quella
dei sistemi dinamici degli **State Space Model**. Non è un caso che Mamba-2 (il
modello che sbiadisce tutta la memoria allo stesso ritmo, ma decidendo il ritmo
parola per parola) compaia due volte, qui tra le attenzioni lineari e là tra
gli SSM: è il ponte fra le due famiglie. I suoi autori lo scrivono infatti in
due modi, una volta come memoria che si aggiorna e una volta come attenzione, e
chiamano *dualità* quella doppia scrittura.

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
  mentre leggevano. Per anni è rimasta lenta sui testi lunghi, perché va fatta
  in fila; poi si è trovato il modo di farla a blocchi, e quindi in
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
- La **delta rule** (Widrow–Hoff, dai fast weights di Schlag e colleghi, che nel
  2021 la portano in **DeltaNet**) scrive solo l’**errore**
  $\mathbf{v}_t - \mathbf{S}_{t-1}\mathbf{k}_t$ scalato da $\beta_t$: $\beta_t=1$ sovrascrive la chiave,
  $\beta_t=0$ la ignora. Yang e colleghi (2024) l'hanno resa
  **parallelizzabile** (algoritmo chunk-parallel, rappresentazione WY), e quindi
  scalabile.
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
  ($\mathbf{I} \to \alpha_t \mathbf{I} \to \operatorname{Diag}(\boldsymbol{\alpha}_t) \to
  \mathbf{I}-\beta_t \mathbf{k}_t \mathbf{k}_t^\top
  \to \alpha_t(\mathbf{I}-\beta_t \mathbf{k}_t \mathbf{k}_t^\top)$): tutto il
  resto della ricorrenza resta identico.
```

`````
