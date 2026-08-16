# A che serve saperlo, e dove sbaglia

Abbiamo speso due sezioni per ottenere un numero. Adesso la domanda che le
giustifica: a che cosa serve. Sono tre mestieri, e sul terzo questa famiglia ha
preso una lezione che vale più dei primi due.

## Primo: comprimere

Il legame è così stretto da essere la stessa cosa detta in due modi, e la
sezione sulla teoria dell'informazione, nei richiami di matematica, l'ha già
stabilito: il numero di bit che servono per scrivere un messaggio con il codice
migliore possibile è $-\log_2 p$ del messaggio. Non «all'incirca a meno di un
fattore»: proprio quella grandezza lì, con uno scarto di un paio di bit su
tutto il file, che è quanto costa scrivere in bit interi una quantità che
intera non è.

Un modello che sa dire $p(\mathbf{x})$ **è** un compressore, e non per
analogia. Gli si dà un file, lui dà una probabilità, un codificatore aritmetico
la trasforma in bit, e il file si ricostruisce esattamente. È la ragione per cui
in questa letteratura la qualità non si misura in punti su cento ma in **bit per
dimensione**: quanti bit costa, in media, ogni numero dell'immagine. Un numero
più basso vuol dire un modello migliore e un file più piccolo, ed è la stessa
frase.

`````{tab} Elementare

Vale la pena fare il conto una volta con numeri veri, perché è lì che il legame
smette di sembrare una metafora.

Una figurina a colori di 32 pixel per lato è fatta di $32 \times 32 \times 3 =
3.072$ numeri, ognuno fra 0 e 255. Chi non sa niente di come sono fatte le
immagini deve spendere 8 bit per ciascuno, cioè 3.072 byte: è il file grezzo.
Il miglior modello fra quelli che questo capitolo racconta costa 2,92 bit per
numero, e $3.072 \times 2{,}92$ fa 8.970 bit, cioè meno di 1.130 byte. Stesso
contenuto, ricostruibile senza perdere niente, in poco più di un terzo dello
spazio. L'unica cosa che il modello ha in più è sapere che cosa aspettarsi.

`````

`````{tab} Superiore

Il codice della sezione precedente stampa nat, la letteratura riporta bit per
dimensione, e il ponte fra i due è una divisione:

$$
\text{bit/dim} = \frac{-\log p(\mathbf{x})}{D \ln 2},
$$

con $D$ il numero di componenti del dato ($D = 32 \times 32 \times 3 = 3.072$
per una immagine di CIFAR-10). Su CIFAR-10, che è il banco di prova storico
della famiglia: NICE $4{,}48$; RealNVP $3{,}49$; Glow $3{,}35$; PixelCNN
$3{,}14$; Gated PixelCNN $3{,}03$; PixelRNN $3{,}00$; PixelCNN++ $2{,}92$. E in
cima, a $8{,}00$, il modello che non sa niente.

Una precisazione che la parola «esatta» rischia di far perdere. Un modello
autoregressivo sui 256 livelli dà la probabilità di *quell'immagine lì*, e il
conto è esatto senza aggiunte. Un flusso invece è continuo, e una densità
continua su valori interi è mal posta (l'entropia differenziale di una
distribuzione discreta è $-\infty$, e la verosimiglianza si può gonfiare a
piacere): si aggiunge allora rumore uniforme ai pixel, cioè si
**dequantizza**, e quel che si ottiene è la verosimiglianza esatta dei dati
dequantizzati. Il legame con il numero che interessa lo dà una disuguaglianza
di Jensen {cite}`theis2016note`: la log-verosimiglianza media del modello
continuo sta **sotto** quella del corrispondente modello discreto, quindi i bit
per dimensione riportati per un flusso sono un limite *superiore* al costo di
codifica vero. Conservativo, il che va benissimo, ma non è la stessa cosa.

`````

C'è anche un motivo per cui questo modo di misurare piace, e il capitolo sulle
GAN lo rende evidente per contrasto. Confrontare due GAN richiede il FID, che
richiede una terza rete addestrata da qualcun altro, che a sua volta ha le sue
idee su che cosa sia una fotografia. Confrontare due modelli a verosimiglianza
esatta richiede un numero solo, misurato su dati mai visti, senza giudici
esterni e senza convenzioni.

```{admonition} Un numero solo, ma su che cosa
:class: caution
Va detto subito, perché è il malinteso più diffuso su questa famiglia. Quel
numero non misura quanto le immagini generate sono belle, e i due giudizi
possono divergere fino a diventare quasi indipendenti. Lucas Theis, Aäron van
den Oord e Matthias Bethge {cite}`theis2016note` lo mostrano già nel 2016: la
verosimiglianza media, le stime di Parzen e la qualità visiva dei campioni
sono, in alta dimensione, criteri largamente slegati, e «ottenere buoni
risultati su un criterio non implica necessariamente ottenerne sugli altri». Un
modello può
avere una verosimiglianza eccellente e produrre campioni mediocri, e viceversa.
La loro conclusione è la regola pratica da tenere: un modello generativo va
valutato rispetto all'uso per cui lo si vuole, non rispetto al numero più
comodo da stampare.
```

## Secondo: ordinare, confrontare, scegliere

Il secondo mestiere è meno vistoso e più usato di quanto sembri. Avere
$p(\mathbf{x})$ vuol dire poter mettere in fila delle ipotesi: quale delle due
trascrizioni è più probabile, quale delle tre ricostruzioni di un dato mancante
sta meglio con il resto, quale delle molte spiegazioni di un'osservazione va
preferita. È il ruolo che un modello di densità gioca dentro sistemi più
grandi, dove non genera niente e serve solo a dare voti confrontabili; ed è il
ruolo che i flussi ricoprono, per esempio, dentro l'inferenza variazionale (il
mestiere di approssimare una distribuzione difficile con una che si sa
maneggiare), che è poi il posto da cui vengono {cite}`rezende2015variational`.

## Terzo: riconoscere ciò che è fuori posto, e qui casca l'asino

L'applicazione più ovvia di tutte è questa. Ho addestrato il modello sulle mie
fotografie; adesso me ne arriva una nuova; se il modello le dà una probabilità
bassissima, vuol dire che è roba diversa da quella che ho visto. Rilevamento di
anomalie, controllo qualità, allarme quando il mondo cambia sotto ai piedi del
sistema (è il tema della sezione su monitoraggio e deriva, nel capitolo su
MLOps). Sembra la cosa più solida del mondo.

Non funziona. E il modo in cui non funziona è così netto da essere diventato un
classico.

Eric Nalisnick e colleghi {cite}`nalisnick2019do` addestrano flussi, VAE e
PixelCNN su CIFAR-10, una raccolta di fotografie di cani, camion, cavalli e
altre cose comuni. Poi mostrano a quei modelli SVHN, che sono fotografie di
numeri civici: un'altra raccolta, un altro mondo, immagini che il modello non
ha mai visto e che non somigliano a niente di ciò che ha visto. E misurano la
verosimiglianza. Il risultato, nelle loro parole, è che quei modelli «non
riescono a distinguere immagini di oggetti comuni come cani, camion e cavalli
da quelle di numeri civici, **assegnando una verosimiglianza più alta a queste
ultime** quando il modello è stato addestrato sulle prime».

Non «faticano a distinguere»: sbagliano nella direzione sbagliata, e con
sicurezza. Il modello dichiara più tipica la roba che non ha mai visto.

`````{tab} Elementare

La spiegazione più accreditata è tanto semplice quanto scomoda, e conviene
arrivarci con un esempio che non ha niente a che fare con le immagini.

Immagina un modello addestrato su testi italiani, e supponi di misurargli
quanto trova probabile una pagina. Adesso dagli in pasto una pagina bianca con
scritto in mezzo «aaaaaaaaaaaa». Non è italiano, non l'ha mai visto, è roba
fuori posto sotto ogni criterio. Ma è anche **facilissima**: ogni carattere è
identico al precedente, quindi il modello ci azzecca ogni volta, e la
probabilità che ne esce è altissima. Più alta di quella di una pagina di
italiano vero, che è piena di scelte difficili.

Le fotografie dei numeri civici fanno la stessa cosa: sono immagini più lisce,
più povere di dettaglio, più prevedibili di quelle di un bosco o del pelo di un
cane. Il modello le trova facili, e «facile» per lui vuol dire «probabile».

Ed è qui la lezione, che vale ben oltre questa famiglia: **«quanto è probabile»
e «l'ho già visto» non sono la stessa domanda**. Le abbiamo confuse perché
nella nostra testa vanno insieme, e per un modello no. Chi vuole sapere se un
dato viene dalla distribuzione su cui il modello è stato addestrato deve
chiederlo con un metodo che risponda a *quella* domanda, e non prendere in
prestito il numero che risponde a un'altra.

Un'ultima cosa, che rende la storia meno amara. Un fallimento così è una
buona notizia per chi legge il libro, e non per masochismo: è un esperimento
piccolo, riproducibile, che ha smontato una convinzione diffusa e non
verificata. La convinzione c'era da anni; per smontarla è bastato addestrare
tre modelli su una raccolta di fotografie e misurarli su un'altra.

`````

`````{tab} Superiore

Il fenomeno non è un difetto dell'addestramento: si presenta su modelli ben
addestrati e con buona verosimiglianza sui dati di prova, ed è stabile fra
famiglie diverse (flussi, VAE, PixelCNN), il che rende implausibile che sia un
artefatto architetturale.

Una spiegazione la danno gli autori stessi, restringendo i flussi a
trasformazioni a volume costante, che si prestano a un conto in forma chiusa:
la differenza di verosimiglianza si spiega con la posizione e la varianza dei
dati e con la curvatura del modello. La lettura più citata dopo di allora è che
la densità in alta dimensione sia dominata dalla **complessità** dell'input più
che dalla sua appartenenza alla distribuzione: dati più lisci ricevono
log-densità più alte
per ragioni che hanno poco a che vedere con il supporto della distribuzione di
addestramento, e SVHN, con sfondi uniformi e poche texture, è esattamente
questo rispetto a CIFAR-10. È l'argomento di Serrà e colleghi
{cite}`serra2020input`, che
da una stima della complessità dell'input ricavano un punteggio, leggibile come
un rapporto di verosimiglianze, che regge il confronto con i metodi dedicati.
Ne discende comunque la stessa diagnosi: la densità non è una statistica di
appartenenza, e usarla come tale confonde $p_\theta(\mathbf{x})$ alto con
$\mathbf{x} \in \operatorname{supp} p_{\text{dati}}$.

Va aggiunta una precisazione di geometria in alta dimensione, che rende il
risultato meno paradossale di quanto sembri: la massa di
probabilità di una gaussiana non sta nel punto di densità massima ma in un
guscio a distanza $\approx \sqrt{D}$ dall'origine (in $D = 1024$ la norma di un
campione gaussiano standard vale in media $32{,}0$). Un campione **tipico** non
è quindi un campione ad alta densità, e i due concetti divergono tanto più
quanto $D$ cresce. Cercare l'atipico guardando la densità è, letteralmente,
guardare l'asse sbagliato, ed è la strada che gli stessi autori prendono subito
dopo, nel 2019, sostituendo alla densità un test di **tipicità**
{cite}`nalisnick2019typicality`.

`````

## Il ponte: dai flussi al *flow matching*

Resta un debito, aperto nella sezione precedente e prima ancora dal capitolo
sui modelli di diffusione: la parola «flusso» del *rectified flow*. Adesso la
si può saldare, e la parentela è più stretta di quanto sembri.

Un flusso, come l'abbiamo costruito, è una **composizione di tanti passi
invertibili**, e ogni passo ha dovuto rinunciare a qualcosa per restare
invertibile e per avere un determinante leggibile. Domanda naturale: e se
invece di comporre venti passi grossi ne componessimo infiniti
infinitamente piccoli? La composizione diventa un'equazione differenziale: si
dichiara una **velocità** in ogni punto dello spazio e in ogni istante, si
lascia scorrere, e il punto di partenza arriva dove deve. La deformazione non è
più fatta di gradini, è un movimento continuo.

Il guadagno è enorme. In quel limite il logaritmo del determinante si riduce
alla **traccia** della jacobiana, cioè alla sola diagonale: è il risultato
delle *neural ODE* di Ricky Chen e colleghi {cite}`chen2018neural`, che al posto di una pila di
strati mettono l'integrazione di un'equazione differenziale. Quella traccia
però va ancora calcolata, e calcolarla per intero costa quanto il quadrato
delle dimensioni; a toglierla di mezzo arriva FFJORD
{cite}`grathwohl2019ffjord`, che la **stima** invece di calcolarla: uno
stimatore stocastico dà una stima non distorta della log-densità «permettendo
architetture di rete senza restrizioni». Tutti i vincoli di questa sezione
cadono in un colpo: niente accoppiamenti, niente metà ferme, niente
determinanti triangolari da costruire a mano.

Il prezzo, però, si sposta: per avere la verosimiglianza bisogna **risolvere
l'equazione differenziale**, ogni volta, e per ogni esempio. Da vincolo
architetturale a costo di calcolo.

E qui arriva la mossa che ha vinto, ed è una rinuncia. Il **flow matching**
{cite}`lipman2023flow` osserva che, se quello che si vuole è generare, la
verosimiglianza durante l'addestramento non serve affatto: basta che la
velocità sia quella giusta. E la velocità giusta si può insegnare per
regressione, mostrando alla rete coppie (punto, velocità) prese da traiettorie
costruite a tavolino, senza mai integrare niente e senza mai calcolare una
traccia. Il *rectified flow* {cite}`liu2023rectified` sceglie come traiettorie
le linee dritte, ed è quello che Stable Diffusion 3 usa, come racconta il
capitolo precedente.

Il cerchio si chiude con un'ironia che vale la pena registrare. La famiglia di
questo capitolo esiste per una proprietà sola, la verosimiglianza esatta; il suo
discendente più usato oggi ha vinto **buttandola via**, e tenendo solo la parte
geometrica, il movimento. La proprietà, però, resta lì: un modello a flow
matching, se qualcuno vuole pagare il conto dell'equazione differenziale, la
verosimiglianza la sa ancora dare. È una rinuncia di comodo, non di struttura,
ed è per questo che vale la pena sapere da dove viene la parola.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- **Sapere quanto è probabile un dato è la stessa cosa che saperlo
  comprimere.** Il numero di bit che serve per scriverlo con il codice migliore
  è $-\log_2$ della sua probabilità, e non per analogia: è quella grandezza lì,
  a un paio di bit su tutto il file. Per questo qui la qualità si misura in
  bit: su una figurina di CIFAR-10 un buon modello costa meno di 3 bit per
  numero, contro gli 8 di chi non sa niente.
- Quel numero però **non dice se le immagini generate sono belle**. I due
  giudizi possono andare per conto loro, ed è documentato dal 2016: un modello
  può avere una verosimiglianza ottima e campioni mediocri.
- L'uso più ovvio, «se la probabilità è bassa allora è roba che non ho mai
  visto», **non funziona**. Modelli addestrati su fotografie di cani e camion
  danno una probabilità *più alta* a fotografie di numeri civici, che non
  hanno mai visto. Il motivo: quelle immagini sono più lisce, e per un modello
  «facile» vuol dire «probabile». «Quanto è probabile» e «l'ho già visto» sono
  due domande diverse.
- Il seguito della storia: se invece di comporre tanti passi grossi si lascia
  scorrere un movimento continuo, tutti i vincoli di progetto cadono, ma
  calcolare la probabilità diventa caro. Il metodo che oggi disegna le immagini
  (il *flow matching*, quello di Stable Diffusion 3) ha vinto **rinunciando** a
  calcolarla e tenendo solo il movimento. Ecco da dove viene quella parola,
  «flusso».
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- $-\log_2 p(\mathbf{x})$ è la lunghezza di codice ottima: verosimiglianza e
  compressione sono la stessa quantità, ed è il motivo per cui la metrica
  standard della famiglia sono i **bit per dimensione**, $-\log p(\mathbf{x}) /
  (D \ln 2)$. Su CIFAR-10 si va da $4{,}48$ (NICE) a $2{,}92$ (PixelCNN++),
  contro $8{,}00$ del modello che non sa niente; per un flusso quel numero è un
  limite **superiore**, perché la densità è misurata su dati dequantizzati.
- Verosimiglianza e qualità dei campioni sono criteri largamente indipendenti
  in alta dimensione {cite}`theis2016note`: vanno scelti in funzione
  dell'applicazione, non estrapolati l'uno dall'altro.
- **Fallimento OOD** {cite}`nalisnick2019do`: flussi, VAE e PixelCNN addestrati
  su CIFAR-10 assegnano log-densità **più alta** a SVHN. La densità in alta
  dimensione è dominata dalla complessità dell'input, non dall'appartenenza al
  supporto; e in $\mathbb{R}^D$ l'insieme tipico non coincide con la regione ad
  alta densità (il guscio a $\approx\sqrt{D}$). $p_\theta$ alto non implica
  $\mathbf{x} \in \operatorname{supp} p_{\text{dati}}$.
- **Limite continuo** (*neural ODE* {cite}`chen2018neural`): la composizione di
  passi discreti diventa una ODE sul campo di velocità e
  $\log\lvert\det\mathbf{J}\rvert$ si riduce a
  $\int \operatorname{tr} (\partial v / \partial \mathbf{x})\, dt$. La traccia
  esatta costa $\mathcal{O}(D^2)$; lo stimatore stocastico di FFJORD
  {cite}`grathwohl2019ffjord` la porta a $\mathcal{O}(D)$ e toglie così ogni
  vincolo architetturale, al prezzo dell'integrazione numerica.
- **Flow matching** {cite}`lipman2023flow` rinuncia alla verosimiglianza in
  addestramento e regredisce direttamente il campo di velocità su cammini
  prescritti; il *rectified flow* {cite}`liu2023rectified` sceglie cammini
  rettilinei ed è la scelta di Stable Diffusion 3. La verosimiglianza resta
  ottenibile a posteriori risolvendo l'ODE: è una rinuncia di convenienza, non
  di struttura.
```

`````
