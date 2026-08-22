# Dai sistemi dinamici a S4

Nel 1960 l'ingegnere ungherese-americano Rudolf Kálmán propose un modo nuovo
di descrivere un sistema che evolve nel tempo: non un groviglio di equazioni
sulle sole grandezze osservabili, ma una manciata di variabili nascoste (lo
**stato**) che riassumono tutto ciò che del passato serve per prevedere il
futuro. Da quella *rappresentazione in spazio degli stati* nacque il filtro di
Kalman, che di lì a pochi anni avrebbe guidato le capsule Apollo verso la
Luna, stimando posizione e velocità da misure rumorose. È un'idea di teoria
del controllo e di elaborazione dei segnali, lontana anni luce dal linguaggio
naturale.

Eppure è la stessa idea che, mezzo secolo dopo, ha dato una seconda strada
verso l'obiettivo che questo capitolo condivide con il precedente: un modello di
sequenze che si addestri in parallelo come un Transformer e che poi, una volta
in servizio, spenda per ogni parola sempre la stessa quantità di tempo e di
memoria, come una rete ricorrente. Usare un modello già addestrato, in gergo,
si dice fare **inferenza**, e da qui in avanti capiterà spesso di leggerlo.

Nel capitolo precedente ci siamo
arrivati partendo dall'attenzione, smontandone il pezzo che costava di più
finché quello che restava era, di nuovo, una memoria di taglia fissa
aggiornata parola per parola. Qui partiamo dal lato opposto (un sistema
dinamico continuo) e arriviamo, sorprendentemente, quasi allo stesso posto.
Alla fine del capitolo Mamba-2 chiuderà il cerchio, mostrando che le due strade
portavano alla stessa città.

## Un sistema che evolve nel tempo

Il mattone di partenza è il più semplice dei sistemi dinamici: qualcosa che
riceve, si modifica e restituisce, senza salti. Entra un segnale, dentro c'è
uno stato che cambia in continuazione, ed esce un altro segnale. Le tre
grandezze sono legate da due regole, che dicono l'una come lo stato cambia da
un istante al successivo e l'altra come si legge l'uscita a partire dallo
stato. La prima delle due mette in relazione lo stato con la sua velocità di
variazione, cioè con quanto sta cambiando in questo momento: quella velocità è
la **derivata** incontrata nella
{doc}`sezione su analisi e ottimizzazione </Matematica/analisi-ottimizzazione>`,
e qui basta leggerla così. La seconda è più semplice, e di derivate non ne ha
dentro: dice come, dal valore dello stato, si ricava l'uscita. Regole della
prima forma, in matematica, si chiamano **equazioni differenziali**.

`````{tab} Elementare

Una vasca da bagno, con il rubinetto aperto e lo scarico non del tutto
chiuso. Il *livello dell'acqua* è lo stato: riassume tutta la storia passata
di quanto hai aperto il rubinetto, senza bisogno di ricordarla minuto per
minuto. Il rubinetto è l'ingresso: quando lo apri di più, il livello sale. Lo
scarico è la dinamica interna: se smetti di versare acqua, il livello cala da
solo, un po’ alla volta. E ciò che *leggi* (magari un galleggiante collegato a
un ago) è l'uscita, che dipende dal livello. A quell'ago se ne potrebbe
affiancare un secondo, attaccato direttamente alla manopola del rubinetto:
direbbe quanto è aperta in questo istante, senza memoria e senza ritardo. È una
scorciatoia dall'ingresso alla lettura, e si tiene da parte, perché la parte
interessante è quella che passa per la vasca.

Quanto in fretta il livello scende, a rubinetto chiuso, dipende da quanto è
aperto lo scarico: largo, e la vasca dimentica in un minuto; stretto, e si
ricorda per un'ora; tappato, e non dimentica più niente, perché tutta l'acqua
che hai versato resta lì. E la vasca non cala soltanto: spingi l'acqua verso un
capo e per qualche secondo ondeggia avanti e indietro prima di quietarsi. Ci
sono poi sistemi che crescono da soli, come il microfono avvicinato troppo alla
cassa che lo amplifica: il fischio si alza da sé finché qualcuno non lo
allontana. Calare, ondeggiare o crescere è quello che un sistema del genere fa
quando lo lasci in pace, e a deciderlo è la sua regola interna, non chi lo
alimenta.

Un'eco in una valle funziona allo stesso modo: gridi (ingresso), il suono
rimbomba e si spegne gradualmente (stato che decade), e quello che senti è una
versione attenuata e ritardata del grido (uscita). In tutti questi casi lo stato
è una fotografia compatta del passato: sapendo il livello dell'acqua *adesso* e
cosa farai col rubinetto *da adesso in poi*, sai prevedere il futuro senza
riavvolgere tutta la storia.

`````

`````{tab} Superiore

Un sistema lineare a tempo continuo, a ingresso e uscita scalari, si scrive

$$
\mathbf{h}'(t) = \mathbf{A}\,\mathbf{h}(t) + \mathbf{B}\,u(t), \qquad y(t) = \mathbf{C}\,\mathbf{h}(t) + D\,u(t),
$$

dove $u(t)\in\mathbb{R}$ è l'ingresso, $y(t)\in\mathbb{R}$ l'uscita e
$\mathbf{h}(t)\in\mathbb{R}^{N}$ lo **stato** interno di dimensione $N$. Le tre matrici
hanno ruoli distinti: $\mathbf{A}\in\mathbb{R}^{N\times N}$ è la **dinamica interna**,
governa come lo stato evolve da solo, in assenza di ingresso (gli autovalori
di $\mathbf{A}$ decidono se lo stato decade, oscilla o esplode);
$\mathbf{B}\in\mathbb{R}^{N\times 1}$ è la **matrice d'ingresso**, dice come il segnale
in arrivo si scrive nello stato; $\mathbf{C}\in\mathbb{R}^{1\times N}$ è la **matrice
d'uscita** (legge lo stato e produce il segnale in uscita). Il termine
$D\,u(t)$ è una scorciatoia diretta dall'ingresso all'uscita, una *skip
connection*: nei modelli che vedremo lo si tiene a parte (equivale a un
residuo) e ci si concentra sulla parte con memoria, ponendo spesso $D=0$ nella
derivazione.

Questa è la *rappresentazione in spazio degli stati* della teoria del controllo:
lo stato $\mathbf{h}(t)$ è, per costruzione, una statistica sufficiente del passato. La
derivata $\mathbf{h}'(t)$ dice come lo stato cambia istante per istante, spinto in parte
dalla propria inerzia ($\mathbf{A}\,\mathbf{h}$) e in parte dal mondo esterno ($\mathbf{B}\,u$).

Una parola sulla notazione, perché qui si incrociano due tradizioni che danno
alle stesse cose lettere diverse, e chi arriva dall'una legge male le formule
dell'altra. Il controllo chiama $\mathbf{x}$ lo stato e $u$ l'ingresso, ed è la
convenzione con cui S4 ({cite}`gu2022s4`) scrive ancora il suo sistema:
$\mathbf{x}'(t) = \mathbf{A}\,\mathbf{x}(t) + \mathbf{B}\,u(t)$. La letteratura
che tratta gli SSM come strati di rete neurale la ribalta e chiama $\mathbf{h}$
lo stato, $x$ l'ingresso: lo fa Mamba ({cite}`gu2023mamba`), con
$\mathbf{h}'(t) = \mathbf{A}\,\mathbf{h}(t) + \mathbf{B}\,x(t)$, ed è la forma
che il campo ha adottato. Non è una questione di gusto: dentro una rete la
lettera $x$ è già presa dal dato che entra nello strato, e in un SSM quel dato
è esattamente l'ingresso della ricorrenza, così che uno stato di nome $x$
finirebbe a dividere la lettera con il proprio ingresso nella stessa equazione.
Il libro segue la seconda convenzione, che è anche quella con cui $\mathbf{h}$
indica lo stato nascosto fin dalle reti ricorrenti. L'unico residuo della prima
è la $u(t)$ del sistema a tempo continuo: appena discretizzeremo, con
l'ingresso diventato una sequenza di campioni, prenderà il nome di $x_t$.

`````

Finora tutto è continuo: il tempo scorre senza gradini. Ma una frase è una
sequenza di **token** (i pezzetti in cui il testo viene diviso, i «passi» di
cui parlavamo), un segnale audio è una sequenza di campioni: dati **discreti**,
uno dopo l'altro. Per usare questo sistema su una sequenza dobbiamo prima
tradurlo dal continuo al discreto.

## Dal continuo al discreto

Il passaggio si chiama **discretizzazione**: al posto di seguire il
cambiamento istante per istante, si va avanti a salti, di lunghezza fissa.
Chiamiamo $\Delta$ la durata di un salto (il tempo che passa tra una misura e
la successiva) e riscriviamo il sistema in modo che vada di stato in stato,
invece di scivolare con continuità.

C'è un punto su cui conviene essere espliciti, perché è una fonte comune di
confusione: **non esiste un solo modo di discretizzare**. Quello che succede
*tra* una misura e l'altra non lo si è visto, e va indovinato; regole diverse
lo indovinano in modi diversi, e i due modelli principali del capitolo ne
usano due che non vanno scambiate.

Il modo più rapido di tenerle separate è pensarle come due figure
geometriche: quella di S4 è un trapezio, quella di Mamba (l'altro protagonista
del capitolo) è un rettangolo. Vediamo perché, e come si chiamano.

`````{tab} Elementare

Discretizzare è come campionare un segnale continuo: invece di seguire l'acqua
della vasca in ogni istante, ne misuri il livello a intervalli regolari (ogni
$\Delta$ secondi) e ti chiedi come passare da una misura alla successiva. Quel
$\Delta$ non dice quante misure prendere, che sono già decise (una per parola):
dice quanto tempo passa fra l'una e l'altra, cioè quanta storia della vasca sta
dentro un salto solo. Con $\Delta$ piccolo, fra un salto e l'altro lo scarico
porta via poco e la memoria si allunga; con $\Delta$ grande, fra un salto e
l'altro succede molto e di quel che c'era prima resta poco. È la manopola che
decide quanto in fretta il sistema dimentica, e più avanti Mamba la girerà a
ogni parola.

Le due regole (bilineare e ZOH) sono due modi diversi di indovinare cosa
succede *tra* un campione e l'altro, e basta un po’ di geometria a
raccontarli. Immagina di dover calcolare quanta acqua è entrata nella vasca
durante il tratto che non hai visto, sapendo solo l'apertura del rubinetto
all'inizio e alla fine. Lo *zero-order hold*, la scelta di Mamba, tiene
l'apertura ferma per tutto il tratto, come se il rubinetto non si fosse mosso:
la figura da misurare è un **rettangolo**. La bilineare, la scelta di S4, tiene
conto di come l'apertura è cambiata da un capo all'altro: il lato di sopra si
inclina, e la figura diventa un **trapezio**. Il risultato è lo stesso tipo di
regola passo dopo passo; cambia quanto errore ti porti dietro a ogni salto, e
l'errore, a forza di salti, si accumula.

Il rettangolo di Mamba tornerà una seconda volta, e conviene saperlo fin
d'ora: anche il conto di quanta acqua, di quella entrata, finisce davvero nella
memoria, Mamba lo fa a rettangoli. È il pezzo che Mamba-3, il modello più
recente della famiglia, rifarà a trapezi.

`````

`````{tab} Superiore

Discretizzare significa ricavare, dalle matrici continue $\mathbf{A}$ e $\mathbf{B}$ e dal passo
$\Delta$, le matrici **discrete** $\bar{\mathbf{A}}$ e $\bar{\mathbf{B}}$ tali che la ricorrenza
$\mathbf{h}_t = \bar{\mathbf{A}}\,\mathbf{h}_{t-1} + \bar{\mathbf{B}}\,x_t$ approssimi l'evoluzione continua
($x_t$ è l'ingresso campionato al passo $t$).

Lo **zero-order hold** assume che l'ingresso resti costante entro ciascun
intervallo $\Delta$ e integra esattamente il sistema su quel tratto:

$$
\bar{\mathbf{A}} = \exp(\Delta \mathbf{A}), \qquad
\bar{\mathbf{B}} = (\Delta \mathbf{A})^{-1}\big(\exp(\Delta \mathbf{A}) - \mathbf{I}\big)\,\Delta \mathbf{B} .
$$

Qui $\exp(\cdot)$ è l'esponenziale **di matrice**, non elemento per elemento.
L'inversa $(\Delta \mathbf{A})^{-1}$ è apparente: la combinazione vale
$\Delta\,\varphi_1(\Delta \mathbf{A})\,\mathbf{B}$ con $\varphi_1(z)=\sum_{k\ge 0} z^k/(k+1)!$,
una serie definita anche quando $\mathbf{A}$ è singolare (per $\mathbf{A}$ diagonale con un
autovalore nullo la formula scritta con l'inversa non si può valutare, la serie
sì). In codice si usa la serie, o `expm1`, non il quoziente: per $a$ piccolo la
differenza $e^{\Delta a}-1$ perde le cifre significative.

Se $\mathbf{A}$ è diagonale, ogni suo autovalore $a$ si discretizza per conto suo:
$\bar{a} = e^{\Delta a}$ e $\bar{b} = \frac{e^{\Delta a}-1}{a}\,b$, ben definito
anche nel limite $a\to 0$, dove vale $\Delta b$ (è $\varphi_1(0)=1$). È la
scelta di Mamba, con una precisazione: lo ZOH
vale per la transizione, $\bar{\mathbf{A}} = \exp(\Delta \mathbf{A})$, mentre per l'ingresso
l'implementazione adotta la semplificazione al prim'ordine (Eulero)
$\bar{\mathbf{B}} = \Delta \mathbf{B}$, che dello ZOH è il troncamento per $\Delta$ piccolo. La
ritroveremo nel codice della prossima sezione.

La **trasformazione bilineare** approssima invece l'esponenziale con la sua
frazione razionale del primo ordine (la regola del trapezio), ottenendo

$$
\bar{\mathbf{A}} = \Big(\mathbf{I} - \tfrac{\Delta}{2}\mathbf{A}\Big)^{-1}\Big(\mathbf{I} + \tfrac{\Delta}{2}\mathbf{A}\Big),
\qquad
\bar{\mathbf{B}} = \Big(\mathbf{I} - \tfrac{\Delta}{2}\mathbf{A}\Big)^{-1}\Delta \mathbf{B} .
$$

È la scelta di S4, e vale la pena ricordarsela: attribuire lo ZOH a S4 è un
errore che si trova spesso in giro. In entrambi i casi la $\mathbf{C}$ resta
invariata ($\bar{\mathbf{C}}=\mathbf{C}$), e i
parametri effettivi del modello sono la quaterna $(\Delta, \mathbf{A}, \mathbf{B}, \mathbf{C})$: le matrici
continue più il passo, da cui si generano le matrici discrete. Il passo $\Delta$
non è un dettaglio: fissa la *scala temporale* del sistema, cioè quanto in fretta
lo stato dimentica.

`````

## Due facce della stessa medaglia: ricorrenza e convoluzione

Ora arriva il fatto che rende speciali questi modelli. Le regole del sistema
sono tre: come lo stato decade da solo, come l'ingresso vi entra, come si legge
l'uscita. Finché queste tre regole sono **le stesse a ogni passo** (nel gergo
della teoria dei segnali il sistema si dice *lineare e tempo-invariante*, in
sigla LTI) la stessa uscita si può calcolare in due modi che sembrano diversi
e non lo sono: una forma **ricorrente**, un passo alla volta, e una forma
**convoluzionale**, tutta la sequenza in un colpo.

`````{tab} Elementare

È la stessa doppia natura che abbiamo visto nel capitolo scorso con l'attenzione
lineare, dove un unico calcolo si poteva leggere in due modi: "passo dopo passo"
oppure "tutto insieme". Qui succede l'identico.

Da un lato la forma **ricorrente**: parti dallo stato, aggiungi il nuovo
ingresso, ottieni il nuovo stato, leggi l'uscita, e ripeti. Un token alla
volta, con una quantità di memoria che non cresce mai: perfetta per generare
testo o processare un flusso audio in tempo reale. È il modo di lavorare di
una RNN.

Dall'altro la forma **convoluzionale**: se il sistema non cambia nel tempo, si
può dimostrare che l'intera uscita è una singola convoluzione dell'ingresso
con un filtro fisso. E la convoluzione la conosciamo dalle reti convoluzionali
del capitolo sul deep learning: un filtro che scorre lungo il segnale e a ogni
posizione moltiplica i propri pesi per i valori che ha sotto, poi somma. Due
differenze. La prima è che qui il filtro è lungo quanto tutta la sequenza, non
una finestrella di pochi elementi. La seconda è che nessuno lo scrive a mano:
si ricava, con un conto, dalle tre regole del sistema. Ed è una buona notizia,
perché il modello ha da imparare le tre regole, che sono poche, e non i numeri
del filtro, che sono tanti quanto il testo è lungo. Quei numeri, uno per
posizione, dicono quanto una parola di venti o di mille passi fa conta ancora
sull'uscita di adesso. Il vantaggio è che una convoluzione si calcola in un
colpo solo, in parallelo su tutta la sequenza: proprio ciò che serve per
sfruttare le GPU in addestramento.

Morale: si **addestra** in forma convoluzionale (veloce, parallela) e si fa
**inferenza** in forma ricorrente (economica, una parola alla volta). La stessa
funzione, due vestiti diversi a seconda dell'occasione.

Tutto questo regge a una condizione: il filtro è uno solo, lo stesso dalla
prima parola all'ultima. Se le tre regole cambiassero da una parola alla
successiva, un filtro da calcolare una volta per tutte non ci sarebbe più, e
resterebbe la sola strada passo dopo passo. Mamba prenderà proprio quella
deviazione, sapendo quel che costa.

`````

`````{tab} Superiore

La forma **ricorrente** srotola la ricorrenza discreta:

$$
\mathbf{h}_t = \bar{\mathbf{A}}\,\mathbf{h}_{t-1} + \bar{\mathbf{B}}\,x_t, \qquad y_t = \mathbf{C}\,\mathbf{h}_t .
$$

Ogni passo costa $O(N^2)$ (o $O(N)$ se $\bar{\mathbf{A}}$ è diagonale) e la memoria è
$O(N)$, **costante** nella lunghezza della sequenza: è l'inferenza a costo fisso
per token tipica delle RNN.

La forma **convoluzionale** si ottiene sostituendo ripetutamente la ricorrenza
in se stessa, con stato iniziale nullo:

$$
y_t = \sum_{j=0}^{t} \mathbf{C}\,\bar{\mathbf{A}}^{\,j}\,\bar{\mathbf{B}}\;x_{t-j}
    = (\mathbf{x} * \bar{\mathbf{K}})_t ,
$$

cioè una convoluzione tra l'ingresso e un **kernel** (o *SSM convolution kernel*)

$$
\bar{\mathbf{K}} = \big(\mathbf{C}\bar{\mathbf{B}},\; \mathbf{C}\bar{\mathbf{A}}\bar{\mathbf{B}},\; \mathbf{C}\bar{\mathbf{A}}^2\bar{\mathbf{B}},\;
\dots,\; \mathbf{C}\bar{\mathbf{A}}^{\,k}\bar{\mathbf{B}},\; \dots\big) ,
$$

dove $\bar{\mathbf{K}}$ è un filtro causale lungo quanto la sequenza. Calcolata questa
volta sola, l'uscita $\mathbf{y} = \mathbf{x} * \bar{\mathbf{K}}$ si ottiene per l'intera sequenza in
parallelo, con la FFT in tempo $O(L \log L)$ ($L$ è la lunghezza). Il termine
$\mathbf{C}\bar{\mathbf{A}}^{\,j}\bar{\mathbf{B}}$ misura quanto un ingresso di $j$ passi fa pesa ancora
sull'uscita di adesso: è la memoria del sistema, e decade con le potenze
$\bar{\mathbf{A}}^{\,j}$.

L'equivalenza $ \text{ricorrenza} \equiv \text{convoluzione} $ vale **solo**
perché $\bar{\mathbf{A}}, \bar{\mathbf{B}}, \mathbf{C}$ sono costanti nel tempo: è la tempo-invarianza a
garantire che il kernel $\bar{\mathbf{K}}$ sia unico e fisso. Quando, con Mamba, faremo
dipendere questi parametri dall'ingresso, il sistema cesserà di essere LTI, il
kernel di convoluzione fisso svanirà, e resterà solo lo scan ricorrente.

`````

Questa dualità è esattamente lo stesso trucco che ha animato il capitolo
sull'attenzione lineare: un'unica funzione con una forma parallela per
l'addestramento e una forma ricorrente per l'inferenza. Che due strade così
diverse (una nata dall'attenzione, l'altra dai sistemi dinamici) approdino
alla stessa struttura non è un caso, come vedremo. La
{numref}`fig-ssm-forma-duale` mostra le due facce affiancate.

```{figure} ../figures/ssm-forma-duale.svg
:name: fig-ssm-forma-duale
:alt: "A sinistra, sotto il titolo Forma ricorrente (inferenza, costo costante per passo), due riquadri uguali marcati SSM in fila: fra l'uno e l'altro corre la freccia dello stato h_t, dal basso entra in ciascuno il proprio ingresso x_t e verso l'alto ne esce la propria uscita y_t; una freccia tratteggiata a destra dice che la stessa cella si ripete lungo la sequenza. Sotto il disegno, le due formule della ricorrenza: h aggiornato come A-bar per h più B-bar per x, e y uguale a C per h. A destra, sotto il titolo Forma convoluzionale (addestramento, in parallelo), un unico riquadro tratteggiato lungo quanto tutta la sequenza di ingresso, il kernel K-bar, con una freccia per ogni posizione che scende sulla riga delle uscite: tutta l'uscita y in un colpo solo. Sotto, la definizione del kernel come (C B-bar, C A-bar B-bar, ...) e y uguale a x convoluto K-bar. Fra le due viste il simbolo ≡ (identicamente uguale), con sotto la scritta \"stessa funzione\" e la precisazione \"(sistema invariante, LTI)\"."
:width: 85%

Le due facce della stessa macchina, quando le sue regole non cambiano da un
passo all'altro. A sinistra si va **passo dopo passo**: lo stato si aggiorna
una parola alla volta, ed è il modo economico per generare. A destra si fa
**tutto insieme**: un unico filtro, lungo quanto la sequenza, produce tutte le
uscite in una volta sola, ed è il modo veloce per addestrare. Il simbolo al
centro dice che non sono due calcoli diversi: è lo stesso, scritto in due modi.
```

## HiPPO e S4: ricordare a lungo

C'è un problema che abbiamo scavalcato. Nei modelli veri lo stato non è un
numero solo, come il livello della vasca: è un pugno di numeri, qualche decina,
che insieme fanno il riassunto. Restano pochi, però, e la domanda è perché un
riassunto così piccolo dovrebbe ricordare qualcosa avvenuto migliaia di passi
prima.

Torniamo alla vasca. A ogni passo lo scarico porta via una fetta di ciò che c'è
dentro, e di quel che era entrato all'inizio sopravvive ogni volta solo una
frazione della frazione di prima. Facciamo che a ogni passo ne resti il novanta
per cento, che come scarico è già lento: dopo cinquanta passi di quella prima
acqua è rimasto $0{,}9^{50}$, cioè circa mezzo per cento; e siccome altri
cinquanta passi rifanno lo stesso lavoro, dopo cento passi ne resta mezzo per
cento di mezzo per cento. Polvere. Una RNN classica soffre esattamente
di questo, ed è la ragione per
cui sono nate LSTM e GRU, che aggiungono dei **cancelli** (in inglese *gate*:
piccole valvole apprese che decidono, a ogni passo, quanto lasciar passare e
quanto trattenere). Per gli SSM la risposta non sta in nuovi cancelli, ma nella
**scelta della regola con cui lo stato decade**: fatta a caso, la memoria è
corta; costruita con criterio, può essere lunghissima.

`````{tab} Elementare

Un romanzo lunghissimo, e una pagina sola di appunti da tenere aggiornata
mentre lo leggi. La pagina non si allunga mai: per far entrare una riga nuova,
una vecchia deve stringersi. Se ogni frase nuova cancella la precedente, alla
fine ti resta in mano solo l'ultimo capitolo.

C'è un modo migliore di riempirla, e nasce da una domanda precisa: fra tutte le
pagine che stanno in quello spazio, quale somiglia di più al romanzo intero? La
risposta ha un nome, **HiPPO** (la sigla è inglese, e sciolta suona «operatori
di proiezione polinomiale di ordine alto»). La pagina viene su a strati: l'idea
generale, gli snodi principali, i dettagli delle ultime righe. Il passato
lontano si assottiglia e non sbiadisce del tutto.

Quella pagina arriva già impostata prima che la lettura cominci, con lo spazio
ripartito fra gli strati. Chi comincia da lì ha la memoria lunga senza fare
altro; chi comincia da una pagina bianca qualunque ricorda le ultime frasi e
basta, e continuare a leggere non gliela allunga.

**S4** prende quella pagina impostata e ci aggiunge la velocità. Per allenarsi
in fretta gli serve sapere in anticipo quanto ogni frase già letta pesa
sull'appunto di adesso: un elenco lungo quanto il romanzo, e ricavarlo voce per
voce vuol dire ripercorrere la storia da capo ogni volta. Ridisegnare la pagina
per renderla comoda da calcolare non era un'opzione: chi tocca quel disegno
butta via la memoria lunga insieme a lui. La regolarità, per fortuna, c'era
già: l'impostazione di HiPPO è fatta di una parte tutta regolare più una
piccola correzione, e da lì l'elenco intero esce in blocco. La memoria gliela
dà la pagina di partenza, la velocità la regolarità: senza tutt'e due sarebbe
rimasto un esercizio.

Con tutt'e due è il primo modello che risolve **Path-X**, la prova più dura del
*Long Range Arena*, la gara sulle dipendenze a lunghissimo raggio. Si guarda
un'immagine di 128 pixel per 128, letta un pixel alla volta, in fila come se
fosse un testo: 128 per 128 fanno $16\,384$ passi. Alla fine si deve dire se
due puntini sono uniti da un tratto oppure no, e per rispondere bisogna tenere
insieme parti dell'immagine lontanissime nella fila. Lì i Transformer restavano
al livello di chi tira a indovinare. Sulle altre prove, dalle immagini al
linguaggio, il distacco si accorcia senza sparire.

Una virtù, però, per strada si perde. Il modo di prendere appunti di HiPPO non
si cura di quanto sia lungo il romanzo: cento pagine o mille, la pagina resta
una e si riadatta da sé man mano che la storia si allunga. S4 lo congela in una
macchina che procede a passi tutti uguali, e da quel momento la distanza su cui
ricordare va decisa prima di cominciare a leggere. Il rimedio è non deciderla
una volta sola: la manopola che regola quanta storia entra in un salto viene
messa su valori sparpagliati, dai lentissimi ai velocissimi, e le pagine tenute
in parallelo si dividono il lavoro, una sull'ultima riga, una sul capitolo
intero.

`````

`````{tab} Superiore

**HiPPO** (*High-order Polynomial Projection Operators*, Gu et al., 2020,
{cite}`gu2020hippo`) formalizza la compressione online di un segnale come la
sua proiezione ottima su una base di **polinomi ortogonali** (per esempio i
polinomi di Legendre) rispetto a una misura sul passato. Lo stato $\mathbf{h}(t)$
diventa il vettore dei coefficienti di quella proiezione: ricostruisce, nel
modo meno sbagliato possibile, tutto il segnale visto fin lì. La variante
**HiPPO-LegS** (*scaled Legendre*) usa una misura che copre uniformemente
tutta la storia, ed è per costruzione robusta alla scala temporale:
l'operatore originale è tempo-variante (il suo passo è $1/t$, non $\Delta$) e
non ha alcun iperparametro di scala, tanto che dilatare l'ingresso dilata
semplicemente l'uscita. Attenzione a cosa si eredita e cosa no: S4 prende la
*matrice* LegS, non quella robustezza, perché la congela dentro un sistema LTI
con passo $\Delta$ costante. Lì la scala temporale torna a essere fissata da
$\Delta$, che infatti non si sceglie a caso ma si inizializza su una gamma
ampia di ordini di grandezza (tipicamente log-uniforme fra $10^{-3}$ e
$10^{-1}$), proprio per coprire orizzonti di memoria diversi. Il risultato
pratico è una matrice $\mathbf{A}$ specifica (la *matrice HiPPO*) con cui
**inizializzare** l'SSM per dotarlo di memoria a lungo raggio.

**S4** (*Structured State Space Sequence model*, Gu, Goel e Ré, ICLR 2022,
{cite}`gu2022s4`) parte proprio da qui: inizializza $\mathbf{A}$ con HiPPO-LegS. Ma sorge
un ostacolo computazionale. Costruire il kernel $\bar{\mathbf{K}}$ richiede le potenze
$\bar{\mathbf{A}}^{\,j}$ fino a $j = L-1$: farlo direttamente costa $O(N^2 L)$ operazioni,
proibitivo per stati e sequenze grandi. La mossa di S4 non è **imporre** ad $\mathbf{A}$
una struttura, e la distinzione va tenuta ferma: se lo facesse perderebbe
proprio la matrice che dà la memoria lunga, e l'argomento crollerebbe. S4 **dimostra** (Teorema 1 del paper) che le matrici HiPPO una
struttura sfruttabile ce l'hanno già, e che è **normale più basso rango**
(NPLR):

$$
\mathbf{A} = \mathbf{V}\boldsymbol{\Lambda} \mathbf{V}^{*} - \mathbf{P} \mathbf{Q}^{\top},
$$

con $\mathbf{V}$ unitaria, $\boldsymbol{\Lambda}$ diagonale e $\mathbf{P}\mathbf{Q}^{\top}$ una correzione di rango
basso ($\mathbf{P}$ e $\mathbf{Q}$ sono matrici «alte e strette»). Coniugando con $\mathbf{V}$ ci si
riduce alla forma *diagonale più basso rango* (DPLR),
$\boldsymbol{\Lambda} - \tilde{\mathbf{P}}\tilde{\mathbf{Q}}^{*}$, che è quella su cui l'algoritmo lavora.
Attenzione a chi sono gli autovalori: $\boldsymbol{\Lambda}$ raccoglie gli autovalori della
**parte normale**, non quelli di $\mathbf{A}$, e per la HiPPO-LegS i due insiemi non si
somigliano affatto (gli autovalori di $\mathbf{A}$ sono reali, $-1, \dots, -N$; quelli
di $\boldsymbol{\Lambda}$ hanno tutti parte reale $-1/2$ e parti immaginarie che crescono).
Gli autovalori di $\mathbf{A}$ restano dove sono, perché una similitudine non li
sposta: la retta verticale appartiene alla parte normale, quella che resta una
volta scorporata la correzione di rango basso. Il guadagno sta lì: una matrice
normale ha una base di autovettori ortonormale, mentre la base di autovettori
di $\mathbf{A}$ si mal condiziona in fretta al crescere di $N$, tanto da rendere
impraticabile la diagonalizzazione diretta. Per questo l'algoritmo lavora sulla
forma DPLR.

Con questa struttura il kernel non si calcola più elevando a potenza una
matrice piena: lo si ottiene passando alla sua *funzione generatrice* valutata
sulle radici dell'unità, e sfruttando l'identità di Woodbury (per l'inversa di
«diagonale + basso rango») e un kernel di Cauchy. Il costo scende a
quasi-lineare in $N + L$.

Il guadagno non è solo teorico. Generando un token alla volta, S4 non ha una
cache che cresce e produce l'uscita a costo costante, mentre un Transformer
deve rileggere tutto il contesto; e sul **Long Range Arena**, il banco di prova
delle dipendenze a lunghissimo raggio, è il primo modello a risolvere
**Path-X**, il compito su sequenze da $16\,384$ elementi su cui i Transformer
restavano al livello del caso. Il paper dichiara di ridurre nettamente (non di
annullare) il divario di qualità con i Transformer su immagini e linguaggio:
la novità che resta, e che è di sostanza, è che una ricorrenza a stato piccolo
arriva dove l'attenzione non arrivava.

`````

## Tappe verso il linguaggio

S4 dimostrò che uno stato piccolo, ben costruito, poteva competere con
l'attenzione sulle sequenze lunghe. Ma tra quel risultato e Mamba (il modello
che porta gli SSM sul linguaggio in modo convincente) ci sono alcune tappe da
nominare, perché ognuna smonta un pezzo del problema.

`````{tab} Elementare

Tre modelli, tre pezzi del problema. **S5** (2023) semplifica la macchina di S4
e, soprattutto, dimostra che la forma «passo dopo passo» non è condannata a
essere lenta. Sembrerebbe di sì, visto che ogni passo ha bisogno del risultato
del precedente. Il fatto è che due passi consecutivi si possono fondere in un
passo solo, che fa il lavoro di tutti e due; e le fusioni di coppie diverse non
si aspettano fra loro, quindi si possono fare tutte nello stesso momento. Un
giro dimezza i passi rimasti, il giro dopo li dimezza ancora, e in una manciata
di giri si è arrivati in fondo. Il conto per esteso sta nella prossima sezione:
è il trucco che Mamba erediterà.

**H3** (2023) affronta invece la memoria «a richiamo»: ritrovare più avanti
una cosa già letta ("chi era il soggetto di quella frase?"). I modelli di
questa famiglia, trattando ogni parola con la stessa regola, faticavano a
farlo; H3 li aiuta accoppiando due memorie e una **valvola** che dosa quanto
passa dall'una all'altra, così il modello riesce a trattenere un'informazione
finché gli serve.

**Hyena** (2023), infine, prova la via più diretta: se l'ingrediente vincente
è un filtro lungo che scorre su tutta la frase (come quelli delle reti
convoluzionali del capitolo sul deep learning, ma lunghi quanto l'intero
testo), tanto vale imparare direttamente il filtro, senza passare dal sistema
dinamico. Con un accorgimento, però: quei numeri Hyena non se li impara a
memoria uno per uno, che sarebbe un elenco lungo quanto il testo. A disegnare
il filtro mette una piccola rete, così le cose da imparare restano poche come
prima; è cambiato soltanto chi tiene la matita. Funziona quasi come
l'attenzione, costando molto meno.

`````

`````{tab} Superiore

**S5** (Smith, Warrington e Linderman, ICLR 2023, {cite}`smith2023s5`)
semplifica S4 su due fronti. Primo: usa un unico SSM **MIMO** (a più ingressi
e più uscite) con matrice $\mathbf{A}$ **diagonale**, invece di tanti SSM scalari
indipendenti. Secondo, e più importante per il seguito: abbandona la
convoluzione via FFT e calcola la ricorrenza con un **parallel scan** (un
algoritmo che, sfruttando l'associatività della ricorrenza lineare, la calcola
in parallelo in tempo logaritmico nella lunghezza). È il ponte diretto verso
lo scan che sarà il cuore di Mamba: la forma ricorrente smette di essere il
modo "lento", diventa anch'essa parallelizzabile.

**H3** (*Hungry Hungry Hippos*, Fu, Dao et al., ICLR 2023, {cite}`fu2023h3`)
attacca il punto debole degli SSM sul linguaggio: il **recall associativo**,
cioè ritrovare a distanza un'informazione già vista ("chi era il soggetto di
quella frase?"). Un SSM LTI puro fatica a copiare e confrontare token, cosa
che l'attenzione fa con naturalezza. H3 impila **due SSM**, uno a spostamento
(*shift*) e uno diagonale, intervallati da un **gating moltiplicativo**, un
prodotto elemento per elemento tra due rami che permette al modello di
confrontare token vicini e "trattenere" un valore fino a quando serve. Con
l'aggiunta di pochissimi strati di attenzione, gli ibridi basati su H3
arrivano a taglie fino a 2,7 miliardi di parametri e reggono il confronto con
i Transformer.

**Hyena** (Poli, Massaroli et al., ICML 2023, {cite}`poli2023hyena`) tira una
riga di sintesi: se l'ingrediente utile è una convoluzione lunga, la si può
apprendere direttamente. Hyena impila **convoluzioni lunghe implicite**
(filtri lunghi quanto la sequenza, ma parametrizzati da una piccola rete
invece che memorizzati numero per numero) alternate a un **gating controllato
dai dati**. Non è un SSM in senso stretto, ma è imparentato: entrambi
calcolano l'uscita come convoluzione lunga, entrambi girano in tempo
$O(L \log L)$ con la FFT. Hyena mostra che si può avvicinare la qualità
dell'attenzione senza attenzione, con sole convoluzioni.

`````

Restava un limite comune a tutti: essendo LTI, questi modelli trattano ogni
token allo stesso modo, incapaci di *scegliere* cosa ricordare in base al
contenuto. Chi legge una parola importante e chi legge una virgola aggiornano
lo stato con la stessa regola fissa. Rompere questo vincolo (rendere il
sistema **tempo-variante**, capace di selezionare) è il passo che porta a
Mamba, ed è il tema della prossima sezione.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un **SSM** nasce da un sistema che evolve nel tempo, come la vasca con il
  rubinetto aperto e lo scarico socchiuso: il livello dell'acqua è lo **stato**,
  una fotografia compatta del passato che basta a prevedere il futuro. Tre
  ingredienti: come lo stato si muove da solo (la vasca cala, altri sistemi
  ondeggiano o crescono), come l'ingresso lo alza, come si legge l'uscita. È
  l'altra strada verso un modello che regge i testi lunghi senza che il costo
  esploda, complementare all'attenzione lineare del capitolo precedente.
- Per usarlo su una sequenza (parole, campioni audio) bisogna **misurare a
  intervalli regolari** e indovinare cosa succede *tra* un campione e il
  successivo. Le ricette non sono una sola e non vanno confuse: **S4 immagina
  quel tratto come un trapezio**; **Mamba tiene l'ingresso fermo per tutto
  l'intervallo** (è lo *zero-order hold*, la tenuta di ordine zero). Per
  calcolare quanta parte di ciò che entra finisce nella memoria, Mamba si
  accontenta del conto più sbrigativo, a rettangoli: è il pezzo che il modello
  più recente della famiglia, Mamba-3, rifarà a trapezi.
- Finché le regole **non cambiano da un passo all'altro**, lo stesso calcolo si
  può fare in due modi: **passo dopo passo** (un token alla volta, con una
  memoria che non cresce mai: economico per generare) oppure **tutto insieme**,
  come un unico filtro lungo che scorre sulla sequenza (parallelo: perfetto per
  addestrare sulle GPU). Si allena nel secondo modo, si usa nel primo: la stessa
  dualità vista con l'attenzione lineare.
- L'equivalenza regge **solo** finché quelle regole restano fisse: Mamba le farà
  dipendere da ciò che legge, e allora resterà solo il modo passo dopo passo.
- **HiPPO** (Gu et al., 2020) è il modo studiato apposta per riassumere una storia
  lunghissima in pochi numeri, come appunti a più livelli su un romanzo: dice
  da quali numeri **partire** perché uno stato piccolo abbia memoria lunga.
  **S4** (2022) parte da lì e rende il conto efficiente sfruttando lo schema
  regolare di quei numeri, ed è il primo a risolvere Path-X (sequenze da
  $16\,384$ elementi), la prova più dura della gara sulle dipendenze a
  lunghissimo raggio, Long Range Arena.
- Le tappe verso il linguaggio: **S5** (mostra che anche il passo dopo passo si
  può svolgere quasi tutto in parallelo), **H3** (due memorie e una valvola che
  dosa quanto passa dall'una all'altra, per ritrovare a distanza una cosa già
  letta), **Hyena** (impara direttamente il filtro lungo). Preparano Mamba.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Un **SSM** nasce da un sistema dinamico continuo
  $\mathbf{h}'(t)=\mathbf{A}\,\mathbf{h}(t)+\mathbf{B}\,u(t)$, $y(t)=\mathbf{C}\,\mathbf{h}(t)$: lo **stato** $\mathbf{h}$ è una fotografia
  compatta del passato, con $\mathbf{A}$ dinamica interna, $\mathbf{B}$
  ingresso, $\mathbf{C}$ uscita. È l'altra strada verso il tempo lineare,
  complementare all'attenzione lineare del capitolo precedente.
- Per usarlo su sequenze discrete serve un passo $\Delta$ di
  **discretizzazione**: una regola per indovinare cosa succede *tra* un campione
  e il successivo. Di regole ce n'è più d'una e non vanno confuse. **S4 usa la
  bilineare** (immagina quel tratto come un trapezio); **Mamba usa lo
  *zero-order hold*** (ZOH: l'ingresso resta fermo per tutto l'intervallo), che
  gli dà la transizione $\bar{\mathbf{A}}=\exp(\Delta \mathbf{A})$. Per il
  termine d'ingresso, però, l'implementazione di Mamba si accontenta del conto a
  rettangoli, $\bar{\mathbf{B}}=\Delta \mathbf{B}$ (il metodo di Eulero, cioè lo
  ZOH troncato al prim'ordine): è il pezzo che Mamba-3 rifarà a trapezi.
- Se il sistema discretizzato è **tempo-invariante** (LTI), la stessa funzione
  ha due forme equivalenti: **ricorrente**
  $\mathbf{h}_t=\bar{\mathbf{A}}\mathbf{h}_{t-1}+\bar{\mathbf{B}}x_t$
  (inferenza $O(1)$ per passo) e **convoluzionale** $\mathbf{y}=\mathbf{x}*\bar{\mathbf{K}}$
  (addestramento parallelo). Si allena convoluzionale, si inferisce
  ricorrente: la stessa dualità vista con l'attenzione lineare.
- Questa equivalenza vale **solo** se $\bar{\mathbf{A}},\bar{\mathbf{B}},
  \mathbf{C}$ sono costanti: Mamba la romperà rendendoli dipendenti
  dall'ingresso, e allora resterà solo lo scan.
- **HiPPO** (Gu et al., 2020) sceglie $\mathbf{A}$ proiettando la storia su
  polinomi ortogonali: è ciò che dà memoria a lungo raggio a uno stato piccolo.
  **S4** (2022) *dimostra* che quelle matrici sono già **normali più basso
  rango** e, coniugando, si riduce a **diagonale + basso rango**: il kernel si
  calcola in tempo quasi-lineare in $N+L$. È il primo a risolvere Path-X
  (sequenze da $16\,384$ elementi) su Long Range Arena.
- Le tappe verso il linguaggio: **S5** (SSM MIMO + parallel scan), **H3** (due
  SSM + gating per il recall associativo), **Hyena** (convoluzioni lunghe
  implicite). Preparano Mamba.
```

`````
