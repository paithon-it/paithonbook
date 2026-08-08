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
verso un vecchio obiettivo di questo capitolo: un modello di sequenze che si
addestri in parallelo come un Transformer e faccia inferenza a costo costante
per token come una rete ricorrente. Nel capitolo precedente ci siamo arrivati
partendo dall'attenzione: abbiamo sostituito la softmax con un prodotto tra
funzioni, e l'attenzione è diventata una RNN lineare a stato fisso. Qui
partiamo dal lato opposto (un sistema dinamico continuo) e arriviamo,
sorprendentemente, quasi allo stesso posto. Stessa meta, radice diversa. Alla
fine del capitolo Mamba-2 chiuderà il cerchio, mostrando che le due strade
portavano alla stessa città.

## Un sistema che evolve nel tempo

Il mattone di partenza è il più semplice dei sistemi dinamici: un sistema
**lineare a tempo continuo**. Un segnale $u(t)$ entra, uno stato interno $h(t)$
evolve, un segnale $y(t)$ esce. La regola che lega le tre grandezze è un paio di
equazioni differenziali lineari.

`````{tab} Elementare

Pensa a una vasca da bagno con il rubinetto aperto e lo scarico non del tutto
chiuso. Il *livello dell'acqua* è lo stato: riassume tutta la storia passata
di quanto hai aperto il rubinetto, senza bisogno di ricordarla minuto per
minuto. Il rubinetto è l'ingresso: quando lo apri di più, il livello sale. Lo
scarico è la dinamica interna: se smetti di versare acqua, il livello cala da
solo, un po' alla volta. E ciò che *leggi* (magari un galleggiante collegato a
un ago) è l'uscita, che dipende dal livello.

Un eco in una valle funziona allo stesso modo: gridi (ingresso), il suono
rimbomba e si spegne gradualmente (stato che decade), e quello che senti è una
versione attenuata e ritardata del grido (uscita). In tutti questi casi lo stato
è una fotografia compatta del passato: sapendo il livello dell'acqua *adesso* e
cosa farai col rubinetto *da adesso in poi*, sai prevedere il futuro senza
riavvolgere tutta la storia.

`````

`````{tab} Superiore

Un sistema lineare a tempo continuo, a ingresso e uscita scalari, si scrive

$$
h'(t) = A\,h(t) + B\,u(t), \qquad y(t) = C\,h(t) + D\,u(t),
$$

dove $u(t)\in\mathbb{R}$ è l'ingresso, $y(t)\in\mathbb{R}$ l'uscita e
$h(t)\in\mathbb{R}^{N}$ lo **stato** interno di dimensione $N$. Le tre matrici
hanno ruoli distinti: $A\in\mathbb{R}^{N\times N}$ è la **dinamica interna**,
governa come lo stato evolve da solo, in assenza di ingresso (gli autovalori
di $A$ decidono se lo stato decade, oscilla o esplode);
$B\in\mathbb{R}^{N\times 1}$ è la **matrice d'ingresso**, dice come il segnale
in arrivo si scrive nello stato; $C\in\mathbb{R}^{1\times N}$ è la **matrice
d'uscita** (legge lo stato e produce il segnale in uscita). Il termine
$D\,u(t)$ è una scorciatoia diretta dall'ingresso all'uscita, una *skip
connection*: nei modelli che vedremo lo si tiene a parte (equivale a un
residuo) e ci si concentra sulla parte con memoria, ponendo spesso $D=0$ nella
derivazione.

Questa è la *rappresentazione in spazio degli stati* della teoria del controllo:
lo stato $h(t)$ è, per costruzione, una statistica sufficiente del passato. La
derivata $h'(t)$ dice come lo stato cambia istante per istante, spinto in parte
dalla propria inerzia ($A\,h$) e in parte dal mondo esterno ($B\,u$).

`````

Finora tutto è continuo: il tempo scorre senza gradini. Ma una frase è una
sequenza di token, un segnale audio è una sequenza di campioni: dati **discreti**,
uno dopo l'altro. Per usare questo sistema su una sequenza dobbiamo prima
tradurlo dal continuo al discreto.

## Dal continuo al discreto

Il passaggio si chiama **discretizzazione** ed è lo stesso problema che
abbiamo incontrato nel capitolo di analisi numerica: sostituire una derivata
continua (un cambiamento infinitesimo) con un passo finito. Introduciamo un
intervallo di campionamento $\Delta$ (il tempo tra due campioni) e riscriviamo
il sistema in modo che salti di stato in stato, da $h_{t-1}$ a $h_t$, invece
di scivolare con continuità.

C'è un punto su cui vale la pena essere espliciti, perché è una fonte comune di
confusione: **non esiste un solo modo di discretizzare**. Sono possibili diverse
regole di integrazione, e due modelli famosi di questo capitolo ne usano due
diverse. S4 adotta la **trasformazione bilineare** (nota anche come metodo di
Tustin); Mamba, che incontreremo più avanti, adotta lo **zero-order hold** (ZOH,
"tenuta di ordine zero"). Non sono intercambiabili nella notazione, e attribuire
lo ZOH a S4 è un errore che si trova spesso in giro.

`````{tab} Elementare

Discretizzare è come campionare un segnale continuo: invece di seguire l'acqua
della vasca in ogni istante, ne misuri il livello a intervalli regolari (ogni
$\Delta$ secondi) e ti chiedi come passare da una misura alla successiva. Se
il passo $\Delta$ è piccolo campioni fitto e cogli ogni sfumatura, ma con più
lavoro; se è grande campioni rado e rischi di perderti quello che succede in
mezzo. È lo stesso compromesso di ogni fotografia a scatti di un movimento
fluido.

Le "due regole" per fare questo passo (bilineare e ZOH) sono due modi
leggermente diversi di indovinare cosa succede *tra* un campione e l'altro. Il
risultato pratico è lo stesso tipo di ricorrenza; cambia la formula esatta con
cui si ricavano le matrici discrete. Basta ricordare quale usa quale modello:
S4 la bilineare, Mamba lo ZOH.

`````

`````{tab} Superiore

Discretizzare significa ricavare, dalle matrici continue $A$ e $B$ e dal passo
$\Delta$, le matrici **discrete** $\bar{A}$ e $\bar{B}$ tali che la ricorrenza
$h_t = \bar{A}\,h_{t-1} + \bar{B}\,x_t$ approssimi l'evoluzione continua
($x_t$ è l'ingresso campionato al passo $t$).

Lo **zero-order hold** assume che l'ingresso resti costante entro ciascun
intervallo $\Delta$ e integra esattamente il sistema su quel tratto:

$$
\bar{A} = \exp(\Delta A), \qquad
\bar{B} = (\Delta A)^{-1}\big(\exp(\Delta A) - I\big)\,\Delta B .
$$

Qui $\exp(\cdot)$ è l'esponenziale **di matrice**, non elemento per elemento.
Se $A$ è diagonale, ogni suo autovalore $a$ si discretizza per conto suo:
$\bar{a} = e^{\Delta a}$ e $\bar{b} = \frac{e^{\Delta a}-1}{a}\,b$, ben definito
anche nel limite $a\to 0$. È la scelta di Mamba, con una precisazione: lo ZOH
vale per la transizione, $\bar{A} = \exp(\Delta A)$, mentre per l'ingresso
l'implementazione adotta la semplificazione al prim'ordine (Eulero)
$\bar{B} = \Delta B$, che dello ZOH è il troncamento per $\Delta$ piccolo. La
ritroveremo nel codice della prossima sezione.

La **trasformazione bilineare** approssima invece l'esponenziale con la sua
frazione razionale del primo ordine (la regola del trapezio), ottenendo

$$
\bar{A} = \Big(I - \tfrac{\Delta}{2}A\Big)^{-1}\Big(I + \tfrac{\Delta}{2}A\Big),
\qquad
\bar{B} = \Big(I - \tfrac{\Delta}{2}A\Big)^{-1}\Delta B .
$$

È la scelta di S4. In entrambi i casi la $C$ resta invariata ($\bar{C}=C$), e i
parametri effettivi del modello sono la quaterna $(\Delta, A, B, C)$: le matrici
continue più il passo, da cui si generano le matrici discrete. Il passo $\Delta$
non è un dettaglio: fissa la *scala temporale* del sistema, cioè quanto in fretta
lo stato dimentica.

`````

## Due facce della stessa medaglia: ricorrenza e convoluzione

Ora arriva il fatto che rende speciali questi modelli. Finché $\bar{A}$,
$\bar{B}$ e $C$ sono **costanti** (non cambiano da un passo all'altro), il
sistema discretizzato è *lineare e tempo-invariante* (in sigla LTI). E per un
sistema LTI vale un'equivalenza esatta tra due modi apparentemente diversi di
calcolare la stessa uscita: una forma **ricorrente** e una forma
**convoluzionale**.

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
con un filtro fisso. E la convoluzione la conosciamo dal capitolo sulle reti
convoluzionali: un filtro che scorre lungo il segnale. La differenza è che qui
il filtro è lungo quanto tutta la sequenza, e nessuno lo impara a mano: nasce
dalle matrici $\bar{A}$, $\bar{B}$, $C$. Il vantaggio è che una convoluzione
si calcola in un colpo solo, in parallelo su tutta la sequenza: proprio ciò
che serve per sfruttare le GPU in addestramento.

Morale: si **addestra** in forma convoluzionale (veloce, parallela) e si
**inferisce** in forma ricorrente (economica, un token alla volta). La stessa
funzione, due vestiti diversi a seconda dell'occasione.

`````

`````{tab} Superiore

La forma **ricorrente** srotola la ricorrenza discreta:

$$
h_t = \bar{A}\,h_{t-1} + \bar{B}\,x_t, \qquad y_t = C\,h_t .
$$

Ogni passo costa $O(N^2)$ (o $O(N)$ se $\bar{A}$ è diagonale) e la memoria è
$O(N)$, **costante** nella lunghezza della sequenza: è l'inferenza a costo fisso
per token tipica delle RNN.

La forma **convoluzionale** si ottiene sostituendo ripetutamente la ricorrenza
in se stessa, con stato iniziale nullo:

$$
y_t = \sum_{j=0}^{t} C\,\bar{A}^{\,j}\,\bar{B}\;x_{t-j}
    = (x * \bar{K})_t ,
$$

cioè una convoluzione tra l'ingresso e un **kernel** (o *SSM convolution kernel*)

$$
\bar{K} = \big(C\bar{B},\; C\bar{A}\bar{B},\; C\bar{A}^2\bar{B},\;
\dots,\; C\bar{A}^{\,k}\bar{B},\; \dots\big) ,
$$

dove $\bar{K}$ è un filtro causale lungo quanto la sequenza. Calcolata questa
volta sola, l'uscita $y = x * \bar{K}$ si ottiene per l'intera sequenza in
parallelo, con la FFT in tempo $O(L \log L)$ ($L$ è la lunghezza). Il termine
$C\bar{A}^{\,j}\bar{B}$ misura quanto un ingresso di $j$ passi fa pesa ancora
sull'uscita di adesso: è la memoria del sistema, e decade con le potenze
$\bar{A}^{\,j}$.

L'equivalenza $ \text{ricorrenza} \equiv \text{convoluzione} $ vale **solo**
perché $\bar{A}, \bar{B}, C$ sono costanti nel tempo: è la tempo-invarianza a
garantire che il kernel $\bar{K}$ sia unico e fisso. Quando, con Mamba, faremo
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
:alt: "A sinistra la forma ricorrente dell'SSM come catena di stati: da h_{t-1} una freccia moltiplicata per A-bar arriva a h_t, l'ingresso x_t entra moltiplicato per B-bar, e da h_t esce y_t moltiplicato per C; la stessa cella si ripete lungo la sequenza. A destra la forma convoluzionale: l'intera sequenza di ingresso x scorre sotto un unico kernel lungo K-bar = (C B-bar, C A-bar B-bar, C A-bar^2 B-bar, ...) e produce l'uscita y in un colpo solo. Una parentesi graffa collega le due viste con la scritta \"stessa funzione\"."
:width: 85%

Le due facce dello stesso SSM lineare tempo-invariante. A sinistra la forma
**ricorrente** $h_t = \bar{A}\,h_{t-1} + \bar{B}\,x_t$, un passo alla volta
(inferenza economica); a destra la forma **convoluzionale** $y = x * \bar{K}$,
tutta la sequenza in parallelo con un unico kernel (addestramento veloce).
```

## HiPPO e S4: ricordare a lungo

C'è un problema che abbiamo scavalcato. Perché uno stato di dimensione piccola
(poche decine di numeri), dovrebbe ricordare qualcosa avvenuto migliaia di
passi prima? Le potenze $\bar{A}^{\,j}$ del kernel tendono in genere a svanire
in fretta: dopo poche decine di passi il contributo del passato è già polvere.
Una RNN classica soffre esattamente di questo, ed è la ragione per cui LSTM e
GRU sono nate. Per gli SSM la risposta non sta in nuovi cancelli, ma nella
**scelta della matrice $A$**: fatta a caso, la memoria è corta; costruita con
criterio, può essere lunghissima.

`````{tab} Elementare

Immagina di dover riassumere un romanzo lunghissimo in una sola pagina di
appunti, aggiornata mentre leggi. Se ogni frase nuova cancella la precedente,
alla fine ti resta in mano solo l'ultimo capitolo. Serve un modo *principiato*
di comprimere: tenere una specie di riassunto a più livelli (l'idea generale,
gli snodi principali, i dettagli recenti), così che ciò che conta del passato
lontano non sbiadisca del tutto.

È l'idea di **HiPPO**: un modo matematicamente fondato di riassumere una storia
lunga in pochi numeri, aggiornandolo a ogni passo, senza che il passato
importante svanisca. Non un'euristica inventata a mano, ma la soluzione ottima
di un problema di approssimazione ben preciso. Chi inizializza la matrice $A$
di un SSM con la ricetta di HiPPO ottiene, gratis, una memoria a lungo raggio.

`````

`````{tab} Superiore

**HiPPO** (*High-order Polynomial Projection Operators*, Gu et al., 2020,
{cite}`gu2020hippo`) formalizza la compressione online di un segnale come la
sua proiezione ottima su una base di **polinomi ortogonali** (per esempio i
polinomi di Legendre) rispetto a una misura sul passato. Lo stato $h(t)$
diventa il vettore dei coefficienti di quella proiezione: ricostruisce, nel
modo meno sbagliato possibile, tutto il segnale visto fin lì. La variante
**HiPPO-LegS** (*scaled Legendre*) usa una misura che copre uniformemente
tutta la storia, ed è robusta alla scala temporale: ricorda a lungo
indipendentemente da $\Delta$. Il risultato pratico è una matrice $A$
specifica (la *matrice HiPPO*) con cui **inizializzare** l'SSM per dotarlo di
memoria a lungo raggio.

**S4** (*Structured State Space Sequence model*, Gu, Goel e Ré, ICLR 2022,
{cite}`gu2022s4`) parte proprio da qui: inizializza $A$ con HiPPO-LegS. Ma sorge
un ostacolo computazionale. Costruire il kernel $\bar{K}$ richiede le potenze
$\bar{A}^{\,j}$ fino a $j = L-1$: farlo direttamente costa $O(N^2 L)$ operazioni,
proibitivo per stati e sequenze grandi. La mossa di S4 è imporre ad $A$ una
**struttura**: la scrive come **diagonale più basso rango** (DPLR, *Diagonal
Plus Low-Rank*),

$$
A = \Lambda - P Q^{*},
$$

dove $\Lambda$ è diagonale e $P Q^{*}$ è una correzione di rango basso ($\Lambda$
raccoglie gli autovalori, $P$ e $Q$ sono matrici "alte e strette"). Con questa
struttura il kernel non si calcola più elevando a potenza una matrice piena: lo
si ottiene passando alla sua *funzione generatrice* valutata sulle radici
dell'unità, e sfruttando l'identità di Woodbury (per l'inversa di
"diagonale + basso rango") e un kernel di Cauchy. Il costo scende a
quasi-lineare in $N + L$.

Il guadagno non è solo teorico. S4 genera circa $60\times$ più in fretta dei
Transformer di pari qualità e, soprattutto, fa un salto su **Long Range Arena**,
il banco di prova delle dipendenze a lunghissimo raggio: è il primo modello a
risolvere **Path-X**, il compito su sequenze da $16\,384$ elementi su cui i
Transformer restavano al livello del caso. Sull'immagine, tratta come sequenza
lunga di pixel, S4 raggiunge circa il 91% su sequential CIFAR-10.

`````

## Tappe verso il linguaggio

S4 dimostrò che uno stato piccolo, ben costruito, poteva competere con
l'attenzione sulle sequenze lunghe. Ma tra quel risultato e Mamba (il modello
che porta gli SSM sul linguaggio in modo convincente) ci sono alcune tappe che
vale la pena nominare, perché ognuna smonta un pezzo del problema.

`````{tab} Elementare

Tre modelli, tre pezzi del problema. **S5** (2023) semplifica la macchina di
S4 e, soprattutto, dimostra che la forma «passo dopo passo» non è condannata a
essere lenta: organizzando i calcoli in modo furbo (si combinano i passi a
coppie, poi a gruppi di quattro, di otto, e così via) anche la ricorrenza si
può svolgere quasi tutta in parallelo. È il trucco che Mamba erediterà.

**H3** (2023) affronta invece la memoria «a richiamo»: ritrovare più avanti
una cosa già letta ("chi era il soggetto di quella frase?"). I modelli di
questa famiglia, trattando ogni parola con la stessa regola, faticavano a
farlo; H3 li aiuta accoppiando due memorie e un rubinetto che dosa il flusso
tra l'una e l'altra, così il modello riesce a trattenere un'informazione
finché gli serve.

**Hyena** (2023), infine, prova la via più diretta: se l'ingrediente vincente
è un filtro lungo che scorre su tutta la frase (come quelli visti nel capitolo
sulle reti convoluzionali, ma lunghi quanto l'intero testo), tanto vale
imparare direttamente il filtro, senza passare dal sistema dinamico. Funziona
quasi come l'attenzione, costando molto meno.

`````

`````{tab} Superiore

**S5** (Smith, Warrington e Linderman, ICLR 2023, {cite}`smith2023s5`)
semplifica S4 su due fronti. Primo: usa un unico SSM **MIMO** (a più ingressi
e più uscite) con matrice $A$ **diagonale**, invece di tanti SSM scalari
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
  ingredienti: come lo stato cala da solo, come l'ingresso lo alza, come si
  legge l'uscita. È l'altra strada verso un modello che regge i testi lunghi
  senza che il costo esploda, complementare all'attenzione lineare del capitolo
  precedente.
- Per usarlo su una sequenza (parole, campioni audio) bisogna **misurare a
  intervalli regolari** e indovinare cosa succede *tra* un campione e il
  successivo. Le ricette non sono una sola e non vanno confuse: **S4 immagina
  quel tratto come un trapezio**; **Mamba tiene l'ingresso fermo per tutto
  l'intervallo** (è lo *zero-order hold*, la tenuta di ordine zero). Per quanto
  di ciò che entra finisce nella memoria, però, Mamba si accontenta del conto
  più sbrigativo, a rettangoli: è il pezzo che Mamba-3 rifarà a trapezi.
- Finché le regole **non cambiano da un passo all'altro**, lo stesso calcolo si
  può fare in due modi: **passo dopo passo** (un token alla volta, con una
  memoria che non cresce mai: economico per generare) oppure **tutto insieme**,
  come un unico filtro lungo che scorre sulla sequenza (parallelo: perfetto per
  addestrare sulle GPU). Si allena nel secondo modo, si usa nel primo: la stessa
  dualità vista con l'attenzione lineare.
- L'equivalenza regge **solo** finché quelle regole restano fisse: Mamba le farà
  dipendere da ciò che legge, e allora resterà solo il modo passo dopo passo.
- **HiPPO** (Gu et al., 2020) è il modo principiato di riassumere una storia
  lunghissima in pochi numeri, come appunti a più livelli su un romanzo: è ciò
  che dà memoria a lungo raggio a uno stato piccolo. **S4** (2022) rende il
  conto efficiente dando alla macchina una forma regolare, ed è il primo a
  risolvere Path-X (sequenze da $16\,384$ elementi), la prova più dura della
  gara sulle dipendenze a lunghissimo raggio, Long Range Arena.
- Le tappe verso il linguaggio: **S5** (mostra che anche il passo dopo passo si
  può svolgere quasi tutto in parallelo), **H3** (due memorie e un rubinetto,
  per ritrovare a distanza una cosa già letta), **Hyena** (impara direttamente
  il filtro lungo). Preparano Mamba.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Un **SSM** nasce da un sistema dinamico continuo
  $\mathbf{h}'(t)=A\,\mathbf{h}(t)+B\,u(t)$,
  $y(t)=C\,\mathbf{h}(t)$: lo **stato** $\mathbf{h}$ è una fotografia
  compatta del passato, con $A$ dinamica interna, $B$
  ingresso, $C$ uscita. È l'altra strada verso il tempo lineare,
  complementare all'attenzione lineare del capitolo precedente.
- Per usarlo su sequenze discrete serve un passo $\Delta$ di
  **discretizzazione**: una regola per indovinare cosa succede *tra* un campione
  e il successivo. Di regole ce n'è più d'una e non vanno confuse. **S4 usa la
  bilineare** (immagina quel tratto come un trapezio); **Mamba usa lo
  *zero-order hold*** (ZOH: l'ingresso resta fermo per tutto l'intervallo), che
  gli dà la transizione $\bar{A}=\exp(\Delta A)$. Per il
  termine d'ingresso, però, l'implementazione di Mamba si accontenta del conto a
  rettangoli, $\bar{B}=\Delta B$ (il metodo di Eulero, cioè lo
  ZOH troncato al prim'ordine): è il pezzo che Mamba-3 rifarà a trapezi.
- Se il sistema discretizzato è **tempo-invariante** (LTI), la stessa funzione
  ha due forme equivalenti: **ricorrente**
  $\mathbf{h}_t=\bar{A}\mathbf{h}_{t-1}+\bar{B}x_t$
  (inferenza $O(1)$ per passo) e **convoluzionale** $y=x*\bar{K}$
  (addestramento parallelo). Si allena convoluzionale, si inferisce
  ricorrente: la stessa dualità vista con l'attenzione lineare.
- Questa equivalenza vale **solo** se $\bar{A},\bar{B},
  C$ sono costanti: Mamba la romperà rendendoli dipendenti
  dall'ingresso, e allora resterà solo lo scan.
- **HiPPO** (Gu et al., 2020) sceglie $A$ proiettando la storia su
  polinomi ortogonali: è ciò che dà memoria a lungo raggio a uno stato piccolo.
  **S4** (2022) rende il calcolo efficiente con una struttura **diagonale +
  basso rango** e risolve per primo Path-X (sequenze da 16k) su Long Range
  Arena.
- Le tappe verso il linguaggio: **S5** (SSM MIMO + parallel scan), **H3** (due
  SSM + gating per il recall associativo), **Hyena** (convoluzioni lunghe
  implicite). Preparano Mamba.
```

`````
