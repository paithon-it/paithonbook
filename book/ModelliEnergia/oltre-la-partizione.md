# Oltre la partizione: tre modi di aggirare $Z$

Nella sezione precedente è comparso, quasi di sfuggita, il personaggio che
domina questo capitolo: la **funzione di partizione**, che il titolo qui sopra
chiama con la sua iniziale, $Z$. È il conto di cui si diceva: la somma su
*tutte* le configurazioni possibili, quella che trasforma un'altezza in una
percentuale. Vale la pena guardarla in faccia, perché è lei a dettare tutto ciò
che segue, e perché la sua intrattabilità non è una difficoltà tecnica fra le
tante, è un muro.

Una rete di venticinque neuroni accesi o spenti, come quella della memoria
associativa, ha trentatré milioni di configurazioni ($2^{25} = 33\,554\,432$),
e $Z$ si calcola davvero: contarle è questione di istanti, e valutare
l'energia di ciascuna, che è il conto vero, richiede meno di un minuto
(misurato: quarantasette secondi). Aggiungiamone settantacinque. Con cento neuroni le
configurazioni diventano un numero lungo trentuno cifre, e a un miliardo al
secondo servirebbero quasi **tremila volte l'età dell'universo** per contarle
tutte ($2^{100} \approx 1{,}27 \times 10^{30}$, cioè circa
$4 \times 10^{13}$ anni). E cento neuroni accesi o spenti sono un'immagine in
bianco e nero di dieci pixel per dieci: nemmeno una figurina. Nessun trucco di
ingegneria recupera trenta ordini di grandezza: se una strada passa da $Z$,
quella strada è chiusa.

`````{tab} Elementare

Torniamo al paesaggio, e immaginiamo che ci piova sopra: l'acqua si raccoglie
nelle valli, e quanto più una valle è profonda tanta più acqua ci finisce
dentro. La pioggia raccolta, allora, è la probabilità: una risposta molto
plausibile è una valle molto profonda, cioè un posto molto bagnato. Per dire
quanto è *alta* una valle rispetto alle
altre non serve nulla di speciale: si guardano le due altezze e si
confrontano. Per dire invece che una valle raccoglie «il 30% di tutta la
pioggia che cade sul continente» bisogna aver misurato l'intero continente,
valle per valle. La funzione di partizione è la misura dell'intero
continente: è ciò che trasforma un'altezza in una percentuale.

E qui verrebbe da chiedersi perché mai dovremmo misurarlo. La risposta è che
imparare, per un modello a energia, sono **due** gesti e non uno: abbassare il
paesaggio dove stanno i dati veri, e alzarlo dove il modello si immagina roba
che non esiste. Il primo è facile, i dati ce li abbiamo in mano. Il secondo
no: per sapere che cosa il modello si immagina bisogna prima fargli produrre
qualcosa, e per farlo per bene bisognerebbe conoscere il paesaggio intero. Se
ci si limitasse ad abbassare, il modello troverebbe subito la scorciatoia:
abbassare tutto, dappertutto, e dire di sì a qualunque cosa gli si presenti.
L'alzare è il gesto che costa, ed è il motivo per cui la misura del continente
continua a ripresentarsi.

Il problema è che il continente, qui, è grande quanto tutte le immagini
possibili. Non lo si percorre. E allora si può fare una di tre cose.
*Campionarlo* (mandare esploratori a caso e accontentarsi di quello che
riportano). *Evitarlo* (accorgersi che per molte domande la percentuale non
serve: basta la pendenza sotto i piedi). Oppure *aggirarlo con un trucco*:
sostituire la domanda «quanto è probabile questo?» con «questo viene dai dati
o l'ho inventato io?», che è una domanda da rispondere sì o no, e per le
domande sì o no sappiamo addestrare un classificatore da trent'anni.

Le tre strade esistono tutte e tre, hanno tutte e tre un nome, e la seconda
(la meno intuitiva) è quella che nel giro di un decennio ha prodotto i modelli
di diffusione del capitolo precedente.

`````

`````{tab} Superiore

Il punto di attrito è il gradiente della log-verosimiglianza. Da
$p_\theta(\mathbf{x}) = e^{-E_\theta(\mathbf{x})}/Z(\theta)$ segue
$\log p_\theta(\mathbf{x}) = -E_\theta(\mathbf{x}) - \log Z(\theta)$, e il primo
addendo si deriva senza storie. Tutto sta nel secondo, ed è il passaggio da
cui dipende il resto del capitolo:

$$
\nabla_\theta \log Z(\theta)
= \frac{1}{Z(\theta)} \int \nabla_\theta e^{-E_\theta(\mathbf{x}')}\, d\mathbf{x}'
= - \int \underbrace{\frac{e^{-E_\theta(\mathbf{x}')}}{Z(\theta)}}_{=\ p_\theta(\mathbf{x}')}
\nabla_\theta E_\theta(\mathbf{x}')\, d\mathbf{x}'
= - \mathbb{E}_{\mathbf{x}' \sim p_\theta}\!\left[\nabla_\theta E_\theta(\mathbf{x}')\right].
$$

Tre mosse, e conviene nominarle: si scambiano derivata e integrale (lecito per
convergenza dominata, se $\nabla_\theta e^{-E_\theta}$ è dominata da una
funzione integrabile in un intorno di $\theta$); si deriva l'esponenziale; e
si riconosce che il rapporto rimasto sotto integrale **è** la densità del
modello, il che trasforma un integrale su tutto lo spazio in un valore atteso
che si può stimare per campionamento. Mettendo insieme:

$$
\nabla_\theta \log p_\theta(\mathbf{x})
= -\nabla_\theta E_\theta(\mathbf{x})
+ \mathbb{E}_{\mathbf{x}' \sim p_\theta}\!\left[\nabla_\theta E_\theta(\mathbf{x}')\right].
$$

La log-verosimiglianza si **massimizza**, quindi i parametri si muovono nel
verso di questo gradiente: il primo termine (**fase positiva**) abbassa allora
l'energia sul dato
osservato e il secondo (**fase negativa**) la rialza sui campioni *del
modello*. Mediando anche il primo su $p_{\text{dati}}$ si ottiene la forma
simmetrica «media sui dati meno media sul modello» già incontrata nella
macchina di Boltzmann. Il termine che dà problemi non è $Z$ in sé, ma quel
valore atteso: la terza mossa lo ha reso stimabile, non gratuito, e per
stimarlo bisogna saper campionare da $p_\theta$, cioè dal modello che
stiamo ancora addestrando.

Il tutorial di LeCun {cite}`lecun2006tutorial` legge la stessa formula in
chiave energetica, e la lettura è illuminante: il termine contrastivo «solleva
l'energia di ogni risposta con una forza proporzionale alla sua
verosimiglianza sotto il modello», e tutte le tecniche di approssimazione
(Monte Carlo, metodi variazionali) si possono vedere come **strategie diverse
per scegliere quali risposte tirare su**. Le tre sezioni che seguono sono, in
questa luce, tre risposte alla stessa domanda: chi solleviamo, e come?

`````

## Prima via: campionare il paesaggio

Se il conto esatto su tutto il paesaggio non si può fare, lo si può stimare
visitandone dei pezzi: non calcolare, campionare. Serve però un modo di
produrre risposte davvero pescate dal modello, cioè che escano fuori con la
frequenza che il paesaggio prescrive, e il modo classico è lasciar vagare
qualcosa sul paesaggio abbastanza a lungo perché passi in ogni punto il tempo
che gli spetta. Quando le risposte non sono acceso e spento ma numeri con la
virgola (un'immagine vera, per dire, dove ogni pixel può avere qualunque
sfumatura), la ricetta più usata porta il nome del fisico francese Paul
Langevin, che nel 1908 la scrisse per il **moto browniano**, il tremolio di un
granello di polline sull'acqua sotto gli urti delle molecole, ed è quasi uno
slogan: **scendere lungo la pendenza dell'energia, con addosso
un po' di rumore**.

`````{tab} Elementare

Una pallina che rotola in discesa finisce nel fondovalle più vicino e lì si
ferma: è la dinamica di Hopfield, e produce sempre la stessa risposta. Ora
immagina la stessa pallina su un tavolo che vibra: continua a scendere,
perché la pendenza c'è ancora, ma i sussulti la fanno anche risalire un po',
uscire dalle conche, passare da una valle all'altra. Se la guardi per molto
tempo e segni dove si trova, scoprirai che passa **più tempo dove il
paesaggio è basso** e pochissimo sulle cime: la frequenza con cui visita
ogni punto *è* la probabilità che il paesaggio definisce.

Questo è il punto elegante della faccenda: per far vibrare e scendere la
pallina serve solo la pendenza locale, quella sotto i suoi piedi. La misura
dell'intero continente (la costante che non sappiamo calcolare) non entra mai
nella discesa, perché una costante non ha pendenza. Campionare non richiede di
normalizzare.

Il prezzo è il tempo. Se due valli sono separate da una montagna alta, la
pallina può restare intrappolata a lungo da una parte, e la fotografia che
ne ricavi è sbilanciata.

E qui va spiegata una parola che tornerà spesso da adesso in poi: **alta
dimensione**. Un'immagine di dieci pixel per dieci è fatta di cento numeri, e
ognuno di quei numeri è una direzione in cui la si può cambiare: il suo
paesaggio non è una collina con un davanti e un dietro, ha cento direzioni
indipendenti, e quello di una fotografia vera ne ha milioni. In un posto così
le valli sono separate da creste lunghissime e le vie per passare da una
all'altra sono rarissime: una pallina può vagare per un tempo lunghissimo
senza trovarne una. È il tallone d'Achille di tutta la famiglia.

`````

`````{tab} Superiore

La **dinamica di Langevin** genera una sequenza di stati

$$
\mathbf{x}_{k+1} = \mathbf{x}_k - \frac{\epsilon}{2}\, \nabla_{\mathbf{x}} E_\theta(\mathbf{x}_k) + \sqrt{\epsilon}\, \mathbf{z}_k,
\qquad \mathbf{z}_k \sim \mathcal{N}(0, \mathbf{I}),
$$

dove $\epsilon > 0$ è il passo (un **tempo**, non una lunghezza) e
$\mathbf{z}_k$ il rumore gaussiano. Per
$k \to \infty$, con $\epsilon \to 0$ e $k\epsilon \to \infty$ (il passo si
accorcia, ma il tempo totale percorso dalla catena deve crescere senza
limite), la distribuzione di $\mathbf{x}_k$ converge a
$p_\theta \propto e^{-E_\theta}$. Il teorema vuole però anche delle ipotesi
sul paesaggio, e vale la pena enunciarle perché l'esempio qui sotto ne viola
una: $\nabla_{\mathbf{x}} E_\theta$ globalmente lipschitziano, o almeno una
condizione di dissipatività che tenga la catena al finito. Con
$E(x) = (x^2-1)^2$ il gradiente cresce come $x^3$, non è lipschitziano, e a
passo fissato la ricorsione **diverge** oltre una soglia:
$|1 - 2\epsilon(x^2-1)| > 1$, cioè $|x| > \sqrt{1 + 1/\epsilon}$, che a
$\epsilon = 0{,}01$ vale $10{,}05$ (verificato: da $10{,}00$ la catena torna
in una buca, da $10{,}05$ esplode in nove passi). Non si vede mai, perché
lassù la densità vale $e^{-9800}$, ma è una divergenza vera e non
un'approssimazione: quella catena, a rigore, è transiente. A passo fissato, com'è nel codice qui sotto
e nella pratica degli EBM, la catena si assesta poi su una distribuzione
leggermente distorta, con un errore dell'ordine di $\epsilon$: lo
eliminerebbe un test di accettazione alla Metropolis (la variante MALA), a
cui di solito si rinuncia in cambio della semplicità. Si noti che compare
**solo** $\nabla_{\mathbf{x}} E_\theta$: la costante $\log Z(\theta)$, non dipendendo da
$\mathbf{x}$, ha gradiente nullo. Il campionamento non ha mai bisogno della
normalizzazione: è l'osservazione su cui poggia tutto il resto della sezione.

La versione stocastica su minibatch, che sostituisce il gradiente esatto con
quello stimato, è la *stochastic gradient Langevin dynamics*
{cite}`welling2011bayesian`, nata per campionare la distribuzione a posteriori
dei *parametri* e poi passata di peso al campionamento dei dati: un passo di
discesa dimezzato ($\epsilon/2$) più un rumore di deviazione standard
$\sqrt{\epsilon}$. Il $\tfrac12$ e la radice non sono scelte di gusto: sono
ciò che rende la ricorsione la discretizzazione di Eulero–Maruyama della
diffusione $d\mathbf{x} = -\tfrac12 \nabla_{\mathbf{x}} E_\theta\, dt + d\mathbf{W}$,
che ha $p_\theta$ come misura invariante. Le due ampiezze **non si
confrontano fra loro**: hanno dimensioni diverse ($[\text{tempo}]^{1/2}$ e
$[\text{tempo}]$), e su un intervallo di tempo fissato i due contributi
restano dello stesso ordine, che è precisamente il motivo per cui il limite
continuo esiste. Quello che in Welling e Teh diventa trascurabile al
decrescere del passo è un'altra cosa ancora: il rumore del **gradiente su
minibatch**, che scala come $\epsilon$ e finisce sotto quello iniettato; ed è
lì che la catena passa senza soluzione di continuità dall'ottimizzazione al
campionamento. Nella pratica degli EBM la catena si tronca dopo poche
decine di passi (*short-run MCMC*) e si conservano i campioni in un serbatoio
da cui ripartire, l'erede diretto della persistent contrastive divergence
della sezione precedente.

`````

Il codice che segue costruisce il paesaggio più semplice in cui la faccenda si
vede: due valli e una collinetta in mezzo. In formula è l'energia a doppia
buca $E(x) = (x^2 - 1)^2$, e i conti si fanno a mente: in $x = 1$ e in
$x = -1$ la parentesi vale zero, quindi l'energia vale zero (sono i due
fondovalle), mentre in $x = 0$ vale $(0-1)^2 = 1$, che è la collinetta.
Ci mette sopra ventimila
palline, le fa vibrare con la ricetta di Langevin e alla fine guarda dove si
sono distribuite, senza aver mai calcolato $Z$. Poi, per pura verifica, $Z$ la
calcola: in un paesaggio a una sola dimensione si può, ed è l'unico modo per
sapere se il campionamento ha detto il vero. Chi non programma può saltare
alla tabella: la colonna «campioni» dice dove sono finite le palline, la
colonna «esatto» dove sarebbero dovute finire.

```python
import numpy as np

rng = np.random.default_rng(0)

# Energia a doppia buca: minimi in x = -1 e x = +1, barriera in x = 0.
def energia(x):
    return (x**2 - 1.0)**2

def gradiente(x):            # dE/dx = 4x(x^2 - 1); lo score e' -gradiente
    return 4.0 * x * (x**2 - 1.0)

# Dinamica di Langevin: ventimila catene in parallelo, passi piccoli.
eps, passi, catene = 0.01, 2000, 20000
x = rng.normal(0.0, 2.0, size=catene)        # partenza qualsiasi
for _ in range(passi):
    x = x - 0.5 * eps * gradiente(x) + np.sqrt(eps) * rng.normal(size=catene)

# Verifica: p(x) = e^{-E(x)}/Z per quadratura numerica (si puo' fare in 1D).
griglia = np.linspace(-3, 3, 60001)
peso = np.exp(-energia(griglia))
Z = np.trapezoid(peso, griglia)
p_esatta = peso / Z

print(f"Z (quadratura)          = {Z:.4f}")
print(f"campioni |x| medio      = {np.abs(x).mean():.3f}")
print(f"esatto   |x| medio      = {np.trapezoid(np.abs(griglia)*p_esatta, griglia):.3f}")
print(f"frazione x>0 (campioni) = {(x > 0).mean():.3f}   (esatto 0.500)")
print()
print(" intervallo   campioni   esatto")
for a, b in [(-2.0, -1.5), (-1.5, -0.5), (-0.5, 0.5), (0.5, 1.5), (1.5, 2.0)]:
    emp = ((x >= a) & (x < b)).mean()
    m = (griglia >= a) & (griglia < b)
    ex = np.trapezoid(p_esatta[m], griglia[m])
    print(f" [{a:+.1f},{b:+.1f})    {emp:6.3f}   {ex:6.3f}")
```

```text
Z (quadratura)          = 1.9737
campioni |x| medio      = 0.822
esatto   |x| medio      = 0.827
frazione x>0 (campioni) = 0.497   (esatto 0.500)

 intervallo   campioni   esatto
 [-2.0,-1.5)     0.011    0.011
 [-1.5,-0.5)     0.380    0.379
 [-0.5,+0.5)     0.225    0.219
 [+0.5,+1.5)     0.373    0.379
 [+1.5,+2.0)     0.012    0.011
```

Le due colonne coincidono entro pochi millesimi (lo scarto più largo è di
0,006, nella riga centrale, quella che raccoglie le palline finite fra $-0,5$
e $+0,5$: un tratto di paesaggio così, in gergo, si chiama **bin**, e da qui
in avanti la parola torna qualche volta): le catene hanno ricostruito la
distribuzione senza
che $Z$ sia mai entrata nel ciclo. Quell'ultimo millesimo, però, non è tutto
rumore statistico. Dentro c'è un errore di natura diversa, ed è colpa del
**passo**: la pallina non scivola giù per il pendio con continuità, lo scende
a saltelli, e $\epsilon$ (nel codice, `eps`) è quanto dura ogni saltello. Una
scala di gradini non è una rampa. Più i saltelli sono brevi, più la fotografia
finale somiglia a quella
vera; con saltelli di durata finita resta uno scarto che non è statistico e
che nessuna quantità di catene fa sparire.

Per vederlo, una sola esecuzione non basta, e vale la pena spiegare perché. Con
ventimila catene l'incertezza statistica su un bin vale circa 0,003, cioè
quanto l'effetto che vogliamo misurare: il numero stampato qui sopra, da solo,
non sa distinguere le due cose. Allora si ripete: sei esecuzioni con sei
sorteggi diversi (i semi da 0 a 5), tutte a parità di *tempo percorso*
($k\epsilon = 20$, dove $k$ è il numero di passi: se si dimezza il passo si
raddoppiano i passi, così la passeggiata dura sempre lo stesso). Lo scarto
**medio** sul bin centrale vale $+0{,}0032 \pm 0{,}0008$ con
$\epsilon = 0{,}01$, poi $+0{,}0024 \pm 0{,}0011$ con $\epsilon = 0{,}002$ e
$-0{,}0002 \pm 0{,}0009$ con $\epsilon = 0{,}0005$ (il numero dopo il $\pm$
dice di quanto quella media balla da un sorteggio all'altro).

E adesso la parte onesta, perché sei ripetizioni **non bastano ancora**: fra i
primi due punti c'è una differenza di 0,0008 con incertezze di 0,0008 e
0,0011, cioè nessuna differenza. Chi si fermasse qui avrebbe due punti
indistinguibili e uno compatibile con zero, e concluderebbe per fede. La
strada che chiude la questione è la stessa che questa sezione ha già usato per
$Z$: in una dimensione **si può calcolare la risposta esatta**. La
distribuzione su cui la catena a passo $\epsilon$ si assesta davvero è
l'autovettore di Perron del suo operatore di transizione, e su una griglia
fine si trova per iterazione. Il suo scarto sul bin centrale vale
$+0{,}00357$ a $\epsilon = 0{,}01$, $+0{,}00071$ a $\epsilon = 0{,}002$ e
$+0{,}00018$ a $\epsilon = 0{,}0005$: sempre positivo (la barriera è
sovrappesata), e il rapporto scarto/$\epsilon$ resta fra $0{,}35$ e $0{,}36$
su due ordini di grandezza di passo. Ecco l'errore «dell'ordine di
$\epsilon$» annunciato nella scheda Superiore, non annunciato ma misurato.
Sparirebbe aggiungendo, dopo ogni saltello, un controllo che
ogni tanto lo rifiuta (si chiama test di accettazione di Metropolis), ed è la
mossa a cui i modelli a energia rinunciano per semplicità.

Il confronto fra le due colonne di numeri è la lezione, e vale ben oltre
questo esempio. Sul singolo seme il segno cambia perfino: con
$\epsilon = 0{,}0005$ tre esecuzioni su sei danno uno scarto negativo, mentre
il valore vero è positivo. E la stima a $\epsilon = 0{,}002$ sbaglia di tre
volte tanto. Un effetto sistematico più piccolo del rumore non si vede
ripetendo di più: si vede cambiando strumento, e un numero solo, per quanto
stampato con quattro cifre, non dimostra niente.

Vale la pena notare anche *perché* qui funziona così bene, per non trarne una
lezione sbagliata: la barriera fra le due buche è alta un'unità di energia
(bassa) e le catene sono ventimila e indipendenti. Alzando la barriera, o
passando a mille dimensioni dove le valli sono separate da creste
lunghissime, la stessa procedura darebbe una fotografia sbilanciata, e nessuno
se ne accorgerebbe: in alta dimensione la colonna «esatto» non si può
stampare.

## Seconda via: imparare la pendenza, non la probabilità

Se il campionamento è costoso perché insegue le percentuali, si può cambiare
bersaglio. Aapo Hyvärinen, nel 2005, propone di smettere di confrontare
*quanta* probabilità il modello mette in ogni punto, e di confrontare invece
la **pendenza** del paesaggio in quel punto {cite}`hyvarinen2005estimation`.
Sembra un dettaglio ed è una liberazione, per una ragione che si dice in una
riga: la misura dell'intero continente è un numero solo, lo stesso
dappertutto, e un numero uguale dappertutto non ha pendenza. Cambiando
bersaglio, sparisce.

`````{tab} Elementare

Immagina di dover descrivere un paesaggio a qualcuno che non lo vedrà mai.
Puoi dirgli, per ogni punto, «qui c'è il 3% della pioggia», e per farlo devi
aver misurato tutto il continente. Oppure puoi dirgli, per ogni punto, «da qui
si scende verso nord-est, con questa pendenza». La seconda descrizione non
richiede di conoscere il continente: è tutta locale. Eppure basta a
ricostruire la forma del paesaggio, a meno di quanto sta in alto o in basso in
assoluto, che per generare non serve.

Attenzione a non immaginare due paesaggi: è sempre lo stesso. L'altezza è
l'energia, e dove l'energia è bassa la probabilità è alta, quindi parlare del
paesaggio dell'una o dell'altra è parlare della stessa carta, una capovolta
rispetto all'altra. La sua pendenza ha un nome tecnico, **score**, ed è la
stessa parola che compare nel capitolo sui modelli di diffusione. Non è una
coincidenza: è la stessa cosa. Insegnare a una rete la pendenza in ogni punto,
invece della percentuale, è ciò che rende addestrabile un generatore di
immagini, e ciò che ha tolto di mezzo, per quella strada, il problema della
normalizzazione.

`````

`````{tab} Superiore

Lo **score** di una densità è $s(\mathbf{x}) = \nabla_{\mathbf{x}} \log p(\mathbf{x})$ (la lettera $s$ qui
non ha niente a che vedere con lo stato della rete delle sezioni precedenti:
cambia mestiere, e in questa sezione vale questo). Per un modello a energia,

$$
\nabla_{\mathbf{x}} \log p_\theta(\mathbf{x}) = -\nabla_{\mathbf{x}} E_\theta(\mathbf{x}),
$$

perché $\log Z(\theta)$ non dipende da $\mathbf{x}$. Lo **score matching** minimizza
la distanza attesa fra lo score del modello e quello dei dati,

$$
J(\theta) = \frac{1}{2}\,
\mathbb{E}_{\mathbf{x} \sim p_{\text{dati}}}
\left\lVert \nabla_{\mathbf{x}} \log p_\theta(\mathbf{x}) - \nabla_{\mathbf{x}} \log p_{\text{dati}}(\mathbf{x})
\right\rVert^2,
$$

che a prima vista è inservibile (lo score dei dati non lo conosciamo) ma che
un'integrazione per parti trasforma in una quantità calcolabile su un
campione. Le ipotesi vale la pena scriverle, perché non sono formalità: oltre
alla regolarità e al decadimento all'infinito ($p_{\text{dati}}(\mathbf{x})\,
\nabla_{\mathbf{x}} \log p_\theta(\mathbf{x}) \to 0$ per $\lVert\mathbf{x}\rVert
\to \infty$, che è ciò che annulla il termine di bordo), servono
$p_{\text{dati}}$ **strettamente positiva e differenziabile su tutto**
$\mathbb{R}^D$ e a **supporto connesso**. Sono le due che si rompono davvero:
i dati veri vivono su una varietà di dimensione molto minore dello spazio in
cui stanno (una fotografia di volti non riempie $\mathbb{R}^{D}$), e senza
connessione due densità con lo stesso score possono differire di un fattore
costante da una componente all'altra, cioè lo score non identifica più la
densità. Sotto quelle ipotesi {cite}`hyvarinen2005estimation`:

$$
J(\theta) = \mathbb{E}_{\mathbf{x} \sim p_{\text{dati}}}
\left[ \operatorname{tr}\!\big(\nabla_{\mathbf{x}}^2 \log p_\theta(\mathbf{x})\big)
+ \tfrac{1}{2} \left\lVert \nabla_{\mathbf{x}} \log p_\theta(\mathbf{x}) \right\rVert^2 \right]
+ \text{cost.},
$$

dove $\nabla_{\mathbf{x}}^2$ è la matrice hessiana rispetto a $\mathbf{x}$ e la costante,
che vale $\tfrac{1}{2}\mathbb{E}\lVert\nabla_{\mathbf{x}} \log
p_{\text{dati}}\rVert^2$, non
dipende da $\theta$ {cite}`hyvarinen2005estimation`. Vale la pena nominarla,
perché è lei a dire *perché* $J$ è un obiettivo sensato: essendo la prima
forma una media di norme al quadrato, $J \ge 0$, e $J = 0$ se e solo se i due
score coincidono quasi ovunque. Niente $Z$, niente
catene di Markov: solo derivate del modello. Il costo si è spostato sulla
traccia dell'hessiana, e va quantificato, perché è l'unico costo del capitolo
che si lascia contare: sono $D$ retropropagazioni per ogni esempio, con $D$ la
dimensione del dato. Su un'immagine è proibitivo, ed è la ragione per cui in
pratica la si stima con una proiezione casuale (lo *sliced score matching*)
invece di calcolarla.

Il colpo di scena arriva nel 2011: Pascal Vincent dimostra che lo score
matching su dati **perturbati con rumore gaussiano** equivale, a meno di
costanti, ad addestrare un *denoising autoencoder*
{cite}`vincent2011connection`. Con $\tilde{\mathbf{x}} = \mathbf{x} + \sigma \boldsymbol{\varepsilon}$ e
$\boldsymbol{\varepsilon} \sim \mathcal{N}(0, \mathbf{I})$, dove $\sigma$ è qui la deviazione
standard del rumore e non la sigmoide di poco fa, e $\boldsymbol{\varepsilon}$
è il rumore iniettato e non il passo $\epsilon$ della catena di Langevin
(stessa lettera greca, due mestieri: qui è un vettore, e va in grassetto), il
bersaglio dello score sul dato perturbato è noto in forma chiusa,
$\nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) = -(\tilde{\mathbf{x}} - \mathbf{x})/\sigma^2 = -\boldsymbol{\varepsilon}/\sigma$,
e l'obiettivo diventa una regressione: predire il rumore iniettato.

Resta però da capire perché regredire sullo score **condizionato** a $\mathbf{x}$
dia lo score della **marginale** $q_\sigma(\tilde{\mathbf{x}})$, che è quello
che serve per generare, e il ponte è il teorema di Vincent. Sta in due
osservazioni: la prima è l'identità

$$
\nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}})
= \mathbb{E}_{\mathbf{x} \mid \tilde{\mathbf{x}}}\!\left[
\nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})\right],
$$

cioè lo score della marginale è la media del bersaglio condizionale sui dati
compatibili con $\tilde{\mathbf{x}}$; la seconda è che il minimo di una
regressione quadratica *è* la media condizionale del bersaglio. Chi minimizza
la regressione, quindi, ottiene esattamente lo score della marginale. È il
**denoising score matching**, niente hessiana e niente MCMC, ed è la loss dei
modelli di diffusione del capitolo precedente {cite}`song2021score` a meno di
una **riponderazione per livello di rumore**, che non è un dettaglio: senza
di essa il bersaglio $-\boldsymbol{\varepsilon}/\sigma$ farebbe esplodere il
peso dei livelli di rumore piccoli, e il fattore che si usa (proporzionale a
$\sigma^2$) è precisamente quello che cancella l'$1/\sigma$ e lascia la
regressione sul rumore in forma pulita.

Il prezzo c'è, e non è quello che si direbbe: ciò che si impara non è lo score
dei dati, è lo score dei dati **sporcati**, cioè della densità marginale
$q_\sigma$, che è $p_{\text{dati}}$ convoluta con la gaussiana (e si noti che
$q_\sigma(\tilde{\mathbf{x}})$ e $q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})$
sono due oggetti diversi, come sempre nella notazione delle densità). I due
score coincidono solo nel limite
$\sigma \to 0$, e a $\sigma$ finito resta un errore sistematico che nessuna
quantità di dati riduce. È esattamente il motivo per cui i modelli di
diffusione non usano un solo livello di rumore ma un'intera scala di livelli,
e il campionamento deve attraversarli in fila.

`````

Qui conviene fermarsi un istante, perché il cerchio che si chiude è largo. Il
compito con cui si addestrano i modelli di diffusione, «indovina il rumore che
ho aggiunto a questa immagine», nasce nel capitolo precedente come una scelta
pratica e felice. Vista da questo capitolo è la soluzione di un problema
vecchio di vent'anni: come dare forma a un paesaggio senza mai misurare il
continente. I modelli di diffusione sono, in questa luce, modelli a energia
addestrati sulla pendenza.

Con una differenza tecnica che vale la pena dire per onestà. Un modello a
energia impara l'altezza del paesaggio, e la pendenza si ricava da quella; un
modello di diffusione impara direttamente la pendenza, una freccia per ogni
punto, e non si preoccupa che esista davvero una superficie di cui quelle
frecce siano la discesa. Sono due cose diverse, e a rigore niente garantisce
che le frecce imparate siano la pendenza di qualcosa. Che sia possibile
sbagliare si vede con quattro frecce: disponile lungo il bordo di un quadrato
in modo che ognuna punti alla successiva, in tondo. Sembrano un pendio, ma
seguendole si torna al punto di partenza dopo essere sempre scesi, e un
paesaggio in cui si scende sempre tornando dove si era non esiste. Ci si guadagna in
stabilità dell'addestramento, e a chi genera immagini l'altra cosa non è mai
importata.

## Terza via: trasformare la densità in una domanda sì o no

La terza strada è la più obliqua e ha il fascino delle idee che spostano il
problema invece di risolverlo. Michael Gutmann e Aapo Hyvärinen, nel 2010,
osservano che stimare una densità è difficile, ma **distinguere** i dati veri
da rumore fabbricato da noi è un problema di classificazione, e a classificare
siamo bravi {cite}`gutmann2010noise`. Il metodo si chiama **stima contrastiva
col rumore**, e la sigla inglese con cui lo si trova ovunque è **NCE**.

`````{tab} Elementare

Invece di chiedere al modello «quanto è probabile questa immagine?», gli si
chiede: «questa l'ho presa dal mondo o l'ho fabbricata io?». Si mescolano
esempi veri e finti (questi ultimi generati da una sorgente di rumore di cui
sappiamo tutto) e si addestra il modello a smistarli. Per riuscirci, il
modello deve implicitamente sapere quanto ogni esempio è tipico dei dati: la
conoscenza che serviva sta tutta lì dentro, ma è arrivata rispondendo a una
domanda facile.

Se suona familiare, è perché lo è. È la stessa mossa di una delle due reti
delle GAN, quella a cui tocca dire se l'immagine che ha davanti viene dal
mondo o l'ha fabbricata l'altra rete. Ed è la stessa mossa con cui si insegna
a un computer a rappresentare con dei numeri le parole di una lingua o i nodi
di un grafo: gli si mostrano accostamenti veri e accostamenti inventati, e gli
si chiede di distinguerli. Per le parole sono gli *word embedding* del
capitolo sul natural language processing; nel capitolo sulle Graph Neural
Network la stessa mossa torna col suo nome inglese, *negative sampling*. La
famiglia è più larga di quanto il nome lasci pensare.

`````

`````{tab} Superiore

La **noise-contrastive estimation** (NCE) affianca ai dati un rumore di
riferimento $p_n$ noto e campionabile, e addestra un classificatore logistico
a distinguere le due sorgenti. Il rumore va scelto **strettamente positivo
dovunque lo siano i dati**, e non è una precauzione da manuale: dove non
arrivano campioni di rumore la densità dei dati non è identificabile, e il
lavoro del 2010 lo enuncia come condizione del teorema, non come consiglio.
Quel lavoro {cite}`gutmann2010noise`
tratta il caso con tanti campioni di rumore quanti dati ($\nu = 1$); nella
formulazione generale, che gli stessi autori danno due anni dopo sul *Journal
of Machine Learning Research* {cite}`gutmann2012noise`, con $\nu$ campioni di
rumore per ogni dato si mescolano le due sorgenti in proporzione
$\tfrac{1}{1+\nu}$ e $\tfrac{\nu}{1+\nu}$, e da Bayes su queste due
probabilità a priori la probabilità a posteriori che $\mathbf{x}$ venga dai dati è

$$
P(\text{dati} \mid \mathbf{x})
= \frac{p_\theta(\mathbf{x})}{p_\theta(\mathbf{x}) + \nu\, p_n(\mathbf{x})}
= \sigma\!\left(\log p_\theta(\mathbf{x}) - \log p_n(\mathbf{x}) - \log \nu\right),
$$

e si massimizza la log-verosimiglianza di questa classificazione binaria. La
mossa decisiva è che $\log Z$ viene trattata come un **parametro in più**,
stimato insieme agli altri: il modello non normalizzato
$\log p_\theta(\mathbf{x}) = -E_\theta(\mathbf{x}) - c$ impara anche $c$, perché al
classificatore la costante *serve* per calibrarsi. Con la massima
verosimiglianza la stessa mossa non è semplicemente inutile, è **impossibile**:
lasciando $c$ libero, la verosimiglianza si fa crescere quanto si vuole
mandando $c \to -\infty$, cioè dichiarando il modello sempre più
«concentrato», e il problema non ha soluzione. È il vincolo di
normalizzazione a impedirlo, ed è esattamente ciò a cui NCE rinuncia
{cite}`gutmann2010noise`.

Il *negative sampling* di word2vec {cite}`mikolov2013distributed` (il secondo
dei due articoli word2vec: il primo usava la softmax gerarchica) è una
semplificazione dichiarata di questa idea, e il
discriminatore delle GAN ne è cugino stretto: in tutti e tre i casi si impara
un rapporto fra densità, non una densità.

`````

## Le tre strade a confronto

{numref}`tab-tre-vie` le mette a confronto.

```{list-table} Tre modi di non pagare il conto del continente, e il prezzo di ciascuno.
:header-rows: 1
:name: tab-tre-vie
:widths: 22 34 44

* - Via
  - Che cosa fa con la misura del continente ($Z$)
  - Che cosa costa
* - **Campionamento** (Langevin e parenti)
  - Non la calcola mai, la sostituisce: quello che serve all'addestramento non
    è il numero, è una media sulle risposte che il modello si immagina, e per
    produrle basta la pendenza sotto i piedi
  - Tempo. Le palline restano intrappolate da una parte quando le montagne
    sono alte, e in alta dimensione lo sono; e da dentro non si vede
* - **Score matching**, cioè imparare la pendenza (e la sua forma *denoising*,
    su dati sporcati apposta)
  - La elimina, perché sparisce nel passaggio dall'altezza alla pendenza
  - Nella forma originale un conto su come la pendenza stessa cambia da un
    punto al vicino (in gergo le derivate seconde, l'hessiana), caro
    in alta dimensione; nella forma *denoising*, il fatto che la pendenza
    imparata è quella dei dati sporcati di rumore, non dei dati
* - **NCE** (*noise-contrastive estimation*, la domanda sì o no) e parenti
  - La stima come un numero qualunque, insieme a tutto il resto
  - Dipende dal rumore che si sceglie: se è troppo diverso dai dati,
    distinguere diventa banale e non si impara nulla
```

Tre modi di non pagare il conto, e nessuno dei tre gratis. Resta una quarta
possibilità, la più radicale, che è anche la tesi della sezione seguente:
**non chiedere mai la probabilità**. Se ciò che serve è decidere,
ordinare, pianificare (non stampare percentuali), l'energia basta da sola, e il
conto non si apre nemmeno.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Misurare l'intero continente non è caro, è impossibile. Cento interruttori
  accesi o spenti danno un numero di configurazioni lungo trentuno cifre: a un
  miliardo di configurazioni al secondo servirebbero quasi tremila volte
  l'età dell'universo, e cento interruttori sono un'immagine in bianco e nero
  di dieci pixel per dieci.
- Imparare vuol dire abbassare il paesaggio dove stanno i dati veri e alzarlo
  dove il modello immagina male. Il primo gesto è facile, i dati ce li
  abbiamo; il secondo no, perché per sapere che cosa il modello immagina
  bisogna prima fargli produrre qualcosa.
- **Prima via, campionare.** La pallina su un tavolo che vibra scende ma ogni
  tanto risale, cambia valle e alla lunga passa più tempo in basso che in
  cima: le serve soltanto la pendenza sotto i piedi, mai la misura del
  continente. Nell'esempio a due valli ricostruisce le proporzioni giuste
  entro pochi millesimi; il prezzo è il tempo, e le montagne alte che la
  tengono prigioniera da una parte sola.
- **Seconda via, la pendenza.** Invece di dire quanta pioggia tocca a ogni
  punto, si dice da che parte si scende e quanto ripido: una descrizione tutta
  locale, che basta a ricostruire la forma del paesaggio. Insegnata su dati
  sporcati apposta, diventa il compito «indovina il rumore che ti ho aggiunto»,
  cioè quello che imparano i modelli di diffusione.
- **Terza via, la domanda sì o no.** Al posto di «quanto è probabile questo?»
  si chiede «viene dal mondo o l'ho fabbricato io?», e si addestra il modello
  a smistare i veri dai finti. Funziona, ma dipende dal rumore che gli si
  mette davanti: se è troppo diverso dai dati, il gioco diventa facile e non
  si impara niente.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- $Z$ non è cara: è impossibile. Con $N = 100$ variabili binarie gli stati
  sono $\approx 1{,}27 \times 10^{30}$, quasi tremila volte l'età
  dell'universo a un miliardo di stati al secondo.
- Il gradiente della log-verosimiglianza ha una **fase positiva** (abbassa
  l'energia sui dati) e una **fase negativa** (la rialza sui campioni del
  modello): è la seconda a richiedere di saper campionare da $p_\theta$.
- **Langevin**:
  $\mathbf{x}_{k+1} = \mathbf{x}_k - \frac{\epsilon}{2}\nabla_{\mathbf{x}} E_\theta(\mathbf{x}_k) + \sqrt{\epsilon}\, \mathbf{z}_k$.
  Usa solo $\nabla_{\mathbf{x}} E$, mai $Z$: nell'esempio a doppia buca
  ricostruisce la distribuzione esatta entro pochi millesimi.
- **Score matching** {cite}`hyvarinen2005estimation` confronta i gradienti
  invece delle densità; la forma **denoising** {cite}`vincent2011connection`
  la riduce a una regressione sul rumore ed è la loss dei modelli di
  diffusione.
- **NCE** {cite}`gutmann2010noise` trasforma la stima di densità in una
  classificazione dati contro rumore, con $\log Z$ come parametro. Il
  *negative sampling* di word2vec è suo discendente.
```
`````
