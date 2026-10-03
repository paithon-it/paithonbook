# Oltre la partizione: tre modi di aggirare $Z$

La macchina di Boltzmann ha lasciato in eredità $Z$, la funzione di
partizione: la somma su *tutte* le configurazioni possibili. È lei a dettare
tutto ciò che segue. Per un'energia scelta senza vincoli sull'architettura, in
alta dimensione, non si riesce a calcolarla, e questo è un muro più che una
difficoltà tecnica fra le tante: i modelli della {doc}`verosimiglianza esatta
</VerosimiglianzaEsatta/overview>` lo evitano proprio pagando in vincoli sulla
forma della rete.

Una rete di venticinque neuroni accesi o spenti, come quella della memoria
associativa, ha trentatré milioni di configurazioni ($2^{25} = 33\,554\,432$),
e $Z$ si calcola davvero: percorrerle tutte e valutare l'energia di ciascuna,
che è il conto vero, prende meno di un minuto su un computer qualunque.

Aggiungiamone settantacinque. Con cento neuroni le configurazioni diventano un
numero lungo trentuno cifre ($2^{100} \approx 1{,}27 \times 10^{30}$), e anche
regalando a quel computer una velocità mille volte superiore a quella che ha,
un miliardo di configurazioni al secondo, servirebbero circa
$4 \times 10^{13}$ anni per percorrerle: quasi tremila volte l'età
dell'universo. E cento neuroni accesi o spenti sono un'immagine in bianco e
nero di dieci pixel per dieci, molto meno di una fotografia. Il costo cresce
come $2^N$, e nessun miglioramento dell'hardware fa sparire trenta zeri: per
un'energia qualunque in alta dimensione, un metodo che deve calcolare $Z$ è
escluso in partenza.

`````{tab} Elementare

È la stessa carta in rilievo dell'apertura del capitolo, con tutte le
risposte possibili messe una accanto all'altra, ma guardata per quello che è:
grande come un continente, e da qui in avanti la chiameremo così. E su quel
continente piove. L'acqua scende e si raccoglie in basso, quindi le valli si
riempiono e le cime restano asciutte, e la pioggia raccolta è la probabilità.
Quanta ne raccolga una valle dipende da due cose, da quanto è profonda e da
quanto è larga: un pozzo strettissimo può essere profondissimo e raccogliere
pochissimo, una conca larga e appena accennata può raccogliere molto.

Confrontare due valli è gratis: si guardano le due altezze. Dire invece che una
valle raccoglie «il 30% di tutta la pioggia che cade sul continente» vuol dire
aver girato il continente intero, valle per valle: quella misura è la funzione
di partizione, ed è ciò che trasforma un'altezza in una percentuale.

E perché mai dovremmo misurarlo? Perché imparare, per un modello a energia,
sono due gesti e non uno: scavare il paesaggio dove stanno i dati veri, e
rialzarlo dove il modello si immagina roba che non esiste. Sono la veglia e il
sogno della macchina di Boltzmann, con altri nomi. Il primo è facile, i
dati ce li abbiamo in mano. Il secondo no: per sapere che cosa il modello si
immagina bisogna prima fargli produrre qualcosa, e produrlo *nelle proporzioni
giuste* sembra richiedere di conoscere il continente intero. E i gesti sono due
per forza: se ci si limitasse a scavare, il modello troverebbe subito la
scorciatoia, cioè abbassare tutto dappertutto e dire di sì a qualunque cosa gli
si presenti. L'alzare è il gesto che costa, ed è il motivo per cui la misura
del continente continua a ripresentarsi.

Il continente è grande quanto tutte le immagini possibili, e non lo si
percorre. Restano tre mosse. *Campionarlo*: mandare esploratori a caso e
accontentarsi di quello che riportano. *Evitarlo*: descrivere il paesaggio con
le pendenze invece che con le percentuali, che è una descrizione locale e non
chiede nessuna misura d'insieme. Oppure *aggirarlo*: sostituire la domanda
«quanto è probabile questo?» con «questo viene dai dati o l'ho inventato io?»,
che è una domanda da sì o no, e a rispondere sì o no sappiamo addestrare un
classificatore da trent'anni. La prima delle tre è la più sorprendente:
camminando secondo la regola giusta, le proporzioni vengono da sé, a lungo
andare, e quel «sembra richiedere» era di troppo.

`````

`````{tab} Superiore

Da qui in avanti si pone $T = 1$, così che $Z$ dipenda soltanto dai
parametri: $Z(\theta)$. Il punto di attrito è il gradiente della
log-verosimiglianza. Da
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

Tre mosse, e vanno nominate: si scambiano derivata e integrale (lecito per
convergenza dominata, se $\nabla_\theta e^{-E_\theta}$ è dominata da una
funzione integrabile in un intorno di $\theta$); si deriva l'esponenziale; e
si riconosce che il rapporto rimasto sotto integrale è la densità del
modello, il che trasforma un integrale su tutto lo spazio in un valore atteso
che si può stimare per campionamento. Mettendo insieme:

$$
\nabla_\theta \log p_\theta(\mathbf{x})
= -\nabla_\theta E_\theta(\mathbf{x})
+ \mathbb{E}_{\mathbf{x}' \sim p_\theta}\!\left[\nabla_\theta E_\theta(\mathbf{x}')\right].
$$

La log-verosimiglianza si massimizza, quindi i parametri si muovono nel
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
(Monte Carlo, metodi variazionali) si possono vedere come strategie diverse
per scegliere quali risposte tirare su. Le tre sezioni che seguono sono, in
questa luce, tre risposte alla stessa domanda: chi solleviamo, e come?

`````

## Prima via: campionare il paesaggio

Se il conto esatto su tutto il paesaggio non si può fare, lo si può stimare
visitandone dei pezzi: non calcolare, campionare. Il modo classico è far
camminare uno stato sul paesaggio, un passo dopo l'altro, con una regola che
guarda soltanto dove si trova adesso, e segnare ogni tanto dove è arrivato. Una
successione di stati fatta così è una {doc}`catena di Markov
</Matematica/catene-di-markov>`, e il metodo si chiama *Monte Carlo a catena di
Markov*, in sigla MCMC. Se la regola ha $p_\theta$
come distribuzione stazionaria, la frazione di tempo che la catena passa in una
regione tende alla probabilità di quella regione, e gli stati visitati dopo un
periodo iniziale di assestamento sono un campione di $p_\theta$. È però un
campione correlato: due stati vicini lungo la catena si somigliano, e per la
stessa precisione servono più punti di quanti ne servirebbero se fossero
pescati indipendenti. Nella pratica se ne fanno camminare migliaia in
parallelo.

Quando le risposte non sono acceso e spento ma numeri con la virgola
(un'immagine vera, per dire, dove ogni pixel può avere qualunque sfumatura), la
regola più usata porta il nome del fisico francese Paul Langevin, che nel 1908
la scrisse per il moto browniano, il tremolio di un granello di polline
sull'acqua sotto gli urti delle molecole {cite}`langevin1908theorie`. È quasi
uno slogan: scendere lungo la pendenza dell'energia, con addosso un po’ di
rumore. E «rumore», qui, non ha niente a che fare con i suoni: vuol dire una
spintarella a caso, diversa a ogni passo, che non si sa da che parte arriverà.
Nelle due vie che seguono la parola cambierà ancora mestiere: nella seconda
sarà lo sporco aggiunto apposta a un dato vero, nella terza gli esempi finti
fabbricati per confronto.

`````{tab} Elementare

Una pallina che rotola in discesa finisce nel fondovalle più vicino e lì si
ferma: è la dinamica di Hopfield, e produce sempre la stessa risposta. La
stessa pallina su un tavolo che vibra, invece, continua a scendere, perché
la pendenza c'è ancora, ma i sussulti la fanno anche risalire un po’,
uscire dalle conche, passare da una valle all'altra. Se la guardi per molto
tempo e segni dove si trova, scoprirai che passa più tempo dove il
paesaggio è basso e pochissimo sulle cime: la frequenza con cui visita
ogni punto *è* la probabilità che il paesaggio definisce.

Questo è il punto elegante della faccenda, ed è il perno di tutta la sezione:
per far vibrare e scendere la pallina serve solo la pendenza locale, quella
sotto i suoi piedi. La misura dell'intero continente, quella che non sappiamo
calcolare, serve a una cosa sola: dividere per il totale la pioggia di ogni
valle, cioè fare lo stesso identico gesto in ogni punto. E fra le due lingue il
cambio è questo: scendere di un gradino non aggiunge una quantità fissa di
pioggia, la moltiplica per un fattore fisso, sempre lo stesso. Se ogni gradino
in giù raddoppia la pioggia, una valle tre gradini più bassa di un'altra ne
raccoglie otto volte tanto, dovunque stiano le due. Leggiamo la stessa regola
al contrario: salire di un gradino dimezza la pioggia, salire di tre la divide
per otto, e questo in ogni punto del continente. Dividere tutta la pioggia per
otto, allora, è la stessa cosa che alzare tutto il paesaggio di tre gradini, e
dividerla per un numero qualunque è alzarlo dappertutto di una stessa
quantità. Ma alzare l'intero paesaggio di dieci metri non cambia di un grado
nessuna salita e nessuna discesa. La pallina, che
sente solo il pendio sotto i piedi, non se ne accorgerebbe nemmeno, e infatti
non ha bisogno di conoscerlo.

Il prezzo è il tempo. Se due valli sono separate da una montagna alta, la
pallina può restare intrappolata a lungo da una parte, e la fotografia che
ne ricavi è sbilanciata. E l'attesa non cresce in proporzione all'altezza:
una collinetta dieci volte più alta si scavalca centinaia di volte meno
spesso.

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
\qquad \mathbf{z}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{I}),
$$

dove $\epsilon > 0$ è il passo (un tempo, non una lunghezza) e $\mathbf{z}_k$
il rumore gaussiano. Per $k \to \infty$, con $\epsilon \to 0$ e $k\epsilon \to
\infty$ (il passo si accorcia, ma il tempo totale percorso dalla catena deve
crescere senza limite), la distribuzione di $\mathbf{x}_k$ converge a $p_\theta
\propto e^{-E_\theta}$. Il teorema vuole però anche delle ipotesi sul
paesaggio, che Roberts e Tweedie hanno reso precise mostrando, fra l'altro, che
la diffusione può convergere mentre la sua discretizzazione non converge
affatto {cite}`roberts1996exponential`; e l'esempio a doppia buca ne viola una:
$\nabla_{\mathbf{x}} E_\theta$ globalmente lipschitziano, o almeno una
condizione di dissipatività che tenga la catena al finito. Con $E(x) =
(x^2-1)^2$ il gradiente cresce come $x^3$, non è lipschitziano, e a passo
fissato la ricorsione diverge oltre una soglia: $|1 - 2\epsilon(x^2-1)| > 1$,
cioè $|x| > \sqrt{1 + 1/\epsilon}$, che a $\epsilon = 0{,}01$ vale $10{,}05$
(da $10{,}00$ la catena torna in una buca, da $10{,}05$ esplode in nove passi).
Non si vede mai, perché lassù la densità vale $e^{-9800}$, ma è una divergenza
vera e non un'approssimazione: quella catena, a rigore, è transiente. A passo
fissato, come nel codice della doppia buca e nella pratica degli EBM, la catena
si assesta poi su una distribuzione leggermente distorta, con un errore
dell'ordine di $\epsilon$: lo eliminerebbe un test di accettazione alla
Metropolis (la variante MALA), a cui di solito si rinuncia in cambio della
semplicità. Si noti che compare solo $\nabla_{\mathbf{x}} E_\theta$: la
costante $\log Z(\theta)$, non dipendendo da $\mathbf{x}$, ha gradiente nullo.
Il campionamento non ha mai bisogno della normalizzazione: è l'osservazione su
cui poggia tutto il resto della sezione. Dove le ipotesi valgono, la
convergenza può essere comunque lentissima: il tempo medio per scavalcare una
barriera di altezza $\Delta E$ cresce come $e^{\Delta E}$ (la legge di Kramers
{cite}`kramers1940brownian`), e un paesaggio con molte buche separate da
barriere alte si campiona in tempi esponenziali nell'altezza delle barriere.

La versione stocastica su minibatch, che sostituisce il gradiente esatto con
quello stimato, è la *stochastic gradient Langevin dynamics*
{cite}`welling2011bayesian`, nata per campionare la distribuzione a posteriori
dei *parametri* e poi passata di peso al campionamento dei dati: un passo di
discesa dimezzato ($\epsilon/2$) più un rumore di deviazione standard
$\sqrt{\epsilon}$. Il $\tfrac12$ e la radice sono ciò che rende la ricorsione
la discretizzazione di Eulero–Maruyama della diffusione $d\mathbf{x} =
-\tfrac12 \nabla_{\mathbf{x}} E_\theta\, dt + d\mathbf{W}$, che ha $p_\theta$
come misura invariante. Le due ampiezze non si confrontano fra loro: hanno
dimensioni diverse (la discesa va come $[\text{tempo}]$, il rumore come
$[\text{tempo}]^{1/2}$), e su un intervallo di tempo fissato i due contributi
restano dello stesso ordine, che è precisamente il motivo per cui il limite
continuo esiste. Quello che in Welling e Teh diventa trascurabile al decrescere
del passo è un'altra cosa ancora: il rumore del gradiente su minibatch, che
scala come $\epsilon$ e finisce sotto quello iniettato; ed è lì che la catena
passa senza soluzione di continuità dall'ottimizzazione al campionamento. Nella
pratica degli EBM la catena si tronca dopo poche decine di passi (*short-run
MCMC*), e la si fa ripartire dal rumore a ogni aggiornamento
{cite}`nijkamp2019learning` oppure da un serbatoio di campioni passati
{cite}`du2019implicit`; il serbatoio è l'erede diretto della persistent
contrastive divergence della sezione precedente.

`````

Il codice che segue costruisce il paesaggio più semplice in cui la faccenda si
vede: due valli e una collinetta in mezzo. In formula è l'energia a doppia
buca $E(x) = (x^2 - 1)^2$, e i conti si fanno a mente: in $x = 1$ e in
$x = -1$ la parentesi vale zero, quindi l'energia vale zero (sono i due
fondovalle), mentre in $x = 0$ vale $(0^2-1)^2 = 1$, che è la collinetta.
Ci mette sopra ventimila palline (nel codice si chiamano `catene`, che è il
nome tecnico di poco fa), le fa vibrare con la ricetta di Langevin e alla fine
guarda dove si sono distribuite, senza aver mai calcolato $Z$. Poi, per pura
verifica, $Z$ la calcola: in un paesaggio a una sola dimensione si può, ed è
l'unico modo per sapere se il campionamento ha detto il vero. Nella tabella
che ne esce, la colonna «campioni» dice dove sono finite le palline e la
colonna «esatto» dove sarebbero dovute finire. Ogni riga raccoglie le palline
finite in un tratto di paesaggio, per esempio fra $-0,5$ e $+0,5$; un tratto
così, in gergo, si chiama **bin**, ed è la parola che tornerà nei conti che
seguono.

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

Le due colonne coincidono entro pochi millesimi: lo scarto più largo vale
0,006, e capita in due righe, quella centrale e quella subito a destra. Le
catene hanno ricostruito le proporzioni giuste senza che $Z$ sia mai entrata
nel ciclo.

Quei pochi millesimi, però, non sono tutti fortuna del sorteggio. Dentro c'è
anche un errore che c'è sempre e sempre nello stesso verso, ed è colpa del
passo: la pallina non scivola giù per il pendio con continuità, lo scende a
saltelli, e $\epsilon$ (nel codice, `eps`) è quanto dura ogni saltello. Una
scala di gradini non è una rampa, e con saltelli di durata finita resta uno
scarto che nessuna quantità di catene fa sparire.

Separare i due effetti ripetendo le esecuzioni costerebbe caro. Con ventimila
catene, due esecuzioni che differiscono soltanto nel sorteggio ballano di
qualche millesimo su un bin, cioè quanto l'effetto da misurare, e il ballo cala
solo come la radice quadrata del numero di esecuzioni: per dimezzarlo ne
servono quattro volte tante. In una dimensione, però, la risposta esatta si
calcola senza tirare nemmeno una pallina, come si è fatto per $Z$. La regola
con cui la catena si sposta si scrive come una matrice su una griglia fine (la
riga di un punto dice con che probabilità da lì si arriva in ciascun altro), e
la distribuzione su cui la catena a passo $\epsilon$ si assesta davvero è
l'unica che quella matrice lascia identica a se stessa, il suo autovettore di
Perron.

```python
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs

energia = lambda x: (x**2 - 1.0)**2
gradiente = lambda x: 4.0 * x * (x**2 - 1.0)

# la distribuzione su cui la catena a passo eps si assesta: quella che la
# regola di transizione, scritta come matrice su una griglia fine, lascia ferma
griglia = np.linspace(-2.6, 2.6, 5201)
centro = (griglia >= -0.5) & (griglia < 0.5)
esatta = np.exp(-energia(griglia))
esatta /= esatta.sum()
for eps in (0.01, 0.002, 0.0005):
    media = griglia - 0.5 * eps * gradiente(griglia)
    K = np.exp(-(griglia[None, :] - media[:, None])**2 / (2 * eps))
    K[K < 1e-12] = 0.0
    K = sparse.csr_matrix(K / K.sum(1, keepdims=True))
    _, v = eigs(K.T, k=1, which="LM")
    p = np.abs(v[:, 0].real)
    p /= p.sum()
    scarto = p[centro].sum() - esatta[centro].sum()
    print(f"eps = {eps:<7} scarto sul bin centrale {scarto:+.5f}"
          f"   scarto/eps {scarto / eps:.3f}")
```

```text
eps = 0.01    scarto sul bin centrale +0.00357   scarto/eps 0.357
eps = 0.002   scarto sul bin centrale +0.00071   scarto/eps 0.353
eps = 0.0005  scarto sul bin centrale +0.00018   scarto/eps 0.352
```

Lo scarto è sempre positivo (la barriera è sovrappesata), e il rapporto fra lo
scarto e la durata del saltello resta fra $0{,}35$ e $0{,}36$ mentre il passo
si accorcia di venti volte. Lo scarto, cioè, cala esattamente in proporzione al
passo: passo cinque volte più corto, scarto cinque volte più piccolo, e zero
soltanto al limite di saltelli di durata nulla. Sparirebbe anche in un altro
modo: aggiungendo dopo ogni saltello un controllo che, come nella ricottura
simulata, accetta la mossa sempre oppure soltanto con una certa probabilità,
con una regola tarata perché le proporzioni finali tornino esatte. È il test di
accettazione di Metropolis, e i modelli a energia ci rinunciano per
semplicità.

La lezione vale ben oltre questo esempio. A $\epsilon = 0{,}002$ l'effetto vero
vale sette decimillesimi, e un'esecuzione sola balla di più: l'effetto c'è
sempre, ma è più piccolo di quanto i numeri ballino da un sorteggio all'altro.
Un numero solo, per quanto stampato con quattro cifre, non dimostra niente se
non si sa di quanto balla.

Qui funziona bene per una ragione che non si generalizza. La collinetta fra le
due buche è alta un'unità di energia, e con la temperatura a uno un'unità è
esattamente la salita che le spintarelle casuali fanno fare a una pallina senza
sforzarsi: barriere così si scavalcano di continuo. Ma la difficoltà di
superare una barriera non cresce in proporzione alla sua altezza, cresce molto
più in fretta. Lo si vede moltiplicando l'energia per un'altezza $h$,
$E(x) = h\,(x^2 - 1)^2$, e contando quante volte ventimila catene passano da
una buca all'altra in duemila passi, con $h = 1$ e con $h = 10$. Le catene
partono metà in una buca e metà nell'altra (con la collinetta alta dieci,
partire lontano dalle buche come nella prima simulazione farebbe esplodere la
ricorsione), e un passaggio conta solo quando la catena arriva in fondo
all'altra buca, oltre $\pm 0{,}8$, così che i saltelli avanti e indietro sulla
cima non vengano contati come scavalcamenti.

```python
import numpy as np

def scavalcamenti(h, eps=0.01, passi=2000, catene=20000, seme=0):
    """Passaggi da una buca all'altra di E(x) = h (x^2 - 1)^2."""
    rng = np.random.default_rng(seme)
    x = rng.choice([-1.0, 1.0], size=catene)   # metà per buca
    lato = x.copy()                            # l'ultima buca toccata
    conta = 0
    for _ in range(passi):
        pendenza = 4.0 * h * x * (x**2 - 1.0)
        x = x - 0.5 * eps * pendenza + np.sqrt(eps) * rng.normal(size=catene)
        # si cambia lato solo arrivando in fondo all'altra buca, oltre ±0,8
        nuovo = np.where(x > 0.8, 1.0, np.where(x < -0.8, -1.0, lato))
        conta += int((nuovo != lato).sum())
        lato = nuovo
    return conta

basso, alto = scavalcamenti(1.0), scavalcamenti(10.0)
print(f"collinetta alta  1: {basso:6d} passaggi")
print(f"collinetta alta 10: {alto:6d} passaggi")
print(f"rapporto {basso / alto:.0f}; "
      f"legge di Kramers, e^9 / 10 = {np.exp(9) / 10:.0f}")
```

```text
collinetta alta  1:  58934 passaggi
collinetta alta 10:     70 passaggi
rapporto 842; legge di Kramers, e^9 / 10 = 810
```

Con la collinetta alta uno le catene passano da una buca all'altra quasi
sessantamila volte, alta dieci settanta volte: quasi tre ordini di grandezza in
meno. È l'ordine che prevede la legge di Kramers {cite}`kramers1940brownian`:
la frequenza degli scavalcamenti cala come $e^{-\Delta E}$, l'esponenziale
dell'altezza della barriera, moltiplicato per un fattore che dipende da quanto
sono curvi il fondo della buca e la cima della collinetta. Qui
$\Delta E = h$ e quel fattore cresce come $h$, quindi da $h = 1$ a $h = 10$ la
frequenza cala di $e^{9}/10$, circa 810 volte. Settanta passaggi sono pochi, e
da un sorteggio all'altro quel numero balla parecchio; l'ordine di grandezza
no. Alzando ancora la collinetta, o passando a mille dimensioni dove le valli
sono separate da creste lunghissime, la stessa procedura darebbe una fotografia
sbilanciata, e nessuno se ne accorgerebbe: in alta dimensione la colonna
«esatto» non si può stampare.

## Seconda via: imparare la pendenza, non la probabilità

Se il campionamento è costoso perché insegue le percentuali, si può cambiare
bersaglio. Aapo Hyvärinen, nel 2005, propone di smettere di confrontare
*quanta* probabilità il modello mette in ogni punto, e di confrontare invece
la pendenza del paesaggio in quel punto {cite}`hyvarinen2005estimation`.
Sembra un dettaglio ed è una liberazione, per una ragione che si dice in una
riga: la misura dell'intero continente è un numero solo, lo stesso
dappertutto, e un numero uguale dappertutto non ha pendenza. Cambiando
bersaglio, sparisce.

`````{tab} Elementare

A chi non lo vedrà mai, un paesaggio si può descrivere in due modi. Puoi
dirgli, per ogni punto, «qui c'è il 3% della pioggia», e per farlo devi aver
misurato tutto il continente. Oppure «da qui si scende verso nord-est, con
questa pendenza»: una descrizione tutta locale, che non chiede di conoscere il
continente. Eppure basta a ricostruire la forma del paesaggio, e quello che si
perde per strada, a che altezza stia nel suo insieme, per produrre risposte non
serve.

Si perde però anche dell'altro, ed è meno innocuo. Se il continente è fatto di
due regioni separate da un deserto dove non piove mai, le pendenze dicono
benissimo com'è fatta ciascuna e non dicono quanta pioggia tocchi all'una
rispetto all'altra: per confrontarle bisognerebbe camminare dall'una all'altra,
e di strada non ce n'è. Un modello che dà metà della pioggia alla prima regione
e uno che gliene dà un decimo disegnano le stesse identiche pendenze. Quando i
dati stanno in gruppi ben separati, che è il caso normale, è lì che questi
metodi sbagliano le proporzioni.

La carta è sempre quella, guardata da sopra o da sotto: l'altezza è l'energia,
e dove il paesaggio scende la pioggia aumenta. La sua pendenza è quella che il
{doc}`capitolo sulla diffusione </ModelliDiffusione/sde-e-ode>` chiamava
punteggio, o *score*: la freccia che in ogni punto indica da che parte
la pioggia si fa più fitta.

Il prezzo si paga al momento di disegnare la carta. In ogni punto va
controllato di quanto la pendenza cambia facendo un passo, e il controllo va
rifatto in ogni direzione: su una collina le direzioni indipendenti sono due,
nel paesaggio di una fotografia sono un milione, e il lavoro non finisce
più.

La scappatoia è sporcare apposta. Si prende un punto vero, gli si dà una
spintarella a caso e si chiede da che parte è arrivato. Chi indovina la
spintarella sta indicando la strada per tornare al punto vero, e siccome i
punti veri stanno nelle valli quella strada è la discesa: gli è bastata una
misura. Il
prezzo è che la carta descrive il paesaggio come si vede dopo la spinta, coi
dettagli fini smussati. Più corta la spinta, più fedele la carta, e una spinta
ci deve essere: per questo i generatori di immagini non ne usano una sola, ma
una scala di spinte.

`````

`````{tab} Superiore

Lo score di una densità è $s(\mathbf{x}) = \nabla_{\mathbf{x}} \log p(\mathbf{x})$ (la lettera $s$ qui
non ha niente a che vedere con lo stato della rete delle sezioni precedenti:
cambia mestiere). Per un modello a energia,

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
campione. Le ipotesi contano.
L'integrazione per parti richiede regolarità, $p_{\text{dati}}$
differenziabile, i due valori attesi
$\mathbb{E}\lVert\nabla_{\mathbf{x}}\log p_\theta\rVert^2$ e
$\mathbb{E}\lVert\nabla_{\mathbf{x}}\log p_{\text{dati}}\rVert^2$ finiti,
e un decadimento all'infinito
($p_{\text{dati}}(\mathbf{x})\, \nabla_{\mathbf{x}} \log p_\theta(\mathbf{x})
\to 0$ per $\lVert\mathbf{x}\rVert \to \infty$, che è ciò che annulla il
termine di bordo); per concludere che il minimo di $J$ identifica il modello
serve in più la densità del modello strettamente positiva ovunque,
ipotesi che nel caso ben specificato si trasmette ai dati
{cite}`hyvarinen2005estimation`.

È lì che le cose si rompono davvero, e per due motivi diversi. Il primo: i dati
veri vivono su una varietà di dimensione molto minore dello spazio in cui
stanno (una fotografia di volti non riempie $\mathbb{R}^{D}$), e la positività
ovunque salta. Il secondo: quando il supporto si spezza in pezzi separati lo
score smette di identificare la densità, perché due densità che differiscono di
un fattore costante da una componente all'altra hanno lo stesso score. La
seconda osservazione la mettono a fuoco, quindici anni dopo l'articolo del
2005, Li K. Wenliang e Heishiro Kanagawa, che la chiamano un fatto poco noto e
ne misurano il danno sulle miscele {cite}`wenliang2020blindness`: è la ragione
per cui questi metodi ne sbagliano i pesi. Sotto le ipotesi
del teorema {cite}`hyvarinen2005estimation`:

$$
J(\theta) = \mathbb{E}_{\mathbf{x} \sim p_{\text{dati}}}
\left[ \operatorname{tr}\!\big(\nabla_{\mathbf{x}}^2 \log p_\theta(\mathbf{x})\big)
+ \tfrac{1}{2} \left\lVert \nabla_{\mathbf{x}} \log p_\theta(\mathbf{x}) \right\rVert^2 \right]
+ \text{cost.},
$$

dove $\nabla_{\mathbf{x}}^2$ è la matrice hessiana rispetto a $\mathbf{x}$ e la
costante, che vale $\tfrac{1}{2}\mathbb{E}\lVert\nabla_{\mathbf{x}} \log
p_{\text{dati}}\rVert^2$, non dipende da $\theta$
{cite}`hyvarinen2005estimation`. È lei a saldare le due forme, e la prima dice
*perché* $J$ è un obiettivo sensato: essendo la prima forma una media di norme
al quadrato, $J \ge 0$, e $J = 0$ se e solo se i due score coincidono quasi
ovunque. Niente $Z$, niente catene di Markov: solo derivate del modello. Il
costo si è spostato sulla traccia dell'hessiana, e va quantificato, perché è
l'unico costo del capitolo che si lascia contare: sono $D$ retropropagazioni
per ogni esempio, con $D$ la dimensione del dato. Su un'immagine è proibitivo.
Lo *sliced score matching* lo evita proiettando gli score su direzioni casuali
$\mathbf{v} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ prima di confrontarli:
la traccia diventa $\mathbb{E}_{\mathbf{v}}\big[\mathbf{v}^\top
\nabla_{\mathbf{x}}^2 \log p_\theta(\mathbf{x})\, \mathbf{v}\big]$, che chiede
un solo prodotto hessiana-vettore per proiezione, cioè due retropropagazioni
invece di $D$ {cite}`song2020sliced`.

Il colpo di scena arriva nel 2011: Pascal Vincent dimostra che lo score
matching su dati perturbati con rumore gaussiano equivale, a meno di
costanti, ad addestrare un *denoising autoencoder*
{cite}`vincent2011connection`. Con $\tilde{\mathbf{x}} = \mathbf{x} + \sigma \boldsymbol{\varepsilon}$ e
$\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, dove $\sigma$ è qui la deviazione
standard del rumore e non la sigmoide di poco fa, e $\boldsymbol{\varepsilon}$
è il rumore iniettato e non il passo $\epsilon$ della catena di Langevin
(stessa lettera greca, due mestieri: qui è un vettore, e va in grassetto), il
bersaglio dello score sul dato perturbato è noto in forma chiusa,
$\nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) = -(\tilde{\mathbf{x}} - \mathbf{x})/\sigma^2 = -\boldsymbol{\varepsilon}/\sigma$,
e l'obiettivo diventa una regressione: predire il rumore iniettato.

Resta però da capire perché regredire sullo score condizionato a $\mathbf{x}$
dia lo score della marginale $q_\sigma(\tilde{\mathbf{x}})$, che è quello
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
modelli di diffusione {cite}`song2021score` a meno di una riponderazione per
livello di rumore, che pesa: senza di essa il bersaglio
$-\boldsymbol{\varepsilon}/\sigma$ farebbe esplodere il peso dei livelli di
rumore piccoli, e il fattore che si usa (proporzionale a $\sigma^2$) è
precisamente quello che cancella l’$1/\sigma$ e lascia la regressione sul
rumore in forma pulita.

Il prezzo c'è, e non è quello che si direbbe: ciò che si impara è lo score dei
dati sporcati e non quello dei dati, cioè della densità marginale
$q_\sigma$, che è $p_{\text{dati}}$ convoluta con la gaussiana (e si noti che
$q_\sigma(\tilde{\mathbf{x}})$ e $q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})$
sono due oggetti diversi, come sempre nella notazione delle densità). I due
score coincidono solo nel limite
$\sigma \to 0$, e a $\sigma$ finito resta un errore sistematico che nessuna
quantità di dati riduce. È il motivo più facile da vedere per cui i modelli di
diffusione non usano un solo livello di rumore ma un'intera scala di livelli,
e il campionamento deve attraversarli in fila.

`````

Il cerchio che si chiude qui è largo. Il compito con cui si addestrano i
modelli di diffusione, «indovina il rumore che ho aggiunto a questa immagine»,
nasce nel capitolo che porta il loro nome come una scelta pratica e felice.
Vista dai modelli a energia è la soluzione di un problema vecchio di vent'anni:
come dare forma a un paesaggio senza mai misurare il continente. I modelli di
diffusione sono, in questa luce, modelli a energia addestrati sulla pendenza.

Con una differenza tecnica da dire per onestà. Un modello a energia impara
l'altezza del paesaggio, e la pendenza si ricava da quella; un modello di
diffusione impara direttamente la pendenza, una freccia per ogni punto, e non
si preoccupa che esista davvero una superficie di cui quelle frecce siano la
discesa. Sono due cose diverse, e a rigore niente garantisce che le frecce
imparate siano la pendenza di qualcosa: un campo di vettori è il gradiente di
una funzione solo se il suo rotore è nullo, e una rete che li impara uno per
uno non ha nessun vincolo che lo imponga. Che sia possibile sbagliare si vede
con quattro frecce: disponile lungo il bordo di un quadrato in modo che ognuna
punti alla successiva, in tondo. Sembrano un pendio, ma seguendole si torna al
punto di partenza dopo essere sempre scesi, e un paesaggio in cui si scende
sempre tornando dove si era non esiste. In cambio l'addestramento è più
stabile, e a chi genera immagini che il paesaggio esista davvero non è mai
importato.

## Terza via: cambiare la domanda, e chiederne una da sì o no

La terza strada è la più obliqua e ha il fascino delle idee che spostano il
problema invece di risolverlo. Michael Gutmann e Aapo Hyvärinen, nel 2010,
osservano che dire quanto è probabile un dato è difficile, mentre
distinguere i dati veri da roba fabbricata da noi è un problema di
classificazione, e a classificare siamo bravi {cite}`gutmann2010noise`. Il
metodo si chiama **stima contrastiva col rumore**, dove «rumore» sono appunto
gli esempi finti che ci fabbrichiamo, e la sigla inglese con cui lo si trova
ovunque è **NCE**.

`````{tab} Elementare

Invece di chiedere al modello «quanto è probabile questa immagine?», gli si
chiede: «questa l'ho presa dal mondo o l'ho fabbricata io?». Si mescolano
esempi veri e finti (fabbricati da una sorgente di rumore di cui sappiamo
tutto) e si addestra il modello a smistarli. Per riuscirci, il modello deve
implicitamente sapere quanto ogni esempio è tipico dei dati: la conoscenza che
serviva sta tutta lì dentro, ma è arrivata rispondendo a una domanda facile.

Per smistare serve anche l'asticella: quanto tipica deve essere un'immagine
perché la si dichiari vera. Quel livello è la misura del continente, e qui
diventa un numero da imparare come tutti gli altri, ritoccato finché lo
smistamento torna. Con la domanda sulle probabilità quel numero non si poteva
lasciare libero: chi rispondeva poteva dichiarare tutto sempre più probabile, e
per smentirlo bisognava aver sommato il continente. Smistando si tradisce al
primo giro, perché a furia di alzare comincia a chiamare veri anche i finti. E
quando lo smistamento torna, quel numero è proprio la misura del continente,
che nessuno ha mai fatto.

Il gioco però vale quanto la fabbrica dei finti. Se sforna macchie grigie e i
veri sono fotografie, smistare è banale: il modello impara a riconoscere il
grigio e nient'altro. E dove la fabbrica non arriva mai non c'è confronto: lì
il modello dice quello che vuole.

La stessa mossa la fa una delle due reti delle GAN, quella a cui tocca dire se
l'immagine che ha davanti viene dal mondo o l'ha fabbricata l'altra rete. Ed è
la stessa con cui si insegna a un computer a dare dei numeri alle parole di una
lingua o ai nodi di un grafo: gli si mostrano accostamenti veri e accostamenti
inventati, e gli si chiede di distinguerli. Per le parole sono i *word
embedding* della {doc}`sezione su come si rappresenta il testo
</NaturalLanguageProcessing/rappresentare-testo>`; nella {doc}`sezione sul
mondo come grafo </GraphNeuralNetwork/dati-a-grafo>`, più avanti nel libro, la
stessa mossa si ritroverà col suo nome inglese, *negative sampling*. La famiglia
è più larga di quanto il nome lasci pensare.

`````

`````{tab} Superiore

La noise-contrastive estimation (NCE) affianca ai dati un rumore di
riferimento $p_n$ noto e campionabile, e addestra un classificatore logistico
a distinguere le due sorgenti. Il rumore va scelto strettamente positivo
dovunque lo siano i dati, e non è una precauzione da manuale: dove non
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

dove $\sigma$ è di nuovo la sigmoide, non la deviazione standard del rumore di
poco fa. Si massimizza la log-verosimiglianza di questa classificazione binaria,

$$
J(\theta) = \mathbb{E}_{\mathbf{x} \sim p_{\text{dati}}}\big[\log h_\theta(\mathbf{x})\big]
+ \nu\, \mathbb{E}_{\mathbf{x} \sim p_n}\big[\log\big(1 - h_\theta(\mathbf{x})\big)\big],
$$

con $h_\theta(\mathbf{x}) = P(\text{dati} \mid \mathbf{x})$ come sopra. Se $p_n$
è positiva dove lo è $p_{\text{dati}}$ e il modello contiene la distribuzione
vera, lo stimatore è consistente per ogni $\nu$; per $\nu \to \infty$ la sua
varianza asintotica smette di dipendere dal rumore e, se il modello è già
normalizzato, raggiunge quella della massima verosimiglianza. A $\nu$ fissato
gli autori consigliano un rumore che somigli ai dati almeno in qualche aspetto,
per esempio nella covarianza, perché con un rumore troppo diverso la
classificazione è facile e insegna poco {cite}`gutmann2012noise`. La mossa
decisiva è che $\log Z$ viene trattata come un parametro in più, stimato insieme
agli altri: il modello non normalizzato
$\log p_\theta(\mathbf{x}) = -E_\theta(\mathbf{x}) - c$ impara anche $c$, perché
al classificatore la costante *serve* per calibrarsi. Con la massima
verosimiglianza la stessa mossa è impossibile, non soltanto inutile: lasciando
$c$ libero, la verosimiglianza si fa crescere quanto si vuole mandando
$c \to -\infty$, cioè dichiarando una densità sempre più alta in ogni punto, e
il problema non ha soluzione. È il vincolo di normalizzazione a impedirlo, ed è
esattamente ciò a cui NCE rinuncia {cite}`gutmann2010noise`. La rinuncia,
però, non costa la normalizzazione: all'ottimo la costante appresa coincide
con il logaritmo della partizione, $\hat c = \log Z(\hat\theta)$, perché il
massimo della funzione obiettivo, preso su tutte le densità non normalizzate,
ha integrale uno da sé (Teorema 1 in {cite}`gutmann2012noise`). NCE normalizza
il modello senza mai sommare su tutto lo spazio, ed è l'unica delle tre vie
che restituisce una densità. La InfoNCE dei metodi contrastivi è la stessa
idea con molti campioni di rumore per ogni dato e una softmax al posto della
sigmoide, e la si ritrova nella sezione {doc}`Perché non collassa, e come si
fa a saperlo </AutoSupervisione/collasso-e-misura>`.

Il discriminatore delle GAN è cugino stretto di NCE: tutti e due imparano un
rapporto fra densità, non una densità. Il *negative sampling* di word2vec
{cite}`mikolov2013distributed` (il secondo dei due articoli word2vec: il primo
usava la softmax gerarchica) è invece una semplificazione dichiarata, che
la garanzia la butta via: tiene i campioni di rumore e getta le loro
probabilità, cioè proprio il termine che rendeva la stima un rapporto, e gli
autori scrivono che quella proprietà per il loro scopo non serve.

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
  - Non la calcola mai, la sostituisce: all'addestramento serve una media sulle
    risposte che il modello si immagina, non quel numero, e per produrle basta
    la pendenza sotto i piedi
  - Tempo. Le palline restano intrappolate da una parte quando le montagne
    sono alte, e in alta dimensione lo sono; e da dentro non si vede
* - **Score matching**, cioè imparare la pendenza (e la sua forma *denoising*,
    su dati sporcati apposta)
  - La elimina, perché sparisce nel passaggio dall'altezza alla pendenza
  - Nella forma originale un conto su come la pendenza stessa cambia da un
    punto al vicino (in gergo, le derivate seconde), caro
    in alta dimensione; nella forma *denoising*, il fatto che la pendenza
    imparata è quella dei dati sporcati di rumore, non dei dati
* - **NCE** (*noise-contrastive estimation*, la domanda sì o no) e parenti
  - La tratta come un parametro in più, da imparare insieme agli altri; e alla
    fine la ritrova giusta, senza averla mai calcolata
  - Dipende dal rumore che si sceglie: la stima resta corretta con
    qualunque rumore che copra i dati, ma se è troppo diverso distinguere
    diventa facile e servono moltissimi esempi per imparare poco
```

Tre modi di non pagare il conto, e nessuno dei tre gratis. Le tre strade
hanno però tutte lo stesso scopo, costruire un modello di com'è fatto il
paesaggio, e si distinguono solo per come pagano il conto.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Su un paesaggio disegnato senza vincoli, misurare l'intero continente è
  impossibile, prima ancora che caro. Cento neuroni, cioè
  cento interruttori accesi o spenti, danno un numero di configurazioni lungo
  trentuno cifre: a un miliardo di configurazioni al secondo servirebbero
  quasi tremila volte l'età dell'universo, e cento interruttori sono
  un'immagine in bianco e nero di dieci pixel per dieci.
- Imparare vuol dire abbassare il paesaggio dove stanno i dati veri e alzarlo
  dove il modello immagina male. Il primo gesto è facile, i dati ce li
  abbiamo; il secondo no, perché per sapere che cosa il modello immagina
  bisogna prima fargli produrre qualcosa.
- Prima via, campionare. La pallina su un tavolo che vibra scende ma ogni
  tanto risale, cambia valle e alla lunga passa più tempo in basso che in
  cima: le serve soltanto la pendenza sotto i piedi, mai la misura del
  continente. Nell'esempio a due valli ricostruisce le proporzioni giuste
  entro pochi millesimi; il prezzo è il tempo, e le montagne alte che la
  tengono prigioniera da una parte sola.
- Seconda via, la pendenza. Invece di dire quanta pioggia tocca a ogni
  punto, si dice da che parte si scende e quanto ripido: una descrizione tutta
  locale, che basta a ricostruire la forma del paesaggio. Insegnata su dati
  sporcati apposta, diventa il compito «indovina il rumore che ti ho aggiunto»,
  cioè quello che imparano i modelli di diffusione.
- Terza via, la domanda sì o no. Al posto di «quanto è probabile questo?»
  si chiede «viene dal mondo o l'ho fabbricato io?», e si addestra il modello
  a smistare i veri dai finti. Funziona, ma dipende dal rumore che gli si
  mette davanti: se è troppo diverso dai dati, il gioco diventa
  facile, e per imparare qualcosa servono moltissimi esempi.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Per un'energia senza vincoli sull'architettura, $Z$ è fuori portata prima
  ancora che cara. Con $N = 100$ variabili binarie gli stati
  sono $\approx 1{,}27 \times 10^{30}$, quasi tremila volte l'età
  dell'universo a un miliardo di stati al secondo.
- Il gradiente della log-verosimiglianza ha una fase positiva (abbassa
  l'energia sui dati) e una fase negativa (la rialza sui campioni del
  modello): è la seconda a richiedere di saper campionare da $p_\theta$.
- Langevin:
  $\mathbf{x}_{k+1} = \mathbf{x}_k - \frac{\epsilon}{2}\nabla_{\mathbf{x}} E_\theta(\mathbf{x}_k) + \sqrt{\epsilon}\, \mathbf{z}_k$.
  Usa solo $\nabla_{\mathbf{x}} E$, mai $Z$: nell'esempio a doppia buca
  ricostruisce la distribuzione esatta entro pochi millesimi, con un errore
  sistematico proporzionale a $\epsilon$. Il prezzo è il tempo: gli
  scavalcamenti di una barriera calano come $e^{-\Delta E}$ (Kramers).
- Score matching {cite}`hyvarinen2005estimation` confronta i gradienti
  invece delle densità; la forma denoising {cite}`vincent2011connection`
  la riduce a una regressione sul rumore ed è la loss dei modelli di
  diffusione.
- NCE {cite}`gutmann2010noise` trasforma la stima di densità in una
  classificazione dati contro rumore, con $\log Z$ come parametro che
  all'ottimo coincide con quello vero {cite}`gutmann2012noise`. Il
  *negative sampling* di word2vec è suo discendente.
```
`````

Resta una quarta possibilità, la più radicale, e cambia lo scopo invece del
metodo: non chiedere mai la probabilità. Se ciò che serve è decidere, ordinare,
pianificare, e non stampare percentuali, un modello della distribuzione non
serve affatto: l'energia basta da sola, e il conto non si apre nemmeno. È la
tesi della {doc}`cornice di LeCun </ModelliEnergia/energia-come-compatibilita>`,
che legge l'energia come un giudizio di compatibilità fra due cose.
