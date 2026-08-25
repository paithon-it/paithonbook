# Come funziona l'addestramento avversario

Dell'idea nata quella sera a Montréal si è già detto in apertura di capitolo; qui la smontiamo pezzo per pezzo.

Il modo consueto di insegnare a una rete a produrre un'immagine è dirle, punto
per punto, quanto la sua uscita si discosta da un'immagine vera che le mettiamo
accanto: questo puntino doveva essere più scuro, quest'altro più chiaro. Il
metodo funziona finché l'originale ce l'abbiamo sotto mano. Ma per *inventare*
un'immagine nuova l'originale non esiste, e senza originale non c'è confronto.
Le **Generative Adversarial Networks** (GAN) {cite}`goodfellow2014generative`
cambiano giudice: al posto del confronto punto per punto mettono una seconda
rete, il cui unico mestiere è smascherare la prima.

I due personaggi dell'apertura prendono qui il loro nome tecnico: il falsario
che dipinge quadri contraffatti è il **generatore** $G$, l'esperto d'arte che
deve smascherarlo è il **discriminatore** $D$. Da qui in avanti usiamo tutti e
due i nomi, e cominciamo da che cosa entra e che cosa esce da ciascuno.

## Il generatore: dal rumore al dato

Il generatore parte dal nulla, letteralmente da un pugno di numeri estratti a
caso, che d'ora in poi chiamiamo **rumore** (è il termine tecnico, e non ha
niente a che vedere con il suono: dice solo che quei numeri non significano
niente in partenza), e deve costruire un dato che sembri autentico.

`````{tab} Elementare

$G$ è un artigiano bendato: bendato perché i quadri autentici non
li vedrà mai, nemmeno uno, e fra poco vedremo che è una scelta di progetto e
non una dimenticanza. Gli consegni una manciata di numeri estratti a caso (il
rumore): è la sua materia prima, sempre diversa. Da quei numeri deve modellare
qualcosa di sensato, per esempio l'immagine di un volto. All'inizio produce
macchie informi. Con l'allenamento impara a trasformare quei numeri casuali in
volti sempre più plausibili. Numeri casuali diversi in ingresso danno volti
diversi in uscita: è così che $G$ genera *varietà*, non una sola immagine
ripetuta. Di volti però non tiene nessun registro: sa fabbricarne uno, non sa
dire quanto un volto sia probabile. E alla fine lo si giudica in blocco: i
volti che sforna, tutti insieme, devono somigliare al mucchio di quelli veri.

`````

`````{tab} Superiore

Il generatore è una funzione $G(\mathbf{z};\theta_G)$ parametrizzata da una rete neurale, che mappa un vettore di rumore $\mathbf{z}\in\mathbb{R}^k$ nello spazio dei dati:

$$
\mathbf{z} \sim p_z(\mathbf{z}) \quad\longmapsto\quad \tilde{\mathbf{x}} = G(\mathbf{z}) .
$$

Il vettore $\mathbf{z}$ è campionato da un *prior* semplice, tipicamente $p_z=\mathcal{N}(\mathbf{0}, \mathbf{I})$ o uniforme. $G$ definisce implicitamente una distribuzione $p_G$ sullo spazio dei dati: spingendo campioni di $\mathbf{z}$ attraverso la rete, otteniamo campioni di dati sintetici. L'obiettivo dell'addestramento è far convergere $p_G$ verso la distribuzione reale $p_{\text{dati}}$, senza mai scrivere esplicitamente la densità: da qui il nome di modello *generativo implicito*.

`````

## Il discriminatore: dal dato alla probabilità

Il discriminatore fa il mestiere opposto, e più familiare: è un classificatore binario.

`````{tab} Elementare

Davanti a $D$, l'esperto d'arte, passano dei quadri, tanti veri (pescati dal
**dataset**, il mucchio di esempi autentici che abbiamo raccolto) quanti falsi
(sfornati da $G$), e su ciascuno deve rispondere a una sola domanda: *è
autentico?* La sua risposta è un numero tra $0$ e $1$, una specie di livello
di fiducia: vicino a $1$ significa "sono quasi certo che sia reale", vicino a
$0$ significa "quasi certo che sia un falso". Il suo mestiere è non farsi
ingannare.

`````

`````{tab} Superiore

Il discriminatore è una funzione $D(\mathbf{x};\theta_D)\in[0,1]$ che stima la probabilità che $\mathbf{x}$ provenga dai dati reali anziché da $G$. La formalizzazione passa da una mistura: il campione arriva metà delle volte dal dataset e metà dal generatore, e $D$ stima la probabilità *a posteriori* che la sorgente sia quella reale, visto il campione:

$$
D(\mathbf{x}) \approx P(\text{reale} \mid \mathbf{x}) .
$$

È un classificatore addestrato con la consueta *cross-entropy* binaria: vuole assegnare $D(\mathbf{x})\to 1$ agli esempi reali e $D(G(\mathbf{z}))\to 0$ a quelli sintetici. L'uscita in $[0,1]$ si ottiene applicando una sigmoide al punteggio grezzo (il *logit*) dell'ultimo strato; nell'implementazione, come d'abitudine in PyTorch, la sigmoide sarà assorbita dentro la loss (`nn.BCEWithLogitsLoss`) per stabilità numerica.

`````

Messi uno di fronte all'altro, $G$ e $D$ compongono la GAN per intero
({numref}`fig-gan-architettura`).

```{figure} ../figures/gan-architettura.svg
:name: fig-gan-architettura
:alt: "Schema di una GAN: un vettore di rumore z entra nel generatore, che produce un dato falso; dati falsi e dati reali entrano nel discriminatore, che restituisce una probabilità reale/falso; in basso una freccia tratteggiata riporta i gradienti dell'errore dal discriminatore al generatore."
:width: 90%

Architettura di una GAN. Il generatore trasforma il rumore in un dato
sintetico; il discriminatore riceve dati reali e dati sintetici e stima, per
ciascuno, quanto è probabile che sia autentico. La freccia tratteggiata in
basso è la correzione che dal giudizio torna indietro verso il generatore.
```

Queste due riprendono il circuito già visto in apertura di capitolo, e la
differenza fra loro è la lingua:
{numref}`fig-gan-architettura` lo scrive con i simboli e chiama il segnale di
ritorno con il suo nome tecnico, i *gradienti* dell'errore (che cosa siano è
proprio l'argomento di queste pagine, e arriva fra poco);
{numref}`fig-gan-circuito` lo dice a parole. Chi preferisce incontrare i
simboli più tardi guardi la seconda (dove di simboli resta solo la $\mathbf{z}$
del rumore), ed è anche quella da tenere sott'occhio
per le pagine che seguono. Tutte e due disegnano soltanto la freccia che torna
al generatore, perché è quella che interessa qui: anche l'esperto impara dal
proprio errore, ma la sua parte del ritorno non è disegnata.

```{figure} ../figures/gan-2014.svg
:name: fig-gan-circuito
:alt: "Circuito di una GAN: il rumore casuale z entra nel generatore G, che produce un campione falso; il discriminatore D riceve sia campioni reali dal dataset sia il falso, e per ciascuno decide se è vero o finto; dal verdetto una freccia tratteggiata torna indietro fino al generatore, ed è il segnale con cui G impara. Fra i dati reali e il generatore non passa nessuna freccia."
:width: 96%

Il circuito completo. La freccia di ritorno verso il generatore è il punto:
G non vede mai i dati reali, impara soltanto da quanto bene ha ingannato D.
```

Vale la pena fermarsi su cosa {numref}`fig-gan-circuito` *non* collega. Fra i
dati reali e il generatore non passa nessuna freccia: G non li copia né li
confronta, e tutto ciò che sa del mondo gli arriva filtrato dal giudizio del
discriminatore. È una scelta di progetto elegante e, come vedremo, fragile.

Elegante per una ragione che conviene dire ad alta voce, perché è la
mossa più furba del disegno: se il falsario vedesse i quadri autentici, la
strategia migliore per ingannare l'esperto sarebbe **ricopiarne uno**, e
avremmo costruito una macchina che restituisce ciò che le abbiamo dato. Tenerlo
bendato è ciò che lo costringe a inventare.

E allora da dove entra la verità, se metà del sistema il mondo non lo vede
mai? Entra dall'altra metà. L'esperto, a ogni turno, guarda anche dei quadri
autentici presi dal dataset e viene corretto su quelli: è lui l'ancora, ed è
l'unica cosa che impedisce ai due di mettersi d'accordo su una schifezza.
Immaginiamo per un attimo di togliergliela, cioè di allenare l'esperto solo
sui falsi. Il caso limite è che gli convenga rispondere "falso" a tutto, e
allora il duello gira a vuoto perché il falsario non ha nessuna strada per
accontentarlo. Ma il caso interessante è l'altro: se anche l'esperto si
inventasse un criterio qualunque per promuovere qualcosa (diciamo che gli
piacciono i quadri verdi), il falsario imparerebbe a dipingere verde e i due
sarebbero contenti tutti e due, con dei quadri che non somigliano a niente.
Sono nemici, ma nessuno dei due ha più un motivo per cambiare, ed è questo il
senso di «mettersi d'accordo». Il senso della
freccia che manca è tutto qui: la realtà entra nel sistema una volta sola, dal
lato di $D$, e da lì raggiunge $G$ di rimbalzo.

## Il gioco minimax

Le due reti non inseguono due obiettivi scollegati: condividono un’**unica
funzione di valore**, cioè un punteggio solo per tutta la partita, che uno
vuole tirare più in alto possibile e l'altro più in basso possibile.

*Minimax* è la contrazione di *minimo* e *massimo*, e dice come si ragiona
quando l'avversario è bravo. Chi gioca da solo sceglie la mossa che gli
promette il guadagno più alto; chi gioca contro qualcuno non può, perché quel
guadagno l'altro glielo toglierà. Deve allora guardare, per ogni mossa
possibile, il peggio che l'avversario può fargli, e scegliere la mossa il cui
peggio è meno peggio di tutti gli altri: il minimo dei massimi danni. È il
modo di ragionare che dà il nome al gioco. Nel codice non lo si vedrà scritto
da nessuna parte, perché i due si limitano a correggersi un pezzetto per volta:
il minimax dice dove la partita vorrebbe andare a finire, non come i due ci
arrivano.

`````{tab} Elementare

I punti si segnano tutti sullo stesso tabellone. L'esperto ne guadagna ogni
volta che indovina, sui quadri autentici come sui falsi; il falsario guadagna
ogni volta che gliene fa perdere. Quello che è un bene per uno è un male per l'altro: è un gioco a
somma zero.

Metà tabellone, però, il falsario non lo tocca. Sui quadri autentici l'esperto
se la vede da solo, e l'unica cosa in suo potere è come vengono i propri:
gioca sul suo mezzo tabellone e basta.

Il gioco cerca un *equilibrio*: il punto in cui nessuno dei due riesce più a
migliorare a spese dell'altro. Lì il falsario è così bravo che
l'esperto, per quanto si sforzi, può solo tirare a indovinare, cinquanta e
cinquanta su ogni quadro. Vale per un esperto ideale, uno a cui è concesso
qualunque criterio e una pazienza infinita; quello vero è una rete con un
numero finito di pesi, e all'ideale ci somiglia soltanto fin dove ci arriva.

`````

`````{tab} Superiore

$G$ e $D$ giocano un gioco minimax sulla funzione di valore

$$
\min_{G}\ \max_{D}\ V(D,G) =
\mathbb{E}_{\mathbf{x}\sim p_{\text{dati}}}\big[\log D(\mathbf{x})\big]
+ \mathbb{E}_{\mathbf{z}\sim p_z}\big[\log\big(1 - D(G(\mathbf{z}))\big)\big] .
$$

Qui $p_{\text{dati}}$ è la distribuzione dei dati reali, $p_z$ il prior del rumore, $D(\mathbf{x})$ la probabilità stimata di autenticità e $G(\mathbf{z})$ il campione generato. $D$ massimizza $V$ (vuole $D(\mathbf{x})$ grande sui reali e $1-D(G(\mathbf{z}))$ grande sui falsi); $G$ minimizza il secondo termine, l'unico che dipenda da lui (vuole $D(G(\mathbf{z}))\to 1$).

La dimostrazione di Goodfellow sta in due passaggi, e conviene rifarli per intero: il secondo è quello che dice *che cosa* una GAN stia davvero minimizzando, ed è un risultato che si cita spesso e si deriva di rado. Prima però le ipotesi, che non sono innocue: le due reti hanno **capacità illimitata**, cioè $D$ è una funzione qualsiasi a valori in $[0,1]$ e non una rete con un numero finito di pesi, e le distribuzioni in gioco hanno densità rispetto alla stessa misura. Su
quest'ultima conviene tenere un dito: fra poco si vedrà che nel caso vero è
proprio lei a cadere.

**Primo passaggio: il discriminatore ottimo.** Fissato $G$, il secondo integrale si riscrive nello spazio dei dati invece che in quello del rumore, perché spingere $\mathbf{z}$ attraverso $G$ è esattamente ciò che definisce $p_G$:

$$
\begin{aligned}
V(D,G) &= \int p_{\text{dati}}(\mathbf{x})\log D(\mathbf{x})\,d\mathbf{x}
        + \int p_z(\mathbf{z})\log\big(1-D(G(\mathbf{z}))\big)\,d\mathbf{z} \\
       &= \int \Big[\, p_{\text{dati}}(\mathbf{x})\log D(\mathbf{x})
        + p_G(\mathbf{x})\log\big(1-D(\mathbf{x})\big) \Big]\,d\mathbf{x}.
\end{aligned}
$$

Ed è qui che serve l'ipotesi di capacità illimitata: poiché $D$ non ha vincoli, l'integrale si massimizza massimizzando l'integrando **punto per punto**, cioè scegliendo per ogni $\mathbf{x}$ separatamente il numero $u = D(\mathbf{x}) \in [0,1]$ che rende massima $a\log u + b\log(1-u)$, con $a = p_{\text{dati}}(\mathbf{x})$ e $b = p_G(\mathbf{x})$. Derivando in $u$:

$$
\frac{d}{du}\big[a\log u + b\log(1-u)\big] = \frac{a}{u} - \frac{b}{1-u} = 0
\;\Longleftrightarrow\; a(1-u) = b\,u \;\Longleftrightarrow\; u = \frac{a}{a+b},
$$

e la derivata seconda $-a/u^2 - b/(1-u)^2$ è negativa, quindi quel punto è un massimo e non un minimo. Da cui

$$
D^*(\mathbf{x}) = \frac{p_{\text{dati}}(\mathbf{x})}{p_{\text{dati}}(\mathbf{x}) + p_G(\mathbf{x})},
$$

cioè proprio l'ottimo bayesiano della mistura descritta sopra. Il conto vale dove $a+b>0$: fuori dall'unione dei due supporti l'integrando è nullo e $D^*$ non è definito, ed è la ragione per cui tutti gli enunciati che seguono dicono «sul supporto dei dati» e non «ovunque».

**Secondo passaggio: che cosa resta da minimizzare.** Si sostituisce $D^*$ in $V$ e si chiama $C(G) = V(D^*,G)$ ciò che rimane, funzione del solo generatore:

$$
C(G) = \mathbb{E}_{\mathbf{x}\sim p_{\text{dati}}}\!\Big[\log \frac{p_{\text{dati}}}{p_{\text{dati}}+p_G}\Big]
     + \mathbb{E}_{\mathbf{x}\sim p_G}\!\Big[\log \frac{p_G}{p_{\text{dati}}+p_G}\Big].
$$

Il passaggio chiave è far comparire la **mistura** $m = (p_{\text{dati}}+p_G)/2$, che si ottiene dividendo per $2$ sopra e sotto dentro ciascun logaritmo: $\frac{p}{p_{\text{dati}}+p_G} = \frac{1}{2}\cdot\frac{p}{m}$, e il fattore $\tfrac12$ esce da ognuno dei due termini come un $-\log 2$. Restano due divergenze di Kullback-Leibler:

$$
C(G) = -\log 4 + \mathrm{KL}\big(p_{\text{dati}} \,\|\, m\big) + \mathrm{KL}\big(p_G \,\|\, m\big)
     = -\log 4 + 2\,\mathrm{JSD}\big(p_{\text{dati}} \,\|\, p_G\big),
$$

dove l'ultima uguaglianza è la definizione stessa della divergenza di Jensen-Shannon, $\mathrm{JSD}(p\,\|\,q) = \tfrac12\mathrm{KL}(p\,\|\,m) + \tfrac12\mathrm{KL}(q\,\|\,m)$. La $\mathrm{JSD}$ è non negativa e si annulla se e solo se le due distribuzioni coincidono, quindi $C(G)$ ha minimo globale $-\log 4 \approx -1{,}386$ esattamente in $p_G = p_{\text{dati}}$; e lì $D^*(\mathbf{x})=\tfrac{1}{2}$ sul supporto dei dati, cioè l'esperto non sa più distinguere.

Questa però è una *caratterizzazione* dell'ottimo, non una promessa di arrivarci, e le due cose vanno tenute separate. La prova di convergenza del paper suppone che a ogni passo $D$ raggiunga il proprio ottimo dato $G$, e soprattutto che a muoversi sia la densità $p_G$, dove $V$ è convessa; nell'addestramento vero si muovono i parametri $\theta_G$ di una rete, e lì la convessità che regge la dimostrazione non c'è più. Lo scrivono gli autori stessi, subito dopo la dimostrazione: usare un percettrone multistrato per definire $G$ introduce molti punti critici nello spazio dei parametri, e le reti funzionano bene in pratica «despite their lack of theoretical guarantees».

`````

Di questo gioco non esiste un fotogramma che lo racconti: quello che conta è
il **movimento**, il falso che si avvicina al vero e l'esperto che perde terreno
mentre succede. {numref}`fig-gan-inseguimento` lo mette in scena su un caso
minuscolo, in sette tappe che si susseguono una dopo l'altra.

Il caso è minuscolo perché al posto delle immagini, che di puntini ne hanno
milioni, ogni esempio è un numero solo: pensa all'altezza di una persona, o
alla temperatura di un giorno. Così i dati si possono
disegnare: si segna su una riga dove cade ciascun esempio e si guarda dove si
ammucchiano. Ne viene una curva a **campana**, alta dove gli esempi sono fitti
e bassa dove sono radi, che è la forma che prende quasi sempre un mucchio di
misure: tanti valori vicini al centro, pochi agli estremi. Una campana per i
dati veri, che sta ferma, e una per quelli del falsario, che si muove.

```{figure} ../figures/gan-inseguimento.svg
:name: fig-gan-inseguimento
:alt: "Due pannelli sovrapposti. Sopra, la campana dei dati veri sta ferma al centro mentre quella del generatore, all'inizio spostata a sinistra e più larga, si sposta e si stringe fino a coprirla. Sotto, la curva del verdetto parte a gobba, alta dove prevalgono i dati veri e bassa dove prevale il generatore, e si appiattisce fino a diventare la retta orizzontale a un mezzo; è disegnata solo nel tratto in cui almeno una delle due campane ha densità apprezzabile."
:width: 92%

Sopra: i dati veri stanno fermi, il generatore li insegue. Sotto: il verdetto
migliore che l'esperto possa dare contro quel generatore. Finché le due
campane sono separate il verdetto è netto; quando si sovrappongono diventa un
mezzo dappertutto, cioè una moneta lanciata in aria.
```

Il riquadro di sotto non è disegnato a mano: è il verdetto migliore possibile
contro quel generatore, e si calcola dalle due curve di sopra con una regola
sola. In ogni punto si prende l'altezza della curva vera e la si divide per la
somma delle due altezze: se lì cade solo roba vera il conto dà uno, se cade
solo roba falsa dà zero, se le due curve sono alte uguali dà un mezzo. Da
questa regola discende una cosa da guardare da vicino: quel che conta in ogni
punto è **quale delle due curve prevale, e di quanto**, non quanto sono
distanti fra loro i due picchi. Dove le due curve sono alte uguali il verdetto
sta a un mezzo, anche se lì di esempi ne cadono pochissimi; dove una prevale
sull'altra il verdetto si allontana da un mezzo, anche se le due curve sono
quasi sovrapposte.

`````{tab} Elementare

Nell'animazione due cose si muovono insieme, e conviene guardarle una per
volta.

La prima è il **confine**: il punto in cui il verdetto passa esattamente per il
mezzo, cioè dove l'esperto smette di dire "falso" e comincia a dire "vero".
All'inizio sta a sinistra, perché a sinistra il falsario è di casa; poi scivola
verso destra mentre il falsario avanza.

La seconda è l’**altezza della gobba**, cioè quanto l'esperto è sicuro nel suo
terreno migliore. Nella prima tappa la gobba arriva a $0{,}93$, che è quasi
certezza; nell'ultima è scesa a $0{,}50$, che è nessuna certezza. Il confine si
sposta e intanto la gobba si sgonfia, e quando la gobba tocca il mezzo il
confine non c'è più: non è che l'esperto abbia sbagliato posto, è che non c'è
più un posto giusto.

C'è infine una ragione per cui la curva di sotto **non attraversa tutto il
riquadro**. Verso i bordi non cade quasi nessun esempio, né vero né falso, e
non c'è niente da giudicare. Il guaio è che la regola di prima una risposta la
darebbe lo stesso, e sarebbe bassa (là fuori la curva del falsario, che è più
larga, resta sopra a quella dei dati veri): l'esperto direbbe «certamente
falso» proprio dove non c'è niente.

`````

`````{tab} Superiore

Il pannello inferiore è $D^*$ del passaggio precedente, valutato sulle due
gaussiane del pannello superiore: i dati veri sono
$\mathcal{N}(0,\ 0{,}55^2)$ e stanno fermi, il generatore parte da
$\mathcal{N}(-1{,}75,\ 1{,}05^2)$ e raggiunge i dati in sette tappe, con media e
deviazione standard interpolate linearmente. Due grandezze si muovono insieme.

Il **punto di indifferenza**, dove $D^*=\tfrac12$ e quindi
$p_{\text{dati}}=p_G$: vale $-0{,}80$, $-0{,}72$, $-0{,}63$, $-0{,}53$,
$-0{,}42$, $-0{,}29$ nelle prime sei tappe (il picco dei dati veri sta
nell'origine), e alla settima non esiste più, perché le due densità coincidono
ovunque. Due gaussiane di varianza diversa si incrociano in realtà in due
punti, e il secondo cade attorno a $x \simeq 2{,}1$: là però entrambe le
densità sono dell'ordine di $10^{-4}$, cioè fuori dal tratto disegnato.

Il **massimo di $D^*$**, cioè la fiducia dell'esperto nel suo terreno migliore:
$0{,}93$, $0{,}90$, $0{,}87$, $0{,}82$, $0{,}74$, $0{,}64$, $0{,}50$. Il
confine si sposta e il contrasto si appiattisce insieme a lui; all'ultima tappa
$D^*$ è la costante $\tfrac12$, e di un confine non c'è più traccia.

La curva **non attraversa tutto il riquadro** per la stessa ragione per cui la
dimostrazione conclude $D^*(\mathbf{x}) = \tfrac{1}{2}$ **sul supporto dei dati**: il
rapporto fra due densità è definito ovunque, ma dove entrambe sono trascurabili
non c'è niente da giudicare, e disegnarlo lì direbbe al lettore «certamente
falso» in una regione vuota. La figura taglia dove la densità totale scende
sotto $0{,}02$.

`````

## L'addestramento alternato

I due obiettivi tirano in direzioni opposte, e non esiste una mossa che li
accontenti tutti e due: quello che fa scendere l'errore di uno lo fa salire
all'altro. Si procede allora **a turni**: un turno per $D$, un turno per $G$, e
così via, con la discesa del gradiente stocastica già incontrata nei capitoli
precedenti (si guarda da che parte l'errore cala e ci si sposta di un passetto
in quella direzione; *stocastica* vuol dire che a ogni passetto si guarda un
pugno di esempi presi a caso, non tutti insieme). Mentre si aggiorna una rete,
i **pesi** dell'altra restano fermi: i pesi sono i numeri che l'addestramento
aggiusta dentro una rete, e tenerli fermi è il «congelamento» di cui fra poco
vedremo che cosa lo garantisce.

Nel codice, il punteggio unico del gioco si spezza in due conti dell'errore,
uno per rete: sono le due **loss** del ciclo, `loss_D` e
`loss_G`. Non sono due giochi diversi: sono le due facce dello stesso
punteggio, ciascuna scritta dal punto di vista di chi la deve far scendere; e
d'ora in avanti, quando parleremo di "loss", parleremo di queste. (Con una
sorpresa in agguato: fra poco vedremo che una delle due righe, nel codice vero,
è scritta in un modo che quel punteggio unico lo incrina. Per adesso teniamolo.)

Il ciclo completo sta in una ventina di righe, e le tre cose che contano sono
spiegate subito sotto: per seguirle bastano le poche paroline del codice che
ricorrono nella spiegazione (`n`, `opt_G`, `opt_D`, `.detach()`).

```{code-block} python
:class: pt-non-eseguibile

import torch
from torch import nn

# G e D sono due nn.Module, ciascuno con il proprio allenatore
# (opt_G e opt_D): aggiornare l'uno non tocca i pesi dell'altro
criterio = nn.BCEWithLogitsLoss()        # sigmoide inclusa nella loss

for epoca in range(n_epoche):
    for batch_reale in loader:
        n = batch_reale.size(0)          # quanti esempi ci sono in questo gruppo
        uni  = torch.ones(n, 1)          # etichette "reale"
        zeri = torch.zeros(n, 1)         # etichette "falso"

        # 1) Passo del discriminatore: distinguere reale da falso
        z = torch.randn(n, dim_rumore)   # rumore
        falsi = G(z).detach()            # campioni sintetici, staccati da G
        loss_D = (criterio(D(batch_reale), uni)   # spinge D(x) -> 1
                  + criterio(D(falsi), zeri))     # spinge D(G(z)) -> 0
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # 2) Passo del generatore: ingannare D (si aggiorna solo G)
        z = torch.randn(n, dim_rumore)
        loss_G = criterio(D(G(z)), uni)  # vuole D(G(z)) -> 1
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()
```

Tre punti di questo ciclo meritano di essere guardati da vicino: che cosa
esattamente torna indietro dall'esperto al falsario, come mai i due
allenamenti non si mescolano, e una piccola astuzia sulla lezione impartita al
generatore. Il primo è il meccanismo centrale del capitolo, e conviene
partire da lì.

### Che cosa torna indietro

`````{tab} Elementare

Riprendiamo la domanda lasciata in sospeso: quando l'esperto boccia un quadro,
che cosa impara il falsario? Se ciò che torna indietro fosse il verdetto
("falso"), non imparerebbe niente di utile: saprebbe di aver sbagliato, e
basta. Ma il ritorno non è il verdetto.

Per una macchina un quadro *è* un elenco di numeri, uno per puntino, che dice
quanto quel puntino è chiaro o scuro. Dipingere vuol dire scegliere quei
numeri.

L'esperto, allora, è fatto in modo che gli si possa chiedere qualcosa di più
fine di un giudizio, e cioè, per ogni singolo puntino del quadro: *se questo
puntino fosse un po’ più chiaro, il tuo giudizio salirebbe o scenderebbe, e di
quanto?* La risposta a quella domanda, posta per tutti i puntini insieme, è
lunga quanto il quadro: per ciascun puntino, da che parte spostarlo e con
quanta forza. È questo che torna indietro. Non un voto, ma una correzione con
un verso, punto per punto: l'esperto non dice "falso", dice "falso, e
soprattutto per via di *questo* qui".

Attenzione però al verso, perché è il punto in cui l'inganno si capovolge.
L'esperto risponde a quella domanda per i propri scopi: quello che lui indica è
come cambierebbe il *suo* giudizio, e a lui il giudizio serve per smascherare.
Il falsario prende la sua risposta e **la percorre al contrario**. Dove
l'esperto dice «se questo puntino fosse più chiaro mi insospettirei di più», il
falsario lo scurisce. Non riceve un consiglio dal nemico: riceve una mappa del
nemico, e la usa contro di lui.

Quell'elenco ha un nome, ed è la parola che si legge nelle figure e in ogni
manuale: si chiama **gradiente**. «I gradienti tornano indietro dal
discriminatore al generatore» vuol dire esattamente questo: l'elenco delle
spintarelle, una per puntino, da percorrere al rovescio.

Ed è anche la risposta alla domanda gemella: perché l'esperto dev'essere una
rete, e non una persona o un elenco di regole? Una persona darebbe lo stesso
verdetto, e magari un consiglio a parole ("la firma non convince"); quello che
non può dare è la lista. A un critico d'arte non si può chiedere di quanto
spostare ciascuno dei due milioni di puntini di una fotografia. A una rete sì,
perché una rete è una formula, e a una formula si può sempre domandare come
cambia il risultato se si muove un ingresso.

Resta un passaggio, ed è quello in cui il falsario impara davvero. L'elenco che
gli arriva parla del *quadro*: dice come dovrebbe venire il prossimo. A lui
serve invece sapere come cambiare *sé stesso*, cioè come ritoccare i propri
pesi. Ma quel pezzo
lo sa già per conto proprio, senza chiedere niente all'esperto: anche lui è una
formula, e sa di quanto si muove ciascun puntino del quadro se ritocca un certo
peso.

I due elenchi si compongono **moltiplicando**, ed è più facile con dei numeri
inventati. Diciamo che schiarire di un'unità quel puntino faccia salire di $2$
il giudizio dell'esperto, e che girare di un'unità un certo peso del falsario
schiarisca quel puntino di $3$: allora girare quel peso di un'unità fa salire
il giudizio di $2 \times 3 = 6$. Il falsario ha ottenuto quello che gli
serviva, «di quanto conviene girare questo peso», ed è un conto in cui l'esperto
ha messo il primo fattore e lui il secondo.

C'è una condizione nascosta, gemella di quella sull'esperto: il falsario deve
dipingere, non scegliere. Il colore si stende un filo di più o un filo di meno,
e il suo mezzo conto ha senso; con parole prese da un elenco, invece, un
ritocco minuscolo a un peso o cambia la parola o non cambia niente, e quel
secondo fattore non esiste più. Per questo le GAN sul testo sono sempre state
faticose.

`````

`````{tab} Superiore

La quantità che il passo di $G$ propaga si scrive con la regola della catena,
spezzata nel punto in cui le due reti si toccano:

$$
\frac{\partial \mathcal{L}_G}{\partial \theta_G} =
\frac{\partial \mathcal{L}_G}{\partial \tilde{\mathbf{x}}} \cdot
\frac{\partial \tilde{\mathbf{x}}}{\partial \theta_G},
\qquad \tilde{\mathbf{x}} = G(\mathbf{z}) .
$$

Il primo fattore è il gradiente della loss del generatore rispetto al **dato
generato**, e vive nello spazio dei dati: ha una componente per ogni numero di
$\tilde{\mathbf{x}}$ (per un'immagine, una per pixel e per canale). È lì che sta la
differenza fra un'informazione utile e un'informazione inutile: il verdetto
$D(\tilde{\mathbf{x}})$ è uno scalare, mentre $\partial \mathcal{L}_G / \partial
\tilde{\mathbf{x}}$ è un vettore che indica, componente per componente, in che verso
spostare il dato perché il verdetto salga. Il secondo fattore è la jacobiana
del generatore rispetto ai propri parametri, e con $D$ non ha niente a che
vedere: è la parte che il falsario conosce già di sé.

Da qui discende un **requisito di progetto**, non un dettaglio di
implementazione: $D$ dev'essere derivabile rispetto al proprio ingresso. Un
giudice umano, o un programma a regole, darebbe lo stesso verdetto e nessun
vettore; la catena si spezzerebbe nel primo fattore e a $G$ non arriverebbe
niente. È anche la ragione per cui le GAN sui dati discreti (il testo, prima di
tutto) sono sempre state faticose: se $\tilde{\mathbf{x}}$ è una sequenza di simboli
campionati, il secondo fattore non esiste, e per aggirare la rottura servono
stimatori a punteggio in stile REINFORCE o rilassamenti continui come
Gumbel-softmax.

Un esempio minimo dà la misura della differenza fra le due informazioni, ed è
un conto che si rifà in cinque righe. Il $D$ giocattolo è
`nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 1))` inizializzato
dopo `torch.manual_seed(8)`, il dato generato è
$\tilde{\mathbf{x}} = [\,0{,}30,\ -0{,}70,\ 1{,}20,\ 0{,}10\,]$ e la loss è la
`binary_cross_entropy_with_logits` verso l'etichetta "reale". Il verdetto è
$D(\tilde{\mathbf{x}}) = 0{,}409823$: un solo numero, che dice "propendo per il falso" e
nient'altro. Il gradiente sullo stesso dato è invece

$$
\frac{\partial \mathcal{L}_G}{\partial \tilde{\mathbf{x}}} =
[\,-0{,}03570,\ -0{,}08145,\ -0{,}07891,\ +0{,}15839\,],
$$

quattro numeri che dicono di alzare le prime tre componenti e di abbassare
nettamente la quarta. Muovendo il dato di mezzo passo in senso opposto al
gradiente, cioè $\tilde{\mathbf{x}} \leftarrow \tilde{\mathbf{x}} - 0{,}5\,\partial
\mathcal{L}_G / \partial \tilde{\mathbf{x}}$, il verdetto sale a $0{,}417891$: poco,
perché il passo è piccolo, ma nella direzione voluta, e ottenuto senza che
nessuno abbia mai mostrato al generatore un dato autentico.

`````

Tenere a mente che il ritorno è un elenco di spintarelle, e non un voto,
serve per tutto il resto della pagina. Quando fra poco parleremo di duelli che
si inceppano, i guasti saranno guasti di quell'elenco: a volte le spintarelle
si assottigliano fino a sparire (è ciò che in gergo si chiama «gradienti che
svaniscono») e il falsario non sa più da che parte andare; altre volte
diventano enormi e tutte diverse fra loro, e lo fanno barcollare invece di
guidarlo. Anche il rimedio più fortunato fra quelli raccolti in fondo a questa
pagina, il *gradient penalty*, riguarda l'elenco: è una multa all'esperto quando le
sue spintarelle si allontanano da una taglia fissa, in su o in giù. Mai il
verdetto.

### Gli altri due dettagli

`````{tab} Elementare

Prima dei due dettagli, la riga con la `n`. I dati non si danno
in pasto alla rete uno per volta ma a gruppetti, e l'ultimo gruppo di ogni giro
può risultare più corto degli altri (se gli esempi sono $1000$ e i gruppi da
$64$, l'ultimo ne contiene $40$). La `n` conta quanti esempi ci sono davvero nel
gruppo di turno, e serve a preparare esattamente altrettante etichette "vero" e
"falso".

Primo dettaglio. Nel codice, il lavoro di girare i pesi non lo fa la rete: lo
fa un pezzo di programma attaccato a lei, che si chiama **allenatore** (`opt_G`
per il falsario, `opt_D` per l'esperto). Non sono due personaggi nuovi della
storia, sono la mano del falsario e la mano dell'esperto: prendono le
correzioni calcolate e le applicano. Il punto è che sono due, e che ciascuno
conosce soltanto i pesi della propria rete: è questo, e nient'altro, a
garantire che ciascuno dei due impari solo nel proprio turno. È il
«congelamento» dei pesi, ed è il modo in cui i due allenatori sono stati messi
su fin dall'inizio.

Chi programma in PyTorch si aspetterebbe qui la parola `.detach()`, che
compare nel codice quando si allena l'esperto sui falsi, e a cui quel merito
viene spesso attribuito. Non è suo: dice soltanto di non calcolare
nemmeno la correzione per il falsario, dato che in quel turno verrebbe comunque
buttata via. Non serve a tenere separati i due allenamenti (a quello bastano i
due allenatori), serve a non sprecare lavoro, e su reti grandi il risparmio è
notevole. Regge finché le righe del ciclo stanno in quest'ordine: cambiandolo,
quel lavoro sprecato smetterebbe di essere innocuo e `.detach()` tornerebbe
indispensabile.

Secondo dettaglio: nel suo turno, il falsario misura il proprio errore come
se i suoi falsi *dovessero* risultare autentici, e impara da quanto il verdetto
se ne discosta. Non sta corrompendo l'arbitro, che infatti non se ne accorge:
sta scegliendo con che metro misurare sé stesso.
Detta così sembra una sfumatura, e invece cambia la domanda che si fa
all'esperto. Non più «quanto è falso questo quadro?», ma «quanto manca perché
passi per vero?».

Le due domande si comportano in modo diverso proprio dove serve. La correzione,
lo abbiamo appena visto, non è il voto: è **di quanto il voto cambierebbe**. Se
l'esperto è sicurissimo che il quadro sia falso, il suo giudizio è schiacciato
contro il fondo della scala e non può scendere oltre: un ritocco al quadro non
lo sposta di una virgola, e alla prima domanda la risposta è sempre la stessa,
«del tutto». Un principiante corretto così è come uno studente che prende zero
a ogni compito senza mai sapere quale zero fosse meno grave: non ha modo di
capire se l'ultimo ritocco andava nella direzione giusta.

La seconda domanda un fondo non ce l'ha, e si vede con due numeri. Supponiamo
che l'esperto dia al quadro una probabilità di essere autentico di $1$ su $100$,
e che un ritocco la porti a $2$ su $100$. Per la prima domanda non è successo
quasi niente: da «falso al $99$ per cento» a «falso al $98$ per cento», un
centesimo di scarto. Per la seconda il quadro ha appena **raddoppiato** le
proprie probabilità, ed è un passo avanti enorme. Stesso ritocco, stesso
esperto: cambia solo quale delle due domande gli si fa, e la seconda continua a
distinguere anche laggiù in fondo, dove la prima ha smesso.

A rigore non è più lo stesso gioco. Il falsario non sta più cercando di far
scendere il punteggio che l'esperto fa salire: ne insegue uno suo, e il
tabellone unico non basta più a raccontare tutti e due i giocatori. Il trucco è
già suggerito nell'articolo del 2014; il prezzo lo vedremo fra poco.

`````

`````{tab} Superiore

La separazione fra i due allenamenti la garantiscono i **due ottimizzatori**:
`opt_D` non conosce i parametri di $G$ e viceversa, quindi nessuno dei due passi
può toccare i pesi dell'altra rete. Il `.detach()` nel passo di $D$ aggiunge un
risparmio: stacca i campioni sintetici dal grafo di $G$, così il gradiente
attraverso il generatore non viene nemmeno calcolato. In questo ciclo, senza
`.detach()`, quel gradiente verrebbe calcolato, si depositerebbe in `.grad` e
sarebbe poi azzerato da `opt_G.zero_grad()` prima di essere usato: il risultato numerico è identico, pesi finali di $G$ compresi, ma su una rete
grande si paga un passaggio all'indietro intero per niente. Attenzione
però che l'innocuità dipende dall'ordine delle righe: in una variante che
azzeri i gradienti in cima all'iterazione, o che legga `.grad` fra i due passi,
`.detach()` torna necessario.

C'è poi una scelta nascosta nella riga `criterio(D(G(z)), uni)`, ed è la
formulazione che si usa davvero, qui come in qualunque implementazione:
chiedere che i falsi siano etichettati "reale" equivale a **massimizzare**
$\log D(G(\mathbf{z}))$, invece di minimizzare $\log(1-D(G(\mathbf{z})))$ come nella formula
minimax. È la *non-saturating loss*, già suggerita nel paper del 2014, e il
motivo per cui la si preferisce sta tutto in una derivata.

Sia $s$ il logit che $D$ produce sul campione falso, cosicché
$D(G(\mathbf{z})) = \sigma(s)$. Scritte entrambe come qualcosa da **minimizzare**, le
due perdite del generatore sono $\mathcal{L}^{\text{sat}} = \log\big(1-\sigma(s)\big)$
e $\mathcal{L}^{\text{ns}} = -\log \sigma(s)$; ricordando che
$\sigma' = \sigma(1-\sigma)$, i due fattori $\sigma$ si semplificano in modi
opposti e restano

$$
\frac{\partial \mathcal{L}^{\text{sat}}}{\partial s} = -\,\sigma(s),
\qquad
\frac{\partial \mathcal{L}^{\text{ns}}}{\partial s} = -\big(1-\sigma(s)\big).
$$

Le due spingono nello stesso verso (verso $s$ grande, cioè $D(G(\mathbf{z}))\to 1$),
ma con forze che agli estremi si scambiano. Quando $G$ è pessimo e $D$ lo
smaschera, diciamo $D(G(\mathbf{z})) = 0{,}01$, la prima ha modulo $0{,}01$ e la
seconda $0{,}99$: **novantanove volte più grande**. In generale il rapporto fra
le due vale $(1-\sigma)/\sigma = e^{-s}$ e cresce senza limite man mano che $D$
si convince, mentre all'equilibrio $\sigma=\tfrac12$ le due coincidono. La loss
minimax non è debole in generale, quindi: è debole **proprio dove servirebbe di
più**, all'inizio dell'addestramento, ed è il senso della frase con cui il
paper la liquida, «same fixed point» ma gradienti «much stronger early in
learning».

Non sono però lo stesso gioco, ed è meglio dirlo esplicitamente perché il
capitolo ha appena costruito due sezioni sull'idea di un punteggio unico: con
questa formulazione il gioco **non è più a somma zero** e non si lascia più
scrivere con un'unica funzione di valore, come nota Goodfellow stesso nel
proprio tutorial NIPS {cite}`goodfellow2016nips`. Arjovsky e Bottou
{cite}`arjovsky2017towards` mostrano
che il gradiente ricevuto qui dal generatore è quello di
$\mathrm{KL}\big(p_G \,\|\, p_{\text{dati}}\big) - 2\,\mathrm{JSD}\big(p_G \,\|\,
p_{\text{dati}}\big)$: una divergenza di Kullback-Leibler rovesciata, più un
termine che spinge le due distribuzioni ad allontanarsi. Conviene tenerlo a
mente fra poco, quando parleremo di *mode collapse*.

`````

## Quando il duello si inceppa

Sulla carta il meccanismo è pulito: due reti, un punteggio, un equilibrio verso
cui tendere. Nella pratica le GAN si sono guadagnate la fama di essere fra le
reti più difficili da addestrare, e tre problemi ricorrono.

`````{tab} Elementare

- **Instabilità.** I due giocatori si rincorrono senza mai fermarsi: migliora
  uno, l'altro peggiora, e il punteggio oscilla invece di stabilizzarsi.
  All'inizio del capitolo avevamo detto che i due "si perfezionano a vicenda",
  ed è ancora vero: la differenza sta in quanto è grossa la correzione che
  ciascuno si applica a ogni turno, e quanto grossa lo fissa chi addestra prima
  di cominciare. Con ritocchi piccoli ogni miglioramento resta acquisito e
  l'equilibrio si sposta un poco alla volta; con correzioni troppo grosse
  ognuna disfa la precedente e nessuno dei due consolida niente, due lottatori
  che si sbilanciano a vicenda invece di allenarsi. Ed è qui che si paga il prezzo
  annunciato poco fa: la domanda «quanto manca perché il quadro passi per
  vero?» tiene viva la correzione anche quando il falsario è pessimo, ma quando
  l'esperto è molto più bravo di lui risponde ogni volta «moltissimo», e
  correzioni tutte grandi e tutte diverse fra loro lo fanno oscillare invece di
  guidarlo.

  E c'è un guasto che col dosaggio non c'entra. Il falsario parte da una
  manciata di numeri, e un quadro di puntini ne ha milioni: tutto quello che sa
  produrre sta su una superficie sottilissima dentro il mondo dei quadri
  possibili, e i quadri veri stanno fuori di lì. All'esperto basta allora un
  dettaglio che nessun falso ha mai, e li boccia tutti con la stessa sicurezza,
  che il falso sia venuto quasi bene o malissimo: la domanda «è autentico?» non
  registra più i progressi del falsario, e alternare meglio i turni non ci mette
  rimedio. Per uscirne bisogna cambiare mestiere all'esperto, e chiedergli
  quanto distano i due mucchi di quadri, quello dei veri e quello dei falsi.
- **Mode collapse.** Il falsario scopre *un solo* falso che inganna sempre
  l'esperto e si limita a rifarlo. Risultato: $G$ genera sempre la stessa
  immagine (o pochissime varianti), buttando via tutta la varietà dei dati
  reali. Verrebbe da chiedersi come mai l'esperto non si insospettisca nel
  vedere sempre lo stesso quadro: il fatto è che li guarda **uno per volta**, e
  uno per volta quel falso è convincente. Per smascherare la ripetizione
  bisognerebbe fargli guardare un gruppo intero in blocco, ed è uno degli
  accorgimenti che si sono messi a punto per rimediare.
- **Mancata convergenza.** L'instabilità e il mode collapse si vedono. Questo è
  più insidioso, perché da fuori non sembra un guasto: il duello continua a
  girare regolarmente e non arriva mai da nessuna parte. Le immagini cambiano a
  ogni turno, non peggiorano e non migliorano, e non esiste un momento in cui
  si possa dire «ecco, è finito».

`````

`````{tab} Superiore

- **Instabilità.** L'ottimizzazione simultanea di un gioco minimax non equivale a minimizzare una singola funzione: la dinamica può divergere o entrare in cicli limite. Se $D$ diventa troppo accurato si ha $D(G(\mathbf{z}))\to 0$, e con l'obiettivo minimax originale questo annulla i gradienti verso $G$ (*vanishing gradients*); la non-saturating loss vista sopra scongiura l'annullamento, ma con un discriminatore quasi ottimo lo paga in aggiornamenti instabili e ad alta varianza {cite}`arjovsky2017towards`. Se invece $D$ è troppo debole, non fornisce segnale utile.

  Che $D$ diventi "troppo accurato" non è però un incidente di dosaggio, ed è
  un punto che cambia il rimedio. Arjovsky e Bottou mostrano che $p_G$, essendo
  l'immagine di uno spazio di rumore a poche decine o centinaia di dimensioni,
  vive su una varietà di dimensione bassa immersa nello spazio dei dati: con
  $\mathbf{z} \in \mathbb{R}^{100}$ e immagini $1024\times1024$ a colori, il supporto di
  $p_G$ ha dimensione al più $100$ dentro $\mathbb{R}^{3\,145\,728}$. Due
  varietà così hanno supporti quasi certamente disgiunti (o intersecantisi in
  un insieme di misura nulla), un discriminatore perfetto esiste, e su supporti
  disgiunti la $\mathrm{JSD}$ vale $\log 2$ **qualunque** sia la distanza fra le
  due distribuzioni. Il gradiente non è piccolo: è nullo, e resta nullo mentre $G$ si avvicina. Ed
  è qui che si chiude il cerchio con le ipotesi del teorema: il conto che dava
  $2\,\mathrm{JSD}$ presupponeva due densità, e un generatore vero una densità
  non ce l'ha. La caratterizzazione dell'ottimo resta vera; è il mondo in cui
  vale a non essere quello dell'addestramento. Alternare meglio i turni non lo
  risolve, ed è da qui che
  nasce l'idea di cambiare misura, cioè la Wasserstein GAN.
- **Mode collapse.** $G$ mappa molti $\mathbf{z}$ diversi su una stessa uscita
  $\tilde{\mathbf{x}}$: $p_G$ collassa su pochi modi di $p_{\text{dati}}$. Sembra un
  paradosso, visto che l'obiettivo ideale ha minimo solo in $p_G =
  p_{\text{dati}}$ e la $\mathrm{JSD}$ i modi mancanti li paga eccome; la
  spiegazione è che l'addestramento non sta ottimizzando quell'obiettivo. Da un
  lato conta l'ordine dei quantificatori {cite}`goodfellow2016nips`: la
  soluzione di $\max_D \min_G$ è
  *esattamente* il generatore che manda ogni $\mathbf{z}$ sul punto che $D$ crede più
  reale, e la discesa alternata non privilegia $\min_G \max_D$ sull'altro
  ordine. Dall'altro c'è la loss non-saturating che stiamo usando, il cui
  termine $\mathrm{KL}(p_G \,\|\, p_{\text{dati}})$ addebita un costo enorme a un
  campione implausibile ($p_G > 0$ dove $p_{\text{dati}} \approx 0$) e un costo
  che tende a zero a un modo abbandonato ($p_{\text{dati}} > 0$ dove $p_G
  \approx 0$): il collasso non lo previene, lo premia.
- **Mancata convergenza.** L'equilibrio di Nash del gioco non è garantito raggiungibile con la sola discesa del gradiente; i parametri possono orbitare indefinitamente attorno all'ottimo senza stabilizzarsi.

`````

## La loss non dice niente: come si misura una GAN

C'è una domanda che a questo punto è inevitabile, e la risposta non è affatto
ovvia: **come si fa a sapere se sta funzionando?**

In tutto il resto del libro la risposta è la stessa: si guarda la loss su un
mucchietto di esempi tenuti da parte apposta, e se scende va bene. Qui non
funziona, per un motivo strutturale. Le due loss non misurano la qualità:
misurano **chi dei due sta vincendo in questo momento**. Se la loss del
generatore scende può voler dire che genera meglio, oppure soltanto che il
discriminatore si è indebolito. Al punto di equilibrio teorico, quando i falsi
sono perfetti, il discriminatore tira a indovinare e le loss si assestano su
valori che non distinguono un capolavoro da un disastro. Guardare le immagini a
occhio, per contro, non regge sui numeri veri (nessuno esamina a una a una
cinquantamila immagini) e soprattutto **non vede il mode collapse**: mille
immagini bellissime e tutte uguali, se le si guarda una per volta, sembrano un
successo.

Si può controllare. Si prende quel ciclo, riga per riga, e gli si dà un
compito minuscolo di cui conosciamo già la risposta.

Il compito è questo: al posto delle immagini, quattromila punti su un foglio,
raccolti attorno a otto mucchietti disposti in cerchio come le ore di un
orologio a otto ore. Ogni mucchietto è stretto, i suoi punti cadono quasi tutti
entro $0{,}15$ dal proprio centro, mentre da un centro al successivo ci sono
circa $1{,}5$: sono otto isolotti ben separati, e da lontano si contano a
occhio. Il falsario deve imparare a produrre punti che sembrino usciti da lì.

Il vantaggio di un compito così è che permette di contare quello che su un
volto non si potrebbe contare. Si fa generare al falsario un mucchio di punti
suoi, e su quelli si guardano due cose: **quanti mucchietti ha imparato** e
**quanti dei suoi punti sono a segno**, cioè cadono dentro un isolotto, entro
$0{,}15$ da un centro. Per il primo conto diciamo che un mucchietto è coperto
se ci finisce almeno l'uno per cento dei punti del falsario: è una soglia
larga, perché uno che avesse imparato bene tutti e otto ne metterebbe in
ciascuno un ottavo, cioè il dodici e mezzo per cento. Un mucchietto non coperto
è quindi un mucchietto proprio abbandonato, non uno servito male.

I mucchietti sono otto campane, di quelle
disegnate poco fa, con la larghezza (in gergo la *deviazione standard*) di
$0{,}05$, disposte sui vertici di un ottagono di raggio $2$; il raggio
dell'isolotto, $0{,}15$, è tre volte quella larghezza. Generatore e
discriminatore sono due reti con due strati nascosti da $128$ unità, con la
funzione di attivazione detta *leaky ReLU* (quella che lascia passare anche i
negativi, molto ridotti: $0{,}2$ volte il loro valore), e il falsario parte da
**due** soli numeri casuali. Come allenatore si usa Adam, quello già visto nei
{doc}`capitoli su PyTorch </PyTorch/overview>`, con correzioni di ampiezza $2\cdot 10^{-4}$: piccole, che
è il modo di tenere a bada l'instabilità detta sopra. I gruppi sono da $256$
esempi e i giri quattrocento, dove un giro vuol dire **una passata sull'intero
insieme** dei quattromila punti, non un singolo gruppo: sono due dettagli che
cambiano tutto, perché con quattrocento gruppi soli il falsario non impara
niente e i mucchietti restano scoperti. L'unica cosa che cambia da un
addestramento all'altro è il numero da cui parte il sorteggio, i semi da $0$ a
$3$.

Prima dei numeri servono due riferimenti, altrimenti le cifre che seguono non
dicono niente. Se l'esperto fosse ridotto a rispondere «cinquanta e cinquanta»
a qualunque cosa, cioè a tirare a indovinare, la sua loss varrebbe $1{,}39$ e
quella del falsario $0{,}69$. È lo stesso conto fatto due volte da una parte e
una dall'altra: l'esperto viene giudicato su due risposte, una su un quadro
vero e una su un falso, e il falsario su una sola, la propria; per questo il
suo numero è la metà.

Quei due valori non sono un voto di promozione: sono il punto in cui il gioco
è in parità. Quello che conta è **di quanto** ciascuna loss se ne allontana, e
la cosa più comoda è tradurla nella domanda vera: quanto l'esperto crede vero
un falso. A $0{,}69$ lo crede vero una volta su due, cioè non lo distingue affatto; a $0{,}81$ scende a poco più di quattro volte su dieci; a $1{,}27$ a meno di
tre su dieci, e lì lo sta smascherando sette volte su dieci. (La conversione è
$e^{-\mathcal{L}_G}$, che è una media di logaritmi riportata indietro: la
frequenza vera sta un pelo più in alto, mai più in basso.)

Il risultato più netto sta dentro un singolo addestramento, e si ripete in
tutti e quattro (uno per seme, quattro addestramenti identici in tutto tranne
il sorteggio iniziale): **la loss del generatore sale mentre il generatore
migliora**. Al primo giro non ha imparato niente (nessun mucchietto coperto,
nessun punto a segno) e la sua loss vale fra $0{,}68$ e $0{,}73$, cioè proprio
lì attorno a $0{,}69$: non perché il gioco sia in parità, ma perché al primo
giro l'esperto non sa ancora riconoscere i falsi, e su quelli risponde più o
meno a caso pure lui. Quattrocento giri dopo la loss del falsario vale di più
in tutti e quattro i casi: fra $0{,}76$ e $0{,}81$ nei tre che sono arrivati a
coprire tutti e otto i mucchietti, $1{,}27$ nel quarto. Un criterio di arresto
che aspettasse la loss più bassa avrebbe fermato tutto al primo giro, con un
generatore buono a niente.

Poi c'è la differenza fra un addestramento e l'altro. I tre che convergono
finiscono con loss tutte in una fascia stretta e appena fuori dalla parità
(`loss_D` fra $1{,}33$ e $1{,}36$, `loss_G` fra $0{,}76$ e $0{,}81$: l'esperto
crede veri i falsi fra il $44$ e il $47$ per cento delle volte, cioè quasi non
li distingue). La qualità invece non si somiglia per niente: i punti a segno
vanno dal $79\%$ al $91\%$, dove un generatore perfetto ne farebbe circa il
$99\%$. Loss quasi identiche, dodici punti di qualità di differenza.

Il quarto è il caso da guardare con attenzione, perché è l'unico finito in
*mode collapse*, con cinque mucchietti coperti su otto. Lì le loss *lo dicono*,
e lo dicono perché escono dalla fascia: $0{,}97$ per l'esperto e $1{,}27$ per
il falsario, cioè un esperto che smaschera i falsi più di sette volte su dieci.
Ma è esattamente la diagnosi di partenza, non una smentita: le loss non hanno
misurato la qualità, hanno misurato chi dei due stesse vincendo. Che nel quarto
caso le due cose coincidano è una fortuna, non un metodo, e i primi tre lo
mostrano.

Serve una misura che giudichi un **insieme** di immagini invece di una sola:
in gergo, la loro *distribuzione*, cioè come si spartiscono fra i vari tipi
possibili, quanti gatti e quanti cani e in quali pose, non soltanto se ciascuna
presa da sé è venuta bene. La strada che si è imposta è obliqua: usare una rete
già addestrata a riconoscere immagini (storicamente Inception, addestrata su
ImageNet) come strumento di misura.

`````{tab} Elementare

Il primo tentativo, l’**Inception Score**, chiede due cose insieme a un
giudice esterno che sa riconoscere gli oggetti. E qui attenzione, perché
entra in scena un personaggio nuovo: non è l'esperto d'arte del duello, è un
giudice terzo, una rete addestrata altrove a riconoscere cani, automobili e
divani, che con la nostra partita non c'entra niente e non ha nessun interesse
a farla finire in un modo o nell'altro. Il falsario e l'esperto restano dove
sono; questo signore arriva a cose fatte e guarda i risultati.

Primo: guardando una singola
immagine generata, il giudice deve saper dire con sicurezza cos'è («questo è
un cane», non «forse un cane, forse un divano»); se esita, l'immagine è
informe. Secondo: guardando tutte le immagini generate insieme, deve trovarci
soggetti diversi; se sono tutti cani, c'è mode collapse. Un punteggio alto
significa immagini nitide e varie.

Il difetto salta all'occhio appena lo si dice: in questa misura **le immagini
vere non entrano mai**. Un generatore potrebbe produrre cani nitidi e assortiti
che non somigliano a nessun cane esistente, e prendere un bel voto.

Il **FID** ripara proprio questo, e comincia da un'osservazione su come lavora
il giudice. Una rete che riconosce oggetti non salta dall'immagine al nome in
un colpo: ci arriva per gradini, e a ogni gradino l'immagine è diventata una
lista di numeri più corta e più riassuntiva di quella di prima (prima i
contorni, poi le parti, poi le cose). L'ultima lista prima del nome descrive
l'immagine senza ancora nominarla, ed è quella che qui interessa: invece di
chiedere al giudice come si chiama l'oggetto, gli si sbircia dentro e ci si
prende quei numeri.

Da lì al disegno di una nuvola il passo è breve. Immagina che quei numeri
siano due soltanto: allora ogni immagine diventa un
punto su un foglio, come una città su una cartina, e mille immagini fanno
mille punti. Immagini che si somigliano finiscono vicine, immagini diverse
lontane, e l'insieme dei punti forma una macchia con una sua posizione e una
sua forma: la **nuvola**. I numeri veri sono più di due (duemila e passa), il
foglio quindi non si può disegnare, ma i conti si fanno lo stesso e la nuvola
c'è.

Una nuvola per le immagini vere, una per
quelle generate. Se le due nuvole si sovrappongono, il generatore ha imparato;
se stanno in due posti diversi, no; e se quella generata è molto più stretta
dell'altra, il generatore sta ripetendo poche cose. Il FID è la distanza fra
le due nuvole, e più è **basso**, meglio è. (Le tre lettere stanno per *Fréchet
Inception Distance*: Inception è il nome del giudice, la distanza è quella fra
le due nuvole, e Fréchet è il matematico che ha definito il modo di misurarla.)

Con un'avvertenza da mettere subito accanto a quel «più è basso, meglio è»: di
una nuvola il conto guarda soltanto dove sta il suo centro e quanto è larga.
Due arcipelaghi, allora, con lo stesso centro e la stessa larghezza ma fatti in
modo diverso: due isole lontane da una parte, un'unica macchia
uniforme che le copre entrambe dall'altra. Per questo conto sono la stessa
cosa, e non lo sono affatto: il secondo ha perso i due gruppi e ha riempito di
roba proprio il vuoto che li separava. Un generatore che schiaccia la varietà
dei soggetti in una poltiglia indistinta, invece di riprodurne i gruppi, può
quindi prendere un ottimo voto.

`````

`````{tab} Superiore

L’**Inception Score** {cite}`salimans2016improved` combina le due richieste in
un'unica quantità:

$$
\text{IS} = \exp\Big( \mathbb{E}_{\mathbf{x} \sim p_G}\big[\, \mathrm{KL}
\big( p(y \mid \mathbf{x})\,\|\,p(y) \big) \,\big] \Big),
$$

dove $p(y\mid \mathbf{x})$ è la distribuzione sulle classi che il classificatore assegna
al campione $\mathbf{x}$ e $p(y) = \mathbb{E}_{\mathbf{x}\sim p_G}[p(y\mid \mathbf{x})]$ è la marginale
sull'intero insieme generato. La divergenza KL è grande quando la prima è
concentrata (campione riconoscibile) e la seconda è piatta (insieme vario):
nitidezza e varietà in una formula sola. Si valuta tipicamente su
decine di migliaia di campioni. I limiti sono noti: non usa mai $p_{\text{dati}}$,
è cieco alla varietà *dentro* una classe, e dipende dalle mille classi di
ImageNet, il che lo rende poco sensato fuori dalle immagini naturali.

La **Fréchet Inception Distance** {cite}`heusel2017gans` abbandona le classi e
lavora sulle attivazioni di uno strato intermedio (il vettore da $2048$
componenti del *pooling* finale di Inception). Si approssimano le due
popolazioni di attivazioni, reali e generate, con due gaussiane
$\mathcal{N}(\boldsymbol{\mu}_r, \boldsymbol{\Sigma}_r)$ e
$\mathcal{N}(\boldsymbol{\mu}_g, \boldsymbol{\Sigma}_g)$, e si misura la
distanza di Fréchet fra le due, che per gaussiane ha forma chiusa:

$$
\text{FID} = \lVert \boldsymbol{\mu}_r - \boldsymbol{\mu}_g \rVert_2^2
+ \operatorname{Tr}\!\Big( \boldsymbol{\Sigma}_r + \boldsymbol{\Sigma}_g
- 2\big(\boldsymbol{\Sigma}_r \boldsymbol{\Sigma}_g\big)^{1/2} \Big).
$$

Il primo termine confronta i centri delle due nuvole, il secondo la loro forma:
è quest'ultimo a far pagare il collasso *di varianza*, perché un generatore che
ripete sempre la stessa uscita ha covarianza nulla e paga
$\operatorname{Tr}(\boldsymbol{\Sigma}_r)$ anche col centro azzeccato. Il FID correla meglio
dell'IS con il giudizio umano ed è oggi lo standard di fatto.

Restano quattro avvertenze da tenere a mente quando si leggono due FID a
confronto. È **distorto verso l'alto con pochi campioni**, quindi due valori
calcolati su numerosità diverse non si confrontano. Dipende dai dettagli
implementativi (come si ridimensionano le immagini, quale versione di Inception,
quale interpolazione), al punto che numeri presi da paper diversi vanno
maneggiati con prudenza. Resta un giudizio dato da un classificatore
addestrato su fotografie: su volti, radiografie o disegni misura qualcosa,
ma non esattamente ciò che dice di misurare.

E soprattutto: il FID vede **solo i primi due momenti**. Approssimare due
popolazioni di attivazioni con due gaussiane significa non poterle distinguere
quando media e covarianza coincidono, per quanto diverse siano davvero. Un
esempio costruito apposta lo mostra bene, e sta in una dimensione sola: i dati
reali sono la mistura in parti uguali di $\mathcal{N}(-3,\,1)$ e
$\mathcal{N}(+3,\,1)$, il generatore emette la sola $\mathcal{N}(0,\,10)$, che
di quella mistura ha esattamente la media e la varianza. Per costruzione il FID
fra le due è **zero**, e su un campione finito di $50\,000$ punti per parte
resta dell'ordine di $10^{-4}$ (fra $2$ e $9 \cdot 10^{-4}$ su tre sorteggi
diversi), cioè indistinguibile da zero. Eppure quel
generatore ha perso per strada l'intera struttura a due modi, e riempie di
campioni proprio la voragine che li separa: nella fascia $|x| < 1$ finisce il
$25\%$ delle sue uscite contro il $2{,}3\%$ dei dati reali. Il termine sulle
covarianze smaschera il collasso su un punto; la perdita di modi a momenti
invariati, no.

`````

Conviene fissare un punto che tornerà: nessuna delle due giudica una singola
immagine, giudicano un **insieme**. L'Inception Score guarda l'insieme
generato e basta, ed è il suo difetto; il FID lo confronta con l'insieme delle
immagini vere. Ma il FID di una foto non esiste, e nemmeno il suo Inception
Score. Ed è coerente con quello che una GAN cerca di fare, cioè avvicinare il
mucchio delle immagini che genera a quello delle immagini vere: si valuta
l'obiettivo dichiarato, non il singolo prodotto. Il FID sarà anche l'unità di
misura con cui, nel {doc}`capitolo sui modelli di diffusione </ModelliDiffusione/overview>`, la nuova famiglia
dimostrerà di aver superato le GAN.

## Accorgimenti pratici (cenni)

La ricerca successiva ha prodotto una cassetta degli attrezzi per domare
l'addestramento. Qui ne diamo solo i titoli, e sono cenni (della sola DCGAN
riparleremo nella prossima sezione); il filo che li unisce è che si può
intervenire su tre cose diverse.

Si può cambiare **com'è fatta** ciascuna delle due reti, dando loro un occhio
adatto alle immagini invece che a una lista qualunque di numeri: è la ricetta
delle **DCGAN** {cite}`radford2016unsupervised`, che la prossima sezione
racconta per esteso.

Si può cambiare **come si misura** la distanza fra i falsi e i veri, dove
"distanza" non è fra due immagini ma fra i due mucchi: quello delle immagini
vere e quello delle generate. È la strada della **Wasserstein GAN**
{cite}`arjovsky2017wasserstein`, che al posto della probabilità "è autentico o
no" adotta una misura dal comportamento più regolare, che cala e cresce con
dolcezza mentre i due mucchi si avvicinano invece di saltare da un estremo
all'altro. (È un'altra distanza da quella del FID, e si usa in un altro
momento: questa la si calcola durante l'addestramento, ed è la cosa che il
falsario cerca di far scendere.) È il rimedio alla situazione peggiore vista
sopra, quella in cui chiedere all'esperto «quanto manca perché passi per vero?» tiene viva la
correzione, ma quando lui è molto più bravo la risposta è sempre «moltissimo»,
e il falsario oscilla invece di avanzare. Una distanza si comporta meglio: dice
quanto manca *e* di quanto ci si è avvicinati all'ultimo ritocco, anche quando i
due mucchi sono ancora lontanissimi.

Il prezzo è un vincolo sull'esperto, che dev'essere prudente: fra due immagini
che si somigliano, i suoi due giudizi non possono essere più distanti di quanto
lo siano le immagini (in gergo, dev'essere 1-Lipschitziano). La WGAN lo otteneva obbligando ogni
peso della rete a restare fra due valori; Gulrajani e colleghi
{cite}`gulrajani2017improved` hanno poi mostrato che così l'esperto si
impoverisce, e che le correzioni finiscono per esplodere o per sparire. Il
rimedio che si è imposto è il loro, il *gradient penalty*: invece di stringere
i pesi, si aggiunge alla loss dell'esperto una multa che cresce quando la sua
risposta cambia troppo in fretta.

E si può cambiare **il regolamento del duello**: chiedere all'esperto di non
essere mai sicuro al cento per cento, ma di fermarsi a "reale al novanta"
(*label smoothing*), perché un giudice mai del tutto certo dà lezioni più
utili; fargli guardare i falsi a gruppi invece che uno per volta
(*minibatch discrimination*), così che un falsario che ripete sempre lo stesso
quadro venga smascherato proprio per la ripetizione; dosare i turni delle due
reti perché nessuna delle due prenda troppo vantaggio sull'altra.

Su due di queste tre leve, però, va messa un'avvertenza, perché cambiare **la
misura** cambia il senso di quel che si fa sul **regolamento**, ed è il genere
di dettaglio che fa perdere pomeriggi. Cambiare la misura cambia anche
il mestiere di chi giudica: con la probabilità l'esperto rispondeva «quanto lo
credo vero», un numero fra zero e uno, e nella rete c'era una funzione apposta
a schiacciare l'uscita dentro quell'intervallo; con la distanza risponde invece
con un punteggio senza tetto né pavimento, quella funzione sparisce, e nei
paper l'esperto non si chiama più discriminatore ma **critico**. Il punto è che
il numero calcolato dal critico *è* la distanza fra i due mucchi soltanto se il
critico ha fatto del suo meglio: se è mediocre, la sua risposta non misura
niente, e la correzione che consegna al falsario indica una direzione che non
porta da nessuna parte. Quindi il consiglio si capovolge. Con il punteggio
classico l'esperto non deve diventare troppo bravo, altrimenti il suo giudizio
si schiaccia sul «falso» e smette di correggere; con la distanza conviene
lasciarlo allenare fino in fondo *prima* di muovere il falsario, e lo si fa a
turni sbilanciati: cinque giri del critico per ogni giro del falsario, nei due
lavori che hanno introdotto la ricetta
{cite}`arjovsky2017wasserstein,gulrajani2017improved`. Con la multa sui gradienti
arriva anche un divieto, e nasce dallo stesso ragionamento. Nelle reti si usa
spesso un accorgimento che, a ogni passaggio, rimette in riga i numeri di un
gruppo di immagini guardandoli **tutte insieme** (si chiama *batch
normalization*): nel critico non ci va, perché così il giudizio su
un'immagine finirebbe per dipendere dalle altre del gruppo, mentre la multa è
scritta per un'immagine alla volta {cite}`gulrajani2017improved`.

Nessuno di questi trucchi è una bacchetta magica, e addestrare una GAN resta in
buona parte un mestiere che si impara provando. È anche il motivo per cui la
storia delle GAN è una fila di ricette, ciascuna che aggiusta un guasto della
precedente: ed è la storia della prossima sezione.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Una GAN è un **duello** fra due reti: il falsario parte da una manciata di
  numeri casuali e ne ricava un dato che sembri autentico, l'esperto guarda un
  dato e dice quanto lo crede vero.
- Quello che torna indietro dall'esperto al falsario **non è il verdetto**: è
  una lista lunga quanto il quadro, che per ogni puntino dice da che parte
  tirare e con quanta forza. Per questo l'esperto dev'essere una rete: una
  persona darebbe lo stesso giudizio e nessuna lista. E la realtà entra nel
  gioco da un lato solo, perché è l'esperto (mai il falsario) a vedere i quadri
  autentici, e a essere corretto su quelli.
- Giocano un **punteggio unico**: quello che è un bene per uno è un male per
  l'altro. L'equilibrio arriva quando i falsi non si distinguono più dai veri,
  e lì l'esperto può soltanto tirare a indovinare.
- Si allenano **a turni**, uno per volta, ed è un addestramento
  capriccioso: attenzione al *mode collapse* (il falsario trova un solo quadro
  che inganna sempre e si limita a rifarlo) e alla mancata convergenza. Quando
  l'esperto è troppo bravo, il suo giudizio è talmente schiacciato sul "falso"
  che non si muove più, e senza movimento non c'è correzione: si rimedia
  chiedendo al falsario, nel suo turno, di far passare i propri quadri per
  autentici; il prezzo sono correzioni più sbalzate, e un duello che non si
  lascia più tenere con un punteggio solo.
- Cambiando **il modo di misurare** (dalla probabilità «quanto lo credo vero»
  alla distanza fra il mucchio dei veri e quello dei falsi) chi giudica cambia
  mestiere e nome: diventa un **critico** che dà un punteggio senza tetto né
  pavimento. E si capovolge il consiglio di prima: il critico va lasciato
  allenare fino in fondo prima di muovere il falsario, cinque suoi giri per
  ogni giro dell'altro, perché soltanto un critico al meglio delle proprie
  possibilità sta misurando davvero qualcosa.
- **La loss, cioè il conto dell'errore, non misura la qualità**: dice solo chi
  dei due sta vincendo. Si giudica confrontando *insiemi* di immagini, mai una
  alla volta: con l’**Inception Score** (nitidezza e varietà secondo un giudice
  esterno, che però le immagini vere non le guarda mai) e soprattutto con il
  **FID**, la distanza fra la nuvola delle immagini vere e quella delle
  generate: più è basso, meglio è. Neanche il FID però è infallibile: vede dove
  sta la nuvola e quanto è larga, quindi smaschera il falsario che ripete
  sempre lo stesso quadro, non quello che perde per strada interi soggetti
  lasciando la nuvola dov'era.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Una GAN è un **duello** tra due reti: il generatore $G$ trasforma rumore in dati sintetici, il discriminatore $D$ stima la probabilità che un dato sia reale.
- Ciò che $D$ restituisce a $G$ è $\partial \mathcal{L}_G / \partial \tilde{\mathbf{x}}$,
  un **vettore** nello spazio dei dati, non il verdetto scalare; la regola della
  catena lo compone con $\partial \tilde{\mathbf{x}} / \partial \theta_G$. Ne segue un
  requisito di progetto: $D$ dev'essere derivabile rispetto al proprio ingresso,
  o si spezza il primo fattore. Sui dati discreti a mancare è invece il secondo,
  perché una sequenza di simboli campionati non si deriva rispetto a $\theta_G$.
- Condividono un'unica **funzione di valore minimax**: $G$ la minimizza, $D$ la
  massimizza; l'obiettivo ideale ha minimo in $p_G = p_{\text{dati}}$, e lì il
  discriminatore ottimo vale $D^*(\mathbf{x})=\tfrac12$ sul supporto dei dati. La
  *caratterizzazione* dell'ottimo non è però una garanzia di convergenza: la
  prova vive nello spazio delle densità, l'addestramento in quello dei
  parametri.
- L'addestramento è **alternato** e notoriamente instabile: attenzione al
  *mode collapse* e alla mancata convergenza. I gradienti che svaniscono, invece,
  riguardano l'obiettivo minimax originale: la *non-saturating loss* usata nel
  codice li evita, al prezzo di aggiornamenti ad alta varianza quando $D$ è
  quasi ottimo, e di un gioco che non è più a somma zero.
- La **Wasserstein GAN** {cite}`arjovsky2017wasserstein` sostituisce la
  probabilità con una stima della distanza fra $p_G$ e $p_{\text{dati}}$: cade
  la sigmoide finale, $D$ diventa un **critico** a valori in $\mathbb{R}$ e va
  portato vicino all'ottimo *prima* di ogni passo di $G$ (cinque iterazioni nei
  due lavori originali), perché quella distanza è definita come un estremo
  superiore sulle funzioni 1-Lipschitziane e solo lì il gradiente che $G$
  riceve la approssima. Il vincolo di Lipschitz è imposto con il *weight
  clipping* nel lavoro originale e con il *gradient penalty*
  {cite}`gulrajani2017improved` in quello che si è affermato; quest'ultimo
  esclude però la *batch normalization* nel critico, perché la penalità è
  definita campione per campione mentre la batchnorm accoppia i campioni del
  minibatch.
- **La loss non misura la qualità**: dice solo chi sta vincendo. Si valuta
  confrontando *distribuzioni*, con l’**Inception Score** (nitidezza e varietà
  secondo un classificatore, ma senza mai guardare i dati veri) e soprattutto
  con il **FID**, la distanza fra la nuvola delle attivazioni reali e quella
  delle generate: più basso è meglio. Il FID però guarda solo i primi due
  momenti: il termine sulle covarianze smaschera il collasso di varianza, non
  la perdita di modi a media e covarianza invariate.
```

`````
