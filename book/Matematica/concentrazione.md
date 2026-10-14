# Quanto può sbagliare una media: le disuguaglianze di concentrazione

Una quantità casuale finisce lontano dalla propria media solo di rado, e lo si
può dire senza sapere come quella quantità è fatta: senza una campana, senza
una forma, senza ipotesi sul meccanismo che la genera. Bastano due numeri, il
centro e la larghezza. Irénée-Jules Bienaymé pubblica la disuguaglianza nel
1853; «disuguaglianza» vuol dire qui una regola della forma «la probabilità di
questo non supera quel numero». Quattordici anni dopo Pafnutij Čebyšëv la
ritrova in modo indipendente e se ne serve per dimostrare, in tre righe, la
legge dei grandi numeri; da allora in Francia si chiama disuguaglianza di
Bienaymé-Čebyšëv, e altrove porta il solo nome del secondo.

Chi lavora con i dati ne ha bisogno ogni giorno. Quante
prove servono per fidarsi di una percentuale? Quanto può sbagliare una media?
Le **disuguaglianze di concentrazione**, che dicono quanto una media resta
concentrata intorno al proprio valore atteso, rispondono con limiti validi per
ogni numero di prove $n$, e non solo in approssimazione come il teorema
del limite centrale. Non chiedono di conoscere la forma dei dati: la curva a
campana della {doc}`sezione su probabilità e statistica
</Matematica/probabilita-statistica>` descrive molte situazioni e non tutte,
mentre queste valgono qualunque sia la forma. Qualcosa chiedono comunque, e
ciascuna dichiara che cosa.

## Con la sola media: la disuguaglianza di Markov

`````{tab} Elementare

In una sala il reddito medio è trentamila euro. Senza sapere altro, quante
persone possono guadagnarne trecentomila?

Al massimo una su dieci, e il ragionamento sta in una riga: se fossero anche
solo undici su cento, quelle undici da sole porterebbero il totale della sala
sopra trentamila euro a testa, e la media sarebbe più alta di quella che ci
hanno detto. In generale la frazione di casi che raggiunge o supera una soglia
non supera la media divisa per la soglia: qui trentamila diviso trecentomila fa
un decimo.

La regola ha due caratteristiche da conoscere.

*Serve che la quantità non possa essere negativa.* Il calcolo funziona perché
nessuno guadagna meno di zero, e quindi chi sta sotto la soglia non può tirare
giù il totale a piacere. Applicata a una quantità che può andare in negativo, la
regola sarebbe falsa.

*La stima è generosa, e a volte è esatta.* Si può costruire una sala in cui la
disuguaglianza è un'uguaglianza: dieci persone, una che guadagna trecentomila
euro e nove che non guadagnano niente. Media trentamila, e uno su dieci
alla soglia. Ne segue che la regola non si può migliorare senza
sapere qualcosa in più. È anche la sua debolezza: sapendo solo la media, di
più non si può dire, e quel «uno su dieci» è di solito molto più alto della
verità.

`````

`````{tab} Superiore

Sia $X\ge 0$ una variabile aleatoria con media finita e $a>0$. La
**disuguaglianza di Markov** afferma

$$
\Pr[X \ge a] \;\le\; \frac{\mathbb{E}[X]}{a} .
$$

La dimostrazione è una riga. Poiché $X\ge 0$,

$$
\mathbb{E}[X] \;\ge\; \mathbb{E}\big[X\,\mathbf{1}\{X\ge a\}\big]
\;\ge\; a\,\Pr[X\ge a],
$$

dove $\mathbf{1}\{\cdot\}$ vale $1$ quando la condizione è vera e $0$
altrimenti: si butta via il contributo dei casi sotto la soglia (non negativo,
quindi buttarlo può solo far scendere) e si sostituisce ogni valore sopra la
soglia con la soglia stessa.

L'ipotesi $X\ge 0$ è essenziale: senza di essa i valori negativi possono
compensare quelli grandi e la media non dice più niente sulla coda. La
disuguaglianza è inoltre ottimale, perché esiste una distribuzione che la
realizza con l'uguaglianza: $X=a$ con probabilità $\mathbb{E}[X]/a$ e $X=0$
altrimenti. Non si può quindi migliorare a parità di informazione, e tutte le
disuguaglianze più forti si ottengono chiedendo di sapere qualcosa in più.

`````

## Con media e larghezza: la disuguaglianza di Čebyšëv

La disuguaglianza di Markov usa la sola media. Applicata al quadrato dello
scarto dalla media, $(X-\mu)^2$, che non è mai negativo e ha per media la
varianza, dà un limite che usa anche la varianza. Lo stesso procedimento,
applicare Markov a una funzione crescente della variabile, serve anche nella
dimostrazione di Hoeffding, dove la funzione è un esponenziale.

`````{tab} Elementare

Lo scarto dalla media può essere positivo o negativo, quindi la regola di prima
non gli si applica. Elevandolo al quadrato però non è mai negativo, e in più i
valori lontani pesano molto di più dei vicini, come serve per parlare di code.
Il valore medio di quello scarto al quadrato è poi una grandezza già nota: la
varianza, la larghezza al quadrato.

Il ragionamento è la regola della sala applicata allo scarto al quadrato.
Essere lontani almeno due larghezze equivale ad avere lo scarto al quadrato
almeno quattro volte la varianza: la frazione di casi che ci arriva non supera
la media dello scarto al quadrato (la varianza) divisa per quattro volte la
varianza, quindi un quarto. Con tre larghezze la soglia diventa nove volte la
varianza, e resta al più un nono. Ne viene una regola che si ricorda a memoria:
qualunque sia la forma dei dati, almeno tre quarti di essi stanno entro due
larghezze dalla media, e almeno otto su nove entro tre. Non si è chiesto
niente: niente campana, niente simmetria, niente code sottili. Vale per i
redditi, per i tempi di risposta di un server, per gli errori di un modello.

Non chiedere niente ha un limite, che si vede confrontando con il caso in cui
invece si chiede molto. Se i dati seguono la curva a campana, entro due
larghezze non c'è il settantacinque per cento ma il novantacinque e mezzo, ed
entro tre il novantanove e sette. La differenza fra quei numeri mostra
quanto vale
sapere la forma della distribuzione. Chi non la sa deve accontentarsi della
garanzia debole, e la garanzia debole, sui dati veri, quasi sempre è molto
più prudente del necessario.

C'è però una situazione in cui la garanzia debole è esatta, e serve a capire
che non si può fare di meglio a scatola chiusa. Se un ottavo dei dati vale meno
due, un ottavo vale più due e i restanti tre quarti valgono zero, la media è
zero; la varianza è un ottavo per quattro più un ottavo per quattro, che fa uno,
e quindi anche la larghezza è uno; e a due larghezze o più dalla media c'è
precisamente un quarto dei dati. La regola diventa un'uguaglianza, e chi
sperasse in un numero migliore avrebbe torto.

`````

`````{tab} Superiore

Sia $X$ con media $\mu$ e varianza $\sigma^2$ finite. Applicando Markov alla
variabile non negativa $(X-\mu)^2$ con soglia $k^2\sigma^2$:

$$
\Pr\big[\,|X-\mu| \ge k\sigma\,\big]
= \Pr\big[(X-\mu)^2 \ge k^2\sigma^2\big]
\le \frac{\mathbb{E}[(X-\mu)^2]}{k^2\sigma^2}
= \frac{1}{k^2} ,
$$

la **disuguaglianza di Čebyšëv**. Nella forma con la soglia assoluta si scrive
$\Pr[|X-\mu|\ge\varepsilon] \le \sigma^2/\varepsilon^2$.

Per $k=2$ dà $\Pr \le 1/4$, per $k=3$ dà $\Pr\le 1/9$, contro le code esatte
di una gaussiana, che valgono $0{,}0455$ e $0{,}0027$: un fattore cinque e mezzo
nel primo caso, quarantuno nel secondo. È quanto si guadagna, sulle code,
sapendo che la distribuzione è gaussiana.

Anche questa disuguaglianza è ottimale, e il testimone è una distribuzione a tre
punti: $X=\pm\sigma k$ ciascuno con probabilità $1/(2k^2)$ e $X=0$ con la
probabilità restante. Media zero, varianza $\sigma^2$, e coda esattamente
$1/k^2$.

Da qui la **legge dei grandi numeri** in tre righe. Per $X_1,\dots,X_n$
indipendenti e identicamente distribuite con varianza $\sigma^2$, la media
campionaria $\bar{X}_n$ ha varianza $\sigma^2/n$, quindi

$$
\Pr\big[\,|\bar{X}_n - \mu| \ge \varepsilon\,\big]
\;\le\; \frac{\sigma^2}{n\,\varepsilon^2}
\;\xrightarrow[n\to\infty]{}\; 0 .
$$

La {doc}`sezione su probabilità e statistica <probabilita-statistica>`
enunciava questo risultato e ne dava la velocità. Questa è la dimostrazione
della forma debole sotto un'ipotesi in più, l'esistenza della varianza; in
compenso non chiede nemmeno l'indipendenza piena, perché per scrivere
$\mathrm{Var}(\bar{X}_n)=\sigma^2/n$ basta che le $X_i$ siano scorrelate a due
a due. La legge debole vale anche con la sola media finita (teorema di
Khinčin), ma la si dimostra con altri strumenti, e senza varianza si perde la
velocità. Si vede anche da dove viene la scala $1/\sqrt{n}$ dell'errore:
$\varepsilon$ compare al quadrato accanto a $n$.

`````

## Quando i valori stanno in un intervallo: Hoeffding

Il limite di Čebyšëv alla probabilità di un errore grande cala come $1/n$, con
$n$ il numero di prove, e $1/n$ cala lentamente. Se in più i valori stanno in
un intervallo noto, il limite cala esponenzialmente in $n$, e il vantaggio
cresce con la sicurezza richiesta. Per una variabile in $[0,1]$, a parità di
tolleranza, con una probabilità di errore ammessa di uno su venti gli esempi
necessari si riducono di un fattore 2,7 rispetto a Čebyšëv (con $\sigma^2\le
1/4$), e per uno su mille di circa 66.

`````{tab} Elementare

Le risposte di un modello a un quiz sono giuste o sbagliate: uno oppure zero,
niente in mezzo, e soprattutto niente che possa schizzare a mille. Questa
informazione in più (i valori stanno dentro un intervallo noto) cambia le
garanzie, perché toglie di mezzo lo scenario che rovina tutto, quello in cui un
solo caso mostruoso sposta la media.

Con i valori limitati, la probabilità di sbagliare di tanto non cala in
proporzione al numero di prove: crolla, perché ogni prova in più la moltiplica
per una frazione fissa minore di uno. Il risultato pratico risponde alla
domanda che ci si fa prima di preparare un banco di prova: quante
domande servono perché l'accuratezza misurata stia a meno di due punti
percentuali da quella vera (la *tolleranza*), con il rischio di mancare quel
bersaglio al più una volta su venti, la stessa volta su venti che i sondaggi
mettono in conto?

- non sapendo niente della forma, la sola larghezza chiede 12 500 domande;
- sapendo che le risposte stanno fra zero e uno, ne bastano 4 612;
- dando per buona la curva a campana, che qui è un'approssimazione ragionevole
  ma resta un'approssimazione, 2 401. È il calcolo dei sondaggi, cento diviso la
  radice delle interviste, perché cento diviso 49 fa circa due punti.

Tre numeri per la stessa domanda, e la differenza fra loro è tutta nelle
ipotesi che si è disposti a fare. Il secondo è quello onesto nella maggior
parte dei casi: non chiede niente che non si possa controllare, e richiede meno
della metà delle domande del primo.

Nessuno dei tre dipende da quanto è grande l'insieme da cui le
domande vengono estratte. Per misurare l'accuratezza di un modello su un
miliardo di casi possibili servono le stesse quattromilaseicento domande che
servirebbero se i casi fossero centomila, purché siano sorteggiate a caso.
È come assaggiare la minestra: un cucchiaio basta a sapere quanto è
salata, sia la pentola piccola o grande, purché sia mescolata.

`````

`````{tab} Superiore

Siano $X_1,\dots,X_n$ indipendenti con $X_i\in[a,b]$ e media comune $\mu$. La
**disuguaglianza di Hoeffding** {cite}`hoeffding1963probability` afferma

$$
\Pr\big[\,|\bar{X}_n - \mu| \ge \varepsilon\,\big]
\;\le\; 2\exp\!\left(-\frac{2n\varepsilon^2}{(b-a)^2}\right) .
$$

Il confronto con Čebyšëv è netto: $1/(n\varepsilon^2)$ contro
$e^{-2n\varepsilon^2}$: decadimento polinomiale contro esponenziale. In
cambio serve l'ipotesi di limitatezza, che l'accuratezza soddisfa sempre; per
precision e recall la media si fa sui soli esempi predetti positivi, o sui
positivi veri, e l’$n$ da mettere nella formula è quel numero, spesso molto più
piccolo.

La dimostrazione ripete il procedimento di Čebyšëv con una funzione più ripida. Per
ogni $s>0$ la funzione $x\mapsto e^{sx}$ è positiva e crescente, quindi Markov
dà

$$
\Pr\big[\bar{X}_n-\mu\ge\varepsilon\big]
\le e^{-sn\varepsilon}\,\mathbb{E}\Big[e^{s\sum_i (X_i-\mu)}\Big]
=e^{-sn\varepsilon}\prod_{i=1}^n\mathbb{E}\big[e^{s(X_i-\mu)}\big],
$$

e il prodotto viene dall'indipendenza: è il solo punto in cui serve, e lì la
garanzia cade se i campioni sono correlati. Il lemma di Hoeffding maggiora
ciascun fattore con $e^{s^2(b-a)^2/8}$ per ogni variabile a valori in $[a,b]$:
posto $\varphi(s)=\log\mathbb{E}\big[e^{s(X_i-\mu)}\big]$, si ha
$\varphi(0)=\varphi'(0)=0$, e $\varphi''(s)$ è la varianza di $X_i$ sotto la
distribuzione ottenuta pesando ogni valore con $e^{sx}$, che sta ancora in
$[a,b]$ e quindi ha varianza al più $(b-a)^2/4$; il resto di Lagrange dà
$\varphi(s)\le s^2(b-a)^2/8$. Resta
$\exp\!\big(-sn\varepsilon+ns^2(b-a)^2/8\big)$, che minimizzato in
$s=4\varepsilon/(b-a)^2$ dà $\exp\!\big(-2n\varepsilon^2/(b-a)^2\big)$, e il
fattore $2$ davanti viene dall'altra coda. Il procedimento (Markov su $e^{sX}$,
poi il migliore $s$) è il metodo di Chernoff, lo stampo dell'intera famiglia.
Hoeffding usa soltanto l'ampiezza dell'intervallo della variabile, e ne ignora
la varianza: quando questa è molto più piccola di $(b-a)^2/4$ (un'accuratezza
vicina al $99\%$, per esempio) conviene la disuguaglianza di Bernstein, che la
mette in conto: se $|X_i-\mu|\le M$ e $\mathrm{Var}(X_i)\le\sigma^2$,

$$
\Pr\big[\,|\bar{X}_n-\mu|\ge\varepsilon\,\big]\;\le\;2\exp\!\left(-\frac{n\,\varepsilon^2}{2\sigma^2+\tfrac23 M\varepsilon}\right).
$$

Per un'accuratezza vera del $99\%$ ($\sigma^2\le 0{,}01$, $M\le1$),
$\varepsilon=0{,}01$ e $\delta=0{,}05$ bastano $984$ esempi, contro i
$18\,445$ di Hoeffding. La varianza vera non si conosce, e si usa un suo
maggiorante.

Risolvendo rispetto a $n$ si ottiene la formula che serve in pratica, per
decidere quanti esempi mettere in un insieme di prova:

$$
n \;\ge\; \frac{(b-a)^2\,\log(2/\delta)}{2\varepsilon^2} ,
$$

con $\varepsilon$ la tolleranza e $\delta$ la probabilità di sbagliare. Per
$[a,b]=[0,1]$, $\varepsilon = 0{,}02$ e $\delta = 0{,}05$ si ottiene
$n\ge 4612$. La stessa domanda posta a Čebyšëv (con $\sigma^2\le 1/4$, il
massimo per una variabile in $[0,1]$) dà $n\ge 12\,500$; l'approssimazione
normale dà $n\ge (1{,}96)^2\cdot 0{,}25/\varepsilon^2 = 2401$. Per l'accuratezza,
che è la media di una variabile di Bernoulli, esiste anche l'intervallo esatto
di Clopper-Pearson: nel caso peggiore ($p=1/2$) chiede circa la metà degli
esempi di Hoeffding, e l'approssimazione normale con $2401$ è già leggermente
ottimista. Hoeffding chiede di più perché usa soltanto l'intervallo $[0,1]$ e
non sa che la variabile è di Bernoulli: è lo strumento giusto per variabili
limitate qualsiasi (punteggi, perdite limitate), meno per contare risposte
esatte.

Due osservazioni cambiano il modo di progettare una valutazione. La prima: $n$
non dipende dalla cardinalità della popolazione, ma solo da $\varepsilon$ e
$\delta$; vale anche per un sorteggio senza reinserimento da una popolazione
finita, che non è indipendente (Hoeffding 1963, teorema 4), e se la popolazione
è più piccola di $n$ il calcolo non serve, perché la si misura tutta. La
seconda: $\varepsilon$ compare al quadrato, quindi dimezzare la tolleranza
quadruplica il costo. È lo stesso andamento $1/\sqrt{n}$ della legge dei grandi
numeri, qui in forma esplicita.

`````

```python
import numpy as np

def hoeffding(eps, delta, ampiezza=1.0):
    """Quante prove servono per stare entro eps con probabilita' 1 - delta."""
    return np.ceil(ampiezza**2 * np.log(2 / delta) / (2 * eps**2))

def chebyshev(eps, delta, var_max=0.25):
    return np.ceil(var_max / (delta * eps**2))

print(chebyshev(0.02, 0.05), hoeffding(0.02, 0.05))   # -> 12500.0 4612.0
print(np.ceil(1.96**2 * 0.25 / 0.02**2))              # normale -> 2401.0

# quanto costa non sapere la forma: la garanzia di Cebysev contro la coda
# vera di una campana
from math import erf, sqrt
for k in (2, 3):
    coda = 1 - erf(k / sqrt(2))
    print(k, round(1 / k**2, 4), round(coda, 4), round(1 / k**2 / coda, 1))
# -> 2 0.25 0.0455 5.5
# -> 3 0.1111 0.0027 41.2

# dimezzare la tolleranza quadruplica il costo
print(hoeffding(0.01, 0.05), hoeffding(0.02, 0.05))   # -> 18445.0 4612.0

# con un'accuratezza vicina al 99% la varianza è al più 0,01, e Bernstein
# la mette in conto
def bernstein(eps, delta, var_max, M=1.0):
    return np.ceil((2 * var_max + 2 * M * eps / 3) * np.log(2 / delta) / eps**2)

print(bernstein(0.01, 0.05, var_max=0.01), hoeffding(0.01, 0.05))
# -> 984.0 18445.0
```

## Il prezzo di guardare molte volte

Le garanzie precedenti valgono per una misura sola, decisa prima di guardare i
dati. Se si confrontano molte varianti sullo stesso insieme di prova e si tiene
la migliore, la stima della vincitrice è distorta verso l'alto (è la
maledizione del vincitore, *winner's curse*) e la garanzia a singola misura non
vale più.

`````{tab} Elementare

Cento varianti di un modello, tutte identiche nella sostanza, tutte con la
stessa accuratezza vera dell'ottanta per cento. Si misurano tutte sullo stesso
insieme di prova da mille esempi e si tiene la migliore.

Su mille domande un modello da ottanta per cento non ne azzecca esattamente
ottocento, come cento lanci di moneta non danno cinquanta teste spaccate: ogni
variante sbaglia domande un po' diverse, e il suo punteggio cade un po' sopra
o un po' sotto. La migliore risulterà sopra l'ottanta per cento, e non di
poco: ripetendo al calcolatore l'intero esperimento duecento volte, la
vincitrice sta tipicamente intorno a $83$, e in una singola prova può arrivare
a $84{,}2$, oltre quattro punti sopra il vero. Nessuna delle cento è
migliore delle altre: la più alta è la più fortunata, e il vantaggio apparente
è tutto caso. È la scatola delle mille monete
della {doc}`sezione su probabilità e statistica
</Matematica/probabilita-statistica>`, in cui una trentina di monete oneste
finiscono accusate per puro caso.

Il meccanismo si capisce contando le occasioni. Ogni singola variante ha una
probabilità piccola di sembrare molto meglio di quel che è; con cento varianti
quelle probabilità piccole si sommano, e la probabilità che almeno una
sembri molto meglio diventa grande.

Le due cure sono note e si usano poco. La prima è tenere da parte un secondo
insieme di prova, mai guardato durante la selezione, su cui misurare soltanto
la vincitrice: quella misura torna onesta perché è una misura sola. La seconda
è alzare l'asticella man mano che le varianti crescono. Il calcolo esiste e
chiede meno di quanto sembri: da dieci varianti a mille l'asticella si alza di
meno del doppio.

`````

`````{tab} Superiore

Questo difetto è la **maledizione del vincitore** (*winner's
curse*),
e il suo controllo è la disuguaglianza dell'unione, o di Boole, la stessa che
nella {doc}`sezione su probabilità e statistica <probabilita-statistica>` fonda
la correzione di Bonferroni: per eventi qualsiasi, dipendenti fra loro quanto
si vuole,

$$
\Pr\left[\bigcup_{i=1}^{k} A_i\right] \le \sum_{i=1}^{k}\Pr[A_i] .
$$

Applicata a $k$ modelli valutati sullo stesso insieme, con $A_i$ l'evento
«la stima $i$-esima si discosta dal vero di almeno $\varepsilon$», la garanzia
di Hoeffding va richiesta a livello $\delta/k$ per ciascuno, e la tolleranza
diventa

$$
\varepsilon(k) = \sqrt{\frac{\log(2k/\delta)}{2n}} ,
$$

che cresce come $\sqrt{\log k}$. La crescita è lenta, ma non nulla: con
$n=1000$ e $\delta = 0{,}05$, passare da dieci a mille varianti allarga la
tolleranza di un terzo scarso.

È lo stesso calcolo su cui poggiano i limiti di generalizzazione della teoria
dell'apprendimento, dove $k$ è la cardinalità della classe di ipotesi: la
garanzia peggiora come la radice del logaritmo del numero di funzioni fra cui
si sceglie. Questa è la ragione formale per cui una classe troppo ricca smette
di dare garanzie. Il caso in cui la classe è infinita si tratta sostituendo il
conteggio con una misura di complessità, la {doc}`dimensione VC
</TeoriaApprendimento/dimensione-vc>` o la {doc}`complessità di Rademacher
</TeoriaApprendimento/rademacher-margine>`, e il capitolo sulla {doc}`teoria
dell'apprendimento </TeoriaApprendimento/overview>` lo svolge a fondo; la
{doc}`sezione su overfitting e validazione
</MachineLearning/overfitting-validazione>` lo affronta dal lato pratico.

La correzione va applicata al
numero di confronti effettivamente fatti, non a quelli dichiarati. Chi
prova venti configurazioni, ne riporta una e chiama le altre esplorazione
preliminare ha comunque fatto venti confronti. La cura alternativa è un
secondo insieme di prova mai guardato durante la selezione: la vincitrice è
allora un modello fissato, indipendente da quell'insieme, e vale la garanzia a
singola misura, $k=1$ e $\varepsilon=\sqrt{\log(2/\delta)/(2n)}$, senza il
termine $\log k$.

`````

```python
rng = np.random.default_rng(0)

# cento varianti tutte uguali nella sostanza, stesso insieme di prova
vera, n_prova, k = 0.80, 1000, 100
stime = rng.binomial(n_prova, vera, size=k) / n_prova

print(round(stime.mean(), 4), stime.max())      # -> 0.8009 0.842
print(round(np.sqrt(vera * (1 - vera) / n_prova), 5))   # scarto tipico -> 0.01265

# ripetendo l'esperimento, la vincitrice è quasi sempre gonfiata
massimi = [(rng.binomial(n_prova, vera, size=k) / n_prova).max()
           for _ in range(200)]
print(round(float(np.median(massimi)), 4))      # -> 0.83

# di quanto va allargata la tolleranza, in funzione di quante varianti si
# sono provate: cresce come la radice del logaritmo, cioè pianissimo
def tolleranza(k, n=1000, delta=0.05):
    return np.sqrt(np.log(2 * k / delta) / (2 * n))

for k in (1, 10, 100, 1000):
    print(k, round(float(tolleranza(k)), 4))
# -> 1 0.0429
# -> 10 0.0547
# -> 100 0.0644
# -> 1000 0.0728
```

Tre punti percentuali di guadagno apparente nella vincitrice tipica (lo
$0{,}83$), e oltre quattro in un'estrazione come quella da $0{,}842$, prodotti
interamente dal caso, sono più di quanto separi molti modelli veri e molti
annunci. Nella simulazione le cento varianti sbagliano in modo indipendente, e
il gonfiamento è quello pieno: varianti simili sbagliano spesso sulle stesse
domande, e il gonfiamento si riduce, ma non sparisce. Chi legge una classifica
dovrebbe sempre chiedersi quante varianti sono state provate prima di quella
pubblicata.

## In pratica: le stesse garanzie su una distribuzione storta

```python
rng = np.random.default_rng(1)

# una distribuzione asimmetrica e con la coda lunga: niente campana
x = rng.exponential(scale=1.0, size=200_000)
mu, sigma = x.mean(), x.std()

for k in (2, 3):
    fuori = np.mean(np.abs(x - mu) >= k * sigma)
    print(k, round(fuori, 4), round(1 / k**2, 4))
# -> 2 0.0498 0.25
# -> 3 0.0184 0.1111
```

Per ciascun numero di larghezze il programma riporta la frazione di dati che
cade fuori e il limite di Čebyšëv. Il limite tiene con larghissimo margine,
come quasi sempre, e tiene senza sapere che i dati sono esponenziali: avrebbe
tenuto anche con dati costruiti apposta per metterlo in difficoltà.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Sapendo solo la media di una quantità che non può essere negativa, la
  frazione di casi che raggiungono una soglia non supera la media divisa per la
  soglia:
  in una sala con reddito medio trentamila euro, al più uno su dieci ne
  guadagna trecentomila.
- Sapendo anche la larghezza, almeno tre quarti dei dati stanno entro due
  larghezze dalla media, e almeno otto su nove entro tre. Vale qualunque forma
  abbiano i dati, e questa è tutta la sua forza; se la forma fosse una campana
  i numeri sarebbero il novantacinque e mezzo e il novantanove e sette, e
  quella differenza è quanto vale saperlo.
- Se in più i valori stanno dentro un intervallo noto (le risposte giuste o
  sbagliate valgono zero oppure uno) la garanzia diventa molto più forte, e si
  può calcolare quante prove servono: per misurare un'accuratezza a meno di due
  punti, sbagliando al massimo una volta su venti, ne bastano circa
  quattromilaseicento. Quel numero non dipende da quanto è grande
  l'insieme da cui le prove vengono estratte.
- Dimezzare la tolleranza quadruplica il numero di prove.
- Tutto questo vale per una misura sola. Provando cento varianti e tenendo la
  migliore, la vincitrice sembra migliore di quanto sia anche quando sono
  tutte identiche: con cento varianti arriva a tre o quattro punti sopra il
  vero. Le cure sono un secondo insieme di prova mai guardato, oppure alzare
  l'asticella man mano che le varianti crescono, e da dieci varianti a mille
  si alza di meno del doppio.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Markov: per $X\ge 0$, $\Pr[X\ge a]\le \mathbb{E}[X]/a$. Usa solo la
  media, è ottimale, e la non negatività è essenziale.
- Čebyšëv: $\Pr[|X-\mu|\ge k\sigma]\le 1/k^2$, ottenuta applicando Markov
  a $(X-\mu)^2$. Applicata a $\bar{X}_n$, con
  $\mathrm{Var}(\bar{X}_n)=\sigma^2/n$, dimostra la legge dei grandi numeri in
  tre righe e spiega perché l'errore cala come $1/\sqrt{n}$.
- Hoeffding: per $X_i\in[a,b]$ indipendenti,
  $\Pr[|\bar{X}_n-\mu|\ge\varepsilon]\le 2\exp(-2n\varepsilon^2/(b-a)^2)$,
  con decadimento esponenziale invece che polinomiale. Esplicitando $n$ si ottiene
  $n \ge (b-a)^2\log(2/\delta)/(2\varepsilon^2)$: per $\varepsilon=0{,}02$ e
  $\delta=0{,}05$ su $[0,1]$, $n\ge 4612$ contro i $12\,500$ di Čebyšëv e i
  $2401$ dell'approssimazione normale.
- Disuguaglianza dell'unione: valutando $k$ ipotesi sullo stesso insieme, la
  tolleranza diventa $\varepsilon(k)=\sqrt{\log(2k/\delta)/(2n)}$, che cresce
  come $\sqrt{\log k}$. È il *winner's curse*, e lo stesso calcolo sostiene i
  limiti di generalizzazione. La correzione si applica ai confronti fatti, non
  a quelli dichiarati.
```
`````

Queste disuguaglianze dicono qualcosa di vero su distribuzioni che non si
conoscono. Il caso opposto richiede di conoscere il meccanismo molto bene:
quando lo stato di un sistema dipende soltanto da dove si trovava un istante
prima, l'algebra lineare e la probabilità si incontrano, e la domanda «dove
finirà, andando avanti per sempre?» ha una risposta esatta.
