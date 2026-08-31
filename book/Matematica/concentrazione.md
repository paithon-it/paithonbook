# Quanto può sbagliare una media: le disuguaglianze di concentrazione

Nel 1853 Irénée-Jules Bienaymé pubblica una disuguaglianza che sta in mezza
pagina e che al tempo passa quasi inosservata. Dice che una quantità casuale
finisce lontano dalla propria media solo di rado, e lo dice senza sapere
niente di come quella quantità è fatta: nessuna campana, nessuna forma,
nessuna ipotesi sul meccanismo che la genera. Bastano due numeri, il centro e
la larghezza. Quattordici anni dopo Pafnutij Čebyšëv la ritrova per conto suo
e ne fa lo strumento con cui dimostra, in tre righe, la legge dei grandi
numeri; da allora in Francia si chiama disuguaglianza di Bienaymé-Čebyšëv, e
altrove porta il solo nome del secondo.

Chi lavora con i dati la incontra ogni giorno senza chiamarla per nome. Ogni
volta che ci si chiede quante prove servano per fidarsi di una percentuale,
quanto sia grande il rumore di una stima o se due modelli siano davvero
diversi, la risposta viene da una disuguaglianza di questa famiglia. E la loro
qualità migliore è che non chiedono di sapere che forma abbiano i dati: la
curva a campana della {doc}`sezione su probabilità e statistica
</Matematica/probabilita-statistica>` descrive tante situazioni e non tutte,
mentre queste valgono qualunque sia la forma. Qualcosa la chiedono comunque, e
ciascuna dichiara che cosa.

## Con la sola media: la disuguaglianza di Markov

`````{tab} Elementare

In una sala il reddito medio è trentamila euro. Senza sapere altro, quante
persone possono guadagnarne trecentomila?

Al massimo una su dieci, e il ragionamento sta in una riga: se fossero anche
solo undici su cento, quelle undici da sole porterebbero il totale della sala
sopra trentamila euro a testa, e la media sarebbe più alta di quella che ci
hanno detto. La regola generale è questa: **la frazione di casi che supera una
certa soglia non può essere più grande della media divisa per la soglia**. Qui
trentamila diviso trecentomila fa un decimo.

Due cose vanno notate, perché dicono che tipo di strumento sia.

*Serve che la quantità non possa essere negativa.* Il conto funziona perché
nessuno guadagna meno di zero, e quindi chi sta sotto la soglia non può tirare
giù il totale a piacere. Applicata a una quantità che può andare in negativo, la
regola sarebbe falsa.

*La stima è generosa, e a volte è esatta.* Si può costruire una sala in cui la
disuguaglianza è un'uguaglianza: dieci persone, una che guadagna trecentomila
euro e nove che non guadagnano niente. Media trentamila, e proprio uno su dieci
sopra la soglia. Il che vuol dire che la regola non si può migliorare senza
sapere qualcosa in più. Ed è anche la sua debolezza: sapendo solo la media, di
più non si può dire, e quel «uno su dieci» è di solito enormemente peggiore
della verità.

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

L'ipotesi $X\ge 0$ è essenziale e non decorativa: senza di essa i valori
negativi possono compensare quelli grandi e la media non dice più niente sulla
coda. La disuguaglianza è inoltre **stretta**, cioè esiste una distribuzione
che la realizza con l'uguale: $X=a$ con probabilità $\mathbb{E}[X]/a$ e $X=0$
altrimenti. Non si può quindi migliorare a parità di informazione, e tutte le
disuguaglianze più forti si ottengono chiedendo di sapere qualcosa in più.

`````

## Con media e larghezza: la disuguaglianza di Čebyšëv

Il salto che rende utile la famiglia consiste nell'applicare la regola di
Markov non alla quantità stessa, ma al **quadrato della sua distanza dalla
media**. È una mossa da guardare da vicino, perché ritorna identica in mezza
statistica.

`````{tab} Elementare

Lo scarto dalla media può essere positivo o negativo, quindi la regola di
prima non gli si applica. Elevandolo al quadrato però diventa sempre positivo,
e in più i valori lontani pesano molto di più dei vicini, che è esattamente
quello che serve per parlare di code. E il valore medio di quello scarto al
quadrato è una grandezza che si conosce già: la varianza, cioè la larghezza al
quadrato.

Fatto il conto viene fuori una regola che si ricorda a memoria. **Qualunque sia
la forma dei dati, almeno tre quarti di essi stanno entro due larghezze dalla
media, e almeno otto su nove entro tre.** Non si è chiesto niente:
niente campana, niente simmetria, niente code sottili. Vale per i redditi, per
i tempi di risposta di un server, per gli errori di un modello.

Il prezzo di non chiedere niente si vede confrontando con il caso in cui invece
si chiede molto. Se i dati seguono davvero la curva a campana, entro due
larghezze non c'è il settantacinque per cento ma il novantacinque e mezzo, ed
entro tre il novantanove e sette. La differenza fra quei numeri è tutta lì: è
quanto vale
sapere la forma della distribuzione. Chi non la sa deve accontentarsi della
garanzia debole, e la garanzia debole, sui dati veri, quasi sempre lascia molto
sul tavolo.

C'è però una situazione in cui la garanzia debole è esattamente quello che
succede, e serve a capire che non si può fare di meglio a scatola chiusa. Se un
ottavo dei dati vale meno due, un ottavo vale più due e i restanti tre quarti
valgono zero, la media è zero, la larghezza è uno, e fuori da due larghezze c'è
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
nel primo caso, quarantuno nel secondo. La perdita misura esattamente quanto
vale l'ipotesi di normalità.

Anche questa disuguaglianza è stretta, e il testimone è una distribuzione a tre
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

La sezione su probabilità e statistica enunciava questo risultato e ne dava la
velocità; questa è la dimostrazione della forma debole, e mostra che serve
pochissimo per averla: la sola esistenza della varianza. Si vede anche perché
l'errore scala come $1/\sqrt{n}$: perché $\varepsilon$ compare al quadrato
accanto a $n$.

`````

## Quando i valori stanno in un intervallo: Hoeffding

La garanzia di Čebyšëv cala come $1/n$, che è lento. Sapendo una cosa in più,
molto facile da verificare, si guadagna un ordine di grandezza.

`````{tab} Elementare

Le risposte di un modello a un quiz sono giuste o sbagliate: uno oppure zero,
niente in mezzo, e soprattutto niente che possa schizzare a mille. Questa
informazione in più (i valori stanno dentro un intervallo noto) cambia
completamente le garanzie, perché toglie di mezzo lo scenario che rovina tutto,
quello in cui un solo caso mostruoso sposta la media da solo.

Con i valori limitati, la probabilità di sbagliare di tanto non cala in
proporzione al numero di prove: **crolla**, come cala una potenza quando si
alza l'esponente. Il risultato pratico si legge sulla domanda che ogni persona
si fa prima di preparare un banco di prova: quante domande servono per
misurare l'accuratezza a meno di due punti percentuali, sbagliando al massimo
una volta su venti?

- non sapendo niente della forma, la sola larghezza chiede dodicimilacinquecento
  domande;
- sapendo che le risposte valgono zero oppure uno, ne bastano
  quattromilaseicentododici;
- dando per buona la curva a campana, che qui è un'approssimazione ragionevole
  ma resta un'approssimazione, duemilaquattrocentouno.

Tre numeri per la stessa domanda, e la differenza fra loro è tutta nelle
ipotesi che si è disposti a fare. Il secondo è quello onesto nella maggior
parte dei casi: non chiede niente che non si possa controllare, e costa meno
della metà del primo.

Va notato che nessuno dei tre dipende da quanto è grande l'insieme da cui le
domande vengono estratte. Per misurare l'accuratezza di un modello su un
miliardo di casi possibili servono le stesse quattromilaseicento domande che
servirebbero se i casi fossero centomila, purché siano sorteggiate davvero a
caso. È uno dei fatti più controintuitivi della statistica, e uno dei più
utili.

`````

`````{tab} Superiore

Siano $X_1,\dots,X_n$ indipendenti con $X_i\in[a,b]$ e media comune $\mu$. La
**disuguaglianza di Hoeffding** afferma

$$
\Pr\big[\,|\bar{X}_n - \mu| \ge \varepsilon\,\big]
\;\le\; 2\exp\!\left(-\frac{2n\varepsilon^2}{(b-a)^2}\right) .
$$

Il confronto con Čebyšëv è netto: $1/(n\varepsilon^2)$ contro
$e^{-2n\varepsilon^2}$, cioè decadimento polinomiale contro esponenziale. Il
prezzo è l'ipotesi di limitatezza, che per una metrica in $[0,1]$ (accuratezza,
precision, recall) è gratis.

Invertendo per $n$ si ottiene la formula che serve davvero, cioè quanti esempi
mettere in un insieme di prova:

$$
n \;\ge\; \frac{(b-a)^2\,\log(2/\delta)}{2\varepsilon^2} ,
$$

con $\varepsilon$ la tolleranza e $\delta$ la probabilità di sbagliare. Per
$[a,b]=[0,1]$, $\varepsilon = 0{,}02$ e $\delta = 0{,}05$ si ottiene
$n\ge 4612$. La stessa domanda posta a Čebyšëv (con $\sigma^2\le 1/4$, il
massimo per una variabile in $[0,1]$) dà $n\ge 12\,500$; l'approssimazione
normale, $n\ge (1{,}96)^2\cdot 0{,}25/\varepsilon^2 = 2401$.

Due osservazioni che cambiano il modo di progettare una valutazione. La prima:
$n$ non dipende dalla **cardinalità della popolazione**, solo dalla precisione
voluta. La seconda: $\varepsilon$ compare al quadrato, quindi dimezzare la
tolleranza quadruplica il costo, ed è la stessa tassa $1/\sqrt{n}$ vista
altrove, qui in forma esplicita.

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
```

## Il prezzo di guardare molte volte

Tutte le garanzie viste finora valgono per una misura sola, decisa in anticipo.
Il modo in cui si lavora davvero, però, consiste nel provare molte varianti e
tenere la migliore, e in quel gesto la garanzia si rompe.

`````{tab} Elementare

Cento varianti di un modello, tutte identiche nella sostanza, tutte con la
stessa accuratezza vera dell'ottanta per cento. Si misurano tutte sullo stesso
insieme di prova da mille esempi e si tiene la migliore.

La migliore risulterà sopra l'ottanta per cento, e non di poco: ripetendo
l'esperimento duecento volte la vincitrice sta tipicamente intorno a $83$, e in
una singola prova può arrivare a $84{,}2$, cioè oltre quattro punti sopra il
vero.
Nessuna delle cento è migliore delle altre, e la differenza è tutta fortuna:
ognuna ha ricevuto un insieme di domande che per lei girava un po' meglio o un
po' peggio, e scegliendo la più fortunata si è scelta esattamente la fortuna.

Il meccanismo si capisce contando le occasioni. Ogni singola variante ha una
probabilità piccola di sembrare molto meglio di quel che è; con cento varianti
quelle probabilità piccole si sommano, e la probabilità che **almeno una**
sembri molto meglio diventa grande. Più cose si guardano, più è probabile che
qualcuna sembri straordinaria per caso.

Le due cure sono note e si usano poco. La prima è tenere da parte un secondo
insieme di prova, mai guardato durante la selezione, su cui misurare soltanto
la vincitrice: quella misura torna onesta perché è una misura sola. La seconda
è alzare l'asticella man mano che le varianti crescono. Il conto c'è, nessuno
lo fa a occhio, e chiede meno di quanto sembri: da dieci varianti a mille
l'asticella si alza di meno del doppio.

`````

`````{tab} Superiore

Il difetto ha un nome ed è la **maledizione del vincitore** (*winner's
curse*),
e il suo controllo è la disuguaglianza dell'unione: per eventi qualsiasi,

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

che cresce come $\sqrt{\log k}$. La crescita è lenta, ed è la buona notizia
della faccenda, ma va pagata: con $n=1000$ e $\delta = 0{,}05$, passare da
dieci a mille varianti allarga la tolleranza di un terzo scarso.

È lo stesso conto che regge i limiti di generalizzazione della teoria
dell'apprendimento, dove $k$ è la cardinalità della classe di ipotesi: la
garanzia peggiora come la radice del logaritmo del numero di funzioni fra cui
si sceglie, ed è la ragione formale per cui una classe troppo ricca smette di
dare garanzie. Il caso in cui la classe è infinita si tratta sostituendo il
conteggio con una misura di complessità, e il {doc}`capitolo sul machine
learning </MachineLearning/overview>` lo affronta dal lato pratico parlando di
sovradattamento.

Una precisazione che evita un errore frequente: la correzione va applicata al
numero di confronti **effettivamente fatti**, non a quelli dichiarati. Chi
prova venti configurazioni, ne riporta una e chiama le altre esplorazione
preliminare ha comunque fatto venti confronti.

`````

```python
rng = np.random.default_rng(0)

# cento varianti tutte uguali nella sostanza, stesso insieme di prova
vera, n_prova, k = 0.80, 1000, 100
stime = rng.binomial(n_prova, vera, size=k) / n_prova

print(round(stime.mean(), 4), stime.max())      # -> 0.8009 0.842
print(round(np.sqrt(vera * (1 - vera) / n_prova), 5))   # scarto tipico -> 0.01265

# ripetendo l'esperimento, la vincitrice e' quasi sempre gonfiata
massimi = [(rng.binomial(n_prova, vera, size=k) / n_prova).max()
           for _ in range(200)]
print(round(float(np.median(massimi)), 4))      # -> 0.83

# di quanto va allargata la tolleranza, in funzione di quante varianti si
# sono provate: cresce come la radice del logaritmo, cioe' pianissimo
def tolleranza(k, n=1000, delta=0.05):
    return np.sqrt(np.log(2 * k / delta) / (2 * n))

for k in (1, 10, 100, 1000):
    print(k, round(float(tolleranza(k)), 4))
# -> 1 0.0429
# -> 10 0.0547
# -> 100 0.0644
# -> 1000 0.0728
```

Il numero $0{,}842$ meritava una riga in più: quattro punti percentuali di
guadagno apparente, prodotti interamente dal caso, sono più di quanto separi
molti modelli veri e molti annunci. Chi legge una classifica dovrebbe sempre
chiedersi quante varianti sono state provate prima di quella pubblicata.

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

Accanto a ogni numero di larghezze ci sono due colonne, quello che succede e
quello che Čebyšëv garantisce, e la garanzia tiene con larghissimo margine,
come quasi sempre. Il punto è che tiene **senza sapere** che i dati erano
esponenziali, e avrebbe tenuto anche se fossero stati fatti apposta per
metterla in difficoltà.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Sapendo solo la **media** di una quantità che non può essere negativa, la
  frazione di casi sopra una soglia non supera la media divisa per la soglia:
  in una sala con reddito medio trentamila euro, al più uno su dieci ne
  guadagna trecentomila.
- Sapendo anche la **larghezza**, almeno tre quarti dei dati stanno entro due
  larghezze dalla media, e almeno otto su nove entro tre. Vale qualunque forma
  abbiano i dati, e questa è tutta la sua forza; se la forma fosse una campana
  i numeri sarebbero il novantacinque e mezzo e il novantanove e sette, e
  quella differenza è quanto vale saperlo.
- Se in più i valori stanno **dentro un intervallo noto** (le risposte giuste o
  sbagliate valgono zero oppure uno) la garanzia diventa molto più forte, e si
  può calcolare quante prove servono: per misurare un'accuratezza a meno di due
  punti, sbagliando al massimo una volta su venti, ne bastano circa
  quattromilaseicento. E quel numero **non dipende** da quanto è grande
  l'insieme da cui le prove vengono estratte.
- Dimezzare la tolleranza **quadruplica** il numero di prove.
- Tutto questo vale per una misura sola. Provando cento varianti e tenendo la
  migliore, la vincitrice sembra migliore di quello che è anche quando sono
  tutte identiche: con cento varianti arriva a tre o quattro punti sopra il
  vero. Le cure sono un secondo insieme di prova mai guardato, oppure alzare
  l'asticella man mano che le varianti crescono, e da dieci varianti a mille
  si alza di meno del doppio.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- **Markov**: per $X\ge 0$, $\Pr[X\ge a]\le \mathbb{E}[X]/a$. Usa solo la
  media, è stretta, e la non negatività è essenziale.
- **Čebyšëv**: $\Pr[|X-\mu|\ge k\sigma]\le 1/k^2$, ottenuta applicando Markov
  a $(X-\mu)^2$. Applicata a $\bar{X}_n$, con
  $\mathrm{Var}(\bar{X}_n)=\sigma^2/n$, dimostra la legge dei grandi numeri in
  tre righe e spiega perché l'errore cala come $1/\sqrt{n}$.
- **Hoeffding**: per $X_i\in[a,b]$ indipendenti,
  $\Pr[|\bar{X}_n-\mu|\ge\varepsilon]\le 2\exp(-2n\varepsilon^2/(b-a)^2)$,
  cioè decadimento esponenziale invece che polinomiale. Invertita dà
  $n \ge (b-a)^2\log(2/\delta)/(2\varepsilon^2)$: per $\varepsilon=0{,}02$ e
  $\delta=0{,}05$ su $[0,1]$, $n\ge 4612$ contro i $12\,500$ di Čebyšëv e i
  $2401$ dell'approssimazione normale.
- **Disuguaglianza dell'unione**: valutando $k$ ipotesi sullo stesso insieme, la
  tolleranza diventa $\varepsilon(k)=\sqrt{\log(2k/\delta)/(2n)}$, che cresce
  come $\sqrt{\log k}$. È il *winner's curse*, ed è lo stesso conto che regge i
  limiti di generalizzazione. La correzione si applica ai confronti fatti, non
  a quelli dichiarati.
```
`````

Queste disuguaglianze hanno una qualità che le rende diverse dal resto del
capitolo: dicono qualcosa di vero su distribuzioni che non si conoscono. Il
prossimo passo va nella direzione opposta, e chiede di conoscere il meccanismo
molto bene: quando lo stato di un sistema dipende soltanto da dove si trovava
un istante prima, l'algebra lineare e la probabilità si incontrano, e la
domanda «dove finirà, andando avanti per sempre?» ha una risposta esatta.
