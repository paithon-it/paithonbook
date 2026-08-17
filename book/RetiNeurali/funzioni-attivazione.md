# Funzioni di attivazione

Prendi una rete profonda: dieci strati, migliaia di neuroni, milioni di
parametri. Ora togli le funzioni di attivazione. Tutta quella profondità si
sgonfia in un istante: quello che resta, per quanto grande sia, è una sola
moltiplicazione, cioè un modello che sa disegnare soltanto righe dritte. Le
funzioni di attivazione sono il piccolo gesto non
lineare che, ripetuto strato dopo strato, trasforma una pila di
moltiplicazioni in un modello capace di riconoscere un volto o tradurre una
frase. Sono l'anima non lineare della rete.

## Perché serve una non linearità

Ogni strato di una rete fa una cosa sola: moltiplica ciascun numero che riceve
per il proprio **peso** (l'importanza che gli assegna), somma i risultati e
aggiunge un numero fisso suo, il **bias**. È un'operazione *lineare*, e il
problema è che comporre due operazioni lineari dà ancora un'operazione lineare:
mille non cambierebbero nulla.

`````{tab} Elementare

Immagina una catena di macchinette, ognuna delle quali "moltiplica per un
numero". La prima moltiplica per $2$, la seconda per $3$. Metterle in fila non
crea niente di nuovo: equivale a una sola macchinetta che moltiplica per $6$.
Puoi impilarne quante vuoi, alla fine resta *una regola proporzionale*.

Una rete fatta solo di strati così, per quanto profonda, non è più potente del
singolo neurone della sezione precedente: per dividere i casi in due gruppi sa
tracciare una riga dritta e nient'altro. Non imparerà mai una spirale, una
lettera scritta a mano, il tono di una frase. Serve, tra uno strato e l'altro,
una "piega": una funzione che *storce* i numeri in modo non proporzionale. È
lei che dà alla rete la libertà di disegnare curve. Non una qualunque, però:
se la piega è a sua volta fatta di sole moltiplicazioni e somme, come elevare
al quadrato, mettere mille neuroni in uno strato non fa disegnare niente di
più che metterne dieci. Le pieghe che incontreremo qui sotto non hanno quel
difetto.

`````

`````{tab} Superiore

In simboli, ogni strato calcola $\mathbf{W}\mathbf{x}+\mathbf{b}$: una
moltiplicazione per una matrice di pesi, più un vettore di bias. Consideriamo
due strati lineari in cascata, senza attivazione:

$$
\mathbf{h} = \mathbf{W}^{[1]}\mathbf{x}+\mathbf{b}^{[1]},
\qquad
\hat{\mathbf{y}} = \mathbf{W}^{[2]}\mathbf{h}+\mathbf{b}^{[2]} .
$$

Sostituendo il primo nel secondo:

$$
\hat{\mathbf{y}} = \mathbf{W}^{[2]}\big(\mathbf{W}^{[1]}\mathbf{x}+\mathbf{b}^{[1]}\big)+\mathbf{b}^{[2]}
= \underbrace{\left(\mathbf{W}^{[2]} \mathbf{W}^{[1]}\right)}_{\mathbf{W}'}\,\mathbf{x}
+ \underbrace{\left(\mathbf{W}^{[2]}\mathbf{b}^{[1]}+\mathbf{b}^{[2]}\right)}_{\mathbf{b}'} .
$$

La composizione collassa in un unico strato lineare con pesi $\mathbf{W}'$ e
bias $\mathbf{b}'$: la profondità è illusoria. Introducendo una non linearità
$g$ tra gli strati,
$\hat{\mathbf{y}} = \mathbf{W}^{[2]}\,g(\mathbf{W}^{[1]}\mathbf{x}+\mathbf{b}^{[1]})+\mathbf{b}^{[2]}$,
la fattorizzazione salta.

Non basta però che $g$ sia non lineare, ed è un punto su cui si scivola spesso.
Se $g$ fosse un polinomio, per esempio $g(x)=x^2$, uno strato nascosto
calcolerebbe $\sum_i c_i\,(w_i x + b_i)^2 + d$, dove $w_i$ e $b_i$ sono peso e
bias dell'$i$-esimo neurone nascosto, $c_i$ il peso con cui l'uscita lo
raccoglie e $d$ il bias d'uscita: comunque si scelgano quei parametri resta un
polinomio di grado $2$, e aggiungere neuroni non servirebbe a
niente. Provato ai minimi quadrati su $x^3$ in $[-1,1]$ (duemila punti
equispaziati, uno strato nascosto, Adam per quattromila passi con
$\eta = 10^{-2}$), dieci neuroni e ottocento danno lo **stesso** errore,
$0{,}1514$ tutti e due, e quell'errore si sa già quanto vale senza addestrare
niente. La miglior approssimazione di $x^3$ con un polinomio di grado
al più $2$, in media quadratica su $[-1,1]$, è $\tfrac{3}{5}x$ (è la proiezione
ortogonale, e si legge nella scrittura di $x^3$ come combinazione di polinomi di
Legendre); lo scarto che resta ha radice

$$
\sqrt{\frac{1}{2}\int_{-1}^{1}\left(x^3 - \tfrac{3}{5}x\right)^2 dx}
= \sqrt{\frac{4}{175}} \simeq 0{,}1512 ,
$$

ed è esattamente il muro contro cui la larghezza si ferma. Con la ReLU, invece,
la larghezza compra davvero qualcosa: con lo stesso addestramento dieci neuroni
scendono a $3{,}0\cdot 10^{-2}$ e ottocento a $1{,}3\cdot 10^{-3}$, cioè oltre
cento volte sotto quel muro.

La condizione esatta è che $g$ **non sia un polinomio** (per la classe di
funzioni in cui il risultato è enunciato: attivazioni continue a tratti e
localmente limitate), e sotto quella
condizione la rete è un **approssimatore universale**: con abbastanza neuroni
avvicina, con errore arbitrariamente piccolo, qualunque funzione continua su un
insieme compatto ({cite}`cybenko1989approximation` per le sigmoidali;
{cite}`leshno1993multilayer` nella forma generale, ReLU compresa). Resta un
teorema di esistenza, e per giunta muto sulla larghezza necessaria, che per una
funzione qualunque di $d$ variabili cresce esponenzialmente in $d$.

`````

Prima di guardarle una per una serve un anticipo su come una rete impara. È il
tema della sezione dopo, ma qui serve subito, perché è il metro con cui si
giudica una funzione di attivazione.

Per correggersi, una rete deve sapere in che direzione muovere ciascun peso, e
lo scopre chiedendosi: *se muovessi questo peso di pochissimo, di quanto
cambierebbe il risultato?* Quel «di quanto cambierebbe» si chiama **pendenza**,
ed è la stessa pendenza di una strada in salita: quanto sali per ogni passo che
fai in avanti. Dove è ripida, un passo cambia molto; dove è pianeggiante, un
passo non cambia niente.

Adesso il punto che riguarda noi. Quella domanda non se la pone un peso alla
volta e in un posto solo: parte dall'uscita della rete, dove l'errore si vede,
e risale gli strati all'indietro, uno dopo l'altro, fino ai primi. È il
meccanismo della sezione dopo, e qui basta sapere che il messaggio viaggia
all'indietro. Ogni volta che attraversa una funzione di attivazione, quel
messaggio viene moltiplicato per la pendenza **di quella funzione**, presa nel
punto in cui il neurone stava lavorando. Se lì la funzione è ripida, il
messaggio passa; se lì la funzione è **piatta**, la sua pendenza vale quasi
zero, e moltiplicare per quasi zero spegne il messaggio.

È tutto qui il metro, ed è quello che applicheremo tre volte: una buona
funzione di attivazione è una che non spegne il messaggio mentre lo lascia
passare.

Le protagoniste degli **strati nascosti** (quelli in mezzo, che non si
affacciano né sull'ingresso né sull'uscita) sono tre, ognuna con un carattere diverso
({numref}`fig-attivazioni`); alla fine della sezione se ne aggiunge una quarta,
la softmax, che fa un altro mestiere e lavora solo sull'ultimo strato.

```{figure} ../figures/attivazioni-sigmoide-tanh-relu.svg
:name: fig-attivazioni
:alt: "Tre grafici affiancati: la sigmoide come curva a S tra 0 e 1, la tanh come curva a S centrata nello zero tra -1 e 1, la ReLU piatta a zero per x negative e lineare per x positive."
:width: 95%

Le tre funzioni di attivazione classiche, in tre grafici affiancati. In
orizzontale il numero che entra nella funzione, in verticale quello che ne
esce; l'incrocio degli assi è lo zero in entrambe le direzioni. Da guardare
soprattutto dove ciascuna curva è **piatta**: è lì che il messaggio che risale
la rete si spegne.
```

## La sigmoide: il primo interruttore morbido

La prima scelta, storicamente: schiaccia qualunque numero in un valore fra $0$
e $1$. Comoda, perché un numero fra zero e uno si legge come un interruttore
acceso a metà, o come «quanto sono convinto». Viene dalla regressione
logistica, il classificatore incontrato nel capitolo di machine learning.

`````{tab} Elementare

La sigmoide prende un numero qualsiasi e lo comprime in un valore tra $0$ e $1$.
Numeri molto negativi diventano quasi $0$, numeri molto positivi quasi $1$, e
lo zero finisce esattamente a metà, $0{,}5$. È un interruttore che invece di
scattare di colpo scivola dolcemente da spento ad acceso. Ecco qualche valore
(sono da calcolatrice: la formula ha dentro lo stesso ingrandimento che
incontreremo con la softmax):

| entra | $-6$ | $-5$ | $-2$ | $0$ | $1$ | $2$ | $5$ |
|---|---|---|---|---|---|---|---|
| esce | $0{,}0025$ | $0{,}0067$ | $0{,}12$ | $0{,}50$ | $0{,}73$ | $0{,}88$ | $0{,}993$ |

Il difetto salta all'occhio guardando la curva: agli estremi diventa
*piattissima*. E «piattissima» si può misurare, con i numeri della tabella.
Vicino allo zero, spostandosi di uno (da $0$ a $1$), l'uscita sale da $0{,}50$
a $0{,}73$: si è mossa di ventitré centesimi. In fondo alla coda, spostandosi
sempre di uno (da $-6$ a $-5$), sale da $0{,}0025$ a $0{,}0067$: si è mossa di
quattro millesimi, cioè più di cinquanta volte meno, per uno spostamento
identico.

E quei due numeri sono, in pratica, la pendenza: la pendenza vera è quanto si
muove l'uscita per un passo *piccolissimo*, e su un passo lungo uno viene fuori
un po’ meno (vicino allo zero la pendenza vera è $0{,}25$, non $0{,}23$). Nelle
code il divario resta, ma i numeri in gioco sono tutti dello stesso ordine: la
pendenza vera vale meno di tre millesimi in $-6$ e quasi sette in $-5$, e i
quattro millesimi letti sulla tabella stanno in mezzo. Il messaggio che risale
la rete viene moltiplicato per un numero così, e si spegne.

`````

`````{tab} Superiore

La sigmoide logistica è

$$
\sigma(x) = \frac{1}{1+e^{-x}} \in (0,1),
\qquad
\sigma'(x) = \sigma(x)\,\big(1-\sigma(x)\big).
$$

La derivata è massima nell'origine, dove vale solo $0{,}25$, e tende a $0$ per
$|x|\to\infty$ (le code sature). Nella *backpropagation* il gradiente che
attraversa uno strato viene moltiplicato per $\sigma'$ *e* per la matrice dei
pesi: con pesi di norma moderata, quella delle inizializzazioni standard, il
fattore complessivo per strato resta sotto $1$ e il prodotto collassa
esponenzialmente con la profondità. È il celebre problema del **gradiente che
svanisce** (*vanishing gradient*), studiato da Hochreiter
{cite}`hochreiter1991untersuchungen` e Bengio {cite}`bengio1994learning`: nelle
reti profonde gli strati vicini all'ingresso smettono di ricevere segnale e non
apprendono. A ciò si aggiunge che l'uscita non è centrata nello zero (sempre
positiva), il che rallenta la convergenza della discesa del gradiente.

La via d'uscita che viene in mente per prima non funziona, e vale la pena
chiudere la porta: non si rimedia alzando i pesi per compensare il fattore
$1/4$. Pesi più grandi spingono $z$ nelle code, dove $\sigma'$ è ancora più
piccola, e i due effetti si mangiano a vicenda. Misurato così: venti strati da
$128$ unità, pesi estratti con l'inizializzazione di Glorot, ingressi normali
standard, e per «fattore» si intende il rapporto fra la norma del gradiente che
esce da uno strato verso l'ingresso e quella del gradiente che vi è entrato
dall'uscita, mediato sugli strati. Con quel protocollo il fattore medio per
strato viene $0{,}24$, in linea con il $0{,}25$ teorico; quadruplicando la
scala dei pesi sale soltanto a $0{,}6$, perché nel frattempo
$\mathbb{E}[\sigma'(z)]$ scende da $0{,}23$ a $0{,}13$. Si guadagna sul modulo
di $\mathbf{W}$ e si perde sulla saturazione, e il fattore resta sotto $1$
comunque: la sigmoide perde da entrambi i lati.

`````

## La tanh: la stessa S, ma centrata nello zero

La `tanh` (per esteso «tangente iperbolica», ma il nome lungo qui non serve a
niente) ha la stessa forma a S della sigmoide e corregge uno dei suoi difetti:
sta a cavallo dello zero invece che tutta sopra.

`````{tab} Elementare

La `tanh` schiaccia i numeri tra $-1$ e $+1$, con lo zero che resta zero. La
differenza con la sigmoide è che ora l'uscita può essere anche negativa: in
media i valori si bilanciano attorno allo zero, e questo aiuta la rete a
imparare un po’ più in fretta. Il motivo, in breve: con la sigmoide tutte le
uscite sono positive, e allora le correzioni dei pesi di uno stesso neurone
tendono ad andare tutte nella stessa direzione insieme, il che fa zigzagare la
discesa invece di farla andare dritta. Con lo zero al centro le uscite si
bilanciano, e la strada si raddrizza. Resta però lo stesso tallone d'Achille: agli
estremi la curva si appiattisce e il messaggio che risale la rete svanisce di
nuovo.

`````

`````{tab} Superiore

$$
\tanh(x) = \frac{e^{x}-e^{-x}}{e^{x}+e^{-x}} = 2\,\sigma(2x)-1 \in (-1,1),
\qquad
\tanh'(x) = 1-\tanh^2(x).
$$

L'uscita è **centrata nello zero**, quindi i gradienti dei pesi non hanno un
segno sistematico: la convergenza è più regolare che con la sigmoide
{cite}`lecun1998efficient`. La derivata arriva fino a $1$ nell'origine, contro
il $0{,}25$ della sigmoide, ma satura comunque agli estremi. Per anni la `tanh`
è stata lo standard negli strati nascosti e sopravvive tuttora nelle celle
ricorrenti LSTM e GRU.

`````

## ReLU: la semplicità che ha sbloccato il deep learning

Nel 2010–2012 una funzione quasi imbarazzante nella sua banalità cambia le
regole del gioco: se il numero è positivo lo lascia passare, altrimenti lo mette
a zero. Nessun conto complicato e, dal lato positivo, nessuna zona piatta.

`````{tab} Elementare

La ReLU (*Rectified Linear Unit*) fa una cosa sola: se l'ingresso è positivo lo
restituisce identico, se è negativo o zero restituisce zero. È uno sportello
che lascia passare i versamenti e blocca i prelievi (è l'esempio con cui si
apre il libro): entra $10$, esce $10$; entra $-3$, esce $0$.

Perché ha sbloccato le reti profonde? Perché dal lato positivo la curva è una
riga inclinata: la sua pendenza è sempre $1$, non si appiattisce mai. Si misura
come prima: da $2$ a $3$ l'uscita passa da $2$ a $3$, si è mossa di uno intero;
da $20$ a $21$ passa da $20$ a $21$, ancora uno intero. Lontano dallo zero
quanto vicino, la pendenza vale sempre $1$. Il
messaggio che risale la rete, moltiplicato per $1$, resta quello di prima; non
si smorza a ogni passaggio come faceva con la sigmoide, e arriva quindi fino ai
primi strati anche in una rete che ne ha decine. Ed è velocissima da calcolare:
un confronto con lo zero.

C'è un rischio, e conviene capirne il motivo perché a prima vista sembra una
maledizione. Se un neurone finisce nella zona negativa per **tutti** gli
esempi (cioè per tutti i dati con cui la rete viene addestrata), la sua uscita
è sempre zero, e allora anche la pendenza che sente è
sempre zero: nessuna indicazione, nessuna correzione, i suoi pesi restano
fermi. È il neurone "morto". Non è del tutto senza ritorno, perché i neuroni
che stanno **prima** di lui continuano a cambiare e possono cominciare a
mandargli numeri diversi; ma da solo non si tira fuori. La **Leaky ReLU**
previene il problema lasciando filtrare una piccola pendenza anche per i valori
negativi, così un po’ di indicazione arriva sempre.

`````

`````{tab} Superiore

$$
\mathrm{ReLU}(x) = \max(0,x),
\qquad
\mathrm{ReLU}'(x) = \begin{cases} 1 & x>0,\\ 0 & x<0.\end{cases}
$$

I due casi non coprono tutta la retta, e l'omissione è voluta: in $x=0$ la
derivata **non esiste**, perché il rapporto incrementale vale $0$ arrivando da
sinistra e $1$ arrivando da destra. Le librerie ne scelgono una per convenzione
(PyTorch restituisce $0$; per `leaky_relu` restituisce $\alpha$), ed è una scelta
innocua: i punti in cui la pre-attivazione è **esattamente** zero sono un
insieme trascurabile, e qualunque valore fra $0$ e $1$ è un sotto-gradiente
legittimo. C'è però una conseguenza pratica che vale un pomeriggio a chi
controlla i conti a mano: verificando il gradiente con le **differenze finite**
proprio in zero si trova $0{,}5$, cioè la media dei due lati, mentre autograd
dà $0$. I due numeri non coincidono e nessuno dei due è sbagliato: è il punto in
cui la derivata non c'è, non un errore nel codice.

Per $x>0$ il gradiente è esattamente $1$: niente saturazione, niente *vanishing*
lungo i cammini attivi. Ciò ha reso addestrabili reti molto profonde
({cite}`nair2010rectified`; {cite}`glorot2011deep`; AlexNet,
{cite}`krizhevsky2012imagenet`) e
induce attivazioni **sparse** (molti neuroni esattamente a zero). Il rovescio è
il *dying ReLU*: un neurone la cui **pre-attivazione** $z$ resta negativa su
tutti i dati ha gradiente esattamente nullo sui propri pesi e smette di
aggiornarsi. Attenzione a leggerlo bene: la condizione è su $z$, non
sull'ingresso $\mathbf{x}$ (a valle di uno strato ReLU gli ingressi sono
$\ge 0$ per costruzione). E «non si aggiorna più» vale per i **suoi**
parametri, non per il suo destino: in uno strato nascosto $z$ continua a
muoversi perché cambiano gli strati a monte, e il neurone può risvegliarsi
senza che nessuno dei suoi pesi si sia mosso. Solo nel primo strato, dove
l'ingresso è il dato e non cambia, la morte è definitiva. La **Leaky ReLU**
introduce una pendenza $\alpha$ piccola (tipicamente $0{,}01$) sul ramo
negativo:

$$
\mathrm{LeakyReLU}(x) = \max(\alpha x,\, x),\qquad \alpha \ll 1 .
$$

Sulla stessa idea nascono PReLU (con $\alpha$ appreso), ELU e, nei Transformer
moderni, la **GELU** {cite}`hendrycks2016gaussian`, cioè $x\,\Phi(x)$: una ReLU
ammorbidita in cui il gradino secco è sostituito da $\Phi$, la **funzione di
ripartizione** della normale standard. Da non confondere con la densità, che
qui chiamiamo $\Phi'$ per non tirare in ballo la $\varphi$ (nel resto del
capitolo $\varphi$ è l'attivazione dello strato d'uscita, e un simbolo con
due mestieri è un errore che aspetta): $x\,\Phi'(x)$ è tutt'altra funzione,
dispari e non monotona, e chi prova a rifarsi il grafico partendo dalla campana
ottiene un disegno diverso.

`````

## Softmax: dalle uscite alle probabilità

Le funzioni viste finora lavorano su un numero alla volta, negli strati
nascosti. Sull'ultimo strato serve un'altra cosa, quando la rete deve scegliere
fra più risposte possibili: qualcosa che guardi *tutte* le uscite insieme e le
trasformi in percentuali che sommano a $100$. È la **softmax**.

`````{tab} Elementare

Immagina che la rete debba decidere tra "gatto", "cane" e "volpe" e produca
tre punteggi grezzi, per esempio $2{,}0$, $1{,}0$, $0{,}1$. La softmax li
converte in tre percentuali che sommano a $100\%$ (qui $66\%$, $24\%$, $10\%$)
esaltando il punteggio più alto ma senza mai azzerare del tutto gli altri. Il
risultato si legge come "quanto la rete è convinta di ciascuna classe".

Non è una semplice divisione, e se provi a farla ti accorgi che i conti non
tornano: $2$ diviso $3{,}1$ farebbe $64{,}5\%$, non $66\%$. Il passaggio in più
è che prima ogni punteggio viene *ingrandito*. Si prende un numero fisso, che
vale $2{,}718\ldots$ e in matematica si chiama $e$, e lo si eleva al punteggio.
Elevarlo a $2$ vuol dire $e \times e$; elevarlo a $0{,}1$ a mente non si fa, ma
la calcolatrice sì, con il tasto `exp`, che riempie anche i buchi fra un
esponente intero e il successivo. Ecco i risultati: $2{,}0$ diventa $7{,}39$,
$1{,}0$ diventa $2{,}72$ (cioè $e$ stesso) e $0{,}1$ diventa $1{,}11$. Sommano
$11{,}21$, e adesso sì che si divide: $7{,}39 / 11{,}21 = 66\%$. È quel primo
ingrandimento a esaltare il punteggio più alto, ed è anche il motivo per cui
nessuna percentuale arriva mai a zero tondo.

Perché proprio $e$ e non, che so, $10$? Con $10$ funzionerebbe lo stesso,
verrebbero solo percentuali più sbilanciate verso il punteggio più alto. La
ragione della scelta è che con $e$ le pendenze, che sono il pane della sezione
successiva, vengono semplicissime.

`````

`````{tab} Superiore

Dato il vettore dei punteggi grezzi dell'ultimo strato, che si chiamano
*logit* e qui indichiamo con $\mathbf{z}\in\mathbb{R}^K$ (dove $K$ è il numero
di classi), la softmax è

$$
\mathrm{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}},
\qquad \sum_{i=1}^{K}\mathrm{softmax}(\mathbf{z})_i = 1 .
$$

È la generalizzazione multiclasse della sigmoide e si accompagna alla loss di
**cross-entropia**. Attenzione: l'esponenziale di logit grandi va facilmente
in overflow. La soluzione standard è sottrarre il massimo,
$z_i \leftarrow z_i - \max_j z_j$, che non cambia il risultato ma lo rende
numericamente stabile: il trucco del *log-sum-exp*, discusso nella sezione di
[analisi numerica](../Matematica/analisi-numerica.md).

`````

## Quale usare, in pratica

Una guida ragionevole per la maggior parte dei casi:

| Dove | Scelta consigliata | Perché |
|---|---|---|
| Strati nascosti (scelta di partenza) | **ReLU** | veloce, il messaggio non si spegne, ottimo punto di partenza |
| Strati nascosti, neuroni "morti" | **Leaky ReLU** | un po’ di pendenza anche sui negativi |
| Celle ricorrenti (le LSTM e le GRU della sezione sui modelli di sequenza, nel capitolo sul linguaggio naturale) | **tanh** + sigmoide | uscita centrata; la sigmoide fa da rubinetto, perché moltiplicare per un numero fra $0$ e $1$ è decidere quanta informazione lasciar passare |
| Uscita, scelta fra due risposte (classificazione binaria) | **sigmoide** | una probabilità, mai esattamente $0$ né $1$ |
| Uscita, scelta fra più risposte (classificazione multiclasse) | **softmax** | una percentuale per ciascuna delle classi in gioco |
| Uscita, previsione di un numero (regressione) | **nessuna** (lineare) | il valore può essere qualunque numero |

La regola pratica di oggi: negli strati nascosti parti da ReLU, cambia solo se
i risultati non convincono. Sull'uscita, invece, la funzione la detta il
problema, non il gusto.

## In pratica, con NumPy

Ogni funzione è una o due righe. La softmax ne chiede due in più, e sono due
precauzioni che vale la pena conoscere. La prima: si sottrae il punteggio più
alto prima di fare gli ingrandimenti, così i numeri restano piccoli e il
computer non va fuori scala (il risultato non cambia). La seconda: si dichiara
su quali numeri fare la somma. Senza, dando alla funzione un gruppo di esempi
tutti insieme, le percentuali sommerebbero a $100$ sull'intero gruppo invece
che su ciascun esempio, e sarebbe un risultato sbagliato che non dà nessun
errore.

```python
import numpy as np

def sigmoide(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)                      # già in NumPy

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)   # pendenza alpha sui negativi

def softmax(z):
    z = z - np.max(z, axis=-1, keepdims=True)   # stabilità numerica
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)    # una riga per esempio

print(softmax(np.array([2.0, 1.0, 0.1])))  # -> [0.65900114 0.24243297 0.09856589]
```

In PyTorch (il framework che incontreremo nel prossimo capitolo) non serve
implementarle a mano: esistono come funzioni (`torch.relu`, `torch.tanh`,
`torch.sigmoid`, `torch.softmax`) o come moduli da impilare tra gli strati
(`nn.ReLU()`, `nn.Sigmoid()`), e sono già scritte nella forma numericamente
stabile.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Fra uno strato e l'altro ci vuole una **piega**: senza, mettere in fila dieci
  strati equivale a metterne uno, e tutta la profondità non serve a niente.
- **Sigmoide** e **tanh** schiacciano i numeri fra due estremi, e proprio agli
  estremi diventano piatte: lì la rete non trova più nessuna pendenza da
  seguire e smette di imparare. La tanh è un po’ meglio perché è centrata sullo
  zero.
- La **ReLU** ("se è positivo lascialo passare, altrimenti zero") dal lato
  positivo non si appiattisce mai, e costa un confronto: è la scelta di
  partenza, ed è ciò che ha reso possibili le reti profonde. La **Leaky ReLU**
  cura i neuroni che restano bloccati a zero.
- Sull'ultimo strato di un classificatore c'è la **softmax**, che trasforma i
  punteggi grezzi in percentuali che sommano a $100$.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Senza una **non linearità** tra gli strati, una rete profonda collassa in un
  singolo strato lineare: la profondità sarebbe inutile. E non basta che sia
  non lineare: deve essere **non polinomiale**, altrimenti la larghezza non
  compra niente.
- **Sigmoide** e **tanh** saturano agli estremi e soffrono il *vanishing
  gradient*; la tanh almeno è centrata nello zero. Alzare i pesi per
  compensare non aiuta: sposta il problema dal modulo di $\mathbf{W}$ alla
  saturazione.
- La **ReLU** ($\max(0,x)$) non satura dal lato positivo ed è velocissima: è ciò
  che ha reso addestrabili le reti profonde. La **Leaky ReLU** cura i neuroni
  "morti", cioè quelli con pre-attivazione negativa su tutto il dataset.
- La **softmax** trasforma i logit dell'ultimo strato in probabilità che sommano
  a $1$; va calcolata nella forma numericamente stabile.
```

`````
