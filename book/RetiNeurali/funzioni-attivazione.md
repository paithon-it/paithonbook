# Funzioni di attivazione

Prendi una rete profonda: dieci strati, migliaia di neuroni, milioni di
parametri. Ora togli le funzioni di attivazione. Tutta quella profondità si
sgonfia in un istante: quello che resta è, matematicamente, una banale
regressione lineare. Le funzioni di attivazione sono il piccolo gesto non
lineare che, ripetuto strato dopo strato, trasforma una pila di
moltiplicazioni in un modello capace di riconoscere un volto o tradurre una
frase. Sono l'anima non lineare della rete.

## Perché serve una non linearità

Ogni strato di una rete calcola $W\mathbf{x}+\mathbf{b}$: una moltiplicazione
per una matrice di pesi, più un bias. È un'operazione *lineare*. Il problema è
che comporre due operazioni lineari dà ancora un'operazione lineare, e mille
non cambierebbero nulla.

`````{tab} Elementare

Immagina una catena di macchinette, ognuna delle quali "moltiplica per un
numero". La prima moltiplica per $2$, la seconda per $3$. Metterle in fila non
crea niente di nuovo: equivale a una sola macchinetta che moltiplica per $6$.
Puoi impilarne quante vuoi, alla fine resta *una regola proporzionale*.

Una rete fatta solo di strati così, per quanto profonda, non è più potente di
una retta: sa tracciare solo confini dritti. Non imparerà mai una spirale, una
lettera scritta a mano, il tono di una frase. Serve, tra uno strato e l'altro,
una "piega": una funzione che *storce* i numeri in modo non proporzionale. È
lei che dà alla rete la libertà di disegnare curve.

`````

`````{tab} Superiore

Consideriamo due strati lineari in cascata, senza attivazione:

$$
\mathbf{h} = W_1\mathbf{x}+\mathbf{b}_1,
\qquad
\hat{\mathbf{y}} = W_2\mathbf{h}+\mathbf{b}_2 .
$$

Sostituendo il primo nel secondo:

$$
\hat{\mathbf{y}} = W_2\big(W_1\mathbf{x}+\mathbf{b}_1\big)+\mathbf{b}_2
= \underbrace{(W_2 W_1)}_{W'}\,\mathbf{x} + \underbrace{(W_2\mathbf{b}_1+\mathbf{b}_2)}_{\mathbf{b}'} .
$$

La composizione collassa in un unico strato lineare con pesi $W'$ e bias
$\mathbf{b}'$: la profondità è illusoria. Introducendo una non linearità $g$
tra gli strati, $\hat{\mathbf{y}} = W_2\,g(W_1\mathbf{x}+\mathbf{b}_1)+\mathbf{b}_2$,
la fattorizzazione salta e la rete diventa un **approssimatore universale**
(Cybenko, 1989; Hornik, 1991): con abbastanza neuroni può avvicinare qualunque
funzione continua.

`````

Le protagoniste di questo capitolo sono tre, ognuna con un carattere diverso
({numref}`fig-attivazioni`).

```{figure} ../figures/attivazioni-sigmoide-tanh-relu.svg
:name: fig-attivazioni
:alt: "Tre grafici affiancati: la sigmoide come curva a S tra 0 e 1, la tanh come curva a S centrata nello zero tra -1 e 1, la ReLU piatta a zero per x negative e lineare per x positive."
:width: 95%

Le tre funzioni di attivazione classiche sullo stesso piano cartesiano.
```

## La sigmoide: il primo interruttore morbido

Storicamente la prima scelta, ereditata dalla regressione logistica: schiaccia
qualunque numero in un valore tra $0$ e $1$, interpretabile come una
probabilità o come un interruttore "acceso/spento" ammorbidito.

`````{tab} Elementare

La sigmoide prende un numero qualsiasi e lo comprime in un valore tra $0$ e $1$.
Numeri molto negativi diventano quasi $0$, numeri molto positivi quasi $1$, e
lo zero finisce esattamente a metà, $0{,}5$. È un interruttore che invece di
scattare di colpo scivola dolcemente da spento ad acceso.

Il difetto salta all'occhio guardando la curva: agli estremi diventa
*piattissima*. Lì un cambiamento grande dell'ingresso muove l'uscita di
pochissimo. Quando la rete impara guardando quanto la curva "pende", nelle zone
piatte non trova quasi nessuna pendenza da seguire: il segnale per correggersi
si assottiglia fino a sparire.

`````

`````{tab} Superiore

La sigmoide logistica è

$$
\sigma(x) = \frac{1}{1+e^{-x}} \in (0,1),
\qquad
\sigma'(x) = \sigma(x)\,\big(1-\sigma(x)\big).
$$

La derivata è massima nell'origine, dove vale solo $0{,}25$, e tende a $0$ per
$|x|\to\infty$ (le code sature). Nel *backpropagation* il gradiente attraversa
gli strati moltiplicandosi: catene di fattori $\le 0{,}25$ collassano
esponenzialmente. È il celebre problema del **gradiente che svanisce**
(*vanishing gradient*), studiato da Hochreiter (1991) e Bengio
{cite}`bengio1994learning`: nelle
reti profonde gli strati vicini all'ingresso smettono di ricevere segnale e non
apprendono. A ciò si aggiunge che l'uscita non è centrata nello zero (sempre
positiva), il che rallenta la convergenza della discesa del gradiente.

`````

## La tanh: la stessa S, ma centrata nello zero

La tangente iperbolica ha la stessa forma a S della sigmoide, ma corregge uno
dei suoi difetti: è simmetrica rispetto all'origine.

`````{tab} Elementare

La `tanh` schiaccia i numeri tra $-1$ e $+1$, con lo zero che resta zero. La
differenza con la sigmoide è che ora l'uscita può essere anche negativa: in
media i valori si bilanciano attorno allo zero, e questo aiuta la rete a
imparare un po' più in fretta. Resta però lo stesso tallone d'Achille: agli
estremi la curva si appiattisce e il segnale di correzione (il *gradiente*,
la pendenza che guida l'apprendimento) svanisce di nuovo.

`````

`````{tab} Superiore

$$
\tanh(x) = \frac{e^{x}-e^{-x}}{e^{x}+e^{-x}} = 2\,\sigma(2x)-1 \in (-1,1),
\qquad
\tanh'(x) = 1-\tanh^2(x).
$$

L'uscita è **centrata nello zero**, quindi i gradienti dei pesi non hanno un
segno sistematico: la convergenza è più regolare che con la sigmoide (LeCun,
*Efficient BackProp*, 1998). La derivata arriva fino a $1$ nell'origine, contro
il $0{,}25$ della sigmoide, ma satura comunque agli estremi. Per anni la `tanh`
è stata lo standard negli strati nascosti e sopravvive tuttora nelle celle
ricorrenti LSTM e GRU.

`````

## ReLU: la semplicità che ha sbloccato il deep learning

Nel 2010–2012 una funzione quasi imbarazzante nella sua banalità cambia le
regole del gioco: se il numero è positivo lo lascia passare, altrimenti lo mette
a zero. Nessuna esponenziale, nessuna saturazione dal lato positivo.

`````{tab} Elementare

La ReLU (*Rectified Linear Unit*) fa una cosa sola: se l'ingresso è positivo lo
restituisce identico, se è negativo o zero restituisce zero. È il filtro che
lascia passare solo i versamenti positivi dell'esempio con cui si apre il libro.

Perché ha sbloccato il deep learning? Perché dal lato positivo la curva è una
retta inclinata: la sua pendenza è sempre $1$, non si appiattisce mai. Il
segnale di apprendimento attraversa decine di strati senza svanire. Ed è
velocissima da calcolare: un confronto con lo zero. C'è un rischio: se un
neurone finisce sempre nella zona negativa resta "morto", bloccato a zero per
sempre. La **Leaky ReLU** lo evita lasciando filtrare una piccola pendenza anche
per i valori negativi.

`````

`````{tab} Superiore

$$
\mathrm{ReLU}(x) = \max(0,x),
\qquad
\mathrm{ReLU}'(x) = \begin{cases} 1 & x>0,\\ 0 & x<0.\end{cases}
$$

Per $x>0$ il gradiente è esattamente $1$: niente saturazione, niente *vanishing*
lungo i cammini attivi. Ciò ha reso addestrabili reti molto profonde
({cite}`nair2010rectified`; {cite}`glorot2011deep`; AlexNet,
{cite}`krizhevsky2012imagenet`) e
induce attivazioni **sparse** (molti neuroni esattamente a zero). Il rovescio è
il *dying ReLU*: un neurone con ingresso sempre negativo ha gradiente nullo e
non si aggiorna più. La **Leaky ReLU** introduce una pendenza $\alpha$ piccola
(tipicamente $0{,}01$) sul ramo negativo:

$$
\mathrm{LeakyReLU}(x) = \max(\alpha x,\, x),\qquad \alpha \ll 1 .
$$

Sulla stessa idea nascono PReLU (con $\alpha$ appreso), ELU e, nei Transformer
moderni, la **GELU**: una ReLU "ammorbidita" pesata dalla gaussiana.

`````

## Softmax: dalle uscite alle probabilità

Le funzioni viste finora agiscono su un singolo numero, negli strati nascosti.
Sull'ultimo strato di un classificatore multiclasse serve invece qualcosa che
guardi *tutte* le uscite insieme e le trasformi in una distribuzione di
probabilità: la **softmax**.

`````{tab} Elementare

Immagina che la rete debba decidere tra "gatto", "cane" e "volpe" e produca
tre punteggi grezzi, per esempio $2{,}0$, $1{,}0$, $0{,}1$. La softmax li
converte in tre percentuali che sommano a $100\%$ (qui $66\%$, $24\%$, $10\%$)
esaltando il punteggio più alto ma senza mai azzerare del tutto gli altri. Il
risultato si legge come "quanto la rete è convinta di ciascuna classe".

`````

`````{tab} Superiore

Dato il vettore di *logit* $\mathbf{z}\in\mathbb{R}^K$, la softmax è

$$
\mathrm{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}},
\qquad \sum_{i=1}^{K}\mathrm{softmax}(\mathbf{z})_i = 1 .
$$

È la generalizzazione multiclasse della sigmoide e si accompagna alla loss di
**cross-entropia**. Attenzione: l'esponenziale di logit grandi va facilmente
in overflow. La soluzione standard è sottrarre il massimo,
$z_i \leftarrow z_i - \max_j z_j$, che non cambia il risultato ma lo rende
numericamente stabile: il trucco del *log-sum-exp*, discusso nel capitolo di
[Analisi numerica](../Matematica/analisi-numerica.md).

`````

## Quale usare, in pratica

Una guida ragionevole per la maggior parte dei casi:

| Dove | Scelta consigliata | Perché |
|---|---|---|
| Strati nascosti (default) | **ReLU** | veloce, niente vanishing, ottimo punto di partenza |
| Strati nascosti, neuroni "morti" | **Leaky ReLU** / GELU | pendenza anche sui negativi |
| Celle ricorrenti (LSTM, GRU) | **tanh** + sigmoide | uscita centrata; sigmoide per i "gate" |
| Uscita, classificazione binaria | **sigmoide** | una probabilità in $[0,1]$ |
| Uscita, classificazione multiclasse | **softmax** | distribuzione su $K$ classi |
| Uscita, regressione | **nessuna** (lineare) | il valore può essere qualunque numero reale |

La regola pratica di oggi: negli strati nascosti parti da ReLU, cambia solo se
i risultati non convincono. Sull'uscita, invece, la funzione la detta il
problema, non il gusto.

## In pratica, con NumPy

Ogni funzione è una o due righe. La softmax si scrive nella versione stabile.

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
    z = z - np.max(z)                      # stabilità numerica (log-sum-exp)
    e = np.exp(z)
    return e / e.sum()

softmax(np.array([2.0, 1.0, 0.1]))         # -> [0.659, 0.242, 0.099]
```

In PyTorch (il framework che incontreremo nel prossimo capitolo) non serve
implementarle a mano: esistono come funzioni (`torch.relu`, `torch.tanh`,
`torch.sigmoid`, `torch.softmax`) o come moduli da impilare tra gli strati
(`nn.ReLU()`, `nn.Sigmoid()`), e sono già scritte nella forma numericamente
stabile.

```{admonition} Da ricordare
:class: important
- Senza una **non linearità** tra gli strati, una rete profonda collassa in un
  singolo strato lineare: la profondità sarebbe inutile.
- **Sigmoide** e **tanh** saturano agli estremi e soffrono il *vanishing
  gradient*; la tanh almeno è centrata nello zero.
- La **ReLU** ($\max(0,x)$) non satura dal lato positivo ed è velocissima: è ciò
  che ha reso addestrabili le reti profonde. La **Leaky ReLU** cura i neuroni
  "morti".
- La **softmax** trasforma i logit dell'ultimo strato in probabilità che sommano
  a $1$; va calcolata nella forma numericamente stabile.
```
