# Insegnare a giudicare: classificare il testo

Tra il 1787 e il 1788, sui giornali di New York, escono ottantacinque saggi
firmati con lo pseudonimo *Publius*: sono i **Federalist Papers**, la campagna
di stampa per convincere lo Stato di New York a ratificare la Costituzione
americana. Dietro lo pseudonimo c'erano tre autori (Alexander Hamilton, James
Madison e John Jay) ma per dodici di quei saggi l'attribuzione restò contesa
per un secolo e mezzo: sia Hamilton (morto nel 1804 in un duello, lasciando
una lista dei "suoi" saggi) sia Madison li rivendicavano, e gli storici non
riuscivano a decidere. Nei primi anni Sessanta due statistici, Frederick
Mosteller e David Wallace, provarono una strada nuova: ignorare del tutto le
idee politiche e contare le **parole funzione** (articoli, preposizioni,
congiunzioni, le parole "invisibili" che ognuno usa a modo suo senza
accorgersene). Scoprirono, per esempio, che Hamilton scriveva *upon* più di
tre volte ogni mille parole, Madison meno di una; e misero a frutto un indizio
che gli storici avevano già notato: Hamilton preferiva *while*, Madison
*whilst*. Applicando la regola di Bayes a questi conteggi, i dodici saggi
contesi risultarono tutti di Madison: un verdetto oggi condiviso dagli storici
{cite}`mosteller1964inference`.

Vale la pena fermarsi su cosa è successo: un problema da archivisti è stato
risolto trasformandolo in un problema di **classificazione di testi**
(assegnare a ogni documento un'etichetta, "Hamilton" o "Madison", sulla base
delle parole che contiene). E lo strumento matematico non era un ritrovato
dell'informatica: era un teorema del Settecento, applicato con più pazienza
che potenza di calcolo. In questa sezione costruiamo proprio quel tipo di
giudice automatico, con gli attrezzi di oggi.

## Dare un'etichetta a un testo

La classificazione è il compito più onnipresente del NLP: e il più onnipresente: è spam o no? Questa recensione è positiva o
negativa? In che lingua è scritto questo tweet? Questa email va allo
sportello "reclami" o "fatturazione"? Chi ha scritto questo saggio? Il formato
è sempre lo stesso: in ingresso un documento, in uscita una scelta tra poche
etichette prefissate.

Gli ingredienti li abbiamo già. Nella sezione sulla rappresentazione del testo
abbiamo imparato a trasformare un documento in un vettore di numeri: il
*bag-of-words* dei conteggi, o la sua versione tarata TF-IDF. Qui aggiungiamo
il pezzo mancante: due modelli che, dato quel vettore, emettono il verdetto.
Il primo, **Naive Bayes**, è il discendente diretto del metodo di Mosteller e
Wallace; il secondo, la **regressione logistica**, l'abbiamo già incontrata
nel {doc}`capitolo sul machine learning </MachineLearning/overview>` e qui la mettiamo al lavoro sul testo. Il
confronto tra i due, vedremo, insegna una distinzione che attraversa tutto il
machine learning.

## Naive Bayes: indizi che votano

L'idea di Naive Bayes è quella del detective che non ha la prova regina ma
tanti piccoli indizi: nessuna parola, da sola, dimostra che un'email è spam,
ma ogni parola *sposta* un po’ il sospetto.

```{figure} ../figures/naive-bayes-filtro-antispam.svg
:name: fig-naive-bayes-spam
:alt: "Le parole di una email, ciascuna con il proprio peso a favore o contro l'ipotesi di spam, confluiscono in un blocco centrale che applica il teorema di Bayes; dal blocco esce un'unica probabilità che il messaggio sia spam."
:width: 92%

Nessun indizio decide da solo. Ogni parola porta il proprio piccolo peso al
calcolo, e il verdetto è la probabilità che ne risulta.
```

In {numref}`fig-naive-bayes-spam` si vede anche dove sta l'ingenuità che dà il
nome al metodo: le frecce entrano nel calcolo tutte allo stesso modo, senza
mai incontrarsi fra loro. «Offerta» e «gratis» in una stessa frase valgono
quanto le stesse due parole in capo opposto al messaggio. Il modello fa votare
tutti gli indizi e sceglie l'etichetta che ne esce meglio. L'aggettivo *naive*,
"ingenuo", è dichiarato nel nome: ogni parola vota per conto suo, come se le
altre non esistessero.

Ne segue una cosa che conviene mettere a fuoco adesso, perché tornerà alla fine
della sezione: un giudice fatto così **non vede l'ordine delle parole**. Riceve
un sacchetto di parole e conta chi c'è dentro; di chi veniva prima e chi dopo
non gli arriva niente. Per lui «Il gatto nero salta sul muro» e «Il muro nero
salta sul gatto» sono lo stesso identico messaggio. Per decidere se una
recensione è entusiasta se ne può fare a meno; per altre cose no, ed è il
motivo per cui questo capitolo va avanti.

`````{tab} Elementare

Il conto che segue ha un nome, **regola di Bayes**, e serve a una cosa sola:
girare una domanda. Noi vorremmo sapere «quant'è probabile che questa email sia
spam, visto che dentro c'è scritto "gratis"?», ma dall'archivio sappiamo
contare solo l'opposto, «fra le email che *erano* spam, quante contenevano
"gratis"?». La regola di Bayes dice come passare dalla seconda alla prima, e in
cambio chiede una sola informazione in più: quanto sono frequenti le spam in
generale. È il teorema del Settecento che risolse i Federalist Papers, e lo
useremo senza nemmeno scriverlo.

Costruiamo dunque un mini-filtro antispam con carta e penna. Nel nostro
archivio ci sono 10 email già lette: 4 sono spam, 6 legittime. Contiamo due
parole sospette:

- «gratis» compare in 3 delle 4 spam, e in 1 delle 6 legittime;
- «offerta» compare in 2 delle 4 spam, e in 1 delle 6 legittime.

Arriva una nuova email che contiene sia «gratis» sia «offerta». Ogni ipotesi
raccoglie i suoi voti, moltiplicandoli. Perché moltiplicare e non sommare? Per
la stessa ragione per cui, lanciando due monete, la probabilità di fare testa
tutte e due le volte è una su due **per** una su due, cioè una su quattro, e
non una su due più una su due (che darebbe la certezza, il che è assurdo). Due
cose che devono capitare insieme si moltiplicano, e qui le cose che devono
capitare insieme sono «l'email contiene *gratis*» e «l'email contiene
*offerta*». Ecco i due conti:

- **voto per "spam"**: la quota di spam nell'archivio (4 su 10, cioè 0,4)
  per la frequenza di «gratis» nelle spam (3 su 4, cioè 0,75) per quella di
  «offerta» (2 su 4, cioè 0,5): $0{,}4 \times 0{,}75 \times 0{,}5 = 0{,}15$;
- **voto per "legittima"**: 0,6 per 1/6 per 1/6, che fa circa 0,017.

Vince lo spam, 0,15 contro 0,017, e per dire di quanto basta guardare che 0,15
è quasi nove volte 0,017. Se invece la volete in percentuale, il passaggio è
uno solo: si mette il punteggio del vincitore sopra la somma dei due punteggi,
$0{,}15 \div (0{,}15 + 0{,}017) = 0{,}90$, cioè il 90 per cento di probabilità
che sia spam. (Quei due numeri non sono probabilità già pronte: sono due
punteggi, e diventano probabilità solo quando li si rapporta al totale, come si
fa con i voti di un'elezione a due candidati.)

Nota l'ingenuità: «gratis» e «offerta» viaggiano spesso insieme, ma qui ognuna
vota come se non conoscesse l'altra. E nota un difetto da riparare: se una
parola della nuova email non fosse *mai* comparsa nelle spam dell'archivio, il
suo voto sarebbe zero, e moltiplicando per zero l'intera ipotesi crollerebbe
per colpa di una parola sola. Il rimedio è quasi comico nella sua semplicità:
si regala **un conteggio in più a tutte le parole**, così nessuna resta a zero.
Attenzione che il regalo si paga: se aggiungo 1 sopra a tutti, per non
sballare i conti devo aggiungere sotto il numero di regali distribuiti. È la
"regola del +1" di Laplace, e la ritroveremo presto con i conti per esteso.

Un'ultima avvertenza, per non restare spiazzati fra due pagine. Qui abbiamo
contato *in quante email* una parola compare (3 spam su 4). C'è un secondo modo
di contare, altrettanto legittimo e più diffuso: quante volte la parola compare
in tutto, sul totale delle parole di quella classe. È quello che userà il
programma di `scikit-learn` più avanti. L'idea non cambia di una virgola, i
decimali sì.

`````

`````{tab} Superiore

Dato un documento $d = (w_1, \dots, w_n)$ e un insieme di classi
$\mathcal{C}$, cerchiamo la classe più probabile alla luce del documento.
La regola di Bayes ribalta la condizione:

$$
P(c \mid d) = \frac{P(d \mid c)\, P(c)}{P(d)},
$$

dove $P(c)$ è la probabilità *a priori* della classe (quanto è frequente di
suo), $P(d \mid c)$ è la verosimiglianza del documento data la classe e $P(d)$
(identico per tutte le classi) si può ignorare nell'argmax. Naive Bayes
aggiunge due ipotesi semplificatrici: il documento è un *bag-of-words* (conta
solo quali parole compaiono, non dove: per il modello «Il gatto nero salta sul
muro» e «Il muro nero salta sul gatto» sono lo stesso documento) e le parole
sono **condizionatamente indipendenti** data la classe. La verosimiglianza si
fattorizza allora in un prodotto e la decisione diventa

$$
\hat{c} = \arg\max_{c \,\in\, \mathcal{C}} \; P(c) \prod_{i=1}^{n} P(w_i \mid c),
$$

dove $\hat{c}$ è la classe predetta e $P(w_i \mid c)$ la probabilità della
parola $w_i$ nei documenti di classe $c$. Le stime di massima verosimiglianza
sono semplici frequenze relative:

$$
P(w \mid c) = \frac{\mathrm{conta}(w, c)}{\sum_{w' \in V} \mathrm{conta}(w', c)},
$$

dove $\mathrm{conta}(w, c)$ è il numero di occorrenze di $w$ nei documenti di
addestramento di classe $c$ e $V$ è il vocabolario. Una parola mai vista in
una classe darebbe probabilità zero e azzererebbe il prodotto: lo
**smoothing add-1 di Laplace** lo evita sommando 1 a ogni conteggio,

$$
P(w \mid c) = \frac{\mathrm{conta}(w, c) + 1}{\sum_{w' \in V} \mathrm{conta}(w', c) + |V|},
$$

un'idea che ritroveremo, identica, nella prossima sezione sui modelli n-gram.
Infine un accorgimento numerico: un prodotto di centinaia di probabilità
minuscole va in *underflow*, perciò in pratica si lavora nello spazio dei
logaritmi, massimizzando $\log P(c) + \sum_i \log P(w_i \mid c)$; il prodotto
diventa una somma e l'argmax non cambia, perché il logaritmo è monotono.

Una precisazione sul modello, perché la formula qui sopra ne individua uno solo
di due. Dividendo le occorrenze di $w$ per il **totale dei token** della classe
si ottiene il Naive Bayes **multinomiale**, quello che il codice qui sotto
usa (`MultinomialNB`) e quello adatto quando conta *quante volte* una
parola compare. Esiste anche la variante di **Bernoulli**, in cui $P(w \mid c)$
è la frazione di **documenti** della classe che contengono $w$, e ogni parola
del vocabolario porta un contributo anche quando è assente. È lo stimatore con
cui è stato svolto il filtro a mano dell'altro livello («gratis» in 3 spam su
4, non 3 occorrenze su tutti i token delle spam): due ricette diverse, e i
numeri di un conto non si ottengono con la formula dell'altro. La variante di
Bernoulli è preferibile quando interessa la presenza e non la quantità (testi
molto corti, vocabolari piccoli) e in `scikit-learn` si chiama `BernoulliNB`.

`````

L'ipotesi di indipendenza è linguisticamente falsa (le parole si tirano a
vicenda) eppure Naive Bayes funziona sorprendentemente bene: si addestra con
un solo passaggio sui dati (basta contare), regge anche con pochi esempi
etichettati, e per decenni è stato il cuore dei filtri antispam reali.

## Alla prova: il sentiment delle recensioni

Il banco di prova classico della classificazione è la **sentiment analysis**:
decidere se un testo esprime un giudizio positivo o negativo.

```{figure} ../figures/sentiment-analysis-python.svg
:name: fig-pipeline-sentiment
:alt: "Catena di tre stadi in fila, più l'esito. Il testo grezzo di una recensione stroncatoria diventa un vettore di pesi TF-IDF, in cui le parole distintive pesano molto e quelle comuni quasi nulla; il vettore passa a una regressione logistica, che somma i pesi e li confronta con una soglia; in uscita, di due etichette possibili, si accende «negativo»."
:width: 96%

La catena, dal testo alla polarità. Ogni stadio è sostituibile: cambiare
tokenizzatore o classificatore non cambia la forma della pipeline.
```

La catena di {numref}`fig-pipeline-sentiment` è fatta di stadi staccabili, e
questo è il motivo per cui il compito è rimasto un banco di prova per
vent'anni: si tiene fisso tutto e si cambia un solo pezzo, ed è così che si
confrontano metodi lontanissimi fra loro, dal conteggio di parole del 2002 ai
modelli di oggi.

Lo studio che aprì il filone è del 2002, e lo firmano Bo Pang, Lillian Lee e
Shivakumar Vaithyanathan {cite}`pang2002thumbs`. Presero 1.400 recensioni di
film, 700 entusiaste e 700 stroncature, e ci misero alla prova tre giudici
automatici diversi: Naive Bayes, un cugino stretto della regressione logistica
che vedremo fra poco, e le *support vector machine*, che nel capitolo sul
machine learning hanno una sezione tutta loro e che cercano il confine più
largo possibile fra due gruppi di esempi.

Due risultati restano istruttivi. Il primo: tutti e tre i giudici, che il
giudizio se lo erano ricavato dagli esempi, arrivavano intorno all'80 per cento
di risposte esatte, cioè nettamente meglio del metodo artigianale con cui si
faceva prima, che era compilare a mano una lista di parole belle e una di
parole brutte e contare chi vince (le rivedremo in fondo alla sezione). Il
secondo: giudicare il tono si rivelò più difficile che riconoscere di che
argomento parla un testo, perché l'argomento sta nelle parole e il giudizio si
nasconde nei giri di frase, che i conteggi prendono male.

Con `scikit-learn`, il filtro che sopra abbiamo fatto a mano diventa poche
righe. Costruiamo un micro-corpus di recensioni in italiano:

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

recensioni = [
    "un capolavoro, attori straordinari e regia impeccabile",
    "film splendido, mi ha emozionato dall'inizio alla fine",
    "divertente e intelligente, lo rivedrei subito",
    "una storia che sorprende, fotografia bellissima",
    "che noia, due ore interminabili e senza idee",
    "recitazione pessima e trama piena di buchi",
    "una delusione totale, soldi buttati",
    "banale e prevedibile, mi sono addormentato",
]
etichette = [1, 1, 1, 1, 0, 0, 0, 0]  # 1 = positiva, 0 = negativa

# CountVectorizer = bag-of-words; alpha=1.0 e' lo smoothing di Laplace
modello = make_pipeline(CountVectorizer(), MultinomialNB(alpha=1.0))
modello.fit(recensioni, etichette)

nuove = ["una regia splendida e attori bravissimi",
         "prevedibile e senza emozioni, che delusione"]
print(modello.predict(nuove))          # [1 0]
print(modello.predict_proba(nuove))    # probabilita' per classe
```

Otto esempi sono pochi per qualunque conclusione seria, ma la meccanica è
tutta qui: conteggi in ingresso, regola di Bayes in mezzo, verdetto in
uscita. Al posto di `CountVectorizer` si può usare il `TfidfVectorizer` già
visto nella sezione precedente.

## La regressione logistica: pesare gli indizi

Naive Bayes conta le parole dentro ciascuna delle due etichette possibili
(«classe» è il nome tecnico per «etichetta», e da qui in avanti si trovano
tutti e due) e lascia che la regola di Bayes tiri le somme. C'è un'alternativa
più diretta: imparare, per ogni parola, un **peso** che dica quanto spinge
verso un'etichetta o l'altra, e sommare le spinte. Si chiama **regressione
logistica**, il {doc}`capitolo sul machine learning </MachineLearning/overview>` la presenta fra i modelli
supervisionati, e qui
la mettiamo al lavoro sul testo.

`````{tab} Elementare

Una bilancia a due piatti, uno "positivo" e uno "negativo": ogni parola della
recensione ci butta sopra un pesetto. I pesetti non li decidiamo noi; li impara
il modello dagli esempi etichettati, aggiustandoli un po’ alla volta finché i
verdetti tornano. Dopo l'addestramento potremmo trovare, per dire: «splendido»
+2,0, «sorprende» +1,5, «noia» −2,2, «delusione» −2,5. La frase «un film
splendido, che sorprende» totalizza $2{,}0 + 1{,}5 = 3{,}5$ sul piatto
positivo.

Resta da tradurre quel 3,5 in una probabilità, e a farlo è una regola fissa,
sempre la stessa, che si chiama **sigmoide** (la curva a S del capitolo sul
machine learning): manda lo zero esattamente a metà, cioè a 0,5, spinge i
punteggi positivi verso 1 e quelli negativi verso 0, senza mai arrivare né
all'uno né all'altro. Più il punteggio è alto, più il risultato si avvicina a
uno: a 3,5 la regola risponde circa 0,97, molto convinta ma non certa. (Quel
0,97 non è a occhio: la sigmoide è una formula sola, $1/(1 + e^{-z})$, e
mettendoci $z = 3{,}5$ esce $0{,}9707$. Se il conto non vi dice niente, tenete
l'idea: punteggio alto, probabilità vicina a uno.) Se le
etichette possibili sono più di due (per esempio lo sportello giusto fra
reclami, fatturazione e informazioni) al posto della sigmoide c'è la sua
sorella maggiore, la **softmax**: un punteggio per ogni etichetta, e i
punteggi trasformati in probabilità che sommano a uno.

`````

`````{tab} Superiore

Il documento è un vettore $\mathbf{x} \in \mathbb{R}^{|V|}$: conteggi o pesi
TF-IDF. Il modello calcola il punteggio lineare e lo schiaccia con la sigmoide:

$$
z = \mathbf{w}^\top \mathbf{x} + b,
\qquad
\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}} = P(y = 1 \mid \mathbf{x}),
$$

dove $\mathbf{w} \in \mathbb{R}^{|V|}$ è il vettore dei pesi (uno per parola:
il segno dice la direzione, il modulo la forza dell'indizio), $b$ il bias, $z$
e $\hat{y}$ due scalari, il punteggio e la probabilità della classe positiva.
È la stessa formula del percettrone nel capitolo sulle reti neurali, con la
stessa grafia: minuscolo grassetto per i vettori. Per $K$ classi i pesi
diventano una **matrice** $\mathbf{W} \in \mathbb{R}^{K \times |V|}$ e la
sigmoide lascia il posto alla softmax,
$\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{W}\mathbf{x} + \mathbf{b})$, con
$\hat{\mathbf{y}} \in \mathbb{R}^{K}$ il vettore delle probabilità. I parametri
si stimano minimizzando la cross-entropia
$\mathcal{L}$ con la discesa del gradiente (non esiste una soluzione in forma
chiusa) tipicamente con una regolarizzazione L2 che scoraggia pesi estremi su
parole rare. Un vantaggio pratico: le feature non devono essere solo parole.
Si possono affiancare bigrammi, la lunghezza del documento, il numero di punti
esclamativi, i conteggi da un lessico di sentiment: il modello impara il peso
di ciascuna, qualunque cosa misuri.

`````

## Generativo contro discriminativo

Perché tenere due modelli per lo stesso compito? Perché incarnano due
filosofie diverse, e la differenza (al di là del testo) è uno dei confini
concettuali del machine learning.

`````{tab} Elementare

Due periti devono attribuire lo stesso quadro a uno di due pittori, e ci
arrivano per strade opposte. Il primo studia *tutto* di ciascun pittore
(tavolozza, pennellate, soggetti) fino a saperne quasi imitare lo stile;
davanti a un quadro nuovo si chiede: "quale dei due è più capace di aver
prodotto proprio questo?". È l'approccio
**generativo**, ed è Naive Bayes: impara com'è fatto un documento tipico di
ogni classe. Il secondo perito non sa dipingere e non gli interessa: ha
imparato solo i *dettagli che distinguono* (quella piega del panneggio, quel
blu). È l'approccio **discriminativo**, ed è la regressione logistica: impara
direttamente il confine tra le classi. La differenza si vede sugli indizi
fotocopia: se «gratis» e «offerta» compaiono quasi sempre insieme, per Naive
Bayes sono due voti pieni (conta due volte lo stesso indizio), mentre la
bilancia della regressione logistica se ne accorge durante l'addestramento e
divide il peso tra le due. In compenso il primo perito impara anche da
pochissimi quadri, mentre il secondo ha bisogno di più esempi per capire quali
dettagli contano davvero.

`````

`````{tab} Superiore

Un modello **generativo** stima la distribuzione congiunta
$P(d, c) = P(d \mid c)\,P(c)$ e classifica passando dalla regola di Bayes: per
Naive Bayes, "generare" un documento di classe $c$ significa estrarre parole
da $P(w \mid c)$. Un modello **discriminativo** stima direttamente la quantità
che serve alla decisione, $P(c \mid d)$, senza mai modellare come sono fatti i
documenti. Le conseguenze pratiche: quando le feature sono correlate (e nel
testo lo sono sempre) Naive Bayes moltiplica evidenze non indipendenti e
produce probabilità mal calibrate, schiacciate verso 0 o 1 (la *decisione*
spesso resta giusta, la *confidenza* no); la regressione logistica,
ottimizzando i pesi congiuntamente, ripartisce il credito tra feature
correlate. In cambio, Naive Bayes ha stime a bassa varianza che convergono con
pochi dati e si addestra in un solo passaggio; la regressione logistica tende
a vincere quando gli esempi abbondano. Già nei confronti di Pang, Lee e
Vaithyanathan sulle recensioni di film i modelli discriminativi tendevano a
superare Naive Bayes, ma di poco {cite}`pang2002thumbs`: su compiti lessicali
con dati scarsi, l'ingenuo resta un avversario dignitoso.

`````

## Il classificatore in PyTorch

La bilancia a due piatti, tradotta in PyTorch, sta in tre righe: un peso per
parola, un ciclo che li aggiusta, un verdetto. Prima di guardarla, la traduzione
dei tre nomi che compaiono nel programma. `nn.Linear` **è** la bilancia: un
peso per parola più una costante che sposta l'ago (il *bias*). Il ciclo `for` è
l'addestramento, cioè trecento passaggi sugli stessi otto esempi, in ciascuno
dei quali i pesi si spostano un pochino nella direzione che fa sbagliare di
meno. E il **logit** è il punteggio grezzo della bilancia, quel 3,5 di prima:
il numero che la curva a S non ha ancora trasformato in probabilità.

Chi ha letto il capitolo sulle reti neurali riconoscerà qui il **percettrone**,
cioè un neurone artificiale solo: la ricetta è la stessa, un peso per ingresso
e una somma, e cambia solo come si schiaccia il risultato alla fine. Riusiamo
il micro-corpus di prima, con vettori TF-IDF in ingresso:

```python
import torch
from torch import nn
from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer()
X = torch.tensor(vec.fit_transform(recensioni).toarray(), dtype=torch.float32)
y = torch.tensor(etichette, dtype=torch.float32).unsqueeze(1)

modello = nn.Linear(X.shape[1], 1)      # un peso per parola, piu' il bias
loss_fn = nn.BCEWithLogitsLoss()        # sigmoide + cross-entropia binaria
ottim = torch.optim.Adam(modello.parameters(), lr=0.05)

for epoca in range(300):
    ottim.zero_grad()
    perdita = loss_fn(modello(X), y)    # il modello emette logit, non probabilita'
    perdita.backward()
    ottim.step()

with torch.no_grad():
    X_nuove = torch.tensor(vec.transform(nuove).toarray(), dtype=torch.float32)
    print(torch.sigmoid(modello(X_nuove)).squeeze())  # probabilita' "positiva"
```

Una nota sul nome più ostico, `BCEWithLogitsLoss`: fonde in un'unica operazione
la curva a S e la misura dell'errore, e lo fa perché eseguire i due passi
separati, su numeri molto grandi o molto piccoli, perde precisione. È per
questo che il modello restituisce il punteggio grezzo e la sigmoide si applica
solo al momento di leggere le probabilità. E i pesi imparati si possono
interrogare, parola per parola:

```python
pesi = modello.weight.detach().squeeze()
parole = vec.get_feature_names_out()
ordine = pesi.argsort().tolist()
print("piu' negative:", [parole[i] for i in ordine[:3]])
print("piu' positive:", [parole[i] for i in ordine[-3:]])
```

Su un corpus vero, in cima e in fondo alla lista compaiono proprio le parole
che un lettore umano sottolineerebbe: il modello è una bilancia trasparente,
e questa leggibilità è uno dei motivi per cui resta un riferimento anche
nell'era dei Transformer.

## Giudicare il giudice

Come si misura un classificatore di testi? Con gli strumenti già costruiti nel
capitolo sul machine learning: la matrice di confusione, la precision, la
recall e la loro sintesi $F_1$. Le due parole inglesi sono quelle che si
trovano ovunque, e conviene ridirle nel modo più corto possibile: di quello
che il sistema ha segnalato, quanto era davvero da segnalare (**precision**)?
E di quello che andava segnalato, quanto ne ha trovato (**recall**)? La prima
misura gli abbagli, la seconda le omissioni; $F_1$ è la loro sintesi in un
numero solo.

```{figure} ../figures/precision-recall-f1.svg
:name: fig-quattro-caselle
:alt: "Matrice di confusione due per due: sulle colonne la previsione del modello, sulle righe la realtà, e nelle quattro caselle i veri positivi, i falsi negativi, i falsi positivi e i veri negativi, ciascuno con il suo esempio. Sotto la matrice corre in orizzontale la formula della precision, con una freccia che scende lungo una colonna; sul fianco destro, scritta in verticale, quella della recall, con una freccia che corre lungo una riga. Le due metriche leggono la stessa matrice in due versi perpendicolari."
:width: 96%

Le stesse quattro caselle, lette in due versi perpendicolari. Della roba
segnalata, quanta era giusta: è la precision, e sulla matrice si legge
scendendo lungo una colonna. Di quella da segnalare, quanta ne è stata trovata:
è la recall, e si legge correndo lungo una riga.
```

Il promemoria di {numref}`fig-quattro-caselle` serve perché le due domande
tirano in direzioni opposte, e la ragione è più semplice di quanto sembri. Il
giudice non risponde sì o no: emette un punteggio, e c'è una soglia oltre la
quale segnala. Abbassate la soglia e segnalerete di più: troverete più roba
vera (la recall sale) ma anche più falsi allarmi (la precision scende).
Alzatela e succede l'esatto contrario. Un solo cursore, due numeri che si
muovono in senso inverso: per questo non ha senso chiedere «quanto è bravo» in
astratto, senza dire quale dei due errori costa di più.

E nei testi il costo è quasi sempre asimmetrico. In un filtro antispam una
mail buona cestinata (falso allarme) è molto peggio di uno spam sfuggito,
quindi comanda la precision. In un sistema che cerca segnalazioni di un difetto
pericoloso è il contrario. La metrica da guardare discende da quel costo, non
da una convenzione.

Quando servono tutte e due in un numero solo si usa $F_1$, che è una media
costruita apposta perché un voto basso non si possa nascondere dietro un voto
alto. La ricetta: si moltiplicano i due numeri, si raddoppia il prodotto, e lo
si divide per la loro somma. In simboli, chiamando $P$ la precision e $R$ la
recall, $F_1 = 2PR/(P+R)$.

Provate con precision $1{,}0$ e recall $0{,}1$, cioè un sistema che segnala
pochissimo e però non sbaglia mai. La media normale, quella di scuola, darebbe
un onorevole $(1{,}0 + 0{,}1)/2 = 0{,}55$. Con $F_1$: il prodotto è $0{,}10$,
raddoppiato fa $0{,}20$, la somma dei due voti è $1{,}1$, e $0{,}20$ diviso
$1{,}1$ fa $0{,}18$. Il voto basso comanda, ed è giusto così: un sistema che
segnala una cosa sola e la azzecca non ha risolto niente.

Resta il tranello che nel capitolo sul machine learning avevamo battezzato
"l'accuratezza inganna", e nei testi è la regola più che l'eccezione, perché le
classi sono quasi sempre **sbilanciate**. Se solo un'email su cento è spam, il
filtro pigro che risponde sempre "legittima" sfoggia il 99% di risposte esatte
senza aver fermato nulla. Quale metrica privilegiare non è un dettaglio
tecnico: è la definizione di "successo" per quel particolare giudice.

## Il termometro delle parole: i lessici di sentiment

Prima di chiudere, un attrezzo più artigianale ma tuttora utile: i **lessici
di sentiment**, liste di parole con la loro polarità compilate una volta per
tutte.

`````{tab} Elementare

Un lessico di sentiment è un dizionario dei giudizi: «splendido» +1, «pessimo»
−1, migliaia di voci. Per stimare il tono di un testo basta contare: più
parole positive che negative, verdetto positivo. Il fascino è che non serve
*nessun* esempio etichettato (niente archivio di recensioni già giudicate) e
il verdetto si spiega da solo, parola per parola. I limiti però sono seri. Il
contesto: «imprevedibile» è un complimento per la trama di un film e un'accusa
per i freni di un'auto, ma nel dizionario ha un solo segno. E la negazione:
«non è affatto male» è un complimento, eppure è fatto soltanto di parole che
un elenco di quel genere marchia come negative o neutre, «non» e «male» in
testa. È lo stesso esempio che ritroveremo nel capitolo sui Transformer, e
conviene anticipare come va a finire. Un conteggio di parole isolate quella
frase non la può prendere, per costruzione: presa una per una, nessuna di
quelle parole è un elogio, e il senso sta tutto in come stanno insieme. Un
modello che legge la frase intera con l'attenzione invece potrebbe, perché ha
davanti anche il «non»; e quello che proveremo là sbaglia lo stesso, per un
soffio, dando alla frase due stelle su cinque, cioè leggendola come una
recensione scontenta. Leggere tutta la frase è la condizione per capirla, non
la garanzia.

`````

`````{tab} Superiore

I lessici hanno una storia lunga: il *General Inquirer* di Philip Stone e
colleghi, a metà anni Sessanta, già annotava migliaia di parole inglesi con
categorie tra cui positivo/negativo. Le voci si costruiscono a mano o in modo
semi-supervisionato: si parte da pochi semi di polarità nota e la si propaga
alle parole che co-occorrono in congiunzioni rivelatrici ("bello e X"
suggerisce X positivo, "bello ma X" il contrario) o che risiedono vicine nello
spazio degli embedding della sezione precedente. In un sistema moderno il
lessico raramente decide da solo: i suoi conteggi entrano come feature in una
regressione logistica, dove convivono con i pesi appresi; un innesto utile
soprattutto quando i dati etichettati del dominio sono pochi. Restano i limiti
strutturali di ogni approccio a sacchetto di parole: polarità dipendente dal
dominio, ironia invisibile, e la negazione, che sposta il segno di intere
porzioni di frase e richiede modelli che leggano le sequenze, non i mucchi.

`````

Ed è proprio questo il passo successivo. Il nostro giudice tratta ancora il
testo come un sacchetto: per lui «Il gatto nero salta sul muro» e «Il muro
nero salta sul gatto» sono indistinguibili. Per andare oltre serve un modello
che prenda sul serio l’*ordine* delle parole: che sappia dire quanto è
plausibile una sequenza, e scommettere sulla parola che viene dopo. È il
modello di linguaggio n-gram della prossima sezione, dove ritroveremo un
vecchio amico appena conosciuto: il +1 di Laplace.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- **Classificare un testo** vuol dire assegnargli un'etichetta fra poche già
  decise (spam o no, recensione entusiasta o stroncatura, lingua, autore): un
  problema risolto già nel 1964 sui Federalist Papers, contando le parole
  «invisibili» che ognuno usa a modo suo senza accorgersene.
- **Naive Bayes** fa votare le parole: ogni parola porta il suo piccolo
  indizio, i voti si moltiplicano fra loro e vince l'ipotesi con il punteggio
  più alto. È ingenuo perché ogni parola vota come se le altre non
  esistessero, e funziona lo stesso. Perché una parola mai vista non azzeri
  tutto, si regala **un conteggio in più a ogni parola**: la regola del $+1$
  di Laplace.
- La **regressione logistica** è la bilancia a due piatti: ogni parola butta
  un pesetto da una parte o dall'altra, i pesetti li impara dagli esempi già
  etichettati, e la curva a S traduce il totale in una probabilità (con più di
  due etichette, un punteggio per etichetta).
- I **due periti** davanti ai quadri: il primo (Naive Bayes) studia com'è
  fatto un quadro tipico di ciascun pittore, il secondo (la regressione
  logistica) impara solo i dettagli che li distinguono. Il primo se la cava
  con pochissimi esempi ma conta due volte gli indizi che viaggiano in coppia;
  il secondo se ne accorge e spartisce il peso, e vince quando gli esempi
  abbondano.
- Per giudicare il giudice servono le misure del capitolo sul machine learning
  (quante delle segnalazioni sono giuste, quante ne ha trovate) e non la
  percentuale secca di risposte esatte: se lo spam è una email su cento, chi
  risponde sempre «legittima» ne azzecca il $99\%$ senza aver fermato niente.
- I **lessici di sentiment**, dizionari di parole con il loro segno, non
  chiedono nessun esempio già giudicato, ma sono ciechi al contesto e alla
  negazione: «non è affatto male» resta il controesempio da ricordare.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- **Classificare un testo** = assegnargli un'etichetta tra poche prefissate
  (spam/non spam, positivo/negativo, lingua, autore): un compito risolto con
  la regola di Bayes già nel 1964, sui Federalist Papers, contando le parole
  funzione.
- **Naive Bayes** sceglie la classe che massimizza
  $P(c)\prod_i P(w_i \mid c)$: le parole votano come indizi indipendenti
  (ipotesi falsa ma efficace). Lo **smoothing add-1 di Laplace** evita gli
  zeri; in pratica si calcola tutto in spazio logaritmico.
- La **regressione logistica** impara un peso per parola e passa la somma
  nella sigmoide (softmax per più classi): stessa ricetta del capitolo sul
  machine learning, applicata ai vettori bag-of-words o TF-IDF.
- **Generativo vs discriminativo**: Naive Bayes modella $P(d \mid c)\,P(c)$,
  la regressione logistica direttamente $P(c \mid d)$; il primo impara da
  pochi dati ma conta due volte gli indizi correlati, la seconda ripartisce
  i pesi e vince quando gli esempi abbondano.
- La valutazione usa **precision, recall e $F_1$** del capitolo sul machine
  learning: con classi sbilanciate (lo spam è raro) l'accuratezza inganna.
- I **lessici di sentiment** funzionano senza dati etichettati ma sono ciechi
  a contesto e negazione: «non è affatto male» resta il controesempio da
  ricordare.
```
`````
