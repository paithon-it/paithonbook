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

La classificazione è il primo dei compiti elencati nella panoramica del
capitolo, e il più onnipresente: è spam o no? Questa recensione è positiva o
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
nel capitolo sul machine learning e qui la mettiamo al lavoro sul testo. Il
confronto tra i due, vedremo, insegna una distinzione che attraversa tutto il
machine learning.

## Naive Bayes: indizi che votano

L'idea di Naive Bayes è quella del detective che non ha la prova regina ma
tanti piccoli indizi: nessuna parola, da sola, dimostra che un'email è spam,
ma ogni parola *sposta* un po' il sospetto. Il modello fa votare tutti gli
indizi e sceglie l'etichetta che raccoglie più voti. L'aggettivo *naive*,
"ingenuo", è dichiarato nel nome: ogni parola vota per conto suo, come se le
altre non esistessero.

`````{tab} Elementare

Costruiamo un mini-filtro antispam con carta e penna. Nel nostro archivio ci
sono 10 email già lette: 4 sono spam, 6 legittime. Contiamo due parole
sospette:

- «gratis» compare in 3 delle 4 spam, e in 1 delle 6 legittime;
- «offerta» compare in 2 delle 4 spam, e in 1 delle 6 legittime.

Arriva una nuova email che contiene sia «gratis» sia «offerta». Ogni ipotesi
raccoglie i suoi voti, moltiplicandoli:

- **voto per "spam"**: la quota di spam nell'archivio (4 su 10, cioè 0,4)
  per la frequenza di «gratis» nelle spam (3 su 4, cioè 0,75) per quella di
  «offerta» (2 su 4, cioè 0,5): $0{,}4 \times 0{,}75 \times 0{,}5 = 0{,}15$;
- **voto per "legittima"**: 0,6 per 1/6 per 1/6, che fa circa 0,017.

0,15 contro 0,017: lo spam vince nove a uno (riportato in percentuale, il 90%
di probabilità che sia spam). Nota l'ingenuità: «gratis» e «offerta» viaggiano
spesso insieme, ma qui ognuna vota come se non conoscesse l'altra. E nota un
difetto da riparare: se una parola della nuova email non fosse *mai* comparsa
nelle spam dell'archivio, il suo voto sarebbe zero, e moltiplicando per zero
l'intera ipotesi crollerebbe per colpa di una parola sola. Il rimedio è quasi
comico nella sua semplicità: si regala **un conteggio in più a tutte le
parole**, così nessuna resta a zero. È la "regola del +1" di Laplace, e la
ritroveremo presto.

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

`````

L'ipotesi di indipendenza è linguisticamente falsa (le parole si tirano a
vicenda) eppure Naive Bayes funziona sorprendentemente bene: si addestra con
un solo passaggio sui dati (basta contare), regge anche con pochi esempi
etichettati, e per decenni è stato il cuore dei filtri antispam reali.

## Alla prova: il sentiment delle recensioni

Il banco di prova classico della classificazione è la **sentiment analysis**:
decidere se un testo esprime un giudizio positivo o negativo. Lo studio che
aprì il filone è del 2002: Pang, Lee e Vaithyanathan presero 1.400 recensioni
di film (700 entusiaste e 700 stroncature) e confrontarono Naive Bayes, un
modello log-lineare parente stretto della regressione logistica e le support
vector machine {cite}`pang2002thumbs`. Due risultati restano istruttivi: i
modelli appresi dai dati arrivavano intorno all'80% di accuratezza, ben sopra
le liste di parole positive e negative compilate a mano; e il sentiment si
rivelò più difficile della classificazione per argomento, perché il giudizio
si nasconde in giri di frase che i conteggi catturano male.

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

Naive Bayes conta le parole *dentro ciascuna classe* e lascia che Bayes tiri
le somme. C'è un'alternativa più diretta: imparare, per ogni parola, un
**peso** che dica quanto spinge verso un'etichetta o l'altra, e sommare le
spinte. È la **regressione logistica** del capitolo sul machine learning
(punteggio lineare più sigmoide) applicata ai vettori di testo.

`````{tab} Elementare

Immagina una bilancia a due piatti: ogni parola della recensione butta un
pesetto sul piatto "positivo" o su quello "negativo", e i pesetti non li
decidiamo noi; li impara il modello dagli esempi etichettati, aggiustandoli un
po' alla volta finché i verdetti tornano. Dopo l'addestramento potremmo
trovare, per dire: «splendido» +2,0, «sorprende» +1,5, «noia» −2,2,
«delusione» −2,5. La frase «un film splendido, che sorprende» totalizza
$2{,}0 + 1{,}5 = 3{,}5$ sul piatto positivo; la curva a S della **sigmoide**
(la stessa incontrata nel capitolo sul machine learning) traduce il punteggio
in una probabilità: circa 0,97. Se le etichette possibili sono più di due (per
esempio lo sportello giusto tra reclami, fatturazione e informazioni), al
posto della sigmoide c'è la softmax delle reti neurali: un punteggio per
etichetta, e tutte le probabilità che sommano a uno.

`````

`````{tab} Superiore

Il documento è un vettore $X \in \mathbb{R}^{|V|}$: conteggi o pesi TF-IDF. Il
modello calcola il punteggio lineare e lo schiaccia con la sigmoide:

$$
z = W^\top X + b,
\qquad
\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}} = P(y = 1 \mid X),
$$

dove $W \in \mathbb{R}^{|V|}$ è il vettore dei pesi (uno per parola: il segno
dice la direzione, il modulo la forza dell'indizio), $b$ il bias e $\hat{y}$
la probabilità della classe positiva. Per $K$ classi si passa a una matrice di
pesi e alla softmax, $\hat{y} = \mathrm{softmax}(W X + b)$, vista nel capitolo
sulle reti neurali. I parametri si stimano minimizzando la cross-entropia
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

Pensa a due periti chiamati a distinguere i quadri di due pittori. Il primo
studia *tutto* di ciascun pittore (tavolozza, pennellate, soggetti) fino a
saperne quasi imitare lo stile; davanti a un quadro nuovo si chiede: "quale
dei due è più capace di aver prodotto proprio questo?". È l'approccio
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

La regressione logistica per il testo è un singolo strato lineare: di fatto un
neurone solo, come il percettrone del capitolo sulle reti neurali, ma con
uscita sigmoidea invece del gradino. L'occasione è perfetta per scriverla in
PyTorch. Riusiamo il micro-corpus di prima, con vettori TF-IDF in ingresso:

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

`BCEWithLogitsLoss` fonde sigmoide e cross-entropia in un'unica operazione
numericamente stabile: per questo il modello restituisce il punteggio grezzo
(il *logit*) e la sigmoide si applica solo al momento di leggere le
probabilità. E i pesi imparati si possono interrogare, parola per parola:

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
recall e la loro sintesi $F_1$. Non li ripetiamo; ricordiamo solo il tranello
che lì avevamo battezzato "l'accuratezza inganna", perché nel testo è la
regola più che l'eccezione: le classi sono quasi sempre **sbilanciate**. Se
solo un'email su cento è spam, il filtro pigro che risponde sempre "legittima"
sfoggia il 99% di accuratezza senza aver fermato nulla; e tra i due errori
possibili, un falso positivo (un'email importante nel cestino), costa più di
uno spam sfuggito, quindi è la precision a comandare. Quale metrica
privilegiare non è un dettaglio tecnico: è la definizione di "successo" per
quel particolare giudice.

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
«non è affatto male» contiene due parole da piatto negativo («non» e «male»)
eppure è un complimento. È lo stesso esempio che abbiamo visto nel capitolo
sui Transformer: lì il modello, leggendo la frase intera con l'attenzione, lo
risolve; un conteggio di parole isolate, per costruzione, non può.

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
che prenda sul serio l'*ordine* delle parole: che sappia dire quanto è
plausibile una sequenza, e scommettere sulla parola che viene dopo. È il
modello di linguaggio n-gram della prossima sezione, dove ritroveremo un
vecchio amico appena conosciuto: il +1 di Laplace.

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
