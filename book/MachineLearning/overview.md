# Machine Learning: imparare dai dati

Nel 1959 un ingegnere dell'IBM di nome Arthur Samuel pubblicò un articolo dal
titolo modesto (*Some Studies in Machine Learning Using the Game of Checkers*
{cite}`samuel1959some`) che oggi suona profetico. Samuel aveva scritto un
programma che giocava a dama, e la parte sorprendente è questa: dopo migliaia
di partite contro sé stesso, il programma giocava **meglio del suo autore**.
Non perché Samuel gli avesse insegnato le mosse giuste una per una, ma perché
il programma le aveva ricavate dall'esperienza. In quelle pagine compare, tra
le prime volte nella storia, l'espressione *machine learning*: la capacità di
un calcolatore di migliorare a un compito senza essere riprogrammato a mano.

È un'idea che ribalta il modo consueto di pensare al software.

## Scrivere le regole, o farle emergere

Il salto concettuale del machine learning si capisce meglio mettendolo accanto
alla programmazione di sempre.

`````{tab} Elementare

Immagina di dover costruire un filtro antispam. Con la programmazione classica
ti siedi e **scrivi tu le regole**: "se l'email contiene la parola *vincita*,
segnala come spam", "se il mittente è sconosciuto, sospetta". Ogni regola la
pensi, la scrivi, la correggi a mano. Funziona finché gli spammer non cambiano
trucco, e allora ricominci da capo.

Il machine learning fa il contrario. Tu non scrivi le regole: raccogli
**migliaia di email già etichettate** come "spam" o "non spam" e le dai in
pasto al programma. È lui a trovare da solo le regolarità: quali parole, quali
mittenti, quali combinazioni ricorrono nello spam. Le regole *emergono* dai
dati, non le scrivi tu.

`````

`````{tab} Superiore

Nella programmazione tradizionale il programmatore conosce la funzione
$f$ che trasforma un input in un output e la codifica esplicitamente:
$\text{output} = f(\text{input})$. Le regole sono note *a priori*.

Nel machine learning $f$ è **ignota**. Disponiamo invece di una collezione di
coppie input-output osservate, e cerchiamo una funzione $f_\theta$, presa da
una famiglia parametrizzata dai parametri $\theta$, che le riproduca bene e
(soprattutto) **generalizzi** a input mai visti. Il compito non è più
*scrivere* $f$, ma *stimare* i parametri $\theta$ a partire dai dati. Il
codice resta fisso; ciò che cambia, con l'esperienza, sono i numeri dentro
$\theta$.

`````

## Che cosa vuol dire "imparare": la definizione di Mitchell

La definizione operativa più citata è quella di Tom Mitchell, nel manuale
*Machine Learning* del 1997. Ha il pregio di essere verificabile: dice quando
un programma sta davvero imparando e quando no.

`````{tab} Elementare

Un programma **impara** se, facendo pratica, diventa più bravo in un compito, e
questo "più bravo" lo possiamo misurare. Servono tre ingredienti:

- il **compito**, cosa deve fare (giocare a dama);
- l'**esperienza**, su cosa fa pratica (le partite giocate);
- la **misura**, come contiamo i progressi (la percentuale di partite vinte).

Il programma di Samuel diventava più bravo (vinceva di più) man mano che
accumulava partite. Questo, e solo questo, è imparare.

`````

`````{tab} Superiore

Mitchell la formula così: un programma apprende da un'esperienza $E$ rispetto a
una classe di compiti $T$ e a una misura di performance $P$, se la sua
performance sui compiti in $T$, misurata da $P$, **migliora con l'esperienza**
$E$.

- $T$ (*task*): il problema, per esempio classificare email.
- $E$ (*experience*): i dati da cui apprende, per esempio $m$ email etichettate.
- $P$ (*performance*): una metrica scalare, per esempio l'accuratezza sul test.

La formulazione è deliberatamente astratta: non nomina reti neurali né alberi
di decisione. È un contratto che qualunque algoritmo di apprendimento deve
rispettare: se all'aumentare di $E$ la $P$ non cresce, non c'è apprendimento.

`````

## Tre modi di imparare

A seconda del tipo di esperienza a disposizione, il machine learning si
divide in tre grandi paradigmi.

**Apprendimento supervisionato.** È il caso del filtro antispam: ogni esempio
arriva con la sua **risposta giusta** (l'etichetta). Il modello impara la
corrispondenza tra domanda e risposta (input → output). Se l'output è una categoria si parla di
*classificazione* (spam / non spam, gatto / cane); se è un numero continuo di
*regressione* (prevedere il prezzo di una casa dai suoi metri quadri). È di gran
lunga il paradigma più usato in pratica.

`````{tab} Elementare

Pensa a uno studente che studia con le soluzioni a fianco degli esercizi. Per
ogni esercizio vede sia il problema sia la risposta corretta, e col tempo
impara a rispondere da solo a esercizi nuovi. Le "soluzioni a fianco" sono le
etichette: senza di esse, questo tipo di apprendimento non funziona.

`````

`````{tab} Superiore

Dato un insieme di addestramento $\{(X^{(i)}, y^{(i)})\}_{i=1}^{m}$, dove
$X^{(i)}$ è il vettore delle feature dell'esempio $i$-esimo e $y^{(i)}$ la sua
etichetta, si cerca $f_\theta$ che minimizzi una funzione di costo (o *loss*)
$\mathcal{L}$ che penalizza le previsioni sbagliate:

$$
\theta^\star = \arg\min_{\theta}\ \frac{1}{m}\sum_{i=1}^{m}
\mathcal{L}\!\left(f_\theta(X^{(i)}),\, y^{(i)}\right).
$$

Qui $f_\theta(X^{(i)}) = \hat{y}^{(i)}$ è la predizione del modello e
$\mathcal{L}$ misura la sua distanza dal valore vero $y^{(i)}$. L'intero
addestramento supervisionato è, in fondo, questo problema di minimizzazione.

`````

**Apprendimento non supervisionato.** Qui le etichette **non ci sono**: il
modello riceve solo gli input e deve scoprire da sé una struttura nascosta.
L'esempio classico è il *clustering*: raggruppare i clienti di un negozio in
segmenti simili senza sapere in anticipo quali segmenti esistano. Rientrano qui
anche la riduzione della dimensionalità (comprimere i dati mantenendone
l'essenza) e i sistemi che rilevano anomalie in una transazione.

**Apprendimento per rinforzo.** Non ci sono né etichette né dataset fisso: c'è
un **agente** che compie azioni in un ambiente e riceve, di tanto in tanto, una
**ricompensa**. L'agente impara per tentativi la strategia che massimizza la
ricompensa nel tempo. È così che AlphaGo di DeepMind ha imparato il Go (2016), e
in fondo è proprio ciò che faceva il programma di dama di Samuel: giocare,
vedere l'esito, e correggere la propria strategia. Vincere era la ricompensa.

## Dall'idea al modello: il flusso di un progetto

Un progetto di machine learning non è mai solo "addestrare un modello". È una
catena di passaggi, e (dettaglio cruciale) non è una linea retta ma un
**ciclo**: i risultati della valutazione ti dicono come tornare indietro e
fare meglio ({numref}`fig-workflow-ml`).

```{figure} ../figures/workflow-ml.svg
:name: fig-workflow-ml
:alt: Cinque blocchi in fila (Dati, Feature, Modello, Valutazione, Deploy) collegati da frecce; una freccia di feedback torna dalla Valutazione alle Feature.
:width: 95%

Il flusso tipico di un progetto ML. Dopo la valutazione si torna quasi sempre
indietro a rivedere feature e modello: l'apprendimento è iterativo.
```

I passaggi, in ordine:

1. **Dati**: raccogliere esempi e ripulirli (valori mancanti, duplicati,
   errori). Spesso è la fase più lunga e ingrata dell'intero progetto.
2. **Feature**, trasformare i dati grezzi nella rappresentazione numerica che
   il modello riceverà: le *feature*. Rappresentare bene un problema è metà
   della soluzione.
3. **Modello**, scegliere una famiglia di modelli e **addestrarla** sui dati:
   l'addestramento cerca i numeri interni del modello (i *parametri* $\theta$)
   che riducono al minimo l'errore (la *loss*).
4. **Valutazione**: misurare le prestazioni su dati **mai visti** in
   addestramento, per stimare come il modello si comporterà nel mondo reale.
5. **Deploy**: se i numeri convincono, mettere il modello in produzione e
   monitorarlo, perché i dati del mondo cambiano nel tempo.

La freccia di ritorno è la parte più importante: quasi mai il primo tentativo
è quello buono. Si osserva dove il modello sbaglia, si tornano a ritoccare le
feature o il modello, e si ricomincia il giro.

`````{tab} Elementare

In pratica il passaggio "training" è sorprendentemente breve da scrivere.
Con una libreria come scikit-learn addestrare un modello e usarlo sono due
righe: `.fit()` per imparare dai dati, `.predict()` per prevedere su casi nuovi.

`````

`````{tab} Superiore

Il metodo `fit` implementa la minimizzazione della loss vista sopra; `predict`
applica la $f_{\theta^\star}$ appresa. La separazione tra dati di addestramento
e dati di test serve a stimare la capacità di **generalizzazione**, non la mera
memorizzazione degli esempi già visti.

`````

```python
from sklearn.tree import DecisionTreeClassifier

# X_train: le feature di ogni esempio, y_train: l'etichetta da prevedere
modello = DecisionTreeClassifier()
modello.fit(X_train, y_train)       # training: il modello impara dai dati
y_pred = modello.predict(X_test)    # previsione su dati mai visti in training
```

## Perché questo capitolo non è archeologia

C'è una scena che si ripete in ogni team alle prime armi. Arriva un problema
(prevedere quali clienti abbandoneranno il servizio, a partire da una tabella
di età, contratti, consumi, reclami) e qualcuno propone subito una rete
neurale profonda, perché è quella di cui parlano tutti. Dopo due settimane di
tuning la rete arriva faticosamente a pareggiare un gradient boosting che un
collega scettico aveva addestrato in dieci minuti con i parametri di default.

`````{tab} Elementare

Il deep learning ha ridefinito cosa è possibile con immagini, audio e
linguaggio. Ma dedurne che sia lo strumento migliore per qualsiasi problema è
un errore di categoria, e su un'ampia fascia di casi reali, probabilmente la
maggioranza di quelli che un'azienda incontra, i metodi di questo capitolo
restano la scelta più sensata.

La distinzione che conta non è l'età dell'algoritmo, ma **la forma dei dati**.
In una tabella, la colonna "codice postale", quella "reddito annuo" e quella
"ha un contratto attivo" non hanno nulla in comune: unità diverse, scale
diverse, significati diversi. Non c'è vicinanza spaziale come fra due pixel
adiacenti, né ordine come fra due parole in una frase. Sono **variabili senza
geografia**, e le reti neurali sono costruite proprio per sfruttare una
geografia: convoluzioni per i pixel vicini, attenzione per le parole in
sequenza. Su una tabella quella struttura non c'è, e il vantaggio evapora.

Gli alberi, al contrario, si trovano a casa: dividono una colonna alla volta con
una soglia, non chiedono che le scale siano confrontabili, e gestiscono
naturalmente variabili categoriche e valori mancanti.

Tre casi in cui il classico resta la scelta giusta:

- **dati tabulari**, che sono la forma più comune dei dati aziendali;
- **pochi esempi**, con qualche migliaio di righe una rete profonda non ha
  abbastanza materiale per imparare le feature da sola;
- **serve spiegare la decisione**: un albero o una regressione si leggono, e
  quando una decisione va motivata a un cliente o a un regolatore questo non è
  un dettaglio.

`````

`````{tab} Superiore

L'osservazione è stata messa alla prova sistematicamente: Grinsztajn, Oyallon e
Varoquaux (NeurIPS 2022) hanno confrontato modelli ad albero e reti neurali su
decine di dataset tabulari, trovando che i primi restano superiori anche a
parità di tuning, e con un budget di ricerca degli iperparametri molto minore.

Le ragioni identificate sono strutturali, non contingenti:

1. le reti hanno un *bias induttivo* verso funzioni regolari, mentre i target
   tabulari sono spesso irregolari a tratti, esattamente ciò che una serie di
   split assiali approssima bene;
2. le reti sono sensibili alle feature non informative, di cui una tabella
   reale abbonda, mentre gli alberi le ignorano per costruzione;
3. le reti sono invarianti per rotazione, proprietà desiderabile sui pixel e
   dannosa su colonne che hanno significati diversi e non intercambiabili.

Il corollario pratico riguarda il **costo**: un gradient boosting si addestra
in minuti su CPU e si mette in produzione senza GPU. Prima di pagare il conto
del deep learning conviene avere una baseline classica ben tarata, e succede
spesso che quella baseline sia già la risposta.

`````

```{admonition} Da ricordare
:class: important
- Nel machine learning **non si scrivono le regole**: si forniscono esempi e le
  regole emergono dai dati (parametri $\theta$).
- Su **dati tabulari**, con pochi esempi o quando serve spiegare la decisione,
  i metodi di questo capitolo battono ancora regolarmente il deep learning.
- Un programma **impara** (Mitchell) se la sua performance $P$ su un compito $T$
  migliora con l'esperienza $E$.
- Tre paradigmi: **supervisionato** (dati etichettati), **non supervisionato**
  (struttura nascosta, senza etichette), **per rinforzo** (agente e ricompense).
- Il flusso (dati, feature, modello, valutazione, deploy) è un **ciclo**: la
  valutazione rimanda indietro, e si itera.
```
