# Machine Learning: imparare dai dati

Nel 1959 un ingegnere dell'IBM di nome Arthur Samuel pubblicò un articolo dal
titolo modesto (*Some Studies in Machine Learning Using the Game of Checkers*
{cite}`samuel1959some`) che oggi suona profetico. Samuel aveva scritto un
programma che giocava a dama, e la parte sorprendente è questa: giocando contro
sé stesso, il programma arrivò a giocare **meglio del suo autore**. Non perché
Samuel gli avesse insegnato le mosse giuste una per una, ma perché le aveva
ricavate dall'esperienza.

Se giocava contro sé stesso, però, chi gli diceva quale mossa fosse quella
buona? Nessuno, ed è qui l'idea. Il programma dava un voto alla posizione che
aveva davanti, poi guardava qualche mossa più in là, e correggeva il voto di
adesso avvicinandolo a quello che vedeva **dopo**. Nessuno gli diceva chi
avesse ragione: a fare da maestro era la propria stessa valutazione, presa un
passo più avanti, dove si vede meglio. Ripetuto per tutta la partita e per
tutte le partite, quel voto diventa un fiuto per le posizioni che portano bene.
È il modo di imparare per tentativi e ricompense, e la sua forma
matura si chiama apprendimento per differenze temporali: il nome torna nella
pagina sul {doc}`Q-learning </ReinforcementLearning/q-learning>`, ed è lì che
si vede per intero.

In quell'articolo compare, tra le prime volte nella storia, l'espressione
*machine learning*: la capacità di un calcolatore di migliorare a un compito
senza essere riprogrammato a mano.

È un'idea che ribalta il modo consueto di pensare al software.

Il {doc}`capitolo di matematica </Matematica/overview>` si era chiuso sull'ultimo strato di un modello
linguistico, cioè su un punto d'arrivo; con Samuel siamo al punto di partenza,
e la strada fra i due è il resto del libro.

## Scrivere le regole, o farle emergere

Il salto concettuale del machine learning si capisce meglio mettendolo accanto
alla programmazione di sempre.

`````{tab} Elementare

Con la programmazione di sempre, le regole di un filtro antispam **le scrivi
tu**: "se l'email contiene la parola *vincita*, segnala come spam", "se il
mittente è sconosciuto, sospetta". Ogni regola la pensi, la scrivi, la correggi
a mano. Funziona finché gli spammer non cambiano trucco, e allora ricominci da
capo.

Il machine learning fa il contrario. Tu non scrivi le regole: raccogli
**migliaia di email già etichettate** come "spam" o "non spam" e le dai in
pasto al programma. È lui a trovare da solo le regolarità: quali parole, quali
mittenti, quali combinazioni ricorrono nello spam. Le regole *emergono* dai
dati, non le scrivi tu.

E devono reggere fuori dal mucchio. Le email che gli hai dato qualcuno le ha
già smistate a mano, quindi rismistarle non serve a nessuno: quello che ti
serve è che se la cavi sulla prossima, quella che nel mucchio non c'era.

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

Il risultato di tutto questo ha un nome, ed è la parola che tornerà da qui alla
fine: il **modello**. Non è il
modellino di un aeroplano né chi sfila in passerella. È il programma *dopo* che
ha visto i dati: la regola che quei dati hanno prodotto. Una regola del genere,
dentro un calcolatore, è fatta di numeri (quanto conta la parola «vincita»,
quanto conta un mittente sconosciuto), e sono proprio quei numeri a rendere la
regola quella lì e non un'altra. Il codice che scriviamo
resta sempre lo stesso; il modello è ciò che ne esce quando lo si è fatto
passare su un mucchio di esempi. E *come* faccia a trovare quelle regolarità,
chi gli dica di guardare le parole di un'email e non il colore dello schermo, è
esattamente la domanda giusta: la risposta arriva un pezzo per volta nelle
prossime sezioni, e per ora basta sapere che quelle regole non le scriviamo noi.

## Che cosa vuol dire "imparare": la definizione di Mitchell

La definizione operativa più citata è quella di Tom Mitchell, nel manuale
*Machine Learning* del 1997 {cite}`mitchell1997machine`. Ha il pregio di essere
verificabile: dice quando un programma sta davvero imparando e quando no.

`````{tab} Elementare

Un programma **impara** se, facendo pratica, diventa più bravo in un compito, e
questo "più bravo" lo possiamo misurare. Servono tre ingredienti:

- il **compito**, cosa deve fare (giocare a dama);
- l’**esperienza**, su cosa fa pratica (le partite giocate);
- la **misura**, come contiamo i progressi (la percentuale di partite vinte).

Il programma di Samuel diventava più bravo (vinceva di più) man mano che
accumulava partite. Questo, e solo questo, è imparare. Se dopo mille partite ne
vincesse la stessa quota di prima non avrebbe imparato niente, per quante ne
avesse giocate, perché a decidere è il numero che sale e non il tempo passato
al tavolo.

I tre ingredienti non parlano di dama. Al posto del compito metti il
riconoscere una firma falsa, al posto dell'esperienza le firme già controllate
una per una, al posto della misura quante ne sbagli su cento. La domanda da
fare resta la stessa, e cioè se quegli errori calano man mano che le firme
controllate aumentano. Funziona con un programma qualunque, senza sapere niente
di come è fatto dentro.

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
divide in tre grandi famiglie (chi scrive di ricerca le chiama *paradigmi*).

**Apprendimento supervisionato.** È il caso del filtro antispam: ogni esempio
arriva con la sua **risposta giusta** (l'etichetta). Il modello impara a legare
la domanda alla risposta: quello che entra si chiama *input*, quello che esce
*output*, e sono due parole che d'ora in poi useremo sempre. Se l'output è una
categoria si parla di *classificazione* (spam / non spam, gatto / cane); se è
un numero su una scala **continua**, cioè una scala in cui fra due valori ce
n'è sempre un altro (2,5 metri quadri esistono, 2,5 stanze no), si parla di
*regressione* (prevedere il prezzo di una casa dai suoi metri quadri). È di gran
lunga il modo più usato in pratica.

`````{tab} Elementare

A fianco di ogni esercizio c'è la soluzione. Chi studia così prova a rispondere
da solo, poi scopre il riquadro accanto e vede di quanto ha sbagliato. Su ogni
esercizio gli resta un numero, lo scarto fra la sua risposta e quella giusta.

Un singolo scarto dice poco. Quello che conta è la media su tutta la raccolta:
cinque esercizi sbagliati di 2, 0, 1, 0 e 2 punti fanno 5 punti in tutto, cioè
1 punto a esercizio, ed è quella media che chi studia cerca di far scendere. Si
guarda lei, e mai il singolo esercizio. Un modo di rispondere che azzecca un
esercizio solo e manda fuori strada gli altri quattro alza la media, quindi
vale meno, per quanto sia perfetto lì. Fra due modi di rispondere si tiene
sempre quello con la media più bassa.

Le "soluzioni a fianco" sono le etichette: senza di esse non c'è niente contro
cui misurare lo scarto, e questo tipo di apprendimento non funziona. La
raccolta, poi, si fa per gli esercizi che ancora non ci sono, quelli senza
soluzione stampata sotto, dove si vede se qualcuno ha davvero imparato.

`````

`````{tab} Superiore

Dato un insieme di addestramento
$\{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^{m}$, dove $\mathbf{x}^{(i)}$ è il
vettore delle feature dell'esempio $i$-esimo e $y^{(i)}$ la sua
etichetta, si cerca $f_\theta$ che minimizzi una funzione di costo (o *loss*)
che penalizza le previsioni sbagliate:

$$
\theta^\star = \arg\min_{\theta}\ \frac{1}{m}\sum_{i=1}^{m}
\ell\!\left(f_\theta(\mathbf{x}^{(i)}),\, y^{(i)}\right).
$$

Qui $f_\theta(\mathbf{x}^{(i)}) = \hat{y}^{(i)}$ è la predizione del modello e
$\ell$ misura la sua distanza dal valore vero $y^{(i)}$; la media di tutti gli
$\ell$ è la loss sull'intero insieme, che il libro scrive $\mathcal{L}$. L'intero
addestramento supervisionato è, in fondo, questo problema di minimizzazione.

`````

**Apprendimento non supervisionato.** Qui le etichette **non ci sono**: il
modello riceve solo gli input e deve scoprire da sé una struttura nascosta.
L'esempio classico è il *clustering*: raggruppare i clienti di un negozio in
segmenti simili senza sapere in anticipo quali segmenti esistano. Rientrano qui
anche la **riduzione della dimensionalità** (le «dimensioni» sono le colonne
della tabella, una per caratteristica, e la prossima sezione spiega perché si
chiamino così), cioè descrivere ogni esempio con
meno numeri senza perderne l'essenza; e i sistemi che rilevano anomalie in una
transazione.

**Apprendimento per rinforzo.** Non ci sono etichette, e non c'è nemmeno un
mucchio di esempi fissato in partenza: c'è un **agente** (un programma che
agisce, non una persona) che compie azioni in un ambiente e riceve, di tanto in
tanto, una **ricompensa**. L'agente impara per tentativi la strategia che
massimizza la ricompensa nel tempo. È il modo in cui il programma AlphaGo, del
laboratorio DeepMind, imparò nel 2016 a battere i campioni del go, un antico
gioco da tavolo orientale: quella prima versione studiò anche partite umane
etichettate, mentre la versione dell'anno dopo imparò solo giocando contro sé
stessa. Che poi è esattamente ciò che faceva il programma di dama di Samuel:
giocare, vedere l'esito, correggere la strategia. Vincere era la ricompensa.

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
indietro a rivedere feature e modello, e si ricomincia il giro.
```

I passaggi, in ordine:

1. **Dati**: raccogliere esempi e ripulirli (valori mancanti, duplicati,
   errori). Spesso è la fase più lunga e ingrata dell'intero progetto. Di
   solito si organizzano in una **tabella**: una riga per esempio, una colonna
   per ogni cosa che di quell'esempio abbiamo misurato.
2. **Feature**: un modello non sa leggere un'email, sa fare conti su dei
   numeri. Le **feature** (in italiano: le *caratteristiche*, e sono proprio le
   colonne della tabella) sono i numeri con
   cui descriviamo ogni esempio, e sceglierli è un lavoro nostro. Di un'email
   possiamo prendere quante parole ha, quanti punti esclamativi, quante volte
   compare la parola «vincita», se il mittente è in rubrica: cinque numeri, e
   quell'email per il modello è diventata quei cinque numeri. Cambiando i numeri
   che si prendono cambia la risposta, ed è per questo che si dice che
   rappresentare bene un problema è metà della soluzione.
3. **Modello**: decidere che *forma* dare al modello (una retta? un albero di
   domande? una rete?) e poi **addestrarlo** sui dati.
   Dentro un modello ci sono dei numeri regolabili, come le manopole di un
   vecchio amplificatore: si chiamano **parametri** (nelle formule del libro:
   $\theta$, la lettera greca *theta*). Addestrare vuol dire girare quelle
   manopole finché il modello sbaglia il meno possibile, e «quanto sbaglia» è a
   sua volta un numero, che si chiama **loss** (la *perdita*: quanto ci costa
   ogni risposta sbagliata). Attenzione a non confondere i due momenti: la
   forma la scegliamo prima, i numeri dentro li trova l'addestramento, e
   «modello» in senso stretto è il risultato dei due messi insieme.
4. **Valutazione**: misurare le prestazioni su dati **mai visti** in
   addestramento, per stimare come il modello si comporterà nel mondo reale.
   Perché non riusare gli esempi di prima, che ci sono già? Perché su quelli un
   modello può cavarsela benissimo limitandosi a ricordarli, e ricordare non è
   un'abilità che ci serva: quello che vogliamo sapere è come se la caverà
   domani, su un'email che nessuno ha ancora scritto.
5. **Deploy**: se i numeri convincono, mettere il modello **in produzione**,
   cioè lasciarlo lavorare sul serio, con utenti veri e dati che arrivano ogni
   giorno, e sorvegliarlo, perché i dati del mondo cambiano nel tempo.

La freccia di ritorno è la parte più importante: quasi mai il primo tentativo
è quello buono. Si osserva dove il modello sbaglia, si tornano a ritoccare le
feature o il modello, e si ricomincia il giro.

`````{tab} Elementare

In pratica l'addestramento (in inglese *training*, ed è la parola che si sente
più spesso) è sorprendentemente breve da scrivere. Con una **libreria**, cioè
una cassetta di attrezzi già pronti che qualcun altro ha costruito, addestrare
un modello e usarlo sono due sole richieste: `.fit()` per imparare dai dati,
`.predict()` per prevedere su casi nuovi. La cassetta degli attrezzi si chiama
scikit-learn.

`````

`````{tab} Superiore

Il metodo `fit` implementa la minimizzazione della loss vista sopra; `predict`
applica la $f_{\theta^\star}$ appresa. La separazione tra dati di addestramento
e dati di test serve a stimare la capacità di **generalizzazione**, non la mera
memorizzazione degli esempi già visti.

`````

```python
from sklearn.tree import DecisionTreeClassifier   # un albero di decisione,
                                                  # cioè una catena di domande
                                                  # sì/no: lo vediamo fra poco

# X_train: le feature di ogni esempio, y_train: l'etichetta da prevedere.
# Per convenzione le X sono maiuscole (una tabella) e le y minuscole (una
# sola colonna di risposte); X_test sono gli esempi tenuti da parte.
modello = DecisionTreeClassifier()
modello.fit(X_train, y_train)       # training: il modello impara dai dati
y_pred = modello.predict(X_test)    # previsione su dati mai visti in training
```

## Questi metodi non sono roba da museo

C'è una scena che si ripete in ogni squadra alle prime armi. Arriva un problema
(prevedere quali clienti abbandoneranno il servizio, a partire da una tabella
di età, contratti, consumi, reclami) e qualcuno propone subito una rete
neurale profonda, perché è quella di cui parlano tutti. Passano due settimane
di messa a punto, e alla fine la rete arriva faticosamente a pareggiare un
*gradient boosting*, cioè tanti
piccoli modelli semplici messi in fila, ognuno a correggere gli errori del
precedente. Quel gradient boosting l'aveva addestrato in dieci minuti un
collega scettico, senza toccare nemmeno una impostazione.

```{figure} ../figures/ml-classico-batte-deep-learning.svg
:name: fig-tabellari-vs-non-strutturati
:alt: "Due domini affiancati. A sinistra i dati tabellari, righe e colonne con significati eterogenei, dove i metodi classici basati su alberi restano competitivi. A destra i dati non strutturati, immagini, audio e testo, dove il deep learning domina perché le caratteristiche utili vanno costruite e non sono già nelle colonne."
:width: 100%

Non c'è un vincitore assoluto, c'è un confine. A sinistra i dati già in
tabella, dove le colonne *sono già* le caratteristiche buone; a destra le
foto, il suono e il testo, dove le caratteristiche vanno ricavate dai puntini
di un'immagine o dall'onda di un suono, ed è lì che le reti profonde non hanno
rivali.
```

C'è un confine, e {numref}`fig-tabellari-vs-non-strutturati` lo disegna. È
quello che la scena di prima ignora ogni volta, e conviene capire dove passa.

`````{tab} Elementare

Sullo schermo c'è la tabella dei clienti, una riga per persona e in colonna
l'età, il codice postale, il reddito annuo, i giga consumati il mese scorso, i
reclami aperti. Fra la colonna del codice postale e quella del reddito non c'è
nessun rapporto: unità diverse, scale diverse, significati diversi. Si possono
scambiare di posto e la tabella dice le stesse cose. Sono **colonne senza
geografia**.

In una fotografia la geografia c'è, perché due puntini vicini appartengono allo
stesso occhio e scambiarli sfigura la faccia. In una frase c'è un ordine, e
spostare una parola cambia chi fa che cosa. Le reti profonde hanno ridefinito
quello che si può fare con immagini, suono e linguaggio proprio perché sono
costruite su quella geografia, con pezzi fatti apposta per i puntini vicini e
altri per capire quali parole si riferiscono a quali. Davanti alla tabella dei
clienti quei pezzi non hanno niente da guardare, e il vantaggio evapora.

I modelli semplici che il collega scettico ha messo in fila lavorano una
domanda alla volta: «i giga consumati sono più di ottanta?», poi «il contratto
è ancora attivo?», poi «il quartiere è Milano?», e avanti così fino a una
risposta. Una catena di domande con risposta sì o no, ciascuna su una colonna
sola, si chiama **albero di decisione**. Che le colonne abbiano unità e scale
diverse non gli dà nessun fastidio, perché non le mescola mai fra loro.

Dentro quella tabella le cose cambiano di scatto. La promozione parte a
cinquanta euro di ricarica. Chi si ferma a quarantanove e novanta non prende i
giga in regalo, chi arriva a cinquanta e un centesimo sì, e di due clienti
quasi identici uno rinnova e l'altro se ne va. Una rete neurale tira
volentieri curve morbide, e attraverso un salto del genere ci passa
arrotondandolo, sbagliando proprio sui clienti a ridosso della soglia.
L'albero quel salto lo fa netto, perché la sua domanda è già «la ricarica
supera i cinquanta euro?», e di soglie così ne mette una dietro l'altra.

Nella stessa tabella parecchie colonne non dicono niente sul problema: il
codice interno del cliente, la data in cui la riga è stata digitata, un campo
che qualcuno ha smesso di compilare due anni fa. La rete se le porta dietro
tutte, e da quel rumore raccoglie qualcosa che scambia per un segnale. L'albero
a ogni passo sceglie una colonna sola, e la sceglie perché divide bene chi
resta da chi se ne va, quindi una colonna muta non la sceglie mai.

Quella gara è stata rifatta su decine di tabelle vere, dando ai due contendenti
lo stesso tempo di messa a punto, e gli alberi hanno vinto. Vale per tabelle
come quella sullo schermo, intorno alle diecimila righe. Su una tabella cento
volte più grande la gara è ancora da correre. Dalla parte opposta, con qualche
migliaio di righe soltanto, una rete profonda ha poco materiale. Sa costruirsi
da sola le caratteristiche buone dai dati grezzi, cosa che un albero non sa
fare, ma di esempi ne vuole tantissimi.

Resta il conto da pagare. Il modello del collega si è addestrato sul suo
portatile e va in servizio così com'è, senza macchine speciali. Prima di
firmare per una rete profonda si mette in piedi il modello semplice e lo si
regola per bene, e molto spesso la risposta è già quella.

`````

`````{tab} Superiore

L'osservazione è stata messa alla prova sistematicamente: Grinsztajn, Oyallon e
Varoquaux (NeurIPS 2022) {cite}`grinsztajn2022why` hanno confrontato modelli ad
albero e reti neurali su decine di dataset tabulari, trovando che i primi
restano superiori anche a parità di ricerca degli iperparametri **sui dati di
taglia media**, dell'ordine dei diecimila esempi, che è la scala su cui il
confronto è stato fatto. Fuori da quella scala il confronto resta aperto.

Le ragioni identificate sono strutturali, non contingenti:

1. le reti hanno un *bias induttivo* verso funzioni regolari, mentre i target
   tabulari sono spesso irregolari a tratti, esattamente ciò che una serie di
   split assiali approssima bene;
2. le reti sono sensibili alle feature non informative, di cui una tabella
   reale abbonda, mentre gli alberi le ignorano per costruzione;
3. le reti sono invarianti per rotazione, cioè indifferenti a una mescolanza
   lineare delle colonne, e una tabella invece ha una base naturale, la sua:
   colonne con significati diversi, che non si scambiano.

Il corollario pratico riguarda il **costo**: un gradient boosting si addestra
in minuti su CPU e si mette in produzione senza GPU. Prima di pagare il conto
del deep learning conviene avere una baseline classica ben tarata, e succede
spesso che quella baseline sia già la risposta.

`````

Ed è il motivo per cui il machine learning classico si impara per primo, e non
per ragioni cronologiche. Da qui in avanti valgono sempre le stesse quattro
parole (modello, feature, parametri, loss) e gli stessi due gesti (addestrare,
valutare su dati mai visti): cambieranno i modelli, non la grammatica.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Nel machine learning **non si scrivono le regole**: si danno migliaia di
  esempi già etichettati (le email marchiate «spam» e «non spam») e le regole
  emergono da sole dai dati.
- Su una **tabella** intorno alle diecimila righe gli alberi battono ancora
  regolarmente le reti profonde: fra le colonne di una tabella non c'è quella
  vicinanza che le reti sanno sfruttare fra i puntini di una foto o fra le
  parole di una frase. Su tabelle cento volte più grandi la gara è ancora da
  correre.
- Un programma **impara** (Mitchell) se, facendo pratica, diventa più bravo in
  un compito e questo «più bravo» si può misurare: servono il **compito**,
  l’**esperienza** e la **misura**.
- Tre modi di imparare: con le **soluzioni a fianco** (supervisionato), senza
  etichette, cercando una struttura nascosta (non supervisionato), per
  tentativi e ricompense (per rinforzo).
- Il flusso (dati, feature, modello, valutazione, deploy) è un **ciclo**: la
  valutazione rimanda indietro, e si ricomincia il giro.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Nel machine learning **non si scrivono le regole**: si forniscono esempi e le
  regole emergono dai dati, cioè si stimano i parametri $\theta$ minimizzando
  una loss $\mathcal{L}$ sugli esempi osservati.
- Su **dati tabulari di taglia media**, dell'ordine dei diecimila esempi, i
  modelli ad albero restano superiori alle reti anche a parità di ricerca degli
  iperparametri {cite}`grinsztajn2022why`. Fuori da quella scala il confronto
  resta aperto.
- Le ragioni sono strutturali: il bias induttivo delle reti verso funzioni
  regolari, contro target irregolari a tratti; la loro sensibilità alle feature
  non informative; la loro invarianza per rotazione, dannosa su colonne che non
  sono intercambiabili.
- Un programma **impara** (Mitchell) se la sua performance $P$ su un compito $T$
  migliora con l'esperienza $E$.
- Tre paradigmi: **supervisionato** (dati etichettati), **non supervisionato**
  (struttura nascosta, senza etichette), **per rinforzo** (agente e ricompense).
- Il flusso (dati, feature, modello, valutazione, deploy) è un **ciclo**: la
  valutazione rimanda indietro, e si itera.
```

`````
