# Una rete, cento lingue

La sezione sulla traduzione neurale si era fermata al 2016, con GNMT che
entrava in produzione: un modello per ogni coppia di lingue, addestrato sui
suoi corpora paralleli. Pochi mesi dopo lo stesso gruppo prova una cosa che
sembra solo un risparmio di ingegneria: invece di decine di modelli, uno solo,
addestrato su tutte le coppie insieme, con un gettone in testa alla frase a
dire in che lingua si vuole l'uscita.

Il risparmio arriva, ma con un effetto collaterale che nessuno aveva ordinato.
Il modello aveva visto giapponese verso inglese e coreano verso inglese, mai
giapponese verso coreano. Gli si chiede giapponese verso coreano, e traduce
{cite}`johnson2017google`. Non benissimo, ma traduce: una coppia su cui non ha
mai visto un solo esempio. Gli autori parlano, con prudenza, di qualcosa che
somiglia a una **interlingua**: una rappresentazione interna dove il
significato di una frase finisce più o meno nello stesso posto qualunque
lingua la vesta.

Quella prudenza non è ancora del tutto superata, e questa sezione racconta
perché. È la sezione più utile a chi legge un libro italiano su una tecnologia
addestrata in prevalenza in inglese.

## Un vocabolario per tutte

Il modo più economico di fare un modello multilingue è non fare niente di
speciale.

`````{tab} Elementare

Prendi BERT, quello degli esercizi a buchi della sezione precedente, e invece
di dargli da leggere la Wikipedia inglese dagli da leggere cento Wikipedia
insieme. Non gli dici in che lingua è la frase che sta leggendo. Non gli dai
un elenco delle lingue. Non gli fai vedere nemmeno una traduzione. Gli chiedi
sempre e solo di indovinare le parole coperte, in un pentolone dove ogni
tanto la frase è in italiano, ogni tanto in turco, ogni tanto in coreano.

Una cosa sola cambia davvero, ed è la scatola dei mattoncini: il vocabolario di
pezzi di parola è **uno solo per tutte le lingue**. Chi impacchetta i pezzi
guarda cento lingue insieme e tiene i più utili complessivamente, così molti
pezzi finiscono per essere condivisi (i numeri, i nomi propri, le radici
latine e greche, i prefissi che l'italiano e lo spagnolo hanno in comune).

Resta un problema di dosi. Se dai al modello i testi in proporzione a quanti
ce ne sono, l'inglese sommerge tutto e le lingue piccole spariscono: la
Wikipedia inglese ha ordini di grandezza più voci di quella in curdo. Allora
si bara sulle proporzioni, e si bara in modo controllato.

`````

`````{tab} Superiore

**mBERT** ha esattamente l'architettura di BERT {cite}`devlin2019bert` e
l'esercizio di BERT (*masked language modeling*: mascherare circa il 15% dei
token e predirli). Cambiano solo i dati, le Wikipedia di 104 lingue, e il
vocabolario, un unico WordPiece da circa 110 000 pezzi condiviso fra tutte.
Nessun identificativo di lingua in ingresso, nessun corpus parallelo, nessun
termine di perdita che chieda di allineare qualcosa. Lo chiameremo
$\text{mMLM}$, per distinguerlo dal MLM monolingue: è la stessa funzione,

$$
\mathcal{L}_{\text{mMLM}} = -\sum_{i \in \mathcal{M}_x} \log P\big(x_i \mid x_{\setminus \mathcal{M}_x}\big),
$$

dove $\mathcal{M}_x$ è l'insieme delle posizioni mascherate e
$x_{\setminus \mathcal{M}_x}$ la frase con i buchi, applicata a un flusso di
testi di lingue diverse messi in fila senza dichiararlo.

Lo squilibrio si corregge con un **ricampionamento a smorzamento
esponenziale**. Se la lingua $i$ vale una frazione $p_i$ del corpus totale, si
campiona invece con

$$
q_i = \frac{p_i^{\,\alpha}}{\sum_j p_j^{\,\alpha}}, \qquad 0 < \alpha \le 1 .
$$

Con $\alpha = 1$ nulla cambia; più $\alpha$ scende, più la distribuzione si
appiattisce verso l'uniforme. Due lingue che stanno nei dati al $99\%$ e
all'$1\%$, con $\alpha = 0{,}3$, vengono campionate all'incirca all'$80\%$ e al
$20\%$: la lingua piccola è sovracampionata di venti volte. È l'unico modo per
cui, quando si costruisce il vocabolario condiviso, alle lingue a bassa
disponibilità tocchino pezzi sensati invece di sole lettere sciolte, che è
esattamente il baratto discusso nella sezione sui tokenizzatori.

`````

## L'allineamento che nessuno ha chiesto

Qui viene la parte interessante, ed è bene isolarla, perché è facile darla per
scontata dopo averla sentita raccontare.

Nel procedimento appena descritto **non c'è nulla che chieda al modello di
mettere vicine le traduzioni**. Nessuno gli mostra mai «il gatto nero salta sul
muro» accanto a «the black cat jumps on the wall». Eppure, a fine
addestramento, le due frasi finiscono in punti vicini dello spazio interno, e
un classificatore addestrato sulle rappresentazioni inglesi funziona su quelle
italiane.

`````{tab} Elementare

L'ipotesi più naturale è che il merito sia dei pezzi condivisi. Italiano e
inglese hanno in comune i numeri, i nomi propri, «computer», «-zione» che
somiglia a «-tion»: il modello troverebbe quei punti d'appoggio, e da lì
tirerebbe su il resto come un ponteggio.

L'ipotesi è ragionevole, e in effetti le lingue che condividono più pezzi si
trasferiscono meglio. Solo che, quando qualcuno l'ha messa alla prova
azzerando **di proposito** la sovrapposizione (rinominando i pezzi di una
lingua in modo che non ne condivida più nemmeno uno), il trasferimento è calato
appena. Il ponteggio, evidentemente, non era quello.

L'altra spiegazione è meno intuitiva e più affascinante: il modello ha una
capacità limitata e cento lingue da imparare, e la strada più economica per
riuscirci non è memorizzarle una per una, è **accorgersi che si somigliano** e
riusare la stessa struttura. Non allinea le lingue perché glielo chiediamo:
allinea per avarizia, perché tenerle separate costerebbe più memoria di quanta
ne abbia.

Detto onestamente: la questione è ancora aperta, e chi vi dice di sapere
esattamente perché mBERT funzioni sta semplificando.

`````

`````{tab} Superiore

Le prove sono di tre tipi, e vale la pena distinguerle perché portano in
direzioni diverse {cite}`pires2019multilingual`.

**Sovrapposizione di vocabolario.** La correlazione fra sovrapposizione dei
sottotoken e trasferimento zero-shot è positiva e robusta su molti compiti
(NER, POS tagging, inferenza, parsing). Ma è correlazione: Karthikeyan e
colleghi {cite}`karthikeyan2020cross` costruiscono lingue sintetiche con
sovrapposizione **nulla** e osservano un calo minimo, e altri mostrano che il
trasferimento avviene comunque purché si rifiniscano tutti gli strati tranne
quello di embedding in ingresso. Il trasferimento sopravvive senza pezzi
condivisi, e nemmeno senza alfabeto condiviso: urdu (in grafia araba) e hindi
(in devanagari) si trasferiscono bene pur non avendo un carattere in comune.

**Architettura.** Ciò che conta è la **profondità**, non il numero di teste: il
trasferimento resta accettabile perfino con una testa sola, mentre crolla con
poche layer. Anche il numero totale di parametri conta meno del numero di
strati.

**Capacità.** L'argomento più curioso è che mBERT trasferirebbe **perché** è
piccolo: la capacità limitata, spartita fra cento lingue, lo costringe a
condividere strutture invece di tenere cento modelli separati in un modello
solo. Se fosse vero, l'allineamento non sarebbe una virtù del metodo ma una
conseguenza della scarsità, e la sezione finale mostrerà che questa lettura ha
un rovescio molto concreto.

Un dato che mette d'accordo tutti: l'allineamento non è uniforme lungo la
pila. Cercando, per una frase in una lingua, la sua traduzione fra molte
candidate in un'altra, sono gli strati **intermedi** a funzionare meglio
(intorno al quinto-ottavo in mBERT), non gli ultimi. Gli strati alti tornano a
specializzarsi sulla lingua; nel mezzo abita quel che c'è di più vicino a una
interlingua. Il che suggerisce che le rappresentazioni siano insieme
*language-agnostic* e *language-specific*, a profondità diverse.

`````

## Allinearlo di proposito

Se l'allineamento emerge da solo, si può anche chiederlo esplicitamente. Le
due strade seguite corrispondono a due granularità: il token e la frase.

`````{tab} Elementare

La prima strada è l'esercizio a buchi fatto su **due frasi appaiate**: si mette
la frase italiana e la sua traduzione inglese una dopo l'altra, e si coprono
parole in tutte e due. A quel punto, per indovinare la parola coperta in
italiano, al modello conviene sbirciare nell'inglese, e viceversa. Non gli si
dice «queste due sono la stessa cosa»: gli si rende conveniente scoprirlo.

La seconda strada lavora su frasi intere e ha una forma che il libro ha già
incontrato, in un capitolo che parlava di tutt'altro. Si prendono due encoder,
si dà a uno la frase italiana e all'altro l'inglese, e si chiede: dato un
mucchietto di frasi da una parte e il mucchietto mescolato delle loro
traduzioni dall'altra, **appaiale**. È la stessa identica figura del capitolo
su visione e linguaggio, quella con la matrice di somiglianza e la diagonale
da massimizzare. Lì i due mucchietti erano immagini e didascalie; qui sono
italiano e inglese. Il meccanismo non cambia di una virgola.

`````

`````{tab} Superiore

**XLM** {cite}`lample2019cross` introduce il *translation language modeling*
(TLM): dato un paio di frasi parallele $(x, y)$, si concatenano e si mascherano
token in entrambe,

$$
\mathcal{L}_{\text{TLM}} = -\sum_{i \in \mathcal{M}_x} \log P\big(x_i \mid x_{\setminus \mathcal{M}_x},\, y_{\setminus \mathcal{M}_y}\big)
\;-\; \sum_{j \in \mathcal{M}_y} \log P\big(y_j \mid x_{\setminus \mathcal{M}_x},\, y_{\setminus \mathcal{M}_y}\big),
$$

cosicché quando il contesto monolingue non basta l'attenzione vada a pescare
nell'altra lingua. È mMLM più un premio all'attenzione cross-lingua, e su
corpora paralleli migliora sensibilmente l'allineamento. Il limite è
altrettanto evidente: richiede testi paralleli, che esistono in abbondanza per
poche decine di coppie e quasi per nulla per le altre. **XLM-R**
{cite}`conneau2020unsupervised` fa la scelta opposta, cioè solo mMLM ma su
CommonCrawl invece che su Wikipedia, ordini di grandezza più testo per un
centinaio di lingue, e mostra che a sufficiente scala di dati il segnale
parallelo esplicito diventa meno necessario.

La via a **doppio encoder** (LaBSE {cite}`feng2022language` e affini) cambia
granularità: non predice token, allinea *frasi*. Con un batch di $N$ coppie
parallele e una loss contrastiva su una matrice di somiglianze $N \times N$,
si massimizza la diagonale e si minimizza il resto. È letteralmente la loss
di CLIP {cite}`radford2021learning`, che il capitolo su visione e linguaggio
disegnerà nella {numref}`fig-clip-matrice`, con la
coppia (immagine, didascalia) sostituita da (frase, traduzione), e ne eredita
tutto: i negativi gratis che crescono come $N^2$, l'importanza del batch
grande, la temperatura. Il prodotto è un embedding di frase confrontabile fra
lingue, che serve a recuperare traduzioni, a ripulire corpora paralleli
raccolti dal web e a cercare in un archivio scrivendo la domanda in un'altra
lingua.

`````

## Rifinire in inglese, usare in italiano

Dal punto di vista di chi lavora, tutto questo serve a una procedura sola, e
conviene enunciarla per intero perché è la ragione pratica per cui i modelli
multilingui esistono.

Si pre-addestra un modello su molte lingue. Lo si rifinisce su un compito
usando i dati etichettati di **una lingua sola**, tipicamente l'inglese,
perché è lì che i dataset stanno. Poi lo si usa su tutte le altre, **senza un
solo esempio etichettato** in quelle lingue. Si chiama *zero-shot
cross-lingual transfer*, e per una lingua come l'italiano, ricca ma non quanto
l'inglese, è spesso la differenza fra avere un classificatore e non averlo.

Due avvertenze, che non si trovano nei tutorial.

La prima: **il fine-tuning consuma l'allineamento**. I pesi che si spostano per
imparare il compito sono gli stessi che tenevano vicine le lingue, e adattare
un modello al POS tagging inglese ne peggiora sensibilmente la capacità di
recuperare frasi parallele. È una forma di dimenticanza catastrofica che
colpisce non il compito precedente ma una proprietà emersa lungo la strada, e
si attenua tenendo in vita l'obiettivo di pre-addestramento durante
l'adattamento.

La seconda: il trasferimento non è uniforme, ed è più facile fra lingue con la
stessa struttura sintattica. Il salto fra lingue soggetto-verbo-oggetto
(italiano, inglese, francese) è molto più agevole di quello verso lingue
soggetto-oggetto-verbo (turco, coreano, giapponese), sia per il parsing sia per
l'etichettatura morfosintattica. Chi misura la resa sull'italiano e ne deduce
la resa «sulle altre lingue» sta misurando la cosa più facile.

## La maledizione della multilingualità

Resta il conto da pagare, e chiude il cerchio con l'argomento della capacità.

`````{tab} Elementare

Se un modello di dimensione fissa deve ospitare più lingue, ciascuna riceve
meno spazio. Per una lingua con pochi testi il baratto conviene: guadagna dal
fatto di somigliare alle vicine più di quanto perda in spazio. Ma continuando
ad aggiungere lingue si arriva a un punto in cui lo spazio manca a tutti, e da
lì in poi ogni lingua in più peggiora un po' tutte le altre, compresa quella
che si voleva aiutare.

Non è un difetto da correggere: è una coperta corta. L'unico modo per
aggiungere lingue senza perderci è ingrandire il modello. Ed è lo stesso
argomento, girato, di quello che spiegava perché l'allineamento emerge:
condividere è economico finché la scarsità costringe a condividere, e diventa
un impaccio quando quello che serviva era distinguere.

`````

`````{tab} Superiore

È la **maledizione della multilingualità** misurata da Conneau e colleghi
{cite}`conneau2020unsupervised`: a capacità fissa, la prestazione media
cross-lingua cresce con il numero di lingue fino a un massimo e poi cala,
perché lo spazio di parametri per lingua si assottiglia. La curva si sposta
verso l'alto e verso destra aumentando i parametri del modello: è un vincolo di
capacità, non di metodo.

Ne discendono due tensioni concrete nel disegno di un modello multilingue. La
prima riguarda il **vocabolario condiviso**: allargarlo dà pezzi migliori alle
lingue piccole ma consuma parametri nella tabella di embedding, che è già una
frazione notevole del modello. La seconda riguarda l'$\alpha$ del
ricampionamento: abbassarlo aiuta le lingue rare e sottrae dati a quelle
comuni, e non esiste un valore giusto in assoluto, solo un valore giusto
rispetto a quali lingue interessano.

Una via d'uscita parziale è smettere di pretendere un modello solo: distillare
un modello monolingue robusto dentro quello multilingue recupera parte del
divario, così come i modelli dedicati a famiglie linguistiche ristrette invece
che a cento lingue insieme.

`````

## Che cosa vuol dire, per l'italiano

Vale la pena tirare le somme dal punto di vista di chi legge.

L'italiano sta in una posizione comoda ma non privilegiata: c'è in tutti i
modelli multilingui, ha abbastanza testo perché il vocabolario condiviso gli
riservi pezzi decenti, e condivide struttura e radici con l'inglese, il che
rende il trasferimento zero-shot particolarmente efficace. In pratica, un
classificatore rifinito su dati inglesi funziona in italiano meglio di quanto
ci si aspetterebbe, ed è la ragione per cui in Italia si costruiscono sistemi
funzionanti senza avere dataset italiani di dimensioni comparabili.

Ma la stessa comodità nasconde due cose. La prima è che *funziona meglio del
previsto* non vuol dire *funziona come in inglese*: il divario esiste, ed è
sistematicamente più grande sui compiti che dipendono dalla morfologia, che in
italiano è più ricca. La seconda è che le lingue davvero minoritarie, incluse
quelle di questo paese, non godono di nulla di tutto ciò: non hanno il testo
per entrare nel vocabolario condiviso, e la maledizione della multilingualità
fa il resto. La copertura linguistica dei modelli è, in buona misura, la
mappa di quali lingue hanno un'abbondante presenza scritta online, che non è la
mappa di quali lingue si parlano.

```{admonition} Da ricordare
:class: important
- Un modello **multilingue** si ottiene senza cambiare nulla
  dell'architettura: stesso esercizio a buchi, testi di molte lingue e un
  **vocabolario di sottotoken condiviso**. Lo squilibrio fra lingue si corregge
  campionando con $q_i \propto p_i^{\alpha}$, che appiattisce le proporzioni.
- L'**allineamento fra lingue emerge da solo**, senza che nessun termine della
  loss lo chieda. Perché, non è del tutto chiaro: la sovrapposizione di
  vocabolario correla ma non è necessaria, la profondità conta più delle teste,
  e c'è chi sostiene che il modello allinei perché la capacità limitata lo
  costringe a condividere.
- Lo si può chiedere esplicitamente: **TLM** maschera coppie di frasi
  parallele; la via a **doppio encoder** allinea frasi intere con la stessa
  loss contrastiva di CLIP, dove al posto di (immagine, didascalia) c'è
  (frase, traduzione).
- Il **zero-shot cross-lingual transfer** (rifinire in inglese, usare in
  italiano) è la ragione pratica di tutto questo. Attenzione: il fine-tuning
  **erode** l'allineamento, e il trasferimento è più facile fra lingue con la
  stessa struttura sintattica.
- La **maledizione della multilingualità**: a capacità fissa, oltre un certo
  numero di lingue ognuna in più peggiora tutte le altre. È un vincolo di
  capacità, e l'unico rimedio pieno è un modello più grande.
```
