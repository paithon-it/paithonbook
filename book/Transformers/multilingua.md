# Una rete, cento lingue

La sezione sulla traduzione neurale si era fermata al 2016, con GNMT (il
traduttore neurale di Google) che entrava in produzione: un modello per ogni
coppia di lingue, addestrato sui *corpora paralleli* di quella coppia, cioè
grandi raccolte di testi già tradotti, frase per frase, da un umano. Pochi mesi
dopo lo stesso gruppo prova una cosa che sembra solo un risparmio di
ingegneria: invece di decine di modelli, uno solo, addestrato su tutte le coppie
insieme, con un gettone in testa alla frase a dire in che lingua si vuole
l'uscita. Il gettone è un *token* come tutti gli altri, uno di quei mattoncini
in cui il testo viene spezzato, e sta lì a fare da etichetta: `<2ko>` davanti a
una frase vuol dire «questa me la vuoi in coreano».

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
Wikipedia inglese ha ordini di grandezza più voci di quella in curdo. Allora si
bara sulle proporzioni, e si bara in modo controllato: alle lingue piccole si
danno più turni di quanti gliene spetterebbero, e a quelle grandi meno, con una
manopola che dice quanto appiattire. Girata a fondo, tutte le lingue avrebbero
lo stesso numero di turni; lasciata ferma, ognuna avrebbe i turni che le
toccano. La si mette in mezzo. Nei fatti mBERT la gira poco (una lingua che vale
l'uno per cento dei testi arriva al quattro), e i modelli venuti dopo, che
puntavano di più sulle lingue rare, la girano molto di più.

`````

`````{tab} Superiore

**mBERT** ha esattamente l'architettura di BERT {cite}`devlin2019bert` e
l'esercizio di BERT (*masked language modeling*: mascherare circa il 15% dei
token e predirli). Cambiano solo i dati, le Wikipedia di 104 lingue, e il
vocabolario, un unico WordPiece da circa 120 000 pezzi condiviso fra tutte.
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
appiattisce verso l'uniforme. Il valore non è universale, ed è bene sapere di
chi è: **mBERT usa $\alpha = 0{,}7$** (il README multilingue ufficiale lo chiama
$S$), XLM {cite}`lample2019cross` $\alpha = 0{,}5$, XLM-R
{cite}`conneau2020unsupervised` $\alpha = 0{,}3$. La differenza non è di
dettaglio. Prendendo due lingue che stanno nei dati al $99\%$ e all'$1\%$: con
l'$\alpha$ di mBERT si campiona al $96\%$ e al $4\%$, cioè la lingua piccola è
sovracampionata di quasi quattro volte; con quello di XLM-R si arriva a $80\%$ e
$20\%$, venti volte. È così che, quando si costruisce il vocabolario condiviso,
alle lingue a bassa disponibilità toccano pezzi sensati invece di sole lettere
sciolte, che è esattamente il baratto discusso nella sezione sui tokenizzatori;
ed è anche la manopola su cui si litiga, perché ogni turno dato a una lingua
rara è un turno tolto a una comune.

`````

## L'allineamento che nessuno ha chiesto

Qui viene la parte interessante, ed è bene isolarla, perché è facile darla per
scontata dopo averla sentita raccontare.

Nel procedimento appena descritto **non c'è nulla che chieda al modello di
mettere vicine le traduzioni**. Nessuno gli mostra mai «il gatto nero salta sul
muro» accanto a «the black cat jumps on the wall». Eppure, a fine
addestramento, succede questo.

Dentro il modello ogni frase diventa una lista di numeri, e una lista di numeri
si può leggere come un indirizzo: due numeri sono un punto su un foglio, tre un
punto nello spazio, settecentosessantotto un punto in un luogo che non si
disegna ma si ragiona allo stesso modo. Chiamiamola **mappa del significato**,
perché la proprietà che conta è quella di una mappa: frasi che vogliono dire
cose simili finiscono in indirizzi vicini. Ebbene, «il gatto nero salta sul
muro» e «the black cat jumps on the wall» finiscono a due passi l'una
dall'altra, pur essendo in due lingue di cui al modello nessuno ha mai
raccontato l'esistenza. E la conseguenza pratica arriva subito: un programma
addestrato a riconoscere qualcosa guardando gli indirizzi delle frasi inglesi
funziona anche sugli indirizzi delle frasi italiane, perché stanno nello stesso
quartiere.

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
direzioni diverse.

**Sovrapposizione di vocabolario.** Che la sovrapposizione dei sottotoken
correli positivamente con il trasferimento zero-shot, e su molti compiti (NER,
POS tagging, inferenza, parsing di dipendenze, question answering), è il
risultato di Wu e Dredze {cite}`wu2019beto`, che misura mBERT su cinque compiti
e trentanove lingue. Ma è correlazione, e due lavori la ridimensionano da
direzioni diverse. Pires e colleghi {cite}`pires2019multilingual` misurano la
stessa quantità su NER e POS e trovano la resa di mBERT **piatta** al variare
della sovrapposizione: resta fra il 40% e il 70% perfino per coppie di lingue
quasi prive di parole in comune, mentre un BERT solo inglese, tenuto come
termine di paragone, dipende direttamente dalla sovrapposizione. Karthikeyan e
colleghi {cite}`karthikeyan2020cross` costruiscono lingue sintetiche con
sovrapposizione **nulla** e osservano un calo minimo. Il trasferimento
sopravvive dunque senza pezzi condivisi, e nemmeno serve l'alfabeto condiviso:
urdu (in grafia araba) e hindi (in devanagari) si trasferiscono al 91% pur non
avendo un carattere in comune.

C'è anche la prova per la strada opposta. Artetxe, Ruder e Yogatama
{cite}`artetxe2020cross` prendono un modello **monolingue**, ne congelano tutto
il corpo e riapprendono **soltanto** la tabella di embedding nella lingua nuova,
con lo stesso esercizio a buchi: il trasferimento avviene lo stesso. Il pezzo
che dipende dalla lingua è quindi l'embedding in ingresso; quel che si
trasferisce è tutto il resto, cioè proprio la parte che nessuno ha toccato.

**Architettura.** Ciò che conta è la **profondità**, non il numero di teste
{cite}`karthikeyan2020cross`: il trasferimento resta accettabile perfino con una
testa sola, mentre crolla con poche layer. Anche il numero totale di parametri
conta meno del numero di strati.

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

La seconda strada lavora su frasi intere e ha una forma che ritroverai più
avanti, in un capitolo che parla di tutt'altro. Si prendono due lettori, si dà a
uno la frase italiana e all'altro l'inglese, e si chiede: dato un mucchietto di
frasi da una parte e il mucchietto mescolato delle loro traduzioni dall'altra,
**appaiale**. Immaginalo come una griglia, le frasi italiane sulle righe e
quelle inglesi sulle colonne: in ogni casella si scrive quanto quelle due si
somigliano, e si allena il modello a far venire i numeri grossi sulla diagonale
(dove ogni frase incontra la sua traduzione) e piccoli dappertutto altrove. È la
stessa identica griglia che disegnerà il capitolo su visione e linguaggio, dove
al posto di italiano e inglese ci sono immagini e didascalie. Il meccanismo non
cambia di una virgola.

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

La prima: **la rifinitura consuma l'allineamento**. I numeri che si spostano
per imparare il compito sono gli stessi che tenevano vicine le lingue. Se
adatti il modello a etichettare le parti del discorso di una frase inglese
(dire per ogni parola se è nome, verbo, aggettivo: si chiama *POS tagging*), poi
quello stesso modello ritrova molto peggio le traduzioni di una frase. È una
forma di quella che il libro ha chiamato **dimenticanza catastrofica**,
imparare una cosa nuova cancellandone una vecchia, con la particolarità che
qui non viene cancellato un compito precedente ma una proprietà emersa per
conto suo lungo la strada; e si attenua continuando a far fare al modello, in
sottofondo, anche il vecchio esercizio a buchi mentre impara il compito nuovo.

La seconda: il trasferimento non è uniforme, ed è più facile fra lingue con la
stessa struttura sintattica. Il salto fra lingue soggetto-verbo-oggetto
(italiano, inglese, francese) è molto più agevole di quello verso lingue
soggetto-oggetto-verbo (turco, coreano, giapponese): sull'etichettatura delle
parti del discorso, l'unica misurata su questa griglia da Pires e colleghi, si
passa da 81,6 punti restando dentro il gruppo SVO a 66,5 uscendone. La tabella
per intero dice però una cosa più precisa di «stessa struttura, salto facile»:
da SOV a SOV si ottiene 64,2, da SOV a SVO 64,0, cioè lo stesso. Non c'è
simmetria, e il caso facile non è «le lingue che si somigliano»: è **SVO verso
SVO**, che è anche il caso in cui casca l'italiano. Chi misura la resa
sull'italiano e ne deduce la resa «sulle altre lingue» sta misurando la cosa
più facile che c'era da misurare.

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

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Un modello **multilingue** non ha niente di speciale nell'architettura:
  stesso esercizio a buchi, cento Wikipedia nel pentolone al posto di una, e
  una sola **scatola di mattoncini** (i pezzi di parola) per tutte le lingue.
  Siccome l'inglese sommergerebbe le altre, si bara sulle dosi in modo
  controllato: alle lingue piccole si dà più turni di quanti ne avrebbero, e
  le proporzioni si appiattiscono verso il pari.
- L'**allineamento fra lingue emerge da solo**: nessuno mostra mai al modello
  una frase accanto alla sua traduzione, eppure le due finiscono vicine.
  Perché, non si sa fino in fondo: i pezzi di parola in comune aiutano ma non
  sono indispensabili (toglierli di proposito costa poco), contano più i piani
  della torre che i lettori in parallelo, e c'è chi sostiene che il modello
  allinei per avarizia, perché tenere cento lingue separate costerebbe più
  memoria di quanta ne abbia.
- Lo si può anche chiedere apertamente, in due modi: coprire parole in una
  frase **e** nella sua traduzione messe una dopo l'altra, così che per
  indovinarle convenga sbirciare nell'altra lingua; oppure dare a due lettori
  un mucchietto di frasi e il mucchietto mescolato delle loro traduzioni e
  chiedere di appaiarle, esattamente come si farà con immagini e didascalie nel
  capitolo su visione e linguaggio.
- Il motivo pratico di tutto questo è **rifinire in inglese e usare in
  italiano**, senza un solo esempio etichettato in italiano. Due avvertenze:
  la rifinitura **consuma** l'allineamento (i pesi che si spostano per
  imparare il compito sono gli stessi che tenevano vicine le lingue), e il
  salto riesce meglio fra lingue costruite allo stesso modo.
- La **maledizione della multilingualità**: lo spazio è una coperta corta.
  Oltre un certo numero di lingue, ognuna in più peggiora un po' tutte le
  altre, e l'unico rimedio pieno è un modello più grande.
```
`````

`````{tab} Superiore
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
`````
