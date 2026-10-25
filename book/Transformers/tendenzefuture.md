---
titolo_ricerca: "Transformer: tendenze di ricerca e limiti aperti"
descrizione: "Dove lavora oggi la ricerca sui Transformer, come si fa pensare
  più a lungo un modello sui problemi difficili e quali limiti restano."
---
# Tendenze e limiti

Prevedere l'evoluzione dei Transformer è rischioso, perché in questo campo le
previsioni invecchiano in pochi mesi. Più utile è descrivere le direzioni di
lavoro visibili oggi e i problemi aperti che le motivano. L'architettura
funziona su una gamma larghissima di compiti, dal testo alle immagini al suono,
ma al crescere delle dimensioni richiede sempre più calcolo e più memoria.

## Dove punta la ricerca

I cantieri aperti sono tre. Il primo è fare di più con meno: i grandi modelli
consumano molto calcolo e molta memoria, per cui buona parte della ricerca
punta a ridurne il costo, fino a farli stare in un telefono invece che in un
centro di calcolo. Il secondo è unire i sensi: modelli che leggono, guardano e
ascoltano insieme, come l'assistente a cui mostri una foto e fai una domanda a
voce.

Il terzo è superare i limiti dell'architettura stessa. Nell'attenzione piena
ogni token si confronta con ogni altro, e su una sequenza di $n$ token il costo
cresce come $n^2$, quindi raddoppiando la lunghezza i confronti
quadruplicano. Questo spinge a cercare modi più economici di far comunicare le
parti di un testo. Due filoni attraversano tutti questi cantieri: il
post-training, che è diventato una fase standard dell'addestramento, e il
calcolo che si adatta alla difficoltà della domanda.

Fra i modi di fare di più con meno c'è la distillazione
{cite}`hinton2015distilling`, che invece di alleggerire un modello ne addestra
uno nuovo, piccolo: l'allievo impara a riprodurre l'intera distribuzione di
probabilità di un maestro grande, e non soltanto la risposta giusta, perché le
probabilità piccole dicono quali alternative erano quasi ragionevoli
({numref}`fig-distillazione`). Il meccanismo, con la temperatura che rende
visibili le probabilità più basse, è spiegato nella {doc}`sezione sul modello
piccolo che imita </Efficienza/un-modello-piccolo-che-imita>`.

```{figure} ../figures/distillazione-insegnante-allievo.svg
:name: fig-distillazione
:alt: "Un modello maestro, grande, riceve un input e produce non una sola risposta ma una distribuzione di probabilità su tutte le risposte possibili. Un modello allievo, molto più piccolo, viene addestrato a riprodurre quella distribuzione intera invece della sola risposta corretta."
:width: 92%

Il maestro passa all'allievo l'intera distribuzione, e non la sola risposta:
la risposta giusta dice soltanto qual è, la distribuzione dice anche quali
errori erano quasi ragionevoli. La temperatura $T$ del disegno è la manopola
che rende visibili le probabilità piccole.
```

`````{tab} Elementare
Gli altri modi di rimpicciolire un modello sono due: scrivere ogni numero con
{doc}`meno cifre </Efficienza/meno-bit>`, e togliere i
{doc}`numeri che contano poco </Efficienza/meno-pesi>`. Detti così sembrano
gratuiti, ma non lo sono. Ad esempio, arrotondare ogni numero a quattro bit,
cioè a soli sedici valori possibili, senza altri accorgimenti, sposta di quasi
un quinto l'uscita di uno strato; e una rete a cui si tolgono nove pesi su dieci
smette di funzionare finché non la si riaddestra.

C'è poi una strada che non rimpicciolisce niente. Il modello resta grosso, ma
per ogni parola ne accende soltanto una parte, come una redazione che manda
l'articolo di sport a chi si occupa di sport invece di farlo rileggere a tutti.
Ha le competenze di tutti i suoi redattori, e su ogni articolo ne impegna pochi.

Sul contesto lungo la ricerca prova a far comunicare le parole senza
convocarle tutte insieme, con l’{doc}`attenzione
lineare </AttenzioneLineare/overview>` e con i {doc}`modelli a spazio di stato
</StateSpaceModel/overview>`. Una terza strada,
{doc}`FlashAttention </GPU/flash-attention>`, non tocca il meccanismo e cambia
il modo di eseguirlo: calcola lo stesso risultato a blocchi, senza scrivere in
memoria la tabella intera dei confronti.

Sul fronte dei sensi si costruiscono
mappe del significato condivise, dove una foto di gatto e la parola «gatto»
cadono vicine. Ci si arriva mostrando al modello milioni di immagini con la
didascalia che le accompagna, e chiedendogli di tenere vicine le coppie giuste
e lontane quelle sbagliate. Un altro cantiere è rendere i modelli utili e non
dannosi con il post-training.
`````

`````{tab} Superiore
Sul fronte dell'efficienza rientrano la {doc}`distillazione
</Efficienza/un-modello-piccolo-che-imita>`, la {doc}`quantizzazione
</Efficienza/meno-bit>` (pesi a 8 o 4 bit invece che a 32, dove a quattro bit
la perdita smette di essere trascurabile e servono metodi che facciano più che
arrotondare), la {doc}`potatura </Efficienza/meno-pesi>`, e le architetture
{doc}`mixture of experts <mixture-of-experts>`, che attivano solo una frazione
dei parametri per ogni token. Per il contesto lungo ci sono le attenzioni
{doc}`sparse <confronti>` e {doc}`lineari </AttenzioneLineare/overview>`, le
ottimizzazioni di memoria come {doc}`FlashAttention </GPU/flash-attention>`, e
gli {doc}`state space model </StateSpaceModel/overview>` (Mamba).
Sul fronte multimodale ci sono spazi di rappresentazione condivisi tra testo,
immagini e audio, a partire da CLIP {cite}`radford2021learning`, addestrato in
modo contrastivo su coppie immagine-didascalia; la {doc}`sezione su un solo
spazio per immagini e parole </VisioneLinguaggio/allineare-due-spazi>` lo
costruisce per esteso. Si aggiunge l'allineamento (RLHF, DPO), trattato nel
{doc}`post-training <post-training>`.
`````

## Pensare più a lungo sulle cose difficili

Un Transformer standard spende lo stesso calcolo su ogni token, facile o
difficile che sia: con sessanta strati, ad esempio, i passaggi sono sempre
sessanta. L'idea di togliere quel vincolo risale al 2018. Per anni non ha
preso piede, poi è tornata da una strada diversa.

`````{tab} Elementare

Sessanta piani, e l'ascensore ferma a tutti. La domanda sale in un vassoio, un
bigliettino per parola, a ogni piano qualcuno ci mette mano, e dal tetto esce
la risposta. «Quanto fa due più due» fa il viaggio intero, e un rompicapo da
dieci mosse pure. Noi no: sulle cose facili rispondiamo di getto, sulle
difficili ci fermiamo a pensare.

Nel 2018 Mostafa Dehghani e colleghi provarono a rifare il palazzo, e lo
chiamarono Universal Transformer. Le sessanta stanze sono tutte diverse, e un
paio in più non si aggiungono: quel numero è murato. Se invece la stanza fosse
una sola, e ci si rientrasse? [I giri li deciderebbe la domanda: due per «due
più due», venti per il rompicapo]{.pt-matita}. A ogni giro un bigliettino può
dire «io ho finito», e resta lì com'è mentre gli altri continuano.

Nei muri della stanza non è scritto nessun massimo. Per questo la stanza sola
promette di riuscire anche in compiti più lunghi di tutti quelli visti in
addestramento, quelli che chiedono più passaggi di lavoro di quanti piani abbia
il palazzo.

Manca un pezzo. Un bigliettino che può fermarsi non ha motivo di farlo, girare
è gratis e un giro in più non fa danno. Perciò all'ingresso si paga un
pedaggio, piccolo, a ogni giro: si paga finché la domanda lo merita, poi si
smette.

Sulla porta c'è una promessa più grossa: con questa stanza si può risolvere
qualunque problema che un computer sappia risolvere, dati tempo e carta a
volontà. È vera, ma con una condizione che quasi nessuno legge: giri quanti ne
servono, senza tetto. A quella condizione mantiene la stessa promessa anche il
palazzo di sessanta piani, se la risposta che esce dal tetto si rimette al
pianterreno quante volte si vuole (lo fa un modello che scrive una parola alla
volta) e se le operazioni si eseguono senza mai arrotondare.

Se invece le operazioni sono arrotondate, come nei modelli veri, il palazzo
che risponde al primo passaggio sa fare poco; ogni giro in più gli aggiunge
potenza, e con abbastanza giri arriva a risolvere i problemi che si risolvono
in un tempo ragionevole. A decidere è il numero di giri concessi, più della
forma dell'edificio.

La stanza sola, però, non prese piede: sessanta giri durano quanto sessanta
stanze, perché il lavoro è lo stesso, ma i mobili da scegliere sono quelli di
una stanza sola, e a parità di spesa si impara meno. Intanto tirare su palazzi
più alti funzionava benissimo.

È tornata da una strada inattesa. Anche i modelli che «ragionano» prima di
rispondere spendono di più sulle domande difficili, ma invece di girare a porta
chiusa scrivono i passaggi su un foglio, una riga alla volta. Ogni riga va
prodotta e poi riletta, e questo allunga il lavoro; in compenso si insegna
meglio, perché esistono montagne di quaderni con i passaggi svolti.

Dal 2024 è tornata anche la porta chiusa: c'è chi addestra i modelli a fare i
loro giri senza scrivere niente. Quale strada convenga resta aperto: la porta
chiusa non riempie fogli, il foglio lascia una traccia da leggere. Che la
traccia racconti quanto è successo nella stanza, però, non è detto, e quando lo
si è misurato spesso non lo racconta.

`````

`````{tab} Superiore

L’**Universal Transformer** {cite}`dehghani2019universal` sostituisce gli
$n_{\text{strati}}$ strati distinti con un solo blocco applicato
ricorrentemente in profondità, legando cioè i pesi fra le iterazioni
(l'articolo, del luglio 2018, è stato presentato a ICLR l'anno successivo). La
motivazione dichiarata è recuperare il *bias induttivo* ricorrente che il
Transformer aveva abbandonato insieme alla ricorrenza temporale, e che serve
sui compiti a struttura gerarchica e sulla generalizzazione a lunghezze non
viste in addestramento.

A questo gli autori affiancano l’**Adaptive Computation Time** di Graves
{cite}`graves2016adaptive`: a ogni iterazione,
per ogni posizione, una piccola unità emette una probabilità di
arresto; le posizioni che si fermano vengono copiate invariate mentre le altre
continuano a essere aggiornate, e una penalità sul numero di passi (il *ponder
cost*) impedisce di iterare all'infinito. [Il calcolo diventa così
condizionato all'ingresso invece che fissato dall'architettura]{.pt-matita}.

Gli autori mostrano che legare i pesi e iterare rende il modello
**Turing-completo**, ma lo dicono sotto certe ipotesi, e quella che pesa è
un numero di passi non limitato a priori. Il ragionamento è
che un Transformer standard esegue un numero di passi sequenziali fissato
dall'architettura, mentre una ricorrenza in profondità lo fa dipendere dai
dati.

La Turing-incompletezza del Transformer standard non è però un fatto assodato,
perché la letteratura contiene anche il risultato opposto: Pérez, Barceló e
Marinković {cite}`perez2021attention` dimostrano che il Transformer
encoder-decoder con attenzione dura (il massimo dei punteggi al posto della
softmax) è Turing-completo, con precisione aritmetica arbitraria e un numero
illimitato di passi di decodifica. Quell'ultima ipotesi coincide con la
generazione autoregressiva, la strada che percorrono oggi i modelli lasciati
scrivere finché serve. I due risultati non si contraddicono, perché cambiano le
ipotesi. In definitiva, il numero di passi conta più del vincolo sui pesi, e
l'espressività di un'architettura non si può separare dalle sue ipotesi.

La teoria successiva dà a questa conclusione una forma quantitativa. Con una
precisione aritmetica finita un Transformer di profondità costante è molto più
debole: con una precisione logaritmica nel numero di token si simula con
circuiti a soglia di profondità costante, la classe $\mathsf{TC}^0$, e se
$\mathsf{L} \neq \mathsf{P}$ non sa nemmeno risolvere con esattezza equazioni
lineari {cite}`merrill2023parallelism`.

La catena di pensiero, invece, allarga l'espressività in proporzione ai passi:
con $T$ passi, un Transformer di
profondità costante, precisione costante e dimensione d'embedding
$O(\log n)$ risolve ogni problema calcolabile da circuiti booleani di
dimensione $T$ {cite}`li2024chain`, e con un numero polinomiale di passi di
decodifica, sotto una lieve generalizzazione della pre-normalizzazione,
riconosce esattamente i problemi risolvibili in tempo polinomiale
{cite}`merrill2024expressive`. In altri termini, il calcolo al momento della
risposta acquista espressività, al costo di un token per ciascun passo.

L'idea del calcolo condizionato è tornata da più parti. Dal lato dei token, i
modelli «ragionanti» della {doc}`sezione sul post-training <post-training>` (o1
{cite}`openai2024learning`, settembre 2024; DeepSeek-R1 {cite}`guo2025deepseek`,
gennaio 2025) allocano più calcolo in inferenza generando una catena di passi
intermedi, e di solito ne generano di più sui problemi difficili. Dal lato
latente, Coconut {cite}`hao2024training` (dicembre 2024) rimette in ingresso
l'ultimo stato nascosto invece di decodificarlo in una parola, e il modello a
profondità ricorrente di Geiping e colleghi {cite}`geiping2025scaling`
(febbraio 2025), da 3,5 miliardi di parametri, itera un blocco al momento
dell'inferenza senza dati speciali di catene.

Dentro i Transformer il calcolo condizionato riemerge anche per un'altra via:
Mixture-of-Depths
{cite}`raposo2024mixture` (aprile 2024) decide con un top-$k$ per ciascuno
strato, come il router della {doc}`sezione sui modelli a
esperti <mixture-of-experts>`, quali token debbano attraversarlo, così che la
spesa totale resti fissata mentre quella per singolo token varia.

Fra la via latente e quella in token il confronto resta aperto: il calcolo
latente non
produce token da scrivere e rileggere, e le iterazioni possono condividere la KV
cache, ma non è ispezionabile; quello in token genera più testo, si addestra con
la supervisione esistente e lascia una traccia che si può leggere.

Stabilire se la traccia sia una descrizione *fedele* del calcolo svolto
(proprietà che il {doc}`capitolo
sull'interpretabilità </Interpretabilita/overview>` distingue dalla
plausibilità) è una domanda a sé, e i dati sperimentali indicano che la fedeltà
non è affatto garantita. Le spiegazioni a catena di pensiero possono travisare
la ragione vera di una previsione, fino a far scendere l'accuratezza del 36%
quando il prompt contiene un indizio fuorviante che la catena non nomina
{cite}`turpin2023unfaithful`. Inoltre, quanto il modello si appoggi alla catena
varia molto con il compito, e i modelli più grandi tendono a produrne di meno
fedeli {cite}`lanham2023faith`. Infine, nei modelli che ragionano, nel 2025,
gli indizi usati compaiono nella catena spesso in meno del 20% dei casi
{cite}`chen2025reasoning`.

`````

## I limiti che restano

Accanto agli entusiasmi vanno considerati i limiti, e i limiti si tengono
l'un l'altro. Il primo è il costo: addestramento e inferenza dei modelli
maggiori richiedono molto calcolo e molta energia
{cite}`strubell2019energy` {cite}`patterson2021carbon`, e chi non dispone di
quelle risorse lavora su scala ridotta.

Quel costo cresce con i dati, e i dati sono il secondo limite. Le grandi
raccolte di testo prese dal web (i *corpora*) potrebbero esaurirsi come fonte
di materiale di qualità, perché i dataset di addestramento crescono più in
fretta dello stock di testo umano pubblico. Una stima del 2022, rivista nel
2024, colloca fra il 2026 e il 2032 il momento in cui i modelli saranno
addestrati su raccolte grandi quanto tutto quel testo, se le tendenze restano
quelle; fra le vie d'uscita indicate ci sono i dati sintetici, il
trasferimento da domini ricchi di dati e un uso più efficiente dei dati
{cite}`villalobos2022run`. I corpora portano con sé, inoltre, le distorsioni
sistematiche dei testi scritti online, i bias, che i modelli assorbono
insieme al resto.

Da quei dati il modello impara a stimare la probabilità di un testo, e qui sta
il terzo limite, l'affidabilità. Le allucinazioni (risposte fluenti ma false)
[hanno una causa nell'obiettivo stesso: il pre-addestramento premia la
verosimiglianza]{.pt-matita}, e un'affermazione falsa e fluente non si
distingue da un fatto se i dati non offrono un modo di farlo. La valutazione fa
il resto, perché la maggior parte dei benchmark dà zero punti a un «non lo so»
e premia così chi tira a indovinare {cite}`kalai2025hallucinate`. Mitigarle è
un problema aperto, affrontato con il {doc}`recupero di fonti esterne <rag>`,
la verifica e la {doc}`calibrazione </MachineLearning/metriche>`, cioè
l'accordo fra la sicurezza che il modello dichiara e quanto spesso ha ragione.

Resta aperta la questione di che cosa i modelli *capiscano*. Bender e Koller
{cite}`bender2020climbing` sostengono che dalla sola forma del testo non si
ricavi il significato; Li e colleghi {cite}`li2023emergent` mostrano che un
modello addestrato soltanto su partite di Otello scritte come testo costruisce
al suo interno una rappresentazione della scacchiera, come racconta la
{doc}`sezione sul dibattito </WorldModels/simulatori-e-dibattito>`. Il secondo
risultato riguarda un gioco con regole fisse e non il linguaggio naturale,
quindi non chiude la questione: attribuire ai modelli intenzioni o ragionamento
senza prove resta un errore, e negarglieli senza prove lo è altrettanto.

## Che cosa c'era prima del Transformer, e che cosa no

Il Transformer riunisce idee che c'erano già: l'attenzione fra le parole viene
dalla traduzione automatica neurale {cite}`bahdanau2015neural`, gli embedding
dalle rappresentazioni distribuite del testo, e la discesa del gradiente, che
sistema i parametri un passo alla volta, è ancora più antica. Risale al 2017
l'idea di farne l'unico meccanismo con cui le posizioni comunicano
{cite}`vaswani2017attention`. Su larga scala [i salti sono venuti più da quanto
calcolo e quanti dati si sono potuti spendere che da pezzi nuovi]{.pt-matita}:
la loss scende in modo regolare al crescere di parametri, dati e calcolo
{cite}`kaplan2020scaling`, e a parità di calcolo conta il bilanciamento fra
parametri e dati {cite}`hoffmann2022training`, come spiega la {doc}`sezione
sulle leggi di scala <llm>`. I pezzi aggiunti dopo, dalla KV cache alla mixture
of experts, dal post-training al retrieval, rispondono per lo più a limiti che
la scala ha reso evidenti.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- La ricerca punta su efficienza (la *distillazione*, cioè l'apprendista
  che impara dal maestro; la *quantizzazione*, cioè scrivere ogni numero con
  meno cifre per farlo stare in un telefono; e i modelli che tengono accesa
  solo una parte di sé per ogni parola), contesto lungo (modi più economici
  di far parlare fra loro le parti di un testo, fino ai modelli a spazio di
  stato)
  e multimodalità (leggere, guardare e ascoltare con lo stesso meccanismo).
- Un Transformer standard spende lo stesso calcolo su ogni ingresso, facile o
  difficile che sia. Nel 2018 si provò a togliere quel vincolo con un piano
  solo riapplicato più volte, lasciando a ogni parola il diritto di dire «io ho
  finito» e fermarsi. Allora non prese piede, e l'idea è tornata attuale dalla
  parte opposta: i modelli che ragionano spendono più calcolo sulle difficili
  scrivendo i passi invece di girare a porta chiusa. La prima strada non riempie
  fogli, la seconda lascia una traccia da leggere, che però spesso non racconta
  quanto è successo nel calcolo.
- I limiti sono strutturali: costi di calcolo e di energia, dati di qualità che
  potrebbero finire, bias dei dati, e il fatto che un modello addestrato a
  stimare il probabile non ha, di per sé, un modo di distinguere il probabile
  dal vero. Su che cosa questi modelli «capiscano» il dibattito resta aperto.
- Il Transformer riunisce idee che c'erano già (l'attenzione viene dalla
  traduzione automatica): i salti degli ultimi anni sono venuti più da
  quanto calcolo e quanti dati si sono potuti spendere che da pezzi nuovi.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Le direzioni di frontiera: efficienza (distillazione, quantizzazione a 8
  o 4 bit, pruning, architetture *mixture-of-experts* che attivano solo una
  frazione dei parametri per token), contesto lungo (attenzioni sparse e
  lineari, ottimizzazioni di memoria come FlashAttention, *state space model*)
  e multimodalità (spazi di rappresentazione condivisi fra testo, immagini
  e audio). Si aggiunge l’allineamento (RLHF, DPO).
- Il calcolo condizionato all'ingresso: l’*Universal Transformer*
  {cite}`dehghani2019universal` lega i pesi fra le iterazioni e vi integra
  l’*Adaptive Computation Time* {cite}`graves2016adaptive`, che a ogni
  iterazione assegna a ogni posizione una probabilità di arresto. La
  Turing-completezza che ne segue vale sotto ipotesi, e quella che pesa è il
  numero di passi non limitato a priori.
- Lo stesso filone è tornato da due parti: i modelli che ragionano allungano la
  generazione e scrivono i passi, e la teoria lega l'espressività al numero di
  passi {cite}`merrill2024expressive`; il ragionamento latente
  {cite}`hao2024training` itera senza scrivere i passi. Il costo sono i token
  prodotti, il guadagno è una traccia leggibile, che però non è necessariamente
  fedele al calcolo svolto {cite}`turpin2023unfaithful`.
- Limiti aperti: il costo quadratico $O(n^2)$ dell'attenzione piena e la KV
  cache, che cresce linearmente con i token generati; costi di calcolo e di
  energia, dati che potrebbero esaurirsi e bias dei dati; e lo scarto fra
  massimizzare la verosimiglianza di una continuazione e stabilire che sia
  vera, che la valutazione aggrava premiando chi tira a indovinare
  {cite}`kalai2025hallucinate`.
- Il Transformer riunisce idee preesistenti (l'attenzione viene dalla
  traduzione neurale {cite}`bahdanau2015neural`; del 2017 è l'idea di farne
  l'unico meccanismo {cite}`vaswani2017attention`); i salti recenti sono venuti
  più dalla scala {cite}`kaplan2020scaling` {cite}`hoffmann2022training` che da
  pezzi nuovi.
```

`````

Il costo quadratico nella lunghezza del testo e la {doc}`KV cache
<attenzione-in-pratica>`, il taccuino delle chiavi e dei valori, che non smette
di crescere, restano i due limiti dell'attenzione piena, e da lì riparte il
{doc}`capitolo sull'attenzione lineare
</AttenzioneLineare/overview>`.
