# LLMOps: operare i grandi modelli

Il 30 novembre 2022 OpenAI mette online ChatGPT. Cinque giorni dopo Sam Altman
annota su Twitter che ha superato il milione di utenti. Nello stesso giorno,
alla domanda su quanto costi tutto questo, risponde con una parola diventata
celebre: i costi di calcolo sono *eye-watering*, da far venire le lacrime agli
occhi, e siamo nell'ordine di qualche centesimo di dollaro a conversazione. Il
dettaglio interessante è quello che *non* dice: il modello dietro ChatGPT, un
GPT-3.5, era già pronto da mesi, addestrato e poi rifinito perché rispondesse
come ci si aspetta da un assistente e non come da un completatore di testi. La
cosa nuova, quella che teneva svegli gli ingegneri, non era *costruire* il
modello, era **operarlo**:
servirlo a milioni di persone, in fretta, in modo affidabile, senza che la
bolletta della GPU divorasse l'azienda.

È lo stesso salto che questo capitolo racconta dall'inizio (dal notebook alla
produzione). Ma quando il «modello» è un grande modello linguistico, in sigla
**LLM** (*large language model*), da miliardi di parametri, i problemi visti
finora si ripresentano *amplificati*, e con sfumature nuove. Due
soprattutto. La prima: il modello, spesso, non lo addestri tu. Lo prendi già
fatto (pesi aperti da ospitare, o un'API di terzi da interrogare) e il tuo
lavoro è *adattarlo* e *servirlo*, non allenarlo da zero. La seconda: l'output
non è più una classe o un numero, ma **testo aperto**, difficile da misurare
quanto è difficile giudicare un tema di italiano. Questa sezione è la mappa di
quel territorio; il suo nome, ormai, è **LLMOps**.

## Che cosa cambia con gli LLM

Il **token**, prima di tutto, perché da qui in avanti si conta tutto così, i
tempi come i costi: è il pezzetto di testo (una parola corta, o un frammento di
parola) che il modello legge e scrive come unità.

Detto questo, il baricentro si sposta, e conviene essere precisi su cosa si
sposta dove. Nel resto del capitolo il problema era caricare i pesi dal disco
alla memoria: si fa una volta all'avvio, poi non ci si pensa più. Qui il
problema è un altro viaggio, molto più corto ma molto più frequente: portare
quei pesi **dalla memoria ai circuiti che fanno i conti**, e questo viaggio va
rifatto per intero **a ogni singolo token**. Il modello scrive la risposta un
token alla volta, e ogni token lo decide guardando tutti quelli già scritti (è
il modo di generare, *autoregressivo*, studiato nel capitolo sui Transformer);
a ogni giro, tutti i miliardi di numeri devono ripassare dalla memoria ai
circuiti. È lì che se ne va il tempo, ed è lì che se ne va la bolletta.

```{figure} ../figures/modelli-locali-memoria.svg
:name: fig-cosa-entra-in-memoria
:alt: "Quattro barre orizzontali, lunghe in proporzione alla memoria disponibile (8, 16, 24 e 48 gigabyte), ciascuna con accanto quanti miliardi di parametri ci entrano se ogni peso è scritto a quattro bit: circa 10, 26, 42 e 90. In fondo la regola che produce quei numeri."
:width: 96%

La domanda pratica che precede ogni altra. Non «quale modello è migliore», ma
«quale entra», perché sotto quella soglia nessuna ottimizzazione serve. A
sinistra ci sono i gigabyte di memoria disponibili, a destra i miliardi di
parametri che ci stanno dentro, scritti come si usa nel settore con la B dei
miliardi (`70B` sono settanta miliardi di parametri: attenzione a non
confonderla con la G dei gigabyte, che sta dall'altra parte).

I quattro numeri di destra non vanno imparati, escono dalla riga in fondo, che
è tutta la figura. Si tolgono tre gigabyte, che servono per tenere in memoria
la conversazione in corso, e si divide il resto per mezzo gigabyte a miliardo,
che è quanto occupa un miliardo di parametri **se ogni peso è scritto con
quattro cifre binarie** invece delle solite sedici (il perché è più avanti, in
«Comprimere per servire»). Da 16 gigabyte, per esempio, restano 13, e tredici
diviso mezzo fa 26.
```

Il vincolo di {numref}`fig-cosa-entra-in-memoria` viene prima di tutte le
tecniche di questa sezione, e ne fissa l'ordine. Prima si stabilisce cosa
entra nella memoria che si ha, poi si discute di quanto vada veloce: un
modello che non ci sta non è lento, semplicemente non parte.

`````{tab} Elementare

Fin qui servire un modello era come guidare un'automobile: la accendi, parte,
sterzi dove vuoi, la fermi. Un LLM è un transatlantico. Ha una massa enorme
(miliardi di «manopole» che devono stare tutte in memoria) e ogni manovra
richiede tempo e spazio: non lo parcheggi in garage, non lo giri in una
stradina. Servirlo non è più una questione di *accenderlo*, ma di
*manovrarlo*: dove lo ormeggi (quanta memoria serve per ospitarlo), quanti
passeggeri imbarchi in una volta sola per far quadrare i conti, come eviti che
resti fermo in rada a bruciare carburante mentre aspetta. E c'è un dettaglio
contro l'intuizione: la parte lenta non è pensare, è *ricordare*. A ogni nuova
parola il modello deve rileggere l'intera stiva dei suoi numeri, e quella
rilettura (non il calcolo) è ciò che scandisce il ritmo.

`````

`````{tab} Superiore

La generazione autoregressiva rende l'inferenza di un LLM **memory-bound**,
non compute-bound. Per produrre un solo token il modello deve leggere *tutti*
i suoi pesi dalla memoria della GPU; l'aritmetica per token è modesta, ma il
traffico di memoria è enorme. Un conto d'ordine di grandezza lo rende
concreto: un modello da 7 miliardi di parametri in 16 bit pesa circa 14 GB, e
una GPU con banda di memoria attorno a 2 TB/s impiega
$14/2000 \approx 0{,}007$ s, cioè circa **7 ms**, solo per far scorrere quei
pesi. Una singola sequenza è così limitata a circa $1/0{,}007 \approx 140$
token al secondo, mentre le unità di calcolo restano quasi inattive. A questo
si somma la **KV cache** vista nel capitolo sui Transformer (key e value dei
token già letti, tenuti in memoria per non ricalcolarli) che cresce con la
lunghezza del contesto e va sommata ai pesi. Due grandezze, quindi, governano
tutto: la memoria (contiene pesi e cache) e la sua **banda** (limita quanti
token al secondo si producono). Buona parte dell'ingegneria di LLMOps è lotta
contro questi due limiti.

`````

## Servire un LLM

Riprendiamo la cosa che si è appena detta, perché tutta questa sezione ne
discende: la parte lenta non è pensare, è ricordare. Per scrivere un token il
modello deve rileggersi tutti i suoi numeri, e quella rilettura costa più del
calcolo che ci fa sopra.

Se è così, però, c'è una conseguenza che salva i conti. La rilettura è la
stessa qualunque cosa il modello stia scrivendo. Farla per servire una persona
sola, o per servirne cento nello stesso istante, costa quasi uguale: i pesi
passano una volta e si usano per tutte e cento le risposte in corso. È lo
stesso mazzo di richieste, il *batch*, che nella sezione «Servire un modello»
serviva a tenere occupata la scheda; qui non è un'ottimizzazione fra le altre,
è il motivo per cui un LLM è economicamente sostenibile.

Solo che formare il mazzo, qui, è molto più difficile, e per due ragioni.

```{figure} ../figures/servire-un-llm-vllm-continuous-batching.svg
:name: fig-continuous-batching
:alt: "Confronto fra due modi di riempire la GPU nel tempo. Con il batching statico le richieste partono insieme e il gruppo si libera solo quando la più lunga ha finito: chi termina prima lascia il posto vuoto. Con il continuous batching ogni posto che si libera viene subito riempito da una richiesta in coda, e la GPU resta occupata."
:width: 100%

Lo stesso hardware, due modi di riempirlo. Nel batching statico i riquadri
tratteggiati e vuoti sono GPU pagata e non usata; nel continuous batching una
richiesta entra non appena un posto si libera.
```

La prima è quella che {numref}`fig-continuous-batching` mette in evidenza: le
risposte non durano tutte uguale, e non si sa in anticipo quanto dureranno. Se
si forma il mazzo e lo si tiene insieme fino alla fine (è il **batching
statico**), chi ha finito presto lascia il suo posto vuoto e nessuno lo occupa
finché non ha finito anche il più lento. In un mazzo grande basta una risposta
lunga per tenere fermi tutti gli altri.

La seconda riguarda la memoria. Mentre scrive, il modello tiene degli appunti
su ciò che ha già letto, per non doverlo rileggere da capo a ogni parola nuova:
sono la **KV cache** incontrata nel capitolo sui Transformer. Ogni risposta in
corso porta con sé i propri appunti, e quegli appunti crescono a ogni token
senza che si sappia fin dove. Chi gestisce la memoria si trova quindi davanti a
una scelta scomoda: o riserva a ciascuno lo spazio del caso peggiore, e allora
in memoria ci stanno pochissime conversazioni, o rischia di restare senza
spazio a metà di una risposta.

`````{tab} Elementare

Immagina un ristorante affollato con una sola sala. Il modo ingenuo di gestire
i tavoli fa due errori. Il primo: per ogni comitiva prenoti *in anticipo* un
tavolone lungo, nel caso arrivino altri amici; così mezza sala resta occupata
da sedie vuote «per sicurezza», e la gente in coda alla porta se ne va. Il
secondo: aspetti che un tavolo si liberi *del tutto* prima di far accomodare
qualcuno di nuovo, e intanto le sedie già libere restano inutilizzate.

Un buon maître fa il contrario. Non riserva tavoloni: sistema gli ospiti su
piccoli gruppi di sedie sparsi dove c'è posto, e tiene un foglietto con
scritto chi siede dove; così nessuna sedia resta vuota «nel caso». E appena
una sedia si libera, ci fa accomodare subito la prossima persona in fila,
senza aspettare che se ne vada l'intera comitiva. Più coperti nella stessa
sala significano più clienti serviti nella stessa serata: esattamente ciò che
permette a un LLM di rispondere a migliaia di persone con lo stesso hardware.

Le due mosse del maître hanno un nome, e vale la pena impararli perché sono le
due parole con cui si parla di questa cosa dappertutto.

Rimpiazzare ogni sedia appena si libera, invece di aspettare che il tavolo si
svuoti del tutto, è il **continuous batching**, il mazzo continuo, e sta nella
figura qui sopra contrapposto al batching statico.

Sistemare gli ospiti su piccoli gruppi di sedie sparsi dove c'è posto, con un
foglietto che tiene il conto di chi siede dove, è la **PagedAttention**: gli
appunti di ogni conversazione non stanno più in un blocco unico prenotato in
anticipo, ma in tanti pezzetti tutti della stessa misura, sparsi dove capita,
e un indice dice quali pezzetti appartengono a chi. Così non si riserva niente
«nel caso», e nella stessa memoria ci stanno molte più conversazioni.

`````

`````{tab} Superiore

Il «tavolone prenotato» è la gestione ingenua della KV cache: si riserva un
blocco di memoria contiguo grande quanto il contesto massimo possibile, anche
se la sequenza resterà corta. Ne nascono due sprechi, **frammentazione
interna** (lo spazio riservato e mai usato) ed **esterna** (i buchi fra
blocchi di taglia diversa), che negli approcci precedenti bruciavano tra il
60% e l'80% della memoria della cache. La soluzione di **vLLM**, la
**PagedAttention** {cite}`kwon2023efficient`, prende in prestito un'idea
vecchia di cinquant'anni dai sistemi operativi: la *paginazione* della memoria
virtuale. La cache di ogni sequenza è spezzata in **blocchi di taglia fissa**,
sistemati in modo non contiguo dove c'è spazio, con una *block table* che
mappa posizioni logiche a fisiche: proprio il foglietto del maître. Lo spreco
scende sotto il 4%, e i blocchi possono perfino essere **condivisi** tra
sequenze (un prompt comune, o le ipotesi di una beam search) senza duplicarli.

L'altra metà è il **continuous batching** (o *in-flight batching*): invece di
attendere che tutte le sequenze di un batch finiscano (costringendo le più
brevi ad aspettare la più lunga) lo scheduler lavora a livello di singola
iterazione, e appena una sequenza emette il suo token di fine, un'altra
richiesta ne prende il posto nel batch. La sala resta piena. Insieme,
PagedAttention e continuous batching permettono batch molto più grandi a
parità di memoria: nella misura riportata dagli autori, contro i sistemi che
c'erano allora, un throughput **da due a quattro volte** maggiore a parità di
latenza. Resta il compromesso di
fondo, già incontrato in «Servire un modello»: batch più grandi alzano il
throughput ma allungano la coda della latenza; il punto di equilibrio dipende
dal prodotto.

`````

## Speculative decoding: far indovinare a un modello piccolo

C'è una seconda strada per accelerare la generazione. Non è in concorrenza con
il batching, si somma a quello, e ha una proprietà rara: **non cambia di una
virgola il testo che esce**.

```{figure} ../figures/speculative-decoding-2024.svg
:name: fig-speculative-decoding
:alt: "In alto un modello bozza, piccolo e veloce, genera in sequenza quattro token candidati. In basso il modello grande li verifica tutti insieme in un'unica passata parallela: accetta i primi tre, che coincidono con quello che avrebbe prodotto lui, e al quarto lo corregge scrivendo di suo il token giusto."
:width: 100%

Il modello piccolo tira a indovinare, che gli costa poco; il grande verifica
tutte le sue proposte in un colpo solo, che gli costa quanto scriverne una. Se
il piccolo azzecca quasi tutto, quella singola verifica consegna quattro token
(i tre accettati più quello che il grande scrive di suo) al prezzo di uno: sono
tre giri risparmiati. E se il piccolo sbaglia, la correzione del grande è
comunque quella giusta.
```

La proprietà rara annunciata sopra si legge nella metà inferiore di
{numref}`fig-speculative-decoding`: il modello grande non si fida mai del
piccolo, lo *controlla*. Ciò che viene accettato è solo ciò che il grande
avrebbe prodotto da sé, e per questo il risultato è identico, token per token,
a quello della generazione normale. Cambia il tempo, non il testo.

`````{tab} Elementare

Il collo di bottiglia della generazione non è quanti calcoli fa la GPU, è
quanta memoria deve leggere. Per produrre *un solo* token il modello carica
dalla memoria tutti i suoi pesi, li usa per una manciata di moltiplicazioni e
li scarta. È come accendere un forno industriale per cuocere un biscotto: il
costo dominante è portare il forno a temperatura, non cuocere.

Da qui l'osservazione decisiva: **far passare i pesi per controllare un token o
quattro costa quasi lo stesso**. Generarli in fila costa un giro del modello
ciascuno; controllarli tutti insieme, un giro solo.

La seconda osservazione riguarda il linguaggio: scrivere non è uniformemente
difficile. Dopo «il gatto si è arrampicato sull’» la parola «albero» è quasi
obbligata; dopo «la capitale della Francia è» segue «Parigi». Solo alcuni
punti (una scelta di argomento, un numero, una svolta del ragionamento)
richiedono davvero tutta la potenza del modello grande.

Lo speculative decoding sfrutta le due cose insieme. Un **modello bozza**,
piccolo e veloce, butta giù qualche token in avanti tirando a indovinare. Il
**modello grande** li verifica tutti in una passata sola: accetta il prefisso
su cui è d'accordo e, alla prima divergenza, scarta il resto e corregge lui.

L'analogia è il correttore di bozze: uno stagista scrive in fretta, il
revisore esperto legge un paragrafo intero in un colpo e si ferma al primo
errore, riscrivendo da lì. Se lo stagista è decente si va molto più veloce; se
sbaglia sempre, si torna al ritmo del revisore: mai peggio.

`````

`````{tab} Superiore

Il metodo è dovuto a Leviathan, Kalman e Matias di Google Research
{cite}`leviathan2023fast` e, indipendentemente, a Chen e colleghi di DeepMind
{cite}`chen2023accelerating`. Il passo è:

1. il modello bozza $q$ genera $\gamma$ token in autoregressione;
2. il modello target $p$ valuta le $\gamma+1$ posizioni **in parallelo**, in
   una sola passata, il costo è quello di un forward, non di $\gamma$;
3. ogni token proposto $x_i$ è accettato con probabilità
   $\min\!\bigl(1,\ p(x_i)/q(x_i)\bigr)$; al primo rifiuto si campiona un token
   correttivo dalla distribuzione residua normalizzata
   $\bigl[p(x)-q(x)\bigr]_+$ e si scarta la coda.

Questa regola di accettazione-rifiuto è ciò che rende il metodo **esatto**: la
distribuzione dei token emessi è identica a quella del solo modello target.
Non è un'approssimazione che scambia qualità per velocità: è la stessa uscita,
più in fretta.

Il guadagno dipende dal **tasso di accettazione** $\alpha$: sotto l'ipotesi
semplificatrice (dichiarata dagli autori) che le accettazioni siano
indipendenti con tasso costante $\alpha$, il numero atteso di token per
passata è

$$
\frac{1-\alpha^{\gamma+1}}{1-\alpha},
$$

che per $\alpha=0{,}8$ e $\gamma=4$ dà circa $3{,}4$ token contro $1$. In
pratica si osservano accelerazioni di 2–3 volte. Il modello bozza dev'essere
molto più economico del target e allineato nella distribuzione, altrimenti
$\alpha$ crolla e il costo delle bozze rifiutate mangia il guadagno.

Le varianti *self-speculative* evitano di mantenere un secondo modello, e vanno
distinte proprio sulla proprietà appena rivendicata. Alcune cambiano solo **chi
propone** e tengono la verifica standard, quindi restano esatte: il *prompt
lookup*, che pesca le proposte dal testo già presente nel contesto, ed EAGLE,
che fa proporre le bozze a una testina leggera addestrata sulle rappresentazioni
interne del modello grande. **Medusa**, nella configurazione che propone e
misura, no: al posto del campionamento per rifiuto adotta la *typical
acceptance*, un criterio a soglia scelto apposta per accettare più token al
prezzo di allontanarsi dalla distribuzione del modello target. Il campionamento
per rifiuto resta disponibile anche lì, ma senza il guadagno in più. È un
compromesso legittimo, e va saputo: chi lo adotta credendo di stare ancora nel
metodo esatto sta scambiando qualità per velocità senza essersene accorto.

Un avvertimento pratico: il metodo aiuta nel regime **memory-bound**, cioè
batch piccoli e bassa latenza. A batch molto grandi la GPU è già satura di
lavoro utile e il vantaggio si assottiglia: si combina male, non bene, con la
spinta al throughput del batching visto poco sopra.

`````

## Comprimere per servire

Se il vincolo è la memoria (quanta ce n'è, e quanto in fretta la si legge), la
leva più diretta è far **pesare meno i pesi**. L'idea l'abbiamo già vista in
«Servire un modello»: la **quantizzazione**, cioè riscrivere i decimali
finissimi dei pesi come interi grossolani, arrotondati ai gradini di una scala;
passando da sedici a quattro cifre binarie ogni peso occupa quattro volte meno.
È da qui, per inciso, che veniva il mezzo gigabyte a miliardo della prima
figura.

Sugli LLM questa leva conta doppio, per due ragioni. La prima è che qui il
tempo se ne va nel rileggere i pesi: alleggerirli non fa solo risparmiare
spazio, accorcia *direttamente* il tempo di ogni token. La seconda è che
riaddestrare un modello da centinaia di miliardi di parametri è fuori portata
per quasi tutti, quindi la quantizzazione va fatta **dopo** l'addestramento,
senza rimettere mano alla ricetta originale.

```{figure} ../figures/quantizzazione-modelli.svg
:name: fig-quantizzazione-memoria
:alt: "Barre che confrontano la memoria occupata dallo stesso modello da 7 miliardi di parametri a precisioni diverse: circa 28 gigabyte con numeri a 32 bit, 14 a 16 bit, 7 a 8 bit e 3,5 a 4 bit. Accanto alla barra più corta, l'annotazione che a 4 bit quel modello entra in un portatile."
:width: 92%

Lo stesso modello, quattro ingombri. Le sigle sono i nomi tecnici delle quattro
scritture (quanti bit occupa ciascun numero, e se ha o no la virgola: `FP32`
sono trentadue bit con la virgola, `INT4` quattro bit senza), ma a decidere
sono i gigabyte, e la decisione è un sì o un no: ventotto vogliono una macchina
da centro dati, tre e mezzo entrano in un portatile. Su quel portatile le prime
due righe non sono «più lente», sono impossibili.
```

Quello che {numref}`fig-quantizzazione-memoria` racconta non è un risparmio
graduale, ed è per questo che la quantizzazione conta più di quanto un taglio
del 75% suggerisca. Le quattro barre non sono quattro sconti sempre più
generosi: sono quattro risposte a una domanda che ammette solo sì o no, cioè
«ci sta nella memoria che ho?». Sul portatile da 8 gigabyte della prima figura,
che dopo il margine per la conversazione ne lascia liberi cinque, le prime tre
righe sono tutte e tre un no, e la differenza fra loro non serve a niente:
l'unica che cambia la vita è la quarta.

`````{tab} Elementare

Comprimere i pesi è come preparare un trasloco: la maggior parte delle cose la
schiacci in scatoloni fitti fitti e risparmi un mucchio di spazio. Ma c'è una
regola che chi ha traslocato conosce: alcuni oggetti sono fragili e
*importanti* (i bicchieri buoni, il vaso della nonna) e se li pigi come gli
altri li rompi, e hai rovinato tutto il trasloco per due centimetri di spazio.

Con gli LLM succede una cosa sorprendentemente simile. Fra i miliardi di numeri
del modello, una manciata è «fragile e portante»: se li arrotondi come tutti
gli altri, la qualità crolla. I metodi buoni di compressione fanno proprio come
un buon traslocatore: schiacciano senza pietà la massa dei numeri ordinari, ma
individuano i pochi delicati e li avvolgono con cura, tenendoli più precisi.
Così il modello diventa quattro volte più leggero e resta quasi identico a
prima.

`````

`````{tab} Superiore

La mappa affine $r = S\,(q - Z)$ della sezione sul deployment vale qui
identica. Tre metodi post-training si sono affermati, e condividono
l'intuizione che *non tutti i numeri contano uguale*. Tutti e tre guardano le
**attivazioni**, cioè i numeri che attraversano il modello mentre risponde: si
distinguono per **che cosa** ne fanno, ed è la distinzione che di solito si
perde.

Il primo, **LLM.int8()**, ci cerca dentro i valori anomali
{cite}`dettmers2022llmint8`. Scopre che oltre una certa scala ne emergono, e
che sono concentrati in poche dimensioni: quantizzare quelle dimensioni a 8 bit
come le altre distrugge la qualità. La soluzione è una moltiplicazione di
matrici a **precisione mista**, con la stragrande maggioranza dei valori in
`int8` e le poche dimensioni anomale tenute in 16 bit. Così la qualità regge
fino a 175 miliardi di parametri.

Il secondo, **GPTQ**, le usa per stimare la **curvatura**
{cite}`frantar2023gptq`. Fa passare per il modello un piccolo insieme di testi
di calibrazione, e dagli ingressi di ogni strato ricava la matrice hessiana,
cioè l'informazione del second'ordine che dice quanto l'uscita di quello strato
soffre se un peso si sposta. Poi quantizza uno strato alla volta, un peso dopo
l'altro, correggendo man mano sui pesi rimasti l'errore appena introdotto. Così
scende a **3 o 4 bit per peso**; un modello da 175 miliardi di parametri gli
costa circa quattro ore di GPU, con degrado trascurabile.

Il terzo, **AWQ** (*Activation-aware Weight Quantization*), le usa per decidere
quali **pesi** proteggere {cite}`lin2024awq`. I pesi che contano non sono i più
grandi ma quelli attraversati dai valori più grandi, e sono circa l'uno per
cento: i canali salienti. La mossa poi non è tenerli in 16 bit, è riscalarli
prima di quantizzarli, e questo evita sia la retropropagazione sia la
ricostruzione su un obiettivo di regressione: è l'obiezione che gli autori di
AWQ muovono a GPTQ, cioè che aderendo al proprio insieme di calibrazione rischi
di generalizzare peggio fuori da quello.

Un'avvertenza sul «circa quattro volte». A 8 bit per tensore i due scalari di
calibrazione sono trascurabili e il conto torna esatto; a 4 bit non più, perché
né GPTQ né AWQ usano una scala per tensore, ma **gruppi** di pesi (tipicamente
128), ed è proprio quella granularità a rendere i 4 bit praticabili. Ogni
gruppo si porta dietro la sua scala e il suo zero (mettiamo sedici bit per la
prima e quattro per il secondo, venti in tutto): spalmati su 128 pesi fanno
$4{,}16$ bit effettivi per peso, su 32 pesi ne fanno $4{,}62$. Il rapporto
reale rispetto ai 16 bit è quindi $3{,}8\times$, non $4\times$: qualche punto
percentuale di bit in più, speso per comprare qualità.

Il compromesso è sempre lo stesso (meno bit significano meno memoria e più
velocità, ma più rischio per la qualità) e vale la regola d'oro della sezione
sul deployment: la quantizzazione va **misurata** su dati di validazione, mai
data per gratuita. I 4 bit sono il punto in cui, per la maggior parte dei
modelli densi, il rapporto fra risparmio e degrado cambia segno: scendendo
sotto, il degrado cresce più in fretta di quanto si risparmi.

`````

### L'altra leva: togliere pesi invece di accorciarli

La quantizzazione scrive gli stessi pesi con meno cifre. La **potatura**
(*pruning*) fa una cosa diversa: ne butta via una parte.

Quanto se ne possa buttare, e perché toglierne il novanta per cento non renda
un modello dieci volte più veloce, li costruisce il capitolo sull'efficienza,
insieme alla distinzione fra sparsità non strutturata e strutturata e agli
schemi a densità fissa che l'hardware sa eseguire. Qui interessa la parte che
di là non c'è, ed è quella che rende gli LLM un caso a sé.

`````{tab} Superiore

Sugli LLM la potatura post-training è più delicata che sulle reti di visione,
perché non si può riaddestrare. **SparseGPT** e **Wanda** affrontano proprio
questo: il secondo, in particolare, sceglie cosa togliere pesando ogni peso
per la norma dell'attivazione corrispondente; lo stesso principio di AWQ, che
i pesi importanti si riconoscono guardando cosa ci passa attraverso, non
quanto sono grandi. Con questi metodi il $50\%$ di sparsità è raggiungibile
senza riaddestramento e con degrado contenuto; oltre, il conto si fa salato.

`````

La ragione per cui su un LLM la potatura è più difficile è tutta in quella
riga: **non si può riaddestrare**. Il capitolo sull'efficienza mostra che il
riaddestramento è la parte non opzionale della potatura, quella che riporta la
rete dov'era; qui non c'è, perché riaddestrare un modello da miliardi di
parametri non è una cosa che si fa a valle di un deploy. Da qui la regola
pratica: a parità di rischio si comincia dalla quantizzazione, che il
riaddestramento non lo chiede. E vale la regola di sempre, **misurare**, perché
il degrado si distribuisce in modo diseguale fra i compiti e una media
aggregata lo nasconde.

## Valutare l'invalutabile

Un modello servito e compresso va poi tenuto d'occhio: funziona ancora bene? E
qui casca l'asino, perché tutti i modi consueti di dargli un voto qui si
rompono.

Il primo è la misura che il modello porta con sé, la **perplessità**, vista nel
capitolo sui Transformer: dice quanto il modello è indeciso a ogni token, ed è
come contare le facce del dado che gli servirebbe per tirare a indovinare al
suo posto. Due facce vuol dire che esita fra due parole, mille facce che non ne
ha idea. È una misura ottima mentre il modello impara la lingua, ma non dice
quasi niente di ciò che conta poi: la risposta è *utile*? *corretta*? *ben
scritta*?

Il secondo sono gli esami standard, i **benchmark**, sempre visti nel capitolo
sui Transformer, che sono compiti in classe uguali per tutti i modelli. Vanno
letti con un sospetto preciso, quello della **contaminazione**: se le domande
dell'esame erano già finite dentro i testi su cui il modello si è allenato, il
suo bel voto non dice niente, perché quelle domande le aveva già viste.

E poi c'è il problema di fondo, che nessuno dei due risolve: per una richiesta
aperta («scrivi una mail di scuse al cliente») non esiste *la* risposta giusta
con cui confrontarsi.

```{figure} ../figures/llm-as-judge.svg
:name: fig-llm-giudice
:alt: "La stessa domanda viene posta a due modelli diversi, che producono due risposte. Le due risposte, insieme alla domanda, vengono passate a un terzo modello che fa da giudice e dichiara quale preferisce. Nessun riferimento assoluto entra nel confronto: si stabilisce solo un ordine fra le due."
:width: 92%

Nessuna risposta giusta, solo un confronto. Il giudice non dice se una
risposta è corretta: dice quale delle due preferisce, ed è una domanda a cui
si può rispondere anche quando la prima non ha risposta.
```

Il cambio di domanda in {numref}`fig-llm-giudice` è ciò che rende il metodo
praticabile, e insieme ciò che ne fissa i limiti. Un ordine fra due risposte si
può stabilire senza un riferimento assoluto. Ma un giudice che *preferisce*
porta con sé i propri gusti, e quei gusti si ripetono sempre uguali: premia chi
gli è stato presentato per primo, premia chi scrive di più, premia chi scrive
come scriverebbe lui. Non sono errori di programmazione, che qualcuno prima o
poi correggerà: sono la conseguenza di aver chiesto una preferenza invece di
una verifica.

`````{tab} Elementare

Chi corregge il tema, se non c'è una sola risposta esatta? A scuola lo fa un
insegnante, che legge, soppesa, dà un voto. Ma un insegnante costa tempo, e di
temi da correggere ne arrivano migliaia al minuto. La scorciatoia è affidare la
correzione a un altro modello: uno studente molto bravo promosso a esaminatore,
che legge la risposta e le dà un voto in un lampo, a costo quasi nullo.

Funziona, ma con difetti da tenere a mente, perché questo esaminatore ha le
sue manie. Tende a dare il voto più alto al **primo** tema che legge, a parità
di tutto il resto, solo perché viene prima. E premia il tema più **lungo e
prolisso**, scambiando l'abbondanza di parole per competenza, anche quando una
risposta breve e centrata sarebbe migliore. Non è un giudice imparziale: è un
correttore rapido ed economico con dei pregiudizi sistematici; comodissimo,
purché si sappia di quali difetti soffre e non lo si prenda per oro colato.

`````

`````{tab} Superiore

Il pattern si chiama **LLM-as-a-judge** {cite}`zheng2023judging`: si usa un
modello forte (nel lavoro originale, GPT-4) per assegnare un punteggio o per
scegliere la migliore fra due risposte. Zheng e colleghi lo validano su due
banchi di prova (**MT-Bench**, ottanta domande a più turni, e **Chatbot
Arena**, confronti a coppie raccolti dal pubblico e aggregati con un punteggio
Elo) e misurano che il giudice-GPT-4 concorda con le preferenze umane oltre
l’**80%** delle volte: lo stesso livello di accordo che due esseri umani hanno
fra loro. Il giudice automatico non è però neutro, e i suoi bias hanno nomi
precisi: il **position bias** (tende a preferire la risposta presentata per
prima), il **verbosity bias** (favorisce le risposte lunghe) e il
**self-enhancement bias** (predilige lo stile dei modelli affini a sé). Alcuni
si mitigano (per il position bias, valutare entrambi gli ordinamenti e
mediare) ma nessuno sparisce del tutto. È la stessa lezione del reward model
del capitolo sui Transformer: un giudice appreso è un surrogato del giudizio
umano, e ottimizzare troppo contro un surrogato porta al *reward hacking*.

`````

Alla valutazione si affianca, quando il servizio è acceso, la **sicurezza di
ciò che esce**. Al modello, durante l'addestramento, si è già insegnato quali
risposte sono preferibili e quali no: sono le due tecniche del capitolo sui
Transformer che là si chiamano per sigla, RLHF e DPO. Quell'insegnamento lo
rende meno incline a rispondere in modo dannoso, ma non offre garanzie: restano
una disposizione appresa, e una disposizione si aggira. Per questo i sistemi
reali aggiungono dei **guardrail**, che in italiano sono proprio i guard rail
dell'autostrada: filtri e classificatori indipendenti dal modello, che
ispezionano quello che entra e quello che esce. Servono a bloccare contenuti
dannosi e dati personali, ma anche due mosse che hanno un nome preciso: le
istruzioni nascoste dentro un testo che il modello deve leggere, scritte
apposta perché le prenda per ordini (la *prompt injection*), e i tentativi di
farsi dire ciò che il modello non dovrebbe dire, aggirandone le regole (il
*jailbreak*). Nessuno di questi strumenti è perfetto; messi insieme, riducono
il rischio senza azzerarlo.

## Il ciclo LLMOps

Tirando le somme di questa sezione: l'anello dell'MLOps torna intatto (dati,
addestramento, valutazione, consegna, sorveglianza), ma con gli LLM cambia
quello che ci gira dentro, cioè le cose di cui si conserva ogni versione.
Spesso non sono i pesi, che arrivano già fatti da qualcun altro: è il
**prompt**, la riga d'istruzione con cui si spiega al modello che cosa deve
fare. Quella riga è codice a tutti gli effetti, ed è fragile come abbiamo visto
nel capitolo sui Transformer, dove basta una parola diversa per cambiare la
risposta: quindi se ne conserva ogni versione, la si prova, e prima di
sostituirla si mettono in campo la vecchia e la nuova su due metà del pubblico
per vedere quale funziona meglio (è il test *A/B* della sezione sul
monitoraggio). Il
monitoraggio, a sua volta, insegue bersagli nuovi: le **allucinazioni**
(risposte sicure di sé e sbagliate), la **deriva** dell'uso rispetto a ciò per
cui il sistema era tarato, e il **costo per token**, che scala con quanto
testo entra ed esce; un prompt gonfio è una bolletta più salata. E poiché il
testo aperto non si collauda con i test unitari del software classico, serve
una **valutazione continua**: una batteria di esempi che gira a ogni cambio di
prompt o di modello, spesso con l'LLM-as-a-judge a fare da metro automatico.

Resta fuori, di proposito, tutto ciò che sta *sopra* il modello: ancorare le
risposte a documenti recuperati al momento (il *retrieval-augmented
generation* nella sua forma avanzata), far usare al modello strumenti esterni,
comporre più passi in un **agente**. Non è un dettaglio di serving: è un
capitolo a sé, quello sugli **Agenti**, che abbiamo già percorso.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Con un grande modello linguistico **la parte lenta non è pensare, è
  ricordare**: per scrivere una sola parola il modello deve rileggersi tutti i
  suoi numeri, e sono miliardi. Il calcolo, in confronto, è quasi fermo.
- La prima domanda non è quale modello sia migliore, è **quale ci sta** nella
  memoria che si ha: se non ci sta, non è lento, proprio non parte.
- Servendo tante richieste **insieme** quella rilettura si paga una volta per
  tutte: è il motivo per cui un buon maître non riserva tavoloni e riempie ogni
  sedia appena si libera.
- Si può far **indovinare in anticipo** un modello piccolo e far verificare al
  grande in blocco quello che ha indovinato: se il piccolo azzecca si va molto
  più veloci, e quello che esce è comunque parola per parola ciò che avrebbe
  scritto il grande.
- Per farlo entrare si **comprime**: si arrotondano i numeri, o se ne buttano
  via una parte. Ma alcuni numeri sono fragili e portanti, come i bicchieri
  buoni in un trasloco, e vanno trattati a parte.
- **Giudicare un testo aperto** non ha una risposta esatta: si usa un altro
  modello come esaminatore, comodo ed economico, sapendo che ha dei
  pregiudizi (premia chi risponde per primo e chi scrive di più).
- Quello che qui si conserva versione per versione non sono i pesi, che spesso
  arrivano già fatti: è **l'istruzione con cui si parla al modello**, che è
  fragile come il codice e come il codice va provata prima di sostituirla.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Con gli LLM il collo di bottiglia si sposta sulla **generazione
  autoregressiva**: l'inferenza è **memory-bound** (leggere i pesi domina sul
  calcolo), e la **KV cache** vista nel capitolo sui Transformer occupa memoria
  che cresce col contesto.
- **Servire** un LLM significa **batchare** tante richieste per ammortizzare
  la lettura dei pesi: la **PagedAttention** di vLLM pagina la KV cache come un
  sistema operativo, e lo spreco passa dal 60–80% a meno del 4%
  {cite}`kwon2023efficient`; il **continuous batching** tiene il batch sempre
  pieno. Insieme, nella misura degli autori, da due a quattro volte di
  throughput a parità di latenza.
- Lo **speculative decoding** è **esatto** grazie alla regola di
  accettazione-rifiuto {cite}`leviathan2023fast`, e non tutte le sue varianti
  lo restano: EAGLE e il *prompt lookup* sì, Medusa no, perché la *typical
  acceptance* scambia la distribuzione del target per un tasso di accettazione
  più alto.
- **Comprimere per servire**: quantizzazione *post-training* (riaddestrare è
  fuori portata) con la mappa affine $r = S(q - Z)$ della sezione sul
  deployment, ma **per gruppi** di pesi, non per tensore: a 4 bit i bit
  effettivi sono circa 4,2. I tre metodi guardano tutti le attivazioni e ne
  fanno cose diverse: **LLM.int8()** ci isola le dimensioni anomale
  {cite}`dettmers2022llmint8`, **GPTQ** ne ricava l'hessiana su un insieme di
  calibrazione e scende a 3–4 bit {cite}`frantar2023gptq`, **AWQ** le usa per
  scegliere l'1% di pesi da proteggere {cite}`lin2024awq`. Sempre **da
  misurare**.
- **Valutare l'invalutabile**: la perplessità non basta e i benchmark si
  contaminano; per l'output aperto si usa **LLM-as-a-judge**, che concorda con
  l'uomo oltre l'80% delle volte {cite}`zheng2023judging`, coi suoi bias (di
  posizione, di verbosità, di auto-preferenza). In produzione servono
  **guardrail** su ingresso e uscita.
- Il **ciclo LLMOps** versiona i **prompt** come codice e monitora
  **allucinazioni**, **deriva** e **costo per token**, con **valutazione
  continua**. RAG avanzato, *tool use* e **agenti** hanno un capitolo dedicato,
  che abbiamo già percorso.
```
`````
