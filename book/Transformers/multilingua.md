# Una rete, cento lingue

Nel 2016 GNMT, il traduttore neurale di Google descritto nella {doc}`sezione
sulla traduzione con le reti </NaturalLanguageProcessing/seq2seq-traduzione>`,
usa ancora reti ricorrenti e un modello per ogni coppia di lingue, addestrato
sui *corpora paralleli* di quella coppia, cioè grandi raccolte di testi già
tradotti, frase per frase, da un umano. Pochi mesi dopo lo stesso gruppo prova
un'alternativa che sembra solo un risparmio di ingegneria: invece di decine di
modelli, uno solo, con la stessa architettura, addestrato su tutte le coppie
insieme, con un'etichetta in testa alla frase che dice in che lingua si vuole
l'uscita. L'etichetta è un *token* come tutti gli altri: ad esempio, `<2es>`
davanti a una frase significa «traducila in spagnolo».

Il risparmio arriva, ma con un effetto collaterale che nessuno aveva previsto.
Il modello aveva visto portoghese e spagnolo solo in coppia con l'inglese, mai
l'uno verso l'altro. Messo alla prova direttamente da portoghese a spagnolo,
traduce {cite}`johnson2017google` con 24,8 punti di BLEU contro i 31,5 di un
modello addestrato apposta su quella coppia: non benissimo, ma senza averne mai
visto un solo esempio. Gli autori parlano, con prudenza, di qualcosa che
somiglia a una **interlingua**: una rappresentazione interna dove il
significato di una frase finisce più o meno nello stesso posto qualunque lingua
la vesta.

Quella prudenza vale ancora: non è del tutto chiaro perché una rete addestrata
su molte lingue finisca per allinearle. La questione pesa soprattutto per chi
lavora in una lingua diversa dall'inglese, dove il trasferimento da una lingua
ricca di dati è una delle fonti da cui un modello ricava le proprie capacità.

## Un vocabolario per tutte

Il modo più economico di costruire un modello multilingue non ha niente
di speciale, e lo segue **mBERT**, il BERT multilingue del 2018: la stessa
architettura e lo stesso esercizio di BERT, su testi di più di cento lingue.

`````{tab} Elementare

Cento Wikipedia versate in un pentolone solo e mescolate. Da lì si tira su una
frase alla volta, con qualche parola coperta, e il modello deve indovinare che
cosa c'è sotto. È l'esercizio a buchi di BERT, e cambia solo il pentolone.
Nessuno gli dice in che lingua è la frase. Nessuno gli dà l'elenco delle
lingue. Nessuno gli fa vedere una traduzione, mai. Ogni tanto sale l'italiano,
ogni tanto il turco, ogni tanto il coreano, e il compito non cambia mai.

Prima però qualcuno prepara la scatola dei mattoncini. I pezzi ammessi (i
*token*) si scelgono all'inizio, una volta per tutte, contando quali frammenti
tornano più spesso nei testi. Le frequenze si contano su cento lingue insieme
invece che su una, e la scatola che ne esce è una sola per tutte, circa
centoventimila pezzi, dentro cui deve stare anche il curdo.

Resta il problema delle dosi. Ogni mestolata è un **turno**, e tocca a una
lingua sola; di turni ce ne sono miliardi. Se si versasse in proporzione ai
testi che esistono, l'inglese sommergerebbe tutto, perché fra la sua Wikipedia
e quella in curdo il divario è di centinaia di volte, e in miliardi di
mestolate il curdo non salirebbe quasi mai. Allora si bara sulle dosi, e si
bara in modo controllato, aggiungendone alle lingue piccole e togliendone alle
grandi, con una manopola che dice quanto appiattire. Girata a fondo, ogni
lingua ha lo stesso numero di turni, il curdo quanti l'inglese. Lasciata ferma,
ognuna ha i turni che le spettano. La si mette in mezzo.

Quanto in mezzo, non c'è accordo. Prendi un pentolone con due sole lingue,
cento frasi in tutto, novantanove inglesi e una curda. mBERT, il primo modello
costruito così, che legge centoquattro Wikipedia, gira la manopola di poco: il
rapporto di novantanove a uno diventa novantasei a quattro, e la lingua piccola
sale a galla quasi quattro volte più spesso di quanto le spetterebbe. I modelli
venuti dopo, più attenti alle lingue rare, la girano molto più a fondo e
arrivano a ottanta contro venti, cioè a venti volte tanto.

Dalla manopola dipende anche che cosa entra nella scatola. Il curdo deve
comparire abbastanza da meritarsi pezzi di parola sensati; se compare poco,
gli toccano lettere sciolte. Ogni turno regalato a una lingua rara viene tolto
a una comune, e su questo si discute.

`````

`````{tab} Superiore

mBERT ha l'architettura di BERT {cite}`devlin2019bert` e ne riprende l'obiettivo
principale, il *masked language modeling* (mascherare circa il 15% dei token e
predirli). Cambiano solo i dati, le Wikipedia di 104 lingue, e il vocabolario,
un unico WordPiece da circa 120 000 token condiviso fra tutte. Non c'è dunque
nessun identificativo di lingua in ingresso, né un corpus parallelo, né un
termine di perdita che chieda di allineare qualcosa. La perdita, che indichiamo
con $\mathcal{L}_{\text{mMLM}}$ per distinguerla da quella monolingue, ha la
stessa forma:

$$
\mathcal{L}_{\text{mMLM}} = -\sum_{i \in \mathcal{M}_x} \log P\big(x_i \mid x_{\setminus \mathcal{M}_x}\big),
$$

dove $\mathcal{M}_x$ è l'insieme delle posizioni mascherate e $x_{\setminus
\mathcal{M}_x}$ la sequenza con i token mascherati. A differenza del caso
monolingue, la funzione è calcolata su un flusso di testi di lingue diverse,
mescolati senza dichiararne la lingua.

Lo squilibrio si corregge con un **ricampionamento a smorzamento
esponenziale**. Se la lingua $i$ vale una frazione $p_i$ del corpus totale, si
campiona invece con

$$
q_i = \frac{p_i^{\,\alpha}}{\sum_j p_j^{\,\alpha}}, \qquad 0 < \alpha \le 1 .
$$

Con $\alpha = 1$ nulla cambia; più $\alpha$ scende, più la distribuzione si
appiattisce verso l'uniforme. Il valore cambia da un modello all'altro: mBERT
usa $\alpha = 0{,}7$ (il README multilingue ufficiale lo chiama $S$), XLM
{cite}`lample2019cross` $\alpha = 0{,}5$, XLM-R {cite}`conneau2020unsupervised`
$\alpha = 0{,}3$. Ad esempio, per due lingue che compaiono nei dati al $99\%$ e
all’$1\%$, l’$\alpha$ di mBERT dà il $96\%$ e il $4\%$, sovracampionando la
lingua piccola di quasi quattro volte; il valore di XLM-R porta le quote
all’$80\%$ e al $20\%$, moltiplicandone la presenza per venti. Lo stesso
campionamento governa il vocabolario condiviso: alle lingue a bassa
disponibilità vengono assegnate sottoparole più lunghe invece di singoli
caratteri, secondo il compromesso discusso nella {doc}`sezione sui tokenizzatori
</NaturalLanguageProcessing/tokenizzatori>`. Ogni quota assegnata a una lingua
rara viene però sottratta a una comune. Il compromesso ha un effetto su ogni
frase: dove il vocabolario condiviso riserva meno sottoparole a una lingua, sale
la *fertilità*, il numero medio di token prodotti per ogni parola del testo, e
lo stesso contenuto occupa più token. Su testi paralleli il rapporto fra le
lunghezze in token di due traduzioni della stessa frase, che Petrov e colleghi
chiamano *premio*, arriva fino a quindici volte {cite}`petrov2023language`; ne
seguono più calcolo, meno testo nella finestra di contesto e, dove l'uso si paga
a token, una tariffa più alta per dire la stessa cosa. Rust e colleghi
{cite}`rust2021good`, che usano la fertilità per misurare i tokenizzatori,
mostrano anche che sostituire il tokenizzatore multilingue con uno monolingue
migliora il modello sulla sua lingua a parità di dati.

`````

## L'allineamento che nessuno ha chiesto

Nell'addestramento di mBERT non c'è nulla che chieda al modello di
mettere vicine le traduzioni. Nessuno gli mostra mai «il gatto nero salta sul
muro» accanto a «the black cat jumps on the wall». Eppure, a fine
addestramento, le due frasi hanno rappresentazioni vicine.

Dentro il modello ogni frase diventa un vettore (in mBERT la media dei vettori
dei suoi token a un dato strato, 768 numeri), cioè un indirizzo: due numeri
sono un punto su un foglio, tre un punto nello spazio, settecentosessantotto un
punto in un luogo che non si disegna ma su cui si ragiona allo stesso modo. Lo
spazio degli indirizzi si legge come una mappa del significato, perché ne ha la
proprietà essenziale: frasi con significato simile finiscono in indirizzi
vicini. Così «il gatto nero salta sul muro» e «the black cat jumps on the wall»
finiscono nella stessa regione, benché al modello nessuno abbia mai detto che
si tratta di due lingue.

La vicinanza va precisata. Le regioni occupate da due lingue sono affiancate e
parallele invece che sovrapposte, e fra l'una e l'altra c'è uno spostamento.
Pires e colleghi {cite}`pires2019multilingual` lo misurano nel modo più
semplice: prendono cinquemila frasi inglesi con accanto la loro traduzione
tedesca, calcolano lo scarto fra le due frasi di ciascuna coppia sulla mappa e
ne fanno la media. Sommato al vettore di una frase inglese, questo scarto medio
la porta così vicino alla sua traduzione che, in tutti gli strati tranne i
primi, per più della metà delle coppie la frase tedesca più vicina fra le
cinquemila è quella corretta. Lo spostamento dipende quindi in gran parte dalla
coppia di lingue e poco dalla singola frase: le due lingue occupano due copie
della stessa mappa, una accanto all'altra.

Lo spostamento sembrerebbe impedire il trasferimento, e invece non lo impedisce.
Se un classificatore impara a riconoscere qualcosa guardando dove cadono le
frasi inglesi (dire se una recensione è arrabbiata, per esempio: impara che gli
arrabbiati stanno da quella parte della regione inglese), lo stesso
classificatore funziona anche sulle frasi italiane, pur senza aver mai visto una
frase italiana e senza che nessuno ne compensi lo spostamento. Una spiegazione
possibile, che nessun lavoro citato dimostra, sta nella direzione di quello
spostamento: porta tutte le frasi italiane dalla stessa parte, scivolando lungo
il confine fra arrabbiate ed entusiaste invece di attraversarlo. Il confine
imparato in inglese, appoggiato sulla regione italiana, cade quindi ancora al
posto giusto. Non ci cade alla perfezione, per cui il trasferimento funziona
bene ma con qualche calo: quanto si perda dipende dalle due lingue, e si misura.

`````{tab} Elementare

L'ipotesi più naturale è che il merito sia dei pezzi condivisi. Italiano e
inglese hanno in comune i numeri, i nomi propri, «computer», «-zione» che
somiglia a «-tion»: il modello troverebbe quei punti d'appoggio, e da lì
tirerebbe su il resto come un ponteggio.

L'ipotesi è ragionevole, e in effetti le lingue che condividono più pezzi si
trasferiscono meglio. Solo che, quando qualcuno l'ha messa alla prova
azzerando di proposito la sovrapposizione (rinominando i pezzi di una
lingua in modo che non ne condivida più nemmeno uno), il trasferimento è calato
appena. Il ponteggio, evidentemente, non era quello.

C'è anche la prova dall'altro capo. Si parte da un modello che ha letto solo
inglese e si congela, cioè si tiene fermo, tutto il modello tranne il primo
passaggio, la tabella che trasforma i mattoncini in numeri. Quel
passaggio si rifà da capo con i mattoncini di una lingua nuova, rimettendo
il modello a fare gli esercizi a buchi in quella lingua. Il trasferimento
avviene lo stesso. La parte specifica di una lingua sta dunque all'ingresso;
quella che passa da una lingua all'altra è tutto il resto, che nessuno ha
toccato.

L'altra spiegazione è meno intuitiva e più affascinante. Il modello ha una
capacità limitata e cento lingue da imparare, e la strada più economica per
riuscirci è accorgersi che si somigliano e riusare la stessa struttura invece
di memorizzarle una per una. Allinea le lingue per avarizia, senza che nessuno
glielo chieda: tenerle separate costerebbe più memoria di quanta ne abbia.

Un indizio a favore viene dallo smontare il modello pezzo per pezzo. Un
modello, ricordiamolo, è una torre di piani uguali, e a ogni piano lavorano in
parallelo diversi lettori con l'evidenziatore (le *teste* di attenzione). Si
può tagliare nell'una o nell'altra direzione, e i due tagli non si
equivalgono: lasciando un lettore solo per piano il trasferimento fra lingue
tiene ancora, mentre togliendo piani peggiora, e tanto più quanti più piani
si tolgono. A costruire la rappresentazione comune alle lingue è più la
profondità della torre che la sua ampiezza.

Che conti la profondità non implica che più si sale, meglio è. Se si ferma la
frase a ogni piano e si chiede di ritrovare la sua traduzione fra tante altre
frasi inglesi, le risposte giuste sono più numerose ai piani di mezzo. In cima
ogni lingua torna riconoscibile come tale e la somiglianza si perde: le lingue
si assomigliano di più a metà della torre che sul tetto.

La questione resta aperta, e chi sostiene di sapere perché mBERT funzioni sta
semplificando.

`````

`````{tab} Superiore

L'algebra dietro la spiegazione possibile sta in una riga. Per un
classificatore lineare $\operatorname{sign}(\mathbf{w}^\top\mathbf{x} + b)$ e
uno scarto medio $\boldsymbol{\delta}$ fra le rappresentazioni delle due
lingue, il punteggio di una frase traslata è
$(\mathbf{w}^\top\mathbf{x} + b) + \mathbf{w}^\top\boldsymbol{\delta}$: la
decisione non cambia se $\mathbf{w}^\top\boldsymbol{\delta} \approx 0$, cioè se
lo scarto è quasi ortogonale al vettore dei pesi e scorre lungo il confine.
Nessuno ha dimostrato che sia sempre così.

Le prove sul perché l'allineamento emerga sono di tre tipi, e portano in
direzioni diverse.

**Sovrapposizione di vocabolario.** Wu e Dredze {cite}`wu2019beto`, misurando
mBERT su cinque compiti e trentanove lingue, riscontrano che la sovrapposizione
dei sottotoken correla positivamente con il trasferimento zero-shot, e questo su
compiti molto diversi fra loro (riconoscimento di entità, POS tagging,
inferenza, parsing di dipendenze, classificazione di documenti). È però una
correlazione, e due lavori la ridimensionano da direzioni diverse. Pires e
colleghi {cite}`pires2019multilingual` mettono in grafico la resa sul
riconoscimento di entità contro la sovrapposizione, per ogni coppia fra sedici
lingue, e la trovano piatta: resta fra il 40% e il 70% di F1 perfino per le
coppie quasi prive di token in comune, mentre un BERT solo inglese, tenuto come
termine di paragone, precipita fino a sfiorare lo zero. Karthikeyan e colleghi
{cite}`karthikeyan2020cross` costruiscono lingue sintetiche con sovrapposizione
nulla e osservano un calo minimo. Il trasferimento sopravvive dunque senza token
condivisi, e nemmeno serve l'alfabeto condiviso: un mBERT rifinito sul solo urdu
(in grafia araba) etichetta le parti del discorso dell'hindi (in devanagari) con
91 punti di accuratezza, pur non avendo mai visto una parola annotata in
devanagari e con una sovrapposizione di vocabolario di fatto nulla. Nel verso
opposto, hindi verso urdu, si scende a 86: la simmetria non è garantita nemmeno
qui.

C'è anche la prova per la strada opposta. Artetxe, Ruder e Yogatama
{cite}`artetxe2020cross` prendono un modello monolingue, ne congelano tutto
il corpo e riapprendono soltanto la tabella di embedding nella lingua nuova,
con lo stesso compito di mascheramento: il trasferimento avviene ugualmente. La
componente che dipende dalla lingua è quindi il solo embedding in ingresso, e il
trasferimento passa attraverso il corpo del modello, rimasto invariato.

**Architettura.** Karthikeyan e colleghi {cite}`karthikeyan2020cross` studiano
BERT bilingui, con il russo e un inglese artificiale (l'inglese con i caratteri
spostati, perché non condivida niente con l'altra lingua), sul compito di
inferenza XNLI e a numero di parametri fissato. Il numero di teste conta poco:
con una testa sola il divario fra le due lingue è di 14,2 punti, con dodici di
13,3. La profondità conta di più: con sei strati il divario sale a 15,2 punti,
con uno solo a 21,6, e con ventiquattro scende a 11,3. Il numero totale di
parametri conta meno della profondità, salvo sotto una soglia minima.

**Capacità.** Dufter e Schütze {cite}`dufter2020identifying` addestrano BERT
piccoli su una lingua e una sua copia artificiale, e trovano che la
multilinguità dipende da pochi elementi, fra cui la scarsità di parametri: un
modello grande quanto BERT-base, su quei pochi dati, ha un indice di
multilinguità più basso (0,58 contro 0,70), e con un'altra modifica lo perde del
tutto. L'interpretazione è che, quando i parametri non bastano a tenere le
lingue separate, il modello sia costretto a condividere strutture; e gli autori
aggiungono che mBERT, con centoquattro Wikipedia da imparare, potrebbe già
trovarsi in quella condizione, senza averlo misurato. Se fosse così,
l'allineamento deriverebbe dalla scarsità di parametri più che dal metodo, e la
scarsità ha un rovescio molto concreto: la maledizione della multilingualità.

Un dato che mette d'accordo tutti: l'allineamento non è uniforme lungo la pila.
Cercando, per una frase in una lingua, la sua traduzione fra molte candidate in
un'altra (dopo averla traslata dello scarto medio $\boldsymbol{\delta}$), sono
gli strati intermedi a funzionare meglio: nei dodici strati di mBERT il massimo
cade fra il sesto e l'ottavo, come si osserva nella Figura 3 di Pires e
colleghi {cite}`pires2019multilingual`, e non in cima. Gli strati alti tornano
a specializzarsi sulla lingua, mentre a metà altezza emerge la configurazione
più vicina a una interlingua. Questo andamento suggerisce che le
rappresentazioni siano insieme *language-agnostic* e *language-specific*, a
profondità diverse.

`````

## Allinearlo di proposito

Se l'allineamento emerge da solo, si può anche chiederlo esplicitamente. Le due
strade seguite si distinguono per la taglia dell'unità da allineare:
la prima lavora sulle singole parole, la seconda sulle frasi intere.

`````{tab} Elementare

La prima strada è l'esercizio a buchi fatto su due frasi appaiate: si mette
la frase italiana e la sua traduzione inglese una dopo l'altra, e si coprono
parole in tutte e due. A quel punto, per indovinare la parola coperta in
italiano, al modello conviene sbirciare nell'inglese, e viceversa. Non gli si
dice «queste due sono la stessa cosa»: gli si rende conveniente scoprirlo.

Le frasi appaiate, però, bisogna averle. Fra l'italiano e l'inglese ce ne sono
milioni, e lo stesso vale per qualche decina di altre coppie di lingue; per
tutte le altre non ce n'è quasi nessuna, e sono le lingue che avrebbero più
bisogno d'aiuto. Da qui la scelta opposta, che funziona lo stesso: niente frasi
appaiate, ma molto più testo qualunque, non le sole Wikipedia ma tutto quello
che si trova sul web. Con abbastanza da leggere le lingue si avvicinano senza
aiuto, e il vantaggio di mostrare al modello le frasi accoppiate si assottiglia.

La seconda strada lavora su frasi intere, e la sua forma tornerà
nell’{doc}`allineamento fra immagini e testo
</VisioneLinguaggio/allineare-due-spazi>`, con immagini e didascalie al posto
di frasi e traduzioni. Ogni frase viene letta senza
che si veda mai l'altra lingua, e ridotta al suo indirizzo sulla mappa; poi si
confrontano gli indirizzi, non le frasi. Si dà da una
parte un mucchietto di frasi italiane e dall'altra il mucchietto mescolato delle
loro traduzioni, e si chiede: appaiale. La griglia ha le frasi italiane
sulle righe e quelle inglesi sulle colonne, e in ogni casella si scrive quanto i
due indirizzi si somigliano; il modello viene allenato a far venire i numeri
grossi sulla diagonale (dove ogni frase incontra la sua traduzione) e piccoli
dappertutto altrove.

Il mucchietto conviene grosso, e la griglia dice perché. Con dieci frasi per
parte le caselle sono cento: dieci sono gli appaiamenti giusti, novanta quelli
sbagliati da respingere. Con cento frasi per parte le caselle diventano
diecimila, e gli appaiamenti sbagliati novemilanovecento. Gli esempi di che cosa
non va appaiato crescono molto più in fretta delle traduzioni che si sono dovute
procurare, e vengono gratis.

Rispetto alla griglia di immagini e didascalie cambia semmai la severità, cioè
quanto nettamente la casella giusta deve vincere. Nel caso delle traduzioni
non le basta prendere il numero più alto della sua riga: deve
prenderlo con un certo scarto su tutte le altre.

`````

`````{tab} Superiore

**XLM** {cite}`lample2019cross` introduce il *translation language modeling*
(TLM): dato un paio di frasi parallele $(x, y)$, si concatenano e si mascherano
token in entrambe,

$$
\mathcal{L}_{\text{TLM}} = -\sum_{i \in \mathcal{M}_x} \log P\big(x_i \mid x_{\setminus \mathcal{M}_x},\, y_{\setminus \mathcal{M}_y}\big)
\;-\; \sum_{j \in \mathcal{M}_y} \log P\big(y_j \mid x_{\setminus \mathcal{M}_x},\, y_{\setminus \mathcal{M}_y}\big),
$$

così che, quando il contesto monolingue non basta, l'attenzione possa attingere
all'altra lingua. In altri termini, TLM è mMLM con in più un incentivo
all'attenzione cross-lingua, e su corpora paralleli migliora sensibilmente
l'allineamento. Il limite è altrettanto evidente: richiede testi paralleli, che
esistono in abbondanza per poche decine di coppie e quasi per nulla per le
altre. **XLM-R** {cite}`conneau2020unsupervised` fa la scelta opposta, cioè solo
mMLM ma su CommonCrawl invece che su Wikipedia, con una quantità di testo
superiore di ordini di grandezza per un centinaio di lingue, e mostra che con un
volume di dati sufficiente il segnale parallelo esplicito diventa meno
necessario.

La via a **doppio encoder** (LaBSE {cite}`feng2022language` e affini) cambia
granularità: non predice token, allinea *frasi*. Con un batch di $N$ coppie
parallele e una loss contrastiva su una matrice di somiglianze $N \times N$, si
massimizza la diagonale e si minimizza il resto. È la stessa forma della loss
illustrata nella {numref}`fig-clip-matrice` per CLIP
{cite}`radford2021learning` nel capitolo su visione e linguaggio, con la coppia
(immagine, didascalia) al posto di (frase, traduzione), e ne condivide il
meccanismo che conta: gli esempi negativi a costo zero, che crescono come $N^2$
e rendono prezioso il batch grande.

LaBSE è del luglio 2020 e CLIP del febbraio 2021, quindi LaBSE non eredita
niente da CLIP. I due modelli non hanno nemmeno lo stesso antenato dichiarato.
LaBSE parte dal doppio encoder con graduatoria sulle traduzioni di Guo e
colleghi {cite}`guo2018effective` e dalla softmax a margine additivo;
l'articolo di CLIP indica invece come fonti la *multi-class N-pair loss* di
Sohn, la InfoNCE di van den Oord e colleghi {cite}`oord2018representation` e
ConVIRT. Li accomuna la forma: una softmax sulle somiglianze di un batch, con i
negativi presi dal batch stesso. Divergono in un dettaglio: dove CLIP scala le
somiglianze con una temperatura appresa, LaBSE sottrae un margine fisso alla
diagonale, cioè chiede alla coppia corretta non solo di prevalere, ma di
superare le altre di uno scarto. Sono due modi diversi per regolare la severità
della loss.

Il prodotto è un embedding di frase confrontabile fra lingue, che serve a
recuperare traduzioni, a ripulire corpora paralleli raccolti dal web e a cercare
in un archivio scrivendo la domanda in un'altra lingua.

`````

## Rifinire in inglese, usare in italiano

Tutto questo serve a una procedura sola, la ragione pratica per cui i modelli
multilingui esistono: lo **zero-shot cross-lingual transfer**, «trasferimento
fra lingue a zero esempi». Si pre-addestra un modello su molte lingue; lo si
rifinisce (*fine-tuning*) su un compito con dati etichettati in una lingua sola,
tipicamente l'inglese, perché lì si concentra la maggior parte dei dataset; e lo
si usa nelle altre senza un solo esempio etichettato in quelle lingue. La
rifinitura sposta i parametri del modello, gli stessi che il pre-addestramento
aveva messo a punto e che decidono dove ogni frase finisce sulla mappa. Per una
lingua come l'italiano, ricca ma non quanto l'inglese, è spesso la differenza
fra avere un programma che funziona e non averlo. Come termine di paragone si
usano due metodi che passano per la traduzione automatica: tradurre nella lingua
d'arrivo i dati etichettati in inglese e rifinire lì (*translate-train*), oppure
tradurre in inglese il testo da giudicare e usare il modello inglese
(*translate-test*), le due righe di confronto che accompagnano i risultati
multilingui fin da XNLI {cite}`conneau2018xnli`.

La prima avvertenza è che la rifinitura consuma l'allineamento, perché i
parametri che si spostano per imparare il compito sono gli stessi che tenevano
vicine le lingue. Liu e colleghi {cite}`liu2020continual` lo misurano rifinendo
un modello multilingue sul *POS tagging* inglese, cioè sull'etichettare ogni
parola di una frase come nome, verbo, aggettivo. Prima della rifinitura,
ritrovare la traduzione inglese di una frase italiana riusciva 45 volte su
cento; dopo, 26. La perplessità sul compito di mascheramento in inglese,
l'indice di quanto il modello resta indeciso sulla parola coperta, peggiora di
venti volte. È **dimenticanza catastrofica**, cioè imparare qualcosa di nuovo
cancellando qualcosa di vecchio, un fenomeno descritto sulle reti neurali fin
dal 1989 {cite}`mccloskey1989catastrophic`. I rimedi mostrano che cosa si è
perso. Rimettere il vecchio compito di mascheramento accanto al compito nuovo
riporta la perplessità dov'era, ma le traduzioni si ritrovano 25 volte su cento,
peggio di prima: l'allineamento non era un semplice prodotto secondario del
mascheramento. Impedire agli errori sul vecchio compito di crescere ne salva una
parte, 32 volte su cento. Per superare il punto di partenza bisogna chiedere
durante la rifinitura la proprietà che si vuole conservare, cioè appaiare le
frasi con le loro traduzioni: con un vincolo che protegge insieme il
mascheramento e l'appaiamento si arriva a 64 volte su cento, e con
l'addestramento congiunto sui due compiti a 74. Tornano così a servire le frasi
appaiate, di cui quasi tutte le lingue sono prive.

La seconda avvertenza è che il trasferimento non è uniforme, e la sua
asimmetria riserva una sorpresa. Le lingue si possono raggruppare per l'ordine
in cui mettono le parole in una frase: l'italiano, l'inglese e il francese sono
lingue **SVO**, soggetto-verbo-oggetto («il gatto morde il cane»); il turco, il
coreano e il giapponese sono lingue **SOV**, soggetto-oggetto-verbo, e
direbbero «il gatto il cane morde». Pires e colleghi
{cite}`pires2019multilingual` hanno misurato il trasferimento in tutte e
quattro le combinazioni possibili, sempre sul POS tagging, e il punteggio è la
percentuale di parole etichettate correttamente, mediata su tutte le coppie di
lingue diverse di ciascun gruppo. Rifinendo su una lingua SVO e valutando su
un'altra SVO si ottiene 81,6; rifinendo su una SVO e valutando su una SOV si
scende a 66,5.

Fin qui sembra la solita morale, «lingue simili, salto facile», che è anche
quella degli autori. Le altre due combinazioni mostrano però la debolezza della
regola: rifinendo su una lingua SOV e valutando su un'altra SOV si ottiene 64,2,
mentre rifinendo su una SOV e valutando su una SVO si ottiene 64,0. Due decimi
di scarto, contro i quindici punti che separano le due combinazioni di prima. Ne
segue che condividere l'ordine delle parole aiuta molto quando si parte da una
lingua SVO e quasi niente quando si parte da una SOV: conta soprattutto da dove
si parte. L'italiano ricade nel caso più favorevole; chi misura la resa
sull'italiano per dedurne quella «sulle altre lingue» sta misurando il caso più
facile.

I due gruppi, però, non differiscono solo per l'ordine delle parole. Quello SVO
raccoglie ventisei varietà, e comprende le due varianti di norvegese, le due di
portoghese e famiglie intere di lingue imparentate; quello SOV ne raccoglie
dieci, in gran parte non imparentate e scritte in sette alfabeti diversi (ma
hindi e urdu sono la stessa lingua in due grafie). Parentela, alfabeto condiviso
e quantità di testo disponibile agiscono tutti nella stessa direzione
dell'ordine delle parole, per cui quei quattro numeri non consentono di isolarne
l'effetto.

## La maledizione della multilingualità

Lo spazio di un modello è finito, e le lingue se lo dividono.

`````{tab} Elementare

Se un modello di dimensione fissa deve ospitare più lingue, ciascuna riceve
meno spazio. Per una lingua con pochi testi il baratto conviene: guadagna dal
fatto di somigliare alle vicine più di quanto perda in spazio. Continuando
ad aggiungere lingue si arriva a un punto in cui lo spazio manca a tutti, e da
lì in poi ogni lingua in più peggiora un po’ tutte le altre, compresa quella
che si voleva aiutare.

Lo spazio se lo contendono anche cose diverse. Allungare l'elenco dei
mattoncini dà pezzi più sensati alle lingue piccole, ma ogni mattoncino in più
occupa un posto nello stesso modello e lo toglie a tutto il resto;
girare la manopola delle dosi verso le lingue rare le aiuta e sottrae turni alle
comuni. Una regolazione giusta in assoluto non esiste; esiste quella giusta
rispetto alle lingue che interessano.

Nessuno ha sbagliato niente: la coperta è corta. L'unico modo di aggiungere
lingue senza rimetterci è ingrandire il modello, e la strada più diretta è
dare a ogni lingua, a ogni piano, un pezzetto di macchina tutto suo, che si
accende solo sui suoi testi, lasciando in comune il resto: il modello cresce
con le lingue, ma ogni frase ne attraversa sempre la stessa quantità. Una
strada più antica è addestrare un modello per una famiglia di lingue vicine,
che ha molto meno da spartire. Il ragionamento è lo stesso che spiegava perché
l'allineamento emerge, rovesciato: condividere è economico finché la scarsità
costringe a farlo, e diventa un impaccio quando serve distinguere.

`````

`````{tab} Superiore

È la **maledizione della multilingualità** misurata da Conneau e colleghi
{cite}`conneau2020unsupervised`: a capacità fissa, la prestazione media
cross-lingua cresce con il numero di lingue fino a un massimo e poi cala, perché
lo spazio di parametri per lingua diminuisce. La curva si sposta verso l'alto e
verso destra aumentando i parametri del modello: è un vincolo di capacità, non
di metodo.

Ne discendono due tensioni concrete nel disegno di un modello multilingue. La
prima riguarda il vocabolario condiviso: allargarlo offre sottoparole migliori
alle lingue piccole ma consuma parametri nella tabella di embedding, che è già
una frazione notevole del modello. La seconda riguarda l’$\alpha$ del
ricampionamento: abbassarlo aiuta le lingue rare e sottrae dati a quelle
comuni, e non esiste un valore giusto in assoluto, solo un valore giusto
rispetto a quali lingue interessano.

Le vie d'uscita smettono di pretendere che tutti i parametri siano di tutti. La
più diretta dà a ogni lingua dei moduli propri (piccoli strati inseriti in ogni
blocco, attivi solo sulla loro lingua) e lascia condiviso il resto: la capacità
totale cresce con le lingue, mentre quella attiva per singolo token resta
costante, e aggiungere una lingua non sottrae risorse alle altre
{cite}`pfeiffer2022lifting`. È la stessa separazione fra parametri totali e
attivi che torna nei {doc}`modelli a esperti <mixture-of-experts>`. Più
tradizionali sono i modelli dedicati a famiglie linguistiche ristrette, che
hanno meno da spartire.

`````

## Che cosa vuol dire, per l'italiano

L'italiano sta in una posizione comoda ma non privilegiata: c'è in tutti i
modelli multilingui, ha abbastanza testo perché il vocabolario condiviso gli
riservi token adeguati, e ha in comune con l'inglese l'ordine delle parole
(soggetto, verbo, oggetto) e l'alfabeto, le condizioni in cui il trasferimento
zero-shot è risultato più efficace. In pratica, un classificatore rifinito su
dati inglesi funziona in italiano meglio di quanto ci si aspetterebbe, e
permette di costruire sistemi funzionanti senza disporre di dataset italiani di
dimensioni comparabili.

Questa comodità nasconde però due insidie. La prima è che *funziona meglio del
previsto* non equivale a *funziona come in inglese*. Il divario esiste e tende a
essere più grande dove conta la **morfologia**, cioè il modo in cui una parola
cambia forma. In inglese, ad esempio, il verbo «to walk» fa «walk», «walks»,
«walked», «walking»: quattro forme. In italiano «camminare» ne fa una
cinquantina, e il modello deve imparare che sono tutte lo stesso verbo, con
molti meno esempi per ciascuna. La seconda è che le lingue minoritarie non
godono di nulla di tutto ciò: il sardo, il friulano, il ladino, il napoletano
scritto non hanno abbastanza testo online per ottenere token dedicati nel
vocabolario condiviso, e la maledizione della multilingualità fa il resto. La
copertura linguistica dei modelli è, in buona misura, la mappa di quali lingue
hanno un'abbondante presenza scritta online, che non è la mappa di quali lingue
si parlano.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Un modello multilingue non ha niente di speciale nell'architettura:
  stesso esercizio a buchi, cento Wikipedia nel pentolone al posto di una, e
  una sola scatola di mattoncini (i pezzi di parola) per tutte le lingue.
  Siccome l'inglese sommergerebbe le altre, si bara sulle dosi in modo
  controllato: alle lingue piccole si danno più turni di quanti ne avrebbero, e
  le proporzioni si appiattiscono verso la parità.
- L’allineamento fra lingue emerge da solo: nessuno mostra mai al modello
  una frase accanto alla sua traduzione, eppure le due finiscono vicine.
  Perché, non si sa fino in fondo: i pezzi di parola in comune aiutano ma non
  sono indispensabili (toglierli di proposito cambia poco), contano più i piani
  della torre che i lettori in parallelo, e c'è chi sostiene che il modello
  allinei per avarizia, perché tenere cento lingue separate costerebbe più
  memoria di quanta ne abbia.
- Lo si può anche chiedere apertamente, in due modi: coprire parole tanto in
  una frase quanto nella sua traduzione, messe una dopo l'altra, così che per
  indovinarle convenga sbirciare nell'altra lingua; oppure ridurre ogni frase,
  letta senza la sua traduzione, al suo indirizzo sulla mappa, e chiedere di
  appaiare un mucchietto di frasi con il mucchietto mescolato delle loro
  traduzioni, come si farà con immagini e didascalie nel {doc}`capitolo su
  visione e linguaggio </VisioneLinguaggio/overview>`.
- Il motivo pratico di tutto questo è rifinire in inglese e usare in
  italiano, senza un solo esempio etichettato in italiano. Due avvertenze:
  la rifinitura consuma l'allineamento (i numeri interni che si spostano per
  imparare il compito sono gli stessi che tenevano vicine le lingue), e il
  salto non riesce uguale in tutte le direzioni: quel che conta è da quale
  lingua si parte, più che la somiglianza fra le due.
- La maledizione della multilingualità: lo spazio è una coperta corta.
  Oltre un certo numero di lingue, ognuna in più peggiora un po’ tutte le
  altre, e l'unico rimedio pieno è un modello più grande.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Un modello multilingue si ottiene senza cambiare nulla
  dell'architettura: stesso compito di mascheramento, testi di molte lingue e un
  vocabolario di sottotoken condiviso. Lo squilibrio fra lingue si corregge
  campionando con $q_i \propto p_i^{\alpha}$, che appiattisce le proporzioni.
- L’allineamento fra lingue emerge da solo, senza che nessun termine della
  loss lo chieda. Perché, non è del tutto chiaro: la sovrapposizione di
  vocabolario correla ma non è necessaria, la profondità conta più delle teste,
  e c'è chi sostiene che il modello allinei perché la capacità limitata lo
  costringe a condividere.
- Lo si può chiedere esplicitamente: TLM maschera coppie di frasi
  parallele; la via a doppio encoder allinea frasi intere con una loss
  contrastiva della stessa forma di quella che userà CLIP, dove al posto di
  (immagine, didascalia) c'è (frase, traduzione).
- Lo zero-shot cross-lingual transfer (rifinire in inglese, usare in
  italiano) è la ragione pratica di tutto questo. Attenzione: il fine-tuning
  erode l'allineamento, e il trasferimento non è simmetrico: conta più la
  lingua di partenza (nel campione, quelle SVO) della somiglianza fra le due.
- La maledizione della multilingualità: a capacità fissa, oltre un certo
  numero di lingue ognuna in più peggiora tutte le altre. È un vincolo di
  capacità, e l'unico rimedio pieno è un modello più grande.
```
`````
