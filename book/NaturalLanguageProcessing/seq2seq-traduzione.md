# Da frase a frase: tradurre con le reti

Prova a tradurre parola per parola: «Il gatto nero salta sul muro»
diventerebbe *«The cat black jumps on the wall»*, e un inglese storcerebbe il
naso, perché l'aggettivo va prima del nome: *«The black cat jumps on the
wall»*. Sei parole sono diventate sette, e due si sono scambiate di posto. La
traduzione prende il *senso* di una sequenza e lo riscrive in un'altra
sequenza, invece di sostituire una parola per volta: di lunghezza diversa e con un
ordine diverso.

Nella sezione precedente abbiamo costruito gli attrezzi: RNN, LSTM, GRU. In
questa li mettiamo alla prova sul compito che più di ogni altro ha spinto
avanti l'NLP: la traduzione automatica. È una storia da seguire da vicino,
perché è proprio qui, tra il 2014 e il 2017, che nasce il meccanismo di
**attenzione**: il ponte diretto verso il {doc}`capitolo sui Transformer </Transformers/overview>`.

## Scommettere sulla prossima parola

Prima di tradurre, un modello deve saper *parlare* la lingua d'arrivo. Lo
strumento è il **modello di linguaggio** che conosciamo dalla sezione sugli
*n-gram*: un sistema che, data una sequenza di parole, assegna una probabilità
alla parola successiva (la tastiera che dopo «a domani e buona» suggerisce
«serata» e quasi mai «carburatore»). La novità è *chi* fa la scommessa: non
più una tabella di conteggi, ma una rete ricorrente con la sua memoria.

`````{tab} Elementare

«Il gatto nero salta sul…»: detto a voce, la maggior parte delle
persone completa con «muro», qualcuno con «tetto» o «divano», nessuno con
«marmellata». Un modello di linguaggio fa esattamente questa scommessa, ma con
i numeri: «muro» 35%, «tetto» 25%, «divano» 10%, e giù fino a briciole di
probabilità per le parole assurde. (Le tre percentuali sono inventate qui per
far vedere l'idea: in un modello vero le calcola la rete, che si è aggiustata
i conti leggendo montagne di testo.) E per giudicare una frase intera si fa lo
stesso gioco parola per parola: si scommette sulla prima, poi sulla seconda
sapendo la prima, poi sulla terza sapendo le prime due, fino in fondo. Quanto
la frase suona giusta lo dicono tutte quelle scommesse messe insieme.

Come misurare se scommette bene? Con la
**perplessità**, che abbiamo incontrato nei richiami di matematica e già
usata come pagella per gli *n-gram*: dice *come se* il modello, a ogni
parola, tirasse un dado con un
certo numero di facce. Perplessità 20 = incerto come un dado a 20 facce;
perplessità 5 = quasi sicuro, il dado ha solo 5 facce. Più bassa, meglio è: il
modello ha capito la lingua abbastanza da restringere le alternative. Agli
estremi, chi indovina sempre ha un dado a una faccia sola e perplessità 1;
chi tira a caso fra le cinquantamila parole che conosce ha un dado da
cinquantamila facce, e perplessità cinquantamila.

`````

`````{tab} Superiore

Un modello di linguaggio stima la probabilità di un'intera sequenza
scomponendola, con la regola della catena, in predizioni della parola
successiva:

$$
P(w_1, \dots, w_n) = \prod_{t=1}^{n} P(w_t \mid w_1, \dots, w_{t-1}),
$$

dove $w_t$ è la parola al passo $t$. Una RNN implementa ciascun fattore in modo
naturale: lo stato nascosto $\mathbf{h}_{t-1}$ riassume il prefisso letto fin
lì, e una softmax sul vocabolario produce la distribuzione
$P(w_t \mid w_{<t}) = \mathrm{softmax}(\mathbf{W}_{hy}\,\mathbf{h}_{t-1})$.
L'addestramento è la cross-entropia sulla parola successiva, e la qualità si
misura con la **perplessità per parola**, vista nella sezione di teoria
dell'informazione e già usata per valutare i modelli *n-gram*:

$$
\mathrm{PP} = 2^{H},
\qquad
H = -\frac{1}{n} \sum_{t=1}^{n} \log_2 P(w_t \mid w_{<t}),
$$

dove $H$ è la cross-entropia media sul testo di test. La perplessità è il
numero di alternative equiprobabili tra cui il modello «esita» a ogni passo:
un modello perfetto avrebbe $\mathrm{PP}=1$, uno che tira a caso su un
vocabolario di $50\,000$ parole avrebbe $\mathrm{PP}=50\,000$.

`````

## Leggere in due direzioni

Le RNN della sezione precedente leggono da sinistra a destra. Ma il senso di
una parola dipende anche da ciò che viene *dopo*: in «La pesca era la sua
passione» e «La pesca era matura», al momento in cui leggi «pesca» non puoi
ancora sapere se si parla di ami o di frutta. Lo scopri solo alla fine. Da qui
un'idea degli anni Novanta {cite}`schuster1997bidirectional`: le **RNN
bidirezionali**.

`````{tab} Elementare

È come rileggere un giallo conoscendo il colpevole: alla seconda lettura ogni
indizio va al suo posto, perché sai già come va a finire. Una rete
bidirezionale fa le due letture insieme: una cella percorre la frase da
sinistra a destra, un'altra da destra a sinistra, e per ogni parola si
incollano i due riassunti (quello di ciò che precede e quello di ciò che
segue). Attenzione però: questo trucco vale solo per **capire** un testo che
esiste già tutto intero. Per **generare** una frase non funziona: mentre
scrivi, le parole future non esistono ancora; nessun giallista può rileggere
il capitolo che deve ancora scrivere. Per questo, come vedremo, chi *legge* la
frase può essere bidirezionale, ma chi la *scrive* procede sempre in avanti.

`````

`````{tab} Superiore

Una RNN bidirezionale mantiene due catene di stati indipendenti: una in avanti,
$\overrightarrow{\mathbf{h}}_t = f(\overrightarrow{\mathbf{h}}_{t-1}, x_t)$, e
una all'indietro,
$\overleftarrow{\mathbf{h}}_t = f'(\overleftarrow{\mathbf{h}}_{t+1}, x_t)$. La
rappresentazione della posizione $t$ è la concatenazione

$$
\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t ; \overleftarrow{\mathbf{h}}_t],
$$

che condensa l'intera frase *vista da quella posizione*: prefisso e suffisso.
È lo standard per i compiti di comprensione (classificazione, NER, encoding),
ma è inapplicabile alla generazione autoregressiva: al passo $t$ la catena
all'indietro richiederebbe $x_{t+1}, \dots, x_n$, che non sono ancora stati
generati. Il decoder resta quindi unidirezionale per costruzione: un vincolo
di causalità che ritroveremo, sotto forma di *maschera*, anche nei
Transformer. Ortogonale a questo è l’**impilamento** (*stacked RNN*): più
strati ricorrenti sovrapposti, dove la sequenza di stati dello strato $\ell-1$
fa da input allo strato $\ell$, per rappresentazioni via via più astratte.

`````

In PyTorch la lettura nei due sensi si chiede scrivendo `bidirectional=True`
quando si costruisce la rete, e non c'è altro da fare. Nella stessa riga
compare anche `num_layers=2`, che chiede due celle impilate una sopra l'altra:
la seconda non legge le parole, legge quello che ha capito la prima. Sono i due
modi in cui una rete ricorrente si può far crescere, in larghezza (le due
direzioni) e in altezza (gli strati):

```python
import torch
from torch import nn

lstm = nn.LSTM(
    input_size=64, hidden_size=128,
    num_layers=2,        # due strati impilati
    bidirectional=True,  # lettura in entrambe le direzioni
    batch_first=True,
)

x = torch.randn(1, 6, 64)  # 1 frase, 6 parole ("Il gatto nero salta sul muro")
out, (h, c) = lstm(x)
print(out.shape)  # torch.Size([1, 6, 256]): 2 direzioni x 128 per ogni parola
print(h.shape)    # torch.Size([4, 1, 128]): 2 strati x 2 direzioni
```

## Comprimere una frase in un vettore

Torniamo alla traduzione. Nel 2014 due gruppi di ricerca arrivano, per strade
loro, alla stessa idea, e l'idea è di una semplicità disarmante: **due reti
ricorrenti, una di fronte all'altra**. La prima legge la frase da tradurre e
non scrive niente; la seconda scrive la traduzione e non legge l'originale. Fra
le due passa una fila di numeri, e basta.

La prima si chiama *encoder*, «chi codifica», e il suo mestiere è ridurre la
frase di partenza a quel pacchetto di numeri; la seconda si chiama *decoder*,
«chi decodifica», e il suo mestiere è srotolare il pacchetto in una frase
dell'altra lingua, una parola per volta. Il pacchetto ha un nome, **vettore di
contesto**, e non è altro che l'ultimo riassunto scritto dall'encoder: quello
che gli resta in mano dopo aver letto l'ultima parola. L'architettura si chiama
**encoder–decoder**, o **seq2seq**, e i due lavori che la propongono nello
stesso anno sono di Kyunghyun Cho e colleghi a Montréal
{cite}`cho2014learning`, che è lo stesso articolo in cui nasce la GRU, e di
Ilya Sutskever, Oriol Vinyals e Quoc Le a Google
{cite}`sutskever2014sequence`.

Il decoder, mentre scrive, fa esattamente quello che fa un modello di
linguaggio: scommette sulla parola successiva. Con una differenza: la sua
scommessa non parte dal nulla, parte dal pacchetto ricevuto. Si dice allora che
è **condizionato** dalla frase di partenza, che è il modo tecnico di dire «gli
è stato detto di che cosa deve parlare».

```{figure} ../figures/seq2seq-2014.svg
:name: fig-encoder-decoder
:alt: "Schema encoder-decoder: l'encoder legge una a una le tre parole della frase inglese «the cat sleeps» e le comprime in un unico vettore di contesto; da quel vettore il decoder genera la traduzione francese «le chat dort», una parola dopo l'altra."
:width: 100%

Due reti e un vettore in mezzo. L'encoder finisce di leggere prima che il
decoder cominci a scrivere: fra i due passa solo quel vettore, e nient'altro.
```

Il «nient'altro» di {numref}`fig-encoder-decoder` è il fatto da cui discende
tutto il resto della sezione, e conviene fissarlo bene. Quel vettore di
contesto è una fila di numeri **di lunghezza decisa in anticipo**, mille per
esempio, e resta di mille numeri sia che la frase da tradurre abbia cinque
parole sia che ne abbia cinquanta: nessuno spazio in più per le frasi lunghe,
per quanto ce ne sarebbe bisogno.

E c'è un dettaglio che tornerà utile fra due pagine, quindi mettiamolo a fuoco
adesso. L'encoder, mentre legge, produce **un riassunto dopo ogni parola**: uno
dopo «il», uno dopo «il gatto», uno dopo «il gatto nero», e così via fino in
fondo. Sono tanti riassunti quante sono le parole, ciascuno con la sua fila di
numeri. Di tutti questi, però, ne viene passato al decoder uno solo, l'ultimo.
Gli altri esistono, sono già stati calcolati, e vengono buttati via.

`````{tab} Elementare

All'interprete del convegno togliamo il blocco per gli appunti: ascolta
l'intervento intero, lo tiene tutto a memoria e solo alla fine lo ripete in
italiano. L'encoder è l'ascolto, il vettore di contesto è ciò che gli resta in
testa, il decoder è la resa in italiano. Nell'articolo di Google c'è un
dettaglio curioso: dare all'encoder la frase di partenza **al contrario**
(«muro sul salta nero gatto Il») migliorava nettamente le traduzioni. Perché?
Così l’*inizio* della frase (la prima cosa che il decoder deve tradurre) viene
letto per *ultimo*, ed è il ricordo più fresco. Un trucco che rivela il
difetto di fondo: se la qualità dipende da quale parola è stata ascoltata più
di recente, la memoria unica è troppo stretta. Sulle frasi brevi regge; su un
discorso lungo l'interprete arranca, perché tutto non entra in un solo
ricordo.

«Migliorava nettamente» qualcuno l'ha dovuto misurare, e il modo assomiglia
alla correzione di un compito di traduzione. Accanto alla versione della
macchina si mette quella di un traduttore in carne e ossa, e si guarda quanto
si somigliano. Non contano solo le parole singole che coincidono, ma anche le
coppie, le terne e le quaterne di parole consecutive, perché è lì che si vede
se anche l'ordine è giusto. Un conteggio così si lascerebbe imbrogliare in due
modi, e il metro si difende da tutti e due. Chi scrivesse «il il il il»
avrebbe quattro parole su quattro presenti nella versione umana, e allora ogni
parola vale al massimo il numero di volte che compare davvero là dentro, e le
ripetizioni in più non contano niente. Chi consegnasse due parole sole, scelte
bene, avrebbe tutto giusto senza aver tradotto, e allora una traduzione più
corta di quella di riferimento paga una penalità, tanto più pesante quanto più
è corta.
Questo metro si chiama **BLEU**, ed è quello con cui la traduzione automatica
ha fatto i conti per vent'anni.

Resta un metro grezzo, e ha due punti ciechi. Va usato su un pacco intero di
frasi e non su una sola, perché su una frase corta basta una quaterna che
manca per mandare il voto a zero. E non riconosce le parafrasi, quindi una
traduzione giusta che sceglie sinonimi diversi da quelli del traduttore umano
prende comunque un voto basso.
E c'è un terzo punto cieco, che si vede
tornando all'interprete. Questo metro guarda le parole che l'interprete ha
detto e va a cercarle in quelle del traduttore umano: castiga chi **si
inventa** un pezzo. L'interprete che invece **salta** il capoverso che contava
lo castiga poco, e solo di rimbalzo, perché una versione più corta paga la
penalità sulla lunghezza. Su una traduzione la scelta ha le sue ragioni, e
qualche riga più sotto si vede quali. Ma il giorno in cui alla macchina si
chiede di **riassumere** invece che di tradurre, saltare diventa il peccato
principale, e il metro si gira: si prendono le parole del riassunto umano e si
va a vedere quante sono finite in quello della macchina. Il voto che ne esce si
chiama **ROUGE** {cite}`lin2004rouge`, e nella pratica di oggi i due versi di
uno stesso conteggio, quello che castiga l'inventare e quello che castiga il
saltare, si mettono insieme con la media prudente dell’$F_1$ della
{doc}`sezione sulle metriche </MachineLearning/metriche>`, quella in cui un
voto basso non si può nascondere dietro un voto alto.
Con questo metro, nel 2014, cinque di queste
reti messe insieme superano di poco il sistema statistico preso come termine
di paragone, e restano sotto ai sistemi migliori dell'anno. La traduzione
neurale non ha ancora vinto; ha fatto vedere che può.

`````

`````{tab} Superiore

Il modello fattorizza la probabilità della frase di arrivo
$y = (y_1, \dots, y_m)$ data quella di partenza $x = (x_1, \dots, x_n)$ come

$$
P(y \mid x) = \prod_{t=1}^{m} P(y_t \mid y_1, \dots, y_{t-1}, \mathbf{c}),
\qquad
\mathbf{c} = \mathbf{h}_n,
$$

dove $\mathbf{c}$ è il vettore di contesto (lo stato finale dell'encoder) e
ogni fattore è calcolato dal decoder, una RNN inizializzata da $\mathbf{c}$ con
softmax sul vocabolario di arrivo.

I risultati che seguono si misurano in **BLEU** {cite}`papineni2002bleu`, il
metro con cui la traduzione automatica si è confrontata per vent'anni, e vale
la pena dire com'è fatto, perché tornerà anche nella sezione sul dialogo. BLEU
confronta la traduzione candidata con uno o più riferimenti umani contando
quanti $n$-grammi hanno in comune, per $n$ da 1 a 4. Due accorgimenti fanno
tutto il lavoro. Il primo è il **clipping**: un $n$-gramma del candidato conta
al massimo il numero di volte che compare nel riferimento, altrimenti «il il il
il» otterrebbe precisione $1$. Il secondo è la **brevity penalty**,
$\mathrm{BP} = \min\!\left(1,\, e^{1 - r/c}\right)$ con $c$ e $r$ le lunghezze
in token del candidato e del riferimento, e serve perché BLEU è fatto di sole
precisioni: un termine di *recall* non c'è (non esiste un modo ovvio di
calcolarlo su più riferimenti insieme) e senza freno la traduzione più corta
sarebbe sempre la migliore. Il punteggio è

$$
\mathrm{BLEU} = \mathrm{BP} \cdot
\exp\!\left(\sum_{n=1}^{4} w_n \log p_n\right),
$$

dove $p_n$ è la precisione clippata degli $n$-grammi e $w_n = 1/4$ il peso
uniforme dei quattro ordini. I limiti vanno detti subito, perché servono a
leggere i punteggi di Sutskever: BLEU è definito **sul corpus** e non sulla singola
frase (le $p_n$ si accumulano su tutto il test set, e su una frase sola un
4-gramma mancante manda il punteggio a zero); dipende dalla tokenizzazione e
dal numero di riferimenti, tanto che due punteggi si confrontano solo a
protocollo identico, ed è la ragione per cui esiste `sacrebleu`; ed è cieco
alla parafrasi corretta. Un punto di differenza è un segnale, non una
sentenza.

Quel «un termine di *recall* non c'è» è la porta da cui entra il metro gemello.
**ROUGE** {cite}`lin2004rouge` (l'acronimo, coniato da Chin-Yew Lin, sta per
*Recall-Oriented Understudy for Gisting Evaluation*) nasce per i riassunti,
dove l'errore che conta è l'omissione. Con **un** riferimento la sua ROUGE-N è
la stessa frazione di BLEU con il denominatore sull'altro lato:

$$
\text{ROUGE-N} = \frac{\sum_{g_n \in R}
\mathrm{Count}_{\text{match}}(g_n)}
{\sum_{g_n \in R} \mathrm{Count}(g_n)} ,
$$

dove $R$ è il riassunto di riferimento, $g_n$ i suoi $n$-grammi,
$\mathrm{Count}(g_n)$ quante volte $g_n$ compare in $R$ e
$\mathrm{Count}_{\text{match}}(g_n)$ quante di quelle occorrenze si ritrovano
nel candidato. Detto a parole: BLEU conta quanti $n$-grammi del candidato
stanno nel riferimento, ROUGE quanti $n$-grammi del riferimento stanno nel
candidato.

La simmetria si rompe appena i riferimenti sono più d'uno, e vale la pena
saperlo perché è il caso normale. BLEU taglia il conteggio del candidato sul
**massimo** fra i riferimenti; la ROUGE-N originale somma invece numeratore e
denominatore **su tutti** i riferimenti, il che dà più peso agli $n$-grammi che
compaiono in parecchi di loro, e il pacchetto dell'autore usa poi una terza
ricetta ancora (il massimo delle ROUGE calcolate a coppie). Su un esempio di
tre parole i numeratori diventano $3$ e $4$: nessuna delle due formule si
ottiene dall'altra scambiando un denominatore.

Accanto alla ROUGE-N si riporta quasi sempre la **ROUGE-L**, che al posto degli
$n$-grammi conta la sottosequenza comune più lunga fra riferimento e candidato.
Le due proprietà per cui esiste sono precise: non chiede che le parole in comune
siano **consecutive**, quindi vede l'ordine senza pretendere la contiguità; e
non chiede di **fissare $n$** in anticipo. Il prezzo è che una sottosequenza
lunga si ottiene anche allungando il candidato, e per questo la si normalizza
sulle lunghezze delle due sequenze prima di confrontare riassunti di taglia
diversa (nel lavoro originale, dove i riassunti erano tagliati a una lunghezza
fissa, quella normalizzazione era regolata per contare il solo richiamo).

E questa è la ragione per cui oggi si riportano quasi sempre le $F_1$, cioè
precisione e richiamo insieme: un richiamo puro non cala mai allungando il
candidato, quindi da solo premia chi ricopia mezzo articolo.

Sutskever et al. usano LSTM a 4 strati con stati da
1000 dimensioni e riportano, sul benchmark WMT'14 inglese→francese, un BLEU di
$34{,}8$ (con un ensemble di cinque modelli) contro il $33{,}3$ del sistema
statistico a frasi di riferimento. Il confronto va letto per quello che è: il
$33{,}3$ è il sistema di *riferimento*, non lo stato dell'arte, che su quel
compito stava a $37{,}0$; la rete pura non lo raggiunge, e ci si avvicina
($36{,}5$) solo quando la si usa per riordinare le mille ipotesi prodotte dal
sistema statistico. Nel 2014 il neurale non ha ancora vinto: la data del
sorpasso è il 2016. L'aneddoto
dell'inversione è invece documentato
nei numeri: invertire l'ordine delle parole sorgente fa scendere la
perplessità di test da $5{,}8$ a $4{,}7$ e salire il BLEU da $25{,}9$ a
$30{,}6$, perché accorcia le dipendenze tra le prime parole di $x$ e le prime
di $y$, semplificando l'ottimizzazione. Ma il limite strutturale resta:
qualunque sia $n$, tutta l'informazione su $x$ deve passare per un vettore di
dimensione fissa. È un **collo di bottiglia**, e le prestazioni degradano
visibilmente al crescere della lunghezza della frase.

`````

## Tornare a guardare: la nascita dell'attenzione

La soluzione arriva nel giro di pochi mesi, da Dzmitry Bahdanau, Kyunghyun Cho
e Yoshua Bengio {cite}`bahdanau2015neural`, e nasce da una domanda tanto ovvia
quanto ben posta: perché costringere il decoder a lavorare a memoria, se i
riassunti intermedi ci sono già?

Ricordate: l'encoder ne aveva prodotto uno per parola, e li avevamo buttati via
tutti tranne l'ultimo. Smettiamo di buttarli. Teniamoli lì tutti, in fila, e
lasciamo che il decoder, ogni volta che deve scrivere una parola, **li guardi
tutti quanti** e decida da sé a quali dare retta adesso. Non «guarda solo il
quinto»: dà a ciascuno un voto, alto per quelli che gli servono, basso per gli
altri, e poi li mescola in proporzione ai voti. Quei voti sono l’**attenzione**:
un numero per ogni parola della frase di partenza, ricalcolato da capo a ogni
parola prodotta.

```{figure} ../figures/attention-prima-dei-transformer.svg
:name: fig-allineamento-traduzione
:alt: "Due file di parole, una sopra l'altra: in alto la frase inglese «the black cat sleeps», in basso la traduzione italiana «il gatto nero dorme». Delle linee collegano le parole delle due file, e il loro spessore è il peso dell'attenzione. Le linee spesse legano «the» a «il» e «sleeps» a «dorme» senza incrociarsi, mentre al centro si incrociano: «black» va a «nero» e «cat» a «gatto». Due linee sottili, fra le stesse parole del centro, sono i pesi piccoli rimasti."
:width: 88%

L'allineamento che nessuno ha annotato. Le linee dicono, per ogni parola
prodotta, dove il modello ha guardato: nessuno gliel'ha insegnato, si leggono
a posteriori dai pesi che si è dato da solo.
```

Il fatto che le due linee centrali di {numref}`fig-allineamento-traduzione` si
incrocino è la notizia, non un difetto: in inglese l'aggettivo precede il nome,
in italiano lo segue, e il modello va a prendersi le parole fuori ordine, terza
prima e seconda dopo. È la cosa che con il vettore unico non si poteva fare, e
non perché fosse difficile: perché nel vettore unico l'informazione su dove
stava ciascuna parola era già stata schiacciata via.

```{figure} ../figures/seq2seq-attenzione.svg
:name: fig-seq2seq-attenzione
:alt: "Encoder-decoder con attenzione: in basso sei stati dell'encoder bidirezionale per «Il gatto nero salta sul muro», in alto il decoder che genera «The black cat»; mentre produce «cat», frecce di spessore diverso collegano ogni parola sorgente al decoder, e la più spessa parte da «gatto»."
:width: 100%

Mentre genera «cat», il decoder consulta *tutti* gli stati dell'encoder: lo
spessore di ogni freccia è il peso di attenzione, massimo su «gatto».
```

Come mostra {numref}`fig-seq2seq-attenzione`, mentre produce *«cat»* il
decoder dà il voto più alto a «gatto» (0,62), tiene d'occhio «nero» (0,20) e
quasi ignora il resto. E a ogni passo la mappa cambia: per *«wall»* il voto
grosso si sposterà su «muro».

(Nella figura quei voti sono chiamati **pesi**, ed è il termine che si usa
ovunque, ma attenzione a non confonderli con i pesi della sezione precedente,
quelli che una rete impara e si tiene: questi cambiano a ogni parola prodotta e
sono roba che il modello *decide sul momento*, non roba che possiede.)

Con questo, il collo di bottiglia del vettore unico sparisce: nessuna fila di
numeri di lunghezza fissa deve più contenere l'intera frase. Resta invece
intatto l'altro collo di bottiglia, quello della sezione precedente, cioè il
fatto che tutto questo si legge e si scrive in fila, un passo dopo l'altro. Due
strozzature diverse: l'attenzione ne toglie una, e sarà il capitolo dopo a
togliere l'altra.

`````{tab} Elementare

Il nostro interprete adesso ha il testo dell'intervento sul tavolo. Mentre
traduce non recita più a memoria: per ogni parola che pronuncia dà un'occhiata
al foglio, e l'occhio cade sul punto che serve in quel momento, come un
evidenziatore che si sposta man mano che la traduzione avanza. Non c'è nessuna
regola scritta a mano che dica dove guardare: la rete impara da sola, durante
l'addestramento, che quando sta per dire *cat* conviene pesare molto «gatto».
Sorpresa in regalo: disegnando dove cade l'evidenziatore si ottiene, gratis,
l'allineamento tra le parole delle due lingue («cat» ↔ «gatto», «wall» ↔
«muro») che nessuno aveva chiesto al modello di imparare.

`````

`````{tab} Superiore

L'encoder (bidirezionale, così che ogni $\mathbf{h}_j$ rappresenti la parola
$j$ con tutto il suo contesto) produce gli stati
$\mathbf{h}_1, \dots, \mathbf{h}_n$. Al passo $i$ il decoder, con stato
$\mathbf{s}_{i-1}$, calcola un punteggio di allineamento verso ogni posizione
sorgente con una piccola rete a un solo strato nascosto:

$$
e_{ij} = \mathbf{v}_a^{\top} \tanh\!\left(\mathbf{W}_a\, \mathbf{s}_{i-1} + \mathbf{U}_a\, \mathbf{h}_j\right),
$$

dove $\mathbf{W}_a$, $\mathbf{U}_a$ e $\mathbf{v}_a$ sono parametri appresi (è
la cosiddetta attenzione **additiva**). I punteggi diventano pesi con una
softmax, e i pesi definiscono un vettore di contesto *diverso a ogni passo*:

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{n} \exp(e_{ik})},
\qquad
\mathbf{c}_i = \sum_{j=1}^{n} \alpha_{ij}\, \mathbf{h}_j,
$$

dove $\alpha_{ij}$ è quanto il passo di decodifica $i$ «guarda» la parola
sorgente $j$ (i pesi sommano a 1) e $\mathbf{c}_i$ è la media pesata degli
stati dell'encoder, che entra nel calcolo di $\mathbf{s}_i$ e della parola
successiva. La matrice dei pesi $\alpha_{ij}$, visualizzata, è una mappa di
allineamento tra le due frasi: appresa senza alcuna supervisione esplicita.

`````

Vale la pena dirlo in modo esplicito, perché è il ponte verso il capitolo
successivo: questa è **la stessa attenzione** dei Transformer. E siccome è
l'idea che di là diventa tutto, conviene guardarla una volta con i numeri sotto
gli occhi.

Immaginiamo che i riassunti siano corti, tre numeri l'uno, e che ce ne siano
tre soltanto:

| riassunto | numeri | voto |
|---|---|---|
| dopo «il» | `2, 0, 1` | 0,10 |
| dopo «il gatto» | `0, 4, 2` | 0,70 |
| dopo «il gatto nero» | `1, 1, 0` | 0,20 |

I voti li ha decisi il decoder, guardando che cosa gli serve *adesso*, e sono
tre numeri positivi che sommano a uno: si spartiscono un totale fisso, proprio
come si spartisce l'attenzione di una persona. Se ne do di più a uno, ne resta
di meno per gli altri, esattamente come quando in classe ascolto il professore
e allora non sento chi mi parla da dietro.

Adesso si mescola. Per la prima casella: $0{,}10 \times 2 + 0{,}70 \times 0 +
0{,}20 \times 1 = 0{,}4$. Per la seconda: $0{,}10 \times 0 + 0{,}70 \times 4 +
0{,}20 \times 1 = 3{,}0$. Per la terza: $0{,}10 \times 1 + 0{,}70 \times 2 +
0{,}20 \times 0 = 1{,}5$. Il risultato è una fila di numeri nuova, `0,4 · 3,0 ·
1,5`, e si vede a occhio che somiglia molto al secondo riassunto e poco agli
altri: è il riassunto che il decoder aveva votato di più. Questa operazione,
mescolare più file di numeri dando a ciascuna un peso, si chiama **media
pesata**, ed è tutta l'attenzione. La fila che ne esce va dritta al decoder, che
la usa per scrivere la parola successiva: al posto del solito pacchetto sempre
uguale, adesso ne riceve uno confezionato apposta per il passo che sta facendo.

Cambieranno due cose, nel capitolo dopo. La prima è **come si decidono i
voti**. Qui li calcola un pezzo di rete apposito, addestrato insieme al resto:
gli si danno il riassunto e lo stato del decoder, e sputa fuori un numero. Là
il conto sarà diretto e molto più economico, fatto sulle due file di numeri
senza nessun pezzo in mezzo. La seconda è che cadrà tutta l'impalcatura
ricorrente attorno: resterà solo l'attenzione, che qui nasce come rattoppo per
la traduzione e là diventa l'architettura intera.

## Con la soluzione accanto: il teacher forcing

Il decoder scrive una parola alla volta, e ogni parola la decide guardando
quella che l'ha preceduta. Quale, però? Mentre si addestra ce ne sono due
disponibili, la parola che il decoder ha appena prodotto e quella che la
traduzione di riferimento ha in quel punto, e non sono la stessa parola.
Prendere la seconda si chiama *teacher forcing*, alla lettera «imporre quella
del maestro».

`````{tab} Elementare

Per insegnarti a tradurre, il professore ha un metodo. Ti dà la frase inglese,
tu scrivi la prima parola italiana, lui te la segna se è sbagliata (è da quel
segno che imparerai) e poi, prima di chiederti la seconda, cancella la tua e ci
mette **la parola giusta**. Poi ti chiede la seconda, e rifà lo stesso. Ogni
parola la scrivi partendo da un inizio corretto, anche quando la tua era
sbagliata. Il foglio delle soluzioni è la traduzione umana che sta nei dati, e
il professore è il conto che confronta la tua parola con quella.

Il metodo ha un vantaggio che non è la gentilezza. Il professore conosce tutte
le venti domande **prima** che tu cominci, perché l'inizio di ognuna sta già
scritto sulla soluzione; se invece dovesse partire dalle tue parole, la settima
domanda non saprebbe formularla finché non hai finito la sesta. Avere le domande
tutte pronte in anticipo è un guadagno enorme per una macchina, che così
organizza il lavoro in blocco invece di fermarsi venti volte, ed è la ragione
per cui a questo metodo nessuno rinuncia. Quanto grande sia il guadagno dipende
da com'è fatta la rete: qui, dove ogni parola aspetta comunque il riassunto di
quella prima, resta un risparmio; nel capitolo che segue, dove quell'attesa
sparisce, diventa un unico passaggio per la frase intera.

Poi arriva il compito in classe, e lì nessuno cancella niente. Scrivi «The», e
la seconda parola la scrivi partendo dal tuo «The». Se in quel punto il
professore avrebbe messo «A», stai continuando da un inizio su cui non ti sei
mai esercitato.

Da lì in poi peggiora, e questa è la parte che sorprende. L'errore non resta
soltanto scritto: cambia anche tutte le domande che vengono dopo, perché ogni
parola successiva la scegli guardando un foglio che nell'esercizio non è mai
comparso. Il modo in cui si continua una frase già sgangherata non l'hai mai
provato, perché nei compiti di esercizio la frase era sempre a posto. Questo
scarto fra come si impara e come si lavora ha un nome, **exposure bias**: alla
lettera «distorsione da esposizione», perché durante l'esercizio si è stati
esposti solo ai testi giusti.

Il rimedio che viene in mente per primo è ammorbidire il metodo: ogni tanto il
professore lasci stare la tua parola invece di correggerla, di rado all'inizio e
sempre più spesso man mano che migliori. Così ogni tanto ti eserciti anche a
continuare da un inizio tuo.

Il secondo rimedio smette di correggere parola per parola: consegni la
traduzione intera e il professore le dà un voto, con uno dei metri automatici
già visti, BLEU o ROUGE, con tutti i limiti che quei metri hanno. Il voto arriva
alla fine e non dice quale parola fosse sbagliata, che è la situazione di chi
impara per tentativi da una ricompensa, cioè il
{doc}`reinforcement learning </ReinforcementLearning/overview>` a cui il libro
dedica una parte intera. Anche questo rimedio, però, parte dal primo metodo e lo
tiene: si comincia correggendo parola per parola, e solo dopo si passa al voto
sul risultato, prima sulla coda della frase e via via su tutta.

Il primo rimedio ha però un buco, e si vede guardando che cosa quell'esercizio
premia davvero. La parola che il professore si aspetta è sempre quella che
segue sulla soluzione, qualunque cosa tu abbia scritto un attimo prima. Ma
allora, per prendere il massimo dei voti, la tua frase non serve nemmeno
guardarla: basta tenere il conto di quante parole sono passate e ricopiare la
soluzione da lì in poi. Chi vince quell'esercizio può farlo ignorando quello che
sta scrivendo, ed è esattamente quello che un traduttore non può fare.

`````

`````{tab} Superiore

La verosimiglianza di una coppia sorgente-traduzione si fattorizza come

$$
\log P(y \mid x) = \sum_{i=1}^{m+1} \log P(y_i \mid y_{<i},\, x),
\qquad y_{m+1} = \texttt{</s>},
$$

dove $y_{<i}$ sono i token della traduzione **di riferimento** che precedono la
posizione $i$, e l'ultimo fattore è quello del token di fine frase, senza il
quale la somma sulle sequenze di ogni lunghezza non farebbe uno. Massimizzarla
prescrive di condizionare sul prefisso vero, e non su quello che il modello
produrrebbe: il *teacher forcing* è la forma esatta della massima
verosimiglianza su queste coppie, e non un'approssimazione adottata per
comodità. Il nome è quello che gli danno Williams e Zipser
{cite}`williams1989learning` in un contesto diverso, le reti ricorrenti
addestrate in continuo, dove la pratica era già in uso.

Ne discende il guadagno computazionale. Con il prefisso vero disponibile in
anticipo, tutti gli **ingressi** del decoder sono noti prima di cominciare,
quindi cade la dipendenza dal campionamento: nessun passo deve attendere che il
precedente estragga un token. Su una RNN resta la dipendenza dallo **stato**,
cioè $m$ passi in sequenza, e il guadagno è che le proiezioni ingresso-stato si
fanno tutte insieme in un prodotto di matrici solo. Nelle architetture del
capitolo seguente, dove la causalità è imposta da una maschera e non da una
ricorrenza, cade anche la dipendenza dallo stato: un solo passaggio in avanti
per l'intera frase.

A generazione, però, $y_{<i}$ non esiste: al suo posto c'è $\hat{y}_{<i}$,
prodotto dal modello stesso. Il modello viene quindi interrogato su una
distribuzione di prefissi che in addestramento non ha mai visto. Ranzato e
colleghi {cite}`ranzato2016sequence` battezzano lo scarto: «ci riferiamo a
questa discrepanza come *exposure bias*, che si verifica quando un modello è
esposto soltanto alla distribuzione dei dati di addestramento invece che alle
proprie predizioni».

Il danno non è la composizione di un tasso d'errore costante. La conditional
$P_\theta(y_i \mid y_{<i}, x)$ è stimata bene dove i prefissi abbondano, cioè
sul supporto dei dati; fuori di lì non c'è nessuna garanzia, perché i dati non
permettono di distinguere fra ipotesi che coincidono sul supporto e divergono
altrove. Un token deviante porta il modello in un contesto raro o inedito, dove
sbaglia di più, il che rende più probabile il token deviante successivo. La
popolazione delle sequenze si sdoppia: quelle ancora sul supporto sbagliano al
tasso misurato in addestramento, quelle uscite sbagliano molto di più, e la
media fra le due peggiora finché le proporzioni si assestano.

I rimedi seguono due strade, e nessuna delle due abbandona il teacher forcing.
Lo *scheduled sampling* di Bengio e colleghi {cite}`bengio2015scheduled`
interpola: a ogni token si tira una moneta e con probabilità $1-\epsilon$ si usa
$\hat{y}_{i-1}$ al posto di $y_{i-1}$, con $\epsilon$ portato da 1 verso 0 lungo
l’**addestramento** (non lungo la frase), cioè un curriculum. MIXER, di Ranzato
e colleghi, ottimizza la metrica di valutazione con il gradiente di policy di
{doc}`REINFORCE </DeepReinforcementLearning/policy-gradient>`, ma parte da un
modello già addestrato con l'entropia incrociata, tiene le due perdite mescolate
e sposta il confine fra le due un pezzo di frase alla volta: gli autori
insistono che entrambi gli ingredienti sono necessari, e chiamano curriculum
anche il proprio.

Il punto di rottura sta sul primo rimedio. Huszár {cite}`huszar2015how` mostra
che l'obiettivo dello scheduled sampling è **improprio**, e che il suo ottimo
non è la distribuzione dei dati nemmeno nel limite di dati e capacità infiniti:
il modello che lo minimizza può ignorare il contenuto del prefisso e limitarsi a
contare le posizioni. La derivazione è svolta su sequenze di lunghezza due e la
generalizzazione è dichiarata dagli autori come una congettura, ma la direzione
è chiara. Lo stesso lavoro, va detto per intero, sostiene che anche la massima
verosimiglianza sia l'obiettivo sbagliato quando lo scopo è generare testo
verosimile. La tensione corre allora fra due obiettivi di cui nessuno dei due è
quello che si vorrebbe davvero, e il rimedio guasto contro il metodo sano è una
lettura più comoda del vero.

`````

Che cosa succeda al tasso d'errore si può guardare senza addestrare niente.
Serve una lingua giocattolo di dieci parole, numerate da 0 a 9, in cui l'unica
frase legale è contare: dopo lo 0 viene 1, dopo il 7 viene 8, dopo il 9 si
ricomincia da 0. E serve un modello che, come i decoder veri, guardi **due**
parole per scegliere la terza, mentre a questa lingua ne basterebbe una: è
questa sovrabbondanza a creare i paia di parole che i dati non contengono mai.
Sui paia che i dati contengono il modello sbaglia una volta su cento; sugli
altri tira a caso fra le dieci parole, e non perché la regola là non valga (vale
identica) ma perché nei dati «la successiva della seconda» e «la seconda dopo la
prima» sono la stessa cosa, e quale delle due il modello abbia imparato si vede
soltanto fuori. La terza riga stampata è la controprova: la stessa cosa, con un
modello che quei paia li sappia continuare.

```python
import numpy as np

V, T, N = 10, 60, 20000   # 10 parole, frasi lunghe 60, 20000 frasi per volta
FUGA = 0.01               # sui paia che i dati contengono sbaglia una volta su cento

def modello(sa_continuare_fuori):
    """Le probabilità della parola dopo, dato il paio che precede."""
    M = np.full((V, V, V), 1.0 / V)      # paia mai viste: il modello tira a caso
    for a in range(V):
        paia = range(V) if sa_continuare_fuori else [(a + 1) % V]
        for b in paia:
            M[a, b] = FUGA / (V - 1)
            M[a, b, (b + 1) % V] = 1 - FUGA
    return M.cumsum(axis=2)

def scrivi(cum, da_se, seme=20260830):
    """N frasi; se `da_se` è falso, prima di ogni parola torna l'inizio giusto."""
    rng = np.random.default_rng(seme)
    seq = np.zeros((N, T), dtype=int)
    seq[:, 1] = 1
    scritte = np.zeros((N, T), dtype=int)
    for t in range(2, T):
        a, b = (seq[:, t-2], seq[:, t-1]) if da_se else ((t-2) % V, (t-1) % V)
        scelta = (cum[a, b] < rng.random(N)[:, None]).sum(axis=1).clip(0, V - 1)
        scritte[:, t] = scelta
        seq[:, t] = scelta if da_se else t % V
    return seq, scritte

reale = modello(False)
_, con_soluzione = scrivi(reale, da_se=False)
libere, _ = scrivi(reale, da_se=True)
ideali, _ = scrivi(modello(True), da_se=True)
ok_sol = con_soluzione[:, 2:] == np.arange(2, T) % V
ok_lib = libere[:, 2:] == (libere[:, 1:-1] + 1) % V
ok_ide = ideali[:, 2:] == (ideali[:, 1:-1] + 1) % V

print("parola  con la soluzione   da sé   da sé, sulle frasi ancora intatte   frasi intatte")
for t in (0, 8, 28, 57):
    intatte = ok_lib[:, :t].all(axis=1)      # nessun errore prima di questa parola
    print(f"{t+2:6}{100*ok_sol[:, t].mean():16.1f}%{100*ok_lib[:, t].mean():8.1f}%"
          f"{100*ok_lib[intatte, t].mean():36.1f}%{100*intatte.mean():16.1f}%")

print()
print("controprova, con un modello che sappia continuare anche fuori strada:")
print("  da sé      " + "  ".join(f"{100*ok_ide[:, t].mean():.1f}%" for t in (0, 8, 28, 57)))
print("  frasi intatte " + "  ".join(
    f"{100*ok_ide[:, :t].all(axis=1).mean():.1f}%" for t in (0, 8, 28, 57)))
```

```text
parola  con la soluzione   da sé   da sé, sulle frasi ancora intatte   frasi intatte
     2            98.9%    98.9%                                98.9%           100.0%
    10            99.1%    94.2%                                99.1%            92.1%
    30            99.0%    91.2%                                99.0%            75.7%
    59            99.0%    90.9%                                99.0%            56.1%

controprova, con un modello che sappia continuare anche fuori strada:
  da sé      98.9%  99.1%  99.0%  99.0%
  frasi intatte 100.0%  92.1%  75.7%  56.1%
```

La prima colonna è piatta: con la soluzione accanto il modello sbaglia una
parola su cento alla seconda posizione e una su cento alla cinquantanovesima,
perché a ogni passo riparte da un inizio corretto. La seconda scende fino a nove
errori su cento e poi si assesta lì, e la terza dice da dove viene quel calo:
fra le frasi che non hanno ancora sbagliato niente, il modello continua a
sbagliare una volta su cento, sempre, esattamente come in addestramento. Le
frasi buone non peggiorano, diventano sempre meno, e la quarta colonna le conta:
da cento su cento scendono a cinquantasei.

La controprova dice quale delle quattro colonne misuri davvero l'exposure bias,
e la risposta è una sola. Con un modello che sappia continuare anche fuori
strada, la seconda colonna resta al novantanove per cento a ogni posizione,
mentre la quarta scende esattamente come prima, fino a quel cinquantasei: le
frasi si rovinano lo stesso, perché anche un modello perfetto fuori strada
sbaglia una parola su cento e in sessanta parole l'errore capita. **Lo scarto
fra la prima colonna e la seconda è tutto l'exposure bias, e le altre due non ne
contengono niente.** Il numero che l'addestramento misura è il primo; quello che
descrive la macchina al lavoro è il secondo; e la distanza fra i due la fanno,
per intero, le frasi su cui il modello non si è mai esercitato.

## Generare la frase: greedy e beam search

Resta un problema che finora abbiamo dato per scontato. Il decoder, a ogni
passo, non sceglie una parola: assegna una percentuale a **tutte** le parole
che conosce, decine di migliaia di numeri che sommano a uno. Come si passa da
quell'elenco a una parola sola? L'istinto dice: prendi la più alta e vai
avanti. Si chiama strategia **greedy**, «ingorda», ed è quello che fa chiunque
abbia fretta. Ma la parola migliore *adesso* non porta sempre alla frase
migliore *alla fine*.

(E chi gli dice di smettere? Fra le voci del suo elenco ce n'è una che non è
una parola, ma il segnale di fine frase, lo stesso `</s>` incontrato con gli
n-gram. Quando il decoder scommette su quello, ha deciso che la traduzione è
finita, e ci si ferma. È una scommessa come le altre, e come le altre può
sbagliare: un modello che lo tira fuori troppo presto tronca la frase, uno che
non lo tira fuori mai continua a scrivere finché qualcuno non lo interrompe.)

`````{tab} Elementare

Un vocabolario da cinquantamila parole apre cinquantamila strade a ogni passo,
e dopo dieci passi le frasi che si possono comporre sono un numero di
quarantasette cifre; nessun calcolatore le percorrerà mai tutte per tenere la
migliore. La frase si costruisce un pezzo alla volta, come un cammino deciso
bivio per bivio, e la sola domanda è quanto guardare avanti prima di
impegnarsi.

Al primo bivio ci sono due cartelli, e per comodità facciamo finta che siano
gli unici. «A» promette 0,50, «The» promette 0,40. Chi ha fretta prende «A» e
non torna più indietro.

Il primo cartello però non decide da solo. Una strada vale il prodotto di tutti
i numeri incontrati lungo il cammino, perché sono cose che devono capitare
tutte insieme, come i voti che si moltiplicavano nel filtro antispam. Al bivio
dopo «A», «black» promette 0,30, e la strada «A black» vale
0,50 × 0,30 = 0,15. Dopo «The», «black» promette 0,60, e «The black» vale
0,40 × 0,60 = 0,24. La strada partita peggio è arrivata meglio.

Chi non vuole cadere nella trappola manda avanti più esploratori invece di uno,
e la mossa si chiama **beam search**, «ricerca a fascio». Quanti mandarne si
decide prima, e quel numero si chiama $k$: con $k=2$ restano in piedi sia «A»
sia «The», al bivio successivo si scopre che «The black» è in testa, e si
prosegue di lì fino a «The black cat…». Le strade lasciate cadere per via si
dicono potate, come i rami di un albero, perché dal tronco parte ogni
continuazione possibile e a ogni bivio se ne tagliano quasi tutte.

C'è un difetto, e si vede dai numeri dei cartelli. Sono tutti minori di uno,
quindi ogni moltiplicazione rimpicciolisce il punteggio. Da 0,50 si scende a
0,15 al secondo bivio e a 0,12 al terzo. Un cammino lungo scende sempre, anche quando sta
andando benissimo. Chi confronta i totali sceglie allora la strada più corta, e
la traduzione esce troncata a metà.

Si rimedia mettendo i cammini sullo stesso metro, e si guarda quanto vale in
media un singolo passo invece del totale. Un cammino di dieci parole e uno di
tre diventano così confrontabili. Questa correzione si chiama **penalità di
lunghezza**, e toglie alle frasi brevi un vantaggio che non si sono
guadagnate.

Resta da decidere quanti esploratori mandare. Con uno solo si torna alla fretta
del primo bivio. Con due, o con dieci, la strada migliore in assoluto può
restare fuori lo stesso: se parte da un cartello che sembrava mediocre, è stata
abbandonata lì, e nessuno torna indietro a riprenderla. Ogni esploratore in più
riduce il rischio e costa, perché è un cammino da seguire fino in fondo. In
traduzione ne bastano quasi sempre una manciata, e non più di una decina.

`````

`````{tab} Superiore

Formalmente cerchiamo $\hat{y} = \arg\max_y P(y \mid x)$, ma le sequenze
possibili sono $|V|^m$ (vocabolario $V$, lunghezza $m$): la ricerca esaustiva
è intrattabile e la scelta greedy è solo l'approssimazione con orizzonte 1.
La beam search è una via di mezzo: a ogni passo estende le $k$ ipotesi
correnti con tutte le parole del vocabolario, ordina le sequenze per
punteggio cumulato

$$
\mathrm{score}(y_{1:t}) = \sum_{i=1}^{t} \log P(y_i \mid y_{<i}, x)
$$

e trattiene le migliori $k$ (con $k=1$ si torna alla greedy). Non è una
ricerca esatta (l'ottimo globale può comunque sfuggire al fascio) ma in
traduzione valori di $k$ tra 4 e 10 bastano quasi sempre. Un dettaglio
pratico: essendo una somma di logaritmi negativi, il punteggio penalizza le
frasi lunghe, e il decoder tenderebbe a traduzioni troppo corte. Si corregge
con una **length penalty**, per esempio dividendo il punteggio per
$|y|^{\alpha}$ con $\alpha \approx 0{,}6$–$0{,}7$. Il sistema di traduzione di
Google {cite}`wu2016google` parte proprio da questa euristica e la sostituisce
poi con una variante appena più elaborata,
$lp(y) = \frac{(5+|y|)^{\alpha}}{6^{\alpha}}$, dove l'esponente agisce su
$(5+|y|)$ e non su $|y|$: per questo il valore che gli autori usano,
$\alpha = 0{,}2$, non è confrontabile con lo $0{,}65$ dell'euristica di
partenza, ed è un dettaglio che vale la pena tenere a mente prima di citare
«l’$\alpha$ di GNMT».

`````

```{figure} ../figures/beam-search.svg
:name: fig-beam-search
:alt: "Albero di beam search con larghezza due su tre passi: al primo passo restano nel fascio «A» (0,50) e «The» (0,40); al secondo «The black» (0,24) supera «A black» (0,15); al terzo l'ipotesi migliore è «The black cat» (0,19). I rami scartati sono in grigio tratteggiato."
:width: 100%

Beam search con $k=2$ sull'esempio del testo. Ogni numero è il punteggio della
strada intera fino a lì, cioè il prodotto di tutte le probabilità incontrate
lungo il cammino. La greedy si sarebbe fermata su «A» al primo passo; il
fascio recupera «The black cat».
```

In {numref}`fig-beam-search` i rami in terracotta sono le due strade tenute
aperte, quelli grigi tratteggiati i rami potati: il ramo spesso è la traduzione
che la strategia ingorda non avrebbe mai trovato.

## 2016: la traduzione neurale entra in produzione

```{figure} ../figures/traduzione-automatica-da-regole-a-llm.svg
:name: fig-paradigmi-traduzione
:alt: "Linea del tempo con i quattro paradigmi della traduzione automatica: i sistemi a regole scritte da linguisti, i metodi statistici che imparano da testi già tradotti da esseri umani, la traduzione neurale con encoder-decoder e attenzione, e infine i modelli linguistici generalisti che traducono senza essere stati costruiti per farlo."
:width: 100%

Quattro modi di tradurre, in settant'anni. A ogni passaggio si sposta chi
fornisce la conoscenza della lingua: prima il linguista che scrive le regole,
poi una montagna di testi già tradotti da esseri umani (un romanzo e la sua
traduzione, gli atti di un parlamento in due lingue), poi un modello addestrato
apposta, infine un modello che non era stato pensato per questo.
```

L'ultimo passaggio di {numref}`fig-paradigmi-traduzione` è il più singolare, e
il libro lo incontrerà nel {doc}`capitolo sui Transformer </Transformers/overview>`: la traduzione ha smesso di
essere un compito con un'architettura propria ed è diventata una delle cose
che un modello generalista sa fare. Qui però siamo alla terza tappa, ed è
quella che ha portato la traduzione neurale in produzione.

Questa storia ha una data di consegna. Nel settembre 2016 Google annuncia GNMT
(*Google Neural Machine Translation*) {cite}`wu2016google`: un encoder–decoder
con l'attenzione, esattamente la ricetta di questa sezione, ma in grande: otto
**strati** di celle impilate per l'encoder e altrettanti per il decoder. L'idea
dell'impilamento è che il primo strato legge le parole, il secondo legge quello
che ha capito il primo, e così via, ogni piano un po’ più astratto del
precedente.

Questa rete prende il posto del sistema che Google usava davvero, quello dietro
al bottone «traduci» che chiunque poteva premere: un sistema statistico che
lavorava a pezzi di frase, imparando da montagne di testi già tradotti quali
gruppi di parole si scambiano con quali. Aveva retto per un decennio. Si parte
dalla coppia cinese-inglese, circa 18 milioni di traduzioni al giorno.

E il miglioramento non è misurato con un punteggio calcolato da un programma,
ma da esseri umani. Google mette delle persone bilingui davanti alla stessa
frase tradotta dal vecchio sistema e dal nuovo, senza dire quale sia quale, e
chiede di dare un voto a ciascuna; poi confronta i voti. Nel mucchio ci mette
anche una traduzione fatta da un traduttore umano, che prende il voto più alto
di tutti ed è il metro di riferimento. Il risultato: della distanza che separava il vecchio sistema dal traduttore umano, il nuovo
ne recupera **almeno il 60 per cento** su tutte e sei le coppie misurate, e in
media quasi il settanta. Per la prima volta le reti
ricorrenti che abbiamo studiato traducono, ogni giorno, per centinaia di
milioni di persone.

Questa storia ha anche un seguito che il capitolo sui Transformer riprende per
intero. Pochi mesi dopo, invece di un modello per coppia di lingue, lo stesso
gruppo ne addestra **uno solo** su tutte le coppie insieme, e scopre che
traduce anche fra due lingue che non ha mai visto appaiate
{cite}`johnson2017google`: è il primo indizio che dentro una rete addestrata su
molte lingue si formi qualcosa di simile a una lingua franca interna.

Ma il rattoppo si stava già mangiando il vestito. Se l'attenzione permette a
ogni parola generata di guardare direttamente tutte le parole sorgente, a cosa
serve ancora far scorrere uno stato passo dopo passo? Nel 2017 un gruppo di
ricercatori di Google si fa esattamente questa domanda, e il titolo della
risposta è tutto un programma: *Attention Is All You Need*
{cite}`vaswani2017attention`; l'attenzione è tutto ciò che serve. Via la
ricorrenza, resta solo il meccanismo che avete appena visto nascere, promosso
da comprimario a protagonista. Come, di preciso, è il tema del capitolo sui
**Transformer**.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Un **modello di linguaggio** scommette sulla parola successiva, e la sua
  pagella è la **perplessità**: quante facce ha il dado con cui esita a ogni
  passo. Più è bassa, più il modello ha ristretto le alternative.
- Le **RNN bidirezionali** rileggono la frase nei due sensi insieme, come un
  giallo di cui si conosce già il colpevole: preziose per *capire* un testo
  che esiste tutto intero, inutilizzabili per *scriverne* uno, perché mentre
  si scrive le parole future non ci sono ancora.
- **Seq2seq** è l'interprete senza appunti: una rete (l'encoder) ascolta
  l'intera frase di partenza e la tiene in un unico ricordo, una seconda (il
  decoder) la ridice nell'altra lingua partendo da lì. Se la frase è lunga,
  in quel ricordo non ci sta tutto: è il collo di bottiglia.
- L’**attenzione di Bahdanau** mette il testo sul tavolo dell'interprete: per
  ogni parola che pronuncia, un'occhiata al punto che serve adesso, con
  un'attenzione che si sposta a ogni passo e che nessuno gli ha insegnato dove
  posare. In regalo si ottiene l'allineamento fra le parole delle due lingue,
  ed è la stessa idea che nei Transformer diventerà protagonista.
- Mentre impara, il decoder riparte dopo ogni parola da quella **giusta** invece
  che dalla propria (*teacher forcing*), e per questo tutte le domande della
  frase sono note in anticipo, il che fa risparmiare molto lavoro. Quando poi
  lavora da sé quella correzione non c'è, e da un inizio sbagliato continua come
  non si è mai esercitato a fare: sbaglia nove parole su cento dove in
  addestramento ne sbagliava una, ed è lì, in quello scarto, che sta tutto il
  danno. Le frasi ancora senza errori continuano invece a sbagliarne una su
  cento, e quel nove è la media fra loro e quelle già uscite di strada. Si
  rimedia lasciandogli ogni tanto la propria parola già mentre impara, oppure
  dandogli un voto sulla traduzione intera; il primo rimedio però si può vincere
  ignorando quello che si è appena scritto, che è la cosa che un traduttore non
  può fare.
- Prendere ogni volta la parola più probabile è **miope**, perché la strada
  che parte peggio può arrivare meglio: la **beam search** tiene aperte le
  poche strade più promettenti e decide qualche passo più avanti (con una
  correzione che le impedisce di preferire sempre le frasi corte).
- Un testo generato si giudica confrontandolo con quello di una persona, e il
  **verso** del confronto dipende da quale errore costa. In traduzione costa
  inventare, e si guarda quante parole della macchina stanno nella versione
  umana (**BLEU**); in un riassunto costa saltare, e si guarda quante parole
  della versione umana stanno in quella della macchina (**ROUGE**). Chi legge
  un punteggio senza sapere in che verso è fatto legge un numero e basta.
- Nel 2016 la traduzione neurale entra in produzione con GNMT di Google; nel
  2017 *Attention Is All You Need* manda in soffitta la lettura passo dopo
  passo e tiene solo l'attenzione.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Un **modello di linguaggio** assegna probabilità alla parola successiva; la
  sua qualità si misura con la **perplessità** $2^H$: il numero di alternative
  equiprobabili tra cui esita a ogni passo.
- Le **RNN bidirezionali** leggono la frase nei due sensi: preziose per
  *capire*, inutilizzabili per *generare* (il futuro non esiste ancora: il
  decoder è unidirezionale per costruzione).
- **Seq2seq**: un encoder comprime la frase sorgente in un vettore di
  contesto, un decoder la riscrive nell'altra lingua. Il vettore fisso è un
  **collo di bottiglia** sulle frasi lunghe.
- L’**attenzione di Bahdanau** lo elimina: a ogni passo il decoder rivede
  *tutti* gli stati dell'encoder con pesi $\alpha_{ij}$ appresi; è il
  precursore diretto della *scaled dot-product attention* dei Transformer.
- L'addestramento condiziona su $y_{<i}$ di riferimento (**teacher forcing**),
  che è la forma esatta della massima verosimiglianza sulle coppie, token di
  fine frase compreso, e rende noti in anticipo gli ingressi del decoder: su una
  RNN restano comunque $m$ passi in sequenza, con una maschera causale il
  passaggio diventa uno solo. In generazione il condizionamento è su
  $\hat{y}_{<i}$, cioè su prefissi fuori dal supporto dei dati: è l’**exposure
  bias** {cite}`ranzato2016sequence`, e si misura come scarto fra l'errore per
  token con prefisso vero e quello con prefisso proprio, non sulla quota di
  sequenze intatte, che scende uguale anche senza. Rimedi: *scheduled sampling*
  {cite}`bengio2015scheduled`, il cui obiettivo è però improprio e lo stimatore
  inconsistente {cite}`huszar2015how`, e MIXER, che mescola entropia incrociata
  e gradiente di policy invece di sostituirla.
- In generazione la scelta **greedy** è miope; la **beam search** tiene
  aperte le $k$ ipotesi migliori (con una *length penalty* per non penalizzare
  le frasi lunghe).
- **BLEU** {cite}`papineni2002bleu` è precisione di $n$-grammi con *clipping*,
  frenata dalla *brevity penalty*: definito sul corpus, dipendente dal
  protocollo, cieco alla parafrasi. Nel 2014 la rete pura ($34{,}8$) supera il
  sistema statistico di riferimento ($33{,}3$) ma non lo stato dell'arte
  ($37{,}0$).
- Il gemello per i riassunti è **ROUGE** {cite}`lin2004rouge`: un **richiamo**
  sugli $n$-grammi del riferimento, perché là il peccato è l'omissione. Con un
  riferimento solo è la frazione di BLEU col denominatore scambiato; con più
  riferimenti no, perché BLEU taglia sul massimo e ROUGE somma. La **ROUGE-L**
  usa la sottosequenza comune più lunga, che vede l'ordine senza pretendere la
  contiguità e non chiede di fissare $n$; si riportano le $F_1$ perché un
  richiamo puro premia il candidato lungo.
- Nel 2016 la traduzione neurale entra in produzione con GNMT; nel 2017
  *Attention Is All You Need* fa cadere la ricorrenza.
```
`````
