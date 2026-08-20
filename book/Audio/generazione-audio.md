# Generare suono e musica

La sezione precedente ci ha lasciato in mano l'alfabeto: un codec neurale sa
ridurre un secondo di suono a qualche centinaio di simboli, e sa rifare il suono
a partire da quelli. Finora l'abbiamo usato per **comprimere**, cioè per
riscrivere un suono che già esisteva. Ma quei simboli non sanno da dove
arrivano: se qualcuno gliene passa una fila inventata, il decoder la trasforma
in suono lo stesso. Prendiamo l'idea alla lettera. Se un secondo di
musica si può trascrivere in una manciata di token, allora la stessa macchina
che indovina la parola dopo «Il gatto nero salta sul…» può indovinare, con lo
stesso identico meccanismo, il *suono* che viene dopo: un token audio alla
volta. Cambia l'alfabeto; la macchina resta quella.

È un passo breve e potente, ed è il ponte che questa sezione attraversa: dal
riscrivere un suono che c'è già all'inventarne uno che non c'è, fino a un
sistema a cui si chiede «una
ballata malinconica al pianoforte» e che la compone. Ma prima dei token c'è
stata una strada più letterale, quasi brutale: generare direttamente l'onda,
campione dopo campione. Conviene partire da lì, perché è la pietra miliare che
ha convinto tutti che una rete *poteva* davvero fabbricare suono.

## La via storica: WaveNet, campione per campione

**WaveNet** {cite}`oord2016wavenet`, presentata da DeepMind nel 2016, è la prima
rete che genera audio grezzo con una qualità mai sentita prima, e lo fa nel modo
più diretto possibile: produce un campione dell'onda alla volta, ciascuno sulla
base di tutti quelli già prodotti.

La ritroveremo nel {doc}`capitolo sul riconoscimento vocale </SpeechRecognition/overview>`, dove farà l'ultimo pezzo
della voce sintetica: quello che riprende l'immagine del suono e ne ricava
l'onda vera e propria, che poi esce dagli altoparlanti (un pezzo di macchina che
si chiama *vocoder*). Ma la sua origine è qui.

`````{tab} Elementare

Immagina di disegnare un'onda sonora su carta millimetrata, puntino per
puntino, da sinistra a destra. Ogni puntino è l'altezza dell'onda in
quell'istante; per decidere dove metterlo guardi tutti quelli che hai già
segnato, così la curva resta coerente. WaveNet fa esattamente questo, ma i
puntini da mettere sono **sedicimila al secondo**: tanti quante sono le
misure al secondo con cui si tratta di solito la voce (i $16\,000$ della prima
sezione). Disegnare un minuto di musica vuol dire
piazzarne quasi un milione, uno dopo l'altro, in fila: e siccome ognuno
dipende dai precedenti, non si può correre avanti, bisogna aspettare che il
puntino di prima sia pronto. Ecco perché WaveNet, pur suonando benissimo, era
proverbialmente lenta. Quanto lenta l'articolo non lo dice, ma un'idea la dà il
seguito che DeepMind pubblicò l'anno dopo, *Parallel WaveNet*: nasce apposta per
rimediare, e si presenta dicendo di generare più di venti volte più in fretta di
quanto il suono duri. L'originale, quella soglia, non la vedeva nemmeno da
lontano. Il difetto non è la qualità, è il ritmo del pennino.

`````

`````{tab} Superiore

WaveNet fattorizza la probabilità dell'onda $\mathbf{x} = (x_1, \dots, x_T)$ in modo
autoregressivo, come un modello di linguaggio sui campioni:

$$
p(\mathbf{x}) = \prod_{t=1}^{T} p(x_t \mid x_1, \dots, x_{t-1}),
$$

dove $x_t$ è il campione audio al passo $t$ (è la forma d'onda campionata della
prima sezione, dove i singoli campioni si scrivevano $x[n]$). Due scelte
architetturali rendono
il tutto praticabile. La prima è la **quantizzazione**: predire un valore
reale continuo sarebbe scomodo, così l'ampiezza viene ridotta a $256$ livelli
(un byte per campione) con la compansione **$\mu$-law** (di cui diciamo tra
poco) e la rete emette una softmax su quelle $256$ classi. La seconda sono le
**convoluzioni causali dilatate**: «causali» perché ogni campione vede solo il
passato (mai il futuro, che non esiste ancora), «dilatate» perché a ogni
strato il filtro salta un numero crescente di campioni ($1, 2, 4, 8, \dots$)
raddoppiando. Così il **campo recettivo** cresce in modo *esponenziale* con la
profondità: pochi strati bastano a coprire migliaia di campioni, cioè
centinaia di millisecondi di contesto, senza il costo di una convoluzione
fitta su tutta la finestra. Resta però il limite strutturale
dell'autoregressione *sui campioni grezzi*: la generazione richiede $T$ passi
sequenziali, sedicimila per ogni secondo a $16$ kHz. Nulla di
parallelizzabile in inferenza, ed è precisamente il collo di bottiglia che la
via dei token, qui sotto, aggira.

`````

La compansione $\mu$-law merita una sosta, perché è il trucco che permette a
WaveNet di descrivere ogni puntino dell'onda scegliendolo fra appena $256$
valori possibili. Il nome è ostico due volte, e conviene scioglierlo subito.
«Compansione» non è un refuso per «compressione»: è una parola composta
(*com*primere ed es*pandere*) e dice che il segnale si comprime prima di
misurarlo e si riespande dopo. E $\mu$ è la lettera greca «mi», che qui fa solo
da nome a una manopola: quanto forte si comprime. Il punto di partenza è una
proprietà dell'orecchio: siamo
sensibili alle variazioni *relative* del suono, e un fruscio debole va
conservato con la stessa cura di un colpo forte.

`````{tab} Elementare

Immagina di misurare le altezze dell'onda con un righello a tacche. Se le
tacche sono tutte alla stessa distanza, i suoni forti ne hanno d'avanzo e i
sussurri finiscono schiacciati fra una tacca e l'altra: arrotondati male,
escono dal disegno coperti da un fruscio. Il trucco della $\mu$-law è spostare
le tacche: fitte dove il suono è debole, più rade dove è forte, così ogni
suono viene arrotondato con la cura che merita. Su una nota che sfuma fino
quasi al silenzio, il righello con le tacche spostate restituisce l'onda quasi
intatta proprio nelle code più delicate, dove quello a tacche uguali la affoga
nel fruscio.

E c'è un rovescio della medaglia che vale la pena guardare, perché insegna
qualcosa sulle misure. Le tacche larghe che la $\mu$-law lascia sui picchi
fanno sì che, sul singolo errore più grosso di tutto il brano, il righello a
tacche uguali sia **cinque volte migliore**. Una misura peggiora di cinque
volte, e il suono migliora lo stesso: perché quell'errore più grosso capita
dove c'è un colpo forte, e un colpo forte copre da sé il proprio difetto,
mentre nel silenzio non c'è niente che copra niente.

Per WaveNet, infine, c'è un guadagno pratico. Indovinare ogni puntino
scegliendolo fra $256$ possibilità è un test a crocette gestibile, e $256$ è
esattamente quello che si scrive con **otto** di quelle risposte sì/no della
sezione sui codec: otto bit, che messi insieme si chiamano un **byte**. La
registrazione di partenza, quella dei CD, ne usa sedici, e sedici risposte sì/no
fanno $65\,536$ possibilità: un test a crocette con sessantacinquemila risposte,
per ogni singolo puntino, non sarebbe gestibile affatto.

`````

`````{tab} Superiore

La quantizzazione lineare, con i suoi gradini tutti uguali, spreca precisione
sui suoni forti e ne lascia troppo poca ai deboli. La $\mu$-law comprime il
segnale con un logaritmo *prima* di quantizzare, dedicando più livelli alle
ampiezze piccole; in decodifica si applica la trasformazione inversa. La
formula è

$$
F(x) = \operatorname{sign}(x)\,
\frac{\ln\!\left(1 + \mu\,|x|\right)}{\ln\!\left(1 + \mu\right)},
\qquad \mu = 255,
$$

dove $x \in [-1, 1]$ è il campione normalizzato, $\operatorname{sign}$ ne
conserva il segno e $\mu = 255$ è il parametro di compressione (con $256$
livelli, cioè $8$ bit). Verifichiamolo su una nota che sfuma quasi al
silenzio, un caso con ampia gamma dinamica, dove la differenza si vede:

```python
import numpy as np

mu = 255  # 8 bit -> 256 livelli, come nel WaveNet originale

def comprimi(x):                 # mu-law: schiaccia verso le ampiezze piccole
    return np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)

def espandi(y):                  # operazione inversa
    return np.sign(y) * ((1 + mu) ** np.abs(y) - 1) / mu

def a_8bit(v):                   # da [-1, 1] a 256 livelli interi e ritorno
    q = np.round((v + 1) / 2 * mu).astype(int)     # 0..255: un byte per campione
    return q / mu * 2 - 1

# segnale di prova: una nota che sfuma quasi al silenzio (ampia gamma dinamica)
t = np.linspace(0, 1, 16000, endpoint=False)       # 1 s a 16 kHz
x = (np.sin(2*np.pi*220*t) + 0.5*np.sin(2*np.pi*440*t)) * np.exp(-5*t)
x = x / np.max(np.abs(x))                          # normalizza in [-1, 1]

x_mulaw   = espandi(a_8bit(comprimi(x)))           # 8 bit CON compansione mu-law
x_lineare = a_8bit(x)                              # 8 bit SENZA (quantizzazione lineare)

def snr_segmentale(x, xh, win=320):                # SNR medio su finestre di 20 ms
    n = (len(x) // win) * win
    ps = np.sum(x[:n].reshape(-1, win)**2, axis=1)
    pe = np.sum((x[:n]-xh[:n]).reshape(-1, win)**2, axis=1)
    m = (ps > 1e-9) & (pe > 1e-12)                 # ignora i frame di puro silenzio
    return np.mean(10*np.log10(ps[m]/pe[m]))

print(f"errore massimo (mu-law):    {np.max(np.abs(x - x_mulaw)):.4f}")
print(f"errore massimo (lineare):   {np.max(np.abs(x - x_lineare)):.4f}")
print(f"SNR segmentale mu-law:      {snr_segmentale(x, x_mulaw):.1f} dB")
print(f"SNR segmentale lineare:     {snr_segmentale(x, x_lineare):.1f} dB")
```

```text
errore massimo (mu-law):    0.0201
errore massimo (lineare):   0.0039
SNR segmentale mu-law:      36.8 dB
SNR segmentale lineare:     26.3 dB
```

L'errore massimo di ricostruzione resta attorno all’$1\%$ dell'escursione
picco-picco (il segnale vive in $[-1, 1]$, quindi l'escursione è 2 e l'errore
$0{,}0201$): con un solo byte per campione l'onda torna indietro quasi intatta.
Si noti però
il rovescio, che è la parte istruttiva: sull'errore *massimo* la quantizzazione
lineare è **cinque volte migliore** ($0{,}0039$ contro $0{,}0201$), perché la
$\mu$-law spende i suoi gradini sulle ampiezze piccole e ne lascia di più larghi
sui picchi. È il baratto voluto, ed è anche una lezione sulle metriche: qui una
misura peggiora di cinque volte e il suono migliora. Il
confronto per finestre (l’**SNR segmentale**, che misura il rapporto
segnale/rumore mediando su spezzoni di $20$ ms e quindi pesa allo stesso modo
i tratti forti e quelli deboli) premia nettamente la $\mu$-law: dieci decibel
di vantaggio, tutti guadagnati sulle code sommesse dove la quantizzazione
lineare annega il suono nel rumore di gradino. La ragione per cui WaveNet si
ferma a $8$ bit, dichiarata nel paper, è però computazionale prima che
percettiva: una softmax su $256$ classi è trattabile, una sui $65\,536$
livelli dei $16$ bit no; la compansione serve a rendere quel risparmio quasi
indolore sul parlato. Non è un pranzo gratis: i successori ad alta fedeltà
torneranno ai $16$ bit, ma per strade diverse. Parallel WaveNet abbandona la
softmax e modella il campione con una **miscela di logistiche**, una densità
continua che non ha bisogno di una classe per livello; WaveRNN resta invece
sulla softmax e la **sdoppia**, una sugli $8$ bit più significativi (la parte
*grossolana*) e una sugli $8$ meno significativi (la parte *fine*),
condizionata sulla prima: due distribuzioni da $256$ classi al posto di una da
$65\,536$.

`````

## La svolta: generare token, non campioni

Il problema di WaveNet non è *cosa* genera, ma *quanti* passi gli servono: uno
per campione, oltre sedicimila al secondo. E se invece di predire il campione
grezzo predicessimo un'unità molto più «densa»: un token che riassume diversi
millisecondi di suono? È esattamente ciò che offre un **codec neurale**: un
autoencoder addestrato a comprimere la forma d'onda in una sequenza *corta* di
token discreti e a ricostruirla da quelli. I token nascono dallo stesso
principio di quantizzazione visto nella sezione precedente, portato alle sue
conseguenze: l'audio diventa qualche centinaio di simboli al secondo invece di
decine di migliaia di campioni.

Ma il numero che conta davvero non è quello: è quanti **passi in fila** servono,
perché è la fila a costare. Prendiamo il codec di MusicGen, il generatore di musica di cui si dice qui
sotto. È lo stesso EnCodec dei codec neurali,
regolato però per la musica: taglia il suono in 50 frame al secondo invece di
75, e per ogni frame produce quattro token invece di otto. Quei quattro il
modello li tira fuori in un colpo solo, con un accorgimento che vedremo. Quindi
i passi in fila sono una cinquantina per ogni secondo di musica, contro i
16.000 di WaveNet: sedicimila diviso cinquanta fa **trecentoventi**.

Una precisazione onesta, perché il numero non va preso alla lettera: i passi non
costano uguale, e un passo di Transformer è ben più pesante di un passo di
WaveNet. Il guadagno vero non è esattamente trecentoventi volte. Ma l'ordine di
grandezza è quello, ed è la differenza fra una strada percorribile e una che non
lo è.

Con l'audio ridotto a token, la generazione cambia natura. Non serve più una
rete su misura per le onde: basta un **Transformer** che li produca in
sequenza, e poi il *decoder* del codec li ritrasforma in suono. Il modo in cui
li produce lo abbiamo già visto per il testo: il modello indovina il token
successivo, quel token gli rientra davanti insieme a tutti i precedenti, e si
ricomincia da capo. Si chiama generazione **autoregressiva**, ed è il motore dei
grandi modelli di linguaggio (gli **LLM**, *large language model*). La via dei
token ha un
precursore: già nel 2020 **Jukebox** {cite}`dhariwal2020jukebox`, di OpenAI,
riduceva la musica a token e li faceva scrivere in sequenza a un Transformer,
arrivando a minuti di canzone con tanto di voce, anche se su quella durata la
struttura del brano si sfilacciava. Ma il sistema che porta la ricetta a
maturazione è **AudioLM** {cite}`borsos2023audiolm`, di Google: il
titolo stesso, *a Language Modeling Approach to Audio Generation*, dichiara il
programma. Il suo contributo chiave non è l'idea in sé, ma capire che *un
solo* tipo di token non basta: ne servono due, uno che tenga la struttura del
pezzo e uno che ne porti il suono.

`````{tab} Elementare

Prova a immaginare come un musicista scrive un brano. Prima abbozza la
struttura: la melodia, l'andamento, dove sale e dove scende (lo scheletro che
tiene insieme il pezzo dall'inizio alla fine). Solo dopo riempie quello
scheletro di *suono vero*: il timbro del pianoforte, il riverbero della sala,
il modo in cui una nota si spegne. Sono due lavori diversi, e conviene farli
in quest'ordine, perché se parti dal timbro senza una struttura ti perdi in
bei suoni che non vanno da nessuna parte.

AudioLM fa proprio così, con due tipi di token. I primi (chiamiamoli token
della **struttura**) catturano l'ossatura a lungo termine: che cosa viene
detto o suonato, in che ordine. I secondi (i token del **suono**) aggiungono
il dettaglio fine: la voce precisa, il colore, la grana. Il modello genera
prima la struttura, dall'inizio alla fine, e solo dopo la riveste di suono. Il
risultato ha insieme le due qualità che, prese da sole, si escludevano:
**coerenza** (il brano ha un filo, non deraglia dopo pochi secondi) e
**fedeltà** (suona come audio vero, non come un'imitazione metallica).

`````

`````{tab} Superiore

La novità di AudioLM sta proprio qui: mettere insieme i **token semantici** di
un modello auto-supervisionato e i **token acustici** di un codec neurale, due
famiglie con ruoli complementari, in quello che il paper chiama uno *schema di
tokenizzazione ibrido*. I **token
semantici** provengono da un modello auto-supervisionato della famiglia vista in
[Imparare dal suono senza etichette](rappresentazioni-auto-supervisionate.md);
nel paper è w2v-BERT, parente stretto di wav2vec 2.0
e HuBERT: catturano il contenuto fonetico e la struttura a lungo termine, ma
buttano via gran parte del dettaglio acustico. I **token acustici** vengono
invece dal codec neurale (SoundStream): codificano il segnale in modo da
poterlo *ricostruire* fedelmente (timbro, identità di chi parla, condizioni di
registrazione). La generazione è **gerarchica**, una cascata che si può
leggere come una fattorizzazione della probabilità dell'audio:

$$
p(\text{audio}) \;\approx\;
\underbrace{\prod_{t} p\!\left(s_t \mid s_{<t}\right)}_{\text{struttura}}
\;\cdot\;
\underbrace{\prod_{t} p\!\left(y_t \mid y_{<t},\, s\right)}_{\text{suono}},
$$

dove $s = (s_1, s_2, \dots)$ sono i token semantici e $y = (y_1, y_2, \dots)$
gli acustici: un primo Transformer genera l'intera sequenza semantica
$s$ (la struttura), un secondo genera quella acustica $y$ *condizionata* su
$s$ (il suono). Nel paper la fase acustica è a sua volta spezzata in due lungo
i livelli della RVQ, uno stadio *grossolano* sui primi codebook e uno *fine*
sui restanti: tre Transformer in cascata in tutto. E va notato che l'indice
$t$ scorre su due griglie diverse: token semantici e acustici hanno frequenze
di frame differenti, quindi le sequenze $s$ e $y$ non sono allineate una a
una. Perché separare? Perché i due obiettivi tirano in direzioni
opposte. Predire direttamente i token acustici darebbe fedeltà ma, senza una
guida a lungo termine, la generazione perde il filo dopo pochi secondi;
predire solo i semantici darebbe coerenza ma un suono povero. La cascata mette
la coerenza a monte e la fedeltà a valle, ottenendo entrambe. AudioLM lo
dimostra continuando sia parlato sia brani di pianoforte a partire da pochi
secondi di traccia, mantenendo identità e stile del frammento iniziale.

`````

## Testo → musica: MusicGen

AudioLM genera *continuazioni*: gli dai un inizio, lo prosegue. Ma la domanda
che ha reso la generazione musicale un fenomeno è un'altra: posso *descrivere
a parole* la musica che voglio e ottenerla? La risposta più netta arriva nel
2023 da Meta con **MusicGen** {cite}`copet2023simple`, il cui titolo (*Simple
and Controllable Music Generation*) rivendica proprio la semplicità: **un
singolo** Transformer autoregressivo, non una cascata, che genera i token di
un codec (EnCodec) condizionato da una descrizione testuale.

Guidare una generazione con una descrizione scritta si chiama
**condizionare**, ed è la stessa idea che ritroveremo con i modelli di
diffusione. La descrizione («un riff di chitarra elettrica anni Settanta,
ritmo incalzante») viene tradotta in numeri da un modello che sa leggere il
testo, e quei numeri restano lì accanto per tutta la generazione, a tirare i
token prodotti verso ciò che si è chiesto. Ciò che MusicGen deve risolvere in
più è un dettaglio tecnico del codec, e conviene capirlo perché è lì che sta
l'ingegno.

`````{tab} Elementare

C'è un intoppo pratico. Per comprimere bene la musica, un codec non descrive
ogni istante con un solo token, ma con una **pila** di token: il primo dà
l'abbozzo grezzo del suono, il secondo corregge ciò che il primo ha sbagliato,
il terzo affina ancora, e così via, come un pittore che parte da una macchia
di colore e la ripassa più volte per avvicinarsi alla tinta giusta. Ottima
idea per la qualità, ma un guaio per chi genera: a ogni istante non c'è *un*
token da indovinare, ce ne sono quattro sovrapposti. (Quattro non è un numero
magico: è quello scelto da MusicGen. Di più darebbe un suono più fedele e più
roba da generare, di meno il contrario.) Metterli tutti in fila,
uno dopo l'altro, allungherebbe la sequenza di quattro volte e renderebbe
tutto lentissimo; produrli tutti insieme in un colpo solo, invece, ignorerebbe
il fatto che il secondo dipende dal primo.

MusicGen trova la via di mezzo, e per capirla bisogna guardare *che cosa vede*
ciascun token nel momento in cui viene prodotto. L'idea è **sfalsare** i quattro
flussi di un passo l'uno dall'altro, come le voci di un canone che entrano una
dopo l'altra. A ogni giro il modello emette ancora quattro token in un colpo
solo, ma non sono più i quattro dello *stesso* istante: sono il primo
dell'istante di adesso, il secondo dell'istante prima, il terzo di due istanti
fa, il quarto di tre. Ed è tutto lì: il secondo token dell'istante prima può
guardare il primo di quello stesso istante, perché quello è già uscito, un giro
fa. Nessuno finge più che siano indipendenti, e la fila non si allunga di
quattro volte: si allunga di tre posizioni in tutto.

`````

`````{tab} Superiore

EnCodec usa la **quantizzazione vettoriale residua** (RVQ): ogni frame è
codificato da $N$ codebook in cascata (la $N$ della sezione sui codec), dove
ciascun codebook quantizza il *residuo* lasciato dai precedenti. Il risultato
è che ogni passo temporale $t$
non ha un token ma $N$ token paralleli $(c_t^1, \dots, c_t^N)$: nel setup base
di MusicGen, $N = 4$. Un modello autoregressivo deve decidere in quale ordine
attraversare questa griglia tempo × codebook. Le due opzioni ingenue sono
entrambe cattive: la **linearizzazione** completa
($c_1^1, c_1^2, c_1^3, c_1^4, c_2^1, \dots$) moltiplica la lunghezza della
sequenza per $N$, con costo quadratico che esplode; la predizione **totalmente
parallela** (tutti gli $N$ token di un frame in un colpo) è veloce ma assume
l'indipendenza tra codebook, che è falsa; il residuo dipende per costruzione
da ciò che lo precede. MusicGen adotta invece un **pattern di interleaving**
dei codebook, in particolare uno schema a **ritardo** (*delay*) che sfasa i
flussi di un passo: al tempo $t$ il modello predice
$c_t^1, c_{t-1}^2, c_{t-2}^3, c_{t-3}^4$. Così ogni token può condizionarsi
sui codebook di livello inferiore già emessi, e la sequenza cresce solo di
poche posizioni invece che di un fattore $N$. Il testo entra come
condizionamento: una descrizione codificata da un encoder testuale (un T5)
guida il Transformer via cross-attention, esattamente come un prompt guida un
generatore di immagini a diffusione. Un unico modello, un unico passaggio di
addestramento, controllabile a parole.

`````

## Lo spartito invece del suono

Tutto quello che abbiamo visto finora genera **suono**: onde, oppure token che
un decoder trasforma in onde. C'è una seconda strada, più vecchia e molto più
leggera, e genera lo **spartito**: non la registrazione, le istruzioni. La
differenza si misura. Tre minuti di musica registrati senza compressione sono
circa trenta milioni di byte ($44\,100$ misure al secondo come su un CD, due
byte ciascuna, due canali per lo stereo); gli stessi tre minuti scritti come
note stanno in qualche decina di migliaia. Chi sceglie questa strada non chiede
alla rete di inventare un timbro: le chiede di inventare la **musica**, e a
fare il suono penserà uno strumento, vero o campionato.

Il formato in cui questa strada si scrive esiste dal 1983 e si chiama **MIDI**:
non contiene audio, contiene messaggi del tipo «premi il tasto SOL, con questa
forza» e «lascia il tasto SOL». Un pianoforte elettrico li esegue; un computer
li disegna, di solito come una griglia in cui il tempo scorre in orizzontale e
l'altezza delle note sale in verticale. Quella griglia porta un nome che viene
da lontano, **piano roll**: è il rotolo di carta perforata delle pianole
meccaniche di inizio Novecento, dove i buchi erano le note e la carta scorreva.

E qui arriva il problema vero, che sta tutto nella parola «sequenza». Un
modello che indovina il prossimo token vuole una fila: un simbolo, poi un
altro, poi un altro. La musica una fila non lo è. In un accordo tre note
partono **insieme**, e ciascuna dura per conto proprio: la linea di basso tiene
una nota lunga mentre quella acuta ne fa passare otto. Ognuna di queste linee,
in musica, si chiama **voce**, anche quando a suonarla è uno strumento e non
qualcuno che canta. Mettere più voci su una riga sola è la domanda tecnica di
questa strada, e le risposte sono essenzialmente due.

`````{tab} Elementare

La prima risposta è **fotografare**. Si taglia il tempo in istanti tutti
uguali (per esempio ogni sedicesimo di battuta) e per
ogni istante si scrive che cosa sta suonando ciascuna voce. Viene fuori una
tabella: una riga per istante, una colonna per voce. Poi la si srotola, primo
istante, secondo istante, e via, ed ecco la fila. Funziona, ed è la cosa più
semplice da programmare.

Ha però due difetti, e sono di quelli che si scoprono tardi. Il primo: una
fotografia dice che cosa **sta suonando**, non che cosa è **cominciato**. Un
SOL tenuto per una battuta intera e lo stesso SOL suonato due volte di seguito
danno esattamente la stessa sequenza di fotografie, mentre all'orecchio sono
due cose diverse. Un rimedio ci sarebbe, aggiungere un simbolo che dica «questa
nota sta continuando», ma è un simbolo in più da imparare, e il difetto che
viene adesso resta tale e quale. Il secondo: gli istanti sono per forza tutti uguali, quindi
la griglia decide in anticipo quali durate esistono al mondo. Per poter
scrivere una terzina (tre note nello spazio di due) bisogna tagliare il tempo
più fine, e la fila si allunga per tutti, anche per le battute che di terzine
non ne hanno nessuna.

La seconda risposta è **raccontare** invece di fotografare. Non si dice che
cosa c'è, si dice che cosa **accade**: «parte il SOL», «finisce il SOL»,
«aspetta mezzo secondo». Tre soli tipi di frase, e ci si scrive qualunque
polifonia: le note che partono insieme sono semplicemente due frasi di fila,
senza attesa in mezzo. Sparisce il primo difetto, perché ora «comincia» e «sta
suonando» sono due cose diverse e si scrivono diversamente. E si allenta il
secondo: l'attesa è un numero che si scrive accanto agli eventi, quindi la fila
si allunga solo dove succede qualcosa, invece di infittirsi per tutta la
musica. Resta posto anche per un quarto tipo di frase, quella che dice *come*
si suona invece di *che cosa*: «da qui in poi più forte».

Una cosa però si perde, ed è quella che la griglia dava gratis: dov'è il
battere. Nella tabella ogni casella cadeva su un punto preciso della battuta;
in una fila di attese misurate in millisecondi la battuta non è scritta da
nessuna parte, e il modello deve indovinarla dai numeri. Per la musica a ritmo
regolare, il pop per dire, questo non basta, e chi la genera il battere ce
l'ha rimesso: eventi che dicono «qui comincia una battuta» e «siamo al terzo
sedicesimo».

`````

`````{tab} Superiore

Le due risposte hanno un nome, e la differenza è quella fra una codifica **di
stato** e una codifica **di evento**.

Nella prima, la *griglia*, il tempo è discretizzato in $T$ passi e a ogni
passo si registra l'altezza suonata da ciascuna delle $V$ voci: la sequenza è
la serializzazione di una matrice $T \times V$ e ha lunghezza $T \cdot V$. La
codifica è **ambigua sugli attacchi**: una nota tenuta per $2k$ passi e due
note uguali da $k$ passi ciascuna producono la stessa matrice, quindi la mappa
non è iniettiva e nessun modello addestrato su questa codifica può imparare la
distinzione, perché nei dati non c'è più. Il rimedio consueto è un simbolo
dedicato di *hold*, oppure un canale separato per gli attacchi: allarga il
vocabolario e lascia intatto il difetto che segue. Il passo $\Delta t$ va inoltre
fissato a priori e vincola l'insieme delle durate rappresentabili: ammettere le
suddivisioni ternarie richiede un $\Delta t$ tre volte più fine, quindi
sequenze tre volte più lunghe, con l'attenzione che costa quadraticamente nella
lunghezza.

Nella seconda si emette una sequenza di istruzioni. Il vocabolario diventato
standard è quello di Oore e colleghi {cite}`oore2018time`, ripreso dal Music
Transformer {cite}`huang2019music`: $128$ eventi `NOTE_ON` (uno per altezza
MIDI), $128$ `NOTE_OFF`, $100$ `TIME_SHIFT` (avanzamenti da $10$ ms a un
secondo, a passi di $10$ ms) e $32$ livelli di `VELOCITY` (nel MIDI la *forza*
con cui il tasto è premuto, non una velocità; il valore vale per le note che
seguono), per un totale di $388$
simboli. La polifonia è gratis (eventi consecutivi senza `TIME_SHIFT` in mezzo
sono simultanei), le durate sono esplicite, e l'espressività (dinamica,
micro-ritardi dell'esecuzione) entra nello stesso alfabeto invece di
richiedere un canale a parte. Il guadagno è misurato: sulle esecuzioni
pianistiche della Piano-e-Competition, un minuto di musica a risoluzione di
$10$ ms sta in circa $2000$ eventi, contro i $6000$–$18\,000$ di una griglia
fissa che porti gli stessi attributi espressivi {cite}`huang2019music`. Il
prezzo è che la lunghezza della sequenza non è
più proporzionale al tempo ma alla **densità di eventi**: un passaggio
virtuosistico occupa molti più token di una nota lunga, e la finestra di
contesto si consuma in modo non uniforme. E ce n'è un secondo, più sottile: con
il solo `TIME_SHIFT` il metro non compare da nessuna parte, e sulla musica a
metro regolare i modelli addestrati così tengono il tempo peggio. È
la ragione della codifica **REMI** del *Pop Music Transformer*
{cite}`huang2020pop`, che rimpiazza gli avanzamenti liberi con coppie `Bar`/`Position` su
una griglia di sedicesimi e rende esplicita la durata di ogni nota: non un
ritorno alla griglia, ma un alfabeto di eventi che si porta dentro il metro.

`````

I due difetti della griglia non sono un'opinione, e per vederli bastano due
funzioni corte. Prendiamo due battute identiche salvo una nota, tokenizziamole
in entrambi i modi e contiamo. (Per leggibilità le attese sono qui in sedicesimi
invece che in millisecondi: cambia l'unità, non il meccanismo.)

```python
# Due battute uguali in tutto tranne una nota. Sotto, un DO grave tenuto per
# tutta la battuta; sopra, un SOL: nel primo caso tenuto anch'esso, nel secondo
# ribattuto a meta'. All'orecchio sono due cose diverse.
# Una nota e' (altezza MIDI, istante d'inizio, durata); i tempi in sedicesimi.
tenuta    = [(67, 0, 16), (48, 0, 16)]              # SOL tenuto per 16 sedicesimi
ribattuta = [(67, 0, 8), (67, 8, 8), (48, 0, 16)]   # SOL suonato due volte

def a_griglia(note, passi=16, voci=2):
    """Fotografia: per ogni istante l'altezza che ciascuna voce sta suonando."""
    griglia = [[0] * voci for _ in range(passi)]     # 0 = silenzio
    scala = passi // 16                              # quanti passi vale un sedicesimo
    for altezza, inizio, durata in note:
        v = 0 if altezza >= 60 else 1                # voce acuta / voce grave
        for t in range(inizio * scala, (inizio + durata) * scala):
            griglia[t][v] = altezza
    return [x for riga in griglia for x in riga]     # srotolata istante per istante

def a_eventi(note):
    """Ricetta: che cosa accade, e quanto si aspetta fra un fatto e il successivo."""
    fatti = []
    for altezza, inizio, durata in note:
        fatti.append((inizio, f"NOTE_ON<{altezza}>"))
        fatti.append((inizio + durata, f"NOTE_OFF<{altezza}>"))
    sequenza, adesso = [], 0
    for istante, evento in sorted(fatti):
        if istante > adesso:
            sequenza.append(f"TIME_SHIFT<{istante - adesso}>")
            adesso = istante
        sequenza.append(evento)
    return sequenza

g1, g2 = a_griglia(tenuta), a_griglia(ribattuta)
e1, e2 = a_eventi(tenuta), a_eventi(ribattuta)

print(f"a griglia: {len(g1)} token ciascuno, e sono identici? {g1 == g2}")
print(f"a eventi:  {len(e1)} e {len(e2)} token, e sono identici? {e1 == e2}")
print(f"  tenuta    -> {' '.join(e1)}")
print(f"  ribattuta -> {' '.join(e2)}")
print(f"la stessa battuta, se la griglia deve reggere le terzine: "
      f"{len(a_griglia(tenuta, passi=48))} token invece di {len(g1)}")
```

```text
a griglia: 32 token ciascuno, e sono identici? True
a eventi:  5 e 8 token, e sono identici? False
  tenuta    -> NOTE_ON<48> NOTE_ON<67> TIME_SHIFT<16> NOTE_OFF<48> NOTE_OFF<67>
  ribattuta -> NOTE_ON<48> NOTE_ON<67> TIME_SHIFT<8> NOTE_OFF<67> NOTE_ON<67> TIME_SHIFT<8> NOTE_OFF<48> NOTE_OFF<67>
la stessa battuta, se la griglia deve reggere le terzine: 96 token invece di 32
```

La prima riga è il difetto: trentadue token per parte, **e sono gli stessi**,
cioè la griglia ha buttato via una differenza che qualunque orecchio sente. La
seconda mostra il rimedio: cinque token contro otto, e questa volta diversi. Il
punto esatto in cui differiscono si legge nelle due righe successive, ed è il
`NOTE_OFF<67>` seguito da un secondo `NOTE_ON<67>` che nella versione tenuta
non c'è. L'ultima riga è il secondo difetto messo in cifre: per lasciare alla
griglia la possibilità di scrivere terzine il conto passa da trentadue a
novantasei token, il triplo, e lo paga tutta la musica.

Quei totali però dicono dove si arriva, non come ci si arriva, e il come è
metà del problema: {numref}`fig-spartito-in-fila` fa scorrere la battuta e
lascia crescere le due file una sotto l'altra. La griglia si allunga di due
caselle a ogni sedicesimo, che ci sia o non ci sia qualcosa da dire; gli
eventi restano fermi per mezza battuta e poi ne aggiungono tre in un colpo
solo, quando il SOL finisce e riparte. È la differenza fra una fila
proporzionale al **tempo** e una proporzionale a **quanto accade**, ed è la
ragione per cui un passaggio fitto di note, nella seconda, riempie la fila
molto più in fretta di una nota lunga.

```{figure} ../figures/spartito-in-fila.svg
:name: fig-spartito-in-fila
:alt: In alto la battuta disegnata come rullo di pianola, con tre rettangoli: un DO grave lungo tutta la battuta e due SOL da mezza battuta ciascuno, separati da uno stacco a metà. Un cursore verticale la percorre da sinistra a destra. Sotto, la fila a griglia, due righe di sedici caselle (voce acuta e voce grave) con dentro l'altezza MIDI di ciò che sta suonando, trentadue in tutto. Più sotto la fila a eventi, otto scatole con i nomi dei token, raccolte nei tre istanti in cui accade qualcosa. A destra due contatori, che arrivano a trentadue e a otto.
:width: 92%

La stessa battuta scritta in tre modi. In alto il rullo (il *piano roll*): tre
rettangoli, il DO grave tenuto per tutta la battuta e i due SOL da mezza
battuta l'uno. In mezzo la fila a griglia, trentadue caselle, due per ogni
sedicesimo, con dentro l'altezza MIDI di ciò che quella voce sta suonando. In
basso la fila a eventi, otto scatole, raccolte nei tre istanti in cui accade
qualcosa. Nella riga della voce acuta i sedici `67` non portano nessuno stacco
a metà battuta: quello che il rullo mostra come due rettangoli separati, e che
fra gli eventi si legge come `NOTE_OFF<67>` seguito da `NOTE_ON<67>`, nella
griglia non c'è più.
```

Chiarito come si mette la musica in fila, la macchina che indovina il simbolo
successivo è quella del {doc}`capitolo sui Transformer </Transformers/overview>`, senza una riga di differenza.
C'è però una ragione per cui proprio qui l'attenzione ha contato più che
altrove, e la dice il titolo del lavoro che l'ha portata nella musica: *Music
Transformer: Generating Music with Long-Term Structure*, di Cheng-Zhi Anna
Huang e colleghi, preprint del settembre 2018 e poi ICLR 2019
{cite}`huang2019music`. La musica è fatta di
**ritorni**: un tema si ripresenta dopo trenta secondi, e trenta secondi sono
circa mille simboli. Una rete ricorrente, che porta con sé un riassunto e lo
aggiorna a ogni passo, a quel punto il tema l'ha dimenticato; l'attenzione può
andarselo a rileggere, ed è lo stesso collo di bottiglia da cui era nata,
raccontato là.

Il contributo tecnico di quel lavoro è però un altro, ed è una lezione
d'ingegneria: l'idea giusta era già pubblicata, e non entrava in memoria.

`````{tab} Elementare

L'idea era guardare a *quanto indietro* nella fila sta un simbolo, invece che
al posto preciso che occupa, e per la musica è proprio ciò che serve: una
battuta fa lo stesso effetto all'inizio o alla fine del pezzo. Il modo in cui
era scritta, però, chiedeva di tenere da parte un appunto per **ogni coppia**
di simboli della fila, e con brani da duemila simboli quel foglio di appunti
pesa $8{,}5$ GB per ogni strato della rete: in una scheda grafica non ci sta,
quindi non si fa. Huang e colleghi si accorgono che il foglio grande non serve
tenerlo: lo si ricava da uno molto più piccolo, facendo scorrere le sue righe,
ognuna di una casella in più della precedente. Il conto scende a $4{,}2$ MB,
circa duemila volte meno, e il risultato è identico. Con quella memoria
liberata il modello arriva a brani di un minuto, attorno ai duemila simboli,
con i temi che tornano davvero.

`````

`````{tab} Superiore

L'attenzione **relativa** di Shaw e colleghi {cite}`shaw2018self` modula i logit
dell'attenzione con la distanza fra le posizioni, ma per farlo materializza un tensore di
rappresentazioni relative indicizzato su ogni coppia, di costo $O(L^2 D)$ in
memoria: per $L = 2048$ e $D = 512$ sono $8{,}5$ GB per strato ($1{,}1$ GB per
testa, con $H = 8$ teste e $D_h = 64$), e su una GPU da $16$ GB la massima
lunghezza addestrabile si ferma a $L = 650$. Huang e colleghi osservano che i
termini che servono si ottengono già da $\mathbf{Q}\mathbf{E}_r^{\top}$, cioè
dal prodotto fra le query e le sole $L$ rappresentazioni di distanza, e che
basta poi uno *skewing* (un riempimento e un rimodellamento che traslano la
riga $i$ di $i$ posizioni) per portare ogni logit al posto giusto. Il termine
intermedio passa da $O(L^2 D)$ a $O(L D)$, cioè da $8{,}5$ GB a $4{,}2$ MB per
strato ($0{,}52$ MB per testa), a parità di risultato, e la lunghezza
addestrabile sale a $L = 3500$ {cite}`huang2019music`. Resta la matrice dei
logit $L \times L$, che nessuna delle due implementazioni evita. Con
$L = 2048$ il contesto copre circa un minuto di esecuzione: è lì che i ritorni
tematici cominciano a essere visibili al modello.

`````

Questa strada non è in concorrenza con l'altra, perché non risponde alla stessa
domanda. Uno spartito il timbro non lo fissa: al più dice quale strumento, e
come quello strumento suoni davvero lo decide chi lo esegue. In compenso è
leggibile, è correggibile nota per
nota da un musicista, e la fila da produrre è più corta: un minuto di
pianoforte sono i duemila simboli di poco fa, mentre un minuto di token audio
con il codec di MusicGen sono cinquanta istanti al secondo per quattro
codebook, cioè dodicimila. Sono due contenuti diversi e il confronto va preso
per quello che è, un ordine di grandezza, non una misura. È il motivo per cui
sopravvive benissimo dove il risultato dev'essere modificato (accompagnamenti,
composizione assistita, colonne sonore da rimaneggiare), mentre la frase che
diventa canzone passa per i token audio e non per lo spartito. E c'è una
morale che vale oltre la musica: la parte difficile non era il modello, era
**decidere come scrivere i dati in fila**. La stessa domanda, con risposte
diverse, torna ogni volta che qualcosa che non è testo va dato in pasto a una
macchina che predice il simbolo successivo.

## Diffusione, e uno sguardo onesto ai limiti

Torniamo al suono. Anche lì la via dei token non è l'unica: c'è un secondo
grande filone, quello dei **modelli di diffusione**, a cui il libro dedica un
capitolo intero più avanti (dove il metodo nasce, per le immagini) e che qui
conviene almeno nominare, perché nell'audio pesa quanto l'altro. L'idea in due
righe: si prende un dato vero e lo si sporca di rumore un po’ alla volta,
finché non resta che rumore; poi si addestra una rete a fare il percorso
inverso, a togliere rumore un passo per volta. Fatto questo, si può partire da
rumore puro e arrivare a un dato nuovo, che nessuno ha mai visto. Il dato,
nell'audio, raramente è l'onda grezza: di solito è il suo **spettrogramma**
(l'immagine tempo-frequenza costruita nella prima sezione del capitolo), che
si può trattare quasi come una figura, oppure il riassunto compatto che
l'encoder di un codec produce prima di arrotondarlo in token, quello che nella
sezione precedente abbiamo chiamato **latente** (è la stessa strategia della
diffusione latente di Stable Diffusion, trasferita al suono). I due filoni
corrono paralleli, e quale dei due convenga dipende dal compito più che
dall'anno.

Detto ciò che funziona, l'onestà impone di dire ciò che ancora non funziona.
La **coerenza a lungo termine** resta fragile: un modello sa produrre trenta
secondi convincenti, ma tenere in piedi la struttura di un brano intero (con
temi che tornano, uno sviluppo, una chiusura) è tuttora un problema aperto, e
il ricorso a token semantici o gerarchie serve proprio ad attenuarlo, non a
risolverlo. Restano poi i difetti tipici del suono fabbricato, che in gergo si
chiamano **artefatti**: un che di metallico che si accende qua e là, gli
attacchi delle note impastati invece che netti, code di riverbero che non
suonano come suonerebbe una stanza vera. Un orecchio allenato li riconosce.

Ma la questione più grande non è tecnica. I modelli di questa sezione imparano
da enormi cataloghi di musica registrata, e questo apre un nodo di **copyright
e consenso** (con le sue cause legali sui dati di addestramento) che
ritroveremo per le immagini generate (nel {doc}`capitolo sui modelli di diffusione </ModelliDiffusione/overview>`)
e per la clonazione vocale (in quello sulla sintesi vocale). A chi appartiene
un brano generato «nello stile di» un artista che non ha mai dato il permesso,
e che non viene pagato? Chi ha diritto sulla musica di addestramento? La voce
è un **dato biometrico**, cioè un dato che identifica una persona e nessun'altra,
come un'impronta digitale; lo stile di un musicista è il lavoro di una vita, e
generarne un surrogato a comando tocca il sostentamento di chi quella musica
la fa. Come sempre in questo libro, lo strumento non sceglie l'uso, ma qui,
più che altrove, le regole del gioco sono ancora tutte da scrivere.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- **WaveNet** {cite}`oord2016wavenet` (2016) disegna l'onda puntino per
  puntino, come su carta millimetrata, guardando ogni volta tutti i puntini già
  messi. Suona benissimo, ma i puntini sono più di sedicimila al secondo e vanno
  in fila uno dopo l'altro: il difetto non è la qualità, è il ritmo del pennino.
- Per farceli stare in $256$ possibilità ciascuno (cioè in un **byte**) si
  sposta il **righello**: tacche fitte dove il suono è debole, più rade dove è
  forte, così anche i sussurri vengono arrotondati con cura. Curiosamente il
  singolo errore più grosso peggiora di cinque volte, e il suono migliora lo
  stesso: capita dove c'è un colpo forte, che il proprio difetto se lo copre da
  solo.
- La svolta è smettere di disegnare puntini e scrivere **token**: un codec
  riassume il suono in poche centinaia di simboli al secondo, e una macchina che
  indovina il simbolo successivo (la stessa che scrive testo) li produce in
  sequenza. Poi il codec li ritrasforma in suono.
- **AudioLM** {cite}`borsos2023audiolm` scopre che di token ne servono due tipi,
  e in quest'ordine: prima quelli della **struttura** (dove va il pezzo), poi
  quelli del **suono** (che timbro ha). Come un musicista che abbozza prima e
  riempie dopo.
- **MusicGen** {cite}`copet2023simple` compone a partire da una **descrizione a
  parole**. A ogni istante il codec produce quattro token sovrapposti, e
  MusicGen li sfalsa di un passo l'uno dall'altro, come le voci di un canone.
- Si può anche generare lo **spartito** invece del suono, e allora la domanda
  difficile è una sola: come si mette su una riga qualcosa in cui più note
  suonano insieme e ciascuna dura per conto suo. Fotografare istante per
  istante è semplice ma confonde una nota lunga con due corte uguali; elencare
  quel che **accade** («parte il SOL», «finisce il SOL», «aspetta») non le
  confonde, ed è quello che si usa, a patto di rimetterci dentro il battere,
  che la griglia dava gratis.
- C'è anche una seconda strada, la **diffusione**: invece di scrivere token, si
  parte da rumore puro e lo si ripulisce un passo alla volta finché non ne esce
  un suono.
- Ciò che ancora non funziona: i brani **lunghi** perdono il filo, restano
  **difetti** che un orecchio allenato sente, e soprattutto c'è la questione di
  **chi possiede** la musica su cui questi modelli hanno imparato, e lo stile
  degli artisti che imitano.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- **WaveNet** {cite}`oord2016wavenet` (2016) genera l'onda **campione per
  campione** in modo autoregressivo, con **convoluzioni causali dilatate** (campo
  recettivo esponenziale) e ampiezza su $256$ livelli via compansione
  **$\mu$-law**. Qualità altissima, ma migliaia di passi sequenziali al secondo:
  lentissimo.
- La svolta è generare **token** invece di campioni: un **codec neurale**
  comprime l'audio in una sequenza corta di simboli, e un **Transformer** li
  produce come un LLM produce parole (poi il decoder del codec li ritrasforma
  in suono). Il guadagno vero è nei **passi sequenziali**: 50 al secondo con il
  codec di MusicGen contro i 16.000 di WaveNet, un fattore ~300.
- **AudioLM** {cite}`borsos2023audiolm` usa due livelli di token in cascata:
  **semantici** (struttura a lungo termine, da modelli auto-supervisionati) e
  **acustici** (dettaglio del suono, dal codec), per avere insieme **coerenza** e
  **fedeltà**.
- **MusicGen** {cite}`copet2023simple` genera i token di EnCodec con un **solo**
  Transformer condizionato dal **testo**. A ogni istante il codec produce
  quattro token sovrapposti: MusicGen li **sfasa di un passo** l'uno dall'altro,
  come le voci di un canone che entrano una dopo l'altra, invece di metterli
  tutti in fila (lento) o di produrli insieme fingendoli indipendenti (falso).
- Nel **simbolico** (MIDI, non audio) il nodo è la codifica. La *griglia*
  serializza una matrice $T \times V$ ma è ambigua sugli attacchi (nota tenuta
  $2k$ e due note da $k$ danno la stessa matrice) e fissa $\Delta t$ a priori.
  La codifica **a eventi** {cite}`oore2018time` la sostituisce con $388$
  simboli ($128$ `NOTE_ON`, $128$ `NOTE_OFF`, $100$ `TIME_SHIFT` a $10$ ms,
  $32$ `VELOCITY`): polifonia e durate esplicite, lunghezza proporzionale alla
  densità di eventi, e metro non rappresentato (che **REMI** rimette fra gli
  eventi con `Bar`/`Position`). Il **Music Transformer** {cite}`huang2019music`
  ci mette sopra l'attenzione relativa, con lo *skewing* che porta il termine
  intermedio da $8{,}5$ GB a $4{,}2$ MB **per strato** ($L = 2048$, $D = 512$;
  per testa, da $1{,}1$ GB a $0{,}52$ MB), e arriva a brani di un minuto.
- La via dei token non è l'unica: c'è anche la **diffusione**, che parte da
  rumore puro e lo ripulisce un passo alla volta finché non ne esce un suono. I
  limiti
  aperti: **coerenza a lungo termine** fragile, **artefatti**, e soprattutto le
  questioni di **copyright e consenso** sulla musica di addestramento e sullo
  stile degli artisti.
```

`````

Questo capitolo ha parlato di tutto il suono tranne uno, e la voce è rimasta
fuori per una ragione e non per dimenticanza. Con la musica e con i suoni
d'ambiente non esiste una continuazione giusta e una sbagliata: lo spartito,
qui, è stato materiale da generare, non un verdetto con cui confrontarsi. Con il
parlato invece una risposta giusta c'è, ed è già scritta: il testo che qualcuno
ha pronunciato davvero. La catena costruita qui ci viene dietro tutta, onda,
spettrogramma, token, e sopra di essi un modello che li produce come
produrrebbe delle parole. «Speech Recognition» la rimette in fila nei due
sensi, dalla voce al testo e dal testo alla voce.
