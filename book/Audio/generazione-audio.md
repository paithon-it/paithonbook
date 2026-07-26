# Generare suono e musica

La sezione precedente si è chiusa con una promessa: le unità discrete del
suono (quell'«alfabeto» che un modello si costruisce ascoltando montagne di
audio) possono diventare un vocabolario su cui *scrivere* suono, come un
modello di linguaggio scrive testo. Prendiamola alla lettera. Se un secondo di
musica si può trascrivere in una manciata di token, allora la stessa macchina
che indovina la parola dopo «Il gatto nero salta sul…» può indovinare, con lo
stesso identico meccanismo, il *suono* che viene dopo: un token audio alla
volta. Cambia l'alfabeto; la macchina resta quella.

È un passo breve e potente, ed è il ponte che questa sezione attraversa: dal
riconoscere il suono al generarlo, fino a un sistema a cui si chiede «una
ballata malinconica al pianoforte» e che la compone. Ma prima dei token c'è
stata una strada più letterale, quasi brutale: generare direttamente l'onda,
campione dopo campione. Conviene partire da lì, perché è la pietra miliare che
ha convinto tutti che una rete *poteva* davvero fabbricare suono.

## La via storica: WaveNet, campione per campione

Lo ritroveremo nel capitolo sul riconoscimento vocale, nella sezione
sulla voce sintetica, dove farà il *vocoder*: quel modello che trasforma un
mel-spettrogramma nell'onda vera e propria. Ma la sua origine è qui. **WaveNet**
{cite}`oord2016wavenet`, presentato da DeepMind nel 2016, è la prima rete che
genera audio grezzo con una qualità mai sentita prima, e lo fa nel modo più
diretto possibile: predice un campione dell'onda alla volta, ciascuno sulla
base di tutti quelli già prodotti.

`````{tab} Elementare

Immagina di disegnare un'onda sonora su carta millimetrata, puntino per
puntino, da sinistra a destra. Ogni puntino è l'altezza dell'onda in
quell'istante; per decidere dove metterlo guardi tutti quelli che hai già
segnato, così la curva resta coerente. WaveNet fa esattamente questo, ma i
puntini da mettere sono **più di sedicimila al secondo**: tanti quante volte
al secondo si misura il suono. Disegnare un minuto di musica vuol dire
piazzarne quasi un milione, uno dopo l'altro, in fila: e siccome ognuno
dipende dai precedenti, non si può correre avanti, bisogna aspettare che il
puntino di prima sia pronto. Ecco perché WaveNet, pur suonando benissimo, era
proverbialmente lento: nella versione originale, generare un secondo di audio
poteva costare minuti di calcolo. Il difetto non è la qualità, è il ritmo del
pennino.

`````

`````{tab} Superiore

WaveNet fattorizza la probabilità dell'onda $a = (a_1, \dots, a_T)$ in modo
autoregressivo, come un modello di linguaggio sui campioni:

$$
p(a) = \prod_{t=1}^{T} p(a_t \mid a_1, \dots, a_{t-1}),
$$

dove $a_t$ è il campione audio al passo $t$. Due scelte architetturali rendono
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
sequenziali, oltre sedicimila per ogni secondo a $16$ kHz. Nulla di
parallelizzabile in inferenza, ed è precisamente il collo di bottiglia che la
via dei token, qui sotto, aggira.

`````

La compansione $\mu$-law merita un esperimento, perché spiega da sola perché
otto bit bastino. L'orecchio è sensibile alle variazioni *relative* del suono:
un fruscio debole va reso con la stessa cura di un colpo forte. La
quantizzazione lineare, con i suoi gradini tutti uguali, spreca precisione sui
suoni forti e ne lascia troppo poca ai deboli. La $\mu$-law comprime il segnale
con un logaritmo *prima* di quantizzare, dedicando più livelli alle ampiezze
piccole; in decodifica si applica la trasformazione inversa. La formula è

$$
F(a) = \operatorname{sign}(a)\,
\frac{\ln\!\left(1 + \mu\,|a|\right)}{\ln\!\left(1 + \mu\right)},
\qquad \mu = 255,
$$

dove $a \in [-1, 1]$ è il campione normalizzato, $\operatorname{sign}$ ne
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
print(f"SNR segmentale mu-law:      {snr_segmentale(x, x_mulaw):.1f} dB")
print(f"SNR segmentale lineare:     {snr_segmentale(x, x_lineare):.1f} dB")
```

```text
errore massimo (mu-law):    0.0201
SNR segmentale mu-law:      36.8 dB
SNR segmentale lineare:     26.3 dB
```

L'errore massimo di ricostruzione resta sotto il $2\%$ dell'escursione totale:
con un solo byte per campione l'onda torna indietro quasi intatta. E il
confronto per finestre (l'**SNR segmentale**, che misura il rapporto
segnale/rumore mediando su spezzoni di $20$ ms e quindi pesa allo stesso modo
i tratti forti e quelli deboli) premia nettamente la $\mu$-law: dieci decibel
di vantaggio, tutti guadagnati sulle code sommesse dove la quantizzazione
lineare annega il suono nel rumore di gradino. È il motivo per cui WaveNet
lavora su otto bit e non su sedici: percettivamente non serve di più.

## La svolta: generare token, non campioni

Il problema di WaveNet non è *cosa* genera, ma *quanti* passi gli servono: uno
per campione, oltre sedicimila al secondo. E se invece di predire il campione
grezzo predicessimo un'unità molto più «densa»: un token che riassume diversi
millisecondi di suono? È esattamente ciò che offre un **codec neurale**: un
autoencoder addestrato a comprimere la forma d'onda in una sequenza *corta* di
token discreti e a ricostruirla da quelli. I token nascono dallo stesso
principio di quantizzazione visto nella sezione precedente, portato alle sue
conseguenze: l'audio diventa una manciata di simboli al secondo invece di
decine di migliaia di campioni.

Con l'audio ridotto a token, la generazione cambia natura. Non serve più una
rete su misura per le onde: basta un **Transformer** che li produca in
sequenza (esattamente il ciclo di generazione autoregressiva visto per gli
LLM, dove il token prodotto rientra come input e si ricomincia) e poi il
*decoder* del codec li ritrasforma in suono. Il primo sistema a mostrare la
potenza di questa idea è **AudioLM** {cite}`borsos2023audiolm`, di Google, nel
2022: il titolo stesso, *a Language Modeling Approach to Audio Generation*,
dichiara il programma. Il suo contributo chiave è capire che *un solo* tipo di
token non basta.

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

AudioLM impiega due famiglie di token con ruoli complementari. I **token
semantici** provengono da un modello auto-supervisionato della famiglia vista
nella sezione precedente, nel paper, w2v-BERT, parente stretto di wav2vec 2.0
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
$s$ (il suono). Perché separare? Perché i due obiettivi tirano in direzioni
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

Il condizionamento sul testo è la stessa idea vista con i modelli di
diffusione: una descrizione («un riff di chitarra elettrica anni Settanta,
ritmo incalzante») viene codificata da un encoder di testo e usata per
*guidare* la generazione, così che i token prodotti realizzino quella
richiesta. Ciò che MusicGen deve risolvere in più è un dettaglio tecnico del
codec, e vale la pena capirlo perché è dove sta l'ingegno.

`````{tab} Elementare

C'è un intoppo pratico. Per comprimere bene la musica, un codec non descrive
ogni istante con un solo token, ma con una **pila** di token: il primo dà
l'abbozzo grezzo del suono, il secondo corregge ciò che il primo ha sbagliato,
il terzo affina ancora, e così via, come un pittore che parte da una macchia
di colore e la ripassa più volte per avvicinarsi alla tinta giusta. Ottima
idea per la qualità, ma un guaio per chi genera: a ogni istante non c'è *un*
token da indovinare, ce ne sono quattro sovrapposti. Metterli tutti in fila,
uno dopo l'altro, allungherebbe la sequenza di quattro volte e renderebbe
tutto lentissimo; produrli tutti insieme in un colpo solo, invece, ignorerebbe
il fatto che il secondo dipende dal primo. MusicGen trova la via di mezzo:
**sfalsa** i quattro flussi di un passo l'uno dall'altro, come le voci di un
canone che entrano una dopo l'altra, così un solo modello può generarli in
parallelo senza fingere che siano indipendenti.

`````

`````{tab} Superiore

EnCodec usa la **quantizzazione vettoriale residua** (RVQ): ogni frame è
codificato da $K$ codebook in cascata, dove il codebook $k$ quantizza il
*residuo* lasciato dai precedenti. Il risultato è che ogni passo temporale $t$
non ha un token ma $K$ token paralleli $(c_t^1, \dots, c_t^K)$: nel setup base
di MusicGen, $K = 4$. Un modello autoregressivo deve decidere in quale ordine
attraversare questa griglia tempo × codebook. Le due opzioni ingenue sono
entrambe cattive: la **linearizzazione** completa
($c_1^1, c_1^2, c_1^3, c_1^4, c_2^1, \dots$) moltiplica la lunghezza della
sequenza per $K$, con costo quadratico che esplode; la predizione **totalmente
parallela** (tutti i $K$ token di un frame in un colpo) è veloce ma assume
l'indipendenza tra codebook, che è falsa; il residuo dipende per costruzione
da ciò che lo precede. MusicGen adotta invece un **pattern di interleaving**
dei codebook, in particolare uno schema a **ritardo** (*delay*) che sfasa i
flussi di un passo: al tempo $t$ il modello predice
$c_t^1, c_{t-1}^2, c_{t-2}^3, c_{t-3}^4$. Così ogni token può condizionarsi
sui codebook di livello inferiore già emessi, e la sequenza cresce solo di
poche posizioni invece che di un fattore $K$. Il testo entra come
condizionamento: una descrizione codificata da un encoder testuale (un T5)
guida il Transformer via cross-attention, esattamente come un prompt guida un
generatore di immagini a diffusione. Un unico modello, un unico passaggio di
addestramento, controllabile a parole.

`````

## Diffusione, e uno sguardo onesto ai limiti

La via dei token non è l'unica. Anche i **modelli di diffusione** (che il
capitolo dedicato mostrerà nascere per le immagini) si applicano all'audio,
con lo stesso schema: si corrompe il dato con rumore e si addestra una rete a
invertire il processo. Il dato, però, raramente è l'onda grezza: di solito è
il suo **spettrogramma** (l'immagine tempo-frequenza costruita nella sezione
sull'elaborazione audio), che si può trattare quasi come una figura, oppure
una rappresentazione **latente** compressa (la stessa strategia della
diffusione latente di Stable Diffusion, trasferita al suono). È un secondo
grande filone, parallelo a quello autoregressivo, e i due si contendono lo
stato dell'arte a seconda del compito.

Detto ciò che funziona, l'onestà impone di dire ciò che ancora non funziona.
La **coerenza a lungo termine** resta fragile: un modello sa produrre trenta
secondi convincenti, ma tenere in piedi la struttura di un brano intero (con
temi che tornano, uno sviluppo, una chiusura) è tuttora un problema aperto, e
il ricorso a token semantici o gerarchie serve proprio ad attenuarlo, non a
risolverlo. Restano poi gli **artefatti**: bagliori metallici, transitori
impastati, code di riverbero innaturali che l'orecchio allenato riconosce.

Ma la questione più grande non è tecnica. I modelli di questa sezione imparano
da enormi cataloghi di musica registrata, e questo apre un nodo di **copyright
e consenso** (con le sue cause legali sui dati di addestramento) che
ritroveremo per le immagini generate (nel capitolo sui modelli di diffusione)
e per la clonazione vocale (in quello sulla sintesi vocale). A chi appartiene
un brano generato «nello stile di» un artista che non ha mai dato il permesso,
e che non viene pagato? Chi ha diritto sulla musica di addestramento? La voce
è un dato biometrico; lo stile di un musicista è il lavoro di una vita, e
generarne un surrogato a comando tocca il sostentamento di chi quella musica
la fa. Come sempre in questo libro, lo strumento non sceglie l'uso, ma qui,
più che altrove, le regole del gioco sono ancora tutte da scrivere.

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
  in suono).
- **AudioLM** {cite}`borsos2023audiolm` usa due livelli di token in cascata:
  **semantici** (struttura a lungo termine, da modelli auto-supervisionati) e
  **acustici** (dettaglio del suono, dal codec), per avere insieme **coerenza** e
  **fedeltà**.
- **MusicGen** {cite}`copet2023simple` genera i token di EnCodec con un **solo**
  Transformer condizionato dal **testo**, gestendo i $K$ flussi paralleli della
  **RVQ** con un pattern di interleaving a ritardo invece di linearizzarli o
  fingerli indipendenti.
- Anche la **diffusione** genera audio (su spettrogramma o su latenti). I limiti
  aperti: **coerenza a lungo termine** fragile, **artefatti**, e soprattutto le
  questioni di **copyright e consenso** sulla musica di addestramento e sullo
  stile degli artisti.
```
