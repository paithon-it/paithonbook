# La voce sintetica: dal testo al parlato

New York, estate del 1939. Nel padiglione della Bell all'Esposizione
universale, una giovane donna siede a una console che ricorda un piccolo
organo: una quindicina di tasti sotto le dita, una barra sotto il polso, un
pedale sotto il piede. Il presentatore invita la macchina a salutare, e dagli
altoparlanti esce una voce, gommosa, un po' subacquea, ma comprensibile:
*«Good afternoon, radio audience»*. Non è una registrazione. Il **Voder** di
Homer Dudley, ingegnere dei Bell Labs, la voce la *genera* sul momento: i
tasti dosano l'energia nelle bande di frequenza, la barra sceglie tra il
ronzio delle vocali e il soffio delle consonanti, il pedale disegna
l'intonazione. La voce, letteralmente, si *suona*, e imparare a suonarla
richiedeva circa un anno di pratica, raccontava Helen Harper, la più abile
delle venti operatrici addestrate per l'occasione.

Quasi novant'anni dopo, la **sintesi vocale** (TTS, *Text-to-Speech*) legge ad
alta voce le indicazioni stradali, gli audiolibri, i messaggi mentre guidiamo.
Il problema è ancora quello del Voder: trasformare simboli scritti in un'onda
di pressione che un orecchio accetti come voce umana. Solo che al posto delle
dita di Helen Harper, oggi, ci sono le reti neurali.

## Il viaggio inverso

Questo capitolo ha percorso finora una sola direzione: dall'onda al testo.
Abbiamo trasformato il suono in numeri, i numeri in spettrogramma, lo
spettrogramma in parole. La sintesi vocale percorre la stessa mappa al
contrario, e il punto d'incontro dei due viaggi è proprio il
**mel-spettrogramma** costruito nel capitolo sull'Audio, nella sezione *Dal
suono alle feature*: quell'immagine a bande che dice, istante per istante,
quanta energia c'è a ciascuna altezza sonora, misurata come la sente
l'orecchio e non come la misurerebbe uno strumento (è questo che significa il
«mel» davanti a «spettrogramma»). L'ASR parte da lì per salire verso le
parole; il TTS ci arriva scendendo dalle parole, e da lì ricostruisce l'onda
({numref}`fig-tts-pipeline`).

Un avvertimento, perché «punto d'incontro» non voglia dire più di quel che
deve: è lo stesso *tipo* di oggetto, non lo stesso oggetto. Chi riconosce e
chi sintetizza scelgono finestre e passi diversi (venticinque millesimi di
secondo e una finestrella ogni dieci da una parte, cinquanta e una ogni dodici
e mezzo dall'altra), perché servono a due mestieri diversi: il
mel-spettrogramma di Tacotron 2 non si dà in pasto all'encoder di Whisper.

```{figure} ../figures/tts-pipeline.svg
:name: fig-tts-pipeline
:alt: "Diagramma di flusso a sei stadi della sintesi vocale: il testo «Sono le 9:30» entra nel blocco testo, passa per la normalizzazione che lo scioglie in «nove e trenta», per la conversione in fonemi, per il modello acustico che produce il mel-spettrogramma e per il vocoder, e ne esce un'onda sonora pronunciata."
:width: 95%

La pipeline della sintesi vocale, speculare a quella dell'ASR: il testo viene
normalizzato, convertito in fonemi, trasformato in mel-spettrogramma dal
modello acustico e infine in onda sonora dal vocoder.
```

Attenzione a un nome che torna e cambia mestiere. Nella pipeline del
riconoscimento il **modello acustico** era il pezzo che, ascoltando il suono,
diceva quali suoni-parola ci fossero dentro; qui è il pezzo che, letto il
testo, decide quali suoni *produrre*. Stesso nome, freccia rovesciata: è il
punto in cui la specularità della figura si vede meglio.

Perché tante tappe? Un secondo di parlato contiene decine di migliaia di
campioni audio, ma solo un'ottantina di colonne di mel-spettrogramma (una ogni
dodici millesimi e mezzo di secondo, nella ricetta di Tacotron 2 che vedremo
fra poco): fra le duecento e le trecento volte di meno. È molto
più facile per un modello decidere prima *che suoni* produrre e *con che
ritmo* (poche colonne compatte), e lasciare a uno specialista (il *vocoder*)
il compito di aggiungere il dettaglio fine dell'onda. Ma prima ancora dei
suoni c'è un lavoro sporco di cui nessuno parla volentieri: sistemare il
testo.

## Mettere ordine nel testo: la normalizzazione

Il testo scritto è pieno di simboli che nessuno pronuncia così come sono:
numeri, date, sigle, abbreviazioni. Prima di qualunque rete neurale, un
sistema TTS deve decidere *quali parole* corrispondono a ciò che è scritto.

`````{tab} Elementare

Immagina di dettare un telegramma al telefono: non puoi dire «1901», devi
scioglierlo a voce («millenovecentouno»). La normalizzazione è esattamente
questo lavoro, fatto da un programma. «Il dott. Rossi arriva l'1/3 alle 9:30»
deve diventare «il dottor Rossi arriva il primo marzo alle nove e trenta»:
«dott.» si scioglie in «dottore» (anzi, «dottor» davanti al nome), «1/3» è una
data e non una frazione, «9:30» è un orario. Sembra banale finché non arrivano
le ambiguità: «1901» si legge «millenovecentouno» se è un anno, ma «uno nove
zero uno» se è un interno telefonico. E c'è di peggio: «ancora» si pronuncia
*àncora* se è l'attrezzo della nave e *ancóra* se vuol dire «di nuovo»; sulla
pagina sono identiche, e per scegliere bisogna capire che mestiere fa la
parola nella frase. È lo stesso problema di analisi grammaticale che abbiamo
incontrato nel capitolo sul linguaggio naturale, nella sezione
sull'etichettatura delle parole.

`````

`````{tab} Superiore

Formalmente la normalizzazione è una trasduzione testo→testo: mappare le
**classi semiotiche** (cardinali, ordinali, date, orari, valute, unità di
misura, sigle) nella loro forma pronunciabile. Storicamente si fa con regole e
trasduttori a stati finiti pesati (WFST), che restano lo standard nei sistemi
di produzione; gli approcci seq2seq neurali funzionano ma sbagliano in un modo
particolarmente insidioso: leggere «1901» come *millenovecentodieci* è un
errore silenzioso, plausibile all'ascolto, che chi sente non ha modo di
correggere perché non si accorge che c'è. Il secondo fronte sono gli **omografi
eterofoni**: *àncora/ancóra*, *sùbito/subìto*, *lèggere/leggère*. La grafia non
basta: serve la categoria grammaticale, cioè il POS tagging visto nel capitolo
NLP, o un contesto più ampio. La normalizzazione è la parte meno glamour della
pipeline e una delle più costose da fare bene: è qui, non nella rete, che un
sistema commerciale si gioca le figuracce.
`````

## Dal grafema al fonema

Finora il capitolo ha parlato di «suoni» senza mai dare loro un nome preciso.
Per la sintesi serve farlo, perché le lettere (i *grafemi*) non corrispondono
ai suoni una a una.

`````{tab} Elementare

Un **fonema** è il più piccolo suono che, cambiando, cambia il significato:
«pane» e «cane» differiscono per un solo suono iniziale, e infatti sono due
parole diverse. Le lettere non bastano a rappresentarli: la «c» di *casa* e la
«c» di *ciao* sono la stessa lettera ma due suoni completamente diversi. Per
scriverli senza ambiguità i linguisti usano l'**alfabeto fonetico
internazionale** (IPA), dove ogni simbolo è un suono e uno solo, e si scrive
fra due barrette: /k/ per la c di *casa*, /tʃ/ per la c di *ciao*. Il
passaggio automatico dalle lettere ai suoni ha una sigla che compare anche
nella figura di poco fa: **G2P**, dall'inglese *grapheme-to-phoneme*, cioè
«dal grafema al fonema», che è il titolo di questa sezione. L'italiano è quasi
«trasparente» (quasi
sempre si legge come si scrive, con poche regole) ma l'inglese no: *though*,
*tough* e *through* finiscono tutte in *-ough* e si pronunciano in tre modi
che non si somigliano affatto. E i fonemi giusti non bastano ancora: prova a
leggere ad alta voce l'elenco della spesa e poi una battuta di teatro. Le
parole sono chiare in entrambi i casi, ma cambiano la melodia della frase, le
pause, la durata delle sillabe. Questa musica si chiama **prosodia**, ed è la
differenza tra leggere e recitare: la parte più difficile da insegnare a una
macchina.

`````

`````{tab} Superiore

Il fonema è l'unità sonora minima con valore distintivo: due suoni sono fonemi
diversi se esiste una **coppia minima** che li oppone (/p/ e /k/ in
*pane*/*cane*). Le varianti che non cambiano mai il significato sono
*allofoni* dello stesso fonema: la nasale di «un cane» e quella di «un pane»
(l'esempio di coarticolazione della panoramica del capitolo) suonano diverse,
e non sono nemmeno intercambiabili (le sceglie il contesto: velare davanti a
velare, bilabiale davanti a bilabiale), ma scambiarle non cambierebbe la
parola, solo la pronuncia. La conversione **grafema→fonema**
(G2P, *grapheme-to-phoneme*) è quasi deterministica per le ortografie
trasparenti come l'italiano (poche regole contestuali: ⟨c⟩ → /tʃ/ davanti a
⟨e, i⟩, /k/ altrove; restano imprevedibili l'apertura delle vocali medie e la
posizione dell'accento, che l'ortografia non marca), mentre per l'inglese è un
problema di apprendimento a tutti gli effetti, spesso risolto con un
dizionario di pronuncia più un modello seq2seq per le parole fuori lista.
Sopra i fonemi c'è la **prosodia**: il contorno della frequenza fondamentale
$F_0$ (l'intonazione), le durate dei foni, le pause. Nei sistemi end-to-end
moderni la prosodia non è annotata: il modello la assorbe implicitamente dai
dati, ed è lì che si gioca la naturalezza.

`````

## Da robot a persona: tre generazioni di sintesi

La prima generazione automatizza proprio il Voder: al posto delle dita di Helen
Harper ci sono regole scritte a mano, una per ogni suono da produrre. Si chiama
**sintesi per formanti**, dal nome della cosa che imita: quando parliamo, la
gola e la bocca cambiano forma e fanno risuonare più forte certe frequenze e
non altre, ed è da quelle poche frequenze esaltate (le *formanti*) che
l'orecchio distingue una «a» da una «i». La macchina non registra nessuna voce:
fabbrica quelle risonanze con generatori di suono e filtri, seguendo la
ricetta. Si capisce benissimo, non si stanca mai, e non somiglia a nessuno: è
la voce robotica per definizione. Il suo capolavoro è il **DECtalk** (1984),
figlio del lavoro di Dennis Klatt al MIT, la cui voce principale (*Perfect
Paul*, modellata sulla voce dello stesso Klatt) è entrata nella storia per un
caso particolare: Stephen Hawking. Dalla metà degli anni Ottanta Hawking parlò
attraverso un sintetizzatore a formanti della stessa famiglia (il CallText 5010
di Speech Plus, anch'esso derivato dal lavoro di Klatt), e quando negli anni
arrivarono voci molto migliori rifiutò sempre di cambiarla: «La tengo perché
non ho ancora sentito una voce che mi piaccia di più, e perché ormai mi ci
identifico». Quel timbro metallico era diventato *la sua* voce: al punto che un
team di ingegneri lavorò anni per emularne l'hardware ormai
introvabile, e Hawking approvò il risultato nel gennaio 2018, due mesi prima di
morire. Klatt, l'uomo che gli aveva prestato la voce, era morto
trent'anni prima, dopo che un tumore alla tiroide gli aveva tolto la propria.

La seconda generazione, dominante dagli anni Novanta, è la **sintesi
concatenativa**: si registrano ore di parlato di uno speaker professionista,
si tagliano in frammenti e si ricuciono i pezzi giusti per comporre la frase
richiesta (*unit selection*: tra i tanti ritagli disponibili si sceglie la
sequenza che si incolla meglio). Sulle frasi facili suona naturale (è vera
voce umana) ma le giunture si sentono, e il sistema è rigido: per cambiare
stile, o anche solo correggere un'intonazione, bisogna tornare in studio di
registrazione.

La terza generazione è quella neurale, e comincia nel 2016, quando WaveNet
{cite}`oord2016wavenet` dimostra che una rete può generare direttamente
l'onda audio, campione per campione, con una qualità mai sentita prima. Nel
giro di due anni la ricetta si assesta nella forma a due stadi della nostra
pipeline.

## Il TTS neurale, in due stadi

### Primo stadio: dal testo al mel-spettrogramma

Il modello di riferimento è **Tacotron 2** {cite}`shen2018natural`, di
Google. Se lo schema ti suona familiare, è perché lo è: è un
encoder-decoder con attenzione, lo stesso della traduzione automatica.

`````{tab} Elementare

Ricordi il traduttore del capitolo sul linguaggio naturale? Una parte della
rete legge tutta la frase, l'altra scrive la traduzione parola per parola,
con l'«evidenziatore» dell'attenzione che si sposta sul punto giusto del
testo di partenza. Tacotron 2 è la stessa macchina, con una differenza: non
scrive parole, scrive *colonne di mel-spettrogramma*, un fotogramma sonoro
alla volta, e l'attenzione scorre sul testo come il dito di chi impara a
leggere segue il rigo. Se una vocale va tenuta a lungo, l'attenzione resta
ferma lì per più colonne. Il risultato è quasi indistinguibile da una
registrazione, ma il metodo ha il difetto di chi scrive una lettera alla
volta: ogni tanto il dito scivola, e il modello salta una parola o balbetta
una sillaba due volte. **FastSpeech 2** {cite}`ren2021fastspeech` rovescia il
metodo: prima decide *quanto dura* ogni suono, poi riempie tutte le colonne
in un colpo solo, in parallelo. Più veloce di ordini di grandezza (cioè
decine o centinaia di volte, non del venti per cento), e niente
balbuzie: al prezzo di una piccola perdita di espressività, è la scelta
tipica quando la voce deve rispondere all'istante.

`````

`````{tab} Superiore

Tacotron 2 è un seq2seq autoregressivo: un encoder (convoluzioni + BiLSTM)
trasforma la sequenza di caratteri o fonemi $c = (c_1, \dots, c_n)$ negli
stati $h_1, \dots, h_n$; un decoder genera il mel-spettrogramma
$M = (m_1, \dots, m_T)$, con $m_t \in \mathbb{R}^{80}$ (80 bande mel), una
colonna alla volta:

$$
m_t = f_\theta(m_1, \dots, m_{t-1}, c),
$$

dove il testo in ingresso si chiama $c$ e non $x$ perché in questo capitolo
$x$ è già il vettore acustico di un frame: qui la freccia va nell'altro verso,
e all'ingresso c'è il testo. È una funzione autoregressiva ma
**deterministica**: niente densità da
massimizzare, la loss è l'errore quadratico sui frame mel, più un predittore
di stop che decide quando la frase è finita. Ogni passo è condizionato da un
vettore di contesto calcolato con l'attenzione di Bahdanau vista nella
traduzione (in variante *location-sensitive*, che tiene traccia di dove ha
già guardato per favorire un avanzamento monotono sul testo). Nei test
d'ascolto degli autori il sistema completo raggiunge un voto medio che
l'intervallo di confidenza non separa da quello del parlato registrato da
professionisti; nel confronto diretto sulla stessa frase, però, gli ascoltatori
preferiscono ancora la registrazione vera, in modo statisticamente
significativo. Le due misure non si contraddicono: dicono che la media dei voti
è uno strumento più grossolano del confronto appaiato, ed è una lezione che
torna nella sezione sul MOS.

FastSpeech 2 elimina l'autoregressione
con un *variance adaptor* a tre rami (durate, pitch, energia): il predittore di
**durate** stima quanti frame occupa ciascun fonema e il *length regulator*
replica gli stati dell'encoder di conseguenza ($T = \sum_i d_i$, dove $d_i$ è
il numero di frame assegnati al fonema $i$-esimo), mentre gli altri due rami
stimano pitch ed energia per frame; il mel si genera poi in parallelo. Niente
attenzione da far convergere: spariscono salti e ripetizioni, e la velocità
cresce di ordini di grandezza.

C'è però un passaggio che la frase «niente attenzione da far convergere»
nasconde, e vale la pena scoprirlo. Il predittore di durate è addestrato in
modo supervisionato, quindi gli servono le durate vere di ogni fonema; ma un
corpus TTS è fatto di coppie testo-audio, non di segmentazioni fonema per
fonema, ed è la stessa impraticabilità che si è vista dal lato del
riconoscimento. Le etichette gliele fornisce un **allineatore forzato**
esterno (nel paper il Montreal Forced Aligner, costruito sopra un sistema
HMM della generazione precedente), applicato ai dati prima
dell'addestramento. L'allineamento, insomma, non sparisce: esce dal modello e
diventa un passo di preparazione dei dati, comprato da un impianto che il
modello dichiara superato. Gli stessi autori indicano il farne a meno come
lavoro futuro.

`````

### Secondo stadio: il vocoder

Il nome chiude un cerchio: *vocoder*, da *voice coder*, è il termine nato ai
Bell Labs proprio per il sistema di analisi e sintesi della voce di Homer
Dudley, di cui il Voder era la vetrina da esposizione. Oggi indica il modello
che trasforma il mel-spettrogramma nell'onda vera e propria.

`````{tab} Elementare

Il mel-spettrogramma è il progetto della casa; il vocoder è l'impresa che la
costruisce mattone su mattone. **WaveNet** (l'abbiamo già incontrata nel
capitolo sull'Audio, quando generava musica) lavora come un amanuense: scrive
l'onda un campione alla volta (e i campioni sono più di ventimila al secondo)
decidendo ognuno sulla base di tutti i precedenti. Qualità mai sentita prima,
ma una lentezza proverbiale: nella versione originale, generare un secondo di
audio poteva costare minuti di calcolo. **HiFi-GAN** risolve il problema con
una vecchia conoscenza: il falsario e l'esperto d'arte del capitolo sulle GAN.
Una rete-falsario impara a produrre l'onda intera in un colpo solo, una
squadra di reti-esperto prova a distinguere l'audio vero da quello fabbricato,
e i due si allenano a vicenda finché il falso diventa indistinguibile.
Risultato: qualità paragonabile a WaveNet, ma molto **più veloce del tempo
reale**, che vuol dire questo: per fabbricare un secondo di parlato ci mette
molto meno di un secondo, tanto che in un secondo di calcolo ne produce minuti.
Il calcolo lo fa su una scheda grafica, che nel libro abbiamo già visto essere
l'attrezzo con cui si fanno i conti delle reti: qui non serve a disegnare
niente, serve a fare migliaia di moltiplicazioni insieme.

`````

`````{tab} Superiore

WaveNet {cite}`oord2016wavenet` è la stessa rete della sezione *Generare suono
e musica* del capitolo sull'Audio; qui compare condizionata sul mel e con i
campioni chiamati $a$, perché $x$ in questa sezione è già il testo in ingresso
a Tacotron 2. Modella l'onda in modo autoregressivo:

$$
p(a \mid M) = \prod_{t=1}^{T'} p(a_t \mid a_1, \dots, a_{t-1}, M),
$$

dove $a_t$ è il campione audio al passo $t$ (quantizzato su 256 livelli con
compansione $\mu$-law nella versione originale), $T'$ è il numero di campioni
dell'onda (da non confondere con il $T$ delle colonne di mel: qui i campioni
sono centinaia di volte più numerosi) e $M$ è il mel-spettrogramma
che condiziona la generazione. L'architettura usa convoluzioni causali
**dilatate**, con dilatazione che raddoppia a ogni strato: il campo recettivo
cresce esponenzialmente e copre centinaia di millisecondi di contesto. Il
limite è strutturale: $T'$ passi sequenziali, cioè oltre ventimila per un
secondo di audio a 22.050 Hz. HiFi-GAN {cite}`kong2020hifi` sostituisce
l'autoregressione con un gioco avversario nel senso esatto del capitolo sulle
GAN: il generatore è una pila di convoluzioni trasposte che sovracampiona il
mel fino alla frequenza dell'onda; i discriminatori sono due famiglie:
*multi-period*, che riorganizzano l'onda per periodi diversi per coglierne le
periodicità, e *multi-scale*, che la ascoltano a risoluzioni diverse. Alla
loss avversaria si sommano una *feature matching loss* e una loss L1 tra i
mel-spettrogrammi dell'audio vero e di quello generato, che stabilizzano
l'addestramento. La differenza di costo non è quantitativa ma strutturale:
WaveNet paga $T'$ passi sequenziali, HiFi-GAN uno solo, e il divario misurato
dagli autori è di due ordini di grandezza sopra il tempo reale su una singola
GPU, con naturalezza percepita alla pari dei vocoder autoregressivi.

`````

## Quanto è naturale? L'orecchio come giudice

Per l'ASR avevamo il WER: si confronta la trascrizione con il riferimento e
si contano gli errori. Per la sintesi un «riferimento» non esiste: la stessa
frase si può pronunciare bene in mille modi diversi.

`````{tab} Elementare

Come si giudica un doppiatore? Lo si ascolta. Il **MOS** (*mean opinion
score*, punteggio medio di opinione) è esattamente questo: si fa ascoltare la
stessa frase a un gruppo di persone e si chiede un voto da 1 («pessima») a 5
(«eccellente»); il MOS è la media. Se trenta ascoltatori danno in media 4,2, il
sistema è vicino, ma non pari, a una registrazione umana, che nella stessa
prova prende di solito tra 4,5 e 4,6.

Un avvertimento che vale più del numero: quel voto medio ha senso *dentro* una
prova, non fra prove diverse. Trenta persone in una stanza con le cuffie
giudicano in un modo, trenta persone a casa propria in un altro, e lo stesso
sistema può prendere 4,5 in uno studio e 3,7 in un altro senza essere
cambiato di una virgola. Confrontare il MOS di un articolo con quello di un
altro non dice niente. L'alternativa più affidabile è il **test A/B**: due
versioni della stessa frase, «quale preferisci?». Chiedere quale delle due si
preferisce è una domanda più facile, e più fine, che chiedere un voto in
assoluto. Non esiste comunque una formula che sostituisca le
orecchie: la naturalezza è un giudizio umano, e due onde matematicamente
diversissime possono suonare entrambe perfette.

`````

`````{tab} Superiore

Il MOS è la media dei giudizi su scala 1–5,
$\mathrm{MOS} = \frac{1}{N}\sum_{i=1}^{N} s_i$, dove $s_i$ è il voto
dell'ascoltatore $i$-esimo; si riporta con l'intervallo di confidenza al 95% e
richiede un protocollo rigoroso (frasi fuori dal training, ascoltatori
madrelingua, cuffie, campioni mescolati con parlato reale come ancoraggio). Le
alternative sono i test di preferenza A/B e ABX. Due debolezze strutturali: il
MOS è relativo al gruppo di ascoltatori e alle condizioni della prova, quindi
i numeri di studi diversi non sono confrontabili (Tacotron 2 vale 4,53 nel
paper che lo presenta e 3,70 in quello di FastSpeech 2, a tre anni di
distanza e senza che il sistema sia cambiato: ottantatré centesimi di scarto
sono la misura di quanto conti il protocollo), e nemmeno due numeri della
stessa tabella si leggono per differenza se i loro intervalli di confidenza si
sovrappongono, come succede fra Tacotron 2 e il parlato registrato; e le
metriche oggettive
(come la distorsione mel-cepstrale dal riferimento) correlano male con la
qualità percepita, perché il TTS è un problema *uno-a-molti* e penalizzano
pronunce legittime solo perché diverse. A oggi, la valutazione seria di un
sistema TTS passa ancora dall'orecchio umano.

`````

## La voce di chi? Cloni, truffe e consenso

C'è un rovescio della medaglia, ed è bene guardarlo senza allarmismi ma senza
sconti. Gli stessi modelli di questa sezione, addestrati sulla voce di una
persona specifica (oggi bastano pochi minuti di registrazione, e i sistemi più
recenti si accontentano di secondi) producono un **clone vocale**. Il fenomeno
delle truffe è documentato: già nel 2019 il *Wall Street Journal* riportò il
caso di un dirigente di un'azienda energetica britannica convinto al telefono,
da una voce clonata che imitava l'amministratore delegato della casa madre
tedesca, a trasferire 220.000 euro a un presunto fornitore ungherese. Da
allora lo schema si è ripetuto, fino alle telefonate-trappola che imitano la
voce di un familiare. La questione di fondo è il **consenso**: la voce è un
dato biometrico e un pezzo di identità (Hawking, che rifiutò per trent'anni
voci «migliori» della sua, lo sapeva bene) e clonarla senza permesso è una
forma di furto. Il *watermarking* (una filigrana inudibile incorporata
nell'onda sintetica per riconoscerla dopo, come il filo di sicurezza in una
banconota) è la contromisura tecnica più promettente, ma insegue. Finché
insegue, la difesa che funziona non è tecnica ed è vecchia come il telefono:
se una voce chiede soldi o dati con urgenza, si riattacca e si richiama il
numero che si conosce; e una parola d'ordine concordata in famiglia costa
niente e non si clona. Vale anche il rovescio del rovescio: la stessa
tecnologia restituisce la voce a chi la sta perdendo per una malattia
neurodegenerativa, registrandola prima che scompaia. Come sempre, lo strumento
non sceglie l'uso.

## In pratica: sintetizzare una frase

`torchaudio` offre pipeline preaddestrate che impacchettano i due stadi. Qui
usiamo Tacotron 2 con un vocoder WaveRNN (un parente autoregressivo
alleggerito di WaveNet) per pronunciare la traduzione del nostro esempio
ricorrente:

```{code-block} python
:class: pt-lento

import torch
import torchaudio
import soundfile as sf

# bundle preaddestrato: Tacotron 2 (da caratteri) + vocoder WaveRNN,
# voce inglese femminile (dataset LJSpeech)
bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH

processor = bundle.get_text_processor()   # testo -> ID dei caratteri
tacotron2 = bundle.get_tacotron2()        # caratteri -> mel-spettrogramma
vocoder = bundle.get_vocoder()            # mel-spettrogramma -> onda

testo = "The black cat jumps on the wall."

with torch.inference_mode():
    token, lunghezze = processor(testo)
    mel, mel_len, _ = tacotron2.infer(token, lunghezze)
    onda, onda_len = vocoder(mel, mel_len)

# salva il risultato: onda ha forma (batch, campioni), qui se ne prende il primo
sf.write("gatto.wav", onda[0].cpu().numpy(), vocoder.sample_rate)
print(onda.shape, vocoder.sample_rate)  # es. torch.Size([1, ...]) e 22050
```

Un *bundle*, qui, è la confezione già pronta: i due modelli e i numeri che
hanno imparato, scaricati insieme. I pesi di questo sono addestrati su una voce
inglese: dagli una frase italiana e la leggerà con un buffo accento anglofono.
Per l'italiano esistono modelli aperti addestrati su corpora nostrani, in
particolare della famiglia **VITS** {cite}`kim2021vits`, che invece di
incatenare due stadi li fonde in un unico modello addestrato da capo a
fondo.

Il cerchio del capitolo si chiude qui: sappiamo trasformare la voce in testo e
il testo in voce. Messe in fila (ASR, un modello che decide cosa rispondere,
TTS) sono lo scheletro di un assistente vocale: la catena che si mette in moto
quando chiedi «che ore sono?» al telefono, e una voce sintetica ti risponde.

Anche qui, il ripasso su due livelli.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- La sintesi vocale rifà la strada del riconoscimento **al contrario**: testo
  → si sciolgono numeri e sigle → si scrivono i suoni → si disegna
  l'immagine a bande del suono → si costruisce l'onda. Quell'immagine a bande
  (il **mel-spettrogramma**) è il punto in cui i due viaggi si toccano.
- Sciogliere il testo (la **normalizzazione**) è il lavoro meno appariscente e
  quello dove si sbaglia di più: «1901» è «millenovecentouno» o «uno nove zero
  uno» a seconda di cosa sia. Poi si passa dalle lettere ai suoni (il
  **G2P**): la c di *casa* e la c di *ciao* sono la stessa lettera e due suoni
  diversi. E sopra tutto c'è la **prosodia**, la musica della frase: pause,
  durate, intonazione.
- Tre generazioni di macchine parlanti: quella **per formanti** (fabbrica i
  suoni da zero seguendo regole, ed è la voce robotica di Hawking), quella
  **concatenativa** (ritagli di voce vera ricuciti insieme), quella
  **neurale**, di oggi.
- Oggi il lavoro è diviso in due: un modello scrive l'immagine del suono
  (**Tacotron 2** una colonna alla volta, **FastSpeech 2** tutte insieme e
  senza balbettare) e un secondo modello, il **vocoder**, la trasforma in
  onda (**WaveNet** un campione alla volta e lentissimo, **HiFi-GAN** tutto in
  un colpo e velocissimo).
- Se la qualità è buona lo decide l'orecchio: si fa votare la gente (**MOS**)
  o le si chiede quale delle due versioni preferisce (**test A/B**). Nessun
  conto automatico basta, perché la stessa frase si può dire bene in mille
  modi diversi.
- Clonare una voce è facile e serve già alle truffe. La voce è un pezzo di
  identità: il permesso di chi parla è il minimo, e al telefono la difesa che
  funziona è riattaccare e richiamare il numero vero.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Il TTS percorre la pipeline dell'ASR **al contrario**: testo →
  normalizzazione → fonemi → mel-spettrogramma → onda. Il mel-spettrogramma
  è il punto d'incontro dei due viaggi, ma non è lo stesso tensore: finestre e
  passi sono diversi dalle due parti (10 ms nell'ASR, 12,5 ms in Tacotron 2).
- La **normalizzazione** scioglie numeri, date e sigle («1901» →
  «millenovecentouno») e disambigua gli omografi (*àncora/ancóra*) col
  contesto; il **G2P** converte i grafemi in **fonemi** (la c di *casa* è
  /k/, quella di *ciao* è /tʃ/): facile in italiano, difficile in inglese.
  Sopra tutto c'è la **prosodia**: intonazione, durate, pause.
- Tre generazioni: sintesi **per formanti** (robotica: DECtalk, la voce di
  Hawking), **concatenativa** (ritagli di voce vera ricuciti), **neurale**.
- Il TTS neurale lavora in **due stadi**: un modello acustico testo→mel
  (**Tacotron 2**, seq2seq con attenzione; **FastSpeech 2**, parallelo e più
  stabile, ma con le durate fornite da un allineatore forzato esterno) e un
  **vocoder** mel→onda (**WaveNet**, autoregressivo e lento; **HiFi-GAN**,
  avversario e di ordini di grandezza più veloce).
- La qualità si misura con l'orecchio: **MOS** e test A/B; nessuna metrica
  automatica è pienamente affidabile, perché la sintesi è un problema
  uno-a-molti. Il MOS vale dentro una prova, non fra prove diverse.
- La **clonazione vocale** è già usata nelle truffe: la voce è un dato
  biometrico, e consenso e watermarking sono il minimo sindacale.
```

`````
