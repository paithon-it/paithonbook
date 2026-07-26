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
suono alle feature*: l'immagine tempo–frequenza del suono, su scala
percettiva. L'ASR parte da lì per salire verso le parole; il TTS ci arriva
scendendo dalle parole, e da lì ricostruisce l'onda
({numref}`fig-tts-pipeline`).

```{figure} ../figures/tts-pipeline.svg
:name: fig-tts-pipeline
:alt: "Diagramma di flusso a sei stadi della sintesi vocale: il testo «Sono le 9:30» entra nel blocco testo, passa per la normalizzazione che lo scioglie in «nove e trenta», per la conversione in fonemi, per il modello acustico che produce il mel-spettrogramma e per il vocoder, e ne esce un'onda sonora pronunciata."
:width: 95%

La pipeline della sintesi vocale, speculare a quella dell'ASR: il testo viene
normalizzato, convertito in fonemi, trasformato in mel-spettrogramma dal
modello acustico e infine in onda sonora dal vocoder.
```

Perché tante tappe? Un secondo di parlato contiene decine di migliaia di
campioni audio, ma solo un'ottantina di colonne di mel-spettrogramma: è molto
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
di produzione; gli approcci seq2seq neurali funzionano ma soffrono di errori
«inaccettabili» in un modo particolare: leggere «1901» come
*millenovecentodieci* è un errore silenzioso, plausibile all'ascolto e quindi
più insidioso di una pronuncia goffa. Il secondo fronte sono gli **omografi
eterofoni**: *àncora/ancóra*, *sùbito/subìto*, *lèggere/leggère*. La grafia
non basta: serve la categoria grammaticale, cioè il POS tagging visto nel
capitolo NLP, o un contesto più ampio. La normalizzazione è la parte meno
glamour della pipeline e una delle più costose da fare bene: è qui, non nella
rete, che un sistema commerciale si gioca le figuracce.

`````

## Dal grafema al fonema

Finora il capitolo ha parlato di «suoni» e di formanti senza mai dare loro un
nome preciso. Per la sintesi serve farlo, perché le lettere (i *grafemi*), non
corrispondono ai suoni una a una.

`````{tab} Elementare

Un **fonema** è il più piccolo suono che, cambiando, cambia il significato:
«pane» e «cane» differiscono per un solo suono iniziale, e infatti sono due
parole diverse. Le lettere non bastano a rappresentarli: la «c» di *casa* e la
«c» di *ciao* sono la stessa lettera ma due suoni completamente diversi. Per
scriverli senza ambiguità i linguisti usano l'**alfabeto fonetico
internazionale** (IPA), dove ogni simbolo è un suono e uno solo: /k/ per la c
di *casa*, /tʃ/ per la c di *ciao*. L'italiano è quasi «trasparente» (quasi
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
(l'esempio di coarticolazione della panoramica del capitolo) suonano diverse
ma sono intercambiabili senza conseguenze. La conversione **grafema→fonema**
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

La prima generazione automatizza proprio il Voder: è la **sintesi per
formanti**, dove regole scritte a mano pilotano oscillatori e filtri per
riprodurre le risonanze del tratto vocale. Intelligibile, instancabile, e
inconfondibilmente robotica. Il suo capolavoro è il **DECtalk** (1984), figlio
del lavoro di Dennis Klatt al MIT, la cui voce principale (*Perfect Paul*,
modellata sulla voce dello stesso Klatt) è entrata nella storia per un caso
particolare: Stephen Hawking. Dalla metà degli anni Ottanta Hawking parlò
attraverso un sintetizzatore a formanti della stessa famiglia (il CallText
5010 di Speech Plus, anch'esso derivato dal lavoro di Klatt), e quando negli
anni arrivarono voci molto migliori rifiutò sempre di cambiarla: «La tengo
perché non ho ancora sentito una voce che mi piaccia di più, e perché ormai mi
ci identifico». Quel timbro metallico era diventato *la sua* voce: al punto
che un team di ingegneri lavorò anni per emularne l'hardware ormai
introvabile, e Hawking approvò il risultato nel gennaio 2018, poche settimane
prima di morire. Klatt, l'uomo che gli aveva prestato la voce, era morto
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
Google. Se lo schema vi suona familiare, è perché lo è: è un
encoder–decoder con attenzione, lo stesso della traduzione automatica.

`````{tab} Elementare

Ricordate il traduttore del capitolo sul linguaggio naturale? Una parte della
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
in un colpo solo, in parallelo. Più veloce di ordini di grandezza, e niente
balbuzie: al prezzo di una piccola perdita di espressività, è la scelta
tipica quando la voce deve rispondere all'istante.

`````

`````{tab} Superiore

Tacotron 2 è un seq2seq autoregressivo: un encoder (convoluzioni + BiLSTM)
trasforma la sequenza di caratteri o fonemi $x = (x_1, \dots, x_n)$ negli
stati $h_1, \dots, h_n$; un decoder genera il mel-spettrogramma
$M = (m_1, \dots, m_T)$, con $m_t \in \mathbb{R}^{80}$ (80 bande mel), una
colonna alla volta:

$$
p(M \mid x) = \prod_{t=1}^{T} p(m_t \mid m_1, \dots, m_{t-1}, x),
$$

dove ogni fattore è condizionato da un vettore di contesto calcolato con
l'attenzione di Bahdanau vista nella traduzione (in variante
*location-sensitive*, che tiene traccia di dove ha già guardato per favorire
un avanzamento monotono sul testo). La loss è l'errore quadratico sui frame
mel, più un predittore di stop che decide quando la frase è finita. Nei test
d'ascolto il sistema completo raggiunge un MOS di 4,53 contro il 4,58 del
parlato registrato da professionisti. FastSpeech 2 elimina l'autoregressione
con un *variance adaptor* a tre rami: un predittore di **durate** stima quanti
frame occupa ciascun fonema, un *length regulator* replica gli stati
dell'encoder di conseguenza ($T = \sum_i d_i$, dove $d_i$ è il numero di frame
assegnati al fonema $i$-esimo), e due predittori analoghi stimano pitch ed
energia per frame; il mel si genera poi in parallelo. Niente attenzione da far
convergere: spariscono salti e ripetizioni, e la velocità cresce di ordini di
grandezza.

`````

### Secondo stadio: il vocoder

Il nome chiude un cerchio: *vocoder*, da *voice coder*, è il termine nato ai
Bell Labs proprio per il sistema di analisi e sintesi della voce di Homer
Dudley, di cui il Voder era la vetrina da esposizione. Oggi indica il modello
che trasforma il mel-spettrogramma nell'onda vera e propria.

`````{tab} Elementare

Il mel-spettrogramma è il progetto della casa; il vocoder è l'impresa che la
costruisce mattone su mattone. **WaveNet** lavora come un amanuense: scrive
l'onda un campione alla volta (e i campioni sono più di ventimila al secondo)
decidendo ognuno sulla base di tutti i precedenti. Qualità mai sentita prima,
ma una lentezza proverbiale: nella versione originale, generare un secondo di
audio poteva costare minuti di calcolo. **HiFi-GAN** risolve il problema con
una vecchia conoscenza: il falsario e l'esperto d'arte del capitolo sulle GAN.
Una rete-falsario impara a produrre l'onda intera in un colpo solo, una
squadra di reti-esperto prova a distinguere l'audio vero da quello fabbricato,
e i due si allenano a vicenda finché il falso diventa indistinguibile.
Risultato: qualità paragonabile a WaveNet, ma oltre centosessanta volte più
veloce del tempo reale su una sola scheda grafica.

`````

`````{tab} Superiore

WaveNet {cite}`oord2016wavenet` modella l'onda in modo autoregressivo:

$$
p(a \mid M) = \prod_{t=1}^{T'} p(a_t \mid a_1, \dots, a_{t-1}, M),
$$

dove $a_t$ è il campione audio al passo $t$ (quantizzato su 256 livelli con
compansione $\mu$-law nella versione originale) e $M$ è il mel-spettrogramma
che condiziona la generazione. L'architettura usa convoluzioni causali
**dilatate**, con dilatazione che raddoppia a ogni strato: il campo recettivo
cresce esponenzialmente e copre centinaia di millisecondi di contesto. Il
limite è strutturale: $T'$ passi sequenziali, cioè oltre ventimila per un
secondo di audio a 22.050 Hz. HiFi-GAN {cite}`kong2020hifi` sostituisce
l'autoregressione con un gioco avversario nel senso esatto del capitolo sulle
GAN: il generatore è una pila di convoluzioni trasposte che sovracampiona il
mel fino alla frequenza dell'onda; i discriminatori sono due famiglie;
*multi-period*, che riorganizzano l'onda per periodi diversi per coglierne le
periodicità, e *multi-scale*, che la ascoltano a risoluzioni diverse. Alla
loss avversaria si sommano una *feature matching loss* e una loss L1 tra i
mel-spettrogrammi dell'audio vero e di quello generato, che stabilizzano
l'addestramento. Il modello genera audio a 22.050 Hz circa 167,9 volte più
veloce del tempo reale su una GPU V100, con naturalezza percepita alla pari
dei vocoder autoregressivi.

`````

## Quanto è naturale? L'orecchio come giudice

Per l'ASR avevamo il WER: si confronta la trascrizione con il riferimento e
si contano gli errori. Per la sintesi un «riferimento» non esiste: la stessa
frase si può pronunciare bene in mille modi diversi.

`````{tab} Elementare

Come si giudica un doppiatore? Lo si ascolta. Il **MOS** (*mean opinion
score*, punteggio medio di opinione) è esattamente questo: si fa ascoltare la
stessa frase a un gruppo di persone e si chiede un voto da 1 («pessima») a 5
(«eccellente»); il MOS è la media. Se trenta ascoltatori danno in media 4,2,
il sistema è vicino (ma non pari), a una registrazione umana, che di solito
prende tra 4,5 e 4,6. L'alternativa è il **test A/B**: due versioni della
stessa frase, «quale preferisci?». Non esiste una formula che sostituisca le
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
i numeri di studi diversi non sono confrontabili; e le metriche oggettive
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
nell'onda sintetica per riconoscerla a posteriori) è la contromisura più
promettente, ma insegue. Vale anche il rovescio del rovescio: la stessa
tecnologia restituisce la voce a chi la sta perdendo per una malattia
neurodegenerativa, registrandola prima che scompaia. Come sempre, lo strumento
non sceglie l'uso.

## In pratica: sintetizzare una frase

`torchaudio` offre pipeline preaddestrate che impacchettano i due stadi. Qui
usiamo Tacotron 2 con un vocoder WaveRNN (un parente autoregressivo
alleggerito di WaveNet) per pronunciare la traduzione del nostro esempio
ricorrente:

```python
import torch
import torchaudio

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

# salva il risultato: onda ha forma (batch, campioni)
torchaudio.save("gatto.wav", onda[0:1].cpu(), sample_rate=vocoder.sample_rate)
print(onda.shape, vocoder.sample_rate)  # es. torch.Size([1, ...]) e 22050
```

I pesi di questo bundle sono addestrati su una voce inglese: dategli una frase
italiana e la leggerà con un buffo accento anglofono. Per l'italiano esistono
modelli open addestrati su corpora nostrani (le famiglie VITS/Piper, i modelli
multilingue di progetti come Coqui TTS), che seguono lo stesso schema a due
stadi (o lo fondono in un unico modello).

Il cerchio del capitolo si chiude qui: sappiamo trasformare la voce in testo e
il testo in voce. Messe in fila (ASR, un modello che decide cosa rispondere,
TTS) sono lo scheletro di un assistente vocale: la catena che si mette in moto
quando dite «che ore sono?» al telefono, e una voce sintetica vi risponde.

```{admonition} Da ricordare
:class: important
- Il TTS percorre la pipeline dell'ASR **al contrario**: testo →
  normalizzazione → fonemi → mel-spettrogramma → onda. Il mel-spettrogramma
  è il punto d'incontro dei due viaggi.
- La **normalizzazione** scioglie numeri, date e sigle («1901» →
  «millenovecentouno») e disambigua gli omografi (*àncora/ancóra*) col
  contesto; il **G2P** converte i grafemi in **fonemi** (la c di *casa* è
  /k/, quella di *ciao* è /tʃ/): facile in italiano, difficile in inglese.
  Sopra tutto c'è la **prosodia**: intonazione, durate, pause.
- Tre generazioni: sintesi **per formanti** (robotica: DECtalk, la voce di
  Hawking), **concatenativa** (ritagli di voce vera ricuciti), **neurale**.
- Il TTS neurale lavora in **due stadi**: un modello acustico testo→mel
  (**Tacotron 2**, seq2seq con attenzione; **FastSpeech 2**, parallelo e più
  stabile) e un **vocoder** mel→onda (**WaveNet**, autoregressivo e lento;
  **HiFi-GAN**, avversario e centinaia di volte più veloce).
- La qualità si misura con l'orecchio: **MOS** e test A/B; nessuna metrica
  automatica è pienamente affidabile, perché la sintesi è un problema
  uno-a-molti.
- La **clonazione vocale** è già usata nelle truffe: la voce è un dato
  biometrico, e consenso e watermarking sono il minimo sindacale.
```
