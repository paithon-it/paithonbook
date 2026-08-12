# I modelli di riconoscimento

Quando pronunci la parola «casa», il microfono del telefono non registra
quattro lettere: registra circa sedicimila numeri al secondo, un fiume di
campioni che descrivono come vibra l'aria. Il compito del riconoscimento
vocale automatico (*Automatic Speech Recognition*, ASR) è tradurre quel fiume
in una manciata di caratteri. Sembra un problema di traduzione come un altro,
ma nasconde una difficoltà tutta sua, che ha condizionato per decenni il modo
in cui si costruiscono questi modelli.

## Il problema dell'allineamento

Prima di dare in pasto l'audio a una rete, lo trasformiamo in uno
**spettrogramma**: tagliamo il segnale in finestrelle di circa 25
millisecondi, una nuova ogni 10 millisecondi (così si sovrappongono: ogni
finestrella sfuma ai bordi, dove il suono conta quasi zero, e senza
sovrapposizione quello che cade sul bordo andrebbe perso), e per ciascuna
misuriamo quanta energia c'è
a ogni frequenza. Ogni finestrella è un *frame*, e un secondo di parlato
diventa così un centinaio di frame. La trascrizione, invece, è lunga poche
decine di caratteri. Due sequenze di lunghezza molto diversa, e nessuno ci
dice quale frame corrisponde a quale lettera.

`````{tab} Elementare
Immagina di dover sottotitolare un video a orecchio, senza conoscere i tempi.
Chi parla lento, chi veloce; una vocale tenuta a lungo («caaasa») occupa molti
fotogrammi ma resta una sola lettera; tra una parola e l'altra ci sono pause e
respiri che non vanno scritti. Sai *cosa* è stato detto, ma non *quando*
comincia e finisce ogni suono. Questo è l'allineamento: appaiare i tanti
pezzetti di audio ai pochi caratteri del testo.
`````

`````{tab} Superiore
Abbiamo un input $X = (x_1, \dots, x_T)$ di $T$ frame e un target
$y = (y_1, \dots, y_U)$ di $U$ token (caratteri o sotto-parole): è la
trascrizione che nella panoramica chiamavamo $W$, vista qui come sequenza di
simboli. Con $T \gg U$, l'allineamento è **monotono** (l'audio scorre in
avanti come il testo) ma **sconosciuto**: non abbiamo etichette frame per
frame. Segmentare a
mano milioni di ore per dire «da qui a qui c'è una a» è impraticabile. Serve un
modello che impari l'allineamento *da solo*, dalla sola coppia
(audio, trascrizione).
`````

## CTC: imparare ad allineare da soli

La svolta arriva nel 2006 con la **Connectionist Temporal Classification** di
Alex Graves e colleghi. L'idea è aggiungere all'alfabeto un simbolo speciale,
il «vuoto» (*blank*, $\varnothing$), che significa «qui non produco nessun
carattere». Per **ogni** frame la rete non sceglie un simbolo secco: dà un voto
a ciascun simbolo dell'alfabeto, vuoto compreso, e i voti sommano a uno (in
gergo, emette una *distribuzione*). Nella figura qui sotto è disegnato, per
ogni frame, il simbolo che ha preso il voto più alto. Poi una regola di
collasso ripulisce la sequenza: prima unisce i caratteri uguali consecutivi,
poi elimina i vuoti ({numref}`fig-ctc-allineamento`).

```{figure} ../figures/ctc-collassa.svg
:name: fig-ctc-allineamento
:alt: Sette frame acustici, ciascuno etichettato con un simbolo, nell'ordine P A A L vuoto L A. Il primo passo unisce i simboli uguali consecutivi, quindi le due A diventano una; il secondo passo toglie il simbolo vuoto. Resta PALLA. In basso, la controprova: invertendo l'ordine dei due passi, tolto per primo il vuoto non separa più le due L, il passo di fusione le unisce, ed esce PALA invece di PALLA.
:width: 90%

Il meccanismo della CTC, un passo alla volta. La rete assegna un simbolo a ogni
frame (anche il simbolo «vuoto» ∅); poi si uniscono i ripetuti consecutivi, e
**solo dopo** si tolgono i vuoti. L'ordine non è un dettaglio: invertendolo la
doppia «L» si perde, ed è per impedirlo che il ∅ esiste. I frame disegnati sono
sette perché ci stiano: per una parola come «palla» ne servirebbero una
cinquantina, uno ogni dieci millesimi di secondo.
```

`````{tab} Elementare
Molti modi di etichettare i frame danno la stessa parola. Per «PALLA» va bene
`P A A L ∅ L A`, ma anche `P P A L ∅ L A`: entrambi, dopo aver unito i doppioni
e tolto i vuoti, diventano `PALLA`.

Ognuno di questi modi ha una sua probabilità, cioè quanto la rete ci crede: si
ottiene moltiplicando fra loro i voti che la rete ha dato ai sette simboli di
quella riga. Diciamo che il primo modo valga il 30% e il secondo il 20%.
Siccome tutti e due, ripuliti, danno «PALLA», la probabilità che la parola sia
«PALLA» è la loro somma: 50%, più di quanto valga ciascuno da solo. Ecco cosa
vuol dire «sommare gli allineamenti».

La CTC non sceglie dunque *un* allineamento giusto e non chiede alla rete di
indovinarlo: li considera tutti insieme, somma le probabilità di quelli che
danno la trascrizione corretta, e spinge la rete ad alzare quel totale. Come
lo alzi (spostando i voti su un modo o sull'altro) sono affari suoi. Nota il
trucco del vuoto: senza il ∅ in mezzo, le due «L» si fonderebbero in una sola.
`````

`````{tab} Superiore
La probabilità di una trascrizione $y$ è la somma su tutti i percorsi
frame-level $\pi$ che, collassati, la producono:

$$
p(y \mid X) = \sum_{\pi \,\in\, \mathcal{B}^{-1}(y)}
\prod_{t=1}^{T} p_t(\pi_t \mid X),
$$

dove $\pi = (\pi_1, \dots, \pi_T)$ è un allineamento a livello di frame,
$p_t(\pi_t \mid X)$ è la probabilità che la rete assegna al simbolo $\pi_t$ al
frame $t$, e $\mathcal{B}$ è la funzione di collasso. La somma ha un numero
esponenziale di termini, ma si calcola in tempo $O(T \cdot U)$ (lineare nella
lunghezza dell'audio, a trascrizione fissata) con l'algoritmo di
programmazione dinamica *forward-backward*, che lavora sul reticolo dei
$2U+1$ simboli della trascrizione estesa con i blank. Si addestra minimizzando
$\mathcal{L} = -\log p(y \mid X)$.

Due limiti strutturali, e sono conseguenze dirette della formula. Il primo: la
CTC emette esattamente un simbolo per frame, quindi $\mathcal{B}^{-1}(y)$ è
vuoto appena $U > T$, e per una parola con due lettere uguali di fila serve
almeno un frame in più per il vuoto che le separa (`PALLA` sta in sette frame,
non in cinque). È il motivo per cui il metodo serve all'ascolto e non alla
sintesi vocale, dove il testo in ingresso è più corto del suono in uscita
{cite}`graves2012sequence`. Il secondo, il più citato: nel prodotto non compare
nessun fattore della forma $p(y_u \mid y_{<u})$, cioè le predizioni ai vari
frame sono **condizionatamente indipendenti** dato $X$. Non è che la CTC
modelli «non bene» le dipendenze fra i caratteri in uscita: non ha il posto
dove metterle. Torneremo su questo punto parlando del modello di linguaggio,
perché è di lì che discende tutto il resto.
`````

## Dalla rete alla frase: la decodifica

Fin qui abbiamo detto come si **addestra** un modello CTC, non come gli si fa
scrivere una frase. Sono due cose diverse, e la differenza è più grossa di
quanto sembri: il passaggio dai voti della rete alla trascrizione si chiama
**decodifica**, ed è un capitolo a sé.

Il modo ovvio è prendere, frame per frame, il simbolo che ha ricevuto il voto
più alto, e poi collassare la sequenza. Si chiama decodifica del **percorso
migliore** (*best path*), costa niente, ed è quello che fa quasi tutto il
codice di esempio che si trova in giro. Solo che risponde alla domanda
sbagliata: cerca il percorso più probabile, non la trascrizione più probabile,
e le due non coincidono.

`````{tab} Elementare

Il caso più piccolo possibile: due frame soltanto, e un alfabeto di due
simboli, il vuoto ∅ e la lettera «A». Su ciascuno dei due frame la rete dà il
60% al vuoto e il 40% alla «A». Ci sono quattro percorsi, e per ognuno la
probabilità è il prodotto dei due voti:

| percorso | probabilità | dopo la ripulitura |
|---|---|---|
| ∅ ∅ | 0,6 × 0,6 = 36% | (niente) |
| ∅ A | 0,6 × 0,4 = 24% | A |
| A ∅ | 0,4 × 0,6 = 24% | A |
| A A | 0,4 × 0,4 = 16% | A (le due si fondono) |

Il percorso più probabile è `∅ ∅`, con il 36%, e non scrive niente. Ma la
lettera «A» esce da tre percorsi diversi, e insieme fanno 24 + 24 + 16 = 64%:
quasi il doppio. Prendendo il percorso migliore avremmo trascritto il
silenzio; sommando come fa la CTC, la risposta è «A».

Non è un caso costruito ad arte: succede ogni volta che una trascrizione è
sostenuta da tanti percorsi mediocri e un'altra da un percorso solo, molto
convinto. E succede più spesso di quanto si creda, perché di percorsi che
danno la stessa parola ce ne sono a milioni.

`````

`````{tab} Superiore

Il *best path* massimizza il singolo allineamento,

$$
\pi^{*} = \arg\max_{\pi} \prod_{t=1}^{T} p_t(\pi_t \mid X),
\qquad \hat{y} = \mathcal{B}(\pi^{*}),
$$

mentre l'obiettivo che il modello è stato addestrato a massimizzare è
$\arg\max_y p(y \mid X)$, cioè la **somma** su $\mathcal{B}^{-1}(y)$. Le due
quantità sono diverse perché $\mathcal{B}$ non è iniettiva: molti percorsi
cadono sulla stessa etichettatura, e la loro massa può battere il massimo
puntuale. Con l'esempio a $T = 2$ della scheda accanto,
$p(\varnothing\varnothing) = 0{,}36$ contro $p(\texttt{A}) = 0{,}64$.

Graves e colleghi lo scrivono già nel paper del 2006
{cite}`graves2006connectionist`, dedicandoci una sezione: il *best path* non
garantisce di trovare l'etichettatura più probabile, e per l'$\arg\max$ esatto
non si conosce un algoritmo trattabile in generale. Si approssima quindi con
una ricerca a fascio.

`````

La ricerca a fascio (**beam search**) l'abbiamo già incontrata nella
traduzione automatica, nel capitolo sul linguaggio naturale: invece di
decidere subito, si tengono aperte le $k$ ipotesi più promettenti e si va
avanti qualche passo prima di scegliere. L'idea è la stessa; una cosa però
cambia, ed è precisamente quella di cui sopra. Nella traduzione le ipotesi
**competono**: due strade diverse sono due frasi diverse, e alla fine ne resta
una. Nel CTC no: due percorsi che si ripuliscono nello stesso testo sono la
stessa ipotesi, e i loro punteggi vanno **sommati** invece di essere messi in
concorrenza. Una beam search che se ne dimentica scarta la trascrizione
giusta, esattamente come fa il percorso migliore.

È anche il punto in cui entrano in scena due cose che vedremo fra poco: un
modello di linguaggio, che a ogni passo della ricerca aggiunge il proprio
giudizio al punteggio, e la lista delle prime $n$ ipotesi, che la ricerca
produce come sottoprodotto e che si può riordinare a posteriori.

Il conto della scheda si rifà in dieci righe, ed è il modo più rapido di
convincersene (chi non programma può saltare il riquadro: fa esattamente i
conti della tabella qui sopra).

```python
import itertools

VUOTO = "∅"
# i voti della rete: due frame, due simboli
voti = [{VUOTO: 0.6, "A": 0.4},
        {VUOTO: 0.6, "A": 0.4}]

def collassa(percorso):
    """Prima unisce i simboli uguali consecutivi, poi toglie i vuoti."""
    uniti = [s for i, s in enumerate(percorso)
             if i == 0 or s != percorso[i - 1]]
    return "".join(s for s in uniti if s != VUOTO)

def probabilita(percorso):
    p = 1.0
    for t, s in enumerate(percorso):
        p *= voti[t][s]
    return p

# tutte le trascrizioni, con la somma dei percorsi che le producono
totali = {}
for percorso in itertools.product(VUOTO + "A", repeat=len(voti)):
    testo = collassa(percorso)
    totali[testo] = totali.get(testo, 0.0) + probabilita(percorso)

# il percorso migliore: il simbolo più votato a ogni frame
migliore = tuple(max(v, key=v.get) for v in voti)

print("percorso migliore:", "".join(migliore),
      "->", repr(collassa(migliore)), round(probabilita(migliore), 3))
for testo, p in sorted(totali.items(), key=lambda kv: -kv[1]):
    print(f"  p(y = {testo!r}) = {p:.2f}")

# la promessa del testo, resa eseguibile
assert totali["A"] > totali[collassa(migliore)]
```

L'uscita dice in tre righe la cosa che conta: il percorso migliore vale 0,36 e
non scrive niente, mentre `A` vale 0,64. Da qui in avanti, quando un sistema
CTC «trascrive», intendiamo sempre una ricerca di questo tipo, non l'argmax
frame per frame.

## Ascoltare e attendere: i modelli con attenzione

Un'alternativa evita del tutto il vuoto. I modelli **sequenza-a-sequenza con
attenzione** hanno un *encoder* che riassume tutto l'audio e un *decoder* che
genera i caratteri uno alla volta, ciascuno tenendo conto di quelli già
scritti (in gergo, in modo *autoregressivo*). A ogni passo il
decoder usa l'**attenzione** per decidere su quali frame concentrarsi: è un
allineamento «morbido», appreso, non deciso a priori. L'architettura di
riferimento è *Listen, Attend and Spell* {cite}`chan2016listen`, ed è da lì che
viene il titolo di questa sezione: «attendere» traduce l'inglese *attend*, che
non vuol dire aspettare ma «fare attenzione a».

`````{tab} Elementare
Pensa a un interprete: prima ascolta l'intera frase, poi la ridice parola per
parola. Mentre pronuncia ogni parola, la sua attenzione torna al punto giusto
di ciò che ha sentito. Il modello fa lo stesso: genera un carattere, si
«riguarda» la porzione di audio più rilevante, genera il prossimo.
`````

`````{tab} Superiore
Al passo $i$ il decoder costruisce un vettore di contesto come media pesata
degli stati dell'encoder $h_j$:

$$
c_i = \sum_{j=1}^{T_{\text{enc}}} \alpha_{ij}\,h_j,
\qquad \sum_j \alpha_{ij} = 1,
$$

dove i pesi di attenzione $\alpha_{ij}$ dicono quanto lo stato $j$ conta per
produrre il token $i$-esimo, e $T_{\text{enc}} \le T$ è il numero di stati
dell'encoder: l'encoder tipicamente sottocampiona l'asse temporale (in LAS di
un fattore 8), proprio per rendere trattabile l'attenzione su sequenze lunghe.
A differenza della CTC, il decoder condiziona ogni token su quelli già emessi:
modella cioè le dipendenze del testo in uscita, ed è per questo che un modello
del genere si porta dentro un modello di linguaggio senza che nessuno
gliel'abbia messo.

Il prezzo si paga altrove, e non è la generazione sequenziale più lenta. È che
**nulla vincola $\alpha_{ij}$ ad avanzare al crescere di $i$**: la CTC ha la
monotonia per costruzione, l'attenzione la deve imparare, e niente le impedisce
di saltare una porzione di audio, di riguardarne una già trascritta o di
restare ferma dov'è. Da lì vengono le parole mancanti, le sillabe ripetute e i
loop di ripetizione, che ritroveremo identici (stessa causa, altro compito)
nella sintesi vocale. C'è poi un secondo prezzo, che conta appena si esce dal
laboratorio: la somma corre su **tutti** gli stati dell'encoder, quindi il
primo token non può uscire prima che sia arrivato l'ultimo frame. Un modello
così non trascrive mentre ascolta.
`````

## Il trasduttore: tenersi tutte e due le cose

Messe una accanto all'altra, le due famiglie sembrano costringere a una
scelta. La CTC scorre l'audio in avanti e non torna mai indietro, quindi può
scrivere mentre ascolta, ma non sa niente di cosa ha già scritto. Il decoder
con attenzione sa benissimo cosa ha già scritto, ma per farlo deve aver
ascoltato tutto, e per giunta può perdere il segno. Nella pratica quella
scelta non esiste, perché esiste una terza famiglia che tiene le due cose
insieme: il **trasduttore neurale** (in sigla RNN-T), proposto da Alex Graves
nel 2012 {cite}`graves2012sequence`, cioè da chi aveva scritto la CTC sei anni
prima.

`````{tab} Elementare

Pensa a uno stenografo che scrive sotto dettatura. Non aspetta la fine del
discorso come l'interprete: scrive mentre ascolta, e non torna mai indietro sul
nastro. Però ha sotto gli occhi il foglio, cioè quello che ha appena scritto, e
quel foglio lo aiuta: se ha scritto «buon», la parola dopo sarà più
probabilmente «giorno» che «gnorno».

Il trasduttore è questo. A ogni istante ha due mosse possibili: scrivere un
carattere (e allora rilegge il foglio aggiornato, ma resta fermo sull'audio) o
passare al frame successivo senza scrivere niente. Alternando le due mosse
copre tutto l'audio e produce tutto il testo, senza mai tornare indietro e
senza mai dimenticare quello che ha già messo giù.

`````

`````{tab} Superiore

Il trasduttore accoppia tre reti: un *encoder* (o *transcription network*) che
produce $h_t$ dai frame acustici, una *prediction network* che produce $g_u$
dai soli token già emessi $y_{<u}$, e una piccola *joint network* che fonde le
due e proietta sul vocabolario esteso col vuoto,

$$
p(k \mid t, u) = \mathrm{softmax}\big(W\,\phi(h_t + g_u)\big).
$$

Lo spazio degli allineamenti non è più una sequenza di $T$ etichette ma un
reticolo $T \times U$: emettere un token muove di uno in verticale, emettere il
vuoto muove di uno in orizzontale, e ogni cammino monotono dall'angolo in basso
a sinistra a quello in alto a destra è un allineamento valido. La probabilità
della trascrizione è ancora la somma su tutti i cammini, calcolata con lo
stesso forward-backward della CTC, e la loss è ancora $-\log p(y \mid X)$.

Due conseguenze, ed è tutto il punto. La prediction network è a tutti gli
effetti un modello di linguaggio interno, condizionato sui token già emessi:
il trasduttore modella cioè le dipendenze uscita-uscita che la CTC non ha dove
mettere. E $h_t$ dipende solo dai frame fino a $t$ (con un encoder causale),
quindi la decodifica è frame-sincrona e non ha bisogno della fine dell'audio:
si trascrive mentre si ascolta.

`````

Non è un'architettura di nicchia: è quella su cui gira, dal 2019, la dettatura
in tempo reale sui telefoni {cite}`he2019streaming`, dove la risposta deve
arrivare mentre si parla e il modello deve stare dentro un dispositivo. Vale la
pena tenerlo a mente nella prossima sezione, quando parleremo di che cosa oggi
serva davvero la vecchia catena a stadi.

## Whisper e i Transformer end-to-end

Nel settembre 2022 OpenAI rilascia **Whisper**: un unico Transformer
encoder-decoder che riceve lo spettrogramma log-mel e produce direttamente il
testo.

```{figure} ../figures/come-funziona-whisper.svg
:name: fig-whisper
:alt: "Catena in cinque stadi: l'onda sonora in ingresso diventa uno spettrogramma, che entra in un encoder Transformer; l'uscita dell'encoder alimenta un decoder, che emette i token di testo uno dopo l'altro. Non ci sono moduli separati per il dizionario di pronuncia o per il modello di linguaggio."
:width: 96%

La catena di Whisper, tutta qui. Fra lo spettrogramma e il testo non c'è
nessuno stadio con regole scritte a mano: encoder e decoder sono addestrati
insieme, in un pezzo solo.
```

Quello che manca in {numref}`fig-whisper` conta quanto quello che c'è. La
pipeline classica aveva un modello acustico, un dizionario di pronuncia e un
modello di linguaggio, ciascuno costruito e messo a punto per una lingua;
qui gli stessi compiti restano, ma sono distribuiti nei pesi e appresi dai
dati, e questo è il motivo per cui un modello solo copre decine di lingue. La
sua forza non è tanto l'architettura quanto i dati: 680.000 ore di audio
raccolte dal web con etichettatura debole, cioè trascrizioni già esistenti in
rete, scritte da qualcuno per altri scopi e non per addestrare un modello.
«Debole» non vuol dire «non curata»: gli autori le ripuliscono con filtri
automatici (via quelle prodotte da altri riconoscitori, via le coppie in cui
la lingua parlata non è quella scritta, via i duplicati) e ispezionano a mano
le fonti che sbagliano di più, buttandole. Il materiale raccolto sfiora il
centinaio di lingue; quelle su cui il modello impara davvero a trascrivere
sono una settantina, e non sono servite allo stesso modo: l'inglese ha due
terzi di quelle ore, e da lì viene buona parte del divario di qualità che si
sente passando all'italiano. Con lo stesso modello Whisper
trascrive, traduce verso l'inglese e riconosce la lingua, guidato da istruzioni
speciali (in gergo *token* speciali, cioè simboli che non si pronunciano)
inserite nel decoder.

Quello che gli autori rivendicano non è che Whisper sbagli meno di tutti, ed è
una distinzione che vale la pena tenere. La grandezza che misurano è la
robustezza *zero-shot*: senza essere mai stato addestrato su un certo
benchmark, Whisper vi si comporta meglio di quanto la sua accuratezza altrove
lascerebbe prevedere, e degrada più lentamente man mano che si alza il rumore
di fondo (con poco rumore i modelli specializzati lo battono ancora). È una
misura di *quanto si peggiora fuori casa*, non di quanto si è bravi. E ha un
fianco scoperto, che il paper dichiara: i dati vengono dal web, dove stanno
anche i benchmark, e il controllo delle sovrapposizioni è stato fatto su un
solo dataset e sulle trascrizioni, non sull'audio.

Non è però Whisper ad aver mandato in pensione la catena a stadi: il passaggio
a una rete sola era cominciato anni prima, con la CTC e con i modelli ad
attenzione delle sezioni precedenti, e Whisper ne è la vetrina più
visibile. La vecchia catena, del resto, non è sparita: dove le parole
da riconoscere sono poche e note in anticipo (i comandi di un centralino
telefonico, i codici letti ad alta voce in un magazzino) i sistemi a stadi
restano in servizio, perché sono più piccoli e più facili da vincolare a un
elenco di parole ammesse. La bassa latenza, invece, non è più una loro
prerogativa da un pezzo: la dettatura che compare sullo schermo mentre si
parla è quella dei trasduttori della sezione precedente, che sono end-to-end e
frame-sincroni. Whisper, quello no: il suo decoder somma su tutto l'encoder e
lavora a finestre di trenta secondi, quindi trascrive a blocchi, non in
diretta.

E da lì vengono anche i suoi limiti, che gli autori dichiarano onestamente. Il
decoder non è vincolato a scorrere l'audio in avanti: quando l'allineamento fra
testo e suono si allenta, quello che resta è un modello di linguaggio molto
bravo che continua a scrivere per conto proprio, senza più guardare il suono.
Sull'audio lungo, che Whisper affronta una finestra di trenta secondi alla
volta e usando i propri tempi predetti per decidere dove tagliare la
successiva, gli errori si propagano da una finestra all'altra e si vedono i
tre sintomi elencati nel paper: le prime o le ultime parole di un segmento non
trascritte, le ripetizioni in loop, il testo inventato di sana pianta (in gergo
*allucinato*: il modello scrive parole che nessuno ha detto). È l'effetto che
si osserva nei sottotitoli automatici quando riempiono di frasi un passaggio in
cui nessuno parla. Ed è anche il motivo per cui una decodifica ingorda non
basta: gli autori usano una ricerca a fascio a cinque ipotesi, con una
temperatura che si alza quando il testo prodotto insospettisce, proprio per
ridurre i loop.

## Il modello di linguaggio, il correttore silenzioso

L'evidenza acustica, da sola, è ambigua. In italiano «l'ago» e «lago», «l'una»
e «luna» suonano identici; è il contesto a decidere. Qui entra il **modello di
linguaggio** (LM), che assegna una probabilità alle sequenze di parole
plausibili e sposta la trascrizione verso ciò che «suona» come italiano
corretto.

Il modo di farlo entrare cambia con l'epoca. Nei sistemi classici il modello di
linguaggio non si affianca al riconoscitore: viene compilato **dentro** il
grafo di decodifica insieme al dizionario di pronuncia, con un peso che regola
quanto contare il suo giudizio rispetto a quello dell'orecchio. Nei modelli
end-to-end quel grafo non c'è più, e lo stesso effetto si ottiene sommando il
punteggio di un LM esterno a quello del modello **a ogni passo della ricerca a
fascio** (si chiama *shallow fusion*, «fusione superficiale»
{cite}`kannan2018analysis`), oppure lasciando finire la ricerca e riordinando
con il LM le prime $n$ ipotesi che ha prodotto.

Quanto serva, però, dipende da quale modello si sta usando, e le due famiglie
non stanno affatto sulla stessa barca. Un modello con decoder autoregressivo,
come Whisper o come LAS, un modello di linguaggio ce l'ha già dentro: lo ha
imparato senza volerlo, perché genera ogni token guardando quelli che ha già
scritto. Per lui un LM esterno è un accessorio, utile sui termini rari o di
dominio (nomi propri, sigle, gergo medico) e poco altro. Un modello CTC no, ed
è esattamente il contrario: l'indipendenza condizionale fra i frame gli vieta
per costruzione di modellare il testo in uscita, quindi un modello di
linguaggio interno **non ce l'ha**. Per lui il LM esterno non è un
miglioramento marginale: è il pezzo che gli manca, e senza di quello i
caratteri escono quasi giusti ma sparpagliati su parole che non esistono. Il
trasduttore sta in mezzo, e ora si capisce perché: la sua *prediction network*
è il modello di linguaggio interno che alla CTC mancava.

## Misurare gli errori: il Word Error Rate

Come diciamo che una trascrizione è «buona»? La metrica standard è il **Word
Error Rate** (WER), e l'idea è quella della **distanza di edit**: il numero
minimo di correzioni (cambia una parola, toglila, aggiungine una) che servono
per trasformare la trascrizione prodotta in quella di riferimento. Si conta a
livello di parola.

$$
\text{WER} = \frac{S + D + I}{N},
$$

dove $S$ è il numero di **sostituzioni**, $D$ le **cancellazioni**, $I$ le
**inserzioni** e $N$ il numero di parole nel riferimento. Un WER di $0$ è la
trascrizione perfetta; può superare $1$ se il modello inserisce più parole di
quante ce ne siano. Calcolarlo è il classico conto della **distanza di
Levenshtein**, che è il nome proprio della distanza di edit e si risolve
riempiendo una tabella (chi non programma può saltare il riquadro: il conto è
quello della formula qui sopra).

```python
import numpy as np

def wer(rif, ip):
    r, h = rif.split(), ip.split()
    # matrice di distanza di edit riempita per programmazione dinamica
    D = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    D[:, 0] = np.arange(len(r) + 1)   # cancellazioni pure
    D[0, :] = np.arange(len(h) + 1)   # inserzioni pure
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            costo = 0 if r[i - 1] == h[j - 1] else 1
            D[i, j] = min(D[i - 1, j] + 1,          # cancellazione
                          D[i, j - 1] + 1,          # inserzione
                          D[i - 1, j - 1] + costo)  # sostituzione (o parola uguale)
    return D[len(r), len(h)] / len(r)

print(round(wer("il gatto nero salta sul muro",
                "il gatto nemo salta muro"), 3))
# -> 0.333  (1 sostituzione + 1 cancellazione su 6 parole)
```

Il WER è comodo ma grezzo: pesa allo stesso modo un errore grave e uno banale,
e penalizza le lingue ricche di composti. Per questo, accanto a esso, si
riporta spesso il *Character Error Rate* (CER), che conta gli stessi errori a
livello di carattere. Nessuna metrica, però, cattura del tutto ciò che conta
davvero: se la frase trascritta, letta da un essere umano, significa ancora la
cosa giusta.

Tiriamo le fila, ciascuno al proprio livello.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- L'audio è lunghissimo e il testo è corto, e **nessuno dice quale pezzetto di
  suono corrisponde a quale lettera**: è il problema centrale del
  riconoscimento vocale.
- La **CTC** lo aggira con il simbolo «vuoto»: prova tutti i modi di
  etichettare i frame e somma le probabilità di quelli che danno la parola
  giusta. I modelli **con attenzione** invece scrivono una lettera alla volta,
  rileggendosi ogni volta il pezzo di audio che serve; il **trasduttore** fa
  come uno stenografo, scrive mentre ascolta senza tornare indietro.
- Trovare la trascrizione non è prendere il simbolo più votato a ogni frame:
  quello è il **percorso** più probabile, che non è la **parola** più
  probabile. Si cerca a fascio, tenendo aperte più strade.
- **Whisper** mette tutto in un modello solo, che fa molte lingue insieme. Se
  perde il filo fra suono e testo continua a scrivere per conto suo: sono le
  frasi inventate che ogni tanto compaiono nei sottotitoli automatici.
- Il **modello di linguaggio** è il pezzo che sceglie fra «l'ago» e «lago»; il
  **WER** conta quante correzioni servono per rimettere a posto una
  trascrizione, diviso il numero di parole.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Audio e testo hanno **lunghezze diverse** e l'allineamento non è dato: è il
  problema centrale dell'ASR.
- La **CTC** lo risolve con il simbolo «vuoto» e sommando tutti gli
  allineamenti possibili, al prezzo dell'**indipendenza condizionale** fra i
  frame; i modelli **con attenzione** lo imparano in modo morbido, un token
  alla volta, ma perdono monotonia e streaming; il **trasduttore**
  {cite}`graves2012sequence` tiene il reticolo monotono e ci aggiunge una
  *prediction network*, cioè un LM interno.
- **Addestramento e decodifica non sono la stessa cosa**: il *best path* non
  massimizza $p(y \mid X)$, e la beam search del CTC **somma** i percorsi che
  collassano nello stesso prefisso invece di metterli in concorrenza.
- I Transformer end-to-end come **Whisper** {cite}`radford2022robust`
  uniscono tutto in un solo modello multilingue; la loro robustezza è
  *zero-shot*, cioè relativa, e i loro loop nascono dall'allineamento
  testo-audio che si stacca.
- Il **modello di linguaggio** disambigua gli omofoni, si integra per *shallow
  fusion* o per riordino delle $n$ ipotesi, ed è indispensabile alla CTC
  proprio perché la CTC non ne ha uno implicito; il **WER** misura gli errori
  come distanza di edit fra parole.
```

`````
