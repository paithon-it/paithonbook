# I modelli di riconoscimento

Quando pronunci la parola «casa», il microfono del telefono non registra
quattro lettere: registra circa sedicimila numeri al secondo. Ogni numero è la
pressione dell'aria misurata in un istante, si chiama **campione**, e messi in
fila quei numeri raccontano come l'aria ha vibrato. Sedicimila non è un numero
magico, è una scelta: bastano a rendere una voce senza sprecare spazio, e chi
registra musica ne usa quasi il triplo, perché lì servono anche gli acuti che
nel parlato non ci sono.

Il compito del riconoscimento vocale automatico (*Automatic Speech
Recognition*, ASR) è tradurre quel fiume di numeri in una manciata di
caratteri. Sembra un problema di traduzione come un altro, ma nasconde una
difficoltà tutta sua, che ha condizionato per decenni il modo in cui si
costruiscono questi modelli.

## Il problema dell'allineamento

Prima di dare in pasto l'audio a una rete lo trasformiamo in uno
**spettrogramma**: tagliamo il segnale in finestrelle di circa 25 millesimi di
secondo, una nuova ogni 10, e per ciascuna misuriamo quanta energia c'è a ogni
frequenza, cioè a ogni altezza sonora. Ogni finestrella è un *frame*, e un
secondo di parlato diventa così un centinaio di frame.

Le finestrelle si sovrappongono, e non è una svista. Il taglio non lo diamo
netto come con le forbici: ai bordi di ciascuna abbassiamo apposta il suono
fino quasi a zero, e la ragione la conosci se hai mai staccato di colpo una
canzone. Quel «tac» che si sente non c'era nella canzone: l'ha fatto lo
stacco. Un taglio brusco aggiunge un suono suo, e in una misura che serve
proprio a dire quali suoni ci sono è l'ultima cosa che vogliamo. Sfumiamo i
bordi per evitarlo; ma allora ai bordi il suono conta quasi niente, e senza
sovrapposizione quello che capita lì andrebbe perso.

La trascrizione, invece, è lunga poche decine di caratteri. Due sequenze di
lunghezza molto diversa, e nessuno ci dice quale frame corrisponde a quale
lettera.

`````{tab} Elementare
Immagina di dover sottotitolare un video a orecchio, senza conoscere i tempi.
Chi parla lento, chi veloce; una vocale tenuta a lungo («caaasa») occupa molti
fotogrammi ma resta una sola lettera; tra una parola e l'altra ci sono pause e
respiri che non vanno scritti. Sai *cosa* è stato detto, ma non *quando*
comincia e finisce ogni suono. Questo è l'allineamento: appaiare i tanti
pezzetti di audio ai pochi caratteri del testo.
`````

`````{tab} Superiore
Abbiamo un input $\mathbf{X} = (\mathbf{x}_1, \dots, \mathbf{x}_T)$ di $T$
frame e un target
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

La svolta arriva nel 2006, e l'idea è di Alex Graves e colleghi: aggiungere
all'alfabeto un simbolo speciale, il «vuoto» (*blank*, $\varnothing$), che
significa «qui non produco nessun carattere». Il metodo si chiama
**Connectionist Temporal Classification**, un nome che non aiuta nessuno e che
infatti si abbrevia sempre: **CTC**, e sono quelle tre lettere a contare.

Per **ogni** frame la rete non sceglie un simbolo secco. Distribuisce cento
punti fra tutti i simboli dell'alfabeto, vuoto compreso: dieci alla «A», due
alla «B», e così via fino a esaurirli. Sono percentuali, e quello che conta è
che siano cento in tutto, cioè che tutta la fiducia della rete finisca da
qualche parte. (In gergo si dice che i voti sommano a uno, perché il 100% si
scrive anche «1», e che la rete emette una *distribuzione*, cioè un modo di
spartire la fiducia fra più possibilità.)

Nella figura qui sotto è disegnato, per ogni frame, il simbolo che ha preso il
voto più alto. Poi una regola di collasso ripulisce la sequenza: prima unisce
i caratteri uguali consecutivi, poi elimina i vuoti
({numref}`fig-ctc-allineamento`).

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

Questi modi hanno un nome, e da qui in avanti useremo quello: si chiamano
**allineamenti**, perché ciascuno dice come i pezzi di suono si appaiano alle
lettere.

Ogni allineamento ha una sua probabilità, cioè quanto la rete ci crede, e si
ottiene moltiplicando fra loro i voti che la rete ha dato ai sette simboli di
quella riga. Si moltiplica perché quel modo si realizza solo se il primo frame
prende quel simbolo **e** il secondo prende quel simbolo **e** così via per
tutti e sette: è la stessa regola per cui, tirando due dadi, la probabilità di
fare due sei è un sesto per un sesto, cioè uno su trentasei.

Sette voti moltiplicati fra loro danno un numero piccolo, quindi mettiamo che
il primo allineamento valga il 3% e il secondo il 2% (sono numeri inventati
per l'esempio; poco più avanti rifaremo il conto per intero, su un caso
piccolo abbastanza da starci tutto). Tutti e due, ripuliti, danno «PALLA»:
quindi finora «PALLA» vale il 5%, più di quanto valga ciascuno da solo. Dico
«finora» perché di modi che danno «PALLA» ce ne sono altri, e vanno sommati
anche quelli. Ecco cosa vuol dire «sommare gli allineamenti».

La CTC non sceglie dunque *un* allineamento giusto e non chiede alla rete di
indovinarlo: li considera tutti insieme, somma le probabilità di quelli che
danno la trascrizione corretta, e spinge la rete ad alzare quel totale. Come
lo alzi (spostando i voti su un modo o sull'altro) sono affari suoi. Nota il
trucco del vuoto: senza il ∅ in mezzo, le due «L» si fonderebbero in una sola.

Gli allineamenti sono tanti anche per una parola di cinque lettere, e quanti
siano non è una magia: si contano, e il conto si può rifare a mano. Il più
corto sta in sei frame, uno per lettera più il vuoto obbligatorio in mezzo
alle due «L»: `P A L ∅ L A`, e in sei frame non ce n'è nessun altro.

Con sette frame ne avanza uno, e lo si può spendere in due modi soltanto. O si
tiene un simbolo per due frame invece che per uno, e allora ci sono sei
possibilità, una per ciascuno dei sei simboli (`P P A L ∅ L A`,
`P A A L ∅ L A`, e così via). O si infila un vuoto in più, e i buchi in cui
infilarlo sono sette: prima della «P», poi i cinque fra un simbolo e l'altro,
poi dopo la «A» finale. Due di quei sette buchi però sono attaccati al vuoto
che c'è già, e metterci un altro vuoto vuol dire tenere il vuoto per due
frame, che è un caso già contato fra i sei di prima. Restano cinque buchi
buoni. Sei più cinque, undici, e su un foglio si controllano tutti in un
minuto.

Con cinquanta frame, che sono quelli che «palla» occupa davvero, lo stesso
conto portato avanti con pazienza dà quasi ventiquattro miliardi di modi.
Sommarli tutti sembra un lavoro impossibile, e invece si fa in fretta. La
ragione è che i modi si somigliano: prendine due che per i primi tre frame
sono identici e si separano al quarto, e vedrai che il conto dei primi tre
frame è lo stesso per tutti e due. Basta farlo una volta e riusarlo. Mettendo
insieme tutti i modi che condividono l'inizio, il lavoro non è più
ventiquattro miliardi di conti: è riempire una tabella lunga cinquanta
colonne, una per frame, e alta quanto la parola scritta con i vuoti in mezzo.
Qualche centinaio di caselle in tutto, ed è lo stesso risparmio del navigatore
di Viterbi che abbiamo visto nel capitolo sul linguaggio naturale.

C'è però un prezzo, e fra qualche pagina conterà. La rete vota un frame alla
volta, e ogni voto lo dà guardando il suono e nient'altro: non si rilegge mai
quello che ha già scritto. Non sa, cioè, che dopo «c-a-n» in italiano viene
molto più facilmente una «e» che una «q». È un'ignoranza che le costerà cara, e
a cui qualcun altro dovrà rimediare al posto suo.
`````

`````{tab} Superiore
La probabilità di una trascrizione $y$ è la somma su tutti i percorsi
frame-level $\pi$ che, collassati, la producono:

$$
p(y \mid \mathbf{X}) = \sum_{\pi \,\in\, \mathcal{B}^{-1}(y)}
\prod_{t=1}^{T} p_t(\pi_t \mid \mathbf{X}),
$$

dove $\pi = (\pi_1, \dots, \pi_T)$ è un allineamento a livello di frame,
$p_t(\pi_t \mid \mathbf{X})$ è la probabilità che la rete assegna al simbolo
$\pi_t$ al frame $t$, e $\mathcal{B}$ è la funzione di collasso. Quanti sono i
termini si conta con la stessa formula, mettendo tutte le $p_t$ a uno: per
`PALLA` sono undici sui sette frame della figura, mille e uno su dieci frame,
quasi ventiquattro miliardi sui cinquanta che quella parola occupa davvero. La
somma si calcola comunque in tempo $O(T \cdot U)$ (lineare nella
lunghezza dell'audio, a trascrizione fissata) con l'algoritmo di
programmazione dinamica *forward-backward*, che lavora sul reticolo dei
$2U+1$ simboli della trascrizione estesa con i blank. Si addestra minimizzando
$\mathcal{L} = -\log p(y \mid \mathbf{X})$.

Due limiti strutturali, e sono conseguenze dirette della formula. Il primo: la
CTC emette esattamente un simbolo per frame, quindi i frame devono bastare, e
la soglia è $T \ge U + r$, dove $r$ conta le coppie di simboli uguali
consecutivi in $y$, ciascuna delle quali vuole in mezzo un vuoto che la
separi; sotto quella soglia $\mathcal{B}^{-1}(y)$ è l'insieme vuoto e la loss
non è nemmeno definita. `PALLA` ha $U = 5$ e $r = 1$: in cinque frame non ci
sta, in sei ci sta in un modo solo (`P A L ∅ L A`), e da lì in poi i modi si
moltiplicano. È il motivo per cui il metodo serve all'ascolto e non alla
sintesi vocale, dove il testo in ingresso è più corto del suono in uscita
{cite}`graves2012sequence`. Il secondo, il più citato: nel prodotto non compare
nessun fattore della forma $p(y_u \mid y_{<u})$, cioè le predizioni ai vari
frame sono **condizionatamente indipendenti** dato $\mathbf{X}$. Non è che la
CTC modelli «non bene» le dipendenze fra i caratteri in uscita: non ha il posto
dove metterle. Torneremo su questo punto parlando del modello di linguaggio,
perché è di lì che discende tutto il resto.
`````

## Dalla rete alla frase: la decodifica

Fin qui abbiamo detto come si **addestra** un modello CTC, non come gli si fa
scrivere una frase. Sono due cose diverse, e la differenza è più grossa di
quanto sembri: il passaggio dai voti della rete alla trascrizione si chiama
**decodifica**, ed è una storia a sé.

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
danno la stessa parola ce ne sono a miliardi, come abbiamo contato poco fa per
«palla».

`````

`````{tab} Superiore

Il *best path* massimizza il singolo allineamento,

$$
\pi^{*} = \arg\max_{\pi} \prod_{t=1}^{T} p_t(\pi_t \mid \mathbf{X}),
\qquad \hat{y} = \mathcal{B}(\pi^{*}),
$$

mentre l'obiettivo che il modello è stato addestrato a massimizzare è
$\arg\max_y p(y \mid \mathbf{X})$, cioè la **somma** su
$\mathcal{B}^{-1}(y)$. Le due
quantità sono diverse perché $\mathcal{B}$ non è iniettiva: molti percorsi
cadono sulla stessa etichettatura, e la loro massa può battere il massimo
puntuale. Il controesempio minimo ha $T = 2$ e alfabeto
$\{\varnothing, \texttt{A}\}$, con $p_t(\varnothing) = 0{,}6$ su entrambi i
frame: il percorso migliore è $\varnothing\varnothing$ e vale $0{,}36$, ma
l'etichettatura `A` raccoglie i tre percorsi restanti e vale
$0{,}24 + 0{,}24 + 0{,}16 = 0{,}64$.

Graves e colleghi lo scrivono già nel paper del 2006
{cite}`graves2006connectionist`, nella sezione in cui costruiscono il
classificatore: per l’$\arg\max$ esatto «non conosciamo un algoritmo di
decodifica trattabile in generale». Al suo posto propongono due metodi
approssimati. Il primo è proprio il *best path*, che costa niente e non
garantisce di trovare l'etichettatura più probabile. Il secondo è la *prefix
search decoding*, che lavora sui **prefissi**
invece che sui percorsi e, dato tempo a sufficienza, l'etichettatura più
probabile la trova davvero. Il tempo però cresce in fretta, perché il numero
di prefissi da espandere cresce esponenzialmente con la lunghezza dell'audio;
gli autori osservano che se la distribuzione in uscita è abbastanza appuntita
la ricerca finisce comunque in tempi ragionevoli, ma per il loro stesso
esperimento servì un'euristica in più (spezzare la sequenza dove il vuoto è
molto probabile). Quella che si usa oggi è la versione col freno a mano: una
ricerca a fascio sui prefissi, che ne tiene aperti $k$ e getta gli altri.

`````

La ricerca a fascio (**beam search**) l'abbiamo già incontrata nella
traduzione automatica, nel capitolo sul linguaggio naturale: invece di
decidere subito, si tengono aperte le $k$ ipotesi più promettenti e si va
avanti qualche passo prima di scegliere. L'idea è la stessa, ma una cosa
cambia, ed è proprio quella di prima: qui molti percorsi diversi danno la
stessa identica parola. Nella traduzione le ipotesi
**competono**: due strade diverse sono due frasi diverse, e alla fine ne resta
una. Nel CTC no: due percorsi che si ripuliscono nello stesso testo sono la
stessa ipotesi, e i loro punteggi vanno **sommati** invece di essere messi in
concorrenza. Una beam search che se ne dimentica scarta la trascrizione
giusta, esattamente come fa il percorso migliore.

È anche il punto in cui entrano in scena due cose che vedremo fra poco: un
modello di linguaggio, che a ogni passo della ricerca aggiunge il proprio
giudizio al punteggio, e la lista delle prime $n$ ipotesi, che la ricerca
produce come sottoprodotto e che si può riordinare a posteriori.

Il caso dei due frame si rifà in poche righe di codice, elencando i quattro
percorsi e sommandoli, ed è il modo più rapido di convincersene: il percorso
più votato vale 0,36 e non scrive niente, la lettera «A» vale 0,64. Chi non
programma può saltare il riquadro senza perdere nulla, perché quei due numeri
sono già qui.

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

Da qui in avanti, quando diremo che un sistema CTC «trascrive», intendiamo
sempre una ricerca di questo tipo, non il simbolo più votato frame per frame.

## Ascoltare e attendere: i modelli con attenzione

C'è una seconda famiglia, e il vuoto non ce l'ha proprio. Il lavoro qui è
diviso fra due parti della stessa rete: una ascolta tutto l'audio e se lo
riassume (si chiama *encoder*, «codificatore»), l'altra scrive il testo un
carattere alla volta (il *decoder*), e ogni carattere lo sceglie tenendo conto
di quelli che ha già scritto. In gergo si dice che procede in modo
*autoregressivo*, cioè rileggendosi.

A ogni passo il decoder deve decidere quale pezzo di audio guardare, e la cosa
che glielo fa decidere si chiama **attenzione**. È di nuovo l'allineamento di
cui questa pagina parla dall'inizio, e anche qui il modello se lo impara da
solo; la differenza con la CTC è che la CTC è obbligata ad andare avanti frame
per frame, mentre l'attenzione può guardare dove le pare, avanti o indietro.
Un allineamento morbido, insomma, invece che a scatti. L'architettura di
riferimento è *Listen,
Attend and Spell* {cite}`chan2016listen`, ed è da lì che viene il titolo di
questa sezione: «attendere» traduce l'inglese *attend*, che non vuol dire
aspettare ma «fare attenzione a».

`````{tab} Elementare
Pensa a un interprete: prima ascolta l'intera frase, poi la ridice parola per
parola. Mentre pronuncia ogni parola, la sua attenzione torna al punto giusto
di ciò che ha sentito. Il modello fa lo stesso: genera un carattere, si
«riguarda» la porzione di audio più rilevante, genera il prossimo.

Due conseguenze, e conviene fissarle perché tornano più avanti. La prima è che
l'interprete, per cominciare, deve aver ascoltato la frase fino in fondo: un
modello così non può scrivere mentre uno sta ancora parlando. La seconda è che
niente gli impedisce di sbagliare punto. Può riguardare un pezzo di audio che
ha già tradotto, o saltarne uno del tutto, e allora ripete una sillaba o si
mangia una parola; nei casi peggiori si impunta e ripete la stessa cosa
all'infinito. La CTC questo difetto non ce l'ha, perché va avanti frame per
frame e indietro non torna mai.
`````

`````{tab} Superiore
Al passo $i$ il decoder costruisce un vettore di contesto come media pesata
degli stati dell'encoder $\mathbf{h}_j$ (è la stessa formula dell'attenzione
di Bahdanau vista nella traduzione automatica, con l'audio al posto della
frase sorgente):

$$
\mathbf{c}_i = \sum_{j=1}^{T_{\text{enc}}} \alpha_{ij}\,\mathbf{h}_j,
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
insieme: il **trasduttore neurale**, proposto da Alex Graves nel 2012
{cite}`graves2012sequence`, cioè da chi aveva scritto la CTC sei anni prima.
Nei testi si trova sempre con la sigla **RNN-T**, dove le prime tre lettere
sono le reti ricorrenti del capitolo sul linguaggio naturale, quelle che
leggono una sequenza un pezzo alla volta tenendosi in mente il pezzo di prima.

Quella data va guardata. Il trasduttore non arriva *dopo* i modelli con
attenzione per rimediare ai loro difetti: li precede di tre anni (*Listen,
Attend and Spell* è del 2015 e arriva in conferenza l'anno dopo). È nato dal
lato della CTC, per togliere alla CTC il difetto che il suo autore le
conosceva meglio di chiunque.

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

Restando fermo sull'audio, in teoria, potrebbe scrivere all'infinito: a
fermarlo non c'è una regola, c'è il fatto che dopo aver scritto quello che
quel pezzo di suono conteneva il vuoto diventa la mossa più votata, e allora
si sposta. Se un modello mal addestrato si incaponisce a scrivere, infatti,
esce proprio quello: una parola ripetuta finché qualcuno non stacca la spina.

E quel foglio, quello che lo stenografo si rilegge, dentro il trasduttore è
una rete a sé, con un nome che vale la pena ricordare perché torna fra due
pagine: la *prediction network*, «la rete che prevede», il cui unico mestiere
è guardare le lettere già scritte e dire cosa ci si aspetta dopo. È la cosa
che alla CTC manca del tutto.

`````

`````{tab} Superiore

Il trasduttore accoppia tre reti: un *encoder* (o *transcription network*) che
produce $\mathbf{h}_t$ dai frame acustici, una *prediction network* che produce
$\mathbf{g}_u$ dai soli token già emessi $y_{<u}$, e una piccola *joint
network* che fonde le due e proietta sul vocabolario esteso col vuoto,

$$
p(k \mid t, u) =
\mathrm{softmax}\big(\mathbf{W}\,\phi(\mathbf{h}_t + \mathbf{g}_u)\big),
$$

dove $k$ è il simbolo candidato, $\phi$ una non linearità (di solito una
tangente iperbolica) e $\mathbf{W}$ la proiezione sul vocabolario. Nella
formulazione originale del 2012 le due reti si sommavano direttamente nello
spazio delle uscite; la *joint network* con la non linearità in mezzo è la
forma che si è imposta dopo, ed è quella che si trova nelle librerie.

Lo spazio degli allineamenti non è più una sequenza di $T$ etichette ma un
reticolo $T \times U$: emettere un token muove di uno in verticale, emettere il
vuoto muove di uno in orizzontale, e ogni cammino monotono dall'angolo in basso
a sinistra a quello in alto a destra è un allineamento valido. La probabilità
della trascrizione è ancora la somma su tutti i cammini, calcolata con lo
stesso forward-backward della CTC, e la loss è ancora
$-\log p(y \mid \mathbf{X})$.

Due conseguenze, ed è tutto il punto. La prediction network è a tutti gli
effetti un modello di linguaggio interno, condizionato sui token già emessi:
il trasduttore modella cioè le dipendenze uscita-uscita che la CTC non ha dove
mettere. E $\mathbf{h}_t$ dipende solo dai frame fino a $t$ (con un encoder
causale),
quindi la decodifica è frame-sincrona e non ha bisogno della fine dell'audio:
si trascrive mentre si ascolta.

`````

Non è un'architettura di nicchia: è quella su cui gira, dal 2019, la dettatura
in tempo reale sui telefoni {cite}`he2019streaming`, dove la risposta deve
arrivare mentre si parla e il modello deve stare dentro un dispositivo. Vale la
pena tenerlo a mente fra qualche pagina, quando ci chiederemo a che cosa serva
ancora, oggi, la vecchia catena a stadi.

## Whisper e i Transformer end-to-end

Nel settembre 2022 OpenAI rilascia **Whisper**, e la novità si dice in una
riga: una rete sola, che riceve l'immagine a bande del suono e restituisce il
testo, senza nessuno stadio in mezzo. La rete è un Transformer, diviso in
encoder e decoder come i modelli con attenzione di poco fa; e l'immagine a
bande è lo spettrogramma di sempre, quello di inizio pagina, in una versione
che si chiama **log-mel** perché misura le altezze sonore come le sente
l'orecchio e non come le misurerebbe uno strumento.

```{figure} ../figures/come-funziona-whisper.svg
:name: fig-whisper
:alt: "Catena di quattro blocchi in fila: l'onda sonora in ingresso, campionata sedicimila volte al secondo, diventa uno spettrogramma log-Mel (tempo per frequenza), che entra in un encoder Transformer; l'uscita dell'encoder alimenta un decoder autoregressivo, e sotto il decoder escono i token di testo uno dopo l'altro, «⟨it⟩ Buon giorno». Non ci sono moduli separati per il dizionario di pronuncia o per il modello di linguaggio."
:width: 96%

La catena di Whisper, tutta qui. Fra lo spettrogramma e il testo non c'è
nessuno stadio con regole scritte a mano: encoder e decoder sono addestrati
insieme, in un pezzo solo.
```

Quello che manca in {numref}`fig-whisper` conta quanto quello che c'è. Nella
catena della panoramica il primo blocco e l'ultimo erano l'audio che entra e
il testo che esce; i pezzi veri, quelli da costruire, erano quelli in mezzo, e
lì ce n'era anche uno che la figura non mostrava. Sono tre, ciascuno messo a
punto lingua per lingua: il modello acustico, il modello di linguaggio e, fra
i due, un **dizionario di pronuncia**, cioè un elenco compilato a mano che
dice di quali suoni è fatta ogni parola.

Quei tre compiti restano anche qui, ma nessuno
li ha più assegnati a un pezzo suo: sono sparsi nei pesi, cioè nei numeri che
la rete ha imparato, ed è questo il motivo per cui un modello solo copre
decine di lingue.

La sua forza, però, non è tanto l'architettura quanto i dati: 680.000 ore di
audio raccolte dal web con **etichettatura debole**, cioè trascrizioni già
esistenti in rete, scritte da qualcuno per i propri scopi e non per addestrare
un modello. «Debole» non vuol dire «non curata». Gli autori le passano al
setaccio con filtri automatici, buttando via quelle prodotte da altri
riconoscitori (imparare da un altro riconoscitore vuol dire ereditarne gli
errori), le coppie in cui la lingua parlata non è quella scritta e i
duplicati; e ispezionano a mano le fonti che sbagliano di più, per eliminarle.

Quelle ore coprono quasi cento lingue, l'inglese e altre novantasei, ma non
allo stesso modo: l'inglese se ne prende circa due terzi, e la maggior parte
delle altre sta sotto le mille ore. È da qui che viene il salto di qualità che
si sente passando all'italiano, e gli autori ne ricavano una regola: la quota
di parole sbagliate (il tasso di errore, che a fine pagina impareremo a
misurare) si dimezza ogni volta che le ore di una lingua si moltiplicano per
sedici. Non è una classifica fra lingue, è una misura di quanto costa fare
meglio, e il costo cresce in fretta.

Con lo stesso modello Whisper trascrive, traduce verso l'inglese e riconosce
la lingua. A dirgli quale dei tre mestieri fare sono delle istruzioni infilate
nel decoder sotto forma di simboli che non si pronunciano (in gergo *token*
speciali): uno dice in che lingua si sta parlando, un altro se il compito è
trascrivere o tradurre.

Quello che gli autori rivendicano non è che Whisper sbagli meno di tutti, ed è
una distinzione che vale la pena tenere. Per confrontare i riconoscitori si
usano dei **benchmark**, che sono prove d'esame standard: raccolte di
registrazioni con accanto la trascrizione giusta, sempre le stesse per tutti.
La grandezza che gli autori misurano è la robustezza *zero-shot*, cioè come se
la cava Whisper su una prova su cui non si è mai allenato: ci va meglio di
quanto la sua bravura altrove lascerebbe prevedere, e peggiora più lentamente
degli altri man mano che si alza il rumore di fondo. Sull'audio pulito, invece,
i modelli allenati apposta per quella prova gli restavano davanti. È una misura
di *quanto si peggiora fuori casa*, non di quanto si è bravi.

E ha un fianco scoperto, che il paper dichiara. I dati vengono dal web, e sul
web stanno anche i benchmark: se le frasi dell'esame erano già dentro il
materiale di studio, il voto è gonfiato. Gli autori il controllo l'hanno
fatto, ma su una raccolta sola (TED-LIUM 3) e confrontando le trascrizioni
scritte, non l'audio; il che vuol dire che una registrazione ripubblicata
altrove con parole leggermente diverse sarebbe passata inosservata.

Due precisazioni, per non lasciare a Whisper meriti che non ha e difetti che
non sono solo suoi.

La prima: non è Whisper ad aver mandato in pensione la catena a stadi. Il
passaggio a una rete sola era cominciato anni prima, con la CTC e con i
modelli ad attenzione di queste pagine; Whisper ne è la vetrina più visibile,
non l'inizio. E la vecchia catena non è nemmeno sparita: dove le parole da
riconoscere sono poche e note in anticipo (i comandi di un centralino
telefonico, i codici letti ad alta voce in un magazzino) i sistemi a stadi
restano in servizio, perché sono più piccoli e si lasciano obbligare a
scegliere solo dentro un elenco di parole ammesse.

La seconda riguarda la trascrizione in diretta, e va detta perché Whisper è
così famoso che si finisce per credere che faccia tutto. Questo no: prima di
scrivere la prima parola deve aver ascoltato tutto quello che gli è stato
dato, e gliene diamo trenta secondi alla volta, quindi trascrive a blocchi. La
dettatura che compare sullo schermo mentre parli è un'altra cosa, ed è quella
dei trasduttori di poco fa, che sono end-to-end come lui ma tengono il passo
dell'audio frame per frame.

E da lì vengono anche i suoi limiti, che gli autori dichiarano onestamente. Il
decoder non è obbligato a scorrere l'audio in avanti, e quando il legame fra
testo e suono si allenta quello che resta è un modello di linguaggio molto
bravo che continua a scrivere per conto proprio, senza più guardare il suono.
Sull'audio lungo il guasto si aggrava, perché ogni finestra di trenta secondi
comincia dove il modello stesso ha deciso che finiva la precedente: se ha
sbagliato a decidere, l'errore passa alla finestra dopo. Il paper elenca tre
sintomi: le prime o le ultime parole di un segmento che non vengono
trascritte, le ripetizioni in loop, e il testo inventato di sana pianta (in
gergo *allucinato*: il modello scrive parole che nessuno ha detto). È quello
che si vede nei sottotitoli automatici quando riempiono di frasi un passaggio
in cui non parla nessuno.

Ecco perché prendere sempre il boccone più grosso, cioè il simbolo più votato
a ogni passo, non basta: è la solita decodifica del percorso migliore, e in
gergo si chiama *ingorda*. Gli autori usano al suo posto una ricerca a fascio
a cinque ipotesi, e quando il testo prodotto insospettisce alzano la
**temperatura**, la stessa manopola della sezione sui grandi modelli
linguistici, nel capitolo sui Transformer.

Vale la pena dire come funziona questo sospetto, perché è ingegnoso e non
serve nessuno che ascolti. Un campanello suona se il testo prodotto si
ripete troppo, e per accorgersene basta comprimerlo: un testo che si ripete si
comprime moltissimo, e se si comprime troppo qualcosa non va. L'altro suona se
la rete stessa, guardando i voti che ha dato, risulta poco convinta di quello
che ha appena scritto. Quando uno dei due suona si rifà il pezzo a temperatura
più alta, e siccome una temperatura alta rende il modello meno incaponito
sulla parola che gli sembra ovvia, gli capiterà di provarne una diversa: è
esattamente quello che serve per uscire da un loop, dove il modello continua a
riscegliere la stessa cosa.

## Il modello di linguaggio, il correttore silenzioso

Quello che dice il suono, da solo, non basta mai. In italiano «l'ago» e
«lago», «l'una» e «luna» si pronunciano allo stesso identico modo: a decidere
è il contesto. Qui entra il **modello di linguaggio** (LM), che sa quali
sequenze di parole sono frasi plausibili e sposta la trascrizione verso ciò
che «suona» come italiano corretto.

Il modo di farlo entrare cambia con l'epoca. Nei sistemi classici il modello
di linguaggio non si affiancava al riconoscitore. Quei sistemi, prima di
ascoltare, si costruivano una specie di mappa stradale di tutto ciò che si
poteva dire: dai suoni alle parole, dalle parole alle frasi, con un costo
scritto su ogni strada. Trascrivere voleva dire attraversare quella mappa
cercando il percorso più economico. Il modello di linguaggio non arrivava dopo:
i suoi giudizi erano già scritti nei costi delle strade, insieme al dizionario
di pronuncia, e una manopola regolava quanto contassero rispetto al parere
dell'orecchio.

Nei modelli end-to-end quella mappa non si costruisce più, e lo stesso effetto
si ottiene in due modi. O si somma il
punteggio di un modello di linguaggio esterno a quello del riconoscitore **a
ogni passo della ricerca a fascio** (si chiama *shallow fusion*, «fusione
superficiale» {cite}`kannan2018analysis`), o si lascia finire la ricerca e si
riordinano con il modello di linguaggio le prime $n$ ipotesi che ha prodotto.

Quanto serva, però, dipende da quale modello si sta usando, e le due famiglie
non stanno affatto sulla stessa barca. Un modello che scrive rileggendosi,
come Whisper o come il *Listen, Attend and Spell* di prima (in gergo: con
decoder autoregressivo), un modello di linguaggio ce l'ha già dentro. Lo ha
imparato senza volerlo, perché sceglie ogni pezzo di testo guardando quelli
che ha già scritto. Per lui un modello di linguaggio esterno è un accessorio,
utile sui termini rari o di dominio (nomi propri, sigle, gergo medico) e poco
altro. Un modello CTC è esattamente il contrario: decide ogni frame per conto
suo, guardando il suono e mai le lettere che ha già scritto, ed è
quell'ignoranza annunciata qualche pagina fa. Non è che modelli male il testo
in uscita: non ha il posto dove metterlo, e un modello di linguaggio interno
**non ce l'ha**. Per lui quello
esterno non è un miglioramento marginale, è il pezzo che gli manca: senza, i
caratteri escono quasi giusti ma sparpagliati su parole che non esistono. Il
trasduttore sta in mezzo, e ora si capisce perché: la sua *prediction network*
(il foglio che lo stenografo si rilegge) è il modello di linguaggio interno
che alla CTC mancava.

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
**inserzioni** e $N$ il numero di parole nel riferimento. Attenzione ai nomi,
perché sono dal punto di vista del sistema e non di chi corregge: una
*cancellazione* è una parola che il sistema si è mangiato, un’*inserzione* è
una parola che ha aggiunto di suo. Chi corregge fa il gesto opposto, ma
l'errore si chiama così. Un WER di $0$ è la trascrizione perfetta; può
superare $1$ se il sistema aggiunge più parole di quante ce ne siano.

Un conto per intero, sull'esempio che il libro si porta dietro dal capitolo
sul linguaggio naturale. Il riferimento è «il gatto nero salta sul muro», sei
parole; il sistema ha scritto «il gatto nemo salta muro». Gli errori sono due:
ha scritto «nemo» al posto di «nero» (una sostituzione) e si è mangiato «sul»
(una cancellazione). Due errori su sei parole, il WER è $2/6 = 0{,}33$, cioè
il 33%.

Trovare quei due errori a occhio è facile su sei parole e impossibile su
seicento, perché le combinazioni sono tante e ne vogliamo il numero **minimo**.
Il conto ha un nome, **distanza di Levenshtein**, e si fa riempiendo una
tabella (chi non programma può saltare il riquadro: il conto è quello appena
fatto a mano).

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

Il WER è comodo ma grezzo. Pesa allo stesso modo un errore grave e uno banale,
e tratta male le lingue che attaccano le parole fra loro: in tedesco
*Geschwindigkeitsbegrenzung* («limite di velocità») è una parola sola, quindi
sbagliarne una sillaba conta come sbagliarla tutta, mentre in italiano lo
stesso inciampo ne intaccherebbe una su tre.
Per questo, accanto al WER, si riporta spesso il *Character Error Rate* (CER),
che conta gli stessi errori a livello di carattere. Nessuna misura, però,
cattura del tutto ciò che conta davvero: se la frase trascritta, letta da un
essere umano, significa ancora la cosa giusta.

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
  allineamenti possibili, al prezzo dell’**indipendenza condizionale** fra i
  frame; i modelli **con attenzione** lo imparano in modo morbido, un token
  alla volta, ma perdono monotonia e streaming; il **trasduttore**
  {cite}`graves2012sequence` tiene il reticolo monotono e ci aggiunge una
  *prediction network*, cioè un LM interno.
- **Addestramento e decodifica non sono la stessa cosa**: il *best path* non
  massimizza $p(y \mid \mathbf{X})$, e la beam search del CTC **somma** i
  percorsi che collassano nello stesso prefisso invece di metterli in
  concorrenza.
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
