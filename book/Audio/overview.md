# Audio oltre la voce

Un mattino di primavera, punti il telefono verso una siepe e apri
un'applicazione. Un merlo canta, invisibile tra le foglie, e sullo schermo
compare un nome: *Turdus merula*. L'app si chiama **BirdNET**, nasce dal
laboratorio di ornitologia della Cornell University insieme all'università
tecnica di Chemnitz, e fa una cosa che cinquant'anni fa sarebbe sembrata magia: riconosce
centinaia di specie di uccelli dal solo canto (un compito che a un umano
richiede anni di orecchio allenato). Nessuna parola, nessuna frase: solo un
fischio modulato, e un modello che sa a chi appartiene.

Il canto degli uccelli è solo un angolo di un mondo enorme. C'è la musica che
un modello genera su richiesta, un brano al pianoforte che non è mai stato
suonato da nessuno. C'è il sistema di sorveglianza di un magazzino che deve
distinguere, nel buio, un allarme da una sirena lontana, il tintinnio di un
vetro rotto dal rumore del vento. C'è il tecnico che, dal ronzio anomalo di un
motore, capisce che un cuscinetto sta per cedere. Tutto questo è **audio**, ed
è quasi tutto tranne la voce.

Quanto è grande questo mondo? Un'idea la dà **AudioSet**, il catalogo con cui
Google, nel 2017, ha provato a mettere ordine {cite}`gemmeke2017audioset`.
Prima hanno fatto l'elenco dei suoni che esistono, e ne sono venute fuori **632
categorie**: dal latrato di un cane al colpo di tosse, dal fruscio della pioggia
al suono di una chitarra elettrica. Poi hanno riempito quelle caselle
ritagliando frammenti da dieci secondi da video di YouTube ed etichettandoli a
mano, e oggi i frammenti raccolti sono **oltre due milioni**. Di categorie ne
hanno usate 527 delle 632: le altre sono rimaste troppo vuote per servire a
qualcosa, ed è già un dato interessante.

Un elenco così lungo dice una cosa sola: il suono che non è parola non è un
rumore indistinto. Ha regole sue, riconoscibili, come le ha una lingua. Un
temporale non comincia a caso; un motore che si sta guastando suona storto in
un modo suo. Solo che qui non ci sono parole.

Il capitolo sullo Speech Recognition, che viene subito dopo, si dedicherà a un
caso particolare e cruciale: la voce, dal parlato al testo e ritorno
(ascoltare e parlare). Questo capitolo viene prima e guarda più in largo
(sentire il mondo, non solo le parole di chi lo abita) e getta le fondamenta
comuni a entrambi.

## Il suono, oltre la voce

Nel riconoscimento vocale il traguardo è sempre lo stesso: da un'onda sonora
ricavare le parole giuste. Qui il traguardo cambia forma. Non c'è più
necessariamente un testo in fondo: a volte vogliamo un’**etichetta** («questo
è un violino»), a volte una **lista di tag** («pioggia, tuono, traffico»), a
volte un suono **nuovo** che prima non esisteva.

Ed è diverso anche il suono da cui si parte. La voce è fatta in un modo tutto
suo: i pochi suoni elementari di una lingua, il timbro che ciascuno di noi ha
perché la gola e la bocca fanno da cassa, il ritmo di chi parla. Un accordo, un
temporale o il rombo di un motore non seguono nessuna di quelle regole.

`````{tab} Elementare

La differenza è quella tra due mestieri. Il primo è la **stenografa** che
trascrive una riunione: sente parole e scrive parole, il senso sta tutto lì.
Il secondo è il **fonico** di un teatro, che a occhi chiusi riconosce ogni
cosa dal suono: «quello è un violino, quello un clacson in strada, là fuori
sta arrivando un temporale». Non trascrive niente (non ci sono parole da
trascrivere) eppure capisce benissimo cosa sta succedendo.

Se il riconoscimento vocale ci rende bravi come la stenografa, questo capitolo
ci insegna il mestiere del fonico: dare un nome ai suoni, elencare tutto ciò che
si sente in una scena, e perfino *inventarne* di nuovi. Sono compiti diversi, su
suoni diversi, e per la maggior parte non c'entrano niente con la voce.

`````

`````{tab} Superiore

Lo Speech Recognition è, formalmente, un problema di **trascrizione di
sequenze**: da $\mathbf{X} = (\mathbf{x}_1, \dots, \mathbf{x}_T)$ acustica a una sequenza di parole
$\hat{W}$. I compiti dell'audio generale hanno firme diverse:

- **classificazione**, dato un segmento, un'unica etichetta su $C$ classi
  ($\hat{y} = \arg\max_c P(c \mid \mathbf{X})$: «violino» vs «pianoforte»);
- **tagging multi-etichetta**, più eventi compresenti, ognuno presente o
  assente ($\hat{\mathbf{y}} \in \{0,1\}^C$: pioggia *e* tuono *e* traffico insieme);
- **rilevazione temporale**, *quando* comincia e finisce ogni evento;
- **generazione**: campionare un $\mathbf{X}$ nuovo da una distribuzione appresa,
  eventualmente condizionata da testo.

La radice comune è che le rappresentazioni tempo–frequenza restano quelle: lo
spettrogramma e la scala mel funzionano per un colpo di tamburo esattamente
come per una vocale. Ciò che cambia è a valle (l'obiettivo, la funzione di
perdita, l'architettura), perché la struttura statistica di musica e suoni
ambientali non è quella, quasi-periodica e vincolata dal tratto vocale, del
parlato.

`````

Questi compiti non sono esercizi da laboratorio. Riconoscere i canti degli
uccelli, moltiplicato per migliaia di microfoni in una foresta, diventa uno
strumento per misurare la biodiversità senza disturbarla. Ascoltare il rumore
di una macchina utensile e accorgersi che «suona storto» è manutenzione
predittiva: si interviene prima che il pezzo si rompa. Distinguere un vetro
infranto o un grido in una registrazione è sorveglianza acustica. E la
classificazione di scene sonore («cucina», «stazione», «ufficio») aiuta un
apparecchio acustico a regolarsi da solo a seconda di dove si trova chi lo
indossa. Ogni volta, il punto di partenza è lo stesso: un suono che non è
parola, e un modello che deve capirlo.

## Le fondamenta, prima di tutto

Prima di correre in avanti c'è un ponte da attraversare, e questo capitolo lo
costruisce **subito**, nella sua prima sezione, *Dal suono alle feature*: il
passaggio da un'onda di pressione ai numeri con cui lavora un modello. È un
ponte che non serve solo qui: regge qualunque suono, e sarà il punto di
partenza anche del capitolo sullo Speech Recognition che segue.

I nomi delle campate, adesso, non diranno niente, ed è normale: la prossima
sezione le monta una per una. Sono elencati qui solo perché si sappia quanto è
lungo il ponte.

- il **campionamento**, che trasforma l'onda continua in una sequenza di numeri,
  con il teorema di Nyquist a dettare quante misure al secondo servono;
- la **trasformata di Fourier** e, applicata a finestre brevi, lo
  **spettrogramma**, l'immagine del suono con il tempo su un asse e le frequenze
  sull'altro;
- la **scala mel** e i **MFCC**, che riassumono quell'immagine imitando
  l'orecchio.

Tutto questo vale identico per il canto di un merlo, per un accordo di chitarra,
per il fragore di un temporale: è il **punto di partenza comune** di ogni
sezione che segue. Da lì in poi diamo per acquisito che un pezzo di audio arrivi
al modello come uno **spettrogramma log-mel**, e siccome quella parola comparirà
in ogni pagina conviene dire subito cosa nasconde. È l'immagine del suono, con
due accorgimenti. Le frequenze sono riscritte come le sente un orecchio, che è
preciso sui suoni gravi e approssimativo sugli acuti (è la parte «mel»); e le
intensità sono schiacciate, in modo che un sussurro si veda accanto a un urlo
invece di sparirci sotto (è la parte «log»). Un oggetto, cioè, che sappiamo
trattare come un'immagine.

## L'idea nuova: l'audio come sequenza di token

Prima della strada nuova conviene guardare quella già battuta, che
{numref}`fig-whisper-pipeline` riassume in quattro passaggi: l'onda diventa
immagine, e dall'immagine un modello ricava le parole. Le parole escono a
destra, una per riquadro, e ciascuno di quei riquadri si chiama **token**. È la
parola che regge tutto il capitolo, e vuol dire una cosa sola: un simbolo preso
da un elenco chiuso, deciso in anticipo. Quanto grosso sia il pezzo che un
token rappresenta cambia da caso a caso (nel disegno è una parola intera,
altrove sarà una sillaba o un frammento di suono), ma la sostanza è quella:
un elenco finito di simboli, e tutto si scrive con quelli.

```{figure} ../figures/whisper-2022.svg
:name: fig-whisper-pipeline
:alt: "Catena in quattro stadi: la forma d'onda dell'audio diventa uno spettrogramma log-mel; lo spettrogramma entra in un unico blocco Transformer, marcato «encoder → decoder»; da lì escono i token di testo, «il», «gatto», «dorme», uno sotto l'altro."
:width: 100%

La via classica, quella del riconoscimento vocale: l'onda diventa immagine,
l'immagine diventa testo. Nel disegno il Transformer è un blocco solo, ma dentro
fa due mestieri: la parte che legge (l’*encoder*) riassume lo spettrogramma e la
parte che scrive (il *decoder*) ne ricava le parole, una alla volta. È la strada
che il capitolo sullo Speech Recognition, subito dopo questo, percorre per
intero. Il pezzo su cui le pagine qui sotto intervengono è il secondo riquadro:
lo spettrogramma resta una tabella di numeri qualsiasi, e c'è un modo di
scriverlo con dei simboli.
```

C'è infatti un secondo modo di guardare l'audio, ed è il vero filo conduttore di
questo capitolo. Poggia su due parole che conviene fissare subito, perché
torneranno in ogni sezione. Riguardano i **valori** che un numero può prendere,
non quanti numeri ci sono. Una grandezza è **discreta** quando i valori
possibili si possono contare a uno a uno: le lettere dell'alfabeto sono ventuno,
e fra la A e la B non c'è niente in mezzo. È **continua** quando non si può: fra
$0{,}3$ e $0{,}4$ ci sono infiniti numeri, e fra due sfumature di grigio ce n'è
sempre una terza.

Un testo nasce discreto, perché è fatto di lettere. Un suono no. Qui c'è un
piccolo tranello, ed è meglio scioglierlo adesso: la fila dei numeri con cui
misuriamo un suono si conta benissimo (sono decine di migliaia al secondo), ma
«continuo» non si riferisce a *quanti* sono, si riferisce a quanto vale
ciascuno. E ciascuno di quei numeri può valere qualsiasi cosa: fra due valori
vicini ce n'è sempre un terzo. È lì che sta la differenza con le lettere.

Lo spettrogramma di {numref}`fig-whisper-pipeline` è dunque una tabella di
numeri *continui*: ogni sua casella può valere qualunque cosa. Ma se
riuscissimo a trasformare un suono in una sequenza di **simboli discreti**
(come le lettere di un testo, o le parole di una frase), allora tutto
l'armamentario che abbiamo costruito per il linguaggio diventerebbe di colpo
applicabile al suono. I Transformer sanno leggere e scrivere sequenze di
simboli: se l'audio *è* una sequenza di simboli, i Transformer sanno leggere e
scrivere audio.

`````{tab} Elementare

Pensa a come è fatto un testo: una manciata di lettere, un alfabeto finito, e
con quelle poche decine di simboli si scrive qualsiasi cosa. Un modello di
linguaggio ha imparato proprio questo: dato un pezzo di frase, indovinare il
simbolo successivo, lettera dopo lettera, parola dopo parola.

L'idea è di dare al suono lo stesso trattamento: ritagliarlo in tanti
**pezzetti** e assegnare a ciascuno un simbolo da un «alfabeto sonoro» finito,
costruito apposta. Un simbolo del genere è esattamente quello che abbiamo
chiamato **token** guardando il disegno qui sopra: sta a un pezzetto di suono
come una lettera sta a una parola scritta. Una volta fatto questo, un brano
musicale diventa una
*frase* scritta in quell'alfabeto, e generare musica nuova diventa la stessa
cosa che generare testo nuovo: indovina il pezzetto successivo, poi il
prossimo, poi il prossimo. La macchina che scrive romanzi impara a comporre
melodie, senza cambiare mestiere.

`````

`````{tab} Superiore

Il passaggio chiave è la **quantizzazione**: sostituire la rappresentazione
continua dell'audio con una sequenza di indici discreti presi da un
*vocabolario* appreso (un *codebook* di vettori prototipo). Un breve segmento
di segnale viene mappato sul vettore del codebook più vicino, e di esso si
tiene solo l'indice intero: un **token**. L'audio diventa così
$z = (z_1, \dots, z_L)$ con $z_i \in \{1, \dots, K\}$, esattamente la forma di
un testo tokenizzato.

Da lì il collegamento con i modelli linguistici è diretto: un Transformer
autoregressivo può modellare

$$
P(z) = \prod_{i=1}^{L} P(z_i \mid z_1, \dots, z_{i-1}),
$$

dove $z_i$ è il token audio in posizione $i$ e ogni fattore è una softmax sul
codebook di dimensione $K$: la stessa fattorizzazione, parola per parola, del
capitolo sui Transformer, con i token audio al posto delle parole. È la
ricetta dietro sistemi come AudioLM e MusicLM: prima si impara un *alfabeto*
del suono, poi ci si scrive sopra con un modello di linguaggio. Le due
difficoltà (costruire un buon alfabeto che perda poco fedeltà, e modellare
bene sequenze di token lunghissime) sono i due poli attorno a cui ruotano le
sezioni sui codec neurali e sulla generazione.

`````

C'è un'asimmetria onesta da segnalare subito. Il testo *nasce* discreto: le
lettere e le parole sono già simboli, l'alfabeto ce lo dà la lingua. L'audio
no: è un'onda continua, e l'alfabeto sonoro non esiste in natura; va
**costruito**, ed è di per sé un problema di apprendimento difficile. Un
alfabeto troppo povero rende il suono metallico e irriconoscibile; uno troppo
ricco produce sequenze di token sterminate, che nessun modello riesce a
gestire. Trovare il giusto compromesso è precisamente il mestiere dei codec
neurali, ed è la ragione per cui la sezione che li tratta viene prima di
quella sulla generazione: senza un buon alfabeto, non c'è nulla su cui
scrivere.

## Come è organizzato il capitolo

Costruite le fondamenta, il capitolo procede in quattro tappe, dalla
comprensione alla creazione.

- **Classificazione audio**: dare un nome ai suoni. Come si passa dallo
  spettrogramma a un'etichetta o a una lista di tag, dalle prime reti
  convoluzionali (CNN) fino ai Transformer audio, con AudioSet come banco di
  prova.
- **Rappresentazioni auto-supervisionate**: come un modello impara com'è fatto
  il suono *senza che nessuno gli dica mai cosa sta ascoltando*, sfruttando le
  montagne di audio che nessuno ha mai trascritto. È la strada che wav2vec e i
  suoi parenti hanno aperto sulla voce.
- **Codec neurali**, l'alfabeto sonoro di cui abbiamo appena parlato: come una
  rete impara a comprimere l'audio in pochi token e a ricostruirlo, fondendo
  compressione e apprendimento.
- **Generazione audio e musica**: scrivere suono nuovo. Due strade, i modelli di
  linguaggio sui token e la **diffusione**, che è il metodo (lo vedremo nascere
  per le immagini, in un capitolo suo) di partire da rumore puro e ripulirlo un
  passo alla volta finché non ne esce qualcosa.

Un filo, quattro nodi: si parte dall'ascoltare per arrivare a comporre, e in
mezzo c'è sempre la stessa idea (trasformare il suono in qualcosa che una rete
sa maneggiare, che sia un'immagine tempo–frequenza o un alfabeto di token).

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- L'audio è molto più della voce: **musica**, **suoni dell'ambiente**, versi di
  animali. AudioSet {cite}`gemmeke2017audioset` (oltre due milioni di frammenti
  da dieci secondi presi da YouTube, su 527 categorie di suoni) dà l'idea di
  quanto sia grande il mondo che c'è là fuori.
- Cambiano le **domande**: non più (solo) «che cosa ha detto», ma «che suono è
  questo», «quali suoni ci sono in questa registrazione», «quando comincia
  ciascuno», e perfino «fammene sentire uno nuovo».
- Il **punto di partenza** (come un suono diventa numeri, e i numeri
  un'immagine) lo costruisce la prossima sezione, *Dal suono alle feature*, e
  vale per qualsiasi suono: dopo di lei nessuno lo rifà da capo.
- Il **filo conduttore**: se il suono si può scrivere con un alfabeto finito di
  simboli (i **token**), allora la stessa macchina che indovina la parola
  successiva di una frase può indovinare il pezzetto di suono successivo di un
  brano.
- Le quattro tappe: **riconoscere** i suoni, **imparare** dal suono senza
  etichette, costruire l’**alfabeto**, **generare** suono nuovo.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- L'audio è molto più della voce: **musica**, **suoni ambientali**,
  **bioacustica**. AudioSet {cite}`gemmeke2017audioset` (ontologia di 632
  categorie; la raccolta pubblicata conta oltre 2 milioni di clip da YouTube su
  527 classi) dà la scala del problema.
- Cambiano i **compiti**: non più (solo) trascrivere, ma **classificare**,
  **taggare** più eventi insieme, **rilevarne** l'istante, **generare** suono
  nuovo. Il segnale di musica e ambiente ha una struttura diversa da quella del
  parlato.
- Le **feature di base** (campionamento, spettrogramma, scala mel, MFCC) sono
  costruite nella prossima sezione, *Dal suono alle feature*, e valgono per
  qualsiasi suono: sono il punto di partenza comune, e non si ripetono più.
- Il **filo conduttore**: trasformare l'audio in una sequenza di **token
  discreti** rende applicabile tutto l'armamentario dei Transformer; è il
  ponte verso i codec neurali e la generazione.
- Le quattro sezioni: **classificazione**, **rappresentazioni
  auto-supervisionate**, **codec neurali**, **generazione audio e musica**.
```

`````
