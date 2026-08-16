# Un'etichetta per ogni parola: POS tagging e riconoscimento di entità

«La vecchia porta la sbarra». Leggi la frase una prima volta: c'è un'anziana
signora («la vecchia»), e sta trasportando («porta») una sbarra di ferro. Ora
rileggila cambiando i ruoli: c'è una porta malandata («la vecchia porta») che
sbarra il passaggio («la sbarra») a una donna, e quel «la» adesso non è più un
articolo, è lei. Nessun trucco di punteggiatura: le stesse cinque parole, nello
stesso ordine, formano due frasi italiane complete e sensate.

Il bivio è tutto grammaticale, e sta in quattro parole su cinque: «vecchia» può
essere nome o aggettivo, «porta» nome o verbo, «la» articolo o pronome,
«sbarra» nome o verbo. Scegliere una lettura significa, senza accorgersene,
assegnare a ogni parola il suo ruolo nella frase.

Noi lo facciamo in una frazione di secondo. Una macchina deve farlo
*esplicitamente*: scrivere accanto a ogni parola un'etichetta con il suo
ruolo. È il **part-of-speech tagging** (POS, etichettatura delle parti del
discorso), uno dei compiti più antichi del NLP; suo cugino stretto è il
**riconoscimento di entità nominate**, che tutti chiamano con la sigla inglese
**NER**, da *named entity recognition*, e che abbiamo già incontrato nella
panoramica del capitolo. Li raccontiamo insieme perché condividono la forma
(un'etichetta per ogni parola) e la stessa storia: prima i modelli
probabilistici degli anni Novanta, la «seconda stagione» della parabola
storica del capitolo, poi le reti ricorrenti delle sezioni precedenti.

## Il mestiere di ogni parola

A scuola si chiamava analisi grammaticale: articolo, nome, verbo, aggettivo…
Il POS tagging è la stessa cosa, fatta da un algoritmo su milioni di frasi.
Il risultato si scrive attaccando a ogni parola la sua etichetta con una barra,
e per l'esempio ricorrente del libro viene così (una avvertenza prima di
leggerlo: la frase dice «sul», e qui trovate «su» e «il» separati, perché
l'italiano fonde preposizione e articolo in una parola sola e chi analizza li
riapre):

> Il/`DET` gatto/`NOUN` nero/`ADJ` salta/`VERB` su/`ADP` il/`DET` muro/`NOUN`

Le sigle sono abbreviazioni inglesi, e conviene scioglierle subito: `DET` è il
*determiner*, cioè l'articolo; `NOUN` il nome; `ADJ` l'aggettivo; `VERB` il
verbo; `ADP` l’*adposition*, che raccoglie le nostre preposizioni e le
posposizioni di quelle lingue che le mettono dopo il nome invece che prima
(una categoria sola per tutte e due, così l'etichetta vale in ogni lingua).
Le altre dodici sono le stesse che si imparano a scuola più qualche
raffinatezza: pronome, avverbio, nome proprio, ausiliare, numerale,
interiezione, i due tipi di congiunzione, le particelle, la punteggiatura, i
simboli, e un'etichetta di riserva per ciò che non è niente di tutto questo.

Perché il gioco funzioni tra lingue diverse serve però un inventario di
categorie condiviso: è il contributo del progetto **Universal Dependencies**
{cite}`nivre2016universal`, un'impresa collettiva di linguisti che prendono
testi veri e ci scrivono sopra, parola per parola, l'analisi giusta (si dice
che li **annotano**), sempre con gli stessi criteri e nella stessa notazione.
Alla presentazione del 2016 le lingue erano 33, oggi sono oltre
centocinquanta. Il nome parla di «dipendenze» e non di categorie perché il
grosso di quel lavoro riguarda un piano più su, quello della struttura della
frase, che è il tema della prossima sezione; le diciassette etichette sono le
fondamenta su cui quella struttura si appoggia.

`````{tab} Elementare

Pensa alle categorie grammaticali come ai *mestieri* delle parole: il nome
indica cose e persone, il verbo racconta azioni, l'articolo fa strada al nome.
Il punto delicato è che molte parole fanno due mestieri, e cambiano divisa
senza avvisare: «porta» è un oggetto in «la porta cigola» e un'azione in
«Maria porta il pane»; «ancora» è un pezzo di nave se la pronunci *àncora* e
un avverbio se la pronunci *ancóra*. Sulla pagina le due «ancora» sono
identiche: solo le parole intorno rivelano quale hai davanti. Etichettare le
parti del discorso è proprio questo: guardare il contesto e decidere, parola
per parola, quale mestiere è in servizio. I linguisti del progetto Universal
Dependencies hanno stilato una lista di 17 mestieri che funziona per
l'italiano come per il finlandese o il giapponese: una specie di stele di
Rosetta della grammatica.

`````

`````{tab} Superiore

Formalmente il POS tagging è un problema di **etichettatura di sequenze**:
data la frase $w_1, \dots, w_n$, produrre la sequenza di etichette
$t_1, \dots, t_n$, una per token, dalla stessa lunghezza dell'input. È una
struttura più semplice della traduzione vista nella sezione precedente
(niente riordini, niente lunghezze diverse), ma la difficoltà si concentra
nell'ambiguità: le parole ambigue sono una minoranza del vocabolario, però
sono tra le più frequenti, tanto che in un testo inglese corrente oltre la
metà delle occorrenze ammette più di un'etichetta.

Lo standard di riferimento è il tagset **universale** di Universal
Dependencies {cite}`nivre2016universal`, 17 categorie valide per tutte le
lingue del progetto:

| Classi aperte | Classi chiuse | Altro |
|---|---|---|
| `NOUN` nome | `DET` determinante | `PUNCT` punteggiatura |
| `PROPN` nome proprio | `PRON` pronome | `SYM` simbolo |
| `VERB` verbo | `ADP` adposizione | `X` altro |
| `ADJ` aggettivo | `AUX` ausiliare | |
| `ADV` avverbio | `CCONJ` cong. coordinante | |
| `INTJ` interiezione | `SCONJ` cong. subordinante | |
| | `NUM` numerale | |
| | `PART` particella | |

Le classi *aperte* accolgono parole nuove di continuo («googlare» è un `VERB`
recente), quelle *chiuse* quasi mai: nessuno conia nuovi articoli.
Sull'inglese giornalistico i tagger moderni superano il 97% di accuratezza per
token: un numero da leggere con prudenza, come vedremo parlando di
valutazione.

`````

E a che cosa serve, oggi, un'etichetta grammaticale? A tre cose almeno.

Alla **lemmatizzazione**, che abbiamo incontrato nella prima sezione: ricondurre
una parola alla forma con cui la si cerca sul vocabolario, il suo *lemma*. Per
«porta» le forme di dizionario sono due, e per scegliere devi sapere prima se
è il nome (e allora il lemma è *porta*) o il verbo (e allora è *portare*).

Alla **sintesi vocale**, cioè ai programmi che leggono un testo ad alta voce,
il percorso inverso del riconoscimento vocale che incontreremo nel capitolo
dedicato alla voce: un lettore automatico davanti ad «ancora» deve scegliere
tra *àncora* e *ancóra*, e l'accento giusto lo decide la categoria
grammaticale.

E all’**analisi sintattica**: le etichette POS sono i mattoni con cui, nella
prossima sezione, si costruisce l'impalcatura della frase.

## Chi, dove, quando: le entità nominate

Il secondo compito lo abbiamo già visto all'opera nella panoramica del
capitolo: in «Enrico Fermi nacque a Roma nel 1901» un sistema NER etichetta
*Enrico Fermi* come persona, *Roma* come luogo, *1901* come data. Il
**riconoscimento di entità nominate** (la sigla nasce alle *Message
Understanding Conference* degli anni Novanta) cerca nel testo persone, luoghi
e organizzazioni, più date, cifre e importi. Serve ogni volta che da un mucchio
di testo bisogna ricavare delle **schede**: chi è nato dove e quando, quale
azienda ha comprato quale altra, quali farmaci compaiono in una cartella
clinica. È anche il primo passo per anonimizzare un documento, perché per
cancellare i nomi bisogna prima sapere quali parole sono nomi di persona.

A prima vista sembra un problema diverso dal POS tagging: lì un'etichetta per
parola, qui *segmenti* da ritagliare; «Enrico Fermi» è un'entità sola, lunga
due parole. Il trucco che riporta tutto alla forma già nota è lo **schema
BIO**: invece di dire dove comincia e dove finisce un segmento, si dà
un'etichetta a ogni parola, e l'etichetta dice se lì un segmento *comincia*,
se lo *continua*, o se lì fuori non c'è niente. Le tre lettere sono le
iniziali inglesi di quelle tre parole (*begin*, *inside*, *outside*). Lo schema
nasce nel 1995 con il lavoro di Lance Ramshaw e Mitchell Marcus sugli spezzoni
di frase, e nella versione che si usa oggi il segnale di «comincia» si mette in
testa a **ogni** entità, anche quando non ce ne sarebbe bisogno per distinguerla
dalla precedente: costa un'etichetta in più e in cambio rende ogni parola
leggibile per conto suo.

`````{tab} Elementare

Immagina di lavorare con degli evidenziatori colorati: giallo per le persone,
azzurro per i luoghi, verde per le date. Il problema è dettare al telefono,
parola per parola, dove passa l'evidenziatore, e con quale regola? Ne bastano
tre per parola: «qui **comincio** un'evidenziatura gialla», «qui la
**continuo**», «qui la penna è **sollevata**». Sulla frase di Fermi: *Enrico*
= comincio-giallo, *Fermi* = continuo, *nacque, a* = penna su, *Roma* =
comincio-azzurro, *nel* = penna su, *1901* = comincio-verde. La distinzione
tra «comincio» e «continuo» sembra pignola ma è preziosa: in «il faccia a
faccia Mattarella Macron», due «comincio-giallo» di fila dicono che le persone
sono *due*; un «comincio» seguito da un «continuo» direbbe che è una sola, un
improbabile signor Mattarella Macron.

`````

`````{tab} Superiore

Lo schema BIO trasforma l'estrazione di segmenti in etichettatura per token.
Per ogni tipo di entità $X$ si definiscono due etichette, `B-X` (*begin*,
primo token del segmento) e `I-X` (*inside*, continuazione), più un'unica
etichetta `O` (*outside*) per i token fuori da ogni entità: con $K$ tipi, il
tagset conta $2K + 1$ etichette. La frase di Fermi diventa:

> Enrico/`B-PER` Fermi/`I-PER` nacque/`O` a/`O` Roma/`B-LOC` nel/`O`
> 1901/`B-DATE`

La marca `B` è ciò che rende lo schema invertibile: senza di essa due entità
adiacenti dello stesso tipo si fonderebbero in una. La sequenza `B-PER B-PER`
codifica due persone consecutive; `B-PER I-PER` una sola entità di due token.

Vale la pena distinguere due varianti che vengono spesso confuse, perché la
differenza è proprio su quel punto. Nello schema originale del 1995 (oggi
chiamato **IOB1**) la `B` era parsimoniosa: compariva **solo** quando un
segmento ne seguiva immediatamente un altro dello stesso tipo, cioè solo dove
serviva davvero a separarli. Sotto IOB1 la frase di Fermi si etichetta
`I-PER I-PER`, non `B-PER I-PER`, e la `B` fa esattamente e soltanto il lavoro
di garantire l'invertibilità. La variante che ha vinto, **IOB2** (la introduce
Adwait Ratnaparkhi nel 1998; il confronto sistematico fra le varianti in
circolazione è di Tjong Kim Sang e Veenstra, 1999), mette la `B` in testa a
ogni segmento senza eccezioni: costa un'etichetta in più dove non servirebbe,
e in cambio rende l'etichetta di un token indipendente da ciò che lo precede,
il che semplifica sia l'annotazione sia l'apprendimento. È quella usata nello
schema qui sopra e in tutti i corpora moderni. Esistono varianti
più ricche (BIOES aggiunge etichette esplicite di fine
segmento e di entità a token singolo), ma l'idea non cambia: una volta ridotto
il NER a un'etichetta per token, *qualunque* modello di etichettatura di
sequenze (HMM, CRF, BiLSTM, Transformer) lo può affrontare.

`````

## La grammatica dietro la tenda: gli HMM

Come si insegna a una macchina a etichettare? La risposta classica, cuore
della stagione statistica del NLP, è un modello dal nome intimidatorio e
dall'idea limpida: lo **Hidden Markov Model** (HMM, modello di Markov
nascosto). Il nome si scioglie pezzo per pezzo. *Markov* è il matematico russo
che due sezioni fa contava le lettere dell’*Onegin*, e la parola richiama il
suo patto: quello che succede adesso dipende solo da quello che è successo
subito prima. *Model*, modello, perché è appunto una descrizione semplificata
di come nasce una frase. E *hidden*, nascosto, che è l'aggettivo importante: le
categorie grammaticali non si vedono mai, perché sulla pagina ci sono soltanto
parole, eppure sono loro a governare quali parole compaiono e in che ordine.

`````{tab} Elementare

Immagina una recita dietro una tenda: tu, in platea, non vedi gli attori,
senti solo le battute. Gli attori sono le *categorie grammaticali*, che si
passano la scena secondo abitudini precise (dopo l'ARTICOLO entra quasi sempre
il NOME, diciamo 7 volte su 10, e raramente il VERBO), e ognuna ha il suo
copione di parole tipiche (quando è in scena l'ARTICOLO senti «la», «il»,
«un»…). Un HMM è questo teatro: due libretti di abitudini (*chi passa la scena
a chi* e *chi dice che cosa*) imparati contando su migliaia di frasi già
etichettate a mano. Etichettare una frase nuova è un ragionamento da
detective: sentite le battute «la porta cigola», qual è la sfilata di attori
dietro la tenda che le spiega meglio? Certezze non ce ne sono («porta»
potrebbe dirla il NOME o il VERBO) ma puoi calcolare quale storia è più
probabile, ed è quella che scrivi.

`````

`````{tab} Superiore

Un HMM per il tagging ha come **stati nascosti** le etichette
$t_1, \dots, t_n$ e come **osservazioni** le parole $w_1, \dots, w_n$. Due
assunzioni lo definiscono: ogni etichetta dipende solo dalla precedente
(catena di Markov del primo ordine) e ogni parola dipende solo dalla
propria etichetta. La probabilità congiunta si fattorizza allora in

$$
P(t_1, \dots, t_n,\; w_1, \dots, w_n)
= \prod_{i=1}^{n} P(t_i \mid t_{i-1})\, P(w_i \mid t_i),
$$

dove $P(t_i \mid t_{i-1})$ sono le probabilità di **transizione** tra
etichette (con $t_0$ simbolo convenzionale di inizio frase) e
$P(w_i \mid t_i)$ le probabilità di **emissione** delle parole; entrambe si
stimano contando su un corpus annotato. Il tagging è la ricerca della
sequenza di stati più probabile date le parole,

$$
\hat{t}_{1:n} = \arg\max_{t_{1:n}} P(t_{1:n} \mid w_{1:n})
= \arg\max_{t_{1:n}} P(t_{1:n}, w_{1:n}),
$$

dove la seconda uguaglianza segue dalla regola di Bayes: il denominatore
$P(w_{1:n})$ non dipende dalle etichette. È un modello **generativo**:
descrive come etichette e parole vengono prodotte insieme, e la lettura
d'obbligo resta il tutorial di Rabiner {cite}`rabiner1989tutorial`.

`````

Questa macchina, del resto, non è nata per la grammatica. Lo scritto che l'ha
resa popolare è del 1989, lo firma Lawrence Rabiner {cite}`rabiner1989tutorial`
e parla di riconoscimento del parlato: gli HMM hanno retto la trascrizione
automatica per trent'anni, e li ritroveremo nel capitolo dedicato.

Cambia solo il cast della recita, e vale la pena vedere come. Dietro la tenda
non ci sono più le categorie grammaticali, ci sono i **suoni elementari** della
lingua, quelli che distinguono «pane» da «cane»: loro sono gli attori. E in
platea non arrivano parole, arriva il suono, che un programma taglia in
fettine da pochi millesimi di secondo e riduce a un pugno di numeri per fetta,
tipo «quanta energia c'è sui toni bassi, quanta sugli alti»: quelle sono le
battute. Il resto è identico: si sente una fila di battute, si cerca la sfilata
di attori che le spiega meglio, e a trovarla è lo stesso navigatore che vedremo
fra poco.

## Viterbi, o l'arte di non provarle tutte

Resta il problema pratico: *trovare* la sequenza di etichette più probabile.
Contiamo quante sono. Per la prima parola posso scegliere fra 17 categorie;
per ciascuna di quelle scelte, la seconda parola me ne offre altre 17, e siamo
già a $17 \times 17$; con venti parole le combinazioni sono $17^{20}$, cioè
circa quattro milioni di miliardi di miliardi. Provarle tutte è fuori
discussione: non c'è computer che finisca.

La salvezza sta in una proprietà che il modello ha per costruzione: **ogni
etichetta dipende solo da quella immediatamente precedente**, e non da tutte
quelle prima ancora. È la stessa regola del patto di Markov, ed è il motivo per
cui questi modelli si dicono «a catena»: come in una catena, ogni anello tocca
solo il precedente e il successivo.

Vale la pena dire in che senso quella proprietà salva. Arrivato alla parola 12,
non ho bisogno di ricordare tutta la storia di come ci sono arrivato: mi basta
sapere, per ciascuna delle 17 categorie possibili, qual era il modo migliore di
arrivarci alla parola 11. Tutto il resto si può buttare, perché non
influenzerà nulla di ciò che viene dopo. Questo modo di procedere (calcolare
una volta sola ogni pezzo che servirà più volte, e tenerselo da parte invece di
rifarlo) si chiama **programmazione dinamica**, e nel libro torna spesso: la
griglia della distanza di edit, nella prima sezione, era la stessa idea.

L'algoritmo che la applica qui porta il nome di Andrew Viterbi
{cite}`viterbi1967error`, nato Andrea a Bergamo nel 1935 ed emigrato bambino
negli Stati Uniti, che lo propose nel 1967 non per la grammatica ma per
decifrare segnali arrivati storti lungo un canale disturbato: lo stesso
algoritmo ha poi viaggiato dentro i telefoni cellulari di mezzo mondo.

Facciamo i conti fino in fondo su un modello giocattolo: tre categorie
(`DET`, `NOME`, `VERBO`) e la frase «la porta cigola», dove «porta» ha la
stessa doppiezza dell'aggancio di questa sezione. I numeri delle due tabelle
qui sotto sono **inventati per l'esempio**, scelti tondi perché i conti si
possano rifare a mente: in un sistema vero verrebbero dai conteggi su un
corpus già etichettato, come si è detto poco fa.

Ecco il primo dei due libretti, quello che dice **chi passa la scena a chi**.
Si legge riga per riga: la riga «inizio frase» dice che sei frasi su dieci
cominciano con un articolo, tre con un nome e una con un verbo; la riga `DET`
dice che dopo un articolo arriva un nome sette volte su dieci, un verbo due e
un altro articolo una.

| da ↓ verso → | `DET` | `NOME` | `VERBO` |
|---|---|---|---|
| inizio frase | 0,6 | 0,3 | 0,1 |
| `DET` | 0,1 | 0,7 | 0,2 |
| `NOME` | 0,2 | 0,3 | 0,5 |
| `VERBO` | 0,4 | 0,4 | 0,2 |

E il secondo libretto, quello che dice **chi pronuncia che cosa**. Si chiama
delle *emissioni*, perché quelle sono le parole che ciascun attore emette
quando è in scena:

| chi è in scena ↓ | dice «la» | dice «porta» | dice «cigola» |
|---|---|---|---|
| `DET` | 0,5 | 0 | 0 |
| `NOME` | 0 | 0,2 | 0 |
| `VERBO` | 0 | 0,2 | 0,3 |

Nessuna di queste righe somma a uno, e non è un errore: il resto della
probabilità va a tutte le altre parole della lingua, che in questo esempio non
compaiono. E siccome siamo in un modello giocattolo con tre sole categorie, il
pronome non c'è: qui «la» la può dire solo l'articolo, anche se in italiano
vero, come si è visto in apertura di sezione, potrebbe essere un pronome.

Nota infine il punto delicato: la parola «porta», da sola, *non decide*, perché
il nome e il verbo la pronunciano con la stessa frequenza, 0,2 contro 0,2.
{numref}`fig-viterbi-traliccio` mostra il **traliccio** (*trellis*): una
colonna per parola, una casella per categoria, e tutti i cammini che
l'algoritmo valuta.

```{figure} ../figures/viterbi-traliccio.svg
:name: fig-viterbi-traliccio
:alt: Traliccio di Viterbi per la frase «la porta cigola» con tre stati DET, NOME e VERBO per colonna. Il cammino ottimo DET, NOME, VERBO è in terracotta con le probabilità parziali 0,30, 0,042 e 0,0063; il cammino alternativo che passa da VERBO su «porta» è in grigio; i cammini a probabilità zero sono tratteggiati.
:width: 100%

Il traliccio di Viterbi su «la porta cigola»: in ogni casella sopravvive
solo il migliore dei cammini che vi arrivano, e alla fine si risale
all'indietro lungo la strada in terracotta.
```

`````{tab} Elementare

Pensa a un navigatore che deve attraversare tre incroci (le tre parole),
scegliendo a ogni incrocio una corsia (la categoria). Il suo segreto è duplice.
Primo: a ogni incrocio, per ogni corsia, conserva **solo il modo migliore di
arrivarci** e butta via gli altri, perché se due strade sbucano nella stessa
corsia dello stesso incrocio, da lì in avanti hanno davanti esattamente le
stesse possibilità, e la più lenta non potrà mai più recuperare. Secondo: ogni
volta che tiene una strada, si segna su un foglietto **da dove veniva**, così
alla fine potrà ricostruire il percorso a ritroso.

Seguiamolo sui numeri della figura.

**«la»**: solo `DET` sa dire «la», quindi c'è una sola casella viva. Vale
$0{,}6 \times 0{,}5 = 0{,}30$, cioè la probabilità che la frase cominci con un
articolo (0,6) per la probabilità che quell'articolo sia proprio «la» (0,5).

**«porta»**: due caselle possibili. Arrivare a `NOME` vale $0{,}30 \times 0{,}7
\times 0{,}2 = 0{,}042$, cioè quanto valeva la casella di prima, per la
probabilità che dopo un articolo venga un nome, per la probabilità che quel
nome sia «porta». Arrivare a `VERBO` vale con lo stesso conto $0{,}30 \times
0{,}2 \times 0{,}2 = 0{,}012$. La parola era in perfetto pareggio (0,2 e 0,2):
a sbilanciare è stata la grammatica, quello 0,7 contro 0,2; dopo un articolo ci
si aspetta un nome.

**«cigola»**: solo `VERBO` può dirla, ma ci si arriva da due strade. Qui
conviene fare il confronto prima e la moltiplicazione finale dopo, perché il
fattore che manca è lo stesso per tutte e due e non cambia la classifica: da
`NOME` si arriva con $0{,}042 \times 0{,}5 = 0{,}021$, da `VERBO` con $0{,}012
\times 0{,}2 = 0{,}0024$. Vince la prima; adesso si moltiplica per la
probabilità che il verbo dica «cigola» ($\times\, 0{,}3$) e la casella chiude a
$0{,}0063$. Ora il navigatore rilegge i foglietti all'indietro: `VERBO` ←
`NOME` ← `DET`. Ecco l'etichettatura: *la*/articolo *porta*/nome
*cigola*/verbo.

Qui i cammini erano una manciata; con 17 categorie e 20 parole sarebbero quei
quattro milioni di miliardi di miliardi di poco fa, ma gli *incroci* restano
appena 17 × 20 = 340. Il
navigatore fa un pugno di moltiplicazioni per incrocio e trova comunque,
garantito, il percorso migliore in assoluto.

`````

`````{tab} Superiore

Definiamo $v_t(s)$ come la probabilità del miglior cammino che arriva allo
stato $s$ dopo aver generato le prime $t$ parole. L'algoritmo di Viterbi la
calcola per ricorrenza:

$$
v_1(s) = \pi_s \, P(w_1 \mid s),
\qquad
v_t(s) = \max_{s'} \big[\, v_{t-1}(s')\, P(s \mid s') \,\big]\, P(w_t \mid s),
$$

dove $\pi_s$ è la probabilità iniziale dello stato $s$ e $s'$ scorre su
tutti gli stati al passo precedente; un **retropuntatore**
$\psi_t(s) = \arg\max_{s'} v_{t-1}(s')\, P(s \mid s')$ memorizza da dove
proviene il massimo. Sul modello giocattolo la tabella dei massimi è:

| stato | «la» | «porta» | «cigola» |
|---|---|---|---|
| `DET` | **0,30** | 0 | 0 |
| `NOME` | 0 | **0,042** ← `DET` | 0 |
| `VERBO` | 0 | 0,012 ← `DET` | **0,0063** ← `NOME` |

Al termine si prende lo stato finale con $v_n$ massimo (qui `VERBO`, con
$0{,}0063$) e si segue $\psi$ a ritroso: `DET` → `NOME` → `VERBO`. Il cammino
alternativo completo `DET` → `VERBO` → `VERBO` vale
$0{,}012 \times 0{,}2 \times 0{,}3 = 0{,}00072$: quasi dieci volte meno. Il
costo è $O(n\,T^2)$ (per ogni parola, per ogni stato, un massimo su $T$
predecessori) contro gli $O(T^n)$ cammini della forza bruta: con $T = 17$ e
$n = 20$, poche migliaia di operazioni al posto di $10^{24}$, e con la
garanzia dell'ottimo globale; a differenza della *beam search* della sezione
precedente, che è un'euristica. In pratica si lavora con i logaritmi, sommando
invece di moltiplicare, per evitare l'underflow.

`````

Una riga di storia successiva. Agli HMM sono succeduti i **Conditional Random
Field** (CRF) {cite}`lafferty2001conditional`, che rinunciano a raccontare come
parole ed etichette nascano insieme e si addestrano soltanto a scegliere
l'etichetta giusta. È la stessa distinzione dei due periti della sezione sulla
classificazione, quello che studia lo stile di ciascun pittore e quello che
impara solo i dettagli che li distinguono: **generativo** il primo,
**discriminativo** il secondo, e qui applicata alle sequenze invece che ai
documenti. Il guadagno è che un CRF può guardare indizi che a un HMM sfuggono,
per esempio la maiuscola iniziale, le ultime tre lettere della parola, la
presenza di un trattino; e per oltre un decennio sono stati il modo migliore di
fare NER. Per trovare il percorso migliore, però, chiamano sempre Viterbi.

## La via neurale: una BiLSTM per etichettare

E le reti ricorrenti? Nella sezione sulla traduzione abbiamo stabilito una
regola: la lettura **bidirezionale** vale solo per *capire* un testo che
esiste già tutto intero, non per generarlo. L'etichettatura è il caso ideale:
la frase è lì, completa, e per decidere l'etichetta di «porta» servono tanto
le parole prima quanto quelle dopo («la porta **cigola**» contro «la porta **a
scuola**»). Un etichettatore neurale minimo (in gergo lo si chiama *tagger*,
dall'inglese *tag*, cartellino) è quindi fatto di tre pezzi, tutti già
incontrati: si trasforma ogni parola nella sua fila di numeri (l'embedding), la
si dà in pasto a una LSTM che legge nei due sensi, e in cima si mette una
bilancia che per **ogni** parola assegna un punteggio a ciascuna delle 17
etichette possibili. Vince l'etichetta col punteggio più alto.

```python
import torch
from torch import nn

class TaggerBiLSTM(nn.Module):
    def __init__(self, vocab=10000, num_tag=17, dim=64, hidden=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim)   # parola -> vettore
        self.lstm = nn.LSTM(dim, hidden, batch_first=True,
                            bidirectional=True)     # legge nei due sensi
        self.out = nn.Linear(2 * hidden, num_tag)   # un logit per etichetta

    def forward(self, x):          # x: (batch, lunghezza), indici di parole
        e = self.embedding(x)      # (batch, lunghezza, dim)
        h, _ = self.lstm(e)        # (batch, lunghezza, 2*hidden)
        return self.out(h)         # logit per OGNI parola, non solo l'ultima
```

Confrontalo con il classificatore di sentiment della sezione sui modelli di
sequenza: là si teneva solo l'ultimo stato (`h[:, -1]`), un'etichetta per
frase; qui si tengono tutti, un'etichetta per parola. La misura dell'errore è
quella di sempre, la cross-entropia (quanto la previsione si discosta
dall'etichetta giusta), applicata però parola per parola, e con un
accorgimento: le frasi di un gruppo hanno lunghezze diverse e si pareggiano
riempiendo le più corte con caselle vuote (il *padding*), che vanno escluse dal
conto, altrimenti la rete si metterebbe a imparare il vuoto. È a questo che
serve il `-100` nel codice qui sotto: è la marca convenzionale che dice
«questa casella non conta».

```{code-block} python
:class: pt-non-eseguibile

modello = TaggerBiLSTM()
perdita = nn.CrossEntropyLoss(ignore_index=-100)  # -100 = padding da ignorare

# frasi: (batch, lunghezza), indici di parole; tag: stessa forma, -100 sul padding
logits = modello(frasi)                    # (batch, lunghezza, 17)
loss = perdita(logits.reshape(-1, 17),     # una riga per token
               tag.reshape(-1))            # un'etichetta per token
loss.backward()                            # poi optimizer.step(), come sempre
```

Questa ricetta ha però un buco: decide ogni parola per conto suo, senza
guardare che cosa ha deciso per quella prima. Può quindi scrivere un
«continuo l'evidenziatura» subito dopo un «penna sollevata», che non vuol dire
niente. Nei sistemi di punta si aggiunge allora in cima uno strato CRF, che
rimette in gioco le transizioni fra etichette e sa che certe successioni sono
impossibili.

Oggi però lo standard del NER è un'altra strada: prendere un modello già
addestrato su montagne di testo, come BERT {cite}`devlin2019bert`, appoggiargli
sopra la stessa testa che assegna un punteggio a ogni etichetta, e proseguire
l'addestramento per pochi giri sul compito specifico. Questa seconda fase corta
si chiama **fine-tuning**, «rifinitura»: non si riparte da zero, si parte da un
modello che la lingua la sa già e gli si insegna soltanto il mestiere nuovo,
con una frazione dei dati e del tempo. Ne parleremo nel capitolo sui
Transformer, dove la lettura nei due sensi, che qui abbiamo dovuto costruire a
mano con due reti affiancate, viene da sé.

## Misurare bene: token o entità?

Un'ultima questione, meno contabile di quanto sembri: come si dà il voto a
un etichettatore? La risposta giusta è diversa per i due compiti, e la
differenza insegna qualcosa.

`````{tab} Elementare

Per le parti del discorso il voto naturale funziona: quante parole hanno
ricevuto l'etichetta giusta, su cento. Ma attenzione alle percentuali gonfiate.
Prova a immaginare il sistema più stupido possibile: per ogni parola guarda
qual è il mestiere che quella parola fa più spesso nei testi già etichettati,
e scrive sempre quello, senza mai guardare il contesto. Un sistema così, che
non ha capito niente di niente, sull'inglese dei giornali azzecca già **92 parole su cento**, perché tantissime
parole sono facili: «il» è quasi sempre un articolo, «velocemente» quasi sempre
un avverbio. Le parole ambigue sono meno numerose, ma sono quelle che si usano
di più, e in un testo vero coprono più della metà delle parole scritte.
I sistemi seri stanno oltre il **97**. Ecco perché quei numeri vanno letti
sapendo da dove si parte: fra il 92 e il 97 c'è tutto il lavoro, e ci sono tutte
le parole ambigue, cioè le uniche su cui valga la pena discutere.

Per le entità quel voto diventa una trappola. Prendi un testo di 100 parole
che contiene una sola entità, «Enrico Fermi». Un sistema pigro che non
evidenzia *niente* azzecca 98 parole su 100: 98%, e non ha trovato nulla! E un
sistema che evidenzia solo «Enrico», lasciando fuori «Fermi», ha prodotto
un'entità sbagliata: mezza persona non serve a nessuno. Per questo il NER si
giudica a evidenziature intere (vale solo il segmento completo, del colore
giusto) e con due domande: di quello che hai evidenziato, quanto era giusto? E
di quello che andava evidenziato, quanto ne hai trovato? Sono la precisione e
il richiamo che abbiamo incontrato nel capitolo sul machine learning e usato
nella sezione sulla classificazione, dove si chiamavano con i loro nomi
inglesi, *precision* e *recall*: sono la stessa identica coppia di domande. Il
voto unico $F_1$ le riunisce con la media severa già vista là: si moltiplicano
i due voti, si raddoppia il prodotto, e lo si divide per la somma dei due voti.

Prova con un testo che contiene dieci entità e un sistema che ne evidenzia una
sola, azzeccandola. Primo voto: $1{,}0$, perché tutto quello che ha segnalato
era giusto. Secondo voto: $0{,}1$, perché di dieci ne ha trovata una. La media
di scuola gli darebbe un onorevole $0{,}55$. Con la nostra: prodotto $0{,}10$,
raddoppiato $0{,}20$, diviso la somma $1{,}1$, fa $0{,}18$. E ha ragione lei.

`````

`````{tab} Superiore

Per il POS tagging la metrica è l’**accuratezza per token**,
$\text{acc} = \frac{\#\{i : \hat{t}_i = t_i\}}{n}$. Va letta contro una base
di confronto onesta: il baseline «assegna a ogni parola la sua etichetta più
frequente nel corpus» supera già il 92% sull'inglese giornalistico, quindi lo
spazio reale di miglioramento (dal 92% al 97% e oltre) sta tutto nelle
occorrenze ambigue, le più difficili.

Per il NER si usano precisione, richiamo e F1 **a livello di entità**, con
il criterio dell’*exact match* reso standard dalle campagne di valutazione
CoNLL dei primi anni Duemila: un'entità predetta conta come corretta solo
se coincidono sia i confini del segmento sia il tipo. Dette $C$ le entità
corrette, $\hat{E}$ quelle predette ed $E$ quelle di riferimento:

$$
P = \frac{|C|}{|\hat{E}|},
\qquad
R = \frac{|C|}{|E|},
\qquad
F_1 = \frac{2PR}{P + R},
$$

dove $P$ (precisione) misura quanto ci si può fidare di ciò che il sistema
estrae e $R$ (richiamo) quanta parte del dovuto viene trovata. La severità
dell'exact match è motivata dall'uso a valle: un'entità dai confini
sbagliati inquina qualunque base di conoscenza la riceva. L'accuratezza per
token, dominata dall'etichetta `O`, qui non discrimina nulla: il sistema
che predice sempre `O` ne uscirebbe con punteggi altissimi e utilità zero.

`````

Chiudiamo dove avevamo aperto. Davanti a «La vecchia porta la sbarra» un
tagger sceglie la lettura più probabile secondo la sua esperienza: quasi
certamente la signora con la sbarra, perché le statistiche della lingua
pendono da quella parte. Ma sapere che «porta» è un verbo non dice ancora *chi
fa che cosa a chi*: per questo bisogna salire di un piano, dalle etichette
alla struttura della frase. È l'analisi sintattica, tema della prossima
sezione, e i suoi mattoni sono esattamente le etichette POS che abbiamo
imparato a mettere.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Il **POS tagging** dà a ogni parola il suo mestiere nella frase (nome,
  verbo, articolo): la lista condivisa fra le lingue è quella dei 17 mestieri
  di Universal Dependencies. Le parole con due mestieri («porta», «ancora»)
  sono una minoranza del vocabolario ma tornano di continuo nei testi, e a
  decidere quale sia in servizio è sempre il contesto.
- Il **NER** cerca persone, luoghi, organizzazioni e date. Lo **schema BIO** è
  il modo di dettare al telefono dove passa l'evidenziatore, con tre soli
  segnali per parola (qui comincio, qui continuo, qui la penna è sollevata):
  così due persone di fila restano due persone e non diventano una sola.
- Lo **HMM** è la recita dietro la tenda: le parole sono le battute che senti,
  le categorie grammaticali gli attori che non vedi, e si passano la scena
  secondo abitudini fisse, ciascuno con il proprio copione di parole tipiche.
  I due libretti di abitudini si imparano contando su frasi già etichettate a
  mano. La stessa macchina, con i suoni al posto delle categorie, ha retto il
  riconoscimento vocale per trent'anni.
- **Viterbi** è il navigatore che a ogni incrocio, per ogni corsia, conserva
  solo il modo migliore di arrivarci e butta via gli altri: invece di provare
  miliardi di percorsi ne visita poche centinaia di caselle, e trova comunque
  il percorso migliore in assoluto, garantito.
- Alla recita dietro la tenda sono poi succeduti metodi che non raccontano più
  come parole ed etichette nascano insieme: si allenano soltanto a scegliere
  l'etichetta giusta, e possono guardare indizi che alla recita sfuggono (la
  maiuscola iniziale, la fine della parola, un trattino). Per trovare il
  percorso migliore, però, chiamano ancora il navigatore.
- Il tagger neurale legge la frase nei due sensi e produce un'etichetta per
  ogni parola, non una per l'intera frase; leggere anche all'indietro è lecito
  perché il testo è già lì tutto intero. Oggi il NER migliore si ottiene
  rifinendo un modello già addestrato (BERT).
- Come si dà il voto: per le parti del discorso si contano le parole
  etichettate bene, ma il numero va letto sapendo da dove si parte. Un sistema
  che dà a ogni parola la sua etichetta più frequente, senza guardare il
  contesto, azzecca già 92 parole su cento; i sistemi seri stanno oltre 97, e
  quei cinque punti sono tutto il mestiere. Per le entità si giudica a
  evidenziature intere, confini e colore compresi, perché mezza persona non
  serve a nessuno.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Il **POS tagging** assegna a ogni parola la sua categoria grammaticale:
  lo standard è il tagset **universale** a 17 categorie di Universal
  Dependencies. Le parole ambigue («porta», «ancora») sono poche nel
  vocabolario ma frequentissime nei testi: decide il contesto.
- Il **NER** trova persone, luoghi, organizzazioni e date; lo **schema
  BIO** (`B-X`, `I-X`, `O`) lo trasforma in un'etichetta per token e tiene
  distinte le entità adiacenti.
- Lo **HMM** è un modello generativo con stati nascosti (le etichette) e
  osservazioni (le parole): $P(t,w) = \prod_i P(t_i \mid t_{i-1}) P(w_i \mid t_i)$;
  transizioni ed emissioni si contano su un corpus annotato. La stessa
  macchina, con i suoni al posto delle etichette, regge l'ASR storico.
- L'algoritmo di **Viterbi** trova la sequenza di stati ottima con la
  programmazione dinamica sul traliccio: $O(n\,T^2)$ invece di $O(T^n)$,
  tenendo in ogni casella solo il miglior cammino in arrivo. I **CRF** sono
  la variante discriminativa.
- Il tagger neurale è una **BiLSTM** con testa lineare per token e
  cross-entropia per token: la bidirezionalità è legittima perché il testo
  è tutto disponibile. Oggi il NER di punta è fine-tuning di BERT.
- Valutazione: **accuratezza per token** per il POS (con baseline già oltre il
  92%), **F1 a livello di entità** con exact match per il NER; perché mezza
  entità è un'entità sbagliata.
```
`````
