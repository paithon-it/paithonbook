# Mondi in miniatura: imparare sognando

C'è un esperimento, nel 2018, che sembra uscito da un racconto più che da un
laboratorio di machine learning. David Ha e Jürgen Schmidhuber prendono un
livello di *Doom* (lo storico sparatutto) in cui bisogna schivare palle di
fuoco, e ci allenano un agente che, durante l'allenamento, il gioco vero non lo
tocca mai. L'ordine delle cose è questo: prima si raccolgono migliaia di
partite giocate premendo i tasti a caso; da quelle partite due reti neurali si
costruiscono una copia compressa e approssimativa del gioco; e l'agente si
allena esclusivamente dentro quella copia, nel proprio «sogno», l'hanno
chiamato proprio così. Riportato nel gioco autentico, schiva le palle di fuoco
ben oltre la soglia che definisce il livello «risolto» {cite}`ha2018world`.

L'articolo ha un titolo di due parole, *World Models* (presentato a NeurIPS
2018 come *Recurrent World Models Facilitate Policy Evolution*) e contiene un
secondo primato: sul gioco di guida *CarRacing-v0*, una pista vista dall'alto
generata a caso a ogni partita, lo stesso schema è il primo sistema dichiarato
in grado di *risolvere* il compito: 906 punti di media su 100 piste, contro la
soglia richiesta di 900. I punti sono quelli che il gioco stesso assegna (più
strada percorsa, meno tempo speso), e i 900 non li hanno scelti gli autori
dell'esperimento: la soglia arriva insieme all'ambiente di gioco, ed è quella
che tutti adoperano proprio perché i risultati si possano confrontare.
Conviene tenerlo a mente ogni volta che si legge «risolto» accanto a un
numero: da qualche parte c'è qualcuno che ha deciso dove mettere l'asticella,
e «risolto» vuol dire soltanto «al di sopra di quell'asticella lì».

L'idea di fondo è antica e molto umana. Quando attraversi la strada non
ragioni sui fotoni che colpiscono la retina: consulti un modello mentale del
traffico («se quell'auto mantiene la velocità, tra tre secondi è qui») e provi
le azioni *nel modello* prima che nel mondo. Un **world model** è questo: una
copia interna, compressa e imparata, dell'ambiente, dentro cui pensare costa
poco e sbagliare non fa male. In questa sezione smontiamo la ricetta di Ha e
Schmidhuber, seguiamo la discendenza fino a DreamerV3 e ai diamanti di
Minecraft, e ricostruiamo i tre moduli in PyTorch.

## Tre lettere per un pilota: V, M e C

La ricetta ha tre ingredienti dai nomi minimalisti: **V** come *visione*,
**M** come *memoria*, **C** come *controller*. V comprime ogni fotogramma in
un piccolo codice; M è una **rete ricorrente** (in sigla RNN: una rete che
legge un passo alla volta portandosi dietro un riassunto di tutto quel che ha
già visto, e quel riassunto, nel disegno qui sotto, si chiama $\mathbf{h}$) e impara
come quel codice evolve in risposta alle azioni; la variante di rete ricorrente
che Ha e Schmidhuber adoperano si chiama **LSTM**, ed è il nome che si legge nel
disegno. C, che dei tre è di gran lunga il più piccolo, legge codice e memoria e
decide. Un avviso sui numeri che seguono: sono quelli del gioco di guida, che è
il più comodo da raccontare; nell'esperimento del sogno, che è l'altro, le
taglie cambiano, e a tempo debito lo diremo. La
{numref}`fig-world-model-vmc` mostra il giro completo: l'azione di C torna
all'ambiente, che produce il fotogramma successivo. E mostra l'anello
tratteggiato che rende speciale l'architettura: M può alimentare se stesso,
sostituendosi all'ambiente. È il circuito del sogno, e ci arriviamo tra poco.

```{figure} ../figures/world-model-vmc.svg
:name: fig-world-model-vmc
:alt: "Pipeline del world model: l'ambiente produce un fotogramma che V comprime in un codice z di 32 numeri; M, una rete ricorrente, predice il prossimo codice; C, un controller lineare da 867 parametri, sceglie l'azione che torna all'ambiente. Un anello tratteggiato sopra M indica il sogno, in cui la predizione di M rientra come suo input."
:width: 100%

I tre moduli di Ha e Schmidhuber: nel gioco vero il ciclo passa
dall'ambiente; nel sogno l'anello tratteggiato lo sostituisce.
```

### V come Visione: il mondo in trentadue numeri

Un fotogramma di *CarRacing*, ridotto a $64 \times 64$ pixel, sono 4.096
puntini; ma ogni puntino è colorato, e per dire un colore servono tre numeri
(quanto rosso, quanto verde, quanto blu), quindi il fotogramma sono
$64 \times 64 \times 3 = 12\,288$ numeri. Troppi, e quasi tutti ridondanti:
alla guida non servono i singoli fili d'erba, serve sapere dove curva la
strada e dove sta l'auto. V è una rete addestrata a spremere ogni fotogramma in
un codice di appena 32 numeri, quasi quattrocento volte meno; il suo nome
tecnico è **autoencoder variazionale**, in sigla VAE, ed è la macchina che il
{doc}`capitolo sui modelli latenti </ModelliLatenti/overview>` deriva per intero {cite}`kingma2014auto`. Quel codice si
chiama $\mathbf{z}$, e
la lettera è soltanto un nome (come la $x$ dell'incognita a scuola): da qui in
avanti «$\mathbf{z}$» vuol dire «il riassunto in 32 numeri di quel che si vede adesso».
Per fare questo mestiere V si porta dietro circa 4,3 milioni di numeri
imparati, i suoi **parametri**: è di gran lunga il più pesante dei tre moduli.

`````{tab} Elementare

Descrivi la schermata di gioco a un amico al telefono. Non
gli detti i 4.096 puntini uno per uno, ciascuno con i suoi tre numeri di
colore: dici «curva a sinistra, auto
al centro, erba sui bordi» (poche informazioni, quelle giuste). Il VAE fa lo
stesso, ma nessuno gli ha suggerito *quali* informazioni tenere: le ha scelte
da solo, perché il suo allenamento è un gioco di andata e ritorno (comprimi il
fotogramma in 32 numeri, poi prova a ridisegnarlo dal solo riassunto). Se il
disegno somiglia all'originale, il riassunto conteneva l'essenziale; se non
somiglia, quei 32 numeri vanno usati meglio. Come al telefono: se dalla tua
descrizione l'amico disegna una scena quasi uguale, la descrizione era buona.

Nell'andata e ritorno c'è una seconda regola. La stessa schermata, descritta
due volte, non viene mai con le stesse parole: «curva a sinistra» oggi, «piega a sinistra»
domani, e l'amico deve cavare una scena sensata da tutt'e due, e da qualunque
frase che ci somigli. Le descrizioni buone smettono così di essere formule
fisse con il vuoto intorno: fra l'una e l'altra ci si sposta senza cadere. Il
vuoto sarebbe un guaio, perché quelle frasi presto se le inventerà M, che lo
schermo non lo guarda mai: se a una frase inventata non corrispondesse nessun
disegno, l'amico poserebbe la matita quasi subito.

`````

`````{tab} Superiore

L'encoder del VAE mappa il fotogramma $\mathbf{x} \in \mathbb{R}^{64 \times 64
\times 3}$ in una distribuzione gaussiana sullo spazio latente:

$$
q_\phi(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}\!\big(\mathbf{z};\, \boldsymbol{\mu}_\phi(\mathbf{x}),\,
\mathrm{diag}(\boldsymbol{\sigma}_\phi^2(\mathbf{x}))\big),
\qquad
\mathbf{z} = \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\epsilon},
\quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}),
$$

dove $\boldsymbol{\mu}_\phi(\mathbf{x})$ e $\boldsymbol{\sigma}_\phi(\mathbf{x})$ sono media e deviazione standard
prodotte da una pila di convoluzioni con parametri $\phi$,
$\mathbf{z} \in \mathbb{R}^{32}$ è il codice latente e la seconda uguaglianza è il
*trucco della riparametrizzazione*, che rende campionabile e derivabile il
passaggio. L'addestramento massimizza l'ELBO, ricostruzione più
regolarizzazione KL verso la prior $\mathcal{N}(\mathbf{0}, \mathbf{I})$, derivato nel
capitolo sui modelli latenti; qui non lo rideriviamo.

```{figure} ../figures/vae-autoencoder-che-immaginano.svg
:name: fig-latente-campionabile
:alt: "Lo spazio latente di un VAE disegnato come un insieme di nuvole gaussiane parzialmente sovrapposte, una per ciascun esempio codificato. Un punto viene campionato in una zona intermedia, che non corrisponde a nessun esempio visto, e il decoder lo trasforma comunque in un'immagine plausibile."
:width: 92%

Perché le nuvole si sovrappongono. Codificando ogni fotogramma in una
distribuzione invece che in un punto, lo spazio resta pieno: anche i punti
mai visti decodificano in qualcosa di sensato.
```

La proprietà mostrata in {numref}`fig-latente-campionabile` è ciò che rende V
utilizzabile da M. Se il latente avesse buchi, la ricorrenza che immagina il
fotogramma successivo produrrebbe presto un codice a cui non corrisponde
nessuna immagine, e il sogno si spezzerebbe dopo pochi passi. Nel paper V viene
addestrato *per primo*, in modo non supervisionato, su fotogrammi raccolti da
una policy casuale; il decoder serve solo in addestramento. Su *CarRacing* V
pesa circa 4,3 milioni di parametri.

`````

### M come Memoria: la fisica del gioco in una RNN

Un fotogramma compresso è una fotografia, non un film: non dice cosa
succederà. Il secondo modulo impara la **dinamica**: dato il codice di adesso
e l'azione scelta, quale sarà il codice di poi? M è una LSTM, la rete
ricorrente con i *gate* (i cancelli che decidono cosa ricordare e cosa
dimenticare) incontrata nel {doc}`capitolo sul Natural Language Processing </NaturalLanguageProcessing/overview>`
{cite}`hochreiter1997long`. Solo che qui la «frase» da proseguire non è fatta
di parole ma di codici $\mathbf{z}$: M vive nel piccolo mondo dei 32 numeri, senza mai
toccare i pixel, perciò è veloce ed economica. Il riassunto che si porta dietro
di passo in passo, cioè la memoria vera e propria, è una fila di 256 numeri, ed
è la $\mathbf{h}$ del disegno; in tutto a M bastano poco più di 400.000 parametri,
meno di un decimo di quelli di V.

`````{tab} Elementare

Un’amica ha giocato mille partite. Le descrivi la situazione in
una frase («sono a metà curva, sto accelerando») e lei ti dice come prosegue:
«esci largo verso l'erba». Non le serve *vedere* lo schermo: le basta il
riassunto, perché la fisica del gioco ce l'ha in testa. Una sfumatura conta:
l'amica onesta non risponde con una certezza ma con un ventaglio, «quasi
sempre esci largo; ogni tanto la tieni». M è costruita così: per ogni
situazione prevede le diverse continuazioni possibili, ciascuna con la sua
probabilità, come le previsioni del tempo che dicono «pioggia al 70%» invece
di giurare sul sole. Il futuro di un gioco (e del mondo) non è mai scritto del
tutto, e un modello che finge di saperlo mente.

`````

`````{tab} Superiore

M è una **MDN-RNN**: una LSTM (256 unità nascoste su *CarRacing*) la cui
testa di uscita è una *mixture density network*, l'idea proposta da
Christopher Bishop nel 1994 per far predire a una rete un'intera
distribuzione anziché un valore. Conviene scrivere la ricorrenza per esteso,
perché è lì che entra l'azione:

$$
\mathbf{h}_{t+1} = \mathrm{LSTM}\big(\mathbf{h}_t,\, [\mathbf{z}_t ; a_t]\big),
$$

$$
P(\mathbf{z}_{t+1} \mid \mathbf{z}_t, a_t, \mathbf{h}_t)
= \prod_{i=1}^{32} \sum_{k=1}^{K} \pi_{k,i}(\mathbf{h}_{t+1})\,
\mathcal{N}\!\big(z_{t+1,i};\, \mu_{k,i}(\mathbf{h}_{t+1}),\,
\sigma_{k,i}^2(\mathbf{h}_{t+1})\big),
$$

dove $\mathbf{h}_t$ è lo stato nascosto della LSTM *prima* del passo $t$ (il riassunto
di tutto ciò che è successo fino a $t-1$ compreso), $[\mathbf{z}_t ; a_t]$ è la
concatenazione del codice corrente e dell'azione scelta, e $\mathbf{h}_{t+1}$ è il
nuovo stato nascosto, funzione deterministica dei tre argomenti che stanno a
destra della barra verticale. È da $\mathbf{h}_{t+1}$, e solo da lì, che escono i
parametri della miscela: se l'azione non entrasse nella ricorrenza, M
predirebbe lo stesso futuro qualunque cosa l'agente faccia, e sarebbe inutile
proprio per la cosa a cui serve, immaginare le conseguenze di una scelta. La
convenzione sui pedici è la stessa del controller, dove $a_t$ si sceglie
leggendo $\mathbf{z}_t$ e $\mathbf{h}_t$: $\mathbf{h}_t$ esiste *prima* che
l'azione sia decisa. Nello scheletro in PyTorch la riga
`nn.LSTM(dim_z + dim_a, dim_h)` monta esattamente questa ricorrenza.

Quanto al resto, $z_{t+1,i}$ è la $i$-esima delle 32 componenti del prossimo
codice: ognuna ha la *propria* miscela di $K = 5$ gaussiane, con pesi
$\pi_{k,i}$ (una softmax, sommano a 1) e con $\mu_{k,i}$, $\sigma_{k,i}$ a
darne centro e incertezza. La fattorizzazione nel prodotto dice che le
scelte di componente sono indipendenti dimensione per dimensione: il modello
non pesa cinque «versioni del futuro» preconfezionate, ne può comporre
$5^{32}$ combinando le alternative di ogni componente.
L'addestramento minimizza la log-verosimiglianza negativa dei codici osservati
nelle partite raccolte; su *Doom*, M predice anche la probabilità che
l'episodio finisca (l'agente muore). Su *CarRacing* M pesa in tutto poco più
di 400.000 parametri, e la miscela è la manopola con cui, tra poco, regoleremo
quanto il «mondo interno» è capriccioso.

`````

### C come Controller: il pilota minimalista

E qui la sorpresa: dopo un compressore da milioni di parametri e una memoria da
centinaia di migliaia, il modulo che *decide* è la cosa più semplice del
capitolo. C prende i 32 numeri del codice visivo e i 256 della memoria, li
mette in fila (288 numeri in tutto) e da quei 288 ricava i tre comandi (sterzo,
acceleratore, freno) facendo per ciascuno una somma pesata: 288 pesi per il
primo comando, 288 per il secondo, 288 per il terzo, più tre numeri che
alzano o abbassano ciascun comando di una quantità fissa. Totale
$288 \times 3 + 3 = 867$ numeri. Non è un vezzo, è la
tesi dell'articolo: se V e M hanno digerito davvero il mondo, per agire bene
basta un riflesso. Tutta l'intelligenza sta nel modello, non nel controllore.

`````{tab} Elementare

Al banco di un fonico c’è una fila di manopole, ciascuna che alza o abbassa
un ingresso, e in uscita tre soli comandi. Gli ingressi, qui, sono i 288 numeri
che descrivono la situazione (che cosa si vede adesso, che cosa si ricorda di
prima), ogni manopola dice quanto ciascuno di quei numeri deve contare, e i tre
comandi in uscita sono sterzo, acceleratore e freno. Imparare a guidare, per C,
vuol dire soltanto trovare la posizione giusta di 867 manopole: pochissimo, se
si pensa che una sola immagine del gioco è fatta di 12.288 numeri. Ed è proprio
questo il punto dell'esperimento: la parte difficile (capire come funziona il
mondo) l'hanno già sbrigata V e M, e a chi deve muovere i pedali resta un
lavoro da riflesso.

Con così poche manopole non serve nemmeno il metodo di addestramento abituale
delle reti, quello che dopo ogni errore ritocca ogni peso di un soffio nella
direzione che conviene (in gergo si chiama seguire il **gradiente**). Basta un
metodo alla Darwin: si provano 64 piloti presi un po’ a caso, si tengono quelli
che hanno guidato meglio, si fa una nuova generazione somigliante a loro, e si
ricomincia. Ci vuole pazienza (nell'articolo le generazioni sono
1.800), ma alla fine si guida.

`````

`````{tab} Superiore

La forma esatta è una moltiplicazione di matrice:

$$
a_t = \tanh\big(\mathbf{W}_c\,[\mathbf{z}_t ; \mathbf{h}_t] + \mathbf{b}_c\big),
$$

dove $[\mathbf{z}_t ; \mathbf{h}_t]$ è la concatenazione del codice visivo e dello stato della
memoria ($32 + 256 = 288$ numeri), $\mathbf{W}_c$ è una matrice $3 \times 288$ e $\mathbf{b}_c$
un vettore di tre bias, uno per azione: sterzo, acceleratore, freno. Totale:
$288 \times 3 + 3 = 867$ parametri. La tangente iperbolica non aggiunge
capacità (schiaccia soltanto le uscite in $[-1, 1]$; acceleratore e freno
vengono poi riportati in $[0, 1]$), quindi la policy è a tutti gli effetti
lineare. Un controllore così piccolo si può addestrare **senza gradiente**:
gli autori usano CMA-ES, una strategia evolutiva che a ogni generazione fa
«gareggiare» 64 varianti del controllore, ne stima media e covarianza e
ricampiona da lì la generazione successiva. Con 867 numeri da scegliere
l'evoluzione basta e avanza, ed è anche la strada più naturale, visto che il
segnale su cui giudicare un pilota (il punteggio) arriva solo a fine episodio.

`````

## Allenarsi nel sogno

Fin qui M ha fatto da spalla a C: gli passava la propria memoria, il riassunto
di quel che era successo prima, e C decideva. Adesso guardiamo l'anello
tratteggiato della
{numref}`fig-world-model-vmc`. M predice il prossimo codice $\mathbf{z}$; e se quel
codice, invece di confrontarlo con la realtà, lo ridessimo in pasto a M come
ingresso del passo successivo? Il modello comincia a raccontarsi il gioco da
solo, un passo dopo l'altro: niente più ambiente, niente pixel, solo codici
che generano codici. Ha e Schmidhuber lo chiamano *dream*, sogno, e
l'esperimento è tutto qui: C viene addestrato **esclusivamente** dentro il
sogno e poi trasferito, senza ritocchi, nel gioco vero.

Sulla parola conviene fermarsi un secondo, perché si porta dietro qualcosa che
qui non c'entra. Un sogno vero è sconclusionato, e da un sogno ci si aspetta
che sbagli; questo invece è una simulazione, e la si vuole fedele: quando si
scolla dal gioco vero non è pittoresco, è un guasto, ed è il guasto di cui
parla il resto della sezione. Per il resto la parola calza: il gioco è
staccato, si procede a occhi chiusi, e quel che si vede se lo sta inventando
chi lo guarda.

Un cambio di scena, però, va dichiarato. I numeri dati finora (32 numeri di
codice, 256 di memoria, 867 parametri di controller) sono quelli di
*CarRacing*, e su *CarRacing* il controller gli autori lo fanno evolvere
nell'ambiente **vero**: l'unico esperimento allenato davvero dentro il sogno è
l'altro, lo sparatutto: per la precisione lo scenario *Take Cover* di VizDoom
(la versione di *Doom* usata nella ricerca), che d'ora in poi chiamiamo con il
suo nome. Lì lo stesso schema usa un codice da 64
numeri, una memoria da 512 numeri e un controller da 1088 parametri: la ricetta
è la stessa, le taglie no.[^taglie-doom]

`````{tab} Elementare

È il pilota che la sera prima della gara ripassa il circuito a occhi chiusi,
curva per curva, i piloti veri lo fanno davvero: costa zero benzina e zero
incidenti. E se il gioco vero è staccato, chi tiene il punteggio? Il sogno
stesso. Nello sparatutto il punteggio è quanto sopravvivi, e M, oltre al
fotogramma dopo, prevede anche se sei stato colpito: quando decide che l'hai
presa, la partita sognata finisce e il punteggio è la sua durata. Ma c'è un
tallone d'Achille: se nella tua testa una curva è più dolce che in pista,
impari una traiettoria che domani ti manda nella ghiaia. All'agente di Ha e
Schmidhuber successe qualcosa di più subdolo: dentro il sogno scoprì dei
*trucchi*. Trovò modi di muoversi per cui i mostri, in certe partite sognate,
non sparavano un colpo, e in certe altre le palle di fuoco svanivano. Stava
barando al *proprio sogno*, sfruttandone i difetti, come uno studente che si
prepara all'esame inventandosi da solo domande facili. Punteggi splendidi nel
mondo immaginato, figuraccia in quello vero.

Il rimedio è rendere il sogno *più capriccioso* del
gioco vero, e si fa girando una manopola sola, la **temperatura**. M non
annuncia una continuazione unica ma un ventaglio di continuazioni con le loro
probabilità: alzando la temperatura escono più spesso quelle improbabili. Il
sogno diventa dispettoso, e un trucco che ha funzionato una volta la volta dopo
non funziona più. Alzarla troppo, però, non conviene: un sogno completamente
sregolato non somiglia più a niente, e lì dentro non si impara nulla. Il punto
giusto della manopola lo si scova provando, e lì l'agente trovò il gioco vero
quasi riposante: vi sopravvisse in media *più a lungo* che nel proprio sogno.

Resta la cosa da cui siamo partiti: il sogno è stato imparato guardando partite
giocate a casaccio. Il pilota è cresciuto dentro la copia di un gioco che
nessun bravo giocatore ha mai giocato, e questo è un limite non del sogno in sé
ma di quello che il sogno ha avuto occasione di vedere. Per un gioco più ricco
di questi due si torna in pista: il pilota che ha imparato qualcosa va a
correre sul serio, e con le situazioni nuove che si porta a casa il ripasso a
occhi chiusi si rifà da capo.

`````

`````{tab} Superiore

Un *rollout* nel modello è la catena

$$
a_t = C(\mathbf{z}_t, \mathbf{h}_t), \qquad
\mathbf{z}_{t+1} \sim P_\tau(\,\cdot \mid \mathbf{z}_t, a_t, \mathbf{h}_t), \qquad
\mathbf{h}_{t+1} = \mathrm{LSTM}\big(\mathbf{h}_t,\, [\mathbf{z}_t ; a_t]\big),
$$

dove la terza uguaglianza è la ricorrenza di M scritta poco fa, qui senza
più nessun fotogramma a rifornirla: il codice che entra al passo dopo è quello
che M ha appena inventato. Il campionamento dalla miscela avviene a
**temperatura** $\tau$: un parametro che gonfia ($\tau > 1$) o spegne
($\tau \to 0$) l'incertezza della distribuzione predetta. Il problema
strutturale è che C viene ottimizzato *contro M*, non contro l'ambiente: ogni
errore sistematico del modello diventa una risorsa da sfruttare, e la ricerca
di policy trova politiche avversarie al proprio stesso mondo interno; nel
paper, rollout in cui i mostri non sparano mai, o in cui certi movimenti
«estinguono» le palle di fuoco. Con $\tau$ basso il sogno è docile e l'inganno
prospera: a $\tau = 0{,}10$ l'agente totalizza $2086 \pm 140$ nel proprio sogno
e $193 \pm 58$ nell'ambiente vero, che è il modo più netto di dire «transfer
disastroso».

Alzare $\tau$ è il rimedio, ma fino a un certo punto, e la tabella di *Take
Cover* dice esattamente dove (punteggi su 100 rollout, media e deviazione
standard):

| $\tau$ | nel sogno | nell'ambiente vero |
|---|---|---|
| 0,10 | $2086 \pm 140$ | $193 \pm 58$ |
| 1,00 | $1145 \pm 690$ | $868 \pm 511$ |
| 1,15 | $918 \pm 546$ | $1092 \pm 556$ |
| 1,30 | $732 \pm 269$ | $753 \pm 139$ |

La curva non è monotona: ha un massimo a $\tau = 1{,}15$, dove l'agente va
*meglio* nella realtà che nella propria immaginazione e supera largamente la
soglia di risoluzione (750); a 1,30 ricade a tre punti sopra quella soglia,
perché il sogno è diventato così rumoroso che dentro non si impara più niente.
Gli autori lo dicono con parole loro: alzare $\tau$ rende più difficile a C
trovare politiche avversarie, ma alzarla troppo rende l'ambiente virtuale troppo
difficile perché l'agente impari alcunché, e quindi è un **iperparametro da
tarare**. Nel paper non c'è alcun criterio per sceglierlo a priori, né la
pretesa che 1,15 valga altrove.

Resta da dire da dove viene il sogno, perché è il vincolo che decide tutto. V e
M non nascono dal nulla: sono addestrati su rollout raccolti **nell'ambiente
vero da una policy casuale**. Su *Take Cover* quella policy totalizza
$210 \pm 108$, contro i 1092 dell'agente finale: il modello del mondo dentro
cui cresce il pilota è stato imparato guardando qualcuno che gioca malissimo.
Gli autori dichiarano che questo basta *perché i due compiti sono semplici*, e
per ambienti più ricchi prescrivono una procedura **iterativa**, in cui
l'agente torna a raccogliere dati veri e il modello viene riaddestrato. Il
limite, quindi, non è solo quanto M è preciso: è che cosa la policy di raccolta
ha avuto occasione di vedere. E il disallineamento tra $P_\tau$ e la vera
dinamica non si annulla comunque: gli errori si accumulano lungo il rollout,
ragione per cui i sogni utili sono brevi.

`````

Che gli errori si accumulino è una di quelle cose che si leggono e si
accettano senza vederle. {numref}`fig-sogno-diverge` la mette in scena sul
mondo più piccolo che si possa immaginare: **un'altalena che qualcuno continua a
spingere**. Va avanti e indietro, a ogni passaggio perde un po’ di slancio per
l'attrito e ne riceve un po’ dalla spinta, e nella finestra disegnata la spinta
vince: l'ampiezza cresce. Il modello che se la immagina sbaglia una cosa sola, e
di poco: **quanto slancio sopravvive** a ogni passaggio. Crede che ne sopravviva
un filo più del vero, il 2,8 per cento in più. Basta quello.

```{figure} ../figures/sogno-diverge.svg
:name: fig-sogno-diverge
:alt: "Due curve che oscillano come un'altalena partono dallo stesso punto e restano sovrapposte per una quindicina di passi, tanto da sembrare una sola; poi si separano sempre di più. L'asse orizzontale conta i passi, quello verticale dice dove si trova l'altalena. Una fascia ombreggiata copre la parte finale del grafico, da dove lo scarto ha superato la tolleranza in poi."
:width: 92%

La stessa spinta iniziale, due altalene quasi identiche: una vera e una
immaginata. Per sedici passi il sogno è una fotocopia della realtà; poi si
stacca. La fascia ombreggiata, a destra della riga tratteggiata, comincia dove
lo scarto **peggiore fin lì** ha superato quello che si era deciso di
tollerare: da lì in avanti il sogno non è più roba su cui allenare nessuno.
```

Tre cose conviene notare in {numref}`fig-sogno-diverge`, e nessuna delle tre
si vede in un fotogramma.

La prima è che l'inizio è **identico**. Chi guardasse solo i primi passi
concluderebbe che il modello è ottimo, ed è esattamente il modo in cui un
modello del mondo viene di solito valutato: **un passo alla volta**, cioè
partendo da una situazione vera, chiedendogli che cosa succede subito dopo e
misurando quanto ha sbagliato, poi ripartendo da un'altra situazione vera. Un
modello promosso a pieni voti da questa prova può essere bocciato appena lo si
lascia andare da solo per venti passi, ed è quello che qui succede.

La seconda richiede di guardare bene, perché è controintuitiva: lo scarto fra
le due curve, misurato passo per passo sulla scala verticale del disegno,
**si richiude**, anche parecchio. Al
passo 17 vale 0,43 e al 19 è sceso a 0,06, perché le due altalene, oscillando,
ogni tanto si ritrovano dalla stessa parte per caso. Quello che non torna più
indietro è il **record**, cioè il peggiore scarto visto fin lì, ed è l'unica
quantità onesta con cui giudicare un sogno: un modello che al passo 19 sembra
tornato buono ha comunque già sbagliato di 0,43, e su quell'errore ci ha
costruito sopra tutti i passi seguenti.

La terza è che il numero di passi affidabili non è una proprietà del modello
da solo: dipende da quanto scarto si è disposti a tollerare. Qui la tolleranza
è 0,25, e nei sedici passi che il sogno regge, fra il punto più alto e il più
basso, l'altalena si sposta di poco più di cinque di quelle stesse
unità.[^scala-altalena] Si sta accettando, insomma, uno scarto pari a un
ventesimo scarso del movimento, che è una scelta e non una legge: chi
accettasse il doppio di scarto si terrebbe cinque passi in più, ventuno invece
di sedici. Dichiararla non è pignoleria: chi non dichiara la tolleranza non
sta dichiarando neanche l’**orizzonte**, cioè fino a che punto il sogno
conviene essere ascoltato.

Una quarta cosa, infine, la figura non può mostrarla, ed è bene non dedurla da
qui. *Quanto in fretta* lo scarto si apra non è una legge universale: dipende
da quanto il sistema amplifica gli scossoni che riceve. Il {doc}`capitolo sul Deep
Reinforcement Learning </DeepReinforcementLearning/overview>` lo scrive per bene, e mostra che su una dinamica
abbastanza mite lo scarto, invece di esplodere, si assesta.

## Dai sogni ai diamanti: la linea Dreamer

*World Models* era una dimostrazione su due videogiochi. Trasformarla in un
metodo generale è stato in buona parte il lavoro di Danijar Hafner e colleghi.
Dreamer (2020) impara i comportamenti senza quasi mai uscire dal proprio
modello: le partite su cui si allena sono tutte immaginate, e sono immaginate
nello spazio dei codici, non in quello dei pixel. Una catena di passi generati
uno dall'altro si chiama **rollout**, ed è esattamente il sogno di poco fa; la
parola vale anche per le partite vere, quando si raccolgono una mossa alla
volta. A imparare da quei rollout sono due reti che si danno il
cambio, e le abbiamo incontrate nel {doc}`capitolo sul Deep Reinforcement Learning </DeepReinforcementLearning/overview>`:
l’**attore**, che sceglie la mossa, e il **critico**, che stima quanto vale la
situazione in cui l'attore si è cacciato, così che l'attore sappia subito se ha
fatto bene invece di dover aspettare la fine della partita. DreamerV2 (2021) è
il primo agente a livello umano sul banco di prova dei giochi Atari imparando
dentro un world model; DreamerV3, pubblicato su *Nature* nel 2025
{cite}`hafner2023mastering`, affronta più di 150 compiti (robot simulati,
Atari, navigazione 3D) con la **stessa identica configurazione**, senza
ritocchi per dominio. Il risultato simbolo: applicato così com'è a Minecraft,
è il primo algoritmo a raccogliere **diamanti** partendo da zero, senza
dimostrazioni umane né curricula. Arrivarci richiede una catena lunghissima di
sotto-obiettivi (legno, banco da lavoro, picconi via via migliori, ferro da
fondere, scavi in profondità) con ricompense rarissime lungo il cammino: il
tipo di compito su cui, come ha mostrato il capitolo sul Deep Reinforcement
Learning con *Montezuma's Revenge*, il DQN si arena.

È qui il raccordo con il capitolo sul Deep Reinforcement Learning: i world
model sono la risposta **model-based** alla fame di esperienza vera dei metodi
*model-free*. In gergo quell'esperienza si conta in **campioni**, dove un
campione è una singola interazione con l'ambiente, e la fame è quella: ne
servono milioni.

`````{tab} Elementare

Ricordate il conto pagato in apertura di capitolo: al DQN servono decine di
milioni di fotogrammi per imparare un gioco Atari dove a un umano bastano
minuti {cite}`mnih2015human`. Ogni esperienza serve solo ad aggiustare di un
soffio le valutazioni, come uno studente che di un'intera lezione trattiene
una riga. Un world model spreme la stessa esperienza molto di più: ogni
partita vera migliora la copia interna del gioco, e dentro la copia ci si
allena quanto si vuole, al solo costo dell'elettricità. L'idea, in piccolo, ha
più di trent'anni: si chiama Dyna, l'architettura con cui Richard Sutton nel
1990 faceva alternare a un agente mosse vere e mosse «ripassate» in un
modellino imparato del labirinto {cite}`sutton1990integrated` (un antenato a
caselle dei sogni di Dreamer, divulgato l'anno dopo in una versione più breve
{cite}`sutton1991dyna`).

C'è un prezzo, però: quel che si impara nella copia vale quanto la copia. Se il
modellino mette un muro dove il labirinto vero ha un corridoio, l'agente impara
benissimo a schivare un muro che non esiste. E dove le partite vere costano
poco, giocarle davvero resta competitivo: fra chi sogna e chi prova, la partita
è ancora aperta.

`````

`````{tab} Superiore

Un metodo *model-free* come il DQN stima direttamente valori o policy
dall'esperienza; un metodo *model-based* impara anche un modello della
dinamica $p(s_{t+1} \mid s_t, a_t)$ e lo usa per generare transizioni
sintetiche. Dyna {cite}`sutton1990integrated` è lo schema capostipite: gli
aggiornamenti di $Q$ attingono sia da transizioni reali sia da transizioni
simulate dal modello appreso, mescolando apprendimento e pianificazione. I
Dreamer ne sono l'erede profondo: un modello ricorrente dello stato (RSSM),
con una componente deterministica e una stocastica, apprende la dinamica nello
spazio latente; attore e critico vengono addestrati per retropropagazione
attraverso rollout immaginati con orizzonte breve (una quindicina di passi)
per contenere l'accumulo degli errori del modello; DreamerV3 aggiunge
normalizzazioni robuste (osservazioni, ricompense, ritorni) che rendono gli
stessi iperparametri validi su domini radicalmente diversi
{cite}`hafner2023mastering`. Il guadagno è l'efficienza campionaria; il tetto
è la qualità del modello: la policy è buona quanto il sogno in cui è
cresciuta, e su dinamiche caotiche o eventi rari i modelli restano il punto
debole. Il confronto con i metodi model-free, competitivi quando i campioni
costano poco, è tutt'altro che chiuso.

`````

Una nota di prospettiva: oggi «world model» è anche un'etichetta di moda per
i grandi modelli generativi di video, promossi a simulatori del mondo fisico.
La parentela concettuale c'è. Quello che quei modelli non hanno ancora mostrato
è proprio la cosa raccontata qui: reggere un intero addestramento al proprio
interno, cioè lasciarci crescere dentro un agente che poi, riportato fuori,
funzioni davvero.

## I tre moduli in PyTorch

Chiudiamo con lo scheletro di V, M e C: poche righe, con le dimensioni di ogni
pacchetto di numeri scritte nei commenti. Manca tutta la parte di addestramento: i tre moduli qui nascono con i pesi a
caso e non imparano niente. Quello che il codice mostra è il percorso dei
dati, cioè chi passa che cosa a chi, ed è quello vero.

```python
import torch
from torch import nn

class EncoderVAE(nn.Module):
    """V: comprime un fotogramma 3x64x64 in un codice z di 32 numeri."""
    def __init__(self, dim_z=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(),    # -> (32, 31, 31)
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),   # -> (64, 14, 14)
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),  # -> (128, 6, 6)
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU(), # -> (256, 2, 2)
            nn.Flatten(),                                # -> 1024
        )
        self.mu = nn.Linear(1024, dim_z)       # media del codice
        self.logvar = nn.Linear(1024, dim_z)   # log-varianza del codice

    def forward(self, x):                      # x: (B, 3, 64, 64)
        h = self.conv(x)                       # (B, 1024)
        mu, logvar = self.mu(h), self.logvar(h)
        eps = torch.randn_like(mu)             # riparametrizzazione
        return mu + torch.exp(0.5 * logvar) * eps   # z: (B, 32)

class ModelloRNN(nn.Module):
    """M: dato il codice e l'azione, predice il codice del passo dopo.
    (Versione deterministica; il paper usa una miscela di gaussiane.)"""
    def __init__(self, dim_z=32, dim_a=3, dim_h=256):
        super().__init__()
        self.lstm = nn.LSTM(dim_z + dim_a, dim_h, batch_first=True)
        self.testa = nn.Linear(dim_h, dim_z)   # media del prossimo z

    def forward(self, z, a, stato=None):       # z: (B, T, 32), a: (B, T, 3)
        ingresso = torch.cat([z, a], dim=-1)   # (B, T, 35)
        h, stato = self.lstm(ingresso, stato)  # h: (B, T, 256)
        return self.testa(h), stato            # z predetto: (B, T, 32)

class Controller(nn.Module):
    """C: policy lineare da codice e memoria all'azione."""
    def __init__(self, dim_z=32, dim_h=256, dim_a=3):
        super().__init__()
        self.lineare = nn.Linear(dim_z + dim_h, dim_a)   # 288*3+3 = 867

    def forward(self, z, h):                   # z: (B, 32), h: (B, 256)
        # azioni in [-1, 1]; gas e freno andrebbero poi riportati in [0, 1]
        return torch.tanh(self.lineare(torch.cat([z, h], dim=-1)))
```

E questo è il circuito del sogno: dieci passi interamente nello spazio dei
codici, con M che fa da ambiente a se stesso. Le azioni qui sono casuali;
nell'addestramento vero le sceglierebbe C, e il punteggio sognato guiderebbe
l'evoluzione dei suoi pochi parametri (867 con le taglie di *CarRacing* usate
in questo scheletro, 1088 su *Take Cover*, che è il gioco in cui il sogno è
stato davvero adoperato per allenare).

Due avvertenze prima di metterci le mani, e servono a non aspettarsi da queste
righe più di quello che danno.

La prima: con i pesi non addestrati la ricorrenza dimentica quasi subito da
dove è partita. Dopo due o tre passi il sogno prosegue uguale qualunque
fotogramma lo abbia iniziato, e cambiare il fotogramma di partenza (che è la
prima cosa che viene in mente di provare) non produce nessun effetto visibile.
Lo si misura così: si parte da cinque fotogrammi diversi invece che da uno, si
danno a tutti e cinque le stesse identiche azioni, e si guarda quanto restano
distanti fra loro i cinque codici sognati. Dopo dieci passi quella distanza è
scesa di più di tremila volte rispetto alla partenza, e i tre comandi finali si
somigliano fino a meno di due millesimi.[^sei-prove]

La seconda: questo M predice *un* codice solo e non un ventaglio di
continuazioni possibili, quindi la manopola della temperatura qui non c'è,
perché non ci sono probabilità da rimescolare.

```python
V, M, C = EncoderVAE(), ModelloRNN(), Controller()
print(sum(p.numel() for p in C.parameters()))   # 867: il pilota è minuscolo

x = torch.rand(1, 3, 64, 64)     # un fotogramma finto: batch 1, RGB, 64x64
z = V(x).unsqueeze(1)            # (1, 1, 32): il codice, come sequenza di 1 passo
stato = None                     # memoria (h, c) della LSTM, vuota all'inizio

for t in range(10):              # dieci passi di sogno: nessun ambiente
    a = torch.rand(1, 1, 3) * 2 - 1        # azione casuale in [-1, 1]
    z, stato = M(z, a, stato)              # il codice sognato: (1, 1, 32)

# il controller legge codice e memoria e restituisce i tre comandi
h = stato[0].squeeze(0)          # stato nascosto della LSTM: (1, 256)
comandi = C(z.squeeze(1), h)     # (1, 3): sterzo, acceleratore, freno
```

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un **world model** è una copia interna del mondo, imparata e ridotta
  all'osso: pensare e sbagliare lì dentro non costa quasi niente.
- La ricetta di Ha e Schmidhuber (2018) è fatta di tre pezzi. **V** guarda e
  riassume: da un'immagine di 12.288 numeri ne tira fuori 32. **M** ricorda e
  prevede: dato il riassunto di adesso e la mossa scelta dice come potrebbe
  continuare, e non con una certezza ma con un ventaglio di possibilità.
  **C** decide, ed è ridicolmente piccolo, 867 manopole. La tesi
  dell'articolo è tutta qui: se i primi due hanno capito il mondo, al terzo
  basta un riflesso.
- Il **sogno** è quel che succede quando si stacca il gioco e si lascia che M
  si racconti la partita da solo, un passo dopo l'altro. Il rischio è lo
  studente che si prepara all'esame inventandosi domande facili: l'agente
  scopre i difetti del proprio sogno e ci sguazza (in certe partite sognate i
  mostri non sparavano un colpo). Il rimedio è rendere il sogno più
  capriccioso del mondo vero, ma con misura: troppo capriccioso, e lì dentro
  non si impara più niente.
- Allenato così e riportato nel gioco vero senza alcun ritocco, l'agente di
  *Doom* se la cava meglio della soglia che definisce il livello superato.
- I **Dreamer**, negli anni successivi, portano l'idea a maturità: l'ultimo
  (*Nature*, 2025) impara più di 150 compiti diversi con le stesse
  impostazioni, e in *Minecraft* arriva a scavare diamanti senza che nessuno
  gli abbia mai mostrato come si fa.
- Il guadagno è l'esperienza risparmiata: chi ha un modello si allena gratis
  nella propria testa, chi non ce l'ha deve provare tutto per davvero. Il
  limite è sempre lo stesso: una strategia è buona quanto il sogno in cui è
  cresciuta, e un sogno è buono quanto le partite che gli sono state date da
  guardare.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Un **world model** è una copia interna, compressa e imparata,
  dell'ambiente: pensare e sbagliare lì dentro costa quasi nulla.
- La ricetta di Ha e Schmidhuber (2018): **V**, un VAE che comprime il
  fotogramma in 32 numeri; **M**, una MDN-RNN che, ingerito $[\mathbf{z}_t ; a_t]$,
  predice la *distribuzione* del prossimo codice; **C**, una policy lineare.
  L'intelligenza sta nel modello, non nel controllore.
- Le taglie cambiano con il gioco, e conta sapere quale: su *CarRacing*
  ($\mathbf{z}$ a 32 numeri, LSTM a 256 unità, C a 867 parametri) il controller è
  evoluto nell'ambiente **vero**; l'esperimento addestrato **solo nel sogno**
  è *VizDoom: Take Cover*, con $\mathbf{z}$ a 64, LSTM a 512 e C a 1088 parametri.
- Il **sogno** è un rollout in cui M alimenta se stesso; ma V e M sono stati
  imparati su rollout raccolti nel mondo vero da una **policy casuale**, e gli
  autori dichiarano che basta *perché i compiti sono semplici*: per ambienti
  più ricchi prescrivono una raccolta **iterativa**.
- Rischio del sogno: sfruttarne i difetti (i mostri che non sparano). Rimedio:
  **tarare** la temperatura $\tau$ verso l'alto, non alzarla e basta. Su
  *Take Cover* l'ottimo è $\tau = 1{,}15$ (1092 nel mondo vero contro 868 a
  $\tau = 1$), e già a 1,30 si ricade a 753, tre punti sopra la soglia di
  risoluzione.
- La linea **Dreamer** porta l'idea a maturità: DreamerV3 (*Nature*, 2025)
  impara nell'immaginazione latente su rollout brevi (una quindicina di
  passi), usa gli stessi iperparametri su più di 150 compiti e trova i
  diamanti in Minecraft senza dimostrazioni umane.
- È la risposta **model-based** alla fame di campioni del DQN, con un
  antenato preciso: Dyna di Sutton (1990). Il limite resta la qualità del
  modello, e prima ancora la copertura dei dati su cui l'ha imparata.
```

`````

[^taglie-doom]: Chi prova a rifare quel 1088 sommando $64 + 512$ non ci arriva,
    e la ragione è una differenza vera: su *Take Cover* il controller legge
    anche il secondo dei due riassunti che una LSTM si porta dietro (lo stato
    di *cella*), quindi in ingresso ha $64 + 512 + 512 = 1088$ numeri, e da lì
    ricava un comando solo, andare a sinistra o a destra. Il conto dei
    parametri, a rigore, farebbe uno in più per via del termine costante: il
    paper riporta la larghezza dell'ingresso.

[^scala-altalena]: Il confronto va fatto sui passi buoni. Prendendo tutto il
    tracciato l'escursione quasi raddoppia, ma i suoi estremi cadono **dopo**
    la rottura, cioè dentro il tratto che stiamo dichiarando inaffidabile:
    dividere per quelli farebbe sembrare la tolleranza più piccola di quello
    che è.

[^sei-prove]: I numeri esatti: ripetendo la prova sei volte, ciascuna con pesi
    casuali diversi, la distanza fra i codici sognati si riduce di un fattore
    compreso fra 3.300 e 4.700.
