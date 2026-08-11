# Tendenze e limiti

Chiudere un capitolo sui Transformer con le previsioni è un esercizio
rischioso: questo campo brucia le profezie in fretta. Più utile fissare le
direzioni di lavoro visibili oggi, e i problemi aperti che le motivano. Perché
il paradosso è proprio questo: mai un'architettura ha funzionato così bene, e
mai è stato così chiaro quanto costa farla funzionare.

## Dove punta la ricerca

```{figure} ../figures/distillazione-insegnante-allievo.svg
:name: fig-distillazione
:alt: "Un modello maestro, grande, riceve un input e produce non una sola risposta ma una distribuzione di probabilità su tutte le risposte possibili. Un modello allievo, molto più piccolo, viene addestrato a riprodurre quella distribuzione intera invece della sola risposta corretta."
:width: 92%

Perché imparare dal maestro batta imparare dalle etichette. L'etichetta dice
solo qual è la risposta giusta; la distribuzione del maestro dice anche quali
errori erano quasi ragionevoli, e quella è informazione in più.
```

Il dettaglio di {numref}`fig-distillazione` che spiega il metodo è la forma
di ciò che passa dal maestro all'allievo. Non è la risposta, è l'intera
graduatoria: fra «gatto» e «lince» il maestro esita, fra «gatto» e «camion»
no, e l'allievo impara anche questa geometria, che nessuna etichetta secca
gli avrebbe insegnato.

`````{tab} Elementare
Tre cantieri, su tutti. Il primo è **fare di più con meno**: i grandi modelli
sono motori potentissimi che consumano moltissimo, e buona parte della ricerca
è una gara di efficienza; modelli più piccoli che imparano dai grandi come
apprendisti dal maestro, versioni "compresse" che girano su un telefono invece
che in un centro di calcolo. Il secondo è **unire i sensi**: modelli che
leggono, guardano e ascoltano insieme, come l'assistente a cui mostri una foto
e fai una domanda a voce. Il terzo è **superare i limiti dell'architettura
stessa**: l'assemblea plenaria delle parole (il costo quadratico visto nel
confronto con le RNN) spinge a cercare modi più economici di collegare le
parti di un testo lungo.
`````

`````{tab} Superiore
Sul fronte dell'efficienza: **distillazione** (un modello piccolo addestrato a
imitare le uscite di uno grande), **quantizzazione** (pesi a 8 o 4 bit invece
che a 32, con perdite di qualità spesso modeste), **pruning**, e architetture
*mixture-of-experts* che attivano solo una frazione dei parametri per ogni
token. Sul fronte del contesto lungo: attenzioni sparse e lineari,
ottimizzazioni di memoria come FlashAttention, e gli *state space model*
(Mamba); a questi ultimi, e alle attenzioni lineari, sono dedicati i due
capitoli che seguono. Sul fronte multimodale: spazi di rappresentazione
condivisi tra testo, immagini e audio, con il transfer learning contrastivo
alla CLIP come collante. A cui si aggiunge il filone dell'**allineamento**:
tecniche (come il fine-tuning con feedback umano) per rendere i modelli più
utili e meno dannosi, che è oggi un'area di ricerca a pieno titolo, non un
ritocco finale.
`````

## Pensare più a lungo sulle cose difficili

C'è un filone che vale la pena isolare, perché nasce da un'osservazione così
semplice da sembrare ingenua e perché la sua storia insegna qualcosa su come
procede questo campo.

`````{tab} Elementare

Un Transformer fa **sempre lo stesso numero di passaggi**. Che gli si chieda
quanto fa due più due o di sbrogliare un ragionamento in dieci mosse, il testo
attraversa esattamente gli stessi strati, e quindi riceve la stessa quantità di
calcolo. Detta così suona strana, perché non è affatto come funzioniamo noi:
sulle cose facili rispondiamo a colpo, sulle difficili ci fermiamo a pensare.

L'idea, allora, è di lasciare che il modello decida **quanto pensare**, e nel
2018 qualcuno ci provò: invece di impilare strati tutti diversi, se ne usa uno
solo applicato più volte di fila (diventa una specie di ricorrenza, non nel
tempo ma in profondità), e a ogni giro ogni parola può dire «io ho finito» e
smettere, mentre le altre continuano.

All'epoca non prese piede. È tornata attuale adesso, per una strada
inaspettata: i modelli che «ragionano» prima di rispondere fanno, in fondo, la
stessa cosa, cioè spendere più calcolo sulle domande difficili. Solo che lo
fanno **scrivendo** il ragionamento, un passo alla volta in parole, invece di
girare più volte dentro sé stessi in silenzio. Quale delle due strade sia la
migliore è una questione aperta: la prima è più economica, la seconda si può
leggere.

`````

`````{tab} Superiore

Il **Universal Transformer** {cite}`dehghani2019universal` sostituisce gli $N$
strati distinti con **un solo blocco applicato ricorrentemente in profondità**,
cioè con i pesi legati fra le iterazioni. La motivazione dichiarata è
recuperare il *bias induttivo* ricorrente che il Transformer aveva buttato via
insieme alla ricorrenza temporale, e che serve sui compiti a struttura
gerarchica e sulla generalizzazione a lunghezze non viste in addestramento.

Sopra ci mettono l'**Adaptive Computation Time** di Graves: a ogni iterazione,
per **ogni posizione**, una piccola unità emette una probabilità di
arresto; le posizioni che si fermano vengono copiate invariate mentre le altre
continuano a essere aggiornate, e una penalità sul numero di passi (il *ponder
cost*) impedisce di pensare all'infinito. Il calcolo diventa così
**condizionato all'ingresso** invece che fissato dall'architettura.

Vale la pena essere precisi su un punto che il titolo lascia intuire: legare i
pesi e iterare rende il modello **computazionalmente universale** in un senso
tecnico. Un Transformer standard esegue un numero di passi sequenziali
indipendente dalla lunghezza dell'ingresso, il che lo rende Turing-incompleto;
una ricorrenza in profondità con numero di passi dipendente dai dati toglie
quel limite.

L'idea è rimasta a lungo marginale e oggi è di nuovo centrale, arrivata però
dall'altra parte. I modelli che **ragionano** allocando più calcolo in
inferenza fanno la stessa cosa nello **spazio dei token** invece che nello
spazio latente: generano una catena di passi intermedi, e più il problema è
difficile più ne generano. Le due vie hanno un compromesso opposto e non
risolto. Il calcolo latente è più economico (nessun token da produrre e
rileggere) e **non è ispezionabile**; quello in token costa di più, è più
facile da addestrare con la supervisione esistente, e lascia una traccia che si
può leggere, il che nel capitolo sull'interpretabilità è tutt'altro che un
dettaglio. Che la traccia sia poi una descrizione *fedele* del calcolo svolto è
una domanda a sé, e la risposta corrente è: non necessariamente.

`````

## I limiti che restano

Un elenco onesto, da tenere accanto agli entusiasmi:

- **Costo**: addestramento e inferenza dei modelli maggiori richiedono risorse
  (economiche, energetiche, di hardware) concentrate in poche aziende; la
  ricerca indipendente lavora per necessità su scala ridotta.
- **Dati**: i corpora del web si stanno esaurendo come fonte "gratuita" di
  testo di qualità, e portano con sé bias e contenuti problematici che i
  modelli assorbono.
- **Affidabilità**: le allucinazioni (risposte fluenti ma false) derivano
  dalla natura stessa del modello autoregressivo, che ottimizza la
  plausibilità, non la verità; mitigarle (con il recupero di fonti esterne, la
  verifica, la calibrazione) è un problema aperto.
- **Comprensione**: su cosa i modelli *capiscano* davvero il dibattito
  scientifico è tutt'altro che chiuso; prudenza nell'attribuire loro
  intenzioni o ragionamento è una virtù, oltre che buona epistemologia.

## Il posto dei Transformer in questo libro

Con questo capitolo si chiude un tratto del percorso tecnico del libro: dai
neuroni del percettrone all'attenzione, ogni pezzo dei Transformer (tensori,
gradienti, softmax, embedding, training loop) è un concetto che hai già
incontrato nei capitoli precedenti, montato in una configurazione nuova. È la
lezione migliore di questa storia: le "rivoluzioni" dell'AI, viste da vicino,
sono quasi sempre ricombinazioni ingegnose di idee semplici, rese possibili da
più dati e più calcolo. Chi conosce le idee semplici non insegue le mode: le
legge.

```{admonition} Da ricordare
:class: important
- La ricerca punta su **efficienza** (distillazione, quantizzazione, esperti
  selettivi), **contesto lungo** (attenzioni economiche, state space model) e
  **multimodalità**.
- Un Transformer spende **lo stesso calcolo** su ogni ingresso, facile o
  difficile che sia. L'**Universal Transformer** (2018) provò a togliere quel
  vincolo legando i pesi fra gli strati e lasciando che ogni posizione decida
  quando fermarsi. Non prese piede allora, ed è tornata attuale dalla parte
  opposta: i modelli che ragionano spendono più calcolo sulle domande difficili
  **scrivendo** i passi invece di girare in silenzio. La prima strada costa
  meno, la seconda si può leggere.
- I limiti sono strutturali: costi concentrati, bias dei dati,
  allucinazioni, e una comprensione ancora dibattuta di cosa i modelli
  sappiano davvero.
- Tutti gli ingredienti dei Transformer li hai già studiati in questo libro:
  ciò che è nuovo è la composizione, non i mattoni.
```
