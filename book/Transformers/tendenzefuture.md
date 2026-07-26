# Tendenze e limiti

Chiudere un capitolo sui Transformer con le previsioni è un esercizio
rischioso: questo campo brucia le profezie in fretta. Più utile fissare le
direzioni di lavoro visibili oggi, e i problemi aperti che le motivano. Perché
il paradosso è proprio questo: mai un'architettura ha funzionato così bene, e
mai è stato così chiaro quanto costa farla funzionare.

## Dove punta la ricerca

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
- I limiti sono strutturali: costi concentrati, bias dei dati,
  allucinazioni, e una comprensione ancora dibattuta di cosa i modelli
  sappiano davvero.
- Tutti gli ingredienti dei Transformer li hai già studiati in questo libro:
  ciò che è nuovo è la composizione, non i mattoni.
```
