# Il contesto è l'interfaccia: la finestra di un agente

Due squadre costruiscono un assistente che risponde ai clienti di un negozio.
Si rivolgono allo stesso identico modello, dallo stesso fornitore, con le
stesse impostazioni. Una ottiene risposte precise e nel tono giusto; l'altra,
risposte vaghe che inventano politiche di rimborso mai esistite. Nessuno ha
addestrato niente: la differenza sta tutta in *cosa* le due squadre scrivono
davanti al modello prima di premere invio, cioè quali istruzioni, quali
esempi, quali documenti e in quale ordine. Il modello è lo stesso: cambia
quello che gli si mette davanti.

Quello che sta «davanti al modello» è il *contesto*, e il suo contenitore è la
finestra di contesto già incontrata nell’{doc}`anatomia di un agente
</Agenti/overview>`: il numero massimo di token che il modello elabora in una
volta, fissato da chi l'ha costruito (per esempio 128.000).

Per chi usa un modello già addestrato senza riaddestrarlo, l'unica leva è il
testo che gli si mette davanti: il contesto è l'interfaccia del modello, il punto
da cui passa ogni istruzione. Nella sezione {doc}`sui grandi modelli linguistici
</Transformers/llm>` questa proprietà ha un nome inglese, *in-context learning*,
apprendimento dal contesto: si descrive il compito nel contesto, magari con due
esempi, e il modello lo esegue senza che i suoi pesi cambino. È un modo di
comandare un programma scrivendo in italiano invece che in codice, e come ogni
comando è potente e fragile: una parola diversa cambia la risposta.

Questa pratica si chiama **context engineering**, ingegneria del contesto:
l'insieme delle strategie con cui si sceglie e si mantiene, a ogni passo, il
testo che entra nella finestra (istruzioni, esempi, documenti recuperati,
cronologia, risultati degli strumenti) {cite}`anthropic2025context`. La
trattazione completa è nel {doc}`capitolo su prompt, contesto e loop
</IngegneriaLLM/overview>`, e in particolare nella {doc}`sezione sul context
engineering </IngegneriaLLM/context-engineering>`: come si progetta il contesto,
come lo si monta entro un budget, quali operazioni lo governano e in quanti modi
si degrada.

Per un agente il problema è più acuto, perché il ciclo osserva → ragiona →
agisci riempie il contesto da sé: ogni pensiero, ogni chiamata a uno strumento,
ogni osservazione di ritorno è testo che si accumula nella *cronologia*, cioè
l'elenco di tutto quel che si è detto e fatto finora. Con
osservazioni di migliaia di token bastano poche decine di passi per occuparne
buona parte. Anche prima del limite la qualità cala, perché la capacità del
modello di ritrovare un'informazione nel contesto diminuisce al crescere dei
token che contiene: Anthropic lo chiama *context rot*, il contesto che marcisce
{cite}`anthropic2025context`. Decidere che cosa tenere e che cosa togliere fa
quindi parte del progetto dell'agente.

## La finestra come budget

La finestra di contesto si misura in token, non in parole né in pagine. La
lunghezza di un token dipende dalla lingua e dal tokenizzatore: in inglese vale
grosso modo una parola, in italiano meno, e una parola ne consuma spesso più di
uno. Ogni modello dichiara il numero massimo di token che riesce a leggere in
una volta.

Riempire la finestra ha tre costi. Il primo è la memoria: per non ricalcolare a
ogni passo ciò che ha già letto, il modello conserva le chiavi e i valori
dell'attenzione di tutto il contesto, il *taccuino* della {doc}`sezione
sull'attenzione in pratica </Transformers/attenzione-in-pratica>`. Il taccuino
cresce linearmente con i token: per il modello da sette miliardi di quella
sezione, nella variante con tutte le teste (MHA), occupa 512 KiB a token. A
8.192 token sono già 4 GiB contro i 13 GiB dei pesi, e a 128.000 token lo spazio
supera di diverse volte il modello stesso. Il secondo è il tempo, perché il
contesto va elaborato per intero prima che compaia il primo token della
risposta. Il terzo è il denaro, perché i fornitori fanno pagare ogni token in
ingresso e in uscita: il *costo per token*, con uno sconto sulla parte già vista
quando si usa la cache dei prefissi, è una delle voci che LLMOps tiene sotto
controllo. Un contesto gonfio comporta una spesa più alta e una risposta più
lenta, e riempire la finestra fino al limite «per sicurezza» è quasi sempre
svantaggioso.

La {numref}`fig-context-window` mostra come la finestra si riempie in una
giornata di lavoro, e due delle sue voci vanno definite. La prima è il *system
prompt*: le istruzioni di fondo che il programma antepone a ogni richiesta, uguali
per tutte, e che l'utente non vede né scrive. La seconda sono le *definizioni
tool*, cioè il catalogo degli strumenti del {doc}`tool use
</Agenti/agenti-e-tool-use>`, con nome, descrizione e argomenti di ciascuno: il
modello lo legge come legge tutto il resto, e quindi lo si paga in token.

```{figure} ../figures/context-window.svg
:name: fig-context-window
:alt: "Una barra orizzontale rappresenta il budget di una finestra di contesto da centoventottomila token, ripartita in cinque segmenti, i primi quattro via via più larghi: il system prompt (circa seimila token), le descrizioni degli strumenti (diecimila), la cronologia della conversazione (trentacinquemila, e cresce a ogni turno), i documenti allegati (cinquantottomila) e, tratteggiato in coda, lo spazio che resta per la risposta: diciannovemila token. In fondo l'avvertenza che i valori sono indicativi."
:width: 92%

La finestra come budget da ripartire. Ogni segmento toglie spazio agli altri,
e l'ultimo (lo spazio per la risposta) è quello che si dimentica di contare
finché il modello non la tronca a metà.
```

La finestra va dunque trattata come un **budget**. I 128.000 token del disegno
(grosso modo un romanzo) sembrano molti finché non li si ripartisce fra cinque
voci in competizione: ogni token assegnato a una è tolto alle altre. Come ogni
budget, si può spendere male: una descrizione di strumento troppo estesa, una
cronologia che nessuno accorcia, dieci documenti recuperati dove ne bastavano
tre. Nessuna di queste scelte è un errore in sé, ma insieme riducono lo spazio
della risposta, l'unica voce che nessuno pensa a contare.

Anche quando lo spazio c'è, riempirlo può danneggiare la risposta: i modelli
usano meglio l'informazione all’inizio e alla fine del contesto e trascurano
quella nel mezzo. L'effetto si chiama *lost in the middle*, dal titolo
dell'articolo in cui Nelson Liu e colleghi lo hanno misurato
{cite}`liu2024lost`. Per un agente la conseguenza è immediata, perché i passi
vecchi della traccia finiscono nel mezzo, fra le istruzioni in testa e l'ultima
osservazione in fondo. La {doc}`sezione sul context engineering
</IngegneriaLLM/context-engineering>` racconta come è stato misurato e come se
ne tiene conto nel montare il contesto.

## La memoria: oltre la finestra

La finestra è la **memoria di lavoro** dell'agente: veloce da consultare,
limitata, e scompare quando la conversazione finisce. Un agente deve però
ricordare anche oltre il singolo scambio: chi è l'utente, che cosa si è detto
ieri, che cosa contengono mille pagine di documentazione che non entrerebbero
mai tutte insieme nella finestra. Serve una **memoria a lungo termine**, che sta
necessariamente *fuori* dal contesto.

`````{tab} Elementare

A mente tieni giusto le poche cose che ti servono *ora*: è veloce, ma ci sta
poco e svanisce. Su un'agenda invece finisce tutto quello che hai annotato nel
tempo: non la leggi tutta insieme, la apri alla pagina giusta quando ti
serve.

Un agente fa lo stesso. Nel breve termine usa un **foglio di brutta** dentro
la finestra: ci scrive i risultati intermedi, i conti a metà, gli appunti del
compito in corso. Nel lungo termine tiene uno **schedario** fuori, e la sua
bravura sta nel pescarne solo la pagina che serve adesso, invece di tenere
tutto aperto sul tavolo (dove non ci starebbe, e dove si perderebbe nel mezzo).

Nello schedario finiscono tre generi di cose, che si recuperano in modi
diversi. I **documenti**, che si vanno a cercare come abbiamo visto parlando di
RAG. I **riassunti** di quello che si è già detto: quando una conversazione si
allunga troppo, invece di portarsela dietro parola per parola se ne tiene un
sunto, che occupa una frazione dello spazio. E i **fatti sull'utente**, cioè le
poche cose che valgono sempre (come si chiama, che lingua parla, cosa ha già
chiesto tre volte), tenute a parte e rimesse davanti al modello quando
c'entrano.

Quando un lavoro è troppo grande per un tavolo solo, se ne passa un pezzo a
un aiutante, che lo sbriga sul suo tavolo e riporta soltanto il risultato: sul
tavolo dell'agente arriva una pagina, non tutte le carte che sono servite a
scriverla.

`````

`````{tab} Superiore

La memoria a breve termine è lo *scratchpad*: uno spazio nel contesto in cui
l'agente scrive i propri stati intermedi (la traccia ReAct del {doc}`ciclo
dell'agente </Agenti/agenti-e-tool-use>` ne è un esempio) e che vive quanto
vive la finestra. La memoria a lungo termine è esterna e persistente, e la
letteratura la organizza a livelli, come la memoria virtuale di un sistema
operativo: in MemGPT {cite}`packer2023memgpt` la finestra è la memoria
principale, l'archivio esterno la secondaria, e il modello stesso decide che
cosa spostare fra le due. Tre forme ricorrono. La prima è il **database
vettoriale**: i ricordi (documenti, scambi passati) vengono codificati in
embedding e recuperati per similarità quando servono; è il RAG dei Transformer,
letto qui come un meccanismo di memoria, non solo di recupero. La seconda è il
**riassunto progressivo**: quando la cronologia della conversazione si allunga,
la si comprime in un sunto che ne conserva l'essenziale a costo di token molto
minore, liberando finestra. La terza sono i **fatti strutturati** (preferenze,
identità, vincoli dell'utente) tenuti a parte e reiniettati quando pertinenti.
Nelle pratiche descritte da Anthropic nel 2025 {cite}`anthropic2025context` le
stesse idee hanno nomi propri: la *compattazione* (riassumere la conversazione
e ripartire con una finestra nuova), gli *appunti strutturati* tenuti fuori
dalla finestra, e i *sotto-agenti*, ciascuno con una finestra sua, che
restituiscono all'agente principale solo un riassunto del proprio lavoro. Il
nodo difficile sta a valle dell'archivio: la **politica di ammissione**, cioè
che cosa di tutto quel materiale merita la finestra a questo passo.

`````

La memoria esterna non ha il limite della finestra, ma ogni voce che rientra
nel contesto ne consuma lo spazio. Scrivere in memoria pone già problemi (che
cosa conservare, come aggiornare un fatto che cambia, quando dimenticare), e a
ogni passo se ne aggiunge un secondo: decidere quale parte del materiale occupa
la finestra *adesso*. Ogni riga dedicata al ricordo sottrae spazio al
ragionamento.

## Pensare costa token: il ragionamento come context engineering

Anche far «ragionare ad alta voce» l'agente prima di agire, cioè fargli scrivere
i passaggi intermedi nel contesto prima della conclusione (la catena di
ragionamento, la chain-of-thought {cite}`wei2022chain`), è ingegneria del
contesto: si spende una parte del budget in token di «pensiero» per ottenere
risposte migliori. Il ragionamento occupa finestra e allunga l'attesa, e il
guadagno misurato si concentra sui compiti matematici e simbolici
{cite}`sprague2025cot`: altrove spesso non ripaga il costo, e va speso dove
serve.

L'idea si può spingere oltre sui problemi che chiedono di pianificare. Invece
di seguire un unico filo fino in fondo, si aprono più strade di ragionamento,
si valuta dove portano e si torna indietro da quelle che non promettono: è il
**Tree of Thoughts** («albero di pensieri») {cite}`yao2023tree`. Il suo banco
di prova più citato è il gioco del 24: quattro numeri da combinare con le
quattro operazioni, usandoli una volta ciascuno, per ottenere 24.

`````{tab} Elementare

Con 4, 9, 10 e 13 bisogna arrivare a 24. Chi va a colpo d'occhio scrive
un'operazione dopo l'altra e spera; se al terzo passo i conti non tornano, il
tentativo è sprecato, e si ricomincia da zero.

Chi va con metodo fa un passo e si ferma a guardare. Prova 13 − 9 = 4, e scrive
su un foglietto che cosa gli resta: 4, 4 e 10. Prova anche 10 − 4 = 6, e su un
altro foglietto: 6, 9 e 13. Per ogni foglietto si chiede se da lì a 24 ci si
arriva: «sicuro», «forse» o «impossibile» (con 1, 1 e 2, per dire, non ci si
arriva in nessun modo). Butta gli impossibili, tiene i più promettenti e
riparte da quelli, un passo per volta; se una strada si chiude, torna al
foglietto di prima e prova un'altra mossa. Da 4, 4 e 10 si arriva a
(10 − 4) × 4 = 24: era la strada buona.

Ogni foglietto in più costa carta e tempo, quindi il confronto va fatto a pari
fatica. Un tentativo solo a colpo d'occhio, sui giochi difficili, va a segno
quattro volte su cento; cento tentativi, contando il gioco come risolto se anche
uno solo va a segno, circa la metà delle volte; il metodo dei foglietti,
tenendone a ogni passo i cinque migliori e con una fatica paragonabile a quei
cento tentativi, tre volte su quattro.

`````

`````{tab} Superiore

Uno stato è $s = [x, z_1, \dots, z_i]$, il problema $x$ con i pensieri
$z_1, \dots, z_i$ prodotti finora, e il metodo ha quattro pezzi
{cite}`yao2023tree`. La *scomposizione* decide che cos'è un pensiero: nel
gioco del 24 un'operazione con i numeri che restano, tre in tutto. Il
*generatore* propone $k$ pensieri successivi a partire da $s$, campionandoli
indipendentemente o, quando lo spazio è stretto, chiedendoli tutti in una
volta con un prompt di proposta, così che non si ripetano. Il *valutatore*,
lo stesso modello, stima quanto prometta ciascuno stato (nel gioco del 24, tre
giudizi «sicuro / forse / impossibile» campionati per ogni stato). La
*ricerca* decide quali stati espandere: in ampiezza (BFS, tenendo i $b$ stati
migliori a ogni livello) o in profondità (DFS, potando lo stato giudicato
senza speranza e risalendo al padre). È la cosa che il filo unico non
permette: abbandonare uno stato che non promette e riprendere da uno
precedente.

I numeri sono di GPT-4 (maggio 2023), su cento giochi difficili di una
raccolta di 1.362. Una sola catena di pensiero ne risolve il 4%, l'albero il
45% con $b = 1$ e il 74% con $b = 5$. Il confronto equo è a token generati
simili: per gioco l'albero con $b = 5$ ne produce circa 5.500 e cento catene
circa 6.700 (stime dell'articolo). Concedendo alla catena di pensiero cento
tentativi e contando il gioco come risolto se almeno uno va a segno, cioè con un
oracolo che sa riconoscere quello giusto, si arriva al 49%, contro il 74%
dell'albero; con $b = 1$ l'albero (45%) sta poco sotto le cento catene, ma
visita molti meno nodi. Il costo cresce in token, in tempo e in denaro.

`````

Il compromesso del context engineering è sempre lo stesso: la finestra è un
budget, ogni elemento che vi entra (istruzioni, esempi, memoria recuperata, il
pensiero stesso del modello) ha un costo, e va collocato dove rende di più. È la
differenza fra le due squadre dell'apertura: che cosa mettere davanti al
modello, e in quale ordine.



`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Con un modello istruito, se non lo si riaddestra, non si programma scrivendo
  codice, ma scrivendo il contesto: quello che gli metti davanti prima di
  fargli la domanda è l'unico comando che hai. Riempire bene quello
  spazio vale più di qualunque «frase magica».
- La finestra è piccola e costosa: ogni parola che ci metti la paghi in
  memoria, in attesa e in denaro, e gli strumenti e la traccia di un agente se
  ne mangiano una parte a ogni passo. Il modello usa bene l'inizio e la fine
  di quello che legge, e trascura il centro (in inglese *lost in the middle*
  {cite}`liu2024lost`): lì finiscono i passi vecchi della traccia.
- Memoria: a breve termine il foglio di brutta dentro la finestra, dove
  l'agente scrive i conti a metà; a lungo termine uno schedario esterno da
  cui pescare solo la pagina che serve adesso (i documenti recuperati, i
  riassunti di quello che si è detto, i fatti sull'utente tenuti a parte), e
  per i lavori grossi un aiutante che riporta solo il risultato. Il problema
  difficile è decidere cosa lasciare fuori dalla finestra adesso: ogni riga
  spesa a ricordare è una riga in meno per ragionare.
- Anche pensare costa: far ragionare il modello a voce alta prima di
  rispondere {cite}`wei2022chain`, o fargli provare più strade e tornare
  indietro da quelle che non promettono (Tree of Thoughts
  {cite}`yao2023tree`), compra qualità spendendo spazio nella finestra. Come
  ogni spesa, va fatta dove rende: il ragionamento scritto aiuta soprattutto
  nei conti e nella logica, e l'albero nei problemi da pianificare, dove batte
  anche cento tentativi alla cieca.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Per chi usa un LLM istruito senza riaddestrarlo, ciò che si mette nella
  finestra prima di chiedere è l'interfaccia. Il context engineering è
  l'insieme delle strategie per scegliere e mantenere a ogni passo i token del
  contesto {cite}`anthropic2025context`. Per un agente la cronologia cresce a
  ogni passo, e la qualità cala con i token anche prima del limite (*context
  rot*).
- La finestra è finita e costosa: ogni token pesa sul taccuino delle chiavi e
  dei valori (lineare nella lunghezza, 512 KiB per token per un modello da
  sette miliardi a 16 bit), sull'attesa del primo token e sul costo per token;
  le definizioni degli strumenti e la cronologia ne occupano una parte che
  cresce a ogni passo. C'è anche il lost in the middle {cite}`liu2024lost`: i
  modelli usano bene l'inizio e la fine del contesto, male il centro, dove
  finiscono i passi vecchi della traccia.
- Memoria: a breve termine lo *scratchpad* nella finestra; a lungo termine
  una memoria esterna, a livelli come la memoria virtuale (MemGPT
  {cite}`packer2023memgpt`), nelle forme del database vettoriale/RAG, dei
  riassunti progressivi (la compattazione), dei fatti strutturati e dei
  sotto-agenti. Il problema difficile sta a valle, nella politica di
  ammissione: che cosa di quel materiale merita la finestra a questo passo.
- Anche il ragionamento è context engineering: chain-of-thought
  {cite}`wei2022chain` e la sua estensione ad albero, il Tree of Thoughts
  {cite}`yao2023tree`, comprano qualità spendendo token di «pensiero». Il
  guadagno della prima si concentra su matematica e logica
  {cite}`sprague2025cot`; il secondo, nel gioco del 24, passa dal 4% al 74% con
  $b = 5$, contro il 49% di cento catene giudicate da un oracolo.
```

`````
