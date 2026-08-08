<!-- GENERATO da scripts/genera-aggiornamenti.py: non modificare a mano.
     La fonte e' book/_dati/aggiornamenti.yml, e la prossima rigenerazione
     cancella qualunque cosa venga scritta qui dentro. -->

(aggiornamenti)=

# Che cosa è cambiato, e quando

Questo libro non esce e finisce: cambia. Una sezione si aggiunge, un conto
sbagliato si corregge, una spiegazione che non ha funzionato si riscrive.
Questa pagina è il registro di quei cambiamenti, dal più recente al più
vecchio, con il link alla pagina toccata. Se hai letto un capitolo un mese fa,
di qui vedi in un minuto se nel frattempo è cambiato qualcosa, e dove.

Non è la storia dei commit, che parla di file e serve a chi scrive il libro:
quella è pubblica e sta [su
GitHub](https://github.com/paithon-it/paithonbook/commits/main). Qui si parla
di quello che legge chi legge.

Se in una pagina trovi un errore, [segnalalo](https://github.com/paithon-it/paithonbook/issues):
le correzioni che arrivano da fuori entrano in questo registro come tutte le
altre, e chi le ha segnalate è citato nel commit che le porta online.

## Come si legge il numero

Il numero della versione è fatto di tre cifre, `impianto.sezioni.correzioni`:

- la **prima** sale quando cambia l'impianto: una parte nuova nell'indice, un
  riordino che cambia il percorso di lettura, la licenza;
- la **seconda** quando arriva un capitolo o una sezione;
- la **terza** quando si corregge o si rifinisce quello che c'è già.

Quando una cifra sale, quelle alla sua destra tornano a zero. Se una
pubblicazione porta sia sezioni nuove sia correzioni, il numero racconta la
cosa più grossa che è successa e l'elenco racconta tutto il resto.

Una versione corrisponde a una **pubblicazione**, non a una giornata di
lavoro: il libro si scrive tutti i giorni e si pubblica quando un pezzo sta in
piedi.

(v1-0-0)=

## 1.0.0 · 8 agosto 2026

*Il libro prende un numero*

La 1.0.0 non è il primo giorno del libro: il primo commit è del giugno 2024 e la storia breve è in fondo a questa pagina. È il primo giorno in cui il libro ha un numero, e il punto da cui si contano i cambiamenti. L'elenco qui sotto è il lavoro di agosto 2026, l'ultimo mese prima che il registro cominciasse.

### Sezioni nuove

- {doc}`La matematica di un LLM </Matematica/matematica-llm>` (Matematica). Gli strumenti dei capitoli di matematica messi all'opera su una cosa sola: un token che entra in un modello linguistico ed esce dall'altra parte. Nasce da un'osservazione di Joseph Breeden, che rileggendo la letteratura sui modelli linguistici scopre che il muro non era la matematica ma il vocabolario preso in prestito da mestieri diversi.
- {doc}`I bandit a più braccia </ReinforcementLearning/banditi>` (Reinforcement Learning). Il dilemma fra esplorare e sfruttare nella forma più pura che esista, dalla domanda che Thompson si pone nel 1933 sulle sperimentazioni cliniche. Arriva prima degli MDP, perché qui manca tutto il resto e si vede solo quello.
- {doc}`I metodi Monte Carlo </ReinforcementLearning/monte-carlo>` (Reinforcement Learning). Imparare giocando fino in fondo, senza la mappa dell'ambiente: il passo che mancava fra la value iteration, che la mappa la pretende, e il Q-learning. Comincia da Ulam e dai suoi solitari a Los Alamos.
- {doc}`I knowledge graph </GraphNeuralNetwork/knowledge-graph>` (Graph Neural Network). Quando gli archi di un grafo non sono collegamenti ma fatti, con un verso e un nome. Da *things, not strings* di Google alle rappresentazioni di entità e relazioni.
- {doc}`Una rete, cento lingue </Transformers/multilingua>` (Transformer). Un modello solo per cento lingue: vocabolari condivisi, trasferimento fra lingue lontane e il prezzo che si paga a metterle tutte nello stesso spazio. Riprende il filo della traduzione neurale, che si era fermato al 2016.
- {doc}`Il conto in energia </MLOps/energia-e-impronta>` (MLOps). Di un modello si dichiara quasi tutto (parametri, FLOP, accuratezza, millisecondi) tranne quanta energia costa. Come si misura, e perché il numero cambia di un ordine di grandezza a seconda di dove lo si misura.

### Pagine ampliate

- {doc}`Message passing </GraphNeuralNetwork/message-passing>` (Graph Neural Network). Da dove viene la matrice di adiacenza normalizzata, e perché l'oversmoothing non è un difetto da correggere ma la conseguenza di come il message passing è fatto.
- {doc}`Confronto coi modelli precedenti </Transformers/confronti>` (Transformer). Il terzo filo del confronto: la self-attention è message passing su un grafo completo, e il capitolo sulle graph neural network e questo si parlano.
- {doc}`L'addestramento avversario </GAN/come-funziona>` (GAN). Come si misura una GAN, che è la domanda che l'addestramento avversario lascia aperta: Inception Score, FID, e che cosa nessuno dei due vede.
- {doc}`GEMM e tensor core </GPU/gemm-e-tensor-core>` (GPU e calcolo parallelo). L'array sistolico: la forma di circuito che sta dentro un tensor core, e perché moltiplicare matrici in hardware somiglia a una catena di montaggio.
- {doc}`I tre errori più comuni </PyTorch/errori-comuni>` (PyTorch). Leggere le curve di addestramento: quali forme dicono overfitting, quali un learning rate sbagliato, quali un errore nei dati.
- {doc}`Pandas e Matplotlib </Python/pandas-matplotlib>` (Python). I quattro tipi di join fra due tabelle e la griglia di subplot, due cose che il capitolo usava senza averle spiegate.
- Ogni capitolo si chiude con un riquadro «Da ricordare», scritto sui due livelli come il resto del libro: cinque righe per rileggere un capitolo in un minuto, o per capire se vale la pena aprirlo.

### Correzioni

- Tutti i capitoli riletti a tre lenti (i fatti sulle fonti primarie, i conti rifatti a mano, il codice eseguito davvero) e corretti dove serviva: date, attribuzioni, esempi numerici, notazione.
- Il codice dei blocchi Python è stato eseguito e i notebook compagni (il pulsante «Esegui il codice») rigenerati dalle pagine corrette: quello che si legge e quello che gira su Colab sono di nuovo la stessa cosa.

### Impianto

- Il tema perde il viola rimasto dalle impostazioni di fabbrica nei fondini delle schede: la palette del libro è quella, e adesso lo è ovunque.
- Questa pagina, e con lei il numero di versione che compare in cima all'indice e nella prima pagina del libro.

## Prima della 1.0

Il libro comincia il 13 giugno 2024, con l'impianto Jupyter Book, la licenza
CC BY-NC-ND e le prime pagine: l'introduzione, il capitolo su Python e, quattro
giorni dopo, i due livelli di lettura che sono poi diventati la regola di tutto
il resto. Nell'ottobre del 2025 si allarga il capitolo sui Transformer. Poi si
ferma.

Riparte nel luglio del 2026, e in tre settimane diventa un'altra cosa: prima
l'ossatura dell'edizione a stampa (matematica, machine learning, reti neurali,
deep learning, visione artificiale, NLP, GAN, reinforcement learning, speech
recognition), poi, uno dietro l'altro, i capitoli che a stampa non c'erano, da
PyTorch e le GPU fino agli agenti, ai modelli di diffusione, agli state space
model, ai sistemi multi-agente.

Agosto 2026 è il mese della rilettura: ogni capitolo ripassato sui fatti, sui
conti e sul codice, i notebook compagni riallineati alle pagine, le figure che
nessuna pagina richiamava messe da parte, e le sezioni nuove scritte per
chiudere i buchi che la rilettura aveva trovato.

Nessuna di queste tappe ha un numero di versione, e non gliene diamo uno
adesso: non erano pubblicazioni, erano lavoro. Il registro comincia dalla
1.0.0.
