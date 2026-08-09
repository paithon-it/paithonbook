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

(v1-1-4)=

## 1.1.4 · 9 agosto 2026

### Correzioni

- Il libro si leggeva peggio sullo schermo che sulla carta, e su un telefono era peggio che altrove. La misura della riga, che è il numero di caratteri che si attraversano prima di tornare a capo, saltava da ventisei su un telefono a settantacinque su un tablet: nel primo caso quattro parole per riga, nel secondo una riga così lunga che tornando indietro si perde il rigo. Ora il corpo del testo cresce con lo schermo senza scalini, la riga resta fra i trentacinque e i settanta caratteri a ogni larghezza, e l'interlinea segue la lunghezza della riga invece di contraddirla. Cambia solo come il testo è messo in pagina: le parole sono le stesse.
- Nella stessa pagina il testo, i blocchi di codice, le figure e le tabelle partivano da tre margini sinistri diversi, e una figura poteva rientrare di quasi cento pixel rispetto al paragrafo che la annunciava, con la didascalia più larga della figura che descrive. Adesso la colonna di lettura è una sola e vale per tutto quello che ci sta dentro. Su uno schermo da portatile l'indice di destra, quello della pagina corrente, finiva mezzo fuori dallo schermo e si leggeva tagliato a metà parola: ora ci sta.

(v1-1-3)=

## 1.1.3 · 8 agosto 2026

### Correzioni

- L'epigrafe in prima pagina era una frase attribuita a Marcus du Sautoy che si trova solo sugli aggregatori di citazioni: nessuno di loro dice da quale libro o intervista venga, e la traduzione italiana lasciava in dubbio chi guardasse che cosa. Al suo posto la domanda con cui Alan Turing apre il discorso sulle macchine che imparano, riscontrata sul testo di «Computing Machinery and Intelligence» del 1950: la strada per l'intelligenza non è simulare una mente adulta, è simularne una che impara.

(v1-1-2)=

## 1.1.2 · 8 agosto 2026

### Correzioni

- {doc}`Introduzione </Introduzione/overview>`. La massima di Weizenbaum con cui si apre il libro era tradotta alla lettera: «si dice che spiegare significhi spiegare via», che in italiano non vuol dire niente, perché *to explain away* è un verbo frasale che l'italiano non ha. Adesso dice «dissolvere», e una nota riporta l'originale inglese per chi vuole controllare. Il richiamo nelle conclusioni la segue.

(v1-1-1)=

## 1.1.1 · 8 agosto 2026

### Correzioni

- {doc}`Prefazione </prefazione>`. L'ultima riga della prefazione diceva, come nel 2019, «genera paura e alimenta false notizie»; adesso dice «false speranze». Il non sapere come funziona una macchina non produce solo timore: produce anche l'attesa che sappia fare cose che non sa fare. Sono la stessa ignoranza, vista dai due lati.

(v1-1-0)=

## 1.1.0 · 8 agosto 2026

*Il libro ha una prefazione*

È saltata fuori la bozza del 2019, quella mandata a un editore: indice di quindici capitoli e una sola parte scritta, la prefazione. Quasi tutto quello che diceva era già nel libro, sparso fra l'introduzione e la sua conclusione. Mancava la prefazione in quanto tale, cioè l'unica pagina in cui l'autore parla in prima persona.

### Sezioni nuove

- {doc}`Prefazione </prefazione>`. Perché questo libro esiste, perché nel 2019 doveva uscire su carta e non è uscito, e perché è stata una fortuna: quell'indice non nominava i Transformer e prometteva TensorFlow. Si legge in tre minuti, prima dell'introduzione, e non è un capitolo: la numerazione degli altri resta quella di prima.

### Pagine ampliate

- {doc}`Conclusione </Introduzione/conclusione>` (Introduzione). La citazione di Andrew Ng sull'AI come «nuova elettricità» arrivava senza rincorsa. Adesso davanti c'è la genealogia a cui allude: le tecnologie che non risolvono un problema, ma cambiano il modo in cui si risolvono tutti gli altri.

### Correzioni

- I numeri dei capitoli nell'indice di sinistra li prendeva anche chi capitolo non è: la pagina degli aggiornamenti compariva come se fosse il trentaquattresimo. Adesso il numero ce l'hanno i capitoli e basta, ed è lo stesso che portano le schede in prima pagina.
- {doc}`Conclusioni </Conclusioni/overview>`. Le conclusioni dicevano di essere partite da una frase di Weizenbaum sull'intelligenza artificiale «straordinariamente resistente al tentativo di una precisa definizione». L'introduzione però apre su un'altra frase, «si dice che spiegare significhi dissolvere», e della prima non si trova riscontro in nessuna fonte primaria: il richiamo adesso cita l'apertura vera.

### Impianto

- Nell'indice di sinistra ogni capitolo tiene le proprie sezioni chiuse, e il comando per aprirle senza entrare nel capitolo era un'icona di dodici pixel dello stesso colore del testo: c'era, ma non la trovava nessuno, e per sapere che cosa copriva un capitolo bisognava aprirlo e tornare indietro. Adesso è un comando vero, alto quanto la riga e in teal, che si accende al passaggio.

(v1-0-3)=

## 1.0.3 · 8 agosto 2026

### Correzioni

- La storia in fondo a questa pagina dava per esistente un'edizione a stampa, e la usava come metro per dire quali capitoli fossero «aggiunti». Il libro era scritto per uscire su carta, ma su carta non è mai uscito: il metro è il manoscritto del 2019, e la prima forma pubblica di questo testo è quella che si sta leggendo.

(v1-0-2)=

## 1.0.2 · 8 agosto 2026

### Correzioni

- La storia in fondo a questa pagina faceva cominciare il libro nel giugno 2024, che è la data del primo commit: il libro nasce nel 2019, e quello che nasce nel 2024 è la sua versione online. La nota di copyright, che diceva la stessa cosa, adesso parte dal 2019.

(v1-0-1)=

## 1.0.1 · 8 agosto 2026

### Correzioni

- La bibliografia e questa pagina si presentavano ai motori di ricerca e alle anteprime dei link condivisi con il titolo e la descrizione della copertina, come se fossero la prima pagina del libro: adesso ognuna porta i propri. Riguardava le voci di primo livello dell'indice, che sono le uniche a non avere un capitolo sopra di sé.

(v1-0-0)=

## 1.0.0 · 8 agosto 2026

*Il libro prende un numero*

La 1.0.0 non è il primo giorno del libro, che nasce nel 2019 (la storia breve è in fondo a questa pagina): è il primo giorno in cui il libro ha un numero, e il punto da cui si contano i cambiamenti. L'elenco qui sotto è il lavoro di agosto 2026, l'ultimo mese prima che il registro cominciasse.

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

Il libro nasce nel 2019, scritto per uscire su carta con un editore. Non è
successo, e il manoscritto è rimasto in un cassetto: la prima forma in cui
questo testo è arrivato a qualcuno è quella che stai leggendo. Il repository è
più giovane di cinque anni, quindi le date qui sotto non sono la storia del
*libro*: sono la storia della sua **versione online**, cioè i giorni in cui il
testo è arrivato qui dentro.

Il 13 giugno 2024 nasce l'impianto Jupyter Book, con la licenza CC BY-NC-ND e
le prime pagine: l'introduzione, il capitolo su Python e, quattro giorni dopo,
i due livelli di lettura, che sono poi diventati la regola di tutto il resto.
Nell'ottobre del 2025 si allarga il capitolo sui Transformer. Poi si ferma.

Riparte nel luglio del 2026, e in tre settimane diventa un'altra cosa: prima
l'ossatura del manoscritto (matematica, machine learning, reti neurali, deep
learning, visione artificiale, NLP, GAN, reinforcement learning, speech
recognition), poi, uno dietro l'altro, i capitoli che nel 2019 non potevano
esserci, da PyTorch e le GPU fino agli agenti, ai modelli di diffusione, agli
state space model, ai sistemi multi-agente.

Agosto 2026 è il mese della rilettura: ogni capitolo ripassato sui fatti, sui
conti e sul codice, i notebook compagni riallineati alle pagine, le figure che
nessuna pagina richiamava messe da parte, e le sezioni nuove scritte per
chiudere i buchi che la rilettura aveva trovato.

Nessuna di queste tappe ha un numero di versione, e non gliene diamo uno
adesso: non erano pubblicazioni, erano lavoro. Il registro comincia dalla
1.0.0.
