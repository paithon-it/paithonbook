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

(v1-4-0)=

## 1.4.0 · 12 agosto 2026

### Pagine ampliate

- Il libro promette due livelli di lettura e un interruttore per scegliere. La promessa era mantenuta dentro le schede e tradita fuori: didascalie delle figure, riquadri «Da ricordare» e paragrafi di raccordo erano scritti nella lingua del livello Superiore, e sono proprio le cose che non si possono saltare. Undici lettori hanno segnalato centodiciotto punti in cui ci si perde. Sono stati riscritti, e i riquadri «Da ricordare» sono ora sui due livelli come il libro dichiara.
- Tredici figure animate nuove, dove il tempo è il contenuto: le fusioni del BPE che accorciano il testo, k-means che si corregge da solo, la finestra che scorre sul suono e riempie lo spettrogramma, la online softmax di Flash Attention che si ricalibra a ogni blocco, una rete di Hopfield che ripara un ricordo scendendo di energia, la validazione a origine mobile che non guarda mai nel futuro, il collasso della CTC con la controprova di cosa succede invertendo i due passi, il feromone delle formiche che decide la strada corta senza che nessuna l'abbia misurata, il falsario della GAN che raggiunge il vero mentre l'esperto si arrende, la deriva dei dati misurata dalla distanza che le dà il nome, il palo che scorre dietro una finestrella e il movimento che si perde, il sogno di un modello del mondo che si stacca dalla realtà dopo sedici passi, e il cammino dalla baseline all'ingresso lungo cui il gradiente è ancora vivo. Ognuna calcola i propri numeri, e si rifiuta di nascere se smettono di combaciare con quelli del testo.
- La notazione matematica adesso è la stessa in tutto il libro. Il grassetto dice che un oggetto ha più di una componente (una matrice, un vettore), e il tondo che è un numero solo: senza quella distinzione una lettera è ambigua, e il lettore deve indovinare dal contesto a ogni riga. Due capitoli su trentatré la applicavano, ed erano i primi due, quelli che la insegnano; da lì in poi spariva. Sono circa duemila simboli rivisti uno per uno, non sostituiti a macchina: i conteggi, le funzioni e le probabilità restano tondi anche quando si scrivono maiuscoli, ed è la trappola in cui una correzione automatica sarebbe caduta.

### Correzioni

- Il libro è stato riletto per intero da tre lenti indipendenti per capitolo: chi verifica i fatti eseguendo il codice, un lettore di tredici anni che legge solo il livello Elementare, e uno studente magistrale che rifà i conti. Questa voce e le successive dicono cosa ne è uscito. La cosa più importante non è nessuna delle singole correzioni: è che il codice del libro, per la prima volta, è stato eseguito davvero invece che letto.
- Alcuni esempi si smentivano da soli, e si vedeva solo lanciandoli. La frase scelta per dimostrare che l'attenzione capisce la negazione veniva classificata male dal modello che la sezione fa girare; l'esempio che doveva mostrare come il BPE non perde informazione usava lettere fuori dal proprio alfabeto; l'analogia «re meno uomo più donna» restituisce «re», se non si esclude il punto di partenza. Adesso il libro lo dice, e dove il modello sbaglia lo mostra invece di nasconderlo: è più interessante di un successo.
- {doc}`Il filtraggio collaborativo </SistemiRaccomandazione/filtraggio-collaborativo>` (Sistemi di raccomandazione). L'unico esperimento del capitolo misurava l'errore sui dati di addestramento e ne traeva una conclusione sulla capacità di generalizzare. Tenendo da parte un quinto dei voti, l'errore passa da 0,019 a 0,418. Un capitolo che insegna a valutare i modelli non può sbagliare la propria valutazione.
- {doc}`La legge dentro la loss </PINN/come-funziona>` (PINN: reti e fisica). L'esempio del capitolo non convergeva come il testo prometteva, e la cosa si vedeva solo addestrando davvero. Adesso il libro mostra il caso che fallisce: una rete con il residuo più basso di tutte e la soluzione più sbagliata, che è esattamente la lezione che la pagina insegna due paragrafi dopo. Con tre controlli nel codice che se ne accorgono, se un domani i numeri cambiano.
- Errori di fatto e attribuzioni sbagliate, corretti in tutto il libro: le sorelle Wachowski (erano «i fratelli»), le connessioni residue di U-Net che sono della FCN, ImageNet che nel 2009 aveva 3,2 milioni di immagini e non quattordici, il margine massimo che è del 1963 e non dei laboratori Bell degli anni Novanta, il forget gate della LSTM che è del 2000 e non del paper del 1997, BIC e AIC che avevano il segno rovesciato. Quando il libro cita una fonte, adesso quella fonte dice davvero quello che il testo le fa dire.
- I notebook scaricabili dal pulsante «Esegui il codice» non si aprivano: centocinquantuno celle su duecentosessantaquattro davano errore di sintassi, in tutti e ventitré i file. Adesso funzionano.
- Alcune figure raccontavano una storia diversa da quella della pagina che le ospita, e sono il tipo di errore che nessuna ricerca nel testo può trovare, perché non c'è una parola sbagliata da cercare. Una svolgeva 17 × 28 mentre il paragrafo accanto svolge 17 × 24; una disegnava AlphaGo addestrato dal solo self-play, che è il suo successore e non lui; una prometteva un blocco di memoria di cui il testo parlava per cinque righe e che nel disegno non c'era; un albero di decisione arrivava per ultimo alla scelta che il testo dice di provare per prima. Adesso testo e figura dicono la stessa cosa.
- Nel farlo sono venute fuori parecchie lettere che significavano due cose diverse nella stessa pagina, e in un capitolo una che ne significava tre. Dove si poteva le abbiamo rinominate, dove il nome era quello standard della materia lo abbiamo detto al lettore invece di nasconderlo: se un simbolo in un articolo si chiama in un altro modo, adesso il libro lo scrive, così chi va a cercare sa cosa cercare.
- Alcuni numeri che il libro stampava erano quelli di una sola esecuzione, e su altre non reggevano la tesi per cui erano stati messi lì. È un difetto che si nasconde bene, perché rilanciando il programma il numero torna identico: la verifica lo conferma invece di scoprirlo. Dove succedeva, il libro adesso misura su più semi e riporta anche quanto ballano i risultati, che è la cosa che al lettore serve davvero sapere.

(v1-3-0)=

## 1.3.0 · 11 agosto 2026

### Pagine ampliate

- {doc}`GraphSAGE, GAT e applicazioni </GraphNeuralNetwork/architetture-applicazioni>` (Graph Neural Network). I Graph Transformer, cioè cosa succede a lasciar parlare ogni nodo con ogni altro invece che solo con i vicini. Se ne ricava un legame fra due capitoli lontani: il modo in cui qui si segna la posizione di un nodo in un grafo generalizza l'idea con cui i Transformer segnano la posizione delle parole in una frase, e sul grafo più semplice che esista, una fila, le due firme diventano onde imparentate. Non le stesse onde: la pagina mostra anche dove si separano.
- {doc}`Confronto coi modelli precedenti </Transformers/confronti>` (Transformer). Le tre strade per far costare meno l'attenzione, ora tutte e tre: uno schema deciso in anticipo, una scelta guidata dai dati (il Reformer, che cerca le coppie che contano invece di calcolarle tutte) e la rinuncia alla softmax. Più gli strati reversibili, che sono il baratto memoria contro calcolo che ricorre a ogni scala.
- {doc}`Multimodalità </Transformers/multimodalita>` (Transformer). ELECTRA, il quarto modo di studiare accanto a GPT, BERT e T5, che nasce da un'obiezione semplice: se si cancella una parola su sette, per sei parole su sette la rete non impara niente. Con il punto in cui l'analogia con le GAN si rompe, che spiega perché qui non ci sia la loro instabilità.
- {doc}`Tendenze e limiti </Transformers/tendenzefuture>` (Transformer). Perché un Transformer spenda lo stesso calcolo su una domanda facile e su una difficile, e i due modi di togliere quel vincolo: pensare più a lungo in silenzio, o pensare scrivendo. Il primo costa meno, il secondo lascia una traccia leggibile.
- {doc}`Il meccanismo di attenzione </Transformers/attenzione>` (Transformer). Un riquadro su un antenato dimenticato: la struttura domanda-contro-archivio dell'attenzione era già stata inventata a metà degli anni Dieci, per far ragionare una rete su un elenco di fatti. È anche la forma dei sistemi che oggi recuperano documenti prima di rispondere.
- {doc}`Dati e pipeline </MLOps/dati-e-pipeline>` (MLOps). In che formato stanno i dati fra uno stadio e l'altro, che sembra una questione tecnica e non lo è: per addestrare si leggono sempre poche colonne su molte, ed è esattamente il caso in cui conservarle per colonna cambia tutto. Perché il CSV sia quasi sempre la scelta sbagliata, e cosa risolvono Parquet e Arrow.
- {doc}`Overfitting e validazione </MachineLearning/overfitting-validazione>` (Machine Learning). Come si decide se conviene raccogliere altri dati o cambiare modello, che è la domanda più cara di ogni progetto: si tracciano due curve e la loro forma dà la risposta. Più l'Elastic Net, il campionamento stratificato, e il fatto che il test set si sporca anche soltanto guardandolo per decidere come impostare il lavoro.
- {doc}`Valutare un modello </MachineLearning/metriche>` (Machine Learning). Quando la cosa da predire ha un ordine (una fascia d'età, le stelle di una recensione, la gravità di una diagnosi) l'accuratezza non sa che sbagliare di poco non è come sbagliare di molto, e sceglie il modello sbagliato. Quali misure usare al suo posto.
- {doc}`Alberi e metodi ensemble </MachineLearning/alberi-ensemble>` (Machine Learning). Come si combinano modelli di tipo diverso, per voto o addestrando un modello a pesarli. Con un risultato che spiazza e che è misurato nella pagina: il voto peggiora rispetto al miglior modello singolo, e l'alternativa lo supera.
- {doc}`Trovare gli iperparametri </MachineLearning/iperparametri>` (Machine Learning). Perché il torneo a eliminazione descritto qui, su un cluster vero, si faccia in versione asincrona: aspettare che tutti finiscano un turno lascia ferme quasi tutte le macchine.
- {doc}`Backpropagation </RetiNeurali/backpropagation>` (Reti neurali). La ragione vera per cui in classificazione si usa la cross-entropia e non l'errore quadratico, che non è una convenzione: con la seconda il modello che sbaglia di più è quello che impara più lentamente, il che è l'esatto contrario di ciò che serve.
- {doc}`Far funzionare le reti profonde </DeepLearning/ottimizzazione-regolarizzazione>` (Deep Learning). Il riscaldamento del learning rate, che in ogni ricetta di addestramento moderna sta all'inizio e che il libro dava per noto: perché i primi passi siano i più pericolosi, e perché convenga farli piano.
- {doc}`Prestazioni e scala </PyTorch/prestazioni>` (PyTorch). Come si misura davvero il tempo su una GPU. È la trappola in cui cade chiunque la prima volta, perché la GPU non esegue quando glielo si chiede ma quando le viene comodo, e un cronometro ingenuo misura il nulla.
- {doc}`Replicare un paper </PyTorch/replicare-un-paper>` (PyTorch). Il diario degli esperimenti, cioè la metà del lavoro che nessuno scrive. Le tre regole che lo rendono utile, e la cosa che gli strumenti di tracciamento non registrano al posto tuo: perché avevi provato.
- {doc}`Dati su misura </PyTorch/dati-su-misura>` (PyTorch). Perché impacchettare un dataset grande convenga, e non per la ragione che si immagina: il costo dominante non è decodificare i file, è aprirli. Nello stesso passaggio si calcolano le statistiche che serviranno per normalizzare.
- {doc}`Modelli di sequenza </NaturalLanguageProcessing/modelli-sequenza>` (Natural Language Processing). Come si addestra davvero una rete ricorrente su una sequenza lunga, e il prezzo che si paga: spezzandola in blocchi, la rete non può più imparare legami più lunghi del blocco. Parte di ciò che chiamiamo «memoria corta» è una scelta di ingegneria, non un limite matematico.
- {doc}`MDP e funzioni valore </ReinforcementLearning/mdp-valore>` (Reinforcement Learning). Che cosa succede quando l'agente non vede lo stato del mondo ma solo un pezzo rumoroso, che è la regola e non l'eccezione. È il motivo per cui certi agenti impilano gli ultimi fotogrammi e altri hanno una memoria.
- {doc}`Gradiente di policy </DeepReinforcementLearning/policy-gradient>` (Deep Reinforcement Learning). Da dove viene il «guinzaglio» di PPO: è la versione economica di un vincolo preciso, formulato qualche anno prima e troppo costoso da calcolare. Non più corretto, abbastanza corretto e molto più semplice.
- {doc}`Forecasting neurale </SerieTemporali/forecasting-neurale>` (Serie temporali). Una terza famiglia fra i modelli classici e le reti: scomporre la serie in tendenza, stagionalità e festività e interpolare una curva. Su molti problemi aziendali basta, e il motivo per cui funziona è anche il motivo del suo limite.
- {doc}`Parallelismo distribuito </GPU/parallelismo-distribuito>` (GPU e calcolo parallelo). Da che cosa ha preso il posto lo schema ad anello con cui le GPU si scambiano i gradienti, e perché quello precedente non reggeva l'aumentare delle macchine.

(v1-2-0)=

## 1.2.0 · 10 agosto 2026

### Sezioni nuove

- {doc}`Geometria e profondità </VisioneArtificiale/geometria-e-profondita>` (Visione artificiale). Il capitolo sulla visione parlava solo di reti applicate alle immagini, e lasciava fuori la metà geometrica del campo: come si forma un'immagine e come si recupera la distanza che lo scatto ha buttato via. Dal modello di fotocamera dimostrato da Brunelleschi con una tavoletta forata al vincolo epipolare, alla profondità dalla disparità, al flusso ottico, fino alla profondità stimata da una sola immagine. Il codice non illustra, verifica: il residuo epipolare vale un decimillesimo di miliardesimo di miliardesimo.
- {doc}`NeRF e splatting </VisioneArtificiale/rendering-neurale>` (Visione artificiale). NeRF e splatting gaussiano: una scena rappresentata addestrando una funzione invece di ricostruire una superficie. Come il rendering volumetrico differenziabile faccia emergere la geometria dalle sole fotografie, perché senza codifica posizionale esca solo nebbia (è lo stesso limite descritto nel capitolo sulle PINN) e perché le pose delle fotocamere restino un ingresso obbligatorio, che arriva dalla sezione precedente.
- {doc}`Una rete, molti compiti </DeepLearning/multi-compito>` (Deep Learning). Addestrare una rete su più compiti insieme, che il libro faceva già in mezza dozzina di posti senza chiamarlo per nome. Perché un compito in più possa aiutare quello che ci interessa, e perché possa anche danneggiarlo: misurato, un compito imparentato toglie il 65% dell'errore e uno estraneo ne aggiunge il 25%.
- {doc}`Imparare guardando </DeepReinforcementLearning/imitazione>` (Deep Reinforcement Learning). Imparare guardando qualcuno che sa già fare la cosa, che è il pezzo che mancava fra il reinforcement learning e l'addestramento degli assistenti conversazionali. E il suo problema caratteristico: gli errori non si sommano, si compongono, e più il maestro è bravo meno insegna a rimediare, perché non si trova mai nella condizione di doverlo fare.

### Pagine ampliate

- {doc}`Riduzione e clustering </MachineLearning/riduzione-clustering>` (Machine Learning). Le misture gaussiane e l'algoritmo EM, che due capitoli più avanti (i tokenizzatori e il riconoscimento vocale) davano già per noti senza che fossero mai stati spiegati. Assegnare una probabilità invece di un'etichetta, imparare la forma di un gruppo e non solo il suo centro, e scegliere quanti gruppi con un criterio invece che a occhio.
- {doc}`Gradiente di policy </DeepReinforcementLearning/policy-gradient>` (Deep Reinforcement Learning). La ricerca ad albero Monte Carlo, che il libro invocava quattro volte come spiegazione di AlphaGo e MuZero senza averla mai insegnata. Le sue quattro mosse, il perché si sceglie la mossa più visitata e non quella con la media migliore, e il fatto che la formula che le fa scoprire dove guardare è la stessa dei bandit a più braccia.
- {doc}`Le basi di Python </Python/basi>` (Python). Il GIL, cioè perché in Python i thread aiutano quando si aspetta e non quando si calcola. Serviva: il capitolo su PyTorch lo usava due volte come spiegazione di scelte importanti, dando per noto un termine mai introdotto.
- {doc}`Rappresentare il testo </NaturalLanguageProcessing/rappresentare-testo>` (Natural Language Processing). Come si ottiene un vettore per una frase intera, e non solo per le singole parole. Il capitolo si fermava alla parola, e intanto due capitoli sul recupero di documenti poggiavano su modelli di embedding mai spiegati. Con la sorpresa che regge la sezione: un BERT preso così com'è dà vettori di frase mediocri, perché è stato addestrato ad altro.
- {doc}`Architetture storiche </DeepLearning/architetture-storiche>` (Deep Learning). La convoluzione separabile in profondità, cioè il mattone di cui è fatta EfficientNet, che il capitolo raccontava senza averlo mai descritto. Stessa forma in uscita, quasi nove volte meno pesi: è la ragione per cui la visione artificiale sta in un telefono.
- {doc}`Componenti e modelli classici </SerieTemporali/componenti-e-classici>` (Serie temporali). La procedura che mancava: come si scelgono i numeri di un modello ARIMA e come si verifica che vada bene, guardando quello che resta invece delle previsioni. Più i due modi in cui le informazioni esterne entrano in una serie, il meteo e le promozioni da una parte, le serie che si influenzano a vicenda dall'altra.
- {doc}`Alberi e metodi ensemble </MachineLearning/alberi-ensemble>` (Machine Learning). Come si combinano modelli di tipo diverso, per voto o addestrando un modello a pesarli. Con un risultato che spiazza e che è misurato nella pagina: il voto peggiora rispetto al miglior modello singolo, e l'alternativa lo supera.

(v1-1-7)=

## 1.1.7 · 9 agosto 2026

### Correzioni

- Il libro si legge scuro, ma si stampava scuro anche lui: pagine nere, figure in negativo e, con gli sfondi disattivati come fa ogni browser, testo chiarissimo su carta bianca, cioè niente. Adesso la stampa esce sempre chiara, qualunque tema si stia usando, e i due livelli di lettura finiscono tutti e due sul foglio, ciascuno col suo nome sopra: su carta non c'è niente da aprire.
- La riga che dice chi ha pronunciato una citazione era di un grigio troppo chiaro per il fondo su cui stava, sotto il minimo di leggibilità raccomandato. Adesso usa il grigio del libro, in entrambi i temi.

(v1-1-6)=

## 1.1.6 · 9 agosto 2026

### Correzioni

- Le figure erano numerate di fila su tutto il libro, e si era arrivati a «Fig. 302»: un numero che non dice in che punto del libro si stia guardando. Adesso portano il numero del capitolo, «Fig. 3.2» come in un libro, e i rimandi nel testo lo seguono. Il numero del capitolo è lo stesso che si legge nell'indice e in prima pagina, così le due numerazioni non possono discordare.
- La figura ingrandita si apre come sta nella finestra e non già zoomata: prima si vede tutto il disegno, poi si sceglie dove guardare da vicino.

(v1-1-5)=

## 1.1.5 · 9 agosto 2026

### Correzioni

- Le figure del libro sono diagrammi, e dentro hanno del testo: i nomi dei passaggi, le etichette, le formule. Su un telefono quel testo finiva sotto i sei pixel, cioè si vedeva la figura ma non si leggeva, e quando una pagina rimandava a una figura il rimando andava a vuoto. Adesso la figura si tocca e si apre a schermo intero, alla grandezza per cui è stata disegnata: si allarga con due dita, si sposta trascinandola, si chiude con un tocco fuori. Da tastiera fanno lo stesso i tasti più, meno e zero, e si esce con Esc.

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
