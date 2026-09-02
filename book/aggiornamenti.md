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

(v1-12-2)=

## 1.12.2 · 1 settembre 2026

### Pagine ampliate

- {doc}`Dalla softmax alla ricorrenza </AttenzioneLineare/dalla-softmax-alla-ricorrenza>` (Attenzione lineare). **L'errore di richiamo adesso lo calcola la pagina.** I due valori con cui il capitolo misura di quanto una memoria di taglia fissa sbaglia la risposta quando le si scrive dentro troppo erano commentati e non prodotti: c'è ora il blocco che li ottiene, con il proprio seme, e accanto l'andamento con cui si confrontano.
- {doc}`Modelli trasparenti </Interpretabilita/modelli-trasparenti-e-importanza>` (Interpretabilità). **I numeri del confronto adesso li stampa la pagina.** Le due accuratezze su cui si regge il dibattito fra un modello leggibile e uno opaco, e il punto e tre di differenza fra loro, erano scritti a mano accanto a un blocco che stampava altro: adesso ci sono le righe che li calcolano, e con esse le cifre che il verdetto richiede.

### Correzioni

- **Diciotto capitoli riletti per intero.** L'Introduzione, i Sistemi di raccomandazione, la GPU, il Deep learning, l'Efficienza, la Ricerca, l'Apprendimento per rinforzo, l'Attenzione lineare, Visione e linguaggio, il Riconoscimento vocale, gli Agenti, l'Ingegneria dei modelli linguistici, i Modelli a energia, le Serie temporali, l'Auto-supervisione, l'Interpretabilità, le Reti neurali su grafo e le Conclusioni sono passati per le quattro lenti: i fatti storici riaperti sulle fonti originali, le condizioni di una formula provate con un controesempio, le citazioni ricontrollate contro quello che il lavoro citato dimostra davvero, i numeri hardware ripresi dalle schede tecniche, e le scene riscritte perché ogni gesto corrisponda a un passaggio del conto. Con questi ultimi la rilettura ha attraversato ogni capitolo del libro.
- {doc}`Introduzione </Introduzione/overview>`. **Quattro fatti storici rimessi alla loro fonte.** L'epigrafe che apre il libro è ora la prima riga dell'articolo del 1966, presa sul testo originale; la frase di Pascal si legge per intero, e la metà che mancava dice quasi il contrario della prima; ELIZA è il capostipite dei chatbot, non il primo, che è più di quanto le fonti sostengano; e i chilometri di ARGO e del VIAC sono quelli dichiarati da chi c'era.
- {doc}`Robotica e AI </Introduzione/applicazioni>` (Introduzione). **La definizione di robot, con la clausola che tiene fuori la lavatrice.** La norma ISO 8373 è quella rifatta nel 2021, e chiede che la macchina agisca con un certo grado di autonomia: era il punto del capoverso, e mancava. E sulla somma scontata delle ricompense: che siano limitate basta a farla convergere, ma non è necessario, mentre la pagina lo dava per obbligatorio.
- {doc}`Conclusione </Introduzione/conclusione>` (Introduzione). **Il confronto fra la rete e i cardiologi, con il suo protocollo vero.** Come era raccontato girava in tondo: gli stessi medici fissavano il riferimento ed erano poi misurati contro di esso. Nello studio i cardiologi del confronto non sono quelli che hanno annotato il riferimento.
- {doc}`Python e l'AI </Introduzione/massimo_comune_divisore>` (Introduzione). **Il quaderno salva il risultato che il testo commenta.** La cella era archiviata senza la sua uscita, mentre la pagina accanto dice che quel numero è stato stampato: adesso c'è.
- {doc}`Il filtraggio collaborativo </SistemiRaccomandazione/filtraggio-collaborativo>` (Sistemi di raccomandazione). **L'esempio commetteva la fuga di informazione che la pagina denuncia.** Novantuno dei milleduecento voti di prova cadevano su celle già viste in addestramento, e i due errori, sul già visto e sul nuovo, erano mescolati in uno solo: adesso è il blocco a contarli e a tenerli separati. Corretto anche che cosa capita a chi si è appena iscritto: la sua scheda non resta dove il caso l'aveva messa, con i minimi quadrati alternati o con un decadimento dei pesi finisce a zero.
- {doc}`La raccomandazione neurale </SistemiRaccomandazione/raccomandazione-neurale>` (Sistemi di raccomandazione). **Una citazione appesa al contrario della sua fonte.** Il lavoro citato non mostra che addestrare sul futuro gonfi i numeri: mostra che rende imprevedibile l'ordine fra i metodi, ed è a quello che ora è agganciato. Corretto anche il passaggio sulla ricerca veloce: cercare il massimo prodotto scalare non è cercare il vicino più prossimo, e le due cose coincidono solo se gli item hanno tutti la stessa norma.
- {doc}`La raccomandazione neurale </SistemiRaccomandazione/raccomandazione-neurale>` (Sistemi di raccomandazione). **Il conto della NDCG si può rifare fino in fondo.** Gli sconti per posizione erano scritti a metà, e senza quelli che mancavano la somma della classifica ideale non si poteva ricostruire. E la figura della vetrina è stata ridisegnata dove il disegno non mostrava il sorpasso che la didascalia promette.
- {doc}`Dentro la GPU </GPU/architettura-gpu>` (GPU e calcolo parallelo). **Il conto che la pagina invita a fare a mente dava la metà.** Ogni unità esegue una moltiplicazione-e-somma per battito, che di conti ne vale due, e senza quella precisazione il lettore arrivava a metà dei mille miliardi di operazioni che le schede tecniche dichiarano. Rifatta anche la figura degli ordini di grandezza, ferma ai numeri di una generazione prima.
- {doc}`La memoria </GPU/gerarchia-memoria>` (GPU e calcolo parallelo). **Due nomi per lo stesso ripiano, e uno era già di un altro.** La cache di secondo livello si chiamava in due modi in due sezioni, e uno dei due era il nome dato altrove a quella di primo livello: il lettore metteva al piano sbagliato la mensola che regge il capitolo. E i dati che traboccano dai registri non rubano lo spazio condiviso della squadra, come diceva la scena.
- {doc}`FlashAttention </GPU/flash-attention>` (GPU e calcolo parallelo). **Otto non erano le teste.** In quel modello le teste che leggono sono trentadue, e otto sono i gruppi che si dividono le chiavi e i valori: il conto della memoria torna solo con la seconda lettura. E i gigabyte del risultato sono tredici e non dodici, perché la riga accanto li contava in base due mentre il capitolo dichiara la base dieci.
- {doc}`GEMM e tensor core </GPU/gemm-e-tensor-core>` (GPU e calcolo parallelo). **Confronti fra grandezze che non erano confrontabili.** Il rapporto fra conti e byte vale un quarto o la metà secondo il formato dei numeri, e il riquadro aveva perso la clausola; un altro paragone metteva a fianco una misura in singola precisione e una soglia in mezza precisione. E il processore dedicato del confronto sull'energia è quello di prima generazione, a interi e per la sola inferenza.
- {doc}`Imparare a imparare </DeepLearning/meta-apprendimento>` (Deep Learning). **La media della famiglia era nella famiglia.** In quattro punti la pagina diceva che la sinusoide media non è una delle sinusoidi ammesse. Il conto dice il contrario, e i suoi due parametri cadono dentro gli intervalli che la pagina stessa dichiara.
- {doc}`Far funzionare le reti profonde </DeepLearning/ottimizzazione-regolarizzazione>` (Deep Learning). **Il decadimento dei pesi dipende dal passo, anche nella versione che li separa.** L'articolo lo scrive moltiplicato per il moltiplicatore dello schedule, e il testo lo dava indipendente: con la correzione cade il fattore dieci che ne discendeva. La riserva sul label smoothing, che su due banchi di prova non guadagna niente, sta ora in tutti e due i livelli, e la figura non disegna più la spiegazione che la pagina smonta.
- {doc}`Architetture storiche </DeepLearning/architetture-storiche>` (Deep Learning). **Un confronto che nessuna fonte sosteneva.** Che una rete residuale di pari accuratezza resti più leggera da addestrare non lo dicono né gli autori né il lavoro citato, e misurato in due modi indipendenti il verso è l'opposto. Rimessi a posto anche i conteggi dei parametri di Inception, e la descrizione testuale della figura, che portava una versione più vecchia di quella della pagina.
- {doc}`Reti convoluzionali (CNN) </DeepLearning/reti-convoluzionali>` (Deep Learning). **L'equivarianza viene dalla condivisione dei pesi, e da quella sola.** Era attribuita a due vincoli insieme, mentre uno strato che guarda una finestra piccola con pesi diversi in ogni posizione equivariante non è, e la pagina lo diceva già giusto poco più su. E il pooling non è solo il massimo: il libro usa anche la media, in un punto che viene prima della definizione.
- {doc}`Un modello che imita </Efficienza/un-modello-piccolo-che-imita>` (Efficienza). **La distillazione riletta sul lavoro originale.** La forma semplice che la pagina mostrava non è implementabile così com'è, e l'equivalenza che la giustifica chiede due condizioni e non una: temperatura alta rispetto ai punteggi grezzi, e punteggi a media nulla su ciascun esempio. Sono dette tutte e due, e il blocco della pagina mostra di quanto si è fuori da quel regime.
- {doc}`Starci non è rispondere </Efficienza/far-rispondere-in-fretta>` (Efficienza). **Quel rapporto non dipende dalla larghezza dello strato solo perché conta i byte dei pesi.** Ingressi e uscite non erano contati, e l'approssimazione regge finché le cose lavorate insieme sono molte meno dei pesi di una riga: all'ultima riga della tabella sbaglia già del dodici per cento, e adesso la pagina lo dichiara.
- {doc}`Le tre cose date per scontate </Ricerca/quando-il-mondo-non-si-conosce>` (Ricerca e pianificazione). **La scena non usciva mai dalla prima mossa.** Raccontava di provare partite a caso e tenere il conto, che è il metodo del 1993, mentre la pagina prometteva la ricerca ad albero: adesso la scena scava, e ogni partita nuova riparte dal fondo del ramo già scavato. E il punteggio della sfida del 1997 è detto per intero, con le patte che valgono mezzo punto a testa.
- {doc}`Ricerca e pianificazione </Ricerca/overview>`. **Profondità media, non massima.** Il riquadro riusava per la lunghezza tipica di una partita il simbolo che dieci righe più su indicava la profondità massima dell'albero, e la fonte parla della media. Corretta anche la descrizione testuale di una figura, che dava a una riga un colore che nel disegno non ha.
- {doc}`Reinforcement Learning </ReinforcementLearning/overview>`. **La manopola dello sconto diceva il contrario nei due livelli.** La scena la descriveva come se al minimo rendesse l'agente lungimirante, mentre la trattazione formale della stessa pagina dice miope: adesso dicono la stessa cosa. E tenere quella manopola sotto uno non è obbligatorio nei compiti senza fine, è la strada consueta.
- {doc}`I bandit a più braccia </ReinforcementLearning/banditi>` (Reinforcement Learning). **«Nessun'altra farà meglio» era smentito quattrocento righe più sotto, nella stessa pagina.** Un'altra strategia arriva al 91,8 per cento contro l'86,6, e il riquadro in fondo la chiamava già la migliore della sezione: l'affermazione ora dice fra quali strategie vale. Il confronto finale gira su tre sorteggi invece di due, perché con due la differenza commentata era più piccola di quanto il numero balla cambiando seme.
- {doc}`I metodi Monte Carlo </ReinforcementLearning/monte-carlo>` (Reinforcement Learning). **La proprietà attribuita alla stima sbagliata.** Il primo dei due modi di pesare le partite prese in prestito coglie il valore giusto in media per qualsiasi numero di partite: illimitata è la sua varianza, non la sua distorsione. E la contrapposizione fra i due vale solo contando la prima visita a ciascuno stato: contandole tutte sono distorti tutti e due.
- {doc}`MDP e funzioni valore </ReinforcementLearning/mdp-valore>` (Reinforcement Learning). **Un rimedio appeso alla scena in cui non funziona.** Guardare gli ultimi cinque o sei fotogrammi serve quando a mancare è il movimento, non quando manca un'informazione che nessun fotogramma mostra: adesso è detto dove vale. E le caselle più lontane dall'obiettivo sono due, non una, come si conta sul disegno.
- {doc}`Q-learning e differenze temporali </ReinforcementLearning/q-learning>` (Reinforcement Learning). **Un numero che il lettore non poteva rifare.** «Una partita su quattro» richiedeva di sapere che il caso sceglie fra quattro direzioni, cosa che la pagina non diceva: chi rifaceva il conto ovvio otteneva il sessantacinque per cento. E la griglia del precipizio adesso dichiara le proprie dimensioni, senza le quali né i passi in più né le caselle affacciate sul vuoto si contano.
- {doc}`Dalla softmax alla ricorrenza </AttenzioneLineare/dalla-softmax-alla-ricorrenza>` (Attenzione lineare). **Due numeri che nella stessa pagina facevano due mestieri.** Nella scena che spiega perché il peso si spezza in due, i tre contributi comparivano prima grezzi e poi, dopo che il testo aveva detto di averli già moltiplicati per il proprio cartellino, tornavano identici: chi rifaceva la somma contava la moltiplicazione una volta sola. E il lato della memoria, trentadue, era prima il punto in cui lo sbaglio diventa grande quanto la risposta e sei righe dopo quante etichette quella memoria tiene separate: adesso viene prima la capienza, poi quello che la erode.
- {doc}`Scrivere nella memoria </AttenzioneLineare/scrivere-nella-memoria>` (Attenzione lineare). **Un'invenzione attribuita a chi l'ha ritrovata.** Il riquadro finale dava i *fast weight programmer* a Schlag e colleghi, che nel 2021 ci hanno riconosciuto dentro l'attenzione lineare: a progettarli era stato Schmidhuber negli anni Novanta, come il capitolo dice già quaranta righe sopra. E la forma WY non è il riassunto che passa da un blocco al successivo, che è lo stato: è il modo compatto di scrivere il prodotto delle correzioni dentro un blocco.
- {doc}`Scrivere nella memoria </AttenzioneLineare/scrivere-nella-memoria>` (Attenzione lineare). **Un piazzamento al posto di un meccanismo.** «Corregge meglio delle altre memorie di taglia fissa e si avvicina all'attenzione» è un confronto, e vale finché non lo battono. Adesso la pagina dice quello che resta vero: a parità di taglia della memoria, correggere batte accumulare e sbiadire, ma il vantaggio non cresce insieme al modello, perché su memorie grandi conta più quanto sono capienti che come le si aggiorna.
- {doc}`Architetture lineari </AttenzioneLineare/architetture-lineari>` (Attenzione lineare). **Tre conteggi presi dal riassunto invece che dal lavoro.** I modelli RWKV-7 rilasciati con i pesi aperti sono sette, non quattro: quattro sono quelli che l'abstract nomina. I limiti della LSTM che gli autori di xLSTM elencano sono tre, non due, perché il libro fondeva la capienza con l'impossibilità di riempire la memoria tutta insieme. E xLSTM-7B non è la prova che la formula regga alla scala dei grandi modelli: a quella taglia va alla pari con i modelli confrontabili, che è quanto la fonte sostiene.
- {doc}`Conclusioni </Conclusioni/overview>`. **La mappa che smentiva la propria didascalia.** Nel grafico che dispone i modelli su due assi, come sono messi a disposizione e quanto sono capaci, i quattro punti più in alto stavano tutti dalla parte dei pesi scaricabili, mentre la didascalia dice che aperto non vuol dire debole e chiuso non vuol dire potente. I punti sono inventati apposta, e a essere sbagliato era il disegno: adesso mostra quello che la pagina sostiene.
- {doc}`Conclusioni </Conclusioni/overview>`. **Due affermazioni che il resto del libro smentiva.** «Sono tutti problemi di ottimizzazione» arrivava tre righe sotto l'elenco dei metodi che un obiettivo globale non ce l'hanno, come DBSCAN, il clustering gerarchico e gli alberi di decisione: adesso dice quasi tutti, e aggiunge che in fondo alla lista cambia perfino se un obiettivo ci sia. E la discesa del gradiente si ferma dove il gradiente si annulla senza garanzia che quello sia il minimo globale, non «e non nel minimo globale»: una rete grande quel minimo lo raggiunge, ed è il punto da cui parte la doppia discesa.
- {doc}`Conclusioni </Conclusioni/overview>`. **Tre numeri attribuiti a qualcuno, senza il lavoro da aprire.** Gli esponenti delle leggi di scala erano detti «degli autori stessi di quelle misure», e nelle ottanta righe che li presentano non compariva nessuna citazione. Adesso c'è, accanto alla figura e dentro il corpo del testo, che è dove un rimando regge anche in stampa.
- {doc}`Conclusioni </Conclusioni/overview>`. **Due rimandi che promettevano una cosa e ne mantenevano un'altra.** Il richiamo all'introduzione parlava di «coprire una parola e chiedere di indovinarla», mentre là si nasconde la parola che viene dopo, che è un compito diverso; e aggiungeva che con un pezzo di immagine funziona uguale, cosa che l'introduzione non fa, perché coprire un pezzo di fotografia serve a scoprire che cosa ha deciso una risposta. L'altro chiamava capitolo una sezione, e chi lo seguiva cercava nell'indice generale.
- {doc}`Conclusioni </Conclusioni/overview>`. **Un manuale consigliato dato per intero a un'altra libreria.** Dei due manuali per cominciare era detto che il codice è in Keras: vero da dove arrivano le reti neurali, mentre la prima metà di Géron sta su scikit-learn, la stessa libreria di queste pagine. Adesso l'avvertenza dice dove passa il confine.
- {doc}`Allineare due spazi </VisioneLinguaggio/allineare-due-spazi>` (Visione e linguaggio). **Un argomento falso sotto una conclusione giusta.** La perdita contrastiva era detta insensibile a qualunque spostamento in blocco di una delle due nuvole, «perché dipende solo dai rapporti fra le similarità»: nessuna delle due cose è vera, e la pagina stessa, cinque righe più giù, racconta che forzando a mano la sovrapposizione il costo sale. Quello che la perdita non vede è una costante aggiunta dentro una riga, mentre scalare tutte le somiglianze è esattamente il mestiere della temperatura. La conclusione, che il minimo non è la sovrapposizione, regge: adesso a sostenerla è un argomento vero.
- {doc}`Allineare due spazi </VisioneLinguaggio/allineare-due-spazi>` (Visione e linguaggio). **Il riassunto e la figura dicevano un'altra cosa dalla pagina.** Il riquadro finale prometteva che la coppia giusta è più vicina di qualunque altra coppia foto-frase, mentre il confronto vale dentro una riga sola, ed è la soglia globale contro cui la sezione mette in guardia due paragrafi prima; e dava la temperatura per «appoggiata al tetto» a $0{,}01$, che è invece il suo valore minimo, perché al tetto ci sta il suo reciproco. Nella figura, poi, la tabella dei punteggi era etichettata con la lettera che il capitolo usa per contare le tessere di un'immagine.
- {doc}`Innestare gli occhi </VisioneLinguaggio/innestare-gli-occhi>` (Visione e linguaggio). **Un numero giusto, letto come un'altra grandezza.** Comprimere l'immagine da seicento pezzi a cinquantadue fa centotrenta volte meno confronti, e la pagina ne ricavava un risparmio di lavoro altrettanto grande. A seicento pezzi i confronti sono meno di un quarantesimo del conto: il lavoro totale cala di undici volte e mezzo, cioè quanto la sequenza, e il quadrato prende il sopravvento solo a decine di migliaia di pezzi. Due sezioni vicine misuravano lo stesso conto con due metri diversi.
- {doc}`Innestare gli occhi </VisioneLinguaggio/innestare-gli-occhi>` (Visione e linguaggio). **Una figura che diceva un numero a chi la guarda e un altro a chi la ascolta.** Il testo alternativo del disegno dei connettori dava al connettore quattro milioni di parametri, che era la cifra della prima versione del modello, mentre il disegno e la didascalia ne dicono ventuno. Chi legge il libro con un lettore di schermo riceveva il numero sbagliato attaccato al disegno giusto.
- {doc}`Il costo del dettaglio </VisioneLinguaggio/risoluzione-e-dettaglio>` (Visione e linguaggio). **Un riassunto rimasto alla versione che la pagina aveva corretto.** I due «Da ricordare» dicevano ancora che riordinare le tessere non perde un grammo e non butta via niente, mentre poche righe sopra il testo spiega che il posto in cui farle stare ha una misura, e che quando è più stretto è quella matrice a decidere che cosa passa. Il riquadro promette adesso quello che la pagina dimostra.
- {doc}`Agenti </Agenti/overview>`. **Due livelli che si contraddicevano, e una data che il capitolo stesso smentiva.** Della capacità di seguire istruzioni scritte nel prompt era detto che a rendere possibili gli agenti sono state quelle righe «e non una macchina più potente», mentre dieci righe sotto la stessa pagina spiega che quella capacità è emersa con la scala. E «fra il 2023 e il 2024 compaiono i sistemi che decidono da soli la mossa successiva»: ReAct, il primo di quei sistemi, è dell'ottobre 2022, come il capitolo dice poche pagine dopo. Quei due anni dicono adesso quando si sono diffusi.
- {doc}`Il ciclo dell'agente </Agenti/agenti-e-tool-use>` (Agenti). **Due difetti nel codice che il capitolo insegna a evitare.** Il calcolatore protetto accettava `True` come se fosse un numero, perché in Python un valore di verità è un intero a tutti gli effetti; e il modello finto che pilota il giro non aveva un ramo per la ricerca andata a vuoto, quindi tirava dritto e inventava la risposta, che è esattamente il comportamento contro cui la sezione mette in guardia. Adesso il primo rifiuta e il secondo dice che non lo sa.
- {doc}`Il ciclo dell'agente </Agenti/agenti-e-tool-use>` (Agenti). **Un guadagno dato per netto, e un banco su cui non c'era.** Della tecnica che fa rileggere all'agente i propri errori era detto che migliora la quota di problemi risolti al primo tentativo, senza eccezioni. Su un banco di problemi di programmazione quella quota scende invece di salire, e adesso la pagina lo dice, con i due numeri accanto.
- {doc}`RAG avanzato </Agenti/rag-avanzato>` (Agenti). **Una soglia che nel caso peggiore lasciava passare tutto.** Il filtro teneva i passaggi che arrivano almeno a metà del punteggio migliore: quando nessun passaggio è pertinente i punteggi valgono tutti zero, e metà di zero non esclude nessuno. Il criterio ha adesso anche un minimo assoluto, così una ricerca a vuoto resta a mani vuote.
- {doc}`Architetture e valutazione </Agenti/architetture-e-valutazione>` (Agenti). **Una fonte usata per metà, nella sezione che dice di collaudare i collaudi.** Dopo i numeri sui difetti del banco di prova per la programmazione, la pagina presentava la versione ripulita come il rimedio; ma il riesame da cui quei numeri vengono dice, due righe più in là nel suo stesso testo, che gli stessi difetti si ritrovano anche lì. Adesso lo dice anche il libro.
- {doc}`Il contesto è l'interfaccia </Agenti/context-engineering>` (Agenti). **Una lettera con tre mestieri, e un testo alternativo che descriveva un disegno diverso.** La stessa $p$ era la probabilità di sbagliare un passo, il passaggio candidato in una formula, e nel codice il tasso di successo, cioè il proprio complemento a dieci righe dalla formula che la usa. E la descrizione della figura del contesto dava i cinque segmenti «via via crescenti», mentre l'ultimo, lo spazio lasciato alla risposta, è più stretto del precedente.
- {doc}`Memoria associativa </ModelliEnergia/memoria-associativa>` (Modelli a energia). **Il codice non faceva quello che la pagina dichiarava, e sei numeri ne dipendevano.** In caso di parità il neurone doveva restare com'è, ma il confronto con lo zero era esatto, e i legami della rete sono frazioni che in virgola mobile non tornano mai esattamente zero: la regola dichiarata non si applicava quasi mai, e la curva pubblicata dipendeva dall'aritmetica della macchina invece che dalla dinamica descritta. Riparato il confronto, le sei misure sono state rifatte su tre semi, e una di esse ha cambiato anche l'ordine: fra i due modi meno frequenti di sbagliare il richiamo, il più comune è l'immagine capovolta.
- {doc}`Memoria associativa </ModelliEnergia/memoria-associativa>` (Modelli a energia). **Due attribuzioni rimesse al testo che le contiene.** La capienza critica di una rete di Hopfield, $0{,}138$, era data per un raffinamento arrivato due anni dopo l'articolo del 1985, che avrebbe scritto $0{,}14$: è il contrario, il $0{,}138$ sta in quell'articolo, sotto la formula e nei due grafici, e il $0{,}14$ ne è l'arrotondamento del sommario. E la rete di Little del 1974 non accende i neuroni oltre una soglia netta: ciascuno si accende con una probabilità che cresce con la spinta, ed è quel carattere aleatorio il suo contributo.
- {doc}`Memoria associativa </ModelliEnergia/memoria-associativa>` (Modelli a energia). **Una figura che contava cinque cambiamenti dove il testo ne annuncia sei.** Nel disegno della rete che si raddrizza da sola, l'ultimo pannello era etichettato «nessun neurone vuole più cambiare», mentre proprio lì avviene l'ultimo capovolgimento e l'energia fa il suo salto finale. Chi contava le etichette non ritrovava il numero della pagina.
- {doc}`Modelli a energia </ModelliEnergia/overview>`. **Una frase fra virgolette che il documento citato non contiene.** La seconda delle quattro rinunce era riportata come una citazione dal documento di posizione del 2022, che quella frase non ha: la tesi c'è, ma detta altrimenti, e la formulazione secca viene dalle conferenze. Adesso il testo dice la tesi e attribuisce al documento l'argomento disteso, che è quello che ci si trova davvero.
- {doc}`Oltre la partizione </ModelliEnergia/oltre-la-partizione>` (Modelli a energia). **Il ponte fra le due immagini del capitolo, che non era mai stato gettato.** La stessa quantità compare prima come la pioggia che cade in totale e poi come un'altezza aggiunta in ogni punto del paesaggio, e il passaggio dall'una all'altra era dato per noto: chi provava a verificarlo con un conto giusto arrivava alla conclusione opposta, perché dividendo la pioggia le differenze si schiacciano invece di spostarsi tutte uguali. Il cambio di lingua adesso è scritto: scendere di un gradino non aggiunge pioggia, la moltiplica.
- {doc}`Paesaggi di oggi </ModelliEnergia/paesaggi-di-oggi>` (Modelli a energia). **Un primato e una formula data per nota.** Alzare l'esponente dell'energia di Hopfield non è un'idea del 2016: quel lavoro riporta nel deep learning un filone di fisica statistica della fine degli anni Ottanta, che cita esso stesso. E la formula da cui tutto il ragionamento parte, quella in cui ogni ricordo abbassa l'energia in proporzione al quadrato della somiglianza, non era scritta in nessun punto del capitolo: adesso c'è, dove nasce, subito dopo la regola che costruisce i legami.
- {doc}`Componenti e modelli classici </SerieTemporali/componenti-e-classici>` (Serie temporali). **Una conclusione che non seguiva dal numero appena stampato.** Da $2{,}49$ a $1{,}49$ il testo leggeva «un terzo in meno»: la riduzione è del quaranta per cento, cioè due quinti, e un terzo in meno avrebbe dato $1{,}66$. Il numero era giusto, era la frase accanto a non seguirne.
- {doc}`Componenti e modelli classici </SerieTemporali/componenti-e-classici>` (Serie temporali). **Una figura che scriveva l'uguale fra pannelli disegnati a scale diverse.** La scomposizione di una serie nei suoi tre ingredienti è aritmeticamente esatta, ma ogni pannello del disegno è alla propria scala verticale, e così tendenza, stagione e residuo sembrano pesare quasi uguale mentre nella serie in alto la stagione vale circa tre volte e mezzo il residuo, come la pagina accanto dice. La didascalia adesso dichiara la scala, e ridisegnare i pannelli a scala comune resta la riparazione migliore.
- {doc}`Componenti e modelli classici </SerieTemporali/componenti-e-classici>` (Serie temporali). **Un primato che il capitolo stesso smentiva mille righe dopo, e un vincolo troppo largo.** Il metodo di Box e Jenkins non è la prima procedura applicabile a una serie qualunque: il lisciamento esponenziale lo è, e il capitolo lo data al 1957. Loro hanno per primi la procedura con cui si *sceglie* il modello, ed è quello che la pagina dice adesso. E i tre fattori di lisciamento non stanno tutti e tre fra zero e uno indipendentemente: nella forma qui scritta il terzo è legato al primo, e con un primo alto il valore che il libro ammetteva era vietato.
- {doc}`Serie temporali </SerieTemporali/overview>`. **Una somma che, come era scritta, non poteva convergere.** Guardando più giorni avanti l'incertezza cresce perché le sorprese non ancora osservate «si sommano»: sommate nude sarebbero una quantità che cresce senza fermarsi, mentre la banda di previsione di una serie stazionaria si assesta. A sommarsi sono i loro contributi, pesati da coefficienti che si smorzano. Detto in due punti, era da riparare in tutti e due, o la pagina si sarebbe contraddetta.
- {doc}`Validazione e feature </SerieTemporali/validazione-e-feature>` (Serie temporali). **La ragione sbagliata per un degrado vero, e una misura messa nella colonna sbagliata.** Le bande di previsione perdono copertura mano a mano che si guarda lontano, e il testo lo spiegava con l'incertezza dei passi intermedi che si accumula: rifatto l'esperimento regalando al modello i parametri veri, la copertura resta quella dichiarata tanto a un passo quanto a cinque, perché quelle sorprese stanno già dentro la banda. A degradarla è l'errore sui parametri stimati. E la copertura empirica compariva in un elenco di misure da rendere piccole, mentre non va né minimizzata né massimizzata: deve coincidere con il livello dichiarato.
- {doc}`Forecasting neurale </SerieTemporali/forecasting-neurale>` (Serie temporali). **Una percentuale che dai numeri della pagina non tornava.** «Al quinto giorno la banda ha già raggiunto il 99,7% del suo limite» stava due righe sotto una banda stampata di $3{,}166$ e un limite di $3{,}204$, che danno il 98,8: il 99,7 è il rapporto calcolato sulla banda teorica, non su quella dell'esperimento. Adesso la frase dice di quale delle due parla.
- {doc}`Collasso e misura </AutoSupervisione/collasso-e-misura>` (Auto-supervisione). **Un tetto calcolato sul numero sbagliato di candidati.** Il limite di informazione che un metodo contrastivo può garantire vale $\log_2$ dei candidati, e con un lotto da quattromila immagini i candidati non sono quattromila: sono le ottomila viste meno sé stessi, $8191$. Il tetto è tredici bit e non dodici, e il capitolo sulla visione artificiale lo scriveva già per esteso: due pagine del libro dicevano due numeri diversi.
- {doc}`Le quattro famiglie </AutoSupervisione/famiglie>` (Auto-supervisione). **Una quantità detta ferma che invece sale, e oltre il suo tetto.** Nel blocco che mostra come si scoraggia il collasso, la diagonale della matrice era descritta come immobile al valore massimo che il rumore consente. Portando l'esperimento a ventimila passi cresce a ogni controllo e finisce sopra quel massimo, il che una fluttuazione non fa: è il proiettore che si adatta agli esempi che ha davanti. Adesso il codice rimisura sugli esempi mai visti, dove il valore torna dov'era, e la pagina lo commenta.
- {doc}`Capire è accorciare </AutoSupervisione/capire-e-accorciare>` (Auto-supervisione). **Un teorema che dice una cosa più debole di quella per cui era citato, e un intervallo più stretto del misurato.** Il teorema di codifica di sorgente riguarda la lunghezza *attesa* e dà una disuguaglianza: a raggiungerla su una stringa data è la codifica aritmetica, come dice il lavoro che la pagina cita. E i tre coefficienti riportati «fra $-0{,}94$ e $-0{,}95$» valgono $-0{,}935$, $-0{,}937$ e $-0{,}953$: due dei tre stavano fuori dall'intervallo dichiarato, che adesso si arrotonda verso l'esterno.
- {doc}`Capire è accorciare </AutoSupervisione/capire-e-accorciare>` (Auto-supervisione). **Una garanzia data per certa e ribaltata centotrenta righe dopo.** Comprimere due cose insieme non costa mai più che comprimerle separate: la scena e i due riquadri lo davano per acquisito, mentre in fondo alla pagina è detto che vale per il compressore ideale, cioè è un'ipotesi e non un risultato. Una condizione che ribalta una promessa entra anche nel riassunto.
- {doc}`Auto-supervisione </AutoSupervisione/overview>`. **Un riassunto che attribuiva a un'immagine i numeri di un testo.** Il riquadro finale dava i bit di una mezza fotografia dove il programma della pagina misura una finestra di parole, e le due grandezze differiscono di un fattore quattro. Il corpo del testo lo diceva già giusto: si era scollato il riquadro.
- {doc}`Il dibattito sul rinforzo </AutoSupervisione/dibattito-rl>` (Auto-supervisione). **Una figura che contava male i propri passi, e una spinta che è un coefficiente.** Il testo alternativo del disegno sul credito dichiarava quattro passi sbagliati mentre il disegno ne mostra cinque, e adesso quei conteggi li calcola il generatore invece di essere ribattuti a mano. Nella stessa sezione, tutti i passi di una prova andata bene non ricevono «la medesima spinta»: ricevono lo stesso coefficiente, e la spinta che ne segue è diversa per ciascuno.
- {doc}`Spiegazioni locali </Interpretabilita/spiegazioni-locali>` (Interpretabilità). **Tre affermazioni riportate a quello che la fonte dice.** La distanza con cui si cerca la spiegazione controfattuale non pesa ogni caratteristica con la sua dispersione tipica: la **divide** per quella, ed è il verso che rende confrontabili grandezze con scale diverse. Shapley, nel 1953, fissò tre requisiti e non quattro, che sono la riscrittura moderna equivalente. E gli studenti dell'esperimento sulla fiducia mal riposta erano già laureati, cosa che in italiano «studenti universitari» non dice.
- {doc}`Attribuzione e meccanicistica </Interpretabilita/attribuzione-e-meccanicistica>` (Interpretabilità). **Due giudizi più netti di quelli dei lavori citati.** Chi ha misurato i pesi di attenzione con gli stessi criteri non li assolve: scrive che si comportano meglio, che è una differenza di grado e non un verdetto. E il metodo che secondo il libro «fallisce del tutto» il controllo, anche su una rete a cui è stato azzerato tutto, non lo fallisce nella versione corrente del lavoro, che ha ritirato in nota proprio quella formulazione.
- {doc}`Interpretabilità </Interpretabilita/overview>`. **Il capitolo si contraddiceva da solo a nove righe di distanza.** Le macchie sulla fotografia del lupo erano chiamate mappa di salienza, mentre poche righe sotto la stessa pagina dice che a disegnarle fu un altro metodo, e la sezione dedicata definisce la salienza come una cosa diversa. E i «tre livelli di rigore crescente» con cui si valuta una spiegazione erano elencati dal più costoso al più debole, cioè in ordine decrescente.
- {doc}`Modelli trasparenti </Interpretabilita/modelli-trasparenti-e-importanza>` (Interpretabilità). **La ragione sbagliata per un fatto vero.** Quando rimescolare una colonna *migliora* le risposte, il libro dava la colpa al caso, dicendo che quei valori stanno dentro l'oscillazione dichiarata accanto: quel margine misura la dispersione fra le ripetizioni, non l'incertezza della loro media, e rifacendo il conto la negatività si ripresenta ogni volta. A renderla difficile da leggere è la scarsità dei dati, non il caso: la conclusione regge, la spiegazione accanto no.
- {doc}`Modelli trasparenti </Interpretabilita/modelli-trasparenti-e-importanza>` (Interpretabilità). **Un esperimento misurato una volta sola, e presentato come una ripartizione.** Due colonne ricevevano un merito che il testo divideva in parti («due terzi non venivano dalla malattia»), sulla base di un'unica estrazione e senza il seme: rifacendola su otto, i valori si spostano tanto da non reggere nessuna ripartizione. Restano i due meccanismi che dentro quel numero si sommano, detti per quello che sono.
- {doc}`Loop engineering </IngegneriaLLM/loop-engineering>` (Prompt, contesto e loop). **Una glossa che, letta da sola, diceva il contrario della formula.** «Siccome quel numero viene moltiplicato per sé stesso venti volte»: a essere elevato a potenza non è la probabilità di riuscire un passo, ma quella di sbagliarlo, e il lettore che copriva la formula ne ricavava un andamento rovesciato. Con il cinque per cento di errore a ogni passo, venti passi di fila riescono trentasei volte su cento; col due e mezzo, sessanta.
- {doc}`Prompt engineering </IngegneriaLLM/prompt-engineering>` (Prompt, contesto e loop). **Una separazione promessa che il modello non ha.** Del testo da tradurre era detto che, arrivando sotto l'etichetta del materiale, «si legge per quello che è»: la stessa pagina, seicento righe più avanti, spiega che un modello non ha modo di distinguere i dati dai comandi, ed è la ragione per cui l'attacco funziona. L'etichetta dice da dove quel testo parte, non lo mette al sicuro.
- {doc}`Prompt engineering </IngegneriaLLM/prompt-engineering>` (Prompt, contesto e loop). **Una figura che attribuiva misure a chi non ne ha.** Nella mappa delle tecniche di prompt, il guadagno misurato da una meta-analisi era esteso anche a una tecnica che quella meta-analisi esclude apposta dal proprio conteggio; e fra le tecniche misurate compariva il chiedere al modello di essere un esperto mondiale, che nessuna delle due fonti della pagina misura. Corrette tutte e due le copie della descrizione del disegno.
- {doc}`Prompt engineering </IngegneriaLLM/prompt-engineering>` (Prompt, contesto e loop). **Un numero del testo che dalla figura non tornava.** Le prime tre voci «scendono a 80» diceva la pagina, mentre il conto rifatto ne dà settantanove e mezzo e le tre barre disegnate, arrotondate, sommano a ottanta: chi le sommava trovava un numero e chi rifaceva il conto un altro. Adesso la frase è vera per tutti e due.
- {doc}`Context engineering </IngegneriaLLM/context-engineering>` (Prompt, contesto e loop). **Lo stesso nome per tre cose in due pagine.** Il foglio di appunti su cui un modello scrive i passaggi intermedi era, nel giro di due capitoli, uno spazio che dura quanto la finestra di contesto, un esempio di stato conservato fuori da essa, e il contrario di sé stesso, con un rimando che mandava proprio alla pagina che diceva l'opposto. Adesso ogni uso ha il suo nome.
- {doc}`Prompt, contesto e loop </IngegneriaLLM/overview>`. **Riserve che i riassunti avevano perso, e un «gratis» smentito due sezioni dopo.** Quattro riquadri riassumevano una promessa senza la condizione che la ribalta: l'indipendenza fra i passi che il conto presuppone e che nella pratica non c'è, la catena di ragionamento che può non corrispondere a quello che il modello ha fatto davvero, e il fatto che una misura vale per un modello solo e per quel momento. Parlare bene al modello, poi, non è gratis: la pagina lo dice altrove, contando i token.
- {doc}`I modelli di riconoscimento </SpeechRecognition/modelli-asr>` (Speech Recognition). **Un conto combinatorio con un termine di troppo.** Gli allineamenti che un trasduttore deve sommare non sono $\binom{T+U}{U}$ ma $\binom{T+U-1}{U}$, perché l'ultimo passo chiude sempre l'ultimo istante: la prova più corta è il caso con un istante e un simbolo, dove di allineamenti ce n'è **uno** e la vecchia formula ne dava due. Verificato enumerandoli in due modi indipendenti. E cinquanta per undici fa cinquecentocinquanta, che non sono «poche centinaia».
- {doc}`I modelli di riconoscimento </SpeechRecognition/modelli-asr>` (Speech Recognition). **Due guasti diversi chiamati con lo stesso nome.** La decodifica più semplice, quella che a ogni passo prende il simbolo più probabile, era presentata come «la solita decodifica del percorso migliore», mentre il lavoro citato la nomina per un difetto suo, la tendenza a incepparsi ripetendo la stessa cosa, che la pagina aveva appena spiegato come un guasto distinto.
- {doc}`I modelli di riconoscimento </SpeechRecognition/modelli-asr>` (Speech Recognition). **Una didascalia che smentiva il testo dieci righe sotto.** Sotto la figura si leggeva che la rete assegna un simbolo a ogni istante, che è esattamente la lettura contro cui la sezione mette in guardia: la rete può anche non assegnare niente, ed è il motivo per cui serve il simbolo vuoto. E la distanza di edit, che un capitolo precedente spiega per esteso con la sua storia, era reintrodotta qui come se fosse nuova: adesso si richiama.
- {doc}`Sintesi vocale </SpeechRecognition/sintesi-vocale>` (Speech Recognition). **Due pagine del libro davano allo stesso modello due frequenze diverse.** Il generatore d'onda che fabbrica il suono un campione alla volta ne produce sedicimila al secondo, come dice il suo articolo e come il capitolo sull'audio scriveva già; i ventiquattromila sono di un altro modello. La correzione andava fatta in due punti, perché il secondo era rimasto indietro di centotrenta righe.
- {doc}`Sintesi vocale </SpeechRecognition/sintesi-vocale>` (Speech Recognition). **Un confronto fra due misure che non erano confrontabili.** «Ventotto centesimi di scarto su un riferimento identico»: i due punteggi vengono da due raccolte di registrazioni diverse, con voci diverse e frequenze diverse, quindi il riferimento identico non c'era e la sottrazione non si poteva fare. E la dilatazione delle convoluzioni non raddoppia a ogni strato all'infinito: raddoppia fino a un limite, poi ricomincia.
- {doc}`Speech Recognition </SpeechRecognition/overview>`. **Un compromesso che invece è un prodotto.** Delle due valutazioni che concorrono alla trascrizione, quella acustica e quella linguistica, era detto che si cerca «il miglior compromesso»: si moltiplicano, e basta che una delle due bocci una frase perché quella frase esca di gara, cosa che un compromesso non fa.
- {doc}`Graph Neural Network </GraphNeuralNetwork/overview>`. **Due fatti riportati alla loro fonte.** L'antibiotico che nel 2020 una rete neurale ha pescato fra migliaia di sostanze già preparate porta come primo firmatario Jonathan Stokes, e il gruppo che ha guidato il lavoro è quello di James Collins e Regina Barzilay. E il lavoro di Euler sui ponti di Königsberg, che il capitolo datava al 1736, fu presentato all'Accademia di San Pietroburgo nel 1735 e stampato sei anni dopo: il 1736 è l'anno del volume, non quello del lavoro.
- {doc}`GraphSAGE, GAT e applicazioni </GraphNeuralNetwork/architetture-applicazioni>` (Graph Neural Network). **Il grafo delle strade aveva perso il caso più comune.** Nella rete che stima i tempi di percorrenza i nodi sono i segmenti di strada, e il libro diceva che a collegarli sono gli incroci. Due segmenti sono collegati anche, e prima di tutto, quando si susseguono sulla stessa strada.
- {doc}`GraphSAGE, GAT e applicazioni </GraphNeuralNetwork/architetture-applicazioni>` (Graph Neural Network). **Una garanzia che lo zero non copre.** La rete che raggiunge il massimo potere distintivo di un passaparola pesa lo stato del nodo con uno scalare appreso, e la sua variante più diffusa lo fissa a zero: il corollario che dà l'iniettività la dà per infinite scelte di quello scalare, fra cui tutti gli irrazionali, e zero non è una di quelle. La variante funziona bene lo stesso, e la pagina ora dice che è una scelta sperimentale e non una conseguenza del teorema.
- {doc}`Message passing </GraphNeuralNetwork/message-passing>` (Graph Neural Network). **Un costo che dimenticava il grado del polinomio.** Il filtro polinomiale sul grafo costa un prodotto fra matrice sparsa e vettore per ogni grado, quindi il conto cresce con il grado; la forma scritta prima coincide con quella giusta soltanto quando il polinomio si ferma al primo.
- {doc}`Message passing </GraphNeuralNetwork/message-passing>` (Graph Neural Network). **Due glosse che la formula smentiva.** Con la normalizzazione che divide per la radice dei gradi, la configurazione più liscia del grafo non dà lo stesso numero a tutti i nodi: dà a ciascuno un valore che cresce con il numero dei suoi vicini, e su una stella il centro vale più del doppio di una punta. E in una catena di quattro nodi gli unici pesi che valgono un mezzo sono quelli che i due nodi di bordo danno a sé stessi, non quelli fra un bordo e l'altro, che non sono nemmeno collegati.
- {doc}`Message passing </GraphNeuralNetwork/message-passing>` (Graph Neural Network). **Il riquadro chiedeva di ricordare un terzo modo di riassumere che la pagina non aveva dato.** I modi per mettere insieme i messaggi dei vicini sono tre, la somma, la media e il massimo, e il racconto si fermava ai primi due mentre il riepilogo li contava tutti e tre. Ora il massimo entra anche nella scena.
- {doc}`I knowledge graph </GraphNeuralNetwork/knowledge-graph>` (Graph Neural Network). **Un conto che cambiava da una riga all'altra.** «Restano tre fatti che una freccia non sa raccontare», diceva la pagina, e la scheda avanzata poco sotto ne elencava altri tre, diversi da quelli: chi leggeva le due cose di fila si trovava a rifare il conto. Gli esempi restano, il numero esce.

(v1-12-1)=

## 1.12.1 · 31 agosto 2026

### Correzioni

- **Dieci capitoli riletti per intero.** MLOps, Visione artificiale, AI responsabile, Deep reinforcement learning, PINN, NLP, GAN, Modelli latenti, Verosimiglianza esatta e World models sono passati per le quattro lenti: rimandi portati al bersaglio giusto, glosse riallineate alle formule che commentano, numeri verificati contro i blocchi che li stampano. In Matematica e Sistemi multi-agente sono state diradate le contrapposizioni rimaste, dove non contrapponevano più niente.
- {doc}`L'addestramento avversario </GAN/come-funziona>` (GAN). **Il discriminatore giocattolo è stato eseguito davvero.** Il conto della pagina è girato riga per riga, seme compreso: il gradiente letto per quello che indica (dove il verdetto scende, e quindi, a ritroso, in che verso salire), la conversione fra perdita e frequenza che dà un minimo e non un massimo, e la curva del verdetto rimessa dove i numeri la mettono. Nel capitolo, quattro buchi dell'isomorfismo chiusi e un verso di metrica raddrizzato.
- {doc}`Comprimere e ricostruire </ModelliLatenti/comprimere-e-ricostruire>` (Modelli latenti). **Le affermazioni che il capitolo smentiva altrove.** La non linearità che serve nel decoder e non «da tutte e due le parti», il copista e l'archivista rimessi ciascuno al proprio mestiere, il rapporto di compressione che una sezione dava cinquanta e un'altra quarantotto, e la media che mancava alla PCA probabilistica: tutto verificato coi conti eseguiti.
- {doc}`A che serve saperlo </VerosimiglianzaEsatta/a-che-serve>` (Verosimiglianza esatta). **Una citazione che cambiava soggetto.** Le virgolette si aprivano su «non riescono a distinguere» e lasciavano fuori chi non distingue, mentre l'originale parla della densità appresa dai modelli a flusso: il soggetto sbagliato proprio nel capoverso la cui tesi è che la densità e il modello non sono la stessa cosa. Ripresa dal testo originale.
- {doc}`Mondi in miniatura </WorldModels/mondi-in-miniatura>` (World Model). **Alla temperatura più alta l'agente impara meno bene, non smette di imparare.** La conclusione non seguiva dai numeri che la pagina stampa: il punteggio resta sopra la soglia, e lontano da quello di una politica che gioca a caso. A cambiare è la deviazione standard, che stava nella tabella e non veniva mai usata.

(v1-12-0)=

## 1.12.0 · 30 agosto 2026

### Sezioni nuove

- {doc}`Imparare a imparare </DeepLearning/meta-apprendimento>` (Deep Learning). **Imparare a imparare.** Una sezione nuova sul meta-apprendimento: che cosa vuol dire scegliere una posizione di partenza da cui pochi passi bastano su un compito mai visto, e il conto che lo dimostra.
- {doc}`Gerarchia e opzioni </DeepReinforcementLearning/gerarchia>` (Deep Reinforcement Learning). **Decidere ogni tanto invece che sempre.** Una sezione nuova sul rinforzo gerarchico: le opzioni, chi decide a quale scala di tempo, e perché un agente che sceglie meno spesso può imparare di più.

### Pagine ampliate

- {doc}`Far funzionare le reti profonde </DeepLearning/ottimizzazione-regolarizzazione>` (Deep Learning). **Quattro cose che il libro usava come note e non aveva mai spiegato.** Il label smoothing (promesso da un altro capitolo e mai mantenuto), il Thompson sampling nei banditi, il teacher forcing e l'exposure bias nei modelli sequenza-a-sequenza, i test d'ipotesi e il p-value con le correzioni per i confronti multipli: ciascuno nella sezione in cui il lettore ha già ciò che serve per capirlo.
- {doc}`Componenti e modelli classici </SerieTemporali/componenti-e-classici>` (Serie temporali). **Sette voci dalla terza tornata sulle fonti.** L'orizzonte di un MA(q) e il nome della passeggiata aleatoria, ROUGE accanto a BLEU e il verso di un metro, il curriculum e il self-play negli agenti che imparano insieme, e l'ordinare per perdita fra le metriche.
- {doc}`Algebra lineare </Matematica/algebra-lineare>` (Matematica). **Altri quattro presupposti spiegati.** La decomposizione ai valori singolari e l'approssimazione di rango basso (che reggono PCA, LoRA e le fattorizzazioni), la calibrazione delle probabilità con il diagramma di affidabilità e il temperature scaling, il filtro di Kalman nelle serie temporali, e i modelli lineari generalizzati con la regressione di Poisson: lineare, logistica e Poisson sono la stessa macchina.

### Correzioni

- **Dieci capitoli riletti con quattro lenti.** Matematica, Modelli di diffusione, Python, Machine learning, Reti neurali, PyTorch, Transformers, State space model, Sistemi multi-agente e Audio sono passati per i tre lettori-agente e per un lettore esperto che apre ogni rimando alla destinazione: rimandi che mandavano dove la cosa non c'era, termini dati per noti prima di essere spiegati, due formule sbagliate, un'affermazione vera in aritmetica esatta e falsa eseguendo la pagina. Ogni correzione è misurata nel suo consuntivo, e i tagli sono più delle aggiunte.

(v1-11-0)=

## 1.11.0 · 30 agosto 2026

### Sezioni nuove

- {doc}`Sistemi lineari </Matematica/sistemi-lineari>` (Matematica). **Il capitolo di matematica arriva a undici sezioni.** Cinque nuove: come si risolve un sistema lineare e che cosa si risponde quando una risposta non c'è, le proiezioni e i minimi quadrati, il determinante come volume, le disuguaglianze che valgono senza sapere niente della distribuzione, e le catene di Markov, che qui servono a tre capitoli più avanti.
- {doc}`Il limite continuo </ModelliDiffusione/sde-e-ode>` (Modelli di diffusione). **La diffusione, dal limite continuo alla generazione in un passo.** Sei sezioni nuove: le equazioni differenziali dietro il rumore che si toglie, il flow matching che sceglie il percorso invece di ereditarlo, i campionatori che tagliano i passi, i generatori che ne fanno uno solo, i modi di dire al modello che cosa generare, e che cosa cambia quando lo stato è fatto di parole invece che di numeri.

### Correzioni

- {doc}`Guida e allineamento </ModelliDiffusione/guida>` (Modelli di diffusione). **Che cosa fa davvero alzare la forza della guida.** Si legge dappertutto che equivale a campionare da una distribuzione «inclinata»: sul banco di prova, dove le due quantità si calcolano esattamente, i campioni escono da tutt'altra parte, e la pagina misura di quanto. Il danno si concentra a metà percorso, dove le due direzioni divergono di più.
- **Meno cadenza, più prosa.** Una passata su tutto il libro ha diradato la costruzione «non è X, è Y», che ricorreva più di settecento volte e a forza di ripetersi non contrapponeva più niente. Dove il contrasto serviva a mettere in guardia da un errore vero è rimasto, in una forma ordinaria.

(v1-10-5)=

## 1.10.5 · 25 agosto 2026

### Correzioni

- {doc}`Analisi e ottimizzazione </Matematica/analisi-ottimizzazione>` (Matematica). **Tre affermazioni del capitolo di matematica che non reggevano.** La sigmoide non restituisce probabilità che sommate fanno uno: quella è la softmax, e le due rispondono a domande diverse. Un punto di sella è un fondo in una direzione e una cima in quella perpendicolare, quindi servono almeno due dimensioni perché esista. E il gradiente taglia le curve di livello ad angolo retto per una ragione che vale in ogni dimensione, non solo sul piano.
- {doc}`Teoria dell'informazione </Matematica/teoria-informazione>` (Matematica). **I bit di un dado non sono il numero di domande che servono.** Le domande si contano intere, quindi su un tiro solo la strategia migliore ne consuma in media due e due terzi; ai 2,585 bit del dado ci si arriva soltanto giocando molti tiri insieme. E il limite della compressione ha ora un conto che si può rifare, stampato dal codice della pagina.
- {doc}`Probabilità e statistica </Matematica/probabilita-statistica>` (Matematica). **Il teorema del limite centrale, con l'ipotesi che gli serve.** Vale quando nessuno dei contributi può essere, da solo, enormemente più grande di tutti gli altri messi insieme, e adesso lo dice anche la spiegazione facile. Nella figura sugli intervalli di confidenza il terzo modello è sceso dove la distanza dagli altri due è più larga dell'incertezza della misura.
- {doc}`La matematica di un LLM </Matematica/matematica-llm>` (Matematica). **Le copie parallele dell'attenzione si specializzano meno di quanto si racconti.** Che due copie che imparano la stessa cosa costino accuratezza è un'ipotesi ragionevole, non un risultato: da un modello addestrato se ne può togliere una quota grossa senza perdite misurabili, e la pagina lo dice adesso con i lavori che l'hanno mostrato.
- **Le figure del capitolo di matematica dicono quello che il disegno mostra.** I simboli che vivevano solo nella scheda avanzata (l'asterisco del punto migliore, il cappello di una stima, le lettere della divergenza) sono sciolti nella didascalia; e in due disegni una scritta finiva fuori dal margine o sopra un'altra.
- **In stampa i riquadri non si aprono più a vuoto.** Una scheda che cominciava in fondo alla pagina lasciava lì una cornice colorata con dentro la sola etichetta, e mandava il contenuto tutto alla pagina dopo. I blocchi di codice spezzati fra due pagine lo segnalano ora con una riga pulita, e dove online si muove una figura, sulla carta i tre fermi immagine dicono di essere tre istanti dello stesso movimento.

(v1-10-4)=

## 1.10.4 · 25 agosto 2026

### Correzioni

- **Nel PDF il codice si copia e gira.** Il carattere con cui il libro compone il codice unisce certe coppie di simboli in un segno solo: in stampa `!=` usciva come ≠, `>=` come ≥ e `->` come una freccia, cioè righe che ricopiate a mano non sono Python. Adesso ogni simbolo resta quello che è, in pagina e nel testo che si seleziona.
- **Si vede quali blocchi sono da provare e quali da leggere.** Qualche esempio usa nomi di comodo, un file che non esiste o un modello che si dà per costruito altrove: adesso lo dichiara, online e in stampa, invece di somigliare a codice pronto da eseguire.
- {doc}`Le basi di Python </Python/basi>` (Python). **Quello che serve per programmare non è più chiuso in metà pagina.** Come si scrive dentro una tabella senza rovinarla, perché un dizionario trova un valore in un colpo mentre una lista va scorsa tutta, come si chiede la copia di una lista, e come si legge un messaggio d'errore: erano cose che arrivavano solo a chi apriva la scheda avanzata, e ora ci sono a tutti e due i livelli.
- {doc}`Python </Python/overview>`. **Da dove venga la velocità di NumPy, raccontato una volta sola.** Le due schede sull'ecosistema rispondevano a due domande diverse, e la facile lasciava credere che il guadagno venisse dal contenitore. Viene invece dal pedaggio che Python paga a ogni giro di un ciclo, e che scrivendo il conto sul blocco intero si paga una volta.
- {doc}`NumPy </Python/numpy>` (Python). **I riquadri di ripasso dicono quello che dice la pagina.** La regola del broadcasting era enunciata in un modo che l'esempio subito sopra smentisce, PyTorch risultava in cima alla torre delle librerie proprio dove il testo spiega che le sta accanto, e dai ripassi mancavano la mutabilità, le viste di NumPy e Matplotlib.
- **I numeri del capitolo di Python tornano, e le fonti ci sono.** La media di gruppo dice ora che i valori mancanti non contano nel divisore; la tassonomia dei dati mancanti porta il riferimento a Rubin; il costo del Python senza lucchetto è quello che dichiara la documentazione ufficiale; il messaggio d'errore stampato è quello che Python produce davvero, freccine comprese; e PyTorch nasce nel 2016 ma arriva ai ricercatori l'anno dopo.
- {doc}`L'addestramento avversario </GAN/come-funziona>` (GAN). **La derivazione del valore di una GAN si legge anche in stampa.** La formula che riscrive il secondo integrale nello spazio dei dati usciva dal margine della pagina e ci lasciava fuori la coda: adesso va a capo sull'uguale.

(v1-10-3)=

## 1.10.3 · 24 agosto 2026

### Correzioni

- {doc}`Introduzione </Introduzione/overview>`. **Una definizione sola di intelligenza artificiale, e regge per tutto il libro.** Si occupa dei compiti per cui nessuno sa scrivere una ricetta che tenga nel mondo vero: ne resta fuori la lavatrice, che una ricetta ce l'ha e funziona, e ci restano dentro i sistemi esperti, che a scriverla provavano lo stesso, a mano. Il loro fallimento diventa così la premessa del capitolo invece di un aneddoto.
- {doc}`Introduzione </Introduzione/overview>`. **L'Introduzione va in una direzione sola.** Il blocco filosofico e le tre discipline in cui la decisione si studiava da prima arrivano ora davanti a Turing, e la funzione obiettivo ha una sezione sua, dopo che il lettore sa che cosa vuol dire addestrare un modello. E ARGO, la Lancia Thema che nel 1998 fece da sola il 94% dei duemila chilometri della Mille Miglia, dice adesso perché sta lì: dentro non c'era niente che avesse imparato qualcosa.
- **I rinvii dicono dove.** «Ci torneremo», «lo vedremo da vicino»: un rimando che non nomina la destinazione chiede di aspettare senza dare niente in cambio. Nell'Introduzione quello sui Transformer è diventato un collegamento, in World Model il collasso nomina la sezione sulle JEPA, in Ricerca e pianificazione la promessa è sparita perché due righe più in là il testo dice già chi se ne occupa.
- **Ogni capitolo scaricato in PDF dice come si può usare.** Il libro intero lo scrive nel colophon, che un capitolo ritagliato non si porta dietro: ora la licenza sta a piè di pagina in ciascuno dei file per capitolo, che sono quelli fatti per essere mandati a qualcuno.
- **Il razzo si trova.** Sulle pagine che sono esse stesse un notebook, il comando che esegue il codice sta fra i pulsanti in cima alla pagina; da telefono è accanto al marchio, non a destra come il testo diceva.

(v1-10-2)=

## 1.10.2 · 23 agosto 2026

### Pagine ampliate

- {doc}`Prefazione </prefazione>`. **La prefazione dice da che parte sta il libro.** Da quando i mestieri di chi lavora con l'intelligenza artificiale sono stati riassorbiti in uno solo, l'apprendimento automatico è diventato facoltativo, e al suo posto è rimasta la fiducia che un modello già addestrato impari al volo dagli esempi che gli si mettono nel prompt. Queste pagine stanno dall'altra parte: meno caccia all'istruzione magica, più architetture, iperparametri e funzioni di costo.

### Correzioni

- {doc}`Prefazione </prefazione>`. **La prefazione si legge più svelta, e riconosce un debito.** Il racconto dei pappagalli stocastici arriva prima al punto e il triangolo impossibile dice quello che deve in un terzo dello spazio; il segno del libro nomina ora *Gödel, Escher, Bach*, da cui viene sia la lettura di quella figura come un anello che torna su sé stesso, sia l'idea di spiegare ogni cosa due volte.
- **Gli strumenti che il libro nomina hanno la loro fonte.** PyTorch, TensorFlow, Keras e Hugging Face si trovano ora in bibliografia con l'articolo di chi li ha fatti e l'indirizzo per approfondire.

(v1-10-1)=

## 1.10.1 · 22 agosto 2026

### Correzioni

- **I due livelli raccontano la stessa struttura, in tutto il libro.** Una rilettura di tutte le coppie di schede ha rimesso nella scena del livello elementare le mosse che stavano soltanto in quello superiore: la condizione che fa valere una garanzia, il caso in cui un metodo si rompe, il conto che il testo commenta. Dove l'esempio illustrava il risultato senza rifare i passaggi, adesso li rifà.
- **Errori di merito, trovati rieseguendo i conti.** Il Davies-Bouldin si minimizza, e stava in fila con due indici che si massimizzano; la perplessità scende quando la frase diventa più prevedibile, non sale; la proiezione di un vettore su un altro non è il prodotto scalare; per una convoluzione il costo per pixel è esattamente il numero dei pesi; e il `top_p` sceglie quanta parte della classifica resta in gara, non quanti candidati.
- **Ipotesi che mancavano accanto alle formule.** Una mistura di due gaussiane fa due gobbe solo se i centri distano più di due deviazioni standard; la simmetria di Shapley chiede meno di quanto il testo dichiarasse; nelle reti su grafo l'equivarianza vuole anche l'indipendenza dall'ordine dei vicini, non i soli pesi condivisi; e la catena degli errori di un modello linguistico presuppone che ogni passo rischi quanto gli altri.
- {doc}`AI responsabile </AIResponsabile/overview>`. **Il paradosso dell'equità si legge per intero anche al primo livello.** Nel caso COMPAS le due parti avevano ragione tutte e due, e il conflitto nasce dalle frequenze diverse nei due gruppi: nessun programma scritto meglio le mette d'accordo. La scena del tribunale adesso arriva fin lì, invece di fermarsi al primo conto.
- {doc}`Generare suono e musica </Audio/generazione-audio>` (Audio oltre la voce). **Un capoverso che si leggeva come codice torna a essere prosa.** In generazione audio la riga che chiude la figura si era attaccata al paragrafo successivo, e il pezzo su Jukebox e AudioLM finiva impaginato come un blocco di programma. E in ricerca ad albero un titolo di sezione era di un livello più basso degli altri due della pagina.
- **Una lettera, un mestiere.** Decine di simboli che ne facevano due nella stessa pagina sono stati rinominati: la sequenza di token e il vettore continuo da cui nasce, la latenza e la durata, il modello bozza e l'intero quantizzato, la precisione e la probabilità.

(v1-10-0)=

## 1.10.0 · 20 agosto 2026

### Sezioni nuove

- {doc}`Il kernel trick </MachineLearning/svm-kernel>` (Machine Learning). **Il kernel trick ha la sua pagina.** La Support Vector Machine, che era la pagina più lunga del libro, si è spezzata alla cucitura naturale: di qua il margine massimo con la strada più larga percorsa passo per passo, di là il kernel trick con la regressione, il riconoscere le anomalie e la pratica con scikit-learn. Due sedute invece di una maratona.
- {doc}`MCTS e AlphaGo </DeepReinforcementLearning/mcts-alphago>` (Deep Reinforcement Learning). **La ricerca ad albero Monte Carlo, AlphaGo e l'allineamento dei modelli linguistici** si leggono ora in una pagina propria, staccata dal gradiente di policy, che si ferma a PPO e chiude con i suoi riquadri. Il pensare prima di muovere comincia dove finisce l'istinto, e adesso anche l'indice lo dice.
- {doc}`Oltre il BPE </NaturalLanguageProcessing/oltre-il-bpe>` (Natural Language Processing). **Oltre il BPE.** WordPiece, SentencePiece e i byte hanno la loro pagina, con le quattro conseguenze che si incontrano davvero lavorando (i numeri spezzati male, l'italiano che costa più token, lo spazio di troppo, il vocabolario congelato). Il BPE resta di qua, con l'esempio svolto e le trenta righe di Python.

### Pagine ampliate

- {doc}`Le basi di Python </Python/basi>` (Python). **Il traceback e il paracadute.** Le basi di Python insegnano a leggere un errore (dal fondo, il che cosa; risalendo, il dove) e a mettere `try`/`except` attorno alla riga fragile, con l'except stretto e lo stile che chiede scusa invece di chiedere permesso. Era il buco più grosso del capitolo, e il primo in cui un lettore inciampa da solo.
- **Cinque disegni nuovi.** I ritagli gemelli dell'apprendimento contrastivo, lo spettrogramma a tessere dell'AST col colpo che guarda la sua eco, la parte coperta di wav2vec col test a risposta multipla, il giro codec-Transformer-decoder della generazione a token; e AlexNet è ridisegnata com'era davvero: due tronchi, uno per GPU, che si parlano soltanto agli incroci.

(v1-9-4)=

## 1.9.4 · 20 agosto 2026

### Pagine ampliate

- {doc}`Deep Q-Network (DQN) </DeepReinforcementLearning/dqn>` (Deep Reinforcement Learning). **Il quaderno degli appunti ha imparato a ripassare ciò che sorprende.** Le due migliorie di DQN che mancavano hanno la loro sezione: il replay con priorità, che pesca più spesso le esperienze su cui la rete ha sbagliato di più (contando un po’ meno i ripassi pescati apposta, per non farsi un'idea storta), e la rete a due rami, che giudica la situazione separatamente dalle mosse e impara anche dove le mosse non contano. Con le formule, le fonti, e la riga su Rainbow che le mette insieme.
- {doc}`Prefazione </prefazione>`. **Il triangolo impossibile adesso si vede.** La prefazione chiede di guardare il segno del libro un angolo alla volta, coprendo il resto con una mano; accanto a quel passaggio ora c'è il disegno grande, gli stessi tre tracciati del logo, così la mano ha dove posarsi.

### Correzioni

- **Rifiniture di lettura.** Nelle conclusioni ogni capitolo citato è un link una volta sola, la prima; e in visione e linguaggio due aperture che chiedevano di immaginare la scena adesso ci entrano direttamente.

(v1-9-3)=

## 1.9.3 · 20 agosto 2026

### Pagine ampliate

- **Le formule che il testo usava senza averle mai scritte adesso ci sono**: l'equazione di Bellman per Q (quella di cui SARSA è il campionamento), la rotazione della RoPE e la RMSNorm, l'obiettivo di word2vec col negative sampling e la pesatura di GloVe, la derivazione del gradiente di policy con la GAE, la riparametrizzazione di SAC con la correzione della tanh, la MMR per la ridondanza dei passaggi recuperati e il PSI per la deriva, colonne categoriche comprese.
- {doc}`Detection e segmentazione </VisioneArtificiale/detection-segmentazione>` (Visione artificiale). **Le ancore dicono in chiaro che il confronto col riquadro vero esiste solo in addestramento**, mentre in inferenza la IoU ricompare soltanto fra le predizioni, nella pulizia dei doppioni. E la famiglia YOLO ha il suo racconto, dalla griglia nuda del 2016 al modello del 2026 che la pulizia non la fa più: con cinque righe di codice da provare, e le confidenze uscite davvero dal modello.

### Correzioni

- {doc}`Prefazione </prefazione>`. **La rilettura avversaria è arrivata alle pagine che mancavano**: la prefazione, le conclusioni e i capitoli su Python, sulla ricerca e su visione e linguaggio. Nella prefazione la riga di Anthropic sulla poesia in rima adesso cita il resoconto in cui sta davvero, il teorema di Penrose ha le sue due voci in bibliografia (il disegno del 1958 col padre, la matematica del 1991), e il giro dei tre rapporti si fa con numeri veri.
- {doc}`Conclusioni </Conclusioni/overview>`. **Il ripasso finale fotografa il libro com'è oggi.** Le famiglie generative sono cinque come nella mappa del libro, i capitoli nati quest'estate sono nominati e linkati, AlphaGo porta al capitolo che lo racconta, i due livelli dicono la stessa cosa sugli errori che si accumulano in un agente (sbagliare tutti insieme è l'estremo buono, non il cattivo), e i riquadri da ricordare stanno prima del congedo, così il libro chiude salutando.
- {doc}`Esplorare lo spazio </Ricerca/esplorare-lo-spazio>` (Ricerca e pianificazione). **La parità dell'otto-tessere dimostrava il contrario di quel che affermava**: se ogni mossa inverte la parità, le disposizioni di parità diversa si raggiungono eccome. L'argomento vero ha due interruttori che scattano insieme, e ora c'è. IDA* e Monte Carlo Go hanno i nomi di chi li ha proposti, Deep Blue il suo punteggio, e la potatura il suo nome completo anche per chi legge il livello Elementare.
- {doc}`Python </Python/overview>`. **Su Ubuntu il primo comando dell'ambiente virtuale si fermava**, e la pagina non lo diceva: ora dice del pezzo che manca, di come si esce dalla scatola e degli altri gestori che si incontrano nei tutorial. La storia dei framework rispetta le date (Theano esisteva prima di AlexNet), e la tabella delle vendite adesso la fabbrica la pagina stessa, così ogni numero si può rifare.
- {doc}`Allineare due spazi </VisioneLinguaggio/allineare-due-spazi>` (Visione e linguaggio). **I numeri del divario fra i due quartieri si leggono alla precisione che il protocollo sostiene**, e il testo dichiara perché; la lettera del batch ha smesso di fare due mestieri, i rimandi atterrano sulle sezioni che contengono la cosa promessa, e le sedici domandine di SigLIP hanno preso il posto di otto interrogazioni, non di quattro.
- **I numeri commentati escono da blocchi che stanno nelle pagine**: il conto del weight decay si stampa da sé, il sorpasso dei banditi arriva dove la misura lo mette (passato il novemillesimo tiro), il giudice di recensioni stampa i tre decimali che il testo discute, e l'ordine ragionato della potatura dichiara quante volte venti ordini a caso lo battono: una.

(v1-9-2)=

## 1.9.2 · 20 agosto 2026

### Correzioni

- **Il libro sapeva una metafora sola, e cominciava sempre allo stesso modo.** Una scheda «Elementare» su cinque apriva con «Immagina di»: presa una per una era una buona apertura, ma chi legge di fila incontrava lo stesso invito ogni cinque pagine, e a quel punto quello che si sente è un tizio che si schiarisce la voce. Adesso sono una su dodici, e nessun capitolo ne ha più del quindici per cento. Nessuna analogia è stata tolta: è cambiato il modo di entrarci, perché entrare nella scena è più forte che chiedere il permesso di immaginarla. «Immagina» è rimasto dove è il verbo esatto, cioè dove si chiede davvero di figurarsi qualcosa che non si può vedere: uno spazio a dodicimila dimensioni, una lingua senza alfabeto, sedicimila puntini in un secondo di suono.
- **E sotto c'era la catena di montaggio.** Tolta la cornice, è venuto fuori che la stessa immagine apriva sette spiegazioni in cinque capitoli diversi, tre con le stesse identiche parole. Dove regge fino in fondo resta (in *GPU* è il filo di tutto il capitolo, e la catena percorsa all'indietro è il modo migliore di raccontare come una rete impara dai propri errori); dove era solo un modo di dire «una cosa dopo l'altra» ha lasciato il posto a un'immagine scelta per quel passaggio: una filiera del latte dove il controllo scarta un lotto intero, un autolavaggio a tunnel, una fila di specchietti che rimbalza un raggio laser.

(v1-9-1)=

## 1.9.1 · 20 agosto 2026

### Pagine ampliate

- **Diciassette voci nuove in bibliografia**, per lavori che il libro nominava senza permettere di risalirci: fra gli altri la policy iteration di Howard, TRPO, DAgger, DistMult, ComplEx e RotatE.

### Correzioni

- **Formule e attribuzioni.** Corretti la risoluzione della trasformata a finestra in *Dal suono alle feature*, le dimensioni della matrice condivisa della GAT, la perdita del VQ-VAE, il costo del calcolo a blocchi di Mamba-2, il reticolo del trasduttore e la penalità di gradiente delle GAN.
- **Numeri rifatti.** La ripartizione dei fallimenti della rete di Hopfield, i punteggi dell'Audio Spectrogram Transformer e di V-JEPA 2, il MOS di Tacotron 2, l'accuratezza di Hearsay-II e la banda di previsione in *Forecasting neurale*, che adesso il codice stampa.
- **Le due schede tornano a dire la stessa cosa** dove si contraddicevano, in *Verosimiglianza esatta*, *Modelli latenti* e *Costo del coordinamento*; e l'apertura delle PINN non attribuisce più l'idea a un lavoro che il capitolo stesso, due pagine dopo, fa risalire al 1994.
- **I rimandi fra capitoli adesso sono link.** Quando il testo cita la backpropagation o la sezione sugli ensemble ci si arriva con un dito, invece di cercarle nell'indice.
- **La prosa è più asciutta** in tutti i capitoli: meno grassetti, meno contrapposizioni «non è X, è Y», e le schede entrano nella scena invece di annunciarla.

(v1-9-0)=

## 1.9.0 · 19 agosto 2026

### Sezioni nuove

- {doc}`Capire è accorciare </AutoSupervisione/capire-e-accorciare>` (Auto-supervisione). **Perché il pre-addestramento funzioni, il libro non lo aveva mai spiegato.** Il capitolo sull'auto-supervisione diceva che cos'è, come si fabbrica un pretesto e che cosa va storto, e si fermava lì. Questa sezione risponde alla domanda che uno studente fa al secondo minuto: perché coprire una parola e farla indovinare dovrebbe produrre qualcosa che sa di biologia. La risposta che una parte del campo dà è che per prevedere bene bisogna accorciare, e per accorciare bisogna aver capito. Dentro ci sono la complessità di Kolmogorov e il rasoio di Occam scritto in bit, il premio da mezzo milione di euro per chi comprime Wikipedia, l'argomento di Ilya Sutskever con il rimpianto che ne è il cuore, le due misure che dal 2023 lo mettono alla prova, e l'obiezione che decide tutto: chi paga il vocabolario. Si apre con Funes, il personaggio di Borges che ricorda ogni cosa e proprio per questo non riesce a pensare. C'è un esperimento che sta in una pagina e si può rilanciare (una lingua inventata in cui contare le lettere non serve a niente), e una figura animata che fa vedere il prezzo di imparare mentre lo si paga.

### Correzioni

- {doc}`Capire è accorciare </AutoSupervisione/capire-e-accorciare>` (Auto-supervisione). **Nove correzioni in una sezione sola.** La più grossa: il 99% che iGPT ottiene su CIFAR-10 è la rete rifinita per intero, non il sondaggio con la rete congelata, che dà 96,3%, ed è proprio il numero su cui poggiava il paragrafo dedicato a quanto sia misteriosa la separabilità lineare. Poi la verosimiglianza scritta senza il condizionamento, cioè la formula di un modello che la pagina stessa mostra non guadagnare niente; il passaggio che chiude l'argomento di Sutskever, che non c'era; l'apertura di Borges, che raccontava tre dettagli assenti dal racconto; e la frase «nessun programma scende sotto il fondo», che la pagina si smentiva da sola venti righe dopo. E una riscrittura di leggibilità, perché il lettore senza basi si fermava a metà: il pezzo più difficile stava fuori dalle schede, dove vale per tutti.
- **Tre figure avevano un'etichetta nera al posto del suo colore.** Su quella nuova della compressione non era cosmesi: le tre curve si distinguono soltanto dal colore, e le tre etichette erano l'unica legenda.

(v1-8-5)=

## 1.8.5 · 19 agosto 2026

### Correzioni

- **La finestra della ricerca restava muta quando il testo era già nel campo.** Succedeva arrivando da un vecchio collegamento alla pagina dei risultati, o tornando indietro col browser su un modulo ripristinato: la domanda si vedeva scritta e sotto non c’era niente, finché non si ribatteva un tasto. Adesso, appena la finestra si apre, cerca quello che trova già scritto. La ragione per cui era sfuggita vale la pena di dirla: il libro qui si costruisce con una versione del tema più recente di quella con cui il sito è pubblicato, e le due caricano l’indice della ricerca in due momenti diversi. Il difetto viveva solo nella seconda. Il controllo adesso prova tutt’e due.

(v1-8-4)=

## 1.8.4 · 19 agosto 2026

### Correzioni

- **L’avviso «Nascondi i risultati della ricerca» compariva dove non c’era niente da nascondere**, per esempio tornando alla copertina dopo una ricerca, o soltanto dopo aver aperto la finestra e averla richiusa senza andare da nessuna parte. Adesso esce solo se in quella pagina qualcosa è stato davvero evidenziato, e dice quante corrispondenze sono. Riparandolo è venuto fuori un difetto più grosso e opposto: sulle versioni recenti del tema l’evidenziazione non partiva affatto, quindi chi arrivava da una ricerca non vedeva marcata nessuna delle parole che aveva cercato. Adesso parte, e sempre allo stesso modo.

### Impianto

- **La ricerca risponde mentre si scrive.** Prima bisognava battere invio e finire su una pagina di risultati a parte; adesso i risultati compaiono a ogni tasto, raggruppati per pagina e con il capitolo scritto sopra, e portano alla **singola sezione** invece che in cima al file. Si scorrono con le frecce e si aprono con invio, senza togliere le mani dalla tastiera (la finestra si apre con Ctrl+K, o ⌘K), e le parole cercate restano evidenziate sulla pagina in cui si arriva. Se c’è un refuso la ricerca lo perdona e lo dichiara: chi scrive «trasformer» trova i Transformer sotto un «forse cercavi», invece di sentirsi rispondere che il libro non ne parla. Un effetto collaterale che si sente: l’indice della ricerca, che è il file più pesante del sito, adesso si scarica solo quando la finestra si apre. Chi legge e basta non lo paga più.

(v1-8-3)=

## 1.8.3 · 19 agosto 2026

### Correzioni

- {doc}`Analisi numerica </Matematica/analisi-numerica>` (Matematica). **La sezione sui due calcolatori prometteva più di quanto si possa mantenere.** Diceva che fissando due variabili d’ambiente le macchine tornano a stampare le stesse cifre: è vero per la parte di conto che passa da quelle due strade, e infatti il residuo dell’aneddoto è tornato identico, ma non in generale. I prodotti fra matrici di PyTorch passano da un’altra libreria ancora, che quelle variabili non toccano, e una terza macchina si è discostata di nuovo nell’ultima cifra. Adesso la pagina lo dice, e ne trae la conseguenza pratica: due esecuzioni su macchine diverse si confrontano con una tolleranza, non con l’uguaglianza.

(v1-8-2)=

## 1.8.2 · 19 agosto 2026

### Correzioni

- {doc}`Python </Python/overview>`. **SIMD adesso vuol dire qualcosa.** La sigla compariva quattro volte nel libro, sempre nuda, come se fosse noto a tutti che cosa sia: adesso è sciolta dove nasce, nel paragrafo che spiega perché NumPy è veloce. È una sola istruzione del processore che opera su più numeri insieme, tanti quanti ne entrano nei suoi registri.
- {doc}`Analisi numerica </Matematica/analisi-numerica>` (Matematica). **La sezione sui due calcolatori non dà più per noto quello che il lettore non ha ancora incontrato.** Nominava SSE2, AVX2 e AVX-512 senza dire che cosa sono (sono generazioni di istruzioni vettoriali, e a distinguerle è la larghezza dei registri: 128, 256 e 512 bit, cioè due, quattro e otto numeri in doppia precisione per volta), e citava PyTorch, che nel percorso di lettura arriva tre capitoli dopo, come se fosse già stato presentato. Sistemate anche due incoerenze interne: la scheda facile contava i numeri per istruzione in singola precisione mentre tutto il resto ragiona in doppia, e la chiusura si appoggiava a «addestramento» e «accuratezza», che a quel punto del libro non sono ancora stati definiti.

(v1-8-1)=

## 1.8.1 · 19 agosto 2026

### Pagine ampliate

- {doc}`Analisi numerica </Matematica/analisi-numerica>` (Matematica). **Perché lo stesso programma, su due calcolatori, stampa due numeri diversi.** È successo a questo libro: il controllo che riesegue il codice delle pagine diceva che andava tutto bene sul computer dell’autore, e la prima volta che è girato sui calcolatori di GitHub, in quattro punti i numeri non combaciavano. Stesso codice, stesse versioni delle librerie, stesso seme. La sezione nuova racconta il perché: in virgola mobile l’addizione **non è associativa**, e a decidere in che ordine si somma è il **kernel**, cioè la versione della routine che la libreria di calcolo sceglie per il processore che si trova sotto. C’è il conto che lo fa vedere in tre righe (gli stessi diecimila numeri sommati nei due versi danno due totali diversi), la spiegazione di che cos’è un kernel, e le due variabili con cui la scelta si fissa quando la riproducibilità serve davvero.

### Correzioni

- {doc}`Meno pesi </Efficienza/meno-pesi>` (Efficienza). **I numeri di potatura e quantizzazione sono stati rimisurati a kernel fissato**, così che chiunque rifaccia il conto ottenga le stesse cifre invece di quelle del proprio processore. Cambiano di qualche decimo, e con loro le frasi che reggevano: comporre le due leve costa 1,1 punti di accuratezza invece di 1,7, e la rete potata di nove pesi su dieci ne perde meno di uno invece di poco più di uno. La conclusione non cambia: il budget di errore non si spartisce a tavolino.
- **Il controllo che confronta i numeri stampati con quelli che il codice produce adesso gira anche sul libro pubblicato.** Ci girava solo qui: lo script non era fra i file che vengono pubblicati, e di là il controllo moriva prima di cominciare, senza che nessuno lo vedesse.

(v1-8-0)=

## 1.8.0 · 18 agosto 2026

### Sezioni nuove

- {doc}`Spline e modelli additivi </MachineLearning/curve-al-posto-di-rette>` (Machine Learning). **Come si piega una retta senza che impazzisca ai bordi.** Fin qui, quando i dati non stavano su una linea, il libro cambiava famiglia di modelli. Questa sezione fa la cosa più ovvia e mai raccontata: tiene la linea e la piega. Si apre con un attrezzo da tavolo da disegno, il listello di legno con cui si tracciavano gli scafi delle navi, e mostra che la curva del legno e quella che la statistica calcola non si somigliano, sono la stessa curva. Dentro: perché un polinomio unico ondeggia proprio dove i dati sono radi, come le spline aggiungono flessibilità **dove serve**, la manopola che va dallo spago al righello di acciaio, e i modelli additivi, che sommano una curva per colonna e per questo si possono guardare uno per uno.
- {doc}`Il bootstrap </MachineLearning/il-bootstrap>` (Machine Learning). **Quanto ci si può credere, a un numero misurato una volta sola.** L’accuratezza è dell’ottantasette per cento: sì, ma su quel test. Rifacendo la prova quanto verrebbe? Per la media una formula c’è da due secoli; per la mediana, per l’accuratezza di un modello, per la differenza fra due modelli, no. La sezione racconta il metodo con cui Bradley Efron nel 1979 ha risposto lo stesso, senza raccogliere un dato in più, e soprattutto **dove si rompe**: sul massimo, su dati non indipendenti, e su un campione raccolto male, dove restituisce un intervallo stretto attorno al numero sbagliato.
- {doc}`Modelli generativi </MachineLearning/modelli-generativi>` (Machine Learning). **Classificare descrivendo, invece che tracciando confini.** Un ornitologo non ti dice dove finisce una cornacchia e comincia una gazza: ti dice com’è fatta ciascuna delle due. È l’altra strada per classificare, e in cambio dà una cosa che l’altra non dà: accorgersi di quello che **non somiglia a niente di visto**, un gettone fra le monete. La sezione costruisce l’analisi discriminante lineare e quadratica e il naive Bayes, misura quando conviene la più flessibile e quando no, e mostra il numero che riassume tutto: sullo stesso punto assurdo lo stesso modello dice «non capita mai» e «sono sicuro al 99,99%».
- {doc}`Valutare un raggruppamento </MachineLearning/valutare-un-raggruppamento>` (Machine Learning). **Come si giudica un raggruppamento, quando la risposta giusta non esiste.** Che Plutone non sia un pianeta lo ha deciso un’alzata di mano: le misure erano note a tutti, a mancare era il criterio. La sezione mostra il caso in cui il voto automatico più usato **preferisce la risposta sbagliata** con un margine confortevole, spiega perché l’indice di Rand grezzo può dare 0,9 al puro caso, e chiude su un teorema del 2002: tre proprietà che sembrano minime non possono valere tutte e tre insieme, e quindi la scelta va dichiarata, perché nessun dato la farà al posto nostro.

### Pagine ampliate

- {doc}`Alberi e metodi ensemble </MachineLearning/alberi-ensemble>` (Machine Learning). **Il capitolo di machine learning adesso dice che cosa stampa il suo codice.** Prima non lo dichiarava da nessuna parte, quindi nessun controllo lo eseguiva mai: ventiquattro uscite sono ora confrontate a ogni pubblicazione con quello che il codice produce davvero. Il primo giro ha trovato due blocchi che misuravano su dati di **un’altra pagina** (una foresta che spiegava l’importanza di 64 pixel in un capitolo che parla di colonne di una tabella) e un rilevatore di deriva che dichiarava «nessuno scostamento» su dati che ne avevano parecchio.

### Correzioni

- **Una revisione a tre letture antagoniste sulle sezioni nuove**, una per chi legge il livello facile, una per chi legge quello formale, una che esegue il codice e apre le fonti. Fra le cose raddrizzate: la dimostrazione di un teorema attribuita all’argomento sbagliato; la provenienza del più famoso insieme di dati della statistica, che Fisher stesso segnala come non omogenea e che il libro dava per omogenea; un esperimento sulla copertura di un intervallo la cui conclusione, misurata bene, si rovesciava sulla seconda cifra decimale; e il testo per lettori di schermo di un’animazione, che un difetto nel programma che la disegna spezzava in tre frammenti e che prometteva una forma che il disegno non ha.

(v1-7-0)=

## 1.7.0 · 18 agosto 2026

### Sezioni nuove

- {doc}`Efficienza </Efficienza/overview>`. **Un capitolo su perché il modello che si addestra non è quello che si usa.** Addestrare può essere lento quanto vuole, rispondere no: sono due mestieri diversi, e quasi sempre si mette in produzione lo stesso identico modello. Il capitolo racconta le tre leve per stringerlo (meno bit per numero, meno numeri, un modello nuovo e più piccolo che impara dal grande) e soprattutto **che cosa costano**, perché è la parte che di solito non si dice. Con qualche sorpresa misurata: a quattro bit arrotondare e basta non basta affatto, una rete alleggerita del novantacinque per cento non gira più veloce sul conto di prima, e il grosso di quello che un maestro insegna a un modello piccolo non sono i suoi dubbi ma il fatto che possa commentare esempi che nessuno ha etichettato.
- {doc}`Ricerca e pianificazione </Ricerca/overview>`. **Un capitolo sull’intelligenza artificiale prima dell’apprendimento.** Prima che le macchine imparassero dai dati, decidevano immaginando: se muovo qui lui risponde là, e allora io potrei. Il capitolo parte da un automa a ingranaggi del 1912 che giocava a scacchi senza pensare, mostra perché l’albero dei futuri esploda oltre ogni possibilità fisica, e racconta le due idee che lo rendono percorribile lo stesso: una stima di quanto manca, e l’accorgersi che certi rami non vanno guardati affatto. Finisce sulle tre cose che tutto questo dava per scontate e che nel mondo vero spesso non ci sono: è da lì che comincia l’apprendimento per rinforzo.
- {doc}`Modelli latenti </ModelliLatenti/overview>`. **Un capitolo su come si spiegano i dati con una causa che non si vede.** È l’idea che il libro usava già in quattro punti senza averla mai derivata: comprimere in poche cifre e ricostruire, e scoprire che fra una cifra e l’altra ci sono buchi in cui il modello non è mai stato. Il capitolo deriva il rimedio (chiedere al modello non un punto ma una nuvola, e quanto costa descriverla) e mostra dove quel latente stava già lavorando: nei codec audio, nel deep reinforcement learning, nella diffusione, nei world model.

### Pagine ampliate

- **Tre capitoli inseriti in mezzo hanno cambiato i numeri di tutti gli altri**, e con essi i rimandi. Le pagine che promettevano «il capitolo seguente» parlando di qualcos’altro sono state riscritte col nome del capitolo, che è la forma che non invecchia. Dove il libro spiegava la stessa cosa due volte (la quantizzazione in MLOps, la potatura fra tre capitoli, il biglietto della lotteria) adesso c’è una spiegazione sola e gli altri la richiamano.

### Correzioni

- **Una revisione a nove letture antagoniste sui capitoli nuovi**, ed è la parte che vale la pena leggere se si è già letto qualcosa. Fra le cose raddrizzate: la spiegazione facile della ricerca A* descriveva in realtà un altro algoritmo, meno buono; un conto sull’efficienza contava la metà di quello del capitolo sulla GPU pur dichiarando di essere lo stesso; un miglioramento attribuito alla potatura era in realtà il riavvio dell’ottimizzatore; e una frase su A*, ripresa da un manuale molto citato, era imprecisa nel manuale stesso.

(v1-6-3)=

## 1.6.3 · 17 agosto 2026

### Correzioni

- **Due spiegazioni facili insegnavano il contrario di quello che succede.** Nel capitolo sull'auto-supervisione, la pagina dice che il guasto da evitare è descrivere tutte le fotografie allo stesso modo, e poi raccontava che a impedirlo è la richiesta di non ripetersi. È il contrario: quel guasto lo ferma l'altra richiesta, quella che due ritagli della stessa foto diano la stessa scheda, e la pagina non diceva la cosa da cui dipende tutto, cioè che una casella che scrive sempre lo stesso numero viene messa da parte prima del confronto. Nei modelli a energia, la descrizione «locale» del paesaggio era data per completa a meno dell'altezza complessiva: su un paesaggio a due valli separate se ne perde una per ogni valle, ed è per questo che quei metodi sbagliano le proporzioni fra gruppi di dati distinti. Corrette tutte e due, e anche i riquadri finali che le ripetevano.
- **Diciotto punti in cui la spiegazione facile e quella formale non raccontavano lo stesso oggetto.** Il passo di tempo dei modelli a spazio di stato era presentato come la finezza di un campionamento, «con più lavoro» se piccolo, mentre è la manopola che decide quanto in fretta il sistema dimentica, e stringerla non costa niente. La divisione che protegge l'attenzione dal saturare non aveva nessuna controparte concreta, e con essa spariva un modo in cui l'attenzione si guasta. «Attenzione lineare» sembrava voler dire «più veloce sempre», mentre sotto qualche migliaio di parole il metodo classico vince ancora. E poi: i lucidi di LoRA adattano ma non insegnano, il metodo diretto di allineamento e quello classico non finiscono nello stesso punto, e la verosimiglianza «esatta» di un flusso è esatta su dati leggermente sporcati, non sul file vero.
- {doc}`Prefazione </prefazione>`. **La chiusa torna a legarsi a quello che la precede.** Le ultime due frasi erano rimaste affiancate senza il ponte che le teneva, e il «Perché» finale restava sospeso. Rimesso il nesso, e riportato il paragrafo sul triangolo impossibile alla lunghezza giusta: dice ancora che l'impossibilità è un teorema e che i tre pezzi sono isomorfismi perfetti che non si compongono, in un terzo dello spazio e senza più separare due frasi che si tenevano.

(v1-6-2)=

## 1.6.2 · 17 agosto 2026

### Pagine ampliate

- {doc}`Prefazione </prefazione>`. **Perché il segno di questo libro è un triangolo impossibile.** La prefazione lo mostrava senza dirlo. Adesso aggiunge la ragione, in un paragrafo: quell'impossibilità è un teorema e non un inganno dell'occhio, e dice una cosa che vale per chi rilegge un libro tecnico. I tre pezzi della figura sono tutti e tre perfetti, corrispondono a oggetti veri, e il guasto non è in nessuno dei tre: è che non si compongono. L'errore non ha un luogo.

### Correzioni

- **Le ultime formule della prima metà, e cinque numeri che dicevano il falso.** La spiegazione facile della precisione media definiva un punteggio che a un rilevatore quasi cieco avrebbe dato il massimo invece del minimo; la colpa ripartita all'indietro in una rete era data come «due terzi e un terzo» quando quella ripartizione non c'è; e i tre numeri di un confronto fra inizializzazioni venivano da due esperimenti diversi. Rifatti i conti, e corretti.

(v1-6-1)=

## 1.6.1 · 17 agosto 2026

### Sezioni nuove

- {doc}`Generare suono e musica </Audio/generazione-audio>` (Audio oltre la voce). **Una figura animata per capire perché uno spartito costa poco e pesa molto.** La battuta scorre come su un rullo di pianola, e sotto crescono le due file di simboli: quella a griglia si allunga a ogni sedicesimo anche quando non succede niente, quella a eventi resta ferma e poi salta. È il compromesso di tutta la sezione, visto invece che descritto.

### Correzioni

- **Adesso ogni capitolo saluta, invece di finire dentro un elenco puntato.** Trentaquattro capitoli su trentasei si chiudevano sull'ultima riga del riquadro «Da ricordare»: chi arrivava in fondo trovava tre punti elenco e la pagina che finiva. Ora ognuno ha due o tre righe che dicono che cosa ci si porta dietro e a che cosa serve nel capitolo dopo, che è nominato per nome. È il genere di difetto che non si vede leggendo un capitolo da solo: si vede solo leggendo di fila.
- **Tutte le formule del libro sono state rifatte a mano, e i conti tornano.** Quattrocento e passa blocchi di formule, letti insieme al paragrafo che li commenta, con le verifiche rifatte in Python invece che a occhio: nessuna formula sbagliata. Sono venuti fuori i difetti *attorno* alle formule, che sono quelli che fanno inciampare chi legge: simboli usati senza essere mai presentati, la stessa lettera per due cose diverse nello stesso capitolo, e una F1 di esempio calcolata prendendo un numero da una tabella e l'altro da un'altra.
- **Dove la spiegazione facile prometteva più di quella difficile.** Il libro spiega ogni cosa due volte, e le due spiegazioni devono essere lo stesso oggetto con altri nomi: chi si costruisce l'intuizione sulla versione elementare non deve **disimparare** niente aprendo l'altra. In quattro punti non era così, e l'elementare affermava come certo ciò che la versione formale smentiva dieci righe più sotto (la discesa del gradiente che «converge da qualsiasi punto», il passo di apprendimento del percettrone che «assesta gradualmente»). Riscritti.
- **Le aperture di alcuni capitoli, e un quadro di Magritte.** Visione e linguaggio apriva con lo stesso aneddoto del 1966 con cui apre Visione artificiale, cioè con una ripetizione. Adesso apre con la pipa dipinta da Magritte nel 1929 e la scritta «questa non è una pipa», che è la stessa domanda del capitolo: il disegno non è la cosa, la parola non è il disegno, e la macchina deve attraversare quel doppio confine.

(v1-6-0)=

## 1.6.0 · 16 agosto 2026

### Sezioni nuove

- {doc}`Auto-supervisione </AutoSupervisione/overview>`. **Un capitolo su come si impara senza che nessuno corregga.** Quasi tutto quello che una macchina sa oggi lo ha imparato da dati che nessuno aveva etichettato: il segnale era già dentro i dati, bastava nascondere un pezzo e chiedere di indovinarlo. Il capitolo racconta i quattro modi di fabbricare quel segnale, il guasto che li minaccia tutti (la scorciatoia di rispondere sempre la stessa cosa, che è la risposta perfetta a una domanda mal posta) e i modi di impedirlo. In fondo, il dibattito aperto: quanto conta davvero il rinforzo, se il grosso lo ha già fatto qualcun altro.
- {doc}`Verosimiglianza esatta </VerosimiglianzaEsatta/overview>`. **Un capitolo sui modelli che sanno dire quanto è probabile quello che vedono.** Un generatore di immagini fabbrica figure ma non sa dirti quanto una figura sia verosimile; questa famiglia sì, e lo dice con un numero. Come ci riesce (un pixel alla volta, oppure deformando una nuvola di punti con trasformazioni che si sanno invertire), quanto costa, e la sorpresa che ha messo in crisi tutti: modelli addestrati su cani danno probabilità più alta a fotografie di numeri civici. La ragione è geometrica, e spiega perché «probabile» non vuol dire «tipico».
- {doc}`L'inferenza attiva </WorldModels/inferenza-attiva>` (World Model). **E se la ricompensa non fosse il punto di partenza?** Una sezione sull'idea che agire e capire siano lo stesso gesto: un agente che si muove per ridurre la propria sorpresa cerca informazione dove non ne ha, senza che nessuno gli abbia scritto un premio per farlo. È un'alternativa di principio al rinforzo, con una sorpresa nel finale: il termine che trattiene un modello di linguaggio vicino a quello di partenza, durante l'allineamento, è formalmente lo stesso.

### Pagine ampliate

- {doc}`Evoluzioni e applicazioni </GAN/applicazioni-evoluzioni>` (GAN). **Sotto il cofano di StyleGAN, e il pezzo che manca a metà dei generatori di oggi.** Come si separa lo stile dal contenuto in un volto sintetico, e perché una goccia comparsa in tutte le immagini restò senza spiegazione per un anno. Poi la quantizzazione delle immagini in un vocabolario di simboli, che è la cerniera fra i generatori di immagini e i modelli di linguaggio.
- {doc}`Generare suono e musica </Audio/generazione-audio>` (Audio oltre la voce). **Generare lo spartito invece del suono.** Un minuto di musica scritta come note e durate sta in venti kilobyte; lo stesso minuto registrato ne occupa trenta megabyte. Che cosa si guadagna e che cosa si perde a generare il primo invece del secondo, e perché la scelta di come si scrivono le note decide se la macchina saprà tenere il tempo.

### Correzioni

- **Il libro è stato riletto da capo con l'idea di metterlo in difficoltà**, pagina per pagina, su tutto quello che è arrivato con l'ultima pubblicazione: un lettore che non sa niente di intelligenza artificiale, uno studente che contesta le formule, e uno che apre le fonti una per una. Sono venuti fuori più di cento difetti, e i peggiori erano numeri: un rapporto sbagliato di cento volte, una memoria dichiarata otto volte più grande del vero, un fattore invertito in una formula spiegata due volte (giusta nella versione difficile, rovesciata in quella facile). Corretti tutti, con la prova accanto.
- **Su quasi ogni pagina c'erano mille apostrofi sbagliati, e nessuno li vedeva.** `«Un po' meno»` usciva stampato come `«un po” meno»`, con una virgoletta doppia al posto dell'apostrofo, perché il programma che compone il testo, in italiano, interpretava così ogni apostrofo a fine parola. Il sorgente era giusto, il difetto nasceva nella composizione: 1086 occorrenze, riparate.
- **Le aperture e i congedi.** Il capitolo di matematica cominciava con quarantasette righe di definizioni, e la frase di Galileo in cima non era di Galileo ma una parafrasi da manuale: adesso c'è il passo vero del *Saggiatore*, e ad aprire è von Neumann che consiglia a Shannon di chiamarla entropia «perché nessuno sa davvero cosa sia». Il capitolo sugli agenti diceva che un modello di linguaggio è passivo e non può agire, una pagina dopo averne mostrato uno che comanda un braccio robotico. E dove un capitolo finiva dentro un elenco puntato, adesso c'è una frase che consegna il passaggio a quello dopo.

(v1-5-8)=

## 1.5.8 · 15 agosto 2026

### Sezioni nuove

- {doc}`Conclusioni </Conclusioni/overview>`. **«È più intelligente di noi?» è una domanda a cui manca il dove.** Una sezione nuova sostiene che una macchina non è migliore di noi in senso assoluto, ma nel mondo digitale sì, e che quel vantaggio non viene dall'intelligenza: viene dal terreno. Là dentro tutto è già un numero, ogni azione costa niente e si ripete un miliardo di volte, ogni partita finita lascia scritto chi ha vinto. È una partita in casa. La consolazione («a noi resta il mondo reale») regge solo a metà, perché il campo si allarga da sé. Quello che resta nostro non è un territorio, che si può perdere, ma due cose che nessuna quantità di dati sposta: rispondere di una scelta, e decidere quale partita giocare.

### Pagine ampliate

- {doc}`Introduzione </Introduzione/overview>`. **I dati non sono un giacimento, sono uno scarto, e adesso il libro lo dice.** Nessuno si alza la mattina per produrre dati: li lascia dietro di sé mentre fa altro. C'è un precedente, ed è successo su scala planetaria: l'ossigeno che respiriamo è il rifiuto di certi batteri che forse tre miliardi di anni fa cominciarono a spezzare l'acqua con la luce del sole, e che per centinaia di milioni di anni non cambiò l'aria di niente. La vita che se ne nutre è arrivata molto dopo. Siamo noi. I dati stanno alle macchine come l'ossigeno sta a noi, e l'analogia tiene anche dalla parte scomoda: quello scarto, prima di diventare respiro, fu un veleno.

### Correzioni

- **Il libro dice di sé una cosa sola, e la dice dappertutto uguale.** La copertina del PDF, l'anteprima che compare condividendo un link e il titolo delle pagine portavano tre versioni diverse della stessa frase, e per un po’ ne hanno portate perfino due in fila, che finivano tutte e due su «due volte» e sembravano un'eco. Adesso è una riga sola. E dalla home il libro si può aprire nel browser, oltre che scaricare.

(v1-5-7)=

## 1.5.7 · 15 agosto 2026

### Pagine ampliate

- **Il libro si legge anche nel browser, senza scaricarlo, e ha un indirizzo che non cambia.** Accanto al pulsante che scarica il PDF ce n'è un secondo che lo apre e basta, pagina per pagina. Quel collegamento è il DOI del libro, cioè il suo identificativo permanente: porta sempre all'ultima versione depositata e dà la scheda già pronta per chi deve citarlo in una bibliografia.

### Correzioni

- **La copertina del PDF diceva come è fatto il libro senza dire che libro è.** Portava soltanto la postilla, «l'AI che spiega se stessa… due volte», e chi apriva il file trovava il nome e nient'altro. Adesso c'è anche la riga che dice che cos'è, la stessa che si legge sulla home e nell'anteprima che compare condividendo il link: le tre superfici dicevano tre cose diverse, ciascuna vera per conto suo.

(v1-5-6)=

## 1.5.6 · 15 agosto 2026

### Correzioni

- **Il libro si chiama Paithon Book, con le iniziali maiuscole.** Fin qui il nome si scriveva tutto minuscolo, e dentro una frase si leggeva come un refuso invece che come un titolo. Il logo resta minuscolo, perché è un disegno e compone il segno della marca: è la convenzione normale dei marchi in minuscolo, dove il segno resta com'è e la prosa scrive il nome in tondo.
- **Il logo del libro ha il triangolo al posto della «a».** Il sito aveva già smesso di mettere il bollo accanto alla parola, perché ripeteva lo stesso segno, e la «a» lo incorpora: il libro era rimasto alla versione di prima. Ne è uscito anche un guasto che nessuno avrebbe visto: il logo stampato sulla copertina del PDF era un file fatto a mano una volta, che nessuno rigenerava, e sarebbe rimasto quello vecchio per sempre senza dare alcun errore.
- **Il colophon del PDF diceva che il copyright è di paithon.it**, che però è un dominio e non una persona: i diritti d'autore un dominio non può averli, e in una contestazione la titolarità la prova chi la rivendica. Adesso è intestato a Francesco Messina. E la licenza del codice degli esempi ha un nome scritto per esteso, Apache License 2.0, invece di un rinvio a un file da cercare su GitHub.

(v1-5-5)=

## 1.5.5 · 15 agosto 2026

### Correzioni

- **Tre didascalie di figura rompevano un collegamento nel PDF.** Contenevano una citazione, e nella versione a stampa quel richiamo spezzava il rimando alla bibliografia: la didascalia si leggeva bene, ma il collegamento non portava da nessuna parte, ed è il motivo per cui nessuno se n'era accorto. Le citazioni sono passate nel corpo del testo. Su una delle tre il testo ci ha guadagnato: adesso dice chi fabbricò quei quattro insiemi di dati, e quando.

(v1-5-4)=

## 1.5.4 · 15 agosto 2026

### Pagine ampliate

- {doc}`Prefazione </prefazione>`. **La prefazione dice perché il libro esiste adesso**, e non solo perché esisteva nel 2019. Con l'intelligenza artificiale usata da tutti, e con gli interessi che le si sono mossi attorno, i «pappagalli stocastici» siamo diventati noi: la stessa frase ripetuta per settimane da chi alla fonte non è mai andato, e il modello scaricato e messo in produzione perché lo stavano facendo tutti, spesso senza che nessuno aprisse la licenza. Chi quell'espressione l'ha coniata, cinque anni dopo, osserva la stessa cosa: «la frase ha superato il paper».

### Correzioni

- **La rilettura per la leggerezza è arrivata in fondo: tutti e trentatré i capitoli.** Gli ultimi undici sono i più tecnici del libro, e sono quelli in cui il difetto ricorrente pesava di più: una pagina prometteva «il conto si rifà in due righe» e consegnava un esperimento al computer, pieno per giunta di parole mai dette a chi legge il percorso semplice. Adesso quel conto si fa su un foglio. Il lettore di tredici anni che ha riletto tutto ha anche corretto la matematica di chi gliela stava spiegando, almeno una volta.
- **Ogni correzione è stata poi messa in dubbio da chi doveva smentirla**, e la cosa è servita più del previsto: quasi tutte le smentite fondate riguardavano difetti che la correzione stessa aveva introdotto. Alcuni esempi finiti nel libro sbagliati e ora sistemati: l'anno in cui si scoprì l'inganno del cavallo che sembrava saper contare (1904, e il 1907 è solo l'anno del libro che lo racconta), il meccanismo con cui negli anni Settanta si telefonava gratis con un fischietto, e l'ipotesi di un teorema sull'equità, dichiarata verificata da chi il paper non lo aveva aperto.
- **Un numero che il testo commenta adesso lo stampa il codice.** In una pagina sulle metriche di servizio due percentuali reggevano un paragrafo intero e non comparivano nella tabella che il libro invitava a riprodurre; una delle due era per giunta un caso fortunato del sorteggio. Riparato aggiungendo la colonna al programma, così quel numero non può più scostarsi dal testo.
- **Due capitoli si contraddicevano fra loro.** Le conclusioni dicevano che il costo dell'attenzione supera quello del resto della rete quando il contesto si avvicina alla dimensione del modello: succede sei volte più in là, e il capitolo sui Transformer lo diceva già con i numeri giusti.

(v1-5-3)=

## 1.5.3 · 14 agosto 2026

### Correzioni

- **Altri quindici capitoli sono stati riscritti per essere più leggeri.** A leggerli è stato un ragazzo di tredici anni che non aveva mai sentito parlare di intelligenza artificiale, e ogni punto in cui si è fermato è stato riscritto. In un capitolo si era fermato duecento volte. Le sue domande sono quelle che un libro dovrebbe prevenire: «da che parte tirare, per andare dove?» sulle GAN, «per me giocare e imparare sono la stessa cosa» sul reinforcement learning, «mi fate credere che un numero si spezza in due pezzi senza farmi mai vedere come» sull'attenzione lineare.
- **Il libro non manda più a leggere l'altro livello.** Era il difetto più diffuso: una pagina del percorso semplice concludeva un ragionamento usando qualcosa spiegato solo nella scheda avanzata, che quel lettore per definizione non apre. In un capitolo c'era scritto a lettere («il fattore otto è quello che il livello Superiore chiama 8× di banda buttata via»), in un altro erano quarantasette fra simboli e formule nel testo comune, ora zero.
- **Fatti e numeri: una citazione che gli autori non hanno mai scritto** (attribuita agli autori di HyDE), due numeri che non stanno in nessuna versione dei paper citati, le operatrici del Voder che erano ventiquattro e non venti, e la frase famosa della loro dimostrazione del 1939, che non compare né nell'articolo né nel filmato superstite. Più un esperimento che il lettore non poteva rifare, perché il protocollo taceva su tre dettagli che decidono il risultato.
- **Alcune figure dicevano il contrario del testo.** In due disegni delle GAN la freccia raccontava che è il verdetto a tornare indietro, mentre il capitolo spende tre pagine a spiegare che il verdetto non è ciò che torna. E un grafico dichiarato in scala logaritmica aveva perso l'asse, cioè l'unica cosa che permette di leggerlo.

(v1-5-2)=

## 1.5.2 · 14 agosto 2026

### Pagine ampliate

- **Tre figure nuove**, dove il testo elencava e basta: le quattro mosse con cui si amministra la finestra di contesto, le tre forme di una rete che impara più compiti insieme, e il righello che traduce l'entropia in «quante facce ha il dado equivalente».

### Correzioni

- **Tutti e trentatré i capitoli sono stati riletti da capo**, con il codice eseguito davvero e i conti rifatti a mano. Sono usciti errori di sostanza in quasi ogni capitolo, e i più gravi erano affermazioni che il libro smentiva da sé qualche pagina dopo. Qualche esempio: il clipping di PPO è asimmetrico e taglia da un lato solo, il gradiente della funzione di partizione era annunciato e mai derivato, TransE la composizione ce l'ha (a mancargli è un'altra cosa), e «PALLA sta in sette frame» è falso, il minimo è sei.
- **I sette capitoli d'apertura sono stati riscritti per essere più leggeri.** Un capitolo può essere tutto vero e restare illeggibile, e quelle sono le pagine dove arriva chi non ha basi. A leggerli è stato un ragazzo di tredici anni che non aveva mai sentito parlare di intelligenza artificiale, e ogni punto in cui si è fermato è stato riscritto: il conto con il seno che «non è un conto che posso rifare, è un numero da credere», la matrice che «gira e stira lo spazio» dopo un esempio che non gira niente, gli alberi che «si trovano a casa» prima che il libro dica che cos'è un albero.
- **Le figure adesso mostrano quello che le didascalie promettono.** Erano decine i casi in cui il testo accreditava un disegno di qualcosa che non c'era: «quasi tutte le celle sono vuote» di una griglia piena al settantadue per cento, «ogni tacca vale dieci volte la precedente» di un disegno senza tacche, «le caselle sono numerate a partire da zero» dove i numeri non c'erano. In una figura sulle GAN una freccia diceva l'esatto contrario di quanto la pagina spiega per tre paragrafi.
- {doc}`Allineamento e governance </AIResponsabile/allineamento-e-governance>` (AI responsabile). L'elenco delle pratiche vietate dall’**AI Act** era rimasto indietro: il regolamento europeo del 24 luglio 2026 lo ha allungato, e dal 2 dicembre 2026 sono vietati anche i sistemi che generano materiale intimo non consensuale e materiale di abuso sessuale su minori.
- **La copertina del PDF non è più mezza pagina vuota**: sotto il titolo c'è una conca disegnata con le sue curve di livello e la discesa che ne trova il fondo, cioè il gesto che il libro racconta dal capitolo di matematica fino all'ultimo. La traiettoria è calcolata, non disegnata a occhio.

(v1-5-1)=

## 1.5.1 · 13 agosto 2026

### Correzioni

- {doc}`Introduzione </Introduzione/overview>`. L'intelligenza artificiale è entrata nella vita di tutti i giorni **due volte**, e la pagina ne faceva un momento solo. Negli anni Dieci ci è entrata senza farsi notare, dentro il traduttore automatico, i suggerimenti di un negozio online, i volti riconosciuti nelle fotografie: la usavano tutti e quasi nessuno la chiamava per nome. Dal novembre 2022, con ChatGPT, è diventata qualcosa con cui si parla apposta. Tenerli separati serve a dire la cosa che conta: il salto che si è percepito alla fine del 2022 era cominciato cinque anni prima, e a cambiare non è stato il motore ma il posto in cui l'abbiamo trovato.
- {doc}`Introduzione </Introduzione/overview>`. Le due lunghe pause dell'AI si chiamano **inverni**, che è il termine del campo e quello che il libro usa già nel capitolo sulle reti neurali. La pagina le chiamava «gelate»: due parole diverse per la stessa cosa, in due capitoli che si richiamano, sono un inciampo per chi legge.

(v1-5-0)=

## 1.5.0 · 13 agosto 2026

### Sezioni nuove

- Il libro si scarica in **PDF**, tutto in un file solo. È un secondo formato e non la stampa delle pagine web: impaginato come un libro, con l'indice, i numeri di pagina, le aperture di capitolo e i due livelli di lettura che si riconoscono dal riquadro. Il collegamento è qui nella pagina di apertura, sotto il numero di versione, e punta sempre all'ultima edizione.
- Dove il libro online muove una figura, il PDF mostra **tre fermi immagine** (l'inizio, il mezzo e la fine) e l'indirizzo della pagina in cui quella figura si muove davvero. Vale per tutte e trentacinque le animazioni: tre fotogrammi non sono un'animazione, ma dicono che c'era un prima e un dopo, che è quello che un fermo immagine solo perde.

### Pagine ampliate

- {doc}`Prefazione </prefazione>`. La Prefazione dice adesso **come** il libro è scritto, perché riguarda chi legge: buona parte di queste pagine nasce lavorando con l'intelligenza artificiale, e quello che resta è ciò che l'autore ha verificato e riscritto. Da qui il sottotesto, «l'AI che spiega se stessa... due volte», dove «due volte» vale in tutti e due i sensi: i due livelli di lettura, e la strada che il testo fa per arrivare in pagina. Quello che una AI scrive lo rilegge un'altra AI, che alla stesura non ha partecipato e ha un compito solo, cercare l'errore; poi il testo passa dall'autore, ed è quel passaggio a decidere che cosa resta. La pagina dice anche su che cosa il libro scommette: che versione dopo versione le correzioni diventino rare, e poi rarissime.

### Correzioni

- {doc}`Algebra lineare </Matematica/algebra-lineare>` (Matematica). Più di metà delle figure del libro (185 su 312) non usava i caratteri del progetto: li chiedeva in modo generico, e ogni lettore se le vedeva disegnate con il carattere del proprio sistema, diverso su Windows, su Mac e su Linux. Adesso le etichette delle figure sono le stesse per tutti, e restano le stesse per tutti.
- {doc}`Modelli n-gram </NaturalLanguageProcessing/modelli-ngram>` (Natural Language Processing). Quattro formule erano scritte in un modo che il browser perdona e la stampa no: tre sistemi di equazioni annidati male e una formula spezzata a metà da un a capo. Online si vedevano lo stesso; qui si vedono giuste in tutti e due i formati.

### Impianto

- {doc}`Prefazione </prefazione>`. La **Prefazione** esce da «Fondamenta» e apre il libro per conto suo, dentro una parte nuova dell'indice, «Prima di cominciare». Una prefazione non è un capitolo di fondamenta: viene prima di tutto, e nel libro stampato si vedeva ancora meglio che online.

(v1-4-2)=

## 1.4.2 · 13 agosto 2026

### Pagine ampliate

- {doc}`Support Vector Machine </MachineLearning/svm>` (Machine Learning). La Support Vector Machine adesso si dimostra invece di enunciarsi. La pagina diceva quanto è largo il corridoio fra le due classi, e diceva che nella seconda forma del problema gli esempi compaiono soltanto a coppie, ma non ricavava né l'una né l'altra cosa: e la seconda è quella su cui poggia il kernel trick, cioè metà del capitolo. C'è una sezione nuova che percorre la derivazione per intero, in cinque passi e su tutti e due i livelli, seguendo la strada che Patrick Winston chiamava «l'approccio della strada più larga» nella sedicesima lezione del corso 6.034 del MIT: la regola per decidere da che parte sta un punto, i due vincoli che diventano uno, la larghezza della strada come ombra di una freccia obliqua, i moltiplicatori di Lagrange, e la sostituzione finale da cui salta fuori che i dati entrano nel conto solo attraverso quanto si «vedono» a due a due. Una figura nuova mostra il passaggio che il testo raccontava a parole.
- {doc}`Support Vector Machine </MachineLearning/svm>` (Machine Learning). Nella derivazione ci sono due pause caffè, e stanno dove le faceva Winston: nei due punti in cui il conto ha un salto che nessuna riga di algebra spiega. Alla prima dice alla classe di essere sicuro che, arrivato lì, anche Vapnik fosse uscito a prendere un caffè; alla seconda aggiunge che quelle pause durano mesi. La sezione si chiude dicendo quanto sono durate davvero: il kernel Vapnik ce l'aveva già nella tesi degli anni Sessanta, e capì che era importante trent'anni dopo.

(v1-4-1)=

## 1.4.1 · 13 agosto 2026

### Pagine ampliate

- Sei figure animate nuove, nei capitoli che non ne avevano nessuna. Si anima solo dove il tempo è il contenuto, cioè dove una figura ferma perde davvero qualcosa: il ciclo di addestramento che si ripete mentre la loss scende, i due modi in cui il DQN rompe una correlazione (la memoria che pesca a caso, la copia congelata che resta indietro e poi scatta), la ricorrenza che si svolge a raddoppio in quattro turni invece che in undici, il ciclo di un agente col contesto che si allunga a ogni giro, la spinta avversaria che cresce finché la risposta si ribalta, e il residuo di una rete guidata dalla fisica che si spegne mentre la curva si accosta alla soluzione esatta.

### Correzioni

- Le figure animate calcolano i numeri che mostrano, e adesso lo verificano: ognuna delle sei porta un controllo che ne impedisce la nascita se il risultato non coincide con quello che il capitolo dichiara. Lo scan a raddoppio si genera solo se dà lo stesso risultato di quello in fila, posizione per posizione; la figura dell'attacco solo se a spinta zero la risposta è giusta e oltre la soglia è ribaltata; quella della rete guidata dalla fisica addestra davvero, per trentamila epoche, e ritrova i due numeri stampati nella pagina. Una figura che smentisce il testo è il libro che dice due cose diverse nella stessa pagina, e non un difetto grafico.
- {doc}`Mamba </StateSpaceModel/mamba>` (State Space Model). Il testo attribuiva a Blelloch lo scan parallelo e gli accreditava lo stesso numero di operazioni della versione in fila, ma il codice stampato poche righe dopo è quello a raddoppio, che di operazioni ne fa di più. Adesso il libro distingue le due versioni e dice quale sta scrivendo: il guadagno è sui turni, e si paga in conti.

(v1-4-0)=

## 1.4.0 · 12 agosto 2026

### Pagine ampliate

- Il libro promette due livelli di lettura e un interruttore per scegliere. La promessa era mantenuta dentro le schede e tradita fuori: didascalie delle figure, riquadri «Da ricordare» e paragrafi di raccordo erano scritti nella lingua del livello Superiore, e sono proprio le cose che non si possono saltare. Undici lettori hanno segnalato centodiciotto punti in cui ci si perde. Sono stati riscritti, e i riquadri «Da ricordare» sono ora sui due livelli come il libro dichiara.
- Tredici figure animate nuove, dove il tempo è il contenuto: le fusioni del BPE che accorciano il testo, k-means che si corregge da solo, la finestra che scorre sul suono e riempie lo spettrogramma, la online softmax di Flash Attention che si ricalibra a ogni blocco, una rete di Hopfield che ripara un ricordo scendendo di energia, la validazione a origine mobile che non guarda mai nel futuro, il collasso della CTC con la controprova di cosa succede invertendo i due passi, il feromone delle formiche che decide la strada corta senza che nessuna l'abbia misurata, il falsario della GAN che raggiunge il vero mentre l'esperto si arrende, la deriva dei dati misurata dalla distanza che le dà il nome, il palo che scorre dietro una finestrella e il movimento che si perde, il sogno di un modello del mondo che si stacca dalla realtà dopo sedici passi, e il cammino dalla baseline all'ingresso lungo cui il gradiente è ancora vivo. Ognuna calcola i propri numeri, e si rifiuta di nascere se smettono di combaciare con quelli del testo.
- La notazione matematica adesso è la stessa in tutto il libro. Il grassetto dice che un oggetto ha più di una componente (una matrice, un vettore), e il tondo che è un numero solo: senza quella distinzione una lettera è ambigua, e il lettore deve indovinare dal contesto a ogni riga. Due capitoli su trentatré la applicavano, ed erano i primi due, quelli che la insegnano; da lì in poi spariva. Sono circa duemila simboli rivisti uno per uno, non sostituiti a macchina: i conteggi, le funzioni e le probabilità restano tondi anche quando si scrivono maiuscoli, ed è la trappola in cui una correzione automatica sarebbe caduta.

### Correzioni

- Correzioni in tutti i capitoli, elencate nelle voci che seguono. La più importante non è nessuna delle singole, ed è che il codice del libro, da questa versione, gira davvero.
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
- {doc}`Dati su misura </PyTorch/dati-su-misura>` (PyTorch). Perché impacchettare un dataset grande convenga, e non per la ragione che si immagina: il costo dominante sta nell’aprire i file, non nel decodificarli. Nello stesso passaggio si calcolano le statistiche che serviranno per normalizzare.
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
- {doc}`MCTS e AlphaGo </DeepReinforcementLearning/mcts-alphago>` (Deep Reinforcement Learning). La ricerca ad albero Monte Carlo, che il libro invocava quattro volte come spiegazione di AlphaGo e MuZero senza averla mai insegnata. Le sue quattro mosse, il perché si sceglie la mossa più visitata e non quella con la media migliore, e il fatto che la formula che le fa scoprire dove guardare è la stessa dei bandit a più braccia.
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

- Correzioni in tutti i capitoli dove servivano: date, attribuzioni, esempi numerici, notazione.
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
