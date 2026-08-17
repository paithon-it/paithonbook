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
- **Su quasi ogni pagina c'erano mille apostrofi sbagliati, e nessuno li vedeva.** «Un po' meno» usciva stampato come «un po” meno», con una virgoletta doppia al posto dell'apostrofo, perché il programma che compone il testo, in italiano, interpretava così ogni apostrofo a fine parola. Il sorgente era giusto, il difetto nasceva nella composizione: 1086 occorrenze, riparate.
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

- **Tutti e trentatré i capitoli sono stati riletti da capo**, con il codice eseguito davvero e i conti rifatti a mano. Non è una rilettura di forma: sono usciti errori di sostanza in quasi ogni capitolo, e i più gravi erano affermazioni che il libro smentiva da sé qualche pagina dopo. Qualche esempio: il clipping di PPO è asimmetrico e taglia da un lato solo, il gradiente della funzione di partizione era annunciato e mai derivato, TransE la composizione ce l'ha (a mancargli è un'altra cosa), e «PALLA sta in sette frame» è falso, il minimo è sei.
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

- Il libro si scarica in **PDF**, tutto in un file solo. Non è la stampa delle pagine web: è un secondo formato, impaginato come un libro, con l'indice, i numeri di pagina, le aperture di capitolo e i due livelli di lettura che si riconoscono dal riquadro. Il collegamento è qui nella pagina di apertura, sotto il numero di versione, e punta sempre all'ultima edizione.
- Dove il libro online muove una figura, il PDF mostra **tre fermi immagine** (l'inizio, il mezzo e la fine) e l'indirizzo della pagina in cui quella figura si muove davvero. Vale per tutte e trentacinque le animazioni: tre fotogrammi non sono un'animazione, ma dicono che c'era un prima e un dopo, che è quello che un fermo immagine solo perde.

### Pagine ampliate

- {doc}`Prefazione </prefazione>`. La Prefazione dice adesso **come** il libro è scritto, perché riguarda chi legge: buona parte di queste pagine nasce lavorando con l'intelligenza artificiale, e quello che resta è ciò che l'autore ha verificato e riscritto. Da qui il sottotesto, «l'AI che spiega se stessa... due volte», dove «due volte» vale in tutti e due i sensi: i due livelli di lettura, e la strada che il testo fa per arrivare in pagina. Quello che una AI scrive lo rilegge un'altra AI, che alla stesura non ha partecipato e ha un compito solo, cercare l'errore; poi il testo passa dall'autore, ed è quel passaggio a decidere che cosa resta. La pagina dice anche su che cosa il libro scommette: che versione dopo versione le correzioni diventino rare, e poi rarissime.

### Correzioni

- {doc}`Algebra lineare </Matematica/algebra-lineare>` (Matematica). Più di metà delle figure del libro (185 su 312) non usava i caratteri del progetto: li chiedeva in modo generico, e ogni lettore se le vedeva disegnate con il carattere del proprio sistema, diverso su Windows, su Mac e su Linux. Adesso le etichette delle figure sono le stesse per tutti, e un controllo automatico impedisce che la cosa ricapiti.
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

- Le figure animate calcolano i numeri che mostrano, e adesso lo verificano: ognuna delle sei porta un controllo che ne impedisce la nascita se il risultato non coincide con quello che il capitolo dichiara. Lo scan a raddoppio si genera solo se dà lo stesso risultato di quello in fila, posizione per posizione; la figura dell'attacco solo se a spinta zero la risposta è giusta e oltre la soglia è ribaltata; quella della rete guidata dalla fisica addestra davvero, per trentamila epoche, e ritrova i due numeri stampati nella pagina. Una figura che smentisce il testo non è un difetto grafico: è il libro che dice due cose diverse nella stessa pagina.
- {doc}`Mamba </StateSpaceModel/mamba>` (State Space Model). Il testo attribuiva a Blelloch lo scan parallelo e gli accreditava lo stesso numero di operazioni della versione in fila, ma il codice stampato poche righe dopo è quello a raddoppio, che di operazioni ne fa di più. Adesso il libro distingue le due versioni e dice quale sta scrivendo: il guadagno è sui turni, e si paga in conti.

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
