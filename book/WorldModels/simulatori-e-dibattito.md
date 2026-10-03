# Simulare il mondo: video generativi e un dibattito aperto

Il 15 febbraio 2024 OpenAI presenta Sora, un modello che da una descrizione
testuale genera video fotorealistici fino a un minuto. Come sia fatto dentro
l'ha già raccontato la {doc}`sezione sui diffusion
transformer </ModelliDiffusione/diffusion-transformer>`; qui interessa che
cosa dimostra. Il rapporto che lo accompagna {cite}`brooks2024video` ha un
titolo che è una tesi: *Video generation models as world simulators*, i modelli
di generazione video come simulatori di mondo. È un documento aziendale con dimostrazioni scelte da chi le pubblica, non un
articolo passato da revisione. La scommessa è dichiarata nell'ultima riga: continuare a scalare i
modelli video è «una strada promettente verso lo sviluppo di simulatori capaci
del mondo fisico e digitale». A sostegno, il rapporto elenca capacità che
dichiara *emerse* senza essere state programmate: coerenza tridimensionale
delle scene, oggetti che continuano a esistere quando escono dall'inquadratura,
un pittore che lascia sulla tela pennellate che restano.

Ma lo stesso rapporto, poche righe più in basso, mostra il video in cui la
scommessa incespica: un bicchiere si solleva da solo e si inclina, il liquido
si stende sul tavolo come una lastra, e quello che resta dentro non cade
nemmeno quando il vetro, a mezz'aria, è coricato su un fianco; poi il bicchiere
ricade senza rompersi. Gli autori lo ammettono senza giri di parole: il modello
«non simula accuratamente la fisica di molte interazioni di base, come il vetro
che va in frantumi», e mangiare qualcosa «non produce sempre il cambiamento di
stato corretto». In quell'immagine c'è per intero la domanda con cui il
capitolo si chiude: un video che *sembra* vero dimostra che il modello ha
*capito* il mondo, o soltanto che ha imparato a imitarne le apparenze? I
pittori fiamminghi del Quattrocento rendevano la luce nel vetro quasi due
secoli prima che Snell scrivesse la legge della rifrazione, nel 1621: copiare
bene un fenomeno e possederne il meccanismo sono due cose diverse, e
distinguere l'una dall'altra, in una rete neurale, è più difficile che in un
quadro.

## Mondi da guardare, mondi da giocare

Un video, per quanto perfetto, si guarda e basta: il futuro che mostra è già
deciso. Nel 2024 un gruppo di Google DeepMind compie il passo successivo con
**Genie** {cite}`bruce2024genie`, presentato alla conferenza ICML e premiato
fra i migliori articoli dell'anno: non generare *video*, ma **ambienti
interattivi**, cioè mondi in cui a ogni fotogramma è chi gioca a decidere la
mossa, e il modello risponde generando il fotogramma coerente con quella
mossa. Il materiale di partenza è sorprendentemente povero: circa 30.000 ore di
video di videogiochi a piattaforme, quelli in due dimensioni con i personaggi
che corrono e saltano alla Super Mario, setacciate da un mucchio iniziale di
244.000 ore raccolte da internet. Video e basta: nessuno ha detto al modello
quali tasti premevano i giocatori.

`````{tab} Elementare

La differenza tra guardare un film e giocare a un videogioco è tutta qui: il
film va avanti da solo, il videogioco deve rispondere alle *tue* mosse. E per
rispondere a una mossa che nessuno ha previsto (premi «salta» proprio su
quell'orlo di burrone), il gioco deve sapere *che cosa succede dopo* in ogni
situazione possibile: deve avere, in qualche forma, un modello del suo piccolo
mondo. Genie è un videogioco senza motore di gioco: dietro non c'è un
programma con le regole scritte da qualcuno, c'è una rete che le regole le ha
assorbite guardando.

Il colpo di scena è che nessuno gli ha mai mostrato un joystick. Come fa a
sapere quali comandi esistono? I ricercatori gli impongono una regola severa:
per spiegare che cosa cambia tra un fotogramma e il successivo può usare
soltanto otto «mosse tipiche», e quali siano le otto più utili deve scoprirlo
da solo, guardando migliaia di ore di partite. Otto lo hanno deciso loro,
pensando a chi poi giocherà: un joystick con cento pulsanti sarebbe
ingovernabile. E da un fotogramma all'altro cambiano molte più cose di otto:
l'acqua che scorre sul fondale, il nemico che avanza, il corpo che ricade a
terra dopo il salto. Nelle otto ci finisce soltanto quello che dipende da chi
gioca; il resto lo tira avanti da sé la parte di rete che disegna il
fotogramma dopo, che di cadute ne ha viste migliaia. Quelle mosse diventano i pulsanti di un
joystick che Genie si è inventato da solo. Quando giochi, scegli un numero da
1 a 8: che il pulsante 3 significhi «salta» lo scopri provando, come davanti a
una console senza libretto di istruzioni.

`````

`````{tab} Superiore

Genie (circa 11 miliardi di parametri in totale) è composto da tre moduli: un
*tokenizer* video che comprime i fotogrammi in token discreti, un **modello di
azioni latenti** e un **modello di dinamica** (di gran lunga il più grande dei
tre) che predice i token del fotogramma successivo. Tutti e tre sono
trasformatori spazio-temporali, con l'attenzione causale nel tempo: il
tokenizer è un autoencoder a quantizzazione vettoriale (VQ-VAE) di questo
tipo, il modello di azioni latenti è anch'esso un VQ-VAE, addestrato a
ricostruire il fotogramma successivo dai precedenti e da un'azione scelta in un
codebook minuscolo, e il modello di dinamica è un trasformatore di tipo
MaskGIT, che riempie i token mascherati del fotogramma successivo in più
passate invece che uno alla volta. L'addestramento ha due fasi: prima il
tokenizer, poi azioni latenti e dinamica insieme {cite}`bruce2024genie`.

Il cuore concettuale è il secondo modulo, che affronta il problema
dell'assenza di etichette: i video di internet non dicono quale azione è stata
premuta. La soluzione è inferirla come variabile latente *discreta*:

$$
\mathbf{z}_t = \mathrm{tok}(\mathbf{x}_{1:t}),
\qquad
\tilde{a}_t = q\big(f_\phi(\mathbf{x}_{1:t+1})\big) \in \{1, \dots, 8\},
\qquad
\hat{\mathbf{z}}_{t+1} = g_\theta\big(\mathbf{z}_{1:t},\, \tilde{a}_{1:t}\big),
$$

dove $\mathbf{x}_{1:t+1}$ sono i fotogrammi osservati fino al tempo $t+1$;
$\mathrm{tok}$ è il tokenizer, il primo dei tre moduli, che di ogni fotogramma
fa una manciata di token discreti $\mathbf{z}_t$ guardando anche quelli che lo
precedono (è causale nel tempo, non lavora un'immagine per volta: fa lo stesso
mestiere del $\mathbf{z}$ del VAE nel world model di Ha e Schmidhuber, ma con
un vocabolario finito invece di 32 numeri continui); $f_\phi$ è un encoder che riassume la
transizione dal fotogramma $t$ al $t+1$; $q$ è una quantizzazione vettoriale
su un codebook di appena $|\mathcal{A}| = 8$ codici (un tetto fissato dagli autori perché il joystick resti
maneggiabile per un giocatore umano); e $g_\theta$ è il modello di dinamica,
che predice i token del fotogramma successivo a partire dai token del
passato *e* dall'azione latente $\tilde{a}_t$. A riportare i token in pixel è
il decoder del tokenizer: il modello di dinamica non vede mai un pixel e non ne
produce mai uno. Il collo di
bottiglia è la chiave: potendo trasmettere a $g_\theta$ solo tre bit per
passo, $\tilde{a}_t$ è spinta a codificare il *cambiamento controllabile*
(la mossa) e a lasciare tutto il resto (sfondo, fisica, inerzia) al modello di
dinamica. In inferenza il modello di azioni sparisce e l'azione la fornisce
l'utente, fotogramma per fotogramma. Le azioni apprese risultano
semanticamente coerenti tra ambienti mai visti; addestrando lo stesso schema
su video di manipolazione robotica (il dataset RT-1), emergono allo stesso
modo azioni consistenti senza alcuna etichetta: indizio che la ricetta non è
legata ai platform.

Fa il contrario di Genie, e per questo aiuta a capirlo, GameNGen di Valevski e
colleghi {cite}`valevski2025diffusion`: un modello di diffusione condizionato
sulle azioni, addestrato sulle partite registrate di un agente di
reinforcement learning a *DOOM*, che simula il gioco a 20 fotogrammi al secondo
su una sola TPU, e di cui valutatori umani distinguono le brevi clip da quelle
vere poco meglio del caso. Qui le azioni sono osservate e non inferite, e il
gioco è uno solo: è un motore di gioco neurale, non un modello che impara da
video qualsiasi.

`````

La discendenza di Genie è andata avanti a passo rapido. Qui la raccontiamo solo
per quello che ha di strutturale: il resto invecchia in fretta, e vale
l'avvertenza fatta per Sora, perché la fonte, qui, sono annunci sul blog di
DeepMind, senza un articolo che descriva il modello. Due passi contano. Il
primo, **Genie 2** (4 dicembre 2024), esce dal piatto del disegno a due
dimensioni: da una singola immagine genera mondi a tre dimensioni, esplorabili
con tastiera e mouse, con acqua, fumo e gravità, e coerenti fino a un minuto
{cite}`parkerholder2024genie2`. Il secondo, **Genie 3** (5 agosto 2025), genera
il mondo in tempo reale, a 24 fotogrammi al secondo, mentre lo si attraversa,
lo tiene coerente per qualche minuto, e accetta eventi richiesti con una frase
(«fa’ piovere», «aggiungi un cane») {cite}`parkerholder2025genie3`.

I limiti li elencano gli autori stessi, e contano più delle immagini
spettacolari. Le sessioni si misurano in minuti. Il repertorio di comandi è
ristretto: quel che scarseggia sono le azioni possibili, non i posti dove
andare. Un testo leggibile compare in scena quasi solo se lo si è scritto nella
descrizione di partenza. E mettere più agenti autonomi nello stesso mondo resta
problematico. È l'elenco che separa una dimostrazione da un
prodotto, e in questo campo la distanza fra le due cose va sempre tenuta a
mente.

## La scacchiera nella macchina: gli LLM hanno un world model?

Genie *deve* avere un modello del suo mondo, per costruzione: è la sua unica
funzione. La domanda si fa più spinosa quando la voltiamo verso i modelli di
linguaggio, i grandi modelli di linguaggio dell'apertura, gli LLM
{cite}`brown2020language`, addestrati a fare
una cosa sola: indovinare la parola che viene dopo (per la precisione il
token successivo, cioè il pezzetto di parola con cui questi modelli
lavorano). L'apertura del capitolo ha lasciato in sospeso un esperimento,
citandolo di sfuggita: è il momento di raccontarlo, perché è il tentativo più
pulito di rispondere con i dati anziché con gli slogan.

Nel 2023 Kenneth Li e colleghi (tra gli altri David Bau, Fernanda Viégas e
Martin Wattenberg, nomi noti dell'interpretabilità, il campo che studia che
cosa succede dentro una rete) pubblicano a
ICLR l'esperimento oggi noto come **Othello-GPT** {cite}`li2023emergent`.
Prendono un piccolo GPT (8 strati, la stessa architettura dei modelli di
linguaggio) e lo addestrano su un solo tipo di testo: sequenze di mosse del
gioco dell'Otello, scritte come liste di caselle («E3, D2, …»), per venti
milioni di partite generate a tavolino da un programma. Una parola sulle
regole, perché tutto l'esperimento poggia lì: nell'Otello si posano pedine su
una griglia di 64 caselle, e una mossa vale soltanto se accerchia almeno una
pedina avversaria, che allora cambia colore. Le caselle in cui è consentito
posare cambiano quindi a ogni turno, e dipendono da com'è messa l'intera
scacchiera. Il modello non ha mai visto una scacchiera, non conosce le regole,
non sa che esistono pedine bianche e nere: per lui le mosse sono token, come
parole. Eppure, a fine addestramento, propone mosse *legali* con un tasso
d'errore dello 0,01%: una mossa illegale ogni diecimila. La domanda, a quel
punto, è una sola: ci
riesce accumulando statistiche di superficie sulle sequenze (dopo «E3, D2»
viene spesso «C4») o si è costruito dentro, da qualche parte, una scacchiera?

`````{tab} Elementare

Una persona che non ha mai visto una scacchiera e ha soltanto
*ascoltato* migliaia di radiocronache di partite: nomi di caselle in fila,
nient'altro. Dopo anni di ascolto sa proseguire una radiocronaca con mosse che
non fanno mai arrabbiare gli arbitri. Ha in testa una scacchiera immaginata, o
solo un enorme orecchio per le frasi tipiche?

Con una persona non potremmo saperlo. Con una rete sì, perché possiamo
guardarle dentro. La rete lavora una mossa in otto tappe di calcolo una dopo
l'altra (gli *strati*), e ogni tappa produce una fila di numeri che si può
leggere uno per uno: è l'attività interna, la sola cosa che la rete abbia in
testa. Gli autori la usano in due passi. Primo passo: addestrano un piccolo
«lettore del pensiero», una seconda rete che guardando soltanto quei numeri
deve indovinare dove sono le pedine. Ne provano due. Il lettore più semplice fa
soltanto somme pesate di quei numeri, e ci capisce poco: sbaglia circa una
casella su cinque, poco meglio di quanto sbaglierebbe davanti a una rete che
non ha mai imparato niente. Uno con un passaggio di calcolo in più scende sotto
le 2 caselle su 100. E poco dopo si è visto che al lettore semplice bastava
cambiare la domanda: non «questa pedina di che colore è» ma «è mia o
dell'avversario». L'informazione «com'è messa la scacchiera» *dentro la rete
c'è*, anche se nessuno gliel'ha mai chiesta; quanto sia facile tirarla fuori
dipende però da chi la cerca e da come la chiede. Secondo passo, il più bello:
il test del falso ricordo. Gli sperimentatori entrano in quei numeri e li
ritoccano, spostando una pedina *nella mente* della rete: non nella sequenza di
mosse, che resta identica. Se la scacchiera interna fosse un ornamento, le
mosse proposte non cambierebbero. Invece cambiano, e in modo coerente con la
scacchiera contraffatta: la rete gioca in base a ciò che «crede» di vedere. Non
è un pappagallo di sequenze: dentro c'è un piccolo mondo, e lo usa.

`````

`````{tab} Superiore

Lo strumento è il **probing**: una sonda $p_\psi$ (un classificatore addestrato
a parte) riceve le attivazioni $\mathbf{h}_t^{(\ell)}$ dello strato $\ell$ al
passo $t$ e deve predire lo stato di ciascuna delle 64 caselle (vuota, nera,
bianca). Le sonde *lineari* falliscono (errore fra il 20,4% e il 23,1% a
seconda dello strato, appena meglio del 26,7-28,9% che si ottiene sondando una
rete con pesi casuali), quelle *non lineari* (un MLP a uno strato nascosto)
arrivano all’1,7% di errore al settimo strato degli otto: lo stato della
partita è ricostruibile quasi per intero dalle attivazioni. Quello sintetico è
anche il caso più favorevole: addestrato su partite di campionato, poco più di
centomila invece dei venti milioni generati a tavolino, lo stesso GPT sbaglia
il 5,17% delle mosse legali, e le sonde non lineari non scendono sotto il 9,4%
di errore. Poiché una sonda potrebbe leggere una correlazione senza ruolo
causale, il passo decisivo è l’intervento: si modificano le attivazioni con una
discesa di gradiente finché la sonda vi legge una scacchiera contraffatta, si
lascia proseguire il calcolo e si osserva che la distribuzione sulle mosse
legali si adegua alla scacchiera modificata, non alla sequenza di input. La
rappresentazione, dunque, *guida* la predizione.

Un poscritto metodologico, prima di tirare le somme. Neel Nanda e collaboratori
(2023) {cite}`nanda2023emergent` hanno mostrato che la rappresentazione è in
realtà *lineare*, purché la si cerchi nel sistema di riferimento giusto: non
«nero/bianco» ma «mia/dell'avversario», relativo a chi muove. L'1,7% delle
sonde non lineari, quindi, non diceva che l'informazione fosse codificata in
modo intricato: quelle sonde stavano compensando una scelta di coordinate. Nel
loro GPT, la sonda lineare su «mia/dell'avversario» arriva fino al 99,6% di
accuratezza, contro il 75% circa di quella su «nero/bianco», e intervenire
lungo le direzioni che trova cambia le mosse previste. È
il monito che vale per ogni probing, ed è lo stesso incontrato con le sonde di
V-JEPA: quel che una sonda estrae dipende dalle coordinate in cui la si fa
guardare e da quanto la si lascia lavorare, e va dichiarato insieme al
risultato.

`````

Come si tengono insieme questi risultati? Il dibattito ha due letture, ed è
onesto dire che nessuna delle due ha vinto.

La prima: Othello-GPT dimostra che la pura predizione del token successivo
*può* far emergere una rappresentazione interna dello stato del mondo. La
caricatura del modello che «rimescola frasi senza rappresentarsi nulla»,
almeno in questo caso controllato, è smentita dai fatti.

La seconda: un mondo di 64 caselle con regole fisse è lontanissimo dal mondo
fisico, e c'è un secondo esperimento che raffredda gli entusiasmi. Un gruppo
guidato da Keyon Vafa {cite}`vafa2024evaluating` ha addestrato una rete sui
percorsi dei taxi di New York. Le indicazioni che dà, svolta per svolta, sono
valide quasi sempre; ma se dalle sue previsioni si ricostruisce la mappa che ha
in testa, vengono fuori strade che non esistono e cavalcavia impossibili, e
basta imporre qualche deviazione perché smetta di funzionare. Le
rappresentazioni emerse per questa via, insomma, possono essere frammentarie:
buone finché si resta nelle situazioni su cui la rete si è addestrata, fragili
appena se ne esce (**fuori distribuzione** si dice appunto di tutto ciò che al
momento dell'addestramento non c'era).

È l'eco di una prudenza già incontrata nella {doc}`sezione su tendenze e limiti
dei Transformer </Transformers/tendenzefuture>`, a
proposito di allucinazioni e comprensione: su che cosa i modelli capiscano
davvero, il dibattito scientifico è tutt'altro che chiuso. Da una parte c'è
chi, come LeCun, ritiene che serva un'architettura pensata apposta per
prevedere il mondo, ed è la strada JEPA delle sezioni precedenti. Dall'altra
c'è chi scommette che basteranno la taglia dei modelli e la quantità di dati
{cite}`kaplan2020scaling` a far maturare un modello del mondo dentro i modelli
di linguaggio. Il lettore arrivato fin qui ha gli strumenti per seguire la
partita senza tifare.

## Dove i world model lavorano già

Mentre il dibattito continua, i world model lavorano. In robotica la strada
l'abbiamo già vista nella {doc}`sezione sulla JEPA </WorldModels/jepa>`: V-JEPA
2, nella variante condizionata sulle azioni, usa le previsioni nello spazio
delle rappresentazioni per *pianificare* (provare mentalmente i comandi
possibili, uno alla volta, e scegliere quello che avvicina il braccio
all'obiettivo) su robot mai visti in addestramento. Nella guida autonoma il
problema sono gli scenari rari: il bambino che sbuca tra due auto, il carico
che cade dal camion. Raccoglierli su strada è impraticabile, oltre che
inaccettabile; un world model generativo li produce in quantità e in sicurezza.
Un esempio pubblico è GAIA-1, presentato dalla società inglese Wayve nel
settembre 2023: impara da video di guida, insieme al testo che li descrive e ai
comandi dell'auto, e genera scene nuove in cui si possono scegliere il
comportamento del veicolo e gli elementi della strada {cite}`hu2023gaia`. Nei
videogiochi e negli ambienti di addestramento, infine, il cerchio si chiude:
DeepMind presenta Genie 2 esplicitamente come generatore di ambienti illimitati
in cui addestrare e valutare agenti; il rimedio a un vizio storico
dell'apprendimento per rinforzo, dove i programmi che imparano per tentativi
finiscono per sapere a memoria i pochi ambienti disponibili invece di imparare
ad adattarsi.

C'è poi una ragione strutturale per cui i video sono un candidato naturale
all'apprendimento auto-supervisionato su larga scala.

`````{tab} Elementare

Qualcuno deve scrivere «gatto» sotto la foto del gatto, tradurre la frase,
assegnare il voto: i dati con le etichette costano, ed è il collo di bottiglia
di sempre. Il video no: è un giacimento sterminato in cui la correzione è
*gratis*. Vuoi sapere se il modello ha previsto bene? Aspetta il fotogramma
successivo: la risposta esatta arriva da sola, trenta volte al secondo,
centomila volte per ogni ora di filmato. Per giunta ogni video è un piccolo
esperimento di fisica già eseguito (bicchieri che cadono, palle che rimbalzano,
porte che sbattono) registrato senza che nessuno lo abbia allestito. E il
testo, invece, non è infinito. La {doc}`sezione su tendenze e limiti dei
Transformer </Transformers/tendenzefuture>` lo dice: più un modello è grande,
più testo pretende per essere addestrato come si deve, e le raccolte prese dal
web potrebbero esaurirsi come fonte di scrittura di qualità. Le pagine scritte
dagli esseri umani restano quelle che sono. Il video è la più grande riserva di
esperienza del mondo non ancora spremuta: ecco perché tutti scavano qui.

Il giacimento però non regala tutto. Chi si mette a prevedere ogni puntino
dello schermo passa il tempo anche sui riflessi del pavimento, che di quel che
sta succedendo non dicono nulla. E indovinare il fotogramma dopo in una scena
vista mille volte non vuol dire aver capito perché le cose cadono: il bicchiere
che si rovescia e tiene il succo dentro anche coricato a mezz'aria sta lì a
ricordarlo.

`````

`````{tab} Superiore

È la stessa logica auto-supervisionata che ha alimentato gli LLM (il bersaglio
dell'addestramento è il dato stesso, spostato nel tempo) applicata a un
serbatoio più grande di ordini di grandezza: il target è $\mathbf{x}_{t+1}$ (o
una sua rappresentazione $\mathbf{s}_{t+1}$, nella scelta JEPA), la loss una
verosimiglianza o una distanza predittiva, l'annotatore nessuno. Se la lezione
delle leggi di scala {cite}`kaplan2020scaling` è che le prestazioni crescono con
dati e calcolo secondo regolarità prevedibili, i video sono il posto naturale
dove proseguire la curva quando il testo si esaurisce. Le incognite però sono
due: *dove* predire (lo spazio dei pixel obbliga a modellare dettagli
irrilevanti; lo spazio latente rischia il collasso e va regolarizzato) e *che
cosa* la predizione garantisce. Sulla seconda i dati cominciano a esserci, e non
vengono dagli annunci. Kang e colleghi addestrano generatori video di taglia
crescente su un mondo bidimensionale sintetico, governato da leggi della
meccanica note, e misurano se i video generati le rispettano. I regimi sono
tre: dentro la distribuzione d'addestramento la generalizzazione è quasi
perfetta; nella ricombinazione di fattori già visti, ma mai insieme, migliora
in modo misurabile con la scala; fuori dalla distribuzione non migliora. E il
modello si comporta come chi richiama il caso d'addestramento più simile invece
di applicare la legge, dando più peso al colore di un oggetto che alla sua
velocità {cite}`kang2024far`. Il banco Physics-IQ, costruito su riprese vere di
esperimenti di meccanica, fluidi e ottica, trova lo stesso sui generatori più
noti: il realismo visivo non predice la correttezza fisica
{cite}`motamed2025generative`. Prevedere bene i fotogrammi tipici, insomma, non
equivale ad aver interiorizzato le leggi che li generano.

`````

## Il filo del capitolo

Riavvolgiamo. Kenneth Craik, 1943: un organismo con un «modello in scala
ridotta» della realtà può provare le alternative nella testa e reagire al futuro
prima che arrivi. Ha e Schmidhuber, 2018: un agente si allena dentro il proprio
sogno e torna nel gioco vero più bravo di prima, e i Dreamer fanno di quel sogno
un metodo generale. Le JEPA: prevedere sì, ma nello spazio delle
rappresentazioni, lasciando cadere i dettagli che non contano; e la lingua in
cui LeCun le ha scritte è quella del {doc}`capitolo sui modelli a energia
</ModelliEnergia/overview>`, un voto di compatibilità dato a ogni coppia di
presente e futuro, la tradizione che nelle reti neurali parte dal lavoro di
Hopfield del 1982. L'inferenza attiva: percepire, agire e imparare come tre modi
di tenere bassa la stessa quantità, con quel che si desidera scritto dentro il
modello invece che in un premio a parte. E infine Sora e Genie: la previsione
fatta spettacolo, fotogrammi interi di futuro. Il filo che attraversa
ottant'anni è uno solo: in questa tradizione di ricerca l'intelligenza è la
capacità di prevedere, e di usare le previsioni per agire.

Che cosa manca, lo si può dire con la stessa calma. Manca la composizionalità: i
simulatori attuali sanno muoversi *fra* le scene che hanno visto, mescolandole e
sfumando dall'una all'altra (in gergo si dice che le **interpolano**);
ricombinare pezzi noti in modi nuovi migliora con la scala ma resta imperfetto,
e le situazioni davvero nuove, fuori da quello che hanno visto, che sono il
forte delle menti biologiche, restano fuori portata. Manca la causalità:
prevedere ciò che *segue* non è capire ciò che *provoca*. Un bambino la
differenza la esplora da sé, rovesciando bicchieri apposta: vedere due cose che
vanno sempre insieme è un conto, andarne a toccare una per vedere che ne è
dell'altra è un altro. Nei modelli quella differenza è ancora poco marcata. E
manca la pianificazione a lungo orizzonte: l'errore dei modelli si accumula
passo dopo passo, ed è il difetto che il capitolo ha incontrato per primo,
quando l'agente si allenava dentro una copia imprecisa del gioco (in gergo si
chiama *model bias*, la piega sistematica del modello). I sogni dentro cui si
può ancora pianificare sono corti: una quindicina di passi immaginati per i
Dreamer, gli eredi del sogno di Ha e Schmidhuber, e un passo solo per il world
model che guida il braccio robotico. I minuti di cui si parla per i simulatori
generativi sono un'altra cosa: sono la lunghezza di un video da *guardare*, non
di un piano da eseguire. Nessuna di queste lacune autorizza il catastrofismo («è
tutto un trucco») né il trionfalismo («è fatta, questione di mesi»): autorizzano
un cantiere. E le date mostrano quanto spesso un'idea preceda di decenni lo
strumento che la rende utilizzabile: il modello in scala ridotta di Craik è del
1943, l'agente che si allena nel proprio sogno del 2018; la rete di Hopfield,
del 1982, ha ricevuto il Nobel per la fisica nel 2024.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Nel 2024 OpenAI presenta i propri generatori di video come «simulatori di
  mondo» {cite}`brooks2024video`, ma è il documento stesso a mostrare dove il
  trucco si vede: il bicchiere che si rovescia e tiene il succo dentro anche
  coricato a mezz'aria. Copiare bene un fenomeno e possederne il meccanismo
  restano due cose diverse, come per i pittori fiamminghi che rendevano la luce
  nel vetro due secoli prima che qualcuno scrivesse la legge della rifrazione.
- Genie {cite}`bruce2024genie` è un videogioco senza motore di gioco:
  guardando trentamila ore di partite altrui, e senza che nessuno gli abbia
  mai detto quali tasti si premessero, si è inventato da solo un joystick a
  otto pulsanti. Le versioni successive (dicembre 2024 e agosto 2025) fanno lo
  stesso in tre dimensioni e in tempo reale, ma sono dimostrazioni scelte da
  chi le pubblica, non prodotti.
- Othello-GPT è la pagina da ricordare: una rete che ha solo «ascoltato»
  radiocronache di partite si è costruita in testa una scacchiera. Lo si
  dimostra in due mosse, e la seconda è quella che conta: prima un lettore del
  pensiero indovina dove sono le pedine guardando l'attività interna della rete
  (uno semplice sbaglia una casella su cinque, uno con un passaggio di calcolo
  in più meno di 2 su 100),
  poi il test del falso ricordo, in cui gli sperimentatori spostano una
  pedina *nella mente* della rete e le mosse cambiano di conseguenza. Non è un
  pappagallo: dentro c'è un piccolo mondo, e lo usa.
- Ma un'altra rete, allenata sui percorsi dei taxi di New York, dà indicazioni
  quasi sempre giuste e ha in testa una mappa piena di strade che non esistono.
  Un modello del mondo, dunque, può nascere da solo; quanto sia coerente e
  quanto regga fuori dai casi su cui si è allenato è la vera posta del
  dibattito, che resta aperto.
- Intanto queste cose lavorano: robot che pianificano immaginando, scenari
  rari generati per addestrare le auto a guida autonoma, ambienti illimitati
  in cui allenare programmi che imparano.
- Perché tutti scavano nei video: sono il grande giacimento non ancora
  spremuto in cui la correzione è gratis. Vuoi sapere se il modello ha
  previsto bene? Aspetta il fotogramma successivo.
- Il filo del capitolo, da Craik (1943) ai simulatori video: l'intelligenza
  come capacità di prevedere. Mancano ancora la capacità di ricombinare i
  pezzi in situazioni davvero nuove, quella di distinguere che cosa *provoca*
  che cosa, e quella di immaginare lontano. Un cantiere, non un verdetto.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Nel 2024 OpenAI presenta i modelli di generazione video come «simulatori di
  mondo» {cite}`brooks2024video`, documentando però essa stessa i limiti:
  fisica delle interazioni di base sbagliata (il bicchiere che si rovescia e
  trattiene il liquido anche coricato, senza rompersi). È un documento
  aziendale con dimostrazioni scelte, non un
  articolo passato da revisione. Imitare le apparenze non equivale a
  possedere il meccanismo.
- Genie {cite}`bruce2024genie` genera *ambienti interattivi* da 30.000 ore
  di video di platform senza etichette: 8 azioni latenti apprese da sole
  (un VQ-VAE con collo di bottiglia) più un modello di dinamica di tipo
  MaskGIT che vive sui token del tokenizer video, non sui pixel. GameNGen,
  all'opposto, simula un solo gioco con azioni osservate. Genie 2, del
  dicembre 2024, e Genie 3, dell'agosto 2025, estendono a mondi 3D in tempo
  reale: annunci via blog con demo selezionate, non ancora prodotti.
- Othello-GPT {cite}`li2023emergent`: un GPT addestrato solo su sequenze
  di mosse sviluppa una rappresentazione interna della scacchiera, leggibile
  con sonde (1,7% di errore con sonde non lineari, sui dati sintetici; 9,4%
  sulle partite di campionato) e *causalmente* efficace
  (interventi sulle attivazioni cambiano le mosse). La rappresentazione è poi
  risultata lineare nel sistema di riferimento «mia/dell'avversario»
  {cite}`nanda2023emergent`, il che ricorda quanto il probing dipenda dalle
  coordinate scelte.
- Qualche world model emerge dunque dalla sola predizione; quanto sia coerente
  e generale è la vera posta, e i taxi di Manhattan {cite}`vafa2024evaluating`
  mostrano il caso opposto: accuratezza locale altissima, mappa implicita
  incoerente, crollo appena si esce dalla distribuzione d'addestramento.
- Applicazioni già al lavoro: pianificazione robotica (V-JEPA 2), scenari
  rari per la guida autonoma, ambienti illimitati per addestrare agenti.
- I video sono il candidato naturale all'auto-supervisione su larga scala:
  dati sterminati, il bersaglio è il fotogramma successivo (o una sua
  rappresentazione, nella scelta JEPA). Quello che la predizione garantisce lo
  misurano Kang e colleghi: generalizzazione quasi perfetta dentro la
  distribuzione, migliorabile con la scala nelle ricombinazioni, assente
  fuori.
- Il filo del capitolo, da Craik (1943) ai simulatori video: l'intelligenza
  come capacità di prevedere. Mancano ancora composizionalità, causalità e
  pianificazione lunga: un cantiere, non un verdetto.
```

`````

Un modello del mondo può nascere da solo, dalla sola previsione, e restare
incoerente proprio dove nessuno lo ha mai messo alla prova. Accorgersene non è
questione di fargli altre domande facili, bisogna interrogarlo dove non è stato
addestrato. La mappa dei taxi lo mostra bene: quali strade siano collegate a
quali, la rete doveva ricostruirlo da sola, a forza di previsioni, e l'ha
sbagliato. Ci sono però dati in cui quei collegamenti arrivano già scritti
insieme ai dati stessi, come le stazioni di una metropolitana con le linee che
le uniscono o gli atomi di una molecola con i loro legami. Si chiamano grafi, e
il {doc}`capitolo sulle reti neurali su grafo </GraphNeuralNetwork/overview>`
parte da lì: in quei dati la struttura non va indovinata, è data, e dove non lo
è, come nei grafi di fatti, costruirla diventa il lavoro più grosso.
