# Simulare il mondo: video generativi e un dibattito aperto

Il 15 febbraio 2024 OpenAI presenta Sora, un modello che da una descrizione
testuale genera video fotorealistici fino a un minuto. Come sia fatto dentro
l'ha già raccontato il {doc}`capitolo sui Modelli di Diffusione </ModelliDiffusione/overview>`; qui interessa che
cosa dimostra. Il rapporto che lo accompagna {cite}`brooks2024video` ha un
titolo che è una tesi: *Video generation models as world simulators*, i modelli
di generazione video come simulatori di mondo. È un documento aziendale con dimostrazioni scelte da chi le pubblica, non un
articolo passato da revisione. La scommessa è dichiarata nell'ultima riga: continuare a scalare i
modelli video è «una strada promettente verso lo sviluppo di simulatori capaci
del mondo fisico e digitale». A sostegno, il rapporto elenca capacità che
dichiara *emerse* senza essere state programmate: coerenza tridimensionale
delle scene, oggetti che continuano a esistere quando escono dall'inquadratura,
un pittore che lascia sulla tela pennellate che restano.

Ma lo stesso rapporto, poche righe più in basso, mostra i video in cui la
scommessa incespica: un bicchiere si rovescia sul tavolo e il liquido non si
versa come dovrebbe; il vetro non si infrange, il contenuto sfida la gravità.
Gli autori lo ammettono senza giri di parole: il modello «non simula
accuratamente la fisica di molte interazioni di base», e mangiare un biscotto
non sempre lascia il segno del morso. In quell'immagine c'è per intero la
domanda di quest'ultima sezione: un video che *sembra* vero dimostra che il
modello ha *capito* il mondo, o soltanto che ha imparato a imitarne le
apparenze? I pittori fiamminghi rendevano alla perfezione la luce nei calici
secoli prima delle leggi dell'ottica: copiare bene un fenomeno e possederne il
meccanismo sono due cose diverse, e distinguere l'una dall'altra, in una rete
neurale, è più difficile che in un quadro.

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
ingovernabile. E funziona perché in quei giochi i cambiamenti tipici sono
davvero pochi: il personaggio si sposta a destra, o a sinistra, o salta, o
cade. Quelle otto mosse diventano i pulsanti di un
joystick che Genie si è inventato da solo. Quando giochi, scegli un numero da
1 a 8: che il pulsante 3 significhi «salta» lo scopri provando, come davanti a
una console senza libretto di istruzioni.

`````

`````{tab} Superiore

Genie (circa 11 miliardi di parametri in totale) è composto da tre moduli: un
*tokenizer* video che comprime i fotogrammi in token discreti, un **modello di
azioni latenti** e un **modello di dinamica** autoregressivo (di gran lunga il
più grande dei tre) che predice i token del fotogramma successivo. Il cuore
concettuale è il secondo modulo, che affronta il problema dell'assenza di
etichette: i video di internet non dicono quale azione è stata premuta. La
soluzione è inferirla come variabile latente *discreta*:

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
mestiere del $\mathbf{z}$ del VAE nella prima sezione, ma con un vocabolario finito
invece di 32 numeri continui); $f_\phi$ è un encoder che riassume la
transizione dal fotogramma $t$ al $t+1$; $q$ è una quantizzazione vettoriale
su un codebook di appena $|\mathcal{A}| = 8$ codici (un tetto fissato dagli autori perché il joystick resti
maneggiabile per un giocatore umano); e $g_\theta$ è il modello di dinamica,
che predice i **token** del fotogramma successivo a partire dai token del
passato *e* dall'azione latente $\tilde{a}_t$. A riportare i token in pixel è
il decoder del tokenizer: il modello di dinamica non vede mai un pixel e non ne
produce mai uno. Il collo di
bottiglia è la chiave: potendo trasmettere a $g_\theta$ solo tre bit per
passo, $\tilde{a}_t$ è costretta a codificare il *cambiamento controllabile*
(la mossa) e a lasciare tutto il resto (sfondo, fisica, inerzia) al modello di
dinamica. In inferenza il modello di azioni sparisce e l'azione la fornisce
l'utente, fotogramma per fotogramma. Le azioni apprese risultano
semanticamente coerenti tra ambienti mai visti; addestrando lo stesso schema
su video di manipolazione robotica (il dataset RT-1), emergono allo stesso
modo azioni consistenti senza alcuna etichetta: indizio che la ricetta non è
legata ai platform.

`````

La discendenza di Genie è andata avanti a passo rapido. Qui la raccontiamo solo
per quello che ha di strutturale: il resto invecchia in fretta, e vale il
cartello appeso poco fa a Sora, perché la fonte sono annunci sul blog di
DeepMind, non articoli passati da revisione. Due passi contano. Il primo,
**Genie 2** (dicembre 2024), esce dal piatto del disegno a due dimensioni: da una singola
immagine genera mondi a tre dimensioni, esplorabili con tastiera e mouse, con
acqua, fumo e gravità. Il secondo, **Genie 3** (agosto 2025), passa dal
differito al **tempo reale**: il mondo si genera mentre lo si attraversa, non
dopo, e si possono richiamare eventi con una frase («fa’ piovere», «aggiungi un
cane»).

I limiti li elencano gli autori stessi, e contano più delle immagini
spettacolari. Le
sessioni si misurano in minuti. Il repertorio di comandi è ristretto: quel che
scarseggia sono le azioni possibili, non i posti dove andare. Il testo che
compare in scena è spesso illeggibile. E mettere più agenti autonomi nello
stesso mondo resta problematico. Anteprime di ricerca, insomma: dimostrazioni
notevoli, non prodotti, e la distanza tra le due cose, in questo campo, va
sempre tenuta a mente.

## La scacchiera nella macchina: gli LLM hanno un world model?

Genie *deve* avere un modello del suo mondo, per costruzione: è la sua unica
funzione. La domanda si fa più spinosa quando la voltiamo verso i modelli di
linguaggio, i grandi modelli di linguaggio dell'apertura, gli LLM
{cite}`brown2020language`, addestrati a fare
una cosa sola: indovinare la parola che viene dopo (per la precisione il
**token** successivo, cioè il pezzetto di parola con cui questi modelli
lavorano). L'apertura del capitolo ha lasciato in sospeso un esperimento,
citandolo di sfuggita: è il momento di raccontarlo, perché è il tentativo più
pulito di rispondere con i dati anziché con gli slogan.

Nel 2023 Kenneth Li e colleghi (tra gli altri David Bau, Fernanda Viégas e
Martin Wattenberg, nomi noti dell'interpretabilità delle reti) pubblicano a
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

Immagina una persona che non ha mai visto una scacchiera in vita sua e ha
solo *ascoltato* migliaia di radiocronache di partite: sequenze di nomi di
caselle, nient'altro. Dopo anni di ascolto, sa proseguire una radiocronaca
con mosse che non fanno mai arrabbiare gli arbitri. Ha in testa una
scacchiera immaginata, o solo un enorme orecchio per le frasi tipiche?

Con una persona non potremmo saperlo. Con una rete sì, perché possiamo
guardarle dentro. Che cosa vuol dire, guardarci dentro? Che mentre la rete
elabora una partita, ogni suo strato produce una fila di numeri, e quei numeri
si possono leggere uno per uno: è l'attività interna, ed è la sola cosa che la
rete abbia in testa. Gli autori la usano in due passi. Primo passo: addestrano
un piccolo «lettore del pensiero» (una seconda rete, molto semplice) che
guardando soltanto quei numeri deve indovinare dove sono le pedine. Ci riesce:
sbaglia meno di 2 caselle su 100. Quindi l'informazione «com'è messa la
scacchiera» *dentro la rete c'è*, anche se nessuno gliel'ha mai chiesta.
Secondo passo, il più bello: il test del falso ricordo. Gli sperimentatori
entrano in quei numeri e li ritoccano, spostando una pedina *nella mente* della
rete: non nella sequenza di mosse, che resta identica. Se la scacchiera interna fosse un ornamento inutile, le
mosse proposte non cambierebbero. Invece cambiano, e in modo coerente con la
scacchiera contraffatta: la rete gioca in base a ciò che «crede» di vedere.
Non è un pappagallo di sequenze: dentro c'è un piccolo mondo, e lo usa.

`````

`````{tab} Superiore

Lo strumento è il **probing**: una sonda $p_\psi$ (un classificatore
addestrato a parte) riceve le attivazioni $\mathbf{h}_t^{(\ell)}$ dello strato $\ell$
al passo $t$ e deve predire lo stato di ciascuna delle 64 caselle (vuota,
nera, bianca). Le sonde *lineari* falliscono (errore fra il 20% e il 24% a
seconda dello strato, appena meglio del 26–29% che si ottiene sondando una rete
con pesi casuali), quelle *non lineari* (un MLP a uno strato nascosto) arrivano
all'1,7% di errore negli strati più profondi: lo stato della partita è
ricostruibile quasi per intero dalle attivazioni. Poiché una sonda potrebbe leggere una correlazione senza
ruolo causale, il passo decisivo è l’**intervento**: si modificano le
attivazioni con una discesa di gradiente finché la sonda vi legge una
scacchiera contraffatta, si lascia proseguire il calcolo e si osserva che la
distribuzione sulle mosse legali si adegua alla scacchiera modificata, non
alla sequenza di input. La rappresentazione, dunque, *guida* la predizione.

Un poscritto metodologico, prima di tirare le somme. Neel Nanda e collaboratori
(2023) {cite}`nanda2023emergent` hanno mostrato che la rappresentazione è in
realtà *lineare*, purché la si cerchi nel sistema di riferimento giusto: non
«nero/bianco» ma «mia/dell'avversario», relativo a chi muove. L'1,7% delle
sonde non lineari, quindi, non diceva che l'informazione fosse codificata in
modo intricato: quelle sonde stavano compensando una scelta di coordinate. È
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

È l'eco di una prudenza già incontrata nel {doc}`capitolo sui Transformer </Transformers/overview>`, a
proposito di allucinazioni e comprensione: su che cosa i modelli capiscano
davvero, il dibattito scientifico è tutt'altro che chiuso. Da una parte c'è
chi, come LeCun, ritiene che serva un'architettura pensata apposta per
prevedere il mondo, ed è la strada JEPA delle sezioni precedenti. Dall'altra
c'è chi scommette che basteranno la taglia dei modelli e la quantità di dati
{cite}`kaplan2020scaling` a far maturare un modello del mondo dentro i modelli
di linguaggio. Il lettore arrivato fin qui ha gli strumenti per seguire la
partita senza tifare.

## Applicazioni con i piedi per terra

Mentre il dibattito continua, i world model lavorano. In **robotica** la
strada l'abbiamo già vista nella sezione precedente: V-JEPA 2, nella variante
condizionata sulle azioni, usa le previsioni nello spazio delle
rappresentazioni per *pianificare* (provare mentalmente i comandi possibili,
uno alla volta, e scegliere quello che avvicina il braccio all'obiettivo) su
robot mai visti in addestramento. Nella **guida autonoma** il problema sono gli
scenari rari: il bambino che sbuca tra due auto, il carico che cade dal camion. Raccoglierli
su strada è impraticabile, oltre che inaccettabile; un world model generativo
li produce in quantità e in sicurezza, ed è dal 2023 la scommessa di più di un
laboratorio del settore (GAIA-1 di Wayve è stato fra i primi a mostrarla in
pubblico).
Nei **videogiochi e negli ambienti di addestramento**, infine,
il cerchio si chiude: DeepMind presenta Genie 2 esplicitamente come generatore
di ambienti illimitati in cui addestrare e valutare agenti; il rimedio a un
vizio storico dell'apprendimento per rinforzo, dove i programmi che imparano
per tentativi finiscono per sapere a memoria i pochi ambienti disponibili
invece di imparare ad adattarsi.

E c'è una ragione strutturale per cui questo fronte è considerato tra i più
caldi della ricerca, al di là delle demo spettacolari.

`````{tab} Elementare

Per tutto il libro il collo di bottiglia è stato lo stesso: i dati con le
etichette costano. Qualcuno deve scrivere «gatto» sotto la foto del gatto,
tradurre la frase, assegnare il voto. Il video no: è un giacimento sterminato in
cui la correzione è *gratis*. Vuoi sapere se il modello ha previsto bene?
Aspetta il fotogramma successivo: la risposta esatta arriva da sola, milioni
di volte per ogni ora di filmato. Per giunta ogni video è un piccolo esperimento di
fisica già eseguito (bicchieri che cadono, palle che rimbalzano, porte che
sbattono) registrato senza che nessuno lo abbia allestito. E il testo, invece,
non è infinito. Il capitolo sui Transformer lo dice in due punti: più un modello è
grande, più testo pretende per essere addestrato come si deve; e il web sta
finendo come fonte gratuita di scrittura di qualità. Le pagine scritte dagli
esseri umani restano quelle che sono. Il video è
la più grande riserva di esperienza del mondo non ancora spremuta: ecco perché
tutti scavano qui.

`````

`````{tab} Superiore

È la stessa logica auto-supervisionata che ha alimentato gli LLM (il bersaglio
dell'addestramento è il dato stesso, spostato nel tempo) applicata a un
serbatoio più grande di ordini di grandezza: il target è $\mathbf{x}_{t+1}$ (o una sua
rappresentazione $\mathbf{z}_{t+1}$, nella scelta JEPA), la loss una verosimiglianza o
una distanza predittiva, l'annotatore nessuno. Se la lezione delle leggi di
scala {cite}`kaplan2020scaling` è che le prestazioni crescono con dati e
calcolo secondo regolarità prevedibili, i video sono il posto naturale dove
proseguire la curva quando il testo si esaurisce. Le incognite però sono due,
e questo capitolo le ha incontrate entrambe: *dove* predire (lo spazio dei
pixel obbliga a modellare dettagli irrilevanti; lo spazio latente rischia il
collasso e va regolarizzato) e *che cosa* la predizione garantisce; perché
prevedere bene i fotogrammi tipici, come mostrano gli errori di Sora, non
equivale ad aver interiorizzato le leggi che li generano.

`````

## Il filo del capitolo

Riavvolgiamo. Kenneth Craik, 1943: un organismo con un «modello in scala
ridotta» della realtà può provare le alternative nella testa e reagire al
futuro prima che arrivi. Ha e Schmidhuber, 2018: un agente si allena dentro il
proprio sogno e torna nel gioco vero più bravo di prima. Hopfield, 1982, e la
tradizione delle energie: dare un voto a ogni combinazione possibile, e poi
lasciare che il sistema scivoli verso quelle che il voto dice compatibili. È
insieme il modo di ricordare e quello di giudicare, ed è la lingua in cui LeCun
ha scritto la sua proposta.
Le JEPA: prevedere sì, ma nello spazio delle rappresentazioni, lasciando
cadere i dettagli che non contano. E infine Sora e Genie: la previsione fatta
spettacolo, fotogrammi interi di futuro. Il filo che attraversa ottant'anni è
uno solo, e conviene dirlo in chiaro: in questa tradizione di ricerca
**l'intelligenza è la capacità di prevedere**, e di usare le previsioni per
agire.

Che cosa manca, lo si può dire con la stessa calma. Manca la
**composizionalità**: i simulatori attuali sanno muoversi *fra* le scene che
hanno visto, mescolandole e sfumando dall'una all'altra (in gergo si dice che
le **interpolano**), ma ricombinare pezzi noti in situazioni radicalmente
nuove, che è il forte delle menti biologiche, resta fragile. Manca la
**causalità**: prevedere ciò che *segue* non è capire ciò che *provoca*. Un
bambino la differenza la esplora da sé, rovesciando bicchieri apposta: vedere
due cose che vanno sempre insieme è un conto, andarne a toccare una per vedere
che ne è dell'altra è un altro. Nei modelli quella differenza è ancora poco
marcata. E manca la **pianificazione a lungo orizzonte**: l'errore dei modelli
si accumula passo dopo passo, ed è il difetto che il capitolo ha incontrato
per primo, quando l'agente si allenava dentro una copia imprecisa del gioco
(in gergo si chiama *model bias*, la piega sistematica del modello). I sogni
dentro cui si può ancora pianificare sono **corti**: una quindicina di passi
immaginati per i Dreamer, gli eredi del sogno di Ha e Schmidhuber, e un passo
solo per il world model che guida il braccio robotico. I minuti di cui si
parla per i simulatori generativi sono un'altra cosa: sono la lunghezza di un
video da *guardare*, non di un piano da eseguire. Nessuna di queste lacune
autorizza il catastrofismo («è tutto un trucco») né il trionfalismo («è fatta,
questione di mesi»): autorizzano un cantiere. Se c'è una lezione nelle date di
questo capitolo (un'idea del 1943 diventata un agente funzionante nel 2018,
una rete del 1982 premiata con il Nobel nel 2024) è che le idee giuste
maturano su tempi lunghi, e che saperle riconoscere prima degli altri è
esattamente ciò per cui conviene studiarle.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Nel 2024 OpenAI presenta i propri generatori di video come «simulatori di
  mondo» {cite}`brooks2024video`, ma è il documento stesso a mostrare dove il
  trucco si vede: il bicchiere che si rovescia e non versa. Copiare bene un
  fenomeno e possederne il meccanismo restano due cose diverse, come per i
  pittori fiamminghi che rendevano la luce nei calici senza sapere niente di
  ottica.
- **Genie** {cite}`bruce2024genie` è un videogioco senza motore di gioco:
  guardando trentamila ore di partite altrui, e senza che nessuno gli abbia
  mai detto quali tasti si premessero, si è inventato da solo un joystick a
  otto pulsanti. Le versioni successive fanno lo stesso in tre dimensioni e in
  tempo reale, ma sono dimostrazioni scelte da chi le pubblica, non prodotti.
- **Othello-GPT** è la pagina da ricordare: una rete che ha solo «ascoltato»
  radiocronache di partite si è costruita in testa una scacchiera. Lo si
  dimostra in due mosse, e la seconda è quella che conta: prima un lettore del
  pensiero indovina dove sono le pedine guardando l'attività interna della
  rete (sbaglia meno di 2 caselle su 100), poi il test del **falso ricordo**,
  in cui gli sperimentatori spostano una pedina *nella mente* della rete e le
  mosse cambiano di conseguenza. Non è un pappagallo: dentro c'è un piccolo
  mondo, e lo usa.
- Ma un'altra rete, allenata sui percorsi dei taxi di New York, dà indicazioni
  quasi sempre giuste e ha in testa una mappa piena di strade che non esistono.
  Un modello del mondo, dunque, può nascere da solo; quanto sia coerente e
  quanto regga fuori dai casi su cui si è allenato è la vera posta del
  dibattito, che resta aperto.
- Intanto queste cose lavorano: robot che pianificano immaginando, scenari
  rari generati per addestrare le auto a guida autonoma, ambienti illimitati
  in cui allenare programmi che imparano.
- Perché il fronte è così caldo: il video è l'unico grande giacimento di dati
  in cui la correzione è gratis. Vuoi sapere se il modello ha previsto bene?
  Aspetta il fotogramma successivo.
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
  fisica delle interazioni di base sbagliata (il bicchiere che si rovescia
  senza versare). È un documento aziendale con dimostrazioni scelte, non un
  articolo passato da revisione. Imitare le apparenze non equivale a
  possedere il meccanismo.
- **Genie** {cite}`bruce2024genie` genera *ambienti interattivi* da 30.000 ore
  di video di platform senza etichette: 8 azioni latenti apprese da sole
  (quantizzazione con collo di bottiglia) più un modello di dinamica
  autoregressivo che vive sui **token** del tokenizer video, non sui pixel.
  Genie 2 e Genie 3 estendono a mondi 3D in tempo reale: annunci via blog con
  demo selezionate, non ancora prodotti.
- **Othello-GPT** {cite}`li2023emergent`: un GPT addestrato solo su sequenze
  di mosse sviluppa una rappresentazione interna della scacchiera, leggibile
  con sonde (1,7% di errore con sonde non lineari) e *causalmente* efficace
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
- Il fronte è caldo perché i video sono dati sterminati con supervisione
  gratuita: il bersaglio è il fotogramma successivo (o una sua
  rappresentazione, nella scelta JEPA).
- Il filo del capitolo, da Craik (1943) ai simulatori video: l'intelligenza
  come capacità di prevedere. Mancano ancora composizionalità, causalità e
  pianificazione lunga: un cantiere, non un verdetto.
```

`````

Un modello del mondo può nascere da solo, dalla sola previsione, e restare
incoerente proprio dove nessuno lo ha mai messo alla prova. Accorgersene non è
questione di fargli altre domande facili, bisogna interrogarlo dove non è stato
addestrato. Con il capitolo sulle reti neurali su grafo si cambia aria, perché
lì la struttura del mondo non va indovinata dai dati, arriva già scritta
insieme a loro.
