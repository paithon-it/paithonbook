# Simulare il mondo: video generativi e un dibattito aperto

Il 15 febbraio 2024 OpenAI presenta Sora, un modello che da una descrizione
testuale genera video fotorealistici fino a un minuto. Il rapporto tecnico che
lo accompagna {cite}`brooks2024video` ha un titolo che è una tesi: *Video
generation models as world simulators*, i modelli di generazione video come
simulatori di mondo. La scommessa è dichiarata nell'ultima riga: continuare a
scalare i modelli video è «una strada promettente verso lo sviluppo di
simulatori capaci del mondo fisico e digitale». A sostegno, il rapporto elenca
capacità *emerse* senza essere state programmate: coerenza tridimensionale
delle scene, oggetti che continuano a esistere quando escono dall'inquadratura,
un pittore che lascia sulla tela pennellate che restano.

Ma lo stesso rapporto, poche righe più in basso, mostra i video in cui la
scommessa incespica: un bicchiere si rovescia sul tavolo e il liquido non si
versa come dovrebbe — il vetro non si infrange, il contenuto sfida la gravità.
Gli autori lo ammettono senza giri di parole: il modello «non simula
accuratamente la fisica di molte interazioni di base», e mangiare un biscotto
non sempre lascia il segno del morso. In quell'immagine c'è per intero la
domanda di quest'ultima sezione:
un video che *sembra* vero dimostra che il modello ha *capito* il mondo — o
soltanto che ha imparato a imitarne le apparenze? I pittori fiamminghi
rendevano alla perfezione la luce nei calici secoli prima delle leggi
dell'ottica: copiare bene un fenomeno e possederne il meccanismo sono due cose
diverse, e distinguere l'una dall'altra, in una rete neurale, è più difficile
che in un quadro.

## Mondi da guardare, mondi da giocare

Un video, per quanto perfetto, si guarda e basta: il futuro che mostra è già
deciso. Nel 2024 un gruppo di Google DeepMind compie il passo successivo con
**Genie** {cite}`bruce2024genie`, presentato a ICML 2024 e premiato tra i
migliori articoli della conferenza: non generare *video*, ma **ambienti
interattivi** — mondi in cui, a ogni fotogramma, è chi gioca a decidere la
mossa, e il modello risponde generando il fotogramma coerente con quella
mossa. Il materiale di partenza è sorprendentemente povero: circa 30.000 ore
di video di piattaforme in 2D — personaggi che corrono e saltano, alla Super
Mario — filtrate da un mucchio iniziale di 244.000 ore raccolte da internet.
Video e basta: nessuno ha detto al modello quali tasti premevano i giocatori.

`````{tab} Elementare

La differenza tra guardare un film e giocare a un videogioco è tutta qui: il
film va avanti da solo, il videogioco deve rispondere alle *tue* mosse. E per
rispondere a una mossa che nessuno ha previsto — premi «salta» proprio su
quell'orlo di burrone — il gioco deve sapere *che cosa succede dopo* in ogni
situazione possibile: deve avere, in qualche forma, un modello del suo piccolo
mondo. Genie è un videogioco senza motore di gioco: dietro non c'è un
programma con le regole scritte da qualcuno, c'è una rete che le regole le ha
assorbite guardando.

Il colpo di scena è che nessuno gli ha mai mostrato un joystick. Come fa a
sapere quali comandi esistono? I ricercatori gli impongono una regola severa:
per spiegare che cosa cambia tra un fotogramma e il successivo può usare
soltanto otto «mosse tipiche» — e quali siano le otto più utili deve
scoprirlo da solo, guardando migliaia di ore di partite. Funziona perché nei
platform i cambiamenti tipici sono davvero pochi: il personaggio si sposta a
destra, o a sinistra, o salta, o cade. Quelle otto mosse diventano i pulsanti
di un joystick che Genie si è inventato da solo. Quando giochi, scegli un
numero da 1 a 8; che il pulsante 3 significhi «salta» lo scopri provando,
come davanti a una console senza libretto di istruzioni.

`````

`````{tab} Superiore

Genie (circa 11 miliardi di parametri in totale) è composto da tre moduli: un
*tokenizer* video che comprime i fotogrammi in token discreti, un **modello di
azioni latenti** e un **modello di dinamica** autoregressivo — di gran lunga
il più grande dei tre — che predice i token del fotogramma successivo. Il
cuore concettuale è il secondo modulo, che affronta il problema dell'assenza
di etichette: i video di internet non dicono quale azione è stata premuta.
La soluzione è inferirla come variabile latente *discreta*:

$$
\tilde{a}_t = q\big(f_\phi(x_{1:t+1})\big) \in \{1, \dots, 8\},
\qquad
\hat{x}_{t+1} = g_\theta\big(x_{1:t},\, \tilde{a}_{1:t}\big),
$$

dove $x_{1:t+1}$ sono i fotogrammi osservati fino al tempo $t+1$, $f_\phi$ è
un encoder che riassume la transizione dal fotogramma $t$ al $t+1$, $q$ è una
quantizzazione vettoriale su un codebook di appena $|A| = 8$ codici — un
tetto fissato dagli autori perché il joystick resti maneggiabile per un
giocatore umano — e
$g_\theta$ è il modello che deve ricostruire il fotogramma successivo a
partire dal passato *e* dall'azione latente $\tilde{a}_t$. Il collo di
bottiglia è la chiave: potendo trasmettere a $g_\theta$ solo tre bit per
passo, $\tilde{a}_t$ è costretta a codificare il *cambiamento controllabile*
— la mossa — e a lasciare tutto il resto (sfondo, fisica, inerzia) al modello
di dinamica. In inferenza il modello di azioni sparisce e l'azione la
fornisce l'utente, fotogramma per fotogramma. Le azioni apprese risultano
semanticamente coerenti tra ambienti mai visti; addestrando lo stesso schema
su video di manipolazione robotica (il dataset RT-1), emergono allo stesso
modo azioni consistenti senza alcuna etichetta — indizio che la ricetta non è
legata ai platform.

`````

La discendenza di Genie è andata avanti a passo rapido, e conviene
raccontarla con le cautele del caso, perché la fonte sono gli annunci sul
blog di DeepMind, con demo selezionate, non articoli passati da revisione.
**Genie 2** (dicembre 2024) passa al 3D: da una singola immagine genera mondi
esplorabili con tastiera e mouse, con effetti di acqua, fumo e gravità,
coerenti fino a un minuto — la maggior parte degli esempi mostrati dura
10–20 secondi. **Genie 3** (agosto 2025) aggiunge il tempo reale: 24
fotogrammi al secondo a risoluzione 720p, mondi che restano coerenti per
alcuni minuti con una «memoria visiva» di circa un minuto, ed eventi
richiamabili con una frase («fa' piovere», «aggiungi un cane»). Gli stessi
autori elencano i limiti: interazioni di pochi minuti, spazio di azioni
ristretto, testo in scena spesso illeggibile, più agenti nello stesso mondo
ancora problematici. Al lancio Genie 3 era un'anteprima di ricerca per una
piccola cerchia di collaudatori: una demo notevole, non un prodotto — e la
distanza tra le due cose, in questo campo, va sempre tenuta a mente.

## La scacchiera nella macchina: gli LLM hanno un world model?

Genie *deve* avere un modello del suo mondo, per costruzione: è la sua unica
funzione. La domanda si fa più spinosa quando la voltiamo verso i modelli di
linguaggio {cite}`brown2020language`, addestrati solo a indovinare il token
successivo. L'apertura del capitolo ha lasciato in sospeso un esperimento,
citandolo di sfuggita: è il momento di raccontarlo, perché è il tentativo più
pulito di rispondere con i dati anziché con gli slogan.

Nel 2023 Kenneth Li e colleghi — tra gli altri David Bau, Fernanda Viégas e
Martin Wattenberg, nomi noti dell'interpretabilità delle reti — pubblicano a
ICLR l'esperimento oggi noto come **Othello-GPT** {cite}`li2023emergent`. Prendono
un piccolo GPT — 8 strati, la stessa architettura dei modelli di linguaggio —
e lo addestrano su un solo tipo di testo: sequenze di mosse del gioco
dell'Otello, scritte come liste di caselle («E3, D2, …»), per venti milioni di
partite sintetiche. Il modello non ha mai visto una scacchiera, non conosce le
regole, non sa che esistono pedine bianche e nere: per lui le mosse sono
token, come parole. Eppure, a fine addestramento, propone mosse *legali* con
un tasso d'errore dello 0,01%. La domanda, a quel punto, è una sola: ci
riesce accumulando statistiche di superficie sulle sequenze — dopo «E3, D2»
viene spesso «C4» — o si è costruito dentro, da qualche parte, una
scacchiera?

`````{tab} Elementare

Immagina una persona che non ha mai visto una scacchiera in vita sua e ha
solo *ascoltato* migliaia di radiocronache di partite: sequenze di nomi di
caselle, nient'altro. Dopo anni di ascolto, sa proseguire una radiocronaca
con mosse che non fanno mai arrabbiare gli arbitri. Ha in testa una
scacchiera immaginata, o solo un enorme orecchio per le frasi tipiche?

Con una persona non potremmo saperlo. Con una rete sì, perché possiamo
guardarle dentro, ed è quel che fanno gli autori, in due passi. Primo passo:
addestrano un piccolo «lettore del pensiero» — una seconda rete, molto
semplice — che guardando solo l'attività interna della prima deve indovinare
dove sono le pedine. Ci riesce: sbaglia meno di 2 caselle su 100. Quindi
l'informazione «com'è messa la scacchiera» *dentro la rete c'è*, anche se
nessuno gliel'ha mai chiesta. Secondo passo, il più bello: il test del falso
ricordo. Gli sperimentatori entrano nell'attività interna e la ritoccano,
spostando una pedina *nella mente* della rete — non nella sequenza di mosse,
che resta identica. Se la scacchiera interna fosse un ornamento inutile, le
mosse proposte non cambierebbero. Invece cambiano, e in modo coerente con la
scacchiera contraffatta: la rete gioca in base a ciò che «crede» di vedere.
Non è un pappagallo di sequenze: dentro c'è un piccolo mondo, e lo usa.

`````

`````{tab} Superiore

Lo strumento è il **probing**: una sonda $p_\psi$ — un classificatore
addestrato a parte — riceve le attivazioni $h_t^{(\ell)}$ dello strato $\ell$
al passo $t$ e deve predire lo stato di ciascuna delle 64 caselle (vuota,
nera, bianca). Le sonde *lineari* falliscono (errore del 20–23%, appena
meglio del 27–29% che si ottiene sondando una rete con pesi casuali), quelle
*non lineari* (un MLP a uno strato nascosto) arrivano
all'1,7% di errore negli strati più profondi: lo stato della partita è ricostruibile
quasi per intero dalle attivazioni. Poiché una sonda potrebbe leggere una
correlazione senza ruolo causale, il passo decisivo è l'**intervento**: si
modificano le attivazioni con una discesa di gradiente finché la sonda vi
legge una scacchiera contraffatta, si lascia proseguire il calcolo e si
osserva che la distribuzione sulle mosse legali si adegua alla scacchiera
modificata, non alla sequenza di input. La rappresentazione, dunque, *guida*
la predizione.

Due poscritti metodologici. Neel Nanda e collaboratori (2023) hanno mostrato
che la rappresentazione è in realtà *lineare*, purché la si cerchi nel
sistema di riferimento giusto: non «nero/bianco» ma «mia/dell'avversario»,
relativo a chi muove — un monito su quanto le conclusioni del probing
dipendano dalle coordinate scelte. E Keyon Vafa e colleghi (NeurIPS 2024)
hanno raffreddato gli entusiasmi sul fronte opposto: un transformer
addestrato sui percorsi dei taxi di Manhattan dà indicazioni svolta-per-svolta
valide nel 99% dei casi, ma la mappa implicita che si può ricostruire dai
suoi output contiene strade inesistenti e cavalcavia impossibili, e basta
imporre qualche deviazione perché le prestazioni crollino. Un modello può
essere localmente accurato e globalmente incoerente: il world model c'è, ma
può essere una mappa sbrindellata.

`````

Come si tengono insieme questi risultati? Il dibattito ha due letture, ed è
onesto dire che nessuna delle due ha vinto. La prima: Othello-GPT dimostra
che la pura predizione del token successivo *può* far emergere una
rappresentazione interna dello stato del mondo — la caricatura del modello
che «rimescola frasi senza rappresentarsi nulla» è, almeno in questo caso
controllato, falsificata. La seconda: un mondo di 64 caselle con regole
fisse è lontanissimo dal mondo fisico, e i risultati alla Vafa suggeriscono
che le rappresentazioni emerse per questa via possono essere frammentarie,
fragili fuori distribuzione, buone per il compito d'addestramento e poco
altro. È l'eco di una prudenza già incontrata nel capitolo sui Transformer, a
proposito di allucinazioni e comprensione: su che cosa i modelli capiscano
davvero, il dibattito scientifico è tutt'altro che chiuso. Chi, come LeCun,
ritiene che serva un'architettura pensata apposta per prevedere il mondo — la
strada JEPA delle sezioni precedenti — e chi scommette che scala e dati
{cite}`kaplan2020scaling` faranno maturare i modelli del mondo dentro i
modelli di linguaggio, oggi si divide esattamente su questo punto. Il lettore
arrivato fin qui ha gli strumenti per seguire la partita senza tifare.

## Applicazioni con i piedi per terra

Mentre il dibattito continua, i world model lavorano. In **robotica** la
strada l'abbiamo già vista nella sezione precedente: V-JEPA 2, nella variante
condizionata sulle azioni, usa le previsioni nello spazio delle
rappresentazioni per *pianificare* — provare mentalmente le sequenze di
comandi e scegliere quella che avvicina il braccio all'obiettivo — su robot
mai visti in addestramento. Nella **guida autonoma** il problema sono gli
scenari rari: il bambino che sbuca tra due auto, il carico che cade dal
camion. Raccoglierli su strada è impraticabile, oltre che inaccettabile; un
world model generativo li produce in quantità e in sicurezza — la scommessa
di sistemi come GAIA-1 di Wayve (2023). Nei **videogiochi e negli ambienti
di addestramento**, infine, il cerchio si chiude: DeepMind presenta Genie 2
esplicitamente come generatore di ambienti illimitati in cui addestrare e
valutare agenti — il rimedio a un vizio storico del reinforcement learning,
dove gli agenti imparano a memoria i pochi ambienti disponibili invece di
imparare ad adattarsi.

E c'è una ragione strutturale per cui questo fronte è considerato tra i più
caldi della ricerca, al di là delle demo spettacolari.

`````{tab} Elementare

Per tutto il libro il collo di bottiglia è stato lo stesso: i dati con le
etichette costano. Qualcuno deve scrivere «gatto» sotto la foto del gatto,
tradurre la frase, assegnare il voto. Il video no: è il primo giacimento in
cui la correzione è *gratis*. Vuoi sapere se il modello ha previsto bene?
Aspetta il fotogramma successivo: la risposta esatta arriva da sola, milioni
di volte per ogni ora di filmato. E ogni video è un piccolo esperimento di
fisica già eseguito — bicchieri che cadono, palle che rimbalzano, porte che
sbattono — registrato senza che nessuno lo abbia allestito. Con il testo di
qualità che comincia a scarseggiare, come si è visto nel capitolo sui
Transformer, il video è la più grande riserva di esperienza del mondo non
ancora spremuta: ecco perché tutti scavano qui.

`````

`````{tab} Superiore

È la stessa logica auto-supervisionata che ha alimentato gli LLM — il
bersaglio dell'addestramento è il dato stesso, spostato nel tempo — applicata
a un serbatoio più grande di ordini di grandezza: il target è $x_{t+1}$ (o
una sua rappresentazione $z_{t+1}$, nella scelta JEPA), la loss una
verosimiglianza o una distanza predittiva, l'annotatore nessuno. Se la
lezione delle leggi di scala {cite}`kaplan2020scaling` è che le prestazioni
crescono con dati e calcolo secondo regolarità prevedibili, i video sono il
posto naturale dove proseguire la curva quando il testo si esaurisce. Le
incognite però sono due, e questo capitolo le ha incontrate entrambe: *dove*
predire (lo spazio dei pixel obbliga a modellare dettagli irrilevanti; lo
spazio latente rischia il collasso e va regolarizzato) e *che cosa* la
predizione garantisce — perché prevedere bene i fotogrammi tipici, come
mostrano gli errori di Sora, non equivale ad aver interiorizzato le leggi che
li generano.

`````

## Il filo del capitolo

Riavvolgiamo. Kenneth Craik, 1943: un organismo con un «modello in scala
ridotta» della realtà può provare le alternative nella testa e reagire al
futuro prima che arrivi. Ha e Schmidhuber, 2018: un agente si allena dentro
il proprio sogno e torna nel gioco vero più bravo di prima. Hopfield, 1982, e
la tradizione delle energie: ricordare e giudicare come discesa verso le
configurazioni compatibili — la lingua in cui LeCun ha scritto la sua
proposta. Le JEPA: prevedere sì, ma nello spazio delle rappresentazioni,
lasciando cadere i dettagli che non contano. E infine Sora e Genie: la
previsione fatta spettacolo, fotogrammi interi di futuro. Il filo che
attraversa ottant'anni è uno solo, e conviene dirlo in chiaro: in questa
tradizione di ricerca **l'intelligenza è la capacità di prevedere** — e di
usare le previsioni per agire.

Che cosa manca, lo si può dire con la stessa calma. Manca la
**composizionalità**: i simulatori attuali interpolano le scene viste, ma
ricombinare pezzi noti in situazioni radicalmente nuove — il forte delle
menti biologiche — resta fragile. Manca la **causalità**: prevedere ciò che
*segue* non è capire ciò che *provoca*, e la differenza tra osservare una
correlazione e intervenire su un meccanismo, che un bambino esplora
rovesciando bicchieri apposta, nei modelli è ancora sottile. E manca la
**pianificazione a lungo orizzonte**: l'errore dei modelli si accumula passo
dopo passo — lo abbiamo chiamato *model bias* all'inizio del capitolo — e i
sogni utili, per ora, durano minuti, non giornate. Nessuna di queste lacune
autorizza il catastrofismo («è tutto un trucco») né il trionfalismo («è fatta,
questione di mesi»): autorizzano un cantiere. Se c'è una lezione nelle date
di questo capitolo — un'idea del 1943 diventata un agente funzionante nel
2018, una rete del 1982 premiata con il Nobel nel 2024 — è che le idee giuste
maturano su tempi lunghi, e che saperle riconoscere prima degli altri è
esattamente ciò per cui vale la pena studiarle.

```{admonition} Da ricordare
:class: important
- Nel 2024 OpenAI presenta i modelli di generazione video come «simulatori di
  mondo» {cite}`brooks2024video`, documentando però essa stessa i limiti:
  fisica delle interazioni di base sbagliata (il bicchiere che si rovescia
  senza versare). Imitare le apparenze non equivale a possedere il meccanismo.
- **Genie** {cite}`bruce2024genie` genera *ambienti interattivi* da 30.000 ore
  di video di platform senza etichette: 8 azioni latenti apprese da sole
  (quantizzazione con collo di bottiglia) più un modello di dinamica
  condizionato sulle azioni. Genie 2 e Genie 3 estendono a mondi 3D in tempo
  reale — annunci via blog con demo selezionate, non ancora prodotti.
- **Othello-GPT** {cite}`li2023emergent`: un GPT addestrato solo su sequenze
  di mosse sviluppa una rappresentazione interna della scacchiera, leggibile
  con sonde (1,7% di errore) e *causalmente* efficace (interventi sulle
  attivazioni cambiano le mosse). Qualche world model emerge dalla sola
  predizione; quanto sia coerente e generale (i taxi di Manhattan di Vafa) è
  la vera posta del dibattito — che resta aperto.
- Applicazioni già al lavoro: pianificazione robotica (V-JEPA 2), scenari
  rari per la guida autonoma, ambienti illimitati per addestrare agenti.
- Il fronte è caldo perché i video sono dati sterminati con supervisione
  gratuita: il bersaglio è il fotogramma successivo.
- Il filo del capitolo, da Craik (1943) ai simulatori video: l'intelligenza
  come capacità di prevedere. Mancano ancora composizionalità, causalità e
  pianificazione lunga: un cantiere, non un verdetto.
```
