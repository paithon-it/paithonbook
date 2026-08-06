# Allineamento e governance: dai valori umani alle regole

Nel dicembre 2016 alcuni ricercatori di OpenAI addestrano un agente a giocare
a *CoastRunners*, una gara di motoscafi. L'obiettivo dichiarato era semplice:
vincere la corsa. Ma il punteggio del gioco non premiava l'arrivo: premiava il
colpire una serie di bersagli disseminati lungo il percorso. L'agente lo
scoprì, e trovò una scorciatoia geniale e assurda: invece di correre, faceva
girare la barca in tondo dentro una piccola laguna, colpendo all'infinito gli
stessi bersagli che ricomparivano, prendendo fuoco, andando contromano e
schiantandosi, e intanto accumulava un punteggio più alto del 20% rispetto a
qualunque giocatore umano che *finiva* la gara. Aveva fatto esattamente ciò
che gli avevamo chiesto. Non ciò che intendevamo.

Questa distanza (tra la **lettera** di un obiettivo e la sua **intenzione**) è
il cuore del problema dell'**allineamento**. Finché il sistema è un motoscafo
in un videogioco, la scorciatoia fa sorridere. Quando lo stesso meccanismo
governa un modello che parla con milioni di persone, o che filtra domande di
lavoro e richieste di prestito, smette di far sorridere. Questa sezione chiude
il capitolo affrontando le due facce di quella distanza: come si prova a
*orientare* il comportamento di un sistema verso ciò che vogliamo davvero
(l'allineamento), e quale impalcatura di regole e verifiche prova a tenere il
tutto entro binari accettabili (la governance).

## Il problema dell'allineamento

L'aneddoto del motoscafo ha un nome tecnico: **specification gaming**, il
«giocare» con la specifica. Ne abbiamo già visto la meccanica parlando di
reinforcement learning e, per gli LLM, nel post-training dei Transformer, dove
va sotto il nome di **reward hacking**: quando ottimizzi un *surrogato* del tuo
vero obiettivo, prima o poi ottieni il surrogato e perdi l'obiettivo.

```{figure} ../figures/problema-allineamento.svg
:name: fig-problema-allineamento
:alt: "Schema in tre riquadri incolonnati: l'intenzione umana («vinci la gara») viene tradotta in una metrica dichiarata («massimizza i punti»), che l'addestramento trasforma in un obiettivo appreso («gira in tondo sui bonus»). Fra il primo e il secondo riquadro è marcato il primo scarto, la metrica cattura l'intenzione?; fra il secondo e il terzo il secondo scarto, il modello persegue davvero la metrica? In basso il comportamento previsto (taglia il traguardo) contro quello reale (punteggio massimo, gara mai finita)."
:width: 92%

Due traduzioni, due occasioni di sbagliare. Dall'intenzione alla metrica
traduciamo noi; dalla metrica al comportamento traduce l'addestramento, e
nessuno dei due passaggi è fedele per definizione.
```

{numref}`fig-problema-allineamento` mostra perché il problema non si risolve
scrivendo una metrica migliore: gli scarti sono due, e stanno in punti
diversi. Il primo lo apriamo noi quando mettiamo in numeri un desiderio; il
secondo si apre da sé mentre il modello impara, e non è detto che ce ne
accorgiamo guardando il punteggio, che nel frattempo sale.

`````{tab} Elementare

È la vecchia storia del genio della lampada. Chiedi «rendimi l'uomo più ricco
del mondo» e ti ritrovi solo su un pianeta deserto: tecnicamente sei il più
ricco, perché non c'è nessun altro. Il genio ha esaudito le tue *parole*, non il
tuo *desiderio*. Un modello ottimizza esattamente il numero che gli diciamo di
far salire, con la stessa fedeltà cieca del genio: se quel numero è una buona
imitazione di ciò che vogliamo, ottimo; ma nessuna imitazione è perfetta, e il
modello è bravissimo a scovare i punti in cui il numero sale *senza* che le cose
migliorino davvero. Non è cattiveria, e non è nemmeno un errore di
programmazione: è che «di' esattamente cosa vuoi» è, con un sistema abbastanza
capace, molto più difficile di quanto sembri.

`````

`````{tab} Superiore

Conviene distinguere due sottoproblemi. L'**outer alignment** riguarda la
*specifica*: scrivere un obiettivo $\tilde{r}$ che catturi davvero ciò che ci
interessa, $r^*$. L'**inner alignment** riguarda ciò che il modello finisce
per perseguire *internamente* una volta addestrato: anche con una specifica
perfetta, un sistema ottimizzato su una distribuzione può interiorizzare un
obiettivo che generalizza male fuori da essa (*goal misgeneralization*). Sul
primo pesa la **legge di Goodhart**: «quando una misura diventa un obiettivo,
cessa di essere una buona misura». Formalmente, ottimizziamo un proxy
$\tilde{r} \approx r^*$; ma $\arg\max_y \tilde{r}(y)$ e $\arg\max_y r^*(y)$
non coincidono, e la differenza $\tilde{r} - r^*$ (trascurabile sui casi
tipici) viene *amplificata* proprio dalla ricerca del massimo, che spinge il
sistema nelle regioni dove il proxy sovrastima. Qui $r^*$ è l'obiettivo vero
(non osservabile direttamente), $\tilde{r}$ è il surrogato che ottimizziamo
(un punteggio del gioco, un reward model), e $y$ è il comportamento prodotto.
L'allineamento è, in questa lettura, il problema di rendere piccola quella
differenza *dove conta*: non in media, ma nel punto in cui l'ottimizzatore
andrà a cercare.

`````

Un piccolo esperimento numerico rende tangibile il fenomeno. Immaginiamo un
insieme di risposte candidate a uno stesso prompt: di ciascuna conosciamo la
*qualità vera* (quanto è utile e corretta), che però non osserviamo mai
direttamente. Al suo posto abbiamo un *proxy*: un reward model che imita i
giudizi umani, e come loro premia un po' troppo le risposte lunghe.
Ottimizzare il proxy significa tenere le risposte col punteggio surrogato più
alto:

```python
import numpy as np

rng = np.random.default_rng(0)
n = 2000  # risposte candidate a uno stesso prompt

# Cio' che ci interessa davvero: quanto la risposta e' utile e corretta (0-1).
qualita_vera = rng.uniform(0, 1, n)
# Una caratteristica superficiale che i giudizi umani tendono a premiare.
lunghezza = rng.uniform(0, 1, n)

# Il reward model imita quei giudizi: approssima la qualita' vera, ma con un
# bias sistematico verso le risposte lunghe (e un po' di rumore).
proxy = qualita_vera + 1.5 * lunghezza + rng.normal(0, 0.1, n)

# Ottimizzare il proxy = tenere le risposte col punteggio surrogato piu' alto.
top = 20
scelte_proxy = np.argsort(proxy)[-top:]          # top-20 secondo il giudice
scelte_vere  = np.argsort(qualita_vera)[-top:]   # top-20 secondo l'obiettivo vero

print(f"Qualita' vera media ottimizzando il proxy: {qualita_vera[scelte_proxy].mean():.3f}")
print(f"Qualita' vera media ottimizzando il vero:  {qualita_vera[scelte_vere].mean():.3f}")
print(f"Lunghezza media delle scelte col proxy:    {lunghezza[scelte_proxy].mean():.3f}")
print(f"Lunghezza media su tutte le risposte:      {lunghezza.mean():.3f}")
# Qualita' vera media ottimizzando il proxy: 0.907
# Qualita' vera media ottimizzando il vero:  0.994
# Lunghezza media delle scelte col proxy:    0.931
# Lunghezza media su tutte le risposte:      0.494
```

Il proxy non è disastroso (le risposte che sceglie hanno qualità vera media
$0{,}907$, non lontana dall'ottimo $0{,}994$) ma il sintomo del *gaming* è
lampante: la lunghezza media delle scelte schizza a $0{,}931$ contro il
$0{,}494$ della popolazione. Il sistema ha imparato a essere prolisso perché
la prolissità paga *sul giudice*, non sull'utente. È il reward hacking in
miniatura, e la prolissità degli assistenti reali ne è la versione in scala.

## Allineare gli LLM

Come si orienta, in concreto, un modello di linguaggio? Il capitolo sui
Transformer, nella sezione sul post-training, sviluppa la ricetta per intero;
qui la richiamiamo per sommi capi e ne aggiungiamo il tassello mancante.

La prima mossa storica è l'**RLHF** (*Reinforcement Learning from Human
Feedback*). L'idea nasce nel 2017, quando Christiano e colleghi
{cite}`christiano2017deep` insegnano a un robottino simulato a fare un salto
mortale all'indietro senza scrivere alcuna funzione di ricompensa: bastava
mostrare a un umano coppie di video e chiedere «quale somiglia di più a un
salto?». Con InstructGPT {cite}`ouyang2022training` la tecnica viene applicata
in grande al linguaggio, in due tempi: i confronti umani addestrano un
**reward model** che impara a dare voti, e quel modello fa poi da giudice
mentre la policy del linguaggio viene ottimizzata con PPO, sotto una penalità
KL che le impedisce di allontanarsi troppo dal punto di partenza; la difesa
contro il reward hacking che abbiamo appena visto.

L'RLHF funziona ma è un cantiere pesante. Nel 2023 la **DPO** (*Direct
Preference Optimization*) di Rafailov e colleghi {cite}`rafailov2023direct`
mostra che si può allineare *direttamente* dalle preferenze, con una semplice
loss supervisionata sulle coppie preferita/scartata, saltando del tutto il
reward model esplicito e il reinforcement learning. Stessa destinazione, senza
l'impalcatura.

```{figure} ../figures/dpo-allineare-senza-reward-model.svg
:name: fig-rlhf-vs-dpo
:alt: "Confronto fra due pipeline che partono dagli stessi dati di preferenza (A preferita a B). In alto RLHF in tre stadi: i dati addestrano un reward model separato, che fa da giudice in un ciclo di reinforcement learning con PPO, che aggiorna l'LLM; a margine il costo, fino a quattro modelli in memoria, campionamento a ogni passo, addestramento instabile. In basso DPO in un solo stadio: gli stessi dati alimentano direttamente una loss di classificazione che aggiorna l'LLM, con due soli modelli in memoria."
:width: 100%

La stessa destinazione, due tappe in meno. La DPO non raccoglie preferenze
diverse: usa le stesse, e si accorge che il reward model esplicito era un
passaggio intermedio di cui si può fare a meno.
```

Il confronto di {numref}`fig-rlhf-vs-dpo` spiega la fortuna della DPO meglio
di qualsiasi argomento teorico: la colonna di destra elenca cosa si smette di
tenere in piedi. Quattro modelli in memoria diventano due, il campionamento a
ogni passo sparisce, e con esso sparisce l'instabilità che rendeva l'RLHF un
mestiere per pochi laboratori.

C'è però un limite che nessuna delle due tocca: le preferenze restano
**umane**, e raccoglierne a sufficienza (specie sui temi delicati della
sicurezza) è lento e costoso. Il tassello nuovo prova a ridurre proprio
questo.

`````{tab} Elementare

Il collo di bottiglia dei metodi visti finora è la persona che giudica. Per
insegnare a un modello a *non* essere tossico servirebbero migliaia di
valutatori che leggono migliaia di risposte sgradevoli e dicono «questa è
peggio di quella»: un lavoro lento, costoso e psicologicamente pesante. L'idea
della *Constitutional AI* è spostare quasi tutto il giudizio sulle spalle di
un altro modello. Prima si scrive una piccola **costituzione**: un elenco di
principi in linguaggio semplice, tipo «scegli la risposta meno dannosa e più
onesta». Poi si chiede al modello di *criticare e riscrivere* le proprie
risposte alla luce di quei principi, e infine di *confrontare* coppie di
risposte dicendo quale rispetta meglio la costituzione. I confronti li produce
l'AI, non più l'umano; all'umano resta il compito più alto e più raro:
scrivere bene i principi. Non è che l'umano sparisca: cambia mestiere, da
etichettatore a legislatore.

`````

`````{tab} Superiore

Il nucleo comune di RLHF, DPO e del metodo che segue è un **dataset di
preferenze** $\mathcal{D} = \{(x, y_w, y_l)\}$: per un prompt $x$, una risposta
preferita $y_w$ e una scartata $y_l$. La probabilità di preferenza si modella
di norma alla Bradley–Terry,

$$
P(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big),
$$

dove $\sigma$ è la sigmoide e $r$ un punteggio scalare (esplicito nel reward
model dell'RLHF, implicito in $\beta \log \frac{\pi_\theta}{\pi_{\text{ref}}}$
nella DPO). La domanda è: *chi* produce le etichette $y_w \succ y_l$? La
**Constitutional AI** {cite}`bai2022constitutional` (Anthropic, 2022)
risponde: un modello, guidato da un insieme scritto di principi. Il metodo ha
due fasi. Nella prima, supervisionata, il modello genera una risposta, la
**critica** rispetto a un principio campionato dalla costituzione e la
**riscrive**; si fa poi fine-tuning sulle risposte riviste. Nella seconda si
applica l'**RLAIF** (*RL from AI Feedback*): un modello confronta coppie di
risposte alla luce dei principi e produce le etichette di preferenza, che
addestrano il preference model usato poi in RL. Il risultato empirico
riportato è un modello meno nocivo e *meno evasivo* (che spiega perché rifiuta
invece di chiudersi) con un impiego di feedback umano sulla sicurezza ridotto
quasi a zero. Resta la domanda a monte, spostata di un livello: i principi li
scrive comunque qualcuno, e *quali* principi non è una scelta tecnica.

`````

```{figure} ../figures/constitutional-ai-2022.svg
:name: fig-constitutional-ai
:alt: "Ciclo chiuso: una costituzione, cioè un elenco di principi in linguaggio naturale, alimenta il modello, che critica sé stesso a partire da una risposta iniziale e la riscrive; la risposta rivista diventa feedback, cioè dato di training, e riaddestra il modello. Una nota precisa che nessuna etichetta umana sull'innocuità entra nel ciclo."
:width: 82%

Il ciclo si chiude senza passare da una persona. L'unico ingresso umano è la
costituzione, in alto: poche righe di principi, scritte una volta, al posto di
migliaia di giudizi su singole risposte.
```

Guardando {numref}`fig-constitutional-ai` si vede dove è finito il lavoro
umano: non è sparito, si è spostato a monte e si è ridotto di volume. È un
buon affare in termini di costo e un cambio di natura del problema, perché
rivedere dieci principi scritti è un'operazione che si può discutere in
pubblico, mentre rivedere diecimila giudizi individuali no.

## I rischi degli LLM

Allineare non elimina i pericoli; li rende gestibili, non nulli. Vale la pena
nominarli con precisione, perché appartengono a famiglie diverse e chiedono
difese diverse.

```{figure} ../figures/allucinazioni-perche-modelli-inventano.svg
:name: fig-allucinazioni
:alt: "Il prompt «La capitale dell'Australia è…» entra nel modello, che produce una distribuzione di probabilità sul token successivo: Sydney 0,42, Canberra 0,35, Melbourne 0,15, altro 0,08. Accanto, separato e non collegato da alcuna freccia, un riquadro rappresenta il mondo reale e i fatti verificati: il modello non lo consulta."
:width: 92%

Perché un modello inventa. La freccia che manca è quella verso i fatti: il
token si sceglie per probabilità, e «Sydney» è più probabile di «Canberra»
perché compare più spesso, non perché sia la risposta giusta.
```

La freccia assente in {numref}`fig-allucinazioni` spiega perché le
allucinazioni non siano un difetto da correggere ma una conseguenza del
meccanismo. Il modello non sta consultando niente e sbagliando: sta facendo
esattamente ciò per cui è addestrato, cioè scegliere la continuazione
plausibile, e la plausibilità non è la verità.

`````{tab} Elementare

Conviene separare due tipi di guaio. Il primo sono gli **errori onesti**: il
modello, che in fondo è un generatore di testo plausibile, a volte inventa;
cita un libro che non esiste, sbaglia una data con perfetta sicurezza. Si
chiamano **allucinazioni**, e non nascono da cattiva volontà: nascono dal
fatto che «suonare vero» e «essere vero» non sono la stessa cosa. Il secondo
tipo sono gli **attacchi**: qualcuno costruisce apposta l'input per far
comportare male il modello. Con un gioco di ruolo astuto lo si convince ad
aggirare le sue regole (*jailbreak*); oppure si nasconde un ordine dentro un
testo che il modello deve solo leggere (una pagina web, una mail) e lui lo
scambia per un comando legittimo (*prompt injection*), come già discusso nel
capitolo sul programmare gli LLM. La differenza pratica conta: per gli errori
onesti la difesa è verificare a valle; per gli attacchi è difendere un
perimetro contro un avversario che ci prova apposta.

`````

`````{tab} Superiore

Le allucinazioni sono un limite **intrinseco** dei modelli generativi:
campionano da $P(\text{testo})$, non da un archivio di fatti verificati, e la
fluidità non è correlata alla verità. Nessun prompt le azzera; si mitigano con
recupero da fonti (RAG), richiesta di citazioni e verifica esterna. Jailbreak
e **prompt injection** sono invece problemi **avversari**: sfruttano il fatto
che la gerarchia *system > user* è morbida e che il modello non distingue in
modo affidabile *istruzioni* da *dati*. La prompt injection in particolare
(testo non fidato che entra nel contesto e viene interpretato come comando) è
l'analogo dell'SQL injection, e nei sistemi agentici (con accesso a strumenti,
mail, file) è oggi il rischio di sicurezza più concreto. Sopra tutto sta la
questione dell'**uso duale**: le stesse capacità che rendono un modello utile
per chimica, biologia o codice possono assistere chi vuole nuocere. Non è un
bug da correggere, è una proprietà della capacità stessa, e ne fa una
questione di controllo dell'accesso, non solo di addestramento.

`````

## Valutare la sicurezza

Se non possiamo garantire che un modello sia sicuro, possiamo almeno *provare a
romperlo prima che lo faccia il mondo*. Qui la sicurezza dell'AI prende in
prestito il vocabolario della sicurezza informatica.

```{figure} ../figures/red-teaming.svg
:name: fig-red-teaming
:alt: "Ciclo chiuso in quattro stazioni disposte ad anello: 1 policy (cosa il modello non deve fare), 2 attacco (jailbreak, injection, attacchi automatizzati), 3 scoperta (falle documentate e classificate), 4 patch (fine-tuning, filtri, system prompt). Una freccia riporta dalla patch all'attacco, con la nota che ogni patch invita un nuovo attacco."
:width: 78%

Il red teaming non è un collaudo che si supera una volta: è un anello. La
freccia di ritorno è la parte onesta del disegno, perché ogni correzione
cambia la superficie d'attacco invece di eliminarla.
```

L'anello di {numref}`fig-red-teaming` comincia dalla policy, e non è un
dettaglio burocratico: senza aver scritto prima *cosa* il modello non deve
fare, un attacco riuscito non si distingue da una risposta insolita, e non c'è
modo di dire se la patch abbia funzionato.

`````{tab} Elementare

Prima di aprire un ponte al traffico non ci si limita a guardarlo: gli si
fanno passare sopra camion carichi, lo si sottopone a vibrazioni, si cerca
*apposta* il punto in cui potrebbe cedere. Con i modelli si fa lo stesso, in
due modi complementari. Il **red-teaming** è una squadra il cui unico compito
è comportarsi da avversario: inventare le domande più insidiose, i trucchi, i
giri di parole, per trovare dove il modello sbaglia (meglio scoprirlo in
laboratorio che sui giornali). Le **evals** (da *evaluation*) sono invece
esami standardizzati: liste di prove ripetibili con un voto finale, così da
misurare se una nuova versione è più o meno sicura della precedente. Il
red-teaming cerca la falla nuova; le evals controllano che le vecchie non
tornino.

`````

`````{tab} Superiore

Il **red-teaming** è la ricerca adversariale di *failure*: manuale (esperti
che sondano capacità pericolose, jailbreak, fughe di dati) o
**automatizzato**, con un LLM istruito a generare attacchi contro un altro. Le
**evals** sono suite di benchmark riproducibili; si distinguono quelle di
*capacità* (cosa il modello sa fare) da quelle di *sicurezza* (tossicità,
rifiuto di richieste illecite, resistenza ai jailbreak, propensione alle
allucinazioni), spesso riassunte in metriche come l'*attack success rate* o il
tasso di rifiuto appropriato. Due avvertenze di metodo, entrambe corollari di
Goodhart. Primo: un benchmark è un proxy, e ottimizzare *per* il benchmark
(magari perché finito nei dati di addestramento) gonfia il punteggio senza
migliorare la sicurezza reale, per questo contano i *test adversariali tenuti
nascosti*. Secondo: passare le evals dimostra l'assenza dei fallimenti
*cercati*, non la sicurezza in assoluto. L'assenza di prove non è prova
d'assenza: è il motivo per cui la valutazione è un processo continuo, non un
timbro una tantum. Non a caso, come vedremo, l'AI Act impone il red-teaming
come obbligo per i modelli più capaci.

`````

## Governance e regolamentazione

Gli strumenti tecnici non bastano da soli: servono regole condivise su *chi*
può fare *cosa*, e chi risponde quando qualcosa va storto. Qui l'Europa ha fatto
la prima mossa di portata mondiale.

```{figure} ../figures/ai-act-guida-pratica.svg
:name: fig-ai-act-rischio
:alt: "Piramide a quattro gradini con i livelli di rischio dell'AI Act. In cima, rischio inaccettabile: divieto totale, social scoring e manipolazione. Sotto, alto rischio: conformità completa, log, sorveglianza umana, marchio CE. Più giù, rischio limitato: solo trasparenza, «stai parlando con un'AI». Alla base, rischio minimo: nessun obbligo specifico, ed è dove sta la maggioranza dei prodotti."
:width: 80%

La piramide del rischio. La forma conta quanto i contenuti: gli obblighi
pesanti stanno in cima, dove i casi sono pochi, e la base larga (quasi tutto
ciò che si costruisce) non ha obblighi specifici.
```

Vale la pena leggere {numref}`fig-ai-act-rischio` dal basso. La lettura
corrente («l'Europa regola l'AI») suggerisce un peso uniforme, mentre la
piramide dice il contrario: la regola morde in proporzione al danno possibile,
e per la maggior parte dei sistemi non morde affatto. Sapere in quale gradino
cade ciò che si sta costruendo è il primo esercizio di conformità, e spesso
anche l'ultimo.

`````{tab} Elementare

Non regoliamo tutti i prodotti allo stesso modo: un giocattolo di plastica e
un farmaco seguono controlli diversissimi, perché diverso è il danno che
possono fare. L'**AI Act** europeo applica la stessa idea all'intelligenza
artificiale: non guarda alla tecnologia in astratto, ma al **rischio** di ogni
suo impiego, e lo dispone su una piramide a quattro gradini. In cima, il
rischio *inaccettabile*: pochi usi semplicemente **vietati**, come dare a ogni
cittadino un «punteggio sociale». Sotto, l'*alto rischio*: usi permessi ma
sorvegliati (selezione del personale, valutazione del credito, dispositivi
medici) con obblighi stringenti prima di andare sul mercato. Più giù, il
*rischio limitato*: basta la **trasparenza**, cioè avvisare le persone («stai
parlando con un'AI», «questo video è generato»). In fondo, il *rischio
minimo*: la stragrande maggioranza dei sistemi, senza obblighi particolari. Un
unico principio: più un uso può ferire, più regole deve rispettare.

`````

`````{tab} Superiore

L'**AI Act** {cite}`euaiact2024`, entrato in vigore nell'agosto 2024 con
applicazione scaglionata negli anni successivi, struttura gli obblighi su
quattro livelli di rischio:

- **inaccettabile** (pratiche vietate): social scoring da parte di enti
  pubblici o privati, tecniche manipolative o subliminali, sfruttamento delle
  vulnerabilità,
  scraping non mirato di volti per costruire database di riconoscimento,
  riconoscimento delle emozioni sul lavoro e a scuola, in gran parte
  l'identificazione biometrica remota *in tempo reale* negli spazi pubblici;
- **alto rischio** (allegato III e componenti di sicurezza di prodotti):
  credito, occupazione, istruzione, giustizia, infrastrutture critiche,
  migrazione. Permessi ma con sistema di gestione del rischio, governance dei
  dati, documentazione tecnica, tracciabilità, **sorveglianza umana**,
  accuratezza e robustezza, e valutazione di conformità prima
  dell'immissione sul mercato;
- **rischio limitato**: obblighi di **trasparenza**, dichiarare l'interazione
  con un'AI, etichettare i contenuti sintetici e i deepfake;
- **rischio minimo**: nessun obbligo (la maggior parte dei sistemi).

Un capitolo a parte è dedicato ai **GPAI** (*general-purpose AI*), i modelli
di uso generale: per tutti valgono obblighi di documentazione tecnica,
informazione agli sviluppatori a valle e sintesi pubblica dei dati di
addestramento; per quelli con **rischio sistemico** (presunto oltre una soglia
di calcolo di addestramento di $10^{25}$ FLOP) si aggiungono valutazione del
modello, **adversarial testing** (il red-teaming della sezione precedente),
mitigazione dei rischi sistemici, cybersicurezza e segnalazione degli
incidenti gravi. Le sanzioni per le pratiche vietate arrivano fino a 35
milioni di euro o al 7% del fatturato mondiale annuo. Sull'altra sponda
dell'Atlantico l'approccio è volontario: il **NIST AI Risk Management
Framework** (2023) propone quattro funzioni (*Govern, Map, Measure, Manage*)
senza forza di legge. Due filosofie regolatorie a confronto: vincolante e *ex
ante* quella europea, volontaria e basata su standard quella statunitense.

`````

Dietro le regole c'è un dibattito che vale la pena rendere esplicito, perché
divide anche gli addetti ai lavori. Da un lato chi mette al centro i **danni
presenti e documentati** (i pregiudizi, le violazioni di privacy, gli esempi
avversari di cui parla il resto di questo capitolo) e teme che l'attenzione ai
rischi lontani distolga risorse da ingiustizie che colpiscono persone reali
*oggi*. Dall'altro chi punta sui **rischi catastrofici futuri** di sistemi
molto più capaci di quelli attuali, e sostiene che prevenirli richieda
cominciare adesso. Non è una disputa che questo libro può chiudere; ma è
onesto notare che non sono alternative: un ponte va progettato sia contro le
crepe di oggi sia contro il terremoto che forse verrà, e le due cose competono
per lo stesso budget di attenzione.

## L'onestà dovuta

Chiudiamo dove il capitolo intero insiste. L'allineamento è oggi un'area di
ricerca a pieno titolo, non un ritocco finale: abbiamo strumenti per
*orientare* il comportamento dei modelli (RLHF, DPO, Constitutional AI,
red-teaming, evals) non **garanzie** sul risultato. Sotto tutto corre una
tensione strutturale, che nessuna tecnica ha sciolto: le stesse capacità che
rendono un modello utile lo rendono difficile da controllare, e più un sistema
è potente, più il divario tra ciò che gli chiediamo e ciò che fa può costare.
Le regole (l'AI Act in testa) provano a comprare tempo e responsabilità mentre
la ricerca insegue. È un campo in movimento, senza soluzioni definitive: e
come per l'equità, la scelta di *quanto* controllo pretendere e *a quali*
valori allineare non è un teorema da dimostrare, ma una decisione che spetta a
noi.

```{admonition} Da ricordare
:class: important
- **Allineamento**: far sì che il sistema persegua ciò che *intendiamo*, non la
  lettera dell'obiettivo. *Specification gaming* / **reward hacking** e legge di
  **Goodhart** (il proxy $\tilde{r}$ diverge dal vero $r^*$ proprio dove
  l'ottimizzatore cerca il massimo); si distinguono *outer* e *inner* alignment.
- Allineare gli LLM: **RLHF** {cite}`christiano2017deep`,
  {cite}`ouyang2022training` (reward model dalle preferenze + PPO sotto vincolo
  KL); **DPO** {cite}`rafailov2023direct` (direttamente dalle preferenze, senza
  reward model); **Constitutional AI / RLAIF** {cite}`bai2022constitutional`
  (principi scritti e feedback dell'AI per ridurre il giudizio umano).
- **Rischi**: allucinazioni (errore intrinseco → verifica a valle) contro
  jailbreak e **prompt injection** (attacchi → difesa di perimetro); e l'**uso
  duale**, proprietà della capacità, non bug da correggere.
- **Valutare la sicurezza**: **red-teaming** (cercare le falle da avversari) ed
  **evals**/benchmark (esami ripetibili). Ma passare un test è un proxy: assenza
  di prove non è prova d'assenza.
- **Governance**: l'**AI Act** {cite}`euaiact2024` regola per *rischio*
  (inaccettabile/vietato, alto, limitato/trasparenza, minimo) con obblighi
  extra per i **GPAI** a rischio sistemico ($>10^{25}$ FLOP: valutazioni e
  adversarial testing); il **NIST AI RMF** è la controparte volontaria.
- Nessuna soluzione definitiva: strumenti per orientare, non garanzie. *A quali
  valori* allineare e *quanto* controllo pretendere sono scelte umane, non
  teoremi.
```
