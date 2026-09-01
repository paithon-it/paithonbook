# Allineamento e governance: dai valori umani alle regole

La barca che gira in tondo dentro una laguna di *CoastRunners*, prendendo
fuoco e andando contromano mentre incassa in media il venti per cento di punti
in più dei giocatori umani senza mai tagliare il traguardo, la racconta per
esteso la {doc}`sezione su esplorazione e ricompensa
</DeepReinforcementLearning/esplorazione-e-ricompensa>`
{cite}`clark2016faulty`. Qui serve la frase che la chiude, perché è da lì che
riparte tutto il resto. Quell'agente aveva fatto esattamente ciò che gli
avevamo chiesto. Non ciò che intendevamo.

Questa distanza (tra la **lettera** di un obiettivo e la sua **intenzione**) è
il cuore del problema dell’**allineamento**. La parola dice quello che sembra
dire: due cose sono allineate quando si sovrappongono, come due righe messe una
sull'altra, e un sistema è allineato quando quello che fa combacia con quello
che volevamo. (Nella sezione sulla traduzione automatica, dentro il capitolo di
Natural Language Processing, la stessa parola indica un'altra cosa: quale
parola di una lingua corrisponde a quale parola dell'altra. Sono due usi
diversi, e qui vale questo.) Finché il sistema è un motoscafo in un videogioco,
la scorciatoia fa sorridere. Quando lo stesso meccanismo governa un modello che
parla con milioni di persone, o che filtra domande di lavoro e richieste di
prestito, smette di far sorridere. Questa sezione chiude
il capitolo affrontando le due facce di quella distanza: come si prova a
*orientare* il comportamento di un sistema verso ciò che vogliamo davvero
(l'allineamento), e quale impalcatura di regole e verifiche prova a tenere il
tutto entro binari accettabili (la governance).

## Il problema dell'allineamento

L'aneddoto del motoscafo ha un nome tecnico, anzi due per la stessa cosa:
**reward hacking**, «scassinare la ricompensa», e **specification
gaming**, il «giocare» con la specifica, cioè con la descrizione precisa di ciò
che abbiamo chiesto. Nell'apprendimento per rinforzo il guaio si incontra a
ogni ricompensa mal scritta, e torna identico nell'ultima fase con cui si
rifiniscono i grandi modelli di linguaggio (gli **LLM**, dall'inglese *large
language model*).

La formula da tenere a mente è questa: quando ottimizzi un **surrogato** del
tuo vero obiettivo, prima o poi ottieni il surrogato e perdi l'obiettivo.
Surrogato, o *proxy*, sono le due parole con cui d'ora in poi si chiamerà una
**imitazione** di ciò che volevamo davvero: un numero che gli somiglia
abbastanza da poterlo misurare, e che proprio per questo non è la cosa vera.
Il punteggio del videogioco era il surrogato di «vinci la gara».

```{figure} ../figures/problema-allineamento.svg
:name: fig-problema-allineamento
:alt: "Schema in tre riquadri messi in fila da sinistra a destra: l'intenzione umana («vinci la gara») viene tradotta in una metrica dichiarata («massimizza i punti»), che l'addestramento trasforma in un obiettivo appreso («gira in tondo sui bonus»). Sotto la freccia fra il primo e il secondo riquadro pende il primo scarto, la metrica cattura l'intenzione?; sotto quella fra il secondo e il terzo il secondo scarto, il modello persegue davvero la metrica? In basso il comportamento previsto (taglia il traguardo) contro quello reale (punteggio massimo, gara mai finita)."
:width: 92%

Due traduzioni, due occasioni di sbagliare. Dall'intenzione al numero che
misuriamo traduciamo noi; da quel numero al comportamento traduce
l'addestramento. Nessuno dei due passaggi è fedele, e non per sbadataggine: un
desiderio non entra mai per intero in un numero, e un numero non determina mai
per intero un comportamento. Nel disegno i due scarti portano i nomi con cui si
chiamano in letteratura, *outer* e *inner alignment*: quello di fuori, fra noi
e la macchina, e quello di dentro, fra ciò che le abbiamo chiesto e ciò che ha
finito per imparare.
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
tuo *desiderio*. Con un modello succede lo stesso: noi gli diamo un numero da
far salire, e lui lo fa salire. Se quel numero è una buona imitazione di ciò che
vogliamo, ottimo; ma nessuna imitazione è perfetta, e il modello è bravissimo a
scovare i punti in cui il numero sale *senza* che le cose migliorino davvero.

Poi c'è un secondo scivolone, e quello il genio non lo fa. Un ragazzo assunto a
raccogliere mele viene pagato per ogni mela matura: il conto è giusto, da
imbrogliare non c'è niente. Solo che nei filari dove ha imparato erano mature
tutte, e lui si è fatto la regola più comoda, staccare quello che arriva a mano.
Nel filare nuovo, dove metà sono acerbe, continua a staccare tutto. La sua
regola e la nostra si somigliavano finché si somigliavano i filari. Anche un
modello impara dai casi che gli capitano, e quale regola si sia fatto lo
scopriamo il giorno in cui gliene arriva uno diverso.

Non è cattiveria, e non è nemmeno un errore di programmazione: con un sistema
abbastanza capace, dire esattamente che cosa si vuole è difficile, e ottenere
che sia proprio quello a essere imparato è un secondo mestiere.

`````

`````{tab} Superiore

Conviene distinguere due sottoproblemi. L’**outer alignment** riguarda la
*specifica*: scrivere un obiettivo $\tilde{r}$ che catturi davvero ciò che ci
interessa, $r^*$. L’**inner alignment** riguarda ciò che il modello finisce
per perseguire *internamente* una volta addestrato: anche con una specifica
perfetta, un sistema ottimizzato su una distribuzione può interiorizzare un
obiettivo che generalizza male fuori da essa (*goal misgeneralization*). Sul
primo pesa la **legge di Goodhart**, nella formulazione resa celebre da
Marilyn Strathern {cite}`strathern1997improving`: «quando una misura diventa un
obiettivo, cessa di essere una buona misura». Formalmente, ottimizziamo un
proxy $\tilde{r} \approx r^*$; ma $\arg\max_y \tilde{r}(y)$ e
$\arg\max_y r^*(y)$ in generale non coincidono, e la differenza
$\tilde{r} - r^*$, piccola dove il proxy è stato stimato (i comportamenti
tipici, quelli su cui esistevano dati), viene
*amplificata* proprio dalla ricerca del massimo, che spinge il sistema fuori da
quella regione, dove il proxy sovrastima. Qui $r^*$ è l'obiettivo vero
(non osservabile direttamente), $\tilde{r}$ è il surrogato che ottimizziamo
(un punteggio del gioco, un reward model), e $y$ è il comportamento prodotto.
L'allineamento è, in questa lettura, il problema di rendere piccola quella
differenza *dove conta*: non in media, ma nel punto in cui l'ottimizzatore
andrà a cercare.

`````

Un piccolo esperimento numerico rende tangibile il fenomeno. Immaginiamo tante
risposte candidate alla stessa domanda. Di ciascuna esiste una *qualità vera*
(quanto è utile e corretta), che però non osserviamo mai: è un numero fra zero
e uno, e più è alto meglio è. Al suo posto abbiamo il surrogato, cioè un
giudice automatico che dà un voto.

Le regole del giocattolo sono tre, e vanno dette tutte prima, altrimenti i
numeri non si possono commentare. La **qualità vera** di una risposta dipende
dal suo merito e dalla sua lunghezza, due numeri anche loro fra zero e uno, e
la lunghezza aiuta fino a un certo punto e poi stanca chi legge: una risposta
troppo lunga è davvero peggio, non solo giudicata peggio. Il **premio alla
lunghezza** è il difetto del giudice, che il merito lo vede ma ci somma un
bonus per le righe in più, e quel bonus
non smette mai di crescere; è la deformazione che i giudici automatici
ereditano dalle persone da cui hanno imparato, e gliel'abbiamo data noi. E c'è
il **tremolio**: il giudice sbaglia anche a caso, di poco, come chiunque dia
molti voti di fila.

Quel che l'esperimento misura è un'altra cosa, e non è affatto ovvia: **quanto
danno fa quel difetto al crescere della pressione con cui si ottimizza**. La
pressione qui si simula generando $k$ risposte, tenendo quella che il giudice
preferisce, e poi alzando $k$.

```python
import numpy as np

rng = np.random.default_rng(0)
n = 200_000          # risposte candidate

# Cio' che ci interessa e non osserviamo: il merito della risposta, e una
# lunghezza che aiuta fino a un punto e oltre quel punto stanca chi legge.
merito = rng.uniform(0, 1, n)
lunghezza = rng.uniform(0, 1, n)
qualita_vera = merito * (1 - ((lunghezza - 0.3) / 0.7) ** 2)

# Il giudice non vede la qualita' vera: vede le caratteristiche di superficie.
# Sui casi tipici ha imparato "piu' lunga = meglio", e lo estrapola oltre.
proxy = merito + 0.4 * lunghezza + rng.normal(0, 0.1, n)

def migliore_di(k):
    """Alza la pressione: fra k candidati tiene quello che il giudice preferisce."""
    m = (n // k) * k
    scelto = proxy[:m].reshape(-1, k).argmax(axis=1)
    riga = np.arange(m // k)
    def media(v):
        return v[:m].reshape(-1, k)[riga, scelto].mean()
    return media(qualita_vera), media(merito), media(lunghezza)

print(f"{'candidati':>10} {'qualita vera':>13} {'merito':>8} {'lunghezza':>10}")
for k in (1, 3, 10, 30, 100, 1000):
    q, me, l = migliore_di(k)
    print(f"{k:>10} {q:>13.3f} {me:>8.3f} {l:>10.3f}")
```

```text
 candidati  qualita vera   merito  lunghezza
         1         0.373    0.499      0.501
         3         0.495    0.722      0.586
        10         0.500    0.857      0.686
        30         0.431    0.907      0.773
       100         0.361    0.934      0.830
      1000         0.267    0.953      0.885
```

La colonna della qualità vera è la storia, e non è la storia che ci si
aspetta. All'inizio ottimizzare il giudice **funziona**: da un candidato solo
a tre, la qualità vera sale da $0{,}373$ a $0{,}495$. Il motivo è che fra tre
risposte prese a caso le differenze di merito sono grosse e quelle di
lunghezza contano poco, quindi scegliere quella che piace al giudice vuol dire
quasi sempre scegliere quella che vale di più.

Poi la salita si ferma attorno a $0{,}50$, e la colonna **scende**: a mille
candidati la qualità vera è $0{,}267$, peggio che non ottimizzare affatto. Il
motivo si legge nelle altre due colonne, ed è la cosa più istruttiva della
tabella: **salgono tutte e due, ma una sola può salire per sempre**. Il merito
delle risposte scelte va da $0{,}499$ a $0{,}953$ e poi si arena, perché più di
uno non può valere: fra cento candidati e mille guadagna appena diciannove
millesimi. La lunghezza invece continua per la sua strada, da $0{,}501$ a
$0{,}885$, e ogni passo in quella direzione costa più del precedente, perché è
già molto oltre la misura che a chi legge fa comodo. Da un certo punto in poi,
quindi, il giudice compra pochissimo merito in più pagandolo con parecchia
lunghezza in più. E lui questo non lo sa: nel suo voto la lunghezza è sempre e
soltanto un pregio.

È la legge di Goodhart in una tabella. Finché la pressione è bassa, il
surrogato e l'obiettivo vero indicano quasi la stessa direzione. Alzandola, la
ricerca del massimo si sposta proprio dove il giudice si sbaglia di più, e da
lì in poi ogni punto guadagnato su di lui è pagato da chi legge. Oltre un certo punto, quindi, il sistema peggiora mentre il suo punteggio
migliora. Non è vero che ottimizzare di più sia sempre meglio, ed è una cosa
che vale ben oltre questo giocattolo. Gli stessi conti
fatti sui giudici automatici veri, quelli addestrati sui giudizi delle persone,
danno una tabella della stessa forma {cite}`gao2023scaling`.

## Allineare gli LLM

Come si orienta, in concreto, un modello di linguaggio? La ricetta per intero
sta nella {doc}`sezione sul post-training </Transformers/post-training>`; qui
restano i due passaggi che servono a leggere il resto, e il tassello che là non
c'era.

La prima mossa storica è l’**RLHF** (*Reinforcement Learning from Human
Feedback*, apprendimento per rinforzo dai giudizi delle persone). L'idea di far
imparare a un sistema da confronti umani circola dai primi anni Dieci (Akrour e
colleghi nel 2011, Wilson e colleghi nel 2012); quello che succede nel 2017 è
che Christiano e colleghi {cite}`christiano2017deep` la
portano sulle reti profonde, insegnando a un robottino simulato a fare un salto
mortale all'indietro senza scrivere da nessuna parte che cosa fosse un bel
salto: bastava mostrare a una persona coppie di video e chiederle «quale
somiglia di più a un salto mortale?». Sono gli autori stessi a dire che il loro
contributo è averlo fatto funzionare in grande, non l'idea in sé.

Poi la tecnica arriva sul linguaggio, e nel 2022, con InstructGPT
{cite}`ouyang2022training`, sul problema specifico di far seguire le
istruzioni. Se il nome non dice niente, questo sì: è in sostanza il metodo con
cui, a novembre di quello stesso anno, sarebbe stato addestrato ChatGPT.

`````{tab} Elementare

Come funziona, senza sigle. Prima si raccolgono un mucchio di coppie di
risposte, e per ognuna una persona dice quale delle due è migliore. Con quei
confronti si addestra un secondo modello, il cui unico mestiere è dare un voto a
una risposta: è la fotografia dei gusti dei valutatori, e da quel momento può
giudicare da solo migliaia di volte al giorno senza stancarsi. Infine si allena
il modello che parla a prendere voti alti da quel giudice.

C'è un particolare importante in quest'ultimo passo: gli si mette un guinzaglio.
Mentre impara a piacere al giudice, il modello viene tenuto vicino a com'era
prima, e non gli si lascia cambiare troppo. La ragione è quella che i mille
candidati hanno mostrato: lasciato libero di spremere il voto, andrebbe a
cercare i punti in cui il giudice si sbaglia. E il guinzaglio è anche
un'ammissione: il giudice resta un'imitazione del metro vero. Se lo fosse
davvero, non ci sarebbe ragione di trattenere il modello dal piacergli il più
possibile.

`````

`````{tab} Superiore

Il procedimento è in due tempi. I confronti umani addestrano un **reward
model** che impara a dare voti; quel modello fa poi da giudice mentre la
*policy* del linguaggio (cioè il modello generativo, visto come la regola che
sceglie il prossimo token) viene ottimizzata con **PPO**, l'algoritmo di
reinforcement learning descritto nel {doc}`capitolo sul deep reinforcement
learning </DeepReinforcementLearning/overview>`,
sotto una **penalità KL** che misura quanto la policy si è allontanata da
quella di partenza e la riporta indietro. Quella penalità è la difesa contro il
reward hacking appena misurato, ed è anche l'ammissione che il reward model è
un surrogato: se fosse
l'obiettivo vero, non ci sarebbe ragione di impedire di ottimizzarlo fino in
fondo.

InstructGPT non è il primo lavoro a portare l'RLHF sul
linguaggio: l'avevano già fatto Ziegler e colleghi nel 2019, sullo stile e sul
riassunto, e Stiennon e colleghi nel 2020, sul solo riassunto. InstructGPT è il
primo a farlo per il *seguire istruzioni*, che è la capacità da cui dipende
tutto l'uso conversazionale.

`````

Tutto questo funziona, ma è un cantiere pesante: due addestramenti in fila, il
giudice da tirare su e poi il modello da allenare contro di lui. Nel 2023 la
**DPO** (*Direct Preference Optimization*, «ottimizzazione diretta dalle
preferenze») di Rafailov e colleghi {cite}`rafailov2023direct` mostra che il
giudice si può togliere di mezzo. Le stesse coppie di risposte, con scritto
quale delle due era migliore, si danno direttamente al modello che parla,
chiedendogli di rendere un po’ più probabile la risposta preferita e un po’
meno quella scartata. Un passaggio solo, e nessun giudice da costruire.

Su che cosa esattamente sia dimostrato, però, conviene essere precisi, perché
la formula con cui la DPO viene di solito riassunta («stessa destinazione, due
tappe in meno») dice più del vero. Quello che il lavoro dimostra è che i due
metodi, se portati fino in fondo, arrivano **allo stesso punto migliore**. Non
dimostra che ci arrivino per la stessa strada, né che sbaglino nello stesso
modo. E una differenza c'è: nella ricetta col giudice il modello, mentre
impara, produce risposte nuove e se le fa valutare, mentre la DPO legge
soltanto le coppie raccolte in partenza, e su tutto ciò che sta fuori da
quelle non viene mai messa alla prova. Ha modi di sbagliare **suoi**, non
versioni più leggere di quelli della ricetta precedente.

```{figure} ../figures/dpo-allineare-senza-reward-model.svg
:name: fig-rlhf-vs-dpo
:alt: "Confronto fra due pipeline che partono dagli stessi dati di preferenza (A preferita a B). In alto RLHF in tre stadi: i dati addestrano un reward model separato, che fa da giudice in un ciclo di reinforcement learning con PPO, che aggiorna l'LLM; a margine il costo, fino a quattro modelli in memoria, campionamento a ogni passo, addestramento instabile. In basso DPO in un solo stadio: gli stessi dati alimentano direttamente una loss di classificazione che aggiorna l'LLM, con due soli modelli in memoria."
:width: 100%

Lo stesso punto d'arrivo, con meno macchinari in mezzo. La DPO non raccoglie
preferenze diverse: usa le stesse, e mostra che il giudizio del reward model si
può assorbire dentro la formula invece di addestrarlo a parte.
```

Il confronto di {numref}`fig-rlhf-vs-dpo` spiega la fortuna della DPO meglio di
qualsiasi argomento teorico, e la ragione riguarda anche chi non addestrerà mai
un modello: è una questione di quanta macchina serve. La riga sotto ciascuna
delle due file elenca che cosa si smette di tenere in piedi. Con il giudice
bisogna tenere accesi insieme fino a quattro modelli (quello che si sta
allenando, la copia di com'era prima che fa da guinzaglio, il giudice, e un
quarto che indovina il voto finale quando la risposta è ancora a metà) e far
generare risposte nuove a ogni passo; con la DPO i modelli sono due e le
risposte sono già scritte. È la
differenza fra una tecnica che possono permettersi pochi laboratori e una che
può usare un gruppo qualsiasi, ed è il motivo per cui l'allineamento ha smesso
di essere una cosa che si fa in tre posti al mondo.

C'è però un limite che nessuna delle due tocca: le preferenze restano
**umane**, e raccoglierne a sufficienza (specie sui temi delicati della
sicurezza) è lento e costoso. Il tassello nuovo prova a ridurre proprio
questo.

`````{tab} Elementare

Il collo di bottiglia dei metodi visti finora è la persona che giudica. Per
insegnare a un modello a *non* scrivere cose offensive o pericolose
servirebbero migliaia di
valutatori che leggono migliaia di risposte sgradevoli e dicono «questa è
peggio di quella»: un lavoro lento, costoso e psicologicamente pesante. L'idea
della *Constitutional AI* è spostare quasi tutto il giudizio sulle spalle di
un altro modello. Prima si scrive una piccola **costituzione**: un elenco di
principi in linguaggio semplice, tipo «scegli la risposta meno dannosa e più
onesta». Poi si chiede al modello di *criticare e riscrivere* le proprie
risposte alla luce di quei principi, e infine di *confrontare* coppie di
risposte dicendo quale rispetta meglio la costituzione. Il modello che ne esce
si chiude di meno: quando dice di no, dice anche perché. I confronti li produce
l'AI, non più l'umano; all'umano resta il compito più alto e più raro:
scrivere bene i principi. Non è che l'umano sparisca: cambia mestiere, da
etichettatore a legislatore.

E il mestiere nuovo si porta dietro un limite del vecchio. Da mille confronti
esce un voto solo: se su una domanda delicata tre valutatori su cinque
preferiscono la risposta prudente e due la risposta schietta, il giudice impara
quella dei tre, e i due finiscono dentro la media. I principi fanno la stessa
cosa alla luce del sole: una riga sola per tutti, e chi la pensa diversamente
non ci si ritrova.

`````

`````{tab} Superiore

Il nucleo comune di RLHF, DPO e del metodo che segue è un **dataset di
preferenze** $\mathcal{D} = \{(x, y_w, y_l)\}$: per un prompt $x$, una risposta
preferita $y_w$ e una scartata $y_l$. La probabilità di preferenza si modella
di norma alla Bradley-Terry,

$$
P(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big),
$$

dove $\sigma$ è la sigmoide e $r$ un punteggio scalare: esplicito nel reward
model dell'RLHF, implicito nella DPO, dove lo stesso ruolo lo fa
$\beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$, con
$\pi_{\text{ref}}$ la policy di partenza e $\beta$ la forza del vincolo che la
tiene vicina (il termine di normalizzazione si cancella nella differenza).

Quel «di norma» porta con sé un'ipotesi, ed è la stessa su cui poggia
l'equivalenza fra DPO e RLHF:
Bradley-Terry assume che esista **un unico punteggio scalare** da cui
tutte le preferenze discendono, quindi che siano transitive e omogenee fra
annotatori. Preferenze intransitive, o popolazioni con valori diversi, non sono
rappresentabili in quel modello e vengono compresse nella media.

La domanda successiva è: *chi* produce le etichette $y_w \succ y_l$? La
**Constitutional AI** {cite}`bai2022constitutional` (Anthropic, 2022)
risponde: un modello, guidato da un insieme scritto di principi. Il metodo ha
due fasi. Nella prima, supervisionata, il modello genera una risposta, la
**critica** rispetto a un principio campionato dalla costituzione e la
**riscrive**; si fa poi fine-tuning sulle risposte riviste. Nella seconda si
applica l’**RLAIF** (*RL from AI Feedback*): un modello confronta coppie di
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

Allineare non elimina i pericoli; li rende gestibili, non nulli. E appartengono
a famiglie diverse, che chiedono difese diverse, quindi vanno nominati con
precisione.

```{figure} ../figures/allucinazioni-perche-modelli-inventano.svg
:name: fig-allucinazioni
:alt: "Il prompt «La capitale dell'Australia è…» entra nel modello, che produce una distribuzione di probabilità sul token successivo: Sydney 0,42, Canberra 0,35, Melbourne 0,15, altro 0,08. In basso a sinistra un riquadro tratteggiato rappresenta il mondo reale e i fatti verificati, con dentro la risposta giusta, Canberra; il tratteggio che lo collegherebbe alla distribuzione è sbarrato da una croce, perché quel collegamento non esiste."
:width: 92%

Perché un modello inventa. L'esempio è inventato apposta per far vedere il
meccanismo (sulla capitale dell'Australia i modelli in circolazione rispondono
bene): quel che conta è il collegamento sbarrato, quello verso i fatti. La
parola si sceglie per probabilità, e in questo schema «Sydney» è più probabile
di «Canberra» perché compare più spesso, non perché sia la risposta giusta.
```

Il collegamento sbarrato di {numref}`fig-allucinazioni` spiega perché le
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
scambia per un comando legittimo (*prompt injection*), come racconta per esteso
{doc}`Attaccare e difendere un LLM </AIResponsabile/sicurezza-llm>`.
La differenza pratica conta: per gli errori
onesti la difesa è verificare a valle; per gli attacchi è difendere un
perimetro contro un avversario che ci prova apposta.

L'attacco cambia peso quando al modello si dà da fare e non solo da dire. Se
legge la posta e può anche rispondere, l'ordine nascosto dentro una mail che
riceve diventa una mail che parte: il danno smette di essere una frase
sbagliata. Resta poi una cosa che
nessuna correzione toglie: le stesse capacità che rendono un modello utile a un
chimico o a un programmatore lo rendono utile a chi vuole nuocere. Nel modello
non c'è niente da aggiustare, e la partita si gioca su chi può metterci le
mani: si chiama **uso duale**.

`````

`````{tab} Superiore

Le allucinazioni sono un limite **intrinseco** dei modelli generativi:
campionano da $P(\text{testo})$, non da un archivio di fatti verificati, e la
fluidità non è una prova di verità. Nessun prompt le azzera; si mitigano con
recupero da fonti (RAG), richiesta di citazioni e verifica esterna. Jailbreak
e **prompt injection** sono invece problemi **avversari**: sfruttano il fatto
che la gerarchia *system > user* è morbida e che il modello non distingue in
modo affidabile *istruzioni* da *dati*. La prompt injection in particolare
(testo non fidato che entra nel contesto e viene interpretato come comando) è
l'analogo dell'SQL injection, come ha mostrato in dettaglio la sezione su
come si attacca e si difende un modello di linguaggio; e nei sistemi agentici
(con accesso a strumenti, mail, file) il danno smette di essere un testo
sbagliato e diventa un'azione. Sopra tutto sta la
questione dell’**uso duale**: le stesse capacità che rendono un modello utile
per chimica, biologia o codice possono assistere chi vuole nuocere. È una
proprietà della capacità stessa, e ne fa una questione di controllo
dell'accesso più che di addestramento.

`````

## Valutare la sicurezza

Se non possiamo garantire che un modello sia sicuro, possiamo almeno *provare
a romperlo prima che lo faccia il mondo*. Qui la sicurezza dell'AI prende in
prestito il vocabolario della sicurezza informatica, e i due mestieri sono
quelli già incontrati attaccando un modello di linguaggio: cercare le falle a
mano, e poi mettere in una lista di prove quelle già trovate.

```{figure} ../figures/red-teaming.svg
:name: fig-red-teaming
:alt: "Ciclo chiuso in quattro stazioni disposte ad anello e percorse in senso orario: 1 policy (cosa il modello non deve fare), 2 attacco (jailbreak, injection, attacchi automatizzati), 3 scoperta (falle documentate e classificate), 4 patch (fine-tuning, filtri, system prompt), e dalla patch di nuovo alla policy. Al centro dell'anello la nota «si ricomincia: ogni patch invita un nuovo attacco»."
:width: 78%

Il red teaming è un anello, non un collaudo che si supera una volta. Il
tratto che dalla patch torna alla policy è la parte onesta del disegno, perché
ogni correzione cambia la superficie d'attacco invece di eliminarla.
```

L'anello di {numref}`fig-red-teaming` comincia dalla policy, ed è una scelta
tecnica prima che burocratica: senza aver scritto prima *cosa* il modello non
deve fare, un attacco riuscito non si distingue da una risposta insolita, e non
c'è modo di dire se la patch abbia funzionato.

`````{tab} Elementare

Prima di aprire un ponte al traffico non ci si limita a guardarlo: gli si
fanno passare sopra camion carichi, lo si sottopone a vibrazioni, si cerca
*apposta* il punto in cui potrebbe cedere. Con i modelli si fa lo stesso, in
due modi complementari, ed è la coppia già vista: una squadra che si comporta
da avversario e inventa le domande più insidiose (meglio scoprirlo in
laboratorio che sui giornali), e una lista di prove ripetibili con un voto
finale, che dice se la versione nuova è più o meno sicura della precedente. Il
**red-teaming** cerca la falla nuova; le **evals** controllano che le vecchie
non tornino.

Un esame superato, però, dice meno di quel che sembra. Con i modelli le
domande girano in anticipo, perché le prove finiscono nel materiale su cui
hanno studiato, e allora il voto sale senza che sia migliorato niente. Per
questo qualche prova si tiene da parte e non si pubblica. E un ponte che ha
passato le prove ha passato quelle prove: se al vento di traverso non ha
pensato nessuno, del vento di traverso non si sa ancora niente.

`````

`````{tab} Superiore

Il **red-teaming** è la ricerca adversariale di *failure*: manuale (esperti
che sondano capacità pericolose, jailbreak, fughe di dati) o
**automatizzato**, con un LLM istruito a generare attacchi contro un altro. Le
**evals** sono suite di benchmark riproducibili; si distinguono quelle di
*capacità* (cosa il modello sa fare) da quelle di *sicurezza* (tossicità,
rifiuto di richieste illecite, resistenza ai jailbreak, propensione alle
allucinazioni), spesso riassunte in metriche che si leggono in versi opposti:
l’*attack success rate*, la quota di attacchi andati a segno, che si vuole
**basso**, e il tasso di rifiuto appropriato, che si vuole **alto**. Due avvertenze di metodo, entrambe corollari di
Goodhart. Primo: un benchmark è un proxy, e ottimizzare *per* il benchmark
(magari perché finito nei dati di addestramento) gonfia il punteggio senza
migliorare la sicurezza reale, per questo contano i *test adversariali tenuti
nascosti*. Secondo: passare le evals dimostra l'assenza dei fallimenti
*cercati*, non la sicurezza in assoluto. L'assenza di prove non è prova
d'assenza, ed è il motivo per cui la valutazione resta un processo continuo
invece di un timbro una tantum. L'AI Act ne fa un obbligo per i modelli più
capaci.

`````

## Governance e regolamentazione

Gli strumenti tecnici non bastano da soli: servono regole condivise su *chi*
può fare *cosa*, e chi risponde quando qualcosa va storto. Qui l'Europa ha
fatto la prima mossa di portata mondiale.

```{figure} ../figures/ai-act-guida-pratica.svg
:name: fig-ai-act-rischio
:alt: "Piramide a quattro gradini con i livelli di rischio dell'AI Act. In cima, rischio inaccettabile: divieto totale, social scoring e manipolazione. Sotto, alto rischio: conformità completa, log, sorveglianza umana, marchio CE. Più giù, rischio limitato: solo trasparenza, «stai parlando con un'AI». Alla base, rischio minimo: nessun obbligo specifico, ed è dove sta la maggioranza dei prodotti."
:width: 80%

La piramide del rischio. La forma conta quanto i contenuti: gli obblighi
pesanti stanno in cima, dove i casi sono pochi, e la base larga (quasi tutto
ciò che si costruisce) non ha obblighi specifici.
```

La {numref}`fig-ai-act-rischio` si legge dal basso. La lettura
corrente («l'Europa regola l'AI») suggerisce un peso uniforme, mentre la
piramide dice il contrario: la regola morde in proporzione al danno possibile,
e per la maggior parte dei sistemi non morde affatto. Sapere in quale gradino
cade ciò che si sta costruendo è il primo esercizio di conformità, e spesso
anche l'ultimo.

E c'è una differenza di natura fra la cima e il gradino sotto, che la forma
della piramide non mostra. Sotto si autorizza a condizioni: il legislatore
ammette che di quel sistema esista una versione fatta bene. In cima no, e la
ragione che il legislatore ne dà è che di certi usi una versione fatta bene non
esiste. Un punteggio unico attaccato a ogni cittadino sposta comunque il potere
su chi tiene il registro, e chi si ritrova il voto basso non ha nemmeno un
posto in cui protestare: non è un problema di taratura, e nessuna
documentazione tecnica lo toglierebbe.

`````{tab} Elementare

Non regoliamo tutti i prodotti allo stesso modo: un giocattolo di plastica e
un farmaco seguono controlli diversissimi, perché diverso è il danno che
possono fare. L’**AI Act** europeo applica la stessa idea all'intelligenza
artificiale: guarda anzitutto al **rischio** di ogni suo impiego, più che alla
tecnologia in astratto, e lo dispone su una piramide a quattro gradini.

In cima, il rischio *inaccettabile*: gli usi semplicemente **vietati**, che
sono pochi. Uno riguarda la vita di chiunque vada a scuola, e conviene
cominciare da lì. **Riconoscere le emozioni sul luogo di lavoro e a scuola**: usare
l'intelligenza artificiale per dedurre dal volto o dalla voce di uno studente
se è attento, annoiato, nervoso, è vietato in Europa, con eccezioni strette
come i motivi medici. Se stai leggendo da un banco, quella legge esiste anche
per te.

Gli altri.

- Dare a ogni cittadino un **punteggio sociale**: un voto unico calcolato dal
  suo comportamento in un ambito, che poi decide cosa può fare in un altro,
  prendere un treno, affittare una casa, iscrivere un figlio da qualche parte.
- **Raccogliere facce da internet o dalle telecamere** senza cercare una
  persona in particolare, per costruire archivi che servono poi a riconoscere
  la gente dal viso. Riconoscere qualcuno da una caratteristica del corpo, il
  volto, la voce, le impronte, si dice **biometrico**.
- **Prevedere chi commetterà un reato** dal solo profilo di una persona, dove
  vive e che tratti ha, invece che da fatti concreti: è il parente stretto del
  software da cui il capitolo è partito.
- **Manipolare** le persone con tecniche che aggirano la loro consapevolezza, o
  **sfruttare le vulnerabilità** di chi è fragile per età, disabilità o
  condizione economica.
- **Fabbricare immagini intime di una persona riconoscibile senza il suo
  consenso**, e il materiale di abuso sessuale su minori. È la voce aggiunta
  nel 2026, e colpisce le applicazioni che «spogliano» una fotografia per quello
  che producono, non per come sono fatte. Il divieto è scritto ma non ancora in
  vigore: si applica dal dicembre 2026.
- Dedurre da un volto la **razza, le opinioni politiche, la religione o
  l'orientamento sessuale** di qualcuno.
- **Riconoscere i volti in tempo reale nei luoghi pubblici**, ma attenzione:
  questo divieto vale per le **forze dell'ordine**, non per chiunque. Fuori da
  quel caso il riconoscimento biometrico resta permesso, ma sorvegliato.

Sotto c'è l’*alto rischio*, ed è il gradino che riguarda le decisioni serie:
chi viene assunto, chi ottiene un prestito, un dispositivo medico, l'ammissione
a una scuola. Questi usi sono permessi, ma prima di andare sul mercato bisogna
avere una documentazione tecnica in ordine, tenere il registro di quello che il
sistema decide, garantire che una persona in carne e ossa possa intervenire e
farsi certificare da fuori. Sembrano adempimenti da ufficio, finché non tocca a
te sentirti dire di no da una macchina.

Più giù, il *rischio limitato*: basta la **trasparenza**, cioè avvisare le
persone («stai parlando con un'AI», «questo video è generato»). In fondo, il
*rischio minimo*: la stragrande maggioranza dei sistemi, senza obblighi
particolari.

Resta una domanda che a questo punto viene naturale: e i modelli come quelli
che usiamo tutti i giorni per farci scrivere un testo, in quale gradino stanno?
In nessuno, ed è il punto in cui la piramide non basta più. Un modello
buono-per-tutto non ha un impiego suo, ce l'hanno le cose che ci si
costruiscono sopra; per questo il regolamento gli dedica un capitolo a parte,
con obblighi che riguardano chi lo fabbrica invece di chi lo usa, e obblighi in
più per i pochi più grossi di tutti. Grossi come? Si guarda quanto calcolo è
servito a costruirli, e di quello che ci si fa non si guarda niente: su quel
gradino la legge smette di misurare l'uso e misura la potenza.

`````

`````{tab} Superiore

L’**AI Act** {cite}`euaiact2024`, entrato in vigore nell'agosto 2024, struttura
gli obblighi su quattro livelli di rischio. I numeri di articolo sono la sola
cosa che permette a chi legge di andare a
verificare, e sono anche la parte più stabile del testo. Stabile non vuol dire
immobile: il regolamento è già stato emendato una volta, nel luglio del 2026
(il regolamento (UE) 2026/1744, il *digital omnibus*), che ha allungato
l'elenco delle pratiche vietate e rinviato alcune scadenze. Le **date** di
applicazione, in particolare, sono scaglionate e si sono già spostate: quelle
vanno guardate sul testo consolidato, non su un libro.

- **inaccettabile** (pratiche vietate, art. 5(1)): tecniche manipolative o
  subliminali (a), sfruttamento delle vulnerabilità dovute a età, disabilità o
  condizione socioeconomica (b), generazione o manipolazione di immagini intime
  di una persona identificabile senza il suo consenso (ba) e di materiale
  pedopornografico (bb), le due voci inserite dal *digital omnibus* e
  applicabili dal 2 dicembre 2026, *social scoring* da parte di enti pubblici o
  privati (c), **polizia predittiva** basata unicamente sulla profilazione o
  sui tratti di personalità (d), scraping non mirato di volti da internet o da
  telecamere per costruire archivi di riconoscimento (e), riconoscimento delle
  emozioni sul luogo di lavoro e negli istituti di istruzione (f),
  categorizzazione biometrica per dedurre razza, opinioni politiche,
  appartenenza sindacale, convinzioni religiose o vita sessuale (g), e
  l'identificazione biometrica remota *in tempo reale* negli spazi accessibili
  al pubblico **a fini di attività di contrasto** (h), con tre eccezioni
  tassative. Quest'ultima qualificazione è quella che si perde più spesso nei
  riassunti, ed è più grande delle eccezioni: fuori dall'ambito delle forze
  dell'ordine il riconoscimento biometrico remoto non è fra le pratiche
  vietate, ricade semmai nell'alto rischio. Si noti anche la lettera (d), che
  chiude il cerchio con l'apertura del capitolo;
- **alto rischio** (art. 6 e allegato III, più i componenti di sicurezza di
  prodotti): credito, occupazione, istruzione, giustizia, infrastrutture
  critiche, migrazione. Permessi ma con sistema di gestione del rischio,
  governance dei dati, documentazione tecnica, tracciabilità, **sorveglianza
  umana**, accuratezza e robustezza, e valutazione di conformità prima
  dell'immissione sul mercato;
- **rischio limitato** (art. 50): obblighi di **trasparenza**, dichiarare
  l'interazione con un'AI, etichettare i contenuti sintetici e i deepfake;
- **rischio minimo**: nessun obbligo (la maggior parte dei sistemi).

Un capo a parte è dedicato ai **GPAI** (*general-purpose AI*), i modelli di uso
generale: per tutti valgono obblighi di documentazione tecnica, informazione
agli sviluppatori a valle e sintesi pubblica dei dati di addestramento (art.
53); per quelli con **rischio sistemico** si aggiungono valutazione del modello,
**adversarial testing**, cioè il red-teaming, mitigazione dei
rischi sistemici, cybersicurezza e segnalazione degli incidenti gravi (art. 55).
Le sanzioni per le pratiche vietate arrivano fino a 35 milioni di euro o al 7%
del fatturato mondiale annuo (art. 99(3)).

Fermiamoci un attimo su come si stabilisce quel «rischio sistemico»,
perché è il punto in cui il regolamento **cambia asse** e contraddice la
formula con cui si riassume di solito («non regola la tecnologia, regola
l'uso»). La presunzione scatta oltre una soglia di calcolo di addestramento di
$10^{25}$ FLOP (art. 51(2)), dove FLOP sta per *floating point operations*, il
conto totale delle operazioni aritmetiche fatte per addestrare il modello. È un
criterio di **capacità**, applicato al fornitore, prima e indipendentemente da
qualunque impiego a valle: della sfera dell'uso non c'è più niente. Che sia
stato necessario dipende dalla natura dell'oggetto, non da una scrittura
difettosa: un modello generalista gli usi non li sceglie, quindi un criterio
d'uso da solo non
aveva presa. E la soglia è un rattoppo dichiarato: il calcolo di addestramento
non misura la capacità, e infatti il regolamento prevede che la Commissione
possa modificarla, il che è un modo elegante di ammettere che nessuno la
considera la grandezza giusta.

Sull'altra sponda dell'Atlantico l'approccio federale è volontario: il **NIST AI
Risk Management Framework** (2023) propone quattro funzioni (*Govern, Map,
Measure, Manage*) senza forza di legge. Due filosofie regolatorie a confronto:
vincolante ed *ex ante* quella europea, volontaria e basata su standard quella
federale statunitense (a livello dei singoli Stati il quadro è più mosso e
cambia in fretta).

`````

Resta una domanda che questo capitolo ha rimandato a lungo, ed è la prima che
farebbe chiunque si trovi dall'altra parte: **e poi con chi ci si lamenta?**
La risposta onesta è che dipende, e che fino a ieri spesso non c'era nessuno.
Le persone scartate dal selezionatore automatico di Amazon, quello che apre la
sezione sull'equità e che penalizzava i curriculum delle donne, non lo hanno
mai saputo: non è stato detto loro che una macchina aveva letto il loro
curriculum, e non avevano modo di scoprirlo. È esattamente il vuoto che le
regole provano a riempire, ed è il motivo per cui gli obblighi che sembrano
burocratici (tenere i registri di cosa il sistema ha deciso, garantire che una
persona possa intervenire, dichiarare che una decisione è stata presa da un
sistema automatico) sono la parte che riguarda chi quelle decisioni le
subisce: senza traccia scritta e senza un umano responsabile, un reclamo non
ha nemmeno un posto dove essere depositato. Chi vuole tirare il filo trova la
parte tecnica della stessa domanda, poter dire *perché* il sistema ha deciso
così, nel {doc}`capitolo sull'interpretabilità </Interpretabilita/overview>`,
che viene appena prima.

Dietro le regole c'è poi un dibattito che va reso esplicito, perché divide
anche gli addetti ai lavori: è quello annunciato all'inizio del capitolo,
quando si è detto che qui ci saremmo occupati dei danni misurabili adesso. Da
un lato chi mette al centro i **danni presenti e documentati** (i pregiudizi,
le violazioni di privacy, gli esempi avversari di cui parla il resto di questo
capitolo) e teme che l'attenzione ai rischi lontani distolga risorse da
ingiustizie che colpiscono persone reali *oggi*. Dall'altro chi punta sui
**rischi catastrofici futuri** di sistemi molto più capaci di quelli attuali,
e sostiene che prevenirli richieda cominciare adesso. Non è una disputa che
si possa chiudere adesso; ma è onesto notare che non sono alternative: un
ponte va progettato sia contro le crepe di oggi sia contro il terremoto che
forse verrà, e le due cose competono per lo stesso budget di attenzione.

## L'onestà dovuta

Chiudiamo dove il capitolo intero insiste. L'allineamento è oggi un'area di
ricerca a pieno titolo, non un ritocco finale: abbiamo strumenti per
*orientare* il comportamento dei modelli (RLHF, DPO, Constitutional AI,
red-teaming, evals), non **garanzie** sul risultato. Sotto tutto corre una
tensione strutturale, che nessuna tecnica ha sciolto: le stesse capacità che
rendono un modello utile lo rendono difficile da controllare, e più un sistema
è potente, più il divario tra ciò che gli chiediamo e ciò che fa può costare.
Le regole (l'AI Act in testa) provano a comprare tempo e responsabilità mentre
la ricerca insegue. È un campo in movimento, senza soluzioni definitive: e
come per l'equità, la scelta di *quanto* controllo pretendere e *a quali*
valori allineare non è un teorema da dimostrare, ma una decisione che spetta a
noi.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- **Allineato** vuol dire che quello che il sistema fa combacia con quello che
  volevamo. Il guaio è che noi non gli diamo il desiderio, gli diamo un numero
  da far salire, e lui quel numero lo fa salire davvero: come il genio della
  lampada, che esaudisce le parole e non l'intenzione.
- Il numero che gli diamo è sempre un’**imitazione** di ciò che ci interessa,
  un *surrogato*. E spremerla **all'inizio funziona e poi peggiora le cose**:
  non è vero che ottimizzare di più sia sempre meglio.
- Come si orienta un modello: gli si mostrano tante coppie di risposte con
  scritto quale delle due è migliore, si addestra un giudice automatico a
  imitare quei giudizi, e poi si allena il modello a piacere al giudice,
  tenendolo però al guinzaglio perché non vada a cercare i punti in cui il
  giudice si sbaglia. C'è anche una versione in cui i confronti li scrive
  un'altra AI seguendo un elenco di princìpi: la persona non sparisce, cambia
  mestiere, da chi giudica caso per caso a chi scrive le regole.
- **Rischi**: le **allucinazioni** (il modello inventa in buona fede, perché
  «suonare vero» non è «essere vero») si combattono verificando a valle; gli
  **attacchi** si combattono difendendo un perimetro. E le stesse capacità che
  rendono un modello utile lo rendono utile anche a chi vuole nuocere, ed è la
  capacità stessa vista dall'altro lato.
- Si può attaccare il proprio sistema apposta per trovarne le falle, e si
  possono fare esami ripetibili per controllare che le falle vecchie non
  tornino. Ma **passare un esame non vuol dire essere sicuri**: vuol dire non
  aver fallito le prove che qualcuno ha pensato di fare.
- L’**AI Act** europeo regola in base a quanto un uso può ferire: pochissimi usi
  vietati del tutto (fra cui riconoscere le emozioni degli studenti a scuola),
  alcuni sorvegliati, alcuni con l'obbligo di dire «stai parlando con un'AI», e
  tutto il resto libero. Con i modelli buoni-per-tutto, che un impiego proprio
  non ce l'hanno, il criterio cambia: per i più grossi si guarda quanto calcolo
  è servito a costruirli, e di quello che ci si fa non si guarda niente.
- Nessuna soluzione definitiva: strumenti per orientare, non garanzie. **A quali
  valori** allineare un sistema è una domanda che tocca a noi, non a un
  teorema.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- **Allineamento**: far sì che il sistema persegua ciò che *intendiamo*, non la
  lettera dell'obiettivo. *Specification gaming* / **reward hacking** e legge di
  **Goodhart** (il proxy $\tilde{r}$ diverge dal vero $r^*$ proprio dove
  l'ottimizzatore cerca il massimo); si distinguono *outer* e *inner* alignment.
- La curva dell’**overottimizzazione** non è monotona: al crescere della
  pressione la qualità vera prima sale, poi ripiega e scende sotto il punto di
  partenza, mentre il punteggio surrogato continua a salire
  {cite}`gao2023scaling`. La penalità KL dell'RLHF è la difesa contro questo, e
  insieme l'ammissione che il reward model è un surrogato.
- Allineare gli LLM: **RLHF** {cite}`christiano2017deep`,
  {cite}`ouyang2022training` (reward model dalle preferenze + PPO sotto vincolo
  KL); **DPO** {cite}`rafailov2023direct` (stesso *ottimo*, sotto le ipotesi di
  Bradley-Terry e con la stessa policy di riferimento, senza reward model
  esplicito: non gli stessi modi di fallire); **Constitutional AI / RLAIF**
  {cite}`bai2022constitutional` (principi scritti e feedback dell'AI per ridurre
  il giudizio umano).
- **Rischi**: allucinazioni (errore intrinseco → verifica a valle) contro
  jailbreak e **prompt injection** (attacchi → difesa di perimetro); e l’**uso
  duale**, proprietà della capacità, non bug da correggere.
- **Valutare la sicurezza**: **red-teaming** (cercare le falle da avversari) ed
  **evals**/benchmark (esami ripetibili). Ma passare un test è un proxy: assenza
  di prove non è prova d'assenza.
- **Governance**: l’**AI Act** {cite}`euaiact2024` regola per *rischio*
  (inaccettabile/vietato art. 5, alto art. 6 e allegato III,
  limitato/trasparenza art. 50, minimo), con obblighi extra per i **GPAI**
  (artt. 53 e 55) a rischio sistemico, presunto oltre $10^{25}$ FLOP (art.
  51(2)): ed è il punto in cui il regolamento passa da un criterio d'uso a un
  criterio di **capacità**. Il **NIST AI RMF** è la controparte federale
  volontaria.
- Nessuna soluzione definitiva: strumenti per orientare, non garanzie. *A quali
  valori* allineare e *quanto* controllo pretendere sono scelte umane, non
  teoremi.
```

`````

Da qui in avanti non si aggiungono più strumenti. Quello che resta è
un'abitudine più che una garanzia, e consiste nel chiedere su quale numero un
sistema è stato premiato, che cosa quel numero lascia fuori e chi risponde
quando la risposta è sbagliata. Restano le Conclusioni, che non insegnano
niente di nuovo: servono a vedere che cosa, di tutto quello che hai letto,
rimane in mano quando si chiude il libro.
