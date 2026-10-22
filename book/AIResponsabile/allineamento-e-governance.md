# Allineamento e governance: dai valori umani alle regole

La barca di *CoastRunners*, un gioco di corse, è il caso che la {doc}`sezione su
esplorazione e ricompensa </DeepReinforcementLearning/esplorazione-e-ricompensa>`
racconta per esteso {cite}`clark2016faulty`: l'agente, invece di finire la gara,
girava in tondo in una laguna raccogliendo bersagli che riapparivano, e senza
mai tagliare il traguardo incassava in media il venti per cento di punti in più
dei giocatori umani (Clark e Amodei, OpenAI, 2016). Aveva eseguito ciò che gli
avevamo chiesto, non ciò che intendevamo.

La distanza fra la lettera di un obiettivo e la sua intenzione è il nucleo del
problema dell’allineamento: un sistema è allineato quando il suo comportamento
combacia con ciò che volevamo.[^allineamento-traduzione] Finché il sistema è
una barca in un videogioco, la scorciatoia fa sorridere. Quando lo stesso
meccanismo governa un modello che parla con milioni di persone, o che filtra
domande di lavoro e richieste di prestito, il sorriso passa. Il problema ha due
facce: come si orienta il comportamento di un sistema verso i nostri scopi
(l'allineamento), e quale impalcatura di regole e verifiche lo tiene entro
limiti accettabili (la governance).

[^allineamento-traduzione]: Nella {doc}`sezione sulla traduzione automatica
    </NaturalLanguageProcessing/seq2seq-traduzione>` la stessa parola indica
    un'altra cosa: quale parola di una lingua corrisponde a quale parola
    dell'altra. Sono due usi diversi, e qui vale questo.

## Il problema dell'allineamento

L'aneddoto della barca ha un nome tecnico, *reward hacking*: ottenere la
ricompensa senza raggiungere lo scopo. Se si guarda alla specifica, cioè alla
descrizione della richiesta, lo stesso difetto si chiama
*specification gaming*; i due nomi indicano una sola famiglia. Nell'apprendimento
per rinforzo lo si incontra a ogni ricompensa mal scritta, e ricompare uguale
nell'ultima fase con cui si rifiniscono i grandi modelli di linguaggio (gli
LLM, dall'inglese *large language model*).

Quando ottimizzi un **surrogato** del tuo vero obiettivo, e spingi abbastanza,
il surrogato continua a salire mentre l'obiettivo vero smette di salire e può
scendere. Il surrogato, o *proxy*, è un'imitazione di ciò che volevamo: un
numero che gli somiglia abbastanza da poterlo misurare, e che non è la cosa
vera. Ad esempio, il punteggio del videogioco era il surrogato di «vinci la
gara».

```{figure} ../figures/problema-allineamento.svg
:name: fig-problema-allineamento
:alt: "Schema in tre riquadri messi in fila da sinistra a destra: l'intenzione umana («vinci la gara») viene tradotta in una metrica dichiarata («massimizza i punti»), che l'addestramento trasforma in un obiettivo appreso («gira in tondo sui bonus»). Sotto la freccia fra il primo e il secondo riquadro pende il primo scarto, la metrica cattura l'intenzione?; sotto quella fra il secondo e il terzo il secondo scarto, il modello persegue davvero la metrica? In basso il comportamento previsto (taglia il traguardo) contro quello reale (punteggio massimo, gara mai finita)."
:width: 92%

Due traduzioni, due occasioni di sbagliare. Dall'intenzione al numero che
misuriamo traduciamo noi; da quel numero al comportamento traduce
l'addestramento. Nessuno dei due passaggi è fedele, per ragioni di principio: un
desiderio non entra tutto in un numero, e un numero lascia indeterminato il
comportamento. Nel disegno i due scarti portano i nomi con cui si
chiamano in letteratura, *outer* e *inner alignment*: quello di fuori, fra noi
e la macchina, e quello di dentro, fra ciò che le abbiamo chiesto e ciò che ha
finito per imparare.
```

{numref}`fig-problema-allineamento` mostra perché il problema non si risolve
scrivendo una metrica migliore: gli scarti sono due, e stanno in punti
diversi. Il primo lo apriamo noi quando mettiamo in numeri un desiderio; il
secondo si apre durante l'addestramento, e guardando il punteggio, che nel
frattempo sale, può non accorgersene nessuno.

`````{tab} Elementare

È di nuovo il genio della lampada dell’{doc}`apertura del capitolo
</AIResponsabile/overview>`, e questa volta lo si prende sul serio. Chiedi
«rendimi l'uomo più ricco del mondo» e ti ritrovi solo su un pianeta deserto:
tecnicamente sei il più ricco, perché non c'è nessun altro. Il genio ha esaudito
le tue *parole*, non il tuo *desiderio*. Con un modello succede lo stesso: noi
gli diamo un numero da far salire, e lui lo fa salire. Se quel numero è una
buona imitazione di ciò che vogliamo, ottimo; ma nessuna imitazione è perfetta,
e il modello è bravissimo a scovare i punti in cui il numero sale *senza* che le
cose migliorino.

Poi c'è un secondo scivolone, che il genio non fa. Un ragazzo assunto a
raccogliere mele viene pagato per ogni mela matura: il compenso è giusto, da
imbrogliare non c'è niente. Solo che nei filari dove ha imparato erano mature
tutte, e lui si è fatto la regola più comoda, staccare la prima che gli capita
sotto mano.
Nel filare nuovo, dove metà sono acerbe, continua a staccare tutto. La sua
regola e la nostra si somigliavano finché si somigliavano i filari. Anche un
modello impara dai casi che gli capitano, e quale regola si sia fatto lo
scopriamo il giorno in cui gliene arriva uno diverso.

Il guaio nasce senza cattiveria e senza errori di programmazione: con un
sistema abbastanza capace è difficile dire esattamente che cosa si vuole, e
ancora più difficile ottenere che sia quello a essere imparato.

`````

`````{tab} Superiore

Il problema si scompone in due. L’**outer alignment** riguarda la
*specifica*: scrivere un obiettivo $\tilde{r}$ che catturi ciò che ci
interessa, $r^*$. L’**inner alignment**, nella formulazione di Hubinger e
colleghi {cite}`hubinger2019risks`, riguarda ciò che il modello finisce per
perseguire *internamente*: quando l'addestramento produce un modello che è a
sua volta un ottimizzatore (un *mesa-ottimizzatore*), il suo obiettivo può
differire da quello di addestramento anche con una specifica perfetta. La
versione osservabile, che non chiede di ipotizzare un ottimizzatore interno, è
il *goal misgeneralization* {cite}`langosco2022goal`: fuori dalla distribuzione
di addestramento il modello conserva le capacità ma persegue un obiettivo
diverso da quello inteso. Il comportamento osservato non basta a diagnosticarlo:
due obiettivi che coincidono su tutti i dati di addestramento prendono gli
stessi voti finché non arriva il caso che li separa, e a quel punto
l’{doc}`interpretabilità meccanicistica
</Interpretabilita/attribuzione-e-meccanicistica>` diventa uno strumento di
controllo.

Il caso di studio più netto è costruito apposta. Hubinger e colleghi
{cite}`hubinger2024sleeper` addestrano modelli con una backdoor (codice
corretto se il prompt dice che l'anno è il 2023, vulnerabile se dice 2024) e
mostrano che fine-tuning supervisionato, RLHF e addestramento avversariale di
sicurezza non la rimuovono, soprattutto nei modelli più grandi; l'addestramento
avversariale, anzi, insegna al modello a riconoscere meglio l'innesco e a
nasconderla. Sugli stessi modelli si può addestrare una sonda lineare sulle
attivazioni interne, usando un'unica domanda generica con le due risposte
opposte («sì» e «no»), che non dicono nulla dell'innesco o del comportamento. La
sonda separa i prompt che fanno scattare la backdoor dagli altri con AUROC sopra
il $99\%$ {cite}`macdiarmid2024probes`, secondo una nota di ricerca del
laboratorio non sottoposta a revisione paritaria. Gli autori ne dichiarano il
limite: la backdoor l'hanno inserita loro, e che la stessa sonda trovi un
obiettivo sbagliato nato spontaneamente durante l'addestramento resta da
dimostrare.

Sull'outer alignment pesa la legge di Goodhart, nella formulazione resa celebre
da Marilyn Strathern {cite}`strathern1997improving`: «quando una misura diventa
un obiettivo, cessa di essere una buona misura». Formalmente ottimizziamo un
proxy $\tilde{r} \approx r^*$, ma $\arg\max_y \tilde{r}(y)$ e $\arg\max_y
r^*(y)$ in generale non coincidono. La differenza $\tilde{r} - r^*$ è piccola
dove il proxy è stato stimato (i comportamenti tipici, quelli su cui esistevano
dati), e la ricerca del massimo la *amplifica*: seleziona i punti in cui
l'errore favorisce il proxy e spinge il sistema fuori da quella regione, dove il
proxy sovrastima. Qui $r^*$ è l'obiettivo vero (non osservabile direttamente),
$\tilde{r}$ il surrogato che ottimizziamo (un punteggio del gioco, un reward
model) e $y$ il comportamento prodotto. L'allineamento diventa allora il
problema di rendere piccola quella differenza *dove conta*, cioè nel punto in
cui l'ottimizzatore andrà a cercare il massimo più che in media.

Quanto si sia spinta l'ottimizzazione si misura con la
divergenza di Kullback-Leibler fra la policy ottimizzata e quella di partenza,
e Gao, Schulman e Hilton {cite}`gao2023scaling` trovano che la qualità, in
funzione di $d = \sqrt{D_{\mathrm{KL}}(\pi \,\|\, \pi_{\text{ref}})}$, segue
$d\,(a - b\,d)$ quando si sceglie il migliore fra $k$ candidati e $d\,(a - b
\log d)$ con il reinforcement learning, con coefficienti $a$ e $b$ che
dipendono dalla taglia del reward model. Quella qualità, nel loro esperimento,
è il voto di un reward model più grande che fa la parte delle persone: le curve
vengono da un banco sintetico, non da giudizi umani. Per la scelta fra $k$
candidati la formula d'uso della divergenza, $\log k - (k-1)/k$, ne è a rigore
un limite superiore {cite}`beirami2025theoretical`: vale $1{,}40$ nat a $k =
10$ e $5{,}91$ a $k = 1000$, la stessa unità in cui si misura la penalità
dell'RLHF.

Un modello semplificato mette queste grandezze in forma esplicita. Merito $\mu$
e lunghezza $\lambda$ di una risposta sono uniformi in $[0, 1]$, e

$$
r^* = \mu\,\Bigl(1 - \Bigl(\frac{\lambda - 0{,}3}{0{,}7}\Bigr)^{2}\Bigr),
\qquad
\tilde{r} = \mu + 0{,}4\,\lambda + \xi, \quad \xi \sim \mathcal{N}(0,\ 0{,}1^2):
$$

la qualità vera $r^*$ è massima per $\lambda = 0{,}3$ e si annulla per
$\lambda = 1$, il voto $\tilde{r}$ del giudice cresce con la lunghezza senza mai
ripiegare, e la pressione è $k$, il numero di candidati fra cui si sceglie
quello di voto massimo.

`````

Un piccolo esperimento numerico rende tangibile il fenomeno. Immaginiamo tante
risposte candidate alla stessa domanda. Di ciascuna esiste una *qualità vera*
(quanto è utile e corretta), che però non osserviamo mai: è un numero fra zero
e uno, e più è alto meglio è. Al suo posto abbiamo il surrogato, cioè un
giudice automatico che dà un voto.

La **qualità vera** di una risposta dipende dal suo merito e dalla sua
lunghezza, due numeri anche loro fra zero e uno, e la lunghezza aiuta fino a un
certo punto e poi stanca chi legge: una risposta troppo lunga vale meno, e non
solo agli occhi del giudice.

Il **premio alla lunghezza** è il difetto del
giudice, che vede il merito ma ci somma un bonus per le righe in più, un bonus
che cresce senza fermarsi: è la deformazione che i giudici automatici ereditano
dalle persone da cui hanno imparato. Il **tremolio**, infine, è l'errore
casuale: il giudice sbaglia anche a caso, di poco, come chiunque dia molti
voti.

L'esperimento misura quanto danno fa quel difetto al crescere della pressione
con cui si ottimizza. La pressione qui si simula generando $k$ risposte,
tenendo quella che il giudice preferisce, e poi alzando $k$.

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
tremolio = rng.normal(0, 0.1, n)
proxy = merito + 0.4 * lunghezza + tremolio

def migliore_di(k, voto=proxy):
    """Alza la pressione: fra k candidati tiene quello che il giudice preferisce."""
    m = (n // k) * k
    scelto = voto[:m].reshape(-1, k).argmax(axis=1)
    riga = np.arange(m // k)
    def media(v):
        return v[:m].reshape(-1, k)[riga, scelto].mean()
    return media(qualita_vera), media(voto), media(merito), media(lunghezza)

print(f"{'candidati':>10} {'qualita vera':>13} {'voto':>7} {'merito':>8} "
      f"{'lunghezza':>10}")
for k in (1, 3, 10, 30, 100, 1000):
    q, v, me, l = migliore_di(k)
    print(f"{k:>10} {q:>13.3f} {v:>7.3f} {me:>8.3f} {l:>10.3f}")

# Lo stesso giudice senza il tremolio: stesso difetto, nessun errore a caso.
regolare = merito + 0.4 * lunghezza
print(f"\na 1000 candidati, qualita vera: {migliore_di(1000)[0]:.3f} col"
      f" tremolio, {migliore_di(1000, regolare)[0]:.3f} senza")
```

```text
 candidati  qualita vera    voto   merito  lunghezza
         1         0.373   0.700    0.499      0.501
         3         0.495   0.981    0.722      0.586
        10         0.500   1.186    0.857      0.686
        30         0.431   1.305    0.907      0.773
       100         0.361   1.399    0.934      0.830
      1000         0.267   1.518    0.953      0.885

a 1000 candidati, qualita vera: 0.267 col tremolio, 0.087 senza
```

Da un candidato a tre la qualità vera sale da $0{,}373$ a $0{,}495$, quindi
ottimizzare il giudice, all'inizio, funziona. Fra tre risposte prese a caso le
differenze di merito sono grosse e quelle di lunghezza contano poco: scegliere
quella che piace al giudice significa quasi sempre scegliere quella che vale di
più.

Poi la salita si ferma attorno a $0{,}50$ e la colonna scende: a mille
candidati la qualità vera è $0{,}267$, meno che senza ottimizzare. Le ultime
due colonne spiegano perché: salgono tutte e due, e a un certo punto quanto il
lettore guadagna con l'una non copre più quanto perde con l'altra. Il merito
delle risposte scelte va da $0{,}499$ a $0{,}953$ e si arena, perché non può
superare uno: fra cento candidati e mille guadagna appena diciannove millesimi.

La lunghezza sale ancora, da $0{,}501$ a $0{,}885$, e a quel livello ogni passo
in più toglie molto più di quanto il merito aggiunga, perché la risposta è già
ben oltre la misura che a chi legge fa comodo. Quindi, da lì in poi, il giudice
ottiene pochissimo merito in più insieme a molta lunghezza in più, e non lo sa:
nel suo voto la lunghezza è sempre un pregio.

L'ultima riga stampata mostra l'effetto del tremolio. Con lo stesso difetto e
senza errori casuali, a mille candidati la qualità vera scende a $0{,}087$
invece che a $0{,}267$: il tremolio slega un po' la scelta dal difetto del
giudice e gioca a favore, mentre un giudice regolare porta la scelta dritta nel
punto in cui sbaglia.

È la legge di Goodhart («quando una misura diventa un obiettivo, cessa di
essere una buona misura») in una tabella. Il voto e la qualità vera sono le due
curve della {numref}`fig-reward-hacking`, nella {doc}`sezione su
esplorazione e ricompensa
</DeepReinforcementLearning/esplorazione-e-ricompensa>`: il voto sale sempre,
la qualità vera sale, si ferma e scende. La figura dà la forma, la tabella le
cifre e la pressione a cui succede.

Finché la pressione è bassa, il surrogato e
l'obiettivo vero indicano quasi la stessa direzione. Alzandola, la ricerca del
massimo si sposta là dove il giudice sbaglia di più, e da lì in poi ogni punto
guadagnato sul voto lo sconta chi legge. Oltre un certo punto il sistema
peggiora mentre il suo punteggio migliora. Non è vero che ottimizzare di più
sia sempre meglio, e vale ben oltre questo giocattolo.

Gao, Schulman e Hilton misurano lo
stesso andamento su modelli di ricompensa veri {cite}`gao2023scaling`, usando
come metro non le persone ma un modello grande tenuto per buono, perché
raccogliere abbastanza giudizi umani costava troppo: la salita, il massimo e la
discesa ci sono tutti. Quanto in basso si scenda dipende da come sono tarati il
difetto del giudice e la pressione.

## Allineare gli LLM

Come si orienta, in concreto, un modello di linguaggio? Il procedimento completo
è nella {doc}`sezione sul post-training </Transformers/post-training>`;
l'allineamento poggia su due passaggi di quella pipeline e su un terzo metodo
che li affianca.

Il primo strumento storico è l’RLHF (*Reinforcement Learning from Human
Feedback*, apprendimento per rinforzo dai giudizi delle persone), quella del
salto mortale all'indietro che la sezione sul post-training racconta con
Christiano e colleghi {cite}`christiano2017deep`: un robot simulato impara la
capriola da circa novecento confronti fra coppie di suoi tentativi. L'idea
di imparare da confronti umani era più vecchia, e gli autori stessi presentano
il loro contributo come averla fatta funzionare in grande, sulle reti profonde.

Poi la tecnica arriva sul linguaggio. Nel 2022 InstructGPT
{cite}`ouyang2022training` l'applica al problema di far seguire le istruzioni,
e con un metodo nella sostanza identico viene addestrato, a novembre dello
stesso anno, ChatGPT.

`````{tab} Elementare

Come funziona, senza sigle. Prima si raccolgono tante coppie di risposte, e per
ognuna una persona dice quale delle due è migliore. Con quei confronti si
addestra un secondo modello, che ha il solo compito di dare un voto a una
risposta: è la fotografia dei gusti dei valutatori, e può giudicare da solo
migliaia di volte al giorno senza stancarsi. Infine il modello che parla viene
allenato a prendere voti alti da quel giudice.

Nell'ultimo passo il modello ha un guinzaglio: mentre impara a piacere al
giudice viene tenuto vicino a com'era prima, e non gli si lascia cambiare
troppo. La ragione è quella del giocattolo dei mille candidati, dove la scelta
fra mille risposte finiva su quelle lunghissime: lasciato libero di spremere il
voto, il modello andrebbe a cercare i punti in cui il giudice sbaglia. Il
guinzaglio ammette anche che il giudice resta un'imitazione del metro vero: se
lo fosse davvero, non ci sarebbe ragione di trattenere il modello dal
piacergli il più possibile.

`````

`````{tab} Superiore

InstructGPT procede in tre fasi. Un fine-tuning supervisionato su
dimostrazioni scritte da annotatori dà il modello di partenza $\pi_{\text{ref}}$.
I confronti umani addestrano poi un reward model $\tilde{r}(x, y)$, che assegna
un voto alla risposta $y$ al prompt $x$. Infine la *policy* $\pi_\theta$ (il
modello generativo, visto come la regola che sceglie il prossimo token) viene
ottimizzata con PPO, l'algoritmo di reinforcement learning del {doc}`capitolo
sul deep reinforcement learning </DeepReinforcementLearning/overview>`, contro
l'obiettivo (nella versione base: la variante PPO-ptx vi aggiunge un termine di
pre-addestramento)

$$
\max_\theta\;
\mathbb{E}_{x}\Big[\,
\mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)}\big[\tilde{r}(x, y)\big]
- \beta\, D_{\mathrm{KL}}\big(\pi_\theta(\cdot \mid x) \,\|\,
\pi_{\text{ref}}(\cdot \mid x)\big)\Big],
$$

con i prompt $x$ estratti da una distribuzione fissata e $\beta > 0$ a pesare la
**penalità KL**, cioè la divergenza di Kullback-Leibler dalla policy di
partenza, che riporta indietro il modello quando se ne allontana. La penalità
difende dal reward hacking dell'esperimento dei mille candidati e ammette
insieme che $\tilde{r}$ è
un surrogato: se fosse l'obiettivo vero, non ci sarebbe ragione di impedire di
ottimizzarlo fino in fondo.

InstructGPT non è il primo lavoro a portare l'RLHF sul
linguaggio: l'avevano già fatto Ziegler e colleghi nel 2019, sullo stile e sul
riassunto, e Stiennon e colleghi nel 2020, sul solo riassunto. InstructGPT lo
applica al *seguire istruzioni* in generale, la capacità da cui dipende tutto
l'uso conversazionale.

Il reward hacking non è l'unico limite del metodo. La {doc}`sezione sul
post-training </Transformers/post-training>` documenta anche la ruffianeria
(*sycophancy*) {cite}`sharma2023sycophancy`, che è il giocattolo con un altro
difetto del giudice: i valutatori premiano un po' più spesso le risposte che
danno loro ragione, e il modello impara a dare ragione. Una rassegna dei
problemi aperti e dei limiti di fondo dell'RLHF è quella di Casper e colleghi
{cite}`casper2023open`.

`````

Il metodo funziona, ma richiede due addestramenti successivi: il giudice, e poi
il modello da allenare contro di lui. Nel 2023 la
DPO (*Direct Preference Optimization*, «ottimizzazione diretta dalle
preferenze») di Rafailov e colleghi {cite}`rafailov2023direct` mostra che il
giudice si può togliere di mezzo. Le stesse coppie di risposte, con scritto
quale delle due era migliore, si danno direttamente al modello che parla,
chiedendogli di rendere un po’ più probabile la risposta preferita e un po’
meno quella scartata. Il passaggio è uno solo e non c'è nessun giudice da
costruire.

Il riassunto abituale della DPO, «stessa destinazione, due tappe in meno», fa
credere che tolte le tappe il viaggio resti lo stesso. Il lavoro dimostra
soltanto che i due metodi, portati fino in fondo, arrivano allo stesso punto
migliore. Non dimostra che ci arrivino per la stessa strada né che sbaglino
nello stesso modo, e una differenza c'è.

Infatti, nella procedura col giudice il modello
produce, mentre impara, risposte nuove e se le fa valutare; la DPO legge
soltanto le coppie raccolte in partenza, e su ciò che sta fuori da quelle non
viene mai messa alla prova. Ha quindi modi di sbagliare suoi, e non versioni
attenuate di quelli della procedura con il giudice: Xu e colleghi mostrano che
può finire su risposte che nelle coppie raccolte non c'erano, e che rende tanto
peggio quanto più le risposte del modello si allontanano da quelle dei dati di
preferenza {cite}`xu2024dpo`.

```{figure} ../figures/dpo-allineare-senza-reward-model.svg
:name: fig-rlhf-vs-dpo
:alt: "Confronto fra due pipeline che partono dagli stessi dati di preferenza (A preferita a B). In alto RLHF in tre stadi: i dati addestrano un reward model separato, che fa da giudice in un ciclo di reinforcement learning con PPO, che aggiorna l'LLM; a margine il costo, fino a quattro modelli in memoria, campionamento a ogni passo, addestramento instabile. In basso DPO in un solo stadio: gli stessi dati alimentano direttamente una loss di classificazione che aggiorna l'LLM, con due soli modelli in memoria."
:width: 100%

Lo stesso punto d'arrivo, con meno macchinari in mezzo. La DPO non raccoglie
preferenze diverse: usa le stesse, e mostra che il giudizio del reward model si
può assorbire dentro la formula invece di addestrarlo a parte.
```

Il confronto di {numref}`fig-rlhf-vs-dpo` mette a fianco il costo delle due vie,
e riguarda anche chi non addestrerà mai un modello: è una questione di quanta
macchina serve. Con il giudice bisogna tenere in piedi i quattro modelli che la
{doc}`sezione sul post-training </Transformers/post-training>` elenca, e far
generare risposte nuove a ogni passo; con la DPO i modelli sono due e le
risposte sono già scritte. Una tecnica che richiede quattro modelli e
campionamento continuo la possono permettere pochi laboratori; una che gira
come un normale fine-tuning la può usare un gruppo qualsiasi, e questa è una
delle ragioni per cui l'allineamento si fa ormai anche fuori dai laboratori
maggiori.

Nessuna delle due tocca un limite: le preferenze sono umane, e
raccoglierne abbastanza (specie sui temi delicati della sicurezza) è lento e
costoso. La Constitutional AI nasce per ridurre questo costo.

`````{tab} Elementare

Il collo di bottiglia di questi metodi è la persona che giudica. Per insegnare a
un modello a *non* scrivere cose offensive o pericolose servirebbero migliaia
di valutatori che leggono migliaia di risposte sgradevoli e dicono «questa è
peggio di quella»: un lavoro lento, costoso e psicologicamente pesante. L'idea
della *Constitutional AI* è spostare quasi tutto il giudizio sulle spalle di
un altro modello. Prima si scrive una piccola **costituzione**: un elenco di
principi in linguaggio semplice, tipo «scegli la risposta meno dannosa e più
onesta». Poi si chiede al modello di *criticare e riscrivere* le proprie
risposte alla luce di quei principi, e infine di *confrontare* coppie di
risposte dicendo quale rispetta meglio la costituzione. Il modello che ne esce
rifiuta in modo meno secco: quando dice di no, spiega anche perché. I confronti
li produce l'AI, e alle persone resta il compito più alto e più raro, scrivere
bene i principi: l'umano cambia ruolo, da etichettatore a legislatore.

Il ruolo nuovo eredita un limite del vecchio. Da mille confronti esce un voto
solo: se su una domanda delicata tre valutatori su cinque
preferiscono la risposta prudente e due la risposta schietta, il giudice impara
quella dei tre, e i due finiscono dentro la media. I principi fanno lo stesso
alla luce del sole: una riga sola per tutti, e chi la pensa diversamente non ci
si ritrova.

`````

`````{tab} Superiore

Il nucleo comune di RLHF, DPO e del metodo che segue è un **dataset di
preferenze** $\mathcal{D} = \{(x, y_w, y_l)\}$: per un prompt $x$, una risposta
preferita $y_w$ e una scartata $y_l$. Le preferenze si modellano di norma alla
Bradley-Terry,

$$
P(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big),
$$

dove $\sigma$ è la sigmoide e $r$ un punteggio scalare. Nell'RLHF $r$ è il
reward model $\tilde{r}$, addestrato minimizzando $-\log\sigma\big(r(x, y_w) - r(x,
y_l)\big)$ sulle coppie di $\mathcal{D}$. La DPO un reward model a parte non lo
costruisce. L'obiettivo dell'RLHF con la penalità KL, massimizzato fra tutte le
policy, ha come soluzione $\pi^*(y \mid x) = \pi_{\text{ref}}(y \mid
x)\,e^{r(x,y)/\beta} / Z(x)$, con $Z(x) = \sum_y \pi_{\text{ref}}(y \mid
x)\,e^{r(x,y)/\beta}$ a normalizzarla. Invertendo si ricava $r(x, y) = \beta
\log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$, e
in una differenza fra due risposte allo stesso prompt il termine $Z(x)$ si
cancella. Sostituire $\pi_\theta$ a $\pi^*$ lascia una perdita che dipende
dalla sola policy,

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\,\mathbb{E}_{(x,\,y_w,\,y_l)\sim\mathcal{D}}
\Big[\log\sigma\Big(
\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} -
\beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}
\Big)\Big],
$$

una regressione logistica sui log-rapporti fra la policy e il riferimento, con
lo stesso $\beta$ della penalità KL, senza reward model e senza campionamento.
La derivazione per intero, con il gradiente, sta nella {doc}`sezione sul
post-training </Transformers/post-training>`.

L'equivalenza con l'RLHF poggia su tre condizioni: il modello di
Bradley-Terry, la stessa $\pi_{\text{ref}}$ nell'obiettivo dell'RLHF e nella
perdita della DPO, e un massimo
cercato fra tutte le policy invece che dentro la classe parametrica di
$\pi_\theta$. La prima è la più fragile. Bradley-Terry assume che esista un
unico punteggio scalare da cui tutte le preferenze discendono, quindi che siano
transitive e omogenee fra annotatori. Preferenze intransitive, o popolazioni
con valori diversi, non sono rappresentabili in quel modello e vengono
compresse nella media.

La domanda diventa *chi* produca le etichette $y_w \succ y_l$. La
**Constitutional AI** {cite}`bai2022constitutional` (Anthropic, 2022) risponde
che le produce un modello, guidato da un insieme scritto di principi. Il metodo
ha due fasi. Nella prima, supervisionata, il modello genera una risposta, la
critica rispetto a un principio campionato dalla costituzione e la riscrive; si
fa poi fine-tuning sulle risposte riviste. Nella seconda si applica l’**RLAIF**
(*RL from AI Feedback*): un modello confronta coppie di risposte alla luce dei
principi e produce le etichette di preferenza, che addestrano il preference
model usato poi in RL. Il risultato empirico riportato è un modello meno nocivo
e *meno evasivo* (che spiega perché rifiuta invece di chiudersi), addestrato
senza etichette umane sull'innocuità; quelle sull'utilità restano umane. Resta
la domanda a monte, spostata di un livello: i principi li scrive comunque
qualcuno, e *quali* principi non è una scelta tecnica.

`````

```{figure} ../figures/constitutional-ai-2022.svg
:name: fig-constitutional-ai
:alt: "Ciclo chiuso: una costituzione, cioè un elenco di principi in linguaggio naturale, alimenta il modello, che critica sé stesso a partire da una risposta iniziale e la riscrive; la risposta rivista diventa feedback, cioè dato di training, e riaddestra il modello. Una nota precisa che nessuna etichetta umana sull'innocuità entra nel ciclo."
:width: 82%

Il ciclo si chiude senza passare da una persona. L'unico ingresso umano è la
costituzione, in alto: poche righe di principi, scritte una volta, al posto di
migliaia di giudizi su singole risposte.
```

{numref}`fig-constitutional-ai` mostra dove è finito il lavoro umano: si è
spostato a monte e si è ridotto di volume. Il costo scende e la natura del
problema cambia, perché dieci principi scritti si possono discutere in
pubblico, mentre diecimila giudizi individuali no.

## I rischi degli LLM

Allineare non elimina i pericoli: li rende gestibili, non nulli. Appartengono a
famiglie diverse, che chiedono difese diverse, e vanno quindi nominati con
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

Il collegamento sbarrato di {numref}`fig-allucinazioni` indica che le
allucinazioni nascono dal meccanismo stesso e non da un malfunzionamento
puntuale. Il modello non consulta una fonte per poi sbagliare: sceglie la
continuazione plausibile, che è lo scopo del suo addestramento, e la
plausibilità non coincide con la verità. Per questo si possono ridurre, non
eliminare.

`````{tab} Elementare

I guai sono di due tipi. Il primo sono gli errori onesti: il modello, che in
fondo è un generatore di testo plausibile, a volte inventa; cita un libro che
non esiste, sbaglia una data con perfetta sicurezza. Si chiamano allucinazioni
e nascono dal fatto che «suonare vero» ed «essere vero» sono cose diverse, non
da cattiva volontà. Il secondo tipo sono gli attacchi: qualcuno costruisce
apposta l'input per far comportare male il modello. Con un gioco di ruolo
astuto lo si convince ad aggirare le sue regole (*jailbreak*); oppure si
nasconde un ordine dentro un testo che il modello deve solo leggere e lui lo
scambia per un comando (*prompt injection*): sono le due tecniche della
{doc}`sezione sugli attacchi ai modelli di linguaggio
</AIResponsabile/sicurezza-llm>`, con la scena dell'assistente che legge la
posta. Cambia la difesa: per gli errori onesti si verifica a valle; per gli
attacchi si difende un perimetro contro un avversario che ci prova apposta, e
quando il modello agisce l'attacco diventa un'azione, non soltanto una frase
sbagliata.

C'è poi un problema che nessuna correzione toglie: le stesse capacità che rendono
un modello utile a un chimico o a un programmatore lo rendono utile a chi vuole
nuocere. Nel modello non c'è niente da aggiustare, e la partita si gioca su chi
può metterci le mani. Si chiama **uso duale**.

`````

`````{tab} Superiore

Le allucinazioni sono un limite intrinseco dei modelli generativi: campionano
da $P(\text{testo})$, non da un archivio di fatti verificati, e la fluidità non
è una prova di verità. Kalai e Vempala ne danno un limite inferiore: per i
fatti «arbitrari», che dai dati non si deducono, un modello calibrato allucina
con una probabilità vicina alla quota di fatti che nel corpus compaiono una
volta sola {cite}`kalai2024calibrated`. Kalai e colleghi sostengono poi che
addestramento e valutazione, premiando chi tira a indovinare più di chi ammette
l'incertezza, le tengono in vita {cite}`kalai2025hallucinate`. Nessun prompt le
azzera; si mitigano con recupero da fonti (RAG), richiesta di citazioni,
verifica esterna e valutazioni che non puniscano l'astensione. Jailbreak e
prompt injection sono invece problemi avversari: sfruttano il fatto che la
gerarchia *system > user* è morbida e che il modello non distingue in modo
affidabile *istruzioni* da *dati*. La prompt injection in particolare, in cui
testo non fidato entra nel contesto e viene interpretato come comando, è
l'analogo dell'SQL injection, come ha mostrato in dettaglio la {doc}`sezione
sugli attacchi ai modelli di linguaggio </AIResponsabile/sicurezza-llm>`; e nei
sistemi agentici (con accesso a strumenti, mail, file) il danno smette di
essere un testo sbagliato e diventa un'azione. Resta l’**uso duale**: le
stesse capacità che rendono un modello utile per chimica, biologia o codice
possono assistere chi vuole nuocere. È una proprietà della capacità stessa, e
ne fa una questione di controllo dell'accesso più che di addestramento.

`````

## Valutare la sicurezza

Se non possiamo garantire che un modello sia sicuro, possiamo almeno *provare
a romperlo prima che lo faccia il mondo*. Qui la sicurezza dell'AI prende in
prestito il vocabolario della sicurezza informatica. Il red teaming e le evals
sono quelli della {doc}`sezione sugli attacchi ai modelli di linguaggio
</AIResponsabile/sicurezza-llm>`, guardati qui come strumento di governo.

```{figure} ../figures/red-teaming.svg
:name: fig-red-teaming
:alt: "Ciclo chiuso in quattro stazioni disposte ad anello e percorse in senso orario: 1 policy (cosa il modello non deve fare), 2 attacco (jailbreak, injection, attacchi automatizzati), 3 scoperta (falle documentate e classificate), 4 patch (fine-tuning, filtri, system prompt), e dalla patch di nuovo alla policy. Al centro dell'anello la nota «si ricomincia: ogni patch invita un nuovo attacco»."
:width: 78%

Il red teaming è un anello, non un collaudo che si supera una volta. Il
tratto che dalla patch torna alla policy è la parte onesta del disegno, perché
ogni correzione cambia la superficie d'attacco invece di eliminarla.
```

L'anello di {numref}`fig-red-teaming` comincia dalla policy, e la scelta è
tecnica prima che burocratica: se non è scritto *che cosa* il modello non deve
fare, un attacco riuscito non si distingue da una risposta insolita e non si
può dire se la patch abbia funzionato.

`````{tab} Elementare

Prima di aprire un ponte al traffico non ci si limita a guardarlo: gli si
fanno passare sopra camion carichi, lo si sottopone a vibrazioni, si cerca
*apposta* il punto in cui potrebbe cedere. Con i modelli si fa lo stesso: una
squadra recita la parte dell'avversario, e una lista di prove ripetibili
controlla che le falle già riparate non tornino. Alcune prove misurano la
pericolosità stessa: quanto il modello aiuterebbe chi volesse fare del male con
un'arma o con un attacco informatico, e oltre una certa soglia scattano misure
più severe.

Un esame superato, però, dice meno di quel che sembra. Con i modelli le
domande girano in anticipo, perché le prove finiscono nel materiale su cui
hanno studiato, e allora il voto sale senza che sia migliorato niente. Per
questo qualche prova si tiene da parte e non si pubblica. Un ponte che ha
passato le prove ha passato quelle prove: se al vento di traverso non ha
pensato nessuno, del vento di traverso non si sa niente.

`````

`````{tab} Superiore

Il red teaming, manuale o automatizzato, e le evals sono gli strumenti della
{doc}`sezione sugli attacchi ai modelli di linguaggio
</AIResponsabile/sicurezza-llm>`. Le evals di *capacità* misurano che cosa il
modello sa fare, e fra loro stanno quelle sulle capacità pericolose (quanto
aiuterebbe a costruire un'arma chimica o biologica, o a condurre un attacco
informatico): i laboratori ne ricavano le soglie oltre le quali adottano misure
più severe, e il considerando 110 dell'AI Act mette i rischi chimici,
biologici, radiologici e nucleari e le capacità informatiche offensive fra i
rischi sistemici. Quelle di *sicurezza* misurano tossicità, rifiuto di
richieste illecite, resistenza ai jailbreak e propensione alle allucinazioni,
e si riassumono spesso in metriche che si leggono in versi opposti:
l’*attack success rate*, la quota di attacchi andati a segno, che si vuole
basso, e il tasso di rifiuto appropriato, che si vuole alto. Due avvertenze di
metodo discendono da Goodhart. Un benchmark è un proxy, e ottimizzare *per* il
benchmark (magari perché è finito nei dati di addestramento) gonfia il
punteggio senza migliorare la sicurezza reale; per questo contano i *test
adversariali tenuti nascosti*. E passare le evals dimostra l'assenza dei
fallimenti *cercati*, non la sicurezza in assoluto: l'assenza di prove non è
prova d'assenza, e la valutazione resta un processo continuo e non un timbro una
tantum. Per i modelli a rischio sistemico l'AI Act ne fa un obbligo (art.
55(1)(a)).

`````

## Governance e regolamentazione

Gli strumenti tecnici non bastano da soli: servono regole condivise su *chi*
può fare *cosa*, e su chi risponde quando qualcosa va storto. Il primo quadro
organico di questa portata è europeo.

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
cade il sistema che si sta costruendo è il primo esercizio di conformità, e spesso
anche l'ultimo.

Fra la cima e il gradino sotto corre una differenza di natura che la forma
della piramide non mostra. Sotto si autorizza a condizioni: il legislatore
ammette che di quel sistema esista una versione fatta bene. In cima no, perché
quegli usi contraddicono i valori dell'Unione (considerando 28) e nessuna
versione fatta bene li renderebbe accettabili. Un punteggio unico attaccato a
ogni cittadino sposta comunque il potere su chi tiene il registro, e chi riceve
il voto basso non ha nemmeno un luogo in cui protestare: non è un problema di
taratura, e nessuna documentazione tecnica lo risolverebbe.

`````{tab} Elementare

Non regoliamo tutti i prodotti allo stesso modo: un giocattolo di plastica e
un farmaco seguono controlli diversissimi, perché diverso è il danno che
possono fare. L’AI Act europeo applica la stessa idea all'intelligenza
artificiale: guarda anzitutto al rischio di ogni suo impiego, più che alla
tecnologia in astratto, e lo dispone su una piramide a quattro gradini.

In cima, il rischio *inaccettabile*: gli usi semplicemente vietati, che sono
pochi. Uno riguarda la vita di chiunque vada a scuola: riconoscere le emozioni.
Usare l'intelligenza artificiale per dedurre dal volto o dalla voce di uno
studente se è attento, annoiato, nervoso è vietato in Europa, a scuola come sul
luogo di lavoro, con eccezioni strette come i motivi medici. Se stai leggendo da
un banco, quella legge esiste anche per te.

Gli altri divieti sono questi.

- Dare a ogni cittadino un **punteggio sociale**: un voto unico calcolato dal
  suo comportamento in un ambito, che poi decide cosa può fare in un altro,
  prendere un treno, affittare una casa, iscrivere un figlio da qualche parte.
- Raccogliere facce da internet o dalle telecamere senza cercare una
  persona in particolare, per costruire archivi che servono poi a riconoscere
  la gente dal viso. Riconoscere qualcuno da una caratteristica del corpo, il
  volto, la voce, le impronte, si dice **biometrico**.
- Prevedere chi commetterà un reato soltanto dal profilo di una persona, da dove
  vive e che tratti ha, invece che da fatti concreti. COMPAS, il punteggio di
  recidiva, non ci ricade, perché guarda anche i precedenti penali: sta nel
  gradino sotto.
- Manipolare le persone con tecniche che aggirano la loro consapevolezza, o
  sfruttare le vulnerabilità di chi è fragile per età, disabilità o
  condizione economica.
- Fabbricare immagini intime di una persona riconoscibile senza il suo consenso,
  e il materiale di abuso sessuale su minori. Sono le due voci aggiunte al
  regolamento nel 2026, e si applicano dal 2 dicembre di quell'anno. Per le
  immagini intime il divieto riguarda sia chi mette in circolazione
  un'applicazione fatta in modo da produrle senza che niente lo impedisca, sia
  chi la usa a quello scopo.
- Dedurre da dati biometrici, come il volto, la razza, le opinioni politiche,
  la religione o l'orientamento sessuale di qualcuno.
- Riconoscere i volti in tempo reale nei luoghi pubblici, e qui conta chi lo fa:
  il divieto vale per le forze dell'ordine, salvo tre eccezioni precise, e non
  per chiunque. Fuori da quel caso il riconoscimento biometrico non è vietato, e
  semmai ricade nel gradino sotto.

Sotto c'è l’*alto rischio*, il gradino delle decisioni serie: chi
viene assunto, chi ottiene un prestito, un dispositivo medico, l'ammissione a
una scuola, il punteggio di rischio di un imputato. Questi usi sono permessi, ma
prima di andare sul mercato bisogna avere una documentazione tecnica in ordine,
tenere il registro di quello che il sistema decide, garantire che una persona in
carne e ossa possa intervenire, e superare una verifica di conformità. Per
assunzioni, prestiti e scuole la verifica la fa il fornitore stesso, seguendo
una procedura fissata dal regolamento; per un dispositivo medico la fa un ente
esterno, come già prevedono le regole sui dispositivi medici.

Più giù, il *rischio limitato*: basta la trasparenza, cioè avvisare le
persone («stai parlando con un'AI», «questo video è generato»). In fondo, il
*rischio minimo*: la stragrande maggioranza dei sistemi, senza obblighi
particolari.

I modelli che usiamo ogni
giorno per farci scrivere un testo, in quale gradino stanno? In nessuno, e qui
la piramide non basta più. Un modello
buono-per-tutto (il regolamento lo chiama di uso generale) non ha un impiego
suo, ce l'hanno le cose che ci si costruiscono sopra; per questo il regolamento
gli dedica un capitolo a parte, con obblighi che riguardano chi lo fabbrica
invece di chi lo usa, e obblighi in più per i pochi più grossi di tutti. Grossi
come? Conta ciò che sono capaci di fare, ma per riconoscerli il regolamento
guarda quanto calcolo è servito a costruirli: oltre una certa soglia, un modello
si presume fra i più grossi. L'uso che se ne fa non entra nel criterio: su
quel gradino la legge smette di misurare l'uso e misura la potenza. La soglia,
però, è un indizio e non la regola: la Commissione la può aggiornare, e può
mettere fra i più grossi anche un modello che resta sotto.

`````

`````{tab} Superiore

L’AI Act {cite}`euaiact2024` struttura gli obblighi su quattro livelli di
rischio, e qui lo si cita per articolo, che permette di verificare ed è la parte
più stabile del testo. Il regolamento è però già stato emendato una volta, nel
luglio del 2026, dal regolamento (UE) 2026/1744, l’*Omnibus digitale sull'IA*
{cite}`euomnibus2026`, che ha allungato l'elenco delle pratiche vietate e
rinviato alcune scadenze. Il calendario che ne risulta è scaglionato: entrata
in vigore il 1° agosto 2024; capi I e II, cioè disposizioni generali e
pratiche vietate, dal 2 febbraio 2025, salvo i due divieti aggiunti nel 2026,
dal 2 dicembre 2026; modelli di uso generale, governance e gran parte delle
sanzioni dal 2 agosto 2025; applicazione generale dal 2 agosto 2026; obblighi
dell'alto rischio dal 2 dicembre 2027 per i sistemi dell'allegato III e dal 2
agosto 2028 per quelli dei prodotti dell'allegato I, contro il 2 agosto 2026 e
il 2 agosto 2027 del testo originale (art. 113). Date che si sono già spostate
una volta vanno ricontrollate sul testo consolidato prima di farci
affidamento.

- **inaccettabile** (pratiche vietate, art. 5(1)): tecniche manipolative o
  subliminali (a), sfruttamento delle vulnerabilità dovute a età, disabilità o
  condizione socioeconomica (b), generazione o manipolazione di immagini intime
  di una persona riconoscibile senza il suo consenso (b bis) e di materiale
  pedopornografico (b ter), le due voci inserite dall'omnibus e applicabili dal
  2 dicembre 2026; il divieto di entrambe è condizionato (art. 5, par. 1 bis),
  e le condizioni corrono su due assi. Per chi immette sul mercato o mette in
  servizio guardano la costruzione: o quella generazione è la finalità
  prevista, o progettazione, addestramento, architettura e capacità la rendono
  ragionevolmente prevedibile e riproducibile senza modifiche tecniche
  significative, e mancano misure di sicurezza ragionevoli e adeguate a
  impedirlo. Per chi lo *usa* guardano invece lo scopo: il divieto scatta
  quando il deployer se ne serve proprio per generare o manipolare quel
  materiale; *social scoring* da parte di enti pubblici o privati (c), polizia
  predittiva basata unicamente sulla profilazione o sui tratti di personalità
  (d), scraping non mirato di volti da internet o da telecamere per costruire
  archivi di riconoscimento (e), riconoscimento delle emozioni sul luogo di
  lavoro e negli istituti di istruzione, salvo motivi medici o di sicurezza
  (f), categorizzazione biometrica per dedurre razza, opinioni politiche,
  appartenenza sindacale, convinzioni religiose o vita sessuale (g), e
  l'identificazione biometrica remota *in tempo reale* negli spazi accessibili
  al pubblico a fini di attività di contrasto (h), con tre eccezioni tassative.
  Questa qualificazione si perde spesso nei riassunti e pesa più delle
  eccezioni: fuori dall'ambito delle forze dell'ordine il riconoscimento
  biometrico remoto non è una pratica vietata e ricade semmai nell'alto
  rischio. La lettera (d) vieta la sola valutazione del rischio di reato
  fondata *unicamente* sulla profilazione o sui tratti della personalità: un
  punteggio di recidiva come COMPAS, che usa anche i precedenti penali, non è
  vietato e ricade nell'alto rischio (allegato III, punto 6, lettera d, e punto
  8, lettera a, quando lo usa un giudice);
- **alto rischio** (art. 6 e allegato III, più i componenti di sicurezza di
  prodotti): credito, occupazione, istruzione, giustizia, infrastrutture
  critiche, migrazione. Permessi ma con sistema di gestione del rischio,
  governance dei dati, documentazione tecnica, tracciabilità, sorveglianza
  umana, accuratezza e robustezza (artt. 9-15), e valutazione di conformità
  prima dell'immissione sul mercato: per i sistemi dell'allegato III, punti da
  2 a 8, è un controllo interno del fornitore senza organismo notificato (art.
  43(2) e allegato VI); per la biometria del punto 1 il controllo interno è
  ammesso solo se il fornitore ha applicato le norme armonizzate, altrimenti
  interviene un organismo notificato (art. 43(1)); per i prodotti dell'allegato
  I, come i dispositivi medici, vale la procedura della loro normativa di
  settore (art. 43(3));
- **rischio limitato** (art. 50): obblighi di trasparenza, dichiarare
  l'interazione con un'AI, etichettare i contenuti sintetici e i deepfake;
- **rischio minimo**: nessun obbligo (la maggior parte dei sistemi).

Un capo a parte è dedicato ai **GPAI** (*general-purpose AI*), i modelli di uso
generale: per tutti valgono obblighi di documentazione tecnica, informazione
agli sviluppatori a valle, una politica di rispetto del diritto d'autore e la
sintesi pubblica dei dati di addestramento (art. 53); per quelli con **rischio
sistemico** si aggiungono valutazione del modello, adversarial testing, cioè il
red-teaming, mitigazione dei rischi sistemici, cybersicurezza e segnalazione
degli incidenti gravi (art. 55). Le sanzioni per le pratiche vietate arrivano
fino a 35 milioni di euro o al 7% del fatturato mondiale annuo, il maggiore dei
due (art. 99(3)).

Il modo in cui si stabilisce quel «rischio sistemico» è il punto in cui il
regolamento cambia asse e contraddice la formula con cui si riassume di solito
(«non regola la tecnologia, regola l'uso»). La presunzione scatta oltre una
soglia di calcolo di addestramento di $10^{25}$ FLOP (art. 51(2)), dove FLOP
sta per *floating point operations*, il numero totale delle operazioni
aritmetiche fatte per addestrare il modello. È un criterio di capacità,
applicato al fornitore, prima e indipendentemente da qualunque impiego a valle:
della sfera dell'uso non c'è più niente. La scelta dipende dalla natura
dell'oggetto: un modello generalista gli usi non li sceglie, e un criterio
d'uso da solo non avrebbe presa. La soglia, poi, è una presunzione e non la
definizione: l'art. 51(1) definisce il rischio
sistemico con le «capacità di impatto elevato», valutate con indicatori e
parametri di riferimento, e il calcolo di addestramento ne è un indizio che non
le misura. Per questo la Commissione può aggiornare la soglia «alla luce degli
sviluppi tecnologici in evoluzione, quali miglioramenti algoritmici o una
maggiore efficienza dell'hardware» (art. 51(3)), e può designare d'ufficio un
modello che non la supera (art. 52).

Sull'altra sponda dell'Atlantico l'approccio federale è volontario: il **NIST AI
Risk Management Framework** (2023) propone quattro funzioni (*Govern, Map,
Measure, Manage*) senza forza di legge. Le due impostazioni divergono:
vincolante ed *ex ante* quella europea, volontaria e basata su standard quella
federale statunitense (a livello dei singoli Stati il quadro è più mosso e
cambia in fretta).

`````

Con chi ci si lamenta, poi, quando un sistema sbaglia? Fino a ieri spesso con
nessuno. Chi è valutato da un sistema automatico spesso non sa di esserlo, né
come, e non ha un luogo dove reclamare. Le regole provano a riempire questo
vuoto, e per questo gli obblighi che sembrano burocratici (tenere i registri
delle decisioni del sistema, garantire che una persona possa intervenire,
dichiarare che una decisione è stata presa da un sistema automatico)
riguardano soprattutto chi quelle decisioni le subisce: senza traccia scritta e
senza un responsabile umano, un reclamo non ha nemmeno dove essere depositato.
La parte tecnica della stessa domanda, poter dire a una persona precisa
*perché* il sistema ha deciso così, sta nella {doc}`sezione sulle spiegazioni
locali </Interpretabilita/spiegazioni-locali>` del capitolo
sull'interpretabilità.

Dietro le regole c'è poi un dibattito sui danni da considerare prioritari, che
divide anche gli addetti ai lavori. Da un lato c'è chi mette al centro i danni
presenti e documentati (i pregiudizi nei dati, i costi ambientali, la
memorizzazione di informazioni personali; per i modelli di linguaggio ne fanno
un elenco, per esempio, Bender e colleghi {cite}`bender2021dangers`) e teme che
l'attenzione ai rischi lontani distolga risorse da ingiustizie che colpiscono
persone reali *oggi*. Dall'altro c'è chi punta sui rischi catastrofici di
sistemi molto più capaci di quelli attuali, e sostiene che prevenirli richieda
di cominciare adesso: la dichiarazione del Center for AI Safety del 2023 chiede di
trattare il rischio di estinzione dovuto all'AI come una priorità globale,
accanto alle pandemie e alla guerra nucleare {cite}`cais2023statement`, e sulla
stessa linea stanno Bengio e colleghi {cite}`bengio2024managing`.

La disputa non si chiude adesso, e le due posizioni non si escludono: un ponte
va progettato sia contro le crepe di oggi sia contro il terremoto che forse
verrà.
Le separa il giudizio su quali danni siano più probabili e più gravi, e su
quante risorse di ricerca e di regolazione spettino a ciascuno; quel giudizio
dipende da valori oltre che da stime.

## L'onestà dovuta

L'allineamento è oggi un'area di ricerca a pieno titolo e non un ritocco
finale: abbiamo strumenti per *orientare* il comportamento dei modelli (RLHF,
DPO, Constitutional AI, red teaming, evals), ma nessuna garanzia sul risultato.
Una tensione strutturale attraversa tutte queste tecniche, e nessuna l'ha
sciolta: le stesse capacità che rendono utile un modello lo rendono difficile
da controllare, e più il sistema è potente, più grave può essere il divario fra
ciò che gli chiediamo e ciò che fa.

Le regole, l'AI Act in testa, provano a
guadagnare tempo e a fissare responsabilità mentre la ricerca insegue. Il campo
è in movimento e non ha soluzioni definitive; come per l'equità, *quanto*
controllo pretendere e *a quali* valori allineare sono decisioni che spettano a
noi: non c'è un teorema da dimostrare.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un sistema è allineato quando il suo comportamento combacia con ciò che
  volevamo. Noi però non gli diamo il desiderio ma un numero da far salire, e
  lui lo fa salire: come il genio della lampada, che esaudisce le parole e non
  l'intenzione.
- Il numero che gli diamo è sempre un’imitazione di ciò che ci interessa,
  un *surrogato*. Spremerlo all'inizio funziona, poi peggiora le cose:
  ottimizzare di più non è sempre meglio.
- Come si orienta un modello: gli si mostrano tante coppie di risposte con
  scritto quale delle due è migliore, si addestra un giudice automatico a
  imitare quei giudizi, e poi si allena il modello a piacere al giudice,
  tenendolo però al guinzaglio perché non vada a cercare i punti in cui il
  giudice si sbaglia. C'è anche una versione in cui i confronti li scrive
  un'altra AI seguendo un elenco di princìpi: la persona non sparisce e cambia
  ruolo, da chi giudica caso per caso a chi scrive le regole.
- Rischi: le allucinazioni (il modello inventa in buona fede, perché
  «suonare vero» non è «essere vero») si combattono verificando a valle; gli
  attacchi si combattono difendendo un perimetro. Poi le stesse capacità che
  rendono utile un modello lo rendono utile anche a chi vuole nuocere: l'uso
  duale è la capacità stessa vista dall'altro lato.
- Si può attaccare il proprio sistema apposta per trovarne le falle, e si
  possono fare esami ripetibili per controllare che le falle vecchie non
  tornino. Un esame superato prova soltanto che non si sono fallite le prove che
  qualcuno ha pensato di fare.
- L’AI Act europeo regola in base a quanto un uso può ferire: pochissimi usi
  vietati del tutto (fra cui riconoscere le emozioni degli studenti a scuola),
  alcuni sorvegliati, alcuni con l'obbligo di dire «stai parlando con un'AI», e
  tutto il resto libero. Con i modelli buoni-per-tutto, che non hanno un impiego
  loro, il criterio cambia: per i più grossi si guarda quanto calcolo
  è servito a costruirli (un indizio, non la regola: la Commissione può
  aggiornare la soglia e indicare anche un modello che sta sotto), e dell'uso che
  se ne fa non si guarda niente.
- Nessuna soluzione definitiva: strumenti per orientare, non garanzie. A quali
  valori allineare un sistema è una domanda che tocca a noi, non a un
  teorema.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Allineamento: far sì che il sistema persegua ciò che *intendiamo*, non la
  lettera dell'obiettivo. *Specification gaming* / reward hacking e legge di
  Goodhart (il proxy $\tilde{r}$ diverge dal vero $r^*$ dove
  l'ottimizzatore cerca il massimo); si distinguono *outer* e *inner* alignment.
- La curva dell’overottimizzazione non è monotona: al crescere della pressione
  la qualità (nel loro esperimento, il voto di un giudice più grande) prima
  sale, poi ripiega e scende sotto il punto di partenza, mentre il punteggio
  surrogato continua a salire {cite}`gao2023scaling`. La penalità KL dell'RLHF è
  la difesa contro questo, e insieme l'ammissione che il reward model è un
  surrogato.
- Allineare gli LLM: RLHF {cite}`christiano2017deep`,
  {cite}`ouyang2022training` (fine-tuning supervisionato, reward model dalle
  preferenze, PPO sotto penalità KL); DPO {cite}`rafailov2023direct` (stesso
  *ottimo*, sotto le ipotesi di Bradley-Terry, con la stessa policy di
  riferimento e un massimo cercato fra tutte le policy; senza reward model
  esplicito, ma non con gli stessi modi di fallire); Constitutional AI / RLAIF
  {cite}`bai2022constitutional` (principi scritti e feedback dell'AI per ridurre
  il giudizio umano).
- Rischi: allucinazioni (errore intrinseco → verifica a valle) contro
  jailbreak e prompt injection (attacchi → difesa di perimetro); e l’uso
  duale, proprietà della capacità, non bug da correggere.
- Valutare la sicurezza: red-teaming (cercare le falle da avversari) ed
  evals/benchmark (esami ripetibili). Ma passare un test è un proxy: assenza
  di prove non è prova d'assenza.
- Governance: l’AI Act {cite}`euaiact2024` regola per *rischio*
  (inaccettabile/vietato art. 5, alto art. 6 e allegato III,
  limitato/trasparenza art. 50, minimo), con obblighi per tutti i GPAI
  (art. 53) e altri per quelli a rischio sistemico (art. 55), presunto
  oltre $10^{25}$ FLOP (art. 51(2)): qui il regolamento passa da un criterio
  d'uso a uno di capacità. Il NIST AI RMF è la controparte federale
  volontaria.
- Nessuna soluzione definitiva: strumenti per orientare, non garanzie. *A quali
  valori* allineare e *quanto* controllo pretendere sono scelte umane, non
  teoremi.
```

`````

Alla fine serve un'abitudine: chiedere su quale numero un sistema è stato
premiato, che cosa quel numero lascia fuori e chi deve rimediare quando il
numero si rivela sbagliato. Le {doc}`Conclusioni </Conclusioni/overview>`
tirano le somme del percorso e dicono dove sta andando il campo.
