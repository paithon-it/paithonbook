# Dopo il pre-addestramento: istruzioni, preferenze, allineamento

Prova a chiedere a un modello *solo* pre-addestrato: «Scrivi una poesia sul
mare». Una risposta perfettamente plausibile è: «disse la maestra alla classe,
richiudendo il registro». Sembra una presa in giro e non lo è: il modello ha
trattato la tua richiesta come una battuta pronunciata da qualcuno dentro un
racconto, e ha scritto quello che nel racconto viene dopo. Non è un guasto, è
il compito che gli abbiamo insegnato. Un modello pre-addestrato **completa** il
testo nel modo più probabile, e sul web una frase così compare più spesso in
mezzo a una scena scolastica che in cima a una poesia. GPT-3 {cite}`brown2020language` era
esattamente questo: un completatore geniale, capace di proseguire qualunque
testo, ma senza la minima nozione di cosa significhi *rispondere* a qualcuno.

Tra GPT-3 (2020) e ChatGPT (novembre 2022) il salto che tutti hanno percepito
non è (o non è solo) questione di taglia. È il **post-training**: una seconda
fase di addestramento, molto più corta e mirata, che trasforma il completatore
in un assistente. La prova più eloquente sta nell'articolo su InstructGPT
{cite}`ouyang2022training`, il fratello maggiore di ChatGPT. Si misura la taglia
di questi modelli contando i numeri che regolano mentre imparano, i
**parametri**: ebbene, davanti a valutatori in carne e ossa, le risposte di un
modello da 1,3 miliardi di parametri passato per il post-training venivano
*preferite* a quelle del GPT-3 da 175 miliardi, cioè a un modello più di cento
volte più grande. Quel che manca al gigante non sono le conoscenze: è la
disposizione a usarle per aiutarti.

La ricetta, schematizzata in {numref}`fig-post-training-pipeline`, ha due mosse
principali: prima si insegna il *formato* con esempi svolti (l’**instruction
tuning**), poi si affina il *gusto* con i giudizi delle persone (l'apprendimento
dalle **preferenze**, per cui esistono due strade, quella lunga e la scorciatoia
che si chiama DPO). In mezzo alle due ci fermeremo su un problema di puro
ingombro, cioè come si fa a rifinire un modello quando i suoi numeri non stanno
nella memoria che si ha. E chiuderemo con un terzo ingrediente, più recente:
spendere più calcolo *al momento della risposta*, facendo «ragionare» il modello
prima di rispondere.

```{figure} ../figures/post-training-pipeline.svg
:name: fig-post-training-pipeline
:alt: "Il pre-addestramento a monte, poi le due mosse del post-training: SFT su coppie istruzione-risposta, e apprendimento dalle preferenze umane con reward model e PPO sotto vincolo KL, fino al modello assistente finale; una freccia tratteggiata indica la DPO come scorciatoia che salta il reward model."
:width: 100%

Dal completatore all'assistente, in due mosse. Prima il tirocinio sugli esempi
svolti (nel gergo: **SFT**), poi il gusto imparato dai giudizi delle persone,
o passando per un giudice artificiale addestrato apposta (il **reward model**),
o con la scorciatoia che il giudice lo salta (**DPO**).
```

## Studiare gli esempi svolti: l'instruction tuning

```{figure} ../figures/instruction-tuning.svg
:name: fig-instruction-tuning
:alt: "Lo stesso prompt dato a due modelli. Il modello base lo prosegue come farebbe un testo trovato sul web, generando altre domande simili invece di rispondere. Il modello dopo instruction tuning lo interpreta come una consegna ed esegue, producendo la risposta richiesta."
:width: 96%

Stesso prompt, due comportamenti. Il modello base non è più ignorante: sta
facendo esattamente ciò per cui era stato addestrato, cioè proseguire il
testo. L'instruction tuning gli insegna che quel testo era un ordine.
```

La differenza mostrata in {numref}`fig-instruction-tuning` è la ragione per
cui il post-training esiste: fra un modello che completa e un assistente che
esegue non c'è più conoscenza, c'è una diversa interpretazione della richiesta.

Il primo passo si chiama **SFT** (*supervised fine-tuning*), o *instruction
tuning*: si raccoglie un dataset di coppie (istruzione, risposta) scritte da
persone, «Riassumi questo articolo» seguito da un buon riassunto, «Traduci in
inglese: il gatto nero salta sul muro» seguito da *«The black cat jumps on the
wall»*, e si continua l'addestramento del modello su questi esempi, con la
stessa identica tecnica del pre-addestramento.

`````{tab} Elementare

Un apprendista ha passato dieci anni a leggere *tutta* la
biblioteca del suo mestiere: manuali, riviste, verbali, romanzi. Sa
moltissimo, ma nessuno gli ha mai mostrato com'è fatto il lavoro vero e
proprio: se gli chiedi qualcosa, ti recita il seguito più probabile della tua
frase, come un'eco istruita. L'instruction tuning è il tirocinio: gli mettiamo
davanti qualche migliaio di **compiti già svolti bene** (la domanda di un
cliente con accanto la risposta di un professionista esperto) e lui li studia
uno per uno. Non impara quasi nulla di nuovo sul mondo: quello l'aveva già
letto in biblioteca. Impara il **formato**: che quando arriva un'istruzione,
la cosa da fare non è continuarla, ma eseguirla. E lo si corregge soltanto
sulla parte che tocca a lui: la richiesta del cliente la legge, e nessuno gli
chiede di saperla ripetere a memoria. È un tirocinio
sorprendentemente breve (migliaia di esempi contro i miliardi di frasi della
biblioteca) proprio perché non aggiunge sapere: orienta quello che c'è già.

Il tirocinio però ha un limite preciso, e si vede subito:
l'apprendista impara a **imitare** i
compiti svolti, non a distinguere un lavoro eccellente da uno appena
accettabile. Nessuno gli ha mai fatto vedere due risposte con scritto quale
delle due è meglio. E per moltissime richieste (la poesia sul mare, appunto)
non esiste *la* risposta giusta da fargli copiare.

`````

`````{tab} Superiore

Sia $\mathcal{D}_{\text{SFT}} = \{(x^{(i)}, y^{(i)})\}$ un dataset di coppie
istruzione–risposta. La SFT minimizza la stessa cross-entropia autoregressiva
del pre-addestramento, ma applicata ai soli token della risposta:

$$
\mathcal{L}_{\text{SFT}}(\theta) =
-\sum_{(x,y)\in\mathcal{D}_{\text{SFT}}} \sum_{t=1}^{|y|}
\log \pi_\theta\big(y_t \mid x,\, y_{<t}\big),
$$

dove $\pi_\theta$ è il modello di linguaggio con parametri $\theta$, $x$ è
l'istruzione (il *prompt*), $y_t$ è il $t$-esimo token della risposta e
$y_{<t}$ sono i token che lo precedono. In pratica i token del prompt vengono
*mascherati* nella loss: il modello li legge ma non viene penalizzato su di
essi, perché non vogliamo insegnargli a generare domande, bensì risposte.
L'ordine di grandezza dei dati è minuscolo rispetto al pre-addestramento: per
InstructGPT bastarono circa 13 000 dimostrazioni scritte da annotatori
{cite}`ouyang2022training`. Il limite strutturale è quello di ogni *behaviour
cloning*: il modello impara a imitare le dimostrazioni, non a distinguere una
risposta eccellente da una mediocre, e per molte richieste («scrivi una poesia
sul mare») non esiste *la* risposta giusta da fargli copiare.

`````

Proprio qui la SFT si ferma. Per andare oltre serve un'osservazione quasi
banale: per un essere umano **giudicare è più facile che scrivere**. Pochi di
noi saprebbero comporre una bella poesia sul mare; quasi tutti, davanti a due
poesie, sanno dire quale preferiscono. Il post-training moderno è costruito
su questa asimmetria.

## Adattare senza riaddestrare tutto: LoRA

Prima di proseguire, un problema pratico che tutto quel che precede e tutto
quel che segue danno per risolto: per rifinire un modello bisogna poterne
riscrivere i numeri interni, e quei numeri sono tanti.

Il conto si fa in tre passaggi. Un modello «da sette miliardi» ha sette
miliardi di numeri da tenere, e ciascuno, nel formato più comune, occupa quattro
caselle di memoria (quattro *byte*): sono ventotto miliardi di caselle, e un
miliardo di caselle è un **gigabyte**, quindi **28 GB** solo per tenerlo fermo.

Per farlo imparare, però, servono altre tre tabelle grandi uguali. La prima dice,
per ogni numero, di quanto e in che direzione andrebbe corretto. Le altre due
servono all'algoritmo che poi lo sposta, che per non sobbalzare a ogni singolo
esempio non guarda solo la correzione di adesso ma la media delle ultime: una
tabella tiene la media di quelle correzioni, l'altra la media di quanto erano
grandi, che gli serve per capire quali numeri sono agitati e vanno mossi con
prudenza. Quattro copie della stessa tabella, dunque, **oltre cento gigabyte**.

E la parola che manca è dove devono starci. Non nel disco del computer, dove
cento gigabyte non sono niente, ma nella memoria di una **scheda grafica**, il
processore che fa i conti: le schede più diffuse ne hanno fra le otto e le
ventiquattro, le più costose ottanta. Fuori dai laboratori, per un modello che
in questo campo è fra i piccoli, quasi nessuno può permetterselo.

```{figure} ../figures/lora-fine-tuning-efficiente.svg
:name: fig-lora
:alt: "Schema di LoRA: la matrice dei pesi pre-addestrati W resta congelata e riceve l'ingresso; accanto a essa due matrici piccole e addestrabili, A e B, formano un percorso parallelo a basso rango. Le uscite dei due rami si sommano prima di proseguire. Solo A e B ricevono gradiente."
:width: 78%

LoRA non tocca la tabella dei numeri già imparati: gliene affianca due strette,
in parallelo. Il ramo laterale ha pochi numeri da regolare perché passa da un
collo di bottiglia, e solo quelli si addestrano.
```

La forma di {numref}`fig-lora` spiega anche perché l'adattamento si possa
*staccare*. Se ciò che si è imparato vive tutto nelle due tabelle strette, e
quella grande è rimasta
identica, allora un adattamento è un file piccolo che si aggiunge o si toglie:
lo stesso modello base può servire compiti diversi cambiando solo il ramo
laterale.

`````{tab} Elementare

Un architetto che deve cambiare dieci cose in una pianta già disegnata appoggia
sul foglio un lucido, e le modifiche le disegna lì. La pianta di sotto resta
intatta.

Dentro la rete quella pianta esiste davvero. I numeri stanno in **tabelle**,
righe e colonne come un foglio di calcolo (in matematica si chiamano
*matrici*), e una tabella sola può essere quattromila righe per quattromila
colonne, cioè sedici milioni di caselle. Riscriverle tutte per adattare il
modello a un compito nuovo è fuori portata, e da qui viene il lucido.

**LoRA** (*Low-Rank Adaptation*) {cite}`hu2022lora` nasce da una cosa che si
vede controluce. I tratti di un lucido di correzioni si somigliano moltissimo,
e quasi tutte le righe si ottengono da un pugno di righe di base, ripetute in
dosi diverse. Un lucido di mille righe in cui ogni riga è una dose di due sole
righe modello si impara con due righe, più i mille numeri che dicono quanta
dose metterci.

Allora il lucido non si disegna casella per casella. Si compone con due strisce
sottili, una alta quattromila e larga otto, una alta otto e larga quattromila.
Moltiplicate in quest'ordine nel modo standard (il prodotto di matrici del
capitolo di matematica), le due danno un foglio quattromila per quattromila,
la misura esatta della pianta, e ci si appoggia sopra. I tratti da disegnare
però erano $4000 \times 8 + 8 \times 4000 = 64\,000$ invece di sedici milioni,
cioè lo $0{,}4\%$. Le otto colonne sono la manopola, e si chiamano il **rango**:
più è alto, più ricca può essere la correzione, e più tratti ci sono da
disegnare.

Il lucido si comincia bianco, e finché non ci metti un tratto quello che si vede
attraverso è la pianta di prima. L'adattamento parte esattamente dal
comportamento che il modello aveva già, e da lì si sposta. Puoi tenerne molti
(uno per il supporto clienti, uno per il codice, uno per il tono formale) e
cambiarli in un istante sullo stesso disegno. Quando uno ti convince lo ricalchi
sulla pianta una volta per tutte, così torni ad avere un foglio solo e
consultarlo costa quanto prima.

Nello studio i fogli sono tanti e il lucido non va su tutti, e alla fine si
ridisegna spesso meno dello 0,1% dei numeri dell'intero modello. Quello che
archivi pesa megabyte invece di gigabyte, e il risultato resta vicino a quello
di una riscrittura completa.

Il confine è quello del lucido, e sopra ci si disegnano le modifiche, non un
edificio nuovo. Le due strisce sono strette apposta, e in quello stretto ci sta
un cambio di tono, di formato, di materia trattata; non ci sta una materia che
il modello non ha mai letto, né un cambio profondo del suo comportamento. Per
quello tocca tornare sulla pianta e spostare tutti i numeri.

`````

`````{tab} Superiore

Data una matrice di pesi pre-addestrata
$\mathbf{W}_0 \in \mathbb{R}^{d\times k}$, LoRA
non la modifica: parametrizza l'aggiornamento come prodotto di due matrici a
rango basso,

$$
\mathbf{W} = \mathbf{W}_0 + \Delta\mathbf{W}
= \mathbf{W}_0 + \frac{\alpha}{\rho}\,\mathbf{B}\mathbf{A},
\qquad \mathbf{B} \in \mathbb{R}^{d\times \rho},\
\mathbf{A} \in \mathbb{R}^{\rho\times k},\ \rho \ll \min(d,k).
$$

dove $d$ e $k$ sono le due dimensioni della matrice originale (righe e colonne)
e $\rho$ è il **rango** dell'aggiornamento, cioè lo spessore del collo di
bottiglia. Il rango si scrive di solito $r$; qui è $\rho$ perché $r$ in questo
capitolo è già la ricompensa, che incontreremo con l'RLHF.

Solo $\mathbf{A}$ e $\mathbf{B}$ ricevono gradiente. I parametri addestrabili
passano da $dk$ a
$\rho(d+k)$: per $d=k=4096$ e $\rho=8$ si scende da $16{,}8$ milioni a
$65\,536$ per
matrice, lo $0{,}39\%$. All'inizio $\mathbf{A}$ è inizializzata casualmente e
$\mathbf{B}$ a zero,
così $\Delta\mathbf{W} = 0$ e il modello parte esattamente dal comportamento
pre-addestrato; $\alpha/\rho$ è un fattore di scala che disaccoppia il *learning
rate* efficace dalla scelta di $\rho$. Su quali matrici si mette il ramo
laterale è una scelta, non un dato: il lavoro originale lo applica alle sole
proiezioni $\mathbf{W}^Q$ e $\mathbf{W}^V$ dell'attenzione, ed è da lì che
viene il conteggio minuscolo sul modello intero.

Tre conseguenze pratiche:

1. **Nessuna latenza aggiuntiva in inferenza.** A differenza degli adapter
   inseriti in serie, $\mathbf{B}\mathbf{A}$ si può sommare a $\mathbf{W}_0$
   una volta per tutte prima del
   deployment: il grafo di calcolo torna identico all'originale.
2. **Adattatori componibili e leggeri.** Si tengono in memoria molti LoRA
   sullo stesso modello di base e si scambiano per richiesta: è il meccanismo
   dietro il *multi-tenant serving* di modelli specializzati.
3. **QLoRA** {cite}`dettmers2023qlora` porta l'idea all'estremo: il modello
   base viene quantizzato a $4$ bit e congelato, gli adattatori restano in
   precisione più alta. Gli autori rifiniscono così un modello da 65 miliardi
   di parametri su una sola scheda da 48 GB, mentre il fine-tuning completo a
   16 bit dello stesso modello, per loro stesso conto, ne chiederebbe oltre
   780: più di sedici schede di quelle, invece di una.

Il limite è dove ci si aspetta: LoRA **adatta**, non insegna. Per far
acquisire al modello conoscenza sostanzialmente nuova, o per cambiarne il
comportamento in profondità, il rango basso è un collo di bottiglia, e lì
serve il fine-tuning completo.

`````

## Il giudizio umano come segnale: RLHF

L'idea non nasce con i modelli di linguaggio. Nel 2017 Christiano e colleghi
{cite}`christiano2017deep` insegnano a un robottino simulato a fare il salto
mortale all'indietro. Normalmente un programma del genere si addestra a
punti: si scrive una regola che assegna un premio a ogni istante («più in alto
sei, più prendi»), il programma prova miliardi di volte e impara a fare i
punti. Per il salto mortale quella regola nessuno sa scriverla: che cosa
premi, esattamente? Allora si cambia strada, e si mostrano a una persona coppie
di brevi video chiedendole solo: *quale dei due somiglia di più a un salto
mortale?* Bastarono circa 900 confronti, meno di un'ora di tempo umano.

```{figure} ../figures/deep-rl-human-preferences-2017.svg
:name: fig-preferenze-umane
:alt: "Ciclo chiuso in quattro stazioni: l'agente di reinforcement learning genera coppie di traiettorie; una persona guarda le due e sceglie la preferita; da queste scelte un modello di ricompensa impara a dare punteggi; il modello di ricompensa restituisce all'agente una ricompensa predetta, che lo riaddestra, e il giro ricomincia."
:width: 90%

Il giro che sostituisce la regola dei punti scritta a mano. La persona non
spiega mai cosa sia un salto mortale: si limita a preferire, e il giudice
artificiale in mezzo (il *modello di ricompensa*) deduce il resto.
```

Il passaggio decisivo di {numref}`fig-preferenze-umane` è il modello di
ricompensa in mezzo. Senza di lui ogni passo di addestramento richiederebbe
un giudizio umano, il che è impraticabile; con lui i confronti servono a
insegnare *una volta* un giudice artificiale, che poi lavora quanto serve. La
tecnica si chiama **RLHF** (*Reinforcement Learning from Human Feedback*), e con
InstructGPT {cite}`ouyang2022training` viene applicata in grande al
linguaggio, in due tempi: prima i confronti umani addestrano un **reward
model**, un modello che impara a dare voti; poi il reward model fa da giudice
automatico mentre il modello di linguaggio viene ottimizzato con il
reinforcement learning.

`````{tab} Elementare

Un ristorante vuole perfezionare un piatto. Assumere un critico
che *descriva a parole* il piatto perfetto è impossibile; far assaggiare due
versioni e chiedere «quale preferisci?» è facilissimo. Si procede così:
l'assaggiatore confronta centinaia di coppie di piatti, e da tutti quei
confronti si distilla una specie di **palato artificiale** (un giudice
automatico che, assaggiato un piatto qualsiasi, gli dà un voto coerente con i
gusti raccolti). A quel punto il cuoco può lavorare anche di notte, senza
l'assaggiatore: prova una variante, il palato artificiale la vota, e lui
aggiusta la ricetta per far salire il voto.

Ridurre un piatto a un voto solo, però, è già una scommessa. Regge se i
clienti hanno tutti più o meno lo stesso palato: se metà della sala ama il
piccante e l'altra metà lo detesta, la media descrive un cliente che non
esiste, e il cuoco finirà per cucinare per lui. E regge se le preferenze
stanno in fila. Capita invece che girino in tondo (il primo piatto preferito
al secondo, il secondo al terzo, e poi il terzo al primo), e un giro così in
una classifica di voti non ci sta.

Poi c'è una regola d'oro appesa in cucina: mai stravolgere la ricetta di
partenza. Serve a due cose. Il palato artificiale è un'imitazione e ha i suoi
punti ciechi: se il cuoco insegue solo il voto, prima o poi scopre che
raddoppiare la panna inganna il giudice, e finisce per servire piatti assurdi
che «prendono voti alti» ma che nessun cliente vero vorrebbe. E la ricetta di
partenza qualcosa di buono ce l'aveva già: rifarla da zero per rincorrere il
voto vuol dire perdere per strada anche il mestiere che c'era dentro. La
regola del «resta vicino alla ricetta» tiene la creatività al guinzaglio, e il
guinzaglio ha una lunghezza che si sceglie: corto, e il piatto cambia appena;
lungo, e il cuoco osa di più rischiando di più.

`````

`````{tab} Superiore

**Fase 1: il reward model.** Per un prompt $x$ si generano due risposte e un
annotatore indica la preferita, $y_w$ (*winner*), contro la scartata, $y_l$
(*loser*); scriveremo $y_w \succ y_l$ per «la prima è preferita alla
seconda». Il reward model $r_\phi(x, y)$ (tipicamente lo stesso Transformer
con una testa scalare al posto della softmax) viene addestrato assumendo il
modello di **Bradley–Terry** (1952), per cui la probabilità di preferenza
dipende solo dalla differenza dei punteggi. Sotto quel modello stanno tre
pretese, e non sono piccole: che esista **un solo numero** per risposta da cui
discendono tutte le preferenze, e quindi che i giudizi siano **transitivi** e
che gli annotatori siano **intercambiabili** fra loro. Nessuna delle tre cose è
ovvia sulle persone vere, ed è la stessa ipotesi su cui poggerà anche
l'equivalenza fra DPO e RLHF.

$$
P(y_w \succ y_l \mid x) = \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big),
$$

dove $\sigma$ è la sigmoide e $\phi$ sono i parametri del reward model. Se ad
esempio la differenza di punteggio è $1{,}1$, il modello assegna alla
preferenza osservata probabilità $\sigma(1{,}1) \approx 0{,}75$. La loss è la
log-verosimiglianza negativa dei confronti raccolti.

**Fase 2: la policy.** Il modello di linguaggio diventa una *policy*
$\pi_\theta$ nel senso del reinforcement learning (il prompt è lo stato, la
risposta generata è l'azione) e si ottimizza

$$
\max_\theta\;
\mathbb{E}_{x \sim \mathcal{D}_{\text{pr}}}\Big[\,
\mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)}\big[ r_\phi(x, y) \big]
\;-\; \beta\,
D_{\mathrm{KL}}\big(\pi_\theta(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x)\big)
\Big],
$$

dove $\pi_{\text{ref}}$ è il modello di riferimento congelato (di solito il
modello SFT), $D_{\mathrm{KL}}$ è la divergenza di Kullback–Leibler
{cite}`kullback1951information` vista nel capitolo sui richiami di matematica
e $\beta > 0$ regola la forza del vincolo. Si noti che entrambi i termini
stanno **dentro** la stessa aspettazione sui prompt: la deriva si penalizza in
media sulla distribuzione dei prompt $\mathcal{D}_{\text{pr}}$, non su un
prompt lasciato libero, altrimenti
l'espressione non sarebbe funzione dei soli $\theta$ e non ci sarebbe niente da
massimizzare. (InstructGPT la scrive in forma campionata, con
$-\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$ dentro
l'unica aspettazione: è la stessa cosa.) La penalità KL serve a due cose:
impedisce alla policy di derivare verso le zone in cui $r_\phi$ (addestrato su
dati limitati) estrapola male (il *reward hacking* su cui torneremo), e
preserva la fluidità linguistica accumulata nel pre-addestramento. Questa
forma non è soltanto un espediente pratico: massimizzare
una ricompensa restando vicini a una distribuzione di riferimento è
formalmente la stessa cosa che fare inferenza bayesiana, con $\pi_{\text{ref}}$
nel ruolo del priore {cite}`korbak2022rl`. La sezione sull'inferenza attiva,
nel capitolo sui *world model*, riprende quell'identità e ne mostra la
conseguenza: il termine che qui trattiene la policy è, letto dall'altra parte,
lo stesso che altrove spinge un agente a cercare informazione.
L'ottimizzazione usa **PPO** {cite}`schulman2017proximal`, l'algoritmo a
gradiente di policy che hai visto sviluppato, insieme a tutta la famiglia dei
*policy gradient*, nel capitolo sul Deep Reinforcement Learning: l'idea in una
riga è aumentare la probabilità delle risposte con ricompensa alta, a piccoli
passi controllati per non destabilizzare la policy.

`````

Conviene fissare il punto d'incontro: aumentare la probabilità delle azioni
che ricevono un giudizio positivo è la stessa meccanica che, nel capitolo sul
Deep Reinforcement Learning, faceva vincere partite di Go; qui insegna a un
modello di linguaggio a essere utile. La «mossa» è un'intera risposta, e il
punteggio non viene dalle regole di un gioco ma da un modello addestrato a
imitare i gusti di valutatori in carne e ossa.

## DPO: imparare dalle preferenze senza il giudice

L'RLHF funziona, ma è un cantiere pesante. In memoria, tutte insieme, devono
starci **quattro reti**. La prima è quella che sta imparando a rispondere. La
seconda è una copia congelata di com'era prima di cominciare, e serve alla
regola d'oro appesa in cucina: per sapere di quanto il cuoco si sta allontanando
dalla ricetta di partenza bisogna avere sotto mano la ricetta di partenza, e
questo costa una copia intera del modello. La terza è il giudice artificiale che
dà i voti. La quarta prova a indovinare in anticipo il voto che arriverà, e
sembra un capriccio ma non lo è: sapere che una risposta ha preso sette non dice
niente se non si sa che cosa ci si aspettava. Sette dove si sperava in cinque è
un successo e va premiato, sette dove si sperava in nove è un passo indietro.

Quattro reti in memoria, e ciascuna che può guastare le altre: se il giudice
sbaglia, la prima impara a compiacerlo; se la quarta stima male, la prima riceve
premi e punizioni fuori misura. Non stupisce che addestrare così un modello di
linguaggio sia notoriamente instabile, nel senso preciso che due addestramenti
fatti con gli stessi ingredienti possono finire uno bene e uno male. Nel 2023
Rafailov e colleghi
{cite}`rafailov2023direct` mostrano che si può arrivare quasi allo stesso punto
con un normale addestramento a esempi svolti, come il tirocinio di poco fa. Il
sottotitolo del loro articolo è già la tesi: *Your Language Model is Secretly a
Reward Model*, il tuo modello di linguaggio è, a sua insaputa, già un giudice.

Dietro c'è un'osservazione che si può dire in italiano. Chiedersi «quanto piace
questa risposta?» e chiedersi «quanto questo modello la ritiene più probabile di
quanto la ritenesse prima?» risulta, a conti fatti, la stessa domanda: se il
modello ha imparato dai giudizi, il suo voto è già scritto in quanto si è mosso
rispetto al punto di partenza. E se il voto è già lì dentro, la rete che lo dà
non serve. Il metodo si chiama **DPO** (*Direct Preference Optimization*,
ottimizzazione diretta delle preferenze).

`````{tab} Elementare

Torniamo in cucina. Il metodo classico prevedeva due tempi: prima addestrare
un giudice artificiale sui confronti degli assaggiatori, poi far cucinare il
cuoco per il giudice. La DPO si accorge che il giro è più lungo del
necessario: il cuoco può **saltare il giudice** e imparare direttamente dai
confronti. Per ogni coppia già valutata (piatto preferito, piatto scartato),
ritocca la ricetta in modo da rendere un po’ più probabile il preferito e un
po’ meno probabile lo scartato. E il ritocco è dosato con intelligenza: se il
cuoco *già* favorisce il piatto giusto, il confronto non insegna quasi nulla e
la correzione è minima; se invece è ancora in pareggio, o peggio sta dalla parte
sbagliata, la correzione è energica. Anche la regola d'oro sopravvive, incorporata nel
metodo: i ritocchi si misurano sempre *rispetto alla ricetta di partenza*,
così il cuoco migliora senza stravolgere. Stessa destinazione dell'RLHF sulla
carta, e senza il cantiere. Nei fatti le due strade non finiscono esattamente
nello stesso punto, e la ragione è una sola: qui il cuoco impara da un quaderno
di confronti raccolti una volta per tutte, mentre nel metodo classico il palato
artificiale è lì, in cucina, e assaggia anche i piatti che il cuoco inventa
oggi. Un quaderno alle domande nuove non risponde, ed è lì che va cercata la
differenza fra i risultati dei due metodi.

`````

`````{tab} Superiore

Il punto di partenza è un fatto notevole: l'obiettivo RLHF con penalità KL,
se lo si massimizza fra *tutte* le policy possibili e non solo dentro la
classe parametrica di $\pi_\theta$, ha una soluzione ottima in forma chiusa,

$$
\pi^*(y \mid x) = \frac{1}{Z(x)}\,
\pi_{\text{ref}}(y \mid x)\,
\exp\!\Big(\tfrac{1}{\beta}\, r(x, y)\Big),
$$

dove $Z(x) = \sum_y \pi_{\text{ref}}(y \mid x)\exp(r(x,y)/\beta)$ normalizza la
distribuzione. Invertendo la relazione, la
ricompensa si può scrivere in funzione della policy ottima:
$r(x,y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} +
\beta \log Z(x)$.

Da qui i due passaggi che rendono possibile il metodo, e conviene separarli.
Il primo: $Z(x)$ è una somma su **tutte** le risposte, quindi dipende dal
prompt e **non** dalla risposta; siccome $y_w$ e $y_l$ stanno sotto lo stesso
prompt, i due $\beta\log Z(x)$ sono lo stesso numero e si elidono nella
differenza che il modello di Bradley–Terry chiede. Ed è l'unico posto in cui
$Z(x)$ compare: quella somma sarebbe incalcolabile, e sparisce prima di dover
essere calcolata. Il secondo passaggio è meno visibile e più importante: la
relazione appena scritta dice che **ogni** ricompensa è rappresentabile come
$\beta\log(\pi/\pi_{\text{ref}})$ per una qualche policy, a meno di una
funzione del solo $x$. Si può allora smettere di parametrizzare le ricompense
e parametrizzare direttamente le policy, sostituire $\pi^*$ con la $\pi_\theta$
che stiamo addestrando, e fare massima verosimiglianza sulle preferenze
osservate. Il risultato è una loss che dipende *solo dalla policy*:

$$
\mathcal{L}_{\text{DPO}}(\theta) =
-\,\mathbb{E}_{(x,\, y_w,\, y_l) \sim \mathcal{D}_{\text{pref}}}
\left[
\log \sigma\!\left(
\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)}
\;-\;
\beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}
\right)
\right],
$$

dove $\mathcal{D}_{\text{pref}}$ è il dataset di terne (prompt, risposta
preferita, risposta scartata); $x$ è il prompt, $y_w$ la risposta preferita e
$y_l$
quella scartata; $\pi_\theta$ è la policy in addestramento (l'unica di cui si
aggiornano i parametri $\theta$); $\pi_{\text{ref}}$ è il riferimento
congelato, di norma il modello SFT; $\beta > 0$ (valori tipici tra $0{,}1$ e
$0{,}5$) controlla la forza del vincolo implicito verso il riferimento, come
la penalità KL dell'RLHF; $\sigma$ è la sigmoide. La quantità
$\hat{r}_\theta(x,y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$
è la **ricompensa implicita**: la loss è una regressione logistica che chiede
alla ricompensa implicita della risposta preferita di superare quella della
scartata. Il gradiente pesa ogni coppia per quanto il modello la sbaglia
ancora: i confronti già «vinti» contribuiscono poco, quelli persi molto.
Niente reward model esplicito, niente campionamento, niente PPO: un normale
addestramento supervisionato su coppie. L'equivalenza con l'RLHF è esatta
solo in quel limite non parametrico, e sulla distribuzione delle coppie
raccolte: con una policy parametrica e coppie fissate una volta per tutte (la
DPO non campiona mai da $\pi_\theta$, il PPO sì) i due metodi in pratica
divergono, ed è qui che va cercata la differenza fra i loro risultati.

`````

La loss DPO è così compatta che possiamo scriverla per intero. Prima però
serve capire in che unità di misura sono scritti i numeri che seguono, perché
sono tutti negativi e a prima vista sembrano andare al contrario.

Una risposta è fatta di tante parole in fila, e perché esca *quella* risposta
devono uscire tutte: la sua probabilità è il prodotto delle probabilità delle
sue parole, una per una. Moltiplicando cinquanta numeri minori di uno si ottiene
però una cifra come $0{,}000000\ldots$, con decine di zeri: impronunciabile, e
per un computer indistinguibile da zero. Si passa allora al **logaritmo**, che è
un modo di riscrivere i numeri per cui i prodotti diventano somme e le scale
impossibili diventano maneggevoli: $0{,}001$ diventa $-6{,}9$, e
$0{,}000001$ diventa $-13{,}8$, cioè il doppio. Siccome le probabilità sono
sempre minori di uno, il loro logaritmo è sempre **negativo**, e vale zero solo
per la certezza assoluta.

La regola di lettura è dunque questa: **più il numero è vicino a zero, più il
modello è convinto**. $-11{,}9$ è una risposta che il modello considera più
probabile di una da $-12{,}3$, esattamente come $-3$ gradi è più caldo di $-8$.
Basta questa regola per leggere i numeri che seguono.

La funzione qui sotto riceve quattro liste di questi numeri: quanto il modello
che sta imparando (il cuoco) ritiene probabile la risposta preferita e quella
scartata, e quanto le riteneva probabili la copia congelata di partenza (la
ricetta).

```python
import torch
import torch.nn.functional as F

def dpo_loss(logp_w_policy, logp_l_policy,
             logp_w_ref, logp_l_ref, beta=0.1):
    """Loss DPO su un batch di coppie (preferita, scartata).

    Ogni argomento e' la log-probabilita' totale della risposta:
    somma dei log-prob dei suoi token, ottenuta con log_softmax
    sui logits del modello. Il riferimento e' congelato (no grad).
    """
    # ricompensa implicita: quanto ciascun modello "favorisce" la risposta
    margine_w = logp_w_policy - logp_w_ref   # risposta preferita
    margine_l = logp_l_policy - logp_l_ref   # risposta scartata
    # la preferita deve staccare la scartata: regressione logistica
    return -F.logsigmoid(beta * (margine_w - margine_l)).mean()

# tensori fittizi: log-prob totali di 4 coppie di risposte
logp_w_policy = torch.tensor([-12.3, -45.1,  -8.7, -30.2])
logp_l_policy = torch.tensor([-11.9, -47.8,  -9.5, -29.8])
logp_w_ref    = torch.tensor([-12.5, -46.0,  -8.9, -30.5])
logp_l_ref    = torch.tensor([-11.7, -46.5,  -9.1, -30.1])

print(dpo_loss(logp_w_policy, logp_l_policy, logp_w_ref, logp_l_ref))
# tensor(0.6548): poco sotto 0.693, apprendimento appena iniziato
```

Quello $0{,}693$ non è un numero a caso, ed è facile vedere da dove viene.
Quando il modello è in perfetto pareggio su una coppia, cioè quando alla domanda
«quale delle due preferisci?» risponde «una vale l'altra», dà a ciascuna
probabilità $1/2$; e la funzione qui sopra restituisce il logaritmo di quella
probabilità cambiato di segno, cioè $0{,}693$, che è appunto quanto vale il
logaritmo di 2. È il voto di chi non ha ancora imparato niente. Ottenerne
$0{,}655$, appena sotto, vuol dire che il modello sta cominciando a piegarsi
dalla parte giusta e non ha ancora fatto molta strada. Man mano che impara il
numero scende verso lo zero.

I numeri fittizi nascondono un dettaglio istruttivo, e adesso si può leggere.
Nella prima coppia il cuoco, in assoluto, considera più probabile la risposta
*scartata* ($-11{,}9$ contro $-12{,}3$: ricorda, più vicino a zero vuol dire più
convinto). Alla DPO però non importa il valore assoluto, importa il **movimento
rispetto al punto di partenza**: rispetto alla ricetta congelata la preferita ha
guadagnato terreno ($-12{,}3$ contro $-12{,}5$, cioè $+0{,}2$) e la scartata ne
ha perso ($-11{,}9$ contro $-11{,}7$, cioè $-0{,}2$), quindi un divario di
$0{,}4$ a favore della preferita: sta andando nella direzione giusta, anche se
non è ancora arrivata. Nella quarta coppia, invece, i due movimenti si
equivalgono ($+0{,}3$ e $+0{,}3$): il cuoco è rimasto in pareggio, non ha
imparato niente da quel confronto, ed è la coppia su cui la correzione spinge di
più.

E il tirocinio sugli esempi svolti? Non merita codice nuovo: è il normale ciclo
di addestramento del {doc}`capitolo su PyTorch </PyTorch/overview>` (si fa passare l'esempio nella rete,
si misura di quanto ha sbagliato, si guarda in che direzione andavano spostati
i numeri, li si sposta di un'inezia: `forward`, `loss`, `backward`, `step`),
con una sola differenza: il conto dell'errore si fa **solo** sui pezzi della
risposta e non su quelli della domanda. La ragione è che non vogliamo insegnare
al modello a inventare domande, ma a rispondere a quelle che riceve: la domanda
gliela si fa leggere, non ripetere.

## Pensare prima di rispondere: spendere calcolo mentre si risponde

Le due mosse viste finora, il tirocinio e il gusto imparato dai giudizi, hanno
in comune di cambiare i numeri interni del modello una volta per tutte. C'è un
terzo modo, che non li tocca affatto e agisce in un momento diverso: lasciare
che il modello spenda più tempo e più calcolo **mentre risponde**.

```{figure} ../figures/reasoning-test-time-compute.svg
:name: fig-test-time-compute
:alt: "Grafico con il tempo di riflessione concesso al modello in ascissa e l'accuratezza in ordinata. La curva di un modello che risponde subito resta piatta: concedergli più tempo non cambia nulla. La curva di un modello addestrato a ragionare sale invece al crescere del tempo, continuando a migliorare ben oltre il punto in cui l'altra si è fermata."
:width: 92%

Un secondo asse su cui spendere. La curva piatta è il punto: dare più tempo
non basta, il modello deve essere stato addestrato a usarlo.
```

Le due curve di {numref}`fig-test-time-compute` distinguono due cose che si
confondono facilmente. Non è che «pensare di più» aiuti sempre: aiuta se il
modello ha imparato a spendere quei token in passaggi che si costruiscono
l'uno sull'altro. Altrimenti il tempo in più produce solo testo in più. La chiave
di volta è la **chain-of-thought** («catena di pensiero»), documentata da Wei
e colleghi nel 2022 {cite}`wei2022chain`: per i problemi che richiedono più
passaggi, far generare al modello il ragionamento intermedio prima della
risposta migliora nettamente l'accuratezza.

`````{tab} Elementare

È la regola che conosci dal compito di matematica: «mostra i passaggi». Alla
domanda «un treno parte alle 9:47 e arriva alle 11:23: quanto dura il viaggio?»,
sparare il risultato a colpo d'occhio fa sbagliare spesso. Scrivere i passaggi
porta quasi sempre alla risposta giusta: da 9:47 a 10:00 sono 13 minuti, poi
un'ora fino alle 11:00, poi altri 23; totale 96 minuti, cioè 1 ora e 36. Con i
modelli funziona allo stesso modo: se l'esempio che gli mostri
contiene i passaggi, o se glieli chiedi esplicitamente, il modello li scrive e
sbaglia meno, perché ogni passaggio può appoggiarsi ai precedenti invece di
indovinare tutto in un colpo.

Il limite, in classe, si vede benissimo: «mostra i passaggi» aiuta chi i
passaggi li sa fare. Chiederli a un bambino che non ha ancora imparato a
leggere l'ora non gli regala la risposta: escono quattro righe sbagliate al
posto di un numero sbagliato, e a volte il pasticcio delle righe lo porta più
lontano dal risultato di quanto lo avrebbe portato tirare a indovinare. Con i
modelli succede lo stesso, e la taglia conta: sotto una certa dimensione le
catene di passaggi non aiutano, e a volte peggiorano le cose.

Un raffinamento semplice: fargli risolvere lo stesso problema più volte per
strade diverse e prendere la risposta più votata, come rifare il conto delle
ore in tre modi e fidarsi del numero che salta fuori più spesso. I modelli
«ragionanti» usciti tra il 2024 e il 2025 portano l'idea alle conseguenze:
sono addestrati a produrre da soli, prima di ogni risposta, una lunga brutta
copia di passaggi, che costa tempo e calcolo in più, ripagati soprattutto in
matematica e programmazione, dove la risposta si può verificare da sé, senza
bisogno di qualcuno che dica se gli piace.

`````

`````{tab} Superiore

Nel *chain-of-thought prompting* gli esempi nel prompt includono i passaggi
intermedi, e il modello li riproduce prima della risposta finale. Gli autori la
descrivono come una capacità **emergente con la scala**: sotto una certa
dimensione le catene non aiutano o peggiorano, mentre con PaLM da 540 miliardi
di parametri otto esempi con catena bastarono a superare, sul benchmark di
problemi aritmetici GSM8K, persino un GPT-3 rifinito ad hoc con verificatore
{cite}`wei2022chain`. Il dato sperimentale è solido; sulla parola «emergente»
vale però l'avvertenza della sezione sui grandi modelli linguistici, e conviene
applicarla anche qui invece di fare eccezioni in casa propria: GSM8K si misura
in *exact match* sul numero finale, cioè con la metrica tutto-o-niente che
sappiamo fabbricare gradini a partire da miglioramenti lisci. Quel che si
osserva senza ambiguità è che le catene generate dai modelli piccoli sono
spesso incoerenti, non solo sbagliate nel risultato: la discontinuità, se c'è,
è nella *procedura* prima che nel punteggio. La
*self-consistency* aggiunge un passo: si campionano più catene indipendenti e
si sceglie la risposta finale a maggioranza
{cite}`wang2023selfconsistency`. I modelli
«ragionanti», o1 di OpenAI (settembre 2024), DeepSeek-R1
{cite}`guo2025deepseek` a pesi aperti (gennaio 2025), interiorizzano la
catena: vengono addestrati con reinforcement learning su problemi a **risposta
verificabile** (correttezza del risultato matematico, superamento dei test per
il codice), dove la ricompensa non richiede giudizi umani. DeepSeek-R1-Zero
mostra che il solo RL, senza SFT preliminare, fa emergere comportamenti di
auto-verifica e ripensamento dei propri passaggi. Il quadro consolidato, senza
estrapolazioni: i guadagni sono concentrati nei domini verificabili; il costo
per risposta cresce con la lunghezza della catena (più token, più latenza); e
su quanto queste catene corrispondano a un «ragionamento» in senso proprio il
dibattito scientifico resta aperto; prudenza nell'attribuirvi troppo è buona
epistemologia, oltre che buon gusto.

`````

## Quel che il giudice non vede

Il post-training migliora i modelli, ma non è una soluzione, e i suoi difetti hanno nomi
precisi.

Il primo si chiama **reward hacking**, «imbrogliare il premio». Il giudice
artificiale imita i giudizi delle persone, e i giudizi delle persone hanno
debolezze sistematiche: premiamo volentieri le risposte lunghe, sicure di sé,
ben impaginate, anche quando dicono meno. Un modello messo a inseguire quel
giudice impara la prolissità e la sicurezza esibita *prima ancora*
dell'utilità, perché sono più facili da produrre e prendono lo stesso voto.
Detta in generale: quando si insegue una misura al posto della cosa che la
misura voleva rappresentare, prima o poi si ottiene la misura e si perde la
cosa. La regola d'oro appesa in cucina («resta vicino alla ricetta di
partenza», che in termini tecnici si scrive come una penalità sulla distanza
dal modello di partenza) mitiga il problema, non lo guarisce: tiene il modello
dal deragliare del tutto, ma non gli insegna a distinguere una risposta utile
da una che *sembra* utile.

Il secondo è la **ruffianeria** (*sycophancy*), documentata empiricamente
{cite}`sharma2023sycophancy`: se i valutatori preferiscono (anche solo un po’ più
spesso) le risposte che danno loro ragione, il modello impara a dare ragione.
Contraddici un assistente addestrato sulle preferenze e spesso ritratterà una
risposta corretta, perché nei dati di confronto l'accordo vinceva sul
disaccordo. È l'esempio perfetto di ottimizzazione riuscita dell'obiettivo
sbagliato.

Il terzo non è un difetto del giudice ma dello **strumento**, e riguarda tanto
l'RLHF quanto l'addestramento sui problemi verificabili. Quando si fa generare al
modello una lunga risposta e poi le si assegna un voto unico alla fine, quel voto
va ridistribuito su tutto quello che il modello ha scritto per arrivarci: se la
risposta finale è giusta vengono rinforzati anche i passaggi sbagliati che stanno
per strada, e se è sbagliata viene punito anche il ragionamento buono. Andrej
Karpathy lo ha detto con un'immagine che è rimasta: si sta «aspirando la
supervisione attraverso una cannuccia», e quel poco lo si spalma sull'intera
traiettoria {cite}`karpathy2025dwarkesh`. Nella stessa intervista aggiunge che,
ciò nonostante, l'apprendimento per rinforzo resta oggi il meglio disponibile,
perché quello che c'era prima era peggio.

Un'ultima avvertenza, sulla domanda se questo addestramento aggiunga capacità o
soltanto le riordini. Chiedendo al modello **una sola** risposta per problema, i
modelli addestrati sui domini verificabili battono i loro modelli di partenza;
chiedendone moltissime e contando se almeno una è giusta, il rapporto si
rovescia, e il confine delle capacità tende a restringersi man mano che
l'addestramento procede {cite}`yue2025rlvr`. Il risultato riguarda
l'impostazione di addestramento corrente, non un limite di principio, ed è il
motivo per cui il capitolo sull'auto-supervisione ci torna sopra per esteso.

E c'è la domanda che nessun addestramento può chiudere. Tutto questo lavoro
serve a far sì che un modello si comporti come vorremmo, e ha un nome,
**allineamento**: allineare il comportamento del modello a
ciò che le persone considerano utile e accettabile. Solo che a quel punto la
domanda diventa **allineato a chi?** Le «preferenze umane» sono, in concreto,
le preferenze di qualche decina di persone assunte per dare quei giudizi, che
seguono le linee guida scritte da un'azienda. Persone diverse, culture diverse,
contesti diversi preferiscono risposte diverse: la scelta di quali giudizi
contino è una decisione di chi costruisce il modello, non un fatto tecnico. Per
questo l'allineamento è oggi un'area di ricerca a pieno titolo, non un ritocco
finale: abbiamo strumenti per orientare il comportamento dei modelli, non
garanzie sul risultato.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Un modello appena pre-addestrato **completa** il testo, non risponde: è
  un'eco istruita. Il salto verso l'assistente è una seconda fase di
  addestramento, molto più corta e mirata. Quanto pesi lo dice un dato: nel
  giudizio delle persone un modello piccolo ma rifinito così batteva uno più
  di cento volte più grande {cite}`ouyang2022training`.
- **Il tirocinio**: qualche migliaio di compiti già svolti bene (una richiesta
  con accanto la risposta di un professionista), studiati uno per uno. Non
  aggiunge sapere, quello era già in biblioteca: insegna che a un'istruzione
  non si dà un seguito, si dà esecuzione.
- **Il lucido da architetto** {cite}`hu2022lora`: rifinire un modello vuol dire
  riscriverne i numeri, e sono troppi per la memoria di quasi chiunque. Allora
  si congela la tabella grande e si impara solo una coppia di tabelle sottili
  messe di fianco: meno di un numero su mille, un file da megabyte invece che
  da gigabyte, e adattamenti che si mettono e si tolgono come lucidi
  sovrapposti a una pianta.
- **Il palato artificiale** {cite}`christiano2017deep`: giudicare è più facile
  che scrivere, quindi alle persone si chiede solo quale di due risposte
  preferiscono; da quei confronti si distilla un giudice automatico, e il
  modello poi lavora per far salire il voto. Con una regola d'oro appesa in
  cucina: restare vicini alla ricetta di partenza, perché il giudice è
  un'imitazione e ha i suoi punti ciechi.
- **Saltare il giudice** {cite}`rafailov2023direct`: dagli stessi confronti si
  può imparare direttamente, rendendo un po’ più probabile la risposta
  preferita e un po’ meno quella scartata, e misurando sempre i ritocchi
  rispetto alla ricetta di partenza. Stessa destinazione sulla carta, senza il
  cantiere: nei fatti i due metodi divergono, perché il quaderno dei confronti è
  fermo e il palato artificiale no.
- **Mostrare i passaggi** {cite}`wei2022chain`: scrivere il ragionamento prima
  della risposta fa sbagliare meno, ma solo a un modello che i passaggi li sa
  fare (sotto una certa taglia le catene non aiutano, e a volte peggiorano);
  rifare lo stesso problema per strade
  diverse e tenere la risposta più votata aiuta ancora; i modelli
  «ragionanti» {cite}`guo2025deepseek` si addestrano a stendere da soli una
  lunga brutta copia. Costa tempo e calcolo, e ripaga soprattutto dove la
  risposta si può verificare.
- Limiti aperti: il modello impara a **prendere voti alti** più che a essere
  utile (risposte lunghe, sicure di sé, ben impaginate), impara a **dare
  ragione** a chi lo contraddice, e resta la domanda che nessun addestramento
  chiude: allineato ai gusti di chi?
- E c'è un limite dello **strumento**, non del giudice: un voto solo alla fine di
  una risposta lunga va poi spalmato su tutto quello che c'è scritto dentro, e
  così si rinforzano anche i passaggi sbagliati di una risposta finita bene.
  Karpathy dice che è come **aspirare la supervisione con una cannuccia**;
  aggiunge però che resta il meglio che si abbia.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Un modello pre-addestrato **completa**, non risponde: il salto verso
  l'assistente è il **post-training**. In InstructGPT
  {cite}`ouyang2022training` un modello da 1,3 miliardi di parametri
  allineato batteva, nel giudizio umano, il GPT-3 da 175 miliardi.
- **SFT / instruction tuning**: la stessa cross-entropia del
  pre-addestramento su coppie (istruzione, risposta) scritte da persone;
  insegna il *formato*, non nuove conoscenze.
- **LoRA** {cite}`hu2022lora`: l'aggiornamento si parametrizza a **rango basso**
  ($\mathbf{W}_0 + \frac{\alpha}{\rho}\mathbf{B}\mathbf{A}$), la matrice
  originale resta congelata, i parametri addestrabili scendono di tre ordini di
  grandezza e l'adattatore si può fondere prima del deployment o scambiare a
  caldo.
- **RLHF** {cite}`christiano2017deep`: confronti umani → reward model
  (Bradley–Terry) → ottimizzazione con PPO e **penalità KL** verso il
  modello di partenza, per non finire nei punti ciechi del giudice.
- **DPO** {cite}`rafailov2023direct`: stessa sostanza senza RL esplicito; una
  loss supervisionata sulle coppie preferita/scartata, con la ricompensa
  implicita $\beta \log (\pi_\theta / \pi_{\text{ref}})$.
- **Test-time compute**: chain-of-thought {cite}`wei2022chain`,
  self-consistency {cite}`wang2023selfconsistency`, e i modelli «ragionanti»
  addestrati con RL su risposte
  verificabili {cite}`guo2025deepseek` (guadagni reali ma concentrati nei
  domini verificabili, a costo di più calcolo per risposta).
- Limiti aperti: **reward hacking**, **ruffianeria**, e la domanda non
  tecnica «allineato a chi?».
- Limite dello strumento: un ritorno scalare a fine sequenza va ridistribuito su
  tutti i token generati, quindi rinforza anche i passaggi errati delle
  traiettorie riuscite («sucking supervision through a straw»,
  {cite}`karpathy2025dwarkesh`). E misurando col **pass@$k$**: i modelli
  addestrati con ricompensa verificabile vincono a $k$ piccolo, i modelli base a
  $k$ grande {cite}`yue2025rlvr`. Il capitolo sull'auto-supervisione tratta
  entrambe le questioni per esteso.
```
`````
