# Agire per non essere sorpresi: l'inferenza attiva

Fino a qui il capitolo ha tenuto separate due cose. Da una parte un modello che
impara a prevedere come va il mondo; dall'altra una strategia che, servendosi di
quel modello, decide che cosa fare. Prima si costruisce il cinema interiore, poi
lo si usa per scegliere la mossa. È la divisione del lavoro dei Dreamer, ed è
anche quella della JEPA più un pianificatore.

Esiste un modo di guardare la faccenda in cui quella separazione non c'è, e i
due mestieri sono lo stesso mestiere. Non nasce nell'informatica ma nelle
neuroscienze teoriche, si chiama **inferenza attiva** e la sua trattazione
d'insieme è un libro di Thomas Parr, Giovanni Pezzulo e Karl Friston
{cite}`parr2022active`. Conviene dedicargli una sezione, non perché sia il
modo in cui oggi si costruiscono i sistemi (non lo è, e più avanti lo diciamo
senza giri di parole) ma perché risponde alla domanda del capitolo da
un'angolatura che le due tappe precedenti non hanno: **a che cosa serve, in
fondo, un modello del mondo a una cosa viva.**

## Un pesce deve restare nell'acqua

Il punto di partenza è biologico, quasi banale, e regge tutto il resto.

Un organismo, per continuare a esistere, deve mantenersi in un piccolo
sottoinsieme degli stati possibili. Un pesce, scrivono gli autori, deve stare
**nell'acqua**; una persona deve tenere temperatura corporea e battito dentro un
intervallo stretto, «altrimenti morirà, o più precisamente diventerà qualcos'altro,
per esempio un cadavere». Quell'intervallo è ciò che definisce quella cosa come
quella cosa, e non un obiettivo assegnato da qualcuno.

Da qui la mossa. Se una creatura si costruisce un modello di come vanno le cose,
allora «trovarsi fuori dall'acqua» è per lei un'osservazione **estremamente
improbabile**. E un'osservazione improbabile è, in senso tecnico, molto
**sorprendente**: la sorpresa di un esito è tanto maggiore quanto meno lo si
riteneva probabile, ed è la misura che i richiami di matematica definiscono
nella sezione sulla teoria dell'informazione. E allora restare vivi e restare
poco sorpresi diventano la stessa cosa, detta in due lingue diverse.

## Due modi di smettere di essere sorpresi

Qui arriva l'idea che dà il nome alla sezione.

`````{tab} Elementare

Ti sei fatto un'idea di come sarà la stanza in cui stai per entrare: luce
accesa, sedia al suo posto. Apri la porta ed è buio. La previsione e la realtà
non combaciano, e tu hai due modi di rimettere le cose a posto.

Il primo è cambiare idea: «ah, evidentemente qualcuno ha spento la luce». Non
hai toccato niente, hai aggiornato quello che credi. Questo si chiama
**percepire**.

Il secondo è cambiare la stanza: allunghi la mano e accendi la luce. Adesso il
mondo assomiglia a quello che ti aspettavi, e la tua idea di partenza torna ad
avere ragione. Questo si chiama **agire**.

Sono due strade diverse verso lo stesso risultato, cioè eliminare lo scarto fra
quello che ti aspettavi e quello che trovi. E la tesi dell'inferenza attiva è
che siano davvero **la stessa operazione**, fatta in due direzioni: percepire
piega le tue idee verso il mondo, agire piega il mondo verso le tue idee.

Quanto sia strana davvero quella stanza al buio, però, non lo sai: quello che
senti è il tuo sconcerto, e lo sconcerto è sempre almeno quanto la stranezza
vera, mai meno. Trovata la spiegazione giusta lo sconcerto cala, e si ferma
contro quel fondo di stranezza che la scena ha comunque. Quel che resta in mezzo
è quanto la tua spiegazione dista dalla migliore che il buio consente.

C'è un terzo modo: **imparare**. Se le sorprese si ripetono, non basta più
aggiustare l'idea di oggi, conviene cambiare il modello che quelle idee le
produce. Gli autori lo dicono così: imparare «non è fondamentalmente diverso dal
percepire, opera semplicemente su una scala di tempo più lenta».

Detto altrimenti: qui non esiste una fase di addestramento separata dall'uso.
C'è una cosa sola che va avanti sempre, a tre velocità diverse.

`````

`````{tab} Superiore

La grandezza minimizzata è l’**energia libera variazionale**. Dato un modello
generativo $P(o, s)$ che lega osservazioni $o$ e stati nascosti $s$, e una
distribuzione approssimata $Q(s)$ sugli stati, si definisce

$$
F \;=\; \mathbb{E}_{Q(s)}\big[\ln Q(s) - \ln P(o, s)\big]
\;=\; \underbrace{D_{\mathrm{KL}}\big[Q(s) \,\|\, P(s \mid o)\big]}_{\ge\, 0}
\;-\; \ln P(o).
$$

La seconda forma è quella che conta. Il termine $-\ln P(o)$ è la **sorpresa**,
cioè l'evidenza logaritmica negativa del modello; la divergenza è non negativa,
quindi $F$ è un **limite superiore** sulla sorpresa, e il divario è esattamente
quanto $Q$ si discosta dalla distribuzione a posteriori che si otterrebbe
facendo l'inferenza esatta. Minimizzare $F$ rispetto a $Q$ stringe il limite ed
è l'inferenza approssimata di sempre: è lo stesso limite variazionale che il
capitolo sui modelli di diffusione deriva sotto il nome di **ELBO**, cambiato di
segno. Il punto nuovo è che $F$ dipende **anche** da $o$, e le osservazioni un
agente se le può andare a prendere.

Da qui i tre modi, che sono tre argomenti diversi rispetto a cui si minimizza la
stessa quantità:

- rispetto a $Q(s)$: **percezione**, cioè aggiornamento delle credenze a
  osservazioni date;
- rispetto alle **azioni**, che cambiano quali $o$ arriveranno: **azione**;
- rispetto ai **parametri** del modello generativo: **apprendimento**, che gli
  autori descrivono come non fondamentalmente diverso dalla percezione, «opera
  semplicemente su una scala di tempo più lenta».

Sull'azione serve una precisazione. Scegliere un gesto istante per istante
minimizza $F$; scegliere fra **politiche**, cioè fra corsi d'azione estesi nel
tempo, richiede una seconda grandezza, l’**energia libera attesa** $G(\pi)$,
dove «attesa» sta per il fatto che le osservazioni future non ci sono ancora e
vanno messe in conto per come ce le si aspetta. Gli autori insistono che le due
sono «matematicamente collegate ma con ruoli distinti e complementari»: $F$
resta la quantità minimizzata nel tempo, e $G$ entra dentro il modello
generativo come **priore sulle politiche** (grosso modo: una politica è
probabile a priori nella misura in cui promette $F$ bassa in futuro)
{cite}`parr2022active`.

Minimizzare la sorpresa equivale infine a **massimizzare l'evidenza** del
modello: sono la stessa espressione col segno cambiato. Un agente che si
comporta bene, in questo linguaggio, è un agente che raccoglie prove a favore di
sé stesso.

`````

## L'obiezione che viene subito, e la sua risposta

Se il fine è non essere sorpresi, la strategia ottima sembra ovvia e assurda:
mettersi in una stanza vuota e buia, immobile, dove non succede mai niente. È
l'obiezione che chiunque solleva alla prima lettura, ed è utile perché la
risposta chiarisce il meccanismo meglio di qualunque formula.

`````{tab} Elementare

La risposta è che la sorpresa non si misura rispetto a quello che capita, ma
rispetto a quello che sei.

Il modello di una creatura non è una fotocopia del mondo: chi avesse in testa
solo una fotocopia descriverebbe la corrente e si lascerebbe portare. Il modello
si porta dentro anche le condizioni in cui quella creatura deve trovarsi per
continuare a esistere. Nessuno ha scritto al pesce, su un foglio a parte, un
premio per l'acqua: il pesce si aspetta l'acqua, e se la aspetta un po’ più di
quanto i fatti gli garantiscano. Da quel po’ di ottimismo viene la nuotata.

Per un pesce, «essere all'asciutto» è la cosa più sorprendente che possa
capitargli, e nessuna quantità di immobilità gliela risparmia: al contrario, se
resta immobile fuori dall'acqua la sorpresa cresce fino alla fine.

Quindi la stanza buia non funziona. Restare fermi in un posto dove non si mangia
e non si beve porta dritti agli stati più sorprendenti che esistano per un
corpo. Per non essere sorpreso, un organismo è **costretto** a muoversi, a
cercare, e persino a esplorare, perché lo stato che vuole occupare non è quello
in cui si trova adesso.

Gli autori mostrano che questo scala molto oltre i riflessi. Restare alla
temperatura giusta, scrivono, è **sudare** (che è fisiologia), ma anche
comprarsi da bere (che è psicologia) e mettere l'aria condizionata in una città
intera (che è una faccenda collettiva). E soprattutto è **cercare l'ombra prima
di surriscaldarsi**, che è l'esempio a cui tengono di più, perché lì il
correttivo arriva *prima* del guaio. Lo stesso imperativo, insomma, soddisfatto
con un anticipo sempre maggiore, e per anticipare serve un modello di quel che
succederà.

`````

`````{tab} Superiore

Formalmente la risposta sta in dove vivono le preferenze. Non c'è una funzione
di ricompensa affiancata al modello: le condizioni preferite sono i **priori**
del modello generativo stesso, cioè la distribuzione delle osservazioni che
l'agente si aspetta di incontrare in quanto agente di quel tipo.

Ne segue una proprietà che gli autori sottolineano: il modello generativo **non
può limitarsi a imitare la dinamica esterna**, altrimenti l'agente si
limiterebbe a seguirla passivamente. Deve anche specificare le regioni di stati
che deve visitare per continuare a esistere, il che equivale ad assumere
implicitamente che le proprie osservazioni preferite siano più probabili di
quanto siano. Gli autori lo chiamano un **bias di ottimismo**, ed è necessario:
è la differenza fra un modello che descrive il mondo e un modello che prescrive
un comportamento.

È anche il punto in cui questa cornice si separa nettamente dai world model
delle sezioni precedenti. Là il modello stima $p_\theta(s_{t+1} \mid s_t, a_t)$
e basta, e a dire che cosa sia desiderabile ci pensa una ricompensa scritta
fuori dal modello. Qui la desiderabilità **è dentro il modello**, sotto forma di
priori, e non esiste un secondo oggetto da specificare.

`````

## Come fa a scegliere una mossa

Fin qui abbiamo detto il *perché*: un pesce si muove per non trovarsi
all'asciutto. Resta il *come*. Davanti a due mosse possibili, che cosa dice a
un agente del genere quale delle due è meglio? La risposta ha una particolarità
che sarebbe un peccato saltare, ed è la ragione per cui questa cornice torna
utile due volte altrove nel libro: nel capitolo sull'auto-supervisione e nella
sezione sull'esplorazione del deep reinforcement learning.

`````{tab} Elementare

Un agente così, prima di muoversi, misura ogni mossa due volte.

La prima misura è quella che ci si aspetta: quanto quella mossa lo porta verso
le condizioni in cui vuole trovarsi. È il valore di **ottenere**.

La seconda è meno ovvia: quanto quella mossa gli farebbe **scoprire** qualcosa
che ancora non sa. È il valore di **sapere**, e non è un premio di consolazione.
Una mossa che non porta da nessuna parte, ma toglie un dubbio, può valere più di
una che avvicina la meta a occhi chiusi.

L'esempio è degli autori. Uno vuole un caffè e conosce due bar, uno aperto nei
giorni feriali e uno nel fine settimana, ma non sa che giorno è. La prima cosa
che fa è guardare il calendario, non andare a un bar: quel gesto non porta un
passo verso il caffè ma apre la porta del bar giusto. Quel che le due misure
pesano, insomma, non è il passo per sé ma la strada che apre. Il capitolo
sull'auto-supervisione lo racconta per esteso.

Lo stesso conto fa evitare i posti da cui si vede male. Un vicolo e una
piazza possono essere ugualmente vicini al bar, ma dal vicolo non si capisce
nemmeno se la saracinesca è alzata: chi si infila lì resta nel dubbio comunque
si giri, e il dubbio era la cosa che stava cercando di togliersi.

Il punto che conta è che quelle due misure sono le due metà di un voto solo,
che l'agente cerca di rendere più alto possibile, e non due voti da mettere
d'accordo. Non c'è nessuna manopola da girare per decidere quanta
curiosità concedere, perché la curiosità era lì dall'inizio, dentro quel voto,
e nessuno l'ha aggiunta.

`````

`````{tab} Superiore

La quantità che ordina le politiche è l’**energia libera attesa** $G(\pi)$
introdotta più su, e la sua utilità sta tutta nel fatto che si riscrive in modi
diversi. Uno la separa in **rischio** e **ambiguità**:

$$
G(\pi) \;=\;
\underbrace{D_{\mathrm{KL}}\big[\,Q(\tilde{o} \mid \pi) \,\big\|\, P(\tilde{o} \mid C)\,\big]}_{\text{rischio}}
\;+\;
\underbrace{\mathbb{E}_{Q(\tilde{s} \mid \pi)}\big[\,H\big[P(\tilde{o} \mid \tilde{s})\big]\,\big]}_{\text{ambiguità}},
$$

dove $\tilde{o}$ e $\tilde{s}$ sono osservazioni e stati futuri sotto la
politica $\pi$, $C$ raccoglie le preferenze (cioè, come si è appena visto, i
priori sulle osservazioni) e $H$ è l'entropia. Il **rischio** è
quanto le osservazioni che $\pi$ promette si discostano da quelle preferite;
l’**ambiguità** è quanto, in media, gli stati che $\pi$ visita lasciano
incerti sull'osservazione, cioè quanto da lì si vede male.

L'altra riscrittura, algebricamente equivalente, dice che $-G(\pi)$ è la somma
di un **valore epistemico** (il guadagno di informazione atteso) e di un
**valore pragmatico** (l'utilità delle osservazioni che $\pi$ promette); è
quella che il capitolo sull'auto-supervisione sviluppa per intero con l'esempio
del caffè. Poiché $G$ si minimizza, quei due valori si massimizzano insieme, ed
è precisamente il motivo per cui in questo quadro non esiste un compromesso fra
esplorazione e sfruttamento da tarare a mano.

Una nota che servirà fra poco: sviluppando la divergenza, il rischio vale
$-H[Q(\tilde{o} \mid \pi)] - \mathbb{E}_{Q(\tilde{o} \mid \pi)}[\ln P(\tilde{o} \mid C)]$,
quindi porta dentro **anche** l'entropia delle osservazioni previste. Annullare
l'ambiguità non spegne dunque ogni spinta a informarsi: sopravvive quel termine
di entropia, che è poi quello che gli autori, nel caso limite in cui si tolgano
anche le preferenze, descrivono come «tenersi aperte le opzioni». Quel che
sparisce è l'altra metà, la spinta a cercare i posti da cui si vede bene.

`````

## Che cosa se ne porta via questo capitolo

Tre cose, e nessuna delle tre chiede di adottare il quadro per intero.

La prima è una risposta alla domanda con cui il capitolo è partito, «a che serve
un modello del mondo». Qui la risposta non è «a pianificare meglio»: senza un
modello di quel che succederà non si può anticipare niente, e un organismo
che non anticipa arriva sempre tardi. Cercare l'ombra prima di avere caldo è un
atto di previsione, e nient'altro.

La seconda è la scomparsa di un pezzo che altrove sembra obbligatorio. Nei
sistemi di questo capitolo ci sono sempre due oggetti da specificare, il modello
e la ricompensa; qui ce n'è **uno solo**, perché quello che si desidera è scritto
nello stesso posto in cui è scritto quello che ci si aspetta. È una semplificazione
concettuale vera, ed è anche la ragione per cui il capitolo sull'auto-supervisione
può usare questa cornice per rispondere a un'obiezione sul rinforzo: lo fa nella
sua ultima sezione, con l'esempio del caffè per esteso.

La terza è la scala dei tempi. Percepire, agire e imparare sono la stessa
operazione a tre velocità, e non tre programmi diversi che si alternano.
Chi legge il capitolo sull'auto-supervisione riconoscerà l'idea, perché è la
versione biologica di quello che quel capitolo dice dei dati: il bersaglio non lo
scrive nessuno, arriva da sé, ed è il segnale successivo.

```{admonition} Due energie, e non sono la stessa cosa
:class: warning
Il capitolo sui modelli a energia chiama **energia** un punteggio di
compatibilità: quanto una configurazione «sta bene insieme», con il buttafuori
che dà i voti e i paesaggi in cui si cerca il punto più basso. L’**energia
libera** di questa sezione misura un'altra cosa: **quanto quel che capita si
discosta da quel che ci si aspettava**, ed è alta quando il mondo ci smentisce.
Le due parole si somigliano perché vengono dallo stesso
posto, la fisica statistica, e i due conti in qualche punto si toccano; ma
scambiarle porta fuori strada, e chi legge «energia libera» pensando al
buttafuori si perde.

La confusione ha anche un appiglio tipografico, ed è meglio toglierlo di
mezzo: la sezione sulla JEPA scrive $\mathcal{E}$ per l'energia di una coppia
(presente, futuro), e qui $F$ è l'energia libera variazionale. Due lettere
diverse per due grandezze diverse, e la differenza serve: sono le due
«energie» che questo riquadro invita a non scambiare.
```

## Onestà sui limiti

Tre avvertenze, perché questa è una sezione su una teoria e non su un risultato.

**Non è così che si addestrano i sistemi di cui parla il libro.** L'inferenza
attiva nasce come teoria del comportamento biologico, e le sue realizzazioni sono
modelli di laboratorio su compiti piccoli, non i sistemi che giocano a
*Minecraft* o generano video. Le sezioni precedenti di questo capitolo
raccontano quello che funziona oggi; questa racconta un modo di pensarci sopra.

**Non si propone come rivale.** Lo dicono gli autori in apertura, e conviene
riportarlo perché evita di arruolarli in una polemica che non hanno cercato: il
quadro «non mira a rimpiazzare altri quadri di riferimento, come la psicologia
comportamentale, la teoria delle decisioni e l'apprendimento per rinforzo»,
piuttosto spera di comprenderli. Diverse cose che il libro ha già visto si
riottengono infatti come casi particolari, e la sezione sull'esplorazione, nel
{doc}`capitolo sul deep reinforcement learning </DeepReinforcementLearning/overview>`, ne mostra una.

**Una cornice che spiega tutto va maneggiata con cura.** Una teoria che
riconduce percezione, azione, apprendimento, attenzione e omeostasi allo stesso
principio è affascinante proprio per questo, ed è anche per questo che va letta
con attenzione a che cosa, in concreto, essa **vieta**. Il libro non prende
posizione su quanto il principio sia empiricamente falsificabile: registra che è
una cornice unificante, ampiamente discussa, e che le sue previsioni specifiche
si valutano modello per modello, come per qualunque altra teoria.

## La stessa formula, in un altro capitolo di questo libro

Fra le riscritture dell'energia libera attesa, gli autori osservano che
togliendone un pezzo si riottengono schemi già noti, e in particolare che «se
si rimuove l'ambiguità, lo schema risultante corrisponde al controllo
**sensibile al rischio** o al **controllo KL** nella teoria del controllo»
{cite}`parr2022active`.

Ora, «controllo KL» vuol dire una cosa precisa: un problema di controllo in cui
al costo si aggiunge la divergenza da una distribuzione di riferimento. E c'è un
posto, in questo libro, in cui compare un obiettivo fatto esattamente così.

`````{tab} Elementare

Nel capitolo sui Transformer, quando si racconta come un modello di linguaggio
viene rifinito sulle preferenze delle persone, salta fuori una regola che lì
viene chiamata la regola d'oro appesa in cucina: «insegui pure il voto più alto,
ma non allontanarti troppo dalla ricetta di partenza». Serve perché il giudice
che dà i voti è un'imitazione e ha punti ciechi, e un cuoco lasciato libero
finirebbe per cucinare per il giudice invece che per chi mangia.

La somiglianza salta all'occhio: anche lì l'obiettivo è fatto di due pezzi,
«ottieni quello che vuoi» più «resta vicino a com'eri». E non è soltanto una
somiglianza. Quella regola pratica è **lo stesso conto** che si fa qui, scritto
con altre lettere: l'hanno verificato tre ricercatori in un articolo del 2022
{cite}`korbak2022rl`, e uno dei tre studia proprio l'inferenza attiva.

La regola della cucina sembrava una toppa: un accorgimento pratico contro un
guaio pratico. Si scopre invece che era la forma giusta fin dall'inizio, e che
il pezzo «resta vicino a com'eri» non è una precauzione ma **metà della
definizione** del problema. È il genere di guadagno che una teoria può dare
anche a chi non la adotta.

I pezzi, però, non fanno lo stesso mestiere di qua e di là. La ricetta di
partenza non è quello che il cuoco vuole: quello gliel'ha scritto il giudice su
un foglio a parte, e la ricetta è soltanto il punto da cui si è messo a
cucinare. Il pesce quel foglio non ce l'ha, e l'acqua a cui deve restare vicino
è già tutto quello che vuole. La ricetta, in cambio, tiene in cucina anche i
piatti che al giudice nessuno ha mai chiesto: è per questo che il cuoco non
finisce a servire sempre lo stesso.

`````

`````{tab} Superiore

Il confronto, per esteso. Nell'RLHF si massimizza

$$
J(\theta) \;=\; \mathbb{E}\big[r_\phi(x, y)\big] \;-\; \beta \, D_{\mathrm{KL}}\big[\pi_\theta \,\|\, \pi_{\text{ref}}\big],
$$

cioè la ricompensa stimata meno la divergenza dalla policy di partenza; nel
controllo KL si minimizza un costo che somma il costo di stato e la divergenza
da una distribuzione di riferimento. La struttura è la stessa, e in tre righe la
somiglianza si stringe fino all'identità.

A prompt $x$ fissato, poniamo
$\tilde{P}(y) = \pi_{\text{ref}}(y \mid x)\,e^{\,r_\phi(x, y)/\beta}$ e
$Z(x) = \sum_y \tilde{P}(y)$; il rapporto $\pi^\star = \tilde{P}/Z(x)$ è la
policy ottima che il capitolo sui Transformer ricava per altra via. Allora

$$
\frac{J(\theta)}{\beta}
\;=\; -\,\mathbb{E}_{\pi_\theta}\big[\ln \pi_\theta - \ln \tilde{P}\big]
\;=\; -\Big(\underbrace{D_{\mathrm{KL}}\big[\pi_\theta \,\|\, \pi^\star\big]}_{\ge\, 0} \;-\; \ln Z(x)\Big)
\;=\; -\,F .
$$

Quella $F$ è **la stessa** dell'inizio della sezione, riga per riga: $\pi_\theta$
fa da $Q$, il modello generativo non normalizzato è $\tilde{P}$, l'osservazione
si riduce al singolo evento «questa uscita è ottima» e $-\ln Z(x)$ è la sorpresa
di quell'evento. Massimizzare l'obiettivo dell'RLHF **è** minimizzare
un'energia libera variazionale. Non è una nostra lettura: Korbak, Perez e
Buckley lo dimostrano nel 2022, ricavando $J(\theta)$ come ELBO sulla
log-verosimiglianza di quell'evento, con $\pi_{\text{ref}}$ nel ruolo di priore
{cite}`korbak2022rl`; il terzo autore lavora sull'inferenza attiva, il che
spiega da dove arrivi il collegamento.

Restano due differenze che nessuna algebra cancella, e sono quelle da tenere in
mano. **Primo**, i due termini KL non sono lo stesso termine. Nel rischio
dell'energia libera attesa la distribuzione di riferimento **è** la preferenza,
$P(\tilde{o} \mid C)$; nell'RLHF il riferimento è il priore $\pi_{\text{ref}}$
(il modello pre-addestrato, un artefatto della procedura) e la preferenza è
scritta a parte, in $r_\phi$. I due mestieri li fanno oggetti diversi, e i due
schemi si toccano solo passando per il quadro comune del **controllo come
inferenza** (trattare la scelta di una traiettoria come l'inferenza di una
distribuzione a posteriori), non con un travaso diretto.

**Secondo**, che cosa sparisce davvero togliendo l'ambiguità. Non il valore
dell'informazione per intero: come si è visto, il rischio conserva
$-H[Q(\tilde{o} \mid \pi)]$, e simmetricamente
$-\beta D_{\mathrm{KL}}[\pi_\theta \| \pi_{\text{ref}}] =
\beta H[\pi_\theta] + \beta \, \mathbb{E}_{\pi_\theta}[\ln \pi_{\text{ref}}]$,
cioè un **bonus di entropia** sulla policy più un richiamo al priore. È
esattamente il termine a cui {cite}`korbak2022rl` attribuisce il fatto che
l'RLHF non collassi su un'unica risposta, e che conservi fluidità e varietà del
modello di partenza. Quello che sparisce è l'ambiguità in senso stretto: nessuno
dei due schemi ha stati nascosti da disambiguare, quindi la spinta a cercare i
posti da cui si vede bene non ha dove attaccarsi.

`````

Resta poi una cosa che il lettore può portarsi via anche senza seguire un
passaggio di algebra. Quando un obiettivo pratico si scrive come «ottieni ciò
che preferisci, ma non allontanarti dal punto di partenza», quella seconda metà
**è sempre un'ipotesi su dove sia lecito cercare**, mai un dettaglio
implementativo, ed è quasi sempre la parte che decide che cosa il sistema
diventerà.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- L’**inferenza attiva** dice una cosa sola e grossa: percepire e agire sono lo
  stesso mestiere in due direzioni. Davanti a uno scarto
  fra quello che ti aspettavi e quello che trovi, o **cambi idea** (percepire) o
  **cambi il mondo** (agire). Anche **imparare** è la stessa cosa, solo più
  lenta.
- L'obiettivo è non essere sorpresi, e l'obiezione ovvia («allora stattene fermo
  in una stanza buia») cade subito: la sorpresa si misura rispetto a **quello che
  sei**, non a quello che capita. Per un pesce, essere all'asciutto è la cosa più
  sorprendente possibile, e restare immobile non lo aiuta affatto.
- Per questo un organismo è **costretto** a muoversi, e ad anticipare: non basta
  sudare quando fa caldo, conviene cercare l'ombra **prima**. Anticipare vuol
  dire avere un modello di quel che succederà, che è poi l'argomento di questo
  capitolo.
- Per scegliere una mossa, un agente così la misura **due volte**: quanto lo
  porta verso le condizioni che vuole (il valore di **ottenere**) e quanto gli
  farebbe scoprire qualcosa che non sa (il valore di **sapere**). Sono le due
  metà di un voto solo, quindi non c'è nessuna manopola da girare per decidere
  quanta curiosità concedere.
- La differenza dai sistemi delle sezioni precedenti: là ci sono **due** cose da
  scrivere, il modello del mondo e il premio; qui ce n'è **una**, perché quello
  che si desidera sta nello stesso posto in cui sta quello che ci si aspetta.
- La regola d'oro della cucina, quella con cui si rifiniscono i modelli di
  linguaggio («insegui il voto, ma non allontanarti dalla ricetta di partenza»),
  è **lo stesso conto** dell'energia libera scritto con altre lettere, e non
  una toppa: e qualcuno l'ha dimostrato nel 2022.
- Attenzione a non prendere l'inferenza attiva per il seguito delle tappe
  precedenti: è una teoria di come funzionano gli esseri viventi, non il modo
  in cui oggi si costruiscono i programmi che giocano o generano video.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- L’**energia libera variazionale**
  $F = D_{\mathrm{KL}}[Q(s)\|P(s\mid o)] - \ln P(o)$ è un **limite superiore
  sulla sorpresa** $-\ln P(o)$, e il divario è la distanza fra l'approssimazione
  $Q$ e la vera distribuzione a posteriori. Minimizzarla equivale a
  **massimizzare l'evidenza** del modello.
- Tre minimizzazioni della **stessa** quantità rispetto ad argomenti diversi:
  su $Q(s)$ è **percezione**, sulle azioni (che decidono quali $o$ arriveranno)
  è **azione**, sui parametri del modello è **apprendimento**, che gli autori
  descrivono come percezione «su una scala di tempo più lenta».
- $F$ e l’**energia libera attesa** $G(\pi)$ sono due oggetti distinti: $F$ è la
  quantità minimizzata nel tempo, $G$ ordina le politiche ed entra nel modello
  come priore su di esse. $G$ si riscrive come **rischio più ambiguità**, e il
  suo opposto $-G$ come **valore epistemico più valore pragmatico**: da lì il
  fatto che esplorazione e sfruttamento non siano due obiettivi da bilanciare.
- Le **preferenze sono priori** del modello generativo, non una ricompensa
  esterna: da qui la risposta all'obiezione della stanza buia, perché gli stati
  non caratteristici (il pesce all'asciutto) sono i più sorprendenti. Il modello
  non può limitarsi a imitare la dinamica esterna: deve prescrivere gli stati da
  occupare, e gli autori chiamano **bias di ottimismo** questa asimmetria.
- Differenza strutturale dai world model delle sezioni precedenti: là
  $p_\theta(s_{t+1}\mid s_t,a_t)$ più una ricompensa specificata fuori; qui un
  solo oggetto, perché la desiderabilità vive nei priori.
- Limiti dichiarati: le realizzazioni sono modelli di laboratorio, non i sistemi
  allo stato dell'arte; gli autori non propongono il quadro come rivale
  dell'apprendimento per rinforzo ma come cornice che lo comprende; e una teoria
  che unifica moltissimo va valutata su ciò che vieta, modello per modello.
```

`````
