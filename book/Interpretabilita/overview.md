# Aprire la scatola nera

Nel 2016 tre ricercatori dell'Università di Washington (Marco Túlio Ribeiro,
Sameer Singh e Carlos Guestrin) addestrarono di proposito un classificatore
truccato a distinguere le foto di **husky** da quelle di **lupo**: scelsero a
mano le venti immagini di addestramento in modo che tutti i lupi comparissero
su sfondo innevato e nessun husky lo facesse. Come previsto, il classificatore
imparò una scorciatoia («c'è neve → lupo») che con l'animale non c'entrava
nulla: un rilevatore di neve travestito da riconoscitore di canidi. Poi
mostrarono dieci sue predizioni, errori compresi, a ventisette studenti di
machine learning, chiedendo se il modello fosse affidabile e come pensavano che
decidesse: senza altro in mano, in dieci dissero di fidarsi, e meno della metà
sospettò della neve. Allora diedero loro le *spiegazioni*, cioè le porzioni di
immagine su cui ogni risposta si era basata, evidenziate sopra la foto.
L'inganno crollò: i fiduciosi scesero a tre, e venticinque studenti su
ventisette indicarono la neve {cite}`ribeiro2016why`. Era una messinscena
costruita apposta per dimostrare una cosa sola: senza una spiegazione, nemmeno
gli addetti ai lavori si accorgono di un modello che funziona per la ragione
sbagliata. Il metodo che disegna quelle macchie lo proposero gli stessi autori
insieme all'esperimento, e lo vedremo in questo capitolo.

La storia ha un antenato illustre. All'inizio del Novecento, a Berlino, un
cavallo di nome **Hans il Sapiente** sembrava saper contare: gli si chiedeva
«quanto fa sette più cinque?» e lui batteva lo zoccolo dodici volte. Nel 1904
lo psicologo Oskar Pfungst scoprì l'inganno, e nel 1907 lo raccontò in un
libro: Hans non faceva aritmetica, leggeva i movimenti involontari di chi gli
poneva la domanda. L'esaminatore si irrigidiva appena lo zoccolo raggiungeva il
numero giusto, e il cavallo si fermava lì. Bastava che l'esaminatore non
conoscesse la risposta, e Hans sbagliava. Da allora si chiama **effetto Clever
Hans** ogni sistema che *sembra* risolvere un problema mentre in realtà ne
risolve un altro, più facile e nascosto. Il classificatore di husky è un Clever
Hans in silicio: accuratissimo, e per la ragione sbagliata.

Il problema è che una rete neurale con milioni di parametri non ci dice, di suo,
*perché* decide come decide. È una **scatola nera**: entra un input, esce una
risposta, e in mezzo c'è un groviglio di numeri che nessuno legge a occhio.

Vale la pena fermarsi un momento su questo groviglio, perché di solito un
computer fa quello che qualcuno gli ha scritto di fare, e verrebbe da dire:
leggiamo il programma e vediamo cosa c'è scritto. Qui non funziona, e la
ragione è che le regole di questo programma non le ha scritte nessuno. Il
modello se le è ricavate da solo guardando gli esempi, e ciò che ne è uscito
non è un elenco di frasi ma una tabella di milioni di numeri senza nome,
nessuno dei quali significa qualcosa preso da solo: ciascuno sposta la
risposta di pochissimo, e la decisione è la somma di tutti quegli spostamenti.
Stampare il programma non serve a niente, perché il programma *è* quella
tabella. Chiamiamo **interpretabilità** la capacità di capire su che cosa si
appoggia la risposta di un modello, e questo capitolo raccoglie i modi di
ottenerla quando il modello non la offre da sé.

Fin quando la posta in gioco è suggerire un film, poco male. Ma quando un
modello decide se concedere un mutuo, se un tumore è maligno o se rilasciare un
imputato, la domanda «perché?» diventa una questione di fiducia e di giustizia.
È anche, da qualche anno, una questione di legge. Il Regolamento generale sulla
protezione dei dati europeo (GDPR, applicabile dal 2018) ha introdotto norme
sulle decisioni automatizzate e un «diritto alla spiegazione» per chi le
subisce. Fin dove arrivi quel diritto, però, i giuristi lo discutono ancora. Il
testo obbliga a fornire «informazioni significative sulla logica utilizzata»,
e il punto controverso è se questo basti a pretendere il motivo della *singola*
decisione.

`````{tab} Elementare

Immagina un professore che dà sempre voti giusti ma non spiega mai come li
assegna. Finché i voti sono corretti ti fidi; ma il giorno che ne prendi uno
che ti sembra ingiusto, senza una spiegazione non puoi né capire dove hai
sbagliato né difenderti. E se scoprissi che il professore, invece di leggere il
compito, guarda di nascosto la calligrafia o il nome sul foglio? Sarebbe come
il cavallo Hans: risposte «giuste» ottenute con il trucco sbagliato.

Aprire la scatola nera vuol dire proprio questo: chiedere al modello non solo
*cosa* ha deciso, ma *su cosa* si è basato. Nel caso degli husky e dei lupi,
guardare le risposte non bastava: il modello ne azzeccava tante, e per capire
che «vedeva» la neve e non il muso bisognava vedere su quali pixel poggiava.
Contare quante volte un modello ha ragione non dice mai *perché* ha ragione, e
un modello può averla per il motivo sbagliato.

`````

`````{tab} Superiore

Il fenomeno degli husky ha un nome tecnico: **correlazione spuria** (o
*shortcut learning*). Il modello minimizza la sua *loss* sui dati disponibili,
e se una feature accessoria (la neve) è statisticamente associata
all'etichetta nel training *e* nel test, l'ottimizzazione la sfrutta senza
scrupoli: è la strategia più economica per abbassare l'errore. La metrica di
generalizzazione non lo cattura perché il bias è presente in entrambe le
partizioni, indistinguibili sotto l'ipotesi che siano campionate dalla stessa
distribuzione. È l'illusione dell'accuratezza: un modello «giusto per la
ragione sbagliata» collassa appena la distribuzione cambia (un lupo su erba,
un husky sulla neve), perché la scorciatoia appresa non è la relazione causale
che ci interessava. L'**interpretabilità** è lo strumento diagnostico che
espone la discrepanza tra ciò che il modello *dovrebbe* usare e ciò che *usa*
davvero, e che l'accuratezza aggregata, per costruzione, non può vedere.

`````

## Perché aprire la scatola

Le ragioni per volere una spiegazione non sono una sola, e non hanno tutte lo
stesso peso. Vale la pena elencarle, perché guidano *che tipo* di spiegazione
cerchiamo.

- **Fiducia.** Un medico non delega una diagnosi a un sistema di cui non
  capisce il ragionamento. La spiegazione è la condizione perché un esperto
  accetti di affidarsi al modello, o di scartarlo, come è giusto quando guarda
  la neve.
- **Debug.** Il caso husky è il manifesto: senza interpretabilità, un bug
  concettuale (la scorciatoia) resta invisibile dietro una buona accuratezza.
  Aprire la scatola è, prima di tutto, uno strumento di ingegneria.
- **Equità.** Un modello può discriminare per genere o provenienza anche senza
  che quelle informazioni gli siano state date: gli basta una colonna che ne
  faccia le veci, quello che si chiama un *proxy*, cioè una spia. Il quartiere
  di residenza, in molte città, dice qualcosa sul reddito e sulla storia di chi
  ci abita; il tipo di contratto di lavoro dice qualcosa sull'età. Il modello
  non ha bisogno di sapere che sta discriminando: gli basta trovare la spia
  nei dati. Solo esaminando *su cosa* si basa una decisione si può scoprirlo:
  un tema che riprenderemo nel capitolo sull'AI responsabile.
- **Scoperta scientifica.** Quando un modello prevede la struttura di una
  proteina o l'attività di un farmaco, capire *cosa ha imparato* può suggerire
  ipotesi nuove ai ricercatori: il modello come microscopio, non solo come
  oracolo.
- **Obblighi normativi.** Dal credito all'assicurazione, un numero crescente di
  ordinamenti richiede che le decisioni automatizzate su una persona siano, in
  qualche misura, spiegabili e contestabili.

C'è un punto sottile che lega tutto: **la stessa decisione richiede spiegazioni
diverse a seconda di chi la riceve**.

`````{tab} Elementare

Chiedi «perché questo prestito è stato rifiutato?» a tre persone diverse e ti
aspetti tre risposte diverse. Al **cliente** serve una spiegazione azionabile:
«il reddito dichiarato è troppo basso rispetto alla rata; con una rata inferiore
la domanda passerebbe». All'**ingegnere** che ha costruito il modello serve
sapere quali variabili pesano e se ce n'è una sospetta (di nuovo: la neve). Al
**regolatore** serve la garanzia che il sistema non discrimini e che la
decisione sia contestabile. Una sola frase non può accontentarli tutti: la
spiegazione «buona» dipende da a chi parli e a cosa gli serve.

`````

`````{tab} Superiore

Doshi-Velez e Kim {cite}`doshi2017towards` insistono su questo:
l'interpretabilità non è una proprietà monolitica del modello, ma è relativa a
un **compito a valle** e a un **destinatario**. Ne deriva la loro tassonomia
della *valutazione* delle spiegazioni, su tre livelli di rigore crescente:
*application-grounded* (esperti reali sul compito reale; un medico che usa la
spiegazione in corsia), *human-grounded* (persone non esperte su compiti
semplificati, per esperimenti controllati), *functionally-grounded* (nessun
umano, ma una definizione formale di interpretabilità come proxy, per esempio
la profondità di un albero). La lezione è che «spiegabile» senza specificare
*per chi* e *per fare cosa* è un aggettivo vuoto: la stessa uscita del modello
va tradotta in linguaggi diversi per lo sviluppatore, l'utente finale e il
regolatore.

`````

## Una mappa delle spiegazioni

Il campo dell'*explainable AI* può sembrare una giungla di sigle. Tre domande,
indipendenti l'una dall'altra, mettono ordine e ci accompagneranno per tutto il
capitolo.

`````{tab} Elementare

Pensa a tre domande da porre a ogni metodo di spiegazione.

- **Il modello è trasparente di suo, o va spiegato dopo?** Un piccolo albero
  di decisione («se il reddito è sotto X e l'età sotto Y, rifiuta») si legge
  come una ricetta: è **trasparente**. Una rete profonda no, e allora serve
  uno strumento esterno che la interroghi *dopo* l'addestramento (spiegazione
  **post-hoc**).
- **Vuoi capire tutto il modello, o una singola decisione?** «In generale questo
  modello dà molto peso al reddito» riguarda il modello **nel suo insieme**
  (spiegazione *globale*). «*Questo* prestito è stato rifiutato per via della
  rata troppo alta» riguarda **una risposta sola** (spiegazione *locale*).
- **Lo strumento funziona solo per un tipo di modello, o per qualunque
  modello?** Alcuni metodi guardano dentro un modello specifico (i pesi di una
  regressione lineare); altri lo trattano come una scatola chiusa e funzionano
  con qualunque cosa: li chiamiamo *agnostici*.

Tre domande, e ogni tecnica del capitolo trova il suo posto nella griglia.

`````

`````{tab} Superiore

Seguendo l'organizzazione di Molnar {cite}`molnar2022interpretable`,
distinguiamo lungo tre assi.

- **Intrinseca vs post-hoc.** L'interpretabilità *intrinseca* è una proprietà
  strutturale del modello, vincolato a priori a essere leggibile: regressione
  lineare/logistica, alberi poco profondi, sistemi a regole. L'interpretabilità
  *post-hoc* si applica dopo l'addestramento a un modello già dato, senza
  alterarne l'architettura (per esempio calcolando importanze delle feature o
  surrogati locali).
- **Globale vs locale.** Una spiegazione *globale* descrive il comportamento
  del modello sull'intero dominio: quali feature contano in media, che forma
  ha la dipendenza. Una spiegazione *locale* riguarda una singola predizione
  $f(\mathbf{x}_0)$: perché *questo* input ha prodotto *questa* uscita. I due livelli
  richiedono metodi diversi; un modello con superficie decisionale complessa
  può essere globalmente incomprensibile ma localmente approssimabile.
- **Model-specific vs model-agnostic.** Un metodo *specifico* sfrutta la
  struttura interna di una classe di modelli (i coefficienti di un GLM, i
  gradienti di una rete). Un metodo *agnostico* accede solo alla funzione
  input→output $f$ e resta valido per qualunque modello, al costo di stimare il
  comportamento per campionamento anziché leggerlo dai parametri.

I tre assi sono largamente indipendenti: LIME, che vedremo, è post-hoc, locale e
agnostico; i coefficienti di una regressione lineare sono intrinseci, globali e
specifici.

`````

Le tre domande dicono *come* lavora un metodo, non che cosa restituisce, e
conviene sapere fin d'ora che sotto la parola «spiegazione» il capitolo mette
oggetti di forma diversissima: un numero per ogni **feature**, cioè per ogni
colonna dei dati («il reddito ha pesato 25»); una regola scritta («finché il
reddito supera 30 000, è sì»); un altro caso, simile al tuo, in cui la risposta
cambia; una macchia colorata sopra una fotografia; una classifica di colonne.
Cinque cose che non si assomigliano per niente. Quando una sezione dice
«spiegazione», la prima domanda utile è quale delle cinque.

C'è poi una quarta domanda, che non riguarda il funzionamento di un metodo ma
decide quale risposta sia quella giusta: **si vuole spiegare il modello, o il
fenomeno?** Sono cose diverse, e la differenza si misura. Mettiamo che due
colonne dicano quasi la stessa cosa e che il modello ne guardi una sola. Un
metodo che chiede «su che cosa si appoggia questo programma» dà tutto il merito
alla colonna usata e zero all'altra; un metodo che chiede «che cosa dice il
dato» lo divide a metà, perché l'informazione sta in tutte e due. Nessuna delle
due risposte è sbagliata: rispondono a domande diverse. Chi cerca un difetto nel
modello vuole la prima; chi cerca una causa nel mondo vuole la seconda, e da un
modello predittivo non la otterrà comunque, perché il modello ha visto
correlazioni e non esperimenti. Questa forcella tornerà quattro volte nel
capitolo, ogni volta che due metodi entrambi ragionevoli daranno due numeri
diversi per lo stesso caso.

L'esempio più pulito di interpretabilità *intrinseca* (cioè che il modello ha
già per come è costruito, senza bisogno di strumenti esterni) è un albero di
decisione poco profondo: la sua logica si stampa per intero.

Il dataset `iris` è un classico: 150 fiori di iris di tre specie, ciascuno
descritto da quattro misure (lunghezza e larghezza del petalo e del sepalo), e
il compito è indovinare la specie.

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text

# Un modello intrinsecamente interpretabile: un albero volutamente basso
iris = load_iris()
X, y = iris.data, iris.target
albero = DecisionTreeClassifier(max_depth=2, random_state=0)
albero.fit(X, y)

# Tutta la "logica" del modello è leggibile come una ricetta di if-then
print(export_text(albero, feature_names=list(iris.feature_names)))
```

Ecco cosa stampa, per intero:

```text
|--- petal width (cm) <= 0.80
|   |--- class: 0
|--- petal width (cm) >  0.80
|   |--- petal width (cm) <= 1.75
|   |   |--- class: 1
|   |--- petal width (cm) >  1.75
|   |   |--- class: 2
```

Tre regole annidate su una misura sola, la larghezza del petalo: petalo
strettissimo, prima specie; sotto 1,75 cm, seconda; sopra, terza. Nessuna
spiegazione post-hoc, il modello *è* la propria spiegazione, e sta in sette
righe. Se lo stesso dato lo diamo in pasto a una foresta casuale da centinaia
di alberi (tanti alberi diversi che votano insieme sbagliano meno di uno solo,
come si è visto nella sezione sugli alberi e gli ensemble del capitolo sul
machine learning) l'accuratezza sale ma la leggibilità sparisce, ed è lì che
nascono i metodi del capitolo.

## Spiegare non è essere fedeli

C'è una trappola, e il campo ci è caduto più di una volta. Una spiegazione
post-hoc è, per sua natura, una ricostruzione fatta *dopo*, a decisione presa,
di ciò che il modello ha fatto: non il modello stesso. E una ricostruzione può essere
**plausibile senza essere fedele**: raccontarci una storia convincente sul
perché di una decisione, mentre il modello dentro faceva tutt'altro.

`````{tab} Elementare

Immagina di chiedere a un amico perché ha scelto un ristorante e lui ti
risponde: «per le recensioni». Suona sensato, ma se la verità è che ci andava
la sua ex e voleva rivederla, la spiegazione è *credibile* e *falsa* insieme.
Con i modelli è lo stesso: uno strumento post-hoc può produrre una motivazione
che a noi umani sembra ragionevole, ma che non corrisponde a come il modello
ha davvero deciso. Chiamiamo **fedeltà** quanto la spiegazione aderisce al
vero funzionamento del modello, e **plausibilità** quanto ci convince. Sono
due cose diverse, e la seconda è pericolosa proprio quando non c'è la prima:
una spiegazione bella e infedele ci fa fidare di un modello che non lo merita.

`````

`````{tab} Superiore

Una famiglia importante di spiegazioni post-hoc costruisce un modello
surrogato $g$, interpretabile, che approssima la scatola nera $f$ in un
intorno del punto di interesse. La sua qualità si misura con la **fedeltà
locale**, cioè quanto $g$ e $f$ concordano sui punti $\mathbf{z}$ campionati
vicino a $\mathbf{x}_0$. La si quantifica per il suo rovescio, misurando quanto
i due *discordano*:

$$
\text{infedeltà}(g; \mathbf{x}_0) = \mathbb{E}_{\mathbf{z} \sim \pi_{\mathbf{x}_0}}
\big[\, \ell\!\left(g(\mathbf{z}),\, f(\mathbf{z})\right) \,\big],
$$

dove $\pi_{\mathbf{x}_0}$ è una distribuzione di prossimità centrata su
$\mathbf{x}_0$ e $\ell$
una loss adatta al tipo di uscita: l'indicatrice di disaccordo per etichette
discrete, uno scarto quadratico per probabilità o punteggi continui. La
quantità cresce quando la fedeltà cala: tanto più è piccola, tanto più $g$ è
fedele a $f$ in quell'intorno (all'estremo, vale zero se il surrogato riproduce
esattamente la scatola nera sui punti campionati). Una fedeltà alta
*sull'intorno* non garantisce nulla *globalmente*, ed è del tutto
scorrelata dalla **plausibilità**: quanto la spiegazione appare sensata a un
umano. Nulla vieta a un surrogato di essere plausibile e infedele, o fedele e
controintuitivo. È il difetto costitutivo dei metodi a surrogato: approssimano,
e un'approssimazione può ingannare. Non tutto il post-hoc è fatto così
(l'importanza per permutazione, i controfattuali e le attribuzioni che vedremo
interrogano $f$ direttamente, senza copie di mezzo), ma la distinzione tra
plausibilità e fedeltà vale per ogni spiegazione, comunque prodotta.

`````

Da qui uno dei dibattiti più netti del campo. Cynthia Rudin
{cite}`rudin2019stop` sostiene una tesi tagliente: per le **decisioni ad alto
rischio** (giustizia, sanità, credito) si dovrebbe **smettere di spiegare le
scatole nere** e usare invece **modelli intrinsecamente interpretabili**.
L'argomento è duplice. Primo: una spiegazione post-hoc infedele è peggio di
niente, perché dà una falsa sensazione di controllo. Secondo, e più radicale:
il presunto compromesso tra accuratezza e interpretabilità spesso *non
esiste*; su molti problemi con feature ben strutturate, un modello trasparente
raggiunge la stessa accuratezza di uno opaco, e allora l'opacità è un costo
senza contropartita.

Non tutti concordano. Su dati non strutturati (immagini, testo, segnali) le
reti profonde restano di gran lunga le più accurate, e rinunciarvi non è
un'opzione: lì il post-hoc è, pragmaticamente, l'unica finestra disponibile
sulla scatola nera. Il monito di Doshi-Velez e Kim {cite}`doshi2017towards`
vale per entrambe le fazioni: qualunque spiegazione va prodotta e *valutata*
con rigore scientifico, dichiarando cosa misura e con quale metodo, non
offerta come rassicurazione qualitativa. Non esistono spiegazioni «gratis»:
esistono spiegazioni verificate e spiegazioni che ci raccontiamo.

## Come è organizzato il capitolo

Con questa mappa in tasca, il capitolo procede seguendo quelle tre domande.

Cominceremo dai **modelli trasparenti**: la regressione lineare e logistica,
gli alberi di decisione, i sistemi a regole. Li abbiamo già incontrati nel
capitolo sul Machine Learning, e qui li rileggiamo con un'altra domanda in
testa, quanto sono leggibili. Vengono poi le misure di **importanza delle
feature**, che dicono su che cosa un modello si appoggia in media, su tutti gli
esempi insieme.

Passeremo quindi alle **spiegazioni locali**, quelle che riguardano *questa*
singola predizione. Sono tre: **LIME**, il metodo del caso husky, che
approssima il modello con un surrogato lineare nell'intorno del punto; **SHAP**,
che distribuisce il «merito» di una predizione tra le feature con una regola
presa dalla teoria dei giochi; e le spiegazioni **controfattuali**, che
rispondono a «cosa sarebbe dovuto cambiare nell'input perché la decisione fosse
diversa?».

Chiuderemo con l'**attribuzione nelle reti profonde**, cioè saliency map,
gradienti integrati e le mappe di attenzione che avevamo visto nascere nel
capitolo sui Transformer. E con la giovane **interpretabilità meccanicistica**,
che prova a smontare una rete circuito per circuito, come un ingegnere inverso
apre un chip per capirne la logica.

Un filo, sopra a tutto, tiene insieme il capitolo con quello sull'**AI
responsabile**: aprire la scatola nera non è un vezzo accademico, ma il primo
passo per costruire sistemi di cui potersi fidare, e da poter contestare
quando sbagliano. Il classificatore che guardava la neve era stato truccato
apposta, per dimostrare quanto è facile non accorgersene: i modelli veri
arrivano da soli alla stessa scorciatoia, e l'unico modo di scoprirlo è
guardarci dentro.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un modello può indovinare tantissimo e farlo **per la ragione sbagliata**: il
  cavallo Hans leggeva la faccia di chi chiedeva, il riconoscitore di lupi
  guardava la neve. Contare quante volte ha ragione non lo smaschera; guardare
  su che cosa si appoggia sì. Questo è l'**interpretabilità**.
- Dentro un modello non c'è un programma da leggere: ci sono milioni di numeri
  che nessuno ha scritto a mano e che il modello si è ricavato dagli esempi.
  Per questo serve un capitolo intero invece di una stampa.
- Si chiede una spiegazione per **fidarsi**, per **trovare i difetti**, per
  **equità**, per **scoprire** cose nuove e perché a volte lo **impone la
  legge**. E la spiegazione buona dipende da chi la riceve: al cliente serve
  sapere cosa cambiare, all'ingegnere quali variabili pesano, al regolatore che
  il sistema non discrimini.
- Tre domande ordinano tutti i metodi del capitolo: il modello si legge da sé
  o va interrogato **dopo**? vuoi capire il modello **intero** o **una risposta
  sola**? lo strumento serve un solo tipo di modello o va bene per
  **qualunque** modello? E una quarta, che decide chi ha ragione quando due
  metodi discordano: vuoi spiegare **il modello** o **il fenomeno**?
- Una spiegazione può essere **convincente e falsa insieme** (l'amico che dice
  «ci vado per le recensioni» e ci va per la sua ex). «Mi convince» e «è vera»
  sono due cose diverse, e la prima senza la seconda è pericolosa: ci fa
  fidare di un modello che non lo merita.
- Il dibattito: per le decisioni che pesano davvero (giustizia, sanità,
  credito) Cynthia Rudin dice che è meglio usare un modello trasparente invece
  di appiccicare una spiegazione a una scatola nera, perché su quei dati la
  scatola nera spesso non indovina nemmeno di più. Su foto, testo e suoni,
  però, la scatola nera resta la più brava, e la spiegazione appiccicata dopo
  è l'unica finestra che abbiamo.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Un modello accurato può esserlo **per la ragione sbagliata** (effetto *Clever
  Hans*, la neve al posto del lupo): l'accuratezza aggregata non smaschera le
  **correlazioni spurie**, l'interpretabilità sì.
- Si spiega per **fiducia, debug, equità, scoperta scientifica e obblighi
  normativi**; la spiegazione «buona» dipende da **a chi** serve:
  sviluppatore, utente finale, regolatore vogliono cose diverse.
- Tre assi ordinano il campo: **intrinseca vs post-hoc**, **globale vs locale**,
  **model-specific vs model-agnostic**. Sono largamente indipendenti. A essi si
  affianca una domanda che non è un asse ma decide quale risposta sia corretta:
  si sta spiegando **il modello** o **il fenomeno**?
- **Plausibilità ≠ fedeltà**: una spiegazione post-hoc può convincere senza
  aderire a come il modello decide davvero. È il rischio di ogni racconto
  costruito dopo, a decisione presa, e cresce quando a essere letto non è il
  modello vero ma una sua copia semplificata.
- Il dibattito: Rudin invita a **usare modelli interpretabili** per le decisioni
  ad alto rischio invece di spiegare scatole nere; sui dati non strutturati il
  post-hoc resta l'unica finestra. In ogni caso, spiegazioni **valutate con
  rigore** (Doshi-Velez & Kim), non rassicurazioni qualitative.
```

`````
