# Aprire la scatola nera

Nel 2016 tre ricercatori dell'Università di Washington (Marco Túlio Ribeiro,
Sameer Singh e Carlos Guestrin) costruirono di proposito un programma truccato.
Doveva guardare una fotografia e dire se ritraeva un **husky** o un **lupo**, e
nessuno gliel'aveva insegnato a parole: l'aveva imparato da solo, guardando
delle foto su cui qualcuno aveva già scritto la risposta giusta. Dare in pasto
a un programma quegli esempi si chiama **addestrarlo**, e il programma che ne
esce, quello che d'ora in poi risponde da solo sulle foto nuove, si chiama
**modello**. È la parola che tornerà in ogni riga di questo capitolo.

Il trucco stava nelle foto. Erano venti soltanto, poche apposta, e scelte a mano
in modo che tutti i lupi comparissero su sfondo innevato e nessun husky lo
facesse. Come previsto, il modello imparò una scorciatoia («c'è neve → lupo»)
che con l'animale non c'entrava nulla: un rilevatore di neve travestito da
riconoscitore di canidi.

Poi venne la parte interessante. I tre mostrarono dieci risposte del modello,
errori compresi, a ventisette studenti universitari che un corso di machine
learning l'avevano già fatto, e chiesero loro se si
fidassero e come pensavano che quel programma decidesse. Con le sole risposte
sotto gli occhi, dieci studenti su ventisette dissero di fidarsi, e meno della
metà sospettò della neve. Allora i ricercatori diedero loro le *spiegazioni*:
sopra ogni foto, evidenziate a colori, le porzioni di immagine su cui quella
risposta si era basata. L'inganno crollò. I fiduciosi scesero a tre, e
venticinque studenti su ventisette indicarono la neve
{cite}`ribeiro2016why`. Era una messinscena costruita apposta per dimostrare
una cosa sola: senza una spiegazione, nemmeno gli addetti ai lavori si
accorgono di un modello che funziona per la ragione sbagliata. Il metodo che
disegna quelle macchie lo proposero gli stessi autori insieme all'esperimento,
e lo vedremo in questo capitolo.

La storia ha un antenato illustre. All'inizio del Novecento, a Berlino, un
cavallo di nome **Hans il Sapiente** sembrava saper contare: gli si chiedeva
«quanto fa sette più cinque?» e lui batteva lo zoccolo dodici volte. Nel
settembre del 1904 una commissione di tredici persone lo esaminò e concluse due
cose: che imbroglio non ce n'era, e che il caso meritava un'indagine seria.
L'indagine cominciò poche settimane dopo, all'Istituto di psicologia
dell'università di Berlino, e a condurla fu lo psicologo Oskar Pfungst: in due
mesi era finita, e a dicembre si sapeva come stavano le cose. Hans non faceva
aritmetica, leggeva i movimenti involontari di chi gli poneva la domanda.
L'esaminatore si irrigidiva appena lo zoccolo raggiungeva il
numero giusto, e il cavallo si fermava lì. Bastava che l'esaminatore non
conoscesse la risposta, e Hans sbagliava. Il resoconto per esteso Pfungst lo
pubblicò in un libro nel 1907, ed è quello che si cita ancora. Da allora si
chiama **effetto Clever
Hans** ogni sistema che *sembra* risolvere un problema mentre in realtà ne
risolve un altro, più facile e nascosto. Il rilevatore di lupi è un Clever
Hans in silicio: azzeccava quasi sempre, e per la ragione sbagliata.

Il problema è che un modello non ci dice, di suo, *perché* decide come decide.
È una **scatola nera**: entra una fotografia, esce una risposta, e in mezzo non
si vede niente.

Vale la pena fermarsi un momento su quel «in mezzo», perché di solito un
computer fa quello che qualcuno gli ha scritto di fare, e verrebbe da dire:
apriamo il programma e leggiamo cosa c'è scritto. Qui non funziona, e la
ragione è che le regole di questo programma non le ha scritte nessuno. Il
modello se le è ricavate da solo guardando gli esempi, e ciò che ne è uscito
non è un elenco di frasi ma una tabella di numeri senza nome: milioni, nei
modelli di oggi. Quei numeri si chiamano **parametri**. E il tipo di
modello che li organizza a strati, dove il primo strato fa un po' di conti sui
dati in arrivo, il secondo rifà i conti sul risultato del primo, e così via fino
in fondo, si chiama **rete neurale**.

Nessuno di quei numeri, preso da solo, significa qualcosa, e conviene vedere in
che senso. Il modello non risponde «lupo» e basta: risponde con un punteggio,
mettiamo 87 su 100 a favore del lupo. Quegli 87 non stanno scritti da nessuna
parte: sono il risultato di milioni di spintarelle minuscole, alcune verso il
lupo e altre contro, che si compensano quasi tutte fra loro e lasciano quel
numero come residuo. Stampare il programma non serve a niente, perché il
programma *è* quella tabella, e lì dentro la parola «neve» non è scritta da
nessuna parte. Chiamiamo **interpretabilità** la capacità di capire su che cosa
si appoggia la risposta di un modello, e questo capitolo raccoglie i modi di
ottenerla quando il modello non la offre da sé.

Fin quando la posta in gioco è suggerire un film, poco male. Ma quando un
modello decide se concedere un mutuo, se un tumore è maligno o se rilasciare un
imputato, la domanda «perché?» diventa una questione di fiducia e di giustizia.

Da qui in avanti gli esempi cambieranno spesso faccia, e conviene sapere come
sono fatti i dati quando non sono fotografie. Sono una tabella: una **riga** per
persona, e una **colonna** per ogni informazione che di lei si conosce, il
reddito, l'età, i debiti in corso. Sono quelle colonne che il capitolo passerà
il tempo a interrogare, e più avanti le chiameremo anche **feature**, che è il
termine di mestiere.

La domanda «perché?», dicevamo, è anche una questione di legge. Il Regolamento generale sulla
protezione dei dati europeo (GDPR, applicabile dal 2018) detta delle regole
sulle decisioni prese da un programma senza che un essere umano ci metta mano,
e obbliga chi le usa a dare all'interessato «informazioni significative sulla
logica utilizzata». Che da lì nasca un vero e proprio «diritto alla
spiegazione» è invece contestato fra i giuristi, e il punto della lite è
proprio questo: quelle informazioni riguardano il funzionamento del sistema in
generale, o si può pretendere il motivo della *propria* decisione, quella e non
un'altra? La differenza fra le due cose, che qui sembra un cavillo da tribunale,
è invece una delle domande con cui fra poco metteremo ordine fra i metodi:
spiegare il modello in generale non è la stessa cosa che spiegare una sua
singola risposta, e non si fa con gli stessi attrezzi.

`````{tab} Elementare

Immagina un professore che dà sempre voti giusti ma non spiega mai come li
assegna. Finché i voti sono corretti ti fidi; ma il giorno che ne prendi uno
che ti sembra ingiusto, senza una spiegazione non puoi né capire dove hai
sbagliato né difenderti. E adesso il colpo di scena: quel professore i compiti
non li legge, guarda la calligrafia. Nella sua classe, per caso, i più bravi
scrivono anche più ordinato, e lui quella regola lì l'ha imparata bene: finché
la classe è quella, i voti tornano davvero. È il cavallo Hans in cattedra:
risposte «giuste» ottenute con il metodo sbagliato, e nessuno se ne accorge
finché non arriva uno bravo con una brutta calligrafia.

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
  capisce il ragionamento. E senza una spiegazione non può nemmeno fare il
  contrario, cioè scartarlo con cognizione di causa: gli mancano gli elementi
  tanto per fidarsi quanto per rifiutare.
- **Trovare i difetti.** Il caso husky è il manifesto. La scorciatoia non è un
  errore di programmazione: il programma faceva esattamente quello che doveva.
  È un difetto che si vede solo guardando *su che cosa* il modello si appoggia,
  perché dai risultati non emerge affatto: il modello indovina tanto, e chi
  guarda i risultati lo promuove. (Cercare e togliere i difetti da un programma
  si chiama fare *debug*.) Aprire la scatola è, prima di tutto, uno strumento
  di ingegneria.
- **Equità.** Un modello può discriminare per genere o provenienza anche senza
  che quelle informazioni gli siano state date: gli basta una colonna che ne
  faccia le veci, cioè che ne sia una spia. Il quartiere di residenza, in molte
  città, dice qualcosa sul reddito di chi ci abita, e un modello che rifiuta un
  prestito «per il quartiere» sta di fatto rifiutandolo per il reddito, che è
  proprio la cosa che non gli era stata data. Il modello non ha bisogno di
  sapere che sta discriminando: gli basta trovare la spia nei dati. Solo
  esaminando *su cosa* si basa una decisione si può scoprirlo, ed è un tema che
  riprenderemo nel capitolo sull'AI responsabile.
- **Scoperta scientifica.** Ci sono modelli che indovinano cose che nessuno
  sapeva prevedere: come si ripiega una proteina, se una molecola nuova sarà un
  farmaco. Lì la domanda cambia segno. Non si chiede una spiegazione per
  controllare il modello, la si chiede per **imparare qualcosa dal modello**:
  se ha capito qualcosa che noi non sappiamo, quel qualcosa è un'ipotesi da
  andare a verificare in laboratorio. Il modello come microscopio, non solo
  come macchina che dà responsi.
- **Obblighi normativi.** Dal credito all'assicurazione, sempre più leggi
  chiedono che una decisione presa da un programma su una persona sia, in
  qualche misura, spiegabile e contestabile.

C'è un punto sottile che lega tutto: **la stessa decisione richiede spiegazioni
diverse a seconda di chi la riceve**.

`````{tab} Elementare

Chiedi «perché questo prestito è stato rifiutato?» a tre persone diverse e ti
aspetti tre risposte diverse. Al **cliente** serve sapere che cosa può fare:
«il reddito dichiarato è troppo basso rispetto alla rata; con una rata inferiore
la domanda passerebbe». All'**ingegnere** che ha costruito il modello serve
sapere quali colonne pesano e se ce n'è una sospetta, cioè se in questa tabella
esiste l'equivalente della neve. All'ufficio
pubblico che vigila sulle banche, il **regolatore**, serve la garanzia che il
sistema non discrimini e che chi si vede dire di no possa protestare con
qualche argomento in mano. Una sola frase non può accontentarli tutti: la
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

I metodi per spiegare un modello sono decine, e presi in blocco sembrano un
elenco senza capo né coda. Tre domande, indipendenti l'una dall'altra, mettono
ordine e ci accompagneranno per tutto il capitolo.

`````{tab} Elementare

Pensa a tre domande da porre a ogni metodo di spiegazione.

- **Il modello si legge da sé, o va spiegato dopo?** Alcuni modelli decidono
  con una catena di domande sì/no («il reddito supera i 30 000? se sì, l'età
  supera i 40? se no, rifiuta»). Un modello così si disegna, e viene fuori una
  cosa che si biforca a ogni domanda: per questo si chiama **albero di
  decisione**. Leggerlo è come leggere una ricetta, e diciamo che è
  **trasparente**. Un modello con milioni di numeri dentro no: lì serve uno
  strumento esterno che lo interroghi *dopo* che ha finito di imparare, e una
  spiegazione ottenuta così si chiama **post-hoc**, che in latino vuol dire
  «dopo il fatto».
- **Vuoi capire tutto il modello, o una singola decisione?** «In generale questo
  modello dà molto peso al reddito» riguarda il modello **nel suo insieme**
  (spiegazione *globale*). «*Questo* prestito è stato rifiutato per via della
  rata troppo alta» riguarda **una risposta sola** (spiegazione *locale*).
- **Lo strumento funziona solo per un tipo di modello, o per qualunque
  modello?** Alcuni metodi hanno bisogno di sapere com'è fatto il modello
  dentro, e valgono solo per quel tipo lì. Altri non lo aprono nemmeno: gli
  passano un caso, si prendono la risposta, e tanto basta. Questi ultimi
  funzionano con qualunque cosa, e si chiamano *agnostici*: la parola, presa a
  prestito dalla filosofia, qui vuol dire soltanto «che non si pronuncia su
  com'è fatto il modello».

Tre domande, e ogni tecnica del capitolo si colloca rispondendo a tutte e tre.

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

Le tre domande dicono *come* lavora un metodo, non che cosa restituisce. E qui
conviene mettere in guardia, perché sotto la parola «spiegazione» questo
capitolo mette oggetti di forma diversissima:

- **una classifica delle colonne**, valida per tutti gli esempi insieme: «in
  questo modello il reddito conta più dell'età, e il colore preferito non conta
  niente»;
- **un conto per un caso solo**, che è un'altra cosa: non l'ordine delle
  colonne, ma di quanto ciascuna ha spinto *questa* risposta. Spinto rispetto a
  che cosa? Rispetto alla risposta che il modello darebbe di un cliente di cui
  non sapesse niente. Ed è un conto vero e proprio, nel senso che le quote
  devono sommare esattamente alla distanza fra le due risposte, senza avanzi;
- **una regola scritta**: «finché il reddito supera 30 000 e non ci sono
  ritardi di pagamento, la risposta è sì»;
- **un altro caso, quasi identico, in cui la risposta cambia**: «con 6 000 euro
  di reddito in più sarebbe stato un sì»;
- **una macchia colorata sopra una fotografia**, come quella che ha smascherato
  la neve.

Cinque cose che non si assomigliano per niente, e la prima e la seconda si
somigliano solo in apparenza: una classifica non ti dice niente sul tuo caso, e
un conto sul tuo caso non è una classifica valida per tutti. Quando una sezione
dice «spiegazione», la prima domanda utile è quale delle cinque.

C'è poi una quarta domanda, che non riguarda il funzionamento di un metodo ma
decide quale risposta sia quella giusta: **si vuole spiegare il modello, o il
mondo?** Sembra la stessa cosa e non lo è, e conviene vederlo su un caso.

Mettiamo che nella tabella ci siano due colonne che dicono quasi la stessa cosa:
lo stipendio del mese e il reddito dichiarato in un anno. Chi ha l'uno alto ha
alto anche l'altro, quindi al modello ne basta una, e mettiamo che abbia scelto
lo stipendio e ignorato il reddito annuo. Adesso chiediamoci quanto vale
ciascuna delle due colonne, e notiamo che ci sono due domande diverse, non una.

- «**Su che cosa si appoggia questo programma?**» Risposta: tutto sullo
  stipendio, zero sul reddito annuo. È la verità sul programma: se cancelli la
  colonna del reddito annuo, non cambia niente.
- «**Che cosa ci dice il dato?**» Risposta: metà e metà. Le due colonne
  portano la stessa informazione, il modello ne ha scelta una per caso, e se lo
  si riaddestrasse domani potrebbe scegliere l'altra. Non c'è nessuna ragione
  per dare più credito a una che all'altra, quindi si divide in parti uguali.

Nessuna delle due risposte è sbagliata: rispondono a domande diverse, e chi
usa un metodo senza sapere a quale delle due sta rispondendo si prende il numero
per quello che non è. Chi cerca un difetto nel modello vuole la prima. Chi cerca
una causa nel mondo vorrebbe la seconda, e conviene dire subito che nemmeno
quella gliela darà: un modello ha visto solo cose che vanno insieme (si dice
che sono **correlate**), non ha mai fatto un esperimento, e due cose che vanno
insieme non sono per forza l'una la causa dell'altra. Questo bivio tornerà più volte
nel capitolo, ogni volta che due metodi entrambi ragionevoli daranno due numeri
diversi per lo stesso caso. (Il bivio ricompare soprattutto nei passaggi più
tecnici del capitolo, quelli del livello Superiore: se ne troverà traccia ogni
volta che due metodi discordano.)

## Un modello che si spiega da sé

Basta di teoria: guardiamone uno. Della prima delle tre domande abbiamo detto
che alcuni modelli si leggono da sé e altri no, e vale la pena vedere in
concreto che aspetto ha un modello del primo tipo. Ricordiamo il nome che gli
diamo: **trasparente**, e nel resto del capitolo si dirà anche che è
interpretabile in modo **intrinseco**, che è la stessa cosa detta col termine
di mestiere.

L'esempio più pulito è l'albero di decisione già incontrato, tenuto basso, con
poche domande. Qui sotto lo costruiamo davvero, con qualche riga di codice: chi
non programma può saltare il riquadro grigio e leggere subito quello che stampa,
che è il pezzo interessante.

I dati sono un classico, l'`iris`: 150 fiori di iris di tre specie diverse,
ciascuno descritto da quattro misure in centimetri (lunghezza e larghezza del
petalo, e le stesse due del sepalo, che è la fogliolina verde che sta sotto al
fiore). Il compito è indovinare la specie a partire dalle quattro misure.

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

Le tre righe che cominciano con `class` sono le tre specie, numerate da zero
come si usa in programmazione. Il resto sono tre regole annidate su una misura
sola, la larghezza del petalo: fino a 0,80 cm, prima specie; fra 0,80 e 1,75,
seconda; oltre 1,75, terza. Nessuno strumento esterno, nessuna spiegazione
aggiunta dopo: il modello *è* la propria spiegazione, e sta in sette righe.

Adesso il confronto. Al posto di un albero solo se ne possono far crescere
centinaia, tutti un po' diversi fra loro, e far votare le loro risposte: si
chiama **foresta casuale**, ed è il metodo visto nella sezione sugli alberi e
gli insiemi di modelli del capitolo sul machine learning. Tanti pareri sbagliano meno di
uno, e infatti su questi fiori la foresta indovina un po' più dell'alberello: il
$96{,}0\%$ contro il $94{,}7\%$. Il modo di misurarlo è quello di sempre, e vale
la pena dirlo: si dividono i 150 fiori in dieci gruppi, si addestra il modello
su nove gruppi e lo si interroga sul decimo, e si ripete dieci volte cambiando
ogni volta il gruppo tenuto da parte, così che ogni fiore prima o poi faccia da
esame. La media di quelle dieci prove è il numero scritto sopra. La differenza,
dunque, è di **un punto e tre**. In cambio, la logica della foresta non si
stampa più, perché non è una ricetta: sono centinaia di ricette che votano. È in questo scambio che nascono i metodi del capitolo, e
vale la pena notare fin d'ora che qui lo scambio conviene poco.

## Una spiegazione può convincere ed essere falsa

C'è una trappola, e i ricercatori ci sono caduti più di una volta. Una
spiegazione che arriva dopo, a decisione presa, è un racconto su quello che è
successo. Non è la cosa che è successa. E un racconto può essere convincente e
falso insieme.

`````{tab} Elementare

Pensa a un telecronista che spiega perché un calciatore ha tirato in quel modo:
«ha visto il portiere spostarsi». Suona sensato, sta guardando la stessa partita
che guardi tu, ma non è stato nella testa del calciatore, che magari è solo
scivolato. La spiegazione è *credibile* e *falsa* insieme, e il telecronista
non sta mentendo: sta indovinando dall'esterno, ed è tutto quello che può fare.

Uno strumento che spiega un modello dall'esterno è nella stessa posizione. Può
produrre una motivazione che a noi sembra ragionevole e che non corrisponde a
come il modello ha davvero deciso. Chiamiamo **fedeltà** quanto la spiegazione
aderisce al vero funzionamento del modello, e **plausibilità** quanto ci
convince. Sono due cose diverse, e la seconda è pericolosa proprio quando non
c'è la prima: una spiegazione bella e infedele ci fa fidare di un modello che
non lo merita.

E il guaio è che a occhio non si distinguono, perché una spiegazione che
convince e una spiegazione vera si presentano allo stesso identico modo. Come
si smaschera una spiegazione infedele? Provando a farla fallire: si cambia
qualcosa nel modello e si guarda se la spiegazione cambia di conseguenza. Se
non cambia, non stava parlando del modello. È esattamente l'esperimento che
faremo nell'ultima sezione del capitolo, quella dentro le reti profonde, e che
boccerà parecchi metodi in uso.

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

Da qui uno dei dibattiti più aspri della materia, e per capirlo serve prima
mettere sul tavolo la convinzione che mette in discussione. La convinzione è
questa: che chiarezza e bravura si paghino l'una con l'altra, cioè che un
modello leggibile sia per forza più scarso di uno oscuro, e che quindi l'oscurità sia un prezzo
che si paga volentieri per avere ragione più spesso. Chiamiamolo il presunto
**scambio fra accuratezza e chiarezza**; sui fiori di poco fa lo abbiamo visto
valere un punto e tre.

Cynthia Rudin {cite}`rudin2019stop` sostiene una tesi tagliente: per le
**decisioni che pesano davvero** (giustizia, sanità, credito) si dovrebbe
**smettere di spiegare le scatole nere** e usare direttamente modelli leggibili.
L'argomento è duplice. Primo: una spiegazione infedele è peggio di niente,
perché dà una falsa sensazione di controllo. Secondo, e più radicale: quello
scambio, su molti problemi veri, semplicemente non esiste. Quando i dati stanno
in una tabella e ogni colonna significa qualcosa di preciso (il reddito, l'età,
la pressione arteriosa), un modello trasparente ben costruito arriva quasi
sempre dove arriva la scatola nera, e allora l'oscurità è un costo senza
contropartita: si paga e non si compra niente.

Non tutti concordano, e la ragione è che non tutti i dati stanno in una
tabella. Su fotografie, testi e suoni, dove le colonne di partenza sono i pixel
o le lettere e presi uno per uno non significano nulla, le reti profonde
restano di gran lunga le più brave, e rinunciarvi non è un'opzione: lì una
spiegazione appiccicata dopo è l'unica finestra che abbiamo.

Su una cosa, però, le due fazioni concordano, e l'hanno scritta nel 2017 due
ricercatrici, Finale Doshi-Velez e Been Kim {cite}`doshi2017towards`: una
spiegazione è una cosa da **misurare**, non da esibire. Chi la produce deve
dichiarare che cosa ha misurato e con quale esperimento, esattamente come si fa
per l'**accuratezza** di un modello, che è la quota di risposte giuste e che
nessuno si sognerebbe di dichiarare senza dire su quali casi l'ha contata. Non
esistono spiegazioni «gratis»: esistono spiegazioni verificate e spiegazioni
che ci raccontiamo.

## Come è organizzato il capitolo

Con questa mappa in tasca, il capitolo procede seguendo quelle tre domande.

Cominceremo dai **modelli trasparenti**, quelli che si leggono senza aiuto:
l'albero di poco fa, e i modelli che rispondono facendo una somma, tanti punti
per il reddito, tanti per l'età. Li abbiamo già incontrati nel capitolo sul
Machine Learning, e qui li rileggiamo con un'altra domanda in testa, quanto sono
leggibili. Vengono poi i modi di misurare su che cosa un modello si appoggia in
media, su tutti gli esempi insieme: è la classifica delle colonne, e si chiama
**importanza delle feature**.

Passeremo quindi alle spiegazioni che riguardano una risposta sola, le
**spiegazioni locali**, e ne vedremo tre. La prima costruisce, lì attorno al
caso da spiegare e solo lì, un modellino semplice che imita quello vero: una
copia leggibile buona in quel punto, che si chiama **surrogato locale**; il
metodo che la costruisce si chiama **LIME**, ed è quello che disegnò le macchie
sulla neve. La seconda spartisce fra le colonne il merito di una risposta. La
regola per farlo non nasce nell'informatica: viene da un problema del 1953, come
dividere in modo equo il guadagno di un'impresa fra i soci che ci hanno
lavorato. Quelle quote si chiamano **valori di Shapley**, e il modo di
calcolarle in fretta si chiama **SHAP**. La terza
risponde alla domanda più pratica di tutte, «che cosa sarebbe dovuto essere
diverso perché la risposta cambiasse?», e sono le spiegazioni
**controfattuali**.

Chiuderemo dentro le reti profonde. Lì la domanda diventa quanto ogni pezzo
dell'ingresso, ogni singolo pixel, ha contribuito a una risposta: quella quota
di merito si chiama **attribuzione**, e la cartina che la disegna sopra la foto
(le macchie dell'inizio) si chiama mappa di **salienza**. Vedremo che quelle
mappe promettono più di quanto mantengano, e la stessa domanda la rivolgeremo
ai pesi di attenzione che abbiamo visto nascere nel capitolo sui Transformer,
che sembrano una spiegazione già pronta e non lo sono. E finiremo con il
tentativo più ambizioso e più giovane, quello di smontare una rete pezzo per pezzo come un
ingegnere apre un chip per capire che cosa fa ciascun componente: si chiama
**interpretabilità meccanicistica**.

Un filo, sopra a tutto, tiene insieme il capitolo con quello sull'**AI
responsabile**: aprire la scatola nera non è un vezzo accademico, ma il primo
passo per costruire sistemi di cui potersi fidare, e da poter contestare
quando sbagliano. Il rilevatore di lupi era stato truccato apposta, per
dimostrare quanto è facile non accorgersene: i modelli veri
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
  sapere cosa cambiare, all'ingegnere quali colonne pesano, all'ufficio che
  vigila che il sistema non discrimini.
- Tre domande ordinano tutti i metodi del capitolo: il modello si legge da sé
  o va interrogato **dopo**? vuoi capire il modello **intero** o **una risposta
  sola**? lo strumento serve un solo tipo di modello o va bene per
  **qualunque** modello? E una quarta, che decide chi ha ragione quando due
  metodi discordano: vuoi spiegare **il modello** o **il mondo**? Se due colonne
  dicono quasi la stessa cosa e il modello ne usa una sola, la prima domanda dà
  tutto a quella e zero all'altra, la seconda divide a metà. Nessuna delle due
  sbaglia: sono domande diverse.
- Una spiegazione può essere **convincente e falsa insieme** (il telecronista
  che spiega il tiro e non è mai stato nella testa del calciatore). «Mi
  convince» e «è vera» sono due cose diverse, e la prima senza la seconda è
  pericolosa: ci fa fidare di un modello che non lo merita. Per distinguerle
  non basta guardare: bisogna provare a far fallire la spiegazione.
- Il dibattito: si crede che un modello chiaro sia per forza più scarso di uno
  oscuro, ma sui dati a righe e colonne quello scambio spesso non c'è (sui
  fiori dell'esempio la foresta guadagna un punto e tre). Per le decisioni
  che pesano davvero (giustizia, sanità, credito) Cynthia Rudin dice quindi che
  è meglio usare un modello trasparente invece di appiccicare una spiegazione a
  una scatola nera. Su foto, testo e suoni, però, la scatola nera resta la più
  brava, e la spiegazione appiccicata dopo è l'unica finestra che abbiamo.
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
  si sta spiegando **il modello** o **il fenomeno**? (Due colonne quasi
  ridondanti di cui il modello ne usa una: la prima domanda attribuisce tutto a
  quella usata, la seconda ripartisce.)
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
