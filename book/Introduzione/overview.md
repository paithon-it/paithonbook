---
titolo_ricerca: "Intelligenza artificiale: dalle origini a oggi"
descrizione: "Che cos'è l'intelligenza artificiale, da ELIZA a oggi: che cosa
  sono un algoritmo e le regole imparate dai dati, e perché il campo è esploso
  adesso."
---
# Introduzione

```{image} ../figures/aperture/introduzione.png
:class: pt-apertura only-light
:width: 100%
:alt: Un cilindro da prestigiatore da cui, al posto del coniglio, escono degli ingranaggi.
```

```{image} ../figures/aperture/introduzione-scura.png
:class: pt-apertura only-dark
:width: 100%
:alt: Un cilindro da prestigiatore da cui, al posto del coniglio, escono degli ingranaggi.
```

Joseph Weizenbaum era un informatico tedesco emigrato negli Stati Uniti e
professore al MIT. Nel 1966 presentò al mondo ELIZA, il capostipite dei
*chatbot*, cioè un programma con cui si poteva conversare per iscritto,
digitando una frase e leggendo la risposta. L'articolo che lo descrive si apre
così {cite}`weizenbaum1966eliza`:

>Si dice che spiegare sia dissolvere l'incanto.[^spiegare-via]

[^spiegare-via]: «It is said that to explain is to explain away»,
    nell'originale. *To explain away* è un verbo frasale che l'italiano non
    ha: vuol dire spiegare una cosa fino a toglierle ogni mistero, e con
    quello ogni importanza. In italiano quel verbo pretende un oggetto che
    l'inglese sottintende, e l'oggetto viene dallo stesso paragrafo
    dell'articolo, dove Weizenbaum scrive che appena il programma è
    smascherato «its magic crumbles away».

Nei programmi che sembrano intelligenti, osservava, questa massima si compie
alla perfezione: finché il meccanismo resta nascosto la macchina appare
prodigiosa; appena qualcuno lo spiega, l'incanto si sgretola. Il meccanismo di
ELIZA sta in poche righe.

Il programma vero e proprio non imitava nessuno. Era un motore che
riconosceva schemi, cioè pezzi di frase fatti a stampo, del tipo «mi sento
___» o «mia madre ___». A stabilire la parte da recitare era un *copione*, un
foglio di regole scritto a parte che si poteva cambiare senza toccare il
programma.

Il copione più celebre si chiamava DOCTOR e gli faceva sostenere la
conversazione di uno psicoterapeuta: riconosceva nella frase dell'utente uno
schema noto e gliela rigirava addosso. Ad esempio, «Mi sento infelice»
diventava «pensa che venire qui la aiuterà a non sentirsi infelice?», e alla
parola «madre» rispondeva «mi parli della sua famiglia». Il programma non
capiva niente, e la sua memoria era una manciata di frasi dette prima
dall'utente, tenute da parte e ripescate quando nella frase nuova non c'era
nessuno schema a cui agganciarsi.

Eppure Weizenbaum voleva dimostrare il contrario di quello che ottenne: far
vedere quanto fosse superficiale la comunicazione fra uomo e macchina. Rimase
sgomento davanti al numero di persone che al suo programma attribuivano
sentimenti umani. La sua stessa segretaria, che l'aveva visto lavorare al
programma per mesi e sapeva benissimo che era soltanto un programma, gli chiese
di uscire dalla stanza per poterci parlare in privato
{cite}`weizenbaum1976computer`.

Il suo programma era soltanto una lista di istruzioni, o c'era qualcosa di
più? E se era un semplice programma, perché attribuirgli una parola così ricca
di significato come l’*intelligenza*?

Che cos'è l'intelligenza artificiale? In inglese si
dice *Artificial Intelligence*, e la sigla **AI** si legge
dappertutto. Ogni anno la si perfeziona, e ogni anno sembra sfuggire un po’ di
più a una definizione precisa.

Su ELIZA la risposta è breve e deludente: no, era soltanto una lista di
istruzioni. Per altri programmi non basta, e dire che cosa sia l'intelligenza
artificiale richiede più spazio: bisogna prima vedere che cosa distingue un
programma scritto riga per riga da uno che ricava le proprie regole dagli
esempi.

In ELIZA la separazione fra il motore e il foglio delle regole va già in quella
direzione. Le regole, infatti, sono un testo che il programma legge, e un testo
si può produrre anche in modi diversi dallo scriverlo a mano. Ma nel 1966 una
persona scriveva ancora ogni regola, una per una. [L'apprendimento automatico
comincia quando nessuno scrive più quel foglio e le regole si ricavano dagli
esempi]{.pt-matita}.

In questo capitolo ripercorriamo le origini dell'intelligenza artificiale, dai
sillogismi di Aristotele al seminario di Dartmouth e ai due inverni della
disciplina. Definiamo poi che cos'è un algoritmo, partendo da quello di
Euclide, e vediamo perché per molti compiti le regole si ricavano dagli esempi
invece di essere scritte a mano. Riprendiamo quindi la funzione obiettivo,
cioè il punteggio che l'addestramento fa salire, e spieghiamo perché i
risultati di oggi sono arrivati così tardi. Seguono il primo programma in
Python, i limiti di ciò che una macchina può calcolare e l'intelligenza
artificiale nei robot.

## Le origini dell'intelligenza artificiale

L'idea di costruire qualcosa che somigli a noi è vecchia quanto il pensiero. La
scienza che ci prova, invece, è giovane: nasce nel decennio successivo
alla seconda guerra mondiale, quando per la prima volta ci sono delle macchine
su cui provare.

Nella vita di tutti i giorni è entrata due volte. Negli anni Dieci (il decennio
2010–2019) ci è entrata senza farsi notare, dentro il traduttore automatico, i
suggerimenti di un negozio online, il riconoscimento dei volti nelle
fotografie; la usavano tutti, ma quasi nessuno la chiamava per nome. Dal
novembre 2022, con ChatGPT, è diventata invece qualcosa con cui si parla
apposta, e in pochi mesi chiunque ne conosceva il nome.

Il salto che il pubblico ha percepito alla fine del 2022 era cominciato, in
realtà, cinque anni prima, in un articolo del 2017 intitolato *Attention Is All
You Need*, «l'attenzione è tutto ciò che serve» {cite}`vaswani2017attention`.
Da lì è nata la famiglia di programmi con cui oggi si conversa, descritta nel
{doc}`capitolo sui Transformer </Transformers/overview>`.

Nel 2022 si aggiunsero due cose: l'ultimo tratto dell'addestramento, fatto con
istruzioni e giudizi di persone {cite}`ouyang2022training`, e il luogo in cui la
tecnologia si incontrava. Prima stava nascosta dentro servizi che facevano altro
(traduci questa pagina, suggerisci un film), e nessuno ci parlava; dal 2022 ci
si scrive direttamente, in una finestra di conversazione.

Le domande da cui nasce l'AI, però, sono più antiche dei calcolatori: per
duemila anni se le sono poste filosofi e matematici, come ricostruiscono
Russell e Norvig nel loro manuale {cite}`russell2020artificial`.

Aristotele, nel IV secolo a.C., è il primo di cui ci sia rimasto un sistema
scritto delle regole del ragionamento corretto. Si regge sui **sillogismi**,
cioè catene di tre frasi in cui la terza discende dalle prime due per la sola
forma in cui sono scritte.

L'esempio che conoscono tutti (probabilmente di Sesto Empirico, non di
Aristotele) è «tutti gli uomini sono mortali; Socrate è un uomo; quindi Socrate
è mortale». Si noti che per trarre la conclusione non serve sapere chi fosse
Socrate: basta la struttura delle prime due frasi. In questo senso le
conclusioni si ottengono «meccanicamente», e il sillogismo è il più antico
tentativo giunto fino a noi di scrivere le regole del pensiero come si
scriverebbero quelle di un gioco.

Dopo Aristotele bisogna aspettare quasi due millenni, fino al Seicento, quando
compaiono le prime macchine che fanno di conto. Hobbes ipotizzò che ragionare
fosse una specie di calcolo, come se dentro la testa eseguissimo addizioni e
sottrazioni. Pascal costruì una di quelle macchine, la Pascalina,[^schickard] e
annotò che la macchina aritmetica produce effetti che sembrano più vicini al
pensiero di tutto ciò che fanno gli animali, ma niente che permetta di
attribuirle una volontà, che invece gli animali hanno.

Cartesio, infine, tracciò una distinzione netta fra mente e materia, sostenendo
che il pensiero è fatto di una sostanza diversa dal corpo e non obbedisce alle
stesse leggi. Una macchina, invece, è materia e nient'altro: se Cartesio avesse
ragione, nessuna macchina potrebbe mai avere una mente.

[^schickard]: Non è la prima di cui si abbia notizia. Wilhelm Schickard ne
    descrisse una a Keplero nel 1623 e di nuovo l'anno dopo, in due lettere che
    sono quasi tutto quello che ce ne resta: la macchina andò distrutta in un
    incendio e la sua esistenza si riscoprì solo tre secoli dopo.

I filosofi hanno posto quasi tutte le domande dell'AI, ma il passaggio a una
scienza vera e propria richiedeva qualcosa che i filosofi non davano. Una
macchina, infatti, opera su numeri e simboli, non su significati; perché un'idea
le arrivi, bisogna prima ridurla a un calcolo, e Aristotele aveva scritto quali
passaggi fossero leciti, non come farli fare a una macchina.

Servivano strumenti che a metà del Novecento la matematica aveva già
{cite}`russell2020artificial`. In particolare, la logica era diventata un
calcolo, con simboli e regole come l'algebra (Boole nel 1847, Frege nel 1879).
La probabilità permetteva di ragionare anche quando l'informazione è incerta.
E una branca nuova, la teoria del calcolo, stabiliva che cosa un procedimento
meccanico sa calcolare e che cosa no. Questa teoria ha una {doc}`sezione tutta
sua <calcolabile>`, perché quei limiti riguardano anche i programmi che
imparano.

Accanto a questi strumenti c'erano le discipline nate per far prendere
decisioni. La **ricerca operativa** studia come scegliere il piano migliore
quando le risorse sono poche (quali camion mandare su quali strade); la
**teoria del controllo** come tenere un sistema sulla rotta voluta,
correggendolo di continuo, ed è la matematica del termostato e del pilota
automatico; la **teoria dei giochi** come decidere quando dall'altra parte c'è
qualcuno che decide a sua volta.

Viste da vicino, queste discipline facevano in fondo la stessa cosa.
Ognuna inventava un punteggio che dice come sta andando (quanto costa
il percorso dei camion, di quanto la temperatura si scosta da quella voluta,
quanto si guadagna in una partita). Il costo e lo scarto si vogliono piccoli, il
guadagno grande, ma il principio è lo stesso: cercare le decisioni che spingono
il punteggio dalla parte giusta. Questo punteggio si chiama
**funzione obiettivo**.

Russell e Norvig chiamano questo schema il **modello standard** dell'AI:
costruire sistemi che agiscono «nel modo migliore possibile», dove a definire
il migliore è la funzione obiettivo che si fornisce
{cite}`russell2020artificial`.

L'AI e la teoria del controllo restano comunque due campi distinti, perché
l'AI nacque in parte per superare i limiti dell'altra
{cite}`russell2020artificial`. Quella matematica
sapeva tenere in rotta un sistema descritto da pochi numeri che cambiano nel
tempo (la velocità, la temperatura, l'angolo del timone), e si fermava lì.
Capire una frase, riconoscere che cosa c'è in una fotografia, decidere in che
ordine fare le cose per arrivare a un obiettivo: erano problemi che si ponevano
completamente fuori dal suo campo d'azione.

Nel 1950 Alan Turing propose di
sostituire la domanda «le macchine possono pensare?» con un esperimento
concreto, il *gioco dell'imitazione*, oggi noto
come **test di Turing**: se conversando a distanza non riesci a capire se
dall'altra parte c'è una persona o un programma, la domanda di partenza si può
mettere da parte {cite}`turing1950computing`.[^gioco-imitazione]

[^gioco-imitazione]: Nella forma in cui Turing lo introduce il gioco non
    riguarda affatto le macchine: i giocatori sono tre, un uomo, una donna e un
    interrogante che deve indovinare chi dei due sia l'uomo. La macchina entra
    subito dopo, e la domanda diventa: che succede se a fare
    quella parte è un calcolatore? La versione «umano contro programma» che
    tutti chiamano test di Turing è la lettura moderna, non l'esperimento
    originale.

Il gioco misura soltanto se la conversazione è credibile, e non dice se
dall'altra parte qualcuno abbia capito. ELIZA, ad esempio, non comprendeva
nulla, eppure bastava a far credere il contrario a chi ci parlava. Oggi questo
fenomeno si chiama **effetto ELIZA**; ne riparla
{doc}`Dialogo e chatbot </NaturalLanguageProcessing/dialogo-chatbot>`.

Il termine *intelligenza artificiale* compare per la prima volta nel 1955,
nella proposta con cui John McCarthy, Marvin Minsky, Nathaniel Rochester e
Claude Shannon chiedevano i fondi per un seminario estivo al Dartmouth College;
è quel seminario, nell'estate del 1956, a essere ricordato come l'atto di
nascita ufficiale della disciplina. La proposta partiva da una congettura: ogni
aspetto dell'apprendimento, o di qualunque altra caratteristica
dell'intelligenza, si può in linea di principio descrivere con tanta precisione
da poterlo far simulare a una macchina {cite}`russell2020artificial`.

Il compito del nuovo campo era rendere automatiche attività che fino ad allora
sapeva fare soltanto una persona: usare il linguaggio, riconoscere immagini,
giocare a scacchi, dimostrare teoremi, guidare un'automobile. L'intelligenza
artificiale tocca quindi, almeno in potenza, ogni attività della mente, eppure
è una delle scienze più giovani. Quando nasce, infatti, la fisica moderna ha
quasi tre secoli di storia, dai *Principia* di Newton (1687), e i calcolatori
elettronici sono in circolazione da una decina d'anni.

Fra il seminario di Dartmouth e i risultati che oggi diamo per scontati, però,
non c'è una linea che sale. Ci sono due lunghi **inverni** (*AI winters*),
periodi in cui le promesse non mantenute fecero tagliare i finanziamenti.

Il primo arriva negli anni Settanta. Le promesse del decennio precedente (una
macchina che traduce, che dimostra teoremi, che vede) erano andate a scadenza
senza essere mantenute. Nel 1973 un rapporto commissionato al matematico James
Lighthill dallo Science Research Council britannico stroncò il campo, e di
conseguenza il governo britannico decise di togliere i fondi alla ricerca
sull'AI in tutte le università tranne due {cite}`russell2020artificial`.

Il secondo inverno arriva a fine anni Ottanta, quando si sgonfia il mercato dei
**sistemi esperti**, cioè programmi che racchiudevano in migliaia di regole
scritte a mano il sapere di uno specialista. Dentro il loro campo ristretto
funzionavano; appena fuori si rompevano, perché ogni caso non previsto
chiedeva una regola nuova. Chi ci lavorava lo chiamava il collo di bottiglia
dell'acquisizione della conoscenza: il sapere c'era, nella testa degli
specialisti, ma travasarlo in regole e tenerle aggiornate costava più di
quanto rendesse.

A questo secondo inverno è seguito il modo di lavorare opposto: ricavare le
regole dagli esempi invece di scriverle a mano. Oltre ai due inverni di tutto
il campo, le reti neurali ne hanno avuto uno tutto loro, che il
{doc}`capitolo che porta il loro nome </RetiNeurali/overview>` racconta per
esteso.

Quell'inverno comincia nel 1969, con *Perceptrons* di Minsky e Papert, che
dimostrava i limiti del percettrone, la versione più semplice di quelle reti
{cite}`minsky1969perceptrons`. Si intreccia poi con l'inverno degli anni
Settanta, perché i tagli seguiti al rapporto Lighthill colpirono tutto il
campo.

L'inverno delle reti neurali si scioglie nel 1986, quando un articolo su
*Nature* di David Rumelhart, Geoffrey Hinton e Ronald Williams mostra come
addestrare una rete a più strati. Il metodo si chiama retropropagazione
dell'errore (*backpropagation*): calcola, per ciascuno dei numeri regolabili
della rete, di quanto cambierebbe l'errore ritoccandolo di poco, e quindi in
che verso correggerlo. Paul Werbos lo aveva formulato dodici anni prima, nella
sua tesi di dottorato {cite}`rumelhart1986learning,werbos1974beyond`.

Le regole scritte a mano, intanto, facevano cose notevoli, e non solo in
laboratorio. Una delle auto che si guidavano da sole negli anni Novanta girava
sulle strade italiane, su una Lancia Thema. A partire dal 1996, all'Università
di Parma, un gruppo di ricercatori e ingegneri guidato da Alberto Broggi
costruì ARGO.

L'auto vedeva con due telecamere in bianco e nero montate in coppia, come i
nostri due occhi, e un computer di bordo del tutto ordinario per l'epoca
prendeva le sue decisioni: un Pentium 200 MMX, un processore incomparabilmente
più lento di quello del telefono che oggi tieni in tasca. ARGO sapeva sterzare
da sola, restare al centro della corsia e accorgersi degli ostacoli davanti. Nel
giugno 1998, nella prova «MilleMiglia in Automatico», percorse circa 2.000 km
sulle autostrade italiane con lo sterzo governato dal programma per circa il 94%
del tragitto; acceleratore e freno restavano al guidatore.

ARGO non fu la prima: negli stessi anni la VaMP di Ernst Dickmanns girava
sulle autostrade europee e la Navlab 5 della Carnegie Mellon University, in
Pennsylvania, attraversava gli Stati Uniti. Colpisce per l'economia
dei mezzi: dove la VaMP portava armadi di elettronica costruita apposta, ARGO
lavorava con due telecamere in bianco e nero e un personal computer da
negozio.

Dentro ARGO, però, non c'era niente che avesse imparato qualcosa. Qualcuno che
sapeva già che aspetto ha una striscia bianca su un asfalto grigio aveva
scritto a mano, una per volta, le regole per riconoscere la corsia e gli
ostacoli.

Nel 2010 il gruppo di Broggi è riuscito a far guidare da soli dei furgoni
elettrici dall'Italia alla Cina. La sfida, chiamata VIAC (VisLab
Intercontinental Autonomous Challenge), prevedeva più di tredicimila chilometri
da Parma a Shanghai, con quattro furgoni che viaggiavano a coppie, due in
marcia mentre gli altri due si ricaricavano, e una regola d'ingaggio precisa.

In ogni coppia i furgoni procedevano in fila: quello di testa apriva la strada
e ogni tanto un umano interveniva, per scegliere il percorso o per risolvere le
situazioni che il sistema non sapeva gestire; quello dietro seguiva il primo in
completa autonomia. Non furono quindi tredicimila chilometri senza nessuno al
volante. I furgoni che seguivano ne percorsero in tutto circa 8.300 in modalità
autonoma, e nel 2010 era comunque un risultato senza precedenti.

## Che cos'è un algoritmo

Un **algoritmo** è una ricetta: una lista finita di istruzioni precise, che non
lasciano dubbi su che cosa fare e che, eseguite nell'ordine giusto, portano a un
risultato in un numero finito di passi. La parola viene dal nome di
al-Khwarizmi, un matematico persiano del IX secolo, ma la cosa è più antica.

Il primo algoritmo non banale è ritenuto quello di Euclide
{cite}`russell2020artificial`, che oltre due millenni fa trovò il modo di
calcolare il massimo comune divisore di due numeri. Applicato a $12$ e $8$, per
esempio, restituisce $4$. Il {doc}`primo programma <massimo_comune_divisore>` lo
scrive in Python, dopo averne studiato il funzionamento.

`````{tab} Elementare
Il **massimo comune divisore** (MCD) di $12$ e $8$ è il numero più grande che
li divide entrambi senza resto. Puoi immaginarlo così: hai un pavimento
rettangolare di $12 \times 8$ mattonelle e vuoi ricoprirlo con piastrelle
quadrate, tutte uguali e senza tagliarne nessuna; la piastrella più grande che
funziona è quella da $4 \times 4$.

Il trucco di Euclide per trovare il MCD è elegante: dividi il numero grande per
il piccolo e guarda il resto. Ad esempio, $12$ diviso $8$ dà resto $4$; ora
ripeti con $8$ e $4$, e ottieni resto $0$. Appena compare il resto zero,
l'ultimo numero *per cui* hai diviso (qui $4$) è il MCD. Niente elenchi di
divisori, niente tentativi: due divisioni e hai finito. E se il resto
viene zero già alla prima, va bene lo stesso, hai solo finito prima:
$12$ diviso $6$ dà resto $0$, e il MCD è $6$.

Perché il trucco funziona lo racconta il pavimento. Dal rettangolo di
$12 \times 8$ ritaglia il quadrato più grande che ci sta, $8 \times 8$: avanza
una striscia di $8 \times 4$, e la larghezza della striscia è proprio il resto
della divisione. Una piastrella che copre senza tagli il pavimento intero
copre senza tagli anche la striscia, e viceversa. Quindi [cercare la
piastrella più grande per $12$ e $8$, o cercarla per $8$ e $4$, è lo stesso
problema, solo più piccolo]{.pt-matita}.

Dalla striscia ritaglia poi due quadrati di $4 \times 4$: non avanza niente, e
quando non avanza niente il lato del quadrato è la piastrella cercata.

Il bello è che le divisioni restano poche anche quando i numeri crescono: per
due numeri lunghi cento cifre ne bastano al massimo cinquecento, cinque per
cifra, mentre provare i divisori uno per uno ne chiederebbe fino a un $1$
seguito da cento zeri. Quel numero è enorme: gli atomi dell'intero universo
osservabile si stimano intorno a un $1$ seguito da ottanta zeri, cento miliardi
di miliardi di volte meno.

Poche, però, non vuol dire poco tempo. Le divisioni crescono con la
*lunghezza* dei numeri, al più cinque per ogni cifra in più (in media meno di
due), e ciascuna, su numeri da
cento cifre, richiede molto più lavoro che su numeri da una cifra.
`````

`````{tab} Superiore
Consideriamo due interi $a \ge 0$ e $b > 0$. L'algoritmo sfrutta l'identità
$\mathrm{MCD}(a, b) = \mathrm{MCD}(b,\, a \bmod b)$, dove $a \bmod b$ è il
resto della divisione intera, $0 \le a \bmod b < b$; il caso base è
$\mathrm{MCD}(a, 0) = a$ per $a > 0$, e $\mathrm{MCD}(0, 0)$ si pone uguale a
$0$ per convenzione. Poiché il secondo argomento scende strettamente a ogni
passo restando non negativo, il ciclo termina. La correttezza segue dal fatto
che ogni divisore comune di $a$ e $b$ divide anche $a \bmod b = a - \lfloor a/b
\rfloor\, b$ e che, viceversa, ogni divisore comune di $b$ e $a \bmod b$ divide
$a = \lfloor a/b \rfloor\, b + (a \bmod b)$. Ne segue che [i due insiemi di
divisori comuni coincidono, e quindi il massimo è invariante a ogni
passo]{.pt-matita}. Ad esempio, per $a=12$ e $b=8$ la successione delle coppie
è $(12, 8) \to (8, 4) \to (4, 0) \Rightarrow 4$.

Il numero di passi $N$ è $O(\log \min(a, b))$. Più precisamente, se
$a > b \ge 1$ vale $b \ge F_{N+1} \ge \varphi^{N-1}$, con $F_k$ i numeri di
Fibonacci e $\varphi = (1+\sqrt{5})/2$ (teorema di Lamé, 1844). Ne segue
$N \le 1 + \log_\varphi b$, cioè al più cinque passi per cifra decimale di
$b$; il caso peggiore è quello di due numeri di Fibonacci consecutivi, con
$\log_\varphi 10 \approx 4{,}79$ passi per cifra. In media i passi sono circa
$\tfrac{12\ln 2}{\pi^2}\ln b$, poco meno di due per cifra
{cite}`knuth1997seminumerical`.

È per questa efficienza (non solo per l'età) che l'idea di Euclide è ancora oggi
nelle librerie standard di quasi tutti i linguaggi (`math.gcd` in Python,
`std::gcd` in C++ dal 2017; la libreria standard del C non ne ha una).

Osserviamo che la stima riguarda i *passi*, e che il logaritmo è preso sul
*valore* di $\min(a, b)$; su due numeri di $n$ cifre, quindi, i passi sono
$O(n)$, non $O(\log n)$. Inoltre, su interi lunghi ogni passo richiede una
divisione, il cui costo è proporzionale alle cifre del quoziente moltiplicate
per quelle del divisore; la somma delle cifre di tutti i quozienti resta però
$O(n)$, perché il prodotto dei quozienti non supera il numero di partenza. Si
ottiene così il tempo totale $O(n^2)$, e non l’$n^3$ che si avrebbe
moltiplicando il numero dei passi per il costo massimo di una divisione.

Le librerie, però, non eseguono questo ciclo così com'è. CPython usa una
variante, l'algoritmo di Lehmer, che lavora sulle cifre di testa in precisione
singola e risparmia molte divisioni su interi lunghi (stessa classe $O(n^2)$,
costante molto migliore); librerie come GMP, per numeri enormi, passano ad
algoritmi subquadratici di tipo *half-GCD* (Knuth, 1970; Schönhage, 1971), con
costo $O(M(n)\log n)$, dove $M(n)$ è il costo di una moltiplicazione fra numeri
di $n$ cifre.
`````

La {numref}`fig-euclide-scende` mostra la stessa discesa su un'altra coppia di
numeri, $60$ e $48$, quelli su cui lavora il primo programma.

```{figure} ../figures/euclide-scende.svg
:name: fig-euclide-scende
:alt: "L'algoritmo di Euclide applicato a 60 e 48. Ogni riga è una tappa, con tre scatole (dividendo, divisore, resto) e accanto la divisione per esteso: 60 = 1 × 48 + 12, poi 48 = 4 × 12 + 0. Due frecce per riga portano il divisore al posto del dividendo e il resto al posto del divisore, così la coppia (60, 48) diventa (48, 12) e poi (12, 0). Quando il divisore è zero il ciclo si ferma, e il 12 rimasto a sinistra è il massimo comune divisore."
:width: 92%

La discesa di Euclide su 60 e 48: a ogni passo il divisore prende il posto del
dividendo e il resto quello del divisore, e i numeri si accorciano. Quando il
divisore arriva a zero ci si ferma, e quello rimasto a sinistra è il massimo
comune divisore. Due divisioni in tutto, senza provare nemmeno un candidato.
```

## Quando le regole non si scrivono

L'algoritmo di Euclide è una ricetta, e le ricette hanno un autore: qualcuno
ha capito come si fa e ne ha scritto i passi. Per duemila anni ogni algoritmo
è stato così, e così è ancora la maggior parte dei programmi che usi ogni
giorno: chi li ha scritti sapeva già che cosa dovevano fare, riga per riga.

Il limite delle ricette si vede bene su una fotografia. Per un computer una
fotografia è un rettangolo di puntini, detti **pixel**, e ogni pixel è una terna
di numeri che dicono quanto rosso, quanto verde e quanto blu ci sono lì. Una
foto da telefono ne ha qualche milione.

Adesso prova a scrivere la ricetta per riconoscere un gatto in una
fotografia. Non può parlare di orecchie a punta: deve dire che cosa fare con
quei milioni di numeri, e deve funzionare anche col gatto di spalle, in
controluce, mezzo nascosto dietro una sedia. Nessuno è mai riuscito a scriverne
una che funzioni in tutti i casi, e non per pigrizia: non conosciamo quella
ricetta, anche se gli occhi e il cervello la eseguono in un decimo di secondo.

E allora si cambia strada. Invece di scrivere le regole, si raccolgono gli
**esempi** (migliaia di fotografie con scritto accanto «gatto» oppure «non
gatto») e si lascia che sia il programma a trovare che cosa distingue le une
dalle altre. In altri termini, [nessuno scrive le regole: emergono dai
dati]{.pt-matita}. In questo senso un programma *impara*; per lo stesso motivo,
in questo modo di lavorare i dati contano quanto il codice.

Contro questa idea c’è un’obiezione vecchia di quasi due secoli, e un pezzo
della sua storia si è svolto in Italia, a Torino. Nel 1840 il matematico
inglese Charles Babbage vi presentò il progetto della sua *macchina analitica*,
un calcolatore fatto di ingranaggi che avrebbe dovuto eseguire in fila le
istruzioni scritte su schede di cartone perforate, un po’ come oggi un computer
esegue un programma. Non fu mai costruita.

A descriverla, in un articolo in francese, fu Luigi Federico Menabrea, un
ufficiale del genio che insegnava all’Accademia militare di Torino e che quasi
trent’anni dopo sarebbe diventato capo del governo italiano. La matematica
inglese Ada Lovelace tradusse l’articolo e lo accompagnò con note lunghe più del
doppio, in cui mostrava fra l’altro come istruire la macchina, passo per passo,
a eseguire un calcolo lungo. In una di quelle note scrisse che la macchina «non
pretende affatto di creare qualcosa da sé: può fare qualunque cosa noi sappiamo
ordinarle di eseguire» {cite}`menabrea1843sketch`.

Nello stesso articolo del 1950 in cui proponeva il suo test, Turing mise quella
frase fra le obiezioni alle macchine pensanti e la chiamò «obiezione di Lady
Lovelace» (Lady è il titolo nobiliare: è la stessa Ada Lovelace).

Qualche pagina più avanti ci tornò sopra con una proposta che è l’idea degli
esempi detta con un’altra immagine: invece di costruire una mente adulta, con
tutte le regole già scritte dentro, costruirne una da bambino, «qualcosa come un
quaderno appena comprato in cartoleria», poco meccanismo e molti fogli bianchi,
e poi educarla {cite}`turing1950computing`. Il quaderno bianco è il programma
prima di vedere le fotografie; l’educazione sono le migliaia di fotografie con
scritto accanto «gatto» oppure «non gatto».

La fase in cui il programma guarda gli esempi e aggiusta sé stesso si chiama
**addestramento**;
il risultato, cioè il programma già aggiustato e pronto a rispondere su
fotografie che non ha mai visto, si chiama **modello**. Il «gatto» scritto
accanto a ciascuna foto si chiama **etichetta**.

Le fotografie etichettate a mano, però, sono il caso più facile da
raccontare, non l'unico né il più diffuso. La risposta giusta può essere già
dentro il materiale che si ha. In un testo, ad esempio, si nasconde la parola
che viene dopo e si chiede al programma di indovinarla; a quel punto le
etichette sono tante quante le parole del testo e non costano niente, perché
le fornisce il testo stesso.

Così si fa la prima fase dell'addestramento dei modelli di cui oggi si parla
di più, e dunque «imparare dagli esempi» non vuol dire per forza che qualcuno
abbia etichettato qualcosa.

Da qui i tre nomi che si incontrano più spesso:

- il **machine learning** (apprendimento automatico) è l'idea di
  ricavare le regole dagli esempi invece di scriverle a mano;
- il **deep learning** (apprendimento profondo) è il machine learning fatto
  con le **reti neurali** (il nome viene dal cervello, che le ha ispirate alla
  lontana), cioè programmi composti da molti passaggi elementari disposti in
  fila per **strati**. Ogni strato trasforma i numeri dello strato precedente in
  una descrizione un po’ più astratta; ad esempio, in una rete che guarda
  fotografie si passa dai pixel ai bordi, dai bordi alle forme, dalle forme al
  gatto. «Profondo» vuol dire soltanto che gli strati sono tanti, uno sopra
  l'altro, e nessuno scrive a mano che cosa ciascuno debba cercare;
- il **reinforcement learning** (apprendimento per rinforzo) è il caso in cui
  gli esempi giusti non esistono e il programma impara dalle conseguenze delle
  proprie azioni: prova, riceve un punteggio, riprova. È il caso che
  incontreremo nella {doc}`sezione sulla robotica <applicazioni>`, con il
  robot che impara a camminare.

Il disegno più diffuso rappresenta l'intelligenza artificiale, il machine
learning e il deep learning come tre cerchi uno dentro l'altro, e per questi tre
è giusto: il deep learning è un modo di fare machine learning, e il machine
learning un modo di fare intelligenza artificiale.

Il reinforcement learning è un altro tipo di machine learning, diverso
dall'imparare da esempi con la risposta scritta accanto: l'unico giudizio è un
punteggio. Può usare le reti profonde (e allora si parla di *deep reinforcement
learning*) o farne a meno. L'intelligenza artificiale, infine, è più ampia di
tutto il machine learning: comprende anche i programmi che ragionano su regole
scritte a mano o che cercano fra le mosse possibili, cioè la sua parte classica,
quella dei sistemi esperti del secondo inverno.

Ne viene una definizione pratica, e provvisoria: l'intelligenza artificiale si
occupa dei compiti per cui nessuno sa scrivere a mano una ricetta che
funzioni nel mondo reale.

Questa definizione esclude la lavatrice. Anche una lavatrice decide da sola
quando fermare il risciacquo, ma per quella decisione la ricetta c'è, sta in
poche righe e funziona: un tecnico ha stabilito quale sensore leggere e sopra
quale soglia fermarsi. Riconoscere un gatto, tradurre una frase, far camminare
un robot su un terreno qualunque: lì la ricetta non esiste, e va fatta
emergere.

D'altra parte, la definizione include i sistemi esperti, che provavano lo stesso
a scriverla, a mano, una regola alla volta. Il loro fallimento è una delle
ragioni per cui oggi le regole si ricavano dagli esempi.

Torna così la domanda su ELIZA. Quel programma non aveva niente di più di una
lista di istruzioni, e per questo il suo incanto si sgretola appena lo si
spiega. I programmi che imparano dagli esempi hanno qualcosa di più, ma è meno
misterioso e più scomodo di quanto si immagini: nessuno ha scritto le regole
che seguono e, nei modelli grandi, nessuno sa elencarle tutte, nemmeno chi li
ha costruiti.

Da lì vengono sia i risultati sia i guai. I modi per guardare comunque là
dentro sono il tema di {doc}`Interpretabilità </Interpretabilita/overview>`, i
danni che quei programmi possono fare quello di {doc}`AI responsabile
</AIResponsabile/overview>`.

## Il punteggio da far salire

Le regole emergono dai dati, ma qualcosa deve pur dire al programma se sta
andando meglio o peggio. È di nuovo la funzione obiettivo, la stessa con cui
la ricerca operativa giudicava un piano e la teoria del controllo misurava lo
scarto dalla rotta. Questa volta, però, il punteggio si vuole alto, e a cercare
le decisioni che lo fanno salire è l'addestramento, non un ingegnere.

L'addestramento tocca una cosa sola. Dentro un programma che impara c'è un
elenco di numeri, spesso lunghissimo, detti **parametri**. Tutti insieme si
indicano con la lettera greca $\theta$ (theta), e la funzione obiettivo, che da
loro dipende, si scrive $J(\theta)$. L'addestramento cambia soltanto
$\theta$; il comportamento viene dietro.

`````{tab} Elementare
A un robot aspirapolvere assegni un punteggio: $+1$ per ogni
briciola raccolta, $-1$ per ogni urto contro un mobile. Ad esempio, se in un
giro di salotto raccoglie $30$ briciole e sbatte $5$ volte, totalizza
$30 - 5 = 25$ punti. Quel modo di dare i punti è la sua funzione obiettivo: non
gli spieghiamo *come* pulire, gli diciamo solo *che punteggio* vogliamo veder
salire.

Dentro, il robot ha una manciata di manopole, i suoi parametri: quanto sterzare
quando il sensore davanti si accende, quanto rallentare vicino a un mobile,
quanto insistere dove il tappeto è sporco. Si girano quelle, si guarda il
punteggio, si girano ancora. Qualunque ritocco dei parametri che porti il
totale medio sopra $25$ è un miglioramento.

«Agire nel modo migliore possibile» significa, alla fine, scegliere le
mosse che rendono quel punteggio il più alto possibile. Buona parte
dell'intelligenza artificiale moderna, sotto sotto, funziona così: [si sceglie
un numero da massimizzare (o un errore da rendere minimo) e si lascia che sia
la macchina a scoprire come]{.pt-matita}.

Un giro solo, però, dice poco. Il robot, infatti, vale per come pulirà le
stanze che deve ancora vedere, come il salotto coi mobili spostati o la casa di
un amico. Quei giri futuri non si possono misurare oggi, per cui si misurano i
giri già fatti, si allena il robot su quelli e si spera che le stanze che
verranno somiglino a quelle già viste. Quanta fiducia meriti quella speranza è
la domanda al centro del {doc}`capitolo sul machine learning
</MachineLearning/overview>`.

C'è poi una crepa: quel punteggio lo scriviamo noi, e non è mai *esattamente*
la cosa che vogliamo. Un aspirapolvere pagato a briciole raccolte,
se è abbastanza bravo, può scoprire che gli conviene rovesciare il cestino e
raccoglierle una seconda volta. Ha fatto il punteggio più alto e ha sporcato il
salotto: ha obbedito alla lettera tradendo l'intenzione. Il fenomeno ha un
nome, *reward hacking*, e lo racconta {doc}`Esplorazione e ricompensa
</DeepReinforcementLearning/esplorazione-e-ricompensa>`, nelle pagine
sull'imparare per tentativi ed errori con le reti neurali.
`````

`````{tab} Superiore
Formalmente, descriviamo il comportamento del sistema con dei parametri
$\theta$ e ne misuriamo la qualità con un’**utilità attesa**

$$
J(\theta) = \mathbb{E}_{\xi \sim p_\theta}\!\left[\, U(\theta, \xi) \,\right],
$$

dove $\xi$ è una situazione (un esempio, un episodio), $U(\theta, \xi)$ il
punteggio che il sistema vi ottiene e $p_\theta$ la distribuzione delle
situazioni che incontrerà, cioè i dati che vedrà o l'ambiente in cui agirà.
Nell'apprendimento supervisionato $p_\theta = p$ non dipende dai parametri, e
$\theta$ agisce solo attraverso $U$. Nel reinforcement learning, invece, le
azioni decidono quali stati si visitano, quindi $p_\theta$ dipende da
$\theta$, e questa dipendenza rende il gradiente di $J$ più difficile da
stimare. Massimizzare $J$ equivale a minimizzare la perdita attesa (*loss*)
$\mathcal{L}(\theta) = -J(\theta)$ e, assumendo che un ottimo esista, si cerca

$$
\theta^\star \in \arg\max_{\theta} J(\theta) = \arg\min_{\theta} \mathcal{L}(\theta),
$$

dove $\theta^\star$ è una configurazione ottima dei parametri.

La differenza fra *ottimizzare* e *imparare* sta tutta qui: l'attesa $J$ non si
sa calcolare, perché la distribuzione su cui è presa è quella dei casi futuri,
l'unica a cui in fase di addestramento non si ha accesso. In pratica si
massimizza la media su $n$ situazioni già raccolte,
$\hat{J}_n(\theta) = \frac{1}{n}\sum_{i=1}^{n} U(\theta, \xi_i)$, con le
$\xi_i$ estratte in modo indipendente dalla stessa $p$ (il caso
supervisionato), e si spera che il punto di massimo $\hat{\theta}$ abbia un
$J(\hat{\theta})$ alto.

Per un $\theta$ fissato, se $\mathbb{E}\,|U(\theta,\xi)| < \infty$, la legge
dei grandi numeri garantisce $\hat{J}_n(\theta) \to J(\theta)$. Per
$\hat{\theta}$, invece, questa garanzia non vale, perché è stato scelto
guardando proprio quei dati, e in media $\hat{J}_n(\hat{\theta})$ non è
inferiore a $J(\hat{\theta})$. Infatti
$\mathbb{E}\,\hat{J}_n(\hat{\theta}) \ge \mathbb{E}\,\hat{J}_n(\theta^\star) =
J(\theta^\star) \ge \mathbb{E}\,J(\hat{\theta})$. Tutto questo presuppone che i
dati futuri vengano dalla stessa $p$ di quelli raccolti; quando non è così si
parla di *dataset shift*, il tema della sezione sui {doc}`dati che cambiano
</MachineLearning/dati-che-cambiano>`.

Massimizzare $\hat{J}_n$ è la minimizzazione del rischio empirico
(Vapnik): $-\hat{J}_n$ è il rischio empirico, $-J$ il rischio, e la loro
distanza è il divario di generalizzazione, di cui si occupa la {doc}`teoria
dell'apprendimento </TeoriaApprendimento/overview>`.

Nei modelli differenziabili il massimo si cerca salendo lungo il gradiente,
$\theta \leftarrow \theta + \eta\,\nabla_\theta \hat{J}_n(\theta)$, con $\eta$
il tasso di apprendimento. In generale, però, $\hat{J}_n$ non è concava, e
la salita si ferma in un punto stazionario, che può essere un massimo locale e
non per forza quello globale. Il {doc}`capitolo sul machine learning
</MachineLearning/overview>` studia come misurare la distanza fra $\hat{J}_n$
e $J$, e quando fidarsene.

Questa è la cornice dell’**agente razionale** {cite}`russell2020artificial`,
cioè di un sistema che sceglie le azioni che massimizzano l'utilità attesa,
date le informazioni disponibili. Osserviamo che qui l'ottimizzazione agisce
sui parametri, non direttamente sulle azioni: $\theta$ determina il
comportamento del sistema, e la configurazione ottima dei parametri induce le
scelte migliori che la famiglia di modelli $\{f_\theta\}$ permette, sempre che
l'ottimizzazione trovi un ottimo globale.

Gran parte dei metodi di apprendimento sono variazioni su questo tema. Ad
esempio, la regressione minimizza un errore quadratico medio, la
classificazione una cross-entropy, il reinforcement learning massimizza una
ricompensa cumulata attesa. [Cambiano $\mathcal{L}$ e il modo di calcolare il
minimo, non lo schema]{.pt-matita}.

Le eccezioni sono poche, e mostrano due modi di uscire dal quadro. Le GAN
sostituiscono la minimizzazione di una funzione con l'equilibrio di un gioco fra
due reti in competizione, e allora la loss smette di dire se le cose stanno
andando bene.

I metodi non parametrici come il k-NN non hanno parametri da
stimare con l'addestramento e conservano i dati stessi. Si noti, però, che «non
parametrico» non vuol dire che non ci siano numeri da scegliere (ne restano,
come il numero di vicini), ma che ciò che il modello conserva cresce con i dati
invece di essere fissato in partenza.

A queste eccezioni si aggiunge un limite della cornice, dichiarato dagli stessi
autori che l'hanno resa canonica: $J$ è il punteggio che *scriviamo noi*, non
quello che vogliamo davvero, e un sistema abbastanza bravo massimizza il
primo anche a spese del secondo. Il fenomeno si chiama *reward hacking*, e lo
tratta {doc}`Esplorazione e ricompensa
</DeepReinforcementLearning/esplorazione-e-ricompensa>`.
`````

## Perché proprio adesso

Se le regole si ricavano dagli esempi, servono molti esempi e macchine capaci
di elaborarli. È la risposta alla domanda che tutti fanno: perché un campo
nato negli anni Cinquanta ha dato i risultati di oggi solo dopo il 2012.
[Servivano insieme i dati, la potenza di calcolo e gli algoritmi capaci di
sfruttarli]{.pt-matita}, e per decenni ne mancava almeno uno.

L'AI deve molto all'informatica e all'elettronica. Le hanno dato i linguaggi
con cui si scrivono i programmi, Python fra questi. Le hanno dato le
**librerie**, cioè raccolte di codice già scritto e collaudato che si usano
invece di rifarlo da capo (PyTorch, con cui addestreremo le reti neurali, e
TensorFlow, un equivalente nato in Google). E le hanno dato
macchine sempre più veloci: prima i processori normali, le CPU; poi le schede
grafiche, le GPU, che erano nate per far girare i videogiochi e si sono
rivelate adatte ad addestrare le reti neurali, perché sanno fare moltissimi
calcoli semplici tutti insieme; infine i chip costruiti apposta per questo
scopo, come le TPU di Google.

Ed è in debito con Internet, che ha permesso di raccogliere i dati su cui i
modelli si addestrano, dalle grandi collezioni di immagini già etichettate,
come ImageNet, al testo del web. Dati, potenza di calcolo e algoritmi sono
arrivati insieme, e il {doc}`capitolo sul deep learning
</DeepLearning/overview>` li riprende uno per uno. Di quanto pesino i dati e il
calcolo, insieme alla taglia del modello, si può anche fare una misura: sono le
leggi di scala, che la {doc}`sezione sui grandi modelli linguistici
</Transformers/llm>` racconta.

Un modo di dire corrente chiama i dati «il petrolio del nostro secolo», cioè un
giacimento che qualcuno è andato a scavare. I dati sono piuttosto uno scarto.

Non che nessuno li produca apposta: una persona, ad esempio, ha etichettato a
mano le fotografie con scritto accanto «gatto», ed è un lavoro pagato. Ma quella
è la fetta piccola, e costa cara perché è l'eccezione. Nessuno produce di
proposito il grosso dei dati: lo lasciamo dietro di noi mentre facciamo altro,
cercando un indirizzo, comprando un libro, scrivendo a un amico, guardando un
video fino in fondo o smettendo al secondo minuto. È la traccia di un passaggio,
non il prodotto di un'intenzione.

C'è un precedente su scala planetaria. Certi batteri, i
cianobatteri, impararono a spezzare l'acqua con la luce del sole per prendersi
la parte che serviva loro a costruirsi il cibo. È la fotosintesi, quella che si
studia a scuola. Buttarono via quel che restava, ed era ossigeno
{cite}`lyons2014rise`. Per loro era un sottoprodotto; per la vita di allora,
cresciuta in un mondo che non ne aveva mai avuto, era un veleno.

E per moltissimo tempo non successe niente. I cianobatteri cominciarono forse
tre miliardi di anni fa, e l'aria restò come prima, perché l'ossigeno finiva
subito, mangiato dalle rocce e dai gas che incontrava. Solo intorno a due
miliardi e trecento milioni di anni fa l'atmosfera cominciò a tenerselo, e ai
livelli di oggi ci arrivò molto più tardi ancora {cite}`lyons2014rise`: quasi
ieri, su quella scala.

Lungo quella strada comparve una forma di vita che di quello scarto faceva il
proprio respiro, e da quel respiro ricavava molta più energia di qualunque modo
di vivere venuto prima. Quella strada è la nostra: respiriamo, alla lettera, il
rifiuto di qualcun altro.

I dati stanno alle macchine come l'ossigeno sta a noi. Sono l'avanzo del nostro
passaggio nel mondo digitale, prodotto senza volerlo e in quantità che nessuno
ha deciso; e sopra quell'avanzo è cresciuta una cosa che di lì trae il proprio
respiro. E torna la domanda «perché proprio adesso»: come l'ossigeno, i
dati sono rimasti lì a lungo prima che qualcosa imparasse a respirarli.

Quello scarto, prima di diventare respiro, fu un veleno, e chi non seppe
conviverci non sparì del tutto: si ritirò. Gli organismi che l'ossigeno
avvelena esistono ancora, ma solo dove l'aria non arriva, nel fango dei fondali
e dentro il nostro intestino.

Nel nostro caso il veleno si chiama privacy: ciò che lasciamo per strada senza
pensarci è ciò di cui vive qualcuno che non abbiamo scelto (di solito un'azienda
di cui non abbiamo mai sentito il nome), e i posti dove non si lascia niente si
fanno più stretti. Se ne occupa
{doc}`Privacy e robustezza </AIResponsabile/privacy-e-robustezza>`.

Il debito con l'informatica, peraltro, è stato ripagato con gli interessi
{cite}`russell2020artificial`: parecchie idee nate nei laboratori di
intelligenza artificiale hanno poi fatto il giro dell'informatica intera e oggi
si usano ovunque senza ricordarne la provenienza. La più diffusa è la gestione
automatica della memoria.

Un programma, mentre gira, chiede continuamente al computer un po’ di spazio in
cui mettere quello che sta maneggiando, e quello spazio prima o poi va
restituito, altrimenti si esaurisce e tutto si ferma. Per anni tenerne traccia è
stato un lavoro di chi programmava, e una fonte inesauribile di errori.

Intorno al 1959 John McCarthy, uno dei quattro della proposta di Dartmouth,
affidò quel lavoro al linguaggio stesso. Nacque così la *garbage collection*
(«raccolta dei rifiuti»), per il Lisp, il linguaggio che McCarthy aveva creato
per scrivere programmi di intelligenza artificiale. Oggi Python lo fa per te,
senza che tu debba accorgertene. Con Python scriveremo il primo algoritmo.
