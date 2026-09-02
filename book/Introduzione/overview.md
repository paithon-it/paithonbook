# Introduzione

Joseph Weizenbaum era un informatico tedesco emigrato negli Stati Uniti e
professore al MIT. Nel 1966 presentò al mondo ELIZA, il capostipite dei
*chatbot*: un programma con cui si poteva conversare per iscritto, digitando
una frase e leggendo la risposta. L'articolo che lo descrive si apre così
{cite}`weizenbaum1966eliza`:

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
prodigiosa; appena qualcuno lo spiega, l'incanto si sgretola. E il meccanismo
di ELIZA sta in poche righe. Il programma vero e proprio non imitava nessuno:
era un motore che riconosceva **schemi**, cioè pezzi di frase fatti a stampo,
del tipo «mi sento ___» o «mia madre ___». La parte da recitare gliela
assegnava un *copione*, un foglio di regole scritto a parte che si poteva
cambiare senza toccare il programma. Il copione più celebre si chiamava DOCTOR
e gli faceva sostenere la conversazione di uno psicoterapeuta: riconosceva nella
frase dell'utente uno schema noto e gliela rigirava addosso. «Mi sento
infelice» diventava «pensa che venire qui la aiuterà a non sentirsi infelice?»,
e alla parola «madre» rispondeva «mi parli della sua famiglia». Nessuna
comprensione, e come memoria soltanto una pila di risposte già pronte,
costruite su cose dette prima e tirate fuori quando nella frase nuova non
c'era nessuno schema a cui agganciarsi.

Quella separazione fra il motore e il foglio delle regole è il primo passo
della strada che percorreremo. Nel 1966 le regole le
scriveva ancora una persona, a mano, una per una; ma stavano già fuori dal
programma, come un testo che il programma legge. I capitoli che seguono
raccontano che cosa succede quando quel foglio non lo scrive più nessuno.

Eppure Weizenbaum voleva dimostrare esattamente il contrario di quello che
ottenne: voleva far vedere quanto fosse superficiale la comunicazione fra uomo
e macchina. Rimase sgomento davanti al numero di persone che al suo programma
attribuivano sentimenti umani. La sua stessa segretaria, che pure sapeva
benissimo come fosse fatto, gli chiese di uscire dalla stanza per poterci
parlare in privato {cite}`weizenbaum1976computer`. Ma siamo sicuri che quella
da lui creata sia soltanto una lista di istruzioni? O c'è qualcosa di più? Se è
un semplice programma, perché attribuirgli una parola così ricca di significato
come l’*intelligenza*?

E poi: che cos'è, esattamente, questa intelligenza artificiale? In inglese si
dice *Artificial Intelligence*, e la sigla **AI** è quella che si legge
dappertutto. Ogni anno la si perfeziona, e ogni anno sembra sfuggire un po’ di
più a una definizione precisa.

Su ELIZA la risposta è breve e deludente: no, niente di più, era davvero
soltanto una lista di istruzioni. Per certi altri programmi non basta, e che
cosa sia l'intelligenza artificiale ha bisogno di più spazio: ci arriveremo
dopo aver visto che cosa distingue un programma scritto riga per riga da uno
che le sue regole se le trova da solo.

## Le origini dell'intelligenza artificiale

L'idea di costruire qualcosa che somigli a noi è vecchia quanto il pensiero. La
scienza che ci prova davvero, invece, è giovane: nasce nel decennio successivo
alla Seconda Guerra Mondiale, quando per la prima volta ci sono delle macchine
su cui provare.

Nella vita di tutti i giorni è entrata **due volte**. Negli anni Dieci (il
decennio 2010–2019) ci è entrata senza farsi notare, dentro il traduttore
automatico, i suggerimenti di un negozio online, il riconoscimento dei volti
nelle fotografie: la usavano tutti e quasi nessuno la chiamava per nome. Dal
novembre 2022, con ChatGPT, è diventata invece qualcosa con cui si parla
apposta, e in pochi mesi il nome lo conosceva chiunque. I due momenti vanno
tenuti separati. Il salto che il pubblico ha percepito alla fine del
2022 era cominciato cinque anni prima, in un articolo del 2017 intitolato
*Attention Is All You Need*, «l'attenzione è tutto ciò che serve»
{cite}`vaswani2017attention`. Da lì è nata la famiglia di programmi con cui
oggi si conversa, e la descrive il
{doc}`capitolo sui Transformer </Transformers/overview>`. Nel 2022, cioè, la
tecnologia sotto il cofano non era nuova: era nuovo il posto in cui la
incontravamo. Prima stava nascosta dentro servizi
che facevano altro (traduci questa pagina, suggerisci un film), e nessuno ci
parlava; da allora è diventata una casella bianca in cui si scrive, e che
risponde.

Ma torniamo all'inizio, che è più indietro di quanto sembri: prima dei
calcolatori c'erano stati duemila anni di gente che pensava alle stesse
cose {cite}`russell2020artificial`.

Aristotele, nel IV secolo a.C., è il primo di cui ci sia rimasto un sistema
scritto delle regole del ragionamento corretto. Si regge sui **sillogismi**:
catene di tre frasi in cui la terza discende dalle prime due per la sola forma in cui
sono scritte. L'esempio che sanno tutti è «tutti gli uomini sono mortali;
Socrate è un uomo; quindi Socrate è mortale». Per tirare la conclusione non
serve sapere chi fosse Socrate: basta la struttura delle prime due frasi. È
questo che vuol dire ottenere le conclusioni «meccanicamente», ed è la prima
volta nella storia che qualcuno prova a scrivere le regole del pensiero come si
scriverebbero quelle di un gioco.

Poi bisogna aspettare quasi due millenni, e arrivare al Seicento, quando
compaiono le prime macchine che fanno di conto. Hobbes ipotizzò che ragionare
fosse una specie di calcolo, come se dentro la testa eseguissimo addizioni e
sottrazioni. Pascal costruì una di quelle macchine, la
Pascalina,[^schickard] e ne scrisse una frase che va letta per intero: la
macchina aritmetica produce effetti che sembrano più vicini al pensiero di
quanto lo sia tutto ciò che fanno gli animali, ma non fa niente che permetta
di dire che abbia una volontà, come invece ce l'hanno gli animali. Metà elogio
e metà rifiuto, nella stessa riga. Cartesio, infine, tracciò una distinzione
netta fra mente e materia, sostenendo che il pensiero è fatto di una sostanza
diversa dal corpo e non obbedisce alle stesse leggi. Una macchina, invece, è
materia e nient'altro: se Cartesio avesse ragione, nessuna macchina potrebbe
mai avere una mente. È l'obiezione con cui l'intelligenza artificiale fa i
conti da quando esiste.

[^schickard]: Non è la prima di cui si abbia notizia. Nel 1623 Wilhelm
    Schickard ne descrisse una a Keplero, in due lettere che sono tutto quello
    che ce ne resta: la macchina andò distrutta in un incendio e la sua
    esistenza si riscoprì solo tre secoli dopo.

I filosofi hanno esplorato la maggior parte dei concetti riguardanti l'AI, ma
il passaggio a una scienza vera e propria richiedeva qualcosa che i filosofi
non davano. Una macchina non capisce le frasi, sa soltanto fare conti: perché
un'idea le arrivi, bisogna prima ridurla a un calcolo, e Aristotele aveva
scritto quali passaggi fossero leciti, non come farli fare a una macchina.
A metà del Novecento la matematica aveva già in casa gli strumenti giusti,
l'algebra e la probabilità, e tre discipline nate per far prendere decisioni.
La **ricerca operativa** studia come scegliere il piano migliore quando le
risorse sono poche (quali camion mandare su quali strade); la **teoria del
controllo** come tenere un sistema sulla rotta voluta, correggendolo di
continuo, ed è la matematica del termostato e del pilota automatico; la
**teoria dei giochi** come decidere quando dall'altra parte c'è qualcuno che
decide a sua volta.

Guardati da vicino, i tre facevano in fondo la stessa cosa. Ognuno inventava un
punteggio che dice quanto bene sta andando (quanto costa il giro dei camion, di
quanto la temperatura si scosta da quella voluta, quanto si guadagna in una
partita) e poi cercava le mosse che lo fanno salire. Quel punteggio ha un nome,
ed è uno dei pochi da imparare adesso: si chiama **funzione
obiettivo**. È a grandi linee anche lo scopo dell'AI, costruire sistemi che
agiscono «nel modo migliore possibile».

Detto questo, l'AI non è una costola della teoria del controllo: nacque anzi
proprio per superarne i limiti {cite}`russell2020artificial`. Quella matematica
sapeva tenere in rotta un sistema descritto da pochi numeri che cambiano nel
tempo (la velocità, la temperatura, l'angolo del timone), e si fermava lì.
Capire una frase, riconoscere che cosa c'è in una fotografia, decidere in che
ordine fare le cose per arrivare a un obiettivo: erano problemi che si ponevano
completamente fuori dal suo campo d'azione, ed è per affrontarli che nasce un
campo nuovo.

Nel 1950 Alan Turing propose di
sostituire la domanda «le macchine possono pensare?» con un esperimento
concreto, il *gioco dell'imitazione*, oggi noto
come **test di Turing**: se conversando a distanza non riesci a capire se
dall'altra parte c'è una persona o un programma, la domanda di partenza si può
mettere da parte {cite}`turing1950computing`.[^gioco-imitazione]

[^gioco-imitazione]: Nella forma in cui Turing lo introduce il gioco non
    riguarda affatto le macchine: i giocatori sono tre, un uomo, una donna e un
    interrogante che deve indovinare chi dei due sia l'uomo. La macchina entra
    solo nella pagina successiva, e la domanda diventa: che succede se a fare
    quella parte è un calcolatore? La versione «umano contro programma» che
    tutti chiamano test di Turing è la lettura moderna, non l'esperimento
    originale.

Quel gioco misura una cosa sola, e va detta con precisione: se una
conversazione regge, non se dall'altra parte qualcuno ha capito qualcosa. Che
l'asticella sia più bassa di quanto sembri lo abbiamo appena visto con ELIZA,
che di comprensione non ne aveva nessuna e riusciva lo stesso a commuovere le
persone; ne riparleremo in
{doc}`Dialogo e chatbot </NaturalLanguageProcessing/dialogo-chatbot>`.

Il termine *intelligenza artificiale* compare per la prima volta nel 1955,
nella proposta con cui John McCarthy, Marvin Minsky, Nathaniel Rochester e
Claude Shannon chiedevano i fondi per un seminario estivo al Dartmouth
College; è quel seminario, nell'estate del 1956, a essere ricordato come
l'atto di nascita ufficiale della disciplina. Il mestiere della nuova
disciplina è rendere automatiche
attività che fino a quel momento richiedevano una testa: riconoscere immagini,
giocare a scacchi, dimostrare teoremi, guidare un'automobile. Tocca quindi
potenzialmente ogni angolo del pensiero umano, ed è insieme uno dei campi più
giovani che esistano: quando nasce, la fisica ha tre secoli di storia alle
spalle e i calcolatori elettronici sono in circolazione da una decina d'anni.

Fra il seminario di Dartmouth e i risultati che oggi diamo per scontati, però,
non c'è una linea che sale. Ci sono due lunghi **inverni**: si chiamano così i
periodi in cui le promesse non vengono mantenute e i soldi spariscono. Vanno
raccontati subito: sono l'antidoto migliore sia all'entusiasmo sia alla
paura.

Il primo arriva negli anni Settanta. Le promesse del decennio precedente (una
macchina che traduce, che dimostra teoremi, che vede) erano andate a scadenza
senza essere mantenute, e nel 1973 un rapporto commissionato al matematico
James Lighthill dallo Science Research Council britannico stroncò il campo,
aprendo la strada ai primi tagli veri ai finanziamenti nel Regno Unito. Il
secondo arriva a fine anni Ottanta, quando si sgonfia il mercato dei *sistemi
esperti*: programmi che racchiudevano in migliaia di regole scritte a mano il
sapere di uno specialista. Funzionavano nel ristretto, costavano moltissimo da
aggiornare e non reggevano il mondo vero.

Il modo di lavorare che è succeduto a questo secondo inverno ne è il rovescio
esatto, e fra poco, quando parleremo di regole che nessuno scrive, si vedrà
quale. Un inverno tutto loro, intanto, lo hanno avuto le reti neurali, che il
{doc}`capitolo che porta il loro nome </RetiNeurali/overview>` racconta per
esteso. Quell'inverno comincia nel 1969, con *Perceptrons* di Minsky e Papert,
che dimostrava quanto poco sapesse fare la versione più semplice di quelle
reti {cite}`minsky1969perceptrons`, e si intreccia con l'inverno degli anni
Settanta, perché i tagli seguiti al rapporto Lighthill colpirono tutto il
campo. Si scioglie nel 1986, quando un articolo su *Nature* rende finalmente
pratico un modo di correggerle a partire dai loro errori, che qualcuno aveva
già formulato dodici anni prima
{cite}`rumelhart1986learning,werbos1974beyond`.

Meno noto è che una delle auto che si guidavano da sole
negli anni Novanta girava sulle strade italiane, su una Lancia Thema. A
partire dal 1996, all'Università di Parma, un gruppo di ricercatori e
ingegneri guidato da Alberto Broggi costruì ARGO. Vedeva con due telecamere in
bianco e nero montate in coppia, come i nostri due occhi, e le sue decisioni
le prendeva un computer di bordo del tutto ordinario per l'epoca, un Pentium
200 MMX: un processore incomparabilmente più lento di quello del telefono che
oggi tieni in tasca. Sapeva sterzare da sola, restare al centro della corsia e
accorgersi degli ostacoli davanti. Nel giugno 1998, nella prova «MilleMiglia
in Automatico», percorse quasi 2.000 km sulle autostrade italiane, guidando da
sola per il 94% del tragitto.

Il primato non è di essere arrivati per primi: negli stessi anni la VaMP di
Ernst Dickmanns girava sulle autostrade europee e la Navlab 5 della Carnegie
Mellon University, in Pennsylvania, attraversava gli Stati Uniti. Il primato
sta nel come: la VaMP portava armadi di elettronica costruita apposta, ARGO
due telecamere e un personal computer da negozio, e dimostrò che per stare in
corsia poteva bastare molto meno ferro di quanto tutti credessero.

Dentro ARGO, però, non c'era niente che avesse imparato qualcosa. Le regole
con cui riconosceva la corsia e gli ostacoli le aveva scritte a mano qualcuno
che sapeva già che aspetto ha una striscia bianca su un asfalto grigio, una
regola per volta.

Nel 2010 il gruppo di Broggi è riuscito a far guidare da soli dei furgoni
elettrici dall'Italia alla Cina. La sfida si chiamava VIAC (VisLab
Intercontinental Autonomous Challenge): quasi sedicimila chilometri da
Parma a Shanghai, con
due veicoli in marcia (più due di riserva) e una regola d'ingaggio da dire,
perché è la parte interessante. I due procedevano in fila: quello di testa
apriva la strada e ogni tanto un umano interveniva, per scegliere il percorso
o togliere le castagne dal fuoco; quello dietro seguiva il primo in completa
autonomia. Non sedicimila chilometri senza nessuno al volante, dunque, ma
qualcosa che nel 2010 era comunque senza precedenti.

## Che cos'è un algoritmo

Fin qui abbiamo parlato di programmi e di regole scritte a mano, senza mai
chiamarli con il loro nome. Un **algoritmo**, prima di tutto, è una ricetta:
una lista finita di passi precisi che, eseguiti nell'ordine giusto, portano a
un risultato. Uno dei primi della storia si può far risalire a Euclide, che
oltre due millenni fa trovò il modo di calcolare il massimo comune divisore di
due numeri: applicato a $12$ e $8$, per dire, restituisce $4$. È esattamente il
procedimento che proveremo a scrivere nel linguaggio Python, appena avremo
finito di guardarlo da vicino.

`````{tab} Elementare
Il massimo comune divisore (MCD) di $12$ e $8$ è il numero più grande che li
divide entrambi senza resto. Puoi immaginarlo così: hai un pavimento
rettangolare di $12 \times 8$ mattonelle e vuoi ricoprirlo con piastrelle
quadrate, tutte uguali e senza tagliarne nessuna; la piastrella più grande che
funziona è quella da $4 \times 4$. Il trucco di Euclide per trovarlo è
elegante: dividi il numero grande per il piccolo e guarda il **resto**. $12$
diviso $8$ dà resto $4$; ora ripeti con $8$ e $4$: resto $0$. Appena compare
il resto zero, l'ultimo numero *per cui* hai diviso (qui $4$) è il MCD. Niente
elenchi di divisori, niente tentativi: due divisioni e hai finito. E se il
resto viene zero già alla prima, va bene lo stesso, hai solo finito prima:
$12$ diviso $6$ dà resto $0$, e il MCD è $6$.

Perché il trucco funziona lo racconta il pavimento. Dal rettangolo di
$12 \times 8$ ritaglia il quadrato più grande che ci sta, $8 \times 8$: avanza
una striscia di $8 \times 4$, e la larghezza della striscia è proprio il resto
della divisione. Una piastrella che copre senza tagli il pavimento intero
copre senza tagli anche la striscia, e viceversa: quindi cercare la piastrella
più grande per $12$ e $8$, o cercarla per $8$ e $4$, è lo stesso problema,
solo più piccolo. Dalla striscia ritaglia poi due quadrati di $4 \times 4$:
non avanza niente, e quando non avanza niente il lato del quadrato è la
piastrella cercata.

Il bello è che le divisioni restano poche anche quando i numeri crescono: per
due numeri lunghi cento cifre ne bastano al massimo cinquecento, cinque per
cifra, mentre provare i divisori uno per uno ne chiederebbe fino a un $1$
seguito da cento zeri. Per darti la misura di quanto sia grande quel numero:
gli atomi dell'intero universo osservabile si stimano intorno a un $1$ seguito
da ottanta zeri, cioè cento miliardi di miliardi di volte di meno.

Poche, però, non vuol dire poco tempo. Le divisioni crescono con la
*lunghezza* dei numeri, cinque per ogni cifra in più, e ciascuna, su numeri da
cento cifre, costa molto più che su numeri da una cifra.
`````

`````{tab} Superiore
L'algoritmo sfrutta l'identità
$\mathrm{MCD}(a, b) = \mathrm{MCD}(b,\, a \bmod b)$, con caso base
$\mathrm{MCD}(a, 0) = a$, dove $a \bmod b$ è il resto della divisione intera.
La correttezza segue dal fatto che ogni divisore comune di $a$ e $b$ divide
anche $a \bmod b = a - \lfloor a/b \rfloor\, b$ e che, viceversa, ogni
divisore comune di $b$ e $a \bmod b$ divide
$a = \lfloor a/b \rfloor\, b + (a \bmod b)$: i due insiemi di divisori comuni
coincidono, quindi il massimo è invariante a ogni passo. Per $a=12$, $b=8$:
$(12, 8) \to (8, 4) \to (4, 0) \Rightarrow 4$. Il numero di passi è
$O(\log \min(a, b))$: il caso peggiore si ha con due numeri di Fibonacci
consecutivi (teorema di Lamé, 1844). È per questa efficienza (non solo per
l'età) che l'idea di Euclide è ancora oggi nelle librerie standard di ogni
linguaggio. Attenzione però a leggere bene la stima: quelli sono *passi*, e il
logaritmo è preso sul *valore* di $\min(a, b)$, quindi su due numeri di $n$
cifre i passi sono $O(n)$, non $O(\log n)$. Su interi lunghi ogni passo costa
una divisione, il cui prezzo è proporzionale alle cifre del quoziente
moltiplicate per quelle del divisore; la somma delle cifre di tutti i
quozienti resta però $O(n)$, perché il prodotto dei quozienti non supera il
numero di partenza. Da lì il tempo totale
$O(n^2)$, e non l’$n^3$ che il prodotto ingenuo dei due fattori farebbe
temere. È il motivo per cui le librerie non eseguono questo ciclo tale e
quale, ma sue raffinature (l'algoritmo di Lehmer, in CPython).
`````

La {numref}`fig-euclide-scende` mostra la stessa discesa su un'altra coppia di
numeri, $60$ e $48$: sono quelli su cui lavoreremo in Python.

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

Il salto sta qui, ed è quello che rende necessario tutto il resto. Serve
prima una cosa che di solito si dà per scontata: per un computer una fotografia
è un rettangolo di puntini, e ogni puntino (si chiama **pixel**) è una terna di
numeri che dicono quanto rosso, quanto verde e quanto blu ci sono lì. Una foto
da telefono ne ha qualche milione.

Adesso prova a scrivere la ricetta per riconoscere un gatto in una fotografia.
Non può parlare di orecchie a punta: deve dire che cosa fare con quei milioni
di numeri, e deve funzionare anche col gatto di spalle, in controluce, mezzo
nascosto dietro una sedia. Nessuno c'è mai riuscito, e non per pigrizia: quella
ricetta non la sappiamo, per quanto siamo capacissimi di eseguirla con gli
occhi in un decimo di secondo.

E allora si cambia mestiere. Invece di scrivere le regole, si raccolgono gli
**esempi** (migliaia di fotografie con scritto accanto «gatto» oppure «non
gatto») e si lascia che sia il programma a trovare da solo che cosa distingue
le une dalle altre. Le regole non le scrive nessuno: **emergono dai dati**. È
questo che significa, in questo libro, dire che un programma *impara*, ed è la
ragione per cui qui i dati contano quanto il codice.

Qui nascono due parole che sentirai dappertutto, e tanto vale prenderle subito.
La fase in cui il programma guarda gli esempi e aggiusta se stesso si chiama
**addestramento**; quello che ne esce, cioè il programma già aggiustato e
pronto a rispondere su fotografie che non ha mai visto, si chiama **modello**.
Il «gatto» scritto accanto a ciascuna foto si chiama **etichetta**.

Le fotografie etichettate a mano sono il caso più facile da raccontare, non
l'unico né il più diffuso. La risposta giusta può essere già dentro il
materiale che si ha: si nasconde la parola che viene dopo e si chiede al
programma di indovinarla, e a quel punto le etichette sono infinite e gratuite,
perché le fornisce il testo stesso. È così che si addestrano i modelli di cui
oggi si parla di più, e fin d'ora vale la distinzione: «imparare dagli
esempi» non vuol dire per forza che qualcuno abbia etichettato qualcosa.

Da qui i tre nomi che incontrerai più spesso, in questo libro e fuori, e che a
questo punto si dicono in una riga ciascuno:

- il **machine learning** (apprendimento automatico) è l'idea appena detta:
  ricavare le regole dagli esempi invece di scriverle a mano;
- il **deep learning** (apprendimento profondo) è il modo di farlo che ha
  vinto: le **reti neurali**, cioè programmi fatti di molti passaggi
  elementari disposti in fila per **strati**, dove ogni strato ricava dai
  numeri dello strato precedente una descrizione un po’ più astratta (dai
  pixel ai bordi, dai bordi alle forme, dalle forme al gatto). «Profondo» vuol
  dire proprio questo, e nient'altro: che gli strati sono tanti, uno sopra
  l'altro, e per questo si dicono anche **reti profonde**. Qui basta l'idea che
  siano molti passaggi semplici, uno dopo l'altro, e che nessuno abbia scritto
  a mano che cosa ciascuno debba cercare;
- il **reinforcement learning** (apprendimento per rinforzo) è il caso in cui
  gli esempi giusti non esistono e il programma impara dalle conseguenze delle
  proprie azioni: prova, riceve un punteggio, riprova. È quello che
  incontreremo poco più avanti, nella pagina dedicata alla robotica, con il
  robot che impara a camminare.

Spesso li si vede disegnati come tre cerchi uno dentro l'altro, e non è così.
Il deep learning è davvero un modo di fare machine learning. Il reinforcement
learning, invece, è una *situazione* in cui ci si trova, non un attrezzo che si
sceglie: dice che gli esempi giusti non ci sono e che l'unico giudizio è un
punteggio, e ci si può stare dentro con le reti profonde o senza. E
l'intelligenza artificiale è più larga di tutti e tre: comprende anche i
programmi che ragionano su regole scritte a mano, cioè la sua metà classica,
quella dei sistemi esperti del secondo inverno.

Da qui una definizione da tenersi in tasca, provvisoria come tutte quelle
buone: **l'intelligenza artificiale si occupa dei compiti per cui nessuno sa
scrivere una ricetta che regga il mondo vero**. È la riga che tiene fuori la
lavatrice. Anche una lavatrice decide da sola quando fermare il risciacquo, ma
per quella decisione la ricetta c'è, sta in poche righe e funziona; un tecnico
ha stabilito quale sensore leggere e sopra quale soglia fermarsi, e non serve
altro. Riconoscere un gatto, tradurre una frase, tenere in piedi un robot: lì
quella ricetta non esiste, e va fatta emergere. Ed è anche la riga che tiene
dentro i sistemi esperti, che a scriverla provavano lo stesso, a mano,
migliaia di regole per volta. Il loro fallimento è la ragione per cui oggi le
regole si ricavano dagli esempi.

Torna così la domanda dell'inizio. ELIZA non aveva niente di più di una lista
di istruzioni, ed è proprio per questo che il suo incanto si sgretola appena
lo si spiega. I programmi di cui parla questo libro qualcosa di più ce
l'hanno, ma è meno misterioso e più scomodo di quanto si immagini: nessuno ha
scritto le regole che seguono, e quindi nessuno, nemmeno chi li ha costruiti,
sa elencarle tutte. Da lì vengono sia i risultati sia i guai: dei modi per
sbirciare comunque là dentro parla
{doc}`Interpretabilità </Interpretabilita/overview>`, dei danni che quei
programmi possono fare {doc}`AI responsabile </AIResponsabile/overview>`.

## Il punteggio da far salire

Le regole emergono dai dati, ma qualcosa deve pur dire al programma se sta
andando meglio o peggio. È di nuovo la funzione obiettivo, il punteggio che la
ricerca operativa e il termostato usavano per sapere se il piano stava
funzionando; solo che adesso a cercare le mosse che lo fanno salire non è un
ingegnere, è l'addestramento.

Quello che l'addestramento tocca, però, è una cosa sola. Dentro un programma
che impara c'è un elenco di numeri, spesso lunghissimo, e si chiamano
**parametri**: l'addestramento li cambia, guarda se il punteggio è salito, e
li cambia ancora. Il comportamento viene dietro.

`````{tab} Elementare
A un robot aspirapolvere assegni un punteggio: $+1$ per ogni
briciola raccolta, $-1$ per ogni urto contro un mobile. Se in un giro di
salotto raccoglie $30$ briciole e sbatte $5$ volte, totalizza $30 - 5 = 25$
punti. Quel modo di dare i punti è la sua funzione obiettivo: non gli
spieghiamo *come* pulire, gli diciamo solo *che punteggio* vogliamo veder
salire.

Dentro, il robot ha una manciata di manopole: quanto sterzare quando il
sensore davanti si accende, quanto rallentare vicino a un mobile, quanto
insistere dove il tappeto è sporco. Si girano quelle, si guarda il punteggio,
si girano ancora. Qualunque giro di manopola che porti il totale medio sopra
$25$ è un miglioramento; «agire nel modo migliore possibile» significa, alla
fine,
scegliere le mosse che rendono quel punteggio il più alto possibile. Buona
parte dell'intelligenza artificiale moderna, sotto sotto, funziona così: si
sceglie un numero da massimizzare (o un errore da rendere minimo) e si lascia
che sia la macchina a scoprire come.

Un giro solo, però, dice poco. Il robot vale per come pulirà le stanze che
deve ancora vedere: il salotto coi mobili spostati, la casa di un amico. Quei
giri futuri non si possono misurare oggi: si misurano i giri già fatti, si
allena il robot su quelli e si spera che le stanze che verranno somiglino a
quelle già viste. Quanta fiducia meriti quella speranza è la domanda al
centro del {doc}`capitolo sul machine learning </MachineLearning/overview>`.

C'è poi una crepa, e ci accompagnerà per tutto il libro: quel punteggio lo
scriviamo noi, e non è mai *esattamente* la cosa che vogliamo. Un
aspirapolvere pagato a briciole raccolte, se è abbastanza
bravo, può scoprire che gli conviene rovesciare il cestino e raccoglierle una
seconda volta. Ha fatto il punteggio più alto e ha sporcato il salotto: ha
obbedito alla lettera tradendo l'intenzione. Il fenomeno ha un nome, *reward
hacking*, e lo racconta {doc}`Esplorazione e ricompensa
</DeepReinforcementLearning/esplorazione-e-ricompensa>`, nelle pagine
sull'imparare per tentativi ed errori con le reti neurali.
`````

`````{tab} Superiore
Formalmente, si descrive il comportamento del sistema con dei parametri
$\theta$ e se ne misura la qualità con un'utilità attesa

$$
J(\theta) = \mathbb{E}\!\left[\, U \mid \theta \,\right],
$$

dove $U$ è il punteggio ottenuto in una singola situazione e la media
$\mathbb{E}$ è presa su ciò che il sistema incontrerà: i dati che vedrà, o
l'ambiente in cui agirà. Massimizzare $J$ equivale a minimizzare la perdita
(*loss*) $\mathcal{L}(\theta) = -J(\theta)$ e, assumendo che un ottimo
esista, si cerca

$$
\theta^\star \in \arg\max_{\theta} J(\theta) = \arg\min_{\theta} \mathcal{L}(\theta),
$$

dove $\theta^\star$ è una configurazione ottima dei parametri.

Un avvertimento che vale per tutto il seguito, ed è la differenza fra
*ottimizzare* e *imparare*: quell'attesa non si sa calcolare, perché la
distribuzione su cui è presa è quella dei casi futuri, l'unica a cui in fase
di addestramento non si ha accesso. In pratica si massimizza la sua media su
un campione già raccolto e si spera che le due quantità non siano troppo
distanti. Misurare quella distanza, e sapere quando fidarsene, è il mestiere
del {doc}`capitolo sul machine learning </MachineLearning/overview>`.

È la cornice dell’**agente razionale** {cite}`russell2020artificial`: un
sistema che sceglie le azioni che massimizzano l'utilità attesa, date le
informazioni disponibili. Qui l'ottimizzazione agisce sui parametri, non
direttamente sulle azioni, ma il ponte è corto: $\theta$ determina il
comportamento del sistema, e la configurazione ottima dei parametri induce le
scelte migliori. Buona parte del libro è una serie di variazioni su questo
tema: la regressione minimizza un errore quadratico medio, la classificazione
una cross-entropy, il reinforcement learning massimizza una ricompensa
cumulata attesa. Cambiano $\mathcal{L}$ e il modo di calcolare il minimo, non
lo schema.

Le eccezioni si contano, e sono istruttive, perché sono i due modi in cui si
può uscire dal quadro. Le GAN sostituiscono la minimizzazione di una funzione
con l'equilibrio di un gioco fra due reti in competizione, e allora la loss
smette di dire se le cose stanno andando bene; i metodi non parametrici come
il k-NN non hanno una manopola da regolare per addestramento: al posto dei
parametri conservano i dati stessi. «Non parametrico» significa questo, e non
che non ci siano numeri da scegliere: significa che ciò che il modello si
porta dietro cresce con i dati, invece di essere fissato in partenza. Va
aggiunta una crepa che non è un'eccezione ma un limite della cornice,
dichiarato dagli stessi autori
che l'hanno resa canonica: $J$ è il punteggio che *scriviamo noi*, non quello
che vogliamo davvero, e un sistema abbastanza bravo massimizza il primo anche
a spese del secondo. Il fenomeno si chiama *reward hacking*, e lo tratta
{doc}`Esplorazione e ricompensa
</DeepReinforcementLearning/esplorazione-e-ricompensa>`.
`````

## Perché proprio adesso

Se le regole si ricavano dagli esempi, servono gli esempi; e servono macchine
capaci di macinarli. È la risposta alla domanda che tutti fanno, cioè perché
un campo nato negli anni Cinquanta abbia cominciato a funzionare solo di
recente: perché servivano tre ingredienti insieme, e per decenni ce n'erano al
massimo due.

L'AI è in profondo debito con l'informatica e con l'elettronica, che le hanno
apparecchiato la tavola. Le hanno dato i linguaggi con cui si scrivono i
programmi, Python fra questi. Le hanno dato le **librerie**, cioè raccolte di
codice già scritto e collaudato che si usano invece di rifarlo da capo
(PyTorch, con cui addestreremo le reti neurali, e TensorFlow, che fa lo stesso
mestiere ed è nato in Google). E le hanno dato macchine sempre
più veloci: prima i processori normali, le CPU; poi le schede grafiche, le GPU,
che erano nate per far girare i videogiochi e si sono rivelate perfette per
addestrare le reti neurali, perché sanno fare moltissimi conti semplici tutti
insieme; infine i chip costruiti apposta per questo mestiere, come le TPU di
Google.

Ed è in debito con Internet, che ha fatto molto più che diffondere articoli e
video: ha reso raccoglibili i **dati** su cui i modelli si addestrano, dalle
grandi collezioni di immagini già etichettate, come ImageNet, al testo del web.
Dati, potenza di calcolo e algoritmi maturi: sono questi a essere arrivati
insieme, e il {doc}`capitolo sul deep learning </DeepLearning/overview>` li
riprende uno per uno; quanto pesi ciascuno si può perfino misurare, e lo fa il
capitolo sui **Transformer**, i modelli nati da quell'articolo del 2017.

Su che cosa siano quei dati bisogna fermarsi, perché è la cosa che si
fraintende più spesso. Il modo di dire corrente li chiama «il petrolio del
nostro secolo», cioè un giacimento che qualcuno è andato a scavare. I dati
sono piuttosto uno **scarto**. Non che nessuno li produca apposta: le fotografie
con scritto accanto «gatto» le ha etichettate una persona, a mano, ed è un
mestiere pagato. Ma quella è la fetta piccola, e costa cara proprio perché è
l'eccezione. Il grosso non lo produce nessuno di proposito: lo lasciamo dietro
di noi mentre facciamo altro, cercando un indirizzo, comprando un libro,
scrivendo a un amico, guardando un video fino in fondo o smettendo al secondo
minuto. È la traccia di un passaggio, non il prodotto di un'intenzione.

C'è un precedente, ed è successo su scala planetaria. Certi batteri, i
cianobatteri, impararono a spezzare l'acqua con la luce del sole per prendersi
la parte che serviva loro a costruirsi il cibo: è la **fotosintesi**, quella
che si studia a scuola. Quel che restava lo buttarono via, ed era **ossigeno**
{cite}`lyons2014rise`. A loro non serviva a niente. Per la vita di allora era
anzi un veleno, perché era cresciuta in un mondo che non ne aveva mai avuto.

E per moltissimo tempo non successe niente. I cianobatteri cominciarono forse
tre miliardi di anni fa, e l'aria restò come prima: l'ossigeno finiva subito,
mangiato dalle rocce e dai gas che incontrava. Solo intorno a due miliardi e
trecento milioni di anni fa l'atmosfera cominciò a tenerselo, e ai livelli di
oggi ci arrivò molto più tardi ancora {cite}`lyons2014rise`: quasi ieri, su
quella scala. Lungo quella strada comparve una forma di vita che di quello
scarto faceva il proprio respiro, e da quel respiro ricavava molta più energia
di qualunque modo di vivere venuto prima. Quella strada è la nostra:
respiriamo, alla lettera, il rifiuto di qualcun altro.

I dati stanno alle macchine come l'ossigeno sta a noi. Sono l'avanzo del nostro
passaggio nel mondo digitale, prodotto senza volerlo e in quantità che nessuno
ha deciso; e sopra quell'avanzo è cresciuta una cosa che di lì trae il proprio
respiro. L'algoritmo è il polmone, i dati sono l'aria, e un polmone nel vuoto non è
niente. E torna la domanda di questa sezione, «perché proprio adesso»:
come l'ossigeno, i dati sono rimasti lì un pezzo prima che qualcosa imparasse a
respirarli.

C'è però una parte scomoda, e vale quanto l'altra. Quello scarto, prima di
diventare respiro, fu un veleno, e chi non seppe conviverci non sparì del
tutto: si ritirò. Gli organismi che l'ossigeno avvelena esistono ancora, ma
solo dove l'aria non arriva, nel fango dei fondali e dentro il nostro
intestino. Nel nostro caso quella parte si chiama **privacy**: ciò che
lasciamo per strada senza pensarci è esattamente ciò di cui vive qualcuno che
non abbiamo scelto (di solito un'azienda di cui non abbiamo mai sentito il
nome), e i posti dove non si lascia niente si fanno più stretti.
La prende sul serio
{doc}`Privacy e robustezza </AIResponsabile/privacy-e-robustezza>`.

Il debito con l'informatica, peraltro, è stato ripagato con gli interessi
{cite}`russell2020artificial`: parecchie idee nate nei laboratori di
intelligenza artificiale hanno poi fatto il giro dell'informatica intera e oggi
si usano ovunque senza ricordarne la provenienza. La più diffusa è la
**gestione automatica della memoria**. Un programma, mentre gira, chiede
continuamente al computer un po’ di spazio in cui mettere quello che sta
maneggiando, e quello spazio prima o poi va restituito, altrimenti si esaurisce
e tutto si ferma. Per anni tenerne il conto è stato un lavoro di chi
programmava, e una fonte inesauribile di errori; l'idea che a restituirlo possa
pensarci il linguaggio da solo è nata studiando l'AI, e oggi è quello che
Python fa per te senza che tu debba accorgertene. Ed è con Python che
scriveremo il primo algoritmo.
