# Introduzione

Joseph Weizenbaum era un informatico tedesco emigrato negli Stati Uniti e
professore al MIT. Nel 1966 presentò al mondo ELIZA, il primo *chatbot* della
storia: un programma con cui si poteva conversare per iscritto, digitando una
frase e leggendo la risposta. L'articolo che lo descrive si apre così
{cite}`weizenbaum1966eliza`:

>Si dice che spiegare significhi dissolvere.[^spiegare-via]

[^spiegare-via]: «It is said that to explain is to explain away»,
    nell'originale. *To explain away* è un verbo frasale che l'italiano non
    ha: vuol dire spiegare una cosa fino a toglierle ogni mistero, e con
    quello ogni importanza. Poco più sotto, nello stesso paragrafo, l'articolo
    dice che appena il programma viene smascherato «its magic crumbles away»:
    l'incanto si sgretola.

Nei programmi che sembrano intelligenti, osservava, questa massima si compie
alla perfezione: finché il meccanismo resta nascosto la macchina appare
prodigiosa; appena qualcuno lo spiega, l'incanto si sgretola. E il meccanismo
di ELIZA sta in poche righe: il programma imitava uno psicoterapeuta,
riconosceva nella frase dell'utente uno schema che conosceva e la rigirava
sotto forma di domanda. «Mi sento triste» diventava «Perché ti senti triste?»;
alla parola «madre» rispondeva «Mi parli della sua famiglia». Nessuna
comprensione, nessuna memoria di ciò che era stato detto due frasi prima: un
elenco di schemi e di sostituzioni.

Eppure Weizenbaum, che voleva dimostrare proprio la superficialità della
comunicazione fra uomo e macchina, rimase sgomento davanti al numero di
persone che al suo programma attribuivano sentimenti umani: la sua stessa
segretaria, che pure sapeva benissimo come fosse fatto, gli chiese di uscire
dalla stanza per poterci parlare in privato {cite}`weizenbaum1976computer`. Ma
siamo
sicuri che sia solo una semplice lista di istruzioni quella da lui creata? O
c'è qualcosa di più? Se è un semplice programma, perché attribuirgli una
parola così ricca di significato come l'*intelligenza*?

E poi, cos'è questa intelligenza artificiale (o **AI** dall'inglese
*Artificial Intelligence*), che così velocemente, anno dopo anno,
perfezionamento dopo perfezionamento, sembra sfuggire inesorabilmente a ogni
tentativo di definizione precisa? Sono le domande da cui parte questo
capitolo. Alla prima si può rispondere subito, ed è un no: ELIZA era davvero
soltanto una lista di istruzioni, e non c'era niente di più. Alla seconda si
risponde alla fine di questa pagina, quando avremo visto che cosa distingue un
programma scritto riga per riga da uno che le sue regole se le trova da solo.

## Le Origini dell'Intelligenza Artificiale

È presente in ognuno di noi quella strana sensazione suscitata dall’osservare un programma che emula i comportamenti umani, come se fosse qualcosa di "vivo" e non un semplice programma che esegue istruzioni.

La possibilità di creare un qualcosa che assomigliasse all'uomo ha sollecitato
la curiosità di tutti i pensatori del passato. Difatti, nonostante sia entrata
nella vita di tutti i giorni solo dagli anni Dieci in poi, l'AI non è una
scienza nuovissima: le sue origini risalgono al periodo immediatamente
successivo alla Seconda Guerra Mondiale. Nel 1950 Alan Turing, in un articolo
destinato a fare storia, propose di sostituire la domanda «le macchine possono
pensare?» con un esperimento concreto, il *gioco dell'imitazione*, oggi noto
come **test di Turing**: se conversando a distanza non riesci a capire se
dall'altra parte c'è una persona o un programma, la questione filosofica
smette di essere decisiva {cite}`turing1950computing`. Vale la pena essere
precisi su che cosa quel gioco misuri, perché il seguito del libro ci
tornerà: misura l'indistinguibilità in una conversazione, non la comprensione.
Che l'asticella sia più bassa di quanto sembri lo abbiamo appena visto con
ELIZA, che di comprensione non ne aveva nessuna e riusciva lo stesso a
commuovere le persone; ne riparleremo nel capitolo sul linguaggio naturale.

Il termine *intelligenza artificiale* compare per la prima volta nel 1955,
nella proposta con cui John McCarthy, Marvin Minsky, Nathaniel Rochester e
Claude Shannon chiedevano i fondi per un seminario estivo al Dartmouth
College; è quel seminario, nell'estate del 1956, a essere ricordato come
l'atto di nascita ufficiale della disciplina. È una scienza che si può
applicare potenzialmente a ogni sfera del pensiero umano, in quanto si occupa
di rendere automatiche alcune attività intellettive come il riconoscimento di
immagini, il gioco degli scacchi, la dimostrazione di teoremi matematici e la
guida autonoma di veicoli. In questo senso, è uno dei campi più antichi e
trasversali.

Fra il seminario di Dartmouth e i risultati che oggi diamo per scontati, però,
non c'è una linea che sale. Ci sono due lunghe gelate, e conviene raccontarle
subito, perché sono l'antidoto migliore sia all'entusiasmo sia alla paura. Le
promesse degli anni Sessanta (una macchina che traduce, che dimostra teoremi,
che vede) arrivarono a scadenza senza essere mantenute: nel 1973 il rapporto
Lighthill, commissionato dal governo britannico, stroncò il campo e portò ai
primi tagli veri ai finanziamenti. Un secondo gelo arrivò a fine anni Ottanta,
quando si sgonfiò il mercato dei *sistemi esperti*, programmi che
racchiudevano in migliaia di regole scritte a mano il sapere di uno
specialista: funzionavano nel ristretto, costavano moltissimo da aggiornare e
non reggevano il mondo vero. Delle ragioni tecniche del primo gelo parla il
capitolo sulle reti neurali, perché riguardano da vicino proprio quelle; del
secondo si sentirà l'eco in tutto il libro, dato che è per non ripetere
l'errore dei sistemi esperti che oggi le regole non si scrivono più a mano.

E se dicessi che uno dei primi test al mondo di auto a guida autonoma è stato
effettuato in Italia, su una “fiammante” Lancia Thema? Ebbene sì: a partire
dal 1996, all’Università di Parma, un team di ricercatori e ingegneri
coordinato dal prof. Alberto Broggi ha sviluppato un vero e proprio prototipo
di guida autonoma, un’autovettura dotata di visione “stereo” (due telecamere
che lavorano in coppia, come gli occhi umani) e di un computer di bordo del
tutto ordinario per l'epoca, un Pentium 200 MMX: un processore incomparabilmente
più lento di quello del telefono che oggi tieni in tasca. Era in grado di sterzare da
sola, restare al centro della corsia di marcia e localizzare gli eventuali
ostacoli sul percorso. Il suo nome è ARGO e nel giugno 1998, nella prova
“MilleMiglia in Automatico”, percorse quasi 2.000 km di strade e autostrade
italiane guidando in autonomia per oltre il 90% del tragitto.

Nel 2010 il gruppo di Broggi è riuscito a far guidare autonomamente delle auto dall’Italia… alla Cina! La sfida si chiamava VIAC (VisLab Intercontinental Autonomous Challenge) e ha coinvolto quattro veicoli su un viaggio di circa 13.000 chilometri da Parma a Shanghai, percorso in larga parte senza intervento umano.

Torniamo indietro, però, e di parecchio: prima di Turing e dei calcolatori
c'erano stati duemila anni di gente che pensava alle stesse
cose {cite}`russell2020artificial`. Aristotele, nel IV secolo a.C., fu il primo
a formulare un insieme preciso di leggi che governano la parte razionale della
mente: sviluppò un sistema basato sui **sillogismi**, cioè ragionamenti in cui
la conclusione discende da due premesse per pura forma. L'esempio che sanno
tutti è «tutti gli uomini sono mortali; Socrate è un uomo; quindi Socrate è
mortale»: per tirare la conclusione non serve sapere chi fosse Socrate, basta
la struttura delle prime due frasi. È questo che vuol dire ottenere le
conclusioni «meccanicamente», ed è la prima volta nella storia che qualcuno
prova a scrivere le regole del pensiero come si scriverebbero quelle di un
gioco. Hobbes, nel Seicento, ipotizzò che il ragionamento umano avesse a che
fare con meccanismi simili al calcolo numerico, come se noi «eseguissimo»
addizioni e sottrazioni nei nostri pensieri. Pascal scrisse che la «macchina
aritmetica» produce effetti che sembrano più vicini al pensiero di tutte le
azioni degli animali, e costruì la Pascalina, una delle prime macchine
calcolatrici (la prima in assoluto risulta essere quella di Wilhelm Schickard,
nel 1623). Infine, Cartesio fornì la prima discussione chiara sulla
distinzione tra mente e materia.

I filosofi hanno esplorato la maggior parte dei concetti riguardanti l'AI, ma
il passaggio a una scienza vera e propria richiedeva qualcosa che i filosofi
non davano: scrivere quelle idee in formule, cioè in una forma che una
macchina possa eseguire. La matematica a metà del '900 ereditava dal passato
tutta una serie di strumenti buoni allo scopo: l'algebra, la probabilità, e
poi tre discipline nate per far prendere decisioni alle macchine e alle
organizzazioni. La **ricerca operativa** studia come scegliere il piano
migliore quando le risorse sono poche (quali camion mandare su quali strade);
la **teoria del controllo** come tenere un sistema sulla rotta voluta,
correggendolo di continuo, ed è la matematica del termostato e del pilota
automatico; la **teoria dei giochi** come decidere quando dall'altra parte c'è
qualcuno che decide a sua volta. D'improvviso, questi campi sono apparsi come
facce dello stesso
poliedro, il cui scopo è la progettazione di sistemi che massimizzano nel
tempo una **funzione obiettivo**, cioè un punteggio che misura quanto bene il
sistema sta svolgendo il proprio compito. Questo, a grandi linee, corrisponde
allo scopo dell’AI: la costruzione di sistemi che agiscono “nel modo migliore
possibile”. L’idea merita un momento di attenzione, perché tornerà in ogni
capitolo del libro.

`````{tab} Elementare
Immagina un robot aspirapolvere a cui assegni un punteggio: $+1$ per ogni
briciola raccolta, $-1$ per ogni urto contro un mobile. Se in un giro di
salotto raccoglie $30$ briciole e sbatte $5$ volte, totalizza $30 - 5 = 25$
punti. Quel numero è la sua funzione obiettivo: non gli spieghiamo *come*
pulire, gli diciamo solo *che punteggio* vogliamo veder salire. Qualunque
cambiamento nel suo comportamento che porti il totale sopra $25$ è un
miglioramento; “agire nel modo migliore possibile” significa, alla fine,
scegliere le mosse che rendono quel punteggio il più alto possibile. Buona
parte dell'intelligenza artificiale moderna, sotto sotto, funziona così: si
sceglie un numero da massimizzare (o un errore da rendere minimo) e si lascia
che sia la macchina a scoprire come.

Conviene sapere subito dov'è la crepa, però, perché ci accompagnerà per tutto
il libro: quel punteggio lo scriviamo noi, e non è mai *esattamente* la cosa
che vogliamo. Un aspirapolvere pagato a briciole raccolte, se è abbastanza
bravo, può scoprire che gli conviene rovesciare il cestino e raccoglierle una
seconda volta. Ha fatto il punteggio più alto e ha sporcato il salotto: ha
obbedito alla lettera tradendo l'intenzione. Il fenomeno ha un nome
(*reward hacking*) e più avanti nel libro una sezione tutta sua.
`````

`````{tab} Superiore
Formalmente, si descrive il comportamento del sistema con dei parametri
$\theta$ e se ne misura la qualità con un’utilità attesa

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
del capitolo sul machine learning.

È la cornice dell'**agente razionale** {cite}`russell2020artificial`: un
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
il k-NN non hanno parametri da ottimizzare affatto. Va aggiunta una crepa che
non è un'eccezione ma un limite della cornice, dichiarato dagli stessi autori
che l'hanno resa canonica: $J$ è il punteggio che *scriviamo noi*, non quello
che vogliamo davvero, e un sistema abbastanza bravo massimizza il primo anche
a spese del secondo. Il fenomeno si chiama *reward hacking* e ha una sezione
sua nel capitolo sul deep reinforcement learning.
`````

Detto questo, l'AI non è una costola della teoria del controllo: nacque anzi
proprio per superarne i limiti {cite}`russell2020artificial`. Quella matematica
sapeva tenere in rotta un sistema descritto da poche variabili numeriche, e si
fermava lì; il linguaggio naturale, la visione artificiale, la pianificazione
erano problemi che si ponevano completamente fuori dal suo campo d'azione, ed
è per affrontarli che il campo si è staccato.

## Algoritmi Intelligenti

Abbiamo appena nominato gli algoritmi, ma cosa sono nello specifico? Un
**algoritmo**, prima di tutto, è una ricetta: una lista finita di passi
precisi che, eseguiti nell'ordine giusto, portano a un risultato. Uno dei
primi algoritmi della storia si può far risalire a Euclide, che oltre due
millenni fa ideò un metodo per calcolare il massimo comune divisore di due
numeri, per esempio, applicandolo a $12$ e $8$ si ottiene $4$. È esattamente
il procedimento che proveremo a scrivere, in Python, nella pagina che segue
(«Python e l'AI»): è un *notebook*, cioè una pagina in cui il codice non si
legge soltanto, si esegue.

`````{tab} Elementare
Il massimo comune divisore (MCD) di $12$ e $8$ è il numero più grande che li
divide entrambi senza resto. Puoi immaginarlo così: hai un pavimento
rettangolare di $12 \times 8$ mattonelle e vuoi ricoprirlo con piastrelle
quadrate, tutte uguali e senza tagliarne nessuna; la piastrella più grande che
funziona è quella da $4 \times 4$. Il trucco di Euclide per trovarlo è
elegante: dividi il numero grande per il piccolo e guarda il **resto**. $12$
diviso $8$ dà resto $4$; ora ripeti con $8$ e $4$: resto $0$. Appena compare
il resto zero, l'ultimo numero *per cui* hai diviso (qui $4$) è il MCD.
Niente elenchi di divisori, niente tentativi: due divisioni e hai finito. E il
bello è che non peggiora quando i numeri crescono: per due numeri lunghi cento
cifre bastano poche centinaia di divisioni, mentre provare i divisori uno per
uno, come si fa a scuola, ne chiederebbe più di quanti sono gli atomi
dell'universo osservabile.
`````

`````{tab} Superiore
L’algoritmo sfrutta l’identità
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
linguaggio. Attenzione però a leggere bene la stima: quelli sono *passi*, e su
numeri molto lunghi ogni passo costa una divisione fra interi grandi, il cui
prezzo cresce quadraticamente nel numero di cifre: il tempo totale non è
$O(\log)$, anche se i passi lo sono. È il motivo per cui le
librerie non eseguono questo ciclo tale e quale, ma sue raffinature
(l'algoritmo di Lehmer, in CPython).
`````

## Quando le Regole non si Scrivono

L'algoritmo di Euclide è una ricetta, e le ricette hanno un autore: qualcuno
ha capito come si fa e ne ha scritto i passi. Per duemila anni ogni algoritmo
è stato così, e così è ancora la maggior parte dei programmi che usi ogni
giorno: chi li ha scritti sapeva già che cosa dovevano fare, riga per riga.

Il salto che rende necessario tutto il resto di questo libro sta qui. Prova a
scrivere la ricetta per riconoscere un gatto in una fotografia. Non i passi
generici («cerca le orecchie a punta»), ma i passi precisi, sui numeri che
compongono l'immagine, che funzionino anche col gatto di spalle, in
controluce, mezzo nascosto dietro una sedia. Nessuno c'è mai riuscito, e non
per pigrizia: quella ricetta non la sappiamo, per quanto siamo capacissimi di
eseguirla con gli occhi in un decimo di secondo.

E allora si cambia mestiere. Invece di scrivere le regole, si raccolgono gli
**esempi** (migliaia di fotografie già marcate «gatto» e «non gatto») e si
lascia che sia il programma a trovare da solo che cosa distingue le une dalle
altre. Le regole non le scrive nessuno: **emergono dai dati**. È questo che
significa, in questo libro, dire che un programma *impara*, ed è la ragione
per cui qui i dati contano quanto il codice.

Da qui i tre nomi del titolo, che a questo punto si dicono in una riga
ciascuno:

- il **machine learning** (apprendimento automatico) è l'idea appena detta:
  ricavare le regole dagli esempi invece di scriverle a mano;
- il **deep learning** (apprendimento profondo) è il modo di farlo che ha
  vinto: reti neurali a molti strati, in cui ogni strato ricava dai numeri
  dello strato sotto una descrizione un po' più astratta (dai pixel ai bordi,
  dai bordi alle forme, dalle forme al gatto);
- il **reinforcement learning** (apprendimento per rinforzo) è il caso in cui
  gli esempi giusti non esistono e il programma impara dalle conseguenze delle
  proprie azioni: prova, riceve un punteggio, riprova. È quello che
  incontreremo nella prossima pagina, con il robot che impara a camminare.

Non sono tre cerchi uno dentro l'altro, benché vengano disegnati spesso così.
Il deep learning è davvero un modo di fare machine learning; ma il
reinforcement learning è un *problema*, non una tecnica, e lo si affronta con
o senza reti profonde. E l'intelligenza artificiale è più larga di tutti e
tre: comprende anche i programmi che ragionano su regole scritte a mano, cioè
la sua metà classica, quella dei sistemi esperti del secondo inverno.

Da qui una definizione da tenersi in tasca, provvisoria come tutte quelle
buone: **l'intelligenza artificiale si occupa di far svolgere a una macchina
compiti per cui non sappiamo scrivere la ricetta**. È la riga che tiene fuori
la lavatrice. Anche una lavatrice decide da sola quando fermare il risciacquo,
e decide bene; ma quella decisione qualcuno l'ha scritta, c'è un tecnico che
ha stabilito quale sensore leggere e sopra quale soglia fermarsi. Riconoscere
un gatto, tradurre una frase, tenere in piedi un robot: lì la ricetta non c'è,
e va fatta emergere.

Torna così la domanda dell'inizio. ELIZA non aveva niente di più di una lista
di istruzioni, ed è proprio per questo che il suo incanto si sgretola appena
lo si spiega. I programmi di cui parla questo libro qualcosa di più ce
l'hanno, ma è meno misterioso e più scomodo di quanto si immagini: nessuno ha
scritto le regole che seguono, e quindi nessuno, nemmeno chi li ha costruiti,
sa elencarle tutte. Da lì vengono sia i risultati sia i guai, e a entrambi il
libro dedica dei capitoli.

## Perché Proprio Adesso

Se le regole si ricavano dagli esempi, servono gli esempi; e servono macchine
capaci di macinarli. È la risposta alla domanda che tutti fanno, cioè perché
un campo nato negli anni Cinquanta abbia cominciato a funzionare solo di
recente: perché servivano tre ingredienti insieme, e per decenni ce n'erano al
massimo due.

L'AI è in profondo debito con l'evoluzione dell'informatica e
dell'elettronica, che hanno messo a disposizione sistemi operativi, linguaggi
di programmazione (es: Python), **librerie** (PyTorch, TensorFlow...), cioè
raccolte di codice già scritto e collaudato che si usa invece di rifarlo, ed
architetture sempre più potenti: i processori (CPU), le schede grafiche (GPU)
nate per i videogiochi e diventate il motore del deep learning, i chip
dedicati (TPU), che permettono la creazione di modelli in tempi brevi. Ed è in
debito con Internet, che ha fatto molto più che diffondere articoli e video:
ha reso raccoglibili i **dati** su cui questi modelli si addestrano, dalle
grandi collezioni di immagini già etichettate al testo del web. Dati, potenza
di calcolo e algoritmi maturi sono i tre ingredienti, e il capitolo sul deep
learning li conta uno per uno; quanto pesi ciascuno, in numeri, lo dicono le
leggi di scala nel capitolo sui Transformer.

Il debito, peraltro, è stato ripagato con gli interessi
{cite}`russell2020artificial`: parecchie idee nate nei laboratori di
intelligenza artificiale hanno poi fatto il giro dell'informatica intera e
oggi si usano ovunque senza ricordarne la provenienza. La più visibile a
chiunque programmi è la **gestione automatica della memoria**: il fatto che un
programma possa creare oggetti a volontà senza doverli poi cancellare a mano
uno per uno, che è esattamente quello che Python fa per te e che in altri
linguaggi resta un lavoro (e una fonte di errori) del programmatore. Ed è con
Python che, nella pagina qui accanto, scriveremo il primo algoritmo di questo
libro.
