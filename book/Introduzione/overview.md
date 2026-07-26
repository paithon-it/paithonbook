# Introduzione

Come direbbe Joseph Weizenbaum, informatico tedesco emigrato negli Stati
Uniti, professore al MIT e creatore di ELIZA, il primo *chatbot* della storia,
un programma che già a metà degli anni Sessanta permetteva a un essere umano
di conversare per iscritto con una macchina {cite}`weizenbaum1966eliza`:

>L’intelligenza artificiale è straordinariamente resistente al tentativo di una precisa definizione.

Weizenbaum, che voleva dimostrare la superficialità della comunicazione tra uomo e macchina, rimase estremamente sorpreso dal numero di persone che attribuivano sentimenti umani al suo programma. Ma siamo sicuri che sia solo una semplice lista di istruzioni quella da lui creata? O c’è qualcosa di più? Se è un semplice programma, perché attribuirgli una parola così ricca di significato come l’*intelligenza*?

E poi, cos’è questa intelligenza artificiale (o **AI** dall’inglese *Artificial Intelligence*), che così velocemente, anno dopo anno, perfezionamento dopo perfezionamento, sfugge inesorabilmente ad una “*precisa definizione*”?

## Le Origini dell'Intelligenza Artificiale

È presente in ognuno di noi quella strana sensazione suscitata dall’osservare un programma che emula i comportamenti umani, come se fosse qualcosa di "vivo" e non un semplice programma che esegue istruzioni.

La possibilità di creare un qualcosa che assomigliasse all’uomo ha sollecitato
la curiosità di tutti i pensatori del passato. Difatti, nonostante abbia
trovato larga applicazione solo nell’ultimo decennio, l’AI non è una scienza
nuovissima: le sue origini risalgono al periodo immediatamente successivo alla
Seconda Guerra Mondiale. Nel 1950 Alan Turing, in un articolo destinato a fare
storia, propose di sostituire la domanda «le macchine possono pensare?» con un
esperimento concreto, il *gioco dell’imitazione*, oggi noto come **test di
Turing**: se conversando a distanza non riesci a capire se dall’altra parte
c’è una persona o un programma, la questione filosofica smette di essere
decisiva {cite}`turing1950computing`. Il termine *intelligenza artificiale* fu
poi coniato, “ufficialmente”, nel 1956 dal matematico statunitense John
McCarthy. È una scienza che si può applicare potenzialmente a ogni sfera del
pensiero umano, in quanto si occupa di rendere automatiche alcune attività
intellettive come il riconoscimento di immagini, il gioco degli scacchi, la
dimostrazione di teoremi matematici e la guida autonoma di veicoli. In questo
senso, è uno dei campi più antichi e trasversali.

E se dicessi che uno dei primi test al mondo di auto a guida autonoma è stato
effettuato in Italia, su una “fiammante” Lancia Thema? Ebbene sì: a partire
dal 1996, all’Università di Parma, un team di ricercatori e ingegneri
coordinato dal prof. Alberto Broggi ha sviluppato un vero e proprio prototipo
di guida autonoma, un’autovettura dotata di visione “stereo” (due telecamere
che lavorano in coppia, come gli occhi umani) e di un computer di bordo del
tutto ordinario per l’epoca (un Pentium 200 MMX): in grado di sterzare da
sola, restare al centro della corsia di marcia e localizzare gli eventuali
ostacoli sul percorso. Il suo nome è ARGO e nel giugno 1998, nella prova
“MilleMiglia in Automatico”, percorse quasi 2.000 km di strade e autostrade
italiane guidando in autonomia per oltre il 90% del tragitto.

Nel 2010 il gruppo di Broggi è riuscito a far guidare autonomamente delle auto dall’Italia… alla Cina! La sfida si chiamava VIAC (VisLab Intercontinental Autonomous Challenge) e ha coinvolto quattro veicoli su un viaggio di quasi 16.000 chilometri (9.900 miglia) da Parma a Shanghai, percorso in larga parte senza intervento umano.

Del resto, i pensatori si interrogano su queste idee da millenni. Aristotele, nel IV secolo a.C., fu il primo a formulare un insieme preciso di leggi che governano la parte razionale della mente: sviluppò un sistema basato sui sillogismi che, in teoria, consentiva a chiunque di ottenere meccanicamente le conclusioni a partire dalle premesse. Hobbes, nel Seicento, ipotizzò che il ragionamento umano avesse a che fare con meccanismi simili al calcolo numerico, come se noi “eseguissimo” addizioni e sottrazioni nei nostri pensieri. Pascal scrisse che la “macchina aritmetica” produce effetti che sembrano più vicini al pensiero di tutte le azioni degli animali, e costruì la Pascalina, una delle prime macchine calcolatrici (la prima in assoluto risulta essere quella di Wilhelm Schickard, nel 1623). Infine, Cartesio fornì la prima discussione chiara sulla distinzione tra mente e materia.

I filosofi hanno esplorato la maggior parte dei concetti riguardanti l’AI, ma
il passaggio ad una scienza che fosse universalmente riconosciuta e apprezzata
richiedeva un livello superiore di formalizzazione. La matematica a metà del
‘900 ereditava dal passato tutta una serie di strumenti necessari a costruire
algoritmi intelligenti: risultati provenienti dall’algebra, dalla ricerca
operativa, dalla teoria del controllo, dalla probabilità, dalla teoria dei
giochi. D’improvviso, questi campi sono apparsi come facce dello stesso
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
scegliere le mosse che rendono quel punteggio il più alto possibile. Tutta
l’intelligenza artificiale moderna, sotto sotto, funziona così: si sceglie
un numero da massimizzare (o un errore da rendere minimo) e si lascia che
sia la macchina a scoprire come.
`````

`````{tab} Superiore
Formalmente, si descrive il comportamento del sistema con dei parametri
$\theta$ e si definisce una funzione da ottimizzare: un’utilità $J(\theta)$
da massimizzare oppure, in modo equivalente, una perdita (*loss*)
$\mathcal{L}(\theta)$ da minimizzare,

$$
\theta^\star = \arg\max_{\theta} J(\theta)
\qquad \text{ovvero} \qquad
\theta^\star = \arg\min_{\theta} \mathcal{L}(\theta),
$$

dove $\theta^\star$ è la configurazione ottima dei parametri. È la cornice
dell’**agente razionale**: un sistema che sceglie le azioni che massimizzano
l’utilità attesa, date le informazioni disponibili. Il resto del libro è una
serie di variazioni su questo tema: la regressione minimizza un errore
quadratico medio, la classificazione una cross-entropy, il reinforcement
learning massimizza una ricompensa cumulata attesa. Cambiano $\mathcal{L}$ e
il modo di calcolare il minimo; lo schema non cambia mai.
`````

L’AI nacque, difatti, proprio per superare i limiti della matematica tipica della teoria del controllo degli anni ’50 e affrontare problemi come il linguaggio naturale, la visione artificiale, la pianificazione, che si ponevano completamente fuori dal suo campo d’azione.

## Algoritmi Intelligenti

Abbiamo parlato di algoritmi intelligenti, ma cosa sono nello specifico? Un
**algoritmo**, prima di tutto, è una ricetta: una lista finita di passi
precisi che, eseguiti nell’ordine giusto, portano a un risultato. Uno dei
primi algoritmi della storia si può far risalire a Euclide, che oltre due
millenni fa ideò un metodo per calcolare il massimo comune divisore di due
numeri, per esempio, applicandolo a $12$ e $8$ si ottiene $4$. È esattamente
il procedimento che proveremo a scrivere, in Python, nel notebook di questo
capitolo.

`````{tab} Elementare
Il massimo comune divisore (MCD) di $12$ e $8$ è il numero più grande che li
divide entrambi senza resto. Puoi immaginarlo così: hai un pavimento
rettangolare di $12 \times 8$ mattonelle e vuoi ricoprirlo con piastrelle
quadrate, tutte uguali e senza tagliarne nessuna; la piastrella più grande che
funziona è quella da $4 \times 4$. Il trucco di Euclide per trovarlo è
elegante: dividi il numero grande per il piccolo e guarda il **resto**. $12$
diviso $8$ dà resto $4$; ora ripeti con $8$ e $4$: resto $0$. Appena compare
il resto zero, l’ultimo divisore usato (qui $4$) è il MCD. Niente elenchi di
divisori, niente tentativi: due divisioni e hai finito, che i numeri siano
piccoli o lunghi cento cifre.
`````

`````{tab} Superiore
L’algoritmo sfrutta l’identità
$\mathrm{MCD}(a, b) = \mathrm{MCD}(b,\, a \bmod b)$, con caso base
$\mathrm{MCD}(a, 0) = a$, dove $a \bmod b$ è il resto della divisione intera.
La correttezza segue dal fatto che ogni divisore comune di $a$ e $b$ divide
anche $a \bmod b = a - \lfloor a/b \rfloor\, b$, quindi l’insieme dei divisori
comuni (e dunque il massimo) è invariante a ogni passo. Per $a=12$, $b=8$:
$(12, 8) \to (8, 4) \to (4, 0) \Rightarrow 4$. Il numero di passi è
$O(\log \min(a, b))$: il caso peggiore si ha con due numeri di Fibonacci
consecutivi (teorema di Lamé, 1844). È per questa efficienza (non solo per
l’età) che l’algoritmo di Euclide è ancora oggi nelle librerie standard di
ogni linguaggio.
`````

L’AI è in profondo debito con l’evoluzione dell’informatica e
dell’elettronica, che hanno messo a disposizione sistemi operativi, linguaggi
di programmazione (es: Python), librerie (PyTorch, TensorFlow...), ed
architetture sempre più potenti, i processori (CPU), le schede grafiche (GPU)
nate per i videogiochi e diventate il motore del deep learning, i chip
dedicati (TPU), che permettono la creazione di modelli in tempi brevi.
Inoltre, Internet ha permesso una capillare diffusione di articoli, blog e
video che favoriscono l’ampia divulgazione dell’argomento.

Il debito, peraltro, è stato ripagato con gli interessi: gli studi sull’intelligenza artificiale hanno esplorato per la prima volta idee che si sono poi diffuse nell’informatica generale, tra cui gli interpreti interattivi, gli ambienti di sviluppo rapido, la gestione automatica della memoria e altri concetti chiave nella programmazione object-oriented.
